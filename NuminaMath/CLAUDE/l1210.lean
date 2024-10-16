import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_bounds_l1210_121020

theorem quadratic_roots_bounds (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m < 0) 
  (hroots : x₁^2 - x₁ - 6 = m ∧ x₂^2 - x₂ - 6 = m) 
  (horder : x₁ < x₂) : 
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_bounds_l1210_121020


namespace NUMINAMATH_CALUDE_remainder_problem_l1210_121002

theorem remainder_problem : (245 * 15 - 20 * 8 + 5) % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1210_121002


namespace NUMINAMATH_CALUDE_inverse_function_value_l1210_121012

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x

-- Define the domain of f
def domain (x : ℝ) : Prop := x < -2

-- Define the inverse function property
def is_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, domain x → f (f_inv (f x)) = f x ∧ f_inv (f x) = x

-- Theorem statement
theorem inverse_function_value :
  ∃ f_inv : ℝ → ℝ, is_inverse f f_inv ∧ f_inv 12 = -6 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_value_l1210_121012


namespace NUMINAMATH_CALUDE_inequality_proof_l1210_121072

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c < d) : a - c > b - d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1210_121072


namespace NUMINAMATH_CALUDE_janous_conjecture_l1210_121077

theorem janous_conjecture (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (z^2 - x^2) / (x + y) + (x^2 - y^2) / (y + z) + (y^2 - z^2) / (z + x) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_janous_conjecture_l1210_121077


namespace NUMINAMATH_CALUDE_batsman_average_batsman_average_proof_l1210_121091

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) : ℕ :=
  let prev_average := 30
  let new_average := prev_average + average_increase
  new_average

#check batsman_average 10 60 3 = 33

theorem batsman_average_proof 
  (total_innings : ℕ) 
  (last_innings_score : ℕ) 
  (average_increase : ℕ) 
  (h1 : total_innings = 10)
  (h2 : last_innings_score = 60)
  (h3 : average_increase = 3) :
  batsman_average total_innings last_innings_score average_increase = 33 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_batsman_average_proof_l1210_121091


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l1210_121051

theorem min_value_sum_fractions (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l1210_121051


namespace NUMINAMATH_CALUDE_intercept_sum_l1210_121090

/-- The line equation 2x - y + 4 = 0 -/
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := -2

/-- The y-intercept of the line -/
def y_intercept : ℝ := 4

/-- Theorem: The sum of the x-intercept and y-intercept of the line 2x - y + 4 = 0 is equal to 2 -/
theorem intercept_sum : x_intercept + y_intercept = 2 := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_l1210_121090


namespace NUMINAMATH_CALUDE_sum_of_powers_of_two_l1210_121030

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_two_l1210_121030


namespace NUMINAMATH_CALUDE_smallest_b_for_nonprime_cubic_l1210_121035

theorem smallest_b_for_nonprime_cubic (x : ℤ) : ∃ (b : ℕ+), ∀ (x : ℤ), ¬ Prime (x^3 + b^2) ∧ ∀ (k : ℕ+), k < b → ∃ (y : ℤ), Prime (y^3 + k^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_nonprime_cubic_l1210_121035


namespace NUMINAMATH_CALUDE_fraction_equality_l1210_121062

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : 1 / x + 1 / y = 2) : 
  (x * y + 3 * x + 3 * y) / (x * y) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1210_121062


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l1210_121041

def f (x : ℝ) : ℝ := -x^2 + 4*x - 3

theorem f_monotone_decreasing : 
  MonotoneOn f (Set.Ici 2) := by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l1210_121041


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_circle_tangent_l1210_121057

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a circle is tangent to a line segment -/
def isTangent (c : Circle) (p1 p2 : Point) : Prop := sorry

/-- Checks if a point lies on a line segment -/
def liesBetween (p : Point) (p1 p2 : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Main theorem -/
theorem isosceles_trapezoid_circle_tangent 
  (ABCD : IsoscelesTrapezoid) 
  (c : Circle) 
  (M N : Point) :
  isTangent c ABCD.A ABCD.B →
  isTangent c ABCD.B ABCD.C →
  liesBetween M ABCD.A ABCD.D →
  liesBetween N ABCD.C ABCD.D →
  distance ABCD.A M / distance ABCD.D M = 1 / 3 →
  distance ABCD.C N / distance ABCD.D N = 4 / 3 →
  distance ABCD.A ABCD.B = 7 →
  distance ABCD.A ABCD.D = 6 →
  distance ABCD.B ABCD.C = 4 + 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_circle_tangent_l1210_121057


namespace NUMINAMATH_CALUDE_division_simplification_l1210_121059

theorem division_simplification (x y : ℝ) : -6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l1210_121059


namespace NUMINAMATH_CALUDE_green_ball_probability_l1210_121071

-- Define the containers and their contents
def containerA : ℕ × ℕ := (5, 7)  -- (red, green)
def containerB : ℕ × ℕ := (8, 6)
def containerC : ℕ × ℕ := (3, 9)

-- Define the probability of selecting each container
def containerProb : ℚ := 1 / 3

-- Define the probability of selecting a green ball from each container
def greenProbA : ℚ := containerA.2 / (containerA.1 + containerA.2)
def greenProbB : ℚ := containerB.2 / (containerB.1 + containerB.2)
def greenProbC : ℚ := containerC.2 / (containerC.1 + containerC.2)

-- Theorem: The probability of selecting a green ball is 127/252
theorem green_ball_probability : 
  containerProb * greenProbA + containerProb * greenProbB + containerProb * greenProbC = 127 / 252 := by
  sorry


end NUMINAMATH_CALUDE_green_ball_probability_l1210_121071


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1210_121034

theorem complex_expression_equality : 
  let x := (11 + 6 * Real.sqrt 2) * Real.sqrt (11 - 6 * Real.sqrt 2) - 
           (11 - 6 * Real.sqrt 2) * Real.sqrt (11 + 6 * Real.sqrt 2)
  let y := Real.sqrt (Real.sqrt 5 + 2) + Real.sqrt (Real.sqrt 5 - 2) - 
           Real.sqrt (Real.sqrt 5 + 1)
  x / y = 28 + 14 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1210_121034


namespace NUMINAMATH_CALUDE_set_union_problem_l1210_121008

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 2} →
  B = {3, a} →
  A ∩ B = {1} →
  A ∪ B = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l1210_121008


namespace NUMINAMATH_CALUDE_one_correct_statement_l1210_121005

theorem one_correct_statement :
  (∃! n : Nat, n = 1 ∧
    (∀ a b : ℝ, a + b = 0 → a = -b) ∧
    (3^2 = 6) ∧
    (∀ a : ℚ, a > -a) ∧
    (∀ a b : ℝ, |a| = |b| → a = b)) :=
sorry

end NUMINAMATH_CALUDE_one_correct_statement_l1210_121005


namespace NUMINAMATH_CALUDE_standard_form_conversion_theta_range_phi_range_l1210_121080

/-- Converts spherical coordinates to standard form -/
def to_standard_spherical (ρ : ℝ) (θ : ℝ) (φ : ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The standard form of (5, 3π/4, 9π/4) is (5, 7π/4, π/4) -/
theorem standard_form_conversion :
  let (ρ, θ, φ) := to_standard_spherical 5 (3 * Real.pi / 4) (9 * Real.pi / 4)
  ρ = 5 ∧ θ = 7 * Real.pi / 4 ∧ φ = Real.pi / 4 :=
by
  sorry

/-- The range of θ in standard spherical coordinates -/
theorem theta_range (θ : ℝ) : 0 ≤ θ ∧ θ < 2 * Real.pi :=
by
  sorry

/-- The range of φ in standard spherical coordinates -/
theorem phi_range (φ : ℝ) : 0 ≤ φ ∧ φ ≤ Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_standard_form_conversion_theta_range_phi_range_l1210_121080


namespace NUMINAMATH_CALUDE_mean_temperature_l1210_121031

def temperatures : List ℤ := [-8, -3, -7, -6, 0, 4, 6, 5, -1, 2]

theorem mean_temperature :
  (List.sum temperatures : ℚ) / temperatures.length = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1210_121031


namespace NUMINAMATH_CALUDE_triangle_side_angle_relation_l1210_121024

/-- Given a triangle ABC with side lengths a, b, and c opposite to angles A, B, and C respectively,
    the sum of the squares of the side lengths equals twice the sum of the products of pairs of
    side lengths and the cosine of their opposite angles. -/
theorem triangle_side_angle_relation (a b c : ℝ) (A B C : ℝ) :
  (a ≥ 0) → (b ≥ 0) → (c ≥ 0) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  a^2 + b^2 + c^2 = 2 * (b * c * Real.cos A + a * c * Real.cos B + a * b * Real.cos C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_angle_relation_l1210_121024


namespace NUMINAMATH_CALUDE_sample_data_properties_l1210_121029

theorem sample_data_properties (x : Fin 6 → ℝ) 
  (h_ordered : ∀ i j, i < j → x i ≤ x j) : 
  (((x 2 + x 3) / 2 = (x 3 + x 4) / 2) ∧ 
  (x 5 - x 2 ≤ x 6 - x 1)) := by sorry

end NUMINAMATH_CALUDE_sample_data_properties_l1210_121029


namespace NUMINAMATH_CALUDE_smiths_children_ages_l1210_121092

def is_divisible (n m : ℕ) : Prop := m ∣ n

theorem smiths_children_ages (children_ages : Finset ℕ) : 
  children_ages.card = 7 ∧ 
  (∀ a ∈ children_ages, 2 ≤ a ∧ a ≤ 11) ∧
  (∃ x, 2 ≤ x ∧ x ≤ 11 ∧ x ∉ children_ages) ∧
  5 ∈ children_ages ∧
  (∀ a ∈ children_ages, is_divisible 3339 a) ∧
  39 ∉ children_ages →
  6 ∉ children_ages :=
by sorry

end NUMINAMATH_CALUDE_smiths_children_ages_l1210_121092


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_revenue_l1210_121055

/-- Revenue function for book sales -/
def revenue (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The optimal price maximizes revenue -/
theorem optimal_price_maximizes_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → revenue p ≥ revenue q ∧
  p = 19 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_revenue_l1210_121055


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1210_121096

/-- Given a quadratic equation x^2 - bx + 20 = 0 where the product of roots is 20,
    prove that the sum of roots is b. -/
theorem sum_of_roots_quadratic (b : ℝ) : 
  (∃ x y : ℝ, x^2 - b*x + 20 = 0 ∧ y^2 - b*y + 20 = 0 ∧ x*y = 20) → 
  (∃ x y : ℝ, x^2 - b*x + 20 = 0 ∧ y^2 - b*y + 20 = 0 ∧ x + y = b) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1210_121096


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l1210_121052

/-- A function f is monotonic on an open interval (a, b) if it is either
    strictly increasing or strictly decreasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∨
  (∀ x y, a < x ∧ x < y ∧ y < b → f y < f x)

theorem quadratic_monotonicity (a : ℝ) :
  IsMonotonic (fun x ↦ x^2 + 2*(a-1)*x + 2) 2 4 →
  a ≤ -3 ∨ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l1210_121052


namespace NUMINAMATH_CALUDE_bernardo_silvia_game_l1210_121022

theorem bernardo_silvia_game (N : ℕ) : N = 24 ↔ 
  (N ≤ 999) ∧ 
  (3 * N < 800) ∧ 
  (3 * N - 30 < 800) ∧ 
  (9 * N - 90 < 800) ∧ 
  (9 * N - 120 < 800) ∧ 
  (27 * N - 360 < 800) ∧ 
  (27 * N - 390 < 800) ∧ 
  (81 * N - 1170 ≥ 800) ∧ 
  (∀ m : ℕ, m < N → 
    (3 * m < 800) ∧ 
    (3 * m - 30 < 800) ∧ 
    (9 * m - 90 < 800) ∧ 
    (9 * m - 120 < 800) ∧ 
    (27 * m - 360 < 800) ∧ 
    (27 * m - 390 < 800) ∧ 
    (81 * m - 1170 < 800)) := by
  sorry

end NUMINAMATH_CALUDE_bernardo_silvia_game_l1210_121022


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l1210_121083

/-- The function f(x) = x^3 - bx^2 - 4 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 - 4

/-- The theorem stating that f has three distinct real zeros iff b < -3 -/
theorem f_has_three_zeros (b : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f b x = 0 ∧ f b y = 0 ∧ f b z = 0) ↔ 
  b < -3 := by sorry

end NUMINAMATH_CALUDE_f_has_three_zeros_l1210_121083


namespace NUMINAMATH_CALUDE_negation_equivalence_l1210_121054

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Teacher : U → Prop)
variable (ExcellentAtMath : U → Prop)
variable (PoorAtMath : U → Prop)

-- Define the statements
def AllTeachersExcellent : Prop := ∀ x, Teacher x → ExcellentAtMath x
def AtLeastOneTeacherPoor : Prop := ∃ x, Teacher x ∧ PoorAtMath x

-- Theorem statement
theorem negation_equivalence : 
  AtLeastOneTeacherPoor U Teacher PoorAtMath ↔ ¬(AllTeachersExcellent U Teacher ExcellentAtMath) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1210_121054


namespace NUMINAMATH_CALUDE_equation_solutions_l1210_121063

theorem equation_solutions : 
  let f (x : ℝ) := (18*x - x^2) / (x + 2) * (x + (18 - x) / (x + 2))
  ∀ x : ℝ, f x = 56 ↔ x = 4 ∨ x = -14/17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1210_121063


namespace NUMINAMATH_CALUDE_triangle_area_l1210_121074

theorem triangle_area (a b c : ℝ) (h1 : a = 14) (h2 : b = 48) (h3 : c = 50) :
  (1/2) * a * b = 336 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1210_121074


namespace NUMINAMATH_CALUDE_plot_length_is_57_l1210_121093

/-- A rectangular plot with specific fencing cost and length-breadth relationship -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_relation : length = breadth + 14
  fencing_cost_equation : total_fencing_cost = fencing_cost_per_meter * (2 * length + 2 * breadth)

/-- The length of the rectangular plot is 57 meters -/
theorem plot_length_is_57 (plot : RectangularPlot) 
    (h1 : plot.fencing_cost_per_meter = 26.5)
    (h2 : plot.total_fencing_cost = 5300) : 
  plot.length = 57 := by
  sorry


end NUMINAMATH_CALUDE_plot_length_is_57_l1210_121093


namespace NUMINAMATH_CALUDE_f_negative_values_l1210_121049

/-- The function f(x) defined as x^2 - ax + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

/-- Theorem stating that f(x) can take negative values iff a > 2 or a < -2 --/
theorem f_negative_values (a : ℝ) :
  (∃ x, f a x < 0) ↔ (a > 2 ∨ a < -2) := by sorry

end NUMINAMATH_CALUDE_f_negative_values_l1210_121049


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_product_product_equals_87526608_l1210_121028

theorem consecutive_even_numbers_product : Int → Prop :=
  fun n => (n - 2) * n * (n + 2) = 87526608

theorem product_equals_87526608 : consecutive_even_numbers_product 444 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_product_product_equals_87526608_l1210_121028


namespace NUMINAMATH_CALUDE_max_vertex_sum_l1210_121027

/-- Represents the set of numbers to be assigned to cube faces -/
def cube_numbers : Finset ℕ := {7, 8, 9, 10, 11, 12}

/-- Represents a valid assignment of numbers to cube faces -/
def valid_assignment (assignment : Fin 6 → ℕ) : Prop :=
  ∀ i : Fin 6, assignment i ∈ cube_numbers ∧ (∀ j : Fin 6, i ≠ j → assignment i ≠ assignment j)

/-- Calculates the sum of products at vertices given a face assignment -/
def vertex_sum (assignment : Fin 6 → ℕ) : ℕ :=
  let opposite_pairs := [(0, 1), (2, 3), (4, 5)]
  (assignment 0 + assignment 1) * (assignment 2 + assignment 3) * (assignment 4 + assignment 5)

/-- Theorem stating the maximum sum of vertex products -/
theorem max_vertex_sum :
  ∃ assignment : Fin 6 → ℕ,
    valid_assignment assignment ∧
    vertex_sum assignment = 6859 ∧
    ∀ other : Fin 6 → ℕ, valid_assignment other → vertex_sum other ≤ 6859 := by
  sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l1210_121027


namespace NUMINAMATH_CALUDE_sum_squared_l1210_121067

theorem sum_squared (x y : ℝ) (h1 : 2*x*(x+y) = 58) (h2 : 3*y*(x+y) = 111) : 
  (x + y)^2 = 28561 / 25 := by
sorry

end NUMINAMATH_CALUDE_sum_squared_l1210_121067


namespace NUMINAMATH_CALUDE_stripe_area_cylindrical_tower_l1210_121058

/-- The area of a horizontal stripe wrapping twice around a cylindrical tower -/
theorem stripe_area_cylindrical_tower (d h w : ℝ) (hd : d = 25) (hh : h = 60) (hw : w = 2) :
  let circumference := π * d
  let stripe_length := 2 * circumference
  let stripe_area := stripe_length * w
  stripe_area = 100 * π :=
sorry

end NUMINAMATH_CALUDE_stripe_area_cylindrical_tower_l1210_121058


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_zero_one_l1210_121047

-- Define set M
def M : Set ℝ := {x | x^2 = x}

-- Define set N
def N : Set ℝ := {-1, 0, 1}

-- Theorem statement
theorem M_intersect_N_eq_zero_one : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_zero_one_l1210_121047


namespace NUMINAMATH_CALUDE_sum_area_15_disks_on_unit_circle_l1210_121078

/-- The sum of areas of 15 congruent disks covering a unit circle --/
theorem sum_area_15_disks_on_unit_circle : 
  ∃ (r : ℝ), 
    0 < r ∧ 
    (15 : ℝ) * (2 * r) = 2 * π ∧ 
    15 * (π * r^2) = π * (105 - 60 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sum_area_15_disks_on_unit_circle_l1210_121078


namespace NUMINAMATH_CALUDE_read_book_in_12_days_l1210_121048

/-- Represents the number of days it takes to read a book -/
def days_to_read_book (total_pages : ℕ) (weekday_pages : ℕ) (weekend_pages : ℕ) : ℕ :=
  let pages_per_week := 5 * weekday_pages + 2 * weekend_pages
  let full_weeks := total_pages / pages_per_week
  let remaining_pages := total_pages % pages_per_week
  let additional_days := 
    if remaining_pages ≤ 5 * weekday_pages
    then (remaining_pages + weekday_pages - 1) / weekday_pages
    else 5 + (remaining_pages - 5 * weekday_pages + weekend_pages - 1) / weekend_pages
  7 * full_weeks + additional_days

/-- Theorem stating that it takes 12 days to read the book under given conditions -/
theorem read_book_in_12_days : 
  days_to_read_book 285 23 35 = 12 := by
  sorry

end NUMINAMATH_CALUDE_read_book_in_12_days_l1210_121048


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1210_121099

/-- The number of sheep on Stewart farm given the ratio of sheep to horses and food consumption. -/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ) (food_per_horse total_food : ℕ),
  sheep * 7 = horses * 6 →  -- Ratio of sheep to horses is 6:7
  food_per_horse = 230 →    -- Each horse eats 230 ounces per day
  horses * food_per_horse = total_food →  -- Total food consumed by horses
  total_food = 12880 →      -- Total food needed is 12,880 ounces
  sheep = 48 := by
sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1210_121099


namespace NUMINAMATH_CALUDE_fraction_addition_l1210_121017

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 11 = (37 : ℚ) / 55 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1210_121017


namespace NUMINAMATH_CALUDE_g_of_two_value_l1210_121070

/-- Given a function g : ℝ → ℝ satisfying g(x) + 3 * g(1 - x) = 4 * x^3 - 5 * x for all real x, 
    prove that g(2) = -19/6 -/
theorem g_of_two_value (g : ℝ → ℝ) 
    (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 - 5 * x) : 
  g 2 = -19/6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_value_l1210_121070


namespace NUMINAMATH_CALUDE_zero_in_interval_l1210_121084

def f (x : ℝ) := x^5 + x - 3

theorem zero_in_interval :
  ∃! x : ℝ, x ∈ Set.Icc 1 2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1210_121084


namespace NUMINAMATH_CALUDE_pool_water_proof_l1210_121011

def initial_volume : ℝ := 300
def evaporation_rate_1 : ℝ := 1
def evaporation_rate_2 : ℝ := 2
def days_1 : ℝ := 15
def days_2 : ℝ := 15
def total_days : ℝ := days_1 + days_2

def remaining_volume : ℝ :=
  initial_volume - (evaporation_rate_1 * days_1 + evaporation_rate_2 * days_2)

theorem pool_water_proof :
  remaining_volume = 255 := by sorry

end NUMINAMATH_CALUDE_pool_water_proof_l1210_121011


namespace NUMINAMATH_CALUDE_coolant_replacement_l1210_121065

/-- Calculates the amount of original coolant left in a car's cooling system after partial replacement. -/
theorem coolant_replacement (initial_volume : ℝ) (initial_concentration : ℝ) 
  (replacement_concentration : ℝ) (final_concentration : ℝ) 
  (h1 : initial_volume = 19) 
  (h2 : initial_concentration = 0.3)
  (h3 : replacement_concentration = 0.8)
  (h4 : final_concentration = 0.5) : 
  initial_volume - (final_concentration * initial_volume - initial_concentration * initial_volume) / 
  (replacement_concentration - initial_concentration) = 11.4 := by
sorry

end NUMINAMATH_CALUDE_coolant_replacement_l1210_121065


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l1210_121097

theorem y_in_terms_of_x (x y : ℝ) : 2 * x + y = 5 → y = 5 - 2 * x := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l1210_121097


namespace NUMINAMATH_CALUDE_sqrt_difference_l1210_121025

theorem sqrt_difference (a b : ℝ) (ha : a < 0) (hb : b < 0) (hab : a - b = 6) :
  Real.sqrt (a^2) - Real.sqrt (b^2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_l1210_121025


namespace NUMINAMATH_CALUDE_smallest_number_l1210_121046

theorem smallest_number (S : Set ℤ) (h : S = {-2, 0, 1, 2}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1210_121046


namespace NUMINAMATH_CALUDE_sin_2theta_third_quadrant_l1210_121082

theorem sin_2theta_third_quadrant (θ : Real) :
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.sin θ)^4 + (Real.cos θ)^4 = 5/9 →
  Real.sin (2*θ) = -2*Real.sqrt 2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_third_quadrant_l1210_121082


namespace NUMINAMATH_CALUDE_one_black_two_white_reachable_l1210_121007

-- Define the urn state as a pair of natural numbers (white, black)
def UrnState := ℕ × ℕ

-- Define the initial state
def initial_state : UrnState := (50, 150)

-- Define the four operations
def op1 (s : UrnState) : UrnState := (s.1, s.2 - 2)
def op2 (s : UrnState) : UrnState := (s.1, s.2 - 1)
def op3 (s : UrnState) : UrnState := (s.1, s.2)
def op4 (s : UrnState) : UrnState := (s.1 - 3, s.2 + 2)

-- Define a predicate for valid states (non-negative marbles)
def valid_state (s : UrnState) : Prop := s.1 ≥ 0 ∧ s.2 ≥ 0

-- Define the reachability relation
inductive reachable : UrnState → Prop where
  | initial : reachable initial_state
  | op1 : ∀ s, reachable s → valid_state (op1 s) → reachable (op1 s)
  | op2 : ∀ s, reachable s → valid_state (op2 s) → reachable (op2 s)
  | op3 : ∀ s, reachable s → valid_state (op3 s) → reachable (op3 s)
  | op4 : ∀ s, reachable s → valid_state (op4 s) → reachable (op4 s)

-- Theorem stating that the configuration (2, 1) is reachable
theorem one_black_two_white_reachable : reachable (2, 1) := by sorry

end NUMINAMATH_CALUDE_one_black_two_white_reachable_l1210_121007


namespace NUMINAMATH_CALUDE_x_value_l1210_121023

theorem x_value (m : ℕ) (x : ℝ) 
  (h1 : m = 34) 
  (h2 : ((x ^ (m + 1)) / (5 ^ (m + 1))) * ((x ^ 18) / (4 ^ 18)) = 1 / (2 * (10 ^ 35))) :
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_x_value_l1210_121023


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1210_121043

theorem absolute_value_equality (x : ℝ) :
  |x^2 - 8*x + 12| = x^2 - 8*x + 12 ↔ x ≤ 2 ∨ x ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1210_121043


namespace NUMINAMATH_CALUDE_roots_negative_reciprocals_implies_a_eq_neg_c_l1210_121001

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the concept of roots
def is_root (r : ℝ) (a b c : ℝ) : Prop := quadratic_equation a b c r

-- Define negative reciprocals
def negative_reciprocals (r s : ℝ) : Prop := r = -1/s ∧ s = -1/r

-- Theorem statement
theorem roots_negative_reciprocals_implies_a_eq_neg_c 
  (a b c r s : ℝ) (h1 : is_root r a b c) (h2 : is_root s a b c) 
  (h3 : negative_reciprocals r s) : a = -c :=
sorry

end NUMINAMATH_CALUDE_roots_negative_reciprocals_implies_a_eq_neg_c_l1210_121001


namespace NUMINAMATH_CALUDE_value_of_M_l1210_121050

theorem value_of_M : ∃ M : ℝ, (0.2 * M = 0.5 * 1000) ∧ (M = 2500) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l1210_121050


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l1210_121061

theorem smallest_perfect_square_divisible_by_5_and_7 : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (m : ℕ), n = m ^ 2) ∧ 
  5 ∣ n ∧ 
  7 ∣ n ∧ 
  (∀ (k : ℕ), k > 0 → (∃ (l : ℕ), k = l ^ 2) → 5 ∣ k → 7 ∣ k → k ≥ n) ∧
  n = 1225 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l1210_121061


namespace NUMINAMATH_CALUDE_students_suggesting_pasta_l1210_121016

theorem students_suggesting_pasta 
  (total_students : ℕ) 
  (mashed_potatoes : ℕ) 
  (bacon : ℕ) 
  (h1 : total_students = 470) 
  (h2 : mashed_potatoes = 230) 
  (h3 : bacon = 140) : 
  total_students - (mashed_potatoes + bacon) = 100 := by
sorry

end NUMINAMATH_CALUDE_students_suggesting_pasta_l1210_121016


namespace NUMINAMATH_CALUDE_tax_discount_order_invariance_l1210_121076

theorem tax_discount_order_invariance 
  (price : ℝ) 
  (tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) 
  (discount_rate_pos : 0 < discount_rate) :
  price * (1 + tax_rate) * (1 - discount_rate) = 
  price * (1 - discount_rate) * (1 + tax_rate) :=
sorry

end NUMINAMATH_CALUDE_tax_discount_order_invariance_l1210_121076


namespace NUMINAMATH_CALUDE_product_factorization_l1210_121068

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def product : ℕ := (List.range 9).foldl (λ acc i => acc * factorial (20 + i)) 1

theorem product_factorization :
  ∃ (n : ℕ), n > 0 ∧ product = 825 * n^3 := by sorry

end NUMINAMATH_CALUDE_product_factorization_l1210_121068


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1210_121026

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x : ℝ, ax^2 + 2*x + c < 0 ↔ -1/3 < x ∧ x < 1/2) → 
  a = 12 ∧ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1210_121026


namespace NUMINAMATH_CALUDE_not_passed_implies_not_all_correct_l1210_121081

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (passed : Student → Prop)
variable (answered_all_correctly : Student → Prop)

-- State the given condition
variable (h : ∀ s : Student, answered_all_correctly s → passed s)

-- Theorem to prove
theorem not_passed_implies_not_all_correct (s : Student) :
  ¬(passed s) → ¬(answered_all_correctly s) :=
by
  sorry


end NUMINAMATH_CALUDE_not_passed_implies_not_all_correct_l1210_121081


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1210_121079

theorem polynomial_coefficient_sum :
  ∀ A B C D E : ℝ,
  (∀ x : ℝ, (x + 3) * (4 * x^3 - 2 * x^2 + 3 * x - 1) = A * x^4 + B * x^3 + C * x^2 + D * x + E) →
  A + B + C + D + E = 16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1210_121079


namespace NUMINAMATH_CALUDE_xyz_product_l1210_121038

theorem xyz_product (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x * (y + z) = 198)
  (h2 : y * (z + x) = 216)
  (h3 : z * (x + y) = 234) :
  x * y * z = 1080 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l1210_121038


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1210_121014

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

-- Define the arithmetic sequence condition
def arithmetic_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  3 * a 1 + 2 * a 3 = 2 * ((1/2) * a 5)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  arithmetic_sequence_condition a q →
  (a 9 + a 10) / (a 7 + a 8) = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1210_121014


namespace NUMINAMATH_CALUDE_root_of_equations_l1210_121021

theorem root_of_equations (a b c d e k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0)
  (eq1 : a * k^4 + b * k^3 + c * k^2 + d * k + e = 0)
  (eq2 : b * k^4 + c * k^3 + d * k^2 + e * k + a = 0) :
  k^5 = 1 :=
sorry

end NUMINAMATH_CALUDE_root_of_equations_l1210_121021


namespace NUMINAMATH_CALUDE_final_distance_to_catch_up_l1210_121006

/-- Represents the state of the race at any given point --/
structure RaceState where
  alex_lead : Int
  distance_covered : Nat

/-- Calculates the new race state after a terrain change --/
def update_race_state (current_state : RaceState) (alex_gain : Int) : RaceState :=
  { alex_lead := current_state.alex_lead + alex_gain,
    distance_covered := current_state.distance_covered }

def race_length : Nat := 5000

theorem final_distance_to_catch_up :
  let initial_state : RaceState := { alex_lead := 0, distance_covered := 200 }
  let after_uphill := update_race_state initial_state 300
  let after_downhill := update_race_state after_uphill (-170)
  let final_state := update_race_state after_downhill 440
  final_state.alex_lead = 570 := by sorry

end NUMINAMATH_CALUDE_final_distance_to_catch_up_l1210_121006


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l1210_121066

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  numDays : Nat
  numFridays : Nat
  firstNotFriday : firstDay ≠ DayOfWeek.Friday
  lastNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : numFridays = 5

/-- Function to determine the day of the week for a given day number -/
def dayOfWeekForDay (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Theorem stating that the 12th day is a Monday -/
theorem twelfth_day_is_monday (m : Month) :
  dayOfWeekForDay m 12 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l1210_121066


namespace NUMINAMATH_CALUDE_derivative_of_two_sin_x_l1210_121003

theorem derivative_of_two_sin_x (x : ℝ) :
  deriv (λ x => 2 * Real.sin x) x = 2 * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_two_sin_x_l1210_121003


namespace NUMINAMATH_CALUDE_concatenated_number_divisibility_l1210_121056

theorem concatenated_number_divisibility
  (n : ℕ) (a : ℕ) (h_n : n > 1) (h_a : 10^(n-1) ≤ a ∧ a < 10^n) :
  let b := a * 10^n + a
  (∃ d : ℕ, b = d * a^2) → b / a^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_concatenated_number_divisibility_l1210_121056


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1210_121037

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  -- Conditions
  (a > 0) → (b > 0) → (c > 0) →  -- Positive sides
  (a^2 + b^2 = c^2) →            -- Right triangle (Pythagorean theorem)
  (a + b = 7) →                  -- Sum of legs
  (a * b / 2 = 6) →              -- Area
  (c = 5) :=                     -- Conclusion: hypotenuse length
by
  sorry

#check right_triangle_hypotenuse

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1210_121037


namespace NUMINAMATH_CALUDE_sum_remainder_theorem_l1210_121045

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : List ℕ :=
  let n := (aₙ - a₁) / d + 1
  List.range n |>.map (fun i => a₁ + i * d)

theorem sum_remainder_theorem (a₁ d aₙ : ℕ) (h₁ : a₁ = 3) (h₂ : d = 8) (h₃ : aₙ = 283) :
  (arithmetic_sequence a₁ d aₙ).sum % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_theorem_l1210_121045


namespace NUMINAMATH_CALUDE_function_composition_problem_l1210_121085

theorem function_composition_problem (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x / 3 + 2) →
  (∀ x, g x = 5 - 2 * x) →
  f (g a) = 6 →
  a = -7/2 := by
sorry

end NUMINAMATH_CALUDE_function_composition_problem_l1210_121085


namespace NUMINAMATH_CALUDE_division_simplification_l1210_121013

theorem division_simplification (a : ℝ) (h : a ≠ 0) : 6 * a / (2 * a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l1210_121013


namespace NUMINAMATH_CALUDE_eric_jogging_time_l1210_121088

/-- Proves the time Eric spent jogging given his running time and return trip time -/
theorem eric_jogging_time 
  (total_time_to_park : ℕ) 
  (running_time : ℕ) 
  (jogging_time : ℕ) 
  (return_trip_time : ℕ) 
  (h1 : total_time_to_park = running_time + jogging_time)
  (h2 : running_time = 20)
  (h3 : return_trip_time = 90)
  (h4 : return_trip_time = 3 * total_time_to_park) :
  jogging_time = 10 := by
sorry

end NUMINAMATH_CALUDE_eric_jogging_time_l1210_121088


namespace NUMINAMATH_CALUDE_min_xy_value_l1210_121095

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/(4*y) = 1) :
  x * y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l1210_121095


namespace NUMINAMATH_CALUDE_interest_rate_is_four_percent_l1210_121036

/-- Given a principal sum and an interest rate, if the simple interest
    for 5 years is one-fifth of the principal, then the interest rate is 4% -/
theorem interest_rate_is_four_percent 
  (P : ℝ) -- Principal sum
  (R : ℝ) -- Interest rate as a percentage
  (h : P > 0) -- Assumption that principal is positive
  (h_interest : P / 5 = (P * R * 5) / 100) -- Condition that interest is one-fifth of principal
  : R = 4 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_is_four_percent_l1210_121036


namespace NUMINAMATH_CALUDE_range_of_m_for_empty_intersection_l1210_121018

/-- The set A defined by the quadratic equation mx^2 + x + m = 0 -/
def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 + x + m = 0}

/-- Theorem stating the range of m for which A has no real solutions -/
theorem range_of_m_for_empty_intersection :
  (∀ m : ℝ, (A m ∩ Set.univ = ∅) ↔ (m < -1/2 ∨ m > 1/2)) := by sorry

end NUMINAMATH_CALUDE_range_of_m_for_empty_intersection_l1210_121018


namespace NUMINAMATH_CALUDE_initial_cow_count_l1210_121000

theorem initial_cow_count (x : ℕ) 
  (h1 : x - 31 + 75 = 83) : x = 39 := by
  sorry

end NUMINAMATH_CALUDE_initial_cow_count_l1210_121000


namespace NUMINAMATH_CALUDE_binary_calculation_l1210_121019

theorem binary_calculation : 
  (0b101010 + 0b11010) * 0b1110 = 0b11000000000 := by sorry

end NUMINAMATH_CALUDE_binary_calculation_l1210_121019


namespace NUMINAMATH_CALUDE_tan_double_angle_l1210_121042

theorem tan_double_angle (α : Real) (h : 3 * Real.cos α + Real.sin α = 0) :
  Real.tan (2 * α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l1210_121042


namespace NUMINAMATH_CALUDE_probability_one_genuine_one_defective_l1210_121004

/-- The probability of selecting exactly one genuine product and one defective product
    when randomly selecting two products from a set of 5 genuine products and 1 defective product. -/
theorem probability_one_genuine_one_defective :
  let total_products : ℕ := 5 + 1
  let genuine_products : ℕ := 5
  let defective_products : ℕ := 1
  let total_selections : ℕ := Nat.choose total_products 2
  let favorable_selections : ℕ := genuine_products * defective_products
  (favorable_selections : ℚ) / total_selections = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_genuine_one_defective_l1210_121004


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1210_121032

theorem complex_magnitude_problem (z : ℂ) (i : ℂ) (h : z = (1 + i) / i) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1210_121032


namespace NUMINAMATH_CALUDE_exists_distinct_singleton_solutions_l1210_121033

/-- Solution set of x^2 + 4x - 2a ≤ 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 4*x - 2*a ≤ 0}

/-- Solution set of x^2 - ax + a + 3 ≤ 0 -/
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + a + 3 ≤ 0}

/-- Theorem stating that there exists an 'a' for which A and B are singleton sets with different elements -/
theorem exists_distinct_singleton_solutions :
  ∃ (a : ℝ), (∃! x, x ∈ A a) ∧ (∃! y, y ∈ B a) ∧ (∀ x y, x ∈ A a → y ∈ B a → x ≠ y) :=
sorry

end NUMINAMATH_CALUDE_exists_distinct_singleton_solutions_l1210_121033


namespace NUMINAMATH_CALUDE_real_part_of_fraction_l1210_121010

theorem real_part_of_fraction (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 2) : 
  (2 / (1 - z)).re = 2/5 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_fraction_l1210_121010


namespace NUMINAMATH_CALUDE_al_sandwich_options_l1210_121039

-- Define the types of ingredients
structure Ingredients :=
  (bread : Nat)
  (meat : Nat)
  (cheese : Nat)

-- Define the restrictions
structure Restrictions :=
  (turkey_swiss : Nat)
  (rye_roast_beef : Nat)

-- Define the function to calculate the number of sandwiches
def calculate_sandwiches (i : Ingredients) (r : Restrictions) : Nat :=
  i.bread * i.meat * i.cheese - r.turkey_swiss - r.rye_roast_beef

-- Theorem statement
theorem al_sandwich_options (i : Ingredients) (r : Restrictions) 
  (h1 : i.bread = 5)
  (h2 : i.meat = 7)
  (h3 : i.cheese = 6)
  (h4 : r.turkey_swiss = 5)
  (h5 : r.rye_roast_beef = 6) :
  calculate_sandwiches i r = 199 := by
  sorry


end NUMINAMATH_CALUDE_al_sandwich_options_l1210_121039


namespace NUMINAMATH_CALUDE_train_length_calculation_l1210_121009

/-- Calculates the length of a train given its speed, the time it takes to pass a bridge, and the length of the bridge. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  bridge_length = 140 →
  passing_time = 50 →
  (train_speed * passing_time) - bridge_length = 485 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1210_121009


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1210_121040

theorem arithmetic_calculations :
  (25 - (3 - (-30 - 2)) = -10) ∧
  ((-80) * (1/2 + 2/5 - 1) = 8) ∧
  (81 / (-3)^3 + (-1/5) * (-10) = -1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1210_121040


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1210_121015

/-- Proves that the average salary of all workers is 750, given the specified conditions. -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℕ)
  (non_technician_salary : ℕ)
  (h1 : total_workers = 20)
  (h2 : technicians = 5)
  (h3 : technician_salary = 900)
  (h4 : non_technician_salary = 700) :
  (technicians * technician_salary + (total_workers - technicians) * non_technician_salary) / total_workers = 750 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l1210_121015


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l1210_121089

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters : ℕ := 8
  let probability : ℚ := unique_letters / alphabet_size
  probability = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l1210_121089


namespace NUMINAMATH_CALUDE_furniture_pricing_l1210_121064

/-- The cost price of furniture before markup and discount -/
def cost_price : ℝ := 7777.78

/-- The markup percentage applied to the cost price -/
def markup_percentage : ℝ := 0.20

/-- The discount percentage applied to the total price -/
def discount_percentage : ℝ := 0.10

/-- The final price paid by the customer after markup and discount -/
def final_price : ℝ := 8400

theorem furniture_pricing :
  final_price = (1 - discount_percentage) * (1 + markup_percentage) * cost_price := by
  sorry

end NUMINAMATH_CALUDE_furniture_pricing_l1210_121064


namespace NUMINAMATH_CALUDE_polynomial_B_value_l1210_121087

def polynomial (z A B C D : ℤ) : ℤ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36

theorem polynomial_B_value :
  ∀ (A B C D : ℤ),
  (∀ r : ℤ, polynomial r A B C D = 0 → r > 0) →
  (∃ r1 r2 r3 r4 r5 r6 : ℤ, 
    ∀ z : ℤ, polynomial z A B C D = (z - r1) * (z - r2) * (z - r3) * (z - r4) * (z - r5) * (z - r6)) →
  B = -136 := by
sorry

end NUMINAMATH_CALUDE_polynomial_B_value_l1210_121087


namespace NUMINAMATH_CALUDE_max_value_a_l1210_121075

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 2 * b) 
  (h2 : b < 3 * c) 
  (h3 : c < 2 * d) 
  (h4 : d < 50) : 
  a ≤ 579 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 579 ∧ 
    a' < 2 * b' ∧ 
    b' < 3 * c' ∧ 
    c' < 2 * d' ∧ 
    d' < 50 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l1210_121075


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1210_121044

theorem complex_number_in_first_quadrant : ∃ z : ℂ, 
  z - Complex.I = Complex.abs (1 + 2 * Complex.I) ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1210_121044


namespace NUMINAMATH_CALUDE_factorization_problems_l1210_121086

theorem factorization_problems (x y : ℝ) : 
  (x^3 - 6*x^2 + 9*x = x*(x-3)^2) ∧ 
  ((x-2)^2 - x + 2 = (x-2)*(x-3)) ∧ 
  ((x^2 + y^2)^2 - 4*x^2*y^2 = (x+y)^2*(x-y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1210_121086


namespace NUMINAMATH_CALUDE_gwen_bookcase_distribution_l1210_121069

/-- Given a bookcase with mystery and picture book shelves, 
    calculates the number of books on each shelf. -/
def books_per_shelf (mystery_shelves : ℕ) (picture_shelves : ℕ) (total_books : ℕ) : ℕ :=
  total_books / (mystery_shelves + picture_shelves)

/-- Proves that Gwen's bookcase has 4 books on each shelf. -/
theorem gwen_bookcase_distribution :
  books_per_shelf 5 3 32 = 4 := by
  sorry

#eval books_per_shelf 5 3 32

end NUMINAMATH_CALUDE_gwen_bookcase_distribution_l1210_121069


namespace NUMINAMATH_CALUDE_find_m_l1210_121053

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5*x + m = 0}

-- Define the complement of A in U
def complement_A (m : ℕ) : Set ℕ := U \ A m

-- Theorem statement
theorem find_m : ∃ m : ℕ, complement_A m = {2, 3} ∧ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1210_121053


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1210_121060

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1210_121060


namespace NUMINAMATH_CALUDE_particle_probability_l1210_121098

def probability (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * probability (x-1) y + (1/3) * probability x (y-1) + (1/3) * probability (x-1) (y-1)

theorem particle_probability :
  probability 3 3 = 7/81 :=
sorry

end NUMINAMATH_CALUDE_particle_probability_l1210_121098


namespace NUMINAMATH_CALUDE_smallest_natural_with_eight_divisors_ending_in_zero_l1210_121094

theorem smallest_natural_with_eight_divisors_ending_in_zero (N : ℕ) :
  (N % 10 = 0) →  -- N ends with 0
  (Finset.card (Nat.divisors N) = 8) →  -- N has exactly 8 divisors
  (∀ M : ℕ, M % 10 = 0 ∧ Finset.card (Nat.divisors M) = 8 → N ≤ M) →  -- N is the smallest such number
  N = 30 := by
sorry

end NUMINAMATH_CALUDE_smallest_natural_with_eight_divisors_ending_in_zero_l1210_121094


namespace NUMINAMATH_CALUDE_right_triangle_median_l1210_121073

/-- Given a right triangle XYZ with ∠XYZ as the right angle, XY = 6, YZ = 8, and N as the midpoint of XZ, prove that YN = 5 -/
theorem right_triangle_median (X Y Z N : ℝ × ℝ) : 
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 6^2 →
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = 8^2 →
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = (X.1 - Y.1)^2 + (X.2 - Y.2)^2 + (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 →
  N = ((X.1 + Z.1) / 2, (X.2 + Z.2) / 2) →
  (Y.1 - N.1)^2 + (Y.2 - N.2)^2 = 5^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_median_l1210_121073
