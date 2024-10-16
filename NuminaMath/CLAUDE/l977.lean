import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l977_97780

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (d : ℤ) :
  is_arithmetic_sequence a d →
  a 3 * a 7 = -16 →
  a 4 + a 6 = 0 →
  ((a 1 = -8 ∧ d = 2) ∨ (a 1 = 8 ∧ d = -2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l977_97780


namespace NUMINAMATH_CALUDE_circle_and_line_equations_l977_97703

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧
                 (2 - a)^2 + (4 - b)^2 = r^2 ∧
                 (1 - a)^2 + (3 - b)^2 = r^2 ∧
                 a - b + 1 = 0

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * x + 1

-- Define the dot product of OM and ON
def dot_product_OM_ON (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem circle_and_line_equations :
  ∀ (k : ℝ),
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_l x₁ y₁ k ∧ line_l x₂ y₂ k ∧
    dot_product_OM_ON x₁ y₁ x₂ y₂ = 12) →
  (∀ (x y : ℝ), circle_C x y ↔ (x - 2)^2 + (y - 3)^2 = 1) ∧
  k = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_equations_l977_97703


namespace NUMINAMATH_CALUDE_population_and_sample_properties_l977_97788

/-- Represents a student in the seventh grade -/
structure Student where
  id : Nat

/-- Represents a population of students -/
structure Population where
  students : Finset Student
  size : Nat
  h_size : students.card = size

/-- Represents a sample of students -/
structure Sample where
  students : Finset Student
  population : Population
  h_subset : students ⊆ population.students

/-- The main theorem stating properties of the population and sample -/
theorem population_and_sample_properties
  (total_students : Finset Student)
  (h_total : total_students.card = 800)
  (sample_students : Finset Student)
  (h_sample : sample_students ⊆ total_students)
  (h_sample_size : sample_students.card = 50) :
  let pop : Population := ⟨total_students, 800, h_total⟩
  let samp : Sample := ⟨sample_students, pop, h_sample⟩
  (pop.size = 800) ∧
  (samp.students ⊆ pop.students) ∧
  (samp.students.card = 50) := by
  sorry


end NUMINAMATH_CALUDE_population_and_sample_properties_l977_97788


namespace NUMINAMATH_CALUDE_complex_first_quadrant_l977_97742

theorem complex_first_quadrant (a : ℝ) : 
  (∃ (z : ℂ), z = (1 : ℂ) / (1 + a * Complex.I) ∧ z.re > 0 ∧ z.im > 0) ↔ a < 0 :=
sorry

end NUMINAMATH_CALUDE_complex_first_quadrant_l977_97742


namespace NUMINAMATH_CALUDE_triangle_problem_l977_97794

theorem triangle_problem (A B C : Real) (a b c : Real) :
  0 < A → A < π / 2 →
  (1 / 2) * b * c * Real.sin A = (Real.sqrt 3 / 4) * b * c →
  c / b = 1 / 2 + Real.sqrt 3 →
  A = π / 3 ∧ Real.tan B = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l977_97794


namespace NUMINAMATH_CALUDE_system_solution_l977_97744

theorem system_solution : 
  ∃! (x y : ℝ), (2 * x + y = 6) ∧ (x - y = 3) ∧ (x = 3) ∧ (y = 0) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l977_97744


namespace NUMINAMATH_CALUDE_range_of_k_l977_97748

theorem range_of_k (x k : ℝ) : 
  (x - 1) / (x - 2) = k / (x - 2) + 2 ∧ x ≥ 0 → k ≤ 3 ∧ k ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l977_97748


namespace NUMINAMATH_CALUDE_marble_difference_l977_97714

theorem marble_difference (total : ℕ) (yellow : ℕ) (h1 : total = 913) (h2 : yellow = 514) :
  yellow - (total - yellow) = 115 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l977_97714


namespace NUMINAMATH_CALUDE_angle_b_measure_l977_97706

theorem angle_b_measure (A B C : ℝ) (a b c : ℝ) : 
  0 < B ∧ B < π →
  0 < A ∧ A < π →
  0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = b * Real.cos C + c * Real.sin B →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  B = π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_b_measure_l977_97706


namespace NUMINAMATH_CALUDE_jerry_removed_figures_l977_97730

/-- The number of old action figures removed from Jerry's shelf. -/
def old_figures_removed (initial : ℕ) (added : ℕ) (current : ℕ) : ℕ :=
  initial + added - current

/-- Theorem stating the number of old action figures Jerry removed. -/
theorem jerry_removed_figures : old_figures_removed 7 11 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jerry_removed_figures_l977_97730


namespace NUMINAMATH_CALUDE_catchup_time_correct_l977_97771

/-- Represents a person walking on the triangle -/
structure Walker where
  speed : ℝ  -- speed in meters per minute
  startVertex : ℕ  -- starting vertex (0, 1, or 2)

/-- Represents the triangle and walking scenario -/
structure TriangleWalk where
  sideLength : ℝ
  walkerA : Walker
  walkerB : Walker
  vertexDelay : ℝ  -- delay at each vertex in seconds

/-- Calculates the time when walker A catches up with walker B -/
def catchUpTime (tw : TriangleWalk) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem catchup_time_correct (tw : TriangleWalk) : 
  tw.sideLength = 200 ∧ 
  tw.walkerA = ⟨100, 0⟩ ∧ 
  tw.walkerB = ⟨80, 1⟩ ∧ 
  tw.vertexDelay = 15 → 
  catchUpTime tw = 1470 :=
sorry

end NUMINAMATH_CALUDE_catchup_time_correct_l977_97771


namespace NUMINAMATH_CALUDE_prob_B_and_C_correct_prob_4_points_correct_prob_at_least_4_points_correct_l977_97761

-- Define the probabilities
def prob_A : ℚ := 1/3
def prob_AB : ℚ := 1/6
def prob_BC : ℚ := 1/5

-- Define the success probabilities for B and C
def prob_B : ℚ := 1/2
def prob_C : ℚ := 2/5

-- Define the probability of scoring 4 points
def prob_4_points : ℚ := 3/10

-- Define the probability of scoring at least 4 points
def prob_at_least_4_points : ℚ := 11/30

-- Theorem statements
theorem prob_B_and_C_correct : 
  prob_A * prob_B = prob_AB ∧ prob_B * prob_C = prob_BC := by sorry

theorem prob_4_points_correct : 
  (1 - prob_A) * prob_B * prob_C + 
  prob_A * (1 - prob_B) * prob_C + 
  prob_A * prob_B * (1 - prob_C) = prob_4_points := by sorry

theorem prob_at_least_4_points_correct : 
  prob_4_points + prob_A * prob_B * prob_C = prob_at_least_4_points := by sorry

end NUMINAMATH_CALUDE_prob_B_and_C_correct_prob_4_points_correct_prob_at_least_4_points_correct_l977_97761


namespace NUMINAMATH_CALUDE_perpendicular_line_to_parallel_plane_l977_97766

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the non-coincidence property
variable (non_coincident_planes : Plane → Plane → Prop)
variable (non_coincident_lines : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_line_to_parallel_plane
  (α β : Plane) (m n : Line)
  (h_non_coincident_planes : non_coincident_planes α β)
  (h_non_coincident_lines : non_coincident_lines m n)
  (h_parallel_lines : parallel_lines m n)
  (h_perp_n_α : perpendicular_line_plane n α)
  (h_parallel_planes : parallel_planes α β) :
  perpendicular_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_parallel_plane_l977_97766


namespace NUMINAMATH_CALUDE_gcd_459_357_l977_97789

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l977_97789


namespace NUMINAMATH_CALUDE_decreasing_linear_function_iff_negative_slope_l977_97726

/-- A linear function f(x) = ax + b -/
def linear_function (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

/-- A function is decreasing if f(x1) > f(x2) whenever x1 < x2 -/
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

theorem decreasing_linear_function_iff_negative_slope (m : ℝ) :
  is_decreasing (linear_function (m + 3) (-2)) ↔ m < -3 :=
sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_iff_negative_slope_l977_97726


namespace NUMINAMATH_CALUDE_child_ticket_cost_l977_97729

theorem child_ticket_cost (total_seats : ℕ) (adult_ticket_cost : ℚ) 
  (num_children : ℕ) (total_revenue : ℚ) :
  total_seats = 250 →
  adult_ticket_cost = 6 →
  num_children = 188 →
  total_revenue = 1124 →
  ∃ (child_ticket_cost : ℚ),
    child_ticket_cost * num_children + 
    adult_ticket_cost * (total_seats - num_children) = total_revenue ∧
    child_ticket_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l977_97729


namespace NUMINAMATH_CALUDE_correct_rounding_l977_97702

def round_to_thousandth (x : ℚ) : ℚ :=
  (⌊x * 1000 + 0.5⌋ : ℚ) / 1000

theorem correct_rounding :
  round_to_thousandth 2.098176 = 2.098 := by sorry

end NUMINAMATH_CALUDE_correct_rounding_l977_97702


namespace NUMINAMATH_CALUDE_eight_digit_divisible_by_eleven_l977_97740

theorem eight_digit_divisible_by_eleven (n : ℕ) : 
  n < 10 →
  (965 * 10^7 + n * 10^6 + 8 * 10^5 + 4 * 10^4 + 3 * 10^3 + 2 * 10^2) % 11 = 0 →
  n = 1 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_divisible_by_eleven_l977_97740


namespace NUMINAMATH_CALUDE_g_composition_sqrt3_l977_97732

noncomputable def g (b c : ℝ) (x : ℝ) : ℝ := b * x + c * x^3 - Real.sqrt 3

theorem g_composition_sqrt3 (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  g b c (g b c (Real.sqrt 3)) = -Real.sqrt 3 → b = 0 ∧ c = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_sqrt3_l977_97732


namespace NUMINAMATH_CALUDE_specific_function_value_l977_97773

def is_odd_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = f x)

def agrees_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → f x = x

theorem specific_function_value
  (f : ℝ → ℝ)
  (h_odd_periodic : is_odd_periodic f)
  (h_unit : agrees_on_unit_interval f) :
  f 2011.5 = -0.5 := by
sorry

end NUMINAMATH_CALUDE_specific_function_value_l977_97773


namespace NUMINAMATH_CALUDE_rectangular_floor_length_l977_97786

theorem rectangular_floor_length (floor_width : ℝ) (square_size : ℝ) (num_squares : ℕ) :
  floor_width = 6 →
  square_size = 2 →
  num_squares = 15 →
  floor_width * (num_squares * square_size^2 / floor_width) = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_floor_length_l977_97786


namespace NUMINAMATH_CALUDE_surface_area_difference_l977_97757

theorem surface_area_difference (large_cube_volume : ℝ) (small_cube_volume : ℝ) (num_small_cubes : ℕ) :
  large_cube_volume = 64 →
  small_cube_volume = 1 →
  num_small_cubes = 64 →
  (num_small_cubes : ℝ) * (6 * small_cube_volume ^ (2/3)) - (6 * large_cube_volume ^ (2/3)) = 288 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_difference_l977_97757


namespace NUMINAMATH_CALUDE_rectangle_square_area_ratio_l977_97759

theorem rectangle_square_area_ratio : 
  let s : ℝ := 20
  let longer_side : ℝ := 1.05 * s
  let shorter_side : ℝ := 0.85 * s
  let area_R : ℝ := longer_side * shorter_side
  let area_S : ℝ := s * s
  area_R / area_S = 357 / 400 := by
sorry

end NUMINAMATH_CALUDE_rectangle_square_area_ratio_l977_97759


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l977_97727

theorem inequality_system_integer_solutions :
  ∀ x : ℤ, (5 * x - 1 > 3 * (x + 1) ∧ (1 + 2 * x) / 3 ≥ x - 1) ↔ (x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l977_97727


namespace NUMINAMATH_CALUDE_permutation_combination_sum_l977_97792

/-- Permutation of n elements taken r at a time -/
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then Nat.factorial n / Nat.factorial (n - r) else 0

/-- Combination of n elements taken r at a time -/
def combination (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r)) else 0

theorem permutation_combination_sum : 3 * (permutation 3 2) + 2 * (combination 4 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_sum_l977_97792


namespace NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l977_97765

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x - 2*y + 4 = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0
def l₃ (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Define the intersection point P
def P : ℝ × ℝ := sorry

-- Define the perpendicular line l
def l (x y : ℝ) : Prop := 4*x + 3*y - 6 = 0

-- Theorem statement
theorem perpendicular_line_through_intersection :
  (l₁ P.1 P.2) ∧ (l₂ P.1 P.2) ∧
  (∀ x y : ℝ, l x y ↔ (x = P.1 ∧ y = P.2 ∨ 
    (x - P.1) * 3 + (y - P.2) * (-4) = 0)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l977_97765


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l977_97769

/-- Given vectors a and b in ℝ², if |a + 2b| = |a - 2b|, then |b| = 2√5 -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  a = (-1, -2) →
  b.1 = m →
  b.2 = 2 →
  ‖a + 2 • b‖ = ‖a - 2 • b‖ →
  ‖b‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l977_97769


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l977_97770

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) →
  (3 * q^2 + 9 * q - 21 = 0) →
  (3 * p - 4) * (6 * q - 8) = 122 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l977_97770


namespace NUMINAMATH_CALUDE_sum_of_cubes_l977_97752

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 11) (h2 : a * b = 21) : a^3 + b^3 = 638 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l977_97752


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l977_97721

/-- A proportional function with a negative slope -/
structure NegativeSlopeProportionalFunction where
  k : ℝ
  k_nonzero : k ≠ 0
  k_negative : k < 0

/-- The linear function y = 2x + k -/
def linear_function (f : NegativeSlopeProportionalFunction) (x : ℝ) : ℝ := 2 * x + f.k

/-- Quadrants of the Cartesian plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Check if a point (x, y) is in a given quadrant -/
def in_quadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I  => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- The theorem stating that the linear function passes through Quadrants I, III, and IV -/
theorem linear_function_quadrants (f : NegativeSlopeProportionalFunction) :
  (∃ x y : ℝ, y = linear_function f x ∧ in_quadrant x y Quadrant.I) ∧
  (∃ x y : ℝ, y = linear_function f x ∧ in_quadrant x y Quadrant.III) ∧
  (∃ x y : ℝ, y = linear_function f x ∧ in_quadrant x y Quadrant.IV) :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l977_97721


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l977_97775

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l977_97775


namespace NUMINAMATH_CALUDE_trigonometric_identity_l977_97701

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : 0 < a) (k : 0 < b) 
  (hyp : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) : 
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l977_97701


namespace NUMINAMATH_CALUDE_trivia_team_absentees_l977_97799

/-- Proves that 6 members didn't show up to a trivia game --/
theorem trivia_team_absentees (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  total_members = 15 →
  points_per_member = 3 →
  total_points = 27 →
  total_members - (total_points / points_per_member) = 6 := by
  sorry


end NUMINAMATH_CALUDE_trivia_team_absentees_l977_97799


namespace NUMINAMATH_CALUDE_line_equation_l977_97723

/-- Circle C with center (3, 5) and radius sqrt(5) -/
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

/-- Line l passing through the center of circle C -/
structure Line_l where
  slope : ℝ
  equation : ℝ → ℝ → Prop
  passes_through_center : equation 3 5

/-- Point on the circle -/
structure Point_on_circle where
  x : ℝ
  y : ℝ
  on_circle : circle_C x y

/-- Point on the y-axis -/
structure Point_on_y_axis where
  y : ℝ

/-- Midpoint condition -/
def is_midpoint (A B P : ℝ × ℝ) : Prop :=
  A.1 = (P.1 + B.1) / 2 ∧ A.2 = (P.2 + B.2) / 2

theorem line_equation (l : Line_l) 
  (A B : Point_on_circle) 
  (P : Point_on_y_axis)
  (h_A_on_l : l.equation A.x A.y)
  (h_B_on_l : l.equation B.x B.y)
  (h_P_on_l : l.equation 0 P.y)
  (h_midpoint : is_midpoint (A.x, A.y) (B.x, B.y) (0, P.y)) :
  (∃ k : ℝ, k = 2 ∨ k = -2) ∧ 
  (∀ x y, l.equation x y ↔ y - 5 = k * (x - 3)) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l977_97723


namespace NUMINAMATH_CALUDE_total_onions_grown_l977_97795

theorem total_onions_grown (nancy_onions dan_onions mike_onions : ℕ) 
  (h1 : nancy_onions = 2)
  (h2 : dan_onions = 9)
  (h3 : mike_onions = 4) :
  nancy_onions + dan_onions + mike_onions = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_onions_grown_l977_97795


namespace NUMINAMATH_CALUDE_not_divisible_by_121_l977_97767

theorem not_divisible_by_121 : ∀ n : ℤ, ¬(121 ∣ (n^2 + 2*n + 2014)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_121_l977_97767


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l977_97782

-- Define the polynomial
def p (x a₄ a₃ a₂ a₁ a₀ : ℝ) : ℝ := (x + 2)^5 - (x + 1)^5 - (a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀)

-- State the theorem
theorem polynomial_coefficients :
  ∃ (a₄ a₃ a₂ : ℝ), ∀ x, p x a₄ a₃ a₂ 75 31 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l977_97782


namespace NUMINAMATH_CALUDE_units_digit_of_product_units_digit_of_27_times_34_l977_97735

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of a product depends only on the units digits of its factors -/
theorem units_digit_of_product (a b : ℕ) : 
  unitsDigit (a * b) = unitsDigit (unitsDigit a * unitsDigit b) := by sorry

theorem units_digit_of_27_times_34 : unitsDigit (27 * 34) = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_units_digit_of_27_times_34_l977_97735


namespace NUMINAMATH_CALUDE_expand_and_simplify_l977_97713

theorem expand_and_simplify (a : ℝ) : (2*a - 3)^2 + (2*a + 3)*(2*a - 3) = 8*a^2 - 12*a := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l977_97713


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l977_97718

/-- Given a line l with equation Ax + By + C = 0, 
    a line perpendicular to l has the equation Bx - Ay + C' = 0, 
    where C' is some constant. -/
theorem perpendicular_line_equation 
  (A B C : ℝ) (x y : ℝ → ℝ) (l : ℝ → Prop) :
  (l = λ t => A * (x t) + B * (y t) + C = 0) →
  ∃ C', ∃ l_perp : ℝ → Prop,
    (l_perp = λ t => B * (x t) - A * (y t) + C' = 0) ∧
    (∀ t, l_perp t → (∀ s, l s → 
      (x t - x s) * (A * (x t - x s) + B * (y t - y s)) + 
      (y t - y s) * (B * (x t - x s) - A * (y t - y s)) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l977_97718


namespace NUMINAMATH_CALUDE_percentage_problem_l977_97750

theorem percentage_problem (x : ℝ) (h : 160 = 320 / 100 * x) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l977_97750


namespace NUMINAMATH_CALUDE_stratified_sampling_high_group_l977_97724

/-- Represents the number of students in each height group -/
structure HeightGroups where
  low : ℕ  -- [120, 130)
  mid : ℕ  -- [130, 140)
  high : ℕ -- [140, 150]

/-- Calculates the number of students to be selected from a group in stratified sampling -/
def stratifiedSample (totalPopulation : ℕ) (groupSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (groupSize * sampleSize + totalPopulation - 1) / totalPopulation

/-- Proves that the number of students to be selected from the [140, 150] group is 3 -/
theorem stratified_sampling_high_group 
  (groups : HeightGroups)
  (h1 : groups.low + groups.mid + groups.high = 100)
  (h2 : groups.low = 20)
  (h3 : groups.mid = 50)
  (h4 : groups.high = 30)
  (totalSample : ℕ)
  (h5 : totalSample = 18) :
  stratifiedSample 100 groups.high totalSample = 3 := by
sorry

#eval stratifiedSample 100 30 18

end NUMINAMATH_CALUDE_stratified_sampling_high_group_l977_97724


namespace NUMINAMATH_CALUDE_equation_solution_exists_l977_97747

theorem equation_solution_exists (m n : ℤ) :
  ∃ (w x y z : ℤ), w + x + 2*y + 2*z = m ∧ 2*w - 2*x + y - z = n := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l977_97747


namespace NUMINAMATH_CALUDE_bus_related_time_trip_time_breakdown_l977_97700

/-- Represents the duration of Luke's trip to London in minutes -/
def total_trip_time : ℕ := 525

/-- Represents the wait time for the first bus in minutes -/
def first_bus_wait : ℕ := 25

/-- Represents the duration of the first bus ride in minutes -/
def first_bus_ride : ℕ := 40

/-- Represents the wait time for the second bus in minutes -/
def second_bus_wait : ℕ := 15

/-- Represents the duration of the second bus ride in minutes -/
def second_bus_ride : ℕ := 10

/-- Represents the walk time to the train station in minutes -/
def walk_time : ℕ := 15

/-- Represents the wait time for the train in minutes -/
def train_wait : ℕ := 2 * walk_time

/-- Represents the duration of the train ride in minutes -/
def train_ride : ℕ := 360

/-- Proves that the total bus-related time is 90 minutes -/
theorem bus_related_time :
  first_bus_wait + first_bus_ride + second_bus_wait + second_bus_ride = 90 :=
by sorry

/-- Proves that the sum of all components equals the total trip time -/
theorem trip_time_breakdown :
  first_bus_wait + first_bus_ride + second_bus_wait + second_bus_ride +
  walk_time + train_wait + train_ride = total_trip_time :=
by sorry

end NUMINAMATH_CALUDE_bus_related_time_trip_time_breakdown_l977_97700


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l977_97768

theorem arithmetic_sequence_formula (a : ℕ → ℤ) (n : ℕ) : 
  (a 1 = 1) → 
  (a 2 = 3) → 
  (a 3 = 5) → 
  (a 4 = 7) → 
  (a 5 = 9) → 
  (∀ k : ℕ, a (k + 1) - a k = 2) → 
  a n = 2 * n - 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l977_97768


namespace NUMINAMATH_CALUDE_complement_union_A_B_complement_A_inter_B_l977_97715

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- State the theorems to be proved
theorem complement_union_A_B : 
  (Set.univ : Set ℝ) \ (A ∪ B) = {x | x ≤ 2 ∨ x ≥ 10} := by sorry

theorem complement_A_inter_B : 
  ((Set.univ : Set ℝ) \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_complement_A_inter_B_l977_97715


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l977_97776

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℤ)
  (h_arith : ArithmeticSequence a)
  (h_diff : a 3 - a 2 = -2)
  (h_seventh : a 7 = -2) :
  a 9 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l977_97776


namespace NUMINAMATH_CALUDE_sequence_properties_l977_97791

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℝ := 2 * n + 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℝ := 2^n

-- Define the sum of the first n terms of a_n + b_n
def S (n : ℕ) : ℝ := n^2 + 2*n + 2^(n+1) - 2

theorem sequence_properties :
  (a 2 = 5) ∧
  (a 1 + a 4 = 12) ∧
  (∀ n, b n > 0) ∧
  (∀ n, b n * b (n+1) = 2^(a n)) ∧
  (∀ n, a n = 2*n + 1) ∧
  (∀ n, b (n+1) / b n = 2) ∧
  (∀ n, S n = n^2 + 2*n + 2^(n+1) - 2) :=
by sorry


end NUMINAMATH_CALUDE_sequence_properties_l977_97791


namespace NUMINAMATH_CALUDE_parallelogram_iff_midpoints_l977_97749

-- Define the points
variable (A B C D P Q E F : ℝ × ℝ)

-- Define the conditions
def is_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def on_diagonal (P Q B D : ℝ × ℝ) : Prop := sorry

def point_order (B P Q D : ℝ × ℝ) : Prop := sorry

def equal_segments (B P Q D : ℝ × ℝ) : Prop := sorry

def line_intersection (A P B C E : ℝ × ℝ) : Prop := sorry

def line_intersection' (A Q C D F : ℝ × ℝ) : Prop := sorry

def is_parallelogram (A B C D : ℝ × ℝ) : Prop := sorry

def is_midpoint (E B C : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem parallelogram_iff_midpoints
  (h1 : is_quadrilateral A B C D)
  (h2 : on_diagonal P Q B D)
  (h3 : point_order B P Q D)
  (h4 : equal_segments B P Q D)
  (h5 : line_intersection A P B C E)
  (h6 : line_intersection' A Q C D F) :
  is_parallelogram A B C D ↔ (is_midpoint E B C ∧ is_midpoint F C D) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_iff_midpoints_l977_97749


namespace NUMINAMATH_CALUDE_cos_negative_1500_degrees_l977_97716

theorem cos_negative_1500_degrees : Real.cos ((-1500 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_1500_degrees_l977_97716


namespace NUMINAMATH_CALUDE_complex_number_opposites_l977_97738

theorem complex_number_opposites (b : ℝ) : 
  (Complex.re ((2 - b * Complex.I) * Complex.I) = 
   -Complex.im ((2 - b * Complex.I) * Complex.I)) → b = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposites_l977_97738


namespace NUMINAMATH_CALUDE_digital_root_of_1999_factorial_l977_97785

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The digital root function -/
def digitalRoot (n : ℕ) : ℕ :=
  if n = 0 then 0 else 1 + (n - 1) % 9

/-- Theorem: The digital root of 1999! is 9 -/
theorem digital_root_of_1999_factorial :
  digitalRoot (factorial 1999) = 9 := by
  sorry

end NUMINAMATH_CALUDE_digital_root_of_1999_factorial_l977_97785


namespace NUMINAMATH_CALUDE_base_five_of_156_l977_97756

def base_five_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_five_of_156 :
  base_five_representation 156 = [1, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_five_of_156_l977_97756


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l977_97743

/-- Sums of arithmetic sequences -/
def S (n : ℕ) : ℝ := sorry

/-- Sums of arithmetic sequences -/
def T (n : ℕ) : ℝ := sorry

/-- Terms of the first arithmetic sequence -/
def a : ℕ → ℝ := sorry

/-- Terms of the second arithmetic sequence -/
def b : ℕ → ℝ := sorry

theorem arithmetic_sequence_ratio :
  (∀ n : ℕ+, S n / T n = n / (2 * n + 1)) →
  a 6 / b 6 = 11 / 23 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l977_97743


namespace NUMINAMATH_CALUDE_pants_cost_is_6_l977_97784

/-- The cost of one pair of pants -/
def pants_cost : ℚ := 6

/-- The cost of one shirt -/
def shirt_cost : ℚ := 10

/-- Theorem stating the cost of one pair of pants is $6 -/
theorem pants_cost_is_6 :
  (2 * pants_cost + 5 * shirt_cost = 62) →
  (2 * shirt_cost = 20) →
  pants_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_is_6_l977_97784


namespace NUMINAMATH_CALUDE_earthwork_inequality_l977_97758

/-- Proves the inequality for the required average daily earthwork to complete the project ahead of schedule. -/
theorem earthwork_inequality (total : ℝ) (days : ℕ) (first_day : ℝ) (ahead : ℕ) (x : ℝ) 
  (h_total : total = 300)
  (h_days : days = 6)
  (h_first_day : first_day = 60)
  (h_ahead : ahead = 2)
  : 3 * x ≥ total - first_day :=
by
  sorry

#check earthwork_inequality

end NUMINAMATH_CALUDE_earthwork_inequality_l977_97758


namespace NUMINAMATH_CALUDE_smallest_number_l977_97798

theorem smallest_number (a b c d : ℝ) (ha : a = -2) (hb : b = 0) (hc : c = 1/2) (hd : d = 2) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l977_97798


namespace NUMINAMATH_CALUDE_exists_top_choice_l977_97790

/- Define the type for houses and people -/
variable {α : Type*} [Finite α]

/- Define the preference relation -/
def Prefers (p : α → α → Prop) : Prop :=
  ∀ x y z, p x y ∧ p y z → p x z

/- Define the assignment function -/
def Assignment (f : α → α) : Prop :=
  Function.Bijective f

/- Define the stability condition -/
def Stable (f : α → α) (p : α → α → Prop) : Prop :=
  ∀ g : α → α, Assignment g →
    ∃ x, p x (f x) ∧ ¬p x (g x)

/- State the theorem -/
theorem exists_top_choice
  (f : α → α)
  (p : α → α → Prop)
  (h_assign : Assignment f)
  (h_prefers : Prefers p)
  (h_stable : Stable f p) :
  ∃ x, ∀ y, p x (f x) ∧ (p x y → y = f x) :=
sorry

end NUMINAMATH_CALUDE_exists_top_choice_l977_97790


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l977_97708

noncomputable def f (x : ℝ) : ℝ := -1 / (x + 2) + (1 / 2) * Real.log (x^2 + 4) + (1 / 2) * Real.arctan (x / 2)

theorem indefinite_integral_proof (x : ℝ) (h : x ≠ -2) : 
  deriv f x = (x^3 + 6*x^2 + 8*x + 8) / ((x + 2)^2 * (x^2 + 4)) :=
by sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l977_97708


namespace NUMINAMATH_CALUDE_power_equality_implies_q_eight_l977_97709

theorem power_equality_implies_q_eight : 16^4 = 4^q → q = 8 := by sorry

end NUMINAMATH_CALUDE_power_equality_implies_q_eight_l977_97709


namespace NUMINAMATH_CALUDE_base7_to_base10_5326_l977_97728

def base7ToBase10 (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem base7_to_base10_5326 : base7ToBase10 5 3 2 6 = 1882 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_5326_l977_97728


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l977_97760

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 2) : 
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l977_97760


namespace NUMINAMATH_CALUDE_share_distribution_l977_97739

theorem share_distribution (x y z : ℚ) (a : ℚ) : 
  (x + y + z = 156) →  -- total amount
  (y = 36) →           -- y's share
  (z = x * (1/2)) →    -- z gets 50 paisa for each rupee x gets
  (y = x * a) →        -- y gets 'a' for each rupee x gets
  (a = 9/20) := by
    sorry

end NUMINAMATH_CALUDE_share_distribution_l977_97739


namespace NUMINAMATH_CALUDE_chastity_lollipop_cost_l977_97781

def lollipop_cost (initial_money : ℚ) (remaining_money : ℚ) (num_lollipops : ℕ) (num_gummy_packs : ℕ) (gummy_pack_cost : ℚ) : ℚ :=
  ((initial_money - remaining_money) - (num_gummy_packs * gummy_pack_cost)) / num_lollipops

theorem chastity_lollipop_cost :
  lollipop_cost 15 5 4 2 2 = (3/2) :=
sorry

end NUMINAMATH_CALUDE_chastity_lollipop_cost_l977_97781


namespace NUMINAMATH_CALUDE_dynaco_shares_sold_is_150_l977_97774

/-- Represents the stock portfolio problem --/
structure StockPortfolio where
  microtron_price : ℝ
  dynaco_price : ℝ
  total_shares : ℕ
  average_price : ℝ

/-- Calculates the number of Dynaco shares sold --/
def dynaco_shares_sold (portfolio : StockPortfolio) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, 150 Dynaco shares were sold --/
theorem dynaco_shares_sold_is_150 : 
  let portfolio := StockPortfolio.mk 36 44 300 40
  dynaco_shares_sold portfolio = 150 := by
  sorry

end NUMINAMATH_CALUDE_dynaco_shares_sold_is_150_l977_97774


namespace NUMINAMATH_CALUDE_distance_between_points_l977_97719

def point1 : ℝ × ℝ := (-5, 3)
def point2 : ℝ × ℝ := (6, -9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 265 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l977_97719


namespace NUMINAMATH_CALUDE_james_singing_lessons_l977_97712

/-- Calculates the number of singing lessons James gets given the conditions --/
def number_of_lessons (lesson_cost : ℕ) (james_payment : ℕ) : ℕ :=
  let total_cost := james_payment * 2
  let initial_paid_lessons := 10
  let remaining_cost := total_cost - (initial_paid_lessons * lesson_cost)
  let additional_paid_lessons := remaining_cost / (lesson_cost * 2)
  1 + initial_paid_lessons + additional_paid_lessons

/-- Theorem stating that James gets 13 singing lessons --/
theorem james_singing_lessons :
  number_of_lessons 5 35 = 13 := by
  sorry


end NUMINAMATH_CALUDE_james_singing_lessons_l977_97712


namespace NUMINAMATH_CALUDE_fraction_evaluation_l977_97717

theorem fraction_evaluation : (3 - (-3)) / (2 - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l977_97717


namespace NUMINAMATH_CALUDE_largest_pot_cost_is_correct_l977_97787

/-- The cost of the largest pot given 6 pots with increasing prices -/
def largest_pot_cost (num_pots : ℕ) (total_cost : ℚ) (price_difference : ℚ) : ℚ :=
  let smallest_pot_cost := (total_cost - (price_difference * (num_pots - 1) * num_pots / 2)) / num_pots
  smallest_pot_cost + price_difference * (num_pots - 1)

/-- Theorem stating the cost of the largest pot -/
theorem largest_pot_cost_is_correct : 
  largest_pot_cost 6 (39/5) (1/4) = 77/40 := by
  sorry

#eval largest_pot_cost 6 (39/5) (1/4)

end NUMINAMATH_CALUDE_largest_pot_cost_is_correct_l977_97787


namespace NUMINAMATH_CALUDE_intersection_of_specific_sets_l977_97796

theorem intersection_of_specific_sets : 
  let A : Set ℕ := {1, 2, 5}
  let B : Set ℕ := {1, 3, 5}
  A ∩ B = {1, 5} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_specific_sets_l977_97796


namespace NUMINAMATH_CALUDE_cycle_selling_price_l977_97763

def cost_price : ℝ := 2800
def loss_percentage : ℝ := 25

theorem cycle_selling_price :
  let loss := (loss_percentage / 100) * cost_price
  let selling_price := cost_price - loss
  selling_price = 2100 := by sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l977_97763


namespace NUMINAMATH_CALUDE_congruence_sufficient_not_necessary_for_similarity_l977_97704

-- Define triangles
variable (T1 T2 : Type)

-- Define congruence and similarity relations
variable (congruent : T1 → T2 → Prop)
variable (similar : T1 → T2 → Prop)

-- Theorem: Triangle congruence is sufficient but not necessary for similarity
theorem congruence_sufficient_not_necessary_for_similarity :
  (∀ t1 : T1, ∀ t2 : T2, congruent t1 t2 → similar t1 t2) ∧
  ¬(∀ t1 : T1, ∀ t2 : T2, similar t1 t2 → congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_congruence_sufficient_not_necessary_for_similarity_l977_97704


namespace NUMINAMATH_CALUDE_expression_bounds_l977_97777

theorem expression_bounds (x y z w : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ 
    Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ∧
  Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l977_97777


namespace NUMINAMATH_CALUDE_fraction_order_l977_97733

theorem fraction_order : 
  (22 : ℚ) / 19 < (18 : ℚ) / 15 ∧ 
  (18 : ℚ) / 15 < (21 : ℚ) / 17 ∧ 
  (21 : ℚ) / 17 < (20 : ℚ) / 16 := by
sorry

end NUMINAMATH_CALUDE_fraction_order_l977_97733


namespace NUMINAMATH_CALUDE_sum_of_odd_three_digit_numbers_l977_97711

/-- The set of odd digits -/
def OddDigits : Finset ℕ := {1, 3, 5, 7, 9}

/-- A three-digit number with odd digits -/
structure OddThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_in_odd_digits : hundreds ∈ OddDigits
  tens_in_odd_digits : tens ∈ OddDigits
  units_in_odd_digits : units ∈ OddDigits

/-- The set of all possible odd three-digit numbers -/
def AllOddThreeDigitNumbers : Finset OddThreeDigitNumber := sorry

/-- The value of an odd three-digit number -/
def value (n : OddThreeDigitNumber) : ℕ := 100 * n.hundreds + 10 * n.tens + n.units

/-- The theorem stating the sum of all odd three-digit numbers -/
theorem sum_of_odd_three_digit_numbers :
  (AllOddThreeDigitNumbers.sum value) = 69375 := by sorry

end NUMINAMATH_CALUDE_sum_of_odd_three_digit_numbers_l977_97711


namespace NUMINAMATH_CALUDE_chase_blue_jays_count_l977_97751

/-- The number of blue jays Chase saw -/
def chase_blue_jays : ℕ := 3

/-- The number of robins Gabrielle saw -/
def gabrielle_robins : ℕ := 5

/-- The number of cardinals Gabrielle saw -/
def gabrielle_cardinals : ℕ := 4

/-- The number of blue jays Gabrielle saw -/
def gabrielle_blue_jays : ℕ := 3

/-- The number of robins Chase saw -/
def chase_robins : ℕ := 2

/-- The number of cardinals Chase saw -/
def chase_cardinals : ℕ := 5

/-- The percentage more birds Gabrielle saw compared to Chase -/
def percentage_difference : ℚ := 1/5

theorem chase_blue_jays_count :
  (gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays : ℚ) =
  (chase_robins + chase_cardinals + chase_blue_jays : ℚ) * (1 + percentage_difference) :=
by sorry

end NUMINAMATH_CALUDE_chase_blue_jays_count_l977_97751


namespace NUMINAMATH_CALUDE_chocolate_box_theorem_l977_97753

/-- Represents a box of chocolates -/
structure ChocolateBox where
  initial_count : ℕ
  rows : ℕ
  columns : ℕ

/-- The state of the box after each rearrangement -/
inductive BoxState
  | Initial
  | AfterFirstRearrange
  | AfterSecondRearrange
  | Final

/-- Function to calculate the number of chocolates at each state -/
def chocolates_at_state (box : ChocolateBox) (state : BoxState) : ℕ :=
  match state with
  | BoxState.Initial => box.initial_count
  | BoxState.AfterFirstRearrange => 3 * box.columns - 1
  | BoxState.AfterSecondRearrange => 5 * box.rows - 1
  | BoxState.Final => box.initial_count / 3

theorem chocolate_box_theorem (box : ChocolateBox) :
  chocolates_at_state box BoxState.Initial = 60 ∧
  chocolates_at_state box BoxState.Initial - chocolates_at_state box BoxState.AfterFirstRearrange = 25 :=
by sorry


end NUMINAMATH_CALUDE_chocolate_box_theorem_l977_97753


namespace NUMINAMATH_CALUDE_product_of_numbers_l977_97772

theorem product_of_numbers (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ = 2 * Real.sqrt 1703)
  (h2 : |x₁ - x₂| = 90) : 
  x₁ * x₂ = -322 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l977_97772


namespace NUMINAMATH_CALUDE_tan_40_plus_4_sin_40_equals_sqrt_3_l977_97710

theorem tan_40_plus_4_sin_40_equals_sqrt_3 : 
  Real.tan (40 * π / 180) + 4 * Real.sin (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_40_plus_4_sin_40_equals_sqrt_3_l977_97710


namespace NUMINAMATH_CALUDE_simplify_expression_l977_97762

theorem simplify_expression (x y : ℝ) : 3*x^2 - 3*(2*x^2 + 4*y) + 2*(x^2 - y) = -x^2 - 14*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l977_97762


namespace NUMINAMATH_CALUDE_vector_sum_in_R2_l977_97725

/-- Given two vectors in R², prove their sum is correct -/
theorem vector_sum_in_R2 (a b : Fin 2 → ℝ) (ha : a = ![5, 2]) (hb : b = ![1, 6]) :
  a + b = ![6, 8] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_in_R2_l977_97725


namespace NUMINAMATH_CALUDE_cone_surface_area_ratio_l977_97707

theorem cone_surface_area_ratio (r l : ℝ) (h : l = 4 * r) :
  let side_area := (1 / 2) * Real.pi * l ^ 2
  let base_area := Real.pi * r ^ 2
  let total_area := side_area + base_area
  (total_area / side_area) = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_ratio_l977_97707


namespace NUMINAMATH_CALUDE_remainder_sum_l977_97745

theorem remainder_sum (D : ℕ) (h1 : D > 0) (h2 : 242 % D = 4) (h3 : 698 % D = 8) :
  (242 + 698) % D = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l977_97745


namespace NUMINAMATH_CALUDE_least_months_to_triple_l977_97734

def interest_rate : ℝ := 1.05

theorem least_months_to_triple (n : ℕ) : (∀ m : ℕ, m < n → interest_rate ^ m ≤ 3) ∧ interest_rate ^ n > 3 ↔ n = 23 := by
  sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l977_97734


namespace NUMINAMATH_CALUDE_train_crossing_time_l977_97755

/-- Given a train and a platform with specific dimensions, calculate the time taken for the train to cross the platform. -/
theorem train_crossing_time (train_length platform_length : ℝ) (time_cross_pole : ℝ) : 
  train_length = 300 → 
  platform_length = 285 → 
  time_cross_pole = 20 → 
  (train_length + platform_length) / (train_length / time_cross_pole) = 39 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l977_97755


namespace NUMINAMATH_CALUDE_calculator_result_l977_97737

def special_key (x : ℚ) : ℚ := 1 / (1 - x)

def apply_n_times (f : ℚ → ℚ) (x : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (apply_n_times f x n)

theorem calculator_result : apply_n_times special_key 3 50 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_calculator_result_l977_97737


namespace NUMINAMATH_CALUDE_candy_problem_l977_97783

theorem candy_problem (x : ℝ) : 
  let day1_remainder := x / 2 - 3
  let day2_remainder := day1_remainder * 3/4 - 5
  let day3_remainder := day2_remainder * 4/5
  day3_remainder = 9 → x = 136 := by sorry

end NUMINAMATH_CALUDE_candy_problem_l977_97783


namespace NUMINAMATH_CALUDE_logarithm_equality_l977_97741

/-- Given the conditions on logarithms and the equation involving x^y, 
    prove that y equals 2q - p - r -/
theorem logarithm_equality (a b c x : ℝ) (p q r y : ℝ) 
  (h1 : x ≠ 1)
  (h2 : Real.log a / p = Real.log b / q)
  (h3 : Real.log b / q = Real.log c / r)
  (h4 : Real.log b / q = Real.log x)
  (h5 : b^2 / (a * c) = x^y) :
  y = 2*q - p - r := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_l977_97741


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l977_97720

theorem easter_egg_distribution (red_total orange_total : ℕ) 
  (h_red : red_total = 20) (h_orange : orange_total = 30) : ∃ (eggs_per_basket : ℕ), 
  eggs_per_basket ≥ 5 ∧ 
  red_total % eggs_per_basket = 0 ∧ 
  orange_total % eggs_per_basket = 0 ∧
  ∀ (n : ℕ), n ≥ 5 ∧ red_total % n = 0 ∧ orange_total % n = 0 → n ≥ eggs_per_basket :=
by
  sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l977_97720


namespace NUMINAMATH_CALUDE_expression_simplification_l977_97793

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - 3 / (x + 2)) / ((x^2 - 1) / (x + 2)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l977_97793


namespace NUMINAMATH_CALUDE_binomial_30_3_l977_97736

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l977_97736


namespace NUMINAMATH_CALUDE_sufficient_condition_transitivity_l977_97731

theorem sufficient_condition_transitivity (p q r : Prop) :
  (p → q) → (q → r) → (p → r) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_transitivity_l977_97731


namespace NUMINAMATH_CALUDE_dot_product_implies_x_value_l977_97754

/-- Given vectors a and b, if their dot product is 1, then the second component of b is 1. -/
theorem dot_product_implies_x_value (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 1) 
  (ha1 : a.1 = 1) (ha2 : a.2 = -1) (hb1 : b.1 = 2) : b.2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_implies_x_value_l977_97754


namespace NUMINAMATH_CALUDE_combination_permutation_equality_permutation_equation_solution_l977_97705

-- Define the combination function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the permutation function
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Theorem 1: Prove that C₁₀⁴ - C₇³ × A₃³ = 0
theorem combination_permutation_equality : C 10 4 - C 7 3 * A 3 3 = 0 := by
  sorry

-- Theorem 2: Prove that the solution to 3A₈ˣ = 4A₉ˣ⁻¹ is x = 6
theorem permutation_equation_solution :
  ∃ x : ℕ, (3 * A 8 x = 4 * A 9 (x - 1)) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_equality_permutation_equation_solution_l977_97705


namespace NUMINAMATH_CALUDE_number_of_boys_l977_97746

/-- Given conditions about men, women, and boys with their earnings, prove the number of boys --/
theorem number_of_boys (men women boys : ℕ) (total_earnings men_wage : ℕ) : 
  men = 5 →
  men = women →
  women = boys →
  total_earnings = 150 →
  men_wage = 10 →
  boys = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l977_97746


namespace NUMINAMATH_CALUDE_spatial_vector_division_not_defined_l977_97797

-- Define a spatial vector
structure SpatialVector where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define valid operations
def add (v w : SpatialVector) : SpatialVector :=
  { x := v.x + w.x, y := v.y + w.y, z := v.z + w.z }

def sub (v w : SpatialVector) : SpatialVector :=
  { x := v.x - w.x, y := v.y - w.y, z := v.z - w.z }

def scalarProduct (v w : SpatialVector) : ℝ :=
  v.x * w.x + v.y * w.y + v.z * w.z

-- Theorem stating that division is not well-defined for spatial vectors
theorem spatial_vector_division_not_defined :
  ¬ ∃ (f : SpatialVector → SpatialVector → SpatialVector),
    ∀ (v w : SpatialVector), w ≠ { x := 0, y := 0, z := 0 } →
      f v w = { x := v.x / w.x, y := v.y / w.y, z := v.z / w.z } :=
by
  sorry


end NUMINAMATH_CALUDE_spatial_vector_division_not_defined_l977_97797


namespace NUMINAMATH_CALUDE_smallest_gcd_of_b_c_l977_97779

theorem smallest_gcd_of_b_c (a b c x y : ℕ+) 
  (hab : Nat.gcd a b = 120)
  (hac : Nat.gcd a c = 1001)
  (hb : b = 120 * x)
  (hc : c = 1001 * y) :
  ∃ (b' c' : ℕ+), Nat.gcd b' c' = 1 ∧ 
    ∀ (b'' c'' : ℕ+), (∃ (x'' y'' : ℕ+), b'' = 120 * x'' ∧ c'' = 1001 * y'') → 
      Nat.gcd b'' c'' ≥ Nat.gcd b' c' :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_of_b_c_l977_97779


namespace NUMINAMATH_CALUDE_oil_percentage_in_dressing_q_l977_97778

/-- Represents the composition of a salad dressing -/
structure Dressing where
  vinegar : ℝ
  oil : ℝ

/-- Represents the mixture of two dressings -/
structure Mixture where
  dressing_p : Dressing
  dressing_q : Dressing
  p_ratio : ℝ
  q_ratio : ℝ
  vinegar : ℝ

/-- Theorem stating that given the conditions of the problem, 
    the oil percentage in dressing Q is 90% -/
theorem oil_percentage_in_dressing_q 
  (p : Dressing)
  (q : Dressing)
  (mix : Mixture)
  (h1 : p.vinegar = 0.3)
  (h2 : p.oil = 0.7)
  (h3 : q.vinegar = 0.1)
  (h4 : mix.dressing_p = p)
  (h5 : mix.dressing_q = q)
  (h6 : mix.p_ratio = 0.1)
  (h7 : mix.q_ratio = 0.9)
  (h8 : mix.vinegar = 0.12)
  : q.oil = 0.9 := by
  sorry

#check oil_percentage_in_dressing_q

end NUMINAMATH_CALUDE_oil_percentage_in_dressing_q_l977_97778


namespace NUMINAMATH_CALUDE_expected_value_after_n_centuries_l977_97764

/-- The expected value of money after n centuries in the 50 Cent game -/
def expected_value (n : ℕ) : ℚ :=
  1/2 + n * (1/4)

/-- The game starts with $0.50 -/
axiom initial_value : expected_value 0 = 1/2

/-- The recurrence relation for the expected value -/
axiom recurrence (n : ℕ) : expected_value (n + 1) = expected_value n + 1/4

theorem expected_value_after_n_centuries (n : ℕ) :
  expected_value n = 1/2 + n * (1/4) := by
  sorry

#eval expected_value 50

end NUMINAMATH_CALUDE_expected_value_after_n_centuries_l977_97764


namespace NUMINAMATH_CALUDE_kim_status_update_time_l977_97722

/-- Kim's morning routine -/
def morning_routine (coffee_time : ℕ) (payroll_time : ℕ) (num_employees : ℕ) (total_time : ℕ) (status_time : ℕ) : Prop :=
  coffee_time + num_employees * status_time + num_employees * payroll_time = total_time

/-- Theorem: Kim spends 2 minutes per employee getting a status update -/
theorem kim_status_update_time :
  ∃ (status_time : ℕ),
    morning_routine 5 3 9 50 status_time ∧
    status_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_kim_status_update_time_l977_97722
