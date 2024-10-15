import Mathlib

namespace NUMINAMATH_CALUDE_prob_same_face_is_37_64_l1481_148134

/-- A cube -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)
  faces : Finset (Finset (Fin 8))

/-- A random vertex of a cube -/
def random_vertex (C : Cube) : Fin 8 := sorry

/-- The probability that three random vertices lie on the same face of a cube -/
def prob_same_face (C : Cube) : ℚ :=
  let P := random_vertex C
  let Q := random_vertex C
  let R := random_vertex C
  sorry

/-- Theorem: The probability that three random vertices of a cube lie on the same face is 37/64 -/
theorem prob_same_face_is_37_64 (C : Cube) : prob_same_face C = 37 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_face_is_37_64_l1481_148134


namespace NUMINAMATH_CALUDE_system_solutions_correct_l1481_148148

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℝ, x - 3 * y = -10 ∧ x + y = 6 ∧ x = 2 ∧ y = 4) ∧
  -- System 2
  (∃ x y : ℝ, x / 2 - (y - 1) / 3 = 1 ∧ 4 * x - y = 8 ∧ x = 12 / 5 ∧ y = 8 / 5) :=
by
  sorry


end NUMINAMATH_CALUDE_system_solutions_correct_l1481_148148


namespace NUMINAMATH_CALUDE_quadratic_equations_properties_l1481_148184

/-- The quadratic equation x^2 + mx + 1 = 0 has two distinct negative real roots -/
def has_two_distinct_negative_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- The quadratic equation 4x^2 + (4m-2)x + 1 = 0 does not have any real roots -/
def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + (4*m-2)*x + 1 ≠ 0

theorem quadratic_equations_properties (m : ℝ) :
  (has_no_real_roots m ↔ 1/2 < m ∧ m < 3/2) ∧
  (has_two_distinct_negative_roots m ∧ ¬has_no_real_roots m ↔ m > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_properties_l1481_148184


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1481_148112

theorem sum_of_solutions_is_zero (x₁ x₂ : ℝ) (y : ℝ) : 
  y = 5 → 
  x₁^2 + y^2 = 169 → 
  x₂^2 + y^2 = 169 → 
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1481_148112


namespace NUMINAMATH_CALUDE_tan_two_theta_l1481_148127

theorem tan_two_theta (θ : Real) 
  (h1 : π / 2 < θ ∧ θ < π) -- θ is an obtuse angle
  (h2 : Real.cos (2 * θ) - Real.sin (2 * θ) = (Real.cos θ)^2) :
  Real.tan (2 * θ) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_two_theta_l1481_148127


namespace NUMINAMATH_CALUDE_lucas_sequence_property_l1481_148133

def L : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => L (n + 1) + L n

theorem lucas_sequence_property (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) :
  p ∣ (L (2 * k) - 2) → p ∣ (L (2 * k + 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_lucas_sequence_property_l1481_148133


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l1481_148116

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations
def parallel_line_line (l₁ l₂ : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def intersect_planes (p₁ p₂ : Plane) (l : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem parallel_lines_theorem 
  (a b c : Line) 
  (α β : Plane) 
  (h_non_overlapping_lines : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_non_overlapping_planes : α ≠ β)
  (h1 : parallel_line_plane a α)
  (h2 : intersect_planes α β b)
  (h3 : line_in_plane a β) :
  parallel_line_line a b :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l1481_148116


namespace NUMINAMATH_CALUDE_triangle_inequality_condition_l1481_148144

theorem triangle_inequality_condition (k : ℕ) : 
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → 
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  k ≥ 6 := by
sorry


end NUMINAMATH_CALUDE_triangle_inequality_condition_l1481_148144


namespace NUMINAMATH_CALUDE_tangent_line_length_range_l1481_148189

-- Define the circles
def circle_C1 (x y α : ℝ) : Prop := (x + Real.cos α)^2 + (y + Real.sin α)^2 = 4
def circle_C2 (x y β : ℝ) : Prop := (x - 5 * Real.sin β)^2 + (y - 5 * Real.cos β)^2 = 1

-- Define the range of α and β
def angle_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the tangent line MN
def tangent_line (M N : ℝ × ℝ) (α β : ℝ) : Prop :=
  circle_C1 M.1 M.2 α ∧ circle_C2 N.1 N.2 β ∧
  ∃ (t : ℝ), N = (M.1 + t * (N.1 - M.1), M.2 + t * (N.2 - M.2))

-- State the theorem
theorem tangent_line_length_range :
  ∀ (M N : ℝ × ℝ) (α β : ℝ),
  angle_range α → angle_range β → tangent_line M N α β →
  2 * Real.sqrt 2 ≤ Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) ∧
  Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) ≤ 3 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_length_range_l1481_148189


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1481_148111

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.b * Real.cos t.C = (2 * t.a - t.c) * Real.cos t.B)
  (h2 : t.b = Real.sqrt 7)
  (h3 : t.a + t.c = 4) :
  t.B = π / 3 ∧ (1/2 : ℝ) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1481_148111


namespace NUMINAMATH_CALUDE_log_equality_l1481_148106

theorem log_equality (y : ℝ) (k : ℝ) 
  (h1 : Real.log 3 / Real.log 8 = y)
  (h2 : Real.log 243 / Real.log 2 = k * y) : 
  k = 15 := by
sorry

end NUMINAMATH_CALUDE_log_equality_l1481_148106


namespace NUMINAMATH_CALUDE_complement_of_A_range_of_c_l1481_148102

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x - 6 ≥ 0}

-- Define set B
def B (c : ℝ) : Set ℝ := {x : ℝ | x > c}

-- Theorem for the complement of A
theorem complement_of_A : 
  {x : ℝ | x ∉ A} = {x : ℝ | -2 < x ∧ x < 3} := by sorry

-- Theorem for the range of c
theorem range_of_c :
  (∀ x : ℝ, x ∈ A ∨ x ∈ B c) → c ∈ Set.Iic (-2) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_range_of_c_l1481_148102


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l1481_148169

theorem polygon_interior_angle_sum (n : ℕ) (h : n > 2) :
  (360 / 72 : ℝ) = n →
  (n - 2) * 180 = 540 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l1481_148169


namespace NUMINAMATH_CALUDE_power_of_two_equality_l1481_148192

theorem power_of_two_equality (x : ℕ) : (1 / 16 : ℚ) * 2^50 = 2^x → x = 46 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l1481_148192


namespace NUMINAMATH_CALUDE_distance_to_school_l1481_148195

/-- The distance from the neighborhood to the school in meters. -/
def school_distance : ℝ := 960

/-- The initial speed of student A in meters per minute. -/
def speed_A_initial : ℝ := 40

/-- The initial speed of student B in meters per minute. -/
def speed_B_initial : ℝ := 60

/-- The speed of student A after increasing in meters per minute. -/
def speed_A_increased : ℝ := 60

/-- The speed of student B after decreasing in meters per minute. -/
def speed_B_decreased : ℝ := 40

/-- The time difference in minutes between A and B's arrival at school. -/
def time_difference : ℝ := 2

/-- Theorem stating that given the conditions, the distance to school is 960 meters. -/
theorem distance_to_school :
  ∀ (distance : ℝ),
  (∃ (time_A time_B : ℝ),
    distance / 2 = speed_A_initial * time_A
    ∧ distance / 2 = speed_A_increased * (time_B - time_A)
    ∧ distance = speed_B_initial * time_A + speed_B_decreased * (time_B - time_A)
    ∧ time_B + time_difference = time_A)
  → distance = school_distance :=
by sorry

end NUMINAMATH_CALUDE_distance_to_school_l1481_148195


namespace NUMINAMATH_CALUDE_equation_solution_l1481_148152

theorem equation_solution (a b : ℝ) (h : a - b = 0) : 
  ∃! x : ℝ, a * x + b = 0 ∧ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1481_148152


namespace NUMINAMATH_CALUDE_fifth_scroll_age_l1481_148154

def scroll_age (n : ℕ) : ℕ → ℕ
  | 0 => 4080
  | (m + 1) => scroll_age n m + (scroll_age n m) / 2

theorem fifth_scroll_age : scroll_age 5 4 = 20655 := by
  sorry

end NUMINAMATH_CALUDE_fifth_scroll_age_l1481_148154


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1481_148190

/-- Represents a cube arrangement with specific properties -/
structure CubeArrangement where
  num_cubes : ℕ
  central_cube_exposed_faces : ℕ
  surrounding_cubes_exposed_faces : ℕ
  extending_cube_exposed_faces : ℕ

/-- Calculate the volume of the cube arrangement -/
def volume (c : CubeArrangement) : ℕ :=
  c.num_cubes

/-- Calculate the surface area of the cube arrangement -/
def surface_area (c : CubeArrangement) : ℕ :=
  c.surrounding_cubes_exposed_faces * 5 + c.central_cube_exposed_faces + c.extending_cube_exposed_faces

/-- The specific cube arrangement described in the problem -/
def special_arrangement : CubeArrangement :=
  { num_cubes := 8,
    central_cube_exposed_faces := 1,
    surrounding_cubes_exposed_faces := 5,
    extending_cube_exposed_faces := 3 }

/-- Theorem stating the ratio of volume to surface area for the special arrangement -/
theorem volume_to_surface_area_ratio :
  (volume special_arrangement : ℚ) / (surface_area special_arrangement : ℚ) = 8 / 29 := by
  sorry


end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1481_148190


namespace NUMINAMATH_CALUDE_ratio_ties_to_losses_l1481_148129

def total_games : ℕ := 56
def losses : ℕ := 12
def wins : ℕ := 38

def ties : ℕ := total_games - (losses + wins)

theorem ratio_ties_to_losses :
  (ties : ℚ) / losses = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_ties_to_losses_l1481_148129


namespace NUMINAMATH_CALUDE_remainder_three_power_2023_mod_5_l1481_148150

theorem remainder_three_power_2023_mod_5 : 3^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_power_2023_mod_5_l1481_148150


namespace NUMINAMATH_CALUDE_non_negative_y_range_l1481_148182

theorem non_negative_y_range (x : Real) :
  0 ≤ x ∧ x ≤ Real.pi / 2 →
  (∃ y : Real, y = 4 * Real.cos x * Real.sin x + 2 * Real.cos x - 2 * Real.sin x - 1 ∧ y ≥ 0) ↔
  0 ≤ x ∧ x ≤ Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_non_negative_y_range_l1481_148182


namespace NUMINAMATH_CALUDE_particle_max_elevation_l1481_148151

/-- The elevation function for a vertically projected particle -/
def elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2 + 20

/-- The maximum elevation achieved by the particle -/
def max_elevation : ℝ := 520

/-- Theorem stating that the maximum elevation is 520 feet -/
theorem particle_max_elevation :
  ∃ t : ℝ, elevation t = max_elevation ∧ ∀ u : ℝ, elevation u ≤ max_elevation := by
  sorry

end NUMINAMATH_CALUDE_particle_max_elevation_l1481_148151


namespace NUMINAMATH_CALUDE_range_of_b_l1481_148187

/-- The function f(x) = x² + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The set A of zeros of f -/
def A (b c : ℝ) : Set ℝ := {x | f b c x = 0}

/-- The set B of x such that f(f(x)) = 0 -/
def B (b c : ℝ) : Set ℝ := {x | f b c (f b c x) = 0}

/-- If there exists x₀ in B but not in A, then b < 0 or b ≥ 4 -/
theorem range_of_b (b c : ℝ) : 
  (∃ x₀, x₀ ∈ B b c ∧ x₀ ∉ A b c) → b < 0 ∨ b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l1481_148187


namespace NUMINAMATH_CALUDE_largest_product_of_three_l1481_148180

def S : Finset Int := {-5, -4, -1, 3, 7, 9}

theorem largest_product_of_three (a b c : Int) :
  a ∈ S → b ∈ S → c ∈ S →
  a ≠ b → b ≠ c → a ≠ c →
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S →
  x ≠ y → y ≠ z → x ≠ z →
  a * b * c ≤ x * y * z →
  x * y * z ≤ 189 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l1481_148180


namespace NUMINAMATH_CALUDE_jean_thursday_calls_correct_l1481_148181

/-- The number of calls Jean answered on each day of the week --/
structure CallData where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day --/
def average_calls : ℕ := 40

/-- The number of working days --/
def working_days : ℕ := 5

/-- Jean's actual call data for the week --/
def jean_calls : CallData where
  monday := 35
  tuesday := 46
  wednesday := 27
  thursday := 61  -- This is what we want to prove
  friday := 31

/-- Theorem stating that Jean's Thursday call count is correct --/
theorem jean_thursday_calls_correct :
  jean_calls.thursday = 
    working_days * average_calls - 
    (jean_calls.monday + jean_calls.tuesday + jean_calls.wednesday + jean_calls.friday) := by
  sorry


end NUMINAMATH_CALUDE_jean_thursday_calls_correct_l1481_148181


namespace NUMINAMATH_CALUDE_adams_balls_l1481_148125

theorem adams_balls (red : ℕ) (blue : ℕ) (pink : ℕ) (orange : ℕ) : 
  red = 20 → 
  blue = 10 → 
  pink = 3 * orange → 
  orange = 5 → 
  red + blue + pink + orange = 50 := by
sorry

end NUMINAMATH_CALUDE_adams_balls_l1481_148125


namespace NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l1481_148118

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equals_negative_two :
  (1 + i)^3 / (1 - i) = -2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l1481_148118


namespace NUMINAMATH_CALUDE_smallest_x_is_correct_l1481_148145

/-- The smallest positive integer x such that 1680x is a perfect cube -/
def smallest_x : ℕ := 44100

theorem smallest_x_is_correct :
  (∀ y : ℕ, y > 0 ∧ y < smallest_x → ¬∃ m : ℤ, 1680 * y = m^3) ∧
  ∃ m : ℤ, 1680 * smallest_x = m^3 := by
  sorry

#eval smallest_x

end NUMINAMATH_CALUDE_smallest_x_is_correct_l1481_148145


namespace NUMINAMATH_CALUDE_tan_alpha_eq_two_l1481_148198

theorem tan_alpha_eq_two (α : ℝ) (h : 2 * Real.sin α + Real.cos α = -Real.sqrt 5) :
  Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_two_l1481_148198


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l1481_148135

theorem quadratic_equation_roots_range (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k - 1) * x^2 - 2 * x + 1 = 0 ∧ (k - 1) * y^2 - 2 * y + 1 = 0) →
  k ≤ 2 ∧ k ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l1481_148135


namespace NUMINAMATH_CALUDE_largest_consecutive_odd_sum_55_l1481_148199

theorem largest_consecutive_odd_sum_55 :
  (∃ (n : ℕ) (x : ℕ),
    n > 0 ∧
    x > 0 ∧
    x % 2 = 1 ∧
    n * (x + n - 1) = 55 ∧
    ∀ (m : ℕ), m > n →
      ¬∃ (y : ℕ), y > 0 ∧ y % 2 = 1 ∧ m * (y + m - 1) = 55) →
  (∃ (x : ℕ),
    x > 0 ∧
    x % 2 = 1 ∧
    11 * (x + 11 - 1) = 55 ∧
    ∀ (m : ℕ), m > 11 →
      ¬∃ (y : ℕ), y > 0 ∧ y % 2 = 1 ∧ m * (y + m - 1) = 55) :=
by sorry

end NUMINAMATH_CALUDE_largest_consecutive_odd_sum_55_l1481_148199


namespace NUMINAMATH_CALUDE_ellipse_equation_l1481_148123

/-- Given an ellipse with foci F₁(0,-4) and F₂(0,4), and the shortest distance from a point on the ellipse to F₁ is 2, the equation of the ellipse is x²/20 + y²/36 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let f₁ : ℝ × ℝ := (0, -4)
  let f₂ : ℝ × ℝ := (0, 4)
  let shortest_distance : ℝ := 2
  x^2 / 20 + y^2 / 36 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1481_148123


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l1481_148159

/-- The minimum distance from a point on the circle x^2 + y^2 = 1 to the line x - y = 2 is √2 - 1 -/
theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | x - y = 2}
  (∃ (p : ℝ × ℝ), p ∈ circle ∧
    (∀ (q : ℝ × ℝ), q ∈ circle →
      ∀ (r : ℝ × ℝ), r ∈ line →
        dist p r ≥ Real.sqrt 2 - 1)) ∧
  (∃ (p : ℝ × ℝ) (r : ℝ × ℝ), p ∈ circle ∧ r ∈ line ∧ dist p r = Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l1481_148159


namespace NUMINAMATH_CALUDE_ice_cream_ratio_l1481_148160

theorem ice_cream_ratio (sunday : ℕ) (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) 
  (h1 : sunday = 4)
  (h2 : monday = 3 * sunday)
  (h3 : tuesday = monday / 3)
  (h4 : wednesday = 18)
  (h5 : sunday + monday + tuesday = wednesday + (sunday + monday + tuesday - wednesday)) :
  (sunday + monday + tuesday - wednesday) / tuesday = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_ratio_l1481_148160


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l1481_148156

theorem geometric_arithmetic_sequence (a b c : ℝ) (q : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Ensure positive terms
  a > b → b > c →  -- Decreasing sequence
  b = a * q →  -- Geometric progression
  c = b * q →  -- Geometric progression
  2 * (2020 * b / 7) = 577 * a + c / 7 →  -- Arithmetic progression condition
  q = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l1481_148156


namespace NUMINAMATH_CALUDE_larger_number_proof_l1481_148173

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1311) (h3 : L = 11 * S + 11) : L = 1441 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1481_148173


namespace NUMINAMATH_CALUDE_count_descending_digit_numbers_l1481_148109

/-- The number of natural numbers with 2 or more digits where each subsequent digit is less than the previous one -/
def descending_digit_numbers : ℕ :=
  (Finset.range 9).sum (fun k => Nat.choose 10 (k + 2))

/-- Theorem stating that the number of natural numbers with 2 or more digits 
    where each subsequent digit is less than the previous one is 1013 -/
theorem count_descending_digit_numbers : descending_digit_numbers = 1013 := by
  sorry

end NUMINAMATH_CALUDE_count_descending_digit_numbers_l1481_148109


namespace NUMINAMATH_CALUDE_slope_of_line_from_equation_l1481_148175

theorem slope_of_line_from_equation (x y : ℝ) (h : (4 / x) + (5 / y) = 0) :
  ∃ m : ℝ, m = -5/4 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (4 / x₁ + 5 / y₁ = 0) → (4 / x₂ + 5 / y₂ = 0) → x₁ ≠ x₂ →
    (y₂ - y₁) / (x₂ - x₁) = m :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_from_equation_l1481_148175


namespace NUMINAMATH_CALUDE_investment_timing_l1481_148107

/-- Given two investments A and B, where A invests for the full year and B invests for part of the year,
    prove that B's investment starts 6 months after A's if their total investment-months are equal. -/
theorem investment_timing (a_amount : ℝ) (b_amount : ℝ) (total_months : ℕ) (x : ℝ) :
  a_amount > 0 →
  b_amount > 0 →
  total_months = 12 →
  a_amount * total_months = b_amount * (total_months - x) →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_investment_timing_l1481_148107


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l1481_148188

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 5) :
  let c := Real.sqrt (a^2 - b^2)
  (5 * Real.sqrt 11) / 2 = min ((a * b) / 2) ((b * c) / 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l1481_148188


namespace NUMINAMATH_CALUDE_total_meals_is_48_l1481_148179

/-- Represents the number of entree options --/
def num_entrees : ℕ := 4

/-- Represents the number of drink options --/
def num_drinks : ℕ := 4

/-- Represents the number of dessert options (including "no dessert") --/
def num_desserts : ℕ := 3

/-- Calculates the total number of possible meal combinations --/
def total_meals : ℕ := num_entrees * num_drinks * num_desserts

/-- Theorem stating that the total number of possible meals is 48 --/
theorem total_meals_is_48 : total_meals = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_meals_is_48_l1481_148179


namespace NUMINAMATH_CALUDE_solve_system_for_x_l1481_148104

theorem solve_system_for_x (x y z : ℚ) 
  (eq1 : 3 * x - 2 * y + z = 8) 
  (eq2 : x + 3 * y - 2 * z = 2) : 
  x = 58 / 21 := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_x_l1481_148104


namespace NUMINAMATH_CALUDE_triangle_property_l1481_148191

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.b * Real.tan t.B = Real.sqrt 3 * (t.a * Real.cos t.C + t.c * Real.cos t.A))
  (h2 : t.b = 2 * Real.sqrt 3)
  (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3) :
  t.B = π / 3 ∧ t.a + t.c = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1481_148191


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l1481_148100

theorem complex_roots_on_circle : 
  ∀ z : ℂ, (z - 2)^4 = 16 * z^4 → Complex.abs (z - (2/3 : ℂ)) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l1481_148100


namespace NUMINAMATH_CALUDE_subtracted_amount_l1481_148178

/-- Given a number N = 200, if 95% of N minus some amount A equals 178, then A must be 12. -/
theorem subtracted_amount (N : ℝ) (A : ℝ) : 
  N = 200 → 0.95 * N - A = 178 → A = 12 := by sorry

end NUMINAMATH_CALUDE_subtracted_amount_l1481_148178


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1481_148193

theorem max_value_of_expression (a b : ℝ) (h : a^2 + b^2 = 9) :
  ∃ (max : ℝ), max = 5 ∧ ∀ (x y : ℝ), x^2 + y^2 = 9 → x * y - y + x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1481_148193


namespace NUMINAMATH_CALUDE_shaded_region_area_l1481_148115

/-- Given a figure composed of 25 congruent squares, where the diagonal of a square 
    formed by 16 of these squares is 10 cm, the total area of the figure is 78.125 square cm. -/
theorem shaded_region_area (num_squares : ℕ) (diagonal : ℝ) (total_area : ℝ) : 
  num_squares = 25 → 
  diagonal = 10 → 
  total_area = 78.125 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_area_l1481_148115


namespace NUMINAMATH_CALUDE_triangle_special_progression_l1481_148136

theorem triangle_special_progression (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Arithmetic progression of sides
  2 * b = a + c →
  -- Geometric progression of sines
  Real.sin B ^ 2 = Real.sin A * Real.sin C →
  -- Conclusion
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_progression_l1481_148136


namespace NUMINAMATH_CALUDE_square_less_than_self_for_unit_interval_l1481_148167

theorem square_less_than_self_for_unit_interval (x : ℝ) : 0 < x → x < 1 → x^2 < x := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_self_for_unit_interval_l1481_148167


namespace NUMINAMATH_CALUDE_seven_twentyfour_twentyfive_pythagorean_triple_l1481_148157

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem seven_twentyfour_twentyfive_pythagorean_triple :
  is_pythagorean_triple 7 24 25 :=
by
  sorry

end NUMINAMATH_CALUDE_seven_twentyfour_twentyfive_pythagorean_triple_l1481_148157


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_l1481_148141

theorem sqrt_expression_equals_three :
  (Real.sqrt 2 + 1)^2 - Real.sqrt 18 + 2 * Real.sqrt (1/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_l1481_148141


namespace NUMINAMATH_CALUDE_vector_parallelism_l1481_148140

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 1]
def b (x : ℝ) : Fin 2 → ℝ := ![2, x]

-- Define the condition for parallel vectors
def parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

-- State the theorem
theorem vector_parallelism (x : ℝ) :
  parallel (λ i => a i + b x i) (λ i => a i - b x i) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallelism_l1481_148140


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1481_148138

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l1481_148138


namespace NUMINAMATH_CALUDE_total_cost_approx_l1481_148194

/-- Calculate the final price of an item after discounts and tax -/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (taxRate : ℝ) : ℝ :=
  let priceAfterDiscount1 := originalPrice * (1 - discount1)
  let priceAfterDiscount2 := priceAfterDiscount1 * (1 - discount2)
  let taxAmount := priceAfterDiscount2 * taxRate
  priceAfterDiscount2 + taxAmount

/-- Calculate the total cost of all items -/
def totalCost (item1Price : ℝ) (item2Price : ℝ) (item3Price : ℝ) : ℝ :=
  let item1 := finalPrice item1Price 0.25 0.15 0.07
  let item2 := finalPrice item2Price 0.30 0 0.10
  let item3 := finalPrice item3Price 0.20 0 0.05
  item1 + item2 + item3

/-- Theorem: The total cost for all three items is approximately $335.93 -/
theorem total_cost_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ abs (totalCost 200 150 100 - 335.93) < ε :=
sorry

end NUMINAMATH_CALUDE_total_cost_approx_l1481_148194


namespace NUMINAMATH_CALUDE_drugstore_inventory_theorem_l1481_148168

def bottles_delivered (initial_inventory : ℕ) (monday_sales : ℕ) (tuesday_sales : ℕ) (daily_sales_wed_to_sun : ℕ) (final_inventory : ℕ) : ℕ :=
  let total_sales := monday_sales + tuesday_sales + (daily_sales_wed_to_sun * 5)
  let remaining_before_delivery := initial_inventory - (monday_sales + tuesday_sales + (daily_sales_wed_to_sun * 4))
  final_inventory - remaining_before_delivery

theorem drugstore_inventory_theorem (initial_inventory : ℕ) (monday_sales : ℕ) (tuesday_sales : ℕ) (daily_sales_wed_to_sun : ℕ) (final_inventory : ℕ) 
  (h1 : initial_inventory = 4500)
  (h2 : monday_sales = 2445)
  (h3 : tuesday_sales = 900)
  (h4 : daily_sales_wed_to_sun = 50)
  (h5 : final_inventory = 1555) :
  bottles_delivered initial_inventory monday_sales tuesday_sales daily_sales_wed_to_sun final_inventory = 600 := by
  sorry

end NUMINAMATH_CALUDE_drugstore_inventory_theorem_l1481_148168


namespace NUMINAMATH_CALUDE_median_salary_is_clerk_salary_l1481_148170

/-- Represents a position in the company -/
inductive Position
  | CEO
  | SeniorManager
  | Manager
  | AssistantManager
  | Clerk

/-- Returns the number of employees for a given position -/
def employeeCount (p : Position) : ℕ :=
  match p with
  | .CEO => 1
  | .SeniorManager => 8
  | .Manager => 12
  | .AssistantManager => 10
  | .Clerk => 40

/-- Returns the salary for a given position -/
def salary (p : Position) : ℕ :=
  match p with
  | .CEO => 180000
  | .SeniorManager => 95000
  | .Manager => 70000
  | .AssistantManager => 55000
  | .Clerk => 28000

/-- The total number of employees in the company -/
def totalEmployees : ℕ := 71

/-- Theorem stating that the median salary is equal to the Clerk's salary -/
theorem median_salary_is_clerk_salary :
  (totalEmployees + 1) / 2 ≤ (employeeCount Position.Clerk) ∧
  (totalEmployees + 1) / 2 > (totalEmployees - employeeCount Position.Clerk) →
  salary Position.Clerk = 28000 := by
  sorry

#check median_salary_is_clerk_salary

end NUMINAMATH_CALUDE_median_salary_is_clerk_salary_l1481_148170


namespace NUMINAMATH_CALUDE_circle_center_on_line_max_ab_l1481_148177

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x - b*y + 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

theorem circle_center_on_line_max_ab :
  ∀ (a b : ℝ),
  line_equation a b (circle_center.1) (circle_center.2) →
  a * b ≤ 1/8 ∧
  ∀ (ε : ℝ), ε > 0 → ∃ (a' b' : ℝ), 
    line_equation a' b' (circle_center.1) (circle_center.2) ∧
    a' * b' > 1/8 - ε :=
by sorry

end NUMINAMATH_CALUDE_circle_center_on_line_max_ab_l1481_148177


namespace NUMINAMATH_CALUDE_softball_opponent_score_l1481_148131

theorem softball_opponent_score :
  let team_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let num_games := team_scores.length
  let num_losses := (team_scores.filter (λ x => x % 2 = 1)).length
  let opponent_scores_losses := (team_scores.filter (λ x => x % 2 = 1)).map (λ x => x + 1)
  let opponent_scores_wins := (team_scores.filter (λ x => x % 2 = 0)).map (λ x => x / 2)
  num_games = 10 →
  num_losses = 5 →
  opponent_scores_losses.sum + opponent_scores_wins.sum = 45 :=
by sorry

end NUMINAMATH_CALUDE_softball_opponent_score_l1481_148131


namespace NUMINAMATH_CALUDE_product_of_4_7_25_l1481_148171

theorem product_of_4_7_25 : 4 * 7 * 25 = 700 := by
  sorry

end NUMINAMATH_CALUDE_product_of_4_7_25_l1481_148171


namespace NUMINAMATH_CALUDE_mixture_cost_july_l1481_148185

/-- The cost of a mixture of milk powder and coffee in July -/
def mixture_cost (june_cost : ℝ) : ℝ :=
  let july_coffee_cost := june_cost * 4
  let july_milk_cost := june_cost * 0.2
  (1.5 * july_coffee_cost) + (1.5 * july_milk_cost)

/-- Theorem: The cost of a 3 lbs mixture of equal parts milk powder and coffee in July is $6.30 -/
theorem mixture_cost_july : ∃ (june_cost : ℝ), 
  (june_cost * 0.2 = 0.20) ∧ (mixture_cost june_cost = 6.30) := by
  sorry

end NUMINAMATH_CALUDE_mixture_cost_july_l1481_148185


namespace NUMINAMATH_CALUDE_total_revenue_is_628_l1481_148128

/-- Represents the characteristics of a pie type -/
structure PieType where
  slices_per_pie : ℕ
  price_per_slice : ℕ
  pies_sold : ℕ

/-- Calculates the revenue for a single pie type -/
def revenue_for_pie_type (pie : PieType) : ℕ :=
  pie.slices_per_pie * pie.price_per_slice * pie.pies_sold

/-- Defines the pumpkin pie -/
def pumpkin_pie : PieType :=
  { slices_per_pie := 8, price_per_slice := 5, pies_sold := 4 }

/-- Defines the custard pie -/
def custard_pie : PieType :=
  { slices_per_pie := 6, price_per_slice := 6, pies_sold := 5 }

/-- Defines the apple pie -/
def apple_pie : PieType :=
  { slices_per_pie := 10, price_per_slice := 4, pies_sold := 3 }

/-- Defines the pecan pie -/
def pecan_pie : PieType :=
  { slices_per_pie := 12, price_per_slice := 7, pies_sold := 2 }

/-- Theorem stating that the total revenue from all pie sales is $628 -/
theorem total_revenue_is_628 :
  revenue_for_pie_type pumpkin_pie +
  revenue_for_pie_type custard_pie +
  revenue_for_pie_type apple_pie +
  revenue_for_pie_type pecan_pie = 628 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_628_l1481_148128


namespace NUMINAMATH_CALUDE_distance_after_three_minutes_l1481_148103

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

/-- Theorem: The distance between two vehicles with speeds 65 km/h and 85 km/h after 3 minutes is 1 km -/
theorem distance_after_three_minutes :
  let v1 : ℝ := 65  -- Speed of the truck in km/h
  let v2 : ℝ := 85  -- Speed of the car in km/h
  let t : ℝ := 3 / 60  -- 3 minutes converted to hours
  distance_between_vehicles v1 v2 t = 1 := by
  sorry


end NUMINAMATH_CALUDE_distance_after_three_minutes_l1481_148103


namespace NUMINAMATH_CALUDE_nested_radical_value_l1481_148183

theorem nested_radical_value : 
  ∃ x : ℝ, x = Real.sqrt (2 + x) ∧ x ≥ 0 ∧ 2 + x ≥ 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l1481_148183


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l1481_148147

/-- The perimeter of a pentagon with side lengths 2, √8, √18, √32, and √62 is 2 + 9√2 + √62 -/
theorem pentagon_perimeter : 
  let side1 : ℝ := 2
  let side2 : ℝ := Real.sqrt 8
  let side3 : ℝ := Real.sqrt 18
  let side4 : ℝ := Real.sqrt 32
  let side5 : ℝ := Real.sqrt 62
  side1 + side2 + side3 + side4 + side5 = 2 + 9 * Real.sqrt 2 + Real.sqrt 62 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_perimeter_l1481_148147


namespace NUMINAMATH_CALUDE_fraction_unchanged_l1481_148132

theorem fraction_unchanged (x y : ℝ) (h : x ≠ y) :
  (3 * x) / (x - y) = (3 * (2 * x)) / ((2 * x) - (2 * y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l1481_148132


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l1481_148105

theorem min_value_of_sum_of_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 25) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 160 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l1481_148105


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_three_l1481_148174

/-- The probability that at least one of three independent events occurs -/
theorem prob_at_least_one_of_three (p₁ p₂ p₃ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) 
  (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) 
  (h₃ : 0 ≤ p₃ ∧ p₃ ≤ 1) : 
  1 - (1 - p₁) * (1 - p₂) * (1 - p₃) = 
  1 - ((1 - p₁) * (1 - p₂) * (1 - p₃)) :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_three_l1481_148174


namespace NUMINAMATH_CALUDE_remainder_17_65_mod_7_l1481_148196

theorem remainder_17_65_mod_7 : 17^65 % 7 = 5 := by sorry

end NUMINAMATH_CALUDE_remainder_17_65_mod_7_l1481_148196


namespace NUMINAMATH_CALUDE_perpendicular_length_l1481_148197

/-- Given a triangle ABC with angle ABC = 135°, AB = 2, and BC = 5,
    if perpendiculars are constructed to AB at A and to BC at C meeting at point D,
    then CD = 5√2 -/
theorem perpendicular_length (A B C D : ℝ × ℝ) : 
  let angleABC : ℝ := 135 * π / 180
  let AB : ℝ := 2
  let BC : ℝ := 5
  ∀ (perpAB : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0)
    (perpBC : (D.1 - C.1) * (B.1 - C.1) + (D.2 - C.2) * (B.2 - C.2) = 0),
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_length_l1481_148197


namespace NUMINAMATH_CALUDE_unique_solution_l1481_148101

theorem unique_solution (x y z : ℝ) :
  (Real.sqrt (x^3 - y) = z - 1) ∧
  (Real.sqrt (y^3 - z) = x - 1) ∧
  (Real.sqrt (z^3 - x) = y - 1) →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l1481_148101


namespace NUMINAMATH_CALUDE_total_seashells_is_58_l1481_148166

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The total number of seashells found -/
def total_seashells : ℕ := tom_seashells + fred_seashells

/-- Theorem stating that the total number of seashells found is 58 -/
theorem total_seashells_is_58 : total_seashells = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_is_58_l1481_148166


namespace NUMINAMATH_CALUDE_physical_examination_count_l1481_148113

theorem physical_examination_count (boys girls examined : ℕ) 
  (h1 : boys = 121)
  (h2 : girls = 83)
  (h3 : examined = 150) :
  boys + girls - examined = 54 := by
  sorry

end NUMINAMATH_CALUDE_physical_examination_count_l1481_148113


namespace NUMINAMATH_CALUDE_carls_weight_l1481_148108

/-- Given the weights of Al, Ben, Carl, and Ed, prove Carl's weight -/
theorem carls_weight (Al Ben Carl Ed : ℕ) 
  (h1 : Al = Ben + 25)
  (h2 : Ben = Carl - 16)
  (h3 : Ed = 146)
  (h4 : Al = Ed + 38) :
  Carl = 175 := by
  sorry

end NUMINAMATH_CALUDE_carls_weight_l1481_148108


namespace NUMINAMATH_CALUDE_infinite_series_equality_l1481_148149

theorem infinite_series_equality (p q : ℝ) 
  (h : ∑' n, p / q^n = 5) :
  ∑' n, p / (p^2 + q)^n = 5 * (q - 1) / (25 * q^2 - 50 * q + 26) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_equality_l1481_148149


namespace NUMINAMATH_CALUDE_cyclists_problem_l1481_148164

/-- The problem of two cyclists traveling between Huntington and Montauk -/
theorem cyclists_problem (x y : ℝ) : 
  (y = x + 6) →                   -- Y is 6 mph faster than X
  (80 / x = (80 + 16) / y) →      -- Time taken by X equals time taken by Y
  (x = 12) :=                     -- X's speed is 12 mph
by sorry

end NUMINAMATH_CALUDE_cyclists_problem_l1481_148164


namespace NUMINAMATH_CALUDE_equation_solutions_l1481_148142

theorem equation_solutions : 
  (∃ x : ℝ, 2 * x + 62 = 248 ∧ x = 93) ∧
  (∃ x : ℝ, x - 12.7 = 2.7 ∧ x = 15.4) ∧
  (∃ x : ℝ, x / 5 = 0.16 ∧ x = 0.8) ∧
  (∃ x : ℝ, 7 * x + 2 * x = 6.3 ∧ x = 0.7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1481_148142


namespace NUMINAMATH_CALUDE_no_overlapping_attendees_l1481_148176

theorem no_overlapping_attendees (total_guests : ℕ) 
  (oates_attendees hall_attendees singh_attendees brown_attendees : ℕ) :
  total_guests = 350 ∧
  oates_attendees = 105 ∧
  hall_attendees = 98 ∧
  singh_attendees = 82 ∧
  brown_attendees = 65 ∧
  oates_attendees + hall_attendees + singh_attendees + brown_attendees = total_guests →
  (∃ (overlapping_attendees : ℕ), overlapping_attendees = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_overlapping_attendees_l1481_148176


namespace NUMINAMATH_CALUDE_product_divisors_24_power_5_l1481_148139

/-- The product of divisors function -/
def prod_divisors (n : ℕ+) : ℕ+ :=
  sorry

theorem product_divisors_24_power_5 (n : ℕ+) :
  prod_divisors n = (24 : ℕ+) ^ 240 → n = (24 : ℕ+) ^ 5 := by
  sorry

end NUMINAMATH_CALUDE_product_divisors_24_power_5_l1481_148139


namespace NUMINAMATH_CALUDE_odd_function_equivalence_l1481_148126

theorem odd_function_equivalence (f : ℝ → ℝ) : 
  (∀ x, f x + f (-x) = 0) ↔ (∀ x, f (-x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_odd_function_equivalence_l1481_148126


namespace NUMINAMATH_CALUDE_donald_oranges_l1481_148162

def final_oranges (initial found given_away : ℕ) : ℕ :=
  initial + found - given_away

theorem donald_oranges : 
  final_oranges 4 5 3 = 6 := by sorry

end NUMINAMATH_CALUDE_donald_oranges_l1481_148162


namespace NUMINAMATH_CALUDE_oil_water_ratio_l1481_148165

/-- Represents the capacity and contents of a bottle -/
structure Bottle where
  capacity : ℝ
  oil : ℝ
  water : ℝ

/-- The problem setup -/
def bottleProblem (C_A : ℝ) : Prop :=
  ∃ (A B C D : Bottle),
    A.capacity = C_A ∧
    A.oil = C_A / 2 ∧
    A.water = C_A / 2 ∧
    B.capacity = 2 * C_A ∧
    B.oil = C_A / 2 ∧
    B.water = 3 * C_A / 2 ∧
    C.capacity = 3 * C_A ∧
    C.oil = C_A ∧
    C.water = 2 * C_A ∧
    D.capacity = 4 * C_A ∧
    D.oil = 0 ∧
    D.water = 0

/-- The theorem to prove -/
theorem oil_water_ratio (C_A : ℝ) (h : C_A > 0) :
  bottleProblem C_A →
  ∃ (D_final : Bottle),
    D_final.capacity = 4 * C_A ∧
    D_final.oil = 2 * C_A ∧
    D_final.water = 3.7 * C_A :=
by
  sorry

#check oil_water_ratio

end NUMINAMATH_CALUDE_oil_water_ratio_l1481_148165


namespace NUMINAMATH_CALUDE_apple_basket_problem_l1481_148137

theorem apple_basket_problem (n : ℕ) : 
  (27 * n > 25 * n) ∧                  -- A has more apples than B
  (27 * n - 4 < 25 * n + 4) ∧          -- Moving 4 apples makes B have more
  (27 * n - 3 ≥ 25 * n + 3) →          -- Moving 3 apples doesn't make B have more
  27 * n + 25 * n = 156 :=              -- Total number of apples
by sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l1481_148137


namespace NUMINAMATH_CALUDE_complement_A_union_B_a_lower_bound_l1481_148124

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for part I
theorem complement_A_union_B :
  (Set.univ \ A) ∪ B = {x : ℝ | x < 1 ∨ x > 2} := by sorry

-- Theorem for part II
theorem a_lower_bound (h : A ⊆ C a) : a ≥ 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_a_lower_bound_l1481_148124


namespace NUMINAMATH_CALUDE_larger_number_proof_l1481_148110

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) : 
  L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1481_148110


namespace NUMINAMATH_CALUDE_equation_solution_l1481_148186

theorem equation_solution : ∃! x : ℝ, 2 * x + 4 = |(-17 + 3)| :=
  by
    -- The unique solution is x = 5
    use 5
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_equation_solution_l1481_148186


namespace NUMINAMATH_CALUDE_dans_cards_count_l1481_148158

/-- The number of Pokemon cards Dan gave to Sally -/
def cards_from_dan (initial_cards : ℕ) (bought_cards : ℕ) (total_cards : ℕ) : ℕ :=
  total_cards - initial_cards - bought_cards

/-- Theorem stating that Dan gave Sally 41 Pokemon cards -/
theorem dans_cards_count : cards_from_dan 27 20 88 = 41 := by
  sorry

end NUMINAMATH_CALUDE_dans_cards_count_l1481_148158


namespace NUMINAMATH_CALUDE_inequality_proof_l1481_148121

theorem inequality_proof (x₃ x₄ : ℝ) (h1 : 1 < x₃) (h2 : x₃ < x₄) :
  x₃ * Real.exp x₄ > x₄ * Real.exp x₃ := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1481_148121


namespace NUMINAMATH_CALUDE_cube_root_three_equation_l1481_148120

theorem cube_root_three_equation (t : ℝ) : 
  t = 1 / (1 - Real.rpow 3 (1/3)) → 
  t = -(1 + Real.rpow 3 (1/3) + Real.rpow 3 (2/3)) / 2 := by
sorry

end NUMINAMATH_CALUDE_cube_root_three_equation_l1481_148120


namespace NUMINAMATH_CALUDE_number_with_given_division_l1481_148153

theorem number_with_given_division : ∃ n : ℕ, n = 100 ∧ n / 11 = 9 ∧ n % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_with_given_division_l1481_148153


namespace NUMINAMATH_CALUDE_f_not_bounded_on_neg_reals_a_range_when_f_bounded_l1481_148130

-- Define the function f(x) = 1 + x + ax^2
def f (a : ℝ) (x : ℝ) : ℝ := 1 + x + a * x^2

-- Part 1: f(x) is not bounded on (-∞, 0) when a = -1
theorem f_not_bounded_on_neg_reals :
  ¬ ∃ (M : ℝ), ∀ (x : ℝ), x < 0 → |f (-1) x| ≤ M :=
sorry

-- Part 2: If |f(x)| ≤ 3 for all x ∈ [1, 4], then a ∈ [-1/2, -1/8]
theorem a_range_when_f_bounded (a : ℝ) :
  (∀ x, x ∈ Set.Icc 1 4 → |f a x| ≤ 3) →
  a ∈ Set.Icc (-1/2) (-1/8) :=
sorry

end NUMINAMATH_CALUDE_f_not_bounded_on_neg_reals_a_range_when_f_bounded_l1481_148130


namespace NUMINAMATH_CALUDE_six_by_six_square_1x4_rectangles_impossible_l1481_148172

theorem six_by_six_square_1x4_rectangles_impossible : ¬ ∃ (a b : ℕ), 
  a + 4*b = 6 ∧ 4*a + b = 6 :=
sorry

end NUMINAMATH_CALUDE_six_by_six_square_1x4_rectangles_impossible_l1481_148172


namespace NUMINAMATH_CALUDE_largest_angle_is_70_l1481_148119

-- Define a right angle in degrees
def right_angle : ℝ := 90

-- Define the triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_eq_180 : angle1 + angle2 + angle3 = 180
  all_positive : 0 < angle1 ∧ 0 < angle2 ∧ 0 < angle3

-- Define the specific conditions of our triangle
def special_triangle (t : Triangle) : Prop :=
  ∃ (x : ℝ), 
    t.angle1 = x ∧ 
    t.angle2 = x + 20 ∧
    t.angle1 + t.angle2 = (4/3) * right_angle

-- Theorem statement
theorem largest_angle_is_70 (t : Triangle) (h : special_triangle t) : 
  max t.angle1 (max t.angle2 t.angle3) = 70 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_is_70_l1481_148119


namespace NUMINAMATH_CALUDE_pumps_to_fill_tires_l1481_148146

/-- Represents the capacity of a single tire in cubic inches -/
def tireCapacity : ℝ := 500

/-- Represents the amount of air injected per pump in cubic inches -/
def airPerPump : ℝ := 50

/-- Calculates the total air needed to fill all tires -/
def totalAirNeeded : ℝ :=
  2 * tireCapacity +  -- Two flat tires
  0.6 * tireCapacity +  -- Tire that's 40% full needs 60% more
  0.3 * tireCapacity  -- Tire that's 70% full needs 30% more

/-- Theorem: The number of pumps required to fill all tires is 29 -/
theorem pumps_to_fill_tires : 
  ⌈totalAirNeeded / airPerPump⌉ = 29 := by sorry

end NUMINAMATH_CALUDE_pumps_to_fill_tires_l1481_148146


namespace NUMINAMATH_CALUDE_remainder_of_n_squared_plus_4n_plus_10_l1481_148117

theorem remainder_of_n_squared_plus_4n_plus_10 (n : ℤ) (a : ℤ) (h : n = 100 * a - 2) :
  (n^2 + 4*n + 10) % 100 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_squared_plus_4n_plus_10_l1481_148117


namespace NUMINAMATH_CALUDE_eustace_milford_age_ratio_l1481_148155

theorem eustace_milford_age_ratio :
  ∀ (eustace_age milford_age : ℕ),
    eustace_age + 3 = 39 →
    milford_age + 3 = 21 →
    eustace_age / milford_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_eustace_milford_age_ratio_l1481_148155


namespace NUMINAMATH_CALUDE_stratified_sampling_l1481_148114

/-- Represents the number of students in each grade and the sample size -/
structure SchoolSample where
  total : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  sample_first : ℕ

/-- The conditions of the problem -/
def school_conditions (s : SchoolSample) : Prop :=
  s.total = 1290 ∧
  s.first_grade = 480 ∧
  s.second_grade = s.third_grade + 30 ∧
  s.total = s.first_grade + s.second_grade + s.third_grade ∧
  s.sample_first = 96

/-- The theorem to prove -/
theorem stratified_sampling (s : SchoolSample) 
  (h : school_conditions s) : 
  (s.sample_first * s.second_grade) / s.first_grade = 78 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_l1481_148114


namespace NUMINAMATH_CALUDE_secret_spread_theorem_l1481_148122

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day of the week given a number of days after Monday -/
def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Monday"
  | 1 => "Tuesday"
  | 2 => "Wednesday"
  | 3 => "Thursday"
  | 4 => "Friday"
  | 5 => "Saturday"
  | _ => "Sunday"

theorem secret_spread_theorem :
  ∃ n : ℕ, secret_spread n = 3280 ∧ day_of_week n = "Monday" :=
by
  sorry

end NUMINAMATH_CALUDE_secret_spread_theorem_l1481_148122


namespace NUMINAMATH_CALUDE_sets_problem_l1481_148163

def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

theorem sets_problem (a : ℝ) :
  (A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 8}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | 5 ≤ x ∧ x < 8}) ∧
  (C a ∩ A = C a → a ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_sets_problem_l1481_148163


namespace NUMINAMATH_CALUDE_power_division_l1481_148161

theorem power_division (x : ℝ) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1481_148161


namespace NUMINAMATH_CALUDE_linear_function_property_l1481_148143

theorem linear_function_property :
  ∀ x y : ℝ, y = -2 * x + 1 → x > (1/2 : ℝ) → y < 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l1481_148143
