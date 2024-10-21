import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_eq_open_unit_interval_l713_71359

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x ≥ 0, y = (1/2) ^ x}
def Q : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the open interval (0, 1]
def OpenUnitInterval : Set ℝ := {x | 0 < x ∧ x ≤ 1}

-- State the theorem
theorem intersection_eq_open_unit_interval : P ∩ Q = OpenUnitInterval := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_eq_open_unit_interval_l713_71359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_linear_function_on_interval_l713_71354

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_linear_function_on_interval :
  ∀ (f : ℝ → ℝ) (a b : ℝ),
  (∀ x, -1 < x ∧ x < 1 → f x = a * x + b) →
  is_odd f →
  (∀ x, -1 < x ∧ x < 1 → f x = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_linear_function_on_interval_l713_71354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l713_71319

-- Define the function f and its domain
def f : ℝ → ℝ := sorry

-- Define the set A as the domain of f(2x+1)
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a ∨ x > a + 1}

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (A ∪ B a = Set.univ) → 3 ≤ a ∧ a < 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l713_71319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l713_71365

-- Define the sets M and N
def M : Set ℝ := {x | (x - 2) / (x - 3) < 0}
def N : Set ℝ := {x | Real.log (x - 2) / Real.log (1/2) ≥ 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioc 2 (5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l713_71365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_range_l713_71358

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then (2 : ℝ)^x - 4 else (2 : ℝ)^(-x) - 4

-- State the theorem
theorem even_function_range (h : ∀ x, f x = f (-x)) : 
  {a : ℝ | f (a - 2) > 0} = {a : ℝ | a < 0 ∨ a > 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_range_l713_71358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_range_is_correct_l713_71339

noncomputable section

-- Define the polar coordinates of point A
def A : ℝ × ℝ := (1, Real.pi)

-- Define the curve C in polar coordinates
def C (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the distance between two points in Cartesian coordinates
def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Convert polar coordinates to Cartesian coordinates
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the range of |PA|
def PA_range : Set ℝ := {x | ∃ θ, Real.sqrt 2 - 1 ≤ x ∧ x ≤ Real.sqrt 2 + 1 ∧
  x = distance (polar_to_cartesian (C θ) θ).1
                (polar_to_cartesian (C θ) θ).2
                (polar_to_cartesian A.1 A.2).1
                (polar_to_cartesian A.1 A.2).2}

-- Theorem statement
theorem PA_range_is_correct : PA_range = Set.Icc (Real.sqrt 2 - 1) (Real.sqrt 2 + 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_range_is_correct_l713_71339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_base_angle_is_sixty_degrees_l713_71311

/-- A regular square pyramid with lateral edge length 1 and base edge length 1 -/
structure RegularSquarePyramid where
  lateral_edge : ℝ
  base_edge : ℝ
  lateral_edge_eq_one : lateral_edge = 1
  base_edge_eq_one : base_edge = 1

/-- The angle formed by a lateral edge and the base of a regular square pyramid -/
noncomputable def lateral_base_angle (p : RegularSquarePyramid) : ℝ :=
  Real.arctan (Real.sqrt 3)

theorem lateral_base_angle_is_sixty_degrees (p : RegularSquarePyramid) :
  lateral_base_angle p = π / 3 := by
  sorry

#check lateral_base_angle_is_sixty_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_base_angle_is_sixty_degrees_l713_71311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l713_71331

theorem sin_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi/2)
  (h2 : -Real.pi/2 < β ∧ β < 0)
  (h3 : Real.cos (α - β) = -5/13)
  (h4 : Real.sin α = 4/5) :
  Real.sin β = -56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l713_71331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_decreasing_function_parameter_l713_71300

noncomputable def f (a : ℤ) (x : ℝ) : ℝ := x^(a^2 - 2*a - 3)

theorem even_decreasing_function_parameter (a : ℤ) 
  (h_even : ∀ x : ℝ, f a x = f a (-x))
  (h_decreasing : ∀ x y : ℝ, 0 < x → x < y → f a y < f a x) :
  a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_decreasing_function_parameter_l713_71300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l713_71333

def p (x : ℝ) : ℝ := 5*x^5 - 8*x^4 + 3*x^3 - x^2 + 4*x - 15

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, p x = (x - 2) * q x + 45 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l713_71333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_7_5_l713_71303

def S : Set ℤ := {-30, -5, -1, 2, 9, 15}

theorem largest_quotient_is_7_5 :
  ∀ a b, a ∈ S → b ∈ S → a ≠ 0 → (a : ℚ) / b ≤ 7.5 ∧ ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ 0 ∧ (x : ℚ) / y = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_7_5_l713_71303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l713_71308

noncomputable def f (x : ℝ) := Real.sin x - Real.cos x

noncomputable def g (x : ℝ) := f ((x - Real.pi/3) / 2)

theorem axis_of_symmetry :
  ∀ x : ℝ, g (11*Real.pi/6 + x) = g (11*Real.pi/6 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l713_71308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l713_71327

/-- Given a hyperbola with certain properties, prove its equation and asymptotes -/
theorem hyperbola_properties (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  (2*a = 4) →  -- Real axis length is 4
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    x1^2/a^2 - (x1 - 2)^2/b^2 = 1 ∧ 
    x2^2/a^2 - (x2 - 2)^2/b^2 = 1 ∧ 
    (x1 - x2)^2 + ((x1 - 2) - (x2 - 2))^2 = 800) →  -- Chord length is 20√2
  (∀ x y : ℝ, x^2/4 - y^2/5 = 1) ∧  -- Resulting hyperbola equation
  (∀ x y : ℝ, (y = Real.sqrt 5/2 * x ∨ y = -Real.sqrt 5/2 * x) ↔ (x^2/4 - y^2/5 = 0))  -- Asymptotes equation
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l713_71327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_three_l713_71340

def S : Finset ℕ := Finset.range 15

theorem probability_divisible_by_three :
  let pairs := (S.product S).filter (λ p : ℕ × ℕ => p.1 ≠ p.2)
  (pairs.filter (λ p : ℕ × ℕ => (p.1 * p.2 - p.1 - p.2) % 3 = 0)).card / pairs.card = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_three_l713_71340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_l713_71325

-- Define the left-hand side of the equation
noncomputable def lhs (x : ℝ) : ℝ := (-2 * x^2 + 5 * x - 7) / (x^3 + 2 * x)

-- Define the right-hand side of the equation
noncomputable def rhs (P Q R x : ℝ) : ℝ := P / x + (Q * x + R) / (x^2 + 2)

-- State the theorem
theorem partial_fraction_decomposition :
  ∃ (P Q R : ℝ), ∀ (x : ℝ), x ≠ 0 → x^2 + 2 ≠ 0 → lhs x = rhs P Q R x ∧ P = -2 ∧ Q = 0 ∧ R = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_l713_71325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l713_71312

noncomputable def sequence_a (a : ℝ) : ℕ → ℝ
  | 0 => a  -- Add this case for Nat.zero
  | 1 => a
  | n + 1 => (5 * sequence_a a n - 8) / (sequence_a a n - 1)

noncomputable def sequence_b (a : ℝ) (n : ℕ) : ℝ :=
  (sequence_a a n - 2) / (sequence_a a n - 4)

theorem sequence_properties (a : ℝ) :
  (∀ n : ℕ, n ≥ 1 → sequence_a a n > 3) ↔ a > 3 ∧
  (a = 3 →
    (∀ n : ℕ, n ≥ 1 → sequence_b a (n + 1) = 3 * sequence_b a n) ∧
    (∀ n : ℕ, n ≥ 1 → sequence_a a n = (4 * 3^(n-1) + 2) / (3^(n-1) + 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l713_71312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_deg_matrix_l713_71352

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos θ, -sin θ],
    ![sin θ,  cos θ]]

noncomputable def angle_150_deg : ℝ := 150 * π / 180

theorem rotation_150_deg_matrix :
  rotation_matrix angle_150_deg = ![![-Real.sqrt 3 / 2, -1 / 2],
                                    ![1 / 2, -Real.sqrt 3 / 2]] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_150_deg_matrix_l713_71352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l713_71342

/-- The distance between two parallel lines -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- The first line: 2x + y - 1 = 0 -/
def line1 (x y : ℝ) : Prop := 2*x + y - 1 = 0

/-- The second line: 4x + 2y + 1 = 0 -/
def line2 (x y : ℝ) : Prop := 4*x + 2*y + 1 = 0

theorem distance_between_lines :
  distance_parallel_lines 4 2 (-2) 1 = 3 * Real.sqrt 5 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l713_71342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_pair_l713_71395

-- Define the rectangle and points
noncomputable def rectangle_width : ℝ := 3
noncomputable def rectangle_height : ℝ := 4
def num_points : ℕ := 6

-- Define the maximum distance
noncomputable def max_distance : ℝ := Real.sqrt 5

-- Theorem statement
theorem exists_close_pair :
  ∃ (points : Finset (ℝ × ℝ)) (p q : ℝ × ℝ),
    points.card = num_points ∧
    (∀ p ∈ points, 0 ≤ p.1 ∧ p.1 ≤ rectangle_width ∧ 0 ≤ p.2 ∧ p.2 ≤ rectangle_height) ∧
    p ∈ points ∧ q ∈ points ∧ p ≠ q ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ max_distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_pair_l713_71395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_is_great_circle_l713_71320

-- Define a point on a sphere by its latitude and longitude
structure Point where
  lat : Real
  long : Real

-- Define the Earth as a sphere
def Earth : Type := Unit

-- Define a path on the Earth
def EarthPath := Point → Point → Type

-- Define the great circle path
def GreatCirclePath : EarthPath := sorry

-- Define the latitude path
def LatitudePath : EarthPath := sorry

-- Define the path through a pole
def PolePath : EarthPath := sorry

-- Define the length of a path
noncomputable def PathLength (p : EarthPath) (a b : Point) : Real := sorry

-- Theorem statement
theorem shortest_path_is_great_circle (a b : Point) 
  (h1 : a.lat = b.lat) 
  (h2 : a.lat ≠ 90 ∧ a.lat ≠ -90) 
  (h3 : a.long ≠ b.long) : 
  PathLength GreatCirclePath a b < min (PathLength LatitudePath a b) (PathLength PolePath a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_is_great_circle_l713_71320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_opposite_eight_l713_71396

theorem cube_root_opposite_eight : Real.rpow (-8) (1/3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_opposite_eight_l713_71396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_pentagon_eq_360_l713_71328

/-- The sum of the exterior angles of a pentagon is 360 degrees. -/
def sum_exterior_angles_pentagon : ℝ := 360

/-- A pentagon is a polygon with 5 sides. -/
structure Pentagon where
  sides : Fin 5 → ℝ
  sides_positive : ∀ i, sides i > 0

/-- The exterior angles of a pentagon. -/
def exterior_angles (p : Pentagon) : Fin 5 → ℝ :=
  sorry

/-- The sum of the exterior angles of a pentagon is 360 degrees. -/
theorem sum_exterior_angles_pentagon_eq_360 (p : Pentagon) :
    (Finset.univ.sum (exterior_angles p)) = sum_exterior_angles_pentagon := by
  sorry

#check sum_exterior_angles_pentagon_eq_360

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_pentagon_eq_360_l713_71328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_l713_71371

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem cylinder_radius (z : ℝ) :
  let h : ℝ := 3
  let volumeIncreaseRadius (r : ℝ) := cylinderVolume (r + 8) h - cylinderVolume r h
  let volumeIncreaseHeight (r : ℝ) := cylinderVolume r (h + 8) - cylinderVolume r h
  ∃ r : ℝ, r > 0 ∧ volumeIncreaseRadius r = z ∧ volumeIncreaseHeight r = z ∧ r = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_radius_l713_71371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_with_remainder_l713_71317

theorem unique_divisor_with_remainder :
  ∃! d : ℕ, d > 0 ∧ (55 * 57) % d = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_with_remainder_l713_71317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_theorem_l713_71323

/-- The combined resistance of two parallel resistors -/
noncomputable def combined_resistance (x : ℝ) : ℝ := 1 / (1 / x + 1 / 5)

theorem parallel_resistors_theorem (x : ℝ) (hx : x > 0) :
  combined_resistance x = 2.2222222222222223 → x = 4 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_theorem_l713_71323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_negative_half_l713_71394

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin (Real.pi * x / 6) else 1 - 2 * x

theorem f_composition_equals_negative_half : f (f 3) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_negative_half_l713_71394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l713_71389

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 8 then (4 - a) * x - 5 else a^(x - 8)

noncomputable def sequence_a (a : ℝ) (n : ℕ+) : ℝ := f a n

theorem a_range (a : ℝ) :
  (∀ n m : ℕ+, n < m → sequence_a a n < sequence_a a m) →
  3 < a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l713_71389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l713_71368

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem sine_function_properties (ω φ : ℝ) 
  (h1 : 0 < ω) (h2 : ω < 3) (h3 : 0 < φ) (h4 : φ < Real.pi)
  (h5 : f ω φ (-Real.pi/4) = 0)
  (h6 : ∀ (x : ℝ), f ω φ (Real.pi/3 + x) = f ω φ (Real.pi/3 - x)) :
  ω = 6/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l713_71368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l713_71309

theorem starting_number_proof (n : ℕ) : 
  (n ≤ 300 ∧ 
   (∃ (m : ℕ), m = 67 ∧ 
    (∀ k : ℕ, n ≤ k ∧ k ≤ 300 ∧ k % 3 = 0 → k ∈ Finset.range (m + 1))) ∧
   (∀ n' : ℕ, n < n' ∧ n' ≤ 300 → 
    ¬(∃ (m : ℕ), m = 67 ∧ 
      (∀ k : ℕ, n' ≤ k ∧ k ≤ 300 ∧ k % 3 = 0 → k ∈ Finset.range (m + 1))))) →
  n = 102 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l713_71309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumference_division_l713_71362

theorem circumference_division (n : ℕ) (d : ℕ) :
  n > 0 ∧ d > 0 ∧ 
  (4 ≡ 11 + d [ZMOD n]) ∧
  (17 ≡ 4 + d [ZMOD n]) →
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumference_division_l713_71362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_differences_l713_71366

def x (n : ℕ) : ℕ := n^2

def first_difference (n : ℕ) : ℕ := x (n + 1) - x n

def second_difference (n : ℕ) : ℕ := first_difference (n + 1) - first_difference n

def kth_difference : ℕ → (ℕ → ℕ)
  | 0 => x
  | 1 => first_difference
  | 2 => second_difference
  | (k + 3) => λ n => kth_difference (k + 2) (n + 1) - kth_difference (k + 2) n

theorem sequence_differences :
  (∀ n, second_difference n = 2) ∧
  (∀ k, k ≥ 3 → ∀ n, kth_difference k n = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_differences_l713_71366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_a_plus_b_l713_71378

/-- A function with specific properties -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ) + 2

/-- The theorem statement -/
theorem min_abs_a_plus_b :
  ∀ (ω φ a b : ℝ),
    ω > 0 →
    |φ| < π / 2 →
    (∀ x, f ω φ (x + 2) = f ω φ x) →
    f ω φ 0 = 3 →
    (∃ x, f ω φ (x + a) = f ω φ (a - x) ∧ f ω φ a = b) →
    (∀ a' b', (∃ x, f ω φ (x + a') = f ω φ (a' - x) ∧ f ω φ a' = b') → |a + b| ≤ |a' + b'|) →
    |a + b| = 1 / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_a_plus_b_l713_71378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_score_percentage_l713_71318

/-- Calculates the percentage of total score made by running between wickets and from no-balls in cricket -/
theorem cricket_score_percentage (total_runs boundaries sixes no_balls : ℕ) :
  total_runs = 180 →
  boundaries = 9 →
  sixes = 7 →
  no_balls = 2 →
  let runs_from_boundaries := boundaries * 4
  let runs_from_sixes := sixes * 6
  let runs_from_no_balls := no_balls
  let runs_from_running := total_runs - (runs_from_boundaries + runs_from_sixes + runs_from_no_balls)
  let percentage := (runs_from_running + runs_from_no_balls : ℚ) / total_runs * 100
  abs (percentage - 56.67) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_score_percentage_l713_71318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l713_71306

noncomputable def g (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 4 * Real.sin x ^ 2 + 3 * Real.sin x + 2 * Real.cos x ^ 2 - 6) / (Real.sin x - 1)

theorem g_range : 
  Set.range g = Set.Icc 1 9 \ {9} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l713_71306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_zeros_l713_71316

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - 1

-- State the theorem
theorem tangent_line_and_zeros (a : ℝ) :
  (∃ m n : ℝ, (deriv (f a)) m = -1 ∧ f a m + 1 = -m + 3) →
  (∃ x₁ x₂ : ℝ, Real.exp (-1) ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.exp 1 ∧ g a x₁ = 0 ∧ g a x₂ = 0) →
  (∀ x > 2, (deriv (f a)) x > 0) ∧
  (∀ x ∈ Set.Ioo 0 2, (deriv (f a)) x < 0) ∧
  2 / Real.exp 1 ≤ a ∧ a < 1 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_zeros_l713_71316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_group_exists_l713_71355

/-- Represents a homework collaboration between three students -/
structure Collaboration where
  students : Finset (Fin 21)
  subject : Bool  -- true for mathematics, false for Russian
  hw_done : students.card = 3

/-- The set of all collaborations in the class -/
def class_collaborations : Set Collaboration :=
  { c | c.students ⊆ Finset.univ }

/-- Any three students have done homework together exactly once -/
axiom collaboration_exists (s : Finset (Fin 21)) :
  s.card = 3 → ∃! c, c ∈ class_collaborations ∧ c.students = s

/-- The main theorem: there exists a group of four students where any three have done homework in the same subject -/
theorem homework_group_exists :
  ∃ g : Finset (Fin 21), g.card = 4 ∧
    (∀ s ⊆ g, s.card = 3 →
      ∃ c ∈ class_collaborations, c.students = s ∧
        (∀ c' ∈ class_collaborations, c'.students ⊆ g → c'.subject = c.subject)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_group_exists_l713_71355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_distinct_numbers_l713_71357

theorem exist_distinct_numbers : ∃ (m n p q : ℕ), 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q ∧
  m + n = p + q ∧
  (Real.sqrt (m : ℝ) + (n : ℝ) ^ (1/3 : ℝ) = Real.sqrt (p : ℝ) + (q : ℝ) ^ (1/3 : ℝ)) ∧
  Real.sqrt (m : ℝ) + (n : ℝ) ^ (1/3 : ℝ) > 2004 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_distinct_numbers_l713_71357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_time_C_is_four_l713_71391

/-- Represents the loan details and interest calculations -/
structure LoanDetails where
  principal_B : ℚ  -- Principal amount lent to B
  principal_C : ℚ  -- Principal amount lent to C
  time_B : ℚ       -- Time (in years) for B's loan
  rate : ℚ         -- Annual interest rate
  total_interest : ℚ  -- Total interest received from both B and C

/-- Calculates the number of years A lent money to C -/
def calculate_time_C (loan : LoanDetails) : ℚ :=
  let interest_B := loan.principal_B * loan.rate * loan.time_B
  let interest_C := loan.total_interest - interest_B
  interest_C / (loan.principal_C * loan.rate)

/-- Theorem stating that the time A lent money to C is 4 years -/
theorem loan_time_C_is_four :
  let loan := LoanDetails.mk 5000 3000 2 (15/100) 3300
  calculate_time_C loan = 4 := by
  -- Proof goes here
  sorry

#eval calculate_time_C (LoanDetails.mk 5000 3000 2 (15/100) 3300)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_time_C_is_four_l713_71391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_field_solutions_l713_71377

def field_area (S : ℝ) (a b : ℤ) (x y : ℝ) : Prop :=
  10 * 300 * S ≤ 10000 ∧
  S > 0 ∧
  2 * a = -b ∧
  |6 * |y| + 3| ≥ 3 ∧
  2 * |2 * |x| - (b : ℝ)| ≤ 9 ∧
  -4.5 ≤ (b : ℝ) ∧ (b : ℝ) ≤ 4.5

theorem valid_field_solutions :
  ∀ S : ℝ, ∀ a b : ℤ, ∀ x y : ℝ,
    field_area S a b x y →
    ((b = -4 ∧ a = 2) ∨ (b = -2 ∧ a = 1)) :=
by
  sorry

#check valid_field_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_field_solutions_l713_71377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_savings_adjustment_l713_71382

/-- Calculates the new required hours per week given the original plan and missed weeks. -/
noncomputable def new_hours_per_week (original_weeks : ℕ) (original_hours : ℝ) (missed_weeks : ℕ) : ℝ :=
  let remaining_weeks := original_weeks - missed_weeks
  original_hours * (original_weeks : ℝ) / (remaining_weeks : ℝ)

/-- The theorem stating the correct calculation of new required hours per week. -/
theorem vacation_savings_adjustment (original_weeks : ℕ) (original_hours : ℝ) (missed_weeks : ℕ) 
    (h1 : original_weeks = 15)
    (h2 : original_hours = 25)
    (h3 : missed_weeks = 3) :
  new_hours_per_week original_weeks original_hours missed_weeks = 31.25 :=
by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues
-- #eval new_hours_per_week 15 25 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_savings_adjustment_l713_71382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_possible_areas_l713_71335

-- Define the points on the first line
noncomputable def G : ℝ := 0
noncomputable def H : ℝ := 1
noncomputable def I : ℝ := 2
noncomputable def J : ℝ := 4
noncomputable def K : ℝ := 7

-- Define the points on the second line
noncomputable def L : ℝ := 0
noncomputable def M : ℝ := 3

-- Define the points on the third line
noncomputable def N : ℝ := 0
noncomputable def O : ℝ := 2

-- Define the distance between parallel lines
noncomputable def d : ℝ := sorry

-- Define the set of possible base lengths
def possible_bases : Set ℝ := {1, 2, 3}

-- Define the function to calculate triangle area
noncomputable def triangle_area (base : ℝ) : ℝ := (1/2) * base * d

-- Theorem stating that there are exactly 3 possible triangle areas
theorem three_possible_areas : 
  ∃! (areas : Finset ℝ), 
    (∀ a ∈ areas, ∃ b ∈ possible_bases, a = triangle_area b) ∧ 
    (Finset.card areas = 3) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_possible_areas_l713_71335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_is_correct_l713_71346

/-- The radius of the largest inscribed circle in a square with side length 2,
    where a quarter-circle of radius 1 is cut out from one corner. -/
noncomputable def largest_inscribed_circle_radius : ℝ := 5 - 3 * Real.sqrt 2

/-- Given a square with side length 2 and a quarter-circle of radius 1 cut out from one corner,
    the radius of the largest inscribed circle in the remaining figure is 5 - 3√2. -/
theorem largest_inscribed_circle_radius_is_correct (square_side : ℝ) (quarter_circle_radius : ℝ)
  (h1 : square_side = 2)
  (h2 : quarter_circle_radius = 1) :
  largest_inscribed_circle_radius = 5 - 3 * Real.sqrt 2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_radius_is_correct_l713_71346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_225_degrees_in_second_quadrant_l713_71379

noncomputable def angle_to_quadrant (angle : ℝ) : ℕ :=
  match Int.floor (angle % 360) % 4 with
  | 0 => 1
  | 1 => 4
  | 2 => 3
  | _ => 2

theorem negative_225_degrees_in_second_quadrant :
  angle_to_quadrant (-225) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_225_degrees_in_second_quadrant_l713_71379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_sixth_numbers_l713_71324

def mySequence : ℕ → ℚ
  | 0 => 1
  | n + 1 => ((n + 3 : ℚ) / (n + 2 : ℚ)) ^ 3

theorem sum_of_fourth_and_sixth_numbers :
  mySequence 3 + mySequence 5 = 6119 / 1728 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_sixth_numbers_l713_71324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_inequality_l713_71322

noncomputable def F (x y : ℝ) : ℝ := (1 + x) ^ y

theorem F_inequality {x y : ℕ} (hx : x > 0) (hy : y > 0) (hxy : x < y) :
  F (x : ℝ) (y : ℝ) > F (y : ℝ) (x : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_inequality_l713_71322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l713_71376

-- Define the cost function C(x)
noncomputable def C (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

-- Define the profit function L(x)
noncomputable def L (x : ℝ) : ℝ :=
  50 * x - C x - 250

-- Theorem statement
theorem max_profit :
  ∃ (x : ℝ), x > 0 ∧ L x = 1000 ∧ ∀ (y : ℝ), y > 0 → L y ≤ L x := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l713_71376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_remaining_l713_71373

theorem paint_remaining (initial_paint : ℚ) : initial_paint > 0 → 
  (initial_paint - (1/4 * initial_paint) - 
   (1/2 * (initial_paint - (1/4 * initial_paint))) - 
   (1/3 * (initial_paint - (1/4 * initial_paint) - 
           (1/2 * (initial_paint - (1/4 * initial_paint)))))) = 
  (1/4) * initial_paint := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_remaining_l713_71373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_length_l713_71313

-- Define the circle and its properties
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Define points A, B, C, D on the circle
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
def C : ℝ × ℝ := (-1, 0)
def D : ℝ × ℝ := (0, -1)

-- Define the theorem
theorem circle_intersection_length (d : ℝ) (h : 0 ≤ d ∧ d ≤ 1) :
  ∃ (M E : ℝ × ℝ),
    M ∈ Circle ∧
    E ∈ Circle ∧
    (M.1 - C.1)^2 + (M.2 - C.2)^2 = (E.1 - M.1)^2 + (E.2 - M.2)^2 ∧
    (E.1 - M.1)^2 + (E.2 - M.2)^2 = d^2 ∧
    (A.1 - M.1)^2 + (A.2 - M.2)^2 = ((- d + Real.sqrt (d^2 + 8)) / 2)^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_length_l713_71313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l713_71310

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sin (5 * x) + Real.sin (7 * x)) / (Real.sin (4 * x) + Real.sin (2 * x)) = -4 * |Real.sin (2 * x)| ↔ 
  (∃ k : ℤ, x = Real.pi - Real.arcsin ((1 - Real.sqrt 2) / 2) + 2 * k * Real.pi) ∨
  (∃ k : ℤ, x = Real.pi - Real.arcsin ((Real.sqrt 2 - 1) / 2) + 2 * k * Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l713_71310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_is_two_l713_71385

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = -x^2 + 4 -/
def on_parabola (p : Point) : Prop :=
  p.y = -p.x^2 + 4

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Predicate to check if four points form a square -/
def is_square (a b c d : Point) : Prop :=
  distance a b = distance b c ∧
  distance b c = distance c d ∧
  distance c d = distance d a ∧
  distance a c = distance b d

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- Theorem stating that if P and Q are on the parabola y = -x^2 + 4,
    and OPQR forms a square with O at the origin,
    then the side length of OPQR is 2 -/
theorem square_side_length_is_two 
  (P Q R : Point)
  (h1 : on_parabola P)
  (h2 : on_parabola Q)
  (h3 : is_square origin P Q R) :
  distance origin P = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_is_two_l713_71385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merry_go_round_revolutions_l713_71390

/-- The number of revolutions needed for a horse on a merry-go-round to travel a given distance -/
noncomputable def revolutions_needed (distance_from_center : ℝ) (target_distance : ℝ) : ℝ :=
  target_distance / (2 * Real.pi * distance_from_center)

theorem merry_go_round_revolutions 
  (horse1_distance : ℝ) 
  (horse1_revolutions : ℝ) 
  (horse2_distance : ℝ) :
  horse1_distance = 30 →
  horse1_revolutions = 40 →
  horse2_distance = 10 →
  revolutions_needed horse2_distance (horse1_distance * horse1_revolutions * 2 * Real.pi) = 120 := by
  sorry

#check merry_go_round_revolutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merry_go_round_revolutions_l713_71390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_properties_l713_71341

/-- Represents a spring with given properties -/
structure Spring where
  initial_length : ℚ
  stretch_rate : ℚ
  max_length : ℚ

/-- Calculates the length of the spring given a mass -/
def spring_length (s : Spring) (mass : ℚ) : ℚ :=
  s.initial_length + s.stretch_rate * mass

/-- Calculates the maximum mass a spring can support -/
def max_mass (s : Spring) : ℚ :=
  (s.max_length - s.initial_length) / s.stretch_rate

/-- Main theorem about the spring properties -/
theorem spring_properties (s : Spring) 
    (h1 : s.initial_length = 12)
    (h2 : s.stretch_rate = 1/2)
    (h3 : s.max_length = 20) : 
  spring_length s 0 = 12 ∧ 
  spring_length s 5 = 29/2 ∧ 
  (∀ x, spring_length s x = 1/2 * x + 12) ∧
  max_mass s = 16 := by
  sorry

#eval spring_length { initial_length := 12, stretch_rate := 1/2, max_length := 20 } 5
#eval max_mass { initial_length := 12, stretch_rate := 1/2, max_length := 20 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_properties_l713_71341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_distribution_l713_71314

theorem library_book_distribution (n : ℕ) (h : n = 8) :
  let valid_distributions := Finset.range (n - 3) |>.filter (λ k ↦ 2 ≤ k + 2)
  Finset.card valid_distributions = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_distribution_l713_71314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_max_k_l713_71348

-- Define the functions
noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (a x : ℝ) := (a * x^2) / 2

-- Theorem 1: Tangent line equation
theorem tangent_line_equation :
  ∃ (y : ℝ → ℝ), (∀ x, y x = 2*x - Real.exp 1 - y (Real.exp 1)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - Real.exp 1| < δ → |f x - y x| < ε * |x - Real.exp 1|) :=
by sorry

-- Theorem 2: Range of a
theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f x₀ < g a x₀) → a > 0 :=
by sorry

-- Theorem 3: Maximum value of k
theorem max_k :
  (∀ x > 1, f x > (5 - 3) * x - 5 + 2) ∧
  ¬(∃ k : ℤ, k > 5 ∧ ∀ x > 1, f x > (k - 3) * x - k + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_max_k_l713_71348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cups_per_batch_is_two_verify_conditions_l713_71344

/-- The number of cups of flour required for one batch of cookies. -/
def cups_per_batch : ℝ := 2

/-- The number of batches Gigi has already baked. -/
def baked_batches : ℕ := 3

/-- The total amount of flour in Gigi's bag. -/
def total_flour : ℝ := 20

/-- The number of additional batches Gigi could make with the remaining flour. -/
def future_batches : ℕ := 7

/-- Theorem stating that the number of cups of flour required for one batch of cookies is 2. -/
theorem cups_per_batch_is_two :
  cups_per_batch = 2 :=
by
  -- Unfold the definition of cups_per_batch
  unfold cups_per_batch
  -- The equality is now trivial
  rfl

/-- Theorem verifying that the given conditions are consistent with the solution. -/
theorem verify_conditions :
  cups_per_batch * (baked_batches : ℝ) + cups_per_batch * (future_batches : ℝ) = total_flour :=
by
  -- Substitute the known values
  simp [cups_per_batch, baked_batches, future_batches, total_flour]
  -- Evaluate the expression
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cups_per_batch_is_two_verify_conditions_l713_71344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_most_axes_l713_71307

-- Define the shapes
inductive Shape
  | EquilateralTriangle
  | Circle
  | Rectangle
  | Square

-- Define a function to count axes of symmetry
def axesOfSymmetry (s : Shape) : Nat ⊕ Unit :=
  match s with
  | Shape.EquilateralTriangle => Sum.inl 3
  | Shape.Circle => Sum.inr ()
  | Shape.Rectangle => Sum.inl 2
  | Shape.Square => Sum.inl 4

-- Define a function to compare number of axes
def hasMoreAxes (a b : Nat ⊕ Unit) : Prop :=
  match a, b with
  | Sum.inl n, Sum.inl m => n > m
  | Sum.inr (), Sum.inl _ => True
  | Sum.inl _, Sum.inr () => False
  | Sum.inr (), Sum.inr () => False

-- Theorem statement
theorem circle_has_most_axes :
  ∀ s : Shape, s ≠ Shape.Circle → hasMoreAxes (axesOfSymmetry Shape.Circle) (axesOfSymmetry s) := by
  intro s h
  cases s
  all_goals (
    simp [axesOfSymmetry, hasMoreAxes]
    try rfl
  )
  contradiction


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_most_axes_l713_71307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l713_71353

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x - φ)

def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

theorem min_shift_for_even_function :
  ∃ φ : ℝ, φ > 0 ∧ is_even (g φ) ∧ ∀ ψ : ℝ, ψ > 0 ∧ is_even (g ψ) → φ ≤ ψ :=
by sorry

#check min_shift_for_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l713_71353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l713_71329

-- Define the clock properties
noncomputable def hours_in_clock : ℕ := 12
noncomputable def degrees_in_circle : ℝ := 360
noncomputable def degrees_per_hour : ℝ := degrees_in_circle / hours_in_clock
noncomputable def degrees_per_minute_for_minute_hand : ℝ := 6
noncomputable def degrees_per_minute_for_hour_hand : ℝ := 0.5

-- Define the specific time
def hours : ℕ := 3
def minutes : ℕ := 15

-- Calculate positions of hands
noncomputable def minute_hand_position : ℝ := minutes * degrees_per_minute_for_minute_hand
noncomputable def hour_hand_position : ℝ := hours * degrees_per_hour + minutes * degrees_per_minute_for_hour_hand

-- Define the theorem
theorem clock_angle_at_3_15 :
  min (|hour_hand_position - minute_hand_position|)
      (degrees_in_circle - |hour_hand_position - minute_hand_position|) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_15_l713_71329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meeting_theorem_l713_71343

/-- Represents the meeting of two cyclists -/
structure CyclistsMeeting where
  speed_a : ℝ  -- Speed of cyclist A in km/h
  speed_b : ℝ  -- Speed of cyclist B in km/h
  midpoint_distance : ℝ  -- Distance from meeting point to midpoint in km

/-- Calculates the meeting time of the cyclists -/
noncomputable def meeting_time (m : CyclistsMeeting) : ℝ :=
  (2 * m.midpoint_distance) / (m.speed_a - m.speed_b)

/-- Calculates the total distance of the journey -/
noncomputable def total_distance (m : CyclistsMeeting) : ℝ :=
  (m.speed_a + m.speed_b) * meeting_time m

/-- Theorem stating the meeting time and total distance for the given scenario -/
theorem cyclists_meeting_theorem (m : CyclistsMeeting) 
  (h1 : m.speed_a = 20)
  (h2 : m.speed_b = 18)
  (h3 : m.midpoint_distance = 8) :
  meeting_time m = 8 ∧ total_distance m = 304 := by
  sorry

-- Remove the #eval statements as they are not computable
-- #eval meeting_time ⟨20, 18, 8⟩
-- #eval total_distance ⟨20, 18, 8⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meeting_theorem_l713_71343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l713_71345

def birthday (n : ℕ) : Fin n → Fin 366 := sorry

theorem birthday_problem (n : ℕ) (h : n = 367) :
  ∃ (i j : Fin n), i ≠ j ∧ birthday n i = birthday n j :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l713_71345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l713_71315

noncomputable def larger_circle_area : ℝ := 100 * Real.pi

noncomputable def circle_A_radius (R : ℝ) : ℝ := R / 2

noncomputable def circle_B_radius (R : ℝ) : ℝ := R / 4

noncomputable def shaded_area (R : ℝ) : ℝ :=
  (larger_circle_area / 2) - 
  (Real.pi * (circle_A_radius R)^2 / 2) - 
  (Real.pi * (circle_B_radius R)^2 / 2)

theorem total_shaded_area :
  ∃ R : ℝ, R > 0 ∧ larger_circle_area = Real.pi * R^2 ∧ shaded_area R = 34.375 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l713_71315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_sets_count_l713_71372

/-- The set A -/
def A : Finset Int := {-1, 0, 1, 2, 3}

/-- The set B -/
def B : Finset Int := {-1, 0, 1}

/-- A function from A to B -/
def f : A → B := sorry

/-- The number of possible value sets for f -/
def num_value_sets : Nat := (Finset.powerset B).filter (· ≠ ∅) |>.card

theorem value_sets_count : num_value_sets = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_sets_count_l713_71372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_percentage_l713_71304

/-- Calculates the percentage of remaining income spent on house rent -/
noncomputable def percentage_on_house_rent (total_income : ℝ) (petrol_expense : ℝ) (house_rent : ℝ) : ℝ :=
  let remaining_income := total_income - petrol_expense
  (house_rent / remaining_income) * 100

/-- Theorem: Given the conditions, the percentage spent on house rent is 20% -/
theorem house_rent_percentage :
  ∀ (total_income : ℝ),
  total_income > 0 →
  total_income * 0.3 = 300 →
  percentage_on_house_rent total_income 300 140 = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_percentage_l713_71304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l713_71397

/-- The function we're maximizing -/
noncomputable def f (t : ℝ) : ℝ := (2^t - 5*t) * t / 4^t

/-- The theorem stating the maximum value of the function -/
theorem f_max_value :
  (∀ t : ℝ, f t ≤ 1/20) ∧ (∃ t : ℝ, f t = 1/20) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l713_71397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_k_ge_one_l713_71356

/-- The function f(x) = kx - ln(x) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x - Real.log x

/-- f is monotonically increasing on (1, +∞) -/
def is_monotone_increasing (k : ℝ) : Prop :=
  ∀ x y, 1 < x → x < y → f k x < f k y

/-- Theorem: If f(x) = kx - ln(x) is monotonically increasing on (1, +∞), then k ≥ 1 -/
theorem monotone_increasing_implies_k_ge_one (k : ℝ) :
  is_monotone_increasing k → k ≥ 1 := by
  sorry

#check monotone_increasing_implies_k_ge_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_k_ge_one_l713_71356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l713_71370

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {5}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {5} := by
  -- We use U \ M instead of Set.compl M to ensure we're working within the same set type
  sorry

#check complement_M_intersect_N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l713_71370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_number_l713_71338

def next_term (n : ℕ) : ℕ :=
  if n < 10 then n + 8 else (n % 10) + 7

def sequence_term (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => next_term (sequence_term n)

theorem last_student_number : sequence_term 2013 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_number_l713_71338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_proof_l713_71332

theorem arc_length_proof (r : ℝ) (h : 2 * Real.sin 1 * r = 2) : 2 * r = 2 / Real.sin 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_proof_l713_71332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_beta_half_l713_71330

theorem cos_alpha_plus_beta_half 
  (α β : ℝ) 
  (h1 : Real.cos (α - β/2) = -1/3)
  (h2 : Real.sin (α/2 - β) = 1/4)
  (h3 : 3*π/2 < α ∧ α < 2*π)
  (h4 : π/2 < β ∧ β < π) :
  Real.cos ((α + β)/2) = -(2*Real.sqrt 2 + Real.sqrt 15)/12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_beta_half_l713_71330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_bounds_l713_71364

/-- The quadratic function f(x) = ax^2 + 4ax - 1 -/
def f (a x : ℝ) : ℝ := a * x^2 + 4 * a * x - 1

/-- The set of a values for which |f(x)| ≤ 4 for all x in [-4, 0] -/
def A : Set ℝ := {a | ∀ x ∈ Set.Icc (-4) 0, |f a x| ≤ 4}

theorem quadratic_function_bounds :
  A = Set.Icc (-5/4) 0 ∪ Set.Ioc 0 (3/4) ∧
  MeasureTheory.volume A = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_bounds_l713_71364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_free_sets_l713_71336

theorem union_free_sets (n : ℕ) (h : n ≥ 5) :
  ∃ (r : ℕ) (S : Finset (Finset ℕ)),
    r = ⌊Real.sqrt (2 * n)⌋ ∧
    S.card = r ∧
    (∀ A ∈ S, ∀ B ∈ S, ∀ C ∈ S, A ≠ B ∪ C) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_free_sets_l713_71336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_fail_percentage_l713_71398

theorem exam_fail_percentage
  (total_candidates : ℕ)
  (num_girls : ℕ)
  (pass_rate : ℚ)
  (h_total : total_candidates = 2000)
  (h_girls : num_girls = 900)
  (h_pass_rate : pass_rate = 32 / 100) :
  let num_boys := total_candidates - num_girls
  let boys_passed := (pass_rate * num_boys : ℚ).floor
  let girls_passed := (pass_rate * num_girls : ℚ).floor
  let total_passed := boys_passed + girls_passed
  let total_failed := total_candidates - total_passed
  (total_failed : ℚ) / total_candidates = 68 / 100 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_fail_percentage_l713_71398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_true_l713_71388

-- Define the options as propositions
def monitor_three_gorges : Prop := True
def monitor_environment : Prop := True
def track_tibetan_antelopes : Prop := True
def address_ecological_issues : Prop := True

-- Define the correct answer
def correct_answer : Prop := monitor_three_gorges ∧ monitor_environment ∧ track_tibetan_antelopes

-- Theorem stating that the correct answer is true
theorem correct_answer_is_true : correct_answer :=
  by
    -- We'll use 'sorry' here as we don't have a formal proof
    sorry

#check correct_answer_is_true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_true_l713_71388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l713_71351

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h0 : a > b
  h1 : b > 0
  h2 : a^2 * (1/4) + b^2 * (1/3) = 1
  eccentricity : ℝ
  h3 : eccentricity = 1/2

/-- A point on the ellipse with specific properties -/
structure PointOnEllipse (ε : SpecialEllipse) where
  P : ℝ × ℝ
  h4 : (P.1^2 / ε.a^2) + (P.2^2 / ε.b^2) = 1
  h5 : P.2 = 0  -- PF₂ is perpendicular to x-axis
  h6 : (P.1 - ε.a)^2 + P.2^2 = (3/2)^2  -- |PF₂| = 3/2

/-- A point M on the positive y-axis -/
noncomputable def M : ℝ × ℝ := (0, Real.sqrt 21 / 7)

/-- Main theorem -/
theorem special_ellipse_properties (ε : SpecialEllipse) (P : PointOnEllipse ε) :
  ε.a = 2 ∧ ε.b = Real.sqrt 3 ∧ M.2 = Real.sqrt 21 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l713_71351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_complex_z_is_pure_imaginary_z_in_second_quadrant_l713_71384

/-- Definition of the complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2 : ℝ) + (m^2 - 3 * m + 2 : ℝ) * Complex.I

/-- z is a real number if and only if m = 1 or m = 2 -/
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = 1 ∨ m = 2 := by sorry

/-- z is a complex number if and only if m ≠ 1 and m ≠ 2 -/
theorem z_is_complex (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ 1 ∧ m ≠ 2 := by sorry

/-- z is a pure imaginary number if and only if m = -1/2 -/
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ↔ m = -1/2 := by sorry

/-- z is in the second quadrant of the complex plane if and only if -1/2 < m < 1 -/
theorem z_in_second_quadrant (m : ℝ) : 
  (z m).re < 0 ∧ (z m).im > 0 ↔ -1/2 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_complex_z_is_pure_imaginary_z_in_second_quadrant_l713_71384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l713_71369

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + c

-- Define the function g
def g (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

-- Define the function φ
def phi (c : ℝ) (lambda : ℝ) (x : ℝ) : ℝ := g c x - lambda * f c x

-- Theorem statement
theorem function_properties (c : ℝ) :
  (∀ x, f c (f c x) = f c (x^2 + 1)) →
  (∀ x, g c x = x^4 + 2*x^2 + 2) ∧
  ∃! lambda, (∀ x ∈ Set.Iic (-1), ∀ δ > 0, phi c lambda (x + δ) ≤ phi c lambda x) ∧
             (∀ x ∈ Set.Ioc (-1) 0, ∀ δ > 0, phi c lambda (x + δ) ≥ phi c lambda x) ∧
             lambda = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l713_71369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_at_distance_two_l713_71347

/-- The distance between two parallel lines ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- Checks if a line ax + by + c = 0 is at distance d from the line x - y - 1 = 0 -/
def is_line_at_distance (a b c d : ℝ) : Prop :=
  a = 1 ∧ b = -1 ∧ distance_between_parallel_lines 1 (-1) (-1) c = d

theorem line_at_distance_two :
  ∀ (a b c : ℝ),
    is_line_at_distance a b c 2 ↔ 
      (a = 1 ∧ b = -1 ∧ c = 2 * Real.sqrt 2 - 1) ∨
      (a = 1 ∧ b = -1 ∧ c = -2 * Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_at_distance_two_l713_71347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l713_71386

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ -1 then a * x^2 + 2*x else (1 - 3*a) * x - 3/2

theorem f_increasing_iff_a_in_range (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 
  0 ≤ a ∧ a ≤ 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l713_71386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_propositions_l713_71305

-- Define what a proposition is
def is_proposition (s : String) : Bool :=
  match s with
  | "|x+2|" => false
  | "-5 ∈ ℤ" => true
  | "π ∉ ℝ" => true
  | "{0} ∈ ℕ" => true
  | _ => false

-- Define our list of statements
def statements : List String := ["|x+2|", "-5 ∈ ℤ", "π ∉ ℝ", "{0} ∈ ℕ"]

-- The theorem to prove
theorem count_propositions :
  (statements.filter is_proposition).length = 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_propositions_l713_71305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_point_area_of_right_triangle_l713_71350

/-- A fold point of a triangle is a point where the creases formed by folding the vertices onto it do not intersect inside the triangle. -/
def FoldPoint (A B C P : ℝ × ℝ) : Prop := sorry

/-- The set of all fold points of a triangle ABC. -/
def FoldPointSet (A B C : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ². -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem fold_point_area_of_right_triangle :
  ∀ A B C : ℝ × ℝ,
  let d_AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let d_BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let angle_A := Real.arccos (((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / (d_AB * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)))
  d_AB = 45 →
  d_BC = 60 →
  angle_A = Real.pi / 2 →
  area (FoldPointSet A B C) = 703.125 * Real.pi - 450 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_point_area_of_right_triangle_l713_71350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_range_l713_71360

-- Define the circle C
def is_on_circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (0, 1)

-- Define the distance function
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the sum of squared distances
def d (P : ℝ × ℝ) : ℝ :=
  distance_squared P A + distance_squared P B

-- Theorem statement
theorem d_range :
  ∀ P : ℝ × ℝ, is_on_circle P.1 P.2 → 32 ≤ d P ∧ d P ≤ 72 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_range_l713_71360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_h_existence_condition_l713_71321

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

def g (a : ℝ) (x : ℝ) : ℝ := -(1 + a) / x

def h (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

theorem monotonicity_of_h (a : ℝ) (h_a : a > 0) :
  (∀ x ∈ Set.Ioo 0 (1 + a), StrictMonoOn (fun x => -h a x) (Set.Ioo 0 (1 + a))) ∧
  (∀ x ∈ Set.Ioi (1 + a), StrictMonoOn (h a) (Set.Ioi (1 + a))) := by sorry

theorem existence_condition (a : ℝ) :
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x < g a x) ↔ 
  (a > (Real.exp 2 + 1) / (Real.exp 1 - 1) ∨ a < -2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_h_existence_condition_l713_71321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_216_l713_71367

/-- Represents a rectangle with given perimeter and aspect ratio -/
structure Rectangle where
  perimeter : ℝ
  aspectRatio : ℝ
  perimeterConstraint : perimeter = 60
  aspectRatioConstraint : aspectRatio = 3/2

/-- Calculates the area of a rectangle given its perimeter and aspect ratio -/
noncomputable def calculateArea (r : Rectangle) : ℝ :=
  let width := r.perimeter / (2 * (1 + r.aspectRatio))
  let length := r.aspectRatio * width
  length * width

/-- Theorem stating that the area of the rectangle with given constraints is 216 -/
theorem rectangle_area_is_216 (r : Rectangle) : calculateArea r = 216 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_216_l713_71367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l713_71383

theorem relationship_abc : 
  let a := (2 : ℝ) ^ (1.2 : ℝ)
  let b := (1/2 : ℝ) ^ (-(0.8 : ℝ))
  let c := 2 * (Real.log 2 / Real.log 5)
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l713_71383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_ten_pound_bag_l713_71399

/-- Represents the cost and weight of a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  cost : ℚ

/-- Represents a purchase of grass seed bags -/
structure Purchase where
  bags : List GrassSeedBag
  totalWeight : Nat
  totalCost : ℚ

def fivePoundBag : GrassSeedBag := ⟨5, 1382/100⟩
def twentyFivePoundBag : GrassSeedBag := ⟨25, 3225/100⟩

/-- The minimum purchase amount in pounds -/
def minPurchase : Nat := 65

/-- The maximum purchase amount in pounds -/
def maxPurchase : Nat := 80

/-- The least possible cost for the purchase -/
def leastCost : ℚ := 9875/100

/-- Function to check if a bag is one of the allowed types -/
def isAllowedBag (b : GrassSeedBag) (tenPoundBag : GrassSeedBag) : Prop :=
  b = fivePoundBag ∨ b = tenPoundBag ∨ b = twentyFivePoundBag

/-- Theorem stating that the cost of a 10-pound bag is $2 -/
theorem cost_of_ten_pound_bag :
  ∃ (tenPoundBag : GrassSeedBag) (purchase : Purchase),
    tenPoundBag.weight = 10 ∧
    tenPoundBag.cost = 2 ∧
    purchase.bags.length ≥ 1 ∧
    (∀ b ∈ purchase.bags, isAllowedBag b tenPoundBag) ∧
    purchase.totalWeight ≥ minPurchase ∧
    purchase.totalWeight ≤ maxPurchase ∧
    purchase.totalCost = leastCost := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_ten_pound_bag_l713_71399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_in_G8_l713_71301

/-- Number of diamonds in design G_n -/
def num_diamonds : ℕ → ℕ
  | 0 => 3  -- We define G_1 as the case for 0 to match Nat indexing
  | n + 1 => num_diamonds n + 4 * (n + 2)

theorem diamonds_in_G8 : num_diamonds 7 = 195 := by
  sorry

#eval num_diamonds 7  -- This will evaluate the function for G_8 (index 7)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamonds_in_G8_l713_71301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cruise_ship_cabins_l713_71334

/-- Represents the total number of cabins on a cruise ship -/
def total_cabins : ℕ := 160

/-- Represents the number of Luxury cabins -/
def luxury_cabins : ℕ := 32

/-- Represents the percentage of Deluxe cabins -/
def deluxe_percentage : ℚ := 1/5

/-- Represents the percentage of Standard cabins -/
def standard_percentage : ℚ := 3/5

/-- Theorem stating that the total number of cabins is 160 -/
theorem cruise_ship_cabins : 
  luxury_cabins + 
  (deluxe_percentage * total_cabins).floor + 
  (standard_percentage * total_cabins).floor = 
  total_cabins := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cruise_ship_cabins_l713_71334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_business_gain_calculation_l713_71337

/-- Represents the business investment scenario -/
structure BusinessInvestment where
  nandan_investment : ℚ
  nandan_time : ℚ
  nandan_gain : ℚ
  krishan_investment : ℚ
  krishan_time : ℚ

/-- Calculates the total gain based on the given investment scenario -/
def total_gain (b : BusinessInvestment) : ℚ :=
  b.nandan_gain * (1 + (b.krishan_investment * b.krishan_time) / (b.nandan_investment * b.nandan_time))

/-- Theorem stating the total gain in the given scenario -/
theorem business_gain_calculation (b : BusinessInvestment) 
  (h1 : b.krishan_investment = 6 * b.nandan_investment)
  (h2 : b.krishan_time = 2 * b.nandan_time)
  (h3 : b.nandan_gain = 6000) :
  total_gain b = 78000 := by
  sorry

/-- Example calculation -/
def example_investment : BusinessInvestment := { 
  nandan_investment := 1, 
  nandan_time := 1, 
  nandan_gain := 6000, 
  krishan_investment := 6, 
  krishan_time := 2 
}

#eval total_gain example_investment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_business_gain_calculation_l713_71337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_congruence_l713_71387

theorem modular_congruence (n : ℕ) (h1 : n < 47) (h2 : (4 * n) % 47 = 1) :
  (3^n)^3 % 47 - 3 % 47 = 38 % 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_congruence_l713_71387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_free_of_arithmetic_progressions_l713_71349

def is_free_of_arithmetic_progressions (A : Set ℕ) : Prop :=
  ∀ a b c, a ∈ A → b ∈ A → c ∈ A → a ≠ b → b ≠ c → a ≠ c → a + b ≠ 2 * c

theorem subset_free_of_arithmetic_progressions :
  ∃ A : Finset ℕ,
    (∀ x ∈ A, x < 3^8) ∧
    is_free_of_arithmetic_progressions A ∧
    Finset.card A ≥ 256 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_free_of_arithmetic_progressions_l713_71349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diff_composite_sum_105_l713_71392

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def sum_to_105 (a b : ℕ) : Prop := a + b = 105

theorem min_diff_composite_sum_105 : 
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ sum_to_105 a b ∧
  ∀ (c d : ℕ), is_composite c → is_composite d → sum_to_105 c d →
  (Int.natAbs (a - b) : ℤ) ≤ (Int.natAbs (c - d) : ℤ) ∧ Int.natAbs (a - b) = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diff_composite_sum_105_l713_71392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l713_71361

theorem polynomial_divisibility (n : ℕ) : 
  (∃ (p : Polynomial ℤ) (m : ℤ), 
    Polynomial.degree p = n ∧ 
    (∀ k : ℤ, k ∈ ({m-1, m, m+1} : Set ℤ) → (3 : ℤ) ∣ p.eval k) → 
    (∀ i : ℕ, i ≤ n → (3 : ℤ) ∣ p.coeff i)) ↔ 
  n ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l713_71361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_to_circumscribed_sphere_volume_ratio_l713_71302

-- Define a regular tetrahedron
structure RegularTetrahedron :=
  (edge : ℝ)
  (edge_positive : edge > 0)

-- Define the inscribed sphere of a regular tetrahedron
noncomputable def inscribed_sphere_volume (t : RegularTetrahedron) : ℝ := 
  (4 * Real.pi * t.edge^3) / (216 * (3^(1/2)))

-- Define the circumscribed sphere of a regular tetrahedron
noncomputable def circumscribed_sphere_volume (t : RegularTetrahedron) : ℝ := 
  (4 * Real.pi * t.edge^3) / (8 * (3^(1/2)))

-- Theorem statement
theorem inscribed_to_circumscribed_sphere_volume_ratio 
  (t : RegularTetrahedron) : 
  inscribed_sphere_volume t / circumscribed_sphere_volume t = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_to_circumscribed_sphere_volume_ratio_l713_71302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abel_arrives_earlier_l713_71375

/-- Represents the travel times of Abel and Alice -/
structure TravelTimes where
  abel : ℚ
  alice : ℚ

/-- Calculates the travel times for Abel and Alice given the conditions -/
noncomputable def calculateTravelTimes (totalDistance : ℚ) (abelInitialSpeed : ℚ) (abelRainSpeed : ℚ) 
  (abelRainDistance : ℚ) (abelBreakTime : ℚ) (aliceSpeed : ℚ) (aliceExtraDistance : ℚ) 
  (aliceLaterStart : ℚ) : TravelTimes :=
  { abel := totalDistance / abelInitialSpeed + abelBreakTime,
    alice := (totalDistance + aliceExtraDistance) / aliceSpeed - aliceLaterStart }

/-- Theorem stating that Abel arrives 225 minutes earlier than Alice -/
theorem abel_arrives_earlier (totalDistance : ℚ) (abelInitialSpeed : ℚ) (abelRainSpeed : ℚ) 
  (abelRainDistance : ℚ) (abelBreakTime : ℚ) (aliceSpeed : ℚ) (aliceExtraDistance : ℚ) 
  (aliceLaterStart : ℚ) :
  totalDistance = 1000 ∧ 
  abelInitialSpeed = 50 ∧ 
  abelRainSpeed = 40 ∧ 
  abelRainDistance = 100 ∧ 
  abelBreakTime = 1/2 ∧ 
  aliceSpeed = 40 ∧ 
  aliceExtraDistance = 30 ∧ 
  aliceLaterStart = 1 →
  let times := calculateTravelTimes totalDistance abelInitialSpeed abelRainSpeed 
                 abelRainDistance abelBreakTime aliceSpeed aliceExtraDistance aliceLaterStart
  (times.alice - times.abel) * 60 = 225 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abel_arrives_earlier_l713_71375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_trajectory_length_l713_71326

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cone with an equilateral triangular axis section -/
structure Cone where
  A : Point3D
  B : Point3D
  S : Point3D
  O : Point3D
  M : Point3D

/-- The length of a vector -/
noncomputable def vectorLength (v : Point3D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2 + v.z^2)

/-- The dot product of two vectors -/
def dotProduct (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

/-- Checks if two vectors are perpendicular -/
def isPerpendicular (v1 v2 : Point3D) : Prop :=
  dotProduct v1 v2 = 0

/-- Vector subtraction -/
def vecSub (v1 v2 : Point3D) : Point3D :=
  ⟨v1.x - v2.x, v1.y - v2.y, v1.z - v2.z⟩

/-- The main theorem -/
theorem cone_trajectory_length (c : Cone) (P : Point3D) : 
  (vectorLength (vecSub c.A c.B) = 2) →  -- Side length of equilateral triangle is 2
  (c.M.z = c.S.z / 2) →  -- M is midpoint of SO
  (P.z = 0) →  -- P is in the base plane
  (isPerpendicular (vecSub c.A c.M) (vecSub c.M P)) →  -- AM ⊥ MP
  (vectorLength P ≤ 1) →  -- P is inside or on the base circle
  (∃ (trajectory : ℝ → Point3D), 
    (∀ t, trajectory t = P) ∧ 
    (vectorLength (vecSub (trajectory 1) (trajectory 0)) = Real.sqrt 7 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_trajectory_length_l713_71326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l713_71380

/-- The circle equation in polar coordinates -/
def circle_eq (ρ θ : ℝ) : Prop := ρ = Real.cos θ + Real.sin θ

/-- The line equation in polar coordinates -/
def line_eq (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

/-- Theorem stating that (1, π/2) is an intersection point of the circle and line -/
theorem intersection_point : 
  ∃ (ρ θ : ℝ), 
    θ ∈ Set.Ioo 0 Real.pi ∧ 
    circle_eq ρ θ ∧ 
    line_eq ρ θ ∧ 
    ρ = 1 ∧ 
    θ = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l713_71380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l713_71393

theorem angle_sum_proof (α β : Real) :
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.cos α = Real.sqrt 5 / 5 →
  Real.sin β = 3 * Real.sqrt 10 / 10 →
  α + β = 3 * π / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l713_71393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_sum_2012_eq_zero_l713_71363

open Complex BigOperators

def imaginary_sum (n : ℕ) : ℂ :=
  ∑ k in Finset.range n, (I ^ (k + 1))

theorem imaginary_sum_2012_eq_zero : imaginary_sum 2012 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_sum_2012_eq_zero_l713_71363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l713_71374

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l713_71374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l713_71381

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |Real.sin x + Real.cos x| + |Real.sin x - Real.cos x|

-- Theorem statement
theorem f_properties : 
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + Real.pi/2) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l713_71381
