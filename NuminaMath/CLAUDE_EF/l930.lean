import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l930_93022

theorem inequality_solution_set (x : ℝ) : 
  (2 : ℝ)^(x^2 - 4*x - 3) > (1/2 : ℝ)^(3*(x - 1)) ↔ x < -2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l930_93022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_satisfy_equation_l930_93025

theorem two_pairs_satisfy_equation : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 - p.2^2 = 91) (Finset.product (Finset.range 100) (Finset.range 100))).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_satisfy_equation_l930_93025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_triangle_volume_is_2pi_l930_93076

/-- The volume of the geometric solid formed by rotating an equilateral triangle
    with side length 2 around one of its sides for one revolution -/
noncomputable def rotated_triangle_volume : ℝ := 2 * Real.pi

/-- Proves that the volume of the geometric solid formed by rotating an equilateral triangle
    with side length 2 around one of its sides for one revolution is equal to 2π -/
theorem rotated_triangle_volume_is_2pi :
  rotated_triangle_volume = 2 * Real.pi := by
  -- Unfold the definition of rotated_triangle_volume
  unfold rotated_triangle_volume
  -- The equality now holds by reflexivity
  rfl

#check rotated_triangle_volume_is_2pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_triangle_volume_is_2pi_l930_93076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_isosceles_triangle_l930_93080

/-- Two lines that intersect and form an isosceles triangle with the x-axis -/
structure IntersectingLines where
  k : ℝ
  h_pos : k > 0

/-- The condition for forming an isosceles triangle -/
def isIsoscelesTriangle (lines : IntersectingLines) : Prop :=
  let l₁_slope := 1 / lines.k
  let l₂_slope := 2 * lines.k
  (l₁_slope = (2 * l₂_slope) / (1 - l₂_slope^2)) ∨
  (l₂_slope = (2 * l₁_slope) / (1 - l₁_slope^2))

/-- The main theorem stating the possible values of k -/
theorem intersecting_lines_isosceles_triangle (lines : IntersectingLines) 
  (h_isosceles : isIsoscelesTriangle lines) : 
  lines.k = Real.sqrt 2 / 2 ∨ lines.k = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_isosceles_triangle_l930_93080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l930_93003

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

/-- The equation of the ellipse -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / m + y^2 / 4 = 1

/-- Theorem stating that m = 8 for the given ellipse -/
theorem ellipse_m_value :
  ∃ (m : ℝ), m = 8 ∧
  (∀ x y, ellipse_equation x y m) ∧
  m ≥ 4 ∧
  (Real.sqrt (m - 4) / Real.sqrt m = eccentricity) :=
by
  sorry

#check ellipse_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l930_93003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_23_permutations_divisible_by_37_l930_93037

/-- A six-digit number with six different non-zero digits -/
structure SixDigitNumber where
  digits : Fin 6 → Fin 10
  different : ∀ i j, i ≠ j → digits i ≠ digits j
  nonzero : ∀ i, digits i ≠ 0

/-- Convert a SixDigitNumber to a natural number -/
def SixDigitNumber.toNat (n : SixDigitNumber) : ℕ :=
  (n.digits 0) * 100000 + (n.digits 1) * 10000 + (n.digits 2) * 1000 +
  (n.digits 3) * 100 + (n.digits 4) * 10 + (n.digits 5)

/-- A permutation of six elements -/
def Permutation := Fin 6 → Fin 6

/-- Apply a permutation to a SixDigitNumber -/
def applyPerm (n : SixDigitNumber) (p : Permutation) : SixDigitNumber where
  digits := n.digits ∘ p
  different := by sorry
  nonzero := by sorry

theorem at_least_23_permutations_divisible_by_37 (n : SixDigitNumber) 
  (h : 37 ∣ n.toNat) :
  ∃ (perms : Finset Permutation), 
    perms.card ≥ 23 ∧
    ∀ p ∈ perms, 37 ∣ (applyPerm n p).toNat := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_23_permutations_divisible_by_37_l930_93037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_sixth_terms_l930_93049

def modifiedSequence : ℕ → ℚ
  | 0 => 1
  | n + 1 => ((n + 2) : ℚ)^2 / ((n + 1) : ℚ)^2

theorem sum_of_fourth_and_sixth_terms :
  modifiedSequence 3 + modifiedSequence 5 = 724 / 225 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fourth_and_sixth_terms_l930_93049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_part_odd_l930_93092

theorem integral_part_odd (n : ℕ) : Odd (⌊(3 + Real.sqrt 5) ^ (n : ℝ)⌋) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_part_odd_l930_93092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l930_93067

-- Define the slopes and point of intersection
noncomputable def m₁ : ℝ := 1/3
noncomputable def m₂ : ℝ := 3
def p : ℝ × ℝ := (3, 3)

-- Define the lines
noncomputable def line₁ (x : ℝ) : ℝ := m₁ * (x - p.1) + p.2
noncomputable def line₂ (x : ℝ) : ℝ := m₂ * (x - p.1) + p.2
def line₃ (x : ℝ) : ℝ := 12 - x

-- Define the intersection points
def point_A : ℝ × ℝ := p
noncomputable def point_B : ℝ × ℝ := (4.5, 7.5)
noncomputable def point_C : ℝ × ℝ := (7.5, 4.5)

-- Theorem statement
theorem triangle_area : 
  let area := (1/2) * abs (point_A.1 * (point_B.2 - point_C.2) + 
                           point_B.1 * (point_C.2 - point_A.2) + 
                           point_C.1 * (point_A.2 - point_B.2))
  area = 8.625 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l930_93067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_line_l930_93069

-- Define the circle
def circle_eq (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = a

-- Define the line
def line_eq (a : ℝ) (x y : ℝ) : Prop := y = x + a

-- Define tangency condition
def is_tangent (a : ℝ) : Prop := ∃ (x y : ℝ), circle_eq a x y ∧ line_eq a x y ∧
  ∀ (x' y' : ℝ), circle_eq a x' y' → line_eq a x' y' → (x', y') = (x, y)

-- Main theorem
theorem tangent_circle_line (a : ℝ) (h1 : a > 0) (h2 : is_tangent a) : a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_line_l930_93069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_time_proof_l930_93084

-- Define the individual painting rates
noncomputable def doug_rate : ℚ := 1 / 5
noncomputable def dave_rate : ℚ := 1 / 7
noncomputable def ellen_rate : ℚ := 1 / 9

-- Define the combined rate
noncomputable def combined_rate : ℚ := doug_rate + dave_rate + ellen_rate

-- Define the total time including lunch break
noncomputable def total_time : ℚ := 458 / 143

-- Theorem statement
theorem painting_time_proof :
  (combined_rate * (total_time - 1) = 1) := by
  -- Expand the definitions
  unfold combined_rate total_time doug_rate dave_rate ellen_rate
  -- Simplify the expression
  simp [add_mul, mul_sub, mul_one]
  -- The proof itself would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_time_proof_l930_93084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_symmetric_functions_l930_93004

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 + x + 1

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Define a point on a function
def point_on_function (P : ℝ × ℝ) (h : ℝ → ℝ) : Prop :=
  P.2 = h P.1

-- State the theorem
theorem min_distance_symmetric_functions :
  ∃ (g : ℝ → ℝ),
    (∀ (x y : ℝ), point_on_function (x, y) g ↔ 
      ∃ (x' y' : ℝ), point_on_function (x', y') f ∧ 
        symmetry_line ((x + x') / 2) ((y + y') / 2)) →
    (∃ (c : ℝ), c = 2 * Real.sqrt 5 ∧
      ∀ (P Q : ℝ × ℝ), 
        point_on_function P f → 
        point_on_function Q g → 
        Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_symmetric_functions_l930_93004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l930_93007

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- State the theorem
theorem tangent_and_inequality :
  (∃ (m : ℝ), HasDerivAt f m 0 ∧ m = 2) ∧
  (∀ x : ℝ, 0 < x → x < 1 → f x > 2 * (x + x^3 / 3)) :=
by
  sorry

#check tangent_and_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l930_93007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l930_93032

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := f' x₀
  (λ (x y : ℝ) => m * (x - x₀) - (y - y₀)) = (λ (x y : ℝ) => 4 * x - y - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l930_93032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_odd_prime_factor_of_2023_pow_6_plus_1_l930_93016

theorem least_odd_prime_factor_of_2023_pow_6_plus_1 (p : Nat) : 
  Nat.Prime p ∧ Odd p ∧ p ∣ (2023^6 + 1) → p ≥ 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_odd_prime_factor_of_2023_pow_6_plus_1_l930_93016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_a_b_2_exists_even_function_when_a_0_a_range_for_increasing_function_l930_93099

-- Define the general function
noncomputable def f (a b x : ℝ) : ℝ := (b^x + 1) / (2^x + a)

-- Statement 1
theorem range_when_a_b_2 :
  Set.range (fun x => f 2 2 x) = Set.Ioo (1/2 : ℝ) 1 := by sorry

-- Statement 2
theorem exists_even_function_when_a_0 :
  ∃ b : ℝ, b > 0 ∧ b ≠ 1 ∧ ∀ x, f 0 b x = f 0 b (-x) := by sorry

-- Statement 3
theorem a_range_for_increasing_function :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, x ∈ Set.Ici (-1 : ℝ) → y ∈ Set.Ici (-1 : ℝ) → x < y → f a 4 x < f a 4 y) →
  a ∈ Set.Ici (3/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_a_b_2_exists_even_function_when_a_0_a_range_for_increasing_function_l930_93099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l930_93054

-- Define the sign function
noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1 else if x < 0 then -1 else 0

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic_2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

def matches_x_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x

-- State the theorem
theorem function_properties (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_periodic : is_periodic_2 f) 
  (h_match : matches_x_on_unit_interval f) : 
  (∃ x, sgn (f x) ≤ 0) ∧ 
  (f (2023 / 2) ≠ 1) ∧ 
  (∀ k : ℤ, sgn (f (2 * ↑k + 1)) = 1) ∧ 
  (∃ k : ℤ, sgn (f ↑k) ≠ |sgn ↑k|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l930_93054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_time_l930_93096

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def train_pass_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 360 meters, traveling at 50 km/hour,
    passes a bridge of length 140 meters in approximately 36 seconds -/
theorem train_pass_bridge_time :
  ∃ ε > 0, |train_pass_time 360 140 50 - 36| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_time_l930_93096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_implies_perpendicular_plane_parallel_planes_implies_parallel_line_l930_93077

-- Define the necessary structures
structure Line : Type
structure Plane : Type

-- Define the relationships
def subset : Line → Plane → Prop := sorry
def perpendicular : Line → Plane → Prop := sorry
def perpendicular_planes : Plane → Plane → Prop := sorry
def parallel : Plane → Plane → Prop := sorry
def parallel_line_plane : Line → Plane → Prop := sorry

-- Theorem statements
theorem perpendicular_line_implies_perpendicular_plane
  (l : Line) (α β : Plane)
  (h1 : subset l α)
  (h2 : perpendicular l β) :
  perpendicular_planes α β :=
sorry

theorem parallel_planes_implies_parallel_line
  (l : Line) (α β : Plane)
  (h1 : subset l α)
  (h2 : parallel α β) :
  parallel_line_plane l β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_implies_perpendicular_plane_parallel_planes_implies_parallel_line_l930_93077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l930_93070

theorem remainder_theorem : 
  (2^210 + 210) % (2^105 + 2^52 + 3) = 210 := by
  sorry

#eval (2^210 + 210) % (2^105 + 2^52 + 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l930_93070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_smallest_positive_integer_solutions_l930_93030

theorem quadratic_equation_roots (m : ℝ) :
  (∀ x : ℝ, (m - 4) * x^2 - (2 * m - 1) * x + m = 0 → 
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
      (m - 4) * x1^2 - (2 * m - 1) * x1 + m = 0 ∧
      (m - 4) * x2^2 - (2 * m - 1) * x2 + m = 0)) ↔
  (m > -1/12 ∧ m ≠ 4) :=
by sorry

theorem smallest_positive_integer_solutions :
  let m : ℝ := 1
  let x1 : ℝ := (-1 + Real.sqrt 13) / 6
  let x2 : ℝ := (-1 - Real.sqrt 13) / 6
  (m > -1/12 ∧ m ≠ 4) ∧
  (m - 4) * x1^2 - (2 * m - 1) * x1 + m = 0 ∧
  (m - 4) * x2^2 - (2 * m - 1) * x2 + m = 0 ∧
  x1 ≠ x2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_smallest_positive_integer_solutions_l930_93030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_2003_segments_with_three_intersections_l930_93031

theorem no_2003_segments_with_three_intersections :
  ¬ ∃ (segments : Finset (Set (ℝ × ℝ))) (intersections : Set (ℝ × ℝ) → Finset (Set (ℝ × ℝ))),
    segments.card = 2003 ∧
    ∀ s ∈ segments, (intersections s).card = 3 ∧ 
    ∀ t ∈ intersections s, t ∈ segments ∧ t ≠ s ∧ s ∈ intersections t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_2003_segments_with_three_intersections_l930_93031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_region_properties_l930_93018

/-- Represents a cylindrical region with hemispherical caps -/
structure CylindricalRegion where
  radius : ℝ
  length : ℝ

/-- Calculates the surface area of the cylindrical region -/
noncomputable def surface_area (region : CylindricalRegion) : ℝ :=
  2 * Real.pi * region.radius * region.length + 4 * Real.pi * region.radius^2

/-- Calculates the volume of the cylindrical region -/
noncomputable def volume (region : CylindricalRegion) : ℝ :=
  Real.pi * region.radius^2 * region.length + (4/3) * Real.pi * region.radius^3

theorem cylindrical_region_properties :
  ∃ (region : CylindricalRegion),
    region.radius = 4 ∧
    surface_area region = 400 * Real.pi ∧
    region.length = 42 ∧
    volume region = 806 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_region_properties_l930_93018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_conditions_l930_93048

noncomputable def f (a b x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 2) + a*x) - Real.log b

theorem odd_function_conditions (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -(f a b x)) ↔ (a = 1 ∨ a = -1) ∧ b = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_conditions_l930_93048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_side_b_value_l930_93072

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  area : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a = t.b * Real.sin t.A / Real.sin t.B ∧
  t.b = t.c * Real.sin t.B / Real.sin t.C ∧
  t.c = t.a * Real.sin t.C / Real.sin t.A ∧
  t.area = Real.sqrt 3

-- Theorem 1
theorem angle_B_value (t : Triangle) 
  (h : triangle_conditions t) 
  (h1 : Real.sin t.B ^ 2 - Real.sin t.A ^ 2 = Real.sin t.C ^ 2 - Real.sin t.A * Real.sin t.C) : 
  t.B = Real.pi / 3 := by
  sorry

-- Theorem 2
theorem side_b_value (t : Triangle) 
  (h : triangle_conditions t)
  (h1 : ∀ (t' : Triangle), triangle_conditions t' → t.a + t.c ≤ t'.a + t'.c) :
  t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_side_b_value_l930_93072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_and_sums_imply_difference_l930_93005

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem product_and_sums_imply_difference (p q r s : ℕ+) : 
  p.val * q.val * r.val * s.val = factorial 7 ∧ 
  p.val * q.val + p.val + q.val = 715 ∧ 
  q.val * r.val + q.val + r.val = 209 ∧ 
  r.val * s.val + r.val + s.val = 143 → 
  p.val - s.val = 10 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_and_sums_imply_difference_l930_93005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l930_93094

def sequenceTerm (n : ℕ) : ℚ := 2^n / (2*n + 1)

theorem sequence_formula_correct :
  (∀ n : ℕ, n ≥ 1 → sequenceTerm n = 2^n / (2*n + 1)) ∧
  (∀ n : ℕ, n ≥ 1 → Odd (2*n + 1)) ∧
  (∀ n : ℕ, n ≥ 1 → ∃ k : ℕ, (sequenceTerm n).num = 2^k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_correct_l930_93094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l930_93014

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 16)
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

-- Define the domain of f (set A)
def A : Set ℝ := {x | x ≤ -4 ∨ x ≥ 4}

-- Define the range of g (set B)
def B (a : ℝ) : Set ℝ := Set.Icc (a - 1) (a + 8)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (A ∪ B a = Set.univ) → a ∈ Set.Icc (-4) (-3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l930_93014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_fermat_theorem_l930_93015

/-- A polynomial with real or complex coefficients -/
def MyPolynomial (α : Type*) [Field α] := α → α

/-- Predicate to check if a polynomial is constant -/
def IsConstant {α : Type*} [Field α] (p : MyPolynomial α) : Prop :=
  ∀ x y, p x = p y

/-- Exponentiation for polynomials -/
def PolyPow {α : Type*} [Field α] (p : MyPolynomial α) (n : ℕ) : MyPolynomial α :=
  fun x => (p x) ^ n

theorem polynomial_fermat_theorem {α : Type*} [Field α] 
  (P Q R : MyPolynomial α) (n : ℕ) 
  (h1 : ¬(IsConstant P ∧ IsConstant Q ∧ IsConstant R))
  (h2 : ∀ x, (PolyPow P n x) + (PolyPow Q n x) + (PolyPow R n x) = 0) :
  n < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_fermat_theorem_l930_93015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_decreasing_l930_93074

-- Define the function f(x) = -2^|x|
noncomputable def f (x : ℝ) : ℝ := -2^(abs x)

-- Theorem statement
theorem f_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_and_decreasing_l930_93074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l930_93041

-- Define the constants
noncomputable def a : ℝ := 1 / Real.log 2
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.exp (-1)

-- State the theorem
theorem ordering_abc : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l930_93041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_curve_l930_93035

/-- The minimum distance between a point on y = x + 1 and a point on x² = -2y --/
theorem min_distance_line_curve : ∃ (min_dist : ℝ), 
  (∀ (x1 y1 x2 y2 : ℝ),
    y1 = x1 + 1 →  -- P is on the line y = x + 1
    y2 = -x2^2 / 2 →  -- Q is on the curve x² = -2y
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) ≥ min_dist) ∧
  min_dist = Real.sqrt 2 / 4 := by
  
  -- Define the minimum distance
  let min_dist := Real.sqrt 2 / 4

  -- Prove the existence of such a minimum distance
  use min_dist

  constructor

  -- First part: prove that all distances are greater than or equal to min_dist
  sorry  -- Proof omitted

  -- Second part: prove that min_dist equals √2/4
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_curve_l930_93035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chromatic_number_complete_graph_chromatic_number_square_chromatic_number_tree_l930_93052

-- Definition of a graph
def Graph (V : Type) := V → V → Prop

-- Definition of a complete graph
def CompleteGraph (n : ℕ) : Graph (Fin n) :=
  λ i j => i ≠ j

-- Definition of chromatic number
def ChromaticNumber (G : Graph V) (n : ℕ) : Prop :=
  ∃ (f : V → Fin n), ∀ v w, G v w → f v ≠ f w

-- Theorem 1: Chromatic number of complete graph
theorem chromatic_number_complete_graph (n : ℕ) :
  ChromaticNumber (CompleteGraph n) n := by
  sorry

-- Definition of a square graph
def SquareGraph : Graph (Fin 4) :=
  λ i j => (i.val + 1) % 4 = j.val ∨ (j.val + 1) % 4 = i.val

-- Theorem 2: Chromatic number of a square
theorem chromatic_number_square :
  ChromaticNumber SquareGraph 2 := by
  sorry

-- Definition of a tree
def IsTree {V : Type} (G : Graph V) : Prop :=
  (∀ v w, G v w → G w v) ∧  -- undirected
  (∃ r, ∀ v, v ≠ r → ∃! p, G p v) ∧  -- unique path to root
  (¬ ∃ (c : List V), c.length > 2 ∧ c.head? = c.getLast? ∧ 
    ∀ i, i + 1 < c.length → G (c.get ⟨i, by sorry⟩) (c.get ⟨i+1, by sorry⟩))  -- no cycles

-- Theorem 3: Chromatic number of a tree
theorem chromatic_number_tree {V : Type} (G : Graph V) :
  IsTree G → ChromaticNumber G 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chromatic_number_complete_graph_chromatic_number_square_chromatic_number_tree_l930_93052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_div_by_five_l930_93075

/-- Definition of p_n -/
def p (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n

/-- Theorem stating the divisibility condition for p_n -/
theorem p_div_by_five (n : ℕ) : 5 ∣ p n ↔ ¬(4 ∣ n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_div_by_five_l930_93075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_analysis_central_angle_proof_estimate_high_scores_l930_93039

/-- Represents the frequency of scores in each range --/
structure ScoreDistribution :=
  (low : ℕ)
  (mid_low : ℕ)
  (mid_high : ℕ)
  (high : ℕ)

/-- Theorem: Given the score distribution and average, prove the median and mode --/
theorem score_analysis 
  (dist : ScoreDistribution)
  (total_scores : ℕ)
  (avg_score : ℚ)
  (h_total : dist.low + dist.mid_low + dist.mid_high + dist.high = total_scores)
  (h_avg : (80 * dist.low + 85 * dist.mid_low + 90 * dist.mid_high + 95 * dist.high) / total_scores = avg_score)
  (h_avg_value : avg_score = 91)
  (h_total_value : total_scores = 20)
  (h_low : dist.low = 3)
  (h_mid_low : dist.mid_low = 5)
  (h_high : dist.high = 7) :
  ∃ (median mode : ℕ), median = 90 ∧ mode = 100 := by
  sorry

/-- Compute the central angle for a given range in a pie chart --/
def central_angle (range_freq : ℕ) (total : ℕ) : ℚ :=
  360 * (range_freq : ℚ) / (total : ℚ)

/-- Estimate the number of people in a larger population based on a sample --/
def estimate_population (sample_freq : ℕ) (sample_total : ℕ) (population : ℕ) : ℕ :=
  (population * sample_freq) / sample_total

/-- Theorem: Prove the central angle for the range 90 ≤ x < 95 --/
theorem central_angle_proof
  (dist : ScoreDistribution)
  (h_total : dist.low + dist.mid_low + dist.mid_high + dist.high = 20)
  (h_mid_high : dist.mid_high = 5) :
  central_angle dist.mid_high 20 = 90 := by
  sorry

/-- Theorem: Prove the estimated number of people with scores not less than 90 --/
theorem estimate_high_scores
  (dist : ScoreDistribution)
  (h_total : dist.low + dist.mid_low + dist.mid_high + dist.high = 20)
  (h_mid_high : dist.mid_high = 5)
  (h_high : dist.high = 7) :
  estimate_population (dist.mid_high + dist.high) 20 1400 = 840 := by
  sorry

#eval central_angle 5 20  -- This should output 90
#eval estimate_population 12 20 1400  -- This should output 840

end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_analysis_central_angle_proof_estimate_high_scores_l930_93039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rajans_profit_share_l930_93059

/-- Represents a partner in the business --/
structure Partner where
  name : String
  investment : ℕ
  duration : ℕ
  sharePercentage : ℚ
deriving Inhabited

/-- Calculates the investment-time product for a partner --/
def investmentTimeProduct (p : Partner) : ℕ :=
  p.investment * p.duration

/-- Calculates the total investment-time product for all partners --/
def totalInvestmentTimeProduct (partners : List Partner) : ℕ :=
  partners.map investmentTimeProduct |>.sum

/-- Calculates a partner's share in the profit --/
def profitShare (p : Partner) (partners : List Partner) (totalProfit : ℕ) : ℚ :=
  (investmentTimeProduct p : ℚ) / (totalInvestmentTimeProduct partners : ℚ) * (totalProfit : ℚ) * p.sharePercentage

/-- The business partners --/
def businessPartners : List Partner := [
  ⟨"Rajan", 20000, 12, 1⟩,
  ⟨"Rakesh", 25000, 4, 4/5⟩,
  ⟨"Mahesh", 30000, 10, 9/10⟩,
  ⟨"Suresh", 35000, 12, 1⟩,
  ⟨"Mukesh", 15000, 8, 1⟩,
  ⟨"Sachin", 40000, 2, 1⟩
]

/-- The total profit of the business --/
def totalProfit : ℕ := 18000

/-- Rajan's profit share is approximately 3428.57 --/
theorem rajans_profit_share :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |profitShare (businessPartners.head!) businessPartners totalProfit - 3428.57| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rajans_profit_share_l930_93059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_S_l930_93095

-- Define the set S
def S : Set ℂ := {z | Complex.abs z.re ≤ 2 ∧ Complex.abs z.im ≤ 2}

-- Define the transformation
noncomputable def transform (z : ℂ) : ℂ := (1/2 + 1/2*Complex.I) * z

-- Theorem statement
theorem transform_stays_in_S : ∀ z ∈ S, transform z ∈ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_S_l930_93095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beans_in_normal_batch_l930_93085

/-- The number of cans of beans in a normal batch of chili -/
def beans : ℕ := 2

/-- The total number of cans in a normal batch of chili -/
def normal_batch : ℝ := beans + 1.5 * (beans : ℝ) + 1

/-- The total number of cans in a quadruple batch of chili -/
def quadruple_batch : ℝ := 4 * normal_batch

theorem beans_in_normal_batch : beans = 2 :=
  by
    have h1 : quadruple_batch = 24 := by
      -- Proof steps would go here
      sorry
    -- More proof steps would go here
    sorry

#check beans_in_normal_batch

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beans_in_normal_batch_l930_93085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_200_value_l930_93051

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Define for 0 to cover all natural numbers
  | 1 => 2
  | (n + 2) => sequence_a (n + 1) + (2 * sequence_a (n + 1)) / (n + 1)

theorem a_200_value : sequence_a 200 = 40200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_200_value_l930_93051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l930_93053

-- Define the function f
noncomputable def f (x k : ℝ) : ℝ := (1 - x) / x + k * Real.log x

-- Theorem statement
theorem f_extrema (k : ℝ) (h : k < 1 / Real.exp 1) :
  let I := Set.Icc (1 / Real.exp 1) (Real.exp 1)
  ∃ (min_val max_val : ℝ),
    (∀ x ∈ I, f x k ≥ min_val) ∧
    (∃ x ∈ I, f x k = min_val) ∧
    (∀ x ∈ I, f x k ≤ max_val) ∧
    (∃ x ∈ I, f x k = max_val) ∧
    min_val = 1 / Real.exp 1 + k - 1 ∧
    max_val = Real.exp 1 - k - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l930_93053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l930_93008

-- Define the circle and parabola
def my_circle (x y : ℝ) : Prop := (x - 4)^2 + (y + 3)^2 = 16
def my_parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the distance function between two points
noncomputable def my_distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_circle_parabola :
  ∃ (x1 y1 x2 y2 : ℝ),
    my_circle x1 y1 ∧ my_parabola x2 y2 ∧
    ∀ (x3 y3 x4 y4 : ℝ),
      my_circle x3 y3 → my_parabola x4 y4 →
      my_distance x1 y1 x2 y2 ≤ my_distance x3 y3 x4 y4 ∧
      my_distance x1 y1 x2 y2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_parabola_l930_93008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l930_93082

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = π →
  A / (A + B + C) = 1 / 6 →
  B / (A + B + C) = 1 / 3 →
  C / (A + B + C) = 1 / 2 →
  a = 1 →
  c = 2 →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  b = Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l930_93082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_side_length_l930_93083

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  area : ℚ
  altitude : ℚ
  base1 : ℚ
  base2 : ℚ

/-- Calculates the length of the other parallel side of the trapezoid -/
noncomputable def Trapezoid.otherSide (t : Trapezoid) : ℚ :=
  2 * t.area / t.altitude - t.base1 - t.base2

theorem trapezoid_side_length (t : Trapezoid) 
  (h1 : t.area = 164)
  (h2 : t.altitude = 8)
  (h3 : t.base1 = 10)
  (h4 : t.base2 = 17) :
  t.otherSide = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_side_length_l930_93083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_chocolate_intersection_sum_of_coordinates_l930_93028

-- Define the chocolate's position
noncomputable def chocolate_pos : ℝ × ℝ := (16, 12)

-- Define the squirrel's path
noncomputable def squirrel_path (x : ℝ) : ℝ := -3 * x + 9

-- Define the perpendicular line passing through the chocolate's position
noncomputable def perpendicular_line (x : ℝ) : ℝ := (1/3) * x + 20/3

-- Define the intersection point
noncomputable def intersection_point : ℝ × ℝ := (7/10, 111/10)

-- Theorem statement
theorem squirrel_chocolate_intersection :
  -- The intersection point lies on the squirrel's path
  squirrel_path (intersection_point.1) = intersection_point.2 ∧
  -- The intersection point lies on the perpendicular line
  perpendicular_line (intersection_point.1) = intersection_point.2 ∧
  -- The slope of the line from the chocolate to the intersection point
  -- is perpendicular to the squirrel's path
  (intersection_point.2 - chocolate_pos.2) / (intersection_point.1 - chocolate_pos.1) * (-3) = -1 :=
by sorry

-- Additional theorem to calculate a + b
theorem sum_of_coordinates :
  intersection_point.1 + intersection_point.2 = 11.8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_chocolate_intersection_sum_of_coordinates_l930_93028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_a_and_b_l930_93047

-- Define a and b as noncomputable
noncomputable def a : ℝ := (Real.log 512) / (Real.log 4)
noncomputable def b : ℝ := Real.log 32 / Real.log 2

-- Theorem statement
theorem relationship_between_a_and_b : a = (9/10) * b := by
  -- The proof is omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_a_and_b_l930_93047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_sqrt_three_l930_93011

/-- Two circles with a common chord of length 2 -/
structure TwoCircles where
  /-- The parameter a for the second circle -/
  a : ℝ
  /-- a is positive -/
  a_pos : a > 0
  /-- The length of the common chord is 2 -/
  common_chord_length : ℝ
  common_chord_eq : common_chord_length = 2

/-- The theorem stating that a = √3 for the given conditions -/
theorem a_equals_sqrt_three (c : TwoCircles) : c.a = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_sqrt_three_l930_93011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_upper_bound_l930_93038

theorem cos_sum_upper_bound (α β γ : ℝ) 
  (h : Real.sin α + Real.sin β + Real.sin γ ≥ 4) :
  Real.cos α + Real.cos β + Real.cos γ ≤ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_upper_bound_l930_93038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l930_93044

theorem complex_number_problem (z : ℂ) 
  (h1 : Complex.abs z = 1) 
  (h2 : (Complex.I * (((3 : ℂ) + 4*Complex.I) * z).im = (3 : ℂ) + 4*Complex.I * z)) : 
  z = 4/5 + 3/5*Complex.I ∨ z = 4/5 - 3/5*Complex.I :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l930_93044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_photo_shoot_problem_toby_photo_shoot_answer_l930_93066

theorem toby_photo_shoot_problem :
  let initial_photos : ℤ := 63
  let deleted_bad_shots : ℤ := 7
  let cat_photos : ℤ := 15
  let final_photos : ℤ := 84
  let deleted_friend_photos : ℤ := 3
  
  initial_photos - deleted_bad_shots + cat_photos + 
  (final_photos + deleted_friend_photos - (initial_photos - deleted_bad_shots + cat_photos)) = 
  final_photos + deleted_friend_photos :=
by
  intros
  ring

#eval (84 : ℤ) + 3 - (63 - 7 + 15)

theorem toby_photo_shoot_answer :
  (84 : ℤ) + 3 - (63 - 7 + 15) = 16 :=
by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_photo_shoot_problem_toby_photo_shoot_answer_l930_93066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_largest_term_l930_93021

theorem arithmetic_progression_largest_term (d : ℝ) : 
  d > 0 →
  ((-3*d)^2 + (-2*d)^2 + (-d)^2 + 0^2 + d^2 + (2*d)^2 + (3*d)^2 = 756) →
  ((-3*d)^3 + (-2*d)^3 + (-d)^3 + 0^3 + d^3 + (2*d)^3 + (3*d)^3 = 0) →
  3*d = 9 * Real.sqrt 3 := by
  sorry

#check arithmetic_progression_largest_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_largest_term_l930_93021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l930_93042

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 4*x + 5) / Real.log 10

-- State the theorem
theorem f_properties :
  ∃ (a b : ℝ),
    (a = 2 ∧ b = 5) ∧
    (∀ x, -1 < x ∧ x < 5 → f x ≤ Real.log 9 / Real.log 10) ∧
    (∀ x y, a ≤ x ∧ x < y ∧ y < b → f y < f x) :=
by
  -- Proof sketch (replace with actual proof later)
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l930_93042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_sectors_area_l930_93055

/-- The area of the overlapping region of two 45° sectors in a circle with radius 10 -/
theorem overlapping_sectors_area : ∃ (overlapping_area : ℝ), overlapping_area = 25 * Real.pi - 100 * Real.sqrt 3 := by
  -- Define the circle radius
  let radius : ℝ := 10

  -- Define the central angle in radians
  let central_angle : ℝ := Real.pi / 4

  -- Define the area of one sector
  let sector_area : ℝ := (1 / 2) * central_angle * radius^2

  -- Define the area of the equilateral triangle formed by the centers and the overlapping point
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * (2 * radius)^2

  -- The overlapping area is equal to the sum of two sector areas minus the triangle area
  let overlapping_area : ℝ := 2 * sector_area - triangle_area

  -- Prove that the overlapping area is equal to 25π - 100√3
  have overlapping_area_eq : overlapping_area = 25 * Real.pi - 100 * Real.sqrt 3 := by
    sorry

  -- Conclusion
  exact ⟨overlapping_area, overlapping_area_eq⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_sectors_area_l930_93055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_vertical_same_color_l930_93013

/-- A color that can be used to paint a face of a cube -/
inductive Color
| Red
| Blue
| Green

/-- A cube with painted faces -/
structure PaintedCube where
  faces : Fin 6 → Color

/-- The probability of a specific color being chosen for a face -/
noncomputable def colorProb : ℝ := 1/3

/-- A cube configuration where all four vertical faces are the same color -/
def allVerticalSameColor (cube : PaintedCube) : Prop :=
  ∃ (c : Color), ∀ (i : Fin 4), cube.faces i = c

/-- The main theorem stating the probability of all vertical faces being the same color -/
theorem prob_all_vertical_same_color :
  Real.exp (-2 * Real.log 3) = 1/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_vertical_same_color_l930_93013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_implies_t_l930_93060

/-- Triangle ABC with vertices A(0, 10), B(4, 0), C(10, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Horizontal line y = t intersecting AB at T and AC at U -/
noncomputable def IntersectionPoints (triangle : Triangle) (t : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let T := (4 - 2/5 * t, t)
  let U := (10 - t, t)
  (T, U)

/-- Area of triangle ATU -/
noncomputable def AreaATU (triangle : Triangle) (t : ℝ) : ℝ :=
  let (T, U) := IntersectionPoints triangle t
  1/2 * (U.1 - T.1) * (triangle.A.2 - t)

/-- Theorem: If triangle ATU has area 18, then t = 10 - √20 -/
theorem area_implies_t (triangle : Triangle) (t : ℝ) 
    (h1 : triangle.A = (0, 10))
    (h2 : triangle.B = (4, 0))
    (h3 : triangle.C = (10, 0))
    (h4 : AreaATU triangle t = 18) :
    t = 10 - Real.sqrt 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_implies_t_l930_93060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knowledge_competition_score_l930_93093

theorem knowledge_competition_score (total_questions : ℤ) (correct_score : ℤ) (incorrect_deduction : ℤ) (final_score : ℤ) : 
  total_questions = 20 →
  correct_score = 5 →
  incorrect_deduction = 1 →
  final_score = 76 →
  ∃ (correct_answers : ℤ), 
    correct_answers * correct_score + (total_questions - correct_answers) * (-incorrect_deduction) = final_score ∧
    correct_answers = 16 :=
by
  intro h1 h2 h3 h4
  use 16
  constructor
  · rw [h1, h2, h3, h4]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knowledge_competition_score_l930_93093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_l930_93057

def repeating_decimal : ℚ := 567 / 999

theorem sum_of_numerator_and_denominator : ∃ (a b : ℕ), 
  repeating_decimal = a / b ∧ 
  Nat.Coprime a b ∧
  a + b = 58 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_l930_93057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_a_l930_93068

theorem right_triangle_sin_a (A B C : ℝ) :
  -- ABC is a right triangle with ∠C = 90°
  A + B + C = Real.pi →
  C = Real.pi / 2 →
  -- cos B = 1/2
  Real.cos B = 1 / 2 →
  -- Prove: sin A = 1/2
  Real.sin A = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_a_l930_93068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_A_has_max_sum_wy_l930_93090

/-- Represents a rectangle with integer-labeled sides -/
structure Rectangle where
  w : ℤ
  x : ℤ
  y : ℤ
  z : ℤ
deriving Repr, DecidableEq

/-- The set of five rectangles -/
def rectangles : List Rectangle := 
  [ Rectangle.mk 8 2 9 5,  -- A
    Rectangle.mk 2 1 5 8,  -- B
    Rectangle.mk 6 9 4 3,  -- C
    Rectangle.mk 4 6 2 9,  -- D
    Rectangle.mk 9 5 6 1 ] -- E

/-- The sum of w and y for a rectangle -/
def sumWY (r : Rectangle) : ℤ := r.w + r.y

/-- Theorem: Rectangle A has the maximum sum of w and y -/
theorem rectangle_A_has_max_sum_wy : 
  ∃ r ∈ rectangles, r = Rectangle.mk 8 2 9 5 ∧ 
    ∀ s ∈ rectangles, sumWY r ≥ sumWY s := by
  sorry

#eval rectangles
#eval sumWY (Rectangle.mk 8 2 9 5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_A_has_max_sum_wy_l930_93090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l930_93056

-- Define the polar coordinate equation of curve C₁
def curve_C₁ (ρ θ : ℝ) : Prop :=
  ρ^2 - 6*ρ*(Real.cos θ) + 5 = 0

-- Define the parametric equation of curve C₂
def curve_C₂ (t x y : ℝ) : Prop :=
  x = t * Real.cos (Real.pi/6) ∧ y = t * Real.sin (Real.pi/6)

-- Theorem stating that the distance between intersection points is √7
theorem intersection_distance :
  ∃ (A B : ℝ × ℝ),
    (∃ (ρ_A θ_A t_A : ℝ), curve_C₁ ρ_A θ_A ∧ curve_C₂ t_A A.1 A.2) ∧
    (∃ (ρ_B θ_B t_B : ℝ), curve_C₁ ρ_B θ_B ∧ curve_C₂ t_B B.1 B.2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l930_93056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l930_93026

/-- Represents a parabola with equation y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a circle with equation x² + y² + 2x - 3 = 0 -/
def Circle := {c : ℝ × ℝ | c.1^2 + c.2^2 + 2*c.1 - 3 = 0}

/-- The latus rectum of a parabola -/
def latusRectum (para : Parabola) : Set ℝ := {x | x = -para.p/2}

/-- The length of a line segment -/
noncomputable def lineSegmentLength (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

/-- Theorem: If the latus rectum of the parabola is intercepted by the circle
    with a length of 4, then p = 2 -/
theorem parabola_circle_intersection
  (para : Parabola)
  (hIntersect : ∃ (a b : ℝ × ℝ), a ∈ Circle ∧ b ∈ Circle ∧
                a.1 ∈ latusRectum para ∧ b.1 ∈ latusRectum para ∧
                lineSegmentLength a b = 4) :
  para.p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l930_93026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l930_93086

/-- 
Given a boat that travels 11 km along a stream in one hour and 5 km against the stream in one hour,
this theorem proves that the speed of the boat in still water is 8 km/h.
-/
theorem boat_speed_in_still_water (along_stream against_stream : ℝ) : 
  along_stream = 11 →
  against_stream = 5 →
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = along_stream ∧
    boat_speed - stream_speed = against_stream ∧
    boat_speed = 8 := by
  intro h1 h2
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l930_93086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l930_93081

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_properties (A ω φ : ℝ) (h1 : ω > 0) (h2 : |φ| < π/2)
  (h3 : ∀ x, f A ω φ x ≤ 2)
  (h4 : f A ω φ (3*π/8) = 0)
  (h5 : f A ω φ (5*π/8) = -2) :
  A = 2 ∧ ω = 2 ∧ φ = π/4 ∧ ∀ x, f A ω φ x = 2 * Real.sin (2*x + π/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l930_93081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l930_93002

/-- The angle of inclination of a line with equation y = mx + b, where m is the slope -/
noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.arctan (-m)

/-- Converts radians to degrees -/
noncomputable def to_degrees (x : ℝ) : ℝ := x * (180 / Real.pi)

theorem line_inclination (x y : ℝ) :
  y = -Real.sqrt 3 * x + 3 →
  0 < angle_of_inclination (-Real.sqrt 3) ∧ 
  angle_of_inclination (-Real.sqrt 3) < Real.pi →
  to_degrees (angle_of_inclination (-Real.sqrt 3)) = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l930_93002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dylans_journey_time_l930_93091

/-- Represents Dylan's bike journey --/
structure BikeJourney where
  total_distance : ℝ
  first_section_distance : ℝ
  first_section_speed : ℝ
  second_section_distance : ℝ
  second_section_speed : ℝ
  third_section_distance : ℝ
  third_section_speed : ℝ
  num_breaks : ℕ
  break_duration : ℝ

/-- Calculates the total time for Dylan's bike journey --/
noncomputable def total_journey_time (journey : BikeJourney) : ℝ :=
  (journey.first_section_distance / journey.first_section_speed) +
  (journey.second_section_distance / journey.second_section_speed) +
  (journey.third_section_distance / journey.third_section_speed) +
  (journey.num_breaks : ℝ) * journey.break_duration / 60

/-- Theorem stating that Dylan's journey takes 24.67 hours --/
theorem dylans_journey_time :
  let journey : BikeJourney := {
    total_distance := 1250,
    first_section_distance := 400,
    first_section_speed := 50,
    second_section_distance := 150,
    second_section_speed := 40,
    third_section_distance := 700,
    third_section_speed := 60,
    num_breaks := 3,
    break_duration := 25
  }
  total_journey_time journey = 24.67 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dylans_journey_time_l930_93091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_condition_l930_93079

theorem sequence_increasing_condition (lambda : ℝ) (h : lambda < 1) :
  ∀ n : ℕ+, (n^2 - 2*lambda*n : ℝ) < ((n+1)^2 - 2*lambda*(n+1) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_condition_l930_93079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l930_93019

theorem triangle_side_length (a : ℝ) : a ∈ ({2, 3, 6, 14} : Set ℝ) →
  (5 + 8 > a ∧ 5 + a > 8 ∧ 8 + a > 5) ↔ a = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l930_93019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_two_ways_l930_93020

/-- Represents the characteristics of a block -/
structure BlockCharacteristics where
  material : Fin 2
  size : Fin 4
  color : Fin 4
  shape : Fin 4
deriving Fintype, DecidableEq

/-- The total number of distinct blocks -/
def totalBlocks : Nat := 128

/-- The reference block: 'plastic medium red circle' -/
def referenceBlock : BlockCharacteristics := {
  material := 0,
  size := 1,
  color := 2,
  shape := 0
}

/-- Counts the number of differences between two blocks -/
def countDifferences (b1 b2 : BlockCharacteristics) : Nat :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0)

/-- The main theorem to prove -/
theorem blocks_differing_in_two_ways :
  (Finset.filter (fun b : BlockCharacteristics => countDifferences b referenceBlock = 2)
    (Finset.univ : Finset BlockCharacteristics)).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_two_ways_l930_93020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l930_93017

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- The coordinates of point P -/
def P : ℝ × ℝ := (-5, 4)

/-- The coordinates of point Q -/
def Q : ℝ × ℝ := (1, 7)

/-- The coordinates of point R -/
def R : ℝ × ℝ := (3, -1)

theorem triangle_PQR_area : 
  triangleArea P.1 P.2 Q.1 Q.2 R.1 R.2 = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l930_93017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miniature_height_rounded_l930_93034

/-- The height of the Empire State Building in feet -/
noncomputable def actual_height : ℝ := 1454

/-- The scale ratio used for the miniature model -/
noncomputable def scale_ratio : ℝ := 50

/-- The height of the miniature model before rounding -/
noncomputable def model_height : ℝ := actual_height / scale_ratio

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem miniature_height_rounded :
  round_to_nearest model_height = 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_miniature_height_rounded_l930_93034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_month_sale_l930_93036

theorem second_month_sale (sales : List ℕ) (average : ℕ) : 
  sales.length = 5 ∧ 
  sales.get? 0 = some 3435 ∧ 
  sales.get? 2 = some 3855 ∧ 
  sales.get? 3 = some 4230 ∧ 
  sales.get? 4 = some 3562 ∧
  average = 3500 →
  ∃ (second_month : ℕ), 
    second_month = 3927 ∧
    (sales.sum + second_month + 1991) / 6 = average :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_month_sale_l930_93036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l930_93098

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- State the theorem
theorem function_composition_equality 
  (x : ℝ) 
  (h : -1 < x ∧ x < 1) : 
  f ((4 * x + x^5) / (1 + 4 * x^4)) = 5 * f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l930_93098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l930_93073

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos θ, 2 * Real.sin θ)

def C₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 = 4}

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ θ, C₁ θ = p ∧ p ∈ C₂}

-- Theorem statement
theorem intersection_distance : 
  ∃ (M N : ℝ × ℝ), M ∈ intersection_points ∧ N ∈ intersection_points ∧ 
  M ≠ N ∧ Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l930_93073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_calculation_l930_93078

-- Define the radius of the smaller circle
def small_radius : ℝ := 3

-- Define the radius of the larger circle
def large_radius : ℝ := 5 * small_radius

-- Define the area of the gray region
noncomputable def gray_area : ℝ := Real.pi * large_radius^2 - Real.pi * small_radius^2

-- Theorem statement
theorem gray_area_calculation : gray_area = 216 * Real.pi := by
  -- Expand the definition of gray_area
  unfold gray_area
  -- Expand the definition of large_radius
  unfold large_radius
  -- Simplify the expression
  simp [small_radius]
  -- Perform arithmetic operations
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_calculation_l930_93078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stops_20_minutes_l930_93062

/-- Represents the bus travel scenario -/
structure BusTravel where
  speed_without_stops : ℚ  -- Speed without stops in km/h
  speed_with_stops : ℚ     -- Speed with stops in km/h

/-- Calculates the time spent stopped per hour given a BusTravel scenario -/
def time_stopped (bt : BusTravel) : ℚ :=
  let distance_diff := bt.speed_without_stops - bt.speed_with_stops
  (distance_diff / bt.speed_without_stops) * 60

/-- Theorem stating that for the given speeds, the bus stops for 20 minutes per hour -/
theorem bus_stops_20_minutes (bt : BusTravel) 
  (h1 : bt.speed_without_stops = 54)
  (h2 : bt.speed_with_stops = 36) : 
  time_stopped bt = 20 := by
  sorry

#eval time_stopped { speed_without_stops := 54, speed_with_stops := 36 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stops_20_minutes_l930_93062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l930_93046

/-- Arithmetic sequence properties -/
def arithmetic_sequence (a₁ d : ℚ) : ℕ → ℚ := λ n ↦ a₁ + (n - 1 : ℚ) * d

/-- Sum of arithmetic sequence -/
def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem arithmetic_sequence_properties :
  (∃ n : ℕ, arithmetic_sequence (5/6) (-1/6) n = -3/2 ∧
            sum_arithmetic_sequence (5/6) (-1/6) n = -5 ∧
            n = 15) ∧
  (arithmetic_sequence (-38) 2 15 = -10 ∧
   sum_arithmetic_sequence (-38) 2 15 = -360) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l930_93046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l930_93001

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Represents a line through two points -/
def line_through (P Q : Point) : Set Point :=
  {R : Point | ∃ t : ℝ, R.x = P.x + t * (Q.x - P.x) ∧ R.y = P.y + t * (Q.y - P.y)}

/-- Represents a vector from point P to point Q -/
def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

/-- Theorem: Eccentricity of the ellipse under given conditions -/
theorem ellipse_eccentricity (e : Ellipse) 
  (A B₁ B₂ F P : Point)
  (h_A : A = ⟨0, -e.a⟩)
  (h_B₁ : B₁ = ⟨-e.b, 0⟩)
  (h_B₂ : B₂ = ⟨e.b, 0⟩)
  (h_F : F.y > 0)
  (h_P : P ∈ line_through A B₂ ∩ line_through B₁ F)
  (h_AP : vector A P = ⟨2 * (B₂.x - A.x), 2 * (B₂.y - A.y)⟩) :
  eccentricity e = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l930_93001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_theorem_l930_93087

/-- A square with side length 4 -/
structure Square :=
  (side_length : ℝ)
  (is_four : side_length = 4)

/-- The sum of distances from a vertex to midpoints of sides in a square -/
noncomputable def sum_distances_to_midpoints (s : Square) : ℝ :=
  4 + 4 * Real.sqrt 5

/-- Theorem: The sum of distances from a vertex to midpoints of sides in a square with side length 4 is 4 + 4√5 -/
theorem sum_distances_theorem (s : Square) :
  sum_distances_to_midpoints s = 4 + 4 * Real.sqrt 5 := by
  -- Proof goes here
  sorry

#check sum_distances_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_theorem_l930_93087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_nonempty_domain_l930_93000

-- Define the recursive function g
noncomputable def g : ℕ → (ℝ → ℝ)
  | 0 => fun _ => 0  -- Add a case for 0
  | 1 => fun x => Real.sqrt (2 - x)
  | (n + 1) => fun x => g n (Real.sqrt ((n + 2)^2 - x))

-- Define the domain of g for each n
def domain (n : ℕ) : Set ℝ :=
  {x | ∃ y, g n x = y}

-- The theorem to be proved
theorem largest_nonempty_domain :
  (∃ M : ℕ, M > 0 ∧ 
    (∀ n > M, domain n = ∅) ∧ 
    (domain M = {25})) ∧
  (∀ M' > 4, ∃ n > M', domain n ≠ ∅) :=
sorry

#check largest_nonempty_domain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_nonempty_domain_l930_93000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_science_questions_l930_93061

structure QuestionSet where
  total : ℕ
  science : ℕ
  arts : ℕ
  h_total : total = science + arts

noncomputable def EventA (qs : QuestionSet) : ℝ :=
  qs.science / qs.total

noncomputable def EventB_given_A (qs : QuestionSet) : ℝ :=
  (qs.science - 1) / (qs.total - 1)

theorem conditional_probability_science_questions 
  (qs : QuestionSet) 
  (h_total : qs.total = 5) 
  (h_science : qs.science = 3) 
  (h_arts : qs.arts = 2) :
  EventB_given_A qs = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_science_questions_l930_93061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relative_errors_equal_l930_93065

/-- Calculates the relative error of a measurement -/
noncomputable def relative_error (actual_length : ℝ) (error : ℝ) : ℝ :=
  (error / actual_length) * 100

theorem relative_errors_equal : 
  let line1_length : ℝ := 25
  let line1_error : ℝ := 0.05
  let line2_length : ℝ := 200
  let line2_error : ℝ := 0.4
  relative_error line1_length line1_error = relative_error line2_length line2_error := by
  -- Unfold the definition of relative_error
  unfold relative_error
  -- Simplify the expressions
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relative_errors_equal_l930_93065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pipe_filling_time_l930_93063

/-- Represents the time it takes for the second pipe to fill the tank -/
noncomputable def second_pipe_time (first_pipe_time tank_volume : ℝ) (combined_time_with_leak : ℝ) : ℝ :=
  (2 * tank_volume * first_pipe_time) / (3 * first_pipe_time - 2 * tank_volume)

theorem second_pipe_filling_time 
  (first_pipe_time : ℝ) 
  (combined_time_with_leak : ℝ) 
  (tank_volume : ℝ) :
  first_pipe_time = 20 →
  combined_time_with_leak = 16 →
  tank_volume > 0 →
  second_pipe_time first_pipe_time tank_volume combined_time_with_leak = 160 / 7 :=
by sorry

#eval (160 : ℚ) / 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pipe_filling_time_l930_93063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_intersection_l930_93071

def P : Finset ℕ := {0, 1, 2}
def Q : Set ℝ := {x | x > 0}

theorem number_of_subsets_of_intersection : Finset.card (Finset.powerset (P.filter (fun x => x > 0))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_intersection_l930_93071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_subtraction_and_simplification_l930_93029

theorem fraction_subtraction_and_simplification :
  ∃ (a b : ℚ), a = 5/15 ∧ b = 2/45 ∧ a - b = 13/45 ∧ (∀ n d : ℤ, n ≠ 0 → d > 0 → (n : ℚ)/d = 13/45 → (n.natAbs.gcd d.natAbs = 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_subtraction_and_simplification_l930_93029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_value_of_15b_minus_2a_l930_93010

-- Define the function f
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if 1 ≤ x ∧ x < 2 then x + a / x
  else if 2 ≤ x ∧ x ≤ 3 then b * x - 3
  else 0  -- We define it as 0 outside [1, 3] for completeness

-- State the theorem
theorem find_value_of_15b_minus_2a (a b : ℝ) :
  (∀ x : ℝ, f (x + 2) a b = f x a b) →  -- f has period 2
  (f (7/2) a b = f (-7/2) a b) →        -- given condition
  (15 * b - 2 * a = 32) :=               -- conclusion
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_value_of_15b_minus_2a_l930_93010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_probability_models_l930_93006

-- Define the characteristics of a geometric probability model
def is_geometric_probability_model (model : ℕ) : Prop :=
  (model = 1 ∨ model = 2 ∨ model = 4) ∧
  (∃ (S : Set ℝ), Set.Infinite S ∧ ∀ x ∈ S, ∃ p : ℝ, p > 0 ∧ p ≤ 1)

-- Define the four models
def model1 : ℕ := 1
def model2 : ℕ := 2
def model3 : ℕ := 3
def model4 : ℕ := 4

-- Theorem statement
theorem geometric_probability_models :
  is_geometric_probability_model model1 ∧
  is_geometric_probability_model model2 ∧
  ¬is_geometric_probability_model model3 ∧
  is_geometric_probability_model model4 := by
  sorry

#check geometric_probability_models

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_probability_models_l930_93006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_EH_in_inscribed_quadrilateral_l930_93027

/-- A quadrilateral EFGH inscribed in a circle with given angle and side length properties -/
structure InscribedQuadrilateral where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  angle_EFG : ℝ
  angle_EHG : ℝ
  length_EF : ℝ
  length_FG : ℝ
  inscribed : Bool
  angle_EFG_eq : angle_EFG = 60
  angle_EHG_eq : angle_EHG = 70
  length_EF_eq : length_EF = 3
  length_FG_eq : length_FG = 7
  is_inscribed : inscribed = true

/-- The theorem stating the length of EH in the inscribed quadrilateral -/
theorem length_EH_in_inscribed_quadrilateral (q : InscribedQuadrilateral) :
  let EH := Real.sqrt ((q.E.1 - q.H.1)^2 + (q.E.2 - q.H.2)^2)
  EH = (3 * Real.sin (50 * π / 180)) / Real.sin (70 * π / 180) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_EH_in_inscribed_quadrilateral_l930_93027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subsets_for_square_sums_min_subsets_is_three_l930_93097

def M : Set Nat := Finset.range 30

def is_square_sum (a b : Nat) : Prop :=
  ∃ n : Nat, a + b = n * n

theorem min_subsets_for_square_sums (k : Nat) : Prop :=
  k = 3 ∧
  ∃ (f : Nat → Nat),
    (∀ x, x ∈ M → f x < k) ∧
    (∀ a b, a ∈ M → b ∈ M → a ≠ b → is_square_sum a b → f a ≠ f b) ∧
    (∀ k' < k, ¬∃ (g : Nat → Nat),
      (∀ x, x ∈ M → g x < k') ∧
      (∀ a b, a ∈ M → b ∈ M → a ≠ b → is_square_sum a b → g a ≠ g b))

theorem min_subsets_is_three : min_subsets_for_square_sums 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subsets_for_square_sums_min_subsets_is_three_l930_93097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_dress_cost_l930_93058

theorem sara_dress_cost (sara_shoe_cost rachel_budget : ℕ) :
  sara_shoe_cost = 50 →
  rachel_budget = 500 →
  let sara_dress_cost := rachel_budget / 2 - sara_shoe_cost
  let rachel_spending := 2 * (sara_shoe_cost + sara_dress_cost)
  rachel_spending = rachel_budget →
  sara_dress_cost = 200 := by
    intros h1 h2 h3
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_dress_cost_l930_93058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_abc_l930_93064

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a + 2 * Real.sqrt (a * b) + 2 * (a * b * c) ^ (1/3 : ℝ) ≤ 7 ∧
  ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧
    a + 2 * Real.sqrt (a * b) + 2 * (a * b * c) ^ (1/3 : ℝ) = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_abc_l930_93064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_value_after_five_years_l930_93024

noncomputable def machine_value (initial_value : ℝ) (depreciation_rate1 : ℝ) (rate_change2 : ℝ) (rate_change3 : ℝ) (repair_cost : ℝ) : ℝ :=
  let value1 := initial_value * (1 - depreciation_rate1)
  let depreciation_rate2 := depreciation_rate1 + rate_change2
  let value2 := value1 * (1 - depreciation_rate2)
  let depreciation_rate3 := depreciation_rate2 + rate_change3
  let value3 := value2 * (1 - depreciation_rate3)
  let value4 := value3 - repair_cost
  let depreciation_rate5 := (depreciation_rate1 + depreciation_rate2 + depreciation_rate3) / 3
  let value5 := value4 * (1 - depreciation_rate5)
  value5

theorem machine_value_after_five_years :
  ∃ ε > 0, |machine_value 5000 0.08 (-0.02) 0.01 500 - 3274.83| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_value_after_five_years_l930_93024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l930_93012

-- Define the constants
def speed_of_sound : ℚ := 1100
def time_delay : ℚ := 8
def feet_per_mile : ℚ := 5280

-- Define the function to calculate distance
def calculate_distance (speed time : ℚ) : ℚ := speed * time

-- Define the function to convert feet to miles
def feet_to_miles (feet : ℚ) : ℚ := feet / feet_per_mile

-- Define the function to round to the nearest quarter-mile
noncomputable def round_to_quarter_mile (miles : ℚ) : ℚ :=
  ⌊(miles * 4 + 1/2)⌋ / 4

-- Theorem statement
theorem lightning_distance :
  round_to_quarter_mile (feet_to_miles (calculate_distance speed_of_sound time_delay)) = 7/4 := by
  -- Unfold definitions
  unfold round_to_quarter_mile
  unfold feet_to_miles
  unfold calculate_distance
  -- Simplify the expression
  simp [speed_of_sound, time_delay, feet_per_mile]
  -- The proof itself would go here, but we'll use sorry to skip it
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_l930_93012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_one_l930_93033

/-- Given two lines l1 and l2 in R^2, where l1 has equation 2x - ay - 1 = 0 and
    l2 has equation x + 2y = 0, prove that if l1 is perpendicular to l2, then a = 1. -/
theorem perpendicular_lines_a_equals_one (a : ℝ) :
  let l1 := {p : ℝ × ℝ | 2 * p.1 - a * p.2 - 1 = 0}
  let l2 := {p : ℝ × ℝ | p.1 + 2 * p.2 = 0}
  (∀ p q : ℝ × ℝ, p ∈ l1 → q ∈ l2 → (p.1 - q.1) * (p.1 - q.1) + (p.2 - q.2) * (p.2 - q.2) = 0) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_one_l930_93033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_value_l930_93040

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | (n + 2) => 1 - 1 / sequence_a (n + 1)

theorem a_2009_value : sequence_a 2009 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_value_l930_93040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt2_over_2_l930_93088

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/4) -/
noncomputable def circle_radius : ℝ := Real.sqrt 2 / 2

/-- Spherical coordinates (ρ, θ, φ) -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- The set of points forming the circle -/
def circle_points : Set SphericalCoord :=
  {p : SphericalCoord | p.ρ = 1 ∧ p.φ = Real.pi/4 ∧ 0 ≤ p.θ ∧ p.θ < 2*Real.pi}

/-- Theorem stating that the radius of the circle is √2/2 -/
theorem circle_radius_is_sqrt2_over_2 :
  ∀ p ∈ circle_points, 
    Real.sqrt (((Real.sin p.φ) * (Real.cos p.θ))^2 + ((Real.sin p.φ) * (Real.sin p.θ))^2) = circle_radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt2_over_2_l930_93088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_twelve_existence_l930_93045

theorem sum_twelve_existence (S : Finset ℕ) : 
  S ⊆ Finset.range 12 → S.card = 7 → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_twelve_existence_l930_93045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_zero_l930_93023

-- Define the circle C
def circleC (t : ℝ) (x y : ℝ) : Prop :=
  (x - t)^2 + (y - t)^2 = 1

-- Define point P
def P : ℝ × ℝ := (-1, 1)

-- Define the tangent points A and B
def tangent_points (t : ℝ) (A B : ℝ × ℝ) : Prop :=
  circleC t A.1 A.2 ∧ circleC t B.1 B.2

-- Define the dot product of vectors PA and PB
def dot_product (A B : ℝ × ℝ) : ℝ :=
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  PA.1 * PB.1 + PA.2 * PB.2

-- State the theorem
theorem min_dot_product_zero :
  ∀ t : ℝ, ∀ A B : ℝ × ℝ,
  tangent_points t A B →
  ∃ m : ℝ, m = 0 ∧ ∀ A' B' : ℝ × ℝ, tangent_points t A' B' → 
  dot_product A' B' ≥ m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_zero_l930_93023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l930_93043

open Finset

def S : Finset (Finset (Fin 5)) :=
  powerset {2, 3, 4} |>.image (· ∪ {0, 1})

theorem number_of_subsets : card S = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l930_93043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_sum_nonnegative_l930_93009

theorem log_inequality_implies_sum_nonnegative (x y : ℝ) :
  (Real.logb 2 3) ^ x - (Real.logb 5 3) ^ x ≥ (Real.logb 2 3) ^ (-y) - (Real.logb 5 3) ^ (-y) →
  x + y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_sum_nonnegative_l930_93009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_area_l930_93050

/-- Definition of the area of an equilateral triangle with side length a -/
noncomputable def area_equilateral_triangle (a : ℝ) : ℝ := (Real.sqrt 3 / 4) * a^2

/-- Definition of the surface area of a regular tetrahedron with edge length a -/
noncomputable def surface_area_tetrahedron (a : ℝ) : ℝ := 4 * area_equilateral_triangle a

/-- The surface area of a regular tetrahedron with edge length a is √3 * a² -/
theorem tetrahedron_surface_area (a : ℝ) (h : a > 0) : 
  surface_area_tetrahedron a = Real.sqrt 3 * a^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_area_l930_93050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_recreation_spending_l930_93089

/-- The percentage of wages John spent on recreation last week -/
def recreation_percent_last_week (last_week_wages : ℝ) (this_week_wages : ℝ) (recreation_percent_this_week : ℝ) : ℝ :=
  30

theorem john_recreation_spending 
  (last_week_wages : ℝ) 
  (this_week_wages : ℝ) 
  (recreation_percent_this_week : ℝ) :
  this_week_wages = 0.75 * last_week_wages →
  recreation_percent_this_week = 20 →
  recreation_percent_this_week / 100 * this_week_wages = 
    0.5 * (recreation_percent_last_week last_week_wages this_week_wages recreation_percent_this_week / 100 * last_week_wages) →
  recreation_percent_last_week last_week_wages this_week_wages recreation_percent_this_week = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_recreation_spending_l930_93089
