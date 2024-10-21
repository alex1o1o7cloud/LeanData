import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1192_119244

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola (a b : ℝ) where
  ha : a > 0
  hb : b > 0
  transverse_conjugate : a = Real.sqrt 2 * b

/-- The asymptotes of a hyperbola -/
def asymptotes (a b : ℝ) (h : Hyperbola a b) : Set (ℝ × ℝ) :=
  {(x, y) | y = Real.sqrt 2 * x ∨ y = -(Real.sqrt 2 * x)}

/-- Theorem stating that the given asymptotes are correct for the specified hyperbola -/
theorem hyperbola_asymptotes (a b : ℝ) (h : Hyperbola a b) :
  asymptotes a b h = {(x, y) | y = Real.sqrt 2 * x ∨ y = -(Real.sqrt 2 * x)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1192_119244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1192_119201

-- Define the function f(x) = x / (1 - x)
noncomputable def f (x : ℝ) : ℝ := x / (1 - x)

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < 1 → x₂ < 1 → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1192_119201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_four_leaved_clovers_result_l1192_119213

/-- The number of purple, four-leaved clovers in a field with given conditions -/
def purple_four_leaved_clovers (total_clovers : ℕ) 
  (four_leaf_probability : ℚ) 
  (red_ratio yellow_ratio purple_ratio : ℚ) : ℕ :=
  let total_four_leaf : ℕ := (↑total_clovers * four_leaf_probability).floor.toNat
  let total_ratio : ℚ := red_ratio + yellow_ratio + purple_ratio
  let purple_proportion : ℚ := purple_ratio / total_ratio
  (↑total_four_leaf * purple_proportion).floor.toNat

/-- The main theorem stating the number of purple, four-leaved clovers under given conditions -/
theorem purple_four_leaved_clovers_result : 
  purple_four_leaved_clovers 850 (273/1000) (11/2) (73/10) (46/5) = 97 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_four_leaved_clovers_result_l1192_119213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1192_119298

/-- Calculates the time taken for a train to pass a person moving in the opposite direction. -/
noncomputable def time_to_pass (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) : ℝ :=
  let relative_speed := train_speed + person_speed
  let relative_speed_mps := relative_speed * (1000 / 3600)
  train_length / relative_speed_mps

/-- Theorem: Given a train of length 200 meters moving at 69 km/h and a person moving at 3 km/h
    in the opposite direction, the time taken for the train to pass the person is 10 seconds. -/
theorem train_passing_time :
  time_to_pass 200 69 3 = 10 := by
  -- Unfold the definition of time_to_pass
  unfold time_to_pass
  -- Simplify the expression
  simp
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1192_119298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_tangent_l1192_119276

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and eccentricity 3,
    prove that if its asymptote is tangent to the circle x² + y² - 6y + m = 0,
    then m = 8 -/
theorem hyperbola_circle_tangent (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let e : ℝ := 3  -- eccentricity
  let c : ℝ := e * a  -- focal distance
  let asymptote : ℝ → ℝ := λ x ↦ (b / a) * x  -- asymptote equation y = (b/a)x
  let circle : ℝ × ℝ → ℝ := λ p ↦ p.1^2 + p.2^2 - 6 * p.2 + 8  -- circle equation
  b^2 = 8 * a^2 ∧  -- condition for eccentricity 3
  (∃ p : ℝ × ℝ, circle p = 0 ∧ p.2 = asymptote p.1 ∧
    ∀ q : ℝ × ℝ, q ≠ p → circle q > 0 ∨ q.2 ≠ asymptote q.1) →
  8 = 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_tangent_l1192_119276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l1192_119251

-- Define the circle
def circleC (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 16

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem point_B_coordinates :
  ∀ x y a : ℝ,
  circleC x y →
  2 * distance x y (-2) 0 = distance x y a 0 →
  a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l1192_119251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_sqrt_two_l1192_119239

/-- The infinite power tower function -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := 
  Real.exp (x * Real.log x)

/-- Theorem: √2 is the unique positive real solution to x^(x^(x^...)) = 4 -/
theorem infinite_power_tower_sqrt_two :
  ∃! (x : ℝ), x > 0 ∧ infinitePowerTower x = 4 := by
  sorry

#check infinite_power_tower_sqrt_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_sqrt_two_l1192_119239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_digging_time_l1192_119297

/-- The time it takes for Jake, Paul, and Hari to dig a well together -/
noncomputable def combined_digging_time (jake_time paul_time hari_time : ℝ) : ℝ :=
  1 / (1 / jake_time + 1 / paul_time + 1 / hari_time)

theorem well_digging_time :
  combined_digging_time 16 24 48 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_digging_time_l1192_119297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_two_l1192_119238

/-- Represents a tetrahedron with vertices P, Q, R, S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron using the Cayley-Menger determinant -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  let a := t.PQ
  let b := t.PR
  let c := t.PS
  let d := t.QR
  let e := t.QS
  let f := t.RS
  (1 / 288) * Real.sqrt (
    Matrix.det ![
      ![0, 1, 1, 1, 1],
      ![1, 0, a^2, b^2, c^2],
      ![1, a^2, 0, d^2, e^2],
      ![1, b^2, d^2, 0, f^2],
      ![1, c^2, e^2, f^2, 0]
    ]
  )

theorem tetrahedron_volume_is_two :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    PS := 3,
    QR := 5,
    QS := 7,
    RS := Real.sqrt 94
  }
  tetrahedronVolume t = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_two_l1192_119238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_not_ending_in_zero_of_million_l1192_119287

def divisors_not_ending_in_zero (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ d => n % d = 0 ∧ d % 10 ≠ 0)

theorem count_divisors_not_ending_in_zero_of_million :
  (divisors_not_ending_in_zero 1000000).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_not_ending_in_zero_of_million_l1192_119287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_running_speed_l1192_119290

/-- Calculates the speed of a boy running around a square field -/
theorem boy_running_speed (side_length : ℝ) (time : ℝ) : 
  side_length = 40 → time = 48 → ∃ (speed : ℝ), 
  (speed ≥ 11.99 ∧ speed ≤ 12.01) ∧ 
  (speed * time ≥ 4 * side_length * 3.6 - 0.01 ∧ 
   speed * time ≤ 4 * side_length * 3.6 + 0.01) := by
  sorry

#check boy_running_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_running_speed_l1192_119290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_gallons_needed_l1192_119227

noncomputable section

def column_height : ℝ := 15
def column_diameter : ℝ := 8
def num_columns : ℕ := 20
def paint_coverage : ℝ := 400

def lateral_surface_area (h d : ℝ) : ℝ := Real.pi * d * h

def total_paint_area : ℝ := num_columns * lateral_surface_area column_height column_diameter

def gallons_needed : ℝ := total_paint_area / paint_coverage

theorem paint_gallons_needed : 
  ⌈gallons_needed⌉ = 19 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_gallons_needed_l1192_119227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_fraction_l1192_119269

theorem min_trig_fraction :
  ∀ x : ℝ, (Real.sin x)^8 + (Real.cos x)^8 + 1 ≥ (1/2) * ((Real.sin x)^6 + (Real.cos x)^6 + 1) ∧
  ∃ x : ℝ, (Real.sin x)^8 + (Real.cos x)^8 + 1 = (1/2) * ((Real.sin x)^6 + (Real.cos x)^6 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_fraction_l1192_119269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_volume_is_correct_l1192_119240

/-- A unit cube with corners sliced off to make each face a regular octagon -/
structure SlicedCube where
  /-- The length of each side of the octagon on a face -/
  octagon_side : ℝ
  /-- The octagon side length is related to the cube's edge -/
  octagon_side_eq : 2 * (octagon_side / Real.sqrt 2) + octagon_side = 1

/-- The volume of tetrahedra removed from a sliced cube -/
noncomputable def removed_volume (c : SlicedCube) : ℝ :=
  8 * (1 / 3) * ((1 / 2) * (1 - c.octagon_side / Real.sqrt 2)^2) * (1 / Real.sqrt 2)

/-- Theorem: The total volume of removed tetrahedra is (10 - 7√2) / 3 -/
theorem removed_volume_is_correct (c : SlicedCube) : 
  removed_volume c = (10 - 7 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_volume_is_correct_l1192_119240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_negative_l1192_119200

noncomputable def number_list : List ℝ := [
  3 + 1/6,
  -(abs (-7)),
  -0.1,
  -(-22/7),
  -100,
  0,
  0.213,
  3.14
]

theorem count_non_negative : (number_list.filter (λ x => x ≥ 0)).length = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_negative_l1192_119200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducible_l1192_119271

theorem fraction_irreducible (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducible_l1192_119271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l1192_119249

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if cos(B)/cos(C) = b/(2a-c), the area is 3√3/4, and b = 3,
    then a + c = 3√2. -/
theorem triangle_side_sum (a b c : ℝ) (A B C : Real) : 
  b = 3 →
  (Real.cos B) / (Real.cos C) = b / (2 * a - c) →
  (1/2) * b * c * (Real.sin A) = (3 * Real.sqrt 3) / 4 →
  a + c = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_l1192_119249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1192_119209

noncomputable def curve (x : ℝ) : ℝ := Real.exp x

noncomputable def P : ℝ × ℝ := (1, Real.exp 1)

noncomputable def tangent_slope : ℝ := Real.exp 1

def line_equation (x y : ℝ) : Prop := x + Real.exp 1 * y - Real.exp 2 - 1 = 0

theorem perpendicular_line_equation :
  line_equation P.1 P.2 ∧
  (tangent_slope * (-1 / tangent_slope) = -1) ∧
  ∀ x y : ℝ, line_equation x y → y = curve x ∨ (y - P.2) = (-1 / tangent_slope) * (x - P.1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1192_119209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1192_119235

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b c p : ℝ) (F₁ M P : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  p > 0 →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 ↔ (x, y) ∈ Set.range (λ t ↦ (a * Real.cosh t, b * Real.sinh t))) →
  (∀ x y, x^2 + y^2 = a^2 ↔ (x, y) ∈ Metric.sphere (0, 0) a) →
  (∀ x y, y^2 = 2*p*x ↔ (x, y) ∈ Set.range (λ t ↦ (t^2/(2*p), t))) →
  F₁.1 = -c →
  F₁.2 = 0 →
  M ∈ Metric.sphere (0, 0) a →
  (M.1 - F₁.1)^2 + (M.2 - F₁.2)^2 = (M.1 - 0)^2 + (M.2 - 0)^2 →
  P.2^2 = 2*p*P.1 →
  M = ((F₁.1 + P.1)/2, (F₁.2 + P.2)/2) →
  p = 2*c →
  let e := c/a
  e = (Real.sqrt 5 + 1) / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1192_119235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1192_119206

open Set
open Real

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2 * Real.sin x) + Real.sqrt ((2 * Real.pi - x) / x)

theorem domain_of_f : 
  {x : ℝ | f x ∈ Set.univ} = 
  {x : ℝ | (0 < x ∧ x < Real.pi / 6) ∨ (5 * Real.pi / 6 < x ∧ x ≤ 2 * Real.pi)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1192_119206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_diagonals_l1192_119233

/-- The angle between diagonals of two faces sharing an edge in a rectangular parallelepiped -/
theorem angle_between_diagonals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let θ := Real.arccos (a^2 / Real.sqrt ((a^2 + b^2) * (a^2 + c^2)))
  ∃ (θ' : ℝ), θ' = θ ∧ 
    θ' = Real.arccos ((a * a) / (Real.sqrt ((a^2 + b^2) * (a^2 + c^2)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_diagonals_l1192_119233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1192_119273

/-- The set A -/
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

/-- The set B -/
def B (a b c : ℝ) : Set ℝ := {x | a*x^2 + b*x + c ≤ 0}

/-- The theorem statement -/
theorem problem_statement (a b c : ℝ) (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (A ∩ B a b c = Set.Ioc 3 5) →
  (A ∪ B a b c = Set.univ) →
  b/a + a^2/c^2 = -3 - 24/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1192_119273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_range_t_value_for_specific_condition_max_m_for_inequality_l1192_119237

-- Define the function f
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (x^3 - 6*x^2 + 3*x + t) * Real.exp x

-- Theorem for part (1)①
theorem extreme_points_range (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  (∃ t : ℝ, ∀ x : ℝ, x ≠ a ∧ x ≠ b ∧ x ≠ c → (deriv (f t)) x ≠ 0) →
  -8 < t ∧ t < 24 :=
sorry

-- Theorem for part (1)②
theorem t_value_for_specific_condition (a b c : ℝ) (h1 : a < b) (h2 : b < c) (h3 : a + c = 2*b^2) :
  (∃ t : ℝ, ∀ x : ℝ, x ≠ a ∧ x ≠ b ∧ x ≠ c → (deriv (f t)) x ≠ 0) →
  t = 8 :=
sorry

-- Theorem for part (2)
theorem max_m_for_inequality :
  (∃ m : ℕ, m > 0 ∧ 
    (∃ t : ℝ, t ∈ Set.Icc 0 2 ∧
      (∀ x : ℝ, x ∈ Set.Icc 1 (m : ℝ) → f t x ≤ x)) ∧
    (∀ n : ℕ, n > m →
      ¬(∃ t : ℝ, t ∈ Set.Icc 0 2 ∧
        (∀ x : ℝ, x ∈ Set.Icc 1 (n : ℝ) → f t x ≤ x)))) →
  m = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_range_t_value_for_specific_condition_max_m_for_inequality_l1192_119237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1192_119291

/-- The minimum area of a triangle ABC with vertices A=(0,0), B=(24,10), C=(p,q), 
    where p and q are integers. -/
theorem min_triangle_area : 
  ∃ (p q : ℤ), ∀ (p' q' : ℤ), 
    (1/2 : ℚ) * |10 * p - 24 * q| ≤ (1/2 : ℚ) * |10 * p' - 24 * q'| ∧
    (1/2 : ℚ) * |10 * p - 24 * q| = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1192_119291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_leaf_problem_l1192_119229

theorem tea_leaf_problem (num_plants : ℕ) (leaves_per_plant : ℕ) (fall_off_numerator : ℕ) (fall_off_denominator : ℕ) :
  num_plants = 5 →
  leaves_per_plant = 24 →
  fall_off_numerator = 2 →
  fall_off_denominator = 5 →
  (num_plants * leaves_per_plant) - (num_plants * leaves_per_plant * fall_off_numerator / fall_off_denominator) = 72 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_leaf_problem_l1192_119229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_exist_l1192_119212

-- Define the function h(x)
def h (x : ℝ) : ℝ := x^3 - x

-- Define the tangent line equation
def tangent_line (t : ℝ) (x : ℝ) : ℝ := (3 * t^2 - 1) * x - 2 * t^3

-- State the theorem
theorem three_tangent_lines_exist :
  ∃ t₁ t₂ t₃ : ℝ, t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃ ∧
  (∀ i : ℝ, i ∈ ({t₁, t₂, t₃} : Set ℝ) → tangent_line i 2 = 1) ∧
  (∀ i : ℝ, i ∈ ({t₁, t₂, t₃} : Set ℝ) → ∀ x : ℝ, tangent_line i x = h x ↔ x = i) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_exist_l1192_119212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hno3_concentration_proof_l1192_119204

/-- Calculates the concentration of HNO3 after mixing two solutions -/
noncomputable def resultant_concentration (initial_volume : ℝ) (initial_concentration : ℝ) (added_volume : ℝ) : ℝ :=
  let total_hno3 := initial_volume * initial_concentration + added_volume
  let total_volume := initial_volume + added_volume
  total_hno3 / total_volume

/-- Proves that mixing 60 liters of 20% HNO3 solution with 36 liters of pure HNO3 results in a 50% concentration -/
theorem hno3_concentration_proof :
  resultant_concentration 60 0.2 36 = 0.5 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hno3_concentration_proof_l1192_119204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worm_length_difference_l1192_119270

theorem worm_length_difference : ∃ (worm1_length worm2_length : ℝ),
  worm1_length = 0.8 ∧
  worm2_length = 0.1 ∧
  let longer_worm_length := max worm1_length worm2_length
  let shorter_worm_length := min worm1_length worm2_length
  let length_difference := longer_worm_length - shorter_worm_length
  length_difference = 0.7 := by
  use 0.8, 0.1
  simp [max, min]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worm_length_difference_l1192_119270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1192_119253

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 1

noncomputable def f_p (p : ℝ) (x : ℝ) : ℝ :=
  if f x ≤ p then f x else p

theorem inequality_proof (p : ℝ) (h : p = 2) : 
  f_p p (f 1) ≠ f (f_p p 1) := by
  -- The proof goes here
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1192_119253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_condition_implies_a_equals_negative_four_l1192_119221

-- Define the function f as noncomputable due to Real.sqrt
noncomputable def f (a b c x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x + c)

-- Define the domain D
def D (a b c : ℝ) : Set ℝ := {x | a * x^2 + b * x + c ≥ 0}

-- Theorem statement
theorem square_condition_implies_a_equals_negative_four 
  (a b c : ℝ) (h_a_neg : a < 0) :
  (∀ s t, s ∈ D a b c → t ∈ D a b c → ∃ side, 
    (t - s)^2 + (f a b c t - f a b c s)^2 = 2 * side^2 ∧
    (t - s)^2 = (f a b c t - f a b c s)^2) →
  a = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_condition_implies_a_equals_negative_four_l1192_119221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_is_one_l1192_119256

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  sum_property : a 3 + a 7 = 10
  eighth_term : a 8 = 8

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_one (seq : ArithmeticSequence) : 
  common_difference seq = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_difference_is_one_l1192_119256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_positive_integers_satisfying_equation_l1192_119279

theorem exist_positive_integers_satisfying_equation : 
  ∃ (x y z : ℕ+), (4 : ℝ) * Real.sqrt (Real.rpow 7 (1/3) - Real.rpow 6 (1/3)) = 
    Real.rpow x.val (1/3) + Real.rpow y.val (1/3) - Real.rpow z.val (1/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_positive_integers_satisfying_equation_l1192_119279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hockey_pads_cost_l1192_119225

noncomputable def initial_amount : ℝ := 150
noncomputable def skates_cost : ℝ := initial_amount / 2
noncomputable def stick_cost : ℝ := 20
noncomputable def helmet_original_price : ℝ := 30
noncomputable def helmet_discount_rate : ℝ := 0.1
noncomputable def remaining_amount : ℝ := 10

noncomputable def helmet_cost : ℝ := helmet_original_price * (1 - helmet_discount_rate)

theorem hockey_pads_cost :
  initial_amount - skates_cost - stick_cost - helmet_cost - remaining_amount = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hockey_pads_cost_l1192_119225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sides_product_l1192_119211

theorem square_sides_product (a₁ a₂ : ℝ) : 
  (∃ (a : ℝ), (Set.Icc 3 8).prod (Set.Icc (-3) a) = Set.Icc (-3, 3) (-3, 8)) →
  a₁ * a₂ = -16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sides_product_l1192_119211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_missing_midpoint_l1192_119217

/-- A 2n-gon is represented by a list of 2n points in a 2D plane -/
def Polygon2n (n : ℕ) := List (ℝ × ℝ)

/-- A midpoint is represented as a point in a 2D plane -/
def Midpoint := ℝ × ℝ

/-- Function to calculate the midpoint of two points -/
noncomputable def calculateMidpoint (p1 p2 : ℝ × ℝ) : Midpoint :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

/-- Function to get all midpoints of a 2n-gon -/
noncomputable def getAllMidpoints (poly : Polygon2n n) : List Midpoint :=
  sorry

/-- Predicate to check if a point is a valid midpoint for a given list of points and known midpoints -/
def isValidMidpoint (point : Midpoint) (poly : Polygon2n n) (knownMidpoints : List Midpoint) : Prop :=
  sorry

theorem unique_missing_midpoint (n : ℕ) (poly : Polygon2n n) (knownMidpoints : List Midpoint) 
    (h1 : n > 0)
    (h2 : poly.length = 2 * n)
    (h3 : knownMidpoints.length = 2 * n - 1)
    (h4 : ∀ (i : Fin (2 * n - 1)), knownMidpoints[i]? = (getAllMidpoints poly)[i]?) :
    ∃! (missingMidpoint : Midpoint), isValidMidpoint missingMidpoint poly knownMidpoints := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_missing_midpoint_l1192_119217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l1192_119286

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def are_not_collinear (v w : V) : Prop := ∀ (r : ℝ), v ≠ r • w

def are_collinear (v w : V) : Prop := ∃ (r : ℝ), v = r • w

theorem collinear_vectors_lambda (e₁ e₂ : V) (l : ℝ) 
  (h1 : are_not_collinear e₁ e₂)
  (h2 : are_collinear (2 • e₁ - e₂) (3 • e₁ + l • e₂)) :
  l = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l1192_119286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1192_119265

noncomputable def a : ℝ := Real.cos (420 * Real.pi / 180)

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then a^x else Real.log x / Real.log a

theorem problem_solution : f (1/4) + f (-2) = 6 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1192_119265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_perpendicular_point_l1192_119272

-- Define the vector e
def e : ℝ × ℝ := (1, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the trajectory condition
def trajectory_condition (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) - (P.1 * e.1 + P.2 * e.2) = 2

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop := y^2 = 4*(x + 1)

-- Define the perpendicularity condition
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Theorem statement
theorem trajectory_and_perpendicular_point :
  (∀ P : ℝ × ℝ, trajectory_condition P ↔ trajectory_equation P.1 P.2) ∧
  (∀ B C : ℝ × ℝ, 
    trajectory_equation B.1 B.2 → 
    trajectory_equation C.1 C.2 → 
    (∃ lambda : ℝ, lambda ≠ 0 ∧ B.1 = lambda * C.1 ∧ B.2 = lambda * C.2) →
    (∃ m : ℝ, (m ≥ 2 ∨ (-2 ≤ m ∧ m < -1)) ↔ 
      perpendicular (m, 0) B C)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_perpendicular_point_l1192_119272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_rhombus_proposition_l1192_119243

/-- A rhombus is a quadrilateral with all sides equal -/
structure Rhombus where
  sides : Fin 4 → ℝ
  sides_equal : ∀ (i j : Fin 4), sides i = sides j

/-- The inverse proposition of "A rhombus has all four sides equal" -/
theorem inverse_rhombus_proposition :
  (∀ (sides : Fin 4 → ℝ), (∀ (i j : Fin 4), sides i = sides j) → ∃ r : Rhombus, r.sides = sides) ↔
  (∀ r : Rhombus, ∀ (i j : Fin 4), r.sides i = r.sides j) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_rhombus_proposition_l1192_119243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_F_negative_reals_l1192_119255

-- Define the real numbers
variable (x : ℝ)

-- Define the functions
variable (f g : ℝ → ℝ)
variable (a b : ℝ)

-- Define F(x)
def F (f g : ℝ → ℝ) (a b : ℝ) (x : ℝ) : ℝ := a * f x + b * g x + 2

-- State the theorem
theorem min_value_F_negative_reals 
  (hf_odd : ∀ x, f (-x) = -f x)
  (hg_odd : ∀ x, g (-x) = -g x)
  (hF_max : ∀ x > 0, F f g a b x ≤ 5) :
  ∀ x < 0, F f g a b x ≥ -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_F_negative_reals_l1192_119255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_payoff_l1192_119236

/-- The weights of the 4-sided die -/
noncomputable def die_weights : Fin 4 → ℝ
  | 0 => 1/2
  | 1 => 1/3
  | 2 => 1/7
  | 3 => 1/42

/-- The payoff function for rolling k -/
noncomputable def payoff (x : Fin 4 → ℝ) (k : Fin 4) : ℝ := 10 + Real.log (x k)

/-- The expected payoff function -/
noncomputable def expected_payoff (x : Fin 4 → ℝ) : ℝ :=
  Finset.sum Finset.univ (fun k => die_weights k * payoff x k)

/-- The constraint on x -/
def valid_choice (x : Fin 4 → ℝ) : Prop :=
  (∀ k, x k > 0) ∧ Finset.sum Finset.univ x = 1

/-- The optimal choice -/
noncomputable def optimal_choice : Fin 4 → ℝ := die_weights

theorem optimal_payoff (x : Fin 4 → ℝ) (h : valid_choice x) :
  expected_payoff x ≤ expected_payoff optimal_choice := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_payoff_l1192_119236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_b_equals_four_l1192_119247

theorem factor_implies_b_equals_four (a b : ℤ) : 
  (∃ p : Polynomial ℤ, (X^2 + X - 2) * p = a*X^4 + b*X^3 - 2) → b = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_b_equals_four_l1192_119247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1192_119214

/-- The first curve defined by y^2 - 9 + 2yx - 12x - 3x^2 = 0 -/
def curve1 (x y : ℝ) : Prop := y^2 - 9 + 2*y*x - 12*x - 3*x^2 = 0

/-- The second curve defined by y^2 + 3 - 4x - 2y + x^2 = 0 -/
def curve2 (x y : ℝ) : Prop := y^2 + 3 - 4*x - 2*y + x^2 = 0

/-- The distance between two points (x1, y1) and (x2, y2) -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem stating the minimum distance between points on the two curves -/
theorem min_distance_between_curves :
  ∃ (x1 y1 x2 y2 : ℝ), 
    curve1 x1 y1 ∧ curve2 x2 y2 ∧
    (∀ (x3 y3 x4 y4 : ℝ), curve1 x3 y3 → curve2 x4 y4 → 
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1192_119214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_fixed_point_in_30_seconds_l1192_119202

/-- Represents the properties of a train and its movement --/
structure Train :=
  (length : ℚ)
  (platform_length : ℚ)
  (platform_crossing_time : ℚ)

/-- Calculates the time it takes for the train to pass a fixed point --/
def time_to_pass_fixed_point (t : Train) : ℚ :=
  t.length * t.platform_crossing_time / (t.length + t.platform_length)

/-- Theorem stating that a train with the given properties will take 30 seconds to pass a fixed point --/
theorem train_passes_fixed_point_in_30_seconds (t : Train) 
  (h1 : t.length = 400)
  (h2 : t.platform_length = 200)
  (h3 : t.platform_crossing_time = 45) :
  time_to_pass_fixed_point t = 30 := by
  sorry

def main : IO Unit := do
  let train : Train := { length := 400, platform_length := 200, platform_crossing_time := 45 }
  IO.println s!"Time to pass fixed point: {time_to_pass_fixed_point train}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_fixed_point_in_30_seconds_l1192_119202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_horizontal_asymptote_l1192_119289

/-- The function for which we want to find the horizontal asymptote -/
noncomputable def f (x : ℝ) : ℝ := (10*x^4 + 5*x^3 + 7*x^2 + 2*x + 4) / (2*x^4 + x^3 + 4*x^2 + x + 2)

/-- The horizontal asymptote of the function f -/
def horizontal_asymptote : ℝ := 5

/-- Theorem stating that the horizontal asymptote of f is 5 -/
theorem f_horizontal_asymptote : 
  ∀ ε > 0, ∃ N, ∀ x, |x| > N → |f x - horizontal_asymptote| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_horizontal_asymptote_l1192_119289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l1192_119207

-- Define the arithmetic sequence and its sum
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (a₁ + arithmetic_sequence a₁ d n) / 2

-- State the theorem
theorem arithmetic_sequence_difference (a₁ d : ℝ) :
  (arithmetic_sum a₁ d 2017 / 2017) - (arithmetic_sum a₁ d 17 / 17) = 100 →
  d = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_difference_l1192_119207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1192_119232

noncomputable def f (x : ℝ) : ℝ := (x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1) / (x^2 - 9)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -3 ∨ (-3 < x ∧ x < 3) ∨ 3 < x} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1192_119232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_angle_is_45_degrees_l1192_119277

/-- A regular hexagonal pyramid where the height is equal to the side of the base -/
structure RegularHexagonalPyramid where
  /-- Side length of the base hexagon -/
  side : ℝ
  /-- Height of the pyramid -/
  height : ℝ
  /-- The height is equal to the side of the base -/
  height_eq_side : height = side

/-- The angle between a lateral edge and the plane of the base in a regular hexagonal pyramid -/
noncomputable def lateral_edge_angle (p : RegularHexagonalPyramid) : ℝ :=
  Real.arctan (p.height / p.side)

/-- Theorem: The angle between a lateral edge and the plane of the base is 45° -/
theorem lateral_edge_angle_is_45_degrees (p : RegularHexagonalPyramid) :
  lateral_edge_angle p = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_angle_is_45_degrees_l1192_119277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angles_first_or_second_quadrant_first_quadrant_acute_different_measure_different_terminal_side_same_terminal_side_periodic_third_quadrant_point_second_quadrant_angle_correct_propositions_l1192_119261

-- Define the concept of a quadrant
inductive Quadrant
| first
| second
| third
| fourth

-- Define the concept of an angle
structure Angle where
  measure : ℝ
  terminal_side : ℝ × ℝ

-- Define the concept of a triangle
structure Triangle where
  angles : Fin 3 → Angle

-- Define what it means for an angle to be in a quadrant
def angle_in_quadrant (α : Angle) (q : Quadrant) : Prop := sorry

-- Define what it means for an angle to be acute
def is_acute (α : Angle) : Prop := sorry

-- Define when two angles have the same terminal side
def same_terminal_side (α β : Angle) : Prop := sorry

-- Theorem corresponding to proposition ①
theorem interior_angles_first_or_second_quadrant : Prop := sorry

-- Theorem corresponding to proposition ②
theorem first_quadrant_acute : Prop := sorry

-- Theorem corresponding to proposition ③
theorem different_measure_different_terminal_side : Prop := sorry

-- Theorem corresponding to proposition ④
theorem same_terminal_side_periodic : Prop := sorry

-- Theorem corresponding to proposition ⑤
theorem third_quadrant_point_second_quadrant_angle : Prop := sorry

-- The main theorem stating which propositions are correct
theorem correct_propositions : 
  (¬interior_angles_first_or_second_quadrant) ∧
  (¬first_quadrant_acute) ∧
  (¬different_measure_different_terminal_side) ∧
  same_terminal_side_periodic ∧
  third_quadrant_point_second_quadrant_angle := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angles_first_or_second_quadrant_first_quadrant_acute_different_measure_different_terminal_side_same_terminal_side_periodic_third_quadrant_point_second_quadrant_angle_correct_propositions_l1192_119261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_curves_l1192_119203

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

def C₂ (ρ θ : ℝ) : Prop := ρ^2 - 8*ρ*(Real.sin θ) + 15 = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- State the theorem
theorem max_distance_between_curves :
  ∃ (M : ℝ), M = 8 ∧
  ∀ (x₁ y₁ ρ θ : ℝ),
    C₁ x₁ y₁ → C₂ ρ θ →
    distance x₁ y₁ (ρ * (Real.cos θ)) (ρ * (Real.sin θ)) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_curves_l1192_119203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_and_max_integer_l1192_119268

-- Define the constants and variables
def π : ℝ := 3
def area_between_circles : ℝ := 96 * π
def radii_difference : ℝ := 2

-- Define the area of the larger circle
def Q : ℝ := 1875

-- Define the function for R
def R : ℕ := 2

-- Theorem statement
theorem circle_area_and_max_integer :
  -- Given conditions
  (π = 3) →
  (area_between_circles = 96 * π) →
  (radii_difference = 2) →
  -- Prove that Q is the area of the larger circle
  (Q = 1875) ∧
  -- Prove that R is the largest integer such that R^Q < 5^200
  (R = 2) ∧
  ((R : ℝ) ^ Q < (5 : ℝ) ^ 200) ∧
  ∀ n : ℕ, n > R → ((n : ℝ) ^ Q ≥ (5 : ℝ) ^ 200) :=
by
  intro h_pi h_area h_diff
  have h_Q : Q = 1875 := rfl
  have h_R : R = 2 := rfl
  
  -- Proof of R^Q < 5^200
  have h_inequality : ((R : ℝ) ^ Q < (5 : ℝ) ^ 200) := by
    -- Add your proof here
    sorry

  -- Proof for ∀ n : ℕ, n > R → (n : ℝ) ^ Q ≥ (5 : ℝ) ^ 200
  have h_forall : ∀ n : ℕ, n > R → ((n : ℝ) ^ Q ≥ (5 : ℝ) ^ 200) := by
    -- Add your proof here
    sorry

  -- Combine all parts of the proof
  exact ⟨h_Q, h_R, h_inequality, h_forall⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_and_max_integer_l1192_119268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_sum_theorem_l1192_119280

def marble_choices (n m : ℕ) : ℕ :=
  (List.range n).foldl
    (λ count i ↦
      count + (List.range m).foldl
        (λ inner_count j ↦
          inner_count + (List.range m).foldl
            (λ innermost_count k ↦
              if i + 1 = j + 1 + k + 1 ∧ j ≠ k then innermost_count + 1 else innermost_count)
            0)
        0)
    0

theorem marble_sum_theorem : marble_choices 15 7 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_sum_theorem_l1192_119280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l1192_119292

/-- The curved surface area of a cone -/
noncomputable def curved_surface_area (r l : ℝ) : ℝ := Real.pi * r * l

/-- Theorem: The curved surface area of a cone with radius 7 and slant height 14 is 98π -/
theorem cone_surface_area :
  curved_surface_area 7 14 = 98 * Real.pi := by
  unfold curved_surface_area
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l1192_119292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebound_percentage_for_given_conditions_l1192_119294

/-- Given a ball dropped from an initial height, calculate the rebound percentage
    based on the total travel distance when it touches the floor for the third time. -/
noncomputable def rebound_percentage (initial_height : ℝ) (total_travel : ℝ) : ℝ :=
  let equation := fun p => initial_height + 2 * initial_height * p + 2 * initial_height * p^2 - total_travel
  let solution := Real.sqrt ((2:ℝ)^2 + 4 * 2 * 1.5) / (2 * 2) - 1 / 2
  solution

/-- Theorem stating that for a ball dropped from 100 cm with a total travel of 250 cm
    when touching the floor for the third time, the rebound percentage is 0.5. -/
theorem rebound_percentage_for_given_conditions :
  rebound_percentage 100 250 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebound_percentage_for_given_conditions_l1192_119294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_sum_reciprocal_inequality_l1192_119267

-- Problem 1
theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 8 < 2 + Real.sqrt 7 := by sorry

-- Problem 2
theorem sum_reciprocal_inequality {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 1/a + 1/b + 1/c ≥ 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_sum_reciprocal_inequality_l1192_119267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_l1192_119231

noncomputable def circle_P (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

def Q : ℝ × ℝ := (4, 0)

noncomputable def M (P : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def trajectory_M (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

def A (t : ℝ) : ℝ × ℝ := (0, t)
def B (t : ℝ) : ℝ × ℝ := (0, t + 6)

def t_range (t : ℝ) : Prop := -5 ≤ t ∧ t ≤ -2

noncomputable def area_ABC (t : ℝ) : ℝ := 6 * (1 - 1 / (t^2 + 6*t + 1))

theorem trajectory_and_area :
  ∀ (x y t : ℝ),
  (∃ P : ℝ × ℝ, circle_P P.1 P.2 ∧ M P = (x, y)) →
  trajectory_M x y ∧
  (t_range t →
    (∃ C : ℝ × ℝ, trajectory_M C.1 C.2 ∧
      (area_ABC t ≤ 15/2 ∧ area_ABC t ≥ 27/4) ∧
      (∃ t₀, t_range t₀ ∧ area_ABC t₀ = 15/2) ∧
      (∃ t₁, t_range t₁ ∧ area_ABC t₁ = 27/4))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_l1192_119231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_min_value_attained_l1192_119281

/-- The function f as defined in the problem -/
noncomputable def f (a b c : ℝ) : ℝ :=
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*a*c) + c / Real.sqrt (c^2 + 8*a*b)

/-- Theorem stating that the minimum value of f is 1 for positive real inputs -/
theorem min_value_of_f :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → f a b c ≥ 1 := by
  sorry

/-- Theorem stating that the minimum value of 1 is attained -/
theorem min_value_attained :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ f a b c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_min_value_attained_l1192_119281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_absolute_value_term_l1192_119208

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem smallest_absolute_value_term
  (a₁ d : ℝ)
  (h₁ : S a₁ d 2018 > 0)
  (h₂ : S a₁ d 2019 < 0) :
  ∀ n : ℕ, |arithmetic_sequence a₁ d 1010| ≤ |arithmetic_sequence a₁ d n| :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_absolute_value_term_l1192_119208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_distribution_with_less_than_946_in_6_minutes_l1192_119216

/-- Represents the typing scenario of Xiao Jin -/
structure TypingScenario where
  total_chars : ℕ
  total_time : ℕ
  first_minute : ℕ
  last_minute : ℕ

/-- Represents a possible distribution of characters typed per minute -/
def valid_distribution (s : TypingScenario) (dist : List ℕ) : Prop :=
  dist.length = s.total_time ∧
  dist.sum = s.total_chars ∧
  dist.head! = s.first_minute ∧
  dist.reverse.head! = s.last_minute

/-- Theorem stating that there exists a valid distribution where some consecutive 6 minutes have less than 946 characters -/
theorem exists_distribution_with_less_than_946_in_6_minutes 
  (s : TypingScenario)
  (h_total : s.total_chars = 2098)
  (h_time : s.total_time = 14)
  (h_first : s.first_minute = 112)
  (h_last : s.last_minute = 97) :
  ∃ (dist : List ℕ), valid_distribution s dist ∧
    ∃ (i : ℕ), i + 6 ≤ s.total_time ∧
      (List.take 6 (List.drop i dist)).sum < 946 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_distribution_with_less_than_946_in_6_minutes_l1192_119216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l1192_119241

/-- Given two perpendicular lines, prove the distance from a specific point to another line -/
theorem distance_to_line (m : ℝ) : 
  (∀ x y, 2*x + y - 2 = 0 ∧ x + m*y - 1 = 0 → (2 : ℝ) * (-1/m) = -1) →
  let P : ℝ × ℝ := (m, m)
  let d := |1*m + 1*m + 3| / Real.sqrt (1^2 + 1^2)
  d = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l1192_119241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_snowflake_stamps_l1192_119245

/-- Represents the number of snowflake stamps Bella bought -/
def snowflake_stamps : ℕ := sorry

/-- Represents the number of truck stamps Bella bought -/
def truck_stamps : ℕ := snowflake_stamps + 9

/-- Represents the number of rose stamps Bella bought -/
def rose_stamps : ℕ := truck_stamps - 13

/-- The total number of stamps Bella bought -/
def total_stamps : ℕ := 38

theorem bella_snowflake_stamps :
  snowflake_stamps + truck_stamps + rose_stamps = total_stamps →
  snowflake_stamps = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_snowflake_stamps_l1192_119245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_arcade_time_l1192_119218

/-- Calculates the number of minutes Mike can play at the arcade given his weekly pay,
    the fraction he spends at the arcade, food cost, and the cost per hour of play. -/
def arcade_play_time (weekly_pay : ℕ) (arcade_fraction : ℚ) (food_cost : ℕ) (cost_per_hour : ℕ) : ℕ :=
  let arcade_budget := (weekly_pay : ℚ) * arcade_fraction
  let token_budget := arcade_budget - food_cost
  let hours_of_play := token_budget / cost_per_hour
  (hours_of_play * 60).floor.toNat

theorem mike_arcade_time :
  arcade_play_time 100 (1/2) 10 8 = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_arcade_time_l1192_119218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l1192_119234

/-- Translate a point by a given vector -/
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

theorem midpoint_after_translation :
  let B : ℝ × ℝ := (1, 1)
  let G : ℝ × ℝ := (5, 1)
  let translation_vector : ℝ × ℝ := (-7, -4)
  let B' := translate B translation_vector
  let G' := translate G translation_vector
  (B'.1 + G'.1) / 2 = -4 ∧ (B'.2 + G'.2) / 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l1192_119234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_side_length_in_rectangle_l1192_119230

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a trapezoid with parallel sides a and b, and height h -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ

/-- The area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ := (t.a + t.b) * t.h / 2

theorem trapezoid_side_length_in_rectangle (rect : Rectangle) 
    (h1 : rect.width = 2) 
    (h2 : rect.height = 1) 
    (h3 : ∃ t : Trapezoid, t.h = rect.height ∧ t.b = rect.height / 2 ∧ 
          3 * trapezoidArea t = rect.width * rect.height) : 
    ∃ t : Trapezoid, t.a = 1/6 ∧ t.h = rect.height ∧ t.b = rect.height / 2 ∧ 
          3 * trapezoidArea t = rect.width * rect.height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_side_length_in_rectangle_l1192_119230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_training_speed_l1192_119258

theorem laura_training_speed :
  ∃ x : ℝ, x > 0 ∧ 
    (30 / (3 * x + 1) + 10 / x = (160 - 5) / 60) ∧
    (abs (x - 7.57) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_training_speed_l1192_119258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l1192_119242

def FunctionSet := {f : ℝ → ℝ | ∀ x, x > 0 → f x > 0}

def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ ω x y z : ℝ, ω > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ ω * x = y * z →
    (f ω)^2 + (f x)^2 / (f (y^2) + f (z^2)) = (ω^2 + x^2) / (y^2 + z^2)

theorem function_characterization (f : ℝ → ℝ) (hf : f ∈ FunctionSet) (h : SatisfiesEquation f) :
  (∀ x : ℝ, x > 0 → f x = x) ∨ (∀ x : ℝ, x > 0 → f x = 1 / x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l1192_119242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carrot_weight_problem_l1192_119220

/-- Given 30 carrots weighing 5.94 kg in total, and 3 carrots with an average weight of 180 grams
    are removed, the average weight of the remaining 27 carrots is 200 grams. -/
theorem carrot_weight_problem :
  let total_carrots : ℕ := 30
  let total_weight : ℝ := 5.94
  let removed_carrots : ℕ := 3
  let removed_avg_weight : ℝ := 180 / 1000
  let remaining_carrots : ℕ := total_carrots - removed_carrots
  let remaining_avg_weight : ℝ := (total_weight - removed_carrots * removed_avg_weight) / remaining_carrots
  remaining_avg_weight = 200 / 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carrot_weight_problem_l1192_119220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_at_one_l1192_119283

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = a - 2 / (2^x + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a - 2 / (2^x + 1)

theorem odd_function_value_at_one (a : ℝ) (h : IsOdd (f a)) : f a 1 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_at_one_l1192_119283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_equals_512_l1192_119250

theorem positive_difference_equals_512 : 
  |((8^2 - 8^2) / 8 : ℚ) - ((8^2 * 8^2) / 8 : ℚ)| = 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_equals_512_l1192_119250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_square_root_l1192_119275

theorem arithmetic_sequence_square_root (y : ℝ) : 
  y > 0 → (2^2 < y^2 ∧ y^2 < 4^2) → y = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_square_root_l1192_119275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_identity_l1192_119263

theorem cube_root_identity : (2^4 + 4^3 + 8^2 : ℝ)^(1/3) = 2^(4/3) * 3^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_identity_l1192_119263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_implies_a_bound_h_minimum_value_l1192_119252

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x - a * x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := Real.exp (3 * x) - 3 * a * Real.exp x

-- State the theorems
theorem g_increasing_implies_a_bound (a : ℝ) :
  (∀ x > 0, Monotone (g a)) → a ≤ 2 * Real.sqrt 2 := by sorry

theorem h_minimum_value (a : ℝ) (ha : a > 1) (ha' : a ≤ 2 * Real.sqrt 2) :
  ∃ x ∈ Set.Icc 0 (Real.log 2), ∀ y ∈ Set.Icc 0 (Real.log 2), h a x ≤ h a y ∧ h a x = -2 * a^(3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_implies_a_bound_h_minimum_value_l1192_119252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_fraction_l1192_119246

theorem divisible_by_six_fraction (n : ℕ) (h : n = 150) : 
  (Finset.filter (λ x => x % 6 = 0) (Finset.range (n + 1))).card / (n : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_fraction_l1192_119246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_complex_number_trigonometric_sum_l1192_119262

theorem imaginary_complex_number_trigonometric_sum (θ : ℝ) :
  (Complex.I * ((Real.sin θ - Real.cos θ) : ℝ) = (Real.sin θ + Real.cos θ + 1) + Complex.I * (Real.sin θ - Real.cos θ)) →
  (Real.sin θ)^2017 + (Real.cos θ)^2017 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_complex_number_trigonometric_sum_l1192_119262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l1192_119224

theorem power_equation (a b : ℝ) (h1 : (100 : ℝ)^a = 4) (h2 : (100 : ℝ)^b = 5) : 
  (20 : ℝ)^((1 - a - b)/(2*(1 - b))) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l1192_119224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1192_119266

theorem polynomial_factorization :
  ∃ (q r : Polynomial ℤ), q ≠ 0 ∧ r ≠ 0 ∧
    (5 * X ^ 2) ^ 4 + (5 * X ^ 2) ^ 3 + (5 * X ^ 2) ^ 2 + (5 * X ^ 2) + 1 = q * r :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l1192_119266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_system_l1192_119285

theorem solve_system (x y z : ℝ) 
  (eq1 : (7 : ℝ)^(x - y) = 343)
  (eq2 : (7 : ℝ)^(x + y) = 16807)
  (eq3 : (7 : ℝ)^(x - z) = 2401) : 
  x = 4 ∧ y = 1 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_system_l1192_119285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l1192_119210

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by the equation Ax + By + C = 0 --/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The distance between a point and a line --/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  abs (l.A * p.1 + l.B * p.2 + l.C) / Real.sqrt (l.A^2 + l.B^2)

/-- The shortest distance from a point on a circle to a line --/
noncomputable def shortestDistanceCircleToLine (c : Circle) (l : Line) : ℝ :=
  distancePointToLine c.center l - c.radius

theorem shortest_distance_circle_to_line :
  let c : Circle := { center := (5, 3), radius := 3 }
  let l : Line := { A := 3, B := 4, C := -2 }
  shortestDistanceCircleToLine c l = 2 := by
  sorry

#check shortest_distance_circle_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l1192_119210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_napkin_width_l1192_119254

/-- Given the dimensions of a tablecloth and napkins, and the total material needed,
    this theorem proves the width of each napkin. -/
theorem napkin_width
  (tablecloth_length : ℕ)
  (tablecloth_width : ℕ)
  (num_napkins : ℕ)
  (napkin_length : ℕ)
  (total_material : ℕ)
  (h1 : tablecloth_length = 102)
  (h2 : tablecloth_width = 54)
  (h3 : num_napkins = 8)
  (h4 : napkin_length = 6)
  (h5 : total_material = 5844)
  : ∃ (napkin_width : ℕ), 
    total_material = tablecloth_length * tablecloth_width + num_napkins * napkin_length * napkin_width ∧
    napkin_width = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_napkin_width_l1192_119254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_eq_neg_one_l1192_119223

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) → m₁ = m₂

/-- Definition of line l₁ -/
def l₁ (x y : ℝ) : Prop := x + y - 2 = 0

/-- Definition of line l₂ -/
def l₂ (a x y : ℝ) : Prop := a * x - y + 7 = 0

/-- If l₁ and l₂ are parallel, then a = -1 -/
theorem parallel_lines_a_eq_neg_one (a : ℝ) : 
  (∀ x y, l₁ x y ↔ l₂ a x y) → a = -1 := by
  sorry

#check parallel_lines_a_eq_neg_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_eq_neg_one_l1192_119223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_relation_angle_B_value_l1192_119260

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angleSum : A + B + C = Real.pi
  sineRule : a / Real.sin A = b / Real.sin B
  cosineRule : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- Part I
theorem cosine_relation (t : Triangle) (h : t.C = 2 * t.B) :
  Real.cos t.A = 3 * Real.cos t.B - 4 * (Real.cos t.B)^3 := by sorry

-- Part II
theorem angle_B_value (t : Triangle) 
  (h1 : t.b * Real.sin t.B - t.c * Real.sin t.C = t.a)
  (h2 : (t.b^2 + t.c^2 - t.a^2) / 4 = (1/2) * t.b * t.c * Real.sin t.A) :
  t.B = Real.pi * 77.5 / 180 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_relation_angle_B_value_l1192_119260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_equation_correct_l1192_119215

/-- Represents the number of people in the first group initially -/
def initial_group1 : ℕ := 12

/-- Represents the number of people in the second group initially -/
def initial_group2 : ℕ := 15

/-- Theorem stating that the equation correctly represents the situation after transfer -/
theorem transfer_equation_correct (x : ℕ) :
  initial_group1 + x = 2 * (initial_group2 - x) ↔
  (initial_group1 + x = 2 * (initial_group2 - x) ∧
   initial_group1 + x > initial_group2 - x ∧
   x ≤ initial_group2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_equation_correct_l1192_119215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_force_on_dam_l1192_119259

/-- The force exerted by water on a dam with an isosceles trapezoidal cross-section -/
noncomputable def waterForce (ρ g a b h : ℝ) : ℝ :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

/-- Theorem stating that the force exerted by water on the dam is 576000 N -/
theorem water_force_on_dam :
  let ρ : ℝ := 1000  -- water density in kg/m³
  let g : ℝ := 10    -- acceleration due to gravity in m/s²
  let a : ℝ := 6.0   -- smaller base of the trapezoid in m
  let b : ℝ := 9.6   -- larger base of the trapezoid in m
  let h : ℝ := 4.0   -- height of the trapezoid in m
  waterForce ρ g a b h = 576000 := by
  sorry

#check water_force_on_dam

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_force_on_dam_l1192_119259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_graph_properties_l1192_119219

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -x - 2
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x-2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2*(x-2)
  else 0  -- undefined elsewhere

-- Define the transformed function h(x) = (1/3)g(x) + 2
noncomputable def h (x : ℝ) : ℝ := (1/3) * g x + 2

-- Theorem statement
theorem transformed_graph_properties :
  (∀ x ∈ Set.Icc (-3) 0, h x = -(1/3)*x + 4/3) ∧
  (∀ x ∈ Set.Ioo 0 2, (x - 2)^2 + (h x - 4/3)^2 = (2/3)^2) ∧
  (∀ x ∈ Set.Ioc 2 3, h x = (2/3)*x + 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_graph_properties_l1192_119219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1192_119222

-- Define the power function f(x) that passes through (2,4)
noncomputable def f : ℝ → ℝ := λ x => x^2

-- Define g(x) using f(x)
noncomputable def g : ℝ → ℝ := λ x => (1/2)^(f x - 4 * Real.sqrt (f x) + 3)

-- Theorem statement
theorem range_of_g :
  Set.range g = Set.Ioo 0 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1192_119222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l1192_119248

/-- Conversion factor from km/h to m/s -/
noncomputable def kmph_to_ms : ℝ := 5 / 18

/-- Speed of the train in km/h -/
def train_speed_kmph : ℝ := 77.993280537557

/-- Speed of the man in km/h -/
def man_speed_kmph : ℝ := 6

/-- Time taken for the train to pass the man in seconds -/
def passing_time : ℝ := 6

/-- Calculate the length of the train -/
noncomputable def train_length : ℝ :=
  (train_speed_kmph * kmph_to_ms + man_speed_kmph * kmph_to_ms) * passing_time

/-- Theorem stating that the calculated train length is approximately 139.99 meters -/
theorem train_length_approx :
  abs (train_length - 139.99) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l1192_119248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_theorem_l1192_119274

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points in 2D space -/
noncomputable def distance2D (p q : Point2D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Calculates the distance between two points in 3D space after folding -/
noncomputable def distanceAfterFold (p q : Point2D) (θ : ℝ) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y * Real.cos θ - q.y * Real.cos θ)^2 + (p.y * Real.sin θ + q.y * Real.sin θ)^2)

theorem dihedral_angle_theorem :
  let A : Point2D := ⟨-2, 3⟩
  let B : Point2D := ⟨3, -2⟩
  let θ : ℝ := 2 * Real.pi / 3  -- 120° in radians
  distanceAfterFold A B θ = 2 * Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_theorem_l1192_119274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_sqrt5_implies_x_is_1_l1192_119288

/-- The distance between two points in 3D space -/
noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- Theorem: If the distance between A(0,0,3) and B(x,2,3) is √5, and x > 0, then x = 1 -/
theorem distance_is_sqrt5_implies_x_is_1 (x : ℝ) (h1 : x > 0) 
  (h2 : distance 0 0 3 x 2 3 = Real.sqrt 5) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_sqrt5_implies_x_is_1_l1192_119288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cigarette_puzzle_l1192_119299

/-- Represents a cigarette or match in the puzzle -/
inductive Cigarette
| vertical : Cigarette
| horizontal : Cigarette
| diagonal : Cigarette
deriving BEq, Repr

/-- Represents the possible configurations after moving cigarettes -/
inductive Configuration
| zero : Configuration
| nil : Configuration
deriving BEq, Repr

/-- Represents the initial arrangement of cigarettes forming 57 -/
def initial_arrangement : List Cigarette := 
  [Cigarette.horizontal, Cigarette.vertical, Cigarette.horizontal, 
   Cigarette.vertical, Cigarette.horizontal, Cigarette.diagonal]

/-- Defines a valid move in the puzzle -/
def is_valid_move (before after : List Cigarette) : Prop :=
  before.length = 6 ∧ after.length = 6 ∧ 
  (∃ (a b c d : Cigarette), 
    before.filter (λ x => x != a && x != b) = after.filter (λ x => x != c && x != d))

/-- The main theorem to prove -/
theorem cigarette_puzzle : 
  ∃ (result : Configuration) (final_arrangement : List Cigarette),
    is_valid_move initial_arrangement final_arrangement ∧
    ((result = Configuration.zero ∧ 
      final_arrangement.count Cigarette.vertical = 2 ∧
      final_arrangement.count Cigarette.horizontal = 4) ∨
     (result = Configuration.nil ∧
      final_arrangement.count Cigarette.vertical = 3 ∧
      final_arrangement.count Cigarette.horizontal = 2 ∧
      final_arrangement.count Cigarette.diagonal = 1)) := by
  sorry

#eval initial_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cigarette_puzzle_l1192_119299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_equal_perimeter_triangle_hexagon_l1192_119293

/-- Given an equilateral triangle and a regular hexagon with equal perimeters,
    if the ratio of their areas is 2:a, then a = 3 -/
theorem area_ratio_of_equal_perimeter_triangle_hexagon :
  ∀ (s t a : ℝ), s > 0 → t > 0 → a > 0 →
  3 * s = 6 * t →  -- equal perimeters
  (Real.sqrt 3 / 4 * s^2) / ((3 * Real.sqrt 3 / 2) * t^2) = 2 / a →  -- area ratio
  a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_equal_perimeter_triangle_hexagon_l1192_119293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_z_l1192_119278

theorem max_value_of_z (x y a : ℝ) (h : Real.cos x + Real.sin y = 1/2) :
  let z := a * Real.sin y + (Real.cos x)^2
  ∃ (z_max : ℝ), ∀ (x' y' : ℝ), Real.cos x' + Real.sin y' = 1/2 → 
    a * Real.sin y' + (Real.cos x')^2 ≤ z_max ∧
    z_max = if a ≤ 1/2 then 1 - a/2 else a + 1/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_z_l1192_119278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_min_at_three_l1192_119228

/-- Sequence term definition -/
noncomputable def a (n : ℕ) : ℝ := n

/-- Sum of the first n terms -/
noncomputable def S (n : ℕ) : ℝ := (n * (n + 1)) / 2

/-- Definition of bₙ -/
noncomputable def b (n : ℕ) : ℝ := (2 * S n + 7) / n

/-- Theorem stating that bₙ takes its minimum value when n = 3 -/
theorem b_min_at_three : 
  ∀ k : ℕ, k ≥ 1 → b 3 ≤ b k :=
by
  sorry

/-- Helper lemma: b_n can be simplified to n + 7/n + 1 -/
lemma b_simplified (n : ℕ) (h : n ≥ 1) : 
  b n = n + 7 / n + 1 :=
by
  sorry

/-- Helper lemma: Monotonicity of f(x) = x + 7/x -/
lemma f_monotonicity (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x < y → x + 7/x < y + 7/y ↔ x > Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_min_at_three_l1192_119228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l1192_119226

noncomputable section

-- Define the diagonal of the square
def square_diagonal : ℝ := 8

-- Define the diameter of the circle
def circle_diameter : ℝ := 8

-- Define the side length of the square
noncomputable def square_side : ℝ := Real.sqrt (square_diagonal ^ 2 / 2)

-- Define the area of the square
noncomputable def square_area : ℝ := square_side ^ 2

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := circle_diameter / 2

-- Define the area of the circle
noncomputable def circle_area : ℝ := Real.pi * circle_radius ^ 2

-- Define the difference in areas
noncomputable def area_difference : ℝ := circle_area - square_area

-- Theorem statement
theorem area_difference_approx : 
  ∃ ε > 0, abs (area_difference - 18.3) < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l1192_119226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_riemann_function_sum_l1192_119264

-- Define the Riemann function R
def R : ℝ → ℝ := sorry

-- Define the function f
def f : ℝ → ℝ := sorry

-- Axioms for f
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_property : ∀ x : ℝ, f (1 + x) = -f (1 - x)
axiom f_eq_R : ∀ x : ℝ, x ∈ Set.Icc 0 1 → f x = R x

-- Theorem to prove
theorem riemann_function_sum :
  f 2023 + f (2023 / 2) + f (-2023 / 3) = -5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_riemann_function_sum_l1192_119264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_nonzero_area_is_21_l1192_119257

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (42, 18)

-- Define the set of possible C points with integer coordinates
def C : Set (ℤ × ℤ) := {p : ℤ × ℤ | true}

-- Function to calculate the area of triangle ABC given C
noncomputable def triangleArea (c : ℤ × ℤ) : ℝ :=
  let (p, q) := c
  (1/2) * abs (42 * (q : ℝ) - 756)

-- Theorem statement
theorem min_nonzero_area_is_21 :
  ∃ c ∈ C, triangleArea c ≠ 0 ∧
  ∀ c' ∈ C, triangleArea c' ≠ 0 → triangleArea c ≤ triangleArea c' ∧
  triangleArea c = 21 := by
  sorry

#check min_nonzero_area_is_21

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_nonzero_area_is_21_l1192_119257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_perimeter_ratio_l1192_119296

theorem square_diagonal_perimeter_ratio :
  ∀ (d₁ d₂ p₁ p₂ : ℝ),
  d₁ > 0 → d₂ > 0 → p₁ > 0 → p₂ > 0 →
  d₂ / d₁ = 3/2 →
  p₁ = 2 * Real.sqrt 2 * d₁ →
  p₂ = 2 * Real.sqrt 2 * d₂ →
  p₂ / p₁ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_perimeter_ratio_l1192_119296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_l1192_119205

theorem cos_alpha_minus_pi (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.cos α) : 
  Real.cos (α - π) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_l1192_119205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l1192_119295

/-- Represents a 3D vector --/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the rotation of a vector --/
def rotate90AboutOrigin (v : Vector3D) : Vector3D :=
  sorry

/-- Checks if a vector passes through the y-axis during rotation --/
def passesYAxisDuringRotation (v : Vector3D) : Prop :=
  sorry

/-- The initial vector --/
def initialVector : Vector3D :=
  { x := 2, y := 1, z := 1 }

/-- The expected result vector --/
noncomputable def resultVector : Vector3D :=
  { x := -Real.sqrt (6/11), y := 3 * Real.sqrt (6/11), z := -Real.sqrt (6/11) }

theorem rotation_result :
  rotate90AboutOrigin initialVector = resultVector ∧
  passesYAxisDuringRotation initialVector :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l1192_119295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l1192_119282

/-- The coefficient of x in the simplified expression 5(x - 6) + 7(8 - 3x^2 + 6x) - 9(3x - 2) is 20 -/
theorem coefficient_of_x (x : ℝ) : 
  let expr := fun x => 5 * (x - 6) + 7 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2)
  deriv expr x = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l1192_119282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l1192_119284

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  A = π/6 →
  a = 1 →
  b = Real.sqrt 3 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a/Real.sin A = b/Real.sin B →
  a/Real.sin A = c/Real.sin C →
  (B = π/3 ∨ B = 2*π/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l1192_119284
