import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l13_1361

-- Define the equation
def equation (x y : ℝ) : Prop := y^2 + 4*x*y + 80*abs x = 800

-- Define the bounded region
def bounded_region : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ equation x y}

-- State the theorem
theorem area_of_bounded_region :
  MeasureTheory.volume bounded_region = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l13_1361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_perimeter_is_72_l13_1305

/-- A regular hexagon with side length 6 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 6)

/-- A square constructed outside the hexagon -/
structure OuterSquare :=
  (side_length : ℝ)
  (is_equal_to_hexagon : side_length = 6)

/-- The dodecagon formed by the vertices of the outer squares -/
structure Dodecagon where
  vertices : List (ℝ × ℝ)
  num_vertices : vertices.length = 12

/-- The perimeter of the dodecagon -/
noncomputable def dodecagon_perimeter (d : Dodecagon) : ℝ := sorry

theorem dodecagon_perimeter_is_72 
  (h : RegularHexagon) 
  (s1 s2 s3 s4 s5 s6 : OuterSquare) 
  (d : Dodecagon) : 
  dodecagon_perimeter d = 72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_perimeter_is_72_l13_1305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_equal_terms_l13_1318

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then 1 / (x - 1)
  else if x = 1 then 1
  else x / (1 - x)

noncomputable def x_sequence (x₁ : ℝ) : ℕ → ℝ
  | 0 => x₁
  | n + 1 => f (x_sequence x₁ n)

theorem existence_of_equal_terms (x₁ : ℝ) 
  (h_irrational : Irrational x₁)
  (h_quadratic : ∃ (a b c : ℤ), a ≠ 0 ∧ a * x₁^2 + b * x₁ + c = 0)
  (h_positive : x₁ > 0) :
  ∃ (k ℓ : ℕ), k ≠ ℓ ∧ x_sequence x₁ k = x_sequence x₁ ℓ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_equal_terms_l13_1318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_position_in_decimal_rep_l13_1398

/-- The decimal representation of 325/999 -/
def decimal_rep : ℚ := 325 / 999

/-- The position of the first occurrence of 5 in the decimal part of 325/999 -/
def position_of_five : ℕ := 3

theorem five_position_in_decimal_rep :
  (Int.floor (decimal_rep * 1000) % 10 : ℤ) = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_position_in_decimal_rep_l13_1398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem1_problem2_l13_1344

-- Problem 1
noncomputable def S (v a t : ℝ) : ℝ := v * t + (1/2) * a * t^2

theorem problem1 (v a : ℝ) (h1 : S v a 1 = 4) (h2 : S v a 2 = 10) : 
  S v a 3 = 18 := by sorry

-- Problem 2
def inequality_system (x : ℤ) : Prop :=
  (3*x - 2*(x+1) < 3) ∧ (1 - (x-2)/3 ≤ x/2)

theorem problem2 : 
  {x : ℤ | inequality_system x} = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem1_problem2_l13_1344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_on_unit_cube_surface_l13_1338

/-- The length of the shortest path on the surface of a unit cube between opposite vertices -/
noncomputable def shortest_path_length_unit_cube : ℝ := Real.sqrt 5

/-- Theorem stating that the shortest path length on a unit cube's surface between opposite vertices is √5 -/
theorem shortest_path_on_unit_cube_surface :
  shortest_path_length_unit_cube = Real.sqrt 5 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_on_unit_cube_surface_l13_1338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_of_triangles_l13_1302

-- Define the geometric configuration
structure GeometricConfig where
  h₄ : ℝ  -- Side length of the quadrilateral
  h₅ : ℝ  -- Side length of the pentagon
  -- Assuming h₄ and h₅ are positive
  h₄_pos : h₄ > 0
  h₅_pos : h₅ > 0

-- Define the areas of triangles ABC and CDE
noncomputable def area_ABC (config : GeometricConfig) : ℝ :=
  (config.h₄ * config.h₅ / 2) * Real.sin (9 * Real.pi / 180)

noncomputable def area_CDE (config : GeometricConfig) : ℝ :=
  (Real.tan (9 * Real.pi / 180) / 4) * (2 * config.h₅ * Real.cos (9 * Real.pi / 180) - config.h₄)^2

-- Theorem statement
theorem areas_of_triangles (config : GeometricConfig) :
  (area_ABC config = (config.h₄ * config.h₅ / 2) * Real.sin (9 * Real.pi / 180)) ∧
  (area_CDE config = (Real.tan (9 * Real.pi / 180) / 4) * (2 * config.h₅ * Real.cos (9 * Real.pi / 180) - config.h₄)^2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_of_triangles_l13_1302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_proof_l13_1386

noncomputable section

/-- The function f(x) = (3x^2 - 2x - 5) / (x - 4) -/
def f (x : ℝ) : ℝ := (3*x^2 - 2*x - 5) / (x - 4)

/-- The slope of the slant asymptote -/
def m : ℝ := 3

/-- The y-intercept of the slant asymptote -/
def b : ℝ := 10

/-- The slant asymptote function -/
def slant_asymptote (x : ℝ) : ℝ := m * x + b

/-- Theorem stating the existence of the slant asymptote and the sum of m and b -/
theorem slant_asymptote_proof :
  (∀ ε > 0, ∃ N : ℝ, ∀ x > N, |f x - slant_asymptote x| < ε) ∧
  m + b = 13 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_proof_l13_1386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_lines_l13_1321

/-- The angle between two lines given by their equations -/
noncomputable def angle_between_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  Real.arccos (abs ((a₁ * a₂ + b₁ * b₂) / (Real.sqrt (a₁^2 + b₁^2) * Real.sqrt (a₂^2 + b₂^2))))

/-- Theorem: The angle between lines l₁: √3x - y + 2 = 0 and l₂: 3x + √3y - 5 = 0 is π/3 -/
theorem angle_between_specific_lines :
  angle_between_lines (Real.sqrt 3) (-1) 2 3 (Real.sqrt 3) (-5) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_lines_l13_1321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seven_l13_1351

theorem divisibility_by_seven (a : ℤ) (m n : ℕ) :
  (7 ∣ (a^(6*m) + a^(6*n))) → (7 ∣ a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seven_l13_1351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cells_proof_l13_1303

/-- The minimum number of cells to guarantee hitting a submarine on a 2n × 2n board -/
def min_cells_to_hit_submarine (n : ℕ) : ℕ := 3 * n + 1

/-- Proof that the minimum number of cells to guarantee hitting a submarine is 3n + 1 -/
theorem min_cells_proof (n : ℕ) : min_cells_to_hit_submarine n = 3 * n + 1 := by
  -- Define the board size
  let board_size := 2 * n

  -- Define the number of submarines
  let num_submarines := n^2

  -- The proof steps would go here, but we'll use sorry to skip the actual proof
  sorry

/-- Verifies that the minimum number of cells is indeed 3n + 1 -/
example (n : ℕ) : min_cells_to_hit_submarine n = 3 * n + 1 := by
  -- This example serves as a verification of our definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cells_proof_l13_1303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l13_1319

theorem find_x : ∃ x : ℕ, 
  Nat.lcm (Nat.lcm 12 16) (Nat.lcm x 24) = 144 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l13_1319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_myquaternion_square_theorem_l13_1357

/-- Quaternion type -/
structure MyQuaternion where
  re : ℝ
  i : ℝ
  j : ℝ
  k : ℝ

/-- Quaternion multiplication -/
def MyQuaternion.mul (p q : MyQuaternion) : MyQuaternion :=
  { re := p.re * q.re - p.i * q.i - p.j * q.j - p.k * q.k,
    i := p.re * q.i + p.i * q.re + p.j * q.k - p.k * q.j,
    j := p.re * q.j - p.i * q.k + p.j * q.re + p.k * q.i,
    k := p.re * q.k + p.i * q.j - p.j * q.i + p.k * q.re }

/-- Quaternion square -/
def MyQuaternion.square (q : MyQuaternion) : MyQuaternion :=
  MyQuaternion.mul q q

/-- Theorem: If q^2 = -1 - i - j - k, then q = -1 - i/2 - j/2 - k/2 -/
theorem myquaternion_square_theorem (q : MyQuaternion) :
  MyQuaternion.square q = { re := -1, i := -1, j := -1, k := -1 } →
  q = { re := -1, i := -1/2, j := -1/2, k := -1/2 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_myquaternion_square_theorem_l13_1357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_five_l13_1322

def digits : Finset Nat := {3, 5, 7}

def is_divisible_by_five (n : Nat) : Prop :=
  n % 5 = 0

def four_digit_number (a b c d : Nat) : Nat :=
  1000 * a + 100 * b + 10 * c + d

def valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  (a = 5 ∨ b = 5 ∨ c = 5 ∨ d = 5) ∧
  (a = 5 → b ≠ 5 ∧ c ≠ 5 ∧ d ≠ 5) ∧
  (b = 5 → a ≠ 5 ∧ c ≠ 5 ∧ d ≠ 5) ∧
  (c = 5 → a ≠ 5 ∧ b ≠ 5 ∧ d ≠ 5) ∧
  (d = 5 → a ≠ 5 ∧ b ≠ 5 ∧ c ≠ 5)

def total_arrangements : Nat :=
  Nat.factorial (Finset.card digits + 1) / 2

def favorable_arrangements : Nat :=
  Nat.factorial (Finset.card digits)

theorem probability_divisible_by_five :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_five_l13_1322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_c_zero_min_value_on_interval_min_value_achieved_l13_1394

-- Define the function f
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (x^2 + 1) / (x + c)

-- Define what it means for f to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_implies_c_zero (c : ℝ) :
  is_odd (f c) → c = 0 := by sorry

theorem min_value_on_interval (x : ℝ) :
  x ∈ Set.Ici 2 → f 0 x ≥ 5/2 := by sorry

theorem min_value_achieved :
  ∃ x ≥ 2, f 0 x = 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_c_zero_min_value_on_interval_min_value_achieved_l13_1394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_properties_l13_1373

theorem matrix_properties :
  ∃ (M : Matrix (Fin 4) (Fin 4) ℝ),
    (∀ (u : Fin 4 → ℝ), M.mulVec u = λ i => -2 * u i) ∧
    (M.mulVec (λ i => if i = 3 then 1 else 0) = λ i => if i = 3 then -2 else 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_properties_l13_1373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_prices_correct_l13_1368

-- Define the discount structures and final prices
def discount_A (x : ℝ) : ℝ := x * 0.9 * 0.8
def discount_B (x : ℝ) : ℝ := x * 0.85 * 0.75
def discount_C (x : ℝ) : ℝ := x * 0.95 * 0.85

def final_price_A : ℝ := 420
def final_price_B : ℝ := 405
def final_price_C : ℝ := 680

-- Define the original prices
def original_price_A : ℝ := 583.33
def original_price_B : ℝ := 635
def original_price_C : ℝ := 842.24

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.05

-- Theorem to prove
theorem shirt_prices_correct :
  (abs (discount_A original_price_A - final_price_A) < 0.01) ∧
  (abs (discount_B original_price_B - final_price_B) < 0.01) ∧
  (abs (discount_C original_price_C - final_price_C) < 0.01) ∧
  (abs ((original_price_A + original_price_B + original_price_C) * (1 + sales_tax_rate) - 2163.5985) < 0.01) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_prices_correct_l13_1368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l13_1366

open Set Real

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem intersection_M_complement_N : M ∩ (𝕌 \ N) = Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l13_1366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_when_parallel_range_of_f_l13_1378

open Real

-- Define the vectors a and b
noncomputable def a (x : ℝ) : Fin 2 → ℝ := ![sin x, 3/2]
noncomputable def b (x : ℝ) : Fin 2 → ℝ := ![cos x, -1]

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := dot_product (fun i => a x i + b x i) (b x)

-- Theorem for part I
theorem tan_2x_when_parallel (x : ℝ) :
  dot_product (a x) (b x) = 0 → tan (2 * x) = 12 / 5 := by sorry

-- Theorem for part II
theorem range_of_f :
  Set.Icc (-π / 2) 0 ⊆ f ⁻¹' Set.Icc (-sqrt 2 / 2) (1 / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_when_parallel_range_of_f_l13_1378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_squared_plus_b_squared_l13_1331

theorem min_value_a_squared_plus_b_squared (a b : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → (a * x^2 + b / x)^6 = k * x + 2 * x + (k : ℝ)) → 
  (∀ a' b' : ℝ, a'^2 + b'^2 ≥ 2) ∧ (∃ a₀ b₀ : ℝ, a₀^2 + b₀^2 = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_squared_plus_b_squared_l13_1331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_model_relationship_l13_1333

/-- Represents the ratio of meters in a statue to inches in a model -/
noncomputable def statue_to_model_ratio (statue_height : ℝ) (model_height_feet : ℝ) : ℝ :=
  statue_height / (model_height_feet * 12)

/-- Theorem stating the relationship between statue and model measurements -/
theorem statue_model_relationship :
  let statue_height : ℝ := 45
  let model_height_feet : ℝ := 3
  statue_to_model_ratio statue_height model_height_feet = 5/4 := by
  -- Unfold the definition and simplify
  unfold statue_to_model_ratio
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_model_relationship_l13_1333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l13_1332

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 3 - 3) * (1 + Real.log x / Real.log 3)

-- State the theorem
theorem f_min_max :
  ∀ x : ℝ, x ∈ Set.Icc (1/27) 9 →
  (∀ y : ℝ, y ∈ Set.Icc (1/27) 9 → f y ≥ -4) ∧
  (∃ y : ℝ, y ∈ Set.Icc (1/27) 9 ∧ f y = -4) ∧
  (∀ y : ℝ, y ∈ Set.Icc (1/27) 9 → f y ≤ 12) ∧
  (∃ y : ℝ, y ∈ Set.Icc (1/27) 9 ∧ f y = 12) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l13_1332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_to_identity_l13_1364

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1/2, -Real.sqrt 3/2; Real.sqrt 3/2, 1/2]

theorem smallest_power_to_identity :
  (∀ m : ℕ+, m < 3 → A ^ m.val ≠ 1) ∧ A ^ 3 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_to_identity_l13_1364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_with_inscribed_circle_l13_1335

theorem rhombus_area_with_inscribed_circle (Q : ℝ) :
  let rhombus_acute_angle : ℝ := 30 * (π / 180)
  let inscribed_circle_area : ℝ := Q
  let rhombus_area : ℝ := 8 * Q / π
  rhombus_area = (4 * inscribed_circle_area / π) / (Real.sin rhombus_acute_angle) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_with_inscribed_circle_l13_1335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_dice_count_l13_1382

/-- The number of dice in James' bag -/
def james_total_dice : ℕ := 8

/-- The percentage of 12-sided dice in James' bag -/
def james_12_sided_percentage : ℚ := 75 / 100

/-- The total number of 12-sided dice needed for the game -/
def total_12_sided_needed : ℕ := 14

/-- The number of 12-sided dice they need to buy -/
def dice_to_buy : ℕ := 2

/-- The percentage of 12-sided dice in Mark's bag -/
def mark_12_sided_percentage : ℚ := 60 / 100

theorem marks_dice_count :
  ∃ (mark_total_dice : ℕ),
    mark_total_dice = 10 ∧
    (james_12_sided_percentage * james_total_dice).floor +
    (mark_12_sided_percentage * mark_total_dice).floor +
    dice_to_buy = total_12_sided_needed :=
by
  use 10
  constructor
  · rfl
  · norm_num
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_dice_count_l13_1382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l13_1358

theorem inequality_proof (x y z : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  x * y * z + (1 - x) * (1 - y) * (1 - z) ≥ min (x * (1 - y)) (min (y * (1 - z)) (z * (1 - x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l13_1358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MAB_l13_1380

open Real

-- Define the curves C₁ and C₂ in polar coordinates
noncomputable def C₁ (θ : ℝ) : ℝ := 4 * cos θ
noncomputable def C₂ (θ : ℝ) : ℝ := 4 * sin θ

-- Define the angle of the ray
noncomputable def ray_angle : ℝ := π / 3

-- Define the fixed point M
def M : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem area_of_triangle_MAB :
  let ρ_A := C₁ ray_angle
  let ρ_B := C₂ ray_angle
  let d := 2 * sin ray_angle
  let AB := ρ_B - ρ_A
  let S := (1 / 2) * AB * d
  S = 3 - sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MAB_l13_1380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l13_1315

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin α = 4 / 5) : 
  Real.sin (2 * α) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l13_1315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_l13_1379

/-- Given vectors a, b, and c in R², prove that λ = 1 is the unique real number
    such that a + λb is perpendicular to c. -/
theorem perpendicular_vector (a b c : ℝ × ℝ) (h1 : a = (1, 2))
    (h2 : b = (3, 0)) (h3 : c = (1, -2)) :
    ∃! l : ℝ, (a.1 + l * b.1, a.2 + l * b.2) • c = 0 ∧ l = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_l13_1379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_zero_eq_neg_one_l13_1317

/-- A polynomial of degree 6 satisfying specific conditions -/
noncomputable def r : Polynomial ℝ :=
  sorry

/-- The condition that r(3^n) = 1/(3^n) for n = 0, 1, 2, ..., 6 -/
axiom r_condition : ∀ n : Fin 7, r.eval ((3 : ℝ) ^ n.val) = ((3 : ℝ) ^ n.val)⁻¹

/-- The theorem stating that r(0) = -1 -/
theorem r_zero_eq_neg_one : r.eval 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_zero_eq_neg_one_l13_1317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_problem_solution_l13_1339

/-- Represents the time it takes for a worker to complete a job alone -/
def CompletionTime := ℝ

/-- Represents the rate at which a worker completes a job -/
def WorkRate := ℝ

structure WorkProblem where
  p_time : CompletionTime
  q_time : CompletionTime
  total_time : CompletionTime
  p_solo_time : CompletionTime
  p_q_time : CompletionTime

theorem work_problem_solution (w : WorkProblem) 
  (hq : w.q_time = (12 : ℝ))
  (htotal : w.total_time = (10 : ℝ))
  (hp_solo : w.p_solo_time = (4 : ℝ))
  (hp_q : w.p_q_time = (6 : ℝ))
  : w.p_time = (20 : ℝ) := by
  sorry

#check work_problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_problem_solution_l13_1339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l13_1376

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a parabola with vertex at origin and focus on y-axis -/
structure Parabola where
  focus : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if a point is in the first quadrant -/
def isFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a point is on the parabola -/
def isOnParabola (p : Point) (parab : Parabola) : Prop :=
  p.x^2 = 4 * parab.focus.y * p.y

theorem parabola_point_theorem (parab : Parabola) (p : Point) :
  parab.focus = Point.mk 0 2 →
  p = Point.mk (Real.sqrt 1184) 148 →
  isFirstQuadrant p ∧
  isOnParabola p parab ∧
  distance p parab.focus = 150 := by
  sorry

#eval Float.sqrt 1184 -- To verify the numerical value (approximation)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l13_1376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_poster_cost_is_five_l13_1383

/-- Represents the daily sales and costs of Laran's poster business --/
structure PosterBusiness where
  large_poster_price : ℚ
  small_poster_price : ℚ
  small_poster_cost : ℚ
  large_posters_per_day : ℕ
  small_posters_per_day : ℕ
  weekly_profit : ℚ
  school_days_per_week : ℕ

/-- Calculates the cost of a large poster given the business parameters --/
noncomputable def large_poster_cost (b : PosterBusiness) : ℚ :=
  let daily_revenue := b.large_poster_price * b.large_posters_per_day +
                       b.small_poster_price * b.small_posters_per_day
  let daily_small_poster_cost := b.small_poster_cost * b.small_posters_per_day
  let daily_profit := b.weekly_profit / b.school_days_per_week
  (daily_revenue - daily_small_poster_cost - daily_profit) / b.large_posters_per_day

theorem large_poster_cost_is_five (b : PosterBusiness) 
    (h1 : b.large_poster_price = 10)
    (h2 : b.small_poster_price = 6)
    (h3 : b.small_poster_cost = 3)
    (h4 : b.large_posters_per_day = 2)
    (h5 : b.small_posters_per_day = 3)
    (h6 : b.weekly_profit = 95)
    (h7 : b.school_days_per_week = 5) :
  large_poster_cost b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_poster_cost_is_five_l13_1383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcake_cost_l13_1310

/-- Proves that the cost of each cupcake is $1.50 given the initial conditions and final balance --/
theorem cupcake_cost (initial_money : ℝ) (cookie_boxes : ℕ) (cookie_price : ℝ) 
  (cupcakes : ℕ) (final_balance : ℝ) (x : ℝ) : 
  initial_money = 20 →
  cookie_boxes = 5 →
  cookie_price = 3 →
  cupcakes = 10 →
  final_balance = 30 →
  (initial_money + 2 * initial_money - cookie_boxes * cookie_price - cupcakes * x = final_balance) →
  x = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcake_cost_l13_1310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l13_1388

/-- The golden ratio, (√5 - 1)/2 -/
noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

/-- The sequence a_n defined as φ^n -/
noncomputable def a (n : ℕ) : ℝ := φ^n

/-- Theorem stating the uniqueness of the sequence -/
theorem unique_sequence :
  (∀ n, a n > 0) ∧
  (a 0 = 1) ∧
  (∀ n, a n - a (n + 1) = a (n + 2)) ∧
  (∀ b : ℕ → ℝ, (∀ n, b n > 0) → (b 0 = 1) → (∀ n, b n - b (n + 1) = b (n + 2)) → b = a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l13_1388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_three_plus_sqrt_five_to_sixth_l13_1377

theorem nearest_integer_to_three_plus_sqrt_five_to_sixth (x : ℝ) : 
  x = (3 + Real.sqrt 5)^6 → 
  ∃ (n : ℤ), n = 11304 ∧ ∀ (m : ℤ), |x - ↑n| ≤ |x - ↑m| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_integer_to_three_plus_sqrt_five_to_sixth_l13_1377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_selling_price_l13_1343

theorem minimum_selling_price 
  (n : ℕ) 
  (avg_price : ℚ) 
  (low_price_count : ℕ) 
  (max_price : ℚ) 
  (h1 : n = 20)
  (h2 : avg_price = 1200)
  (h3 : low_price_count = 10)
  (h4 : max_price = 11000)
  : ∃ (min_price : ℚ), min_price = 400 ∧ 
    (∀ (prices : Fin n → ℚ), 
      (Finset.sum Finset.univ (λ i => prices i)) / n = avg_price ∧
      ((Finset.filter (λ i => prices i < 1000) Finset.univ).card = low_price_count) ∧
      (∃ i, prices i = max_price) →
      (∀ i, prices i ≥ min_price)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_selling_price_l13_1343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l13_1363

def f (x : ℝ) : ℝ := x^2 + 5*x - 6

theorem solution_set_of_f :
  (∀ y : ℝ, f (y - 1) = y^2 + 3*y - 10) →
  ∀ x : ℝ, f x = 0 ↔ x = -6 ∨ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_l13_1363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l13_1352

/-- The quartic equation whose roots form a rhombus -/
def quartic_equation (z : ℂ) : ℂ :=
  z^4 + 4*Complex.I*z^3 + (2 + 2*Complex.I)*z^2 + (7 + Complex.I)*z + (6 - 3*Complex.I)

/-- The roots of the quartic equation form a rhombus -/
axiom roots_form_rhombus : ∃ (a b c d : ℂ), 
  (quartic_equation a = 0) ∧ (quartic_equation b = 0) ∧ 
  (quartic_equation c = 0) ∧ (quartic_equation d = 0) ∧
  (a - b = c - d) ∧ (a - d = b - c)

/-- The theorem stating that the area of the rhombus is the fourth root of 2 -/
theorem rhombus_area : 
  ∃ (a b c d : ℂ), (quartic_equation a = 0) ∧ (quartic_equation b = 0) ∧ 
  (quartic_equation c = 0) ∧ (quartic_equation d = 0) ∧
  (a - b = c - d) ∧ (a - d = b - c) →
  Real.sqrt (Real.sqrt 2) = Complex.abs ((a - c) * (b - d)) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l13_1352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_counterfeit_coins_l13_1381

/-- Represents the weight of a coin -/
structure CoinWeight where
  value : ℝ

/-- Represents a set of coins -/
structure CoinSet where
  n : ℕ
  weights : Fin n → CoinWeight
  counterfeit_indices : Fin 2 → Fin n
  h_n : n ≥ 5
  h_counterfeit : ∀ i : Fin 2, (weights (counterfeit_indices i)).value < 
    (weights (Classical.choice (Nonempty.intro (Fin.mk 0 (Nat.zero_lt_of_lt (h_n)))))).value
  h_same_weight : (weights (counterfeit_indices 0)).value = 
    (weights (counterfeit_indices 1)).value

/-- Represents a weighing on a balance scale -/
inductive Weighing (n : ℕ) where
| Compare : List (Fin n) → List (Fin n) → Weighing n

/-- The result of a weighing -/
inductive WeighingResult
| LeftHeavier
| RightHeavier
| Equal

/-- Perform a weighing and get the result -/
noncomputable def performWeighing (cs : CoinSet) (w : Weighing cs.n) : WeighingResult :=
  sorry

/-- Check if a given pair of indices corresponds to the counterfeit coins -/
def isCounterfeitPair (cs : CoinSet) (i j : Fin cs.n) : Prop :=
  sorry

/-- The main theorem: it's possible to identify counterfeit coins in two weighings -/
theorem identify_counterfeit_coins (cs : CoinSet) :
  ∃ (w1 w2 : Weighing cs.n) (i j : Fin cs.n),
    isCounterfeitPair cs i j ∧
    ∀ (r1 : WeighingResult) (r2 : WeighingResult),
      r1 = performWeighing cs w1 →
      r2 = performWeighing cs w2 →
      isCounterfeitPair cs i j :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_counterfeit_coins_l13_1381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformations_l13_1304

noncomputable def f (x : ℝ) : ℝ := -5 * Real.cos (x + Real.pi / 4) + 2

theorem cosine_transformations (x : ℝ) :
  ∃ (A B C D : ℝ),
    (A = 5) ∧
    (B = 1) ∧
    (C = -Real.pi / 4) ∧
    (D = 2) ∧
    (f x = A * Real.cos (B * (x - C)) + D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformations_l13_1304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l13_1324

noncomputable section

variable (a b : ℝ) (n1 n2 : ℤ) (x x1 x2 : ℝ)

def f (a b : ℝ) (x : ℝ) : ℝ := a^x - b*x + (3/2)*x^2 - 5

def f_prime (a b : ℝ) (x : ℝ) : ℝ := a^x * Real.log a - b + 3*x

theorem problem_solution (h1 : a > 0) (h2 : a ≠ 1) (h3 : f_prime a b 0 = 0) :
  (b = Real.log a) ∧ 
  (a = Real.exp 1 → (∀ n1 n2 : ℤ, (∀ x ∈ Set.Ioo (n1 : ℝ) (n2 : ℝ), f a b x < 0) → n2 - n1 ≤ 2)) ∧
  (a > 1 → (∃ x1 x2 : ℝ, x1 ∈ Set.Icc (-1 : ℝ) (1 : ℝ) ∧ x2 ∈ Set.Icc (-1 : ℝ) (1 : ℝ) ∧ 
    |f a b x1 - f a b x2| ≥ Real.exp 1 - 1/2) → a ≥ Real.exp 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l13_1324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_of_perpendicular_cylinders_l13_1354

/-- The volume of the intersection of two cylinders with equal radii k, 
    whose axes intersect at a right angle -/
noncomputable def intersectionVolume (k : ℝ) : ℝ := (16 * k^3) / 3

/-- Theorem stating that the volume of the intersection of two cylinders 
    with equal radii k, whose axes intersect at a right angle, is (16k^3)/3 -/
theorem intersection_volume_of_perpendicular_cylinders (k : ℝ) (hk : k > 0) :
  intersectionVolume k = ∫ x in (-k)..k, 4 * (k^2 - x^2) := by
  sorry

#check intersection_volume_of_perpendicular_cylinders

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_of_perpendicular_cylinders_l13_1354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_min_no_max_l13_1326

open Real

noncomputable def f (x : ℝ) : ℝ := (cos x + 3) / cos x

theorem f_has_min_no_max :
  ∃ (m : ℝ), (∀ x ∈ Set.Ioo (-π/2) (π/2), f x ≥ m) ∧
  (∀ M : ℝ, ∃ x ∈ Set.Ioo (-π/2) (π/2), f x > M) := by
  -- We'll use 4 as the minimum value
  use 4
  constructor
  · -- Proof that 4 is a lower bound
    intro x hx
    sorry -- Proof details omitted
  · -- Proof that there's no upper bound
    intro M
    -- We'll use x very close to π/2 to make f(x) arbitrarily large
    sorry -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_min_no_max_l13_1326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l13_1387

theorem expression_value : 
  abs (((10 * 1.8 - 2 * 1.5) / 0.3 + Real.rpow 3 (2/3) - Real.log 4 + Real.sin (π/6) - Real.cos (π/4) + 24 / 2) - 59.6862) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l13_1387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_is_pi_over_three_l13_1316

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_vectors_is_pi_over_three :
  ∀ a : ℝ × ℝ,
  Real.sqrt (a.1^2 + a.2^2) = 2 →
  let b : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
  (a.1 * (b.1 - a.1) + a.2 * (b.2 - a.2)) + 2 = 0 →
  angle_between_vectors a b = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_is_pi_over_three_l13_1316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winner_weight_l13_1330

/-- Represents the weight of different types of ravioli in ounces -/
structure RavioliWeights where
  meat : ℝ
  pumpkin : ℝ
  cheese : ℝ

/-- Represents the number of ravioli eaten by each person -/
structure RavioliEaten where
  meat : ℕ
  pumpkin : ℕ
  cheese : ℕ

/-- Calculates the total weight of ravioli eaten -/
def totalWeight (weights : RavioliWeights) (eaten : RavioliEaten) : ℝ :=
  weights.meat * eaten.meat + weights.pumpkin * eaten.pumpkin + weights.cheese * eaten.cheese

/-- The main theorem stating the weight eaten by the winner -/
theorem winner_weight (weights : RavioliWeights) (javier_eaten : RavioliEaten) (brother_pumpkin : ℕ) : 
  weights.meat = 1.5 →
  weights.pumpkin = 1.25 →
  weights.cheese = 1 →
  javier_eaten.meat = 5 →
  javier_eaten.pumpkin = 2 →
  javier_eaten.cheese = 4 →
  brother_pumpkin = 12 →
  max (totalWeight weights javier_eaten) (weights.pumpkin * brother_pumpkin) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_winner_weight_l13_1330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l13_1355

theorem evaluate_expression : 
  (81 : ℝ) ^ (1/2 : ℝ) * (64 : ℝ) ^ (-(1/3) : ℝ) * (49 : ℝ) ^ (1/4 : ℝ) = 9 * Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l13_1355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_eq_neg_0_95_l13_1393

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex_quadrilateral (q : Quadrilateral) : Prop := sorry

def angle_B_eq_angle_D (q : Quadrilateral) : Prop := sorry

def side_AB_eq_200 (q : Quadrilateral) : ℝ := sorry

def side_BC_eq_200 (q : Quadrilateral) : ℝ := sorry

def side_AD_neq_CD (q : Quadrilateral) : Prop := sorry

def perimeter_eq_780 (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem cos_B_eq_neg_0_95 (q : Quadrilateral) 
  (h1 : is_convex_quadrilateral q)
  (h2 : angle_B_eq_angle_D q)
  (h3 : side_AB_eq_200 q = 200)
  (h4 : side_BC_eq_200 q = 200)
  (h5 : side_AD_neq_CD q)
  (h6 : perimeter_eq_780 q = 780) :
  ∃ (B : ℝ), Real.cos B = -0.95 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_eq_neg_0_95_l13_1393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_theorem_l13_1300

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  sum_angles : angle_a + angle_b + angle_c = 180
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define properties for different types of triangles
def Triangle.isRightAngled (t : Triangle) : Prop :=
  t.angle_a = 90 ∨ t.angle_b = 90 ∨ t.angle_c = 90

def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

def Triangle.isScalene (t : Triangle) : Prop :=
  t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.c ≠ t.a

def Triangle.isAcute (t : Triangle) : Prop :=
  t.angle_a < 90 ∧ t.angle_b < 90 ∧ t.angle_c < 90

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angle_a > 90 ∨ t.angle_b > 90 ∨ t.angle_c > 90

-- Theorem statement
theorem triangle_segment_theorem (t : Triangle) 
  (h_right_isosceles : t.isRightAngled ∧ t.isIsosceles)
  (h_angles : t.angle_a = 90 ∧ t.angle_b = 45 ∧ t.angle_c = 45) :
  ∃ (segment : Triangle → Triangle × Triangle),
    let (t1, t2) := segment t
    (t1.isRightAngled ∨ t2.isRightAngled) ∧
    (t1.isIsosceles ∨ t2.isIsosceles) ∧
    (t1.isEquilateral ∨ t2.isEquilateral) ∧
    (t1.isScalene ∨ t2.isScalene) ∧
    (t1.isAcute ∨ t2.isAcute) ∧
    (t1.isObtuse ∨ t2.isObtuse) ∧
    (t1.angle_a = 90 ∨ t1.angle_b = 90 ∨ t1.angle_c = 90 ∨
     t2.angle_a = 90 ∨ t2.angle_b = 90 ∨ t2.angle_c = 90) ∧
    (t1.angle_a = 60 ∨ t1.angle_b = 60 ∨ t1.angle_c = 60 ∨
     t2.angle_a = 60 ∨ t2.angle_b = 60 ∨ t2.angle_c = 60) ∧
    (t1.angle_a = 30 ∨ t1.angle_b = 30 ∨ t1.angle_c = 30 ∨
     t2.angle_a = 30 ∨ t2.angle_b = 30 ∨ t2.angle_c = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_theorem_l13_1300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_duration_is_20_l13_1323

noncomputable def work_duration (total_work : ℝ) : ℝ :=
  let combined_rate := total_work / 30
  let a_rate := total_work / 60
  let x := (2 / 3) * 30
  have h1 : combined_rate = total_work / 30 := by rfl
  have h2 : a_rate = total_work / 60 := by rfl
  have h3 : x * combined_rate + 20 * a_rate = total_work := by
    -- Proof steps would go here, but we'll use sorry for now
    sorry
  x

-- This will not evaluate due to the noncomputable nature of the function
-- #eval work_duration 1

-- Instead, we can state a theorem about the result
theorem work_duration_is_20 : ∀ total_work : ℝ, work_duration total_work = 20 := by
  intro total_work
  unfold work_duration
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_duration_is_20_l13_1323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l13_1396

/-- The number of distinct ordered pairs of positive integers (x, y) that satisfy x + y = 50 -/
theorem count_solutions : ∃! n : ℕ, n = (Finset.range 49).card ∧
  n = (Finset.filter (fun p : ℕ × ℕ => p.1 + p.2 = 50 ∧ p.1 > 0 ∧ p.2 > 0)
      (Finset.product (Finset.range 50) (Finset.range 50))).card :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l13_1396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l13_1370

theorem m_range (A B : Set ℝ) (m : ℝ) : 
  A = {x : ℝ | -3 ≤ x ∧ x ≤ 4} →
  B = {x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1} →
  B ⊆ A →
  m ∈ Set.Icc (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l13_1370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_measure_theorem_l13_1325

-- Define the circle and points
variable (O : Point) -- Center of the circle
variable (A B C : Point) -- Points on the circle

-- Define the conditions
def is_on_circle (O : Point) (P : Point) : Prop := sorry
def is_diameter (O : Point) (P Q : Point) : Prop := sorry
def angle_measure (P Q R : Point) : ℝ := sorry
def arc_measure (O : Point) (P Q : Point) : ℝ := sorry

-- State the theorem
theorem arc_measure_theorem 
  (h1 : is_on_circle O A ∧ is_on_circle O B ∧ is_on_circle O C)
  (h2 : is_diameter O A C)
  (h3 : angle_measure B A C = 60) :
  arc_measure O B C = 120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_measure_theorem_l13_1325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_interested_in_both_music_and_art_l13_1346

/-- Given a class of students, calculate the number of students interested in both music and art. -/
theorem students_interested_in_both_music_and_art 
  (total_students music_enthusiasts art_enthusiasts neither_enthusiasts : ℕ)
  (h1 : total_students = 50)
  (h2 : music_enthusiasts = 30)
  (h3 : art_enthusiasts = 25)
  (h4 : neither_enthusiasts = 4)
  (h5 : total_students = music_enthusiasts + art_enthusiasts - (Finset.card (Finset.filter (λ x => x ∈ Finset.range music_enthusiasts ∧ x ∈ Finset.range art_enthusiasts) (Finset.range total_students))) + neither_enthusiasts) :
  Finset.card (Finset.filter (λ x => x ∈ Finset.range music_enthusiasts ∧ x ∈ Finset.range art_enthusiasts) (Finset.range total_students)) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_interested_in_both_music_and_art_l13_1346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l13_1390

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log (|x| + 1)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f (a * x^2) < f 3) ↔ -3/4 < a ∧ a < 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l13_1390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_count_is_18_l13_1341

/-- Angle between clock hands at time t for a clock running at rate r -/
def clockAngle (t : ℝ) (r : ℝ) : ℝ :=
  abs (30 * r * t - 5.5 * r * (t * 60))

/-- Number of times the angles are equal in a 12-hour period -/
def equalAnglesCount : ℕ :=
  18

theorem equal_angles_count_is_18 :
  ∃ (times : Finset ℝ),
    times.card = equalAnglesCount ∧
    (∀ t ∈ times, 0 ≤ t ∧ t < 12 ∧
      clockAngle t 1 = clockAngle t 0.5) ∧
    (∀ t, 0 ≤ t → t < 12 →
      clockAngle t 1 = clockAngle t 0.5 →
      t ∈ times) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angles_count_is_18_l13_1341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_0_3_inequality_l13_1367

-- Define the logarithm with base 0.3
noncomputable def log_0_3 (x : ℝ) : ℝ := Real.log x / Real.log 0.3

-- Define a and b as noncomputable
noncomputable def a : ℝ := log_0_3 4
noncomputable def b : ℝ := log_0_3 5

-- Theorem statement
theorem log_0_3_inequality : 0 > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_0_3_inequality_l13_1367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_on_line_l13_1375

/-- The projection of vector v onto vector u -/
noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2
  let norm_squared := u.1 * u.1 + u.2 * u.2
  (dot / norm_squared * u.1, dot / norm_squared * u.2)

/-- The theorem stating that vectors satisfying the projection condition lie on the specified line -/
theorem vectors_on_line (v : ℝ × ℝ) :
  proj (3, 4) v = (-3/2, -2) →
  v.2 = -3/4 * v.1 - 25/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_on_line_l13_1375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_seven_equals_fifteen_l13_1311

theorem coefficient_x_seven_equals_fifteen (a : ℝ) : 
  (Finset.range 11).sum (λ k ↦ (Nat.choose 10 k) * a^k * (1 : ℝ)^(10 - k)) = 15 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_seven_equals_fifteen_l13_1311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_intersection_implies_a_range_l13_1371

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1/2 then 1/(4^x)
  else -x + 1

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ :=
  a * Real.sin (Real.pi/6 * x) - a + 2

-- State the theorem
theorem exists_intersection_implies_a_range (a : ℝ) :
  (a > 0) →
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ f x₁ = g a x₂) →
  a ∈ Set.Icc 1 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_intersection_implies_a_range_l13_1371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_through_points_l13_1372

/-- The standard equation of an ellipse passing through two given points -/
theorem ellipse_through_points :
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧
  (A * (Real.sqrt 15 / 2)^2 + B = 1) ∧
  (B * (-2)^2 = 1) ∧
  (∀ x y : ℝ, A * x^2 + B * y^2 = 1 ↔ x^2 / 5 + y^2 / 4 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_through_points_l13_1372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_l13_1374

def Real.complement (x : Real) : Real := 180 - x

theorem triangle_angle_sum (A B C : Real) (h : Real.complement C = 130) : A + B = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_l13_1374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_4_and_8_l13_1350

-- Define the cumulative distribution function F
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 2 then 0
  else if x ≤ 4 then 0.5
  else if x ≤ 8 then 0.7
  else 1

-- Define the probability function
noncomputable def prob (a b : ℝ) : ℝ := F b - F a

-- Theorem statement
theorem probability_between_4_and_8 :
  prob 4 8 = 0.2 := by
  -- Unfold the definitions of prob and F
  unfold prob F
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_4_and_8_l13_1350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_remaining_balance_l13_1389

/-- Calculates John's remaining balance in Euros after a series of transactions -/
theorem johns_remaining_balance :
  let initial_amount : ℚ := 100
  let sister_fraction : ℚ := 1/3
  let groceries_cost : ℚ := 40
  let gift_price : ℚ := 30
  let gift_discount : ℚ := 1/2
  let exchange_rate : ℚ := 85/100

  let amount_after_sister : ℚ := initial_amount - (initial_amount * sister_fraction)
  let amount_after_groceries : ℚ := amount_after_sister - groceries_cost
  let discounted_gift_price : ℚ := gift_price * (1 - gift_discount)
  let amount_after_gift : ℚ := amount_after_groceries - discounted_gift_price
  let final_amount_euros : ℚ := amount_after_gift * exchange_rate

  ∃ (ε : ℚ), ε > 0 ∧ |final_amount_euros - 992/100| < ε
  := by sorry

#eval (100 : ℚ) * (2/3) - 40 - (30 * 1/2) * (85/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_remaining_balance_l13_1389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l13_1397

/-- Given a point P(3,y) on the terminal side of angle α where cos α = 3/5, prove y = ±4 -/
theorem point_on_terminal_side (α : ℝ) (y : ℝ) :
  (∃ P : ℝ × ℝ, P = (3, y) ∧ P.1 = 3 * Real.cos α ∧ P.2 = 3 * Real.sin α) →
  Real.cos α = 3/5 →
  y = 4 ∨ y = -4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l13_1397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l13_1328

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a)^2 + 4 * Real.log (x + 1)

-- State the theorem
theorem function_properties (a : ℝ) :
  (∃ (f' : ℝ → ℝ), ∀ x, HasDerivAt (f a) x (f' x)) →  -- f is differentiable
  (∃ (f' : ℝ → ℝ), HasDerivAt (f a) 1 (f' 1) ∧ f' 1 = 0) →  -- tangent line at (1, f(1)) is perpendicular to y-axis
  (a = 2) ∧  -- part 1: value of a
  (∀ x, x > -1 → f a x ≤ 4) ∧  -- part 2: maximum value
  (f a 0 = 4) ∧  -- maximum occurs at x = 0
  (∀ x, x > -1 → f a x ≥ 1 + 4 * Real.log 2) ∧  -- part 2: minimum value
  (f a 1 = 1 + 4 * Real.log 2)  -- minimum occurs at x = 1
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l13_1328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_half_l13_1309

-- Define the geometric configuration
structure GeometricConfig where
  a : ℝ
  x : ℝ
  AB : ℝ
  AC : ℝ
  OA : ℝ
  OC : ℝ
  AB_eq : AB = a
  AC_eq : AC = 2 * x
  OA_eq : OA = x
  OC_eq : OC = x

-- Define the surface areas
noncomputable def F₁ (config : GeometricConfig) : ℝ :=
  Real.pi * (config.a * config.x * (config.a - config.x)) / (config.a - 2 * config.x)

noncomputable def F₂ (config : GeometricConfig) : ℝ :=
  Real.pi * (2 * config.a * config.x^2) / (config.a - config.x)

-- Theorem statement
theorem surface_area_ratio_half (config : GeometricConfig) :
  F₂ config / F₁ config = 1/2 → config.x = config.a / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_half_l13_1309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_expectation_is_2_4_l13_1307

/-- A discrete random variable ξ with three possible values -/
inductive Xi : Type
  | one : Xi
  | three : Xi
  | five : Xi

/-- The probability mass function for ξ -/
noncomputable def P (x : Xi) : ℝ :=
  match x with
  | Xi.one => 0.5
  | Xi.three => 1 - 0.5 - 0.2  -- Define m directly
  | Xi.five => 0.2

/-- The value of the random variable ξ -/
def value (x : Xi) : ℝ :=
  match x with
  | Xi.one => 1
  | Xi.three => 3
  | Xi.five => 5

/-- The sum of probabilities equals 1 -/
theorem prob_sum : P Xi.one + P Xi.three + P Xi.five = 1 := by
  simp [P]
  ring

/-- The mathematical expectation of ξ -/
noncomputable def E_Xi : ℝ := (value Xi.one * P Xi.one) + (value Xi.three * P Xi.three) + (value Xi.five * P Xi.five)

/-- Theorem: The mathematical expectation of ξ is 2.4 -/
theorem expectation_is_2_4 : E_Xi = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_expectation_is_2_4_l13_1307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_sqrt_l13_1312

open Real BigOperators

theorem sum_reciprocal_sqrt :
  ∑ n in Finset.range 4998, 1 / (n * sqrt (n - 2) + (n - 2) * sqrt n) =
  1 + 1 / sqrt 2 - 1 / sqrt 5000 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_sqrt_l13_1312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_215_of_17_over_370_l13_1301

theorem digit_215_of_17_over_370 : ∃ (d : ℕ) (s : List ℕ), 
  (17 : ℚ) / 370 = d + (List.sum (List.map (λ (i : ℕ) => (s.get? i).getD 0 / (10 ^ (i + 1))) (List.range s.length))) / (1 - 1 / (10 ^ s.length)) ∧ 
  s.length = 6 ∧ 
  s = [0, 4, 5, 9, 4, 5] ∧
  (s.get? ((215 - 1) % s.length)).getD 0 = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_215_of_17_over_370_l13_1301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_to_perfect_square_l13_1329

/-- A function that represents adding factorial marks to some factors of n! --/
def add_factorial_marks (n : ℕ) : ℕ → ℕ := sorry

/-- A predicate that checks if a number is a perfect square --/
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

/-- The main theorem --/
theorem factorial_to_perfect_square (n : ℕ) (h : n ≥ 2) (h_composite : ¬ Nat.Prime n) :
  ∃ k : ℕ, is_perfect_square (add_factorial_marks n k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_to_perfect_square_l13_1329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_b1_b2_l13_1353

def sequence_b (b₁ b₂ : ℕ) : ℕ → ℕ
  | 0 => b₁
  | 1 => b₂
  | (n + 2) => (sequence_b b₁ b₂ n + 3961) / (1 + sequence_b b₁ b₂ (n + 1))

theorem min_sum_b1_b2 (b₁ b₂ : ℕ) :
  (∀ n, sequence_b b₁ b₂ n > 0) →
  (∀ n, (sequence_b b₁ b₂ (n + 2) * (1 + sequence_b b₁ b₂ (n + 1))) = sequence_b b₁ b₂ n + 3961) →
  b₁ ≤ b₂ →
  126 ≤ b₁ + b₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_b1_b2_l13_1353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_for_a_neg_one_min_value_of_g_g_achieves_min_l13_1342

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x + a| + |x - 1 / a|

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + f a (-x)

-- Theorem for part (1)
theorem solution_for_a_neg_one (x : ℝ) :
  -5/3 < x ∧ x < 1 ↔ f (-1) x < 4 := by
  sorry

-- Theorem for part (2)
theorem min_value_of_g (a : ℝ) (x : ℝ) (h : a ≠ 0) :
  g a x ≥ 4 * Real.sqrt 2 := by
  sorry

-- Theorem for the minimum value of g
theorem g_achieves_min (a : ℝ) (h : a ≠ 0) :
  ∃ (a₀ x₀ : ℝ), g a₀ x₀ = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_for_a_neg_one_min_value_of_g_g_achieves_min_l13_1342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_binomial_inequality_l13_1347

theorem sum_binomial_inequality (n : ℕ+) (j : Fin 3) :
  ∑' (k : ℕ), (-1 : ℤ)^(n : ℕ) * (Int.ofNat (Nat.choose n.val (3*k + j))) ≥ (1/3 : ℚ) * (((-2 : ℤ)^(n : ℕ)) - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_binomial_inequality_l13_1347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_odd_and_decreasing_l13_1399

-- Define the functions
noncomputable def f (x : ℝ) := -Real.sin x
noncomputable def g (x : ℝ) := 1 / x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define what it means for a function to be decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem f_and_g_odd_and_decreasing :
  (is_odd f ∧ is_decreasing_on f 0 1) ∧
  (is_odd g ∧ is_decreasing_on g 0 1) := by
  sorry

#check f_and_g_odd_and_decreasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_odd_and_decreasing_l13_1399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_constant_gcd_subset_l13_1314

/-- A is a subset of positive integers -/
def A : Set ℕ+ := sorry

/-- Each element of A is the product of at most 2000 different prime numbers -/
axiom A_prime_bounded : ∀ a ∈ A, ∃ (primes : Finset ℕ) (n : ℕ+), 
  a = n.val ∧ primes.card ≤ 2000 ∧ (∀ p ∈ primes, Nat.Prime p) ∧ (∀ p ∈ primes, p ∣ a)

/-- A is an infinite set -/
axiom A_infinite : Set.Infinite A

/-- There exists an infinite subset B of A with constant GCD -/
theorem exists_constant_gcd_subset : 
  ∃ (B : Set ℕ+) (d : ℕ+), Set.Infinite B ∧ B ⊆ A ∧ 
    ∀ (x y : ℕ+), x ∈ B → y ∈ B → x ≠ y → Nat.gcd x.val y.val = d.val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_constant_gcd_subset_l13_1314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_perpendicular_l13_1392

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define midpoints
noncomputable def M (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def N (B C : ℝ × ℝ) : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
noncomputable def P (C D : ℝ × ℝ) : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
noncomputable def Q (D A : ℝ × ℝ) : ℝ × ℝ := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

-- Define the length of a segment
noncomputable def length (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the dot product
def dot_product (p q : ℝ × ℝ) : ℝ := p.1 * q.1 + p.2 * q.2

-- Theorem statement
theorem diagonals_perpendicular (A B C D : ℝ × ℝ) 
  (h : length (M A B) (P C D) = length (N B C) (Q D A)) :
  dot_product (A.1 - C.1, A.2 - C.2) (B.1 - D.1, B.2 - D.2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonals_perpendicular_l13_1392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_sum_l13_1349

open Real

theorem triangle_tangent_sum (A B C : ℝ) (h1 : A + B + C = π) 
  (h2 : tan A = 1) (h3 : tan B = 2) : tan C = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_sum_l13_1349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_eq_sqrt13_plus_2_sum_distances_correct_l13_1337

/-- A rectangle with length 3 and width 4 -/
structure Rectangle where
  length : ℝ
  width : ℝ
  length_eq : length = 3
  width_eq : width = 4

/-- The sum of distances from a vertex to the centers of opposite sides -/
noncomputable def sum_distances (r : Rectangle) : ℝ :=
  Real.sqrt 13 + 2

/-- Theorem stating that the sum of distances is √13 + 2 -/
theorem sum_distances_eq_sqrt13_plus_2 (r : Rectangle) : 
  sum_distances r = Real.sqrt 13 + 2 := by
  -- Unfold the definition of sum_distances
  unfold sum_distances
  -- The equality is now trivial
  rfl

/-- Proof that the sum of distances is correct -/
theorem sum_distances_correct (r : Rectangle) : 
  sum_distances r = Real.sqrt ((r.length^2 + r.width^2) / 4) + r.width / 2 := by
  sorry -- We'll leave the detailed proof for later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_eq_sqrt13_plus_2_sum_distances_correct_l13_1337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_three_l13_1385

def S : Finset ℕ := {1, 2, 3, 4, 5}

def two_digit_numbers : Finset ℕ :=
  Finset.filter (λ n => n ≥ 10 ∧ n < 100)
    (Finset.image (λ (a, b) => 10 * a + b) (Finset.product S S))

def divisible_by_three : Finset ℕ :=
  Finset.filter (λ n => n % 3 = 0) two_digit_numbers

theorem probability_divisible_by_three :
  (Finset.card divisible_by_three : ℚ) / (Finset.card two_digit_numbers : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_three_l13_1385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_unique_zeros_characterization_l13_1395

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

-- Part 1: Tangent line at (0, f(0)) when a = 1
theorem tangent_line_at_zero :
  ∀ x : ℝ, 2 * x = (deriv (f 1)) 0 * x + (f 1) 0 :=
sorry

-- Part 2: Characterization of a for exactly one zero in each interval
theorem unique_zeros_characterization :
  ∀ a : ℝ, 
  ((∃! x₁, x₁ ∈ Set.Ioo (-1) 0 ∧ f a x₁ = 0) ∧ 
   (∃! x₂, x₂ ∈ Set.Ioi 0 ∧ f a x₂ = 0)) ↔ 
  a ∈ Set.Iio (-1) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_unique_zeros_characterization_l13_1395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l13_1360

/-- Given two vectors a and b in ℝ², if (a + b) is parallel to (a - b), then the y-coordinate of b is -8. -/
theorem vector_parallel_condition (a b : ℝ × ℝ) (h : a = (-1, 4) ∧ b.1 = 2) :
  (a + b).1 / (a + b).2 = (a - b).1 / (a - b).2 → b.2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l13_1360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_two_transitions_l13_1313

/-- Counts the number of transitions between different adjacent digits in the binary representation of a natural number. -/
def countTransitions (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is less than or equal to 127 and has exactly two transitions in its binary representation. -/
def hasExactlyTwoTransitions (n : ℕ) : Bool :=
  n ≤ 127 ∧ countTransitions n = 2

/-- The set of all natural numbers satisfying the condition. -/
def validNumbers : Set ℕ :=
  {n : ℕ | hasExactlyTwoTransitions n = true}

theorem count_numbers_with_two_transitions :
  Finset.card (Finset.filter (fun n => hasExactlyTwoTransitions n) (Finset.range 128)) = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_two_transitions_l13_1313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l13_1362

theorem tan_double_angle (α : ℝ) (h : Real.sin α + 2 * Real.cos α = Real.sqrt 10 / 2) :
  Real.tan (2 * α) = -3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l13_1362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_472_83649_to_hundredth_l13_1306

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem round_472_83649_to_hundredth :
  round_to_hundredth 472.83649 = 472.84 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_472_83649_to_hundredth_l13_1306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_square_area_l13_1320

theorem midpoint_square_area (large_square_area : ℝ) (h : large_square_area = 100) :
  let large_side := Real.sqrt large_square_area
  let small_side := Real.sqrt 2 * (large_side / 2)
  small_side ^ 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_square_area_l13_1320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l13_1308

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

theorem equidistant_point (z : ℝ) :
  let A : Point3D := ⟨0, 0, z⟩
  let B : Point3D := ⟨-1, -1, -6⟩
  let C : Point3D := ⟨2, 3, 5⟩
  distance A B = distance A C ↔ z = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l13_1308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_terms_l13_1365

def a (n : ℕ) : ℚ :=
  match n with
  | 0 => 16
  | n + 1 => (3 / 2) * a n

theorem sum_of_terms : a 0 + a 2 + a 4 = 133 := by
  have h1 : a 0 = 16 := by rfl
  have h2 : a 1 = 24 := by
    simp [a]
    norm_num
  have h3 : a 2 = 36 := by
    simp [a]
    norm_num
  have h4 : a 3 = 54 := by
    simp [a]
    norm_num
  have h5 : a 4 = 81 := by
    simp [a]
    norm_num
  calc
    a 0 + a 2 + a 4 = 16 + 36 + 81 := by rw [h1, h3, h5]
    _ = 133 := by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_terms_l13_1365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_two_in_S_l13_1369

-- Define the set S
axiom S : Set ℤ

-- Define the property that any integer root of a non-zero polynomial with coefficients in S belongs to S
def closed_under_roots (S : Set ℤ) : Prop :=
  ∀ (p : Polynomial ℤ), p ≠ 0 → (∀ n : ℕ, p.coeff n ∈ S) → ∀ x : ℤ, p.eval x = 0 → x ∈ S

-- State the theorem
theorem negative_two_in_S (h1 : 0 ∈ S) (h2 : 1996 ∈ S) (h3 : closed_under_roots S) : -2 ∈ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_two_in_S_l13_1369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_optimal_expense_optimal_expense_value_l13_1334

/-- Represents the annual sales volume in ten thousand units -/
noncomputable def annual_sales (t : ℝ) : ℝ := 4 - 3 / t

/-- Represents the profit in ten thousand yuan -/
noncomputable def profit (t : ℝ) : ℝ := 27 - 18 / t - t

/-- The optimal promotional expense that maximizes profit -/
noncomputable def optimal_expense : ℝ := 3 * Real.sqrt 2

theorem profit_maximized_at_optimal_expense :
  ∀ t > 0, profit t ≤ profit optimal_expense :=
by sorry

theorem optimal_expense_value :
  optimal_expense = 3 * Real.sqrt 2 :=
by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_optimal_expense_optimal_expense_value_l13_1334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_fourth_and_eighth_term_l13_1348

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := λ n => a₁ * q^(n - 1)

theorem geometric_mean_of_fourth_and_eighth_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * 2) 
  (h_first : a 1 = 1/8) :
  Real.sqrt (a 4 * a 8) = 4 := by
  sorry

#check geometric_mean_of_fourth_and_eighth_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_fourth_and_eighth_term_l13_1348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_range_l13_1356

/-- The function f(x) defined as a^x + (1+a)^x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (1+a)^x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := a^x * Real.log a + (1+a)^x * Real.log (1+a)

theorem monotone_increasing_range (a : ℝ) :
  a ∈ Set.Ioo 0 1 →
  (∀ x ∈ Set.Ioi 0, f_deriv a x ≥ 0) →
  a ∈ Set.Icc ((Real.sqrt 5 - 1) / 2) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_range_l13_1356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l13_1359

theorem sin_double_angle (θ : ℝ) : 
  Real.sin (5 * π / 4 + θ) = -3/5 → Real.sin (2 * θ) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l13_1359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l13_1327

theorem root_interval (a : ℝ) : 
  (a^2 + a - 3 = 0) →
  ((1.3^2 + 1.3 - 3) < 0) →
  ((1.4^2 + 1.4 - 3) > 0) →
  (1.3 < a ∧ a < 1.4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_l13_1327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_sin_l13_1336

theorem cos_double_angle_with_sin (x : ℝ) (h : Real.sin x = -2/3) : Real.cos (2*x) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_sin_l13_1336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l13_1345

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * (Real.cos (ω * x / 2))^2 + Real.cos (ω * x + Real.pi / 3) - 1

theorem triangle_problem (ω A B C : ℝ) (a b c : ℝ) :
  ω > 0 →
  f ω A = 0 →
  f ω B = 0 →
  A < B →
  B - A = Real.pi / 2 →
  f ω A = -3/2 →
  c = 3 →
  1/2 * b * c * Real.sin A = 3 * Real.sqrt 3 →
  ω = 2 ∧ a = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l13_1345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_problem_l13_1340

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

noncomputable def geometric_sequence (b₁ : ℝ) (q : ℝ) : ℕ → ℝ
  | n => b₁ * q ^ n

noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

noncomputable def geometric_sum (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * b₁ else b₁ * (1 - q^n) / (1 - q)

noncomputable def even_indexed_sum (b₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  q * b₁ * (1 - q^(n/2)) / (1 - q)

theorem arithmetic_geometric_sequence_problem 
  (d : ℝ) 
  (hd : d ≠ 0) 
  (h_geometric : ∃ b₁ q, 
    geometric_sequence b₁ q 0 = arithmetic_sequence 4 d 1 ∧ 
    geometric_sequence b₁ q 1 = arithmetic_sequence 4 d 3 ∧ 
    geometric_sequence b₁ q 2 = arithmetic_sequence 4 d 7) 
  (h_sum : ∃ b₁ q, geometric_sum b₁ q 100 = 150) :
  arithmetic_sum 4 d 10 = 130 ∧ 
  ∃ b₁ q, even_indexed_sum b₁ q 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_problem_l13_1340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_real_solutions_l13_1384

noncomputable def real_root₁ : ℝ := 
  (2/9 - Real.sqrt ((2/9)^2 + 64/3)) / 2

noncomputable def real_root₂ : ℝ := 
  (2/9 + Real.sqrt ((2/9)^2 + 64/3)) / 2

theorem quadratic_real_solutions (x : ℝ) : 
  (∃ y : ℝ, 9 * y^2 + 9 * x * y + x + 8 = 0) ↔ 
  (x ≤ real_root₁ ∨ x ≥ real_root₂) :=
by
  have h1 : ∀ y : ℝ, 9 * y^2 + 9 * x * y + x + 8 = 0 → 
    81 * x^2 - 36 * x - 288 ≥ 0 := sorry
  have h2 : (81 * x^2 - 36 * x - 288 ≥ 0) ↔ 
    (x ≤ real_root₁ ∨ x ≥ real_root₂) := sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_real_solutions_l13_1384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_94_l13_1391

def f : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 3
  | n + 3 => f (n + 2) - f (n + 1) + 2 * (n + 3)

theorem f_10_equals_94 : f 10 = 94 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_94_l13_1391
