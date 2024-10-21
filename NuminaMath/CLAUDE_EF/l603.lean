import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_49_l603_60314

def divisors_of_245 : List Nat := [5, 7, 35, 49, 245]

def has_common_factor_greater_than_one (a b : Nat) : Prop :=
  ∃ (f : Nat), f > 1 ∧ f ∣ a ∧ f ∣ b

def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i : Nat, i < arr.length → has_common_factor_greater_than_one (arr[i]!) (arr[(i + 1) % arr.length]!)

theorem sum_of_adjacent_to_49 (arr : List Nat) :
  arr.toFinset = divisors_of_245.toFinset →
  is_valid_arrangement arr →
  (∃ i : Nat, i < arr.length ∧ arr[i]! = 49) →
  ∃ i : Nat, i < arr.length ∧ arr[i]! = 49 ∧
    (arr[(i - 1 + arr.length) % arr.length]! + arr[(i + 1) % arr.length]! = 280) :=
by sorry

#check sum_of_adjacent_to_49

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_49_l603_60314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundred_points_possible_hundred_points_impossible_main_result_l603_60392

/-- Represents the number of points on a cube that map into themselves under all rotations. -/
def RotationInvariantPoints (n : ℕ) : Prop :=
  n % 6 = 0 ∨ n % 6 = 2

/-- 200 points can be placed on a cube in a rotation-invariant manner. -/
theorem two_hundred_points_possible : RotationInvariantPoints 200 := by
  unfold RotationInvariantPoints
  apply Or.inr
  rfl

/-- 100 points cannot be placed on a cube in a rotation-invariant manner. -/
theorem hundred_points_impossible : ¬RotationInvariantPoints 100 := by
  unfold RotationInvariantPoints
  intro h
  cases h with
  | inl h₁ => norm_num at h₁
  | inr h₂ => norm_num at h₂

/-- The main theorem stating which of 100 or 200 points are possible. -/
theorem main_result : RotationInvariantPoints 200 ∧ ¬RotationInvariantPoints 100 := by
  exact ⟨two_hundred_points_possible, hundred_points_impossible⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundred_points_possible_hundred_points_impossible_main_result_l603_60392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l603_60379

noncomputable def z : ℂ := (Complex.exp (Real.pi / 4 * Complex.I))^2 / (1 - Complex.I)

def first_quadrant (w : ℂ) : Prop := w.re > 0 ∧ w.im > 0
def second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0
def third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0
def fourth_quadrant (w : ℂ) : Prop := w.re > 0 ∧ w.im < 0

theorem z_in_second_quadrant : second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l603_60379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banquet_hall_tables_l603_60334

/-- Represents the number of tables in the banquet hall -/
def num_tables : ℕ := 21

/-- Represents the number of chairs per table -/
def chairs_per_table : ℕ := 8

/-- Represents the number of legs per chair -/
def legs_per_chair : ℕ := 4

/-- Represents the number of legs per table -/
def legs_per_table : ℕ := 6

/-- Represents the total number of legs (tables and chairs combined) -/
def total_legs : ℕ := 798

/-- Theorem stating that the number of tables is 21 given the conditions -/
theorem banquet_hall_tables : num_tables = 21 := by
  rfl

#check banquet_hall_tables

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banquet_hall_tables_l603_60334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbered_triangles_l603_60322

/-- 
Given an equilateral triangle divided into n^2 equal smaller equilateral triangles,
where m of these triangles are numbered consecutively such that triangles with 
consecutive numbers have adjacent sides, the maximum value of m is n^2 - n + 1.
-/
theorem max_numbered_triangles (n : ℕ) (m : ℕ) 
  (h1 : n > 0) 
  (h2 : m > 0) 
  (h3 : ∃ (triangle : Type) (numbering : Fin m → triangle), 
    (∀ i j : Fin m, i.val + 1 = j.val → ∃ (adjacent : triangle → triangle → Prop), adjacent (numbering i) (numbering j)) ∧
    (∀ t : triangle, ∃ i : Fin m, numbering i = t ∨ ¬(∃ j : Fin m, numbering j = t))) :
  m ≤ n^2 - n + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbered_triangles_l603_60322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_values_l603_60376

noncomputable def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

noncomputable def line_slope (A B : ℝ) : ℝ := -A / B

theorem parallel_lines_a_values :
  ∀ a : ℝ,
  let l1_slope := line_slope 1 (2 * a)
  let l2_slope := line_slope (a + 1) (-a)
  parallel_lines l1_slope l2_slope →
  a = -3/2 ∨ a = 0 := by
  sorry

#check parallel_lines_a_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_values_l603_60376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_values_l603_60378

/-- The equation of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

/-- Theorem: Given an ellipse with equation x^2/9 + y^2/(4+k) = 1 and eccentricity 4/5,
    the value of k is either -19/25 or 21 -/
theorem ellipse_k_values (k : ℝ) :
  (∀ x y : ℝ, ellipse_equation x y 3 (Real.sqrt (4 + k))) →
  eccentricity 3 (Real.sqrt (4 + k)) = 4/5 →
  k = -19/25 ∨ k = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_values_l603_60378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_size_ranking_l603_60380

/-- Represents the cost and quantity of a bottle size --/
structure BottleSize where
  cost : ℝ
  quantity : ℝ

/-- Calculates the discounted cost per ounce --/
noncomputable def discountedCostPerOunce (b : BottleSize) : ℝ :=
  (0.9 * b.cost) / b.quantity

/-- Theorem stating the ranking of bottle sizes based on discounted cost per ounce --/
theorem bottle_size_ranking (s m l : BottleSize)
  (h_m_cost : m.cost = 1.4 * s.cost)
  (h_m_quantity : m.quantity = 1.3 * s.quantity)
  (h_l_cost : l.cost = 1.2 * m.cost)
  (h_l_quantity : l.quantity = 1.5 * m.quantity) :
  discountedCostPerOunce l < discountedCostPerOunce s ∧
  discountedCostPerOunce s < discountedCostPerOunce m :=
by
  sorry

#check bottle_size_ranking

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_size_ranking_l603_60380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_mileage_l603_60353

/-- Calculates the average gas mileage for a two-leg trip -/
noncomputable def average_gas_mileage (distance1 : ℝ) (mpg1 : ℝ) (distance2 : ℝ) (mpg2 : ℝ) : ℝ :=
  (distance1 + distance2) / (distance1 / mpg1 + distance2 / mpg2)

/-- Proves that the average gas mileage for the given trip is approximately 18.333 mpg -/
theorem trip_average_mileage :
  let trip_avg := average_gas_mileage 150 25 180 15
  ∃ ε > 0, |trip_avg - 18.333| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_mileage_l603_60353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_square_prime_1001_n_l603_60393

theorem no_square_prime_1001_n : ¬ ∃ (n : ℕ), n ≥ 2 ∧ ∃ (p : ℕ), Nat.Prime p ∧ n^3 + 1 = p^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_square_prime_1001_n_l603_60393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_l603_60300

/-- Given a segment AC with point B such that AB = 14 cm and BC = 28 cm,
    and semicircles constructed on AB, BC, and AC in one half-plane,
    the radius of the circle tangent to all three semicircles is 6 cm. -/
theorem tangent_circle_radius (A B C : EuclideanSpace ℝ (Fin 2))
    (h_AB : dist A B = 14)
    (h_BC : dist B C = 28)
    (semicircle_AB : Set (EuclideanSpace ℝ (Fin 2)))
    (semicircle_BC : Set (EuclideanSpace ℝ (Fin 2)))
    (semicircle_AC : Set (EuclideanSpace ℝ (Fin 2)))
    (h_halfplane : Set (EuclideanSpace ℝ (Fin 2))) :
    ∃ (P : EuclideanSpace ℝ (Fin 2)) (r : ℝ),
      r = 6 ∧ 
      (∀ x ∈ semicircle_AB, dist x P = r + 7) ∧
      (∀ x ∈ semicircle_BC, dist x P = r + 14) ∧
      (∀ x ∈ semicircle_AC, dist x P = r + 21) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_l603_60300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_bar_placement_optimal_l603_60342

/-- Represents a placement of bars in storage rooms -/
def BarPlacement := List Nat

/-- Checks if a list of room numbers is valid (within 1 to 121 and sorted) -/
def isValidPlacement (placement : BarPlacement) : Prop :=
  List.Sorted (· ≤ ·) placement ∧ placement.all (λ x => 1 ≤ x ∧ x ≤ 121)

/-- Calculates the minimum distance between any two bars -/
def minDistance (placement : BarPlacement) : Nat :=
  (placement.zip placement.tail).foldl (λ acc (a, b) => min acc (b - a)) (placement.length + 1)

/-- The given placement of the first 5 bars -/
def initialPlacement : BarPlacement := [1, 31, 61, 91, 121]

/-- Possible placements for the 6th bar -/
def sixthBarPlacements : List Nat := [12, 34, 56, 78]

/-- Theorem: The 6th bar can be placed in rooms 12, 34, 56, or 78 to maximize the minimum distance -/
theorem sixth_bar_placement_optimal :
  isValidPlacement initialPlacement →
  ∀ (room : Nat), room ∈ sixthBarPlacements →
  ∀ (otherRoom : Nat), 1 ≤ otherRoom ∧ otherRoom ≤ 121 →
  otherRoom ∉ sixthBarPlacements →
  minDistance (room :: initialPlacement) ≥ minDistance (otherRoom :: initialPlacement) := by
  sorry

#eval minDistance initialPlacement
#eval sixthBarPlacements.map (λ room => minDistance (room :: initialPlacement))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_bar_placement_optimal_l603_60342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_12_miles_l603_60387

/-- The distance Bob walked when he met Yolanda -/
noncomputable def bobsDistance (totalDistance : ℝ) (yolandaSpeed : ℝ) (bobSpeed : ℝ) (bobDelay : ℝ) : ℝ :=
  let meetingTime := (totalDistance - yolandaSpeed * bobDelay) / (yolandaSpeed + bobSpeed)
  bobSpeed * meetingTime

/-- Theorem stating that Bob walked 12 miles when he met Yolanda -/
theorem bob_walked_12_miles :
  bobsDistance 24 3 4 1 = 12 := by
  sorry

-- Use #eval only for computable functions
-- #eval bobsDistance 24 3 4 1

-- Instead, we can use the following to check the result:
#check bob_walked_12_miles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_12_miles_l603_60387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1993rd_term_l603_60361

/-- Sequence function: returns the nth term of the sequence -/
def sequenceFunc (n : ℕ) : ℕ := sorry

/-- Sum of first n integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sequence_1993rd_term :
  ∃ (k : ℕ), sequenceFunc 1993 = 63 ∧ 63 % 5 = 3 ∧ sum_first_n 62 ≤ 1993 ∧ 1993 < sum_first_n 63 := by
  sorry

#check sequence_1993rd_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1993rd_term_l603_60361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_α_value_complex_expression_value_l603_60383

-- Define the angle α and point P
variable (α : Real)
variable (P : Real × Real)

-- Define the conditions
axiom x_negative : P.1 < 0
axiom terminal_side : P.2 = -1
axiom cos_α : Real.cos α = (Real.sqrt 5 / 5) * P.1

-- State the theorems to be proved
theorem tan_α_value : Real.tan α = 1/2 := by sorry

theorem complex_expression_value :
  (1 - Real.cos (2 * α)) / (Real.sqrt 2 * Real.cos (α - π/4) + Real.sin (π + α)) = -(Real.sqrt 5 / 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_α_value_complex_expression_value_l603_60383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_bc_l603_60374

theorem least_possible_bc (AB AC DC BD BC : ℝ) : 
  AB = 5 → AC = 12 → DC = 8 → BD = 20 → 
  AB + AC > BC → BD + DC > BC → 
  ∃ (n : ℤ), BC = n → 
  (∀ x : ℤ, x < BC → x < 13) :=
by
  sorry

#check least_possible_bc

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_bc_l603_60374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_perimeter_l603_60303

/-- Represents a rhombus with given diagonal lengths and side length -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  side : ℝ

/-- Calculates the area of a rhombus -/
noncomputable def area (r : Rhombus) : ℝ :=
  (r.diagonal1 * r.diagonal2) / 2

/-- Calculates the perimeter of a rhombus -/
noncomputable def perimeter (r : Rhombus) : ℝ :=
  4 * r.side

theorem rhombus_area_and_perimeter (r : Rhombus)
    (h1 : r.diagonal1 = 18)
    (h2 : r.diagonal2 = 16)
    (h3 : r.side = 10) :
    area r = 144 ∧ perimeter r = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_perimeter_l603_60303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_shift_without_K_change_l603_60340

/-- Represents the equilibrium constant -/
def K : ℝ → ℝ := sorry

/-- Represents the equilibrium state -/
def equilibrium_state : Type := ℝ

/-- Represents a shift in equilibrium -/
def equilibrium_shift (initial final : equilibrium_state) : Prop := sorry

/-- Represents temperature -/
def temperature : ℝ := sorry

/-- K is a function of temperature only -/
axiom K_depends_on_temperature :
  ∀ t₁ t₂, t₁ = t₂ → K t₁ = K t₂

/-- Equilibrium can shift due to changes in factors other than temperature -/
axiom equilibrium_shift_non_temperature :
  ∃ initial final : equilibrium_state,
    equilibrium_shift initial final ∧
    initial = final

/-- Theorem: It's possible for equilibrium to shift without changing K -/
theorem equilibrium_shift_without_K_change :
  ∃ initial final : equilibrium_state,
    equilibrium_shift initial final ∧
    K initial = K final := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_shift_without_K_change_l603_60340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_after_13_years_l603_60375

/-- Calculates the total amount of principal and interest for annual deposits over a period of time with compound interest. -/
noncomputable def totalAmount (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  (a / r) * ((1 + r) ^ (n + 1) - r - 1)

/-- Theorem: The total amount of principal and interest after 13 years of annual deposits
    is equal to (a/r) * ((1+r)^14 - r - 1), where 'a' is the annual deposit amount and 'r' is the annual interest rate. -/
theorem total_amount_after_13_years (a r : ℝ) (h1 : r ≠ 0) (h2 : r > 0) (h3 : a > 0) :
  totalAmount a r 13 = (a / r) * ((1 + r)^14 - r - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_after_13_years_l603_60375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_ratio_l603_60343

noncomputable def ω : ℂ := -1/2 + (Complex.I * Real.sqrt 3) / 2

theorem cube_root_ratio {a b c : ℂ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h1 : a/b = b/c) (h2 : b/c = c/a) : 
  (a + b - c) / (a - b + c) = 1 ∨ 
  (a + b - c) / (a - b + c) = ω ∨ 
  (a + b - c) / (a - b + c) = ω^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_ratio_l603_60343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_increase_cube_to_cuboids_l603_60333

/-- The surface area after cutting a cube into cuboids -/
noncomputable def surface_area_after_cutting (edge_length : ℝ) (num_cuboids : ℕ) : ℝ :=
  sorry

/-- The surface area increase when cutting a cube into cuboids -/
theorem surface_area_increase_cube_to_cuboids :
  ∀ (edge_length : ℝ) (num_cuboids : ℕ),
  edge_length = 5 →
  num_cuboids = 3 →
  (6 * edge_length^2) + 100 = 
    surface_area_after_cutting edge_length num_cuboids :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_increase_cube_to_cuboids_l603_60333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_properties_l603_60365

noncomputable section

def A : Fin 2 → ℝ := ![1, 0]
def B : Fin 2 → ℝ := ![2, 2]
def C : Fin 2 → ℝ := ![3, 0]

def M : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.of !![Real.sqrt 2 / 2, Real.sqrt 2 / 2; -Real.sqrt 2 / 2, Real.sqrt 2 / 2]

def clockwise_rotation_45 (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  M.mulVec v

def area_triangle (p q r : Fin 2 → ℝ) : ℝ :=
  abs ((q 0 - p 0) * (r 1 - p 1) - (r 0 - p 0) * (q 1 - p 1)) / 2

theorem rotation_matrix_properties :
  (M = Matrix.of !![Real.sqrt 2 / 2, Real.sqrt 2 / 2; -Real.sqrt 2 / 2, Real.sqrt 2 / 2]) ∧
  (M⁻¹ = Matrix.of !![Real.sqrt 2 / 2, -Real.sqrt 2 / 2; Real.sqrt 2 / 2, Real.sqrt 2 / 2]) ∧
  (area_triangle (clockwise_rotation_45 A) (clockwise_rotation_45 B) (clockwise_rotation_45 C) = 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_matrix_properties_l603_60365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l603_60369

noncomputable def floor (x : ℝ) := ⌊x⌋

theorem divisibility_condition (n : ℕ) : 
  (∃ k : ℕ, 2 * n = k * (1 + floor (Real.sqrt (2 * ↑n)))) ↔ 
  (∃ x : ℕ, n = x * (x + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l603_60369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_disjoint_and_sum_range_l603_60307

/-- Line l in parametric form -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t + 4 * Real.sqrt 2)

/-- Curve C in polar form -/
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.cos (θ + Real.pi / 4)

/-- Curve C in Cartesian form -/
def on_curve_C (x y : ℝ) : Prop :=
  (x - Real.sqrt 2 / 2)^2 + (y + Real.sqrt 2 / 2)^2 = 1

theorem line_curve_disjoint_and_sum_range :
  (∀ t x y : ℝ, line_l t ≠ (x, y) ∨ ¬on_curve_C x y) ∧
  (∀ x y : ℝ, on_curve_C x y → -Real.sqrt 2 ≤ x + y ∧ x + y ≤ Real.sqrt 2) := by
  sorry

#check line_curve_disjoint_and_sum_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_disjoint_and_sum_range_l603_60307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_prime_in_sequence_l603_60390

def sequenceDigits (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 2
  | 1 => 0
  | 2 => 1
  | _ => 6

def sequenceNumber : ℕ → ℕ
  | 0 => 2
  | n + 1 => sequenceNumber n * 10 + sequenceDigits n

def isPrime (n : ℕ) : Prop := Nat.Prime n

theorem one_prime_in_sequence :
  ∃! k : ℕ, isPrime (sequenceNumber k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_prime_in_sequence_l603_60390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l603_60305

-- Define the function f(x) = x + 4/(x-1)
noncomputable def f (x : ℝ) : ℝ := x + 4 / (x - 1)

-- State the theorem
theorem min_value_of_f :
  ∀ x : ℝ, x > 1 → f x ≥ 5 ∧ ∃ x₀ : ℝ, x₀ > 1 ∧ f x₀ = 5 :=
by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l603_60305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_eccentricity_special_ellipse_equation_l603_60330

/-- An ellipse with a point satisfying specific conditions -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  P : ℝ × ℝ
  h_P_on_ellipse : (P.1^2 / a^2) + (P.2^2 / b^2) = 1
  h_P_coords : P = (3, 4)
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_foci : F₁.1 = -F₂.1 ∧ F₁.2 = 0 ∧ F₂.2 = 0
  h_PF_perp : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0
  h_PF_ratio : (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = 4 * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)

/-- The eccentricity of the special ellipse is √5/3 -/
theorem special_ellipse_eccentricity (e : SpecialEllipse) :
  (e.F₁.1^2 + e.F₁.2^2).sqrt / e.a = Real.sqrt 5 / 3 := by sorry

/-- The standard equation of the special ellipse is x²/45 + y²/20 = 1 -/
theorem special_ellipse_equation (e : SpecialEllipse) :
  e.a^2 = 45 ∧ e.b^2 = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_eccentricity_special_ellipse_equation_l603_60330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equals_two_m_range_l603_60347

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then 1/x - 1
  else if x ≥ 1 then 1 - 1/x
  else 0  -- Define a default value for x ≤ 0

-- Theorem 1
theorem sum_reciprocals_equals_two (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  1/a + 1/b = 2 := by sorry

-- Theorem 2
theorem m_range (a b m : ℝ) (h1 : 1 < a) (h2 : a < b)
  (h3 : ∀ x, a ≤ x ∧ x ≤ b → m * a ≤ f x ∧ f x ≤ m * b) :
  0 < m ∧ m < 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equals_two_m_range_l603_60347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l603_60325

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + 4^x) / Real.log 4 - (1/2) * x

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≥ 0 → x₂ ≥ 0 → x₁ > x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f x ≥ 1/2) ∧
  (∃ x : ℝ, f x = 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l603_60325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_coprime_solution_l603_60315

theorem no_coprime_solution (n : ℕ) 
  (h_no_square_divisor : ∀ d : ℕ, d > 1 → d ∣ n → ¬∃ k : ℕ, d = k^2) :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ Nat.gcd x y = 1 ∧ (x^n + y^n) % ((x + y)^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_coprime_solution_l603_60315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_properties_l603_60345

/-- Properties of a right parallelepiped with rhombus base -/
structure RhombusBaseParallelepiped where
  Q : ℝ  -- Area of the rhombus base
  S₁ : ℝ  -- Area of one diagonal section
  S₂ : ℝ  -- Area of the other diagonal section
  Q_pos : 0 < Q
  S₁_pos : 0 < S₁
  S₂_pos : 0 < S₂

/-- Volume of the parallelepiped -/
noncomputable def volume (p : RhombusBaseParallelepiped) : ℝ :=
  Real.sqrt ((p.S₁ * p.S₂ * p.Q) / 2)

/-- Lateral surface area of the parallelepiped -/
noncomputable def lateralSurfaceArea (p : RhombusBaseParallelepiped) : ℝ :=
  2 * Real.sqrt (p.S₁^2 + p.S₂^2)

theorem parallelepiped_properties (p : RhombusBaseParallelepiped) :
  (volume p = Real.sqrt ((p.S₁ * p.S₂ * p.Q) / 2)) ∧
  (lateralSurfaceArea p = 2 * Real.sqrt (p.S₁^2 + p.S₂^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_properties_l603_60345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l603_60310

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2*x - 8)*(x - 4) / x

-- State the theorem
theorem solution_set (x : ℝ) :
  (x ≠ 0) → (f x ≥ 0 ↔ x < 0 ∨ x > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l603_60310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l603_60323

theorem cos_double_angle_special_case (θ : Real) :
  (2 : Real)^(-2 + 3 * Real.cos θ) + 1 = (2 : Real)^(1/2 + Real.cos θ) →
  Real.cos (2 * θ) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l603_60323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_discount_is_ten_percent_l603_60388

/-- Calculates the final price after two discounts -/
noncomputable def final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  let price_after_first_discount := initial_price * (1 - discount1 / 100)
  price_after_first_discount * (1 - discount2 / 100)

/-- Theorem: Given the initial price, second discount, and final price, the first discount is 10% -/
theorem first_discount_is_ten_percent 
  (initial_price : ℝ) 
  (second_discount : ℝ) 
  (final_price_value : ℝ)
  (h1 : initial_price = 400)
  (h2 : second_discount = 8)
  (h3 : final_price_value = 331.2)
  : ∃ (first_discount : ℝ), 
    first_discount = 10 ∧ 
    final_price initial_price first_discount second_discount = final_price_value :=
by
  sorry

#check first_discount_is_ten_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_discount_is_ten_percent_l603_60388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_monotonicity_l603_60316

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f x + a * x

theorem min_value_and_monotonicity :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x) ∧
  f 1 = 1 ∧
  (∀ (a : ℝ), (∀ (x y : ℝ), 2 ≤ x ∧ x < y → F a x ≤ F a y) ↔ 
    (a ≤ -1/4 ∨ 0 ≤ a)) := by
  sorry

#check min_value_and_monotonicity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_monotonicity_l603_60316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_inscribed_circles_tangent_points_l603_60386

-- Define the necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

def dist (A B : Point) : ℝ := sorry

def altitude (A B C H : Point) : Prop := sorry

def is_inscribed_circle (A B C P : Point) : Prop := sorry

def is_tangent_point (P A B : Point) : Prop := sorry

-- Main theorem
theorem altitude_inscribed_circles_tangent_points 
  (A B C H P Q : Point) 
  (h_altitude : altitude A B C H) 
  (h_inscribed_ACH : is_inscribed_circle A C H P) 
  (h_inscribed_BCH : is_inscribed_circle B C H Q) 
  (h_tangent_P : is_tangent_point P C H) 
  (h_tangent_Q : is_tangent_point Q C H) 
  (h_AB : dist A B = 2023) 
  (h_AC : dist A C = 2022) 
  (h_BC : dist B C = 2021) : 
  dist P Q = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_inscribed_circles_tangent_points_l603_60386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_properties_l603_60336

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a shape in 2D space -/
inductive Shape
  | Triangle (p1 p2 p3 : Point)
  | Parallelogram (p1 p2 p3 p4 : Point)
  | Square (p1 p2 p3 p4 : Point)
  | Rhombus (p1 p2 p3 p4 : Point)

/-- Applies oblique projection to a point -/
noncomputable def obliqueProject (p : Point) : Point :=
  { x := p.x, y := p.y / 2 }

/-- Applies oblique projection to a shape -/
noncomputable def obliqueProjectShape (s : Shape) : Shape :=
  match s with
  | Shape.Triangle p1 p2 p3 => Shape.Triangle (obliqueProject p1) (obliqueProject p2) (obliqueProject p3)
  | Shape.Parallelogram p1 p2 p3 p4 => Shape.Parallelogram (obliqueProject p1) (obliqueProject p2) (obliqueProject p3) (obliqueProject p4)
  | Shape.Square p1 p2 p3 p4 => Shape.Parallelogram (obliqueProject p1) (obliqueProject p2) (obliqueProject p3) (obliqueProject p4)
  | Shape.Rhombus p1 p2 p3 p4 => Shape.Parallelogram (obliqueProject p1) (obliqueProject p2) (obliqueProject p3) (obliqueProject p4)

theorem oblique_projection_properties :
  ∀ (t p s r : Shape),
  (∃ (t' : Shape), t' = obliqueProjectShape t ∧ (∃ p1 p2 p3, t' = Shape.Triangle p1 p2 p3)) ∧
  (∃ (p' : Shape), p' = obliqueProjectShape p ∧ (∃ p1 p2 p3 p4, p' = Shape.Parallelogram p1 p2 p3 p4)) ∧
  (∀ (s' : Shape), s' = obliqueProjectShape s → ¬∃ p1 p2 p3 p4, s' = Shape.Square p1 p2 p3 p4) ∧
  (∀ (r' : Shape), r' = obliqueProjectShape r → ¬∃ p1 p2 p3 p4, r' = Shape.Rhombus p1 p2 p3 p4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_properties_l603_60336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l603_60395

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_intersection_distance : ∃ x₁ x₂, x₁ < x₂ ∧ f ω x₁ = 1 ∧ f ω x₂ = 1 ∧ x₂ - x₁ = Real.pi) :
  (ω = 2) ∧ 
  (∀ x, f ω (Real.pi/6 + x) = f ω (Real.pi/6 - x)) ∧
  (∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.pi ∧ f ω x₁ = 0 ∧ f ω x₂ = 0 ∧ 
   ∀ x, 0 ≤ x ∧ x ≤ Real.pi ∧ f ω x = 0 → x = x₁ ∨ x = x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l603_60395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l603_60377

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side length opposite to A
  b : ℝ  -- Side length opposite to B
  c : ℝ  -- Side length opposite to C
  sum_angles : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the specific triangle condition
def SpecialTriangle (t : Triangle) : Prop :=
  t.a^2 + t.c^2 = t.b^2 + Real.sqrt 2 * t.a * t.c

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.B = π/4 ∧ 
  (∃ (x : ℝ), ∀ (A C : ℝ), 0 ≤ A ∧ A + C = 3*π/4 → Real.cos A + Real.sqrt 2 * Real.cos C ≤ x) ∧
  (∃ (A C : ℝ), 0 ≤ A ∧ A + C = 3*π/4 ∧ Real.cos A + Real.sqrt 2 * Real.cos C = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l603_60377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l603_60372

theorem no_solution_exists : ¬∃ (x : ℕ), 1^(x+3) + 2^(x+2) + 3^x + 4^(x+1) = 1958 := by
  intro h
  cases' h with x hx
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l603_60372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l603_60318

noncomputable def f (x : ℝ) : ℝ := Real.rpow (x - 3) (1/3) + Real.rpow (6 - x) (1/3) + Real.sqrt (x^2 - 9)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≤ -3 ∨ x ≥ 3} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l603_60318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_average_speed_l603_60367

/-- Represents the distance covered in a segment of the journey as a multiple of x -/
structure SegmentDistance (x : ℝ) where
  multiple : ℝ

/-- Represents the speed of the train in a segment of the journey -/
structure SegmentSpeed where
  speed : ℝ

/-- Represents a segment of the train journey -/
structure Segment (x : ℝ) where
  distance : SegmentDistance x
  speed : SegmentSpeed

/-- Calculates the time taken for a segment of the journey -/
noncomputable def segmentTime (x : ℝ) (s : Segment x) : ℝ :=
  (s.distance.multiple * x) / s.speed.speed

/-- Represents the entire train journey -/
structure Journey (x : ℝ) where
  segments : List (Segment x)

/-- Calculates the total distance of the journey -/
noncomputable def totalDistance (x : ℝ) (j : Journey x) : ℝ :=
  (j.segments.map (λ s => s.distance.multiple * x)).sum

/-- Calculates the total time of the journey -/
noncomputable def totalTime (x : ℝ) (j : Journey x) : ℝ :=
  (j.segments.map (segmentTime x)).sum

/-- Calculates the average speed of the journey -/
noncomputable def averageSpeed (x : ℝ) (j : Journey x) : ℝ :=
  totalDistance x j / totalTime x j

/-- The main theorem stating the average speed of the train journey -/
theorem train_journey_average_speed (x : ℝ) (j : Journey x) :
  j.segments = [
    { distance := { multiple := 1 }, speed := { speed := 50 } },
    { distance := { multiple := 2 }, speed := { speed := 30 } },
    { distance := { multiple := 0.5 }, speed := { speed := 35 } },
    { distance := { multiple := 1.5 }, speed := { speed := 40 } }
  ] →
  averageSpeed x j = 1050 / 29.075 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_average_speed_l603_60367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_form_triangle_l603_60344

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
axiom triangle_inequality (a b c : ℝ) : 
  (0 < a ∧ 0 < b ∧ 0 < c) → (a + b > c ∧ b + c > a ∧ c + a > b) ↔ ∃ (t : Set ℝ), t = {a, b, c}

/-- Given three positive real numbers, determines if they can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: The line segments 7cm, 7cm, and 15cm cannot form a triangle -/
theorem cannot_form_triangle : ¬(can_form_triangle 7 7 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_form_triangle_l603_60344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l603_60335

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | x * f x > 0}

theorem solution_set_characterization
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_f_1 : f 1 = 0)
  (h_inequality : ∀ x > 0, x * (deriv (deriv f) x) - f x > 0) :
  solution_set f = Set.Ioo (-1) 0 ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l603_60335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_distinct_roots_iff_a_in_range_l603_60348

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := 2*x + a

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then f a x else f_prime a x

-- Define the composition g(f(x))
noncomputable def g_comp_f (a : ℝ) (x : ℝ) : ℝ := g a (f a x)

-- State the theorem
theorem four_distinct_roots_iff_a_in_range (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    g_comp_f a x₁ = 0 ∧ g_comp_f a x₂ = 0 ∧ g_comp_f a x₃ = 0 ∧ g_comp_f a x₄ = 0) ↔
  (a < 0 ∨ a > 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_distinct_roots_iff_a_in_range_l603_60348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_in_five_years_l603_60328

/-- The annual interest rate that doubles a sum of money in 5 years with compound interest -/
noncomputable def annual_interest_rate : ℝ :=
  Real.log 2 / 5

/-- Theorem stating that the annual interest rate doubles a sum of money in 5 years -/
theorem double_in_five_years (r : ℝ) :
  r = annual_interest_rate ↔ (1 + r)^5 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_in_five_years_l603_60328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l603_60397

theorem exponential_equation_solution :
  ∀ x : ℚ, (3 : ℝ) ^ (2 * x^2 - 5*x + 2 : ℝ) = 3 ^ (2 * x^2 + 7*x - 4 : ℝ) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l603_60397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_power_monotonicity_l603_60398

theorem inverse_power_monotonicity (n : ℕ) :
  let f := fun (x : ℝ) => x^(-(n : ℤ))
  (∀ x y, 0 < x ∧ x < y → f y < f x) ∧
  (∀ x y, x < y ∧ y < 0 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_power_monotonicity_l603_60398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_proof_l603_60341

/-- The time it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (a b v : ℝ) : ℝ :=
  (a + b) / v

/-- Theorem: The time for a train to cross a bridge is (a + b) / v -/
theorem train_crossing_time_proof (a b v : ℝ) (h : v ≠ 0) :
  train_crossing_time a b v = (a + b) / v :=
by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- The equation is now trivially true
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_proof_l603_60341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrangea_percentage_l603_60337

-- Define the total number of flowers
variable (total : ℕ)

-- Define the number of yellow flowers
def yellow (total : ℕ) : ℕ := (4 * total) / 10

-- Define the number of blue flowers
def blue (total : ℕ) : ℕ := total - yellow total

-- Define the number of tulips
def tulips (total : ℕ) : ℕ := blue total / 4

-- Define the number of daisies
def daisies (total : ℕ) : ℕ := yellow total / 3

-- Define the number of hydrangeas
def hydrangeas (total : ℕ) : ℕ := blue total - tulips total

-- Theorem statement
theorem hydrangea_percentage (total : ℕ) (h : total > 0) :
  (hydrangeas total : ℚ) / total = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrangea_percentage_l603_60337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_unit_circle_l603_60313

theorem tan_double_angle_unit_circle (α : ℝ) :
  (∃ P : ℝ × ℝ, P.1 = 3/5 ∧ P.2 = 4/5 ∧ P.1^2 + P.2^2 = 1) →
  Real.tan (2 * α) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_unit_circle_l603_60313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_and_sequence_sum_l603_60350

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + 1 - a * x

noncomputable def seq (n : ℕ) : ℝ :=
  match n with
  | 0 => 2/3
  | n+1 => Real.log ((seq n + 1) / 2) + 1

theorem function_zeros_and_sequence_sum (a : ℝ) :
  (∀ x > 0, f a x = 0 → (a < 1 → False) ∧ (a = 1 → ∃! x, f a x = 0) ∧ (a > 1 → ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0)) ∧
  (∀ n : ℕ, (Finset.range n).sum (λ i => 1 / seq i) < n + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_and_sequence_sum_l603_60350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_incircle_radii_implies_ratio_l603_60338

/-- Triangle ABC with given side lengths -/
structure Triangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ

/-- Point N on side AC of triangle ABC -/
def PointN (t : Triangle) := { x : ℝ // 0 < x ∧ x < t.AC }

/-- Incircle radius of a triangle given its semiperimeter and area -/
noncomputable def incircleRadius (s area : ℝ) : ℝ := area / s

/-- Semiperimeter of triangle ABN -/
noncomputable def sABN (t : Triangle) (n : PointN t) : ℝ :=
  (t.AB + n.val + Real.sqrt (t.AB^2 - 2*t.AB*n.val + n.val^2)) / 2

/-- Semiperimeter of triangle BCN -/
noncomputable def sBCN (t : Triangle) (n : PointN t) : ℝ :=
  (t.BC + (t.AC - n.val) + Real.sqrt (t.BC^2 - 2*t.BC*(t.AC - n.val) + (t.AC - n.val)^2)) / 2

/-- Theorem statement -/
theorem equal_incircle_radii_implies_ratio (t : Triangle) (n : PointN t) :
  t.AB = 10 ∧ t.BC = 14 ∧ t.AC = 16 →
  incircleRadius (sABN t n) (n.val * t.AB / 2) = incircleRadius (sBCN t n) ((t.AC - n.val) * t.BC / 2) →
  n.val / (t.AC - n.val) = 17 / 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_incircle_radii_implies_ratio_l603_60338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_products_l603_60370

theorem max_value_of_sum_products (x y z w : ℕ) : 
  x ∈ ({1, 3, 5, 7} : Set ℕ) ∧ 
  y ∈ ({1, 3, 5, 7} : Set ℕ) ∧ 
  z ∈ ({1, 3, 5, 7} : Set ℕ) ∧ 
  w ∈ ({1, 3, 5, 7} : Set ℕ) ∧ 
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
  x * y + y * z + z * w + w * x ≤ 64 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_products_l603_60370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_girls_in_class_l603_60391

/-- Represents the number of students in the class -/
def total_students : ℕ := 20

/-- Represents the number of girls in the class -/
def num_girls : ℕ → ℕ := λ d => d

/-- Represents the number of boys in the class -/
def num_boys (d : ℕ) : ℕ := total_students - num_girls d

/-- Represents the maximum number of possible unique list lengths -/
def max_unique_lists (d : ℕ) : ℕ := 2 * (d + 1)

/-- The constraint that the number of boys cannot exceed the number of possible unique list lengths -/
def constraint (d : ℕ) : Prop := num_boys d ≤ max_unique_lists d

/-- The theorem stating the minimum number of girls in the class -/
theorem min_girls_in_class : 
  ∃ d : ℕ, d = 6 ∧ constraint d ∧ ∀ k < d, ¬constraint k :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_girls_in_class_l603_60391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_post_office_distance_l603_60357

/-- The distance from the village to the post office in kilometers -/
noncomputable def D : ℝ := 20

/-- The speed to the post office in km/h -/
noncomputable def speed_to : ℝ := 25

/-- The speed from the post office in km/h -/
noncomputable def speed_from : ℝ := 4

/-- The total journey time in hours -/
noncomputable def total_time : ℝ := 5 + 48 / 60

theorem post_office_distance :
  D = 20 ∧ D / speed_to + D / speed_from = total_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_post_office_distance_l603_60357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_profit_growth_rate_l603_60319

/-- The annual growth rate of a company's profit -/
noncomputable def annual_growth_rate (profit_2021 profit_2023 : ℝ) : ℝ :=
  Real.sqrt (profit_2023 / profit_2021) - 1

/-- Theorem stating that the annual growth rate of the company's profit is 0.2 -/
theorem company_profit_growth_rate : 
  let profit_2021 : ℝ := 3000
  let profit_2023 : ℝ := 4320
  annual_growth_rate profit_2021 profit_2023 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_profit_growth_rate_l603_60319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_line_ratio_l603_60363

/-- A rhombus with an acute angle α and a line dividing it -/
structure RhombusWithLine where
  α : ℝ
  acute_angle : 0 < α ∧ α < π/2

/-- The ratio in which the line divides the side of the rhombus -/
noncomputable def side_ratio (r : RhombusWithLine) : ℝ × ℝ :=
  (Real.cos (r.α/6), Real.cos (r.α/2))

/-- Theorem stating the ratio in which the line divides the side -/
theorem rhombus_line_ratio (r : RhombusWithLine) :
  let (a, b) := side_ratio r
  ∃ (k : ℝ), k > 0 ∧ a = k * Real.cos (r.α/6) ∧ b = k * Real.cos (r.α/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_line_ratio_l603_60363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_always_wins_l603_60381

/-- Represents a player in the game -/
inductive Player
| Teacher
| Student (n : Fin 30)

/-- Represents a unit segment on the grid -/
structure Segment where
  x : ℕ
  y : ℕ
  horizontal : Bool

/-- Represents the state of the game -/
structure GameState where
  coloredSegments : Set Segment
  currentPlayer : Player

/-- Represents a move in the game -/
def Move := Segment

/-- Checks if a segment is valid (not already colored) -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  move ∉ state.coloredSegments

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { coloredSegments := state.coloredSegments.insert move
    currentPlayer := 
      match state.currentPlayer with
      | Player.Teacher => Player.Student 0
      | Player.Student n => 
          if n.val = 29 then Player.Teacher
          else Player.Student (Fin.succ n)
  }

/-- Checks if the teacher has won -/
def teacherWins (state : GameState) : Prop :=
  ∃ x y, 
    (Segment.mk x y true ∈ state.coloredSegments ∧
     Segment.mk x (y+1) true ∈ state.coloredSegments ∧
     Segment.mk x y false ∈ state.coloredSegments ∧
     Segment.mk (x+1) y false ∈ state.coloredSegments ∧
     Segment.mk x (y+1) false ∉ state.coloredSegments) ∨
    (Segment.mk x y false ∈ state.coloredSegments ∧
     Segment.mk (x+1) y false ∈ state.coloredSegments ∧
     Segment.mk x y true ∈ state.coloredSegments ∧
     Segment.mk x (y+1) true ∈ state.coloredSegments ∧
     Segment.mk (x+1) y true ∉ state.coloredSegments)

/-- The main theorem stating that the teacher can always win -/
theorem teacher_always_wins :
  ∀ (strategy : GameState → Move),
  ∃ (n : ℕ) (moves : Fin n → Move),
  let finalState := (List.range n).foldl (fun state i => applyMove state (moves ⟨i, sorry⟩)) {coloredSegments := ∅, currentPlayer := Player.Teacher}
  teacherWins finalState ∧ 
  (∀ i : Fin n, isValidMove 
    ((List.range i.val).foldl (fun state j => applyMove state (moves ⟨j, sorry⟩)) {coloredSegments := ∅, currentPlayer := Player.Teacher})
    (moves i)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_always_wins_l603_60381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_bipartite_graph_l603_60332

/-- A bipartite graph with sets A and B -/
structure BipartiteGraph (α β : Type*) where
  A : Finset α
  B : Finset β
  edges : Finset (α × β)

/-- Definition of a connected component in a bipartite graph -/
def isConnectedComponent {α β : Type*} (g : BipartiteGraph α β) : Prop :=
  ∀ (a₁ a₂ : α) (b₁ b₂ : β), a₁ ∈ g.A → a₂ ∈ g.A → b₁ ∈ g.B → b₂ ∈ g.B →
    ∃ (path : List (α ⊕ β)), path.head? = some (Sum.inl a₁) ∧ path.getLast? = some (Sum.inl a₂)

/-- Each node in B is connected to at least two distinct nodes in A -/
def hasAtLeastTwoConnections {α β : Type*} (g : BipartiteGraph α β) : Prop :=
  ∀ b ∈ g.B, ∃ (a₁ a₂ : α), a₁ ≠ a₂ ∧ (a₁, b) ∈ g.edges ∧ (a₂, b) ∈ g.edges

/-- The main theorem -/
theorem max_edges_bipartite_graph :
  ∀ (g : BipartiteGraph Nat Nat),
    g.A.card = 25 →
    g.B.card = 15 →
    isConnectedComponent g →
    hasAtLeastTwoConnections g →
    g.edges.card ≤ 375 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_bipartite_graph_l603_60332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l603_60384

-- Define the points A, B, C in 3D space
variable (A B C : ℝ × ℝ × ℝ)

-- Define the real numbers l, m, n, p, q, r
variable (l m n p q r : ℝ)

-- Define the midpoints
def midpoint_BC : ℝ × ℝ × ℝ := (l, 0, p)
def midpoint_AC : ℝ × ℝ × ℝ := (0, m, q)
def midpoint_AB : ℝ × ℝ × ℝ := (0, 0, r)

-- Define the distance function
def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := P
  let (x₂, y₂, z₂) := Q
  (x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2

-- State the theorem
theorem triangle_ratio_theorem (h : n = Real.sqrt (p^2 + q^2 + r^2)) :
  (distance A B + distance A C + distance B C) / (l^2 + m^2 + n^2 + p^2 + q^2 + r^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l603_60384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l603_60349

-- Define the line and circle
def line (m : ℝ) (x y : ℝ) : Prop := x + y + m = 0
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the condition for intersection
def intersect (m : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ 
    line m A.1 A.2 ∧ line m B.1 B.2 ∧
    my_circle A.1 A.2 ∧ my_circle B.1 B.2

-- Define the vector condition
def vector_condition (A B : ℝ × ℝ) : Prop :=
  let OA := (A.1 - origin.1, A.2 - origin.2)
  let OB := (B.1 - origin.1, B.2 - origin.2)
  let AB := (B.1 - A.1, B.2 - A.2)
  OA.1^2 + OA.2^2 + OB.1^2 + OB.2^2 + 2*(OA.1*OB.1 + OA.2*OB.2) ≥ AB.1^2 + AB.2^2

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (intersect m ∧ 
    (∀ A B : ℝ × ℝ, line m A.1 A.2 ∧ line m B.1 B.2 ∧ 
      my_circle A.1 A.2 ∧ my_circle B.1 B.2 → vector_condition A B)) →
    (m ∈ Set.Icc (-2*Real.sqrt 2) (-2) ∪ Set.Ico 2 (2*Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l603_60349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l603_60396

open Real

theorem calculation_proof :
  ((3/2)^(-2) - (-4.5)^0 - (8/27)^(2/3) = -1) ∧
  (2/3 * log 8 / log 10 + log 25 / log 10 - 3^(2 * log 5 / log 3) + 16^(3/4) = -15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l603_60396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_equation_l603_60364

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem complex_arithmetic_equation : 
  (8^2 - factorial 3) * (4 * Real.sqrt 9) / (factorial 6 / (5 * 4 * 3^2)) = 174 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_equation_l603_60364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l603_60346

-- Define the right triangle
noncomputable def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Define the median to the hypotenuse
noncomputable def median_to_hypotenuse (c m : ℝ) : Prop := m = c / 2

-- Define the area of a triangle
noncomputable def triangle_area (a b : ℝ) : ℝ := (a * b) / 2

-- Theorem statement
theorem right_triangle_area :
  ∀ (a b c m : ℝ),
  a = 6 →
  b = 8 →
  right_triangle a b c →
  median_to_hypotenuse c m →
  m = 5 →
  triangle_area a b = 24 :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l603_60346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_l603_60352

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  side : ℝ
  side_positive : side > 0

/-- A square -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Predicate to check if a square is inscribed in a triangle -/
def IsInscribed (square : Square) (triangle : IsoscelesRightTriangle) : Prop :=
  square.side ≤ triangle.side

/-- Area of a square -/
def area (square : Square) : ℝ :=
  square.side ^ 2

/-- Given an isosceles right triangle with a square inscribed as in Figure 1 
    with an area of 529 cm², the area of a second square inscribed as in Figure 2 
    is 4232/9 cm². -/
theorem inscribed_squares_area (triangle : IsoscelesRightTriangle) 
  (square1 square2 : Square) : 
  IsInscribed square1 triangle ∧ 
  IsInscribed square2 triangle ∧ 
  area square1 = 529 → 
  area square2 = 4232 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_l603_60352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_side_length_l603_60320

theorem triangle_max_side_length 
  (A B C : ℝ) 
  (a b : ℝ)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = Real.pi)
  (h3 : Real.cos (2 * A) + Real.cos (2 * B) + Real.cos (2 * C) = 1)
  (h4 : a = 7 ∧ b = 24) :
  ∃ c : ℝ, c ≤ Real.sqrt 457 ∧ 
    c^2 = a^2 + b^2 - 2 * a * b * Real.cos C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_side_length_l603_60320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l603_60373

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * Real.log x

-- State the theorem
theorem f_derivative_at_one : 
  deriv f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l603_60373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_problems_l603_60339

/-- The number of problems solved per hour by me -/
def p : ℕ := sorry

/-- The number of hours it takes me to finish the homework -/
def t : ℕ := sorry

/-- The theorem stating the conditions and the result to be proved -/
theorem homework_problems (hp : p > 10) (ht : t > 2) 
  (h_equal : p * t = (2 * p - 4) * (t - 2)) : p * t = 60 := by
  sorry

#check homework_problems

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_problems_l603_60339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_range_l603_60311

/-- Curve C1 in polar coordinates -/
noncomputable def C1 (θ : ℝ) : ℝ := Real.sqrt (2 / (1 + Real.sin θ ^ 2))

/-- Curve C2 in polar coordinates -/
noncomputable def C2 (θ : ℝ) : ℝ := 2 * Real.sin θ

/-- The sum of squares of distances from origin to intersection points -/
noncomputable def sum_of_squares (α : ℝ) : ℝ := C1 α ^ 2 + C2 α ^ 2

theorem intersection_distance_range :
  ∀ α, 0 < α → α < π/4 → 2 < sum_of_squares α ∧ sum_of_squares α < 10/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_range_l603_60311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_five_fourths_l603_60366

-- Define the functions
def f (x : ℝ) : ℝ := 4 * x + 4
noncomputable def g (x : ℝ) : ℝ := 3 * x + Real.log x

-- Define the intersection points
noncomputable def M (a : ℝ) : ℝ × ℝ := (((a - 4) / 4), a)
noncomputable def N (a : ℝ) : ℝ × ℝ := (Real.exp (a - 3 * Real.exp (a - 3)), a)

-- Define the distance between M and N
noncomputable def distance (a : ℝ) : ℝ := abs ((N a).1 - (M a).1)

-- Theorem statement
theorem min_distance_is_five_fourths :
  ∃ (a : ℝ), ∀ (b : ℝ), distance a ≤ distance b ∧ distance a = 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_five_fourths_l603_60366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_prime_power_equation_l603_60359

theorem quadruple_prime_power_equation :
  ∀ p q a b : ℕ,
    Nat.Prime p → Nat.Prime q → a > 1 →
    p^a = 1 + 5*q^b →
    ((p = 2 ∧ q = 3 ∧ a = 4 ∧ b = 1) ∨ (p = 3 ∧ q = 2 ∧ a = 4 ∧ b = 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_prime_power_equation_l603_60359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_one_inverse_power_l603_60351

theorem negative_one_inverse_power : (-1 : ℚ)^(-1 : ℤ) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_one_inverse_power_l603_60351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l603_60368

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 2*x - 3) / Real.log (1/2)

-- Theorem statement
theorem f_increasing_on_interval :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < -3 → f x₁ < f x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l603_60368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipes_filling_time_l603_60389

/-- Represents the time in minutes that both pipes were open together -/
noncomputable def t : ℝ := 2

/-- The rate at which pipe p fills the cistern (fraction per minute) -/
noncomputable def rate_p : ℝ := 1 / 12

/-- The rate at which pipe q fills the cistern (fraction per minute) -/
noncomputable def rate_q : ℝ := 1 / 15

/-- The additional time pipe q runs after pipe p is turned off -/
noncomputable def additional_time : ℝ := 10.5

theorem pipes_filling_time :
  t * (rate_p + rate_q) + additional_time * rate_q = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipes_filling_time_l603_60389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ellipse_tangent_and_locus_l603_60326

-- Define the circle and ellipse
def my_circle (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2
def my_ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the conditions
variable (a b : ℝ)
variable (h_ab : a > b ∧ b > 0)

-- Define points A, B, C, and P
variable (A : ℝ × ℝ)
variable (h_A_circle : my_circle a A.1 A.2)
variable (h_A_not_axis : A.1 ≠ 0 ∧ A.2 ≠ 0)

variable (B : ℝ × ℝ)
variable (h_B_ellipse : my_ellipse a b B.1 B.2)
variable (h_AB_vertical : A.1 = B.1)

variable (C : ℝ × ℝ)
variable (h_C_x_intercept : C.2 = 0)

variable (P : ℝ × ℝ)

-- Define the theorem
theorem circle_ellipse_tangent_and_locus :
  (∃ (m : ℝ), ∀ (x y : ℝ), y = m * (x - C.1) → my_circle a x y → x = A.1 ∧ y = A.2) ∧
  (P.1^2 + P.2^2 = (a + b)^2 ∧ P.1 ≠ 0 ∧ P.2 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ellipse_tangent_and_locus_l603_60326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_min_difference_value_l603_60371

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem smallest_difference (p q r : ℕ+) : 
  p * q * r = factorial 9 → p < q → q < r → 
  ∀ (p' q' r' : ℕ+), p' * q' * r' = factorial 9 → p' < q' → q' < r' → 
  r - p ≤ r' - p' := by
sorry

theorem min_difference_value (p q r : ℕ+) :
  p * q * r = factorial 9 → p < q → q < r → 
  (∀ (p' q' r' : ℕ+), p' * q' * r' = factorial 9 → p' < q' → q' < r' → r - p ≤ r' - p') →
  r - p = 39 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_min_difference_value_l603_60371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l603_60399

theorem integral_proof (x : ℝ) : 
  deriv (λ x => (1 / (3 * Real.sqrt 5)) * Real.arctan ((3 * Real.tan x) / Real.sqrt 5)) x = 
  1 / (5 * (Real.cos x)^2 + 9 * (Real.sin x)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l603_60399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_median_l603_60329

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- The height of the trapezoid
  height : ℝ
  -- The length of the lateral side (and diagonal)
  lateral_side : ℝ
  -- Condition that the trapezoid is isosceles
  is_isosceles : True
  -- Condition that the diagonal equals the lateral side
  diagonal_eq_lateral : True

/-- The median of an isosceles trapezoid with given properties -/
noncomputable def median (t : IsoscelesTrapezoid) : ℝ :=
  3 * Real.sqrt 3

/-- Theorem stating that for a trapezoid with height 2 and lateral side 4, the median is 3√3 -/
theorem isosceles_trapezoid_median :
  ∀ t : IsoscelesTrapezoid, t.height = 2 → t.lateral_side = 4 → median t = 3 * Real.sqrt 3 :=
by
  sorry

#check isosceles_trapezoid_median

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_median_l603_60329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_preserving_l603_60356

-- Define the plane as ℝ × ℝ
def Plane := ℝ × ℝ

-- Define the distance function
noncomputable def distance (p q : Plane) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the properties of the function f
def preserves_unit_distance (f : Plane → Plane) : Prop :=
  ∀ x y : Plane, distance x y = 1 → distance (f x) (f y) = 1

-- State the theorem
theorem distance_preserving (f : Plane → Plane) 
  (h : preserves_unit_distance f) :
  ∀ a b : Plane, distance (f a) (f b) = distance a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_preserving_l603_60356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_cost_l603_60317

/-- The cost of the candy in cents -/
def C : ℕ := sorry

/-- The amount of money Jane has in cents -/
def J : ℕ := sorry

/-- The amount of money John has in cents -/
def H : ℕ := sorry

/-- Theorem stating that the candy costs 7 cents -/
theorem candy_cost : C = 7 :=
  by
    -- Jane needs 7 more cents to buy the candy
    have h1 : J + 7 = C := sorry
    -- John needs 1 more cent to buy the candy
    have h2 : H + 1 = C := sorry
    -- They do not have enough money when combining their funds
    have h3 : J + H < C := sorry
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_cost_l603_60317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l603_60354

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Slope of a line -/
noncomputable def slopeOfLine (l : Line) : ℝ :=
  (l.p2.y - l.p1.y) / (l.p2.x - l.p1.x)

/-- Function to check if a point lies on the hyperbola -/
def onHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

theorem hyperbola_theorem (h : Hyperbola) (p : Point) (a b c d : Point) :
  onHyperbola h p ∧ 
  p.x = 4 ∧ p.y^2 = 3 ∧
  a.x = -2 ∧ a.y = 0 ∧
  b.x = 2 ∧ b.y = 0 ∧
  c.x ≠ a.x ∧ c.x ≠ b.x ∧
  d.x ≠ a.x ∧ d.x ≠ b.x ∧
  (∃ (t : Point), t.x = 4 ∧ t.y = 0 ∧ 
    (c.y - t.y) * (d.x - t.x) = (d.y - t.y) * (c.x - t.x)) →
  h.a^2 = 4 ∧ h.b^2 = 1 ∧
  (slopeOfLine (Line.mk a c) / slopeOfLine (Line.mk b d) = -1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l603_60354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_inequality_l603_60358

theorem largest_lambda_inequality : 
  ∃ (lambda_max : ℝ), (∀ (lambda : ℝ), (∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b + lambda*b*c + 2*c*d + a*d) → lambda ≤ lambda_max) ∧ 
  (∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b + lambda_max*b*c + 2*c*d + a*d) ∧
  lambda_max = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_inequality_l603_60358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l603_60308

theorem problem_statement :
  (∀ a : ℝ, a > 0 → a + 1/a ≥ 2) ∧
  (¬ ∃ x₀ : ℝ, Real.sin x₀ + Real.cos x₀ = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l603_60308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l603_60306

-- Define the ellipse C₁
def C₁ (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the line l
def l (x y : ℝ) : Prop := y = x + 2

-- Define the circle (renamed to avoid conflict)
def circleEquation (b x y : ℝ) : Prop := x^2 + y^2 = b^2

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the dot product
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem ellipse_properties :
  ∀ a b : ℝ,
  (∃ x y : ℝ, C₁ a b x y) →
  (∃ x y : ℝ, l x y ∧ circleEquation b x y) →
  eccentricity a b = Real.sqrt 3 / 3 →
  (∀ x y : ℝ, C₁ a b x y ↔ x^2 / 3 + y^2 / 2 = 1) ∧
  (∀ x y : ℝ, C₂ x y) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ,
    C₂ x₁ y₁ → C₂ x₂ y₂ →
    dot_product x₁ y₁ (x₂ - x₁) (y₂ - y₁) = 0 →
    Real.sqrt ((x₂^2 / 4)^2 + x₂^2) ≥ 8 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l603_60306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_line_points_form_hyperbola_l603_60382

/-- The set of points lying on all fold lines when folding a circle onto an internal point -/
def foldLinePoints (R a : ℝ) (h : 0 < a ∧ a < R) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (x - a/2)^2 / (R/2)^2 + y^2 / ((R/2)^2 - (a/2)^2) = 1}

/-- Predicate to check if a set of points forms a hyperbola -/
def IsHyperbola (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), (x, y) ∈ S ↔ (x - c)^2 / a^2 - y^2 / b^2 = 1

/-- Theorem stating that the set of fold line points forms a hyperbola -/
theorem fold_line_points_form_hyperbola (R a : ℝ) (h : 0 < a ∧ a < R) :
  IsHyperbola (foldLinePoints R a h) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_line_points_form_hyperbola_l603_60382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l603_60302

noncomputable def sequenceProperty (a : ℕ → ℝ) : Prop :=
  (∀ n ≥ 2, Real.log (a (n + 1)) = |Real.log (a n) - Real.log (a (n - 1))|) ∧
  a 1 = 2 ∧ a 2 = 3

theorem sequence_properties (a : ℕ → ℝ) (h : sequenceProperty a) :
  (a 3 = 3/2 ∧ a 4 = 2 ∧ a 5 = 4/3) ∧
  (∃ k : ℕ, k ≥ 1 ∧ Real.log (a k) = 0 ↔ ∀ m : ℕ, ∃ n : ℕ, n > m ∧ a n = 1) ∧
  ∃ k : ℕ, k ≥ 1 ∧ 1 ≤ a k ∧ a k < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l603_60302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_on_x_axis_l603_60312

/-- Represents a quadratic function of the form f(x) = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of the vertex of a quadratic function -/
noncomputable def vertexX (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
noncomputable def vertexY (f : QuadraticFunction) : ℝ := f.c - f.b^2 / (4 * f.a)

/-- Theorem: For the parabola y = x² - 6x + c, the vertex lies on the x-axis iff c = 9 -/
theorem vertex_on_x_axis (c : ℝ) :
  let f : QuadraticFunction := { a := 1, b := -6, c := c }
  vertexY f = 0 ↔ c = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_on_x_axis_l603_60312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l603_60327

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 1 → f x + f (1 / (1 - x)) = x

/-- The specific function form that satisfies the equation -/
noncomputable def SpecificFunction (c : ℝ) : ℝ → ℝ := fun x =>
  if x = 0 then c
  else if x = 1 then -c
  else (x^3 - 2*x) / (2*x*(x-1))

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
  ∃ c : ℝ, ∀ x : ℝ, f x = SpecificFunction c x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l603_60327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_C_perpendicular_points_sum_l603_60321

-- Define the curve C in polar coordinates
noncomputable def C (θ : Real) : Real := Real.sqrt (9 / (Real.cos θ ^ 2 + 9 * Real.sin θ ^ 2))

-- Define the Cartesian coordinates of a point on C
noncomputable def cartesian_C (θ : Real) : Real × Real :=
  (C θ * Real.cos θ, C θ * Real.sin θ)

-- Theorem 1: Standard equation of curve C
theorem standard_equation_C :
  ∀ (x y : Real), (x, y) ∈ Set.range cartesian_C ↔ x^2/9 + y^2 = 1 := by
  sorry

-- Theorem 2: Sum of reciprocal squared distances for perpendicular points
theorem perpendicular_points_sum :
  ∀ (θ₁ θ₂ : Real), θ₂ = θ₁ + π/2 →
    1 / (C θ₁)^2 + 1 / (C θ₂)^2 = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_C_perpendicular_points_sum_l603_60321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_divisibility_existence_of_m_l603_60355

theorem power_of_two_divisibility (m n : ℕ) :
  (∃ k : ℕ, (2^n - 1) * k = m^2 + 9) → (∃ i : ℕ, n = 2^i) :=
sorry

theorem existence_of_m (n : ℕ) :
  (∃ i : ℕ, n = 2^i) → (∃ m k : ℕ, (2^n - 1) * k = m^2 + 9) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_divisibility_existence_of_m_l603_60355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l603_60362

def is_valid_subset (M : Finset Nat) : Prop :=
  M ⊆ Finset.range 2012 ∧
  ∀ a b c, a ∈ M → b ∈ M → c ∈ M → (a ∣ b ∨ b ∣ a) ∨ (a ∣ c ∨ c ∣ a) ∨ (b ∣ c ∨ c ∣ b)

theorem max_subset_size :
  ∃ M : Finset Nat, is_valid_subset M ∧ M.card = 18 ∧
  ∀ N : Finset Nat, is_valid_subset N → N.card ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l603_60362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_four_l603_60324

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then 9 * x + 16 else 2 * x - 14

-- Theorem statement
theorem sum_of_solutions_is_four :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = -2 ∧ f x₂ = -2 ∧ x₁ + x₂ = 4) ∧
  (∀ (x : ℝ), f x = -2 → x = -2 ∨ x = 6) := by
  sorry

#check sum_of_solutions_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_four_l603_60324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_ratio_l603_60394

-- Define the cone properties
noncomputable def cone_sector_angle : ℝ := 120 * Real.pi / 180
noncomputable def cone_sector_radius : ℝ := 1

-- Define the lateral surface area
noncomputable def lateral_surface_area : ℝ := Real.pi * cone_sector_radius^2 * (cone_sector_angle / (2 * Real.pi))

-- Define the base radius
noncomputable def base_radius : ℝ := cone_sector_radius * cone_sector_angle / (2 * Real.pi)

-- Define the base area
noncomputable def base_area : ℝ := Real.pi * base_radius^2

-- Define the total surface area
noncomputable def surface_area : ℝ := lateral_surface_area + base_area

-- State the theorem
theorem cone_surface_area_ratio : 
  (surface_area / lateral_surface_area) = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_ratio_l603_60394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_is_exp_l603_60385

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the inverse function
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => Real.exp (x * Real.log a)

-- State the theorem
theorem inverse_log_is_exp (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 2) :
  ∀ x, f a x = 2^x :=
by
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_is_exp_l603_60385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_moves_to_turn_all_lamps_on_l603_60309

/-- Represents the state of a lamp (on or off) -/
inductive LampState
| On : LampState
| Off : LampState

/-- Represents a configuration of n lamps -/
def LampConfiguration (n : ℕ) := Fin n → LampState

/-- Represents a move that switches the state of the first i lamps -/
def switchLamps (config : LampConfiguration n) (i : Fin n) : LampConfiguration n :=
  fun j => if j.val < i.val then
    match config j with
    | LampState.On => LampState.Off
    | LampState.Off => LampState.On
    else config j

/-- Predicate to check if all lamps are on -/
def allLampsOn (config : LampConfiguration n) : Prop :=
  ∀ i, config i = LampState.On

/-- The main theorem: The smallest number of moves to turn all lamps on is n -/
theorem smallest_moves_to_turn_all_lamps_on (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, k = n ∧
  (∀ initialConfig : LampConfiguration n,
    ∃ moves : List (Fin n),
      moves.length ≤ k ∧
      allLampsOn (moves.foldl switchLamps initialConfig)) ∧
  (∀ k' : ℕ, k' < n →
    ∃ initialConfig : LampConfiguration n,
      ∀ moves : List (Fin n),
        moves.length ≤ k' →
        ¬ allLampsOn (moves.foldl switchLamps initialConfig)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_moves_to_turn_all_lamps_on_l603_60309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_widget_price_reduction_l603_60331

theorem widget_price_reduction (total_money : ℚ) (original_quantity : ℕ) (reduced_quantity : ℕ)
  (h1 : total_money = 36)
  (h2 : original_quantity = 6)
  (h3 : reduced_quantity = 8)
  (h4 : total_money / original_quantity = total_money / reduced_quantity + price_reduction) :
  price_reduction = (3/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_widget_price_reduction_l603_60331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l603_60301

theorem triangle_angle_B (a b : ℝ) (A B : ℝ) :
  a = 1 →
  b = Real.sqrt 3 →
  A = π / 6 →
  (B = π / 3 ∨ B = 2 * π / 3) →
  Real.sin B = (b * Real.sin A) / a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l603_60301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_formation_coloring_iff_mobot_at_1_1_l603_60360

/-- Represents a formation with dimensions n and m, and a function to check if a mobot exists at a given position -/
structure Formation (n m : ℕ) where
  hasMobot : ℕ → ℕ → Bool

/-- Represents the number of possible colorings for a formation -/
def PossibleColorings (f : Formation n m) : ℕ := sorry

/-- Conditions for coloring as mentioned in problem 3 -/
def SatisfiesColoringConditions (f : Formation n m) : Prop := sorry

theorem formation_coloring_iff_mobot_at_1_1 
  {n m : ℕ} (f : Formation n m) 
  (h1 : n ≥ 3) (h2 : m ≥ 3) :
  (PossibleColorings f = 6 ∧ SatisfiesColoringConditions f) ↔ f.hasMobot 1 1 = true := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_formation_coloring_iff_mobot_at_1_1_l603_60360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_equals_37_l603_60304

def sequence_a : ℕ → ℕ
  | 0 => 1
  | (n + 1) => sequence_a n + (n + 1)

theorem a_9_equals_37 : sequence_a 9 = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_equals_37_l603_60304
