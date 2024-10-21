import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gifts_and_charity_amount_l1143_114364

noncomputable def monthly_salary : ℝ := 3400

noncomputable def discretionary_income (salary : ℝ) : ℝ := salary / 5

noncomputable def vacation_fund (income : ℝ) : ℝ := 0.30 * income

noncomputable def savings (income : ℝ) : ℝ := 0.20 * income

noncomputable def social_spending (income : ℝ) : ℝ := 0.35 * income

noncomputable def gifts_and_charity (income : ℝ) : ℝ :=
  income - (vacation_fund income + savings income + social_spending income)

theorem gifts_and_charity_amount :
  gifts_and_charity (discretionary_income monthly_salary) = 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gifts_and_charity_amount_l1143_114364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1143_114393

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x/3))^2 + Real.pi * Real.arcsin (x/3) - (Real.arcsin (x/3))^2 + (Real.pi^2/18) * (x^2 + 9*x + 27)

theorem g_range :
  Set.range g = Set.Icc (11*Real.pi^2/24) (59*Real.pi^2/24) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1143_114393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1143_114302

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/2 * sequence_a n + 1

theorem sequence_a_formula (n : ℕ) : 
  sequence_a n = (2^(n+1) - 1) / 2^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1143_114302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_cakes_l1143_114384

/-- Calculates the final number of cakes Baker has after selling some and baking more. -/
def final_cakes (initial_cakes : ℕ) (sold_percentage : ℚ) (new_batch_ratio : ℚ) : ℕ :=
  let remaining_cakes := initial_cakes - Int.floor (↑initial_cakes * sold_percentage)
  let new_cakes := Int.floor (↑remaining_cakes * new_batch_ratio)
  (remaining_cakes + new_cakes).toNat

/-- Theorem stating that given the initial conditions, Baker ends up with 107 cakes. -/
theorem baker_cakes : final_cakes 110 (2/5) (5/8) = 107 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_cakes_l1143_114384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_A_coordinates_l1143_114336

noncomputable section

-- Define the triangle OAB
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (6, 0)

-- Define the angles
def angle_ABO : ℝ := 90 * Real.pi / 180
def angle_AOB : ℝ := 45 * Real.pi / 180

-- Define the rotation angle
def rotation_angle : ℝ := 60 * Real.pi / 180

-- Define point A (we don't know its exact coordinates yet, but we know it's in the first quadrant)
def A : ℝ × ℝ := sorry

-- Function to rotate a point around the origin
def rotate (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos θ - p.2 * Real.sin θ, p.1 * Real.sin θ + p.2 * Real.cos θ)

theorem rotated_A_coordinates :
  rotate A rotation_angle = (3 + 3 * Real.sqrt 3, 3 - 3 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_A_coordinates_l1143_114336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1143_114387

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := log x + x^2 - 1
noncomputable def g (x : ℝ) : ℝ := exp x - exp 1

-- State the theorem
theorem function_inequality (m : ℝ) :
  (∀ x > 1, m * g x > f x) ↔ m ≥ 3 / exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1143_114387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_point_coordinates_l1143_114378

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- A point is on the z-axis if its x and y coordinates are 0 -/
def onZAxis (p : Point3D) : Prop :=
  p.x = 0 ∧ p.y = 0

theorem symmetrical_point_coordinates :
  ∀ (M : Point3D),
    onZAxis M →
    distance M ⟨1, 0, 2⟩ = distance M ⟨1, -3, 1⟩ →
    ∃ (S : Point3D),
      S.x = 0 ∧ S.y = 0 ∧ S.z = 7/2 ∧
      S.x = -M.x ∧ S.y = -M.y ∧ S.z = -M.z :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_point_coordinates_l1143_114378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_selection_size_l1143_114309

def valid_selection (S : Finset ℕ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → a * b ∉ S

theorem max_selection_size :
  ∃ (S : Finset ℕ), S ⊆ Finset.range 1984 ∧ valid_selection S ∧ S.card = 1939 ∧
  ∀ (T : Finset ℕ), T ⊆ Finset.range 1984 → valid_selection T → T.card ≤ 1939 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_selection_size_l1143_114309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1143_114332

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x + a / 16)

-- Define proposition p
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, f a x = y

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x : ℝ, 3^x - 9^x < a

-- Main theorem
theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) → (a > 2 ∨ a ≤ 1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1143_114332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_half_triangle_l1143_114380

theorem gray_area_half_triangle (a : ℝ) (p q : ℝ) : 
  let A : ℝ × ℝ := (-a, 0)
  let B : ℝ × ℝ := (a, 0)
  let C : ℝ × ℝ := (0, Real.sqrt 3 * a)
  let P : ℝ × ℝ := (p, q)
  let area_left := (1 / 2) * (a + p) * q
  let area_right := (Real.sqrt 3 * a^2) / 8 - (Real.sqrt 3 * a * p) / 4 + (a * q) / 4 + 
                    (Real.sqrt 3 * p^2) / 8 - (p * q) / 4 - (Real.sqrt 3 * q^2) / 8
  let area_above := (3 * Real.sqrt 3 * a^2) / 8 + (Real.sqrt 3 * a * p) / 4 - (3 * a * q) / 4 - 
                    (Real.sqrt 3 * p^2) / 8 - (p * q) / 4 + (Real.sqrt 3 * q^2) / 8
  let area_triangle := (Real.sqrt 3 / 2) * a^2
  area_left + area_right + area_above = (1 / 2) * area_triangle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_half_triangle_l1143_114380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1143_114351

-- Define the function f(x) = log_a(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem statement
theorem log_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : StrictMono (f a)) : f a (a + 1) > f a 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1143_114351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_sides_theorem_l1143_114386

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  convex : Bool

/-- Represents the projection of a polyhedron onto a plane -/
def projection (p : ConvexPolyhedron) (s : ℕ) : Prop := sorry

/-- The maximum number of sides in the projection of a convex polyhedron -/
def max_projection_sides (p : ConvexPolyhedron) : ℕ :=
  2 * p.faces - 4

/-- Theorem stating that the maximum number of sides in the projection of a convex polyhedron with n faces is 2n - 4 -/
theorem max_projection_sides_theorem (p : ConvexPolyhedron) (h : p.convex = true) :
  ∀ (s : ℕ), projection p s → s ≤ max_projection_sides p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_sides_theorem_l1143_114386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ABP_l1143_114375

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B P : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (P.2 - A.2) - (P.1 - A.1) * (B.2 - A.2))

/-- The maximum area of triangle ABP given points A(1,2), B(4,1), and P(x,y) on the circle x^2 + y^2 = 25 -/
theorem max_area_triangle_ABP :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (4, 1)
  let circle : Set (ℝ × ℝ) := {P | P.1^2 + P.2^2 = 25}
  ∃ (max_area : ℝ), max_area = (7 + 5 * Real.sqrt 10) / 2 ∧
    ∀ (P : ℝ × ℝ), P ∈ circle →
      area_triangle A B P ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ABP_l1143_114375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_loss_fraction_for_given_parameters_l1143_114320

/-- The energy loss fraction per bounce for a ball falling inelastically -/
noncomputable def energy_loss_fraction (h t g : ℝ) : ℝ :=
  let A := Real.sqrt (2 * h / g)
  let y := ((t - A) / (t + A)) ^ 2
  1 - y

/-- Theorem stating the energy loss fraction for given parameters -/
theorem energy_loss_fraction_for_given_parameters :
  let h := 0.2  -- 20 cm in meters
  let t := 18   -- 18 seconds
  let g := 10   -- 10 m/s²
  ‖energy_loss_fraction h t g - 0.36‖ < 0.001 := by
  sorry

/-- Compute an approximation of the energy loss fraction -/
def approx_energy_loss_fraction (h t g : Float) : Float :=
  let A := Float.sqrt (2 * h / g)
  let y := ((t - A) / (t + A)) ^ 2
  1 - y

#eval approx_energy_loss_fraction 0.2 18 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_loss_fraction_for_given_parameters_l1143_114320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l1143_114308

/-- The time taken for a train to cross a bridge -/
noncomputable def time_to_cross_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 110 meters, traveling at 36 km/hr, 
    takes 24.2 seconds to cross a bridge of length 132 meters -/
theorem train_bridge_crossing_time :
  let train_length : ℝ := 110
  let train_speed_kmh : ℝ := 36
  let bridge_length : ℝ := 132
  time_to_cross_bridge train_length train_speed_kmh bridge_length = 24.2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l1143_114308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_value_l1143_114381

def sequence_a : ℕ → ℚ
| 0 => 3/4
| n + 1 => 1 - 1 / sequence_a n

theorem a_2018_value : sequence_a 2017 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_value_l1143_114381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l1143_114390

/-- The equation of circle C -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 2*y - 9 = -y^2 + 18*x + 9

/-- The center of circle C -/
def center (a b r : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

/-- The radius of circle C -/
def radius (a b r : ℝ) : Prop :=
  r > 0 ∧ ∀ x y, circle_equation x y → (x - a)^2 + (y - b)^2 = r^2

theorem circle_center_radius_sum :
  ∃ a b r : ℝ, center a b r ∧ radius a b r ∧ a + b + r = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l1143_114390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_on_truck_l1143_114372

theorem oranges_on_truck (bags : ℕ) : 
  (30 * bags - 50 - 30 = 220) →  -- Equation representing the problem
  bags = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_on_truck_l1143_114372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_hexagon_probability_is_half_l1143_114377

/-- Represents a hexagonal dart board with a central hexagon and six surrounding triangles -/
structure HexagonalDartboard where
  /-- Side length of the large hexagon -/
  s : ℝ
  /-- Assumption that s is positive -/
  s_pos : s > 0

/-- The area of the central hexagon in the dartboard -/
noncomputable def centralHexagonArea (board : HexagonalDartboard) : ℝ :=
  3 * Real.sqrt 3 * (board.s / 2)^2 / 2

/-- The total area of the six surrounding triangles in the dartboard -/
noncomputable def surroundingTrianglesArea (board : HexagonalDartboard) : ℝ :=
  6 * (Real.sqrt 3 * (board.s / 2)^2 / 4)

/-- The total area of the entire dartboard -/
noncomputable def totalDartboardArea (board : HexagonalDartboard) : ℝ :=
  centralHexagonArea board + surroundingTrianglesArea board

/-- The probability of a dart landing in the central hexagon -/
noncomputable def centralHexagonProbability (board : HexagonalDartboard) : ℝ :=
  centralHexagonArea board / totalDartboardArea board

/-- Theorem stating that the probability of a dart landing in the central hexagon is 1/2 -/
theorem central_hexagon_probability_is_half (board : HexagonalDartboard) :
  centralHexagonProbability board = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_hexagon_probability_is_half_l1143_114377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_partition_l1143_114368

/-- A partition of the set {1, 2, ..., 33} into groups of 3 -/
def Partition := List (Fin 33 × Fin 33 × Fin 33)

/-- Predicate to check if a group satisfies the sum condition -/
def ValidGroup (group : Fin 33 × Fin 33 × Fin 33) : Bool :=
  let (a, b, c) := group
  (a.val + 1 = b.val + 1 + c.val + 1) || 
  (b.val + 1 = a.val + 1 + c.val + 1) || 
  (c.val + 1 = a.val + 1 + b.val + 1)

/-- Predicate to check if a partition is valid -/
def ValidPartition (p : Partition) : Prop :=
  p.length = 11 ∧ 
  p.all ValidGroup ∧ 
  (p.map (fun (a, b, c) => [a, b, c])).join.toFinset.card = 33

theorem no_valid_partition : ¬∃ (p : Partition), ValidPartition p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_partition_l1143_114368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_worth_is_500_l1143_114397

/-- The initial worth of wears Mrs. Smith wanted to buy -/
noncomputable def W : ℝ := sorry

/-- The amount Mrs. Smith needs after adding two-fifths more -/
noncomputable def needed_amount : ℝ := W + (2/5) * W

/-- The discounted price after 15% discount -/
noncomputable def discounted_price : ℝ := 0.85 * needed_amount

/-- Theorem stating that the initial worth of wears is $500 -/
theorem initial_worth_is_500 :
  discounted_price = W + 95 → W = 500 := by
  sorry

#check initial_worth_is_500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_worth_is_500_l1143_114397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_divisible_by_p_l1143_114376

theorem infinitely_many_n_divisible_by_p (p : ℕ) (hp : Nat.Prime p) :
  ∃ f : ℕ → ℕ, Function.Injective f ∧ ∀ k, p ∣ (2^(f k) - f k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_divisible_by_p_l1143_114376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solutions_l1143_114312

theorem cosine_equation_solutions : 
  ∃ (S : Set ℝ), (∀ x ∈ S, x / 50 = Real.cos x ∧ x ∈ Set.Icc (-50) 50) ∧ 
                 (∃ (f : Fin 32 → S), Function.Bijective f) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solutions_l1143_114312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l1143_114326

open BigOperators

noncomputable def f (x : ℝ) : ℝ := ∏ i in Finset.range 51, (x - i)

theorem derivative_f_at_zero :
  deriv f 0 = (50 : ℕ).factorial :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_l1143_114326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_conversion_l1143_114357

/-- Given a map scale where 312 inches represents 136 km, 
    prove that 28 inches on the map corresponds to approximately 12.205 km in actual distance. -/
theorem map_distance_conversion (map_distance : ℝ) (actual_distance : ℝ) (ram_map_distance : ℝ) :
  map_distance = 312 →
  actual_distance = 136 →
  ram_map_distance = 28 →
  ∃ (ram_actual_distance : ℝ), 
    abs (ram_actual_distance - 12.205) < 0.001 ∧
    ram_actual_distance = (actual_distance / map_distance) * ram_map_distance :=
by
  intro h1 h2 h3
  let scale := actual_distance / map_distance
  let ram_actual_distance := scale * ram_map_distance
  use ram_actual_distance
  constructor
  · sorry -- Proof of approximation
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_conversion_l1143_114357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equations_l1143_114399

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 1)
def C : ℝ × ℝ := (-1, 0)

-- Define the equations
def median_eq (x y : ℝ) : Prop := x = 1
def altitude_eq (x y : ℝ) : Prop := 4 * x + y - 7 = 0
def perp_bisector_eq (x y : ℝ) : Prop := 8 * x + 2 * y - 9 = 0

-- Helper functions (not proven, just declared)
def is_median (A B C : ℝ × ℝ) : (ℝ → ℝ → Prop) := sorry
def is_altitude (A B C : ℝ × ℝ) : (ℝ → ℝ → Prop) := sorry
def is_perpendicular_bisector (B C : ℝ × ℝ) : (ℝ → ℝ → Prop) := sorry

-- State the theorem
theorem triangle_equations :
  (∀ x y, median_eq x y ↔ is_median A B C x y) ∧
  (∀ x y, altitude_eq x y ↔ is_altitude A B C x y) ∧
  (∀ x y, perp_bisector_eq x y ↔ is_perpendicular_bisector B C x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equations_l1143_114399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_with_ten_divisors_and_two_prime_factors_l1143_114362

theorem natural_number_with_ten_divisors_and_two_prime_factors :
  ∃! (S : Finset ℕ), S.card = 2 ∧
    (∀ N ∈ S,
      (∃ a b : ℕ, N = 2^a * 3^b) ∧
      (Finset.card (Finset.filter (λ d : ℕ ↦ d ∣ N) (Finset.range (N + 1))) = 10) ∧
      (∀ p : ℕ, Nat.Prime p → p ∣ N → (p = 2 ∨ p = 3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_with_ten_divisors_and_two_prime_factors_l1143_114362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sphere_radius_l1143_114310

-- Define the volume of a sphere
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- State the theorem
theorem original_sphere_radius : 
  ∃ (R : ℝ), R > 0 ∧ sphereVolume R = 64 * sphereVolume 1 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sphere_radius_l1143_114310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1143_114350

theorem triangle_properties (a b c A B C : ℝ) (h1 : a * Real.cos A - b * Real.cos B = 0) 
  (h2 : a ≠ b) (h3 : 0 < A ∧ A < π) (h4 : 0 < B ∧ B < π) (h5 : 0 < C ∧ C < π) 
  (h6 : A + B + C = π) (h7 : 0 < a ∧ 0 < b ∧ 0 < c) : 
  C = π / 2 ∧ 
  ∀ y : ℝ, y = (Real.sin A + Real.sin B) / (Real.sin A * Real.sin B) → 
  y > 2 * Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1143_114350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_25_equals_125_times_5_l1143_114318

theorem power_of_25_equals_125_times_5 (y : ℝ) : (25 : ℝ)^y = 125 * 5 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_25_equals_125_times_5_l1143_114318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1143_114305

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 + 1

-- Theorem statement
theorem f_properties :
  -- 1. Minimum positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T)) ∧

  -- 2. Monotonic increase interval
  (∀ (k : ℤ), ∀ (x y : ℝ), 
    x ∈ Set.Icc (-Real.pi/6 + k*Real.pi) (Real.pi/3 + k*Real.pi) →
    y ∈ Set.Icc (-Real.pi/6 + k*Real.pi) (Real.pi/3 + k*Real.pi) →
    x < y → f x < f y) ∧

  -- 3. Range on the given interval
  (∀ (y : ℝ), y ∈ Set.Icc (-2) 1 ↔ 
    ∃ (x : ℝ), x ∈ Set.Icc (-5*Real.pi/12) (Real.pi/6) ∧ f x = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1143_114305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_abc_l1143_114343

theorem ascending_order_abc :
  let a := Real.log (1/2)
  let b := (1/3 : Real) ^ (0.8 : Real)
  let c := (2 : Real) ^ (1/3 : Real)
  a < b ∧ b < c := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_abc_l1143_114343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_box_dimensions_l1143_114325

/-- The height that maximizes the volume of a box created from an isosceles right triangle --/
noncomputable def optimal_height (b : ℝ) : ℝ := (2 - Real.sqrt 2) * b / 6

/-- The maximum volume of the box --/
noncomputable def max_volume (b : ℝ) : ℝ := (2 - Real.sqrt 2) * b^3 / 27

/-- Theorem stating the optimal height and maximum volume for the box --/
theorem optimal_box_dimensions (b : ℝ) (h : b > 0) :
  ∃ (x : ℝ), x = optimal_height b ∧ 
  ∀ (y : ℝ), y ≥ 0 → y ≤ b / (2 + Real.sqrt 2) → 
  (1/2 * (b - (2 + Real.sqrt 2) * y)^2 * y) ≤ max_volume b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_box_dimensions_l1143_114325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_l1143_114388

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => if n % 2 = 0 then sequence_a (n / 2 + 1) else sequence_a (n / 2 + 1) + 1

theorem sequence_a_property (t : ℕ) :
  (∀ n : ℕ, n > 0 → (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → sequence_a (k * n) = sequence_a n) ↔ n = 2^t - 1) ∧
  (∀ k : ℕ, k > 0 → sequence_a (k * 2^t) ≥ sequence_a (2^t)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_l1143_114388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lineChartIsBestForTemperatureChanges_l1143_114331

inductive ChartType
  | PieChart
  | LineChart
  | BarChart

def bestChartForTemperatureChanges : ChartType := ChartType.LineChart

theorem lineChartIsBestForTemperatureChanges :
  bestChartForTemperatureChanges = ChartType.LineChart := by
  rfl

#check lineChartIsBestForTemperatureChanges

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lineChartIsBestForTemperatureChanges_l1143_114331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_dolls_difference_l1143_114300

/-- The number of dolls Rene's sister has -/
def sister_dolls : ℕ := sorry

/-- The number of dolls Rene has -/
def rene_dolls : ℕ := sorry

/-- The number of dolls the grandmother has -/
def grandmother_dolls : ℕ := 50

/-- The total number of dolls -/
def total_dolls : ℕ := 258

theorem sister_dolls_difference :
  rene_dolls = 3 * sister_dolls →
  total_dolls = rene_dolls + sister_dolls + grandmother_dolls →
  sister_dolls - grandmother_dolls = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_dolls_difference_l1143_114300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_and_distance_l1143_114366

/-- Line l in polar coordinates -/
def line_l (ρ θ : ℝ) : Prop :=
  ρ * Real.cos (θ + Real.pi/6) = (Real.sqrt 3 - 1) / 2

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop :=
  ρ * (1 - Real.cos θ ^ 2) - 2 * Real.cos θ = 0

/-- Line l' in rectangular coordinates -/
def line_l' (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - 2)

/-- Point M -/
def M : ℝ × ℝ := (2, 0)

/-- The main theorem -/
theorem polar_to_rectangular_and_distance :
  ∃ (P Q : ℝ × ℝ),
    (∀ x y : ℝ, (Real.sqrt 3 * x - y - Real.sqrt 3 + 1 = 0) ↔ 
      (∃ ρ θ : ℝ, line_l ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)) ∧
    (∀ x y : ℝ, (y^2 = 2*x) ↔ 
      (∃ ρ θ : ℝ, curve_C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)) ∧
    line_l' P.1 P.2 ∧ P.2^2 = 2*P.1 ∧
    line_l' Q.1 Q.2 ∧ Q.2^2 = 2*Q.1 ∧
    (P.1 - M.1)^2 + (P.2 - M.2)^2 + (Q.1 - M.1)^2 + (Q.2 - M.2)^2 = 112/9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_and_distance_l1143_114366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1143_114311

def M : Set ℕ := {1, 3, 5, 7, 9}

def N : Set ℕ := {x : ℕ | 2 * x > 7}

theorem intersection_M_N : M ∩ N = {5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1143_114311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1143_114358

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the focus of the hyperbola
def hyperbola_focus : ℝ × ℝ := (2, 0)

-- State the theorem
theorem parabola_triangle_area 
  (p : ℝ) 
  (F K A : ℝ × ℝ) 
  (hF : F = hyperbola_focus) 
  (hA : parabola p A.1 A.2) 
  (hK : K.2 = 0) 
  (hAK : (A.1 - K.1)^2 + (A.2 - K.2)^2 = 2 * ((A.1 - F.1)^2 + (A.2 - F.2)^2)) :
  (1/2) * abs ((A.1 - F.1) * (K.2 - F.2) - (K.1 - F.1) * (A.2 - F.2)) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1143_114358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_schools_l1143_114383

/-- Represents a student in the math contest -/
structure Student where
  name : String
  rank : Nat

/-- Represents a team of 3 students from a high school -/
structure Team where
  members : Fin 3 → Student

/-- The math contest in the city of Archimedes -/
structure Contest where
  teams : List Team
  unique_ranks : ∀ t1 t2 : Team, ∀ i j : Fin 3, 
    (t1.members i).rank = (t2.members j).rank → t1 = t2 ∧ i = j

theorem number_of_schools (contest : Contest) 
  (andrea sam charlie : Student)
  (andrea_in_team : ∃ t : Team, t ∈ contest.teams ∧ 
    (∃ i : Fin 3, t.members i = andrea) ∧
    (∃ j : Fin 3, t.members j = sam) ∧
    (∃ k : Fin 3, t.members k = charlie))
  (andrea_median : andrea.rank > charlie.rank ∧ andrea.rank < sam.rank)
  (charlie_rank : charlie.rank = 40)
  (sam_rank : sam.rank = 51)
  : contest.teams.length = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_schools_l1143_114383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_properties_l1143_114352

noncomputable def geometric_progression (a₁ q : ℝ) : ℕ → ℝ := λ n ↦ a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

noncomputable def geometric_product (a₁ q : ℝ) (n : ℕ) : ℝ := a₁^n * q^(n * (n - 1) / 2)

noncomputable def common_difference (a₁ q : ℝ) : ℕ → ℝ := λ n ↦ a₁ * (3 / 2^(n + 1))

theorem geometric_progression_properties (a₁ q : ℝ) (h₁ : a₁ = 2011) (h₂ : q = -1/2) :
  (∀ n : ℕ, geometric_sum a₁ q 2 ≤ geometric_sum a₁ q n ∧ geometric_sum a₁ q n ≤ geometric_sum a₁ q 1) ∧
  (∀ n : ℕ, geometric_product a₁ q 12 ≥ geometric_product a₁ q n) ∧
  (∃ r : ℝ, ∀ n : ℕ, common_difference a₁ q (n + 1) = r * common_difference a₁ q n ∧ r = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_properties_l1143_114352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l1143_114323

noncomputable section

/-- Represents the time in hours for a given distance and speed -/
def time_hours (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

theorem route_time_difference :
  let route_x_distance : ℝ := 8
  let route_x_speed : ℝ := 40
  let route_y_total_distance : ℝ := 7
  let route_y_normal_distance : ℝ := 6.5
  let route_y_construction_distance : ℝ := 0.5
  let route_y_normal_speed : ℝ := 50
  let route_y_construction_speed : ℝ := 10

  let route_x_time := hours_to_minutes (time_hours route_x_distance route_x_speed)
  let route_y_time := hours_to_minutes (time_hours route_y_normal_distance route_y_normal_speed) +
                      hours_to_minutes (time_hours route_y_construction_distance route_y_construction_speed)

  route_x_time - route_y_time = 1.2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l1143_114323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_difference_is_075_l1143_114301

/-- The price of the item in dollars -/
noncomputable def item_price : ℚ := 50

/-- The first tax rate as a percentage -/
noncomputable def tax_rate1 : ℚ := 85/10

/-- The second tax rate as a percentage -/
noncomputable def tax_rate2 : ℚ := 7

/-- The difference between the two tax amounts -/
noncomputable def tax_difference : ℚ := item_price * (tax_rate1 / 100) - item_price * (tax_rate2 / 100)

theorem tax_difference_is_075 : tax_difference = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_difference_is_075_l1143_114301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l1143_114317

/-- The number of ways to represent n as a sum of powers of 2 -/
def f (n : ℕ) : ℕ := sorry

/-- Main theorem: bounds for f(2^n) -/
theorem f_bounds (n : ℕ) (h : n ≥ 3) :
  (2 : ℝ)^(n^2/4) < (f (2^n) : ℝ) ∧ (f (2^n) : ℝ) < (2 : ℝ)^(n^2/2) := by
  sorry

#check f_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l1143_114317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_domain_l1143_114344

-- Define the function f with domain [1, +∞)
noncomputable def f : ℝ → ℝ := sorry

-- Define the property that f is defined on [1, +∞)
axiom f_domain : ∀ x : ℝ, x ≥ 1 → f x ≠ 0

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := f (x - 1) + f (4 - x)

-- Theorem statement
theorem y_domain :
  {x : ℝ | y x ≠ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_domain_l1143_114344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piston_max_height_l1143_114307

/-- The maximum height reached by a piston above a cylindrical vessel containing an ideal gas -/
noncomputable def max_height (M P a g c_v R : ℝ) : ℝ :=
  2 * P^2 / (M^2 * g * a^2 * (1 + c_v/R)^2)

/-- Theorem stating the maximum height reached by the piston -/
theorem piston_max_height 
  (M P a g c_v R : ℝ) 
  (h_M : M > 0)
  (h_P : P > 0)
  (h_a : a > 0)
  (h_g : g > 0)
  (h_c_v : c_v > 0)
  (h_R : R > 0) :
  ∃ (h : ℝ), h = max_height M P a g c_v R ∧ 
  h = (2 * P^2) / (M^2 * g * a^2 * (1 + c_v/R)^2) := by
  sorry

#check piston_max_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piston_max_height_l1143_114307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_section_l1143_114370

/-- Given a prism with base ABC and parallel top face A₁B₁C₁, we define a section DKMN. -/
theorem prism_section (a : ℝ) (h_a : a > 0) :
  let BK := 3*a/4
  let BD := a/2
  let BQ := a/4
  let BG := a/6
  let DK := a * Real.sqrt 7 / 4
  let BH := 3 * a * Real.sqrt 3 / (4 * Real.sqrt 7)
  let BL := 3 * a * Real.sqrt 6 / (4 * Real.sqrt 7)
  let S_ABC := a^2 * Real.sqrt 3 / 4
  let V_LBHK := S_ABC * BL / 8
  let V_LB_NM := V_LBHK / 27
  let V_BDKB_NM := 26 * V_LBHK / 27
  let V_total := 2 * S_ABC * BL / 3
  let V_ACKDA_CN := V_total - V_BDKB_NM
  ∀ α : ℝ, Real.cos α = Real.sqrt (2/3) → 
    (V_LB_NM = 91 * Real.sqrt 2 / 6 ∧ 
     V_ACKDA_CN = 413 * Real.sqrt 2 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_section_l1143_114370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_minus_2sqrt3_sinx_cosx_range_l1143_114379

theorem cos_2x_minus_2sqrt3_sinx_cosx_range :
  ∀ x : ℝ, -3 ≤ Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1 ∧
           Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_minus_2sqrt3_sinx_cosx_range_l1143_114379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_equation_l1143_114313

theorem no_solution_for_equation : ¬∃ (a b : ℕ), (3^a : ℤ) - (2^b : ℤ) = 41 ∧ (3^a : ℤ) - (2^b : ℤ) = -41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_equation_l1143_114313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_reduction_percentage_l1143_114342

-- Define the original and reduced number of faculty members
noncomputable def original_faculty : ℝ := 229.41
noncomputable def reduced_faculty : ℝ := 195

-- Define the percentage reduction function
noncomputable def percentage_reduction (original : ℝ) (reduced : ℝ) : ℝ :=
  (original - reduced) / original * 100

-- Theorem statement
theorem faculty_reduction_percentage :
  abs (percentage_reduction original_faculty reduced_faculty - 15) < 0.1 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_reduction_percentage_l1143_114342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sparrow_swallow_weight_problem_l1143_114327

/-- Represents the weight of a sparrow in taels -/
def x : ℝ := sorry

/-- Represents the weight of a swallow in taels -/
def y : ℝ := sorry

/-- The number of sparrows -/
def num_sparrows : ℕ := 5

/-- The number of swallows -/
def num_swallows : ℕ := 6

/-- The total weight of all birds in taels -/
def total_weight : ℝ := 16

/-- The system of equations correctly represents the sparrow and swallow weight problem -/
theorem sparrow_swallow_weight_problem :
  (4 * x + y = 5 * y + x) ∧
  (num_sparrows * x + num_swallows * y = total_weight) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sparrow_swallow_weight_problem_l1143_114327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_equals_g_l1143_114385

-- Define the original piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -4 ∧ x ≤ -1 then -1 - x
  else if x > -1 ∧ x ≤ 3 then Real.sqrt (9 - (x - 3)^2) - 1
  else if x > 3 ∧ x ≤ 4 then 2 * (x - 3)
  else 0  -- undefined for other x values

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -2 ∧ x ≤ 1 then 4 - x
  else if x > 1 ∧ x ≤ 5 then Real.sqrt (9 - (x - 5)^2) + 2
  else if x > 5 ∧ x ≤ 6 then 2 * x - 7
  else 0  -- undefined for other x values

-- Theorem stating that g is the transformation of f
theorem transform_f_equals_g :
  ∀ x : ℝ, x ≥ -2 ∧ x ≤ 6 → g x = f (x - 2) + 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_equals_g_l1143_114385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_draws_is_eight_l1143_114330

/-- A color of a ball -/
inductive Color
| Red
| Yellow
| Blue
| Green

/-- The probability of drawing a ball of any color -/
noncomputable def prob : Color → ℝ
| _ => 1/4

/-- The expected number of draws until two consecutive red balls are drawn -/
noncomputable def expected_draws : ℝ := sorry

/-- Theorem stating that the expected number of draws is 8 -/
theorem expected_draws_is_eight : expected_draws = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_draws_is_eight_l1143_114330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_evaluation_multiplications_l1143_114394

/-- Represents a polynomial of degree n -/
def MyPolynomial (α : Type*) (n : ℕ) := Fin (n + 1) → α

/-- Number of multiplications required by Horner's method -/
def horner_multiplications (n : ℕ) : ℕ := n

/-- Number of multiplications required by direct summation method -/
def direct_summation_multiplications (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating the number of multiplications for both methods -/
theorem polynomial_evaluation_multiplications (n : ℕ) (P : MyPolynomial ℝ n) (x₀ : ℝ) :
  (horner_multiplications n = n) ∧
  (direct_summation_multiplications n = n * (n + 1) / 2) := by
  constructor
  . rfl
  . rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_evaluation_multiplications_l1143_114394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1143_114337

noncomputable def f (x : ℝ) : ℝ := (2*x - 1) / (x + 1)

theorem f_range : 
  ∀ y ∈ Set.range f,
  (∃ x ∈ Set.Icc 0 2, f x = y) ↔ y ∈ Set.Icc (-1) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1143_114337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1143_114339

def strictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, x < y → f x < f y

theorem function_properties (f : ℕ → ℕ) 
  (h1 : strictlyIncreasing f) 
  (h2 : ∀ k : ℕ, k > 0 → f (f k) = 3 * k) :
  (∀ k : ℕ, k > 0 → f (3 * k) = 3 * f k) ∧ 
  (∀ k : ℕ, k > 0 → f (3^(k-1)) = 2 * 3^(k-1)) ∧
  (∃ k : ℕ, ∃ p : ℕ, k > 0 ∧ p = 3^(k-1) + 1 ∧ 
    ∀ i : ℕ, 1 ≤ i ∧ i ≤ p → f (i + 1) = f i + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1143_114339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_trisected_segment_l1143_114319

/-- Predicate stating that B and C trisect the segment AD -/
def trisect (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D

/-- Predicate stating that M is the midpoint of segment AD -/
def is_midpoint (M A D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A M = dist M D

/-- Given a line segment AD with points B and C trisecting it, and M as its midpoint,
    prove that the length of AD is 24 when MC = 4 and AB is twice MC. -/
theorem length_of_trisected_segment (A B C D M : EuclideanSpace ℝ (Fin 2))
    (h1 : trisect A B C D)
    (h2 : is_midpoint M A D)
    (h3 : dist M C = 4)
    (h4 : dist A B = 2 * dist M C) :
    dist A D = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_trisected_segment_l1143_114319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1143_114391

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 4 = 0

/-- First tangent line -/
def tangent1 (x y : ℝ) : Prop :=
  y = (3/4) * x

/-- Second tangent line -/
def tangent2 (x : ℝ) : Prop :=
  x = 0

/-- Theorem stating that the two lines are tangent to the circle -/
theorem tangent_lines_to_circle :
  (∃ (x y : ℝ), circle_eq x y ∧ tangent1 x y ∧ (x, y) ≠ (0, 0)) ∧
  (∃ (y : ℝ), circle_eq 0 y ∧ tangent2 0) ∧
  (∀ (x y : ℝ), circle_eq x y → (tangent1 x y ∨ tangent2 x) → (x, y) = (0, 0) ∨ 
    (∃ (t : ℝ), x = t ∧ y = (3/4) * t) ∨ x = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1143_114391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interiorIntersects_l1143_114329

-- Define the type for arcs
def Arc : Type := Set (ℝ × ℝ)

-- Define the interior of an arc
noncomputable def arcInterior (a : Arc) : Set (ℝ × ℝ) := sorry

-- Define the union of two arcs
def arcUnion (a b : Arc) : Set (ℝ × ℝ) := a ∪ b

-- Define the complement of a set in ℝ²
def complement (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := (Set.univ : Set (ℝ × ℝ)) \ s

-- Define the condition for arcs sharing endpoints but not intersecting otherwise
def shareEndpointsOnly (a b c : Arc) : Prop := sorry

-- Define the condition for the complement consisting of exactly three regions
def threeRegions (a b c : Arc) : Prop := sorry

-- Define the condition for P being an arc between interiors of P₁ and P₃
def betweenInteriors (p p1 p3 : Arc) : Prop := sorry

-- Define the condition for P's interior being in the specified region
def interiorInRegion (p p1 p2 p3 : Arc) : Prop := sorry

-- The main theorem
theorem interiorIntersects (P P1 P2 P3 : Arc) : 
  shareEndpointsOnly P1 P2 P3 →
  threeRegions P1 P2 P3 →
  betweenInteriors P P1 P3 →
  interiorInRegion P P1 P2 P3 →
  (arcInterior P ∩ arcInterior P2).Nonempty := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interiorIntersects_l1143_114329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_rent_and_revenue_l1143_114389

/-- Represents the car rental company's properties and revenue function -/
structure RentalCompany where
  totalCars : ℕ := 100
  baseRent : ℕ := 3000
  rentIncrement : ℕ := 50
  rentedMaintenance : ℕ := 150
  unrentedMaintenance : ℕ := 50

/-- Calculates the number of rented cars given a rent amount -/
def rentedCars (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.totalCars - (rent - company.baseRent) / company.rentIncrement

/-- Calculates the revenue given a rent amount -/
def revenue (company : RentalCompany) (rent : ℕ) : ℚ :=
  let rented := rentedCars company rent
  (rented : ℚ) * ((rent : ℚ) - company.rentedMaintenance) -
    ((company.totalCars - rented : ℚ) * company.unrentedMaintenance)

/-- The main theorem stating the optimal rent and maximum revenue -/
theorem optimal_rent_and_revenue (company : RentalCompany) :
  rentedCars company 3600 = 88 ∧
  (∀ x : ℕ, revenue company x ≤ revenue company 4050) ∧
  revenue company 4050 = 307050 := by
  sorry

def defaultCompany : RentalCompany := {
  totalCars := 100
  baseRent := 3000
  rentIncrement := 50
  rentedMaintenance := 150
  unrentedMaintenance := 50
}

#eval rentedCars defaultCompany 3600
#eval revenue defaultCompany 4050

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_rent_and_revenue_l1143_114389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_2015_l1143_114363

theorem function_value_at_2015 (f : ℝ → ℝ) 
  (h1 : ∀ a b : ℝ, f ((2*a + b)/3) = (2*f a + f b)/3)
  (h2 : f 1 = 1)
  (h3 : f 4 = 7) :
  f 2015 = 4029 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_2015_l1143_114363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l1143_114322

/-- The positive slope of the asymptotes for the hyperbola (x^2/16) - (y^2/25) = 1 -/
noncomputable def asymptote_slope : ℝ := 5/4

/-- The equation of the given hyperbola -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2/16 - y^2/25 = 1

/-- The equation of the asymptotes -/
def is_on_asymptote (x y : ℝ) : Prop :=
  y = asymptote_slope * x ∨ y = -asymptote_slope * x

theorem hyperbola_asymptote_slope :
  ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    is_on_hyperbola x y → |x| > δ →
    ∃ y' : ℝ, is_on_asymptote x y' ∧ |y - y'| < ε := by
  sorry

#check hyperbola_asymptote_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l1143_114322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1143_114324

-- Define the sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N : Set ℝ := {x | Real.log (x + 2) ≥ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1143_114324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_in_tank_l1143_114306

/-- The volume of water in a cylindrical tank lying on its side -/
noncomputable def water_volume (r h d : ℝ) : ℝ := 
  h * (2 * r^2 * Real.arccos ((r - d) / r) - (r - d) * Real.sqrt (2 * r * d - d^2))

/-- Proof that the volume of water in the given cylindrical tank is 48π - 36√3 cubic feet -/
theorem water_volume_in_tank : 
  water_volume 4 9 2 = 48 * Real.pi - 36 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_in_tank_l1143_114306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_propositions_l1143_114348

-- Define Triangle as an unspecified type
structure Triangle : Type := mk ::

-- Define interior_angle_sum as a function from Triangle to Real
noncomputable def interior_angle_sum : Triangle → ℝ := sorry

theorem negation_propositions :
  (¬ (∀ t : Triangle, interior_angle_sum t = 180) = False) ∧
  (¬ (∀ x : ℝ, x^2 > 0) = True) ∧
  (¬ (∃ x : ℝ, x^2 = 1) = False) ∧
  (¬ (∃ x : ℝ, x^2 - 3*x + 2 = 0) = False) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_propositions_l1143_114348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_sixth_l1143_114346

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability that a randomly selected point (x,y) from the given rectangle satisfies x + 1 < y --/
noncomputable def probability_x_plus_one_less_than_y (r : Rectangle) : ℝ :=
  let favorable_area := (r.x_max - r.x_min) * (r.y_max - r.y_min) / 2
  let total_area := (r.x_max - r.x_min) * (r.y_max - r.y_min)
  favorable_area / total_area

/-- The specific rectangle from the problem --/
def problem_rectangle : Rectangle where
  x_min := 0
  x_max := 4
  y_min := 0
  y_max := 3
  h_x := by norm_num
  h_y := by norm_num

theorem probability_is_one_sixth :
  probability_x_plus_one_less_than_y problem_rectangle = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_sixth_l1143_114346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_centroid_on_incircle_l1143_114304

/-- Predicate to represent that the centroid is on the incircle --/
def centroid_on_incircle (α : Real) : Prop :=
  let ρ := (Real.sin α + Real.cos α - 1) / 2
  (Real.cos α / 3 - ρ)^2 + (Real.sin α / 3 - ρ)^2 = ρ^2

/-- A right triangle with centroid on the incircle has a specific relation for its acute angle --/
theorem right_triangle_centroid_on_incircle (α : Real) 
  (h_right : α > 0 ∧ α < Real.pi/2)  -- α is an acute angle
  (h_centroid : centroid_on_incircle α) : 
  Real.sin α + Real.cos α = 4 * Real.sqrt 3 / 3 - 1 := by
  sorry

#check right_triangle_centroid_on_incircle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_centroid_on_incircle_l1143_114304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_17x_relation_l1143_114328

theorem cos_sin_17x_relation (f : ℝ → ℝ) :
  (∀ x : ℝ, Real.cos (17 * x) = f (Real.cos x)) →
  (∀ x : ℝ, Real.sin (17 * x) = f (Real.sin x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_17x_relation_l1143_114328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1143_114355

open Set
open Function

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

-- Define the domain of the function
def domain : Set ℝ := {x | x ≠ -2}

-- Theorem statement
theorem range_of_f :
  range (f ∘ (coe : domain → ℝ)) = (Iio 1) ∪ (Ioi 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1143_114355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1143_114373

-- Define the functions
noncomputable def f1 (x : ℝ) := (1 : ℝ)
noncomputable def f2 (x : ℝ) := x ^ (0 : ℝ)
noncomputable def f3 (x : ℝ) := x
noncomputable def f4 (x : ℝ) := x^2 / x
noncomputable def f5 (x : ℝ) := Real.log (Real.exp x)
noncomputable def f6 (x : ℝ) := abs x
noncomputable def f7 (x : ℝ) := (Real.sqrt x)^2

-- State the theorem
theorem function_equality :
  (∃ x : ℝ, f1 x ≠ f2 x) ∧
  (∃ x : ℝ, f3 x ≠ f4 x) ∧
  (∀ x : ℝ, f3 x = f5 x) ∧
  (∃ x : ℝ, f6 x ≠ f7 x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1143_114373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_nested_calculation_l1143_114303

-- Define the ⊗ operation as noncomputable
noncomputable def otimes (a b c : ℝ) : ℝ := a / (b - c)

-- State the theorem
theorem otimes_nested_calculation :
  otimes (otimes 1 3 4) (otimes 2 4 3) (otimes 4 3 2) = 1/2 :=
by
  -- Unfold the definition of otimes
  unfold otimes
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_nested_calculation_l1143_114303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_q_p_l1143_114316

/-- A type representing the numbers on the cards -/
def CardNumber : Type := Fin 10

/-- The total number of cards -/
def totalCards : Nat := 50

/-- The number of cards for each number -/
def cardsPerNumber : Nat := 5

/-- The number of cards drawn -/
def drawnCards : Nat := 5

/-- The probability of drawing five cards with the same number -/
def p' : ℚ :=
  (10 : ℚ) / Nat.choose totalCards drawnCards

/-- The probability of drawing four cards with one number and one card with a different number -/
def q' : ℚ :=
  (2250 : ℚ) / Nat.choose totalCards drawnCards

/-- Theorem stating that the ratio of q' to p' is 225 -/
theorem ratio_q_p : q' / p' = 225 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_q_p_l1143_114316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1143_114398

/-- The function f(x) = 2x^2 + 4x + 6 + 2√x --/
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x + 6 + 2 * Real.sqrt x

/-- Theorem stating that the minimum value of f(x) for x ≥ 0 is 6 --/
theorem min_value_of_f : 
  ∀ x : ℝ, x ≥ 0 → f x ≥ 6 ∧ ∃ y : ℝ, y ≥ 0 ∧ f y = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1143_114398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wages_theorem_l1143_114392

/-- The number of days the sum can pay both A and B's wages -/
noncomputable def both_days : ℝ := 12

/-- The number of days the sum can pay B's wages -/
noncomputable def b_days : ℝ := 30

/-- The daily wage of A -/
noncomputable def a_wage : ℝ := (3 / 2) * (b_days / both_days)

/-- The number of days the sum can pay A's wages -/
noncomputable def a_days : ℝ := b_days / a_wage

theorem wages_theorem : a_days = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wages_theorem_l1143_114392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_m_range_l1143_114341

-- Define the function f as noncomputable due to its dependency on Real
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + x else Real.log x / Real.log (1/3)

-- State the theorem
theorem f_upper_bound_implies_m_range (m : ℝ) :
  (∀ x, f x ≤ 5/4*m - m^2) → m ∈ Set.Icc (1/4) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_m_range_l1143_114341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1143_114395

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi / 4) + Real.cos (x - Real.pi / 4)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1143_114395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_six_l1143_114345

theorem opposite_of_six (a : ℤ) : a = -6 ↔ a = -6 := by
  apply Iff.refl

#check opposite_of_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_six_l1143_114345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_increase_four_squares_l1143_114335

/-- Calculates the percent increase between two values -/
noncomputable def percentIncrease (initial : ℝ) (final : ℝ) : ℝ :=
  (final - initial) / initial * 100

/-- Represents a sequence of squares with increasing side lengths -/
structure SquareSequence where
  initialSideLength : ℝ
  growthFactor : ℝ
  numSquares : ℕ

/-- Calculates the side length of the nth square in the sequence -/
noncomputable def nthSquareSideLength (seq : SquareSequence) (n : ℕ) : ℝ :=
  seq.initialSideLength * (seq.growthFactor ^ (n - 1))

/-- Calculates the perimeter of a square given its side length -/
noncomputable def squarePerimeter (sideLength : ℝ) : ℝ :=
  4 * sideLength

theorem perimeter_increase_four_squares (seq : SquareSequence) :
  seq.initialSideLength = 3 ∧ 
  seq.growthFactor = 2 ∧ 
  seq.numSquares = 4 →
  percentIncrease 
    (squarePerimeter (nthSquareSideLength seq 1)) 
    (squarePerimeter (nthSquareSideLength seq 4)) = 700 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_increase_four_squares_l1143_114335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_POQ_approx_l1143_114315

/-- Represents the configuration of squares in the diagram -/
structure SquareGrid where
  side_length : ℝ
  rows : ℕ
  columns : ℕ

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the angle between two vectors -/
noncomputable def angle_between (v1 v2 : Point) : ℝ :=
  Real.arccos ((v1.x * v2.x + v1.y * v2.y) / 
    (Real.sqrt (v1.x^2 + v1.y^2) * Real.sqrt (v2.x^2 + v2.y^2)))

/-- Main theorem statement -/
theorem angle_POQ_approx (grid : SquareGrid) (O P Q : Point) : 
  grid.side_length = 2 ∧ 
  grid.rows = 2 ∧ 
  grid.columns = 3 ∧
  O = ⟨3, 2⟩ ∧ 
  P = ⟨6, 4⟩ ∧ 
  Q = ⟨0, 0⟩ →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |angle_between (Point.mk (P.x - O.x) (P.y - O.y)) 
                 (Point.mk (Q.x - O.x) (Q.y - O.y)) * (180 / Real.pi) - 26.6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_POQ_approx_l1143_114315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_runs_in_twenty_over_match_l1143_114371

/-- Represents a cricket match with specific constraints. -/
structure CricketMatch where
  totalOvers : Nat
  maxSixesPerOver : Nat
  firstHalfOvers : Nat
  secondHalfOvers : Nat
  firstHalfOutfielders : Nat
  secondHalfOutfielders : Nat
  maxOversPerBowler : Nat

/-- Calculates the maximum runs possible in a cricket match under given constraints. -/
def maxRunsInMatch (m : CricketMatch) : Nat :=
  let runsPerOver := 6 * m.maxSixesPerOver + 4 * (6 - m.maxSixesPerOver)
  runsPerOver * m.totalOvers

/-- Theorem stating the maximum runs possible in the given cricket match scenario. -/
theorem max_runs_in_twenty_over_match :
  let m : CricketMatch := {
    totalOvers := 20,
    maxSixesPerOver := 3,
    firstHalfOvers := 10,
    secondHalfOvers := 10,
    firstHalfOutfielders := 2,
    secondHalfOutfielders := 5,
    maxOversPerBowler := 4
  }
  maxRunsInMatch m = 600 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_runs_in_twenty_over_match_l1143_114371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_max_no_min_l1143_114349

open Real

theorem omega_range_for_max_no_min (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ (cos (ω * x))^2 + 2 * sin (ω * x) * cos (ω * x) - (sin (ω * x))^2
  (∃ (x_max : ℝ), x_max ∈ Set.Ioo (π / 12) (π / 3) ∧
    ∀ (x : ℝ), x ∈ Set.Ioo (π / 12) (π / 3) → f x ≤ f x_max) ∧
  (∀ (x_min : ℝ), x_min ∈ Set.Ioo (π / 12) (π / 3) →
    ∃ (x : ℝ), x ∈ Set.Ioo (π / 12) (π / 3) ∧ f x < f x_min) →
  3/8 < ω ∧ ω < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_max_no_min_l1143_114349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_and_muffin_buyers_l1143_114347

theorem cake_and_muffin_buyers (total_buyers : ℕ) (cake_buyers : ℕ) (muffin_buyers : ℕ) 
  (prob_neither : ℚ) (h1 : total_buyers = 100) (h2 : cake_buyers = 50) (h3 : muffin_buyers = 40) 
  (h4 : prob_neither = 28/100) : 
  ∃ both_buyers : ℕ, both_buyers = 18 ∧ 
  cake_buyers + muffin_buyers - both_buyers + (prob_neither * ↑total_buyers).floor = total_buyers :=
by
  -- We use 'floor' instead of 'toNat' to convert the rational number to a natural number
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_and_muffin_buyers_l1143_114347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_distances_l1143_114338

-- Define the locations
noncomputable def adam : ℝ × ℝ := (2, -15)
noncomputable def ben : ℝ × ℝ := (-3, 10)
noncomputable def calum : ℝ × ℝ := (1, 5)
noncomputable def danielle : ℝ × ℝ := (1, 17)

-- Define the meeting point
noncomputable def meeting_point : ℝ × ℝ := ((adam.1 + ben.1) / 2, (adam.2 + ben.2) / 2)

-- Theorem statement
theorem walking_distances :
  (calum.2 - meeting_point.2 = 7.5) ∧
  (danielle.2 - calum.2 = 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_distances_l1143_114338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_event_attendance_l1143_114367

/-- The number of students attending the event -/
def num_students : ℕ := sorry

/-- The number of benches in the auditorium -/
def num_benches : ℕ := sorry

/-- When 9 people sit per bench, one student can't sit -/
axiom condition1 : num_students = 9 * num_benches + 1

/-- When 10 people sit per bench, all benches are full except one empty bench -/
axiom condition2 : num_students = 10 * num_benches - 10

/-- The number of students attending the event is 100 -/
theorem event_attendance : num_students = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_event_attendance_l1143_114367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_charge_difference_l1143_114356

/-- Represents a company's charges for an air conditioner -/
structure CompanyCharges where
  price : ℚ
  surchargePercent : ℚ
  installationCharge : ℚ

/-- Calculates the total charge for a company -/
def totalCharge (c : CompanyCharges) : ℚ :=
  c.price + (c.price * c.surchargePercent / 100) + c.installationCharge

/-- The charges for Company X -/
def companyX : CompanyCharges :=
  { price := 575
  , surchargePercent := 4
  , installationCharge := 82.50 }

/-- The charges for Company Y -/
def companyY : CompanyCharges :=
  { price := 530
  , surchargePercent := 3
  , installationCharge := 93.00 }

/-- Theorem stating the difference in total charges between Company X and Company Y -/
theorem total_charge_difference :
  totalCharge companyX - totalCharge companyY = 41.60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_charge_difference_l1143_114356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_difference_converges_to_zero_l1143_114314

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry

axiom a_recurrence : ∀ n : ℕ, n ≥ 1 → sequence_a (n + 1) = (sequence_b (n - 1) + sequence_b n) / 2
axiom b_recurrence : ∀ n : ℕ, n ≥ 1 → sequence_b (n + 1) = (sequence_a (n - 1) + sequence_a n) / 2

theorem sequence_difference_converges_to_zero :
  ∀ ε > 0, ∃ N : ℕ, ∀ n : ℕ, n > N → |sequence_a n - sequence_b n| < ε :=
by
  sorry

#check sequence_difference_converges_to_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_difference_converges_to_zero_l1143_114314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_increase_cube_cutting_l1143_114334

/-- Calculates the surface area of a cube given its side length -/
noncomputable def cubeSurfaceArea (sideLength : ℝ) : ℝ := 6 * sideLength^2

/-- Calculates the number of smaller cubes when cutting a larger cube -/
noncomputable def numberOfSmallerCubes (originalSideLength smallerSideLength : ℝ) : ℝ :=
  (originalSideLength / smallerSideLength)^3

/-- Calculates the percentage increase in surface area -/
noncomputable def percentageIncrease (originalArea newArea : ℝ) : ℝ :=
  ((newArea - originalArea) / originalArea) * 100

theorem surface_area_increase_cube_cutting :
  let originalSideLength : ℝ := 7
  let smallerSideLength : ℝ := 1
  let originalArea := cubeSurfaceArea originalSideLength
  let numSmallerCubes := numberOfSmallerCubes originalSideLength smallerSideLength
  let newTotalArea := numSmallerCubes * (cubeSurfaceArea smallerSideLength)
  percentageIncrease originalArea newTotalArea = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_increase_cube_cutting_l1143_114334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1143_114353

theorem sum_of_coefficients (a : ℝ) : 
  (∃ k : ℕ, (Nat.choose 8 k) * ((-a)^k) * (1^(8-2*k)) = 1120) →
  ((1 - a)^8 = 1 ∨ (1 - a)^8 = 6561) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1143_114353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_on_interval_l1143_114382

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin x * Real.sin (x + Real.pi / 6)

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

-- Theorem for the range of f(x) when x ∈ [0, π/2]
theorem range_on_interval :
  ∀ (y : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = y) ↔ 
  y ∈ Set.Icc 0 (1 / 2 + Real.sqrt 3 / 4) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_on_interval_l1143_114382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poly_identity_l1143_114396

/-- Polynomial of degree 5 with integer coefficients -/
def Poly5 (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) : ℝ → ℝ :=
  λ x => (a₅ : ℝ) * x^5 + (a₄ : ℝ) * x^4 + (a₃ : ℝ) * x^3 + (a₂ : ℝ) * x^2 + (a₁ : ℝ) * x + (a₀ : ℝ)

/-- Polynomial of degree 3 with integer coefficients -/
def Poly3 (b₀ b₁ b₂ b₃ : ℤ) : ℝ → ℝ :=
  λ x => (b₃ : ℝ) * x^3 + (b₂ : ℝ) * x^2 + (b₁ : ℝ) * x + (b₀ : ℝ)

/-- Polynomial of degree 2 with integer coefficients -/
def Poly2 (c₀ c₁ c₂ : ℤ) : ℝ → ℝ :=
  λ x => (c₂ : ℝ) * x^2 + (c₁ : ℝ) * x + (c₀ : ℝ)

theorem poly_identity 
  (a₀ a₁ a₂ a₃ a₄ a₅ b₀ b₁ b₂ b₃ c₀ c₁ c₂ : ℤ) 
  (h_a : ∀ i ∈ ({a₀, a₁, a₂, a₃, a₄, a₅} : Set ℤ), abs i ≤ 4)
  (h_b : ∀ i ∈ ({b₀, b₁, b₂, b₃} : Set ℤ), abs i ≤ 1)
  (h_c : ∀ i ∈ ({c₀, c₁, c₂} : Set ℤ), abs i ≤ 1)
  (h_eq : Poly5 a₀ a₁ a₂ a₃ a₄ a₅ 10 = Poly3 b₀ b₁ b₂ b₃ 10 * Poly2 c₀ c₁ c₂ 10) :
  ∀ x, Poly5 a₀ a₁ a₂ a₃ a₄ a₅ x = Poly3 b₀ b₁ b₂ b₃ x * Poly2 c₀ c₁ c₂ x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poly_identity_l1143_114396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_median_bisector_l1143_114359

/-- The angle between the median and angle bisector from vertex A in triangle ABC -/
noncomputable def angle_between_median_and_bisector (a b c : ℝ) : ℝ := sorry

/-- Given a triangle ABC with base a and sum of other sides b + c, 
    the angle between the median and angle bisector from A is maximized 
    when c - b = (a/2) * √2 -/
theorem max_angle_median_bisector (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) :
  let x := angle_between_median_and_bisector a b c
  ∀ b' c', b' + c' = b + c → 
    angle_between_median_and_bisector a b' c' ≤ x ↔ 
    c - b = a / 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_median_bisector_l1143_114359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1143_114374

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a m : ℝ) : 
  (∀ x, f a x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 ∧ m = 3 := by sorry

-- Part II
theorem part_two (t : ℝ) (h : t ≥ 0) :
  (∀ x, f 2 x + t ≥ f 2 (x + 2*t)) ↔
    (t = 0 ∧ ∀ x : ℝ, True) ∨
    (t > 0 ∧ ∀ x : ℝ, x ≤ 2 - t/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1143_114374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l1143_114340

/-- The speed of the man in kmph given the train's parameters --/
noncomputable def man_speed (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_length / crossing_time
  let man_speed_ms := relative_speed - train_speed_ms
  man_speed_ms * 3600 / 1000

/-- Theorem stating the speed of the man given the train's parameters --/
theorem man_speed_calculation :
  man_speed 100 54.99520038396929 6 = 5.00479961403071 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l1143_114340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_equation_l1143_114369

/-- Given a table of values for the polynomial ax - b, prove that x = 0 is the solution to ax = b - 2 --/
theorem solution_of_equation (a b : ℝ) (h : ℝ → ℝ) : ∃ (x : ℝ), x = 0 ∧ a * x = b - 2 :=
by
  -- Define the function h that represents the table of values
  have h_def : h = λ x ↦ if x = -3 then 4
                       else if x = -2 then 2
                       else if x = -1 then 0
                       else if x = 0 then -2
                       else if x = 1 then -4
                       else 0 := by sorry
  
  -- Prove that h(0) = -2
  have h_zero : h 0 = -2 := by sorry
  
  -- Prove that h(x) = ax - b for all x in the domain of h
  have h_eq : ∀ x, h x = a * x - b := by sorry
  
  -- Combine the above to prove that a * 0 - b = -2
  have eq_zero : a * 0 - b = -2 := by sorry
  
  -- Conclude that x = 0 is the solution to ax = b - 2
  exact ⟨0, rfl, by linarith⟩

#check solution_of_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_equation_l1143_114369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_intervals_f_center_of_symmetry_f_minimum_value_and_occurrences_l1143_114365

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := -2 * sin (3 * x + π / 4) + 5

-- Theorem for strictly increasing intervals
theorem f_strictly_increasing_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (2 * π * (k : ℝ) / 3 + π / 12) (2 * π * (k : ℝ) / 3 + 5 * π / 12)) :=
sorry

-- Theorem for center of symmetry
theorem f_center_of_symmetry (k : ℤ) :
  ∀ x : ℝ, f (-π / 12 + (k : ℝ) * π / 3 + x) = f (-π / 12 + (k : ℝ) * π / 3 - x) :=
sorry

-- Theorem for minimum value and its occurrences
theorem f_minimum_value_and_occurrences (k : ℤ) :
  (∀ x : ℝ, f x ≥ 3) ∧ (f (2 * π * (k : ℝ) / 3 + π / 12) = 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_intervals_f_center_of_symmetry_f_minimum_value_and_occurrences_l1143_114365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_split_contribution_l1143_114361

-- Define the earnings of the five friends
def earnings : List ℝ := [18, 23, 28, 35, 45]

-- Define the number of friends
def num_friends : ℕ := 5

-- Theorem to prove
theorem equal_split_contribution :
  let total := earnings.sum
  let equal_share := total / num_friends
  let max_earner := earnings.maximum?
  ∀ m, max_earner = some m → m - equal_share = 15.2 := by
  intro total equal_share max_earner m h_max
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_split_contribution_l1143_114361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_lines_through_A1_l1143_114333

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a line in 3D space -/
structure Line where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Calculates the angle between two lines -/
noncomputable def angle_between_lines (l1 l2 : Line) : ℝ :=
  sorry

/-- Checks if a line passes through a point -/
def line_passes_through (l : Line) (p : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- The main theorem: there are exactly 3 lines through A₁ forming 60° angles with AC and BC₁ -/
theorem three_lines_through_A1 (c : Cube) : 
  ∃! (lines : Finset Line), 
    lines.card = 3 ∧ 
    ∀ l ∈ lines, 
      line_passes_through l (c.vertices 4) ∧ 
      angle_between_lines l (Line.mk (c.vertices 0) (c.vertices 2 - c.vertices 0)) = Real.pi / 3 ∧
      angle_between_lines l (Line.mk (c.vertices 1) (c.vertices 6 - c.vertices 1)) = Real.pi / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_lines_through_A1_l1143_114333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_length_l1143_114321

-- Define the circle and points
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 25}

structure PointOnCircle where
  point : ℝ × ℝ
  on_circle : point ∈ Circle

def A : PointOnCircle := sorry
def B : PointOnCircle := sorry

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the condition that AB = 6
axiom AB_length : distance A.point B.point = 6

-- Define C as the midpoint of the minor arc AB
def C : PointOnCircle := sorry

-- State that C is the midpoint of the minor arc AB
axiom C_is_midpoint : ∃ (center : ℝ × ℝ), 
  center ∈ Circle ∧ 
  distance center C.point = distance center A.point ∧
  distance center C.point = distance center B.point

-- Theorem to prove
theorem AC_length : distance A.point C.point = Real.sqrt 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_length_l1143_114321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_inequality_l1143_114354

theorem solution_to_inequality : ∀ x : ℝ, x ∈ ({0, 3, 4, 6} : Set ℝ) → (x + 1 > 5 ↔ x = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_inequality_l1143_114354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_is_correct_l1143_114360

-- Define the function f
def f : Set ℝ → (ℝ → ℝ)
  | S => λ x => sorry -- We don't need to define the function explicitly for this proof

-- Define the property that f should satisfy
def satisfies_property (f : Set ℝ → (ℝ → ℝ)) (S : Set ℝ) : Prop :=
  ∀ x ∈ S, (1 / x) ∈ S ∧ f S x + f S (1 / x) = 3 * x

-- Define the largest set that satisfies the property
def largest_domain : Set ℝ :=
  {x : ℝ | x = 1 ∨ x = -1}

-- Theorem statement
theorem largest_domain_is_correct :
  (∃ f : Set ℝ → (ℝ → ℝ), satisfies_property f largest_domain) ∧
  ∀ (f : Set ℝ → (ℝ → ℝ)) (S : Set ℝ), satisfies_property f S → S ⊆ largest_domain :=
by
  sorry -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_is_correct_l1143_114360
