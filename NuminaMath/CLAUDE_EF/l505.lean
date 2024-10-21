import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l505_50550

-- Define the curve C
noncomputable def curve_C (φ : ℝ) : ℝ × ℝ := (3 * Real.cos φ, 2 * Real.sin φ)

-- Define the polar equation
noncomputable def polar_equation (θ : ℝ) : ℝ := 36 / (4 * (Real.cos θ) ^ 2 + 9 * (Real.sin θ) ^ 2)

-- Define a point on the curve
noncomputable def point_on_curve (θ : ℝ) : ℝ × ℝ := 
  let ρ := Real.sqrt (polar_equation θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Theorem statement
theorem curve_C_properties :
  -- Part 1: The polar equation is correct
  (∀ θ : ℝ, ∃ φ : ℝ, curve_C φ = point_on_curve θ) ∧
  -- Part 2: For perpendicular points, the sum of reciprocals of squared distances is constant
  (∀ θ : ℝ, 
    let A := point_on_curve θ
    let B := point_on_curve (θ + π/2)
    1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) = 13/36) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l505_50550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_grazing_area_l505_50526

/-- The area outside a regular triangle that can be reached by a rope tied to one vertex -/
noncomputable def grazing_area (side_length rope_length : ℝ) (π : ℝ) : ℝ :=
  let circle_area := π * rope_length^2
  let sector_area := (5/6) * circle_area
  let triangle_area := (Real.sqrt 3 / 4) * side_length^2
  let triangle_part := triangle_area / 3
  let grazing_per_vertex := sector_area - triangle_part
  3 * grazing_per_vertex

theorem sheep_grazing_area :
  let side_length : ℝ := 5
  let rope_length : ℝ := 7
  let π : ℝ := 3.14
  abs (grazing_area side_length rope_length π - 373.8252) < 0.0001 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_grazing_area_l505_50526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l505_50542

/-- The area of a triangle with sides a, b, and c using Heron's formula -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 26, 24, and 15 is approximately 175.95 -/
theorem triangle_area_specific : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |triangle_area 26 24 15 - 175.95| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l505_50542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_prime_power_l505_50586

/-- 
Given an arithmetic progression of integers a₁, a₂, ..., aₙ,
if i divides aᵢ for i = 1, 2, ..., n-1 and n does not divide aₙ,
then n is a prime power.
-/
theorem arithmetic_progression_prime_power (n : ℕ) (a : ℕ → ℤ) :
  (∃ d : ℤ, ∀ i : ℕ, i ≤ n → a i = a 1 + (i - 1) * d) →
  (∀ i : ℕ, i < n → (i : ℤ) ∣ a i) →
  ¬((n : ℤ) ∣ a n) →
  ∃ p : ℕ, Prime p ∧ ∃ k : ℕ, n = p ^ k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_prime_power_l505_50586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_one_two_zeros_condition_l505_50536

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log (x + 2) + Real.log a - 2

-- Part 1: Maximum value when a = 1 on [-1, 1]
theorem max_value_when_a_is_one :
  ∃ (x : ℝ), x ∈ Set.Icc (-1) 1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-1) 1 → f 1 y ≤ f 1 x ∧
  f 1 x = Real.exp 1 - Real.log 3 - 2 := by
  sorry

-- Part 2: Condition for two zeros
theorem two_zeros_condition :
  ∀ (a : ℝ), a > 0 →
  (∃ (x y : ℝ), x < y ∧ x > -2 ∧ f a x = 0 ∧ f a y = 0) ↔
  (a > 0 ∧ a < Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_one_two_zeros_condition_l505_50536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2009_l505_50537

def our_sequence (a : ℕ → ℤ) : Prop :=
  a 3 = 5 ∧ a 5 = 8 ∧ ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 7

theorem sequence_2009 (a : ℕ → ℤ) (h : our_sequence a) : a 2009 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2009_l505_50537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_color_invariance_l505_50517

def circle_arc_product (n m : ℕ) : ℚ :=
  2^(2*m - n)

def arc_value (a b : Bool) : ℚ :=
  match a, b with
  | true, true => 2
  | false, false => 1/2
  | _, _ => 1

def product_of_arcs (arrangement : List Bool) : ℚ :=
  let pairs := List.zip arrangement (arrangement.rotateRight 1)
  List.foldl (λ acc (a, b) => acc * arc_value a b) 1 pairs

theorem circle_color_invariance (n m : ℕ) (h : m ≤ n) :
  ∀ (arrangement : List Bool),
  arrangement.length = n →
  arrangement.count true = m →
  product_of_arcs arrangement = circle_arc_product n m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_color_invariance_l505_50517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l505_50583

/-- The sum of an arithmetic sequence with first term -25, last term 19, and common difference 2 is -69. -/
theorem arithmetic_sequence_sum : 
  ∀ (a : List Int), 
    a.length > 0 → 
    a.head! = -25 → 
    a[a.length - 1]! = 19 → 
    (∀ i, i ∈ Finset.range (a.length - 1) → a[i+1]! - a[i]! = 2) →
    a.sum = -69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l505_50583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upsize_cost_is_one_l505_50589

-- Define the base cost of the burger meal
noncomputable def base_cost : ℚ := 6

-- Define the number of days
def days : ℕ := 5

-- Define the total cost for all days
noncomputable def total_cost : ℚ := 35

-- Define the upsize cost as a function of the other parameters
noncomputable def upsize_cost : ℚ := (total_cost / days) - base_cost

-- Theorem to prove
theorem upsize_cost_is_one : upsize_cost = 1 := by
  -- Expand the definition of upsize_cost
  unfold upsize_cost
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upsize_cost_is_one_l505_50589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_passes_through_fixed_point_l505_50562

/-- A parabola defined by y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The line x + 2 = 0 -/
def line (x : ℝ) : Prop := x + 2 = 0

/-- A circle with center (a, b) and radius r -/
def myCircle (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

/-- The fixed point (2, 0) -/
def fixed_point : ℝ × ℝ := (2, 0)

theorem moving_circle_passes_through_fixed_point :
  ∀ (a b r : ℝ),
  parabola a b →
  (∃ (x : ℝ), line x ∧ myCircle x 0 a b r) →
  myCircle (fixed_point.1) (fixed_point.2) a b r :=
by
  sorry

#check moving_circle_passes_through_fixed_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_passes_through_fixed_point_l505_50562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_time_difference_l505_50501

-- Define the ferries and their properties
noncomputable def ferry_p_speed : ℝ := 6
noncomputable def ferry_p_time : ℝ := 3
noncomputable def ferry_q_speed : ℝ := ferry_p_speed + 3
noncomputable def ferry_p_distance : ℝ := ferry_p_speed * ferry_p_time
noncomputable def ferry_q_distance : ℝ := 3 * ferry_p_distance
noncomputable def ferry_q_time : ℝ := ferry_q_distance / ferry_q_speed

-- Theorem to prove
theorem ferry_time_difference :
  ferry_q_time - ferry_p_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_time_difference_l505_50501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_B_cardinality_l505_50546

def A (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (fun k => (Nat.factors (Nat.gcd n k)).length % 2 = 0)

def B (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (fun k => (Nat.factors (Nat.gcd n k)).length % 2 = 1)

theorem A_B_cardinality (n : ℕ) :
  (Even n → Finset.card (A n) = Finset.card (B n)) ∧
  (Odd n → Finset.card (A n) > Finset.card (B n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_B_cardinality_l505_50546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_invertible_labels_l505_50515

-- Define the function types
def CubicPolynomial := ℝ → ℝ
def LinearFunction := Set ℤ → ℤ
def TangentFunction := { x : ℝ // -Real.pi/2 < x ∧ x < Real.pi/2 } → ℝ
def HyperbolaFunction := { x : ℝ // x ≠ 0 } → ℝ

-- Define the specific functions
noncomputable def f2 : CubicPolynomial := sorry
noncomputable def f3 : LinearFunction := sorry
noncomputable def f4 : TangentFunction := sorry
noncomputable def f5 : HyperbolaFunction := sorry

-- Define invertibility
def isInvertible {α β : Type} (f : α → β) : Prop := sorry

-- Theorem statement
theorem product_of_invertible_labels :
  (isInvertible f3) ∧
  (isInvertible f4) ∧
  (isInvertible f5) ∧
  ¬(isInvertible f2) →
  3 * 4 * 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_invertible_labels_l505_50515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_power_sum_characterization_l505_50569

theorem integer_power_sum_characterization (x : ℝ) :
  (∀ n : ℤ, ∃ m : ℤ, x^n + x^(-n) = m) →
  ∃ k : ℤ, |k| ≥ 2 ∧ (x = (k + Real.sqrt (k^2 - 4)) / 2 ∨ x = (k - Real.sqrt (k^2 - 4)) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_power_sum_characterization_l505_50569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_complex_theorem_l505_50574

noncomputable def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

noncomputable def point_to_complex (p : ℝ × ℝ) : ℂ := ⟨p.1, p.2⟩

theorem midpoint_complex_theorem :
  let A : ℂ := 4 + 5*I
  let B : ℂ := -2 + I
  let C : ℂ := (A + B) / 2
  C = 1 + 3*I := by
    -- Proof steps would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_complex_theorem_l505_50574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l505_50538

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (speed platform_length crossing_time : ℝ) :
  speed = 72 → platform_length = 240 → crossing_time = 26 →
  (speed * (5/18) * crossing_time) - platform_length = 280 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l505_50538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_D_smallest_unshaded_area_l505_50585

/-- Unshaded area of Figure D -/
noncomputable def unshaded_area_D : ℝ := 12 - 4 * Real.pi

/-- Unshaded area of Figure E -/
noncomputable def unshaded_area_E : ℝ := 20 - Real.pi

/-- Unshaded area of Figure F -/
noncomputable def unshaded_area_F : ℝ := 15 - 4.5 * Real.pi

/-- Theorem stating that Figure D has the smallest unshaded area -/
theorem figure_D_smallest_unshaded_area :
  unshaded_area_D < unshaded_area_E ∧ unshaded_area_D < unshaded_area_F := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_D_smallest_unshaded_area_l505_50585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l505_50578

theorem triangle_angle_calculation (A B C : Real) (h_triangle : A + B + C = Real.pi)
  (h_acute : 0 < A ∧ A < Real.pi / 2)
  (h_eq1 : Real.sin A ^ 2 + Real.sin B ^ 2 + 2 * Real.cos A * Real.cos B * Real.sin C = 9/10)
  (h_eq2 : Real.sin B ^ 2 + Real.sin C ^ 2 + 2 * Real.cos B * Real.cos C * Real.sin A = 11/12)
  (h_eq3 : Real.sin C ^ 2 + Real.sin A ^ 2 + 2 * Real.cos C * Real.cos A * Real.sin B = 1/2) :
  Real.sin A = Real.sqrt 2 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l505_50578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lame_rook_paths_corner_vs_diagonal_l505_50568

/-- A chessboard is represented as a simple graph where vertices are cells and edges connect adjacent cells --/
def Chessboard : Type := SimpleGraph (Fin 8 × Fin 8)

/-- A corner cell of the chessboard --/
def CornerCell (board : Chessboard) : Fin 8 × Fin 8 := sorry

/-- A cell diagonally adjacent to a corner cell --/
def DiagonalAdjacentCell (board : Chessboard) (corner : Fin 8 × Fin 8) : Fin 8 × Fin 8 := sorry

/-- A Hamiltonian path on the chessboard representing a lame rook's traversal --/
def HamiltonianPath (board : Chessboard) : Type := List (Fin 8 × Fin 8)

/-- Predicate to check if a path is valid for a lame rook --/
def IsValidLameRookPath (board : Chessboard) (path : HamiltonianPath board) : Prop := sorry

/-- The number of valid Hamiltonian paths for a lame rook starting from a given cell --/
noncomputable def NumPaths (board : Chessboard) (start : Fin 8 × Fin 8) : ℕ := sorry

theorem lame_rook_paths_corner_vs_diagonal (board : Chessboard) :
  let A := CornerCell board
  let B := DiagonalAdjacentCell board A
  NumPaths board A > NumPaths board B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lame_rook_paths_corner_vs_diagonal_l505_50568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l505_50511

theorem function_property (f : ℕ → ℕ) 
  (h1 : ∀ n, f (f n) + f n = 2*n + 3)
  (h2 : f 0 = 1)
  (h3 : f 2014 = 2015) :
  f 2013 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l505_50511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_growth_x_squared_l505_50503

theorem faster_growth_x_squared (x : ℝ) (h : x > 1) :
  ∃ c : ℝ, c > 1 ∧ ∀ x' : ℝ, x' > c → (2 * x') > (Real.log x' + 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_growth_x_squared_l505_50503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandth_number_with_remainder_5_mod_8_l505_50543

theorem thousandth_number_with_remainder_5_mod_8 : 
  Finset.sup (Finset.range 1000) (λ n => 8*n + 5) = 7997 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandth_number_with_remainder_5_mod_8_l505_50543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_delivery_time_l505_50534

noncomputable def freshness_loss (t : ℝ) : ℝ :=
  if 0 ≤ t ∧ t < 10 then t^2 / 1000
  else if 10 ≤ t ∧ t ≤ 100 then (1/20) * 2^((20+t)/30)
  else 0

def log_2_3_approx : ℝ := 1.6

theorem max_delivery_time :
  ∃ (t : ℝ), t = 28 ∧
  (∀ (s : ℝ), 0 ≤ s ∧ s ≤ t → freshness_loss s ≤ 0.15) ∧
  (∀ (u : ℝ), u > t → freshness_loss u > 0.15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_delivery_time_l505_50534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_dimensions_l505_50575

/-- Given a right triangle DEF and an inscribed rectangle GHIJ, prove the dimensions of GHIJ -/
theorem inscribed_rectangle_dimensions (DE EF DF GH GI : ℝ) : 
  DE = 6 →
  EF = 8 →
  DF = 10 →
  (∃ (v : ℝ × ℝ), v.1 * GH = v.2 * DE) →  -- Parallelism condition
  GH * GI = (25/3) * (40/6) →
  (GH = 25/3 ∧ GI = 40/6) ∨ (GH = 40/6 ∧ GI = 25/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_dimensions_l505_50575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_iff_a_in_range_l505_50544

/-- A function f is monotonically increasing on ℝ if for all x₁, x₂ ∈ ℝ, x₁ < x₂ implies f(x₁) ≤ f(x₂) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

/-- The function f(x) = x - (1/3)sin(2x) + asin(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * Real.sin (2*x) + a * Real.sin x

theorem monotonic_f_iff_a_in_range :
  ∀ a : ℝ, MonotonicallyIncreasing (f a) ↔ -1/3 ≤ a ∧ a ≤ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_iff_a_in_range_l505_50544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_unique_intersection_point_l505_50521

-- Define the real number m > 1
variable (m : ℝ) (h_m : m > 1)

-- Define points A and B
def A (m : ℝ) : ℝ × ℝ := (-m, 0)
def B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the moving point S
variable (S : ℝ × ℝ)

-- Define the product of slopes condition
def slope_product (m : ℝ) (S : ℝ × ℝ) : Prop :=
  let (x, y) := S
  x ≠ m ∧ x ≠ -m ∧ (y / (x + m)) * (y / (x - m)) = -1 / m^2

-- Define the trajectory C
def trajectory_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m^2 + y^2 = 1

-- Theorem for part (1)
theorem trajectory_is_ellipse (m : ℝ) (h_m : m > 1) :
  ∀ x y, x ≠ m ∧ x ≠ -m → (slope_product m (x, y) ↔ trajectory_C m x y) :=
sorry

-- Theorem for part (2)
theorem unique_intersection_point (t : ℝ) :
  m = Real.sqrt 2 →
  t > 0 →
  (∃! p : ℝ × ℝ, trajectory_C (Real.sqrt 2) p.1 p.2 ∧ 2 * p.1 - p.2 + t = 0) ↔
  t = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_unique_intersection_point_l505_50521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l505_50581

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  straight_favorable : ℝ
  straight_unfavorable : ℝ
  bend_favorable : ℝ
  bend_unfavorable : ℝ

/-- Represents the environmental conditions -/
structure Environment where
  current_straight : ℝ
  current_bend : ℝ
  wind_impact : ℝ

/-- Calculates the average speed against the current -/
noncomputable def average_speed_against_current (boat : BoatSpeed) (env : Environment) : ℝ :=
  let straight_speed := (boat.straight_favorable - env.current_straight - env.wind_impact + 
                         boat.straight_unfavorable - env.current_straight + env.wind_impact) / 2
  let bend_speed := (boat.bend_favorable - env.current_bend - env.wind_impact + 
                     boat.bend_unfavorable - env.current_bend + env.wind_impact) / 2
  (straight_speed + bend_speed) / 2

/-- The theorem to be proved -/
theorem average_speed_theorem (boat : BoatSpeed) (env : Environment) :
  boat.straight_favorable = 15 ∧ 
  boat.straight_unfavorable = 12 ∧ 
  boat.bend_favorable = 10 ∧ 
  boat.bend_unfavorable = 7 ∧
  env.current_straight = 2.5 ∧ 
  env.current_bend = 3.5 ∧ 
  env.wind_impact = 1 →
  average_speed_against_current boat env = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_theorem_l505_50581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smith_total_payment_l505_50502

/-- Calculates the total amount Mr. Smith will pay after three months, given his unpaid balances and their respective interest rates. -/
theorem smith_total_payment (balance1 balance2 balance3 : ℚ)
  (rate1_month1 rate1_month2 rate1_month3 : ℚ)
  (rate2_month2 rate2_month3 : ℚ)
  (rate3 : ℚ) :
  let interest1 := balance1 * (rate1_month1 + rate1_month2 + rate1_month3)
  let interest2 := balance2 * (rate2_month2 + rate2_month3)
  let interest3 := balance3 * (3 * rate3)
  let total := (balance1 + interest1) + (balance2 + interest2) + (balance3 + interest3)
  balance1 = 150 →
  balance2 = 220 →
  balance3 = 75 →
  rate1_month1 = 2/100 →
  rate1_month2 = 1/100 →
  rate1_month3 = 5/1000 →
  rate2_month2 = 1/100 →
  rate2_month3 = 5/1000 →
  rate3 = 15/1000 →
  ⌈total * 100⌉ / 100 = 45693/100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smith_total_payment_l505_50502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_cost_per_mile_l505_50577

/-- Calculates the cost per mile for a car rental given the daily rate, total payment, and miles driven -/
theorem car_rental_cost_per_mile 
  (daily_rate : ℚ) 
  (total_payment : ℚ) 
  (miles_driven : ℚ) 
  (h1 : daily_rate = 29)
  (h2 : total_payment = 46.12)
  (h3 : miles_driven = 214)
  : (total_payment - daily_rate) / miles_driven = 0.08 := by
  sorry

#eval (46.12 - 29) / 214

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_cost_per_mile_l505_50577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_iff_l505_50565

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2/2 - Real.log x

-- Define the property of f being not monotonic on an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (x y z : ℝ), a < x ∧ x < y ∧ y < z ∧ z < b ∧
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- State the theorem
theorem f_not_monotonic_iff (m : ℝ) :
  not_monotonic f m (m + 1/2) ↔ 1/2 < m ∧ m < 1 := by
  sorry

#check f_not_monotonic_iff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_iff_l505_50565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_weight_l505_50548

/-- The acceleration due to gravity on Earth -/
noncomputable def g : ℝ := 9.81

/-- The force applied to the string -/
def F : ℝ := 50

/-- The mass of the pulley -/
noncomputable def M : ℝ := 100 / g

/-- The radius of the pulley -/
noncomputable def r : ℝ := F * 2 / (M * g)

/-- The moment of inertia of a solid cylindrical pulley -/
noncomputable def I : ℝ := (1 / 2) * M * r^2

/-- The angular acceleration of the pulley -/
noncomputable def α : ℝ := g / r

/-- The weight of the pulley -/
noncomputable def W : ℝ := M * g

theorem pulley_weight :
  (F * r = I * α) ∧ (W = 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_weight_l505_50548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l505_50535

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  ab : ℝ
  cd : ℝ
  distance : ℝ
  angle : ℝ

/-- The volume of a tetrahedron with given properties -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  1/3 * t.ab * t.cd * t.distance * Real.sin (t.angle * Real.pi / 180)

/-- Theorem stating the volume of a specific tetrahedron -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    ab := 1,
    cd := 3,
    distance := 2,
    angle := 60
  }
  tetrahedron_volume t = Real.sqrt 3 := by
  sorry

#eval "Tetrahedron volume theorem loaded successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l505_50535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_bound_l505_50507

open Real

-- Define the set M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.sin (12 * p.1 + 5 * p.2) = Real.sin (12 * p.1) + Real.sin (5 * p.2)}

-- Define a circle with center c and radius r
def Circle (c : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

-- Theorem statement
theorem circle_radius_bound
  (c : ℝ × ℝ) (r : ℝ)
  (h : ∀ p ∈ M, p ∉ Circle c r) :
  0 < r ∧ r < π / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_bound_l505_50507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_loss_percentage_l505_50564

/-- Proves that the loss percentage is 10% when a car is sold at a loss and then resold with a 20% gain -/
theorem car_sale_loss_percentage 
  (original_price : ℝ) 
  (friend_selling_price : ℝ) 
  (friend_gain_percentage : ℝ) 
  (h1 : original_price = 50000)
  (h2 : friend_selling_price = 54000)
  (h3 : friend_gain_percentage = 20) :
  (original_price - friend_selling_price / (1 + friend_gain_percentage / 100)) / original_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_loss_percentage_l505_50564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_377_l505_50504

theorem greatest_prime_factor_of_377 : 
  (Nat.factors 377).maximum? = some 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_377_l505_50504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l505_50508

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + 2 * (Real.cos x)^2

-- State the theorem
theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = π) ∧
  -- Monotonically decreasing interval
  (∀ k : ℤ, ∀ x y : ℝ, x ≥ π/8 + k*π → y ≤ 5*π/8 + k*π → x ≤ y → f x ≥ f y) ∧
  -- Set of x values where f(x) ≥ 3
  (∀ x : ℝ, f x ≥ 3 ↔ ∃ k : ℤ, k*π ≤ x ∧ x ≤ π/4 + k*π) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l505_50508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l505_50553

/-- The speed of a train given distance and time -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- The theorem stating the speed of the first train -/
theorem first_train_speed 
  (ratio : ℚ) -- Ratio of speeds between the two trains
  (distance_second : ℝ) -- Distance traveled by the second train
  (time_second : ℝ) -- Time taken by the second train
  (h1 : ratio = 7 / 8) -- The ratio is 7:8
  (h2 : distance_second = 400) -- The second train travels 400 km
  (h3 : time_second = 4) -- The second train takes 4 hours
  : speed (ratio * distance_second) time_second = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l505_50553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l505_50560

/-- The area of a square inscribed in a specific ellipse -/
theorem inscribed_square_area : 
  ∃ (t : ℝ), t > 0 ∧ 
    (∀ (x y : ℝ), x^2 + y^2 = 2*t^2 → x^2/4 + y^2/8 = 1) ∧
    2*t^2 = 32/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_l505_50560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l505_50582

/-- Represents an arrangement of integers -/
def Arrangement := List Nat

/-- Checks if an arrangement is valid according to the problem conditions -/
def is_valid_arrangement (arr : Arrangement) : Prop :=
  (arr.length = 100) ∧ 
  (∀ k : Nat, k ≥ 1 ∧ k ≤ 50 → arr.count k = 2) ∧
  (∀ k : Nat, k ≥ 1 ∧ k ≤ 50 → 
    ∃ i j : Nat, i < j ∧ arr.get! i = k ∧ arr.get! j = k ∧ j - i - 1 = k)

/-- The main theorem stating that no valid arrangement exists -/
theorem no_valid_arrangement : ¬∃ arr : Arrangement, is_valid_arrangement arr := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l505_50582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l505_50540

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (|x + 1| + |x + 2| - 5)

def A : Set ℝ := {x | x ≤ -4 ∨ x ≥ 1}

theorem inequality_proof (a b : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) :
  |a + b| / 2 < |1 + a * b / 4| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l505_50540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_restoration_theorem_l505_50594

-- Define the discount percentages
def initial_discount : ℚ := 25 / 100
def second_discount : ℚ := 20 / 100
def limited_time_discount : ℚ := 10 / 100

-- Define the price after each discount
noncomputable def price_after_initial (original_price : ℚ) : ℚ := original_price * (1 - initial_discount)
noncomputable def price_after_second (original_price : ℚ) : ℚ := price_after_initial original_price * (1 - second_discount)
noncomputable def price_after_limited (original_price : ℚ) : ℚ := price_after_second original_price * (1 - limited_time_discount)

-- Define the required increase percentages
def required_increase_without_limited : ℚ := 200 / 3
def required_increase_with_limited : ℚ := 4600 / 54

-- Theorem statement
theorem price_restoration_theorem (original_price : ℚ) (original_price_pos : original_price > 0) :
  price_after_second original_price * (1 + required_increase_without_limited / 100) = original_price ∧
  price_after_limited original_price * (1 + required_increase_with_limited / 100) = original_price :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_restoration_theorem_l505_50594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l505_50595

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : S 3 = 12)
  (h3 : ∀ n, S n = arithmetic_sum (a 1) ((a 2) - (a 1)) n)
  (h4 : ∀ n, a n = arithmetic_sequence (a 1) ((a 2) - (a 1)) n) :
  (a 24 = 70 ∧ S 7 = 70) ∧
  (∀ n : ℕ, (∃ m : ℕ, a m = S n) ↔ ∃ k : ℕ, n = 3 * k + 1) := by
  sorry

#check arithmetic_sequence_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l505_50595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l505_50509

/-- Given a geometric sequence {a_n} with common ratio q ≠ 1, a_1 = 1/2, and S_n as the sum of the first n terms. 
    If a_2 + S_2, a_3 + S_3, a_4 + S_4 form an arithmetic sequence, then a_n + S_n = 1 for all n ≥ 1. -/
theorem geometric_sequence_sum (q : ℝ) (hq : q ≠ 1) : 
  let a : ℕ → ℝ := fun n => (1/2) * q^(n-1)
  let S : ℕ → ℝ := fun n => (1/2) * (1 - q^n) / (1 - q)
  (2 * (a 3 + S 3) = (a 2 + S 2) + (a 4 + S 4)) → 
  ∀ n : ℕ, n ≥ 1 → a n + S n = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l505_50509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l505_50519

theorem trig_identity (α : ℝ) (h : Real.sin (α + π / 12) = 1 / 3) :
  Real.cos (α + 7 * π / 12) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l505_50519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_MAB_l505_50500

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- The directrix of the parabola -/
def directrix (x : ℝ) : Prop := x = -1

/-- Point M: intersection of directrix and x-axis -/
def M : ℝ × ℝ := (-1, 0)

/-- Point A: upper intersection of parabola and line through focus -/
def A : ℝ × ℝ := (1, 2)

/-- Point B: lower intersection of parabola and line through focus -/
def B : ℝ × ℝ := (1, -2)

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

/-- Theorem: The circle passing through M, A, and B has the equation (x - 1)^2 + y^2 = 4 -/
theorem circle_through_MAB :
  circle_eq M.1 M.2 ∧ circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_MAB_l505_50500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_3_minus_4i_l505_50556

theorem modulus_of_3_minus_4i :
  Complex.abs (3 - 4 * Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_3_minus_4i_l505_50556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_FAB_l505_50590

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The left focus F -/
def F : ℝ × ℝ := (-1, 0)

/-- A point on the line l -/
def point_on_l : ℝ × ℝ := (1, 1)

/-- Points A and B are on the ellipse -/
def A_on_ellipse (A : ℝ × ℝ) : Prop := ellipse A.1 A.2
def B_on_ellipse (B : ℝ × ℝ) : Prop := ellipse B.1 B.2

/-- A and B are on the line l -/
def A_B_on_l (A B : ℝ × ℝ) : Prop :=
  A = point_on_l ∨ B = point_on_l ∨
  (A.2 - point_on_l.2) / (A.1 - point_on_l.1) = (B.2 - point_on_l.2) / (B.1 - point_on_l.1)

/-- The area of triangle FAB -/
noncomputable def area_FAB (A B : ℝ × ℝ) : ℝ :=
  (1/2) * abs (F.1 * (A.2 - B.2) + A.1 * (B.2 - F.2) + B.1 * (F.2 - A.2))

/-- The theorem statement -/
theorem max_area_FAB :
  ∀ A B : ℝ × ℝ,
  A_on_ellipse A → B_on_ellipse B → A_B_on_l A B →
  area_FAB A B ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_FAB_l505_50590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_minimal_values_l505_50572

-- Define monic quadratic polynomials
def monicQuadratic (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

-- Define the polynomials R and S
noncomputable def R : ℝ → ℝ := monicQuadratic 112 280
noncomputable def S : ℝ → ℝ := monicQuadratic 32 100

-- Define the composite functions
noncomputable def R_of_S : ℝ → ℝ := fun x ↦ R (S x)
noncomputable def S_of_R : ℝ → ℝ := fun x ↦ S (R x)

-- State the theorem
theorem sum_of_minimal_values :
  (∀ x ∈ ({-20, -18, -14, -12} : Set ℝ), R_of_S x = 0) ∧
  (∀ x ∈ ({-61, -59, -53, -51} : Set ℝ), S_of_R x = 0) →
  (R (-56) + S (-16) = -3114) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_minimal_values_l505_50572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l505_50516

/-- A parabola is defined by its quadratic equation coefficients -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ :=
  let h := -p.b / (2 * p.a)
  let k := p.c - p.b^2 / (4 * p.a)
  (h, k + 1 / (4 * p.a))

/-- Theorem: The focus of the parabola y = 2x^2 + 8x - 1 is at (-2, -71/8) -/
theorem focus_of_specific_parabola :
  let p : Parabola := { a := 2, b := 8, c := -1 }
  focus p = (-2, -71/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l505_50516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l505_50539

/-- Hyperbola C with equation x²/4 - y²/b² = 1 (b > 0) -/
def Hyperbola (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 / b^2 = 1 ∧ b > 0}

/-- Asymptote of the hyperbola -/
def Asymptote : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (Real.sqrt 6 / 2) * p.1}

/-- Left focus of the hyperbola -/
noncomputable def LeftFocus (b : ℝ) : ℝ × ℝ := sorry

/-- Right focus of the hyperbola -/
noncomputable def RightFocus (b : ℝ) : ℝ × ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem hyperbola_property (b : ℝ) (P : ℝ × ℝ) :
  P ∈ Hyperbola b →
  Asymptote.Nonempty →
  distance P (LeftFocus b) / distance P (RightFocus b) = 3 →
  distance P (LeftFocus b) + distance P (RightFocus b) = 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l505_50539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_63_l505_50573

/-- Represents the price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- Represents the price of a table in dollars -/
def table_price : ℝ := sorry

/-- The price of 2 chairs and 1 table is 60% of the price of 1 chair and 2 tables -/
axiom price_relation : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The price of 1 table and 1 chair is $72 -/
axiom total_price : chair_price + table_price = 72

/-- Theorem stating that the price of a table is $63 -/
theorem table_price_is_63 : table_price = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_63_l505_50573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_prism_side_length_l505_50596

-- Define the prism
structure Prism where
  base_side : ℝ
  height : ℝ

-- Define the volume function for the prism
noncomputable def volume (p : Prism) : ℝ :=
  (1/2) * p.base_side^2 * p.height

-- Theorem statement
theorem isosceles_right_prism_side_length 
  (p : Prism) 
  (h1 : p.height = 10) 
  (h2 : volume p = 25) : 
  p.base_side = Real.sqrt 5 := by
  sorry

#check isosceles_right_prism_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_prism_side_length_l505_50596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_eleven_dividing_factorial_200_l505_50514

theorem largest_power_of_eleven_dividing_factorial_200 :
  (11 : ℕ) ^ 19 ∣ Nat.factorial 200 ∧ ¬((11 : ℕ) ^ 20 ∣ Nat.factorial 200) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_eleven_dividing_factorial_200_l505_50514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_N_values_l505_50563

/-- Represents the total number of students in the class -/
def N : ℕ → ℕ := id

/-- Represents the number of bullies in the class -/
def num_bullies : ℕ := 8

/-- Represents the number of honor students in the class -/
def num_honor_students (n : ℕ) : ℕ := n - num_bullies

/-- Represents the condition for a bully's statement to be a lie -/
def bully_condition (n : ℕ) : Prop :=
  (num_bullies : ℚ) / ((n : ℚ) - 1) < 1 / 3

/-- Represents the condition for an honor student's statement to be true -/
def honor_condition (n : ℕ) : Prop :=
  (num_bullies : ℚ) / ((n : ℚ) - 1) ≥ 1 / 3

/-- Theorem stating the possible values for N -/
theorem possible_N_values (n : ℕ) :
  (n = 23 ∨ n = 24 ∨ n = 25) ↔
  (n > num_bullies ∧ bully_condition n ∧ honor_condition n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_N_values_l505_50563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_not_complementary_l505_50524

-- Define the ball colors
inductive Color where
  | Red
  | White
  | Black
deriving DecidableEq

-- Define the box contents
def box : Multiset Color := 
  Multiset.replicate 3 Color.Red + Multiset.replicate 2 Color.White + Multiset.replicate 1 Color.Black

-- Define the event of drawing at least one white ball
def atLeastOneWhite (draw : Multiset Color) : Prop :=
  (Multiset.count Color.White draw) ≥ 1

-- Define the event of drawing one red ball and one black ball
def oneRedOneBlack (draw : Multiset Color) : Prop :=
  (Multiset.count Color.Red draw = 1) ∧ (Multiset.count Color.Black draw = 1)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  ∃ (draw : Multiset Color), 
    Multiset.card draw = 2 ∧ 
    draw ⊆ box ∧
    (¬(atLeastOneWhite draw ∧ oneRedOneBlack draw)) ∧
    (∃ (other_draw : Multiset Color), 
      Multiset.card other_draw = 2 ∧ 
      other_draw ⊆ box ∧
      ¬(atLeastOneWhite other_draw ∨ oneRedOneBlack other_draw)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_not_complementary_l505_50524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_l505_50599

theorem division_with_remainder (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x % y = 3)
  (h2 : (x : ℝ) / (y : ℝ) = 96.12) : 
  y = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_l505_50599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_panda_bears_count_l505_50579

theorem panda_bears_count :
  ∃ (x : ℕ),
    let small_panda_count := x
    let big_panda_count : ℕ := 5
    let small_panda_daily_consumption : ℕ := 25
    let big_panda_daily_consumption : ℕ := 40
    let weekly_total_consumption : ℕ := 2100
    small_panda_daily_consumption * small_panda_count + 
    big_panda_daily_consumption * big_panda_count = 
    weekly_total_consumption / 7 ∧
    x = 4 :=
by
  sorry

#check panda_bears_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_panda_bears_count_l505_50579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_significant_digits_l505_50505

-- Define the area of the square
def square_area : ℝ := 2.401

-- Define the precision of the area measurement (to the nearest thousandth)
def area_precision : ℝ := 0.001

-- Define a function to calculate the number of significant digits
def count_significant_digits (x : ℝ) : ℕ := sorry

-- Define a function to calculate the side length of a square given its area
noncomputable def square_side_length (area : ℝ) : ℝ := Real.sqrt area

-- Theorem statement
theorem side_length_significant_digits :
  count_significant_digits (square_side_length square_area) = 4 := by
  sorry

#eval square_area
#eval area_precision

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_significant_digits_l505_50505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_area_l505_50570

theorem l_shaped_area
  (full_length : ℝ)
  (full_width : ℝ)
  (removed_length_diff : ℝ)
  (removed_width_diff : ℝ)
  (h1 : full_length = 10)
  (h2 : full_width = 7)
  (h3 : removed_length_diff = 3)
  (h4 : removed_width_diff = 4) :
  full_length * full_width - (full_length - removed_length_diff) * (full_width - removed_width_diff) = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_area_l505_50570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_extrema_implies_a_b_values_l505_50559

/-- A function y defined in terms of x, a, and b -/
noncomputable def y (x a b : ℝ) : ℝ := (a * x^2 - 8*x + b) / (x^2 + 1)

/-- The maximum value of y -/
def y_max : ℝ := 9

/-- The minimum value of y -/
def y_min : ℝ := 1

/-- Theorem stating that if y has a maximum of 9 and a minimum of 1, then a = 5 and b = 5 -/
theorem y_extrema_implies_a_b_values (a b : ℝ) :
  (∀ x : ℝ, y x a b ≤ y_max) ∧
  (∀ x : ℝ, y x a b ≥ y_min) ∧
  (∃ x1 x2 : ℝ, y x1 a b = y_max ∧ y x2 a b = y_min) →
  a = 5 ∧ b = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_extrema_implies_a_b_values_l505_50559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_attitude_gender_relationship_expected_value_X_l505_50587

/-- Represents the contingency table data -/
structure ContingencyTable :=
  (male_agree : ℕ)
  (male_disagree : ℕ)
  (female_agree : ℕ)
  (female_disagree : ℕ)

/-- Calculates the chi-square statistic -/
def chi_square (ct : ContingencyTable) : ℚ :=
  let n := ct.male_agree + ct.male_disagree + ct.female_agree + ct.female_disagree
  let ad := (ct.male_agree : ℚ) * ct.female_disagree
  let bc := (ct.female_agree : ℚ) * ct.male_disagree
  let num := (n : ℚ) * (ad - bc)^2
  let denom := ((ct.male_agree + ct.female_agree) * (ct.male_disagree + ct.female_disagree) *
                (ct.male_agree + ct.male_disagree) * (ct.female_agree + ct.female_disagree) : ℚ)
  num / denom

/-- The critical value for 99% confidence level -/
def critical_value : ℚ := 6635 / 1000

/-- Theorem stating the relationship between attitudes and gender -/
theorem attitude_gender_relationship (ct : ContingencyTable) 
  (h1 : ct.male_agree = 70)
  (h2 : ct.male_disagree = 30)
  (h3 : ct.female_agree = 50)
  (h4 : ct.female_disagree = 50) :
  chi_square ct > critical_value := by sorry

/-- Expected value of a binomial distribution -/
def expected_value (n : ℕ) (p : ℚ) : ℚ := (n : ℚ) * p

/-- Theorem stating the expected value of X -/
theorem expected_value_X :
  expected_value 3 (3/5) = 9/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_attitude_gender_relationship_expected_value_X_l505_50587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_table_sum_l505_50584

def numbers : List ℕ := [1, 4, 6, 8, 9, 10]

def is_valid_arrangement (top : List ℕ) (left : List ℕ) : Prop :=
  top.length = 3 ∧ left.length = 3 ∧ (top ++ left).toFinset = numbers.toFinset

def table_sum (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum * left.sum)

theorem max_table_sum :
  ∀ (top left : List ℕ), is_valid_arrangement top left →
    table_sum top left ≤ 361 :=
  sorry

#check max_table_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_table_sum_l505_50584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_scores_product_l505_50567

def shooting_scores (a b : ℚ) : Finset ℚ := {6, a, 10, 8, b}

theorem shooting_scores_product (a b : ℚ) : 
  (Finset.card (shooting_scores a b) = 5) →
  ((Finset.sum (shooting_scores a b) id) / 5 = 8) →
  ((Finset.sum (shooting_scores a b) (λ x => (x - 8)^2)) / 5 = 8/5) →
  a * b = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_scores_product_l505_50567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_problem_l505_50552

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def trapezium_area (t : Trapezium) : ℝ := (t.side1 + t.side2) * t.height / 2

/-- Theorem statement for the trapezium problem -/
theorem trapezium_problem (t : Trapezium) 
  (h1 : t.side1 = 18)
  (h2 : t.height = 15)
  (h3 : t.area = 285)
  (h4 : trapezium_area t = t.area) :
  t.side2 = 20 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_problem_l505_50552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seven_l505_50557

/-- Represents a number with 2023 repetitions of a two-digit number -/
def repeatedNumber (a b : ℕ) : ℕ :=
  (a * 10 + b) * (10^2023 - 1) / 99

/-- The main theorem stating the divisibility condition -/
theorem divisibility_by_seven :
  ∃ (x : ℕ), x < 10 ∧ (repeatedNumber 6 6 * repeatedNumber 5 5) % 7 = 0 ∧ x = 6 := by
  sorry

#check divisibility_by_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seven_l505_50557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l505_50545

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem f_sum_property (x₁ x₂ : ℝ) (h : x₁ + x₂ = 1) :
  f x₁ + f x₂ = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l505_50545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l505_50580

theorem cos_pi_minus_alpha (α : ℝ) (h1 : Real.sin α = 5/13) (h2 : 0 < α ∧ α < π/2) :
  Real.cos (π - α) = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l505_50580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_is_fibonacci_l505_50593

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- Define R_m
def R (m : ℕ) : ℕ := 
  Finset.prod (Finset.range (fib m)) (fun k => k^k) % fib m

-- Theorem statement
theorem R_is_fibonacci (m : ℕ) (h : m > 2) : 
  ∃ n, R m = fib n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_is_fibonacci_l505_50593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l505_50520

-- Define the function f
noncomputable def f : ℝ → ℝ := λ x => 
  if x < 0 then x * (x - 1) else -x * (x + 1)

-- State the theorem
theorem odd_function_extension :
  (∀ x, f (-x) = -f x) ∧                  -- f is an odd function
  (∀ x, x < 0 → f x = x * (x - 1)) →      -- f(x) = x(x-1) for x < 0
  (∀ x, x > 0 → f x = -x * (x + 1)) :=    -- f(x) = -x(x+1) for x > 0
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l505_50520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_difference_l505_50527

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - a*Real.log x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a x + (a + 2)*Real.log x - (a + 2*b - 2)*x

-- State the theorem
theorem min_value_of_g_difference (a b : ℝ) (x₁ x₂ : ℝ) :
  b ≥ 1 + 4*Real.sqrt 3/3 →
  x₁ < x₂ →
  (∀ x, x ≠ x₁ → x ≠ x₂ → (deriv (g a b)) x ≠ 0) →
  (deriv (g a b)) x₁ = 0 →
  (deriv (g a b)) x₂ = 0 →
  g a b x₁ - g a b x₂ ≥ 8/3 - 2*Real.log 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_difference_l505_50527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l505_50558

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x+a)ln((2x-1)/(2x+1)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

/-- If f(x) = (x+a)ln((2x-1)/(2x+1)) is an even function, then a = 0 -/
theorem even_function_implies_a_zero (a : ℝ) :
    IsEven (f a) → a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l505_50558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_div_2_pow_2011_l505_50598

/-- Defines a function that generates a nested square root expression with n levels -/
noncomputable def nestedSqrt (n : ℕ) : ℝ :=
  match n with
  | 0 => 2
  | n + 1 => Real.sqrt (2 + nestedSqrt n)

/-- Theorem stating the relationship between sin(π/2^2011) and a nested square root expression -/
theorem sin_pi_div_2_pow_2011 :
  Real.sin (π / 2^2011) = (Real.sqrt (2 - nestedSqrt 2009)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_div_2_pow_2011_l505_50598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_ordinate_zero_if_three_solutions_l505_50566

/-- A quadratic polynomial type -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a point -/
noncomputable def QuadraticPolynomial.eval (f : QuadraticPolynomial) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The equation (f(x))^3 - f(x) = 0 has exactly three solutions -/
def has_three_solutions (f : QuadraticPolynomial) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (f.eval x)^3 - f.eval x = 0 ∧
    (f.eval y)^3 - f.eval y = 0 ∧
    (f.eval z)^3 - f.eval z = 0 ∧
    ∀ w, (f.eval w)^3 - f.eval w = 0 → w = x ∨ w = y ∨ w = z

/-- The ordinate of the vertex of a quadratic polynomial -/
noncomputable def vertex_ordinate (f : QuadraticPolynomial) : ℝ :=
  f.c - (f.b^2) / (4 * f.a)

theorem vertex_ordinate_zero_if_three_solutions (f : QuadraticPolynomial) 
  (h : has_three_solutions f) : vertex_ordinate f = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_ordinate_zero_if_three_solutions_l505_50566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_line_intersects_circle_l505_50576

-- Define the points A and B
def A : ℝ × ℝ := (-2, 3)
def B (a : ℝ) : ℝ × ℝ := (0, a)

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := (x + 3)^2 + (y + 2)^2 = 1

-- Define the symmetrical line
def on_symmetrical_line (a x y : ℝ) : Prop :=
  (3 - a) * x - 2 * y + 2 * a = 0

-- Main theorem
theorem symmetrical_line_intersects_circle (a : ℝ) :
  (∃ x y : ℝ, on_symmetrical_line a x y ∧ is_on_circle x y) →
  1/3 ≤ a ∧ a ≤ 3/2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_line_intersects_circle_l505_50576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_A_l505_50506

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ := cos x * (a * sin x - cos x) + (cos (π / 2 - x))^2

theorem range_of_f_A (a b c : ℝ) (A B C : ℝ) :
  f (-π/3) = f 0 →
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  (a^2 + c^2 - b^2) / (a^2 + b^2 - c^2) = c / (2*a - c) →
  ∃ (y : ℝ), 1 < y ∧ y ≤ 2 ∧ f (2*sqrt 3) A = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_A_l505_50506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_equivalence_l505_50549

theorem remainder_equivalence (a p : ℕ) (h_prime : Nat.Prime p) :
  (∀ m n : ℕ, m > 0 → n > 0 → (a ^ (2 ^ n) % p ^ n ≠ 0) ∧
    (a ^ (2 ^ n) % p ^ n = a ^ (2 ^ m) % p ^ m)) ↔
  (p = 2 ∧ ∃ k : ℕ, a = 2 * k + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_equivalence_l505_50549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_five_thirtysix_l505_50551

/-- The sum of the infinite series Σ(1 / (n(n+1)(n+3))) for n from 1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' n : ℕ+, (1 : ℝ) / (n * (n + 1) * (n + 3))

/-- Theorem stating that the infinite series sum equals 5/36 -/
theorem infiniteSeries_eq_five_thirtysix : infiniteSeries = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_five_thirtysix_l505_50551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_triangle_l505_50530

/-- Given an ellipse with short axis AB and focus F₁, if triangle ABF₁ is equilateral,
    then the eccentricity of the ellipse is √3/2. -/
theorem ellipse_eccentricity_equilateral_triangle (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensuring positive values
  a = 2*b →  -- Major axis is twice the minor axis (equilateral triangle condition)
  a^2 = b^2 + c^2 →  -- Ellipse equation
  c / a = Real.sqrt 3 / 2 :=  -- Eccentricity is √3/2
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_triangle_l505_50530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l505_50571

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 2*m^2 - 4

theorem quadratic_function_properties (m : ℝ) :
  (∃! x, f m (2^x) ≤ 0) = (m = 2 ∨ -Real.sqrt 2 < m ∧ m < Real.sqrt 2) ∧
  (∃ x > 2, f m x = 0) = (m > 0 ∧ m < 2) ∧
  (∀ x > 2, f m x ≠ 0) = (m ≤ 0 ∨ m ≥ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l505_50571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_range_l505_50525

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the axis of symmetry
def axis_of_symmetry (x : ℝ) : Prop := x = -1

-- Define a point Q on the axis of symmetry
def point_Q (t : ℝ) : ℝ × ℝ := (-1, t)

-- Define the line l passing through Q and perpendicular to OQ
def line_l (t : ℝ) (x y : ℝ) : Prop := x - t*y + t^2 + 1 = 0

-- Define the distance function from a point to a line
noncomputable def distance_to_line (px py t : ℝ) : ℝ :=
  abs (px - t*py + t^2) / Real.sqrt (1 + t^2)

theorem min_distance_range :
  ∀ t : ℝ, t ≠ 0 →
    ∃ min_d : ℝ, 
      (∀ px py : ℝ, parabola px py → distance_to_line px py t ≥ min_d) ∧
      0 < min_d ∧ min_d < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_range_l505_50525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_ratio_l505_50541

/-- Represents a pyramid in 3D space -/
structure Pyramid where
  base : Set (Fin 3 → ℝ)
  apex : Fin 3 → ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Fin 3 → ℝ
  point : Fin 3 → ℝ

/-- Calculate the altitude of a pyramid -/
noncomputable def altitude (P : Pyramid) : ℝ :=
  sorry

/-- Check if two planes are parallel -/
def is_parallel (p1 p2 : Plane) : Prop :=
  sorry

/-- Calculate the base plane of a pyramid -/
noncomputable def base_plane (P : Pyramid) : Plane :=
  sorry

/-- Calculate the distance between a plane and a point -/
noncomputable def distance (pl : Plane) (pt : Fin 3 → ℝ) : ℝ :=
  sorry

/-- Calculate the volume of a pyramid -/
noncomputable def volume (P : Pyramid) : ℝ :=
  sorry

/-- Construct the smaller pyramid formed by the intersection of the original pyramid and a plane -/
noncomputable def smaller_pyramid (P : Pyramid) (π : Plane) : Pyramid :=
  sorry

theorem pyramid_volume_ratio (P : Pyramid) (π : Plane) (h : ℝ) :
  altitude P = h →
  is_parallel π (base_plane P) →
  distance π (base_plane P).point = (2/3) * h →
  volume (smaller_pyramid P π) / volume P = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_ratio_l505_50541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_factors_of_2_pow_20_minus_1_l505_50554

theorem two_digit_factors_of_2_pow_20_minus_1 :
  (Finset.filter (fun n : ℕ => 10 ≤ n ∧ n < 100 ∧ (2^20 - 1) % n = 0) (Finset.range 100)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_factors_of_2_pow_20_minus_1_l505_50554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_measure_less_than_mean_l505_50532

theorem acute_angle_measure_less_than_mean (θ : Real) (h : 0 < θ ∧ θ < π/2) : 
  θ < (Real.sin θ + Real.tan θ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_measure_less_than_mean_l505_50532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_area_of_intersecting_boards_l505_50510

/-- Given two boards intersecting at a 45-degree angle, calculate the area of the unpainted region -/
theorem unpainted_area_of_intersecting_boards
  (width1 : ℝ) -- Width of the first board
  (width2 : ℝ) -- Width of the second board
  (angle : ℝ) -- Angle of intersection in radians
  (h1 : width1 = 5) -- First board is 5 inches wide
  (h2 : width2 = 7) -- Second board is 7 inches wide
  (h3 : angle = π/4) -- Intersection angle is 45 degrees (π/4 radians)
  : ℝ :=
by
  -- The area of the unpainted region on the five-inch board is 35√2 square inches
  sorry

-- Example usage (commented out to avoid evaluation errors)
-- #eval unpainted_area_of_intersecting_boards 5 7 (Real.pi/4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_area_of_intersecting_boards_l505_50510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_squared_l505_50518

open BigOperators Finset

theorem binomial_sum_squared (n : ℕ) : 
  ∑ k in range (n + 1), (2 * n).factorial / ((k.factorial)^2 * ((n - k).factorial)^2) = (Nat.choose (2 * n) n)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_squared_l505_50518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_higher_interest_rate_l505_50588

/-- Calculates simple interest given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem higher_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (rate : ℝ) 
  (base_rate : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 2000)
  (h2 : time = 2)
  (h3 : base_rate = 12)
  (h4 : interest_difference = 240)
  (h5 : simpleInterest principal rate time - simpleInterest principal base_rate time = interest_difference) :
  rate = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_higher_interest_rate_l505_50588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l505_50547

noncomputable def g (x : ℝ) : ℝ := (3^x + 2) / (3^x + 3)

theorem g_neither_even_nor_odd :
  ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) :=
by
  apply And.intro
  · intro h
    have h1 : g 1 ≠ g (-1) := by
      -- Proof that g(1) ≠ g(-1)
      sorry
    exact h1 (h 1)
  · intro h
    have h2 : g 1 ≠ -g (-1) := by
      -- Proof that g(1) ≠ -g(-1)
      sorry
    exact h2 (h 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l505_50547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l505_50592

/-- Given a point P on the terminal side of angle α, prove properties about α and β. -/
theorem angle_properties (α β : Real) (P : Real × Real) :
  α ∈ Set.Ioo 0 π →
  β ∈ Set.Ioo 0 π →
  P.1 = 3 →
  P.2 = 1 →
  Real.tan α = 1 / 3 →
  Real.tan (α - β) = (Real.sin (2 * (π / 2 - α)) + 4 * (Real.cos α)^2) / (10 * (Real.cos α)^2 + Real.cos (3 * π / 2 - 2 * α)) →
  (Real.tan (α - β) = 1 / 2 ∧ Real.tan β = -1 / 7 ∧ 2 * α - β = -3 * π / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l505_50592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_negative_two_of_three_l505_50531

theorem power_negative_two_of_three : (3 : ℝ) ^ ((-2) : ℝ) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_negative_two_of_three_l505_50531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_theorem_l505_50561

/-- The speed of the second train given the conditions of the problem -/
noncomputable def second_train_speed (first_train_length : ℝ) (second_train_length : ℝ)
  (first_train_speed : ℝ) (clearing_time_seconds : ℝ) : ℝ :=
  let total_distance := first_train_length + second_train_length
  let clearing_time_hours := clearing_time_seconds / 3600
  let relative_speed := total_distance / 1000 / clearing_time_hours
  relative_speed - first_train_speed

/-- Theorem stating that the speed of the second train is approximately 65.069 km/h -/
theorem second_train_speed_theorem :
  ∃ ε > 0, |second_train_speed 120 165 80 7.0752960452818945 - 65.069| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_theorem_l505_50561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_or_indivisibility_l505_50512

theorem divisibility_or_indivisibility (m n : ℕ) (a : ℕ → ℕ) 
  (h_mn : m > 0 ∧ n > 0)
  (h_length : ∀ i, i ≤ m * n → a i < a (i + 1))
  (h_positive : ∀ i, i ≤ m * n + 1 → a i > 0) :
  (∃ s : Finset ℕ, s.card = m + 1 ∧ 
    (∀ i j, i ∈ s → j ∈ s → i ≠ j → ¬(a i ∣ a j) ∧ ¬(a j ∣ a i))) ∨
  (∃ s : Finset ℕ, s.card = n + 1 ∧ 
    (∀ i j, i ∈ s → j ∈ s → i < j → a i ∣ a j)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_or_indivisibility_l505_50512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_has_15_elements_l505_50555

def oplus (m n : ℕ) : ℕ :=
  if (m % 2 = 0 ∧ n % 2 = 0) ∨ (m % 2 ≠ 0 ∧ n % 2 ≠ 0) then
    m + n
  else
    m * n

def M : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ oplus p.1 p.2 = 12}

theorem M_has_15_elements : Finset.card (Finset.filter (fun p => p.1 > 0 ∧ p.2 > 0 ∧ oplus p.1 p.2 = 12) (Finset.range 13 ×ˢ Finset.range 13)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_has_15_elements_l505_50555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_profit_percentage_is_30_percent_l505_50591

noncomputable def calculate_new_profit_percentage (original_selling_price : ℝ) 
                                    (original_profit_percentage : ℝ) 
                                    (purchase_price_reduction_percentage : ℝ)
                                    (additional_revenue : ℝ) : ℝ :=
  let original_purchase_price := original_selling_price / (1 + original_profit_percentage)
  let new_purchase_price := original_purchase_price * (1 - purchase_price_reduction_percentage)
  let new_selling_price := original_selling_price + additional_revenue
  let new_profit := new_selling_price - new_purchase_price
  (new_profit / new_purchase_price) * 100

theorem new_profit_percentage_is_30_percent :
  calculate_new_profit_percentage 1100 0.10 0.10 70 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_profit_percentage_is_30_percent_l505_50591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_360_l505_50523

theorem divisors_of_360 : 
  (Finset.filter (fun n => 360 % n = 0) (Finset.range 361)).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_360_l505_50523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l505_50533

noncomputable def ceiling (x : ℝ) := Int.ceil x
noncomputable def floor (x : ℝ) := Int.floor x

theorem solution_range (x : ℝ) : 
  (ceiling x)^2 + 4 * (floor x + 1) + 4 = 0 → -3 < x ∧ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l505_50533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_two_three_l505_50528

noncomputable def f (x : ℝ) : ℝ := 2^x - 5

theorem zero_of_f_in_interval_two_three :
  ∃ (x : ℝ), x ∈ Set.Icc 2 3 ∧ f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_in_interval_two_three_l505_50528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l505_50513

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a parabola, its focus, and points A, B, and M, prove the required properties -/
theorem parabola_properties (C : Parabola) (F A B M : Point)
  (hF : F.x = C.p / 2 ∧ F.y = 0)  -- F is the focus
  (hA : A.y^2 = 2 * C.p * A.x ∧ A.x > 0 ∧ A.y > 0)  -- A is on C and in first quadrant
  (hB : B.y^2 = 2 * C.p * B.x)  -- B is on C
  (hM : M.x = C.p ∧ M.y = 0)  -- M is (p, 0)
  (hAF_AM : (A.x - F.x)^2 + A.y^2 = (A.x - M.x)^2 + A.y^2)  -- |AF| = |AM|
  : (∃ (m : ℝ), m = 2 * Real.sqrt 6 ∧ m = (B.y - A.y) / (B.x - A.x))  -- Slope of AB is 2√6
  ∧ (A.x - B.x)^2 + (A.y - B.y)^2 > 16 * (F.x^2 + F.y^2)  -- |AB| > 4|OF|
  ∧ Real.arccos ((A.x * M.x + A.y * M.y) / (Real.sqrt (A.x^2 + A.y^2) * Real.sqrt (M.x^2 + M.y^2)))
    + Real.arccos ((B.x * M.x + B.y * M.y) / (Real.sqrt (B.x^2 + B.y^2) * Real.sqrt (M.x^2 + M.y^2)))
    < π  -- ∠OAM + ∠OBM < 180°
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l505_50513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marbles_l505_50522

/-- The number of red marbles -/
def r : ℝ := sorry

/-- The number of blue marbles -/
noncomputable def b : ℝ := r / 1.1

/-- The number of green marbles -/
def g : ℝ := 1.8 * r

/-- The total number of marbles -/
noncomputable def total : ℝ := r + b + g

theorem total_marbles : total = 3.709 * r := by
  unfold total b g
  ring
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marbles_l505_50522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l505_50597

/-- Given a train crossing a platform, calculate the time to cross a signal pole -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 350)
  (h3 : platform_crossing_time = 39)
  : ∃ (signal_pole_crossing_time : ℝ),
    signal_pole_crossing_time = train_length / ((train_length + platform_length) / platform_crossing_time) ∧
    abs (signal_pole_crossing_time - 18) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l505_50597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_midpoint_triangle_l505_50529

/-- Area of a triangle given its vertices -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Check if two line segments are parallel and in opposite directions -/
def parallel_opposite_direction (seg1 seg2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

/-- Check if a point is the midpoint of a line segment -/
def is_midpoint (M A B : ℝ × ℝ) : Prop := sorry

/-- Given two triangles ABC and A'B'C' with the following properties:
    1. Area of ABC is 1
    2. Area of A'B'C' is 2025
    3. AB is parallel to A'B' and in opposite direction
    4. BC is parallel to B'C' and in opposite direction
    5. CA is parallel to C'A' and in opposite direction
    6. A'' is the midpoint of AA'
    7. B'' is the midpoint of BB'
    8. C'' is the midpoint of CC'
    Then the area of triangle A''B''C'' is 484 -/
theorem area_midpoint_triangle (A B C A' B' C' A'' B'' C'' : ℝ × ℝ) 
  (h1 : area_triangle A B C = 1)
  (h2 : area_triangle A' B' C' = 2025)
  (h3 : parallel_opposite_direction (A, B) (A', B'))
  (h4 : parallel_opposite_direction (B, C) (B', C'))
  (h5 : parallel_opposite_direction (C, A) (C', A'))
  (h6 : is_midpoint A'' A A')
  (h7 : is_midpoint B'' B B')
  (h8 : is_midpoint C'' C C') :
  area_triangle A'' B'' C'' = 484 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_midpoint_triangle_l505_50529
