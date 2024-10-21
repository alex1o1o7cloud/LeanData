import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_proof_l570_57079

def median (x y z : ℕ) : ℕ := 
  if x ≤ y ∧ y ≤ z then y
  else if x ≤ z ∧ z ≤ y then z
  else if y ≤ x ∧ x ≤ z then x
  else if y ≤ z ∧ z ≤ x then z
  else if z ≤ x ∧ x ≤ y then x
  else y

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 30 →
  median a b c = 28 →
  max a (max b c) = 28 + 6 →
  min a (min b c) = 28 :=
by
  sorry

#check smallest_number_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_proof_l570_57079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_points_l570_57011

open Real

-- Define the equation of the curve
def curve_equation (x y : ℝ) : Prop :=
  y^2 + x^4 = 2 * x^2 * y + 4

-- Define the two points on the curve
def point_a (a : ℝ) : Prop :=
  curve_equation Real.pi a

def point_b (b : ℝ) : Prop :=
  curve_equation Real.pi b

-- Theorem statement
theorem absolute_difference_of_points :
  ∀ a b : ℝ, 
  point_a a → point_b b → a ≠ b → 
  |a - b| = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_points_l570_57011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_seller_loss_percentage_l570_57028

/-- Represents the selling price that incurs a loss -/
noncomputable def selling_price_loss : ℝ := 9

/-- Represents the selling price that would give a 5% profit -/
noncomputable def selling_price_profit : ℝ := 11.8125

/-- Represents the profit percentage that would be achieved at the profit selling price -/
noncomputable def profit_percentage : ℝ := 5

/-- Calculates the cost price given the selling price for profit and the profit percentage -/
noncomputable def cost_price : ℝ := selling_price_profit / (1 + profit_percentage / 100)

/-- Calculates the loss amount -/
noncomputable def loss : ℝ := cost_price - selling_price_loss

/-- Calculates the percentage of loss -/
noncomputable def loss_percentage : ℝ := (loss / cost_price) * 100

theorem fruit_seller_loss_percentage :
  loss_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_seller_loss_percentage_l570_57028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_satisfies_equation_l570_57056

noncomputable def smallest_angle : ℝ := 45 / 7

theorem smallest_angle_satisfies_equation :
  let y := smallest_angle
  Real.tan (6 * y * Real.pi / 180) = (Real.cos (y * Real.pi / 180) - Real.sin (y * Real.pi / 180)) / (Real.cos (y * Real.pi / 180) + Real.sin (y * Real.pi / 180)) ∧
  ∀ z, 0 < z ∧ z < y →
    Real.tan (6 * z * Real.pi / 180) ≠ (Real.cos (z * Real.pi / 180) - Real.sin (z * Real.pi / 180)) / (Real.cos (z * Real.pi / 180) + Real.sin (z * Real.pi / 180)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_satisfies_equation_l570_57056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_square_cx_squared_l570_57035

/-- Unit square ABCD with point X outside -/
structure UnitSquareWithPoint where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  X : ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Distance from a point to a line segment -/
noncomputable def distanceToLineSegment (p a b : ℝ × ℝ) : ℝ :=
  sorry

theorem unit_square_cx_squared (sq : UnitSquareWithPoint) :
  sq.A = (0, 0) →
  sq.B = (1, 0) →
  sq.C = (1, 1) →
  sq.D = (0, 1) →
  distanceToLineSegment sq.X sq.A sq.C = distanceToLineSegment sq.X sq.B sq.D →
  distance sq.A sq.X = Real.sqrt 2 / 2 →
  (distance sq.C sq.X)^2 = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_square_cx_squared_l570_57035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_count_problem_l570_57061

/-- The number of oranges in a group of apples and oranges --/
def numOranges (applePrice orangePrice : ℚ) (totalPrice : ℚ) (numApples : ℕ) : ℕ :=
  Nat.floor ((totalPrice - (numApples : ℚ) * applePrice) / orangePrice)

theorem orange_count_problem :
  let applePrice : ℚ := 21 / 100
  let firstGroupTotal : ℚ := 177 / 100
  let secondGroupTotal : ℚ := 127 / 100
  ∃ (orangePrice : ℚ),
    (2 : ℕ) * applePrice + 5 * orangePrice = secondGroupTotal ∧
    numOranges applePrice orangePrice firstGroupTotal 6 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_count_problem_l570_57061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_real_parts_l570_57024

theorem product_of_real_parts : 
  ∃ (x₁ x₂ : ℂ), x₁^2 + 2*x₁ = -1 + 2*Complex.I ∧ 
                 x₂^2 + 2*x₂ = -1 + 2*Complex.I ∧ 
                 Complex.re x₁ * Complex.re x₂ = 1 - Real.sqrt 5 * (Real.cos (Real.arctan 2 / 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_real_parts_l570_57024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_squares_l570_57091

/-- The area covered by two overlapping congruent squares -/
noncomputable def area_covered_by_overlapping_squares (side_length : ℝ) : ℝ :=
  2 * side_length^2 - (side_length / 2)^2

/-- Theorem: The area covered by two congruent squares with side length 12,
    where one vertex of the second square coincides with a vertex of the first square,
    is equal to 252 -/
theorem area_of_overlapping_squares :
  area_covered_by_overlapping_squares 12 = 252 := by
  -- Unfold the definition of area_covered_by_overlapping_squares
  unfold area_covered_by_overlapping_squares
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

-- We can't use #eval with noncomputable definitions, so we'll use #check instead
#check area_covered_by_overlapping_squares 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_squares_l570_57091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_correct_l570_57070

noncomputable def original_equation (x : ℝ) : Prop :=
  (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 14

noncomputable def smallest_solution : ℝ := (11 - Real.sqrt 445) / 6

theorem smallest_solution_correct :
  original_equation smallest_solution ∧
  ∀ y, original_equation y → y ≥ smallest_solution :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_correct_l570_57070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l570_57023

def S : Finset ℕ := {3^27 * 5^36, 3^28 * 5^35, 3^29 * 5^34, 3^30 * 5^33, 3^31 * 5^32, 
                     3^32 * 5^31, 3^33 * 5^30, 3^34 * 5^29, 3^35 * 5^28, 3^36 * 5^27}

theorem problem_solution :
  (S.card = 10) ∧ 
  (∀ n, n ∈ S → 15 ∣ n) ∧
  (∀ a b, a ∈ S → b ∈ S → a ≠ b → ¬(a ∣ b)) ∧
  (∀ a b, a ∈ S → b ∈ S → b^3 ∣ a^4) := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l570_57023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l570_57098

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

noncomputable def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 = 1 →
  sum_of_arithmetic_sequence a 11 = 33 →
  (∀ n : ℕ, b n = (1/4) ^ (a n)) →
  (∀ n : ℕ, a n = n / 2) ∧
  geometric_sequence b ∧
  (∀ n : ℕ, sum_of_arithmetic_sequence b n = 1 - (1/2)^n) := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l570_57098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l570_57019

theorem sqrt_equation_solution :
  ∃ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t)^(1/4 : ℝ)) ∧ t = 37/10 := by
  use 37/10
  apply And.intro
  · -- Proof of the equation
    sorry
  · -- Proof that t = 37/10
    rfl

#eval (37/10 : Float)  -- This will output 3.7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l570_57019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_of_triangle_l570_57030

/-- The radius of the circumcircle of a triangle with sides 5, 12, and 13 is 6.5 -/
theorem circumcircle_radius_of_triangle (a b c : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) :
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c))) = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_of_triangle_l570_57030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_x_value_l570_57083

/-- Given that 2x, x+1, and x+2 form an arithmetic sequence, prove that x = 0 -/
theorem arithmetic_sequence_x_value :
  ∀ x : ℝ, (∀ d : ℝ, 2*x + d = x+1 ∧ x+1 + d = x+2) → x = 0 :=
by
  intro x h
  specialize h ((x+2) - (x+1))
  have h1 : 2*x + ((x+2) - (x+1)) = x+1 := h.left
  have h2 : x+1 + ((x+2) - (x+1)) = x+2 := h.right
  linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_x_value_l570_57083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_12_l570_57044

open Real

-- Define the function f
noncomputable def f (α : ℝ) : ℝ := 
  (cos (-α) * sin (π + α)) / cos (3*π + α) + 
  (sin (-2*π - α) * sin (α + π/2)) / cos (3*π/2 - α)

-- State the theorem
theorem f_value_at_pi_12 : f (π/12) = (Real.sqrt 2 + Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_12_l570_57044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_sum_of_roots_l570_57026

theorem cubic_root_equation_sum_of_roots :
  let f (x : ℝ) := (x + 2) ^ (1/3 : ℝ) + (3 * x - 1) ^ (1/3 : ℝ) - (16 * x + 4) ^ (1/3 : ℝ)
  ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = 1.25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_sum_of_roots_l570_57026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l570_57068

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.cos (x - Real.pi/3) - 2

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
    T = Real.pi ∧
    (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f x ≤ 1) ∧
    (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f x = 1) ∧
    (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f x ≥ -2) ∧
    (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f x = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l570_57068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line₁_correct_line₂_correct_l570_57071

-- Define the slope for the first line
noncomputable def slope₁ : ℝ := Real.sqrt 3

-- Define the point for the first line
def point₁ : ℝ × ℝ := (2, 1)

-- Define the inclination angle for the first line
noncomputable def angle₁ : ℝ := Real.pi / 3

-- Define the equation for the first line
def line₁ (x y : ℝ) : Prop := slope₁ * x - y - 2 * slope₁ + 1 = 0

-- Define the point for the second line
def point₂ : ℝ × ℝ := (-3, 2)

-- Define the two possible equations for the second line
def line₂₁ (x y : ℝ) : Prop := 2 * x + 3 * y = 0
def line₂₂ (x y : ℝ) : Prop := x + y + 1 = 0

-- Theorem for the first line
theorem line₁_correct :
  (line₁ point₁.1 point₁.2) ∧
  (Real.tan angle₁ = slope₁) := by sorry

-- Theorem for the second line
theorem line₂_correct :
  (line₂₁ point₂.1 point₂.2 ∨ line₂₂ point₂.1 point₂.2) ∧
  (∃ a : ℝ, a ≠ 0 ∧ ((line₂₁ a 0 ∧ line₂₁ 0 a) ∨ (line₂₂ a 0 ∧ line₂₂ 0 a))) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line₁_correct_line₂_correct_l570_57071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_angle_bisector_l570_57029

/-- Triangle ABC with angle bisector BL -/
structure AngleBisectorTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  L : ℝ × ℝ
  is_angle_bisector : Bool
  AL : ℝ
  BL : ℝ
  CL : ℝ

/-- The area of a triangle with an angle bisector -/
noncomputable def triangle_area (t : AngleBisectorTriangle) : ℝ := sorry

/-- Theorem: Area of triangle ABC with given conditions -/
theorem triangle_area_with_angle_bisector (t : AngleBisectorTriangle) 
  (h1 : t.is_angle_bisector = true)
  (h2 : t.AL = 3)
  (h3 : t.BL = 6 * Real.sqrt 5)
  (h4 : t.CL = 4) :
  triangle_area t = (21 * Real.sqrt 55) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_angle_bisector_l570_57029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_constant_bound_exists_l570_57069

/-- Definition of the function f on a set of integers -/
def f (A : Finset ℤ) : Finset ℤ :=
  (A.product A).image (fun (x, y) => x^2 + x*y + y^2)

/-- Theorem stating that no constant c exists satisfying the given condition -/
theorem no_constant_bound_exists :
  ¬∃ c : ℝ, ∀ n : ℕ, ∃ A : Finset ℤ, (A.card = n) ∧ ((f A).card ≤ ⌈c * n⌉) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_constant_bound_exists_l570_57069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_l570_57067

/-- A parabola with equation y^2 = 4x -/
structure Parabola where
  C : Set (ℝ × ℝ)
  eq : ∀ (x y : ℝ), (x, y) ∈ C ↔ y^2 = 4*x

/-- The focus of a parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: For a parabola y^2 = 4x, if |AF| = |BF| where F is the focus,
    A is on the parabola, and B is (3,0), then |AB| = 2√2 -/
theorem parabola_distance (p : Parabola) (A : ℝ × ℝ) 
    (h1 : A ∈ p.C) 
    (h2 : distance A focus = distance (3, 0) focus) :
  distance A (3, 0) = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_l570_57067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storage_volume_calculation_l570_57086

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the storage details for a company -/
structure StorageDetails where
  boxDimensions : BoxDimensions
  costPerBoxPerMonth : ℝ
  totalMonthlyPayment : ℝ

/-- Calculates the number of boxes stored based on the storage details -/
noncomputable def numberOfBoxes (s : StorageDetails) : ℝ :=
  s.totalMonthlyPayment / s.costPerBoxPerMonth

/-- Calculates the total volume occupied by all boxes -/
noncomputable def totalVolume (s : StorageDetails) : ℝ :=
  (numberOfBoxes s) * (boxVolume s.boxDimensions)

theorem storage_volume_calculation (s : StorageDetails) 
  (h1 : s.boxDimensions.length = 15)
  (h2 : s.boxDimensions.width = 12)
  (h3 : s.boxDimensions.height = 10)
  (h4 : s.costPerBoxPerMonth = 0.5)
  (h5 : s.totalMonthlyPayment = 300) :
  totalVolume s = 1080000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_storage_volume_calculation_l570_57086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_profit_l570_57063

-- Define the passenger capacity function
noncomputable def p (t : ℝ) : ℝ :=
  if 10 ≤ t ∧ t ≤ 20 then 1300
  else if 2 ≤ t ∧ t < 10 then 1300 - 10 * (10 - t)^2
  else 0

-- Define the net profit per minute function
noncomputable def Q (t : ℝ) : ℝ := (6 * p t - 3960) / t - 350

-- Theorem statement
theorem max_net_profit :
  ∃ (t_max : ℝ), 2 ≤ t_max ∧ t_max ≤ 20 ∧
  Q t_max = 130 ∧
  ∀ (t : ℝ), 2 ≤ t ∧ t ≤ 20 → Q t ≤ Q t_max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_profit_l570_57063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_2x_minus_1_F_sum_zero_h_zeroes_h_range_l570_57047

-- Define the function F
noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℤ :=
  if x < f x then 1
  else if x = f x then 0
  else -1

-- Theorem 1: Analytical expression for F(2x-1)
theorem F_2x_minus_1 (x : ℝ) : F (λ y => 2*y - 1) x = 
  if x > 1 then 1
  else if x = 1 then 0
  else -1 := by sorry

-- Theorem 2: Solution for F(|x-a|) + F(2x-1) = 0
theorem F_sum_zero : ∃ a : ℝ, ∀ x : ℝ, 
  F (λ y => |y - a|) x + F (λ y => 2*y - 1) x = 0 ∧ (a = 0 ∨ a = 2) := by sorry

-- Theorem 3: Number of zeroes of h(x) in [π/3, 4π/3]
theorem h_zeroes : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁ ∈ Set.Icc (π/3) (4*π/3) ∧ x₂ ∈ Set.Icc (π/3) (4*π/3) ∧
  Real.cos x₁ * F (λ y => y + Real.sin y) x₁ = 0 ∧
  Real.cos x₂ * F (λ y => y + Real.sin y) x₂ = 0 := by sorry

-- Theorem 4: Range of h(x) in [π/3, 4π/3]
theorem h_range : ∀ x ∈ Set.Icc (π/3) (4*π/3),
  ∃ y ∈ Set.Ioo (-1) 1, y = Real.cos x * F (λ z => z + Real.sin z) x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_2x_minus_1_F_sum_zero_h_zeroes_h_range_l570_57047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l570_57052

/-- Triangle with sides 9, 12, and 15 has area 54 -/
theorem triangle_area (a b c : ℝ) : a = 9 ∧ b = 12 ∧ c = 15 → (a * b) / 2 = 54 := by
  intro h
  rw [h.1, h.2.1]
  norm_num

#check triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l570_57052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_theorem_l570_57093

def grasshopper_jumps (r : ℕ) : ℕ := Nat.choose (2 * r) r

noncomputable def grasshopper_catalan (r : ℕ) : ℚ := (grasshopper_jumps r : ℚ) / (r + 1 : ℚ)

theorem grasshopper_theorem (r : ℕ) :
  (∃ (total_ways positive_ways : ℕ),
    total_ways = grasshopper_jumps r ∧
    positive_ways = Nat.floor (grasshopper_catalan r) ∧
    total_ways ≥ positive_ways) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_theorem_l570_57093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l570_57042

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 18 * a^2 + 32 * a - 12 = 0 →
  b^3 - 18 * b^2 + 32 * b - 12 = 0 →
  c^3 - 18 * c^2 + 32 * c - 12 = 0 →
  (a / (a + b * c)) + (b / (b + c * a)) + (c / (c + a * b)) = 65 / 74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l570_57042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l570_57034

/-- The number of days it takes for two workers to complete a job together,
    given their individual work rates. -/
noncomputable def days_to_complete (rate_A rate_B : ℝ) : ℝ :=
  1 / (rate_A + rate_B)

/-- Theorem stating that if worker A's work rate is half of worker B's,
    and worker B can finish a job in 22.5 days, then workers A and B
    together can finish the job in 15 days. -/
theorem workers_completion_time
  (rate_B : ℝ)
  (h1 : rate_B > 0)
  (h2 : rate_B = 1 / 22.5) :
  days_to_complete (rate_B / 2) rate_B = 15 := by
  sorry

#check workers_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_completion_time_l570_57034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_through_given_points_l570_57004

/-- The slope of a line passing through two points (x₁, y₁) and (x₂, y₂) is (y₂ - y₁) / (x₂ - x₁) -/
def line_slope (x₁ y₁ x₂ y₂ : ℚ) : ℚ := (y₂ - y₁) / (x₂ - x₁)

/-- The slope of the line passing through (-4, 7) and (3, -4) is -11/7 -/
theorem slope_through_given_points :
  line_slope (-4) 7 3 (-4) = -11/7 := by
  -- Unfold the definition of line_slope
  unfold line_slope
  -- Simplify the arithmetic
  simp [sub_eq_add_neg, add_comm, add_left_comm]
  -- The result should now be obvious to Lean
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_through_given_points_l570_57004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l570_57082

theorem trig_identity (k : ℤ) :
  let z : ℝ := π * k / 3
  (Real.sin z)^3 * Real.sin (3 * z) + (Real.cos z)^3 * Real.cos (3 * z) = (Real.cos (4 * z))^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l570_57082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_equal_l570_57064

-- Define the curve y = 1/x
noncomputable def inverse_curve (x : ℝ) : ℝ := 1 / x

-- Define a point on the curve
structure PointOnCurve where
  x : ℝ
  h_pos : x > 0

-- Define the area bounded by OA, OB, and arc AB
noncomputable def area_OAB (A B : PointOnCurve) : ℝ := sorry

-- Define the area bounded by AH_A, BH_B, x-axis, and arc AB
noncomputable def area_AHBH (A B : PointOnCurve) : ℝ := sorry

-- Theorem statement
theorem areas_equal (A B : PointOnCurve) : area_OAB A B = area_AHBH A B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_equal_l570_57064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_all_black_probability_l570_57092

/-- Represents the color of a square -/
inductive Color
| White
| Black
| Grey

/-- Represents a 4x4 grid of colored squares -/
def Grid := Fin 4 → Fin 4 → Color

/-- The probability of a single square being initially colored with a specific color -/
noncomputable def initialColorProb : ℝ := 1 / 3

/-- The probability of a square ending up black after the rotation operation -/
noncomputable def blackAfterRotationProb : ℝ := 11 / 27

/-- The number of squares affected by rotation (excluding the center) -/
def rotatedSquaresCount : ℕ := 12

/-- The number of center squares that remain unchanged -/
def centerSquaresCount : ℕ := 4

/-- Calculates the probability of the entire grid being black after the operation -/
noncomputable def probabilityAllBlack : ℝ :=
  (initialColorProb ^ centerSquaresCount) * (blackAfterRotationProb ^ rotatedSquaresCount)

theorem grid_all_black_probability :
  probabilityAllBlack = 1 / 81 * (11 / 27) ^ 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_all_black_probability_l570_57092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_total_profit_l570_57081

/-- Profit function for product A -/
noncomputable def profit_A (m : ℝ) : ℝ := (1/2) * m + 60

/-- Profit function for product B -/
noncomputable def profit_B (m : ℝ) : ℝ := 70 + 6 * Real.sqrt m

/-- Total capital to be invested -/
def total_capital : ℝ := 200

/-- Minimum investment for each product -/
def min_investment : ℝ := 25

/-- Total profit function -/
noncomputable def total_profit (x : ℝ) : ℝ := profit_A (total_capital - x) + profit_B x

/-- Theorem stating the maximum total profit and optimal allocation -/
theorem max_total_profit :
  ∃ (x : ℝ), x ≥ min_investment ∧ 
             x ≤ total_capital - min_investment ∧
             total_profit x = 248 ∧
             ∀ (y : ℝ), y ≥ min_investment → 
                         y ≤ total_capital - min_investment → 
                         total_profit y ≤ total_profit x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_total_profit_l570_57081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scout_troop_profit_l570_57095

/-- Represents the profit calculation for a scout troop's candy bar sale --/
theorem scout_troop_profit :
  ∀ (total_bars : ℕ) (cost_per_8_bars : ℚ) (price_per_3_bars : ℚ) (discount_price_per_20_bars : ℚ),
  total_bars = 1500 →
  cost_per_8_bars = 3 →
  price_per_3_bars = 2 →
  discount_price_per_20_bars = 12 →
  (cost_per_8_bars / 8 * total_bars : ℚ) = 562.5 ∧
  (discount_price_per_20_bars / 20 * total_bars : ℚ) = 900 ∧
  900 - 562.5 = 337.5 := by
  intro total_bars cost_per_8_bars price_per_3_bars discount_price_per_20_bars
  intro h_total h_cost h_price h_discount
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scout_troop_profit_l570_57095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l570_57058

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Given an acute triangle ABC inscribed in circle ω -/
def inscribed_triangle (ABC : Triangle) (ω : Circle) : Prop :=
  sorry

/-- A circle through A, O, C intersects BC at P -/
def circle_intersects (A O C P : Point) (BC : Set Point) : Prop :=
  sorry

/-- Tangents to ω at A and C intersect at T -/
def tangents_intersect (ω : Circle) (A C T : Point) : Prop :=
  sorry

/-- TP intersects AC at K -/
def line_intersects (T P K : Point) (AC : Set Point) : Prop :=
  sorry

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  sorry

/-- Length of a line segment -/
noncomputable def length (A B : Point) : ℝ :=
  sorry

/-- Angle between three points -/
noncomputable def angle (A B C : Point) : ℝ :=
  sorry

/-- Main theorem -/
theorem triangle_properties
  (ABC : Triangle) (ω : Circle) (O P T K : Point) :
  inscribed_triangle ABC ω →
  circle_intersects ABC.A O ABC.C P {ABC.B, ABC.C} →
  tangents_intersect ω ABC.A ABC.C T →
  line_intersects T P K {ABC.A, ABC.C} →
  area (Triangle.mk ABC.A P K) = 15 →
  area (Triangle.mk ABC.C P K) = 13 →
  angle ABC.A ABC.B ABC.C = Real.arctan (4/7) →
  area ABC = 784/13 ∧ length ABC.A ABC.C = 14 / Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l570_57058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_bounds_l570_57005

/-- Given a consumer product with the following properties:
  - Initial cost: 120 yuan per item
  - Initial annual sales: 800,000 items
  - Tax rate: r yuan per 100 yuan of sales
  - Sales volume decrease: (20/3)r * 10,000 items
  - Minimum annual tax revenue: 2.56 million yuan

  Prove that the tax rate r satisfies 4 ≤ r ≤ 8 to ensure the annual tax
  revenue is not less than 2.56 million yuan. -/
theorem tax_rate_bounds (r : ℝ) : 
  (let initial_cost : ℝ := 120
  let initial_sales : ℝ := 800000
  let sales_decrease : ℝ → ℝ := λ r => (20/3) * r * 10000
  let current_sales : ℝ → ℝ := λ r => initial_sales - sales_decrease r
  let tax_revenue : ℝ → ℝ := λ r => (current_sales r * initial_cost * r) / 100
  let min_revenue : ℝ := 2.56 * 1000000
  tax_revenue r ≥ min_revenue) → 4 ≤ r ∧ r ≤ 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_bounds_l570_57005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_inequality_l570_57040

theorem max_a_inequality (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - a * Real.sqrt (x^2 + 1) + 3 ≥ 0) → 
  a ≤ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_inequality_l570_57040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_childrens_ticket_price_l570_57031

theorem childrens_ticket_price
  (total_tickets : ℕ)
  (adult_price : ℚ)
  (total_receipts : ℚ)
  (adult_tickets : ℕ)
  (h1 : total_tickets = 522)
  (h2 : adult_price = 15)
  (h3 : total_receipts = 5086)
  (h4 : adult_tickets = 130) :
  (total_receipts - (adult_tickets : ℚ) * adult_price) / ((total_tickets - adult_tickets) : ℚ) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_childrens_ticket_price_l570_57031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l570_57032

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_problem (a b : ℝ × ℝ) (h1 : a = (1, -2)) (h2 : a + b = (0, 2)) :
  magnitude b = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l570_57032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_α_l570_57003

noncomputable def angle_α : ℝ := Real.arctan (-4 / -3)

theorem cos_pi_half_plus_α (p₀ : ℝ × ℝ) (h : p₀ = (-3, -4)) :
  Real.cos (π / 2 + angle_α) = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_α_l570_57003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_evaluation_l570_57094

theorem propositions_evaluation :
  (Real.log 5 ≥ Real.sqrt 5 * Real.log 2) ∧
  ((2 : Real)^(Real.sqrt 11) < 11) ∧
  (3 * Real.exp 1 * Real.log 2 < 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_evaluation_l570_57094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiple_of_given_numbers_l570_57027

theorem smallest_multiple_of_given_numbers : ∃! n : ℕ,
  (∀ m ∈ ({2, 4, 6, 8, 10, 12, 14, 16, 18, 20} : Finset ℕ), m ∣ n) ∧
  (∀ k : ℕ, 0 < k → k < n → ∃ m ∈ ({2, 4, 6, 8, 10, 12, 14, 16, 18, 20} : Finset ℕ), ¬(m ∣ k)) :=
by
  use 5040
  sorry

#eval Nat.lcm 2 (Nat.lcm 4 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 10 (Nat.lcm 12 (Nat.lcm 14 (Nat.lcm 16 (Nat.lcm 18 20))))))))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiple_of_given_numbers_l570_57027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_and_sum_theorem_l570_57036

/-- The region S in the coordinate plane -/
def S : Set (ℝ × ℝ) :=
  {p | |p.1^2 - 4*p.1 - 5| + p.2 ≤ 15 ∧ p.2 - 2*p.1 ≥ 10}

/-- The line around which S is revolved -/
def revolveLine (x y : ℝ) : Prop := y - 2*x = 10

/-- The volume of the solid formed by revolving S around the line -/
noncomputable def volumeOfSolid : ℝ := 343*Real.pi/(24*Real.sqrt 2)

/-- Theorem stating the volume and the sum of coefficients -/
theorem volume_and_sum_theorem :
  (∃ (a b c : ℕ), volumeOfSolid = (a:ℝ)*Real.pi/((b:ℝ)*Real.sqrt (c:ℝ)) ∧
                   Nat.Coprime a b ∧
                   (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ c)) ∧
                   a + b + c = 369) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_and_sum_theorem_l570_57036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l570_57099

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x - Real.pi / 3) + Real.sqrt 3 / 2

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (∀ x, f x ≤ 1) →
  (0 < A) → (A < Real.pi / 2) →
  (0 < B) → (B < Real.pi / 2) →
  (0 < C) → (C < Real.pi / 2) →
  (A + B + C = Real.pi) →
  (a = 2 * Real.sqrt 2) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →
  (f A = Real.sqrt 3 / 2) →
  (A = Real.pi / 3 ∧ b + c = 2 * Real.sqrt 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l570_57099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l570_57050

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 5*x + 2) / (x^4 + 2*x^3 - x^2 - 2*x)

def num_holes (f : ℝ → ℝ) : ℕ := sorry
def num_vertical_asymptotes (f : ℝ → ℝ) : ℕ := sorry
def num_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := sorry
def num_oblique_asymptotes (f : ℝ → ℝ) : ℕ := sorry

theorem asymptote_sum : 
  num_holes f + 2 * num_vertical_asymptotes f + 3 * num_horizontal_asymptotes f + 4 * num_oblique_asymptotes f = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l570_57050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_travel_distance_l570_57000

/-- Proves that the distance between points A and B is approximately 217.14 km given the specified conditions. -/
theorem boat_travel_distance (total_time : ℝ) (stream_velocity : ℝ) (boat_speed : ℝ) (wind_speed : ℝ) :
  total_time = 38 →
  stream_velocity = 4 →
  boat_speed = 14 →
  wind_speed = 2 →
  let downstream_speed := boat_speed + stream_velocity + wind_speed
  let upstream_speed := boat_speed - stream_velocity - wind_speed
  let distance := (total_time * downstream_speed * upstream_speed) / (2 * downstream_speed + upstream_speed)
  abs (distance - 217.14) < 0.01 := by
  sorry

#check boat_travel_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_travel_distance_l570_57000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_fencing_cost_per_meter_l570_57012

/-- Represents a rectangular plot with fencing. -/
structure Plot where
  length : ℚ
  breadth : ℚ
  total_cost : ℚ

/-- Calculates the cost per meter of fencing for a given plot. -/
def cost_per_meter (p : Plot) : ℚ :=
  p.total_cost / (2 * (p.length + p.breadth))

/-- Theorem stating the cost per meter for the given plot specifications. -/
theorem plot_fencing_cost_per_meter :
  let p : Plot := {
    length := 57,
    breadth := 57 - 14,
    total_cost := 5300
  }
  cost_per_meter p = 265 / 10 := by
  sorry

#eval cost_per_meter { length := 57, breadth := 43, total_cost := 5300 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_fencing_cost_per_meter_l570_57012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l570_57008

/-- Given a circle C and a line l, if the distance from the center of C to l is √2, then a = 1 -/
theorem circle_line_distance (a : ℝ) (h_a : a > 0) :
  let C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 1}
  let l : Set (ℝ × ℝ) := {p | p.1 - p.2 + a = 0}
  let center : ℝ × ℝ := (1, 0)
  (center ∈ C) →
  (∀ p ∈ l, Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) ≥ Real.sqrt 2) →
  (∃ p ∈ l, Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) = Real.sqrt 2) →
  a = 1 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l570_57008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l570_57007

noncomputable def h (x : ℝ) : ℝ := (x^3 - 3*x^2 + 6*x - 8) / (x^2 - 5*x + 6)

theorem h_domain :
  {x : ℝ | ∃ y, h x = y} = Set.union (Set.union (Set.Iio 2) (Set.Ioo 2 3)) (Set.Ioi 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l570_57007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_arithmetic_sequence_l570_57014

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n+1 => arithmetic_sequence a₁ d n + d

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem minimize_sum_arithmetic_sequence :
  ∃ (d : ℝ),
    let a := arithmetic_sequence (-6) d
    let S := sum_arithmetic_sequence (-6) d
    S 6 = a 7 - 3 →
    (∀ n : ℕ, S n ≥ S 2 ∨ S n ≥ S 3) ∧
    (S 2 = S 3 ∨ (∀ n : ℕ, n ≠ 2 ∧ n ≠ 3 → S n > S 2 ∧ S n > S 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_arithmetic_sequence_l570_57014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_of_symmetry_l570_57053

-- Define the two functions as noncomputable
noncomputable def f (x : ℝ) : ℝ := 4^(-x)
noncomputable def g (x : ℝ) : ℝ := 2^(2*x - 3)

-- State the theorem
theorem line_of_symmetry :
  ∃ (c : ℝ), c = 9/4 ∧
  ∀ (x : ℝ), f (c - x) = g (c + x) := by
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_of_symmetry_l570_57053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equivalence_l570_57048

theorem scientific_notation_equivalence : ∀ (x : ℝ), 
  x = 0.000688 → x = 6.88 * (10 : ℝ)^(-4 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equivalence_l570_57048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_terms_l570_57080

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry

axiom recurrence_relation (n : ℕ) :
  n ≥ 1 → (sequence_a (n + 1), sequence_b (n + 1)) = (2 * sequence_a n - Real.sqrt 3 * sequence_b n, 2 * sequence_b n + Real.sqrt 3 * sequence_a n)

axiom fiftieth_term :
  (sequence_a 50, sequence_b 50) = (1, Real.sqrt 3)

theorem sum_of_first_terms :
  sequence_a 1 + sequence_b 1 = Real.sqrt 3 / 7^24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_terms_l570_57080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_one_l570_57010

-- Define the function f
noncomputable def f (x : ℝ) (θ : ℝ) : ℝ := 3 * Real.cos (Real.pi * x + θ)

-- State the theorem
theorem f_value_at_one (θ : ℝ) 
  (h : ∀ x, f x θ = f (2 - x) θ) : 
  f 1 θ = 3 ∨ f 1 θ = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_one_l570_57010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_is_twenty_percent_l570_57077

/-- Represents the hourly wages of different positions at Joe's Steakhouse -/
structure Wages where
  manager : ℚ
  chef : ℚ
  dishwasher : ℚ

/-- The conditions given in the problem -/
def wage_conditions (w : Wages) : Prop :=
  w.chef > w.dishwasher ∧
  w.dishwasher = w.manager / 2 ∧
  w.manager = 17/2 ∧
  w.chef = w.manager - 17/5

/-- The percentage increase in hourly wage of a chef compared to a dishwasher -/
def percentage_increase (w : Wages) : ℚ :=
  (w.chef - w.dishwasher) / w.dishwasher * 100

/-- Theorem stating that the percentage increase is 20% -/
theorem wage_increase_is_twenty_percent (w : Wages) :
  wage_conditions w → percentage_increase w = 20 := by
  sorry

#eval let w : Wages := ⟨17/2, 51/10, 17/4⟩; percentage_increase w

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_is_twenty_percent_l570_57077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_eq_interval_l570_57001

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a

def f_iter (a : ℝ) : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => f a (f_iter a n x)

def M : Set ℝ := {a | ∀ n : ℕ, n > 0 → |f_iter a n 0| ≤ 2}

theorem M_eq_interval : M = Set.Icc (-2 : ℝ) (1/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_eq_interval_l570_57001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_circle_area_integral_l570_57045

open Set
open MeasureTheory

theorem half_circle_area_integral (a : ℝ) (ha : 0 < a) :
  ∫ x in -a..a, Real.sqrt (a^2 - x^2) = (1/2) * Real.pi * a^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_circle_area_integral_l570_57045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_properties_l570_57046

-- Define the relationships and analysis method
structure FunctionRelationship : Type
structure CorrelationRelationship : Type
structure RegressionAnalysis : Type

-- Define properties
def isDeterministic (r : Type) : Prop := sorry
def isNonDeterministic (r : Type) : Prop := sorry
def usedFor (a : Type) (r : Type) : Prop := sorry

-- State the theorem
theorem relationship_properties :
  (isDeterministic FunctionRelationship) ∧
  (isNonDeterministic CorrelationRelationship) ∧
  (usedFor RegressionAnalysis CorrelationRelationship) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_properties_l570_57046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l570_57025

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

-- Define the interval [-2, 2]
def I : Set ℝ := Set.Icc (-2) 2

-- Define the maximum value of f on the interval I
noncomputable def M (a : ℝ) : ℝ := ⨆ (x ∈ I), f a x

-- Define the minimum value of f on the interval I
noncomputable def m (a : ℝ) : ℝ := ⨅ (x ∈ I), f a x

-- Define the function g
noncomputable def g (a : ℝ) : ℝ := M a - m a

-- State the theorem
theorem g_min_value :
  ∃ (a_min : ℝ), ∀ (a : ℝ), g a ≥ g a_min ∧ g a_min = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l570_57025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_difference_of_T_l570_57075

-- Define the triangle PQR
def P : ℝ × ℝ := (0, 10)
def Q : ℝ × ℝ := (3, 0)
def R : ℝ × ℝ := (10, 0)

-- Define the vertical line intersection points
noncomputable def T : ℝ × ℝ := (10 - Real.sqrt 30, Real.sqrt 30)
noncomputable def U : ℝ × ℝ := (10 - Real.sqrt 30, 0)

-- Define the area of triangle TUR
def area_TUR : ℝ := 15

-- State the theorem
theorem coordinate_difference_of_T :
  abs (T.1 - T.2) = 10 - 2 * Real.sqrt 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_difference_of_T_l570_57075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oates_reunion_attendance_l570_57097

theorem oates_reunion_attendance
  (total_guests : ℕ)
  (yellow_reunion : ℕ)
  (both_reunions : ℕ)
  (h1 : total_guests = 100)
  (h2 : yellow_reunion = 65)
  (h3 : both_reunions = 7)
  (h4 : ∀ g : ℕ, g ≤ total_guests → g ≤ yellow_reunion ∨ g ≤ (total_guests - yellow_reunion + both_reunions)) :
  total_guests - yellow_reunion + both_reunions = 42 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oates_reunion_attendance_l570_57097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circle_cover_square_l570_57073

theorem min_circle_cover_square (a : ℝ) : 
  (∃ (c1 c2 c3 : Set (ℝ × ℝ)), 
    (∀ x y, (x, y) ∈ c1 ∨ (x, y) ∈ c2 ∨ (x, y) ∈ c3 → 
      0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1) ∧
    (∀ i ∈ ({1, 2, 3} : Set Nat), ∃ x y : ℝ, 
      Set.prod (Set.Icc 0 1) (Set.Icc 0 1) ⊆ 
        {p : ℝ × ℝ | (p.1 - x)^2 + (p.2 - y)^2 ≤ (a/2)^2})) →
  a ≥ Real.sqrt 65 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circle_cover_square_l570_57073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_tank_l570_57009

/-- Represents a right circular cone water tank -/
structure WaterTank where
  baseRadius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem water_height_in_tank (tank : WaterTank) (water_volume_ratio : ℝ) :
  tank.baseRadius = 12 →
  tank.height = 72 →
  water_volume_ratio = 0.2 →
  ∃ (waterHeight : ℝ), waterHeight = 36 * (2/5)^(1/3) ∧
    coneVolume tank.baseRadius tank.height * water_volume_ratio =
    coneVolume (tank.baseRadius * (waterHeight / tank.height)) waterHeight :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_tank_l570_57009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partners_profit_share_l570_57055

/-- Given three partners A, B, and C with capitals satisfying certain ratios,
    prove that B's share of a total profit of 16500 is 6000. -/
theorem partners_profit_share (a b c : ℝ) : 
  (2 * a = 3 * b) →  -- Twice A's capital equals thrice B's capital
  (b = 4 * c) →      -- B's capital is 4 times C's capital
  (b / (a + b + c)) * 16500 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partners_profit_share_l570_57055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_difference_not_22_l570_57057

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem divisor_difference_not_22 (p₁ p₂ p₃ p₄ : ℕ) (n : ℕ) 
  (h_prime₁ : is_prime p₁) (h_prime₂ : is_prime p₂) (h_prime₃ : is_prime p₃) (h_prime₄ : is_prime p₄)
  (h_distinct : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄)
  (h_n : n = p₁ * p₂ * p₃ * p₄)
  (h_bound : n < 1995)
  (divisors : List ℕ)
  (h_divisors : List.Pairwise (·<·) divisors ∧ divisors.length = 16 ∧ divisors.head? = some 1 ∧ divisors.getLast? = some n ∧ 
    ∀ d ∈ divisors, n % d = 0) :
  (divisors.get? 8).bind (λ d₈ => (divisors.get? 9).map (λ d₉ => d₉ - d₈)) ≠ some 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_difference_not_22_l570_57057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_6_l570_57037

/-- A geometric sequence with first term a and common ratio r -/
noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ := fun n => a * r^(n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_6 (a r : ℝ) :
  let seq := geometric_sequence a r
  let sum := geometric_sum a r
  seq 1 + seq 3 = 5/2 →
  seq 2 + seq 4 = 5/4 →
  sum 6 = 63/16 := by sorry

#check geometric_sequence_sum_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_6_l570_57037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_squares_exist_l570_57002

/-- Represents a square divided into rectangles -/
structure DividedSquare where
  total_rectangles : Nat
  vertical_lines : Nat
  horizontal_lines : Nat
  num_squares : Nat

/-- Helper function to represent the size of a square within the divided square -/
def square_size (ds : DividedSquare) (square_index : Nat) : Nat :=
  sorry

/-- Theorem stating that if a square is divided into 100 rectangles by 9 vertical and 9 horizontal lines,
    and there are exactly 9 squares among these rectangles, then at least two of these squares must be identical in size -/
theorem identical_squares_exist (ds : DividedSquare) 
    (h1 : ds.total_rectangles = 100)
    (h2 : ds.vertical_lines = 9)
    (h3 : ds.horizontal_lines = 9)
    (h4 : ds.num_squares = 9) :
    ∃ (s1 s2 : Nat), s1 ≠ s2 ∧ square_size ds s1 = square_size ds s2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_squares_exist_l570_57002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l570_57060

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

-- Define the conditions
def EllipseConditions (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧
  (0, Real.sqrt 3) ∈ Ellipse a b ∧
  c ^ 2 = a ^ 2 - b ^ 2 ∧
  c = b * Real.tan (Real.pi / 6)

-- Theorem statement
theorem ellipse_theorem (a b c : ℝ) (h : EllipseConditions a b c) :
  (Ellipse 2 (Real.sqrt 3) = Ellipse a b) ∧
  (∀ (A B E : ℝ × ℝ) (k : ℝ),
    A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧ E ∈ Ellipse a b ∧
    A.1 = B.1 ∧ E.2 = k * (E.1 - 4) →
    ∃ (t : ℝ), t * (E.2 + A.2) = (1 - t) * (E.1 - A.1) ∧ t ∈ Set.Ioo 0 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l570_57060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_horse_pony_difference_l570_57076

/-- Represents the number of ponies on the ranch -/
def P : ℕ := sorry

/-- Represents the number of horses on the ranch -/
def H : ℕ := sorry

/-- The fraction of ponies with horseshoes -/
def ponies_with_horseshoes : ℚ := 3/10

/-- The fraction of ponies with horseshoes that are from Iceland -/
def icelandic_ponies_with_horseshoes : ℚ := 5/8

theorem minimum_horse_pony_difference :
  (P + H = 163) →
  (H > P) →
  (ponies_with_horseshoes * icelandic_ponies_with_horseshoes * ↑P).num % (ponies_with_horseshoes * icelandic_ponies_with_horseshoes * ↑P).den = 0 →
  (∀ P' H', P' + H' = 163 → H' > P' → P' < P → False) →
  H - P = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_horse_pony_difference_l570_57076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l570_57066

theorem cos_alpha_plus_pi_fourth (α β : Real) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : 0 < β) (h4 : β < π/2) 
  (h5 : Real.cos (α + β) = 3/5) 
  (h6 : Real.sin (β - π/4) = 5/13) : 
  Real.cos (α + π/4) = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l570_57066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l570_57022

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + Real.sin (2 * x)

theorem min_translation_for_symmetry :
  ∃ (m : ℝ), m > 0 ∧
  (∀ (x : ℝ), f (x + m) = -f (-x - m)) ∧
  (∀ (m' : ℝ), m' > 0 ∧ (∀ (x : ℝ), f (x + m') = -f (-x - m')) → m ≤ m') ∧
  m = 3 * Real.pi / 8 := by
  sorry

#check min_translation_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l570_57022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l570_57021

noncomputable def f (x : ℝ) := Real.sqrt (x - 5) + (x + 4) ^ (1/3 : ℝ)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l570_57021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisor_of_all_l570_57065

def has_dividing_or_summing_property (S : Finset ℕ) : Prop :=
  ∀ a b c d, a ∈ S → b ∈ S → c ∈ S → d ∈ S →
    (a ∣ b ∧ a ∣ c ∧ a ∣ d) ∨ (b ∣ a ∧ b ∣ c ∧ b ∣ d) ∨ 
    (c ∣ a ∧ c ∣ b ∧ c ∣ d) ∨ (d ∣ a ∧ d ∣ b ∧ d ∣ c) ∨
    a = b + c + d ∨ b = a + c + d ∨ c = a + b + d ∨ d = a + b + c

theorem exists_divisor_of_all (S : Finset ℕ) 
  (h_size : S.card = 100) 
  (h_prop : has_dividing_or_summing_property S) :
  ∃ x ∈ S, ∀ y ∈ S, x ≠ y → x ∣ y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisor_of_all_l570_57065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l570_57013

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  d : ℝ
  seq_def : ∀ n : ℕ+, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) : ℕ+ → ℝ :=
  λ n => (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : S seq 5 > S seq 6 ∧ S seq 6 > S seq 4) :
  seq.d < 0 ∧ S seq 10 > 0 ∧ S seq 11 < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l570_57013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l570_57018

-- Define the quadratic function
noncomputable def f (a b x : ℝ) := x^2 - a*x - b

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a b x < 0 ↔ 1 < x ∧ x < 3

-- Define the inequality function
noncomputable def g (a b x : ℝ) := (2*x + a) / (x + b)

theorem quadratic_inequality_solution :
  ∃ a b : ℝ, solution_set a b ∧
    (a = 4 ∧ b = -3) ∧
    (∀ x, g a b x > 1 ↔ x > 3 ∨ x < -7) := by
  sorry

#check quadratic_inequality_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l570_57018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l570_57017

-- Define the ellipse
noncomputable def Γ (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, -1)

-- Define lines l₁ and l₂
def l₁ (x : ℝ) : Prop := x = -2
def l₂ (y : ℝ) : Prop := y = -1

-- Define point P
def P (x₀ y₀ : ℝ) : Prop := Γ x₀ y₀ ∧ x₀ > 0 ∧ y₀ > 0

-- Define tangent line l₃
def l₃ (x₀ y₀ x y : ℝ) : Prop := x₀*x/4 + y₀*y = 1

-- Define intersection points
def C : ℝ × ℝ := (-2, -1)
noncomputable def D (x₀ y₀ : ℝ) : ℝ × ℝ := (4*(1 + y₀)/x₀, -1)
noncomputable def E (x₀ y₀ : ℝ) : ℝ × ℝ := (-2, (2 + x₀)/(2*y₀))

-- Helper function to define a line given two points
def line (p q : ℝ × ℝ) (x : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, x = (1 - t) • p + t • q

-- Theorem statement
theorem lines_concurrent (x₀ y₀ : ℝ) (h₁ : P x₀ y₀) :
  ∃ (Q : ℝ × ℝ), line A (D x₀ y₀) Q ∧ line B (E x₀ y₀) Q ∧ line C (x₀, y₀) Q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l570_57017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l570_57074

/-- Given two squares ABCD and DEFG with integer side lengths, where point E is on line segment CD
    with CE < DE, and CF = 5 cm, prove that the area of pentagon ABCFG is 71 square centimeters. -/
theorem pentagon_area (a b c d e f g : EuclideanSpace ℝ (Fin 2)) : 
  let ab := ‖a - b‖
  let bc := ‖b - c‖
  let cd := ‖c - d‖
  let de := ‖d - e‖
  let ef := ‖e - f‖
  let fg := ‖f - g‖
  let cf := ‖c - f‖
  let ce := ‖c - e‖
  -- ABCD is a square with integer side length
  ab = bc ∧ bc = cd ∧ cd = ‖d - a‖ ∧ ∃ n : ℕ, ab = n ∧
  -- DEFG is a square with integer side length
  de = ef ∧ ef = fg ∧ fg = ‖g - d‖ ∧ ∃ m : ℕ, de = m ∧
  -- E is on CD and CE < DE
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ e = c + t • (d - c) ∧ ce < de ∧
  -- CF = 5
  cf = 5 →
  -- Area of pentagon ABCFG is 71
  ab * bc + de * ef - (1/2 * ce * ef) = 71 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l570_57074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_open_zero_one_l570_57090

-- Define the sets A and B
def A : Set ℝ := {x | (2 : ℝ)^x > 1}
def B : Set ℝ := {x | -4 < x ∧ x < 1}

-- State the theorem
theorem A_intersect_B_equals_open_zero_one : A ∩ B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_open_zero_one_l570_57090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_regular_rate_l570_57039

/-- Bob's payment structure and work hours -/
structure BobsWork where
  regularRate : ℚ  -- Bob's regular hourly rate
  overtimeRate : ℚ  -- Bob's overtime hourly rate
  regularHoursPerWeek : ℚ  -- Regular hours per week
  firstWeekHours : ℚ  -- Total hours worked in the first week
  secondWeekHours : ℚ  -- Total hours worked in the second week
  totalPay : ℚ  -- Total pay for two weeks

/-- Calculate overtime hours for a given week -/
def overtimeHours (totalHours : ℚ) (regularHours : ℚ) : ℚ :=
  max (totalHours - regularHours) 0

/-- Calculate total pay for two weeks -/
def calculateTotalPay (b : BobsWork) : ℚ :=
  let regularPay := 2 * b.regularHoursPerWeek * b.regularRate
  let overtimePay := b.overtimeRate * (overtimeHours b.firstWeekHours b.regularHoursPerWeek +
                                       overtimeHours b.secondWeekHours b.regularHoursPerWeek)
  regularPay + overtimePay

/-- Theorem: Bob's regular hourly rate is $5 -/
theorem bobs_regular_rate (b : BobsWork)
  (h_overtime : b.overtimeRate = 6)
  (h_regular_hours : b.regularHoursPerWeek = 40)
  (h_first_week : b.firstWeekHours = 44)
  (h_second_week : b.secondWeekHours = 48)
  (h_total_pay : b.totalPay = 472)
  (h_calc_pay : b.totalPay = calculateTotalPay b) :
  b.regularRate = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_regular_rate_l570_57039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_partition_l570_57043

def is_obtuse_triple (a b c : ℕ) : Prop :=
  a + b > c ∧ a * a + b * b < c * c

def set_range (n : ℕ) : Set ℕ :=
  {x | 2 ≤ x ∧ x ≤ 3 * n + 1}

theorem obtuse_triangle_partition (n : ℕ) :
  ∃ (partition : List (ℕ × ℕ × ℕ)),
    (∀ (triple : ℕ × ℕ × ℕ), triple ∈ partition → 
      let (a, b, c) := triple
      a ∈ set_range n ∧ b ∈ set_range n ∧ c ∈ set_range n ∧
      is_obtuse_triple a b c) ∧
    (∀ x ∈ set_range n, ∃ (triple : ℕ × ℕ × ℕ), triple ∈ partition ∧ 
      (x = triple.1 ∨ x = triple.2.1 ∨ x = triple.2.2)) ∧
    partition.length = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_partition_l570_57043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_at_quarter_perimeter_l570_57072

/-- Represents the perimeter of the circular path in blocks -/
def perimeter : ℕ := 24

/-- Represents Hector's speed in blocks per unit time -/
def hector_speed : ℚ := 1

/-- Represents Jane's speed in blocks per unit time -/
def jane_speed : ℚ := 2 * hector_speed

/-- Represents the time it takes for Jane and Hector to meet -/
noncomputable def meeting_time : ℚ := perimeter / (hector_speed + jane_speed)

/-- Represents the distance Hector walks before meeting Jane -/
noncomputable def hector_distance : ℚ := hector_speed * meeting_time

/-- Represents the distance Jane walks before meeting Hector -/
noncomputable def jane_distance : ℚ := jane_speed * meeting_time

theorem meet_at_quarter_perimeter :
  hector_distance = jane_distance ∧ hector_distance = perimeter / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_at_quarter_perimeter_l570_57072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l570_57087

noncomputable def bounded_area : ℝ := (8 * Real.pi / 3) + 8 * Real.sqrt 3

noncomputable def curve1 (φ : ℝ) : ℝ := 4 * Real.cos (3 * φ)

def curve2 : ℝ := 2

theorem area_of_bounded_figure :
  ∃ (area : ℝ), area = bounded_area ∧
  (∀ φ : ℝ, curve1 φ ≥ curve2 → 
    area = 6 * (1/2) * ∫ φ in (-π/9)..(π/9), (curve1 φ)^2 - curve2^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l570_57087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_m_l570_57038

theorem existence_of_m (n : ℕ) (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h1 : (p * q) ∣ (n^p + 2))
  (h2 : (n + 2) ∣ (n^p + q^p)) :
  ∃ m : ℕ, q ∣ (4^m * n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_m_l570_57038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_problem_l570_57096

/-- A function representing the inverse square relationship between c and d -/
noncomputable def inverse_square (k : ℝ) (d : ℝ) : ℝ := k / (d * d)

/-- Theorem stating that given the inverse square relationship and initial condition,
    the value of c when d = 6 is 64/9 -/
theorem inverse_square_problem (k : ℝ) :
  (inverse_square k 4 = 16) →
  (inverse_square k 6 = 64/9) :=
by
  intro h
  -- Proof steps would go here
  sorry

#check inverse_square_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_problem_l570_57096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l570_57089

def sequence_a : ℕ → ℚ
  | 0 => 1/3  -- Adding the base case for 0
  | 1 => 1/3
  | (n+2) => (n+2) / ((2 * sequence_a (n+1) + (n+1)) / sequence_a (n+1))

theorem sequence_a_formula (n : ℕ) : sequence_a n = n / (2*n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l570_57089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_2016_subsequence_l570_57033

def sequenceDigit (a b c d : Nat) : Nat :=
  (a + b + c + d) % 10

def sequenceN : Nat → Nat
  | 0 => 2
  | 1 => 0
  | 2 => 1
  | 3 => 7
  | n + 4 => sequenceDigit (sequenceN (n+3)) (sequenceN (n+2)) (sequenceN (n+1)) (sequenceN n)

theorem no_2016_subsequence :
  ∀ n : Nat, n ≥ 4 →
    ¬(sequenceN n = 2 ∧ sequenceN (n+1) = 0 ∧ sequenceN (n+2) = 1 ∧ sequenceN (n+3) = 6) := by
  sorry

#eval [sequenceN 0, sequenceN 1, sequenceN 2, sequenceN 3, sequenceN 4, sequenceN 5, sequenceN 6, sequenceN 7, sequenceN 8, sequenceN 9]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_2016_subsequence_l570_57033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_polynomial_degree_for_selection_l570_57015

theorem min_polynomial_degree_for_selection (a : Fin 13 → ℝ) 
  (h_distinct : ∀ i j : Fin 13, i ≠ j → a i ≠ a j) :
  (∃ (n : ℕ) (P : ℝ → ℝ) (S : Finset (Fin 13)),
    S.card = 6 ∧ 
    (∀ x : ℝ, ∃ (p : Polynomial ℝ), Polynomial.degree p ≤ n ∧ P x = p.eval x) ∧
    (∀ i ∈ S, ∀ j ∉ S, P (a i) > P (a j))) →
  (∀ n : ℕ, n < 12 → 
    ¬∃ (P : ℝ → ℝ) (S : Finset (Fin 13)),
      S.card = 6 ∧ 
      (∀ x : ℝ, ∃ (p : Polynomial ℝ), Polynomial.degree p ≤ n ∧ P x = p.eval x) ∧
      (∀ i ∈ S, ∀ j ∉ S, P (a i) > P (a j))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_polynomial_degree_for_selection_l570_57015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_radian_conversion_l570_57054

/-- Conversion factor between degrees and radians -/
noncomputable def deg_to_rad : ℝ := Real.pi / 180

/-- Conversion from degrees to radians -/
noncomputable def to_radians (degrees : ℝ) : ℝ := degrees * deg_to_rad

/-- Conversion from radians to degrees -/
noncomputable def to_degrees (radians : ℝ) : ℝ := radians / deg_to_rad

theorem degree_radian_conversion :
  (to_radians 210 = 7 * Real.pi / 6) ∧ 
  (to_degrees (-5 * Real.pi / 2) = -450) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_radian_conversion_l570_57054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_5432_l570_57088

noncomputable def x : ℝ := 4 + Real.sqrt 15
noncomputable def y : ℝ := 4 - Real.sqrt 15

noncomputable def S (n : ℕ) : ℝ := (1 / 2) * (x^n + y^n)

theorem units_digit_S_5432 : ∃ k : ℕ, S 5432 = 10 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_5432_l570_57088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l570_57041

theorem division_remainder_problem (a b : ℕ) 
  (h1 : a - b = 1355)
  (h2 : a = 1608)
  (h3 : ∃ q r, a = b * q + r ∧ q = 6 ∧ r < b) :
  a % b = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l570_57041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_occurrence_l570_57084

theorem subset_occurrence (n : ℕ) (M : Finset ℕ) (P : Finset (Finset ℕ)) :
  n ≥ 3 →
  M = Finset.range n →
  (∀ p ∈ P, p.card = 2 ∧ p ⊆ M) →
  (∀ i j, i ∈ M → j ∈ M → i ≠ j → (∃ p ∈ P, i ∈ p ∧ j ∈ p) → (∃ k ∈ M, {i, j} ∈ P)) →
  (∀ i ∈ M, (P.filter (λ p => i ∈ p)).card = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_occurrence_l570_57084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_bicycle_flow_l570_57059

-- Define the flow function Q(v)
noncomputable def Q (v : ℝ) : ℝ := 600 * v / (v^2 + 2*v + 400)

-- Theorem statement
theorem electric_bicycle_flow :
  -- Part 1: Range of v where Q(v) ≥ 10
  (∀ v : ℝ, 0 < v ∧ v ≤ 25 → (Q v ≥ 10 ↔ 8 ≤ v ∧ v ≤ 25)) ∧
  -- Part 2: Maximum value of Q(v) occurs at v = 20
  (∀ v : ℝ, 0 < v ∧ v ≤ 25 → Q 20 ≥ Q v) ∧
  -- Part 3: Maximum value of Q(v) is 100/7
  Q 20 = 100 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_bicycle_flow_l570_57059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_five_halves_equals_half_l570_57020

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  let y := x - 2 * ⌊x / 2⌋  -- Adjust x to the equivalent point in [-1, 1)
  if -1 ≤ y ∧ y < 0 then
    -4 * y^2 + 2
  else if 0 ≤ y ∧ y < 1 then
    y
  else
    0  -- This case should never occur due to the periodicity

-- State the theorem
theorem f_five_halves_equals_half :
  f (5/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_five_halves_equals_half_l570_57020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l570_57051

-- Define the function f(x) = x / (x - 2)
noncomputable def f (x : ℝ) : ℝ := x / (x - 2)

-- State the theorem about the domain of f
theorem domain_of_f : 
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x ≠ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l570_57051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_trip_distance_l570_57085

/-- Represents the bus trip scenario -/
structure BusTrip where
  v : ℝ  -- Original speed of the bus in miles per hour
  T : ℝ  -- Total distance of the trip in miles

/-- Calculates the travel time for the first scenario -/
noncomputable def travel_time_scenario1 (trip : BusTrip) : ℝ :=
  2 + 0.75 + (3 * (trip.T - 2 * trip.v)) / (2 * trip.v)

/-- Calculates the travel time for the second scenario -/
noncomputable def travel_time_scenario2 (trip : BusTrip) : ℝ :=
  (2 * trip.v + 120) / trip.v + 0.75 + (3 * (trip.T - 2 * trip.v - 120)) / (2 * trip.v)

/-- Theorem stating the total distance of the trip -/
theorem bus_trip_distance :
  ∃ (trip : BusTrip),
    travel_time_scenario1 trip - 4 = travel_time_scenario2 trip - 3 ∧
    trip.T = 720 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_trip_distance_l570_57085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_cardinality_l570_57062

def A : Finset ℕ := {1, 2, 3, 4}
def B : Finset ℕ := {3, 4, 5}
def U : Finset ℕ := A ∪ B

theorem complement_intersection_cardinality : Finset.card (U \ (A ∩ B)) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_cardinality_l570_57062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocal_squares_l570_57078

/-- The line l in the xy-plane -/
noncomputable def line_l (x y : ℝ) : Prop :=
  Real.sqrt 3 * x - y - Real.sqrt 3 = 0

/-- The curve C in the xy-plane -/
def curve_C (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Point P -/
def point_P : ℝ × ℝ := (1, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_sum_reciprocal_squares :
  ∀ A B : ℝ × ℝ,
  line_l A.1 A.2 → line_l B.1 B.2 →
  curve_C A.1 A.2 → curve_C B.1 B.2 →
  A ≠ B →
  (1 / (distance A point_P)^2) + (1 / (distance B point_P)^2) = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocal_squares_l570_57078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_density_and_inner_circle_l570_57006

/-- Probability density function for a point (x, y) in a circle of radius R -/
noncomputable def f (R : ℝ) (C : ℝ) (x y : ℝ) : ℝ :=
  if x^2 + y^2 ≤ R^2 then C * (R - Real.sqrt (x^2 + y^2)) else 0

/-- The constant C for the probability density function -/
noncomputable def C (R : ℝ) : ℝ := 3 / (Real.pi * R^3)

/-- The probability that a random point (X, Y) falls within a circle of radius r centered at the origin -/
noncomputable def prob (R r : ℝ) : ℝ :=
  ∫ x in -r..r, ∫ y in -Real.sqrt (r^2 - x^2)..Real.sqrt (r^2 - x^2), f R (C R) x y

theorem probability_density_and_inner_circle (R : ℝ) (h : R > 0) :
  (∫ x in -R..R, ∫ y in -Real.sqrt (R^2 - x^2)..Real.sqrt (R^2 - x^2), f R (C R) x y) = 1 ∧
  prob 2 1 = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_density_and_inner_circle_l570_57006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_implies_b_equals_two_l570_57049

noncomputable section

-- Define the line and parabola
def line (b : ℝ) (x : ℝ) : ℝ := x + b
def parabola (x : ℝ) : ℝ := (1/2) * x^2

-- Define the intersection points
structure IntersectionPoint (b : ℝ) where
  x : ℝ
  y : ℝ
  on_line : y = line b x
  on_parabola : y = parabola x

-- Define the perpendicularity condition
def perpendicular (b : ℝ) (A B : IntersectionPoint b) : Prop :=
  A.x * B.x + A.y * B.y = 0

theorem intersection_perpendicular_implies_b_equals_two :
  ∀ b : ℝ,
  ∀ A B : IntersectionPoint b,
  A ≠ B →
  perpendicular b A B →
  b = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_implies_b_equals_two_l570_57049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_price_reduction_l570_57016

/-- Calculates the percentage reduction in price given the original price and the additional quantity that can be purchased for the same total cost after the price reduction. -/
noncomputable def percentage_price_reduction (original_price : ℝ) (additional_quantity : ℝ) : ℝ :=
  let original_quantity := 400 / original_price
  let new_quantity := original_quantity + additional_quantity
  let new_price := 400 / new_quantity
  (original_price - new_price) / original_price * 100

/-- Theorem stating that given the original price of salt is Rs. 10 per kg and a price reduction allows purchasing 10 kgs more for Rs. 400, the percentage reduction in the price of salt is 20%. -/
theorem salt_price_reduction :
  percentage_price_reduction 10 10 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_price_reduction_l570_57016
