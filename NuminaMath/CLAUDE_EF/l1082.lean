import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_points_ratio_l1082_108280

/-- A point in a plane represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: For any set of n ≥ 3 points in a plane, there exist three points A, B, C 
    such that 1 ≤ AB/AC ≤ (n+1)/(n-1), where AB and AC are the distances between the points -/
theorem exist_three_points_ratio (n : ℕ) (points : Finset Point) 
    (h : n ≥ 3) (h_card : points.card = n) : 
    ∃ (A B C : Point), A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ 
    1 ≤ (distance A B) / (distance A C) ∧ 
    (distance A B) / (distance A C) ≤ (n + 1) / (n - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_points_ratio_l1082_108280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinate_l1082_108238

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three vertices -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

theorem third_vertex_coordinate (x : ℝ) :
  let a : Point := ⟨8, 6⟩
  let b : Point := ⟨0, 0⟩
  let c : Point := ⟨x, 0⟩
  x < 0 →
  triangleArea a b c = 48 →
  x = -16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinate_l1082_108238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_present_age_ratio_l1082_108214

-- Define the father's age
def father_age : ℕ → ℕ := sorry

-- Define the son's age
def son_age : ℕ → ℕ := sorry

-- Axiom: The sum of their present ages is 220
axiom sum_of_ages : father_age 0 + son_age 0 = 220

-- Axiom: 10 years later, the ratio of their ages will be 5:3
axiom future_ratio : (father_age 10) * 3 = (son_age 10) * 5

-- Theorem: The ratio of their present ages is 7:4
theorem present_age_ratio :
  (father_age 0) * 4 = (son_age 0) * 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_present_age_ratio_l1082_108214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_gallons_purchased_l1082_108229

def discount_rate : ℚ := 1 / 10
def isabella_gallons : ℚ := 25
def discount_ratio : ℚ := 108571428571428610 / 100000000000000000
def non_discounted_gallons : ℚ := 6

theorem kim_gallons_purchased :
  ∃ (k : ℚ),
    (isabella_gallons - non_discounted_gallons) * discount_rate =
    discount_ratio * (k - non_discounted_gallons) * discount_rate ∧
    abs (k - 235 / 10) < 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_gallons_purchased_l1082_108229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_linear_function_through_points_l1082_108249

-- Problem 1
theorem cube_root_plus_abs_plus_power : (8 : ℝ) ^ (1/3) + |(-5)| + (-1)^2023 = 5 := by sorry

-- Problem 2
theorem linear_function_through_points :
  ∀ (k b : ℝ),
  (∀ x y : ℝ, y = k * x + b → ((x = 0 ∧ y = 1) ∨ (x = 2 ∧ y = 5))) →
  (∀ x : ℝ, k * x + b = 2 * x + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_linear_function_through_points_l1082_108249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1082_108288

/-- A vector in R^2 -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The line y = (3/4)x - 2 -/
def onLine (v : Vector2D) : Prop :=
  v.y = (3/4) * v.x - 2

/-- Dot product of two vectors -/
def dot (v w : Vector2D) : ℝ :=
  v.x * w.x + v.y * w.y

/-- Squared norm of a vector -/
def normSquared (v : Vector2D) : ℝ :=
  v.x^2 + v.y^2

/-- Projection of v onto w -/
noncomputable def proj (v w : Vector2D) : Vector2D :=
  let scalar := (dot v w) / (normSquared w)
  { x := scalar * w.x, y := scalar * w.y }

/-- The theorem to be proved -/
theorem projection_theorem (w : Vector2D) :
  (∃ p : Vector2D, ∀ v : Vector2D, onLine v → proj v w = p) →
  ∃ p : Vector2D, p.x = 24/25 ∧ p.y = -32/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1082_108288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clive_olive_change_l1082_108292

/-- Given Clive's olive purchasing scenario, calculate his change. -/
theorem clive_olive_change (budget : ℚ) (olives_needed : ℕ) (jar_capacity : ℕ) (jar_cost : ℚ) :
  budget = 10 →
  olives_needed = 80 →
  jar_capacity = 20 →
  jar_cost = 3/2 →
  budget - (↑(olives_needed / jar_capacity + 
    (if olives_needed % jar_capacity = 0 then 0 else 1)) * jar_cost) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clive_olive_change_l1082_108292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cherry_cost_per_kg_l1082_108266

/-- The cost per kilogram of cherries -/
noncomputable def cost_per_kg (initial_money short_amount total_kg : ℝ) : ℝ :=
  (initial_money + short_amount) / total_kg

/-- Theorem stating the cost per kilogram of cherries -/
theorem cherry_cost_per_kg : cost_per_kg 1600 400 250 = 8 := by
  -- Unfold the definition of cost_per_kg
  unfold cost_per_kg
  -- Simplify the arithmetic
  simp [add_div]
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cherry_cost_per_kg_l1082_108266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_when_a_neg_one_increasing_condition_l1082_108252

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Define the domain
def domain : Set ℝ := Set.Icc (-5) 5

-- Theorem for part 1
theorem monotonicity_intervals_when_a_neg_one :
  let a := -1
  (∀ x ∈ Set.Icc (-5) (1/2), ∀ y ∈ Set.Icc (-5) (1/2), x < y → f a x > f a y) ∧
  (∀ x ∈ Set.Ioc (1/2) 5, ∀ y ∈ Set.Ioc (1/2) 5, x < y → f a x < f a y) := by
  sorry

-- Theorem for part 2
theorem increasing_condition :
  ∀ a : ℝ, (∀ x y, x ∈ domain → y ∈ domain → x < y → f a x < f a y) ↔ a ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_when_a_neg_one_increasing_condition_l1082_108252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l1082_108273

theorem undefined_values_count : ∃! (S : Finset ℝ), 
  (∀ x ∈ S, (x^2 - 5*x + 6) * (x + 1) = 0) ∧ 
  (∀ x ∉ S, (x^2 - 5*x + 6) * (x + 1) ≠ 0) ∧ 
  Finset.card S = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l1082_108273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1082_108254

def U : Set ℕ := {1,2,3,4,5,6,7,8}

def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | 1 ≤ x ∧ x ≤ 5}

def P : Set ℕ := {x ∈ U | 2 < x ∧ x < 9}

theorem set_operations :
  (A ∪ (B ∩ P) = {1,2,3,4,5}) ∧
  ((U \ B) ∩ (U \ P) = {1,2,6,7,8}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1082_108254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_red_stamps_l1082_108250

noncomputable section

def red_stamps : ℕ → Prop := λ _ => True
def blue_stamps : ℕ := 80
def yellow_stamps : ℕ := 7
def red_price : ℚ := 11/10
def blue_price : ℚ := 4/5
def yellow_price : ℚ := 2
def total_sale : ℚ := 100

axiom blue_stamps_count : blue_stamps = 80
axiom yellow_stamps_count : yellow_stamps = 7
axiom red_stamp_price : red_price = 11/10
axiom blue_stamp_price : blue_price = 4/5
axiom yellow_stamp_price : yellow_price = 2
axiom total_sale_amount : total_sale = 100

theorem max_red_stamps :
  ∃ n : ℕ, red_stamps n ∧ 
  (n : ℚ) * red_price + (blue_stamps : ℚ) * blue_price + (yellow_stamps : ℚ) * yellow_price = total_sale ∧
  n = 20 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_red_stamps_l1082_108250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l1082_108291

/-- The y-intercept of a line is the point where it crosses the y-axis (x = 0). -/
noncomputable def y_intercept (a b c : ℝ) : ℝ × ℝ := (0, c / b)

/-- A point (x, y) lies on a line ax + by = c if it satisfies the equation. -/
def on_line (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 = c

theorem y_intercept_of_line :
  y_intercept 4 7 28 = (0, 4) ∧ on_line 4 7 28 (y_intercept 4 7 28) := by
  sorry

#check y_intercept_of_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l1082_108291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_angle_l1082_108208

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- sum of angles in a triangle is 180°
  (α = β ∨ α = γ ∨ β = γ) →  -- isosceles triangle condition
  (α = 50 ∨ β = 50 ∨ γ = 50) →  -- one angle is 50°
  ((α = 50 ∧ β = γ) ∨ (β = 50 ∧ α = γ) ∨ (γ = 50 ∧ α = β) ∨ -- vertex angle is 50°
  (α = 80 ∧ β = 50 ∧ γ = 50) ∨ (β = 80 ∧ α = 50 ∧ γ = 50) ∨ (γ = 80 ∧ α = 50 ∧ β = 50)) -- vertex angle is 80°
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_angle_l1082_108208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1082_108244

noncomputable section

-- Define the triangles
def triangle1 : (ℝ × ℝ × ℝ) := (15, 15 * Real.sqrt 3, 30)
def triangle2 : (ℝ × ℝ × ℝ) := (39, 52, 65)

-- Function to check if a triangle is right-angled
def is_right_triangle (t : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := t
  a^2 + b^2 = c^2

-- Function to calculate the angle using Law of Cosines
noncomputable def angle_between (a b c : ℝ) : ℝ :=
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

-- Main theorem
theorem triangle_properties :
  (is_right_triangle triangle1) ∧
  (is_right_triangle triangle2) ∧
  (angle_between 15 (15 * Real.sqrt 3) 30 = π / 2) ∧
  (65 > 0 ∧ ∀ (p q : ℝ), p^2 + q^2 = 65^2 / 4 → p^2 + q^2 ≤ 65^2 / 4) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1082_108244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_parallel_lines_l1082_108257

-- Define the necessary structures
structure Line : Type :=
  (id : ℕ)

structure Angle : Type :=
  (id : ℕ)

-- Define the necessary functions
def Parallel (m n : Line) : Prop := sorry
def MeasureAngle (a : Angle) : ℝ := sorry
def AdjacentAngles (a b : Angle) : Prop := sorry
def AlternateInteriorAngles (a b : Angle) (m n : Line) : Prop := sorry

theorem angle_measure_in_parallel_lines (m n : Line) (y : Angle) :
  Parallel m n →
  ∃ (a b c d : Angle),
    MeasureAngle a = 40 ∧
    MeasureAngle b = 40 ∧
    MeasureAngle c = 90 ∧
    MeasureAngle d = 90 ∧
    AdjacentAngles a c ∧
    AdjacentAngles b d ∧
    AlternateInteriorAngles a b m n →
  MeasureAngle y = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_parallel_lines_l1082_108257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_removable_count_l1082_108279

theorem largest_removable_count (n : ℕ) (h : n = 1000) : 
  ∃ m : ℕ, m = 499 ∧ 
  (∀ S : Finset ℕ, S ⊆ Finset.range n → S.card = n - m → 
    ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b) ∧
  (∀ k : ℕ, k > m → ∃ S : Finset ℕ, S ⊆ Finset.range n ∧ S.card = n - k ∧ 
    ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(a ∣ b ∨ b ∣ a)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_removable_count_l1082_108279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_disjoint_condition_l1082_108255

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - x) + Real.log (3^x - 9)

-- Define the domain A of f
def A : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 4}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x : ℝ | (x - a) * (x - (a + 3)) < 0}

-- Theorem for the first case
theorem subset_condition (a : ℝ) : A ⊆ B a → 1 < a ∧ a ≤ 2 := by sorry

-- Theorem for the second case
theorem disjoint_condition (a : ℝ) : A ∩ B a = ∅ → a ≤ -1 ∨ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_disjoint_condition_l1082_108255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_tank_trips_l1082_108220

/-- The volume of a hemisphere with given radius -/
noncomputable def hemisphereVolume (radius : ℝ) : ℝ := (2 / 3) * Real.pi * radius^3

/-- The volume of a cylinder with given radius and height -/
noncomputable def cylinderVolume (radius height : ℝ) : ℝ := Real.pi * radius^2 * height

/-- The number of trips required to fill a cylinder with a hemisphere -/
noncomputable def tripsRequired (cylinderRadius cylinderHeight hemisphereRadius : ℝ) : ℕ :=
  Int.toNat (Int.ceil (cylinderVolume cylinderRadius cylinderHeight / hemisphereVolume hemisphereRadius))

theorem fill_tank_trips :
  tripsRequired 12 20 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_tank_trips_l1082_108220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1082_108274

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x - Real.pi / 6) - 1

theorem f_properties :
  (∀ x, f x ≥ -Real.sqrt 3 - 1) ∧
  (∃ x, f x = -Real.sqrt 3 - 1) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1082_108274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l1082_108269

/-- The volume of the solid formed by rotating the region bounded by y = x^2, x = 2, and y = 0 around the Oy axis -/
noncomputable def rotationVolume : ℝ := 8 * Real.pi

/-- The function defining the curve y = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The upper bound of x -/
def upperBoundX : ℝ := 2

/-- The lower bound of y -/
def lowerBoundY : ℝ := 0

/-- The upper bound of y -/
def upperBoundY : ℝ := f upperBoundX

/-- Theorem stating the volume of rotation -/
theorem volume_of_rotation : 
  ∃ V : ℝ, V = rotationVolume ∧ 
  V = Real.pi * ∫ y in lowerBoundY..upperBoundY, (upperBoundX^2 - y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l1082_108269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_duration_is_24_hours_l1082_108247

/-- Represents the duration of a car trip with varying speeds. -/
structure CarTrip where
  initialHours : ℝ  -- Initial hours at slower speed
  initialSpeed : ℝ  -- Speed during initial hours (miles per hour)
  additionalSpeed : ℝ  -- Speed during additional hours (miles per hour)
  averageSpeed : ℝ  -- Average speed for the entire trip (miles per hour)

/-- Calculates the total trip time given the conditions. -/
noncomputable def totalTripTime (trip : CarTrip) : ℝ :=
  let initialDistance := trip.initialHours * trip.initialSpeed
  let additionalHours := (trip.averageSpeed * trip.initialHours - initialDistance) / (trip.additionalSpeed - trip.averageSpeed)
  trip.initialHours + additionalHours

/-- Theorem stating that the total trip time is 24 hours under the given conditions. -/
theorem trip_duration_is_24_hours :
  let trip := CarTrip.mk 4 35 53 50
  totalTripTime trip = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_duration_is_24_hours_l1082_108247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₂_f₃_same_cluster_l1082_108270

/-- Definition of a periodic function -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

/-- Definition of the amplitude of a function -/
def Amplitude (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ c, ∀ x, |f x - c| ≤ a ∧ ∃ x₀, |f x₀ - c| = a

/-- Definition of functions of the same cluster -/
def SameCluster (f g : ℝ → ℝ) : Prop :=
  ∃ p a, IsPeriodic f p ∧ IsPeriodic g p ∧ Amplitude f a ∧ Amplitude g a

/-- The two functions from the problem -/
noncomputable def f₂ (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 4)
noncomputable def f₃ (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

/-- Theorem stating that f₂ and f₃ are of the same cluster -/
theorem f₂_f₃_same_cluster : SameCluster f₂ f₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₂_f₃_same_cluster_l1082_108270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_approx_l1082_108230

noncomputable def expression (x y z u : Real) : Real :=
  ((Real.log x / Real.log 10) * (0.625 * Real.sin y) * Real.sqrt 0.0729 * Real.cos z * 28.9) /
  (0.0017 * 0.025 * 8.1 * Real.tan u)

noncomputable def deg_to_rad (deg : Real) : Real := deg * (Real.pi / 180)

theorem expression_value_approx :
  ∃ (ε : Real), ε > 0 ∧ ε < 0.1 ∧ 
  abs (expression 23 (deg_to_rad 58) (deg_to_rad 19) (deg_to_rad 33) - 1472.8) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_approx_l1082_108230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_divisibility_l1082_108241

-- Define the ceiling function as noncomputable
noncomputable def ceiling (x : ℝ) : ℤ := Int.ceil x

-- State the theorem
theorem smallest_integer_divisibility (n : ℕ) (hn : n > 0) :
  ∃ k : ℤ, ceiling ((Real.sqrt 3 + 1) ^ (2 * n)) = k * (2 ^ (n + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_divisibility_l1082_108241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_triple_existence_l1082_108216

-- Define the set of numbers from 2 to 70
def S : Set ℕ := {n | 2 ≤ n ∧ n ≤ 70}

-- Define a partition of S into four disjoint subsets
def is_partition (X₁ X₂ X₃ X₄ : Set ℕ) : Prop :=
  (X₁ ∪ X₂ ∪ X₃ ∪ X₄ = S) ∧
  (X₁ ∩ X₂ = ∅) ∧ (X₁ ∩ X₃ = ∅) ∧ (X₁ ∩ X₄ = ∅) ∧
  (X₂ ∩ X₃ = ∅) ∧ (X₂ ∩ X₄ = ∅) ∧ (X₃ ∩ X₄ = ∅)

-- Define the property we want to prove
def has_triple (X : Set ℕ) : Prop :=
  ∃ a b c, a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ 71 ∣ (a * b - c)

-- The main theorem
theorem partition_triple_existence :
  ∀ X₁ X₂ X₃ X₄ : Set ℕ,
  is_partition X₁ X₂ X₃ X₄ →
  has_triple X₁ ∨ has_triple X₂ ∨ has_triple X₃ ∨ has_triple X₄ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_triple_existence_l1082_108216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tip_percentage_under_20_l1082_108260

/-- Represents the maximum tip percentage for a given meal cost and total amount paid -/
noncomputable def max_tip_percentage (meal_cost : ℝ) (total_paid : ℝ) : ℝ :=
  ((total_paid - meal_cost) / meal_cost) * 100

/-- Theorem stating the maximum tip percentage for the given problem -/
theorem max_tip_percentage_under_20 (meal_cost : ℝ) (total_paid : ℝ) 
  (h1 : meal_cost = 37.25)
  (h2 : total_paid = 40.975)
  (h3 : max_tip_percentage meal_cost total_paid > 10)
  (h4 : max_tip_percentage meal_cost total_paid < 20) :
  ∃ (ε : ℝ), ε > 0 ∧ max_tip_percentage meal_cost total_paid = 20 - ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tip_percentage_under_20_l1082_108260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_circleplus_equality_l1082_108234

-- Define the operation ⊕
noncomputable def circleplus (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Define y as the nested operation from 3 to 101
noncomputable def y : ℝ := sorry

-- Theorem statement
theorem nested_circleplus_equality :
  circleplus 2 (circleplus 3 (circleplus 4 (circleplus 100 101))) = (2 + y) / (1 + 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_circleplus_equality_l1082_108234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_l1082_108240

/-- The maximum value of sin²(θ/2) * (1 + cos θ) is 1/2 for 0 < θ < π. -/
theorem max_value_sin_cos (θ : ℝ) (h : 0 < θ ∧ θ < π) :
  (∀ φ : ℝ, 0 < φ ∧ φ < π → Real.sin (φ / 2)^2 * (1 + Real.cos φ) ≤ Real.sin (θ / 2)^2 * (1 + Real.cos θ)) →
  Real.sin (θ / 2)^2 * (1 + Real.cos θ) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_l1082_108240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1082_108207

/-- The distance between two parallel lines in the form ax + by + c = 0 -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- The first line equation: x + y - 2 = 0 -/
def line1 (x y : ℝ) : Prop := x + y - 2 = 0

/-- The second line equation: x + y + 1 = 0 -/
def line2 (x y : ℝ) : Prop := x + y + 1 = 0

theorem distance_between_given_lines :
  distance_between_parallel_lines 1 1 (-2) (-1) = 3 * Real.sqrt 2 / 2 := by
  sorry

#check distance_between_given_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1082_108207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pioneer_club_attendance_l1082_108212

def pioneer_clubs (n m : ℕ) : Fin n → Set (Fin m) := sorry

theorem pioneer_club_attendance (n m : ℕ) (hn : n = 11) (hm : m = 5) :
  ∃ (A B : Fin n), A ≠ B ∧ (∀ (c : Fin m), c ∈ (pioneer_clubs n m A) → c ∈ (pioneer_clubs n m B)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pioneer_club_attendance_l1082_108212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expo_park_area_scientific_notation_l1082_108285

/-- Scientific notation representation of a real number -/
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

/-- Definition of valid scientific notation -/
def is_valid_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10

theorem expo_park_area_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    scientific_notation a n = 5280000 ∧
    is_valid_scientific_notation a n ∧
    a = 5.28 ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expo_park_area_scientific_notation_l1082_108285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_but_not_complementary_l1082_108272

/-- Represents the group of students -/
structure StudentGroup :=
  (boys : ℕ)
  (girls : ℕ)

/-- Represents a selection of students -/
structure Selection :=
  (boys : ℕ)
  (girls : ℕ)

/-- Defines the event "At least 1 boy is selected" -/
def at_least_one_boy (s : Selection) : Prop :=
  s.boys > 0

/-- Defines the event "All selected are girls" -/
def all_girls (s : Selection) : Prop :=
  s.girls = 2 ∧ s.boys = 0

/-- Defines mutual exclusivity of two events -/
def mutually_exclusive (e1 e2 : Selection → Prop) : Prop :=
  ∀ s : Selection, ¬(e1 s ∧ e2 s)

/-- Defines complementarity of two events -/
def complementary (e1 e2 : Selection → Prop) : Prop :=
  ∀ s : Selection, e1 s ∨ e2 s

/-- The main theorem to be proved -/
theorem events_mutually_exclusive_but_not_complementary 
  (g : StudentGroup) (h1 : g.boys = 3) (h2 : g.girls = 2) : 
  mutually_exclusive at_least_one_boy all_girls ∧ 
  ¬(complementary at_least_one_boy all_girls) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_but_not_complementary_l1082_108272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_squares_l1082_108297

noncomputable section

-- Define the side length of the squares
def side_length : ℝ := 12

-- Define the angle of rotation
def rotation_angle : ℝ := Real.pi / 4  -- 45 degrees in radians

-- Theorem statement
theorem area_of_overlapping_squares :
  let square_area := side_length ^ 2
  let overlap_side := side_length * Real.sin rotation_angle
  let overlap_area := overlap_side ^ 2
  let total_area := 2 * square_area - overlap_area
  total_area = 216 := by
  -- Proof steps would go here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_squares_l1082_108297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_significant_digits_l1082_108276

/-- The number of significant digits in a real number -/
def significantDigits (x : ℝ) : ℕ := sorry

/-- The side length of a square given its area -/
noncomputable def squareSideLength (area : ℝ) : ℝ := Real.sqrt area

theorem square_side_significant_digits (area : ℝ) (h : area = 3.2400) :
  significantDigits (squareSideLength area) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_significant_digits_l1082_108276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_isosceles_triangle_l1082_108278

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We represent angles in degrees as integers for simplicity
  angle1 : ℕ
  angle2 : ℕ
  angle3 : ℕ
  -- Conditions for an isosceles triangle
  sum_of_angles : angle1 + angle2 + angle3 = 180
  equal_angles : angle1 = angle2
  angle_30 : angle1 = 30

-- Theorem statement
theorem largest_angle_in_isosceles_triangle (t : IsoscelesTriangle) : 
  max t.angle1 (max t.angle2 t.angle3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_isosceles_triangle_l1082_108278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1082_108204

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 * a - 1) * x - 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-3/2) 2

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ interval, f a x ≤ 1) ∧ (∃ x ∈ interval, f a x = 1) ↔ 
  (a = 3/4 ∨ a = (-3 - 2 * Real.sqrt 2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1082_108204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1082_108298

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem: A train 100 meters long, traveling at 18 kmph, will take 50 seconds to cross a bridge 150 meters long -/
theorem train_crossing_bridge_time :
  train_crossing_time 100 150 18 = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1082_108298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1082_108218

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^5 + Real.tan x - 3

-- State the theorem
theorem f_symmetry (m : ℝ) (hm : f (-m) = -2) : f m = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l1082_108218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_matrix_l1082_108232

/-- A 10x10 matrix of natural numbers -/
def SpecialMatrix := Fin 10 → Fin 10 → ℕ

/-- The product of a row in the matrix -/
def rowProduct (m : SpecialMatrix) (i : Fin 10) : ℕ := (Finset.univ.prod fun j => m i j)

/-- The product of a column in the matrix -/
def columnProduct (m : SpecialMatrix) (j : Fin 10) : ℕ := (Finset.univ.prod fun i => m i j)

/-- Check if a sequence forms a nontrivial arithmetic progression -/
def isNontrivialAP (s : Fin 10 → ℕ) : Prop :=
  ∃ (a d : ℕ), d ≠ 0 ∧ ∀ (i : Fin 10), s i = a + i.val * d

/-- The main theorem statement -/
theorem exists_special_matrix : ∃ (m : SpecialMatrix),
  (∀ (i j k l : Fin 10), i ≠ k ∨ j ≠ l → m i j ≠ m k l) ∧
  isNontrivialAP (rowProduct m) ∧
  isNontrivialAP (columnProduct m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_matrix_l1082_108232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_inequality_solution_l1082_108294

theorem fraction_inequality_solution (x : ℝ) : 
  (x - 1) / (x + 2) ≥ 0 ↔ x ∈ Set.Iic (-2) ∪ Set.Ici 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_inequality_solution_l1082_108294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_set_theorem_l1082_108290

def has_no_square_factors (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → ¬(m * m ∣ n)

def satisfies_condition (p : ℕ) : Prop :=
  Nat.Prime p ∧
  ∀ q : ℕ, Nat.Prime q → q < p →
    has_no_square_factors (p - p / q * q)

theorem prime_set_theorem :
  {p : ℕ | satisfies_condition p} = {2, 3, 5, 7, 13} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_set_theorem_l1082_108290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_first_quadrant_l1082_108237

/-- Two lines in the xy-plane -/
structure Lines where
  a : ℝ
  line1 : ℝ × ℝ → Prop := fun (x, y) ↦ a * x + y - 4 = 0
  line2 : ℝ × ℝ → Prop := fun (x, y) ↦ x - y - 2 = 0

/-- The intersection point of two lines -/
noncomputable def intersection (l : Lines) : ℝ × ℝ :=
  (6 / (l.a + 1), (4 - 2 * l.a) / (l.a + 1))

/-- Predicate for a point being in the first quadrant -/
def inFirstQuadrant : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x > 0 ∧ y > 0

/-- Theorem stating the equivalence between the intersection point being in the first quadrant
    and the parameter a being in the open interval (-1, 2) -/
theorem intersection_in_first_quadrant (l : Lines) :
  inFirstQuadrant (intersection l) ↔ -1 < l.a ∧ l.a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_first_quadrant_l1082_108237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_arithmetic_sequence_prime_or_power_of_two_l1082_108227

theorem coprime_arithmetic_sequence_prime_or_power_of_two (n : ℕ) :
  n > 6 →
  (∃ (k : ℕ) (a : ℕ → ℕ) (d : ℕ),
    (∀ i, i ∈ Finset.range k → 0 < a i ∧ a i < n ∧ Nat.Coprime (a i) n) ∧
    (∀ i, i ∈ Finset.range (k - 1) → a (i + 1) - a i = d) ∧
    d > 0 ∧
    (∀ m, 0 < m ∧ m < n ∧ Nat.Coprime m n → ∃ i, i ∈ Finset.range k ∧ m = a i)) →
  Nat.Prime n ∨ ∃ m, n = 2^m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_arithmetic_sequence_prime_or_power_of_two_l1082_108227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_sector_l1082_108211

/-- Represents a circular sector --/
structure Sector where
  radius : ℝ
  angle : ℝ
  perimeter : ℝ

/-- The area of a circular sector --/
noncomputable def sectorArea (s : Sector) : ℝ :=
  (1/2) * s.radius * s.radius * s.angle

/-- The arc length of a circular sector --/
noncomputable def arcLength (s : Sector) : ℝ :=
  s.radius * s.angle

/-- Theorem: Maximum area of a sector with perimeter 8 cm occurs when the angle is 2 radians --/
theorem max_area_sector (s : Sector) (h : s.perimeter = 8) :
  sectorArea s ≤ 4 ∧ (sectorArea s = 4 ↔ s.angle = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_sector_l1082_108211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1082_108263

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) / (x - 1)

-- State the theorem
theorem min_value_of_f :
  ∀ x > 1, f x ≥ 2 * Real.sqrt 3 + 2 ∧
  ∃ x > 1, f x = 2 * Real.sqrt 3 + 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1082_108263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1082_108206

def my_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  a 1 = 1 ∧
  ∀ n, a (n + 2) = 1 / (a n + 1)

theorem sequence_sum (a : ℕ → ℝ) (h : my_sequence a) (h100 : a 100 = a 96) :
  a 2018 + a 3 = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1082_108206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l1082_108281

/-- The side length of the base of a right pyramid with an equilateral triangular base. -/
noncomputable def base_side_length (lateral_face_area : ℝ) (slant_height : ℝ) : ℝ :=
  (2 * lateral_face_area) / slant_height

/-- Theorem stating that the side length of the base is 7.5 meters under given conditions. -/
theorem pyramid_base_side_length :
  base_side_length 150 40 = 7.5 := by
  -- Unfold the definition of base_side_length
  unfold base_side_length
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l1082_108281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_logarithm_l1082_108233

theorem max_value_logarithm (x y : ℝ) 
  (h1 : 1 ≤ Real.log (x * y^2) ∧ Real.log (x * y^2) ≤ 2 * Real.log 10) 
  (h2 : -Real.log 10 ≤ Real.log (x^2 / y) ∧ Real.log (x^2 / y) ≤ 2 * Real.log 10) : 
  (∀ z, z = Real.log (x^3 / y^4) → z ≤ 3 * Real.log 10) ∧ 
  (∃ x y, Real.log (x^3 / y^4) = 3 * Real.log 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_logarithm_l1082_108233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_lengths_l1082_108215

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  right_angle_A : (A.x - D.x) * (A.x - B.x) + (A.y - D.y) * (A.y - B.y) = 0
  right_angle_C : (C.x - B.x) * (C.x - D.x) + (C.y - B.y) * (C.y - D.y) = 0
  E_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E.x = A.x + t * (C.x - A.x) ∧ E.y = A.y + t * (C.y - A.y)
  F_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F.x = A.x + t * (C.x - A.x) ∧ F.y = A.y + t * (C.y - A.y)
  DE_perp_AC : (D.x - E.x) * (C.x - A.x) + (D.y - E.y) * (C.y - A.y) = 0
  BF_perp_AC : (B.x - F.x) * (C.x - A.x) + (B.y - F.y) * (C.y - A.y) = 0
  AE_length : (A.x - E.x)^2 + (A.y - E.y)^2 = 16
  DE_length : (D.x - E.x)^2 + (D.y - E.y)^2 = 36
  CE_length : (C.x - E.x)^2 + (C.y - E.y)^2 = 64

/-- The main theorem to be proved -/
theorem quadrilateral_lengths (q : Quadrilateral) : 
  abs (Real.sqrt ((q.B.x - q.F.x)^2 + (q.B.y - q.F.y)^2) - 11.08) < 0.01 ∧
  abs (Real.sqrt ((q.A.x - q.B.x)^2 + (q.A.y - q.B.y)^2) - 13.56) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_lengths_l1082_108215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_longer_leg_l1082_108231

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  shorterLeg : ℝ
  longerLeg : ℝ
  hypotenuse : ℝ
  ratio_constraint : shorterLeg * Real.sqrt 3 = longerLeg ∧ 2 * shorterLeg = hypotenuse

/-- Represents a sequence of four 30-60-90 triangles -/
structure FourTriangleSequence where
  t1 : Triangle30_60_90
  t2 : Triangle30_60_90
  t3 : Triangle30_60_90
  t4 : Triangle30_60_90
  adjacent_constraint : t1.longerLeg = t2.hypotenuse ∧ 
                        t2.longerLeg = t3.hypotenuse ∧ 
                        t3.longerLeg = t4.hypotenuse
  largest_hypotenuse : t1.hypotenuse = 16

theorem smallest_triangle_longer_leg 
  (seq : FourTriangleSequence) : seq.t4.longerLeg = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_longer_leg_l1082_108231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_surd_l1082_108228

-- Define a function to check if a number is a quadratic surd in its simplest form
def is_simplest_quadratic_surd (x : ℝ) : Prop :=
  ∃ (n : ℕ), n > 1 ∧ ¬ (∃ (m : ℕ), m^2 = n) ∧ x = Real.sqrt n

-- State the theorem
theorem simplest_quadratic_surd :
  ¬ is_simplest_quadratic_surd (Real.sqrt (1/2)) ∧
  ¬ is_simplest_quadratic_surd 2 ∧
  is_simplest_quadratic_surd (Real.sqrt 3) ∧
  ¬ is_simplest_quadratic_surd (Real.sqrt 16) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_surd_l1082_108228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_decrease_l1082_108223

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

theorem equilateral_triangle_area_decrease
  (original_area : ℝ)
  (side_decrease : ℝ)
  (h1 : original_area = 121 * Real.sqrt 3)
  (h2 : side_decrease = 6)
  (h3 : ∃ (s : ℝ), equilateral_triangle_area s = original_area) :
  ∃ (s : ℝ), 
    equilateral_triangle_area s = original_area ∧
    equilateral_triangle_area (s - side_decrease) = original_area - 57 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_decrease_l1082_108223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equality_l1082_108282

/-- Given a triangle ABC with area 10 and a point D on AB such that AD:DB = 2:3,
    if there exists a point E on BC forming triangle ABE with the same area as
    quadrilateral DBEF (where F is on CA), then the area of triangle ABE is 6. -/
theorem triangle_area_equality (A B C D E F : ℝ × ℝ) : 
  let triangle_area (p q r : ℝ × ℝ) := abs ((p.1 - r.1) * (q.2 - r.2) - (q.1 - r.1) * (p.2 - r.2)) / 2
  triangle_area A B C = 10 →
  D.1 = (2 * B.1 + 3 * A.1) / 5 ∧ D.2 = (2 * B.2 + 3 * A.2) / 5 →
  ∃ t, E.1 = B.1 + t * (C.1 - B.1) ∧ E.2 = B.2 + t * (C.2 - B.2) →
  ∃ s, F.1 = C.1 + s * (A.1 - C.1) ∧ F.2 = C.2 + s * (A.2 - C.2) →
  triangle_area A B E = triangle_area D B E + triangle_area D E F →
  triangle_area A B E = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equality_l1082_108282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_property_l1082_108284

-- Define the operation (a,b) as noncomputable
noncomputable def operation (a b : ℝ) : ℝ := Real.log b / Real.log a

-- Theorem statement
theorem operation_property :
  operation 3 7 + operation 3 8 = operation 3 56 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_property_l1082_108284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_volume_proof_l1082_108242

/-- The original volume of ice that melts at given rates over 4 hours -/
noncomputable def original_ice_volume (final_volume : ℝ) : ℝ :=
  let first_hour_remaining := 1/4
  let second_hour_remaining := 1/4
  let third_hour_remaining := 1/3
  let fourth_hour_remaining := 1/2
  final_volume / (first_hour_remaining * second_hour_remaining * third_hour_remaining * fourth_hour_remaining)

/-- Theorem stating that given the melting rates and final volume, the original ice volume is 48 cubic inches -/
theorem ice_volume_proof (final_volume : ℝ) (h_final : final_volume = 0.5) :
  original_ice_volume final_volume = 48 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval original_ice_volume 0.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_volume_proof_l1082_108242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_two_implies_a_value_non_negative_for_x_ge_one_implies_a_range_sum_of_squared_logs_inequality_l1082_108275

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1/x - a * Real.log x

theorem extremum_at_two_implies_a_value (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f a x ≥ f a 2 ∨ f a x ≤ f a 2) →
  a = 5/2 :=
by sorry

theorem non_negative_for_x_ge_one_implies_a_range (a : ℝ) :
  (∀ x ≥ 1, f a x ≥ 0) →
  a ≤ 2 :=
by sorry

theorem sum_of_squared_logs_inequality (n : ℕ) :
  n > 0 →
  (Finset.range n).sum (λ i => Real.log ((i + 2 : ℝ) / (i + 1 : ℝ))^2) < n / (n + 1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_two_implies_a_value_non_negative_for_x_ge_one_implies_a_range_sum_of_squared_logs_inequality_l1082_108275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_deducted_salary_l1082_108267

noncomputable def weekly_salary : ℚ := 1379
def work_days_per_week : ℕ := 5
def missed_days : ℕ := 2

noncomputable def daily_salary : ℚ := weekly_salary / work_days_per_week
noncomputable def deduction : ℚ := daily_salary * missed_days
noncomputable def deducted_salary : ℚ := weekly_salary - deduction

theorem anna_deducted_salary : 
  deducted_salary = 827.40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_deducted_salary_l1082_108267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_add_and_round_to_ten_thousandth_l1082_108213

/-- Rounds a given real number to the nearest ten-thousandth -/
noncomputable def round_to_ten_thousandth (x : ℝ) : ℝ :=
  ⌊x * 10000 + 0.5⌋ / 10000

/-- Proves that adding 174.39875 and 28.06754, and then rounding the result
    to the nearest ten-thousandth, equals 202.4663 -/
theorem add_and_round_to_ten_thousandth :
  round_to_ten_thousandth (174.39875 + 28.06754) = 202.4663 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_add_and_round_to_ten_thousandth_l1082_108213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_curved_surface_area_l1082_108221

/-- The area of the curved surface of a right circular cone -/
noncomputable def curvedSurfaceArea (slantHeight height : ℝ) : ℝ :=
  let radius := Real.sqrt (slantHeight ^ 2 - height ^ 2)
  Real.pi * radius * slantHeight

/-- Theorem: The curved surface area of a right circular cone with slant height 10 and height 8 is 60π -/
theorem cone_curved_surface_area :
  curvedSurfaceArea 10 8 = 60 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_curved_surface_area_l1082_108221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_second_derivative_l1082_108203

open Real

theorem log2_second_derivative (x : ℝ) (h : x > 0) :
  (deriv (deriv (λ y => log y / log 2))) x = 1 / (x * log 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_second_derivative_l1082_108203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_x_squared_solution_set_l1082_108293

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x + 2 else x + 2

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem f_geq_x_squared_solution_set :
  {x : ℝ | f x ≥ x^2} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_x_squared_solution_set_l1082_108293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l1082_108201

theorem cos_double_angle (α : ℝ) (h : Real.cos α = 2/3) : Real.cos (2*α) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l1082_108201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1082_108253

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The perpendicular bisector of BC intersects AB at D and BC at E -/
structure Quadrilateral (t : Triangle) where
  D : ℝ
  E : ℝ

/-- Helper function to calculate the area of the quadrilateral ACED -/
noncomputable def area_quadrilateral (t : Triangle) (q : Quadrilateral t) : ℝ :=
  sorry

theorem triangle_theorem (t : Triangle) (q : Quadrilateral t) :
  t.b * Real.cos t.C + t.c * Real.cos t.B = 4 * t.a * (Real.sin (t.A / 2))^2 →
  t.A = π / 3 ∧
  (t.a = 3 ∧ t.b = Real.sqrt 6 →
    area_quadrilateral t q = (9 + 6 * Real.sqrt 3) / 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1082_108253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_mean_score_l1082_108265

/-- Represents a class with students and their scores -/
structure ClassInfo where
  boys : ℕ
  girls : ℕ
  total_score : ℝ

/-- Calculate the mean score of a group -/
noncomputable def mean_score (number : ℕ) (total_score : ℝ) : ℝ :=
  total_score / number

theorem class_mean_score (class_7A class_7B : ClassInfo) : 
  class_7A.boys + class_7A.girls = 30 →
  class_7B.boys + class_7B.girls = 30 →
  mean_score class_7A.girls (class_7A.total_score - mean_score class_7A.boys class_7A.total_score * class_7A.boys) = 48 →
  mean_score (class_7A.boys + class_7A.girls + class_7B.boys + class_7B.girls) (class_7A.total_score + class_7B.total_score) = 60 →
  mean_score (class_7A.girls + class_7B.girls) ((class_7A.total_score - mean_score class_7A.boys class_7A.total_score * class_7A.boys) + (class_7B.total_score - mean_score class_7B.boys class_7B.total_score * class_7B.boys)) = 60 →
  class_7B.girls = 5 →
  class_7A.boys = 15 →
  mean_score class_7B.girls (class_7B.total_score - mean_score class_7B.boys class_7B.total_score * class_7B.boys) = 2 * mean_score class_7A.boys class_7A.total_score →
  10 * mean_score class_7B.boys class_7B.total_score = 672 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_mean_score_l1082_108265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1082_108222

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition on f
def condition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, Differentiable ℝ f ∧ Differentiable ℝ (deriv f) ∧
    x * (deriv (deriv f) x) + 2 * f x > 0

-- Define the inequality
def inequality (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (x + 2016) * f (x + 2016) / 5 < 5 * f 5 / (x + 2016)

-- Theorem statement
theorem solution_set (f : ℝ → ℝ) (h : condition f) :
  {x : ℝ | inequality f x} = {x : ℝ | -2016 < x ∧ x < -2011} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1082_108222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_white_balls_l1082_108264

theorem estimate_white_balls
  (total_balls : ℚ)
  (total_draws : ℚ)
  (red_draws : ℚ)
  (h1 : total_balls = 10)
  (h2 : total_draws = 100)
  (h3 : red_draws = 80) :
  Int.floor (total_balls - (total_balls * red_draws / total_draws)) = 2 :=
by
  -- Replace the natural numbers with rationals for more precise calculations
  -- Use Int.floor instead of ℕ.floor
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_white_balls_l1082_108264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_l1082_108226

/-- Given a triangle ABC with side a = 2, cos A = 1/3, and area = √2, prove that side b = √3 -/
theorem triangle_side_b (A B C : Real) (a b c : Real) : 
  a = 2 → 
  Real.cos A = 1/3 → 
  (1/2) * b * c * Real.sin A = Real.sqrt 2 → 
  b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_l1082_108226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_decrease_l1082_108243

/-- Proves that the average age of a class decreases by 4 years when new students join. -/
theorem average_age_decrease (original_avg new_students_avg : ℝ) 
  (original_strength new_students : ℕ) 
  (h1 : original_avg = 40)
  (h2 : new_students_avg = 32)
  (h3 : original_strength = 12)
  (h4 : new_students = 12)
  : (original_avg * (original_strength : ℝ) + new_students_avg * (new_students : ℝ)) / 
    ((original_strength + new_students : ℕ) : ℝ) = original_avg - 4 := by
  sorry

#check average_age_decrease

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_decrease_l1082_108243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_is_two_std_dev_below_mean_l1082_108289

/-- Calculates the number of standard deviations a value is from the mean -/
noncomputable def standardDeviationsFromMean (mean stdDev value : ℝ) : ℝ :=
  (value - mean) / stdDev

theorem value_is_two_std_dev_below_mean :
  let mean : ℝ := 10.5
  let stdDev : ℝ := 1
  let value : ℝ := 8.5
  standardDeviationsFromMean mean stdDev value = -2 := by
  -- Unfold the definition and simplify
  unfold standardDeviationsFromMean
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_is_two_std_dev_below_mean_l1082_108289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_c_is_valid_two_variable_linear_equation_l1082_108224

-- Define a structure for a linear equation
structure LinearEquation where
  lhs : List (ℚ × String)  -- List of (coefficient, variable) pairs
  rhs : ℚ                  -- Right-hand side constant

-- Function to count unique variables in an equation
def countUniqueVariables (eq : LinearEquation) : Nat :=
  (eq.lhs.map (λ p => p.2)).eraseDups.length

-- Function to check if all variables have degree 1
def allVariablesDegreeOne (eq : LinearEquation) : Bool :=
  eq.lhs.all (λ p => p.1 ≠ 0)

-- Function to check if equation has variables in denominator
def noVariablesInDenominator (eq : LinearEquation) : Bool :=
  true  -- This is always true for our linear equation representation

-- Define the equations from the problem
def eqA : LinearEquation := ⟨[(1, "x"), (-2, "x")], 1⟩
def eqB : LinearEquation := ⟨[(1, "x")], 1⟩  -- Simplified representation
def eqC : LinearEquation := ⟨[(1, "x"), (1, "z")], 3⟩
def eqD : LinearEquation := ⟨[(1, "x"), (-1, "y"), (1, "z")], 1⟩

-- Theorem statement
theorem equation_c_is_valid_two_variable_linear_equation :
  (countUniqueVariables eqC = 2) ∧
  allVariablesDegreeOne eqC ∧
  noVariablesInDenominator eqC ∧
  ¬((countUniqueVariables eqA = 2) ∧ allVariablesDegreeOne eqA ∧ noVariablesInDenominator eqA) ∧
  ¬((countUniqueVariables eqB = 2) ∧ allVariablesDegreeOne eqB ∧ noVariablesInDenominator eqB) ∧
  ¬((countUniqueVariables eqD = 2) ∧ allVariablesDegreeOne eqD ∧ noVariablesInDenominator eqD) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_c_is_valid_two_variable_linear_equation_l1082_108224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sulfur_produced_l1082_108236

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Avogadro's constant -/
def avogadro : ℝ := 6.02e23

/-- Represents the balanced chemical equation for the reaction -/
structure Reaction where
  h2s : Moles  -- moles of H₂S
  so2 : Moles  -- moles of SO₂
  s : Moles    -- moles of S
  h2o : Moles  -- moles of H₂O

/-- The stoichiometric coefficients of the reaction -/
def reaction_coefficients : Reaction :=
  { h2s := (2 : ℝ), so2 := (1 : ℝ), s := (3 : ℝ), h2o := (2 : ℝ) }

/-- The number of electrons transferred in the reaction -/
def electrons_transferred : ℝ := 4 * avogadro

/-- Theorem stating that when 4 × 6.02 × 10²³ electrons are transferred in the given reaction,
    3 mol of elemental sulfur is produced -/
theorem sulfur_produced (r : Reaction) (h : r = reaction_coefficients) :
  electrons_transferred = 4 * avogadro → r.s = (3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sulfur_produced_l1082_108236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_AB_length_l1082_108287

-- Define the ellipse Q
def ellipse (a : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + y^2 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define the perpendicular bisector of AB
noncomputable def perp_bisector (k : ℝ) (x : ℝ) : ℝ := 
  let x₀ := -2 * k^2 / (1 + 2 * k^2)
  let y₀ := k / (1 + 2 * k^2)
  y₀ - (1/k) * (x - x₀)

-- Define the x-coordinate of point P
noncomputable def x_P (k : ℝ) : ℝ := -1/2 + 1 / (4 * k^2 + 2)

-- Define the length of AB
noncomputable def AB_length (k : ℝ) : ℝ := 
  2 * Real.sqrt 2 * (1/2 + 1 / (2 * (2 * k^2 + 1)))

theorem min_AB_length :
  ∀ a : ℝ, a > 1 →
  ∀ k : ℝ, k ≠ 0 →
  (∀ x y : ℝ, ellipse a x y → 
    (∃ t : ℝ, y = line_l k x ∧ t ∈ Set.Icc (-1/4 : ℝ) 0 ∧ perp_bisector k t = 0)) →
  ∃ min_length : ℝ, 
    (∀ k : ℝ, k ≠ 0 → AB_length k ≥ min_length) ∧
    min_length = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_AB_length_l1082_108287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_pi_twelfth_l1082_108271

theorem sin_plus_pi_twelfth (α : Real) 
  (h1 : Real.sin α + Real.cos α = Real.sqrt 2 / 3) 
  (h2 : 0 < α ∧ α < Real.pi) : 
  Real.sin (α + Real.pi / 12) = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_pi_twelfth_l1082_108271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_y_axis_l1082_108239

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by x^2 = 4y -/
def Parabola := {p : Point | p.x^2 = 4 * p.y}

/-- The focus of the parabola -/
def focus : Point := ⟨0, 1⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

theorem parabola_point_distance_to_y_axis 
  (M : Point) 
  (h1 : M ∈ Parabola) 
  (h2 : distance M focus = 3) :
  distanceToYAxis M = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_y_axis_l1082_108239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_closer_to_CD_is_15_28_l1082_108283

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  AD : ℝ
  angle_A : ℝ
  h : AB ≠ CD
  i : AD > 0
  j : angle_A > 0 ∧ angle_A < π

/-- The fraction of the area closer to the longer base CD in a trapezoid -/
noncomputable def fraction_closer_to_longer_base (t : Trapezoid) : ℝ :=
  (t.AB + 3 * t.CD) / (4 * (t.AB + t.CD))

/-- Theorem stating the fraction of area closer to CD in the given trapezoid -/
theorem fraction_closer_to_CD_is_15_28 (t : Trapezoid) 
  (h1 : t.AB = 150)
  (h2 : t.CD = 200)
  (h3 : t.AD = 130)
  (h4 : t.angle_A = 75 * π / 180) :
  fraction_closer_to_longer_base t = 15 / 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_closer_to_CD_is_15_28_l1082_108283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l1082_108209

/-- Represents a triangle with given perimeter and inradius -/
structure Triangle where
  perimeter : ℝ
  inradius : ℝ

/-- The area of a triangle given its perimeter and inradius -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  t.inradius * (t.perimeter / 2)

/-- Theorem: A triangle with perimeter 20 and inradius 2.5 has an area of 25 -/
theorem triangle_area_theorem :
  ∃ (t : Triangle), t.perimeter = 20 ∧ t.inradius = 2.5 ∧ triangleArea t = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l1082_108209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_existence_l1082_108286

-- Define the circle C
def Circle (r : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}

-- Define the line l
def Line (k : ℝ) := {(x, y) : ℝ × ℝ | y - 1 = k * (x + 1)}

-- Define the origin
def Origin : ℝ × ℝ := (0, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_and_line_existence (r : ℝ) (h1 : r > 0) :
  ∃ (k : ℝ),
    (1, 2) ∈ Circle r ∧
    (-1, 1) ∈ Line k ∧
    ∃ (A B : ℝ × ℝ),
      A ∈ Circle r ∧ B ∈ Circle r ∧
      A ∈ Line k ∧ B ∈ Line k ∧
      A ≠ B ∧
      distance Origin ((-1, 1)) = distance Origin A + distance Origin B →
    r = 2 ∧ k = 1 := by
  sorry

#check circle_and_line_existence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_existence_l1082_108286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_symmetric_point_l1082_108219

/-- Given a point M(4, 7, 6) in a three-dimensional Cartesian coordinate system,
    this theorem proves that the coordinates of the projection of the point symmetric to M
    with respect to the y-axis on the xOz coordinate plane are (-4, 0, -6). -/
theorem projection_of_symmetric_point (M : ℝ × ℝ × ℝ) 
  (h : M = (4, 7, 6)) :
  let M' := (-(M.fst), M.snd.fst, -(M.snd.snd))  -- Point symmetric to M with respect to y-axis
  (M'.1, 0, M'.2.2) = (-4, 0, -6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_symmetric_point_l1082_108219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_correct_l1082_108246

/-- Represents an electrical circuit with a resistor, capacitor, and galvanic cell. -/
structure Circuit where
  R : ℝ  -- Resistance of the resistor
  C : ℝ  -- Capacitance of the capacitor
  ε : ℝ  -- Electromotive force (EMF) of the galvanic cell
  r : ℝ  -- Internal resistance of the galvanic cell

/-- Calculates the heat released in the resistor during capacitor charging. -/
noncomputable def heat_released (circuit : Circuit) : ℝ :=
  (circuit.C * circuit.ε^2 * circuit.R) / (2 * (circuit.R + circuit.r))

/-- Theorem stating that the heat released in the resistor is correctly calculated. -/
theorem heat_released_correct (circuit : Circuit) :
  heat_released circuit = (circuit.C * circuit.ε^2 * circuit.R) / (2 * (circuit.R + circuit.r)) :=
by
  -- Unfold the definition of heat_released
  unfold heat_released
  -- The equation is now trivially true
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_correct_l1082_108246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1082_108258

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 3 - x) - Real.cos (Real.pi / 6 + x)

theorem min_value_of_f :
  ∃ (min : ℝ), (∀ x, f x ≥ min) ∧ (∃ x, f x = min) ∧ min = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1082_108258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_comparison_l1082_108296

/-- Represents a square divided into regions -/
structure DividedSquare where
  total_area : ℝ
  shaded_area : ℝ

/-- Square I divided by diagonals with one triangle shaded -/
noncomputable def square_I : DividedSquare := {
  total_area := 1,
  shaded_area := 1/4
}

/-- Square II divided into four rectangles with two adjacent shaded -/
noncomputable def square_II : DividedSquare := {
  total_area := 1,
  shaded_area := 1/2
}

/-- Square III divided into eight triangles with four alternate shaded -/
noncomputable def square_III : DividedSquare := {
  total_area := 1,
  shaded_area := 1/2
}

/-- Theorem stating the relationship between shaded areas -/
theorem shaded_areas_comparison :
  square_II.shaded_area = square_III.shaded_area ∧
  square_I.shaded_area ≠ square_II.shaded_area ∧
  square_I.shaded_area ≠ square_III.shaded_area := by
  sorry

#eval "Theorem stated and proof skipped with 'sorry'"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_comparison_l1082_108296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l1082_108225

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Function to check if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Function to check if two lines are parallel -/
def areParallel (l1 l2 : Line) : Prop := sorry

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Function to check if a circle touches a line -/
def circleTouchesLine (c : Circle) (l : Line) : Prop := sorry

/-- Function to get the inscribed circle of a triangle -/
noncomputable def inscribedCircle (t : Triangle) : Circle := sorry

/-- Function to check if two geometric objects have exactly one common point -/
def haveOneCommonPoint {α β : Type} (obj1 : α) (obj2 : β) : Prop := sorry

/-- Main theorem -/
theorem inscribed_circle_radius
  (t : Triangle)
  (l1 l2 : Line)
  (c : Circle)
  (h1 : isIsosceles t)
  (h2 : areParallel l1 l2)
  (h3 : pointOnLine t.a l1)
  (h4 : pointOnLine t.c l1)
  (h5 : pointOnLine t.b l2)
  (h6 : c.radius = 1)
  (h7 : circleTouchesLine c l1)
  (h8 : circleTouchesLine c l2)
  (h9 : haveOneCommonPoint t c)
  (h10 : ∃ p : Point, pointOnLine p (Line.mk 0 0 0) ∧ pointOnLine p (Line.mk 0 0 0)) :
  (inscribedCircle t).radius = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l1082_108225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1082_108235

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 6)) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  f (2 * Real.pi / 3) = -2 ∧
  (∀ x ∈ Set.Icc (Real.pi / 12) (Real.pi / 2), f x ≤ 2) ∧
  (∃ x ∈ Set.Icc (Real.pi / 12) (Real.pi / 2), f x = 2) ∧
  (∀ x ∈ Set.Icc (Real.pi / 12) (Real.pi / 2), f x ≥ 0) ∧
  (∃ x ∈ Set.Icc (Real.pi / 12) (Real.pi / 2), f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1082_108235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_bound_l1082_108256

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := log x + 1/x
noncomputable def g (a x : ℝ) : ℝ := x + 1/(x-a)

-- State the theorem
theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ > a, f x₁ ≥ g a x₂) →
  a ≤ -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_bound_l1082_108256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_l1082_108248

theorem triangle_third_side (a b c θ : ℝ) : 
  a = 6 → b = 9 → θ = 150 * Real.pi / 180 → 
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) → 
  c = Real.sqrt (117 + 54 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_l1082_108248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l1082_108299

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (Real.sqrt 3 / 2) * t)

def on_curve_C (p : ℝ × ℝ) : Prop := p.1^2 + 2 * p.2^2 = 2

def point_P : ℝ × ℝ := (1, 0)

def dist_squared (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem intersection_reciprocal_sum :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧
  on_curve_C (line_l t₁) ∧
  on_curve_C (line_l t₂) ∧
  (1 / dist_squared (line_l t₁) point_P) + (1 / dist_squared (line_l t₂) point_P) = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l1082_108299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_satisfying_conditions_l1082_108295

theorem smallest_integer_satisfying_conditions : ℕ := by
  -- Define the fractions
  let f1 : ℚ := 5 / 7
  let f2 : ℚ := 7 / 9
  let f3 : ℚ := 9 / 11
  let f4 : ℚ := 11 / 13

  -- Define the fractional parts
  let p1 : ℚ := 2 / 5
  let p2 : ℚ := 2 / 7
  let p3 : ℚ := 2 / 9
  let p4 : ℚ := 2 / 11

  -- Define the conditions
  let satisfies_conditions (n : ℕ) : Prop :=
    (n : ℚ) / f1 - ⌊(n : ℚ) / f1⌋ = p1 ∧
    (n : ℚ) / f2 - ⌊(n : ℚ) / f2⌋ = p2 ∧
    (n : ℚ) / f3 - ⌊(n : ℚ) / f3⌋ = p3 ∧
    (n : ℚ) / f4 - ⌊(n : ℚ) / f4⌋ = p4

  -- The theorem statement
  have smallest_satisfying : ∀ n : ℕ, n > 1 → satisfies_conditions n → n ≥ 3466 := by
    sorry

  have smallest_is_3466 : satisfies_conditions 3466 := by
    sorry

  exact 3466

#check smallest_integer_satisfying_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_satisfying_conditions_l1082_108295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_legs_unique_plane_l1082_108205

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Vector3D

/-- A plane in 3D space -/
structure Plane3D where
  normal : Vector3D
  point : Point3D

/-- A trapezoid in 3D space -/
structure Trapezoid where
  leg1 : Line3D
  leg2 : Line3D

/-- Define a contains relation between a plane and a line -/
def contains (p : Plane3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p.normal.x * (l.point.x + t * l.direction.x - p.point.x) +
           p.normal.y * (l.point.y + t * l.direction.y - p.point.y) +
           p.normal.z * (l.point.z + t * l.direction.z - p.point.z) = 0

/-- Theorem: There exists a unique plane containing the lines on which the legs of a trapezoid lie -/
theorem trapezoid_legs_unique_plane (t : Trapezoid) : 
  ∃! (p : Plane3D), contains p t.leg1 ∧ contains p t.leg2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_legs_unique_plane_l1082_108205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_children_who_can_do_both_l1082_108200

/-- Given a group of children where:
  - The total number of children is 48
  - 38 children can do A
  - 29 children can do B
  - Every child can do at least one of A or B
  Prove that the number of children who can do both A and B is 19 -/
theorem children_who_can_do_both (total : ℕ) (can_do_A : ℕ) (can_do_B : ℕ) 
  (h_total : total = 48)
  (h_A : can_do_A = 38)
  (h_B : can_do_B = 29)
  (h_at_least_one : total = can_do_A + can_do_B - (can_do_A + can_do_B - total)) :
  can_do_A + can_do_B - total = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_children_who_can_do_both_l1082_108200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_linearity_l1082_108245

theorem equation_linearity (k : ℤ) : 
  (∀ x : ℚ, ∃ a b : ℚ, (x : ℚ)^(k.natAbs) + 5*k + 1 = a*x + b) ↔ (k = 1 ∨ k = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_linearity_l1082_108245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_positive_implies_inputs_sum_positive_l1082_108261

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1)) + 2*x + Real.sin x

-- State the theorem
theorem sum_positive_implies_inputs_sum_positive (x₁ x₂ : ℝ) 
  (h : f x₁ + f x₂ > 0) : x₁ + x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_positive_implies_inputs_sum_positive_l1082_108261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_bridge_time_is_80_seconds_l1082_108268

/-- Represents the train crossing scenario -/
structure TrainCrossing where
  train_length : ℝ
  signal_cross_time : ℝ
  bridge_cross_time : ℝ

/-- Calculates the time required for the train to cross just the bridge -/
noncomputable def time_to_cross_bridge (tc : TrainCrossing) : ℝ :=
  let train_speed := tc.train_length / tc.signal_cross_time
  let bridge_length := train_speed * tc.bridge_cross_time - tc.train_length
  bridge_length / train_speed

/-- Theorem stating that the time to cross just the bridge is 80 seconds -/
theorem cross_bridge_time_is_80_seconds (tc : TrainCrossing) 
  (h1 : tc.train_length = 600)
  (h2 : tc.signal_cross_time = 40)
  (h3 : tc.bridge_cross_time = 120) :
  time_to_cross_bridge tc = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_bridge_time_is_80_seconds_l1082_108268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_f_equals_three_halves_iff_l1082_108210

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then 1 + 1/x
  else if -1 ≤ x ∧ x ≤ 1 then x^2 + 1
  else 2*x + 3

-- Theorem 1: f(f(f(-2))) = 3/2
theorem f_composition_negative_two : f (f (f (-2))) = 3/2 := by sorry

-- Theorem 2: f(a) = 3/2 if and only if a = 2 or a = ± √2/2
theorem f_equals_three_halves_iff (a : ℝ) : 
  f a = 3/2 ↔ a = 2 ∨ a = Real.sqrt 2 / 2 ∨ a = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_two_f_equals_three_halves_iff_l1082_108210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1082_108259

/-- A quadrilateral with diagonals that are equal in length and bisect each other -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has diagonals that are equal in length -/
  equal_diagonals : Bool
  /-- The quadrilateral has diagonals that bisect each other -/
  bisecting_diagonals : Bool

/-- Predicate to check if a quadrilateral is a parallelogram -/
def is_parallelogram (q : SpecialQuadrilateral) : Prop :=
  q.equal_diagonals ∧ q.bisecting_diagonals

/-- Proposition p: Every quadrilateral with diagonals that are equal in length and bisect each other is a parallelogram -/
def proposition_p : Prop :=
  ∀ q : SpecialQuadrilateral, q.equal_diagonals ∧ q.bisecting_diagonals → is_parallelogram q

/-- The negation of proposition p -/
def negation_p : Prop :=
  ∃ q : SpecialQuadrilateral, q.equal_diagonals ∧ q.bisecting_diagonals ∧ ¬is_parallelogram q

theorem negation_equivalence : ¬proposition_p ↔ negation_p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1082_108259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1082_108277

/-- Hyperbola with one focus at (4,0) and eccentricity 2 -/
structure Hyperbola where
  focus : ℝ × ℝ := (4, 0)
  eccentricity : ℝ := 2

/-- Standard form of the hyperbola equation -/
def standard_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

/-- Asymptote equations of the hyperbola -/
def asymptote_equations (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- Distance from a point to a line ax + by + c = 0 -/
noncomputable def distance_point_to_line (px py a b c : ℝ) : ℝ :=
  |a * px + b * py + c| / Real.sqrt (a^2 + b^2)

theorem hyperbola_properties (h : Hyperbola) :
  (∀ x y, standard_equation x y ↔ 
    x^2 / 4 - y^2 / 12 = 1) ∧
  (∀ x y, asymptote_equations x y ↔ 
    y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) ∧
  (distance_point_to_line 4 0 (Real.sqrt 3) (-1) 0 = 2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1082_108277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1082_108217

-- Define the ellipse
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / m + y^2 = 1

-- Define the foci
noncomputable def leftFocus (m : ℝ) : ℝ × ℝ := (-Real.sqrt (m - 1), 0)
noncomputable def rightFocus (m : ℝ) : ℝ × ℝ := (Real.sqrt (m - 1), 0)

-- Define a point on the ellipse
def pointOnEllipse (m : ℝ) (P : ℝ × ℝ) : Prop :=
  ellipse m P.1 P.2

-- Define a point on the circle with diameter F₁F₂
def pointOnCircle (m : ℝ) (P : ℝ × ℝ) : Prop := by
  let F₁ := leftFocus m
  let F₂ := rightFocus m
  exact (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2

-- Define the slope condition for chord AB and OM
def slopeCondition (m : ℝ) (A B M : ℝ × ℝ) : Prop := by
  let KAB := (B.2 - A.2) / (B.1 - A.1)
  let KOM := M.2 / M.1
  exact KAB * KOM = -1/4

-- Define the midpoint condition
def midpointCondition (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Main theorem
theorem ellipse_properties (m : ℝ) :
  (∃ P, pointOnEllipse m P ∧ pointOnCircle m P) →
  (∃ A B M, pointOnEllipse m A ∧ pointOnEllipse m B ∧
            midpointCondition A B M ∧ slopeCondition m A B M) →
  (m = 4 ∧ Real.sqrt ((m - 1) / m) ∈ Set.Icc (Real.sqrt 2 / 2) 1) := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1082_108217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_has_winning_strategy_l1082_108251

/-- Represents a player in the game -/
inductive Player
| Petya
| Vasya

/-- Represents a strip placement on the board -/
structure Move where
  player : Player
  row : Nat
  col : Nat

/-- Represents the game state -/
structure GameState where
  board : List (List Bool)
  currentPlayer : Player

/-- Checks if a move is valid in the current game state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  sorry

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over (no more valid moves) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Determines the winner of the game -/
def getWinner (state : GameState) : Option Player :=
  sorry

/-- Initial game state -/
def initialGameState : GameState :=
  { board := List.replicate 3 (List.replicate 2021 false),
    currentPlayer := Player.Petya }

/-- Theorem stating that Petya has a winning strategy -/
theorem petya_has_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (game : List Move),
      (game.length % 2 = 0) →
      (∀ (i : Fin game.length), isValidMove (game.take i.val |> List.foldl applyMove initialGameState) (game[i])) →
      (isGameOver (game |> List.foldl applyMove initialGameState)) →
      (getWinner (game |> List.foldl applyMove initialGameState) = some Player.Petya) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_has_winning_strategy_l1082_108251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l1082_108202

theorem power_equation (x : ℝ) : (2 : ℝ)^(3*x) = 128 → (2 : ℝ)^(-x) = 1 / (2 : ℝ)^(7/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l1082_108202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_quarter_pi_l1082_108262

theorem cosine_sum_quarter_pi (α β : Real) : 
  α ∈ Set.Ioo (3 * Real.pi / 4) Real.pi →
  β ∈ Set.Ioo (3 * Real.pi / 4) Real.pi →
  Real.sin (α + β) = -3/5 →
  Real.sin (β - Real.pi/4) = 12/13 →
  Real.cos (α + Real.pi/4) = -56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_quarter_pi_l1082_108262
