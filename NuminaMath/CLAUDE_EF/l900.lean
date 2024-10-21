import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_specific_angles_l900_90068

theorem sin_sum_specific_angles (α β : ℝ) 
  (h1 : Real.sin α = 2/3)
  (h2 : α ∈ Set.Ioo (π/2) π)
  (h3 : Real.cos β = -3/5)
  (h4 : β ∈ Set.Ioo π (3*π/2)) :
  Real.sin (α + β) = (4 * Real.sqrt 5 - 6) / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_specific_angles_l900_90068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_good_n_is_correct_l900_90092

/-- A set is "good" if there are two elements whose product is a multiple of the GCD of the remaining two elements -/
def is_good (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ ∃ a b c d, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧
    (a * b % (Nat.gcd c d) = 0 ∨
     a * c % (Nat.gcd b d) = 0 ∨
     a * d % (Nat.gcd b c) = 0 ∨
     b * c % (Nat.gcd a d) = 0 ∨
     b * d % (Nat.gcd a c) = 0 ∨
     c * d % (Nat.gcd a b) = 0)

/-- The greatest possible value of n such that any four-element set with elements ≤ n is good -/
def greatest_good_n : ℕ := 230

theorem greatest_good_n_is_correct :
  (∀ s : Finset ℕ, s.card = 4 → (∀ x ∈ s, x ≤ greatest_good_n) → is_good s) ∧
  ∃ s : Finset ℕ, s.card = 4 ∧ (∀ x ∈ s, x ≤ greatest_good_n + 1) ∧ ¬is_good s :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_good_n_is_correct_l900_90092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_cube_root_l900_90001

theorem segment_length_cube_root (x₁ x₂ : ℝ) : 
  (|x₁ - Real.rpow 2 (1/3)| = 4) → (|x₂ - Real.rpow 2 (1/3)| = 4) → |x₁ - x₂| = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_cube_root_l900_90001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_inequality_l900_90039

theorem max_sum_of_inequality (O sq : ℕ+) : 
  (O : ℚ) / 11 - 7 / (sq : ℚ) < 4 / 5 → O + sq ≤ 393 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_inequality_l900_90039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_theorem_l900_90050

/-- The area of a rhombus with diagonals d1 and d2 is (d1 * d2) / 2 -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem: The area of a rhombus with diagonals d1 = 5x + 10 and d2 = 4y - 8 
    is equal to 10xy - 20x + 20y - 40 -/
theorem rhombus_area_theorem (x y : ℝ) : 
  rhombusArea (5 * x + 10) (4 * y - 8) = 10 * x * y - 20 * x + 20 * y - 40 := by
  -- Unfold the definition of rhombusArea
  unfold rhombusArea
  -- Simplify the expression
  simp [mul_add, add_mul, sub_mul, mul_sub]
  -- Algebraic manipulation
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_theorem_l900_90050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vitya_car_collection_l900_90082

noncomputable def probability_threshold : Real := 0.99

def models_found : Nat := 12

noncomputable def min_additional_offers (p : Real) (m : Nat) : Nat :=
  Nat.ceil (Real.log p / Real.log ((m : Real) / (m + 1 : Real)))

theorem vitya_car_collection (p : Real) (m : Nat) 
  (h1 : p = probability_threshold) 
  (h2 : m = models_found) : 
  min_additional_offers p m = 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vitya_car_collection_l900_90082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_one_l900_90041

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Definition of the function f -/
noncomputable def f : ℝ → ℝ 
| x => if x ≤ 0 then 2 * x^2 - x else -(2 * (-x)^2 - (-x))

theorem f_value_at_one :
  IsOdd f ∧ (∀ x ≤ 0, f x = 2 * x^2 - x) → f 1 = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_one_l900_90041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l900_90023

variable (x y : ℝ)

noncomputable def A : ℝ → ℝ → ℝ := λ x y => (x + y) * (y - 3 * x)
noncomputable def B : ℝ → ℝ → ℝ := λ x y => (x - y)^4 / (x - y)^2

theorem problem_solution :
  2 * y + A x y = B x y - 6 →
  (A x y = y^2 - 2*x*y - 3*x^2) ∧
  (B x y = x^2 - 2*x*y + y^2) ∧
  (y = 2*x^2 - 3) ∧
  ((y + 3)^2 - 2*x*(x*y - 3) - 6*x*(x + 1) = 0) := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l900_90023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_12_divisors_divisible_by_15_l900_90009

def has_exactly_n_divisors (n : ℕ) (k : ℕ) : Prop :=
  (Finset.filter (λ m ↦ n % m = 0) (Finset.range (n + 1))).card = k

theorem smallest_number_with_12_divisors_divisible_by_15 :
  ∃ (n : ℕ), n > 0 ∧ n % 15 = 0 ∧ has_exactly_n_divisors n 12 ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬(m % 15 = 0 ∧ has_exactly_n_divisors m 12) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_12_divisors_divisible_by_15_l900_90009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l900_90048

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  law_of_sines : a / Real.sin A = b / Real.sin B
  side_condition : a = b + 2 * b * Real.cos C

theorem triangle_properties (abc : Triangle) :
  abc.C = 2 * abc.B ∧ 1 < (abc.a + abc.c) / abc.b ∧ (abc.a + abc.c) / abc.b < 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l900_90048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_minimizes_sum_squared_distances_l900_90030

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron with four vertices -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the squared distance between two points -/
noncomputable def squaredDistance (p q : Point3D) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2

/-- Calculates the centroid of a tetrahedron -/
noncomputable def centroid (t : Tetrahedron) : Point3D :=
  { x := (t.A.x + t.B.x + t.C.x + t.D.x) / 4,
    y := (t.A.y + t.B.y + t.C.y + t.D.y) / 4,
    z := (t.A.z + t.B.z + t.C.z + t.D.z) / 4 }

/-- Calculates the sum of squared distances from a point to all vertices of a tetrahedron -/
noncomputable def sumSquaredDistances (t : Tetrahedron) (p : Point3D) : ℝ :=
  squaredDistance p t.A + squaredDistance p t.B + squaredDistance p t.C + squaredDistance p t.D

/-- Theorem: The centroid minimizes the sum of squared distances to all vertices of a tetrahedron -/
theorem centroid_minimizes_sum_squared_distances (t : Tetrahedron) :
  ∀ p : Point3D, sumSquaredDistances t (centroid t) ≤ sumSquaredDistances t p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_minimizes_sum_squared_distances_l900_90030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_T_l900_90079

/-- Geometric sequence with common ratio 2 -/
def geometricSequence (a₁ : ℝ) : ℕ+ → ℝ :=
  fun n => a₁ * (2 ^ (n.val - 1))

/-- Sum of the first n terms of the geometric sequence -/
noncomputable def S (a₁ : ℝ) (n : ℕ+) : ℝ :=
  a₁ * (1 - 2^n.val) / (1 - 2)

/-- Definition of Tn -/
noncomputable def T (a₁ : ℝ) (n : ℕ+) : ℝ :=
  (9 * S a₁ n - S a₁ (2 * n)) / (geometricSequence a₁ (n + 1))

/-- The maximum value of the sequence Tn is 3 -/
theorem max_value_of_T (a₁ : ℝ) (h : a₁ ≠ 0) :
  ∃ (M : ℝ), M = 3 ∧ ∀ (n : ℕ+), T a₁ n ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_T_l900_90079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_n_plus_reciprocal_l900_90049

theorem x_power_n_plus_reciprocal (θ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ) (h2 : θ < π/2) (h3 : x + 1/x = 2 * Real.sin θ) : 
  x^n + (1/x)^n = 2 * Real.cos (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_n_plus_reciprocal_l900_90049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_of_a_l900_90063

/-- Given a space vector a = (2, 2, -1), prove that its unit vector is (2/3, 2/3, -1/3) -/
theorem unit_vector_of_a :
  let a : Fin 3 → ℝ := ![2, 2, -1]
  let norm := Real.sqrt (a 0^2 + a 1^2 + a 2^2)
  (a 0 / norm, a 1 / norm, a 2 / norm) = (2/3, 2/3, -1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_of_a_l900_90063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_distribution_sum_l900_90014

-- Define f(n, k) as the number of ways to distribute k chocolates to n children
-- with each child receiving 0, 1, or 2 chocolates
def f : ℕ → ℕ → ℕ
| n, k => sorry  -- We'll leave the implementation as 'sorry' for now

-- Define the sequence of k values: 1, 4, 7, ..., 4027, 4030
def k_sequence : List ℕ := List.range 1344 |>.map (fun i => 3 * i + 1)

-- State the theorem
theorem chocolate_distribution_sum :
  (k_sequence.map (f 2016)).sum = 3^2015 :=
by
  sorry

-- Auxiliary lemmas

-- The recursive relation for f
axiom f_recursive (k : ℕ) (h : k ≥ 2) :
  f 2016 k = f 2015 k + f 2015 (k-1) + f 2015 (k-2)

-- Each child can receive at most 2 chocolates
axiom f_max_chocolates (n k : ℕ) :
  k > 2 * n → f n k = 0

-- The sum of f(2015, k) for all k is 3^2015
axiom f_2015_sum :
  (List.range 4031 |>.map (f 2015)).sum = 3^2015

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_distribution_sum_l900_90014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_given_equation_l900_90075

noncomputable def area_of_region (equation : ℝ → ℝ → Prop) : ℝ := 19 * Real.pi

theorem area_of_given_equation :
  area_of_region (λ x y => x^2 + y^2 + 6*x - 2*y - 9 = 0) = 19 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_given_equation_l900_90075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freddy_travel_time_l900_90021

/-- Represents the travel scenario between Eddy and Freddy -/
structure TravelScenario where
  distance_AB : ℝ  -- Distance from A to B in km
  distance_AC : ℝ  -- Distance from A to C in km
  time_E : ℝ        -- Eddy's travel time in hours
  speed_ratio : ℝ   -- Ratio of Eddy's speed to Freddy's speed

/-- Calculates Freddy's travel time given a TravelScenario -/
noncomputable def freddy_time (scenario : TravelScenario) : ℝ :=
  (scenario.distance_AC * scenario.time_E) / (scenario.distance_AB * scenario.speed_ratio)

/-- Theorem stating that Freddy's travel time is 4 hours under the given conditions -/
theorem freddy_travel_time :
  ∀ (scenario : TravelScenario),
    scenario.distance_AB = 450 ∧
    scenario.distance_AC = 300 ∧
    scenario.time_E = 3 ∧
    scenario.speed_ratio = 2 →
    freddy_time scenario = 4 := by
  intro scenario h
  simp [freddy_time]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_freddy_travel_time_l900_90021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_l900_90017

/-- Calculates the unoccupied volume in a cube-shaped container -/
theorem unoccupied_volume
  (container_side : ℝ)
  (water_fraction : ℝ)
  (ice_cube_side : ℝ)
  (ice_cube_count : ℕ)
  (h1 : container_side = 12)
  (h2 : water_fraction = 2/3)
  (h3 : ice_cube_side = 3)
  (h4 : ice_cube_count = 8) :
  container_side^3 - (water_fraction * container_side^3 + ↑ice_cube_count * ice_cube_side^3) = 360 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and might cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_l900_90017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l900_90016

-- Define the hyperbola
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2 / b^2 = 1

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define the asymptote of the hyperbola
def asymptote (b : ℝ) (x y : ℝ) : Prop := y = b * x ∨ y = -b * x

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (b : ℝ) : ℝ := Real.sqrt (1 + b^2)

-- The theorem to prove
theorem hyperbola_eccentricity (b : ℝ) (h1 : b > 0) :
  (∃ (x y : ℝ), hyperbola b x y ∧ asymptote b x y ∧ circleEq x y) →
  (∀ (x1 y1 x2 y2 : ℝ), asymptote b x1 y1 ∧ circleEq x1 y1 ∧ asymptote b x2 y2 ∧ circleEq x2 y2 → x1 = x2 ∧ y1 = y2) →
  eccentricity b = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l900_90016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_retailer_discount_l900_90005

/-- Represents the discount percentage calculation for a pen retailer --/
theorem pen_retailer_discount (P : ℝ) (h1 : P > 0) : 
  (let cost_price := 36 * P
   let profit_rate := 0.09999999999999996
   let profit := cost_price * profit_rate
   let total_revenue := cost_price + profit
   let selling_price := total_revenue / 40
   let discount := P - selling_price
   let discount_percentage := (discount / P) * 100
   discount_percentage) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_retailer_discount_l900_90005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_x_values_for_1001_l900_90037

/-- Define the sequence recursively -/
def mySequence (x : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => 1000
  | (n + 2) => mySequence x n * mySequence x (n + 1) - 1

/-- Predicate to check if 1001 appears in the first 5 terms of the sequence -/
def has_1001 (x : ℝ) : Prop :=
  ∃ n : ℕ, n < 5 ∧ mySequence x n = 1001

/-- The main theorem -/
theorem count_x_values_for_1001 :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x, x ∈ s ↔ (x > 0 ∧ has_1001 x) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_x_values_for_1001_l900_90037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_addition_for_divisibility_l900_90059

theorem least_addition_for_divisibility (n a b : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Prime b) : 
  ∃ (k : ℕ), k = (a * b - n % (a * b)) % (a * b) ∧ 
  (n + k) % a = 0 ∧ (n + k) % b = 0 :=
by
  let lcm := a * b
  let remainder := n % lcm
  let k := (lcm - remainder) % lcm
  
  use k
  
  constructor
  · -- Prove that k is the least non-negative number to add
    rfl
  
  constructor
  · -- Prove (n + k) % a = 0
    sorry
  
  -- Prove (n + k) % b = 0
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_addition_for_divisibility_l900_90059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_product_l900_90074

theorem cube_root_product (x : ℝ) (h : x > 0) :
  (108 * x^5)^(1/3) * (27 * x^4)^(1/3) * (8 * x)^(1/3) = 18 * x^3 * (4 * x)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_product_l900_90074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l900_90084

def a : ℝ × ℝ := (1, 3)
def c : ℝ × ℝ := (3, 4)

theorem vector_problem (m : ℝ) : 
  let b : ℝ × ℝ := (m, 2)
  let orthogonal := (a.1 - 3 * b.1) * c.1 + (a.2 - 3 * b.2) * c.2 = 0
  orthogonal →
    (m = -1 ∧ 
     Real.arccos ((a.1 * b.1 + a.2 * b.2) / 
      (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l900_90084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_extra_credit_l900_90062

/-- Represents a class with students and their test scores -/
structure ClassInfo where
  numStudents : ℕ
  scores : Fin numStudents → ℝ

/-- Calculates the median score of a class -/
noncomputable def medianScore (c : ClassInfo) : ℝ :=
  sorry

/-- Counts the number of students whose scores exceed the median -/
def countExceedingMedian (c : ClassInfo) : ℕ :=
  sorry

/-- Theorem stating the maximum number of students who can receive extra credit -/
theorem max_extra_credit (c : ClassInfo) (h : c.numStudents = 120) :
  countExceedingMedian c ≤ 60 ∧
  ∃ (scores : Fin 120 → ℝ), countExceedingMedian {numStudents := 120, scores := scores} = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_extra_credit_l900_90062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_function_l900_90042

noncomputable def f (x y : ℕ+) : ℕ+ := ⟨Nat.lcm x.val y.val, Nat.lcm_pos x.2 y.2⟩

theorem f_unique_function (g : ℕ+ → ℕ+ → ℕ+) :
  (∀ x y : ℕ+, g x y = g y x) →
  (∀ x : ℕ+, g x x = x) →
  (∀ x y : ℕ+, y > x → (y - x) * g x y = y * g x (y - x)) →
  g = f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_function_l900_90042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_x_axis_intersection_l900_90061

/-- Given a circle with diameter endpoints (2,2) and (10,8), 
    the x-coordinate of the second intersection point with the x-axis is 6. -/
theorem circle_x_axis_intersection :
  ∀ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) →
    (2, 2) ∈ C →
    (10, 8) ∈ C →
    center = ((2 + 10) / 2, (2 + 8) / 2) →
    radius = Real.sqrt ((2 - center.1)^2 + (2 - center.2)^2) →
    ∃ (x : ℝ), x ≠ 2 ∧ (x, 0) ∈ C ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_x_axis_intersection_l900_90061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_C_is_8_75_l900_90035

/-- Represents the hiking scenario with two hikers C and D -/
structure HikingScenario where
  initial_distance : ℝ
  landmark_to_meeting : ℝ
  speed_difference : ℝ

/-- Calculates the speed of hiker C given the hiking scenario -/
noncomputable def speed_of_C (scenario : HikingScenario) : ℝ :=
  let d := scenario.initial_distance
  let m := scenario.landmark_to_meeting
  let s := scenario.speed_difference
  ((d + m) * (d - m)) / (2 * d * s) - s / 2

/-- Theorem stating that given the specific scenario, the speed of hiker C is 8.75 mph -/
theorem speed_of_C_is_8_75 :
  let scenario : HikingScenario := {
    initial_distance := 90,
    landmark_to_meeting := 20,
    speed_difference := 5
  }
  speed_of_C scenario = 8.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_C_is_8_75_l900_90035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_condition_l900_90072

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle ABCD -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if two angles are equal -/
def anglesEqual (a b c d e f : Point) : Prop :=
  (distance a b)^2 + (distance a c)^2 - (distance b c)^2 =
  (distance d e)^2 + (distance d f)^2 - (distance e f)^2

theorem rectangle_point_condition (ABCD : Rectangle) (M N : Point) (x : ℝ) :
  distance ABCD.A ABCD.B = 8 →
  distance ABCD.B ABCD.C = 4 →
  M.x = (ABCD.A.x + ABCD.B.x) / 2 →
  M.y = ABCD.A.y →
  N.x = ABCD.C.x →
  N.y = ABCD.C.y + x →
  anglesEqual M N ABCD.D M N ABCD.B →
  x = 4 - 4 * Real.sqrt 2 := by
  sorry

#check rectangle_point_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_condition_l900_90072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentytwo_power_ends_in_two_twentysecond_n_is_85_l900_90046

theorem twentytwo_power_ends_in_two (n : ℕ) : n > 0 → (
  (22^n : ℕ) % 10 = 2 ↔ n % 4 = 1
) := by
  sorry

theorem twentysecond_n_is_85 : 
  (List.filter (λ n : ℕ ↦ n > 0 ∧ (22^n : ℕ) % 10 = 2) (List.range 86)).length = 22 ∧
  (List.filter (λ n : ℕ ↦ n > 0 ∧ (22^n : ℕ) % 10 = 2) (List.range 86)).getLast? = some 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentytwo_power_ends_in_two_twentysecond_n_is_85_l900_90046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_reciprocal_l900_90076

/-- A sequence defined recursively -/
def a : ℕ → ℚ
  | 0 => 1  -- Adding case for 0 to cover all natural numbers
  | 1 => 1
  | (n + 1) => a n / (1 + a n)

/-- The theorem stating that a_n = 1/n for all n ≥ 1 -/
theorem a_eq_reciprocal (n : ℕ) (h : n ≥ 1) : a n = 1 / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_reciprocal_l900_90076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_is_80_l900_90013

/-- Represents the cost, initial price, and sales data for a wine store --/
structure WineStore where
  cost_price : ℚ
  initial_price : ℚ
  initial_sales : ℚ
  price_reduction_effect : ℚ
  target_profit : ℚ

/-- Calculates the selling price that achieves the target profit --/
def calculate_selling_price (store : WineStore) : ℚ :=
  let a := -2
  let b := store.initial_price * store.price_reduction_effect + store.initial_sales - 
           store.price_reduction_effect * store.cost_price
  let c := store.target_profit - store.initial_price * store.initial_sales + 
           store.cost_price * store.initial_sales
  (-b - (b^2 - 4*a*c).sqrt) / (2*a)

/-- Theorem stating that the calculated selling price is 80 for the given conditions --/
theorem selling_price_is_80 (store : WineStore) 
  (h1 : store.cost_price = 60)
  (h2 : store.initial_price = 100)
  (h3 : store.initial_sales = 40)
  (h4 : store.price_reduction_effect = 2)
  (h5 : store.target_profit = 1600) :
  calculate_selling_price store = 80 := by
  sorry

#eval calculate_selling_price { 
  cost_price := 60, 
  initial_price := 100, 
  initial_sales := 40, 
  price_reduction_effect := 2, 
  target_profit := 1600 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_is_80_l900_90013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_MNC_l900_90051

-- Define the square ABCD
noncomputable def square_ABCD : Set (ℝ × ℝ) := sorry

-- Define the side length of the square
def side_length : ℝ := 1

-- Define point B
def B : ℝ × ℝ := (0, 0)

-- Define point C
def C : ℝ × ℝ := (side_length, 0)

-- Define point N
def N : ℝ × ℝ := (side_length, side_length)

-- Define point M as the midpoint of BC
noncomputable def M : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define triangle BCN
noncomputable def triangle_BCN : Set (ℝ × ℝ) := sorry

-- Define triangle MNC
noncomputable def triangle_MNC : Set (ℝ × ℝ) := sorry

-- State that BCN is a right isosceles triangle
axiom BCN_right_isosceles : 
  Real.pi / 2 = sorry ∧ 
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = Real.sqrt ((B.1 - N.1)^2 + (B.2 - N.2)^2) ∧ 
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = side_length

-- Function to calculate the area of a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem stating that the area of triangle MNC is √2/4
theorem area_MNC : 
  triangle_area M N C = Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_MNC_l900_90051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_eq_sqrt45_plus_sqrt8_plus_sqrt41_l900_90000

/-- The total distance traveled from A(-3,6) to C(6,-3) passing through B(0,0) and D(2,2) -/
noncomputable def total_distance : ℝ :=
  let A : ℝ × ℝ := (-3, 6)
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (6, -3)
  let D : ℝ × ℝ := (2, 2)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A B + dist B D + dist D C

/-- Theorem stating that the total distance is equal to √45 + √8 + √41 -/
theorem total_distance_eq_sqrt45_plus_sqrt8_plus_sqrt41 :
  total_distance = Real.sqrt 45 + Real.sqrt 8 + Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_eq_sqrt45_plus_sqrt8_plus_sqrt41_l900_90000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_perimeter_l900_90085

/-- A rectangle subdivided into 9 squares with specific properties -/
structure SpecialRectangle where
  W : ℕ
  H : ℕ
  b : Fin 9 → ℕ
  rel_prime : Nat.Coprime W H
  largest_square : b 8 = W / 2
  sum_rule1 : b 0 + b 1 = b 2
  sum_rule2 : b 0 + b 2 = b 3
  sum_rule3 : b 2 + b 3 = b 4
  sum_rule4 : b 3 + b 4 = b 5
  sum_rule5 : b 1 + b 2 + b 4 = b 6
  sum_rule6 : b 1 + b 6 = b 7
  sum_rule7 : b 0 + b 3 + b 5 = b 8
  sum_rule8 : b 5 + b 8 = b 6 + b 7
  distinct : ∀ i j, i ≠ j → b i ≠ b j

/-- The perimeter of the special rectangle is 266 -/
theorem special_rectangle_perimeter (r : SpecialRectangle) : 
  2 * (r.W + r.H) = 266 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_perimeter_l900_90085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_zero_additions_l900_90070

/-- A multiset of integers representing the numbers on the blackboard -/
def Blackboard := Multiset ℤ

/-- The operation of adding a and b to a pair of equal numbers -/
def add_to_equal (board : Blackboard) (a b : ℤ) : Blackboard :=
  sorry

/-- The operation of adding two zeros to the board -/
def add_zeros (board : Blackboard) : Blackboard :=
  sorry

/-- A sequence of operations on the blackboard -/
inductive Operation (a b : ℤ)
  | AddToEqual : Operation a b
  | AddZeros : Operation a b

/-- Predicate to check if an operation is AddZeros -/
def isAddZeros {a b : ℤ} : Operation a b → Bool
  | Operation.AddZeros => true
  | _ => false

/-- The theorem statement -/
theorem finite_zero_additions 
  (a b : ℤ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a ≠ b) :
  ∃ (N : ℕ), ∀ (ops : List (Operation a b)),
    (ops.filter isAddZeros).length ≤ N :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_zero_additions_l900_90070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_seven_l900_90004

theorem smallest_divisible_by_seven : 
  ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m < n → ¬(7 ∣ 517324 + m * 10)) ∧ 
  (7 ∣ 517324 + n * 10) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_seven_l900_90004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_T_l900_90024

def T (n : ℕ) : ℕ := n * (10^(n.pred)) * 285

theorem smallest_even_T : 
  ∀ k : ℕ, 0 < k ∧ k < 1 → ¬ Even (T k) ∧ Even (T 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_even_T_l900_90024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheapest_plan_l900_90019

/-- Taxi fare function -/
noncomputable def fare (x : ℝ) : ℝ :=
  if x ≤ 3 then 5
  else if x ≤ 10 then 1.2 * x + 1.4
  else 1.8 * x - 4.6

/-- Cost of n equal segments of length l -/
noncomputable def segmentCost (n : ℕ) (l : ℝ) : ℝ := n * fare l

theorem cheapest_plan :
  let plan1 := segmentCost 1 30
  let plan2 := segmentCost 2 15
  let plan3 := segmentCost 3 10
  plan3 < plan2 ∧ plan3 < plan1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheapest_plan_l900_90019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l900_90077

-- Define the function representing the left side of the equation
def f (x : ℝ) : ℝ := (3 - x)^(1/4) + (x + 2)^(1/2)

-- State the theorem
theorem equation_solutions :
  ∃ x₁ x₂ : ℝ, 
    (x₁ ≠ x₂) ∧ 
    (f x₁ = 2) ∧ 
    (f x₂ = 2) ∧ 
    (x₁ = 2 ∨ |x₁ - 0.990| < 0.001) ∧
    (x₂ = 2 ∨ |x₂ - 0.990| < 0.001) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l900_90077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_4_l900_90020

/-- A function f satisfying the given conditions -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ) - 3

/-- The theorem stating the properties of f and its value at π/4 -/
theorem f_value_at_pi_over_4 (ω φ : ℝ) (h_ω : ω > 0) 
  (h_symmetry : ∀ x, f ω φ (x + π/6) = f ω φ (π/3 - x)) :
  f ω φ (π/4) = -5 ∨ f ω φ (π/4) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_4_l900_90020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l900_90090

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + 1/y^2) * (1/x^2 + 4*y^2) ≥ 9 ∧
  ((x^2 + 1/y^2) * (1/x^2 + 4*y^2) = 9 ↔ x*y = 1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l900_90090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_l900_90060

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation (x : ℝ) : 
  (x ∈ ({2, 3, 8, 12} : Set ℝ)) →
  (can_form_triangle 4 7 x ↔ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_l900_90060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_bisector_slope_l900_90038

/-- A rectangle in a 2D plane -/
structure Rectangle where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the slope of a line -/
noncomputable def slopeLine (l : Line) : ℝ :=
  (l.point2.2 - l.point1.2) / (l.point2.1 - l.point1.1)

/-- Check if a line passes through the origin -/
def passesOrigin (l : Line) : Prop :=
  l.point1 = (0, 0) ∨ l.point2 = (0, 0)

/-- Calculate the center of a rectangle -/
noncomputable def center (r : Rectangle) : ℝ × ℝ :=
  ((r.topRight.1 + r.bottomLeft.1) / 2, (r.topRight.2 + r.bottomLeft.2) / 2)

/-- Check if a line passes through the center of a rectangle -/
def passesCenter (l : Line) (r : Rectangle) : Prop :=
  l.point1 = center r ∨ l.point2 = center r

theorem rectangle_bisector_slope :
  ∀ (r : Rectangle) (l : Line),
    r.bottomLeft = (1, 0) →
    r.topRight = (9, 2) →
    passesOrigin l →
    passesCenter l r →
    slopeLine l = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_bisector_slope_l900_90038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_theorem_l900_90029

/-- Represents the daily changes in tonnage over 6 days -/
def daily_changes : List Int := [31, -32, -16, 35, -38, -20]

/-- The final tonnage after 6 days -/
def final_tonnage : Int := 460

/-- The loading and unloading fee per ton -/
def fee_per_ton : Int := 5

/-- Theorem stating the initial tonnage and total fees -/
theorem warehouse_theorem :
  let initial_tonnage := final_tonnage - daily_changes.sum
  let total_fees := (daily_changes.map abs).sum * fee_per_ton
  initial_tonnage = 500 ∧ total_fees = 860 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_theorem_l900_90029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shifted_correct_l900_90018

-- Define the original function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-4) 0 then -x - 2
  else if x ∈ Set.Icc 0 2 then Real.sqrt (4 - (x - 1)^2) - 2
  else if x ∈ Set.Icc 2 4 then 2 * (x - 1)
  else 0  -- Define a default value for x outside the given intervals

-- Define the shifted function g(x + 3)
noncomputable def g_shifted (x : ℝ) : ℝ := g (x + 3)

-- State the theorem
theorem g_shifted_correct :
  ∀ x : ℝ, 
    (x ∈ Set.Icc (-7) (-3) → g_shifted x = -x - 5) ∧
    (x ∈ Set.Icc (-3) (-1) → g_shifted x = Real.sqrt (4 - (x + 2)^2) - 2) ∧
    (x ∈ Set.Icc (-1) 1 → g_shifted x = 2 * (x + 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shifted_correct_l900_90018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l900_90003

def M : Set ℝ := {x | x^2 - 4*x < 0}
def N : Set ℝ := {x | |x| ≤ 2}

theorem union_of_M_and_N : M ∪ N = Set.Icc (-2) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l900_90003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l900_90026

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 1/3
  sum_condition : a 2 + a 5 = 4

/-- The theorem to be proved -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) :
  (∃ n : ℕ, seq.a n = 33) → ∃ n : ℕ, seq.a n = 33 ∧ n = 50 :=
by
  intro h
  -- The proof goes here
  sorry

#check arithmetic_sequence_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l900_90026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_bound_l900_90056

/-- Represents a triangle with side lengths a, b, c where c ≤ b ≤ a -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : 0 < c
  h2 : c ≤ b
  h3 : b ≤ a

/-- Represents a point inside a triangle -/
structure PointInTriangle (t : Triangle) where
  x : ℝ
  y : ℝ
  inside : x > 0 ∧ y > 0 ∧ x + y < 1 -- Simple representation of a point inside a triangle

/-- The sum of distances from a point to each side of the triangle -/
def sumOfDistances (t : Triangle) (p : PointInTriangle t) : ℝ :=
  p.x + p.y + (1 - p.x - p.y) -- Simplified representation of distances

theorem sum_of_distances_bound (t : Triangle) (p : PointInTriangle t) :
  sumOfDistances t p < 2 * t.a + t.b := by
  sorry -- Proof is omitted

#check sum_of_distances_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_bound_l900_90056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_positive_l900_90064

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2-a)*x - a * Real.log x

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*x + (2-a) - a/x

theorem tangent_slope_positive (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : f a x₁ = 0) (h₂ : f a x₂ = 0) (h₃ : x₁ > 0) (h₄ : x₂ > 0) (h₅ : x₁ ≠ x₂) :
  f_deriv a ((x₁ + x₂) / 2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_positive_l900_90064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_trig_expression_l900_90040

theorem third_quadrant_trig_expression (α : Real) :
  (α > π ∧ α < 3*π/2) →
  (Real.cos α / Real.sqrt (1 - Real.sin α ^ 2) + Real.sin α / Real.sqrt (1 - Real.cos α ^ 2)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_trig_expression_l900_90040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l900_90099

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
noncomputable def given_triangle : Triangle where
  a := 2
  c := Real.sqrt 2
  A := Real.arccos (-Real.sqrt 2 / 4)
  b := 1  -- We know this from the solution, so we can include it
  B := sorry
  C := sorry

-- State the theorem
theorem triangle_properties (t : Triangle) (h1 : t = given_triangle) :
  Real.sin t.C = Real.sqrt 7 / 4 ∧
  t.b = 1 ∧
  Real.cos (2 * t.A + π / 3) = (-3 + Real.sqrt 21) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l900_90099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l900_90067

noncomputable def f (x : ℝ) : ℝ := (2^x) / (2^x + 1)

theorem f_properties :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x : ℝ, f x + f (-x) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l900_90067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_l900_90080

def G : ℕ → ℚ
  | 0 => 3  -- Add this case to handle Nat.zero
  | 1 => 3
  | (n + 1) => (3 * G n + 1) / 3

theorem G_51 : G 51 = 59 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_l900_90080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_one_half_l900_90044

theorem inverse_one_half : (1 / 2 : ℝ)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_one_half_l900_90044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l900_90034

-- Define necessary structures and functions
structure Point := (x y : ℝ)

def TriangleXYZ (X Y Z : Point) : Prop := sorry
def AltitudeXH (X Y Z H : Point) : Prop := sorry
def SegmentLength (A B : Point) : ℝ := sorry
def RectangleABCD (A B C D : Point) : Prop := sorry
def InscribedRectangle (X Y Z A B C D : Point) : Prop := sorry
def OnLine (A D Y Z : Point) : Prop := sorry
def AreaRectangle (A B C D : Point) : ℝ := sorry

theorem inscribed_rectangle_area (X Y Z A B C D H : Point) :
  TriangleXYZ X Y Z →
  AltitudeXH X Y Z H →
  SegmentLength H X = 9 →
  SegmentLength Y Z = 15 →
  RectangleABCD A B C D →
  InscribedRectangle X Y Z A B C D →
  OnLine A D Y Z →
  SegmentLength A B = (1/3) * SegmentLength A D →
  AreaRectangle A B C D = 675/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l900_90034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l900_90097

theorem abc_inequality : 
  let a : ℝ := Real.log (1/2)
  let b : ℝ := (1/3) ^ (0.8 : ℝ)
  let c : ℝ := 2 ^ (1/3 : ℝ)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l900_90097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_chair_problem_l900_90047

-- Define the problem parameters
def total_budget : ℕ := 12000
def initial_chairs : ℕ := 1093
def total_classrooms : ℕ := 35
def large_classrooms : ℕ := 20
def small_classrooms : ℕ := 15
def large_capacity : ℕ := 40
def small_capacity : ℕ := 30

-- Define the theorem
theorem school_chair_problem :
  let total_capacity := large_classrooms * large_capacity + small_classrooms * small_capacity
  let additional_chairs := total_capacity - initial_chairs
  let cost_per_chair : ℚ := (total_budget : ℚ) / (initial_chairs : ℚ)
  additional_chairs = 157 ∧ 
  (cost_per_chair ≥ 10.97 ∧ cost_per_chair ≤ 10.99) := by
  sorry

-- Note: We use an inequality to represent the approximate equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_chair_problem_l900_90047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrafluoroethylene_polymerization_l900_90091

/-- Represents a chemical compound -/
structure Compound where
  formula : String
  name : String

/-- Represents a polymerization process -/
def polymerize (monomer : Compound) : Compound :=
  sorry -- Placeholder implementation

/-- Tetrafluoroethylene compound -/
def tetrafluoroethylene : Compound :=
  { formula := "CF2=CF2", name := "Tetrafluoroethylene" }

/-- Teflon compound -/
def teflon : Compound :=
  { formula := "(-CF2-CF2-)n", name := "Teflon" }

/-- Theorem stating that polymerization of tetrafluoroethylene results in Teflon -/
theorem tetrafluoroethylene_polymerization :
  polymerize tetrafluoroethylene = teflon := by
  sorry

#check tetrafluoroethylene_polymerization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrafluoroethylene_polymerization_l900_90091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_apartment_number_l900_90086

def phone_number : List Nat := [8, 6, 5, 3, 4, 2, 1]

def sum_digits (n : Nat) : Nat :=
  (n.digits 10).sum

def is_valid_apartment (n : Nat) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  sum_digits n = phone_number.sum ∧
  (n.digits 10).toFinset.card = 4

theorem largest_apartment_number :
  ∀ n : Nat, is_valid_apartment n → n ≤ 9875 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_apartment_number_l900_90086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_arithmetic_progression_l900_90071

/-- A sequence (a_n) where the sum of its first n terms is given by S_n = a n^2 + b n -/
def SumSequence (a b : ℝ) : ℕ → ℝ := λ n ↦ a * n^2 + b * n

/-- The n-th term of the sequence -/
def NthTerm (a b : ℝ) : ℕ → ℝ := λ n ↦ 
  SumSequence a b n - SumSequence a b (n-1)

/-- Definition of arithmetic progression -/
def IsArithmeticProgression (seq : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), seq (n+1) - seq n = d

theorem sequence_is_arithmetic_progression (a b : ℝ) :
  IsArithmeticProgression (NthTerm a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_is_arithmetic_progression_l900_90071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_m_1_monotonicity_min_integer_m_l900_90088

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x^2 + (1 - 2*m) * x + 1

-- Theorem for the maximum value when m = 1
theorem max_value_when_m_1 :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 1 y ≤ f 1 x ∧ f 1 x = 1/4 - Real.log 2 := by
  sorry

-- Theorem for monotonicity
theorem monotonicity (m : ℝ) :
  (m ≤ 0 → ∀ (x y : ℝ), 0 < x ∧ x < y → f m x < f m y) ∧
  (m > 0 → ∃ (z : ℝ), z > 0 ∧ 
    (∀ (x y : ℝ), 0 < x ∧ x < y ∧ y < z → f m x < f m y) ∧
    (∀ (x y : ℝ), z < x ∧ x < y → f m x > f m y)) := by
  sorry

-- Theorem for minimum integer m
theorem min_integer_m :
  ∀ (m : ℤ), (∀ (x : ℝ), x > 0 → f (↑m) x ≤ 0) → m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_m_1_monotonicity_min_integer_m_l900_90088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_35_74_l900_90031

/-- Represents a number in base 8 --/
structure OctalNumber where
  value : ℕ

/-- Converts an OctalNumber to its decimal (ℤ) representation --/
def octal_to_decimal (n : OctalNumber) : ℤ := sorry

/-- Converts a decimal (ℤ) number to its octal representation --/
def decimal_to_octal (n : ℤ) : OctalNumber := sorry

/-- Subtracts two OctalNumbers and returns the result as an OctalNumber --/
def octal_subtract (a b : OctalNumber) : OctalNumber :=
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

/-- Helper function to create an OctalNumber from a natural number --/
def mk_octal (n : ℕ) : OctalNumber := ⟨n⟩

theorem octal_subtraction_35_74 :
  octal_subtract (mk_octal 35) (mk_octal 74) = decimal_to_octal (-37) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_35_74_l900_90031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_minimum_l900_90032

/-- The circle equation -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 + 8*x + 2*y + 1 = 0

/-- The line equation -/
def lineEq (a b x y : ℝ) : Prop := a*x + b*y + 1 = 0

/-- The symmetry condition -/
def symmetric (a b : ℝ) : Prop := 4*a + b = 1

theorem circle_symmetry_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_sym : symmetric a b) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → symmetric a' b' → 1/a + 4/b ≤ 1/a' + 4/b') ∧ 
  (1/a + 4/b = 16) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_minimum_l900_90032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_l900_90065

def plane_equation (x y z : ℝ) : Prop := 4*x - 5*y - z - 7 = 0

def is_symmetric (M M' : ℝ × ℝ × ℝ) (plane : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (t : ℝ),
    let M₀ := (
      M.1 + t * 4,
      M.2.1 - t * 5,
      M.2.2 - t
    )
    plane M₀.1 M₀.2.1 M₀.2.2 ∧
    M'.1 = 2 * M₀.1 - M.1 ∧
    M'.2.1 = 2 * M₀.2.1 - M.2.1 ∧
    M'.2.2 = 2 * M₀.2.2 - M.2.2

theorem symmetric_point :
  is_symmetric (-1, (2, 0)) (3, (-3, -1)) plane_equation := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_l900_90065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l900_90027

-- Define the parametric equations of the line
noncomputable def line_param (t : ℝ) : ℝ × ℝ := (2 * t, 2 * Real.sqrt 3 * t + 1/2)

-- Define the polar equation of circle C
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the Cartesian equation of circle C1
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the general equation of line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1/2 = 0

-- Define the equation of curve C2
def curve_C2 (x y : ℝ) : Prop := x^2 + (y - 1/2)^2 = 1/4

-- Define points M and N
noncomputable def point_M : ℝ × ℝ := (1/4, Real.sqrt 3 / 4 + 1/2)
noncomputable def point_N : ℝ × ℝ := (-1/4, -Real.sqrt 3 / 4 + 1/2)

-- Theorem statement
theorem intersection_points :
  ∀ x y : ℝ, curve_C2 x y ∧ line_l x y → (x, y) = point_M ∨ (x, y) = point_N :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l900_90027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y1_gt_y3_gt_y2_l900_90055

-- Define the values
noncomputable def y1 : ℝ := 2^(1.8 : ℝ)
noncomputable def y2 : ℝ := 8^(0.48 : ℝ)
noncomputable def y3 : ℝ := (1/2)^(-(1.5 : ℝ))

-- State the theorem
theorem y1_gt_y3_gt_y2 : y1 > y3 ∧ y3 > y2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y1_gt_y3_gt_y2_l900_90055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_excluding_stoppages_l900_90008

/-- The speed of a train excluding stoppages, given its speed including stoppages and stop time per hour. -/
theorem train_speed_excluding_stoppages 
  (speed_with_stops : ℝ) 
  (stop_time : ℝ) 
  (h1 : speed_with_stops = 36) 
  (h2 : stop_time = 15) : 
  speed_with_stops * 60 / (60 - stop_time) = 48 := by
  sorry

#check train_speed_excluding_stoppages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_excluding_stoppages_l900_90008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_volume_l900_90022

/-- The volume of a specific solid with a square base --/
theorem solid_volume (s : ℝ) (h : s = 6 * Real.sqrt 2) : 
  (1 / 3) * s^2 * (Real.sqrt ((s / 2)^2 + s^2)) = 288 :=
by
  -- Replace this with the actual proof steps
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_volume_l900_90022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l900_90052

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := a * x + b / x + c

-- State the theorem
theorem function_properties (a b c : ℝ) :
  (∀ x, f a b c x = -f a b c (-x)) →  -- f is odd
  f a b c 1 = b + c + a →             -- f(1) condition
  f a b c 2 = 2 * a + b / 2 + c →     -- f(2) condition
  (a = 2 ∧ b = -4 ∧ c = 0) ∧          -- Part 1: values of a, b, c
  (∀ x > 0, f 2 (-4) 0 x ≥ 2)         -- Part 2: minimum value on (0, +∞)
  := by
    -- The proof is omitted for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l900_90052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_after_21_rounds_l900_90089

/-- Represents the state of the game with token counts for three players -/
structure GameState where
  playerA : Nat
  playerB : Nat
  playerC : Nat

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  let max := max state.playerA (max state.playerB state.playerC)
  if max = state.playerA then
    { playerA := state.playerA - 6, playerB := state.playerB + 2, playerC := state.playerC + 2 }
  else if max = state.playerB then
    { playerA := state.playerA + 2, playerB := state.playerB - 6, playerC := state.playerC + 2 }
  else
    { playerA := state.playerA + 2, playerB := state.playerB + 2, playerC := state.playerC - 6 }

/-- Checks if the game has ended (any player has 0 tokens) -/
def isGameOver (state : GameState) : Bool :=
  state.playerA = 0 || state.playerB = 0 || state.playerC = 0

/-- Simulates n rounds of the game -/
def playNRounds (n : Nat) (state : GameState) : GameState :=
  match n with
  | 0 => state
  | n + 1 => playRound (playNRounds n state)

/-- Theorem stating that the game ends after exactly 21 rounds -/
theorem game_ends_after_21_rounds :
  let initialState := { playerA := 24, playerB := 21, playerC := 20 : GameState }
  let finalState := playNRounds 21 initialState
  isGameOver finalState ∧ ¬isGameOver (playNRounds 20 initialState) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_after_21_rounds_l900_90089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_with_more_than_three_factors_of_2550_l900_90010

/-- The number of positive integer factors of 2550 that have more than 3 factors -/
def factors_with_more_than_three_factors : ℕ :=
  (Finset.filter (fun d => (Nat.divisors d).card > 3) (Nat.divisors 2550)).card

/-- Theorem stating that the number of positive integer factors of 2550 
    that have more than 3 factors is equal to 8 -/
theorem factors_with_more_than_three_factors_of_2550 :
  factors_with_more_than_three_factors = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_with_more_than_three_factors_of_2550_l900_90010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_cosine_function_l900_90012

/-- Given a function f(x) = 2cos(2x - π/4), prove that its range on the interval [-π/8, π/2] is [-√2, 2] -/
theorem range_of_cosine_function (f : ℝ → ℝ) (h : ∀ x, f x = 2 * Real.cos (2 * x - π / 4)) :
  Set.range (fun x => f x) ∩ Set.Icc (-π / 8) (π / 2) = Set.Icc (-Real.sqrt 2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_cosine_function_l900_90012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l900_90081

-- Define set A
def A : Set ℝ := {x : ℝ | |x| < 3}

-- Define set B
def B : Set ℝ := {x : ℝ | (2 : ℝ)^x > 1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l900_90081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_properties_l900_90028

def is_in_set_A (f : ℕ → ℕ) : Prop :=
  (∀ k ∈ Finset.range 2007, (f^[k]) ≠ id) ∧
  (f^[2008] = id)

theorem set_A_properties :
  ∃ (A : Set (ℕ → ℕ)),
    (∀ f ∈ A, is_in_set_A f) ∧
    (∃ f, f ∈ A) ∧
    (Set.Infinite A) ∧
    (∀ f ∈ A, Function.Bijective f) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_properties_l900_90028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_box_radius_l900_90054

/-- A rectangular box inscribed in a right circular cylinder -/
structure InscribedBox where
  x : ℝ  -- length
  y : ℝ  -- width
  z : ℝ  -- height (equal to cylinder height)
  surface_area : ℝ
  edge_sum : ℝ

/-- The radius of the cylinder containing the inscribed box -/
noncomputable def cylinder_radius (box : InscribedBox) : ℝ :=
  (1 / 2) * Real.sqrt (box.x^2 + box.y^2)

theorem inscribed_box_radius 
  (box : InscribedBox) 
  (h1 : box.surface_area = 600)
  (h2 : box.edge_sum = 160)
  (h3 : 2 * (box.x * box.y + box.y * box.z + box.x * box.z) = box.surface_area)
  (h4 : 4 * (box.x + box.y + box.z) = box.edge_sum) :
  cylinder_radius box = 15 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_box_radius_l900_90054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_is_correct_l900_90083

/-- The smallest positive integer b for which x^2 + bx + 4032 factors into two integer binomials -/
def smallest_b : ℕ := 127

/-- A function that checks if a given b allows x^2 + bx + 4032 to factor into two integer binomials -/
def has_integer_factorization (b : ℕ) : Prop :=
  ∃ (r s : ℤ), (r * s = 4032) ∧ (r + s = b) ∧
  (∀ x : ℤ, x^2 + b*x + 4032 = (x + r) * (x + s))

theorem smallest_b_is_correct :
  (has_integer_factorization smallest_b) ∧
  (∀ b : ℕ, b < smallest_b → ¬(has_integer_factorization b)) :=
by sorry

#check smallest_b_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_is_correct_l900_90083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_pentagon_and_triangle_angles_l900_90069

noncomputable def interior_angle (n : ℕ) : ℝ := 180 * (n - 2 : ℝ) / n

theorem sum_of_pentagon_and_triangle_angles :
  interior_angle 5 + interior_angle 3 = 168 := by
  -- Unfold the definition of interior_angle
  unfold interior_angle
  -- Simplify the arithmetic
  simp [mul_div_assoc, mul_comm, mul_assoc]
  -- The proof is completed by normalization of real numbers
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_pentagon_and_triangle_angles_l900_90069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_is_integer_iff_n_is_zero_or_144_l900_90033

noncomputable def expression (n : ℤ) : ℝ :=
  Real.sqrt ((25 / 2) + Real.sqrt ((625 / 4) - n)) + Real.sqrt ((25 / 2) - Real.sqrt ((625 / 4) - n))

theorem expression_is_integer_iff_n_is_zero_or_144 :
  ∀ n : ℤ, (∃ k : ℤ, expression n = k) ↔ (n = 0 ∨ n = 144) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_is_integer_iff_n_is_zero_or_144_l900_90033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_l900_90094

/-- The distance to school in miles -/
noncomputable def distance : ℝ := sorry

/-- The usual speed during rush hour in miles per hour -/
noncomputable def usual_speed : ℝ := sorry

/-- Time taken during rush hour in hours -/
noncomputable def rush_hour_time : ℝ := 20 / 60

/-- Time taken with no traffic in hours -/
noncomputable def no_traffic_time : ℝ := 15 / 60

/-- Speed increase on the no-traffic day in miles per hour -/
def speed_increase : ℝ := 20

theorem distance_to_school :
  (distance = usual_speed * rush_hour_time) ∧
  (distance = (usual_speed + speed_increase) * no_traffic_time) →
  distance = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_l900_90094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_wash_time_l900_90073

/-- The time it takes to wash a car -/
def W : ℕ := sorry

/-- The time it takes to change oil on a car -/
def oil_change_time : ℕ := 15

/-- The time it takes to change a set of tires -/
def tire_change_time : ℕ := 30

/-- The number of cars Mike washed -/
def cars_washed : ℕ := 9

/-- The number of cars Mike changed oil on -/
def oil_changes : ℕ := 6

/-- The number of sets of tires Mike changed -/
def tire_changes : ℕ := 2

/-- The total time Mike worked in minutes -/
def total_work_time : ℕ := 4 * 60

theorem car_wash_time : W = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_wash_time_l900_90073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_extrema_l900_90011

noncomputable def f (x : ℝ) := x^3 - 12*x + 2

theorem tangent_and_extrema :
  (∃ (a b : ℝ), f = λ x ↦ a*x^3 + b*x + 2) ∧
  (f 2 = -14) ∧
  (∀ x, x ∈ Set.Icc (-3 : ℝ) 3 → f x ≥ -14) ∧
  (∀ x, x ∈ Set.Icc (-3 : ℝ) 3 → f x ≤ 18) ∧
  (f (-2) = 18) ∧
  (∀ x y, y = -(9*x) ↔ (x - 1) * (deriv f 1) + f 1 = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_extrema_l900_90011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l900_90053

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

theorem g_is_odd : ∀ x, g (-x) = -g x := by
  intro x
  simp [g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l900_90053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_l900_90043

noncomputable section

structure RectPrism where
  l : ℝ
  w : ℝ
  h : ℝ

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def center_base (p : RectPrism) : Point3D :=
  { x := p.l / 2, y := p.w / 2, z := 0 }

noncomputable def midpoint_DD1 (p : RectPrism) : Point3D :=
  { x := p.l / 2, y := p.w / 2, z := p.h / 2 }

noncomputable def point_on_CC1 (p : RectPrism) (y z : ℝ) : Point3D :=
  { x := p.w / 2, y := y, z := z }

def is_parallel (v1 v2 v3 : Point3D) : Prop :=
  (v1.x * (v2.y * v3.z - v2.z * v3.y) +
   v1.y * (v2.z * v3.x - v2.x * v3.z) +
   v1.z * (v2.x * v3.y - v2.y * v3.x)) = 0

theorem parallel_planes (p : RectPrism) (y z : ℝ) :
  let O := center_base p
  let P := midpoint_DD1 p
  let Q := point_on_CC1 p y z
  let D1 := Point3D.mk 0 0 p.h
  let B := Point3D.mk p.l 0 0
  let A := Point3D.mk (-p.l) 0 0
  is_parallel (Point3D.mk (D1.x - B.x) (D1.y - B.y) (D1.z - B.z))
              (Point3D.mk (Q.x - B.x) (Q.y - B.y) (Q.z - B.z))
              (Point3D.mk (A.x - P.x) (A.y - P.y) (A.z - P.z)) ↔
  y = (z * p.w - p.w * p.h / 2 + p.h * p.w / 2) / (5 * p.h) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_l900_90043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l900_90057

theorem triangle_angle_relation (a b c : ℝ) (α β γ : Real)
  (ha : a = 20) (hb : b = 15) (hc : c = 7)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b)
  (h_angles : α + β + γ = Real.pi)
  (h_cos_law_a : Real.cos α = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_cos_law_b : Real.cos β = (a^2 + c^2 - b^2) / (2 * a * c))
  (h_cos_law_c : Real.cos γ = (a^2 + b^2 - c^2) / (2 * a * b)) :
  α = 3 * β + γ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l900_90057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_not_always_equal_probability_l900_90066

-- Define an event
def Event : Type := Unit

-- Define a probability function
def probability (e : Event) : ℝ := sorry

-- Define a frequency function
def frequency (e : Event) (n : ℕ) : ℝ := sorry

-- Theorem stating that frequency is not always equal to probability
theorem frequency_not_always_equal_probability :
  ∃ (e : Event) (n : ℕ), frequency e n ≠ probability e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_not_always_equal_probability_l900_90066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_heads_probability_l900_90093

/-- The probability of getting exactly k successes in n trials with probability p for each trial. -/
noncomputable def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The number of coin tosses -/
def numTosses : ℕ := 3

/-- The number of heads we want to get -/
def numHeads : ℕ := 2

/-- The probability of getting heads on a single toss of a fair coin -/
def probHeads : ℚ := 1/2

theorem exactly_two_heads_probability :
  binomialProbability numTosses numHeads (probHeads : ℝ) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_heads_probability_l900_90093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l900_90006

/-- A vector in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The line y = 3x + 6 -/
def onLine (v : Vector2D) : Prop :=
  v.y = 3 * v.x + 6

/-- Projection of v onto u -/
noncomputable def proj (v u : Vector2D) : Vector2D :=
  let dot := v.x * u.x + v.y * u.y
  let norm_sq := u.x^2 + u.y^2
  { x := (dot / norm_sq) * u.x
  , y := (dot / norm_sq) * u.y }

/-- The theorem to be proved -/
theorem projection_theorem :
  ∃ u : Vector2D, ∀ v : Vector2D, onLine v →
    proj v u = { x := -9/5, y := 3/5 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l900_90006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_time_l900_90007

/-- Represents the time taken to fill a pool with given inlet and outlet configurations -/
structure PoolFilling where
  /-- Time to fill with outlet and first two inlets open -/
  time_two_inlets_1 : ℝ
  /-- Time to fill with outlet and second two inlets open -/
  time_two_inlets_2 : ℝ
  /-- Time to fill with outlet and last two inlets open -/
  time_two_inlets_3 : ℝ
  /-- Time to fill with outlet and all three inlets open -/
  time_all_inlets : ℝ
  /-- Ensures all times are positive -/
  all_positive : time_two_inlets_1 > 0 ∧ time_two_inlets_2 > 0 ∧ time_two_inlets_3 > 0 ∧ time_all_inlets > 0

/-- Calculates the time to fill the pool with all inlets open and outlet closed -/
noncomputable def timeToFillAllInlets (p : PoolFilling) : ℝ :=
  60 / 23

/-- Theorem stating that given the conditions, the time to fill with all inlets open and outlet closed is 2 14/23 hours -/
theorem pool_filling_time (p : PoolFilling) 
    (h1 : p.time_two_inlets_1 = 6)
    (h2 : p.time_two_inlets_2 = 5)
    (h3 : p.time_two_inlets_3 = 4)
    (h4 : p.time_all_inlets = 3) :
    timeToFillAllInlets p = 2 + 14 / 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_time_l900_90007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freds_walking_speed_l900_90045

/-- Proof that Fred's walking speed is 5 miles per hour -/
theorem freds_walking_speed :
  ∀ (total_distance : ℝ) (sams_speed : ℝ) (sams_distance : ℝ) (freds_speed : ℝ),
  total_distance = 50 →
  sams_speed = 5 →
  sams_distance = 25 →
  sams_distance = total_distance / 2 →
  sams_distance / sams_speed = (total_distance - sams_distance) / freds_speed →
  freds_speed = 5 := by
  intros total_distance sams_speed sams_distance freds_speed h1 h2 h3 h4 h5
  -- The proof steps would go here
  sorry

#check freds_walking_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freds_walking_speed_l900_90045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_ratio_points_exist_l900_90096

/-- A line in a plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if a point is on a line -/
def on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  ∃ t : ℝ, p = (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Ratio of distances from a point to two fixed points -/
noncomputable def distance_ratio (p a b : ℝ × ℝ) : ℝ :=
  distance p a / distance p b

/-- Set of points on a line -/
def Line.toSet (l : Line) : Set (ℝ × ℝ) :=
  {p | on_line p l}

theorem max_min_ratio_points_exist (e : Line) (a b : ℝ × ℝ) :
  ∃ p1 p2 : ℝ × ℝ,
    on_line p1 e ∧ on_line p2 e ∧
    (∀ p : ℝ × ℝ, on_line p e → distance_ratio p a b ≤ distance_ratio p2 a b) ∧
    (∀ p : ℝ × ℝ, on_line p e → distance_ratio p a b ≥ distance_ratio p1 a b) ∧
    (∃ k k' : Set (ℝ × ℝ), 
      (∀ x : ℝ × ℝ, x ∈ k → distance_ratio x a b = distance_ratio p1 a b) ∧
      (∀ x : ℝ × ℝ, x ∈ k' → distance_ratio x a b = distance_ratio p2 a b) ∧
      (p1 ∈ k ∨ (p1 ∈ e.toSet ∩ k)) ∧
      (p2 ∈ k' ∨ (p2 ∈ e.toSet ∩ k'))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_ratio_points_exist_l900_90096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_turns_in_two_hours_l900_90036

/-- Represents a wheel in the machine system -/
structure Wheel where
  radius : ℚ
  turnsPerCycle : ℚ

/-- Calculates the number of turns a wheel makes in a given time -/
def calculateTurns (w : Wheel) (time : ℚ) : ℚ :=
  (time / 30) * 6 * w.turnsPerCycle

/-- The machine system with three interconnected wheels -/
structure MachineSystem where
  wheelA : Wheel
  wheelB : Wheel
  wheelC : Wheel

/-- Theorem stating the number of turns each wheel makes in two hours -/
theorem wheel_turns_in_two_hours (m : MachineSystem) 
  (h1 : m.wheelA.radius = 4)
  (h2 : m.wheelB.radius = 2)
  (h3 : m.wheelC.radius = 3)
  (h4 : m.wheelA.turnsPerCycle = 1)
  (h5 : m.wheelB.turnsPerCycle = 2)
  (h6 : m.wheelC.turnsPerCycle = 3) :
  let twoHours : ℚ := 2 * 60 * 60
  (calculateTurns m.wheelA twoHours = 240) ∧
  (calculateTurns m.wheelB twoHours = 480) ∧
  (calculateTurns m.wheelC twoHours = 720) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_turns_in_two_hours_l900_90036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l900_90058

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := 
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Sum of distances AC + BC -/
noncomputable def sumDistances (k : ℝ) : ℝ :=
  let a : Point := ⟨7, 7⟩
  let b : Point := ⟨4, 1⟩
  let c : Point := ⟨k, 0⟩
  distance a c + distance b c

/-- The value of k that minimizes the sum of distances is 11/2 -/
theorem min_sum_distances : 
  ∃ (k : ℝ), k = 11/2 ∧ ∀ (x : ℝ), sumDistances k ≤ sumDistances x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l900_90058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_constant_property_l900_90087

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

/-- Definition of point Q -/
noncomputable def Q : ℝ × ℝ := (6 * Real.sqrt 5 / 5, 0)

/-- Distance from a point to a line -/
noncomputable def distanceToLine (x y : ℝ) : ℝ :=
  abs (x - y + 3 * Real.sqrt 2) / Real.sqrt 2

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem stating the constant value property for chords passing through Q -/
theorem chord_constant_property :
  ∃ (a b : ℝ),
    a > b ∧ b > 0 ∧
    (∃ (c : ℝ), c > 0 ∧ distanceToLine c 0 = 5) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      ellipse x1 y1 ∧ ellipse x2 y2 ∧
      distance x1 y1 x2 y2 = Real.sqrt 10) →
    ∀ (A B : ℝ × ℝ),
      ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
      (∃ (t : ℝ), 0 < t ∧ t < 1 ∧
        Q.1 = t * A.1 + (1 - t) * B.1 ∧
        Q.2 = t * A.2 + (1 - t) * B.2) →
      1 / (distance Q.1 Q.2 A.1 A.2)^2 +
      1 / (distance Q.1 Q.2 B.1 B.2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_constant_property_l900_90087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_at_least_twice_min_distance_l900_90025

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Theorem statement
theorem max_distance_at_least_twice_min_distance (points : Fin 10 → Point) :
  ∃ (i j k l : Fin 10), i ≠ j ∧ k ≠ l ∧ 
    distance (points i) (points j) ≥ 2 * distance (points k) (points l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_at_least_twice_min_distance_l900_90025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l900_90078

theorem inequality_solution_set (x : ℝ) :
  (x ≠ 2) → ((3 * x + 1) / (x - 2) ≤ 0 ↔ x ∈ Set.Icc (-1/3) 2 \ {2}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l900_90078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l900_90095

theorem inequality_solution_set (x : ℝ) :
  (x - 3)^2 - 2 * Real.sqrt ((x - 3)^2) - 3 < 0 ↔ 0 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l900_90095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_MON_l900_90098

-- Define the ellipse and circle
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the point P on the ellipse in the first quadrant
def P : ℝ × ℝ → Prop :=
  λ p => ellipse p.1 p.2 ∧ p.1 > 0 ∧ p.2 > 0

-- Define the tangent line from P to the circle
def tangent_line (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  p.1 * x + p.2 * y = 9

-- Define the intersection points M and N
noncomputable def M (p : ℝ × ℝ) : ℝ := 9 / (4 * p.1)
noncomputable def N (p : ℝ × ℝ) : ℝ := 3 / p.2

-- Define the area of triangle MON
noncomputable def area_MON (p : ℝ × ℝ) : ℝ :=
  (1/2) * (M p) * (N p)

-- State the theorem
theorem min_area_MON :
  ∃ (min_area : ℝ), min_area = 27/4 ∧
  ∀ (p : ℝ × ℝ), P p → area_MON p ≥ min_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_MON_l900_90098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonic_decreasing_l900_90015

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_monotonic_decreasing :
  ∃ α : ℝ, 
    (power_function α 2 = 1/2) ∧ 
    (∀ x y : ℝ, x < y ∧ x > 0 ∧ y > 0 → power_function α x > power_function α y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonic_decreasing_l900_90015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_intersection_point_value_l900_90002

theorem tan_intersection_point_value (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_intersection : ∀ (x₁ x₂ : ℝ), Real.tan (ω * x₁) = 3 → Real.tan (ω * x₂) = 3 → x₁ ≠ x₂ → |x₁ - x₂| = π/4) :
  Real.tan (ω * π/12) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_intersection_point_value_l900_90002
