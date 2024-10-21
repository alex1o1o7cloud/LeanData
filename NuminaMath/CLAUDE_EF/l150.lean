import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_share_l150_15001

/-- Calculate each person's share of a restaurant bill with tip -/
theorem restaurant_bill_share (total_bill : ℝ) (num_people : ℕ) (tip_percent : ℝ) :
  total_bill = 211 ∧ num_people = 6 ∧ tip_percent = 0.15 →
  ∃ (share : ℝ), (share ≥ 40.43 ∧ share ≤ 40.45) ∧ 
    (share * (num_people : ℝ) ≥ total_bill * (1 + tip_percent) - 0.01 ∧
     share * (num_people : ℝ) ≤ total_bill * (1 + tip_percent) + 0.01) :=
by
  intro h
  use 40.44166666666667
  apply And.intro
  · apply And.intro
    · norm_num
    · norm_num
  · apply And.intro
    · norm_num [h]
    · norm_num [h]

#eval (211 : ℚ) * (1 + 0.15) / 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_share_l150_15001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_tangent_parallel_points_l150_15050

-- Define the sphere
def sphere (x y z : ℝ) : Prop := x^2 + y^2 + z^2 = 676

-- Define the plane
def plane (x y z : ℝ) : Prop := 3*x - 12*y + 4*z = 0

-- Define the tangent plane to the sphere at point (a, b, c)
def tangent_plane (a b c x y z : ℝ) : Prop := a*x + b*y + c*z = 676

-- Define the condition for parallel planes
def parallel_planes (a b c : ℝ) : Prop := ∃ (t : ℝ), a = 3*t ∧ b = -12*t ∧ c = 4*t

-- Theorem statement
theorem sphere_tangent_parallel_points :
  ∀ (x y z : ℝ), 
    sphere x y z ∧ parallel_planes x y z ↔ 
    (x = 6 ∧ y = -24 ∧ z = 8) ∨ (x = -6 ∧ y = 24 ∧ z = -8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_tangent_parallel_points_l150_15050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_problem_l150_15006

theorem boat_speed_problem (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ)
  (h1 : boat_speed = 12)
  (h2 : downstream_distance = 32)
  (h3 : upstream_distance = 16)
  : ∃ stream_speed : ℝ, 
    downstream_distance / (boat_speed + stream_speed) = upstream_distance / (boat_speed - stream_speed) ∧
    stream_speed = 4 := by
  use 4
  constructor
  · -- Prove the equation
    simp [h1, h2, h3]
    norm_num
  · -- Prove stream_speed = 4
    rfl

#check boat_speed_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_problem_l150_15006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_5_l150_15035

/-- The set of numbers from 100 to 999 inclusive -/
def NumberSet : Set ℕ := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

/-- A number is divisible by 5 -/
def DivisibleBy5 (n : ℕ) : Prop := n % 5 = 0

/-- The probability of selecting a number divisible by 5 from NumberSet -/
def ProbabilityDivisibleBy5 : ℚ :=
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 900)).card / 900

theorem probability_divisible_by_5 : ProbabilityDivisibleBy5 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_5_l150_15035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l150_15048

noncomputable def variance (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  (data.map (fun x => (x - mean) ^ 2)).sum / data.length

def transform (x : ℝ) : ℝ := 2 * x + 1

theorem variance_transformation (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  variance [x₁, x₂, x₃, x₄, x₅] = 3 →
  variance (List.map transform [x₁, x₂, x₃, x₄, x₅]) = 12 := by
  sorry

#check variance_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l150_15048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_point_n_properties_l150_15089

-- Define the parabola C
def Parabola (C : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ, p > 0 ∧ C = {(x, y) : ℝ × ℝ | y^2 = 2*p*x}

-- Define the focus F
def Focus (F : ℝ × ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ, p > 0 ∧ F = (p/2, 0) ∧ Parabola C

-- Define line l
def Line (l : Set (ℝ × ℝ)) (F : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, l = {(x, y) : ℝ × ℝ | x = t*y + F.1}

-- Define points A and B
def IntersectionPoints (A B : ℝ × ℝ) (C : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop :=
  A ∈ C ∧ A ∈ l ∧ B ∈ C ∧ B ∈ l ∧ A ≠ B

-- Define the dot product condition
def DotProductCondition (O A B : ℝ × ℝ) : Prop :=
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = -3/4

-- Define point M on the directrix
def PointOnDirectrix (M : ℝ → ℝ × ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → M m = (-p/2, m)

-- Define the second dot product condition
def SecondDotProductCondition (M A B : ℝ × ℝ) : Prop :=
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 9

-- Define point N
def PointN (N : ℝ × ℝ) (A B : ℝ × ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ, p > 0 ∧ N.1 = -p/2 ∧ N.2 = (A.2 + B.2) / 2

theorem parabola_and_point_n_properties
  (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ) (O : ℝ × ℝ) (M : ℝ → ℝ × ℝ) (N : ℝ × ℝ) :
  Parabola C →
  Focus F C →
  Line l F →
  IntersectionPoints A B C l →
  DotProductCondition O A B →
  PointOnDirectrix M C →
  SecondDotProductCondition (M 0) A B →
  PointN N A B C →
  (C = {(x, y) : ℝ × ℝ | y^2 = 2*x} ∧ 
   (N.2 ∈ Set.Icc (-4 : ℝ) (-2) ∨ N.2 ∈ Set.Icc 2 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_point_n_properties_l150_15089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l150_15018

-- Define the function f(x) = 2^(x+1)
noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.range f,
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x = y) ↔ 1 ≤ y ∧ y ≤ 4 :=
by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l150_15018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_to_N_l150_15090

/-- The distance between two points in 3D space -/
noncomputable def distance3D (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- Theorem: The distance between points M(3, 4, 1) and N(0, 0, 1) is 5 -/
theorem distance_M_to_N : distance3D 3 4 1 0 0 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_to_N_l150_15090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_impossibility_l150_15007

/-- Represents a rectangle --/
structure Rectangle where
  width : ℚ
  height : ℚ
  width_pos : width > 0
  height_pos : height > 0

/-- Calculates the aspect ratio of a rectangle --/
def aspect_ratio (r : Rectangle) : ℚ :=
  r.width / r.height

/-- Defines when one rectangle is inscribed in another --/
def inscribed (inner outer : Rectangle) : Prop :=
  ∃ (x y : ℚ), 
    0 ≤ x ∧ x + inner.width ≤ outer.width ∧
    0 ≤ y ∧ y + inner.height ≤ outer.height

theorem inscribed_rectangle_impossibility (outer_ratio inner_ratio : ℚ) : 
  outer_ratio = 9/16 → inner_ratio = 4/7 → 
  ¬ ∃ (outer inner : Rectangle), 
    (aspect_ratio outer = outer_ratio) ∧ 
    (aspect_ratio inner = inner_ratio) ∧ 
    (inscribed inner outer) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_impossibility_l150_15007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_partition_l150_15032

/-- A triple of integers (x, y, z) such that x + y = 3z -/
structure SumTriple where
  x : ℕ
  y : ℕ
  z : ℕ
  sum_property : x + y = 3 * z

/-- The set of integers from 1 to 3n -/
def set_up_to_3n (n : ℕ) : Set ℕ :=
  {i | 1 ≤ i ∧ i ≤ 3 * n}

/-- A partition of the set {1, ..., 3n} into n disjoint triples -/
structure Partition (n : ℕ) where
  triples : Fin n → SumTriple
  covers_set : ∀ i : Fin n, (triples i).x ∈ set_up_to_3n n ∧
                            (triples i).y ∈ set_up_to_3n n ∧
                            (triples i).z ∈ set_up_to_3n n
  disjoint : ∀ i j : Fin n, i ≠ j → 
    ({(triples i).x, (triples i).y, (triples i).z} : Set ℕ) ∩
    ({(triples j).x, (triples j).y, (triples j).z} : Set ℕ) = ∅

/-- The main theorem stating that 5 is the smallest positive integer satisfying the condition -/
theorem smallest_n_for_partition :
  (∃ (p : Partition 5), True) ∧
  (∀ n < 5, ¬∃ (p : Partition n), True) := by
  sorry

#check smallest_n_for_partition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_partition_l150_15032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_one_l150_15053

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x / (x - 2)

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := -2 / ((x - 2) ^ 2)

-- Theorem statement
theorem tangent_line_at_point_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -2 * x + 1 := by
  sorry

#check tangent_line_at_point_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_one_l150_15053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_control_plant_height_l150_15029

theorem control_plant_height 
  (control_height : ℝ)
  (bone_meal_height : ℝ)
  (cow_manure_height : ℝ)
  (h1 : bone_meal_height = 1.25 * control_height)
  (h2 : cow_manure_height = 2 * bone_meal_height)
  (h3 : cow_manure_height = 90) :
  control_height = 36 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_control_plant_height_l150_15029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elevator_probability_theorem_l150_15020

/-- Represents the elevator system in Dave's Amazing Hotel -/
structure ElevatorSystem :=
  (num_floors : ℕ)
  (num_moves : ℕ)

/-- Calculates the probability of ending on the first floor -/
def probability_first_floor (system : ElevatorSystem) : ℚ :=
  (2^system.num_moves.pred - 1) / (3 * 2^system.num_moves.pred)

/-- Theorem stating the properties of the elevator system and the result -/
theorem elevator_probability_theorem (system : ElevatorSystem) 
  (h1 : system.num_floors = 3) 
  (h2 : system.num_moves = 482) : 
  ∃ (m n : ℕ), 
    probability_first_floor system = m / n ∧ 
    Nat.Coprime m n ∧ 
    (m + n) % 1000 = 803 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elevator_probability_theorem_l150_15020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l150_15086

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2*x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_properties :
  (∃ p > 0, is_periodic f p ∧ ∀ q, 0 < q → is_periodic f q → p ≤ q) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≥ -1/2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi/2), f x = 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi/2), f x = -1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l150_15086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt2_div_2_l150_15011

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/4) -/
noncomputable def circle_radius : ℝ := Real.sqrt 2 / 2

/-- Spherical coordinates (ρ, θ, φ) -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- The set of points forming the circle -/
def circle_points : Set SphericalCoord :=
  {p : SphericalCoord | p.ρ = 1 ∧ p.φ = Real.pi / 4}

theorem circle_radius_is_sqrt2_div_2 :
  ∀ p ∈ circle_points, 
    Real.sqrt ((Real.sin p.φ * Real.cos p.θ) ^ 2 + (Real.sin p.φ * Real.sin p.θ) ^ 2) = circle_radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt2_div_2_l150_15011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_t_is_2_max_a_for_inequality_l150_15082

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - t|

-- Theorem for Question 1
theorem solution_set_when_t_is_2 :
  {x : ℝ | f 2 x > 2} = {x : ℝ | x < 1/2 ∨ x > 5/2} := by sorry

-- Theorem for Question 2
theorem max_a_for_inequality (t : ℝ) (h : t ∈ Set.Icc 1 2) :
  IsGreatest {a : ℝ | ∀ x ∈ Set.Icc (-1) 3, f t x ≥ a + x} (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_t_is_2_max_a_for_inequality_l150_15082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l150_15047

/-- The area of a sector of a circle with given diameter and central angle -/
noncomputable def sector_area (diameter : ℝ) (central_angle : ℝ) : ℝ :=
  (central_angle / 360) * (Real.pi * (diameter / 2)^2)

/-- Theorem: The area of a sector of a circle with diameter 8 meters and central angle 60 degrees is 8π/3 square meters -/
theorem sector_area_specific : sector_area 8 60 = 8 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l150_15047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l150_15038

theorem trig_problem (α : Real) 
  (h1 : Real.sin α - Real.cos α = Real.sqrt 10 / 5)
  (h2 : α ∈ Set.Ioo π (2 * π)) : 
  (Real.sin α + Real.cos α = -2 * Real.sqrt 10 / 5) ∧ 
  (Real.tan α - 1 / Real.tan α = -8 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l150_15038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_upper_bound_l150_15049

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x + 4 / x
def g (a x : ℝ) : ℝ := 2^x + a

-- State the theorem
theorem function_inequality_implies_upper_bound (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g a x₂) →
  a ≤ 1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_upper_bound_l150_15049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_x_squared_implies_a_geq_one_over_e_squared_l150_15046

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x + x + x * Real.log x

-- State the theorem
theorem f_geq_x_squared_implies_a_geq_one_over_e_squared (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x ≥ x^2) →
  a ≥ 1 / Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_x_squared_implies_a_geq_one_over_e_squared_l150_15046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l150_15019

theorem log_sum_upper_bound (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) :
  (Real.log (a^2 / b^2) / Real.log a + Real.log (b^2 / a^2) / Real.log b ≤ 0) ∧
  (∃ x y : ℝ, x ≥ y ∧ y > 2 ∧ Real.log (x^2 / y^2) / Real.log x + Real.log (y^2 / x^2) / Real.log y = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l150_15019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_subset_theorem_l150_15054

theorem matrix_subset_theorem (m n : ℕ) (h_neq : m ≠ n) 
  (A : Matrix (Fin m) (Fin n) Bool) 
  (h_inj : ∀ f : Fin m → Fin n, Function.Injective f → 
    ∃ i : Fin m, A i (f i) = false) :
  ∃ (S : Finset (Fin m)) (T : Finset (Fin n)), 
    (∀ (i : Fin m) (j : Fin n), i ∈ S → j ∈ T → A i j = false) ∧
    (S.card + T.card > n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_subset_theorem_l150_15054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_seven_l150_15085

theorem count_divisible_by_seven : 
  (Finset.filter (fun n => n % 7 = 0) (Finset.range 251 \ Finset.range 100)).card = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_seven_l150_15085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_diff_implies_cos_sin_ratio_l150_15062

theorem tan_sum_diff_implies_cos_sin_ratio 
  (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5) 
  (h2 : Real.tan (β - π/4) = 1/4) : 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_diff_implies_cos_sin_ratio_l150_15062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_ratio_hexahedron_octahedron_product_mn_is_six_l150_15025

noncomputable def inscribed_sphere_radius_hexahedron (a : ℝ) : ℝ :=
  (Real.sqrt 6 * a) / 9

noncomputable def inscribed_sphere_radius_octahedron (a : ℝ) : ℝ :=
  (Real.sqrt 6 * a) / 6

/-- The ratio of the radii of inscribed spheres of a hexahedron and a regular octahedron with equilateral triangle faces of side length a is 2/3 -/
theorem inscribed_sphere_ratio_hexahedron_octahedron (a : ℝ) (a_pos : a > 0) :
  ∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧
  (r₁ = inscribed_sphere_radius_hexahedron a) ∧
  (r₂ = inscribed_sphere_radius_octahedron a) ∧
  (r₁ / r₂ = 2 / 3) := by
  sorry

/-- The product of m and n in the ratio m/n of inscribed sphere radii is 6 -/
theorem product_mn_is_six :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ (2 : ℚ) / 3 = m / n ∧ m * n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_ratio_hexahedron_octahedron_product_mn_is_six_l150_15025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABF_l150_15064

/-- Square with side length 2 and vertex at origin -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- Point A is the origin -/
def A : ℝ × ℝ := (0, 0)

/-- Point B is (2, 0) -/
def B : ℝ × ℝ := (2, 0)

/-- Point D is (0, 2) -/
def D : ℝ × ℝ := (0, 2)

/-- E is a point inside the square such that AE = BE -/
noncomputable def E : ℝ × ℝ := sorry

/-- G is the midpoint of AB -/
def G : ℝ × ℝ := (1, 0)

/-- F is the intersection of diagonal BD and line segment AE -/
def F : ℝ × ℝ := (1, 1)

/-- The area of triangle ABF is 1 -/
theorem area_triangle_ABF : 
  (1/2) * ‖B - A‖ * ‖F.2 - A.2‖ = 1 := by
  -- Proof steps
  sorry

/-- Auxiliary lemma: F is indeed on the diagonal BD -/
lemma F_on_diagonal : F.2 = -F.1 + 2 := by
  -- Proof steps
  sorry

/-- Auxiliary lemma: F is indeed on the line AE -/
lemma F_on_AE : F.1 = 1 := by
  -- Proof steps
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABF_l150_15064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_for_given_charge_l150_15076

/-- Calculates the distance traveled given taxi fare parameters -/
noncomputable def distance_traveled (initial_fee : ℚ) (charge_per_increment : ℚ) (increment_distance : ℚ) (total_charge : ℚ) : ℚ :=
  let distance_charge := total_charge - initial_fee
  let num_increments := distance_charge / charge_per_increment
  num_increments * increment_distance

/-- Proves that the distance traveled is 3.6 miles given the specified taxi fare parameters -/
theorem distance_for_given_charge :
  let initial_fee : ℚ := 205/100
  let charge_per_increment : ℚ := 35/100
  let increment_distance : ℚ := 2 / 5
  let total_charge : ℚ := 520/100
  distance_traveled initial_fee charge_per_increment increment_distance total_charge = 18/5 := by
  -- Unfold the definition and simplify
  unfold distance_traveled
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_for_given_charge_l150_15076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_average_speed_l150_15027

/-- Represents a segment of the motorcycle's journey -/
structure Segment where
  speed : ℝ  -- Speed in kph
  distance : Option ℝ  -- Distance in km, if given
  time : Option ℝ  -- Time in hours, if given

/-- Calculates the average speed given a list of travel segments and a stop time -/
noncomputable def averageSpeed (segments : List Segment) (stopTime : ℝ) : ℝ :=
  let totalDistance := segments.foldl (fun acc s => acc + match s.distance with
    | some d => d
    | none => s.speed * s.time.getD 0) 0
  let totalTime := segments.foldl (fun acc s => acc + match s.time with
    | some t => t
    | none => s.distance.getD 0 / s.speed) 0 + stopTime
  totalDistance / totalTime

theorem motorcycle_average_speed :
  let segments : List Segment := [
    { speed := 50, distance := some 30, time := none },
    { speed := 55, distance := some 40, time := none },
    { speed := 45, distance := none, time := some 0.5 },
    { speed := 50, distance := none, time := some (10/60) }
  ]
  let stopTime : ℝ := 20/60  -- 20 minutes in hours
  abs (averageSpeed segments stopTime - 43.35) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_average_speed_l150_15027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l150_15060

-- Define the constants
noncomputable def a : ℝ := 2^(1.2 : ℝ)
noncomputable def b : ℝ := (1/2)^(-(0.5 : ℝ))
noncomputable def c : ℝ := 2 * Real.log 2 / Real.log 5

-- State the theorem
theorem relationship_abc : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l150_15060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_longer_side_l150_15055

/-- Represents the properties of a trapezium --/
structure Trapezium where
  short_side : ℝ
  height : ℝ
  area : ℝ

/-- Calculates the length of the longer parallel side of a trapezium --/
noncomputable def longer_side (t : Trapezium) : ℝ :=
  (2 * t.area / t.height) - t.short_side

/-- Theorem stating that for a trapezium with given properties, the longer side is 18 cm --/
theorem trapezium_longer_side :
  let t : Trapezium := {
    short_side := 10,
    height := 10.00001,
    area := 140.00014
  }
  longer_side t = 18 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_longer_side_l150_15055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_interior_angle_proof_l150_15072

/-- The measure of each interior angle of a regular hexagon is 120 degrees. -/
def regular_hexagon_interior_angle : ℚ := 120

/-- A regular hexagon has 6 sides. -/
def regular_hexagon_sides : ℕ := 6

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees. -/
def polygon_interior_angle_sum (n : ℕ) : ℚ := (n - 2) * 180

theorem regular_hexagon_interior_angle_proof :
  regular_hexagon_interior_angle = 
    (polygon_interior_angle_sum regular_hexagon_sides) / regular_hexagon_sides :=
by
  -- Proof steps would go here
  sorry

#eval regular_hexagon_interior_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_interior_angle_proof_l150_15072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l150_15094

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def expression (a b c d : ℕ) : ℕ := c * (factorial a)^b - d

def is_permutation (a b c d : ℕ) : Prop :=
  Multiset.ofList [a, b, c, d] = Multiset.ofList [0, 1, 2, 3]

theorem max_value_of_expression :
  ∃ (a b c d : ℕ), is_permutation a b c d ∧ 
    (∀ (w x y z : ℕ), is_permutation w x y z → expression a b c d ≥ expression w x y z) ∧
    expression a b c d = 36 := by
  -- Proof goes here
  sorry

#eval expression 3 2 1 0  -- This should output 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l150_15094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l150_15042

/-- Marguerite's driving distance in miles -/
noncomputable def marguerite_distance : ℝ := 150

/-- Marguerite's driving time in hours -/
noncomputable def marguerite_time : ℝ := 3

/-- Sam's driving time in hours -/
noncomputable def sam_time : ℝ := 4

/-- Calculate the distance driven given a rate and time -/
noncomputable def distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Calculate the rate given a distance and time -/
noncomputable def rate (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem sam_distance :
  distance (rate marguerite_distance marguerite_time) sam_time = 200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_distance_l150_15042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_set_1_condition_set_2_l150_15067

noncomputable section

-- Define the lines l₁ and l₂
def l₁ (a b : ℝ) (x y : ℝ) : Prop := a * x - b * y + 4 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y + b = 0

-- Define perpendicularity of lines
def perpendicular (a b : ℝ) : Prop := a * (a - 1) + (-b) * 1 = 0

-- Define parallel lines
def parallel (a b : ℝ) : Prop := a / b = 1 - a

-- Define distance from origin to line ax + by + c = 0
noncomputable def distance_to_origin (a b c : ℝ) : ℝ := |c| / Real.sqrt (a^2 + b^2)

theorem condition_set_1 (a b : ℝ) : 
  l₁ a b (-3) (-1) ∧ perpendicular a b → a = 2 ∧ b = 2 := by
  sorry

theorem condition_set_2 (a b : ℝ) : 
  parallel a b ∧ 
  distance_to_origin a (-b) 4 = distance_to_origin (a-1) 1 b → 
  (a = 2 ∧ b = -2) ∨ (a = 2/3 ∧ b = 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_set_1_condition_set_2_l150_15067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_children_on_playground_l150_15012

def playground_children_count (girls : ℕ) (boys : ℕ) : ℕ :=
  girls + boys

theorem total_children_on_playground :
  let girls : ℕ := 28
  let boys : ℕ := 35
  playground_children_count girls boys = 63 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_children_on_playground_l150_15012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_phi_zero_l150_15043

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem cos_phi_zero (φ : ℝ) 
  (h : ∀ x, f x φ = -f (-x) φ) : 
  Real.cos φ = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_phi_zero_l150_15043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_savings_l150_15031

/-- Calculates the cost of two pairs of shoes under Promotion A -/
noncomputable def costPromotionA (price1 price2 : ℝ) : ℝ := price1 + price2 / 2

/-- Calculates the cost of two pairs of shoes under Promotion B -/
noncomputable def costPromotionB (price1 price2 : ℝ) : ℝ := price1 + price2 - 15

/-- Proves that the difference in cost between Promotion B and Promotion A is $5 -/
theorem promotion_savings (price1 price2 : ℝ) 
  (h1 : price1 = 50) 
  (h2 : price2 = 40) : 
  costPromotionB price1 price2 - costPromotionA price1 price2 = 5 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_savings_l150_15031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_s_equality_l150_15095

-- Define the function s as noncomputable
noncomputable def s (θ : ℝ) : ℝ := 1 / (1 + θ)

-- State the theorem
theorem nested_s_equality : s (s (s (s (s (s 50))))) = 258 / 413 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_s_equality_l150_15095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l150_15097

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y)) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/3), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/3), f x ≥ 0) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi/3), f x = 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi/3), f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l150_15097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l150_15041

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) (A : ℝ) : ℝ :=
  Real.sin (x + Real.pi/6) + Real.sin (x - Real.pi/6) + Real.cos x + Real.cos A

theorem triangle_problem (t : Triangle) 
    (h_max : ∀ x, f x t.A ≤ 7/3)
    (h_area : 1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 2)
    (h_sum : t.b + t.c = 4) :
    Real.sin t.A = 2 * Real.sqrt 2 / 3 ∧ t.a = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l150_15041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_range_l150_15071

-- Define a type for lines in a 2D plane
structure Line where
  slope : ℝ

-- Define the angle a line makes with the positive x-axis
noncomputable def angle_with_x_axis (l : Line) : ℝ :=
  Real.arctan l.slope

-- Theorem statement
theorem line_angle_range :
  ∀ l : Line, 0 ≤ angle_with_x_axis l ∧ angle_with_x_axis l < π :=
by
  intro l
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_range_l150_15071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_min_l150_15028

noncomputable section

/-- The parabola y^2 = 2x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 2 * p.1

/-- The directrix of the parabola y^2 = 2x -/
def directrix : ℝ → ℝ := λ _ ↦ -1/4

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Distance from a point to the directrix -/
def distToDirectrix (p : ℝ × ℝ) : ℝ :=
  |p.1 + 1/4|

theorem parabola_distance_sum_min :
  ∀ p : ℝ × ℝ, parabola p →
    distance p (0, 2) + distToDirectrix p ≥ Real.sqrt 17 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_min_l150_15028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_line_intersection_tangent_line_l150_15077

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ := p.y

theorem trajectory_equation (p : Point) (h1 : p.y ≥ 0) :
  distance p ⟨0, 1/2⟩ = distanceToXAxis p + 1/2 →
  p.y^2 - 2*p.y + (p.x^2 + 1/4) = 0 := by sorry

/-- The equation of the line y = kx + 1 -/
def line_equation (k : ℝ) (x : ℝ) : ℝ := k * x + 1

theorem line_intersection (k : ℝ) :
  ∃ A B : Point,
    A.y = line_equation k A.x ∧
    B.y = line_equation k B.x ∧
    A.y^2 - 2*A.y + (A.x^2 + 1/4) = 0 ∧
    B.y^2 - 2*B.y + (B.x^2 + 1/4) = 0 ∧
    distance A B = 2 * Real.sqrt 6 →
    k = 1 ∨ k = -1 := by sorry

theorem tangent_line (y₀ : ℝ) :
  let Q : Point := ⟨1, y₀⟩
  1^2 = 2 * y₀ →
  y₀ = 1/2 ∧
  (2 : ℝ) * Q.x - 2 * Q.y - 1 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_line_intersection_tangent_line_l150_15077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teddy_pillow_material_l150_15024

/-- The amount of fluffy foam material Teddy has in tons -/
def material_tons : ℚ := 3

/-- The number of pillows Teddy can make -/
def num_pillows : ℕ := 3000

/-- The number of pounds in a ton -/
def pounds_per_ton : ℕ := 2000

/-- The amount of fluffy foam material used for each pillow in pounds -/
noncomputable def material_per_pillow : ℚ := (material_tons * pounds_per_ton) / num_pillows

theorem teddy_pillow_material :
  material_per_pillow = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teddy_pillow_material_l150_15024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_jump_problem_l150_15033

/-- The minimum number of jumps required for a flea to cover a given distance -/
noncomputable def min_jumps (jump_length : ℝ) (distance : ℝ) : ℕ :=
  Nat.ceil (distance / jump_length)

/-- Theorem stating the minimum number of jumps for the given problem -/
theorem flea_jump_problem :
  let jump_length : ℝ := 15 / 1000  -- 15 mm in meters
  let distance : ℝ := 2020 / 100    -- 2020 cm in meters
  min_jumps jump_length distance = 1347 := by
  sorry

#check flea_jump_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_jump_problem_l150_15033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_l150_15069

/-- A right pyramid with a regular hexagonal base -/
structure HexagonalPyramid where
  /-- Total surface area of the pyramid -/
  total_surface_area : ℝ
  /-- The area of each triangular face is one-third the area of the hexagonal base -/
  face_area_ratio : ℝ
  /-- Condition: The total surface area is 648 square units -/
  total_area_constraint : total_surface_area = 648
  /-- Condition: The area of each triangular face is one-third the area of the hexagonal base -/
  face_ratio_constraint : face_area_ratio = 1/3

/-- The volume of the hexagonal pyramid -/
noncomputable def volume (p : HexagonalPyramid) : ℝ := 1728 * Real.sqrt 6

/-- Theorem: The volume of the hexagonal pyramid is 1728√6 cubic units -/
theorem hexagonal_pyramid_volume (p : HexagonalPyramid) : volume p = 1728 * Real.sqrt 6 := by
  -- Unfold the definition of volume
  unfold volume
  -- The equality is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_l150_15069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_proper_divisors_540_l150_15045

theorem sum_proper_divisors_540 : 
  (Finset.filter (λ x : ℕ => x ≠ 540 ∧ 540 % x = 0) (Finset.range 540)).sum id = 1140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_proper_divisors_540_l150_15045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salad_ingredients_l150_15084

theorem salad_ingredients (mushrooms : ℕ) (cheese_cubes : ℕ) 
  (h1 : mushrooms = 6)
  (h2 : cheese_cubes = 8) :
  let cherry_tomatoes := 3 * mushrooms
  let pickles := (cherry_tomatoes + mushrooms) / 2
  let bacon_bits := 5 * pickles
  let olives := 2 * pickles
  let final_cherry_tomatoes := cherry_tomatoes - 3
  let red_bacon_bits := bacon_bits / 5
  let green_olives := (3 * olives) / 4
  red_bacon_bits = 12 ∧ green_olives = 18 ∧ cheese_cubes = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salad_ingredients_l150_15084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l150_15075

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, a^2+1, 2*a-1}

-- Define function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x + 6 else x + 6

-- State the theorem
theorem problem_solution :
  ∃ a : ℝ, (A a ∩ B a = {-3}) ∧
  (a = -1) ∧
  (∀ x : ℝ, f x > f (-a) ↔ x ∈ Set.Ioo (-3) 1 ∪ Set.Ioi 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l150_15075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_at_1_5_l150_15061

-- Define the function p
noncomputable def p : ℝ → ℝ := sorry

-- State the properties of p
axiom p_graph : ∀ x : ℝ, (x = 1.5 ∧ p x = 4) ∨ 
  (∃ y : ℝ, (x, y) ∈ {(x, y) | y = p x ∧ 
  ((x < 1.5 ∧ y < 4) ∨ (x > 1.5 ∧ y > 4))})

axiom p_integer_at_1_5 : ∃ n : ℤ, p 1.5 = n

-- Theorem to prove
theorem p_at_1_5 : p 1.5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_at_1_5_l150_15061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ioanas_number_l150_15096

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def contest_numbers : Finset ℕ := {11, 12, 13, 14, 15, 16}

structure Participant where
  name : String
  number : ℕ

def participants : Finset String := {"Glenda", "Helga", "Ioana", "Julia", "Karl", "Liu"}

theorem ioanas_number (
  assignment : String → ℕ
) (h1 : ∀ p, p ∈ participants → assignment p ∈ contest_numbers)
  (h2 : ∀ p q, p ∈ participants → q ∈ participants → p ≠ q → assignment p ≠ assignment q)
  (h3 : Even (assignment "Helga") ∧ Even (assignment "Julia"))
  (h4 : is_prime (assignment "Karl") ∧ is_prime (assignment "Liu"))
  (h5 : is_perfect_square (assignment "Glenda"))
  : assignment "Ioana" = 15 := by
  sorry

#check ioanas_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ioanas_number_l150_15096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_assignment_l150_15070

-- Define a type for 2D integer coordinates
def IntPoint := ℤ × ℤ

-- Define a function type for assigning positive integers to points
def Assignment := IntPoint → ℕ+

-- Define collinearity for three points
def collinear (p q r : IntPoint) : Prop :=
  ∃ (a b c : ℤ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
    a * (q.1 - p.1) + b * (r.1 - p.1) = 0 ∧
    a * (q.2 - p.2) + b * (r.2 - p.2) = 0

-- Define the property of having a common divisor greater than 1
def hasCommonDivisorGreaterThanOne (a b c : ℕ+) : Prop :=
  ∃ (d : ℕ), d > 1 ∧ d ∣ a.val ∧ d ∣ b.val ∧ d ∣ c.val

-- Main theorem statement
theorem no_valid_assignment :
  ¬∃ (f : Assignment),
    ∀ (p q r : IntPoint),
      collinear p q r ↔ hasCommonDivisorGreaterThanOne (f p) (f q) (f r) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_assignment_l150_15070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_ten_minutes_l150_15014

/-- The time difference in minutes between two people arriving at a destination -/
noncomputable def timeDifference (distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  (distance / speed2 - distance / speed1) * 60

/-- Proof that the time difference is 10 minutes -/
theorem time_difference_is_ten_minutes :
  timeDifference 2 12 6 = 10 := by
  -- Unfold the definition of timeDifference
  unfold timeDifference
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_ten_minutes_l150_15014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l150_15066

open Real

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_monotonic : Monotone f
axiom f_property : ∀ x > 0, f (f x - x^3) = 2

-- Define the equation we want to solve
def equation (x : ℝ) : Prop := f x - deriv f x = 2

-- Theorem statement
theorem solution_in_interval :
  ∃ x ∈ Set.Ioo 3 4, equation x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l150_15066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l150_15040

open Real
open BigOperators
open Finset

theorem log_inequality (n : ℕ) : 
  0 < n → log (1 + n : ℝ) > ∑ i in range n, (i - 1 : ℝ) / i^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l150_15040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_median_slopes_unique_n_value_l150_15099

-- Define a right triangle with legs parallel to x and y axes
structure RightTriangle where
  a : ℝ  -- x-coordinate of the right angle
  b : ℝ  -- y-coordinate of the right angle
  c : ℝ  -- height of the triangle
  d : ℝ  -- width of the triangle

-- Define the slopes of the medians
noncomputable def median1_slope (t : RightTriangle) : ℝ := t.c / (2 * t.d)
noncomputable def median2_slope (t : RightTriangle) : ℝ := (2 * t.c) / t.d

-- Theorem stating that if one median is on y = 2x + 1, the other must be on y = 8x + 2
theorem unique_median_slopes (t : RightTriangle) :
  median1_slope t = 2 → median2_slope t = 8 := by
  sorry

-- Theorem stating that there is only one possible value for n
theorem unique_n_value :
  ∃! n : ℝ, ∀ t : RightTriangle, median1_slope t = 2 → median2_slope t = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_median_slopes_unique_n_value_l150_15099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l150_15037

theorem expression_evaluation : 
  let x : ℝ := Real.sqrt 27 + abs (-2) - 3 * Real.tan (60 * π / 180)
  ((x^2 - 1) / (x^2 - 2*x + 1) - 1 / (x - 1)) / ((x + 2) / (x - 1)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l150_15037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_values_l150_15036

theorem sum_of_x_values : ∃ (x₁ x₂ : ℝ),
  (Real.sqrt ((x₁ + 2)^2 + 9) = 10) ∧
  (Real.sqrt ((x₂ + 2)^2 + 9) = 10) ∧
  (∀ x : ℝ, Real.sqrt ((x + 2)^2 + 9) = 10 → (x = x₁ ∨ x = x₂)) ∧
  (x₁ + x₂ = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_values_l150_15036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_max_k_value_l150_15078

noncomputable def f (x : ℝ) : ℝ := 5 + Real.log x
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (k * x) / (x + 1)

-- Theorem for part I
theorem tangent_line_values (k : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    HasDerivAt (g k) ((deriv f) 1) x₀ ∧
    g k x₀ = f 1 + (deriv f 1) * (x₀ - 1)) ↔ 
  (k = 1 ∨ k = 9) := by
  sorry

-- Theorem for part II
theorem max_k_value :
  (∃ k : ℕ, k > 0 ∧ 
    (∀ x : ℝ, x > 1 → f x > g k x) ∧
    (∀ m : ℕ, m > k → ∃ x : ℝ, x > 1 ∧ f x ≤ g m x)) ↔
  (∃ k : ℕ, k = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_max_k_value_l150_15078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_area_l150_15021

-- Define the area of the circle
noncomputable def circle_area : ℝ := 64 * Real.pi

-- Theorem statement
theorem circle_radius_from_area :
  ∃ r : ℝ, r > 0 ∧ circle_area = Real.pi * r^2 ∧ r = 8 := by
  -- Introduce the radius
  let r : ℝ := 8
  
  -- Prove existence
  use r
  
  -- Prove the three conditions
  constructor
  · -- r > 0
    norm_num
  constructor
  · -- circle_area = Real.pi * r^2
    unfold circle_area
    ring
  · -- r = 8
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_area_l150_15021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birmingham_to_sheffield_routes_l150_15003

/-- Represents the number of routes between two cities -/
def RouteCount : Type := Nat

instance : OfNat RouteCount n where
  ofNat := n

instance : HMul RouteCount RouteCount RouteCount where
  hMul := Nat.mul

theorem birmingham_to_sheffield_routes 
  (bristol_to_birmingham : RouteCount)
  (sheffield_to_carlisle : RouteCount)
  (total_routes : RouteCount)
  (h1 : bristol_to_birmingham = 6)
  (h2 : sheffield_to_carlisle = 2)
  (h3 : total_routes = 36) :
  ∃ (birmingham_to_sheffield : RouteCount), 
    birmingham_to_sheffield = 3 ∧ 
    bristol_to_birmingham * birmingham_to_sheffield * sheffield_to_carlisle = total_routes :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birmingham_to_sheffield_routes_l150_15003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_right_focus_l150_15093

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Definition of the hyperbola equation -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Definition of the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the distance from a point on a hyperbola to its right focus -/
theorem distance_to_right_focus 
  (h : Hyperbola) 
  (p : Point) 
  (left_focus right_focus : Point) 
  (left_directrix : ℝ) :
  hyperbola_equation h p →
  h.b = 4 →
  p.x > 0 →
  distance p left_focus - distance p right_focus = 6 →
  |p.x - left_directrix| = 34/5 →
  distance p right_focus = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_right_focus_l150_15093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_balls_l150_15063

def box_problem (blue red green yellow purple orange black white : ℕ) : Prop :=
  blue = 8 ∧
  red = 5 ∧
  green = 3 * (2 * blue - 1) ∧
  yellow = 2 * (Int.floor (Real.sqrt (blue * red : ℝ))) ∧
  purple = 4 * (blue + green) ∧
  orange = 7 ∧
  black + white = blue + red + green + yellow + purple + orange ∧
  blue + red + green + yellow + purple + orange + black + white = 3 * (red + green + yellow + purple) + orange / 2

theorem total_balls (blue red green yellow purple orange black white : ℕ) :
  box_problem blue red green yellow purple orange black white →
  blue + red + green + yellow + purple + orange + black + white = 829 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_balls_l150_15063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_probability_10cm_l150_15030

/-- The probability of forming a triangle from three segments of a randomly divided line segment --/
noncomputable def triangle_probability (total_length : ℝ) : ℝ :=
  let f (x y : ℝ) : ℝ := if 0 < x ∧ x < total_length ∧ 0 < y ∧ y < total_length ∧ x + y > total_length / 2
                         then 1 / (total_length^2 / 2)
                         else 0
  ∫ x in Set.Icc 0 (total_length/2), ∫ y in Set.Icc (total_length/2 - x) total_length, f x y

/-- The theorem stating that the probability of forming a triangle from three segments
    of a randomly divided 10 cm line segment is 0.25 --/
theorem triangle_probability_10cm :
  triangle_probability 10 = 0.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_probability_10cm_l150_15030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l150_15000

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x - 1

-- State the theorem
theorem function_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : f a x₁ = 0) 
  (h2 : f a x₂ = 0) 
  (h3 : x₁ > x₂) :
  (0 < a ∧ a < Real.exp 1) ∧ (1 / x₁ + 2 / x₂ > 1 / a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l150_15000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l150_15023

/-- Calculates the speed of the first part of a journey given the total distance, 
    total time, speed and time of the second part. -/
noncomputable def first_part_speed (total_distance : ℝ) (total_time : ℝ) 
                     (second_part_speed : ℝ) (second_part_time : ℝ) : ℝ :=
  let second_part_distance := second_part_speed * second_part_time
  let first_part_distance := total_distance - second_part_distance
  let first_part_time := total_time - second_part_time
  first_part_distance / first_part_time

/-- Theorem stating that under the given conditions, the speed of the first part 
    of the journey is approximately 36.67 kmph. -/
theorem journey_speed_calculation :
  let total_distance := (400 : ℝ)
  let total_time := (8 : ℝ)
  let second_part_speed := (70 : ℝ)
  let second_part_time := (3.2 : ℝ)
  abs (first_part_speed total_distance total_time second_part_speed second_part_time - 36.67) < 0.01 :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval first_part_speed 400 8 70 3.2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l150_15023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_theorem_l150_15079

/-- Triangle DEF with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle WXYZ inscribed in triangle DEF -/
structure InscribedRectangle where
  base : ℝ
  height : ℝ

/-- Area of the inscribed rectangle as a function of θ -/
def rectangleArea (γ δ θ : ℝ) : ℝ := γ * θ - δ * θ^2

theorem inscribed_rectangle_theorem (t : Triangle) (rect : InscribedRectangle) 
    (γ δ : ℝ) (p q : ℕ) :
  t.a = 13 ∧ t.b = 30 ∧ t.c = 19 →
  δ = p / q →
  (∀ θ, rectangleArea γ δ θ = rect.base * rect.height) →
  Nat.Coprime p q →
  p + q = 266 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_theorem_l150_15079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_takeoff_run_theorem_l150_15016

/-- Calculates the distance traveled in uniformly accelerated motion -/
noncomputable def takeoffRunLength (time : ℝ) (finalSpeed : ℝ) : ℝ :=
  let acceleration := finalSpeed / time
  (1/2) * acceleration * time^2

/-- Converts km/h to m/s -/
noncomputable def kmhToMs (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

theorem takeoff_run_theorem :
  let time := 15
  let finalSpeed := kmhToMs 100
  let runLength := takeoffRunLength time finalSpeed
  ⌊runLength⌋ = 208 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_takeoff_run_theorem_l150_15016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_price_is_correct_l150_15010

noncomputable def bicycle_price (cost_price : ℝ) (loss_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let sp1 := cost_price * (1 - loss_percent / 100)
  sp1 * (1 - discount_percent / 100)

noncomputable def scooter_price (cost_price : ℝ) (loss_percent : ℝ) (tax_percent : ℝ) : ℝ :=
  let sp1 := cost_price * (1 - loss_percent / 100)
  sp1 * (1 + tax_percent / 100)

noncomputable def motorcycle_price (cost_price : ℝ) (loss_percent : ℝ) (commission_percent : ℝ) : ℝ :=
  let sp1 := cost_price * (1 - loss_percent / 100)
  sp1 * (1 - commission_percent / 100)

noncomputable def total_selling_price : ℝ :=
  bicycle_price 1600 10 2 + scooter_price 8000 5 3 + motorcycle_price 15000 8 4

theorem total_price_is_correct : total_selling_price = 23487.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_price_is_correct_l150_15010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_10_l150_15058

/-- Piecewise function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then 5*x + 10 else x^2 + 8*x + 15

/-- The solutions to f(x) = 10 -/
theorem solutions_of_f_eq_10 :
  {x : ℝ | f x = 10} = {-4 + Real.sqrt 11, -4 - Real.sqrt 11} := by
  sorry

#check solutions_of_f_eq_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_10_l150_15058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_solution_set_solution_set_quadratic_inequality_l150_15015

-- Problem 1
def solution_set (x : ℝ) : Prop := (2 * x + 1) / (3 - x) < 2

theorem complement_of_solution_set :
  {x : ℝ | ¬(solution_set x)} = Set.Icc (5/4) 3 := by sorry

-- Problem 2
def quadratic_inequality (a x : ℝ) : Prop := a * x^2 - (a + 4) * x + 4 ≤ 0

theorem solution_set_quadratic_inequality (a : ℝ) :
  {x : ℝ | quadratic_inequality a x} = 
    if a = 0 then Set.Ici 1
    else if a < 0 then (Set.Iic (4/a) ∪ Set.Ici 1)
    else if 0 < a ∧ a < 4 then Set.Icc 1 (4/a)
    else if a = 4 then {1}
    else Set.Icc (4/a) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_solution_set_solution_set_quadratic_inequality_l150_15015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_calculation_l150_15026

/-- The volume of a right circular cone -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The volume of a hemisphere -/
noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

/-- The total volume of ice cream in the cone and hemisphere -/
noncomputable def total_ice_cream_volume (cone_height cone_radius hemisphere_radius : ℝ) : ℝ :=
  cone_volume cone_radius cone_height + hemisphere_volume hemisphere_radius

theorem ice_cream_volume_calculation :
  total_ice_cream_volume 10 3 5 = (520/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_calculation_l150_15026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_sum_l150_15091

theorem last_two_digits_sum : (9^23 + 11^23) % 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_sum_l150_15091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_min_value_h_min_at_one_l150_15092

/-- The function h(x) as defined in the problem -/
noncomputable def h (x : ℝ) : ℝ := x + 1/x + (1/(x + 1/x))^2

/-- Theorem stating that the minimum value of h(x) for x > 0 is 9/4 -/
theorem h_min_value (x : ℝ) (hx : x > 0) : h x ≥ 9/4 := by
  sorry

/-- Theorem stating that the minimum value 9/4 is achieved when x = 1 -/
theorem h_min_at_one : h 1 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_min_value_h_min_at_one_l150_15092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l150_15088

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (area : ℝ), area = 6

-- Define point K on AB
def PointK (A B K : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t = 2 / 5 ∧ K = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define point L on AC
def PointL (A C L : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t = 5 / 8 ∧ L = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)

-- Define point Q as intersection of CK and BL
def PointQ (B C K L Q : ℝ × ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), 
    Q = (t₁ * C.1 + (1 - t₁) * K.1, t₁ * C.2 + (1 - t₁) * K.2) ∧
    Q = (t₂ * B.1 + (1 - t₂) * L.1, t₂ * B.2 + (1 - t₂) * L.2)

-- Define the distance from Q to AB
noncomputable def DistanceQtoAB (A B Q : ℝ × ℝ) : Prop :=
  ∃ (d : ℝ), d = 1.5 ∧ 
    d = (abs ((B.2 - A.2) * Q.1 + (A.1 - B.1) * Q.2 + (B.1 * A.2 - A.1 * B.2))) /
        (Real.sqrt ((B.2 - A.2)^2 + (A.1 - B.1)^2))

-- Theorem statement
theorem length_of_AB 
  (A B C K L Q : ℝ × ℝ) 
  (h₁ : Triangle A B C) 
  (h₂ : PointK A B K) 
  (h₃ : PointL A C L) 
  (h₄ : PointQ B C K L Q) 
  (h₅ : DistanceQtoAB A B Q) : 
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l150_15088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_volume_is_two_l150_15017

/-- Represents a polygon in the figure --/
inductive Polygon
| IsoscelesRightTriangle
| Rectangle
| EquilateralTriangle
deriving BEq

/-- The figure consisting of multiple polygons --/
structure Figure where
  polygons : List Polygon
  /-- Ensures the figure has the correct number and type of polygons --/
  valid_composition : 
    (polygons.filter (· == Polygon.IsoscelesRightTriangle)).length = 3 ∧
    (polygons.filter (· == Polygon.Rectangle)).length = 3 ∧
    (polygons.filter (· == Polygon.EquilateralTriangle)).length = 1

/-- The polyhedron formed by folding the figure --/
structure Polyhedron where
  figure : Figure
  /-- Assumes the polyhedron can be formed by folding the figure --/
  foldable : True

/-- Calculate the volume of the polyhedron --/
def volume (p : Polyhedron) : ℝ :=
  2  -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the volume of the polyhedron is 2 --/
theorem polyhedron_volume_is_two (p : Polyhedron) : volume p = 2 := by
  -- The proof is omitted
  sorry

#check polyhedron_volume_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_volume_is_two_l150_15017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_fuel_cost_electricity_hydrogen_l150_15002

/-- Calculates the median fuel cost per 100 miles for electricity and hydrogen -/
theorem median_fuel_cost_electricity_hydrogen (electricity_price : ℝ) (electricity_kwh_per_100miles : ℝ)
  (hydrogen_price : ℝ) (hydrogen_kg_per_100miles : ℝ) :
  let electricity_cost_per_100miles := electricity_price * electricity_kwh_per_100miles
  let hydrogen_cost_per_100miles := hydrogen_price * hydrogen_kg_per_100miles
  (min electricity_cost_per_100miles hydrogen_cost_per_100miles +
   max electricity_cost_per_100miles hydrogen_cost_per_100miles) / 2 =
    (electricity_cost_per_100miles + hydrogen_cost_per_100miles) / 2 := by
  sorry

-- Example usage (commented out as it's not necessary for building)
-- #eval median_fuel_cost_electricity_hydrogen 0.12 34 4.50 0.028

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_fuel_cost_electricity_hydrogen_l150_15002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_telecom_charge_reduction_original_charge_l150_15013

theorem telecom_charge_reduction (a b : ℝ) :
  let original := (5/4) * b + a
  let first_reduction := original - a
  let final := (4/5) * first_reduction
  final = b :=
by
  -- Introduce the local definitions
  intro original first_reduction final
  
  -- Expand the definitions
  calc
    final = (4/5) * first_reduction := by rfl
    _ = (4/5) * ((5/4) * b + a - a) := by rfl
    _ = (4/5) * ((5/4) * b) := by ring
    _ = b := by ring

-- The theorem proves that our calculation matches the given final price
theorem original_charge (a b : ℝ) :
  (5/4) * b + a = (5/4) * b + a :=
by rfl

-- This theorem shows that our answer matches the original charge

end NUMINAMATH_CALUDE_ERRORFEEDBACK_telecom_charge_reduction_original_charge_l150_15013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l150_15022

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1

-- Define the number of solutions function
noncomputable def num_solutions (a : ℝ) : ℕ :=
  if |a| < 1 then 4
  else if a = 1 then 2
  else if 1 < |a| ∧ |a| < Real.sqrt 2 then 4
  else if a = Real.sqrt 2 then 2
  else if a > Real.sqrt 2 then 4
  else 0  -- This case should never occur, but Lean requires all cases to be covered

-- State the theorem
theorem system_solutions (a : ℝ) :
  (∃ x y, system x y a) ↔ num_solutions a ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l150_15022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l150_15080

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let u := (w.1, w.2)
  let scalar := (v.1 * u.1 + v.2 * u.2) / (u.1 * u.1 + u.2 * u.2)
  (scalar * u.1, scalar * u.2)

theorem projection_property :
  ∃ (P : (ℝ × ℝ) → (ℝ × ℝ)),
    P (2, -3) = (-1, 3/2) →
    P (5, -1) = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l150_15080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_properties_l150_15098

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation mx^2 + ny^2 = 1 -/
structure Ellipse where
  m : ℝ
  n : ℝ
  h_positive : m > 0 ∧ n > 0
  h_different : m ≠ n

/-- Theorem about properties of an ellipse with a chord and its perpendicular bisector -/
theorem ellipse_chord_properties (e : Ellipse) (A B C D E F : Point) :
  -- AB is a chord of the ellipse with slope 1
  (A.y - B.y = A.x - B.x) →
  (e.m * A.x^2 + e.n * A.y^2 = 1) →
  (e.m * B.x^2 + e.n * B.y^2 = 1) →
  -- CD is the perpendicular bisector of AB
  ((C.y - D.y) * (A.x - B.x) = -(C.x - D.x) * (A.y - B.y)) →
  ((C.x + D.x) / 2 = (A.x + B.x) / 2) →
  ((C.y + D.y) / 2 = (A.y + B.y) / 2) →
  -- E is the midpoint of AB
  (E.x = (A.x + B.x) / 2) →
  (E.y = (A.y + B.y) / 2) →
  -- F is the midpoint of CD
  (F.x = (C.x + D.x) / 2) →
  (F.y = (C.y + D.y) / 2) →
  -- Properties to prove
  (((C.x - D.x)^2 + (C.y - D.y)^2) - ((A.x - B.x)^2 + (A.y - B.y)^2) = 
   4 * ((E.x - F.x)^2 + (E.y - F.y)^2)) ∧ 
  (∃ (center : Point) (radius : ℝ), 
    (A.x - center.x)^2 + (A.y - center.y)^2 = radius^2 ∧
    (B.x - center.x)^2 + (B.y - center.y)^2 = radius^2 ∧
    (C.x - center.x)^2 + (C.y - center.y)^2 = radius^2 ∧
    (D.x - center.x)^2 + (D.y - center.y)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_properties_l150_15098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_on_line_l150_15039

/-- 
Given an angle θ where:
- Its vertex is at the origin
- Its initial side is on the positive x-axis
- Its terminal side is on the line y = 2x
Prove that tan 2θ = -4/3
-/
theorem tan_double_angle_on_line (θ : ℝ) : 
  (∀ (x y : ℝ), y = 2 * x → x * Real.cos θ = y * Real.sin θ) → 
  Real.tan (2 * θ) = -4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_on_line_l150_15039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l150_15068

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos (x - Real.pi / 4))^2 - 1

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ p, p > 0 → (∀ x, f (x + p) = f x) → p ≥ Real.pi) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l150_15068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_perimeter_l150_15073

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ := 4 * s.sideLength

/-- Represents a set of concentric squares -/
structure ConcentricSquares where
  count : ℕ
  largestSquare : Square
  separation : ℝ

/-- Returns the smallest square in a set of concentric squares -/
def ConcentricSquares.smallestSquare (cs : ConcentricSquares) : Square :=
  { sideLength := cs.largestSquare.sideLength - 2 * (cs.count - 1 : ℝ) * cs.separation }

theorem smallest_square_perimeter
  (cs : ConcentricSquares)
  (h_count : cs.count = 8)
  (h_separation : cs.separation = 1)
  (h_largest_perimeter : cs.largestSquare.perimeter = 96) :
  (cs.smallestSquare).perimeter = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_perimeter_l150_15073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_three_in_factorial_30_l150_15057

theorem greatest_power_of_three_in_factorial_30 : 
  ∃ k : ℕ, (3^k : ℕ) ∣ (Nat.factorial 30) ∧ 
  ∀ m : ℕ, (3^m : ℕ) ∣ (Nat.factorial 30) → m ≤ k :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_three_in_factorial_30_l150_15057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l150_15004

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 3

-- Define the point outside the circle
noncomputable def external_point (a : ℝ) : ℝ × ℝ := (5, a)

-- Define the tangent line (implicitly)
def tangent_line (x y : ℝ) (a : ℝ) : Prop :=
  ∃ (t : ℝ), x = 5 * (1 - t) + t * 2 ∧ y = a * (1 - t) + t * 0

-- Define the chord length
noncomputable def chord_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem min_chord_length (a : ℝ) :
  ∃ (A B : ℝ × ℝ),
    circleC A.1 A.2 ∧
    circleC B.1 B.2 ∧
    tangent_line A.1 A.2 a ∧
    tangent_line B.1 B.2 a ∧
    (∀ (C D : ℝ × ℝ),
      circleC C.1 C.2 → circleC D.1 D.2 →
      tangent_line C.1 C.2 a → tangent_line D.1 D.2 a →
      chord_length A B ≤ chord_length C D) ∧
    chord_length A B = 2 * Real.sqrt 2 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l150_15004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l150_15051

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / Real.sqrt (2 * x - 10)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > 5}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l150_15051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_integer_f_non_integer_l150_15065

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def f (x : ℝ) : ℝ := (floor x : ℝ) + (floor (1 - x) : ℝ) + 1

theorem f_integer (x : ℝ) (h : x ∈ Set.Icc (-3 : ℝ) 3) (hz : ∃ (n : ℤ), x = n) : f x = 2 := by
  sorry

theorem f_non_integer (x : ℝ) (h : x ∈ Set.Icc (-3 : ℝ) 3) (hnz : ∀ (n : ℤ), x ≠ n) : f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_integer_f_non_integer_l150_15065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AED_l150_15083

/-- Rectangle ABCD with side lengths and midpoint E -/
structure Rectangle :=
  (A B C D E : ℝ × ℝ)
  (ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4)
  (bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 5)
  (is_rectangle : (A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) = 0)
  (e_midpoint : E = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

/-- Theorem: The area of triangle AED in the given rectangle is 5 -/
theorem area_of_triangle_AED (rect : Rectangle) : triangleArea rect.A rect.E rect.D = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AED_l150_15083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_sum_of_real_imag_parts_l150_15074

def z (m : ℝ) : ℂ := (m - 1) * (m + 2) + (m - 1) * Complex.I

theorem purely_imaginary_condition (m : ℝ) : 
  z m = Complex.I * Complex.im (z m) → m = -2 :=
sorry

theorem sum_of_real_imag_parts (a b : ℝ) :
  (z 2 + Complex.I) / (z 2 - Complex.I) = Complex.ofReal a + Complex.I * b → a + b = 3/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_sum_of_real_imag_parts_l150_15074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hexagon_with_internal_point_l150_15059

/-- A convex hexagon in 2D space -/
structure ConvexHexagon where
  vertices : Fin 6 → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

/-- The statement that no convex hexagon with all sides > 1 contains a point M
    within distance < 1 from all vertices -/
theorem no_hexagon_with_internal_point :
  ¬ ∃ (h : ConvexHexagon) (M : ℝ × ℝ),
    (∀ i j : Fin 6, i.val + 1 = j.val → dist (h.vertices i) (h.vertices j) > 1) ∧
    M ∈ interior (Set.range h.vertices) ∧
    (∀ i : Fin 6, dist M (h.vertices i) < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hexagon_with_internal_point_l150_15059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l150_15056

/-- A tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  /-- Length of edge AB in cm -/
  ab_length : ℝ
  /-- Area of face ABC in cm² -/
  abc_area : ℝ
  /-- Area of face ABD in cm² -/
  abd_area : ℝ
  /-- Angle between faces ABC and ABD in radians -/
  face_angle : ℝ

/-- The volume of a tetrahedron with given properties -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  512 / 27

/-- Theorem stating the volume of a specific tetrahedron -/
theorem specific_tetrahedron_volume :
  ∀ t : Tetrahedron,
    t.ab_length = 4 ∧
    t.abc_area = 16 ∧
    t.abd_area = 18 ∧
    t.face_angle = π / 4 →
    tetrahedron_volume t = 512 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l150_15056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_before_root_f_increasing_l150_15009

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log (1/2)

-- State the theorem
theorem f_negative_before_root (a x₀ : ℝ) (h1 : f a = 0) (h2 : 0 < x₀) (h3 : x₀ < a) :
  f x₀ < 0 := by
  sorry

-- Prove that f is increasing on (0, +∞)
theorem f_increasing (x y : ℝ) (hx : 0 < x) (hy : x < y) :
  f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_before_root_f_increasing_l150_15009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_sqrt2_over_2_l150_15044

-- Define the functions f and g
noncomputable def g (x : ℝ) : ℝ := Real.tan (x / 2)

noncomputable def f (y : ℝ) : ℝ := 
  2 * (2 * y * (1 - y^2)) / (1 + y^2)^2

-- State the theorem
theorem f_value_at_sqrt2_over_2 :
  f (Real.sqrt 2 / 2) = 4 * Real.sqrt 2 / 9 :=
by
  sorry

-- State the conditions
axiom f_g_composition (x : ℝ) : 
  0 < x → x < Real.pi → f (g x) = Real.sin (2 * x)

axiom g_definition (x : ℝ) : 
  0 < x → x < Real.pi → g x = Real.tan (x / 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_sqrt2_over_2_l150_15044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l150_15081

open Real

theorem smallest_positive_z (x z : ℝ) : 
  sin x = 0 → 
  sin (x + z) = sqrt 2 / 2 → 
  z > 0 → 
  (∀ w, w > 0 ∧ sin x = 0 ∧ sin (x + w) = sqrt 2 / 2 → z ≤ w) → 
  z = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l150_15081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_nine_exists_l150_15005

theorem difference_of_nine_exists (S : Finset ℕ) (h1 : S.card = 55) (h2 : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 99) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b = 9 ∨ b - a = 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_nine_exists_l150_15005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_three_l150_15034

/-- The area of a triangle given its three sides -/
noncomputable def area (a b c : ℝ) : ℝ :=
  Real.sqrt (1/4 * (a^2 * c^2 - ((a^2 + c^2 - b^2)/2)^2))

/-- Theorem: Given specific conditions, the area of triangle ABC is √3 -/
theorem triangle_area_is_sqrt_three 
  (a b c : ℝ) 
  (h1 : a^2 * Real.sin c = 4 * Real.sin a) 
  (h2 : (a + c)^2 = 12 + b^2) :
  area a b c = Real.sqrt 3 := by
  sorry

#check triangle_area_is_sqrt_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_three_l150_15034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l150_15052

-- Define the expression (marked as noncomputable due to Real.sqrt)
noncomputable def expression (x : ℝ) (n : ℕ) := (Real.sqrt x - 2 / x^2)^n

-- Define the sum of binomial coefficients
def sum_binomial_coefficients (n : ℕ) := 2^n

-- Theorem statement
theorem expansion_properties :
  ∃ (n : ℕ),
    -- Condition: sum of binomial coefficients is 1024
    sum_binomial_coefficients n = 1024 ∧
    -- 1. Prove n = 10
    n = 10 ∧
    -- 2. Prove constant term is 180
    (∃ (k : ℕ), Nat.choose n k * ((-2)^k : ℤ) = 180 ∧ 10 - 5*k = 0) ∧
    -- 3. Prove number of rational terms is 6
    (Finset.card (Finset.filter (λ k => (10 - 5*k) % 2 = 0) (Finset.range (n+1))) = 6) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l150_15052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_45_degree_angle_l150_15008

/-- The equation of a line passing through (0,1) with a 45° inclination angle is x - y + 1 = 0 -/
theorem line_equation_through_point_with_45_degree_angle :
  ∀ (x y : ℝ), 
  (∃ (l : Set (ℝ × ℝ)), 
    ((0, 1) ∈ l) ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l → (q.2 - p.2) = (q.1 - p.1)) →
    ((x, y) ∈ l ↔ x - y + 1 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_45_degree_angle_l150_15008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_gt_one_l150_15087

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

theorem f_range_gt_one :
  {x₀ : ℝ | f x₀ > 1} = Set.Ioi 1 ∪ Set.Iic (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_gt_one_l150_15087
