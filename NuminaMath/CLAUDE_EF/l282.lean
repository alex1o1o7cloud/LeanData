import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_arrangement_l282_28240

/-- The number of arrangements of 5 different products satisfying the conditions -/
def num_arrangements : ℕ := 36

/-- The total number of ways to arrange 5 different products -/
def total_arrangements : ℕ := Nat.factorial 5

/-- The number of arrangements where A is adjacent to B -/
def adjacent_ab : ℕ := 2 * Nat.factorial 4

/-- The number of arrangements where A is adjacent to both B and C -/
def adjacent_abc : ℕ := 2 * Nat.factorial 3

theorem product_arrangement :
  num_arrangements = adjacent_ab - adjacent_abc :=
by
  -- The proof goes here
  sorry

#eval num_arrangements
#eval adjacent_ab - adjacent_abc

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_arrangement_l282_28240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l282_28203

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (x^3) ^ (1/3)
def g (x : ℝ) : ℝ := x

-- State the theorem
theorem f_equiv_g : ∀ x : ℝ, f x = g x := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l282_28203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_chord_equation_l282_28230

/-- Theorem: Chord equation for tangents to a hyperbola
  Given a point P₀(x₀, y₀) outside the hyperbola x²/a² - y²/b² = 1 (a > 0, b > 0),
  if P₁ and P₂ are the intersection points of the tangents drawn from P₀ to the hyperbola,
  then the equation of the line containing the chord P₁P₂ is x₀x/a² - y₀y/b² = 1.
-/
theorem hyperbola_chord_equation 
  (a b x₀ y₀ : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_outside : x₀^2 / a^2 - y₀^2 / b^2 > 1) :
  ∃ (P₁ P₂ : ℝ × ℝ),
    (∀ (x y : ℝ), x₀ * x / a^2 - y₀ * y / b^2 = 1 ↔ 
      ((x, y) : ℝ × ℝ) ∈ Set.Icc P₁ P₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_chord_equation_l282_28230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_product_l282_28297

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (sum_eq : x + y + z = 9) (prod_sum_eq : x*y + y*z + z*x = 24) :
  ∃ (m : ℝ), m = min (x*y) (min (y*z) (z*x)) ∧ ∀ (m' : ℝ), m' = min (x*y) (min (y*z) (z*x)) → m' ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_product_l282_28297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_elements_theorem_l282_28291

theorem marked_elements_theorem (m : ℕ) (M : Finset ℕ) (M_sets : Finset (Finset ℕ)) :
  M.card = m →
  M_sets.card = 1986 →
  (∀ Mi ∈ M_sets, (Mi ∩ M).card > m / 2) →
  ∃ S : Finset ℕ, S ⊆ M ∧ S.card ≤ 10 ∧ ∀ Mi ∈ M_sets, (S ∩ Mi).Nonempty := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_elements_theorem_l282_28291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l282_28206

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi → -- Triangle angles are positive and sum to π
  a > 0 ∧ b > 0 ∧ c > 0 → -- Side lengths are positive
  a = 1 → -- Given condition
  Real.cos A = 4/5 → -- Given condition
  Real.cos C = 5/13 → -- Given condition
  Real.sin A = a / (2 * (a * b * Real.sin C + b * c * Real.sin A + c * a * Real.sin B) / (a + b + c)) → -- Law of sines
  Real.sin B = b / (2 * (a * b * Real.sin C + b * c * Real.sin A + c * a * Real.sin B) / (a + b + c)) → -- Law of sines
  Real.sin C = c / (2 * (a * b * Real.sin C + b * c * Real.sin A + c * a * Real.sin B) / (a + b + c)) → -- Law of sines
  b = 21/13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l282_28206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l282_28285

theorem roots_of_equation :
  let f : ℝ → ℝ := λ x => 4 * Real.sqrt x + 3 * x^(-(1/2 : ℝ)) - 10
  let root1 : ℝ := (19 + 5 * Real.sqrt 13) / 8
  let root2 : ℝ := (19 - 5 * Real.sqrt 13) / 8
  (∀ x : ℝ, x > 0 → (f x = 0 ↔ x = root1 ∨ x = root2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l282_28285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l282_28263

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 + x

-- State the theorem
theorem m_range (m : ℝ) (h : f (m^2 - 2*m) + f (m - 6) < 0) : -2 < m ∧ m < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l282_28263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_two_ln_two_l282_28251

-- Define the boundaries of the figure
noncomputable def left_boundary : ℝ := 1/2
noncomputable def right_boundary : ℝ := 2

-- Define the function representing the upper boundary
noncomputable def upper_boundary (x : ℝ) : ℝ := 1/x

-- Define the area of the figure
noncomputable def area : ℝ := ∫ x in left_boundary..right_boundary, upper_boundary x

-- Theorem statement
theorem area_equals_two_ln_two : area = 2 * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_two_ln_two_l282_28251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_triangle_area_is_half_l282_28289

/-- The area of the triangle formed by the lines y = 3, y = 2 + 2x, and y = 2 - 2x -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let line1 := fun x : ℝ => (3 : ℝ)
    let line2 := fun x : ℝ => 2 + 2 * x
    let line3 := fun x : ℝ => 2 - 2 * x
    let triangle := {p : ℝ × ℝ | ∃ x, (p.1 = x ∧ p.2 = line1 x) ∨ 
                                    (p.1 = x ∧ p.2 = line2 x) ∨ 
                                    (p.1 = x ∧ p.2 = line3 x)}
    area = (1/2 : ℝ)

/-- The area of the triangle is 0.5 -/
theorem triangle_area_is_half : triangle_area (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_triangle_area_is_half_l282_28289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supplementary_angles_difference_l282_28221

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 3 →  -- ratio of angles is 5:3
  |a - b| = 45 :=  -- positive difference is 45°
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supplementary_angles_difference_l282_28221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l282_28249

/-- Calculate the average speed of a car given four segments of travel -/
theorem car_average_speed (d1 d2 : ℝ) (t3 t4 : ℝ) : 
  let v1 : ℝ := 30
  let v2 : ℝ := 55
  let v3 : ℝ := 70
  let v4 : ℝ := 36
  let d3 : ℝ := v3 * t3
  let d4 : ℝ := v4 * t4
  let total_distance : ℝ := d1 + d2 + d3 + d4
  let total_time : ℝ := d1 / v1 + d2 / v2 + t3 + t4
  let average_speed : ℝ := total_distance / total_time
  d1 = 30 ∧ d2 = 35 ∧ t3 = 0.5 ∧ t4 = 2/3 →
  |average_speed - 44.238| < 0.001 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l282_28249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jar_capacity_theorem_l282_28207

/-- The number of regular toenails that can fit in Hilary's jar -/
def jar_capacity : ℕ := 60

/-- The number of big toenails already in the jar -/
def big_toenails : ℕ := 20

/-- The number of regular toenails already in the jar -/
def regular_toenails_in_jar : ℕ := 40

/-- The number of additional regular toenails that can fit in the jar -/
def additional_regular_toenails : ℕ := 20

/-- The total number of regular toenails that can fit in the jar -/
def total_regular_toenails : ℕ := regular_toenails_in_jar + additional_regular_toenails

theorem jar_capacity_theorem : total_regular_toenails = jar_capacity := by
  rfl

#eval total_regular_toenails

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jar_capacity_theorem_l282_28207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_durer_O_l282_28273

/-- The area of a shape formed by two intersecting unit circles with a 30° angle at their intersection --/
theorem area_of_durer_O : 
  let r : ℝ := 1  -- radius of the circles
  let θ : ℝ := 30 * (π / 180)  -- angle in radians
  let sector_area : ℝ := (π * r^2) * (2*θ) / (2*π)
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * r^2
  let intersection_area : ℝ := 2 * (sector_area - triangle_area)
  let circle_area : ℝ := π * r^2
  circle_area - intersection_area = (2*π)/3 + Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_durer_O_l282_28273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_stars_always_remain_and_six_stars_can_be_removed_l282_28296

-- Define a 4x4 grid as a function from (Fin 4 × Fin 4) to Bool
def Grid := Fin 4 → Fin 4 → Bool

-- Define a function to count stars in a grid
def countStars (g : Grid) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 4)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 4)) fun j =>
      if g i j then 1 else 0)

-- Define a function to check if a star exists after removing two rows and two columns
def starExistsAfterRemoval (g : Grid) (r1 r2 c1 c2 : Fin 4) : Prop :=
  ∃ (i j : Fin 4), i ≠ r1 ∧ i ≠ r2 ∧ j ≠ c1 ∧ j ≠ c2 ∧ g i j

theorem seven_stars_always_remain_and_six_stars_can_be_removed :
  (∃ (g : Grid), countStars g = 7 ∧
    ∀ (r1 r2 c1 c2 : Fin 4), r1 ≠ r2 → c1 ≠ c2 →
      starExistsAfterRemoval g r1 r2 c1 c2) ∧
  (∀ (g : Grid), countStars g ≤ 6 →
    ∃ (r1 r2 c1 c2 : Fin 4), r1 ≠ r2 ∧ c1 ≠ c2 ∧
      ∀ (i j : Fin 4), (i ≠ r1 ∧ i ≠ r2 ∧ j ≠ c1 ∧ j ≠ c2) → ¬(g i j)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_stars_always_remain_and_six_stars_can_be_removed_l282_28296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_seventh_pair_is_two_ten_l282_28257

/-- Defines the sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Defines the group number for a given pair index -/
def group_number (index : ℕ) : ℕ :=
  (List.range 100).find? (λ n => triangular_number n ≥ index) |>.getD 0

/-- Defines the position within a group for a given pair index -/
def position_in_group (index : ℕ) : ℕ :=
  index - triangular_number (group_number index - 1)

/-- Defines the pair at a given index in the sequence -/
def pair_at_index (index : ℕ) : ℕ × ℕ :=
  let group := group_number index
  let pos := position_in_group index
  (pos, group + 1 - pos)

/-- The main theorem stating that the 57th pair is (2,10) -/
theorem fifty_seventh_pair_is_two_ten :
  pair_at_index 57 = (2, 10) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_seventh_pair_is_two_ten_l282_28257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_difference_theorem_l282_28271

def digits : List Nat := [9, 2, 1, 5]

def largest_number (digits : List Nat) : Nat :=
  (digits.toArray.qsort (fun a b => b < a)).toList.foldl (fun acc d => acc * 10 + d) 0

def least_number (digits : List Nat) : Nat :=
  (digits.toArray.qsort (fun a b => a < b)).toList.foldl (fun acc d => acc * 10 + d) 0

theorem digit_difference_theorem :
  largest_number digits - least_number digits = 8262 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_difference_theorem_l282_28271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dallas_apple_bags_l282_28248

theorem dallas_apple_bags (dallas_pears austin_total : ℕ) 
  (h1 : dallas_pears = 9)
  (h2 : austin_total = 24) : ℕ := by
  let dallas_apples := austin_total - (dallas_pears - 5) - 6
  have : dallas_apples = 14 := by
    -- Proof goes here
    sorry
  exact dallas_apples


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dallas_apple_bags_l282_28248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_minimum_value_l282_28267

-- Define the tangent line function
def tangent_line (a : ℝ) (x : ℝ) : ℝ := a * x + a

-- Define the curve function
noncomputable def curve (b : ℝ) (x : ℝ) : ℝ := Real.log x + b

-- Define the point of tangency
noncomputable def point_of_tangency (x₀ : ℝ) : ℝ × ℝ := (x₀, Real.log x₀)

-- Define the condition for the line to be tangent to the curve
def is_tangent (a b x₀ : ℝ) : Prop :=
  tangent_line a x₀ = curve b x₀ ∧
  a = 1 / x₀

-- Define the expression to be minimized
def expr_to_minimize (a b : ℝ) : ℝ := 5 * a - b

-- State the theorem
theorem tangent_line_minimum_value :
  ∀ a b x₀ : ℝ, x₀ > 0 → is_tangent a b x₀ →
  expr_to_minimize a b ≥ 2 * Real.log 2 := by
  sorry

#check tangent_line_minimum_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_minimum_value_l282_28267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_permutation_l282_28233

def factorial (n : ℕ) : ℕ := Finset.prod (Finset.range n) (λ i => i + 1)

def permutation (n k : ℕ) : ℕ := 
  if k ≤ n then factorial n / factorial (n - k) else 0

theorem product_equals_permutation (n : ℕ) (h : n > 19) :
  Finset.prod (Finset.range 20) (λ i => n - i) = permutation n 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_permutation_l282_28233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_calls_correct_l282_28242

/-- Represents the minimum number of phone calls needed for information sharing. -/
def min_calls (n : ℕ) : ℕ := 2 * n - 2

/-- 
Represents the actual minimum number of calls needed for n people to share all information.
This function is not explicitly defined and serves as a placeholder for the true minimum.
-/
noncomputable def minimum_calls_needed (n : ℕ) : ℕ := sorry

/-- 
Theorem: For n people (n ≥ 1), each knowing one unique piece of information,
the minimum number of phone calls needed for everyone to know everything is 2n - 2,
where in each call the caller shares all their information and the receiver shares nothing.
-/
theorem min_calls_correct (n : ℕ) (h : n ≥ 1) : 
  min_calls n = minimum_calls_needed n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_calls_correct_l282_28242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_sequence_aperiodic_l282_28232

/-- A sequence of daily payments -/
def DailyPayment := ℕ → ℕ

/-- The total payment after n days -/
def TotalPayment (p : DailyPayment) (n : ℕ) : ℕ := (Finset.range n).sum p

/-- A payment sequence satisfying the given condition -/
def ValidPayment (p : DailyPayment) : Prop :=
  (∀ n, p n = 1 ∨ p n = 2) ∧
  (∀ n, |(TotalPayment p n : ℝ) - n * Real.sqrt 2| ≤ 1/2)

/-- A sequence is periodic if it repeats after some point -/
def IsPeriodic (p : DailyPayment) : Prop :=
  ∃ N T, T > 0 ∧ ∀ n ≥ N, p (n + T) = p n

theorem payment_sequence_aperiodic (p : DailyPayment) (h : ValidPayment p) :
  ¬ IsPeriodic p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_sequence_aperiodic_l282_28232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_determine_sphere_l282_28268

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a function to check if 4 points are coplanar
def are_coplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define membership for Point3D in Sphere
instance : Membership Point3D Sphere where
  mem p s := (p.x - s.center.x)^2 + (p.y - s.center.y)^2 + (p.z - s.center.z)^2 = s.radius^2

-- Theorem statement
theorem four_points_determine_sphere (p1 p2 p3 p4 : Point3D) :
  ¬(are_coplanar p1 p2 p3 p4) → ∃! s : Sphere, p1 ∈ s ∧ p2 ∈ s ∧ p3 ∈ s ∧ p4 ∈ s :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_determine_sphere_l282_28268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_can_be_built_l282_28259

/-- Represents a block consisting of 4 unit cubes in the shape shown in the problem -/
structure Block where
  shape : List (Fin 4 × Fin 4 × Fin 4)

/-- Represents a 4 × 4 × 4 cube -/
def Cube : Set (Fin 4 × Fin 4 × Fin 4) := Set.univ

/-- Function to check if a given arrangement of blocks forms a 4 × 4 × 4 cube -/
def forms_cube (arrangement : List Block) : Prop :=
  ∃ (c : Set (Fin 4 × Fin 4 × Fin 4)), c = Cube ∧
    ∀ (x y z : Fin 4), (x, y, z) ∈ c ↔ 
      ∃ (b : Block), b ∈ arrangement ∧ (x, y, z) ∈ b.shape

/-- The main theorem stating that it's possible to build a 4 × 4 × 4 cube from the given blocks -/
theorem cube_can_be_built : ∃ (arrangement : List Block), forms_cube arrangement := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_can_be_built_l282_28259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_transformation_l282_28243

theorem determinant_transformation (a b c d : ℝ) :
  Matrix.det !![a, b; c, d] = 6 →
  Matrix.det !![a, 5*a + 2*b; c, 5*c + 2*d] = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_transformation_l282_28243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l282_28238

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A line passing through a point with nonzero slope -/
structure Line where
  m : ℝ
  slope : ℝ
  h : slope ≠ 0 ∧ m > 3/4

/-- The theorem statement -/
theorem ellipse_and_triangle_properties
  (E : Ellipse)
  (l : Line)
  (h1 : E.a^2 * E.b^2 * 2 = (E.a^2 + E.b^2)) -- Eccentricity condition
  (h2 : ∀ x y : ℝ, y^2 = -4*x → x = -1) -- Parabola focus condition
  (h3 : ∀ A B : ℝ × ℝ, 
        (A.1 - 5/4) * (B.1 - 5/4) + A.2 * B.2 = 
        (l.m^2 + 2) / (l.slope^2 + 2) - 7/16) -- Constant dot product condition
  : (∀ x y : ℝ, x^2/2 + y^2 = 1 ↔ x^2/E.a^2 + y^2/E.b^2 = 1) ∧ 
    (∃ s : ℝ, s = Real.sqrt 2/2 ∧ 
      ∀ A B : ℝ × ℝ, 
        A.1^2/E.a^2 + A.2^2/E.b^2 = 1 → 
        B.1^2/E.a^2 + B.2^2/E.b^2 = 1 → 
        abs ((A.1 * B.2 - A.2 * B.1) / 2) ≤ s) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l282_28238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_hypotenuse_l282_28288

/-- Given a right-angled triangle LMN where M and N are on the horizontal axis
    and L is directly above M, if sin N = 2/3 and LM = 18, then LN = 27 -/
theorem triangle_hypotenuse (L M N : ℝ × ℝ) :
  (M.2 = N.2) →  -- M and N are on the horizontal axis
  (L.1 = M.1) →  -- L is directly above M
  (N.1 - M.1)^2 + (L.2 - M.2)^2 = (N.1 - L.1)^2 + (N.2 - L.2)^2 →  -- LMN is a right-angled triangle
  Real.sin (Real.arctan ((L.2 - M.2) / (N.1 - M.1))) = 2/3 →  -- sin N = 2/3
  Real.sqrt ((L.1 - M.1)^2 + (L.2 - M.2)^2) = 18 →  -- LM = 18
  Real.sqrt ((N.1 - L.1)^2 + (N.2 - L.2)^2) = 27 :=  -- LN = 27
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_hypotenuse_l282_28288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l282_28241

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.cos (2 * α) = 3/7)
  (h2 : Real.cos α < 0)
  (h3 : Real.tan α < 0) : 
  Real.sin α = Real.sqrt 14 / 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l282_28241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l282_28272

def e₁ : ℝ × ℝ := (1, 0)
def e₂ : ℝ × ℝ := (0, 1)
def a : ℝ × ℝ := (3 * e₁.1 - 3 * e₂.1, 3 * e₁.2 - 3 * e₂.2)
def b : ℝ × ℝ := (4 * e₁.1 + e₂.1, 4 * e₁.2 + e₂.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

noncomputable def cos_angle (v w : ℝ × ℝ) : ℝ := (dot_product v w) / (magnitude v * magnitude w)

theorem vector_properties :
  (dot_product a b = 9) ∧
  (magnitude (vector_sum a b) = Real.sqrt 53) ∧
  (cos_angle a b = (3 * Real.sqrt 34) / 34) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l282_28272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_max_volume_ob_length_l282_28265

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a circular cone -/
structure CircularCone where
  vertex : Point3D
  baseCenter : Point3D
  baseRadius : ℝ

/-- Condition: The axial section of the cone is an isosceles right triangle -/
def isIsoscelesRightTriangle (cone : CircularCone) : Prop :=
  sorry

/-- Condition: A point is on the circumference of the base circle -/
def isOnBaseCircumference (cone : CircularCone) (p : Point3D) : Prop :=
  sorry

/-- Condition: A point is within the base circle -/
def isWithinBaseCircle (cone : CircularCone) (p : Point3D) : Prop :=
  sorry

/-- Condition: Two line segments are perpendicular -/
def arePerpendicular (p1 p2 p3 p4 : Point3D) : Prop :=
  sorry

/-- Condition: A point is the midpoint of a line segment -/
def isMidpoint (m p1 p2 : Point3D) : Prop :=
  sorry

/-- The volume of the triangular prism O-HPC -/
def prismVolume (o h p c : Point3D) : ℝ :=
  sorry

/-- Distance between two points -/
def dist (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

theorem cone_max_volume_ob_length 
  (cone : CircularCone)
  (a b h c : Point3D)
  (hCone : isIsoscelesRightTriangle cone)
  (hA : isOnBaseCircumference cone a)
  (hB : isWithinBaseCircle cone b)
  (hPerp1 : arePerpendicular a b cone.baseCenter b)
  (hPerp2 : arePerpendicular cone.baseCenter h cone.vertex b)
  (hPA : dist cone.vertex a = 4)
  (hC : isMidpoint c cone.vertex a)
  (hMaxVol : ∀ (b' : Point3D), isWithinBaseCircle cone b' → 
    prismVolume cone.baseCenter h cone.vertex c ≥ prismVolume cone.baseCenter h' cone.vertex c)
  : dist cone.baseCenter b = 2 * Real.sqrt 6 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_max_volume_ob_length_l282_28265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coronavirus_size_scientific_notation_l282_28213

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem coronavirus_size_scientific_notation :
  toScientificNotation 0.0000012 = ScientificNotation.mk 1.2 (-6) (by sorry) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coronavirus_size_scientific_notation_l282_28213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_properties_l282_28220

noncomputable def ω : ℂ := (3 * Complex.I - 1) / Complex.I

theorem omega_properties : 
  Complex.im ω = 1 ∧ Complex.abs ω = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_properties_l282_28220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_is_sqrt_two_l282_28205

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define point P
def point_P : ℝ × ℝ := (1, 0)

-- Define the intersection points A and B (existence assumed)
axiom exists_intersection_points : ∃ (t₁ t₂ : ℝ), 
  t₁ ≠ t₂ ∧ 
  circle_C (line_l t₁).1 (line_l t₁).2 ∧ 
  circle_C (line_l t₂).1 (line_l t₂).2

-- Theorem statement
theorem distance_difference_is_sqrt_two : 
  ∃ (A B : ℝ × ℝ), 
    (∃ (t₁ t₂ : ℝ), A = line_l t₁ ∧ B = line_l t₂ ∧ 
      circle_C A.1 A.2 ∧ circle_C B.1 B.2) → 
    |dist A point_P - dist B point_P| = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_is_sqrt_two_l282_28205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_l282_28298

/-- The number of sides of the regular polygon -/
def n : ℕ := 12

/-- The radius of the circle in which the polygon is inscribed -/
noncomputable def R : ℝ := Real.sqrt 2

/-- The area of the regular polygon -/
def A : ℝ := 6

/-- Theorem stating that n equals 12 for the given conditions -/
theorem polygon_sides : 
  (∀ (m : ℕ), m > 0 → A = (m : ℝ) * R^2 * Real.sin (2 * Real.pi / m) / 2) → n = 12 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_l282_28298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_squarish_numbers_l282_28215

/-- A function to check if a number is a perfect square in base 8 --/
def isPerfectSquareBase8 (n : Nat) : Prop := sorry

/-- A function to get the digits of a number in base 8 --/
def getBase8Digits (n : Nat) : List Nat := sorry

/-- A function to check if a number is squarish --/
def isSquarish (n : Nat) : Prop :=
  let digits := getBase8Digits n
  (digits.length = 5) ∧
  (∀ d ∈ digits, d ≠ 0) ∧
  (isPerfectSquareBase8 n) ∧
  (isPerfectSquareBase8 (digits[0]! * 8 + digits[1]!)) ∧
  (isPerfectSquareBase8 digits[2]!) ∧
  (isPerfectSquareBase8 (digits[3]! * 8 + digits[4]!)) ∧
  (isPerfectSquareBase8 (digits.sum))

/-- Theorem: There are no squarish numbers --/
theorem no_squarish_numbers : ¬∃ n : Nat, isSquarish n := by
  sorry

#check no_squarish_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_squarish_numbers_l282_28215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_Z_notA_B_subset_A_iff_l282_28293

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≥ 0}

-- Define set B (parametrized by m)
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 2 ≤ x ∧ x ≤ m + 2}

-- Define the complement of A in ℝ
def notA : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_Z_notA : (Set.Icc 0 2 ∩ notA) = {0, 1, 2} := by sorry

theorem B_subset_A_iff (m : ℝ) : B m ⊆ A ↔ m ≤ -3 ∨ m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_Z_notA_B_subset_A_iff_l282_28293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l282_28244

theorem sin_double_angle_special (α : ℝ) :
  (π/2 < α) ∧ (α < π) ∧ (Real.sin (α + π/6) = 1/3) →
  Real.sin (2*α + π/3) = -4*Real.sqrt 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l282_28244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_focal_distance_ratio_l282_28246

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point is on an ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Checks if a point is on a hyperbola -/
def isOnHyperbola (p : Point) (h : Hyperbola) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The ratio of focal distances for intersection points -/
theorem intersection_focal_distance_ratio
  (Q F1 F2 : Point) (e : Ellipse) (h : Hyperbola) :
  isOnEllipse Q e → isOnHyperbola Q h →
  e.a = 7 → h.a = 4 →
  (distance Q F1 - distance Q F2) / (distance Q F1 + distance Q F2) = 4/7 := by
  sorry

#check intersection_focal_distance_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_focal_distance_ratio_l282_28246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt_three_over_two_l282_28223

theorem cos_squared_difference_equals_sqrt_three_over_two :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt_three_over_two_l282_28223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_partition_l282_28255

/-- A triangle represented by its vertices -/
structure Triangle :=
  (a b c : ℝ × ℝ)

/-- Check if a triangle has a specific angle -/
def has_angle (t : Triangle) (x : ℝ) : Prop :=
  sorry -- Actual implementation would go here

/-- Check if a set of triangles covers an equilateral triangle -/
def covers_equilateral (triangles : Finset Triangle) : Prop :=
  sorry -- Actual implementation would go here

/-- A partition of an equilateral triangle into triangles with a specific angle -/
structure TrianglePartition (x : ℝ) :=
  (triangles : Finset Triangle)
  (covers_equilateral : covers_equilateral triangles)
  (all_have_angle : ∀ t ∈ triangles, has_angle t x)

/-- The theorem stating the condition for partitioning an equilateral triangle -/
theorem equilateral_triangle_partition (x : ℝ) :
  (∃ p : TrianglePartition x, True) ↔ (0 < x ∧ x ≤ 120) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_partition_l282_28255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l282_28212

noncomputable def h (x : ℝ) : ℝ := (5 * x - 2) / (x^2 + 2*x - 15)

theorem domain_of_h :
  {x : ℝ | ∃ y, h x = y} = {x : ℝ | x < -5 ∨ (-5 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l282_28212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_at_max_f_l282_28266

noncomputable section

/-- The function f(x) given in the problem -/
def f (x : ℝ) : ℝ := 3 * Real.sin (x + Real.pi / 6)

/-- The theorem stating that when f(x) reaches its maximum at x=θ, tan(θ) = √3 -/
theorem tan_theta_at_max_f (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.tan θ = Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_at_max_f_l282_28266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_characterization_l282_28219

/-- The property described in the problem -/
def has_polynomial_property (n : ℕ) : Prop :=
  ∀ (k : ℕ) (a : Fin k → ℕ),
    (∀ i j : Fin k, i ≠ j → a i % n ≠ a j % n) →
    ∃ (f : Polynomial ℤ),
      (∀ x : ℕ, (∃ i : Fin k, x % n = a i % n) ↔ (f.eval (x : ℤ)) % n = 0) ∧
      (∀ x : ℕ, (f.eval (x : ℤ)) % n = 0 → ∃ i : Fin k, x % n = a i % n)

/-- The main theorem -/
theorem polynomial_property_characterization (n : ℕ) (hn : n ≥ 2) :
  has_polynomial_property n ↔ n = 2 ∨ n = 4 ∨ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_characterization_l282_28219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_town_population_l282_28226

def total_population (males : ℕ) (females_with_glasses : ℕ) (female_glasses_percentage : ℚ) : ℕ :=
  males + (females_with_glasses / female_glasses_percentage.num * female_glasses_percentage.den).toNat

theorem town_population :
  let males := 2000
  let females_with_glasses := 900
  let female_glasses_percentage := 30 / 100
  total_population males females_with_glasses female_glasses_percentage = 5000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_town_population_l282_28226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l282_28202

open Real

-- Define the interval
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ arctan 500 }

-- Define the condition for tan θ > θ
axiom tan_gt_theta : ∀ θ : ℝ, 0 < θ → θ < Real.pi / 2 → tan θ > θ

-- Define the equation
def equation (x : ℝ) : Prop := tan x = tan (2 * x)

-- State the theorem
theorem solution_count : ∃ (S : Finset ℝ), (∀ x ∈ S, x ∈ interval ∧ equation x) ∧ S.card = 159 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l282_28202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_bounce_height_l282_28216

noncomputable def building_height : ℝ := 96

noncomputable def bounce_height (n : ℕ) : ℝ :=
  building_height / (2 ^ n)

theorem fifth_bounce_height :
  bounce_height 5 = 3 := by
  -- Unfold the definitions
  unfold bounce_height building_height
  -- Simplify the expression
  simp [pow_succ]
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_bounce_height_l282_28216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_intersection_l282_28250

noncomputable def α : ℝ := Real.arccos (4/5)

theorem unit_circle_intersection :
  (Real.sin α = 3/5) ∧ 
  (Real.cos α = 4/5) ∧ 
  (Real.tan α = 3/4) ∧ 
  ((Real.sin (α/2) - Real.cos (α/2))^2 = 2/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_intersection_l282_28250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_rate_is_9_point_6_percent_l282_28258

/-- Calculates the total interest rate for a two-share investment --/
noncomputable def calculate_total_interest_rate (total_investment : ℝ) (rate1 : ℝ) (rate2 : ℝ) (amount_at_rate2 : ℝ) : ℝ :=
  let amount_at_rate1 := total_investment - amount_at_rate2
  let interest1 := amount_at_rate1 * rate1
  let interest2 := amount_at_rate2 * rate2
  let total_interest := interest1 + interest2
  (total_interest / total_investment) * 100

/-- Theorem stating that the total interest rate is 9.6% given the specified conditions --/
theorem total_interest_rate_is_9_point_6_percent :
  calculate_total_interest_rate 100000 0.09 0.11 29999.999999999993 = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_rate_is_9_point_6_percent_l282_28258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colonization_combinations_count_l282_28282

/-- The number of Earth-like planets -/
def earth_like_planets : ℕ := 8

/-- The number of Mars-like planets -/
def mars_like_planets : ℕ := 8

/-- The number of colony units required for an Earth-like planet -/
def earth_like_units : ℕ := 3

/-- The number of colony units required for a Mars-like planet -/
def mars_like_units : ℕ := 1

/-- The total number of colony units available -/
def total_units : ℕ := 18

/-- The number of ways to select planets for colonization -/
def colonization_combinations : ℕ := 5124

/-- Theorem stating that the number of ways to select planets for colonization is 5124 -/
theorem colonization_combinations_count :
  (Finset.sum (Finset.range (earth_like_planets + 1)) (fun a : ℕ =>
    if a ≤ earth_like_planets ∧ (total_units - a * earth_like_units) ≤ mars_like_planets
    then Nat.choose earth_like_planets a *
         Nat.choose mars_like_planets (total_units - a * earth_like_units)
    else 0)) = colonization_combinations := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colonization_combinations_count_l282_28282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_with_given_conditions_l282_28200

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  side_length : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- The area of a rhombus is half the product of its diagonals. -/
noncomputable def area (r : Rhombus) : ℝ := (r.diagonal1 * r.diagonal2) / 2

/-- The diagonals of a rhombus are perpendicular bisectors of each other. -/
axiom diagonals_perpendicular_bisectors (r : Rhombus) : 
  r.diagonal1 * r.diagonal2 = 4 * r.side_length^2

theorem rhombus_area_with_given_conditions : 
  ∀ (r : Rhombus), 
    r.side_length = Real.sqrt 89 →
    |r.diagonal1 - r.diagonal2| = 6 →
    area r = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_with_given_conditions_l282_28200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_weight_proof_l282_28290

noncomputable def marco_strawberries : ℚ := 5
noncomputable def marco_blueberries : ℚ := 3
noncomputable def marco_raspberries : ℚ := 6

noncomputable def dad_multiplier : ℚ := 2
noncomputable def sister_multiplier : ℚ := 1/2

noncomputable def total_weight : ℚ := 76

theorem fruit_weight_proof :
  let marco_total := marco_strawberries + marco_blueberries + marco_raspberries
  let dad_total := dad_multiplier * marco_total
  let sister_total := sister_multiplier * marco_total
  dad_total + sister_total = 35 := by
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_weight_proof_l282_28290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l282_28294

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4 / (x + 1)

theorem min_value_of_f :
  ∀ x : ℝ, x > 0 → f x ≥ 4 * Real.sqrt 3 - 3 :=
by
  intro x hx
  -- The proof goes here
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l282_28294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_isosceles_triangle_l282_28253

theorem largest_isosceles_triangle (wire_length : ℝ) (fixed_side : ℝ) 
  (h1 : wire_length = 20) 
  (h2 : fixed_side = 6) 
  (h3 : wire_length > fixed_side) : 
  (wire_length - fixed_side) / 2 = 7 := by
  have remaining_wire : ℝ := wire_length - fixed_side
  have other_side : ℝ := remaining_wire / 2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_isosceles_triangle_l282_28253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wipe_sale_average_decrease_l282_28286

noncomputable def percentDecrease (oldPrice newPrice : ℝ) : ℝ :=
  (oldPrice - newPrice) / oldPrice * 100

noncomputable def average (a b : ℝ) : ℝ :=
  (a + b) / 2

theorem wipe_sale_average_decrease : 
  let smallPackOldPrice := 7 / 3
  let smallPackNewPrice := 5 / 4
  let largePackOldPrice := 8 / 2
  let largePackNewPrice := 9 / 3
  let smallPackDecrease := percentDecrease smallPackOldPrice smallPackNewPrice
  let largePackDecrease := percentDecrease largePackOldPrice largePackNewPrice
  let avgDecrease := average smallPackDecrease largePackDecrease
  ∃ ε > 0, |avgDecrease - 35| < ε :=
by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wipe_sale_average_decrease_l282_28286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_l282_28260

/-- Given vectors a and b, function f, and triangle ABC with specific properties,
    prove that the area of the circumcircle of triangle ABC is 7π. -/
theorem circumcircle_area (x θ A : Real) (a b : Real × Real) :
  a = (Real.sqrt 2 * Real.sin (2 * x), Real.sqrt 2 * Real.cos (2 * x)) →
  b = (Real.cos θ, Real.sin θ) →
  |θ| < π / 2 →
  (fun x ↦ a.1 * b.1 + a.2 * b.2) = (fun x ↦ Real.sqrt 2 * Real.sin (2 * x + π / 6)) →
  (fun x ↦ Real.sqrt 2 * Real.sin (2 * x + π / 6)) A = Real.sqrt 2 →
  5 = 5 →  -- Representing b = 5
  2 * Real.sqrt 3 = 2 * Real.sqrt 3 →  -- Representing c = 2√3
  π * (Real.sqrt 7) ^ 2 = 7 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_l282_28260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l282_28292

def mySequence (n : ℕ) : ℤ :=
  (-1)^(n+1) * 2

theorem sequence_proof (n : ℕ) : 
  (mySequence 1 = 2) ∧ 
  (∀ k : ℕ, k ≥ 1 → mySequence (k + 1) + 2 * mySequence k = 3) →
  ∀ m : ℕ, m ≥ 1 → mySequence m = (-1)^(m+1) * 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l282_28292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_correct_l282_28274

/-- The maximum value of f(x) = (4-3m)x² - 2x + m in the interval [0,1] -/
noncomputable def max_value (m : ℝ) : ℝ :=
  if m < 2/3 then 2 - 2*m else m

/-- The function f(x) = (4-3m)x² - 2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := (4 - 3*m)*x^2 - 2*x + m

theorem max_value_correct (m : ℝ) :
  ∀ x ∈ Set.Icc 0 1, f m x ≤ max_value m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_correct_l282_28274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_section_theorem_l282_28237

/-- Given a triangle ABC and a point D on side BC such that BD = 3DC, 
    prove that AD = (1/4)AB + (3/4)AC -/
theorem section_theorem (A B C D : EuclideanSpace ℝ (Fin 2)) : 
  (D - B) = 3 • (C - D) → 
  (D - A) = (1/4) • (B - A) + (3/4) • (C - A) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_section_theorem_l282_28237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_correct_l282_28214

-- Define the equation
def equation (x : ℝ) : Prop :=
  3 * x / (x - 3) + (3 * x^2 - 45) / x + 3 = 14

-- Define the smallest solution
noncomputable def smallest_solution : ℝ := (1 - Real.sqrt 649) / 12

-- Theorem statement
theorem smallest_solution_correct :
  equation smallest_solution ∧ 
  ∀ y, y < smallest_solution → ¬(equation y) := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_correct_l282_28214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_with_smaller_x_coordinate_l282_28276

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  (x - 5)^2 / 3^2 - (y - 20)^2 / 7^2 = 1

-- Define the focus coordinates
noncomputable def focus_coordinates : ℝ × ℝ := (5 - Real.sqrt 58, 20)

-- Theorem statement
theorem focus_with_smaller_x_coordinate :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ 
  focus_coordinates = (x, y) ∧
  ∀ (x' y' : ℝ), hyperbola_equation x' y' → 
    (x' ≠ x ∨ y' ≠ y) → x ≤ x' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_with_smaller_x_coordinate_l282_28276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_neg_sqrt3_over_2_l282_28270

theorem sin_eq_neg_sqrt3_over_2 (x : ℝ) :
  Real.sin (π / 2 - x) = -Real.sqrt 3 / 2 → π < x → x < 2 * π → x = 7 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_neg_sqrt3_over_2_l282_28270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_perimeter_ratio_l282_28262

/-- The ratio of the perimeter of a polygon formed by cutting a square with a line to the side length of the square -/
theorem square_cut_perimeter_ratio (a : ℝ) (h : a > 0) :
  let square := {(x, y) : ℝ × ℝ | max (abs x) (abs y) = a}
  let line := {(x, y) : ℝ × ℝ | y = 2 * x}
  let polygon := {p : ℝ × ℝ | p ∈ square ∧ (p.2 ≤ 2 * p.1 ∨ p.1 = a)}
  let perimeter := Real.sqrt ((3 * a)^2 + (3 * a)^2) + Real.sqrt ((2 * a)^2 + (4 * a)^2) + a
  perimeter / a = 4 + 2 * Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_perimeter_ratio_l282_28262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broccoli_heads_count_l282_28278

def free_delivery_minimum : ℚ := 35
def chicken_weight : ℚ := 3/2
def chicken_price_per_pound : ℚ := 6
def lettuce_price : ℚ := 3
def cherry_tomatoes_price : ℚ := 5/2
def sweet_potatoes_quantity : ℕ := 4
def sweet_potato_price : ℚ := 3/4
def broccoli_price : ℚ := 2
def brussel_sprouts_price : ℚ := 5/2
def additional_spend_needed : ℚ := 11

def cart_subtotal : ℚ :=
  chicken_weight * chicken_price_per_pound +
  lettuce_price +
  cherry_tomatoes_price +
  (sweet_potatoes_quantity : ℚ) * sweet_potato_price +
  brussel_sprouts_price

def broccoli_heads_in_cart : ℚ :=
  (free_delivery_minimum - cart_subtotal - additional_spend_needed) / broccoli_price

theorem broccoli_heads_count : broccoli_heads_in_cart = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broccoli_heads_count_l282_28278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_arc_length_part2_max_area_l282_28299

-- Define the sector
structure Sector where
  α : ℝ  -- Central angle in radians
  r : ℝ  -- Radius

-- Part 1
def arc_length (s : Sector) : ℝ := s.α * s.r

theorem part1_arc_length :
  let s : Sector := { α := 2 * Real.pi / 3, r := 6 }
  arc_length s = 4 * Real.pi := by sorry

-- Part 2
noncomputable def perimeter (s : Sector) : ℝ := s.α * s.r + 2 * s.r

noncomputable def area (s : Sector) : ℝ := 1/2 * s.α * s.r^2

theorem part2_max_area :
  ∃ (s : Sector), perimeter s = 24 ∧ 
    (∀ (s' : Sector), perimeter s' = 24 → area s' ≤ area s) ∧
    area s = 36 ∧ s.α = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_arc_length_part2_max_area_l282_28299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l282_28261

theorem angle_values (θ : ℝ) (m : ℝ) :
  let P : ℝ × ℝ := (-Real.sqrt 3, m)
  Real.sin θ = (Real.sqrt 2 / 4) * m →
  (m = 0 → Real.cos θ = -1 ∧ Real.tan θ = 0) ∧
  (m = Real.sqrt 5 → Real.cos θ = -(Real.sqrt 6 / 4) ∧ Real.tan θ = -(Real.sqrt 15 / 3)) ∧
  (m = -Real.sqrt 5 → Real.cos θ = -(Real.sqrt 6 / 4) ∧ Real.tan θ = Real.sqrt 15 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l282_28261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_element_in_M_l282_28229

-- Define the set M
def M : Set (ℕ × ℕ) :=
  {p | (1 / Real.sqrt (p.1 : ℝ) - 1 / Real.sqrt (p.2 : ℝ) = 1 / Real.sqrt 45) ∧ 
       p.1 > 0 ∧ p.2 > 0}

-- Theorem statement
theorem unique_element_in_M : ∃! p, p ∈ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_element_in_M_l282_28229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_rep_235_property_l282_28227

def binary_representation (n : ℕ) : List Bool :=
  sorry -- Implementation details omitted for brevity

theorem binary_rep_235_property :
  let bin_235 := binary_representation 235
  let x := (bin_235.filter (· = false)).length
  let y := (bin_235.filter (· = true)).length
  2 * y - 3 * x = 11 := by
    sorry -- Proof details omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_rep_235_property_l282_28227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_assignment_l282_28217

-- Define the types
inductive Student : Type
| boris : Student
| vasily : Student
| nikolay : Student
| peter : Student

inductive Surname : Type
| karpov : Surname
| orlov : Surname
| ivanov : Surname
| krylov : Surname

inductive Course : Type
| first : Course
| second : Course
| third : Course
| fourth : Course

-- Define the problem conditions
def is_stipend_recipient : Student → Prop := sorry
def goes_to_omsk : Student → Prop := sorry
def goes_to_donbass : Surname → Prop := sorry
def is_leningrader : Student → Prop := sorry
def is_leningrader_surname : Surname → Prop := sorry
def finished_school_last_year : Surname → Prop := sorry
def uses_notes_from : Student → Student → Prop := sorry

-- Define the assignment function
def assignment : Student → (Surname × Course) := sorry

-- Theorem statement
theorem student_assignment :
  ∀ (s : Student) (n : Surname) (c : Course),
  (assignment s = (n, c)) →
  (
    (s = Student.boris ∧ n = Surname.karpov ∧ c = Course.third) ∨
    (s = Student.vasily ∧ n = Surname.orlov ∧ c = Course.fourth) ∨
    (s = Student.nikolay ∧ n = Surname.ivanov ∧ c = Course.second) ∨
    (s = Student.peter ∧ n = Surname.krylov ∧ c = Course.first)
  ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_assignment_l282_28217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_five_l282_28247

/-- Represents a 3x3 grid with numbers from 0 to 8 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2.val < q.2.val ∧ q.2.val = p.2.val + 1 ∨
               q.2.val < p.2.val ∧ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val < q.1.val ∧ q.1.val = p.1.val + 1 ∨
               q.1.val < p.1.val ∧ p.1.val = q.1.val + 1))

/-- Checks if the grid satisfies the consecutive number condition -/
def consecutive_condition (g : Grid) : Prop :=
  ∀ p q : Fin 3 × Fin 3, (g p.1 p.2).val + 1 = (g q.1 q.2).val ∨ (g q.1 q.2).val + 1 = (g p.1 p.2).val → adjacent p q

/-- Checks if the sum of corner numbers is 20 -/
def corner_sum_condition (g : Grid) : Prop :=
  (g 0 0).val + (g 0 2).val + (g 2 0).val + (g 2 2).val = 20

/-- The main theorem -/
theorem center_is_five (g : Grid) 
  (h1 : consecutive_condition g)
  (h2 : corner_sum_condition g) : 
  (g 1 1).val = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_five_l282_28247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l282_28225

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 - x + 1)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-2/3) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l282_28225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_param_iff_l282_28281

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A vector parameterization of a line -/
structure VectorParam where
  v : Point  -- initial point
  d : Point  -- direction vector

/-- The line y = 3x + 4 -/
def line (p : Point) : Prop := p.y = 3 * p.x + 4

/-- The slope of a non-vertical vector -/
noncomputable def vectorSlope (v : Point) : ℝ := v.y / v.x

/-- A valid parameterization of the line y = 3x + 4 -/
def valid_param (param : VectorParam) : Prop :=
  line param.v ∧ vectorSlope param.d = 3

theorem valid_param_iff (param : VectorParam) :
  valid_param param ↔ (line param.v ∧ vectorSlope param.d = 3) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_param_iff_l282_28281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l282_28287

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (Real.pi - x) * Real.cos x + 2 * (Real.cos x) ^ 2

noncomputable def g (x : ℝ) : ℝ := f x - 1

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 3 ↔ ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/6) (Real.pi/3) ∧ f x = y) ∧
  (∀ (m : ℝ), (∃! (a b : ℝ), a ∈ Set.Icc (-Real.pi/6) m ∧ b ∈ Set.Icc (-Real.pi/6) m ∧ a ≠ b ∧ g a = 0 ∧ g b = 0) ↔ 
    m ∈ Set.Icc (5*Real.pi/12) (11*Real.pi/12)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l282_28287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l282_28280

theorem divisor_problem : ∃! D : ℕ, 
  D > 0 ∧
  242 % D = 4 ∧
  698 % D = 8 ∧
  354 % D = 5 ∧
  1294 % D = 9 ∧
  D = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l282_28280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l282_28283

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 3^x

-- State the theorem
theorem function_value_theorem :
  ∃ a : ℝ, f a = 1/3 ∧ (a = -1 ∨ a = Real.rpow 3 (1/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l282_28283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_purple_page_difference_l282_28284

theorem orange_purple_page_difference 
  (purple_pages_per_book : ℕ)
  (orange_pages_per_book : ℕ)
  (purple_books_read : ℕ)
  (orange_books_read : ℕ) :
  purple_pages_per_book = 230 →
  orange_pages_per_book = 510 →
  purple_books_read = 5 →
  orange_books_read = 4 →
  orange_pages_per_book * orange_books_read - purple_pages_per_book * purple_books_read = 890 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_purple_page_difference_l282_28284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_shadow_boundary_l282_28236

noncomputable section

-- Define the sphere
def sphere_radius : ℝ := 2
def sphere_center : ℝ × ℝ × ℝ := (0, 0, 2)

-- Define the light source
def light_source : ℝ × ℝ × ℝ := (1, -2, 4)

-- Define the shadow boundary function
def shadow_boundary (x : ℝ) : ℝ := -1/2 * x - 13/2

-- Theorem statement
theorem sphere_shadow_boundary :
  ∀ x y : ℝ,
  (x, y, 0) ∈ {p : ℝ × ℝ × ℝ | ∃ t : ℝ, p = light_source + t • (p - light_source)} ∩
              {p : ℝ × ℝ × ℝ | ‖p - sphere_center‖ = sphere_radius} →
  y = shadow_boundary x :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_shadow_boundary_l282_28236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_l282_28218

-- Define the points and lines
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 1)

def l₁ (x y : ℝ) : Prop := 3 * x - 2 * y + 3 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Define the intersection point C
def C : ℝ × ℝ := (-1, 0)

-- Define the area of a triangle given three points
noncomputable def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  (1/2) * abs ((x₂ - x₁) * (y₃ - y₁) - (x₃ - x₁) * (y₂ - y₁))

-- Theorem statement
theorem intersection_and_area :
  (l₁ C.1 C.2 ∧ l₂ C.1 C.2) ∧ triangle_area A B C = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_l282_28218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_page_number_l282_28210

/-- Represents the count of available '5' digits -/
def available_fives : ℕ := 20

/-- Represents the set of all digits -/
def Digit : Type := Fin 10

/-- Represents the set of digits that are available in unlimited supply -/
def UnlimitedDigits : Set Digit :=
  {⟨0, by norm_num⟩, ⟨1, by norm_num⟩, ⟨2, by norm_num⟩, ⟨3, by norm_num⟩, 
   ⟨4, by norm_num⟩, ⟨6, by norm_num⟩, ⟨7, by norm_num⟩, ⟨8, by norm_num⟩, 
   ⟨9, by norm_num⟩}

/-- Counts the number of '5's needed to represent a natural number -/
def count_fives (n : ℕ) : ℕ :=
  (n.digits 10).count 5

/-- Represents the property that a number can be represented with the available digits -/
def can_represent (n : ℕ) : Prop :=
  count_fives n ≤ available_fives ∧
  ∀ d : Digit, d ∉ UnlimitedDigits → (n.digits 10).count d.val ≤ available_fives

/-- The main theorem stating the maximum page number that can be reached -/
theorem max_page_number :
  ∀ n : ℕ, n ≤ 104 → can_represent n ∧
  ∀ m : ℕ, m > 104 → ¬can_represent m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_page_number_l282_28210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l282_28231

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 4*x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 1 ∨ (1 < x ∧ x < 3) ∨ 3 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l282_28231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_l282_28224

theorem candle_height_ratio (initial_height : ℝ) (h_pos : initial_height > 0) : 
  ∃ t : ℝ, t = 40 / 11 ∧ 
    (initial_height - (initial_height / 5) * t) = 
    3 * (initial_height - (initial_height / 4) * t) :=
by
  -- Define the burn rates and height functions
  let burn_rate1 := initial_height / 5
  let burn_rate2 := initial_height / 4
  let height1 (t : ℝ) := initial_height - burn_rate1 * t
  let height2 (t : ℝ) := initial_height - burn_rate2 * t
  
  -- Prove the existence of t satisfying the conditions
  use 40 / 11
  apply And.intro
  · -- Prove t = 40 / 11
    rfl
  · -- Prove height1 t = 3 * height2 t
    -- This part would require algebraic manipulation
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_l282_28224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_on_axis_l282_28201

-- Define a bounded figure
structure BoundedFigure where
  -- Add necessary properties for a bounded figure
  isBounded : Bool

-- Define an axis of symmetry
structure AxisOfSymmetry where
  -- Add necessary properties for an axis of symmetry

-- Define a center of symmetry
structure CenterOfSymmetry where
  -- Add necessary properties for a center of symmetry

-- Define a predicate to check if a center lies on an axis
def CenterLiesOnAxis (c : CenterOfSymmetry) (a : AxisOfSymmetry) : Prop :=
  sorry -- Define the condition for a center to lie on an axis

-- Define the main theorem
theorem center_on_axis
  (F : BoundedFigure)
  (A : AxisOfSymmetry)
  (C : CenterOfSymmetry)
  (h1 : F.isBounded)
  (h2 : ∃ (a : AxisOfSymmetry), a = A) -- Figure has an axis of symmetry
  (h3 : ∃ (c : CenterOfSymmetry), c = C) -- Figure has a center of symmetry
  : CenterLiesOnAxis C A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_on_axis_l282_28201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_neg_eight_range_of_a_min_value_of_expression_l282_28209

-- Define the inequality function
def f (x a : ℝ) : Prop := |x - 3| + |x + 2| ≤ |a + 1|

-- Theorem 1: Solution set when a = -8
theorem solution_set_a_neg_eight :
  {x : ℝ | f x (-8)} = Set.Icc (-3) 4 := by sorry

-- Theorem 2: Range of a for which the inequality has solutions
theorem range_of_a :
  {a : ℝ | ∃ x, f x a} = Set.Ici 4 ∪ Set.Iic (-6) := by sorry

-- Helper theorem: Minimum value of |x-3|+|x+2|
theorem min_value_of_expression :
  ∀ x : ℝ, |x - 3| + |x + 2| ≥ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_neg_eight_range_of_a_min_value_of_expression_l282_28209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_greater_than_one_l282_28239

-- Define the logarithm function with base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := lg ((2 / (1 - x)) + a)

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem range_of_f_greater_than_one (a : ℝ) :
  (is_odd_function (f a)) →
  (∀ x, f a x > 1 ↔ 9/11 < x ∧ x < 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_greater_than_one_l282_28239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_rotation_area_l282_28211

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle given side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Represents a rotation in 2D space -/
def rotate90Clockwise (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Calculates the centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Calculates the area of the union of two triangles -/
noncomputable def unionArea (t1 t2 : Triangle) : ℝ := sorry

/-- Main theorem -/
theorem isosceles_triangle_rotation_area (t : Triangle) :
  t.A.1 = 0 ∧ t.A.2 = 0 ∧   -- Assume A is at origin for simplicity
  t.B.1^2 + t.B.2^2 = 17^2 ∧
  t.C.1^2 + t.C.2^2 = 17^2 ∧
  (t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2 = 16^2 →
  let g := centroid t
  let t' := Triangle.mk
    (rotate90Clockwise t.A g)
    (rotate90Clockwise t.B g)
    (rotate90Clockwise t.C g)
  unionArea t t' = 240 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_rotation_area_l282_28211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l282_28264

/-- The ellipse C with its properties -/
structure Ellipse :=
  (a : ℝ) (b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The intersection point of lines AB and CF -/
def intersection_point (e : Ellipse) : ℝ × ℝ := (3 * e.a, 16)

/-- Theorem: Standard equation of the ellipse and constant sum of squared distances -/
theorem ellipse_properties (e : Ellipse) :
  (∀ x y : ℝ, ellipse_equation e x y ↔ x^2 / 25 + y^2 / 16 = 1) ∧
  (∀ m : ℝ, ∃ s t : ℝ × ℝ,
    let p := (m, 0)
    let line_equation := λ (x y : ℝ) ↦ y = 4/5 * (x - m)
    ellipse_equation e s.1 s.2 ∧
    ellipse_equation e t.1 t.2 ∧
    line_equation s.1 s.2 ∧
    line_equation t.1 t.2 ∧
    (s.1 - p.1)^2 + (s.2 - p.2)^2 + (t.1 - p.1)^2 + (t.2 - p.2)^2 = 41) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l282_28264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_shift_l282_28279

theorem cosine_shift (x : ℝ) :
  Real.cos (1/2 * x + π/6) = Real.cos (1/2 * (x + π/3)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_shift_l282_28279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l282_28277

-- Define the function f
noncomputable def f (x : ℝ) := Real.cos x ^ 2 + Real.sin x

-- State the theorem
theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ -1) ∧ (∃ x : ℝ, f x = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l282_28277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_y_l282_28235

-- Define the function f
def f (x : ℝ) : ℝ := x + 2

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc 1 9

-- Define the function y
def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- Theorem statement
theorem domain_and_range_of_y :
  {x : ℝ | x ∈ dom_f ∧ x^2 ∈ dom_f} = Set.Icc 1 3 ∧
  Set.range (fun x => y x) = Set.Icc 12 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_y_l282_28235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l282_28208

noncomputable def line1 (a : ℝ) (s : ℝ) : ℝ × ℝ × ℝ := (2 + s*a, -1 + s*(-3), 0 + s*2)
noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ := (1 + t*2, -3/2 + t*1, -5 + t*3)

def direction_vector1 (a : ℝ) : ℝ × ℝ × ℝ := (a, -3, 2)
def direction_vector2 : ℝ × ℝ × ℝ := (2, 1, 3)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  v1 * w1 + v2 * w2 + v3 * w3

theorem perpendicular_lines (a : ℝ) :
  dot_product (direction_vector1 a) direction_vector2 = 0 ↔ a = -3/2 :=
by sorry

#check perpendicular_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l282_28208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l282_28254

noncomputable def A (a : ℝ) : ℝ × ℝ × ℝ := (-2, -2*a, 7)
noncomputable def B (a : ℝ) : ℝ × ℝ × ℝ := (a+1, a+4, 2)
noncomputable def M (a : ℝ) : ℝ × ℝ := (a^2/4, a)
def F : ℝ × ℝ := (1, 0)

def p (a : ℝ) : Prop := dist (A a) (B a) < 3 * Real.sqrt 10
def q (a : ℝ) : Prop := dist (M a) F > 2

theorem range_of_a :
  ∀ a : ℝ, (¬(¬(p a)) ∧ ¬(p a ∧ q a)) → a ∈ Set.Icc (-2) 1 :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l282_28254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_for_views_l282_28222

/-- Represents a 3D grid of unit cubes -/
def CubeGrid := List (List (List Bool))

/-- Checks if a cube exists at the given coordinates -/
def hasCube (grid : CubeGrid) (x y z : Nat) : Bool :=
  match grid.get? x with
  | none => false
  | some plane => match plane.get? y with
    | none => false
    | some column => match column.get? z with
      | none => false
      | some cube => cube

/-- Checks if the grid satisfies the condition that each cube shares a face -/
def sharesface (grid : CubeGrid) : Bool :=
  sorry

/-- Generates the front view of the grid -/
def frontView (grid : CubeGrid) : List (List Bool) :=
  sorry

/-- Generates the side view of the grid -/
def sideView (grid : CubeGrid) : List (List Bool) :=
  sorry

/-- The given front view -/
def givenFrontView : List (List Bool) :=
  [[true, true, true], [true, true, false], [true, false, false]]

/-- The given side view -/
def givenSideView : List (List Bool) :=
  [[true, true, true], [true, true, false], [true, false, false]]

/-- Counts the number of cubes in the grid -/
def countCubes (grid : CubeGrid) : Nat :=
  sorry

theorem min_cubes_for_views :
  ∃ (grid : CubeGrid),
    sharesface grid ∧
    frontView grid = givenFrontView ∧
    sideView grid = givenSideView ∧
    countCubes grid = 5 ∧
    (∀ (grid' : CubeGrid),
      sharesface grid' ∧
      frontView grid' = givenFrontView ∧
      sideView grid' = givenSideView →
      countCubes grid' ≥ 5) :=
by
  sorry

#check min_cubes_for_views

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_for_views_l282_28222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_l282_28245

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f(x) = x + [x]
noncomputable def f (x : ℝ) : ℝ := x + (floor x)

-- Theorem stating the properties of f(x)
theorem properties_of_f :
  (∀ x : ℝ, ∃ y : ℝ, f y = x) ∧  -- Range is ℝ
  (¬ ∀ x : ℝ, f (-x) = -f x) ∧   -- Not an odd function
  (¬ ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧  -- Not periodic
  (∀ x y : ℝ, x < y → f x < f y) -- Increasing function
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_l282_28245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pentagonal_pyramid_properties_l282_28256

/-- A right pyramid with a regular pentagonal base inscribed in a circle and equilateral triangular lateral faces -/
structure RightPentagonalPyramid where
  /-- Radius of the circle inscribing the base pentagon -/
  base_radius : ℝ
  /-- The base is a regular pentagon -/
  base_is_regular_pentagon : True
  /-- The lateral faces are equilateral triangles -/
  faces_are_equilateral : True

/-- Calculate the total surface area of the pyramid -/
noncomputable def total_surface_area (p : RightPentagonalPyramid) : ℝ :=
  11.25 * Real.sqrt (10 - 2 * Real.sqrt 5) * (1 + Real.sqrt 5 + Real.sqrt (30 - 6 * Real.sqrt 5))

/-- Calculate the volume of the pyramid -/
noncomputable def volume (p : RightPentagonalPyramid) : ℝ :=
  45 * Real.sqrt (10 - 2 * Real.sqrt 5)

/-- Theorem: The total surface area and volume of the pyramid are as calculated -/
theorem right_pentagonal_pyramid_properties (p : RightPentagonalPyramid) 
  (h : p.base_radius = 6) : 
  total_surface_area p = 11.25 * Real.sqrt (10 - 2 * Real.sqrt 5) * (1 + Real.sqrt 5 + Real.sqrt (30 - 6 * Real.sqrt 5)) ∧ 
  volume p = 45 * Real.sqrt (10 - 2 * Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pentagonal_pyramid_properties_l282_28256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_shift_quadrant_l282_28295

open Real

-- Define the fourth quadrant
def fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, α > 2 * k * Real.pi - Real.pi / 2 ∧ α < 2 * k * Real.pi

-- Define the second quadrant
def second_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, α > 2 * k * Real.pi + Real.pi / 2 ∧ α < 2 * k * Real.pi + Real.pi

-- Theorem statement
theorem angle_shift_quadrant (α : ℝ) :
  fourth_quadrant α → second_quadrant (Real.pi + α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_shift_quadrant_l282_28295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_minus_one_l282_28275

theorem power_of_three_minus_one :
  (∃! p : ℕ × ℕ, p.1^3 = 3^p.2 - 1 ∧ p = (2, 3)) ∧
  (∀ n : ℕ, n > 1 ∧ n ≠ 3 → ¬∃ (x k : ℕ), x^n = 3^k - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_minus_one_l282_28275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_parameter_range_l282_28269

open Real Set

theorem function_inequality_implies_parameter_range 
  (f g : ℝ → ℝ) 
  (a : ℝ) 
  (hf : ∀ x, f x = sin (2 * x) - 1) 
  (hg : ∀ x, g x = 2 * a * (sin x + cos x) - 4 * a * x) 
  (h_exists : ∃ x ∈ Icc 0 (π / 2), f x ≥ (deriv (deriv g)) x) :
  a ∈ Ici (1 / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_parameter_range_l282_28269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l282_28228

-- Define the slope range
def slope_range : Set ℝ := { k | -1 ≤ k ∧ k ≤ Real.sqrt 3 }

-- Define the inclination angle range
def inclination_range : Set ℝ := { α | (0 ≤ α ∧ α ≤ Real.pi/3) ∨ (3*Real.pi/4 ≤ α ∧ α < Real.pi) }

-- Theorem statement
theorem inclination_angle_range (k : ℝ) (α : ℝ) :
  k ∈ slope_range → α = Real.arctan k → α ∈ inclination_range := by
  sorry

#check inclination_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l282_28228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_women_not_speaking_french_l282_28204

theorem percentage_women_not_speaking_french 
  (total_employees : ℝ) 
  (h1 : total_employees > 0) 
  (percent_men : ℝ)
  (percent_french_speakers : ℝ)
  (percent_men_french_speakers : ℝ)
  (h2 : total_employees * 0.7 = total_employees * (percent_men / 100)) 
  (h3 : total_employees * 0.4 = total_employees * (percent_french_speakers / 100)) 
  (h4 : total_employees * 0.7 * 0.5 = total_employees * (percent_men / 100) * (percent_men_french_speakers / 100)) :
  let women := total_employees * (1 - percent_men / 100)
  let women_not_french := women - (total_employees * (percent_french_speakers / 100) - total_employees * (percent_men / 100) * (percent_men_french_speakers / 100))
  ∃ ε > 0, abs ((women_not_french / women) * 100 - 83.33) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_women_not_speaking_french_l282_28204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_percentage_l282_28234

def shoes_price : ℝ := 95
def cap_price : ℝ := 25
def shoes_discount : ℝ := 0.25
def cap_discount : ℝ := 0.60

theorem total_savings_percentage :
  let total_original_price := shoes_price + cap_price
  let shoes_savings := shoes_price * shoes_discount
  let cap_savings := cap_price * cap_discount
  let total_savings := shoes_savings + cap_savings
  let savings_percentage := (total_savings / total_original_price) * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |savings_percentage - 32.29| < ε := by
  -- The proof goes here
  sorry

#eval (((95 * 0.25 + 25 * 0.60) / (95 + 25)) * 100 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_percentage_l282_28234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clara_receives_52_50_l282_28252

/-- The amount Clara receives from selling state quarters to a collector -/
def claras_payment (face_value : ℚ) (num_quarters : ℕ) (price_multiplier : ℚ) : ℚ :=
  face_value * num_quarters * price_multiplier

/-- Theorem: Clara receives $52.5 for her seven state quarters -/
theorem clara_receives_52_50 :
  claras_payment (1/4) 7 30 = 105/2 := by
  -- Unfold the definition of claras_payment
  unfold claras_payment
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clara_receives_52_50_l282_28252
