import Mathlib

namespace NUMINAMATH_CALUDE_coffee_table_price_is_330_l2547_254792

/-- Represents the living room set purchase -/
structure LivingRoomSet where
  sofa_price : ℕ
  armchair_price : ℕ
  num_armchairs : ℕ
  total_invoice : ℕ

/-- Calculates the price of the coffee table -/
def coffee_table_price (set : LivingRoomSet) : ℕ :=
  set.total_invoice - (set.sofa_price + set.armchair_price * set.num_armchairs)

/-- Theorem stating that the coffee table price is 330 -/
theorem coffee_table_price_is_330 (set : LivingRoomSet) 
  (h1 : set.sofa_price = 1250)
  (h2 : set.armchair_price = 425)
  (h3 : set.num_armchairs = 2)
  (h4 : set.total_invoice = 2430) :
  coffee_table_price set = 330 := by
  sorry

#check coffee_table_price_is_330

end NUMINAMATH_CALUDE_coffee_table_price_is_330_l2547_254792


namespace NUMINAMATH_CALUDE_negation_equivalence_l2547_254784

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 1 < 0) ↔ (∀ x : ℝ, x^2 - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2547_254784


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2547_254723

/-- Given an ellipse C with semi-major axis a and semi-minor axis b,
    and a circle with diameter 2a tangent to a line,
    prove that the eccentricity of C is √(6)/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let L := {(x, y) : ℝ × ℝ | b * x - a * y + 2 * a * b = 0}
  let circle_diameter := 2 * a
  (∃ (p : ℝ × ℝ), p ∈ C ∧ p ∈ L) →  -- The circle is tangent to the line
  let e := Real.sqrt (1 - b^2 / a^2)  -- Eccentricity definition
  e = Real.sqrt 6 / 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2547_254723


namespace NUMINAMATH_CALUDE_length_OP_specific_case_l2547_254706

/-- Given a circle with center O and radius r, and two intersecting chords AB and CD,
    this function calculates the length of OP, where P is the intersection point of the chords. -/
def length_OP (r : ℝ) (chord_AB : ℝ) (chord_CD : ℝ) (midpoint_distance : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for a circle with radius 20 and two intersecting chords of lengths 24 and 18,
    if the distance between their midpoints is 10, then the length of OP is approximately 14.8. -/
theorem length_OP_specific_case :
  let r := 20
  let chord_AB := 24
  let chord_CD := 18
  let midpoint_distance := 10
  ∃ ε > 0, |length_OP r chord_AB chord_CD midpoint_distance - 14.8| < ε :=
by sorry

end NUMINAMATH_CALUDE_length_OP_specific_case_l2547_254706


namespace NUMINAMATH_CALUDE_parking_theorem_l2547_254793

/-- The number of parking spaces in a row -/
def total_spaces : ℕ := 7

/-- The number of cars to be parked -/
def num_cars : ℕ := 4

/-- The number of consecutive empty spaces required -/
def consecutive_empty : ℕ := 3

/-- The number of different parking arrangements -/
def parking_arrangements : ℕ := 120

/-- Theorem stating that the number of ways to arrange 4 cars and 3 consecutive
    empty spaces in a row of 7 parking spaces is equal to 120 -/
theorem parking_theorem :
  (total_spaces = 7) →
  (num_cars = 4) →
  (consecutive_empty = 3) →
  (parking_arrangements = 120) :=
by sorry

end NUMINAMATH_CALUDE_parking_theorem_l2547_254793


namespace NUMINAMATH_CALUDE_difference_of_squares_l2547_254778

theorem difference_of_squares : (502 : ℤ) * 502 - (501 : ℤ) * 503 = 1 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2547_254778


namespace NUMINAMATH_CALUDE_circle_properties_l2547_254718

-- Define the circle equation type
def CircleEquation := ℝ → ℝ → ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the properties of the circles and points
def CircleProperties (f : CircleEquation) (P₁ P₂ : Point) :=
  f P₁.x P₁.y = 0 ∧ f P₂.x P₂.y ≠ 0

-- Define the new circle equation
def NewCircleEquation (f : CircleEquation) (P₁ P₂ : Point) : CircleEquation :=
  fun x y => f x y - f P₁.x P₁.y - f P₂.x P₂.y

-- Theorem statement
theorem circle_properties
  (f : CircleEquation)
  (P₁ P₂ : Point)
  (h : CircleProperties f P₁ P₂) :
  let g := NewCircleEquation f P₁ P₂
  (g P₂.x P₂.y = 0) ∧
  (∀ x y, g x y = 0 → f x y = f P₂.x P₂.y) := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l2547_254718


namespace NUMINAMATH_CALUDE_janes_shadow_length_l2547_254794

/-- Given a tree and a person (Jane) casting shadows, this theorem proves
    the length of Jane's shadow based on the heights of the tree and Jane,
    and the length of the tree's shadow. -/
theorem janes_shadow_length
  (tree_height : ℝ)
  (tree_shadow : ℝ)
  (jane_height : ℝ)
  (h_tree_height : tree_height = 30)
  (h_tree_shadow : tree_shadow = 10)
  (h_jane_height : jane_height = 1.5) :
  jane_height * tree_shadow / tree_height = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_janes_shadow_length_l2547_254794


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l2547_254738

theorem smallest_undefined_inverse (a : ℕ) : 
  (a > 0) → 
  (¬ ∃ (x : ℕ), x * a ≡ 1 [MOD 77]) → 
  (¬ ∃ (y : ℕ), y * a ≡ 1 [MOD 91]) → 
  (∀ (b : ℕ), b > 0 ∧ b < a → 
    (∃ (x : ℕ), x * b ≡ 1 [MOD 77]) ∨ 
    (∃ (y : ℕ), y * b ≡ 1 [MOD 91])) → 
  a = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l2547_254738


namespace NUMINAMATH_CALUDE_margaux_lending_problem_l2547_254791

/-- Margaux's money lending problem -/
theorem margaux_lending_problem (brother_payment cousin_payment total_days total_collection : ℕ) 
  (h1 : brother_payment = 8)
  (h2 : cousin_payment = 4)
  (h3 : total_days = 7)
  (h4 : total_collection = 119) :
  ∃ (friend_payment : ℕ), 
    friend_payment * total_days + brother_payment * total_days + cousin_payment * total_days = total_collection ∧ 
    friend_payment = 5 := by
  sorry

end NUMINAMATH_CALUDE_margaux_lending_problem_l2547_254791


namespace NUMINAMATH_CALUDE_motion_equation_l2547_254737

/-- Given a point's rectilinear motion with velocity v(t) = t^2 - 8t + 3,
    prove that its displacement function s(t) satisfies
    s(t) = t^3/3 - 4t^2 + 3t + C for some constant C. -/
theorem motion_equation (v : ℝ → ℝ) (s : ℝ → ℝ) :
  (∀ t, v t = t^2 - 8*t + 3) →
  (∀ t, (deriv s) t = v t) →
  ∃ C, ∀ t, s t = t^3/3 - 4*t^2 + 3*t + C :=
sorry

end NUMINAMATH_CALUDE_motion_equation_l2547_254737


namespace NUMINAMATH_CALUDE_total_shaded_area_is_one_third_l2547_254736

/-- Represents the fractional area shaded in each step of the square division pattern. -/
def shadedAreaSequence : ℕ → ℚ
  | 0 => 1/4
  | n + 1 => (1/4) * shadedAreaSequence n

/-- The sum of the infinite geometric series representing the total shaded area. -/
noncomputable def totalShadedArea : ℚ := ∑' n, shadedAreaSequence n

/-- Theorem stating that the total shaded area is equal to 1/3. -/
theorem total_shaded_area_is_one_third :
  totalShadedArea = 1/3 := by sorry

end NUMINAMATH_CALUDE_total_shaded_area_is_one_third_l2547_254736


namespace NUMINAMATH_CALUDE_first_coordinate_on_line_l2547_254761

theorem first_coordinate_on_line (n : ℝ) (a : ℝ) :
  (a = 4 * n + 5 ∧ a + 2 = 4 * (n + 0.5) + 5) → a = 4 * n + 5 :=
by sorry

end NUMINAMATH_CALUDE_first_coordinate_on_line_l2547_254761


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2547_254763

def a : Fin 2 → ℝ := ![3, -2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 4]

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), k ≠ 0 ∧ b x = k • a) → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2547_254763


namespace NUMINAMATH_CALUDE_three_planes_max_parts_l2547_254759

/-- The maximum number of parts into which three planes can divide three-dimensional space -/
def max_parts_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of parts into which three planes can divide three-dimensional space is 8 -/
theorem three_planes_max_parts :
  max_parts_three_planes = 8 := by sorry

end NUMINAMATH_CALUDE_three_planes_max_parts_l2547_254759


namespace NUMINAMATH_CALUDE_unpainted_cubes_6x6x6_l2547_254752

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  painted_area : Nat
  total_cubes : Nat

/-- Calculate the number of unpainted cubes in a PaintedCube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  cube.total_cubes - (6 * cube.painted_area - 4 * cube.painted_area + 8)

/-- Theorem: In a 6x6x6 cube with central 4x4 areas painted, there are 160 unpainted cubes -/
theorem unpainted_cubes_6x6x6 :
  let cube : PaintedCube := { size := 6, painted_area := 16, total_cubes := 216 }
  unpainted_cubes cube = 160 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_6x6x6_l2547_254752


namespace NUMINAMATH_CALUDE_alice_above_quota_l2547_254758

def alice_sales (adidas_price nike_price reebok_price : ℕ)
                (adidas_qty nike_qty reebok_qty : ℕ)
                (quota : ℕ) : ℤ :=
  (adidas_price * adidas_qty + nike_price * nike_qty + reebok_price * reebok_qty) - quota

theorem alice_above_quota :
  alice_sales 45 60 35 6 8 9 1000 = 65 := by
  sorry

end NUMINAMATH_CALUDE_alice_above_quota_l2547_254758


namespace NUMINAMATH_CALUDE_g_range_l2547_254730

noncomputable def g (x : ℝ) : ℝ := (Real.arccos x)^4 + (Real.arcsin x)^4

theorem g_range : 
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, 
  ∃ y ∈ Set.Icc (Real.pi^4 / 16) ((3 * Real.pi^4) / 32), 
  g x = y ∧ 
  ∀ z, g x = z → z ∈ Set.Icc (Real.pi^4 / 16) ((3 * Real.pi^4) / 32) :=
sorry

end NUMINAMATH_CALUDE_g_range_l2547_254730


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l2547_254721

theorem arithmetic_simplification :
  2 - (-3) - 6 - (-8) - 10 - (-12) = 9 := by sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l2547_254721


namespace NUMINAMATH_CALUDE_max_candies_is_18_l2547_254753

/-- Represents the candy store's pricing structure and discount policy. -/
structure CandyStore where
  individual_price : ℕ
  pack4_price : ℕ
  pack7_price : ℕ
  double_pack7_discount : ℕ

/-- Calculates the maximum number of candies that can be bought with a given amount of money. -/
def max_candies (store : CandyStore) (budget : ℕ) : ℕ :=
  sorry

/-- The theorem states that with $25 and the given pricing structure, 
    the maximum number of candies that can be bought is 18. -/
theorem max_candies_is_18 : 
  let store : CandyStore := {
    individual_price := 2,
    pack4_price := 6,
    pack7_price := 10,
    double_pack7_discount := 3
  }
  max_candies store 25 = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_is_18_l2547_254753


namespace NUMINAMATH_CALUDE_geometric_sequence_in_arithmetic_progression_l2547_254757

theorem geometric_sequence_in_arithmetic_progression (x : ℚ) (hx : x > 0) :
  ∃ (i j k : ℕ), i < j ∧ j < k ∧ (x + i) * (x + k) = (x + j)^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_in_arithmetic_progression_l2547_254757


namespace NUMINAMATH_CALUDE_unique_ages_solution_l2547_254725

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem unique_ages_solution :
  ∃! (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a = 2 * b ∧
    b = c - 7 ∧
    is_prime (a + b + c) ∧
    a + b + c < 70 ∧
    sum_of_digits (a + b + c) = 13 ∧
    a = 30 ∧ b = 15 ∧ c = 22 :=
sorry

end NUMINAMATH_CALUDE_unique_ages_solution_l2547_254725


namespace NUMINAMATH_CALUDE_triangle_area_sum_l2547_254712

theorem triangle_area_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + y^2 = 3^2)
  (h2 : y^2 + y*z + z^2 = 4^2)
  (h3 : x^2 + Real.sqrt 3 * x*z + z^2 = 5^2) :
  2*x*y + x*z + Real.sqrt 3 * y*z = 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_sum_l2547_254712


namespace NUMINAMATH_CALUDE_section_b_average_weight_l2547_254727

/-- Proves that the average weight of section B is 30 kg given the conditions of the problem -/
theorem section_b_average_weight 
  (students_a : ℕ) 
  (students_b : ℕ) 
  (total_students : ℕ) 
  (avg_weight_a : ℝ) 
  (avg_weight_total : ℝ) 
  (h1 : students_a = 36)
  (h2 : students_b = 24)
  (h3 : total_students = students_a + students_b)
  (h4 : avg_weight_a = 30)
  (h5 : avg_weight_total = 30) :
  (total_students * avg_weight_total - students_a * avg_weight_a) / students_b = 30 :=
by sorry

end NUMINAMATH_CALUDE_section_b_average_weight_l2547_254727


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2547_254772

theorem systematic_sampling_interval 
  (total_items : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_items = 2005)
  (h2 : sample_size = 20) :
  ∃ (removed : ℕ) (interval : ℕ),
    removed < sample_size ∧
    interval * sample_size = total_items - removed ∧
    interval = 100 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2547_254772


namespace NUMINAMATH_CALUDE_sound_propagation_all_directions_l2547_254768

/-- Represents the medium through which sound travels -/
inductive Medium
| Air
| Water
| Solid

/-- Represents a direction in 3D space -/
structure Direction where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents sound as a mechanical wave -/
structure Sound where
  medium : Medium
  frequency : ℝ
  amplitude : ℝ

/-- Represents the propagation of sound in a medium -/
def SoundPropagation (s : Sound) (d : Direction) : Prop :=
  match s.medium with
  | Medium.Air => true
  | Medium.Water => true
  | Medium.Solid => true

/-- Theorem stating that sound propagates in all directions in a classroom -/
theorem sound_propagation_all_directions 
  (s : Sound) 
  (h1 : s.medium = Medium.Air) 
  (h2 : ∀ (d : Direction), SoundPropagation s d) : 
  ∀ (d : Direction), SoundPropagation s d :=
sorry

end NUMINAMATH_CALUDE_sound_propagation_all_directions_l2547_254768


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l2547_254701

-- Define the necessary types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (belongs_to : Point → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (α β : Plane) (l n : Line) :
  parallel α β →
  perpendicular l α →
  contained_in n β →
  perpendicular_lines l n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l2547_254701


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2547_254799

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation y = -2x + 1 in slope-intercept form. -/
theorem perpendicular_line_equation (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 6 * y = 9
  let P : ℝ × ℝ := (2, -3)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = -2 * x + 1
  (∀ x y, L1 x y ↔ y = (1/2) * x - (3/2)) →
  (L2 P.1 P.2) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (y₂ - y₁) * (x₂ - x₁) = -1 / ((y₂ - y₁) / (x₂ - x₁))) →
  ∀ x y, L2 x y ↔ y = -2 * x + 1 :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l2547_254799


namespace NUMINAMATH_CALUDE_annika_hikes_four_km_l2547_254735

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  flatRate : ℝ  -- Rate on flat terrain in minutes per kilometer
  initialDistance : ℝ  -- Initial distance hiked east in kilometers
  totalTime : ℝ  -- Total time available for the round trip in minutes
  uphillDistance : ℝ  -- Distance of uphill section in kilometers
  uphillRate : ℝ  -- Rate on uphill section in minutes per kilometer
  downhillDistance : ℝ  -- Distance of downhill section in kilometers
  downhillRate : ℝ  -- Rate on downhill section in minutes per kilometer

/-- Calculates the total distance hiked east given the hiking scenario -/
def totalDistanceEast (scenario : HikingScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, Annika will hike 4 km east -/
theorem annika_hikes_four_km : 
  let scenario : HikingScenario := {
    flatRate := 10,
    initialDistance := 2.75,
    totalTime := 45,
    uphillDistance := 0.5,
    uphillRate := 15,
    downhillDistance := 0.5,
    downhillRate := 5
  }
  totalDistanceEast scenario = 4 := by
  sorry

end NUMINAMATH_CALUDE_annika_hikes_four_km_l2547_254735


namespace NUMINAMATH_CALUDE_cube_volume_from_circumscribed_sphere_l2547_254785

theorem cube_volume_from_circumscribed_sphere (V_sphere : ℝ) :
  V_sphere = (32 / 3) * Real.pi →
  ∃ (V_cube : ℝ), V_cube = (64 * Real.sqrt 3) / 9 ∧ 
  (∃ (a : ℝ), V_cube = a^3 ∧ V_sphere = (4 / 3) * Real.pi * ((a * Real.sqrt 3) / 2)^3) :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_from_circumscribed_sphere_l2547_254785


namespace NUMINAMATH_CALUDE_pythagorean_triple_check_l2547_254779

def isPythagoreanTriple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem pythagorean_triple_check :
  ¬(isPythagoreanTriple 2 3 4) ∧
  (isPythagoreanTriple 3 4 5) ∧
  (isPythagoreanTriple 6 8 10) ∧
  (isPythagoreanTriple 5 12 13) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_check_l2547_254779


namespace NUMINAMATH_CALUDE_inequality_proof_l2547_254782

theorem inequality_proof (p q r : ℝ) (n : ℕ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0) 
  (h_product : p * q * r = 1) : 
  1 / (p^n + q^n + 1) + 1 / (q^n + r^n + 1) + 1 / (r^n + p^n + 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2547_254782


namespace NUMINAMATH_CALUDE_external_internal_triangles_form_parallelogram_l2547_254733

-- Define the basic structures
structure Point :=
  (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Quadrilateral :=
  (A B C D : Point)

-- Define the properties
def isEquilateral (t : Triangle) : Prop :=
  sorry

def areSimilar (t1 t2 : Triangle) : Prop :=
  sorry

def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

def constructedExternally (base outer : Triangle) : Prop :=
  sorry

def constructedInternally (base inner : Triangle) : Prop :=
  sorry

-- State the theorem
theorem external_internal_triangles_form_parallelogram
  (ABC : Triangle)
  (AB₁C AC₁B BA₁C : Triangle)
  (ABB₁AC₁ : Quadrilateral) :
  isEquilateral AB₁C ∧
  isEquilateral AC₁B ∧
  areSimilar AB₁C ABC ∧
  areSimilar AC₁B ABC ∧
  constructedExternally ABC AB₁C ∧
  constructedExternally ABC AC₁B ∧
  constructedInternally ABC BA₁C ∧
  ABB₁AC₁.A = ABC.A ∧
  ABB₁AC₁.B = AB₁C.B ∧
  ABB₁AC₁.C = AC₁B.C ∧
  ABB₁AC₁.D = BA₁C.A →
  isParallelogram ABB₁AC₁ :=
sorry

end NUMINAMATH_CALUDE_external_internal_triangles_form_parallelogram_l2547_254733


namespace NUMINAMATH_CALUDE_ten_thousand_one_divides_same_first_last_four_digits_l2547_254767

-- Define an 8-digit number type
def EightDigitNumber := { n : ℕ // 10000000 ≤ n ∧ n < 100000000 }

-- Define the property of having the same first and last four digits
def SameFirstLastFourDigits (n : EightDigitNumber) : Prop :=
  ∃ (a b c d : ℕ), 
    0 ≤ a ∧ a < 10 ∧
    0 ≤ b ∧ b < 10 ∧
    0 ≤ c ∧ c < 10 ∧
    0 ≤ d ∧ d < 10 ∧
    n.val = 10000000 * a + 1000000 * b + 100000 * c + 10000 * d +
            1000 * a + 100 * b + 10 * c + d

-- Theorem statement
theorem ten_thousand_one_divides_same_first_last_four_digits 
  (n : EightDigitNumber) (h : SameFirstLastFourDigits n) : 
  10001 ∣ n.val :=
sorry

end NUMINAMATH_CALUDE_ten_thousand_one_divides_same_first_last_four_digits_l2547_254767


namespace NUMINAMATH_CALUDE_extreme_value_and_intersection_l2547_254716

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (a - Real.log x) / x

def g (x : ℝ) : ℝ := -1

theorem extreme_value_and_intersection (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ x ≤ Real.exp 1 ∧ f a x = g x) →
  (∀ (x : ℝ), x > 0 → f a x ≥ -Real.exp (-a - 1)) ∧
  (f a (Real.exp (a + 1)) = -Real.exp (-a - 1)) ∧
  (a ≤ -1 ∨ (0 ≤ a ∧ a ≤ Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_and_intersection_l2547_254716


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l2547_254729

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 13) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1027 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l2547_254729


namespace NUMINAMATH_CALUDE_two_times_larger_by_one_l2547_254704

theorem two_times_larger_by_one (a : ℝ) : 
  (2 * a + 1) = (2 * a) + 1 := by sorry

end NUMINAMATH_CALUDE_two_times_larger_by_one_l2547_254704


namespace NUMINAMATH_CALUDE_trumpet_cost_l2547_254744

/-- The cost of the trumpet given the total spent and the costs of other items. -/
theorem trumpet_cost (total_spent music_tool_cost song_book_cost : ℚ) 
  (h1 : total_spent = 163.28)
  (h2 : music_tool_cost = 9.98)
  (h3 : song_book_cost = 4.14) :
  total_spent - (music_tool_cost + song_book_cost) = 149.16 := by
  sorry

end NUMINAMATH_CALUDE_trumpet_cost_l2547_254744


namespace NUMINAMATH_CALUDE_linear_equation_m_value_l2547_254771

theorem linear_equation_m_value (m : ℤ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (m - 1) * x^(abs m) - 2 = a * x + b) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_m_value_l2547_254771


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2547_254790

/-- A positive geometric sequence -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_problem (a : ℕ → ℝ)
    (h_geom : IsPositiveGeometricSequence a)
    (h_sum : a 1 + 2/3 * a 2 = 3)
    (h_prod : a 4 ^ 2 = 1/9 * a 3 * a 7) :
    a 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2547_254790


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l2547_254734

theorem smallest_x_for_equation : 
  ∃ (x : ℕ+), x = 9 ∧ 
  (∃ (y : ℕ+), (9 : ℚ) / 10 = (y : ℚ) / (151 + x)) ∧ 
  (∀ (x' : ℕ+), x' < x → 
    ¬∃ (y : ℕ+), (9 : ℚ) / 10 = (y : ℚ) / (151 + x')) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l2547_254734


namespace NUMINAMATH_CALUDE_breakfast_cost_theorem_l2547_254755

/-- The cost of breakfast for Francis and Kiera -/
def breakfast_cost (muffin_price fruit_cup_price : ℕ) 
  (francis_muffins francis_fruit_cups : ℕ)
  (kiera_muffins kiera_fruit_cups : ℕ) : ℕ :=
  (francis_muffins + kiera_muffins) * muffin_price +
  (francis_fruit_cups + kiera_fruit_cups) * fruit_cup_price

/-- Theorem stating the total cost of Francis and Kiera's breakfast -/
theorem breakfast_cost_theorem : 
  breakfast_cost 2 3 2 2 2 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cost_theorem_l2547_254755


namespace NUMINAMATH_CALUDE_sum_of_distances_bound_l2547_254719

/-- A rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ
  length_ge_width : length ≥ width

/-- A point inside a rectangle -/
structure PointInRectangle (rect : Rectangle) where
  x : ℝ
  y : ℝ
  x_bounds : 0 ≤ x ∧ x ≤ rect.length
  y_bounds : 0 ≤ y ∧ y ≤ rect.width

/-- The sum of distances from a point to the extensions of all sides of a rectangle -/
def sum_of_distances (rect : Rectangle) (p : PointInRectangle rect) : ℝ :=
  p.x + (rect.length - p.x) + p.y + (rect.width - p.y)

/-- The theorem stating that the sum of distances is at most 2l + 2w -/
theorem sum_of_distances_bound (rect : Rectangle) (p : PointInRectangle rect) :
  sum_of_distances rect p ≤ 2 * rect.length + 2 * rect.width := by
  sorry


end NUMINAMATH_CALUDE_sum_of_distances_bound_l2547_254719


namespace NUMINAMATH_CALUDE_addition_multiplication_equality_l2547_254786

theorem addition_multiplication_equality : 300 + 5 * 8 = 340 := by
  sorry

end NUMINAMATH_CALUDE_addition_multiplication_equality_l2547_254786


namespace NUMINAMATH_CALUDE_m_range_l2547_254769

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 ≥ 0

def q (m : ℝ) : Prop := ∀ x, (8*x + 4*(m - 1)) ≠ 0

-- Define the theorem
theorem m_range (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → ((-2 ≤ m ∧ m < 1) ∨ m > 2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l2547_254769


namespace NUMINAMATH_CALUDE_library_books_end_of_month_l2547_254700

theorem library_books_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) : 
  initial_books = 75 →
  loaned_books = 20 →
  return_rate = 65 / 100 →
  initial_books - loaned_books + (↑loaned_books * return_rate).floor = 68 :=
by sorry

end NUMINAMATH_CALUDE_library_books_end_of_month_l2547_254700


namespace NUMINAMATH_CALUDE_ratio_transitive_l2547_254740

theorem ratio_transitive (a b c : ℝ) 
  (h1 : a / b = 7 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transitive_l2547_254740


namespace NUMINAMATH_CALUDE_father_sons_ages_l2547_254787

theorem father_sons_ages (father_age : ℕ) (youngest_son_age : ℕ) (years_until_equal : ℕ) :
  father_age = 33 →
  youngest_son_age = 2 →
  years_until_equal = 12 →
  ∃ (middle_son_age oldest_son_age : ℕ),
    (father_age + years_until_equal = youngest_son_age + years_until_equal + 
                                      middle_son_age + years_until_equal + 
                                      oldest_son_age + years_until_equal) ∧
    (middle_son_age = 3 ∧ oldest_son_age = 4) :=
by sorry

end NUMINAMATH_CALUDE_father_sons_ages_l2547_254787


namespace NUMINAMATH_CALUDE_total_green_is_seven_l2547_254788

/-- The number of green marbles Sara has -/
def sara_green : ℕ := 3

/-- The number of green marbles Tom has -/
def tom_green : ℕ := 4

/-- The total number of green marbles Sara and Tom have together -/
def total_green : ℕ := sara_green + tom_green

/-- Theorem stating that the total number of green marbles is 7 -/
theorem total_green_is_seven : total_green = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_green_is_seven_l2547_254788


namespace NUMINAMATH_CALUDE_partner_contribution_b_contribution_is_31500_l2547_254756

/-- Given a business partnership scenario, calculate partner B's capital contribution. -/
theorem partner_contribution (a_initial : ℕ) (a_months : ℕ) (b_months : ℕ) (profit_ratio_a : ℕ) (profit_ratio_b : ℕ) : ℕ :=
  let total_months := a_months
  let b_contribution := (a_initial * total_months * profit_ratio_b) / (b_months * profit_ratio_a)
  b_contribution

/-- Prove that B's contribution is 31500 rupees given the specific scenario. -/
theorem b_contribution_is_31500 :
  partner_contribution 3500 12 2 2 3 = 31500 := by
  sorry

end NUMINAMATH_CALUDE_partner_contribution_b_contribution_is_31500_l2547_254756


namespace NUMINAMATH_CALUDE_sum_f_two_and_neg_two_l2547_254724

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -1 / x else x^2

theorem sum_f_two_and_neg_two : f 2 + f (-2) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_two_and_neg_two_l2547_254724


namespace NUMINAMATH_CALUDE_half_inequality_l2547_254774

theorem half_inequality (a b : ℝ) (h : a > b) : (1/2) * a > (1/2) * b := by
  sorry

end NUMINAMATH_CALUDE_half_inequality_l2547_254774


namespace NUMINAMATH_CALUDE_complex_number_problem_l2547_254781

theorem complex_number_problem (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) * (1 - Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) →
  (a = -1 ∧ Complex.abs (z + Complex.I) = 3) := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2547_254781


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2547_254702

theorem simplify_square_roots : 
  2 * Real.sqrt 12 - Real.sqrt 27 - Real.sqrt 3 * Real.sqrt (1/9) = (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2547_254702


namespace NUMINAMATH_CALUDE_perfect_square_factors_count_l2547_254749

/-- Given a natural number with prime factorization 2^6 × 3^3, 
    this function returns the count of its positive integer factors that are perfect squares -/
def count_perfect_square_factors (N : ℕ) : ℕ :=
  8

/-- The theorem stating that for a number with prime factorization 2^6 × 3^3,
    the count of its positive integer factors that are perfect squares is 8 -/
theorem perfect_square_factors_count (N : ℕ) 
  (h : N = 2^6 * 3^3) : 
  count_perfect_square_factors N = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factors_count_l2547_254749


namespace NUMINAMATH_CALUDE_total_purchase_cost_l2547_254773

def snake_toy_cost : ℚ := 11.76
def cage_cost : ℚ := 14.54

theorem total_purchase_cost : snake_toy_cost + cage_cost = 26.30 := by
  sorry

end NUMINAMATH_CALUDE_total_purchase_cost_l2547_254773


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2547_254717

def U : Set ℕ := {1,2,3,4,5,6,7,8,9}
def A : Set ℕ := {2,4,5,7}
def B : Set ℕ := {3,4,5,6,8}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {3,6,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2547_254717


namespace NUMINAMATH_CALUDE_average_difference_is_negative_six_point_fifteen_l2547_254775

/- Define the parameters of the problem -/
def total_students : ℕ := 120
def total_teachers : ℕ := 6
def dual_enrolled_students : ℕ := 10
def class_enrollments : List ℕ := [40, 30, 25, 15, 5, 5]

/- Define the average number of students per teacher -/
def t : ℚ := (total_students : ℚ) / total_teachers

/- Define the average number of students per student, including dual enrollments -/
def s : ℚ :=
  let total_enrollments := total_students + dual_enrolled_students
  (class_enrollments.map (λ x => (x : ℚ) * x / total_enrollments)).sum

/- The theorem to be proved -/
theorem average_difference_is_negative_six_point_fifteen :
  t - s = -315 / 100 := by sorry

end NUMINAMATH_CALUDE_average_difference_is_negative_six_point_fifteen_l2547_254775


namespace NUMINAMATH_CALUDE_marbles_in_first_jar_l2547_254748

theorem marbles_in_first_jar (jar1 jar2 jar3 : ℕ) : 
  jar2 = 2 * jar1 →
  jar3 = jar1 / 4 →
  jar1 + jar2 + jar3 = 260 →
  jar1 = 80 := by
sorry

end NUMINAMATH_CALUDE_marbles_in_first_jar_l2547_254748


namespace NUMINAMATH_CALUDE_vkontakte_problem_l2547_254795

-- Define predicates for each person being on VKontakte
variable (M I A P : Prop)

-- State the theorem
theorem vkontakte_problem :
  (M → (I ∧ A)) →  -- If M is on VKontakte, then both I and A are on VKontakte
  (A ↔ ¬P) →       -- Only one of A or P is on VKontakte
  (I ∨ M) →        -- At least one of I or M is on VKontakte
  (P ↔ I) →        -- P and I are either both on or both not on VKontakte
  (I ∧ P ∧ ¬M ∧ ¬A) -- Conclusion: I and P are on VKontakte, M and A are not
  := by sorry

end NUMINAMATH_CALUDE_vkontakte_problem_l2547_254795


namespace NUMINAMATH_CALUDE_marbleCombinations_eq_twelve_l2547_254762

/-- The number of ways to select 4 marbles from a set of 5 indistinguishable red marbles,
    4 indistinguishable blue marbles, and 2 indistinguishable black marbles -/
def marbleCombinations : ℕ :=
  let red := 5
  let blue := 4
  let black := 2
  let totalSelect := 4
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    t.1 + t.2.1 + t.2.2 = totalSelect ∧ 
    t.1 ≤ red ∧ 
    t.2.1 ≤ blue ∧ 
    t.2.2 ≤ black
  ) (Finset.product (Finset.range (red + 1)) (Finset.product (Finset.range (blue + 1)) (Finset.range (black + 1))))).card

theorem marbleCombinations_eq_twelve : marbleCombinations = 12 := by
  sorry

end NUMINAMATH_CALUDE_marbleCombinations_eq_twelve_l2547_254762


namespace NUMINAMATH_CALUDE_union_M_N_equals_N_l2547_254798

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x > 0}
def N : Set ℝ := {x | |x| > 2}

-- State the theorem
theorem union_M_N_equals_N : M ∪ N = N := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_N_l2547_254798


namespace NUMINAMATH_CALUDE_at_hash_sum_l2547_254708

def at_operation (a b : ℕ+) : ℚ := (a.val * b.val : ℚ) / (a.val + b.val)

def hash_operation (a b : ℕ+) : ℚ := (a.val + 3 * b.val : ℚ) / (b.val + 3 * a.val)

theorem at_hash_sum :
  (at_operation 3 9) + (hash_operation 3 9) = 47 / 12 := by sorry

end NUMINAMATH_CALUDE_at_hash_sum_l2547_254708


namespace NUMINAMATH_CALUDE_job_completion_time_l2547_254742

/-- Given two workers A and B, where A completes a job in 10 days and B completes it in 6 days,
    prove that they can complete the job together in 3.75 days. -/
theorem job_completion_time (a_time b_time combined_time : ℝ) 
  (ha : a_time = 10) 
  (hb : b_time = 6) 
  (hc : combined_time = (a_time * b_time) / (a_time + b_time)) : 
  combined_time = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2547_254742


namespace NUMINAMATH_CALUDE_correct_answer_points_l2547_254714

/-- Represents the scoring system for a math competition --/
structure ScoringSystem where
  total_problems : ℕ
  wang_score : ℤ
  zhang_score : ℤ
  correct_points : ℕ
  incorrect_points : ℕ

/-- Theorem stating that the given scoring system results in 25 points for correct answers --/
theorem correct_answer_points (s : ScoringSystem) : 
  s.total_problems = 20 ∧ 
  s.wang_score = 328 ∧ 
  s.zhang_score = 27 ∧ 
  s.correct_points ≥ 10 ∧ s.correct_points ≤ 99 ∧
  s.incorrect_points ≥ 10 ∧ s.incorrect_points ≤ 99 →
  s.correct_points = 25 := by
  sorry


end NUMINAMATH_CALUDE_correct_answer_points_l2547_254714


namespace NUMINAMATH_CALUDE_annual_insurance_payment_l2547_254705

/-- The number of quarters in a year -/
def quarters_per_year : ℕ := 4

/-- The quarterly insurance payment in dollars -/
def quarterly_payment : ℕ := 378

/-- The annual insurance payment in dollars -/
def annual_payment : ℕ := quarterly_payment * quarters_per_year

theorem annual_insurance_payment :
  annual_payment = 1512 :=
by sorry

end NUMINAMATH_CALUDE_annual_insurance_payment_l2547_254705


namespace NUMINAMATH_CALUDE_uncle_lou_peanuts_l2547_254728

/-- Calculates the number of peanuts in each bag given the conditions of Uncle Lou's flight. -/
theorem uncle_lou_peanuts (bags : ℕ) (flight_duration : ℕ) (eating_rate : ℕ) : bags = 4 → flight_duration = 120 → eating_rate = 1 → (flight_duration / bags : ℕ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_uncle_lou_peanuts_l2547_254728


namespace NUMINAMATH_CALUDE_rectangle_not_stable_l2547_254713

-- Define the shape type
inductive Shape
| AcuteTriangle
| Rectangle
| RightTriangle
| IsoscelesTriangle

-- Define stability property
def IsStable (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => False
  | _ => True

-- State the theorem
theorem rectangle_not_stable :
  ∀ (s : Shape), ¬(IsStable s) ↔ s = Shape.Rectangle :=
by sorry

end NUMINAMATH_CALUDE_rectangle_not_stable_l2547_254713


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l2547_254722

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (5 * x + 1 > 3 * (x - 1)) ∧ ((x - 1) / 2 ≥ 2 * x - 4)}
  S = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l2547_254722


namespace NUMINAMATH_CALUDE_sum_of_digits_2017_power_l2547_254709

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that S(S(S(S(2017^2017)))) = 1 -/
theorem sum_of_digits_2017_power : S (S (S (S (2017^2017)))) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_2017_power_l2547_254709


namespace NUMINAMATH_CALUDE_circle_center_correct_l2547_254751

/-- The equation of a circle in the form x^2 - 2ax + y^2 - 2by + c = 0 -/
def CircleEquation (a b c : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 - 2*a*x + y^2 - 2*b*y + c = 0

/-- The center of a circle given by its equation -/
def CircleCenter (a b c : ℝ) : ℝ × ℝ := (a, b)

theorem circle_center_correct (x y : ℝ) :
  CircleEquation 1 2 (-28) x y → CircleCenter 1 2 (-28) = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_correct_l2547_254751


namespace NUMINAMATH_CALUDE_trig_simplification_l2547_254797

theorem trig_simplification (x y : ℝ) :
  (Real.cos (x + π/4))^2 + (Real.cos (x + y + π/2))^2 - 
  2 * Real.cos (x + π/4) * Real.cos (y + π/4) * Real.cos (x + y + π/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2547_254797


namespace NUMINAMATH_CALUDE_corn_harvest_difference_l2547_254707

theorem corn_harvest_difference (greg_harvest sharon_harvest : ℝ) 
  (h1 : greg_harvest = 0.4)
  (h2 : sharon_harvest = 0.1) :
  greg_harvest - sharon_harvest = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_corn_harvest_difference_l2547_254707


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l2547_254720

/-- Calculates the total cost of power cable for a neighborhood with the given specifications. -/
theorem neighborhood_cable_cost
  (ew_streets : ℕ)
  (ew_length : ℝ)
  (ns_streets : ℕ)
  (ns_length : ℝ)
  (cable_per_mile : ℝ)
  (cable_cost : ℝ)
  (h1 : ew_streets = 18)
  (h2 : ew_length = 2)
  (h3 : ns_streets = 10)
  (h4 : ns_length = 4)
  (h5 : cable_per_mile = 5)
  (h6 : cable_cost = 2000) :
  (ew_streets * ew_length + ns_streets * ns_length) * cable_per_mile * cable_cost = 760000 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_cable_cost_l2547_254720


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2547_254796

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 + (4*m + 1)*x₁ + m = 0) ∧
  (x₂^2 + (4*m + 1)*x₂ + m = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2547_254796


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2547_254711

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ≥ 0 → x^2 + x - 1 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 + x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2547_254711


namespace NUMINAMATH_CALUDE_gcd_105_45_l2547_254770

theorem gcd_105_45 : Nat.gcd 105 45 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_45_l2547_254770


namespace NUMINAMATH_CALUDE_existence_of_x_l2547_254710

theorem existence_of_x (a : Fin 1997 → ℕ)
  (h1 : ∀ i j : Fin 1997, i + j ≤ 1997 → a i + a j ≤ a (i + j))
  (h2 : ∀ i j : Fin 1997, i + j ≤ 1997 → a (i + j) ≤ a i + a j + 1) :
  ∃ x : ℝ, ∀ n : Fin 1997, a n = ⌊n * x⌋ := by
sorry

end NUMINAMATH_CALUDE_existence_of_x_l2547_254710


namespace NUMINAMATH_CALUDE_unique_modular_solution_l2547_254754

theorem unique_modular_solution : ∃! n : ℕ, n ≤ 9 ∧ n ≡ -1345 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l2547_254754


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2547_254726

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x - 1)

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2547_254726


namespace NUMINAMATH_CALUDE_gas_bill_calculation_l2547_254780

/-- Represents the household bills and payments -/
structure HouseholdBills where
  electricity : ℕ
  water : ℕ
  internet : ℕ
  gas : ℕ
  gasPaidFraction : ℚ
  gasAdditionalPayment : ℕ
  remainingPayment : ℕ

/-- Theorem stating that given the household bill conditions, the gas bill is $120 -/
theorem gas_bill_calculation (bills : HouseholdBills) 
  (h1 : bills.electricity = 60)
  (h2 : bills.water = 40)
  (h3 : bills.internet = 25)
  (h4 : bills.gasPaidFraction = 3/4)
  (h5 : bills.gasAdditionalPayment = 5)
  (h6 : bills.remainingPayment = 30)
  (h7 : bills.water / 2 + (bills.internet - 4 * 5) + (bills.gas * (1 - bills.gasPaidFraction) - bills.gasAdditionalPayment) = bills.remainingPayment) :
  bills.gas = 120 := by
  sorry

#check gas_bill_calculation

end NUMINAMATH_CALUDE_gas_bill_calculation_l2547_254780


namespace NUMINAMATH_CALUDE_bakers_new_cakes_l2547_254765

/-- Baker's cake problem -/
theorem bakers_new_cakes 
  (initial_cakes : ℕ) 
  (sold_initial : ℕ) 
  (sold_difference : ℕ) 
  (h1 : initial_cakes = 170)
  (h2 : sold_initial = 78)
  (h3 : sold_difference = 47)
  : ∃ (new_cakes : ℕ), 
    sold_initial + sold_difference = new_cakes + sold_difference ∧ 
    new_cakes = 78 :=
by sorry

end NUMINAMATH_CALUDE_bakers_new_cakes_l2547_254765


namespace NUMINAMATH_CALUDE_transformed_circle_equation_l2547_254789

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the scaling transformation
def scaling_transform (x y x' y' : ℝ) : Prop := x' = 5*x ∧ y' = 3*y

-- State the theorem
theorem transformed_circle_equation (x y x' y' : ℝ) :
  original_circle x y ∧ scaling_transform x y x' y' →
  x'^2 / 25 + y'^2 / 9 = 1 :=
by sorry

end NUMINAMATH_CALUDE_transformed_circle_equation_l2547_254789


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2547_254731

theorem rectangle_dimensions (x : ℝ) : 
  x > 3 →
  (x - 3) * (3 * x + 6) = 9 * x - 9 →
  x = (21 + Real.sqrt 549) / 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2547_254731


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2547_254703

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 where a > 0, b > 0,
    and one of its asymptotes is y = √2 x, prove that its eccentricity is √3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x : ℝ, y = Real.sqrt 2 * x) →
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2547_254703


namespace NUMINAMATH_CALUDE_train_passing_jogger_l2547_254783

theorem train_passing_jogger (jogger_speed train_speed : ℝ) 
  (train_length initial_distance : ℝ) :
  jogger_speed = 9 →
  train_speed = 45 →
  train_length = 120 →
  initial_distance = 240 →
  (train_speed - jogger_speed) * (5 / 18) * 
    ((initial_distance + train_length) / ((train_speed - jogger_speed) * (5 / 18))) = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_l2547_254783


namespace NUMINAMATH_CALUDE_eugene_pencils_l2547_254750

def distribute_pencils (initial : ℕ) (received : ℕ) (per_friend : ℕ) : ℕ :=
  (initial + received) % per_friend

theorem eugene_pencils : distribute_pencils 127 14 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l2547_254750


namespace NUMINAMATH_CALUDE_exists_tangent_circle_l2547_254764

/-- Two parallel lines in a plane -/
structure ParallelLines where
  distance : ℝ
  distance_pos : distance > 0

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- The configuration of our geometry problem -/
structure Configuration where
  lines : ParallelLines
  given_circle : Circle
  circle_between_lines : given_circle.center.2 > 0 ∧ given_circle.center.2 < lines.distance

/-- The theorem stating the existence of the sought circle -/
theorem exists_tangent_circle (config : Configuration) :
  ∃ (tangent_circle : Circle),
    tangent_circle.radius = config.lines.distance / 2 ∧
    (tangent_circle.center.2 = config.lines.distance / 2 ∨
     tangent_circle.center.2 = config.lines.distance / 2) ∧
    ((tangent_circle.center.1 - config.given_circle.center.1) ^ 2 +
     (tangent_circle.center.2 - config.given_circle.center.2) ^ 2 =
     (tangent_circle.radius + config.given_circle.radius) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_exists_tangent_circle_l2547_254764


namespace NUMINAMATH_CALUDE_prob_two_sixes_one_four_l2547_254747

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of rolling a specific number on a single die -/
def single_prob : ℚ := 1 / num_sides

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The number of ways to arrange two 6's and one 4 in three dice rolls -/
def num_arrangements : ℕ := 3

/-- The probability of rolling exactly two 6's and one 4 when rolling three six-sided dice simultaneously -/
theorem prob_two_sixes_one_four : 
  (single_prob ^ num_dice * num_arrangements : ℚ) = 1 / 72 := by sorry

end NUMINAMATH_CALUDE_prob_two_sixes_one_four_l2547_254747


namespace NUMINAMATH_CALUDE_triangle_max_side_length_range_l2547_254741

theorem triangle_max_side_length_range (P : ℝ) (a b c : ℝ) (h_triangle : a + b + c = P) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_max : c = max a (max b c)) : P / 3 ≤ c ∧ c < P / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_side_length_range_l2547_254741


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2547_254776

theorem fraction_evaluation : 
  (1 - 1/4) / (1 - 2/3) + 1/6 = 29/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2547_254776


namespace NUMINAMATH_CALUDE_girls_in_class_l2547_254760

/-- 
Given a class with a total of 60 people and a ratio of girls to boys to teachers of 3:2:1,
prove that the number of girls in the class is 30.
-/
theorem girls_in_class (total : ℕ) (girls boys teachers : ℕ) : 
  total = 60 →
  girls + boys + teachers = total →
  girls = 3 * teachers →
  boys = 2 * teachers →
  girls = 30 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l2547_254760


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2547_254715

theorem geometric_arithmetic_sequence_problem :
  ∀ a b c : ℝ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (∃ r : ℝ, b = a * r ∧ c = b * r) →
  (a * b * c = 512) →
  (2 * b = (a - 2) + (c - 2)) →
  ((a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = 16 ∧ b = 8 ∧ c = 4)) :=
by sorry


end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2547_254715


namespace NUMINAMATH_CALUDE_initial_experiment_range_is_appropriate_l2547_254766

-- Define the types of microorganisms
inductive Microorganism
| Bacteria
| Actinomycetes
| Fungi
| Unknown

-- Define a function to represent the typical dilution range for each microorganism
def typicalDilutionRange (m : Microorganism) : Set ℕ :=
  match m with
  | Microorganism.Bacteria => {4, 5, 6}
  | Microorganism.Actinomycetes => {3, 4, 5}
  | Microorganism.Fungi => {2, 3, 4}
  | Microorganism.Unknown => {}

-- Define the general dilution range for initial experiments
def initialExperimentRange : Set ℕ := {n | 1 ≤ n ∧ n ≤ 7}

-- Theorem statement
theorem initial_experiment_range_is_appropriate :
  ∀ m : Microorganism, (typicalDilutionRange m).Subset initialExperimentRange :=
sorry

end NUMINAMATH_CALUDE_initial_experiment_range_is_appropriate_l2547_254766


namespace NUMINAMATH_CALUDE_rat_value_l2547_254777

/-- Represents the alphabet with corresponding numeric values. --/
def alphabet : List (Char × Nat) := [
  ('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7), ('h', 8), ('i', 9), ('j', 10),
  ('k', 11), ('l', 12), ('m', 13), ('n', 14), ('o', 15), ('p', 16), ('q', 17), ('r', 18), ('s', 19),
  ('t', 20), ('u', 21), ('v', 22), ('w', 23), ('x', 24), ('y', 25), ('z', 26)
]

/-- Gets the numeric value of a character based on its position in the alphabet. --/
def letterValue (c : Char) : Nat :=
  (alphabet.find? (fun p => p.1 == c.toLower)).map Prod.snd |>.getD 0

/-- Calculates the number value of a word based on the given rules. --/
def wordValue (word : String) : Nat :=
  let letterSum := word.toList.map letterValue |>.sum
  letterSum * word.length

/-- Theorem stating that the number value of "rat" is 117. --/
theorem rat_value : wordValue "rat" = 117 := by
  sorry

end NUMINAMATH_CALUDE_rat_value_l2547_254777


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l2547_254732

theorem reciprocal_of_negative_two_thirds :
  let x : ℚ := -2/3
  let reciprocal (y : ℚ) : ℚ := if y ≠ 0 then 1 / y else 0
  reciprocal x = -3/2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l2547_254732


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2547_254743

theorem cube_root_equation_solution :
  ∀ x : ℝ, (((5 - x / 3) ^ (1/3 : ℝ) = -2) ↔ (x = 39)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2547_254743


namespace NUMINAMATH_CALUDE_min_value_sqrt_a_plus_four_over_sqrt_a_plus_one_l2547_254745

theorem min_value_sqrt_a_plus_four_over_sqrt_a_plus_one (a : ℝ) (ha : a > 0) :
  Real.sqrt a + 4 / (Real.sqrt a + 1) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_a_plus_four_over_sqrt_a_plus_one_l2547_254745


namespace NUMINAMATH_CALUDE_smallest_side_of_triangle_l2547_254739

theorem smallest_side_of_triangle : ∃ (s : ℕ),
  (s : ℝ) > 0 ∧ 
  7.5 + (s : ℝ) > 11 ∧ 
  7.5 + 11 > (s : ℝ) ∧ 
  11 + (s : ℝ) > 7.5 ∧
  ∀ (t : ℕ), t > 0 → 
    (7.5 + (t : ℝ) > 11 ∧ 
     7.5 + 11 > (t : ℝ) ∧ 
     11 + (t : ℝ) > 7.5) → 
    s ≤ t ∧
  s = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_side_of_triangle_l2547_254739


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l2547_254746

theorem added_number_after_doubling (original : ℕ) (added : ℕ) : 
  original = 5 → 3 * (2 * original + added) = 57 → added = 9 := by
  sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l2547_254746
