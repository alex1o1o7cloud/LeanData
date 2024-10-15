import Mathlib

namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l1301_130124

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let original : Point2D := { x := 2, y := 3 }
  reflectAcrossXAxis original = { x := 2, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l1301_130124


namespace NUMINAMATH_CALUDE_right_pentagonal_pyramid_base_side_length_l1301_130134

/-- Represents a right pyramid with a regular pentagonal base -/
structure RightPentagonalPyramid where
  base_side_length : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- 
Theorem: For a right pyramid with a regular pentagonal base, 
if the area of one lateral face is 120 square meters and the slant height is 40 meters, 
then the length of the side of its base is 6 meters.
-/
theorem right_pentagonal_pyramid_base_side_length 
  (pyramid : RightPentagonalPyramid) 
  (h1 : pyramid.lateral_face_area = 120) 
  (h2 : pyramid.slant_height = 40) : 
  pyramid.base_side_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_pentagonal_pyramid_base_side_length_l1301_130134


namespace NUMINAMATH_CALUDE_find_divisor_l1301_130118

theorem find_divisor (n : ℕ) (added : ℕ) (divisor : ℕ) : 
  (n + added) % divisor = 0 ∧ 
  (∀ m : ℕ, m < added → (n + m) % divisor ≠ 0) →
  divisor = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1301_130118


namespace NUMINAMATH_CALUDE_least_possible_difference_l1301_130174

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → Odd y → Odd z → 
  (∀ (x' y' z' : ℤ), x' < y' → y' < z' → y' - x' > 5 → Even x' → Odd y' → Odd z' → z' - x' ≥ z - x) →
  z - x = 9 := by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l1301_130174


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1301_130160

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → m ≤ n) → n + (n + 1) = 43 :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1301_130160


namespace NUMINAMATH_CALUDE_sum_outside_angles_inscribed_pentagon_l1301_130194

/-- A pentagon inscribed in a circle -/
structure InscribedPentagon where
  -- Define the circle
  circle : Set (ℝ × ℝ)
  -- Define the pentagon
  pentagon : Set (ℝ × ℝ)
  -- Ensure the pentagon is inscribed in the circle
  is_inscribed : pentagon ⊆ circle

/-- An angle inscribed in a segment outside the pentagon -/
def OutsideAngle (p : InscribedPentagon) : Type :=
  { θ : ℝ // 0 ≤ θ ∧ θ ≤ 2 * Real.pi }

/-- The theorem stating that the sum of angles inscribed in the five segments
    outside an inscribed pentagon is equal to 5π/2 radians (900°) -/
theorem sum_outside_angles_inscribed_pentagon (p : InscribedPentagon) 
  (α β γ δ ε : OutsideAngle p) : 
  α.val + β.val + γ.val + δ.val + ε.val = 5 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_outside_angles_inscribed_pentagon_l1301_130194


namespace NUMINAMATH_CALUDE_work_ratio_man_to_boy_l1301_130116

theorem work_ratio_man_to_boy :
  ∀ (m b : ℝ),
  m > 0 → b > 0 →
  7 * m + 2 * b = 6 * (m + b) →
  m / b = 4 := by
sorry

end NUMINAMATH_CALUDE_work_ratio_man_to_boy_l1301_130116


namespace NUMINAMATH_CALUDE_tissue_diameter_calculation_l1301_130104

/-- Given a circular piece of tissue magnified by an electron microscope,
    calculate its actual diameter in millimeters. -/
theorem tissue_diameter_calculation
  (magnification : ℕ)
  (magnified_diameter_meters : ℝ)
  (h_magnification : magnification = 5000)
  (h_magnified_diameter : magnified_diameter_meters = 0.15) :
  magnified_diameter_meters * 1000 / magnification = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_tissue_diameter_calculation_l1301_130104


namespace NUMINAMATH_CALUDE_one_real_root_l1301_130100

-- Define the determinant function
def det (x a b d : ℝ) : ℝ := x * (x^2 + a^2 + b^2 + d^2)

-- State the theorem
theorem one_real_root (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃! x : ℝ, det x a b d = 0 :=
sorry

end NUMINAMATH_CALUDE_one_real_root_l1301_130100


namespace NUMINAMATH_CALUDE_miranda_savings_l1301_130184

theorem miranda_savings (total_cost sister_contribution saving_period : ℕ) 
  (h1 : total_cost = 260)
  (h2 : sister_contribution = 50)
  (h3 : saving_period = 3) :
  (total_cost - sister_contribution) / saving_period = 70 := by
  sorry

end NUMINAMATH_CALUDE_miranda_savings_l1301_130184


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1301_130143

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a survey with its characteristics --/
structure Survey where
  population_size : ℕ
  sample_size : ℕ
  has_subgroups : Bool
  is_uniform : Bool

/-- Determines the most appropriate sampling method for a given survey --/
def best_sampling_method (s : Survey) : SamplingMethod :=
  if s.has_subgroups then SamplingMethod.Stratified
  else if s.is_uniform && s.population_size > s.sample_size * 10 then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

/-- The three surveys described in the problem --/
def survey1 : Survey := { population_size := 10, sample_size := 3, has_subgroups := false, is_uniform := true }
def survey2 : Survey := { population_size := 32 * 40, sample_size := 32, has_subgroups := false, is_uniform := true }
def survey3 : Survey := { population_size := 160, sample_size := 20, has_subgroups := true, is_uniform := false }

theorem correct_sampling_methods :
  best_sampling_method survey1 = SamplingMethod.SimpleRandom ∧
  best_sampling_method survey2 = SamplingMethod.Systematic ∧
  best_sampling_method survey3 = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1301_130143


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1301_130110

/-- The discriminant of the quadratic equation x² - (m + 3)x + m + 1 = 0 -/
def discriminant (m : ℝ) : ℝ := (m + 1)^2 + 4

theorem quadratic_equation_properties :
  (∀ m : ℝ, discriminant m > 0) ∧
  ({m : ℝ | discriminant m = 5} = {0, -2}) := by
  sorry

#check quadratic_equation_properties

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1301_130110


namespace NUMINAMATH_CALUDE_triangle_area_sum_form_sum_of_coefficients_l1301_130105

/-- Represents a 2x2x2 cube -/
structure Cube :=
  (side : ℝ)
  (is_two : side = 2)

/-- Represents the sum of areas of all triangles with vertices on the cube -/
def triangle_area_sum (c : Cube) : ℝ := sorry

/-- The sum can be expressed as q + √r + √s where q, r, s are integers -/
theorem triangle_area_sum_form (c : Cube) :
  ∃ (q r s : ℤ), triangle_area_sum c = ↑q + Real.sqrt (↑r) + Real.sqrt (↑s) :=
sorry

/-- The sum of q, r, and s is 7728 -/
theorem sum_of_coefficients (c : Cube) :
  ∃ (q r s : ℤ),
    triangle_area_sum c = ↑q + Real.sqrt (↑r) + Real.sqrt (↑s) ∧
    q + r + s = 7728 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_sum_form_sum_of_coefficients_l1301_130105


namespace NUMINAMATH_CALUDE_complex_simplification_l1301_130159

/-- Given that i^2 = -1, prove that 3(4-2i) - 2i(3-i) + i(1+2i) = 8 - 11i -/
theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  3 * (4 - 2*i) - 2*i*(3 - i) + i*(1 + 2*i) = 8 - 11*i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l1301_130159


namespace NUMINAMATH_CALUDE_emily_big_garden_seeds_l1301_130126

def emily_garden_problem (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  total_seeds - (small_gardens * seeds_per_small_garden)

theorem emily_big_garden_seeds :
  emily_garden_problem 42 3 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_emily_big_garden_seeds_l1301_130126


namespace NUMINAMATH_CALUDE_infinite_n_squared_plus_one_divides_and_not_divides_factorial_l1301_130181

theorem infinite_n_squared_plus_one_divides_and_not_divides_factorial :
  (∃ S : Set ℤ, Set.Infinite S ∧ ∀ n ∈ S, (n^2 + 1) ∣ n!) ∧
  (∃ T : Set ℤ, Set.Infinite T ∧ ∀ n ∈ T, ¬((n^2 + 1) ∣ n!)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_n_squared_plus_one_divides_and_not_divides_factorial_l1301_130181


namespace NUMINAMATH_CALUDE_triangle_problem_l1301_130140

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where c = √3, b = 1, and B = 30°, prove that C is either 60° or 120°,
    and the corresponding area S is either √3/2 or √3/4. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  c = Real.sqrt 3 →
  b = 1 →
  B = 30 * π / 180 →
  ((C = 60 * π / 180 ∧ S = Real.sqrt 3 / 2) ∨
   (C = 120 * π / 180 ∧ S = Real.sqrt 3 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1301_130140


namespace NUMINAMATH_CALUDE_tunneled_cube_surface_area_l1301_130108

/-- Represents a cube with a tunnel carved through it -/
structure TunneledCube where
  side_length : ℝ
  tunnel_distance : ℝ

/-- Calculates the total surface area of a tunneled cube -/
def total_surface_area (c : TunneledCube) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the specific tunneled cube -/
theorem tunneled_cube_surface_area :
  ∃ (c : TunneledCube), c.side_length = 10 ∧ c.tunnel_distance = 3 ∧
  total_surface_area c = 600 + 73.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tunneled_cube_surface_area_l1301_130108


namespace NUMINAMATH_CALUDE_a_2_times_a_3_l1301_130180

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

theorem a_2_times_a_3 : a 2 * a 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_a_2_times_a_3_l1301_130180


namespace NUMINAMATH_CALUDE_smallest_with_12_divisors_l1301_130175

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Check if a number has exactly 12 divisors -/
def has_12_divisors (n : ℕ+) : Prop := num_divisors n = 12

theorem smallest_with_12_divisors :
  ∃ (n : ℕ+), has_12_divisors n ∧ ∀ (m : ℕ+), has_12_divisors m → n ≤ m :=
by
  use 288
  sorry

end NUMINAMATH_CALUDE_smallest_with_12_divisors_l1301_130175


namespace NUMINAMATH_CALUDE_circle_intersection_x_coordinate_l1301_130121

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle defined by two points on its diameter -/
structure Circle where
  p1 : Point
  p2 : Point

/-- Function to check if a given x-coordinate is one of the intersection points -/
def isIntersectionPoint (c : Circle) (x : ℝ) : Prop :=
  let center_x := (c.p1.x + c.p2.x) / 2
  let center_y := (c.p1.y + c.p2.y) / 2
  let radius := Real.sqrt ((c.p1.x - center_x)^2 + (c.p1.y - center_y)^2)
  (x - center_x)^2 + (1 - center_y)^2 = radius^2

/-- Theorem stating that one of the intersection points has x-coordinate 3 or 5 -/
theorem circle_intersection_x_coordinate 
  (c : Circle) 
  (h1 : c.p1 = ⟨1, 5⟩) 
  (h2 : c.p2 = ⟨7, 3⟩) : 
  isIntersectionPoint c 3 ∨ isIntersectionPoint c 5 := by
  sorry


end NUMINAMATH_CALUDE_circle_intersection_x_coordinate_l1301_130121


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l1301_130170

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the ages of Bipin, Alok, and Chandan -/
structure Ages where
  bipin : Age
  alok : Age
  chandan : Age

/-- The conditions given in the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.alok.years = 5 ∧
  ages.chandan.years = 10 ∧
  ages.bipin.years + 10 = 2 * (ages.chandan.years + 10)

/-- The theorem to prove -/
theorem age_ratio_theorem (ages : Ages) :
  problem_conditions ages →
  (ages.bipin.years : ℚ) / ages.alok.years = 6 / 1 := by
  sorry


end NUMINAMATH_CALUDE_age_ratio_theorem_l1301_130170


namespace NUMINAMATH_CALUDE_swimming_pool_containers_l1301_130122

/-- The minimum number of containers needed to fill a pool -/
def min_containers (pool_capacity : ℕ) (container_capacity : ℕ) : ℕ :=
  (pool_capacity + container_capacity - 1) / container_capacity

/-- Theorem: 30 containers of 75 liters each are needed to fill a 2250-liter pool -/
theorem swimming_pool_containers : 
  min_containers 2250 75 = 30 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_containers_l1301_130122


namespace NUMINAMATH_CALUDE_triangle_perimeters_l1301_130117

/-- The possible side lengths of the triangle -/
def triangle_sides : Set ℝ := {3, 6}

/-- Check if three numbers can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of possible perimeters of the triangle -/
def possible_perimeters : Set ℝ := {9, 15, 18}

/-- Theorem stating that the possible perimeters are 9, 15, or 18 -/
theorem triangle_perimeters :
  ∀ a b c : ℝ,
  a ∈ triangle_sides → b ∈ triangle_sides → c ∈ triangle_sides →
  is_triangle a b c →
  a + b + c ∈ possible_perimeters :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeters_l1301_130117


namespace NUMINAMATH_CALUDE_cut_tetrahedron_unfolds_to_given_config_l1301_130115

/-- Represents a polyhedron with vertices and edges -/
structure Polyhedron where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)

/-- Represents the unfolded configuration of a polyhedron -/
structure UnfoldedConfig where
  vertices : Finset (ℝ × ℝ)
  edges : Finset ((ℝ × ℝ) × (ℝ × ℝ))

/-- The given unfolded configuration from the problem -/
def given_config : UnfoldedConfig := sorry

/-- A tetrahedron with a smaller tetrahedron removed -/
def cut_tetrahedron : Polyhedron where
  vertices := {1, 2, 3, 4, 5, 6, 7, 8}
  edges := {(1,2), (1,3), (1,4), (2,3), (2,4), (3,4),
            (1,5), (2,6), (3,7), (4,8),
            (5,6), (6,7), (7,8)}

/-- Function to unfold a polyhedron onto a plane -/
def unfold (p : Polyhedron) : UnfoldedConfig := sorry

theorem cut_tetrahedron_unfolds_to_given_config :
  unfold cut_tetrahedron = given_config := by sorry

end NUMINAMATH_CALUDE_cut_tetrahedron_unfolds_to_given_config_l1301_130115


namespace NUMINAMATH_CALUDE_calculation_problem_linear_system_solution_l1301_130128

-- Problem 1
theorem calculation_problem : -Real.sqrt 3 + (-5/2)^0 + |1 - Real.sqrt 3| = 0 := by sorry

-- Problem 2
theorem linear_system_solution :
  ∃ (x y : ℝ), 4*x + 3*y = 10 ∧ 3*x + y = 5 ∧ x = 1 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_calculation_problem_linear_system_solution_l1301_130128


namespace NUMINAMATH_CALUDE_bird_migration_l1301_130190

/-- Bird migration problem -/
theorem bird_migration 
  (total_families : ℕ)
  (africa_families : ℕ)
  (asia_families : ℕ)
  (south_america_families : ℕ)
  (africa_days : ℕ)
  (asia_days : ℕ)
  (south_america_days : ℕ)
  (h1 : total_families = 200)
  (h2 : africa_families = 60)
  (h3 : asia_families = 95)
  (h4 : south_america_families = 30)
  (h5 : africa_days = 7)
  (h6 : asia_days = 14)
  (h7 : south_america_days = 10) :
  (total_families - (africa_families + asia_families + south_america_families) = 15) ∧
  (africa_families * africa_days + asia_families * asia_days + south_america_families * south_america_days = 2050) := by
  sorry


end NUMINAMATH_CALUDE_bird_migration_l1301_130190


namespace NUMINAMATH_CALUDE_bug_final_position_l1301_130111

def CirclePoints : Nat := 7

def jump (start : Nat) : Nat :=
  if start % 2 == 0 then
    (start + 2 - 1) % CirclePoints + 1
  else
    (start + 3 - 1) % CirclePoints + 1

def bug_position (start : Nat) (jumps : Nat) : Nat :=
  match jumps with
  | 0 => start
  | n + 1 => jump (bug_position start n)

theorem bug_final_position :
  bug_position 7 2023 = 1 := by sorry

end NUMINAMATH_CALUDE_bug_final_position_l1301_130111


namespace NUMINAMATH_CALUDE_cos_90_degrees_l1301_130161

theorem cos_90_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_l1301_130161


namespace NUMINAMATH_CALUDE_real_part_of_z_l1301_130152

def complex_number_z : ℂ → Prop :=
  λ z ↦ z * Complex.I = 2 * Complex.I

theorem real_part_of_z (z : ℂ) (h : complex_number_z z) :
  z.re = 3/2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l1301_130152


namespace NUMINAMATH_CALUDE_marble_replacement_l1301_130101

theorem marble_replacement (total : ℕ) (red blue yellow white black : ℕ) : 
  total = red + blue + yellow + white + black →
  red = (40 * total) / 100 →
  blue = (25 * total) / 100 →
  yellow = (10 * total) / 100 →
  white = (15 * total) / 100 →
  black = 20 →
  (blue + red / 3 : ℕ) = 77 := by
  sorry

end NUMINAMATH_CALUDE_marble_replacement_l1301_130101


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_twenty_l1301_130192

theorem log_expression_equals_negative_twenty :
  (Real.log (1/4) - Real.log 25) / (100 ^ (-1/2 : ℝ)) = -20 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_twenty_l1301_130192


namespace NUMINAMATH_CALUDE_five_digit_numbers_without_specific_digits_l1301_130176

/-- The number of digits allowed in each place (excluding the first place) -/
def allowed_digits : ℕ := 8

/-- The number of digits allowed in the first place -/
def first_place_digits : ℕ := 7

/-- The total number of places in the number -/
def total_places : ℕ := 5

/-- The expected total count of valid numbers -/
def expected_total : ℕ := 28672

theorem five_digit_numbers_without_specific_digits (d : ℕ) (h : d ≠ 7) :
  first_place_digits * (allowed_digits ^ (total_places - 1)) = expected_total :=
sorry

end NUMINAMATH_CALUDE_five_digit_numbers_without_specific_digits_l1301_130176


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l1301_130149

theorem quadratic_roots_difference_squared : 
  ∀ Φ φ : ℝ, 
  (Φ ^ 2 = Φ + 2) → 
  (φ ^ 2 = φ + 2) → 
  (Φ ≠ φ) → 
  (Φ - φ) ^ 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l1301_130149


namespace NUMINAMATH_CALUDE_difference_of_largest_and_smallest_l1301_130144

def digits : List ℕ := [2, 7, 4, 9]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  ∃ (a b c : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = 100 * a + 10 * b + c

def largest_number : ℕ := 974
def smallest_number : ℕ := 247

theorem difference_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_number) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ smallest_number) ∧
  largest_number - smallest_number = 727 :=
sorry

end NUMINAMATH_CALUDE_difference_of_largest_and_smallest_l1301_130144


namespace NUMINAMATH_CALUDE_parabola_focus_l1301_130177

/-- A parabola is defined by the equation x = -1/4 * y^2 -/
def parabola (x y : ℝ) : Prop := x = -1/4 * y^2

/-- The focus of a parabola is a point (f, 0) where f is a real number -/
def is_focus (f : ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, parabola x y → 
    ((x - f)^2 + y^2 = (x - (-f))^2) ∧ 
    (∀ g : ℝ, g ≠ f → ∃ x y : ℝ, parabola x y ∧ (x - g)^2 + y^2 ≠ (x - (-g))^2)

/-- The focus of the parabola x = -1/4 * y^2 is at the point (-1, 0) -/
theorem parabola_focus : is_focus (-1) parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l1301_130177


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1301_130103

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : 0 < d ∧ d < 1) 
  (h3 : ∀ k : ℤ, a 5 ≠ k * (π / 2)) 
  (h4 : Real.sin (a 3) ^ 2 + 2 * Real.sin (a 5) * Real.cos (a 5) = Real.sin (a 7) ^ 2) : 
  d = π / 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1301_130103


namespace NUMINAMATH_CALUDE_triangle_height_l1301_130182

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 9.31 → base = 4.9 → height = (2 * area) / base → height = 3.8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l1301_130182


namespace NUMINAMATH_CALUDE_caleb_hamburger_cost_l1301_130148

def total_burgers : ℕ := 50
def single_burger_cost : ℚ := 1
def double_burger_cost : ℚ := 1.5
def double_burgers_bought : ℕ := 29

theorem caleb_hamburger_cost :
  let single_burgers := total_burgers - double_burgers_bought
  let total_cost := (single_burgers : ℚ) * single_burger_cost +
                    (double_burgers_bought : ℚ) * double_burger_cost
  total_cost = 64.5 := by sorry

end NUMINAMATH_CALUDE_caleb_hamburger_cost_l1301_130148


namespace NUMINAMATH_CALUDE_train_bus_cost_l1301_130147

theorem train_bus_cost (bus_cost : ℝ) (train_extra_cost : ℝ) : 
  bus_cost = 1.40 →
  train_extra_cost = 6.85 →
  bus_cost + (bus_cost + train_extra_cost) = 9.65 := by
sorry

end NUMINAMATH_CALUDE_train_bus_cost_l1301_130147


namespace NUMINAMATH_CALUDE_unique_prime_triplet_l1301_130156

theorem unique_prime_triplet :
  ∃! (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    3 * p^4 - 5 * q^4 - 4 * r^2 = 26 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_triplet_l1301_130156


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1301_130150

/-- Given that x^2 and ∛y vary inversely, and x = 3 when y = 64, prove that y = 15 * ∛15 when xy = 90 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) :
  (∀ x y, x^2 * y^(1/3) = k) →  -- x^2 and ∛y vary inversely
  (3^2 * 64^(1/3) = k) →        -- x = 3 when y = 64
  (x * y = 90) →                -- xy = 90
  (y = 15 * 15^(1/5)) :=        -- y = 15 * ∛15
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1301_130150


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1301_130166

theorem average_of_remaining_numbers 
  (total_average : ℝ) 
  (avg_group1 : ℝ) 
  (avg_group2 : ℝ) 
  (h1 : total_average = 3.9) 
  (h2 : avg_group1 = 3.4) 
  (h3 : avg_group2 = 3.85) : 
  (6 * total_average - 2 * avg_group1 - 2 * avg_group2) / 2 = 4.45 := by
sorry

#eval (6 * 3.9 - 2 * 3.4 - 2 * 3.85) / 2

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1301_130166


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1301_130137

/-- A geometric sequence with first term 1 and the sum of the third and fifth terms equal to 6 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  (∃ q : ℝ, ∀ n : ℕ, a n = q ^ (n - 1)) ∧
  a 3 + a 5 = 6

/-- The sum of the fifth and seventh terms of the geometric sequence is 12 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 5 + a 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1301_130137


namespace NUMINAMATH_CALUDE_caleb_gallons_per_trip_l1301_130153

/-- Prove that Caleb adds 7 gallons per trip to fill a pool --/
theorem caleb_gallons_per_trip 
  (pool_capacity : ℕ) 
  (cynthia_gallons : ℕ) 
  (total_trips : ℕ) 
  (h1 : pool_capacity = 105)
  (h2 : cynthia_gallons = 8)
  (h3 : total_trips = 7)
  : ∃ (caleb_gallons : ℕ), 
    caleb_gallons * total_trips + cynthia_gallons * total_trips = pool_capacity ∧ 
    caleb_gallons = 7 := by
  sorry

end NUMINAMATH_CALUDE_caleb_gallons_per_trip_l1301_130153


namespace NUMINAMATH_CALUDE_sock_inventory_theorem_l1301_130145

/-- Represents the number of socks of a particular color --/
structure SockCount where
  pairs : ℕ
  singles : ℕ

/-- Represents the total sock inventory --/
structure SockInventory where
  blue : SockCount
  green : SockCount
  red : SockCount

def initial_inventory : SockInventory := {
  blue := { pairs := 20, singles := 0 },
  green := { pairs := 15, singles := 0 },
  red := { pairs := 15, singles := 0 }
}

def lost_socks : SockInventory := {
  blue := { pairs := 0, singles := 3 },
  green := { pairs := 0, singles := 2 },
  red := { pairs := 0, singles := 2 }
}

def donated_socks : SockInventory := {
  blue := { pairs := 0, singles := 10 },
  green := { pairs := 0, singles := 15 },
  red := { pairs := 0, singles := 10 }
}

def purchased_socks : SockInventory := {
  blue := { pairs := 5, singles := 0 },
  green := { pairs := 3, singles := 0 },
  red := { pairs := 2, singles := 0 }
}

def gifted_socks : SockInventory := {
  blue := { pairs := 2, singles := 0 },
  green := { pairs := 1, singles := 0 },
  red := { pairs := 0, singles := 0 }
}

def update_inventory (inv : SockInventory) (change : SockInventory) : SockInventory :=
  { blue := { pairs := inv.blue.pairs + change.blue.pairs - (inv.blue.singles + change.blue.singles) / 2,
              singles := (inv.blue.singles + change.blue.singles) % 2 },
    green := { pairs := inv.green.pairs + change.green.pairs - (inv.green.singles + change.green.singles) / 2,
               singles := (inv.green.singles + change.green.singles) % 2 },
    red := { pairs := inv.red.pairs + change.red.pairs - (inv.red.singles + change.red.singles) / 2,
             singles := (inv.red.singles + change.red.singles) % 2 } }

def total_pairs (inv : SockInventory) : ℕ :=
  inv.blue.pairs + inv.green.pairs + inv.red.pairs

theorem sock_inventory_theorem :
  let final_inventory := update_inventory 
                          (update_inventory 
                            (update_inventory 
                              (update_inventory initial_inventory lost_socks) 
                            donated_socks) 
                          purchased_socks) 
                        gifted_socks
  total_pairs final_inventory = 43 := by
  sorry

end NUMINAMATH_CALUDE_sock_inventory_theorem_l1301_130145


namespace NUMINAMATH_CALUDE_magazine_cost_l1301_130139

theorem magazine_cost (book magazine : ℚ)
  (h1 : 2 * book + 2 * magazine = 26)
  (h2 : book + 3 * magazine = 27) :
  magazine = 7 := by
sorry

end NUMINAMATH_CALUDE_magazine_cost_l1301_130139


namespace NUMINAMATH_CALUDE_decimal_number_problem_l1301_130127

theorem decimal_number_problem :
  ∃ x : ℝ, 
    0 ≤ x ∧ 
    x < 10 ∧ 
    (∃ n : ℤ, ⌊x⌋ = n) ∧
    ⌊x⌋ + 4 * x = 21.2 ∧
    x = 4.3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_number_problem_l1301_130127


namespace NUMINAMATH_CALUDE_cos_equality_solution_l1301_130187

theorem cos_equality_solution (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_solution_l1301_130187


namespace NUMINAMATH_CALUDE_rope_cutting_probability_l1301_130193

theorem rope_cutting_probability (rope_length : ℝ) (min_segment_length : ℝ) : 
  rope_length = 4 →
  min_segment_length = 1.5 →
  (rope_length - 2 * min_segment_length) / rope_length = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_rope_cutting_probability_l1301_130193


namespace NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l1301_130195

def is_not_divisible_by_five (b : ℤ) : Prop :=
  ¬ (5 ∣ (2 * b^3 - 2 * b^2))

theorem base_b_not_divisible_by_five :
  ∀ b : ℤ, b ∈ ({4, 5, 7, 8, 10} : Set ℤ) →
    (is_not_divisible_by_five b ↔ b ∈ ({4, 7, 8} : Set ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l1301_130195


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1301_130189

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_d_nonzero : d ≠ 0
  h_a1_nonzero : a 1 ≠ 0
  h_geometric : (a 2) ^ 2 = a 1 * a 4

/-- The main theorem -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) :
  (seq.a 1 + seq.a 14) / seq.a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1301_130189


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l1301_130102

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l1301_130102


namespace NUMINAMATH_CALUDE_unique_number_proof_l1301_130132

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ n / 1000 = 6

def move_first_to_last (n : ℕ) : ℕ :=
  (n % 1000) * 10 + (n / 1000)

theorem unique_number_proof :
  ∃! n : ℕ, is_valid_number n ∧ move_first_to_last n = n - 1152 :=
by
  use 6538
  sorry

end NUMINAMATH_CALUDE_unique_number_proof_l1301_130132


namespace NUMINAMATH_CALUDE_am_gm_inequality_and_specific_case_l1301_130131

theorem am_gm_inequality_and_specific_case :
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 → (a + b + c) / 3 ≥ (a * b * c) ^ (1/3)) ∧
  ((4 + 9 + 16) / 3 - (4 * 9 * 16) ^ (1/3) ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_and_specific_case_l1301_130131


namespace NUMINAMATH_CALUDE_megans_earnings_l1301_130167

/-- Calculates the total earnings for a given work schedule and hourly rate -/
def total_earnings (hours_per_day : ℕ) (hourly_rate : ℚ) (days_per_month : ℕ) (months : ℕ) : ℚ :=
  hours_per_day * hourly_rate * days_per_month * months

/-- Proves that Megan's total earnings for two months equal $2400 -/
theorem megans_earnings :
  let hours_per_day : ℕ := 8
  let hourly_rate : ℚ := 15/2
  let days_per_month : ℕ := 20
  let months : ℕ := 2
  total_earnings hours_per_day hourly_rate days_per_month months = 2400 := by
  sorry

#eval total_earnings 8 (15/2) 20 2

end NUMINAMATH_CALUDE_megans_earnings_l1301_130167


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l1301_130168

theorem units_digit_of_quotient (n : ℕ) : 
  (4^1993 + 5^1993) % 3 = 0 ∧ ((4^1993 + 5^1993) / 3) % 10 = 3 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l1301_130168


namespace NUMINAMATH_CALUDE_space_diagonal_of_rectangular_solid_l1301_130123

theorem space_diagonal_of_rectangular_solid (l w h : ℝ) (hl : l = 12) (hw : w = 4) (hh : h = 3) :
  Real.sqrt (l^2 + w^2 + h^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_space_diagonal_of_rectangular_solid_l1301_130123


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1301_130158

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1301_130158


namespace NUMINAMATH_CALUDE_sandwich_bread_count_l1301_130196

/-- The number of pieces of bread needed for a given number of regular and double meat sandwiches -/
def breadNeeded (regularCount : ℕ) (doubleMeatCount : ℕ) : ℕ :=
  2 * regularCount + 3 * doubleMeatCount

/-- Theorem stating that 14 regular sandwiches and 12 double meat sandwiches require 64 pieces of bread -/
theorem sandwich_bread_count : breadNeeded 14 12 = 64 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_bread_count_l1301_130196


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1301_130109

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6 * Complex.I
  let z₂ : ℂ := 4 - 6 * Complex.I
  (z₁ / z₂) + (z₂ / z₁) = (-10 : ℚ) / 13 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1301_130109


namespace NUMINAMATH_CALUDE_tank_fill_time_l1301_130113

theorem tank_fill_time (r1 r2 r3 : ℚ) 
  (h1 : r1 = 1 / 18) 
  (h2 : r2 = 1 / 30) 
  (h3 : r3 = -1 / 45) : 
  (1 / (r1 + r2 + r3)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l1301_130113


namespace NUMINAMATH_CALUDE_unique_grid_solution_l1301_130197

-- Define the grid type
def Grid := Fin 3 → Fin 3 → ℕ

-- Define adjacency
def adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨ 
  (i = k ∧ l.val + 1 = j.val) ∨ 
  (j = l ∧ i.val + 1 = k.val) ∨ 
  (j = l ∧ k.val + 1 = i.val)

-- Define the property of sum of adjacent cells being less than 12
def valid_sum (g : Grid) : Prop :=
  ∀ i j k l, adjacent i j k l → g i j + g k l < 12

-- Define the given partial grid
def partial_grid (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 0 2 = 9 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7

-- Define the property that all numbers from 1 to 9 are used
def all_numbers_used (g : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → ∃ i j, g i j = n

-- The main theorem
theorem unique_grid_solution :
  ∀ g : Grid, 
    valid_sum g → 
    partial_grid g → 
    all_numbers_used g → 
    g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_grid_solution_l1301_130197


namespace NUMINAMATH_CALUDE_intersection_circles_sum_l1301_130186

/-- Given two circles intersecting at (2,3) and (m,2), with centers on the line x+y+n=0, prove m+n = -2 -/
theorem intersection_circles_sum (m n : ℝ) : 
  (∃ (c₁ c₂ : ℝ × ℝ), 
    (c₁.1 + c₁.2 + n = 0) ∧ 
    (c₂.1 + c₂.2 + n = 0) ∧
    ((2 - c₁.1)^2 + (3 - c₁.2)^2 = (2 - c₂.1)^2 + (3 - c₂.2)^2) ∧
    ((m - c₁.1)^2 + (2 - c₁.2)^2 = (m - c₂.1)^2 + (2 - c₂.2)^2) ∧
    ((2 - c₁.1)^2 + (3 - c₁.2)^2 = (m - c₁.1)^2 + (2 - c₁.2)^2) ∧
    ((2 - c₂.1)^2 + (3 - c₂.2)^2 = (m - c₂.1)^2 + (2 - c₂.2)^2)) →
  m + n = -2 := by
sorry

end NUMINAMATH_CALUDE_intersection_circles_sum_l1301_130186


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1301_130198

/-- For an arithmetic sequence {a_n}, if a_3 + a_11 = 22, then a_7 = 11 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence property
  a 3 + a 11 = 22 →                                -- given condition
  a 7 = 11                                         -- conclusion to prove
:= by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1301_130198


namespace NUMINAMATH_CALUDE_ratio_problem_l1301_130138

theorem ratio_problem (N : ℝ) (h1 : (1/1) * (1/3) * (2/5) * N = 25) (h2 : (40/100) * N = 300) :
  (25 : ℝ) / ((1/3) * (2/5) * N) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1301_130138


namespace NUMINAMATH_CALUDE_taehyung_calculation_l1301_130165

theorem taehyung_calculation (x : ℝ) (h : 5 * x = 30) : x / 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_taehyung_calculation_l1301_130165


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1301_130179

/-- Probability of selecting two non-defective pens from a box of pens -/
theorem prob_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (h1 : total_pens = 12) 
  (h2 : defective_pens = 6) : 
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1301_130179


namespace NUMINAMATH_CALUDE_peach_problem_l1301_130136

theorem peach_problem (jake steven jill : ℕ) 
  (h1 : jake = steven - 6)
  (h2 : steven = jill + 18)
  (h3 : jake = 17) : 
  jill = 5 := by
sorry

end NUMINAMATH_CALUDE_peach_problem_l1301_130136


namespace NUMINAMATH_CALUDE_equal_population_after_15_years_l1301_130107

/-- The rate of population increase in Village Y that results in equal populations after 15 years -/
def rate_of_increase_village_y (
  initial_population_x : ℕ
  ) (initial_population_y : ℕ
  ) (decrease_rate_x : ℕ
  ) (years : ℕ
  ) : ℕ :=
  (initial_population_x - decrease_rate_x * years - initial_population_y) / years

theorem equal_population_after_15_years 
  (initial_population_x : ℕ)
  (initial_population_y : ℕ)
  (decrease_rate_x : ℕ)
  (years : ℕ) :
  initial_population_x = 72000 →
  initial_population_y = 42000 →
  decrease_rate_x = 1200 →
  years = 15 →
  rate_of_increase_village_y initial_population_x initial_population_y decrease_rate_x years = 800 :=
by
  sorry

#eval rate_of_increase_village_y 72000 42000 1200 15

end NUMINAMATH_CALUDE_equal_population_after_15_years_l1301_130107


namespace NUMINAMATH_CALUDE_sphere_wedge_volume_l1301_130157

/-- Given a sphere with circumference 18π inches cut into 6 congruent wedges,
    the volume of one wedge is 162π cubic inches. -/
theorem sphere_wedge_volume :
  ∀ (r : ℝ),
  2 * π * r = 18 * π →
  (4 / 3 * π * r^3) / 6 = 162 * π := by
sorry

end NUMINAMATH_CALUDE_sphere_wedge_volume_l1301_130157


namespace NUMINAMATH_CALUDE_floor_sum_existence_l1301_130185

theorem floor_sum_existence : ∃ (a b c : ℝ), 
  (⌊a⌋ + ⌊b⌋ = ⌊a + b⌋) ∧ (⌊a⌋ + ⌊c⌋ < ⌊a + c⌋) := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_existence_l1301_130185


namespace NUMINAMATH_CALUDE_mean_temperature_is_88_75_l1301_130199

def temperatures : List ℚ := [85, 84, 85, 88, 91, 93, 94, 90]

theorem mean_temperature_is_88_75 :
  (temperatures.sum / temperatures.length : ℚ) = 355/4 := by sorry

end NUMINAMATH_CALUDE_mean_temperature_is_88_75_l1301_130199


namespace NUMINAMATH_CALUDE_container_volume_ratio_l1301_130130

theorem container_volume_ratio (V1 V2 V3 : ℚ) : 
  (3/7 : ℚ) * V1 = V2 →  -- First container's juice fills second container
  (3/5 : ℚ) * V3 + (2/3 : ℚ) * ((3/7 : ℚ) * V1) = (4/5 : ℚ) * V3 →  -- Third container's final state
  V1 / V2 = 7/3 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l1301_130130


namespace NUMINAMATH_CALUDE_swimming_speed_problem_l1301_130191

/-- Given a person who swims for 2 hours at speed S and runs for 1 hour at speed 4S, 
    if the total distance covered is 12 miles, then S must equal 2 miles per hour. -/
theorem swimming_speed_problem (S : ℝ) : 
  (2 * S + 1 * (4 * S) = 12) → S = 2 := by
  sorry

end NUMINAMATH_CALUDE_swimming_speed_problem_l1301_130191


namespace NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l1301_130135

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ := m * 100

/-- The dimensions of the large wooden box in meters -/
def largeBoxDimMeters : BoxDimensions := {
  length := 4,
  width := 2,
  height := 4
}

/-- The dimensions of the large wooden box in centimeters -/
def largeBoxDimCm : BoxDimensions := {
  length := metersToCentimeters largeBoxDimMeters.length,
  width := metersToCentimeters largeBoxDimMeters.width,
  height := metersToCentimeters largeBoxDimMeters.height
}

/-- The dimensions of the small rectangular box in centimeters -/
def smallBoxDimCm : BoxDimensions := {
  length := 4,
  width := 2,
  height := 2
}

theorem max_small_boxes_in_large_box :
  (boxVolume largeBoxDimCm) / (boxVolume smallBoxDimCm) = 2000000 := by
  sorry

end NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l1301_130135


namespace NUMINAMATH_CALUDE_trig_identity_l1301_130120

theorem trig_identity (α : ℝ) : -Real.sin α + Real.sqrt 3 * Real.cos α = 2 * Real.sin (α + 2 * Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1301_130120


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l1301_130162

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  long_base : ℝ

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating that the area of the given isosceles trapezoid is approximately 2457.2 -/
theorem isosceles_trapezoid_area :
  let t : IsoscelesTrapezoid := ⟨40, 50, 65⟩
  ∃ ε > 0, |area t - 2457.2| < ε :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l1301_130162


namespace NUMINAMATH_CALUDE_car_pushing_speed_l1301_130155

theorem car_pushing_speed (total_distance total_time first_segment second_segment third_segment : ℝ)
  (second_speed third_speed : ℝ) :
  total_distance = 10 ∧
  total_time = 2 ∧
  first_segment = 3 ∧
  second_segment = 3 ∧
  third_segment = 4 ∧
  second_speed = 3 ∧
  third_speed = 8 ∧
  total_time = first_segment / v + second_segment / second_speed + third_segment / third_speed →
  v = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_car_pushing_speed_l1301_130155


namespace NUMINAMATH_CALUDE_mother_daughter_ages_l1301_130142

/-- Proves the ages of a mother and daughter given certain conditions --/
theorem mother_daughter_ages :
  ∀ (daughter_age mother_age : ℕ),
  mother_age = daughter_age + 22 →
  2 * (2 * daughter_age) = 2 * daughter_age + 22 →
  daughter_age = 11 ∧ mother_age = 33 :=
by
  sorry

#check mother_daughter_ages

end NUMINAMATH_CALUDE_mother_daughter_ages_l1301_130142


namespace NUMINAMATH_CALUDE_opposite_gold_is_black_l1301_130188

-- Define the set of colors
inductive Color
  | Blue
  | Orange
  | Yellow
  | Black
  | Silver
  | Gold

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  right : Face
  left : Face

-- Define the three views of the cube
def view1 (c : Cube) : Prop :=
  c.top.color = Color.Black ∧ c.front.color = Color.Blue ∧ c.right.color = Color.Orange

def view2 (c : Cube) : Prop :=
  c.top.color = Color.Black ∧ c.front.color = Color.Yellow ∧ c.right.color = Color.Orange

def view3 (c : Cube) : Prop :=
  c.top.color = Color.Black ∧ c.front.color = Color.Silver ∧ c.right.color = Color.Orange

-- Theorem statement
theorem opposite_gold_is_black (c : Cube) :
  view1 c → view2 c → view3 c → c.bottom.color = Color.Gold → c.top.color = Color.Black :=
by sorry

end NUMINAMATH_CALUDE_opposite_gold_is_black_l1301_130188


namespace NUMINAMATH_CALUDE_negative_square_cubed_l1301_130112

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l1301_130112


namespace NUMINAMATH_CALUDE_painter_scenario_proof_l1301_130154

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculate the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Theorem: For the given painting scenario, the time to paint the remaining rooms is 49 hours. -/
theorem painter_scenario_proof :
  time_to_paint_remaining 12 7 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_painter_scenario_proof_l1301_130154


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1301_130141

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 169
def C₂ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 9

-- Define the moving circle
structure MovingCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency conditions
def internallyTangent (c : MovingCircle) : Prop :=
  C₁ (c.center.1 - c.radius) (c.center.2)

def externallyTangent (c : MovingCircle) : Prop :=
  C₂ (c.center.1 + c.radius) (c.center.2)

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := x^2 / 64 + y^2 / 48 = 1

-- The theorem to prove
theorem moving_circle_trajectory :
  ∀ (c : MovingCircle),
    internallyTangent c →
    externallyTangent c →
    trajectory c.center.1 c.center.2 :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1301_130141


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1301_130125

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 = b^2 + c^2) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 5 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1301_130125


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l1301_130183

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q ∧ a n > 0

/-- Theorem: For a geometric sequence with positive terms satisfying 2a_1 + a_2 = a_3, 
    the common ratio is 2 -/
theorem geometric_sequence_ratio_two (a : ℕ → ℝ) (q : ℝ) 
    (h_geom : GeometricSequence a q)
    (h_eq : 2 * a 1 + a 2 = a 3) : q = 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_two_l1301_130183


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1301_130151

/-- Given a rectangle with area 540 square centimeters, if its length is increased by 15%
    and its width is decreased by 20%, then its new area is 496.8 square centimeters. -/
theorem rectangle_area_change (l w : ℝ) (h1 : l * w = 540) : 
  (1.15 * l) * (0.8 * w) = 496.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1301_130151


namespace NUMINAMATH_CALUDE_connie_markers_count_l1301_130146

/-- The number of red markers Connie has -/
def red_markers : ℕ := 2315

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 1028

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers

theorem connie_markers_count : total_markers = 3343 := by
  sorry

end NUMINAMATH_CALUDE_connie_markers_count_l1301_130146


namespace NUMINAMATH_CALUDE_inequality_relationship_l1301_130171

theorem inequality_relationship (a b : ℝ) : 
  ¬(((2 : ℝ)^a > (2 : ℝ)^b → (1 : ℝ)/a < (1 : ℝ)/b) ∧ 
    ((1 : ℝ)/a < (1 : ℝ)/b → (2 : ℝ)^a > (2 : ℝ)^b)) :=
sorry

end NUMINAMATH_CALUDE_inequality_relationship_l1301_130171


namespace NUMINAMATH_CALUDE_unique_three_digit_reborn_number_l1301_130129

def is_reborn_number (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    n = 100 * a + 10 * b + c ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    (a ≠ b ∨ b ≠ c ∨ a ≠ c) ∧
    n = (100 * max a (max b c) + 10 * max (min a b) (max (min a c) (min b c)) + min a (min b c)) -
        (100 * min a (min b c) + 10 * min (max a b) (min (max a c) (max b c)) + max a (max b c))

theorem unique_three_digit_reborn_number :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ is_reborn_number n ↔ n = 495 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_reborn_number_l1301_130129


namespace NUMINAMATH_CALUDE_adam_room_capacity_l1301_130119

/-- The number of action figures that can fit on each shelf. -/
def figures_per_shelf : ℕ := 10

/-- The number of shelves in Adam's room. -/
def number_of_shelves : ℕ := 8

/-- The total number of action figures Adam's room could hold. -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

/-- Theorem stating that the total number of action figures Adam's room could hold is 80. -/
theorem adam_room_capacity : total_figures = 80 := by sorry

end NUMINAMATH_CALUDE_adam_room_capacity_l1301_130119


namespace NUMINAMATH_CALUDE_lisa_dvd_rental_l1301_130169

theorem lisa_dvd_rental (total_cost : ℚ) (cost_per_dvd : ℚ) (h1 : total_cost = 4.80) (h2 : cost_per_dvd = 1.20) :
  total_cost / cost_per_dvd = 4 := by
  sorry

end NUMINAMATH_CALUDE_lisa_dvd_rental_l1301_130169


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1301_130164

-- Define the set [1,3]
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - 2

-- Define the solution set
def X : Set ℝ := {x : ℝ | x < -1 ∨ x > 2/3}

-- State the theorem
theorem quadratic_inequality (x : ℝ) :
  (∃ a ∈ A, f a x > 0) → x ∈ X :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1301_130164


namespace NUMINAMATH_CALUDE_checker_rearrangement_impossible_l1301_130173

/-- Represents a 5x5 chessboard -/
def Chessboard := Fin 5 → Fin 5 → Bool

/-- A function that determines if two positions are adjacent (horizontally or vertically) -/
def adjacent (p1 p2 : Fin 5 × Fin 5) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- A function representing the initial placement of checkers -/
def initial_placement : Fin 5 × Fin 5 → Fin 5 × Fin 5 :=
  fun p => p

/-- A function representing the final placement of checkers -/
def final_placement : Fin 5 × Fin 5 → Fin 5 × Fin 5 :=
  sorry

/-- Theorem stating that it's impossible to rearrange the checkers as required -/
theorem checker_rearrangement_impossible :
  ¬ (∀ p : Fin 5 × Fin 5, adjacent (initial_placement p) (final_placement p)) ∧
    (∀ p : Fin 5 × Fin 5, ∃ q, final_placement q = p) :=
sorry

end NUMINAMATH_CALUDE_checker_rearrangement_impossible_l1301_130173


namespace NUMINAMATH_CALUDE_circular_plate_ratio_l1301_130163

theorem circular_plate_ratio (radius : ℝ) (circumference : ℝ) 
  (h1 : radius = 15)
  (h2 : circumference = 90) :
  circumference / (2 * radius) = 3 := by
  sorry

end NUMINAMATH_CALUDE_circular_plate_ratio_l1301_130163


namespace NUMINAMATH_CALUDE_cost_per_load_is_25_cents_l1301_130114

/-- Represents the detergent scenario -/
structure DetergentScenario where
  loads_per_bottle : ℕ
  regular_price : ℚ
  sale_price : ℚ

/-- Calculates the cost per load in cents for a given detergent scenario -/
def cost_per_load_cents (scenario : DetergentScenario) : ℚ :=
  (2 * scenario.sale_price * 100) / (2 * scenario.loads_per_bottle)

/-- Theorem stating that the cost per load is 25 cents for the given scenario -/
theorem cost_per_load_is_25_cents (scenario : DetergentScenario)
  (h1 : scenario.loads_per_bottle = 80)
  (h2 : scenario.regular_price = 25)
  (h3 : scenario.sale_price = 20) :
  cost_per_load_cents scenario = 25 := by
  sorry

#eval cost_per_load_cents { loads_per_bottle := 80, regular_price := 25, sale_price := 20 }

end NUMINAMATH_CALUDE_cost_per_load_is_25_cents_l1301_130114


namespace NUMINAMATH_CALUDE_min_tan_sum_l1301_130106

theorem min_tan_sum (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π) 
  (h3 : α + β < π) 
  (h4 : (Real.cos α - Real.sin α) / (Real.cos α + Real.sin α) = Real.cos (2 * β)) :
  ∃ (m : Real), ∀ (α' β' : Real), 
    (0 < α' ∧ α' < π) → (0 < β' ∧ β' < π) → (α' + β' < π) → 
    ((Real.cos α' - Real.sin α') / (Real.cos α' + Real.sin α') = Real.cos (2 * β')) →
    (Real.tan α' + Real.tan β' ≥ m) ∧ 
    (∃ (α₀ β₀ : Real), Real.tan α₀ + Real.tan β₀ = m) ∧ 
    m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_tan_sum_l1301_130106


namespace NUMINAMATH_CALUDE_max_projection_area_of_special_tetrahedron_l1301_130178

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  side_length : ℝ
  dihedral_angle : ℝ

/-- The area of the projection of the tetrahedron onto a plane -/
noncomputable def projection_area (t : Tetrahedron) (rotation_angle : ℝ) : ℝ :=
  sorry

/-- The maximum area of the projection over all rotation angles -/
noncomputable def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

theorem max_projection_area_of_special_tetrahedron :
  let t : Tetrahedron := { side_length := 1, dihedral_angle := π/3 }
  max_projection_area t = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_projection_area_of_special_tetrahedron_l1301_130178


namespace NUMINAMATH_CALUDE_right_triangle_ab_length_l1301_130172

/-- Given a right triangle ABC in the x-y plane where:
  - ∠B = 90°
  - The length of AC is 100
  - The slope of line segment AC is 4/3
  Prove that the length of AB is 80 -/
theorem right_triangle_ab_length 
  (A B C : ℝ × ℝ) 
  (right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0)
  (ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 100)
  (ac_slope : (C.2 - A.2) / (C.1 - A.1) = 4/3) :
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ab_length_l1301_130172


namespace NUMINAMATH_CALUDE_beat_kevin_record_l1301_130133

/-- The number of additional wings Alan must eat per minute to beat Kevin's record -/
def additional_wings_per_minute (
  kevin_record : ℕ
  ) (
  alan_current_rate : ℕ
  ) (
  time_frame : ℕ
  ) : ℕ :=
  ((kevin_record + 1) - alan_current_rate * time_frame + time_frame - 1) / time_frame

theorem beat_kevin_record (
  kevin_record : ℕ
  ) (
  alan_current_rate : ℕ
  ) (
  time_frame : ℕ
  ) (
  h1 : kevin_record = 64
  ) (
  h2 : alan_current_rate = 5
  ) (
  h3 : time_frame = 8
  ) : additional_wings_per_minute kevin_record alan_current_rate time_frame = 3 := by
  sorry

end NUMINAMATH_CALUDE_beat_kevin_record_l1301_130133
