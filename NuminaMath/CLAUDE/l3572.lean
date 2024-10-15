import Mathlib

namespace NUMINAMATH_CALUDE_max_planes_with_six_points_l3572_357209

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def pointOnPlane (pt : Point3D) (pl : Plane3D) : Prop :=
  pl.a * pt.x + pl.b * pt.y + pl.c * pt.z + pl.d = 0

/-- Check if four points are collinear -/
def areCollinear (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (a b c d : ℝ), ∀ (x y z : ℝ),
    a*x + b*y + c*z + d = 0 ↔ (x = p1.x ∧ y = p1.y ∧ z = p1.z) ∨
                            (x = p2.x ∧ y = p2.y ∧ z = p2.z) ∨
                            (x = p3.x ∧ y = p3.y ∧ z = p3.z) ∨
                            (x = p4.x ∧ y = p4.y ∧ z = p4.z)

/-- Main theorem -/
theorem max_planes_with_six_points
  (points : Fin 6 → Point3D)
  (h_not_collinear : ∀ (i j k l : Fin 6), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
                     ¬ areCollinear (points i) (points j) (points k) (points l)) :
  ∃ (planes : Fin 6 → Plane3D),
    (∀ (i : Fin 6), ∃ (p1 p2 p3 p4 : Fin 6),
      p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
      pointOnPlane (points p1) (planes i) ∧
      pointOnPlane (points p2) (planes i) ∧
      pointOnPlane (points p3) (planes i) ∧
      pointOnPlane (points p4) (planes i)) ∧
    (∀ (newPlane : Plane3D),
      (∃ (p1 p2 p3 p4 : Fin 6),
        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
        pointOnPlane (points p1) newPlane ∧
        pointOnPlane (points p2) newPlane ∧
        pointOnPlane (points p3) newPlane ∧
        pointOnPlane (points p4) newPlane) →
      ∃ (i : Fin 6), newPlane = planes i) :=
by
  sorry


end NUMINAMATH_CALUDE_max_planes_with_six_points_l3572_357209


namespace NUMINAMATH_CALUDE_special_function_properties_l3572_357216

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∃ x, f x ≠ 0) ∧
  (∀ a b : ℝ, f (a * b) = a * f b + b * f a) ∧
  f (1 / 2) = 1

theorem special_function_properties (f : ℝ → ℝ) (h : special_function f) :
  f (1 / 4) = 1 ∧
  f (1 / 8) = 3 / 4 ∧
  f (1 / 16) = 1 / 2 ∧
  ∀ n : ℕ, n > 0 → f (2 ^ (-n : ℝ)) = n * (1 / 2) ^ (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l3572_357216


namespace NUMINAMATH_CALUDE_third_circle_radius_l3572_357294

/-- Given two externally tangent circles and a third circle tangent to both and their common external tangent, prove the radius of the third circle -/
theorem third_circle_radius (center_P center_Q center_R : ℝ × ℝ) 
  (radius_P radius_Q radius_R : ℝ) : 
  radius_P = 2 →
  radius_Q = 6 →
  (center_P.1 - center_Q.1)^2 + (center_P.2 - center_Q.2)^2 = (radius_P + radius_Q)^2 →
  (center_P.1 - center_R.1)^2 + (center_P.2 - center_R.2)^2 = (radius_P + radius_R)^2 →
  (center_Q.1 - center_R.1)^2 + (center_Q.2 - center_R.2)^2 = (radius_Q + radius_R)^2 →
  (∃ (t : ℝ), center_R.2 = t * center_P.2 + (1 - t) * center_Q.2 ∧ 
              center_R.1 = t * center_P.1 + (1 - t) * center_Q.1 ∧ 
              0 ≤ t ∧ t ≤ 1) →
  radius_R = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l3572_357294


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3572_357225

/-- Given two similar triangles PQR and STU, prove that if PQ = 12, QR = 10, and ST = 18, then TU = 15 -/
theorem similar_triangles_side_length 
  (PQ QR ST TU : ℝ) 
  (h_similar : ∃ k : ℝ, k > 0 ∧ PQ = k * ST ∧ QR = k * TU) 
  (h_PQ : PQ = 12) 
  (h_QR : QR = 10) 
  (h_ST : ST = 18) : 
  TU = 15 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3572_357225


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3572_357289

theorem inequality_equivalence (x : ℝ) : 
  5 - 3 / (3 * x - 2) < 7 ↔ x < 1/6 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3572_357289


namespace NUMINAMATH_CALUDE_ellipse_area_l3572_357206

def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + 8 * x + 3 * y^2 - 9 * y + 12 = 0

theorem ellipse_area : 
  ∃ (A : ℝ), A = Real.pi * Real.sqrt 6 / 6 ∧ 
  ∀ (x y : ℝ), ellipse_equation x y → A = Real.pi * Real.sqrt ((1 / 2) * (1 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_area_l3572_357206


namespace NUMINAMATH_CALUDE_custom_mult_example_l3572_357282

/-- Custom multiplication operation for fractions -/
def custom_mult (m n p q : ℚ) : ℚ := m * p * (n / q)

/-- Theorem stating that 5/4 * 6/2 = 60 under the custom multiplication -/
theorem custom_mult_example : custom_mult 5 4 6 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_example_l3572_357282


namespace NUMINAMATH_CALUDE_max_area_enclosure_l3572_357249

/-- Represents a rectangular enclosure with length and width. -/
structure Enclosure where
  length : ℝ
  width : ℝ

/-- The perimeter of the enclosure is 500 feet. -/
def perimeterConstraint (e : Enclosure) : Prop :=
  2 * e.length + 2 * e.width = 500

/-- The length of the enclosure is at least 100 feet. -/
def minLengthConstraint (e : Enclosure) : Prop :=
  e.length ≥ 100

/-- The width of the enclosure is at least 60 feet. -/
def minWidthConstraint (e : Enclosure) : Prop :=
  e.width ≥ 60

/-- The area of the enclosure. -/
def area (e : Enclosure) : ℝ :=
  e.length * e.width

/-- Theorem stating that the maximum area of the enclosure satisfying all constraints is 15625 square feet. -/
theorem max_area_enclosure :
  ∃ (e : Enclosure),
    perimeterConstraint e ∧
    minLengthConstraint e ∧
    minWidthConstraint e ∧
    (∀ (e' : Enclosure),
      perimeterConstraint e' ∧
      minLengthConstraint e' ∧
      minWidthConstraint e' →
      area e' ≤ area e) ∧
    area e = 15625 :=
  sorry

end NUMINAMATH_CALUDE_max_area_enclosure_l3572_357249


namespace NUMINAMATH_CALUDE_original_average_theorem_l3572_357210

theorem original_average_theorem (S : Finset ℝ) (f : ℝ → ℝ) :
  S.card = 7 →
  (∀ x ∈ S, f x = 5 * x) →
  (S.sum f) / S.card = 75 →
  (S.sum id) / S.card = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_original_average_theorem_l3572_357210


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l3572_357247

theorem geometric_progression_ratio (x y z w r : ℂ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 ∧
  x * (y - w) ≠ 0 ∧ y * (z - x) ≠ 0 ∧ z * (w - y) ≠ 0 ∧ w * (x - z) ≠ 0 ∧
  x * (y - w) ≠ y * (z - x) ∧ y * (z - x) ≠ z * (w - y) ∧ z * (w - y) ≠ w * (x - z) ∧
  ∃ (a : ℂ), a ≠ 0 ∧
    y * (z - x) = r * (x * (y - w)) ∧
    z * (w - y) = r * (y * (z - x)) ∧
    w * (x - z) = r * (z * (w - y)) →
  r^3 + r^2 + r + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l3572_357247


namespace NUMINAMATH_CALUDE_unique_recovery_l3572_357227

theorem unique_recovery (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_operations : ∃ (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0),
    ({a, b, c, d} : Set ℝ) = {x + y, x - y, x / y, x * y}) :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧ ({a, b, c, d} : Set ℝ) = {x + y, x - y, x / y, x * y} :=
sorry

end NUMINAMATH_CALUDE_unique_recovery_l3572_357227


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l3572_357283

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 
    1043 = 23 * q + r ∧ 
    r > 0 ∧ 
    ∀ (q' r' : ℕ), 1043 = 23 * q' + r' ∧ r' > 0 → q' - r' ≤ q - r ∧ 
    q - r = 37 := by
  sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l3572_357283


namespace NUMINAMATH_CALUDE_max_y_value_l3572_357245

/-- Given that x and y are negative integers satisfying y = 10x / (10 - x), 
    the maximum value of y is -5 -/
theorem max_y_value (x y : ℤ) 
  (h1 : x < 0) 
  (h2 : y < 0) 
  (h3 : y = 10 * x / (10 - x)) : 
  (∀ z : ℤ, z < 0 ∧ ∃ w : ℤ, w < 0 ∧ z = 10 * w / (10 - w) → z ≤ -5) ∧ 
  (∃ u : ℤ, u < 0 ∧ -5 = 10 * u / (10 - u)) :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l3572_357245


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l3572_357229

def A (a b c d e : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![a, 1, 0, b;
     0, 3, 2, 0;
     c, 4, d, 5;
     6, 0, 7, e]

def B (f g h : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![-7, f,  0, -15;
      g, -20, h,   0;
      0,  2,  5,   0;
      3,  0,  8,   6]

theorem inverse_matrices_sum (a b c d e f g h : ℝ) :
  A a b c d e * B f g h = 1 →
  a + b + c + d + e + f + g + h = 27 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l3572_357229


namespace NUMINAMATH_CALUDE_fashion_show_duration_l3572_357267

/-- The total time for a fashion show runway -/
def fashion_show_time (num_models : ℕ) (bathing_suits_per_model : ℕ) (evening_wear_per_model : ℕ) (time_per_trip : ℕ) : ℕ :=
  (num_models * (bathing_suits_per_model + evening_wear_per_model)) * time_per_trip

/-- Theorem: The fashion show with 6 models, 2 bathing suits and 3 evening wear per model, and 2 minutes per trip takes 60 minutes -/
theorem fashion_show_duration :
  fashion_show_time 6 2 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_fashion_show_duration_l3572_357267


namespace NUMINAMATH_CALUDE_rotation_transformation_l3572_357262

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def ABC : Triangle := ⟨(0, 0), (0, 10), (20, 0)⟩
def DEF : Triangle := ⟨(20, 10), (30, 10), (20, 2)⟩

def rotatePoint (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

def rotateTriangle (center : ℝ × ℝ) (angle : ℝ) (t : Triangle) : Triangle := sorry

theorem rotation_transformation (n x y : ℝ) 
  (h1 : 0 < n ∧ n < 180) 
  (h2 : rotateTriangle (x, y) n ABC = DEF) : 
  n + x + y = 92 := by sorry

end NUMINAMATH_CALUDE_rotation_transformation_l3572_357262


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3572_357211

/-- A rectangular solid with prime edge lengths and volume 231 has surface area 262 -/
theorem rectangular_solid_surface_area : 
  ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 231 →
  2 * (a * b + b * c + a * c) = 262 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3572_357211


namespace NUMINAMATH_CALUDE_locust_jump_symmetry_l3572_357201

/-- A locust on a line -/
structure Locust where
  position : ℝ

/-- A configuration of locusts on a line -/
def LocustConfiguration := List Locust

/-- A property that can be achieved by a locust configuration -/
def ConfigurationProperty := LocustConfiguration → Prop

/-- Jumping to the right -/
def jumpRight (config : LocustConfiguration) : LocustConfiguration := sorry

/-- Jumping to the left -/
def jumpLeft (config : LocustConfiguration) : LocustConfiguration := sorry

/-- Two locusts are 1 mm apart -/
def twoLocustsOneMillimeterApart (config : LocustConfiguration) : Prop := sorry

theorem locust_jump_symmetry 
  (initial_config : LocustConfiguration) 
  (h : ∃ (final_config : LocustConfiguration), 
       twoLocustsOneMillimeterApart final_config ∧ 
       ∃ (n : ℕ), final_config = (jumpRight^[n]) initial_config) :
  ∃ (left_final_config : LocustConfiguration), 
    twoLocustsOneMillimeterApart left_final_config ∧ 
    ∃ (m : ℕ), left_final_config = (jumpLeft^[m]) initial_config := 
by sorry

end NUMINAMATH_CALUDE_locust_jump_symmetry_l3572_357201


namespace NUMINAMATH_CALUDE_equation_root_l3572_357263

theorem equation_root : ∃ x : ℝ, (18 / (x^3 - 8) - 2 / (x - 2) = 1) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_l3572_357263


namespace NUMINAMATH_CALUDE_exponential_decreasing_inequality_l3572_357241

theorem exponential_decreasing_inequality (m n : ℝ) (h1 : m > n) (h2 : n > 0) : 
  (0.3 : ℝ) ^ m < (0.3 : ℝ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_exponential_decreasing_inequality_l3572_357241


namespace NUMINAMATH_CALUDE_subset_size_bound_l3572_357258

/-- Given a natural number n ≥ 2, we define a set A and a family of subsets S with certain properties. -/
theorem subset_size_bound (n : ℕ) (h_n : n ≥ 2) :
  ∃ (A : Finset ℕ) (S : Finset (Finset ℕ)),
    (A = Finset.range (2^(n+1) + 1)) ∧
    (S.card = 2^n) ∧
    (∀ s ∈ S, s ⊆ A) ∧
    (∀ (a b : Finset ℕ) (x y z : ℕ),
      a ∈ S → b ∈ S → x ∈ A → y ∈ A → z ∈ A →
      x < y → y < z → y ∈ a → z ∈ a → x ∈ b → z ∈ b →
      a.card < b.card) →
    ∃ s ∈ S, s.card ≤ 4 * n :=
by
  sorry


end NUMINAMATH_CALUDE_subset_size_bound_l3572_357258


namespace NUMINAMATH_CALUDE_festival_profit_margin_is_five_percent_l3572_357242

/-- Represents the pricing and profit information for an item -/
structure ItemPricing where
  regular_discount : ℝ
  regular_profit_margin : ℝ

/-- Calculates the profit margin during a "buy one get one free" offer -/
def festival_profit_margin (item : ItemPricing) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem stating the profit margin during the shopping festival -/
theorem festival_profit_margin_is_five_percent (item : ItemPricing) 
  (h1 : item.regular_discount = 0.3)
  (h2 : item.regular_profit_margin = 0.47) :
  festival_profit_margin item = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_festival_profit_margin_is_five_percent_l3572_357242


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3572_357297

/-- A line passing through a point and perpendicular to another line -/
structure PerpendicularLine where
  point : ℝ × ℝ
  other_line : ℝ → ℝ → ℝ → ℝ

/-- The equation of the perpendicular line -/
def perpendicular_line_equation (l : PerpendicularLine) : ℝ → ℝ → ℝ → ℝ :=
  fun x y c => 3 * x + 2 * y + c

theorem perpendicular_line_through_point (l : PerpendicularLine)
  (h1 : l.point = (-1, 2))
  (h2 : l.other_line = fun x y c => 2 * x - 3 * y + c) :
  perpendicular_line_equation l (-1) 2 (-1) = 0 ∧
  perpendicular_line_equation l = fun x y c => 3 * x + 2 * y - 1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3572_357297


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l3572_357266

theorem geometric_sequence_second_term :
  ∀ (a : ℕ+) (r : ℕ+),
    a = 5 →
    a * r^4 = 1280 →
    a * r = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l3572_357266


namespace NUMINAMATH_CALUDE_clives_box_balls_l3572_357260

/-- The number of balls in Clive's box -/
def total_balls (blue red green yellow : ℕ) : ℕ := blue + red + green + yellow

/-- Theorem: The total number of balls in Clive's box is 36 -/
theorem clives_box_balls : 
  ∃ (blue red green yellow : ℕ),
    blue = 6 ∧ 
    red = 4 ∧ 
    green = 3 * blue ∧ 
    yellow = 2 * red ∧ 
    total_balls blue red green yellow = 36 := by
  sorry

end NUMINAMATH_CALUDE_clives_box_balls_l3572_357260


namespace NUMINAMATH_CALUDE_inequality_proof_l3572_357219

theorem inequality_proof (a b c : ℝ) (h1 : a > -b) (h2 : -b > 0) (h3 : c < 0) :
  a * (1 - c) > b * (c - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3572_357219


namespace NUMINAMATH_CALUDE_equation_satisfied_at_one_l3572_357200

/-- The function f(x) = 3x - 5 -/
def f (x : ℝ) : ℝ := 3 * x - 5

/-- Theorem stating that the equation 2 * [f(x)] - 16 = f(x - 6) is satisfied when x = 1 -/
theorem equation_satisfied_at_one :
  2 * (f 1) - 16 = f (1 - 6) := by sorry

end NUMINAMATH_CALUDE_equation_satisfied_at_one_l3572_357200


namespace NUMINAMATH_CALUDE_staircase_perimeter_l3572_357212

/-- Represents a staircase-shaped region -/
structure StaircaseRegion where
  -- Eight sides of length 1
  unit_sides : Fin 8 → ℝ
  unit_sides_length : ∀ i, unit_sides i = 1
  -- Area of the region
  area : ℝ
  area_value : area = 53
  -- Other properties of the staircase shape are implicit

/-- The perimeter of a staircase region -/
def perimeter (s : StaircaseRegion) : ℝ := sorry

/-- Theorem stating that the perimeter of the given staircase region is 32 -/
theorem staircase_perimeter (s : StaircaseRegion) : perimeter s = 32 := by
  sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l3572_357212


namespace NUMINAMATH_CALUDE_quadratic_radical_combination_l3572_357237

theorem quadratic_radical_combination (a : ℝ) : 
  (∃ k : ℝ, (k * Real.sqrt 2)^2 = a + 1) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_radical_combination_l3572_357237


namespace NUMINAMATH_CALUDE_chord_length_l3572_357296

-- Define the circle and points
variable (O A B C D : Point)
variable (r : ℝ)

-- Define the circle properties
def is_circle (O : Point) (r : ℝ) : Prop := sorry

-- Define diameter
def is_diameter (O A D : Point) : Prop := sorry

-- Define chord
def is_chord (O A B C : Point) : Prop := sorry

-- Define arc measure
def arc_measure (O C D : Point) : ℝ := sorry

-- Define angle measure
def angle_measure (A B O : Point) : ℝ := sorry

-- Define distance between points
def distance (P Q : Point) : ℝ := sorry

-- Theorem statement
theorem chord_length 
  (h_circle : is_circle O r)
  (h_diameter : is_diameter O A D)
  (h_chord : is_chord O A B C)
  (h_BO : distance B O = 7)
  (h_angle : angle_measure A B O = 45)
  (h_arc : arc_measure O C D = 90) :
  distance B C = 7 := by sorry

end NUMINAMATH_CALUDE_chord_length_l3572_357296


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l3572_357232

theorem sum_of_x_solutions_is_zero :
  ∀ (x₁ x₂ : ℝ),
  (∃ y : ℝ, y = 8 ∧ x₁^2 + y^2 = 145) ∧
  (∃ y : ℝ, y = 8 ∧ x₂^2 + y^2 = 145) ∧
  (∀ x : ℝ, (∃ y : ℝ, y = 8 ∧ x^2 + y^2 = 145) → (x = x₁ ∨ x = x₂)) →
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l3572_357232


namespace NUMINAMATH_CALUDE_negative_215_in_fourth_quadrant_l3572_357291

-- Define a function to convert degrees to the equivalent angle in the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Define a function to determine the quadrant of an angle
def getQuadrant (angle : Int) : Nat :=
  let normalizedAngle := normalizeAngle angle
  if 0 < normalizedAngle && normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle && normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle && normalizedAngle < 270 then 3
  else 4

-- Theorem stating that -215° is in the fourth quadrant
theorem negative_215_in_fourth_quadrant :
  getQuadrant (-215) = 4 := by sorry

end NUMINAMATH_CALUDE_negative_215_in_fourth_quadrant_l3572_357291


namespace NUMINAMATH_CALUDE_test_score_calculation_l3572_357269

theorem test_score_calculation (total_questions correct_answers incorrect_answers score : ℕ) : 
  total_questions = 100 →
  correct_answers + incorrect_answers = total_questions →
  score = correct_answers - 2 * incorrect_answers →
  score = 70 →
  correct_answers = 90 := by
sorry

end NUMINAMATH_CALUDE_test_score_calculation_l3572_357269


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l3572_357287

theorem units_digit_of_expression : ∃ n : ℕ, 
  (13 + Real.sqrt 196)^13 + (13 + Real.sqrt 196)^71 = 10 * n :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l3572_357287


namespace NUMINAMATH_CALUDE_walk_time_calculation_l3572_357214

/-- Represents the walking times between different locations in minutes -/
structure WalkingTimes where
  parkOfficeToHiddenLake : ℝ
  hiddenLakeToParkOffice : ℝ
  parkOfficeToLakeParkRestaurant : ℝ

/-- Represents the wind effect on walking times -/
structure WindEffect where
  favorableReduction : ℝ
  adverseIncrease : ℝ

theorem walk_time_calculation (w : WindEffect) (t : WalkingTimes) : 
  w.favorableReduction = 0.2 →
  w.adverseIncrease = 0.25 →
  t.parkOfficeToHiddenLake * (1 - w.favorableReduction) = 15 →
  t.hiddenLakeToParkOffice * (1 + w.adverseIncrease) = 7 →
  t.parkOfficeToHiddenLake * (1 - w.favorableReduction) + 
    t.hiddenLakeToParkOffice * (1 + w.adverseIncrease) + 
    t.parkOfficeToLakeParkRestaurant * (1 - w.favorableReduction) = 32 →
  t.parkOfficeToLakeParkRestaurant = 12.5 := by
  sorry

#check walk_time_calculation

end NUMINAMATH_CALUDE_walk_time_calculation_l3572_357214


namespace NUMINAMATH_CALUDE_correct_assignment_count_l3572_357202

/-- The number of ways to assign doctors and nurses to schools -/
def assignment_methods (doctors nurses schools : ℕ) : ℕ :=
  if doctors = 2 ∧ nurses = 4 ∧ schools = 2 then 12 else 0

/-- Theorem stating that there are 12 different assignment methods -/
theorem correct_assignment_count :
  assignment_methods 2 4 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_correct_assignment_count_l3572_357202


namespace NUMINAMATH_CALUDE_N_q_odd_iff_prime_power_l3572_357273

/-- The number of integers a such that 0 < a < q/4 and gcd(a,q) = 1 -/
def N_q (q : ℕ) : ℕ :=
  (Finset.filter (fun a => a > 0 ∧ a < q / 4 ∧ Nat.gcd a q = 1) (Finset.range q)).card

/-- A prime p is congruent to 5 or 7 modulo 8 -/
def is_prime_5_or_7_mod_8 (p : ℕ) : Prop :=
  Nat.Prime p ∧ (p % 8 = 5 ∨ p % 8 = 7)

theorem N_q_odd_iff_prime_power (q : ℕ) (h_odd : Odd q) :
  Odd (N_q q) ↔ ∃ (p k : ℕ), q = p^k ∧ k > 0 ∧ is_prime_5_or_7_mod_8 p :=
sorry

end NUMINAMATH_CALUDE_N_q_odd_iff_prime_power_l3572_357273


namespace NUMINAMATH_CALUDE_optimal_division_l3572_357252

theorem optimal_division (a : ℝ) (h : a > 0) :
  let f := fun (x : ℝ) => x / (a - x) + (a - x) / x
  ∃ (x : ℝ), 0 < x ∧ x < a ∧ ∀ (y : ℝ), 0 < y ∧ y < a → f x ≤ f y :=
by sorry

end NUMINAMATH_CALUDE_optimal_division_l3572_357252


namespace NUMINAMATH_CALUDE_building_height_l3572_357299

theorem building_height :
  let standard_floor_height : ℝ := 3
  let taller_floor_height : ℝ := 3.5
  let num_standard_floors : ℕ := 18
  let num_taller_floors : ℕ := 2
  let total_floors : ℕ := num_standard_floors + num_taller_floors
  total_floors = 20 →
  (num_standard_floors : ℝ) * standard_floor_height + (num_taller_floors : ℝ) * taller_floor_height = 61 := by
  sorry

end NUMINAMATH_CALUDE_building_height_l3572_357299


namespace NUMINAMATH_CALUDE_sufficient_condition_for_product_greater_than_four_l3572_357251

theorem sufficient_condition_for_product_greater_than_four (a b : ℝ) :
  a > 2 → b > 2 → a * b > 4 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_product_greater_than_four_l3572_357251


namespace NUMINAMATH_CALUDE_digit_placement_theorem_l3572_357203

def number_of_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  Nat.choose n k * Nat.factorial m

theorem digit_placement_theorem :
  number_of_arrangements 6 2 4 = 360 :=
sorry

end NUMINAMATH_CALUDE_digit_placement_theorem_l3572_357203


namespace NUMINAMATH_CALUDE_consecutive_powers_of_two_divisible_by_six_l3572_357217

theorem consecutive_powers_of_two_divisible_by_six (n : ℕ) :
  6 ∣ (2^n + 2^(n+1)) := by sorry

end NUMINAMATH_CALUDE_consecutive_powers_of_two_divisible_by_six_l3572_357217


namespace NUMINAMATH_CALUDE_left_handed_rock_fans_under_25_l3572_357253

/-- Represents the number of people with specific characteristics in a workshop. -/
structure WorkshopPeople where
  total : ℕ
  leftHanded : ℕ
  rockMusicFans : ℕ
  rightHandedNotRockFans : ℕ
  under25 : ℕ
  rightHandedUnder25RockFans : ℕ

/-- Theorem stating the number of left-handed, rock music fans under 25 in the workshop. -/
theorem left_handed_rock_fans_under_25 (w : WorkshopPeople) 
  (h1 : w.total = 30)
  (h2 : w.leftHanded = 12)
  (h3 : w.rockMusicFans = 18)
  (h4 : w.rightHandedNotRockFans = 5)
  (h5 : w.under25 = 9)
  (h6 : w.rightHandedUnder25RockFans = 3)
  (h7 : w.leftHanded + (w.total - w.leftHanded) = w.total) :
  ∃ x : ℕ, x = 5 ∧ 
    x + (w.leftHanded - x) + (w.rockMusicFans - x) + w.rightHandedNotRockFans + 
    w.rightHandedUnder25RockFans + (w.total - w.leftHanded - w.rightHandedNotRockFans - 
    w.rightHandedUnder25RockFans - x) = w.total :=
  sorry


end NUMINAMATH_CALUDE_left_handed_rock_fans_under_25_l3572_357253


namespace NUMINAMATH_CALUDE_jake_balloons_l3572_357270

theorem jake_balloons (allan_balloons : ℕ) (difference : ℕ) : 
  allan_balloons = 5 → difference = 2 → allan_balloons - difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_jake_balloons_l3572_357270


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l3572_357285

theorem simplify_sqrt_difference : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |((Real.sqrt 528 / Real.sqrt 32) - (Real.sqrt 297 / Real.sqrt 99)) - 2.318| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l3572_357285


namespace NUMINAMATH_CALUDE_volleyball_not_basketball_l3572_357254

theorem volleyball_not_basketball (total : Nat) (basketball : Nat) (volleyball : Nat) (neither : Nat)
  (h1 : total = 40)
  (h2 : basketball = 15)
  (h3 : volleyball = 20)
  (h4 : neither = 10)
  (h5 : total = basketball + volleyball - (basketball + volleyball - (total - neither)) + neither) :
  volleyball - (basketball + volleyball - (total - neither)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_not_basketball_l3572_357254


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l3572_357293

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 15 →
  (n * original_mean - n * decrement) / n = 185 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l3572_357293


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3572_357298

theorem necessary_but_not_sufficient (a b : ℝ) :
  (((a > 1) ∧ (b > 1)) → (a + b > 2)) ∧
  (∃ a b : ℝ, (a + b > 2) ∧ ¬((a > 1) ∧ (b > 1))) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3572_357298


namespace NUMINAMATH_CALUDE_f_24_18_mod_89_l3572_357259

/-- The function f(x) = x^2 - 2 -/
def f (x : ℤ) : ℤ := x^2 - 2

/-- f^n denotes f applied n times -/
def f_iter (n : ℕ) : ℤ → ℤ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ f_iter n

/-- The main theorem stating that f^24(18) ≡ 47 (mod 89) -/
theorem f_24_18_mod_89 : f_iter 24 18 ≡ 47 [ZMOD 89] := by
  sorry


end NUMINAMATH_CALUDE_f_24_18_mod_89_l3572_357259


namespace NUMINAMATH_CALUDE_stratified_sampling_model_c_l3572_357233

theorem stratified_sampling_model_c (total_units : ℕ) (model_c_units : ℕ) (sample_size : ℕ) :
  total_units = 1000 →
  model_c_units = 300 →
  sample_size = 60 →
  (model_c_units * sample_size) / total_units = 18 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_model_c_l3572_357233


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_l3572_357248

theorem max_sum_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 10) : 
  x + y ≤ 93 / 44 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_l3572_357248


namespace NUMINAMATH_CALUDE_unique_prime_value_l3572_357278

def f (n : ℕ) : ℤ := n^3 - 9*n^2 + 23*n - 15

theorem unique_prime_value : ∃! (n : ℕ), n > 0 ∧ Nat.Prime (Int.natAbs (f n)) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_value_l3572_357278


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3572_357271

/-- Given a hyperbola with asymptote equation 4x - 3y = 0 and sharing foci with the ellipse x²/30 + y²/5 = 1, 
    its equation is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ k : ℝ, 4 * x - 3 * y = k) ∧ 
  (∃ c : ℝ, c > 0 ∧ c^2 = 25 ∧ (∀ x y : ℝ, x^2 / 30 + y^2 / 5 = 1 → x^2 ≤ c^2)) →
  (x^2 / 9 - y^2 / 16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3572_357271


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_first_term_l3572_357277

/-- An arithmetic sequence with common difference 2 where a_1, a_2, and a_4 form a geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧ 
  (a 2)^2 = a 1 * a 4

theorem arithmetic_geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h : ArithmeticGeometricSequence a) : 
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_first_term_l3572_357277


namespace NUMINAMATH_CALUDE_seashell_count_l3572_357230

theorem seashell_count (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_l3572_357230


namespace NUMINAMATH_CALUDE_fraction_problem_l3572_357218

theorem fraction_problem (f : ℚ) : f * 300 = (3/5 * 125) + 45 → f = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3572_357218


namespace NUMINAMATH_CALUDE_puppies_difference_l3572_357264

/-- The number of puppies Yuri adopted in the first week -/
def first_week : ℕ := 20

/-- The number of puppies Yuri adopted in the second week -/
def second_week : ℕ := (2 * first_week) / 5

/-- The number of puppies Yuri adopted in the third week -/
def third_week : ℕ := 2 * second_week

/-- The total number of puppies Yuri has after four weeks -/
def total_puppies : ℕ := 74

/-- The number of puppies Yuri adopted in the fourth week -/
def fourth_week : ℕ := total_puppies - (first_week + second_week + third_week)

theorem puppies_difference : fourth_week - first_week = 10 := by
  sorry

end NUMINAMATH_CALUDE_puppies_difference_l3572_357264


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l3572_357257

theorem largest_multiple_of_8_under_100 : ∃ n : ℕ, n * 8 = 96 ∧ 
  (∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l3572_357257


namespace NUMINAMATH_CALUDE_f_fixed_points_l3572_357272

def f (x : ℝ) : ℝ := x^3 - 4*x

theorem f_fixed_points (x : ℝ) : 
  (x = 0 ∨ x = 2 ∨ x = -2) → f (f x) = f x :=
by
  sorry

end NUMINAMATH_CALUDE_f_fixed_points_l3572_357272


namespace NUMINAMATH_CALUDE_least_possible_value_a_2008_l3572_357223

theorem least_possible_value_a_2008 (a : ℕ → ℤ) 
  (h_increasing : ∀ n : ℕ, n ≥ 1 → a n < a (n + 1))
  (h_inequality : ∀ i j k l : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ k ∧ k < l ∧ i + l = j + k → 
    a i + a l > a j + a k) :
  a 2008 ≥ 2015029 := by
sorry

end NUMINAMATH_CALUDE_least_possible_value_a_2008_l3572_357223


namespace NUMINAMATH_CALUDE_bakery_customers_l3572_357250

theorem bakery_customers (total_pastries : ℕ) (regular_customers : ℕ) (pastry_difference : ℕ) :
  total_pastries = 392 →
  regular_customers = 28 →
  pastry_difference = 6 →
  ∃ (actual_customers : ℕ),
    actual_customers * (total_pastries / regular_customers - pastry_difference) = total_pastries ∧
    actual_customers = 49 := by
  sorry

end NUMINAMATH_CALUDE_bakery_customers_l3572_357250


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3572_357280

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 < 0 → x > a) ∧
  (∃ x : ℝ, x > a ∧ ¬(x^2 - 2*x - 3 < 0)) →
  a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3572_357280


namespace NUMINAMATH_CALUDE_consecutive_numbers_proof_l3572_357204

theorem consecutive_numbers_proof (x y z : ℤ) : 
  (x = y + 1) →  -- x, y are consecutive
  (y = z + 1) →  -- y, z are consecutive
  (x > y) →      -- x > y
  (y > z) →      -- y > z
  (2*x + 3*y + 3*z = 5*y + 11) →  -- given equation
  (z = 3) →      -- given value of z
  (3*y = 12) :=  -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_proof_l3572_357204


namespace NUMINAMATH_CALUDE_f_equals_g_l3572_357265

-- Define the functions
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := (x^3)^(1/3)

-- Statement to prove
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_f_equals_g_l3572_357265


namespace NUMINAMATH_CALUDE_lewis_weekly_earnings_l3572_357290

/-- Lewis's earnings during the harvest -/
def total_earnings : ℕ := 1216

/-- Duration of the harvest in weeks -/
def harvest_duration : ℕ := 76

/-- Weekly earnings of Lewis during the harvest -/
def weekly_earnings : ℚ := total_earnings / harvest_duration

theorem lewis_weekly_earnings : weekly_earnings = 16 := by
  sorry

end NUMINAMATH_CALUDE_lewis_weekly_earnings_l3572_357290


namespace NUMINAMATH_CALUDE_equal_cheese_division_l3572_357235

/-- Represents an equilateral triangle cheese -/
structure EquilateralTriangleCheese where
  side_length : ℝ
  area : ℝ

/-- Represents a division of the cheese -/
structure CheeseDivision where
  num_pieces : ℕ
  piece_area : ℝ

/-- The number of people to divide the cheese among -/
def num_people : ℕ := 5

theorem equal_cheese_division 
  (cheese : EquilateralTriangleCheese) 
  (division : CheeseDivision) :
  division.num_pieces = 25 ∧
  division.piece_area * division.num_pieces = cheese.area ∧
  division.num_pieces % num_people = 0 →
  ∃ (pieces_per_person : ℕ), 
    pieces_per_person * num_people = division.num_pieces ∧
    pieces_per_person = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_cheese_division_l3572_357235


namespace NUMINAMATH_CALUDE_line_relationships_l3572_357275

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_relationships (a b c : Line) 
  (h1 : skew a b) (h2 : parallel c a) : 
  ¬ parallel c b := by sorry

end NUMINAMATH_CALUDE_line_relationships_l3572_357275


namespace NUMINAMATH_CALUDE_optimal_allocation_l3572_357295

/-- Represents the production capacity of workers in a workshop --/
structure Workshop where
  total_workers : ℕ
  bolts_per_worker : ℕ
  nuts_per_worker : ℕ
  nuts_per_bolt : ℕ

/-- Represents the allocation of workers to bolt and nut production --/
structure WorkerAllocation where
  bolt_workers : ℕ
  nut_workers : ℕ

/-- Checks if a given allocation is valid for the workshop --/
def is_valid_allocation (w : Workshop) (a : WorkerAllocation) : Prop :=
  a.bolt_workers + a.nut_workers = w.total_workers ∧
  a.bolt_workers * w.bolts_per_worker * w.nuts_per_bolt = a.nut_workers * w.nuts_per_worker

/-- The theorem stating the optimal allocation for the given workshop conditions --/
theorem optimal_allocation (w : Workshop) 
    (h1 : w.total_workers = 28)
    (h2 : w.bolts_per_worker = 12)
    (h3 : w.nuts_per_worker = 18)
    (h4 : w.nuts_per_bolt = 2) :
  ∃ (a : WorkerAllocation), 
    is_valid_allocation w a ∧ 
    a.bolt_workers = 12 ∧ 
    a.nut_workers = 16 := by
  sorry

end NUMINAMATH_CALUDE_optimal_allocation_l3572_357295


namespace NUMINAMATH_CALUDE_problem_2_l3572_357205

def f (x a : ℝ) : ℝ := |x - a|

theorem problem_2 (a m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_solution_set : ∀ x, f x a ≤ 2 ↔ -1 ≤ x ∧ x ≤ 3)
  (h_equation : m + 2*n = 2*m*n - 3*a) : 
  m + 2*n ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_problem_2_l3572_357205


namespace NUMINAMATH_CALUDE_ab_value_l3572_357236

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
    Real.sqrt (log a) = m ∧
    Real.sqrt (log b) = n ∧
    log (Real.sqrt a) = m^2 / 2 ∧
    log (Real.sqrt b) = n^2 / 2 ∧
    m + n + m^2 / 2 + n^2 / 2 = 100) →
  a * b = 10^164 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l3572_357236


namespace NUMINAMATH_CALUDE_square_field_area_l3572_357261

/-- The area of a square field with side length 20 meters is 400 square meters. -/
theorem square_field_area : ∀ (side_length : ℝ), side_length = 20 → side_length * side_length = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l3572_357261


namespace NUMINAMATH_CALUDE_abs_negative_seventeen_l3572_357207

theorem abs_negative_seventeen : |(-17 : ℤ)| = 17 := by sorry

end NUMINAMATH_CALUDE_abs_negative_seventeen_l3572_357207


namespace NUMINAMATH_CALUDE_plate_cup_cost_l3572_357228

/-- Given that 100 paper plates and 200 paper cups cost $6.00 in total,
    prove that 20 paper plates and 40 paper cups cost $1.20. -/
theorem plate_cup_cost (plate_cost cup_cost : ℝ) 
  (h : 100 * plate_cost + 200 * cup_cost = 6) :
  20 * plate_cost + 40 * cup_cost = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_plate_cup_cost_l3572_357228


namespace NUMINAMATH_CALUDE_no_prime_solution_l3572_357292

theorem no_prime_solution : ¬∃ p : ℕ, Nat.Prime p ∧ 2 * p^3 - p^2 - 16 * p + 26 = 0 := by sorry

end NUMINAMATH_CALUDE_no_prime_solution_l3572_357292


namespace NUMINAMATH_CALUDE_total_jelly_beans_l3572_357274

/-- The number of vanilla jelly beans -/
def vanilla : ℕ := 120

/-- The number of grape jelly beans -/
def grape : ℕ := 5 * vanilla + 50

/-- The number of strawberry jelly beans -/
def strawberry : ℕ := (2 * vanilla) / 3

/-- The total number of jelly beans -/
def total : ℕ := grape + vanilla + strawberry

/-- Theorem stating that the total number of jelly beans is 850 -/
theorem total_jelly_beans : total = 850 := by
  sorry

end NUMINAMATH_CALUDE_total_jelly_beans_l3572_357274


namespace NUMINAMATH_CALUDE_samanthas_number_l3572_357221

theorem samanthas_number (x : ℚ) : 5 * ((3 * x + 6) / 2) = 100 → x = 34 / 3 := by
  sorry

end NUMINAMATH_CALUDE_samanthas_number_l3572_357221


namespace NUMINAMATH_CALUDE_woojung_high_school_students_l3572_357276

theorem woojung_high_school_students (first_year : ℕ) (non_first_year : ℕ) : 
  non_first_year = 954 → 
  first_year = non_first_year - 468 → 
  first_year + non_first_year = 1440 := by
sorry

end NUMINAMATH_CALUDE_woojung_high_school_students_l3572_357276


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3572_357234

theorem smallest_n_congruence (n : ℕ+) : (23 * n.val ≡ 5678 [ZMOD 11]) ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3572_357234


namespace NUMINAMATH_CALUDE_ellipse_chord_theorem_l3572_357224

/-- The ellipse type -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- The chord type -/
structure Chord where
  p1 : Point
  p2 : Point

/-- The theorem statement -/
theorem ellipse_chord_theorem (e : Ellipse) (c : Chord) (F1 F2 : Point) :
  e.a = 5 →
  e.b = 4 →
  F1.x = -3 →
  F1.y = 0 →
  F2.x = 3 →
  F2.y = 0 →
  (c.p1.x^2 / 25 + c.p1.y^2 / 16 = 1) →
  (c.p2.x^2 / 25 + c.p2.y^2 / 16 = 1) →
  (c.p1.x - F1.x) * (c.p2.y - F1.y) = (c.p1.y - F1.y) * (c.p2.x - F1.x) →
  (Real.pi = 2 * Real.pi * (Real.sqrt (5 * 5 / 36))) →
  |c.p1.y - c.p2.y| = 5/3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_chord_theorem_l3572_357224


namespace NUMINAMATH_CALUDE_perpendicular_condition_false_l3572_357222

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpLine : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpPlane : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset : Line → Plane → Prop)

theorem perpendicular_condition_false
  (α β : Plane) (b : Line)
  (h_diff : α ≠ β)
  (h_subset : subset b β) :
  ¬(∀ (α β : Plane) (b : Line),
    α ≠ β →
    subset b β →
    (perpLine b α → perpPlane α β) ∧
    ¬(perpPlane α β → perpLine b α)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_false_l3572_357222


namespace NUMINAMATH_CALUDE_polar_equations_and_intersection_ratio_l3572_357240

-- Define the line l in Cartesian coordinates
def line_l (x y : ℝ) : Prop := x = 4

-- Define the curve C in Cartesian coordinates
def curve_C (x y φ : ℝ) : Prop := x = 1 + Real.sqrt 2 * Real.cos φ ∧ y = 1 + Real.sqrt 2 * Real.sin φ

-- Define the transformation from Cartesian to polar coordinates
def to_polar (x y ρ θ : ℝ) : Prop := x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- State the theorem
theorem polar_equations_and_intersection_ratio :
  ∀ (x y ρ θ φ α : ℝ),
  (line_l x y → ρ * Real.cos θ = 4) ∧
  (curve_C x y φ → ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) ∧
  (0 < α ∧ α < Real.pi / 4 →
    ∃ (ρ_A ρ_B : ℝ),
      ρ_A = 2 * (Real.cos α + Real.sin α) ∧
      ρ_B = 4 / Real.cos α ∧
      1 / 2 < ρ_A / ρ_B ∧ ρ_A / ρ_B ≤ (Real.sqrt 2 + 1) / 4) := by
  sorry

end NUMINAMATH_CALUDE_polar_equations_and_intersection_ratio_l3572_357240


namespace NUMINAMATH_CALUDE_black_balloons_problem_l3572_357279

theorem black_balloons_problem (x y : ℝ) (h1 : x = 4 * y) (h2 : x = 7.0) : y = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_black_balloons_problem_l3572_357279


namespace NUMINAMATH_CALUDE_area_difference_of_square_fields_l3572_357268

/-- Given two square fields where the second field's side length is 1% longer than the first,
    and the area of the first field is 1 hectare (10,000 square meters),
    prove that the difference in area between the two fields is 201 square meters. -/
theorem area_difference_of_square_fields (a : ℝ) : 
  a^2 = 10000 → (1.01 * a)^2 - a^2 = 201 := by
  sorry

end NUMINAMATH_CALUDE_area_difference_of_square_fields_l3572_357268


namespace NUMINAMATH_CALUDE_basketball_average_points_l3572_357213

/-- Given a basketball player who scored 60 points in 5 games, 
    prove that their average points per game is 12. -/
theorem basketball_average_points (total_points : ℕ) (num_games : ℕ) 
  (h1 : total_points = 60) (h2 : num_games = 5) : 
  total_points / num_games = 12 := by
  sorry

end NUMINAMATH_CALUDE_basketball_average_points_l3572_357213


namespace NUMINAMATH_CALUDE_smallest_natural_numbers_for_nested_root_l3572_357220

theorem smallest_natural_numbers_for_nested_root (a b : ℕ) : 
  (b > 1) → 
  (Real.sqrt (a * Real.sqrt (a * Real.sqrt a)) = b) → 
  (∀ a' b' : ℕ, b' > 1 → Real.sqrt (a' * Real.sqrt (a' * Real.sqrt a')) = b' → a ≤ a' ∧ b ≤ b') →
  a = 256 ∧ b = 128 := by
sorry

end NUMINAMATH_CALUDE_smallest_natural_numbers_for_nested_root_l3572_357220


namespace NUMINAMATH_CALUDE_vacation_pictures_l3572_357284

theorem vacation_pictures (zoo museum beach deleted : ℕ) :
  zoo = 120 →
  museum = 75 →
  beach = 45 →
  deleted = 93 →
  zoo + museum + beach - deleted = 147 :=
by sorry

end NUMINAMATH_CALUDE_vacation_pictures_l3572_357284


namespace NUMINAMATH_CALUDE_star_four_six_l3572_357226

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2

-- Theorem statement
theorem star_four_six : star 4 6 = 100 := by
  sorry

end NUMINAMATH_CALUDE_star_four_six_l3572_357226


namespace NUMINAMATH_CALUDE_smallest_shift_l3572_357244

-- Define a function that repeats every 15 units horizontally
def is_periodic_15 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 15) = f x

-- Define the property we're looking for
def satisfies_shift (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f ((x - b) / 3) = f (x / 3)

-- State the theorem
theorem smallest_shift (f : ℝ → ℝ) (h : is_periodic_15 f) :
  ∃ b : ℝ, b > 0 ∧ satisfies_shift f b ∧ ∀ b' : ℝ, b' > 0 ∧ satisfies_shift f b' → b ≤ b' :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_l3572_357244


namespace NUMINAMATH_CALUDE_average_tree_height_l3572_357231

def tree_heights : List ℝ := [1000, 500, 500, 1200]

theorem average_tree_height : (tree_heights.sum / tree_heights.length : ℝ) = 800 := by
  sorry

end NUMINAMATH_CALUDE_average_tree_height_l3572_357231


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l3572_357243

theorem max_product_sum_2000 : 
  ∃ (a b : ℤ), a + b = 2000 ∧ 
    ∀ (x y : ℤ), x + y = 2000 → x * y ≤ a * b ∧ 
    a * b = 1000000 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l3572_357243


namespace NUMINAMATH_CALUDE_ryan_to_bill_ratio_l3572_357238

/-- Represents the number of math problems composed by each person -/
structure ProblemCounts where
  bill : ℕ
  ryan : ℕ
  frank : ℕ

/-- Represents the conditions of the problem -/
def problem_conditions (p : ProblemCounts) : Prop :=
  p.bill = 20 ∧
  p.frank = 3 * p.ryan ∧
  p.frank = 30 * 4

/-- The theorem to be proved -/
theorem ryan_to_bill_ratio 
  (p : ProblemCounts) 
  (h : problem_conditions p) : 
  p.ryan / p.bill = 2 := by
  sorry

#check ryan_to_bill_ratio

end NUMINAMATH_CALUDE_ryan_to_bill_ratio_l3572_357238


namespace NUMINAMATH_CALUDE_coin_toss_probability_l3572_357239

def toss_outcomes (n : ℕ) : ℕ := 2^n

def favorable_outcomes : ℕ := 5

theorem coin_toss_probability :
  let mina_tosses : ℕ := 2
  let liam_tosses : ℕ := 3
  let total_outcomes : ℕ := toss_outcomes mina_tosses * toss_outcomes liam_tosses
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 32 := by sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l3572_357239


namespace NUMINAMATH_CALUDE_circle_properties_l3572_357215

-- Define the points
def O : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (1, 1)
def N : ℝ × ℝ := (4, 2)

-- Define the circle equation
def circle_equation (x y : ℝ) := x^2 + y^2 - 4*x + 3*y

-- Define the center and radius
def center : ℝ × ℝ := (4, -3)
def radius : ℝ := 5

theorem circle_properties :
  (circle_equation O.1 O.2 = 0) ∧
  (circle_equation M.1 M.2 = 0) ∧
  (circle_equation N.1 N.2 = 0) ∧
  (∀ (x y : ℝ), circle_equation x y = 0 ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3572_357215


namespace NUMINAMATH_CALUDE_sum_of_exponents_l3572_357281

theorem sum_of_exponents (a b c : ℕ+) : 
  4^(a : ℕ) * 5^(b : ℕ) * 6^(c : ℕ) = 8^8 * 9^9 * 10^10 → 
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 36 := by
sorry

end NUMINAMATH_CALUDE_sum_of_exponents_l3572_357281


namespace NUMINAMATH_CALUDE_jack_email_difference_l3572_357246

/-- Given the number of emails Jack received at different times of the day,
    prove that he received 2 more emails in the morning than in the afternoon. -/
theorem jack_email_difference (morning afternoon evening : ℕ) 
    (h1 : morning = 5)
    (h2 : afternoon = 3)
    (h3 : evening = 16) :
    morning - afternoon = 2 := by
  sorry

end NUMINAMATH_CALUDE_jack_email_difference_l3572_357246


namespace NUMINAMATH_CALUDE_range_of_a_in_system_with_one_integer_solution_l3572_357288

/-- Given a system of inequalities with exactly one integer solution, prove the range of a -/
theorem range_of_a_in_system_with_one_integer_solution :
  ∀ a : ℝ,
  (∃! x : ℤ, (2 * (x : ℝ) + 3 > 5 ∧ (x : ℝ) - a ≤ 0)) →
  (2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_in_system_with_one_integer_solution_l3572_357288


namespace NUMINAMATH_CALUDE_triangles_5_4_l3572_357256

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of triangles formed by points on two parallel lines -/
def triangles_on_parallel_lines (points_on_line_a points_on_line_b : ℕ) : ℕ :=
  choose points_on_line_a 2 * choose points_on_line_b 1 +
  choose points_on_line_a 1 * choose points_on_line_b 2

/-- Theorem: The number of triangles formed by 5 points on one line and 4 points on a parallel line -/
theorem triangles_5_4 : triangles_on_parallel_lines 5 4 = choose 5 2 * choose 4 1 + choose 5 1 * choose 4 2 := by
  sorry

end NUMINAMATH_CALUDE_triangles_5_4_l3572_357256


namespace NUMINAMATH_CALUDE_earth_surface_area_scientific_notation_l3572_357255

/-- The surface area of the Earth in square kilometers -/
def earth_surface_area : ℝ := 510000000

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Conversion of a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem earth_surface_area_scientific_notation :
  to_scientific_notation earth_surface_area = ScientificNotation.mk 5.1 8 sorry := by
  sorry

end NUMINAMATH_CALUDE_earth_surface_area_scientific_notation_l3572_357255


namespace NUMINAMATH_CALUDE_solve_jury_duty_problem_l3572_357208

def jury_duty_problem (jury_selection_days : ℕ) (trial_multiplier : ℕ) (deliberation_full_days : ℕ) (total_days : ℕ) : Prop :=
  let trial_days : ℕ := trial_multiplier * jury_selection_days
  let selection_and_trial_days : ℕ := jury_selection_days + trial_days
  let actual_deliberation_days : ℕ := total_days - selection_and_trial_days
  let deliberation_hours : ℕ := deliberation_full_days * 24
  deliberation_hours / actual_deliberation_days = 16

theorem solve_jury_duty_problem : 
  jury_duty_problem 2 4 6 19 := by
  sorry

end NUMINAMATH_CALUDE_solve_jury_duty_problem_l3572_357208


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3572_357286

theorem quadratic_inequality_solution (x : ℝ) :
  2 * x^2 - 4 * x - 70 > 0 ∧ x ≠ -2 ∧ x ≠ 0 →
  x < -5 ∨ x > 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3572_357286
