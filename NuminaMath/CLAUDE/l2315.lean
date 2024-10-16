import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2315_231514

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2315_231514


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l2315_231520

theorem reciprocal_sum_of_roots (m n : ℝ) : 
  m^2 - 4*m - 2 = 0 → n^2 - 4*n - 2 = 0 → m ≠ n → 1/m + 1/n = -2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l2315_231520


namespace NUMINAMATH_CALUDE_jordan_oreos_l2315_231553

theorem jordan_oreos (jordan : ℕ) (james : ℕ) : 
  james = 2 * jordan + 3 → 
  jordan + james = 36 → 
  jordan = 11 := by
sorry

end NUMINAMATH_CALUDE_jordan_oreos_l2315_231553


namespace NUMINAMATH_CALUDE_circle_center_locus_l2315_231524

-- Define the circle C
def circle_C (a x y : ℝ) : Prop :=
  x^2 + y^2 - (2*a^2 - 4)*x - 4*a^2*y + 5*a^4 - 4 = 0

-- Define the locus of the center
def center_locus (x y : ℝ) : Prop :=
  y = 2*x + 4 ∧ -2 ≤ x ∧ x < 0

-- Theorem statement
theorem circle_center_locus :
  ∀ a x y : ℝ, circle_C a x y → ∃ h k : ℝ, center_locus h k ∧ 
  (h = a^2 - 2 ∧ k = 2*a^2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_locus_l2315_231524


namespace NUMINAMATH_CALUDE_a_profit_calculation_l2315_231551

def total_subscription : ℕ := 50000
def total_profit : ℕ := 36000

def subscription_difference_a_b : ℕ := 4000
def subscription_difference_b_c : ℕ := 5000

def c_subscription (x : ℕ) : ℕ := x
def b_subscription (x : ℕ) : ℕ := x + subscription_difference_b_c
def a_subscription (x : ℕ) : ℕ := x + subscription_difference_b_c + subscription_difference_a_b

theorem a_profit_calculation :
  ∃ x : ℕ,
    c_subscription x + b_subscription x + a_subscription x = total_subscription ∧
    (a_subscription x : ℚ) / (total_subscription : ℚ) * (total_profit : ℚ) = 15120 :=
  sorry

end NUMINAMATH_CALUDE_a_profit_calculation_l2315_231551


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2315_231599

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  ∃ (M : ℝ), M = 30 + 20*Real.sqrt 3 ∧ 
  ∀ (z w : ℝ), z > 0 → w > 0 → z^2 - 2*z*w + 3*w^2 = 10 → 
  z^2 + 2*z*w + 3*w^2 ≤ M := by
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2315_231599


namespace NUMINAMATH_CALUDE_max_integer_squared_inequality_l2315_231578

theorem max_integer_squared_inequality : ∃ (n : ℕ),
  n = 30499 ∧ 
  n^2 ≤ 160 * 170 * 180 * 190 ∧
  ∀ (m : ℕ), m > n → m^2 > 160 * 170 * 180 * 190 := by
  sorry

end NUMINAMATH_CALUDE_max_integer_squared_inequality_l2315_231578


namespace NUMINAMATH_CALUDE_transformed_triangle_area_equality_l2315_231513

-- Define the domain
variable (x₁ x₂ x₃ : ℝ)

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the area function for a triangle given three points
def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem transformed_triangle_area_equality 
  (h₁ : triangle_area (x₁, f x₁) (x₂, f x₂) (x₃, f x₃) = 50)
  (h₂ : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁) :
  triangle_area (x₁/3, 3 * f x₁) (x₂/3, 3 * f x₂) (x₃/3, 3 * f x₃) = 50 := by
  sorry

end NUMINAMATH_CALUDE_transformed_triangle_area_equality_l2315_231513


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2315_231526

/-- A quadratic function with vertex form (x + h)^2 + k -/
def QuadraticFunction (a h k : ℝ) (x : ℝ) : ℝ := a * (x + h)^2 + k

theorem quadratic_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = QuadraticFunction a 3 0 x) →  -- vertex at (-3, 0)
  f 2 = -50 →                              -- passes through (2, -50)
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2315_231526


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2315_231547

theorem polynomial_remainder (x : ℝ) : 
  (x^4 - 4*x^2 + 7) % (x - 1) = 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2315_231547


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2315_231579

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1}

theorem complement_of_A_in_U : (U \ A) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2315_231579


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l2315_231525

def english_marks : ℕ := 96
def math_marks : ℕ := 98
def physics_marks : ℕ := 99
def biology_marks : ℕ := 98
def average_marks : ℚ := 98.2
def num_subjects : ℕ := 5

theorem davids_chemistry_marks :
  ∃ (chemistry_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    chemistry_marks = 100 := by
  sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l2315_231525


namespace NUMINAMATH_CALUDE_age_problem_l2315_231545

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 47 →
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2315_231545


namespace NUMINAMATH_CALUDE_sum_product_ratio_l2315_231592

theorem sum_product_ratio (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : x ≠ z) (h4 : x + y + z = 1) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = (x * y + y * z + z * x) / (1 - 2 * (x * y + y * z + z * x)) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_ratio_l2315_231592


namespace NUMINAMATH_CALUDE_simons_blueberry_pies_l2315_231570

/-- Simon's blueberry pie problem -/
theorem simons_blueberry_pies :
  ∀ (own_blueberries nearby_blueberries blueberries_per_pie : ℕ),
    own_blueberries = 100 →
    nearby_blueberries = 200 →
    blueberries_per_pie = 100 →
    (own_blueberries + nearby_blueberries) / blueberries_per_pie = 3 :=
by
  sorry

#check simons_blueberry_pies

end NUMINAMATH_CALUDE_simons_blueberry_pies_l2315_231570


namespace NUMINAMATH_CALUDE_banana_distribution_l2315_231574

/-- The number of bananas each child would normally receive -/
def normal_bananas : ℕ := 2

/-- The number of absent children -/
def absent_children : ℕ := 330

/-- The number of extra bananas each child received due to absences -/
def extra_bananas : ℕ := 2

/-- The actual number of children in the school -/
def actual_children : ℕ := 660

theorem banana_distribution (total_bananas : ℕ) :
  (total_bananas = normal_bananas * actual_children) ∧
  (total_bananas = (normal_bananas + extra_bananas) * (actual_children - absent_children)) →
  actual_children = 660 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l2315_231574


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2315_231541

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2315_231541


namespace NUMINAMATH_CALUDE_ellipse_properties_l2315_231558

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_major_axis : 2 * b = a
  h_rhombus_area : 4 * a * b = 8

/-- A line passing through a point on the ellipse -/
structure IntersectingLine (ε : Ellipse) where
  k : ℝ
  h_length : (4 * Real.sqrt 2) / 5 = 4 * Real.sqrt (1 + k^2) / (1 + 4 * k^2)

/-- A point on the perpendicular bisector of the chord -/
structure PerpendicularPoint (ε : Ellipse) (l : IntersectingLine ε) where
  y₀ : ℝ
  h_dot_product : 4 = (y₀^2 + ε.a^2) - (y₀^2 + (ε.a * (1 - k^2) / (1 + k^2))^2)

/-- The main theorem capturing the problem's assertions -/
theorem ellipse_properties (ε : Ellipse) (l : IntersectingLine ε) (p : PerpendicularPoint ε l) :
  ε.a = 2 ∧ ε.b = 1 ∧
  (l.k = 1 ∨ l.k = -1) ∧
  (p.y₀ = 2 * Real.sqrt 2 ∨ p.y₀ = -2 * Real.sqrt 2 ∨
   p.y₀ = 2 * Real.sqrt 14 / 5 ∨ p.y₀ = -2 * Real.sqrt 14 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2315_231558


namespace NUMINAMATH_CALUDE_concurrent_iff_concyclic_l2315_231512

/-- Two circles in a plane -/
structure TwoCircles where
  C₁ : Set (ℝ × ℝ)
  C₂ : Set (ℝ × ℝ)

/-- Points on the circles -/
structure CirclePoints (tc : TwoCircles) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  h_AB_intersect : A ∈ tc.C₁ ∧ A ∈ tc.C₂ ∧ B ∈ tc.C₁ ∧ B ∈ tc.C₂
  h_CD_on_C₁ : C ∈ tc.C₁ ∧ D ∈ tc.C₁
  h_EF_on_C₂ : E ∈ tc.C₂ ∧ F ∈ tc.C₂

/-- Define a line through two points -/
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Three lines are concurrent if they all intersect at a single point -/
def AreConcurrent (l₁ l₂ l₃ : Set (ℝ × ℝ)) : Prop := sorry

/-- Four points are concyclic if they lie on the same circle -/
def AreConcyclic (p q r s : ℝ × ℝ) : Prop := sorry

/-- The main theorem -/
theorem concurrent_iff_concyclic (tc : TwoCircles) (pts : CirclePoints tc) :
  AreConcurrent (Line pts.E pts.F) (Line pts.C pts.D) (Line pts.A pts.B) ↔
  AreConcyclic pts.E pts.F pts.C pts.D := by sorry

end NUMINAMATH_CALUDE_concurrent_iff_concyclic_l2315_231512


namespace NUMINAMATH_CALUDE_vanessa_camera_pictures_l2315_231582

/-- The number of pictures Vanessa uploaded from her camera -/
def camera_pictures (phone_pictures album_count pictures_per_album : ℕ) : ℕ :=
  album_count * pictures_per_album - phone_pictures

/-- Proof that Vanessa uploaded 7 pictures from her camera -/
theorem vanessa_camera_pictures :
  camera_pictures 23 5 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_camera_pictures_l2315_231582


namespace NUMINAMATH_CALUDE_expected_value_8_sided_die_l2315_231586

def standard_8_sided_die : Finset ℕ := Finset.range 8

theorem expected_value_8_sided_die :
  let outcomes := standard_8_sided_die
  let probability (n : ℕ) := (1 : ℚ) / 8
  let expected_value := (outcomes.sum (λ n => (n + 1 : ℚ) * probability n)) / outcomes.card
  expected_value = 9/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_8_sided_die_l2315_231586


namespace NUMINAMATH_CALUDE_jeff_cabinet_count_l2315_231517

/-- The number of cabinets Jeff has after installation and removal -/
def total_cabinets : ℕ :=
  let initial_cabinets := 3
  let counters_with_double := 4
  let cabinets_per_double_counter := 2 * initial_cabinets
  let additional_cabinets := [3, 5, 7]
  let cabinets_to_remove := 2

  initial_cabinets + 
  counters_with_double * cabinets_per_double_counter + 
  additional_cabinets.sum - 
  cabinets_to_remove

theorem jeff_cabinet_count : total_cabinets = 37 := by
  sorry

end NUMINAMATH_CALUDE_jeff_cabinet_count_l2315_231517


namespace NUMINAMATH_CALUDE_gcd_768_288_l2315_231537

theorem gcd_768_288 : Int.gcd 768 288 = 96 := by sorry

end NUMINAMATH_CALUDE_gcd_768_288_l2315_231537


namespace NUMINAMATH_CALUDE_event_probability_l2315_231521

theorem event_probability (n : ℕ) (p_at_least_once : ℚ) (p_single : ℚ) : 
  n = 4 →
  p_at_least_once = 65 / 81 →
  (1 - p_single) ^ n = 1 - p_at_least_once →
  p_single = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_event_probability_l2315_231521


namespace NUMINAMATH_CALUDE_jinas_teddies_l2315_231569

/-- Proves that the initial number of teddies is 5 given the conditions in Jina's mascot collection problem -/
theorem jinas_teddies :
  ∀ (initial_teddies : ℕ),
  let bunnies := 3 * initial_teddies
  let additional_teddies := 2 * bunnies
  let total_mascots := initial_teddies + bunnies + additional_teddies + 1
  total_mascots = 51 →
  initial_teddies = 5 := by
sorry

end NUMINAMATH_CALUDE_jinas_teddies_l2315_231569


namespace NUMINAMATH_CALUDE_fraction_inequality_l2315_231518

theorem fraction_inequality (x y : ℝ) (h : x / y = 3 / 4) :
  (2 * x + y) / y ≠ 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2315_231518


namespace NUMINAMATH_CALUDE_alex_has_more_listens_l2315_231597

/-- Calculates total listens over 3 months given initial listens and monthly growth rate -/
def totalListens (initial : ℝ) (growthRate : ℝ) : ℝ :=
  initial + initial * growthRate + initial * growthRate^2

/-- Represents the streaming statistics for a song -/
structure SongStats where
  spotify : ℝ
  appleMusic : ℝ
  youtube : ℝ

/-- Calculates total listens across all platforms -/
def overallListens (initial : SongStats) (growth : SongStats) : ℝ :=
  totalListens initial.spotify growth.spotify +
  totalListens initial.appleMusic growth.appleMusic +
  totalListens initial.youtube growth.youtube

/-- Jordan's initial listens -/
def jordanInitial : SongStats := ⟨60000, 35000, 45000⟩

/-- Jordan's monthly growth rates -/
def jordanGrowth : SongStats := ⟨2, 1.5, 1.25⟩

/-- Alex's initial listens -/
def alexInitial : SongStats := ⟨75000, 50000, 65000⟩

/-- Alex's monthly growth rates -/
def alexGrowth : SongStats := ⟨1.5, 1.8, 1.1⟩

theorem alex_has_more_listens :
  overallListens alexInitial alexGrowth > overallListens jordanInitial jordanGrowth :=
by sorry

end NUMINAMATH_CALUDE_alex_has_more_listens_l2315_231597


namespace NUMINAMATH_CALUDE_sandbox_sand_weight_l2315_231538

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ := r.length * r.width

/-- Calculates the total area of two rectangles -/
def totalArea (r1 r2 : Rectangle) : ℕ := rectangleArea r1 + rectangleArea r2

/-- Calculates the number of bags needed to fill an area -/
def bagsNeeded (area : ℕ) (areaPerBag : ℕ) : ℕ := (area + areaPerBag - 1) / areaPerBag

/-- Theorem: The total weight of sand needed to fill the sandbox -/
theorem sandbox_sand_weight :
  let rectangle1 : Rectangle := ⟨50, 30⟩
  let rectangle2 : Rectangle := ⟨20, 15⟩
  let areaPerBag : ℕ := 80
  let weightPerBag : ℕ := 30
  let totalSandboxArea : ℕ := totalArea rectangle1 rectangle2
  let bags : ℕ := bagsNeeded totalSandboxArea areaPerBag
  bags * weightPerBag = 690 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_sand_weight_l2315_231538


namespace NUMINAMATH_CALUDE_range_of_a_l2315_231540

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (a*x^2 - x + 2)

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
    (a > 0) → 
    (a ≠ 1) → 
    ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
    ((0 < a ∧ a ≤ 1/8) ∨ (a ≥ 1)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2315_231540


namespace NUMINAMATH_CALUDE_parabola_vertex_l2315_231544

/-- The vertex of the parabola y = 3x^2 + 2 has coordinates (0, 2) -/
theorem parabola_vertex (x y : ℝ) : y = 3 * x^2 + 2 → (0, 2) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2315_231544


namespace NUMINAMATH_CALUDE_income_calculation_l2315_231550

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 8 = expenditure * 15 →
  income - expenditure = savings →
  savings = 7000 →
  income = 15000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l2315_231550


namespace NUMINAMATH_CALUDE_bike_shop_wheels_l2315_231527

/-- The number of wheels on all vehicles in a bike shop -/
def total_wheels (num_bicycles num_tricycles : ℕ) : ℕ :=
  2 * num_bicycles + 3 * num_tricycles

/-- Theorem stating the total number of wheels in the bike shop -/
theorem bike_shop_wheels : total_wheels 50 20 = 160 := by
  sorry

end NUMINAMATH_CALUDE_bike_shop_wheels_l2315_231527


namespace NUMINAMATH_CALUDE_three_equal_differences_l2315_231554

theorem three_equal_differences (n : ℕ) (a : Fin (2*n) → ℕ) 
  (h1 : n > 2)
  (h2 : ∀ i j, i ≠ j → a i ≠ a j)
  (h3 : ∀ i, a i ≤ n^2)
  (h4 : ∀ i, a i > 0) :
  ∃ i₁ j₁ i₂ j₂ i₃ j₃, 
    i₁ ≠ j₁ ∧ i₂ ≠ j₂ ∧ i₃ ≠ j₃ ∧
    (i₁, j₁) ≠ (i₂, j₂) ∧ (i₁, j₁) ≠ (i₃, j₃) ∧ (i₂, j₂) ≠ (i₃, j₃) ∧
    a i₁ - a j₁ = a i₂ - a j₂ ∧ a i₁ - a j₁ = a i₃ - a j₃ :=
by sorry

end NUMINAMATH_CALUDE_three_equal_differences_l2315_231554


namespace NUMINAMATH_CALUDE_room_tiles_proof_l2315_231522

/-- Calculates the least number of square tiles required to pave a rectangular floor -/
def leastSquareTiles (length width : ℕ) : ℕ :=
  let gcd := Nat.gcd length width
  (length * width) / (gcd * gcd)

theorem room_tiles_proof (length width : ℕ) 
  (h_length : length = 5000)
  (h_width : width = 1125) :
  leastSquareTiles length width = 360 := by
  sorry

#eval leastSquareTiles 5000 1125

end NUMINAMATH_CALUDE_room_tiles_proof_l2315_231522


namespace NUMINAMATH_CALUDE_max_rabbits_l2315_231505

theorem max_rabbits (N : ℕ) 
  (long_ears : ℕ) (jump_far : ℕ) (both : ℕ) :
  long_ears = 13 →
  jump_far = 17 →
  both ≥ 3 →
  long_ears + jump_far - both ≤ N →
  N ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_rabbits_l2315_231505


namespace NUMINAMATH_CALUDE_linear_function_property_l2315_231539

/-- A linear function is a function of the form f(x) = mx + b for some constants m and b. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) (h_linear : LinearFunction g) 
    (h_diff : g 4 - g 1 = 9) : g 10 - g 1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l2315_231539


namespace NUMINAMATH_CALUDE_OPRQ_shape_l2315_231530

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A figure formed by four points -/
structure Quadrilateral where
  O : Point
  P : Point
  R : Point
  Q : Point

/-- Check if three points are collinear -/
def collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) = (C.x - A.x) * (B.y - A.y)

/-- Check if two line segments are parallel -/
def parallel (A B C D : Point) : Prop :=
  (B.x - A.x) * (D.y - C.y) = (D.x - C.x) * (B.y - A.y)

/-- Check if a quadrilateral is a straight line -/
def isStraightLine (quad : Quadrilateral) : Prop :=
  collinear quad.O quad.P quad.Q ∧ collinear quad.O quad.R quad.Q

/-- Check if a quadrilateral is a trapezoid -/
def isTrapezoid (quad : Quadrilateral) : Prop :=
  (parallel quad.O quad.P quad.Q quad.R ∧ ¬parallel quad.O quad.Q quad.P quad.R) ∨
  (¬parallel quad.O quad.P quad.Q quad.R ∧ parallel quad.O quad.Q quad.P quad.R)

/-- Check if a quadrilateral is a parallelogram -/
def isParallelogram (quad : Quadrilateral) : Prop :=
  parallel quad.O quad.P quad.Q quad.R ∧ parallel quad.O quad.Q quad.P quad.R

/-- The main theorem -/
theorem OPRQ_shape (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂ ∨ y₁ ≠ y₂) :
  let P : Point := ⟨x₁, y₁⟩
  let Q : Point := ⟨x₂, y₂⟩
  let R : Point := ⟨x₁ - x₂, y₁ - y₂⟩
  let O : Point := ⟨0, 0⟩
  let quad : Quadrilateral := ⟨O, P, R, Q⟩
  (isStraightLine quad ∨ isTrapezoid quad) ∧ ¬isParallelogram quad := by
  sorry


end NUMINAMATH_CALUDE_OPRQ_shape_l2315_231530


namespace NUMINAMATH_CALUDE_student_meeting_probability_l2315_231566

/-- The probability of two students meeting given specific conditions -/
theorem student_meeting_probability (α : ℝ) (h : 0 < α ∧ α < 60) :
  let p := 1 - ((60 - α) / 60)^2
  0 ≤ p ∧ p ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_student_meeting_probability_l2315_231566


namespace NUMINAMATH_CALUDE_andrew_age_proof_l2315_231542

/-- Andrew's age in years -/
def andrew_age : ℚ := 30 / 7

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℚ := 15 * andrew_age

theorem andrew_age_proof :
  (grandfather_age - andrew_age = 60) ∧ (grandfather_age = 15 * andrew_age) → andrew_age = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_andrew_age_proof_l2315_231542


namespace NUMINAMATH_CALUDE_marble_probability_l2315_231523

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) : 
  total = 84 →
  p_white = 1/4 →
  p_green = 1/7 →
  (total : ℚ) * (1 - p_white - p_green) / total = 17/28 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l2315_231523


namespace NUMINAMATH_CALUDE_total_apples_is_75_l2315_231548

/-- The number of apples Benny picked from each tree -/
def benny_apples_per_tree : ℕ := 2

/-- The number of trees Benny picked from -/
def benny_trees : ℕ := 4

/-- The number of apples Dan picked from each tree -/
def dan_apples_per_tree : ℕ := 9

/-- The number of trees Dan picked from -/
def dan_trees : ℕ := 5

/-- Calculate the total number of apples picked by Benny -/
def benny_total : ℕ := benny_apples_per_tree * benny_trees

/-- Calculate the total number of apples picked by Dan -/
def dan_total : ℕ := dan_apples_per_tree * dan_trees

/-- Calculate the number of apples picked by Sarah (half of Dan's total, rounded down) -/
def sarah_total : ℕ := dan_total / 2

/-- The total number of apples picked by all three people -/
def total_apples : ℕ := benny_total + dan_total + sarah_total

theorem total_apples_is_75 : total_apples = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_75_l2315_231548


namespace NUMINAMATH_CALUDE_problem_grid_paths_l2315_231573

/-- Represents a grid with forbidden segments -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (forbidden_segments : List (ℕ × ℕ × ℕ × ℕ))

/-- Calculates the number of paths in a grid with forbidden segments -/
def count_paths (g : Grid) : ℕ :=
  sorry

/-- The specific grid described in the problem -/
def problem_grid : Grid :=
  { rows := 4
  , cols := 7
  , forbidden_segments := [(1, 2, 3, 4), (2, 3, 5, 6)] }

/-- Theorem stating that the number of paths in the problem grid is 64 -/
theorem problem_grid_paths :
  count_paths problem_grid = 64 :=
sorry

end NUMINAMATH_CALUDE_problem_grid_paths_l2315_231573


namespace NUMINAMATH_CALUDE_average_difference_l2315_231589

theorem average_difference (x : ℝ) : (10 + 60 + x) / 3 = (20 + 40 + 60) / 3 - 5 ↔ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2315_231589


namespace NUMINAMATH_CALUDE_f_of_two_equals_zero_l2315_231564

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2 - 1

-- State the theorem
theorem f_of_two_equals_zero : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_zero_l2315_231564


namespace NUMINAMATH_CALUDE_factor_quadratic_l2315_231532

theorem factor_quadratic (x : ℝ) : 3 * x^2 + 12 * x + 12 = 3 * (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_quadratic_l2315_231532


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2315_231507

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := m * (x^2 - 2*x) + 2*x + 3

-- Define the condition for m1 and m2
def condition (m : ℝ) : Prop :=
  ∃ (a b : ℝ), quadratic_equation m a = 0 ∧ quadratic_equation m b = 0 ∧ a/b + b/a = 3/2

-- Theorem statement
theorem root_sum_theorem (m1 m2 : ℝ) :
  condition m1 ∧ condition m2 → m1/m2 + m2/m1 = 833/64 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2315_231507


namespace NUMINAMATH_CALUDE_digit_sum_divisible_by_11_l2315_231510

/-- The sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: In any sequence of 39 consecutive natural numbers, 
    there exists at least one number whose digit sum is divisible by 11 -/
theorem digit_sum_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (digitSum (N + k) % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_digit_sum_divisible_by_11_l2315_231510


namespace NUMINAMATH_CALUDE_interest_calculation_l2315_231535

/-- Calculates the compound interest earned over a period of time -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the compound interest earned on $2000 at 5% for 5 years is approximately $552.56 -/
theorem interest_calculation :
  let principal := 2000
  let rate := 0.05
  let years := 5
  abs (compoundInterest principal rate years - 552.56) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l2315_231535


namespace NUMINAMATH_CALUDE_pure_imaginary_m_l2315_231585

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_m (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 1) (m^2 + 2*m - 3)
  is_pure_imaginary z → m = -1 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_l2315_231585


namespace NUMINAMATH_CALUDE_no_seventh_power_sum_l2315_231557

def a : ℕ → ℤ
  | 0 => 8
  | 1 => 20
  | (n + 2) => (a (n + 1))^2 + 12 * (a (n + 1)) * (a n) + (a (n + 1)) + 11 * (a n)

def seventh_power_sum_mod_29 (x y z : ℤ) : ℤ :=
  ((x^7 % 29) + (y^7 % 29) + (z^7 % 29)) % 29

theorem no_seventh_power_sum (n : ℕ) :
  ∀ x y z : ℤ, (a n) % 29 ≠ seventh_power_sum_mod_29 x y z :=
by sorry

end NUMINAMATH_CALUDE_no_seventh_power_sum_l2315_231557


namespace NUMINAMATH_CALUDE_stock_price_change_l2315_231504

theorem stock_price_change (P1 P2 D : ℝ) (h1 : D = 0.18 * P1) (h2 : D = 0.12 * P2) :
  P2 = 1.5 * P1 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l2315_231504


namespace NUMINAMATH_CALUDE_sticks_left_in_yard_l2315_231571

def sticks_picked_up : ℕ := 14
def difference : ℕ := 10

theorem sticks_left_in_yard : sticks_picked_up - difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_sticks_left_in_yard_l2315_231571


namespace NUMINAMATH_CALUDE_certain_number_threshold_l2315_231562

theorem certain_number_threshold (k : ℤ) : 0.0010101 * (10 : ℝ)^(k : ℝ) > 10.101 → k ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_threshold_l2315_231562


namespace NUMINAMATH_CALUDE_smaller_root_of_quadratic_l2315_231572

theorem smaller_root_of_quadratic (x : ℚ) : 
  (x - 4/5) * (x - 4/5) + (x - 4/5) * (x - 2/3) + 1/15 = 0 →
  (x = 11/15 ∨ x = 4/5) ∧ 11/15 < 4/5 :=
sorry

end NUMINAMATH_CALUDE_smaller_root_of_quadratic_l2315_231572


namespace NUMINAMATH_CALUDE_crosswalk_distance_l2315_231503

/-- Given a parallelogram with one side of length 20 feet, height of 60 feet,
    and another side of length 80 feet, the distance between the 20-foot side
    and the 80-foot side is 15 feet. -/
theorem crosswalk_distance (side1 side2 height : ℝ) : 
  side1 = 20 → side2 = 80 → height = 60 → 
  (side1 * height) / side2 = 15 := by sorry

end NUMINAMATH_CALUDE_crosswalk_distance_l2315_231503


namespace NUMINAMATH_CALUDE_prob_same_color_l2315_231576

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of black balls in the bag -/
def black_balls : ℕ := 4

/-- The number of red balls in the bag -/
def red_balls : ℕ := 5

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls + red_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- The probability of drawing at least two balls of the same color -/
theorem prob_same_color : 
  (1 : ℚ) - (white_balls * black_balls * red_balls : ℚ) / (total_balls * (total_balls - 1) * (total_balls - 2) / 6) = 8/11 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_l2315_231576


namespace NUMINAMATH_CALUDE_dice_probability_l2315_231593

theorem dice_probability : 
  let n : ℕ := 8  -- total number of dice
  let k : ℕ := 4  -- number of dice showing even
  let p : ℚ := 1/2  -- probability of rolling even (or odd) on a single die
  Nat.choose n k * p^n = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l2315_231593


namespace NUMINAMATH_CALUDE_sqrt_seven_fraction_inequality_l2315_231587

theorem sqrt_seven_fraction_inequality (m n : ℤ) 
  (h1 : m ≥ 1) (h2 : n ≥ 1) (h3 : Real.sqrt 7 - (m : ℝ) / n > 0) : 
  Real.sqrt 7 - (m : ℝ) / n > 1 / (m * n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_fraction_inequality_l2315_231587


namespace NUMINAMATH_CALUDE_sine_product_inequality_l2315_231590

theorem sine_product_inequality :
  1/8 < Real.sin (20 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) ∧
  Real.sin (20 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sine_product_inequality_l2315_231590


namespace NUMINAMATH_CALUDE_square_side_length_l2315_231598

theorem square_side_length (area : ℝ) (h : area = 9/16) :
  ∃ (side : ℝ), side > 0 ∧ side^2 = area ∧ side = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2315_231598


namespace NUMINAMATH_CALUDE_complement_U_P_l2315_231559

-- Define the set U
def U : Set ℝ := {x | x^2 - 2*x < 3}

-- Define the set P
def P : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Theorem statement
theorem complement_U_P : 
  (U \ P) = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_U_P_l2315_231559


namespace NUMINAMATH_CALUDE_keyboard_warrior_disapproval_l2315_231506

theorem keyboard_warrior_disapproval 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (sample_approving : ℕ) 
  (h1 : total_population = 9600) 
  (h2 : sample_size = 50) 
  (h3 : sample_approving = 14) :
  ⌊(total_population : ℚ) * ((sample_size - sample_approving) : ℚ) / (sample_size : ℚ)⌋ = 6912 := by
  sorry

#check keyboard_warrior_disapproval

end NUMINAMATH_CALUDE_keyboard_warrior_disapproval_l2315_231506


namespace NUMINAMATH_CALUDE_hanks_fruit_purchase_cost_hanks_fruit_purchase_cost_is_1704_l2315_231509

/-- Calculates the total cost of Hank's fruit purchase at Clark's Food Store --/
theorem hanks_fruit_purchase_cost : ℝ :=
  let apple_price_per_dozen : ℝ := 40
  let pear_price_per_dozen : ℝ := 50
  let orange_price_per_dozen : ℝ := 30
  let apple_dozens_bought : ℝ := 14
  let pear_dozens_bought : ℝ := 18
  let orange_dozens_bought : ℝ := 10
  let apple_discount_rate : ℝ := 0.1

  let apple_cost : ℝ := apple_price_per_dozen * apple_dozens_bought
  let discounted_apple_cost : ℝ := apple_cost * (1 - apple_discount_rate)
  let pear_cost : ℝ := pear_price_per_dozen * pear_dozens_bought
  let orange_cost : ℝ := orange_price_per_dozen * orange_dozens_bought

  let total_cost : ℝ := discounted_apple_cost + pear_cost + orange_cost

  1704

/-- Proves that Hank's total fruit purchase cost is 1704 dollars --/
theorem hanks_fruit_purchase_cost_is_1704 : hanks_fruit_purchase_cost = 1704 := by
  sorry

end NUMINAMATH_CALUDE_hanks_fruit_purchase_cost_hanks_fruit_purchase_cost_is_1704_l2315_231509


namespace NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l2315_231581

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- The slope of the tangent line at a point on the parabola -/
def m (a : ℝ) : ℝ := 2*(a - 1)

/-- The y-coordinate of the intersection point of tangent lines -/
def y_intersection (a b : ℝ) : ℝ := a*b - a - b + 2

theorem tangent_intersection_y_coordinate 
  (a b : ℝ) 
  (ha : f a = a^2 - 2*a - 3) 
  (hb : f b = b^2 - 2*b - 3) 
  (h_perp : m a * m b = -1) : 
  y_intersection a b = -1/4 := by
  sorry

#check tangent_intersection_y_coordinate

end NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l2315_231581


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2315_231502

/-- A configuration of squares and rectangles -/
structure SquareRectConfig where
  /-- Side length of the inner square -/
  s : ℝ
  /-- Shorter side of each rectangle -/
  y : ℝ
  /-- Longer side of each rectangle -/
  x : ℝ
  /-- The shorter side of each rectangle is half the side of the inner square -/
  short_side_half : y = s / 2
  /-- The area of the outer square is 9 times that of the inner square -/
  area_ratio : (s + 2 * y)^2 = 9 * s^2
  /-- The longer side of the rectangle forms the side of the outer square with the inner square -/
  outer_square_side : x + s / 2 = 3 * s

/-- The ratio of the longer side to the shorter side of each rectangle is 5 -/
theorem rectangle_ratio (config : SquareRectConfig) : config.x / config.y = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2315_231502


namespace NUMINAMATH_CALUDE_epipen_cost_l2315_231536

/-- Proves that the cost of each EpiPen is $500, given the specified conditions -/
theorem epipen_cost (epipen_per_year : ℕ) (insurance_coverage : ℚ) (annual_payment : ℚ) :
  epipen_per_year = 2 ∧ insurance_coverage = 3/4 ∧ annual_payment = 250 →
  ∃ (cost : ℚ), cost = 500 ∧ epipen_per_year * (1 - insurance_coverage) * cost = annual_payment :=
by sorry

end NUMINAMATH_CALUDE_epipen_cost_l2315_231536


namespace NUMINAMATH_CALUDE_circle_point_distance_relation_l2315_231583

/-- Given a circle with radius r and a point F constructed as described in the problem,
    prove the relationship between distances u and v from F to specific lines. -/
theorem circle_point_distance_relation (r u v : ℝ) : v^2 = u^3 / (2*r - u) := by
  sorry

end NUMINAMATH_CALUDE_circle_point_distance_relation_l2315_231583


namespace NUMINAMATH_CALUDE_english_chinese_difference_l2315_231567

/-- The number of hours Ryan spends learning English daily -/
def hours_english : ℕ := 6

/-- The number of hours Ryan spends learning Chinese daily -/
def hours_chinese : ℕ := 2

/-- The difference in hours between English and Chinese learning -/
def hour_difference : ℕ := hours_english - hours_chinese

theorem english_chinese_difference : hour_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_english_chinese_difference_l2315_231567


namespace NUMINAMATH_CALUDE_friends_money_distribution_l2315_231568

structure Friend :=
  (name : String)
  (initialMoney : ℚ)

def giveMoneyTo (giver receiver : Friend) (fraction : ℚ) : ℚ :=
  giver.initialMoney * fraction

theorem friends_money_distribution (loki moe nick ott pam : Friend) 
  (h1 : ott.initialMoney = 0)
  (h2 : pam.initialMoney = 0)
  (h3 : giveMoneyTo moe ott (1/6) = giveMoneyTo loki ott (1/5))
  (h4 : giveMoneyTo moe ott (1/6) = giveMoneyTo nick ott (1/4))
  (h5 : giveMoneyTo moe pam (1/6) = giveMoneyTo loki pam (1/5))
  (h6 : giveMoneyTo moe pam (1/6) = giveMoneyTo nick pam (1/4)) :
  let totalInitialMoney := loki.initialMoney + moe.initialMoney + nick.initialMoney
  let moneyReceivedByOttAndPam := 2 * (giveMoneyTo moe ott (1/6) + giveMoneyTo loki ott (1/5) + giveMoneyTo nick ott (1/4))
  moneyReceivedByOttAndPam / totalInitialMoney = 2/5 := by
    sorry

#check friends_money_distribution

end NUMINAMATH_CALUDE_friends_money_distribution_l2315_231568


namespace NUMINAMATH_CALUDE_mba_committee_size_l2315_231543

/-- Represents the number of second-year MBAs -/
def total_mbas : ℕ := 6

/-- Represents the number of committees -/
def num_committees : ℕ := 2

/-- Represents the probability that Jane and Albert are on the same committee -/
def same_committee_prob : ℚ := 2/5

/-- Represents the number of members in each committee -/
def committee_size : ℕ := total_mbas / num_committees

theorem mba_committee_size :
  (committee_size = 3) ∧
  (same_committee_prob = (committee_size - 1 : ℚ) / (total_mbas - 1 : ℚ)) :=
sorry

end NUMINAMATH_CALUDE_mba_committee_size_l2315_231543


namespace NUMINAMATH_CALUDE_compare_sqrt_l2315_231563

theorem compare_sqrt : 2 * Real.sqrt 3 < 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_compare_sqrt_l2315_231563


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2315_231575

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 6*x + 8 = 0 →
  x + 3 > 6 ∧ x + 6 > 3 ∧ 3 + 6 > x →
  x + 3 + 6 = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2315_231575


namespace NUMINAMATH_CALUDE_base8_addition_and_conversion_l2315_231591

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 16 --/
def base10_to_base16 (n : ℕ) : ℕ := sorry

/-- Adds two base 8 numbers and returns the result in base 8 --/
def add_base8 (a b : ℕ) : ℕ := 
  base10_to_base8 (base8_to_base10 a + base8_to_base10 b)

theorem base8_addition_and_conversion :
  let a : ℕ := 537 -- In base 8
  let b : ℕ := 246 -- In base 8
  let sum_base8 : ℕ := add_base8 a b
  let sum_base16 : ℕ := base10_to_base16 (base8_to_base10 sum_base8)
  sum_base8 = 1005 ∧ sum_base16 = 0x205 := by sorry

end NUMINAMATH_CALUDE_base8_addition_and_conversion_l2315_231591


namespace NUMINAMATH_CALUDE_solution_count_l2315_231511

-- Define the equation
def equation (x a : ℝ) : Prop :=
  Real.log (2 - x^2) / Real.log (x - a) = 2

-- Theorem statement
theorem solution_count (a : ℝ) :
  (∀ x, ¬ equation x a) ∨
  (∃! x, equation x a) ∨
  (∃ x y, x ≠ y ∧ equation x a ∧ equation y a) :=
by
  -- Case 1: No solution
  have h1 : a ≤ -2 ∨ a = 0 ∨ a ≥ Real.sqrt 2 → ∀ x, ¬ equation x a := by sorry
  -- Case 2: One solution
  have h2 : (-Real.sqrt 2 < a ∧ a < 0) ∨ (0 < a ∧ a < Real.sqrt 2) → ∃! x, equation x a := by sorry
  -- Case 3: Two solutions
  have h3 : -2 < a ∧ a < -Real.sqrt 2 → ∃ x y, x ≠ y ∧ equation x a ∧ equation y a := by sorry
  sorry -- Complete the proof using h1, h2, and h3


end NUMINAMATH_CALUDE_solution_count_l2315_231511


namespace NUMINAMATH_CALUDE_calculate_expression_l2315_231580

theorem calculate_expression : (Real.pi - Real.sqrt 3) ^ 0 - 2 * Real.sin (45 * π / 180) + |-Real.sqrt 2| + Real.sqrt 8 = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2315_231580


namespace NUMINAMATH_CALUDE_domino_arrangements_count_l2315_231555

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a domino with length and width -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the number of distinct arrangements of dominoes on a grid -/
def count_arrangements (g : Grid) (d : Domino) (num_dominoes : ℕ) : ℕ :=
  Nat.choose (g.width + g.height - 2) (g.width - 1)

/-- Theorem stating the number of distinct arrangements -/
theorem domino_arrangements_count (g : Grid) (d : Domino) (num_dominoes : ℕ) :
  g.width = 6 →
  g.height = 4 →
  d.length = 2 →
  d.width = 1 →
  num_dominoes = 5 →
  count_arrangements g d num_dominoes = 56 :=
by sorry

end NUMINAMATH_CALUDE_domino_arrangements_count_l2315_231555


namespace NUMINAMATH_CALUDE_sum_of_variables_l2315_231516

theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 15 - 4*x)
  (eq2 : x + z = -17 - 4*y)
  (eq3 : x + y = 8 - 4*z) :
  2*x + 2*y + 2*z = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l2315_231516


namespace NUMINAMATH_CALUDE_complement_to_set_l2315_231500

def U : Set ℤ := {-1, 0, 1, 2, 4}

theorem complement_to_set (M : Set ℤ) (h : {x : ℤ | x ∈ U ∧ x ∉ M} = {-1, 1}) : 
  M = {0, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_to_set_l2315_231500


namespace NUMINAMATH_CALUDE_bushes_for_zucchinis_l2315_231528

/-- The number of containers of blueberries each bush yields -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries that can be traded for zucchinis -/
def containers_for_trade : ℕ := 6

/-- The number of zucchinis received in trade for containers_for_trade -/
def zucchinis_from_trade : ℕ := 3

/-- The target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- The number of bushes needed to obtain the target number of zucchinis -/
def bushes_needed : ℕ := 12

theorem bushes_for_zucchinis :
  bushes_needed * containers_per_bush * zucchinis_from_trade = 
  target_zucchinis * containers_for_trade := by
  sorry

end NUMINAMATH_CALUDE_bushes_for_zucchinis_l2315_231528


namespace NUMINAMATH_CALUDE_miley_bought_two_cellphones_l2315_231549

/-- The number of cellphones Miley bought -/
def num_cellphones : ℕ := 2

/-- The cost of each cellphone in dollars -/
def cost_per_cellphone : ℝ := 800

/-- The discount rate for buying more than one cellphone -/
def discount_rate : ℝ := 0.05

/-- The total amount Miley paid in dollars -/
def total_paid : ℝ := 1520

/-- Theorem stating that the number of cellphones Miley bought is 2 -/
theorem miley_bought_two_cellphones :
  num_cellphones = 2 ∧
  num_cellphones > 1 ∧
  (1 - discount_rate) * (num_cellphones : ℝ) * cost_per_cellphone = total_paid :=
by sorry

end NUMINAMATH_CALUDE_miley_bought_two_cellphones_l2315_231549


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2315_231534

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a = 3 →
  b = 2 * Real.sqrt 6 →
  B = 2 * A →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b / Real.sin B = c / Real.sin C →
  (Real.cos A = Real.sqrt 6 / 3) ∧ (c = 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2315_231534


namespace NUMINAMATH_CALUDE_complex_root_modulus_l2315_231556

open Complex

theorem complex_root_modulus (c d : ℝ) (h : (1 + I)^2 + c*(1 + I) + d = 0) : 
  abs (c + d*I) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_modulus_l2315_231556


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l2315_231552

theorem consecutive_integers_sum_of_squares (a : ℕ) (h1 : a > 1) 
  (h2 : (a - 1) * a * (a + 1) = 10 * (3 * a)) : 
  (a - 1)^2 + a^2 + (a + 1)^2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l2315_231552


namespace NUMINAMATH_CALUDE_gunny_bag_fill_proof_l2315_231594

/-- Conversion factor from tons to pounds -/
def tons_to_pounds : ℝ := 2200

/-- Conversion factor from pounds to ounces -/
def pounds_to_ounces : ℝ := 16

/-- Conversion factor from grams to ounces -/
def grams_to_ounces : ℝ := 0.035274

/-- Capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℝ := 13.5

/-- Weight of a packet in pounds -/
def packet_weight_pounds : ℝ := 16

/-- Weight of a packet in additional ounces -/
def packet_weight_extra_ounces : ℝ := 4

/-- Weight of a packet in additional grams -/
def packet_weight_extra_grams : ℝ := 350

/-- The number of packets needed to fill the gunny bag -/
def packets_needed : ℕ := 1745

theorem gunny_bag_fill_proof : 
  ⌈(gunny_bag_capacity * tons_to_pounds * pounds_to_ounces) / 
   (packet_weight_pounds * pounds_to_ounces + packet_weight_extra_ounces + 
    packet_weight_extra_grams * grams_to_ounces)⌉ = packets_needed := by
  sorry

end NUMINAMATH_CALUDE_gunny_bag_fill_proof_l2315_231594


namespace NUMINAMATH_CALUDE_quadrilateral_problem_l2315_231588

/-- Prove that for a quadrilateral PQRS with specific vertex coordinates,
    consecutive integer side lengths, and an area of 50,
    the product of the odd integer scale factor and the sum of side lengths is 5. -/
theorem quadrilateral_problem (a b k : ℤ) : 
  a > b ∧ b > 0 ∧  -- a and b are consecutive integers with a > b > 0
  ∃ n : ℤ, a = b + 1 ∧  -- a and b are consecutive integers
  ∃ m : ℤ, k = 2 * m + 1 ∧  -- k is an odd integer
  2 * k^2 * (a - b) * (a + b) = 50 →  -- area of PQRS is 50
  k * (a + b) = 5 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_problem_l2315_231588


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2315_231515

def diamond (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 2

theorem diamond_equation_solution :
  ∀ X : ℝ, diamond X 6 = 35 → X = 51 / 4 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2315_231515


namespace NUMINAMATH_CALUDE_fraction_simplification_l2315_231560

theorem fraction_simplification (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^(2*b) * b^a) / (b^(2*a) * a^b) = (a/b)^b := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2315_231560


namespace NUMINAMATH_CALUDE_largest_positive_solution_l2315_231533

theorem largest_positive_solution : 
  ∃ (x : ℝ), x > 0 ∧ 
    (2 * x^3 - x^2 - x + 1)^(1 + 1/(2*x + 1)) = 1 ∧ 
    ∀ (y : ℝ), y > 0 → 
      (2 * y^3 - y^2 - y + 1)^(1 + 1/(2*y + 1)) = 1 → 
      y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_positive_solution_l2315_231533


namespace NUMINAMATH_CALUDE_g_of_2_l2315_231501

def g (x : ℝ) : ℝ := x^2 + 3*x - 1

theorem g_of_2 : g 2 = 9 := by sorry

end NUMINAMATH_CALUDE_g_of_2_l2315_231501


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l2315_231519

theorem incorrect_average_calculation (n : ℕ) (incorrect_num correct_num : ℝ) (correct_avg : ℝ) :
  n = 10 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 65 ∧ 
  correct_avg = 50 →
  ∃ (other_sum : ℝ),
    (other_sum + correct_num) / n = correct_avg ∧
    (other_sum + incorrect_num) / n = 46 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l2315_231519


namespace NUMINAMATH_CALUDE_cubic_one_real_root_l2315_231584

theorem cubic_one_real_root (c : ℝ) :
  ∃! x : ℝ, x^3 - 4*x^2 + 9*x + c = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_cubic_one_real_root_l2315_231584


namespace NUMINAMATH_CALUDE_evaluate_expression_l2315_231561

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 1) :
  z * (z - 4 * x) = -15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2315_231561


namespace NUMINAMATH_CALUDE_angle_ratio_not_determine_right_triangle_l2315_231595

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = 180

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Define the angle ratio condition
def angle_ratio_condition (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.A = 6 * k ∧ t.B = 8 * k ∧ t.C = 10 * k

-- Theorem statement
theorem angle_ratio_not_determine_right_triangle :
  ∃ (t : Triangle), angle_ratio_condition t ∧ ¬(is_right_triangle t) :=
sorry

end NUMINAMATH_CALUDE_angle_ratio_not_determine_right_triangle_l2315_231595


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2315_231596

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : Nat
  sampleSize : Nat
  interval : Nat
  firstElement : Nat

/-- Checks if a number is in the systematic sample -/
def isInSample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, n = s.firstElement + k * s.interval ∧ n ≤ s.total

theorem systematic_sample_fourth_element 
  (s : SystematicSample)
  (h_total : s.total = 52)
  (h_size : s.sampleSize = 4)
  (h_interval : s.interval = s.total / s.sampleSize)
  (h_first : s.firstElement = 6)
  (h_in_32 : isInSample s 32)
  (h_in_45 : isInSample s 45) :
  isInSample s 19 :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2315_231596


namespace NUMINAMATH_CALUDE_work_completion_time_l2315_231546

/-- Given a work that B can complete in 10 days, and when A and B work together,
    B's share of the total 5000 Rs wages is 3333 Rs, prove that A alone can do the work in 20 days. -/
theorem work_completion_time
  (b_time : ℝ)
  (total_wages : ℝ)
  (b_wages : ℝ)
  (h1 : b_time = 10)
  (h2 : total_wages = 5000)
  (h3 : b_wages = 3333)
  : ∃ (a_time : ℝ), a_time = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2315_231546


namespace NUMINAMATH_CALUDE_power_of_negative_two_a_cubed_l2315_231565

theorem power_of_negative_two_a_cubed (a : ℝ) : (-2 * a^3)^3 = -8 * a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_two_a_cubed_l2315_231565


namespace NUMINAMATH_CALUDE_area_four_intersecting_circles_l2315_231529

/-- The area common to four intersecting circles with specific configuration -/
theorem area_four_intersecting_circles (R : ℝ) (R_pos : R > 0) : ℝ := by
  /- Given two circles of radius R that intersect such that each passes through the center of the other,
     and two additional circles of radius R with centers at the intersection points of the first two circles,
     the area common to all four circles is: -/
  have area : ℝ := R^2 * (2 * Real.pi - 3 * Real.sqrt 3) / 6
  
  /- Proof goes here -/
  sorry

#check area_four_intersecting_circles

end NUMINAMATH_CALUDE_area_four_intersecting_circles_l2315_231529


namespace NUMINAMATH_CALUDE_inequality_proof_l2315_231508

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 3) : 
  1 / (Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1)) + 
  1 / (Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1)) + 
  1 / (Real.sqrt (3 * z + 1) + Real.sqrt (3 * x + 1)) ≥ 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2315_231508


namespace NUMINAMATH_CALUDE_park_area_l2315_231577

/-- Given a rectangular park with length to breadth ratio of 1:2, where a cyclist completes one round along the boundary in 6 minutes at an average speed of 6 km/hr, prove that the area of the park is 20,000 square meters. -/
theorem park_area (length width : ℝ) (average_speed : ℝ) (time_taken : ℝ) : 
  length > 0 ∧ 
  width > 0 ∧ 
  length = (1/2) * width ∧ 
  average_speed = 6 ∧ 
  time_taken = 1/10 ∧ 
  2 * (length + width) = average_speed * time_taken * 1000 →
  length * width = 20000 := by sorry

end NUMINAMATH_CALUDE_park_area_l2315_231577


namespace NUMINAMATH_CALUDE_total_people_count_l2315_231531

theorem total_people_count (people_in_front : ℕ) (people_behind : ℕ) (total_lines : ℕ) :
  people_in_front = 2 →
  people_behind = 4 →
  total_lines = 8 →
  (people_in_front + 1 + people_behind) * total_lines = 56 :=
by sorry

end NUMINAMATH_CALUDE_total_people_count_l2315_231531
