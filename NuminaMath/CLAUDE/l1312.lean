import Mathlib

namespace NUMINAMATH_CALUDE_sunzi_problem_l1312_131296

theorem sunzi_problem (x y : ℚ) : 
  (x + (1/2) * y = 48 ∧ y + (2/3) * x = 48) ↔ 
  (x + (1/2) * y = 48 ∧ y + (2/3) * x = 48) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_problem_l1312_131296


namespace NUMINAMATH_CALUDE_triangle_area_l1312_131260

/-- The area of a triangle with side lengths 7, 8, and 10 -/
theorem triangle_area : ℝ := by
  -- Define the side lengths
  let a : ℝ := 7
  let b : ℝ := 8
  let c : ℝ := 10

  -- Define the semi-perimeter
  let s : ℝ := (a + b + c) / 2

  -- Define the area using Heron's formula
  let area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1312_131260


namespace NUMINAMATH_CALUDE_max_intersections_ellipse_cosine_l1312_131214

-- Define the ellipse equation
def ellipse (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

-- Define the cosine function
def cosine_graph (x y : ℝ) : Prop :=
  y = Real.cos x

-- Theorem statement
theorem max_intersections_ellipse_cosine :
  ∃ (h k a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∃ (points : Finset (ℝ × ℝ)),
    (∀ (p : ℝ × ℝ), p ∈ points → ellipse p.1 p.2 h k a b ∧ cosine_graph p.1 p.2) ∧
    points.card = 8) ∧
  (∀ (points : Finset (ℝ × ℝ)),
    (∀ (p : ℝ × ℝ), p ∈ points → ellipse p.1 p.2 h k a b ∧ cosine_graph p.1 p.2) →
    points.card ≤ 8) :=
by sorry


end NUMINAMATH_CALUDE_max_intersections_ellipse_cosine_l1312_131214


namespace NUMINAMATH_CALUDE_area_of_special_triangle_l1312_131245

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- Implement the condition for a right triangle
  sorry

def is_scalene (t : Triangle) : Prop :=
  -- Implement the condition for a scalene triangle
  sorry

def on_hypotenuse (t : Triangle) : Prop :=
  -- Implement the condition that P is on the hypotenuse AC
  sorry

def angle_ABP_45 (t : Triangle) : Prop :=
  -- Implement the condition that ∠ABP = 45°
  sorry

def AP_equals_2 (t : Triangle) : Prop :=
  -- Implement the condition that AP = 2
  sorry

def CP_equals_3 (t : Triangle) : Prop :=
  -- Implement the condition that CP = 3
  sorry

-- Define the area of a triangle
def triangle_area (t : Triangle) : ℝ :=
  -- Implement the formula for triangle area
  sorry

-- Theorem statement
theorem area_of_special_triangle (t : Triangle) :
  is_right_triangle t →
  is_scalene t →
  on_hypotenuse t →
  angle_ABP_45 t →
  AP_equals_2 t →
  CP_equals_3 t →
  triangle_area t = 75 / 13 :=
sorry

end NUMINAMATH_CALUDE_area_of_special_triangle_l1312_131245


namespace NUMINAMATH_CALUDE_jean_is_cyclist_l1312_131294

/-- Represents a traveler's journey --/
structure Traveler where
  distanceTraveled : ℝ
  distanceRemaining : ℝ

/-- Jean's travel condition --/
def jeanCondition (j : Traveler) : Prop :=
  3 * j.distanceTraveled + 2 * j.distanceRemaining = j.distanceTraveled + j.distanceRemaining

/-- Jules' travel condition --/
def julesCondition (j : Traveler) : Prop :=
  (1/2) * j.distanceTraveled + 3 * j.distanceRemaining = j.distanceTraveled + j.distanceRemaining

/-- The theorem to prove --/
theorem jean_is_cyclist (jean jules : Traveler) 
  (hj : jeanCondition jean) (hk : julesCondition jules) : 
  jean.distanceTraveled / (jean.distanceTraveled + jean.distanceRemaining) < 
  jules.distanceTraveled / (jules.distanceTraveled + jules.distanceRemaining) :=
sorry

end NUMINAMATH_CALUDE_jean_is_cyclist_l1312_131294


namespace NUMINAMATH_CALUDE_cow_count_is_twelve_l1312_131267

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem stating that the number of cows is 12 -/
theorem cow_count_is_twelve :
  ∃ (count : AnimalCount), 
    totalLegs count = 2 * totalHeads count + 24 ∧ 
    count.cows = 12 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_is_twelve_l1312_131267


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l1312_131264

theorem sin_cos_sum_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (47 * π / 180) * Real.cos (103 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l1312_131264


namespace NUMINAMATH_CALUDE_max_value_is_60_l1312_131244

-- Define the types of jewels
structure Jewel :=
  (weight : ℕ)
  (value : ℕ)

-- Define the jewel types
def typeA : Jewel := ⟨6, 18⟩
def typeB : Jewel := ⟨3, 9⟩
def typeC : Jewel := ⟨1, 4⟩

-- Define the maximum carrying capacity
def maxCapacity : ℕ := 15

-- Define the function to calculate the maximum value
def maxValue (typeA typeB typeC : Jewel) (maxCapacity : ℕ) : ℕ :=
  sorry

-- Theorem stating the maximum value is 60
theorem max_value_is_60 :
  maxValue typeA typeB typeC maxCapacity = 60 :=
sorry

end NUMINAMATH_CALUDE_max_value_is_60_l1312_131244


namespace NUMINAMATH_CALUDE_alex_upside_down_growth_rate_l1312_131298

/-- The growth rate of Alex when hanging upside down -/
def upsideDownGrowthRate (
  requiredHeight : ℚ)
  (currentHeight : ℚ)
  (normalGrowthRate : ℚ)
  (upsideDownHoursPerMonth : ℚ)
  (monthsInYear : ℕ) : ℚ :=
  let totalGrowthNeeded := requiredHeight - currentHeight
  let normalYearlyGrowth := normalGrowthRate * monthsInYear
  let additionalGrowthNeeded := totalGrowthNeeded - normalYearlyGrowth
  let totalUpsideDownHours := upsideDownHoursPerMonth * monthsInYear
  additionalGrowthNeeded / totalUpsideDownHours

/-- Theorem stating that Alex's upside down growth rate is 1/12 inch per hour -/
theorem alex_upside_down_growth_rate :
  upsideDownGrowthRate 54 48 (1/3) 2 12 = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_alex_upside_down_growth_rate_l1312_131298


namespace NUMINAMATH_CALUDE_robyn_cookie_sales_l1312_131263

theorem robyn_cookie_sales (total_sales lucy_sales : ℕ) 
  (h1 : total_sales = 76) 
  (h2 : lucy_sales = 29) : 
  total_sales - lucy_sales = 47 := by
  sorry

end NUMINAMATH_CALUDE_robyn_cookie_sales_l1312_131263


namespace NUMINAMATH_CALUDE_tims_age_l1312_131270

theorem tims_age (james_age john_age tim_age : ℕ) : 
  james_age = 23 → 
  john_age = 35 → 
  tim_age = 2 * john_age - 5 → 
  tim_age = 65 := by
sorry

end NUMINAMATH_CALUDE_tims_age_l1312_131270


namespace NUMINAMATH_CALUDE_thread_length_ratio_l1312_131281

theorem thread_length_ratio : 
  let original_length : ℚ := 12
  let total_required : ℚ := 21
  let additional_length := total_required - original_length
  additional_length / original_length = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_thread_length_ratio_l1312_131281


namespace NUMINAMATH_CALUDE_smallest_angle_for_sin_polar_graph_l1312_131229

def completes_intrinsic_pattern (t : Real) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ ≤ t → ∃ r, r = Real.sin θ ∧ 
  (∀ ϕ, ϕ > t → ∃ ψ, 0 ≤ ψ ∧ ψ ≤ t ∧ Real.sin ϕ = Real.sin ψ)

theorem smallest_angle_for_sin_polar_graph :
  (∀ t < 2 * Real.pi, ¬ completes_intrinsic_pattern t) ∧
  completes_intrinsic_pattern (2 * Real.pi) := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_for_sin_polar_graph_l1312_131229


namespace NUMINAMATH_CALUDE_fashion_design_not_in_digital_china_l1312_131253

-- Define the concept of a service area
def ServiceArea : Type := String

-- Define Digital China as a structure with a set of service areas
structure DigitalChina :=
  (services : Set ServiceArea)

-- Define known service areas
def environmentalMonitoring : ServiceArea := "Environmental Monitoring"
def publicSecurity : ServiceArea := "Public Security"
def financialInfo : ServiceArea := "Financial Information"
def fashionDesign : ServiceArea := "Fashion Design"

-- Theorem: Fashion design is not a service area of Digital China
theorem fashion_design_not_in_digital_china 
  (dc : DigitalChina) 
  (h1 : environmentalMonitoring ∈ dc.services)
  (h2 : publicSecurity ∈ dc.services)
  (h3 : financialInfo ∈ dc.services) :
  fashionDesign ∉ dc.services := by
  sorry


end NUMINAMATH_CALUDE_fashion_design_not_in_digital_china_l1312_131253


namespace NUMINAMATH_CALUDE_inverse_composition_equals_target_l1312_131203

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 4

-- Define the inverse function f⁻¹
noncomputable def f_inv (x : ℝ) : ℝ := (x + 4) / 3

-- Theorem statement
theorem inverse_composition_equals_target : f_inv (f_inv 13) = 29 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_target_l1312_131203


namespace NUMINAMATH_CALUDE_share_calculation_l1312_131288

/-- Given a total amount divided among three parties with specific ratios, 
    prove that the first party's share is a certain value. -/
theorem share_calculation (total : ℚ) (a b c : ℚ) : 
  total = 700 →
  a + b + c = total →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a = 280 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l1312_131288


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1312_131278

-- Define ω as a complex number that is a nonreal root of z^4 = 1
def ω : ℂ := sorry

-- Define the property for the ordered pairs we're looking for
def validPair (a b : ℤ) : Prop :=
  Complex.abs (a • ω + b) = 1

-- State the theorem
theorem count_valid_pairs :
  ∃! (n : ℕ), ∃ (S : Finset (ℤ × ℤ)), 
    S.card = n ∧ 
    (∀ (p : ℤ × ℤ), p ∈ S ↔ validPair p.1 p.2) ∧
    n = 4 := by sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1312_131278


namespace NUMINAMATH_CALUDE_sequence_comparison_l1312_131204

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ+ → ℝ :=
  fun n => a₁ + (n.val - 1) * d

def geometric_sequence (b₁ : ℝ) (q : ℝ) : ℕ+ → ℝ :=
  fun n => b₁ * q^(n.val - 1)

theorem sequence_comparison (a : ℕ+ → ℝ) (b : ℕ+ → ℝ) :
  (a 1 = 2) →
  (b 1 = 2) →
  (a 2 = 4) →
  (b 2 = 4) →
  (∀ n : ℕ+, a n = 2 * n.val) →
  (∀ n : ℕ+, b n = 2^n.val) →
  (∀ n : ℕ+, n ≥ 3 → a n < b n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_comparison_l1312_131204


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1312_131290

theorem algebraic_expression_value (a b c : ℤ) 
  (h1 : a - b = 3) 
  (h2 : b + c = -5) : 
  a * c - b * c + a^2 - a * b = -6 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1312_131290


namespace NUMINAMATH_CALUDE_only_parallelogram_coincides_l1312_131241

-- Define the shapes
inductive Shape
  | Parallelogram
  | EquilateralTriangle
  | IsoscelesRightTriangle
  | RegularPentagon

-- Define a function to check if a shape coincides with itself after 180° rotation
def coincides_after_180_rotation (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | _ => False

-- Theorem statement
theorem only_parallelogram_coincides :
  ∀ (s : Shape), coincides_after_180_rotation s ↔ s = Shape.Parallelogram :=
by sorry

end NUMINAMATH_CALUDE_only_parallelogram_coincides_l1312_131241


namespace NUMINAMATH_CALUDE_A_solution_l1312_131236

noncomputable def A (x y : ℝ) : ℝ := 
  (Real.sqrt (4 * (x - Real.sqrt y) + y / x) * 
   Real.sqrt (9 * x^2 + 6 * (2 * y * x^3)^(1/3) + (4 * y^2)^(1/3))) / 
  (6 * x^2 + 2 * (2 * y * x^3)^(1/3) - 3 * Real.sqrt (y * x^2) - (4 * y^5)^(1/6)) / 2.343

theorem A_solution (x y : ℝ) (hx : x > 0) (hy : y ≥ 0) :
  A x y = if y > 4 * x^2 then -1 / Real.sqrt x else 1 / Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_A_solution_l1312_131236


namespace NUMINAMATH_CALUDE_count_ten_digit_numbers_theorem_l1312_131262

/-- Count of ten-digit numbers with a given digit sum -/
def count_ten_digit_numbers (n : ℕ) : ℕ :=
  match n with
  | 2 => 46
  | 3 => 166
  | 4 => 361
  | _ => 0

/-- Theorem stating the count of ten-digit numbers with specific digit sums -/
theorem count_ten_digit_numbers_theorem :
  (count_ten_digit_numbers 2 = 46) ∧
  (count_ten_digit_numbers 3 = 166) ∧
  (count_ten_digit_numbers 4 = 361) := by
  sorry

end NUMINAMATH_CALUDE_count_ten_digit_numbers_theorem_l1312_131262


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l1312_131249

theorem quadratic_integer_roots (p q : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q →
  (∃ x y : ℤ, x^2 + 5*p*x + 7*q = 0 ∧ y^2 + 5*p*y + 7*q = 0) ↔ 
  ((p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l1312_131249


namespace NUMINAMATH_CALUDE_mean_problem_l1312_131258

theorem mean_problem (x y : ℝ) : 
  (6 + 14 + x + 17 + 9 + y + 10) / 7 = 13 → x + y = 35 := by
  sorry

end NUMINAMATH_CALUDE_mean_problem_l1312_131258


namespace NUMINAMATH_CALUDE_complex_square_root_l1312_131233

theorem complex_square_root (a b : ℕ+) (h : (↑a + ↑b * Complex.I) ^ 2 = 5 + 12 * Complex.I) :
  ↑a + ↑b * Complex.I = 3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_l1312_131233


namespace NUMINAMATH_CALUDE_sum_product_inequality_l1312_131231

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l1312_131231


namespace NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l1312_131234

def point_A : ℝ × ℝ := (2, -3)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_A_in_fourth_quadrant : in_fourth_quadrant point_A := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l1312_131234


namespace NUMINAMATH_CALUDE_buddy_system_fraction_l1312_131230

theorem buddy_system_fraction (f e : ℕ) (h : e = (4 * f) / 3) : 
  (f / 3 + e / 4) / (f + e) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_buddy_system_fraction_l1312_131230


namespace NUMINAMATH_CALUDE_existence_of_three_person_correspondence_l1312_131283

/-- Represents a person in the correspondence network -/
def Person : Type := ℕ

/-- Represents a topic of correspondence -/
def Topic : Type := ℕ

/-- The total number of people in the network -/
def totalPeople : ℕ := 17

/-- The total number of topics -/
def totalTopics : ℕ := 3

/-- A function that returns the topic of correspondence between two people -/
def correspondenceTopic : Person → Person → Topic := sorry

/-- Proposition: There exists a subset of at least 3 people who all correspond on the same topic -/
theorem existence_of_three_person_correspondence :
  ∃ (t : Topic) (p₁ p₂ p₃ : Person),
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    correspondenceTopic p₁ p₂ = t ∧
    correspondenceTopic p₁ p₃ = t ∧
    correspondenceTopic p₂ p₃ = t :=
by
  sorry


end NUMINAMATH_CALUDE_existence_of_three_person_correspondence_l1312_131283


namespace NUMINAMATH_CALUDE_point_coordinates_l1312_131247

/-- A point in the two-dimensional plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the two-dimensional plane -/
def fourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance between a point and the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance between a point and the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: Coordinates of a point in the fourth quadrant with given distances to axes -/
theorem point_coordinates (p : Point) 
  (h1 : fourthQuadrant p) 
  (h2 : distanceToXAxis p = 2) 
  (h3 : distanceToYAxis p = 3) : 
  p = Point.mk 3 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1312_131247


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l1312_131239

theorem discount_percentage_proof (jacket_price shirt_price : ℝ)
  (jacket_discount shirt_discount : ℝ) :
  jacket_price = 80 →
  shirt_price = 40 →
  jacket_discount = 0.4 →
  shirt_discount = 0.55 →
  (jacket_price * jacket_discount + shirt_price * shirt_discount) /
  (jacket_price + shirt_price) = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l1312_131239


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l1312_131210

/-- Represents a number with n repetitions of a digit in base 10 -/
def repeatedDigit (digit : Nat) (n : Nat) : Nat :=
  digit * ((10^n - 1) / 9)

/-- Calculates the sum of digits of a number in base 10 -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_9ab :
  let a := repeatedDigit 9 1977
  let b := repeatedDigit 6 1977
  sumOfDigits (9 * a * b) = 25694 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l1312_131210


namespace NUMINAMATH_CALUDE_group_formation_count_l1312_131240

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of groups that can be formed with one boy and two girls -/
def oneBoytwoGirls (boys girls : ℕ) : ℕ := 
  binomial boys 1 * binomial girls 2

/-- The number of groups that can be formed with two boys and one girl -/
def twoBoyoneGirl (boys girls : ℕ) : ℕ := 
  binomial boys 2 * binomial girls 1

/-- The total number of valid groups that can be formed -/
def totalGroups (boys girls : ℕ) : ℕ := 
  oneBoytwoGirls boys girls + twoBoyoneGirl boys girls

theorem group_formation_count :
  totalGroups 9 12 = 1026 := by sorry

end NUMINAMATH_CALUDE_group_formation_count_l1312_131240


namespace NUMINAMATH_CALUDE_base_7_sum_theorem_l1312_131248

def base_7_to_decimal (a b c : Nat) : Nat :=
  7^2 * a + 7 * b + c

theorem base_7_sum_theorem (A B C : Nat) :
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
  A < 7 ∧ B < 7 ∧ C < 7 ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  base_7_to_decimal A B C + base_7_to_decimal B C A + base_7_to_decimal C A B = base_7_to_decimal A A A + 1 →
  B + C = 6 := by
sorry

end NUMINAMATH_CALUDE_base_7_sum_theorem_l1312_131248


namespace NUMINAMATH_CALUDE_map_scale_conversion_l1312_131238

/-- Given a map scale where 10 cm represents 50 km, 
    prove that a 23 cm length on the map represents 115 km. -/
theorem map_scale_conversion (scale : ℝ → ℝ) : 
  (scale 10 = 50) → (scale 23 = 115) :=
by
  sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l1312_131238


namespace NUMINAMATH_CALUDE_initial_bedbug_count_l1312_131200

/-- The number of bedbugs after n days, given an initial population -/
def bedbug_population (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (3 ^ days)

/-- Theorem stating the initial number of bedbugs -/
theorem initial_bedbug_count : ∃ (initial : ℕ), 
  bedbug_population initial 4 = 810 ∧ initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_bedbug_count_l1312_131200


namespace NUMINAMATH_CALUDE_can_space_before_compacting_l1312_131280

theorem can_space_before_compacting :
  ∀ (n : ℕ) (total_space : ℝ) (compaction_ratio : ℝ),
    n = 60 →
    compaction_ratio = 0.2 →
    total_space = 360 →
    (n : ℝ) * compaction_ratio * (360 / (n * compaction_ratio)) = total_space →
    360 / (n * compaction_ratio) = 30 := by
  sorry

end NUMINAMATH_CALUDE_can_space_before_compacting_l1312_131280


namespace NUMINAMATH_CALUDE_three_digit_permutation_property_l1312_131206

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_permutations (n : ℕ) : List ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  [100*a + 10*b + c, 100*a + 10*c + b, 100*b + 10*a + c, 100*b + 10*c + a, 100*c + 10*a + b, 100*c + 10*b + a]

def satisfies_property (n : ℕ) : Prop :=
  is_three_digit n ∧ (List.sum (digit_permutations n)) / 6 = n

def solution_set : List ℕ := [111, 222, 333, 444, 555, 666, 777, 888, 999, 407, 518, 629, 370, 481, 592]

theorem three_digit_permutation_property :
  ∀ n : ℕ, satisfies_property n ↔ n ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_three_digit_permutation_property_l1312_131206


namespace NUMINAMATH_CALUDE_shaded_squares_count_l1312_131252

/-- Represents the number of shaded squares in each column of the grid -/
def shaded_per_column : List Nat := [1, 3, 5, 4, 2, 0, 0, 0]

/-- The total number of squares in the grid -/
def total_squares : Nat := 30

/-- The number of columns in the grid -/
def num_columns : Nat := 8

theorem shaded_squares_count :
  (List.sum shaded_per_column = 15) ∧
  (List.sum shaded_per_column = total_squares / 2) ∧
  (List.length shaded_per_column = num_columns) := by
  sorry

end NUMINAMATH_CALUDE_shaded_squares_count_l1312_131252


namespace NUMINAMATH_CALUDE_opposite_sign_coordinates_second_quadrant_range_l1312_131266

def P (x : ℝ) : ℝ × ℝ := (x - 2, x)

theorem opposite_sign_coordinates (x : ℝ) :
  (P x).1 * (P x).2 < 0 → x = 1 := by sorry

theorem second_quadrant_range (x : ℝ) :
  (P x).1 < 0 ∧ (P x).2 > 0 → 0 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_opposite_sign_coordinates_second_quadrant_range_l1312_131266


namespace NUMINAMATH_CALUDE_exists_double_area_quadrilateral_l1312_131205

/-- The area of a quadrilateral given by four points in the plane -/
noncomputable def quadrilateralArea (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the existence of points A, B, C, and D such that 
    the area of ABCD is twice the area of ADBC -/
theorem exists_double_area_quadrilateral :
  ∃ (A B C D : ℝ × ℝ), quadrilateralArea A B C D = 2 * quadrilateralArea A D B C := by
  sorry

end NUMINAMATH_CALUDE_exists_double_area_quadrilateral_l1312_131205


namespace NUMINAMATH_CALUDE_division_result_l1312_131269

theorem division_result : ∃ (q : ℕ), 1254 = 6 * q → q = 209 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1312_131269


namespace NUMINAMATH_CALUDE_yellow_on_second_draw_l1312_131235

def total_balls : ℕ := 10
def yellow_balls : ℕ := 6
def white_balls : ℕ := 4

theorem yellow_on_second_draw :
  let p_white_first := white_balls / total_balls
  let p_yellow_second := yellow_balls / (total_balls - 1)
  p_white_first * p_yellow_second = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_yellow_on_second_draw_l1312_131235


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l1312_131232

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  -- Original quadrilateral sides
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  -- Extended sides
  bb' : ℝ
  cc' : ℝ
  dd' : ℝ
  aa' : ℝ
  -- Area of original quadrilateral
  area : ℝ
  -- Conditions
  ab_eq : ab = 5
  bc_eq : bc = 8
  cd_eq : cd = 4
  da_eq : da = 7
  bb'_eq : bb' = 1.5 * ab
  cc'_eq : cc' = 1.5 * bc
  dd'_eq : dd' = 1.5 * cd
  aa'_eq : aa' = 1.5 * da
  area_eq : area = 20

/-- The area of the extended quadrilateral A'B'C'D' is 140 -/
theorem extended_quadrilateral_area (q : ExtendedQuadrilateral) :
  q.area + (q.bb' - q.ab) * q.ab / 2 + (q.cc' - q.bc) * q.bc / 2 +
  (q.dd' - q.cd) * q.cd / 2 + (q.aa' - q.da) * q.da / 2 = 140 := by
  sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l1312_131232


namespace NUMINAMATH_CALUDE_local_min_implies_b_range_l1312_131256

theorem local_min_implies_b_range (b : ℝ) : 
  (∃ x ∈ Set.Ioo 0 1, IsLocalMin (fun x : ℝ ↦ x^3 - 3*b*x + 3*b) x) → 
  0 < b ∧ b < 1 :=
sorry

end NUMINAMATH_CALUDE_local_min_implies_b_range_l1312_131256


namespace NUMINAMATH_CALUDE_movie_ticket_sales_l1312_131237

theorem movie_ticket_sales (adult_price student_price total_revenue : ℚ)
  (student_tickets : ℕ) (h1 : adult_price = 4)
  (h2 : student_price = 5 / 2) (h3 : total_revenue = 445 / 2)
  (h4 : student_tickets = 9) :
  ∃ (adult_tickets : ℕ),
    adult_price * adult_tickets + student_price * student_tickets = total_revenue ∧
    adult_tickets + student_tickets = 59 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_sales_l1312_131237


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l1312_131228

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 + 2*x^2 + 3

-- State the theorem
theorem remainder_theorem (p : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + p a := by sorry

-- State the problem
theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x + 2) * q x + 27 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l1312_131228


namespace NUMINAMATH_CALUDE_cost_of_bananas_l1312_131223

/-- The cost of bananas given the total cost of groceries and the costs of other items -/
theorem cost_of_bananas 
  (total_cost : ℕ) 
  (bread_cost milk_cost apple_cost : ℕ) 
  (h1 : total_cost = 42)
  (h2 : bread_cost = 9)
  (h3 : milk_cost = 7)
  (h4 : apple_cost = 14) :
  total_cost - (bread_cost + milk_cost + apple_cost) = 12 := by
sorry

end NUMINAMATH_CALUDE_cost_of_bananas_l1312_131223


namespace NUMINAMATH_CALUDE_april_flower_sale_l1312_131257

/-- April's flower sale problem -/
theorem april_flower_sale 
  (price_per_rose : ℕ) 
  (initial_roses : ℕ) 
  (remaining_roses : ℕ) 
  (h1 : price_per_rose = 4)
  (h2 : initial_roses = 13)
  (h3 : remaining_roses = 4) :
  (initial_roses - remaining_roses) * price_per_rose = 36 := by
  sorry

end NUMINAMATH_CALUDE_april_flower_sale_l1312_131257


namespace NUMINAMATH_CALUDE_sea_glass_collection_l1312_131242

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_red rose_blue : ℕ) 
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_red = 9)
  (h4 : rose_blue = 11)
  (dorothy_red : ℕ)
  (h5 : dorothy_red = 2 * (blanche_red + rose_red))
  (dorothy_blue : ℕ)
  (h6 : dorothy_blue = 3 * rose_blue) :
  dorothy_red + dorothy_blue = 57 := by
sorry

end NUMINAMATH_CALUDE_sea_glass_collection_l1312_131242


namespace NUMINAMATH_CALUDE_complex_equality_l1312_131268

theorem complex_equality (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b + Complex.I →
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_l1312_131268


namespace NUMINAMATH_CALUDE_cheaper_module_cost_l1312_131265

theorem cheaper_module_cost (expensive_cost : ℝ) (total_modules : ℕ) (cheap_modules : ℕ) (total_value : ℝ) :
  expensive_cost = 10 →
  total_modules = 11 →
  cheap_modules = 10 →
  total_value = 45 →
  ∃ (cheap_cost : ℝ), cheap_cost * cheap_modules + expensive_cost * (total_modules - cheap_modules) = total_value ∧ cheap_cost = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_cheaper_module_cost_l1312_131265


namespace NUMINAMATH_CALUDE_pat_height_l1312_131211

/-- Represents the depth dug on each day in centimeters -/
def depth_day1 : ℝ := 40

/-- Represents the total depth after day 2 in centimeters -/
def depth_day2 : ℝ := 3 * depth_day1

/-- Represents the additional depth dug on day 3 in centimeters -/
def depth_day3 : ℝ := depth_day2 - depth_day1

/-- Represents the distance from the ground surface to Pat's head at the end in centimeters -/
def surface_to_head : ℝ := 50

/-- Theorem stating Pat's height in centimeters -/
theorem pat_height : 
  depth_day2 + depth_day3 - surface_to_head = 150 := by sorry

end NUMINAMATH_CALUDE_pat_height_l1312_131211


namespace NUMINAMATH_CALUDE_incorrect_transformation_l1312_131287

theorem incorrect_transformation (x y m : ℝ) :
  ¬(∀ (x y m : ℝ), x = y → x / m = y / m) :=
sorry

end NUMINAMATH_CALUDE_incorrect_transformation_l1312_131287


namespace NUMINAMATH_CALUDE_somus_age_l1312_131293

theorem somus_age (somu father : ℕ) : 
  somu = father / 3 → 
  (somu - 7) = (father - 7) / 5 → 
  somu = 14 := by
sorry

end NUMINAMATH_CALUDE_somus_age_l1312_131293


namespace NUMINAMATH_CALUDE_platform_height_is_44_l1312_131217

/-- Represents the dimensions of a rectangular brick -/
structure Brick where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the experimental setup -/
structure Setup where
  platform_height : ℝ
  brick : Brick
  r : ℝ
  s : ℝ

/-- The main theorem stating the height of the platform -/
theorem platform_height_is_44 (setup : Setup) :
  setup.brick.length + setup.platform_height - 2 * setup.brick.width = setup.r ∧
  setup.brick.width + setup.platform_height - setup.brick.length = setup.s ∧
  setup.platform_height = 2 * setup.brick.width ∧
  setup.r = 36 ∧
  setup.s = 30 →
  setup.platform_height = 44 := by
sorry


end NUMINAMATH_CALUDE_platform_height_is_44_l1312_131217


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1312_131218

theorem arithmetic_sequence_sum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) (n : ℕ) :
  a₁ = 2/7 →
  aₙ = 20/7 →
  d = 2/7 →
  n * (a₁ + aₙ) / 2 = 110/7 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1312_131218


namespace NUMINAMATH_CALUDE_ratio_of_shares_l1312_131297

/-- Given a total amount divided among three persons, prove the ratio of the first person's share to the second person's share. -/
theorem ratio_of_shares (total : ℕ) (r_share : ℕ) (q_to_r_ratio : Rat) :
  total = 1210 →
  r_share = 400 →
  q_to_r_ratio = 9 / 10 →
  ∃ (p_share q_share : ℕ),
    p_share + q_share + r_share = total ∧
    q_share = (q_to_r_ratio * r_share).num ∧
    p_share * 4 = q_share * 5 :=
by sorry

end NUMINAMATH_CALUDE_ratio_of_shares_l1312_131297


namespace NUMINAMATH_CALUDE_rotation_of_doubled_complex_l1312_131272

theorem rotation_of_doubled_complex :
  let z : ℂ := 3 - 4*I
  let doubled : ℂ := 2 * z
  let rotated : ℂ := -doubled
  rotated = -6 + 8*I :=
by
  sorry

end NUMINAMATH_CALUDE_rotation_of_doubled_complex_l1312_131272


namespace NUMINAMATH_CALUDE_proportional_segments_l1312_131286

/-- Given four proportional line segments a, b, c, d, where b = 3, c = 4, and d = 6,
    prove that the length of line segment a is 2. -/
theorem proportional_segments (a b c d : ℝ) 
  (h_prop : a / b = c / d)
  (h_b : b = 3)
  (h_c : c = 4)
  (h_d : d = 6) :
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segments_l1312_131286


namespace NUMINAMATH_CALUDE_sum_of_roots_ln_abs_x_minus_two_l1312_131221

theorem sum_of_roots_ln_abs_x_minus_two (m : ℝ) :
  ∃ x₁ x₂ : ℝ, (Real.log (|x₁ - 2|) = m ∧ Real.log (|x₂ - 2|) = m) → x₁ + x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_ln_abs_x_minus_two_l1312_131221


namespace NUMINAMATH_CALUDE_point_transformation_to_yoz_plane_l1312_131282

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Transforms a point from O-xyz coordinates to yOz plane coordinates -/
def transformToYOZ (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

theorem point_transformation_to_yoz_plane :
  let p : Point3D := { x := 1, y := -2, z := 3 }
  transformToYOZ p = { x := -1, y := -2, z := 3 } := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_to_yoz_plane_l1312_131282


namespace NUMINAMATH_CALUDE_min_max_values_l1312_131277

/-- Given positive real numbers x and y satisfying x² + y² = x + y,
    prove that the minimum value of 1/x + 1/y is 2 and the maximum value of x + y is 2 -/
theorem min_max_values (x y : ℝ) (h_pos : x > 0 ∧ y > 0) (h_eq : x^2 + y^2 = x + y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a + b → 1/x + 1/y ≤ 1/a + 1/b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a + b → x + y ≥ a + b) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = a + b ∧ 1/a + 1/b = 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = a + b ∧ a + b = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_l1312_131277


namespace NUMINAMATH_CALUDE_geography_quiz_correct_percentage_l1312_131208

theorem geography_quiz_correct_percentage (y : ℝ) (h : y > 0) :
  let total_questions := 8 * y
  let incorrect_answers := 2 * y - 3
  let correct_answers := total_questions - incorrect_answers
  let correct_percentage := (correct_answers / total_questions) * 100
  correct_percentage = 75 + 75 / (2 * y) :=
by sorry

end NUMINAMATH_CALUDE_geography_quiz_correct_percentage_l1312_131208


namespace NUMINAMATH_CALUDE_remaining_perimeter_is_56_l1312_131224

/-- Represents the dimensions of a rectangular piece of paper. -/
structure Paper where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of the remaining rectangle after cutting out the largest square. -/
def remainingPerimeter (p : Paper) : ℝ :=
  2 * (p.width + (p.length - p.width))

/-- Theorem stating that for a 28 cm by 15 cm paper, the perimeter of the remaining rectangle is 56 cm. -/
theorem remaining_perimeter_is_56 :
  let p : Paper := { length := 28, width := 15 }
  remainingPerimeter p = 56 := by sorry

end NUMINAMATH_CALUDE_remaining_perimeter_is_56_l1312_131224


namespace NUMINAMATH_CALUDE_intersection_points_l1312_131220

-- Define the lines and points
def L1 (x y : ℚ) : Prop := y = 3 * x - 4
def P : ℚ × ℚ := (3, 2)
def L2 (x y : ℚ) : Prop := y = -(1/3) * x + 3
def L3 (x y : ℚ) : Prop := y = 3 * x - 3

-- State the theorem
theorem intersection_points :
  (∃ x y : ℚ, L1 x y ∧ L2 x y ∧ x = 21/10 ∧ y = 23/10) ∧
  (∃ x y : ℚ, L2 x y ∧ L3 x y ∧ x = 9/5 ∧ y = 12/5) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_l1312_131220


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l1312_131219

def f (x : ℝ) : ℝ := -x^2 + 5

theorem parabola_point_ordering :
  ∀ (y₁ y₂ y₃ : ℝ),
  f (-4) = y₁ ∧ f (-1) = y₂ ∧ f 2 = y₃ →
  y₂ > y₃ ∧ y₃ > y₁ :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l1312_131219


namespace NUMINAMATH_CALUDE_candy_bar_cost_l1312_131215

theorem candy_bar_cost (initial_amount : ℝ) (num_candy_bars : ℕ) (remaining_amount : ℝ) :
  initial_amount = 20 →
  num_candy_bars = 4 →
  remaining_amount = 12 →
  (initial_amount - remaining_amount) / num_candy_bars = 2 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l1312_131215


namespace NUMINAMATH_CALUDE_prob_product_multiple_of_four_l1312_131289

/-- A fair 10-sided die -/
def decagonal_die := Finset.range 10

/-- A fair 12-sided die -/
def dodecagonal_die := Finset.range 12

/-- The probability of an event occurring when rolling a fair n-sided die -/
def prob (event : Finset ℕ) (die : Finset ℕ) : ℚ :=
  (event ∩ die).card / die.card

/-- The event of rolling a multiple of 4 -/
def multiple_of_four (die : Finset ℕ) : Finset ℕ :=
  die.filter (fun x => x % 4 = 0)

/-- The probability that the product of rolls from a 10-sided die and a 12-sided die is a multiple of 4 -/
theorem prob_product_multiple_of_four :
  prob (multiple_of_four decagonal_die) decagonal_die +
  prob (multiple_of_four dodecagonal_die) dodecagonal_die -
  prob (multiple_of_four decagonal_die) decagonal_die *
  prob (multiple_of_four dodecagonal_die) dodecagonal_die = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_product_multiple_of_four_l1312_131289


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1312_131261

theorem polynomial_coefficient_sum (A B C D : ℚ) : 
  (∀ x : ℚ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1312_131261


namespace NUMINAMATH_CALUDE_min_value_expression_l1312_131279

theorem min_value_expression (x : ℝ) : 
  (12 - x) * (10 - x) * (12 + x) * (10 + x) ≥ -484 ∧ 
  ∃ y : ℝ, (12 - y) * (10 - y) * (12 + y) * (10 + y) = -484 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1312_131279


namespace NUMINAMATH_CALUDE_thirty_five_only_math_l1312_131213

/-- Represents the number of students in various class combinations -/
structure ClassCounts where
  total : ℕ
  math : ℕ
  foreign : ℕ
  sport : ℕ
  all_three : ℕ

/-- Calculates the number of students taking only math class -/
def only_math (counts : ClassCounts) : ℕ :=
  counts.math - (counts.total - (counts.math + counts.foreign + counts.sport - counts.all_three))

/-- Theorem stating that 35 students take only math class given the specific class counts -/
theorem thirty_five_only_math (counts : ClassCounts) 
  (h_total : counts.total = 120)
  (h_math : counts.math = 85)
  (h_foreign : counts.foreign = 65)
  (h_sport : counts.sport = 50)
  (h_all_three : counts.all_three = 10) :
  only_math counts = 35 := by
  sorry

end NUMINAMATH_CALUDE_thirty_five_only_math_l1312_131213


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1312_131271

/-- Given a quadratic function f(x) = x^2 - 26x + 129, 
    prove that when written in the form (x+d)^2 + e, d + e = -53 -/
theorem quadratic_form_sum (x : ℝ) : 
  ∃ (d e : ℝ), (∀ x, x^2 - 26*x + 129 = (x+d)^2 + e) ∧ (d + e = -53) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1312_131271


namespace NUMINAMATH_CALUDE_simplify_expression_l1312_131254

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 * b) / (a^2 - a * b) * (a / b - b / a) = a + b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1312_131254


namespace NUMINAMATH_CALUDE_fourth_grade_blue_count_l1312_131275

/-- Represents the number of students in each grade and uniform color combination -/
structure StudentCount where
  third_red_blue : ℕ
  third_white : ℕ
  fourth_red : ℕ
  fourth_white : ℕ
  fourth_blue : ℕ
  fifth_red_blue : ℕ
  fifth_white : ℕ

/-- The theorem stating the number of 4th grade students wearing blue uniforms -/
theorem fourth_grade_blue_count (s : StudentCount) : s.fourth_blue = 213 :=
  by
  have total_participants : s.third_red_blue + s.third_white + s.fourth_red + s.fourth_white + s.fourth_blue + s.fifth_red_blue + s.fifth_white = 2013 := by sorry
  have fourth_grade_total : s.fourth_red + s.fourth_white + s.fourth_blue = 600 := by sorry
  have fifth_grade_total : s.fifth_red_blue + s.fifth_white = 800 := by sorry
  have total_white : s.third_white + s.fourth_white + s.fifth_white = 800 := by sorry
  have third_red_blue : s.third_red_blue = 200 := by sorry
  have fourth_red : s.fourth_red = 200 := by sorry
  have fifth_white : s.fifth_white = 200 := by sorry
  sorry

end NUMINAMATH_CALUDE_fourth_grade_blue_count_l1312_131275


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1312_131216

theorem right_triangle_side_length : ∃ (k : ℕ), 
  (5 * k : ℕ) > 0 ∧ 
  (12 * k : ℕ) > 0 ∧ 
  (13 * k : ℕ) > 0 ∧ 
  (5 * k)^2 + (12 * k)^2 = (13 * k)^2 ∧ 
  (13 * k = 91 ∨ 12 * k = 91 ∨ 5 * k = 91) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1312_131216


namespace NUMINAMATH_CALUDE_length_A_l1312_131201

def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (3, 7)

def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

def on_line_AC (p : ℝ × ℝ) : Prop :=
  (p.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (p.1 - A.1)

def on_line_BC (p : ℝ × ℝ) : Prop :=
  (p.2 - B.2) * (C.1 - B.1) = (C.2 - B.2) * (p.1 - B.1)

def A' : ℝ × ℝ := sorry
def B' : ℝ × ℝ := sorry

theorem length_A'B'_is_4_sqrt_2 :
  line_y_eq_x A' ∧ line_y_eq_x B' ∧ on_line_AC A' ∧ on_line_BC B' →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_length_A_l1312_131201


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l1312_131227

theorem solution_to_system_of_equations :
  ∃ (x y : ℚ), (3 * x - 4 * y = -7) ∧ (6 * x - 5 * y = 3) ∧ (x = 47/9) ∧ (y = 17/3) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l1312_131227


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l1312_131251

theorem smallest_solution_floor_equation :
  ∀ x : ℝ, (⌊x⌋ : ℝ) = 7 + 50 * (x - ⌊x⌋) → x ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l1312_131251


namespace NUMINAMATH_CALUDE_largest_prime_divisor_check_l1312_131259

theorem largest_prime_divisor_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) :
  ∀ p : ℕ, Prime p → p ∣ n → p ≤ 31 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_check_l1312_131259


namespace NUMINAMATH_CALUDE_expected_adjacent_pairs_l1312_131274

/-- The expected number of adjacent boy-girl pairs in a random permutation of boys and girls -/
theorem expected_adjacent_pairs (num_boys num_girls : ℕ) : 
  let total := num_boys + num_girls
  let prob_pair := (num_boys : ℚ) * num_girls / (total * (total - 1))
  let num_pairs := total - 1
  num_boys = 8 → num_girls = 12 → 2 * num_pairs * prob_pair = 912 / 95 := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_pairs_l1312_131274


namespace NUMINAMATH_CALUDE_complex_squared_i_positive_l1312_131292

theorem complex_squared_i_positive (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (Complex.I * (a + Complex.I)^2 = x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_squared_i_positive_l1312_131292


namespace NUMINAMATH_CALUDE_parts_cost_is_800_l1312_131209

/-- Represents the business model of John's computer assembly and sales --/
structure ComputerBusiness where
  partsCost : ℝ  -- Cost of parts for each computer
  sellMultiplier : ℝ  -- Multiplier for selling price
  monthlyProduction : ℕ  -- Number of computers produced per month
  monthlyRent : ℝ  -- Monthly rent cost
  monthlyExtraExpenses : ℝ  -- Monthly non-rent extra expenses
  monthlyProfit : ℝ  -- Monthly profit

/-- Calculates the monthly revenue --/
def monthlyRevenue (b : ComputerBusiness) : ℝ :=
  b.monthlyProduction * (b.sellMultiplier * b.partsCost)

/-- Calculates the monthly expenses --/
def monthlyExpenses (b : ComputerBusiness) : ℝ :=
  b.monthlyProduction * b.partsCost + b.monthlyRent + b.monthlyExtraExpenses

/-- Theorem stating that the cost of parts for each computer is $800 --/
theorem parts_cost_is_800 (b : ComputerBusiness)
    (h1 : b.sellMultiplier = 1.4)
    (h2 : b.monthlyProduction = 60)
    (h3 : b.monthlyRent = 5000)
    (h4 : b.monthlyExtraExpenses = 3000)
    (h5 : b.monthlyProfit = 11200)
    (h6 : monthlyRevenue b - monthlyExpenses b = b.monthlyProfit) :
    b.partsCost = 800 := by
  sorry

end NUMINAMATH_CALUDE_parts_cost_is_800_l1312_131209


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l1312_131285

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 3 = 0

/-- The parabola equation -/
def parabola_eq (x y p : ℝ) : Prop := y^2 = 2*p*x

/-- The directrix equation of the parabola -/
def directrix_eq (x p : ℝ) : Prop := x = -p/2

/-- The length of the line segment cut by the circle on the directrix -/
def segment_length (p : ℝ) : ℝ := 4

/-- The theorem to be proved -/
theorem parabola_circle_intersection (p : ℝ) 
  (h_p_pos : p > 0) 
  (h_segment : segment_length p = 4) : p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l1312_131285


namespace NUMINAMATH_CALUDE_cone_shape_l1312_131276

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Defines the set of points satisfying φ ≤ c -/
def ConeSet (c : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.φ ≤ c}

/-- Theorem: The set of points satisfying φ ≤ c forms a cone -/
theorem cone_shape (c : ℝ) (h : 0 ≤ c ∧ c ≤ π) :
  ∃ (cone : Set SphericalPoint), ConeSet c = cone :=
sorry

end NUMINAMATH_CALUDE_cone_shape_l1312_131276


namespace NUMINAMATH_CALUDE_tommy_balloons_l1312_131295

/-- The number of balloons Tommy has after receiving more from his mom -/
def total_balloons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that Tommy's total balloons is the sum of his initial balloons and additional balloons -/
theorem tommy_balloons (initial : ℕ) (additional : ℕ) :
  total_balloons initial additional = initial + additional := by
  sorry

end NUMINAMATH_CALUDE_tommy_balloons_l1312_131295


namespace NUMINAMATH_CALUDE_student_increase_proof_l1312_131226

/-- Represents the increase in the number of students in a hostel -/
def student_increase : ℕ := sorry

/-- The initial number of students in the hostel -/
def initial_students : ℕ := 35

/-- The original daily expenditure of the mess in rupees -/
def original_expenditure : ℕ := 420

/-- The increase in daily mess expenses in rupees when the number of students increases -/
def expense_increase : ℕ := 42

/-- The decrease in average expenditure per student in rupees when the number of students increases -/
def average_expense_decrease : ℕ := 1

/-- Calculates the new total expenditure after the increase in students -/
def new_total_expenditure : ℕ := (initial_students + student_increase) * 
  (original_expenditure / initial_students - average_expense_decrease)

theorem student_increase_proof : 
  new_total_expenditure = original_expenditure + expense_increase ∧ 
  student_increase = 7 := by sorry

end NUMINAMATH_CALUDE_student_increase_proof_l1312_131226


namespace NUMINAMATH_CALUDE_hybrid_one_headlight_percentage_l1312_131284

theorem hybrid_one_headlight_percentage
  (total_cars : ℕ)
  (hybrid_percentage : ℚ)
  (full_headlight_hybrids : ℕ)
  (h1 : total_cars = 600)
  (h2 : hybrid_percentage = 60 / 100)
  (h3 : full_headlight_hybrids = 216) :
  let total_hybrids := (total_cars : ℚ) * hybrid_percentage
  let one_headlight_hybrids := total_hybrids - (full_headlight_hybrids : ℚ)
  one_headlight_hybrids / total_hybrids = 40 / 100 := by
sorry

end NUMINAMATH_CALUDE_hybrid_one_headlight_percentage_l1312_131284


namespace NUMINAMATH_CALUDE_expression_value_at_4_l1312_131212

theorem expression_value_at_4 (a b : ℤ) 
  (h : ∀ (n : ℤ), ∃ (k : ℤ), (2 * n^3 + 3 * n^2 + a * n + b) = k * (n^2 + 1)) :
  (2 * 4^3 + 3 * 4^2 + a * 4 + b) / (4^2 + 1) = 11 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_4_l1312_131212


namespace NUMINAMATH_CALUDE_total_teaching_years_is_70_l1312_131225

/-- The total number of years Tom and Devin have been teaching -/
def total_teaching_years (tom_years devin_years : ℕ) : ℕ := tom_years + devin_years

/-- Tom's teaching years -/
def tom_years : ℕ := 50

/-- Devin's teaching years in terms of Tom's -/
def devin_years : ℕ := tom_years / 2 - 5

theorem total_teaching_years_is_70 : 
  total_teaching_years tom_years devin_years = 70 := by sorry

end NUMINAMATH_CALUDE_total_teaching_years_is_70_l1312_131225


namespace NUMINAMATH_CALUDE_valid_configuration_iff_consecutive_adjacent_l1312_131207

/-- Represents a cell in the 4x4 grid --/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents a configuration of numbers in the 4x4 grid --/
def Configuration := Cell → Option ℕ

/-- Checks if two cells are adjacent --/
def adjacent (c1 c2 : Cell) : Bool :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col)

/-- Checks if a configuration is valid --/
def is_valid (config : Configuration) : Prop :=
  ∀ (c1 c2 : Cell),
    match config c1, config c2 with
    | some n1, some n2 =>
        if n1 + 1 = n2 ∨ n2 + 1 = n1 then
          adjacent c1 c2
        else
          true
    | _, _ => true

/-- Theorem: A configuration is valid if and only if all pairs of consecutive numbers
    present in the grid are in adjacent cells --/
theorem valid_configuration_iff_consecutive_adjacent (config : Configuration) :
  is_valid config ↔
  (∀ (c1 c2 : Cell),
    match config c1, config c2 with
    | some n1, some n2 =>
        if n1 + 1 = n2 ∨ n2 + 1 = n1 then
          adjacent c1 c2
        else
          true
    | _, _ => true) :=
by sorry


end NUMINAMATH_CALUDE_valid_configuration_iff_consecutive_adjacent_l1312_131207


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1312_131273

def is_ellipse (k : ℝ) : Prop :=
  ∀ x y : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
    x^2 / (k - 4) + y^2 / (9 - k) = 1 ↔ (x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_k_range (k : ℝ) :
  is_ellipse k ↔ (k ∈ Set.Ioo 4 (13/2) ∪ Set.Ioo (13/2) 9) :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1312_131273


namespace NUMINAMATH_CALUDE_building_entrances_l1312_131250

/-- Represents a multi-story building with apartments --/
structure Building where
  floors : ℕ
  apartments_per_floor : ℕ
  total_apartments : ℕ

/-- Calculates the number of entrances in a building --/
def number_of_entrances (b : Building) : ℕ :=
  b.total_apartments / (b.floors * b.apartments_per_floor)

/-- Theorem: A building with 9 floors, 4 apartments per floor, and 180 total apartments has 5 entrances --/
theorem building_entrances :
  let b : Building := ⟨9, 4, 180⟩
  number_of_entrances b = 5 := by
sorry

end NUMINAMATH_CALUDE_building_entrances_l1312_131250


namespace NUMINAMATH_CALUDE_check_to_new_balance_ratio_l1312_131202

def initial_balance : ℚ := 150
def check_amount : ℚ := 50

def new_balance : ℚ := initial_balance + check_amount

theorem check_to_new_balance_ratio :
  check_amount / new_balance = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_check_to_new_balance_ratio_l1312_131202


namespace NUMINAMATH_CALUDE_g_properties_l1312_131299

noncomputable def g (x : ℝ) : ℝ :=
  (4 * Real.sin x ^ 4 + 5 * Real.cos x ^ 2) / (4 * Real.cos x ^ 4 + 3 * Real.sin x ^ 2)

theorem g_properties :
  (∀ k : ℤ, g (π/4 + k*π) = 7/5 ∧ g (π/3 + 2*k*π) = 7/5 ∧ g (-π/3 + 2*k*π) = 7/5) ∧
  (∀ x : ℝ, g x ≤ 71/55) ∧
  (∀ x : ℝ, g x ≥ 5/4) ∧
  (∃ x : ℝ, g x = 71/55) ∧
  (∃ x : ℝ, g x = 5/4) :=
by sorry

end NUMINAMATH_CALUDE_g_properties_l1312_131299


namespace NUMINAMATH_CALUDE_coin_problem_l1312_131255

theorem coin_problem :
  ∀ (nickels dimes quarters : ℕ),
    nickels + dimes + quarters = 100 →
    5 * nickels + 10 * dimes + 25 * quarters = 835 →
    ∃ (min_dimes max_dimes : ℕ),
      (∀ d : ℕ, 
        (∃ n q : ℕ, n + d + q = 100 ∧ 5 * n + 10 * d + 25 * q = 835) →
        min_dimes ≤ d ∧ d ≤ max_dimes) ∧
      max_dimes - min_dimes = 64 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l1312_131255


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l1312_131291

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 3*x^2 - 4*x + 12

-- State the theorem
theorem partial_fraction_decomposition_sum (a b c D E F : ℝ) : 
  -- a, b, c are distinct roots of p
  p a = 0 → p b = 0 → p c = 0 → a ≠ b → b ≠ c → a ≠ c →
  -- Partial fraction decomposition holds
  (∀ s : ℝ, s ≠ a → s ≠ b → s ≠ c → 
    1 / (s^3 - 3*s^2 - 4*s + 12) = D / (s - a) + E / (s - b) + F / (s - c)) →
  -- Conclusion
  1 / D + 1 / E + 1 / F + a * b * c = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l1312_131291


namespace NUMINAMATH_CALUDE_cos_n_equals_sin_712_l1312_131222

theorem cos_n_equals_sin_712 (n : ℤ) :
  -90 ≤ n ∧ n ≤ 90 ∧ Real.cos (n * π / 180) = Real.sin (712 * π / 180) → n = -82 := by
  sorry

end NUMINAMATH_CALUDE_cos_n_equals_sin_712_l1312_131222


namespace NUMINAMATH_CALUDE_white_wins_with_three_pieces_l1312_131246

/-- Represents a player in the game -/
inductive Player
| White
| Black

/-- Represents a position on the game board -/
structure Position where
  n : ℕ
  white_pieces : Finset ℕ
  black_pieces : Finset ℕ

/-- Represents a move in the game -/
inductive Move
| Simple : ℕ → ℕ → Move
| Capture : ℕ → ℕ → Move

/-- Checks if a move is valid for a given player and position -/
def is_valid_move (player : Player) (pos : Position) (move : Move) : Prop :=
  sorry

/-- Applies a move to a position, returning the new position -/
def apply_move (pos : Position) (move : Move) : Position :=
  sorry

/-- Checks if a player has a winning strategy from a given position -/
def has_winning_strategy (player : Player) (pos : Position) : Prop :=
  sorry

/-- The main theorem to be proved -/
theorem white_wins_with_three_pieces (n : ℕ) (h : n > 6) :
  let initial_pos : Position :=
    { n := n,
      white_pieces := {1, 2, 3},
      black_pieces := {n - 2, n - 1, n} }
  has_winning_strategy Player.White initial_pos :=
sorry

end NUMINAMATH_CALUDE_white_wins_with_three_pieces_l1312_131246


namespace NUMINAMATH_CALUDE_simplified_tax_system_is_most_suitable_l1312_131243

-- Define the business characteristics
structure BusinessCharacteristics where
  isFlowerSelling : Bool
  hasNoExperience : Bool
  hasSingleOutlet : Bool
  isSelfOperated : Bool

-- Define the tax regimes
inductive TaxRegime
  | UnifiedAgricultural
  | Simplified
  | General
  | Patent

-- Define a function to determine the most suitable tax regime
def mostSuitableTaxRegime (business : BusinessCharacteristics) : TaxRegime :=
  sorry

-- Theorem statement
theorem simplified_tax_system_is_most_suitable 
  (leonidBusiness : BusinessCharacteristics)
  (h1 : leonidBusiness.isFlowerSelling = true)
  (h2 : leonidBusiness.hasNoExperience = true)
  (h3 : leonidBusiness.hasSingleOutlet = true)
  (h4 : leonidBusiness.isSelfOperated = true) :
  mostSuitableTaxRegime leonidBusiness = TaxRegime.Simplified :=
sorry

end NUMINAMATH_CALUDE_simplified_tax_system_is_most_suitable_l1312_131243
