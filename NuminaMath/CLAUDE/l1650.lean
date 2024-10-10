import Mathlib

namespace square_root_difference_equals_two_sqrt_three_l1650_165093

theorem square_root_difference_equals_two_sqrt_three :
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end square_root_difference_equals_two_sqrt_three_l1650_165093


namespace max_value_theorem_l1650_165052

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2/2 = 1) :
  ∃ (M : ℝ), M = (3 * Real.sqrt 2) / 4 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 + y'^2/2 = 1 →
    x' * Real.sqrt (1 + y'^2) ≤ M :=
by sorry

end max_value_theorem_l1650_165052


namespace sqrt_18_times_sqrt_72_l1650_165039

theorem sqrt_18_times_sqrt_72 : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end sqrt_18_times_sqrt_72_l1650_165039


namespace trapezoid_sides_l1650_165000

/-- A rectangular trapezoid with an inscribed circle -/
structure RectangularTrapezoid (r : ℝ) where
  /-- The radius of the inscribed circle -/
  radius : r > 0
  /-- The shorter base of the trapezoid -/
  short_base : ℝ
  /-- The longer base of the trapezoid -/
  long_base : ℝ
  /-- One of the non-parallel sides of the trapezoid -/
  side1 : ℝ
  /-- The other non-parallel side of the trapezoid -/
  side2 : ℝ
  /-- The shorter base is equal to 4r/3 -/
  short_base_eq : short_base = 4*r/3
  /-- The circle is inscribed, so one non-parallel side equals the diameter -/
  side1_eq_diameter : side1 = 2*r
  /-- Property of trapezoids with an inscribed circle -/
  inscribed_circle_property : side1 + long_base = short_base + side2

/-- Theorem: The sides of the rectangular trapezoid with an inscribed circle of radius r 
    and shorter base 4r/3 are 4r, 10r/3, and 2r -/
theorem trapezoid_sides (r : ℝ) (t : RectangularTrapezoid r) : 
  t.short_base = 4*r/3 ∧ t.long_base = 10*r/3 ∧ t.side1 = 2*r ∧ t.side2 = 8*r/3 := by
  sorry

end trapezoid_sides_l1650_165000


namespace correct_result_l1650_165073

variables {a b c : ℤ}

theorem correct_result (A : ℤ) (h : A + 2 * (a * b + 2 * b * c - 4 * a * c) = 3 * a * b - 2 * a * c + 5 * b * c) :
  A - 2 * (a * b + 2 * b * c - 4 * a * c) = -a * b + 14 * a * c - 3 * b * c := by
  sorry

end correct_result_l1650_165073


namespace triangle_with_100_degree_angle_is_obtuse_l1650_165038

/-- A triangle is obtuse if it has an interior angle greater than 90 degrees. -/
def IsObtuse (a b c : ℝ) : Prop := 
  (a + b + c = 180) ∧ (max a (max b c) > 90)

/-- If a triangle has an interior angle of 100 degrees, then it is obtuse. -/
theorem triangle_with_100_degree_angle_is_obtuse (a b c : ℝ) : 
  (a + b + c = 180) → (max a (max b c) = 100) → IsObtuse a b c := by
  sorry

#check triangle_with_100_degree_angle_is_obtuse

end triangle_with_100_degree_angle_is_obtuse_l1650_165038


namespace division_problem_l1650_165015

theorem division_problem (number quotient remainder divisor : ℕ) : 
  number = quotient * divisor + remainder →
  divisor = 163 →
  quotient = 76 →
  remainder = 13 →
  number = 12401 := by
sorry

end division_problem_l1650_165015


namespace infantry_column_problem_l1650_165061

/-- Given an infantry column of length 1 km, moving at speed x km/h,
    and Sergeant Kim moving at speed 3x km/h, if the infantry column
    covers 2.4 km while Kim travels to the front of the column and back,
    then Kim's total distance traveled is 3.6 km. -/
theorem infantry_column_problem (x : ℝ) (h : x > 0) :
  let column_length : ℝ := 1
  let column_speed : ℝ := x
  let kim_speed : ℝ := 3 * x
  let column_distance : ℝ := 2.4
  let time := column_distance / column_speed
  let kim_distance := kim_speed * time
  kim_distance = 3.6 := by sorry

end infantry_column_problem_l1650_165061


namespace lcm_48_180_l1650_165007

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end lcm_48_180_l1650_165007


namespace quadratic_rewrite_ratio_l1650_165018

/-- Given a quadratic equation x^2 + 2100x + 4200, prove that when rewritten in the form (x+b)^2 + c, the value of c/b is -1034 -/
theorem quadratic_rewrite_ratio : 
  ∃ (b c : ℝ), (∀ x, x^2 + 2100*x + 4200 = (x + b)^2 + c) ∧ c/b = -1034 := by
sorry

end quadratic_rewrite_ratio_l1650_165018


namespace isochronous_growth_law_l1650_165067

theorem isochronous_growth_law (k α : ℝ) (h1 : k > 0) (h2 : α > 0) :
  (∀ (x y : ℝ), y = k * x^α → (16 * x)^α = 8 * y) → α = 3/4 := by
  sorry

end isochronous_growth_law_l1650_165067


namespace sin_minus_cos_equals_one_l1650_165032

theorem sin_minus_cos_equals_one (x : Real) : 
  0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = 1 ↔ x = Real.pi / 2 ∨ x = Real.pi) := by
  sorry

end sin_minus_cos_equals_one_l1650_165032


namespace tangent_intersection_y_coordinate_l1650_165090

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

end tangent_intersection_y_coordinate_l1650_165090


namespace function_properties_and_triangle_perimeter_l1650_165092

noncomputable def f (m : ℝ) (θ : ℝ) (x : ℝ) : ℝ := (m + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

theorem function_properties_and_triangle_perimeter
  (m : ℝ)
  (θ : ℝ)
  (h1 : ∀ x, f m θ x = -f m θ (-x))  -- f is an odd function
  (h2 : f m θ (π/4) = 0)
  (h3 : 0 < θ)
  (h4 : θ < π) :
  (∀ x, f m θ x = -1/2 * Real.sin (4*x)) ∧
  (∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    f m θ (Real.arccos (1/2) / 2 + π/24) = -1/2 ∧
    1 = 1 ∧
    a * b = 2 * Real.sqrt 3 ∧
    a + b + 1 = 3 + Real.sqrt 3) :=
by sorry

end function_properties_and_triangle_perimeter_l1650_165092


namespace viviana_vanilla_chips_l1650_165051

/-- Given the conditions about chocolate and vanilla chips, prove that Viviana has 20 vanilla chips. -/
theorem viviana_vanilla_chips 
  (viviana_chocolate : ℕ) 
  (viviana_vanilla : ℕ) 
  (susana_chocolate : ℕ) 
  (susana_vanilla : ℕ) 
  (h1 : viviana_chocolate = susana_chocolate + 5)
  (h2 : susana_vanilla = (3 * viviana_vanilla) / 4)
  (h3 : susana_chocolate = 25)
  (h4 : viviana_chocolate + susana_chocolate + viviana_vanilla + susana_vanilla = 90) :
  viviana_vanilla = 20 := by
sorry

end viviana_vanilla_chips_l1650_165051


namespace set_union_problem_l1650_165075

theorem set_union_problem (a b : ℝ) :
  let A : Set ℝ := {-1, a}
  let B : Set ℝ := {2^a, b}
  A ∩ B = {1} → A ∪ B = {-1, 1, 2} := by
  sorry

end set_union_problem_l1650_165075


namespace geometric_sequence_common_ratio_l1650_165048

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a₁ + a₃ = 20 and a₂ + a₄ = 40, then q = 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = q * a n) 
  (h2 : a 1 + a 3 = 20) 
  (h3 : a 2 + a 4 = 40) : 
  q = 2 := by
  sorry

end geometric_sequence_common_ratio_l1650_165048


namespace ribbon_remaining_l1650_165021

/-- Proves that given a ribbon of 51 meters, after cutting 100 pieces of 15 centimeters each, 
    the remaining ribbon length is 36 meters. -/
theorem ribbon_remaining (total_length : ℝ) (num_pieces : ℕ) (piece_length : ℝ) :
  total_length = 51 →
  num_pieces = 100 →
  piece_length = 0.15 →
  total_length - (num_pieces : ℝ) * piece_length = 36 :=
by sorry

end ribbon_remaining_l1650_165021


namespace average_speed_calculation_l1650_165069

theorem average_speed_calculation (local_distance : ℝ) (local_speed : ℝ) 
  (highway_distance : ℝ) (highway_speed : ℝ) : 
  local_distance = 40 ∧ local_speed = 20 ∧ highway_distance = 180 ∧ highway_speed = 60 →
  (local_distance + highway_distance) / ((local_distance / local_speed) + (highway_distance / highway_speed)) = 44 := by
  sorry

end average_speed_calculation_l1650_165069


namespace angle_complement_half_supplement_l1650_165046

theorem angle_complement_half_supplement : 
  ∃ (x : ℝ), x > 0 ∧ x < 90 ∧ (90 - x) = (1/2) * (180 - x) ∧ x = 60 := by
  sorry

end angle_complement_half_supplement_l1650_165046


namespace f_properties_l1650_165059

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3) + 1

theorem f_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → x₂ < 5 * Real.pi / 12 → f x₁ < f x₂) ∧
  (∀ x₁ x₂ x₃, x₁ ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) →
               x₂ ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) →
               x₃ ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) →
               f x₁ + f x₃ > f x₂) :=
by sorry

end f_properties_l1650_165059


namespace sequence_properties_l1650_165016

/-- Geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- Arithmetic sequence with positive common difference -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), d > 0 ∧ ∀ n, b (n + 1) = b n + d

theorem sequence_properties (a b : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_arith : arithmetic_sequence b)
  (h_eq3 : a 3 = b 3)
  (h_eq7 : a 7 = b 7) :
  a 5 < b 5 ∧ a 1 > b 1 ∧ a 9 > b 9 := by
  sorry

end sequence_properties_l1650_165016


namespace system_solution_unique_l1650_165083

theorem system_solution_unique :
  ∃! (x y : ℝ), 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 :=
by
  sorry

end system_solution_unique_l1650_165083


namespace chosen_numbers_sum_l1650_165057

theorem chosen_numbers_sum (S : Finset ℕ) : 
  S.card = 5 ∧ 
  S ⊆ Finset.range 9 ∧ 
  S.sum id = ((Finset.range 9).sum id - S.sum id) / 2 → 
  S.sum id = 15 := by sorry

end chosen_numbers_sum_l1650_165057


namespace smallest_non_factor_product_of_36_l1650_165055

theorem smallest_non_factor_product_of_36 (x y : ℕ+) : 
  x ≠ y →
  x ∣ 36 →
  y ∣ 36 →
  ¬(x * y ∣ 36) →
  (∀ a b : ℕ+, a ≠ b → a ∣ 36 → b ∣ 36 → ¬(a * b ∣ 36) → x * y ≤ a * b) →
  x * y = 8 := by
sorry

end smallest_non_factor_product_of_36_l1650_165055


namespace isosceles_right_triangle_hypotenuse_l1650_165081

/-- A right triangle with two 45° angles and one 90° angle, and an inscribed circle of radius 8 cm has a hypotenuse of length 16(√2 + 1) cm. -/
theorem isosceles_right_triangle_hypotenuse (r : ℝ) (h : r = 8) :
  ∃ (a : ℝ), a > 0 ∧ 
  (a * a = 2 * r * r * (2 + Real.sqrt 2)) ∧
  (a * Real.sqrt 2 = 16 * (Real.sqrt 2 + 1)) := by
  sorry

end isosceles_right_triangle_hypotenuse_l1650_165081


namespace inequality_proof_l1650_165033

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end inequality_proof_l1650_165033


namespace angle_ratio_not_determine_right_triangle_l1650_165086

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

end angle_ratio_not_determine_right_triangle_l1650_165086


namespace smallest_solution_l1650_165042

-- Define the equation
def equation (x : ℝ) : Prop :=
  3 * x / (x - 3) + (3 * x^2 - 27) / x = 15

-- State the theorem
theorem smallest_solution :
  ∃ (s : ℝ), s = 1 - Real.sqrt 10 ∧
  equation s ∧
  (∀ (x : ℝ), equation x → x ≥ s) :=
sorry

end smallest_solution_l1650_165042


namespace one_twenty_million_properties_l1650_165099

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

/-- Counts the number of significant figures in a scientific notation -/
def countSignificantFigures (sn : ScientificNotation) : ℕ := sorry

/-- Determines the place value of accuracy for a number -/
inductive PlaceValue
  | Ones
  | Tens
  | Hundreds
  | Thousands
  | TenThousands
  | HundredThousands
  | Millions
  | TenMillions
  | HundredMillions

def getPlaceValueAccuracy (x : ℝ) : PlaceValue := sorry

theorem one_twenty_million_properties :
  let x : ℝ := 120000000
  let sn := toScientificNotation x
  countSignificantFigures sn = 2 ∧ getPlaceValueAccuracy x = PlaceValue.Millions := by sorry

end one_twenty_million_properties_l1650_165099


namespace negation_of_implication_l1650_165091

theorem negation_of_implication (x : ℝ) :
  ¬(x > 0 → x^2 > 0) ↔ (x ≤ 0 → x^2 ≤ 0) :=
by
  sorry

end negation_of_implication_l1650_165091


namespace sequence_general_term_l1650_165072

def sequence_property (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  ∀ n : ℕ, n > 0 → (n + 1 : ℝ) * a (n + 1) - n * (a n)^2 + (n + 1 : ℝ) * a n * a (n + 1) - n * a n = 0

theorem sequence_general_term (a : ℕ → ℝ) (h : sequence_property a) :
  ∀ n : ℕ, n > 0 → a n = 1 / n :=
sorry

end sequence_general_term_l1650_165072


namespace round_984530_to_nearest_ten_thousand_l1650_165095

-- Define a function to round to the nearest ten thousand
def roundToNearestTenThousand (n : ℤ) : ℤ :=
  (n + 5000) / 10000 * 10000

-- State the theorem
theorem round_984530_to_nearest_ten_thousand :
  roundToNearestTenThousand 984530 = 980000 := by
  sorry

end round_984530_to_nearest_ten_thousand_l1650_165095


namespace student_meeting_probability_l1650_165089

/-- The probability of two students meeting given specific conditions -/
theorem student_meeting_probability (α : ℝ) (h : 0 < α ∧ α < 60) :
  let p := 1 - ((60 - α) / 60)^2
  0 ≤ p ∧ p ≤ 1 :=
by sorry

end student_meeting_probability_l1650_165089


namespace proposition_s_range_p_or_q_and_not_q_range_l1650_165003

-- Define propositions p, q, and s
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / (4 - m) + y^2 / m = 1 → x^2 / a^2 + y^2 / b^2 = 1

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0

def s (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

-- Theorem 1
theorem proposition_s_range (m : ℝ) : s m → m < 0 ∨ m ≥ 1 := by sorry

-- Theorem 2
theorem p_or_q_and_not_q_range (m : ℝ) : (p m ∨ q m) ∧ ¬(q m) → 1 ≤ m ∧ m < 2 := by sorry

end proposition_s_range_p_or_q_and_not_q_range_l1650_165003


namespace exists_rational_rearrangement_l1650_165050

/-- Represents an infinite decimal fraction as a sequence of digits. -/
def InfiniteDecimal := ℕ → Fin 10

/-- Represents a rearrangement of digits. -/
def Rearrangement := ℕ → ℕ

/-- A number is rational if it can be expressed as a ratio of two integers. -/
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

/-- Converts an InfiniteDecimal to a real number. -/
noncomputable def toReal (d : InfiniteDecimal) : ℝ := sorry

/-- Applies a rearrangement to an InfiniteDecimal. -/
def applyRearrangement (d : InfiniteDecimal) (r : Rearrangement) : InfiniteDecimal :=
  fun n => d (r n)

/-- Theorem: For any infinite decimal, there exists a rearrangement that results in a rational number. -/
theorem exists_rational_rearrangement (d : InfiniteDecimal) :
  ∃ (r : Rearrangement), IsRational (toReal (applyRearrangement d r)) := by sorry

end exists_rational_rearrangement_l1650_165050


namespace positive_sum_and_product_imply_positive_quadratic_root_conditions_quadratic_root_conditions_not_sufficient_l1650_165082

-- Statement 3
theorem positive_sum_and_product_imply_positive (a b : ℝ) :
  a + b > 0 → a * b > 0 → a > 0 ∧ b > 0 := by sorry

-- Statement 4
def has_two_distinct_positive_roots (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0

theorem quadratic_root_conditions (a b c : ℝ) (h : a ≠ 0) :
  has_two_distinct_positive_roots a b c →
  b / a < 0 ∧ c / a > 0 := by sorry

theorem quadratic_root_conditions_not_sufficient (a b c : ℝ) (h : a ≠ 0) :
  b / a < 0 ∧ c / a > 0 →
  ¬(has_two_distinct_positive_roots a b c ↔ True) := by sorry

end positive_sum_and_product_imply_positive_quadratic_root_conditions_quadratic_root_conditions_not_sufficient_l1650_165082


namespace problem_solution_l1650_165004

theorem problem_solution (m n : ℕ+) 
  (h1 : ∃ k : ℕ, m = 111 * k)
  (h2 : ∃ l : ℕ, n = 31 * l)
  (h3 : m + n = 2017) :
  n - m = 463 := by
  sorry

end problem_solution_l1650_165004


namespace winter_wheat_harvest_scientific_notation_l1650_165027

theorem winter_wheat_harvest_scientific_notation :
  239000000 = 2.39 * (10 ^ 8) := by
  sorry

end winter_wheat_harvest_scientific_notation_l1650_165027


namespace ship_distance_constant_l1650_165010

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircular path -/
structure SemicircularPath where
  center : Point
  radius : ℝ

/-- Represents the ship's journey -/
structure ShipJourney where
  path1 : SemicircularPath
  path2 : SemicircularPath

/-- Represents the ship's position along its journey -/
structure ShipPosition where
  t : ℝ  -- Time parameter (0 ≤ t ≤ 2)
  isOnFirstPath : Bool

/-- Distance function for the ship's position -/
def distance (journey : ShipJourney) (pos : ShipPosition) : ℝ :=
  if pos.isOnFirstPath then journey.path1.radius else journey.path2.radius

theorem ship_distance_constant (journey : ShipJourney) :
  ∀ t1 t2 : ℝ, 0 ≤ t1 ∧ t1 ≤ 1 → 0 ≤ t2 ∧ t2 ≤ 1 →
    distance journey { t := t1, isOnFirstPath := true } =
    distance journey { t := t2, isOnFirstPath := true } ∧
  ∀ t3 t4 : ℝ, 1 < t3 ∧ t3 ≤ 2 → 1 < t4 ∧ t4 ≤ 2 →
    distance journey { t := t3, isOnFirstPath := false } =
    distance journey { t := t4, isOnFirstPath := false } ∧
  journey.path1.radius ≠ journey.path2.radius →
    ∃ t5 t6 : ℝ, 0 ≤ t5 ∧ t5 ≤ 1 ∧ 1 < t6 ∧ t6 ≤ 2 ∧
      distance journey { t := t5, isOnFirstPath := true } ≠
      distance journey { t := t6, isOnFirstPath := false } :=
by
  sorry

end ship_distance_constant_l1650_165010


namespace sphere_surface_area_circumscribing_cube_l1650_165028

theorem sphere_surface_area_circumscribing_cube (edge_length : ℝ) (h : edge_length = Real.sqrt 3) : 
  let cube_diagonal := Real.sqrt (3 * edge_length ^ 2)
  let sphere_radius := cube_diagonal / 2
  4 * Real.pi * sphere_radius ^ 2 = 9 * Real.pi := by
  sorry

end sphere_surface_area_circumscribing_cube_l1650_165028


namespace perry_vs_phil_l1650_165063

/-- The number of games won by each player -/
structure GolfWins where
  phil : ℕ
  charlie : ℕ
  dana : ℕ
  perry : ℕ

/-- The conditions of the golf game results -/
def golf_conditions (w : GolfWins) : Prop :=
  w.perry = w.dana + 5 ∧
  w.charlie = w.dana - 2 ∧
  w.phil = w.charlie + 3 ∧
  w.phil = 12

theorem perry_vs_phil (w : GolfWins) (h : golf_conditions w) : w.perry = w.phil + 4 :=
sorry

end perry_vs_phil_l1650_165063


namespace line_slope_problem_l1650_165062

/-- Given a line passing through points (-1, -4) and (3, k) with slope k, prove k = 4/3 -/
theorem line_slope_problem (k : ℚ) : 
  (k - (-4)) / (3 - (-1)) = k → k = 4/3 := by
  sorry

end line_slope_problem_l1650_165062


namespace base_2_representation_of_101_l1650_165065

theorem base_2_representation_of_101 : 
  ∃ (a b c d e f g : Nat), 
    (a = 1 ∧ b = 1 ∧ c = 0 ∧ d = 0 ∧ e = 1 ∧ f = 0 ∧ g = 1) ∧
    101 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end base_2_representation_of_101_l1650_165065


namespace sprinter_target_heart_rate_l1650_165077

/-- Calculates the maximum heart rate given the age --/
def maxHeartRate (age : ℕ) : ℕ := 225 - age

/-- Calculates the target heart rate as a percentage of the maximum heart rate --/
def targetHeartRate (maxRate : ℕ) : ℚ := 0.85 * maxRate

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (x : ℚ) : ℤ := (x + 1/2).floor

theorem sprinter_target_heart_rate :
  let age : ℕ := 30
  let max_rate := maxHeartRate age
  let target_rate := targetHeartRate max_rate
  roundToNearest target_rate = 166 := by sorry

end sprinter_target_heart_rate_l1650_165077


namespace min_sum_of_product_l1650_165012

theorem min_sum_of_product (a b : ℤ) : 
  a ≤ 0 → b ≤ 0 → a * b = 144 → (∀ x y : ℤ, x ≤ 0 → y ≤ 0 → x * y = 144 → a + b ≤ x + y) → a + b = -30 :=
by sorry

end min_sum_of_product_l1650_165012


namespace night_crew_ratio_l1650_165008

theorem night_crew_ratio (D N : ℚ) (h1 : D > 0) (h2 : N > 0) : 
  (N * (3/4)) / (D + N * (3/4)) = 1/3 → N/D = 2/3 := by
  sorry

end night_crew_ratio_l1650_165008


namespace set_A_is_correct_l1650_165094

-- Define the universe set U
def U : Set ℝ := {x | x > 0}

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := {x | 0 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | x ≥ 3}

-- Theorem statement
theorem set_A_is_correct : A = U \ complement_A_in_U := by sorry

end set_A_is_correct_l1650_165094


namespace square_minus_product_plus_square_l1650_165023

theorem square_minus_product_plus_square : 7^2 - 4*5 + 6^2 = 65 := by
  sorry

end square_minus_product_plus_square_l1650_165023


namespace thomas_needs_2000_more_l1650_165080

/-- Thomas's savings scenario over two years -/
def thomas_savings_scenario (first_year_allowance : ℕ) (second_year_hourly_rate : ℕ) 
  (second_year_weekly_hours : ℕ) (car_cost : ℕ) (weekly_expenses : ℕ) : Prop :=
  let weeks_per_year : ℕ := 52
  let total_weeks : ℕ := 2 * weeks_per_year
  let first_year_earnings : ℕ := first_year_allowance * weeks_per_year
  let second_year_earnings : ℕ := second_year_hourly_rate * second_year_weekly_hours * weeks_per_year
  let total_earnings : ℕ := first_year_earnings + second_year_earnings
  let total_expenses : ℕ := weekly_expenses * total_weeks
  let savings : ℕ := total_earnings - total_expenses
  car_cost - savings = 2000

/-- Theorem stating Thomas needs $2000 more to buy the car -/
theorem thomas_needs_2000_more :
  thomas_savings_scenario 50 9 30 15000 35 := by sorry

end thomas_needs_2000_more_l1650_165080


namespace min_value_of_expression_range_of_a_l1650_165013

theorem min_value_of_expression (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

theorem range_of_a (a : ℝ) : (∃ x > 1, a ≤ x + 1 / (x - 1)) ↔ a ≤ 3 := by
  sorry

end min_value_of_expression_range_of_a_l1650_165013


namespace cube_occupation_percentage_l1650_165011

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of cubes that can fit along a given dimension -/
def cubesFit (dimension cubeSide : ℕ) : ℕ :=
  dimension / cubeSide

/-- Calculates the volume occupied by cubes in the box -/
def occupiedVolume (d : BoxDimensions) (cubeSide : ℕ) : ℕ :=
  (cubesFit d.length cubeSide) * (cubesFit d.width cubeSide) * (cubesFit d.height cubeSide) * (cubeSide ^ 3)

/-- Theorem: The percentage of volume occupied by 4-inch cubes in a 
    8x7x12 inch box is equal to 4/7 -/
theorem cube_occupation_percentage :
  let boxDim : BoxDimensions := { length := 8, width := 7, height := 12 }
  let cubeSide : ℕ := 4
  (occupiedVolume boxDim cubeSide : ℚ) / (boxVolume boxDim : ℚ) = 4 / 7 := by
  sorry

end cube_occupation_percentage_l1650_165011


namespace cricket_team_average_age_l1650_165064

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (A : ℚ),
    team_size = 11 →
    captain_age = 26 →
    wicket_keeper_age_diff = 5 →
    (team_size : ℚ) * A - (captain_age + (captain_age + wicket_keeper_age_diff)) = 
      (team_size - 2 : ℚ) * (A - 1) →
    A = 24 := by
  sorry

end cricket_team_average_age_l1650_165064


namespace underdog_wins_probability_l1650_165001

def best_of_five_probability (p : ℚ) : ℚ :=
  (p^5) + 5 * (p^4) * (1 - p) + 10 * (p^3) * ((1 - p)^2)

theorem underdog_wins_probability :
  best_of_five_probability (1/3) = 17/81 := by
  sorry

end underdog_wins_probability_l1650_165001


namespace inequality_always_true_l1650_165022

theorem inequality_always_true : ∀ x : ℝ, 4 * x^2 - 4 * x + 1 ≥ 0 := by
  sorry

end inequality_always_true_l1650_165022


namespace pages_difference_l1650_165020

theorem pages_difference (beatrix_pages cristobal_pages : ℕ) : 
  beatrix_pages = 704 →
  cristobal_pages = 15 + 3 * beatrix_pages →
  cristobal_pages - beatrix_pages = 1423 := by
sorry

end pages_difference_l1650_165020


namespace tessellating_nonagon_angles_l1650_165009

/-- A nonagon that tessellates the plane and can be decomposed into seven triangles -/
structure TessellatingNonagon where
  /-- The vertices of the nonagon -/
  vertices : Fin 9 → ℝ × ℝ
  /-- The nonagon tessellates the plane -/
  tessellates : sorry
  /-- The nonagon can be decomposed into seven triangles -/
  decomposable : sorry
  /-- Some sides of the nonagon form rhombuses with equal side lengths -/
  has_rhombuses : sorry

/-- The angles of a tessellating nonagon -/
def nonagon_angles (n : TessellatingNonagon) : Fin 9 → ℝ := sorry

/-- Theorem stating the angles of the tessellating nonagon -/
theorem tessellating_nonagon_angles (n : TessellatingNonagon) :
  nonagon_angles n = ![105, 60, 195, 195, 195, 15, 165, 165, 165] := by sorry

end tessellating_nonagon_angles_l1650_165009


namespace absolute_value_at_zero_l1650_165043

-- Define a fourth-degree polynomial with real coefficients
def fourthDegreePolynomial (a b c d e : ℝ) : ℝ → ℝ := λ x => a*x^4 + b*x^3 + c*x^2 + d*x + e

-- State the theorem
theorem absolute_value_at_zero (a b c d e : ℝ) :
  let g := fourthDegreePolynomial a b c d e
  (|g 1| = 16 ∧ |g 3| = 16 ∧ |g 4| = 16 ∧ |g 5| = 16 ∧ |g 6| = 16 ∧ |g 7| = 16) →
  |g 0| = 54 := by
  sorry


end absolute_value_at_zero_l1650_165043


namespace monic_polynomial_problem_l1650_165084

theorem monic_polynomial_problem (g : ℝ → ℝ) :
  (∃ b c : ℝ, ∀ x, g x = x^2 + b*x + c) →  -- g is a monic polynomial of degree 2
  g 0 = 6 →                               -- g(0) = 6
  g 1 = 12 →                              -- g(1) = 12
  ∀ x, g x = x^2 + 5*x + 6 :=              -- Conclusion: g(x) = x^2 + 5x + 6
by
  sorry

end monic_polynomial_problem_l1650_165084


namespace cookie_pack_cost_l1650_165034

/-- Prove that the cost of each pack of cookies is $1.50 --/
theorem cookie_pack_cost
  (cookies_per_chore : ℕ)
  (chores_per_week : ℕ)
  (num_siblings : ℕ)
  (num_weeks : ℕ)
  (cookies_per_pack : ℕ)
  (total_money : ℚ)
  (h1 : cookies_per_chore = 3)
  (h2 : chores_per_week = 4)
  (h3 : num_siblings = 2)
  (h4 : num_weeks = 10)
  (h5 : cookies_per_pack = 24)
  (h6 : total_money = 15) :
  (total_money / (cookies_per_chore * chores_per_week * num_siblings * num_weeks / cookies_per_pack) : ℚ) = 3/2 := by
  sorry

end cookie_pack_cost_l1650_165034


namespace systematic_sample_fourth_element_l1650_165087

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

end systematic_sample_fourth_element_l1650_165087


namespace geometric_sequence_property_l1650_165066

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence {a_n} where a_2 * a_3 = 5 and a_5 * a_6 = 10, prove that a_8 * a_9 = 20. -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_23 : a 2 * a 3 = 5) 
    (h_56 : a 5 * a 6 = 10) : 
  a 8 * a 9 = 20 := by
  sorry

end geometric_sequence_property_l1650_165066


namespace trapezoid_bases_l1650_165047

/-- An isosceles trapezoid with a circumscribed circle -/
structure IsoscelesTrapezoid where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The ratio of the lower part of the height to the total height -/
  heightRatio : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The shorter base of the trapezoid -/
  shorterBase : ℝ
  /-- The longer base of the trapezoid -/
  longerBase : ℝ
  /-- The radius is positive -/
  radiusPos : radius > 0
  /-- The height ratio is between 0 and 1 -/
  heightRatioValid : 0 < heightRatio ∧ heightRatio < 1
  /-- The height is positive -/
  heightPos : height > 0
  /-- The bases are positive -/
  basesPos : shorterBase > 0 ∧ longerBase > 0
  /-- The longer base is longer than the shorter base -/
  basesOrder : shorterBase < longerBase
  /-- The center of the circle divides the height in the given ratio -/
  centerDivision : heightRatio = 4 / 7
  /-- The median is equal to the height -/
  medianEqualsHeight : (shorterBase + longerBase) / 2 = height

/-- The theorem stating the bases of the trapezoid given the conditions -/
theorem trapezoid_bases (t : IsoscelesTrapezoid) (h : t.radius = 10) :
  t.shorterBase = 12 ∧ t.longerBase = 16 := by
  sorry

end trapezoid_bases_l1650_165047


namespace probability_two_heads_two_tails_l1650_165098

theorem probability_two_heads_two_tails : 
  let n : ℕ := 4  -- total number of coins
  let k : ℕ := 2  -- number of heads (or tails) we want
  let p : ℚ := 1/2  -- probability of getting heads (or tails) on a single toss
  Nat.choose n k * p^n = 3/8 := by
  sorry

end probability_two_heads_two_tails_l1650_165098


namespace units_digit_of_fraction_l1650_165096

theorem units_digit_of_fraction (n : ℕ) : n = 30 * 31 * 32 * 33 * 34 * 35 / 7200 → n % 10 = 2 := by
  sorry

end units_digit_of_fraction_l1650_165096


namespace xy_sum_product_l1650_165053

theorem xy_sum_product (x y : ℝ) (h1 : x + y = 2 * Real.sqrt 3) (h2 : x * y = Real.sqrt 6) :
  x^2 * y + x * y^2 = 6 * Real.sqrt 2 := by
  sorry

end xy_sum_product_l1650_165053


namespace min_value_expression_min_value_achievable_l1650_165044

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + 1/y^2 + 1) * (x^2 + 1/y^2 - 1000) + (y^2 + 1/x^2 + 1) * (y^2 + 1/x^2 - 1000) ≥ -498998 :=
by sorry

theorem min_value_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧
  (x^2 + 1/y^2 + 1) * (x^2 + 1/y^2 - 1000) + (y^2 + 1/x^2 + 1) * (y^2 + 1/x^2 - 1000) = -498998 :=
by sorry

end min_value_expression_min_value_achievable_l1650_165044


namespace binomial_expansion_coefficient_l1650_165097

/-- 
Given that the coefficient of the third term in the binomial expansion 
of (x - 1/(2x))^n is 7, prove that n = 8.
-/
theorem binomial_expansion_coefficient (n : ℕ) : 
  (1/4 : ℚ) * (n.choose 2) = 7 → n = 8 := by
sorry

end binomial_expansion_coefficient_l1650_165097


namespace inscribed_circles_area_ratio_l1650_165017

theorem inscribed_circles_area_ratio (s : ℝ) (h : s > 0) : 
  let square_side := s
  let large_circle_radius := s / 2
  let triangle_side := s * (Real.sqrt 3) / 2
  let small_circle_radius := s * (Real.sqrt 3) / 12
  (π * (small_circle_radius ^ 2)) / (square_side ^ 2) = π / 48 :=
by sorry

end inscribed_circles_area_ratio_l1650_165017


namespace star_difference_l1650_165054

-- Define the ⋆ operation
def star (x y : ℤ) : ℤ := x * y - 3 * x + 1

-- Theorem statement
theorem star_difference : star 5 3 - star 3 5 = -6 := by sorry

end star_difference_l1650_165054


namespace min_value_expression_l1650_165085

theorem min_value_expression (a b c : ℕ+) :
  ∃ (x y z : ℕ+), 
    (⌊(8 * (x + y) : ℚ) / z⌋ + ⌊(8 * (x + z) : ℚ) / y⌋ + ⌊(8 * (y + z) : ℚ) / x⌋ = 46) ∧
    ∀ (a b c : ℕ+), 
      ⌊(8 * (a + b) : ℚ) / c⌋ + ⌊(8 * (a + c) : ℚ) / b⌋ + ⌊(8 * (b + c) : ℚ) / a⌋ ≥ 46 :=
by
  sorry

end min_value_expression_l1650_165085


namespace negation_of_existence_negation_of_quadratic_inequality_l1650_165036

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_inequality_l1650_165036


namespace principal_calculation_l1650_165035

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  (principal * rate * time) / 100

theorem principal_calculation (principal : ℚ) :
  simple_interest principal (4 : ℚ) (5 : ℚ) = principal - 2000 →
  principal = 2500 := by
  sorry

end principal_calculation_l1650_165035


namespace office_network_connections_l1650_165014

/-- Represents a network of switches with their connections -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- Calculates the total number of connections in the network -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- Theorem stating that a network of 30 switches, each connected to 4 others, has 60 connections -/
theorem office_network_connections :
  let network : SwitchNetwork := { num_switches := 30, connections_per_switch := 4 }
  total_connections network = 60 := by
  sorry

end office_network_connections_l1650_165014


namespace youngest_child_age_l1650_165060

/-- Represents the age of the youngest child in a group of 5 children -/
def youngest_age (total_age : ℕ) : ℕ :=
  (total_age - 20) / 5

/-- Theorem stating that if the sum of ages of 5 children born at 2-year intervals is 50,
    then the age of the youngest child is 6 years -/
theorem youngest_child_age :
  youngest_age 50 = 6 := by
  sorry

end youngest_child_age_l1650_165060


namespace sixty_degrees_is_hundred_clerts_l1650_165078

/-- Represents the number of clerts in a full circle in the Martian system -/
def full_circle_clerts : ℕ := 600

/-- Represents the number of degrees in a full circle in the Earth system -/
def full_circle_degrees : ℕ := 360

/-- Converts degrees to clerts -/
def degrees_to_clerts (degrees : ℕ) : ℚ :=
  (degrees : ℚ) * full_circle_clerts / full_circle_degrees

theorem sixty_degrees_is_hundred_clerts :
  degrees_to_clerts 60 = 100 := by sorry

end sixty_degrees_is_hundred_clerts_l1650_165078


namespace simplify_expression_l1650_165045

theorem simplify_expression : (81 * (10 ^ 12)) / (9 * (10 ^ 4)) = 900000000 := by
  sorry

end simplify_expression_l1650_165045


namespace divide_by_fraction_twelve_divided_by_one_fourth_l1650_165071

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_fourth : 12 / (1 / 4) = 48 := by sorry

end divide_by_fraction_twelve_divided_by_one_fourth_l1650_165071


namespace triangle_angle_calculation_l1650_165049

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    if a = 1, C = 60°, and c = √3, then A = π/6 -/
theorem triangle_angle_calculation (a b c A B C : Real) : 
  a = 1 → C = π / 3 → c = Real.sqrt 3 → A = π / 6 := by sorry

end triangle_angle_calculation_l1650_165049


namespace equation_solutions_l1650_165037

theorem equation_solutions :
  (∃ x : ℝ, 4 * x + 3 = 5 * x - 1 ∧ x = 4) ∧
  (∃ x : ℝ, 4 * (x - 1) = 1 - x ∧ x = 1) :=
by
  sorry

end equation_solutions_l1650_165037


namespace sequence_general_term_l1650_165088

/-- Given a sequence {a_n} with n ∈ ℕ, if S_n = 2a_n - 2^n + 1 represents
    the sum of the first n terms, then a_n = n × 2^(n-1) for all n ∈ ℕ. -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = 2 * a n - 2^n + 1) →
  ∀ n : ℕ, a n = n * 2^(n-1) := by
  sorry

end sequence_general_term_l1650_165088


namespace fifth_term_of_specific_arithmetic_sequence_l1650_165076

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifth_term_of_specific_arithmetic_sequence :
  let a₁ : ℝ := 1
  let d : ℝ := 2
  arithmeticSequence a₁ d 5 = 9 := by
sorry

end fifth_term_of_specific_arithmetic_sequence_l1650_165076


namespace taxi_ride_cost_l1650_165074

/-- Calculates the total cost of a taxi ride -/
def taxi_cost (base_fare : ℝ) (per_mile_rate : ℝ) (tax_rate : ℝ) (distance : ℝ) : ℝ :=
  let fare_without_tax := base_fare + per_mile_rate * distance
  let tax := tax_rate * fare_without_tax
  fare_without_tax + tax

/-- Theorem: The total cost of an 8-mile taxi ride is $4.84 -/
theorem taxi_ride_cost :
  taxi_cost 2.00 0.30 0.10 8 = 4.84 := by
  sorry

end taxi_ride_cost_l1650_165074


namespace regular_polygon_with_140_degree_interior_angles_l1650_165079

theorem regular_polygon_with_140_degree_interior_angles (n : ℕ) 
  (h_regular : n ≥ 3) 
  (h_interior_angle : (n - 2) * 180 / n = 140) : n = 9 := by
  sorry

end regular_polygon_with_140_degree_interior_angles_l1650_165079


namespace chenny_spoons_l1650_165041

/-- Given the following:
  * Chenny bought 9 plates at $2 each
  * Spoons cost $1.50 each
  * The total paid for plates and spoons is $24
  Prove that Chenny bought 4 spoons -/
theorem chenny_spoons (num_plates : ℕ) (price_plate : ℚ) (price_spoon : ℚ) (total_paid : ℚ) :
  num_plates = 9 →
  price_plate = 2 →
  price_spoon = 3/2 →
  total_paid = 24 →
  (total_paid - num_plates * price_plate) / price_spoon = 4 :=
by sorry

end chenny_spoons_l1650_165041


namespace banana_arrangements_l1650_165068

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  factorial total_letters / (factorial a_count * factorial n_count * factorial b_count) = 60 := by
  sorry

end banana_arrangements_l1650_165068


namespace truck_gas_calculation_l1650_165031

/-- Calculates the amount of gas already in a truck's tank given the truck's fuel efficiency, 
    distance to travel, and additional gas needed to complete the journey. -/
def gas_in_tank (miles_per_gallon : ℚ) (distance : ℚ) (additional_gas : ℚ) : ℚ :=
  distance / miles_per_gallon - additional_gas

/-- Theorem stating that for a truck traveling 3 miles per gallon, needing to cover 90 miles,
    and requiring 18 more gallons, the amount of gas already in the tank is 12 gallons. -/
theorem truck_gas_calculation :
  gas_in_tank 3 90 18 = 12 := by
  sorry

end truck_gas_calculation_l1650_165031


namespace right_triangle_perimeter_equals_area_l1650_165002

theorem right_triangle_perimeter_equals_area :
  ∀ a b c : ℕ,
    a ≤ b → b ≤ c →
    a^2 + b^2 = c^2 →
    a + b + c = (a * b) / 2 →
    ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 6 ∧ b = 8 ∧ c = 10)) :=
by sorry

end right_triangle_perimeter_equals_area_l1650_165002


namespace shaded_area_is_74_l1650_165024

/-- Represents a square with shaded and unshaded areas -/
structure ShadedSquare where
  side_length : ℝ
  unshaded_rectangles : ℕ
  unshaded_area : ℝ

/-- Calculates the area of the shaded part of the square -/
def shaded_area (s : ShadedSquare) : ℝ :=
  s.side_length ^ 2 - s.unshaded_area

/-- Theorem stating the area of the shaded part for the given conditions -/
theorem shaded_area_is_74 (s : ShadedSquare) 
    (h1 : s.side_length = 10)
    (h2 : s.unshaded_rectangles = 4)
    (h3 : s.unshaded_area = 26) : 
  shaded_area s = 74 := by
  sorry


end shaded_area_is_74_l1650_165024


namespace average_team_size_l1650_165070

theorem average_team_size (boys girls teams : ℕ) (h1 : boys = 83) (h2 : girls = 77) (h3 : teams = 4) :
  (boys + girls) / teams = 40 := by
  sorry

end average_team_size_l1650_165070


namespace probability_of_a_l1650_165058

theorem probability_of_a (p_a p_b : ℝ) (h_pb : p_b = 2/5)
  (h_independent : p_a * p_b = 0.22857142857142856) :
  p_a = 0.5714285714285714 := by
  sorry

end probability_of_a_l1650_165058


namespace permutation_combination_sum_l1650_165030

/-- Permutation of n elements taken r at a time -/
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then Nat.factorial n / Nat.factorial (n - r) else 0

/-- Combination of n elements taken r at a time -/
def combination (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r)) else 0

theorem permutation_combination_sum : 3 * (permutation 3 2) + 2 * (combination 4 2) = 30 := by
  sorry

end permutation_combination_sum_l1650_165030


namespace log_equation_solution_l1650_165005

theorem log_equation_solution (x : ℝ) :
  x > 1 →
  (Real.log (x^3 - 9*x + 8) / Real.log (x + 1)) * (Real.log (x + 1) / Real.log (x - 1)) = 3 →
  x = 3 :=
by sorry

end log_equation_solution_l1650_165005


namespace green_marble_fraction_l1650_165040

theorem green_marble_fraction (total : ℝ) (h1 : total > 0) : 
  let initial_green := (1/4) * total
  let initial_yellow := total - initial_green
  let new_green := 3 * initial_green
  let new_total := new_green + initial_yellow
  new_green / new_total = 1/2 := by sorry

end green_marble_fraction_l1650_165040


namespace composition_ratio_l1650_165019

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : f (g (f 3)) / g (f (g 3)) = 151 / 121 := by
  sorry

end composition_ratio_l1650_165019


namespace expression_simplification_l1650_165025

theorem expression_simplification (α : ℝ) : 
  (2 * Real.tan (π/4 - α)) / (1 - Real.tan (π/4 - α)^2) * 
  (Real.sin α * Real.cos α) / (Real.cos α^2 - Real.sin α^2) = 4 := by
sorry

end expression_simplification_l1650_165025


namespace area_of_inscribed_rectangle_l1650_165026

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.side * s.side

/-- The theorem stating the area of the inscribed rectangle R -/
theorem area_of_inscribed_rectangle (largerSquare : Square) 
  (smallerSquare : Square) 
  (rect1 rect2 : Rectangle) :
  smallerSquare.side = 2 ∧ 
  rect1.width = 2 ∧ rect1.height = 4 ∧
  rect2.width = 1 ∧ rect2.height = 2 ∧
  largerSquare.side = 6 →
  largerSquare.area - (smallerSquare.area + rect1.area + rect2.area) = 22 := by
  sorry

end area_of_inscribed_rectangle_l1650_165026


namespace jason_final_pears_l1650_165029

def initial_pears : ℕ := 46
def pears_given_to_keith : ℕ := 47
def pears_received_from_mike : ℕ := 12

theorem jason_final_pears :
  (if initial_pears ≥ pears_given_to_keith
   then initial_pears - pears_given_to_keith
   else 0) + pears_received_from_mike = 12 := by
  sorry

end jason_final_pears_l1650_165029


namespace mary_saw_36_snakes_l1650_165006

/-- The total number of snakes Mary saw -/
def total_snakes (breeding_balls : ℕ) (snakes_per_ball : ℕ) (additional_pairs : ℕ) : ℕ :=
  breeding_balls * snakes_per_ball + additional_pairs * 2

/-- Theorem stating that Mary saw 36 snakes in total -/
theorem mary_saw_36_snakes :
  total_snakes 3 8 6 = 36 := by
  sorry

end mary_saw_36_snakes_l1650_165006


namespace square_roots_problem_l1650_165056

theorem square_roots_problem (m : ℝ) (a : ℝ) (h1 : a > 0) 
  (h2 : (2 * m - 6)^2 = a) (h3 : (m + 3)^2 = a) (h4 : 2 * m - 6 ≠ m + 3) : m = 1 := by
  sorry

end square_roots_problem_l1650_165056
