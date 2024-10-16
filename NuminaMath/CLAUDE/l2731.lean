import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l2731_273148

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + 3*z = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2731_273148


namespace NUMINAMATH_CALUDE_set_relations_and_cardinality_l2731_273101

-- Define the cardinality function for finite sets
def card {α : Type*} (s : Set α) : ℕ := sorry

-- State the theorem
theorem set_relations_and_cardinality {α : Type*} (A B : Set α) :
  (A ∩ B = ∅ ↔ card (A ∪ B) = card A + card B) ∧
  (A ⊆ B → ¬(card A ≤ card B - 1)) ∧
  (card A ≤ card B → ¬(A ⊆ B)) ∧
  (card A = card B → ¬(A = B)) := by
  sorry


end NUMINAMATH_CALUDE_set_relations_and_cardinality_l2731_273101


namespace NUMINAMATH_CALUDE_curve_properties_l2731_273181

/-- The curve y = ax³ + bx passing through point (2,2) with tangent slope 9 -/
def Curve (a b : ℝ) : Prop :=
  2 * a * 8 + 2 * b = 2 ∧ 3 * a * 4 + b = 9

/-- The function f(x) = ax³ + bx -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x

theorem curve_properties :
  ∀ a b : ℝ, Curve a b →
  (a * b = -3) ∧
  (∀ x : ℝ, -3/2 ≤ x ∧ x ≤ 3 → -2 ≤ f a b x ∧ f a b x ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l2731_273181


namespace NUMINAMATH_CALUDE_parsley_decoration_problem_l2731_273197

/-- Calculates the number of plates decorated with one whole parsley sprig -/
def plates_with_whole_sprig (initial_sprigs : ℕ) (remaining_sprigs : ℕ) (half_sprig_plates : ℕ) : ℕ :=
  initial_sprigs - remaining_sprigs - (half_sprig_plates / 2)

/-- Proves that the number of plates decorated with one whole parsley sprig is 8 -/
theorem parsley_decoration_problem (initial_sprigs : ℕ) (remaining_sprigs : ℕ) (half_sprig_plates : ℕ)
  (h1 : initial_sprigs = 25)
  (h2 : remaining_sprigs = 11)
  (h3 : half_sprig_plates = 12) :
  plates_with_whole_sprig initial_sprigs remaining_sprigs half_sprig_plates = 8 := by
  sorry

#eval plates_with_whole_sprig 25 11 12

end NUMINAMATH_CALUDE_parsley_decoration_problem_l2731_273197


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l2731_273106

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 6)
  (h2 : new_price = 7.5) : 
  (1 - initial_price / new_price) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l2731_273106


namespace NUMINAMATH_CALUDE_taxi_driver_problem_l2731_273195

def distances : List Int := [8, -6, 3, -4, 8, -4, 4, -3]

def total_time : Rat := 4/3

theorem taxi_driver_problem (distances : List Int) (total_time : Rat) :
  (distances.sum = 6) ∧
  (((distances.map abs).sum : Rat) / total_time = 30) :=
by sorry

end NUMINAMATH_CALUDE_taxi_driver_problem_l2731_273195


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l2731_273125

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l2731_273125


namespace NUMINAMATH_CALUDE_apple_cost_l2731_273163

/-- The cost of an apple and an orange given two price combinations -/
theorem apple_cost (apple orange : ℝ) 
  (h1 : 6 * apple + 3 * orange = 1.77)
  (h2 : 2 * apple + 5 * orange = 1.27) :
  apple = 0.21 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_l2731_273163


namespace NUMINAMATH_CALUDE_quadratic_completion_l2731_273143

theorem quadratic_completion (c : ℝ) (n : ℝ) : 
  c < 0 → 
  (∀ x, x^2 + c*x + (1/4 : ℝ) = (x + n)^2 + (1/8 : ℝ)) → 
  c = -Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_l2731_273143


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l2731_273198

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l2731_273198


namespace NUMINAMATH_CALUDE_point_on_line_proof_l2731_273133

def point_on_line (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

theorem point_on_line_proof : point_on_line 2 1 10 5 14 7 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_proof_l2731_273133


namespace NUMINAMATH_CALUDE_cos_D_is_zero_l2731_273136

-- Define the triangle DEF
structure Triangle (DE EF : ℝ) where
  -- Ensure DE and EF are positive
  de_pos : DE > 0
  ef_pos : EF > 0

-- Define the right triangle DEF with given side lengths
def rightTriangleDEF : Triangle 9 40 where
  de_pos := by norm_num
  ef_pos := by norm_num

-- Theorem: In the right triangle DEF where angle D is 90°, cos D = 0
theorem cos_D_is_zero (t : Triangle 9 40) : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_D_is_zero_l2731_273136


namespace NUMINAMATH_CALUDE_birds_on_fence_l2731_273184

theorem birds_on_fence (initial_birds landing_birds : ℕ) :
  initial_birds = 12 →
  landing_birds = 8 →
  initial_birds + landing_birds = 20 := by
sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2731_273184


namespace NUMINAMATH_CALUDE_zero_in_interval_l2731_273116

theorem zero_in_interval (a b : ℝ) (ha : a > 1) (hb : 0 < b) (hb' : b < 1) :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ a^x + x - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2731_273116


namespace NUMINAMATH_CALUDE_green_chips_count_l2731_273158

theorem green_chips_count (total : ℕ) (blue_fraction : ℚ) (red : ℕ) : 
  total = 60 →
  blue_fraction = 1 / 6 →
  red = 34 →
  (total : ℚ) * blue_fraction + red + (total - (total : ℚ) * blue_fraction - red) = total →
  total - (total : ℚ) * blue_fraction - red = 16 :=
by sorry

end NUMINAMATH_CALUDE_green_chips_count_l2731_273158


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l2731_273151

theorem chocolate_milk_probability :
  let n : ℕ := 5  -- number of days
  let k : ℕ := 4  -- number of successful days
  let p : ℚ := 2/3  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 80/243 := by
sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l2731_273151


namespace NUMINAMATH_CALUDE_quadrilateral_area_theorem_l2731_273185

-- Define a structure for a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the signed area of a triangle
def signedArea (A B C : Point) : ℝ := sorry

-- Define the theorem
theorem quadrilateral_area_theorem 
  (A B C D O K L : Point) 
  (h1 : K.x = (A.x + C.x) / 2 ∧ K.y = (A.y + C.y) / 2)  -- K is midpoint of AC
  (h2 : L.x = (B.x + D.x) / 2 ∧ L.y = (B.y + D.y) / 2)  -- L is midpoint of BD
  : (signedArea A O B) + (signedArea C O D) - 
    ((signedArea B O C) - (signedArea D O A)) = 
    4 * (signedArea K O L) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_theorem_l2731_273185


namespace NUMINAMATH_CALUDE_rectangle_dg_length_l2731_273168

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

theorem rectangle_dg_length :
  ∀ (r1 r2 r3 : Rectangle),
  area r1 = area r2 ∧ area r2 = area r3 ∧   -- Equal areas
  r1.width = 23 ∧                           -- BC = 23
  r2.width = r1.height ∧                    -- DE = AB
  r3.width = r1.height - r2.height ∧        -- CE = AB - DE
  r3.height = r1.width →                    -- CH = BC
  r2.height = 552                           -- DG = 552
  := by sorry

end NUMINAMATH_CALUDE_rectangle_dg_length_l2731_273168


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_relation_l2731_273119

theorem rectangular_prism_volume_relation (c : ℝ) (hc : c > 0) :
  let a := (4 : ℝ)^(1/3) * c
  let b := (2 : ℝ)^(1/3) * c
  2 * c^3 = a * b * c := by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_relation_l2731_273119


namespace NUMINAMATH_CALUDE_all_points_collinear_l2731_273109

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  ∃ l : Line, p.onLine l ∧ q.onLine l ∧ r.onLine l

/-- The main theorem -/
theorem all_points_collinear (S : Set Point) (h_finite : Set.Finite S)
    (h_three_point : ∀ p q r : Point, p ∈ S → q ∈ S → r ∈ S → p ≠ q → 
      (∃ l : Line, p.onLine l ∧ q.onLine l) → r.onLine l) :
    ∀ p q r : Point, p ∈ S → q ∈ S → r ∈ S → collinear p q r :=
  sorry

end NUMINAMATH_CALUDE_all_points_collinear_l2731_273109


namespace NUMINAMATH_CALUDE_conditional_probability_l2731_273164

theorem conditional_probability (P_AB P_A : ℝ) (h1 : P_AB = 3/10) (h2 : P_A = 3/5) :
  P_AB / P_A = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_l2731_273164


namespace NUMINAMATH_CALUDE_satellite_altitude_scientific_notation_l2731_273108

/-- The altitude of a Beidou satellite in meters -/
def satellite_altitude : ℝ := 21500000

/-- Scientific notation representation of the satellite altitude -/
def scientific_notation : ℝ := 2.15 * (10 ^ 7)

/-- Theorem stating that the satellite altitude is equal to its scientific notation representation -/
theorem satellite_altitude_scientific_notation : 
  satellite_altitude = scientific_notation := by sorry

end NUMINAMATH_CALUDE_satellite_altitude_scientific_notation_l2731_273108


namespace NUMINAMATH_CALUDE_leading_coefficient_is_negative_fourteen_l2731_273103

def polynomial (x : ℝ) : ℝ := -5 * (x^5 - x^4 + 2*x) + 9 * (x^5 + 3) - 6 * (3*x^5 + x^3 + 2)

theorem leading_coefficient_is_negative_fourteen :
  ∃ (a : ℝ) (p : ℝ → ℝ), (∀ x, polynomial x = a * x^5 + p x) ∧ (∀ x, x ≠ 0 → |p x| / |x|^5 < 1) ∧ a = -14 :=
sorry

end NUMINAMATH_CALUDE_leading_coefficient_is_negative_fourteen_l2731_273103


namespace NUMINAMATH_CALUDE_manufacturing_department_percentage_l2731_273174

theorem manufacturing_department_percentage (total_degrees : ℝ) (manufacturing_degrees : ℝ) 
  (h1 : total_degrees = 360) 
  (h2 : manufacturing_degrees = 162) : 
  (manufacturing_degrees / total_degrees) * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_department_percentage_l2731_273174


namespace NUMINAMATH_CALUDE_trapezoid_area_sum_properties_l2731_273146

/-- Represents a trapezoid with four side lengths -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the sum of all possible areas of a trapezoid -/
def sum_of_areas (t : Trapezoid) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_square_prime (n : ℕ) : Prop := sorry

/-- Theorem stating the properties of the sum of areas for the given trapezoid -/
theorem trapezoid_area_sum_properties :
  ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
    let t := Trapezoid.mk 4 6 8 10
    sum_of_areas t = r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃ ∧
    not_divisible_by_square_prime n₁ ∧
    not_divisible_by_square_prime n₂ ∧
    ⌊r₁ + r₂ + r₃ + n₁ + n₂⌋ = 742 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_sum_properties_l2731_273146


namespace NUMINAMATH_CALUDE_cannot_form_triangle_l2731_273111

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that line segments of lengths 2, 4, and 6 cannot form a triangle -/
theorem cannot_form_triangle : ¬(can_form_triangle 2 4 6) := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_triangle_l2731_273111


namespace NUMINAMATH_CALUDE_digit_multiplication_l2731_273191

theorem digit_multiplication (A B : ℕ) : 
  A < 10 ∧ B < 10 ∧ A ≠ B ∧ A * (10 * A + B) = 100 * B + 11 * A → A = 8 ∧ B = 6 := by
  sorry

end NUMINAMATH_CALUDE_digit_multiplication_l2731_273191


namespace NUMINAMATH_CALUDE_power_four_inequality_l2731_273107

theorem power_four_inequality (a b : ℝ) : (a^4 + b^4) / 2 ≥ ((a + b) / 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_power_four_inequality_l2731_273107


namespace NUMINAMATH_CALUDE_parallel_line_plane_perpendicular_transitivity_l2731_273155

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Axioms for the properties of lines and planes
axiom different_lines : ∀ (m n : Line), m ≠ n
axiom different_planes : ∀ (α β γ : Plane), α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Theorem 1
theorem parallel_line_plane (m n : Line) (α : Plane) :
  parallel m n → parallelLP n α → (parallelLP m α ∨ subset m α) := by sorry

-- Theorem 2
theorem perpendicular_transitivity (m : Line) (α β γ : Plane) :
  parallelPP α β → parallelPP β γ → perpendicular m α → perpendicular m γ := by sorry

end NUMINAMATH_CALUDE_parallel_line_plane_perpendicular_transitivity_l2731_273155


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2731_273157

theorem polynomial_factorization (x : ℝ) : 
  x^5 + x^4 + 1 = (x^2 + x + 1) * (x^3 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2731_273157


namespace NUMINAMATH_CALUDE_repeating_decimal_proof_l2731_273187

/-- The repeating decimal 0.4̅67̅ as a rational number -/
def repeating_decimal : ℚ := 463 / 990

/-- Proof that 0.4̅67̅ is equal to 463/990 and is in lowest terms -/
theorem repeating_decimal_proof :
  repeating_decimal = 463 / 990 ∧
  (∀ n d : ℤ, n / d = 463 / 990 → d ≠ 0 → d.natAbs ≤ 990 → d = 990) :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_proof_l2731_273187


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2731_273128

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- A line l -/
structure Line where
  l : ℝ × ℝ → Prop

/-- The distance from a point to a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Determines if a line intersects a circle -/
def intersects (c : Circle) (l : Line) : Prop :=
  distancePointToLine c.O l < c.r

theorem line_circle_intersection (c : Circle) (l : Line) :
  distancePointToLine c.O l < c.r → intersects c l :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2731_273128


namespace NUMINAMATH_CALUDE_poodle_bark_count_l2731_273118

/-- The number of times the terrier's owner says "hush" -/
def hush_count : ℕ := 6

/-- The ratio of poodle barks to terrier barks -/
def poodle_terrier_ratio : ℕ := 2

/-- The number of barks in a poodle bark set -/
def poodle_bark_set : ℕ := 5

/-- The number of times the terrier barks -/
def terrier_barks : ℕ := hush_count * 2

/-- The number of times the poodle barks -/
def poodle_barks : ℕ := terrier_barks * poodle_terrier_ratio

theorem poodle_bark_count : poodle_barks = 24 := by
  sorry

end NUMINAMATH_CALUDE_poodle_bark_count_l2731_273118


namespace NUMINAMATH_CALUDE_problem_statement_l2731_273160

theorem problem_statement :
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 → x^3 + 1/x^3 - 3 = 15) ∧
  (∀ x a b c : ℝ, a = 1/20*x + 20 ∧ b = 1/20*x + 19 ∧ c = 1/20*x + 21 →
    a^2 + b^2 + c^2 - a*b - b*c - a*c = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2731_273160


namespace NUMINAMATH_CALUDE_det2_specific_values_det2_quadratic_relation_l2731_273169

def det2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem det2_specific_values :
  det2 5 6 7 8 = -2 :=
sorry

theorem det2_quadratic_relation (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  det2 (x + 1) (3*x) (x - 2) (x - 1) = 6*x + 1 :=
sorry

end NUMINAMATH_CALUDE_det2_specific_values_det2_quadratic_relation_l2731_273169


namespace NUMINAMATH_CALUDE_wireless_mice_ratio_l2731_273190

/-- Proves that the ratio of wireless mice to total mice sold is 1:2 -/
theorem wireless_mice_ratio (total_mice : ℕ) (optical_mice : ℕ) (trackball_mice : ℕ) :
  total_mice = 80 →
  optical_mice = total_mice / 4 →
  trackball_mice = 20 →
  let wireless_mice := total_mice - (optical_mice + trackball_mice)
  (wireless_mice : ℚ) / total_mice = 1 / 2 := by
  sorry

#check wireless_mice_ratio

end NUMINAMATH_CALUDE_wireless_mice_ratio_l2731_273190


namespace NUMINAMATH_CALUDE_equation_solution_set_l2731_273179

theorem equation_solution_set : 
  ∃ (S : Set ℝ), S = {x : ℝ | 16 * Real.sin (Real.pi * x) * Real.cos (Real.pi * x) = 16 * x + 1 / x} ∧ 
  S = {-(1/4), 1/4} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_set_l2731_273179


namespace NUMINAMATH_CALUDE_fraction_problem_l2731_273131

theorem fraction_problem (f : ℝ) : 
  (0.5 * 100 = f * 100 - 10) → f = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2731_273131


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2731_273141

open Real

def quadratic_inequality (k : ℝ) (x : ℝ) : Prop :=
  2 * k * x^2 + k * x - 3/8 < 0

theorem quadratic_inequality_range :
  ∀ k : ℝ, (∀ x : ℝ, quadratic_inequality k x) ↔ k ∈ Set.Ioo (-3/2) 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2731_273141


namespace NUMINAMATH_CALUDE_race_time_difference_l2731_273138

theorem race_time_difference (apple_rate mac_rate : ℝ) (race_distance : ℝ) : 
  apple_rate = 3 ∧ mac_rate = 4 ∧ race_distance = 24 → 
  (race_distance / apple_rate - race_distance / mac_rate) * 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l2731_273138


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l2731_273194

def arithmetic_sequence (a₁ a₂ a₃ : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

theorem tenth_term_of_sequence (h₁ : arithmetic_sequence (1/2) (5/6) (7/6) 2 = 5/6) 
                               (h₂ : arithmetic_sequence (1/2) (5/6) (7/6) 3 = 7/6) :
  arithmetic_sequence (1/2) (5/6) (7/6) 10 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l2731_273194


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l2731_273127

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_150_is_identity :
  B ^ 150 = 1 := by sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l2731_273127


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l2731_273171

/-- Given three real numbers a, b, and c satisfying certain conditions,
    prove that the average of a and b is 35. -/
theorem average_of_a_and_b (a b c : ℝ) 
    (h1 : (a + b) / 2 = 35)
    (h2 : (b + c) / 2 = 80)
    (h3 : c - a = 90) : 
  (a + b) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l2731_273171


namespace NUMINAMATH_CALUDE_parallel_line_through_point_line_equation_proof_l2731_273173

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space in the form ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (givenLine : Line2D) 
  (point : Point2D) 
  (resultLine : Line2D) : Prop :=
  parallelLines givenLine resultLine ∧ 
  pointOnLine point resultLine

/-- The main theorem to prove -/
theorem line_equation_proof : 
  let givenLine : Line2D := { a := 2, b := 3, c := 5 }
  let point : Point2D := { x := 1, y := -4 }
  let resultLine : Line2D := { a := 2, b := 3, c := 10 }
  parallel_line_through_point givenLine point resultLine := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_line_equation_proof_l2731_273173


namespace NUMINAMATH_CALUDE_some_frames_are_not_tars_l2731_273120

universe u

-- Define the types
variable (Tar Kite Rope Frame : Type u)

-- Define the relations
variable (is_tar : Tar → Prop)
variable (is_kite : Kite → Prop)
variable (is_rope : Rope → Prop)
variable (is_frame : Frame → Prop)

-- Hypotheses
variable (h1 : ∀ t : Tar, ∃ k : Kite, is_kite k)
variable (h2 : ∀ k : Kite, ∀ r : Rope, ¬(is_kite k ∧ is_rope r))
variable (h3 : ∃ r : Rope, ∃ f : Frame, is_rope r ∧ is_frame f)

-- Theorem to prove
theorem some_frames_are_not_tars :
  ∃ f : Frame, ¬∃ t : Tar, is_frame f ∧ is_tar t :=
sorry

end NUMINAMATH_CALUDE_some_frames_are_not_tars_l2731_273120


namespace NUMINAMATH_CALUDE_f_is_generalized_distance_l2731_273166

-- Define the binary function f
def f (x y : ℝ) : ℝ := x^2 + y^2

-- State the theorem
theorem f_is_generalized_distance :
  (∀ x y : ℝ, f x y ≥ 0 ∧ (f x y = 0 ↔ x = 0 ∧ y = 0)) ∧ 
  (∀ x y : ℝ, f x y = f y x) ∧
  (∀ x y z : ℝ, f x y ≤ f x z + f z y) :=
sorry

end NUMINAMATH_CALUDE_f_is_generalized_distance_l2731_273166


namespace NUMINAMATH_CALUDE_reading_speed_ratio_l2731_273199

/-- Given that Emery takes 20 days to read a book and the average number of days
    for Emery and Serena to read the book is 60, prove that the ratio of
    Emery's reading speed to Serena's reading speed is 5:1 -/
theorem reading_speed_ratio
  (emery_days : ℕ)
  (average_days : ℚ)
  (h_emery : emery_days = 20)
  (h_average : average_days = 60) :
  ∃ (emery_speed serena_speed : ℚ), 
    emery_speed / serena_speed = 5 / 1 :=
by sorry

end NUMINAMATH_CALUDE_reading_speed_ratio_l2731_273199


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_2023_l2731_273110

theorem tens_digit_of_13_pow_2023 : ∃ n : ℕ, 13^2023 ≡ 90 + n [ZMOD 100] ∧ n < 10 :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_2023_l2731_273110


namespace NUMINAMATH_CALUDE_min_keystrokes_for_2018_l2731_273192

/-- Represents the state of the screen and copy buffer -/
structure ScreenState where
  screen : ℕ  -- number of 'a's on screen
  buffer : ℕ  -- number of 'a's in copy buffer

/-- Represents the possible operations -/
inductive Operation
  | Copy
  | Paste

/-- Applies an operation to the screen state -/
def applyOperation (state : ScreenState) (op : Operation) : ScreenState :=
  match op with
  | Operation.Copy => { state with buffer := state.screen }
  | Operation.Paste => { state with screen := state.screen + state.buffer }

/-- Applies a sequence of operations to the initial screen state -/
def applyOperations (ops : List Operation) : ScreenState :=
  ops.foldl applyOperation { screen := 1, buffer := 0 }

/-- Checks if a sequence of operations achieves the goal -/
def achievesGoal (ops : List Operation) : Prop :=
  (applyOperations ops).screen ≥ 2018

theorem min_keystrokes_for_2018 :
  ∃ (ops : List Operation), achievesGoal ops ∧ ops.length = 21 ∧
  (∀ (other_ops : List Operation), achievesGoal other_ops → other_ops.length ≥ 21) :=
sorry

end NUMINAMATH_CALUDE_min_keystrokes_for_2018_l2731_273192


namespace NUMINAMATH_CALUDE_expression_value_l2731_273140

theorem expression_value : 
  |1 - Real.sqrt 3| - 2 * Real.sin (π / 3) + (π - 2023) ^ 0 = 0 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2731_273140


namespace NUMINAMATH_CALUDE_pool_capacity_theorem_l2731_273105

/-- Represents the dimensions of a pool -/
structure PoolDimensions where
  width : ℝ
  length : ℝ
  depth : ℝ

/-- Calculates the volume of a pool given its dimensions -/
def poolVolume (d : PoolDimensions) : ℝ := d.width * d.length * d.depth

/-- Represents the draining parameters of a pool -/
structure DrainParameters where
  rate : ℝ
  time : ℝ

/-- Calculates the amount of water drained given drain parameters -/
def waterDrained (p : DrainParameters) : ℝ := p.rate * p.time

/-- Theorem: The initial capacity of the pool was 80% of its total volume -/
theorem pool_capacity_theorem (d : PoolDimensions) (p : DrainParameters) :
  d.width = 60 ∧ d.length = 150 ∧ d.depth = 10 ∧
  p.rate = 60 ∧ p.time = 1200 →
  waterDrained p / poolVolume d = 0.8 := by
  sorry

#eval (80 : ℚ) / 100  -- Expected output: 4/5

end NUMINAMATH_CALUDE_pool_capacity_theorem_l2731_273105


namespace NUMINAMATH_CALUDE_problem_solution_l2731_273188

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -6)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2731_273188


namespace NUMINAMATH_CALUDE_circle_trajectory_and_max_area_l2731_273117

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 49
def F₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the property of point Q
def Q_property (x y : ℝ) : Prop := C x y ∧ y ≠ 0

-- Define the line MN parallel to OQ and passing through F₂
def MN_parallel_OQ (m : ℝ) (x y : ℝ) : Prop := x = m * y + 2

-- Define the distinct intersection points M and N
def distinct_intersections (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
  C x₁ y₁ ∧ C x₂ y₂ ∧
  MN_parallel_OQ m x₁ y₁ ∧ MN_parallel_OQ m x₂ y₂

-- Theorem statement
theorem circle_trajectory_and_max_area :
  (∀ x y, C x y → (∃ R, (∀ x' y', F₁ x' y' → (x - x')^2 + (y - y')^2 = (7 - R)^2) ∧
                      (∀ x' y', F₂ x' y' → (x - x')^2 + (y - y')^2 = (R - 1)^2))) ∧
  (∀ m, distinct_intersections m →
    ∃ x₃ y₃, Q_property x₃ y₃ ∧
    (∀ A, (∃ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ MN_parallel_OQ m x₁ y₁ ∧ MN_parallel_OQ m x₂ y₂ ∧
           A = (1/2) * abs ((x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁))) →
    A ≤ 10/3)) :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_and_max_area_l2731_273117


namespace NUMINAMATH_CALUDE_m_shaped_area_l2731_273152

/-- The area of the M-shaped region formed by folding a 12 × 18 rectangle along its diagonal -/
theorem m_shaped_area (width : ℝ) (height : ℝ) (diagonal : ℝ) (m_area : ℝ) : 
  width = 12 → 
  height = 18 → 
  diagonal = (width^2 + height^2).sqrt →
  m_area = 138 → 
  m_area = (width * height / 2) + 2 * (width * height / 2 - (13 / 36) * (width * height / 2)) :=
by sorry

end NUMINAMATH_CALUDE_m_shaped_area_l2731_273152


namespace NUMINAMATH_CALUDE_find_B_over_A_l2731_273123

-- Define the equation
def equation (A B x : ℝ) : Prop :=
  A / (x + 6) + B / (x^2 - 5*x) = (x^3 - 3*x^2 + 12) / (x^3 + x^2 - 30*x)

-- Theorem statement
theorem find_B_over_A (A B : ℤ) :
  (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 → equation A B x) →
  (B : ℝ) / (A : ℝ) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_find_B_over_A_l2731_273123


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2731_273170

theorem abs_neg_three_eq_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2731_273170


namespace NUMINAMATH_CALUDE_different_winning_scores_l2731_273135

def cross_country_meet (n : ℕ) : Prop :=
  n = 12 ∧ ∃ (team_size : ℕ), team_size = 6

def sum_of_positions (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def winning_score (total_score : ℕ) (score : ℕ) : Prop :=
  score ≤ total_score / 2

def min_winning_score (team_size : ℕ) : ℕ :=
  sum_of_positions team_size

theorem different_winning_scores (total_runners : ℕ) (team_size : ℕ) : 
  cross_country_meet total_runners →
  (winning_score (sum_of_positions total_runners) (sum_of_positions total_runners / 2) ∧
   min_winning_score team_size = sum_of_positions team_size) →
  (sum_of_positions total_runners / 2 - min_winning_score team_size + 1 = 19) :=
by sorry

end NUMINAMATH_CALUDE_different_winning_scores_l2731_273135


namespace NUMINAMATH_CALUDE_water_balloon_puddle_depth_l2731_273172

/-- The depth of water in a cylindrical puddle formed from a burst spherical water balloon -/
theorem water_balloon_puddle_depth (r_sphere r_cylinder : ℝ) (h : ℝ) : 
  r_sphere = 3 → 
  r_cylinder = 12 → 
  (4 / 3) * π * r_sphere^3 = π * r_cylinder^2 * h → 
  h = 1 / 4 := by
  sorry

#check water_balloon_puddle_depth

end NUMINAMATH_CALUDE_water_balloon_puddle_depth_l2731_273172


namespace NUMINAMATH_CALUDE_equidistant_line_proof_l2731_273182

-- Define the two given lines
def line1 (x y : ℝ) : ℝ := 3 * x + 2 * y - 6
def line2 (x y : ℝ) : ℝ := 6 * x + 4 * y - 3

-- Define the proposed equidistant line
def equidistant_line (x y : ℝ) : ℝ := 12 * x + 8 * y - 15

-- Theorem statement
theorem equidistant_line_proof :
  ∀ (x y : ℝ), |equidistant_line x y| = |line1 x y| ∧ |equidistant_line x y| = |line2 x y| :=
sorry

end NUMINAMATH_CALUDE_equidistant_line_proof_l2731_273182


namespace NUMINAMATH_CALUDE_hyperbola_symmetric_intersection_l2731_273139

/-- The hyperbola and its symmetric curve with respect to a line have common points for all real k -/
theorem hyperbola_symmetric_intersection (k : ℝ) : ∃ (x y : ℝ), 
  (x^2 - y^2 = 1) ∧ 
  (∃ (x' y' : ℝ), (x'^2 - y'^2 = 1) ∧ 
    ((x + x') / 2 = (y + y') / (2*k) - 1/k) ∧
    ((y + y') / 2 = k * ((x + x') / 2) - 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_symmetric_intersection_l2731_273139


namespace NUMINAMATH_CALUDE_may_greatest_drop_l2731_273112

/-- Represents the months of the year --/
inductive Month
| january
| february
| march
| april
| may
| june

/-- Price change for a given month --/
def price_change : Month → ℝ
| Month.january  => -1.00
| Month.february => 3.50
| Month.march    => -3.00
| Month.april    => 4.00
| Month.may      => -5.00
| Month.june     => 2.00

/-- Returns true if the price change is negative (a drop) --/
def is_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- The month with the greatest price drop --/
def greatest_drop : Month :=
  Month.may

theorem may_greatest_drop :
  ∀ m : Month, is_price_drop m → price_change greatest_drop ≤ price_change m :=
by sorry

end NUMINAMATH_CALUDE_may_greatest_drop_l2731_273112


namespace NUMINAMATH_CALUDE_even_number_divisibility_property_l2731_273114

theorem even_number_divisibility_property (n : ℕ) :
  n % 2 = 0 →
  (∀ p : ℕ, Prime p → p ∣ n → (p - 1) ∣ (n - 1)) →
  ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_even_number_divisibility_property_l2731_273114


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2731_273159

theorem isosceles_triangle_perimeter : 
  ∀ x : ℝ, 
  x^2 - 8*x + 15 = 0 → 
  x > 0 →
  2*x + 7 > x →
  2*x + 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2731_273159


namespace NUMINAMATH_CALUDE_plane_relations_theorem_l2731_273124

-- Define a type for planes
def Plane : Type := Unit

-- Define the relations between planes
def perpendicular (p q : Plane) : Prop := sorry
def parallel (p q : Plane) : Prop := sorry

-- Define a predicate for three non-collinear points on a plane being equidistant from another plane
def three_points_equidistant (p q : Plane) : Prop := sorry

-- The theorem to be proven
theorem plane_relations_theorem (α β γ : Plane) : 
  ¬((perpendicular α β ∧ perpendicular β γ → parallel α γ) ∨ 
    (three_points_equidistant α β → parallel α β)) := by sorry

end NUMINAMATH_CALUDE_plane_relations_theorem_l2731_273124


namespace NUMINAMATH_CALUDE_last_passenger_correct_seat_prob_l2731_273161

/-- Represents a bus with n seats and n passengers -/
structure Bus (n : ℕ) where
  seats : Fin n → Passenger
  tickets : Fin n → Seat

/-- Represents a passenger -/
inductive Passenger
| scientist
| regular (id : ℕ)

/-- Represents a seat -/
def Seat := ℕ

/-- The seating process for the bus -/
def seatingProcess (b : Bus n) : Bus n := sorry

/-- The probability that the last passenger sits in their assigned seat -/
def lastPassengerInCorrectSeat (b : Bus n) : ℚ := sorry

/-- Theorem stating that the probability of the last passenger sitting in their assigned seat is 1/2 -/
theorem last_passenger_correct_seat_prob (n : ℕ) (b : Bus n) :
  lastPassengerInCorrectSeat (seatingProcess b) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_last_passenger_correct_seat_prob_l2731_273161


namespace NUMINAMATH_CALUDE_factorization_proof_l2731_273137

theorem factorization_proof (x y : ℝ) : x^2 + y^2 + 2*x*y - 1 = (x + y + 1) * (x + y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2731_273137


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l2731_273167

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- M is defined as the square root of 36^49 * 49^36 -/
def M : ℕ := sorry

/-- Theorem stating that the sum of digits of M is 37 -/
theorem sum_of_digits_M : sum_of_digits M = 37 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l2731_273167


namespace NUMINAMATH_CALUDE_cloth_trimming_l2731_273153

theorem cloth_trimming (x : ℝ) :
  (x > 0) →
  (x - 4 > 0) →
  (x - 3 > 0) →
  ((x - 4) * (x - 3) = 120) →
  (x = 12) :=
by sorry

end NUMINAMATH_CALUDE_cloth_trimming_l2731_273153


namespace NUMINAMATH_CALUDE_range_of_a_l2731_273147

theorem range_of_a (e : ℝ) (h_e : e = Real.exp 1) :
  ∀ a : ℝ, (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + a * (y - 2 * e * x) * (Real.log y - Real.log x) = 0) ↔
  a < 0 ∨ a ≥ 2 / e := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2731_273147


namespace NUMINAMATH_CALUDE_right_triangle_altitude_l2731_273154

theorem right_triangle_altitude (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^2 + b^2 = c^2) (h5 : 1/a + 1/b = 3/c) :
  ∃ m_c : ℝ, m_c = c * (1 + Real.sqrt 10) / 9 ∧ m_c^2 * c = a * b := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_l2731_273154


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2731_273149

/-- The function f(x) = a^(x-1) + 2 passes through the point (1, 3) for any a > 0 and a ≠ 1 -/
theorem function_passes_through_point (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2731_273149


namespace NUMINAMATH_CALUDE_hiking_equipment_cost_l2731_273196

/-- Calculates the total cost of hiking equipment given specific prices and discounts -/
theorem hiking_equipment_cost
  (hoodie_price : ℝ)
  (flashlight_percentage : ℝ)
  (boots_original_price : ℝ)
  (boots_discount_percentage : ℝ)
  (h1 : hoodie_price = 80)
  (h2 : flashlight_percentage = 0.20)
  (h3 : boots_original_price = 110)
  (h4 : boots_discount_percentage = 0.10)
  : hoodie_price +
    (flashlight_percentage * hoodie_price) +
    (boots_original_price - (boots_discount_percentage * boots_original_price)) = 195 :=
by sorry

end NUMINAMATH_CALUDE_hiking_equipment_cost_l2731_273196


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_dne_l2731_273129

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan x * Real.sin (7 / x) else 0

theorem derivative_f_at_zero_dne :
  ¬ DifferentiableAt ℝ f 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_dne_l2731_273129


namespace NUMINAMATH_CALUDE_jackson_holidays_l2731_273134

/-- The number of holidays taken in a year given the number of days off per month and the number of months in a year -/
def holidays_in_year (days_off_per_month : ℕ) (months_in_year : ℕ) : ℕ :=
  days_off_per_month * months_in_year

/-- Theorem stating that taking 3 days off every month for 12 months results in 36 holidays in a year -/
theorem jackson_holidays :
  holidays_in_year 3 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jackson_holidays_l2731_273134


namespace NUMINAMATH_CALUDE_max_value_of_function_l2731_273113

theorem max_value_of_function (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  (2*x^2 - 2*x + 3) / (x^2 - x + 1) ≤ 10/3 ∧
  ∃ y : ℝ, (2*y^2 - 2*y + 3) / (y^2 - y + 1) = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2731_273113


namespace NUMINAMATH_CALUDE_polynomial_identity_l2731_273142

theorem polynomial_identity (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (x - 1)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅ + a₇)^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2731_273142


namespace NUMINAMATH_CALUDE_tom_lake_crossing_cost_l2731_273100

/-- The cost of hiring an assistant for crossing a lake back and forth -/
def lake_crossing_cost (one_way_time : ℕ) (hourly_rate : ℕ) : ℕ :=
  2 * one_way_time * hourly_rate

/-- Theorem: The cost for Tom to hire an assistant for crossing the lake back and forth is $80 -/
theorem tom_lake_crossing_cost :
  lake_crossing_cost 4 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_tom_lake_crossing_cost_l2731_273100


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l2731_273178

def num_questions : ℕ := 7
def num_positive : ℕ := 3
def prob_positive : ℚ := 3/7

theorem magic_8_ball_probability :
  (Nat.choose num_questions num_positive : ℚ) *
  (prob_positive ^ num_positive) *
  ((1 - prob_positive) ^ (num_questions - num_positive)) =
  242112/823543 := by sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l2731_273178


namespace NUMINAMATH_CALUDE_total_fireworks_count_l2731_273144

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of boxes Cherie has -/
def cherie_boxes : ℕ := 1

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box) +
  cherie_boxes * (cherie_sparklers + cherie_whistlers)

theorem total_fireworks_count : total_fireworks = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_fireworks_count_l2731_273144


namespace NUMINAMATH_CALUDE_tire_cost_l2731_273180

theorem tire_cost (window_cost tire_count total_cost : ℕ) 
  (h1 : window_cost = 700)
  (h2 : tire_count = 3)
  (h3 : total_cost = 1450)
  (h4 : tire_count * (total_cost - window_cost) / tire_count = 250) :
  ∃ (single_tire_cost : ℕ), 
    single_tire_cost * tire_count + window_cost = total_cost ∧ 
    single_tire_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_tire_cost_l2731_273180


namespace NUMINAMATH_CALUDE_wage_increase_percentage_l2731_273102

theorem wage_increase_percentage (W W' H H' : ℝ) : 
  W > 0 → H > 0 →
  W * H = W' * H' →  -- Total income remains unchanged
  H' = (2/3) * H →   -- Hours decreased by 1/3
  W' = (3/2) * W     -- Wage increased by 50%
  := by sorry

end NUMINAMATH_CALUDE_wage_increase_percentage_l2731_273102


namespace NUMINAMATH_CALUDE_unique_digit_solution_l2731_273132

theorem unique_digit_solution :
  ∃! (square boxplus boxtimes boxminus : ℕ),
    square < 10 ∧ boxplus < 10 ∧ boxtimes < 10 ∧ boxminus < 10 ∧
    square = 423 / 47 ∧
    1448 = 282 * boxminus + square * boxtimes ∧
    423 * boxplus = 282 * 3 ∧
    square = 9 ∧ boxplus = 2 ∧ boxtimes = 8 ∧ boxminus = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l2731_273132


namespace NUMINAMATH_CALUDE_cow_population_characteristics_l2731_273175

/-- Represents the number of cows in each category --/
structure CowPopulation where
  total : ℕ
  male : ℕ
  female : ℕ
  transgender : ℕ

/-- Represents the characteristics of cows in each category --/
structure CowCharacteristics where
  hornedMalePercentage : ℚ
  spottedFemalePercentage : ℚ
  uniquePatternTransgenderPercentage : ℚ

/-- Theorem stating the relation between spotted females and the sum of horned males and uniquely patterned transgender cows --/
theorem cow_population_characteristics 
  (pop : CowPopulation)
  (char : CowCharacteristics)
  (h1 : pop.total = 450)
  (h2 : pop.male = 3 * pop.female / 2)
  (h3 : pop.female = 2 * pop.transgender)
  (h4 : pop.total = pop.male + pop.female + pop.transgender)
  (h5 : char.hornedMalePercentage = 3/5)
  (h6 : char.spottedFemalePercentage = 1/2)
  (h7 : char.uniquePatternTransgenderPercentage = 7/10) :
  ↑(pop.female * 1) * char.spottedFemalePercentage = 
  ↑(pop.male * 1) * char.hornedMalePercentage + ↑(pop.transgender * 1) * char.uniquePatternTransgenderPercentage - 112 :=
sorry

end NUMINAMATH_CALUDE_cow_population_characteristics_l2731_273175


namespace NUMINAMATH_CALUDE_count_words_beginning_ending_with_A_l2731_273162

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering --/
def word_length : ℕ := 5

/-- The number of variable positions in the word --/
def variable_positions : ℕ := word_length - 2

/-- The number of five-letter words beginning and ending with 'A' --/
def words_beginning_ending_with_A : ℕ := alphabet_size ^ variable_positions

theorem count_words_beginning_ending_with_A :
  words_beginning_ending_with_A = 17576 :=
sorry

end NUMINAMATH_CALUDE_count_words_beginning_ending_with_A_l2731_273162


namespace NUMINAMATH_CALUDE_st_length_l2731_273156

/-- Triangle PQR with given side lengths and points S, T on its sides --/
structure TrianglePQR where
  /-- Side length PQ --/
  pq : ℝ
  /-- Side length PR --/
  pr : ℝ
  /-- Side length QR --/
  qr : ℝ
  /-- Point S on side PQ --/
  s : ℝ
  /-- Point T on side PR --/
  t : ℝ
  /-- PQ = 13 --/
  pq_eq : pq = 13
  /-- PR = 14 --/
  pr_eq : pr = 14
  /-- QR = 15 --/
  qr_eq : qr = 15
  /-- S is between P and Q --/
  s_between : 0 ≤ s ∧ s ≤ pq
  /-- T is between P and R --/
  t_between : 0 ≤ t ∧ t ≤ pr
  /-- ST is parallel to QR --/
  st_parallel_qr : (s / pq) = (t / pr)
  /-- ST contains the incenter of triangle PQR --/
  st_contains_incenter : ∃ (k : ℝ), 0 < k ∧ k < 1 ∧
    k * s / (1 - k) * (pq - s) = pr / (pr + qr) ∧
    k * t / (1 - k) * (pr - t) = pq / (pq + qr)

/-- The main theorem --/
theorem st_length (tri : TrianglePQR) : (tri.s * tri.pr + tri.t * tri.pq) / (tri.pq + tri.pr) = 135 / 14 := by
  sorry

end NUMINAMATH_CALUDE_st_length_l2731_273156


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2731_273145

theorem quadratic_inequality_condition (a : ℝ) :
  (a ≥ 0 → ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ∧
  (∃ a : ℝ, a < 0 ∧ ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2731_273145


namespace NUMINAMATH_CALUDE_max_stuck_guests_l2731_273193

/-- Represents a guest with their galosh size -/
structure Guest where
  size : Nat

/-- Represents the state of remaining guests and galoshes -/
structure State where
  guests : List Guest
  galoshes : List Nat

/-- Checks if a guest can wear a galosh -/
def canWear (g : Guest) (s : Nat) : Bool :=
  g.size ≤ s

/-- Defines a valid initial state with 10 guests and galoshes -/
def validInitialState (s : State) : Prop :=
  s.guests.length = 10 ∧ 
  s.galoshes.length = 10 ∧
  s.guests.map Guest.size = s.galoshes ∧
  s.galoshes.Nodup

/-- Defines a stuck state where no remaining guest can wear any remaining galosh -/
def isStuckState (s : State) : Prop :=
  ∀ g ∈ s.guests, ∀ sz ∈ s.galoshes, ¬ canWear g sz

/-- The main theorem stating the maximum number of guests that could be left -/
theorem max_stuck_guests (s : State) (h : validInitialState s) :
  ∀ s' : State, (∃ seq : List (Guest × Nat), s.guests.Sublist s'.guests ∧ 
                                             s.galoshes.Sublist s'.galoshes ∧
                                             isStuckState s') →
    s'.guests.length ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_stuck_guests_l2731_273193


namespace NUMINAMATH_CALUDE_cycle_loss_percentage_l2731_273126

/-- Given a cost price and selling price, calculate the percentage of loss -/
def percentageLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice * 100

theorem cycle_loss_percentage :
  let costPrice : ℚ := 2800
  let sellingPrice : ℚ := 2100
  percentageLoss costPrice sellingPrice = 25 := by
  sorry

end NUMINAMATH_CALUDE_cycle_loss_percentage_l2731_273126


namespace NUMINAMATH_CALUDE_prob_ice_skating_given_skiing_l2731_273122

/-- The probability that a randomly selected student likes ice skating -/
def P_ice_skating : ℝ := 0.6

/-- The probability that a randomly selected student likes skiing -/
def P_skiing : ℝ := 0.5

/-- The probability that a randomly selected student likes either ice skating or skiing -/
def P_ice_skating_or_skiing : ℝ := 0.7

/-- Theorem stating that the probability of a student liking ice skating given that they like skiing is 0.8 -/
theorem prob_ice_skating_given_skiing :
  (P_ice_skating + P_skiing - P_ice_skating_or_skiing) / P_skiing = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_prob_ice_skating_given_skiing_l2731_273122


namespace NUMINAMATH_CALUDE_remainder_problem_l2731_273121

theorem remainder_problem (n : ℤ) (h : n % 20 = 11) : (2 * n) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2731_273121


namespace NUMINAMATH_CALUDE_sum_possible_side_lengths_is_330_l2731_273115

/-- Represents a convex quadrilateral with specific properties -/
structure ConvexQuadrilateral where
  EF : ℝ
  angleE : ℝ
  sidesArithmeticProgression : Bool
  EFisMaxLength : Bool
  EFparallelGH : Bool

/-- Calculates the sum of all possible values for the length of one of the other sides -/
def sumPossibleSideLengths (q : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem stating the sum of all possible values for the length of one of the other sides is 330 -/
theorem sum_possible_side_lengths_is_330 (q : ConvexQuadrilateral) :
  q.EF = 24 ∧ q.angleE = 45 ∧ q.sidesArithmeticProgression ∧ q.EFisMaxLength ∧ q.EFparallelGH →
  sumPossibleSideLengths q = 330 :=
by sorry

end NUMINAMATH_CALUDE_sum_possible_side_lengths_is_330_l2731_273115


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2731_273176

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := (1 + 2*I) / (a - I)
  (∃ (b : ℝ), z = b*I) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2731_273176


namespace NUMINAMATH_CALUDE_fourth_power_sum_l2731_273183

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a^2 + b^2 + c^2 = 5) 
  (h3 : a^3 + b^3 + c^3 = 8) : 
  a^4 + b^4 + c^4 = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l2731_273183


namespace NUMINAMATH_CALUDE_sin_A_value_area_ABC_l2731_273130

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Conditions
  c = Real.sqrt 2 ∧
  a = 1 ∧
  Real.cos C = 3/4

-- Theorem for sin A
theorem sin_A_value (A B C : ℝ) (a b c : ℝ) 
  (h : triangle_ABC A B C a b c) : Real.sin A = Real.sqrt 14 / 8 := by
  sorry

-- Theorem for the area of triangle ABC
theorem area_ABC (A B C : ℝ) (a b c : ℝ) 
  (h : triangle_ABC A B C a b c) : (1/2) * a * b * Real.sin C = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_A_value_area_ABC_l2731_273130


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2731_273189

/-- An isosceles triangle with side lengths 6 and 7 has a perimeter of either 19 or 20 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  (a = 6 ∧ b = 7) ∨ (a = 7 ∧ b = 6) →  -- Given side lengths
  ((a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)) →  -- Isosceles condition
  a + b + c = 19 ∨ a + b + c = 20 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2731_273189


namespace NUMINAMATH_CALUDE_log_equation_solution_l2731_273186

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2731_273186


namespace NUMINAMATH_CALUDE_min_value_on_interval_l2731_273150

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem min_value_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 20) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y ∧ f a x = -7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l2731_273150


namespace NUMINAMATH_CALUDE_digit_multiplication_puzzle_l2731_273104

theorem digit_multiplication_puzzle :
  ∃! (A B C D E F : ℕ),
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧ F ≤ 9 ∧
    A * (10 * B + A) = 10 * C + D ∧
    F * (10 * B + E) = 10 * D + C ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F :=
by sorry

end NUMINAMATH_CALUDE_digit_multiplication_puzzle_l2731_273104


namespace NUMINAMATH_CALUDE_candies_given_away_l2731_273177

/-- Given a girl who initially had 60 candies and now has 20 left, 
    prove that she gave away 40 candies. -/
theorem candies_given_away (initial : ℕ) (remaining : ℕ) (given_away : ℕ) : 
  initial = 60 → remaining = 20 → given_away = initial - remaining → given_away = 40 := by
  sorry

end NUMINAMATH_CALUDE_candies_given_away_l2731_273177


namespace NUMINAMATH_CALUDE_conference_handshakes_l2731_273165

theorem conference_handshakes (n : ℕ) (m : ℕ) : 
  n = 15 →  -- number of married couples
  m = 3 →   -- number of men who don't shake hands with each other
  (2 * n * (2 * n - 1) - 2 * n) / 2 - (m * (m - 1)) / 2 = 417 :=
by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l2731_273165
