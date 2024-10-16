import Mathlib

namespace NUMINAMATH_CALUDE_unique_function_divisibility_l1829_182915

theorem unique_function_divisibility (k : ℕ) :
  ∃! f : ℕ → ℕ, ∀ m n : ℕ, (f m + f n) ∣ (m + n)^k :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_divisibility_l1829_182915


namespace NUMINAMATH_CALUDE_plot_perimeter_l1829_182993

/-- Proves that the perimeter of a rectangular plot is 300 meters given specific conditions -/
theorem plot_perimeter : 
  ∀ (width length perimeter : ℝ),
  length = width + 10 →
  1950 = (perimeter * 6.5) →
  perimeter = 2 * (length + width) →
  perimeter = 300 := by
sorry

end NUMINAMATH_CALUDE_plot_perimeter_l1829_182993


namespace NUMINAMATH_CALUDE_divisibility_condition_l1829_182934

theorem divisibility_condition (x : ℤ) : (x - 1) ∣ (x - 3) ↔ x ∈ ({-1, 0, 2, 3} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1829_182934


namespace NUMINAMATH_CALUDE_crease_lines_form_ellipse_l1829_182986

/-- Given a circle with radius R and an interior point A at distance a from the center,
    this theorem states that the set of points on all crease lines formed by folding
    the circle so that a point on the circumference coincides with A is described by
    the equation of an ellipse. -/
theorem crease_lines_form_ellipse (R a : ℝ) (h : 0 < a ∧ a < R) :
  ∀ x y : ℝ, (x - a / 2)^2 / (R / 2)^2 + y^2 / ((R / 2)^2 - (a / 2)^2) = 1 ↔ 
  (∃ A' : ℝ × ℝ, (A'.1^2 + A'.2^2 = R^2) ∧ 
   ((x - a)^2 + y^2 = (x - A'.1)^2 + (y - A'.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_crease_lines_form_ellipse_l1829_182986


namespace NUMINAMATH_CALUDE_product_repeating_decimal_three_and_eight_l1829_182983

/-- The product of 0.3̄ and 8 is equal to 8/3 -/
theorem product_repeating_decimal_three_and_eight :
  (∃ x : ℚ, x = 1/3 ∧ (∃ d : ℕ → ℕ, ∀ n, d n < 10 ∧ x = ∑' k, (d k : ℚ) / 10^(k+1)) ∧ x * 8 = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_three_and_eight_l1829_182983


namespace NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l1829_182950

theorem angle_measure_in_acute_triangle (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ A < Real.pi/2 →
  0 < B ∧ B < Real.pi/2 →
  0 < C ∧ C < Real.pi/2 →
  A + B + C = Real.pi →
  a = Real.sin B * (Real.sin C / Real.sin A) →
  b = Real.sin C * (Real.sin A / Real.sin B) →
  2 * a * Real.sin B = Real.sqrt 3 * b →
  A = Real.pi/3 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l1829_182950


namespace NUMINAMATH_CALUDE_largest_y_coordinate_l1829_182997

theorem largest_y_coordinate (x y : ℝ) :
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_l1829_182997


namespace NUMINAMATH_CALUDE_q_transformation_l1829_182929

theorem q_transformation (w d z z' : ℝ) (hw : w > 0) (hd : d > 0) (hz : z ≠ 0) :
  let q := 5 * w / (4 * d * z^2)
  let q' := 5 * (4 * w) / (4 * (2 * d) * z'^2)
  q' / q = 2/9 ↔ z' = 3 * Real.sqrt 2 * z := by
sorry

end NUMINAMATH_CALUDE_q_transformation_l1829_182929


namespace NUMINAMATH_CALUDE_no_perfect_power_in_sequence_l1829_182942

/-- Represents a triple in the sequence -/
structure Triple where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Generates the next triple from the current one -/
def nextTriple (t : Triple) : Triple :=
  { a := t.a * t.b,
    b := t.b * t.c,
    c := t.c * t.a }

/-- Checks if a number is a perfect power -/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ k m : ℕ, m ≥ 2 ∧ n = k^m

/-- The sequence of triples starting with (2,3,5) -/
def tripleSequence : ℕ → Triple
  | 0 => { a := 2, b := 3, c := 5 }
  | n + 1 => nextTriple (tripleSequence n)

/-- Theorem: No number in any triple of the sequence is a perfect power -/
theorem no_perfect_power_in_sequence :
  ∀ n : ℕ, ¬(isPerfectPower (tripleSequence n).a ∨
            isPerfectPower (tripleSequence n).b ∨
            isPerfectPower (tripleSequence n).c) :=
by
  sorry


end NUMINAMATH_CALUDE_no_perfect_power_in_sequence_l1829_182942


namespace NUMINAMATH_CALUDE_savings_fraction_proof_l1829_182995

def total_savings : ℕ := 180000
def ppf_savings : ℕ := 72000

theorem savings_fraction_proof :
  let nsc_savings : ℕ := total_savings - ppf_savings
  let fraction : ℚ := (1/3 : ℚ) * nsc_savings / ppf_savings
  fraction = (1/2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_savings_fraction_proof_l1829_182995


namespace NUMINAMATH_CALUDE_integral_inequality_l1829_182992

theorem integral_inequality (n : ℕ) (hn : n ≥ 2) :
  (1 : ℝ) / n < ∫ x in (0 : ℝ)..(π / 2), 1 / (1 + Real.cos x) ^ n ∧
  ∫ x in (0 : ℝ)..(π / 2), 1 / (1 + Real.cos x) ^ n < (n + 5 : ℝ) / (n * (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_integral_inequality_l1829_182992


namespace NUMINAMATH_CALUDE_exists_valid_layout_18_rectangles_l1829_182933

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a position on a 2D grid --/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents a layout of rectangles on a grid --/
def Layout := Position → Option Rectangle

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : Position) : Prop :=
  (p1.x = p2.x ∧ (p1.y + 1 = p2.y ∨ p2.y + 1 = p1.y)) ∨
  (p1.y = p2.y ∧ (p1.x + 1 = p2.x ∨ p2.x + 1 = p1.x))

/-- Checks if two rectangles form a larger rectangle when adjacent --/
def formsLargerRectangle (r1 r2 : Rectangle) : Prop :=
  r1.width = r2.width ∨ r1.height = r2.height

/-- Checks if a layout satisfies the non-adjacency condition --/
def validLayout (l : Layout) : Prop :=
  ∀ p1 p2, adjacent p1 p2 →
    match l p1, l p2 with
    | some r1, some r2 => ¬formsLargerRectangle r1 r2
    | _, _ => True

/-- The main theorem: there exists a valid layout with 18 rectangles --/
theorem exists_valid_layout_18_rectangles :
  ∃ (l : Layout) (r : Rectangle),
    validLayout l ∧
    (∃ (positions : Finset Position), positions.card = 18 ∧
      ∀ p, p ∈ positions ↔ ∃ (smallR : Rectangle), l p = some smallR) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_layout_18_rectangles_l1829_182933


namespace NUMINAMATH_CALUDE_green_lab_coat_pairs_l1829_182902

theorem green_lab_coat_pairs 
  (total_students : ℕ) 
  (white_coat_students : ℕ) 
  (green_coat_students : ℕ) 
  (total_pairs : ℕ) 
  (white_white_pairs : ℕ) 
  (h1 : total_students = 142)
  (h2 : white_coat_students = 68)
  (h3 : green_coat_students = 74)
  (h4 : total_pairs = 71)
  (h5 : white_white_pairs = 29)
  (h6 : total_students = white_coat_students + green_coat_students)
  (h7 : total_students = 2 * total_pairs) :
  ∃ (green_green_pairs : ℕ), green_green_pairs = 32 ∧ 
    green_green_pairs + white_white_pairs + (white_coat_students - 2 * white_white_pairs) = total_pairs :=
by
  sorry

end NUMINAMATH_CALUDE_green_lab_coat_pairs_l1829_182902


namespace NUMINAMATH_CALUDE_sophia_age_in_eight_years_l1829_182900

/-- Represents the ages of individuals in the problem -/
structure Ages where
  jeremy : ℕ
  sebastian : ℕ
  isabella : ℕ
  sophia : ℕ
  lucas : ℕ
  olivia : ℕ
  ethan : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.jeremy + ages.sebastian + ages.isabella + ages.sophia + ages.lucas + ages.olivia + ages.ethan + 42 = 495) ∧
  (ages.sebastian = ages.jeremy + 4) ∧
  (ages.isabella = ages.sebastian - 3) ∧
  (ages.sophia = 2 * ages.lucas) ∧
  (ages.lucas = ages.jeremy - 5) ∧
  (ages.olivia = ages.isabella) ∧
  (ages.ethan = ages.olivia / 2) ∧
  (ages.jeremy + ages.sebastian + ages.isabella + 6 = 150) ∧
  (ages.jeremy = 40)

/-- The theorem to be proved -/
theorem sophia_age_in_eight_years (ages : Ages) :
  problem_conditions ages → ages.sophia + 8 = 78 := by
  sorry


end NUMINAMATH_CALUDE_sophia_age_in_eight_years_l1829_182900


namespace NUMINAMATH_CALUDE_shekar_average_marks_l1829_182938

def shekar_marks : List ℕ := [76, 65, 82, 67, 85]

theorem shekar_average_marks :
  (shekar_marks.sum / shekar_marks.length : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l1829_182938


namespace NUMINAMATH_CALUDE_decagon_triangles_l1829_182914

theorem decagon_triangles : 
  let n : ℕ := 10  -- number of vertices in a regular decagon
  let k : ℕ := 3   -- number of vertices needed to form a triangle
  Nat.choose n k = 120 := by
sorry

end NUMINAMATH_CALUDE_decagon_triangles_l1829_182914


namespace NUMINAMATH_CALUDE_nancy_carrots_l1829_182932

/-- The number of carrots Nancy picked the next day -/
def carrots_picked_next_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : ℕ :=
  total_carrots - (initial_carrots - thrown_out)

/-- Proof that Nancy picked 21 carrots the next day -/
theorem nancy_carrots :
  carrots_picked_next_day 12 2 31 = 21 := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrots_l1829_182932


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_one_l1829_182963

theorem negation_of_forall_geq_one :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_one_l1829_182963


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_17_l1829_182989

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_mod_17 :
  arithmetic_sequence_sum 4 6 100 % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_17_l1829_182989


namespace NUMINAMATH_CALUDE_sector_angle_when_arc_equals_radius_l1829_182901

theorem sector_angle_when_arc_equals_radius (r : ℝ) (θ : ℝ) :
  r > 0 → r * θ = r → θ = 1 := by sorry

end NUMINAMATH_CALUDE_sector_angle_when_arc_equals_radius_l1829_182901


namespace NUMINAMATH_CALUDE_scientific_notation_of_given_number_l1829_182948

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The given number -/
def givenNumber : ℝ := 0.0000046

/-- Theorem: The scientific notation of 0.0000046 is 4.6 × 10^(-6) -/
theorem scientific_notation_of_given_number :
  toScientificNotation givenNumber = ScientificNotation.mk 4.6 (-6) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_given_number_l1829_182948


namespace NUMINAMATH_CALUDE_square_of_complex_number_l1829_182988

theorem square_of_complex_number (z : ℂ) (i : ℂ) :
  z = 5 + 2 * i →
  i^2 = -1 →
  z^2 = 21 + 20 * i :=
by sorry

end NUMINAMATH_CALUDE_square_of_complex_number_l1829_182988


namespace NUMINAMATH_CALUDE_complement_intersection_equality_l1829_182973

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {2, 4}
def N : Set Nat := {3, 5}

theorem complement_intersection_equality :
  (U \ M) ∩ N = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equality_l1829_182973


namespace NUMINAMATH_CALUDE_gcd_45_75_l1829_182924

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_75_l1829_182924


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1829_182949

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →
  y = 10 →
  (x + y + z) / 3 = x + 20 →
  (x + y + z) / 3 = z - 25 →
  x + y + z = 45 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1829_182949


namespace NUMINAMATH_CALUDE_cube_volume_increase_l1829_182953

theorem cube_volume_increase (s : ℝ) (h : s > 0) : 
  let new_edge := 1.6 * s
  let original_volume := s^3
  let new_volume := new_edge^3
  (new_volume - original_volume) / original_volume * 100 = 309.6 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_increase_l1829_182953


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_implies_m_greater_than_three_l1829_182926

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The point P with coordinates (3-m, m-1) -/
def P (m : ℝ) : ℝ × ℝ := (3 - m, m - 1)

/-- If point P(3-m, m-1) is in the second quadrant, then m > 3 -/
theorem point_in_second_quadrant_implies_m_greater_than_three (m : ℝ) :
  second_quadrant (P m).1 (P m).2 → m > 3 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_implies_m_greater_than_three_l1829_182926


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l1829_182917

/-- Given points A(-2,a), B(-1,b), and C(3,c) on the graph of y = 4/x, prove b < a < c -/
theorem inverse_proportion_order (a b c : ℝ) : 
  ((-2 : ℝ) * a = 4) → ((-1 : ℝ) * b = 4) → ((3 : ℝ) * c = 4) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l1829_182917


namespace NUMINAMATH_CALUDE_jills_herd_sale_fraction_l1829_182903

/-- Represents the number of llamas in Jill's herd -/
structure LlamaHerd where
  initial : ℕ
  single_births : ℕ
  twin_births : ℕ
  traded_calves : ℕ
  traded_adults : ℕ
  final : ℕ

/-- Calculates the fraction of the herd sold at the market -/
def fraction_sold (herd : LlamaHerd) : ℚ :=
  let total_calves := herd.single_births + 2 * herd.twin_births
  let before_trade := herd.initial + total_calves
  let after_trade := before_trade - herd.traded_calves + herd.traded_adults
  let sold := after_trade - herd.final
  sold / before_trade

/-- Theorem stating the fraction of the herd Jill sold at the market -/
theorem jills_herd_sale_fraction : 
  ∀ (herd : LlamaHerd), 
  herd.single_births = 9 → 
  herd.twin_births = 5 → 
  herd.traded_calves = 8 → 
  herd.traded_adults = 2 → 
  herd.final = 18 → 
  fraction_sold herd = 4 / 13 := by
  sorry


end NUMINAMATH_CALUDE_jills_herd_sale_fraction_l1829_182903


namespace NUMINAMATH_CALUDE_distance_between_red_lights_l1829_182970

/-- Represents the color of a light -/
inductive LightColor
| Red
| Green

/-- Calculates the position of the nth red light in the sequence -/
def redLightPosition (n : ℕ) : ℕ :=
  (n - 1) / 3 * 7 + (n - 1) % 3 + 1

/-- The distance between lights in inches -/
def light_spacing : ℕ := 8

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The main theorem stating the distance between the 4th and 19th red lights -/
theorem distance_between_red_lights :
  (redLightPosition 19 - redLightPosition 4) * light_spacing / inches_per_foot = 
    (22671 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_distance_between_red_lights_l1829_182970


namespace NUMINAMATH_CALUDE_prism_properties_l1829_182985

/-- A right triangular prism with rectangular base ABCD and height DE -/
structure RightTriangularPrism where
  AB : ℝ
  BC : ℝ
  DE : ℝ
  ab_eq_dc : AB = 8
  bc_eq : BC = 15
  de_eq : DE = 7

/-- The perimeter of the base ABCD -/
def basePerimeter (p : RightTriangularPrism) : ℝ :=
  2 * (p.AB + p.BC)

/-- The area of the base ABCD -/
def baseArea (p : RightTriangularPrism) : ℝ :=
  p.AB * p.BC

/-- The volume of the right triangular prism -/
def volume (p : RightTriangularPrism) : ℝ :=
  p.AB * p.BC * p.DE

theorem prism_properties (p : RightTriangularPrism) :
  basePerimeter p = 46 ∧ baseArea p = 120 ∧ volume p = 840 := by
  sorry

end NUMINAMATH_CALUDE_prism_properties_l1829_182985


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1829_182972

theorem min_reciprocal_sum (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_sum : x + y + z = 2) :
  (1 / x + 1 / y + 1 / z) ≥ 4.5 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 ∧ 1 / x + 1 / y + 1 / z = 4.5 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1829_182972


namespace NUMINAMATH_CALUDE_arrange_crosses_and_zeros_theorem_l1829_182919

def arrange_crosses_and_zeros (n : ℕ) : ℕ :=
  if n = 27 then 14
  else if n = 26 then 105
  else 0

theorem arrange_crosses_and_zeros_theorem :
  (arrange_crosses_and_zeros 27 = 14) ∧
  (arrange_crosses_and_zeros 26 = 105) :=
sorry

end NUMINAMATH_CALUDE_arrange_crosses_and_zeros_theorem_l1829_182919


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1829_182999

theorem compound_interest_problem (P : ℝ) (t : ℝ) : 
  P * (1 + 0.1)^t = 2420 → 
  P * (1 + 0.1)^(t+3) = 2662 → 
  t = 3 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l1829_182999


namespace NUMINAMATH_CALUDE_prop_a_prop_b_prop_d_l1829_182930

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Proposition A
theorem prop_a (t : Triangle) (h : t.A > t.B) : Real.sin t.A > Real.sin t.B := by sorry

-- Proposition B
theorem prop_b (t : Triangle) (h : t.A < π/2 ∧ t.B < π/2 ∧ t.C < π/2) : Real.sin t.A > Real.cos t.B := by sorry

-- Proposition D
theorem prop_d (t : Triangle) (h1 : t.B = π/3) (h2 : t.b^2 = t.a * t.c) : t.A = π/3 ∧ t.B = π/3 ∧ t.C = π/3 := by sorry

end NUMINAMATH_CALUDE_prop_a_prop_b_prop_d_l1829_182930


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1829_182941

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum (1/4) (1/4) 6 = 4095/12288 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1829_182941


namespace NUMINAMATH_CALUDE_top_coat_drying_time_l1829_182935

/-- Given nail polish drying times, prove the top coat drying time -/
theorem top_coat_drying_time 
  (base_coat_time : ℕ) 
  (color_coat_time : ℕ) 
  (num_color_coats : ℕ) 
  (total_drying_time : ℕ) 
  (h1 : base_coat_time = 2)
  (h2 : color_coat_time = 3)
  (h3 : num_color_coats = 2)
  (h4 : total_drying_time = 13) :
  total_drying_time - (base_coat_time + num_color_coats * color_coat_time) = 5 := by
  sorry

end NUMINAMATH_CALUDE_top_coat_drying_time_l1829_182935


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1829_182916

theorem min_value_sqrt_sum_squares (m n : ℝ) (h : ∃ α : ℝ, m * Real.sin α + n * Real.cos α = 5) :
  (∀ x y : ℝ, (∃ β : ℝ, x * Real.sin β + y * Real.cos β = 5) → Real.sqrt (x^2 + y^2) ≥ 5) ∧
  (∃ p q : ℝ, (∃ γ : ℝ, p * Real.sin γ + q * Real.cos γ = 5) ∧ Real.sqrt (p^2 + q^2) = 5) :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1829_182916


namespace NUMINAMATH_CALUDE_k_range_for_empty_intersection_l1829_182955

-- Define the sets A and B
def A (k : ℝ) : Set ℝ := {x | k * x^2 - (k + 3) * x - 1 ≥ 0}
def B : Set ℝ := {y | ∃ x, y = 2 * x + 1}

-- State the theorem
theorem k_range_for_empty_intersection :
  (∀ k : ℝ, (A k ∩ B = ∅)) ↔ (∀ k : ℝ, -9 < k ∧ k < -1) :=
sorry

end NUMINAMATH_CALUDE_k_range_for_empty_intersection_l1829_182955


namespace NUMINAMATH_CALUDE_function_max_min_difference_l1829_182956

theorem function_max_min_difference (a : ℝ) (h1 : a > 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x-1)
  (∀ x ∈ Set.Icc 2 3, f x ≤ f 3) ∧
  (∀ x ∈ Set.Icc 2 3, f 2 ≤ f x) ∧
  (f 3 - f 2 = a / 2) →
  a = 3/2 := by sorry

end NUMINAMATH_CALUDE_function_max_min_difference_l1829_182956


namespace NUMINAMATH_CALUDE_john_weight_lifting_l1829_182946

/-- John's weight lifting problem -/
theorem john_weight_lifting 
  (weight_per_rep : ℕ) 
  (reps_per_set : ℕ) 
  (num_sets : ℕ) 
  (h1 : weight_per_rep = 15)
  (h2 : reps_per_set = 10)
  (h3 : num_sets = 3) :
  weight_per_rep * reps_per_set * num_sets = 450 := by
  sorry

#check john_weight_lifting

end NUMINAMATH_CALUDE_john_weight_lifting_l1829_182946


namespace NUMINAMATH_CALUDE_average_xyz_in_terms_of_k_l1829_182987

theorem average_xyz_in_terms_of_k (x y z k : ℝ) 
  (eq1 : 2 * x + y - z = 26)
  (eq2 : x + 2 * y + z = 10)
  (eq3 : x - y + z = k) :
  (x + y + z) / 3 = (36 + k) / 6 := by
  sorry

end NUMINAMATH_CALUDE_average_xyz_in_terms_of_k_l1829_182987


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1829_182962

def z : ℂ := (2 + Complex.I) * Complex.I

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1829_182962


namespace NUMINAMATH_CALUDE_pillsbury_sugar_needed_l1829_182908

/-- Chef Pillsbury's recipe ratios -/
structure RecipeRatios where
  eggs_to_flour : ℚ
  milk_to_eggs : ℚ
  sugar_to_milk : ℚ

/-- Calculate the number of tablespoons of sugar needed for a given amount of flour -/
def sugar_needed (ratios : RecipeRatios) (flour_cups : ℚ) : ℚ :=
  let eggs := flour_cups * ratios.eggs_to_flour
  let milk := eggs * ratios.milk_to_eggs
  milk * ratios.sugar_to_milk

/-- Theorem: For 24 cups of flour, Chef Pillsbury needs 90 tablespoons of sugar -/
theorem pillsbury_sugar_needed :
  let ratios : RecipeRatios := {
    eggs_to_flour := 7 / 2,
    milk_to_eggs := 5 / 14,
    sugar_to_milk := 3 / 1
  }
  sugar_needed ratios 24 = 90 := by
  sorry

end NUMINAMATH_CALUDE_pillsbury_sugar_needed_l1829_182908


namespace NUMINAMATH_CALUDE_flower_problem_solution_l1829_182925

/-- Given initial flowers and minimum flowers per bouquet, 
    calculate additional flowers needed and number of bouquets -/
def flower_arrangement (initial_flowers : ℕ) (min_per_bouquet : ℕ) : 
  {additional_flowers : ℕ // ∃ (num_bouquets : ℕ), 
    num_bouquets * min_per_bouquet = initial_flowers + additional_flowers ∧
    num_bouquets * min_per_bouquet > initial_flowers ∧
    ∀ (k : ℕ), k * min_per_bouquet > initial_flowers → 
      k * min_per_bouquet ≥ num_bouquets * min_per_bouquet} :=
sorry

theorem flower_problem_solution : 
  (flower_arrangement 1273 89).val = 62 ∧ 
  ∃ (num_bouquets : ℕ), num_bouquets = 15 ∧
    num_bouquets * 89 = 1273 + (flower_arrangement 1273 89).val :=
sorry

end NUMINAMATH_CALUDE_flower_problem_solution_l1829_182925


namespace NUMINAMATH_CALUDE_truck_meeting_distance_difference_l1829_182991

theorem truck_meeting_distance_difference 
  (initial_distance : ℝ) 
  (speed_a : ℝ) 
  (speed_b : ℝ) 
  (head_start : ℝ) :
  initial_distance = 855 →
  speed_a = 90 →
  speed_b = 80 →
  head_start = 1 →
  let relative_speed := speed_a + speed_b
  let meeting_time := (initial_distance - speed_a * head_start) / relative_speed
  let distance_a := speed_a * (meeting_time + head_start)
  let distance_b := speed_b * meeting_time
  distance_a - distance_b = 135 := by sorry

end NUMINAMATH_CALUDE_truck_meeting_distance_difference_l1829_182991


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1829_182954

-- Define the quadratic function
def f (a c x : ℝ) := a * x^2 + x + c

-- Define the solution set condition
def solution_set (a c : ℝ) : Set ℝ := {x | 1 < x ∧ x < 3}

-- Define the theorem
theorem quadratic_inequality_theorem (a c : ℝ) 
  (h : ∀ x, f a c x > 0 ↔ x ∈ solution_set a c) :
  a = -1/4 ∧ c = -3/4 ∧ 
  ∀ m : ℝ, (∀ x, -1/4 * x^2 + 2*x - 3 > 0 → x + m > 0) → m ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1829_182954


namespace NUMINAMATH_CALUDE_lowest_hundred_year_flood_level_l1829_182927

/-- Represents the frequency distribution of water levels -/
structure WaterLevelDistribution where
  -- Add necessary fields to represent the distribution
  -- This is a simplified representation
  lowest_hundred_year_flood : ℝ

/-- The hydrological observation point data -/
def observation_point : WaterLevelDistribution :=
  { lowest_hundred_year_flood := 50 }

/-- Theorem stating the lowest water level of the hundred-year flood -/
theorem lowest_hundred_year_flood_level :
  observation_point.lowest_hundred_year_flood = 50 := by
  sorry

#check lowest_hundred_year_flood_level

end NUMINAMATH_CALUDE_lowest_hundred_year_flood_level_l1829_182927


namespace NUMINAMATH_CALUDE_solution_count_l1829_182909

def is_solution (a b : ℕ+) : Prop :=
  (1 : ℚ) / a.val - (1 : ℚ) / b.val = (1 : ℚ) / 2018

theorem solution_count :
  ∃! (s : Finset (ℕ+ × ℕ+)), 
    (∀ (p : ℕ+ × ℕ+), p ∈ s ↔ is_solution p.1 p.2) ∧ 
    s.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_solution_count_l1829_182909


namespace NUMINAMATH_CALUDE_robin_candy_problem_l1829_182921

theorem robin_candy_problem (initial_candy : ℕ) (sister_candy : ℕ) (final_candy : ℕ) 
  (h1 : initial_candy = 23)
  (h2 : sister_candy = 21)
  (h3 : final_candy = 37) :
  initial_candy - (final_candy - sister_candy) = 7 := by
  sorry

end NUMINAMATH_CALUDE_robin_candy_problem_l1829_182921


namespace NUMINAMATH_CALUDE_joan_pinball_spending_l1829_182905

def half_dollar_value : ℚ := 0.5

theorem joan_pinball_spending (wednesday_spent : ℕ) (total_spent : ℚ) 
  (h1 : wednesday_spent = 4)
  (h2 : total_spent = 9)
  : ℕ := by
  sorry

#check joan_pinball_spending

end NUMINAMATH_CALUDE_joan_pinball_spending_l1829_182905


namespace NUMINAMATH_CALUDE_largest_n_value_l1829_182994

/-- The largest possible value of n for regular polygons Q1 (m-gon) and Q2 (n-gon) 
    satisfying the given conditions -/
theorem largest_n_value (m n : ℕ) : m ≥ n → n ≥ 3 → 
  (m - 2) * n = (n - 2) * m * 8 / 7 → 
  (∀ k, k > n → (k - 2) * m ≠ (m - 2) * k * 8 / 7) →
  n = 112 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_value_l1829_182994


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_absolute_value_location_l1829_182980

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

theorem absolute_value_location (a : ℝ) : |a| = 0 ∨ |a| > 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_absolute_value_location_l1829_182980


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1829_182947

theorem simple_interest_problem (r : ℝ) (n : ℝ) :
  (400 * r * n) / 100 + 200 = (400 * (r + 5) * n) / 100 →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1829_182947


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1829_182967

theorem smallest_positive_solution :
  ∀ x : ℝ, x > 0 ∧ Real.sqrt x = 9 * x^2 → x ≥ 1/81 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1829_182967


namespace NUMINAMATH_CALUDE_miser_knight_theorem_l1829_182937

theorem miser_knight_theorem (n : ℕ) (h : n > 0) :
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 76 → ∃ m : ℕ, n = m * k) →
  ∃ m : ℕ, n = m * 77 :=
by sorry

end NUMINAMATH_CALUDE_miser_knight_theorem_l1829_182937


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1829_182977

def f (a x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = 2) →
  a = -1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1829_182977


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l1829_182971

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ + k = 0 ∧ x₂^2 + 4*x₂ + k = 0) ↔ k ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l1829_182971


namespace NUMINAMATH_CALUDE_candy_distribution_l1829_182957

theorem candy_distribution (total_candy : ℕ) (total_bags : ℕ) (chocolate_heart_bags : ℕ) (chocolate_kiss_bags : ℕ)
  (h1 : total_candy = 63)
  (h2 : total_bags = 9)
  (h3 : chocolate_heart_bags = 2)
  (h4 : chocolate_kiss_bags = 3)
  (h5 : total_candy % total_bags = 0) :
  let candy_per_bag := total_candy / total_bags
  let chocolate_bags := chocolate_heart_bags + chocolate_kiss_bags
  let non_chocolate_bags := total_bags - chocolate_bags
  non_chocolate_bags * candy_per_bag = 28 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1829_182957


namespace NUMINAMATH_CALUDE_track_team_boys_l1829_182936

theorem track_team_boys (initial_girls : ℕ) (girls_joined : ℕ) (boys_quit : ℕ) (final_total : ℕ) :
  initial_girls = 18 →
  girls_joined = 7 →
  boys_quit = 4 →
  final_total = 36 →
  ∃ initial_boys : ℕ,
    initial_boys = 15 ∧
    final_total = (initial_girls + girls_joined) + (initial_boys - boys_quit) :=
by
  sorry

end NUMINAMATH_CALUDE_track_team_boys_l1829_182936


namespace NUMINAMATH_CALUDE_jayas_rank_from_bottom_l1829_182944

/-- Given a class of students, calculate the rank from the bottom based on the rank from the top -/
def rankFromBottom (totalStudents : ℕ) (rankFromTop : ℕ) : ℕ :=
  totalStudents - rankFromTop + 1

/-- Theorem stating Jaya's rank from the bottom in a class of 53 students -/
theorem jayas_rank_from_bottom :
  let totalStudents : ℕ := 53
  let jayasRankFromTop : ℕ := 5
  rankFromBottom totalStudents jayasRankFromTop = 50 := by
  sorry


end NUMINAMATH_CALUDE_jayas_rank_from_bottom_l1829_182944


namespace NUMINAMATH_CALUDE_residential_building_capacity_l1829_182984

/-- The number of households that can be accommodated in multiple identical residential buildings. -/
def total_households (floors_per_building : ℕ) (households_per_floor : ℕ) (num_buildings : ℕ) : ℕ :=
  floors_per_building * households_per_floor * num_buildings

/-- Theorem stating that 10 buildings with 16 floors and 12 households per floor can accommodate 1920 households. -/
theorem residential_building_capacity :
  total_households 16 12 10 = 1920 := by
  sorry

end NUMINAMATH_CALUDE_residential_building_capacity_l1829_182984


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1829_182965

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  (∀ n : ℕ, a n > 0) →
  a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 16 →
  a 2 + a 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1829_182965


namespace NUMINAMATH_CALUDE_inequality_solution_l1829_182952

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x ≥ 3 }
  else if a < 0 then { x | x ≥ 3 ∨ x ≤ 2/a }
  else if 0 < a ∧ a < 2/3 then { x | 3 ≤ x ∧ x ≤ 2/a }
  else if a = 2/3 then { x | x = 3 }
  else { x | 2/a ≤ x ∧ x ≤ 3 }

theorem inequality_solution (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 - (3*a + 2) * x + 6 ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1829_182952


namespace NUMINAMATH_CALUDE_grandpa_to_uncle_ratio_l1829_182939

/-- Represents the number of toy cars in various scenarios --/
structure ToyCars where
  initial : ℕ
  final : ℕ
  fromDad : ℕ
  fromMum : ℕ
  fromAuntie : ℕ
  fromUncle : ℕ
  fromGrandpa : ℕ

/-- Theorem stating the ratio of Grandpa's gift to Uncle's gift --/
theorem grandpa_to_uncle_ratio (cars : ToyCars)
  (h1 : cars.initial = 150)
  (h2 : cars.final = 196)
  (h3 : cars.fromDad = 10)
  (h4 : cars.fromMum = cars.fromDad + 5)
  (h5 : cars.fromAuntie = 6)
  (h6 : cars.fromUncle = cars.fromAuntie - 1)
  (h7 : cars.final = cars.initial + cars.fromDad + cars.fromMum + cars.fromAuntie + cars.fromUncle + cars.fromGrandpa) :
  cars.fromGrandpa = 2 * cars.fromUncle := by
  sorry

#check grandpa_to_uncle_ratio

end NUMINAMATH_CALUDE_grandpa_to_uncle_ratio_l1829_182939


namespace NUMINAMATH_CALUDE_quadratic_root_c_value_l1829_182978

theorem quadratic_root_c_value (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 13 * x + c = 0 ↔ x = (-13 + Real.sqrt 19) / 4 ∨ x = (-13 - Real.sqrt 19) / 4) →
  c = 18.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_c_value_l1829_182978


namespace NUMINAMATH_CALUDE_power_of_fraction_l1829_182959

theorem power_of_fraction : (5 / 3 : ℚ) ^ 3 = 125 / 27 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_l1829_182959


namespace NUMINAMATH_CALUDE_solve_equation_l1829_182906

theorem solve_equation (x : ℝ) : (x^3)^(1/2) = 18 * 18^(1/9) → x = 18^(20/27) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1829_182906


namespace NUMINAMATH_CALUDE_quadratic_roots_l1829_182982

theorem quadratic_roots (x y : ℝ) : 
  x + y = 8 → 
  |x - y| = 10 → 
  x^2 - 8*x - 9 = 0 ∧ y^2 - 8*y - 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1829_182982


namespace NUMINAMATH_CALUDE_eighteen_is_seventyfive_percent_of_twentyfour_l1829_182964

theorem eighteen_is_seventyfive_percent_of_twentyfour (x : ℝ) : 
  18 = 0.75 * x → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_is_seventyfive_percent_of_twentyfour_l1829_182964


namespace NUMINAMATH_CALUDE_cupcake_packages_l1829_182923

theorem cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) :
  initial_cupcakes = 18 →
  eaten_cupcakes = 8 →
  cupcakes_per_package = 2 →
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcake_packages_l1829_182923


namespace NUMINAMATH_CALUDE_power_equality_l1829_182998

theorem power_equality : (8 : ℕ) ^ 8 = (4 : ℕ) ^ 12 ∧ (8 : ℕ) ^ 8 = (2 : ℕ) ^ 24 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1829_182998


namespace NUMINAMATH_CALUDE_parametric_to_circle_equation_l1829_182961

/-- Given parametric equations for a curve and a relationship between parameters,
    prove that the resulting equation is that of a circle with specific center and radius,
    excluding two points on the x-axis. -/
theorem parametric_to_circle_equation 
  (u v : ℝ) (m : ℝ) (hm : m ≠ 0)
  (hx : ∀ u v, x = (1 - u^2 - v^2) / ((1 - u)^2 + v^2))
  (hy : ∀ u v, y = 2 * v / ((1 - u)^2 + v^2))
  (hv : v = m * u) :
  x^2 + (y - 1/m)^2 = 1 + 1/m^2 ∧ 
  (x ≠ 1 ∨ y ≠ 0) ∧ (x ≠ -1 ∨ y ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_parametric_to_circle_equation_l1829_182961


namespace NUMINAMATH_CALUDE_cyclic_fraction_inequality_l1829_182945

theorem cyclic_fraction_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z := by
  sorry

end NUMINAMATH_CALUDE_cyclic_fraction_inequality_l1829_182945


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1829_182931

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 4) : 
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1829_182931


namespace NUMINAMATH_CALUDE_smallest_k_for_900_digit_sum_l1829_182990

def digit_sum (n : ℕ) : ℕ := sorry

def repeated_7 (k : ℕ) : ℕ := (10^k - 1) / 9

theorem smallest_k_for_900_digit_sum : 
  ∀ k : ℕ, k > 0 → 
  (∀ j : ℕ, 0 < j ∧ j < k → digit_sum (9 * repeated_7 j) ≠ 900) ∧ 
  digit_sum (9 * repeated_7 k) = 900 → 
  k = 100 := by sorry

end NUMINAMATH_CALUDE_smallest_k_for_900_digit_sum_l1829_182990


namespace NUMINAMATH_CALUDE_cake_cutting_l1829_182958

/-- Represents a rectangular grid --/
structure RectangularGrid :=
  (rows : ℕ)
  (cols : ℕ)

/-- The maximum number of pieces created by a single straight line cut in a rectangular grid --/
def max_pieces (grid : RectangularGrid) : ℕ :=
  grid.rows * grid.cols + (grid.rows + grid.cols - 1)

/-- The minimum number of straight cuts required to intersect all cells in a rectangular grid --/
def min_cuts (grid : RectangularGrid) : ℕ :=
  min grid.rows grid.cols

theorem cake_cutting (grid : RectangularGrid) 
  (h1 : grid.rows = 3) 
  (h2 : grid.cols = 5) : 
  max_pieces grid = 22 ∧ min_cuts grid = 3 := by
  sorry

#eval max_pieces ⟨3, 5⟩
#eval min_cuts ⟨3, 5⟩

end NUMINAMATH_CALUDE_cake_cutting_l1829_182958


namespace NUMINAMATH_CALUDE_lcm_of_20_45_28_l1829_182981

theorem lcm_of_20_45_28 : Nat.lcm (Nat.lcm 20 45) 28 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_28_l1829_182981


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_pairs_l1829_182904

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

theorem infinitely_many_divisible_pairs :
  ∀ n : ℕ, ∃ a b : ℕ,
    a = fib (2 * n + 1) ∧
    b = fib (2 * n + 3) ∧
    a > 0 ∧
    b > 0 ∧
    a ∣ (b^2 + 1) ∧
    b ∣ (a^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_pairs_l1829_182904


namespace NUMINAMATH_CALUDE_domain_of_shifted_function_l1829_182911

-- Define the function f with domain [0,2]
def f : Set ℝ := Set.Icc 0 2

-- Define the function g(x) = f(x+1)
def g (x : ℝ) : Prop := ∃ y ∈ f, y = x + 1

-- Theorem statement
theorem domain_of_shifted_function :
  Set.Icc (-1) 1 = {x | g x} := by sorry

end NUMINAMATH_CALUDE_domain_of_shifted_function_l1829_182911


namespace NUMINAMATH_CALUDE_double_dimensions_volume_l1829_182968

/-- A cylindrical container with volume, height, and radius. -/
structure CylindricalContainer where
  volume : ℝ
  height : ℝ
  radius : ℝ
  volume_formula : volume = Real.pi * radius^2 * height

/-- Given a cylindrical container of 5 gallons, doubling its dimensions results in a 40-gallon container -/
theorem double_dimensions_volume (c : CylindricalContainer) 
  (h_volume : c.volume = 5) :
  let new_container : CylindricalContainer := {
    volume := Real.pi * (2 * c.radius)^2 * (2 * c.height),
    height := 2 * c.height,
    radius := 2 * c.radius,
    volume_formula := by sorry
  }
  new_container.volume = 40 := by
  sorry

end NUMINAMATH_CALUDE_double_dimensions_volume_l1829_182968


namespace NUMINAMATH_CALUDE_race_speed_ratio_l1829_182922

theorem race_speed_ratio (v_a v_b : ℝ) (h : v_a > 0 ∧ v_b > 0) :
  (1 / v_a = (1 - 13/30) / v_b) → v_a / v_b = 30 / 17 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l1829_182922


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l1829_182974

/-- The quadratic function f(x) = (x-2)^2 + 1 -/
def f (x : ℝ) : ℝ := (x - 2)^2 + 1

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (2, 1)

/-- Theorem: The vertex of the quadratic function f(x) = (x-2)^2 + 1 is (2,1) -/
theorem vertex_of_quadratic :
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l1829_182974


namespace NUMINAMATH_CALUDE_ripe_orange_harvest_l1829_182969

/-- The number of days of harvest -/
def harvest_days : ℕ := 73

/-- The number of sacks of ripe oranges harvested per day -/
def daily_ripe_harvest : ℕ := 5

/-- The total number of sacks of ripe oranges harvested over the entire period -/
def total_ripe_harvest : ℕ := harvest_days * daily_ripe_harvest

theorem ripe_orange_harvest :
  total_ripe_harvest = 365 := by
  sorry

end NUMINAMATH_CALUDE_ripe_orange_harvest_l1829_182969


namespace NUMINAMATH_CALUDE_sum_difference_is_450_l1829_182920

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_down_to_10 (n : ℕ) : ℕ := (n / 10) * 10

def kate_sum (n : ℕ) : ℕ := 
  (List.range n).map round_down_to_10 |> List.sum

theorem sum_difference_is_450 (n : ℕ) (h : n = 100) : 
  (sum_first_n n) - (kate_sum n) = 450 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_is_450_l1829_182920


namespace NUMINAMATH_CALUDE_green_apples_count_l1829_182976

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := sorry

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 6

/-- The number of students who took fruit -/
def students_taking_fruit : ℕ := 5

/-- The number of extra apples left over -/
def extra_apples : ℕ := 16

/-- Theorem stating that the number of green apples ordered is 15 -/
theorem green_apples_count : green_apples = 15 := by
  sorry

end NUMINAMATH_CALUDE_green_apples_count_l1829_182976


namespace NUMINAMATH_CALUDE_factor_quadratic_l1829_182912

theorem factor_quadratic (m : ℤ) : 
  let s : ℤ := 5
  m^2 - s*m - 24 = (m - 8) * (m + 3) :=
by sorry

end NUMINAMATH_CALUDE_factor_quadratic_l1829_182912


namespace NUMINAMATH_CALUDE_builder_purchase_cost_l1829_182940

/-- Calculates the total cost of a builder's purchase with specific items, taxes, and discounts --/
theorem builder_purchase_cost : 
  let drill_bits_cost : ℚ := 5 * 6
  let hammers_cost : ℚ := 3 * 8
  let toolbox_cost : ℚ := 25
  let nails_cost : ℚ := (50 / 2) * 0.1
  let drill_bits_tax : ℚ := drill_bits_cost * 0.1
  let toolbox_tax : ℚ := toolbox_cost * 0.15
  let hammers_discount : ℚ := hammers_cost * 0.05
  let total_before_discount : ℚ := drill_bits_cost + drill_bits_tax + hammers_cost - hammers_discount + toolbox_cost + toolbox_tax + nails_cost
  let overall_discount : ℚ := if total_before_discount > 60 then total_before_discount * 0.05 else 0
  let final_total : ℚ := total_before_discount - overall_discount
  ∃ (rounded_total : ℚ), (rounded_total ≥ final_total) ∧ (rounded_total < final_total + 0.005) ∧ (rounded_total = 82.70) :=
by sorry


end NUMINAMATH_CALUDE_builder_purchase_cost_l1829_182940


namespace NUMINAMATH_CALUDE_price_reduction_order_invariance_l1829_182966

theorem price_reduction_order_invariance :
  let reduction1 := 0.1
  let reduction2 := 0.15
  let total_reduction1 := 1 - (1 - reduction1) * (1 - reduction2)
  let total_reduction2 := 1 - (1 - reduction2) * (1 - reduction1)
  total_reduction1 = total_reduction2 ∧ total_reduction1 = 0.235 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_order_invariance_l1829_182966


namespace NUMINAMATH_CALUDE_shrimp_cost_per_pound_l1829_182979

/-- Calculates the cost per pound of shrimp for Wayne's shrimp cocktail appetizer. -/
theorem shrimp_cost_per_pound 
  (shrimp_per_guest : ℕ) 
  (num_guests : ℕ) 
  (shrimp_per_pound : ℕ) 
  (total_cost : ℚ) : 
  shrimp_per_guest = 5 → 
  num_guests = 40 → 
  shrimp_per_pound = 20 → 
  total_cost = 170 → 
  (total_cost / (shrimp_per_guest * num_guests / shrimp_per_pound : ℚ)) = 17 :=
by sorry

end NUMINAMATH_CALUDE_shrimp_cost_per_pound_l1829_182979


namespace NUMINAMATH_CALUDE_tangent_line_cubic_curve_l1829_182907

/-- Given a cubic function f(x) = x^3 + ax + b represented by curve C,
    if the line y = kx - 2 is tangent to C at point (1, 0),
    then k = 2 and f(x) = x^3 - x -/
theorem tangent_line_cubic_curve (a b k : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^3 + a*x + b
  let tangent_line : ℝ → ℝ := fun x ↦ k*x - 2
  (f 1 = 0) →
  (tangent_line 1 = 0) →
  (∀ x, tangent_line x ≤ f x) →
  (∃ x₀, x₀ ≠ 1 ∧ tangent_line x₀ = f x₀) →
  (k = 2 ∧ ∀ x, f x = x^3 - x) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_curve_l1829_182907


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_find_value_l1829_182943

-- Question 1
theorem simplify_expression (a b : ℝ) :
  8 * (a + b) + 6 * (a + b) - 2 * (a + b) = 12 * (a + b) := by sorry

-- Question 2
theorem evaluate_expression (x y : ℝ) (h : x + y = 1/2) :
  9 * (x + y)^2 + 3 * (x + y) + 7 * (x + y)^2 - 7 * (x + y) = 2 := by sorry

-- Question 3
theorem find_value (x y : ℝ) (h : x^2 - 2*y = 4) :
  -3 * x^2 + 6 * y + 2 = -10 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_find_value_l1829_182943


namespace NUMINAMATH_CALUDE_power_sum_cosine_l1829_182913

theorem power_sum_cosine (θ : Real) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : x + 1/x = 2 * Real.cos θ) :
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_cosine_l1829_182913


namespace NUMINAMATH_CALUDE_equilateral_iff_rhombus_l1829_182996

-- Define a parallelogram
structure Parallelogram :=
  (sides : Fin 4 → ℝ)
  (is_parallelogram : sides 0 = sides 2 ∧ sides 1 = sides 3)

-- Define an equilateral parallelogram
def is_equilateral (p : Parallelogram) : Prop :=
  p.sides 0 = p.sides 1 ∧ p.sides 1 = p.sides 2 ∧ p.sides 2 = p.sides 3

-- Define a rhombus
def is_rhombus (p : Parallelogram) : Prop :=
  p.sides 0 = p.sides 1 ∧ p.sides 1 = p.sides 2 ∧ p.sides 2 = p.sides 3

-- Theorem: A parallelogram is equilateral if and only if it is a rhombus
theorem equilateral_iff_rhombus (p : Parallelogram) :
  is_equilateral p ↔ is_rhombus p :=
sorry

end NUMINAMATH_CALUDE_equilateral_iff_rhombus_l1829_182996


namespace NUMINAMATH_CALUDE_roundness_of_1280000_l1829_182960

/-- Roundness of a positive integer is the sum of exponents in its prime factorization. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The number we're calculating the roundness for -/
def our_number : ℕ+ := 1280000

/-- Theorem stating that the roundness of 1,280,000 is 19 -/
theorem roundness_of_1280000 : roundness our_number = 19 := by
  sorry

end NUMINAMATH_CALUDE_roundness_of_1280000_l1829_182960


namespace NUMINAMATH_CALUDE_yulin_school_sampling_l1829_182910

/-- Systematic sampling function that calculates the number of elements to be removed -/
def systematicSamplingRemoval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize % sampleSize

/-- Theorem stating that for the given population and sample size, 
    the number of students to be removed is 2 -/
theorem yulin_school_sampling :
  systematicSamplingRemoval 254 42 = 2 := by
  sorry

end NUMINAMATH_CALUDE_yulin_school_sampling_l1829_182910


namespace NUMINAMATH_CALUDE_subtraction_is_perfect_square_l1829_182951

def A : ℕ := (10^1001 - 1) / 9
def B : ℕ := (10^2002 - 1) / 9
def C : ℕ := 2 * A

theorem subtraction_is_perfect_square : B - C = (3 * A)^2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_is_perfect_square_l1829_182951


namespace NUMINAMATH_CALUDE_num_unique_heights_equals_multiples_of_five_l1829_182918

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the configuration of a tower of bricks -/
def TowerConfiguration := List Nat

/-- The number of bricks in the tower -/
def numBricks : Nat := 80

/-- The dimensions of each brick -/
def brickDimensions : BrickDimensions := { small := 3, medium := 8, large := 18 }

/-- Calculate the height of a tower given its configuration -/
def towerHeight (config : TowerConfiguration) : Nat :=
  config.sum

/-- Generate all possible tower configurations -/
def allConfigurations : List TowerConfiguration :=
  sorry

/-- Calculate the number of unique tower heights -/
def numUniqueHeights : Nat :=
  (allConfigurations.map towerHeight).toFinset.card

/-- The main theorem to prove -/
theorem num_unique_heights_equals_multiples_of_five :
  numUniqueHeights = (((numBricks * brickDimensions.large) - (numBricks * brickDimensions.small)) / 5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_num_unique_heights_equals_multiples_of_five_l1829_182918


namespace NUMINAMATH_CALUDE_chess_pieces_remaining_l1829_182975

theorem chess_pieces_remaining (initial_pieces : ℕ) (scarlett_lost : ℕ) (hannah_lost : ℕ)
  (h1 : initial_pieces = 32)
  (h2 : scarlett_lost = 6)
  (h3 : hannah_lost = 8) :
  initial_pieces - (scarlett_lost + hannah_lost) = 18 :=
by sorry

end NUMINAMATH_CALUDE_chess_pieces_remaining_l1829_182975


namespace NUMINAMATH_CALUDE_circle_radius_zero_l1829_182928

/-- The radius of a circle defined by the equation 4x^2 - 8x + 4y^2 + 16y + 20 = 0 is 0 -/
theorem circle_radius_zero (x y : ℝ) : 
  (4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0) → 
  ∃ (h k : ℝ), ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l1829_182928
