import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l2_225

theorem simplify_expression (x y : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 9*y = 45*x + 9*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2_225


namespace NUMINAMATH_CALUDE_fred_balloons_l2_262

theorem fred_balloons (initial : ℕ) (to_sandy : ℕ) (to_bob : ℕ) :
  initial = 709 →
  to_sandy = 221 →
  to_bob = 153 →
  initial - to_sandy - to_bob = 335 := by
  sorry

end NUMINAMATH_CALUDE_fred_balloons_l2_262


namespace NUMINAMATH_CALUDE_grape_ratio_theorem_l2_205

/-- Represents the contents and cost of a fruit basket -/
structure FruitBasket where
  banana_count : ℕ
  apple_count : ℕ
  strawberry_count : ℕ
  avocado_count : ℕ
  banana_price : ℚ
  apple_price : ℚ
  strawberry_price : ℚ
  avocado_price : ℚ
  grape_portion_price : ℚ
  total_cost : ℚ

/-- Calculates the cost of fruits excluding grapes -/
def cost_excluding_grapes (fb : FruitBasket) : ℚ :=
  fb.banana_count * fb.banana_price +
  fb.apple_count * fb.apple_price +
  fb.strawberry_count / 12 * fb.strawberry_price +
  fb.avocado_count * fb.avocado_price

/-- Calculates the cost of grapes in the basket -/
def grape_cost (fb : FruitBasket) : ℚ :=
  fb.total_cost - cost_excluding_grapes fb

/-- Represents the ratio of grapes in the basket to a whole bunch -/
structure GrapeRatio where
  numerator : ℚ
  denominator : ℚ

/-- Theorem stating the ratio of grapes in the basket to a whole bunch -/
theorem grape_ratio_theorem (fb : FruitBasket) (x : ℚ) :
  fb.banana_count = 4 →
  fb.apple_count = 3 →
  fb.strawberry_count = 24 →
  fb.avocado_count = 2 →
  fb.banana_price = 1 →
  fb.apple_price = 2 →
  fb.strawberry_price = 4 →
  fb.avocado_price = 3 →
  fb.grape_portion_price = 2 →
  fb.total_cost = 28 →
  x > 2 →
  ∃ (gr : GrapeRatio), gr.numerator = 2 ∧ gr.denominator = x :=
by sorry

end NUMINAMATH_CALUDE_grape_ratio_theorem_l2_205


namespace NUMINAMATH_CALUDE_lower_bound_of_expression_l2_207

theorem lower_bound_of_expression (L : ℤ) : 
  (∃ (S : Finset ℤ), 
    (∀ n ∈ S, L < 4*n + 7 ∧ 4*n + 7 < 120) ∧ 
    S.card = 30) →
  L = 5 :=
sorry

end NUMINAMATH_CALUDE_lower_bound_of_expression_l2_207


namespace NUMINAMATH_CALUDE_binary_to_octal_equivalence_l2_288

-- Define the binary number
def binary_num : ℕ := 11011

-- Define the octal number
def octal_num : ℕ := 33

-- Theorem stating the equivalence of the binary and octal representations
theorem binary_to_octal_equivalence :
  (binary_num.digits 2).foldl (· + 2 * ·) 0 = (octal_num.digits 8).foldl (· + 8 * ·) 0 :=
by sorry

end NUMINAMATH_CALUDE_binary_to_octal_equivalence_l2_288


namespace NUMINAMATH_CALUDE_number_composition_proof_l2_281

theorem number_composition_proof : 
  let ones : ℕ := 5
  let tenths : ℕ := 7
  let hundredths : ℕ := 21
  let thousandths : ℕ := 53
  let composed_number := 
    (ones : ℝ) + 
    (tenths : ℝ) * 0.1 + 
    (hundredths : ℝ) * 0.01 + 
    (thousandths : ℝ) * 0.001
  10 * composed_number = 59.63 := by
sorry

end NUMINAMATH_CALUDE_number_composition_proof_l2_281


namespace NUMINAMATH_CALUDE_inequality_solution_l2_245

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 3/4) ↔ (-2 < x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2_245


namespace NUMINAMATH_CALUDE_jack_gerald_notebook_difference_l2_218

theorem jack_gerald_notebook_difference :
  ∀ (jack_initial gerald : ℕ),
    jack_initial > gerald →
    gerald = 8 →
    jack_initial - 5 - 6 = 10 →
    jack_initial - gerald = 13 := by
  sorry

end NUMINAMATH_CALUDE_jack_gerald_notebook_difference_l2_218


namespace NUMINAMATH_CALUDE_joes_lift_ratio_l2_229

/-- Joe's weight-lifting competition results -/
def JoesLifts (first second : ℕ) : Prop :=
  first + second = 600 ∧ first = 300 ∧ 2 * first = second + 300

theorem joes_lift_ratio :
  ∀ first second : ℕ, JoesLifts first second → first = second :=
by
  sorry

end NUMINAMATH_CALUDE_joes_lift_ratio_l2_229


namespace NUMINAMATH_CALUDE_value_of_M_l2_206

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.55 * 4500) ∧ (M = 9900) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l2_206


namespace NUMINAMATH_CALUDE_masons_father_age_l2_283

theorem masons_father_age :
  ∀ (mason_age sydney_age father_age : ℕ),
    mason_age = 20 →
    sydney_age = 3 * mason_age →
    father_age = sydney_age + 6 →
    father_age = 66 := by
  sorry

end NUMINAMATH_CALUDE_masons_father_age_l2_283


namespace NUMINAMATH_CALUDE_general_solution_second_order_recurrence_l2_243

/-- Second-order linear recurrence sequence -/
def RecurrenceSequence (a b : ℝ) (u : ℕ → ℝ) : Prop :=
  ∀ n, u (n + 2) = a * u (n + 1) + b * u n

/-- Characteristic polynomial of the recurrence sequence -/
def CharacteristicPolynomial (a b : ℝ) (X : ℝ) : ℝ :=
  X^2 - a*X - b

theorem general_solution_second_order_recurrence
  (a b : ℝ) (u : ℕ → ℝ) (r₁ r₂ : ℝ) :
  RecurrenceSequence a b u →
  r₁ ≠ r₂ →
  CharacteristicPolynomial a b r₁ = 0 →
  CharacteristicPolynomial a b r₂ = 0 →
  ∃ c d : ℝ, ∀ n, u n = c * r₁^n + d * r₂^n ∧
    c = (u 1 - u 0 * r₂) / (r₁ - r₂) ∧
    d = (u 0 * r₁ - u 1) / (r₁ - r₂) :=
sorry

end NUMINAMATH_CALUDE_general_solution_second_order_recurrence_l2_243


namespace NUMINAMATH_CALUDE_angelina_walking_speed_l2_210

/-- Angelina's walking problem -/
theorem angelina_walking_speed 
  (distance_home_to_grocery : ℝ) 
  (distance_grocery_to_gym : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance_home_to_grocery = 100)
  (h2 : distance_grocery_to_gym = 180)
  (h3 : time_difference = 40)
  : ∃ (v : ℝ), 
    (distance_home_to_grocery / v - distance_grocery_to_gym / (2 * v) = time_difference) ∧ 
    (2 * v = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_angelina_walking_speed_l2_210


namespace NUMINAMATH_CALUDE_parking_problem_l2_267

/-- Represents the number of parking spaces -/
def total_spaces : ℕ := 7

/-- Represents the number of cars -/
def num_cars : ℕ := 3

/-- Represents the number of consecutive empty spaces -/
def empty_spaces : ℕ := 4

/-- Represents the total number of units to arrange (cars + empty space block) -/
def total_units : ℕ := num_cars + 1

/-- The number of different parking arrangements -/
def parking_arrangements : ℕ := Nat.factorial total_units

theorem parking_problem :
  parking_arrangements = 24 :=
sorry

end NUMINAMATH_CALUDE_parking_problem_l2_267


namespace NUMINAMATH_CALUDE_correct_proposition_l2_296

-- Define proposition p
def p : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

-- Theorem to prove
theorem correct_proposition : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l2_296


namespace NUMINAMATH_CALUDE_largest_root_is_four_l2_277

/-- The polynomial P(x) -/
def P (x r s : ℝ) : ℝ := x^6 - 12*x^5 + 40*x^4 - r*x^3 + s*x^2

/-- The line L(x) -/
def L (x d e : ℝ) : ℝ := d*x - e

/-- Theorem stating that the largest root of P(x) = L(x) is 4 -/
theorem largest_root_is_four 
  (r s d e : ℝ) 
  (h : ∃ (x₁ x₂ x₃ : ℝ), 
    (x₁ < x₂ ∧ x₂ < x₃) ∧ 
    (∀ x : ℝ, P x r s = L x d e ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    (∀ x : ℝ, (x - x₁)^2 * (x - x₂)^2 * (x - x₃) = P x r s - L x d e)) : 
  (∃ (x : ℝ), P x r s = L x d e ∧ ∀ y : ℝ, P y r s = L y d e → y ≤ x) ∧ 
  (∀ x : ℝ, P x r s = L x d e → x ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_largest_root_is_four_l2_277


namespace NUMINAMATH_CALUDE_x_plus_y_values_l2_248

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 5) (h2 : y = Real.sqrt 9) :
  x + y = -2 ∨ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l2_248


namespace NUMINAMATH_CALUDE_exactly_seventeen_solutions_l2_247

/-- The number of ordered pairs of complex numbers satisfying the given equations -/
def num_solutions : ℕ := 17

/-- The property that a pair of complex numbers satisfies the given equations -/
def satisfies_equations (a b : ℂ) : Prop :=
  a^5 * b^3 = 1 ∧ a^9 * b^2 = 1

/-- The theorem stating that there are exactly 17 solutions -/
theorem exactly_seventeen_solutions :
  ∃! (s : Set (ℂ × ℂ)), 
    (∀ (p : ℂ × ℂ), p ∈ s ↔ satisfies_equations p.1 p.2) ∧
    Finite s ∧
    Nat.card s = num_solutions :=
sorry

end NUMINAMATH_CALUDE_exactly_seventeen_solutions_l2_247


namespace NUMINAMATH_CALUDE_smallest_number_l2_200

theorem smallest_number (a b c d : ℝ) 
  (ha : a = 1) 
  (hb : b = -3) 
  (hc : c = -Real.sqrt 2) 
  (hd : d = -Real.pi) : 
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
sorry

end NUMINAMATH_CALUDE_smallest_number_l2_200


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l2_257

def original_width : ℕ := 5
def original_height : ℕ := 6
def original_black_tiles : ℕ := 12
def original_white_tiles : ℕ := 18
def border_width : ℕ := 1

def extended_width : ℕ := original_width + 2 * border_width
def extended_height : ℕ := original_height + 2 * border_width

def total_extended_tiles : ℕ := extended_width * extended_height
def new_white_tiles : ℕ := total_extended_tiles - (original_width * original_height)
def total_white_tiles : ℕ := original_white_tiles + new_white_tiles

theorem extended_pattern_ratio :
  (original_black_tiles : ℚ) / total_white_tiles = 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l2_257


namespace NUMINAMATH_CALUDE_wire_length_proof_l2_289

-- Define the area of the square field
def field_area : ℝ := 69696

-- Define the number of times the wire goes around the field
def rounds : ℕ := 15

-- Theorem statement
theorem wire_length_proof :
  let side_length := Real.sqrt field_area
  let perimeter := 4 * side_length
  rounds * perimeter = 15840 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_proof_l2_289


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_lines_l2_273

/-- Represents a line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle --/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Check if three lines are coplanar and equidistant --/
def are_coplanar_equidistant (l1 l2 l3 : Line) : Prop :=
  l1.slope = l2.slope ∧ l2.slope = l3.slope ∧
  |l2.intercept - l1.intercept| = |l3.intercept - l2.intercept|

/-- Check if a triangle is equilateral --/
def is_equilateral (t : Triangle) : Prop :=
  let d1 := ((t.b.x - t.a.x)^2 + (t.b.y - t.a.y)^2).sqrt
  let d2 := ((t.c.x - t.b.x)^2 + (t.c.y - t.b.y)^2).sqrt
  let d3 := ((t.a.x - t.c.x)^2 + (t.a.y - t.c.y)^2).sqrt
  d1 = d2 ∧ d2 = d3

/-- Check if a point lies on a line --/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Theorem: Given three coplanar and equidistant lines, 
    it is possible to construct an equilateral triangle 
    with its vertices lying on these lines --/
theorem equilateral_triangle_on_lines 
  (l1 l2 l3 : Line) 
  (h : are_coplanar_equidistant l1 l2 l3) :
  ∃ (t : Triangle), 
    is_equilateral t ∧ 
    point_on_line t.a l1 ∧ 
    point_on_line t.b l2 ∧ 
    point_on_line t.c l3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_lines_l2_273


namespace NUMINAMATH_CALUDE_min_triangle_forming_number_l2_208

def CanFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def MinTriangleForming : ℕ → Prop
| n => ∀ (S : Finset ℕ), S.card = n → (∀ x ∈ S, x ≥ 1 ∧ x ≤ 1000) →
       ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ CanFormTriangle a b c

theorem min_triangle_forming_number : MinTriangleForming 16 ∧ ∀ k < 16, ¬MinTriangleForming k :=
  sorry

end NUMINAMATH_CALUDE_min_triangle_forming_number_l2_208


namespace NUMINAMATH_CALUDE_matrix_cube_eq_matrix_plus_identity_det_positive_l2_294

open Matrix

theorem matrix_cube_eq_matrix_plus_identity_det_positive :
  ∀ (n : ℕ), ∃ (A : Matrix (Fin n) (Fin n) ℝ), A ^ 3 = A + 1 →
  ∀ (A : Matrix (Fin n) (Fin n) ℝ), A ^ 3 = A + 1 → 0 < det A :=
by sorry

end NUMINAMATH_CALUDE_matrix_cube_eq_matrix_plus_identity_det_positive_l2_294


namespace NUMINAMATH_CALUDE_fifty_cent_items_count_l2_274

def total_items : ℕ := 35
def total_price : ℕ := 4000  -- in cents

def is_valid_purchase (x y z : ℕ) : Prop :=
  x + y + z = total_items ∧
  50 * x + 300 * y + 400 * z = total_price

theorem fifty_cent_items_count : ∃ (x y z : ℕ), is_valid_purchase x y z ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_fifty_cent_items_count_l2_274


namespace NUMINAMATH_CALUDE_carla_class_size_l2_298

theorem carla_class_size :
  let students_in_restroom : ℕ := 2
  let absent_students : ℕ := 3 * students_in_restroom - 1
  let total_desks : ℕ := 4 * 6
  let occupied_desks : ℕ := (2 * total_desks) / 3
  let students_present : ℕ := occupied_desks
  students_in_restroom + absent_students + students_present = 23 := by
sorry

end NUMINAMATH_CALUDE_carla_class_size_l2_298


namespace NUMINAMATH_CALUDE_sum_RS_ST_l2_232

/-- Represents a polygon PQRSTU -/
structure Polygon :=
  (area : ℝ)
  (PQ : ℝ)
  (QR : ℝ)
  (TU : ℝ)

/-- Theorem stating the sum of RS and ST in the polygon PQRSTU -/
theorem sum_RS_ST (poly : Polygon) (h1 : poly.area = 70) (h2 : poly.PQ = 10) 
  (h3 : poly.QR = 7) (h4 : poly.TU = 6) : ∃ (RS ST : ℝ), RS + ST = 80 := by
  sorry

#check sum_RS_ST

end NUMINAMATH_CALUDE_sum_RS_ST_l2_232


namespace NUMINAMATH_CALUDE_beths_underwater_time_l2_282

/-- Calculates the total underwater time for a scuba diver -/
def total_underwater_time (primary_tank_time : ℕ) (supplemental_tanks : ℕ) (time_per_supplemental_tank : ℕ) : ℕ :=
  primary_tank_time + supplemental_tanks * time_per_supplemental_tank

/-- Proves that Beth's total underwater time is 8 hours -/
theorem beths_underwater_time :
  let primary_tank_time : ℕ := 2
  let supplemental_tanks : ℕ := 6
  let time_per_supplemental_tank : ℕ := 1
  total_underwater_time primary_tank_time supplemental_tanks time_per_supplemental_tank = 8 := by
  sorry

#eval total_underwater_time 2 6 1

end NUMINAMATH_CALUDE_beths_underwater_time_l2_282


namespace NUMINAMATH_CALUDE_vector_collinearity_l2_261

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

/-- Collinearity of two vectors in ℝ² -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

/-- The main theorem -/
theorem vector_collinearity (m : ℝ) :
  collinear ((m * a.1 + 4 * b.1, m * a.2 + 4 * b.2)) (a.1 - 2 * b.1, a.2 - 2 * b.2) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2_261


namespace NUMINAMATH_CALUDE_circle_reconstruction_uniqueness_l2_293

-- Define the types for lines and circles
def Line : Type := ℝ × ℝ → Prop
def Circle : Type := ℝ × ℝ → Prop

-- Define the property of two lines being parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the property of two lines being perpendicular
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the property of a line being tangent to a circle
def tangent_to (l : Line) (c : Circle) : Prop := sorry

-- Define the distance between two parallel lines
def distance_between_parallel_lines (l1 l2 : Line) : ℝ := sorry

-- Main theorem
theorem circle_reconstruction_uniqueness 
  (e1 e2 f1 f2 : Line) 
  (h_parallel_e : parallel e1 e2) 
  (h_parallel_f : parallel f1 f2) 
  (h_not_perp_e1f1 : ¬ perpendicular e1 f1) 
  (h_not_perp_e2f2 : ¬ perpendicular e2 f2) :
  (∃! (k1 k2 : Circle), 
    tangent_to e1 k1 ∧ tangent_to e2 k2 ∧ 
    tangent_to f1 k1 ∧ tangent_to f2 k2 ∧ 
    (∃ (e f : Line), tangent_to e k1 ∧ tangent_to e k2 ∧ 
                     tangent_to f k1 ∧ tangent_to f k2)) ↔ 
  distance_between_parallel_lines e1 e2 ≠ distance_between_parallel_lines f1 f2 :=
sorry

end NUMINAMATH_CALUDE_circle_reconstruction_uniqueness_l2_293


namespace NUMINAMATH_CALUDE_return_amount_calculation_l2_272

-- Define the borrowed amount
def borrowed_amount : ℝ := 100

-- Define the interest rate
def interest_rate : ℝ := 0.10

-- Theorem to prove
theorem return_amount_calculation :
  borrowed_amount * (1 + interest_rate) = 110 := by
  sorry

end NUMINAMATH_CALUDE_return_amount_calculation_l2_272


namespace NUMINAMATH_CALUDE_central_angle_invariant_under_doubling_l2_275

theorem central_angle_invariant_under_doubling 
  (r : ℝ) (l : ℝ) (h_r : r > 0) (h_l : l > 0) :
  l / r = (2 * l) / (2 * r) :=
by sorry

end NUMINAMATH_CALUDE_central_angle_invariant_under_doubling_l2_275


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l2_217

theorem smallest_solution_quartic_equation :
  ∃ x : ℝ, x^4 - 14*x^2 + 49 = 0 ∧ 
  (∀ y : ℝ, y^4 - 14*y^2 + 49 = 0 → x ≤ y) ∧
  x = -Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l2_217


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2_241

theorem imaginary_part_of_z (θ : ℝ) :
  let z : ℂ := Complex.mk (Real.sin (2 * θ) - 1) (Real.sqrt 2 * Real.cos θ - 1)
  (z.re = 0) → z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2_241


namespace NUMINAMATH_CALUDE_bulb_cost_difference_l2_271

theorem bulb_cost_difference (lamp_cost : ℝ) (total_cost : ℝ) (bulb_cost : ℝ) : 
  lamp_cost = 7 → 
  2 * lamp_cost + 6 * bulb_cost = 32 → 
  bulb_cost < lamp_cost →
  lamp_cost - bulb_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_bulb_cost_difference_l2_271


namespace NUMINAMATH_CALUDE_geometric_progression_values_l2_242

theorem geometric_progression_values (p : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (3*p + 1) = |p - 3| * r ∧ (9*p + 10) = (3*p + 1) * r) ↔ 
  (p = -1 ∨ p = 29/18) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_values_l2_242


namespace NUMINAMATH_CALUDE_frog_count_l2_215

theorem frog_count (total_eyes : ℕ) (eyes_per_frog : ℕ) (h1 : total_eyes > 0) (h2 : eyes_per_frog > 0) :
  total_eyes / eyes_per_frog = 4 →
  total_eyes = 8 ∧ eyes_per_frog = 2 :=
by sorry

end NUMINAMATH_CALUDE_frog_count_l2_215


namespace NUMINAMATH_CALUDE_journey_mpg_is_28_l2_292

/-- Calculates the average miles per gallon for a car journey -/
def average_mpg (initial_odometer final_odometer : ℕ) 
                (initial_fill first_refill second_refill : ℕ) : ℚ :=
  let total_distance := final_odometer - initial_odometer
  let total_gas := initial_fill + first_refill + second_refill
  (total_distance : ℚ) / total_gas

/-- Theorem stating that the average MPG for the given journey is 28 -/
theorem journey_mpg_is_28 :
  let initial_odometer := 56100
  let final_odometer := 57500
  let initial_fill := 10
  let first_refill := 15
  let second_refill := 25
  average_mpg initial_odometer final_odometer initial_fill first_refill second_refill = 28 := by
  sorry

#eval average_mpg 56100 57500 10 15 25

end NUMINAMATH_CALUDE_journey_mpg_is_28_l2_292


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2_227

theorem complex_equation_sum (a b : ℝ) : (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (2 + Complex.I) * (1 - b * Complex.I) = a + Complex.I →
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2_227


namespace NUMINAMATH_CALUDE_marble_distribution_correct_l2_224

/-- Represents the distribution of marbles among four boys -/
structure MarbleDistribution where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The rule for distributing marbles based on a parameter x -/
def distributionRule (x : ℕ) : MarbleDistribution :=
  { first := 3 * x + 2
  , second := x + 1
  , third := 2 * x - 1
  , fourth := x }

/-- Theorem stating that the given distribution satisfies the problem conditions -/
theorem marble_distribution_correct : ∃ x : ℕ, 
  let d := distributionRule x
  d.first = 22 ∧
  d.second = 8 ∧
  d.third = 12 ∧
  d.fourth = 7 ∧
  d.first + d.second + d.third + d.fourth = 49 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_correct_l2_224


namespace NUMINAMATH_CALUDE_car_speed_theorem_l2_266

/-- Calculates the speed of a car in miles per hour -/
def car_speed (distance_yards : ℚ) (time_seconds : ℚ) (yards_per_mile : ℚ) : ℚ :=
  (distance_yards / yards_per_mile) * (3600 / time_seconds)

/-- Theorem stating that a car traveling 22 yards in 0.5 seconds has a speed of 90 miles per hour -/
theorem car_speed_theorem :
  car_speed 22 0.5 1760 = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_theorem_l2_266


namespace NUMINAMATH_CALUDE_magical_stack_size_l2_258

/-- A stack of cards is magical if it satisfies certain conditions -/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (card_197_position : ℕ)
  (card_197_retains_position : card_197_position = 197)
  (is_magical : ∃ (a b : ℕ), a ≤ n ∧ b > n ∧ b ≤ total_cards)

/-- The number of cards in a magical stack where card 197 retains its position is 590 -/
theorem magical_stack_size (stack : MagicalStack) : stack.total_cards = 590 :=
by sorry

end NUMINAMATH_CALUDE_magical_stack_size_l2_258


namespace NUMINAMATH_CALUDE_beef_cost_calculation_l2_279

/-- Proves that the cost of a pound of beef is $5 given the initial amount,
    cheese cost, quantities purchased, and remaining amount. -/
theorem beef_cost_calculation (initial_amount : ℕ) (cheese_cost : ℕ) 
  (cheese_quantity : ℕ) (beef_quantity : ℕ) (remaining_amount : ℕ) :
  initial_amount = 87 →
  cheese_cost = 7 →
  cheese_quantity = 3 →
  beef_quantity = 1 →
  remaining_amount = 61 →
  initial_amount - remaining_amount - (cheese_cost * cheese_quantity) = 5 :=
by sorry

end NUMINAMATH_CALUDE_beef_cost_calculation_l2_279


namespace NUMINAMATH_CALUDE_pot_count_l2_211

/-- The number of pots given the number of flowers and sticks per pot and the total number of flowers and sticks -/
def number_of_pots (flowers_per_pot : ℕ) (sticks_per_pot : ℕ) (total_items : ℕ) : ℕ :=
  total_items / (flowers_per_pot + sticks_per_pot)

/-- Theorem stating that there are 466 pots given the conditions -/
theorem pot_count : number_of_pots 53 181 109044 = 466 := by
  sorry

#eval number_of_pots 53 181 109044

end NUMINAMATH_CALUDE_pot_count_l2_211


namespace NUMINAMATH_CALUDE_exterior_angle_HGI_exterior_angle_is_81_degrees_l2_286

-- Define the polygons
def Octagon : Type := Unit
def Decagon : Type := Unit

-- Define the properties of the polygons
axiom is_regular_octagon : Octagon → Prop
axiom is_regular_decagon : Decagon → Prop

-- Define the interior angles
def interior_angle_octagon (o : Octagon) (h : is_regular_octagon o) : ℝ := 135
def interior_angle_decagon (d : Decagon) (h : is_regular_decagon d) : ℝ := 144

-- Define the configuration
structure Configuration :=
  (o : Octagon)
  (d : Decagon)
  (ho : is_regular_octagon o)
  (hd : is_regular_decagon d)
  (share_side : Prop)

-- State the theorem
theorem exterior_angle_HGI (c : Configuration) : ℝ :=
  360 - interior_angle_octagon c.o c.ho - interior_angle_decagon c.d c.hd

-- The main theorem to prove
theorem exterior_angle_is_81_degrees (c : Configuration) :
  exterior_angle_HGI c = 81 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_HGI_exterior_angle_is_81_degrees_l2_286


namespace NUMINAMATH_CALUDE_min_sum_of_product_l2_239

theorem min_sum_of_product (a b c : ℕ+) (h : a * b * c = 2450) :
  ∃ (x y z : ℕ+), x * y * z = 2450 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 76 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l2_239


namespace NUMINAMATH_CALUDE_sharons_salary_increase_l2_259

theorem sharons_salary_increase (S : ℝ) (x : ℝ) : 
  S * 1.08 = 324 → S * (1 + x / 100) = 330 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sharons_salary_increase_l2_259


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2_249

theorem complex_fraction_equality : (1 + I : ℂ) / (2 - I) = 1/5 + 3/5 * I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2_249


namespace NUMINAMATH_CALUDE_labeling_existence_condition_l2_238

/-- A labeling of lattice points in Z^2 with positive integers -/
def Labeling := ℤ × ℤ → ℕ+

/-- The property that only finitely many distinct labels occur -/
def FiniteLabels (l : Labeling) : Prop :=
  ∃ (n : ℕ), ∀ (p : ℤ × ℤ), l p ≤ n

/-- The distance condition for a given c > 0 -/
def DistanceCondition (c : ℝ) (l : Labeling) : Prop :=
  ∀ (i : ℕ+) (p q : ℤ × ℤ), l p = i ∧ l q = i → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 : ℝ) ≥ c^(i : ℝ)

/-- The main theorem -/
theorem labeling_existence_condition (c : ℝ) :
  (c > 0 ∧
   ∃ (l : Labeling), FiniteLabels l ∧ DistanceCondition c l) ↔
  (c > 0 ∧ c < Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_labeling_existence_condition_l2_238


namespace NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l2_297

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l2_297


namespace NUMINAMATH_CALUDE_adjacent_complementary_angles_are_complementary_l2_264

/-- Two angles are complementary if their sum is 90 degrees -/
def Complementary (α β : ℝ) : Prop := α + β = 90

/-- Two angles are adjacent if they share a common vertex and a common side,
    but do not overlap -/
def Adjacent (α β : ℝ) : Prop := True  -- We simplify this for the purpose of the statement

theorem adjacent_complementary_angles_are_complementary
  (α β : ℝ) (h1 : Adjacent α β) (h2 : Complementary α β) :
  Complementary α β :=
sorry

end NUMINAMATH_CALUDE_adjacent_complementary_angles_are_complementary_l2_264


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l2_204

def x : ℕ := 9 * 36 * 54

theorem smallest_y_for_perfect_cube (y : ℕ) : 
  (∀ z < y, ∃ (a b : ℕ), x * z = a^3 → False) ∧
  (∃ (a : ℕ), x * y = a^3) →
  y = 9 := by
sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l2_204


namespace NUMINAMATH_CALUDE_intersection_m_value_l2_222

theorem intersection_m_value (x y : ℝ) (m : ℝ) : 
  (3 * x + y = m) →
  (-0.75 * x + y = -22) →
  (x = 6) →
  (m = 0.5) := by
  sorry

end NUMINAMATH_CALUDE_intersection_m_value_l2_222


namespace NUMINAMATH_CALUDE_arithmetic_sum_10_l2_236

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) : ℕ := n * (2 * n + 1)

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence with general term a_n = 2n + 1 is 120 -/
theorem arithmetic_sum_10 : arithmetic_sum 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_10_l2_236


namespace NUMINAMATH_CALUDE_sample_size_proof_l2_226

theorem sample_size_proof (n : ℕ) 
  (h1 : ∃ k : ℕ, 2*k + 3*k + 4*k = 27) 
  (h2 : ∃ k : ℕ, n = 2*k + 3*k + 4*k + 6*k + 4*k + k) : n = 60 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_proof_l2_226


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l2_291

/-- The number of vertices in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem pentadecagon_triangles : num_triangles = 455 := by sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l2_291


namespace NUMINAMATH_CALUDE_characterize_valid_functions_l2_255

def is_valid_function (f : ℤ → ℤ) : Prop :=
  f 1 ≠ f (-1) ∧ ∀ m n : ℤ, (f (m + n))^2 ∣ (f m - f n)

theorem characterize_valid_functions :
  ∀ f : ℤ → ℤ, is_valid_function f →
    (∀ x : ℤ, f x = 1 ∨ f x = -1) ∨
    (∀ x : ℤ, f x = 2 ∨ f x = -2) ∧ f 1 = -f (-1) :=
by sorry

end NUMINAMATH_CALUDE_characterize_valid_functions_l2_255


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2_256

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 2 → x > 3)) ↔ (∃ x : ℝ, x > 2 ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2_256


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2_203

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2_203


namespace NUMINAMATH_CALUDE_carrots_per_pound_l2_234

/-- Given the number of carrots in three beds and the total weight of the harvest,
    calculate the number of carrots that weigh one pound. -/
theorem carrots_per_pound 
  (bed1 bed2 bed3 : ℕ) 
  (total_weight : ℕ) 
  (h1 : bed1 = 55)
  (h2 : bed2 = 101)
  (h3 : bed3 = 78)
  (h4 : total_weight = 39) :
  (bed1 + bed2 + bed3) / total_weight = 6 := by
  sorry

#check carrots_per_pound

end NUMINAMATH_CALUDE_carrots_per_pound_l2_234


namespace NUMINAMATH_CALUDE_room_length_calculation_l2_233

theorem room_length_calculation (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 10 → width = 2 → area = length * width → length = 5 := by
sorry

end NUMINAMATH_CALUDE_room_length_calculation_l2_233


namespace NUMINAMATH_CALUDE_smallest_valid_survey_size_l2_276

def is_valid_survey_size (N : ℕ) : Prop :=
  (N * 1 / 10 : ℚ).num % (N * 1 / 10 : ℚ).den = 0 ∧
  (N * 3 / 10 : ℚ).num % (N * 3 / 10 : ℚ).den = 0 ∧
  (N * 2 / 5 : ℚ).num % (N * 2 / 5 : ℚ).den = 0

theorem smallest_valid_survey_size :
  ∃ (N : ℕ), N > 0 ∧ is_valid_survey_size N ∧ ∀ (M : ℕ), M > 0 ∧ is_valid_survey_size M → N ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_survey_size_l2_276


namespace NUMINAMATH_CALUDE_log_equality_implies_x_value_log_inequality_implies_x_range_l2_237

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the conditions
variable (a : ℝ)
variable (x : ℝ)
variable (h1 : a > 0)
variable (h2 : a ≠ 1)

-- Theorem 1
theorem log_equality_implies_x_value :
  log a (3*x + 1) = log a (-3*x) → x = -1/6 :=
by sorry

-- Theorem 2
theorem log_inequality_implies_x_range :
  log a (3*x + 1) > log a (-3*x) →
  ((0 < a ∧ a < 1 → -1/3 < x ∧ x < -1/6) ∧
   (a > 1 → -1/6 < x ∧ x < 0)) :=
by sorry

end NUMINAMATH_CALUDE_log_equality_implies_x_value_log_inequality_implies_x_range_l2_237


namespace NUMINAMATH_CALUDE_prop_A_sufficient_not_necessary_for_prop_B_l2_252

theorem prop_A_sufficient_not_necessary_for_prop_B :
  (∀ a b : ℝ, a < b ∧ b < 0 → a * b > b^2) ∧
  (∃ a b : ℝ, a * b > b^2 ∧ ¬(a < b ∧ b < 0)) :=
by sorry

end NUMINAMATH_CALUDE_prop_A_sufficient_not_necessary_for_prop_B_l2_252


namespace NUMINAMATH_CALUDE_fermat_quotient_perfect_square_no_fermat_quotient_perfect_square_l2_216

theorem fermat_quotient_perfect_square (p : ℕ) (h : Prime p) :
  (∃ (x : ℕ), (7^(p-1) - 1) / p = x^2) ↔ p = 3 :=
sorry

theorem no_fermat_quotient_perfect_square (p : ℕ) (h : Prime p) :
  ¬∃ (x : ℕ), (11^(p-1) - 1) / p = x^2 :=
sorry

end NUMINAMATH_CALUDE_fermat_quotient_perfect_square_no_fermat_quotient_perfect_square_l2_216


namespace NUMINAMATH_CALUDE_expression_value_l2_260

theorem expression_value : 6 * (3/2 + 2/3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2_260


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l2_253

theorem sin_cos_sum_equals_half : 
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (225 * π / 180) * Real.sin (15 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l2_253


namespace NUMINAMATH_CALUDE_expression_simplification_l2_220

theorem expression_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 - 3*x^3 =
  -x^3 - x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2_220


namespace NUMINAMATH_CALUDE_pete_reads_300_books_l2_263

/-- The number of books Pete reads across two years given the conditions -/
def petes_total_books (matts_second_year_books : ℕ) : ℕ :=
  let matts_first_year_books := matts_second_year_books * 2 / 3
  let petes_first_year_books := matts_first_year_books * 2
  let petes_second_year_books := petes_first_year_books * 2
  petes_first_year_books + petes_second_year_books

/-- Theorem stating that Pete reads 300 books across both years -/
theorem pete_reads_300_books : petes_total_books 75 = 300 := by
  sorry

end NUMINAMATH_CALUDE_pete_reads_300_books_l2_263


namespace NUMINAMATH_CALUDE_product_remainder_by_five_l2_254

theorem product_remainder_by_five : 
  (2685 * 4932 * 91406) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_five_l2_254


namespace NUMINAMATH_CALUDE_lion_weight_is_41_3_l2_268

/-- The weight of a lion in kilograms -/
def lion_weight : ℝ := 41.3

/-- The weight of a tiger in kilograms -/
def tiger_weight : ℝ := lion_weight - 4.8

/-- The weight of a panda in kilograms -/
def panda_weight : ℝ := tiger_weight - 7.7

/-- Theorem stating that the weight of a lion is 41.3 kg given the conditions -/
theorem lion_weight_is_41_3 : 
  lion_weight = 41.3 ∧ 
  tiger_weight = lion_weight - 4.8 ∧
  panda_weight = tiger_weight - 7.7 ∧
  lion_weight + tiger_weight + panda_weight = 106.6 := by
  sorry

#check lion_weight_is_41_3

end NUMINAMATH_CALUDE_lion_weight_is_41_3_l2_268


namespace NUMINAMATH_CALUDE_expression_value_l2_290

theorem expression_value : 3^(1^(2^8)) + ((3^1)^2)^8 = 43046724 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2_290


namespace NUMINAMATH_CALUDE_sqrt_two_plus_three_times_sqrt_two_minus_three_l2_285

theorem sqrt_two_plus_three_times_sqrt_two_minus_three : (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_three_times_sqrt_two_minus_three_l2_285


namespace NUMINAMATH_CALUDE_expression_simplification_l2_295

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 + 1) :
  (x + 1) / (x + 2) / (x - 2 + 3 / (x + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2_295


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2_221

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2) + f (x * y) = f x * f y + y * f x + x * f (x + y)

/-- The main theorem stating that any function satisfying the functional equation
    is either constantly zero or the negation function. -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = -x) := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l2_221


namespace NUMINAMATH_CALUDE_investment_proof_l2_202

def total_investment : ℝ := 3000
def part_one_investment : ℝ := 800
def part_one_interest_rate : ℝ := 0.10
def total_yearly_interest : ℝ := 256

theorem investment_proof :
  ∃ (part_two_interest_rate : ℝ),
    part_one_investment * part_one_interest_rate +
    (total_investment - part_one_investment) * part_two_interest_rate =
    total_yearly_interest :=
by sorry

end NUMINAMATH_CALUDE_investment_proof_l2_202


namespace NUMINAMATH_CALUDE_fourth_month_sale_problem_l2_284

/-- Calculates the sale in the fourth month given the sales of other months and the average --/
def fourthMonthSale (sale1 sale2 sale3 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Theorem stating the sale in the fourth month given the problem conditions --/
theorem fourth_month_sale_problem :
  fourthMonthSale 5420 5660 6200 6500 8270 6400 = 6350 := by
  sorry

#eval fourthMonthSale 5420 5660 6200 6500 8270 6400

end NUMINAMATH_CALUDE_fourth_month_sale_problem_l2_284


namespace NUMINAMATH_CALUDE_soap_cost_l2_213

/-- The cost of a bar of soap given monthly usage and two-year expenditure -/
theorem soap_cost (monthly_usage : ℕ) (two_year_expenditure : ℚ) :
  monthly_usage = 1 →
  two_year_expenditure = 96 →
  two_year_expenditure / (24 : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_soap_cost_l2_213


namespace NUMINAMATH_CALUDE_quadrilaterals_equal_area_l2_201

/-- Represents a quadrilateral on a geoboard -/
structure Quadrilateral where
  area : ℝ

/-- Quadrilateral I can be rearranged to form a 3x1 rectangle -/
def quadrilateral_I : Quadrilateral :=
  { area := 3 * 1 }

/-- Quadrilateral II can be rearranged to form two 1x1.5 rectangles -/
def quadrilateral_II : Quadrilateral :=
  { area := 2 * (1 * 1.5) }

/-- Theorem: Quadrilateral I and Quadrilateral II have the same area -/
theorem quadrilaterals_equal_area : quadrilateral_I.area = quadrilateral_II.area := by
  sorry

#check quadrilaterals_equal_area

end NUMINAMATH_CALUDE_quadrilaterals_equal_area_l2_201


namespace NUMINAMATH_CALUDE_candy_distribution_l2_223

theorem candy_distribution (S M L : ℕ) 
  (total : S + M + L = 110)
  (without_jelly : S + L = 100)
  (relation : M + L = S + M + 20) :
  S = 40 ∧ L = 60 ∧ M = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2_223


namespace NUMINAMATH_CALUDE_sidney_wednesday_jumping_jacks_l2_246

/-- The number of jumping jacks Sidney did on Wednesday -/
def sidney_wednesday : ℕ := sorry

/-- The total number of jumping jacks Sidney did -/
def sidney_total : ℕ := sorry

/-- The number of jumping jacks Brooke did -/
def brooke_total : ℕ := 438

theorem sidney_wednesday_jumping_jacks :
  sidney_wednesday = 40 ∧
  sidney_total = sidney_wednesday + 106 ∧
  brooke_total = 3 * sidney_total :=
by sorry

end NUMINAMATH_CALUDE_sidney_wednesday_jumping_jacks_l2_246


namespace NUMINAMATH_CALUDE_solve_equation_l2_270

theorem solve_equation : ∃ x : ℝ, 90 + (x * 12) / (180 / 3) = 91 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2_270


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2_240

/-- Given vectors a and b, prove that |a + 2b| = √7 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  a = (Real.cos (5 * π / 180), Real.sin (5 * π / 180)) →
  b = (Real.cos (65 * π / 180), Real.sin (65 * π / 180)) →
  ‖a + 2 • b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2_240


namespace NUMINAMATH_CALUDE_oreo_cheesecake_problem_l2_287

theorem oreo_cheesecake_problem (graham_boxes_initial : ℕ) (graham_boxes_per_cake : ℕ) (oreo_packets_per_cake : ℕ) (graham_boxes_leftover : ℕ) :
  graham_boxes_initial = 14 →
  graham_boxes_per_cake = 2 →
  oreo_packets_per_cake = 3 →
  graham_boxes_leftover = 4 →
  let cakes_made := (graham_boxes_initial - graham_boxes_leftover) / graham_boxes_per_cake
  ∃ oreo_packets_bought : ℕ, oreo_packets_bought = cakes_made * oreo_packets_per_cake :=
by sorry

end NUMINAMATH_CALUDE_oreo_cheesecake_problem_l2_287


namespace NUMINAMATH_CALUDE_double_elimination_tournament_players_l2_269

/-- Represents a double elimination tournament -/
structure DoubleEliminationTournament where
  num_players : ℕ
  num_matches : ℕ

/-- Theorem: In a double elimination tournament with 63 matches, there are 32 players -/
theorem double_elimination_tournament_players (t : DoubleEliminationTournament) 
  (h : t.num_matches = 63) : t.num_players = 32 := by
  sorry

end NUMINAMATH_CALUDE_double_elimination_tournament_players_l2_269


namespace NUMINAMATH_CALUDE_ball_arrangements_count_l2_244

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of different arrangements of placing 5 numbered balls into 3 boxes,
    where two boxes contain 2 balls each and one box contains 1 ball --/
def ball_arrangements : ℕ :=
  choose 3 2 * choose 5 2 * choose 3 2

theorem ball_arrangements_count : ball_arrangements = 90 := by sorry

end NUMINAMATH_CALUDE_ball_arrangements_count_l2_244


namespace NUMINAMATH_CALUDE_seating_arrangements_l2_212

/-- The number of seats on the bench -/
def total_seats : ℕ := 7

/-- The number of people to be seated -/
def people_to_seat : ℕ := 4

/-- The number of empty seats -/
def empty_seats : ℕ := total_seats - people_to_seat

/-- The total number of unrestricted seating arrangements -/
def total_arrangements : ℕ := 840

theorem seating_arrangements :
  (∃ (arrangements_with_adjacent : ℕ),
    arrangements_with_adjacent = total_arrangements - 24 ∧
    arrangements_with_adjacent = 816) ∧
  (∃ (arrangements_without_all_empty_adjacent : ℕ),
    arrangements_without_all_empty_adjacent = total_arrangements - 120 ∧
    arrangements_without_all_empty_adjacent = 720) := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2_212


namespace NUMINAMATH_CALUDE_hyperbola_dimensions_l2_219

/-- Proves that for a hyperbola with given conditions, a = 3 and b = 4 -/
theorem hyperbola_dimensions (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_conjugate : 2 * b = 8)
  (h_distance : a * b / Real.sqrt (a^2 + b^2) = 12/5) :
  a = 3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_dimensions_l2_219


namespace NUMINAMATH_CALUDE_f_properties_g_inequality_l2_230

/-- The function f(x) = a ln x + 1/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / x

/-- The function g(x) = f(x) - 1/x -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 1 / x

theorem f_properties (a : ℝ) :
  (a > 0 → (∃ (x : ℝ), x > 0 ∧ f a x = a - a * Real.log a ∧ ∀ y > 0, f a y ≥ f a x) ∧
            (¬∃ (M : ℝ), ∀ x > 0, f a x ≤ M)) ∧
  (a ≤ 0 → ¬∃ (x : ℝ), x > 0 ∧ (∀ y > 0, f a y ≥ f a x ∨ ∀ y > 0, f a y ≤ f a x)) :=
sorry

theorem g_inequality (m n : ℝ) (h1 : 0 < m) (h2 : m < n) :
  (g 1 n - g 1 m) / 2 > (n - m) / (n + m) :=
sorry

end NUMINAMATH_CALUDE_f_properties_g_inequality_l2_230


namespace NUMINAMATH_CALUDE_counterclockwise_notation_l2_280

/-- Represents the direction of rotation -/
inductive RotationDirection
  | Clockwise
  | Counterclockwise

/-- Represents a rotation with a direction and an angle -/
structure Rotation :=
  (direction : RotationDirection)
  (angle : ℝ)

/-- Notation for a rotation -/
def rotationNotation (r : Rotation) : ℝ :=
  match r.direction with
  | RotationDirection.Clockwise => r.angle
  | RotationDirection.Counterclockwise => -r.angle

theorem counterclockwise_notation 
  (h : rotationNotation { direction := RotationDirection.Clockwise, angle := 60 } = 60) :
  rotationNotation { direction := RotationDirection.Counterclockwise, angle := 15 } = -15 :=
by
  sorry

end NUMINAMATH_CALUDE_counterclockwise_notation_l2_280


namespace NUMINAMATH_CALUDE_remainder_problem_l2_209

theorem remainder_problem (s t u : ℕ) 
  (hs : s % 12 = 4)
  (ht : t % 12 = 5)
  (hu : u % 12 = 7)
  (hst : s > t)
  (htu : t > u) :
  ((s - t) + (t - u)) % 12 = 9 :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l2_209


namespace NUMINAMATH_CALUDE_alpha_value_l2_251

theorem alpha_value (α : Real) 
  (h1 : (1 - 4 * Real.sin α) / Real.tan α = Real.sqrt 3)
  (h2 : α ∈ Set.Ioo 0 (Real.pi / 2)) :
  α = Real.pi / 18 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2_251


namespace NUMINAMATH_CALUDE_motorboat_stream_speed_l2_278

/-- Proves that the speed of the stream is 3 kmph given the conditions of the motorboat problem -/
theorem motorboat_stream_speed 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (total_time : ℝ) 
  (h1 : boat_speed = 21) 
  (h2 : distance = 72) 
  (h3 : total_time = 7) :
  ∃ (stream_speed : ℝ), 
    stream_speed = 3 ∧ 
    distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed) = total_time :=
by
  sorry

end NUMINAMATH_CALUDE_motorboat_stream_speed_l2_278


namespace NUMINAMATH_CALUDE_lg_sum_equals_two_l2_214

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_two : 2 * lg 2 + lg 25 = 2 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_two_l2_214


namespace NUMINAMATH_CALUDE_expand_difference_of_squares_l2_250

theorem expand_difference_of_squares (x y : ℝ) : 
  (x - y + 1) * (x - y - 1) = x^2 - 2*x*y + y^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_difference_of_squares_l2_250


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_of_seven_l2_228

theorem least_four_digit_multiple_of_seven : 
  (∀ n : ℕ, n < 1001 → n % 7 ≠ 0 ∨ n < 1000) ∧ 1001 % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_of_seven_l2_228


namespace NUMINAMATH_CALUDE_unique_digits_for_divisibility_l2_231

-- Define the number 13xy45z as a function of x, y, z
def number (x y z : ℕ) : ℕ := 13000000 + x * 100000 + y * 10000 + 4500 + z

-- Define the divisibility condition
def is_divisible_by_792 (n : ℕ) : Prop := n % 792 = 0

-- Theorem statement
theorem unique_digits_for_divisibility :
  ∃! (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ is_divisible_by_792 (number x y z) ∧ x = 2 ∧ y = 3 ∧ z = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_digits_for_divisibility_l2_231


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2_265

theorem right_triangle_inequality (a b c : ℝ) (n : ℕ) 
  (h_right_triangle : a^2 = b^2 + c^2)
  (h_order : a > b ∧ b > c)
  (h_n : n > 2) : 
  a^n > b^n + c^n := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2_265


namespace NUMINAMATH_CALUDE_quadratic_root_one_iff_sum_coeffs_zero_l2_299

theorem quadratic_root_one_iff_sum_coeffs_zero (a b c : ℝ) :
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = 1) ↔ a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_one_iff_sum_coeffs_zero_l2_299


namespace NUMINAMATH_CALUDE_boys_joined_school_l2_235

theorem boys_joined_school (initial_boys final_boys : ℕ) 
  (h1 : initial_boys = 214)
  (h2 : final_boys = 1124) :
  final_boys - initial_boys = 910 := by
  sorry

end NUMINAMATH_CALUDE_boys_joined_school_l2_235
