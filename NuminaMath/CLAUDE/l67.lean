import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_l67_6713

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 23 → 
  a * b + b * c + a * c = 131 → 
  a^2 + b^2 + c^2 = 267 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l67_6713


namespace NUMINAMATH_CALUDE_square_difference_of_solutions_l67_6780

theorem square_difference_of_solutions (α β : ℝ) : 
  α ≠ β ∧ α^2 = 3*α + 1 ∧ β^2 = 3*β + 1 → (α - β)^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_solutions_l67_6780


namespace NUMINAMATH_CALUDE_order_of_powers_l67_6703

theorem order_of_powers : 
  let a : ℝ := (2/5: ℝ)^(3/5: ℝ)
  let b : ℝ := (2/5: ℝ)^(2/5: ℝ)
  let c : ℝ := (3/5: ℝ)^(2/5: ℝ)
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_order_of_powers_l67_6703


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l67_6767

open Real
open InnerProductSpace

theorem min_sum_squared_distances (a b c : EuclideanSpace ℝ (Fin 2)) 
  (ha : ‖a‖^2 = 4)
  (hb : ‖b‖^2 = 1)
  (hc : ‖c‖^2 = 9) :
  ∃ (min : ℝ), min = 2 ∧ 
    ∀ (x y z : EuclideanSpace ℝ (Fin 2)), 
      ‖x‖^2 = 4 → ‖y‖^2 = 1 → ‖z‖^2 = 9 →
      ‖x - y‖^2 + ‖x - z‖^2 + ‖y - z‖^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l67_6767


namespace NUMINAMATH_CALUDE_jackson_chairs_count_l67_6741

/-- The number of chairs Jackson needs to buy for his restaurant -/
def total_chairs (four_seat_tables six_seat_tables : ℕ) (seats_per_four_seat_table seats_per_six_seat_table : ℕ) : ℕ :=
  four_seat_tables * seats_per_four_seat_table + six_seat_tables * seats_per_six_seat_table

/-- Proof that Jackson needs to buy 96 chairs for his restaurant -/
theorem jackson_chairs_count :
  total_chairs 6 12 4 6 = 96 := by
  sorry

end NUMINAMATH_CALUDE_jackson_chairs_count_l67_6741


namespace NUMINAMATH_CALUDE_intercepted_arc_measure_l67_6777

/-- An equilateral triangle with a circle rolling along its side -/
structure TriangleWithCircle where
  /-- Side length of the equilateral triangle -/
  side : ℝ
  /-- Radius of the circle (equal to the height of the triangle) -/
  radius : ℝ
  /-- The radius is equal to the height of the equilateral triangle -/
  height_eq_radius : radius = side * Real.sqrt 3 / 2

/-- The theorem stating that the intercepted arc measure is 60° -/
theorem intercepted_arc_measure (tc : TriangleWithCircle) :
  let arc_measure := Real.pi / 3  -- 60° in radians
  ∃ (center : ℝ × ℝ) (point_on_side : ℝ × ℝ),
    arc_measure = Real.arccos ((point_on_side.1 - center.1) / tc.radius) :=
sorry

end NUMINAMATH_CALUDE_intercepted_arc_measure_l67_6777


namespace NUMINAMATH_CALUDE_mowgli_nuts_theorem_l67_6776

/-- The number of monkeys --/
def num_monkeys : ℕ := 5

/-- The number of nuts each monkey gathered initially --/
def nuts_per_monkey : ℕ := 8

/-- The number of nuts thrown by each monkey during the quarrel --/
def nuts_thrown_per_monkey : ℕ := num_monkeys - 1

/-- The total number of nuts thrown during the quarrel --/
def total_nuts_thrown : ℕ := num_monkeys * nuts_thrown_per_monkey

/-- The number of nuts Mowgli received --/
def nuts_received : ℕ := (num_monkeys * nuts_per_monkey) / 2

theorem mowgli_nuts_theorem :
  nuts_received = total_nuts_thrown :=
by sorry

end NUMINAMATH_CALUDE_mowgli_nuts_theorem_l67_6776


namespace NUMINAMATH_CALUDE_min_k_inequality_k_lower_bound_l67_6784

theorem min_k_inequality (x y z : ℝ) :
  (16/9 : ℝ) * (x^2 - x + 1) * (y^2 - y + 1) * (z^2 - z + 1) ≥ (x*y*z)^2 - x*y*z + 1 :=
by sorry

theorem k_lower_bound (k : ℝ) 
  (h : ∀ x y z : ℝ, k * (x^2 - x + 1) * (y^2 - y + 1) * (z^2 - z + 1) ≥ (x*y*z)^2 - x*y*z + 1) :
  k ≥ 16/9 :=
by sorry

end NUMINAMATH_CALUDE_min_k_inequality_k_lower_bound_l67_6784


namespace NUMINAMATH_CALUDE_intersection_unique_l67_6753

/-- Two lines in 2D space -/
def line1 (t : ℝ) : ℝ × ℝ := (4 + 3 * t, 1 - 2 * t)
def line2 (u : ℝ) : ℝ × ℝ := (-2 + 4 * u, 5 - u)

/-- The point of intersection -/
def intersection_point : ℝ × ℝ := (-2, 5)

/-- Theorem stating that the intersection_point is the unique point of intersection for the two lines -/
theorem intersection_unique :
  (∃ (t : ℝ), line1 t = intersection_point) ∧
  (∃ (u : ℝ), line2 u = intersection_point) ∧
  (∀ (p : ℝ × ℝ), (∃ (t : ℝ), line1 t = p) ∧ (∃ (u : ℝ), line2 u = p) → p = intersection_point) :=
by sorry

end NUMINAMATH_CALUDE_intersection_unique_l67_6753


namespace NUMINAMATH_CALUDE_lara_flowers_in_vase_l67_6715

def flowers_in_vase (total_flowers mom_flowers grandma_extra : ℕ) : ℕ :=
  total_flowers - (mom_flowers + (mom_flowers + grandma_extra))

theorem lara_flowers_in_vase :
  flowers_in_vase 52 15 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_lara_flowers_in_vase_l67_6715


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l67_6748

/-- Two arithmetic sequences a and b with their respective sums S and T -/
structure ArithmeticSequences where
  a : ℕ → ℚ
  b : ℕ → ℚ
  S : ℕ → ℚ
  T : ℕ → ℚ

/-- The ratio of sums S_n and T_n for any n -/
def sum_ratio (seq : ArithmeticSequences) : ℕ → ℚ :=
  fun n => seq.S n / seq.T n

/-- The given condition that S_n / T_n = (2n + 1) / (3n + 2) -/
def sum_ratio_condition (seq : ArithmeticSequences) : Prop :=
  ∀ n : ℕ, sum_ratio seq n = (2 * n + 1) / (3 * n + 2)

/-- The theorem to be proved -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequences) 
  (h : sum_ratio_condition seq) : 
  (seq.a 3 + seq.a 11 + seq.a 19) / (seq.b 7 + seq.b 15) = 129 / 130 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l67_6748


namespace NUMINAMATH_CALUDE_combined_weight_equals_3655_574_l67_6722

-- Define molar masses of elements
def mass_C : ℝ := 12.01
def mass_H : ℝ := 1.008
def mass_O : ℝ := 16.00
def mass_Na : ℝ := 22.99

-- Define molar masses of compounds
def mass_citric_acid : ℝ := 6 * mass_C + 8 * mass_H + 7 * mass_O
def mass_sodium_carbonate : ℝ := 2 * mass_Na + mass_C + 3 * mass_O
def mass_sodium_citrate : ℝ := 3 * mass_Na + 6 * mass_C + 5 * mass_H + 7 * mass_O
def mass_carbon_dioxide : ℝ := mass_C + 2 * mass_O
def mass_water : ℝ := 2 * mass_H + mass_O

-- Define number of moles for each substance
def moles_citric_acid : ℝ := 3
def moles_sodium_carbonate : ℝ := 4.5
def moles_sodium_citrate : ℝ := 9
def moles_carbon_dioxide : ℝ := 4.5
def moles_water : ℝ := 4.5

-- Theorem statement
theorem combined_weight_equals_3655_574 :
  moles_citric_acid * mass_citric_acid +
  moles_sodium_carbonate * mass_sodium_carbonate +
  moles_sodium_citrate * mass_sodium_citrate +
  moles_carbon_dioxide * mass_carbon_dioxide +
  moles_water * mass_water = 3655.574 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_equals_3655_574_l67_6722


namespace NUMINAMATH_CALUDE_hot_dogs_dinner_l67_6783

def hot_dogs_today : ℕ := 11
def hot_dogs_lunch : ℕ := 9

theorem hot_dogs_dinner : hot_dogs_today - hot_dogs_lunch = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_dinner_l67_6783


namespace NUMINAMATH_CALUDE_fixed_point_on_circle_l67_6789

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- The parabola x^2 = 12y -/
def on_parabola (p : Point) : Prop :=
  p.x^2 = 12 * p.y

/-- The line y = -3 -/
def on_line (p : Point) : Prop :=
  p.y = -3

/-- Check if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The circle is tangent to the line y = -3 -/
def tangent_to_line (c : Circle) : Prop :=
  c.center.y + c.radius = -3

/-- Main theorem -/
theorem fixed_point_on_circle :
  ∀ (c : Circle),
    on_parabola c.center →
    tangent_to_line c →
    on_circle ⟨0, 3⟩ c :=
sorry

end NUMINAMATH_CALUDE_fixed_point_on_circle_l67_6789


namespace NUMINAMATH_CALUDE_tangent_line_equation_l67_6729

/-- The curve S defined by y = x³ + 4 -/
def S : ℝ → ℝ := fun x ↦ x^3 + 4

/-- The point A -/
def A : ℝ × ℝ := (1, 5)

/-- The first possible tangent line equation: 3x - y - 2 = 0 -/
def tangent1 (x y : ℝ) : Prop := 3 * x - y - 2 = 0

/-- The second possible tangent line equation: 3x - 4y + 17 = 0 -/
def tangent2 (x y : ℝ) : Prop := 3 * x - 4 * y + 17 = 0

/-- Theorem: The tangent line to curve S passing through point A
    is either tangent1 or tangent2 -/
theorem tangent_line_equation :
  ∃ (x y : ℝ), (y = S x ∧ (x, y) ≠ A) →
  (∀ (h k : ℝ), tangent1 h k ∨ tangent2 h k ↔ 
    (k - A.2) / (h - A.1) = 3 * x^2 ∧ k = S h) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l67_6729


namespace NUMINAMATH_CALUDE_cube_sum_not_2016_l67_6762

theorem cube_sum_not_2016 (a b : ℤ) : a^3 + 5*b^3 ≠ 2016 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_not_2016_l67_6762


namespace NUMINAMATH_CALUDE_titu_andreescu_inequality_l67_6702

theorem titu_andreescu_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end NUMINAMATH_CALUDE_titu_andreescu_inequality_l67_6702


namespace NUMINAMATH_CALUDE_expression_equals_three_l67_6791

theorem expression_equals_three :
  |Real.sqrt 3 - 1| + (2023 - Real.pi)^0 - (-1/3)⁻¹ - 3 * Real.tan (30 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_three_l67_6791


namespace NUMINAMATH_CALUDE_hyperbola_equation_l67_6763

/-- Given an ellipse and a hyperbola with shared foci, prove the equation of the hyperbola -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    -- Ellipse equation
    (x^2 + 4*y^2 = 64) ∧
    -- Hyperbola shares foci with ellipse
    (a^2 + b^2 = 48) ∧
    -- Asymptote equation
    (x - Real.sqrt 3 * y = 0)) →
  -- Hyperbola equation
  x^2/36 - y^2/12 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l67_6763


namespace NUMINAMATH_CALUDE_valid_numbers_l67_6769

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  (n / 10000 % 10) * 5 = (n / 1000 % 10) ∧
  (n / 10000 % 10) * (n / 1000 % 10) * (n / 100 % 10) * (n / 10 % 10) * (n % 10) = 1000

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 15558 ∨ n = 15585 ∨ n = 15855 :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l67_6769


namespace NUMINAMATH_CALUDE_rectangle_existence_l67_6700

theorem rectangle_existence : ∃ (x y : ℝ), 
  (2 * (x + y) = 2 * (2 + 1) * 2) ∧ 
  (x * y = 2 * 1 * 2) ∧ 
  (x > 0) ∧ (y > 0) ∧
  (x = 3 + Real.sqrt 5) ∧ (y = 3 - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_existence_l67_6700


namespace NUMINAMATH_CALUDE_min_value_of_function_l67_6717

theorem min_value_of_function (a : ℝ) (h : a > 1) :
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧
  ∀ x > 1, x + x^2 / (x - 1) ≥ min_val ∧
  ∃ y > 1, y + y^2 / (y - 1) = min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l67_6717


namespace NUMINAMATH_CALUDE_geometric_subsequence_exists_l67_6760

/-- An arithmetic progression with first term 1 -/
def ArithmeticProgression (d : ℕ) : ℕ → ℕ :=
  fun n => 1 + (n - 1) * d

/-- A geometric progression -/
def GeometricProgression (a : ℕ) : ℕ → ℕ :=
  fun k => a^k

theorem geometric_subsequence_exists :
  ∃ (d a : ℕ), ∃ (start : ℕ),
    (∀ k, k ∈ Finset.range 2015 →
      ArithmeticProgression d (start + k) = GeometricProgression a (k + 1)) :=
sorry

end NUMINAMATH_CALUDE_geometric_subsequence_exists_l67_6760


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l67_6716

theorem sum_of_absolute_values_zero (a b : ℝ) :
  |a - 5| + |b + 8| = 0 → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l67_6716


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l67_6773

/-- The total number of boxes in the game -/
def total_boxes : ℕ := 30

/-- The number of boxes containing at least $50,000 -/
def high_value_boxes : ℕ := 9

/-- The target probability (50%) expressed as a fraction -/
def target_probability : ℚ := 1/2

/-- The minimum number of boxes that need to be eliminated -/
def boxes_to_eliminate : ℕ := 12

theorem deal_or_no_deal_probability :
  boxes_to_eliminate = total_boxes - 2 * high_value_boxes :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l67_6773


namespace NUMINAMATH_CALUDE_product_in_fourth_quadrant_l67_6745

def z₁ : ℂ := 3 + Complex.I
def z₂ : ℂ := 1 - Complex.I

theorem product_in_fourth_quadrant :
  let z := z₁ * z₂
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_product_in_fourth_quadrant_l67_6745


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l67_6792

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℚ)
  (h_geometric : geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a6 : a 6 = 1/8) :
  ∃ q : ℚ, (q = 1/2 ∨ q = -1/2) ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l67_6792


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_shifted_l67_6775

def f (x : ℝ) : ℝ := x^2 + 2*x - 5

theorem decreasing_interval_of_f_shifted :
  let g := fun (x : ℝ) => f (x - 1)
  ∀ x y : ℝ, x < y ∧ y ≤ 0 → g x > g y :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_shifted_l67_6775


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l67_6739

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros : 
  let a : ℕ := 35
  let b : ℕ := 4900
  let a_factorization := 5 * 7
  let b_factorization := 2^2 * 5^2 * 7^2
  trailing_zeros (a * b) = 2 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l67_6739


namespace NUMINAMATH_CALUDE_triangle_solution_l67_6737

/-- Given a triangle ABC with side lengths a, b, c and angles α, β, γ,
    if a : b = 1 : 2, α : β = 1 : 3, and c = 5 cm,
    then a = 5√3/3 cm, b = 10√3/3 cm, α = 30°, β = 90°, and γ = 60°. -/
theorem triangle_solution (a b c : ℝ) (α β γ : ℝ) : 
  a / b = 1 / 2 →
  α / β = 1 / 3 →
  c = 5 →
  a = 5 * Real.sqrt 3 / 3 ∧
  b = 10 * Real.sqrt 3 / 3 ∧
  α = Real.pi / 6 ∧
  β = Real.pi / 2 ∧
  γ = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_solution_l67_6737


namespace NUMINAMATH_CALUDE_field_trip_total_l67_6756

theorem field_trip_total (num_vans num_buses students_per_van students_per_bus teachers_per_van teachers_per_bus : ℕ) 
  (h1 : num_vans = 6)
  (h2 : num_buses = 8)
  (h3 : students_per_van = 6)
  (h4 : students_per_bus = 18)
  (h5 : teachers_per_van = 1)
  (h6 : teachers_per_bus = 2) :
  num_vans * students_per_van + num_buses * students_per_bus + 
  num_vans * teachers_per_van + num_buses * teachers_per_bus = 202 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_total_l67_6756


namespace NUMINAMATH_CALUDE_greatest_valid_number_l67_6786

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  ∃ k : ℕ, n = 9 * k + 2 ∧
  ∃ m : ℕ, n = 5 * m + 3

theorem greatest_valid_number : 
  is_valid_number 9962 ∧ ∀ n : ℕ, is_valid_number n → n ≤ 9962 :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_number_l67_6786


namespace NUMINAMATH_CALUDE_unique_sequence_existence_l67_6772

theorem unique_sequence_existence :
  ∃! a : ℕ → ℝ,
    (∀ n, a n > 0) ∧
    a 0 = 1 ∧
    (∀ n : ℕ, a (n + 1) = a (n - 1) - a n) :=
by sorry

end NUMINAMATH_CALUDE_unique_sequence_existence_l67_6772


namespace NUMINAMATH_CALUDE_log_sqrt7_343sqrt7_equals_7_l67_6730

theorem log_sqrt7_343sqrt7_equals_7 :
  Real.log (343 * Real.sqrt 7) / Real.log (Real.sqrt 7) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt7_343sqrt7_equals_7_l67_6730


namespace NUMINAMATH_CALUDE_divisible_by_45_digits_l67_6728

theorem divisible_by_45_digits (a b : Nat) : 
  a < 10 → b < 10 → (72000 + 100 * a + 30 + b) % 45 = 0 → 
  ((a = 6 ∧ b = 0) ∨ (a = 1 ∧ b = 5)) := by
sorry

end NUMINAMATH_CALUDE_divisible_by_45_digits_l67_6728


namespace NUMINAMATH_CALUDE_xy_fraction_sum_l67_6788

theorem xy_fraction_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y + x * y = 1) :
  x * y + 1 / (x * y) - y / x - x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_fraction_sum_l67_6788


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_l67_6799

theorem sum_mod_thirteen : (9010 + 9011 + 9012 + 9013 + 9014) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_l67_6799


namespace NUMINAMATH_CALUDE_geometric_series_problem_l67_6731

theorem geometric_series_problem (b₁ q : ℝ) (h_decrease : |q| < 1) : 
  (b₁ / (1 - q^2) = 2 + b₁ * q / (1 - q^2)) →
  (b₁^2 / (1 - q^4) - b₁^2 * q^2 / (1 - q^4) = 36/5) →
  (b₁ = 3 ∧ q = 1/2) := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l67_6731


namespace NUMINAMATH_CALUDE_solution_system_l67_6735

theorem solution_system (x y : ℝ) 
  (eq1 : ⌊x⌋ + (y - ⌊y⌋) = 7.2)
  (eq2 : (x - ⌊x⌋) + ⌊y⌋ = 10.3) : 
  |x - y| = 2.9 := by
  sorry

end NUMINAMATH_CALUDE_solution_system_l67_6735


namespace NUMINAMATH_CALUDE_average_matches_is_four_l67_6727

/-- Represents the distribution of matches played in a badminton club --/
structure MatchDistribution :=
  (one_match : Nat)
  (two_matches : Nat)
  (four_matches : Nat)
  (six_matches : Nat)
  (eight_matches : Nat)

/-- Calculates the average number of matches played, rounded to the nearest whole number --/
def averageMatchesPlayed (d : MatchDistribution) : Nat :=
  let totalMatches := d.one_match * 1 + d.two_matches * 2 + d.four_matches * 4 + d.six_matches * 6 + d.eight_matches * 8
  let totalPlayers := d.one_match + d.two_matches + d.four_matches + d.six_matches + d.eight_matches
  let average := totalMatches / totalPlayers
  if totalMatches % totalPlayers >= totalPlayers / 2 then average + 1 else average

/-- The specific distribution of matches in the badminton club --/
def clubDistribution : MatchDistribution :=
  { one_match := 4
  , two_matches := 3
  , four_matches := 2
  , six_matches := 2
  , eight_matches := 8 }

theorem average_matches_is_four :
  averageMatchesPlayed clubDistribution = 4 := by sorry

end NUMINAMATH_CALUDE_average_matches_is_four_l67_6727


namespace NUMINAMATH_CALUDE_finitely_many_odd_divisors_l67_6734

theorem finitely_many_odd_divisors (k : ℕ+) :
  (∃ c : ℕ, k + 1 = 2^c) ↔
  (∃ S : Finset ℕ, ∀ n : ℕ, n % 2 = 1 → (n ∣ k^n + 1) → n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_finitely_many_odd_divisors_l67_6734


namespace NUMINAMATH_CALUDE_green_ball_probability_l67_6709

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a container -/
def containerProb : ℚ := 1 / 3

/-- Calculate the probability of drawing a green ball from a container -/
def greenProb (c : Container) : ℚ := c.green / (c.red + c.green)

/-- The three containers A, B, and C -/
def containerA : Container := ⟨5, 5⟩
def containerB : Container := ⟨8, 2⟩
def containerC : Container := ⟨3, 7⟩

/-- The probability of selecting a green ball -/
def probGreenBall : ℚ :=
  containerProb * greenProb containerA +
  containerProb * greenProb containerB +
  containerProb * greenProb containerC

theorem green_ball_probability :
  probGreenBall = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l67_6709


namespace NUMINAMATH_CALUDE_spatial_relationships_l67_6746

-- Define the basic concepts
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships
def intersect (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def line_of_intersection (p1 p2 : Plane) : Line := sorry

-- State the propositions
def proposition_1 (l1 l2 l3 l4 : Line) : Prop :=
  intersect l1 l3 → intersect l2 l4 → skew l3 l4 → skew l1 l2

def proposition_2 (l1 l2 : Line) (p1 p2 : Plane) : Prop :=
  parallel_planes p1 p2 → parallel_lines l1 p1 → parallel_lines l2 p2 → parallel_lines l1 l2

def proposition_3 (l1 l2 : Line) (p : Plane) : Prop :=
  perpendicular_to_plane l1 p → perpendicular_to_plane l2 p → parallel_lines l1 l2

def proposition_4 (p1 p2 : Plane) (l : Line) : Prop :=
  perpendicular_planes p1 p2 →
  line_in_plane l p1 →
  ¬perpendicular_to_plane l (line_of_intersection p1 p2) →
  ¬perpendicular_to_plane l p2

theorem spatial_relationships :
  (∀ l1 l2 l3 l4 : Line, ¬proposition_1 l1 l2 l3 l4) ∧
  (∀ l1 l2 : Line, ∀ p1 p2 : Plane, ¬proposition_2 l1 l2 p1 p2) ∧
  (∀ l1 l2 : Line, ∀ p : Plane, proposition_3 l1 l2 p) ∧
  (∀ p1 p2 : Plane, ∀ l : Line, proposition_4 p1 p2 l) :=
sorry

end NUMINAMATH_CALUDE_spatial_relationships_l67_6746


namespace NUMINAMATH_CALUDE_pricing_scenario_l67_6764

/-- The number of articles in a pricing scenario -/
def num_articles : ℕ := 50

/-- The number of articles used for selling price comparison -/
def comparison_articles : ℕ := 45

/-- The gain percentage as a rational number -/
def gain_percentage : ℚ := 1 / 9

theorem pricing_scenario :
  (∀ (cost_price selling_price : ℚ),
    cost_price * num_articles = selling_price * comparison_articles →
    selling_price = cost_price * (1 + gain_percentage)) →
  num_articles = 50 :=
sorry

end NUMINAMATH_CALUDE_pricing_scenario_l67_6764


namespace NUMINAMATH_CALUDE_total_length_eleven_segments_l67_6797

/-- The total length of 11 congruent segments -/
def total_length (segment_length : ℝ) (num_segments : ℕ) : ℝ :=
  segment_length * (num_segments : ℝ)

/-- Theorem: The total length of 11 congruent segments of 7 cm each is 77 cm -/
theorem total_length_eleven_segments :
  total_length 7 11 = 77 := by sorry

end NUMINAMATH_CALUDE_total_length_eleven_segments_l67_6797


namespace NUMINAMATH_CALUDE_expression_value_l67_6740

theorem expression_value (x y : ℝ) (h1 : x + y = 17) (h2 : x * y = 17) :
  (x^2 - 17*x) * (y + 17/y) = -289 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l67_6740


namespace NUMINAMATH_CALUDE_exists_valid_distribution_with_plate_B_size_l67_6726

/-- Represents a distribution of balls across three plates -/
structure BallDistribution where
  plateA : List Nat
  plateB : List Nat
  plateC : List Nat

/-- Checks if a given distribution satisfies the problem conditions -/
def isValidDistribution (d : BallDistribution) : Prop :=
  let allBalls := d.plateA ++ d.plateB ++ d.plateC
  (∀ n ∈ allBalls, 1 ≤ n ∧ n ≤ 15) ∧ 
  (allBalls.length = 15) ∧
  (d.plateA.length ≥ 4 ∧ d.plateB.length ≥ 4 ∧ d.plateC.length ≥ 4) ∧
  ((d.plateA.sum : Rat) / d.plateA.length = 3) ∧
  ((d.plateB.sum : Rat) / d.plateB.length = 8) ∧
  ((d.plateC.sum : Rat) / d.plateC.length = 13)

/-- The main theorem to be proved -/
theorem exists_valid_distribution_with_plate_B_size :
  ∃ d : BallDistribution, isValidDistribution d ∧ (d.plateB.length = 7 ∨ d.plateB.length = 5) := by
  sorry


end NUMINAMATH_CALUDE_exists_valid_distribution_with_plate_B_size_l67_6726


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l67_6795

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum of first n terms
  h1 : a 1 < 0
  h2 : a 10 + a 15 = a 12
  h3 : ∀ n, a n = a 1 + (n - 1) * d
  h4 : ∀ n, S n = n * a 1 + n * (n - 1) / 2 * d

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n m, n < m → seq.a n < seq.a m) ∧
  (∀ n, n ≠ 12 ∧ n ≠ 13 → seq.S 12 ≤ seq.S n ∧ seq.S 13 ≤ seq.S n) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l67_6795


namespace NUMINAMATH_CALUDE_lower_bound_second_inequality_l67_6757

theorem lower_bound_second_inequality (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : ∃ n, n < x ∧ x < 18)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) :
  ∀ n, n < x → n ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_lower_bound_second_inequality_l67_6757


namespace NUMINAMATH_CALUDE_f_sin_75_eq_zero_l67_6765

-- Define the function f
def f (a₄ a₃ a₂ a₁ a₀ : ℤ) (x : ℝ) : ℝ :=
  a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

-- State the theorem
theorem f_sin_75_eq_zero 
  (a₄ a₃ a₂ a₁ a₀ : ℤ) 
  (h₁ : f a₄ a₃ a₂ a₁ a₀ (Real.cos (75 * π / 180)) = 0) 
  (h₂ : a₄ ≠ 0) : 
  f a₄ a₃ a₂ a₁ a₀ (Real.sin (75 * π / 180)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sin_75_eq_zero_l67_6765


namespace NUMINAMATH_CALUDE_min_sum_of_product_36_l67_6710

theorem min_sum_of_product_36 (a b : ℤ) (h : a * b = 36) : 
  ∀ (x y : ℤ), x * y = 36 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 36 ∧ a₀ + b₀ = -37 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_36_l67_6710


namespace NUMINAMATH_CALUDE_ship_speed_upstream_l67_6778

/-- Given a ship traveling downstream at 26 km/h and a water flow speed of v km/h,
    the speed of the ship traveling upstream is 26 - 2v km/h. -/
theorem ship_speed_upstream 
  (v : ℝ) -- Water flow speed in km/h
  (h1 : v > 0) -- Assumption that water flow speed is positive
  (h2 : v < 26) -- Assumption that water flow speed is less than downstream speed
  : ℝ :=
  26 - 2 * v

#check ship_speed_upstream

end NUMINAMATH_CALUDE_ship_speed_upstream_l67_6778


namespace NUMINAMATH_CALUDE_count_multiples_of_three_is_12960_l67_6744

/-- A function that returns the count of six-digit multiples of 3 where each digit is not greater than 5 -/
def count_multiples_of_three : ℕ :=
  let first_digit_options := 5  -- digits 1 to 5
  let other_digit_options := 6  -- digits 0 to 5
  let last_digit_options := 2   -- two options to make the sum divisible by 3
  first_digit_options * (other_digit_options ^ 4) * last_digit_options

/-- Theorem stating that the count of six-digit multiples of 3 where each digit is not greater than 5 is 12960 -/
theorem count_multiples_of_three_is_12960 : count_multiples_of_three = 12960 := by
  sorry

#eval count_multiples_of_three

end NUMINAMATH_CALUDE_count_multiples_of_three_is_12960_l67_6744


namespace NUMINAMATH_CALUDE_cone_altitude_to_radius_ratio_l67_6793

/-- The ratio of a cone's altitude to its base radius, given that its volume is one-third of a sphere with the same radius -/
theorem cone_altitude_to_radius_ratio (r h : ℝ) (h_pos : 0 < r) : 
  (1 / 3 * π * r^2 * h = 1 / 3 * (4 / 3 * π * r^3)) → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_altitude_to_radius_ratio_l67_6793


namespace NUMINAMATH_CALUDE_f_less_than_three_zeros_l67_6771

/-- The cubic function f(x) = x³ - ax² + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

/-- Theorem: f(x) has less than 3 zeros if and only if a ≤ 3 -/
theorem f_less_than_three_zeros (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → (∃ y z : ℝ, y ≠ z ∧ f a y = 0 ∧ f a z = 0 → False)) ↔ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_f_less_than_three_zeros_l67_6771


namespace NUMINAMATH_CALUDE_square_area_increase_l67_6701

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.25 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l67_6701


namespace NUMINAMATH_CALUDE_figure_reassemble_to_square_l67_6724

/-- Represents a figure on a graph paper --/
structure GraphFigure where
  area : ℝ
  triangles : ℕ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Function to check if a figure can be reassembled into a square --/
def can_reassemble_to_square (figure : GraphFigure) (square : Square) : Prop :=
  figure.area = square.side ^ 2 ∧ figure.triangles = 5

/-- Theorem stating that the given figure can be reassembled into a square --/
theorem figure_reassemble_to_square :
  ∃ (figure : GraphFigure) (square : Square),
    figure.area = 20 ∧ can_reassemble_to_square figure square :=
by sorry

end NUMINAMATH_CALUDE_figure_reassemble_to_square_l67_6724


namespace NUMINAMATH_CALUDE_one_third_to_fifth_power_l67_6759

theorem one_third_to_fifth_power :
  (1 / 3 : ℚ) ^ 5 = 1 / 243 := by sorry

end NUMINAMATH_CALUDE_one_third_to_fifth_power_l67_6759


namespace NUMINAMATH_CALUDE_triangle_ratio_l67_6711

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  A = Real.pi / 3 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
sorry


end NUMINAMATH_CALUDE_triangle_ratio_l67_6711


namespace NUMINAMATH_CALUDE_prime_power_equation_solutions_l67_6736

theorem prime_power_equation_solutions :
  ∀ p n : ℕ,
    Nat.Prime p →
    n > 0 →
    p^3 - 2*p^2 + p + 1 = 3^n →
    ((p = 2 ∧ n = 1) ∨ (p = 5 ∧ n = 4)) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_equation_solutions_l67_6736


namespace NUMINAMATH_CALUDE_f_five_eq_zero_l67_6720

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f x

-- State the theorem
theorem f_five_eq_zero : f 5 = 0 := by sorry

end NUMINAMATH_CALUDE_f_five_eq_zero_l67_6720


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l67_6796

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit in one direction -/
def tilesInOneDimension (floorSize tileSize : ℕ) : ℕ :=
  floorSize / tileSize

/-- Calculates the total number of tiles for a given orientation -/
def totalTiles (floor tile : Dimensions) : ℕ :=
  (tilesInOneDimension floor.length tile.length) * (tilesInOneDimension floor.width tile.width)

/-- Theorem stating the maximum number of tiles that can be accommodated -/
theorem max_tiles_on_floor (floor : Dimensions) (tile : Dimensions) 
    (h_floor : floor = ⟨1000, 210⟩) (h_tile : tile = ⟨35, 30⟩) :
  max (totalTiles floor tile) (totalTiles floor ⟨tile.width, tile.length⟩) = 198 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l67_6796


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l67_6755

-- Define the vectors
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

-- Theorem statement
theorem parallel_vectors_x_value :
  parallel a (b x) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l67_6755


namespace NUMINAMATH_CALUDE_upper_limit_correct_l67_6725

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def upper_limit : ℕ := 7533

theorem upper_limit_correct :
  ∀ h : ℕ, h > 0 ∧ digit_product h = 210 → h < upper_limit :=
by sorry

end NUMINAMATH_CALUDE_upper_limit_correct_l67_6725


namespace NUMINAMATH_CALUDE_smallest_fruit_distribution_l67_6706

theorem smallest_fruit_distribution (N : ℕ) : N = 79 ↔ 
  N > 0 ∧
  (N - 1) % 3 = 0 ∧
  (2 * (N - 1) / 3 - 1) % 3 = 0 ∧
  ((2 * N - 5) / 3 - 1) % 3 = 0 ∧
  ((4 * N - 28) / 9 - 1) % 3 = 0 ∧
  ((8 * N - 56) / 27 - 1) % 3 = 0 ∧
  ∀ (M : ℕ), M < N → 
    (M > 0 ∧
    (M - 1) % 3 = 0 ∧
    (2 * (M - 1) / 3 - 1) % 3 = 0 ∧
    ((2 * M - 5) / 3 - 1) % 3 = 0 ∧
    ((4 * M - 28) / 9 - 1) % 3 = 0 ∧
    ((8 * M - 56) / 27 - 1) % 3 = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_smallest_fruit_distribution_l67_6706


namespace NUMINAMATH_CALUDE_contrapositive_x_squared_greater_than_one_l67_6785

theorem contrapositive_x_squared_greater_than_one (x : ℝ) : 
  x ≤ 1 → x^2 ≤ 1 := by sorry

end NUMINAMATH_CALUDE_contrapositive_x_squared_greater_than_one_l67_6785


namespace NUMINAMATH_CALUDE_max_value_of_z_l67_6723

-- Define the system of inequalities and z
def system (x y : ℝ) : Prop :=
  x + y - Real.sqrt 2 ≤ 0 ∧
  x - y + Real.sqrt 2 ≥ 0 ∧
  y ≥ 0

def z (x y : ℝ) : ℝ := 2 * x - y

-- State the theorem
theorem max_value_of_z :
  ∃ (max_z : ℝ) (x_max y_max : ℝ),
    system x_max y_max ∧
    z x_max y_max = max_z ∧
    max_z = 2 * Real.sqrt 2 ∧
    x_max = Real.sqrt 2 ∧
    y_max = 0 ∧
    ∀ (x y : ℝ), system x y → z x y ≤ max_z :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_z_l67_6723


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_properties_l67_6742

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The asymptote of a hyperbola -/
def asymptote (h : Hyperbola) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

/-- The focus of the parabola y^2 = 16x -/
def parabola_focus : ℝ × ℝ := (4, 0)

/-- Theorem: If a hyperbola has the given properties, its equation is x^2/4 - y^2/12 = 1 -/
theorem hyperbola_equation_from_properties (h : Hyperbola) :
  (∃ x y : ℝ, asymptote h x y) →
  (∃ x y : ℝ, hyperbola_equation h x y ∧ (x, y) = parabola_focus) →
  (∀ x y : ℝ, hyperbola_equation h x y ↔ x^2/4 - y^2/12 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_properties_l67_6742


namespace NUMINAMATH_CALUDE_set_operations_and_range_l67_6743

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- State the theorem
theorem set_operations_and_range :
  (∃ (a : ℝ),
    (A ∩ B = {x | 2 ≤ x ∧ x < 3}) ∧
    ((U \ A) ∪ B = {x | x < -1 ∨ x ≥ 2}) ∧
    (B ∪ C a = C a → a > 4)) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l67_6743


namespace NUMINAMATH_CALUDE_min_cans_for_drinks_l67_6712

/-- Represents the available can sizes in liters -/
inductive CanSize
  | half
  | one
  | two

/-- Calculates the number of cans needed for a given volume and can size -/
def cansNeeded (volume : ℕ) (size : CanSize) : ℕ :=
  match size with
  | CanSize.half => volume * 2
  | CanSize.one => volume
  | CanSize.two => volume / 2

/-- Finds the minimum number of cans needed for a given volume -/
def minCansForVolume (volume : ℕ) : ℕ :=
  min (cansNeeded volume CanSize.half)
    (min (cansNeeded volume CanSize.one)
      (cansNeeded volume CanSize.two))

/-- The main theorem stating the minimum number of cans required -/
theorem min_cans_for_drinks :
  minCansForVolume 60 +
  minCansForVolume 220 +
  minCansForVolume 500 +
  minCansForVolume 315 +
  minCansForVolume 125 = 830 := by
  sorry


end NUMINAMATH_CALUDE_min_cans_for_drinks_l67_6712


namespace NUMINAMATH_CALUDE_quadrilateral_sum_l67_6766

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a quadrilateral PQRS -/
structure Quadrilateral where
  P : Point
  Q : Point
  R : Point
  S : Point

def area (q : Quadrilateral) : ℚ :=
  sorry  -- Area calculation implementation

theorem quadrilateral_sum (a b : ℤ) :
  a > b ∧ b > 0 →
  let q := Quadrilateral.mk
    (Point.mk (2*a) (2*b))
    (Point.mk (2*b) (2*a))
    (Point.mk (-2*a) (-2*b))
    (Point.mk (-2*b) (-2*a))
  area q = 32 →
  a + b = 4 := by
    sorry

end NUMINAMATH_CALUDE_quadrilateral_sum_l67_6766


namespace NUMINAMATH_CALUDE_number_sequence_count_l67_6761

theorem number_sequence_count : ∀ (N : ℕ) (S : ℝ),
  S / N = 44 →
  (11 * 48 + 11 * 41 - 55) / N = 44 →
  N = 21 := by
sorry

end NUMINAMATH_CALUDE_number_sequence_count_l67_6761


namespace NUMINAMATH_CALUDE_bounded_sequence_characterization_l67_6768

def sequence_rule (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) = (a (n + 1) + a n) / (Nat.gcd (a (n + 1)) (a n))

def is_bounded (a : ℕ → ℕ) : Prop :=
  ∃ M, ∀ n, a n ≤ M

theorem bounded_sequence_characterization (a : ℕ → ℕ) :
  (∀ n, a n > 0) →
  sequence_rule a →
  is_bounded a ↔ a 1 = 2 ∧ a 2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_bounded_sequence_characterization_l67_6768


namespace NUMINAMATH_CALUDE_point_coordinates_l67_6721

/-- A point in the second quadrant with specific x and y values -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  second_quadrant : x < 0 ∧ y > 0
  x_abs : |x| = 2
  y_squared : y^2 = 1

/-- The coordinates of the point P are (-2, 1) -/
theorem point_coordinates (P : SecondQuadrantPoint) : P.x = -2 ∧ P.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l67_6721


namespace NUMINAMATH_CALUDE_certain_number_problem_l67_6758

theorem certain_number_problem : ∃ x : ℝ, 45 * x = 0.45 * 900 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l67_6758


namespace NUMINAMATH_CALUDE_injective_function_characterization_l67_6749

theorem injective_function_characterization (f : ℤ → ℤ) :
  Function.Injective f ∧ (∀ x y : ℤ, |f x - f y| ≤ |x - y|) →
  ∃ a : ℤ, (∀ x : ℤ, f x = a + x) ∨ (∀ x : ℤ, f x = a - x) :=
by sorry

end NUMINAMATH_CALUDE_injective_function_characterization_l67_6749


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l67_6719

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let perimeter : ℝ := 3 * side_length
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  area / perimeter = (5 * Real.sqrt 3) / 6 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l67_6719


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l67_6718

theorem square_sum_given_difference_and_product (a b : ℝ) 
  (h1 : a - b = 2) 
  (h2 : a * b = 10.5) : 
  a^2 + b^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l67_6718


namespace NUMINAMATH_CALUDE_correct_total_paths_l67_6708

/-- The number of paths from Wolfburg to the Green Meadows -/
def paths_wolfburg_to_meadows : ℕ := 6

/-- The number of paths from the Green Meadows to Sheep Village -/
def paths_meadows_to_village : ℕ := 20

/-- Wolfburg and Sheep Village are separated by the Green Meadows -/
axiom separated_by_meadows : True

/-- The number of different ways to travel from Wolfburg to Sheep Village -/
def total_paths : ℕ := paths_wolfburg_to_meadows * paths_meadows_to_village

theorem correct_total_paths : total_paths = 120 := by sorry

end NUMINAMATH_CALUDE_correct_total_paths_l67_6708


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l67_6774

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 500! has 124 trailing zeroes -/
theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l67_6774


namespace NUMINAMATH_CALUDE_fixed_circle_theorem_l67_6704

noncomputable section

-- Define the hyperbola C
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / (3 * a^2) = 1

-- Define the foci F₁ and F₂
def F₁ (a : ℝ) : ℝ × ℝ := (-2, 0)
def F₂ (a : ℝ) : ℝ × ℝ := (2, 0)

-- Define the distance from F₂ to the asymptote
def distance_to_asymptote (a : ℝ) : ℝ := Real.sqrt 3

-- Define a line passing through the left vertex and not coinciding with x-axis
def line_through_left_vertex (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define the intersection point B
def point_B (a k : ℝ) : ℝ × ℝ := ((3 + k^2) / (3 - k^2), 6 * k / (3 - k^2))

-- Define the intersection point P
def point_P (k : ℝ) : ℝ × ℝ := (1/2, k * 3/2)

-- Define the line parallel to PF₂ passing through F₁
def parallel_line (k : ℝ) (x : ℝ) : ℝ := -k * (x + 2)

-- Define the theorem
theorem fixed_circle_theorem (a : ℝ) (k : ℝ) :
  a > 0 →
  ∀ Q : ℝ × ℝ,
  (∃ x, Q.1 = x ∧ Q.2 = line_through_left_vertex k x) →
  (∃ x, Q.1 = x ∧ Q.2 = parallel_line k x) →
  (Q.1 - (F₂ a).1)^2 + (Q.2 - (F₂ a).2)^2 = 16 :=
by sorry

end

end NUMINAMATH_CALUDE_fixed_circle_theorem_l67_6704


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l67_6738

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l67_6738


namespace NUMINAMATH_CALUDE_insertion_sort_comparison_bounds_l67_6705

/-- Insertion sort comparison count bounds -/
theorem insertion_sort_comparison_bounds (n : ℕ) :
  ∀ (list : List ℕ), list.length = n →
  ∃ (comparisons : ℕ),
    (n - 1 : ℝ) ≤ comparisons ∧ comparisons ≤ (n * (n - 1) : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_insertion_sort_comparison_bounds_l67_6705


namespace NUMINAMATH_CALUDE_damaged_polynomial_satisfies_equation_damaged_polynomial_value_l67_6754

-- Define the damaged polynomial
def damaged_polynomial (x y : ℚ) : ℚ := -3 * x + y^2

-- Define the given equation
def equation_holds (x y : ℚ) : Prop :=
  damaged_polynomial x y + 2 * (x - 1/3 * y^2) = -x + 1/3 * y^2

-- Theorem 1: The damaged polynomial satisfies the equation
theorem damaged_polynomial_satisfies_equation :
  ∀ x y : ℚ, equation_holds x y :=
sorry

-- Theorem 2: The value of the damaged polynomial for given x and y
theorem damaged_polynomial_value :
  damaged_polynomial (-3) (3/2) = 45/4 :=
sorry

end NUMINAMATH_CALUDE_damaged_polynomial_satisfies_equation_damaged_polynomial_value_l67_6754


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l67_6779

/-- The line equation y = kx + 1 -/
def line_equation (k x : ℝ) : ℝ := k * x + 1

/-- The circle equation x^2 + y^2 - 2ax + a^2 - 2a - 4 = 0 -/
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 - 2*a*x + a^2 - 2*a - 4 = 0

/-- The condition that the line always intersects with the circle -/
def always_intersects (k a : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation k x = y → circle_equation x y a

theorem line_circle_intersection_range :
  ∀ k : ℝ, (∀ a : ℝ, always_intersects k a) ↔ -1 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l67_6779


namespace NUMINAMATH_CALUDE_circle_radius_is_4_l67_6794

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + y^2 + 4*y + 13 = 0

-- Define the radius of the circle
def circle_radius : ℝ := 4

-- Theorem statement
theorem circle_radius_is_4 :
  ∀ x y : ℝ, circle_equation x y → 
  ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_4_l67_6794


namespace NUMINAMATH_CALUDE_quadratic_roots_coefficients_l67_6782

theorem quadratic_roots_coefficients (p q : ℝ) :
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) →
  ((p = 0 ∧ q = 0) ∨ (p = 1 ∧ q = -2)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_coefficients_l67_6782


namespace NUMINAMATH_CALUDE_square_even_implies_even_sqrt_2_irrational_l67_6770

-- Part 1: If p² is even, then p is even
theorem square_even_implies_even (p : ℤ) : Even (p^2) → Even p := by sorry

-- Part 2: √2 is irrational
theorem sqrt_2_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a : ℚ) / b = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_square_even_implies_even_sqrt_2_irrational_l67_6770


namespace NUMINAMATH_CALUDE_max_distance_AB_l67_6732

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 0)

-- Define the line passing through M and intersecting C at A and B
def line_through_M (k : ℝ) (x : ℝ) : ℝ := k * (x - 2)

-- Define the condition for A and B being on both the line and the ellipse
def A_B_on_line_and_C (k x y : ℝ) : Prop :=
  C x y ∧ y = line_through_M k x

-- Define the vector addition condition
def vector_addition_condition (xA yA xB yB xP yP t : ℝ) : Prop :=
  xA + xB = t * xP ∧ yA + yB = t * yP

-- Main theorem
theorem max_distance_AB :
  ∀ (k xA yA xB yB xP yP t : ℝ),
    A_B_on_line_and_C k xA yA →
    A_B_on_line_and_C k xB yB →
    C xP yP →
    vector_addition_condition xA yA xB yB xP yP t →
    2 * Real.sqrt 6 / 3 < t →
    t < 2 →
    ∃ (max_dist : ℝ), max_dist = 2 * Real.sqrt 5 / 3 ∧
      ((xA - xB)^2 + (yA - yB)^2)^(1/2 : ℝ) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_AB_l67_6732


namespace NUMINAMATH_CALUDE_james_bed_purchase_l67_6752

theorem james_bed_purchase (bed_frame_price : ℝ) (discount_rate : ℝ) : 
  bed_frame_price = 75 →
  discount_rate = 0.2 →
  let bed_price := 10 * bed_frame_price
  let total_before_discount := bed_frame_price + bed_price
  let discount_amount := discount_rate * total_before_discount
  let final_price := total_before_discount - discount_amount
  final_price = 660 := by sorry

end NUMINAMATH_CALUDE_james_bed_purchase_l67_6752


namespace NUMINAMATH_CALUDE_ragnar_wood_chopping_l67_6790

/-- Represents the number of blocks of wood obtained from chopping trees over a period of time. -/
structure WoodChopping where
  trees_per_day : ℕ
  days : ℕ
  total_blocks : ℕ

/-- Calculates the number of blocks of wood obtained from one tree. -/
def blocks_per_tree (w : WoodChopping) : ℚ :=
  w.total_blocks / (w.trees_per_day * w.days)

/-- Theorem stating that given the specific conditions, the number of blocks per tree is 3. -/
theorem ragnar_wood_chopping :
  let w : WoodChopping := { trees_per_day := 2, days := 5, total_blocks := 30 }
  blocks_per_tree w = 3 := by sorry

end NUMINAMATH_CALUDE_ragnar_wood_chopping_l67_6790


namespace NUMINAMATH_CALUDE_largest_integer_l67_6707

theorem largest_integer (a b c d : ℤ) 
  (sum_abc : a + b + c = 160)
  (sum_abd : a + b + d = 185)
  (sum_acd : a + c + d = 205)
  (sum_bcd : b + c + d = 230) :
  max a (max b (max c d)) = 100 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_l67_6707


namespace NUMINAMATH_CALUDE_liz_total_spent_l67_6787

/-- The total amount spent by Liz on her baking purchases -/
def total_spent (recipe_book_cost : ℕ) (ingredient_cost : ℕ) (num_ingredients : ℕ) : ℕ :=
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_total_cost := ingredient_cost * num_ingredients
  let apron_cost := recipe_book_cost + 1
  recipe_book_cost + baking_dish_cost + ingredients_total_cost + apron_cost

/-- Theorem stating that Liz spent $40 in total -/
theorem liz_total_spent : total_spent 6 3 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_liz_total_spent_l67_6787


namespace NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l67_6781

theorem unique_solution_for_rational_equation :
  ∃! x : ℝ, x ≠ 3 ∧ (x^2 - 9) / (x - 3) = 3 * x :=
by
  -- The unique solution is x = 3/2
  use 3/2
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l67_6781


namespace NUMINAMATH_CALUDE_sock_count_proof_l67_6747

def total_socks (john_initial mary_initial kate_initial : ℕ)
                (john_thrown john_bought : ℕ)
                (mary_thrown mary_bought : ℕ)
                (kate_thrown kate_bought : ℕ) : ℕ :=
  (john_initial - john_thrown + john_bought) +
  (mary_initial - mary_thrown + mary_bought) +
  (kate_initial - kate_thrown + kate_bought)

theorem sock_count_proof :
  total_socks 33 20 15 19 13 6 10 5 8 = 69 := by
  sorry

end NUMINAMATH_CALUDE_sock_count_proof_l67_6747


namespace NUMINAMATH_CALUDE_max_wins_l67_6733

/-- Given the ratio of Chloe's wins to Max's wins and Chloe's total wins,
    calculate Max's wins. -/
theorem max_wins (chloe_wins : ℕ) (chloe_ratio : ℕ) (max_ratio : ℕ) 
    (h1 : chloe_wins = 24)
    (h2 : chloe_ratio = 8)
    (h3 : max_ratio = 3) :
  chloe_wins * max_ratio / chloe_ratio = 9 := by
  sorry

#check max_wins

end NUMINAMATH_CALUDE_max_wins_l67_6733


namespace NUMINAMATH_CALUDE_four_color_arrangement_l67_6751

theorem four_color_arrangement : ∀ n : ℕ, n = 4 → (Nat.factorial n) = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_color_arrangement_l67_6751


namespace NUMINAMATH_CALUDE_sprinting_competition_races_verify_sprinting_competition_races_l67_6798

/-- Calculates the number of races needed to determine a champion in a sprinting competition. -/
def races_needed (total_sprinters : ℕ) (sprinters_per_race : ℕ) (eliminations_per_race : ℕ) : ℕ :=
  (total_sprinters - 1) / eliminations_per_race

/-- Theorem stating that 43 races are needed for the given competition setup. -/
theorem sprinting_competition_races : 
  races_needed 216 6 5 = 43 := by
  sorry

/-- Verifies the result by simulating rounds of the competition. -/
def verify_races (total_sprinters : ℕ) (sprinters_per_race : ℕ) : ℕ :=
  let first_round := total_sprinters / sprinters_per_race
  let second_round := first_round / sprinters_per_race
  let third_round := if second_round ≥ sprinters_per_race then 1 else 0
  first_round + second_round + third_round

/-- Theorem stating that the verification method also yields 43 races. -/
theorem verify_sprinting_competition_races :
  verify_races 216 6 = 43 := by
  sorry

end NUMINAMATH_CALUDE_sprinting_competition_races_verify_sprinting_competition_races_l67_6798


namespace NUMINAMATH_CALUDE_blocks_in_specific_box_l67_6750

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks that can fit in the box -/
def blocksInBox (box : BoxDimensions) (block : BoxDimensions) : ℕ :=
  (box.length / block.length) * (box.width / block.width) * (box.height / block.height)

theorem blocks_in_specific_box :
  let box := BoxDimensions.mk 4 3 2
  let block := BoxDimensions.mk 3 1 1
  blocksInBox box block = 6 := by sorry

end NUMINAMATH_CALUDE_blocks_in_specific_box_l67_6750


namespace NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l67_6714

theorem smallest_solution_for_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 10 ∧
  ∀ (y : ℝ), y > 0 → (⌊y^2⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 10 → y ≥ x :=
by
  use 131 / 11
  sorry

end NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l67_6714
