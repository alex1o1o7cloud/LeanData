import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1383_138372

theorem quadratic_always_positive (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - a > 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1383_138372


namespace NUMINAMATH_CALUDE_problem_solution_l1383_138382

theorem problem_solution (x y a : ℝ) 
  (h1 : Real.sqrt (3 * x + 4) + y^2 + 6 * y + 9 = 0)
  (h2 : a * x * y - 3 * x = y) : 
  a = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1383_138382


namespace NUMINAMATH_CALUDE_log_x2y2_value_l1383_138375

-- Define the logarithm function (assuming it's the natural logarithm)
noncomputable def log : ℝ → ℝ := Real.log

-- Define the main theorem
theorem log_x2y2_value (x y : ℝ) (h1 : log (x * y^4) = 1) (h2 : log (x^3 * y) = 1) :
  log (x^2 * y^2) = 10/11 := by
  sorry

end NUMINAMATH_CALUDE_log_x2y2_value_l1383_138375


namespace NUMINAMATH_CALUDE_must_divide_p_l1383_138364

theorem must_divide_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 28)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 63)
  (h4 : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) :
  11 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_must_divide_p_l1383_138364


namespace NUMINAMATH_CALUDE_black_hair_ratio_l1383_138313

/-- Represents the ratio of hair colors in the class -/
structure HairColorRatio :=
  (red : ℕ)
  (blonde : ℕ)
  (black : ℕ)

/-- Represents the class information -/
structure ClassInfo :=
  (ratio : HairColorRatio)
  (redHairedKids : ℕ)
  (totalKids : ℕ)

/-- The main theorem -/
theorem black_hair_ratio (c : ClassInfo) 
  (h1 : c.ratio = HairColorRatio.mk 3 6 7)
  (h2 : c.redHairedKids = 9)
  (h3 : c.totalKids = 48) : 
  (c.ratio.black * c.redHairedKids / c.ratio.red : ℚ) / c.totalKids = 7 / 16 := by
  sorry

#check black_hair_ratio

end NUMINAMATH_CALUDE_black_hair_ratio_l1383_138313


namespace NUMINAMATH_CALUDE_cube_root_of_product_powers_l1383_138367

theorem cube_root_of_product_powers (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (m^4 * n^4)^(1/3) = (m * n)^(4/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_product_powers_l1383_138367


namespace NUMINAMATH_CALUDE_common_zero_condition_l1383_138374

/-- The first polynomial -/
def P (k : ℝ) (x : ℝ) : ℝ := 1988 * x^2 + k * x + 8891

/-- The second polynomial -/
def Q (k : ℝ) (x : ℝ) : ℝ := 8891 * x^2 + k * x + 1988

/-- Theorem stating the condition for common zeros -/
theorem common_zero_condition (k : ℝ) :
  (∃ x : ℝ, P k x = 0 ∧ Q k x = 0) ↔ (k = 10879 ∨ k = -10879) := by sorry

end NUMINAMATH_CALUDE_common_zero_condition_l1383_138374


namespace NUMINAMATH_CALUDE_jason_borrowed_amount_l1383_138377

/-- Calculates the total earnings for a given number of hours based on the described payment structure -/
def jasonEarnings (hours : ℕ) : ℕ :=
  let fullCycles := hours / 9
  let remainingHours := hours % 9
  let earningsPerCycle := (List.range 9).sum
  fullCycles * earningsPerCycle + (List.range remainingHours).sum

theorem jason_borrowed_amount :
  jasonEarnings 27 = 135 := by
  sorry

end NUMINAMATH_CALUDE_jason_borrowed_amount_l1383_138377


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l1383_138370

/-- Calculates the total cost of circus tickets -/
def total_ticket_cost (adult_price children_price senior_price : ℕ) 
  (adult_count children_count senior_count : ℕ) : ℕ :=
  adult_price * adult_count + children_price * children_count + senior_price * senior_count

/-- Proves that the total cost of circus tickets for the given quantities and prices is $318 -/
theorem circus_ticket_cost : 
  total_ticket_cost 55 28 42 4 2 1 = 318 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l1383_138370


namespace NUMINAMATH_CALUDE_angle_terminal_side_value_l1383_138344

/-- Given a point P(-4t, 3t) on the terminal side of angle θ, where t ≠ 0,
    the value of 2sinθ + cosθ is either 2/5 or -2/5. -/
theorem angle_terminal_side_value (t : ℝ) (θ : ℝ) (h : t ≠ 0) :
  let P : ℝ × ℝ := (-4 * t, 3 * t)
  (∃ (k : ℝ), k > 0 ∧ P = k • (Real.cos θ, Real.sin θ)) →
  2 * Real.sin θ + Real.cos θ = 2 / 5 ∨ 2 * Real.sin θ + Real.cos θ = -2 / 5 :=
by sorry


end NUMINAMATH_CALUDE_angle_terminal_side_value_l1383_138344


namespace NUMINAMATH_CALUDE_third_vertex_coordinates_l1383_138397

/-- Given a triangle with vertices (2,3), (0,0), and (0,y) where y > 0,
    if the area of the triangle is 36 square units, then y = 39 -/
theorem third_vertex_coordinates (y : ℝ) (h1 : y > 0) : 
  (1/2 : ℝ) * |2 * (3 - y)| = 36 → y = 39 := by
  sorry

end NUMINAMATH_CALUDE_third_vertex_coordinates_l1383_138397


namespace NUMINAMATH_CALUDE_no_solution_base_conversion_l1383_138390

theorem no_solution_base_conversion : ¬∃ (d : ℕ), d ≤ 9 ∧ d * 5 + 2 = d * 9 + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_base_conversion_l1383_138390


namespace NUMINAMATH_CALUDE_max_area_is_10000_l1383_138331

/-- Represents a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- The perimeter of the garden is 400 feet -/
def perimeterConstraint (g : Garden) : Prop :=
  2 * g.length + 2 * g.width = 400

/-- The length of the garden is at least 100 feet -/
def lengthConstraint (g : Garden) : Prop :=
  g.length ≥ 100

/-- The width of the garden is at least 50 feet -/
def widthConstraint (g : Garden) : Prop :=
  g.width ≥ 50

/-- The area of the garden -/
def area (g : Garden) : ℝ :=
  g.length * g.width

/-- The maximum area of the garden satisfying all constraints is 10000 square feet -/
theorem max_area_is_10000 :
  ∃ (g : Garden),
    perimeterConstraint g ∧
    lengthConstraint g ∧
    widthConstraint g ∧
    area g = 10000 ∧
    ∀ (g' : Garden),
      perimeterConstraint g' ∧
      lengthConstraint g' ∧
      widthConstraint g' →
      area g' ≤ 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_area_is_10000_l1383_138331


namespace NUMINAMATH_CALUDE_flight_savings_l1383_138327

theorem flight_savings (delta_price united_price : ℝ) 
  (delta_discount united_discount : ℝ) :
  delta_price = 850 →
  united_price = 1100 →
  delta_discount = 0.20 →
  united_discount = 0.30 →
  united_price * (1 - united_discount) - delta_price * (1 - delta_discount) = 90 := by
  sorry

end NUMINAMATH_CALUDE_flight_savings_l1383_138327


namespace NUMINAMATH_CALUDE_dispersion_measures_l1383_138389

/-- A sample of data points -/
def Sample : Type := List ℝ

/-- Standard deviation of a sample -/
noncomputable def standardDeviation (s : Sample) : ℝ := sorry

/-- Range of a sample -/
noncomputable def range (s : Sample) : ℝ := sorry

/-- Median of a sample -/
noncomputable def median (s : Sample) : ℝ := sorry

/-- Mean of a sample -/
noncomputable def mean (s : Sample) : ℝ := sorry

/-- A measure of dispersion is a function that quantifies the spread of a sample -/
def isDispersionMeasure (f : Sample → ℝ) : Prop := sorry

theorem dispersion_measures (s : Sample) :
  isDispersionMeasure standardDeviation ∧
  isDispersionMeasure range ∧
  ¬isDispersionMeasure median ∧
  ¬isDispersionMeasure mean :=
sorry

end NUMINAMATH_CALUDE_dispersion_measures_l1383_138389


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1383_138326

/-- A rhombus with diagonals in the ratio 3:4 and sum 56 has perimeter 80 -/
theorem rhombus_perimeter (d₁ d₂ s : ℝ) : 
  d₁ > 0 → d₂ > 0 → s > 0 →
  d₁ / d₂ = 3 / 4 → 
  d₁ + d₂ = 56 → 
  s^2 = (d₁/2)^2 + (d₂/2)^2 → 
  4 * s = 80 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1383_138326


namespace NUMINAMATH_CALUDE_prob_three_heads_eight_tosses_l1383_138386

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

/-- Theorem: The probability of getting exactly 3 heads in 8 tosses of a fair coin is 7/32 -/
theorem prob_three_heads_eight_tosses :
  prob_k_heads 8 3 = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_heads_eight_tosses_l1383_138386


namespace NUMINAMATH_CALUDE_function_identity_l1383_138318

theorem function_identity (f : ℝ → ℝ) 
  (h1 : Set.Finite {y | ∃ x ≠ 0, y = f x / x})
  (h2 : ∀ x : ℝ, f (x - 1 - f x) = f x - x - 1) :
  ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_CALUDE_function_identity_l1383_138318


namespace NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l1383_138359

/-- A linear function f(x) = x + 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem linear_function_not_in_fourth_quadrant :
  ∀ x : ℝ, ¬(fourth_quadrant x (f x)) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l1383_138359


namespace NUMINAMATH_CALUDE_self_inverse_sum_zero_l1383_138388

def is_self_inverse (a d : ℝ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 5; -12, d]
  M * M = 1

theorem self_inverse_sum_zero (a d : ℝ) (h : is_self_inverse a d) : a + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_self_inverse_sum_zero_l1383_138388


namespace NUMINAMATH_CALUDE_anya_andrea_erasers_l1383_138320

theorem anya_andrea_erasers : 
  ∀ (andrea_erasers : ℕ) (anya_multiplier : ℕ),
    andrea_erasers = 4 →
    anya_multiplier = 4 →
    anya_multiplier * andrea_erasers - andrea_erasers = 12 := by
  sorry

end NUMINAMATH_CALUDE_anya_andrea_erasers_l1383_138320


namespace NUMINAMATH_CALUDE_complex_subtraction_l1383_138321

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 4 + I) :
  a - 3*b = -7 - 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1383_138321


namespace NUMINAMATH_CALUDE_perpendicular_lines_theorem_l1383_138343

/-- A line in 3D space -/
structure Line3D where
  -- We represent a line by a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Perpendicular relation between two lines -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Definition of perpendicularity
  sorry

/-- Parallel relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  -- Definition of parallelism
  sorry

theorem perpendicular_lines_theorem (a b c d : Line3D) 
  (h1 : perpendicular a b)
  (h2 : perpendicular b c)
  (h3 : perpendicular c d)
  (h4 : perpendicular d a) :
  parallel b d ∨ parallel a c :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_theorem_l1383_138343


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l1383_138395

/-- The capacity of a water tank in liters. -/
def tank_capacity : ℝ := 120

/-- The difference in liters between 70% full and 40% full. -/
def difference : ℝ := 36

theorem tank_capacity_proof :
  tank_capacity * 0.7 - tank_capacity * 0.4 = difference ∧
  tank_capacity > 0 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l1383_138395


namespace NUMINAMATH_CALUDE_angle_bisector_inequality_l1383_138340

/-- Represents a triangle with side lengths and angle bisectors -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  aa_prime : ℝ → ℝ → ℝ
  bb_prime : ℝ → ℝ → ℝ

/-- Theorem: In a triangle ABC with angle bisectors AA' and BB', if a > b, then CA' > CB' and BA' > AB' -/
theorem angle_bisector_inequality (t : Triangle) (h : t.a > t.b) :
  (t.c * t.a) / (t.b + t.c) > (t.a * t.b) / (t.a + t.c) ∧
  (t.a * t.b) / (t.b + t.c) > (t.c * t.b) / (t.a + t.c) := by
  sorry


end NUMINAMATH_CALUDE_angle_bisector_inequality_l1383_138340


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1383_138339

theorem isosceles_triangle_base_angle (apex_angle : ℝ) (base_angle : ℝ) :
  apex_angle = 100 → -- The apex angle is 100°
  apex_angle + 2 * base_angle = 180 → -- Sum of angles in a triangle is 180°
  base_angle = 40 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1383_138339


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1383_138365

theorem triangle_perimeter (m n : ℝ) : 
  let side1 := 3 * m
  let side2 := side1 - (m - n)
  let side3 := side2 + 2 * n
  side1 + side2 + side3 = 7 * m + 4 * n :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1383_138365


namespace NUMINAMATH_CALUDE_abs_z_squared_l1383_138361

-- Define a complex number z
variable (z : ℂ)

-- State the theorem
theorem abs_z_squared (h : z + Complex.abs z = 2 + 8*I) : Complex.abs z ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_squared_l1383_138361


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l1383_138360

theorem line_intercepts_sum (c : ℝ) : 
  (∃ x y : ℝ, 3 * x + 5 * y + c = 0 ∧ x + y = 16) → c = -30 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l1383_138360


namespace NUMINAMATH_CALUDE_rational_root_count_l1383_138304

def polynomial (a₁ : ℤ) (x : ℚ) : ℚ := 12 * x^3 - 4 * x^2 + a₁ * x + 18

def is_possible_root (x : ℚ) : Prop :=
  ∃ (p q : ℤ), x = p / q ∧ 
  (p ∣ 18 ∨ p = 0) ∧ 
  (q ∣ 12 ∧ q ≠ 0)

theorem rational_root_count :
  ∃! (roots : Finset ℚ), 
    (∀ x ∈ roots, is_possible_root x) ∧
    (∀ x, is_possible_root x → x ∈ roots) ∧
    roots.card = 20 :=
sorry

end NUMINAMATH_CALUDE_rational_root_count_l1383_138304


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1383_138338

theorem polynomial_remainder (x : ℝ) : (x^13 + 1) % (x - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1383_138338


namespace NUMINAMATH_CALUDE_jacob_needs_26_more_fish_l1383_138307

def fishing_tournament (jacob_initial : ℕ) (alex_multiplier : ℕ) (alex_loss : ℕ) : ℕ :=
  let alex_initial := jacob_initial * alex_multiplier
  let alex_final := alex_initial - alex_loss
  let jacob_target := alex_final + 1
  jacob_target - jacob_initial

theorem jacob_needs_26_more_fish :
  fishing_tournament 8 7 23 = 26 := by
  sorry

end NUMINAMATH_CALUDE_jacob_needs_26_more_fish_l1383_138307


namespace NUMINAMATH_CALUDE_max_residents_top_floor_l1383_138334

/-- Represents the number of people living on a floor --/
def residents (floor : ℕ) : ℕ := floor

/-- The number of floors in the building --/
def num_floors : ℕ := 10

/-- Theorem: The floor with the most residents is the top floor --/
theorem max_residents_top_floor :
  ∀ k : ℕ, k ≤ num_floors → residents k ≤ residents num_floors :=
by
  sorry

#check max_residents_top_floor

end NUMINAMATH_CALUDE_max_residents_top_floor_l1383_138334


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1383_138330

theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a → Prime b → Prime c → 
  a * b * c = 399 → 
  2 * (a * b + b * c + c * a) = 422 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1383_138330


namespace NUMINAMATH_CALUDE_total_weight_calculation_l1383_138336

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight : ℝ := 72

/-- The number of moles of the compound -/
def number_of_moles : ℝ := 4

/-- The total weight of the compound in grams -/
def total_weight : ℝ := molecular_weight * number_of_moles

theorem total_weight_calculation : total_weight = 288 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_calculation_l1383_138336


namespace NUMINAMATH_CALUDE_rebecca_hours_less_than_toby_l1383_138396

theorem rebecca_hours_less_than_toby (x : ℕ) : 
  x + (2 * x - 10) + 56 = 157 → 64 - 56 = 8 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_hours_less_than_toby_l1383_138396


namespace NUMINAMATH_CALUDE_functional_equation_problem_l1383_138392

theorem functional_equation_problem (f : ℝ → ℝ) 
  (h1 : ∀ a b : ℝ, f (a + b) = f a * f b) 
  (h2 : f 1 = 2) : 
  f 1^2 + f 2 / f 1 + f 2^2 + f 4 / f 3 + f 3^2 + f 6 / f 5 + f 4^2 + f 8 / f 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l1383_138392


namespace NUMINAMATH_CALUDE_jimmy_stair_climbing_time_jimmy_total_time_l1383_138358

/-- The sum of an arithmetic sequence with 5 terms, first term 20, and common difference 5 -/
def arithmetic_sum : ℕ := by sorry

/-- The number of flights Jimmy climbs -/
def num_flights : ℕ := 5

/-- The time taken to climb the first flight -/
def first_flight_time : ℕ := 20

/-- The increase in time for each subsequent flight -/
def time_increase : ℕ := 5

theorem jimmy_stair_climbing_time :
  arithmetic_sum = num_flights * (2 * first_flight_time + (num_flights - 1) * time_increase) / 2 :=
by sorry

theorem jimmy_total_time : arithmetic_sum = 150 := by sorry

end NUMINAMATH_CALUDE_jimmy_stair_climbing_time_jimmy_total_time_l1383_138358


namespace NUMINAMATH_CALUDE_correct_multiplication_l1383_138347

theorem correct_multiplication (x : ℝ) : x * 51 = 244.8 → x * 15 = 72 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplication_l1383_138347


namespace NUMINAMATH_CALUDE_power_exceeds_million_l1383_138349

theorem power_exceeds_million : ∃ (n₁ n₂ n₃ : ℕ+),
  (1.01 : ℝ) ^ (n₁ : ℕ) > 1000000 ∧
  (1.001 : ℝ) ^ (n₂ : ℕ) > 1000000 ∧
  (1.000001 : ℝ) ^ (n₃ : ℕ) > 1000000 := by
  sorry

end NUMINAMATH_CALUDE_power_exceeds_million_l1383_138349


namespace NUMINAMATH_CALUDE_special_line_equation_l1383_138394

/-- A line passing through point (-2, 3) with an x-intercept twice its y-intercept -/
structure SpecialLine where
  -- Slope-intercept form: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (-2, 3)
  passes_through : m * (-2) + b = 3
  -- The x-intercept is twice the y-intercept
  intercept_relation : -b / m = 2 * b

/-- The equation of the special line is x + 2y - 4 = 0 or 3x + 2y = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.m = -1/2 ∧ l.b = 2) ∨ (l.m = -3/2 ∧ l.b = 0) := by
  sorry

end NUMINAMATH_CALUDE_special_line_equation_l1383_138394


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1383_138335

theorem price_reduction_percentage (original_price reduction : ℝ) : 
  original_price = 500 → reduction = 200 → (reduction / original_price) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l1383_138335


namespace NUMINAMATH_CALUDE_ellipse_sum_is_twelve_l1383_138305

/-- Represents an ellipse with center (h, k) and semi-axes a and b -/
structure Ellipse where
  h : ℝ  -- x-coordinate of center
  k : ℝ  -- y-coordinate of center
  a : ℝ  -- length of semi-major axis
  b : ℝ  -- length of semi-minor axis

/-- The sum of center coordinates and semi-axes lengths for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: The sum of center coordinates and semi-axes lengths for the given ellipse is 12 -/
theorem ellipse_sum_is_twelve : 
  let e : Ellipse := { h := 3, k := -2, a := 7, b := 4 }
  ellipse_sum e = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_is_twelve_l1383_138305


namespace NUMINAMATH_CALUDE_min_value_theorem_l1383_138362

noncomputable section

variables (a m n : ℝ)

-- Define the function f
def f (x : ℝ) := a^(x - 1) - 2

-- State the conditions
axiom a_pos : a > 0
axiom a_neq_one : a ≠ 1
axiom m_pos : m > 0
axiom n_pos : n > 0

-- Define the fixed point A
def A : ℝ × ℝ := (1, -1)

-- State that A lies on the line mx - ny - 1 = 0
axiom A_on_line : m * A.1 - n * A.2 - 1 = 0

-- State the theorem to be proved
theorem min_value_theorem : 
  (∀ x : ℝ, f x = f (A.1) → x = A.1) → 
  (∃ (m' n' : ℝ), m' > 0 ∧ n' > 0 ∧ m' * A.1 - n' * A.2 - 1 = 0 ∧ 1/m' + 2/n' < 1/m + 2/n) → 
  1/m + 2/n ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end

end NUMINAMATH_CALUDE_min_value_theorem_l1383_138362


namespace NUMINAMATH_CALUDE_sabrina_profit_is_35_l1383_138398

def sabrina_profit (total_loaves : ℕ) (morning_price : ℚ) (afternoon_price_ratio : ℚ) (evening_price : ℚ) (production_cost : ℚ) : ℚ :=
  let morning_loaves : ℕ := (2 * total_loaves) / 3
  let morning_revenue : ℚ := morning_loaves * morning_price
  let afternoon_loaves : ℕ := (total_loaves - morning_loaves) / 2
  let afternoon_revenue : ℚ := afternoon_loaves * (afternoon_price_ratio * morning_price)
  let evening_loaves : ℕ := total_loaves - morning_loaves - afternoon_loaves
  let evening_revenue : ℚ := evening_loaves * evening_price
  let total_revenue : ℚ := morning_revenue + afternoon_revenue + evening_revenue
  let total_cost : ℚ := total_loaves * production_cost
  total_revenue - total_cost

theorem sabrina_profit_is_35 :
  sabrina_profit 60 2 (1/4) 1 1 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sabrina_profit_is_35_l1383_138398


namespace NUMINAMATH_CALUDE_perfume_price_change_l1383_138371

def original_price : ℝ := 1200
def increase_rate : ℝ := 0.10
def decrease_rate : ℝ := 0.15

theorem perfume_price_change :
  let increased_price := original_price * (1 + increase_rate)
  let final_price := increased_price * (1 - decrease_rate)
  original_price - final_price = 78 := by
sorry

end NUMINAMATH_CALUDE_perfume_price_change_l1383_138371


namespace NUMINAMATH_CALUDE_min_value_z_l1383_138380

/-- The function z(x) = 5x^2 + 10x + 20 has a minimum value of 15 -/
theorem min_value_z (x : ℝ) : ∀ y : ℝ, 5 * x^2 + 10 * x + 20 ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l1383_138380


namespace NUMINAMATH_CALUDE_lap_time_improvement_is_12_seconds_l1383_138333

-- Define the initial condition
def initial_laps : ℕ := 25
def initial_time : ℕ := 50

-- Define the later condition
def later_laps : ℕ := 30
def later_time : ℕ := 54

-- Define the function to calculate lap time in seconds
def lap_time_seconds (laps : ℕ) (time : ℕ) : ℚ :=
  (time * 60) / laps

-- Define the improvement in lap time
def lap_time_improvement : ℚ :=
  lap_time_seconds initial_laps initial_time - lap_time_seconds later_laps later_time

-- Theorem statement
theorem lap_time_improvement_is_12_seconds :
  lap_time_improvement = 12 := by sorry

end NUMINAMATH_CALUDE_lap_time_improvement_is_12_seconds_l1383_138333


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l1383_138323

theorem book_arrangement_proof :
  let total_books : ℕ := 8
  let geometry_books : ℕ := 5
  let number_theory_books : ℕ := 3
  Nat.choose total_books geometry_books = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l1383_138323


namespace NUMINAMATH_CALUDE_system_solution_l1383_138391

theorem system_solution (x y : ℝ) : 2*x - y = 5 ∧ x - 2*y = 1 → x - y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1383_138391


namespace NUMINAMATH_CALUDE_cupcake_frosting_problem_l1383_138378

/-- Represents the number of cupcakes frosted in a given time -/
def cupcakes_frosted (rate : ℚ) (time : ℚ) : ℚ := rate * time

/-- Represents the combined rate of two people frosting cupcakes -/
def combined_rate (rate1 : ℚ) (rate2 : ℚ) : ℚ := 1 / (1 / rate1 + 1 / rate2)

theorem cupcake_frosting_problem :
  let cagney_rate : ℚ := 1 / 18  -- Cagney's frosting rate (cupcakes per second)
  let lacey_rate : ℚ := 1 / 40   -- Lacey's frosting rate (cupcakes per second)
  let total_time : ℚ := 6 * 60   -- Total time in seconds
  let lacey_delay : ℚ := 60      -- Lacey's delay in seconds

  let cagney_solo_time := lacey_delay
  let combined_time := total_time - lacey_delay
  let combined_frosting_rate := combined_rate cagney_rate lacey_rate

  let total_cupcakes := 
    cupcakes_frosted cagney_rate cagney_solo_time + 
    cupcakes_frosted combined_frosting_rate combined_time

  ⌊total_cupcakes⌋ = 27 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_frosting_problem_l1383_138378


namespace NUMINAMATH_CALUDE_true_discount_example_l1383_138351

/-- Given a banker's discount and sum due, calculate the true discount -/
def true_discount (bankers_discount : ℚ) (sum_due : ℚ) : ℚ :=
  bankers_discount / (1 + bankers_discount / sum_due)

/-- Theorem stating that for a banker's discount of 18 and sum due of 90, the true discount is 15 -/
theorem true_discount_example : true_discount 18 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_true_discount_example_l1383_138351


namespace NUMINAMATH_CALUDE_max_value_of_g_l1383_138301

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l1383_138301


namespace NUMINAMATH_CALUDE_page_shoe_collection_l1383_138399

theorem page_shoe_collection (initial_shoes : ℕ) (donation_percentage : ℚ) (new_shoes : ℕ) : 
  initial_shoes = 120 →
  donation_percentage = 45 / 100 →
  new_shoes = 15 →
  initial_shoes - (initial_shoes * donation_percentage).floor + new_shoes = 81 :=
by sorry

end NUMINAMATH_CALUDE_page_shoe_collection_l1383_138399


namespace NUMINAMATH_CALUDE_reciprocal_in_fourth_quadrant_l1383_138308

theorem reciprocal_in_fourth_quadrant (i : ℂ) (z : ℂ) :
  i * i = -1 →
  z = 1 + i →
  let w := 1 / z
  0 < w.re ∧ w.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_in_fourth_quadrant_l1383_138308


namespace NUMINAMATH_CALUDE_adlai_animal_legs_l1383_138368

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The number of dogs Adlai has -/
def adlai_dogs : ℕ := 2

/-- The number of chickens Adlai has -/
def adlai_chickens : ℕ := 1

/-- The total number of animal legs Adlai has -/
def total_legs : ℕ := adlai_dogs * dog_legs + adlai_chickens * chicken_legs

theorem adlai_animal_legs : total_legs = 10 := by
  sorry

end NUMINAMATH_CALUDE_adlai_animal_legs_l1383_138368


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l1383_138366

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the tangent line l
def tangent_line_l (x y : ℝ) : Prop := x = 2 ∨ 4*x - 3*y + 1 = 0

-- Theorem statement
theorem circle_and_tangent_line :
  -- Circle E passes through (0,0) and (1,1)
  circle_E 0 0 ∧ circle_E 1 1 ∧
  -- One of the three conditions is satisfied
  (circle_E 2 0 ∨ 
   (∀ m : ℝ, ∃ x y : ℝ, circle_E x y ∧ m*x - y - m = 0) ∨
   (∃ x : ℝ, circle_E x 0 ∧ x = 0)) →
  -- The tangent line passes through (2,3)
  (∃ x y : ℝ, circle_E x y ∧ tangent_line_l x y ∧
   ((x - 2)^2 + (y - 3)^2).sqrt = 1) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l1383_138366


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1383_138376

theorem book_arrangement_count : ℕ := by
  -- Define the total number of books
  let total_books : ℕ := 6
  -- Define the number of identical copies for each book type
  let identical_copies1 : ℕ := 3
  let identical_copies2 : ℕ := 2
  let unique_book : ℕ := 1

  -- Assert that the sum of all book types equals the total number of books
  have h_total : identical_copies1 + identical_copies2 + unique_book = total_books := by sorry

  -- Define the number of distinct arrangements
  let arrangements : ℕ := Nat.factorial total_books / (Nat.factorial identical_copies1 * Nat.factorial identical_copies2)

  -- Prove that the number of distinct arrangements is 60
  have h_result : arrangements = 60 := by sorry

  -- Return the result
  exact 60

end NUMINAMATH_CALUDE_book_arrangement_count_l1383_138376


namespace NUMINAMATH_CALUDE_triangle_angle_C_l1383_138346

theorem triangle_angle_C (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → 
  5 * Real.sin A + 2 * Real.cos B = 3 →
  2 * Real.sin B + 5 * Real.tan A = 7 →
  Real.sin C = Real.sin (A + B) := by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l1383_138346


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_double_root_at_three_l1383_138383

/-- The quadratic equation (k-2)x^2 - 2x + 1 = 0 has two real roots if and only if k ≤ 3 and k ≠ 2.
    When k = 3, the equation has a double root at x = 1. -/
theorem quadratic_roots_condition (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k - 2) * x₁^2 - 2 * x₁ + 1 = 0 ∧ (k - 2) * x₂^2 - 2 * x₂ + 1 = 0) ↔
  (k ≤ 3 ∧ k ≠ 2) :=
sorry

/-- When k = 3, the quadratic equation x^2 - 2x + 1 = 0 has a double root at x = 1. -/
theorem double_root_at_three :
  ∀ x : ℝ, x^2 - 2*x + 1 = 0 ↔ x = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_double_root_at_three_l1383_138383


namespace NUMINAMATH_CALUDE_friends_average_age_l1383_138328

def average_age (m : ℝ) : ℝ := 1.05 * m + 21.6

theorem friends_average_age (m : ℝ) :
  let john := 1.5 * m
  let mary := m
  let tonya := 60
  let sam := 0.8 * tonya
  let carol := 2.75 * m
  (john + mary + tonya + sam + carol) / 5 = average_age m := by
  sorry

end NUMINAMATH_CALUDE_friends_average_age_l1383_138328


namespace NUMINAMATH_CALUDE_solution_range_l1383_138363

theorem solution_range (m : ℝ) : 
  (∃ x y : ℝ, x + y = -1 ∧ 5 * x + 2 * y = 6 * m + 7 ∧ 2 * x - y < 19) → 
  m < 3/2 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l1383_138363


namespace NUMINAMATH_CALUDE_line_through_d_divides_equally_l1383_138393

-- Define the shape
structure Shape :=
  (area : ℝ)
  (is_unit_squares : Bool)

-- Define points
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a line
structure Line :=
  (point1 : Point)
  (point2 : Point)

-- Define the problem setup
def problem_setup (s : Shape) (p a b c d e : Point) : Prop :=
  s.is_unit_squares ∧
  s.area = 9 ∧
  b.x = (a.x + c.x) / 2 ∧
  b.y = (a.y + c.y) / 2 ∧
  d.x = (c.x + e.x) / 2 ∧
  d.y = (c.y + e.y) / 2

-- Define the division of area by a line
def divides_area_equally (l : Line) (s : Shape) : Prop :=
  ∃ (area1 area2 : ℝ), 
    area1 = area2 ∧
    area1 + area2 = s.area

-- Theorem statement
theorem line_through_d_divides_equally 
  (s : Shape) (p a b c d e : Point) (l : Line) :
  problem_setup s p a b c d e →
  l.point1 = p →
  l.point2 = d →
  divides_area_equally l s :=
sorry

end NUMINAMATH_CALUDE_line_through_d_divides_equally_l1383_138393


namespace NUMINAMATH_CALUDE_probability_same_color_is_correct_l1383_138312

def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def green_marbles : ℕ := 4

def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

def probability_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem probability_same_color_is_correct : probability_same_color = 106 / 109725 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_correct_l1383_138312


namespace NUMINAMATH_CALUDE_road_trip_gas_cost_l1383_138387

/-- Calculates the total cost of filling a gas tank at multiple stations -/
def total_gas_cost (tank_capacity : ℝ) (prices : List ℝ) : ℝ :=
  List.sum (List.map (· * tank_capacity) prices)

/-- Proves that the total cost of filling a 12-gallon tank at 4 stations with given prices is $180 -/
theorem road_trip_gas_cost :
  let tank_capacity : ℝ := 12
  let gas_prices : List ℝ := [3, 3.5, 4, 4.5]
  total_gas_cost tank_capacity gas_prices = 180 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_gas_cost_l1383_138387


namespace NUMINAMATH_CALUDE_system_solution_l1383_138319

theorem system_solution :
  let f (x y : ℝ) := y + Real.sqrt (y - 3*x) + 3*x = 12
  let g (x y : ℝ) := y^2 + y - 3*x - 9*x^2 = 144
  ∀ x y : ℝ, (f x y ∧ g x y) ↔ ((x = -4/3 ∧ y = 12) ∨ (x = -24 ∧ y = 72)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1383_138319


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1383_138310

theorem polynomial_expansion (x : ℝ) : 
  (3*x^3 + 4*x^2 + 12)*(x + 1) - (x + 1)*(2*x^3 + 6*x^2 - 42) + (6*x^2 - 28)*(x + 1)*(x - 2) = 
  7*x^4 - 7*x^3 - 42*x^2 + 82*x + 110 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1383_138310


namespace NUMINAMATH_CALUDE_sum_remainder_theorem_l1383_138379

theorem sum_remainder_theorem :
  (9256 + 9257 + 9258 + 9259 + 9260) % 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_theorem_l1383_138379


namespace NUMINAMATH_CALUDE_binomial_variance_l1383_138303

variable (p : ℝ)

-- Define the random variable X
def X : ℕ → ℝ
| 0 => 1 - p
| 1 => p
| _ => 0

-- Conditions
axiom p_range : 0 < p ∧ p < 1

-- Define the probability mass function
def pmf (k : ℕ) : ℝ := X p k

-- Define the expected value
def expectation : ℝ := p

-- Define the variance
def variance : ℝ := p * (1 - p)

-- Theorem statement
theorem binomial_variance : 
  ∀ (p : ℝ), 0 < p ∧ p < 1 → variance p = p * (1 - p) :=
by sorry

end NUMINAMATH_CALUDE_binomial_variance_l1383_138303


namespace NUMINAMATH_CALUDE_geometry_propositions_l1383_138329

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations between lines and planes
def subset (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def intersect (p1 p2 : Plane) (l : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem geometry_propositions 
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n) 
  (h_distinct_planes : α ≠ β) :
  (∀ (m : Line) (α β : Plane), 
    subset m α → perpendicular m β → perpendicular_planes α β) ∧
  (∃ (m : Line) (α β : Plane) (n : Line), 
    subset m α ∧ intersect α β n ∧ perpendicular_planes α β ∧ ¬(perpendicular m n)) ∧
  (∃ (m n : Line) (α β : Plane), 
    subset m α ∧ subset n β ∧ parallel_planes α β ∧ ¬(parallel_lines m n)) ∧
  (∀ (m n : Line) (α β : Plane), 
    parallel_line_plane m α → subset m β → intersect α β n → parallel_lines m n) := by
  sorry


end NUMINAMATH_CALUDE_geometry_propositions_l1383_138329


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1383_138381

theorem quadratic_root_difference (p q : ℝ) : 
  let r := (p + Real.sqrt (p^2 + q))
  let s := (p - Real.sqrt (p^2 + q))
  abs (r - s) = Real.sqrt (2 * p^2 + 2 * q) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1383_138381


namespace NUMINAMATH_CALUDE_success_probability_given_expectation_l1383_138345

/-- A two-point distribution random variable -/
structure TwoPointDistribution where
  X : ℝ → ℝ  -- The random variable
  p : ℝ      -- Success probability
  h1 : 0 ≤ p ∧ p ≤ 1  -- Probability is between 0 and 1

/-- Expected value of a two-point distribution -/
def expectedValue (T : TwoPointDistribution) : ℝ :=
  T.p * 1 + (1 - T.p) * 0

theorem success_probability_given_expectation 
  (T : TwoPointDistribution) 
  (h : expectedValue T = 0.7) : 
  T.p = 0.7 := by
  sorry


end NUMINAMATH_CALUDE_success_probability_given_expectation_l1383_138345


namespace NUMINAMATH_CALUDE_equation_solution_l1383_138311

theorem equation_solution : ∃ x : ℝ, 0.05 * x + 0.07 * (25 + x) = 15.1 ∧ x = 111.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1383_138311


namespace NUMINAMATH_CALUDE_billys_restaurant_bill_l1383_138369

/-- Calculates the total bill for a group at Billy's Restaurant -/
theorem billys_restaurant_bill (adults children meal_cost : ℕ) : 
  adults = 2 → children = 5 → meal_cost = 3 → 
  (adults + children) * meal_cost = 21 := by
sorry

end NUMINAMATH_CALUDE_billys_restaurant_bill_l1383_138369


namespace NUMINAMATH_CALUDE_new_average_after_changes_l1383_138353

theorem new_average_after_changes (numbers : Finset ℕ) (original_sum : ℕ) : 
  numbers.card = 15 → 
  original_sum = numbers.sum id →
  original_sum / numbers.card = 40 →
  let new_sum := original_sum + 9 * 10 - 6 * 5
  new_sum / numbers.card = 44 := by
sorry

end NUMINAMATH_CALUDE_new_average_after_changes_l1383_138353


namespace NUMINAMATH_CALUDE_project_budget_decrease_l1383_138316

/-- Proves that the annual decrease in project V's budget is $30,000 --/
theorem project_budget_decrease (q_initial v_initial q_increase : ℕ) 
  (h1 : q_initial = 540000)
  (h2 : v_initial = 780000)
  (h3 : q_increase = 30000) :
  ∃ v_decrease : ℕ, 
    q_initial + 4 * q_increase = v_initial - 4 * v_decrease ∧ 
    v_decrease = 30000 := by
  sorry

end NUMINAMATH_CALUDE_project_budget_decrease_l1383_138316


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l1383_138302

theorem max_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y - 25 = 0}
  ∃ (p : ℝ × ℝ), p ∈ circle ∧
    (∀ (q : ℝ × ℝ), q ∈ circle →
      ∃ (r : ℝ × ℝ), r ∈ line ∧
        dist p r ≥ dist q r) ∧
    (∃ (s : ℝ × ℝ), s ∈ line ∧ dist p s = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l1383_138302


namespace NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l1383_138325

theorem smallest_gcd_of_multiples (m n : ℕ+) (h : Nat.gcd m n = 15) :
  ∃ (k : ℕ), k ≥ 30 ∧ Nat.gcd (14 * m) (20 * n) = k ∧
  ∀ (j : ℕ), j < 30 → Nat.gcd (14 * m) (20 * n) ≠ j :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l1383_138325


namespace NUMINAMATH_CALUDE_number_is_perfect_square_l1383_138348

def N : ℕ := (10^1998 * ((10^1997 - 1) / 9)) + 2 * ((10^1998 - 1) / 9)

theorem number_is_perfect_square : 
  N = (10^1998 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_number_is_perfect_square_l1383_138348


namespace NUMINAMATH_CALUDE_n_has_9_digits_l1383_138324

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_minimal : ∀ m : ℕ, m > 0 → 30 ∣ m → (∃ k : ℕ, m^2 = k^3) → (∃ k : ℕ, m^3 = k^2) → n ≤ m

/-- The number of digits in n -/
def num_digits (x : ℕ) : ℕ := sorry

/-- Theorem stating that n has 9 digits -/
theorem n_has_9_digits : num_digits n = 9 := by sorry

end NUMINAMATH_CALUDE_n_has_9_digits_l1383_138324


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1383_138306

-- Define the quadratic expression
def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - 2

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x ≥ -1 }
  else if a > 0 then { x | -1 ≤ x ∧ x ≤ 2/a }
  else if -2 < a ∧ a < 0 then { x | x ≤ 2/a ∨ x ≥ -1 }
  else if a < -2 then { x | x ≤ -1 ∨ x ≥ 2/a }
  else Set.univ

-- State the theorem
theorem quadratic_inequality_solution (a : ℝ) :
  { x : ℝ | f a x ≤ 0 } = solution_set a :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1383_138306


namespace NUMINAMATH_CALUDE_trigonometric_expressions_l1383_138355

theorem trigonometric_expressions :
  (2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1 / 2) ∧
  ((-1)^2023 + 2 * Real.sin (45 * π / 180) - Real.cos (30 * π / 180) + Real.sin (60 * π / 180) + (Real.tan (60 * π / 180))^2 = 2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_l1383_138355


namespace NUMINAMATH_CALUDE_circle_line_tangency_l1383_138356

/-- The circle equation -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 - 2*k*x - 2*y = 0

/-- The line equation -/
def line_equation (x y k : ℝ) : Prop :=
  x + y = 2*k

/-- The tangency condition -/
def are_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y k ∧ line_equation x y k

/-- The main theorem -/
theorem circle_line_tangency (k : ℝ) :
  are_tangent k → k = -1 := by sorry

end NUMINAMATH_CALUDE_circle_line_tangency_l1383_138356


namespace NUMINAMATH_CALUDE_window_width_calculation_l1383_138332

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12
def door_length : ℝ := 6
def door_width : ℝ := 3
def window_height : ℝ := 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 3
def total_cost : ℝ := 2718

theorem window_width_calculation (W : ℝ) :
  (2 * (room_length * room_height + room_width * room_height) -
   door_length * door_width - num_windows * W * window_height) * cost_per_sqft = total_cost →
  W = 4 := by sorry

end NUMINAMATH_CALUDE_window_width_calculation_l1383_138332


namespace NUMINAMATH_CALUDE_boys_on_playground_l1383_138315

theorem boys_on_playground (total_children girls : ℕ) 
  (h1 : total_children = 62) 
  (h2 : girls = 35) : 
  total_children - girls = 27 := by
sorry

end NUMINAMATH_CALUDE_boys_on_playground_l1383_138315


namespace NUMINAMATH_CALUDE_magic_box_result_l1383_138322

def magic_box (a b : ℝ) : ℝ := a^2 + b + 1

theorem magic_box_result : 
  let m := magic_box (-2) 3
  magic_box m 1 = 66 := by sorry

end NUMINAMATH_CALUDE_magic_box_result_l1383_138322


namespace NUMINAMATH_CALUDE_sum_of_squares_l1383_138357

theorem sum_of_squares (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 120) : 
  x^2 + y^2 = 2424 / 49 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1383_138357


namespace NUMINAMATH_CALUDE_vertex_in_second_quadrant_l1383_138337

/-- The quadratic function f(x) = -(x+1)^2 + 2 -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 2

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := -1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := f vertex_x

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The vertex of f(x) = -(x+1)^2 + 2 is in the second quadrant -/
theorem vertex_in_second_quadrant : is_in_second_quadrant vertex_x vertex_y := by
  sorry

end NUMINAMATH_CALUDE_vertex_in_second_quadrant_l1383_138337


namespace NUMINAMATH_CALUDE_gwen_spent_nothing_l1383_138373

/-- Represents the amount of money Gwen received from her mom -/
def mom_money : ℤ := 8

/-- Represents the amount of money Gwen received from her dad -/
def dad_money : ℤ := 5

/-- Represents the difference in money Gwen has from her mom compared to her dad after spending -/
def difference_after_spending : ℤ := 3

/-- Represents the amount of money Gwen spent -/
def money_spent : ℤ := 0

theorem gwen_spent_nothing :
  (mom_money - money_spent) - (dad_money - money_spent) = difference_after_spending :=
sorry

end NUMINAMATH_CALUDE_gwen_spent_nothing_l1383_138373


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1383_138384

theorem smallest_solution_of_equation :
  ∃ (x : ℝ), x = -3 ∧ 
  (3 * x) / (x + 3) + (3 * x^2 - 18) / x = 9 ∧
  (∀ y : ℝ, (3 * y) / (y + 3) + (3 * y^2 - 18) / y = 9 → y ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1383_138384


namespace NUMINAMATH_CALUDE_trisection_line_equation_l1383_138317

/-- Given points A and B in the xy-plane, this function returns the coordinates of the two trisection points of the line segment AB. -/
def trisectionPoints (A B : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- Given three points, this function returns true if they are collinear, false otherwise. -/
def areCollinear (P Q R : ℝ × ℝ) : Prop := sorry

/-- Given three points, this function returns the equation of the line passing through them in the form ax + by + c = 0, represented as the triple (a, b, c). -/
def lineEquation (P Q R : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

theorem trisection_line_equation :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (8, 3)
  let P : ℝ × ℝ := (2, 1)
  let (C, D) := trisectionPoints A B
  areCollinear P C D ∧ lineEquation P C D = (1, -1, 1) := by sorry

end NUMINAMATH_CALUDE_trisection_line_equation_l1383_138317


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l1383_138309

/-- In a right-angled triangle ABC, the sum of two specific arctangent expressions equals π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  Real.arctan (a / (Real.sqrt b + Real.sqrt c)) + Real.arctan (b / (Real.sqrt a + Real.sqrt c)) = π/4 := by
  sorry

#check right_triangle_arctan_sum

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l1383_138309


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l1383_138352

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def N : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- Theorem statement
theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l1383_138352


namespace NUMINAMATH_CALUDE_hyperbola_min_value_l1383_138341

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    with one asymptote having a slope angle of π/3 and eccentricity e,
    the minimum value of (a² + e)/b is 2√6/3 -/
theorem hyperbola_min_value (a b : ℝ) (e : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b / a = Real.sqrt 3) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ e = c / a) →
  (∀ k : ℝ, k > 0 → (a^2 + e) / b ≥ 2 * Real.sqrt 6 / 3) ∧
  (∃ k : ℝ, k > 0 ∧ (a^2 + e) / b = 2 * Real.sqrt 6 / 3) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_min_value_l1383_138341


namespace NUMINAMATH_CALUDE_system_solution_l1383_138300

theorem system_solution (x y : ℝ) : 
  (6 * (1 - x)^2 = 1 / y ∧ 6 * (1 - y)^2 = 1 / x) ↔ 
  ((x = 3/2 ∧ y = 2/3) ∨ 
   (x = 2/3 ∧ y = 3/2) ∨ 
   (x = (1/6) * (4 + 2^(2/3) + 2^(4/3)) ∧ y = (1/6) * (4 + 2^(2/3) + 2^(4/3)))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1383_138300


namespace NUMINAMATH_CALUDE_monotonic_quadratic_l1383_138350

/-- A function f is monotonic on an interval [a,b] if it is either 
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The statement of the problem -/
theorem monotonic_quadratic (a : ℝ) :
  IsMonotonic (fun x => x^2 + (1-a)*x + 3) 1 4 ↔ a ≥ 9 ∨ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_l1383_138350


namespace NUMINAMATH_CALUDE_right_triangle_ab_length_l1383_138314

/-- 
Given a right triangle ABC in the x-y plane where:
- ∠B = 90°
- The length of AC is 225
- The slope of line segment AC is 4/3
Prove that the length of AB is 180.
-/
theorem right_triangle_ab_length 
  (A B C : ℝ × ℝ) -- Points in the plane
  (right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) -- ∠B = 90°
  (ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 225) -- Length of AC is 225
  (ac_slope : (C.2 - A.2) / (C.1 - A.1) = 4 / 3) -- Slope of AC is 4/3
  : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 180 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ab_length_l1383_138314


namespace NUMINAMATH_CALUDE_coin_machine_possible_amount_l1383_138354

theorem coin_machine_possible_amount :
  ∃ (m n p : ℕ), 298 = 5 + 25 * m + 2 * n + 29 * p :=
sorry

end NUMINAMATH_CALUDE_coin_machine_possible_amount_l1383_138354


namespace NUMINAMATH_CALUDE_star_value_for_specific_conditions_l1383_138342

-- Define the * operation for non-zero integers
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_value_for_specific_conditions (a b : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : a + b = 16) 
  (h4 : a^2 + b^2 = 136) : 
  star a b = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_star_value_for_specific_conditions_l1383_138342


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1383_138385

theorem arithmetic_sequence_sum (a₁ aₙ : ℤ) (n : ℕ) (h : n > 0) :
  (a₁ = -4) → (aₙ = 37) → (n = 10) →
  (∃ d : ℚ, ∀ k : ℕ, k < n → a₁ + k * d = aₙ - (n - 1 - k) * d) →
  (n : ℚ) * (a₁ + aₙ) / 2 = 165 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1383_138385
