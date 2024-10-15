import Mathlib

namespace NUMINAMATH_CALUDE_subtracted_value_l1187_118763

theorem subtracted_value (chosen_number : ℕ) (final_answer : ℕ) : 
  chosen_number = 848 → final_answer = 6 → 
  ∃ x : ℚ, (chosen_number / 8 : ℚ) - x = final_answer ∧ x = 100 := by
sorry

end NUMINAMATH_CALUDE_subtracted_value_l1187_118763


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1187_118714

/-- A sequence where the ratio of consecutive terms is constant -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence with a_4 = 2, prove that a_2 * a_6 = 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : GeometricSequence a) (h_a4 : a 4 = 2) : 
    a 2 * a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1187_118714


namespace NUMINAMATH_CALUDE_cylinder_water_properties_l1187_118759

/-- Represents a cylindrical tank lying on its side -/
structure HorizontalCylinder where
  radius : ℝ
  height : ℝ

/-- Represents the water level in the tank -/
def WaterLevel : ℝ := 3

/-- The volume of water in the cylindrical tank -/
def waterVolume (c : HorizontalCylinder) (h : ℝ) : ℝ := sorry

/-- The submerged surface area of the cylindrical side of the tank -/
def submergedSurfaceArea (c : HorizontalCylinder) (h : ℝ) : ℝ := sorry

theorem cylinder_water_properties :
  let c : HorizontalCylinder := { radius := 5, height := 10 }
  (waterVolume c WaterLevel = 290.7 * Real.pi - 40 * Real.sqrt 6) ∧
  (submergedSurfaceArea c WaterLevel = 91.5) := by sorry

end NUMINAMATH_CALUDE_cylinder_water_properties_l1187_118759


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l1187_118778

theorem purely_imaginary_z (α : ℝ) :
  let z : ℂ := Complex.mk (Real.sin α) (-(1 - Real.cos α))
  z.re = 0 → ∃ k : ℤ, α = (2 * k + 1) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l1187_118778


namespace NUMINAMATH_CALUDE_water_in_tank_after_40_days_l1187_118741

/-- Calculates the final amount of water in a tank given initial conditions and events. -/
def finalWaterAmount (initialWater : ℝ) (evaporationRate : ℝ) (daysBeforeAddition : ℕ) 
  (addedWater : ℝ) (remainingDays : ℕ) : ℝ :=
  let waterAfterFirstEvaporation := initialWater - evaporationRate * daysBeforeAddition
  let waterAfterAddition := waterAfterFirstEvaporation + addedWater
  waterAfterAddition - evaporationRate * remainingDays

/-- The final amount of water in the tank is 520 liters. -/
theorem water_in_tank_after_40_days :
  finalWaterAmount 500 2 15 100 25 = 520 := by
  sorry

end NUMINAMATH_CALUDE_water_in_tank_after_40_days_l1187_118741


namespace NUMINAMATH_CALUDE_composite_sum_of_squares_l1187_118700

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
   x₁^2 + a*x₁ + 1 = b ∧ x₂^2 + a*x₂ + 1 = b) →
  ∃ m n : ℤ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end NUMINAMATH_CALUDE_composite_sum_of_squares_l1187_118700


namespace NUMINAMATH_CALUDE_roots_sum_of_sixth_powers_l1187_118711

theorem roots_sum_of_sixth_powers (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 7 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 7 + 1 = 0 →
  r^6 + s^6 = 389374 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_sixth_powers_l1187_118711


namespace NUMINAMATH_CALUDE_excellent_students_problem_l1187_118708

theorem excellent_students_problem (B₁ B₂ B₃ : Finset ℕ) 
  (h_total : (B₁ ∪ B₂ ∪ B₃).card = 100)
  (h_math : B₁.card = 70)
  (h_phys : B₂.card = 65)
  (h_chem : B₃.card = 75)
  (h_math_phys : (B₁ ∩ B₂).card = 40)
  (h_math_chem : (B₁ ∩ B₃).card = 45)
  (h_all : (B₁ ∩ B₂ ∩ B₃).card = 25) :
  ((B₂ ∩ B₃) \ B₁).card = 25 := by
  sorry

end NUMINAMATH_CALUDE_excellent_students_problem_l1187_118708


namespace NUMINAMATH_CALUDE_max_sum_of_square_roots_l1187_118790

theorem max_sum_of_square_roots (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 2013) : 
  Real.sqrt (3 * a + 12) + Real.sqrt (3 * b + 12) + Real.sqrt (3 * c + 12) ≤ 135 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_square_roots_l1187_118790


namespace NUMINAMATH_CALUDE_male_students_count_l1187_118710

/-- Represents the number of students in a grade. -/
def total_students : ℕ := 800

/-- Represents the size of the stratified sample. -/
def sample_size : ℕ := 20

/-- Represents the number of female students in the sample. -/
def females_in_sample : ℕ := 8

/-- Calculates the number of male students in the entire grade based on stratified sampling. -/
def male_students_in_grade : ℕ := 
  (total_students * (sample_size - females_in_sample)) / sample_size

/-- Theorem stating that the number of male students in the grade is 480. -/
theorem male_students_count : male_students_in_grade = 480 := by sorry

end NUMINAMATH_CALUDE_male_students_count_l1187_118710


namespace NUMINAMATH_CALUDE_ap_triangle_centroid_incenter_parallel_l1187_118724

/-- A triangle with sides in arithmetic progression -/
structure APTriangle where
  a : ℝ
  b : ℝ
  hab : a ≠ b
  hab_pos : 0 < a ∧ 0 < b

/-- The centroid of a triangle -/
def centroid (t : APTriangle) : ℝ × ℝ := sorry

/-- The incenter of a triangle -/
def incenter (t : APTriangle) : ℝ × ℝ := sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : ℝ × ℝ → ℝ × ℝ → Prop) : Prop := sorry

/-- The line passing through two points -/
def line_through (p1 p2 : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ → Prop := sorry

/-- The side AB of the triangle -/
def side_AB (t : APTriangle) : ℝ × ℝ → ℝ × ℝ → Prop := sorry

theorem ap_triangle_centroid_incenter_parallel (t : APTriangle) :
  parallel (line_through (centroid t) (incenter t)) (side_AB t) := by
  sorry

end NUMINAMATH_CALUDE_ap_triangle_centroid_incenter_parallel_l1187_118724


namespace NUMINAMATH_CALUDE_total_repair_cost_is_4850_l1187_118746

-- Define the repair costs
def engine_labor_rate : ℕ := 75
def engine_labor_hours : ℕ := 16
def engine_part_cost : ℕ := 1200

def brake_labor_rate : ℕ := 85
def brake_labor_hours : ℕ := 10
def brake_part_cost : ℕ := 800

def tire_labor_rate : ℕ := 50
def tire_labor_hours : ℕ := 4
def tire_part_cost : ℕ := 600

-- Define the total repair cost function
def total_repair_cost : ℕ :=
  (engine_labor_rate * engine_labor_hours + engine_part_cost) +
  (brake_labor_rate * brake_labor_hours + brake_part_cost) +
  (tire_labor_rate * tire_labor_hours + tire_part_cost)

-- Theorem statement
theorem total_repair_cost_is_4850 : total_repair_cost = 4850 := by
  sorry

end NUMINAMATH_CALUDE_total_repair_cost_is_4850_l1187_118746


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1187_118707

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x, -3 ≤ x^2 - 2*a*x + a ∧ x^2 - 2*a*x + a ≤ -2) → (a = 2 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1187_118707


namespace NUMINAMATH_CALUDE_p_iff_between_two_and_three_l1187_118775

def p (x : ℝ) : Prop := x^2 - 5*x + 6 < 0

theorem p_iff_between_two_and_three :
  ∀ x : ℝ, p x ↔ 2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_p_iff_between_two_and_three_l1187_118775


namespace NUMINAMATH_CALUDE_find_x_l1187_118760

def A : Set ℝ := {0, 2, 3}
def B (x : ℝ) : Set ℝ := {x + 1, x^2 + 4}

theorem find_x : ∃ x : ℝ, A ∩ B x = {3} → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1187_118760


namespace NUMINAMATH_CALUDE_a_range_l1187_118721

def f (a x : ℝ) := x^2 - 2*a*x + 7

theorem a_range (a : ℝ) : 
  (∀ x y, 1 ≤ x ∧ x < y → f a x < f a y) → 
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_a_range_l1187_118721


namespace NUMINAMATH_CALUDE_sum_of_squares_l1187_118762

theorem sum_of_squares (x y : ℚ) (h1 : x + 2*y = 20) (h2 : 3*x + y = 19) : x^2 + y^2 = 401/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1187_118762


namespace NUMINAMATH_CALUDE_smallest_b_factorization_l1187_118766

/-- The smallest positive integer b for which x^2 + bx + 2304 factors into a product of two polynomials with integer coefficients -/
def smallest_factorizable_b : ℕ := 96

/-- Predicate to check if a polynomial factors with integer coefficients -/
def factors_with_integer_coeffs (a b c : ℤ) : Prop :=
  ∃ (p q : ℤ), ∀ (x : ℤ), a * x^2 + b * x + c = (x + p) * (x + q)

theorem smallest_b_factorization :
  (factors_with_integer_coeffs 1 smallest_factorizable_b 2304) ∧
  (∀ b : ℕ, b < smallest_factorizable_b →
    ¬(factors_with_integer_coeffs 1 b 2304)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_factorization_l1187_118766


namespace NUMINAMATH_CALUDE_product_expansion_l1187_118767

theorem product_expansion (a b c d : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 5 * x + 4) * (7 - 2 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l1187_118767


namespace NUMINAMATH_CALUDE_lcm_of_6_and_15_l1187_118718

theorem lcm_of_6_and_15 : Nat.lcm 6 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_6_and_15_l1187_118718


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l1187_118757

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    calculate the cost of one dozen pens -/
theorem cost_of_dozen_pens (total_cost : ℕ) (ratio_pen_pencil : ℕ) :
  total_cost = 260 →
  ratio_pen_pencil = 5 →
  ∃ (pen_cost : ℕ) (pencil_cost : ℕ),
    3 * pen_cost + 5 * pencil_cost = total_cost ∧
    pen_cost = ratio_pen_pencil * pencil_cost ∧
    12 * pen_cost = 780 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l1187_118757


namespace NUMINAMATH_CALUDE_boat_distance_proof_l1187_118753

/-- The distance covered by a boat traveling downstream -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

theorem boat_distance_proof (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ)
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 5)
  (h3 : time = 8) :
  distance_downstream boat_speed stream_speed time = 168 := by
  sorry

#eval distance_downstream 16 5 8

end NUMINAMATH_CALUDE_boat_distance_proof_l1187_118753


namespace NUMINAMATH_CALUDE_power_sum_prime_l1187_118735

theorem power_sum_prime (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → (2^p + 3^p = a^n) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_prime_l1187_118735


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1187_118723

def A : Set ℤ := {1, 3}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1187_118723


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l1187_118784

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) ∧
  (a.val = 18 ∧ b.val = 1) := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l1187_118784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1187_118713

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) (m : ℕ) :
  arithmetic_sequence a d →
  d ≠ 0 →
  a 3 + a 6 + a 10 + a 13 = 32 →
  a m = 8 →
  m = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1187_118713


namespace NUMINAMATH_CALUDE_abc_product_l1187_118773

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 168) (h2 : b * (c + a) = 153) (h3 : c * (a + b) = 147) :
  a * b * c = 720 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l1187_118773


namespace NUMINAMATH_CALUDE_choir_robe_cost_l1187_118745

/-- Calculates the cost of buying additional robes for a school choir. -/
theorem choir_robe_cost (total_robes : ℕ) (existing_robes : ℕ) (cost_per_robe : ℕ) : 
  total_robes = 30 → existing_robes = 12 → cost_per_robe = 2 → 
  (total_robes - existing_robes) * cost_per_robe = 36 :=
by
  sorry

#check choir_robe_cost

end NUMINAMATH_CALUDE_choir_robe_cost_l1187_118745


namespace NUMINAMATH_CALUDE_first_quartile_of_list_l1187_118738

def number_list : List ℝ := [42, 24, 30, 22, 26, 27, 33, 35]

def median (l : List ℝ) : ℝ := sorry

def first_quartile (l : List ℝ) : ℝ :=
  let m := median l
  median (l.filter (λ x => x < m))

theorem first_quartile_of_list :
  first_quartile number_list = 25 := by sorry

end NUMINAMATH_CALUDE_first_quartile_of_list_l1187_118738


namespace NUMINAMATH_CALUDE_student_arrangements_l1187_118796

/-- Represents a student with a unique height -/
structure Student :=
  (height : ℕ)

/-- The set of 7 students with different heights -/
def Students : Finset Student :=
  sorry

/-- Predicate for a valid arrangement in a row -/
def ValidRowArrangement (arrangement : List Student) : Prop :=
  sorry

/-- Predicate for a valid arrangement in two rows and three columns -/
def Valid2x3Arrangement (arrangement : List (List Student)) : Prop :=
  sorry

theorem student_arrangements :
  (∃ (arrangements : Finset (List Student)),
    (∀ arr ∈ arrangements, ValidRowArrangement arr) ∧
    Finset.card arrangements = 20) ∧
  (∃ (arrangements : Finset (List (List Student))),
    (∀ arr ∈ arrangements, Valid2x3Arrangement arr) ∧
    Finset.card arrangements = 630) :=
  sorry

end NUMINAMATH_CALUDE_student_arrangements_l1187_118796


namespace NUMINAMATH_CALUDE_gcd_13n_plus_4_7n_plus_2_max_2_l1187_118770

theorem gcd_13n_plus_4_7n_plus_2_max_2 :
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) ≤ 2) ∧
  (∃ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_gcd_13n_plus_4_7n_plus_2_max_2_l1187_118770


namespace NUMINAMATH_CALUDE_jellybean_difference_l1187_118795

theorem jellybean_difference (black green orange : ℕ) : 
  black = 8 →
  green = black + 2 →
  black + green + orange = 27 →
  green - orange = 1 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_difference_l1187_118795


namespace NUMINAMATH_CALUDE_sum_of_two_squares_condition_l1187_118702

theorem sum_of_two_squares_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ a b : ℤ, p = a^2 + b^2) ↔ p % 4 = 1 ∨ p = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_condition_l1187_118702


namespace NUMINAMATH_CALUDE_floor_product_equation_l1187_118756

theorem floor_product_equation : ∃! (x : ℝ), x > 0 ∧ x * ⌊x⌋ = 50 ∧ |x - (50 / 7)| < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_floor_product_equation_l1187_118756


namespace NUMINAMATH_CALUDE_merry_go_round_cost_per_child_l1187_118730

/-- The cost of a merry-go-round ride per child given the following conditions:
  - There are 5 children
  - 3 children rode the Ferris wheel
  - Ferris wheel cost is $5 per child
  - Everyone rode the merry-go-round
  - Each child bought 2 ice cream cones
  - Each ice cream cone costs $8
  - Total spent is $110
-/
theorem merry_go_round_cost_per_child 
  (num_children : ℕ)
  (ferris_wheel_riders : ℕ)
  (ferris_wheel_cost : ℚ)
  (ice_cream_cones_per_child : ℕ)
  (ice_cream_cone_cost : ℚ)
  (total_spent : ℚ)
  (h1 : num_children = 5)
  (h2 : ferris_wheel_riders = 3)
  (h3 : ferris_wheel_cost = 5)
  (h4 : ice_cream_cones_per_child = 2)
  (h5 : ice_cream_cone_cost = 8)
  (h6 : total_spent = 110) :
  (total_spent - (ferris_wheel_riders * ferris_wheel_cost) - (num_children * ice_cream_cones_per_child * ice_cream_cone_cost)) / num_children = 3 :=
by sorry

end NUMINAMATH_CALUDE_merry_go_round_cost_per_child_l1187_118730


namespace NUMINAMATH_CALUDE_exists_common_divisor_l1187_118750

/-- A function from positive integers to integers greater than or equal to 2 -/
def PositiveIntegerFunction := ℕ+ → ℕ

/-- The property that f(m+n) divides f(m) + f(n) for all positive integers m and n -/
def HasDivisibilityProperty (f : PositiveIntegerFunction) : Prop :=
  ∀ m n : ℕ+, (f (m + n) : ℤ) ∣ (f m + f n : ℤ)

/-- The main theorem -/
theorem exists_common_divisor
  (f : PositiveIntegerFunction)
  (h1 : ∀ n : ℕ+, f n ≥ 2)
  (h2 : HasDivisibilityProperty f) :
  ∃ c : ℕ+, c > 1 ∧ ∀ n : ℕ+, (c : ℤ) ∣ (f n : ℤ) :=
sorry

end NUMINAMATH_CALUDE_exists_common_divisor_l1187_118750


namespace NUMINAMATH_CALUDE_water_speed_calculation_l1187_118739

/-- A person's swimming speed in still water (in km/h) -/
def still_water_speed : ℝ := 4

/-- The time taken to swim against the current (in hours) -/
def time_against_current : ℝ := 3

/-- The distance swum against the current (in km) -/
def distance_against_current : ℝ := 6

/-- The speed of the water (in km/h) -/
def water_speed : ℝ := 2

theorem water_speed_calculation :
  (distance_against_current = (still_water_speed - water_speed) * time_against_current) →
  water_speed = 2 := by
sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l1187_118739


namespace NUMINAMATH_CALUDE_cartesian_product_eq_expected_set_l1187_118765

-- Define the set of possible x and y values
def X : Set ℕ := {1, 2}
def Y : Set ℕ := {1, 2}

-- Define the Cartesian product set
def cartesianProduct : Set (ℕ × ℕ) := {p | p.1 ∈ X ∧ p.2 ∈ Y}

-- Define the expected result set
def expectedSet : Set (ℕ × ℕ) := {(1, 1), (1, 2), (2, 1), (2, 2)}

-- Theorem stating that the Cartesian product is equal to the expected set
theorem cartesian_product_eq_expected_set : cartesianProduct = expectedSet := by
  sorry

end NUMINAMATH_CALUDE_cartesian_product_eq_expected_set_l1187_118765


namespace NUMINAMATH_CALUDE_percentage_difference_l1187_118781

theorem percentage_difference : (0.55 * 40) - (4/5 * 25) = 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1187_118781


namespace NUMINAMATH_CALUDE_increasing_derivative_relation_l1187_118768

open Set
open Function
open Real

-- Define the interval (a, b)
variable (a b : ℝ) (hab : a < b)

-- Define a real-valued function on the interval (a, b)
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on (a, b)
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Define the derivative of f
variable (f' : ℝ → ℝ)
variable (hf' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)

-- State the theorem
theorem increasing_derivative_relation :
  (∀ x ∈ Ioo a b, f' x > 0 → IsIncreasing f a b) ∧
  ∃ f : ℝ → ℝ, IsIncreasing f a b ∧ ¬(∀ x ∈ Ioo a b, f' x > 0) :=
sorry

end NUMINAMATH_CALUDE_increasing_derivative_relation_l1187_118768


namespace NUMINAMATH_CALUDE_system_solution_l1187_118780

theorem system_solution : ∃ (x y z : ℤ), 
  (7*x + 3*y = 2*z + 1) ∧ 
  (4*x - 5*y = 3*z - 30) ∧ 
  (x + 2*y = 5*z + 15) ∧ 
  (x = -1) ∧ (y = 2) ∧ (z = 7) := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l1187_118780


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_hcf_lcm_l1187_118782

theorem product_of_numbers_with_given_hcf_lcm :
  ∀ (a b : ℕ+),
  Nat.gcd a b = 33 →
  Nat.lcm a b = 2574 →
  a * b = 84942 :=
by sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_hcf_lcm_l1187_118782


namespace NUMINAMATH_CALUDE_jelly_overlap_l1187_118701

/-- The number of jellies -/
def num_jellies : ℕ := 12

/-- The length of each jelly in centimeters -/
def jelly_length : ℝ := 18

/-- The circumference of the ring in centimeters -/
def ring_circumference : ℝ := 210

/-- The overlapping portion of each jelly in millimeters -/
def overlap_mm : ℝ := 5

theorem jelly_overlap :
  (num_jellies : ℝ) * jelly_length - ring_circumference = num_jellies * overlap_mm / 10 := by
  sorry

end NUMINAMATH_CALUDE_jelly_overlap_l1187_118701


namespace NUMINAMATH_CALUDE_composite_function_difference_l1187_118789

theorem composite_function_difference (A B : ℝ) (h : A ≠ B) :
  let f := λ x : ℝ => A * x + B
  let g := λ x : ℝ => B * x + A
  (∀ x, f (g x) - g (f x) = 2 * (B - A)) →
  A + B = -2 := by
sorry

end NUMINAMATH_CALUDE_composite_function_difference_l1187_118789


namespace NUMINAMATH_CALUDE_square_area_subtraction_l1187_118752

theorem square_area_subtraction (s : ℝ) (x : ℝ) : 
  s = 4 → s^2 + s - x = 4 → x = 16 := by sorry

end NUMINAMATH_CALUDE_square_area_subtraction_l1187_118752


namespace NUMINAMATH_CALUDE_hyperbola_sufficient_condition_l1187_118712

-- Define the equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / (4 - m) = 1

-- Define the condition for a hyperbola with foci on the x-axis
def is_hyperbola_x_axis (m : ℝ) : Prop :=
  m - 1 > 0 ∧ 4 - m < 0

-- The theorem to prove
theorem hyperbola_sufficient_condition :
  ∃ (m : ℝ), m > 5 → is_hyperbola_x_axis m ∧
  ∃ (m' : ℝ), is_hyperbola_x_axis m' ∧ m' ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_sufficient_condition_l1187_118712


namespace NUMINAMATH_CALUDE_product_of_integers_l1187_118783

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 20)
  (diff_squares_eq : x^2 - y^2 = 40) :
  x * y = 99 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l1187_118783


namespace NUMINAMATH_CALUDE_simplify_square_root_l1187_118749

theorem simplify_square_root : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by sorry

end NUMINAMATH_CALUDE_simplify_square_root_l1187_118749


namespace NUMINAMATH_CALUDE_least_number_of_sweets_l1187_118744

theorem least_number_of_sweets (s : ℕ) : s > 0 ∧ 
  s % 6 = 5 ∧ 
  s % 8 = 3 ∧ 
  s % 9 = 6 ∧ 
  s % 11 = 10 ∧ 
  (∀ t : ℕ, t > 0 → t % 6 = 5 → t % 8 = 3 → t % 9 = 6 → t % 11 = 10 → s ≤ t) → 
  s = 2095 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_sweets_l1187_118744


namespace NUMINAMATH_CALUDE_characterization_of_f_l1187_118793

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the conditions
def NonNegative (f : RealFunction) : Prop :=
  ∀ x : ℝ, f x ≥ 0

def SatisfiesEquation (f : RealFunction) : Prop :=
  ∀ a b c d : ℝ, a * b + b * c + c * d = 0 →
    f (a - b) + f (c - d) = f a + f (b + c) + f d

-- Main theorem
theorem characterization_of_f (f : RealFunction)
  (h1 : NonNegative f)
  (h2 : SatisfiesEquation f) :
  ∃ c : ℝ, c ≥ 0 ∧ ∀ x : ℝ, f x = c * x^2 :=
sorry

end NUMINAMATH_CALUDE_characterization_of_f_l1187_118793


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_11_18_l1187_118747

theorem smallest_divisible_by_10_11_18 : ∃ n : ℕ+, (∀ m : ℕ+, 10 ∣ m ∧ 11 ∣ m ∧ 18 ∣ m → n ≤ m) ∧ 10 ∣ n ∧ 11 ∣ n ∧ 18 ∣ n :=
  by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_11_18_l1187_118747


namespace NUMINAMATH_CALUDE_van_speed_problem_l1187_118776

theorem van_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 288 →
  original_time = 6 →
  new_time_factor = 3 / 2 →
  let new_time := original_time * new_time_factor
  let new_speed := distance / new_time
  new_speed = 32 := by
sorry

end NUMINAMATH_CALUDE_van_speed_problem_l1187_118776


namespace NUMINAMATH_CALUDE_cubic_polynomials_l1187_118734

-- Define the polynomials A and B
def A (x : ℝ) : ℝ := 5 * x^3 - 6 * x^2 + 10
def B (x e f : ℝ) : ℝ := x^2 + e * x + f

-- Define the alternative form of A
def A_alt (x a b c d : ℝ) : ℝ := a * (x - 1)^3 + b * (x - 1)^2 + c * (x - 1) + d

-- State the theorem
theorem cubic_polynomials (a b c d e f : ℝ) (hf : f ≠ 0) (he : e ≠ 0) :
  (∀ x, A x = A_alt x a b c d) →
  (∀ x, ∃ k₁ k₂ k₃, A x + B x e f = k₁ * x^3 + k₂ * x^2 + k₃ * x + (10 + f)) →
  (a + b + c = 17) ∧
  (∃ x₀, ∀ x, B x e f = 0 ↔ x = x₀) →
  (f = -10 ∧ a + b + c = 17 ∧ e^2 = 4 * f) := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_l1187_118734


namespace NUMINAMATH_CALUDE_polynomial_positive_intervals_l1187_118792

/-- The polynomial (x+1)(x-1)(x-3) is positive if and only if x is in the interval (-1, 1) or (3, ∞) -/
theorem polynomial_positive_intervals (x : ℝ) : 
  (x + 1) * (x - 1) * (x - 3) > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_positive_intervals_l1187_118792


namespace NUMINAMATH_CALUDE_triangle_angle_range_l1187_118728

theorem triangle_angle_range (A B C : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Triangle conditions
  2 * Real.sin A + Real.sin B = Real.sqrt 3 * Real.sin C →  -- Given equation
  π / 6 ≤ A ∧ A ≤ π / 2 := by  -- Conclusion to prove
sorry

end NUMINAMATH_CALUDE_triangle_angle_range_l1187_118728


namespace NUMINAMATH_CALUDE_paint_combinations_l1187_118761

theorem paint_combinations (num_colors num_methods : ℕ) :
  num_colors = 5 → num_methods = 4 → num_colors * num_methods = 20 := by
  sorry

end NUMINAMATH_CALUDE_paint_combinations_l1187_118761


namespace NUMINAMATH_CALUDE_triangle_condition_implies_right_angle_l1187_118704

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition from the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a - 3)^2 + Real.sqrt (t.b - 4) + |t.c - 5| = 0

-- Define what it means for a triangle to be right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- Theorem statement
theorem triangle_condition_implies_right_angle (t : Triangle) :
  satisfiesCondition t → isRightTriangle t :=
by sorry

end NUMINAMATH_CALUDE_triangle_condition_implies_right_angle_l1187_118704


namespace NUMINAMATH_CALUDE_candy_bar_cost_l1187_118703

/-- Given that the total cost of 2 candy bars is $4 and each candy bar costs the same amount,
    prove that the cost of each candy bar is $2. -/
theorem candy_bar_cost (total_cost : ℝ) (num_bars : ℕ) (cost_per_bar : ℝ) : 
  total_cost = 4 → num_bars = 2 → total_cost = num_bars * cost_per_bar → cost_per_bar = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l1187_118703


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1187_118799

theorem min_reciprocal_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 3) :
  (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x + y + z = 3 → 1/x + 1/y + 1/z ≥ 1/a + 1/b + 1/c) →
  1/a + 1/b + 1/c = 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1187_118799


namespace NUMINAMATH_CALUDE_committee_formation_count_l1187_118758

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of people in the club -/
def total_people : ℕ := 12

/-- The number of board members -/
def board_members : ℕ := 3

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- The number of regular members (non-board members) -/
def regular_members : ℕ := total_people - board_members

theorem committee_formation_count :
  choose total_people committee_size - choose regular_members committee_size = 666 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l1187_118758


namespace NUMINAMATH_CALUDE_disjoint_sets_cardinality_relation_l1187_118736

theorem disjoint_sets_cardinality_relation (a b : ℕ+) (A B : Finset ℤ) :
  Disjoint A B →
  (∀ i : ℤ, i ∈ A ∪ B → (i + a) ∈ A ∨ (i - b) ∈ B) →
  a * A.card = b * B.card := by
  sorry

end NUMINAMATH_CALUDE_disjoint_sets_cardinality_relation_l1187_118736


namespace NUMINAMATH_CALUDE_product_digit_count_l1187_118764

def x : ℕ := 3659893456789325678
def y : ℕ := 342973489379256

theorem product_digit_count :
  (String.length (toString (x * y))) = 34 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_count_l1187_118764


namespace NUMINAMATH_CALUDE_three_digit_sum_not_always_three_digits_l1187_118731

theorem three_digit_sum_not_always_three_digits : ∃ (a b : ℕ), 
  100 ≤ a ∧ a ≤ 999 ∧ 100 ≤ b ∧ b ≤ 999 ∧ 1000 ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_not_always_three_digits_l1187_118731


namespace NUMINAMATH_CALUDE_narcissistic_numbers_l1187_118794

theorem narcissistic_numbers : 
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ 
    n = (n / 100)^3 + ((n % 100) / 10)^3 + (n % 10)^3} = 
  {153, 370, 371, 407} := by
sorry

end NUMINAMATH_CALUDE_narcissistic_numbers_l1187_118794


namespace NUMINAMATH_CALUDE_solve_for_a_l1187_118742

theorem solve_for_a (a : ℝ) :
  (∀ x, |2*x - a| + a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_solve_for_a_l1187_118742


namespace NUMINAMATH_CALUDE_six_people_arrangement_l1187_118754

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

def permutations (n : ℕ) (k : ℕ) : ℕ := 
  if k > n then 0
  else factorial n / factorial (n - k)

theorem six_people_arrangement : 
  let total_arrangements := permutations 6 6
  let a_head_b_tail := permutations 4 4
  let a_head_b_not_tail := permutations 4 1 * permutations 4 4
  let a_not_head_b_tail := permutations 4 1 * permutations 4 4
  total_arrangements - a_head_b_tail - a_head_b_not_tail - a_not_head_b_tail = 504 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l1187_118754


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1187_118727

/-- A geometric sequence is a sequence where the ratio of any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that for a geometric sequence satisfying certain conditions, 
    the sum of specific terms equals 3. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h1 : a 1 + a 3 = 8) 
    (h2 : a 5 + a 7 = 4) : 
  a 9 + a 11 + a 13 + a 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1187_118727


namespace NUMINAMATH_CALUDE_workers_problem_l1187_118720

/-- Given a number of workers that can complete a job in 25 days, 
    and adding 10 workers reduces the time to 15 days, 
    prove that the original number of workers is 15. -/
theorem workers_problem (W : ℕ) : 
  W * 25 = (W + 10) * 15 → W = 15 := by sorry

end NUMINAMATH_CALUDE_workers_problem_l1187_118720


namespace NUMINAMATH_CALUDE_hannah_restaurant_bill_hannah_restaurant_bill_proof_l1187_118726

/-- The total amount Hannah spent on the entree and dessert is $23, 
    given that the entree costs $14 and it is $5 more than the dessert. -/
theorem hannah_restaurant_bill : ℕ → ℕ → ℕ → Prop :=
  fun entree_cost dessert_cost total_cost =>
    (entree_cost = 14) →
    (entree_cost = dessert_cost + 5) →
    (total_cost = entree_cost + dessert_cost) →
    (total_cost = 23)

/-- Proof of hannah_restaurant_bill -/
theorem hannah_restaurant_bill_proof : hannah_restaurant_bill 14 9 23 := by
  sorry

end NUMINAMATH_CALUDE_hannah_restaurant_bill_hannah_restaurant_bill_proof_l1187_118726


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1187_118743

def p (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 2
def q (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem coefficient_of_x_cubed (x : ℝ) : 
  ∃ (a b c d e : ℝ), p x * q x = a * x^5 + b * x^4 - 25 * x^3 + c * x^2 + d * x + e :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1187_118743


namespace NUMINAMATH_CALUDE_book_profit_percentage_l1187_118733

/-- Given a book with cost price $1800, if selling it for $90 more than the initial
    selling price would result in a 15% profit, then the initial profit percentage is 10% -/
theorem book_profit_percentage (cost_price : ℝ) (additional_price : ℝ) 
  (higher_profit_percentage : ℝ) (initial_selling_price : ℝ) :
  cost_price = 1800 →
  additional_price = 90 →
  higher_profit_percentage = 15 →
  initial_selling_price + additional_price = cost_price * (1 + higher_profit_percentage / 100) →
  (initial_selling_price - cost_price) / cost_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_percentage_l1187_118733


namespace NUMINAMATH_CALUDE_stuart_reward_points_l1187_118779

/-- Represents the reward points earned per $25 spent at the Gauss Store. -/
def reward_points_per_unit : ℕ := 5

/-- Represents the amount Stuart spends at the Gauss Store in dollars. -/
def stuart_spend : ℕ := 200

/-- Represents the dollar amount that earns one unit of reward points. -/
def dollars_per_unit : ℕ := 25

/-- Calculates the number of reward points earned based on the amount spent. -/
def calculate_reward_points (spend : ℕ) : ℕ :=
  (spend / dollars_per_unit) * reward_points_per_unit

/-- Theorem stating that Stuart earns 40 reward points when spending $200 at the Gauss Store. -/
theorem stuart_reward_points : 
  calculate_reward_points stuart_spend = 40 := by
  sorry

end NUMINAMATH_CALUDE_stuart_reward_points_l1187_118779


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_distance_l1187_118729

/-- The ellipse with equation 9x² + 16y² = 114 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | 9 * p.1^2 + 16 * p.2^2 = 114}

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The distance from a point to a line defined by two points -/
noncomputable def distanceToLine (p q r : ℝ × ℝ) : ℝ :=
  sorry

theorem ellipse_perpendicular_distance :
  ∀ (P Q : ℝ × ℝ),
  P ∈ Ellipse →
  Q ∈ Ellipse →
  (P.1 - O.1) * (Q.1 - O.1) + (P.2 - O.2) * (Q.2 - O.2) = 0 →
  distanceToLine O P Q = 12/5 := by
    sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_distance_l1187_118729


namespace NUMINAMATH_CALUDE_world_book_day_purchase_l1187_118797

theorem world_book_day_purchase (planned_spending : ℝ) (price_reduction : ℝ) (additional_books_ratio : ℝ) :
  planned_spending = 180 →
  price_reduction = 9 →
  additional_books_ratio = 1/4 →
  ∃ (planned_books actual_books : ℝ),
    planned_books > 0 ∧
    actual_books = planned_books * (1 + additional_books_ratio) ∧
    planned_spending / planned_books - planned_spending / actual_books = price_reduction ∧
    actual_books = 5 := by
  sorry

end NUMINAMATH_CALUDE_world_book_day_purchase_l1187_118797


namespace NUMINAMATH_CALUDE_new_year_markup_verify_new_year_markup_l1187_118706

/-- Calculates the New Year season markup percentage given other price adjustments and final profit -/
theorem new_year_markup (initial_markup : ℝ) (february_discount : ℝ) (final_profit : ℝ) : ℝ :=
  let new_year_markup := 
    ((1 + final_profit) / ((1 + initial_markup) * (1 - february_discount)) - 1) * 100
  by
    -- The proof would go here
    sorry

/-- Verifies that the New Year markup is 25% given the problem conditions -/
theorem verify_new_year_markup : 
  new_year_markup 0.20 0.09 0.365 = 25 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_new_year_markup_verify_new_year_markup_l1187_118706


namespace NUMINAMATH_CALUDE_xyz_value_l1187_118722

theorem xyz_value (x y z : ℝ) 
  (eq1 : (x + y + z) * (x * y + x * z + y * z) = 35)
  (eq2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 23 / 3 := by sorry

end NUMINAMATH_CALUDE_xyz_value_l1187_118722


namespace NUMINAMATH_CALUDE_larrys_to_keiths_score_ratio_l1187_118719

/-- Given that Keith scored 3 points, Danny scored 5 more marks than Larry,
    and the total amount of marks scored by the three students is 26,
    prove that the ratio of Larry's score to Keith's score is 3:1 -/
theorem larrys_to_keiths_score_ratio (keith_score larry_score danny_score : ℕ) : 
  keith_score = 3 →
  danny_score = larry_score + 5 →
  keith_score + larry_score + danny_score = 26 →
  larry_score / keith_score = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_larrys_to_keiths_score_ratio_l1187_118719


namespace NUMINAMATH_CALUDE_problem_solution_l1187_118709

theorem problem_solution :
  ∀ m n : ℕ+,
  (m : ℝ)^2 - (n : ℝ) = 32 →
  (∃ x : ℝ, x = (m + n^(1/2))^(1/5) + (m - n^(1/2))^(1/5) ∧ x^5 - 10*x^3 + 20*x - 40 = 0) →
  (m : ℕ) + n = 388 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1187_118709


namespace NUMINAMATH_CALUDE_unique_solution_system_l1187_118798

theorem unique_solution_system :
  ∃! (x y z : ℝ), x + 3 * y = 10 ∧ y = 3 ∧ 2 * x - y + z = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1187_118798


namespace NUMINAMATH_CALUDE_complex_arithmetic_simplification_l1187_118732

theorem complex_arithmetic_simplification :
  ((6 - 3 * Complex.I) - (2 + 4 * Complex.I)) * (2 * Complex.I) = 14 + 8 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_simplification_l1187_118732


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1187_118716

theorem quadratic_inequality_theorem (c : ℝ) : 
  (∀ (a b : ℝ), (c^2 - 2*a*c + b) * (c^2 + 2*a*c + b) ≥ a^2 - 2*a^2 + b) ↔ 
  (c = 1/2 ∨ c = -1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1187_118716


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l1187_118772

theorem sphere_surface_area_from_cube (a : ℝ) (h : a > 0) :
  ∃ (cube_edge : ℝ) (sphere_radius : ℝ),
    cube_edge > 0 ∧
    sphere_radius > 0 ∧
    (6 * cube_edge ^ 2 = a) ∧
    (cube_edge * Real.sqrt 3 = 2 * sphere_radius) ∧
    (4 * π * sphere_radius ^ 2 = π / 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l1187_118772


namespace NUMINAMATH_CALUDE_ellipse_equation_l1187_118785

/-- An ellipse with center at the origin, a focus on a coordinate axis,
    eccentricity √3/2, and passing through (2,0) -/
structure Ellipse where
  -- The focus is either on the x-axis or y-axis
  focus_on_axis : Bool
  -- The equation of the ellipse in the form x²/a² + y²/b² = 1
  a : ℝ
  b : ℝ
  -- Conditions
  center_origin : a > 0 ∧ b > 0
  passes_through_2_0 : (2 : ℝ)^2 / a^2 + 0^2 / b^2 = 1
  eccentricity : Real.sqrt (1 - b^2 / a^2) = Real.sqrt 3 / 2

/-- The equation of the ellipse is either x²/4 + y² = 1 or x²/4 + y²/16 = 1 -/
theorem ellipse_equation (e : Ellipse) :
  (e.a^2 = 4 ∧ e.b^2 = 1) ∨ (e.a^2 = 16 ∧ e.b^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1187_118785


namespace NUMINAMATH_CALUDE_talia_age_in_seven_years_l1187_118725

/-- Proves Talia's age in seven years given the conditions of the problem -/
theorem talia_age_in_seven_years :
  ∀ (talia_age mom_age dad_age : ℕ),
    mom_age = 3 * talia_age →
    dad_age + 3 = mom_age →
    dad_age = 36 →
    talia_age + 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_talia_age_in_seven_years_l1187_118725


namespace NUMINAMATH_CALUDE_pedal_triangle_area_l1187_118748

/-- Given a triangle with area S and circumradius R, and a point at distance d from the circumcenter,
    S₁ is the area of the triangle formed by the feet of the perpendiculars from this point
    to the sides of the original triangle. -/
theorem pedal_triangle_area (S R d S₁ : ℝ) (h_pos_S : S > 0) (h_pos_R : R > 0) :
  S₁ = (S / 4) * |1 - (d^2 / R^2)| := by
  sorry

end NUMINAMATH_CALUDE_pedal_triangle_area_l1187_118748


namespace NUMINAMATH_CALUDE_circles_have_three_common_tangents_l1187_118769

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 9

-- Define the centers and radii
def center1 : ℝ × ℝ := (-1, -2)
def center2 : ℝ × ℝ := (2, 2)
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem statement
theorem circles_have_three_common_tangents :
  ∃! (n : ℕ), n = 3 ∧ 
  (∃ (tangents : Finset (ℝ → ℝ)), tangents.card = n ∧ 
    (∀ f ∈ tangents, ∀ x y : ℝ, 
      (circle1 x y → (y = f x ∨ y = -f x)) ∧ 
      (circle2 x y → (y = f x ∨ y = -f x)))) := by sorry

end NUMINAMATH_CALUDE_circles_have_three_common_tangents_l1187_118769


namespace NUMINAMATH_CALUDE_fraction_equality_l1187_118787

theorem fraction_equality (a b c d : ℚ) 
  (h1 : b / a = 1 / 2)
  (h2 : d / c = 1 / 2)
  (h3 : a ≠ c) :
  (2 * b - d) / (2 * a - c) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1187_118787


namespace NUMINAMATH_CALUDE_escalator_walking_rate_l1187_118751

/-- Calculates the walking rate of a person on an escalator -/
theorem escalator_walking_rate 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 15)
  (h2 : escalator_length = 180)
  (h3 : time_taken = 10) :
  (escalator_length / time_taken) - escalator_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_escalator_walking_rate_l1187_118751


namespace NUMINAMATH_CALUDE_original_number_l1187_118777

theorem original_number (x : ℝ) : 
  (x * 10) * 0.001 = 0.375 → x = 37.5 := by
sorry

end NUMINAMATH_CALUDE_original_number_l1187_118777


namespace NUMINAMATH_CALUDE_two_p_plus_q_l1187_118717

theorem two_p_plus_q (p q : ℚ) (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_p_plus_q_l1187_118717


namespace NUMINAMATH_CALUDE_conference_theorem_l1187_118774

/-- A graph with vertices labeled 1 to n, where edges are colored either red or blue -/
structure ColoredGraph (n : ℕ) where
  edge_color : Fin n → Fin n → Bool

/-- Predicate to check if a subgraph of 4 vertices satisfies the given conditions -/
def valid_subgraph (G : ColoredGraph n) (a b c d : Fin n) : Prop :=
  let edges := [G.edge_color a b, G.edge_color a c, G.edge_color a d, 
                G.edge_color b c, G.edge_color b d, G.edge_color c d]
  let red_count := (edges.filter id).length
  let blue_count := (edges.filter not).length
  (red_count + blue_count) % 2 = 0 ∧ 
  red_count > 0 ∧ 
  (blue_count = 0 ∨ blue_count ≥ red_count)

/-- Theorem statement -/
theorem conference_theorem :
  ∃ (G : ColoredGraph 2017),
    (∀ (a b c d : Fin 2017), valid_subgraph G a b c d) →
    ∃ (S : Finset (Fin 2017)),
      S.card = 673 ∧
      ∀ (x y : Fin 2017), x ∈ S → y ∈ S → x ≠ y → G.edge_color x y = true :=
by sorry

end NUMINAMATH_CALUDE_conference_theorem_l1187_118774


namespace NUMINAMATH_CALUDE_function_symmetry_l1187_118791

/-- The function f(x) defined as √3 sin(2x) + 2 cos²x is symmetric about the line x = π/6 -/
theorem function_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2
  f (π / 6 + x) = f (π / 6 - x) :=
by sorry

end NUMINAMATH_CALUDE_function_symmetry_l1187_118791


namespace NUMINAMATH_CALUDE_circle_symmetry_l1187_118788

-- Define the line l: x + y = 0
def line_l (x y : ℝ) : Prop := x + y = 0

-- Define circle C: (x-2)^2 + (y-1)^2 = 4
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Define circle C': (x+1)^2 + (y+2)^2 = 4
def circle_C' (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 4

-- Function to reflect a point (x, y) across the line l
def reflect_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

-- Theorem stating that C' is symmetric to C with respect to l
theorem circle_symmetry :
  ∀ x y : ℝ, circle_C x y ↔ circle_C' (reflect_point x y).1 (reflect_point x y).2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1187_118788


namespace NUMINAMATH_CALUDE_lost_money_proof_l1187_118737

def money_lost (initial_amount spent_amount remaining_amount : ℕ) : ℕ :=
  (initial_amount - spent_amount) - remaining_amount

theorem lost_money_proof (initial_amount spent_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 2)
  (h3 : remaining_amount = 3) :
  money_lost initial_amount spent_amount remaining_amount = 6 := by
  sorry

#eval money_lost 11 2 3

end NUMINAMATH_CALUDE_lost_money_proof_l1187_118737


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l1187_118771

/-- Given a triangle ABC with B > A, prove that C₁ - C₂ = B - A,
    where C₁ and C₂ are parts of angle C divided by the altitude,
    and C₂ is adjacent to side a. -/
theorem triangle_angle_relation (A B C C₁ C₂ : Real) : 
  B > A → 
  C = C₁ + C₂ → 
  A + B + C = Real.pi → 
  C₁ - C₂ = B - A := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l1187_118771


namespace NUMINAMATH_CALUDE_net_population_increase_per_day_l1187_118786

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℚ := 5

/-- Represents the death rate in people per two seconds -/
def death_rate : ℚ := 3

/-- Calculates the net population increase per second -/
def net_increase_per_second : ℚ := (birth_rate - death_rate) / 2

/-- Theorem stating the net population increase over one day -/
theorem net_population_increase_per_day :
  (net_increase_per_second * seconds_per_day : ℚ) = 86400 := by
  sorry

end NUMINAMATH_CALUDE_net_population_increase_per_day_l1187_118786


namespace NUMINAMATH_CALUDE_not_perfect_square_if_last_two_digits_odd_l1187_118705

-- Define a function to get the last two digits of an integer
def lastTwoDigits (n : ℤ) : ℤ × ℤ :=
  let d₁ := n % 10
  let d₂ := (n / 10) % 10
  (d₂, d₁)

-- Define a predicate for an integer being odd
def isOdd (n : ℤ) : Prop := n % 2 ≠ 0

-- Theorem statement
theorem not_perfect_square_if_last_two_digits_odd (n : ℤ) :
  let (d₂, d₁) := lastTwoDigits n
  isOdd d₂ ∧ isOdd d₁ → ¬∃ (m : ℤ), n = m ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_if_last_two_digits_odd_l1187_118705


namespace NUMINAMATH_CALUDE_unique_solution_is_76_l1187_118715

/-- The cubic equation in question -/
def cubic_equation (p x : ℝ) : ℝ := 5*x^3 - 5*(p+1)*x^2 + (71*p-1)*x + 1 - 66*p

/-- A function that checks if a number is a natural number -/
def is_natural (x : ℝ) : Prop := ∃ n : ℕ, x = n

/-- The main theorem stating that p = 76 is the unique solution -/
theorem unique_solution_is_76 :
  ∃! p : ℝ, p = 76 ∧ 
    ∃ x y : ℝ, x ≠ y ∧ 
      is_natural x ∧ is_natural y ∧
      cubic_equation p x = 0 ∧ 
      cubic_equation p y = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_is_76_l1187_118715


namespace NUMINAMATH_CALUDE_complex_power_30_150_deg_l1187_118740

theorem complex_power_30_150_deg : (Complex.exp (Complex.I * Real.pi * (5/6)))^30 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_30_150_deg_l1187_118740


namespace NUMINAMATH_CALUDE_restaurant_table_difference_l1187_118755

/-- Represents the number of tables and seating capacity in a restaurant --/
structure Restaurant where
  new_tables : ℕ
  original_tables : ℕ
  new_table_capacity : ℕ
  original_table_capacity : ℕ

/-- Calculates the total number of tables in the restaurant --/
def Restaurant.total_tables (r : Restaurant) : ℕ :=
  r.new_tables + r.original_tables

/-- Calculates the total seating capacity of the restaurant --/
def Restaurant.total_capacity (r : Restaurant) : ℕ :=
  r.new_tables * r.new_table_capacity + r.original_tables * r.original_table_capacity

/-- Theorem stating the difference between new and original tables --/
theorem restaurant_table_difference (r : Restaurant) 
  (h1 : r.total_tables = 40)
  (h2 : r.total_capacity = 212)
  (h3 : r.new_table_capacity = 6)
  (h4 : r.original_table_capacity = 4) :
  r.new_tables - r.original_tables = 12 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_table_difference_l1187_118755
