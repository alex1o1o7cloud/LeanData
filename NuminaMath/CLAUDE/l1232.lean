import Mathlib

namespace NUMINAMATH_CALUDE_probability_solution_l1232_123293

def probability_equation (p q : ℝ) : Prop :=
  q = 1 - p ∧ 
  (Nat.choose 10 7 : ℝ) * p^7 * q^3 = (Nat.choose 10 6 : ℝ) * p^6 * q^4

theorem probability_solution :
  ∀ p q : ℝ, probability_equation p q → p = 7/11 := by
  sorry

end NUMINAMATH_CALUDE_probability_solution_l1232_123293


namespace NUMINAMATH_CALUDE_A_intersect_B_l1232_123232

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x : ℝ | (x + 1) * (x - 2) < 0}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1232_123232


namespace NUMINAMATH_CALUDE_percentage_of_hindu_boys_l1232_123279

theorem percentage_of_hindu_boys (total_boys : ℕ) 
  (muslim_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  muslim_percentage = 44 / 100 →
  sikh_percentage = 10 / 100 →
  other_boys = 153 →
  (total_boys - (muslim_percentage * total_boys + sikh_percentage * total_boys + other_boys)) / total_boys = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_hindu_boys_l1232_123279


namespace NUMINAMATH_CALUDE_least_squares_for_25x25_l1232_123200

theorem least_squares_for_25x25 (n : Nat) (h1 : n = 25) (h2 : n * n = 625) :
  ∃ f : Nat → Nat, f n ≥ (n^2 - 1) / 2 ∧ f n ≥ 312 := by
  sorry

end NUMINAMATH_CALUDE_least_squares_for_25x25_l1232_123200


namespace NUMINAMATH_CALUDE_other_communities_count_l1232_123208

theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) :
  total = 1500 →
  muslim_percent = 37.5 / 100 →
  hindu_percent = 25.6 / 100 →
  sikh_percent = 8.4 / 100 →
  ↑(round ((1 - (muslim_percent + hindu_percent + sikh_percent)) * total)) = 428 :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l1232_123208


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_in_right_triangle_l1232_123273

/-- For a right-angled triangle with perimeter k, the radius r of the largest inscribed circle
    is given by r = k/2 * (3 - 2√2). -/
theorem largest_inscribed_circle_in_right_triangle (k : ℝ) (h : k > 0) :
  ∃ (r : ℝ), r = k / 2 * (3 - 2 * Real.sqrt 2) ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a + b + c = k →   -- perimeter condition
  2 * (a * b) / (a + b + c) ≤ r :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_in_right_triangle_l1232_123273


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l1232_123228

/-- Given two cubes with edge lengths a and b, where a/b = 3/1 and the volume of the cube
    with edge length a is 27 units, prove that the volume of the cube with edge length b is 1 unit. -/
theorem cube_volume_ratio (a b : ℝ) (h1 : a / b = 3 / 1) (h2 : a^3 = 27) : b^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l1232_123228


namespace NUMINAMATH_CALUDE_students_meeting_time_l1232_123224

/-- Two students walking towards each other -/
theorem students_meeting_time 
  (distance : ℝ) 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (h1 : distance = 350) 
  (h2 : speed1 = 1.6) 
  (h3 : speed2 = 1.9) : 
  distance / (speed1 + speed2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_students_meeting_time_l1232_123224


namespace NUMINAMATH_CALUDE_range_of_m_l1232_123202

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 5) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 1) →
  m ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1232_123202


namespace NUMINAMATH_CALUDE_circle_point_range_l1232_123229

theorem circle_point_range (a : ℝ) : 
  ((-1 + a)^2 + (-1 - a)^2 < 4) → (-1 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_point_range_l1232_123229


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l1232_123271

/-- Given three vectors OA, OB, and OC in ℝ², where points A, B, and C are collinear,
    prove that the x-coordinate of OA is 18. -/
theorem collinear_points_k_value (k : ℝ) :
  let OA : ℝ × ℝ := (k, 12)
  let OB : ℝ × ℝ := (4, 5)
  let OC : ℝ × ℝ := (10, 8)
  (∃ (t : ℝ), OC - OA = t • (OB - OA)) →
  k = 18 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l1232_123271


namespace NUMINAMATH_CALUDE_inequality_proof_l1232_123286

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1/2) :
  (1 - a^2 + c^2) / (c * (a + 2*b)) + 
  (1 - b^2 + a^2) / (a * (b + 2*c)) + 
  (1 - c^2 + b^2) / (b * (c + 2*a)) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1232_123286


namespace NUMINAMATH_CALUDE_min_integer_solution_l1232_123260

def is_solution (x : ℤ) : Prop :=
  (3 - x > 0) ∧ ((4 * x : ℚ) / 3 + 3 / 2 > -x / 6)

theorem min_integer_solution :
  is_solution 0 ∧ ∀ y : ℤ, y < 0 → ¬is_solution y :=
sorry

end NUMINAMATH_CALUDE_min_integer_solution_l1232_123260


namespace NUMINAMATH_CALUDE_guppies_theorem_l1232_123211

def guppies_problem (haylee jose charliz nicolai : ℕ) : Prop :=
  haylee = 3 * 12 ∧
  jose = haylee / 2 ∧
  charliz = jose / 3 ∧
  nicolai = 4 * charliz ∧
  haylee + jose + charliz + nicolai = 84

theorem guppies_theorem : ∃ haylee jose charliz nicolai : ℕ, guppies_problem haylee jose charliz nicolai :=
sorry

end NUMINAMATH_CALUDE_guppies_theorem_l1232_123211


namespace NUMINAMATH_CALUDE_larger_integer_value_l1232_123207

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 3 / 2)
  (h_product : (a : ℕ) * b = 180) :
  (a : ℝ) = 3 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l1232_123207


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1232_123296

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1232_123296


namespace NUMINAMATH_CALUDE_money_distribution_l1232_123292

theorem money_distribution (x : ℝ) (h : x > 0) :
  let moe_original := 6 * x
  let loki_original := 5 * x
  let kai_original := 2 * x
  let total_original := moe_original + loki_original + kai_original
  let ott_received := 3 * x
  ott_received / total_original = 3 / 13 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1232_123292


namespace NUMINAMATH_CALUDE_tim_total_amount_l1232_123217

-- Define the value of each coin type
def nickel_value : ℚ := 0.05
def dime_value : ℚ := 0.10
def half_dollar_value : ℚ := 0.50

-- Define the number of each coin type Tim received
def nickels_from_shining : ℕ := 3
def dimes_from_shining : ℕ := 13
def dimes_from_tip_jar : ℕ := 7
def half_dollars_from_tip_jar : ℕ := 9

-- Calculate the total amount Tim received
def total_amount : ℚ :=
  nickels_from_shining * nickel_value +
  (dimes_from_shining + dimes_from_tip_jar) * dime_value +
  half_dollars_from_tip_jar * half_dollar_value

-- Theorem statement
theorem tim_total_amount : total_amount = 6.65 := by
  sorry

end NUMINAMATH_CALUDE_tim_total_amount_l1232_123217


namespace NUMINAMATH_CALUDE_circle_intersection_exists_l1232_123294

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given elements
variable (A B : Point)
variable (S : Circle)
variable (α : ℝ)

-- Define the intersection angle between two circles
def intersectionAngle (c1 c2 : Circle) : ℝ := sorry

-- Define a function to check if a point is on a circle
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Theorem statement
theorem circle_intersection_exists :
  ∃ (C : Circle), isOnCircle A C ∧ isOnCircle B C ∧ intersectionAngle C S = α := by sorry

end NUMINAMATH_CALUDE_circle_intersection_exists_l1232_123294


namespace NUMINAMATH_CALUDE_sum_ge_sum_sqrt_products_l1232_123291

theorem sum_ge_sum_sqrt_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (a * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_ge_sum_sqrt_products_l1232_123291


namespace NUMINAMATH_CALUDE_smallest_x_and_y_l1232_123288

theorem smallest_x_and_y (x y : ℕ+) (h : (3 : ℚ) / 4 = y / (242 + x)) : 
  (x = 2 ∧ y = 183) ∧ ∀ (x' y' : ℕ+), ((3 : ℚ) / 4 = y' / (242 + x')) → x ≤ x' :=
sorry

end NUMINAMATH_CALUDE_smallest_x_and_y_l1232_123288


namespace NUMINAMATH_CALUDE_red_balls_count_l1232_123268

theorem red_balls_count (x : ℕ) (h : (4 : ℝ) / (x + 4) = (1 : ℝ) / 5) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l1232_123268


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_l1232_123214

/-- Two planar vectors a and b are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_k (k : ℝ) :
  let a : ℝ × ℝ := (k, 3)
  let b : ℝ × ℝ := (1, 4)
  perpendicular a b → k = -12 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_l1232_123214


namespace NUMINAMATH_CALUDE_min_value_of_m_range_of_x_l1232_123276

-- Define the conditions
def conditions (a b m : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a^2 + b^2 = 9/2 ∧ a + b ≤ m

-- Part I: Minimum value of m
theorem min_value_of_m (a b m : ℝ) (h : conditions a b m) :
  m ≥ 3 :=
sorry

-- Part II: Range of x
theorem range_of_x (x : ℝ) :
  (∀ a b m, conditions a b m → 2*|x-1| + |x| ≥ a + b) →
  x ≤ -1/3 ∨ x ≥ 5/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_m_range_of_x_l1232_123276


namespace NUMINAMATH_CALUDE_dance_troupe_arrangement_l1232_123218

theorem dance_troupe_arrangement (n : ℕ) : n > 0 ∧ 
  6 ∣ n ∧ 9 ∣ n ∧ 12 ∣ n ∧ 5 ∣ n → n ≥ 180 :=
by sorry

end NUMINAMATH_CALUDE_dance_troupe_arrangement_l1232_123218


namespace NUMINAMATH_CALUDE_circle_range_theta_l1232_123226

/-- The range of θ for a circle with center (2cos θ, 2sin θ) and radius 1,
    where all points (x,y) on the circle satisfy x ≤ y -/
theorem circle_range_theta :
  ∀ θ : ℝ,
  (∀ x y : ℝ, (x - 2 * Real.cos θ)^2 + (y - 2 * Real.sin θ)^2 = 1 → x ≤ y) →
  0 ≤ θ →
  θ ≤ 2 * Real.pi →
  5 * Real.pi / 12 ≤ θ ∧ θ ≤ 13 * Real.pi / 12 :=
by sorry

end NUMINAMATH_CALUDE_circle_range_theta_l1232_123226


namespace NUMINAMATH_CALUDE_raised_beds_planks_l1232_123269

/-- Calculates the number of 8-foot long planks needed for raised beds --/
def planks_needed (num_beds : ℕ) (bed_height : ℕ) (bed_width : ℕ) (bed_length : ℕ) (plank_width : ℕ) (plank_length : ℕ) : ℕ :=
  let long_sides := 2 * bed_height
  let short_sides := 2 * bed_height * bed_width / plank_length
  let planks_per_bed := long_sides + short_sides
  num_beds * planks_per_bed

theorem raised_beds_planks :
  planks_needed 10 2 2 8 1 8 = 50 := by
  sorry

end NUMINAMATH_CALUDE_raised_beds_planks_l1232_123269


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l1232_123209

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 25 ∧ initial_mean = 190 ∧ incorrect_value = 130 ∧ correct_value = 165 →
  (n : ℚ) * initial_mean - incorrect_value + correct_value = n * 191.4 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l1232_123209


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sides_l1232_123251

theorem regular_polygon_interior_angle_sides : ∀ n : ℕ,
  n > 2 →
  (180 * (n - 2) : ℝ) / n = 150 →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sides_l1232_123251


namespace NUMINAMATH_CALUDE_green_balls_count_l1232_123242

theorem green_balls_count (total : ℕ) (blue : ℕ) : 
  total = 40 → 
  blue = 11 → 
  ∃ (red green : ℕ), 
    red = 2 * blue ∧ 
    green = total - (red + blue) ∧ 
    green = 7 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l1232_123242


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l1232_123215

/-- Given a convex hexagon ABCDEF with the following properties:
  - Angles A, B, and C are congruent
  - Angles D and E are congruent
  - Angle A is 30° less than angle D
  - Angle F is equal to angle A
  Prove that the measure of angle D is 140° -/
theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  A = B ∧ B = C ∧                      -- Angles A, B, and C are congruent
  D = E ∧                              -- Angles D and E are congruent
  A = D - 30 ∧                         -- Angle A is 30° less than angle D
  F = A ∧                              -- Angle F is equal to angle A
  A + B + C + D + E + F = 720          -- Sum of angles in a hexagon
  → D = 140 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l1232_123215


namespace NUMINAMATH_CALUDE_dogs_in_park_l1232_123245

theorem dogs_in_park (total_legs : ℕ) (legs_per_dog : ℕ) (h1 : total_legs = 436) (h2 : legs_per_dog = 4) :
  total_legs / legs_per_dog = 109 := by
  sorry

end NUMINAMATH_CALUDE_dogs_in_park_l1232_123245


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_l1232_123241

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum :
  unitsDigit (sumFactorials 15) = 3 :=
by
  sorry

/- Hint: You may want to use the following lemma -/
lemma units_digit_factorial_ge_5 (n : ℕ) (h : n ≥ 5) :
  unitsDigit (factorial n) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_l1232_123241


namespace NUMINAMATH_CALUDE_product_equals_57_over_168_l1232_123267

def product : ℚ :=
  (2^3 - 1) / (2^3 + 1) *
  (3^3 - 1) / (3^3 + 1) *
  (4^3 - 1) / (4^3 + 1) *
  (5^3 - 1) / (5^3 + 1) *
  (6^3 - 1) / (6^3 + 1) *
  (7^3 - 1) / (7^3 + 1)

theorem product_equals_57_over_168 : product = 57 / 168 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_57_over_168_l1232_123267


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l1232_123239

/-- The volume of a slice of pizza -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) 
  (h1 : thickness = 1/4)
  (h2 : diameter = 16)
  (h3 : num_slices = 8) :
  (π * (diameter/2)^2 * thickness) / num_slices = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_volume_l1232_123239


namespace NUMINAMATH_CALUDE_joanne_earnings_theorem_l1232_123256

/-- Calculates Joanne's total weekly earnings based on her work schedule and pay rates -/
def joanne_weekly_earnings (main_job_hours_per_day : ℕ) (main_job_rate : ℚ) 
  (part_time_hours_per_day : ℕ) (part_time_rate : ℚ) (days_per_week : ℕ) : ℚ :=
  (main_job_hours_per_day * main_job_rate + part_time_hours_per_day * part_time_rate) * days_per_week

/-- Theorem stating that Joanne's weekly earnings are $775.00 -/
theorem joanne_earnings_theorem : 
  joanne_weekly_earnings 8 16 2 (27/2) 5 = 775 := by
  sorry

end NUMINAMATH_CALUDE_joanne_earnings_theorem_l1232_123256


namespace NUMINAMATH_CALUDE_exponent_simplification_l1232_123237

theorem exponent_simplification :
  (1 : ℝ) / ((5 : ℝ)^2)^4 * (5 : ℝ)^15 = (5 : ℝ)^7 := by sorry

end NUMINAMATH_CALUDE_exponent_simplification_l1232_123237


namespace NUMINAMATH_CALUDE_max_ab_min_a2_b2_l1232_123249

theorem max_ab_min_a2_b2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + 2 * b = 2) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 2 → x * y ≤ a * b) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 2 → a^2 + b^2 ≤ x^2 + y^2) ∧
  a * b = 1/2 ∧ a^2 + b^2 = 4/5 := by
sorry

end NUMINAMATH_CALUDE_max_ab_min_a2_b2_l1232_123249


namespace NUMINAMATH_CALUDE_probability_not_monday_l1232_123223

theorem probability_not_monday (p_monday : ℚ) (h : p_monday = 1/7) : 
  1 - p_monday = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_monday_l1232_123223


namespace NUMINAMATH_CALUDE_tangent_line_at_pi_over_2_l1232_123238

noncomputable def f (x : ℝ) := Real.sin x - 2 * Real.cos x

theorem tangent_line_at_pi_over_2 :
  let Q : ℝ × ℝ := (π / 2, 1)
  let m : ℝ := Real.cos (π / 2) + 2 * Real.sin (π / 2)
  let tangent_line (x : ℝ) := m * (x - Q.1) + Q.2
  ∀ x, tangent_line x = 2 * x - π + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_pi_over_2_l1232_123238


namespace NUMINAMATH_CALUDE_division_remainder_3005_98_l1232_123236

theorem division_remainder_3005_98 : ∃ q : ℤ, 3005 = 98 * q + 65 ∧ 0 ≤ 65 ∧ 65 < 98 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_3005_98_l1232_123236


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_315_l1232_123285

/-- The sum of the digits in the binary representation of 315 is 6 -/
theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_315_l1232_123285


namespace NUMINAMATH_CALUDE_triangle_trig_identity_l1232_123299

/-- Given a triangle ABC with sides AB = 6, AC = 5, and BC = 4,
    prove that (cos((A - B)/2) / sin(C/2)) - (sin((A - B)/2) / cos(C/2)) = 5/3 -/
theorem triangle_trig_identity (A B C : ℝ) (hABC : A + B + C = π) 
  (hAB : Real.cos A * 6 = Real.cos B * 5 + Real.cos C * 4)
  (hBC : Real.cos B * 4 = Real.cos C * 5 + Real.cos A * 6)
  (hAC : Real.cos C * 5 = Real.cos A * 6 + Real.cos B * 4) :
  (Real.cos ((A - B)/2) / Real.sin (C/2)) - (Real.sin ((A - B)/2) / Real.cos (C/2)) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_identity_l1232_123299


namespace NUMINAMATH_CALUDE_complement_M_in_U_l1232_123253

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define the set M
def M : Set ℝ := {x | 2 * x - x^2 > 0}

-- Statement to prove
theorem complement_M_in_U : 
  {x : ℝ | x ∈ U ∧ x ∉ M} = {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l1232_123253


namespace NUMINAMATH_CALUDE_min_area_quadrilateral_on_parabola_l1232_123259

/-- Parabola type -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 4*x

/-- Point on a parabola -/
structure PointOnParabola (par : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : par.eq x y

/-- Chord of a parabola -/
structure Chord (par : Parabola) where
  p1 : PointOnParabola par
  p2 : PointOnParabola par

/-- Theorem: Minimum area of quadrilateral ABCD -/
theorem min_area_quadrilateral_on_parabola (par : Parabola)
  (A B C D : PointOnParabola par) 
  (chord_AC chord_BD : Chord par)
  (perp : chord_AC.p1.x = A.x ∧ chord_AC.p1.y = A.y ∧ 
          chord_AC.p2.x = C.x ∧ chord_AC.p2.y = C.y ∧
          chord_BD.p1.x = B.x ∧ chord_BD.p1.y = B.y ∧
          chord_BD.p2.x = D.x ∧ chord_BD.p2.y = D.y ∧
          (chord_AC.p2.y - chord_AC.p1.y) * (chord_BD.p2.y - chord_BD.p1.y) = 
          -(chord_AC.p2.x - chord_AC.p1.x) * (chord_BD.p2.x - chord_BD.p1.x))
  (through_focus : ∃ t : ℝ, 
    chord_AC.p1.x + t * (chord_AC.p2.x - chord_AC.p1.x) = par.p / 2 ∧
    chord_AC.p1.y + t * (chord_AC.p2.y - chord_AC.p1.y) = 0 ∧
    chord_BD.p1.x + t * (chord_BD.p2.x - chord_BD.p1.x) = par.p / 2 ∧
    chord_BD.p1.y + t * (chord_BD.p2.y - chord_BD.p1.y) = 0) :
  ∃ area : ℝ, area ≥ 32 ∧ 
    area = (1/2) * Real.sqrt ((A.x - C.x)^2 + (A.y - C.y)^2) * 
                    Real.sqrt ((B.x - D.x)^2 + (B.y - D.y)^2) := by
  sorry

end NUMINAMATH_CALUDE_min_area_quadrilateral_on_parabola_l1232_123259


namespace NUMINAMATH_CALUDE_random_co_captains_probability_l1232_123295

def team_sizes : List Nat := [4, 5, 6, 7]
def co_captains_per_team : Nat := 3

def prob_both_co_captains (n : Nat) : Rat :=
  (co_captains_per_team.choose 2) / (n.choose 2)

theorem random_co_captains_probability :
  (1 / team_sizes.length : Rat) *
  (team_sizes.map prob_both_co_captains).sum = 2/7 := by sorry

end NUMINAMATH_CALUDE_random_co_captains_probability_l1232_123295


namespace NUMINAMATH_CALUDE_smallest_a_is_eight_l1232_123284

-- Define the polynomial function
def f (a x : ℤ) : ℤ := x^4 + a^2 + 2*a*x

-- Define what it means for a number to be composite
def is_composite (n : ℤ) : Prop := ∃ m k : ℤ, m > 1 ∧ k > 1 ∧ n = m * k

-- State the theorem
theorem smallest_a_is_eight :
  (∀ x : ℤ, is_composite (f 8 x)) ∧
  (∀ a : ℤ, 0 < a → a < 8 → ∃ x : ℤ, ¬ is_composite (f a x)) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_eight_l1232_123284


namespace NUMINAMATH_CALUDE_two_stamps_theorem_l1232_123216

/-- The cost of a single stamp in dollars -/
def single_stamp_cost : ℚ := 34/100

/-- The cost of three stamps in dollars -/
def three_stamps_cost : ℚ := 102/100

/-- The cost of two stamps in dollars -/
def two_stamps_cost : ℚ := 68/100

theorem two_stamps_theorem :
  (single_stamp_cost * 2 = two_stamps_cost) ∧
  (single_stamp_cost * 3 = three_stamps_cost) := by
  sorry

end NUMINAMATH_CALUDE_two_stamps_theorem_l1232_123216


namespace NUMINAMATH_CALUDE_circle_radius_zero_l1232_123278

/-- The radius of a circle given by the equation 4x^2 + 8x + 4y^2 - 16y + 20 = 0 is 0 -/
theorem circle_radius_zero (x y : ℝ) : 
  4*x^2 + 8*x + 4*y^2 - 16*y + 20 = 0 → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l1232_123278


namespace NUMINAMATH_CALUDE_cost_of_stationery_l1232_123235

/-- Given the cost of different combinations of erasers, pens, and markers,
    prove that 3 erasers, 4 pens, and 6 markers cost 520 rubles. -/
theorem cost_of_stationery (E P M : ℕ) : 
  (E + 3 * P + 2 * M = 240) →
  (2 * E + 5 * P + 4 * M = 440) →
  (3 * E + 4 * P + 6 * M = 520) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_stationery_l1232_123235


namespace NUMINAMATH_CALUDE_square_difference_plus_constant_l1232_123246

theorem square_difference_plus_constant : (262^2 - 258^2) + 150 = 2230 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_constant_l1232_123246


namespace NUMINAMATH_CALUDE_probability_theorem_l1232_123289

def total_cups : ℕ := 8
def white_cups : ℕ := 3
def red_cups : ℕ := 3
def black_cups : ℕ := 2
def selected_cups : ℕ := 5

def probability_specific_sequence : ℚ :=
  (white_cups * (white_cups - 1) * red_cups * (red_cups - 1) * black_cups) /
  (total_cups * (total_cups - 1) * (total_cups - 2) * (total_cups - 3) * (total_cups - 4))

def number_of_arrangements : ℕ := Nat.factorial selected_cups / 
  (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

theorem probability_theorem :
  probability_specific_sequence * number_of_arrangements = 9 / 28 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1232_123289


namespace NUMINAMATH_CALUDE_remaining_family_member_age_l1232_123234

/-- Represents the ages of family members -/
structure FamilyAges where
  total : ℕ
  father : ℕ
  mother : ℕ
  brother : ℕ
  sister : ℕ
  remaining : ℕ

/-- Theorem stating the age of the remaining family member -/
theorem remaining_family_member_age 
  (family : FamilyAges)
  (h_total : family.total = 200)
  (h_father : family.father = 60)
  (h_mother : family.mother = family.father - 2)
  (h_brother : family.brother = family.father / 2)
  (h_sister : family.sister = 40)
  (h_sum : family.total = family.father + family.mother + family.brother + family.sister + family.remaining) :
  family.remaining = 12 :=
by sorry

end NUMINAMATH_CALUDE_remaining_family_member_age_l1232_123234


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l1232_123247

/-- A polygon in a plane -/
structure Polygon where
  sides : ℕ

/-- A regular polygon -/
structure RegularPolygon extends Polygon

/-- An irregular polygon -/
structure IrregularPolygon extends Polygon

/-- Two polygons that overlap but share no complete side -/
structure OverlappingPolygons where
  P₁ : RegularPolygon
  P₂ : IrregularPolygon
  overlap : Bool
  no_shared_side : Bool

/-- The maximum number of intersection points between two polygons -/
def max_intersections (op : OverlappingPolygons) : ℕ :=
  op.P₁.sides * op.P₂.sides

/-- Theorem: The maximum number of intersections between a regular polygon P₁
    and an irregular polygon P₂, where they overlap but share no complete side,
    is the product of their number of sides -/
theorem max_intersections_theorem (op : OverlappingPolygons)
    (h : op.P₁.sides ≤ op.P₂.sides) :
    max_intersections op = op.P₁.sides * op.P₂.sides :=
  sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l1232_123247


namespace NUMINAMATH_CALUDE_journey_distance_l1232_123255

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 40 ∧ speed1 = 20 ∧ speed2 = 30 →
  ∃ (distance : ℝ), 
    distance / speed1 / 2 + distance / speed2 / 2 = total_time ∧ 
    distance = 960 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l1232_123255


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l1232_123212

theorem minimum_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l1232_123212


namespace NUMINAMATH_CALUDE_roberts_trip_l1232_123274

/-- Proves that given the conditions of Robert's trip, the return trip takes 2.5 hours -/
theorem roberts_trip (distance : ℝ) (outbound_time : ℝ) (saved_time : ℝ) (avg_speed : ℝ) :
  distance = 180 →
  outbound_time = 3 →
  saved_time = 0.5 →
  avg_speed = 80 →
  (2 * distance) / (outbound_time + (outbound_time + saved_time - 2 * saved_time) - 2 * saved_time) = avg_speed →
  outbound_time + saved_time - 2 * saved_time = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_roberts_trip_l1232_123274


namespace NUMINAMATH_CALUDE_shoes_alteration_problem_l1232_123230

theorem shoes_alteration_problem (cost_per_shoe : ℕ) (total_cost : ℕ) (num_pairs : ℕ) :
  cost_per_shoe = 29 →
  total_cost = 986 →
  num_pairs = total_cost / (2 * cost_per_shoe) →
  num_pairs = 17 :=
by sorry

end NUMINAMATH_CALUDE_shoes_alteration_problem_l1232_123230


namespace NUMINAMATH_CALUDE_inequality_proof_l1232_123283

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + b * c) / a + (1 + c * a) / b + (1 + a * b) / c > 
  Real.sqrt (a^2 + 2) + Real.sqrt (b^2 + 2) + Real.sqrt (c^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1232_123283


namespace NUMINAMATH_CALUDE_range_of_a_l1232_123252

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1232_123252


namespace NUMINAMATH_CALUDE_find_a_l1232_123248

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else -x

-- State the theorem
theorem find_a : ∃ (a : ℝ), f (1/3) = (1/3) * f a ∧ a = 1/27 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1232_123248


namespace NUMINAMATH_CALUDE_consecutive_numbers_problem_l1232_123240

theorem consecutive_numbers_problem (x y z w : ℚ) : 
  x > y ∧ y > z ∧  -- x, y, z are consecutive and in descending order
  w > x ∧  -- w is greater than x
  w = (5/3) * x ∧  -- ratio of x to w is 3:5
  w^2 = x * z ∧  -- w^2 = xz
  2*x + 3*y + 3*z = 5*y + 11 ∧  -- given equation
  x - y = y - z  -- x, y, z are equally spaced
  → z = 3 := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_problem_l1232_123240


namespace NUMINAMATH_CALUDE_triangle_to_hexagon_proportionality_l1232_123280

-- Define the original triangle
structure Triangle where
  x : Real
  y : Real
  z : Real
  angle_sum : x + y + z = 180

-- Define the resulting hexagon
structure Hexagon where
  a : Real -- Length of vector a
  b : Real -- Length of vector b
  c : Real -- Length of vector c
  u : Real -- Length of vector u
  v : Real -- Length of vector v
  w : Real -- Length of vector w
  angle1 : Real -- (x-1)°
  angle2 : Real -- 181°
  angle3 : Real
  angle4 : Real
  angle5 : Real
  angle6 : Real
  angle_sum : angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 720
  non_convex : angle2 > 180

-- Define the transformation from triangle to hexagon
def transform (t : Triangle) (h : Hexagon) : Prop :=
  h.angle1 = t.x - 1 ∧ h.angle2 = 181

-- Theorem to prove
theorem triangle_to_hexagon_proportionality (t : Triangle) (h : Hexagon) 
  (trans : transform t h) : 
  ∃ (k : Real), k > 0 ∧ 
    h.a / t.x = h.b / t.y ∧ 
    h.b / t.y = h.c / t.z ∧ 
    h.c / t.z = k :=
  sorry

end NUMINAMATH_CALUDE_triangle_to_hexagon_proportionality_l1232_123280


namespace NUMINAMATH_CALUDE_common_difference_is_three_l1232_123221

/-- Arithmetic sequence with 10 terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n < 9 → a (n + 1) = a n + d

/-- Sum of odd terms is 15 -/
def sum_odd_terms (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 + a 5 + a 7 + a 9 = 15

/-- Sum of even terms is 30 -/
def sum_even_terms (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 + a 8 + a 10 = 30

theorem common_difference_is_three (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : sum_odd_terms a) 
  (h3 : sum_even_terms a) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, n < 9 → a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_three_l1232_123221


namespace NUMINAMATH_CALUDE_helium_pressure_change_l1232_123272

-- Define the variables and constants
variable (v₁ v₂ p₁ p₂ : ℝ)

-- State the given conditions
def initial_volume : ℝ := 3.6
def initial_pressure : ℝ := 8
def final_volume : ℝ := 4.5

-- Define the inverse proportionality relationship
def inverse_proportional (v₁ v₂ p₁ p₂ : ℝ) : Prop :=
  v₁ * p₁ = v₂ * p₂

-- State the theorem
theorem helium_pressure_change :
  v₁ = initial_volume →
  p₁ = initial_pressure →
  v₂ = final_volume →
  inverse_proportional v₁ v₂ p₁ p₂ →
  p₂ = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_helium_pressure_change_l1232_123272


namespace NUMINAMATH_CALUDE_complex_expression_magnitude_l1232_123297

theorem complex_expression_magnitude : 
  Complex.abs ((18 - 5 * Complex.I) * (14 + 6 * Complex.I) - (3 - 12 * Complex.I) * (4 + 9 * Complex.I)) = Real.sqrt 146365 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_magnitude_l1232_123297


namespace NUMINAMATH_CALUDE_candy_distribution_l1232_123266

theorem candy_distribution (total : ℝ) (total_pos : total > 0) : 
  let initial_shares := [4/10, 3/10, 2/10, 1/10]
  let first_round := initial_shares.map (· * total)
  let remaining_after_first := total - first_round.sum
  let second_round := initial_shares.map (· * remaining_after_first)
  remaining_after_first - second_round.sum = 0 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l1232_123266


namespace NUMINAMATH_CALUDE_nigella_sold_three_houses_l1232_123261

/-- Represents a house with its cost -/
structure House where
  cost : ℝ

/-- Represents a realtor's earnings -/
structure RealtorEarnings where
  baseSalary : ℝ
  commissionRate : ℝ
  totalEarnings : ℝ

def calculateCommission (house : House) (commissionRate : ℝ) : ℝ :=
  house.cost * commissionRate

def nigellaEarnings : RealtorEarnings := {
  baseSalary := 3000
  commissionRate := 0.02
  totalEarnings := 8000
}

def houseA : House := { cost := 60000 }
def houseB : House := { cost := 3 * houseA.cost }
def houseC : House := { cost := 2 * houseA.cost - 110000 }

theorem nigella_sold_three_houses :
  let commission := calculateCommission houseA nigellaEarnings.commissionRate +
                    calculateCommission houseB nigellaEarnings.commissionRate +
                    calculateCommission houseC nigellaEarnings.commissionRate
  nigellaEarnings.totalEarnings = nigellaEarnings.baseSalary + commission ∧
  (houseA.cost > 0 ∧ houseB.cost > 0 ∧ houseC.cost > 0) →
  3 = 3 := by
  sorry

#check nigella_sold_three_houses

end NUMINAMATH_CALUDE_nigella_sold_three_houses_l1232_123261


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l1232_123243

theorem square_perimeters_sum (x y : ℝ) (h1 : x^2 + y^2 = 65) (h2 : x^2 - y^2 = 33) :
  4*x + 4*y = 44 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l1232_123243


namespace NUMINAMATH_CALUDE_add_negative_numbers_l1232_123257

theorem add_negative_numbers : -10 + (-12) = -22 := by
  sorry

end NUMINAMATH_CALUDE_add_negative_numbers_l1232_123257


namespace NUMINAMATH_CALUDE_fraction_product_squared_l1232_123206

theorem fraction_product_squared :
  (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_squared_l1232_123206


namespace NUMINAMATH_CALUDE_tower_height_difference_l1232_123204

theorem tower_height_difference (grace_height clyde_height : ℕ) :
  grace_height = 40 ∧ grace_height = 8 * clyde_height →
  grace_height - clyde_height = 35 := by
  sorry

end NUMINAMATH_CALUDE_tower_height_difference_l1232_123204


namespace NUMINAMATH_CALUDE_cost_of_pens_l1232_123233

/-- Given a box of 150 pens costing $45, prove that the cost of 3600 pens is $1080 -/
theorem cost_of_pens (box_size : ℕ) (box_cost : ℚ) (total_pens : ℕ) :
  box_size = 150 →
  box_cost = 45 →
  total_pens = 3600 →
  (total_pens : ℚ) / box_size * box_cost = 1080 :=
by
  sorry


end NUMINAMATH_CALUDE_cost_of_pens_l1232_123233


namespace NUMINAMATH_CALUDE_peter_large_glasses_bought_l1232_123219

def small_glass_cost : ℕ := 3
def large_glass_cost : ℕ := 5
def initial_amount : ℕ := 50
def small_glasses_bought : ℕ := 8
def change : ℕ := 1

def large_glasses_bought : ℕ := (initial_amount - change - small_glass_cost * small_glasses_bought) / large_glass_cost

theorem peter_large_glasses_bought :
  large_glasses_bought = 5 :=
by sorry

end NUMINAMATH_CALUDE_peter_large_glasses_bought_l1232_123219


namespace NUMINAMATH_CALUDE_unique_diametric_circle_l1232_123264

/-- An equilateral triangle in a 2D plane -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_equilateral : ∀ (i j : Fin 3), i ≠ j → 
    Real.sqrt ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 
    Real.sqrt ((vertices 0).1 - (vertices 1).1)^2 + ((vertices 0).2 - (vertices 1).2)^2

/-- A circle defined by two points as its diameter -/
structure DiametricCircle (T : EquilateralTriangle) where
  endpoint1 : Fin 3
  endpoint2 : Fin 3
  is_diameter : endpoint1 ≠ endpoint2

/-- The theorem stating that there's only one unique diametric circle for an equilateral triangle -/
theorem unique_diametric_circle (T : EquilateralTriangle) : 
  ∃! (c : DiametricCircle T), True := by sorry

end NUMINAMATH_CALUDE_unique_diametric_circle_l1232_123264


namespace NUMINAMATH_CALUDE_apple_tv_cost_l1232_123231

theorem apple_tv_cost (iphone_count : ℕ) (iphone_cost : ℝ)
                      (ipad_count : ℕ) (ipad_cost : ℝ)
                      (apple_tv_count : ℕ)
                      (total_avg_cost : ℝ) :
  iphone_count = 100 →
  iphone_cost = 1000 →
  ipad_count = 20 →
  ipad_cost = 900 →
  apple_tv_count = 80 →
  total_avg_cost = 670 →
  (iphone_count * iphone_cost + ipad_count * ipad_cost + apple_tv_count * (iphone_count * iphone_cost + ipad_count * ipad_cost + apple_tv_count * 200) / (iphone_count + ipad_count + apple_tv_count)) / (iphone_count + ipad_count + apple_tv_count) = total_avg_cost →
  (iphone_count * iphone_cost + ipad_count * ipad_cost + apple_tv_count * 200) / (iphone_count + ipad_count + apple_tv_count) = total_avg_cost :=
by sorry

#check apple_tv_cost

end NUMINAMATH_CALUDE_apple_tv_cost_l1232_123231


namespace NUMINAMATH_CALUDE_negation_existence_real_gt_one_l1232_123227

theorem negation_existence_real_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_real_gt_one_l1232_123227


namespace NUMINAMATH_CALUDE_range_of_a_l1232_123244

-- Define an odd function that is monotonically increasing on [0, +∞)
def is_odd_and_increasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f x < f y)

-- Theorem statement
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_odd_incr : is_odd_and_increasing f) 
  (h_ineq : f (2 - a^2) + f a > 0) : 
  -1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1232_123244


namespace NUMINAMATH_CALUDE_quadratic_root_l1232_123254

/-- A quadratic polynomial with coefficients a, b, and c. -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Predicate to check if a quadratic polynomial has exactly one root. -/
def has_one_root (a b c : ℝ) : Prop := b^2 = 4 * a * c

theorem quadratic_root (a b c : ℝ) (ha : a ≠ 0) :
  has_one_root a b c →
  has_one_root (-a) (b - 30*a) (17*a - 7*b + c) →
  ∃! x : ℝ, quadratic a b c x = 0 ∧ x = -11 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_l1232_123254


namespace NUMINAMATH_CALUDE_jogger_speed_l1232_123220

/-- The speed of the jogger in km/hr given the following conditions:
  1. The jogger is 200 m ahead of the train engine
  2. The train is 210 m long
  3. The train is running at 45 km/hr
  4. The train and jogger are moving in the same direction
  5. The train passes the jogger in 41 seconds
-/
theorem jogger_speed : ℝ := by
  -- Define the given conditions
  let initial_distance : ℝ := 200 -- meters
  let train_length : ℝ := 210 -- meters
  let train_speed : ℝ := 45 -- km/hr
  let passing_time : ℝ := 41 -- seconds

  -- Define the jogger's speed as a variable
  let jogger_speed : ℝ := 9 -- km/hr

  sorry -- Proof omitted

#check jogger_speed

end NUMINAMATH_CALUDE_jogger_speed_l1232_123220


namespace NUMINAMATH_CALUDE_function_value_at_inverse_point_l1232_123201

noncomputable def log_log_2_10 : ℝ := Real.log (Real.log 10 / Real.log 2)

theorem function_value_at_inverse_point 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h : ∀ x, f x = a * x^3 + b * Real.sin x + 4) 
  (h1 : f log_log_2_10 = 5) : 
  f (- log_log_2_10) = 3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_inverse_point_l1232_123201


namespace NUMINAMATH_CALUDE_tangent_line_is_perpendicular_and_tangent_l1232_123250

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

-- Define the given curve
def given_curve (x y : ℝ) : Prop := y = x^3 + 3 * x^2 - 5

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x + y + 6 = 0

-- Theorem statement
theorem tangent_line_is_perpendicular_and_tangent :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    given_curve x₀ y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The tangent line is perpendicular to the given line
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      given_line x₁ y₁ ∧ given_line x₂ y₂ ∧ x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * ((y₀ + 6) / (-3) - y₀) / (((y₀ + 6) / (-3)) - x₀) = -1) ∧
    -- The tangent line is indeed tangent to the curve
    (∀ (x : ℝ), x ≠ x₀ → ∃ (y : ℝ), given_curve x y ∧ ¬tangent_line x y) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_perpendicular_and_tangent_l1232_123250


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1232_123262

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  perimeter : ℝ
  is_isosceles : side ≥ 0 ∧ base ≥ 0 ∧ perimeter = 2 * side + base

-- Theorem statement
theorem isosceles_triangle_base_length 
  (t : IsoscelesTriangle) 
  (h_perimeter : t.perimeter = 26) 
  (h_side : t.side = 11 ∨ t.base = 11) : 
  t.base = 11 ∨ t.base = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1232_123262


namespace NUMINAMATH_CALUDE_eduardo_classes_l1232_123263

theorem eduardo_classes (x : ℕ) : 
  x + 2 * x = 9 → x = 3 := by sorry

end NUMINAMATH_CALUDE_eduardo_classes_l1232_123263


namespace NUMINAMATH_CALUDE_negative_three_star_negative_two_nested_star_op_l1232_123282

-- Define the custom operation
def star_op (a b : ℤ) : ℤ := a^2 - b + a * b

-- Theorem statements
theorem negative_three_star_negative_two : star_op (-3) (-2) = 17 := by sorry

theorem nested_star_op : star_op (-2) (star_op (-3) (-2)) = -47 := by sorry

end NUMINAMATH_CALUDE_negative_three_star_negative_two_nested_star_op_l1232_123282


namespace NUMINAMATH_CALUDE_odd_function_zero_value_l1232_123265

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_zero_value (f : ℝ → ℝ) (h : OddFunction f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_value_l1232_123265


namespace NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l1232_123290

theorem cos_pi_half_plus_alpha (α : Real) 
  (h : (Real.sin (π + α) * Real.cos (-α + 4*π)) / Real.cos α = 1/2) : 
  Real.cos (π/2 + α) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_half_plus_alpha_l1232_123290


namespace NUMINAMATH_CALUDE_floor_neg_seven_fourths_l1232_123275

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_seven_fourths_l1232_123275


namespace NUMINAMATH_CALUDE_limit_of_sequence_l1232_123287

def a (n : ℕ) : ℚ := (5 * n + 1) / (10 * n - 3)

theorem limit_of_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 1/2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l1232_123287


namespace NUMINAMATH_CALUDE_number_times_x_minus_3y_l1232_123203

/-- Given that 2x - y = 4 and kx - 3y = 12, prove that k = 6 -/
theorem number_times_x_minus_3y (x y k : ℝ) 
  (h1 : 2 * x - y = 4) 
  (h2 : k * x - 3 * y = 12) : 
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_times_x_minus_3y_l1232_123203


namespace NUMINAMATH_CALUDE_orthocenter_on_line_l1232_123205

/-
  Define the necessary geometric objects and properties
-/

-- Define a Point type
structure Point := (x y : ℝ)

-- Define a Line type
structure Line := (a b c : ℝ)

-- Define a Circle type
structure Circle := (center : Point) (radius : ℝ)

-- Define a Triangle type
structure Triangle := (A B C : Point)

-- Function to check if a triangle is acute-angled
def is_acute_triangle (t : Triangle) : Prop := sorry

-- Function to get the circumcenter of a triangle
def circumcenter (t : Triangle) : Point := sorry

-- Function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Function to get the orthocenter of a triangle
def orthocenter (t : Triangle) : Point := sorry

-- Function to check if a circle passes through a point
def circle_passes_through (c : Circle) (p : Point) : Prop := sorry

-- Function to get the intersection points of a circle and a line segment
def circle_line_intersection (c : Circle) (l : Line) : List Point := sorry

-- Main theorem
theorem orthocenter_on_line 
  (A B C : Point) 
  (O : Point) 
  (c : Circle) 
  (P Q : Point) :
  is_acute_triangle (Triangle.mk A B C) →
  O = circumcenter (Triangle.mk A B C) →
  circle_passes_through c B →
  circle_passes_through c O →
  P ∈ circle_line_intersection c (Line.mk 0 1 0) → -- Assuming BC is on y-axis
  Q ∈ circle_line_intersection c (Line.mk 1 0 0) → -- Assuming BA is on x-axis
  point_on_line (orthocenter (Triangle.mk P O Q)) (Line.mk 1 1 0) -- Assuming AC is y = x
  := by sorry

end NUMINAMATH_CALUDE_orthocenter_on_line_l1232_123205


namespace NUMINAMATH_CALUDE_oak_grove_total_books_l1232_123281

def public_library_books : ℕ := 1986
def school_library_books : ℕ := 5106

theorem oak_grove_total_books :
  public_library_books + school_library_books = 7092 :=
by sorry

end NUMINAMATH_CALUDE_oak_grove_total_books_l1232_123281


namespace NUMINAMATH_CALUDE_exactlyTwoVisitCount_l1232_123225

/-- Represents a visitor with a visiting frequency -/
structure Visitor where
  frequency : ℕ

/-- Calculates the number of days when exactly two out of three visitors visit -/
def exactlyTwoVisit (v1 v2 v3 : Visitor) (days : ℕ) : ℕ :=
  sorry

theorem exactlyTwoVisitCount :
  let alice : Visitor := ⟨2⟩
  let beatrix : Visitor := ⟨5⟩
  let claire : Visitor := ⟨7⟩
  exactlyTwoVisit alice beatrix claire 365 = 55 := by sorry

end NUMINAMATH_CALUDE_exactlyTwoVisitCount_l1232_123225


namespace NUMINAMATH_CALUDE_code_problem_l1232_123213

theorem code_problem (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 →
  B > A →
  A < C →
  11 * B + 11 * A + 11 * C = 242 →
  ((A = 5 ∧ B = 8 ∧ C = 9) ∨ (A = 5 ∧ B = 9 ∧ C = 8)) :=
by sorry

end NUMINAMATH_CALUDE_code_problem_l1232_123213


namespace NUMINAMATH_CALUDE_smaller_factor_of_5610_l1232_123210

theorem smaller_factor_of_5610 (a b : Nat) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 5610 → 
  min a b = 34 := by
sorry

end NUMINAMATH_CALUDE_smaller_factor_of_5610_l1232_123210


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l1232_123258

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = -1/2 + (5/4) * Complex.I →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = -1/2 - (5/4) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l1232_123258


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_24_and_864_l1232_123298

theorem smallest_n_divisible_by_24_and_864 :
  ∃ n : ℕ+, (∀ m : ℕ+, m < n → (¬(24 ∣ m^2) ∨ ¬(864 ∣ m^3))) ∧ 
  (24 ∣ n^2) ∧ (864 ∣ n^3) ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_24_and_864_l1232_123298


namespace NUMINAMATH_CALUDE_average_of_tenths_and_thousandths_l1232_123277

theorem average_of_tenths_and_thousandths :
  let a : ℚ := 4/10  -- 4 tenths
  let b : ℚ := 5/1000  -- 5 thousandths
  (a + b) / 2 = 2025/10000 := by
sorry

end NUMINAMATH_CALUDE_average_of_tenths_and_thousandths_l1232_123277


namespace NUMINAMATH_CALUDE_g_of_3_equals_5_l1232_123270

-- Define the function g
def g (y : ℝ) : ℝ := 2 * (y - 2) + 3

-- State the theorem
theorem g_of_3_equals_5 : g 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_5_l1232_123270


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_property_l1232_123222

theorem quadratic_equation_solution_property (k : ℝ) : 
  (∃ a b : ℝ, 
    (3 * a^2 + 6 * a + k = 0) ∧ 
    (3 * b^2 + 6 * b + k = 0) ∧ 
    (abs (a - b) = 2 * (a^2 + b^2))) ↔ 
  (k = 3 ∨ k = 45/16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_property_l1232_123222
