import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_set_l23_2356

theorem inequality_solution_set (x : ℝ) :
  ((x + 1) / (3 - x) < 0) ↔ (x < -1 ∨ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l23_2356


namespace NUMINAMATH_CALUDE_inradius_plus_circumradius_le_height_l23_2319

/-- An acute-angled triangle is a triangle where all angles are less than 90 degrees. -/
structure AcuteTriangle where
  /-- The greatest height of the triangle -/
  height : ℝ
  /-- The inradius of the triangle -/
  inradius : ℝ
  /-- The circumradius of the triangle -/
  circumradius : ℝ
  /-- All angles are less than 90 degrees -/
  acute : height > 0 ∧ inradius > 0 ∧ circumradius > 0

/-- For any acute-angled triangle, the sum of its inradius and circumradius
    is less than or equal to its greatest height. -/
theorem inradius_plus_circumradius_le_height (t : AcuteTriangle) :
  t.inradius + t.circumradius ≤ t.height := by
  sorry

end NUMINAMATH_CALUDE_inradius_plus_circumradius_le_height_l23_2319


namespace NUMINAMATH_CALUDE_base5_multiplication_l23_2383

/-- Converts a base 5 number to base 10 --/
def baseConvert5To10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 5 --/
def baseConvert10To5 (n : ℕ) : ℕ := sorry

/-- Multiplies two base 5 numbers --/
def multiplyBase5 (a b : ℕ) : ℕ :=
  baseConvert10To5 (baseConvert5To10 a * baseConvert5To10 b)

theorem base5_multiplication :
  multiplyBase5 132 22 = 4004 := by sorry

end NUMINAMATH_CALUDE_base5_multiplication_l23_2383


namespace NUMINAMATH_CALUDE_spent_sixty_four_l23_2372

/-- The total amount spent by Victor and his friend on trick decks -/
def total_spent (deck_price : ℕ) (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * (victor_decks + friend_decks)

/-- Theorem: Victor and his friend spent $64 on trick decks -/
theorem spent_sixty_four :
  total_spent 8 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_spent_sixty_four_l23_2372


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l23_2360

theorem complex_number_quadrant : ∃ (z : ℂ), 
  z / (1 - z) = Complex.I * 2 ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 :=
sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l23_2360


namespace NUMINAMATH_CALUDE_isosceles_triangle_80_vertex_angle_l23_2333

/-- An isosceles triangle with one angle of 80 degrees -/
structure IsoscelesTriangle80 where
  /-- The measure of the vertex angle in degrees -/
  vertex_angle : ℝ
  /-- The measure of one of the base angles in degrees -/
  base_angle : ℝ
  /-- The triangle is isosceles -/
  isosceles : base_angle = 180 - vertex_angle - base_angle
  /-- One angle is 80 degrees -/
  has_80_degree : vertex_angle = 80 ∨ base_angle = 80
  /-- The sum of angles is 180 degrees -/
  angle_sum : vertex_angle + 2 * base_angle = 180

/-- The vertex angle in an isosceles triangle with one 80-degree angle is either 80 or 20 degrees -/
theorem isosceles_triangle_80_vertex_angle (t : IsoscelesTriangle80) :
  t.vertex_angle = 80 ∨ t.vertex_angle = 20 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_80_vertex_angle_l23_2333


namespace NUMINAMATH_CALUDE_range_of_a_l23_2349

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → -(5-2*a)^x > -(5-2*a)^y

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l23_2349


namespace NUMINAMATH_CALUDE_circle_area_ratio_l23_2361

/-- A square with side length 2 -/
structure Square :=
  (side_length : ℝ)
  (is_two : side_length = 2)

/-- A circle outside the square -/
structure Circle :=
  (radius : ℝ)
  (center : ℝ × ℝ)

/-- The configuration of the two circles and the square -/
structure Configuration :=
  (square : Square)
  (circle1 : Circle)
  (circle2 : Circle)
  (tangent_to_PQ : circle1.center.2 = square.side_length / 2)
  (tangent_to_RS : circle2.center.2 = -square.side_length / 2)
  (tangent_to_QR_extension : circle1.center.1 + circle1.radius = circle2.center.1 - circle2.radius)

/-- The theorem stating the ratio of the areas -/
theorem circle_area_ratio (config : Configuration) : 
  (π * config.circle2.radius^2) / (π * config.circle1.radius^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l23_2361


namespace NUMINAMATH_CALUDE_weaving_factory_profit_maximization_l23_2342

/-- Represents the profit maximization problem in a weaving factory --/
theorem weaving_factory_profit_maximization 
  (total_workers : ℕ) 
  (fabric_per_worker : ℕ) 
  (clothing_per_worker : ℕ) 
  (fabric_per_clothing : ℚ) 
  (fabric_profit : ℚ) 
  (clothing_profit : ℕ) 
  (h_total : total_workers = 150)
  (h_fabric : fabric_per_worker = 30)
  (h_clothing : clothing_per_worker = 4)
  (h_fabric_clothing : fabric_per_clothing = 3/2)
  (h_fabric_profit : fabric_profit = 2)
  (h_clothing_profit : clothing_profit = 25) :
  ∃ (x : ℕ), 
    x ≤ total_workers ∧ 
    (clothing_profit * clothing_per_worker * x : ℚ) + 
    (fabric_profit * (fabric_per_worker * (total_workers - x) - 
    fabric_per_clothing * clothing_per_worker * x)) = 11800 ∧
    x = 100 := by
  sorry

end NUMINAMATH_CALUDE_weaving_factory_profit_maximization_l23_2342


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_l23_2305

def tickets_bought : ℕ := 13
def tickets_left : ℕ := 4
def ticket_cost : ℕ := 9

theorem ferris_wheel_cost : (tickets_bought - tickets_left) * ticket_cost = 81 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_l23_2305


namespace NUMINAMATH_CALUDE_determinant_problem_l23_2364

theorem determinant_problem (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = -3 →
  Matrix.det !![p, 5*p + 4*q; r, 5*r + 4*s] = -12 := by
  sorry

end NUMINAMATH_CALUDE_determinant_problem_l23_2364


namespace NUMINAMATH_CALUDE_simplify_radical_product_l23_2390

theorem simplify_radical_product (q : ℝ) (h : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 14 * q * Real.sqrt (39 * q) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l23_2390


namespace NUMINAMATH_CALUDE_budget_supplies_percent_l23_2309

theorem budget_supplies_percent (salaries research_dev utilities equipment transportation : ℝ)
  (h1 : salaries = 60)
  (h2 : research_dev = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : transportation = 72 * 100 / 360)
  (h6 : salaries + research_dev + utilities + equipment + transportation < 100) :
  100 - (salaries + research_dev + utilities + equipment + transportation) = 2 := by
  sorry

end NUMINAMATH_CALUDE_budget_supplies_percent_l23_2309


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l23_2376

def sequence_sum (n : ℕ) : ℚ :=
  if n = 0 then 3 / 3
  else if n = 1 then 4 + 1/3 * (sequence_sum 0)
  else (2003 - n + 1) + 1/3 * (sequence_sum (n-1))

theorem sequence_sum_formula : 
  sequence_sum 2000 = 3004.5 - 1 / (2 * 3^1999) := by sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l23_2376


namespace NUMINAMATH_CALUDE_sqrt_product_equation_l23_2318

theorem sqrt_product_equation (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (8 * x) * Real.sqrt (10 * x) * Real.sqrt (3 * x) * Real.sqrt (15 * x) = 15) : 
  x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equation_l23_2318


namespace NUMINAMATH_CALUDE_associated_equation_k_range_l23_2340

/-- Definition of an associated equation -/
def is_associated_equation (eq_sol : ℝ) (ineq_set : Set ℝ) : Prop :=
  eq_sol ∈ ineq_set

/-- The system of inequalities -/
def inequality_system (x : ℝ) : Prop :=
  (x - 3) / 2 ≥ x ∧ (2 * x + 5) / 2 > x / 2

/-- The solution set of the system of inequalities -/
def solution_set : Set ℝ :=
  {x | inequality_system x}

/-- The equation 2x - k = 6 -/
def equation (k : ℝ) (x : ℝ) : Prop :=
  2 * x - k = 6

theorem associated_equation_k_range :
  ∀ k : ℝ, (∃ x : ℝ, equation k x ∧ is_associated_equation x solution_set) →
    -16 < k ∧ k ≤ -12 :=
by sorry

end NUMINAMATH_CALUDE_associated_equation_k_range_l23_2340


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l23_2354

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * (x - 1) = 2 - 5 * (x + 2)
def equation2 (x : ℝ) : Prop := (5 * x + 1) / 2 - (6 * x + 2) / 4 = 1

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = -6/7 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l23_2354


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l23_2343

def income : ℕ := 10000
def savings : ℕ := 2000

def expenditure : ℕ := income - savings

def ratio_simplify (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem income_expenditure_ratio :
  ratio_simplify income expenditure = (5, 4) := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l23_2343


namespace NUMINAMATH_CALUDE_expand_product_l23_2380

theorem expand_product (x : ℝ) : (x + 3) * (x - 6) * (x + 2) = x^3 - x^2 - 24*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l23_2380


namespace NUMINAMATH_CALUDE_fourth_day_income_l23_2374

def cab_driver_income (day1 day2 day3 day4 day5 : ℝ) : Prop :=
  day1 = 200 ∧ day2 = 150 ∧ day3 = 750 ∧ day5 = 500 ∧
  (day1 + day2 + day3 + day4 + day5) / 5 = 400

theorem fourth_day_income (day1 day2 day3 day4 day5 : ℝ) :
  cab_driver_income day1 day2 day3 day4 day5 → day4 = 400 := by
  sorry

end NUMINAMATH_CALUDE_fourth_day_income_l23_2374


namespace NUMINAMATH_CALUDE_sequence_periodicity_l23_2398

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n

theorem sequence_periodicity (a : ℕ → ℝ) (h : sequence_property a) :
  ∃ m₀ : ℕ, ∀ m ≥ m₀, a (m + 9) = a m :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l23_2398


namespace NUMINAMATH_CALUDE_sqrt_two_sin_twenty_equals_cos_minus_sin_theta_l23_2399

theorem sqrt_two_sin_twenty_equals_cos_minus_sin_theta (θ : Real) :
  θ > 0 ∧ θ < Real.pi / 2 →
  Real.sqrt 2 * Real.sin (20 * Real.pi / 180) = Real.cos θ - Real.sin θ →
  θ = 25 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_sqrt_two_sin_twenty_equals_cos_minus_sin_theta_l23_2399


namespace NUMINAMATH_CALUDE_area_of_three_arc_region_l23_2338

/-- The area of a region bounded by three identical circular arcs -/
theorem area_of_three_arc_region :
  let r : ℝ := 5 -- radius of each arc
  let θ : ℝ := Real.pi / 2 -- central angle in radians (90 degrees)
  let segment_area : ℝ := r^2 * (θ - Real.sin θ) / 2 -- area of one circular segment
  let total_area : ℝ := 3 * segment_area -- area of the entire region
  total_area = (75 * Real.pi - 150) / 4 :=
by sorry

end NUMINAMATH_CALUDE_area_of_three_arc_region_l23_2338


namespace NUMINAMATH_CALUDE_tea_mixture_theorem_l23_2337

/-- Calculates the price of a tea mixture per kg -/
def tea_mixture_price (price1 price2 price3 : ℚ) (ratio1 ratio2 ratio3 : ℚ) : ℚ :=
  let total_cost := price1 * ratio1 + price2 * ratio2 + price3 * ratio3
  let total_quantity := ratio1 + ratio2 + ratio3
  total_cost / total_quantity

/-- Theorem stating the price of the specific tea mixture -/
theorem tea_mixture_theorem : 
  tea_mixture_price 126 135 177.5 1 1 2 = 154 := by
  sorry

#eval tea_mixture_price 126 135 177.5 1 1 2

end NUMINAMATH_CALUDE_tea_mixture_theorem_l23_2337


namespace NUMINAMATH_CALUDE_add_negative_numbers_add_positive_negative_add_negative_positive_inverse_subtract_larger_from_smaller_subtract_negative_add_negative_positive_real_abs_value_add_negative_multiply_negative_mixed_multiply_two_negatives_l23_2317

-- 1. (-51) + (-37) = -88
theorem add_negative_numbers : (-51) + (-37) = -88 := by sorry

-- 2. (+2) + (-11) = -9
theorem add_positive_negative : (2 : Int) + (-11) = -9 := by sorry

-- 3. (-12) + (+12) = 0
theorem add_negative_positive_inverse : (-12) + (12 : Int) = 0 := by sorry

-- 4. 8 - 14 = -6
theorem subtract_larger_from_smaller : (8 : Int) - 14 = -6 := by sorry

-- 5. 15 - (-8) = 23
theorem subtract_negative : (15 : Int) - (-8) = 23 := by sorry

-- 6. (-3.4) + 4.3 = 0.9
theorem add_negative_positive_real : (-3.4) + 4.3 = 0.9 := by sorry

-- 7. |-2.25| + (-0.5) = 1.75
theorem abs_value_add_negative : |(-2.25 : ℝ)| + (-0.5) = 1.75 := by sorry

-- 8. -4 * 1.5 = -6
theorem multiply_negative_mixed : (-4 : ℝ) * 1.5 = -6 := by sorry

-- 9. -3 * (-6) = 18
theorem multiply_two_negatives : (-3 : Int) * (-6) = 18 := by sorry

end NUMINAMATH_CALUDE_add_negative_numbers_add_positive_negative_add_negative_positive_inverse_subtract_larger_from_smaller_subtract_negative_add_negative_positive_real_abs_value_add_negative_multiply_negative_mixed_multiply_two_negatives_l23_2317


namespace NUMINAMATH_CALUDE_expand_expression_l23_2362

theorem expand_expression (x y : ℝ) : (x + 7) * (3 * y + 8) = 3 * x * y + 8 * x + 21 * y + 56 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l23_2362


namespace NUMINAMATH_CALUDE_milk_bottles_count_l23_2345

theorem milk_bottles_count (bread : ℕ) (total : ℕ) (h1 : bread = 37) (h2 : total = 52) :
  total - bread = 15 := by
  sorry

end NUMINAMATH_CALUDE_milk_bottles_count_l23_2345


namespace NUMINAMATH_CALUDE_union_M_N_l23_2369

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N with respect to U
def complement_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := U \ complement_N

-- Theorem statement
theorem union_M_N : M ∪ N = {x : ℝ | -3 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_l23_2369


namespace NUMINAMATH_CALUDE_circle_C_radius_range_l23_2373

-- Define the triangle vertices
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (3, 2)

-- Define the circumcircle H
def H : ℝ × ℝ := (0, 3)

-- Define the line BH
def lineBH (x y : ℝ) : Prop := 3 * x + y - 3 = 0

-- Define a point P on line segment BH
def P (m : ℝ) : ℝ × ℝ := (m, 3 - 3 * m)

-- Define the circle with center C
def circleC (r : ℝ) (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = r^2

-- Define the theorem
theorem circle_C_radius_range :
  ∀ (r : ℝ), 
  (∀ (m : ℝ), 0 ≤ m ∧ m ≤ 1 →
    ∃ (x y : ℝ), 
      circleC r x y ∧
      circleC r ((x + m) / 2) ((y + (3 - 3 * m)) / 2) ∧
      x ≠ m ∧ y ≠ (3 - 3 * m)) →
  (∀ (m : ℝ), 0 ≤ m ∧ m ≤ 1 →
    (m - 3)^2 + (1 - 3 * m)^2 > r^2) →
  Real.sqrt 10 / 3 ≤ r ∧ r < 4 * Real.sqrt 10 / 5 :=
sorry


end NUMINAMATH_CALUDE_circle_C_radius_range_l23_2373


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l23_2329

def repeating_decimal : ℚ := 7 + 2/10 + 34/99/100

theorem repeating_decimal_as_fraction : 
  repeating_decimal = 36357 / 4950 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l23_2329


namespace NUMINAMATH_CALUDE_valid_systematic_sample_l23_2382

/-- Represents a systematic sample -/
structure SystematicSample where
  population : ℕ
  sampleSize : ℕ
  startPoint : ℕ
  interval : ℕ

/-- Checks if a given list is a valid systematic sample -/
def isValidSystematicSample (sample : List ℕ) (s : SystematicSample) : Prop :=
  sample.length = s.sampleSize ∧
  sample.all (·≤ s.population) ∧
  sample.all (·> 0) ∧
  ∀ i, i < sample.length - 1 → sample[i + 1]! - sample[i]! = s.interval

theorem valid_systematic_sample :
  let sample := [3, 13, 23, 33, 43]
  let s : SystematicSample := {
    population := 50,
    sampleSize := 5,
    startPoint := 3,
    interval := 10
  }
  isValidSystematicSample sample s := by
  sorry

end NUMINAMATH_CALUDE_valid_systematic_sample_l23_2382


namespace NUMINAMATH_CALUDE_largest_A_k_l23_2367

-- Define A_k
def A (k : ℕ) : ℝ := (Nat.choose 1000 k) * (0.2 ^ k)

-- State the theorem
theorem largest_A_k : ∃ (k : ℕ), k = 166 ∧ ∀ (j : ℕ), j ≠ k → j ≤ 1000 → A k ≥ A j := by
  sorry

end NUMINAMATH_CALUDE_largest_A_k_l23_2367


namespace NUMINAMATH_CALUDE_stratified_sample_grade12_l23_2363

/-- Represents the number of students in a grade -/
structure GradePopulation where
  total : ℕ
  sampled : ℕ

/-- Represents the school population -/
structure SchoolPopulation where
  grade11 : GradePopulation
  grade12 : GradePopulation

/-- Checks if the sampling is stratified (same ratio across grades) -/
def isStratifiedSample (school : SchoolPopulation) : Prop :=
  school.grade11.sampled * school.grade12.total = school.grade12.sampled * school.grade11.total

/-- The main theorem -/
theorem stratified_sample_grade12 (school : SchoolPopulation) 
    (h1 : school.grade11.total = 500)
    (h2 : school.grade12.total = 450)
    (h3 : school.grade11.sampled = 20)
    (h4 : isStratifiedSample school) :
  school.grade12.sampled = 18 := by
  sorry

#check stratified_sample_grade12

end NUMINAMATH_CALUDE_stratified_sample_grade12_l23_2363


namespace NUMINAMATH_CALUDE_candidate_b_votes_l23_2397

/-- Proves that candidate B received 4560 valid votes given the election conditions -/
theorem candidate_b_votes (total_eligible : Nat) (abstention_rate : Real) (invalid_vote_rate : Real) 
  (c_vote_percentage : Real) (a_vote_reduction : Real) 
  (h1 : total_eligible = 12000)
  (h2 : abstention_rate = 0.1)
  (h3 : invalid_vote_rate = 0.2)
  (h4 : c_vote_percentage = 0.05)
  (h5 : a_vote_reduction = 0.2) : 
  ∃ (b_votes : Nat), b_votes = 4560 := by
  sorry

end NUMINAMATH_CALUDE_candidate_b_votes_l23_2397


namespace NUMINAMATH_CALUDE_sequence_end_point_sequence_end_point_proof_l23_2332

theorem sequence_end_point : ℕ → Prop :=
  fun n =>
    (∃ k : ℕ, 9 * k ≥ 10 ∧ 9 * (k + 11109) = n) →
    n = 99999

-- The proof is omitted
theorem sequence_end_point_proof : sequence_end_point 99999 := by
  sorry

end NUMINAMATH_CALUDE_sequence_end_point_sequence_end_point_proof_l23_2332


namespace NUMINAMATH_CALUDE_eight_four_two_power_l23_2303

theorem eight_four_two_power : 8^8 * 4^4 / 2^28 = 16 := by sorry

end NUMINAMATH_CALUDE_eight_four_two_power_l23_2303


namespace NUMINAMATH_CALUDE_paths_equal_choose_l23_2370

/-- The number of paths on a 3x3 grid from top-left to bottom-right -/
def num_paths : ℕ := sorry

/-- The number of ways to choose 3 items from a set of 6 items -/
def choose_3_from_6 : ℕ := Nat.choose 6 3

/-- Theorem stating that the number of paths is equal to choosing 3 from 6 -/
theorem paths_equal_choose :
  num_paths = choose_3_from_6 := by sorry

end NUMINAMATH_CALUDE_paths_equal_choose_l23_2370


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l23_2351

theorem complete_square_with_integer (y : ℝ) :
  ∃ (k : ℤ) (b : ℝ), y^2 + 12*y + 44 = (y + b)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l23_2351


namespace NUMINAMATH_CALUDE_triangle_pairs_lower_bound_l23_2355

/-- Given n points in a plane and l line segments, this theorem proves a lower bound
    for the number of triangle pairs formed. -/
theorem triangle_pairs_lower_bound
  (n : ℕ) (l : ℕ) (h_n : n ≥ 4) (h_l : l ≥ n^2 / 4 + 1)
  (no_three_collinear : sorry) -- Hypothesis for no three points being collinear
  (T : ℕ) (h_T : T = sorry) -- Definition of T as the number of triangle pairs
  : T ≥ (l * (4 * l - n^2) * (4 * l - n^2 - n)) / (2 * n^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_pairs_lower_bound_l23_2355


namespace NUMINAMATH_CALUDE_height_C_ceiling_l23_2335

/-- A right-angled triangle with given heights -/
structure RightTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Heights
  hA : ℝ
  hB : ℝ
  hC : ℝ
  -- Conditions
  right_angle : c^2 = a^2 + b^2
  height_A : hA = 5
  height_B : hB = 15
  -- Area consistency
  area_consistency : a * hA = b * hB

/-- The smallest integer ceiling of the height from C to AB is 5 -/
theorem height_C_ceiling (t : RightTriangle) : ⌈t.hC⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_height_C_ceiling_l23_2335


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l23_2306

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 24 →
  a^2 + b^2 + c^2 = 2500 →
  a^2 + b^2 = c^2 →
  c = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l23_2306


namespace NUMINAMATH_CALUDE_evaluate_expression_l23_2358

theorem evaluate_expression : 
  Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + 1 / 3 = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l23_2358


namespace NUMINAMATH_CALUDE_propositions_p_and_not_q_l23_2357

theorem propositions_p_and_not_q :
  (∃ x₀ : ℝ, Real.log x₀ ≥ x₀ - 1) ∧
  ¬(∀ θ : ℝ, Real.sin θ + Real.cos θ < 1) :=
by sorry

end NUMINAMATH_CALUDE_propositions_p_and_not_q_l23_2357


namespace NUMINAMATH_CALUDE_even_function_m_value_l23_2336

/-- A function f is even if f(x) = f(-x) for all x in its domain --/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The given function f(x) = x^2 + (m+2)x + 3 --/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m+2)*x + 3

theorem even_function_m_value :
  ∀ m : ℝ, IsEven (f m) → m = -2 := by
sorry

end NUMINAMATH_CALUDE_even_function_m_value_l23_2336


namespace NUMINAMATH_CALUDE_gumballs_last_days_l23_2316

def gumballs_per_earring : ℕ := 9
def first_day_earrings : ℕ := 3
def second_day_earrings : ℕ := 2 * first_day_earrings
def third_day_earrings : ℕ := second_day_earrings - 1
def daily_consumption : ℕ := 3

def total_earrings : ℕ := first_day_earrings + second_day_earrings + third_day_earrings
def total_gumballs : ℕ := total_earrings * gumballs_per_earring

theorem gumballs_last_days : total_gumballs / daily_consumption = 42 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_last_days_l23_2316


namespace NUMINAMATH_CALUDE_investment_interest_rate_exists_and_unique_l23_2334

theorem investment_interest_rate_exists_and_unique :
  ∃! r : ℝ, 
    r > 0 ∧ 
    6000 * (1 + r)^10 = 24000 ∧ 
    6000 * (1 + r)^15 = 48000 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_rate_exists_and_unique_l23_2334


namespace NUMINAMATH_CALUDE_polygon_intersection_theorem_l23_2328

-- Define a convex polygon
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define what it means for a point to be inside a circle
def point_inside_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define what it means for a polygon to be inside a circle
def polygon_inside_circle {n : ℕ} (p : ConvexPolygon n) (c : Circle) : Prop :=
  ∀ i : Fin n, point_inside_circle (p.vertices i) c

-- Define what it means for two polygons to intersect
def polygons_intersect {n m : ℕ} (p1 : ConvexPolygon n) (p2 : ConvexPolygon m) : Prop :=
  sorry

-- The main theorem
theorem polygon_intersection_theorem {n m : ℕ}
  (p1 : ConvexPolygon n) (p2 : ConvexPolygon m) (c1 c2 : Circle)
  (h1 : polygon_inside_circle p1 c1)
  (h2 : polygon_inside_circle p2 c2)
  (h3 : polygons_intersect p1 p2) :
  (∃ i : Fin n, point_inside_circle (p1.vertices i) c2) ∨
  (∃ j : Fin m, point_inside_circle (p2.vertices j) c1) :=
sorry

end NUMINAMATH_CALUDE_polygon_intersection_theorem_l23_2328


namespace NUMINAMATH_CALUDE_speed_in_still_water_l23_2301

/-- Given a man's upstream and downstream speeds, calculate his speed in still water -/
theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 37) 
  (h2 : downstream_speed = 53) : 
  (upstream_speed + downstream_speed) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l23_2301


namespace NUMINAMATH_CALUDE_peanut_difference_l23_2341

theorem peanut_difference (jose_peanuts kenya_peanuts : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_peanuts = 133)
  (h3 : kenya_peanuts > jose_peanuts) :
  kenya_peanuts - jose_peanuts = 48 := by
  sorry

end NUMINAMATH_CALUDE_peanut_difference_l23_2341


namespace NUMINAMATH_CALUDE_train_speed_l23_2365

/-- Proves that the speed of a train is 90 km/hr, given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 225) (h2 : time = 9) :
  (length / 1000) / (time / 3600) = 90 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l23_2365


namespace NUMINAMATH_CALUDE_tangent_line_and_lower_bound_l23_2377

noncomputable def f (x : ℝ) := Real.exp x - 3 * x^2 + 4 * x

theorem tangent_line_and_lower_bound :
  (∃ (b : ℝ), ∀ x, (Real.exp 1 - 2) * x + b = f 1 + (Real.exp 1 - 6 + 4) * (x - 1)) ∧
  (∀ x ≥ 1, f x > 3) ∧
  (∃ x₀ ≥ 1, f x₀ < 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_lower_bound_l23_2377


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l23_2387

/-- Given a hyperbola with equation x²/4 - y²/9 = -1, its asymptotes are y = ±(3/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 4 - y^2 / 9 = -1 →
  ∃ (k : ℝ), k = 3/2 ∧ (y = k*x ∨ y = -k*x) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l23_2387


namespace NUMINAMATH_CALUDE_average_score_proof_l23_2315

theorem average_score_proof (total_students : Nat) (abc_students : Nat) (de_students : Nat)
  (total_average : ℚ) (abc_average : ℚ) :
  total_students = 5 →
  abc_students = 3 →
  de_students = 2 →
  total_average = 80 →
  abc_average = 78 →
  (total_students * total_average - abc_students * abc_average) / de_students = 83 := by
  sorry

end NUMINAMATH_CALUDE_average_score_proof_l23_2315


namespace NUMINAMATH_CALUDE_binary_equals_octal_l23_2322

-- Define the binary number
def binary_num : List Bool := [true, true, false, true, false, true]

-- Define the octal number
def octal_num : Nat := 65

-- Function to convert binary to decimal
def binary_to_decimal (bin : List Bool) : Nat :=
  bin.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

-- Function to convert decimal to octal
def decimal_to_octal (n : Nat) : Nat :=
  if n < 8 then n
  else 10 * (decimal_to_octal (n / 8)) + (n % 8)

-- Theorem stating that the binary number is equal to the octal number when converted
theorem binary_equals_octal :
  decimal_to_octal (binary_to_decimal binary_num) = octal_num := by
  sorry


end NUMINAMATH_CALUDE_binary_equals_octal_l23_2322


namespace NUMINAMATH_CALUDE_three_digit_sum_theorem_l23_2393

/-- Given three distinct single-digit numbers, returns the largest three-digit number that can be formed using these digits. -/
def largest_three_digit (a b c : Nat) : Nat := sorry

/-- Given three distinct single-digit numbers, returns the second largest three-digit number that can be formed using these digits. -/
def second_largest_three_digit (a b c : Nat) : Nat := sorry

/-- Given three distinct single-digit numbers, returns the smallest three-digit number that can be formed using these digits. -/
def smallest_three_digit (a b c : Nat) : Nat := sorry

theorem three_digit_sum_theorem :
  let a := 2
  let b := 5
  let c := 8
  largest_three_digit a b c + smallest_three_digit a b c + second_largest_three_digit a b c = 1935 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_sum_theorem_l23_2393


namespace NUMINAMATH_CALUDE_thief_speed_l23_2379

/-- Proves that the thief's speed is 8 km/hr given the problem conditions -/
theorem thief_speed (initial_distance : ℝ) (policeman_speed : ℝ) (thief_distance : ℝ)
  (h1 : initial_distance = 100) -- Initial distance in meters
  (h2 : policeman_speed = 10) -- Policeman's speed in km/hr
  (h3 : thief_distance = 400) -- Distance thief runs before being overtaken in meters
  : ∃ (thief_speed : ℝ), thief_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_thief_speed_l23_2379


namespace NUMINAMATH_CALUDE_magazine_sale_gain_l23_2346

/-- Calculates the total gain from selling magazines -/
def total_gain (cost_price selling_price : ℝ) (num_magazines : ℕ) : ℝ :=
  (selling_price - cost_price) * num_magazines

/-- Proves that the total gain from selling 10 magazines at $3.50 each, 
    bought at $3 each, is $5 -/
theorem magazine_sale_gain : 
  total_gain 3 3.5 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_magazine_sale_gain_l23_2346


namespace NUMINAMATH_CALUDE_min_distance_parabola_point_l23_2312

/-- The minimum distance from a point on the parabola to Q plus its y-coordinate -/
theorem min_distance_parabola_point (x y : ℝ) (h : x^2 = -4*y) :
  ∃ (min : ℝ), min = 2 ∧ 
  ∀ (x' y' : ℝ), x'^2 = -4*y' → 
    abs y + Real.sqrt ((x' + 2*Real.sqrt 2)^2 + y'^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_parabola_point_l23_2312


namespace NUMINAMATH_CALUDE_better_performance_against_teamB_l23_2330

/-- Represents the statistics for a team --/
structure TeamStats :=
  (points : List Nat)
  (rebounds : List Nat)
  (turnovers : List Nat)

/-- Calculate the average of a list of numbers --/
def average (l : List Nat) : Rat :=
  (l.sum : Rat) / l.length

/-- Calculate the comprehensive score for a team --/
def comprehensiveScore (stats : TeamStats) : Rat :=
  average stats.points + 1.2 * average stats.rebounds - average stats.turnovers

/-- Xiao Bin's statistics against Team A --/
def teamA : TeamStats :=
  { points := [21, 29, 24, 26],
    rebounds := [10, 10, 14, 10],
    turnovers := [2, 2, 3, 5] }

/-- Xiao Bin's statistics against Team B --/
def teamB : TeamStats :=
  { points := [25, 31, 16, 22],
    rebounds := [17, 15, 12, 8],
    turnovers := [2, 0, 4, 2] }

/-- Theorem: Xiao Bin's comprehensive score against Team B is higher than against Team A --/
theorem better_performance_against_teamB :
  comprehensiveScore teamB > comprehensiveScore teamA :=
by
  sorry


end NUMINAMATH_CALUDE_better_performance_against_teamB_l23_2330


namespace NUMINAMATH_CALUDE_horner_method_proof_l23_2325

def horner_polynomial (x : ℝ) : ℝ := 
  ((((2 * x - 5) * x - 4) * x + 3) * x - 6) * x + 7

theorem horner_method_proof : horner_polynomial 5 = 2677 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_proof_l23_2325


namespace NUMINAMATH_CALUDE_strawberry_picking_problem_l23_2344

/-- Strawberry picking problem -/
theorem strawberry_picking_problem 
  (brother_baskets : ℕ) 
  (strawberries_per_basket : ℕ) 
  (kimberly_multiplier : ℕ) 
  (equal_share : ℕ) 
  (family_members : ℕ) 
  (h1 : brother_baskets = 3)
  (h2 : strawberries_per_basket = 15)
  (h3 : kimberly_multiplier = 8)
  (h4 : equal_share = 168)
  (h5 : family_members = 4) :
  kimberly_multiplier * (brother_baskets * strawberries_per_basket) - 
  (family_members * equal_share - 
   kimberly_multiplier * (brother_baskets * strawberries_per_basket) - 
   (brother_baskets * strawberries_per_basket)) = 93 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_picking_problem_l23_2344


namespace NUMINAMATH_CALUDE_coronavirus_recoveries_day2_l23_2331

/-- Proves that the number of recoveries on day 2 is 50, given the conditions of the Coronavirus case problem. -/
theorem coronavirus_recoveries_day2 
  (initial_cases : ℕ) 
  (day2_increase : ℕ) 
  (day3_new_cases : ℕ) 
  (day3_recoveries : ℕ) 
  (total_cases_day3 : ℕ) 
  (h1 : initial_cases = 2000)
  (h2 : day2_increase = 500)
  (h3 : day3_new_cases = 1500)
  (h4 : day3_recoveries = 200)
  (h5 : total_cases_day3 = 3750) :
  ∃ (day2_recoveries : ℕ), 
    initial_cases + day2_increase - day2_recoveries + day3_new_cases - day3_recoveries = total_cases_day3 ∧ 
    day2_recoveries = 50 := by
  sorry

end NUMINAMATH_CALUDE_coronavirus_recoveries_day2_l23_2331


namespace NUMINAMATH_CALUDE_solution_l23_2308

def problem (a b c d : ℚ) : Prop :=
  a + b + c + d = 406 ∧
  a = (1/2) * b ∧
  b = (1/2) * c ∧
  d = (1/3) * c

theorem solution :
  ∃ a b c d : ℚ,
    problem a b c d ∧
    a = 48.72 ∧
    b = 97.44 ∧
    c = 194.88 ∧
    d = 64.96 :=
by sorry

end NUMINAMATH_CALUDE_solution_l23_2308


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l23_2381

theorem consecutive_integers_cube_sum : 
  ∀ x : ℕ, x > 0 → 
  (x - 1) * x * (x + 1) = 12 * (3 * x) →
  (x - 1)^3 + x^3 + (x + 1)^3 = 3 * (37 : ℝ).sqrt^3 + 6 * (37 : ℝ).sqrt :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l23_2381


namespace NUMINAMATH_CALUDE_sticker_distribution_jeremy_sticker_problem_l23_2327

/-- The number of ways to distribute n identical objects among k distinct groups -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute n identical objects among k distinct groups,
    with each group receiving at least one object -/
def distributeAtLeastOne (n k : ℕ) : ℕ := distribute (n - k) k

theorem sticker_distribution (n k : ℕ) (hn : n ≥ k) (hk : k > 0) :
  distributeAtLeastOne n k = distribute (n - k) k :=
by sorry

theorem jeremy_sticker_problem :
  distributeAtLeastOne 10 3 = 36 :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_jeremy_sticker_problem_l23_2327


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l23_2326

theorem davids_chemistry_marks
  (english : ℕ) (mathematics : ℕ) (physics : ℕ) (biology : ℕ) (average : ℕ)
  (h_english : english = 61)
  (h_mathematics : mathematics = 65)
  (h_physics : physics = 82)
  (h_biology : biology = 85)
  (h_average : average = 72)
  : ∃ chemistry : ℕ,
    chemistry = 67 ∧
    (english + mathematics + physics + chemistry + biology) / 5 = average :=
by sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l23_2326


namespace NUMINAMATH_CALUDE_total_legs_in_farm_l23_2314

theorem total_legs_in_farm (total_animals : Nat) (num_ducks : Nat) (duck_legs : Nat) (dog_legs : Nat) :
  total_animals = 8 →
  num_ducks = 4 →
  duck_legs = 2 →
  dog_legs = 4 →
  (num_ducks * duck_legs + (total_animals - num_ducks) * dog_legs) = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_farm_l23_2314


namespace NUMINAMATH_CALUDE_sum_of_squares_l23_2350

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 21) (h2 : x * y = 43) : x^2 + y^2 = 355 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l23_2350


namespace NUMINAMATH_CALUDE_cubic_root_inequality_l23_2353

theorem cubic_root_inequality (R : ℚ) (h : R ≥ 0) : 
  let a : ℤ := 1
  let b : ℤ := 1
  let c : ℤ := 2
  let d : ℤ := 1
  let e : ℤ := 1
  let f : ℤ := 1
  |((a * R^2 + b * R + c) / (d * R^2 + e * R + f) : ℚ) - (2 : ℚ)^(1/3)| < |R - (2 : ℚ)^(1/3)| :=
by
  sorry

#check cubic_root_inequality

end NUMINAMATH_CALUDE_cubic_root_inequality_l23_2353


namespace NUMINAMATH_CALUDE_expression_equals_one_l23_2385

theorem expression_equals_one : 
  (2001 * 2021 + 100) * (1991 * 2031 + 400) / (2011^4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l23_2385


namespace NUMINAMATH_CALUDE_work_completion_time_l23_2394

theorem work_completion_time (a_time b_time joint_time b_remaining_time : ℝ) 
  (ha : a_time = 45)
  (hjoint : joint_time = 9)
  (hb_remaining : b_remaining_time = 23)
  (h_work_rate : (joint_time * (1 / a_time + 1 / b_time)) + 
                 (b_remaining_time * (1 / b_time)) = 1) :
  b_time = 40 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l23_2394


namespace NUMINAMATH_CALUDE_round_trip_distance_l23_2311

/-- Calculates the total distance of a round trip given the times for each leg and the average speed -/
theorem round_trip_distance (t1 t2 : ℚ) (avg_speed : ℚ) (h1 : t1 = 15/60) (h2 : t2 = 25/60) (h3 : avg_speed = 3) :
  (t1 + t2) * avg_speed = 2 := by
  sorry

#check round_trip_distance

end NUMINAMATH_CALUDE_round_trip_distance_l23_2311


namespace NUMINAMATH_CALUDE_max_abs_sum_of_quadratic_coeffs_l23_2359

/-- Given a quadratic polynomial ax^2 + bx + c where |ax^2 + bx + c| ≤ 1 for all x in [-1,1],
    the maximum value of |a| + |b| + |c| is 3. -/
theorem max_abs_sum_of_quadratic_coeffs (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) →
  |a| + |b| + |c| ≤ 3 ∧ ∃ a' b' c' : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a' * x^2 + b' * x + c'| ≤ 1) ∧ |a'| + |b'| + |c'| = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_of_quadratic_coeffs_l23_2359


namespace NUMINAMATH_CALUDE_sum_of_angles_in_three_triangles_l23_2388

theorem sum_of_angles_in_three_triangles :
  ∀ (angle1 angle2 angle3 angle4 angle5 angle6 angle7 angle8 angle9 : ℝ),
    angle1 > 0 → angle2 > 0 → angle3 > 0 → angle4 > 0 → angle5 > 0 →
    angle6 > 0 → angle7 > 0 → angle8 > 0 → angle9 > 0 →
    angle1 + angle2 + angle3 = 180 →
    angle4 + angle5 + angle6 = 180 →
    angle7 + angle8 + angle9 = 180 →
    angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 + angle8 + angle9 = 540 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_three_triangles_l23_2388


namespace NUMINAMATH_CALUDE_curve_equivalence_l23_2368

-- Define the set of points satisfying the original equation
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + p.2 - 1) * Real.sqrt (p.1^2 + p.2^2 - 4) = 0}

-- Define the set of points on the line outside or on the circle
def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ p.1^2 + p.2^2 ≥ 4}

-- Define the set of points on the circle
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Theorem stating the equivalence of the sets
theorem curve_equivalence : S = L ∪ C := by sorry

end NUMINAMATH_CALUDE_curve_equivalence_l23_2368


namespace NUMINAMATH_CALUDE_extra_apples_l23_2391

theorem extra_apples (red_apples green_apples students : ℕ) 
  (h1 : red_apples = 25)
  (h2 : green_apples = 17)
  (h3 : students = 10) :
  red_apples + green_apples - students = 32 := by
  sorry

end NUMINAMATH_CALUDE_extra_apples_l23_2391


namespace NUMINAMATH_CALUDE_calculation_proof_l23_2307

theorem calculation_proof :
  (125 * 76 * 4 * 8 * 25 = 7600000) ∧
  ((6742 + 6743 + 6738 + 6739 + 6741 + 6743) / 6 = 6741) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l23_2307


namespace NUMINAMATH_CALUDE_conference_handshakes_l23_2395

/-- Represents a conference with two groups of people -/
structure Conference :=
  (total : ℕ)
  (group1 : ℕ)
  (group2 : ℕ)
  (h_total : total = group1 + group2)

/-- Calculates the number of handshakes in the conference -/
def handshakes (conf : Conference) : ℕ :=
  conf.group2 * (conf.group1 + conf.group2 - 1)

theorem conference_handshakes :
  ∃ (conf : Conference),
    conf.total = 40 ∧
    conf.group1 = 25 ∧
    conf.group2 = 15 ∧
    handshakes conf = 480 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l23_2395


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l23_2366

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l23_2366


namespace NUMINAMATH_CALUDE_worst_player_is_son_or_sister_l23_2371

-- Define the family members
inductive FamilyMember
  | Woman
  | Brother
  | Son
  | Daughter
  | Sister

-- Define the chess skill level
def ChessSkill := Nat

structure Family where
  members : List FamilyMember
  skills : FamilyMember → ChessSkill
  worst_player : FamilyMember
  best_player : FamilyMember
  twin : FamilyMember

def is_opposite_sex (a b : FamilyMember) : Prop :=
  (a = FamilyMember.Woman ∨ a = FamilyMember.Daughter ∨ a = FamilyMember.Sister) ∧
  (b = FamilyMember.Brother ∨ b = FamilyMember.Son) ∨
  (b = FamilyMember.Woman ∨ b = FamilyMember.Daughter ∨ b = FamilyMember.Sister) ∧
  (a = FamilyMember.Brother ∨ a = FamilyMember.Son)

def are_siblings (a b : FamilyMember) : Prop :=
  (a = FamilyMember.Brother ∧ b = FamilyMember.Sister) ∨
  (a = FamilyMember.Sister ∧ b = FamilyMember.Brother)

def is_between_woman_and_sister (a : FamilyMember) : Prop :=
  a = FamilyMember.Brother ∨ a = FamilyMember.Son ∨ a = FamilyMember.Daughter

theorem worst_player_is_son_or_sister (f : Family) :
  (f.members = [FamilyMember.Woman, FamilyMember.Brother, FamilyMember.Son, FamilyMember.Daughter, FamilyMember.Sister]) →
  (is_opposite_sex f.twin f.best_player) →
  (f.skills f.worst_player = f.skills f.best_player) →
  (are_siblings f.twin f.worst_player ∨ is_between_woman_and_sister f.twin) →
  (f.worst_player = FamilyMember.Son ∨ f.worst_player = FamilyMember.Sister) :=
by sorry

end NUMINAMATH_CALUDE_worst_player_is_son_or_sister_l23_2371


namespace NUMINAMATH_CALUDE_toy_car_growth_l23_2310

theorem toy_car_growth (initial_count : ℕ) (growth_factor : ℚ) (final_multiplier : ℕ) : 
  initial_count = 50 → growth_factor = 5/2 → final_multiplier = 3 →
  (↑initial_count * growth_factor * ↑final_multiplier : ℚ) = 375 := by
  sorry

end NUMINAMATH_CALUDE_toy_car_growth_l23_2310


namespace NUMINAMATH_CALUDE_tech_students_formula_l23_2348

/-- The number of students in technology elective courses -/
def tech_students (m : ℕ) : ℚ :=
  (1 / 3 : ℚ) * (m : ℚ) + 8

/-- The number of students in subject elective courses -/
def subject_students (m : ℕ) : ℕ := m

/-- The number of students in physical education and arts elective courses -/
def pe_arts_students (m : ℕ) : ℕ := m + 9

theorem tech_students_formula (m : ℕ) :
  tech_students m = (1 / 3 : ℚ) * (pe_arts_students m : ℚ) + 5 :=
by sorry

end NUMINAMATH_CALUDE_tech_students_formula_l23_2348


namespace NUMINAMATH_CALUDE_man_speed_man_speed_proof_l23_2323

/-- Calculates the speed of a man moving opposite to a bullet train -/
theorem man_speed (train_length : ℝ) (train_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_length / passing_time * 3.6
  relative_speed - train_speed

/-- Proves that the man's speed is 4 kmph given the specific conditions -/
theorem man_speed_proof :
  man_speed 120 50 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_proof_l23_2323


namespace NUMINAMATH_CALUDE_fold_length_is_ten_rectangle_fold_length_l23_2304

/-- Represents a folded rectangle with specific properties -/
structure FoldedRectangle where
  short_side : ℝ
  long_side : ℝ
  fold_length : ℝ
  congruent_triangles : Prop

/-- The folded rectangle satisfies the problem conditions -/
def satisfies_conditions (r : FoldedRectangle) : Prop :=
  r.short_side = 12 ∧
  r.long_side = r.short_side * 3/2 ∧
  r.congruent_triangles

/-- The theorem to be proved -/
theorem fold_length_is_ten 
  (r : FoldedRectangle) 
  (h : satisfies_conditions r) : 
  r.fold_length = 10 := by
  sorry

/-- The main theorem restated in terms of the problem -/
theorem rectangle_fold_length :
  ∃ (r : FoldedRectangle), 
    satisfies_conditions r ∧ 
    r.fold_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_fold_length_is_ten_rectangle_fold_length_l23_2304


namespace NUMINAMATH_CALUDE_swimmer_distance_proof_l23_2384

/-- Calculates the distance swam against a current given the swimmer's speed in still water,
    the current's speed, and the time spent swimming. -/
def distance_swam_against_current (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
  (swimmer_speed - current_speed) * time

/-- Proves that a swimmer with a given speed in still water, swimming against a specific current
    for a certain amount of time, covers the expected distance. -/
theorem swimmer_distance_proof (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ)
    (h1 : swimmer_speed = 3)
    (h2 : current_speed = 1.7)
    (h3 : time = 2.3076923076923075)
    : distance_swam_against_current swimmer_speed current_speed time = 3 := by
  sorry

#eval distance_swam_against_current 3 1.7 2.3076923076923075

end NUMINAMATH_CALUDE_swimmer_distance_proof_l23_2384


namespace NUMINAMATH_CALUDE_obese_employee_is_male_prob_l23_2313

-- Define the company's employee structure
structure Company where
  male_ratio : ℚ
  female_ratio : ℚ
  male_obese_ratio : ℚ
  female_obese_ratio : ℚ

-- Define the probability function
def prob_obese_is_male (c : Company) : ℚ :=
  (c.male_ratio * c.male_obese_ratio) / 
  (c.male_ratio * c.male_obese_ratio + c.female_ratio * c.female_obese_ratio)

-- Theorem statement
theorem obese_employee_is_male_prob 
  (c : Company) 
  (h1 : c.male_ratio = 3/5) 
  (h2 : c.female_ratio = 2/5)
  (h3 : c.male_obese_ratio = 1/5)
  (h4 : c.female_obese_ratio = 1/10) :
  prob_obese_is_male c = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_obese_employee_is_male_prob_l23_2313


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l23_2396

theorem sine_cosine_inequality (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ c < 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l23_2396


namespace NUMINAMATH_CALUDE_inequality_product_sum_l23_2389

theorem inequality_product_sum (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_inequality_product_sum_l23_2389


namespace NUMINAMATH_CALUDE_wheel_radius_increase_wheel_radius_increase_approx_011_l23_2321

/-- Calculates the increase in wheel radius given the original and new measurements -/
theorem wheel_radius_increase (original_radius : ℝ) (original_distance : ℝ) 
  (new_odometer_distance : ℝ) (new_actual_distance : ℝ) : ℝ :=
  let original_circumference := 2 * Real.pi * original_radius
  let original_rotations := original_distance * 63360 / original_circumference
  let new_radius := new_actual_distance * 63360 / (2 * Real.pi * original_rotations)
  new_radius - original_radius

/-- The increase in wheel radius is approximately 0.11 inches -/
theorem wheel_radius_increase_approx_011 :
  abs (wheel_radius_increase 12 300 310 315 - 0.11) < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_wheel_radius_increase_wheel_radius_increase_approx_011_l23_2321


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l23_2392

/-- Given an ellipse and a hyperbola with the same a and b parameters,
    prove that if the ellipse has eccentricity 1/2,
    then the hyperbola has eccentricity √7/2 -/
theorem ellipse_hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (∃ c : ℝ, c/a = 1/2)) :
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    (∃ c' : ℝ, c'/a = Real.sqrt 7 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l23_2392


namespace NUMINAMATH_CALUDE_symmetric_line_and_distance_theorem_l23_2324

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x + y + 3 = 0
def l₂ (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the symmetric line l₃
def l₃ (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, -1)

-- Define the line m
def m (x y : ℝ) : Prop := 3 * x + 4 * y + 10 = 0 ∨ x = -2

-- Theorem statement
theorem symmetric_line_and_distance_theorem :
  (∀ x y : ℝ, l₃ x y ↔ l₁ x (-y)) ∧
  (l₂ P.1 P.2 ∧ l₃ P.1 P.2) ∧
  (m P.1 P.2 ∧ 
   ∀ x y : ℝ, m x y → 
     (x * x + y * y = 4 ∨ 
      (3 * x + 4 * y + 10)^2 / (3 * 3 + 4 * 4) = 4)) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_and_distance_theorem_l23_2324


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l23_2347

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 9) :
  let sphere_diameter := outer_cube_edge
  let inscribed_cube_space_diagonal := sphere_diameter
  let inscribed_cube_edge := inscribed_cube_space_diagonal / Real.sqrt 3
  let inscribed_cube_volume := inscribed_cube_edge ^ 3
  inscribed_cube_volume = 81 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l23_2347


namespace NUMINAMATH_CALUDE_louies_previous_goals_correct_l23_2320

/-- Calculates the number of goals Louie scored in previous matches before the last match -/
def louies_previous_goals (
  louies_last_match_goals : ℕ
  ) (
  brother_seasons : ℕ
  ) (
  games_per_season : ℕ
  ) (
  total_goals : ℕ
  ) : ℕ := by
  sorry

theorem louies_previous_goals_correct :
  louies_previous_goals 4 3 50 1244 = 40 := by
  sorry

end NUMINAMATH_CALUDE_louies_previous_goals_correct_l23_2320


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l23_2300

theorem consecutive_integers_square_sum : ∃ (a : ℕ), 
  (a > 0) ∧ 
  ((a - 1) * a * (a + 1) = 8 * (3 * a)) ∧ 
  ((a - 1)^2 + a^2 + (a + 1)^2 = 77) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l23_2300


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l23_2375

theorem cubic_sum_theorem (a b c d : ℕ) 
  (h : (a + b + c + d) * (a^2 + b^2 + c^2 + d^2)^2 = 2023) : 
  a^3 + b^3 + c^3 + d^3 = 43 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l23_2375


namespace NUMINAMATH_CALUDE_rogers_cookie_price_l23_2352

/-- Represents a cookie shape -/
inductive CookieShape
| Trapezoid
| Rectangle

/-- Represents a baker's cookie production -/
structure Baker where
  name : String
  shape : CookieShape
  numCookies : ℕ
  pricePerCookie : ℕ

/-- Calculates the total earnings for a baker -/
def totalEarnings (baker : Baker) : ℕ :=
  baker.numCookies * baker.pricePerCookie

/-- Theorem: Roger's cookie price for equal earnings -/
theorem rogers_cookie_price 
  (art roger : Baker)
  (h1 : art.shape = CookieShape.Trapezoid)
  (h2 : roger.shape = CookieShape.Rectangle)
  (h3 : art.numCookies = 12)
  (h4 : art.pricePerCookie = 60)
  (h5 : totalEarnings art = totalEarnings roger) :
  roger.pricePerCookie = 40 :=
sorry

end NUMINAMATH_CALUDE_rogers_cookie_price_l23_2352


namespace NUMINAMATH_CALUDE_intersection_theorem_union_theorem_complement_union_theorem_l23_2302

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 3 < x ∧ x < 10}

-- State the theorems to be proved
theorem intersection_theorem : A ∩ B = {x | 3 < x ∧ x < 7} := by sorry

theorem union_theorem : A ∪ B = {x | 2 ≤ x ∧ x < 10} := by sorry

theorem complement_union_theorem : (Set.univ \ A) ∪ (Set.univ \ B) = {x | x ≤ 3 ∨ x ≥ 7} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_union_theorem_complement_union_theorem_l23_2302


namespace NUMINAMATH_CALUDE_four_roots_iff_t_in_range_l23_2378

-- Define the function f(x) = |xe^x|
noncomputable def f (x : ℝ) : ℝ := |x * Real.exp x|

-- Define the equation f^2(x) + tf(x) + 2 = 0
def has_four_distinct_roots (t : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (f x₁)^2 + t * f x₁ + 2 = 0 ∧
    (f x₂)^2 + t * f x₂ + 2 = 0 ∧
    (f x₃)^2 + t * f x₃ + 2 = 0 ∧
    (f x₄)^2 + t * f x₄ + 2 = 0

-- The theorem to be proved
theorem four_roots_iff_t_in_range :
  ∀ t : ℝ, has_four_distinct_roots t ↔ t < -(2 * Real.exp 2 + 1) / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_four_roots_iff_t_in_range_l23_2378


namespace NUMINAMATH_CALUDE_valid_fractions_characterization_l23_2339

def is_valid_fraction (a b : ℕ) : Prop :=
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧
  (a % 10 = b / 10) ∧
  (a : ℚ) / b = (a / 10 : ℚ) / (b % 10)

def valid_fractions : Set (ℕ × ℕ) :=
  {(a, b) | is_valid_fraction a b}

theorem valid_fractions_characterization :
  valid_fractions = {(19, 95), (49, 98), (11, 11), (22, 22), (33, 33),
                     (44, 44), (55, 55), (66, 66), (77, 77), (88, 88),
                     (99, 99), (16, 64), (26, 65)} :=
by sorry

end NUMINAMATH_CALUDE_valid_fractions_characterization_l23_2339


namespace NUMINAMATH_CALUDE_root_transformation_l23_2386

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 8 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 8 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 8 = 0) → 
  ((3*r₁)^3 - 9*(3*r₁)^2 + 216 = 0) ∧
  ((3*r₂)^3 - 9*(3*r₂)^2 + 216 = 0) ∧
  ((3*r₃)^3 - 9*(3*r₃)^2 + 216 = 0) := by
sorry

end NUMINAMATH_CALUDE_root_transformation_l23_2386
