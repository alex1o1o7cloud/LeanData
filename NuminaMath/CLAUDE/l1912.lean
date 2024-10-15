import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_triangle_l1912_191230

/-- If in a triangle with sides a and b, and their opposite angles α and β, 
    the equation a / cos(α) = b / cos(β) holds, then a = b. -/
theorem isosceles_triangle (a b α β : Real) : 
  0 < a ∧ 0 < b ∧ 0 < α ∧ α < π ∧ 0 < β ∧ β < π →  -- Ensuring valid triangle
  a / Real.cos α = b / Real.cos β →
  a = b :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1912_191230


namespace NUMINAMATH_CALUDE_initial_wax_amount_l1912_191256

/-- Given the total required amount of wax and the additional amount needed,
    calculate the initial amount of wax available. -/
theorem initial_wax_amount (total_required additional_needed : ℕ) :
  total_required ≥ additional_needed →
  total_required - additional_needed = total_required - additional_needed :=
by sorry

end NUMINAMATH_CALUDE_initial_wax_amount_l1912_191256


namespace NUMINAMATH_CALUDE_semicircle_problem_l1912_191257

theorem semicircle_problem (N : ℕ) (r : ℝ) (h_positive : r > 0) : 
  let A := N * (π * r^2 / 2)
  let B := (π * (N*r)^2 / 2) - A
  A / B = 1 / 9 → N = 10 := by
sorry

end NUMINAMATH_CALUDE_semicircle_problem_l1912_191257


namespace NUMINAMATH_CALUDE_sector_central_angle_l1912_191292

/-- A sector with radius 1 cm and circumference 4 cm has a central angle of 2 radians. -/
theorem sector_central_angle (r : ℝ) (circ : ℝ) (h1 : r = 1) (h2 : circ = 4) :
  let arc_length := circ - 2 * r
  arc_length / r = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1912_191292


namespace NUMINAMATH_CALUDE_parallel_vectors_angle_l1912_191220

theorem parallel_vectors_angle (θ : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_parallel : ∃ (k : Real), k ≠ 0 ∧ (1 - Real.sin θ, 1) = k • (1/2, 1 + Real.sin θ)) :
  θ = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_angle_l1912_191220


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1912_191231

theorem triangle_angle_measure (y : ℝ) : 
  y > 0 ∧ 
  y < 180 ∧ 
  3*y > 0 ∧ 
  3*y < 180 ∧
  y + 3*y + 40 = 180 → 
  y = 35 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1912_191231


namespace NUMINAMATH_CALUDE_problem_solution_l1912_191298

theorem problem_solution (x y : ℝ) (h1 : y = 3) (h2 : x * y + 4 = 19) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1912_191298


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1912_191212

theorem sqrt_equation_solution : 
  {x : ℝ | Real.sqrt (2*x - 4) - Real.sqrt (x + 5) = 1} = {4, 20} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1912_191212


namespace NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l1912_191253

theorem min_value_trig_expression (α β : ℝ) :
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 ≥ 100 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ α β : ℝ, (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l1912_191253


namespace NUMINAMATH_CALUDE_orange_stack_theorem_l1912_191289

/-- Calculates the number of oranges in a trapezoidal layer -/
def trapezoidalLayer (a b h : ℕ) : ℕ := (a + b) * h / 2

/-- Calculates the total number of oranges in the stack -/
def orangeStack (baseA baseB height : ℕ) : ℕ :=
  let rec stackLayers (a b h : ℕ) : ℕ :=
    if h = 0 then 0
    else trapezoidalLayer a b h + stackLayers (a - 1) (b - 1) (h - 1)
  stackLayers baseA baseB height

theorem orange_stack_theorem :
  orangeStack 7 5 6 = 90 := by sorry

end NUMINAMATH_CALUDE_orange_stack_theorem_l1912_191289


namespace NUMINAMATH_CALUDE_candy_box_prices_l1912_191241

theorem candy_box_prices (total_money : ℕ) (metal_boxes : ℕ) (paper_boxes : ℕ) :
  -- A buys 4 fewer boxes than B
  metal_boxes + 4 = paper_boxes →
  -- A has 6 yuan left
  ∃ (metal_price : ℕ), metal_price * metal_boxes + 6 = total_money →
  -- B uses all the money
  ∃ (paper_price : ℕ), paper_price * paper_boxes = total_money →
  -- If A used three times his original amount of money
  ∃ (new_metal_boxes : ℕ), 
    -- He would buy 31 more boxes than B
    new_metal_boxes = paper_boxes + 31 →
    -- And still have 6 yuan left
    metal_price * new_metal_boxes + 6 = 3 * total_money →
  -- Then the price of metal boxes is 12 yuan
  metal_price = 12 ∧
  -- And the price of paper boxes is 10 yuan
  paper_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_prices_l1912_191241


namespace NUMINAMATH_CALUDE_disc_probability_l1912_191268

theorem disc_probability (p_A p_B p_C p_D p_E : ℝ) : 
  p_A = 1/5 →
  p_B = 1/5 →
  p_C = p_D →
  p_C = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_disc_probability_l1912_191268


namespace NUMINAMATH_CALUDE_sample_capacity_l1912_191262

theorem sample_capacity (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ) 
  (h1 : frequency = 36)
  (h2 : frequency_rate = 1/4)
  (h3 : frequency_rate = frequency / n) : n = 144 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_l1912_191262


namespace NUMINAMATH_CALUDE_circular_seating_nine_seven_l1912_191234

/-- The number of ways to choose 7 people from 9 and seat them around a circular table -/
def circular_seating_arrangements (total_people : ℕ) (seats : ℕ) : ℕ :=
  (total_people.choose (total_people - seats)) * (seats - 1).factorial

theorem circular_seating_nine_seven :
  circular_seating_arrangements 9 7 = 25920 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_nine_seven_l1912_191234


namespace NUMINAMATH_CALUDE_shed_area_calculation_l1912_191291

/-- The total inside surface area of a rectangular shed -/
def shedSurfaceArea (width length height : ℝ) : ℝ :=
  2 * (width * height + length * height + width * length)

theorem shed_area_calculation :
  let width : ℝ := 12
  let length : ℝ := 15
  let height : ℝ := 7
  shedSurfaceArea width length height = 738 := by
  sorry

end NUMINAMATH_CALUDE_shed_area_calculation_l1912_191291


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l1912_191288

-- Define binary numbers as natural numbers
def bin1010 : ℕ := 10
def bin111 : ℕ := 7
def bin1001 : ℕ := 9
def bin1011 : ℕ := 11
def bin10111 : ℕ := 23

-- Theorem statement
theorem binary_arithmetic_equality :
  (bin1010 + bin111) - bin1001 + bin1011 = bin10111 := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l1912_191288


namespace NUMINAMATH_CALUDE_square_equation_solution_l1912_191263

theorem square_equation_solution (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 
  3*x^2*y^2 = 588 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1912_191263


namespace NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_90_l1912_191278

theorem smallest_of_three_consecutive_sum_90 (x y z : ℤ) :
  y = x + 1 ∧ z = x + 2 ∧ x + y + z = 90 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_90_l1912_191278


namespace NUMINAMATH_CALUDE_f_composition_value_l1912_191232

def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x + 1

theorem f_composition_value : f (f 2) = 78652 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1912_191232


namespace NUMINAMATH_CALUDE_sum_of_y_values_is_0_04_l1912_191207

-- Define the function g
def g (x : ℝ) : ℝ := (5*x)^2 - 5*x + 2

-- State the theorem
theorem sum_of_y_values_is_0_04 :
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ g y₁ = 12 ∧ g y₂ = 12 ∧ y₁ + y₂ = 0.04 ∧
  ∀ y : ℝ, g y = 12 → y = y₁ ∨ y = y₂ := by
  sorry

end NUMINAMATH_CALUDE_sum_of_y_values_is_0_04_l1912_191207


namespace NUMINAMATH_CALUDE_subtraction_addition_equality_l1912_191248

theorem subtraction_addition_equality : ∃ x : ℤ, 100 - 70 = 70 + x ∧ x = -40 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_addition_equality_l1912_191248


namespace NUMINAMATH_CALUDE_reflection_distance_C_l1912_191270

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflection_distance (p : ℝ × ℝ) : ℝ :=
  2 * |p.2|

theorem reflection_distance_C : reflection_distance (-3, 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_reflection_distance_C_l1912_191270


namespace NUMINAMATH_CALUDE_two_dice_probability_l1912_191296

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of favorable outcomes for the first die (rolling less than 4) -/
def favorableFirst : ℕ := 3

/-- The number of favorable outcomes for the second die (rolling greater than 4) -/
def favorableSecond : ℕ := 4

/-- The probability of the desired outcome when rolling two eight-sided dice -/
theorem two_dice_probability :
  (favorableFirst / numSides) * (favorableSecond / numSides) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_two_dice_probability_l1912_191296


namespace NUMINAMATH_CALUDE_parabola_trajectory_parabola_trajectory_is_parabola_l1912_191237

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

def parallelogram_point (A B F : Point) : Point :=
  Point.mk (A.x + B.x - F.x) (A.y + B.y - F.y)

def intersect_parabola_line (p : Parabola) (l : Line) : Set Point :=
  {P : Point | P.x^2 = 4 * P.y ∧ P.y = l.slope * P.x + l.intercept}

theorem parabola_trajectory (p : Parabola) (l : Line) (F : Point) :
  p.a = 1 ∧ p.h = 0 ∧ p.k = 0 ∧ 
  F.x = 0 ∧ F.y = 1 ∧
  l.intercept = -1 →
  ∃ (R : Point),
    (∃ (A B : Point), A ∈ intersect_parabola_line p l ∧ 
                      B ∈ intersect_parabola_line p l ∧ 
                      R = parallelogram_point A B F) ∧
    R.x^2 = 4 * (R.y + 3) ∧
    abs R.x > 4 :=
sorry

theorem parabola_trajectory_is_parabola (p : Parabola) (l : Line) (F : Point) :
  p.a = 1 ∧ p.h = 0 ∧ p.k = 0 ∧ 
  F.x = 0 ∧ F.y = 1 ∧
  l.intercept = -1 →
  ∃ (new_p : Parabola),
    new_p.a = 1 ∧ new_p.h = 0 ∧ new_p.k = -3 ∧
    (∀ (R : Point),
      (∃ (A B : Point), A ∈ intersect_parabola_line p l ∧ 
                        B ∈ intersect_parabola_line p l ∧ 
                        R = parallelogram_point A B F) →
      R.x^2 = 4 * (R.y + 3) ∧ abs R.x > 4) :=
sorry

end NUMINAMATH_CALUDE_parabola_trajectory_parabola_trajectory_is_parabola_l1912_191237


namespace NUMINAMATH_CALUDE_g_of_two_equals_eighteen_l1912_191286

-- Define g as a function from ℝ to ℝ
variable (g : ℝ → ℝ)

-- State the theorem
theorem g_of_two_equals_eighteen
  (h : ∀ x : ℝ, g (3 * x - 7) = 4 * x + 6) :
  g 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_equals_eighteen_l1912_191286


namespace NUMINAMATH_CALUDE_divisibility_of_ones_l1912_191205

theorem divisibility_of_ones (p : ℕ) (h_prime : Nat.Prime p) (h_ge_7 : p ≥ 7) :
  ∃ k : ℤ, (10^(p-1) - 1) / 9 = k * p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_ones_l1912_191205


namespace NUMINAMATH_CALUDE_find_a_value_l1912_191261

theorem find_a_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l1912_191261


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1912_191249

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def possible_b_values : Set ℝ := {-2, -1, 0, 2}

theorem point_in_second_quadrant (b : ℝ) :
  is_in_second_quadrant (-3) b ∧ b ∈ possible_b_values → b = 2 :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1912_191249


namespace NUMINAMATH_CALUDE_lawn_mowing_payment_l1912_191245

theorem lawn_mowing_payment (payment_rate : ℚ) (lawns_mowed : ℚ) : 
  payment_rate = 13/3 → lawns_mowed = 11/4 → payment_rate * lawns_mowed = 143/12 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_payment_l1912_191245


namespace NUMINAMATH_CALUDE_not_perfect_square_l1912_191275

theorem not_perfect_square : ¬ ∃ (n : ℕ), n^2 = 425102348541 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1912_191275


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2537_l1912_191259

theorem smallest_prime_factor_of_2537 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2537 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2537 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2537_l1912_191259


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l1912_191285

theorem thirty_percent_less_than_ninety (x : ℝ) : x = 42 → x + x/2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l1912_191285


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1912_191239

theorem arithmetic_progression_first_term
  (n : ℕ)
  (a_n : ℝ)
  (d : ℝ)
  (h1 : n = 15)
  (h2 : a_n = 44)
  (h3 : d = 3) :
  ∃ a₁ : ℝ, a₁ = 2 ∧ a_n = a₁ + (n - 1) * d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1912_191239


namespace NUMINAMATH_CALUDE_binomial_27_6_l1912_191206

theorem binomial_27_6 (h1 : Nat.choose 26 4 = 14950)
                      (h2 : Nat.choose 26 5 = 65780)
                      (h3 : Nat.choose 26 6 = 230230) :
  Nat.choose 27 6 = 296010 := by
  sorry

end NUMINAMATH_CALUDE_binomial_27_6_l1912_191206


namespace NUMINAMATH_CALUDE_parallel_to_same_plane_are_parallel_perpendicular_to_same_line_are_parallel_l1912_191266

-- Define the basic types
variable (P : Type) -- Type for planes
variable (L : Type) -- Type for lines

-- Define the relationships
variable (parallel : P → P → Prop) -- Parallel planes
variable (perpendicular : P → L → Prop) -- Plane perpendicular to a line

-- Axioms
axiom parallel_trans (p q r : P) : parallel p q → parallel q r → parallel p r
axiom parallel_symm (p q : P) : parallel p q → parallel q p

-- Theorem 1: Two different planes that are parallel to the same plane are parallel to each other
theorem parallel_to_same_plane_are_parallel (p q r : P) 
  (hp : parallel p r) (hq : parallel q r) (hne : p ≠ q) : 
  parallel p q :=
sorry

-- Theorem 2: Two different planes that are perpendicular to the same line are parallel to each other
theorem perpendicular_to_same_line_are_parallel (p q : P) (l : L)
  (hp : perpendicular p l) (hq : perpendicular q l) (hne : p ≠ q) :
  parallel p q :=
sorry

end NUMINAMATH_CALUDE_parallel_to_same_plane_are_parallel_perpendicular_to_same_line_are_parallel_l1912_191266


namespace NUMINAMATH_CALUDE_transformed_system_solution_l1912_191271

theorem transformed_system_solution 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h₁ : a₁ * 3 - b₁ * 5 = c₁) 
  (h₂ : a₂ * 3 + b₂ * 5 = c₂) :
  a₁ * (11 - 2) - b₁ * 15 = 3 * c₁ ∧ 
  a₂ * (11 - 2) + b₂ * 15 = 3 * c₂ := by
  sorry

end NUMINAMATH_CALUDE_transformed_system_solution_l1912_191271


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1912_191250

theorem inequality_implies_upper_bound (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1912_191250


namespace NUMINAMATH_CALUDE_remainder_415_420_mod_16_l1912_191238

theorem remainder_415_420_mod_16 : 415^420 ≡ 1 [MOD 16] := by
  sorry

end NUMINAMATH_CALUDE_remainder_415_420_mod_16_l1912_191238


namespace NUMINAMATH_CALUDE_julie_work_hours_l1912_191244

/-- Calculates the number of hours Julie needs to work per week during the school year -/
def school_year_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℚ) 
                      (school_year_weeks : ℕ) (school_year_earnings : ℚ) : ℚ :=
  let hourly_wage := summer_earnings / (summer_weeks * summer_hours_per_week)
  let weekly_earnings := school_year_earnings / school_year_weeks
  weekly_earnings / hourly_wage

theorem julie_work_hours :
  school_year_hours 10 60 7500 50 7500 = 12 := by
  sorry

end NUMINAMATH_CALUDE_julie_work_hours_l1912_191244


namespace NUMINAMATH_CALUDE_trapezoid_bases_count_l1912_191235

theorem trapezoid_bases_count : ∃! n : ℕ, 
  n = (Finset.filter (fun p : ℕ × ℕ => 
    10 ∣ p.1 ∧ 10 ∣ p.2 ∧ 
    (p.1 + p.2) * 30 = 1800 ∧ 
    0 < p.1 ∧ 0 < p.2) (Finset.product (Finset.range 181) (Finset.range 181))).card ∧
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_bases_count_l1912_191235


namespace NUMINAMATH_CALUDE_triangle_median_and_symmetric_point_l1912_191255

/-- Triangle OAB with vertices O(0,0), A(2,0), and B(3,2) -/
structure Triangle :=
  (O : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)

/-- Line l containing the median on side OA -/
structure MedianLine :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Point symmetric to A with respect to line l -/
structure SymmetricPoint :=
  (x : ℝ)
  (y : ℝ)

theorem triangle_median_and_symmetric_point 
  (t : Triangle)
  (l : MedianLine)
  (h1 : t.O = (0, 0))
  (h2 : t.A = (2, 0))
  (h3 : t.B = (3, 2))
  (h4 : l.slope = 1)
  (h5 : l.intercept = -1)
  : ∃ (p : SymmetricPoint), 
    (∀ (x y : ℝ), y = l.slope * x + l.intercept ↔ y = x - 1) ∧
    p.x = 1 ∧ p.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_median_and_symmetric_point_l1912_191255


namespace NUMINAMATH_CALUDE_complex_power_32_l1912_191260

open Complex

theorem complex_power_32 : (((1 : ℂ) - I) / (Real.sqrt 2 : ℂ)) ^ 32 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_32_l1912_191260


namespace NUMINAMATH_CALUDE_final_student_count_l1912_191213

/-- The number of students in Beth's class at different stages --/
def students_in_class (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the final number of students in Beth's class --/
theorem final_student_count :
  students_in_class 150 30 15 = 165 := by
  sorry

end NUMINAMATH_CALUDE_final_student_count_l1912_191213


namespace NUMINAMATH_CALUDE_count_strictly_ordered_three_digit_numbers_l1912_191297

/-- The number of three-digit numbers with digits from 1 to 9 in strictly increasing or decreasing order -/
def strictly_ordered_three_digit_numbers : ℕ :=
  2 * (Nat.choose 9 3)

/-- Theorem: The number of three-digit numbers with digits from 1 to 9 
    in strictly increasing or decreasing order is 168 -/
theorem count_strictly_ordered_three_digit_numbers :
  strictly_ordered_three_digit_numbers = 168 := by
  sorry

end NUMINAMATH_CALUDE_count_strictly_ordered_three_digit_numbers_l1912_191297


namespace NUMINAMATH_CALUDE_annual_croissant_expenditure_is_858_l1912_191274

/-- The total annual expenditure on croissants -/
def annual_croissant_expenditure : ℚ :=
  let regular_cost : ℚ := 7/2
  let almond_cost : ℚ := 11/2
  let chocolate_cost : ℚ := 9/2
  let ham_cheese_cost : ℚ := 6
  let weeks_per_year : ℕ := 52
  regular_cost * weeks_per_year +
  almond_cost * weeks_per_year +
  chocolate_cost * weeks_per_year +
  ham_cheese_cost * (weeks_per_year / 2)

/-- Theorem stating that the annual croissant expenditure is $858.00 -/
theorem annual_croissant_expenditure_is_858 :
  annual_croissant_expenditure = 858 := by
  sorry

end NUMINAMATH_CALUDE_annual_croissant_expenditure_is_858_l1912_191274


namespace NUMINAMATH_CALUDE_equation_solutions_l1912_191299

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 2/3 ∧ ∀ x : ℝ, 3*x*(x-2) = 2*(x-2) ↔ x = x₁ ∨ x = x₂) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 7/2 ∧ y₂ = -2 ∧ ∀ x : ℝ, 2*x^2 - 3*x - 14 = 0 ↔ x = y₁ ∨ x = y₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1912_191299


namespace NUMINAMATH_CALUDE_sin_2α_minus_π_over_6_l1912_191254

theorem sin_2α_minus_π_over_6 (α : Real) 
  (h : Real.cos (α + 2 * Real.pi / 3) = 3 / 5) : 
  Real.sin (2 * α - Real.pi / 6) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2α_minus_π_over_6_l1912_191254


namespace NUMINAMATH_CALUDE_bob_fruit_drink_cost_l1912_191252

/-- The cost of Bob's fruit drink -/
def fruit_drink_cost (andy_total bob_sandwich_cost : ℕ) : ℕ :=
  andy_total - bob_sandwich_cost

theorem bob_fruit_drink_cost :
  let andy_total := 5
  let bob_sandwich_cost := 3
  fruit_drink_cost andy_total bob_sandwich_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_bob_fruit_drink_cost_l1912_191252


namespace NUMINAMATH_CALUDE_intersection_line_intersection_distance_l1912_191217

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line
def line (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B

-- Theorem for the line equation
theorem intersection_line (A B : ℝ × ℝ) (h : intersection_points A B) :
  line A.1 A.2 ∧ line B.1 B.2 :=
sorry

-- Theorem for the distance between intersection points
theorem intersection_distance (A B : ℝ × ℝ) (h : intersection_points A B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_intersection_distance_l1912_191217


namespace NUMINAMATH_CALUDE_jennas_number_l1912_191219

theorem jennas_number (x : ℝ) : 3 * ((3 * x + 20) - 5) = 225 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_jennas_number_l1912_191219


namespace NUMINAMATH_CALUDE_price_reduction_l1912_191293

theorem price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h1 : original_price = 120)
  (h2 : final_price = 85)
  (h3 : x > 0 ∧ x < 1) -- Assuming x is a valid percentage
  (h4 : final_price = original_price * (1 - x)^2) :
  120 * (1 - x)^2 = 85 := by sorry

end NUMINAMATH_CALUDE_price_reduction_l1912_191293


namespace NUMINAMATH_CALUDE_inequality_solution_set_range_l1912_191242

/-- The inequality we're working with -/
def inequality (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 - m * x - 1 < 2 * x^2 - 2 * x

/-- The solution set of the inequality with respect to x is R -/
def solution_set_is_R (m : ℝ) : Prop :=
  ∀ x : ℝ, inequality m x

/-- The range of m values for which the solution set is R -/
def m_range : Set ℝ :=
  {m : ℝ | m > -2 ∧ m ≤ 2}

/-- The main theorem to prove -/
theorem inequality_solution_set_range :
  ∀ m : ℝ, solution_set_is_R m ↔ m ∈ m_range :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_range_l1912_191242


namespace NUMINAMATH_CALUDE_max_sum_of_coefficients_l1912_191209

/-- Given a temperature function T(t) = a * sin(t) + b * cos(t) where a and b are positive real
    numbers, and the maximum temperature difference is 10 degrees Celsius, 
    the maximum value of a + b is 5√2. -/
theorem max_sum_of_coefficients (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ t : ℝ, t > 0 → ∃ T : ℝ, T = a * Real.sin t + b * Real.cos t) →
  (∃ t₁ t₂ : ℝ, t₁ > 0 ∧ t₂ > 0 ∧ 
    (a * Real.sin t₁ + b * Real.cos t₁) - (a * Real.sin t₂ + b * Real.cos t₂) = 10) →
  a + b ≤ 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_coefficients_l1912_191209


namespace NUMINAMATH_CALUDE_cone_volume_l1912_191226

/-- A cone with lateral area √5π and whose unfolded lateral area forms a sector with central angle 2√5π/5 has volume 2π/3 -/
theorem cone_volume (lateral_area : ℝ) (central_angle : ℝ) :
  lateral_area = Real.sqrt 5 * Real.pi →
  central_angle = 2 * Real.sqrt 5 * Real.pi / 5 →
  ∃ (r h : ℝ), 
    r > 0 ∧ h > 0 ∧
    lateral_area = Real.pi * r * Real.sqrt (r^2 + h^2) ∧
    (1/3) * Real.pi * r^2 * h = (2/3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l1912_191226


namespace NUMINAMATH_CALUDE_largest_cube_surface_area_l1912_191222

-- Define the dimensions of the cuboid
def cuboid_width : ℝ := 12
def cuboid_length : ℝ := 16
def cuboid_height : ℝ := 14

-- Define the function to calculate the surface area of a cube
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length^2

-- Theorem statement
theorem largest_cube_surface_area :
  let max_side_length := min cuboid_width (min cuboid_length cuboid_height)
  cube_surface_area max_side_length = 864 := by
sorry

end NUMINAMATH_CALUDE_largest_cube_surface_area_l1912_191222


namespace NUMINAMATH_CALUDE_problem_solution_l1912_191272

theorem problem_solution (x : ℚ) : (2 * x + 10 - 2) / 7 = 15 → x = 97 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1912_191272


namespace NUMINAMATH_CALUDE_minimum_k_value_l1912_191208

theorem minimum_k_value : ∃ (k : ℝ), 
  (∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 → 
    (∃ (a b : ℝ), (a = x ∧ b = y) ∨ (a = x ∧ b = z) ∨ (a = y ∧ b = z) ∧ 
      (|a - b| ≤ k ∨ |1/a - 1/b| ≤ k))) ∧
  (∀ (k' : ℝ), 
    (∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 → 
      (∃ (a b : ℝ), (a = x ∧ b = y) ∨ (a = x ∧ b = z) ∨ (a = y ∧ b = z) ∧ 
        (|a - b| ≤ k' ∨ |1/a - 1/b| ≤ k'))) → 
    k ≤ k') ∧
  k = 3/2 := by
sorry

end NUMINAMATH_CALUDE_minimum_k_value_l1912_191208


namespace NUMINAMATH_CALUDE_tan_double_angle_l1912_191243

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by sorry

end NUMINAMATH_CALUDE_tan_double_angle_l1912_191243


namespace NUMINAMATH_CALUDE_april_rainfall_calculation_l1912_191240

def march_rainfall : ℝ := 0.81
def april_decrease : ℝ := 0.35

theorem april_rainfall_calculation :
  march_rainfall - april_decrease = 0.46 := by sorry

end NUMINAMATH_CALUDE_april_rainfall_calculation_l1912_191240


namespace NUMINAMATH_CALUDE_roses_per_set_l1912_191236

theorem roses_per_set (days_in_week : ℕ) (sets_per_day : ℕ) (total_roses : ℕ) :
  days_in_week = 7 →
  sets_per_day = 2 →
  total_roses = 168 →
  total_roses / (days_in_week * sets_per_day) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_roses_per_set_l1912_191236


namespace NUMINAMATH_CALUDE_rbc_divisibility_l1912_191276

theorem rbc_divisibility (r b c : ℕ) : 
  r < 10 → b < 10 → c < 10 →
  (523 * 100 + r * 10 + b) * 10 + c ≡ 0 [MOD 7] →
  (523 * 100 + r * 10 + b) * 10 + c ≡ 0 [MOD 89] →
  r * b * c = 36 := by
sorry

end NUMINAMATH_CALUDE_rbc_divisibility_l1912_191276


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1912_191269

open Real

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Define the line l
def line_l (x y m : ℝ) : Prop := x - y + m = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l A.1 A.2 m ∧ line_l B.1 B.2 m

-- Define the angle ACB
def angle_ACB (A B : ℝ × ℝ) : ℝ := sorry

-- Define the distance between two points
def distance (P Q : ℝ × ℝ) : ℝ := sorry

theorem circle_intersection_theorem (m : ℝ) (A B : ℝ × ℝ) :
  intersection_points A B m →
  ((angle_ACB A B = 2 * π / 3) ∨ (distance A B = 2 * sqrt 3)) →
  (m = sqrt 2 - 1 ∨ m = -sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1912_191269


namespace NUMINAMATH_CALUDE_closest_to_sqrt_two_l1912_191264

theorem closest_to_sqrt_two : 
  let a := Real.sqrt 3 * Real.cos (14 * π / 180) + Real.sin (14 * π / 180)
  let b := Real.sqrt 3 * Real.cos (24 * π / 180) + Real.sin (24 * π / 180)
  let c := Real.sqrt 3 * Real.cos (64 * π / 180) + Real.sin (64 * π / 180)
  let d := Real.sqrt 3 * Real.cos (74 * π / 180) + Real.sin (74 * π / 180)
  abs (d - Real.sqrt 2) < min (abs (a - Real.sqrt 2)) (min (abs (b - Real.sqrt 2)) (abs (c - Real.sqrt 2))) := by
  sorry

end NUMINAMATH_CALUDE_closest_to_sqrt_two_l1912_191264


namespace NUMINAMATH_CALUDE_typist_salary_l1912_191225

theorem typist_salary (x : ℝ) : 
  (x * 1.1 * 0.95 = 1045) → x = 1000 := by sorry

end NUMINAMATH_CALUDE_typist_salary_l1912_191225


namespace NUMINAMATH_CALUDE_impossible_to_achieve_goal_state_l1912_191210

/-- Represents a jar with a certain volume of tea and amount of sugar -/
structure Jar where
  volume : ℕ
  sugar : ℕ

/-- Represents the state of the system with three jars -/
structure SystemState where
  jar1 : Jar
  jar2 : Jar
  jar3 : Jar

/-- Represents a transfer of tea between jars -/
inductive Transfer where
  | from1to2 : Transfer
  | from1to3 : Transfer
  | from2to1 : Transfer
  | from2to3 : Transfer
  | from3to1 : Transfer
  | from3to2 : Transfer

def initial_state : SystemState :=
  { jar1 := { volume := 0, sugar := 0 },
    jar2 := { volume := 700, sugar := 50 },
    jar3 := { volume := 800, sugar := 60 } }

def transfer_amount : ℕ := 100

def is_valid_state (s : SystemState) : Prop :=
  s.jar1.volume + s.jar2.volume + s.jar3.volume = 1500 ∧
  s.jar1.volume % transfer_amount = 0 ∧
  s.jar2.volume % transfer_amount = 0 ∧
  s.jar3.volume % transfer_amount = 0

def apply_transfer (s : SystemState) (t : Transfer) : SystemState :=
  sorry

def is_goal_state (s : SystemState) : Prop :=
  s.jar1.volume = 0 ∧ s.jar2.sugar = s.jar3.sugar

theorem impossible_to_achieve_goal_state :
  ∀ (transfers : List Transfer),
    let final_state := transfers.foldl apply_transfer initial_state
    is_valid_state final_state → ¬is_goal_state final_state :=
  sorry

end NUMINAMATH_CALUDE_impossible_to_achieve_goal_state_l1912_191210


namespace NUMINAMATH_CALUDE_min_y_value_l1912_191281

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 8*x + 54*y) :
  ∃ (y_min : ℝ), y_min = 27 - Real.sqrt 745 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 8*x' + 54*y' → y' ≥ y_min :=
sorry

end NUMINAMATH_CALUDE_min_y_value_l1912_191281


namespace NUMINAMATH_CALUDE_collinear_probability_l1912_191228

/-- The number of dots in one side of the square grid -/
def grid_size : ℕ := 5

/-- The total number of dots in the grid -/
def total_dots : ℕ := grid_size * grid_size

/-- The number of dots to be chosen -/
def chosen_dots : ℕ := 4

/-- The number of ways to choose 4 dots from the grid -/
def total_choices : ℕ := Nat.choose total_dots chosen_dots

/-- The number of collinear sets of 4 dots in the grid -/
def collinear_sets : ℕ := 28

/-- The probability of choosing 4 collinear dots from a 5x5 grid -/
theorem collinear_probability : 
  (collinear_sets : ℚ) / total_choices = 14 / 6325 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_l1912_191228


namespace NUMINAMATH_CALUDE_solve_fruit_problem_l1912_191295

def fruit_problem (apple_price orange_price : ℚ) (total_fruit : ℕ) (initial_avg_price : ℚ) (final_avg_price : ℚ) : Prop :=
  ∀ (apples oranges : ℕ),
    apple_price = 40 / 100 →
    orange_price = 60 / 100 →
    total_fruit = 10 →
    apples + oranges = total_fruit →
    (apple_price * apples + orange_price * oranges) / total_fruit = initial_avg_price →
    initial_avg_price = 56 / 100 →
    final_avg_price = 50 / 100 →
    ∃ (oranges_to_remove : ℕ),
      oranges_to_remove = 6 ∧
      (apple_price * apples + orange_price * (oranges - oranges_to_remove)) / (total_fruit - oranges_to_remove) = final_avg_price

theorem solve_fruit_problem :
  fruit_problem (40/100) (60/100) 10 (56/100) (50/100) :=
sorry

end NUMINAMATH_CALUDE_solve_fruit_problem_l1912_191295


namespace NUMINAMATH_CALUDE_equality_implies_a_equals_two_l1912_191218

theorem equality_implies_a_equals_two (a : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x + a = (x + 1)*(x + 2)) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_equality_implies_a_equals_two_l1912_191218


namespace NUMINAMATH_CALUDE_total_fuel_consumption_l1912_191284

/-- Fuel consumption for city driving in liters per km -/
def city_fuel_rate : ℝ := 6

/-- Fuel consumption for highway driving in liters per km -/
def highway_fuel_rate : ℝ := 4

/-- Distance of the city-only trip in km -/
def city_trip : ℝ := 50

/-- Distance of the highway-only trip in km -/
def highway_trip : ℝ := 35

/-- Distance of the mixed trip's city portion in km -/
def mixed_trip_city : ℝ := 15

/-- Distance of the mixed trip's highway portion in km -/
def mixed_trip_highway : ℝ := 10

/-- Theorem stating that the total fuel consumption for all trips is 570 liters -/
theorem total_fuel_consumption :
  city_fuel_rate * city_trip +
  highway_fuel_rate * highway_trip +
  city_fuel_rate * mixed_trip_city +
  highway_fuel_rate * mixed_trip_highway = 570 := by
  sorry


end NUMINAMATH_CALUDE_total_fuel_consumption_l1912_191284


namespace NUMINAMATH_CALUDE_missing_number_is_1255_l1912_191283

def given_numbers : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755]
def total_count : ℕ := 10
def average : ℕ := 750

theorem missing_number_is_1255 :
  let sum_given := given_numbers.sum
  let total_sum := total_count * average
  total_sum - sum_given = 1255 := by sorry

end NUMINAMATH_CALUDE_missing_number_is_1255_l1912_191283


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l1912_191294

theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l1912_191294


namespace NUMINAMATH_CALUDE_function_equation_implies_linear_l1912_191280

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f x - y * f y = (x - y) * f (x + y)

/-- The theorem stating that any function satisfying the equation must be linear -/
theorem function_equation_implies_linear (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := by
  sorry


end NUMINAMATH_CALUDE_function_equation_implies_linear_l1912_191280


namespace NUMINAMATH_CALUDE_tournament_rounds_theorem_l1912_191287

/-- Represents a person in the tournament -/
inductive Person : Type
  | A
  | B
  | C

/-- Represents the tournament data -/
structure TournamentData where
  rounds_played : Person → Nat
  referee_rounds : Person → Nat

/-- The total number of rounds in the tournament -/
def total_rounds (data : TournamentData) : Nat :=
  (data.rounds_played Person.A + data.rounds_played Person.B + data.rounds_played Person.C + 
   data.referee_rounds Person.A + data.referee_rounds Person.B + data.referee_rounds Person.C) / 2

theorem tournament_rounds_theorem (data : TournamentData) 
  (h1 : data.rounds_played Person.A = 5)
  (h2 : data.rounds_played Person.B = 6)
  (h3 : data.referee_rounds Person.C = 2) :
  total_rounds data = 9 := by
  sorry

end NUMINAMATH_CALUDE_tournament_rounds_theorem_l1912_191287


namespace NUMINAMATH_CALUDE_sum_x_y_z_l1912_191215

theorem sum_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 2 * y) :
  x + y + z = 10 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_z_l1912_191215


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1912_191200

-- Define the universal set U
def U : Finset Nat := {2, 3, 4, 5, 6}

-- Define set A
def A : Finset Nat := {2, 5, 6}

-- Define set B
def B : Finset Nat := {3, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ B) ∩ A = {2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1912_191200


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1912_191265

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line -/
def slope1 (a : ℝ) : ℝ := a

/-- The slope of the second line -/
def slope2 (a : ℝ) : ℝ := a + 2

/-- If the line y = ax - 2 is perpendicular to the line y = (a+2)x + 1, then a = -1 -/
theorem perpendicular_lines (a : ℝ) : 
  perpendicular (slope1 a) (slope2 a) → a = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1912_191265


namespace NUMINAMATH_CALUDE_function_has_positive_zero_l1912_191267

/-- The function f(x) = xe^x - ax - 1 has at least one positive zero for any real a. -/
theorem function_has_positive_zero (a : ℝ) : ∃ x : ℝ, x > 0 ∧ x * Real.exp x - a * x - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_has_positive_zero_l1912_191267


namespace NUMINAMATH_CALUDE_markup_calculation_l1912_191227

/-- The markup required for an article with given purchase price, overhead percentage, and desired net profit. -/
def required_markup (purchase_price : ℝ) (overhead_percent : ℝ) (net_profit : ℝ) : ℝ :=
  purchase_price * overhead_percent + net_profit

/-- Theorem stating that the required markup for the given conditions is $34.80 -/
theorem markup_calculation :
  required_markup 48 0.35 18 = 34.80 := by
  sorry

end NUMINAMATH_CALUDE_markup_calculation_l1912_191227


namespace NUMINAMATH_CALUDE_iron_conducts_electricity_is_deductive_reasoning_l1912_191229

-- Define the set of all objects
variable (Object : Type)

-- Define the property of being a metal
variable (is_metal : Object → Prop)

-- Define the property of conducting electricity
variable (conducts_electricity : Object → Prop)

-- Define iron as an object
variable (iron : Object)

-- Theorem statement
theorem iron_conducts_electricity_is_deductive_reasoning
  (h1 : ∀ x, is_metal x → conducts_electricity x)  -- All metals conduct electricity
  (h2 : is_metal iron)                             -- Iron is a metal
  : conducts_electricity iron                      -- Therefore, iron conducts electricity
  := by sorry

-- The fact that this can be proved using only the given premises
-- demonstrates that this is deductive reasoning

end NUMINAMATH_CALUDE_iron_conducts_electricity_is_deductive_reasoning_l1912_191229


namespace NUMINAMATH_CALUDE_triangle_median_theorem_l1912_191246

/-- Represents a triangle with given side lengths and medians -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  area : ℝ

/-- The theorem stating the properties of the specific triangle -/
theorem triangle_median_theorem (t : Triangle) 
  (h1 : t.a = 5)
  (h2 : t.b = 8)
  (h3 : t.m_a = 4)
  (h4 : t.area = 12) :
  t.m_b = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_median_theorem_l1912_191246


namespace NUMINAMATH_CALUDE_total_cost_is_48_l1912_191233

/-- The cost of a pencil case in yuan -/
def pencil_case_cost : ℕ := 8

/-- The cost of a backpack in yuan -/
def backpack_cost : ℕ := 5 * pencil_case_cost

/-- The total cost of a backpack and a pencil case in yuan -/
def total_cost : ℕ := backpack_cost + pencil_case_cost

/-- Theorem stating that the total cost of a backpack and a pencil case is 48 yuan -/
theorem total_cost_is_48 : total_cost = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_48_l1912_191233


namespace NUMINAMATH_CALUDE_unique_solution_system_l1912_191201

theorem unique_solution_system (x : ℝ) : 
  (3 * x^2 + 8 * x - 3 = 0 ∧ 3 * x^4 + 2 * x^3 - 10 * x^2 + 30 * x - 9 = 0) ↔ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1912_191201


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1912_191279

theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) :
  (∃ x y, x^2 / m^2 - y^2 = 1 ∧ x + Real.sqrt 3 * y = 0) → m = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1912_191279


namespace NUMINAMATH_CALUDE_ratio_equality_l1912_191273

theorem ratio_equality (x y : ℝ) (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1912_191273


namespace NUMINAMATH_CALUDE_exam_average_l1912_191224

theorem exam_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) (h₁ : n₁ = 15) (h₂ : n₂ = 10)
  (h₃ : avg₁ = 75/100) (h₄ : avg_total = 83/100) (h₅ : n₁ + n₂ = 25) :
  let avg₂ := (((n₁ + n₂ : ℚ) * avg_total) - (n₁ * avg₁)) / n₂
  avg₂ = 95/100 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l1912_191224


namespace NUMINAMATH_CALUDE_range_of_a_l1912_191223

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 / Real.exp x - 2 * x

theorem range_of_a (a : ℝ) (h : f (a - 3) + f (2 * a^2) ≤ 0) :
  -3/2 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1912_191223


namespace NUMINAMATH_CALUDE_restaurant_bill_division_l1912_191221

/-- Given a group of friends dividing a restaurant bill evenly, this theorem proves
    the number of friends in the group based on the total bill and individual payment. -/
theorem restaurant_bill_division (total_bill : ℕ) (individual_payment : ℕ) 
    (h1 : total_bill = 135)
    (h2 : individual_payment = 45) :
    total_bill / individual_payment = 3 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_division_l1912_191221


namespace NUMINAMATH_CALUDE_expression_value_l1912_191258

theorem expression_value (m n a b x : ℝ) : 
  (m = -n) → 
  (a * b = 1) → 
  (abs x = 3) → 
  (x^3 - (1 + m + n - a*b) * x^2010 + (m + n) * x^2007 + (-a*b)^2009 = 26 ∨
   x^3 - (1 + m + n - a*b) * x^2010 + (m + n) * x^2007 + (-a*b)^2009 = -28) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l1912_191258


namespace NUMINAMATH_CALUDE_bee_swarm_puzzle_l1912_191282

theorem bee_swarm_puzzle :
  ∃ (x : ℚ),
    x > 0 ∧
    (x / 5 + x / 3 + 3 * (x / 3 - x / 5) + 1 = x) ∧
    x = 15 :=
by sorry

end NUMINAMATH_CALUDE_bee_swarm_puzzle_l1912_191282


namespace NUMINAMATH_CALUDE_system_solution_l1912_191216

theorem system_solution (x y : ℝ) : 
  (x + y = 5 ∧ 2 * x - 3 * y = 20) ↔ (x = 7 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1912_191216


namespace NUMINAMATH_CALUDE_minimum_score_needed_l1912_191204

def current_scores : List ℕ := [90, 80, 70, 60, 85]
def num_current_tests : ℕ := current_scores.length
def sum_current_scores : ℕ := current_scores.sum
def current_average : ℚ := sum_current_scores / num_current_tests
def target_increase : ℚ := 3
def num_total_tests : ℕ := num_current_tests + 1

theorem minimum_score_needed (min_score : ℕ) : 
  (sum_current_scores + min_score) / num_total_tests ≥ current_average + target_increase ∧
  ∀ (score : ℕ), score < min_score → 
    (sum_current_scores + score) / num_total_tests < current_average + target_increase →
  min_score = 95 := by
  sorry

end NUMINAMATH_CALUDE_minimum_score_needed_l1912_191204


namespace NUMINAMATH_CALUDE_apps_deleted_proof_l1912_191202

/-- The number of apps Dave had at the start -/
def initial_apps : ℕ := 23

/-- The number of apps Dave had after deleting some -/
def remaining_apps : ℕ := 5

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := initial_apps - remaining_apps

theorem apps_deleted_proof : deleted_apps = 18 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_proof_l1912_191202


namespace NUMINAMATH_CALUDE_range_of_a_l1912_191214

/-- Custom binary operation on real numbers -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of 'a' given the inequality condition -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1912_191214


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l1912_191247

theorem least_positive_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 2 = 1) ∧ 
  (x % 5 = 2) ∧ 
  (x % 6 = 3) ∧ 
  (∀ y : ℕ, y > 0 ∧ y % 2 = 1 ∧ y % 5 = 2 ∧ y % 6 = 3 → x ≤ y) ∧
  x = 27 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l1912_191247


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1912_191290

theorem smallest_k_no_real_roots : ∃ k : ℤ, 
  (∀ x : ℝ, 3 * x * (k * x - 5) - x^2 + 8 - x^3 ≠ 0) ∧ 
  (∀ m : ℤ, m < k → ∃ x : ℝ, 3 * x * (m * x - 5) - x^2 + 8 - x^3 = 0) ∧
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1912_191290


namespace NUMINAMATH_CALUDE_max_cylinder_volume_in_cube_l1912_191251

/-- The maximum volume of a cylinder inscribed in a cube with side length √3,
    where the cylinder's axis is along a diagonal of the cube. -/
theorem max_cylinder_volume_in_cube :
  let cube_side : ℝ := Real.sqrt 3
  let max_volume : ℝ := π / 2
  ∀ (cylinder_volume : ℝ),
    (∃ (cylinder_radius height : ℝ),
      cylinder_volume = π * cylinder_radius^2 * height ∧
      0 < cylinder_radius ∧
      0 < height ∧
      2 * Real.sqrt 2 * cylinder_radius + height = cube_side) →
    cylinder_volume ≤ max_volume :=
by sorry

end NUMINAMATH_CALUDE_max_cylinder_volume_in_cube_l1912_191251


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_sq_eq_one_l1912_191277

theorem x_eq_one_sufficient_not_necessary_for_x_sq_eq_one :
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧
  (∃ x : ℝ, x ≠ 1 ∧ x^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_sq_eq_one_l1912_191277


namespace NUMINAMATH_CALUDE_unique_modular_solution_l1912_191203

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1453 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l1912_191203


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_interval_l1912_191211

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {x | ∃ y, x^2 + y^2 = 2}

-- Define the interval [0, √2]
def interval : Set ℝ := {x | 0 ≤ x ∧ x ≤ Real.sqrt 2}

-- State the theorem
theorem M_intersect_N_eq_interval : M ∩ N = interval := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_interval_l1912_191211
