import Mathlib

namespace NUMINAMATH_CALUDE_house_price_calculation_l3153_315374

theorem house_price_calculation (P : ℝ) 
  (h1 : P > 0)
  (h2 : 0.56 * P = 56000) : P = 100000 :=
by
  sorry

end NUMINAMATH_CALUDE_house_price_calculation_l3153_315374


namespace NUMINAMATH_CALUDE_unique_g_50_l3153_315314

/-- A function from ℕ to ℕ satisfying the given property -/
def special_function (g : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 2 * g (a^2 + 2*b^2) = (g a)^2 + 3*(g b)^2

theorem unique_g_50 (g : ℕ → ℕ) (h : special_function g) : g 50 = 0 := by
  sorry

#check unique_g_50

end NUMINAMATH_CALUDE_unique_g_50_l3153_315314


namespace NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l3153_315341

/-- A board is a 22x22 grid of natural numbers -/
def Board : Type := Fin 22 → Fin 22 → ℕ

/-- A cell is a position on the board -/
def Cell : Type := Fin 22 × Fin 22

/-- Two cells are adjacent if they share a side or vertex -/
def adjacent (c1 c2 : Cell) : Prop :=
  let (x1, y1) := c1
  let (x2, y2) := c2
  (x1 = x2 ∧ y1.val + 1 = y2.val) ∨
  (x1 = x2 ∧ y2.val + 1 = y1.val) ∨
  (y1 = y2 ∧ x1.val + 1 = x2.val) ∨
  (y1 = y2 ∧ x2.val + 1 = x1.val) ∨
  (x1.val + 1 = x2.val ∧ y1.val + 1 = y2.val) ∨
  (x2.val + 1 = x1.val ∧ y1.val + 1 = y2.val) ∨
  (x1.val + 1 = x2.val ∧ y2.val + 1 = y1.val) ∨
  (x2.val + 1 = x1.val ∧ y2.val + 1 = y1.val)

/-- A valid board contains numbers from 1 to 22² -/
def valid_board (b : Board) : Prop :=
  ∀ x y, 1 ≤ b x y ∧ b x y ≤ 22^2

theorem adjacent_sum_divisible_by_four (b : Board) (h : valid_board b) :
  ∃ c1 c2 : Cell, adjacent c1 c2 ∧ (b c1.1 c1.2 + b c2.1 c2.2) % 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l3153_315341


namespace NUMINAMATH_CALUDE_calculation_proof_l3153_315330

theorem calculation_proof : 
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 10.5 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3153_315330


namespace NUMINAMATH_CALUDE_f_max_value_l3153_315308

/-- A function f(x) that is symmetric about x = -1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := -x^2 * (x^2 + a*x + b)

/-- Symmetry condition: f(x) = f(-2-x) for all x -/
def is_symmetric (a b : ℝ) : Prop := ∀ x, f a b x = f a b (-2-x)

/-- The maximum value of f(x) is 0 -/
theorem f_max_value (a b : ℝ) (h : is_symmetric a b) : 
  ∃ x₀, ∀ x, f a b x ≤ f a b x₀ ∧ f a b x₀ = 0 := by
  sorry

#check f_max_value

end NUMINAMATH_CALUDE_f_max_value_l3153_315308


namespace NUMINAMATH_CALUDE_marie_messages_per_day_l3153_315322

/-- Represents the problem of calculating the number of messages read per day -/
def messages_read_per_day (initial_unread : ℕ) (new_messages_per_day : ℕ) (days_to_clear : ℕ) : ℕ :=
  (initial_unread + new_messages_per_day * days_to_clear) / days_to_clear

/-- Theorem stating that Marie reads 20 messages per day -/
theorem marie_messages_per_day :
  messages_read_per_day 98 6 7 = 20 := by
  sorry

#eval messages_read_per_day 98 6 7

end NUMINAMATH_CALUDE_marie_messages_per_day_l3153_315322


namespace NUMINAMATH_CALUDE_quadratic_roots_equal_integral_l3153_315340

/-- The roots of the quadratic equation 3x^2 - 6x + c = 0 are equal and integral when the discriminant is zero -/
theorem quadratic_roots_equal_integral (c : ℝ) :
  (∀ x : ℝ, 3 * x^2 - 6 * x + c = 0 ↔ x = 1) ∧ ((-6)^2 - 4 * 3 * c = 0) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_equal_integral_l3153_315340


namespace NUMINAMATH_CALUDE_perfect_cube_factors_of_72_is_two_l3153_315383

/-- A function that returns the number of positive factors of 72 that are perfect cubes -/
def perfect_cube_factors_of_72 : ℕ :=
  -- The function should return the number of positive factors of 72 that are perfect cubes
  sorry

/-- Theorem stating that the number of positive factors of 72 that are perfect cubes is 2 -/
theorem perfect_cube_factors_of_72_is_two : perfect_cube_factors_of_72 = 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_factors_of_72_is_two_l3153_315383


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_as_sum_of_three_squares_l3153_315378

theorem largest_multiple_of_seven_as_sum_of_three_squares :
  ∃ n : ℕ, 
    (∃ a : ℕ, n = a^2 + (a+1)^2 + (a+2)^2) ∧ 
    7 ∣ n ∧
    n < 10000 ∧
    (∀ m : ℕ, (∃ b : ℕ, m = b^2 + (b+1)^2 + (b+2)^2) → 7 ∣ m → m < 10000 → m ≤ n) ∧
    n = 8750 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_as_sum_of_three_squares_l3153_315378


namespace NUMINAMATH_CALUDE_product_of_polynomials_l3153_315336

/-- Given two constants m and k, and the equation
    (9d^2 - 5d + m) * (4d^2 + kd - 6) = 36d^4 + 11d^3 - 59d^2 + 10d + 12
    prove that m + k = -7 -/
theorem product_of_polynomials (m k : ℝ) : 
  (∀ d : ℝ, (9*d^2 - 5*d + m) * (4*d^2 + k*d - 6) = 36*d^4 + 11*d^3 - 59*d^2 + 10*d + 12) →
  m + k = -7 := by
  sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l3153_315336


namespace NUMINAMATH_CALUDE_quadratic_roots_relations_l3153_315312

/-- Given complex numbers a, b, c satisfying certain conditions, prove specific algebraic relations -/
theorem quadratic_roots_relations (a b c : ℂ) 
  (h1 : a + b ≠ 0)
  (h2 : b + c ≠ 0)
  (h3 : c + a ≠ 0)
  (h4 : ∀ (x : ℂ), (x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0) → 
    ∃ (y : ℂ), y^2 + a*y + b = 0 ∧ y^2 + b*y + c = 0 ∧ x = -y)
  (h5 : ∀ (x : ℂ), (x^2 + b*x + c = 0 ∧ x^2 + c*x + a = 0) → 
    ∃ (y : ℂ), y^2 + b*y + c = 0 ∧ y^2 + c*y + a = 0 ∧ x = -y)
  (h6 : ∀ (x : ℂ), (x^2 + c*x + a = 0 ∧ x^2 + a*x + b = 0) → 
    ∃ (y : ℂ), y^2 + c*y + a = 0 ∧ y^2 + a*y + b = 0 ∧ x = -y) :
  a^2 + b^2 + c^2 = 18 ∧ 
  a^2*b + b^2*c + c^2*a = 27 ∧ 
  a^3*b^2 + b^3*c^2 + c^3*a^2 = -162 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relations_l3153_315312


namespace NUMINAMATH_CALUDE_coefficient_x3_is_negative_540_l3153_315325

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (3x^2 - 1/x)^6
def coefficient_x3 : ℤ :=
  -3^3 * binomial 6 3

-- Theorem statement
theorem coefficient_x3_is_negative_540 : coefficient_x3 = -540 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_is_negative_540_l3153_315325


namespace NUMINAMATH_CALUDE_glazed_doughnut_cost_l3153_315395

/-- Proves that the cost of each glazed doughnut is $1 given the conditions of the problem -/
theorem glazed_doughnut_cost :
  let total_students : ℕ := 25
  let chocolate_lovers : ℕ := 10
  let glazed_lovers : ℕ := 15
  let chocolate_cost : ℚ := 2
  let total_cost : ℚ := 35
  chocolate_lovers + glazed_lovers = total_students →
  chocolate_lovers * chocolate_cost + glazed_lovers * (total_cost - chocolate_lovers * chocolate_cost) / glazed_lovers = total_cost →
  (total_cost - chocolate_lovers * chocolate_cost) / glazed_lovers = 1 := by
sorry

end NUMINAMATH_CALUDE_glazed_doughnut_cost_l3153_315395


namespace NUMINAMATH_CALUDE_equal_sum_sequence_18th_term_l3153_315390

/-- An equal sum sequence is a sequence where the sum of each term and its next term is constant. -/
def EqualSumSequence (a : ℕ → ℝ) :=
  ∃ s : ℝ, ∀ n : ℕ, a n + a (n + 1) = s

theorem equal_sum_sequence_18th_term 
  (a : ℕ → ℝ) 
  (h_equal_sum : EqualSumSequence a)
  (h_first_term : a 1 = 2)
  (h_common_sum : ∃ s : ℝ, s = 5 ∧ ∀ n : ℕ, a n + a (n + 1) = s) :
  a 18 = 3 := by
  sorry

#check equal_sum_sequence_18th_term

end NUMINAMATH_CALUDE_equal_sum_sequence_18th_term_l3153_315390


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l3153_315346

theorem isabellas_hair_growth (initial_length growth : ℕ) (h1 : initial_length = 18) (h2 : growth = 6) :
  initial_length + growth = 24 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l3153_315346


namespace NUMINAMATH_CALUDE_gcd_1337_382_l3153_315385

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1337_382_l3153_315385


namespace NUMINAMATH_CALUDE_half_inequality_l3153_315364

theorem half_inequality (a b : ℝ) (h : a > b) : a/2 > b/2 := by
  sorry

end NUMINAMATH_CALUDE_half_inequality_l3153_315364


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3153_315363

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -x^2 + 2*x + 3 > 0}
def B : Set ℝ := {x : ℝ | x - 2 < 0}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3153_315363


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3153_315389

theorem exponential_equation_solution (x : ℝ) :
  3^(3*x + 2) = (1 : ℝ) / 27 → x = -(5 : ℝ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3153_315389


namespace NUMINAMATH_CALUDE_wheat_flour_price_l3153_315315

theorem wheat_flour_price (initial_amount : ℕ) (rice_price : ℕ) (rice_packets : ℕ)
  (soda_price : ℕ) (wheat_packets : ℕ) (remaining_balance : ℕ) :
  initial_amount = 500 →
  rice_price = 20 →
  rice_packets = 2 →
  soda_price = 150 →
  wheat_packets = 3 →
  remaining_balance = 235 →
  ∃ (wheat_price : ℕ),
    wheat_price * wheat_packets = initial_amount - remaining_balance - (rice_price * rice_packets + soda_price) ∧
    wheat_price = 25 := by
  sorry

#check wheat_flour_price

end NUMINAMATH_CALUDE_wheat_flour_price_l3153_315315


namespace NUMINAMATH_CALUDE_product_of_reals_l3153_315384

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 10) (sum_cubes_eq : a^3 + b^3 = 172) :
  a * b = 27.6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reals_l3153_315384


namespace NUMINAMATH_CALUDE_max_value_chord_intersection_l3153_315392

theorem max_value_chord_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ (2*a*x + b*y = 2) ∧
   ∃ (x1 y1 x2 y2 : ℝ), x1^2 + y1^2 = 4 ∧ x2^2 + y2^2 = 4 ∧
   2*a*x1 + b*y1 = 2 ∧ 2*a*x2 + b*y2 = 2 ∧
   (x1 - x2)^2 + (y1 - y2)^2 = 12) →
  (∀ c : ℝ, c ≤ (9 * Real.sqrt 2) / 8 ∨ ∃ d : ℝ, d > c ∧ d = a * Real.sqrt (1 + 2*b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_chord_intersection_l3153_315392


namespace NUMINAMATH_CALUDE_farmer_wheat_harvest_l3153_315355

theorem farmer_wheat_harvest 
  (estimated_harvest : ℕ) 
  (additional_harvest : ℕ) 
  (h1 : estimated_harvest = 48097)
  (h2 : additional_harvest = 684) :
  estimated_harvest + additional_harvest = 48781 :=
by sorry

end NUMINAMATH_CALUDE_farmer_wheat_harvest_l3153_315355


namespace NUMINAMATH_CALUDE_correct_propositions_l3153_315375

-- Define the propositions
def proposition1 : Prop := sorry
def proposition2 : Prop := sorry
def proposition3 : Prop := sorry
def proposition4 : Prop := sorry

-- Define a function to check if a proposition is correct
def is_correct (p : Prop) : Prop := sorry

-- Theorem statement
theorem correct_propositions :
  is_correct proposition2 ∧ 
  is_correct proposition3 ∧ 
  ¬is_correct proposition1 ∧ 
  ¬is_correct proposition4 :=
sorry

end NUMINAMATH_CALUDE_correct_propositions_l3153_315375


namespace NUMINAMATH_CALUDE_triangle_area_fraction_l3153_315398

/-- The size of the grid -/
def gridSize : ℕ := 6

/-- The coordinates of the triangle vertices -/
def triangleVertices : List (ℕ × ℕ) := [(3, 3), (3, 5), (5, 5)]

/-- The area of the triangle -/
def triangleArea : ℚ := 2

/-- The area of the entire grid -/
def gridArea : ℕ := gridSize * gridSize

/-- The fraction of the grid area occupied by the triangle -/
def areaFraction : ℚ := triangleArea / gridArea

theorem triangle_area_fraction :
  areaFraction = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_triangle_area_fraction_l3153_315398


namespace NUMINAMATH_CALUDE_average_speed_barney_schwinn_l3153_315391

/-- Proves that the average speed is 31 miles per hour given the problem conditions --/
theorem average_speed_barney_schwinn : 
  let initial_reading : ℕ := 2552
  let final_reading : ℕ := 2992
  let total_time : ℕ := 14
  let distance := final_reading - initial_reading
  let exact_speed := (distance : ℚ) / total_time
  Int.floor (exact_speed + 1/2) = 31 := by sorry

end NUMINAMATH_CALUDE_average_speed_barney_schwinn_l3153_315391


namespace NUMINAMATH_CALUDE_arrangement_count_l3153_315320

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem arrangement_count :
  let total_people : ℕ := 5
  let total_arrangements := factorial total_people
  let arrangements_with_A_first := factorial (total_people - 1)
  let arrangements_with_B_last := factorial (total_people - 1)
  let arrangements_with_A_first_and_B_last := factorial (total_people - 2)
  total_arrangements - arrangements_with_A_first - arrangements_with_B_last + arrangements_with_A_first_and_B_last = 78 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3153_315320


namespace NUMINAMATH_CALUDE_amy_garden_space_l3153_315356

/-- Calculates the total square footage of garden beds -/
def total_sq_ft (num_beds1 num_beds2 : ℕ) (length1 width1 length2 width2 : ℝ) : ℝ :=
  (num_beds1 * length1 * width1) + (num_beds2 * length2 * width2)

/-- Proves that Amy's garden beds have a total of 42 sq ft of growing space -/
theorem amy_garden_space : total_sq_ft 2 2 3 3 4 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_amy_garden_space_l3153_315356


namespace NUMINAMATH_CALUDE_min_cookies_eaten_is_five_l3153_315353

/-- Represents the number of cookies Paco had, ate, and bought -/
structure CookieCount where
  initial : ℕ
  eaten_first : ℕ
  bought : ℕ
  eaten_second : ℕ

/-- The conditions of the cookie problem -/
def cookie_problem (c : CookieCount) : Prop :=
  c.initial = 25 ∧
  c.bought = 3 ∧
  c.eaten_second = c.bought + 2

/-- The minimum number of cookies Paco ate -/
def min_cookies_eaten (c : CookieCount) : ℕ :=
  c.eaten_second

/-- Theorem stating that the minimum number of cookies Paco ate is 5 -/
theorem min_cookies_eaten_is_five :
  ∀ c : CookieCount, cookie_problem c → min_cookies_eaten c = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_min_cookies_eaten_is_five_l3153_315353


namespace NUMINAMATH_CALUDE_trigonometric_sum_l3153_315311

theorem trigonometric_sum (θ φ : Real) 
  (h : (Real.cos θ)^6 / (Real.cos φ)^2 + (Real.sin θ)^6 / (Real.sin φ)^2 = 1) :
  (Real.sin φ)^6 / (Real.sin θ)^2 + (Real.cos φ)^6 / (Real.cos θ)^2 = (1 + (Real.cos (2 * φ))^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_l3153_315311


namespace NUMINAMATH_CALUDE_three_plants_three_colors_l3153_315331

/-- Represents the number of ways to assign plants to colored lamps -/
def plant_lamp_assignments (num_plants : ℕ) (num_identical_plants : ℕ) (num_colors : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of ways to assign 3 plants to 3 colors of lamps -/
theorem three_plants_three_colors :
  plant_lamp_assignments 3 2 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_three_plants_three_colors_l3153_315331


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l3153_315366

theorem coefficient_x3y5_in_expansion_of_x_plus_y_to_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * 1^k * 1^(8-k)) = 256 ∧
  Nat.choose 8 3 = 56 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l3153_315366


namespace NUMINAMATH_CALUDE_total_hamburgers_made_l3153_315382

def initial_hamburgers : ℝ := 9.0
def additional_hamburgers : ℝ := 3.0

theorem total_hamburgers_made :
  initial_hamburgers + additional_hamburgers = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_total_hamburgers_made_l3153_315382


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3153_315396

theorem complex_equation_solution (z : ℂ) : (1 + z) * Complex.I = 1 - z → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3153_315396


namespace NUMINAMATH_CALUDE_cathy_doughnuts_l3153_315371

/-- Prove that Cathy bought 3 dozen doughnuts given the conditions of the problem -/
theorem cathy_doughnuts : 
  ∀ (samuel_dozens cathy_dozens : ℕ),
  samuel_dozens = 2 →
  (samuel_dozens * 12 + cathy_dozens * 12 = (8 + 2) * 6) →
  cathy_dozens = 3 := by sorry

end NUMINAMATH_CALUDE_cathy_doughnuts_l3153_315371


namespace NUMINAMATH_CALUDE_perfect_square_sum_l3153_315399

theorem perfect_square_sum (x y : ℕ) 
  (h : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / (x + 2) + (1 : ℚ) / (y - 2)) : 
  ∃ n : ℕ, x * y + 1 = n ^ 2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l3153_315399


namespace NUMINAMATH_CALUDE_min_days_correct_l3153_315380

/-- Represents the problem of scheduling warriors for duty --/
structure WarriorSchedule where
  total_warriors : ℕ
  min_duty : ℕ
  max_duty : ℕ
  min_days : ℕ

/-- The specific instance of the problem --/
def warrior_problem : WarriorSchedule :=
  { total_warriors := 33
  , min_duty := 9
  , max_duty := 10
  , min_days := 7 }

/-- Theorem stating that the minimum number of days is correct --/
theorem min_days_correct (w : WarriorSchedule) (h1 : w = warrior_problem) :
  ∃ (k l m : ℕ),
    k + l = w.min_days ∧
    w.min_duty * k + w.max_duty * l = w.total_warriors * m ∧
    (∀ (k' l' : ℕ), k' + l' < w.min_days →
      ¬∃ (m' : ℕ), w.min_duty * k' + w.max_duty * l' = w.total_warriors * m') :=
by sorry

end NUMINAMATH_CALUDE_min_days_correct_l3153_315380


namespace NUMINAMATH_CALUDE_donut_distribution_unique_l3153_315317

/-- The distribution of donuts among five people -/
def DonutDistribution : Type := ℕ × ℕ × ℕ × ℕ × ℕ

/-- The total number of donuts -/
def total_donuts : ℕ := 60

/-- Check if a distribution satisfies the given conditions -/
def is_valid_distribution (d : DonutDistribution) : Prop :=
  let (alpha, beta, gamma, delta, epsilon) := d
  delta = 8 ∧
  beta = 3 * gamma ∧
  alpha = 2 * delta ∧
  epsilon = gamma - 4 ∧
  alpha + beta + gamma + delta + epsilon = total_donuts

/-- The correct distribution of donuts -/
def correct_distribution : DonutDistribution := (16, 24, 8, 8, 4)

/-- Theorem stating that the correct distribution is the only valid distribution -/
theorem donut_distribution_unique :
  ∀ d : DonutDistribution, is_valid_distribution d → d = correct_distribution := by
  sorry

end NUMINAMATH_CALUDE_donut_distribution_unique_l3153_315317


namespace NUMINAMATH_CALUDE_student_marks_calculation_l3153_315388

theorem student_marks_calculation (total_marks : ℕ) (passing_percentage : ℚ) (failing_margin : ℕ) (student_marks : ℕ) : 
  total_marks = 500 →
  passing_percentage = 33 / 100 →
  failing_margin = 40 →
  student_marks = total_marks * passing_percentage - failing_margin →
  student_marks = 125 := by
sorry

end NUMINAMATH_CALUDE_student_marks_calculation_l3153_315388


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l3153_315316

theorem sarahs_bowling_score (greg_score sarah_score : ℝ) : 
  sarah_score = greg_score + 50 →
  (greg_score + sarah_score) / 2 = 122.4 →
  sarah_score = 147.4 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l3153_315316


namespace NUMINAMATH_CALUDE_exponential_logarithmic_sum_implies_cosine_sum_l3153_315326

theorem exponential_logarithmic_sum_implies_cosine_sum :
  ∃ (x y z : ℝ),
    (Real.exp x + Real.exp y + Real.exp z = 3) ∧
    (Real.log (1 + x^2) + Real.log (1 + y^2) + Real.log (1 + z^2) = 3) ∧
    (Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) = 3) := by
  sorry

end NUMINAMATH_CALUDE_exponential_logarithmic_sum_implies_cosine_sum_l3153_315326


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l3153_315338

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 64 * π) :
  2 * π * r^2 + π * r^2 = 192 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l3153_315338


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_3_l3153_315359

theorem consecutive_integers_around_sqrt_3 (a b : ℤ) :
  (b = a + 1) →
  (a < Real.sqrt 3) →
  (Real.sqrt 3 < b) →
  a + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_3_l3153_315359


namespace NUMINAMATH_CALUDE_koschei_stopped_month_l3153_315300

/-- The number of children Baba Yaga helps per month -/
def baba_yaga_rate : ℕ := 77

/-- The number of children Koschei helps per month -/
def koschei_rate : ℕ := 12

/-- The number of months between the start and end of the competition -/
def competition_duration : ℕ := 120

/-- The ratio of Baba Yaga's total good deeds to Koschei's at the end -/
def final_ratio : ℕ := 5

/-- Theorem stating when Koschei stopped doing good deeds -/
theorem koschei_stopped_month :
  ∃ (m : ℕ), m * koschei_rate * final_ratio = competition_duration * baba_yaga_rate ∧ m = 154 := by
  sorry

end NUMINAMATH_CALUDE_koschei_stopped_month_l3153_315300


namespace NUMINAMATH_CALUDE_no_real_solution_cubic_equation_l3153_315387

theorem no_real_solution_cubic_equation :
  ∀ x : ℂ, (x^3 + 3*x^2 + 4*x + 6) / (x + 5) = x^2 + 10 →
  (x = (-3 + Complex.I * Real.sqrt 79) / 2 ∨ x = (-3 - Complex.I * Real.sqrt 79) / 2) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_cubic_equation_l3153_315387


namespace NUMINAMATH_CALUDE_four_Z_three_l3153_315386

-- Define the Z operation
def Z (x y : ℤ) : ℤ := x^2 - 3*x*y + y^2

-- Theorem to prove
theorem four_Z_three : Z 4 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_three_l3153_315386


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l3153_315393

theorem sine_cosine_inequality (x a b : ℝ) :
  (Real.sin x + a * Real.cos x) * (Real.sin x + b * Real.cos x) ≤ 1 + ((a + b) / 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l3153_315393


namespace NUMINAMATH_CALUDE_dvd_sales_l3153_315305

theorem dvd_sales (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * cd →
  dvd + cd = 273 →
  dvd = 168 := by
sorry

end NUMINAMATH_CALUDE_dvd_sales_l3153_315305


namespace NUMINAMATH_CALUDE_multiple_problem_l3153_315349

theorem multiple_problem (n : ℝ) (m : ℝ) (h1 : n = 25.0) (h2 : 2 * n = m * n - 25) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l3153_315349


namespace NUMINAMATH_CALUDE_mersenne_prime_implies_prime_exponent_l3153_315310

theorem mersenne_prime_implies_prime_exponent (n : ℕ) :
  Nat.Prime (2^n - 1) → Nat.Prime n := by
sorry

end NUMINAMATH_CALUDE_mersenne_prime_implies_prime_exponent_l3153_315310


namespace NUMINAMATH_CALUDE_wall_length_is_800_l3153_315397

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  width : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.height * w.width

/-- Theorem: The length of the wall is 800 cm -/
theorem wall_length_is_800 (brick : BrickDimensions) (wall : WallDimensions) 
    (h1 : brick.length = 40)
    (h2 : brick.width = 11.25)
    (h3 : brick.height = 6)
    (h4 : wall.height = 600)
    (h5 : wall.width = 22.5)
    (h6 : wallVolume wall / brickVolume brick = 4000) :
    wall.length = 800 := by
  sorry

end NUMINAMATH_CALUDE_wall_length_is_800_l3153_315397


namespace NUMINAMATH_CALUDE_sin_75_cos_75_double_l3153_315344

theorem sin_75_cos_75_double : 2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_75_double_l3153_315344


namespace NUMINAMATH_CALUDE_solve_equation_l3153_315377

theorem solve_equation : ∃ x : ℝ, 2*x + 3*x = 600 - (4*x + 6*x) ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3153_315377


namespace NUMINAMATH_CALUDE_vector_dot_product_cosine_l3153_315303

theorem vector_dot_product_cosine (x : ℝ) : 
  let a : ℝ × ℝ := (Real.cos x, Real.sin x)
  let b : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
  (a.1 * b.1 + a.2 * b.2 = 8/5) → Real.cos (x - π/4) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_cosine_l3153_315303


namespace NUMINAMATH_CALUDE_student_arrangement_equality_l3153_315352

theorem student_arrangement_equality (n : ℕ) : 
  n = 48 → 
  (Nat.factorial n) = (Nat.factorial n) :=
by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_equality_l3153_315352


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l3153_315339

theorem max_value_of_fraction (x y z u v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0) (hv : v > 0) :
  (x*y + y*z + z*u + u*v) / (2*x^2 + y^2 + 2*z^2 + u^2 + 2*v^2) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l3153_315339


namespace NUMINAMATH_CALUDE_simplify_expression_l3153_315354

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  3 * x^2 * y * (2 / (9 * x^3 * y)) = 2 / (3 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3153_315354


namespace NUMINAMATH_CALUDE_tims_car_initial_price_l3153_315304

/-- The initial price of a car, given its depreciation rate and value after a certain time -/
def initial_price (depreciation_rate : ℕ → ℚ) (years : ℕ) (final_value : ℚ) : ℚ :=
  final_value + (years : ℚ) * depreciation_rate years

/-- Theorem: The initial price of Tim's car is $20,000 -/
theorem tims_car_initial_price :
  let depreciation_rate : ℕ → ℚ := λ _ => 1000
  let years : ℕ := 6
  let final_value : ℚ := 14000
  initial_price depreciation_rate years final_value = 20000 := by
  sorry

end NUMINAMATH_CALUDE_tims_car_initial_price_l3153_315304


namespace NUMINAMATH_CALUDE_fuel_cost_difference_l3153_315362

-- Define the parameters
def num_vans : ℝ := 6.0
def num_buses : ℝ := 8
def people_per_van : ℝ := 6
def people_per_bus : ℝ := 18
def van_distance : ℝ := 120
def bus_distance : ℝ := 150
def van_efficiency : ℝ := 20
def bus_efficiency : ℝ := 6
def van_fuel_cost : ℝ := 2.5
def bus_fuel_cost : ℝ := 3

-- Define the theorem
theorem fuel_cost_difference : 
  let van_total_distance := num_vans * van_distance
  let bus_total_distance := num_buses * bus_distance
  let van_fuel_consumed := van_total_distance / van_efficiency
  let bus_fuel_consumed := bus_total_distance / bus_efficiency
  let van_total_cost := van_fuel_consumed * van_fuel_cost
  let bus_total_cost := bus_fuel_consumed * bus_fuel_cost
  bus_total_cost - van_total_cost = 510 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_difference_l3153_315362


namespace NUMINAMATH_CALUDE_sum_set_bounds_l3153_315365

theorem sum_set_bounds (A : Finset ℕ) (S : Finset ℕ) :
  A.card = 100 →
  S = Finset.image (λ (p : ℕ × ℕ) => p.1 + p.2) (A.product A) →
  199 ≤ S.card ∧ S.card ≤ 5050 := by
  sorry

end NUMINAMATH_CALUDE_sum_set_bounds_l3153_315365


namespace NUMINAMATH_CALUDE_equation_value_l3153_315342

theorem equation_value (x y z w : ℝ) 
  (h1 : 4 * x * z + y * w = 3)
  (h2 : (2 * x + y) * (2 * z + w) = 15) :
  x * w + y * z = 6 := by
sorry

end NUMINAMATH_CALUDE_equation_value_l3153_315342


namespace NUMINAMATH_CALUDE_units_digit_of_l_squared_plus_two_to_l_l3153_315327

def l : ℕ := 15^2 + 2^15

theorem units_digit_of_l_squared_plus_two_to_l (l : ℕ) (h : l = 15^2 + 2^15) : 
  (l^2 + 2^l) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_l_squared_plus_two_to_l_l3153_315327


namespace NUMINAMATH_CALUDE_calculation_problem_l3153_315361

theorem calculation_problem (x : ℝ) : 10 * 1.8 - (2 * x / 0.3) = 50 ↔ x = -4.8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_problem_l3153_315361


namespace NUMINAMATH_CALUDE_sqrt_23_bound_l3153_315337

theorem sqrt_23_bound : 4.5 < Real.sqrt 23 ∧ Real.sqrt 23 < 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_23_bound_l3153_315337


namespace NUMINAMATH_CALUDE_eggs_for_bread_l3153_315347

/-- The number of dozens of eggs needed given the total weight required, weight per egg, and eggs per dozen -/
def eggs_needed (total_weight : ℚ) (weight_per_egg : ℚ) (eggs_per_dozen : ℕ) : ℚ :=
  (total_weight / weight_per_egg) / eggs_per_dozen

/-- Theorem stating that 8 dozens of eggs are needed for the given conditions -/
theorem eggs_for_bread : eggs_needed 6 (1/16) 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eggs_for_bread_l3153_315347


namespace NUMINAMATH_CALUDE_triangle_heights_l3153_315351

theorem triangle_heights (ha hb : ℝ) (d : ℕ) :
  ha = 3 →
  hb = 7 →
  (∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a * ha = b * hb ∧
    b * hb = c * d ∧
    a * ha = c * d ∧
    a + b > c ∧ a + c > b ∧ b + c > a) →
  d = 3 ∨ d = 4 ∨ d = 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_heights_l3153_315351


namespace NUMINAMATH_CALUDE_bens_previous_salary_l3153_315307

/-- Prove that Ben's previous job's annual salary was $75,000 given the conditions of his new job --/
theorem bens_previous_salary (new_base_salary : ℝ) (commission_rate : ℝ) (sale_price : ℝ) (min_sales : ℝ) :
  new_base_salary = 45000 →
  commission_rate = 0.15 →
  sale_price = 750 →
  min_sales = 266.67 →
  ∃ (previous_salary : ℝ), 
    previous_salary ≥ new_base_salary + commission_rate * sale_price * min_sales ∧
    previous_salary < new_base_salary + commission_rate * sale_price * min_sales + 1 :=
by
  sorry

#eval (45000 : ℝ) + 0.15 * 750 * 266.67

end NUMINAMATH_CALUDE_bens_previous_salary_l3153_315307


namespace NUMINAMATH_CALUDE_product_of_zeros_range_l3153_315367

noncomputable section

def f (x : ℝ) : ℝ := 
  if x ≥ 1 then Real.log x else 1 - x / 2

def F (m : ℝ) (x : ℝ) : ℝ := f (f x + 1) + m

theorem product_of_zeros_range (m : ℝ) 
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F m x₁ = 0 ∧ F m x₂ = 0) :
  ∃ p : ℝ, p < Real.sqrt (Real.exp 1) ∧ 
    ∀ q : ℝ, q < p → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F m x₁ = 0 ∧ F m x₂ = 0 ∧ x₁ * x₂ = q :=
sorry

end

end NUMINAMATH_CALUDE_product_of_zeros_range_l3153_315367


namespace NUMINAMATH_CALUDE_min_speed_against_current_l3153_315368

/-- The minimum speed against the current given the following conditions:
    - Man's speed with the current is 35 km/hr
    - Speed of the current varies between 5.6 km/hr and 8.4 km/hr
    - Wind resistance provides a decelerating force between 0.1 to 0.3 times his speed -/
theorem min_speed_against_current (speed_with_current : ℝ) 
  (current_speed_min current_speed_max : ℝ) 
  (wind_resistance_min wind_resistance_max : ℝ) :
  speed_with_current = 35 →
  current_speed_min = 5.6 →
  current_speed_max = 8.4 →
  wind_resistance_min = 0.1 →
  wind_resistance_max = 0.3 →
  ∃ (speed_against_current : ℝ), 
    speed_against_current ≥ 14.7 ∧ 
    (∀ (actual_current_speed actual_wind_resistance : ℝ),
      actual_current_speed ≥ current_speed_min →
      actual_current_speed ≤ current_speed_max →
      actual_wind_resistance ≥ wind_resistance_min →
      actual_wind_resistance ≤ wind_resistance_max →
      speed_against_current ≤ speed_with_current - actual_current_speed - 
        actual_wind_resistance * (speed_with_current - actual_current_speed)) :=
by sorry

end NUMINAMATH_CALUDE_min_speed_against_current_l3153_315368


namespace NUMINAMATH_CALUDE_sum_floor_equals_n_l3153_315306

/-- For any natural number n, the sum of floor((n + 2^k) / 2^(k+1)) from k = 0 to infinity equals n -/
theorem sum_floor_equals_n (n : ℕ) :
  (∑' k : ℕ, ⌊(n + 2^k : ℝ) / (2^(k+1) : ℝ)⌋) = n :=
sorry

end NUMINAMATH_CALUDE_sum_floor_equals_n_l3153_315306


namespace NUMINAMATH_CALUDE_smallest_difference_of_valid_units_digits_l3153_315376

def is_multiple_of_five (n : ℕ) : Prop := ∃ k, n = 5 * k

def valid_units_digit (x : ℕ) : Prop :=
  x < 10 ∧ is_multiple_of_five (520 + x)

theorem smallest_difference_of_valid_units_digits :
  ∃ (a b : ℕ), valid_units_digit a ∧ valid_units_digit b ∧
  (∀ (c d : ℕ), valid_units_digit c → valid_units_digit d →
    a - b ≤ c - d ∨ b - a ≤ c - d) ∧
  a - b = 5 ∨ b - a = 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_of_valid_units_digits_l3153_315376


namespace NUMINAMATH_CALUDE_canoe_kayak_rental_difference_l3153_315369

theorem canoe_kayak_rental_difference :
  ∀ (canoe_cost kayak_cost : ℚ) 
    (canoe_count kayak_count : ℕ) 
    (total_revenue : ℚ),
  canoe_cost = 12 →
  kayak_cost = 18 →
  canoe_count = (3 * kayak_count) / 2 →
  total_revenue = canoe_cost * canoe_count + kayak_cost * kayak_count →
  total_revenue = 504 →
  canoe_count - kayak_count = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_rental_difference_l3153_315369


namespace NUMINAMATH_CALUDE_expression_values_l3153_315394

theorem expression_values (x y : ℝ) (h1 : x + y = 2) (h2 : y > 0) (h3 : x ≠ 0) :
  (1 / |x| + |x| / (y + 2) = 3/4) ∨ (1 / |x| + |x| / (y + 2) = 5/4) :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l3153_315394


namespace NUMINAMATH_CALUDE_max_value_of_f_l3153_315323

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * k * x + 1

theorem max_value_of_f (k : ℝ) : 
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f k x ≤ 4) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, f k x = 4) →
  k = -3 ∨ k = 3/8 := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3153_315323


namespace NUMINAMATH_CALUDE_quadratic_satisfies_conditions_l3153_315333

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

-- State the theorem
theorem quadratic_satisfies_conditions : 
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_satisfies_conditions_l3153_315333


namespace NUMINAMATH_CALUDE_mary_james_seating_probability_l3153_315301

/-- The number of chairs in the row -/
def totalChairs : ℕ := 10

/-- The set of broken chair numbers -/
def brokenChairs : Finset ℕ := {4, 7}

/-- The set of available chairs -/
def availableChairs : Finset ℕ := Finset.range totalChairs \ brokenChairs

/-- The probability that Mary and James do not sit next to each other -/
def probabilityNotAdjacent : ℚ := 3/4

theorem mary_james_seating_probability :
  let totalWays := Nat.choose availableChairs.card 2
  let adjacentWays := (availableChairs.filter (fun n => n + 1 ∈ availableChairs)).card
  (totalWays - adjacentWays : ℚ) / totalWays = probabilityNotAdjacent :=
sorry

end NUMINAMATH_CALUDE_mary_james_seating_probability_l3153_315301


namespace NUMINAMATH_CALUDE_intimate_interval_is_two_three_l3153_315370

def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

def intimate_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, |f x - g x| ≤ 1

theorem intimate_interval_is_two_three :
  ∃ (a b : ℝ), a = 2 ∧ b = 3 ∧
  intimate_functions f g a b ∧
  ∀ (c d : ℝ), c < 2 ∨ d > 3 → ¬intimate_functions f g c d :=
sorry

end NUMINAMATH_CALUDE_intimate_interval_is_two_three_l3153_315370


namespace NUMINAMATH_CALUDE_remainder_theorem_l3153_315379

theorem remainder_theorem (n : ℤ) (h : n % 9 = 4) : (5 * n - 12) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3153_315379


namespace NUMINAMATH_CALUDE_decimal_addition_l3153_315313

theorem decimal_addition : (0.8 : ℝ) + 0.02 = 0.82 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l3153_315313


namespace NUMINAMATH_CALUDE_todds_snow_cone_business_l3153_315321

/-- Todd's snow-cone business problem -/
theorem todds_snow_cone_business 
  (borrowed : ℝ) 
  (repay : ℝ) 
  (ingredients_cost : ℝ) 
  (num_sold : ℕ) 
  (price_per_cone : ℝ) 
  (h1 : borrowed = 100)
  (h2 : repay = 110)
  (h3 : ingredients_cost = 75)
  (h4 : num_sold = 200)
  (h5 : price_per_cone = 0.75)
  : borrowed - ingredients_cost + (num_sold : ℝ) * price_per_cone - repay = 65 :=
by
  sorry


end NUMINAMATH_CALUDE_todds_snow_cone_business_l3153_315321


namespace NUMINAMATH_CALUDE_inscribed_triangle_polygon_sides_l3153_315319

/-- A triangle inscribed in a circle with specific angle relationships -/
structure InscribedTriangle where
  -- The circle in which the triangle is inscribed
  circle : Real
  -- The angles of the triangle
  angleA : Real
  angleB : Real
  angleC : Real
  -- The number of sides of the regular polygon
  n : ℕ
  -- Conditions
  angle_sum : angleA + angleB + angleC = 180
  angle_B : angleB = 3 * angleA
  angle_C : angleC = 5 * angleA
  polygon_arc : (360 : Real) / n = 140

/-- Theorem: The number of sides of the regular polygon is 5 -/
theorem inscribed_triangle_polygon_sides (t : InscribedTriangle) : t.n = 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_polygon_sides_l3153_315319


namespace NUMINAMATH_CALUDE_total_items_in_jar_l3153_315348

/-- The total number of items in a jar with candy and secret eggs -/
theorem total_items_in_jar (candy : ℝ) (secret_eggs : ℝ) (h1 : candy = 3409.0) (h2 : secret_eggs = 145.0) :
  candy + secret_eggs = 3554.0 := by
  sorry

end NUMINAMATH_CALUDE_total_items_in_jar_l3153_315348


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l3153_315358

/-- Given a train of length 1200 meters that crosses a tree in 120 seconds,
    calculate the time required to pass a platform of length 900 meters. -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (tree_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200) 
  (h2 : tree_passing_time = 120) 
  (h3 : platform_length = 900) :
  (train_length + platform_length) / (train_length / tree_passing_time) = 210 := by
  sorry

#check train_platform_passing_time

end NUMINAMATH_CALUDE_train_platform_passing_time_l3153_315358


namespace NUMINAMATH_CALUDE_trailing_zeroes_of_six_factorial_l3153_315335

-- Define the function z(n) that counts trailing zeroes in n!
def z (n : ℕ) : ℕ := 
  (n / 5) + (n / 25) + (n / 125)

-- State the theorem
theorem trailing_zeroes_of_six_factorial : z (z 6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeroes_of_six_factorial_l3153_315335


namespace NUMINAMATH_CALUDE_rotation_symmetry_l3153_315329

-- Define the directions
inductive Direction
  | Up
  | Down
  | Left
  | Right

-- Define a square configuration
def SquareConfig := List Direction

-- Define a rotation function
def rotate90Clockwise (config : SquareConfig) : SquareConfig :=
  match config with
  | [a, b, c, d] => [d, a, b, c]
  | _ => []  -- Return empty list for invalid configurations

-- Theorem statement
theorem rotation_symmetry (original : SquareConfig) :
  original = [Direction.Up, Direction.Right, Direction.Down, Direction.Left] →
  rotate90Clockwise original = [Direction.Right, Direction.Down, Direction.Left, Direction.Up] :=
by
  sorry


end NUMINAMATH_CALUDE_rotation_symmetry_l3153_315329


namespace NUMINAMATH_CALUDE_x_value_when_y_is_negative_four_l3153_315343

theorem x_value_when_y_is_negative_four :
  ∀ x y : ℝ, 16 * (3 : ℝ)^x = 7^(y + 4) → y = -4 → x = -4 * (Real.log 2 / Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_negative_four_l3153_315343


namespace NUMINAMATH_CALUDE_disjunction_false_l3153_315334

theorem disjunction_false :
  ¬(
    (∃ x : ℝ, x^2 + 1 < 2*x) ∨
    (∀ m : ℝ, (∀ x : ℝ, m*x^2 - m*x + 1 > 0) → (0 < m ∧ m < 4))
  ) := by sorry

end NUMINAMATH_CALUDE_disjunction_false_l3153_315334


namespace NUMINAMATH_CALUDE_remainder_11_power_4001_mod_13_l3153_315357

theorem remainder_11_power_4001_mod_13 : 11^4001 % 13 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_power_4001_mod_13_l3153_315357


namespace NUMINAMATH_CALUDE_tinas_pens_l3153_315360

theorem tinas_pens (pink_pens green_pens blue_pens : ℕ) : 
  pink_pens = 12 →
  green_pens < pink_pens →
  blue_pens = green_pens + 3 →
  pink_pens + green_pens + blue_pens = 21 →
  pink_pens - green_pens = 9 := by
sorry

end NUMINAMATH_CALUDE_tinas_pens_l3153_315360


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l3153_315309

theorem sqrt_sum_problem (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l3153_315309


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l3153_315328

theorem smallest_n_for_candy_purchase : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → 24 * m = Nat.lcm (Nat.lcm 18 16) 20 → n ≤ m) ∧
  24 * n = Nat.lcm (Nat.lcm 18 16) 20 ∧ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l3153_315328


namespace NUMINAMATH_CALUDE_complement_of_M_l3153_315345

def M : Set ℝ := {x | x + 3 > 0}

theorem complement_of_M : 
  (Set.univ : Set ℝ) \ M = {x : ℝ | x ≤ -3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l3153_315345


namespace NUMINAMATH_CALUDE_ratio_problem_l3153_315324

theorem ratio_problem (x y z w : ℝ) 
  (h1 : 0.1 * x = 0.2 * y) 
  (h2 : 0.3 * y = 0.4 * z) 
  (h3 : 0.5 * z = 0.6 * w) : 
  ∃ (k : ℝ), k > 0 ∧ x = 8 * k ∧ y = 4 * k ∧ z = 3 * k ∧ w = 2.5 * k :=
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3153_315324


namespace NUMINAMATH_CALUDE_divisibility_theorem_l3153_315318

theorem divisibility_theorem (K M N : ℤ) (hK : K ≠ 0) (hM : M ≠ 0) (hN : N ≠ 0) (hcoprime : Nat.Coprime K.natAbs M.natAbs) :
  ∃ x : ℤ, ∃ y : ℤ, M * x + N = K * y := by
sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l3153_315318


namespace NUMINAMATH_CALUDE_whitney_cant_afford_all_items_l3153_315350

def poster_cost : ℕ := 7
def notebook_cost : ℕ := 5
def bookmark_cost : ℕ := 3
def pencil_cost : ℕ := 1

def poster_quantity : ℕ := 3
def notebook_quantity : ℕ := 4
def bookmark_quantity : ℕ := 5
def pencil_quantity : ℕ := 2

def available_funds : ℕ := 2 * 20

theorem whitney_cant_afford_all_items :
  poster_cost * poster_quantity +
  notebook_cost * notebook_quantity +
  bookmark_cost * bookmark_quantity +
  pencil_cost * pencil_quantity > available_funds :=
by sorry

end NUMINAMATH_CALUDE_whitney_cant_afford_all_items_l3153_315350


namespace NUMINAMATH_CALUDE_tangent_equality_mod_180_l3153_315373

theorem tangent_equality_mod_180 (m : ℤ) : 
  -180 < m ∧ m < 180 ∧ Real.tan (m * π / 180) = Real.tan (2530 * π / 180) → m = 10 := by
  sorry

end NUMINAMATH_CALUDE_tangent_equality_mod_180_l3153_315373


namespace NUMINAMATH_CALUDE_ratio_problem_l3153_315332

theorem ratio_problem (x y : ℝ) (h : (3*x - 2*y) / (2*x + y) = 4/5) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3153_315332


namespace NUMINAMATH_CALUDE_sale_price_calculation_l3153_315302

/-- Calculates the sale price including tax given the cost price, profit rate, and tax rate -/
def salePriceWithTax (costPrice : ℝ) (profitRate : ℝ) (taxRate : ℝ) : ℝ :=
  let sellingPrice := costPrice * (1 + profitRate)
  sellingPrice * (1 + taxRate)

/-- Theorem stating that the sale price including tax is approximately 677.60 -/
theorem sale_price_calculation :
  let costPrice : ℝ := 545.13
  let profitRate : ℝ := 0.13
  let taxRate : ℝ := 0.10
  abs (salePriceWithTax costPrice profitRate taxRate - 677.60) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l3153_315302


namespace NUMINAMATH_CALUDE_sand_amount_l3153_315372

/-- The total amount of sand in tons -/
def total_sand : ℕ := 180

/-- The originally scheduled daily transport rate in tons -/
def scheduled_rate : ℕ := 15

/-- The actual daily transport rate in tons -/
def actual_rate : ℕ := 20

/-- The number of days the task was completed ahead of schedule -/
def days_ahead : ℕ := 3

/-- Theorem stating that the total amount of sand is 180 tons -/
theorem sand_amount :
  ∃ (scheduled_days : ℕ),
    scheduled_days * scheduled_rate = total_sand ∧
    (scheduled_days - days_ahead) * actual_rate = total_sand :=
by sorry

end NUMINAMATH_CALUDE_sand_amount_l3153_315372


namespace NUMINAMATH_CALUDE_problem_statement_l3153_315381

theorem problem_statement : 103^4 - 4*103^3 + 6*103^2 - 4*103 + 1 = 108243216 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3153_315381
