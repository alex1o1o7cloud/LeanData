import Mathlib

namespace NUMINAMATH_CALUDE_adams_earnings_l3006_300622

theorem adams_earnings (daily_wage : ℝ) (tax_rate : ℝ) (work_days : ℕ) :
  daily_wage = 40 →
  tax_rate = 0.1 →
  work_days = 30 →
  (daily_wage * (1 - tax_rate) * work_days : ℝ) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_adams_earnings_l3006_300622


namespace NUMINAMATH_CALUDE_triangle_theorem_l3006_300618

/-- Triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) (h1 : t.a * (Real.cos (t.C / 2))^2 + t.c * (Real.cos (t.A / 2))^2 = (3/2) * t.b)
  (h2 : t.B = π/3) (h3 : (1/2) * t.a * t.c * Real.sin t.B = 8 * Real.sqrt 3) :
  (2 * t.b = t.a + t.c) ∧ (t.b = 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3006_300618


namespace NUMINAMATH_CALUDE_tan_expression_equality_l3006_300659

theorem tan_expression_equality (θ : Real) (h : Real.tan θ = 3) :
  let k : Real := 1/2
  (1 - k * Real.cos θ) / Real.sin θ - (2 * Real.sin θ) / (1 + Real.cos θ) =
  (20 - Real.sqrt 10) / (3 * Real.sqrt 10) - (6 * Real.sqrt 10) / (10 + Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_tan_expression_equality_l3006_300659


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3006_300634

theorem complex_fraction_simplification :
  (Complex.I - 1) / (1 + Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3006_300634


namespace NUMINAMATH_CALUDE_wholesale_price_correct_l3006_300699

/-- The retail price of the machine -/
def retail_price : ℝ := 167.99999999999997

/-- The discount rate applied to the retail price -/
def discount_rate : ℝ := 0.10

/-- The profit rate as a percentage of the wholesale price -/
def profit_rate : ℝ := 0.20

/-- The wholesale price of the machine -/
def wholesale_price : ℝ := 126.00

/-- Theorem stating that the given wholesale price is correct -/
theorem wholesale_price_correct : 
  wholesale_price = (retail_price * (1 - discount_rate)) / (1 + profit_rate) :=
sorry

end NUMINAMATH_CALUDE_wholesale_price_correct_l3006_300699


namespace NUMINAMATH_CALUDE_initial_amount_satisfies_equation_l3006_300681

/-- The initial amount of money the man has --/
def initial_amount : ℝ := 6.25

/-- The amount spent at each shop --/
def amount_spent : ℝ := 10

/-- The equation representing the man's transactions --/
def transaction_equation (x : ℝ) : Prop :=
  2 * (2 * (2 * x - amount_spent) - amount_spent) - amount_spent = 0

/-- Theorem stating that the initial amount satisfies the transaction equation --/
theorem initial_amount_satisfies_equation : 
  transaction_equation initial_amount := by sorry

end NUMINAMATH_CALUDE_initial_amount_satisfies_equation_l3006_300681


namespace NUMINAMATH_CALUDE_negation_of_ln_positive_l3006_300695

theorem negation_of_ln_positive :
  (¬ ∀ x : ℝ, x > 0 → Real.log x > 0) ↔ (∃ x : ℝ, x > 0 ∧ Real.log x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_ln_positive_l3006_300695


namespace NUMINAMATH_CALUDE_inequality_proof_l3006_300607

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (1/x + 1/y + 1/z) - (x + y + z) ≥ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3006_300607


namespace NUMINAMATH_CALUDE_cost_effective_plan_l3006_300641

/-- Represents the ticket purchasing scenario for a group of employees visiting a scenic spot. -/
structure TicketScenario where
  totalEmployees : ℕ
  regularPrice : ℕ
  groupDiscountRate : ℚ
  womenDiscountRate : ℚ
  minGroupSize : ℕ

/-- Calculates the cost of tickets with women's discount applied. -/
def womenDiscountCost (s : TicketScenario) (numWomen : ℕ) : ℚ :=
  s.regularPrice * s.womenDiscountRate * numWomen + s.regularPrice * (s.totalEmployees - numWomen)

/-- Calculates the cost of tickets with group discount applied. -/
def groupDiscountCost (s : TicketScenario) : ℚ :=
  s.totalEmployees * s.regularPrice * (1 - s.groupDiscountRate)

/-- Theorem stating the conditions for the most cost-effective ticket purchasing plan. -/
theorem cost_effective_plan (s : TicketScenario) (numWomen : ℕ) :
  s.totalEmployees = 30 ∧
  s.regularPrice = 80 ∧
  s.groupDiscountRate = 1/5 ∧
  s.womenDiscountRate = 1/2 ∧
  s.minGroupSize = 30 ∧
  numWomen ≤ s.totalEmployees →
  (numWomen < 12 → groupDiscountCost s < womenDiscountCost s numWomen) ∧
  (numWomen = 12 → groupDiscountCost s = womenDiscountCost s numWomen) ∧
  (numWomen > 12 → groupDiscountCost s > womenDiscountCost s numWomen) :=
by sorry

end NUMINAMATH_CALUDE_cost_effective_plan_l3006_300641


namespace NUMINAMATH_CALUDE_no_valid_n_l3006_300620

theorem no_valid_n : ¬∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (y : ℤ), n^2 - 18*n + 80 = y^2) ∧ 
  (∃ (k : ℤ), 15 = n * k) := by
sorry

end NUMINAMATH_CALUDE_no_valid_n_l3006_300620


namespace NUMINAMATH_CALUDE_fraction_addition_l3006_300632

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3006_300632


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3006_300630

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) :
  (1 / a^2) + 2 * a^2 + 3 * b^2 + 4 * a * b ≥ Real.sqrt (8 / 3) := by
  sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a ≠ 0 ∧ (1 / a^2) + 2 * a^2 + 3 * b^2 + 4 * a * b = Real.sqrt (8 / 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3006_300630


namespace NUMINAMATH_CALUDE_tv_show_cost_per_episode_l3006_300627

/-- Given a TV show season with the following properties:
  * The season has 22 episodes
  * The total cost of the season is $35,200
  * The second half of the season costs 120% more per episode than the first half
  Prove that the cost per episode for the first half of the season is $1,000. -/
theorem tv_show_cost_per_episode 
  (total_episodes : ℕ) 
  (total_cost : ℚ) 
  (second_half_increase : ℚ) :
  total_episodes = 22 →
  total_cost = 35200 →
  second_half_increase = 1.2 →
  let first_half_cost := total_cost / (total_episodes / 2 * (1 + 1 + second_half_increase))
  first_half_cost = 1000 := by
sorry

end NUMINAMATH_CALUDE_tv_show_cost_per_episode_l3006_300627


namespace NUMINAMATH_CALUDE_division_438_by_4_result_l3006_300686

/-- Represents the place value of a digit in a number -/
inductive PlaceValue
  | Ones
  | Tens
  | Hundreds
  | Thousands

/-- Represents a division operation with its result -/
structure DivisionResult (dividend : ℕ) (divisor : ℕ) where
  quotient : ℕ
  remainder : ℕ
  highest_place_value : PlaceValue
  valid : dividend = quotient * divisor + remainder
  remainder_bound : remainder < divisor

/-- The division of 438 by 4 -/
def division_438_by_4 : DivisionResult 438 4 := sorry

theorem division_438_by_4_result :
  division_438_by_4.highest_place_value = PlaceValue.Hundreds ∧
  division_438_by_4.remainder = 2 := by sorry

end NUMINAMATH_CALUDE_division_438_by_4_result_l3006_300686


namespace NUMINAMATH_CALUDE_prob_different_given_alone_is_half_l3006_300655

/-- The number of people visiting tourist spots -/
def num_people : ℕ := 3

/-- The number of tourist spots -/
def num_spots : ℕ := 3

/-- The number of ways person A can visit a spot alone -/
def ways_A_alone : ℕ := num_spots * (num_spots - 1) * (num_spots - 1)

/-- The number of ways all three people can visit different spots -/
def ways_all_different : ℕ := num_spots * (num_spots - 1) * (num_spots - 2)

/-- The probability that three people visit different spots given that one person visits a spot alone -/
def prob_different_given_alone : ℚ := ways_all_different / ways_A_alone

theorem prob_different_given_alone_is_half : prob_different_given_alone = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_given_alone_is_half_l3006_300655


namespace NUMINAMATH_CALUDE_polynomial_coefficient_C_l3006_300656

theorem polynomial_coefficient_C (A B C D : ℤ) : 
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ z : ℂ, z^6 - 15*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆)) ∧
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 15)) → 
  C = -92 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_C_l3006_300656


namespace NUMINAMATH_CALUDE_jane_earnings_l3006_300652

/-- Represents the number of flower bulbs planted for each type --/
structure FlowerBulbs where
  tulips : ℕ
  iris : ℕ
  daffodils : ℕ
  crocus : ℕ

/-- Calculates the total earnings from planting flower bulbs --/
def calculate_earnings (bulbs : FlowerBulbs) (price_per_bulb : ℚ) : ℚ :=
  price_per_bulb * (bulbs.tulips + bulbs.iris + bulbs.daffodils + bulbs.crocus)

/-- The main theorem stating Jane's earnings --/
theorem jane_earnings : ∃ (bulbs : FlowerBulbs),
  bulbs.tulips = 20 ∧
  bulbs.iris = bulbs.tulips / 2 ∧
  bulbs.daffodils = 30 ∧
  bulbs.crocus = 3 * bulbs.daffodils ∧
  calculate_earnings bulbs (1/2) = 75 := by
  sorry


end NUMINAMATH_CALUDE_jane_earnings_l3006_300652


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3006_300672

theorem fraction_sum_equality (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 =
  1 / (b - c) + 1 / (c - a) + 1 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3006_300672


namespace NUMINAMATH_CALUDE_angle_30_implies_sqrt3_l3006_300637

/-- Given two vectors a and b in ℝ², prove that if they form an angle of 30°, then the first component of a is √3. -/
theorem angle_30_implies_sqrt3 (a b : ℝ × ℝ) (h : a.1 = m ∧ a.2 = 3 ∧ b.1 = Real.sqrt 3 ∧ b.2 = 1) :
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = Real.sqrt 3 / 2 →
  m = Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_angle_30_implies_sqrt3_l3006_300637


namespace NUMINAMATH_CALUDE_better_performance_criterion_l3006_300658

/-- Represents a shooter's performance statistics -/
structure ShooterStats where
  average_score : ℝ
  standard_deviation : ℝ

/-- Defines when a shooter has better performance than another -/
def better_performance (a b : ShooterStats) : Prop :=
  a.average_score > b.average_score ∧ a.standard_deviation < b.standard_deviation

/-- Theorem stating that a shooter with higher average score and lower standard deviation
    has better performance -/
theorem better_performance_criterion (shooter_a shooter_b : ShooterStats)
  (h1 : shooter_a.average_score > shooter_b.average_score)
  (h2 : shooter_a.standard_deviation < shooter_b.standard_deviation) :
  better_performance shooter_a shooter_b := by
  sorry

end NUMINAMATH_CALUDE_better_performance_criterion_l3006_300658


namespace NUMINAMATH_CALUDE_mary_shirts_fraction_l3006_300640

theorem mary_shirts_fraction (blue_initial : ℕ) (brown_initial : ℕ) (total_left : ℕ) :
  blue_initial = 26 →
  brown_initial = 36 →
  total_left = 37 →
  ∃ (f : ℚ), 
    (blue_initial / 2 + brown_initial * (1 - f) = total_left) ∧
    (f = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_mary_shirts_fraction_l3006_300640


namespace NUMINAMATH_CALUDE_no_natural_solution_l3006_300662

theorem no_natural_solution :
  ¬ ∃ (x y z t : ℕ), 16^x + 21^y + 26^z = t^2 := by
sorry

end NUMINAMATH_CALUDE_no_natural_solution_l3006_300662


namespace NUMINAMATH_CALUDE_min_groups_for_class_l3006_300682

theorem min_groups_for_class (total_students : ℕ) (max_group_size : ℕ) (h1 : total_students = 30) (h2 : max_group_size = 12) :
  ∃ (num_groups : ℕ) (group_size : ℕ),
    num_groups * group_size = total_students ∧
    group_size ≤ max_group_size ∧
    (∀ (other_num_groups : ℕ) (other_group_size : ℕ),
      other_num_groups * other_group_size = total_students →
      other_group_size ≤ max_group_size →
      num_groups ≤ other_num_groups) ∧
    num_groups = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_groups_for_class_l3006_300682


namespace NUMINAMATH_CALUDE_exists_monochromatic_parallelepiped_l3006_300646

-- Define the set A as points in ℤ³
def A : Set (ℤ × ℤ × ℤ) := Set.univ

-- Define a color assignment function
def colorAssignment (p : ℕ) : (ℤ × ℤ × ℤ) → Fin p := sorry

-- Define a rectangular parallelepiped
def isRectangularParallelepiped (vertices : Finset (ℤ × ℤ × ℤ)) : Prop := sorry

-- Theorem statement
theorem exists_monochromatic_parallelepiped (p : ℕ) (hp : p > 0) :
  ∃ (vertices : Finset (ℤ × ℤ × ℤ)),
    vertices.card = 8 ∧
    isRectangularParallelepiped vertices ∧
    ∃ (c : Fin p), ∀ v ∈ vertices, colorAssignment p v = c :=
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_parallelepiped_l3006_300646


namespace NUMINAMATH_CALUDE_exam_students_count_l3006_300670

theorem exam_students_count :
  ∀ (n : ℕ) (T : ℝ),
    n > 0 →
    T = n * 90 →
    T - 120 = (n - 3) * 95 →
    n = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l3006_300670


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l3006_300667

-- Define the function f
def f (a x : ℝ) := |x - a| + |2*x - 2|

-- Theorem 1: Solution set of f(x) > 2 when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f 2 x > 2} = {x : ℝ | x < 2/3 ∨ x > 2} :=
sorry

-- Theorem 2: Range of a when f(x) ≥ 2 for all x
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 2) ↔ (a ≤ -1 ∨ a ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l3006_300667


namespace NUMINAMATH_CALUDE_expand_product_l3006_300604

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3006_300604


namespace NUMINAMATH_CALUDE_smallest_product_smallest_product_is_neg_32_l3006_300611

def S : Finset Int := {-8, -3, -2, 2, 4}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y ≤ a * b :=
by
  sorry

theorem smallest_product_is_neg_32 :
  ∃ (a b : Int), a ∈ S ∧ b ∈ S ∧ a * b = -32 ∧
  ∀ (x y : Int), x ∈ S → y ∈ S → a * b ≤ x * y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_product_smallest_product_is_neg_32_l3006_300611


namespace NUMINAMATH_CALUDE_sunnydale_farm_arrangement_l3006_300687

/-- The number of ways to arrange animals in a row -/
def arrange_animals (chickens dogs cats rabbits : ℕ) : ℕ :=
  Nat.factorial 4 * Nat.factorial chickens * Nat.factorial dogs * Nat.factorial cats * Nat.factorial rabbits

/-- Theorem stating the number of arrangements for the given animal counts -/
theorem sunnydale_farm_arrangement :
  arrange_animals 5 3 4 3 = 2488320 :=
by sorry

end NUMINAMATH_CALUDE_sunnydale_farm_arrangement_l3006_300687


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3006_300653

theorem quadratic_inequality_solution (x : ℝ) :
  (-x^2 + 5*x - 4 < 0) ↔ (1 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3006_300653


namespace NUMINAMATH_CALUDE_steves_gold_bars_l3006_300654

theorem steves_gold_bars (friends : ℕ) (lost_bars : ℕ) (bars_per_friend : ℕ) : 
  friends = 4 → lost_bars = 20 → bars_per_friend = 20 →
  friends * bars_per_friend + lost_bars = 100 := by
  sorry

end NUMINAMATH_CALUDE_steves_gold_bars_l3006_300654


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l3006_300657

theorem quadratic_form_equivalence (b : ℝ) (n : ℝ) :
  b < 0 →
  (∀ x, x^2 + b*x - 36 = (x + n)^2 - 20) →
  b = -8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l3006_300657


namespace NUMINAMATH_CALUDE_shaded_perimeter_value_l3006_300649

/-- The perimeter of the shaded region formed by four quarter-circle arcs in a unit square --/
def shadedPerimeter : ℝ := sorry

/-- The square PQRS with side length 1 --/
def unitSquare : Set (ℝ × ℝ) := sorry

/-- Arc TRU with center P --/
def arcTRU : Set (ℝ × ℝ) := sorry

/-- Arc VPW with center R --/
def arcVPW : Set (ℝ × ℝ) := sorry

/-- Arc UV with center S --/
def arcUV : Set (ℝ × ℝ) := sorry

/-- Arc WT with center Q --/
def arcWT : Set (ℝ × ℝ) := sorry

/-- The theorem stating that the perimeter of the shaded region is (2√2 - 1)π --/
theorem shaded_perimeter_value : shadedPerimeter = (2 * Real.sqrt 2 - 1) * Real.pi := by sorry

end NUMINAMATH_CALUDE_shaded_perimeter_value_l3006_300649


namespace NUMINAMATH_CALUDE_brick_length_is_20_l3006_300635

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 20

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 10

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 7.5

/-- The length of the wall in meters -/
def wall_length : ℝ := 27

/-- The width of the wall in meters -/
def wall_width : ℝ := 2

/-- The height of the wall in meters -/
def wall_height : ℝ := 0.75

/-- The number of bricks required for the wall -/
def num_bricks : ℕ := 27000

/-- Conversion factor from cubic meters to cubic centimeters -/
def m3_to_cm3 : ℝ := 1000000

theorem brick_length_is_20 :
  brick_length = 20 ∧
  brick_width = 10 ∧
  brick_height = 7.5 ∧
  wall_length = 27 ∧
  wall_width = 2 ∧
  wall_height = 0.75 ∧
  num_bricks = 27000 →
  wall_length * wall_width * wall_height * m3_to_cm3 =
    brick_length * brick_width * brick_height * num_bricks :=
by sorry

end NUMINAMATH_CALUDE_brick_length_is_20_l3006_300635


namespace NUMINAMATH_CALUDE_sports_school_sections_l3006_300638

theorem sports_school_sections (total : ℕ) (skiing : ℕ) (speed_skating : ℕ) (hockey : ℕ) : 
  total = 96 ∧ 
  speed_skating = (8 * skiing) / 10 ∧ 
  hockey = (skiing + speed_skating) / 3 ∧ 
  total = skiing + speed_skating + hockey →
  skiing = 40 ∧ speed_skating = 32 ∧ hockey = 24 := by
  sorry

end NUMINAMATH_CALUDE_sports_school_sections_l3006_300638


namespace NUMINAMATH_CALUDE_xiao_bing_winning_probability_l3006_300684

-- Define the game parameters
def dice_outcomes : ℕ := 6 * 6
def same_number_outcomes : ℕ := 6
def xiao_cong_score : ℕ := 10
def xiao_bing_score : ℕ := 2

-- Define the probabilities
def prob_same_numbers : ℚ := same_number_outcomes / dice_outcomes
def prob_different_numbers : ℚ := 1 - prob_same_numbers

-- Define the expected scores
def xiao_cong_expected_score : ℚ := prob_same_numbers * xiao_cong_score
def xiao_bing_expected_score : ℚ := prob_different_numbers * xiao_bing_score

-- Theorem: The probability of Xiao Bing winning is 1/2
theorem xiao_bing_winning_probability : 
  xiao_cong_expected_score = xiao_bing_expected_score → 
  (1 : ℚ) / 2 = prob_different_numbers := by
  sorry

end NUMINAMATH_CALUDE_xiao_bing_winning_probability_l3006_300684


namespace NUMINAMATH_CALUDE_complex_sum_problem_l3006_300690

theorem complex_sum_problem (x y u v w z : ℝ) : 
  y = 5 →
  w = -x - u →
  Complex.I * (x + y + u + v + w + z) = 4 * Complex.I →
  v + z = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l3006_300690


namespace NUMINAMATH_CALUDE_square_eq_necessary_condition_l3006_300648

theorem square_eq_necessary_condition (x h k : ℝ) :
  (x + h)^2 = k → k ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_eq_necessary_condition_l3006_300648


namespace NUMINAMATH_CALUDE_divisibility_by_264_l3006_300678

theorem divisibility_by_264 (n : ℕ) (h : n > 1) :
  ∃ k : ℤ, (7 : ℤ)^(2*n) - (4 : ℤ)^(2*n) - 297 = 264 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_264_l3006_300678


namespace NUMINAMATH_CALUDE_ln_neg_implies_a_less_than_one_a_less_than_one_not_sufficient_for_ln_neg_l3006_300693

theorem ln_neg_implies_a_less_than_one :
  ∀ a : ℝ, Real.log a < 0 → a < 1 :=
sorry

theorem a_less_than_one_not_sufficient_for_ln_neg :
  ∃ a : ℝ, a < 1 ∧ ¬(Real.log a < 0) :=
sorry

end NUMINAMATH_CALUDE_ln_neg_implies_a_less_than_one_a_less_than_one_not_sufficient_for_ln_neg_l3006_300693


namespace NUMINAMATH_CALUDE_new_quad_inscribable_l3006_300600

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define the points on the circle
variable (A₁ A₂ B₁ B₂ C₁ C₂ D₁ D₂ : ℝ × ℝ)

-- Define the convex quadrilateral
variable (quad : Set (ℝ × ℝ))

-- Define the condition that the quadrilateral is inscribed in the circle
variable (quad_inscribed : quad ⊆ circle)

-- Define the condition that the extended sides intersect the circle at the given points
variable (extended_sides : 
  A₁ ∈ circle ∧ A₂ ∈ circle ∧ 
  B₁ ∈ circle ∧ B₂ ∈ circle ∧ 
  C₁ ∈ circle ∧ C₂ ∈ circle ∧ 
  D₁ ∈ circle ∧ D₂ ∈ circle)

-- Define the equality condition
variable (equality_condition : 
  dist A₁ B₂ = dist B₁ C₂ ∧ 
  dist B₁ C₂ = dist C₁ D₂ ∧ 
  dist C₁ D₂ = dist D₁ A₂)

-- Define the quadrilateral formed by the lines A₁A₂, B₁B₂, C₁C₂, D₁D₂
def new_quad : Set (ℝ × ℝ) := sorry

-- The theorem to be proved
theorem new_quad_inscribable :
  ∃ (new_circle : Set (ℝ × ℝ)), new_quad ⊆ new_circle :=
sorry

end NUMINAMATH_CALUDE_new_quad_inscribable_l3006_300600


namespace NUMINAMATH_CALUDE_beaus_sons_correct_number_of_sons_l3006_300696

theorem beaus_sons (sons_age_today : ℕ) (beaus_age_today : ℕ) : ℕ :=
  let sons_age_three_years_ago := sons_age_today - 3
  let beaus_age_three_years_ago := beaus_age_today - 3
  let num_sons := beaus_age_three_years_ago / sons_age_three_years_ago
  num_sons

theorem correct_number_of_sons : beaus_sons 16 42 = 3 := by
  sorry

end NUMINAMATH_CALUDE_beaus_sons_correct_number_of_sons_l3006_300696


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l3006_300608

/-- The number of cakes Baker initially made -/
def initial_cakes : ℕ := 48

/-- The number of cakes Baker sold -/
def sold_cakes : ℕ := 44

/-- Theorem: Baker still has 4 cakes -/
theorem baker_remaining_cakes : initial_cakes - sold_cakes = 4 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l3006_300608


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3006_300694

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*m = 0 ∧ x₂^2 + 2*x₂ + 2*m = 0) → m < (1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3006_300694


namespace NUMINAMATH_CALUDE_pencil_pen_difference_l3006_300669

-- Define the given conditions
def paige_pencils_home : ℕ := 15
def paige_pens_backpack : ℕ := 7

-- Define the theorem
theorem pencil_pen_difference : 
  paige_pencils_home - paige_pens_backpack = 8 := by
  sorry


end NUMINAMATH_CALUDE_pencil_pen_difference_l3006_300669


namespace NUMINAMATH_CALUDE_sunday_reading_time_is_46_l3006_300613

def book_a_assignment : ℕ := 60
def book_b_assignment : ℕ := 45
def friday_book_a : ℕ := 16
def saturday_book_a : ℕ := 28
def saturday_book_b : ℕ := 15

def sunday_reading_time : ℕ := 
  (book_a_assignment - (friday_book_a + saturday_book_a)) + 
  (book_b_assignment - saturday_book_b)

theorem sunday_reading_time_is_46 : sunday_reading_time = 46 := by
  sorry

end NUMINAMATH_CALUDE_sunday_reading_time_is_46_l3006_300613


namespace NUMINAMATH_CALUDE_base_conversion_and_addition_l3006_300661

/-- Converts a number from base 8 to base 10 -/
def base8To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10To7 (n : ℕ) : ℕ := sorry

/-- Adds two numbers in base 7 -/
def addBase7 (a b : ℕ) : ℕ := sorry

theorem base_conversion_and_addition :
  addBase7 (base10To7 (base8To10 123)) 25 = 264 := by sorry

end NUMINAMATH_CALUDE_base_conversion_and_addition_l3006_300661


namespace NUMINAMATH_CALUDE_progression_existence_l3006_300615

theorem progression_existence (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) :
  (∃ q : ℝ, q > 0 ∧ b = a * q ∧ c = b * q) ∧
  ¬(∃ d : ℝ, b = a + d ∧ c = b + d) := by
  sorry

end NUMINAMATH_CALUDE_progression_existence_l3006_300615


namespace NUMINAMATH_CALUDE_calculation_one_l3006_300663

theorem calculation_one : 6.8 - (-4.2) + (-4) * (-3) = 23 := by
  sorry

end NUMINAMATH_CALUDE_calculation_one_l3006_300663


namespace NUMINAMATH_CALUDE_descending_order_original_statement_l3006_300671

theorem descending_order : 0.38 > 0.373 ∧ 0.373 > 0.37 := by
  sorry

-- Define 37% as 0.37
def thirty_seven_percent : ℝ := 0.37

-- Prove that the original statement holds
theorem original_statement : 0.38 > 0.373 ∧ 0.373 > thirty_seven_percent := by
  sorry

end NUMINAMATH_CALUDE_descending_order_original_statement_l3006_300671


namespace NUMINAMATH_CALUDE_green_square_area_percentage_l3006_300692

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  side : ℝ
  crossWidth : ℝ
  crossArea : ℝ
  greenSide : ℝ

/-- The cross is symmetric and occupies 49% of the flag's area -/
def isValidCrossFlag (flag : CrossFlag) : Prop :=
  flag.crossArea = 0.49 * flag.side^2 ∧
  flag.greenSide = 2 * flag.crossWidth

/-- Theorem: The green square occupies 6.01% of the flag's area -/
theorem green_square_area_percentage (flag : CrossFlag) 
  (h : isValidCrossFlag flag) : 
  (flag.greenSide^2) / (flag.side^2) = 0.0601 := by
  sorry

end NUMINAMATH_CALUDE_green_square_area_percentage_l3006_300692


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l3006_300677

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a 2D translation --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola y = 8x² --/
def original_parabola : Parabola := { a := 8, b := 0, c := 0 }

/-- The translation of 3 units left and 5 units down --/
def translation : Translation := { dx := -3, dy := -5 }

/-- Applies a translation to a parabola --/
def apply_translation (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a
    b := -2 * p.a * t.dx
    c := p.a * t.dx^2 + p.b * t.dx + p.c + t.dy }

theorem parabola_translation_theorem :
  apply_translation original_parabola translation = { a := 8, b := 48, c := -5 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l3006_300677


namespace NUMINAMATH_CALUDE_min_group_size_with_94_percent_boys_l3006_300688

theorem min_group_size_with_94_percent_boys (boys girls : ℕ) :
  boys > 0 →
  girls > 0 →
  (boys : ℚ) / (boys + girls : ℚ) > 94 / 100 →
  boys + girls ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_min_group_size_with_94_percent_boys_l3006_300688


namespace NUMINAMATH_CALUDE_age_sum_problem_l3006_300628

theorem age_sum_problem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 256 →
  a + b + c = 38 := by
sorry

end NUMINAMATH_CALUDE_age_sum_problem_l3006_300628


namespace NUMINAMATH_CALUDE_fifth_match_goals_l3006_300664

/-- A football player's goal-scoring record over 5 matches -/
structure FootballRecord where
  total_matches : Nat
  total_goals : Nat
  fifth_match_goals : Nat
  average_increase : Rat

/-- The conditions of the problem -/
def problem_conditions (record : FootballRecord) : Prop :=
  record.total_matches = 5 ∧
  record.total_goals = 8 ∧
  record.average_increase = 1/10 ∧
  (4 * (record.total_goals - record.fifth_match_goals)) / 4 + record.average_increase = 
    record.total_goals / record.total_matches

/-- The theorem stating that under the given conditions, the player scored 2 goals in the fifth match -/
theorem fifth_match_goals (record : FootballRecord) 
  (h : problem_conditions record) : record.fifth_match_goals = 2 := by
  sorry


end NUMINAMATH_CALUDE_fifth_match_goals_l3006_300664


namespace NUMINAMATH_CALUDE_room_width_calculation_l3006_300697

/-- Given a room with length 5.5 m and a floor paving cost of 400 Rs per sq metre
    resulting in a total cost of 8250 Rs, prove that the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) :
  length = 5.5 →
  cost_per_sqm = 400 →
  total_cost = 8250 →
  width = total_cost / cost_per_sqm / length →
  width = 3.75 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l3006_300697


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3006_300626

/-- A regular nonagon is a 9-sided polygon with all sides equal and all angles equal. -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices. -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- Two diagonals intersect if they have a point in common inside the nonagon. -/
def Intersect (n : RegularNonagon) (d1 d2 : Diagonal n) : Prop := sorry

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes. -/
def Probability (event : Prop) : ℚ := sorry

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  Probability (∃ (d1 d2 : Diagonal n), Intersect n d1 d2) = 14 / 39 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3006_300626


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3006_300633

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (α + π / 4) = 1 / 3) : 
  Real.sin (2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3006_300633


namespace NUMINAMATH_CALUDE_number_equation_l3006_300625

theorem number_equation (x : ℚ) (N : ℚ) : x = 9 → (N - 5 / x = 4 + 4 / x ↔ N = 5) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3006_300625


namespace NUMINAMATH_CALUDE_log_eight_x_equals_three_halves_l3006_300642

theorem log_eight_x_equals_three_halves (x : ℝ) :
  Real.log x / Real.log 8 = 3/2 → x = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_eight_x_equals_three_halves_l3006_300642


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_not_square_l3006_300603

theorem product_of_five_consecutive_integers_not_square (n : ℤ) : 
  ∃ (m : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ≠ m ^ 2 := by
sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_not_square_l3006_300603


namespace NUMINAMATH_CALUDE_star_neg_two_three_l3006_300636

-- Define the new operation ※
def star (a b : ℤ) : ℤ := a^2 + 2*a*b

-- Theorem statement
theorem star_neg_two_three : star (-2) 3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_star_neg_two_three_l3006_300636


namespace NUMINAMATH_CALUDE_sin_shift_stretch_l3006_300674

/-- Given a function f(x) = sin(2x), prove that shifting it right by π/12 and
    stretching x-coordinates by a factor of 2 results in g(x) = sin(x - π/6) -/
theorem sin_shift_stretch (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x)
  let shift : ℝ → ℝ := λ x => x - π / 12
  let stretch : ℝ → ℝ := λ x => x / 2
  let g : ℝ → ℝ := λ x => Real.sin (x - π / 6)
  (f ∘ shift ∘ stretch) x = g x :=
by sorry

end NUMINAMATH_CALUDE_sin_shift_stretch_l3006_300674


namespace NUMINAMATH_CALUDE_multiplication_preserves_odd_positives_l3006_300614

def P : Set ℕ := {n : ℕ | n % 2 = 1 ∧ n > 0}

def M : Set ℕ := {x : ℕ | ∃ (a b : ℕ), a ∈ P ∧ b ∈ P ∧ x = a * b}

theorem multiplication_preserves_odd_positives (h : M ⊆ P) :
  ∀ (a b : ℕ), a ∈ P → b ∈ P → (a * b) ∈ P := by
  sorry

end NUMINAMATH_CALUDE_multiplication_preserves_odd_positives_l3006_300614


namespace NUMINAMATH_CALUDE_red_bottles_count_l3006_300623

/-- The number of red water bottles in the fridge -/
def red_bottles : ℕ := 2

/-- The number of black water bottles in the fridge -/
def black_bottles : ℕ := 3

/-- The number of blue water bottles in the fridge -/
def blue_bottles : ℕ := 4

/-- The total number of water bottles initially in the fridge -/
def total_bottles : ℕ := 9

theorem red_bottles_count : red_bottles + black_bottles + blue_bottles = total_bottles := by
  sorry

end NUMINAMATH_CALUDE_red_bottles_count_l3006_300623


namespace NUMINAMATH_CALUDE_soda_price_after_increase_l3006_300609

theorem soda_price_after_increase (candy_price_new : ℝ) (candy_increase : ℝ) (soda_increase : ℝ) (combined_price_old : ℝ) 
  (h1 : candy_price_new = 15)
  (h2 : candy_increase = 0.25)
  (h3 : soda_increase = 0.5)
  (h4 : combined_price_old = 16) :
  ∃ (soda_price_new : ℝ), soda_price_new = 6 := by
  sorry

#check soda_price_after_increase

end NUMINAMATH_CALUDE_soda_price_after_increase_l3006_300609


namespace NUMINAMATH_CALUDE_point_c_in_second_quadrant_l3006_300647

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Given points -/
def pointA : Point := ⟨5, 3⟩
def pointB : Point := ⟨5, -3⟩
def pointC : Point := ⟨-5, 3⟩
def pointD : Point := ⟨-5, -3⟩

/-- Theorem: Point C is the only point in the second quadrant -/
theorem point_c_in_second_quadrant :
  isInSecondQuadrant pointC ∧
  ¬isInSecondQuadrant pointA ∧
  ¬isInSecondQuadrant pointB ∧
  ¬isInSecondQuadrant pointD :=
by sorry

end NUMINAMATH_CALUDE_point_c_in_second_quadrant_l3006_300647


namespace NUMINAMATH_CALUDE_beef_stew_duration_l3006_300624

/-- The number of days the beef stew lasts for 2 people -/
def days_for_two : ℝ := 7

/-- The number of days the beef stew lasts for 5 people -/
def days_for_five : ℝ := 2.8

/-- The number of people in the original scenario -/
def original_people : ℕ := 2

/-- The number of people in the new scenario -/
def new_people : ℕ := 5

theorem beef_stew_duration :
  days_for_two * original_people = days_for_five * new_people :=
by sorry

end NUMINAMATH_CALUDE_beef_stew_duration_l3006_300624


namespace NUMINAMATH_CALUDE_soccer_league_games_l3006_300666

theorem soccer_league_games (n : ℕ) (h : n = 12) : 
  (n * (n - 1)) / 2 = 66 := by
  sorry

#check soccer_league_games

end NUMINAMATH_CALUDE_soccer_league_games_l3006_300666


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l3006_300679

/-- Reflects a point (x, y) across the line y = -x -/
def reflect_across_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The original center of the circle -/
def original_center : ℝ × ℝ := (3, -7)

/-- The expected center after reflection -/
def expected_reflected_center : ℝ × ℝ := (7, -3)

theorem reflection_of_circle_center :
  reflect_across_y_neg_x original_center = expected_reflected_center :=
by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l3006_300679


namespace NUMINAMATH_CALUDE_bible_yellow_tickets_l3006_300621

-- Define the conversion rates
def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10

-- Define Tom's current tickets
def tom_yellow : ℕ := 8
def tom_red : ℕ := 3
def tom_blue : ℕ := 7

-- Define the additional blue tickets needed
def additional_blue : ℕ := 163

-- Define the function to calculate the total blue tickets equivalent
def total_blue_equivalent (yellow red blue : ℕ) : ℕ :=
  yellow * yellow_to_red * red_to_blue + red * red_to_blue + blue

-- Theorem statement
theorem bible_yellow_tickets :
  ∃ (required_yellow : ℕ),
    required_yellow = 10 ∧
    total_blue_equivalent tom_yellow tom_red tom_blue + additional_blue =
    required_yellow * yellow_to_red * red_to_blue :=
by sorry

end NUMINAMATH_CALUDE_bible_yellow_tickets_l3006_300621


namespace NUMINAMATH_CALUDE_special_polyhedron_sum_l3006_300691

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  t : ℕ  -- number of triangular faces
  h : ℕ  -- number of hexagonal faces
  T : ℕ  -- number of triangular faces meeting at each vertex
  H : ℕ  -- number of hexagonal faces meeting at each vertex

/-- The theorem about the special polyhedron -/
theorem special_polyhedron_sum (p : SpecialPolyhedron) : 
  p.F = 30 ∧ 
  p.F = p.t + p.h ∧
  p.T = 3 ∧ 
  p.H = 2 ∧
  p.V - p.E + p.F = 2 ∧ 
  p.E = (3 * p.t + 6 * p.h) / 2 →
  100 * p.H + 10 * p.T + p.V = 262 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_sum_l3006_300691


namespace NUMINAMATH_CALUDE_harry_monday_speed_l3006_300689

/-- Harry's marathon running speeds throughout the week -/
def harry_speeds (monday_speed : ℝ) : Fin 5 → ℝ
  | 0 => monday_speed  -- Monday
  | 1 => 1.5 * monday_speed  -- Tuesday
  | 2 => 1.5 * monday_speed  -- Wednesday
  | 3 => 1.5 * monday_speed  -- Thursday
  | 4 => 1.6 * 1.5 * monday_speed  -- Friday

theorem harry_monday_speed :
  ∃ (monday_speed : ℝ), 
    (harry_speeds monday_speed 4 = 24) ∧ 
    (monday_speed = 10) := by
  sorry

end NUMINAMATH_CALUDE_harry_monday_speed_l3006_300689


namespace NUMINAMATH_CALUDE_number_of_routes_P_to_Q_l3006_300643

/-- Represents the points in the diagram --/
inductive Point : Type
| P | Q | R | S | T

/-- Represents a direct path between two points --/
def DirectPath : Point → Point → Prop :=
  fun p q => match p, q with
  | Point.P, Point.R => True
  | Point.P, Point.S => True
  | Point.R, Point.T => True
  | Point.R, Point.Q => True
  | Point.S, Point.T => True
  | Point.T, Point.Q => True
  | _, _ => False

/-- Represents a route from one point to another --/
def Route : Point → Point → Type :=
  fun p q => List (Σ' x y : Point, DirectPath x y)

/-- Counts the number of routes between two points --/
def countRoutes : Point → Point → Nat :=
  fun p q => sorry

theorem number_of_routes_P_to_Q :
  countRoutes Point.P Point.Q = 3 := by sorry

end NUMINAMATH_CALUDE_number_of_routes_P_to_Q_l3006_300643


namespace NUMINAMATH_CALUDE_smallest_bench_configuration_l3006_300612

theorem smallest_bench_configuration (adults_per_bench children_per_bench : ℕ) 
  (adults_per_bench_pos : adults_per_bench > 0)
  (children_per_bench_pos : children_per_bench > 0)
  (adults_per_bench_def : adults_per_bench = 9)
  (children_per_bench_def : children_per_bench = 15) :
  ∃ (M : ℕ), M > 0 ∧ M * adults_per_bench = M * children_per_bench ∧
  ∀ (N : ℕ), N > 0 → N * adults_per_bench = N * children_per_bench → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_smallest_bench_configuration_l3006_300612


namespace NUMINAMATH_CALUDE_interest_rate_for_doubling_l3006_300685

/-- The time in years for the money to double --/
def doubling_time : ℝ := 4

/-- The interest rate as a decimal --/
def interest_rate : ℝ := 0.25

/-- Simple interest formula: Final amount = Principal * (1 + rate * time) --/
def simple_interest (principal rate time : ℝ) : ℝ := principal * (1 + rate * time)

theorem interest_rate_for_doubling :
  simple_interest 1 interest_rate doubling_time = 2 := by sorry

end NUMINAMATH_CALUDE_interest_rate_for_doubling_l3006_300685


namespace NUMINAMATH_CALUDE_puzzle_pieces_left_l3006_300644

theorem puzzle_pieces_left (total_pieces : ℕ) (num_children : ℕ) (reyn_pieces : ℕ) : 
  total_pieces = 500 →
  num_children = 4 →
  reyn_pieces = 25 →
  total_pieces - (reyn_pieces + 2*reyn_pieces + 3*reyn_pieces + 4*reyn_pieces) = 250 := by
sorry

end NUMINAMATH_CALUDE_puzzle_pieces_left_l3006_300644


namespace NUMINAMATH_CALUDE_unique_b_solution_l3006_300629

def base_83_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (83 ^ i)) 0

theorem unique_b_solution : ∃! b : ℤ, 
  (0 ≤ b ∧ b ≤ 20) ∧ 
  (∃ k : ℤ, base_83_to_decimal [2, 5, 7, 3, 6, 4, 5] - b = 17 * k) ∧
  b = 8 := by sorry

end NUMINAMATH_CALUDE_unique_b_solution_l3006_300629


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3006_300698

def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (1, 1)

theorem vector_perpendicular : (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3006_300698


namespace NUMINAMATH_CALUDE_percentage_problem_l3006_300650

theorem percentage_problem (x : ℝ) : (27 / x = 45 / 100) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3006_300650


namespace NUMINAMATH_CALUDE_omar_coffee_cup_size_l3006_300660

/-- Represents the size of Omar's coffee cup in ounces -/
def coffee_cup_size : ℝ := 6

theorem omar_coffee_cup_size :
  ∀ (remaining_after_work : ℝ) (remaining_after_office : ℝ),
  remaining_after_work = coffee_cup_size - (1/4 * coffee_cup_size + 1/2 * coffee_cup_size) →
  remaining_after_office = remaining_after_work - 1 →
  remaining_after_office = 2 →
  coffee_cup_size = 6 := by
sorry

end NUMINAMATH_CALUDE_omar_coffee_cup_size_l3006_300660


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3006_300619

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.sin (10 * π / 180) - 
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3006_300619


namespace NUMINAMATH_CALUDE_product_b_original_price_l3006_300610

theorem product_b_original_price 
  (price_a : ℝ) 
  (price_b : ℝ) 
  (initial_relation : price_a = 1.2 * price_b)
  (price_a_after : ℝ)
  (price_a_decrease : price_a_after = 0.9 * price_a)
  (price_a_final : price_a_after = 198)
  : price_b = 183.33 := by
  sorry

end NUMINAMATH_CALUDE_product_b_original_price_l3006_300610


namespace NUMINAMATH_CALUDE_sand_amount_l3006_300668

/-- The amount of gravel bought by the company in tons -/
def gravel : ℝ := 5.91

/-- The total amount of material bought by the company in tons -/
def total_material : ℝ := 14.02

/-- The amount of sand bought by the company in tons -/
def sand : ℝ := total_material - gravel

theorem sand_amount : sand = 8.11 := by
  sorry

end NUMINAMATH_CALUDE_sand_amount_l3006_300668


namespace NUMINAMATH_CALUDE_yun_lost_paperclips_l3006_300616

theorem yun_lost_paperclips : ∀ (yun_current : ℕ),
  yun_current ≤ 20 →
  (1 + 1/4 : ℚ) * yun_current + 7 = 9 →
  20 - yun_current = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_yun_lost_paperclips_l3006_300616


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3006_300617

/-- Given two vectors a and b in ℝ², prove that |a - b| = 5 -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  a = (-2, 1) →
  a + b = (-1, -2) →
  ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3006_300617


namespace NUMINAMATH_CALUDE_expression_evaluation_l3006_300645

theorem expression_evaluation : 
  2 * (7 ^ (1/3 : ℝ)) + 16 ^ (3/4 : ℝ) + (4 / (Real.sqrt 3 - 1)) ^ (0 : ℝ) + (-3) ^ (-1 : ℝ) = 44/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3006_300645


namespace NUMINAMATH_CALUDE_average_rounds_played_l3006_300680

/-- Represents the distribution of golf rounds played by members -/
def golf_distribution : List (Nat × Nat) := [(1, 3), (2, 4), (3, 6), (4, 3), (5, 2)]

/-- Calculates the total number of rounds played -/
def total_rounds (dist : List (Nat × Nat)) : Nat :=
  dist.foldr (fun p acc => p.1 * p.2 + acc) 0

/-- Calculates the total number of golfers -/
def total_golfers (dist : List (Nat × Nat)) : Nat :=
  dist.foldr (fun p acc => p.2 + acc) 0

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : Rat) : Int :=
  if x - x.floor < 1/2 then x.floor else x.ceil

theorem average_rounds_played : 
  round_to_nearest ((total_rounds golf_distribution : Rat) / total_golfers golf_distribution) = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_rounds_played_l3006_300680


namespace NUMINAMATH_CALUDE_inequality_proof_l3006_300602

theorem inequality_proof (a b t : ℝ) (h1 : 0 < t) (h2 : t < 1) (h3 : a * b > 0) :
  (a^2 / t^3) + (b^2 / (1 - t^3)) ≥ (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3006_300602


namespace NUMINAMATH_CALUDE_garden_outer_radius_l3006_300683

/-- Given a circular park with a central fountain and a surrounding garden ring,
    this theorem proves the radius of the garden's outer boundary. -/
theorem garden_outer_radius (fountain_diameter : ℝ) (garden_width : ℝ) :
  fountain_diameter = 12 →
  garden_width = 10 →
  fountain_diameter / 2 + garden_width = 16 := by
  sorry

end NUMINAMATH_CALUDE_garden_outer_radius_l3006_300683


namespace NUMINAMATH_CALUDE_consecutive_integers_product_mod_three_l3006_300673

theorem consecutive_integers_product_mod_three (n : ℤ) : 
  (n * (n + 1) / 2) % 3 = 0 ∨ (n * (n + 1) / 2) % 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_mod_three_l3006_300673


namespace NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l3006_300675

theorem no_natural_numbers_satisfying_condition : 
  ¬∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y - (x + y) = 2021 :=
by sorry

end NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l3006_300675


namespace NUMINAMATH_CALUDE_min_n_plus_d_for_arithmetic_sequence_l3006_300676

/-- An arithmetic sequence with positive integral terms -/
def ArithmeticSequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem min_n_plus_d_for_arithmetic_sequence :
  ∀ a : ℕ → ℕ,
  ∀ d : ℕ,
  ArithmeticSequence a d →
  a 1 = 1 →
  (∃ n : ℕ, a n = 51) →
  (∃ n d : ℕ, ArithmeticSequence a d ∧ a 1 = 1 ∧ a n = 51 ∧ n + d = 16 ∧
    ∀ m k : ℕ, ArithmeticSequence a k ∧ a 1 = 1 ∧ a m = 51 → m + k ≥ 16) :=
by sorry

end NUMINAMATH_CALUDE_min_n_plus_d_for_arithmetic_sequence_l3006_300676


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3006_300631

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : (Real.sin α) / (Real.tan α) > 0) 
  (h2 : (Real.tan α) / (Real.cos α) < 0) : 
  0 < α ∧ α < π / 2 ∧ Real.sin α < 0 ∧ Real.cos α > 0 :=
sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3006_300631


namespace NUMINAMATH_CALUDE_unoccupied_chair_fraction_l3006_300606

theorem unoccupied_chair_fraction :
  let total_chairs : ℕ := 40
  let chair_capacity : ℕ := 2
  let total_members : ℕ := total_chairs * chair_capacity
  let attending_members : ℕ := 48
  let unoccupied_chairs : ℕ := (total_members - attending_members) / chair_capacity
  (unoccupied_chairs : ℚ) / total_chairs = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_unoccupied_chair_fraction_l3006_300606


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l3006_300605

theorem remaining_cooking_time (recommended_time_minutes : ℕ) (cooked_time_seconds : ℕ) : 
  recommended_time_minutes = 5 → cooked_time_seconds = 45 → 
  recommended_time_minutes * 60 - cooked_time_seconds = 255 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l3006_300605


namespace NUMINAMATH_CALUDE_intersection_M_N_l3006_300601

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3006_300601


namespace NUMINAMATH_CALUDE_solve_business_partnership_l3006_300651

/-- Represents the problem of determining when Hari joined Praveen's business --/
def business_partnership_problem (praveen_investment : ℕ) (hari_investment : ℕ) (profit_ratio_praveen : ℕ) (profit_ratio_hari : ℕ) (total_months : ℕ) : Prop :=
  ∃ (x : ℕ), 
    x ≤ total_months ∧
    (praveen_investment * total_months) * profit_ratio_hari = 
    (hari_investment * (total_months - x)) * profit_ratio_praveen

/-- Theorem stating the solution to the business partnership problem --/
theorem solve_business_partnership : 
  business_partnership_problem 3360 8640 2 3 12 → 
  ∃ (x : ℕ), x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_business_partnership_l3006_300651


namespace NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l3006_300639

theorem no_bounded_function_satisfying_inequality :
  ¬∃ (f : ℝ → ℝ), 
    (∃ (M : ℝ), ∀ x, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y, (f (x + y))^2 ≥ (f x)^2 + 2*(f (x*y)) + (f y)^2) := by
  sorry

end NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l3006_300639


namespace NUMINAMATH_CALUDE_power_three_1234_mod_5_l3006_300665

theorem power_three_1234_mod_5 : 3^1234 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_three_1234_mod_5_l3006_300665
