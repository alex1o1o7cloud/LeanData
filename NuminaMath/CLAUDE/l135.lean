import Mathlib

namespace NUMINAMATH_CALUDE_negative_integers_abs_leq_four_l135_13518

theorem negative_integers_abs_leq_four :
  {x : ℤ | x < 0 ∧ |x| ≤ 4} = {-1, -2, -3, -4} := by sorry

end NUMINAMATH_CALUDE_negative_integers_abs_leq_four_l135_13518


namespace NUMINAMATH_CALUDE_solve_problem_l135_13579

def problem (basketballs soccer_balls volleyballs : ℕ) : Prop :=
  (soccer_balls = basketballs + 23) ∧
  (volleyballs + 18 = soccer_balls) ∧
  (volleyballs = 40)

theorem solve_problem :
  ∃ (basketballs soccer_balls volleyballs : ℕ),
    problem basketballs soccer_balls volleyballs ∧ basketballs = 35 := by
  sorry

end NUMINAMATH_CALUDE_solve_problem_l135_13579


namespace NUMINAMATH_CALUDE_cd_length_l135_13500

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (N : Point)
  (M : Point)

/-- Represents the problem setup -/
structure ProblemSetup :=
  (ABNM : Quadrilateral)
  (C : Point)
  (A' : Point)
  (D : Point)
  (x : ℝ)
  (AB : ℝ)
  (AM : ℝ)
  (AC : ℝ)

/-- Main theorem: CD = AC * cos(x) -/
theorem cd_length
  (setup : ProblemSetup)
  (h1 : setup.ABNM.A.y = setup.ABNM.B.y) -- AB is initially horizontal
  (h2 : setup.ABNM.N.y = setup.ABNM.M.y) -- MN is horizontal
  (h3 : setup.C.y = setup.ABNM.M.y) -- C is on line MN
  (h4 : setup.A'.x - setup.ABNM.B.x = setup.AB * Real.cos setup.x) -- A' position after rotation
  (h5 : setup.A'.y - setup.ABNM.B.y = setup.AB * Real.sin setup.x)
  (h6 : Real.sqrt ((setup.A'.x - setup.ABNM.B.x)^2 + (setup.A'.y - setup.ABNM.B.y)^2) = setup.AB) -- A'B = AB
  : Real.sqrt ((setup.D.x - setup.C.x)^2 + (setup.D.y - setup.C.y)^2) = setup.AC * Real.cos setup.x :=
sorry

end NUMINAMATH_CALUDE_cd_length_l135_13500


namespace NUMINAMATH_CALUDE_function_intersects_x_axis_l135_13543

theorem function_intersects_x_axis (a : ℝ) : 
  (∀ m : ℝ, ∃ x : ℝ, m * x^2 + x - m - a = 0) → 
  a ∈ Set.Icc (-1 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_function_intersects_x_axis_l135_13543


namespace NUMINAMATH_CALUDE_rectangle_length_l135_13528

theorem rectangle_length (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 4 * width →
  area = length * width →
  area = 100 →
  length = 20 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_l135_13528


namespace NUMINAMATH_CALUDE_min_value_product_quotient_equality_condition_l135_13508

theorem min_value_product_quotient (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2*a + 1) * (b^2 + 2*b + 1) * (c^2 + 2*c + 1) / (a * b * c) ≥ 64 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2*a + 1) * (b^2 + 2*b + 1) * (c^2 + 2*c + 1) / (a * b * c) = 64 ↔ a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_equality_condition_l135_13508


namespace NUMINAMATH_CALUDE_expression_is_integer_l135_13509

theorem expression_is_integer (m : ℕ+) : ∃ k : ℤ, (m^4 / 24 : ℚ) + (m^3 / 4 : ℚ) + (11 * m^2 / 24 : ℚ) + (m / 4 : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_expression_is_integer_l135_13509


namespace NUMINAMATH_CALUDE_stream_speed_l135_13553

/-- Represents the speed of a boat in a stream -/
structure BoatSpeed where
  boatStillWater : ℝ  -- Speed of the boat in still water
  stream : ℝ          -- Speed of the stream

/-- Calculates the effective speed of the boat -/
def effectiveSpeed (b : BoatSpeed) (downstream : Bool) : ℝ :=
  if downstream then b.boatStillWater + b.stream else b.boatStillWater - b.stream

/-- Theorem: Given the conditions, the speed of the stream is 3 km/h -/
theorem stream_speed (b : BoatSpeed) 
  (h1 : effectiveSpeed b true * 4 = 84)  -- Downstream condition
  (h2 : effectiveSpeed b false * 4 = 60) -- Upstream condition
  : b.stream = 3 := by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l135_13553


namespace NUMINAMATH_CALUDE_joes_steakhouse_wage_difference_l135_13554

/-- Proves the wage difference between a manager and a chef at Joe's Steakhouse -/
theorem joes_steakhouse_wage_difference :
  let manager_wage : ℚ := 85/10
  let dishwasher_wage : ℚ := manager_wage / 2
  let chef_wage : ℚ := dishwasher_wage * (1 + 1/4)
  manager_wage - chef_wage = 3187/1000 := by
  sorry

end NUMINAMATH_CALUDE_joes_steakhouse_wage_difference_l135_13554


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_triangle_equality_theorem_l135_13583

/-- A triangle with sides a, b, c and area S -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  area_positive : 0 < S

/-- The inequality holds for all triangles -/
theorem triangle_inequality_theorem (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * t.S * Real.sqrt 3 :=
sorry

/-- The equality holds if and only if the triangle is equilateral -/
theorem triangle_equality_theorem (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 = 4 * t.S * Real.sqrt 3 ↔ t.a = t.b ∧ t.b = t.c :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_triangle_equality_theorem_l135_13583


namespace NUMINAMATH_CALUDE_mulch_cost_calculation_l135_13520

/-- The cost of mulch in dollars per cubic foot -/
def mulch_cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The volume of mulch in cubic yards -/
def mulch_volume_cubic_yards : ℝ := 7

/-- The cost of mulch for a given volume in cubic yards -/
def mulch_cost (volume_cubic_yards : ℝ) : ℝ :=
  volume_cubic_yards * cubic_feet_per_cubic_yard * mulch_cost_per_cubic_foot

theorem mulch_cost_calculation :
  mulch_cost mulch_volume_cubic_yards = 1512 := by
  sorry

end NUMINAMATH_CALUDE_mulch_cost_calculation_l135_13520


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l135_13534

theorem min_value_exponential_sum (x y : ℝ) (h : x + y = 5) :
  3^x + 3^y ≥ 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l135_13534


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l135_13555

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def constant_ratio (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, a n ≠ 0 → a (2 * n) ≠ 0 → a n / a (2 * n) = k

theorem arithmetic_sequence_constant_ratio 
  (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : constant_ratio a) :
  ∃ k : ℝ, (k = 1 ∨ k = 1/2) ∧ ∀ n : ℕ, a n ≠ 0 → a (2 * n) ≠ 0 → a n / a (2 * n) = k :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l135_13555


namespace NUMINAMATH_CALUDE_multiply_106_94_l135_13571

theorem multiply_106_94 : 106 * 94 = 9964 := by
  sorry

end NUMINAMATH_CALUDE_multiply_106_94_l135_13571


namespace NUMINAMATH_CALUDE_measure_11_grams_l135_13540

/-- Represents the number of ways to measure a weight using given weights -/
def measure_ways (one_gram : ℕ) (two_gram : ℕ) (four_gram : ℕ) (target : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 4 ways to measure 11 grams
    given three 1-gram weights, four 2-gram weights, and two 4-gram weights -/
theorem measure_11_grams :
  measure_ways 3 4 2 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_measure_11_grams_l135_13540


namespace NUMINAMATH_CALUDE_interior_angles_sum_l135_13551

theorem interior_angles_sum (n : ℕ) (h : 180 * (n - 2) = 2340) :
  180 * ((n + 3) - 2) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l135_13551


namespace NUMINAMATH_CALUDE_truck_distance_l135_13574

/-- Prove that a truck traveling b/4 feet every t seconds will cover 20b/t yards in 4 minutes -/
theorem truck_distance (b t : ℝ) (h1 : t > 0) : 
  let feet_per_t_seconds := b / 4
  let seconds_in_4_minutes := 4 * 60
  let feet_in_yard := 3
  let yards_in_4_minutes := (feet_per_t_seconds * seconds_in_4_minutes / t) / feet_in_yard
  yards_in_4_minutes = 20 * b / t :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_l135_13574


namespace NUMINAMATH_CALUDE_hyperbola_center_l135_13539

/-- The center of the hyperbola given by the equation (4x+8)^2/16 - (5y-5)^2/25 = 1 is (-2, 1) -/
theorem hyperbola_center : ∃ (h k : ℝ), 
  (∀ x y : ℝ, (4*x + 8)^2 / 16 - (5*y - 5)^2 / 25 = 1 ↔ 
    (x - h)^2 - (y - k)^2 = 1) ∧ 
  h = -2 ∧ k = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l135_13539


namespace NUMINAMATH_CALUDE_seven_digit_nondecreasing_integers_l135_13530

theorem seven_digit_nondecreasing_integers (n : ℕ) (h : n = 7) :
  (Nat.choose (10 + n - 1) n) % 1000 = 440 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_nondecreasing_integers_l135_13530


namespace NUMINAMATH_CALUDE_four_integers_problem_l135_13547

theorem four_integers_problem (x y z u n : ℤ) :
  x + y + z + u = 36 →
  x + n = y - n ∧ y - n = z * n ∧ z * n = u / n →
  n = 1 ∧ x = 8 ∧ y = 10 ∧ z = 9 ∧ u = 9 :=
by sorry

end NUMINAMATH_CALUDE_four_integers_problem_l135_13547


namespace NUMINAMATH_CALUDE_job_completion_time_l135_13513

/-- The time it takes for Annie to complete the job alone -/
def annie_time : ℝ := 10

/-- The time the person works before Annie takes over -/
def person_work_time : ℝ := 3

/-- The time it takes Annie to complete the remaining work after the person stops -/
def annie_remaining_time : ℝ := 8

/-- The time it takes for the person to complete the job alone -/
def person_total_time : ℝ := 15

theorem job_completion_time :
  (person_work_time / person_total_time) + (annie_remaining_time / annie_time) = 1 :=
sorry

end NUMINAMATH_CALUDE_job_completion_time_l135_13513


namespace NUMINAMATH_CALUDE_hyperbola_properties_l135_13503

-- Define the hyperbola
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passing_point : ℝ × ℝ

-- Define the point M
structure PointM where
  on_right_branch : Bool
  dot_product_zero : Bool

-- Theorem statement
theorem hyperbola_properties (h : Hyperbola) (m : PointM) 
    (h_center : h.center = (0, 0))
    (h_foci : h.foci_on_x_axis = true)
    (h_eccentricity : h.eccentricity = Real.sqrt 2)
    (h_passing_point : h.passing_point = (4, -2 * Real.sqrt 2))
    (h_m_right : m.on_right_branch = true)
    (h_m_dot : m.dot_product_zero = true) :
    (∃ (x y : ℝ), x^2 - y^2 = 8) ∧ 
    (∃ (area : ℝ), area = 8) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l135_13503


namespace NUMINAMATH_CALUDE_gathering_dancers_l135_13585

theorem gathering_dancers (men : ℕ) (women : ℕ) : 
  men = 15 →
  men * 4 = women * 3 →
  women = 20 := by
sorry

end NUMINAMATH_CALUDE_gathering_dancers_l135_13585


namespace NUMINAMATH_CALUDE_volunteer_selection_probabilities_l135_13549

/-- Represents the number of calligraphy competition winners -/
def calligraphy_winners : ℕ := 4

/-- Represents the number of painting competition winners -/
def painting_winners : ℕ := 2

/-- Represents the total number of winners -/
def total_winners : ℕ := calligraphy_winners + painting_winners

/-- Represents the number of volunteers to be selected -/
def volunteers_needed : ℕ := 2

/-- The probability of selecting both volunteers from calligraphy winners -/
def prob_both_calligraphy : ℚ := 2 / 5

/-- The probability of selecting one volunteer from each competition -/
def prob_one_each : ℚ := 8 / 15

theorem volunteer_selection_probabilities :
  (Nat.choose calligraphy_winners volunteers_needed) / (Nat.choose total_winners volunteers_needed) = prob_both_calligraphy ∧
  (calligraphy_winners * painting_winners) / (Nat.choose total_winners volunteers_needed) = prob_one_each :=
sorry

end NUMINAMATH_CALUDE_volunteer_selection_probabilities_l135_13549


namespace NUMINAMATH_CALUDE_votes_for_candidate_D_l135_13511

def total_votes : ℕ := 1000000
def invalid_percentage : ℚ := 25 / 100
def candidate_A_percentage : ℚ := 45 / 100
def candidate_B_percentage : ℚ := 30 / 100
def candidate_C_percentage : ℚ := 20 / 100
def candidate_D_percentage : ℚ := 5 / 100

theorem votes_for_candidate_D :
  (total_votes : ℚ) * (1 - invalid_percentage) * candidate_D_percentage = 37500 := by
  sorry

end NUMINAMATH_CALUDE_votes_for_candidate_D_l135_13511


namespace NUMINAMATH_CALUDE_four_digit_with_four_or_five_l135_13575

/-- The number of four-digit positive integers -/
def total_four_digit : ℕ := 9000

/-- The number of four-digit positive integers without 4 or 5 -/
def without_four_or_five : ℕ := 3584

/-- The number of four-digit positive integers with at least one 4 or 5 -/
def with_four_or_five : ℕ := total_four_digit - without_four_or_five

theorem four_digit_with_four_or_five : with_four_or_five = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_with_four_or_five_l135_13575


namespace NUMINAMATH_CALUDE_max_guaranteed_profit_l135_13537

/-- Represents the number of balls -/
def n : ℕ := 10

/-- Represents the cost of a test and the price of a non-radioactive ball -/
def cost : ℕ := 1

/-- Represents the triangular number function -/
def H (k : ℕ) : ℕ := k * (k + 1) / 2

/-- Theorem stating the maximum guaranteed profit for n balls -/
theorem max_guaranteed_profit :
  ∃ k : ℕ, H k < n ∧ n ≤ H (k + 1) ∧ n - (k + 1) = 5 :=
sorry

end NUMINAMATH_CALUDE_max_guaranteed_profit_l135_13537


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l135_13532

theorem largest_x_sqrt_3x_eq_5x : 
  ∃ (x_max : ℚ), x_max = 3/25 ∧ 
  (∀ x : ℚ, x ≥ 0 → Real.sqrt (3*x) = 5*x → x ≤ x_max) ∧
  Real.sqrt (3*x_max) = 5*x_max := by
sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l135_13532


namespace NUMINAMATH_CALUDE_chick_hit_at_least_five_l135_13531

/-- Represents the number of times each toy was hit -/
structure ToyHits where
  chick : ℕ
  monkey : ℕ
  dog : ℕ

/-- Calculates the total score based on the number of hits for each toy -/
def calculateScore (hits : ToyHits) : ℕ :=
  9 * hits.chick + 5 * hits.monkey + 2 * hits.dog

/-- Checks if the given hits satisfy all conditions of the game -/
def isValidGame (hits : ToyHits) : Prop :=
  hits.chick > 0 ∧ hits.monkey > 0 ∧ hits.dog > 0 ∧
  hits.chick + hits.monkey + hits.dog = 10 ∧
  calculateScore hits = 61

/-- The minimum number of times the chick was hit in a valid game -/
def minChickHits : ℕ := 5

theorem chick_hit_at_least_five :
  ∀ hits : ToyHits, isValidGame hits → hits.chick ≥ minChickHits := by
  sorry

end NUMINAMATH_CALUDE_chick_hit_at_least_five_l135_13531


namespace NUMINAMATH_CALUDE_profit_maximized_at_12_marginal_profit_decreasing_l135_13516

-- Define the revenue and cost functions
def R (x : ℕ) : ℝ := 3700 * x + 45 * x^2 - 10 * x^3
def C (x : ℕ) : ℝ := 460 * x + 5000

-- Define the profit function
def P (x : ℕ) : ℝ := R x - C x

-- Define the marginal function
def M (f : ℕ → ℝ) (x : ℕ) : ℝ := f (x + 1) - f x

-- Define the marginal profit function
def MP (x : ℕ) : ℝ := M P x

-- Theorem: Profit is maximized when 12 ships are built
theorem profit_maximized_at_12 :
  ∀ x : ℕ, 1 ≤ x → x ≤ 20 → P 12 ≥ P x :=
sorry

-- Theorem: Marginal profit function is decreasing on [1, 19]
theorem marginal_profit_decreasing :
  ∀ x y : ℕ, 1 ≤ x → x < y → y ≤ 19 → MP y < MP x :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_12_marginal_profit_decreasing_l135_13516


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l135_13538

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 39) →
  (a 3 + a 6 + a 9 = 33) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l135_13538


namespace NUMINAMATH_CALUDE_negation_of_exponential_inequality_l135_13567

theorem negation_of_exponential_inequality :
  (¬ ∀ a : ℝ, a > 0 → Real.exp a ≥ 1) ↔ (∃ a : ℝ, a > 0 ∧ Real.exp a < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exponential_inequality_l135_13567


namespace NUMINAMATH_CALUDE_average_w_x_is_half_l135_13580

theorem average_w_x_is_half 
  (w x y : ℝ) 
  (h1 : 5 / w + 5 / x = 5 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_average_w_x_is_half_l135_13580


namespace NUMINAMATH_CALUDE_dinner_cakes_count_l135_13545

def lunch_cakes : ℕ := 6
def dinner_difference : ℕ := 3

theorem dinner_cakes_count : lunch_cakes + dinner_difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cakes_count_l135_13545


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l135_13557

/-- A rectangular prism with length 4, width 3, and height 2 -/
structure RectangularPrism where
  length : ℕ := 4
  width : ℕ := 3
  height : ℕ := 2

/-- The number of diagonals in a rectangular prism -/
def num_diagonals (prism : RectangularPrism) : ℕ := sorry

/-- Theorem stating that a rectangular prism with length 4, width 3, and height 2 has 16 diagonals -/
theorem rectangular_prism_diagonals :
  ∀ (prism : RectangularPrism), num_diagonals prism = 16 := by sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l135_13557


namespace NUMINAMATH_CALUDE_james_candy_bar_sales_l135_13512

/-- Proves that James sells 5 boxes of candy bars given the conditions of the fundraiser -/
theorem james_candy_bar_sales :
  let boxes_to_bars : ℕ → ℕ := λ x => 10 * x
  let selling_price : ℚ := 3/2
  let buying_price : ℚ := 1
  let profit_per_bar : ℚ := selling_price - buying_price
  let total_profit : ℚ := 25
  ∃ (num_boxes : ℕ), 
    (boxes_to_bars num_boxes : ℚ) * profit_per_bar = total_profit ∧ 
    num_boxes = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_james_candy_bar_sales_l135_13512


namespace NUMINAMATH_CALUDE_school_gender_difference_l135_13592

theorem school_gender_difference (girls boys : ℕ) 
  (h1 : girls = 34) 
  (h2 : boys = 841) : 
  boys - girls = 807 := by
  sorry

end NUMINAMATH_CALUDE_school_gender_difference_l135_13592


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l135_13556

/-- An isosceles right triangle with leg length 6 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  leg_length_eq : leg_length = 6

/-- A square inscribed in the triangle with one vertex at the right angle -/
structure InscribedSquareA (triangle : IsoscelesRightTriangle) where
  side_length : ℝ
  vertex_at_right_angle : True
  side_along_leg : True

/-- A square inscribed in the triangle with one side along the other leg -/
structure InscribedSquareB (triangle : IsoscelesRightTriangle) where
  side_length : ℝ
  side_along_leg : True

/-- The theorem statement -/
theorem inscribed_squares_ratio 
  (triangle : IsoscelesRightTriangle) 
  (square_a : InscribedSquareA triangle) 
  (square_b : InscribedSquareB triangle) : 
  square_a.side_length / square_b.side_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l135_13556


namespace NUMINAMATH_CALUDE_max_gcd_lcm_value_l135_13577

theorem max_gcd_lcm_value (a b c : ℕ) 
  (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) : 
  Nat.gcd (Nat.lcm a b) c ≤ 10 ∧ 
  ∃ (a' b' c' : ℕ), Nat.gcd (Nat.lcm a' b') c' = 10 ∧ 
    Nat.gcd (Nat.lcm a' b') c' * Nat.lcm (Nat.gcd a' b') c' = 200 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_lcm_value_l135_13577


namespace NUMINAMATH_CALUDE_trouser_sale_price_l135_13515

theorem trouser_sale_price 
  (original_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : original_price = 100)
  (h2 : discount_percentage = 80) : 
  original_price * (1 - discount_percentage / 100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_trouser_sale_price_l135_13515


namespace NUMINAMATH_CALUDE_square_roots_problem_l135_13505

theorem square_roots_problem (a m : ℝ) : 
  ((2 - m)^2 = a ∧ (2*m + 1)^2 = a) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l135_13505


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l135_13521

theorem max_sum_on_circle (x y : ℕ) : x^2 + y^2 = 64 → x + y ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l135_13521


namespace NUMINAMATH_CALUDE_decimal_multiplication_l135_13548

theorem decimal_multiplication (h : 213 * 16 = 3408) : 0.16 * 2.13 = 0.3408 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l135_13548


namespace NUMINAMATH_CALUDE_sheet_area_difference_l135_13599

/-- The difference in combined area of front and back between two rectangular sheets of paper -/
theorem sheet_area_difference (l1 w1 l2 w2 : ℕ) : 
  l1 = 14 ∧ w1 = 12 ∧ l2 = 9 ∧ w2 = 14 → 2 * (l1 * w1) - 2 * (l2 * w2) = 84 := by
  sorry

#check sheet_area_difference

end NUMINAMATH_CALUDE_sheet_area_difference_l135_13599


namespace NUMINAMATH_CALUDE_lemonade_distribution_theorem_l135_13578

/-- Represents the distribution of lemonade cups --/
structure LemonadeDistribution where
  total : ℕ
  sold_to_kids : ℕ

/-- Checks if the lemonade distribution is valid --/
def is_valid_distribution (d : LemonadeDistribution) : Prop :=
  d.total = 56 ∧
  d.sold_to_kids + d.sold_to_kids / 2 + 1 = d.total / 2

/-- Theorem stating that the valid distribution has 18 cups sold to kids --/
theorem lemonade_distribution_theorem (d : LemonadeDistribution) :
  is_valid_distribution d → d.sold_to_kids = 18 := by
  sorry

#check lemonade_distribution_theorem

end NUMINAMATH_CALUDE_lemonade_distribution_theorem_l135_13578


namespace NUMINAMATH_CALUDE_distance_from_origin_implies_k_range_l135_13564

theorem distance_from_origin_implies_k_range (k : ℝ) (h1 : k > 0) :
  (∃ x : ℝ, x ≠ 0 ∧ x^2 + (k/x)^2 = 1) → 0 < k ∧ k ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_implies_k_range_l135_13564


namespace NUMINAMATH_CALUDE_boys_in_biology_class_l135_13581

/-- Given a Physics class with 200 students and a Biology class with half as many students,
    where the ratio of girls to boys in Biology is 3:1, prove that there are 25 boys in Biology. -/
theorem boys_in_biology_class
  (physics_students : ℕ)
  (biology_students : ℕ)
  (girls_to_boys_ratio : ℚ)
  (h1 : physics_students = 200)
  (h2 : biology_students = physics_students / 2)
  (h3 : girls_to_boys_ratio = 3)
  : biology_students / (girls_to_boys_ratio + 1) = 25 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_biology_class_l135_13581


namespace NUMINAMATH_CALUDE_divisibility_properties_l135_13506

theorem divisibility_properties :
  (∃ k : ℤ, 2^41 + 1 = 83 * k) ∧
  (∃ m : ℤ, 2^70 + 3^70 = 13 * m) ∧
  (∃ n : ℤ, 2^60 - 1 = 20801 * n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_properties_l135_13506


namespace NUMINAMATH_CALUDE_min_value_theorem_l135_13587

theorem min_value_theorem (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) :
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 ∧
  ((x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) = 9 ↔ x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l135_13587


namespace NUMINAMATH_CALUDE_f_analytical_expression_l135_13510

def f : Set ℝ := {x : ℝ | x ≠ -1}

theorem f_analytical_expression :
  ∀ x : ℝ, x ∈ f ↔ x ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_f_analytical_expression_l135_13510


namespace NUMINAMATH_CALUDE_smallest_positive_angle_for_negative_2015_l135_13572

-- Define the concept of angle equivalence
def angle_equivalent (a b : ℤ) : Prop := ∃ k : ℤ, b = a + 360 * k

-- Define the function to find the smallest positive equivalent angle
def smallest_positive_equivalent (a : ℤ) : ℤ :=
  (a % 360 + 360) % 360

-- Theorem statement
theorem smallest_positive_angle_for_negative_2015 :
  smallest_positive_equivalent (-2015) = 145 ∧
  angle_equivalent (-2015) 145 ∧
  ∀ x : ℤ, 0 < x ∧ x < 145 → ¬(angle_equivalent (-2015) x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_for_negative_2015_l135_13572


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l135_13561

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25             -- Shorter leg length
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l135_13561


namespace NUMINAMATH_CALUDE_two_interviewers_passing_l135_13570

def number_of_interviewers : ℕ := 5
def interviewers_to_choose : ℕ := 2

theorem two_interviewers_passing :
  Nat.choose number_of_interviewers interviewers_to_choose = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_interviewers_passing_l135_13570


namespace NUMINAMATH_CALUDE_quadratic_even_iff_b_zero_l135_13522

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Definition of an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem quadratic_even_iff_b_zero (a b c : ℝ) :
  is_even (quadratic a b c) ↔ b = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_even_iff_b_zero_l135_13522


namespace NUMINAMATH_CALUDE_arctan_cos_solution_l135_13529

theorem arctan_cos_solution (x : Real) :
  -π ≤ x ∧ x ≤ π →
  Real.arctan (Real.cos x) = x / 3 →
  x = Real.arccos (Real.sqrt ((Real.sqrt 5 - 1) / 2)) ∨
  x = -Real.arccos (Real.sqrt ((Real.sqrt 5 - 1) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_arctan_cos_solution_l135_13529


namespace NUMINAMATH_CALUDE_nested_expression_value_l135_13550

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l135_13550


namespace NUMINAMATH_CALUDE_abs_neg_one_third_l135_13576

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_l135_13576


namespace NUMINAMATH_CALUDE_pinterest_group_pins_l135_13535

/-- Calculates the number of pins in a Pinterest group after one month -/
def pinsAfterOneMonth (
  initialPins : ℕ
  ) (
  members : ℕ
  ) (
  dailyContributionPerMember : ℕ
  ) (
  weeklyDeletionPerMember : ℕ
  ) : ℕ :=
  let monthlyContribution := members * dailyContributionPerMember * 30
  let monthlyDeletion := members * weeklyDeletionPerMember * 30 / 7
  initialPins + monthlyContribution - monthlyDeletion

theorem pinterest_group_pins :
  pinsAfterOneMonth 1000 20 10 5 = 6571 := by
  sorry

end NUMINAMATH_CALUDE_pinterest_group_pins_l135_13535


namespace NUMINAMATH_CALUDE_square_number_problem_l135_13559

theorem square_number_problem : ∃ x : ℤ, 
  (∃ m : ℤ, x + 15 = m^2) ∧ 
  (∃ n : ℤ, x - 74 = n^2) ∧ 
  x = 2010 := by
  sorry

end NUMINAMATH_CALUDE_square_number_problem_l135_13559


namespace NUMINAMATH_CALUDE_fraction_representation_l135_13588

theorem fraction_representation (n : ℕ) : ∃ x y : ℕ, n = x^2 / y^3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_representation_l135_13588


namespace NUMINAMATH_CALUDE_simplify_expression_l135_13546

theorem simplify_expression (x : ℝ) : x - 2*(1+x) + 3*(1-x) - 4*(1+2*x) = -12*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l135_13546


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l135_13514

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, x^2 - 2*(3*x - 2) + (x + 1) = 0 ↔ x^2 - 5*x + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l135_13514


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l135_13533

-- Define arithmetic sequences a_n and b_n
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define the problem statement
theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  arithmetic_sequence b →
  a 1 = 25 →
  b 1 = 75 →
  a 2 + b 2 = 100 →
  a 37 + b 37 = 100 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l135_13533


namespace NUMINAMATH_CALUDE_power_of_three_mod_ten_l135_13560

theorem power_of_three_mod_ten : 3^19 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_power_of_three_mod_ten_l135_13560


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_necessity_not_sufficient_l135_13591

/-- Defines an ellipse in terms of its equation coefficients -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

/-- The main theorem stating that mn > 0 is necessary but not sufficient for mx^2 + ny^2 = 1 to be an ellipse -/
theorem mn_positive_necessary_not_sufficient :
  (∀ m n : ℝ, is_ellipse m n → m * n > 0) ∧
  (∃ m n : ℝ, m * n > 0 ∧ ¬is_ellipse m n) :=
sorry

/-- Proving necessity: if mx^2 + ny^2 = 1 is an ellipse, then mn > 0 -/
theorem necessity (m n : ℝ) (h : is_ellipse m n) : m * n > 0 :=
sorry

/-- Proving not sufficient: there exist m and n where mn > 0 but mx^2 + ny^2 = 1 is not an ellipse -/
theorem not_sufficient : ∃ m n : ℝ, m * n > 0 ∧ ¬is_ellipse m n :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_necessity_not_sufficient_l135_13591


namespace NUMINAMATH_CALUDE_problem_solution_l135_13582

theorem problem_solution (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (7 * x) * Real.sqrt (21 * x) = 42) : 
  x = Real.sqrt (21 / 47) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l135_13582


namespace NUMINAMATH_CALUDE_gloria_pencils_l135_13568

theorem gloria_pencils (G : ℕ) (h : G + 99 = 101) : G = 2 := by
  sorry

end NUMINAMATH_CALUDE_gloria_pencils_l135_13568


namespace NUMINAMATH_CALUDE_fraction_comparison_l135_13562

theorem fraction_comparison : (2 : ℝ) / 3 > (5 - Real.sqrt 11) / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l135_13562


namespace NUMINAMATH_CALUDE_target_hit_probability_l135_13501

theorem target_hit_probability (p_a p_b : ℝ) (h_a : p_a = 0.6) (h_b : p_b = 0.5) :
  let p_hit := 1 - (1 - p_a) * (1 - p_b)
  (p_a / p_hit) = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l135_13501


namespace NUMINAMATH_CALUDE_union_complement_equals_l135_13586

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 4}

-- Define set N
def N : Finset Nat := {2, 5}

-- Theorem statement
theorem union_complement_equals : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equals_l135_13586


namespace NUMINAMATH_CALUDE_pauls_savings_l135_13593

/-- Paul's initial savings in dollars -/
def initial_savings : ℕ := sorry

/-- Cost of one toy in dollars -/
def toy_cost : ℕ := 5

/-- Number of toys Paul wants to buy -/
def num_toys : ℕ := 2

/-- Additional money Paul receives in dollars -/
def additional_money : ℕ := 7

theorem pauls_savings :
  initial_savings = 3 ∧
  initial_savings + additional_money = num_toys * toy_cost :=
by sorry

end NUMINAMATH_CALUDE_pauls_savings_l135_13593


namespace NUMINAMATH_CALUDE_election_majority_proof_l135_13517

/-- 
In an election with a total of 4500 votes, where the winning candidate receives 60% of the votes,
prove that the majority of votes by which the candidate won is 900.
-/
theorem election_majority_proof (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 4500 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).num / (winning_percentage * total_votes : ℚ).den -
  ((1 - winning_percentage) * total_votes : ℚ).num / ((1 - winning_percentage) * total_votes : ℚ).den = 900 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_proof_l135_13517


namespace NUMINAMATH_CALUDE_smallest_block_size_l135_13563

/-- Given a rectangular block made of N identical 1-cm cubes, where 378 cubes are not visible
    when three faces are visible, the smallest possible value of N is 560. -/
theorem smallest_block_size (N : ℕ) : 
  (∃ l m n : ℕ, (l - 1) * (m - 1) * (n - 1) = 378 ∧ N = l * m * n) →
  (∀ N' : ℕ, (∃ l' m' n' : ℕ, (l' - 1) * (m' - 1) * (n' - 1) = 378 ∧ N' = l' * m' * n') → N' ≥ N) →
  N = 560 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_size_l135_13563


namespace NUMINAMATH_CALUDE_series_sum_equals_four_l135_13541

/-- The sum of the series ∑(4n+2)/3^n from n=1 to infinity equals 4. -/
theorem series_sum_equals_four :
  (∑' n : ℕ, (4 * n + 2) / (3 : ℝ) ^ n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_four_l135_13541


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l135_13544

theorem interest_rate_calculation (principal_B principal_C time_B time_C total_interest : ℕ) 
  (h1 : principal_B = 5000)
  (h2 : principal_C = 3000)
  (h3 : time_B = 2)
  (h4 : time_C = 4)
  (h5 : total_interest = 2640) :
  let rate := total_interest * 100 / (principal_B * time_B + principal_C * time_C)
  rate = 12 := by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l135_13544


namespace NUMINAMATH_CALUDE_carol_bought_three_packs_l135_13536

-- Define the problem parameters
def total_invitations : ℕ := 12
def invitations_per_pack : ℕ := 4

-- Define the function to calculate the number of packs
def number_of_packs : ℕ := total_invitations / invitations_per_pack

-- Theorem statement
theorem carol_bought_three_packs : number_of_packs = 3 := by
  sorry

end NUMINAMATH_CALUDE_carol_bought_three_packs_l135_13536


namespace NUMINAMATH_CALUDE_percentage_calculation_l135_13523

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 4800 → 
  (P / 100) * (30 / 100) * (50 / 100) * N = 108 → 
  P = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l135_13523


namespace NUMINAMATH_CALUDE_right_triangle_median_length_l135_13598

theorem right_triangle_median_length (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_sides : (a = 6 ∧ b = 8) ∨ (a = 6 ∧ c = 8) ∨ (b = 6 ∧ c = 8)) :
  ∃ m : ℝ, (m = 4 ∨ m = 5) ∧ 2 * m = c :=
sorry

end NUMINAMATH_CALUDE_right_triangle_median_length_l135_13598


namespace NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l135_13526

/-- Converts a quinary (base-5) number to decimal --/
def quinary_to_decimal (q : ℕ) : ℕ := sorry

/-- Converts a decimal number to octal --/
def decimal_to_octal (d : ℕ) : ℕ := sorry

/-- Theorem: The quinary number 444₅ is equal to the octal number 174₈ --/
theorem quinary_444_equals_octal_174 : 
  decimal_to_octal (quinary_to_decimal 444) = 174 := by sorry

end NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l135_13526


namespace NUMINAMATH_CALUDE_shirt_coat_ratio_l135_13590

/-- Given a shirt costing $150 and a total cost of $600 for the shirt and coat,
    prove that the ratio of the cost of the shirt to the cost of the coat is 1:3. -/
theorem shirt_coat_ratio (shirt_cost coat_cost total_cost : ℕ) : 
  shirt_cost = 150 → 
  total_cost = 600 → 
  total_cost = shirt_cost + coat_cost →
  (shirt_cost : ℚ) / coat_cost = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shirt_coat_ratio_l135_13590


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l135_13595

theorem coefficient_x_squared_in_binomial_expansion :
  let binomial := (x + 2/x)^4
  ∃ (a b c d e : ℝ), binomial = a*x^4 + b*x^3 + c*x^2 + d*x + e ∧ c = 8 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l135_13595


namespace NUMINAMATH_CALUDE_regular_polygon_with_135_degree_angles_has_8_sides_l135_13569

/-- The number of sides of a regular polygon with interior angles measuring 135 degrees. -/
def regular_polygon_sides : ℕ := 8

/-- The measure of each interior angle of the regular polygon in degrees. -/
def interior_angle : ℝ := 135

/-- Theorem stating that a regular polygon with interior angles measuring 135 degrees has 8 sides. -/
theorem regular_polygon_with_135_degree_angles_has_8_sides :
  (interior_angle * regular_polygon_sides : ℝ) = 180 * (regular_polygon_sides - 2) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_135_degree_angles_has_8_sides_l135_13569


namespace NUMINAMATH_CALUDE_arithmetic_computation_l135_13558

theorem arithmetic_computation : 12 + 4 * (2 * 3 - 8 + 1)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l135_13558


namespace NUMINAMATH_CALUDE_sam_initial_watermelons_l135_13507

/-- The number of watermelons Sam grew initially -/
def initial_watermelons : ℕ := sorry

/-- The number of additional watermelons Sam grew -/
def additional_watermelons : ℕ := 3

/-- The total number of watermelons Sam has now -/
def total_watermelons : ℕ := 7

/-- Theorem stating that Sam grew 4 watermelons initially -/
theorem sam_initial_watermelons : 
  initial_watermelons + additional_watermelons = total_watermelons → initial_watermelons = 4 := by
  sorry

end NUMINAMATH_CALUDE_sam_initial_watermelons_l135_13507


namespace NUMINAMATH_CALUDE_robin_phone_pictures_l135_13565

theorem robin_phone_pictures (phone_pics camera_pics albums pics_per_album : ℕ) :
  camera_pics = 5 →
  albums = 5 →
  pics_per_album = 8 →
  phone_pics + camera_pics = albums * pics_per_album →
  phone_pics = 35 := by
  sorry

end NUMINAMATH_CALUDE_robin_phone_pictures_l135_13565


namespace NUMINAMATH_CALUDE_sqrt_ratio_eq_three_implies_x_eq_eighteen_sevenths_l135_13594

theorem sqrt_ratio_eq_three_implies_x_eq_eighteen_sevenths :
  ∀ x : ℚ, (Real.sqrt (8 * x) / Real.sqrt (4 * (x - 2)) = 3) → x = 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ratio_eq_three_implies_x_eq_eighteen_sevenths_l135_13594


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l135_13524

theorem unique_solution_exponential_equation :
  ∃! y : ℝ, (10 : ℝ)^(2*y) * (100 : ℝ)^y = (1000 : ℝ)^3 * (10 : ℝ)^y :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l135_13524


namespace NUMINAMATH_CALUDE_three_digit_factorial_sum_l135_13525

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem three_digit_factorial_sum : ∃ a b c : ℕ, 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  100 * a + 10 * b + c = factorial a + factorial b + factorial c := by
  sorry

end NUMINAMATH_CALUDE_three_digit_factorial_sum_l135_13525


namespace NUMINAMATH_CALUDE_divide_by_repeating_decimal_l135_13552

def repeating_decimal_to_fraction (a b : ℕ) : ℚ :=
  (a : ℚ) / (99 : ℚ)

theorem divide_by_repeating_decimal (a b : ℕ) :
  (7 : ℚ) / (repeating_decimal_to_fraction a b) = 38.5 :=
sorry

end NUMINAMATH_CALUDE_divide_by_repeating_decimal_l135_13552


namespace NUMINAMATH_CALUDE_PQ_perpendicular_RS_l135_13542

-- Define the points
variable (A B C D M P Q R S : ℝ × ℝ)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D M : ℝ × ℝ) : Prop := sorry

-- Define centroid
def is_centroid (P A M D : ℝ × ℝ) : Prop := sorry

-- Define orthocenter
def is_orthocenter (R D M C : ℝ × ℝ) : Prop := sorry

-- Define perpendicularity of vectors
def vectors_perpendicular (P Q R S : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem PQ_perpendicular_RS 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : diagonals_intersect_at A B C D M)
  (h3 : is_centroid P A M D)
  (h4 : is_centroid Q C M B)
  (h5 : is_orthocenter R D M C)
  (h6 : is_orthocenter S M A B) :
  vectors_perpendicular P Q R S := by sorry

end NUMINAMATH_CALUDE_PQ_perpendicular_RS_l135_13542


namespace NUMINAMATH_CALUDE_no_base_for_172_four_digit_odd_final_l135_13596

theorem no_base_for_172_four_digit_odd_final (b : ℕ) : ¬ (
  (b ^ 3 ≤ 172 ∧ 172 < b ^ 4) ∧  -- four-digit number condition
  (172 % b % 2 = 1)              -- odd final digit condition
) := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_172_four_digit_odd_final_l135_13596


namespace NUMINAMATH_CALUDE_stationary_train_length_is_1296_l135_13504

/-- The length of a stationary train given the time it takes for another train to pass it. -/
def stationary_train_length (time_to_pass_pole : ℝ) (time_to_cross_stationary : ℝ) (train_speed : ℝ) : ℝ :=
  train_speed * time_to_cross_stationary - train_speed * time_to_pass_pole

/-- Theorem stating that the length of the stationary train is 1296 meters under the given conditions. -/
theorem stationary_train_length_is_1296 :
  stationary_train_length 5 25 64.8 = 1296 := by
  sorry

#eval stationary_train_length 5 25 64.8

end NUMINAMATH_CALUDE_stationary_train_length_is_1296_l135_13504


namespace NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l135_13597

theorem cube_diff_even_iff_sum_even (p q : ℕ) :
  Even (p^3 - q^3) ↔ Even (p + q) := by
  sorry

end NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l135_13597


namespace NUMINAMATH_CALUDE_positive_integer_solution_of_equation_l135_13573

theorem positive_integer_solution_of_equation (x : ℕ+) :
  (4 * x.val^2 - 16 * x.val - 60 = 0) → x.val = 6 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solution_of_equation_l135_13573


namespace NUMINAMATH_CALUDE_ice_cream_cost_is_734_l135_13589

/-- The cost of Mrs. Hilt's ice cream purchase -/
def ice_cream_cost : ℚ :=
  let vanilla_price : ℚ := 99 / 100
  let chocolate_price : ℚ := 129 / 100
  let strawberry_price : ℚ := 149 / 100
  let vanilla_quantity : ℕ := 2
  let chocolate_quantity : ℕ := 3
  let strawberry_quantity : ℕ := 1
  vanilla_price * vanilla_quantity +
  chocolate_price * chocolate_quantity +
  strawberry_price * strawberry_quantity

theorem ice_cream_cost_is_734 : ice_cream_cost = 734 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_is_734_l135_13589


namespace NUMINAMATH_CALUDE_fraction_of_married_women_l135_13502

theorem fraction_of_married_women (total : ℕ) (total_pos : total > 0) :
  let women := (58 : ℚ) / 100 * total
  let married := (60 : ℚ) / 100 * total
  let men := total - women
  let single_men := (2 : ℚ) / 3 * men
  let married_men := men - single_men
  let married_women := married - married_men
  (married_women / women) = 23 / 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_married_women_l135_13502


namespace NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l135_13519

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is perpendicular to two different planes, then the planes are parallel -/
theorem line_perp_two_planes_implies_parallel 
  (l : Line3D) (α β : Plane3D) 
  (h_diff : α ≠ β) 
  (h_perp_α : perpendicular l α) 
  (h_perp_β : perpendicular l β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l135_13519


namespace NUMINAMATH_CALUDE_digit_multiplication_sum_l135_13584

theorem digit_multiplication_sum (p q : ℕ) : 
  p < 10 → q < 10 → (40 + p) * (10 * q + 5) = 190 → p + q = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_multiplication_sum_l135_13584


namespace NUMINAMATH_CALUDE_intersection_with_complement_l135_13566

open Set

def P : Set ℝ := {1, 2, 3, 4, 5}
def Q : Set ℝ := {4, 5, 6}

theorem intersection_with_complement : P ∩ (univ \ Q) = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l135_13566


namespace NUMINAMATH_CALUDE_cube_root_of_eight_l135_13527

theorem cube_root_of_eight (x y : ℝ) : x^(3*y) = 8 ∧ x = 8 → y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_eight_l135_13527
