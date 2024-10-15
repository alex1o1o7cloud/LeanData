import Mathlib

namespace NUMINAMATH_CALUDE_investment_period_ratio_l3625_362514

/-- Represents the profit distribution in a joint business venture -/
structure JointBusiness where
  investment_a : ℚ
  investment_b : ℚ
  period_a : ℚ
  period_b : ℚ
  profit_b : ℚ
  total_profit : ℚ

/-- Theorem stating the ratio of investment periods given the conditions -/
theorem investment_period_ratio (jb : JointBusiness)
  (h1 : jb.investment_a = 3 * jb.investment_b)
  (h2 : ∃ k : ℚ, jb.period_a = k * jb.period_b)
  (h3 : jb.profit_b = 4000)
  (h4 : jb.total_profit = 28000) :
  jb.period_a / jb.period_b = 2 := by
  sorry

#check investment_period_ratio

end NUMINAMATH_CALUDE_investment_period_ratio_l3625_362514


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l3625_362572

theorem smallest_lcm_with_gcd_five (k ℓ : ℕ) : 
  k ≥ 1000 → k < 10000 → ℓ ≥ 1000 → ℓ < 10000 → Nat.gcd k ℓ = 5 → 
  Nat.lcm k ℓ ≥ 201000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l3625_362572


namespace NUMINAMATH_CALUDE_binomial_20_19_l3625_362528

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end NUMINAMATH_CALUDE_binomial_20_19_l3625_362528


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l3625_362595

theorem largest_n_with_unique_k : ∃ (n : ℕ), n > 0 ∧ 
  (∃! (k : ℤ), (5 : ℚ)/18 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 9/17) ∧
  (∀ (m : ℕ), m > n → ¬(∃! (k : ℤ), (5 : ℚ)/18 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 9/17)) ∧
  n = 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l3625_362595


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_to_height_ratio_l3625_362561

/-- A regular tetrahedron with height H and an inscribed sphere of radius R -/
structure RegularTetrahedron where
  H : ℝ
  R : ℝ
  H_pos : H > 0
  R_pos : R > 0

/-- The ratio of the radius of the inscribed sphere to the height of a regular tetrahedron is 1:4 -/
theorem inscribed_sphere_radius_to_height_ratio (t : RegularTetrahedron) : t.R / t.H = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_to_height_ratio_l3625_362561


namespace NUMINAMATH_CALUDE_floor_x_floor_x_eq_48_l3625_362503

open Real

theorem floor_x_floor_x_eq_48 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 48 ↔ 8 ≤ x ∧ x < 49 / 6 := by
  sorry

end NUMINAMATH_CALUDE_floor_x_floor_x_eq_48_l3625_362503


namespace NUMINAMATH_CALUDE_difference_of_squares_l3625_362532

theorem difference_of_squares (a b : ℕ+) : 
  ∃ (x y z w : ℤ), (a : ℤ) = x^2 - y^2 ∨ (b : ℤ) = z^2 - w^2 ∨ ((a + b) : ℤ) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3625_362532


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3625_362523

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3625_362523


namespace NUMINAMATH_CALUDE_smallest_c_for_negative_three_in_range_l3625_362531

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

-- State the theorem
theorem smallest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (c' : ℝ), (∃ (x : ℝ), f c' x = -3) → c ≤ c') ∧
  (∃ (x : ℝ), f (-3/4) x = -3) :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_negative_three_in_range_l3625_362531


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3625_362546

theorem chess_tournament_games (n : ℕ) (h : n = 12) :
  2 * n * (n - 1) / 2 = 264 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3625_362546


namespace NUMINAMATH_CALUDE_problem_solution_l3625_362522

theorem problem_solution (t : ℚ) :
  let x := 3 - 2 * t
  let y := 5 * t + 6
  x = 0 → y = 27 / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3625_362522


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3625_362504

-- Define the speed of the stream
def stream_speed : ℝ := 5

-- Define the distance traveled downstream
def downstream_distance : ℝ := 81

-- Define the time taken to travel downstream
def downstream_time : ℝ := 3

-- Define the speed of the boat in still water
def boat_speed : ℝ := 22

-- Theorem statement
theorem boat_speed_in_still_water :
  boat_speed = downstream_distance / downstream_time - stream_speed :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3625_362504


namespace NUMINAMATH_CALUDE_rubble_purchase_l3625_362558

/-- Calculates the remaining money after a purchase. -/
def remaining_money (initial_amount notebook_cost pen_cost : ℚ) : ℚ :=
  initial_amount - (2 * notebook_cost + 2 * pen_cost)

/-- Proves that Rubble will have $4.00 left after his purchase. -/
theorem rubble_purchase : remaining_money 15 4 (3/2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_rubble_purchase_l3625_362558


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3625_362556

theorem polynomial_expansion (x : ℝ) : 
  (3*x^3 + x^2 - 5*x + 9)*(x + 2) - (x + 2)*(2*x^3 - 4*x + 8) + (x^2 - 6*x + 13)*(x + 2)*(x - 3) = 
  2*x^4 + x^3 + 9*x^2 + 23*x + 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3625_362556


namespace NUMINAMATH_CALUDE_solve_john_age_problem_l3625_362533

def john_age_problem (john_current_age : ℕ) (sister_age_multiplier : ℕ) (sister_future_age : ℕ) : Prop :=
  let sister_current_age := john_current_age * sister_age_multiplier
  let age_difference := sister_current_age - john_current_age
  let john_future_age := sister_future_age - age_difference
  john_future_age = sister_future_age - age_difference

theorem solve_john_age_problem :
  john_age_problem 10 2 60 = true :=
sorry

end NUMINAMATH_CALUDE_solve_john_age_problem_l3625_362533


namespace NUMINAMATH_CALUDE_narcissus_count_is_75_l3625_362557

/-- The number of narcissus flowers in a florist's inventory -/
def narcissus_count : ℕ := 75

/-- The number of chrysanthemums in the florist's inventory -/
def chrysanthemum_count : ℕ := 90

/-- The number of bouquets that can be made -/
def bouquet_count : ℕ := 33

/-- The number of flowers in each bouquet -/
def flowers_per_bouquet : ℕ := 5

/-- Theorem stating that the number of narcissus flowers is 75 -/
theorem narcissus_count_is_75 : 
  narcissus_count = bouquet_count * flowers_per_bouquet - chrysanthemum_count :=
by
  sorry

#eval narcissus_count -- Should output 75

end NUMINAMATH_CALUDE_narcissus_count_is_75_l3625_362557


namespace NUMINAMATH_CALUDE_exponent_division_l3625_362554

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3625_362554


namespace NUMINAMATH_CALUDE_custom_operation_solution_l3625_362571

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 2 * a^2 - b

-- State the theorem
theorem custom_operation_solution :
  ∃ x : ℝ, (star 3 (star 4 x) = 8) ∧ (x = 22) := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_solution_l3625_362571


namespace NUMINAMATH_CALUDE_plywood_perimeter_difference_l3625_362578

def plywood_width : ℝ := 3
def plywood_length : ℝ := 9
def num_pieces : ℕ := 6

def is_valid_cut (w h : ℝ) : Prop :=
  w * h * num_pieces = plywood_width * plywood_length ∧
  (w = plywood_width ∨ h = plywood_width ∨ w = plywood_length ∨ h = plywood_length ∨
   w * num_pieces = plywood_width ∨ h * num_pieces = plywood_width ∨
   w * num_pieces = plywood_length ∨ h * num_pieces = plywood_length)

def piece_perimeter (w h : ℝ) : ℝ := 2 * (w + h)

def max_perimeter : ℝ := 20
def min_perimeter : ℝ := 8

theorem plywood_perimeter_difference :
  ∀ w h, is_valid_cut w h →
  ∃ max_w max_h min_w min_h,
    is_valid_cut max_w max_h ∧
    is_valid_cut min_w min_h ∧
    piece_perimeter max_w max_h = max_perimeter ∧
    piece_perimeter min_w min_h = min_perimeter ∧
    max_perimeter - min_perimeter = 12 :=
sorry

end NUMINAMATH_CALUDE_plywood_perimeter_difference_l3625_362578


namespace NUMINAMATH_CALUDE_circle_tangent_origin_l3625_362593

/-- A circle in the xy-plane -/
structure Circle where
  G : ℝ
  E : ℝ
  F : ℝ

/-- Predicate to check if a circle is tangent to the x-axis at the origin -/
def isTangentAtOrigin (c : Circle) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + c.G * x + c.E * y + c.F = 0 ∧
                (x = 0 ∧ y = 0) ∧
                ∀ (x' y' : ℝ), x' ≠ 0 → (x'^2 + y'^2 + c.G * x' + c.E * y' + c.F > 0)

theorem circle_tangent_origin (c : Circle) :
  isTangentAtOrigin c → c.G = 0 ∧ c.F = 0 ∧ c.E ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_origin_l3625_362593


namespace NUMINAMATH_CALUDE_prime_factors_power_l3625_362599

/-- Given an expression containing 4^11, 7^5, and 11^x, 
    if the total number of prime factors is 29, then x = 2 -/
theorem prime_factors_power (x : ℕ) : 
  (22 + 5 + x = 29) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_power_l3625_362599


namespace NUMINAMATH_CALUDE_spanish_books_count_l3625_362501

theorem spanish_books_count (total : ℕ) (english : ℕ) (french : ℕ) (italian : ℕ) (spanish : ℕ) :
  total = 280 ∧
  english = total / 5 ∧
  french = total / 7 ∧
  italian = total / 4 ∧
  spanish = total - (english + french + italian) →
  spanish = 114 := by
sorry

end NUMINAMATH_CALUDE_spanish_books_count_l3625_362501


namespace NUMINAMATH_CALUDE_rectangle_area_l3625_362543

/-- The length of the shorter side of each small rectangle -/
def short_side : ℝ := 4

/-- The number of small rectangles -/
def num_rectangles : ℕ := 4

/-- The aspect ratio of each small rectangle -/
def aspect_ratio : ℝ := 2

/-- The length of the longer side of each small rectangle -/
def long_side : ℝ := short_side * aspect_ratio

/-- The width of rectangle EFGH -/
def width : ℝ := long_side

/-- The length of rectangle EFGH -/
def length : ℝ := 2 * long_side

/-- The area of rectangle EFGH -/
def area : ℝ := width * length

theorem rectangle_area : area = 128 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3625_362543


namespace NUMINAMATH_CALUDE_lucas_chocolate_problem_l3625_362584

theorem lucas_chocolate_problem (total_students : ℕ) 
  (candy_per_student : ℕ) 
  (h1 : total_students * candy_per_student = 40) 
  (h2 : (total_students - 3) * candy_per_student = 28) :
  candy_per_student = 4 := by
  sorry

end NUMINAMATH_CALUDE_lucas_chocolate_problem_l3625_362584


namespace NUMINAMATH_CALUDE_g_x_plus_3_l3625_362547

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem
theorem g_x_plus_3 : ∀ x : ℝ, g (x + 3) = 3 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_g_x_plus_3_l3625_362547


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3625_362509

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 < 0) → (a < -2 ∨ a > 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3625_362509


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3625_362569

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.sin x > 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3625_362569


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3625_362574

theorem interest_rate_calculation (P : ℝ) (R : ℝ) : 
  P * (1 + 5 * R / 100) = 9800 →
  P * (1 + 8 * R / 100) = 12005 →
  R = 12 :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3625_362574


namespace NUMINAMATH_CALUDE_copy_machine_rate_copy_machine_rate_proof_l3625_362591

/-- Given two copy machines working together for 30 minutes to produce 3000 copies,
    where one machine produces 65 copies per minute, prove that the other machine
    must produce 35 copies per minute. -/
theorem copy_machine_rate : ℕ → Prop :=
  fun x =>
    -- x is the number of copies per minute for the first machine
    -- 65 is the number of copies per minute for the second machine
    -- 30 is the number of minutes they work
    -- 3000 is the total number of copies produced
    30 * x + 30 * 65 = 3000 →
    x = 35

-- The proof would go here, but we're skipping it as requested
theorem copy_machine_rate_proof : copy_machine_rate 35 := by sorry

end NUMINAMATH_CALUDE_copy_machine_rate_copy_machine_rate_proof_l3625_362591


namespace NUMINAMATH_CALUDE_mets_fan_count_l3625_362513

/-- Represents the number of fans for each team -/
structure FanCount where
  yankees : ℕ
  mets : ℕ
  redsox : ℕ

/-- The conditions of the problem -/
def fan_conditions (fc : FanCount) : Prop :=
  -- Ratio of Yankees to Mets fans is 3:2
  3 * fc.mets = 2 * fc.yankees ∧
  -- Ratio of Mets to Red Sox fans is 4:5
  4 * fc.redsox = 5 * fc.mets ∧
  -- Total number of fans is 360
  fc.yankees + fc.mets + fc.redsox = 360

/-- The theorem stating that under the given conditions, there are 96 Mets fans -/
theorem mets_fan_count (fc : FanCount) : fan_conditions fc → fc.mets = 96 := by
  sorry

end NUMINAMATH_CALUDE_mets_fan_count_l3625_362513


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_value_l3625_362525

/-- Given two positive integers with a specific ratio and value, prove their LCM --/
theorem lcm_of_ratio_and_value (a b : ℕ+) (h1 : a = 45) (h2 : 4 * a = 3 * b) : 
  Nat.lcm a b = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_value_l3625_362525


namespace NUMINAMATH_CALUDE_min_value_expression_l3625_362559

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (a + 1) + 4 / (b + 1) ≥ 9 / 4 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / (a₀ + 1) + 4 / (b₀ + 1) = 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3625_362559


namespace NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l3625_362596

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| - |2*x - 1|

-- Part I
theorem solution_set_part_I :
  {x : ℝ | f x 2 + 3 ≥ 0} = {x : ℝ | -4 ≤ x ∧ x ≤ 2} := by sorry

-- Part II
theorem range_of_a_part_II :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, f x a ≤ 3) ↔ a ∈ Set.Icc (-3) 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l3625_362596


namespace NUMINAMATH_CALUDE_reading_days_l3625_362517

-- Define the reading speed in words per hour
def reading_speed : ℕ := 100

-- Define the number of words in each book
def book1_words : ℕ := 200
def book2_words : ℕ := 400
def book3_words : ℕ := 300

-- Define the average reading time per day in minutes
def avg_reading_time : ℕ := 54

-- Define the total number of words
def total_words : ℕ := book1_words + book2_words + book3_words

-- Theorem to prove
theorem reading_days : 
  (total_words / reading_speed : ℚ) / (avg_reading_time / 60 : ℚ) = 10 := by
  sorry


end NUMINAMATH_CALUDE_reading_days_l3625_362517


namespace NUMINAMATH_CALUDE_complex_number_location_l3625_362582

theorem complex_number_location (z : ℂ) (h : z * (1 - 2*I) = I) :
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3625_362582


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3625_362512

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (a ≥ 2 → monotonic_on (f a) 1 2) ∧ 
  ¬(monotonic_on (f a) 1 2 → a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3625_362512


namespace NUMINAMATH_CALUDE_optimal_allocation_l3625_362565

/-- Represents an investment project --/
structure Project where
  maxProfitRate : ℝ
  maxLossRate : ℝ

/-- Represents an investment allocation --/
structure Allocation where
  projectA : ℝ
  projectB : ℝ

/-- Calculates the potential profit for a given allocation --/
def potentialProfit (projects : Project × Project) (alloc : Allocation) : ℝ :=
  alloc.projectA * projects.1.maxProfitRate + alloc.projectB * projects.2.maxProfitRate

/-- Calculates the potential loss for a given allocation --/
def potentialLoss (projects : Project × Project) (alloc : Allocation) : ℝ :=
  alloc.projectA * projects.1.maxLossRate + alloc.projectB * projects.2.maxLossRate

/-- Theorem: The optimal allocation maximizes profit while satisfying constraints --/
theorem optimal_allocation
  (projectA : Project)
  (projectB : Project)
  (totalLimit : ℝ)
  (lossLimit : ℝ)
  (h1 : projectA.maxProfitRate = 1)
  (h2 : projectB.maxProfitRate = 0.5)
  (h3 : projectA.maxLossRate = 0.3)
  (h4 : projectB.maxLossRate = 0.1)
  (h5 : totalLimit = 100000)
  (h6 : lossLimit = 18000) :
  ∃ (alloc : Allocation),
    alloc.projectA = 40000 ∧
    alloc.projectB = 60000 ∧
    alloc.projectA + alloc.projectB ≤ totalLimit ∧
    potentialLoss (projectA, projectB) alloc ≤ lossLimit ∧
    ∀ (otherAlloc : Allocation),
      otherAlloc.projectA + otherAlloc.projectB ≤ totalLimit →
      potentialLoss (projectA, projectB) otherAlloc ≤ lossLimit →
      potentialProfit (projectA, projectB) alloc ≥ potentialProfit (projectA, projectB) otherAlloc :=
by sorry

end NUMINAMATH_CALUDE_optimal_allocation_l3625_362565


namespace NUMINAMATH_CALUDE_als_original_investment_l3625_362583

-- Define the original investment amounts
variable (a b c d : ℝ)

-- Define the conditions
axiom total_investment : a + b + c + d = 1200
axiom different_amounts : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom final_total : (a - 150) + 3*b + 3*c + 2*d = 1800

-- Theorem to prove
theorem als_original_investment : a = 825 := by
  sorry

end NUMINAMATH_CALUDE_als_original_investment_l3625_362583


namespace NUMINAMATH_CALUDE_paint_container_rectangle_perimeter_l3625_362562

theorem paint_container_rectangle_perimeter :
  ∀ x : ℝ,
  -- Old rectangle conditions
  x > 0 →
  let old_width := x
  let old_length := 3 * x
  let old_area := old_width * old_length
  -- New rectangle conditions
  let new_width := x + 8
  let new_length := 3 * x - 18
  let new_area := new_width * new_length
  -- Equal area condition
  old_area = new_area →
  -- Perimeter calculation
  let new_perimeter := 2 * (new_width + new_length)
  -- Theorem statement
  new_perimeter = 172 :=
by
  sorry


end NUMINAMATH_CALUDE_paint_container_rectangle_perimeter_l3625_362562


namespace NUMINAMATH_CALUDE_max_roses_for_680_l3625_362538

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : ℚ  -- Price of an individual rose
  dozen : ℚ       -- Price of a dozen roses
  twoDozen : ℚ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def maxRoses (budget : ℚ) (pricing : RosePricing) : ℕ :=
  sorry

/-- The specific pricing for the problem -/
def problemPricing : RosePricing :=
  { individual := 730/100,  -- $7.30
    dozen := 36,            -- $36
    twoDozen := 50 }        -- $50

theorem max_roses_for_680 :
  maxRoses 680 problemPricing = 316 :=
sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l3625_362538


namespace NUMINAMATH_CALUDE_kim_morning_routine_time_l3625_362519

/-- Represents Kim's morning routine and calculates the total time taken. -/
def morning_routine_time (total_employees : ℕ) (senior_employees : ℕ) (overtime_employees : ℕ)
  (coffee_time : ℕ) (regular_status_time : ℕ) (senior_status_extra_time : ℕ)
  (overtime_payroll_time : ℕ) (regular_payroll_time : ℕ)
  (email_time : ℕ) (task_allocation_time : ℕ) : ℕ :=
  let regular_employees := total_employees - senior_employees
  let non_overtime_employees := total_employees - overtime_employees
  coffee_time +
  (regular_employees * regular_status_time) +
  (senior_employees * (regular_status_time + senior_status_extra_time)) +
  (overtime_employees * overtime_payroll_time) +
  (non_overtime_employees * regular_payroll_time) +
  email_time +
  task_allocation_time

/-- Theorem stating that Kim's morning routine takes 60 minutes given the specified conditions. -/
theorem kim_morning_routine_time :
  morning_routine_time 9 3 4 5 2 1 3 1 10 7 = 60 := by
  sorry

end NUMINAMATH_CALUDE_kim_morning_routine_time_l3625_362519


namespace NUMINAMATH_CALUDE_refrigerator_profit_theorem_l3625_362527

def refrigerator_profit (cost_price marked_price : ℝ) : Prop :=
  let profit_20_off := 0.8 * marked_price - cost_price
  let profit_margin := profit_20_off / cost_price
  profit_20_off = 200 ∧
  profit_margin = 0.1 ∧
  0.85 * marked_price - cost_price = 337.5

theorem refrigerator_profit_theorem :
  ∃ (cost_price marked_price : ℝ),
    refrigerator_profit cost_price marked_price :=
  sorry

end NUMINAMATH_CALUDE_refrigerator_profit_theorem_l3625_362527


namespace NUMINAMATH_CALUDE_mountain_bike_pricing_l3625_362585

/-- Represents the sales and pricing of mountain bikes over three months -/
structure MountainBikeSales where
  january_sales : ℝ
  february_price_decrease : ℝ
  february_sales : ℝ
  march_price_decrease_percentage : ℝ
  march_profit_percentage : ℝ

/-- Theorem stating the selling price in February and the cost price of each mountain bike -/
theorem mountain_bike_pricing (sales : MountainBikeSales)
  (h1 : sales.january_sales = 27000)
  (h2 : sales.february_price_decrease = 100)
  (h3 : sales.february_sales = 24000)
  (h4 : sales.march_price_decrease_percentage = 0.1)
  (h5 : sales.march_profit_percentage = 0.44) :
  ∃ (february_price cost_price : ℝ),
    february_price = 800 ∧ cost_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_mountain_bike_pricing_l3625_362585


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3625_362505

theorem no_solution_absolute_value_equation :
  ¬ ∃ x : ℝ, 3 * |x + 2| + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3625_362505


namespace NUMINAMATH_CALUDE_square_perimeter_9cm_l3625_362576

/-- The perimeter of a square with side length 9 centimeters is 36 centimeters. -/
theorem square_perimeter_9cm (s : ℝ) (h : s = 9) : 4 * s = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_9cm_l3625_362576


namespace NUMINAMATH_CALUDE_reporters_not_covering_politics_l3625_362563

/-- The percentage of reporters who cover local politics in country X -/
def local_politics_coverage : ℝ := 30

/-- The percentage of reporters who cover politics but not local politics in country X -/
def non_local_politics_coverage : ℝ := 25

/-- Theorem stating that 60% of reporters do not cover politics -/
theorem reporters_not_covering_politics :
  let total_reporters : ℝ := 100
  let reporters_covering_local_politics : ℝ := local_politics_coverage
  let reporters_covering_politics : ℝ := reporters_covering_local_politics / (1 - non_local_politics_coverage / 100)
  let reporters_not_covering_politics : ℝ := total_reporters - reporters_covering_politics
  reporters_not_covering_politics / total_reporters = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_reporters_not_covering_politics_l3625_362563


namespace NUMINAMATH_CALUDE_joan_found_70_seashells_l3625_362568

/-- The number of seashells Sam gave to Joan -/
def seashells_from_sam : ℕ := 27

/-- The total number of seashells Joan has now -/
def total_seashells : ℕ := 97

/-- The number of seashells Joan found on the beach -/
def seashells_found_on_beach : ℕ := total_seashells - seashells_from_sam

theorem joan_found_70_seashells : seashells_found_on_beach = 70 := by
  sorry

end NUMINAMATH_CALUDE_joan_found_70_seashells_l3625_362568


namespace NUMINAMATH_CALUDE_unique_solution_3m_plus_4n_eq_5k_l3625_362592

theorem unique_solution_3m_plus_4n_eq_5k :
  ∀ m n k : ℕ+, 3 * m + 4 * n = 5 * k → m = 2 ∧ n = 2 ∧ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_3m_plus_4n_eq_5k_l3625_362592


namespace NUMINAMATH_CALUDE_section_b_average_weight_l3625_362549

/-- Given a class with two sections A and B, prove that the average weight of section B is 35 kg. -/
theorem section_b_average_weight
  (students_a : ℕ)
  (students_b : ℕ)
  (total_students : ℕ)
  (avg_weight_a : ℝ)
  (avg_weight_total : ℝ)
  (h1 : students_a = 30)
  (h2 : students_b = 20)
  (h3 : total_students = students_a + students_b)
  (h4 : avg_weight_a = 40)
  (h5 : avg_weight_total = 38)
  : (total_students * avg_weight_total - students_a * avg_weight_a) / students_b = 35 := by
  sorry

#check section_b_average_weight

end NUMINAMATH_CALUDE_section_b_average_weight_l3625_362549


namespace NUMINAMATH_CALUDE_black_squares_in_37th_row_l3625_362508

/-- Represents the number of squares in the nth row of a stair-step figure -/
def num_squares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of black squares in the nth row of a stair-step figure -/
def num_black_squares (n : ℕ) : ℕ := (num_squares n - 1) / 2

theorem black_squares_in_37th_row :
  num_black_squares 37 = 36 := by sorry

end NUMINAMATH_CALUDE_black_squares_in_37th_row_l3625_362508


namespace NUMINAMATH_CALUDE_intersection_when_m_eq_2_sufficient_not_necessary_condition_l3625_362594

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | (x-1+m)*(x-1-m) ≤ 0}

-- Theorem for part (1)
theorem intersection_when_m_eq_2 : 
  A ∩ B 2 = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

-- Theorem for part (2)
theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) ↔ m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_eq_2_sufficient_not_necessary_condition_l3625_362594


namespace NUMINAMATH_CALUDE_custom_op_result_l3625_362536

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- Theorem statement
theorem custom_op_result :
  let y : ℤ := 11
  customOp y 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l3625_362536


namespace NUMINAMATH_CALUDE_toothpick_grid_60_32_l3625_362579

/-- Calculates the total number of toothpicks in a rectangular grid -/
def total_toothpicks (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem: A 60x32 toothpick grid uses 3932 toothpicks -/
theorem toothpick_grid_60_32 :
  total_toothpicks 60 32 = 3932 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grid_60_32_l3625_362579


namespace NUMINAMATH_CALUDE_total_lives_calculation_l3625_362524

/-- Given 7 initial friends, 2 additional players, and 7 lives per player,
    the total number of lives for all players is 63. -/
theorem total_lives_calculation (initial_friends : ℕ) (additional_players : ℕ) (lives_per_player : ℕ)
    (h1 : initial_friends = 7)
    (h2 : additional_players = 2)
    (h3 : lives_per_player = 7) :
    (initial_friends + additional_players) * lives_per_player = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_calculation_l3625_362524


namespace NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achieved_l3625_362555

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 8 := by
  sorry

theorem max_sum_achieved : ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achieved_l3625_362555


namespace NUMINAMATH_CALUDE_garden_sprinkler_morning_usage_garden_sprinkler_conditions_l3625_362520

/-- A sprinkler system that waters a desert garden twice daily -/
structure SprinklerSystem where
  morning_usage : ℝ
  evening_usage : ℝ
  days : ℕ
  total_usage : ℝ

/-- The specific sprinkler system described in the problem -/
def garden_sprinkler : SprinklerSystem where
  morning_usage := 4  -- This is what we want to prove
  evening_usage := 6
  days := 5
  total_usage := 50

/-- Theorem stating that the morning usage of the garden sprinkler is 4 liters -/
theorem garden_sprinkler_morning_usage :
  garden_sprinkler.morning_usage = 4 :=
by sorry

/-- Theorem proving that the given conditions are satisfied by the garden sprinkler -/
theorem garden_sprinkler_conditions :
  garden_sprinkler.evening_usage = 6 ∧
  garden_sprinkler.days = 5 ∧
  garden_sprinkler.total_usage = 50 ∧
  garden_sprinkler.days * (garden_sprinkler.morning_usage + garden_sprinkler.evening_usage) = garden_sprinkler.total_usage :=
by sorry

end NUMINAMATH_CALUDE_garden_sprinkler_morning_usage_garden_sprinkler_conditions_l3625_362520


namespace NUMINAMATH_CALUDE_crew_average_weight_increase_l3625_362548

theorem crew_average_weight_increase (initial_average : ℝ) : 
  let initial_total_weight := 20 * initial_average
  let new_total_weight := initial_total_weight + (80 - 40)
  let new_average := new_total_weight / 20
  new_average - initial_average = 2 := by
sorry

end NUMINAMATH_CALUDE_crew_average_weight_increase_l3625_362548


namespace NUMINAMATH_CALUDE_at_least_one_boy_and_girl_l3625_362588

def probability_boy_or_girl : ℚ := 1 / 2

def number_of_children : ℕ := 4

theorem at_least_one_boy_and_girl :
  let p := probability_boy_or_girl
  let n := number_of_children
  (1 : ℚ) - (p^n + (1 - p)^n) = 7 / 8 := by sorry

end NUMINAMATH_CALUDE_at_least_one_boy_and_girl_l3625_362588


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_t_for_f_geq_t_squared_minus_t_l3625_362552

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 1|

-- Theorem for part I
theorem solution_set_f_leq_x : 
  {x : ℝ | f x ≤ x} = {x : ℝ | x ≥ 1/4} :=
sorry

-- Theorem for part II
theorem range_of_t_for_f_geq_t_squared_minus_t : 
  {t : ℝ | ∀ x ∈ Set.Icc (-2) (-1), f x ≥ t^2 - t} = 
  Set.Icc ((1 - Real.sqrt 5) / 2) ((1 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_t_for_f_geq_t_squared_minus_t_l3625_362552


namespace NUMINAMATH_CALUDE_candace_hiking_ratio_l3625_362577

/-- Candace's hiking scenario -/
def hiking_scenario (old_speed new_speed hike_duration blister_interval blister_slowdown : ℝ) : Prop :=
  let blisters := hike_duration / blister_interval
  let total_slowdown := blisters * blister_slowdown
  let final_new_speed := new_speed - total_slowdown
  final_new_speed / old_speed = 7 / 6

/-- The theorem representing Candace's hiking problem -/
theorem candace_hiking_ratio :
  hiking_scenario 6 11 4 2 2 :=
by
  sorry

end NUMINAMATH_CALUDE_candace_hiking_ratio_l3625_362577


namespace NUMINAMATH_CALUDE_derivative_of_odd_function_is_even_l3625_362507

theorem derivative_of_odd_function_is_even 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_odd : ∀ x, f (-x) = -f x) : 
  ∀ x, (deriv f) (-x) = (deriv f) x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_odd_function_is_even_l3625_362507


namespace NUMINAMATH_CALUDE_average_age_problem_l3625_362526

theorem average_age_problem (devin_age eden_age mom_age : ℕ) : 
  devin_age = 12 →
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  (devin_age + eden_age + mom_age) / 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_problem_l3625_362526


namespace NUMINAMATH_CALUDE_trapezoid_height_l3625_362521

/-- A trapezoid with given area and sum of diagonals has a specific height -/
theorem trapezoid_height (area : ℝ) (sum_diagonals : ℝ) (height : ℝ) :
  area = 2 →
  sum_diagonals = 4 →
  height = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_height_l3625_362521


namespace NUMINAMATH_CALUDE_lab_items_per_tech_l3625_362597

/-- Proves that each lab tech gets 14 items (coats and uniforms combined) given the problem conditions -/
theorem lab_items_per_tech (uniforms : ℕ) (coats : ℕ) (lab_techs : ℕ) : 
  uniforms = 12 →
  coats = 6 * uniforms →
  lab_techs = uniforms / 2 →
  (coats + uniforms) / lab_techs = 14 :=
by
  sorry

#check lab_items_per_tech

end NUMINAMATH_CALUDE_lab_items_per_tech_l3625_362597


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l3625_362515

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l3625_362515


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3625_362590

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) :
  1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 9 / 4 ∧
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 ∧
    1 / (a + 3 * b) + 1 / (b + 3 * c) + 1 / (c + 3 * a) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3625_362590


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3625_362542

theorem right_triangle_sides (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  (a + b + c) / 2 - a = 2/3 * r →  -- Relation derived from circle touching sides
  c = 5/3 * r →  -- Hypotenuse relation
  a * b / 2 = 2 * r →  -- Area of the triangle
  (a = 4/3 * r ∧ b = r) ∨ (a = r ∧ b = 4/3 * r) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3625_362542


namespace NUMINAMATH_CALUDE_total_passengers_l3625_362573

def bus_problem (initial_a initial_b new_a new_b : ℕ) : ℕ :=
  (initial_a + new_a) + (initial_b + new_b)

theorem total_passengers :
  bus_problem 4 7 13 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_l3625_362573


namespace NUMINAMATH_CALUDE_min_pieces_correct_l3625_362545

/-- Represents a chessboard of size n x n -/
structure Chessboard (n : ℕ) where
  size : ℕ
  size_pos : size > 0
  size_eq : size = n

/-- A piece on the chessboard -/
structure Piece (n : ℕ) where
  x : Fin n
  y : Fin n

/-- A configuration of pieces on the chessboard -/
def Configuration (n : ℕ) := List (Piece n)

/-- Checks if a configuration satisfies the line coverage property -/
def satisfiesLineCoverage (n : ℕ) (config : Configuration n) : Prop := sorry

/-- The minimum number of pieces required for a valid configuration -/
def minPieces (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 * n else 2 * n + 1

/-- Theorem stating the minimum number of pieces required for a valid configuration -/
theorem min_pieces_correct (n : ℕ) (h : n > 0) :
  ∀ (config : Configuration n),
    satisfiesLineCoverage n config →
    config.length ≥ minPieces n :=
  sorry

end NUMINAMATH_CALUDE_min_pieces_correct_l3625_362545


namespace NUMINAMATH_CALUDE_sphere_tangent_planes_properties_l3625_362570

/-- Given a sphere with radius r, this theorem proves various geometric properties related to
    tangent planes, spherical caps, and conical frustums. -/
theorem sphere_tangent_planes_properties (r : ℝ) (hr : r > 0) :
  ∃ (locus_radius : ℝ) (cap_area conical_area : ℝ),
    -- The locus of points P forms a sphere with radius r√3
    locus_radius = r * Real.sqrt 3 ∧
    -- The surface area of the smaller spherical cap
    cap_area = 2 * Real.pi * r^2 * (1 - Real.sqrt (2/3)) ∧
    -- The surface area of the conical frustum
    conical_area = Real.pi * r^2 * (2 * Real.sqrt 3 / 3) ∧
    -- The ratio of the two surface areas
    cap_area / conical_area = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_tangent_planes_properties_l3625_362570


namespace NUMINAMATH_CALUDE_function_simplification_l3625_362580

theorem function_simplification (x : ℝ) : 
  Real.sqrt (4 * Real.sin x ^ 4 - 2 * Real.cos (2 * x) + 3) + 
  Real.sqrt (4 * Real.cos x ^ 4 + 2 * Real.cos (2 * x) + 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_simplification_l3625_362580


namespace NUMINAMATH_CALUDE_grade_assignments_12_4_l3625_362537

/-- The number of ways to assign grades to students -/
def gradeAssignments (numStudents : ℕ) (numGrades : ℕ) : ℕ :=
  numGrades ^ numStudents

/-- Theorem: The number of ways to assign 4 possible grades to 12 students is 16777216 -/
theorem grade_assignments_12_4 : gradeAssignments 12 4 = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignments_12_4_l3625_362537


namespace NUMINAMATH_CALUDE_derivative_of_y_l3625_362566

noncomputable def y (x : ℝ) : ℝ := (1 + Real.cos (2 * x))^3

theorem derivative_of_y (x : ℝ) :
  deriv y x = -48 * (Real.cos x)^5 * Real.sin x := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l3625_362566


namespace NUMINAMATH_CALUDE_third_grade_girls_sample_l3625_362551

theorem third_grade_girls_sample (total_students : ℕ) (first_grade : ℕ) (second_grade : ℕ) (third_grade : ℕ)
  (first_boys : ℕ) (first_girls : ℕ) (second_boys : ℕ) (second_girls : ℕ) (third_boys : ℕ) (third_girls : ℕ)
  (sample_size : ℕ) :
  total_students = 3000 →
  first_grade = 800 →
  second_grade = 1000 →
  third_grade = 1200 →
  first_boys = 500 →
  first_girls = 300 →
  second_boys = 600 →
  second_girls = 400 →
  third_boys = 800 →
  third_girls = 400 →
  sample_size = 150 →
  first_grade + second_grade + third_grade = total_students →
  first_boys + first_girls = first_grade →
  second_boys + second_girls = second_grade →
  third_boys + third_girls = third_grade →
  (third_grade : ℚ) / (total_students : ℚ) * (sample_size : ℚ) * (third_girls : ℚ) / (third_grade : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_grade_girls_sample_l3625_362551


namespace NUMINAMATH_CALUDE_log10_graph_property_l3625_362539

-- Define the logarithm function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the condition for a point to be on the graph of y = log₁₀ x
def on_log10_graph (p : ℝ × ℝ) : Prop :=
  p.2 = log10 p.1

-- State the theorem
theorem log10_graph_property (a b : ℝ) (h1 : on_log10_graph (a, b)) (h2 : a ≠ 1) :
  on_log10_graph (a^2, 2*b) :=
sorry

end NUMINAMATH_CALUDE_log10_graph_property_l3625_362539


namespace NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l3625_362534

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 :=
sorry

end NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l3625_362534


namespace NUMINAMATH_CALUDE_apple_stack_theorem_l3625_362544

/-- Calculates the number of apples in a cubic-like stack --/
def appleStack (baseSize : Nat) : Nat :=
  let numLayers := baseSize
  List.range numLayers
    |> List.map (fun i => (baseSize - i) ^ 3)
    |> List.sum

theorem apple_stack_theorem :
  appleStack 4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_apple_stack_theorem_l3625_362544


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3625_362587

theorem cube_surface_area_from_volume (V : ℝ) (h : V = 64) :
  ∃ (a : ℝ), a > 0 ∧ a^3 = V ∧ 6 * a^2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3625_362587


namespace NUMINAMATH_CALUDE_base4_to_decimal_equality_l3625_362510

/-- Converts a base 4 number represented as a list of digits to its decimal (base 10) equivalent. -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number we want to convert. -/
def base4Number : List Nat := [3, 0, 1, 2, 1]

/-- Theorem stating that the base 4 number 30121₄ is equal to 793 in base 10. -/
theorem base4_to_decimal_equality :
  base4ToDecimal base4Number = 793 := by
  sorry

end NUMINAMATH_CALUDE_base4_to_decimal_equality_l3625_362510


namespace NUMINAMATH_CALUDE_parallelogram_angle_difference_parallelogram_angle_difference_proof_l3625_362518

/-- 
In a parallelogram with a smaller angle of 55 degrees, 
the difference between the larger and smaller angles is 70 degrees.
-/
theorem parallelogram_angle_difference : ℝ → Prop :=
  fun smaller_angle : ℝ =>
    smaller_angle = 55 →
    ∃ larger_angle : ℝ,
      smaller_angle + larger_angle = 180 ∧
      larger_angle - smaller_angle = 70

-- The proof is omitted
theorem parallelogram_angle_difference_proof : 
  parallelogram_angle_difference 55 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_difference_parallelogram_angle_difference_proof_l3625_362518


namespace NUMINAMATH_CALUDE_digit_swap_difference_multiple_of_nine_l3625_362564

theorem digit_swap_difference_multiple_of_nine (a b : ℕ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) : 
  ∃ k : ℤ, (10 * a + b) - (10 * b + a) = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_digit_swap_difference_multiple_of_nine_l3625_362564


namespace NUMINAMATH_CALUDE_square_side_length_l3625_362535

/-- The radius of the circles -/
def r : ℝ := 1000

/-- The side length of the square -/
def square_side : ℝ := 400

/-- Two circles touch each other and a horizontal line is tangent to both circles -/
axiom circles_touch_and_tangent_to_line : True

/-- A square fits snugly between the horizontal line and the two circles -/
axiom square_fits_snugly : True

/-- The theorem stating that the side length of the square is 400 -/
theorem square_side_length : 
  square_side = 400 :=
sorry

end NUMINAMATH_CALUDE_square_side_length_l3625_362535


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3625_362567

theorem inequality_system_solution :
  ∃ (x y : ℝ), 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧ 
    (2 * x - 4 * y ≤ -3) ∧
    (x = -1/3) ∧ 
    (y = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3625_362567


namespace NUMINAMATH_CALUDE_deepak_age_l3625_362581

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 10 = 26 →
  deepak_age = 12 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l3625_362581


namespace NUMINAMATH_CALUDE_interest_rate_difference_l3625_362506

/-- Given a principal amount, time period, and difference in interest earned between two simple interest rates, 
    this theorem proves that the difference between these rates is 5%. -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_diff : ℝ)
  (h1 : principal = 600)
  (h2 : time = 10)
  (h3 : interest_diff = 300) :
  let rate_diff := interest_diff / (principal * time / 100)
  rate_diff = 5 := by sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l3625_362506


namespace NUMINAMATH_CALUDE_mary_performance_l3625_362540

theorem mary_performance (total_days : ℕ) (adequate_rate : ℕ) (outstanding_rate : ℕ) (total_amount : ℕ) :
  total_days = 15 ∧ 
  adequate_rate = 4 ∧ 
  outstanding_rate = 7 ∧ 
  total_amount = 85 →
  ∃ (adequate_days outstanding_days : ℕ),
    adequate_days + outstanding_days = total_days ∧
    adequate_days * adequate_rate + outstanding_days * outstanding_rate = total_amount ∧
    outstanding_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_mary_performance_l3625_362540


namespace NUMINAMATH_CALUDE_first_person_speed_l3625_362511

/-- Two persons walk in opposite directions for a given time, ending up at a specific distance apart. -/
def opposite_walk (x : ℝ) (time : ℝ) (distance : ℝ) : Prop :=
  (x + 7) * time = distance

/-- The theorem states that given the conditions of the problem, the speed of the first person is 6 km/hr. -/
theorem first_person_speed : ∃ x : ℝ, opposite_walk x 3.5 45.5 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_first_person_speed_l3625_362511


namespace NUMINAMATH_CALUDE_xy_value_l3625_362516

theorem xy_value (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 44) : x*y = -24 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3625_362516


namespace NUMINAMATH_CALUDE_alex_score_l3625_362529

/-- Represents the number of shots attempted for each type --/
structure ShotAttempts where
  free_throws : ℕ
  three_points : ℕ
  two_points : ℕ

/-- Calculates the total points scored given the shot attempts --/
def calculate_score (attempts : ShotAttempts) : ℕ :=
  (attempts.free_throws * 8 / 10) +
  (attempts.three_points * 3 * 1 / 10) +
  (attempts.two_points * 2 * 5 / 10)

theorem alex_score :
  ∃ (attempts : ShotAttempts),
    attempts.free_throws + attempts.three_points + attempts.two_points = 40 ∧
    calculate_score attempts = 28 := by
  sorry

end NUMINAMATH_CALUDE_alex_score_l3625_362529


namespace NUMINAMATH_CALUDE_log_ten_seven_in_terms_of_p_q_l3625_362502

theorem log_ten_seven_in_terms_of_p_q (p q : ℝ) 
  (hp : Real.log 3 / Real.log 4 = p)
  (hq : Real.log 7 / Real.log 5 = q) :
  Real.log 7 / Real.log 10 = (2 * p * q + 2 * p) / (1 + 2 * p) := by
  sorry

end NUMINAMATH_CALUDE_log_ten_seven_in_terms_of_p_q_l3625_362502


namespace NUMINAMATH_CALUDE_vector_simplification_l3625_362541

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification 
  (A B C D : V) : 
  ((B - A) - (D - C)) - ((C - A) - (D - B)) = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_simplification_l3625_362541


namespace NUMINAMATH_CALUDE_simultaneous_cycle_is_twenty_l3625_362530

/-- The length of the letter sequence -/
def letter_cycle_length : ℕ := 5

/-- The length of the digit sequence -/
def digit_cycle_length : ℕ := 4

/-- The number of cycles needed for both sequences to return to their original state simultaneously -/
def simultaneous_cycle : ℕ := Nat.lcm letter_cycle_length digit_cycle_length

theorem simultaneous_cycle_is_twenty : simultaneous_cycle = 20 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_cycle_is_twenty_l3625_362530


namespace NUMINAMATH_CALUDE_problem_solution_l3625_362589

theorem problem_solution (x y z : ℕ) : 
  x > 0 ∧ 
  x = 10 * y + 3 ∧ 
  2 * x = 7 * (3 * y) + 1 ∧ 
  3 * x = 5 * z + 2 → 
  11 * y - x + 7 * z = 219 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3625_362589


namespace NUMINAMATH_CALUDE_sallys_gold_card_balance_fraction_l3625_362550

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard

/-- The conditions of Sally's credit cards -/
def sallys_cards_conditions (cards : SallysCards) : Prop :=
  cards.platinum.limit = 2 * cards.gold.limit ∧
  cards.platinum.balance = (1 / 6) * cards.platinum.limit ∧
  cards.platinum.balance + cards.gold.balance = (1 / 3) * cards.platinum.limit

/-- The theorem representing the problem -/
theorem sallys_gold_card_balance_fraction (cards : SallysCards) 
  (h : sallys_cards_conditions cards) : 
  cards.gold.balance = (1 / 3) * cards.gold.limit := by
  sorry

end NUMINAMATH_CALUDE_sallys_gold_card_balance_fraction_l3625_362550


namespace NUMINAMATH_CALUDE_total_friends_l3625_362598

/-- The number of friends who attended the movie -/
def M : ℕ := 10

/-- The number of friends who attended the picnic -/
def P : ℕ := 20

/-- The number of friends who attended the games -/
def G : ℕ := 5

/-- The number of friends who attended both movie and picnic -/
def MP : ℕ := 4

/-- The number of friends who attended both movie and games -/
def MG : ℕ := 2

/-- The number of friends who attended both picnic and games -/
def PG : ℕ := 0

/-- The number of friends who attended all three events -/
def MPG : ℕ := 2

/-- The total number of unique friends -/
def N : ℕ := M + P + G - MP - MG - PG + MPG

theorem total_friends : N = 31 := by sorry

end NUMINAMATH_CALUDE_total_friends_l3625_362598


namespace NUMINAMATH_CALUDE_binary_ones_factorial_divisibility_l3625_362500

-- Define a function to count the number of ones in the binary representation of a natural number
def countOnes (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem binary_ones_factorial_divisibility (n : ℕ) (h : n > 0) (h_ones : countOnes n = 1995) :
  (2^(n - 1995) : ℕ) ∣ n! :=
sorry

end NUMINAMATH_CALUDE_binary_ones_factorial_divisibility_l3625_362500


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3625_362575

theorem binomial_square_constant (a : ℚ) : 
  (∃ b c : ℚ, ∀ x, 9 * x^2 + 21 * x + a = (b * x + c)^2) → a = 49 / 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3625_362575


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3625_362560

theorem quadrilateral_area (S_ABCD S_OKSL S_ONAM S_OMBK : ℝ) 
  (h1 : S_ABCD = 4 * (S_OKSL + S_ONAM))
  (h2 : S_OKSL = 6)
  (h3 : S_ONAM = 12)
  (h4 : S_OMBK = S_ABCD - S_OKSL - 24 - S_ONAM) :
  S_ABCD = 72 ∧ S_OMBK = 30 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3625_362560


namespace NUMINAMATH_CALUDE_A_equals_two_three_l3625_362553

def A : Set ℤ := {x | (3 : ℚ) / (x - 1) > 1}

theorem A_equals_two_three : A = {2, 3} := by sorry

end NUMINAMATH_CALUDE_A_equals_two_three_l3625_362553


namespace NUMINAMATH_CALUDE_expression_evaluation_l3625_362586

theorem expression_evaluation : 
  let x : ℝ := 2
  let y : ℝ := -1
  (2*x - y)^2 + (x - 2*y) * (x + 2*y) = 25 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3625_362586
