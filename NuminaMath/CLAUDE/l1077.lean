import Mathlib

namespace NUMINAMATH_CALUDE_percentage_problem_l1077_107707

theorem percentage_problem (x : ℝ) :
  (15 / 100) * (30 / 100) * (50 / 100) * x = 108 → x = 4800 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1077_107707


namespace NUMINAMATH_CALUDE_adult_ticket_price_l1077_107717

theorem adult_ticket_price (total_tickets : ℕ) (senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) :
  total_tickets = 510 →
  senior_price = 15 →
  total_receipts = 8748 →
  senior_tickets = 327 →
  ∃ (adult_price : ℕ), adult_price = 21 ∧ 
    total_receipts = senior_price * senior_tickets + adult_price * (total_tickets - senior_tickets) :=
by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l1077_107717


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1077_107779

def set_A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def set_B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_implies_a_value :
  ∀ a : ℝ, set_A a ∩ set_B a = {9} → a = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1077_107779


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1077_107714

theorem quadratic_equation_solution :
  let a : ℝ := 2
  let b : ℝ := -5
  let c : ℝ := 3
  let x₁ : ℝ := 3/2
  let x₂ : ℝ := 1
  (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_solution_l1077_107714


namespace NUMINAMATH_CALUDE_employee_salary_calculation_l1077_107703

/-- Proves that given two employees with a total weekly pay of $572, 
    where one employee's salary is 120% of the other's, 
    the lower-paid employee's weekly salary is $260. -/
theorem employee_salary_calculation (total_pay n_salary : ℝ) : 
  total_pay = 572 →
  total_pay = n_salary + 1.2 * n_salary →
  n_salary = 260 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_calculation_l1077_107703


namespace NUMINAMATH_CALUDE_fraction_simplification_l1077_107765

theorem fraction_simplification :
  (6 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + 2 * Real.sqrt 18) = (3 * Real.sqrt 2) / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1077_107765


namespace NUMINAMATH_CALUDE_cranberry_juice_price_per_ounce_l1077_107761

/-- Given a can of cranberry juice with volume in ounces and price in cents,
    calculate the price per ounce in cents. -/
def price_per_ounce (volume : ℕ) (price : ℕ) : ℚ :=
  price / volume

/-- Theorem stating that the price per ounce of cranberry juice is 7 cents
    given that a 12 ounce can sells for 84 cents. -/
theorem cranberry_juice_price_per_ounce :
  price_per_ounce 12 84 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cranberry_juice_price_per_ounce_l1077_107761


namespace NUMINAMATH_CALUDE_min_third_side_right_triangle_l1077_107776

theorem min_third_side_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a = 8 ∧ b = 15) ∨ (a = 8 ∧ c = 15) ∨ (b = 8 ∧ c = 15) →
  a^2 + b^2 = c^2 →
  min a (min b c) ≥ Real.sqrt 161 :=
by sorry

end NUMINAMATH_CALUDE_min_third_side_right_triangle_l1077_107776


namespace NUMINAMATH_CALUDE_middle_number_problem_l1077_107724

theorem middle_number_problem (x y z : ℝ) 
  (h_order : x < y ∧ y < z)
  (h_sum1 : x + y = 24)
  (h_sum2 : x + z = 29)
  (h_sum3 : y + z = 34) :
  y = 14.5 := by
sorry

end NUMINAMATH_CALUDE_middle_number_problem_l1077_107724


namespace NUMINAMATH_CALUDE_price_per_cup_l1077_107758

/-- Represents the number of trees each sister has -/
def trees : ℕ := 110

/-- Represents the number of oranges Gabriela's trees produce per tree -/
def gabriela_oranges_per_tree : ℕ := 600

/-- Represents the number of oranges Alba's trees produce per tree -/
def alba_oranges_per_tree : ℕ := 400

/-- Represents the number of oranges Maricela's trees produce per tree -/
def maricela_oranges_per_tree : ℕ := 500

/-- Represents the number of oranges needed to make one cup of juice -/
def oranges_per_cup : ℕ := 3

/-- Represents the total earnings from selling the juice -/
def total_earnings : ℕ := 220000

/-- Calculates the total number of oranges harvested by all sisters -/
def total_oranges : ℕ := 
  trees * gabriela_oranges_per_tree + 
  trees * alba_oranges_per_tree + 
  trees * maricela_oranges_per_tree

/-- Calculates the total number of cups of juice that can be made -/
def total_cups : ℕ := total_oranges / oranges_per_cup

/-- Theorem stating that the price per cup of juice is $4 -/
theorem price_per_cup : total_earnings / total_cups = 4 := by
  sorry

end NUMINAMATH_CALUDE_price_per_cup_l1077_107758


namespace NUMINAMATH_CALUDE_simplify_expression_l1077_107726

theorem simplify_expression (w : ℝ) : 4*w + 6*w + 8*w + 10*w + 12*w + 24 = 40*w + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1077_107726


namespace NUMINAMATH_CALUDE_shovel_time_closest_to_17_l1077_107720

/-- Represents the snow shoveling problem --/
structure SnowShoveling where
  /-- Initial shoveling rate in cubic yards per hour --/
  initial_rate : ℕ
  /-- Decrease in shoveling rate per hour --/
  rate_decrease : ℕ
  /-- Break duration in hours --/
  break_duration : ℚ
  /-- Hours of shoveling before a break --/
  hours_before_break : ℕ
  /-- Driveway width in yards --/
  driveway_width : ℕ
  /-- Driveway length in yards --/
  driveway_length : ℕ
  /-- Snow depth in yards --/
  snow_depth : ℕ

/-- Calculates the time taken to shovel the driveway clean, including breaks --/
def time_to_shovel (problem : SnowShoveling) : ℚ :=
  sorry

/-- Theorem stating that the time taken to shovel the driveway is closest to 17 hours --/
theorem shovel_time_closest_to_17 (problem : SnowShoveling) 
  (h1 : problem.initial_rate = 25)
  (h2 : problem.rate_decrease = 1)
  (h3 : problem.break_duration = 1/2)
  (h4 : problem.hours_before_break = 2)
  (h5 : problem.driveway_width = 5)
  (h6 : problem.driveway_length = 12)
  (h7 : problem.snow_depth = 4) :
  ∃ (t : ℚ), time_to_shovel problem = t ∧ abs (t - 17) < abs (t - 14) ∧ 
             abs (t - 17) < abs (t - 15) ∧ abs (t - 17) < abs (t - 16) ∧ 
             abs (t - 17) < abs (t - 18) :=
  sorry

end NUMINAMATH_CALUDE_shovel_time_closest_to_17_l1077_107720


namespace NUMINAMATH_CALUDE_negp_sufficient_not_necessary_for_negq_l1077_107722

def p (x : ℝ) : Prop := x < -1 ∨ x > 1

def q (x : ℝ) : Prop := x < -2 ∨ x > 1

theorem negp_sufficient_not_necessary_for_negq :
  (∀ x, ¬(p x) → ¬(q x)) ∧ ¬(∀ x, ¬(q x) → ¬(p x)) := by sorry

end NUMINAMATH_CALUDE_negp_sufficient_not_necessary_for_negq_l1077_107722


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1077_107744

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + 2*m - 2022 = 0) → 
  (n^2 + 2*n - 2022 = 0) → 
  (m^2 + 3*m + n = 2020) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1077_107744


namespace NUMINAMATH_CALUDE_f_max_value_f_solution_set_max_ab_plus_bc_l1077_107730

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| - 2 * |x + 1|

-- Theorem for the maximum value of f
theorem f_max_value : ∃ m : ℝ, m = 4 ∧ ∀ x : ℝ, f x ≤ m :=
sorry

-- Theorem for the solution set of f(x) < 1
theorem f_solution_set : ∀ x : ℝ, f x < 1 ↔ x < -4 ∨ x > 0 :=
sorry

-- Theorem for the maximum value of ab + bc
theorem max_ab_plus_bc :
  ∀ a b c : ℝ, a > 0 → b > 0 → a^2 + 2*b^2 + c^2 = 4 →
  ∃ max : ℝ, max = 2 ∧ a*b + b*c ≤ max :=
sorry

end NUMINAMATH_CALUDE_f_max_value_f_solution_set_max_ab_plus_bc_l1077_107730


namespace NUMINAMATH_CALUDE_bakery_sugar_amount_l1077_107706

/-- Given the ratios of ingredients in a bakery storage room, 
    prove the amount of sugar stored. -/
theorem bakery_sugar_amount 
  (sugar flour baking_soda : ℕ) 
  (h1 : sugar = flour)  -- sugar to flour ratio is 5:5, which simplifies to 1:1
  (h2 : flour = 10 * baking_soda)  -- flour to baking soda ratio is 10:1
  (h3 : flour = 8 * (baking_soda + 60))  -- if 60 more pounds of baking soda, ratio would be 8:1
  : sugar = 2400 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sugar_amount_l1077_107706


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_l1077_107701

theorem polynomial_roots_sum (c d : ℝ) : 
  c^2 - 6*c + 10 = 0 ∧ d^2 - 6*d + 10 = 0 → c^3 + c^5*d^3 + c^3*d^5 + d^3 = 16156 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_l1077_107701


namespace NUMINAMATH_CALUDE_price_restoration_l1077_107729

theorem price_restoration (original_price : ℝ) (original_price_positive : original_price > 0) :
  let reduced_price := original_price * (1 - 0.15)
  let restoration_factor := (1 + 0.1765)
  (reduced_price * restoration_factor - original_price) / original_price < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_price_restoration_l1077_107729


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l1077_107780

theorem sum_of_roots_zero (z₁ z₂ z₃ : ℝ) : 
  (4096 * z₁^3 + 16 * z₁ - 9 = 0) →
  (4096 * z₂^3 + 16 * z₂ - 9 = 0) →
  (4096 * z₃^3 + 16 * z₃ - 9 = 0) →
  z₁ + z₂ + z₃ = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l1077_107780


namespace NUMINAMATH_CALUDE_shark_sightings_l1077_107728

theorem shark_sightings (cape_may daytona_beach : ℕ) : 
  cape_may + daytona_beach = 40 →
  cape_may = 2 * daytona_beach - 8 →
  cape_may = 24 :=
by sorry

end NUMINAMATH_CALUDE_shark_sightings_l1077_107728


namespace NUMINAMATH_CALUDE_wage_decrease_theorem_l1077_107750

theorem wage_decrease_theorem (x : ℝ) : 
  (100 - x) * 1.5 = 75 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_wage_decrease_theorem_l1077_107750


namespace NUMINAMATH_CALUDE_infinitely_many_a_without_solution_l1077_107731

-- Define τ(n) as the number of positive divisors of n
def tau (n : ℕ+) : ℕ := sorry

-- Statement of the theorem
theorem infinitely_many_a_without_solution :
  ∃ (S : Set ℕ+), (Set.Infinite S) ∧ 
  (∀ (a : ℕ+), a ∈ S → ∀ (n : ℕ+), tau (a * n) ≠ n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_a_without_solution_l1077_107731


namespace NUMINAMATH_CALUDE_dividend_calculation_l1077_107764

theorem dividend_calculation (divisor : ℕ) (partial_quotient : ℕ) 
  (h1 : divisor = 12) 
  (h2 : partial_quotient = 909809) : 
  divisor * partial_quotient = 10917708 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1077_107764


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1077_107751

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = 5 - x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1077_107751


namespace NUMINAMATH_CALUDE_triangle_area_is_six_l1077_107725

-- Define the triangle vertices
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 3)

-- Define the line on which C lies
def line_C (x y : ℝ) : Prop := x + y = 7

-- Define the area of the triangle
def triangle_area (C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_is_six :
  ∀ C : ℝ × ℝ, line_C C.1 C.2 → triangle_area C = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_six_l1077_107725


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l1077_107711

theorem reciprocal_of_sum : (1 / (1/3 + 1/4) : ℚ) = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l1077_107711


namespace NUMINAMATH_CALUDE_cone_surface_area_and_volume_l1077_107788

/-- Represents a cone with given dimensions and properties -/
structure Cone where
  height : ℝ
  lateral_to_total_ratio : ℝ

/-- Calculates the surface area of the cone -/
def surface_area (c : Cone) : ℝ := sorry

/-- Calculates the volume of the cone -/
def volume (c : Cone) : ℝ := sorry

/-- Theorem stating the surface area and volume of a specific cone -/
theorem cone_surface_area_and_volume :
  let c := Cone.mk 96 (25/32)
  (surface_area c = 3584 * Real.pi) ∧ (volume c = 25088 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_and_volume_l1077_107788


namespace NUMINAMATH_CALUDE_emani_money_l1077_107760

/-- Proves that Emani has $150, given the conditions of the problem -/
theorem emani_money :
  (∀ (emani howard : ℕ),
    emani = howard + 30 →
    emani + howard = 2 * 135 →
    emani = 150) :=
by sorry

end NUMINAMATH_CALUDE_emani_money_l1077_107760


namespace NUMINAMATH_CALUDE_base_19_representation_of_1987_l1077_107787

theorem base_19_representation_of_1987 :
  ∃! (x y z b : ℕ), 
    x * b^2 + y * b + z = 1987 ∧
    x + y + z = 25 ∧
    x < b ∧ y < b ∧ z < b ∧
    x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 := by
  sorry

end NUMINAMATH_CALUDE_base_19_representation_of_1987_l1077_107787


namespace NUMINAMATH_CALUDE_optimal_container_l1077_107715

-- Define the container parameters
def volume : ℝ := 8
def length : ℝ := 2
def min_height : ℝ := 3
def bottom_cost : ℝ := 40
def lateral_cost : ℝ := 20

-- Define the cost function
def cost (width height : ℝ) : ℝ :=
  bottom_cost * length * width + lateral_cost * (2 * (length + width) * height)

-- State the theorem
theorem optimal_container :
  ∃ (width height : ℝ),
    width > 0 ∧
    height ≥ min_height ∧
    length * width * height = volume ∧
    cost width height = 1520 / 3 ∧
    width = 4 / 3 ∧
    ∀ (w h : ℝ), w > 0 → h ≥ min_height → length * w * h = volume → cost w h ≥ 1520 / 3 := by
  sorry

end NUMINAMATH_CALUDE_optimal_container_l1077_107715


namespace NUMINAMATH_CALUDE_consecutive_integral_roots_properties_l1077_107792

-- Define the properties of p and q
def is_valid_pq (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧
  ∃ (r : ℤ), (r + 1)^2 - p * (r + 1) + q = 0 ∧ r^2 - p * r + q = 0

-- Define the theorem
theorem consecutive_integral_roots_properties (p q : ℕ) (h : is_valid_pq p q) :
  (∃ (x y : ℤ), x^2 - p * x + q = 0 ∧ y^2 - p * y + q = 0 ∧ y = x + 1) →
  (Nat.Prime p) ∧
  (Nat.Prime q) ∧
  (Nat.Prime (p + q)) ∧
  ¬(Nat.Prime (p^2 - 4*q)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integral_roots_properties_l1077_107792


namespace NUMINAMATH_CALUDE_platform_length_l1077_107749

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and a signal pole in 10 seconds, prove that the platform length is 870 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_time = 39)
  (h3 : pole_time = 10) :
  let train_speed := train_length / pole_time
  let platform_length := train_speed * platform_time - train_length
  platform_length = 870 := by sorry

end NUMINAMATH_CALUDE_platform_length_l1077_107749


namespace NUMINAMATH_CALUDE_two_face_cube_probability_l1077_107735

/-- The number of small cubes a large painted cube is sawed into -/
def total_cubes : ℕ := 1000

/-- The number of edges in a cube -/
def cube_edges : ℕ := 12

/-- The number of small cubes along each edge of the large cube -/
def edge_cubes : ℕ := 10

/-- The number of small cubes with two painted faces -/
def two_face_cubes : ℕ := cube_edges * edge_cubes

/-- The probability of randomly picking a small cube with two painted faces -/
def two_face_probability : ℚ := two_face_cubes / total_cubes

theorem two_face_cube_probability :
  two_face_probability = 12 / 125 := by sorry

end NUMINAMATH_CALUDE_two_face_cube_probability_l1077_107735


namespace NUMINAMATH_CALUDE_max_quotient_value_l1077_107756

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) :
  (∀ x y, 100 ≤ x ∧ x ≤ 300 → 500 ≤ y ∧ y ≤ 1500 → y / x ≤ b / a) → b / a = 15 :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l1077_107756


namespace NUMINAMATH_CALUDE_first_discount_percentage_l1077_107748

theorem first_discount_percentage (original_price final_price : ℝ) 
  (second_discount : ℝ) (h1 : original_price = 480) 
  (h2 : final_price = 306) (h3 : second_discount = 25) : 
  ∃ (first_discount : ℝ), 
    first_discount = 15 ∧ 
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l1077_107748


namespace NUMINAMATH_CALUDE_binary_arrangements_count_l1077_107798

/-- The number of ways to arrange 3 ones and 3 zeros in a binary string -/
def binaryArrangements : ℕ := 20

/-- The length of the binary string -/
def stringLength : ℕ := 6

/-- The number of ones in the binary string -/
def numberOfOnes : ℕ := 3

theorem binary_arrangements_count :
  binaryArrangements = Nat.choose stringLength numberOfOnes := by
  sorry

end NUMINAMATH_CALUDE_binary_arrangements_count_l1077_107798


namespace NUMINAMATH_CALUDE_vector_sum_with_scalar_mult_l1077_107737

/-- Given two 2D vectors and a scalar, prove that their sum after scalar multiplication equals the expected result -/
theorem vector_sum_with_scalar_mult (v1 v2 result : Fin 2 → ℝ) (scalar : ℝ) 
  (h1 : v1 = ![4, -9])
  (h2 : v2 = ![-3, 5])
  (h3 : scalar = 2)
  (h4 : result = ![-2, 1]) :
  v1 + scalar • v2 = result :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_with_scalar_mult_l1077_107737


namespace NUMINAMATH_CALUDE_election_majority_l1077_107794

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 5200 → 
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 1040 :=
by sorry

end NUMINAMATH_CALUDE_election_majority_l1077_107794


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l1077_107772

theorem unfair_coin_probability (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (35 * p^4 * (1-p)^3 = 343/3125) →
  p = 0.7 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l1077_107772


namespace NUMINAMATH_CALUDE_total_selling_price_is_correct_l1077_107767

def cycle_price : ℕ := 2000
def scooter_price : ℕ := 25000
def bike_price : ℕ := 60000

def cycle_loss_percent : ℚ := 10 / 100
def scooter_loss_percent : ℚ := 15 / 100
def bike_loss_percent : ℚ := 5 / 100

def selling_price (price : ℕ) (loss_percent : ℚ) : ℚ :=
  price - (price * loss_percent)

def total_selling_price : ℚ :=
  selling_price cycle_price cycle_loss_percent +
  selling_price scooter_price scooter_loss_percent +
  selling_price bike_price bike_loss_percent

theorem total_selling_price_is_correct :
  total_selling_price = 80050 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_is_correct_l1077_107767


namespace NUMINAMATH_CALUDE_camping_trip_distance_l1077_107791

/-- Represents a camping trip with increasing daily distances -/
structure CampingTrip where
  dailyIncrease : ℕ  -- Daily increase in distance
  daysToGo : ℕ       -- Days to reach the destination
  daysToReturn : ℕ   -- Days to return

/-- Calculates the one-way distance for a camping trip -/
def oneWayDistance (trip : CampingTrip) : ℕ :=
  let totalDays := trip.daysToGo + trip.daysToReturn
  let lastDayDistance := totalDays - 1
  (lastDayDistance * (lastDayDistance + 1) - trip.daysToReturn * (trip.daysToReturn - 1)) / 2

/-- Theorem: The one-way distance for the given camping trip is 42 km -/
theorem camping_trip_distance :
  oneWayDistance { dailyIncrease := 1, daysToGo := 4, daysToReturn := 3 } = 42 := by
  sorry

end NUMINAMATH_CALUDE_camping_trip_distance_l1077_107791


namespace NUMINAMATH_CALUDE_chord_line_equation_l1077_107768

/-- The equation of a line passing through a chord of an ellipse -/
theorem chord_line_equation (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁^2 / 36 + y₁^2 / 9 = 1) →
  (x₂^2 / 36 + y₂^2 / 9 = 1) →
  ((x₁ + x₂) / 2 = 4) →
  ((y₁ + y₂) / 2 = 2) →
  (∀ x y : ℝ, y - 2 = -(1/2) * (x - 4) ↔ x + 2*y - 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l1077_107768


namespace NUMINAMATH_CALUDE_parabola_focus_l1077_107721

/-- The parabola defined by y = 2x² -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The theorem stating that (0, -1/8) is the focus of the parabola y = 2x² -/
theorem parabola_focus :
  ∃ (f : Focus), f.x = 0 ∧ f.y = -1/8 ∧
  ∀ (x y : ℝ), parabola x y →
    (x - f.x)^2 + (y - f.y)^2 = (y + 1/8)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1077_107721


namespace NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1077_107763

theorem richmond_tigers_ticket_sales (total_tickets : ℕ) (first_half_tickets : ℕ) (second_half_tickets : ℕ) :
  total_tickets = 9570 →
  first_half_tickets = 3867 →
  second_half_tickets = total_tickets - first_half_tickets →
  second_half_tickets = 5703 :=
by
  sorry

end NUMINAMATH_CALUDE_richmond_tigers_ticket_sales_l1077_107763


namespace NUMINAMATH_CALUDE_inverse_variation_example_l1077_107746

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_variation_example :
  ∀ x y : ℝ → ℝ,
  VaryInversely x y →
  y 1500 = 0.4 →
  y 3000 = 0.2 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_example_l1077_107746


namespace NUMINAMATH_CALUDE_green_triangle_cost_l1077_107773

/-- Calculates the cost of greening a right-angled triangle -/
theorem green_triangle_cost 
  (a b c : ℝ) 
  (cost_per_sqm : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_a : a = 8) 
  (h_b : b = 15) 
  (h_c : c = 17) 
  (h_cost : cost_per_sqm = 50) : 
  (1/2 * a * b) * cost_per_sqm = 3000 := by
sorry

end NUMINAMATH_CALUDE_green_triangle_cost_l1077_107773


namespace NUMINAMATH_CALUDE_nationality_deduction_l1077_107736

-- Define the types for people and nationalities
inductive Person : Type
  | A | B | C | D

inductive Nationality : Type
  | UK | US | Germany | France

-- Define the occupations
inductive Occupation : Type
  | Doctor | Teacher

-- Define the predicates
def isNationalityOf : Person → Nationality → Prop := sorry
def hasOccupation : Person → Occupation → Prop := sorry
def canSwim : Person → Prop := sorry
def playsSportsWith : Person → Person → Prop := sorry

-- State the theorem
theorem nationality_deduction :
  -- Condition 1 and 2 are implicitly handled by the type definitions
  -- Condition 3
  (∀ p : Person, ∃! o : Occupation, hasOccupation p o) →
  -- Condition 4
  (hasOccupation Person.A Occupation.Doctor ∧
   ∃ p : Person, isNationalityOf p Nationality.US ∧ hasOccupation p Occupation.Doctor) →
  -- Condition 5
  (hasOccupation Person.B Occupation.Teacher ∧
   ∃ p : Person, isNationalityOf p Nationality.Germany ∧ hasOccupation p Occupation.Teacher) →
  -- Condition 6
  (canSwim Person.C ∧
   ∀ p : Person, isNationalityOf p Nationality.Germany → ¬canSwim p) →
  -- Condition 7
  (∃ p : Person, isNationalityOf p Nationality.France ∧ playsSportsWith Person.A p) →
  -- Conclusion
  (isNationalityOf Person.A Nationality.UK ∧ isNationalityOf Person.D Nationality.Germany) :=
by
  sorry

end NUMINAMATH_CALUDE_nationality_deduction_l1077_107736


namespace NUMINAMATH_CALUDE_pete_backward_speed_calculation_l1077_107740

/-- Pete's backward walking speed in miles per hour -/
def pete_backward_speed : ℝ := 12

/-- Susan's forward walking speed in miles per hour -/
def susan_forward_speed : ℝ := 4

/-- Tracy's cartwheel speed in miles per hour -/
def tracy_cartwheel_speed : ℝ := 8

/-- Pete's hand-walking speed in miles per hour -/
def pete_hand_speed : ℝ := 2

theorem pete_backward_speed_calculation :
  (pete_backward_speed = 3 * susan_forward_speed) ∧
  (tracy_cartwheel_speed = 2 * susan_forward_speed) ∧
  (pete_hand_speed = (1/4) * tracy_cartwheel_speed) ∧
  (pete_hand_speed = 2) →
  pete_backward_speed = 12 := by
sorry

end NUMINAMATH_CALUDE_pete_backward_speed_calculation_l1077_107740


namespace NUMINAMATH_CALUDE_vector_simplification_l1077_107733

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem vector_simplification :
  (1 / 2 : ℝ) • ((2 : ℝ) • a + (8 : ℝ) • b) - ((4 : ℝ) • a - (2 : ℝ) • b) = (6 : ℝ) • b - (3 : ℝ) • a :=
by sorry

end NUMINAMATH_CALUDE_vector_simplification_l1077_107733


namespace NUMINAMATH_CALUDE_tiles_for_dining_room_l1077_107745

/-- Calculates the number of tiles needed for a rectangular room with a border --/
def tiles_needed (room_length room_width border_width : ℕ) 
  (small_tile_size large_tile_size : ℕ) : ℕ :=
  let border_tiles := 
    2 * (2 * (room_length - 2 * border_width) + 2 * (room_width - 2 * border_width)) + 
    4 * border_width * border_width / (small_tile_size * small_tile_size)
  let inner_area := (room_length - 2 * border_width) * (room_width - 2 * border_width)
  let large_tiles := (inner_area + large_tile_size * large_tile_size - 1) / 
    (large_tile_size * large_tile_size)
  border_tiles + large_tiles

/-- Theorem stating that for the given room dimensions and tile sizes, 
    the total number of tiles needed is 144 --/
theorem tiles_for_dining_room : 
  tiles_needed 20 15 2 1 3 = 144 := by sorry

end NUMINAMATH_CALUDE_tiles_for_dining_room_l1077_107745


namespace NUMINAMATH_CALUDE_max_students_distribution_l1077_107732

theorem max_students_distribution (pens pencils erasers notebooks : ℕ) 
  (h_pens : pens = 4261)
  (h_pencils : pencils = 2677)
  (h_erasers : erasers = 1759)
  (h_notebooks : notebooks = 1423) :
  (∃ (n : ℕ), n > 0 ∧ 
    pens % n = 0 ∧ pencils % n = 0 ∧ erasers % n = 0 ∧ notebooks % n = 0 ∧
    (∀ m : ℕ, m > n → (pens % m ≠ 0 ∨ pencils % m ≠ 0 ∨ erasers % m ≠ 0 ∨ notebooks % m ≠ 0))) →
  1 = Nat.gcd (Nat.gcd (Nat.gcd pens pencils) erasers) notebooks :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l1077_107732


namespace NUMINAMATH_CALUDE_james_age_is_35_l1077_107738

/-- The age James turned when John turned 35 -/
def james_age : ℕ := sorry

/-- John's age when James turned james_age -/
def john_age : ℕ := 35

/-- Tim's current age -/
def tim_age : ℕ := 79

theorem james_age_is_35 : james_age = 35 :=
  by
    have h1 : tim_age = 2 * john_age - 5 := by sorry
    have h2 : james_age = john_age := by sorry
    sorry

#check james_age_is_35

end NUMINAMATH_CALUDE_james_age_is_35_l1077_107738


namespace NUMINAMATH_CALUDE_fraction_simplification_l1077_107795

theorem fraction_simplification (b x : ℝ) (h : b^2 + x^4 ≠ 0) :
  (Real.sqrt (b^2 + x^4) - (x^4 - b^2) / (2 * Real.sqrt (b^2 + x^4))) / (b^2 + x^4) =
  (3 * b^2 + x^4) / (2 * (b^2 + x^4)^(3/2)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1077_107795


namespace NUMINAMATH_CALUDE_otimes_equation_solution_l1077_107705

/-- Custom operation ⊗ for real numbers -/
def otimes (a b : ℝ) : ℝ := a^2 + b^2 - a*b

/-- Theorem stating that if x ⊗ (x-1) = 3, then x = 2 or x = -1 -/
theorem otimes_equation_solution (x : ℝ) : 
  otimes x (x - 1) = 3 → x = 2 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_otimes_equation_solution_l1077_107705


namespace NUMINAMATH_CALUDE_no_odd_total_students_l1077_107710

theorem no_odd_total_students (B : ℕ) (T : ℕ) : 
  (T = B + (7.25 * B : ℚ).floor) → 
  (50 ≤ T ∧ T ≤ 150) → 
  ¬(T % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_odd_total_students_l1077_107710


namespace NUMINAMATH_CALUDE_sum_of_nineteen_terms_l1077_107700

/-- Given an arithmetic sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) : ℝ := sorry

/-- {a_n} is an arithmetic sequence -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop := sorry

/-- The second, ninth, and nineteenth terms of the sequence sum to 6 -/
axiom sum_condition (a : ℕ → ℝ) : a 2 + a 9 + a 19 = 6

theorem sum_of_nineteen_terms (a : ℕ → ℝ) (h : isArithmeticSequence a) : 
  S 19 = 38 := by sorry

end NUMINAMATH_CALUDE_sum_of_nineteen_terms_l1077_107700


namespace NUMINAMATH_CALUDE_room_length_with_veranda_l1077_107782

/-- Represents a rectangular room with a surrounding veranda -/
structure RoomWithVeranda where
  roomLength : ℝ
  roomWidth : ℝ
  verandaWidth : ℝ

/-- Calculates the area of the veranda -/
def verandaArea (r : RoomWithVeranda) : ℝ :=
  (r.roomLength + 2 * r.verandaWidth) * (r.roomWidth + 2 * r.verandaWidth) - r.roomLength * r.roomWidth

theorem room_length_with_veranda (r : RoomWithVeranda) :
  r.roomWidth = 12 ∧ r.verandaWidth = 2 ∧ verandaArea r = 144 → r.roomLength = 20 := by
  sorry

end NUMINAMATH_CALUDE_room_length_with_veranda_l1077_107782


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l1077_107775

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l1077_107775


namespace NUMINAMATH_CALUDE_workshop_schedule_l1077_107774

theorem workshop_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 10))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_workshop_schedule_l1077_107774


namespace NUMINAMATH_CALUDE_equation_solution_l1077_107799

theorem equation_solution (x y : ℝ) (h : 17 * x + 51 * y = 102) : 9 * x + 27 * y = 54 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1077_107799


namespace NUMINAMATH_CALUDE_teacher_weight_l1077_107747

theorem teacher_weight (num_students : ℕ) (student_avg_weight : ℝ) (avg_increase : ℝ) :
  num_students = 24 →
  student_avg_weight = 35 →
  avg_increase = 0.4 →
  let total_student_weight := num_students * student_avg_weight
  let new_avg := student_avg_weight + avg_increase
  let total_weight_with_teacher := new_avg * (num_students + 1)
  total_weight_with_teacher - total_student_weight = 45 := by
  sorry

end NUMINAMATH_CALUDE_teacher_weight_l1077_107747


namespace NUMINAMATH_CALUDE_subtract_percentage_equivalent_to_multiply_l1077_107755

theorem subtract_percentage_equivalent_to_multiply (a : ℝ) : 
  a - (0.04 * a) = 0.96 * a := by sorry

end NUMINAMATH_CALUDE_subtract_percentage_equivalent_to_multiply_l1077_107755


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1077_107781

theorem quadratic_inequality (x : ℝ) : -15 * x^2 + 10 * x + 5 > 0 ↔ -1/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1077_107781


namespace NUMINAMATH_CALUDE_sales_difference_prove_sales_difference_l1077_107754

def morning_sales (remy_bottles : ℕ) (nick_bottles : ℕ) (price_per_bottle : ℚ) : ℚ :=
  (remy_bottles + nick_bottles) * price_per_bottle

theorem sales_difference (remy_morning_bottles : ℕ) (evening_sales : ℚ) : ℚ :=
  let nick_morning_bottles : ℕ := remy_morning_bottles - 6
  let price_per_bottle : ℚ := 1/2
  let morning_total : ℚ := morning_sales remy_morning_bottles nick_morning_bottles price_per_bottle
  evening_sales - morning_total

theorem prove_sales_difference : sales_difference 55 55 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sales_difference_prove_sales_difference_l1077_107754


namespace NUMINAMATH_CALUDE_distance_before_collision_l1077_107713

/-- Theorem: Distance between boats 3 minutes before collision -/
theorem distance_before_collision
  (river_current : ℝ)
  (boat1_speed : ℝ)
  (boat2_speed : ℝ)
  (initial_distance : ℝ)
  (h1 : river_current = 2)
  (h2 : boat1_speed = 5)
  (h3 : boat2_speed = 25)
  (h4 : initial_distance = 20) :
  let relative_speed := (boat1_speed - river_current) + (boat2_speed - river_current)
  let time_before_collision : ℝ := 3 / 60
  let distance_covered := relative_speed * time_before_collision
  initial_distance - distance_covered = 1.3 := by
  sorry

#check distance_before_collision

end NUMINAMATH_CALUDE_distance_before_collision_l1077_107713


namespace NUMINAMATH_CALUDE_field_constraint_l1077_107704

def field_area (b : ℤ) : ℚ :=
  let a := -b / 2
  2 * (5 / (2 * (2 - b))) * (5 / (2 * 3))

theorem field_constraint (b : ℤ) :
  b ≥ -4 ∧ b ≤ 4 →
  (∀ x y : ℚ, 2 * |2 * |x| - b| + |6 * |y| + 3| = 12) →
  field_area b > 0 →
  field_area b ≤ 10 / 3 →
  b = -2 ∧ -b / 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_field_constraint_l1077_107704


namespace NUMINAMATH_CALUDE_ray_reflection_l1077_107784

/-- Given a point A, a line l, and a point B, prove the equations of the incident and reflected rays --/
theorem ray_reflection (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : 
  A = (2, 3) → 
  B = (1, 1) → 
  (∀ x y, l x y ↔ x + y + 1 = 0) →
  ∃ (incident reflected : ℝ → ℝ → Prop),
    (∀ x y, incident x y ↔ 9*x - 7*y + 3 = 0) ∧
    (∀ x y, reflected x y ↔ 7*x - 9*y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ray_reflection_l1077_107784


namespace NUMINAMATH_CALUDE_guanghua_community_households_l1077_107762

theorem guanghua_community_households (num_buildings : ℕ) (floors_per_building : ℕ) (households_per_floor : ℕ) 
  (h1 : num_buildings = 14)
  (h2 : floors_per_building = 7)
  (h3 : households_per_floor = 8) :
  num_buildings * floors_per_building * households_per_floor = 784 := by
  sorry

end NUMINAMATH_CALUDE_guanghua_community_households_l1077_107762


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1077_107742

-- System (1)
theorem system_one_solution (x y : ℚ) :
  (4 * x - 3 * y = 1 ∧ 3 * x - 2 * y = -1) ↔ (x = -5 ∧ y = 7) :=
sorry

-- System (2)
theorem system_two_solution (x y : ℚ) :
  ((y + 1) / 4 = (x + 2) / 3 ∧ 2 * x - 3 * y = 1) ↔ (x = -3 ∧ y = -7/3) :=
sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1077_107742


namespace NUMINAMATH_CALUDE_cubic_equation_root_l1077_107790

theorem cubic_equation_root (c d : ℚ) : 
  (3 + 2 * Real.sqrt 5)^3 + c * (3 + 2 * Real.sqrt 5)^2 + d * (3 + 2 * Real.sqrt 5) + 45 = 0 →
  c = -10 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l1077_107790


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1077_107797

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  ArithmeticSequence a →
  (a 1 + a 4 + a 7 = 39) →
  (a 2 + a 5 + a 8 = 33) →
  (a 3 + a 6 + a 9 = 27) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1077_107797


namespace NUMINAMATH_CALUDE_total_seashells_l1077_107771

def sam_seashells : ℕ := 18
def mary_seashells : ℕ := 47
def john_seashells : ℕ := 32
def emily_seashells : ℕ := 26

theorem total_seashells : 
  sam_seashells + mary_seashells + john_seashells + emily_seashells = 123 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l1077_107771


namespace NUMINAMATH_CALUDE_semicircle_area_theorem_l1077_107759

theorem semicircle_area_theorem (x y z : ℝ) : 
  x^2 + y^2 = z^2 →
  (1/8) * π * x^2 = 50 * π →
  (1/8) * π * y^2 = 288 * π →
  (1/8) * π * z^2 = 338 * π :=
sorry

end NUMINAMATH_CALUDE_semicircle_area_theorem_l1077_107759


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1077_107709

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + 2*k*y + 4*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  x^2 * z / y^3 = 125 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1077_107709


namespace NUMINAMATH_CALUDE_frustum_smaller_base_area_l1077_107752

/-- A frustum with given properties -/
structure Frustum where
  r : ℝ  -- radius of the smaller base
  h : ℝ  -- slant height
  S : ℝ  -- lateral area

/-- The theorem stating the properties of the frustum and its smaller base area -/
theorem frustum_smaller_base_area (f : Frustum) 
  (h1 : f.h = 3)
  (h2 : f.S = 84 * Real.pi)
  (h3 : 2 * Real.pi * (3 * f.r) = 3 * (2 * Real.pi * f.r)) :
  f.r^2 * Real.pi = 49 * Real.pi := by
  sorry

#check frustum_smaller_base_area

end NUMINAMATH_CALUDE_frustum_smaller_base_area_l1077_107752


namespace NUMINAMATH_CALUDE_parallelogram_area_theorem_l1077_107785

/-- Represents the area of a parallelogram with a square removed -/
def parallelogram_area_with_square_removed (base : ℝ) (height : ℝ) (square_side : ℝ) : ℝ :=
  base * height - square_side * square_side

/-- Theorem stating that a parallelogram with base 20 and height 4, 
    after removing a 2x2 square, has an area of 76 square feet -/
theorem parallelogram_area_theorem :
  parallelogram_area_with_square_removed 20 4 2 = 76 := by
  sorry

#eval parallelogram_area_with_square_removed 20 4 2

end NUMINAMATH_CALUDE_parallelogram_area_theorem_l1077_107785


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1077_107702

theorem quadratic_equation_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (∀ x : ℝ, (x - 1) * (x + 5) = 3 * x + 1 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1077_107702


namespace NUMINAMATH_CALUDE_square_sum_equality_l1077_107739

theorem square_sum_equality : 107 * 107 + 93 * 93 = 20098 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l1077_107739


namespace NUMINAMATH_CALUDE_complement_of_range_l1077_107712

def f (x : ℝ) : ℝ := x^2 - 2*x - 3

def domain : Set ℝ := Set.univ

def range : Set ℝ := {y | ∃ x, f x = y}

theorem complement_of_range :
  (domain \ range) = {x | x < -4} :=
sorry

end NUMINAMATH_CALUDE_complement_of_range_l1077_107712


namespace NUMINAMATH_CALUDE_string_folding_theorem_l1077_107786

/-- The number of layers after folding a string n times -/
def layers (n : ℕ) : ℕ := 2^n

/-- The number of longer strings after folding and cutting -/
def longer_strings (total_layers : ℕ) : ℕ := total_layers - 1

/-- The number of shorter strings after folding and cutting -/
def shorter_strings (total_layers : ℕ) (num_cuts : ℕ) : ℕ :=
  (num_cuts - 2) * total_layers + 2

theorem string_folding_theorem (num_folds num_cuts : ℕ) 
  (h1 : num_folds = 10) (h2 : num_cuts = 10) :
  longer_strings (layers num_folds) = 1023 ∧
  shorter_strings (layers num_folds) num_cuts = 8194 := by
  sorry

#eval longer_strings (layers 10)
#eval shorter_strings (layers 10) 10

end NUMINAMATH_CALUDE_string_folding_theorem_l1077_107786


namespace NUMINAMATH_CALUDE_lions_after_one_year_l1077_107734

/-- Calculates the number of lions after a given number of months -/
def lions_after_months (initial_population : ℕ) (birth_rate : ℕ) (death_rate : ℕ) (months : ℕ) : ℕ :=
  initial_population + birth_rate * months - death_rate * months

/-- Theorem stating that given the initial conditions, there will be 148 lions after 12 months -/
theorem lions_after_one_year :
  lions_after_months 100 5 1 12 = 148 := by
  sorry

end NUMINAMATH_CALUDE_lions_after_one_year_l1077_107734


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l1077_107793

def U : Set Int := {x | -3 < x ∧ x < 3}
def A : Set Int := {1, 2}
def B : Set Int := {-2, -1, 2}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l1077_107793


namespace NUMINAMATH_CALUDE_line_intercept_sum_l1077_107757

/-- A line passing through (5, 3) with slope 3 has x-intercept + y-intercept = -8 -/
theorem line_intercept_sum : ∀ (f : ℝ → ℝ),
  (f 5 = 3) →                        -- The line passes through (5, 3)
  (∀ x y, f y - f x = 3 * (y - x)) → -- The slope is 3
  (∃ a, f a = 0) →                   -- x-intercept exists
  (∃ b, f 0 = b) →                   -- y-intercept exists
  (∃ a b, f a = 0 ∧ f 0 = b ∧ a + b = -8) :=
by sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l1077_107757


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1077_107716

/-- Given a circle with two inscribed squares:
    - The first square is inscribed in the circle
    - The second square is inscribed in the segment of the circle cut off by one side of the first square
    This theorem states that the ratio of the side lengths of these squares is 5:1 -/
theorem inscribed_squares_ratio (r : ℝ) (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (2 * a) ^ 2 + (2 * a) ^ 2 = (2 * r) ^ 2 →  -- First square inscribed in circle
  (a + 2 * b) ^ 2 + b ^ 2 = r ^ 2 →          -- Second square inscribed in segment
  a / b = 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1077_107716


namespace NUMINAMATH_CALUDE_bulls_win_in_seven_l1077_107783

/-- The probability of the Knicks winning a single game -/
def p_knicks_win : ℚ := 3/4

/-- The probability of the Bulls winning a single game -/
def p_bulls_win : ℚ := 1 - p_knicks_win

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The total number of games played when the series goes to 7 games -/
def total_games : ℕ := 7

/-- The number of ways to choose 3 games out of 6 -/
def ways_to_choose_3_of_6 : ℕ := 20

theorem bulls_win_in_seven (
  p_knicks_win : ℚ) 
  (p_bulls_win : ℚ) 
  (games_to_win : ℕ) 
  (total_games : ℕ) 
  (ways_to_choose_3_of_6 : ℕ) :
  p_knicks_win = 3/4 →
  p_bulls_win = 1 - p_knicks_win →
  games_to_win = 4 →
  total_games = 7 →
  ways_to_choose_3_of_6 = 20 →
  (ways_to_choose_3_of_6 : ℚ) * p_bulls_win^3 * p_knicks_win^3 * p_bulls_win = 540/16384 :=
by sorry

end NUMINAMATH_CALUDE_bulls_win_in_seven_l1077_107783


namespace NUMINAMATH_CALUDE_smallest_square_area_l1077_107741

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The smallest square that can contain two rectangles without overlap -/
def smallest_containing_square (r1 r2 : Rectangle) : ℕ :=
  (min r1.width r1.height + min r2.width r2.height) ^ 2

/-- Theorem stating the smallest possible area of the square -/
theorem smallest_square_area (r1 r2 : Rectangle) 
  (h1 : r1 = ⟨2, 3⟩) 
  (h2 : r2 = ⟨3, 4⟩) : 
  smallest_containing_square r1 r2 = 25 := by
  sorry

#eval smallest_containing_square ⟨2, 3⟩ ⟨3, 4⟩

end NUMINAMATH_CALUDE_smallest_square_area_l1077_107741


namespace NUMINAMATH_CALUDE_triangle_side_relationship_l1077_107778

/-- Given a triangle with perimeter 12 and one side 5, prove the relationship between the other two sides -/
theorem triangle_side_relationship (x y : ℝ) : 
  (0 < x ∧ x < 6) → 
  (0 < y ∧ y < 6) → 
  (5 + x + y = 12) → 
  y = 7 - x :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_relationship_l1077_107778


namespace NUMINAMATH_CALUDE_fixed_point_on_all_parabolas_l1077_107777

/-- The parabola family defined by a real parameter t -/
def parabola (t : ℝ) (x : ℝ) : ℝ := 4 * x^2 + 2 * t * x - 3 * t

/-- The fixed point through which all parabolas pass -/
def fixed_point : ℝ × ℝ := (3, 36)

/-- Theorem stating that the fixed point lies on all parabolas in the family -/
theorem fixed_point_on_all_parabolas :
  ∀ t : ℝ, parabola t (fixed_point.1) = fixed_point.2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_all_parabolas_l1077_107777


namespace NUMINAMATH_CALUDE_man_speed_against_current_l1077_107769

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific conditions,
    the man's speed against the current is 18 kmph. -/
theorem man_speed_against_current :
  speed_against_current 20 1 = 18 := by
  sorry

#eval speed_against_current 20 1

end NUMINAMATH_CALUDE_man_speed_against_current_l1077_107769


namespace NUMINAMATH_CALUDE_train_passing_time_l1077_107796

/-- The time it takes for two trains to pass each other -/
theorem train_passing_time (v1 l1 v2 l2 : ℝ) : 
  v1 > 0 → l1 > 0 → v2 > 0 → l2 > 0 →
  (l1 / v1 = 5) →
  (v1 = 2 * v2) →
  (l1 = 3 * l2) →
  (l1 + l2) / (v1 + v2) = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1077_107796


namespace NUMINAMATH_CALUDE_power_product_square_l1077_107753

theorem power_product_square (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_square_l1077_107753


namespace NUMINAMATH_CALUDE_xoxoxox_probability_l1077_107743

/-- The probability of arranging 4 X tiles and 3 O tiles in the specific order XOXOXOX -/
theorem xoxoxox_probability (n : ℕ) (x o : ℕ) (h1 : n = 7) (h2 : x = 4) (h3 : o = 3) :
  (1 : ℚ) / (n.choose x) = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_xoxoxox_probability_l1077_107743


namespace NUMINAMATH_CALUDE_subtraction_result_l1077_107770

theorem subtraction_result : 
  let total : ℚ := 8000
  let fraction1 : ℚ := 1 / 10
  let fraction2 : ℚ := 1 / 20 * (1 / 100)
  (total * fraction1) - (total * fraction2) = 796 :=
by sorry

end NUMINAMATH_CALUDE_subtraction_result_l1077_107770


namespace NUMINAMATH_CALUDE_basketball_weight_proof_l1077_107789

/-- The weight of a skateboard in pounds -/
def skateboard_weight : ℝ := 32

/-- The number of skateboards that balance with the basketballs -/
def num_skateboards : ℕ := 4

/-- The number of basketballs that balance with the skateboards -/
def num_basketballs : ℕ := 8

/-- The weight of a single basketball in pounds -/
def basketball_weight : ℝ := 16

theorem basketball_weight_proof :
  num_basketballs * basketball_weight = num_skateboards * skateboard_weight :=
by sorry

end NUMINAMATH_CALUDE_basketball_weight_proof_l1077_107789


namespace NUMINAMATH_CALUDE_inequality_proof_l1077_107723

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / a + 1 / b + 1 / c = 1) :
  Real.sqrt (a * b + c) + Real.sqrt (b * c + a) + Real.sqrt (c * a + b) ≥
  Real.sqrt (a * b * c) + Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1077_107723


namespace NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l1077_107766

/-- The average speed of a car traveling different distances in two hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : 
  d1 ≥ 0 → d2 ≥ 0 → (d1 + d2) / 2 = (d1 / 1 + d2 / 1) / 2 := by
  sorry

/-- The average speed of a car traveling 10 km in the first hour and 60 km in the second hour is 35 km/h -/
theorem car_average_speed : 
  let d1 : ℝ := 10  -- Distance traveled in the first hour
  let d2 : ℝ := 60  -- Distance traveled in the second hour
  (d1 + d2) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l1077_107766


namespace NUMINAMATH_CALUDE_graveyard_skeletons_l1077_107719

/-- Represents the number of skeletons in the graveyard -/
def S : ℕ := sorry

/-- The number of bones in an adult woman's skeleton -/
def womanBones : ℕ := 20

/-- The number of bones in an adult man's skeleton -/
def manBones : ℕ := womanBones + 5

/-- The number of bones in a child's skeleton -/
def childBones : ℕ := womanBones / 2

/-- The total number of bones in the graveyard -/
def totalBones : ℕ := 375

theorem graveyard_skeletons :
  (S / 2 * womanBones + S / 4 * manBones + S / 4 * childBones = totalBones) →
  S = 20 := by sorry

end NUMINAMATH_CALUDE_graveyard_skeletons_l1077_107719


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1077_107727

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 12 → x ≥ 7 ∧ 7 < 3*7 - 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1077_107727


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1077_107708

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 + 2*a*x + a > 0) ↔ a > -1/3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1077_107708


namespace NUMINAMATH_CALUDE_factor_w4_minus_81_l1077_107718

theorem factor_w4_minus_81 (w : ℂ) : w^4 - 81 = (w-3)*(w+3)*(w-3*I)*(w+3*I) := by
  sorry

end NUMINAMATH_CALUDE_factor_w4_minus_81_l1077_107718
