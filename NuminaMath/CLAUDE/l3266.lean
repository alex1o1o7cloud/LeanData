import Mathlib

namespace NUMINAMATH_CALUDE_balloon_solution_l3266_326662

/-- The number of balloons Allan and Jake have in the park -/
def balloon_problem (allan_balloons jake_initial_balloons jake_bought_balloons : ℕ) : Prop :=
  allan_balloons - (jake_initial_balloons + jake_bought_balloons) = 1

/-- Theorem stating the solution to the balloon problem -/
theorem balloon_solution :
  balloon_problem 6 2 3 := by
  sorry

end NUMINAMATH_CALUDE_balloon_solution_l3266_326662


namespace NUMINAMATH_CALUDE_largest_angle_is_75_l3266_326664

-- Define the angles of the triangle
def triangle_angles (a b c : ℝ) : Prop :=
  -- The sum of all angles in a triangle is 180°
  a + b + c = 180 ∧
  -- Two angles sum to 7/6 of a right angle (90°)
  b + c = 7/6 * 90 ∧
  -- One angle is 10° more than twice the other
  c = 2 * b + 10 ∧
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c

-- Theorem statement
theorem largest_angle_is_75 (a b c : ℝ) :
  triangle_angles a b c → max a (max b c) = 75 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_is_75_l3266_326664


namespace NUMINAMATH_CALUDE_remainder_problem_l3266_326665

theorem remainder_problem : 123456789012 % 360 = 108 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3266_326665


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l3266_326692

def arithmetic_sequence_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_specific :
  arithmetic_sequence_sum (-41) 3 2 = -437 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l3266_326692


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l3266_326693

theorem existence_of_special_numbers :
  ∃ (S : Finset ℕ), Finset.card S = 100 ∧
  ∀ (a b c d e : ℕ), a ∈ S → b ∈ S → c ∈ S → d ∈ S → e ∈ S →
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → b ≠ c → b ≠ d → b ≠ e → c ≠ d → c ≠ e → d ≠ e →
  (a * b * c * d * e) % (a + b + c + d + e) = 0 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l3266_326693


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3266_326625

theorem least_addition_for_divisibility : ∃! x : ℕ, x < 37 ∧ (1052 + x) % 37 = 0 ∧ ∀ y : ℕ, y < x → (1052 + y) % 37 ≠ 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3266_326625


namespace NUMINAMATH_CALUDE_sin_240_degrees_l3266_326623

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l3266_326623


namespace NUMINAMATH_CALUDE_movie_choice_l3266_326632

-- Define the set of all movies
def Movies : Set Char := {'A', 'B', 'C', 'D', 'E'}

-- Define the acceptable movies for each person
def Zhao : Set Char := Movies \ {'B'}
def Zhang : Set Char := {'B', 'C', 'D', 'E'}
def Li : Set Char := Movies \ {'C'}
def Liu : Set Char := Movies \ {'E'}

-- Theorem statement
theorem movie_choice : Zhao ∩ Zhang ∩ Li ∩ Liu = {'D'} := by
  sorry

end NUMINAMATH_CALUDE_movie_choice_l3266_326632


namespace NUMINAMATH_CALUDE_slope_product_theorem_l3266_326637

theorem slope_product_theorem (m n : ℝ) : 
  m ≠ 0 → n ≠ 0 →  -- non-horizontal lines
  (∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) →  -- angle relationship
  m = 6 * n →  -- slope relationship
  m * n = 9 / 17 := by
sorry

end NUMINAMATH_CALUDE_slope_product_theorem_l3266_326637


namespace NUMINAMATH_CALUDE_central_angle_of_sector_l3266_326668

-- Define the sector
structure Sector where
  radius : ℝ
  area : ℝ

-- Define the theorem
theorem central_angle_of_sector (s : Sector) (h1 : s.radius = 2) (h2 : s.area = 8) :
  (2 * s.area) / (s.radius ^ 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_of_sector_l3266_326668


namespace NUMINAMATH_CALUDE_four_greater_than_sqrt_fourteen_l3266_326609

theorem four_greater_than_sqrt_fourteen : 4 > Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_four_greater_than_sqrt_fourteen_l3266_326609


namespace NUMINAMATH_CALUDE_a_eq_2_sufficient_not_necessary_for_abs_a_eq_2_l3266_326659

theorem a_eq_2_sufficient_not_necessary_for_abs_a_eq_2 :
  (∃ a : ℝ, a = 2 → |a| = 2) ∧ 
  (∃ a : ℝ, |a| = 2 ∧ a ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_2_sufficient_not_necessary_for_abs_a_eq_2_l3266_326659


namespace NUMINAMATH_CALUDE_range_of_m_l3266_326675

/-- The function f(x) = x² + mx - 1 --/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

/-- Theorem stating the range of m given the conditions --/
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc m (m + 1), f m x < 0) →
  m ∈ Set.Ioo (-Real.sqrt 2 / 2) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3266_326675


namespace NUMINAMATH_CALUDE_money_spending_l3266_326661

theorem money_spending (M : ℚ) : 
  (2 / 7 : ℚ) * M = 500 →
  M = 1750 := by
  sorry

end NUMINAMATH_CALUDE_money_spending_l3266_326661


namespace NUMINAMATH_CALUDE_sum_equality_implies_expression_value_l3266_326643

theorem sum_equality_implies_expression_value 
  (A B : ℝ) (h : A + B = 5) : B - 3 + A = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_equality_implies_expression_value_l3266_326643


namespace NUMINAMATH_CALUDE_girl_multiplication_problem_l3266_326684

theorem girl_multiplication_problem (incorrect_multiplier : ℕ) (difference : ℕ) (base_number : ℕ) :
  incorrect_multiplier = 34 →
  difference = 1242 →
  base_number = 138 →
  ∃ (correct_multiplier : ℕ), 
    base_number * correct_multiplier = base_number * incorrect_multiplier + difference ∧
    correct_multiplier = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_girl_multiplication_problem_l3266_326684


namespace NUMINAMATH_CALUDE_max_product_with_sum_and_diff_l3266_326615

/-- Given two real numbers with a difference of 4 and a sum of 35, 
    their product is maximized when the numbers are 19.5 and 15.5 -/
theorem max_product_with_sum_and_diff (x y : ℝ) : 
  x - y = 4 → x + y = 35 → x * y ≤ 19.5 * 15.5 :=
by sorry

end NUMINAMATH_CALUDE_max_product_with_sum_and_diff_l3266_326615


namespace NUMINAMATH_CALUDE_total_weight_is_63_l3266_326656

/-- The weight of beeswax used in each candle, in ounces -/
def beeswax_weight : ℕ := 8

/-- The weight of coconut oil used in each candle, in ounces -/
def coconut_oil_weight : ℕ := 1

/-- The number of candles Ethan makes -/
def num_candles : ℕ := 10 - 3

/-- The total weight of all candles made by Ethan, in ounces -/
def total_weight : ℕ := num_candles * (beeswax_weight + coconut_oil_weight)

/-- Theorem stating that the total weight of candles is 63 ounces -/
theorem total_weight_is_63 : total_weight = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_63_l3266_326656


namespace NUMINAMATH_CALUDE_simplify_fraction_l3266_326629

theorem simplify_fraction : 
  1 / (1 / ((1/2)^0) + 1 / ((1/2)^1) + 1 / ((1/2)^2) + 1 / ((1/2)^3) + 1 / ((1/2)^4)) = 1 / 31 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3266_326629


namespace NUMINAMATH_CALUDE_xyz_value_l3266_326600

theorem xyz_value (x y z : ℝ) 
  (eq1 : (x + y + z) * (x * y + x * z + y * z) = 35)
  (eq2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 23 / 3 := by sorry

end NUMINAMATH_CALUDE_xyz_value_l3266_326600


namespace NUMINAMATH_CALUDE_chloe_apples_l3266_326641

theorem chloe_apples (chloe_apples dylan_apples : ℕ) : 
  chloe_apples = dylan_apples + 8 →
  dylan_apples = chloe_apples / 3 →
  chloe_apples = 12 := by
sorry

end NUMINAMATH_CALUDE_chloe_apples_l3266_326641


namespace NUMINAMATH_CALUDE_widest_strip_width_l3266_326695

theorem widest_strip_width (w1 w2 w3 : ℕ) (hw1 : w1 = 45) (hw2 : w2 = 60) (hw3 : w3 = 70) :
  Nat.gcd w1 (Nat.gcd w2 w3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_widest_strip_width_l3266_326695


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3266_326649

-- Define the quadratic inequality
def quadratic_inequality (a x : ℝ) : Prop := a * x^2 - 2 * x + a ≤ 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | quadratic_inequality a x}

theorem quadratic_inequality_solution :
  (∃! x, x ∈ solution_set 1) ∧
  (0 ∈ solution_set a ∧ -1 ∉ solution_set a → a ∈ Set.Ioc (-1) 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3266_326649


namespace NUMINAMATH_CALUDE_absolute_difference_always_less_than_one_l3266_326676

theorem absolute_difference_always_less_than_one :
  ∀ (m : ℝ), ∀ (x : ℝ), |x - m| < 1 :=
by sorry

end NUMINAMATH_CALUDE_absolute_difference_always_less_than_one_l3266_326676


namespace NUMINAMATH_CALUDE_least_addend_proof_l3266_326689

/-- The least non-negative integer that, when added to 11002, results in a number divisible by 11 -/
def least_addend : ℕ := 9

/-- The original number we start with -/
def original_number : ℕ := 11002

theorem least_addend_proof :
  (∀ k : ℕ, k < least_addend → ¬((original_number + k) % 11 = 0)) ∧
  ((original_number + least_addend) % 11 = 0) :=
sorry

end NUMINAMATH_CALUDE_least_addend_proof_l3266_326689


namespace NUMINAMATH_CALUDE_dvd_pack_cost_l3266_326691

theorem dvd_pack_cost (total_cost : ℝ) (num_packs : ℕ) (h1 : total_cost = 120) (h2 : num_packs = 6) :
  total_cost / num_packs = 20 := by
sorry

end NUMINAMATH_CALUDE_dvd_pack_cost_l3266_326691


namespace NUMINAMATH_CALUDE_intersection_A_B_l3266_326653

def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {-2, 0, 2}

theorem intersection_A_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3266_326653


namespace NUMINAMATH_CALUDE_pet_groomer_problem_l3266_326614

theorem pet_groomer_problem (total_animals : ℕ) (cats : ℕ) (selected : ℕ) (prob : ℚ) :
  total_animals = 7 →
  cats = 2 →
  selected = 4 →
  prob = 2/7 →
  (Nat.choose cats cats * Nat.choose (total_animals - cats) (selected - cats)) / Nat.choose total_animals selected = prob →
  total_animals - cats = 5 := by
sorry

end NUMINAMATH_CALUDE_pet_groomer_problem_l3266_326614


namespace NUMINAMATH_CALUDE_rectangular_field_width_l3266_326639

theorem rectangular_field_width (length width : ℝ) : 
  length = 24 ∧ length = 2 * width - 3 → width = 13.5 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l3266_326639


namespace NUMINAMATH_CALUDE_quarterly_charge_is_80_l3266_326667

/-- The Kwik-e-Tax Center pricing structure and sales data -/
structure TaxCenter where
  federal_charge : ℕ
  state_charge : ℕ
  federal_sold : ℕ
  state_sold : ℕ
  quarterly_sold : ℕ
  total_revenue : ℕ

/-- The charge for quarterly business taxes -/
def quarterly_charge (tc : TaxCenter) : ℕ :=
  (tc.total_revenue - (tc.federal_charge * tc.federal_sold + tc.state_charge * tc.state_sold)) / tc.quarterly_sold

/-- Theorem stating the charge for quarterly business taxes is $80 -/
theorem quarterly_charge_is_80 (tc : TaxCenter) 
  (h1 : tc.federal_charge = 50)
  (h2 : tc.state_charge = 30)
  (h3 : tc.federal_sold = 60)
  (h4 : tc.state_sold = 20)
  (h5 : tc.quarterly_sold = 10)
  (h6 : tc.total_revenue = 4400) :
  quarterly_charge tc = 80 := by
  sorry

#eval quarterly_charge { federal_charge := 50, state_charge := 30, federal_sold := 60, state_sold := 20, quarterly_sold := 10, total_revenue := 4400 }

end NUMINAMATH_CALUDE_quarterly_charge_is_80_l3266_326667


namespace NUMINAMATH_CALUDE_negation_of_existential_real_exp_l3266_326635

theorem negation_of_existential_real_exp (p : Prop) : 
  (p ↔ ∃ x : ℝ, Real.exp x < 0) → 
  (¬p ↔ ∀ x : ℝ, Real.exp x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_real_exp_l3266_326635


namespace NUMINAMATH_CALUDE_six_couples_handshakes_l3266_326642

/-- Represents the number of handshakes in a gathering of couples -/
def handshakes_in_gathering (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 6 couples, where each person shakes hands with
    everyone except their spouse and one other person, there are 54 handshakes -/
theorem six_couples_handshakes :
  handshakes_in_gathering 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_six_couples_handshakes_l3266_326642


namespace NUMINAMATH_CALUDE_fuel_cost_calculation_l3266_326680

/-- Calculates the new fuel cost after a price increase and capacity increase -/
def new_fuel_cost (original_cost : ℚ) (price_increase_percent : ℚ) (capacity_multiplier : ℚ) : ℚ :=
  original_cost * (1 + price_increase_percent / 100) * capacity_multiplier

/-- Proves that the new fuel cost is $480 given the specified conditions -/
theorem fuel_cost_calculation :
  new_fuel_cost 200 20 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_calculation_l3266_326680


namespace NUMINAMATH_CALUDE_binomial_sum_theorem_l3266_326699

theorem binomial_sum_theorem :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (5*x - 4)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 25 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_theorem_l3266_326699


namespace NUMINAMATH_CALUDE_water_added_to_tank_l3266_326624

/-- The amount of water added to a tank -/
def water_added (capacity : ℚ) (initial_fraction : ℚ) (final_fraction : ℚ) : ℚ :=
  capacity * (final_fraction - initial_fraction)

/-- Theorem: The amount of water added to a 40-gallon tank, 
    initially 3/4 full and ending up 7/8 full, is 5 gallons -/
theorem water_added_to_tank : 
  water_added 40 (3/4) (7/8) = 5 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_tank_l3266_326624


namespace NUMINAMATH_CALUDE_hannahs_running_distance_l3266_326633

/-- Hannah's running distances problem -/
theorem hannahs_running_distance :
  -- Define the distances
  let monday_distance : ℕ := 9000
  let friday_distance : ℕ := 2095
  let additional_distance : ℕ := 2089

  -- Define the relation between distances
  ∀ wednesday_distance : ℕ,
    monday_distance = wednesday_distance + friday_distance + additional_distance →
    wednesday_distance = 4816 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_running_distance_l3266_326633


namespace NUMINAMATH_CALUDE_circle_intersection_equality_l3266_326603

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary relations and functions
variable (on_circle : Point → Circle → Prop)
variable (center : Circle → Point)
variable (intersect : Circle → Circle → Point × Point)
variable (line_intersect : Point → Point → Circle → Point)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem circle_intersection_equality 
  (circle1 circle2 : Circle) 
  (O P Q C A B : Point) :
  on_circle O circle1 ∧ 
  center circle2 = O ∧
  intersect circle1 circle2 = (P, Q) ∧
  on_circle C circle1 ∧
  line_intersect C P circle2 = A ∧
  line_intersect C Q circle2 = B →
  distance A B = distance P Q :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_equality_l3266_326603


namespace NUMINAMATH_CALUDE_sqrt_seven_squared_minus_four_l3266_326638

theorem sqrt_seven_squared_minus_four : (Real.sqrt 7 + 2) * (Real.sqrt 7 - 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_squared_minus_four_l3266_326638


namespace NUMINAMATH_CALUDE_task_completion_time_l3266_326611

/-- Given two workers can complete a task in 35 days, and one worker can complete it in 60 days,
    prove that the other worker can complete the task in 84 days. -/
theorem task_completion_time (total_time : ℝ) (worker1_time : ℝ) (worker2_time : ℝ) : 
  (1 / total_time = 1 / worker1_time + 1 / worker2_time) →
  (total_time = 35) →
  (worker1_time = 60) →
  (worker2_time = 84) := by
sorry

end NUMINAMATH_CALUDE_task_completion_time_l3266_326611


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3266_326620

def A : Set ℤ := {1, 3}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3266_326620


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3266_326655

/-- Given complex numbers z₁, z₂, z₃ satisfying (z₃ - z₁) / (z₂ - z₁) = ai 
    where a ∈ ℝ and a ≠ 0, the angle between vectors ⃗Z₁Z₂ and ⃗Z₁Z₃ is π/2. -/
theorem angle_between_vectors (z₁ z₂ z₃ : ℂ) (a : ℝ) 
    (h : (z₃ - z₁) / (z₂ - z₁) = Complex.I * a) 
    (ha : a ≠ 0) : 
  Complex.arg ((z₃ - z₁) / (z₂ - z₁)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l3266_326655


namespace NUMINAMATH_CALUDE_max_kids_on_bus_l3266_326674

/-- Represents the school bus configuration -/
structure SchoolBus where
  lowerDeckRows : Nat
  upperDeckRows : Nat
  lowerDeckCapacity : Nat
  upperDeckCapacity : Nat
  staffMembers : Nat
  reservedSeats : Nat

/-- Calculates the maximum number of kids that can ride the school bus -/
def maxKids (bus : SchoolBus) : Nat :=
  (bus.lowerDeckRows * bus.lowerDeckCapacity + bus.upperDeckRows * bus.upperDeckCapacity)
  - bus.staffMembers - bus.reservedSeats

/-- The theorem stating the maximum number of kids that can ride the school bus -/
theorem max_kids_on_bus :
  let bus : SchoolBus := {
    lowerDeckRows := 15,
    upperDeckRows := 10,
    lowerDeckCapacity := 5,
    upperDeckCapacity := 3,
    staffMembers := 4,
    reservedSeats := 10
  }
  maxKids bus = 91 := by
  sorry

#eval maxKids {
  lowerDeckRows := 15,
  upperDeckRows := 10,
  lowerDeckCapacity := 5,
  upperDeckCapacity := 3,
  staffMembers := 4,
  reservedSeats := 10
}

end NUMINAMATH_CALUDE_max_kids_on_bus_l3266_326674


namespace NUMINAMATH_CALUDE_weighted_sum_square_inequality_l3266_326650

theorem weighted_sum_square_inequality (x y a b : ℝ) 
  (h1 : a + b = 1) (h2 : a ≥ 0) (h3 : b ≥ 0) : 
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 := by
  sorry

end NUMINAMATH_CALUDE_weighted_sum_square_inequality_l3266_326650


namespace NUMINAMATH_CALUDE_mila_visible_area_l3266_326622

/-- The area visible to Mila as she walks around a square -/
theorem mila_visible_area (side_length : ℝ) (visibility_radius : ℝ) : 
  side_length = 4 →
  visibility_radius = 1 →
  (side_length - 2 * visibility_radius)^2 + 
  4 * side_length * visibility_radius + 
  π * visibility_radius^2 = 28 + π := by
  sorry

end NUMINAMATH_CALUDE_mila_visible_area_l3266_326622


namespace NUMINAMATH_CALUDE_jose_investment_is_45000_l3266_326648

/-- Represents the investment and profit scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_investment : ℕ
  tom_months : ℕ
  jose_months : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given conditions --/
def calculate_jose_investment (s : ShopInvestment) : ℕ :=
  let tom_time_investment := s.tom_investment * s.tom_months
  let tom_profit := s.total_profit - s.jose_profit
  (tom_time_investment * s.jose_profit) / (tom_profit * s.jose_months)

/-- Theorem stating that Jose's investment is 45000 given the problem conditions --/
theorem jose_investment_is_45000 :
  let s : ShopInvestment := {
    tom_investment := 30000,
    tom_months := 12,
    jose_months := 10,
    total_profit := 63000,
    jose_profit := 35000
  }
  calculate_jose_investment s = 45000 := by sorry


end NUMINAMATH_CALUDE_jose_investment_is_45000_l3266_326648


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l3266_326679

theorem slope_angle_of_line (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ θ : ℝ, θ ∈ Set.Icc 0 π ∧ θ = π - Real.arctan (a / b) := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l3266_326679


namespace NUMINAMATH_CALUDE_arrangements_count_is_24_l3266_326690

/-- The number of ways to arrange 5 people in a line, where two specific people
    must stand next to each other but not at the ends. -/
def arrangements_count : ℕ :=
  /- Number of ways to arrange A and B together -/
  (2 * 1) *
  /- Number of positions for A and B together (excluding ends) -/
  3 *
  /- Number of ways to arrange the other 3 people -/
  (3 * 2 * 1)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count_is_24 : arrangements_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_is_24_l3266_326690


namespace NUMINAMATH_CALUDE_total_eggs_per_week_l3266_326640

/-- Represents the three chicken breeds -/
inductive Breed
  | BCM  -- Black Copper Marans
  | RIR  -- Rhode Island Reds
  | LH   -- Leghorns

/-- Calculates the number of chickens for a given breed -/
def chickenCount (b : Breed) : Nat :=
  match b with
  | Breed.BCM => 125
  | Breed.RIR => 200
  | Breed.LH  => 175

/-- Calculates the number of hens for a given breed -/
def henCount (b : Breed) : Nat :=
  match b with
  | Breed.BCM => 81
  | Breed.RIR => 110
  | Breed.LH  => 105

/-- Represents the egg-laying rates for each breed -/
def eggRates (b : Breed) : List Nat :=
  match b with
  | Breed.BCM => [3, 4, 5]
  | Breed.RIR => [5, 6, 7]
  | Breed.LH  => [6, 7, 8]

/-- Represents the distribution of hens for each egg-laying rate -/
def henDistribution (b : Breed) : List Nat :=
  match b with
  | Breed.BCM => [32, 24, 25]
  | Breed.RIR => [22, 55, 33]
  | Breed.LH  => [26, 47, 32]

/-- Calculates the total eggs produced by a breed per week -/
def eggsByBreed (b : Breed) : Nat :=
  List.sum (List.zipWith (· * ·) (eggRates b) (henDistribution b))

/-- The main theorem: total eggs produced by all hens per week is 1729 -/
theorem total_eggs_per_week :
  (eggsByBreed Breed.BCM) + (eggsByBreed Breed.RIR) + (eggsByBreed Breed.LH) = 1729 := by
  sorry

#eval (eggsByBreed Breed.BCM) + (eggsByBreed Breed.RIR) + (eggsByBreed Breed.LH)

end NUMINAMATH_CALUDE_total_eggs_per_week_l3266_326640


namespace NUMINAMATH_CALUDE_pencil_packing_problem_l3266_326682

theorem pencil_packing_problem :
  ∃ (a k m : ℤ),
    200 ≤ a ∧ a ≤ 300 ∧
    a % 10 = 7 ∧
    a % 12 = 9 ∧
    a = 60 * m + 57 ∧
    (m = 3 ∨ m = 4) :=
by sorry

end NUMINAMATH_CALUDE_pencil_packing_problem_l3266_326682


namespace NUMINAMATH_CALUDE_smallest_valid_n_l3266_326658

def is_valid_sequence (n : ℕ) (xs : List ℕ) : Prop :=
  xs.length = n ∧
  (∀ x ∈ xs, 1 ≤ x ∧ x ≤ n) ∧
  xs.sum = n * (n + 1) / 2 ∧
  xs.prod = Nat.factorial n ∧
  xs.toFinset ≠ Finset.range n

theorem smallest_valid_n : 
  (∀ m < 9, ¬ ∃ xs : List ℕ, is_valid_sequence m xs) ∧
  (∃ xs : List ℕ, is_valid_sequence 9 xs) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l3266_326658


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l3266_326683

/-- Calculates the total wet surface area of a rectangular cistern. -/
def totalWetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * length * depth + 2 * width * depth

/-- Theorem: The total wet surface area of a cistern with given dimensions is 62 square meters. -/
theorem cistern_wet_surface_area :
  totalWetSurfaceArea 8 4 1.25 = 62 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l3266_326683


namespace NUMINAMATH_CALUDE_exists_city_that_reaches_all_l3266_326657

-- Define the type for cities
variable {City : Type}

-- Define the "can reach" relation
variable (canReach : City → City → Prop)

-- Define the properties of the "can reach" relation
variable (h_reflexive : ∀ x : City, canReach x x)
variable (h_transitive : ∀ x y z : City, canReach x y → canReach y z → canReach x z)

-- Define the condition that for any two cities, there's a city that can reach both
variable (h_common_reachable : ∀ x y : City, ∃ z : City, canReach z x ∧ canReach z y)

-- State the theorem
theorem exists_city_that_reaches_all [Finite City] :
  ∃ c : City, ∀ x : City, canReach c x :=
sorry

end NUMINAMATH_CALUDE_exists_city_that_reaches_all_l3266_326657


namespace NUMINAMATH_CALUDE_stuffed_animal_sales_difference_stuffed_animal_sales_difference_proof_l3266_326601

theorem stuffed_animal_sales_difference : ℕ → ℕ → ℕ → Prop :=
  fun thor jake quincy =>
    (jake = thor + 10) →
    (quincy = thor * 10) →
    (quincy = 200) →
    (quincy - jake = 170)

-- The proof would go here, but we're skipping it as requested
theorem stuffed_animal_sales_difference_proof :
  ∃ (thor jake quincy : ℕ), stuffed_animal_sales_difference thor jake quincy :=
sorry

end NUMINAMATH_CALUDE_stuffed_animal_sales_difference_stuffed_animal_sales_difference_proof_l3266_326601


namespace NUMINAMATH_CALUDE_next_perfect_square_l3266_326613

theorem next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m^2) ∧ 
  ∀ y : ℕ, y > x → (∃ l : ℕ, y = l^2) → y ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_next_perfect_square_l3266_326613


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3266_326646

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3266_326646


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3266_326628

/-- The minimum value of (a+1)^2 + b^2 for a point (a, b) on the line y = √3x - √3 is 3 -/
theorem min_distance_to_line : 
  ∀ a b : ℝ, 
  b = Real.sqrt 3 * a - Real.sqrt 3 → 
  (∀ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 → (a + 1)^2 + b^2 ≤ (x + 1)^2 + y^2) → 
  (a + 1)^2 + b^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3266_326628


namespace NUMINAMATH_CALUDE_cubic_sum_equality_l3266_326687

theorem cubic_sum_equality (x y z : ℝ) (h1 : x = y + z) (h2 : x = 2) :
  x^3 + 2*y^3 + 2*z^3 + 6*x*y*z = 24 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_equality_l3266_326687


namespace NUMINAMATH_CALUDE_distinct_collections_eq_110_l3266_326660

def vowels : ℕ := 5
def consonants : ℕ := 4
def indistinguishable_consonants : ℕ := 2
def vowels_to_select : ℕ := 3
def consonants_to_select : ℕ := 4

def distinct_collections : ℕ :=
  (Nat.choose vowels vowels_to_select) *
  (Nat.choose consonants consonants_to_select +
   Nat.choose consonants (consonants_to_select - 1) +
   Nat.choose consonants (consonants_to_select - 2))

theorem distinct_collections_eq_110 :
  distinct_collections = 110 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_eq_110_l3266_326660


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3266_326652

theorem quadratic_coefficient (b : ℝ) (n : ℝ) : 
  b < 0 ∧ 
  (∀ x, x^2 + b*x + 1/5 = (x+n)^2 + 1/20) →
  b = -Real.sqrt (3/5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3266_326652


namespace NUMINAMATH_CALUDE_ap_triangle_centroid_incenter_parallel_l3266_326621

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

end NUMINAMATH_CALUDE_ap_triangle_centroid_incenter_parallel_l3266_326621


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3266_326696

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3266_326696


namespace NUMINAMATH_CALUDE_geometric_sum_2_power_63_l3266_326698

theorem geometric_sum_2_power_63 : 
  (Finset.range 64).sum (fun i => 2^i) = 2^64 - 1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_2_power_63_l3266_326698


namespace NUMINAMATH_CALUDE_quadruplet_babies_l3266_326602

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1500)
  (h_triplets : ∃ (b c : ℕ), b = 3 * c)
  (h_twins : ∃ (a b : ℕ), a = 5 * b)
  (h_sum : ∃ (a b c : ℕ), 2 * a + 3 * b + 4 * c = total_babies) :
  ∃ (c : ℕ), 4 * c = 136 ∧ c * 4 ≤ total_babies := by
sorry

#eval 136

end NUMINAMATH_CALUDE_quadruplet_babies_l3266_326602


namespace NUMINAMATH_CALUDE_total_cost_is_53_l3266_326644

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def sandwich_quantity : ℕ := 7
def soda_quantity : ℕ := 10
def discount_threshold : ℕ := 15
def discount_amount : ℕ := 5

def total_items : ℕ := sandwich_quantity + soda_quantity

def total_cost : ℕ :=
  sandwich_cost * sandwich_quantity + soda_cost * soda_quantity - 
  if total_items > discount_threshold then discount_amount else 0

theorem total_cost_is_53 : total_cost = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_53_l3266_326644


namespace NUMINAMATH_CALUDE_musical_group_seats_l3266_326610

/-- Represents the number of seats needed for a musical group --/
def total_seats (F T Tr D C H S P V G : ℕ) : ℕ :=
  F + T + Tr + D + C + H + S + P + V + G

/-- Theorem stating the total number of seats needed for the musical group --/
theorem musical_group_seats :
  ∀ (F T Tr D C H S P V G : ℕ),
    F = 5 →
    T = 3 * F →
    Tr = T - 8 →
    D = Tr + 11 →
    C = 2 * F →
    H = Tr + 3 →
    S = (T + Tr) / 2 →
    P = D + 2 →
    V = H - C →
    G = 3 * F →
    total_seats F T Tr D C H S P V G = 111 :=
by
  sorry

end NUMINAMATH_CALUDE_musical_group_seats_l3266_326610


namespace NUMINAMATH_CALUDE_count_squares_below_line_l3266_326673

/-- The number of 1x1 squares in the first quadrant lying entirely below the line 6x + 216y = 1296 -/
def squaresBelowLine : ℕ :=
  -- Definition goes here
  sorry

/-- The equation of the line -/
def lineEquation (x y : ℝ) : Prop :=
  6 * x + 216 * y = 1296

theorem count_squares_below_line :
  squaresBelowLine = 537 := by
  sorry

end NUMINAMATH_CALUDE_count_squares_below_line_l3266_326673


namespace NUMINAMATH_CALUDE_pencil_sale_ratio_l3266_326645

theorem pencil_sale_ratio :
  ∀ (C S : ℚ),
  C > 0 → S > 0 →
  80 * C = 80 * S + 30 * S →
  (80 * C) / (80 * S) = 11 / 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_sale_ratio_l3266_326645


namespace NUMINAMATH_CALUDE_costume_cost_theorem_l3266_326616

/-- Calculates the total cost of materials for a costume --/
def costume_cost (skirt_length : ℝ) (skirt_width : ℝ) (num_skirts : ℕ) 
                 (skirt_cost_per_sqft : ℝ) (bodice_shirt_area : ℝ) 
                 (bodice_sleeve_area : ℝ) (bodice_cost_per_sqft : ℝ)
                 (bonnet_length : ℝ) (bonnet_width : ℝ) (bonnet_cost_per_sqft : ℝ)
                 (shoe_cover_length : ℝ) (shoe_cover_width : ℝ) 
                 (num_shoe_covers : ℕ) (shoe_cover_cost_per_sqft : ℝ) : ℝ :=
  let skirt_total_area := skirt_length * skirt_width * num_skirts
  let skirt_cost := skirt_total_area * skirt_cost_per_sqft
  let bodice_total_area := bodice_shirt_area + 2 * bodice_sleeve_area
  let bodice_cost := bodice_total_area * bodice_cost_per_sqft
  let bonnet_area := bonnet_length * bonnet_width
  let bonnet_cost := bonnet_area * bonnet_cost_per_sqft
  let shoe_cover_total_area := shoe_cover_length * shoe_cover_width * num_shoe_covers
  let shoe_cover_cost := shoe_cover_total_area * shoe_cover_cost_per_sqft
  skirt_cost + bodice_cost + bonnet_cost + shoe_cover_cost

/-- The total cost of materials for the costume is $479.63 --/
theorem costume_cost_theorem : 
  costume_cost 12 4 3 3 2 5 2.5 2.5 1.5 1.5 1 1.5 2 4 = 479.63 := by
  sorry

end NUMINAMATH_CALUDE_costume_cost_theorem_l3266_326616


namespace NUMINAMATH_CALUDE_paint_area_is_123_l3266_326669

/-- Calculates the area to be painted on a wall with given dimensions and window areas -/
def area_to_paint (wall_height wall_length window1_height window1_width window2_height window2_width : ℝ) : ℝ :=
  let wall_area := wall_height * wall_length
  let window1_area := window1_height * window1_width
  let window2_area := window2_height * window2_width
  wall_area - (window1_area + window2_area)

/-- Theorem: The area to be painted is 123 square feet -/
theorem paint_area_is_123 :
  area_to_paint 10 15 3 5 2 6 = 123 := by
  sorry

#eval area_to_paint 10 15 3 5 2 6

end NUMINAMATH_CALUDE_paint_area_is_123_l3266_326669


namespace NUMINAMATH_CALUDE_sum_integer_part_l3266_326634

theorem sum_integer_part : ⌊(2010 : ℝ) / 1000 + (1219 : ℝ) / 100 + (27 : ℝ) / 10⌋ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_integer_part_l3266_326634


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l3266_326627

theorem cos_alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/6) = 1/3) :
  Real.cos (α + π/3) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l3266_326627


namespace NUMINAMATH_CALUDE_shift_cosine_to_sine_l3266_326678

theorem shift_cosine_to_sine (x : ℝ) :
  let original := λ x : ℝ => 2 * Real.cos (2 * x)
  let shifted := λ x : ℝ => original (x - π / 8)
  let target := λ x : ℝ => 2 * Real.sin (2 * x + π / 4)
  0 < π / 8 ∧ π / 8 < π / 2 →
  shifted = target := by sorry

end NUMINAMATH_CALUDE_shift_cosine_to_sine_l3266_326678


namespace NUMINAMATH_CALUDE_solve_equation_l3266_326688

theorem solve_equation (m : ℝ) : m + (m + 2) + (m + 4) = 21 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3266_326688


namespace NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l3266_326617

/-- The number of steps Petya walks from the first to the third floor -/
def petya_steps : ℕ := 36

/-- The number of steps Vasya walks from the first floor to his floor -/
def vasya_steps : ℕ := 72

/-- The floor on which Vasya lives -/
def vasya_floor : ℕ := 5

/-- Theorem stating that Vasya lives on the 5th floor given the conditions -/
theorem vasya_lives_on_fifth_floor :
  (petya_steps / 2 = vasya_steps / vasya_floor) →
  vasya_floor = 5 := by
  sorry

end NUMINAMATH_CALUDE_vasya_lives_on_fifth_floor_l3266_326617


namespace NUMINAMATH_CALUDE_plant_structure_unique_solution_l3266_326619

/-- Represents a plant with branches and small branches -/
structure Plant where
  branches : ℕ
  smallBranchesPerBranch : ℕ

/-- The total number of parts (main stem, branches, and small branches) in a plant -/
def totalParts (p : Plant) : ℕ :=
  1 + p.branches + p.branches * p.smallBranchesPerBranch

/-- Theorem stating that a plant with 6 small branches per branch satisfies the given conditions -/
theorem plant_structure : ∃ (p : Plant), p.smallBranchesPerBranch = 6 ∧ totalParts p = 43 :=
  sorry

/-- Theorem proving that 6 is the unique solution for the number of small branches per branch -/
theorem unique_solution (p : Plant) (h : totalParts p = 43) : p.smallBranchesPerBranch = 6 :=
  sorry

end NUMINAMATH_CALUDE_plant_structure_unique_solution_l3266_326619


namespace NUMINAMATH_CALUDE_pipe_length_is_35_l3266_326666

/-- The length of the pipe in meters -/
def pipe_length : ℝ := 35

/-- The length of Yura's step in meters -/
def step_length : ℝ := 1

/-- The number of steps Yura took against the movement of the tractor -/
def steps_against : ℕ := 20

/-- The number of steps Yura took with the movement of the tractor -/
def steps_with : ℕ := 140

/-- Theorem stating that the pipe length is 35 meters -/
theorem pipe_length_is_35 : 
  ∃ (x : ℝ), 
    (step_length * steps_against : ℝ) = pipe_length - x ∧ 
    (step_length * steps_with : ℝ) = pipe_length + 7 * x ∧
    pipe_length = 35 := by sorry

end NUMINAMATH_CALUDE_pipe_length_is_35_l3266_326666


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3266_326647

theorem quadratic_inequality_theorem (c : ℝ) : 
  (∀ (a b : ℝ), (c^2 - 2*a*c + b) * (c^2 + 2*a*c + b) ≥ a^2 - 2*a^2 + b) ↔ 
  (c = 1/2 ∨ c = -1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3266_326647


namespace NUMINAMATH_CALUDE_product_maximized_at_11_l3266_326631

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ

/-- Calculates the nth term of a geometric sequence -/
def nthTerm (gs : GeometricSequence) (n : ℕ) : ℝ :=
  gs.a₁ * gs.q ^ (n - 1)

/-- Calculates the product of the first n terms of a geometric sequence -/
def productFirstNTerms (gs : GeometricSequence) (n : ℕ) : ℝ :=
  (gs.a₁ ^ n) * (gs.q ^ (n * (n - 1) / 2))

/-- Theorem: The product of the first n terms is maximized when n = 11 for the given sequence -/
theorem product_maximized_at_11 (gs : GeometricSequence) 
    (h1 : gs.a₁ = 1536) (h2 : gs.q = -1/2) :
    ∀ k : ℕ, k ≠ 11 → productFirstNTerms gs 11 ≥ productFirstNTerms gs k := by
  sorry

end NUMINAMATH_CALUDE_product_maximized_at_11_l3266_326631


namespace NUMINAMATH_CALUDE_magnitude_of_5_minus_12i_l3266_326608

theorem magnitude_of_5_minus_12i : Complex.abs (5 - 12 * Complex.I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_5_minus_12i_l3266_326608


namespace NUMINAMATH_CALUDE_function_relationship_l3266_326604

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = f (-x + 2))

-- State the theorem
theorem function_relationship : f 2.5 > f 1 ∧ f 1 > f 3.5 :=
sorry

end NUMINAMATH_CALUDE_function_relationship_l3266_326604


namespace NUMINAMATH_CALUDE_person_height_calculation_l3266_326630

/-- The height of a person used to determine the depth of water -/
def personHeight : ℝ := 6

/-- The depth of the water in feet -/
def waterDepth : ℝ := 60

/-- The relationship between the water depth and the person's height -/
def depthRelation : Prop := waterDepth = 10 * personHeight

theorem person_height_calculation : 
  depthRelation → personHeight = 6 := by sorry

end NUMINAMATH_CALUDE_person_height_calculation_l3266_326630


namespace NUMINAMATH_CALUDE_sin_30_deg_value_l3266_326606

theorem sin_30_deg_value (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (3 * x)) :
  f (Real.sin (π / 6)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_deg_value_l3266_326606


namespace NUMINAMATH_CALUDE_square_difference_equals_sixteen_l3266_326697

theorem square_difference_equals_sixteen
  (x y : ℝ)
  (sum_eq : x + y = 6)
  (product_eq : x * y = 5) :
  (x - y)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_sixteen_l3266_326697


namespace NUMINAMATH_CALUDE_value_of_a_l3266_326670

theorem value_of_a : ∀ a : ℕ, 
  (a * (9^3) = 3 * (15^5)) → 
  (a = 5^5) → 
  (a = 3125) := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3266_326670


namespace NUMINAMATH_CALUDE_determinant_equality_l3266_326663

theorem determinant_equality (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 7 →
  Matrix.det !![p + r, q + s; r, s] = 7 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l3266_326663


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3266_326686

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℂ, X^44 + X^33 + X^22 + X^11 + 1 = (X^4 + X^3 + X^2 + X + 1) * q :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3266_326686


namespace NUMINAMATH_CALUDE_point_on_line_l3266_326651

/-- Given five points O, A, B, C, D on a straight line with specified distances,
    and points P and Q satisfying certain ratio conditions, prove that OQ has the given value. -/
theorem point_on_line (a b c d : ℝ) :
  let O := 0
  let A := 2 * a
  let B := 4 * b
  let C := 5 * c
  let D := 7 * d
  let P := (14 * b * d - 10 * a * c) / (2 * a - 4 * b + 7 * d - 5 * c)
  let Q := (14 * c * d - 10 * b * c) / (5 * c - 7 * d)
  (A - P) / (P - D) = (B - P) / (P - C) →
  (Q - C) / (D - Q) = (B - C) / (D - C) →
  Q = (14 * c * d - 10 * b * c) / (5 * c - 7 * d) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_l3266_326651


namespace NUMINAMATH_CALUDE_emily_coloring_books_l3266_326612

/-- 
Given Emily's initial number of coloring books, the number she gave away,
and her current total, prove that she bought 14 coloring books.
-/
theorem emily_coloring_books 
  (initial : ℕ) 
  (given_away : ℕ) 
  (current_total : ℕ) 
  (h1 : initial = 7)
  (h2 : given_away = 2)
  (h3 : current_total = 19) :
  current_total - (initial - given_away) = 14 := by
  sorry

end NUMINAMATH_CALUDE_emily_coloring_books_l3266_326612


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3266_326685

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 3}

theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3266_326685


namespace NUMINAMATH_CALUDE_sine_of_angle_l3266_326618

/-- Given an angle α with vertex at the origin, initial side on the non-negative x-axis,
    and terminal side in the third quadrant intersecting the unit circle at (-√5/5, m),
    prove that sin α = -2√5/5 -/
theorem sine_of_angle (α : Real) (m : Real) : 
  ((-Real.sqrt 5 / 5) ^ 2 + m ^ 2 = 1) →  -- Point on unit circle
  (m < 0) →  -- In third quadrant
  (Real.sin α = m) →  -- Definition of sine
  (Real.sin α = -2 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sine_of_angle_l3266_326618


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l3266_326654

theorem walking_rate_ratio 
  (D : ℝ) -- Distance to school
  (R : ℝ) -- Usual walking rate
  (R' : ℝ) -- New walking rate
  (h1 : D = R * 21) -- Usual time equation
  (h2 : D = R' * 18) -- New time equation
  : R' / R = 7 / 6 := by sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l3266_326654


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3266_326607

theorem fraction_meaningful (a : ℝ) : 
  (∃ x : ℝ, x = (a + 1) / (2 * a - 1)) ↔ a ≠ 1/2 :=
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3266_326607


namespace NUMINAMATH_CALUDE_correct_percentage_calculation_l3266_326626

/-- Calculates the overall percentage of correct answers across multiple tests -/
def overallPercentage (testSizes : List Nat) (scores : List Rat) : Rat :=
  sorry

/-- Rounds a rational number to the nearest whole number -/
def roundToNearest (x : Rat) : Nat :=
  sorry

theorem correct_percentage_calculation :
  let testSizes : List Nat := [40, 30, 20]
  let scores : List Rat := [65/100, 85/100, 75/100]
  roundToNearest (overallPercentage testSizes scores * 100) = 74 :=
by sorry

end NUMINAMATH_CALUDE_correct_percentage_calculation_l3266_326626


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_42_l3266_326605

theorem no_primes_divisible_by_42 : 
  ∀ p : ℕ, Prime p → ¬(42 ∣ p) :=
by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_42_l3266_326605


namespace NUMINAMATH_CALUDE_white_balls_count_l3266_326694

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) 
  (h_total : total = 60)
  (h_green : green = 18)
  (h_yellow : yellow = 8)
  (h_red : red = 5)
  (h_purple : purple = 7)
  (h_prob : prob_not_red_purple = 4/5) :
  total - (green + yellow + red + purple) = 22 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l3266_326694


namespace NUMINAMATH_CALUDE_min_socks_for_pair_is_four_l3266_326677

/-- Represents a drawer of socks with three colors -/
structure SockDrawer :=
  (white : Nat)
  (green : Nat)
  (red : Nat)

/-- Ensures that there is at least one sock of each color -/
def hasAllColors (drawer : SockDrawer) : Prop :=
  drawer.white > 0 ∧ drawer.green > 0 ∧ drawer.red > 0

/-- The minimum number of socks needed to ensure at least two of the same color -/
def minSocksForPair (drawer : SockDrawer) : Nat :=
  4

theorem min_socks_for_pair_is_four (drawer : SockDrawer) 
  (h : hasAllColors drawer) : 
  minSocksForPair drawer = 4 := by
  sorry

#check min_socks_for_pair_is_four

end NUMINAMATH_CALUDE_min_socks_for_pair_is_four_l3266_326677


namespace NUMINAMATH_CALUDE_unique_cyclic_number_l3266_326681

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def same_digits (a b : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → (∃ k, a / 10^k % 10 = d) ↔ (∃ k, b / 10^k % 10 = d)

theorem unique_cyclic_number : 
  ∃! N : ℕ, is_six_digit N ∧ 
    (∀ k : ℕ, 2 ≤ k ∧ k ≤ 6 → 
      is_six_digit (k * N) ∧ 
      same_digits N (k * N) ∧ 
      N ≠ k * N) ∧
    N = 142857 :=
sorry

end NUMINAMATH_CALUDE_unique_cyclic_number_l3266_326681


namespace NUMINAMATH_CALUDE_min_detectors_for_gold_coins_l3266_326671

/-- Represents a grid of unit squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a subgrid within a larger grid -/
structure Subgrid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the minimum number of detectors needed -/
def min_detectors (g : Grid) (s : Subgrid) : ℕ := sorry

/-- The main theorem stating the minimum number of detectors needed -/
theorem min_detectors_for_gold_coins (g : Grid) (s : Subgrid) :
  g.rows = 2017 ∧ g.cols = 2017 ∧ s.rows = 1500 ∧ s.cols = 1500 →
  min_detectors g s = 1034 := by sorry

end NUMINAMATH_CALUDE_min_detectors_for_gold_coins_l3266_326671


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l3266_326672

theorem min_value_of_function (x : ℝ) (h : x > 0) : 4 * x + 9 / x ≥ 12 :=
sorry

theorem min_value_achieved : ∃ x : ℝ, x > 0 ∧ 4 * x + 9 / x = 12 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l3266_326672


namespace NUMINAMATH_CALUDE_pharmacy_tubs_in_storage_l3266_326636

theorem pharmacy_tubs_in_storage (total_needed : ℕ) (bought_usual : ℕ) : ℕ :=
  let tubs_in_storage := total_needed - (bought_usual + bought_usual / 3)
  by
    sorry

#check pharmacy_tubs_in_storage 100 60

end NUMINAMATH_CALUDE_pharmacy_tubs_in_storage_l3266_326636
