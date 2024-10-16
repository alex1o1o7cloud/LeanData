import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_sum_l1687_168748

theorem quadratic_sum (a b : ℝ) : 
  ({b} : Set ℝ) = {x : ℝ | a * x^2 - 4 * x + 1 = 0} → 
  a + b = 1/4 ∨ a + b = 9/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1687_168748


namespace NUMINAMATH_CALUDE_scissors_in_drawer_final_scissors_count_l1687_168710

theorem scissors_in_drawer (initial : ℕ) (added : ℕ) (removed : ℕ) : ℕ :=
  initial + added - removed

theorem final_scissors_count : scissors_in_drawer 54 22 15 = 61 := by
  sorry

end NUMINAMATH_CALUDE_scissors_in_drawer_final_scissors_count_l1687_168710


namespace NUMINAMATH_CALUDE_divisibility_by_3804_l1687_168728

theorem divisibility_by_3804 (n : ℕ+) :
  ∃ k : ℤ, (n.val ^ 3 - n.val : ℤ) * (5 ^ (8 * n.val + 4) + 3 ^ (4 * n.val + 2)) = 3804 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_3804_l1687_168728


namespace NUMINAMATH_CALUDE_no_negative_one_in_sequence_l1687_168793

def recurrence_sequence (p : ℕ) : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * recurrence_sequence p (n + 1) - p * recurrence_sequence p n

theorem no_negative_one_in_sequence (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) (h_not_five : p ≠ 5) :
  ∀ n, recurrence_sequence p n ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_no_negative_one_in_sequence_l1687_168793


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1687_168756

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the number of quadruplets that must be in the starting lineup
def quadruplets_in_lineup : ℕ := 3

-- Theorem statement
theorem basketball_team_selection :
  (Nat.choose num_quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)) = 1980 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1687_168756


namespace NUMINAMATH_CALUDE_cos_negative_300_degrees_l1687_168783

theorem cos_negative_300_degrees :
  Real.cos ((-300 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_300_degrees_l1687_168783


namespace NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_1365_l1687_168741

theorem sum_largest_smallest_prime_factors_1365 : ∃ (p q : ℕ), 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  p ∣ 1365 ∧ 
  q ∣ 1365 ∧ 
  (∀ r : ℕ, Nat.Prime r → r ∣ 1365 → p ≤ r ∧ r ≤ q) ∧ 
  p + q = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_1365_l1687_168741


namespace NUMINAMATH_CALUDE_jane_reading_probability_l1687_168735

theorem jane_reading_probability (p : ℚ) (h : p = 5/8) :
  1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_jane_reading_probability_l1687_168735


namespace NUMINAMATH_CALUDE_x_equation_value_l1687_168753

theorem x_equation_value (x : ℝ) (h : x + 1/x = 3) :
  x^10 - 6*x^6 + x^2 = -328*x^2 := by
sorry

end NUMINAMATH_CALUDE_x_equation_value_l1687_168753


namespace NUMINAMATH_CALUDE_general_term_formula_l1687_168745

/-- The sequence term for a given positive integer n -/
def a (n : ℕ+) : ℚ :=
  (4 * n^2 + n - 1) / (2 * n + 1)

/-- The first part of each term in the sequence -/
def b (n : ℕ+) : ℕ :=
  2 * n - 1

/-- The second part of each term in the sequence -/
def c (n : ℕ+) : ℚ :=
  n / (2 * n + 1)

/-- Theorem stating that the general term formula is correct -/
theorem general_term_formula (n : ℕ+) : 
  a n = (b n : ℚ) + c n :=
sorry

end NUMINAMATH_CALUDE_general_term_formula_l1687_168745


namespace NUMINAMATH_CALUDE_binary_to_base4_example_l1687_168742

/-- Converts a binary (base 2) number to its base 4 representation -/
def binary_to_base4 (binary : ℕ) : ℕ := sorry

/-- The binary number 1011010010₂ -/
def binary_num : ℕ := 722  -- 1011010010₂ in decimal

/-- Theorem stating that the base 4 representation of 1011010010₂ is 3122₄ -/
theorem binary_to_base4_example : binary_to_base4 binary_num = 3122 := by sorry

end NUMINAMATH_CALUDE_binary_to_base4_example_l1687_168742


namespace NUMINAMATH_CALUDE_absolute_value_fraction_less_than_one_l1687_168705

theorem absolute_value_fraction_less_than_one (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : |(x - y) / (1 - x * y)| < 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_less_than_one_l1687_168705


namespace NUMINAMATH_CALUDE_ellen_dinner_calories_l1687_168725

/-- Calculates the remaining calories for dinner given a daily limit and calories consumed for breakfast, lunch, and snack. -/
def remaining_calories_for_dinner (daily_limit : ℕ) (breakfast : ℕ) (lunch : ℕ) (snack : ℕ) : ℕ :=
  daily_limit - (breakfast + lunch + snack)

/-- Proves that given the specific calorie values in the problem, the remaining calories for dinner is 832. -/
theorem ellen_dinner_calories : 
  remaining_calories_for_dinner 2200 353 885 130 = 832 := by
sorry

end NUMINAMATH_CALUDE_ellen_dinner_calories_l1687_168725


namespace NUMINAMATH_CALUDE_laptop_selection_theorem_l1687_168722

-- Define the number of laptops of each type
def typeA : ℕ := 4
def typeB : ℕ := 5

-- Define the total number of laptops to be selected
def selectTotal : ℕ := 3

-- Define the function to calculate the number of selections
def numSelections : ℕ := 
  Nat.choose typeA 2 * Nat.choose typeB 1 + 
  Nat.choose typeA 1 * Nat.choose typeB 2

-- Theorem statement
theorem laptop_selection_theorem : numSelections = 70 := by
  sorry

end NUMINAMATH_CALUDE_laptop_selection_theorem_l1687_168722


namespace NUMINAMATH_CALUDE_candy_store_spend_l1687_168703

def weekly_allowance : ℚ := 3/2

def arcade_spend (allowance : ℚ) : ℚ := (3/5) * allowance

def toy_store_spend (remaining : ℚ) : ℚ := (1/3) * remaining

theorem candy_store_spend :
  let remaining_after_arcade := weekly_allowance - arcade_spend weekly_allowance
  let remaining_after_toy := remaining_after_arcade - toy_store_spend remaining_after_arcade
  remaining_after_toy = 2/5 := by sorry

end NUMINAMATH_CALUDE_candy_store_spend_l1687_168703


namespace NUMINAMATH_CALUDE_contradiction_proof_l1687_168768

theorem contradiction_proof (a b : ℕ) : a < 2 → b < 2 → a + b < 3 := by
  sorry

end NUMINAMATH_CALUDE_contradiction_proof_l1687_168768


namespace NUMINAMATH_CALUDE_area_of_triangle_MOI_l1687_168761

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  AB = 15 ∧ AC = 8 ∧ BC = 7

-- Define the circumcenter O
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the incenter I
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the point M
def point_M (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the tangency condition for M
def tangency_condition (M : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop := sorry

-- Calculate the area of a triangle given three points
def triangle_area (P Q R : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_MOI 
  (A B C : ℝ × ℝ) 
  (h_triangle : triangle_ABC A B C) :
  let O := circumcenter A B C
  let I := incenter A B C
  let M := point_M A B C
  tangency_condition M A B C →
  triangle_area M O I = 7/4 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_MOI_l1687_168761


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l1687_168715

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.03 * L) (h2 : B' = B * (1 + 0.06)) :
  L' * B' = 1.0918 * (L * B) :=
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l1687_168715


namespace NUMINAMATH_CALUDE_full_spots_is_186_l1687_168770

/-- Represents a parking garage with open spots on each level -/
structure ParkingGarage where
  levels : Nat
  spotsPerLevel : Nat
  openSpotsFirstLevel : Nat
  openSpotsSecondLevel : Nat
  openSpotsThirdLevel : Nat
  openSpotsFourthLevel : Nat

/-- Calculates the number of full parking spots in the garage -/
def fullParkingSpots (garage : ParkingGarage) : Nat :=
  garage.levels * garage.spotsPerLevel -
  (garage.openSpotsFirstLevel + garage.openSpotsSecondLevel +
   garage.openSpotsThirdLevel + garage.openSpotsFourthLevel)

/-- Theorem stating that the number of full parking spots is 186 -/
theorem full_spots_is_186 (garage : ParkingGarage)
  (h1 : garage.levels = 4)
  (h2 : garage.spotsPerLevel = 100)
  (h3 : garage.openSpotsFirstLevel = 58)
  (h4 : garage.openSpotsSecondLevel = garage.openSpotsFirstLevel + 2)
  (h5 : garage.openSpotsThirdLevel = garage.openSpotsSecondLevel + 5)
  (h6 : garage.openSpotsFourthLevel = 31) :
  fullParkingSpots garage = 186 := by
  sorry

#eval fullParkingSpots {
  levels := 4,
  spotsPerLevel := 100,
  openSpotsFirstLevel := 58,
  openSpotsSecondLevel := 60,
  openSpotsThirdLevel := 65,
  openSpotsFourthLevel := 31
}

end NUMINAMATH_CALUDE_full_spots_is_186_l1687_168770


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l1687_168709

theorem alpha_beta_sum (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α = 1) 
  (h2 : β^3 - 3*β^2 + 5*β = 5) : 
  α + β = 2 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l1687_168709


namespace NUMINAMATH_CALUDE_balloon_permutations_l1687_168711

/-- The number of distinct arrangements of letters in "balloon" -/
def balloon_arrangements : ℕ := 1260

/-- The total number of letters in "balloon" -/
def total_letters : ℕ := 7

/-- The frequency of each letter in "balloon" -/
def letter_frequency : List ℕ := [1, 1, 2, 2, 1]

theorem balloon_permutations :
  balloon_arrangements = Nat.factorial total_letters / (List.prod letter_frequency) := by
  sorry

end NUMINAMATH_CALUDE_balloon_permutations_l1687_168711


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1687_168767

theorem trigonometric_identities (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin (π - θ) + Real.cos (θ - π)) / (Real.sin (θ + π) + Real.cos (θ + π)) = -1/3 ∧
  Real.sin (2 * θ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1687_168767


namespace NUMINAMATH_CALUDE_math_team_count_is_480_l1687_168752

/-- The number of ways to form a six-member math team with 3 girls and 3 boys 
    from a club of 4 girls and 6 boys, where one team member is selected as captain -/
def math_team_count : ℕ := sorry

/-- The number of girls in the math club -/
def girls_in_club : ℕ := 4

/-- The number of boys in the math club -/
def boys_in_club : ℕ := 6

/-- The number of girls required in the team -/
def girls_in_team : ℕ := 3

/-- The number of boys required in the team -/
def boys_in_team : ℕ := 3

/-- The total number of team members -/
def team_size : ℕ := girls_in_team + boys_in_team

theorem math_team_count_is_480 : 
  math_team_count = (Nat.choose girls_in_club girls_in_team) * 
                    (Nat.choose boys_in_club boys_in_team) * 
                    team_size := by sorry

end NUMINAMATH_CALUDE_math_team_count_is_480_l1687_168752


namespace NUMINAMATH_CALUDE_journey_speed_proof_l1687_168718

/-- Proves that given a journey of 120 miles in 120 minutes, with average speeds of 50 mph and 60 mph
for the first and second 40-minute segments respectively, the average speed for the last 40-minute
segment is 70 mph. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 120 →
  total_time = 120 →
  speed1 = 50 →
  speed2 = 60 →
  ∃ (speed3 : ℝ), speed3 = 70 ∧ (speed1 + speed2 + speed3) / 3 = total_distance / (total_time / 60) :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l1687_168718


namespace NUMINAMATH_CALUDE_grace_total_pennies_l1687_168760

/-- The value of a dime in pennies -/
def dime_value : ℕ := 10

/-- The value of a coin in pennies -/
def coin_value : ℕ := 5

/-- The number of dimes Grace has -/
def grace_dimes : ℕ := 10

/-- The number of coins Grace has -/
def grace_coins : ℕ := 10

/-- The total value of Grace's dimes and coins in pennies -/
def total_value : ℕ := grace_dimes * dime_value + grace_coins * coin_value

theorem grace_total_pennies : total_value = 150 := by sorry

end NUMINAMATH_CALUDE_grace_total_pennies_l1687_168760


namespace NUMINAMATH_CALUDE_trains_crossing_time_l1687_168724

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length_A length_B speed_A speed_B : ℝ) 
  (h1 : length_A = 200)
  (h2 : length_B = 250)
  (h3 : speed_A = 72)
  (h4 : speed_B = 18) : 
  (length_A + length_B) / ((speed_A + speed_B) * (1000 / 3600)) = 18 := by
  sorry

#check trains_crossing_time

end NUMINAMATH_CALUDE_trains_crossing_time_l1687_168724


namespace NUMINAMATH_CALUDE_monthly_income_of_P_l1687_168778

/-- Given the average monthly incomes of three people, prove the monthly income of P. -/
theorem monthly_income_of_P (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  (P + R) / 2 = 5200 →
  P = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_of_P_l1687_168778


namespace NUMINAMATH_CALUDE_system_solution_unique_l1687_168714

def system_solution (a₁ a₂ a₃ a₄ : ℝ) (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧
  (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
  (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
  (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
  (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1)

theorem system_solution_unique (a₁ a₂ a₃ a₄ : ℝ) :
  ∃! (x₁ x₂ x₃ x₄ : ℝ), system_solution a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ ∧
    x₁ = 1 / (a₄ - a₁) ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 1 / (a₄ - a₁) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1687_168714


namespace NUMINAMATH_CALUDE_inequality_proof_l1687_168782

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * Real.sqrt (a * b)) / (Real.sqrt a + Real.sqrt b) ≤ (a * b) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1687_168782


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_solutions_equation3_l1687_168751

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := 4 * (x - 1)^2 - 36 = 0
def equation2 (x : ℝ) : Prop := x^2 + 2*x - 3 = 0
def equation3 (x : ℝ) : Prop := x*(x - 4) = 8 - 2*x

-- Theorem stating the solutions for equation1
theorem solutions_equation1 : 
  ∀ x : ℝ, equation1 x ↔ (x = 4 ∨ x = -2) :=
sorry

-- Theorem stating the solutions for equation2
theorem solutions_equation2 : 
  ∀ x : ℝ, equation2 x ↔ (x = -3 ∨ x = 1) :=
sorry

-- Theorem stating the solutions for equation3
theorem solutions_equation3 : 
  ∀ x : ℝ, equation3 x ↔ (x = 4 ∨ x = -2) :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_solutions_equation3_l1687_168751


namespace NUMINAMATH_CALUDE_box_price_difference_l1687_168720

/-- The price difference between box C and box A -/
def price_difference (a b c : ℚ) : ℚ := c - a

/-- The conditions from the problem -/
def problem_conditions (a b c : ℚ) : Prop :=
  a + b + c = 9 ∧ 3*a + 2*b + c = 16

theorem box_price_difference :
  ∀ a b c : ℚ, problem_conditions a b c → price_difference a b c = 2 := by
  sorry

end NUMINAMATH_CALUDE_box_price_difference_l1687_168720


namespace NUMINAMATH_CALUDE_system_solution_conditions_l1687_168750

theorem system_solution_conditions (m : ℝ) :
  let x := (1 + 2*m) / 3
  let y := (1 - m) / 3
  (x + 2*y = 1 ∧ x - y = m) ∧ (x > 1 ∧ y ≥ -1) ↔ 1 < m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_conditions_l1687_168750


namespace NUMINAMATH_CALUDE_race_result_l1687_168717

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ → ℝ

/-- The race setup -/
structure Race where
  sasha : Runner
  lesha : Runner
  kolya : Runner
  length : ℝ

def Race.valid (race : Race) : Prop :=
  race.sasha.speed > 0 ∧ 
  race.lesha.speed > 0 ∧ 
  race.kolya.speed > 0 ∧
  race.length > 0 ∧
  -- When Sasha finishes, Lesha is 10 meters behind
  race.sasha.position (race.length / race.sasha.speed) = race.length ∧
  race.lesha.position (race.length / race.sasha.speed) = race.length - 10 ∧
  -- When Lesha finishes, Kolya is 10 meters behind
  race.lesha.position (race.length / race.lesha.speed) = race.length ∧
  race.kolya.position (race.length / race.lesha.speed) = race.length - 10 ∧
  -- All runners have constant speeds
  ∀ t, race.sasha.position t = race.sasha.speed * t ∧
       race.lesha.position t = race.lesha.speed * t ∧
       race.kolya.position t = race.kolya.speed * t

theorem race_result (race : Race) (h : race.valid) : 
  race.kolya.position (race.length / race.sasha.speed) = race.length - 19 := by
  sorry

end NUMINAMATH_CALUDE_race_result_l1687_168717


namespace NUMINAMATH_CALUDE_max_truck_speed_l1687_168719

theorem max_truck_speed (distance : ℝ) (hourly_cost : ℝ) (fixed_cost : ℝ) (max_total_cost : ℝ) :
  distance = 125 →
  hourly_cost = 30 →
  fixed_cost = 1000 →
  max_total_cost = 1200 →
  ∃ (max_speed : ℝ),
    max_speed = 75 ∧
    ∀ (speed : ℝ),
      speed > 0 →
      (distance / speed) * hourly_cost + fixed_cost + 2 * speed ≤ max_total_cost →
      speed ≤ max_speed :=
sorry

end NUMINAMATH_CALUDE_max_truck_speed_l1687_168719


namespace NUMINAMATH_CALUDE_stickers_used_for_decoration_l1687_168766

def initial_stickers : ℕ := 20
def bought_stickers : ℕ := 26
def birthday_stickers : ℕ := 20
def given_away_stickers : ℕ := 6
def left_stickers : ℕ := 2

theorem stickers_used_for_decoration :
  initial_stickers + bought_stickers + birthday_stickers - given_away_stickers - left_stickers = 58 :=
by sorry

end NUMINAMATH_CALUDE_stickers_used_for_decoration_l1687_168766


namespace NUMINAMATH_CALUDE_every_nonzero_nat_is_product_of_primes_l1687_168749

theorem every_nonzero_nat_is_product_of_primes :
  ∀ n : ℕ, n ≠ 0 → ∃ (primes : List ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ n = primes.prod := by
  sorry

end NUMINAMATH_CALUDE_every_nonzero_nat_is_product_of_primes_l1687_168749


namespace NUMINAMATH_CALUDE_alternating_number_composite_l1687_168737

def alternating_number (k : ℕ) : ℕ := 
  (10^(2*k+1) - 1) / 99

theorem alternating_number_composite (k : ℕ) (h : k ≥ 2) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ alternating_number k = a * b :=
sorry

end NUMINAMATH_CALUDE_alternating_number_composite_l1687_168737


namespace NUMINAMATH_CALUDE_sequence_sum_l1687_168721

/-- Given a geometric sequence and an arithmetic sequence with specific properties, 
    prove that the sum of two terms in the arithmetic sequence equals 8. -/
theorem sequence_sum (a b : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  (a 3 * a 11 = 4 * a 7) →                              -- given condition for geometric sequence
  (∀ n : ℕ, b (n + 1) - b n = b (n + 2) - b (n + 1)) →  -- arithmetic sequence condition
  (a 7 = b 7) →                                         -- given condition relating both sequences
  b 5 + b 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1687_168721


namespace NUMINAMATH_CALUDE_playground_area_l1687_168797

/-- The area of a rectangular playground with given conditions -/
theorem playground_area : 
  ∀ (width length : ℝ),
  length = 3 * width + 30 →
  2 * (width + length) = 730 →
  width * length = 23554.6875 := by
sorry

end NUMINAMATH_CALUDE_playground_area_l1687_168797


namespace NUMINAMATH_CALUDE_no_solution_for_sock_problem_l1687_168776

theorem no_solution_for_sock_problem : ¬∃ (n m : ℕ), 
  n + m = 2009 ∧ (n^2 - 2*n*m + m^2) = (n + m) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_sock_problem_l1687_168776


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l1687_168789

theorem complex_magnitude_theorem (r : ℝ) (z : ℂ) 
  (h1 : |r| < 6) 
  (h2 : z + 9 / z = r) : 
  Complex.abs z = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l1687_168789


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1687_168763

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2, 3}

-- Define set N
def N : Set Nat := {2, 3, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ M) ∩ N = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1687_168763


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_ii_l1687_168798

/-- A linear function with slope k and y-intercept b -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

/-- Quadrant II is the region where x < 0 and y > 0 -/
def InQuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem linear_function_not_in_quadrant_ii :
  ∀ x : ℝ, ¬InQuadrantII x (LinearFunction 3 (-2) x) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_ii_l1687_168798


namespace NUMINAMATH_CALUDE_product_equals_two_sevenths_l1687_168747

/-- Sequence defined by a₀ = 1/3 and aₙ = 2 + (aₙ₋₁ - 2)² for n ≥ 1 -/
def a : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 2 + (a n - 2)^2

/-- The infinite product of the sequence elements -/
noncomputable def infiniteProduct : ℚ := ∏' n, a n

theorem product_equals_two_sevenths : infiniteProduct = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_two_sevenths_l1687_168747


namespace NUMINAMATH_CALUDE_biking_problem_solution_l1687_168707

/-- Represents the problem of Andrea and Lauren biking in a park -/
def BikingProblem (park_length : ℝ) (distance_decrease_rate : ℝ) (andrea_initial_time : ℝ) (andrea_wait_time : ℝ) : Prop :=
  ∃ (lauren_speed : ℝ),
    lauren_speed > 0 ∧
    2 * lauren_speed + lauren_speed = distance_decrease_rate ∧
    let initial_distance := distance_decrease_rate * andrea_initial_time
    let remaining_distance := park_length - initial_distance
    let lauren_time := remaining_distance / lauren_speed
    andrea_initial_time + andrea_wait_time + lauren_time = 79

/-- The theorem stating the solution to the biking problem -/
theorem biking_problem_solution :
  BikingProblem 24 0.8 7 3 := by
  sorry

end NUMINAMATH_CALUDE_biking_problem_solution_l1687_168707


namespace NUMINAMATH_CALUDE_same_tangent_line_implies_b_value_l1687_168740

def f (x : ℝ) : ℝ := 2 * x^3 + 1
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x^2 - b

theorem same_tangent_line_implies_b_value :
  ∀ b : ℝ, (∃ x₀ : ℝ, (deriv f x₀ = deriv (g b) x₀) ∧ 
    (f x₀ = g b x₀)) → (b = 0 ∨ b = -1) :=
by sorry

end NUMINAMATH_CALUDE_same_tangent_line_implies_b_value_l1687_168740


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l1687_168771

theorem square_plus_inverse_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l1687_168771


namespace NUMINAMATH_CALUDE_fine_on_fifth_day_l1687_168706

/-- Calculates the fine for a given day based on the previous day's fine -/
def nextDayFine (prevFine : ℚ) : ℚ :=
  min (prevFine + 0.3) (prevFine * 2)

/-- Calculates the fine for a specified number of days -/
def fineAfterDays (days : ℕ) : ℚ :=
  match days with
  | 0 => 0
  | 1 => 0.07
  | n + 1 => nextDayFine (fineAfterDays n)

theorem fine_on_fifth_day :
  fineAfterDays 5 = 0.86 := by sorry

end NUMINAMATH_CALUDE_fine_on_fifth_day_l1687_168706


namespace NUMINAMATH_CALUDE_wallpaper_removal_time_l1687_168794

/-- Calculates the time required to remove wallpaper from remaining walls -/
def time_to_remove_wallpaper (time_per_wall : ℕ) (dining_room_walls : ℕ) (living_room_walls : ℕ) (walls_completed : ℕ) : ℕ :=
  time_per_wall * (dining_room_walls + living_room_walls - walls_completed)

/-- Proves that given the conditions, the time required to remove the remaining wallpaper is 14 hours -/
theorem wallpaper_removal_time :
  let time_per_wall : ℕ := 2
  let dining_room_walls : ℕ := 4
  let living_room_walls : ℕ := 4
  let walls_completed : ℕ := 1
  time_to_remove_wallpaper time_per_wall dining_room_walls living_room_walls walls_completed = 14 := by
  sorry

#eval time_to_remove_wallpaper 2 4 4 1

end NUMINAMATH_CALUDE_wallpaper_removal_time_l1687_168794


namespace NUMINAMATH_CALUDE_sarah_candy_duration_l1687_168700

/-- The number of days Sarah's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  (neighbors_candy + sister_candy) / daily_consumption

/-- Proof that Sarah's candy will last 9 days -/
theorem sarah_candy_duration :
  candy_duration 66 15 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sarah_candy_duration_l1687_168700


namespace NUMINAMATH_CALUDE_prob_at_least_two_hits_eq_81_125_l1687_168795

/-- The probability of hitting a target in one shot. -/
def p : ℝ := 0.6

/-- The number of shots taken. -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in n shots. -/
def prob_at_least_two_hits : ℝ := 
  Finset.sum (Finset.range (n + 1) \ Finset.range 2) (λ k => 
    (n.choose k : ℝ) * p^k * (1 - p)^(n - k))

theorem prob_at_least_two_hits_eq_81_125 : 
  prob_at_least_two_hits = 81 / 125 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_hits_eq_81_125_l1687_168795


namespace NUMINAMATH_CALUDE_factorization_difference_l1687_168774

theorem factorization_difference (a b : ℤ) : 
  (∀ y : ℝ, 3 * y^2 - y - 18 = (3 * y + a) * (y + b)) → 
  a - b = -11 := by
sorry

end NUMINAMATH_CALUDE_factorization_difference_l1687_168774


namespace NUMINAMATH_CALUDE_tangent_line_at_neg_one_l1687_168716

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Theorem statement
theorem tangent_line_at_neg_one (a : ℝ) :
  f_derivative a 1 = 1 →
  ∃ y : ℝ, 9 * (-1) - y + 3 = 0 ∧
  y = f a (-1) ∧
  f_derivative a (-1) = 9 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_neg_one_l1687_168716


namespace NUMINAMATH_CALUDE_letter_value_puzzle_l1687_168733

theorem letter_value_puzzle (L E A D : ℤ) : 
  L = 15 →
  L + E + A + D = 41 →
  D + E + A + L = 45 →
  A + D + D + E + D = 53 →
  D = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_letter_value_puzzle_l1687_168733


namespace NUMINAMATH_CALUDE_artist_paintings_l1687_168758

theorem artist_paintings (paint_per_large : ℕ) (paint_per_small : ℕ) 
  (small_paintings : ℕ) (total_paint : ℕ) :
  paint_per_large = 3 →
  paint_per_small = 2 →
  small_paintings = 4 →
  total_paint = 17 →
  ∃ (large_paintings : ℕ), 
    large_paintings * paint_per_large + small_paintings * paint_per_small = total_paint ∧
    large_paintings = 3 :=
by sorry

end NUMINAMATH_CALUDE_artist_paintings_l1687_168758


namespace NUMINAMATH_CALUDE_polynomial_inequality_l1687_168779

theorem polynomial_inequality (n : ℕ) (hn : n > 1) :
  ∀ x : ℝ, x > 0 → x^n - n*x + n - 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l1687_168779


namespace NUMINAMATH_CALUDE_mary_change_theorem_l1687_168708

-- Define the ticket prices and discounts
def adult_price : ℚ := 2
def child_price : ℚ := 1
def first_child_discount : ℚ := 0.5
def second_child_discount : ℚ := 0.75
def third_child_discount : ℚ := 1
def sales_tax_rate : ℚ := 0.08
def amount_paid : ℚ := 20

-- Calculate the total cost before tax
def total_cost_before_tax : ℚ :=
  adult_price +
  child_price * first_child_discount +
  child_price * second_child_discount +
  child_price * third_child_discount

-- Calculate the sales tax
def sales_tax : ℚ := total_cost_before_tax * sales_tax_rate

-- Calculate the total cost including tax
def total_cost_with_tax : ℚ := total_cost_before_tax + sales_tax

-- Calculate the change
def change : ℚ := amount_paid - total_cost_with_tax

-- Theorem to prove
theorem mary_change_theorem : change = 15.41 := by sorry

end NUMINAMATH_CALUDE_mary_change_theorem_l1687_168708


namespace NUMINAMATH_CALUDE_first_question_percentage_l1687_168726

theorem first_question_percentage
  (second_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : second_correct = 0.55)
  (h2 : neither_correct = 0.20)
  (h3 : both_correct = 0.55) :
  ∃ (first_correct : ℝ), first_correct = 0.80 := by
sorry

end NUMINAMATH_CALUDE_first_question_percentage_l1687_168726


namespace NUMINAMATH_CALUDE_cubic_quartic_relation_l1687_168701

theorem cubic_quartic_relation (x y : ℝ) 
  (h1 : x^3 + y^3 + 1 / (x^3 + y^3) = 3) 
  (h2 : x + y = 2) : 
  x^4 + y^4 + 1 / (x^4 + y^4) = 257/16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_quartic_relation_l1687_168701


namespace NUMINAMATH_CALUDE_broken_line_exists_l1687_168746

-- Define the type for points
structure Point where
  x : ℝ
  y : ℝ

-- Define the set of marked points
def MarkedPoints : Set Point := sorry

-- Define the function to check if a point is marked
def isMarked (p : Point) : Prop := p ∈ MarkedPoints

-- Define the function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define the function to check if three points are collinear
def areCollinear (p1 p2 p3 : Point) : Prop := sorry

-- Define the start and end points
def A : Point := sorry
def B : Point := sorry

-- Theorem statement
theorem broken_line_exists :
  ∃ (P Q R : Point),
    isMarked P ∧ isMarked Q ∧ isMarked R ∧
    distance A P = distance P Q ∧
    distance P Q = distance Q R ∧
    distance Q R = distance R B ∧
    ¬ areCollinear A P Q ∧
    ¬ areCollinear P Q R ∧
    ¬ areCollinear Q R B ∧
    ∀ (X : Point),
      (X ∈ MarkedPoints ∧ X ≠ A ∧ X ≠ P ∧ X ≠ Q ∧ X ≠ R ∧ X ≠ B) →
      (distance A X > distance A P ∨ distance X P > distance A P) ∧
      (distance P X > distance P Q ∨ distance X Q > distance P Q) ∧
      (distance Q X > distance Q R ∨ distance X R > distance Q R) ∧
      (distance R X > distance R B ∨ distance X B > distance R B) :=
sorry

end NUMINAMATH_CALUDE_broken_line_exists_l1687_168746


namespace NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l1687_168772

theorem inscribed_circles_area_ratio (s : ℝ) (hs : s > 0) :
  let square_area := s^2
  let semicircle_area := (π * s^2) / 8
  let quarter_circle_area := (π * s^2) / 16
  let combined_area := semicircle_area + quarter_circle_area
  combined_area / square_area = 3 * π / 16 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l1687_168772


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l1687_168704

theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 960) 
  (h2 : num_moles = 5) : 
  total_weight / num_moles = 192 := by
sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l1687_168704


namespace NUMINAMATH_CALUDE_pradeep_exam_marks_l1687_168755

/-- The maximum marks in Pradeep's exam -/
def maximum_marks : ℕ := 928

/-- The percentage required to pass the exam -/
def pass_percentage : ℚ := 55 / 100

/-- The marks Pradeep obtained -/
def pradeep_marks : ℕ := 400

/-- The number of marks Pradeep fell short by -/
def shortfall : ℕ := 110

theorem pradeep_exam_marks :
  (pass_percentage * maximum_marks : ℚ) = pradeep_marks + shortfall ∧
  maximum_marks * pass_percentage = (pradeep_marks + shortfall : ℚ) ∧
  ∀ m : ℕ, m > maximum_marks → 
    (pass_percentage * m : ℚ) > (pradeep_marks + shortfall : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_pradeep_exam_marks_l1687_168755


namespace NUMINAMATH_CALUDE_min_expression_l1687_168743

theorem min_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x * y) / 2 + 18 / (x * y) ≥ 6 ∧
  ((x * y) / 2 + 18 / (x * y) = 6 → y / 2 + x / 3 ≥ 2) ∧
  ((x * y) / 2 + 18 / (x * y) = 6 ∧ y / 2 + x / 3 = 2 → x = 3 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_expression_l1687_168743


namespace NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l1687_168792

def f (a x : ℝ) := x^2 + 2*a*x + 2

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∨
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem f_monotonicity_and_minimum (a : ℝ) :
  (is_monotonic (f a) (-5) 5 ↔ a ≥ 5 ∨ a ≤ -5) ∧
  (∀ x ∈ Set.Icc (-5) 5, f a x ≥
    if a ≥ 5 then 27 - 10*a
    else if a ≥ -5 then 2 - a^2
    else 27 + 10*a) :=
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l1687_168792


namespace NUMINAMATH_CALUDE_unique_solution_l1687_168730

/-- Given a real number a, returns the sum of coefficients of odd powers of x 
    in the expansion of (1+ax)^2(1-x)^5 -/
def oddPowerSum (a : ℝ) : ℝ := sorry

theorem unique_solution : 
  ∃! (a : ℝ), a > 0 ∧ oddPowerSum a = -64 :=
by
  use 3
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1687_168730


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l1687_168713

theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = bags ∧ bags = days ∧ days = 40) :
  (1 : ℕ) * days = 40 := by sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l1687_168713


namespace NUMINAMATH_CALUDE_root_sum_fraction_l1687_168786

theorem root_sum_fraction (m₁ m₂ : ℝ) : 
  (∃ a b : ℝ, m₁ * (a^2 - 3*a) + 2*a + 7 = 0 ∧ 
              m₂ * (b^2 - 3*b) + 2*b + 7 = 0 ∧ 
              a/b + b/a = 7/3) →
  m₁/m₂ + m₂/m₁ = 15481/324 := by
sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l1687_168786


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1687_168787

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (a - 2 * Complex.I) / (1 + 2 * Complex.I)) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1687_168787


namespace NUMINAMATH_CALUDE_max_sin_angle_ellipse_l1687_168739

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

def is_focus (F : ℝ × ℝ) (a b : ℝ) : Prop :=
  F.1^2 + F.2^2 = a^2 - b^2 ∧ a > b ∧ a > 0 ∧ b > 0

def angle_sin (A B C : ℝ × ℝ) : ℝ := sorry

theorem max_sin_angle_ellipse :
  ∃ (a b : ℝ) (F₁ F₂ : ℝ × ℝ),
    a = 3 ∧ b = Real.sqrt 5 ∧
    is_focus F₁ a b ∧ is_focus F₂ a b ∧
    (∀ (P : ℝ × ℝ), ellipse P.1 P.2 →
      angle_sin F₁ P F₂ ≤ 4 * Real.sqrt 5 / 9) ∧
    (∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧
      angle_sin F₁ P F₂ = 4 * Real.sqrt 5 / 9) :=
sorry

end NUMINAMATH_CALUDE_max_sin_angle_ellipse_l1687_168739


namespace NUMINAMATH_CALUDE_sand_art_project_jason_sand_needed_l1687_168799

/-- The amount of sand needed for Jason's sand art project -/
theorem sand_art_project (rectangular_length : ℕ) (rectangular_width : ℕ) 
  (square_side : ℕ) (sand_per_inch : ℕ) : ℕ :=
  let rectangular_area := rectangular_length * rectangular_width
  let square_area := square_side * square_side
  let total_area := rectangular_area + square_area
  total_area * sand_per_inch

/-- Proof that Jason needs 201 grams of sand -/
theorem jason_sand_needed : sand_art_project 6 7 5 3 = 201 := by
  sorry

end NUMINAMATH_CALUDE_sand_art_project_jason_sand_needed_l1687_168799


namespace NUMINAMATH_CALUDE_marble_basket_problem_l1687_168738

theorem marble_basket_problem :
  ∀ (w o p : ℝ),
    o + p = 10 →
    w + p = 12 →
    w + o = 5 →
    w + o + p = 13.5 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_basket_problem_l1687_168738


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1687_168790

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x | x^2 < 10}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1687_168790


namespace NUMINAMATH_CALUDE_circle_origin_inside_l1687_168784

theorem circle_origin_inside (m : ℝ) : 
  (∀ x y : ℝ, (x - m)^2 + (y + m)^2 = 8 → (0 : ℝ)^2 + (0 : ℝ)^2 < 8) → 
  -2 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_origin_inside_l1687_168784


namespace NUMINAMATH_CALUDE_group_b_more_stable_l1687_168754

-- Define the structure for a group's statistics
structure GroupStats where
  mean : ℝ
  variance : ℝ

-- Define the stability comparison function
def more_stable (a b : GroupStats) : Prop :=
  a.variance < b.variance

-- Theorem statement
theorem group_b_more_stable (group_a group_b : GroupStats) 
  (h1 : group_a.mean = group_b.mean)
  (h2 : group_a.variance = 36)
  (h3 : group_b.variance = 30) :
  more_stable group_b group_a :=
by
  sorry

end NUMINAMATH_CALUDE_group_b_more_stable_l1687_168754


namespace NUMINAMATH_CALUDE_division_problem_l1687_168759

/-- Proves that given a total amount of 544, if A gets 2/3 of what B gets, and B gets 1/4 of what C gets, then A gets 64. -/
theorem division_problem (total : ℚ) (a b c : ℚ) 
  (h_total : total = 544)
  (h_ab : a = (2/3) * b)
  (h_bc : b = (1/4) * c)
  (h_sum : a + b + c = total) : 
  a = 64 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1687_168759


namespace NUMINAMATH_CALUDE_fraction_inequality_l1687_168777

theorem fraction_inequality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -3) :
  (x^2 + 3*x + 2) / (x^2 + x - 6) ≠ (x + 2) / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1687_168777


namespace NUMINAMATH_CALUDE_randy_walks_dog_twice_daily_l1687_168736

/-- The number of times Randy walks his dog per day -/
def walks_per_day (wipes_per_pack : ℕ) (packs_for_360_days : ℕ) : ℕ :=
  (wipes_per_pack * packs_for_360_days) / 360

theorem randy_walks_dog_twice_daily :
  walks_per_day 120 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_randy_walks_dog_twice_daily_l1687_168736


namespace NUMINAMATH_CALUDE_map_distance_theorem_l1687_168788

/-- Given a map scale and an actual distance, calculate the distance on the map --/
def map_distance (scale : ℚ) (actual_distance_km : ℚ) : ℚ :=
  (actual_distance_km * 100000) / (1 / scale)

/-- Theorem: The distance between two points on a map with scale 1/250000 and actual distance 5 km is 2 cm --/
theorem map_distance_theorem :
  map_distance (1 / 250000) 5 = 2 := by
  sorry

#eval map_distance (1 / 250000) 5

end NUMINAMATH_CALUDE_map_distance_theorem_l1687_168788


namespace NUMINAMATH_CALUDE_sum_always_positive_l1687_168781

variable {f : ℝ → ℝ}
variable {a : ℕ → ℝ}

def MonotonicIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (h_incr : MonotonicIncreasing f)
  (h_odd : OddFunction f)
  (h_arith : ArithmeticSequence a)
  (h_a3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l1687_168781


namespace NUMINAMATH_CALUDE_toy_price_difference_is_250_l1687_168765

def toy_price_difference : ℝ → Prop :=
  λ price_diff : ℝ =>
    ∃ (a b : ℝ),
      a > 150 ∧ b > 150 ∧
      (∀ p : ℝ, a ≤ p ∧ p ≤ b →
        (0.2 * p ≥ 40 ∧ 0.2 * p ≥ 0.3 * (p - 150))) ∧
      (∀ p : ℝ, p < a ∨ p > b →
        (0.2 * p < 40 ∨ 0.2 * p < 0.3 * (p - 150))) ∧
      price_diff = b - a

theorem toy_price_difference_is_250 :
  toy_price_difference 250 :=
sorry

end NUMINAMATH_CALUDE_toy_price_difference_is_250_l1687_168765


namespace NUMINAMATH_CALUDE_ed_doug_marble_difference_l1687_168785

-- Define the initial number of marbles for Ed and Doug
def ed_marbles : ℕ := 45
def doug_initial_marbles : ℕ := ed_marbles - 10

-- Define the number of marbles Doug lost
def doug_lost_marbles : ℕ := 11

-- Define Doug's final number of marbles
def doug_final_marbles : ℕ := doug_initial_marbles - doug_lost_marbles

-- Theorem statement
theorem ed_doug_marble_difference :
  ed_marbles - doug_final_marbles = 21 :=
by sorry

end NUMINAMATH_CALUDE_ed_doug_marble_difference_l1687_168785


namespace NUMINAMATH_CALUDE_problem_solution_l1687_168764

theorem problem_solution (x y : ℝ) (hx : x = Real.sqrt 3 + 1) (hy : y = Real.sqrt 3 - 1) :
  (x^2 + 2*x*y + y^2 = 12) ∧ (x^2 - y^2 = 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1687_168764


namespace NUMINAMATH_CALUDE_investment_equation_l1687_168702

/-- Proves the equation for the investment problem -/
theorem investment_equation (x : ℝ) (h : x > 0) : (106960 / (x + 500)) - (50760 / x) = 20 := by
  sorry

end NUMINAMATH_CALUDE_investment_equation_l1687_168702


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l1687_168780

theorem modulo_residue_problem : (392 + 6 * 51 + 8 * 221 + 3^2 * 23) % 17 = 11 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l1687_168780


namespace NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l1687_168723

/-- The vertex of a parabola in the form y = a(x-h)^2 + k is (h,k) -/
theorem parabola_vertex (a : ℝ) (h k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * (x - h)^2 + k
  (h, k) = (h, f h) ∧ ∀ x, f x ≥ f h := by sorry

/-- The vertex of the parabola y = 1/3 * (x-7)^2 + 5 is (7,5) -/
theorem specific_parabola_vertex :
  let f : ℝ → ℝ := λ x ↦ (1/3) * (x - 7)^2 + 5
  (7, 5) = (7, f 7) ∧ ∀ x, f x ≥ f 7 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l1687_168723


namespace NUMINAMATH_CALUDE_books_in_childrens_section_l1687_168796

theorem books_in_childrens_section
  (initial_books : ℕ)
  (books_left : ℕ)
  (history_books : ℕ)
  (fiction_books : ℕ)
  (misplaced_books : ℕ)
  (h1 : initial_books = 51)
  (h2 : books_left = 16)
  (h3 : history_books = 12)
  (h4 : fiction_books = 19)
  (h5 : misplaced_books = 4) :
  initial_books + misplaced_books - history_books - fiction_books - books_left = 8 :=
by sorry

end NUMINAMATH_CALUDE_books_in_childrens_section_l1687_168796


namespace NUMINAMATH_CALUDE_a_upper_bound_l1687_168744

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - x ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2^(1-x) + a ≤ 0}

-- State the theorem
theorem a_upper_bound (a : ℝ) (h : A ⊆ B a) : a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_a_upper_bound_l1687_168744


namespace NUMINAMATH_CALUDE_min_cards_xiaohua_l1687_168769

def greeting_cards (x y z : ℕ) : Prop :=
  (Nat.lcm x (Nat.lcm y z) = 60) ∧
  (Nat.gcd x y = 4) ∧
  (Nat.gcd y z = 3) ∧
  (x ≥ 5)

theorem min_cards_xiaohua :
  ∀ x y z : ℕ, greeting_cards x y z → x ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_min_cards_xiaohua_l1687_168769


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l1687_168727

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 22 / 21 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l1687_168727


namespace NUMINAMATH_CALUDE_function_minimum_condition_l1687_168762

theorem function_minimum_condition (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, Real.exp x + a * Real.exp (-x) ≥ 1 + a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_condition_l1687_168762


namespace NUMINAMATH_CALUDE_range_of_expression_l1687_168729

theorem range_of_expression (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 ≤ β ∧ β ≤ π/2) :
  -π/6 < 2*α - β/3 ∧ 2*α - β/3 < π := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l1687_168729


namespace NUMINAMATH_CALUDE_square_of_negative_square_l1687_168732

theorem square_of_negative_square (m : ℝ) : (-m^2)^2 = m^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_square_l1687_168732


namespace NUMINAMATH_CALUDE_initial_dozens_of_doughnuts_l1687_168773

theorem initial_dozens_of_doughnuts (eaten : ℕ) (left : ℕ) (dozen : ℕ) : 
  eaten = 8 → left = 16 → dozen = 12 → (eaten + left) / dozen = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_dozens_of_doughnuts_l1687_168773


namespace NUMINAMATH_CALUDE_problem_solution_l1687_168775

def circle_times (a b : ℚ) : ℚ := (a + b) / (a - b)

def circle_plus (a b : ℚ) : ℚ := 2 * (circle_times a b)

theorem problem_solution : circle_plus (circle_plus 8 6) 2 = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1687_168775


namespace NUMINAMATH_CALUDE_cone_volume_theorem_l1687_168734

-- Define the cone properties
def base_radius : ℝ := 1
def lateral_area_ratio : ℝ := 2

-- Theorem statement
theorem cone_volume_theorem :
  let r := base_radius
  let l := lateral_area_ratio * r -- slant height
  let h := Real.sqrt (l^2 - r^2) -- height
  (1/3 : ℝ) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_theorem_l1687_168734


namespace NUMINAMATH_CALUDE_expression_value_l1687_168757

theorem expression_value : 2^2 + (-3)^2 - 7^2 - 2*2*(-3) + 3*7 = -15 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1687_168757


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1687_168791

-- Problem 1
theorem solve_equation_one (x : ℝ) : 4 * x^2 = 25 ↔ x = 5/2 ∨ x = -5/2 := by sorry

-- Problem 2
theorem solve_equation_two (x : ℝ) : (x + 1)^3 - 8 = 56 ↔ x = 3 := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1687_168791


namespace NUMINAMATH_CALUDE_tan_alpha_and_expression_l1687_168731

theorem tan_alpha_and_expression (α : Real) (h : Real.tan (α / 2) = 2) :
  Real.tan α = -4/3 ∧ (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_and_expression_l1687_168731


namespace NUMINAMATH_CALUDE_actual_speed_calculation_l1687_168712

/-- 
Given a person who travels a certain distance at an unknown speed, 
this theorem proves that if walking at 10 km/hr would allow them 
to travel 20 km more in the same time, and the actual distance 
traveled is 20 km, then their actual speed is 5 km/hr.
-/
theorem actual_speed_calculation 
  (actual_distance : ℝ) 
  (faster_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 20) 
  (h2 : faster_speed = 10) 
  (h3 : additional_distance = 20) 
  (h4 : actual_distance / actual_speed = (actual_distance + additional_distance) / faster_speed) :
  actual_speed = 5 :=
sorry

#check actual_speed_calculation

end NUMINAMATH_CALUDE_actual_speed_calculation_l1687_168712
