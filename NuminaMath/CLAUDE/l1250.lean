import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_value_given_condition_l1250_125010

theorem polynomial_value_given_condition (x : ℝ) : 
  3 * x^3 - x = 1 → 9 * x^4 + 12 * x^3 - 3 * x^2 - 7 * x + 2001 = 2001 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_given_condition_l1250_125010


namespace NUMINAMATH_CALUDE_triangle_side_difference_l1250_125060

theorem triangle_side_difference (x : ℕ) : 
  x > 0 → x + 8 > 10 → x + 10 > 8 → 8 + 10 > x → 
  (∃ (max min : ℕ), 
    (∀ y : ℕ, y > 0 → y + 8 > 10 → y + 10 > 8 → 8 + 10 > y → y ≤ max) ∧
    (∀ y : ℕ, y > 0 → y + 8 > 10 → y + 10 > 8 → 8 + 10 > y → y ≥ min) ∧
    max - min = 14) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l1250_125060


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1250_125088

theorem sufficient_not_necessary (m : ℝ) (h : m > 0) :
  (∀ a b : ℝ, a > b ∧ b > 0 → (b + m) / (a + m) > b / a) ∧
  (∃ a b : ℝ, (b + m) / (a + m) > b / a ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1250_125088


namespace NUMINAMATH_CALUDE_increase_by_fifty_percent_l1250_125056

theorem increase_by_fifty_percent (initial : ℝ) (increase : ℝ) (result : ℝ) : 
  initial = 350 → increase = 0.5 → result = initial * (1 + increase) → result = 525 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_fifty_percent_l1250_125056


namespace NUMINAMATH_CALUDE_find_x1_l1250_125042

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h_order : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h_eq1 : (1-x1)^2 + (x1-x2)^2 + (x2-x3)^2 + (x3-x4)^2 + x4^2 = 1/3)
  (h_sum : x1 + x2 + x3 + x4 = 2) : 
  x1 = 4/5 := by
sorry

end NUMINAMATH_CALUDE_find_x1_l1250_125042


namespace NUMINAMATH_CALUDE_mutated_frogs_percentage_l1250_125019

def total_frogs : ℕ := 27
def mutated_frogs : ℕ := 9

def percentage_mutated : ℚ := (mutated_frogs : ℚ) / (total_frogs : ℚ) * 100

def rounded_percentage : ℕ := 
  (percentage_mutated + 0.5).floor.toNat

theorem mutated_frogs_percentage :
  rounded_percentage = 33 := by sorry

end NUMINAMATH_CALUDE_mutated_frogs_percentage_l1250_125019


namespace NUMINAMATH_CALUDE_officer_selection_theorem_l1250_125005

/-- Represents a club with members of two genders -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ

/-- Calculates the number of ways to choose officers from a single gender -/
def waysToChooseOfficers (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- Calculates the total number of ways to choose officers in the club -/
def totalWaysToChooseOfficers (club : Club) : ℕ :=
  waysToChooseOfficers club.boys + waysToChooseOfficers club.girls

/-- The main theorem stating the number of ways to choose officers -/
theorem officer_selection_theorem (club : Club) 
    (h1 : club.total_members = 30)
    (h2 : club.boys = 18)
    (h3 : club.girls = 12) :
    totalWaysToChooseOfficers club = 6216 := by
  sorry


end NUMINAMATH_CALUDE_officer_selection_theorem_l1250_125005


namespace NUMINAMATH_CALUDE_complex_power_difference_l1250_125076

theorem complex_power_difference (x : ℂ) (h : x - 1/x = 2*I) : x^2048 - 1/x^2048 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1250_125076


namespace NUMINAMATH_CALUDE_security_deposit_is_1110_l1250_125024

/-- Calculates the security deposit for a cabin rental --/
def calculate_security_deposit (weeks : ℕ) (daily_rate : ℚ) (pet_fee : ℚ) (service_fee_rate : ℚ) (deposit_rate : ℚ) : ℚ :=
  let days := weeks * 7
  let rental_fee := daily_rate * days
  let total_rental := rental_fee + pet_fee
  let service_fee := service_fee_rate * total_rental
  let total_cost := total_rental + service_fee
  deposit_rate * total_cost

/-- Theorem: The security deposit for the given conditions is $1,110.00 --/
theorem security_deposit_is_1110 :
  calculate_security_deposit 2 125 100 (1/5) (1/2) = 1110 := by
  sorry

end NUMINAMATH_CALUDE_security_deposit_is_1110_l1250_125024


namespace NUMINAMATH_CALUDE_distribute_negative_three_l1250_125090

theorem distribute_negative_three (x y : ℝ) : -3 * (x - x * y) = -3 * x + 3 * x * y := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_three_l1250_125090


namespace NUMINAMATH_CALUDE_triangle_condition_implies_right_angle_l1250_125083

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition from the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a - 3)^2 + Real.sqrt (t.b - 4) + |t.c - 5| = 0

-- Define what it means for a triangle to be right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- Theorem statement
theorem triangle_condition_implies_right_angle (t : Triangle) :
  satisfiesCondition t → isRightTriangle t :=
by sorry

end NUMINAMATH_CALUDE_triangle_condition_implies_right_angle_l1250_125083


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l1250_125074

/-- Represents a ball with a color and label -/
structure Ball where
  color : String
  label : Char

/-- Represents the bag of balls -/
def bag : List Ball := [
  { color := "yellow", label := 'a' },
  { color := "yellow", label := 'b' },
  { color := "red", label := 'c' },
  { color := "red", label := 'd' }
]

/-- Calculates the probability of drawing a yellow ball on the first draw -/
def probYellowFirst (bag : List Ball) : ℚ :=
  (bag.filter (fun b => b.color = "yellow")).length / bag.length

/-- Calculates the probability of drawing a yellow ball on the second draw -/
def probYellowSecond (bag : List Ball) : ℚ :=
  let totalOutcomes := bag.length * (bag.length - 1)
  let favorableOutcomes := 2 * (bag.length - 2)
  favorableOutcomes / totalOutcomes

theorem yellow_ball_probability (bag : List Ball) :
  probYellowFirst bag = 1/2 ∧ probYellowSecond bag = 1/2 :=
sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l1250_125074


namespace NUMINAMATH_CALUDE_reduce_to_single_letter_l1250_125028

/-- Represents a circular arrangement of letters A and B -/
def CircularArrangement := List Bool

/-- Represents the operations that can be performed on the arrangement -/
inductive Operation
  | replaceABA
  | replaceBAB

/-- Applies an operation to a circular arrangement -/
def applyOperation (arr : CircularArrangement) (op : Operation) : CircularArrangement :=
  sorry

/-- Checks if the arrangement consists of only one type of letter -/
def isSingleLetter (arr : CircularArrangement) : Bool :=
  sorry

/-- Theorem stating that any initial arrangement of 41 letters can be reduced to a single letter -/
theorem reduce_to_single_letter (initial : CircularArrangement) :
  initial.length = 41 → ∃ (final : CircularArrangement), isSingleLetter final ∧ 
  ∃ (ops : List Operation), final = ops.foldl applyOperation initial :=
  sorry

end NUMINAMATH_CALUDE_reduce_to_single_letter_l1250_125028


namespace NUMINAMATH_CALUDE_smallest_possible_b_l1250_125061

def is_valid_polynomial (Q : ℤ → ℤ) (b : ℕ) : Prop :=
  b > 0 ∧
  Q 0 = b ∧ Q 4 = b ∧ Q 6 = b ∧ Q 10 = b ∧
  Q 1 = -b ∧ Q 5 = -b ∧ Q 7 = -b ∧ Q 11 = -b

theorem smallest_possible_b :
  ∀ Q : ℤ → ℤ, ∀ b : ℕ,
  is_valid_polynomial Q b →
  (∀ b' : ℕ, is_valid_polynomial Q b' → b ≤ b') →
  b = 1350 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l1250_125061


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l1250_125012

theorem rectangular_plot_length 
  (width : ℝ) 
  (num_poles : ℕ) 
  (pole_distance : ℝ) 
  (h1 : width = 40) 
  (h2 : num_poles = 52) 
  (h3 : pole_distance = 5) : 
  let perimeter := (num_poles - 1 : ℝ) * pole_distance
  let length := perimeter / 2 - width
  length = 87.5 := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l1250_125012


namespace NUMINAMATH_CALUDE_game_cost_before_tax_l1250_125068

theorem game_cost_before_tax 
  (weekly_savings : ℝ) 
  (weeks : ℕ) 
  (tax_rate : ℝ) 
  (total_saved : ℝ) 
  (h1 : weekly_savings = 5)
  (h2 : weeks = 11)
  (h3 : tax_rate = 0.1)
  (h4 : total_saved = weekly_savings * weeks)
  : ∃ (pre_tax_cost : ℝ), pre_tax_cost = 50 ∧ total_saved = pre_tax_cost * (1 + tax_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_game_cost_before_tax_l1250_125068


namespace NUMINAMATH_CALUDE_yogurt_combinations_l1250_125038

theorem yogurt_combinations (flavors : Nat) (toppings : Nat) : 
  flavors = 5 → toppings = 8 → 
  flavors * (toppings.choose 3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l1250_125038


namespace NUMINAMATH_CALUDE_miller_rabin_correct_for_primes_l1250_125055

/-- Miller-Rabin primality test function -/
def miller_rabin (n : ℕ) : Bool := sorry

/-- Definition of primality -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem miller_rabin_correct_for_primes (n : ℕ) (h : is_prime n) : 
  miller_rabin n = true := by sorry

end NUMINAMATH_CALUDE_miller_rabin_correct_for_primes_l1250_125055


namespace NUMINAMATH_CALUDE_below_warning_level_notation_l1250_125000

/-- Represents the water level relative to a warning level -/
def water_level_notation (warning_level : ℝ) (actual_level : ℝ) : ℝ :=
  actual_level - warning_level

theorem below_warning_level_notation 
  (warning_level : ℝ) (distance_below : ℝ) (distance_below_positive : distance_below > 0) :
  water_level_notation warning_level (warning_level - distance_below) = -distance_below :=
by sorry

end NUMINAMATH_CALUDE_below_warning_level_notation_l1250_125000


namespace NUMINAMATH_CALUDE_percussion_probability_l1250_125049

def total_sounds : ℕ := 6
def percussion_sounds : ℕ := 3

theorem percussion_probability :
  (percussion_sounds.choose 2 : ℚ) / (total_sounds.choose 2) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_percussion_probability_l1250_125049


namespace NUMINAMATH_CALUDE_large_pizza_slices_l1250_125021

def large_pizza_cost : ℝ := 10
def first_topping_cost : ℝ := 2
def next_two_toppings_cost : ℝ := 1
def remaining_toppings_cost : ℝ := 0.5
def num_toppings : ℕ := 7
def cost_per_slice : ℝ := 2

def total_pizza_cost : ℝ :=
  large_pizza_cost + first_topping_cost + 2 * next_two_toppings_cost + 
  (num_toppings - 3 : ℝ) * remaining_toppings_cost

theorem large_pizza_slices :
  (total_pizza_cost / cost_per_slice : ℝ) = 8 :=
sorry

end NUMINAMATH_CALUDE_large_pizza_slices_l1250_125021


namespace NUMINAMATH_CALUDE_two_numbers_subtracted_from_32_l1250_125041

theorem two_numbers_subtracted_from_32 : ∃ (A B : ℤ), 
  A ≠ B ∧
  ((32 - A = 23 ∧ 32 - B = 13) ∨ (32 - A = 13 ∧ 32 - B = 23)) ∧
  ¬ (∃ (k : ℤ), |A - B| = 11 * k) ∧
  A = 9 ∧ B = 19 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_subtracted_from_32_l1250_125041


namespace NUMINAMATH_CALUDE_contest_end_time_l1250_125003

def contest_start : Nat := 15 * 60  -- 3:00 p.m. in minutes since midnight
def contest_duration : Nat := 850   -- total duration in minutes
def break_duration : Nat := 30      -- break duration in minutes

def minutes_in_day : Nat := 24 * 60 -- number of minutes in a day

def contest_end : Nat :=
  (contest_start + contest_duration - break_duration) % minutes_in_day

theorem contest_end_time :
  contest_end = 4 * 60 + 40 := by sorry

end NUMINAMATH_CALUDE_contest_end_time_l1250_125003


namespace NUMINAMATH_CALUDE_intersection_M_N_l1250_125035

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 2*x ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1250_125035


namespace NUMINAMATH_CALUDE_solution_set_characterization_l1250_125032

def is_solution_set (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, x ∈ S ↔ 2^(1 + f x) + 2^(1 - f x) + 2 * f (x^2) ≤ 7

theorem solution_set_characterization
  (f : ℝ → ℝ)
  (h1 : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
  (h2 : ∀ x, x > 0 → f x > 0)
  (h3 : f 1 = 1) :
  is_solution_set f (Set.Icc (-1) 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l1250_125032


namespace NUMINAMATH_CALUDE_nancy_shoe_count_nancy_has_168_shoes_l1250_125004

/-- Calculates the total number of individual shoes Nancy has given her shoe collection. -/
theorem nancy_shoe_count (boots : ℕ) (slippers : ℕ) (heels : ℕ) : ℕ :=
  let total_pairs := boots + slippers + heels
  2 * total_pairs

/-- Proves that Nancy has 168 individual shoes given the conditions of her shoe collection. -/
theorem nancy_has_168_shoes : nancy_shoe_count 6 15 63 = 168 := by
  sorry

#check nancy_has_168_shoes

end NUMINAMATH_CALUDE_nancy_shoe_count_nancy_has_168_shoes_l1250_125004


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l1250_125097

/-- Given a quadratic inequality ax^2 - bx + 2 < 0 with solution set {x | 1 < x < 2},
    prove that the sum of coefficients a + b equals 4. -/
theorem quadratic_inequality_coefficient_sum (a b : ℝ) : 
  (∀ x, (1 < x ∧ x < 2) ↔ (a * x^2 - b * x + 2 < 0)) → 
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l1250_125097


namespace NUMINAMATH_CALUDE_melissa_commission_l1250_125023

/-- Calculates the commission earned by Melissa based on vehicle sales --/
def calculate_commission (coupe_price suv_price luxury_sedan_price motorcycle_price truck_price : ℕ)
  (coupe_sold suv_sold luxury_sedan_sold motorcycle_sold truck_sold : ℕ) : ℕ :=
  let total_sales := coupe_price * coupe_sold + (2 * coupe_price) * suv_sold +
                     luxury_sedan_price * luxury_sedan_sold + motorcycle_price * motorcycle_sold +
                     truck_price * truck_sold
  let total_vehicles := coupe_sold + suv_sold + luxury_sedan_sold + motorcycle_sold + truck_sold
  let commission_rate := if total_vehicles ≤ 2 then 2
                         else if total_vehicles ≤ 4 then 25
                         else 3
  (total_sales * commission_rate) / 100

theorem melissa_commission :
  calculate_commission 30000 60000 80000 15000 40000 3 2 1 4 2 = 12900 :=
by sorry

end NUMINAMATH_CALUDE_melissa_commission_l1250_125023


namespace NUMINAMATH_CALUDE_toys_sold_second_week_is_26_l1250_125096

/-- The number of toys sold in the second week at an online toy store. -/
def toys_sold_second_week (initial_stock : ℕ) (sold_first_week : ℕ) (toys_left : ℕ) : ℕ :=
  initial_stock - sold_first_week - toys_left

/-- Theorem stating that 26 toys were sold in the second week. -/
theorem toys_sold_second_week_is_26 :
  toys_sold_second_week 83 38 19 = 26 := by
  sorry

#eval toys_sold_second_week 83 38 19

end NUMINAMATH_CALUDE_toys_sold_second_week_is_26_l1250_125096


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1250_125006

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 20 - 5 / 200 + 7 / 2000 = 0.1285 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1250_125006


namespace NUMINAMATH_CALUDE_sum_reciprocals_l1250_125085

/-- Given two positive integers m and n with sum 60, HCF 6, and LCM 210, prove that 1/m + 1/n = 1/21 -/
theorem sum_reciprocals (m n : ℕ+) 
  (h_sum : m + n = 60)
  (h_hcf : Nat.gcd m.val n.val = 6)
  (h_lcm : Nat.lcm m.val n.val = 210) : 
  1 / (m : ℚ) + 1 / (n : ℚ) = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l1250_125085


namespace NUMINAMATH_CALUDE_part_one_part_two_l1250_125011

-- Define the inequalities p and q
def p (x a : ℝ) : Prop := x^2 - 6*a*x + 8*a^2 < 0
def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Part (1)
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x ≤ 3) :=
sorry

-- Part (2)
theorem part_two :
  ∀ a : ℝ, (∀ x : ℝ, p x a → q x) ∧ (∃ x : ℝ, q x ∧ ¬(p x a)) ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1250_125011


namespace NUMINAMATH_CALUDE_train_journey_solution_l1250_125002

/-- Represents the number of passengers from Zhejiang and Shanghai at a given point in the journey -/
structure PassengerCount where
  zhejiang : Nat
  shanghai : Nat

/-- Represents the train journey with passenger counts at each stage -/
structure TrainJourney where
  initial : PassengerCount
  afterB : PassengerCount
  afterC : PassengerCount
  afterD : PassengerCount
  afterE : PassengerCount
  final : PassengerCount

def total_passengers (pc : PassengerCount) : Nat :=
  pc.zhejiang + pc.shanghai

/-- The conditions of the train journey -/
def journey_conditions (j : TrainJourney) : Prop :=
  total_passengers j.initial = 19 ∧
  total_passengers j.afterB = 12 ∧
  total_passengers j.afterD = 7 ∧
  total_passengers j.final = 0 ∧
  j.initial.zhejiang = (total_passengers j.initial - total_passengers j.afterB) ∧
  j.afterB.zhejiang = (total_passengers j.afterB - total_passengers j.afterC) ∧
  j.afterC.zhejiang = (total_passengers j.afterC - total_passengers j.afterD) ∧
  j.afterD.zhejiang = (total_passengers j.afterD - total_passengers j.afterE) ∧
  j.afterE.zhejiang = (total_passengers j.afterE - total_passengers j.final)

/-- The theorem stating that given the conditions, the journey matches the solution -/
theorem train_journey_solution (j : TrainJourney) :
  journey_conditions j →
  j.initial = ⟨7, 12⟩ ∧
  j.afterB = ⟨3, 9⟩ ∧
  j.afterC = ⟨2, 7⟩ ∧
  j.afterD = ⟨2, 5⟩ :=
by sorry

end NUMINAMATH_CALUDE_train_journey_solution_l1250_125002


namespace NUMINAMATH_CALUDE_graveyard_bone_ratio_l1250_125050

theorem graveyard_bone_ratio :
  let total_skeletons : ℕ := 20
  let adult_women_skeletons : ℕ := total_skeletons / 2
  let remaining_skeletons : ℕ := total_skeletons - adult_women_skeletons
  let adult_men_skeletons : ℕ := remaining_skeletons / 2
  let children_skeletons : ℕ := remaining_skeletons / 2
  let adult_woman_bones : ℕ := 20
  let adult_man_bones : ℕ := adult_woman_bones + 5
  let total_bones : ℕ := 375
  let child_bones : ℕ := (total_bones - (adult_women_skeletons * adult_woman_bones + adult_men_skeletons * adult_man_bones)) / children_skeletons
  (child_bones : ℚ) / (adult_woman_bones : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_graveyard_bone_ratio_l1250_125050


namespace NUMINAMATH_CALUDE_crayons_remaining_l1250_125046

theorem crayons_remaining (initial : ℕ) (taken : ℕ) (remaining : ℕ) : 
  initial = 7 → taken = 3 → remaining = initial - taken → remaining = 4 := by
sorry

end NUMINAMATH_CALUDE_crayons_remaining_l1250_125046


namespace NUMINAMATH_CALUDE_fixed_fee_calculation_l1250_125037

/-- Represents a monthly bill for an online service provider -/
structure MonthlyBill where
  fixed_fee : ℝ
  hourly_rate : ℝ
  hours_used : ℝ

/-- Calculates the total bill amount -/
def MonthlyBill.total (bill : MonthlyBill) : ℝ :=
  bill.fixed_fee + bill.hourly_rate * bill.hours_used

theorem fixed_fee_calculation (feb_bill mar_bill : MonthlyBill) 
  (h1 : feb_bill.total = 20.72)
  (h2 : mar_bill.total = 35.28)
  (h3 : feb_bill.fixed_fee = mar_bill.fixed_fee)
  (h4 : feb_bill.hourly_rate = mar_bill.hourly_rate)
  (h5 : mar_bill.hours_used = 3 * feb_bill.hours_used) :
  feb_bill.fixed_fee = 13.44 := by
  sorry

#check fixed_fee_calculation

end NUMINAMATH_CALUDE_fixed_fee_calculation_l1250_125037


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1250_125063

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n, b (n + 1) = b n + d) →  -- arithmetic sequence
  (b 5 * b 6 = 21) →  -- given condition
  (b 4 * b 7 = -779 ∨ b 4 * b 7 = -11) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1250_125063


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1250_125036

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (sum_eq : x + y = 3 * x * y) (diff_eq : x - y = 2) : 
  1 / x + 1 / y = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1250_125036


namespace NUMINAMATH_CALUDE_f_has_local_minimum_in_interval_l1250_125054

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log x

theorem f_has_local_minimum_in_interval :
  ∃ x₀ : ℝ, 1/2 < x₀ ∧ x₀ < 1 ∧ IsLocalMin f x₀ := by sorry

end NUMINAMATH_CALUDE_f_has_local_minimum_in_interval_l1250_125054


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l1250_125072

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_identification :
  ¬ is_pythagorean_triple 7 8 9 ∧
  ¬ is_pythagorean_triple 5 6 7 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 21 25 28 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l1250_125072


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1250_125027

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {0, 2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1250_125027


namespace NUMINAMATH_CALUDE_line_slope_problem_l1250_125013

/-- Given a line passing through points (1, -1) and (3, m) with slope 2, prove that m = 3 -/
theorem line_slope_problem (m : ℝ) : 
  (m - (-1)) / (3 - 1) = 2 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l1250_125013


namespace NUMINAMATH_CALUDE_inverse_function_zero_solution_l1250_125030

/-- Given a function f(x) = 2 / (ax + b) where a and b are nonzero constants,
    prove that the solution to f⁻¹(x) = 0 is x = 2/b -/
theorem inverse_function_zero_solution
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x => 2 / (a * x + b)
  (f⁻¹) 0 = 2 / b :=
sorry

end NUMINAMATH_CALUDE_inverse_function_zero_solution_l1250_125030


namespace NUMINAMATH_CALUDE_min_value_at_one_third_l1250_125034

def y (x : ℝ) : ℝ := |x - 1| + |2*x - 1| + |3*x - 1| + |4*x - 1| + |5*x - 1|

theorem min_value_at_one_third :
  ∀ x : ℝ, y (1/3 : ℝ) ≤ y x := by
  sorry

end NUMINAMATH_CALUDE_min_value_at_one_third_l1250_125034


namespace NUMINAMATH_CALUDE_triangle_side_difference_l1250_125022

/-- Given a triangle ABC with side lengths satisfying specific conditions, prove that b - a = 0 --/
theorem triangle_side_difference (a b : ℤ) : 
  a > 1 → 
  b > 1 → 
  ∃ (AB BC CA : ℝ), 
    AB = b^2 - 1 ∧ 
    BC = a^2 ∧ 
    CA = 2*a ∧ 
    AB + BC > CA ∧ 
    BC + CA > AB ∧ 
    CA + AB > BC → 
    b - a = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l1250_125022


namespace NUMINAMATH_CALUDE_inequality_solution_l1250_125001

/-- Given an inequality (ax-1)/(x+1) < 0 with solution set {x | x < -1 or x > -1/2}, prove that a = -2 -/
theorem inequality_solution (a : ℝ) : 
  (∀ x : ℝ, (a * x - 1) / (x + 1) < 0 ↔ (x < -1 ∨ x > -1/2)) → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1250_125001


namespace NUMINAMATH_CALUDE_smallest_solution_abs_quadratic_l1250_125069

theorem smallest_solution_abs_quadratic (x : ℝ) :
  (|2 * x^2 + 3 * x - 1| = 33) →
  x ≥ ((-3 - Real.sqrt 281) / 4) ∧
  (|2 * (((-3 - Real.sqrt 281) / 4)^2) + 3 * ((-3 - Real.sqrt 281) / 4) - 1| = 33) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_quadratic_l1250_125069


namespace NUMINAMATH_CALUDE_square_difference_representation_l1250_125018

theorem square_difference_representation (n : ℕ) :
  (∃ (a b : ℤ), n + a^2 = b^2) ↔ n % 4 ≠ 2 := by sorry

end NUMINAMATH_CALUDE_square_difference_representation_l1250_125018


namespace NUMINAMATH_CALUDE_sum_of_variables_l1250_125033

theorem sum_of_variables (x y z : ℚ) 
  (eq1 : y + z = 20 - 5*x)
  (eq2 : x + z = -18 - 5*y)
  (eq3 : x + y = 10 - 5*z) :
  3*x + 3*y + 3*z = 36/7 := by sorry

end NUMINAMATH_CALUDE_sum_of_variables_l1250_125033


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_l1250_125075

/-- An arithmetic-geometric sequence -/
def arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r q : ℝ, ∀ n : ℕ, a (n + 1) = r * a n + q

/-- The statement to be proven -/
theorem arithmetic_geometric_sum (a : ℕ → ℝ) :
  arithmetic_geometric_sequence a →
  a 4 + a 6 = 5 →
  a 4 * a 6 = 6 →
  a 3 * a 5 + a 5 * a 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_l1250_125075


namespace NUMINAMATH_CALUDE_triangle_area_range_l1250_125064

-- Define the triangle ABC
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

-- Define the variable points P and Q
structure VariablePoints :=
  (P : ℝ) -- distance AP
  (Q : ℝ) -- distance AQ

-- Define the perpendiculars x and y
structure Perpendiculars :=
  (x : ℝ)
  (y : ℝ)

-- Define the main theorem
theorem triangle_area_range (ABC : Triangle) (PQ : VariablePoints) (perp : Perpendiculars) :
  ABC.AB = 4 ∧ ABC.BC = 5 ∧ ABC.CA = 3 →
  0 < PQ.P ∧ PQ.P ≤ ABC.AB →
  0 < PQ.Q ∧ PQ.Q ≤ ABC.CA →
  perp.x = PQ.Q / 2 →
  perp.y = PQ.P / 2 →
  PQ.P * PQ.Q = 6 →
  6 ≤ 2 * perp.y + 3 * perp.x ∧ 2 * perp.y + 3 * perp.x ≤ 6.5 :=
by sorry


end NUMINAMATH_CALUDE_triangle_area_range_l1250_125064


namespace NUMINAMATH_CALUDE_wandas_walk_l1250_125052

/-- Proves that if Wanda walks 2 miles per day and 40 miles in 4 weeks, then she walks 5 days per week. -/
theorem wandas_walk (miles_per_day : ℝ) (total_miles : ℝ) (weeks : ℕ) (days_per_week : ℝ) : 
  miles_per_day = 2 → total_miles = 40 → weeks = 4 → 
  miles_per_day * days_per_week * weeks = total_miles → 
  days_per_week = 5 := by
sorry

end NUMINAMATH_CALUDE_wandas_walk_l1250_125052


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1250_125026

theorem complex_equation_solution (x y : ℝ) : 
  (Complex.I * (x + y) = x - 1) → (x = 1 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1250_125026


namespace NUMINAMATH_CALUDE_triangle_BC_equation_l1250_125045

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line in general form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.medianAB (t : Triangle) : Line :=
  { a := 5, b := -3, c := -3 }

def Triangle.medianAC (t : Triangle) : Line :=
  { a := 7, b := -3, c := -5 }

def Triangle.sideBC (t : Triangle) : Line :=
  { a := 2, b := -1, c := -2 }

theorem triangle_BC_equation (t : Triangle) 
  (h1 : t.A = (1, 2))
  (h2 : t.medianAB = { a := 5, b := -3, c := -3 })
  (h3 : t.medianAC = { a := 7, b := -3, c := -5 }) :
  t.sideBC = { a := 2, b := -1, c := -2 } := by
  sorry


end NUMINAMATH_CALUDE_triangle_BC_equation_l1250_125045


namespace NUMINAMATH_CALUDE_stream_speed_l1250_125043

/-- Proves that the speed of a stream is 2.5 km/h, given a man's swimming speed in still water
    and the time ratio of upstream to downstream swimming. -/
theorem stream_speed (swimming_speed : ℝ) (time_ratio : ℝ) :
  swimming_speed = 7.5 ∧ time_ratio = 2 →
  ∃ stream_speed : ℝ,
    stream_speed = 2.5 ∧
    (swimming_speed - stream_speed) / (swimming_speed + stream_speed) = 1 / time_ratio :=
by
  sorry

/-- Calculates the speed of the stream based on the given conditions. -/
def calculate_stream_speed (swimming_speed : ℝ) (time_ratio : ℝ) : ℝ :=
  2.5

#check stream_speed
#check calculate_stream_speed

end NUMINAMATH_CALUDE_stream_speed_l1250_125043


namespace NUMINAMATH_CALUDE_marble_difference_l1250_125094

theorem marble_difference (drew_original : ℕ) (marcus_original : ℕ) : 
  (drew_original / 4 = 35) →  -- Drew gave 1/4 of his marbles, which is 35
  (drew_original * 3 / 4 = 35) →  -- Drew has 35 marbles after giving 1/4 away
  (marcus_original + 35 = 35) →  -- Marcus has 35 marbles after receiving Drew's 1/4
  (drew_original - marcus_original = 140) := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l1250_125094


namespace NUMINAMATH_CALUDE_fraction_product_l1250_125044

theorem fraction_product : (2 : ℚ) / 9 * 5 / 8 = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1250_125044


namespace NUMINAMATH_CALUDE_sam_gave_29_cards_l1250_125025

/-- The number of new Pokemon cards Sam gave to Mary -/
def new_cards (initial : ℕ) (torn : ℕ) (final : ℕ) : ℕ :=
  final - (initial - torn)

/-- Proof that Sam gave Mary 29 new Pokemon cards -/
theorem sam_gave_29_cards : new_cards 33 6 56 = 29 := by
  sorry

end NUMINAMATH_CALUDE_sam_gave_29_cards_l1250_125025


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l1250_125067

theorem quadratic_completion_of_square :
  ∀ x : ℝ, (x^2 - 8*x + 10 = 0) ↔ ((x - 4)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l1250_125067


namespace NUMINAMATH_CALUDE_geometric_series_proof_l1250_125059

def geometric_series (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_proof 
  (a : ℚ) 
  (h_a : a = 4/7) 
  (r : ℚ) 
  (h_r : r = 4/7) :
  r = 4/7 ∧ geometric_series a r 3 = 372/343 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_proof_l1250_125059


namespace NUMINAMATH_CALUDE_mary_hourly_wage_l1250_125057

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ
  tue_thu_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours

/-- Calculates the hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Mary's work schedule --/
def mary_schedule : WorkSchedule :=
  { mon_wed_fri_hours := 9
  , tue_thu_hours := 5
  , weekly_earnings := 407 }

/-- Theorem stating Mary's hourly wage is $11 --/
theorem mary_hourly_wage :
  hourly_wage mary_schedule = 11 := by sorry

end NUMINAMATH_CALUDE_mary_hourly_wage_l1250_125057


namespace NUMINAMATH_CALUDE_sequence_periodicity_l1250_125089

/-- A cubic polynomial with rational coefficients -/
def CubicPolynomial (α : Type) [Field α] := α → α

/-- A sequence of rational numbers -/
def RationalSequence := ℕ → ℚ

/-- The statement that a sequence satisfies q_n = p(q_{n+1}) for all positive n -/
def SatisfiesRelation (p : CubicPolynomial ℚ) (q : RationalSequence) :=
  ∀ n : ℕ, q n = p (q (n + 1))

/-- The theorem stating the existence of a period for the sequence -/
theorem sequence_periodicity
  (p : CubicPolynomial ℚ)
  (q : RationalSequence)
  (h : SatisfiesRelation p q) :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, q (n + k) = q n :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l1250_125089


namespace NUMINAMATH_CALUDE_abs_4x_minus_6_not_positive_l1250_125008

theorem abs_4x_minus_6_not_positive (x : ℚ) : 
  ¬(0 < |4 * x - 6|) ↔ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_4x_minus_6_not_positive_l1250_125008


namespace NUMINAMATH_CALUDE_truck_profit_analysis_l1250_125087

def initial_cost : ℕ := 490000
def first_year_expense : ℕ := 60000
def annual_expense_increase : ℕ := 20000
def annual_income : ℕ := 250000

def profit_function (n : ℕ) : ℤ := -n^2 + 20*n - 49

def option1_sell_price : ℕ := 40000
def option2_sell_price : ℕ := 130000

theorem truck_profit_analysis :
  -- 1. Profit function
  (∀ n : ℕ, profit_function n = annual_income * n - (first_year_expense * n + (n * (n - 1) / 2) * annual_expense_increase) - initial_cost) ∧
  -- 2. Profit exceeds 150,000 in 5th year
  (profit_function 5 > 150 ∧ ∀ k < 5, profit_function k ≤ 150) ∧
  -- 3. Maximum profit at n = 10
  (∀ n : ℕ, profit_function n ≤ profit_function 10) ∧
  -- 4. Maximum average annual profit at n = 7
  (∀ n : ℕ, n ≠ 0 → profit_function n / n ≤ profit_function 7 / 7) ∧
  -- 5. Both options yield 550,000 total profit
  (profit_function 10 + option1_sell_price = 550000 ∧
   profit_function 7 + option2_sell_price = 550000) ∧
  -- 6. Option 2 is more time-efficient
  (7 < 10) :=
by sorry

end NUMINAMATH_CALUDE_truck_profit_analysis_l1250_125087


namespace NUMINAMATH_CALUDE_philips_banana_collection_l1250_125014

/-- The number of groups of bananas in Philip's collection -/
def num_groups : ℕ := 196

/-- The number of bananas in each group -/
def bananas_per_group : ℕ := 2

/-- The total number of bananas in Philip's collection -/
def total_bananas : ℕ := num_groups * bananas_per_group

theorem philips_banana_collection : total_bananas = 392 := by
  sorry

end NUMINAMATH_CALUDE_philips_banana_collection_l1250_125014


namespace NUMINAMATH_CALUDE_horner_method_v₃_l1250_125073

def horner_polynomial (x : ℝ) : ℝ := 2*x^6 + 5*x^4 + x^3 + 7*x^2 + 3*x + 1

def horner_v₀ : ℝ := 2
def horner_v₁ (x : ℝ) : ℝ := horner_v₀ * x + 0
def horner_v₂ (x : ℝ) : ℝ := horner_v₁ x * x + 5
def horner_v₃ (x : ℝ) : ℝ := horner_v₂ x * x + 1

theorem horner_method_v₃ : horner_v₃ 3 = 70 := by sorry

end NUMINAMATH_CALUDE_horner_method_v₃_l1250_125073


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1250_125029

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 →  -- Positive lengths
  a^2 + b^2 = c^2 →                        -- Right triangle (Pythagorean theorem)
  a / b = 3 / 4 →                          -- Ratio of legs
  r * c = a^2 →                            -- Altitude theorem for r
  s * c = b^2 →                            -- Altitude theorem for s
  r + s = c →                              -- Segments sum to hypotenuse
  r / s = 9 / 16 :=                        -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1250_125029


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l1250_125099

/-- An ellipse with semi-major axis 3 and semi-minor axis 2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / 9) = 1}

/-- An equilateral triangle inscribed in the ellipse -/
structure InscribedTriangle where
  vertices : Fin 3 → ℝ × ℝ
  on_ellipse : ∀ i, vertices i ∈ Ellipse
  is_equilateral : ∀ i j, i ≠ j → dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)
  centroid_origin : (vertices 0 + vertices 1 + vertices 2) / 3 = (0, 0)

/-- The square of the area of the inscribed equilateral triangle -/
def square_area (t : InscribedTriangle) : ℝ := sorry

/-- The main theorem: The square of the area of the inscribed equilateral triangle is 507/16 -/
theorem inscribed_triangle_area (t : InscribedTriangle) : square_area t = 507/16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l1250_125099


namespace NUMINAMATH_CALUDE_travis_apple_sale_price_l1250_125048

/-- Calculates the price per box of apples given the total number of apples,
    apples per box, and desired total revenue. -/
def price_per_box (total_apples : ℕ) (apples_per_box : ℕ) (total_revenue : ℕ) : ℚ :=
  (total_revenue : ℚ) / ((total_apples / apples_per_box) : ℚ)

/-- Proves that given Travis's conditions, he must sell each box for $35. -/
theorem travis_apple_sale_price :
  price_per_box 10000 50 7000 = 35 := by
  sorry

end NUMINAMATH_CALUDE_travis_apple_sale_price_l1250_125048


namespace NUMINAMATH_CALUDE_corrected_mean_l1250_125080

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 40 ∧ original_mean = 36 ∧ incorrect_value = 20 ∧ correct_value = 34 →
  (n * original_mean + (correct_value - incorrect_value)) / n = 36.35 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l1250_125080


namespace NUMINAMATH_CALUDE_object_max_height_l1250_125084

/-- The height of the object as a function of time -/
def h (t : ℝ) : ℝ := -15 * (t - 3)^2 + 150

/-- The time at which the object reaches its maximum height -/
def t_max : ℝ := 3

theorem object_max_height :
  (∀ t : ℝ, h t ≤ h t_max) ∧
  h (t_max + 2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_object_max_height_l1250_125084


namespace NUMINAMATH_CALUDE_breakfast_customers_count_l1250_125017

/-- The number of customers during breakfast on Friday -/
def breakfast_customers : ℕ := 73

/-- The number of customers during lunch on Friday -/
def lunch_customers : ℕ := 127

/-- The number of customers during dinner on Friday -/
def dinner_customers : ℕ := 87

/-- The predicted number of customers for Saturday -/
def saturday_prediction : ℕ := 574

/-- Theorem stating that the number of customers during breakfast on Friday is 73 -/
theorem breakfast_customers_count : 
  breakfast_customers = 
    saturday_prediction / 2 - (lunch_customers + dinner_customers) :=
by
  sorry

#check breakfast_customers_count

end NUMINAMATH_CALUDE_breakfast_customers_count_l1250_125017


namespace NUMINAMATH_CALUDE_distance_between_points_l1250_125070

theorem distance_between_points (A B : ℝ) : A = 3 ∧ B = -7 → |A - B| = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1250_125070


namespace NUMINAMATH_CALUDE_result_2011th_operation_l1250_125081

/-- Represents the sequence of operations starting with 25 -/
def operationSequence : ℕ → ℕ
| 0 => 25
| 1 => 133
| 2 => 55
| 3 => 250
| (n + 4) => operationSequence n

/-- The result of the nth operation in the sequence -/
def nthOperationResult (n : ℕ) : ℕ := operationSequence (n % 4)

theorem result_2011th_operation :
  nthOperationResult 2011 = 133 := by sorry

end NUMINAMATH_CALUDE_result_2011th_operation_l1250_125081


namespace NUMINAMATH_CALUDE_complex_magnitude_l1250_125092

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1250_125092


namespace NUMINAMATH_CALUDE_sqrt_square_eq_identity_power_zero_eq_one_l1250_125007

-- Option C
theorem sqrt_square_eq_identity (x : ℝ) (h : x ≥ -2) :
  (Real.sqrt (x + 2))^2 = x + 2 := by sorry

-- Option D
theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) :
  x^0 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_identity_power_zero_eq_one_l1250_125007


namespace NUMINAMATH_CALUDE_factorization_of_ax2_minus_16ay2_l1250_125016

theorem factorization_of_ax2_minus_16ay2 (a x y : ℝ) : 
  a * x^2 - 16 * a * y^2 = a * (x + 4*y) * (x - 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_ax2_minus_16ay2_l1250_125016


namespace NUMINAMATH_CALUDE_bug_triangle_probability_sum_of_numerator_denominator_l1250_125078

/-- Probability of being at the starting vertex after n moves -/
def prob_at_start (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then 0
  else
    let prev := prob_at_start (n - 1)
    let prev_prev := prob_at_start (n - 2)
    (prev_prev + 2 * prev) / 4

theorem bug_triangle_probability :
  prob_at_start 10 = 171 / 1024 :=
sorry

#eval Nat.gcd 171 1024  -- To verify that 171 and 1024 are coprime

theorem sum_of_numerator_denominator :
  171 + 1024 = 1195 :=
sorry

end NUMINAMATH_CALUDE_bug_triangle_probability_sum_of_numerator_denominator_l1250_125078


namespace NUMINAMATH_CALUDE_sum_of_digits_divisible_by_13_l1250_125020

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For any 79 consecutive positive integers, there exists at least one whose sum of digits is divisible by 13 -/
theorem sum_of_digits_divisible_by_13 (start : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 79 ∧ (sum_of_digits (start + k)) % 13 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_divisible_by_13_l1250_125020


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1250_125047

theorem compound_interest_problem (P r : ℝ) : 
  P > 0 → r > 0 →
  P * (1 + r)^2 = 7000 →
  P * (1 + r)^3 = 9261 →
  P = 4000 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l1250_125047


namespace NUMINAMATH_CALUDE_candy_bar_cost_l1250_125082

/-- Given that the total cost of 2 candy bars is $4 and each candy bar costs the same amount,
    prove that the cost of each candy bar is $2. -/
theorem candy_bar_cost (total_cost : ℝ) (num_bars : ℕ) (cost_per_bar : ℝ) : 
  total_cost = 4 → num_bars = 2 → total_cost = num_bars * cost_per_bar → cost_per_bar = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l1250_125082


namespace NUMINAMATH_CALUDE_even_sum_theorem_l1250_125040

theorem even_sum_theorem (n : ℕ) (h1 : Odd n) 
  (h2 : (Finset.sum (Finset.filter Even (Finset.range n)) id) = 95 * 96) : 
  n = 191 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_theorem_l1250_125040


namespace NUMINAMATH_CALUDE_derivative_from_second_derivative_l1250_125079

open Real

theorem derivative_from_second_derivative
  (f : ℝ → ℝ)
  (h : ∀ x, deriv^[2] f x = 3) :
  ∀ x, deriv f x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_from_second_derivative_l1250_125079


namespace NUMINAMATH_CALUDE_solve_inequality_range_of_a_l1250_125098

-- Part 1
def inequality_solution_set (x : ℝ) : Prop :=
  x^2 - 5*x + 4 > 0

theorem solve_inequality :
  ∀ x : ℝ, inequality_solution_set x ↔ (x < 1 ∨ x > 4) :=
sorry

-- Part 2
def always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a*x + 4 > 0

theorem range_of_a :
  ∀ a : ℝ, always_positive a ↔ -4 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_solve_inequality_range_of_a_l1250_125098


namespace NUMINAMATH_CALUDE_roots_transformation_l1250_125009

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 5*r₁ + 12 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + 5*r₂ + 12 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + 5*r₃ + 12 = 0) →
  ((3*r₁)^3 - 12*(3*r₁)^2 + 45*(3*r₁) + 324 = 0) ∧
  ((3*r₂)^3 - 12*(3*r₂)^2 + 45*(3*r₂) + 324 = 0) ∧
  ((3*r₃)^3 - 12*(3*r₃)^2 + 45*(3*r₃) + 324 = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_transformation_l1250_125009


namespace NUMINAMATH_CALUDE_skiing_scavenger_ratio_is_two_to_one_l1250_125039

/-- Given a total number of students and the number of students for a scavenger hunting trip,
    calculates the ratio of skiing trip students to scavenger hunting trip students. -/
def skiing_to_scavenger_ratio (total : ℕ) (scavenger : ℕ) : ℚ :=
  let skiing := total - scavenger
  (skiing : ℚ) / (scavenger : ℚ)

/-- Theorem stating that given 12000 total students and 4000 for scavenger hunting,
    the ratio of skiing to scavenger hunting students is 2:1. -/
theorem skiing_scavenger_ratio_is_two_to_one :
  skiing_to_scavenger_ratio 12000 4000 = 2 := by
  sorry

end NUMINAMATH_CALUDE_skiing_scavenger_ratio_is_two_to_one_l1250_125039


namespace NUMINAMATH_CALUDE_cube_volume_problem_l1250_125051

theorem cube_volume_problem (a : ℝ) : 
  a > 0 →
  a^3 - ((a-1)*(a-1)*(a+1)) = 7 →
  a^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l1250_125051


namespace NUMINAMATH_CALUDE_faulty_faucet_leak_l1250_125015

/-- The amount of water leaked by a faulty faucet in half an hour -/
def water_leaked (leak_rate : ℝ) (time : ℝ) : ℝ :=
  leak_rate * time

theorem faulty_faucet_leak : 
  let leak_rate : ℝ := 65  -- grams per minute
  let time : ℝ := 30       -- half an hour in minutes
  water_leaked leak_rate time = 1950 := by
sorry

end NUMINAMATH_CALUDE_faulty_faucet_leak_l1250_125015


namespace NUMINAMATH_CALUDE_system_solution_l1250_125062

theorem system_solution : ∃! (x y : ℝ), 
  (2 * Real.sqrt (2 * x + 3 * y) + Real.sqrt (5 - x - y) = 7) ∧ 
  (3 * Real.sqrt (5 - x - y) - Real.sqrt (2 * x + y - 3) = 1) ∧ 
  x = 3 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1250_125062


namespace NUMINAMATH_CALUDE_amount_spent_on_toys_l1250_125095

def initial_amount : ℕ := 16
def amount_left : ℕ := 8

theorem amount_spent_on_toys :
  initial_amount - amount_left = 8 :=
by sorry

end NUMINAMATH_CALUDE_amount_spent_on_toys_l1250_125095


namespace NUMINAMATH_CALUDE_every_multiple_of_2_is_even_is_universal_l1250_125086

-- Define what it means for a number to be even
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define what it means for a number to be a multiple of 2
def MultipleOf2 (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define what a universal proposition is
def UniversalProposition (P : ℤ → Prop) : Prop :=
  ∀ x : ℤ, P x

-- Statement to prove
theorem every_multiple_of_2_is_even_is_universal :
  UniversalProposition (λ n : ℤ => MultipleOf2 n → IsEven n) :=
sorry

end NUMINAMATH_CALUDE_every_multiple_of_2_is_even_is_universal_l1250_125086


namespace NUMINAMATH_CALUDE_race_elimination_proof_l1250_125091

/-- The number of racers at the start of the race -/
def initial_racers : ℕ := 100

/-- The number of racers in the final section -/
def final_racers : ℕ := 30

/-- The fraction of racers remaining after the second segment -/
def second_segment_fraction : ℚ := 2/3

/-- The fraction of racers remaining after the third segment -/
def third_segment_fraction : ℚ := 1/2

/-- The number of racers eliminated after the first segment -/
def eliminated_first_segment : ℕ := 10

theorem race_elimination_proof :
  (↑final_racers : ℚ) = third_segment_fraction * second_segment_fraction * (initial_racers - eliminated_first_segment) :=
sorry

end NUMINAMATH_CALUDE_race_elimination_proof_l1250_125091


namespace NUMINAMATH_CALUDE_zoo_animals_l1250_125077

theorem zoo_animals (birds : ℕ) (non_birds : ℕ) : 
  birds = 450 → 
  birds = 5 * non_birds → 
  birds - non_birds = 360 := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_l1250_125077


namespace NUMINAMATH_CALUDE_original_average_age_proof_l1250_125066

theorem original_average_age_proof (initial_avg : ℝ) (new_students : ℕ) (new_students_avg : ℝ) (avg_decrease : ℝ) :
  initial_avg = 40 →
  new_students = 12 →
  new_students_avg = 34 →
  avg_decrease = 4 →
  initial_avg = 40 := by
sorry

end NUMINAMATH_CALUDE_original_average_age_proof_l1250_125066


namespace NUMINAMATH_CALUDE_hypotenuse_length_l1250_125065

/-- A right triangle with specific properties -/
structure RightTriangle where
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The side opposite to the 30° angle -/
  opposite_30 : ℝ
  /-- Condition: The triangle has a right angle -/
  right_angle : True
  /-- Condition: One angle is 30° -/
  angle_30 : True
  /-- Condition: The perimeter is 120 units -/
  perimeter_120 : perimeter = 120
  /-- Condition: The side opposite to 30° is half the hypotenuse -/
  opposite_half_hypotenuse : opposite_30 = hypotenuse / 2

/-- Theorem: The hypotenuse of the specified right triangle is 40(3 - √3) -/
theorem hypotenuse_length (t : RightTriangle) : t.hypotenuse = 40 * (3 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l1250_125065


namespace NUMINAMATH_CALUDE_function_value_theorem_l1250_125058

/-- Given functions f and g, prove that f(2) = 2 under certain conditions -/
theorem function_value_theorem (a b c : ℝ) (h_abc : a * b * c ≠ 0) :
  let f := fun (x : ℝ) ↦ a * x^2 + b * Real.cos x
  let g := fun (x : ℝ) ↦ c * Real.sin x
  (f 2 + g 2 = 3) → (f (-2) + g (-2) = 1) → f 2 = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_function_value_theorem_l1250_125058


namespace NUMINAMATH_CALUDE_village_x_current_population_l1250_125071

/-- The current population of Village X -/
def village_x_population : ℕ := 74000

/-- The annual decrease in Village X's population -/
def x_decrease_rate : ℕ := 1200

/-- The current population of Village Y -/
def village_y_population : ℕ := 42000

/-- The annual increase in Village Y's population -/
def y_increase_rate : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 16

/-- Theorem stating that the current population of Village X is 74,000 -/
theorem village_x_current_population :
  village_x_population = village_y_population + y_increase_rate * years_until_equal + x_decrease_rate * years_until_equal :=
by sorry

end NUMINAMATH_CALUDE_village_x_current_population_l1250_125071


namespace NUMINAMATH_CALUDE_product_of_sum_equals_three_times_product_l1250_125053

theorem product_of_sum_equals_three_times_product (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : x + y = 3 * x * y) (h4 : x + y ≠ 0) : x * y = (x + y) / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_equals_three_times_product_l1250_125053


namespace NUMINAMATH_CALUDE_cube_surface_area_l1250_125031

/-- Given a cube with volume 1728 cubic centimeters, its surface area is 864 square centimeters. -/
theorem cube_surface_area (volume : ℝ) (side : ℝ) :
  volume = 1728 →
  volume = side^3 →
  6 * side^2 = 864 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1250_125031


namespace NUMINAMATH_CALUDE_water_consumption_l1250_125093

/-- Proves that given a 1.5-quart bottle of water and a can of water,
    if the total amount of water drunk is 60 ounces,
    and 1 quart is equivalent to 32 ounces,
    then the can of water contains 12 ounces. -/
theorem water_consumption (bottle : ℚ) (can : ℚ) (total : ℚ) (quart_to_ounce : ℚ → ℚ) :
  bottle = 1.5 →
  total = 60 →
  quart_to_ounce 1 = 32 →
  can = total - quart_to_ounce bottle :=
by
  sorry

end NUMINAMATH_CALUDE_water_consumption_l1250_125093
