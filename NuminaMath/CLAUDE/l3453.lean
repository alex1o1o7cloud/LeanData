import Mathlib

namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3453_345399

def U : Set ℕ := {1, 2, 3, 4}

def A : Set ℕ := {x : ℕ | x ^ 2 - 5 * x + 4 < 0}

theorem complement_of_A_in_U :
  U \ A = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3453_345399


namespace NUMINAMATH_CALUDE_min_balls_to_draw_for_given_counts_l3453_345374

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat

/-- The minimum number of balls to draw to guarantee the desired outcome -/
def minBallsToDraw (counts : BallCounts) : Nat :=
  sorry

/-- The theorem stating the minimum number of balls to draw for the given problem -/
theorem min_balls_to_draw_for_given_counts :
  let counts := BallCounts.mk 30 25 20 15 10
  minBallsToDraw counts = 81 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_for_given_counts_l3453_345374


namespace NUMINAMATH_CALUDE_constant_expression_implies_a_equals_three_l3453_345386

theorem constant_expression_implies_a_equals_three (a : ℝ) :
  (∀ x : ℝ, x < 0 → ∃ c : ℝ, ∀ y : ℝ, y < 0 → 
    |y| + 2 * (y^2022)^(1/2022) + a * (y^2023)^(1/2023) = c) → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_constant_expression_implies_a_equals_three_l3453_345386


namespace NUMINAMATH_CALUDE_min_probability_cards_l3453_345370

/-- Represents the probability of a card being red-side up after two flips -/
def probability_red (k : ℕ) : ℚ :=
  if k ≤ 25 then
    (676 - 52 * k + 2 * k^2 : ℚ) / 676
  else
    (676 - 52 * (51 - k) + 2 * (51 - k)^2 : ℚ) / 676

/-- The total number of cards -/
def total_cards : ℕ := 50

/-- The number of cards flipped in each operation -/
def flip_size : ℕ := 25

/-- Theorem stating that cards 13 and 38 have the lowest probability of being red-side up -/
theorem min_probability_cards :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ total_cards →
    (probability_red 13 ≤ probability_red k ∧
     probability_red 38 ≤ probability_red k) :=
by sorry

end NUMINAMATH_CALUDE_min_probability_cards_l3453_345370


namespace NUMINAMATH_CALUDE_janet_piano_hours_l3453_345316

/-- Represents the number of hours per week Janet takes piano lessons -/
def piano_hours : ℕ := sorry

/-- The cost per hour of clarinet lessons -/
def clarinet_cost_per_hour : ℕ := 40

/-- The number of hours per week of clarinet lessons -/
def clarinet_hours_per_week : ℕ := 3

/-- The cost per hour of piano lessons -/
def piano_cost_per_hour : ℕ := 28

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The additional amount spent on piano lessons compared to clarinet lessons in a year -/
def additional_piano_cost : ℕ := 1040

theorem janet_piano_hours :
  piano_hours = 5 ∧
  clarinet_cost_per_hour * clarinet_hours_per_week * weeks_per_year + additional_piano_cost =
  piano_cost_per_hour * piano_hours * weeks_per_year :=
by sorry

end NUMINAMATH_CALUDE_janet_piano_hours_l3453_345316


namespace NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l3453_345311

/-- The number of pumpkins at Moonglow Orchard -/
def x : ℕ := 14

/-- The number of pumpkins at Sunshine Orchard -/
def y : ℕ := 3 * x^2 + 12

theorem sunshine_orchard_pumpkins : y = 600 := by
  sorry

end NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l3453_345311


namespace NUMINAMATH_CALUDE_greatest_b_value_l3453_345364

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, x^2 - 14*x + 45 ≤ 0 → x ≤ 9) ∧ 
  (9^2 - 14*9 + 45 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l3453_345364


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3453_345352

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (a + 1)*x + a > 0}
  if a < 1 then S = {x : ℝ | x < a ∨ x > 1}
  else if a = 1 then S = {x : ℝ | x ≠ 1}
  else S = {x : ℝ | x < 1 ∨ x > a} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3453_345352


namespace NUMINAMATH_CALUDE_intersection_not_in_third_quadrant_l3453_345390

/-- The intersection point of y = 2x + m and y = -x + 3 cannot be in the third quadrant -/
theorem intersection_not_in_third_quadrant (m : ℝ) : 
  ∀ x y : ℝ, y = 2*x + m ∧ y = -x + 3 → ¬(x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_not_in_third_quadrant_l3453_345390


namespace NUMINAMATH_CALUDE_function_is_periodic_l3453_345328

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the parameters a and b
variable (a b : ℝ)

-- State the conditions
axiom cond1 : ∀ x, f x = f (2 * b - x)
axiom cond2 : ∀ x, f (a + x) = -f (a - x)
axiom cond3 : a ≠ b

-- State the theorem
theorem function_is_periodic : ∀ x, f x = f (x + 4 * (a - b)) := by sorry

end NUMINAMATH_CALUDE_function_is_periodic_l3453_345328


namespace NUMINAMATH_CALUDE_junk_mail_per_block_l3453_345381

/-- Given that a mailman distributes junk mail to blocks with the following conditions:
  1. The mailman gives 8 mails to each house in a block.
  2. There are 4 houses in a block.
Prove that the number of pieces of junk mail given to each block is 32. -/
theorem junk_mail_per_block (mails_per_house : ℕ) (houses_per_block : ℕ) 
  (h1 : mails_per_house = 8) (h2 : houses_per_block = 4) : 
  mails_per_house * houses_per_block = 32 := by
  sorry

#check junk_mail_per_block

end NUMINAMATH_CALUDE_junk_mail_per_block_l3453_345381


namespace NUMINAMATH_CALUDE_commission_rate_is_four_percent_l3453_345384

/-- Calculates the commission rate given base pay, goal earnings, and required sales. -/
def calculate_commission_rate (base_pay : ℚ) (goal_earnings : ℚ) (required_sales : ℚ) : ℚ :=
  ((goal_earnings - base_pay) / required_sales) * 100

/-- Proves that the commission rate is 4% given the problem conditions. -/
theorem commission_rate_is_four_percent
  (base_pay : ℚ)
  (goal_earnings : ℚ)
  (required_sales : ℚ)
  (h1 : base_pay = 190)
  (h2 : goal_earnings = 500)
  (h3 : required_sales = 7750) :
  calculate_commission_rate base_pay goal_earnings required_sales = 4 := by
  sorry

#eval calculate_commission_rate 190 500 7750

end NUMINAMATH_CALUDE_commission_rate_is_four_percent_l3453_345384


namespace NUMINAMATH_CALUDE_beverage_production_l3453_345346

/-- Represents the number of bottles of beverage A -/
def bottles_A : ℕ := sorry

/-- Represents the number of bottles of beverage B -/
def bottles_B : ℕ := sorry

/-- The amount of additive (in grams) required for one bottle of beverage A -/
def additive_A : ℚ := 1/5

/-- The amount of additive (in grams) required for one bottle of beverage B -/
def additive_B : ℚ := 3/10

/-- The total number of bottles produced -/
def total_bottles : ℕ := 200

/-- The total amount of additive used (in grams) -/
def total_additive : ℚ := 54

theorem beverage_production :
  bottles_A + bottles_B = total_bottles ∧
  additive_A * bottles_A + additive_B * bottles_B = total_additive ∧
  bottles_A = 60 ∧
  bottles_B = 140 := by sorry

end NUMINAMATH_CALUDE_beverage_production_l3453_345346


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_l3453_345300

/-- Calculates the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Calculates the number of digits of a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Converts a positive integer to its binary representation -/
def to_binary (n : ℕ) : List ℕ := sorry

/-- Calculates the cost of transmitting a number using Option 1 -/
def cost_option1 (n : ℕ) : ℕ :=
  sum_of_digits n + num_digits n

/-- Calculates the cost of transmitting a number using Option 2 -/
def cost_option2 (n : ℕ) : ℕ :=
  let binary := to_binary n
  (binary.filter (· = 1)).length + (binary.filter (· = 0)).length + binary.length

/-- Checks if the costs are equal for both options -/
def costs_equal (n : ℕ) : Prop :=
  cost_option1 n = cost_option2 n

theorem largest_equal_cost_number :
  ∀ n : ℕ, n < 2000 → n > 1539 → ¬(costs_equal n) := by sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_l3453_345300


namespace NUMINAMATH_CALUDE_andrew_remaining_vacation_days_l3453_345375

/-- Calculates the remaining vacation days for an employee given their work days and vacation days taken. -/
def remaining_vacation_days (work_days : ℕ) (march_vacation : ℕ) : ℕ :=
  let earned_days := work_days / 10
  let taken_days := march_vacation + 2 * march_vacation
  earned_days - taken_days

/-- Theorem stating that Andrew has 15 remaining vacation days. -/
theorem andrew_remaining_vacation_days :
  remaining_vacation_days 300 5 = 15 := by
  sorry

#eval remaining_vacation_days 300 5

end NUMINAMATH_CALUDE_andrew_remaining_vacation_days_l3453_345375


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l3453_345321

/-- Theorem: Weight of replaced person in a group
Given a group of 9 persons, if replacing one person with a new person weighing 87.5 kg
increases the average weight by 2.5 kg, then the weight of the replaced person was 65 kg. -/
theorem weight_of_replaced_person
  (n : ℕ) -- number of persons in the group
  (w : ℝ) -- total weight of the original group
  (new_weight : ℝ) -- weight of the new person
  (avg_increase : ℝ) -- increase in average weight
  (h1 : n = 9)
  (h2 : new_weight = 87.5)
  (h3 : avg_increase = 2.5)
  (h4 : (w - (w / n) + new_weight) / n = (w / n) + avg_increase) :
  w / n = 65 :=
sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l3453_345321


namespace NUMINAMATH_CALUDE_divisor_problem_l3453_345302

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 199 →
  quotient = 11 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 18 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l3453_345302


namespace NUMINAMATH_CALUDE_cleaning_earnings_l3453_345363

/-- Calculates the total earnings for cleaning a building -/
def total_earnings (floors : ℕ) (rooms_per_floor : ℕ) (hours_per_room : ℕ) (hourly_rate : ℕ) : ℕ :=
  floors * rooms_per_floor * hours_per_room * hourly_rate

/-- Proves that cleaning a 4-floor building with 10 rooms per floor,
    taking 6 hours per room at $15 per hour, results in $3600 earnings -/
theorem cleaning_earnings :
  total_earnings 4 10 6 15 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_earnings_l3453_345363


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3453_345357

theorem fraction_to_decimal : (11 : ℚ) / 16 = 0.6875 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3453_345357


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3453_345314

theorem quadratic_inequality (a b c A B C : ℝ) 
  (ha : a ≠ 0) (hA : A ≠ 0)
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4*a*c| ≤ |B^2 - 4*A*C| := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3453_345314


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3453_345376

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (x - 2) > 0 ↔ x < 1/3 ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3453_345376


namespace NUMINAMATH_CALUDE_car_capacities_and_rental_plans_l3453_345365

/-- The capacity of a type A car in tons -/
def capacity_A : ℕ := 3

/-- The capacity of a type B car in tons -/
def capacity_B : ℕ := 4

/-- The total weight of goods to be transported -/
def total_weight : ℕ := 31

/-- A rental plan is a pair of natural numbers (a, b) where a is the number of type A cars and b is the number of type B cars -/
def RentalPlan := ℕ × ℕ

/-- The set of all valid rental plans -/
def valid_rental_plans : Set RentalPlan :=
  {plan | plan.1 * capacity_A + plan.2 * capacity_B = total_weight}

theorem car_capacities_and_rental_plans :
  (2 * capacity_A + capacity_B = 10) ∧
  (capacity_A + 2 * capacity_B = 11) ∧
  (valid_rental_plans = {(1, 7), (5, 4), (9, 1)}) := by
  sorry


end NUMINAMATH_CALUDE_car_capacities_and_rental_plans_l3453_345365


namespace NUMINAMATH_CALUDE_real_number_pure_imaginary_condition_l3453_345349

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for a pure imaginary number
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem real_number_pure_imaginary_condition (m : ℝ) : 
  isPureImaginary (m^2 * (1 + i) + (m - i) - 2) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_real_number_pure_imaginary_condition_l3453_345349


namespace NUMINAMATH_CALUDE_world_cup_stats_l3453_345331

def world_cup_data : List ℕ := [32, 31, 16, 16, 14, 12]

def median (l : List ℕ) : ℚ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem world_cup_stats :
  median world_cup_data = 16 ∧ mode world_cup_data = 16 := by sorry

end NUMINAMATH_CALUDE_world_cup_stats_l3453_345331


namespace NUMINAMATH_CALUDE_square_area_increase_l3453_345350

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.3 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.69 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l3453_345350


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l3453_345315

/-- Given a right triangle with legs of length 8 and 15, the diameter of its inscribed circle is 6 -/
theorem inscribed_circle_diameter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_a : a = 8) (h_b : b = 15) : 
  2 * (a * b) / (a + b + c) = 6 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l3453_345315


namespace NUMINAMATH_CALUDE_sequence_sum_l3453_345342

theorem sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) :
  x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1 →
  4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 12 →
  9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 123 →
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 334 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3453_345342


namespace NUMINAMATH_CALUDE_triangle_uniqueness_l3453_345385

/-- Triangle defined by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

/-- Two triangles are congruent if their corresponding sides are equal -/
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

theorem triangle_uniqueness (t1 t2 : Triangle) : 
  t1.a = t2.a → t1.b = t2.b → t1.c = t2.c → congruent t1 t2 := by
  sorry

#check triangle_uniqueness

end NUMINAMATH_CALUDE_triangle_uniqueness_l3453_345385


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3453_345369

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 + x / 3) ^ (1/3 : ℝ) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3453_345369


namespace NUMINAMATH_CALUDE_channel_transmission_theorem_l3453_345387

/-- Channel transmission probabilities -/
structure ChannelProb where
  α : ℝ
  β : ℝ
  h_α_pos : 0 < α
  h_α_lt_one : α < 1
  h_β_pos : 0 < β
  h_β_lt_one : β < 1

/-- Single transmission probability for sequence 1, 0, 1 -/
def single_trans_prob (cp : ChannelProb) : ℝ := (1 - cp.α) * (1 - cp.β)^2

/-- Triple transmission probability for decoding 0 as 0 -/
def triple_trans_prob_0 (cp : ChannelProb) : ℝ :=
  (1 - cp.α)^3 + 3 * cp.α * (1 - cp.α)^2

/-- Single transmission probability for decoding 0 as 0 -/
def single_trans_prob_0 (cp : ChannelProb) : ℝ := 1 - cp.α

theorem channel_transmission_theorem (cp : ChannelProb) :
  single_trans_prob cp = (1 - cp.α) * (1 - cp.β)^2 ∧
  (cp.α < 1/2 → triple_trans_prob_0 cp > single_trans_prob_0 cp) := by sorry

end NUMINAMATH_CALUDE_channel_transmission_theorem_l3453_345387


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l3453_345325

/-- Given a stamp price of 25 cents and $40 available, 
    the maximum number of stamps that can be purchased is 160. -/
theorem max_stamps_purchasable (stamp_price : ℕ) (available_money : ℕ) :
  stamp_price = 25 → available_money = 4000 → 
  (∀ n : ℕ, n * stamp_price ≤ available_money → n ≤ 160) ∧
  160 * stamp_price ≤ available_money :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l3453_345325


namespace NUMINAMATH_CALUDE_hotel_room_cost_l3453_345383

theorem hotel_room_cost (total_rooms : ℕ) (double_room_cost : ℕ) (total_revenue : ℕ) (single_rooms : ℕ) :
  total_rooms = 260 →
  double_room_cost = 60 →
  total_revenue = 14000 →
  single_rooms = 64 →
  ∃ (single_room_cost : ℕ),
    single_room_cost = 35 ∧
    single_room_cost * single_rooms + double_room_cost * (total_rooms - single_rooms) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_hotel_room_cost_l3453_345383


namespace NUMINAMATH_CALUDE_C_is_hyperbola_l3453_345335

/-- The curve C is defined by the equation 3y^2 - 4(x+1)y + 12(x-2) = 0 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.2^2 - 4 * (p.1 + 1) * p.2 + 12 * (p.1 - 2) = 0}

/-- The discriminant of the quadratic equation in y -/
def discriminant (x : ℝ) : ℝ :=
  16 * x^2 - 112 * x + 304

/-- Theorem: The curve C is a hyperbola -/
theorem C_is_hyperbola : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1^2 / a^2) - (p.2^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_C_is_hyperbola_l3453_345335


namespace NUMINAMATH_CALUDE_three_number_problem_l3453_345396

theorem three_number_problem :
  ∃ (x y z : ℝ),
    x = 45 ∧ y = 37.5 ∧ z = 22.5 ∧
    x - y = (1/3) * z ∧
    y - z = (1/3) * x ∧
    z - 10 = (1/3) * y :=
by
  sorry

end NUMINAMATH_CALUDE_three_number_problem_l3453_345396


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3453_345332

theorem digit_sum_problem (a p v e s r : ℕ) 
  (h1 : a + p = v)
  (h2 : v + e = s)
  (h3 : s + a = r)
  (h4 : p + e + r = 14)
  (h5 : a ≠ 0 ∧ p ≠ 0 ∧ v ≠ 0 ∧ e ≠ 0 ∧ s ≠ 0 ∧ r ≠ 0) :
  s = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3453_345332


namespace NUMINAMATH_CALUDE_weitzenboeck_inequality_tetrahedron_l3453_345333

/-- A tetrahedron with edge lengths a, b, c, d, e, f and surface area S. -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  S : ℝ

/-- The Weitzenböck inequality for tetrahedra. -/
theorem weitzenboeck_inequality_tetrahedron (t : Tetrahedron) :
  t.S ≤ (Real.sqrt 3 / 6) * (t.a^2 + t.b^2 + t.c^2 + t.d^2 + t.e^2 + t.f^2) := by
  sorry

end NUMINAMATH_CALUDE_weitzenboeck_inequality_tetrahedron_l3453_345333


namespace NUMINAMATH_CALUDE_no_solution_when_p_is_seven_l3453_345312

theorem no_solution_when_p_is_seven (p : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - p) / (x - 8)) ↔ p = 7 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_when_p_is_seven_l3453_345312


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3453_345356

theorem cubic_root_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  24 * a^3 - 36 * a^2 + 16 * a - 1 = 0 →
  24 * b^3 - 36 * b^2 + 16 * b - 1 = 0 →
  24 * c^3 - 36 * c^2 + 16 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3453_345356


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_315_l3453_345372

/-- The sum of the digits in the binary representation of 315 is 6. -/
theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_315_l3453_345372


namespace NUMINAMATH_CALUDE_paint_per_statue_l3453_345306

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) 
  (h1 : total_paint = 7/8)
  (h2 : num_statues = 7) : 
  total_paint / num_statues = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_paint_per_statue_l3453_345306


namespace NUMINAMATH_CALUDE_odd_spaced_stone_selections_count_l3453_345359

/-- The number of ways to select 5 stones from 15 stones in a line, 
    such that there are an odd number of stones between any two selected stones. -/
def oddSpacedStoneSelections : ℕ := 77

/-- The total number of stones in the line. -/
def totalStones : ℕ := 15

/-- The number of stones to be selected. -/
def stonesToSelect : ℕ := 5

/-- The number of odd-numbered stones in the line. -/
def oddNumberedStones : ℕ := 8

/-- The number of even-numbered stones in the line. -/
def evenNumberedStones : ℕ := 7

theorem odd_spaced_stone_selections_count :
  oddSpacedStoneSelections = Nat.choose oddNumberedStones stonesToSelect + Nat.choose evenNumberedStones stonesToSelect :=
by sorry

end NUMINAMATH_CALUDE_odd_spaced_stone_selections_count_l3453_345359


namespace NUMINAMATH_CALUDE_all_crop_to_diagonal_l3453_345373

/-- A symmetric kite-shaped field -/
structure KiteField where
  long_side : ℝ
  short_side : ℝ
  angle : ℝ
  long_side_positive : 0 < long_side
  short_side_positive : 0 < short_side
  angle_range : 0 < angle ∧ angle < π

/-- The fraction of the field area closer to the longer diagonal -/
def fraction_closer_to_diagonal (k : KiteField) : ℝ :=
  1 -- Definition, not proof

/-- The theorem statement -/
theorem all_crop_to_diagonal (k : KiteField) 
  (h1 : k.long_side = 100)
  (h2 : k.short_side = 70)
  (h3 : k.angle = 2 * π / 3) :
  fraction_closer_to_diagonal k = 1 := by
  sorry

end NUMINAMATH_CALUDE_all_crop_to_diagonal_l3453_345373


namespace NUMINAMATH_CALUDE_complement_intersection_eq_set_l3453_345304

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_eq_set : (Aᶜ ∩ Bᶜ) = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_eq_set_l3453_345304


namespace NUMINAMATH_CALUDE_hostel_provisions_l3453_345355

/-- The number of men initially in the hostel -/
def initial_men : ℕ := 250

/-- The number of days the provisions last initially -/
def initial_days : ℕ := 40

/-- The number of men who leave the hostel -/
def men_who_leave : ℕ := 50

/-- The number of days the provisions last after some men leave -/
def days_after_leaving : ℕ := 50

theorem hostel_provisions :
  initial_men * initial_days = (initial_men - men_who_leave) * days_after_leaving :=
by sorry

#check hostel_provisions

end NUMINAMATH_CALUDE_hostel_provisions_l3453_345355


namespace NUMINAMATH_CALUDE_modulo_residue_sum_l3453_345348

theorem modulo_residue_sum : (255 + 7 * 51 + 9 * 187 + 5 * 34) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_sum_l3453_345348


namespace NUMINAMATH_CALUDE_certain_number_bound_l3453_345391

theorem certain_number_bound (x y z : ℤ) (N : ℝ) 
  (h1 : x < y ∧ y < z)
  (h2 : (y - x : ℝ) > N)
  (h3 : Even x)
  (h4 : Odd y ∧ Odd z)
  (h5 : ∀ (a b : ℤ), (Even a ∧ Odd b ∧ a < b) → (b - a ≥ 7) → (z - x ≤ b - a)) :
  N < 3 := by
sorry

end NUMINAMATH_CALUDE_certain_number_bound_l3453_345391


namespace NUMINAMATH_CALUDE_parabola_equation_l3453_345303

/-- A parabola passing through the point (4, -2) has a standard equation of either x² = -8y or y² = x -/
theorem parabola_equation (p : ℝ × ℝ) (h : p = (4, -2)) :
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ {(x, y) | x^2 = -8*y}) ∨
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ {(x, y) | y^2 = x}) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3453_345303


namespace NUMINAMATH_CALUDE_quadratic_intersection_l3453_345377

/-- Given two quadratic functions f(x) = ax^2 + bx + c and g(x) = 4ax^2 + 2bx + c,
    where b ≠ 0 and c ≠ 0, their intersection points are x = 0 and x = -b/(3a) -/
theorem quadratic_intersection
  (a b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f := fun x : ℝ => a * x^2 + b * x + c
  let g := fun x : ℝ => 4 * a * x^2 + 2 * b * x + c
  (∃ y, f 0 = y ∧ g 0 = y) ∧
  (∃ y, f (-b / (3 * a)) = y ∧ g (-b / (3 * a)) = y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l3453_345377


namespace NUMINAMATH_CALUDE_greater_solution_of_quadratic_l3453_345395

theorem greater_solution_of_quadratic (x : ℝ) : 
  x^2 + 20*x - 96 = 0 → x ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_greater_solution_of_quadratic_l3453_345395


namespace NUMINAMATH_CALUDE_function_symmetry_range_l3453_345393

open Real

theorem function_symmetry_range (a : ℝ) : 
  (∃ x ∈ Set.Icc (1/ℯ) ℯ, a + 8 * log x = x^2 + 2) ↔ 
  a ∈ Set.Icc (6 - 8 * log 2) (10 + 1 / ℯ^2) :=
sorry

end NUMINAMATH_CALUDE_function_symmetry_range_l3453_345393


namespace NUMINAMATH_CALUDE_total_project_hours_l3453_345354

def project_hours (kate_hours : ℝ) : ℝ × ℝ × ℝ :=
  let pat_hours := 2 * kate_hours
  let mark_hours := kate_hours + 75
  (pat_hours, kate_hours, mark_hours)

theorem total_project_hours :
  ∃ (kate_hours : ℝ),
    let (pat_hours, _, mark_hours) := project_hours kate_hours
    pat_hours = (1/3) * mark_hours ∧
    (pat_hours + kate_hours + mark_hours) = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_project_hours_l3453_345354


namespace NUMINAMATH_CALUDE_tournament_outcomes_l3453_345341

/-- Represents the number of players in the tournament -/
def num_players : Nat := 5

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : Nat := 2

/-- Represents the number of elimination rounds -/
def num_rounds : Nat := 4

/-- Calculates the total number of possible outcomes for the tournament -/
def total_outcomes : Nat := outcomes_per_match ^ num_rounds

/-- Theorem stating that the total number of possible outcomes is 16 -/
theorem tournament_outcomes :
  total_outcomes = 16 := by sorry

end NUMINAMATH_CALUDE_tournament_outcomes_l3453_345341


namespace NUMINAMATH_CALUDE_expression_equals_one_l3453_345309

theorem expression_equals_one :
  let x : ℝ := 40 * π / 180  -- 40 degrees in radians
  let y : ℝ := 50 * π / 180  -- 50 degrees in radians
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) / (Real.cos x - Real.sqrt (1 - Real.sin y ^ 2)) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3453_345309


namespace NUMINAMATH_CALUDE_sum_of_roots_l3453_345327

def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3453_345327


namespace NUMINAMATH_CALUDE_otimes_calculation_l3453_345397

-- Define the new operation ⊗
def otimes (a b : ℚ) : ℚ := a^2 - a*b

-- State the theorem
theorem otimes_calculation :
  otimes (-5) (otimes 3 (-2)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_otimes_calculation_l3453_345397


namespace NUMINAMATH_CALUDE_sum_of_lengths_l3453_345360

-- Define the conversion factors
def meters_to_cm : ℝ := 100
def meters_to_mm : ℝ := 1000

-- Define the values in their original units
def length_m : ℝ := 2
def length_cm : ℝ := 3
def length_mm : ℝ := 5

-- State the theorem
theorem sum_of_lengths :
  length_m + length_cm / meters_to_cm + length_mm / meters_to_mm = 2.035 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_lengths_l3453_345360


namespace NUMINAMATH_CALUDE_cosine_sum_difference_l3453_345358

theorem cosine_sum_difference : 
  Real.cos (π / 15) - Real.cos (2 * π / 15) - Real.cos (4 * π / 15) + Real.cos (7 * π / 15) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_difference_l3453_345358


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_line_passes_through_fixed_point_l3453_345334

/-- A circle that passes through (1, 0) and is tangent to x = -1 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_F : (center.1 - 1)^2 + center.2^2 = radius^2
  tangent_to_l : center.1 + radius = 1

/-- The trajectory of the center of the TangentCircle -/
def trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Two distinct points on the trajectory, neither being the origin -/
structure TrajectoryPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  A_on_trajectory : A ∈ trajectory
  B_on_trajectory : B ∈ trajectory
  A_not_origin : A ≠ (0, 0)
  B_not_origin : B ≠ (0, 0)
  A_ne_B : A ≠ B
  y_product_ne_neg16 : A.2 * B.2 ≠ -16

theorem trajectory_is_parabola (c : TangentCircle) : c.center ∈ trajectory := by sorry

theorem line_passes_through_fixed_point (p : TrajectoryPoints) :
  ∃ t : ℝ, t * (p.B.1 - p.A.1) + p.A.1 = 4 ∧ t * (p.B.2 - p.A.2) + p.A.2 = 0 := by sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_line_passes_through_fixed_point_l3453_345334


namespace NUMINAMATH_CALUDE_f_sum_positive_l3453_345343

def f (x : ℝ) := x^3 + x

theorem f_sum_positive (a b : ℝ) (h : a + b > 0) : f a + f b > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_positive_l3453_345343


namespace NUMINAMATH_CALUDE_quartic_polynomial_extrema_bounds_l3453_345339

/-- A polynomial of degree 4 with real coefficients -/
def QuarticPolynomial (a₀ a₁ a₂ : ℝ) : ℝ → ℝ := λ x ↦ x^4 + a₁*x^3 + a₂*x^2 + a₁*x + a₀

/-- The local maximum of a function -/
noncomputable def LocalMax (f : ℝ → ℝ) : ℝ := sorry

/-- The local minimum of a function -/
noncomputable def LocalMin (f : ℝ → ℝ) : ℝ := sorry

/-- Theorem: Bounds for the difference between local maximum and minimum of a quartic polynomial -/
theorem quartic_polynomial_extrema_bounds (a₀ a₁ a₂ : ℝ) :
  let f := QuarticPolynomial a₀ a₁ a₂
  let M := LocalMax f
  let m := LocalMin f
  3/10 * (a₁^2/4 - 2*a₂/9)^2 < M - m ∧ M - m < 3 * (a₁^2/4 - 2*a₂/9)^2 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_extrema_bounds_l3453_345339


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_l3453_345329

theorem smallest_integer_with_remainder (k : ℕ) : k = 275 ↔ 
  k > 1 ∧ 
  k % 13 = 2 ∧ 
  k % 7 = 2 ∧ 
  k % 3 = 2 ∧ 
  ∀ m : ℕ, m > 1 → m % 13 = 2 → m % 7 = 2 → m % 3 = 2 → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_l3453_345329


namespace NUMINAMATH_CALUDE_area_of_triangle_PMF_l3453_345301

/-- A parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  directrix : ℝ
  focus : ℝ × ℝ

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- The foot of the perpendicular from a point to the directrix -/
def footOfPerpendicular (p : Parabola) (point : PointOnParabola p) : ℝ × ℝ :=
  (p.directrix, point.y)

/-- The theorem stating the area of the triangle PMF -/
theorem area_of_triangle_PMF (p : Parabola) (P : PointOnParabola p) :
  p.equation = (fun x y => y^2 = 4*x) →
  p.directrix = -1 →
  p.focus = (1, 0) →
  (P.x - p.directrix)^2 + P.y^2 = 5^2 →
  let M := footOfPerpendicular p P
  let F := p.focus
  let area := (1/2) * |P.y| * 5
  area = 10 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_PMF_l3453_345301


namespace NUMINAMATH_CALUDE_binomial_sum_l3453_345326

theorem binomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) : 
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 233 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l3453_345326


namespace NUMINAMATH_CALUDE_sara_coin_collection_value_l3453_345322

/-- Calculates the total value in cents of a coin collection --/
def total_cents (quarters dimes nickels pennies : ℕ) : ℕ :=
  quarters * 25 + dimes * 10 + nickels * 5 + pennies

/-- Proves that Sara's coin collection totals 453 cents --/
theorem sara_coin_collection_value :
  total_cents 11 8 15 23 = 453 := by
  sorry

end NUMINAMATH_CALUDE_sara_coin_collection_value_l3453_345322


namespace NUMINAMATH_CALUDE_rectangle_perimeter_13km_l3453_345317

/-- The perimeter of a rectangle with both sides equal to 13 km is 52 km. -/
theorem rectangle_perimeter_13km (l w : ℝ) : 
  l = 13 → w = 13 → 2 * (l + w) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_13km_l3453_345317


namespace NUMINAMATH_CALUDE_stating_three_card_draw_probability_value_l3453_345338

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck := Fin 52

/-- The probability of drawing a specific sequence of three cards from a standard deck -/
def three_card_draw_probability : ℚ :=
  -- Probability of first card being a non-heart King
  (3 : ℚ) / 52 *
  -- Probability of second card being a heart (not King of hearts)
  12 / 51 *
  -- Probability of third card being a spade or diamond
  26 / 50

/-- 
Theorem stating that the probability of drawing a non-heart King, 
then a heart (not King of hearts), then a spade or diamond 
from a standard 52-card deck is 26/3675
-/
theorem three_card_draw_probability_value : 
  three_card_draw_probability = 26 / 3675 := by
  sorry

end NUMINAMATH_CALUDE_stating_three_card_draw_probability_value_l3453_345338


namespace NUMINAMATH_CALUDE_all_cards_same_number_l3453_345347

theorem all_cards_same_number (n : ℕ) (c : Fin n → ℕ) : 
  (∀ i : Fin n, c i ∈ Finset.range n) →
  (∀ s : Finset (Fin n), (s.sum c) % (n + 1) ≠ 0) →
  (∀ i j : Fin n, c i = c j) :=
by sorry

end NUMINAMATH_CALUDE_all_cards_same_number_l3453_345347


namespace NUMINAMATH_CALUDE_constant_integral_equals_one_l3453_345366

theorem constant_integral_equals_one : ∫ x in (0:ℝ)..1, (1:ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_constant_integral_equals_one_l3453_345366


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3453_345388

-- Problem 1
theorem problem_1 : (4 - Real.pi) ^ 0 + (1/3)⁻¹ - 2 * Real.cos (45 * π / 180) = 4 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3453_345388


namespace NUMINAMATH_CALUDE_farmer_initial_apples_l3453_345308

/-- The number of apples given away by the farmer -/
def apples_given_away : ℕ := 88

/-- The number of apples the farmer has left -/
def apples_left : ℕ := 39

/-- The initial number of apples the farmer had -/
def initial_apples : ℕ := apples_given_away + apples_left

/-- Theorem: The farmer initially had 127 apples -/
theorem farmer_initial_apples : initial_apples = 127 := by
  sorry

end NUMINAMATH_CALUDE_farmer_initial_apples_l3453_345308


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l3453_345379

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 Real.pi) ∧ 
  (∀ x ∈ Set.Icc 0 Real.pi, Real.cos (x + y) ≥ Real.cos x + Real.cos y) → 
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l3453_345379


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l3453_345313

/-- The rectangular coordinate equation of a curve given its polar equation -/
theorem polar_to_rectangular (ρ θ : ℝ) (h : ρ * Real.cos θ = 2) : 
  ∃ x : ℝ, x = 2 := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l3453_345313


namespace NUMINAMATH_CALUDE_no_prime_power_solution_l3453_345394

theorem no_prime_power_solution : 
  ¬ ∃ (p : ℕ) (x : ℕ) (k : ℕ), 
    Nat.Prime p ∧ x^5 + 2*x + 3 = p^k :=
sorry

end NUMINAMATH_CALUDE_no_prime_power_solution_l3453_345394


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3453_345307

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of Quadrant I -/
def inQuadrantI (p : Point2D) : Prop := p.x > 0 ∧ p.y > 0

/-- Definition of Quadrant II -/
def inQuadrantII (p : Point2D) : Prop := p.x < 0 ∧ p.y > 0

/-- The set of points satisfying the given inequalities -/
def satisfiesInequalities (p : Point2D) : Prop :=
  p.y > 3 * p.x ∧ p.y > 6 - 2 * p.x

theorem points_in_quadrants_I_and_II :
  ∀ p : Point2D, satisfiesInequalities p → inQuadrantI p ∨ inQuadrantII p :=
by sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3453_345307


namespace NUMINAMATH_CALUDE_max_value_a_l3453_345368

theorem max_value_a (a : ℝ) : 
  (∀ x > 0, x * Real.exp x - a * (x + 1) ≥ Real.log x) → a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l3453_345368


namespace NUMINAMATH_CALUDE_birthday_cake_icing_l3453_345323

/-- Represents a 3D cube --/
structure Cube :=
  (size : ℕ)

/-- Represents the icing configuration on the cube --/
structure IcingConfig :=
  (top : Bool)
  (bottom : Bool)
  (side1 : Bool)
  (side2 : Bool)
  (side3 : Bool)
  (side4 : Bool)

/-- Counts the number of unit cubes with exactly two iced sides --/
def countTwoSidedIcedCubes (c : Cube) (ic : IcingConfig) : ℕ :=
  sorry

/-- The main theorem --/
theorem birthday_cake_icing (c : Cube) (ic : IcingConfig) :
  c.size = 5 →
  ic.top = true →
  ic.bottom = true →
  ic.side1 = true →
  ic.side2 = true →
  ic.side3 = false →
  ic.side4 = false →
  countTwoSidedIcedCubes c ic = 20 :=
sorry

end NUMINAMATH_CALUDE_birthday_cake_icing_l3453_345323


namespace NUMINAMATH_CALUDE_prob_different_grades_is_two_thirds_l3453_345362

/-- Represents the number of students in each grade -/
def students_per_grade : ℕ := 2

/-- Represents the total number of students -/
def total_students : ℕ := 2 * students_per_grade

/-- Represents the number of students to be selected -/
def selected_students : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Represents the probability of selecting students from different grades -/
def prob_different_grades : ℚ := 
  (choose students_per_grade 1 * choose students_per_grade 1 : ℚ) / 
  (choose total_students selected_students : ℚ)

theorem prob_different_grades_is_two_thirds : 
  prob_different_grades = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_different_grades_is_two_thirds_l3453_345362


namespace NUMINAMATH_CALUDE_jorge_corn_yield_l3453_345310

/-- Represents the yield calculation for Jorge's corn plantation --/
def jorge_yield (total_acres : ℝ) (clay_rich_fraction : ℝ) (total_yield : ℝ) (other_soil_yield : ℝ) : Prop :=
  let clay_rich_acres := clay_rich_fraction * total_acres
  let other_soil_acres := (1 - clay_rich_fraction) * total_acres
  let clay_rich_yield := (other_soil_yield / 2) * clay_rich_acres
  let other_soil_total_yield := other_soil_yield * other_soil_acres
  clay_rich_yield + other_soil_total_yield = total_yield

theorem jorge_corn_yield :
  jorge_yield 60 (1/3) 20000 400 := by
  sorry

end NUMINAMATH_CALUDE_jorge_corn_yield_l3453_345310


namespace NUMINAMATH_CALUDE_unique_root_of_unity_polynomial_l3453_345318

def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z^n = 1

def is_cube_root_of_unity (z : ℂ) : Prop :=
  ∃ k : ℕ, z^(3*k) = 1

theorem unique_root_of_unity_polynomial (c d : ℤ) :
  ∃! z : ℂ, is_root_of_unity z ∧ is_cube_root_of_unity z ∧ z^3 + c*z + d = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_root_of_unity_polynomial_l3453_345318


namespace NUMINAMATH_CALUDE_correct_sums_l3453_345351

theorem correct_sums (total : ℕ) (wrong_ratio : ℕ) (correct : ℕ) : 
  total = 36 → 
  wrong_ratio = 2 → 
  total = correct + wrong_ratio * correct → 
  correct = 12 := by sorry

end NUMINAMATH_CALUDE_correct_sums_l3453_345351


namespace NUMINAMATH_CALUDE_scientific_notation_864000_l3453_345371

theorem scientific_notation_864000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 864000 = a * (10 : ℝ) ^ n :=
by
  use 8.64, 5
  sorry

end NUMINAMATH_CALUDE_scientific_notation_864000_l3453_345371


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3453_345398

theorem unique_integer_solution : ∃! (x y : ℤ), 10*x + 18*y = 28 ∧ 18*x + 10*y = 56 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3453_345398


namespace NUMINAMATH_CALUDE_probability_product_less_than_30_l3453_345378

def paco_spinner : Finset ℕ := Finset.range 5
def manu_spinner : Finset ℕ := Finset.range 12

def product_less_than_30 (x : ℕ) (y : ℕ) : Bool :=
  x * y < 30

theorem probability_product_less_than_30 :
  (Finset.filter (λ (pair : ℕ × ℕ) => product_less_than_30 (pair.1 + 1) (pair.2 + 1))
    (paco_spinner.product manu_spinner)).card / (paco_spinner.card * manu_spinner.card : ℚ) = 51 / 60 :=
sorry

end NUMINAMATH_CALUDE_probability_product_less_than_30_l3453_345378


namespace NUMINAMATH_CALUDE_class_test_problem_l3453_345320

theorem class_test_problem (first_correct : Real) (second_correct : Real) (both_correct : Real)
  (h1 : first_correct = 0.75)
  (h2 : second_correct = 0.65)
  (h3 : both_correct = 0.60) :
  1 - (first_correct + second_correct - both_correct) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_class_test_problem_l3453_345320


namespace NUMINAMATH_CALUDE_binomial_representation_l3453_345382

theorem binomial_representation (n : ℕ) :
  ∃ x y z : ℕ, n = Nat.choose x 1 + Nat.choose y 2 + Nat.choose z 3 ∧
  ((0 ≤ x ∧ x < y ∧ y < z) ∨ (x = 0 ∧ y = 0 ∧ 0 < z)) :=
sorry

end NUMINAMATH_CALUDE_binomial_representation_l3453_345382


namespace NUMINAMATH_CALUDE_exists_function_45_composition_l3453_345344

def compose_n_times (f : ℝ → ℝ) (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (compose_n_times f n)

theorem exists_function_45_composition :
  ∃ (f : ℝ → ℝ), ∀ (x : ℝ), x ≥ 0 →
    compose_n_times f 45 x = 1 + x + 2 * Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_exists_function_45_composition_l3453_345344


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3453_345330

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  roots_property : a 1 + a 2011 = 10 ∧ a 1 * a 2011 = 16

/-- The sum of specific terms in the arithmetic sequence is 15 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) : 
  seq.a 2 + seq.a 1006 + seq.a 2010 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3453_345330


namespace NUMINAMATH_CALUDE_runner_problem_l3453_345367

/-- Proves that for a 40-mile run where the speed is halved halfway through,
    and the second half takes 12 hours longer than the first half,
    the time to complete the second half is 24 hours. -/
theorem runner_problem (v : ℝ) (h1 : v > 0) : 
  (40 / v = 20 / v + 12) → (40 / (v / 2) = 24) :=
by sorry

end NUMINAMATH_CALUDE_runner_problem_l3453_345367


namespace NUMINAMATH_CALUDE_odd_prime_square_root_l3453_345340

theorem odd_prime_square_root (p : ℕ) (hp : Prime p) (hp_odd : Odd p) :
  ∀ k : ℕ, (∃ m : ℕ, m > 0 ∧ m * m = k * k - p * k) ↔ k = ((p + 1) / 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_square_root_l3453_345340


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3453_345337

-- Define condition p
def condition_p (x y : ℝ) : Prop := x > 2 ∧ y > 3

-- Define condition q
def condition_q (x y : ℝ) : Prop := x + y > 5 ∧ x * y > 6

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, condition_p x y → condition_q x y) ∧
  ¬(∀ x y : ℝ, condition_q x y → condition_p x y) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3453_345337


namespace NUMINAMATH_CALUDE_point_movement_l3453_345389

/-- Given three points A, B, and C on a number line, where:
    - B is 4 units to the right of A
    - C is 2 units to the left of B
    - C represents the number -3
    Prove that A represents the number -5 -/
theorem point_movement (A B C : ℝ) 
  (h1 : B = A + 4)
  (h2 : C = B - 2)
  (h3 : C = -3) :
  A = -5 := by sorry

end NUMINAMATH_CALUDE_point_movement_l3453_345389


namespace NUMINAMATH_CALUDE_students_without_A_l3453_345305

theorem students_without_A (total : ℕ) (chemistry_A : ℕ) (physics_A : ℕ) (both_A : ℕ) 
  (h1 : total = 35)
  (h2 : chemistry_A = 9)
  (h3 : physics_A = 15)
  (h4 : both_A = 5) :
  total - (chemistry_A + physics_A - both_A) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_without_A_l3453_345305


namespace NUMINAMATH_CALUDE_empty_subset_of_set_l3453_345361

theorem empty_subset_of_set : ∅ ⊆ ({2, 0, 1} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_empty_subset_of_set_l3453_345361


namespace NUMINAMATH_CALUDE_fundraiser_full_price_revenue_l3453_345345

/-- Represents the fundraiser event ticket sales -/
structure FundraiserTickets where
  fullPrice : ℕ  -- Number of full-price tickets
  halfPrice : ℕ  -- Number of half-price tickets
  price : ℕ      -- Price of a full-price ticket in dollars

/-- The conditions of the fundraiser ticket sales -/
def fundraiserConditions (t : FundraiserTickets) : Prop :=
  t.fullPrice + t.halfPrice = 200 ∧
  t.fullPrice * t.price + t.halfPrice * (t.price / 2) = 2700

/-- The theorem to prove -/
theorem fundraiser_full_price_revenue :
  ∃ t : FundraiserTickets, fundraiserConditions t ∧ t.fullPrice * t.price = 600 :=
sorry

end NUMINAMATH_CALUDE_fundraiser_full_price_revenue_l3453_345345


namespace NUMINAMATH_CALUDE_bread_rising_times_l3453_345380

/-- Represents the bread-making process with given time constraints --/
def BreadMaking (total_time rising_time kneading_time baking_time : ℕ) :=
  {n : ℕ // n * rising_time + kneading_time + baking_time = total_time}

/-- Theorem stating that Mark lets the bread rise twice --/
theorem bread_rising_times :
  BreadMaking 280 120 10 30 = {n : ℕ // n = 2} :=
by sorry

end NUMINAMATH_CALUDE_bread_rising_times_l3453_345380


namespace NUMINAMATH_CALUDE_log_difference_equals_negative_three_l3453_345353

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_difference_equals_negative_three :
  log10 4 - log10 4000 = -3 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_negative_three_l3453_345353


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l3453_345319

/-- Isabella's hair growth problem -/
theorem isabellas_hair_growth (initial_length : ℝ) : 
  initial_length + 6 = 24 → initial_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l3453_345319


namespace NUMINAMATH_CALUDE_bill_profit_l3453_345324

-- Define the given conditions
def total_milk : ℚ := 16
def butter_ratio : ℚ := 1/4
def sour_cream_ratio : ℚ := 1/4
def milk_to_butter : ℚ := 4
def milk_to_sour_cream : ℚ := 2
def butter_price : ℚ := 5
def sour_cream_price : ℚ := 6
def whole_milk_price : ℚ := 3

-- Define the theorem
theorem bill_profit : 
  let milk_for_butter := total_milk * butter_ratio
  let milk_for_sour_cream := total_milk * sour_cream_ratio
  let butter_gallons := milk_for_butter / milk_to_butter
  let sour_cream_gallons := milk_for_sour_cream / milk_to_sour_cream
  let whole_milk_gallons := total_milk - milk_for_butter - milk_for_sour_cream
  let butter_profit := butter_gallons * butter_price
  let sour_cream_profit := sour_cream_gallons * sour_cream_price
  let whole_milk_profit := whole_milk_gallons * whole_milk_price
  butter_profit + sour_cream_profit + whole_milk_profit = 41 := by
sorry

end NUMINAMATH_CALUDE_bill_profit_l3453_345324


namespace NUMINAMATH_CALUDE_square_equal_area_rectangle_l3453_345392

theorem square_equal_area_rectangle (rectangle_length rectangle_width square_side : ℝ) :
  rectangle_length = 25 ∧ 
  rectangle_width = 9 ∧ 
  square_side = 15 →
  rectangle_length * rectangle_width = square_side * square_side :=
by sorry

end NUMINAMATH_CALUDE_square_equal_area_rectangle_l3453_345392


namespace NUMINAMATH_CALUDE_soda_bottle_difference_l3453_345336

theorem soda_bottle_difference :
  let diet_soda : ℕ := 4
  let regular_soda : ℕ := 83
  regular_soda - diet_soda = 79 :=
by sorry

end NUMINAMATH_CALUDE_soda_bottle_difference_l3453_345336
