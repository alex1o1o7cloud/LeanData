import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l277_27729

theorem rectangle_dimension_change (L B : ℝ) (h₁ : L > 0) (h₂ : B > 0) : 
  let new_B := 1.25 * B
  let new_area := 1.375 * (L * B)
  ∃ x : ℝ, x = 10 ∧ new_area = (L * (1 + x / 100)) * new_B := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l277_27729


namespace NUMINAMATH_CALUDE_mia_sock_purchase_l277_27764

/-- Represents the number of pairs of socks at each price point --/
structure SockPurchase where
  twoDoller : ℕ
  threeDoller : ℕ
  fiveDoller : ℕ

/-- Checks if the given SockPurchase satisfies the problem conditions --/
def isValidPurchase (p : SockPurchase) : Prop :=
  p.twoDoller + p.threeDoller + p.fiveDoller = 15 ∧
  2 * p.twoDoller + 3 * p.threeDoller + 5 * p.fiveDoller = 35 ∧
  p.twoDoller ≥ 1 ∧ p.threeDoller ≥ 1 ∧ p.fiveDoller ≥ 1

theorem mia_sock_purchase :
  ∃ (p : SockPurchase), isValidPurchase p ∧ p.twoDoller = 12 := by
  sorry

end NUMINAMATH_CALUDE_mia_sock_purchase_l277_27764


namespace NUMINAMATH_CALUDE_mean_of_xyz_l277_27736

theorem mean_of_xyz (x y z : ℝ) 
  (eq1 : 9*x + 3*y - 5*z = -4)
  (eq2 : 5*x + 2*y - 2*z = 13) :
  (x + y + z) / 3 = 10 := by
sorry

end NUMINAMATH_CALUDE_mean_of_xyz_l277_27736


namespace NUMINAMATH_CALUDE_reroll_one_die_probability_l277_27792

def dice_sum (d1 d2 d3 : Nat) : Nat := d1 + d2 + d3

def is_valid_die (d : Nat) : Prop := 1 ≤ d ∧ d ≤ 6

def reroll_one_probability : ℚ :=
  let total_outcomes : Nat := 6^3
  let favorable_outcomes : Nat := 19 * 6
  favorable_outcomes / total_outcomes

theorem reroll_one_die_probability :
  ∀ (d1 d2 d3 : Nat),
    is_valid_die d1 → is_valid_die d2 → is_valid_die d3 →
    (∃ (r : Nat), is_valid_die r ∧ dice_sum d1 d2 r = 9 ∨
                  dice_sum d1 r d3 = 9 ∨
                  dice_sum r d2 d3 = 9) →
    reroll_one_probability = 19/216 :=
by sorry

end NUMINAMATH_CALUDE_reroll_one_die_probability_l277_27792


namespace NUMINAMATH_CALUDE_xiao_wang_speed_l277_27754

/-- Represents the cycling speed of Xiao Wang in km/h -/
def cycling_speed : ℝ := 10

/-- The total distance between City A and City B in km -/
def total_distance : ℝ := 55

/-- The distance Xiao Wang cycled in km -/
def cycling_distance : ℝ := 25

/-- The time difference between cycling and bus ride in hours -/
def time_difference : ℝ := 1

theorem xiao_wang_speed :
  cycling_speed = 10 ∧
  cycling_speed > 0 ∧
  total_distance = 55 ∧
  cycling_distance = 25 ∧
  time_difference = 1 ∧
  (cycling_distance / cycling_speed) = 
    ((total_distance - cycling_distance) / (2 * cycling_speed)) + time_difference :=
by sorry

end NUMINAMATH_CALUDE_xiao_wang_speed_l277_27754


namespace NUMINAMATH_CALUDE_ellipse_chord_fixed_point_l277_27784

/-- The fixed point theorem for ellipse chords -/
theorem ellipse_chord_fixed_point 
  (a b A B : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hAB : A ≠ 0 ∧ B ≠ 0) :
  ∃ M : ℝ × ℝ, ∀ P : ℝ × ℝ,
    (A * P.1 + B * P.2 = 1) →  -- P is on line l
    ∃ Q R : ℝ × ℝ,
      (R.1^2 / a^2 + R.2^2 / b^2 = 1) ∧  -- R is on ellipse Γ
      (∃ t : ℝ, Q = ⟨t * P.1, t * P.2⟩) ∧  -- Q is on ray OP
      (Q.1^2 + Q.2^2) * (P.1^2 + P.2^2) = (R.1^2 + R.2^2)^2 →  -- |OQ| * |OP| = |OR|^2
      ∃ l_P : Set (ℝ × ℝ),
        (∀ X ∈ l_P, ∃ s : ℝ, X = ⟨s * Q.1, s * Q.2⟩) ∧  -- l_P is a line through Q
        M ∈ l_P ∧  -- M is on l_P
        M = (A * a^2, B * b^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_fixed_point_l277_27784


namespace NUMINAMATH_CALUDE_wendy_shoes_theorem_l277_27790

/-- The number of pairs of shoes Wendy gave away -/
def shoes_given_away (total : ℕ) (left : ℕ) : ℕ := total - left

/-- Theorem stating that Wendy gave away 14 pairs of shoes -/
theorem wendy_shoes_theorem (total : ℕ) (left : ℕ) 
  (h1 : total = 33) 
  (h2 : left = 19) : 
  shoes_given_away total left = 14 := by
  sorry

end NUMINAMATH_CALUDE_wendy_shoes_theorem_l277_27790


namespace NUMINAMATH_CALUDE_smallest_AAB_existence_AAB_l277_27727

-- Define the structure for our number
structure SpecialNumber where
  A : Nat
  B : Nat
  AB : Nat
  AAB : Nat

-- Define the conditions
def validNumber (n : SpecialNumber) : Prop :=
  1 ≤ n.A ∧ n.A ≤ 9 ∧
  1 ≤ n.B ∧ n.B ≤ 9 ∧
  n.AB = 10 * n.A + n.B ∧
  n.AAB = 110 * n.A + n.B ∧
  n.AB = n.AAB / 8

-- Define the theorem
theorem smallest_AAB :
  ∀ n : SpecialNumber, validNumber n → n.AAB ≥ 221 := by
  sorry

-- Define the existence of such a number
theorem existence_AAB :
  ∃ n : SpecialNumber, validNumber n ∧ n.AAB = 221 := by
  sorry

end NUMINAMATH_CALUDE_smallest_AAB_existence_AAB_l277_27727


namespace NUMINAMATH_CALUDE_bowTie_equation_solution_l277_27712

/-- The infinite nested radical operation -/
noncomputable def bowTie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

/-- Theorem stating that if 5 ⋈ z = 12, then z = 42 -/
theorem bowTie_equation_solution :
  ∃ z : ℝ, bowTie 5 z = 12 → z = 42 := by
  sorry

end NUMINAMATH_CALUDE_bowTie_equation_solution_l277_27712


namespace NUMINAMATH_CALUDE_grape_heap_division_l277_27758

theorem grape_heap_division (n : ℕ) (h1 : n ≥ 105) 
  (h2 : (n + 1) % 3 = 1) (h3 : (n + 1) % 5 = 1) :
  ∃ x : ℕ, x > 5 ∧ (n + 1) % x = 1 ∧ ∀ y : ℕ, 5 < y ∧ y < x → (n + 1) % y ≠ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_grape_heap_division_l277_27758


namespace NUMINAMATH_CALUDE_digit_150_of_5_over_13_l277_27723

def decimal_representation (n d : ℕ) : ℚ := n / d

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem digit_150_of_5_over_13 : 
  nth_digit_after_decimal (decimal_representation 5 13) 150 = 5 := by sorry

end NUMINAMATH_CALUDE_digit_150_of_5_over_13_l277_27723


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l277_27747

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 30 → b = 40 → h^2 = a^2 + b^2 → h = 50 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l277_27747


namespace NUMINAMATH_CALUDE_jose_chickens_l277_27773

/-- Given that Jose has 46 fowls in total and 18 ducks, prove that he has 28 chickens. -/
theorem jose_chickens : 
  ∀ (total_fowls ducks chickens : ℕ), 
    total_fowls = 46 → 
    ducks = 18 → 
    total_fowls = ducks + chickens → 
    chickens = 28 := by
  sorry

end NUMINAMATH_CALUDE_jose_chickens_l277_27773


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l277_27748

/-- Given a geometric sequence with common ratio q ≠ 1, if the ratio of the sum of the first 10 terms
    to the sum of the first 5 terms is 1:2, then the ratio of the sum of the first 15 terms to the
    sum of the first 5 terms is 3:4. -/
theorem geometric_sequence_sum_ratio (q : ℝ) (a : ℕ → ℝ) (h1 : q ≠ 1) 
  (h2 : ∀ n, a (n + 1) = q * a n) 
  (h3 : (1 - q^10) / (1 - q^5) = 1 / 2) :
  (1 - q^15) / (1 - q^5) = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l277_27748


namespace NUMINAMATH_CALUDE_cheryl_tournament_cost_l277_27739

/-- Calculates the total amount Cheryl pays for a golf tournament given her expenses -/
def tournament_cost (electricity_bill : ℕ) (phone_bill_difference : ℕ) (tournament_percentage : ℕ) : ℕ :=
  let phone_bill := electricity_bill + phone_bill_difference
  let tournament_additional_cost := phone_bill * tournament_percentage / 100
  phone_bill + tournament_additional_cost

/-- Proves that Cheryl pays $1440 for the golf tournament given the specified conditions -/
theorem cheryl_tournament_cost :
  tournament_cost 800 400 20 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_tournament_cost_l277_27739


namespace NUMINAMATH_CALUDE_pizza_buffet_l277_27772

theorem pizza_buffet (A B C : ℕ) (h1 : ∃ x : ℕ, A = x * B) 
  (h2 : B * 8 = C) (h3 : A + B + C = 360) : 
  ∃ x : ℕ, A = 351 * B := by
  sorry

end NUMINAMATH_CALUDE_pizza_buffet_l277_27772


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l277_27725

theorem greatest_integer_less_than_negative_seventeen_thirds :
  Int.floor (-17 / 3 : ℚ) = -6 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l277_27725


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_is_zero_l277_27708

/-- Given q(x) = x⁴ - 4x + 5, the coefficient of x³ in (q(x))² is 0 -/
theorem coefficient_x_cubed_is_zero (x : ℝ) : 
  let q := fun (x : ℝ) => x^4 - 4*x + 5
  (q x)^2 = x^8 - 8*x^5 + 10*x^4 + 16*x^2 - 40*x + 25 :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_is_zero_l277_27708


namespace NUMINAMATH_CALUDE_garage_sale_games_l277_27757

/-- The number of games Luke bought from a friend -/
def games_from_friend : ℕ := 2

/-- The number of games that didn't work -/
def broken_games : ℕ := 2

/-- The number of good games Luke ended up with -/
def good_games : ℕ := 2

/-- The number of games Luke bought at the garage sale -/
def games_from_garage_sale : ℕ := 2

theorem garage_sale_games :
  games_from_friend + games_from_garage_sale - broken_games = good_games :=
by sorry

end NUMINAMATH_CALUDE_garage_sale_games_l277_27757


namespace NUMINAMATH_CALUDE_problem_statement_l277_27710

theorem problem_statement : (Real.sqrt 5 + 2)^2 + (-1/2)⁻¹ - Real.sqrt 49 = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l277_27710


namespace NUMINAMATH_CALUDE_least_product_of_reciprocal_sum_l277_27701

theorem least_product_of_reciprocal_sum (a b : ℕ+) 
  (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (9 : ℚ)⁻¹) : 
  (∀ c d : ℕ+, (c : ℚ)⁻¹ + (3 * d : ℚ)⁻¹ = (9 : ℚ)⁻¹ → (a * b : ℕ) ≤ (c * d : ℕ)) ∧ 
  (a * b : ℕ) = 144 := by
sorry

end NUMINAMATH_CALUDE_least_product_of_reciprocal_sum_l277_27701


namespace NUMINAMATH_CALUDE_min_value_of_expression_l277_27713

theorem min_value_of_expression (a b : ℝ) (ha : a > 1) (hab : a * b = 2 * a + b) :
  (∀ x y : ℝ, x > 1 ∧ x * y = 2 * x + y → (a + 1) * (b + 2) ≤ (x + 1) * (y + 2)) ∧
  (a + 1) * (b + 2) = 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l277_27713


namespace NUMINAMATH_CALUDE_color_change_probability_l277_27756

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  green_duration : ℕ
  yellow_duration : ℕ
  red_duration : ℕ

/-- Calculates the total duration of a traffic light cycle -/
def cycle_duration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green_duration + cycle.yellow_duration + cycle.red_duration

/-- Calculates the number of seconds where a color change can be observed in a 4-second interval -/
def change_observation_duration (cycle : TrafficLightCycle) : ℕ := 12

/-- Theorem: The probability of observing a color change during a random 4-second interval
    in the given traffic light cycle is 0.12 -/
theorem color_change_probability (cycle : TrafficLightCycle) 
  (h1 : cycle.green_duration = 45)
  (h2 : cycle.yellow_duration = 5)
  (h3 : cycle.red_duration = 50)
  (h4 : change_observation_duration cycle = 12) :
  (change_observation_duration cycle : ℚ) / (cycle_duration cycle) = 12 / 100 := by
  sorry

end NUMINAMATH_CALUDE_color_change_probability_l277_27756


namespace NUMINAMATH_CALUDE_min_max_sum_l277_27788

theorem min_max_sum (a b c d e f : ℕ+) 
  (sum_eq : a + b + c + d + e + f = 4020) : 
  804 ≤ max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f)))) := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_l277_27788


namespace NUMINAMATH_CALUDE_fish_rice_trade_l277_27799

/-- Represents the value of one fish in terms of bags of rice -/
def fish_value (fish bread apple rice : ℚ) : Prop :=
  (5 * fish = 3 * bread) ∧
  (bread = 6 * apple) ∧
  (2 * apple = rice) →
  fish = 9/5 * rice

theorem fish_rice_trade : ∀ (fish bread apple rice : ℚ),
  fish_value fish bread apple rice :=
by
  sorry

end NUMINAMATH_CALUDE_fish_rice_trade_l277_27799


namespace NUMINAMATH_CALUDE_geometry_theorem_l277_27786

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (intersects : Line → Plane → Prop)
variable (distinct : Line → Line → Prop)
variable (distinctP : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- State the theorem
theorem geometry_theorem 
  (h_distinct_lines : distinct m n)
  (h_distinct_planes : distinctP α β) :
  (perpendicularPP α β ∧ perpendicularLP m α → ¬(intersects m β)) ∧
  (perpendicular m n ∧ perpendicularLP m α → ¬(intersects n α)) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l277_27786


namespace NUMINAMATH_CALUDE_solution_set_f_derivative_positive_l277_27769

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem solution_set_f_derivative_positive :
  {x : ℝ | (deriv f) x > 0} = {x : ℝ | x < -2 ∨ x > 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_derivative_positive_l277_27769


namespace NUMINAMATH_CALUDE_condition_not_well_defined_l277_27795

-- Define a type for students
structure Student :=
  (height : ℝ)
  (school : String)

-- Define a type for conditions
inductive Condition
  | TallStudents : Condition
  | PointsAwayFromOrigin : Condition
  | PrimesLessThan100 : Condition
  | QuadraticEquationSolutions : Condition

-- Define a predicate for well-defined sets
def IsWellDefinedSet (c : Condition) : Prop :=
  match c with
  | Condition.TallStudents => false
  | Condition.PointsAwayFromOrigin => true
  | Condition.PrimesLessThan100 => true
  | Condition.QuadraticEquationSolutions => true

-- Theorem statement
theorem condition_not_well_defined :
  ∃ c : Condition, ¬(IsWellDefinedSet c) ∧
  ∀ c' : Condition, c' ≠ c → IsWellDefinedSet c' :=
sorry

end NUMINAMATH_CALUDE_condition_not_well_defined_l277_27795


namespace NUMINAMATH_CALUDE_distance_walked_proof_l277_27722

/-- Calculates the distance walked by a person given their step length, steps per minute, and duration of walk. -/
def distanceWalked (stepLength : ℝ) (stepsPerMinute : ℝ) (durationMinutes : ℝ) : ℝ :=
  stepLength * stepsPerMinute * durationMinutes

/-- Proves that a person walking 0.75 meters per step at 70 steps per minute for 13 minutes covers 682.5 meters. -/
theorem distance_walked_proof :
  distanceWalked 0.75 70 13 = 682.5 := by
  sorry

#eval distanceWalked 0.75 70 13

end NUMINAMATH_CALUDE_distance_walked_proof_l277_27722


namespace NUMINAMATH_CALUDE_cost_price_calculation_l277_27704

theorem cost_price_calculation (cost_price : ℝ) : 
  cost_price * (1 + 0.2) * 0.9 - cost_price = 8 → cost_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l277_27704


namespace NUMINAMATH_CALUDE_committee_selection_ways_l277_27779

-- Define the total number of members
def total_members : ℕ := 30

-- Define the number of ineligible members
def ineligible_members : ℕ := 3

-- Define the size of the committee
def committee_size : ℕ := 5

-- Define the number of eligible members
def eligible_members : ℕ := total_members - ineligible_members

-- Theorem statement
theorem committee_selection_ways :
  Nat.choose eligible_members committee_size = 80730 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l277_27779


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l277_27797

/-- If |a-4|+(b+3)^2=0, then a > 0 and b < 0 -/
theorem point_in_fourth_quadrant (a b : ℝ) (h : |a - 4| + (b + 3)^2 = 0) : a > 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l277_27797


namespace NUMINAMATH_CALUDE_equation_solution_l277_27744

theorem equation_solution (C D : ℝ) : 
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ 5 → (C * x - 20) / (x^2 - 3*x - 10) = D / (x + 2) + 4 / (x - 5)) →
  C + D = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l277_27744


namespace NUMINAMATH_CALUDE_investment_sum_proof_l277_27733

/-- Proves that a sum invested at 15% p.a. simple interest for two years yields
    Rs. 420 more interest than if invested at 12% p.a. for the same period,
    then the sum is Rs. 7000. -/
theorem investment_sum_proof (P : ℚ) 
  (h1 : P * (15 / 100) * 2 - P * (12 / 100) * 2 = 420) : P = 7000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_proof_l277_27733


namespace NUMINAMATH_CALUDE_count_distinct_terms_l277_27735

/-- The number of distinct terms in the expansion of (x+y+z)^2026 + (x-y-z)^2026 -/
def num_distinct_terms : ℕ := 1028196

/-- The exponent in the original expression -/
def exponent : ℕ := 2026

-- Theorem stating the number of distinct terms
theorem count_distinct_terms : 
  num_distinct_terms = (exponent / 2 + 1)^2 := by sorry

end NUMINAMATH_CALUDE_count_distinct_terms_l277_27735


namespace NUMINAMATH_CALUDE_transport_cost_tripled_bags_reduced_weight_l277_27702

/-- The cost of transporting cement bags -/
def transport_cost (bags : ℕ) (weight : ℚ) : ℚ :=
  (6000 : ℚ) * bags * weight / (80 * 50)

/-- Theorem: The cost of transporting 240 bags weighing 30 kgs each is $10800 -/
theorem transport_cost_tripled_bags_reduced_weight :
  transport_cost 240 30 = 10800 := by
  sorry

end NUMINAMATH_CALUDE_transport_cost_tripled_bags_reduced_weight_l277_27702


namespace NUMINAMATH_CALUDE_range_of_a_l277_27777

-- Define the function f
def f (x : ℝ) : ℝ := 3*x + 2*x^3

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo (-2 : ℝ) 2, f x = 3*x + 2*x^3) →
  (f (a - 1) + f (1 - 2*a) < 0) →
  a ∈ Set.Ioo 0 (3/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l277_27777


namespace NUMINAMATH_CALUDE_beth_crayons_l277_27798

theorem beth_crayons (initial : ℕ) (given_away : ℕ) (left : ℕ) : 
  given_away = 54 → left = 52 → initial = given_away + left → initial = 106 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_l277_27798


namespace NUMINAMATH_CALUDE_stamps_problem_l277_27752

theorem stamps_problem (cj kj aj : ℕ) : 
  cj = 2 * kj + 5 →  -- CJ has 5 more than twice the number of stamps that KJ has
  kj * 2 = aj →      -- KJ has half as many stamps as AJ
  cj + kj + aj = 930 →  -- The three boys have 930 stamps in total
  aj = 370 :=         -- Prove that AJ has 370 stamps
by
  sorry


end NUMINAMATH_CALUDE_stamps_problem_l277_27752


namespace NUMINAMATH_CALUDE_triangle_inequality_l277_27794

theorem triangle_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l277_27794


namespace NUMINAMATH_CALUDE_max_snack_bags_l277_27740

def granola_bars : ℕ := 24
def dried_fruit : ℕ := 36
def nuts : ℕ := 60

theorem max_snack_bags : 
  ∃ (n : ℕ), n > 0 ∧ 
  granola_bars % n = 0 ∧ 
  dried_fruit % n = 0 ∧ 
  nuts % n = 0 ∧
  ∀ (m : ℕ), m > n → 
    (granola_bars % m = 0 ∧ dried_fruit % m = 0 ∧ nuts % m = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_max_snack_bags_l277_27740


namespace NUMINAMATH_CALUDE_three_speakers_from_different_companies_l277_27774

/-- The number of companies in the meeting -/
def total_companies : ℕ := 5

/-- The number of representatives from Company A -/
def company_a_reps : ℕ := 2

/-- The number of representatives from each of the other companies -/
def other_company_reps : ℕ := 1

/-- The number of speakers at the meeting -/
def num_speakers : ℕ := 3

/-- The number of scenarios where 3 speakers come from 3 different companies -/
def num_scenarios : ℕ := 16

theorem three_speakers_from_different_companies :
  (Nat.choose company_a_reps 1 * Nat.choose (total_companies - 1) 2) +
  (Nat.choose (total_companies - 1) 3) = num_scenarios :=
sorry

end NUMINAMATH_CALUDE_three_speakers_from_different_companies_l277_27774


namespace NUMINAMATH_CALUDE_problem_one_l277_27793

theorem problem_one : (1 / 3)⁻¹ + Real.sqrt 12 - |Real.sqrt 3 - 2| - (π - 2023)^0 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_l277_27793


namespace NUMINAMATH_CALUDE_binomial_seven_four_l277_27731

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_four_l277_27731


namespace NUMINAMATH_CALUDE_value_of_x_l277_27768

theorem value_of_x (x y z : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 4) * z → 
  z = 96 → 
  x = 8 := by sorry

end NUMINAMATH_CALUDE_value_of_x_l277_27768


namespace NUMINAMATH_CALUDE_circle_tangent_and_center_range_l277_27778

-- Define the given points and lines
def A : ℝ × ℝ := (0, 3)
def l (x : ℝ) : ℝ := 2 * x - 4

-- Define the circle C
def C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 1}

-- Define the condition for the center of C
def center_condition (center : ℝ × ℝ) : Prop :=
  center.2 = l center.1 ∧ center.2 = center.1 - 1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  3 * x + 4 * y - 12 = 0

-- Define the condition for point M
def M_condition (center : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  M ∈ C center ∧ (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4 * ((M.1 - center.1)^2 + (M.2 - center.2)^2)

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 12/5

-- State the theorem
theorem circle_tangent_and_center_range :
  ∃ (center : ℝ × ℝ),
    center_condition center ∧
    (∀ (x y : ℝ), (x, y) ∈ C center → tangent_line x y) ∧
    (∃ (M : ℝ × ℝ), M_condition center M) ∧
    a_range center.1 := by sorry

end NUMINAMATH_CALUDE_circle_tangent_and_center_range_l277_27778


namespace NUMINAMATH_CALUDE_ellipse_equation_from_properties_l277_27714

/-- An ellipse with specified properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  min_distance_to_focus : ℝ
  eccentricity : ℝ

/-- The equation of an ellipse given its properties -/
def ellipse_equation (E : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 8 + y^2 / 4 = 1

/-- Theorem stating that an ellipse with the given properties has the specified equation -/
theorem ellipse_equation_from_properties (E : Ellipse) 
  (h1 : E.center = (0, 0))
  (h2 : E.foci_on_x_axis = true)
  (h3 : E.min_distance_to_focus = 2 * Real.sqrt 2 - 2)
  (h4 : E.eccentricity = Real.sqrt 2 / 2) :
  ellipse_equation E = fun x y => x^2 / 8 + y^2 / 4 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_properties_l277_27714


namespace NUMINAMATH_CALUDE_average_weight_calculation_l277_27782

theorem average_weight_calculation (num_men num_women : ℕ) (avg_weight_men avg_weight_women : ℚ) :
  num_men = 8 →
  num_women = 6 →
  avg_weight_men = 170 →
  avg_weight_women = 130 →
  let total_weight := num_men * avg_weight_men + num_women * avg_weight_women
  let total_people := num_men + num_women
  abs ((total_weight / total_people) - 153) < 1 := by
sorry

end NUMINAMATH_CALUDE_average_weight_calculation_l277_27782


namespace NUMINAMATH_CALUDE_total_value_of_coins_l277_27709

/-- Represents the value of a coin in paise -/
inductive CoinType
| OneRupee
| FiftyPaise
| TwentyFivePaise

/-- The number of coins of each type in the bag -/
def coinsPerType : ℕ := 120

/-- Converts a coin type to its value in paise -/
def coinValueInPaise (c : CoinType) : ℕ :=
  match c with
  | CoinType.OneRupee => 100
  | CoinType.FiftyPaise => 50
  | CoinType.TwentyFivePaise => 25

/-- Calculates the total value of all coins of a given type in rupees -/
def totalValueOfCoinType (c : CoinType) : ℚ :=
  (coinsPerType * coinValueInPaise c : ℚ) / 100

/-- Theorem: The total value of all coins in the bag is 210 rupees -/
theorem total_value_of_coins :
  totalValueOfCoinType CoinType.OneRupee +
  totalValueOfCoinType CoinType.FiftyPaise +
  totalValueOfCoinType CoinType.TwentyFivePaise = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_value_of_coins_l277_27709


namespace NUMINAMATH_CALUDE_complement_union_A_B_l277_27761

open Set

def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_A_B :
  (A ∪ B)ᶜ = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l277_27761


namespace NUMINAMATH_CALUDE_janet_earnings_l277_27730

/-- Calculates the hourly earnings for checking social media posts. -/
def hourly_earnings (pay_per_post : ℚ) (seconds_per_post : ℕ) : ℚ :=
  let posts_per_hour : ℕ := 3600 / seconds_per_post
  pay_per_post * posts_per_hour

/-- Proves that given a pay rate of 25 cents per post and a checking time of 10 seconds per post, the hourly earnings are $90. -/
theorem janet_earnings : hourly_earnings (25 / 100) 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_janet_earnings_l277_27730


namespace NUMINAMATH_CALUDE_arithmetic_sequence_angles_l277_27781

theorem arithmetic_sequence_angles (angles : Fin 5 → ℝ) : 
  (∀ i j : Fin 5, i < j → angles i < angles j) →  -- angles are strictly increasing
  (∀ i : Fin 4, angles (i + 1) - angles i = angles (i + 2) - angles (i + 1)) →  -- arithmetic sequence
  angles 0 = 25 →  -- smallest angle
  angles 4 = 105 →  -- largest angle
  ∀ i : Fin 4, angles (i + 1) - angles i = 20 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_angles_l277_27781


namespace NUMINAMATH_CALUDE_measure_when_unit_changed_l277_27759

-- Define segments a and b as positive real numbers
variable (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

-- Define m as the measure of a when b is the unit length
variable (m : ℝ) (hm : a = m * b)

-- Theorem statement
theorem measure_when_unit_changed : 
  (b / a : ℝ) = 1 / m :=
sorry

end NUMINAMATH_CALUDE_measure_when_unit_changed_l277_27759


namespace NUMINAMATH_CALUDE_expected_ones_three_dice_l277_27776

-- Define a standard die
def standardDie : Finset Nat := Finset.range 6

-- Define the probability of rolling a 1 on a single die
def probOne : ℚ := 1 / 6

-- Define the probability of not rolling a 1 on a single die
def probNotOne : ℚ := 1 - probOne

-- Define the number of dice
def numDice : Nat := 3

-- Define the expected value function for discrete random variables
def expectedValue (outcomes : Finset Nat) (prob : Nat → ℚ) : ℚ :=
  Finset.sum outcomes (λ x => x * prob x)

-- Statement of the theorem
theorem expected_ones_three_dice :
  expectedValue (Finset.range (numDice + 1)) (λ k =>
    (numDice.choose k : ℚ) * probOne ^ k * probNotOne ^ (numDice - k)) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_expected_ones_three_dice_l277_27776


namespace NUMINAMATH_CALUDE_reinforcement_arrival_time_l277_27765

/-- Calculates the number of days passed before reinforcement arrived -/
def days_before_reinforcement (initial_garrison : ℕ) (initial_provisions : ℕ) 
  (reinforcement : ℕ) (remaining_provisions : ℕ) : ℕ :=
  (initial_garrison * initial_provisions - (initial_garrison + reinforcement) * remaining_provisions) / initial_garrison

/-- Theorem stating that 15 days passed before reinforcement arrived -/
theorem reinforcement_arrival_time :
  days_before_reinforcement 2000 65 3000 20 = 15 := by sorry

end NUMINAMATH_CALUDE_reinforcement_arrival_time_l277_27765


namespace NUMINAMATH_CALUDE_farmers_wheat_cleaning_l277_27746

theorem farmers_wheat_cleaning (original_rate : ℕ) (new_rate : ℕ) (last_day_acres : ℕ) :
  original_rate = 80 →
  new_rate = original_rate + 10 →
  last_day_acres = 30 →
  ∃ (total_acres : ℕ) (planned_days : ℕ),
    total_acres = 480 ∧
    planned_days * original_rate = total_acres ∧
    (planned_days - 1) * new_rate + last_day_acres = total_acres :=
by
  sorry

end NUMINAMATH_CALUDE_farmers_wheat_cleaning_l277_27746


namespace NUMINAMATH_CALUDE_work_completion_time_l277_27700

/-- Given workers a, b, and c with their work rates, prove the time taken when all work together -/
theorem work_completion_time
  (total_work : ℝ)
  (time_ab : ℝ)
  (time_a : ℝ)
  (time_c : ℝ)
  (h1 : time_ab = 9)
  (h2 : time_a = 18)
  (h3 : time_c = 24)
  : (total_work / (total_work / time_ab + total_work / time_a + total_work / time_c)) = 72 / 11 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l277_27700


namespace NUMINAMATH_CALUDE_max_min_difference_z_l277_27787

theorem max_min_difference_z (x y z : ℝ) 
  (sum_eq : x + y + z = 3) 
  (sum_squares_eq : x^2 + y^2 + z^2 = 18) : 
  ∃ (z_max z_min : ℝ), 
    (∀ w : ℝ, (∃ u v : ℝ, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 18) → w ≤ z_max) ∧
    (∀ w : ℝ, (∃ u v : ℝ, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 18) → w ≥ z_min) ∧
    z_max - z_min = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l277_27787


namespace NUMINAMATH_CALUDE_fran_required_speed_l277_27732

/-- Given Joann's bike ride parameters and Fran's ride time, calculate Fran's required speed -/
theorem fran_required_speed 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4) 
  (h3 : fran_time = 2.5) : 
  (joann_speed * joann_time) / fran_time = 24 := by
sorry

end NUMINAMATH_CALUDE_fran_required_speed_l277_27732


namespace NUMINAMATH_CALUDE_sharp_composition_l277_27737

def sharp (N : ℝ) : ℝ := 0.4 * N * 1.5

theorem sharp_composition : sharp (sharp (sharp 80)) = 17.28 := by
  sorry

end NUMINAMATH_CALUDE_sharp_composition_l277_27737


namespace NUMINAMATH_CALUDE_circle_area_below_line_l277_27751

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 - 14*y + 33 = 0

-- Define the line equation
def line_equation (y : ℝ) : Prop :=
  y = 7

-- Theorem statement
theorem circle_area_below_line :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_y = 7 ∧
    radius > 0 ∧
    (π * radius^2 / 2 : ℝ) = 25 * π / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_area_below_line_l277_27751


namespace NUMINAMATH_CALUDE_absolute_difference_of_points_on_curve_l277_27726

theorem absolute_difference_of_points_on_curve (e p q : ℝ) : 
  (p^2 + e^4 = 4 * e^2 * p + 6) →
  (q^2 + e^4 = 4 * e^2 * q + 6) →
  |p - q| = 2 * Real.sqrt (3 * e^4 + 6) := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_points_on_curve_l277_27726


namespace NUMINAMATH_CALUDE_sum_infinite_geometric_series_l277_27742

def geometric_series (a : ℝ) (r : ℝ) := 
  fun n : ℕ => a * r ^ n

theorem sum_infinite_geometric_series :
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let series := geometric_series a r
  (∑' n, series n) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sum_infinite_geometric_series_l277_27742


namespace NUMINAMATH_CALUDE_divisibility_by_nine_l277_27743

theorem divisibility_by_nine : ∃ d : ℕ, d < 10 ∧ (2345 * 10 + d) % 9 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_l277_27743


namespace NUMINAMATH_CALUDE_platform_length_l277_27724

/-- Given a train of length 300 meters that crosses a platform in 33 seconds
    and a signal pole in 18 seconds, the length of the platform is 250 meters. -/
theorem platform_length
  (train_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 33)
  (h3 : pole_crossing_time = 18) :
  (train_length + platform_crossing_time * (train_length / pole_crossing_time) - train_length) = 250 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l277_27724


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l277_27750

theorem least_k_for_inequality (k : ℤ) : 
  (∀ j : ℤ, j < k → (0.00010101 : ℝ) * (10 : ℝ)^j ≤ 1000) ∧ 
  ((0.00010101 : ℝ) * (10 : ℝ)^k > 1000) → 
  k = 8 :=
sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l277_27750


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l277_27716

theorem lcm_factor_proof (A B : ℕ+) (X : ℕ) : 
  Nat.gcd A B = 23 →
  Nat.lcm A B = 23 * 15 * X →
  A = 368 →
  X = 16 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l277_27716


namespace NUMINAMATH_CALUDE_smaller_number_problem_l277_27760

theorem smaller_number_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 45) (h4 : y = 4 * x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l277_27760


namespace NUMINAMATH_CALUDE_will_toy_purchase_l277_27718

theorem will_toy_purchase (initial_amount : ℕ) (spent_amount : ℕ) (toy_cost : ℕ) : 
  initial_amount = 83 → spent_amount = 47 → toy_cost = 4 →
  (initial_amount - spent_amount) / toy_cost = 9 := by
sorry

end NUMINAMATH_CALUDE_will_toy_purchase_l277_27718


namespace NUMINAMATH_CALUDE_candy_bar_profit_l277_27717

/-- Represents the candy bar sale problem -/
structure CandyBarSale where
  total_bars : ℕ
  bulk_price : ℚ
  bulk_quantity : ℕ
  regular_price : ℚ
  regular_quantity : ℕ
  selling_price : ℚ
  selling_quantity : ℕ

/-- Calculates the profit for the candy bar sale -/
def calculate_profit (sale : CandyBarSale) : ℚ :=
  let cost_per_bar := sale.bulk_price / sale.bulk_quantity
  let total_cost := cost_per_bar * sale.total_bars
  let revenue_per_bar := sale.selling_price / sale.selling_quantity
  let total_revenue := revenue_per_bar * sale.total_bars
  total_revenue - total_cost

/-- The main theorem stating that the profit is $350 -/
theorem candy_bar_profit :
  let sale : CandyBarSale := {
    total_bars := 1200,
    bulk_price := 3,
    bulk_quantity := 8,
    regular_price := 2,
    regular_quantity := 5,
    selling_price := 2,
    selling_quantity := 3
  }
  calculate_profit sale = 350 := by
  sorry


end NUMINAMATH_CALUDE_candy_bar_profit_l277_27717


namespace NUMINAMATH_CALUDE_expression_evaluation_l277_27783

theorem expression_evaluation : (5 + 2 + 6) * 2 / 3 - 4 / 3 = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l277_27783


namespace NUMINAMATH_CALUDE_probability_three_defective_before_two_good_l277_27796

/-- Represents the number of good products in the box -/
def goodProducts : ℕ := 9

/-- Represents the number of defective products in the box -/
def defectiveProducts : ℕ := 3

/-- Represents the total number of products in the box -/
def totalProducts : ℕ := goodProducts + defectiveProducts

/-- Calculates the probability of selecting 3 defective products before 2 good products -/
def probabilityThreeDefectiveBeforeTwoGood : ℚ :=
  (4 : ℚ) / 55

/-- Theorem stating that the probability of selecting 3 defective products
    before 2 good products is 4/55 -/
theorem probability_three_defective_before_two_good :
  probabilityThreeDefectiveBeforeTwoGood = (4 : ℚ) / 55 := by
  sorry

#eval probabilityThreeDefectiveBeforeTwoGood

end NUMINAMATH_CALUDE_probability_three_defective_before_two_good_l277_27796


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_area_l277_27770

/-- Given two similar triangles with corresponding median lengths and sum of perimeters,
    prove their individual perimeters and area ratio -/
theorem similar_triangles_perimeter_area (median1 median2 perimeter_sum : ℝ) :
  median1 = 10 →
  median2 = 4 →
  perimeter_sum = 140 →
  ∃ (perimeter1 perimeter2 area1 area2 : ℝ),
    perimeter1 = 100 ∧
    perimeter2 = 40 ∧
    perimeter1 + perimeter2 = perimeter_sum ∧
    (area1 / area2 = 25 / 4) ∧
    (median1 / median2)^2 = area1 / area2 ∧
    median1 / median2 = perimeter1 / perimeter2 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_area_l277_27770


namespace NUMINAMATH_CALUDE_unique_positive_number_l277_27763

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l277_27763


namespace NUMINAMATH_CALUDE_inequality_proof_l277_27711

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x + y + z)^2 / 3 ≥ x * Real.sqrt (y * z) + y * Real.sqrt (z * x) + z * Real.sqrt (x * y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l277_27711


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l277_27707

theorem correct_quotient_proof (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 49) : D / 21 = 28 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l277_27707


namespace NUMINAMATH_CALUDE_child_ticket_cost_l277_27738

/-- Proves that the cost of a child ticket is 1 dollar given the conditions of the problem -/
theorem child_ticket_cost
  (adult_ticket_cost : ℕ)
  (total_attendees : ℕ)
  (total_revenue : ℕ)
  (child_attendees : ℕ)
  (h1 : adult_ticket_cost = 8)
  (h2 : total_attendees = 22)
  (h3 : total_revenue = 50)
  (h4 : child_attendees = 18) :
  let adult_attendees : ℕ := total_attendees - child_attendees
  let child_ticket_cost : ℚ := (total_revenue - adult_ticket_cost * adult_attendees) / child_attendees
  child_ticket_cost = 1 := by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l277_27738


namespace NUMINAMATH_CALUDE_inequalities_always_satisfied_l277_27734

theorem inequalities_always_satisfied (a b c x y z : ℝ) 
  (hx : x < a) (hy : y < b) (hz : z < c) : 
  (x * y * c < a * b * z) ∧ 
  (x^2 + c < a^2 + z) ∧ 
  (x^2 * y^2 * z^2 < a^2 * b^2 * c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_always_satisfied_l277_27734


namespace NUMINAMATH_CALUDE_emma_coins_l277_27791

theorem emma_coins (x : ℚ) (hx : x > 0) : 
  let lost := x / 3
  let found := (3 / 4) * lost
  x - (x - lost + found) = x / 12 := by sorry

end NUMINAMATH_CALUDE_emma_coins_l277_27791


namespace NUMINAMATH_CALUDE_CO2_yield_calculation_l277_27703

-- Define the chemical equation
def chemical_equation : String := "HCl + NaHCO3 → NaCl + H2O + CO2"

-- Define the molar quantities of reactants
def moles_HCl : ℝ := 1
def moles_NaHCO3 : ℝ := 1

-- Define the molar mass of CO2
def molar_mass_CO2 : ℝ := 44.01

-- Define the theoretical yield function
def theoretical_yield (moles_reactant : ℝ) (molar_mass_product : ℝ) : ℝ :=
  moles_reactant * molar_mass_product

-- Theorem statement
theorem CO2_yield_calculation :
  theoretical_yield (min moles_HCl moles_NaHCO3) molar_mass_CO2 = 44.01 := by
  sorry


end NUMINAMATH_CALUDE_CO2_yield_calculation_l277_27703


namespace NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l277_27745

/-- Calculates the length of a train given its speed, the time it takes to pass a bridge, and the length of the bridge. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (time_to_pass : Real) : Real :=
  let speed_ms := train_speed * 1000 / 3600
  let total_distance := speed_ms * time_to_pass
  total_distance - bridge_length

/-- Proves that a train traveling at 45 km/hour passing a 140-meter bridge in 42 seconds has a length of 385 meters. -/
theorem train_length_proof :
  train_length_calculation 45 140 42 = 385 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l277_27745


namespace NUMINAMATH_CALUDE_intersection_sum_l277_27775

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 2
def g (x y : ℝ) : Prop := x + 2*y = 2

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
  f x₁ = y₁ ∧ g x₁ y₁ ∧
  f x₂ = y₂ ∧ g x₂ y₂ ∧
  f x₃ = y₃ ∧ g x₃ y₃

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    f x₁ = y₁ ∧ g x₁ y₁ ∧
    f x₂ = y₂ ∧ g x₂ y₂ ∧
    f x₃ = y₃ ∧ g x₃ y₃ ∧
    x₁ + x₂ + x₃ = 4 ∧
    y₁ + y₂ + y₃ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l277_27775


namespace NUMINAMATH_CALUDE_willies_stickers_l277_27789

/-- The final sticker count for Willie -/
def final_sticker_count (initial_count : ℝ) (received_count : ℝ) : ℝ :=
  initial_count + received_count

/-- Theorem stating that Willie's final sticker count is the sum of his initial count and received stickers -/
theorem willies_stickers (initial_count received_count : ℝ) :
  final_sticker_count initial_count received_count = initial_count + received_count :=
by sorry

end NUMINAMATH_CALUDE_willies_stickers_l277_27789


namespace NUMINAMATH_CALUDE_correct_operation_l277_27719

theorem correct_operation (a : ℝ) : 3 * a^3 - 2 * a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l277_27719


namespace NUMINAMATH_CALUDE_a_range_l277_27715

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem a_range (a : ℝ) : 
  (∀ x : ℝ, otimes (x - a) (x + 1) < 1) → -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l277_27715


namespace NUMINAMATH_CALUDE_probability_both_selected_l277_27741

theorem probability_both_selected (prob_X prob_Y : ℚ) 
  (h1 : prob_X = 1/3) (h2 : prob_Y = 2/5) : 
  prob_X * prob_Y = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l277_27741


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l277_27705

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation of a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the equation of a parabola in the form ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
structure ParabolaEq where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The focus of the parabola -/
def focus : Point := ⟨5, 2⟩

/-- The directrix of the parabola -/
def directrix : Line := ⟨5, 2, 25⟩

/-- The equation of the parabola -/
def parabolaEq : ParabolaEq := ⟨4, -20, 25, -40, -16, -509⟩

/-- Checks if the given parabola equation satisfies the conditions -/
def isValidParabolaEq (eq : ParabolaEq) : Prop :=
  eq.a > 0 ∧ Int.gcd eq.a.natAbs (Int.gcd eq.b.natAbs (Int.gcd eq.c.natAbs (Int.gcd eq.d.natAbs (Int.gcd eq.e.natAbs eq.f.natAbs)))) = 1

/-- Theorem stating that the given parabola equation is correct and satisfies the conditions -/
theorem parabola_equation_correct :
  isValidParabolaEq parabolaEq ∧
  ∀ (p : Point),
    (p.x - focus.x)^2 + (p.y - focus.y)^2 = 
    ((directrix.a * p.x + directrix.b * p.y - directrix.c)^2) / (directrix.a^2 + directrix.b^2) ↔
    parabolaEq.a * p.x^2 + parabolaEq.b * p.x * p.y + parabolaEq.c * p.y^2 + 
    parabolaEq.d * p.x + parabolaEq.e * p.y + parabolaEq.f = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l277_27705


namespace NUMINAMATH_CALUDE_fuel_mixture_cost_l277_27720

/-- Represents the cost of the other liquid per gallon -/
def other_liquid_cost : ℝ := 3

/-- The total volume of the mixture in gallons -/
def total_volume : ℝ := 12

/-- The cost of the final fuel mixture per gallon -/
def final_fuel_cost : ℝ := 8

/-- The cost of oil per gallon -/
def oil_cost : ℝ := 15

/-- The volume of one of the liquids used in the mixture -/
def one_liquid_volume : ℝ := 7

theorem fuel_mixture_cost : 
  one_liquid_volume * other_liquid_cost + (total_volume - one_liquid_volume) * oil_cost = 
  total_volume * final_fuel_cost :=
sorry

end NUMINAMATH_CALUDE_fuel_mixture_cost_l277_27720


namespace NUMINAMATH_CALUDE_min_value_expression_l277_27785

theorem min_value_expression (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a < 1) (hb : 0 ≤ b ∧ b < 1) (hc : 0 ≤ c ∧ c < 1) : 
  (1 / ((2 - a) * (2 - b) * (2 - c))) + (1 / ((2 + a) * (2 + b) * (2 + c))) ≥ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l277_27785


namespace NUMINAMATH_CALUDE_f_derivative_and_tangent_line_l277_27749

noncomputable def f (x : ℝ) : ℝ := Real.sin x / x

theorem f_derivative_and_tangent_line :
  (∃ (f' : ℝ → ℝ), ∀ x, x ≠ 0 → HasDerivAt f (f' x) x) ∧
  (∀ x, x ≠ 0 → (deriv f) x = (x * Real.cos x - Real.sin x) / x^2) ∧
  (HasDerivAt f (-1/π) π) ∧
  (∀ x, -x/π + 1 = (-1/π) * (x - π)) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_and_tangent_line_l277_27749


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l277_27767

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l277_27767


namespace NUMINAMATH_CALUDE_single_elimination_games_tournament_with_23_teams_l277_27753

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  num_teams : ℕ
  num_games : ℕ

/-- The number of games in a single-elimination tournament is one less than the number of teams. -/
theorem single_elimination_games (t : SingleEliminationTournament) 
  (h : t.num_teams > 0) : t.num_games = t.num_teams - 1 := by
  sorry

/-- In a single-elimination tournament with 23 teams, 22 games are required to declare a winner. -/
theorem tournament_with_23_teams : 
  ∃ t : SingleEliminationTournament, t.num_teams = 23 ∧ t.num_games = 22 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_games_tournament_with_23_teams_l277_27753


namespace NUMINAMATH_CALUDE_auto_credit_percentage_l277_27755

def auto_finance_credit : ℝ := 40
def total_consumer_credit : ℝ := 342.857

theorem auto_credit_percentage :
  let total_auto_credit := 3 * auto_finance_credit
  let percentage := (total_auto_credit / total_consumer_credit) * 100
  ∃ ε > 0, |percentage - 35| < ε :=
sorry

end NUMINAMATH_CALUDE_auto_credit_percentage_l277_27755


namespace NUMINAMATH_CALUDE_xiaoyings_journey_equations_correct_l277_27706

/-- Represents a journey with uphill and downhill sections -/
structure Journey where
  total_distance : ℝ
  total_time : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- Xiaoying's journey to school -/
def xiaoyings_journey : Journey where
  total_distance := 1.2  -- 1200 meters converted to kilometers
  total_time := 16
  uphill_speed := 3
  downhill_speed := 5

/-- The system of equations representing Xiaoying's journey -/
def journey_equations (j : Journey) (x y : ℝ) : Prop :=
  (j.uphill_speed / 60 * x + j.downhill_speed / 60 * y = j.total_distance) ∧
  (x + y = j.total_time)

theorem xiaoyings_journey_equations_correct :
  journey_equations xiaoyings_journey = λ x y ↦ 
    (3 / 60 * x + 5 / 60 * y = 1.2) ∧ (x + y = 16) := by sorry

end NUMINAMATH_CALUDE_xiaoyings_journey_equations_correct_l277_27706


namespace NUMINAMATH_CALUDE_hall_breadth_is_15_meters_l277_27721

-- Define the hall length in meters
def hall_length : ℝ := 36

-- Define stone dimensions in decimeters
def stone_length : ℝ := 6
def stone_width : ℝ := 5

-- Define the number of stones
def num_stones : ℕ := 1800

-- Define the conversion factor from square decimeters to square meters
def dm2_to_m2 : ℝ := 0.01

-- Statement to prove
theorem hall_breadth_is_15_meters :
  let stone_area_m2 := stone_length * stone_width * dm2_to_m2
  let total_area_m2 := stone_area_m2 * num_stones
  let hall_breadth := total_area_m2 / hall_length
  hall_breadth = 15 := by sorry

end NUMINAMATH_CALUDE_hall_breadth_is_15_meters_l277_27721


namespace NUMINAMATH_CALUDE_topsoil_cost_for_seven_cubic_yards_l277_27766

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards_of_topsoil : ℝ := 7

/-- The cost of topsoil for a given number of cubic yards -/
def topsoil_cost (cubic_yards : ℝ) : ℝ :=
  cubic_yards * cubic_feet_per_cubic_yard * topsoil_cost_per_cubic_foot

theorem topsoil_cost_for_seven_cubic_yards :
  topsoil_cost cubic_yards_of_topsoil = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_for_seven_cubic_yards_l277_27766


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l277_27762

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def power_of_two (n : ℕ) : ℕ := 2^n

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def sum_powers_of_two (n : ℕ) : ℕ := (List.range n).map power_of_two |>.sum

theorem units_digit_of_sum (n : ℕ) : 
  (sum_factorials n + sum_powers_of_two n) % 10 = 9 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l277_27762


namespace NUMINAMATH_CALUDE_one_common_color_l277_27780

/-- Given a set of n ≥ 5 colors and n+1 distinct 3-element subsets,
    there exist two subsets that share exactly one element. -/
theorem one_common_color (n : ℕ) (C : Finset ℕ) (A : Fin (n + 1) → Finset ℕ)
  (h_n : n ≥ 5)
  (h_C : C.card = n)
  (h_A_subset : ∀ i, A i ⊆ C)
  (h_A_card : ∀ i, (A i).card = 3)
  (h_A_distinct : ∀ i j, i ≠ j → A i ≠ A j) :
  ∃ i j, i ≠ j ∧ (A i ∩ A j).card = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_common_color_l277_27780


namespace NUMINAMATH_CALUDE_sum_of_coordinates_of_D_l277_27728

/-- Given that N is the midpoint of CD and C's coordinates, prove the sum of D's coordinates -/
theorem sum_of_coordinates_of_D (N C : ℝ × ℝ) (h1 : N = (2, 6)) (h2 : C = (6, 2)) :
  ∃ D : ℝ × ℝ, N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) ∧ D.1 + D.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_of_D_l277_27728


namespace NUMINAMATH_CALUDE_lawn_width_proof_l277_27771

/-- The width of a rectangular lawn with specific conditions -/
def lawn_width : ℝ := 50

theorem lawn_width_proof (length : ℝ) (road_width : ℝ) (total_road_area : ℝ) :
  length = 80 →
  road_width = 10 →
  total_road_area = 1200 →
  lawn_width = (total_road_area - (length * road_width) + (road_width * road_width)) / road_width :=
by
  sorry

#check lawn_width_proof

end NUMINAMATH_CALUDE_lawn_width_proof_l277_27771
