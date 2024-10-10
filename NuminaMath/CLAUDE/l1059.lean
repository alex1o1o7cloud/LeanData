import Mathlib

namespace lily_pad_half_coverage_l1059_105909

theorem lily_pad_half_coverage (total_days : ℕ) (half_coverage_days : ℕ) : 
  (total_days = 34) → 
  (half_coverage_days = total_days - 1) →
  (half_coverage_days = 33) :=
by sorry

end lily_pad_half_coverage_l1059_105909


namespace bus_arrival_time_difference_l1059_105979

/-- Proves that a person walking to a bus stand will arrive 10 minutes early when doubling their speed -/
theorem bus_arrival_time_difference (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (miss_time : ℝ) : 
  distance = 2.2 →
  speed1 = 3 →
  speed2 = 6 →
  miss_time = 12 →
  (distance / speed2 * 60) = ((distance / speed1 * 60) - miss_time) - 10 :=
by sorry

end bus_arrival_time_difference_l1059_105979


namespace abs_neg_one_third_l1059_105900

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end abs_neg_one_third_l1059_105900


namespace jarry_secretary_or_treasurer_prob_l1059_105987

/-- A club with 10 members, including Jarry -/
structure Club where
  members : Finset Nat
  jarry : Nat
  total_members : members.card = 10
  jarry_in_club : jarry ∈ members

/-- The probability of Jarry being either secretary or treasurer -/
def probability_jarry_secretary_or_treasurer (club : Club) : ℚ :=
  19 / 90

/-- Theorem stating the probability of Jarry being secretary or treasurer -/
theorem jarry_secretary_or_treasurer_prob (club : Club) :
  probability_jarry_secretary_or_treasurer club = 19 / 90 := by
  sorry

end jarry_secretary_or_treasurer_prob_l1059_105987


namespace cylinder_radius_equals_8_l1059_105958

/-- Given a cylinder and a cone with equal volumes, prove that the cylinder's radius is 8 cm -/
theorem cylinder_radius_equals_8 (h_cyl : ℝ) (r_cyl : ℝ) (h_cone : ℝ) (r_cone : ℝ)
  (h_cyl_val : h_cyl = 2)
  (h_cone_val : h_cone = 6)
  (r_cone_val : r_cone = 8)
  (volume_equal : π * r_cyl^2 * h_cyl = (1/3) * π * r_cone^2 * h_cone) :
  r_cyl = 8 := by
sorry

end cylinder_radius_equals_8_l1059_105958


namespace glucose_solution_volume_l1059_105917

/-- Given a glucose solution where 45 cubic centimeters contain 6.75 grams of glucose,
    prove that the volume containing 15 grams of glucose is 100 cubic centimeters. -/
theorem glucose_solution_volume (volume : ℝ) (glucose_mass : ℝ) :
  (45 : ℝ) / volume = 6.75 / glucose_mass →
  glucose_mass = 15 →
  volume = 100 := by
  sorry

end glucose_solution_volume_l1059_105917


namespace complex_number_value_l1059_105941

theorem complex_number_value : Complex.I ^ 2 * (1 + Complex.I) = -1 - Complex.I := by
  sorry

end complex_number_value_l1059_105941


namespace physics_marks_l1059_105907

theorem physics_marks (P C M : ℝ) 
  (total_avg : (P + C + M) / 3 = 70)
  (physics_math_avg : (P + M) / 2 = 90)
  (physics_chem_avg : (P + C) / 2 = 70) :
  P = 110 := by
sorry

end physics_marks_l1059_105907


namespace compute_expression_l1059_105911

theorem compute_expression : 9 * (2 / 3) ^ 4 = 16 / 9 := by
  sorry

end compute_expression_l1059_105911


namespace inequality_preservation_l1059_105991

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y := by
  sorry

end inequality_preservation_l1059_105991


namespace both_runners_in_picture_probability_zero_l1059_105920

/-- Represents a runner on the circular track -/
structure Runner where
  direction : Bool  -- true for counterclockwise, false for clockwise
  lap_time : ℕ      -- time to complete one lap in seconds

/-- Represents the photographer's picture -/
structure Picture where
  coverage : ℝ      -- fraction of the track covered by the picture
  center : ℝ        -- position of the center of the picture on the track (0 ≤ center < 1)

/-- Calculates the probability of both runners being in the picture -/
def probability_both_in_picture (lydia : Runner) (lucas : Runner) (pic : Picture) : ℝ :=
  sorry

/-- Theorem stating that the probability of both runners being in the picture is 0 -/
theorem both_runners_in_picture_probability_zero 
  (lydia : Runner) 
  (lucas : Runner) 
  (pic : Picture) 
  (h1 : lydia.direction = true) 
  (h2 : lydia.lap_time = 120) 
  (h3 : lucas.direction = false) 
  (h4 : lucas.lap_time = 100) 
  (h5 : pic.coverage = 1/3) 
  (h6 : pic.center = 0) : 
  probability_both_in_picture lydia lucas pic = 0 :=
sorry

end both_runners_in_picture_probability_zero_l1059_105920


namespace unique_positive_solution_l1059_105988

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ x^8 + 8*x^7 + 28*x^6 + 2023*x^5 - 1807*x^4 = 0 :=
by sorry

end unique_positive_solution_l1059_105988


namespace knight_count_l1059_105968

def is_correct_arrangement (knights liars : ℕ) : Prop :=
  knights + liars = 2019 ∧
  knights > 2 * liars ∧
  knights ≤ 2 * liars + 1

theorem knight_count : ∃ (knights liars : ℕ), 
  is_correct_arrangement knights liars ∧ knights = 1346 := by
  sorry

end knight_count_l1059_105968


namespace most_cars_are_blue_l1059_105989

theorem most_cars_are_blue (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow := by
  sorry

end most_cars_are_blue_l1059_105989


namespace peach_price_to_friends_peach_price_proof_l1059_105906

/-- The price of peaches sold to friends, given the following conditions:
  * Lilia has 15 peaches
  * She sold 10 peaches to friends
  * She sold 4 peaches to relatives for $1.25 each
  * She kept 1 peach for herself
  * She earned $25 in total from selling 14 peaches
-/
theorem peach_price_to_friends : ℝ :=
  let total_peaches : ℕ := 15
  let peaches_to_friends : ℕ := 10
  let peaches_to_relatives : ℕ := 4
  let peaches_kept : ℕ := 1
  let price_to_relatives : ℝ := 1.25
  let total_earned : ℝ := 25
  let price_to_friends : ℝ := (total_earned - peaches_to_relatives * price_to_relatives) / peaches_to_friends
  2

theorem peach_price_proof (total_peaches : ℕ) (peaches_to_friends : ℕ) (peaches_to_relatives : ℕ) 
    (peaches_kept : ℕ) (price_to_relatives : ℝ) (total_earned : ℝ) :
    total_peaches = peaches_to_friends + peaches_to_relatives + peaches_kept →
    total_earned = peaches_to_friends * peach_price_to_friends + peaches_to_relatives * price_to_relatives →
    peach_price_to_friends = 2 :=
by sorry

end peach_price_to_friends_peach_price_proof_l1059_105906


namespace revenue_ratio_theorem_l1059_105938

/-- Represents the revenue data for a product line -/
structure ProductLine where
  lastYearRevenue : ℝ
  projectedIncrease : ℝ
  actualDecrease : ℝ

/-- Calculates the projected revenue for a product line -/
def projectedRevenue (p : ProductLine) : ℝ :=
  p.lastYearRevenue * (1 + p.projectedIncrease)

/-- Calculates the actual revenue for a product line -/
def actualRevenue (p : ProductLine) : ℝ :=
  p.lastYearRevenue * (1 - p.actualDecrease)

/-- Theorem stating that the ratio of total actual revenue to total projected revenue
    is approximately 0.5276 for the given product lines -/
theorem revenue_ratio_theorem (standardGum sugarFreeGum bubbleGum : ProductLine)
    (h1 : standardGum.lastYearRevenue = 100000)
    (h2 : standardGum.projectedIncrease = 0.3)
    (h3 : standardGum.actualDecrease = 0.2)
    (h4 : sugarFreeGum.lastYearRevenue = 150000)
    (h5 : sugarFreeGum.projectedIncrease = 0.5)
    (h6 : sugarFreeGum.actualDecrease = 0.3)
    (h7 : bubbleGum.lastYearRevenue = 200000)
    (h8 : bubbleGum.projectedIncrease = 0.4)
    (h9 : bubbleGum.actualDecrease = 0.25) :
    let totalActualRevenue := actualRevenue standardGum + actualRevenue sugarFreeGum + actualRevenue bubbleGum
    let totalProjectedRevenue := projectedRevenue standardGum + projectedRevenue sugarFreeGum + projectedRevenue bubbleGum
    abs (totalActualRevenue / totalProjectedRevenue - 0.5276) < 0.0001 := by
  sorry

end revenue_ratio_theorem_l1059_105938


namespace factorial_division_l1059_105908

theorem factorial_division : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end factorial_division_l1059_105908


namespace special_sequence_coprime_l1059_105947

/-- A polynomial with integer coefficients that maps 0 and 1 to 1 -/
def SpecialPolynomial (p : ℤ → ℤ) : Prop :=
  (∀ x y : ℤ, p (x + y) = p x + p y - 1) ∧ p 0 = 1 ∧ p 1 = 1

/-- The sequence defined by the special polynomial -/
def SpecialSequence (p : ℤ → ℤ) (a : ℕ → ℤ) : Prop :=
  a 0 ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = p (a n)

/-- The theorem stating that any two terms in the sequence are coprime -/
theorem special_sequence_coprime (p : ℤ → ℤ) (a : ℕ → ℤ) 
  (hp : SpecialPolynomial p) (ha : SpecialSequence p a) :
  ∀ i j : ℕ, Nat.gcd (a i).natAbs (a j).natAbs = 1 :=
sorry

end special_sequence_coprime_l1059_105947


namespace divisibility_by_43_l1059_105986

theorem divisibility_by_43 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  43 ∣ (7^p - 6^p - 1) := by
  sorry

end divisibility_by_43_l1059_105986


namespace spinner_probability_divisible_by_3_l1059_105930

/-- A spinner with 8 equal sections numbered from 1 to 8 -/
def Spinner := Finset (Fin 8)

/-- The set of numbers on the spinner that are divisible by 3 -/
def DivisibleBy3 (s : Spinner) : Finset (Fin 8) :=
  s.filter (fun n => n % 3 = 0)

/-- The probability of an event on the spinner -/
def Probability (event : Finset (Fin 8)) (s : Spinner) : ℚ :=
  event.card / s.card

theorem spinner_probability_divisible_by_3 (s : Spinner) :
  Probability (DivisibleBy3 s) s = 1 / 4 :=
sorry

end spinner_probability_divisible_by_3_l1059_105930


namespace inequality_always_holds_l1059_105978

theorem inequality_always_holds (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 1| > a) ↔ a < 2 := by sorry

end inequality_always_holds_l1059_105978


namespace sum_not_five_implies_not_two_or_not_three_l1059_105939

theorem sum_not_five_implies_not_two_or_not_three (a b : ℝ) : 
  a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3 := by
sorry

end sum_not_five_implies_not_two_or_not_three_l1059_105939


namespace max_value_of_exponential_difference_l1059_105932

theorem max_value_of_exponential_difference :
  ∃ (M : ℝ), M = 1/4 ∧ ∀ x : ℝ, 5^x - 25^x ≤ M :=
sorry

end max_value_of_exponential_difference_l1059_105932


namespace history_books_count_l1059_105964

theorem history_books_count (total : ℕ) (reading : ℕ) (math : ℕ) (science : ℕ) (history : ℕ) : 
  total = 10 →
  reading = 2 * total / 5 →
  math = 3 * total / 10 →
  science = math - 1 →
  history = total - (reading + math + science) →
  history = 1 := by
sorry

end history_books_count_l1059_105964


namespace pinterest_group_pins_l1059_105927

theorem pinterest_group_pins 
  (num_members : ℕ) 
  (initial_pins : ℕ) 
  (daily_contribution : ℕ) 
  (weekly_deletion : ℕ) 
  (days_in_month : ℕ) 
  (h1 : num_members = 20)
  (h2 : initial_pins = 1000)
  (h3 : daily_contribution = 10)
  (h4 : weekly_deletion = 5)
  (h5 : days_in_month = 30) :
  let total_new_pins := num_members * daily_contribution * days_in_month
  let total_deleted_pins := num_members * weekly_deletion * (days_in_month / 7)
  initial_pins + total_new_pins - total_deleted_pins = 6600 := by
  sorry

end pinterest_group_pins_l1059_105927


namespace inverse_f_90_l1059_105905

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_f_90 : f⁻¹ 90 = 3 := by
  sorry

end inverse_f_90_l1059_105905


namespace ceiling_sum_of_square_roots_l1059_105956

theorem ceiling_sum_of_square_roots : 
  ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end ceiling_sum_of_square_roots_l1059_105956


namespace arithmetic_sequence_property_l1059_105982

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  a 9 - (1/3) * a 11 = 16 := by
  sorry

end arithmetic_sequence_property_l1059_105982


namespace intersection_M_N_l1059_105972

def U := ℝ

def M : Set ℝ := {-1, 1, 2}

def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_M_N : N ∩ M = {1} := by sorry

end intersection_M_N_l1059_105972


namespace angle_A_is_60_degrees_triangle_is_equilateral_l1059_105999

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = t.a^2 + t.b * t.c

-- Define the law of cosines
def lawOfCosines (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - 2 * t.b * t.c * Real.cos t.A

-- Define the law of sines
def lawOfSines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C

-- Define the equilateral triangle condition
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Theorem 1: Prove that angle A is 60 degrees
theorem angle_A_is_60_degrees (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : satisfiesCondition t) 
  (h3 : lawOfCosines t) : 
  t.A = Real.pi / 3 :=
sorry

-- Theorem 2: Prove that the triangle is equilateral
theorem triangle_is_equilateral (t : Triangle)
  (h1 : isValidTriangle t)
  (h2 : satisfiesCondition t)
  (h3 : lawOfSines t)
  (h4 : Real.sin t.B * Real.sin t.C = Real.sin t.A ^ 2) :
  isEquilateral t :=
sorry

end angle_A_is_60_degrees_triangle_is_equilateral_l1059_105999


namespace sum_of_four_squares_sum_of_four_squares_proof_l1059_105973

theorem sum_of_four_squares : ℕ → ℕ → ℕ → Prop :=
  fun triangle circle square =>
    triangle + circle + triangle + square = 27 ∧
    circle + triangle + circle + square = 25 ∧
    square + square + square + triangle = 39 →
    4 * square = 44

-- The proof would go here, but we're skipping it as per instructions
theorem sum_of_four_squares_proof (triangle circle square : ℕ) 
  (h : sum_of_four_squares triangle circle square) : 4 * square = 44 :=
by
  sorry

end sum_of_four_squares_sum_of_four_squares_proof_l1059_105973


namespace division_remainder_l1059_105970

def largest_three_digit : Nat := 975
def smallest_two_digit : Nat := 23

theorem division_remainder :
  largest_three_digit % smallest_two_digit = 9 := by
  sorry

end division_remainder_l1059_105970


namespace tan_alpha_2_implications_l1059_105966

theorem tan_alpha_2_implications (α : Real) (h : Real.tan α = 2) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -1/3 ∧
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 3 := by
  sorry

end tan_alpha_2_implications_l1059_105966


namespace sqrt_fourth_power_equals_256_l1059_105944

theorem sqrt_fourth_power_equals_256 (y : ℝ) : (Real.sqrt y)^4 = 256 → y = 16 := by
  sorry

end sqrt_fourth_power_equals_256_l1059_105944


namespace world_cup_2018_21st_edition_l1059_105913

/-- Represents the year of the nth World Cup -/
def worldCupYear (n : ℕ) : ℕ := 1950 + 4 * (n - 4)

/-- Theorem stating that the 2018 World Cup was the 21st edition -/
theorem world_cup_2018_21st_edition :
  ∃ n : ℕ, n = 21 ∧ worldCupYear n = 2018 :=
sorry

end world_cup_2018_21st_edition_l1059_105913


namespace abs_neg_one_fifth_l1059_105936

theorem abs_neg_one_fifth : |(-1 : ℚ) / 5| = 1 / 5 := by
  sorry

end abs_neg_one_fifth_l1059_105936


namespace max_sum_of_factors_max_sum_of_factors_achieved_l1059_105918

theorem max_sum_of_factors (a b : ℕ) : a * b = 48 → a ≠ b → a + b ≤ 49 := by sorry

theorem max_sum_of_factors_achieved : ∃ a b : ℕ, a * b = 48 ∧ a ≠ b ∧ a + b = 49 := by sorry

end max_sum_of_factors_max_sum_of_factors_achieved_l1059_105918


namespace perfect_square_quadratic_l1059_105997

theorem perfect_square_quadratic (x k : ℝ) : 
  (∃ a b : ℝ, x^2 - 18*x + k = (a*x + b)^2) ↔ k = 81 := by
sorry

end perfect_square_quadratic_l1059_105997


namespace jonathan_daily_burn_l1059_105912

-- Define Jonathan's daily calorie intake
def daily_intake : ℕ := 2500

-- Define Jonathan's extra calorie intake on Saturday
def saturday_extra : ℕ := 1000

-- Define Jonathan's weekly caloric deficit
def weekly_deficit : ℕ := 2500

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Theorem to prove Jonathan's daily calorie burn
theorem jonathan_daily_burn :
  (6 * daily_intake + (daily_intake + saturday_extra) + weekly_deficit) / days_in_week = 3000 :=
by sorry

end jonathan_daily_burn_l1059_105912


namespace bus_driver_compensation_theorem_l1059_105974

/-- Represents the compensation structure and work hours of a bus driver -/
structure BusDriverCompensation where
  regular_rate : ℝ
  overtime_rate : ℝ
  total_compensation : ℝ
  total_hours : ℝ
  regular_hours_limit : ℝ

/-- Calculates the overtime rate based on the regular rate -/
def overtime_rate (regular_rate : ℝ) : ℝ :=
  regular_rate * 1.75

/-- Theorem stating the conditions and the result to be proved -/
theorem bus_driver_compensation_theorem (driver : BusDriverCompensation) :
  driver.regular_rate = 16 ∧
  driver.overtime_rate = overtime_rate driver.regular_rate ∧
  driver.total_compensation = 920 ∧
  driver.total_hours = 50 →
  driver.regular_hours_limit = 40 := by
  sorry


end bus_driver_compensation_theorem_l1059_105974


namespace popcorn_servings_for_jared_and_friends_l1059_105993

/-- Calculate the number of popcorn servings needed for a group -/
def popcorn_servings (pieces_per_serving : ℕ) (jared_pieces : ℕ) (num_friends : ℕ) (friend_pieces : ℕ) : ℕ :=
  ((jared_pieces + num_friends * friend_pieces) + pieces_per_serving - 1) / pieces_per_serving

theorem popcorn_servings_for_jared_and_friends :
  popcorn_servings 30 90 3 60 = 9 := by
  sorry

end popcorn_servings_for_jared_and_friends_l1059_105993


namespace total_snowfall_l1059_105924

theorem total_snowfall (morning_snow afternoon_snow : ℝ) 
  (h1 : morning_snow = 0.125)
  (h2 : afternoon_snow = 0.5) :
  morning_snow + afternoon_snow = 0.625 := by
  sorry

end total_snowfall_l1059_105924


namespace solve_for_q_l1059_105959

theorem solve_for_q (n d p q : ℝ) (h1 : d ≠ 0) (h2 : p ≠ 0) (h3 : q ≠ 0) 
  (h4 : n = (2 * d * p * q) / (p - q)) : 
  q = (n * p) / (2 * d * p + n) := by
  sorry

end solve_for_q_l1059_105959


namespace sum_of_x_and_y_in_arithmetic_sequence_l1059_105975

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The given sequence 3, 9, x, y, 27 -/
def givenSequence (x y : ℝ) : ℕ → ℝ
  | 0 => 3
  | 1 => 9
  | 2 => x
  | 3 => y
  | 4 => 27
  | _ => 0  -- For indices beyond 4, we return 0 (this part is not relevant to our problem)

theorem sum_of_x_and_y_in_arithmetic_sequence (x y : ℝ) 
    (h : isArithmeticSequence (givenSequence x y)) : x + y = 36 := by
  sorry

end sum_of_x_and_y_in_arithmetic_sequence_l1059_105975


namespace smallest_integer_satisfying_conditions_l1059_105902

theorem smallest_integer_satisfying_conditions : ∃ x : ℤ, 
  (∀ y : ℤ, (|3*y - 4| ≤ 25 ∧ 3 ∣ y) → x ≤ y) ∧ 
  |3*x - 4| ≤ 25 ∧ 
  3 ∣ x ∧
  x = -6 := by
sorry

end smallest_integer_satisfying_conditions_l1059_105902


namespace grandma_molly_statues_l1059_105948

/-- The number of statues Grandma Molly created in the first year -/
def initial_statues : ℕ := sorry

/-- The total number of statues after four years -/
def total_statues : ℕ := 31

/-- The number of statues broken in the third year -/
def broken_statues : ℕ := 3

theorem grandma_molly_statues :
  initial_statues = 4 ∧
  (4 * initial_statues + 12 - broken_statues + 2 * broken_statues = total_statues) :=
sorry

end grandma_molly_statues_l1059_105948


namespace special_checkerboard_black_squares_l1059_105928

/-- A checkerboard with alternating colors -/
structure Checkerboard where
  size : ℕ
  isRed : Fin size → Fin size → Bool

/-- The properties of our specific checkerboard -/
def specialCheckerboard : Checkerboard where
  size := 32
  isRed := fun i j => 
    (i.val + j.val) % 2 = 0 ∨ i.val = 0 ∨ i.val = 31 ∨ j.val = 0 ∨ j.val = 31

/-- Count of black squares on the checkerboard -/
def blackSquareCount (c : Checkerboard) : ℕ :=
  (c.size * c.size) - (Finset.sum (Finset.univ : Finset (Fin c.size × Fin c.size)) 
    fun (i, j) => if c.isRed i j then 1 else 0)

/-- Theorem stating the number of black squares on our special checkerboard -/
theorem special_checkerboard_black_squares :
  blackSquareCount specialCheckerboard = 511 := by sorry

end special_checkerboard_black_squares_l1059_105928


namespace no_perfect_square_sum_of_prime_powers_l1059_105951

theorem no_perfect_square_sum_of_prime_powers (p k m : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ¬∃ x : ℕ, p^k + p^m = x^2 := by
  sorry

end no_perfect_square_sum_of_prime_powers_l1059_105951


namespace expression_value_l1059_105963

theorem expression_value :
  let x : ℝ := 1
  let y : ℝ := 1
  let z : ℝ := 3
  let p : ℝ := 2
  let q : ℝ := 4
  let r : ℝ := 2
  let s : ℝ := 3
  let t : ℝ := 3
  (p + x)^2 * y * z - q * r * (x * y * z)^2 + s^t = -18 := by sorry

end expression_value_l1059_105963


namespace sum_problem_l1059_105922

/-- The sum of 38 and twice a number is a certain value. The number is 43. -/
theorem sum_problem (x : ℕ) (h : x = 43) : 38 + 2 * x = 124 := by
  sorry

end sum_problem_l1059_105922


namespace average_of_w_and_x_l1059_105969

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end average_of_w_and_x_l1059_105969


namespace polygon_area_is_400_l1059_105950

-- Define the polygon vertices
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (20, 0)
def C : ℝ × ℝ := (30, 10)
def D : ℝ × ℝ := (20, 20)
def E : ℝ × ℝ := (10, 10)
def F : ℝ × ℝ := (0, 20)

-- Define the polygon as a list of vertices
def polygon : List (ℝ × ℝ) := [A, B, C, D, E, F]

-- Function to calculate the area of a polygon given its vertices
def polygonArea (vertices : List (ℝ × ℝ)) : ℝ :=
  sorry -- Implementation not required for this task

-- Theorem statement
theorem polygon_area_is_400 : polygonArea polygon = 400 := by
  sorry -- Proof not required for this task

end polygon_area_is_400_l1059_105950


namespace reflection_curve_coefficient_product_l1059_105957

/-- The reflection of the curve xy = 1 over the line y = 2x -/
def ReflectedCurve (x y : ℝ) : Prop :=
  ∃ (b c d : ℝ), 12 * x^2 + b * x * y + c * y^2 + d = 0

/-- The product of coefficients b and c in the reflected curve equation -/
def CoefficientProduct (b c : ℝ) : ℝ := b * c

theorem reflection_curve_coefficient_product :
  ∃ (b c : ℝ), ReflectedCurve x y ∧ CoefficientProduct b c = 84 := by
  sorry

end reflection_curve_coefficient_product_l1059_105957


namespace tangent_and_trigonometric_identity_l1059_105952

theorem tangent_and_trigonometric_identity (α : Real) 
  (h : Real.tan (α + π/3) = 2 * Real.sqrt 3) : 
  (Real.tan (α - 2*π/3) = 2 * Real.sqrt 3) ∧ 
  (2 * Real.sin α ^ 2 - Real.cos α ^ 2 = -43/52) := by
  sorry

end tangent_and_trigonometric_identity_l1059_105952


namespace line_passes_through_fixed_point_l1059_105915

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k : ℝ) = k * 2 + (-1) + 1 := by sorry

end line_passes_through_fixed_point_l1059_105915


namespace max_programs_max_programs_achievable_max_programs_optimal_l1059_105940

theorem max_programs (n : ℕ) : n ≤ 4 :=
  sorry

theorem max_programs_achievable : ∃ (P : Fin 4 → Finset (Fin 12)),
  (∀ i : Fin 4, (P i).card = 6) ∧
  (∀ i j : Fin 4, i ≠ j → (P i ∩ P j).card ≤ 2) :=
  sorry

theorem max_programs_optimal :
  ¬∃ (P : Fin 5 → Finset (Fin 12)),
    (∀ i : Fin 5, (P i).card = 6) ∧
    (∀ i j : Fin 5, i ≠ j → (P i ∩ P j).card ≤ 2) :=
  sorry

end max_programs_max_programs_achievable_max_programs_optimal_l1059_105940


namespace inequality_subtraction_l1059_105960

theorem inequality_subtraction (a b c d : ℝ) (h1 : a > b) (h2 : c < d) : a - c > b - d := by
  sorry

end inequality_subtraction_l1059_105960


namespace gum_distribution_l1059_105910

theorem gum_distribution (cousins : ℕ) (total_gum : ℕ) (gum_per_cousin : ℕ) :
  cousins = 4 →
  total_gum = 20 →
  total_gum = cousins * gum_per_cousin →
  gum_per_cousin = 5 :=
by
  sorry

end gum_distribution_l1059_105910


namespace rectangular_triangular_field_equal_area_l1059_105962

/-- Proves that a rectangular field with width 4 m and length 6.3 m has the same area
    as a triangular field with base 7.2 m and height 7 m. -/
theorem rectangular_triangular_field_equal_area :
  let triangle_base : ℝ := 7.2
  let triangle_height : ℝ := 7
  let triangle_area : ℝ := (triangle_base * triangle_height) / 2
  let rectangle_width : ℝ := 4
  let rectangle_length : ℝ := 6.3
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  triangle_area = rectangle_area := by sorry

end rectangular_triangular_field_equal_area_l1059_105962


namespace fraction_ordering_l1059_105904

theorem fraction_ordering : 19/15 < 17/13 ∧ 17/13 < 15/11 := by
  sorry

end fraction_ordering_l1059_105904


namespace triangle_properties_l1059_105921

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  -- The given equation
  (Real.sqrt 3 * Real.sin A * Real.cos A - Real.sin A ^ 2 = 0) ∧
  -- Collinearity condition
  (∃ (k : Real), k ≠ 0 ∧ k * 1 = 2 ∧ k * Real.sin C = Real.sin B) ∧
  -- Given side length
  (a = 3) →
  -- Prove angle A and perimeter
  (A = π / 3) ∧
  (a + b + c = 3 * (1 + Real.sqrt 3)) := by
sorry

end triangle_properties_l1059_105921


namespace vector_collinearity_l1059_105919

theorem vector_collinearity (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (2 • a + b) = k • b) → x = -2 := by
  sorry

end vector_collinearity_l1059_105919


namespace dodecagon_diagonal_equality_l1059_105933

/-- A regular dodecagon -/
structure RegularDodecagon where
  /-- Side length of the dodecagon -/
  a : ℝ
  /-- Length of shortest diagonal spanning three sides -/
  b : ℝ
  /-- Length of longest diagonal spanning six sides -/
  d : ℝ
  /-- Positive side length -/
  a_pos : 0 < a

/-- In a regular dodecagon, the length of the shortest diagonal spanning three sides
    is equal to the length of the longest diagonal spanning six sides -/
theorem dodecagon_diagonal_equality (poly : RegularDodecagon) : poly.b = poly.d := by
  sorry

end dodecagon_diagonal_equality_l1059_105933


namespace consecutive_integers_permutation_divisibility_l1059_105901

theorem consecutive_integers_permutation_divisibility
  (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
  (m : ℕ → ℕ) (hm : ∀ i ∈ Finset.range p, m (i + 1) = m i + 1)
  (σ : Fin p → Fin p) (hσ : Function.Bijective σ) :
  ∃ k l : Fin p, k ≠ l ∧ p ∣ (m k * m (σ k) - m l * m (σ l)) :=
sorry

end consecutive_integers_permutation_divisibility_l1059_105901


namespace girls_in_club_l1059_105961

theorem girls_in_club (total : Nat) (girls : Nat) (boys : Nat) : 
  total = 36 →
  girls + boys = total →
  (∀ (group : Nat), group = 33 → girls > group / 2) →
  (∃ (group : Nat), group = 31 ∧ boys ≥ group / 2) →
  girls = 20 :=
by sorry

end girls_in_club_l1059_105961


namespace problem_statement_l1059_105945

theorem problem_statement (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12)
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) :
  a + b^2 + c^3 = 14 := by
sorry

end problem_statement_l1059_105945


namespace printing_completion_time_l1059_105914

def start_time : ℕ := 9 * 60  -- 9:00 AM in minutes since midnight
def quarter_completion_time : ℕ := 12 * 60 + 30  -- 12:30 PM in minutes since midnight

-- Time taken to complete one-fourth of the job in minutes
def quarter_job_duration : ℕ := quarter_completion_time - start_time

-- Total time required to complete the entire job in minutes
def total_job_duration : ℕ := 4 * quarter_job_duration

-- Completion time in minutes since midnight
def completion_time : ℕ := start_time + total_job_duration

theorem printing_completion_time :
  completion_time = 23 * 60 := by sorry

end printing_completion_time_l1059_105914


namespace total_apples_l1059_105983

theorem total_apples (kayla_apples kylie_apples : ℕ) : 
  kayla_apples = 40 → 
  kayla_apples = kylie_apples / 4 → 
  kayla_apples + kylie_apples = 200 :=
by
  sorry

end total_apples_l1059_105983


namespace students_after_yoongi_l1059_105946

/-- Given a group of students waiting for a bus, this theorem proves
    the number of students who came after a specific student. -/
theorem students_after_yoongi 
  (total : ℕ) 
  (before_jungkook : ℕ) 
  (h1 : total = 20) 
  (h2 : before_jungkook = 11) 
  (h3 : ∃ (before_yoongi : ℕ), before_yoongi + 1 = before_jungkook) : 
  ∃ (after_yoongi : ℕ), after_yoongi = total - (before_jungkook - 1) - 1 ∧ after_yoongi = 9 :=
by sorry

end students_after_yoongi_l1059_105946


namespace bee_puzzle_l1059_105916

theorem bee_puzzle (B : ℕ) 
  (h1 : B > 0)
  (h2 : B % 5 = 0)
  (h3 : B % 3 = 0)
  (h4 : B = B / 5 + B / 3 + 3 * (B / 3 - B / 5) + 1) :
  B = 15 := by
sorry

end bee_puzzle_l1059_105916


namespace h_of_two_eq_two_l1059_105923

/-- The function h satisfying the given equation for all x -/
noncomputable def h : ℝ → ℝ :=
  fun x => ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) - 1) / (x^(2^4 - 1) - 1)

/-- Theorem stating that h(2) = 2 -/
theorem h_of_two_eq_two : h 2 = 2 := by sorry

end h_of_two_eq_two_l1059_105923


namespace only_negative_number_l1059_105953

theorem only_negative_number (a b c d : ℚ) : 
  a = 0 → b = -(-3) → c = -1/2 → d = 3.2 → 
  (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) ∧ 
  (a ≥ 0 ∧ b ≥ 0 ∧ d ≥ 0) ∧ 
  c < 0 := by
sorry

end only_negative_number_l1059_105953


namespace distance_probability_l1059_105929

/-- Represents a city in our problem -/
inductive City : Type
| Bangkok : City
| CapeTown : City
| Honolulu : City
| London : City

/-- The distance between two cities in miles -/
def distance (c1 c2 : City) : ℕ :=
  match c1, c2 with
  | City.Bangkok, City.CapeTown => 6300
  | City.Bangkok, City.Honolulu => 6609
  | City.Bangkok, City.London => 5944
  | City.CapeTown, City.Honolulu => 11535
  | City.CapeTown, City.London => 5989
  | City.Honolulu, City.London => 7240
  | _, _ => 0  -- Same city or reverse order

/-- The total number of unique city pairs -/
def totalPairs : ℕ := 6

/-- The number of city pairs with distance less than 8000 miles -/
def pairsLessThan8000 : ℕ := 5

/-- The probability of selecting two cities with distance less than 8000 miles -/
def probability : ℚ := 5 / 6

theorem distance_probability :
  (probability : ℚ) = (pairsLessThan8000 : ℚ) / (totalPairs : ℚ) :=
by sorry

end distance_probability_l1059_105929


namespace tan_half_less_than_x_l1059_105965

theorem tan_half_less_than_x (x : ℝ) (h1 : 0 < x) (h2 : x ≤ π / 2) : Real.tan (x / 2) < x := by
  sorry

end tan_half_less_than_x_l1059_105965


namespace jerrys_age_l1059_105994

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 14 → 
  mickey_age = 3 * jerry_age - 4 → 
  jerry_age = 6 := by
sorry

end jerrys_age_l1059_105994


namespace percentage_calculation_l1059_105980

theorem percentage_calculation (P : ℝ) : 
  P * 5600 = 126 → 
  (0.3 * 0.5 * 5600 : ℝ) = 840 → 
  P = 0.0225 := by
sorry

end percentage_calculation_l1059_105980


namespace expand_quadratic_l1059_105926

theorem expand_quadratic (a : ℝ) : a * (a - 3) = a^2 - 3*a := by sorry

end expand_quadratic_l1059_105926


namespace line_equation_of_parabola_points_l1059_105949

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 3*y

-- Define the quadratic equation
def quadratic_equation (x p q : ℝ) : Prop := x^2 + p*x + q = 0

theorem line_equation_of_parabola_points (p q : ℝ) (h : p^2 - 4*q > 0) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    quadratic_equation x₁ p q ∧ quadratic_equation x₂ p q ∧
    x₁ ≠ x₂ ∧
    ∀ (x y : ℝ), (p*x + 3*y + q = 0) ↔ (y - y₁) = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁) :=
sorry

end line_equation_of_parabola_points_l1059_105949


namespace square_grid_divisible_four_parts_l1059_105942

/-- A square grid of cells that can be divided into four equal parts -/
structure SquareGrid where
  n : ℕ
  n_even : Even n
  n_ge_2 : n ≥ 2

/-- The number of cells in each part when the grid is divided into four equal parts -/
def cells_per_part (grid : SquareGrid) : ℕ := grid.n * grid.n / 4

/-- Theorem stating that a square grid can be divided into four equal parts -/
theorem square_grid_divisible_four_parts (grid : SquareGrid) :
  ∃ (part_size : ℕ), part_size = cells_per_part grid ∧ 
  grid.n * grid.n = 4 * part_size :=
sorry

end square_grid_divisible_four_parts_l1059_105942


namespace cube_edge_length_l1059_105954

theorem cube_edge_length (surface_area : ℝ) (edge_length : ℝ) : 
  surface_area = 54 → 
  surface_area = 6 * edge_length ^ 2 → 
  edge_length = 3 := by
  sorry

end cube_edge_length_l1059_105954


namespace cindy_added_25_pens_l1059_105971

/-- Calculates the number of pens Cindy added given the initial count, pens received, pens given away, and final count. -/
def pens_added_by_cindy (initial : ℕ) (received : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial + received - given_away)

/-- Theorem stating that Cindy added 25 pens given the specific conditions of the problem. -/
theorem cindy_added_25_pens :
  pens_added_by_cindy 5 20 10 40 = 25 := by
  sorry

#eval pens_added_by_cindy 5 20 10 40

end cindy_added_25_pens_l1059_105971


namespace lee_cookies_l1059_105935

/-- Given an initial ratio of flour to cookies and a new amount of flour, 
    calculate the number of cookies that can be made. -/
def cookies_from_flour (initial_flour : ℚ) (initial_cookies : ℚ) (new_flour : ℚ) : ℚ :=
  (initial_cookies / initial_flour) * new_flour

/-- Theorem stating that with the given initial ratio and remaining flour, 
    Lee can make 36 cookies. -/
theorem lee_cookies : 
  let initial_flour : ℚ := 2
  let initial_cookies : ℚ := 18
  let initial_available : ℚ := 5
  let spilled : ℚ := 1
  let remaining_flour : ℚ := initial_available - spilled
  cookies_from_flour initial_flour initial_cookies remaining_flour = 36 := by
  sorry

end lee_cookies_l1059_105935


namespace harry_seed_purchase_cost_l1059_105903

/-- Given the prices of seed packets and the quantities Harry wants to buy, 
    prove that the total cost is $18.00 -/
theorem harry_seed_purchase_cost : 
  let pumpkin_price : ℚ := 25/10
  let tomato_price : ℚ := 15/10
  let chili_price : ℚ := 9/10
  let pumpkin_quantity : ℕ := 3
  let tomato_quantity : ℕ := 4
  let chili_quantity : ℕ := 5
  (pumpkin_price * pumpkin_quantity + 
   tomato_price * tomato_quantity + 
   chili_price * chili_quantity) = 18
:= by sorry

end harry_seed_purchase_cost_l1059_105903


namespace jeep_speed_calculation_l1059_105981

theorem jeep_speed_calculation (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 420 ∧ original_time = 7 ∧ new_time_factor = 3/2 →
  distance / (new_time_factor * original_time) = 40 := by
  sorry

end jeep_speed_calculation_l1059_105981


namespace third_term_of_geometric_series_l1059_105985

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    prove that the third term is 3/4. -/
theorem third_term_of_geometric_series 
  (a : ℝ) -- First term of the series
  (h1 : 0 < (1 : ℝ) - (1/4 : ℝ)) -- Condition for convergence of infinite geometric series
  (h2 : a / (1 - (1/4 : ℝ)) = 16) -- Sum formula for infinite geometric series
  : a * (1/4)^2 = 3/4 := by
  sorry

end third_term_of_geometric_series_l1059_105985


namespace fourth_power_sum_l1059_105977

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a^2 + b^2 + c^2 = 5) 
  (h3 : a^3 + b^3 + c^3 = 8) : 
  a^4 + b^4 + c^4 = 19.5 := by
  sorry

end fourth_power_sum_l1059_105977


namespace algorithm_output_l1059_105943

def sum_odd_numbers (n : Nat) : Nat :=
  List.sum (List.range n |>.filter (λ x => x % 2 = 1))

theorem algorithm_output : 1 + sum_odd_numbers 6 = 10 := by
  sorry

end algorithm_output_l1059_105943


namespace expression_equals_eight_l1059_105996

theorem expression_equals_eight : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) = 8 := by
  sorry

end expression_equals_eight_l1059_105996


namespace man_swimming_speed_l1059_105984

/-- The speed of a man in still water, given his downstream and upstream swimming times and distances -/
theorem man_swimming_speed 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_distance : ℝ) 
  (upstream_time : ℝ) 
  (h1 : downstream_distance = 48) 
  (h2 : downstream_time = 3) 
  (h3 : upstream_distance = 34) 
  (h4 : upstream_time = 4) : 
  ∃ (man_speed stream_speed : ℝ), 
    man_speed = 12.25 ∧ 
    downstream_distance = (man_speed + stream_speed) * downstream_time ∧
    upstream_distance = (man_speed - stream_speed) * upstream_time :=
by sorry

end man_swimming_speed_l1059_105984


namespace school_dinner_drink_choice_l1059_105990

theorem school_dinner_drink_choice (total_students : ℕ) 
  (juice_percentage : ℚ) (water_percentage : ℚ) (juice_students : ℕ) :
  juice_percentage = 3/4 →
  water_percentage = 1/4 →
  juice_students = 90 →
  ∃ water_students : ℕ, water_students = 30 ∧ 
    (juice_students : ℚ) / total_students = juice_percentage ∧
    (water_students : ℚ) / total_students = water_percentage :=
by sorry

end school_dinner_drink_choice_l1059_105990


namespace moving_circle_center_path_l1059_105955

/-- A moving circle M with center (x, y) passes through (3, 2) and is tangent to y = 1 -/
def MovingCircle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 2)^2 = (y - 1)^2

/-- The equation of the path traced by the center of the moving circle -/
def CenterPath (x y : ℝ) : Prop :=
  x^2 - 6*x + 2*y + 12 = 0

/-- Theorem: The equation of the path traced by the center of the moving circle
    is x^2 - 6x + 2y + 12 = 0 -/
theorem moving_circle_center_path :
  ∀ x y : ℝ, MovingCircle x y → CenterPath x y := by
  sorry

end moving_circle_center_path_l1059_105955


namespace smallest_rearranged_multiple_of_nine_l1059_105967

/-- A function that returns the digits of a natural number as a list -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- A predicate that checks if two natural numbers have the same digits -/
def same_digits (a b : ℕ) : Prop :=
  digits a = digits b

/-- The theorem stating that 1089 is the smallest natural number
    that when multiplied by 9, results in a number with the same digits -/
theorem smallest_rearranged_multiple_of_nine :
  (∀ n : ℕ, n < 1089 → ¬(same_digits n (9 * n))) ∧
  (same_digits 1089 (9 * 1089)) :=
sorry

end smallest_rearranged_multiple_of_nine_l1059_105967


namespace ashley_champagne_bottles_l1059_105992

/-- The number of bottles of champagne needed for a wedding toast -/
def bottles_needed (glasses_per_guest : ℕ) (num_guests : ℕ) (servings_per_bottle : ℕ) : ℕ :=
  (glasses_per_guest * num_guests + servings_per_bottle - 1) / servings_per_bottle

/-- Proof that Ashley needs 40 bottles of champagne for her wedding toast -/
theorem ashley_champagne_bottles :
  bottles_needed 2 120 6 = 40 := by
  sorry

end ashley_champagne_bottles_l1059_105992


namespace fibSeriesSum_l1059_105937

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define the series sum
noncomputable def fibSeries : ℝ := ∑' n : ℕ, (fib (2 * n + 1) : ℝ) / (5 : ℝ) ^ n

-- Theorem statement
theorem fibSeriesSum : fibSeries = 35 / 3 := by sorry

end fibSeriesSum_l1059_105937


namespace milkshake_production_theorem_l1059_105995

/-- Represents the milkshake production scenario -/
structure MilkshakeProduction where
  augustus_rate : ℕ
  luna_rate : ℕ
  neptune_rate : ℕ
  total_hours : ℕ
  neptune_start : ℕ
  break_interval : ℕ
  extra_break : ℕ
  break_consumption : ℕ

/-- Calculates the total number of milkshakes produced -/
def total_milkshakes (prod : MilkshakeProduction) : ℕ :=
  sorry

/-- The main theorem stating that given the conditions, 93 milkshakes are produced -/
theorem milkshake_production_theorem (prod : MilkshakeProduction)
  (h1 : prod.augustus_rate = 3)
  (h2 : prod.luna_rate = 7)
  (h3 : prod.neptune_rate = 5)
  (h4 : prod.total_hours = 12)
  (h5 : prod.neptune_start = 3)
  (h6 : prod.break_interval = 3)
  (h7 : prod.extra_break = 7)
  (h8 : prod.break_consumption = 18) :
  total_milkshakes prod = 93 :=
sorry

end milkshake_production_theorem_l1059_105995


namespace joggers_speed_ratio_l1059_105998

theorem joggers_speed_ratio (v₁ v₂ : ℝ) (h1 : v₁ > v₂) (h2 : (v₁ + v₂) * 2 = 8) (h3 : (v₁ - v₂) * 4 = 8) : v₁ / v₂ = 3 := by
  sorry

end joggers_speed_ratio_l1059_105998


namespace vector_subtraction_l1059_105925

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := ![2, -x]
def b : Fin 2 → ℝ := ![-1, 3]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem vector_subtraction (x : ℝ) : 
  dot_product (a x) b = 4 → 
  (λ i : Fin 2 => (a x) i - 2 * (b i)) = ![4, -4] := by
  sorry

end vector_subtraction_l1059_105925


namespace divisibility_condition_exists_divisibility_for_all_implies_equality_l1059_105931

-- Part (a)
theorem divisibility_condition_exists (n : ℕ+) :
  ∃ (x y : ℕ+), x ≠ y ∧ ∀ j ∈ Finset.range n, (x + j) ∣ (y + j) := by sorry

-- Part (b)
theorem divisibility_for_all_implies_equality (x y : ℕ+) :
  (∀ j : ℕ+, (x + j) ∣ (y + j)) → x = y := by sorry

end divisibility_condition_exists_divisibility_for_all_implies_equality_l1059_105931


namespace equidistant_line_proof_l1059_105976

-- Define the two given lines
def line1 (x y : ℝ) : ℝ := 3 * x + 2 * y - 6
def line2 (x y : ℝ) : ℝ := 6 * x + 4 * y - 3

-- Define the proposed equidistant line
def equidistant_line (x y : ℝ) : ℝ := 12 * x + 8 * y - 15

-- Theorem statement
theorem equidistant_line_proof :
  ∀ (x y : ℝ), |equidistant_line x y| = |line1 x y| ∧ |equidistant_line x y| = |line2 x y| :=
sorry

end equidistant_line_proof_l1059_105976


namespace ellipse_line_slope_l1059_105934

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 3 = 1

-- Define a line l
def line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

-- Theorem statement
theorem ellipse_line_slope (x₁ y₁ x₂ y₂ k m : ℝ) :
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
  line k m x₁ y₁ ∧ line k m x₂ y₂ ∧
  is_midpoint x₁ y₁ x₂ y₂ 1 1 →
  k = -1/2 := by
  sorry

end ellipse_line_slope_l1059_105934
