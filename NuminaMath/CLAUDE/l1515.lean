import Mathlib

namespace NUMINAMATH_CALUDE_joey_age_digit_sum_l1515_151504

-- Define the current ages
def zoe_current_age : ℕ := 2
def chloe_current_age : ℕ := 38  -- Derived from the condition that Chloe will be 40 when her age is first a multiple of Zoe's
def joey_current_age : ℕ := chloe_current_age + 1

-- Define the future point when Chloe's age is first a multiple of Zoe's
def years_until_chloe_multiple : ℕ := 40 - chloe_current_age

-- Define a function to calculate the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- The main theorem
theorem joey_age_digit_sum : 
  ∃ (m : ℕ), 
    m > years_until_chloe_multiple ∧ 
    (joey_current_age + m) % (zoe_current_age + m) = 0 ∧
    sum_of_digits (joey_current_age + m) = 13 := by
  sorry

end NUMINAMATH_CALUDE_joey_age_digit_sum_l1515_151504


namespace NUMINAMATH_CALUDE_min_questions_to_determine_l1515_151530

def questions_to_determine (x : ℕ) : ℕ :=
  if x ≥ 10 ∧ x ≤ 19 then
    if x ≤ 14 then
      if x ≤ 12 then
        if x = 11 then 3 else 3
      else
        if x = 13 then 3 else 3
    else
      if x ≤ 17 then
        if x ≤ 16 then
          if x = 15 then 4 else 4
        else 3
      else
        if x = 18 then 3 else 3
  else 0

theorem min_questions_to_determine :
  ∀ x : ℕ, x ≥ 10 ∧ x ≤ 19 → questions_to_determine x ≤ 3 ∧
  (∀ y : ℕ, y ≥ 10 ∧ y ≤ 19 ∧ y ≠ x → ∃ q : ℕ, q < questions_to_determine x ∧
    (∀ z : ℕ, z ≥ 10 ∧ z ≤ 19 → questions_to_determine z < q → z ≠ x ∧ z ≠ y)) :=
sorry

end NUMINAMATH_CALUDE_min_questions_to_determine_l1515_151530


namespace NUMINAMATH_CALUDE_rectangle_division_l1515_151517

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Represents the large rectangle ABCD -/
def large_rectangle : Rectangle := { a := 18, b := 16 }

/-- Represents a small rectangle within ABCD -/
structure SmallRectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.a + r.b)

/-- The perimeter of a small rectangle -/
def small_perimeter (r : SmallRectangle) : ℝ := 2 * (r.x + r.y)

/-- The theorem to be proved -/
theorem rectangle_division (small1 small2 small3 small4 : SmallRectangle) :
  large_rectangle.a = 18 ∧ large_rectangle.b = 16 ∧
  small_perimeter small1 = small_perimeter small2 ∧
  small_perimeter small2 = small_perimeter small3 ∧
  small_perimeter small3 = small_perimeter small4 ∧
  small1.x + small2.x + small3.x = large_rectangle.a ∧
  small1.y + small2.y + small3.y + small4.y = large_rectangle.b →
  (small1.x = 2 ∧ small1.y = 18 ∧
   small2.x = 6 ∧ small2.y = 14 ∧
   small3.x = 6 ∧ small3.y = 14 ∧
   small4.x = 6 ∧ small4.y = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l1515_151517


namespace NUMINAMATH_CALUDE_triangle_area_fraction_l1515_151529

/-- The area of a triangle given the coordinates of its vertices -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The theorem stating that the area of the given triangle divided by the area of the grid equals 5/28 -/
theorem triangle_area_fraction :
  let a := (2, 2)
  let b := (6, 3)
  let c := (3, 6)
  let gridArea := 7 * 6
  (triangleArea a.1 a.2 b.1 b.2 c.1 c.2) / gridArea = 5 / 28 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_fraction_l1515_151529


namespace NUMINAMATH_CALUDE_complex_magnitude_l1515_151523

theorem complex_magnitude (a : ℝ) (z : ℂ) : 
  z = a + Complex.I ∧ z^2 + z = 1 - 3*Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1515_151523


namespace NUMINAMATH_CALUDE_factorial_division_l1515_151538

theorem factorial_division : Nat.factorial 9 / Nat.factorial (9 - 3) = 504 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1515_151538


namespace NUMINAMATH_CALUDE_min_a_for_p_half_ge_p_23_value_l1515_151551

def p (a : ℕ) : ℚ :=
  (Nat.choose (41 - a) 2 + Nat.choose (a - 1) 2) / Nat.choose 50 2

theorem min_a_for_p_half_ge :
  ∀ a : ℕ, 1 ≤ a → a ≤ 40 → (∀ b : ℕ, 1 ≤ b → b < a → p b < 1/2) → p a ≥ 1/2 → a = 23 :=
sorry

theorem p_23_value : p 23 = 34/49 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_p_half_ge_p_23_value_l1515_151551


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l1515_151535

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2*x - 3}

-- Define set N
def N : Set ℝ := {x | -5 ≤ x ∧ x ≤ 2}

-- Define the complement of N with respect to ℝ
def complement_N : Set ℝ := {x | x < -5 ∨ 2 < x}

-- Theorem statement
theorem intersection_M_complement_N : M ∩ complement_N = {y | y > 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l1515_151535


namespace NUMINAMATH_CALUDE_quadratic_coefficient_values_l1515_151581

/-- A quadratic function f(x) = ax^2 + 2ax + 1 with a maximum value of 5 on the interval [-2, 3] -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

/-- The maximum value of the quadratic function on the given interval -/
def max_value : ℝ := 5

/-- The lower bound of the interval -/
def lower_bound : ℝ := -2

/-- The upper bound of the interval -/
def upper_bound : ℝ := 3

/-- Theorem stating that the value of 'a' in the quadratic function
    with the given properties is either 4/15 or -4 -/
theorem quadratic_coefficient_values :
  ∃ (a : ℝ), (∀ x ∈ Set.Icc lower_bound upper_bound,
    quadratic_function a x ≤ max_value) ∧
  (∃ x ∈ Set.Icc lower_bound upper_bound,
    quadratic_function a x = max_value) ∧
  (a = 4/15 ∨ a = -4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_values_l1515_151581


namespace NUMINAMATH_CALUDE_correct_train_sequence_l1515_151595

-- Define the steps as an enumeration
inductive TrainStep
  | BuyTicket
  | WaitInWaitingRoom
  | CheckTicketAtGate
  | BoardTrain

def correct_sequence : List TrainStep :=
  [TrainStep.BuyTicket, TrainStep.WaitInWaitingRoom, TrainStep.CheckTicketAtGate, TrainStep.BoardTrain]

-- Define a function to check if a given sequence is correct
def is_correct_sequence (sequence : List TrainStep) : Prop :=
  sequence = correct_sequence

-- Theorem stating that the given sequence is correct
theorem correct_train_sequence : 
  is_correct_sequence [TrainStep.BuyTicket, TrainStep.WaitInWaitingRoom, TrainStep.CheckTicketAtGate, TrainStep.BoardTrain] :=
by sorry


end NUMINAMATH_CALUDE_correct_train_sequence_l1515_151595


namespace NUMINAMATH_CALUDE_seedling_count_l1515_151571

theorem seedling_count (packets : ℕ) (seeds_per_packet : ℕ) 
  (h1 : packets = 60) (h2 : seeds_per_packet = 7) :
  packets * seeds_per_packet = 420 := by
  sorry

end NUMINAMATH_CALUDE_seedling_count_l1515_151571


namespace NUMINAMATH_CALUDE_earring_ratio_l1515_151597

theorem earring_ratio (bella_earrings monica_earrings rachel_earrings : ℕ) :
  bella_earrings = 10 ∧
  bella_earrings = monica_earrings / 4 ∧
  bella_earrings + monica_earrings + rachel_earrings = 70 →
  monica_earrings / rachel_earrings = 2 := by
  sorry

end NUMINAMATH_CALUDE_earring_ratio_l1515_151597


namespace NUMINAMATH_CALUDE_necklace_profit_is_1500_l1515_151561

/-- Calculates the profit from selling necklaces --/
def calculate_profit (charms_per_necklace : ℕ) (cost_per_charm : ℕ) (selling_price : ℕ) (necklaces_sold : ℕ) : ℕ :=
  let cost_per_necklace := charms_per_necklace * cost_per_charm
  let profit_per_necklace := selling_price - cost_per_necklace
  profit_per_necklace * necklaces_sold

/-- Proves that the profit from selling 30 necklaces is $1500 --/
theorem necklace_profit_is_1500 :
  calculate_profit 10 15 200 30 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_necklace_profit_is_1500_l1515_151561


namespace NUMINAMATH_CALUDE_inequality_solution_l1515_151598

theorem inequality_solution (x : ℝ) :
  x ≤ 4 ∧ |2*x - 3| + |x + 1| < 7 → -5/3 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1515_151598


namespace NUMINAMATH_CALUDE_allie_toys_count_l1515_151544

/-- Proves that given a set of toys with specified values, the total number of toys is correct -/
theorem allie_toys_count (total_worth : ℕ) (special_toy_worth : ℕ) (regular_toy_worth : ℕ) :
  total_worth = 52 →
  special_toy_worth = 12 →
  regular_toy_worth = 5 →
  ∃ (n : ℕ), n * regular_toy_worth + special_toy_worth = total_worth ∧ n + 1 = 9 :=
by sorry

end NUMINAMATH_CALUDE_allie_toys_count_l1515_151544


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l1515_151522

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  longerBase : ℝ
  baseAngle : ℝ
  height : ℝ

/-- Calculate the area of the isosceles trapezoid -/
def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific trapezoid is 100 -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    longerBase := 20,
    baseAngle := Real.arcsin 0.6,
    height := 9
  }
  trapezoidArea t = 100 := by sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l1515_151522


namespace NUMINAMATH_CALUDE_trishul_investment_percentage_l1515_151545

/-- Represents the investment amounts of Vishal, Trishul, and Raghu -/
structure Investments where
  vishal : ℝ
  trishul : ℝ
  raghu : ℝ

/-- The conditions of the investment problem -/
def InvestmentConditions (i : Investments) : Prop :=
  i.vishal = 1.1 * i.trishul ∧
  i.raghu = 2300 ∧
  i.vishal + i.trishul + i.raghu = 6647

/-- The theorem stating that Trishul invested 10% less than Raghu -/
theorem trishul_investment_percentage (i : Investments) 
  (h : InvestmentConditions i) : 
  (i.raghu - i.trishul) / i.raghu = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_trishul_investment_percentage_l1515_151545


namespace NUMINAMATH_CALUDE_initial_number_of_girls_l1515_151537

/-- The number of girls in the initial group -/
def n : ℕ := sorry

/-- The initial average weight of the girls -/
def A : ℝ := sorry

/-- The weight of the new girl -/
def new_weight : ℝ := 100

/-- The weight of the girl being replaced -/
def replaced_weight : ℝ := 50

/-- The increase in average weight -/
def avg_increase : ℝ := 5

theorem initial_number_of_girls :
  (n * A - replaced_weight + new_weight) / n = A + avg_increase →
  n = 10 := by sorry

end NUMINAMATH_CALUDE_initial_number_of_girls_l1515_151537


namespace NUMINAMATH_CALUDE_students_travel_speed_l1515_151540

/-- Proves that given the conditions of the problem, student B's bicycle speed is 14.4 km/h -/
theorem students_travel_speed (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ)
  (h_distance : distance = 2.4)
  (h_speed_ratio : speed_ratio = 4)
  (h_time_difference : time_difference = 0.5) :
  let walking_speed := distance / (distance / (speed_ratio * walking_speed) + time_difference)
  speed_ratio * walking_speed = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_students_travel_speed_l1515_151540


namespace NUMINAMATH_CALUDE_triangle_side_length_l1515_151583

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a = 2 → B = π / 3 → c = 3 → b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1515_151583


namespace NUMINAMATH_CALUDE_triangular_array_coin_sum_l1515_151580

theorem triangular_array_coin_sum (N : ℕ) : 
  (N * (N + 1)) / 2 = 2016 → (N / 10 + N % 10 = 9) := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_coin_sum_l1515_151580


namespace NUMINAMATH_CALUDE_ship_cargo_after_loading_l1515_151558

/-- The total cargo on a ship after loading additional cargo in the Bahamas -/
theorem ship_cargo_after_loading (initial_cargo additional_cargo : ℕ) :
  initial_cargo = 5973 →
  additional_cargo = 8723 →
  initial_cargo + additional_cargo = 14696 := by
  sorry

end NUMINAMATH_CALUDE_ship_cargo_after_loading_l1515_151558


namespace NUMINAMATH_CALUDE_product_of_real_and_imaginary_parts_l1515_151577

theorem product_of_real_and_imaginary_parts : ∃ (z : ℂ), 
  z = (2 + 3*Complex.I) / (1 + Complex.I) ∧ 
  (z.re * z.im = 5/4) := by sorry

end NUMINAMATH_CALUDE_product_of_real_and_imaginary_parts_l1515_151577


namespace NUMINAMATH_CALUDE_exactly_three_sequences_l1515_151552

/-- Represents a sequence of 10 positive integers -/
def Sequence := Fin 10 → ℕ+

/-- Checks if a sequence satisfies the recurrence relation -/
def satisfies_recurrence (s : Sequence) : Prop :=
  ∀ n : Fin 8, s (n.succ.succ) = s (n.succ) + s n

/-- Checks if a sequence has the required last term -/
def has_correct_last_term (s : Sequence) : Prop :=
  s 9 = 2002

/-- The main theorem stating that there are exactly 3 valid sequences -/
theorem exactly_three_sequences :
  ∃! (sequences : Finset Sequence),
    sequences.card = 3 ∧
    ∀ s ∈ sequences, satisfies_recurrence s ∧ has_correct_last_term s :=
sorry

end NUMINAMATH_CALUDE_exactly_three_sequences_l1515_151552


namespace NUMINAMATH_CALUDE_apple_lovers_joined_correct_number_joined_l1515_151526

theorem apple_lovers_joined (total_apples : ℕ) (initial_per_person : ℕ) (decrease : ℕ) : ℕ :=
  let initial_group_size := total_apples / initial_per_person
  let final_per_person := initial_per_person - decrease
  let final_group_size := total_apples / final_per_person
  final_group_size - initial_group_size

theorem correct_number_joined :
  apple_lovers_joined 1430 22 9 = 45 :=
by sorry

end NUMINAMATH_CALUDE_apple_lovers_joined_correct_number_joined_l1515_151526


namespace NUMINAMATH_CALUDE_carrot_count_l1515_151586

theorem carrot_count (initial_carrots thrown_out_carrots picked_next_day : ℕ) :
  initial_carrots = 48 →
  thrown_out_carrots = 11 →
  picked_next_day = 15 →
  initial_carrots - thrown_out_carrots + picked_next_day = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_carrot_count_l1515_151586


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l1515_151536

theorem quadratic_form_equivalence (b : ℝ) (h1 : b > 0) :
  (∃ n : ℝ, ∀ x : ℝ, x^2 + b*x + 36 = (x + n)^2 + 20) → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l1515_151536


namespace NUMINAMATH_CALUDE_only_14_satisfies_l1515_151506

def is_multiple_of_three (n : ℤ) : Prop := ∃ k : ℤ, n = 3 * k

def is_perfect_square (n : ℤ) : Prop := ∃ k : ℤ, n = k * k

def sum_of_digits (n : ℤ) : ℕ :=
  (n.natAbs.repr.toList.map (λ c => c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

def satisfies_conditions (n : ℤ) : Prop :=
  ¬(is_multiple_of_three n) ∧
  ¬(is_perfect_square n) ∧
  is_prime (sum_of_digits n)

theorem only_14_satisfies :
  satisfies_conditions 14 ∧
  ¬(satisfies_conditions 12) ∧
  ¬(satisfies_conditions 16) ∧
  ¬(satisfies_conditions 21) ∧
  ¬(satisfies_conditions 26) :=
sorry

end NUMINAMATH_CALUDE_only_14_satisfies_l1515_151506


namespace NUMINAMATH_CALUDE_vanessa_missed_days_l1515_151589

/-- Represents the number of days missed by each student -/
structure MissedDays where
  vanessa : ℕ
  mike : ℕ
  sarah : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (d : MissedDays) : Prop :=
  d.vanessa + d.mike + d.sarah = 17 ∧
  d.vanessa + d.mike = 14 ∧
  d.mike + d.sarah = 12

/-- The theorem to prove -/
theorem vanessa_missed_days (d : MissedDays) (h : satisfiesConditions d) : d.vanessa = 5 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_missed_days_l1515_151589


namespace NUMINAMATH_CALUDE_leg_head_difference_l1515_151585

/-- Represents the number of legs for a buffalo -/
def buffalo_legs : ℕ := 4

/-- Represents the number of legs for a duck -/
def duck_legs : ℕ := 2

/-- Represents the number of heads for any animal -/
def animal_head : ℕ := 1

/-- The number of buffaloes in the group -/
def num_buffaloes : ℕ := 12

theorem leg_head_difference (num_ducks : ℕ) :
  (num_buffaloes * buffalo_legs + num_ducks * duck_legs) -
  2 * (num_buffaloes * animal_head + num_ducks * animal_head) = 24 :=
by sorry

end NUMINAMATH_CALUDE_leg_head_difference_l1515_151585


namespace NUMINAMATH_CALUDE_production_proof_l1515_151528

def average_production_problem (n : ℕ) (past_average current_average : ℚ) : Prop :=
  let past_total := n * past_average
  let today_production := (n + 1) * current_average - past_total
  today_production = 95

theorem production_proof :
  average_production_problem 8 50 55 := by
  sorry

end NUMINAMATH_CALUDE_production_proof_l1515_151528


namespace NUMINAMATH_CALUDE_monochromatic_subgrid_exists_l1515_151560

/-- Represents a cell color -/
inductive Color
| Black
| White

/-- Represents the grid -/
def Grid := Fin 3 → Fin 7 → Color

/-- Checks if a 2x2 subgrid has all cells of the same color -/
def has_monochromatic_2x2_subgrid (g : Grid) : Prop :=
  ∃ (i : Fin 2) (j : Fin 6),
    g i j = g i (j + 1) ∧
    g i j = g (i + 1) j ∧
    g i j = g (i + 1) (j + 1)

/-- Main theorem: Any 3x7 grid with black and white cells contains a monochromatic 2x2 subgrid -/
theorem monochromatic_subgrid_exists (g : Grid) : 
  has_monochromatic_2x2_subgrid g :=
sorry

end NUMINAMATH_CALUDE_monochromatic_subgrid_exists_l1515_151560


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l1515_151501

/-- A two-digit natural number -/
def TwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

/-- The first digit of a two-digit number -/
def firstDigit (n : ℕ) : ℕ := n / 10

/-- The second digit of a two-digit number -/
def secondDigit (n : ℕ) : ℕ := n % 10

/-- The condition given in the problem -/
def satisfiesCondition (n : ℕ) : Prop :=
  4 * (firstDigit n) + 2 * (secondDigit n) = n / 2

theorem two_digit_number_theorem (n : ℕ) :
  TwoDigitNumber n ∧ satisfiesCondition n → n = 32 ∨ n = 64 ∨ n = 96 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l1515_151501


namespace NUMINAMATH_CALUDE_trigonometric_calculations_l1515_151539

theorem trigonometric_calculations :
  (2 * Real.cos (60 * π / 180) + |1 - 2 * Real.sin (45 * π / 180)| + (1/2)^0 = Real.sqrt 2 + 1) ∧
  (Real.sqrt (1 - 2 * Real.tan (60 * π / 180) + Real.tan (60 * π / 180)^2) - Real.tan (60 * π / 180) = -1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_calculations_l1515_151539


namespace NUMINAMATH_CALUDE_inequality_proof_l1515_151563

theorem inequality_proof (a b : ℝ) (ha : a > 1/2) (hb : b > 1/2) :
  a + 2*b - 5*a*b < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1515_151563


namespace NUMINAMATH_CALUDE_square_gt_abs_square_l1515_151546

theorem square_gt_abs_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_gt_abs_square_l1515_151546


namespace NUMINAMATH_CALUDE_distance_ratio_on_rough_terrain_l1515_151509

theorem distance_ratio_on_rough_terrain
  (total_distance : ℝ)
  (speed_ratio : ℝ → ℝ → Prop)
  (rough_terrain_speed : ℝ → ℝ)
  (rough_terrain_length : ℝ)
  (meeting_point : ℝ)
  (h1 : speed_ratio 2 3)
  (h2 : ∀ x, rough_terrain_speed x = x / 2)
  (h3 : rough_terrain_length = 2 / 3 * total_distance)
  (h4 : meeting_point = total_distance / 2) :
  ∃ (d1 d2 : ℝ), d1 + d2 = rough_terrain_length ∧ d1 / d2 = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_distance_ratio_on_rough_terrain_l1515_151509


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1515_151516

theorem isosceles_triangle_perimeter (m : ℝ) :
  (3 : ℝ) ^ 2 - (m + 1) * 3 + 2 * m = 0 →
  ∃ (a b : ℝ),
    a ^ 2 - (m + 1) * a + 2 * m = 0 ∧
    b ^ 2 - (m + 1) * b + 2 * m = 0 ∧
    ((a = b ∧ a + a + b = 10) ∨ (a ≠ b ∧ a + a + b = 11)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1515_151516


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_ratio_l1515_151566

theorem triangle_square_perimeter_ratio : 
  let square_side : ℝ := 4
  let square_perimeter : ℝ := 4 * square_side
  let triangle_leg : ℝ := square_side
  let triangle_hypotenuse : ℝ := square_side * Real.sqrt 2
  let triangle_perimeter : ℝ := 2 * triangle_leg + triangle_hypotenuse
  triangle_perimeter / square_perimeter = 1/2 + Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_ratio_l1515_151566


namespace NUMINAMATH_CALUDE_division_by_negative_fraction_l1515_151555

theorem division_by_negative_fraction :
  5 / (-1/2 : ℚ) = -10 := by sorry

end NUMINAMATH_CALUDE_division_by_negative_fraction_l1515_151555


namespace NUMINAMATH_CALUDE_sqrt_expressions_equality_l1515_151512

theorem sqrt_expressions_equality : 
  (2 * Real.sqrt (2/3) - 3 * Real.sqrt (3/2) + Real.sqrt 24 = (7 * Real.sqrt 6) / 6) ∧
  (Real.sqrt (25/2) + Real.sqrt 32 - Real.sqrt 18 - (Real.sqrt 2 - 1)^2 = (11 * Real.sqrt 2) / 2 - 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_equality_l1515_151512


namespace NUMINAMATH_CALUDE_arithmetic_operations_l1515_151575

theorem arithmetic_operations : 
  (100 - 54 - 46 = 0) ∧ 
  (234 - (134 + 45) = 55) ∧ 
  (125 * 7 * 8 = 7000) ∧ 
  (15 * (61 - 45) = 240) ∧ 
  (318 / 6 + 165 = 218) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l1515_151575


namespace NUMINAMATH_CALUDE_hyperbola_foci_l1515_151590

theorem hyperbola_foci (x y : ℝ) :
  (x^2 / 3 - y^2 / 4 = 1) →
  (∃ f : ℝ, f = Real.sqrt 7 ∧ 
    ((x = f ∧ y = 0) ∨ (x = -f ∧ y = 0)) →
    (x^2 / 3 - y^2 / 4 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l1515_151590


namespace NUMINAMATH_CALUDE_cousins_distribution_eq_67_l1515_151520

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 cousins into 4 rooms -/
def cousins_distribution : ℕ := distribute 5 4

theorem cousins_distribution_eq_67 : cousins_distribution = 67 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_eq_67_l1515_151520


namespace NUMINAMATH_CALUDE_total_chickens_l1515_151518

def farm_animals (ducks rabbits : ℕ) : Prop :=
  ∃ (hens roosters chickens : ℕ),
    hens = ducks + 20 ∧
    roosters = rabbits - 10 ∧
    chickens = hens + roosters ∧
    chickens = 80

theorem total_chickens : farm_animals 40 30 := by
  sorry

end NUMINAMATH_CALUDE_total_chickens_l1515_151518


namespace NUMINAMATH_CALUDE_greatest_number_of_bouquets_l1515_151587

theorem greatest_number_of_bouquets (white_tulips red_tulips : ℕ) 
  (h_white : white_tulips = 21) (h_red : red_tulips = 91) : 
  (∃ (bouquets_count : ℕ) (white_per_bouquet red_per_bouquet : ℕ), 
    bouquets_count * white_per_bouquet = white_tulips ∧ 
    bouquets_count * red_per_bouquet = red_tulips ∧ 
    ∀ (other_count : ℕ) (other_white other_red : ℕ), 
      other_count * other_white = white_tulips → 
      other_count * other_red = red_tulips → 
      other_count ≤ bouquets_count) ∧ 
  (∃ (max_bouquets : ℕ), max_bouquets = 3 ∧ 
    ∀ (bouquets_count : ℕ) (white_per_bouquet red_per_bouquet : ℕ), 
      bouquets_count * white_per_bouquet = white_tulips → 
      bouquets_count * red_per_bouquet = red_tulips → 
      bouquets_count ≤ max_bouquets) := by
sorry

end NUMINAMATH_CALUDE_greatest_number_of_bouquets_l1515_151587


namespace NUMINAMATH_CALUDE_power_equation_solution_l1515_151541

theorem power_equation_solution (n : ℕ) : 5^29 * 4^15 = 2 * 10^n → n = 29 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1515_151541


namespace NUMINAMATH_CALUDE_log_sqrt10_1000_sqrt10_l1515_151568

theorem log_sqrt10_1000_sqrt10 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt10_1000_sqrt10_l1515_151568


namespace NUMINAMATH_CALUDE_SetA_eq_SetB_l1515_151565

/-- Set A: integers representable as x^2 + 2y^2 where x and y are integers -/
def SetA : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 2*y^2}

/-- Set B: integers representable as x^2 + 6xy + 11y^2 where x and y are integers -/
def SetB : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 6*x*y + 11*y^2}

/-- Theorem stating that Set A and Set B are equal -/
theorem SetA_eq_SetB : SetA = SetB := by sorry

end NUMINAMATH_CALUDE_SetA_eq_SetB_l1515_151565


namespace NUMINAMATH_CALUDE_dons_average_speed_l1515_151584

theorem dons_average_speed 
  (ambulance_speed : ℝ) 
  (ambulance_time : ℝ) 
  (don_time : ℝ) 
  (h1 : ambulance_speed = 60) 
  (h2 : ambulance_time = 1/4) 
  (h3 : don_time = 1/2) : 
  (ambulance_speed * ambulance_time) / don_time = 30 := by
sorry

end NUMINAMATH_CALUDE_dons_average_speed_l1515_151584


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1515_151591

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊2 * x⌋ - 1/2⌋ = ⌊x + 2⌋ ↔ 5/2 ≤ x ∧ x < 7/2 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1515_151591


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1515_151550

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem smallest_two_digit_prime_with_composite_reverse :
  ∃ (n : ℕ), is_two_digit n ∧ 
             Nat.Prime n ∧ 
             tens_digit n = 2 ∧ 
             ¬(Nat.Prime (reverse_digits n)) ∧
             (∀ m : ℕ, is_two_digit m → 
                       Nat.Prime m → 
                       tens_digit m = 2 → 
                       ¬(Nat.Prime (reverse_digits m)) → 
                       n ≤ m) ∧
             n = 23 := by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1515_151550


namespace NUMINAMATH_CALUDE_simplify_fraction_multiplication_l1515_151524

theorem simplify_fraction_multiplication : (123 : ℚ) / 9999 * 41 = 1681 / 3333 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_multiplication_l1515_151524


namespace NUMINAMATH_CALUDE_coeff_x3_is_42_l1515_151554

/-- First polynomial: x^5 - 4x^4 + 7x^3 - 5x^2 + 3x - 2 -/
def p1 (x : ℝ) : ℝ := x^5 - 4*x^4 + 7*x^3 - 5*x^2 + 3*x - 2

/-- Second polynomial: 3x^2 - 5x + 6 -/
def p2 (x : ℝ) : ℝ := 3*x^2 - 5*x + 6

/-- The product of the two polynomials -/
def product (x : ℝ) : ℝ := p1 x * p2 x

/-- Theorem: The coefficient of x^3 in the product of p1 and p2 is 42 -/
theorem coeff_x3_is_42 : ∃ (a b c d e f : ℝ), product x = a*x^5 + b*x^4 + 42*x^3 + d*x^2 + e*x + f :=
sorry

end NUMINAMATH_CALUDE_coeff_x3_is_42_l1515_151554


namespace NUMINAMATH_CALUDE_paper_fold_unfold_holes_l1515_151579

/-- Represents a rectangular piece of paper --/
structure Paper where
  height : ℕ
  width : ℕ

/-- Represents the position of a hole on the paper --/
structure HolePosition where
  x : ℕ
  y : ℕ

/-- Represents the state of the paper after folding and hole punching --/
structure FoldedPaper where
  original : Paper
  folds : List (Paper → Paper)
  holePosition : HolePosition

/-- Function to calculate the number and arrangement of holes after unfolding --/
def unfoldAndCount (fp : FoldedPaper) : ℕ × List HolePosition :=
  sorry

/-- Theorem stating the result of folding and unfolding the paper --/
theorem paper_fold_unfold_holes :
  ∀ (fp : FoldedPaper),
    fp.original = Paper.mk 8 12 →
    fp.folds.length = 3 →
    (unfoldAndCount fp).1 = 8 ∧
    (∃ (col1 col2 : ℕ), ∀ (pos : HolePosition),
      pos ∈ (unfoldAndCount fp).2 →
      (pos.x = col1 ∨ pos.x = col2) ∧
      pos.y ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_paper_fold_unfold_holes_l1515_151579


namespace NUMINAMATH_CALUDE_tape_circle_length_l1515_151553

/-- The total length of a circle formed by overlapping tape pieces -/
def circle_length (num_pieces : ℕ) (piece_length : ℝ) (overlap : ℝ) : ℝ :=
  num_pieces * (piece_length - overlap)

/-- Theorem stating the total length of the circle-shaped colored tapes -/
theorem tape_circle_length :
  circle_length 16 10.4 3.5 = 110.4 := by
  sorry

end NUMINAMATH_CALUDE_tape_circle_length_l1515_151553


namespace NUMINAMATH_CALUDE_emily_orange_count_l1515_151519

/-- The number of oranges each person has -/
structure OrangeCount where
  betty : ℕ
  sandra : ℕ
  emily : ℕ

/-- The conditions of the orange distribution problem -/
def orange_distribution (oc : OrangeCount) : Prop :=
  oc.emily = 7 * oc.sandra ∧
  oc.sandra = 3 * oc.betty ∧
  oc.betty = 12

/-- Theorem stating that Emily has 252 oranges given the conditions -/
theorem emily_orange_count (oc : OrangeCount) 
  (h : orange_distribution oc) : oc.emily = 252 := by
  sorry


end NUMINAMATH_CALUDE_emily_orange_count_l1515_151519


namespace NUMINAMATH_CALUDE_cricketer_average_increase_l1515_151513

/-- Represents a cricketer's batting statistics -/
structure CricketerStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (stats : CricketerStats) (newInningRuns : ℕ) : ℚ :=
  (stats.totalRuns + newInningRuns) / (stats.innings + 1)

/-- Theorem: If a cricketer's average increases by 8 after scoring 140 in the 15th inning, 
    the new average is 28 -/
theorem cricketer_average_increase 
  (stats : CricketerStats) 
  (h1 : stats.innings = 14)
  (h2 : newAverage stats 140 = stats.average + 8)
  : newAverage stats 140 = 28 := by
  sorry

#check cricketer_average_increase

end NUMINAMATH_CALUDE_cricketer_average_increase_l1515_151513


namespace NUMINAMATH_CALUDE_paper_cut_end_time_l1515_151527

def minutes_per_cut : ℕ := 3
def rest_minutes : ℕ := 1
def start_time : ℕ := 9 * 60 + 40  -- 9:40 in minutes since midnight
def num_cuts : ℕ := 10

def total_time : ℕ := (num_cuts - 1) * (minutes_per_cut + rest_minutes) + minutes_per_cut

def end_time : ℕ := start_time + total_time

theorem paper_cut_end_time :
  (end_time / 60, end_time % 60) = (10, 19) := by
  sorry

end NUMINAMATH_CALUDE_paper_cut_end_time_l1515_151527


namespace NUMINAMATH_CALUDE_proposition_conjunction_false_perpendicular_lines_condition_converse_equivalence_l1515_151574

-- Statement 1
theorem proposition_conjunction_false :
  ¬(∃ x : ℝ, Real.tan x = 1 ∧ ¬(∀ x : ℝ, x^2 + 1 > 0)) := by sorry

-- Statement 2
theorem perpendicular_lines_condition :
  ∃ a b : ℝ, (∀ x y : ℝ, a * x + 3 * y - 1 = 0 ↔ x + b * y + 1 = 0) ∧
             (a * 1 + b * 3 = 0) ∧
             (a / b ≠ -3) := by sorry

-- Statement 3
theorem converse_equivalence :
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) := by sorry

end NUMINAMATH_CALUDE_proposition_conjunction_false_perpendicular_lines_condition_converse_equivalence_l1515_151574


namespace NUMINAMATH_CALUDE_complex_magnitude_l1515_151525

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1515_151525


namespace NUMINAMATH_CALUDE_lighter_box_weight_l1515_151543

/-- Given a shipment of boxes with the following properties:
  * There are 30 boxes in total
  * Some boxes weigh W pounds (lighter boxes)
  * The rest of the boxes weigh 20 pounds (heavier boxes)
  * The initial average weight is 18 pounds
  * After removing 15 of the 20-pound boxes, the new average weight is 16 pounds
  Prove that the weight of the lighter boxes (W) is 16 pounds. -/
theorem lighter_box_weight (total_boxes : ℕ) (W : ℝ) (heavy_box_weight : ℝ) 
  (initial_avg : ℝ) (new_avg : ℝ) (removed_boxes : ℕ) :
  total_boxes = 30 →
  heavy_box_weight = 20 →
  initial_avg = 18 →
  new_avg = 16 →
  removed_boxes = 15 →
  (∃ (light_boxes heavy_boxes : ℕ), 
    light_boxes + heavy_boxes = total_boxes ∧
    (light_boxes * W + heavy_boxes * heavy_box_weight) / total_boxes = initial_avg ∧
    ((light_boxes * W + (heavy_boxes - removed_boxes) * heavy_box_weight) / 
      (total_boxes - removed_boxes) = new_avg)) →
  W = 16 := by
sorry

end NUMINAMATH_CALUDE_lighter_box_weight_l1515_151543


namespace NUMINAMATH_CALUDE_ratio_of_trigonometric_equation_l1515_151573

theorem ratio_of_trigonometric_equation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a * Real.sin (π / 5) + b * Real.cos (π / 5)) / (a * Real.cos (π / 5) - b * Real.sin (π / 5)) = Real.tan (8 * π / 15)) : 
  b / a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_trigonometric_equation_l1515_151573


namespace NUMINAMATH_CALUDE_correct_observation_value_l1515_151510

theorem correct_observation_value (n : ℕ) (initial_mean corrected_mean wrong_value : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : corrected_mean = 36.5)
  (h4 : wrong_value = 23) :
  let total_sum := n * initial_mean
  let corrected_sum := n * corrected_mean
  corrected_sum = total_sum - wrong_value + (total_sum - wrong_value + corrected_sum - total_sum) := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l1515_151510


namespace NUMINAMATH_CALUDE_circle_area_difference_l1515_151576

theorem circle_area_difference (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 24) (h₂ : r₂ = 36) :
  r₃ ^ 2 * π = (r₂ ^ 2 - r₁ ^ 2) * π → r₃ = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1515_151576


namespace NUMINAMATH_CALUDE_circle_square_area_ratio_l1515_151592

theorem circle_square_area_ratio :
  ∀ (r : ℝ) (s : ℝ),
  r > 0 →
  s > 0 →
  r = s * (Real.sqrt 2) / 2 →
  (π * r^2) / (s^2) = π / 2 := by
sorry

end NUMINAMATH_CALUDE_circle_square_area_ratio_l1515_151592


namespace NUMINAMATH_CALUDE_cost_difference_l1515_151570

def candy_bar_cost : ℝ := 6
def chocolate_cost : ℝ := 3

theorem cost_difference : candy_bar_cost - chocolate_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l1515_151570


namespace NUMINAMATH_CALUDE_system_solutions_correct_l1515_151593

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℝ, 3 * x + 2 * y = 10 ∧ x / 2 - (y + 1) / 3 = 1 ∧ x = 3 ∧ y = 1/2) ∧
  -- System 2
  (∃ x y : ℝ, 4 * x - 5 * y = 3 ∧ (x - 2 * y) / 0.4 = 0.6 ∧ x = 1.6 ∧ y = 0.68) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l1515_151593


namespace NUMINAMATH_CALUDE_marble_problem_l1515_151500

theorem marble_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) (emma : ℚ)
  (h1 : angela = a)
  (h2 : brian = 2 * a)
  (h3 : caden = 3 * brian)
  (h4 : daryl = 5 * caden)
  (h5 : emma = 2 * daryl)
  (h6 : angela + brian + caden + daryl + emma = 212) :
  a = 212 / 99 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l1515_151500


namespace NUMINAMATH_CALUDE_bottle_cap_boxes_l1515_151542

theorem bottle_cap_boxes (total_caps : ℕ) (caps_per_box : ℕ) (h1 : total_caps = 316) (h2 : caps_per_box = 4) :
  total_caps / caps_per_box = 79 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_boxes_l1515_151542


namespace NUMINAMATH_CALUDE_perfect_non_spiral_shells_l1515_151556

theorem perfect_non_spiral_shells (total_perfect : ℕ) (total_broken : ℕ) 
  (h1 : total_perfect = 17)
  (h2 : total_broken = 52)
  (h3 : total_broken / 2 = total_broken - total_broken / 2)  -- Half of broken shells are spiral
  (h4 : total_broken / 2 = (total_perfect - (total_perfect - (total_broken / 2 - 21))) + 21) :
  total_perfect - (total_perfect - (total_broken / 2 - 21)) = 12 := by
  sorry

#check perfect_non_spiral_shells

end NUMINAMATH_CALUDE_perfect_non_spiral_shells_l1515_151556


namespace NUMINAMATH_CALUDE_parent_chaperones_count_l1515_151564

/-- The number of parent chaperones on a school field trip -/
def num_parent_chaperones (total_students : ℕ) (num_teachers : ℕ) (students_left : ℕ) (chaperones_left : ℕ) (remaining_individuals : ℕ) : ℕ :=
  (remaining_individuals + students_left + chaperones_left) - (total_students + num_teachers)

theorem parent_chaperones_count :
  num_parent_chaperones 20 2 10 2 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_parent_chaperones_count_l1515_151564


namespace NUMINAMATH_CALUDE_product_remainder_theorem_l1515_151572

theorem product_remainder_theorem (x : ℤ) : 
  (37 * x) % 31 = 15 ↔ ∃ k : ℤ, x = 18 + 31 * k := by sorry

end NUMINAMATH_CALUDE_product_remainder_theorem_l1515_151572


namespace NUMINAMATH_CALUDE_point_outside_circle_l1515_151549

theorem point_outside_circle (a b : ℝ) 
  (h : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y = 1) : 
  a^2 + b^2 > 1 :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1515_151549


namespace NUMINAMATH_CALUDE_union_S_T_l1515_151548

def U : Finset Nat := {1,2,3,4,5,6}
def S : Finset Nat := {1,3,5}
def T : Finset Nat := {2,3,4,5}

theorem union_S_T : S ∪ T = {1,2,3,4,5} := by sorry

end NUMINAMATH_CALUDE_union_S_T_l1515_151548


namespace NUMINAMATH_CALUDE_jerry_weighted_mean_l1515_151521

/-- Represents different currencies --/
inductive Currency
| USD
| EUR
| GBP
| CAD

/-- Represents a monetary amount with its currency --/
structure Money where
  amount : Float
  currency : Currency

/-- Represents a gift with its source --/
structure Gift where
  amount : Money
  isFamily : Bool

/-- Exchange rates to USD --/
def exchangeRate (c : Currency) : Float :=
  match c with
  | Currency.USD => 1
  | Currency.EUR => 1.20
  | Currency.GBP => 1.38
  | Currency.CAD => 0.82

/-- Convert Money to USD --/
def toUSD (m : Money) : Float :=
  m.amount * exchangeRate m.currency

/-- List of all gifts Jerry received --/
def jerryGifts : List Gift := [
  { amount := { amount := 9.73, currency := Currency.USD }, isFamily := true },
  { amount := { amount := 9.43, currency := Currency.EUR }, isFamily := true },
  { amount := { amount := 22.16, currency := Currency.USD }, isFamily := false },
  { amount := { amount := 23.51, currency := Currency.USD }, isFamily := false },
  { amount := { amount := 18.72, currency := Currency.EUR }, isFamily := false },
  { amount := { amount := 15.53, currency := Currency.GBP }, isFamily := false },
  { amount := { amount := 22.84, currency := Currency.USD }, isFamily := false },
  { amount := { amount := 7.25, currency := Currency.USD }, isFamily := true },
  { amount := { amount := 20.37, currency := Currency.CAD }, isFamily := true }
]

/-- Calculate weighted mean of Jerry's gifts --/
def weightedMean (gifts : List Gift) : Float :=
  let familyGifts := gifts.filter (λ g => g.isFamily)
  let friendGifts := gifts.filter (λ g => ¬g.isFamily)
  let familySum := familyGifts.foldl (λ acc g => acc + toUSD g.amount) 0
  let friendSum := friendGifts.foldl (λ acc g => acc + toUSD g.amount) 0
  familySum * 0.4 + friendSum * 0.6

/-- Theorem: The weighted mean of Jerry's birthday money in USD is $85.4442 --/
theorem jerry_weighted_mean :
  weightedMean jerryGifts = 85.4442 := by
  sorry

end NUMINAMATH_CALUDE_jerry_weighted_mean_l1515_151521


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l1515_151569

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- State the theorem
theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l1515_151569


namespace NUMINAMATH_CALUDE_set_intersection_complement_l1515_151594

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := {1, 4}

def N : Set ℕ := {1, 3, 5}

theorem set_intersection_complement :
  N ∩ (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_set_intersection_complement_l1515_151594


namespace NUMINAMATH_CALUDE_segment_ratio_l1515_151562

/-- Given a line segment AD with points B and C on it, where AB = 3BD and AC = 5CD,
    the length of BC is 1/12 of the length of AD. -/
theorem segment_ratio (A B C D : ℝ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : C ≤ D)
  (h4 : B - A = 3 * (D - B)) (h5 : C - A = 5 * (D - C)) :
  (C - B) = (1 / 12) * (D - A) := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l1515_151562


namespace NUMINAMATH_CALUDE_least_k_for_convergence_l1515_151559

def u : ℕ → ℚ
  | 0 => 1/4
  | n + 1 => 2 * u n - 2 * (u n)^2

def L : ℚ := 1/2

theorem least_k_for_convergence :
  (∀ k : ℕ, k < 10 → |u k - L| > 1/2^1000) ∧
  |u 10 - L| ≤ 1/2^1000 := by sorry

end NUMINAMATH_CALUDE_least_k_for_convergence_l1515_151559


namespace NUMINAMATH_CALUDE_configuration_contains_triangle_l1515_151578

/-- A configuration of points and segments on a plane. -/
structure Configuration (n : ℕ) where
  points : Fin (2*n) → Plane
  segments : Finset (Fin (2*n) × Fin (2*n))
  segment_count : segments.card = n^2 + 1

/-- A triangle in the configuration. -/
def Triangle (config : Configuration n) :=
  ∃ (a b c : Fin (2*n)), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a, b) ∈ config.segments ∧ (b, c) ∈ config.segments ∧ (a, c) ∈ config.segments

/-- The main theorem: any configuration with n^2 + 1 segments contains a triangle. -/
theorem configuration_contains_triangle (n : ℕ) (h : n ≥ 2) (config : Configuration n) :
  Triangle config :=
sorry

end NUMINAMATH_CALUDE_configuration_contains_triangle_l1515_151578


namespace NUMINAMATH_CALUDE_gcf_of_75_and_100_l1515_151505

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_100_l1515_151505


namespace NUMINAMATH_CALUDE_partition_natural_numbers_l1515_151534

theorem partition_natural_numbers : 
  ∃ (partition : ℕ → Fin 100), 
    (∀ i : Fin 100, ∃ n : ℕ, partition n = i) ∧ 
    (∀ a b c : ℕ, a + 99 * b = c → 
      partition a = partition c ∨ 
      partition a = partition b ∨ 
      partition b = partition c) :=
sorry

end NUMINAMATH_CALUDE_partition_natural_numbers_l1515_151534


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1515_151547

theorem scientific_notation_equivalence : 22000000 = 2.2 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1515_151547


namespace NUMINAMATH_CALUDE_sector_area_l1515_151502

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 4) (h2 : θ = π / 4) :
  (1 / 2) * θ * r^2 = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1515_151502


namespace NUMINAMATH_CALUDE_oplus_2_4_1_3_l1515_151503

-- Define the # operation
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Define the ⊕ operation
def oplus (a b c d : ℝ) : ℝ := hash a (b + d) c - hash a b c

-- Theorem statement
theorem oplus_2_4_1_3 : oplus 2 4 1 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_oplus_2_4_1_3_l1515_151503


namespace NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l1515_151582

/-- 
Given two equations relating velocity (V), displacement (S), time (t), 
acceleration (g), and initial velocity (V₀), prove that t can be expressed 
in terms of S, V, and V₀.
-/
theorem time_from_velocity_and_displacement 
  (V g t V₀ S : ℝ) 
  (hV : V = g * t + V₀) 
  (hS : S = (1/2) * g * t^2 + V₀ * t) : 
  t = 2 * S / (V + V₀) := by
sorry

end NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l1515_151582


namespace NUMINAMATH_CALUDE_focus_of_our_parabola_l1515_151531

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola x^2 = 4y -/
def our_parabola : Parabola :=
  { equation := fun x y => x^2 = 4*y }

theorem focus_of_our_parabola :
  focus our_parabola = (0, 1) := by sorry

end NUMINAMATH_CALUDE_focus_of_our_parabola_l1515_151531


namespace NUMINAMATH_CALUDE_pencil_difference_l1515_151508

/-- The price of a single pencil in dollars -/
def pencil_price : ℚ := 0.04

/-- The number of pencils Jamar bought -/
def jamar_pencils : ℕ := 81

/-- The number of pencils Michael bought -/
def michael_pencils : ℕ := 104

/-- The amount Jamar paid in dollars -/
def jamar_paid : ℚ := 2.32

/-- The amount Michael paid in dollars -/
def michael_paid : ℚ := 3.24

theorem pencil_difference : 
  (pencil_price > 0.01) ∧ 
  (jamar_paid = pencil_price * jamar_pencils) ∧
  (michael_paid = pencil_price * michael_pencils) ∧
  (∃ n : ℕ, n^2 = jamar_pencils) →
  michael_pencils - jamar_pencils = 23 := by
sorry

end NUMINAMATH_CALUDE_pencil_difference_l1515_151508


namespace NUMINAMATH_CALUDE_card_arrangement_probability_l1515_151599

/-- The probability of arranging cards to form a specific word --/
theorem card_arrangement_probability (n : ℕ) (n1 n2 : ℕ) (h1 : n = n1 + n2) (h2 : n1 = 2) (h3 : n2 = 3) :
  (1 : ℚ) / (n.factorial / (n1.factorial * n2.factorial)) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_card_arrangement_probability_l1515_151599


namespace NUMINAMATH_CALUDE_halloween_candy_division_l1515_151515

/-- Represents the fraction of candy taken by each person --/
def candy_fraction (total : ℚ) (remaining : ℚ) (ratio : ℚ) : ℚ :=
  ratio * remaining / total

/-- The problem of dividing Halloween candy --/
theorem halloween_candy_division :
  let total := 1
  let al_ratio := 4 / 10
  let bert_ratio := 3 / 10
  let carl_ratio := 2 / 10
  let dana_ratio := 1 / 10
  
  let al_takes := candy_fraction total total al_ratio
  let bert_takes := candy_fraction total (total - al_takes) bert_ratio
  let carl_takes := candy_fraction total (total - al_takes - bert_takes) carl_ratio
  let dana_takes := candy_fraction total (total - al_takes - bert_takes - carl_takes) dana_ratio
  
  total - (al_takes + bert_takes + carl_takes + dana_takes) = 27 / 125 :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_division_l1515_151515


namespace NUMINAMATH_CALUDE_simplify_product_l1515_151511

theorem simplify_product : (625 : ℝ) ^ (1/4) * (343 : ℝ) ^ (1/3) = 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_l1515_151511


namespace NUMINAMATH_CALUDE_calculate_expression_l1515_151557

theorem calculate_expression : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1515_151557


namespace NUMINAMATH_CALUDE_movie_collection_size_l1515_151533

theorem movie_collection_size :
  ∀ (dvd blu : ℕ),
  dvd > 0 ∧ blu > 0 →
  dvd / blu = 17 / 4 →
  dvd / (blu - 4) = 9 / 2 →
  dvd + blu = 378 := by
sorry

end NUMINAMATH_CALUDE_movie_collection_size_l1515_151533


namespace NUMINAMATH_CALUDE_kickball_students_total_l1515_151596

theorem kickball_students_total (wednesday : ℕ) (fewer_thursday : ℕ) : 
  wednesday = 37 → fewer_thursday = 9 → 
  wednesday + (wednesday - fewer_thursday) = 65 := by
  sorry

end NUMINAMATH_CALUDE_kickball_students_total_l1515_151596


namespace NUMINAMATH_CALUDE_circle_properties_l1515_151567

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 5 = 0

-- Define the point M
def point_M : ℝ × ℝ := (3, -1)

-- Define the point N
def point_N : ℝ × ℝ := (1, 2)

-- Define the equation of the required circle
def required_circle (x y : ℝ) : Prop := (x - 20/7)^2 + (y - 15/14)^2 = 845/196

-- Theorem statement
theorem circle_properties :
  -- The required circle passes through point M
  required_circle point_M.1 point_M.2 ∧
  -- The required circle passes through point N
  required_circle point_N.1 point_N.2 ∧
  -- The required circle is tangent to circle C at point N
  (∃ (t : ℝ), t ≠ 0 ∧
    ∀ (x y : ℝ),
      circle_C x y ↔ required_circle x y ∨
      ((x - point_N.1) = t * (40/7 - 2*point_N.1) ∧
       (y - point_N.2) = t * (30/7 - 2*point_N.2))) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l1515_151567


namespace NUMINAMATH_CALUDE_fib_50_div_5_l1515_151532

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The 50th Fibonacci number is divisible by 5 -/
theorem fib_50_div_5 : 5 ∣ fib 50 := by sorry

end NUMINAMATH_CALUDE_fib_50_div_5_l1515_151532


namespace NUMINAMATH_CALUDE_equal_fractions_imply_one_third_l1515_151588

theorem equal_fractions_imply_one_third (x : ℝ) (h1 : x > 0) 
  (h2 : (2/3) * x = (16/216) * (1/x)) : x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equal_fractions_imply_one_third_l1515_151588


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l1515_151507

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate for angles forming an arithmetic progression -/
def angles_in_arithmetic_progression (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

/-- Predicate for sides forming a geometric progression -/
def sides_in_geometric_progression (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c

/-- Theorem stating that a triangle with angles in arithmetic progression
    and sides in geometric progression is equilateral -/
theorem triangle_is_equilateral (t : Triangle)
  (h1 : angles_in_arithmetic_progression t)
  (h2 : sides_in_geometric_progression t) :
  t.A = 60 ∧ t.B = 60 ∧ t.C = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l1515_151507


namespace NUMINAMATH_CALUDE_pascal_triangle_47_l1515_151514

/-- Pascal's Triangle contains the number 47 in exactly one row -/
theorem pascal_triangle_47 (p : ℕ) (h_prime : Nat.Prime p) (h_p : p = 47) : 
  (∃! n : ℕ, ∃ k : ℕ, Nat.choose n k = p) :=
sorry

end NUMINAMATH_CALUDE_pascal_triangle_47_l1515_151514
