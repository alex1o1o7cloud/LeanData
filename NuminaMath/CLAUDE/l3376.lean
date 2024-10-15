import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_sum_base8_l3376_337626

/-- The units digit of the sum of two numbers in base 8 -/
def unitsDigitBase8 (a b : ℕ) : ℕ :=
  (a + b) % 8

/-- 35 in base 8 -/
def a : ℕ := 3 * 8 + 5

/-- 47 in base 8 -/
def b : ℕ := 4 * 8 + 7

theorem units_digit_sum_base8 :
  unitsDigitBase8 a b = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_base8_l3376_337626


namespace NUMINAMATH_CALUDE_durand_more_likely_to_win_l3376_337634

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of ways to roll a sum of 7 with two dice -/
def ways_to_roll_7 : ℕ := 6

/-- The number of ways to roll a sum of 8 with two dice -/
def ways_to_roll_8 : ℕ := 5

/-- The probability of rolling a sum of 7 with two dice -/
def prob_7 : ℚ := ways_to_roll_7 / total_outcomes

/-- The probability of rolling a sum of 8 with two dice -/
def prob_8 : ℚ := ways_to_roll_8 / total_outcomes

theorem durand_more_likely_to_win : prob_7 > prob_8 := by
  sorry

end NUMINAMATH_CALUDE_durand_more_likely_to_win_l3376_337634


namespace NUMINAMATH_CALUDE_product_place_value_l3376_337669

theorem product_place_value : 
  (216 * 5 ≥ 1000 ∧ 216 * 5 < 10000) ∧ 
  (126 * 5 ≥ 100 ∧ 126 * 5 < 1000) := by
  sorry

end NUMINAMATH_CALUDE_product_place_value_l3376_337669


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3376_337605

theorem inequality_solution_set (a : ℝ) (h : a^3 < a ∧ a < a^2) :
  {x : ℝ | x + a > 1 - a * x} = {x : ℝ | x < (1 - a) / (1 + a)} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3376_337605


namespace NUMINAMATH_CALUDE_min_cubes_for_specific_box_l3376_337697

/-- Calculates the minimum number of cubes required to build a box -/
def min_cubes_for_box (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

/-- Theorem stating the minimum number of cubes required for the specific box -/
theorem min_cubes_for_specific_box :
  min_cubes_for_box 10 18 4 12 = 60 := by
  sorry

#eval min_cubes_for_box 10 18 4 12

end NUMINAMATH_CALUDE_min_cubes_for_specific_box_l3376_337697


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3376_337657

theorem largest_integer_satisfying_inequality :
  ∃ (n : ℕ), n > 0 ∧ n^200 < 5^300 ∧ ∀ (m : ℕ), m > n → m^200 ≥ 5^300 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3376_337657


namespace NUMINAMATH_CALUDE_orange_flowers_killed_l3376_337618

/-- Represents the number of flowers of each color --/
structure FlowerCount where
  red : ℕ
  yellow : ℕ
  orange : ℕ
  purple : ℕ

/-- Represents the number of flowers killed by fungus for each color --/
structure FungusKilled where
  red : ℕ
  yellow : ℕ
  orange : ℕ
  purple : ℕ

def flowersPerBouquet : ℕ := 9
def seedsPlanted : ℕ := 125
def bouquetsMade : ℕ := 36

def totalFlowersNeeded : ℕ := flowersPerBouquet * bouquetsMade

def fungusKilled : FungusKilled := {
  red := 45,
  yellow := 61,
  orange := 0,  -- This is what we need to prove
  purple := 40
}

def survivingFlowers : FlowerCount := {
  red := seedsPlanted - fungusKilled.red,
  yellow := seedsPlanted - fungusKilled.yellow,
  orange := seedsPlanted - fungusKilled.orange,
  purple := seedsPlanted - fungusKilled.purple
}

theorem orange_flowers_killed (x : ℕ) :
  x = fungusKilled.orange →
  x = 30 ∧
  totalFlowersNeeded = survivingFlowers.red + survivingFlowers.yellow + survivingFlowers.orange + survivingFlowers.purple :=
by sorry

end NUMINAMATH_CALUDE_orange_flowers_killed_l3376_337618


namespace NUMINAMATH_CALUDE_sheila_attend_picnic_l3376_337651

/-- The probability of rain tomorrow -/
def rain_prob : ℝ := 0.5

/-- The probability Sheila decides to go if it rains -/
def go_if_rain : ℝ := 0.4

/-- The probability Sheila decides to go if it's sunny -/
def go_if_sunny : ℝ := 0.9

/-- The probability Sheila finishes her homework -/
def finish_homework : ℝ := 0.7

/-- The overall probability that Sheila attends the picnic -/
def attend_prob : ℝ := rain_prob * go_if_rain * finish_homework + 
                       (1 - rain_prob) * go_if_sunny * finish_homework

theorem sheila_attend_picnic : attend_prob = 0.455 := by
  sorry

end NUMINAMATH_CALUDE_sheila_attend_picnic_l3376_337651


namespace NUMINAMATH_CALUDE_larger_number_problem_l3376_337676

theorem larger_number_problem (x y : ℝ) (h1 : x > y) (h2 : x + y = 30) (h3 : 2 * y - x = 6) : x = 18 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3376_337676


namespace NUMINAMATH_CALUDE_motorboat_distance_l3376_337653

theorem motorboat_distance (boat_speed : ℝ) (time_with_current time_against_current : ℝ) :
  boat_speed = 10 →
  time_with_current = 2 →
  time_against_current = 3 →
  ∃ (distance current_speed : ℝ),
    distance = (boat_speed + current_speed) * time_with_current ∧
    distance = (boat_speed - current_speed) * time_against_current ∧
    distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_motorboat_distance_l3376_337653


namespace NUMINAMATH_CALUDE_dice_roll_sums_theorem_l3376_337615

/-- Represents the possible movements based on dice rolls -/
inductive Movement
  | West : Movement
  | East : Movement
  | North : Movement
  | South : Movement
  | Stay : Movement
  | NorthThree : Movement

/-- Converts a die roll to a movement -/
def rollToMovement (roll : Nat) : Movement :=
  match roll with
  | 1 => Movement.West
  | 2 => Movement.East
  | 3 => Movement.North
  | 4 => Movement.South
  | 5 => Movement.Stay
  | 6 => Movement.NorthThree
  | _ => Movement.Stay  -- Default case

/-- Represents a position in 2D space -/
structure Position where
  x : Int
  y : Int

/-- Updates the position based on a movement -/
def updatePosition (pos : Position) (mov : Movement) : Position :=
  match mov with
  | Movement.West => ⟨pos.x - 1, pos.y⟩
  | Movement.East => ⟨pos.x + 1, pos.y⟩
  | Movement.North => ⟨pos.x, pos.y + 1⟩
  | Movement.South => ⟨pos.x, pos.y - 1⟩
  | Movement.Stay => pos
  | Movement.NorthThree => ⟨pos.x, pos.y + 3⟩

/-- Calculates the final position after a sequence of rolls -/
def finalPosition (rolls : List Nat) : Position :=
  rolls.foldl (fun pos roll => updatePosition pos (rollToMovement roll)) ⟨0, 0⟩

/-- Theorem: Given the movement rules and final position of 1 km east,
    the possible sums of five dice rolls are 12, 15, 18, 22, and 25 -/
theorem dice_roll_sums_theorem (rolls : List Nat) :
  rolls.length = 5 ∧ 
  (finalPosition rolls).x = 1 ∧ 
  (finalPosition rolls).y = 0 →
  rolls.sum ∈ [12, 15, 18, 22, 25] :=
sorry


end NUMINAMATH_CALUDE_dice_roll_sums_theorem_l3376_337615


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3376_337659

theorem linear_equation_solution (a b : ℝ) :
  (a ≠ 0 → ∃! x : ℝ, a * x + b = 0 ∧ x = -b / a) ∧
  (a = 0 ∧ b = 0 → ∀ x : ℝ, a * x + b = 0) ∧
  (a = 0 ∧ b ≠ 0 → ¬∃ x : ℝ, a * x + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3376_337659


namespace NUMINAMATH_CALUDE_equation_roots_l3376_337675

theorem equation_roots : ∃! (s : Set ℝ), s = {x : ℝ | (x^2 + x)^2 - 4*(x^2 + x) - 12 = 0} ∧ s = {-3, 2} := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l3376_337675


namespace NUMINAMATH_CALUDE_trapezoid_QR_squared_l3376_337692

/-- Represents a trapezoid PQRS with specific properties -/
structure Trapezoid where
  PQ : ℝ
  PS : ℝ
  RS : ℝ
  QR : ℝ
  perp_QR_PQ_RS : True  -- QR is perpendicular to PQ and RS
  perp_diagonals : True -- Diagonals PR and QS are perpendicular

/-- The theorem stating the properties of the specific trapezoid and its conclusion -/
theorem trapezoid_QR_squared (T : Trapezoid) 
  (h1 : T.PQ = Real.sqrt 41)
  (h2 : T.PS = Real.sqrt 2001)
  (h3 : T.PQ + T.RS = Real.sqrt 2082) :
  T.QR ^ 2 = 410 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_QR_squared_l3376_337692


namespace NUMINAMATH_CALUDE_kim_class_hours_l3376_337619

def class_hours (initial_classes : ℕ) (hours_per_class : ℕ) (dropped_classes : ℕ) : ℕ :=
  (initial_classes - dropped_classes) * hours_per_class

theorem kim_class_hours : 
  class_hours 4 2 1 = 6 := by sorry

end NUMINAMATH_CALUDE_kim_class_hours_l3376_337619


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3376_337693

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3376_337693


namespace NUMINAMATH_CALUDE_proportion_solutions_l3376_337695

theorem proportion_solutions :
  (∃ x : ℚ, 0.75 / (1/2) = 12 / x ∧ x = 8) ∧
  (∃ x : ℚ, 0.7 / x = 14 / 5 ∧ x = 0.25) ∧
  (∃ x : ℚ, (2/15) / (1/6) = x / (2/3) ∧ x = 8/15) ∧
  (∃ x : ℚ, 4 / 4.5 = x / 27 ∧ x = 24) := by
  sorry

end NUMINAMATH_CALUDE_proportion_solutions_l3376_337695


namespace NUMINAMATH_CALUDE_x_plus_y_equals_six_l3376_337604

theorem x_plus_y_equals_six (x y : ℝ) 
  (h1 : |x| - x + y = 42)
  (h2 : x + |y| + y = 24) :
  x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_six_l3376_337604


namespace NUMINAMATH_CALUDE_dandelion_counts_l3376_337691

/-- Represents the state of dandelions in a meadow on a given day -/
structure DandelionState :=
  (yellow : ℕ)
  (white : ℕ)

/-- Represents the lifecycle of dandelions over three consecutive days -/
structure DandelionLifecycle :=
  (dayBeforeYesterday : DandelionState)
  (yesterday : DandelionState)
  (today : DandelionState)

/-- The theorem statement -/
theorem dandelion_counts 
  (lifecycle : DandelionLifecycle)
  (h1 : lifecycle.yesterday.yellow = 20)
  (h2 : lifecycle.yesterday.white = 14)
  (h3 : lifecycle.today.yellow = 15)
  (h4 : lifecycle.today.white = 11) :
  lifecycle.dayBeforeYesterday.yellow = 25 ∧ 
  lifecycle.today.yellow - (lifecycle.yesterday.white - lifecycle.today.white) = 9 :=
by sorry

end NUMINAMATH_CALUDE_dandelion_counts_l3376_337691


namespace NUMINAMATH_CALUDE_total_container_weight_l3376_337635

def container_weight (steel_weight tin_weight copper_weight aluminum_weight : ℝ) : ℝ :=
  10 * steel_weight + 15 * tin_weight + 12 * copper_weight + 8 * aluminum_weight

theorem total_container_weight :
  ∀ (steel_weight tin_weight copper_weight aluminum_weight : ℝ),
    steel_weight = 2 * tin_weight →
    steel_weight = copper_weight + 20 →
    copper_weight = 90 →
    aluminum_weight = tin_weight + 10 →
    container_weight steel_weight tin_weight copper_weight aluminum_weight = 3525 := by
  sorry

end NUMINAMATH_CALUDE_total_container_weight_l3376_337635


namespace NUMINAMATH_CALUDE_largest_circle_equation_l3376_337699

/-- The standard equation of the circle with the largest radius, which is tangent to a line and has its center at (1, 0) -/
theorem largest_circle_equation (m : ℝ) : 
  ∃ (x y : ℝ), 
    (∀ (x' y' : ℝ), (2 * m * x' - y' - 4 * m + 1 = 0) → 
      ((x' - 1)^2 + y'^2 ≤ (x - 1)^2 + y^2)) ∧ 
    ((x - 1)^2 + y^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_circle_equation_l3376_337699


namespace NUMINAMATH_CALUDE_mollys_age_l3376_337623

/-- Given three friends with a total average age of 40, where Jared is ten years older than Hakimi,
    and Hakimi is 40 years old, prove that Molly is 30 years old. -/
theorem mollys_age (total_average : ℕ) (hakimi_age : ℕ) (jared_age : ℕ) (molly_age : ℕ) : 
  total_average = 40 →
  hakimi_age = 40 →
  jared_age = hakimi_age + 10 →
  (hakimi_age + jared_age + molly_age) / 3 = total_average →
  molly_age = 30 := by
sorry

end NUMINAMATH_CALUDE_mollys_age_l3376_337623


namespace NUMINAMATH_CALUDE_problem_solution_l3376_337683

theorem problem_solution : 
  let a : Float := 0.137
  let b : Float := 0.098
  let c : Float := 0.123
  let d : Float := 0.086
  ((a + b)^2 - (a - b)^2) / (c * d) + (d^3 - c^3) / (a * b * (a + b)) = 4.6886 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3376_337683


namespace NUMINAMATH_CALUDE_nonreal_cube_root_unity_sum_l3376_337603

theorem nonreal_cube_root_unity_sum (ω : ℂ) : 
  ω^3 = 1 ∧ ω ≠ 1 → (1 - 2*ω + 2*ω^2)^6 + (1 + 2*ω - 2*ω^2)^6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_nonreal_cube_root_unity_sum_l3376_337603


namespace NUMINAMATH_CALUDE_s_99_digits_l3376_337685

/-- s(n) is the number formed by concatenating the first n perfect squares -/
def s (n : ℕ) : ℕ := sorry

/-- Count the number of digits in a natural number -/
def countDigits (n : ℕ) : ℕ := sorry

/-- The main theorem: s(99) has 353 digits -/
theorem s_99_digits : countDigits (s 99) = 353 := by sorry

end NUMINAMATH_CALUDE_s_99_digits_l3376_337685


namespace NUMINAMATH_CALUDE_total_amount_is_500_l3376_337601

/-- Calculate the total amount received after applying simple interest -/
def total_amount_with_simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal + (principal * rate * time / 100)

/-- Theorem stating that the total amount received is 500 given the specified conditions -/
theorem total_amount_is_500 :
  total_amount_with_simple_interest 468.75 4 (5/3) = 500 := by
  sorry

#eval total_amount_with_simple_interest 468.75 4 (5/3)

end NUMINAMATH_CALUDE_total_amount_is_500_l3376_337601


namespace NUMINAMATH_CALUDE_travelers_checks_average_l3376_337680

theorem travelers_checks_average (total_checks : ℕ) (total_worth : ℕ) 
  (spent_checks : ℕ) (h1 : total_checks = 30) (h2 : total_worth = 1800) 
  (h3 : spent_checks = 18) :
  let fifty_checks := (2 * total_worth - 100 * total_checks) / 50
  let hundred_checks := total_checks - fifty_checks
  let remaining_fifty := fifty_checks - spent_checks
  let remaining_total := remaining_fifty + hundred_checks
  let remaining_worth := 50 * remaining_fifty + 100 * hundred_checks
  remaining_worth / remaining_total = 75 := by
sorry

end NUMINAMATH_CALUDE_travelers_checks_average_l3376_337680


namespace NUMINAMATH_CALUDE_period_multiple_l3376_337636

/-- A function f is periodic with period l if f(x + l) = f(x) for all x in the domain of f -/
def IsPeriodic (f : ℝ → ℝ) (l : ℝ) : Prop :=
  ∀ x, f (x + l) = f x

/-- If l is a period of f, then nl is also a period of f for any natural number n -/
theorem period_multiple {f : ℝ → ℝ} {l : ℝ} (h : IsPeriodic f l) (n : ℕ) :
  IsPeriodic f (n * l) := by
  sorry


end NUMINAMATH_CALUDE_period_multiple_l3376_337636


namespace NUMINAMATH_CALUDE_baxter_peanut_purchase_l3376_337633

/-- The cost of peanuts per pound -/
def cost_per_pound : ℚ := 3

/-- The minimum purchase in pounds -/
def minimum_purchase : ℚ := 15

/-- The amount Baxter spent on peanuts -/
def amount_spent : ℚ := 105

/-- The number of pounds Baxter purchased over the minimum -/
def pounds_over_minimum : ℚ := (amount_spent / cost_per_pound) - minimum_purchase

theorem baxter_peanut_purchase :
  pounds_over_minimum = 20 :=
by sorry

end NUMINAMATH_CALUDE_baxter_peanut_purchase_l3376_337633


namespace NUMINAMATH_CALUDE_max_remaining_pairwise_sums_l3376_337670

theorem max_remaining_pairwise_sums (a b c d : ℝ) : 
  let sums : List ℝ := [a + b, a + c, a + d, b + c, b + d, c + d]
  (172 ∈ sums) ∧ (305 ∈ sums) ∧ (250 ∈ sums) ∧ (215 ∈ sums) →
  (∃ (x y : ℝ), x ∈ sums ∧ y ∈ sums ∧ x ≠ 172 ∧ x ≠ 305 ∧ x ≠ 250 ∧ x ≠ 215 ∧
                 y ≠ 172 ∧ y ≠ 305 ∧ y ≠ 250 ∧ y ≠ 215 ∧ x ≠ y ∧
                 x + y ≤ 723) ∧
  (∃ (a' b' c' d' : ℝ), 
    let sums' : List ℝ := [a' + b', a' + c', a' + d', b' + c', b' + d', c' + d']
    (172 ∈ sums') ∧ (305 ∈ sums') ∧ (250 ∈ sums') ∧ (215 ∈ sums') ∧
    (∃ (x' y' : ℝ), x' ∈ sums' ∧ y' ∈ sums' ∧ x' ≠ 172 ∧ x' ≠ 305 ∧ x' ≠ 250 ∧ x' ≠ 215 ∧
                     y' ≠ 172 ∧ y' ≠ 305 ∧ y' ≠ 250 ∧ y' ≠ 215 ∧ x' ≠ y' ∧
                     x' + y' = 723)) :=
by sorry

end NUMINAMATH_CALUDE_max_remaining_pairwise_sums_l3376_337670


namespace NUMINAMATH_CALUDE_log_product_equals_three_fourths_l3376_337655

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_product_equals_three_fourths :
  log 4 3 * log 9 8 = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_log_product_equals_three_fourths_l3376_337655


namespace NUMINAMATH_CALUDE_solution_value_l3376_337678

theorem solution_value (a b x y : ℝ) : 
  x = 1 ∧ y = 1 ∧ 
  a * x + b * y = 2 ∧ 
  x - b * y = 3 →
  a - b = 6 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l3376_337678


namespace NUMINAMATH_CALUDE_circle_and_line_equations_l3376_337627

-- Define the circle C
def circle_C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 4}

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 + 4 = 0}

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 - 3}

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem circle_and_line_equations :
  ∃ (center : ℝ × ℝ) (k : ℝ),
    center.1 > 0 ∧ center.2 = 0 ∧
    (∃ (p : ℝ × ℝ), p ∈ circle_C center ∩ tangent_line) ∧
    (∃ (A B : ℝ × ℝ), A ≠ B ∧ A ∈ circle_C center ∩ line_l k ∧ B ∈ circle_C center ∩ line_l k) ∧
    (∀ (A B : ℝ × ℝ), A ∈ circle_C center ∩ line_l k → B ∈ circle_C center ∩ line_l k → dot_product A B = 3) →
    center = (2, 0) ∧ k = 1 := by
  sorry

#check circle_and_line_equations

end NUMINAMATH_CALUDE_circle_and_line_equations_l3376_337627


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3376_337640

/-- Given an arithmetic sequence, if A is the sum of the first n terms
    and B is the sum of the first 2n terms, then the sum of the first 3n terms
    is equal to 3(B - A) -/
theorem arithmetic_sequence_sum (n : ℕ) (A B : ℝ) :
  (∃ a d : ℝ, A = (n : ℝ) / 2 * (2 * a + (n - 1) * d) ∧
               B = (2 * n : ℝ) / 2 * (2 * a + (2 * n - 1) * d)) →
  (3 * n : ℝ) / 2 * (2 * a + (3 * n - 1) * d) = 3 * (B - A) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3376_337640


namespace NUMINAMATH_CALUDE_distance_travelled_downstream_l3376_337625

/-- The distance travelled downstream by a boat -/
theorem distance_travelled_downstream 
  (boat_speed : ℝ) -- Speed of the boat in still water (km/hr)
  (current_speed : ℝ) -- Speed of the current (km/hr)
  (travel_time : ℝ) -- Travel time (minutes)
  (h1 : boat_speed = 20) -- Given boat speed
  (h2 : current_speed = 4) -- Given current speed
  (h3 : travel_time = 24) -- Given travel time
  : (boat_speed + current_speed) * (travel_time / 60) = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_distance_travelled_downstream_l3376_337625


namespace NUMINAMATH_CALUDE_negation_proofs_l3376_337650

-- Define a multi-digit number
def MultiDigitNumber (n : ℕ) : Prop := n ≥ 10

-- Define the last digit of a number
def LastDigit (n : ℕ) : ℕ := n % 10

-- Define divisibility
def Divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem negation_proofs :
  (∃ n : ℕ, MultiDigitNumber n ∧ LastDigit n ≠ 0 ∧ ¬(Divides 5 n)) = False ∧
  (∃ n : ℕ, Even n ∧ ¬(Divides 2 n)) = False :=
by sorry

end NUMINAMATH_CALUDE_negation_proofs_l3376_337650


namespace NUMINAMATH_CALUDE_ellipse_properties_l3376_337672

/-- Definition of the ellipse C -/
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Definition of a point being on the ellipse -/
def OnEllipse (p : ℝ × ℝ) : Prop :=
  Ellipse p.1 p.2

/-- The focus of the ellipse -/
def F : ℝ × ℝ := (1, 0)

/-- A line perpendicular to the x-axis passing through F -/
def PerpendicularLine (y : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 1}

/-- The dot product of two points -/
def DotProduct (p q : ℝ × ℝ) : ℝ :=
  p.1 * q.1 + p.2 * q.2

/-- The circle with diameter MN passes through a fixed point -/
def CirclePassesThroughFixedPoint (m n : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t = 1 ∨ t = 3 ∧ 
    (2 - t)^2 + (m.2 - 0) * (n.2 - 0) / ((m.1 - t) * (n.1 - t)) = 1

theorem ellipse_properties :
  ∀ (a b : ℝ × ℝ),
    OnEllipse a ∧ OnEllipse b ∧
    (∃ y, a ∈ PerpendicularLine y ∧ b ∈ PerpendicularLine y) →
    DotProduct a b = 1/2 →
    (∀ (m n : ℝ × ℝ), 
      OnEllipse m ∧ OnEllipse n ∧ m.1 = 2 ∧ n.1 = 2 →
      CirclePassesThroughFixedPoint m n) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3376_337672


namespace NUMINAMATH_CALUDE_eleanor_distance_between_meetings_l3376_337688

/-- The distance of the circular track in meters -/
def track_length : ℝ := 720

/-- The time Eric takes to complete one circuit in minutes -/
def eric_time : ℝ := 4

/-- The time Eleanor takes to complete one circuit in minutes -/
def eleanor_time : ℝ := 5

/-- The theorem stating the distance Eleanor runs between consecutive meetings -/
theorem eleanor_distance_between_meetings :
  let eric_speed := track_length / eric_time
  let eleanor_speed := track_length / eleanor_time
  let relative_speed := eric_speed + eleanor_speed
  let time_between_meetings := track_length / relative_speed
  eleanor_speed * time_between_meetings = 320 := by
  sorry

end NUMINAMATH_CALUDE_eleanor_distance_between_meetings_l3376_337688


namespace NUMINAMATH_CALUDE_pizza_area_increase_l3376_337614

/-- Given that the radius of a large pizza is 40% larger than the radius of a medium pizza,
    prove that the percent increase in area between a medium and a large pizza is 96%. -/
theorem pizza_area_increase (r : ℝ) (h : r > 0) : 
  let large_radius := 1.4 * r
  let medium_area := Real.pi * r^2
  let large_area := Real.pi * large_radius^2
  (large_area - medium_area) / medium_area * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l3376_337614


namespace NUMINAMATH_CALUDE_inverse_variation_result_l3376_337639

/-- Represents the inverse relationship between y^2 and √⁴z -/
def inverse_relationship (y z : ℝ) : Prop :=
  ∃ k : ℝ, y^2 * z^(1/4) = k

theorem inverse_variation_result :
  ∀ y z : ℝ,
  inverse_relationship y z →
  inverse_relationship 3 16 →
  y = 6 →
  z = 1/16 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_result_l3376_337639


namespace NUMINAMATH_CALUDE_parabola_vertex_l3376_337647

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -x^2 + 15

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (0, 15)

/-- Theorem: The vertex of the parabola y = -x^2 + 15 is at the point (0, 15) -/
theorem parabola_vertex :
  (∀ x : ℝ, parabola x ≤ parabola (vertex.1)) ∧ parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3376_337647


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3376_337656

/-- The minimum distance between points on two specific curves -/
theorem min_distance_between_curves : ∃ (d : ℝ),
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 →
    let p := (x₁, (1/2) * Real.exp x₁)
    let q := (x₂, Real.log (2 * x₂))
    d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
  d = Real.sqrt 2 * (1 - Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3376_337656


namespace NUMINAMATH_CALUDE_sunday_necklace_production_l3376_337621

/-- The number of necklaces made by the first machine on Sunday -/
def first_machine_necklaces : ℕ := 45

/-- The ratio of necklaces made by the second machine compared to the first -/
def second_machine_ratio : ℝ := 2.4

/-- The total number of necklaces made on Sunday -/
def total_necklaces : ℕ := 153

/-- Theorem stating that the total number of necklaces made on Sunday is 153 -/
theorem sunday_necklace_production :
  (first_machine_necklaces : ℝ) + first_machine_necklaces * second_machine_ratio = total_necklaces := by
  sorry

end NUMINAMATH_CALUDE_sunday_necklace_production_l3376_337621


namespace NUMINAMATH_CALUDE_right_triangle_area_from_broken_stick_l3376_337667

theorem right_triangle_area_from_broken_stick : ∀ a : ℝ,
  0 < a →
  a < 24 →
  a^2 + 24^2 = (48 - a)^2 →
  (1/2) * a * 24 = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_from_broken_stick_l3376_337667


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l3376_337671

theorem triangle_determinant_zero (A B C : ℝ) (h : A + B + C = π) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.cos A ^ 2, Real.tan A, 1],
    ![Real.cos B ^ 2, Real.tan B, 1],
    ![Real.cos C ^ 2, Real.tan C, 1]
  ]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l3376_337671


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3376_337610

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) : 
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3376_337610


namespace NUMINAMATH_CALUDE_journey_time_calculation_l3376_337645

theorem journey_time_calculation (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_distance = 112 ∧ speed1 = 21 ∧ speed2 = 24 →
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l3376_337645


namespace NUMINAMATH_CALUDE_book_purchase_ratio_l3376_337637

/-- The number of people who purchased only book A -/
def only_A : ℕ := 1000

/-- The number of people who purchased both books A and B -/
def both_A_and_B : ℕ := 500

/-- The number of people who purchased only book B -/
def only_B : ℕ := both_A_and_B / 2

/-- The total number of people who purchased book A -/
def total_A : ℕ := only_A + both_A_and_B

/-- The total number of people who purchased book B -/
def total_B : ℕ := only_B + both_A_and_B

/-- The ratio of people who purchased book A to those who purchased book B -/
def ratio : ℚ := total_A / total_B

theorem book_purchase_ratio : ratio = 2 := by sorry

end NUMINAMATH_CALUDE_book_purchase_ratio_l3376_337637


namespace NUMINAMATH_CALUDE_married_men_fraction_l3376_337617

theorem married_men_fraction (total_women : ℕ) (total_people : ℕ) 
  (h1 : total_women > 0)
  (h2 : total_people > 0)
  (h3 : (3 : ℚ) / 7 = (single_women : ℚ) / total_women) 
  (h4 : total_people = total_women + married_men)
  (h5 : married_women = total_women - single_women)
  (h6 : married_men = married_women) :
  (married_men : ℚ) / total_people = 4 / 11 :=
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l3376_337617


namespace NUMINAMATH_CALUDE_additional_money_needed_l3376_337609

def dictionary_cost : ℕ := 11
def dinosaur_book_cost : ℕ := 19
def cookbook_cost : ℕ := 7
def savings : ℕ := 8

theorem additional_money_needed : 
  dictionary_cost + dinosaur_book_cost + cookbook_cost - savings = 29 := by
sorry

end NUMINAMATH_CALUDE_additional_money_needed_l3376_337609


namespace NUMINAMATH_CALUDE_regina_farm_earnings_l3376_337607

def farm_earnings (num_cows : ℕ) (pig_cow_ratio : ℕ) (price_pig : ℕ) (price_cow : ℕ) : ℕ :=
  let num_pigs := pig_cow_ratio * num_cows
  (num_cows * price_cow) + (num_pigs * price_pig)

theorem regina_farm_earnings :
  farm_earnings 20 4 400 800 = 48000 := by
  sorry

end NUMINAMATH_CALUDE_regina_farm_earnings_l3376_337607


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3376_337648

-- Define the inequality
def inequality (x : ℝ) : Prop := (2*x - 1) / (x + 3) > 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ x > 1/2}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3376_337648


namespace NUMINAMATH_CALUDE_katies_new_games_l3376_337630

/-- Katie's new games problem -/
theorem katies_new_games :
  ∀ (k : ℕ),  -- k represents Katie's new games
  (k + 8 = 92) →  -- Total new games between Katie and her friends is 92
  (k = 84)  -- Katie has 84 new games
  := by sorry

end NUMINAMATH_CALUDE_katies_new_games_l3376_337630


namespace NUMINAMATH_CALUDE_inequality_proof_l3376_337658

theorem inequality_proof (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  let L := (a^3 / (b + c + d)) + (b^3 / (c + d + a)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c))
  L ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3376_337658


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_range_of_m_value_of_m_l3376_337686

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + (2*m - 1)*x + m^2

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (2*m - 1)^2 - 4*m^2

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop := discriminant m > 0

-- Define the sum and product of roots
def sum_of_roots (m : ℝ) : ℝ := -(2*m - 1)
def product_of_roots (m : ℝ) : ℝ := m^2

theorem quadratic_roots_theorem (m : ℝ) :
  has_two_distinct_real_roots m →
  (∃ α β : ℝ, α ≠ β ∧ 
    quadratic m α = 0 ∧ 
    quadratic m β = 0 ∧ 
    α + β = sum_of_roots m ∧ 
    α * β = product_of_roots m) :=
sorry

theorem range_of_m (m : ℝ) :
  has_two_distinct_real_roots m → m < (1/4) :=
sorry

theorem value_of_m (m : ℝ) (α β : ℝ) :
  has_two_distinct_real_roots m →
  quadratic m α = 0 →
  quadratic m β = 0 →
  α^2 + β^2 = 1 →
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_range_of_m_value_of_m_l3376_337686


namespace NUMINAMATH_CALUDE_smallest_a_value_l3376_337674

theorem smallest_a_value (a b : ℤ) : 
  (a + 2 * b = 32) → 
  (abs a > 2) → 
  (∀ x : ℤ, x + 2 * b = 32 → abs x > 2 → x ≥ 4) → 
  (a = 4) → 
  (b = 14) := by
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3376_337674


namespace NUMINAMATH_CALUDE_range_of_a_l3376_337664

theorem range_of_a (x a : ℝ) : 
  (∀ x, (1/2 ≤ x ∧ x ≤ 1) → (a ≤ x ∧ x ≤ a + 1)) ∧
  (∃ x, (1/2 ≤ x ∧ x ≤ 1) ∧ ¬(a ≤ x ∧ x ≤ a + 1)) →
  (0 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3376_337664


namespace NUMINAMATH_CALUDE_ln_one_eq_zero_l3376_337606

theorem ln_one_eq_zero : Real.log 1 = 0 := by sorry

end NUMINAMATH_CALUDE_ln_one_eq_zero_l3376_337606


namespace NUMINAMATH_CALUDE_stating_basketball_tournament_wins_l3376_337629

/-- Represents the number of games won in a basketball tournament -/
def games_won (total_games : ℕ) (total_points : ℕ) : ℕ :=
  total_games - (2 * total_games - total_points)

/-- 
Theorem stating that given 8 games where wins earn 2 points and losses earn 1 point, 
if the total points earned is 13, then the number of games won is 5.
-/
theorem basketball_tournament_wins :
  games_won 8 13 = 5 := by
  sorry

#eval games_won 8 13  -- Should output 5

end NUMINAMATH_CALUDE_stating_basketball_tournament_wins_l3376_337629


namespace NUMINAMATH_CALUDE_math_students_count_l3376_337660

theorem math_students_count (total : ℕ) (difference : ℕ) (math_students : ℕ) : 
  total = 1256 →
  difference = 408 →
  math_students < 500 →
  math_students + difference + math_students = total →
  math_students = 424 := by
sorry

end NUMINAMATH_CALUDE_math_students_count_l3376_337660


namespace NUMINAMATH_CALUDE_jessica_and_sibling_ages_l3376_337612

-- Define the variables
def jessica_age_at_passing : ℕ := sorry
def mother_age_at_passing : ℕ := sorry
def current_year : ℕ := sorry
def sibling_age : ℕ := sorry

-- Define the conditions
def jessica_half_mother_age : Prop :=
  jessica_age_at_passing = mother_age_at_passing / 2

def mother_age_if_alive : Prop :=
  mother_age_at_passing + 10 = 70

def sibling_age_difference : Prop :=
  sibling_age - (jessica_age_at_passing + 10) = (70 - mother_age_at_passing) / 2

-- Theorem to prove
theorem jessica_and_sibling_ages :
  jessica_half_mother_age →
  mother_age_if_alive →
  sibling_age_difference →
  jessica_age_at_passing + 10 = 40 ∧ sibling_age = 45 := by
  sorry

end NUMINAMATH_CALUDE_jessica_and_sibling_ages_l3376_337612


namespace NUMINAMATH_CALUDE_bryce_raisins_l3376_337643

theorem bryce_raisins (bryce carter : ℚ) 
  (h1 : bryce = carter + 10)
  (h2 : carter = (1 / 4) * bryce) : 
  bryce = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_bryce_raisins_l3376_337643


namespace NUMINAMATH_CALUDE_football_team_size_l3376_337684

/-- Represents the number of players on a football team -/
def total_players : ℕ := 70

/-- Represents the number of throwers on the team -/
def throwers : ℕ := 31

/-- Represents the total number of right-handed players -/
def right_handed_total : ℕ := 57

/-- Represents the number of left-handed players (non-throwers) -/
def left_handed_non_throwers : ℕ := (total_players - throwers) / 3

/-- Represents the number of right-handed non-throwers -/
def right_handed_non_throwers : ℕ := right_handed_total - throwers

theorem football_team_size :
  total_players = throwers + left_handed_non_throwers + right_handed_non_throwers ∧
  left_handed_non_throwers * 2 = right_handed_non_throwers ∧
  right_handed_total = throwers + right_handed_non_throwers :=
by
  sorry

end NUMINAMATH_CALUDE_football_team_size_l3376_337684


namespace NUMINAMATH_CALUDE_consecutive_squares_not_equal_consecutive_fourth_powers_l3376_337624

theorem consecutive_squares_not_equal_consecutive_fourth_powers :
  ∀ a b : ℕ, a^2 + (a+1)^2 ≠ b^4 + (b+1)^4 := by sorry

end NUMINAMATH_CALUDE_consecutive_squares_not_equal_consecutive_fourth_powers_l3376_337624


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3376_337690

theorem polynomial_simplification (r : ℝ) :
  (2 * r^2 + 5 * r - 3) + (3 * r^2 - 4 * r + 2) = 5 * r^2 + r - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3376_337690


namespace NUMINAMATH_CALUDE_min_cube_sum_l3376_337600

theorem min_cube_sum (a b t : ℝ) (h : a + b = t) :
  ∃ (min : ℝ), min = t^3 / 4 ∧ ∀ (x y : ℝ), x + y = t → x^3 + y^3 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_cube_sum_l3376_337600


namespace NUMINAMATH_CALUDE_no_integer_roots_for_odd_coefficients_l3376_337641

theorem no_integer_roots_for_odd_coefficients (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) : 
  ¬ ∃ x : ℤ, a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_for_odd_coefficients_l3376_337641


namespace NUMINAMATH_CALUDE_product_equals_19404_l3376_337682

theorem product_equals_19404 : 3^2 * 4 * 7^2 * 11 = 19404 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_19404_l3376_337682


namespace NUMINAMATH_CALUDE_train_crossing_time_l3376_337602

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 50 →
  train_speed_kmh = 60 →
  crossing_time = 3 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3376_337602


namespace NUMINAMATH_CALUDE_gas_fee_calculation_l3376_337632

/-- Calculates the gas fee for a given usage --/
def gas_fee (usage : ℕ) : ℚ :=
  if usage ≤ 60 then
    0.8 * usage
  else
    0.8 * 60 + 1.2 * (usage - 60)

/-- Represents the average cost per cubic meter --/
def average_cost (usage : ℕ) (fee : ℚ) : ℚ :=
  fee / usage

theorem gas_fee_calculation (usage : ℕ) (h : average_cost usage (gas_fee usage) = 0.88) :
  gas_fee usage = 66 := by
  sorry

end NUMINAMATH_CALUDE_gas_fee_calculation_l3376_337632


namespace NUMINAMATH_CALUDE_angle_calculation_l3376_337687

-- Define the angles in degrees
def small_triangle_angle1 : ℝ := 70
def small_triangle_angle2 : ℝ := 50
def large_triangle_angle1 : ℝ := 45
def large_triangle_angle2 : ℝ := 50

-- Define α and β
def α : ℝ := 120
def β : ℝ := 85

-- Theorem statement
theorem angle_calculation :
  let small_triangle_angle3 := 180 - (small_triangle_angle1 + small_triangle_angle2)
  α = 180 - small_triangle_angle3 ∧
  β = 180 - (large_triangle_angle1 + large_triangle_angle2) := by
sorry

end NUMINAMATH_CALUDE_angle_calculation_l3376_337687


namespace NUMINAMATH_CALUDE_smallest_integer_sequence_sum_l3376_337698

theorem smallest_integer_sequence_sum (B : ℤ) : B = -2022 ↔ 
  (∃ n : ℕ, (Finset.range n).sum (λ i => B + i) = 2023) ∧ 
  (∀ k < B, ¬∃ m : ℕ, (Finset.range m).sum (λ i => k + i) = 2023) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_sequence_sum_l3376_337698


namespace NUMINAMATH_CALUDE_integral_odd_function_integral_even_function_integral_positive_function_exists_counterexample_for_D_incorrect_proposition_l3376_337673

-- Define the necessary concepts
def continuous (f : ℝ → ℝ) : Prop := sorry
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def integral (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

-- Theorem A
theorem integral_odd_function (f : ℝ → ℝ) (α : ℝ) :
  continuous f → odd_function f → integral f (-α) α = 0 := by sorry

-- Theorem B
theorem integral_even_function (f : ℝ → ℝ) (a : ℝ) :
  continuous f → even_function f → integral f (-a) a = 2 * integral f 0 a := by sorry

-- Theorem C
theorem integral_positive_function (f : ℝ → ℝ) (a b : ℝ) :
  continuous f → (∀ x ∈ [a, b], f x > 0) → integral f a b > 0 := by sorry

-- Theorem D (false)
theorem exists_counterexample_for_D :
  ∃ f : ℝ → ℝ, ∃ a b : ℝ,
    continuous f ∧ 
    integral f a b > 0 ∧ 
    ¬(∀ x ∈ [a, b], f x > 0) := by sorry

-- Main theorem
theorem incorrect_proposition :
  ¬(∀ f : ℝ → ℝ, ∀ a b : ℝ,
    continuous f → integral f a b > 0 → (∀ x ∈ [a, b], f x > 0)) := by sorry

end NUMINAMATH_CALUDE_integral_odd_function_integral_even_function_integral_positive_function_exists_counterexample_for_D_incorrect_proposition_l3376_337673


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3376_337661

theorem imaginary_part_of_complex_number (z : ℂ) : z = (1 - Complex.I) * Complex.I → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3376_337661


namespace NUMINAMATH_CALUDE_ratio_problem_l3376_337622

theorem ratio_problem (a b c d : ℝ) 
  (h1 : b / a = 3)
  (h2 : c / b = 4)
  (h3 : d = 5 * b)
  : (a + b + d) / (b + c + d) = 19 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3376_337622


namespace NUMINAMATH_CALUDE_function_equality_l3376_337696

theorem function_equality : 
  (∀ x : ℝ, |x| = Real.sqrt (x^2)) ∧ 
  (∀ x : ℝ, x^2 = (fun t => t^2) x) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l3376_337696


namespace NUMINAMATH_CALUDE_one_color_triangle_l3376_337662

/-- Represents a stick with a color and length -/
structure Stick where
  color : Bool  -- True for blue, False for yellow
  length : ℝ
  positive : length > 0

/-- Represents a hexagon formed by 6 sticks -/
structure Hexagon where
  sticks : Fin 6 → Stick
  alternating : ∀ i, (sticks i).color ≠ (sticks (i + 1)).color
  three_yellow : ∃ a b c, (sticks a).color = false ∧ (sticks b).color = false ∧ (sticks c).color = false ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c
  three_blue : ∃ a b c, (sticks a).color = true ∧ (sticks b).color = true ∧ (sticks c).color = true ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Checks if three sticks can form a triangle -/
def canFormTriangle (a b c : Stick) : Prop :=
  a.length + b.length > c.length ∧
  b.length + c.length > a.length ∧
  c.length + a.length > b.length

/-- Main theorem: In a hexagon with alternating colored sticks where any three consecutive sticks can form a triangle, 
    it's possible to form a triangle using sticks of only one color -/
theorem one_color_triangle (h : Hexagon)
  (consecutive_triangle : ∀ i, canFormTriangle (h.sticks i) (h.sticks (i + 1)) (h.sticks (i + 2))) :
  (∃ a b c, (h.sticks a).color = (h.sticks b).color ∧ 
            (h.sticks b).color = (h.sticks c).color ∧ 
            canFormTriangle (h.sticks a) (h.sticks b) (h.sticks c)) :=
by sorry

end NUMINAMATH_CALUDE_one_color_triangle_l3376_337662


namespace NUMINAMATH_CALUDE_original_denominator_proof_l3376_337649

theorem original_denominator_proof (d : ℕ) : 
  (4 : ℚ) / d ≠ 0 →
  (4 + 3 : ℚ) / (d + 3) = 1 / 3 →
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l3376_337649


namespace NUMINAMATH_CALUDE_two_digit_sum_divisible_by_11_l3376_337665

theorem two_digit_sum_divisible_by_11 (A B : ℕ) (h1 : A < 10) (h2 : B < 10) :
  ∃ k : ℤ, (10 * A + B : ℤ) + (10 * B + A : ℤ) = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_divisible_by_11_l3376_337665


namespace NUMINAMATH_CALUDE_total_yellow_marbles_l3376_337677

theorem total_yellow_marbles (mary joan tim lisa : ℕ) 
  (h1 : mary = 9) 
  (h2 : joan = 3) 
  (h3 : tim = 5) 
  (h4 : lisa = 7) : 
  mary + joan + tim + lisa = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_yellow_marbles_l3376_337677


namespace NUMINAMATH_CALUDE_gcd_of_390_455_546_l3376_337613

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_390_455_546_l3376_337613


namespace NUMINAMATH_CALUDE_tv_cost_l3376_337679

def original_savings : ℚ := 600

def furniture_fraction : ℚ := 2/4

theorem tv_cost (savings : ℚ) (frac : ℚ) (h1 : savings = original_savings) (h2 : frac = furniture_fraction) : 
  savings * (1 - frac) = 300 := by
  sorry

end NUMINAMATH_CALUDE_tv_cost_l3376_337679


namespace NUMINAMATH_CALUDE_direct_proportion_through_3_6_l3376_337642

/-- A direct proportion function passing through (3,6) -/
def f (x : ℝ) : ℝ := 2 * x

theorem direct_proportion_through_3_6 :
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x) ∧ 
  f 3 = 6 ∧
  ∀ x : ℝ, f x = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_through_3_6_l3376_337642


namespace NUMINAMATH_CALUDE_cos_tan_arcsin_four_fifths_l3376_337611

theorem cos_tan_arcsin_four_fifths :
  (∃ θ : ℝ, θ = Real.arcsin (4/5)) →
  (Real.cos (Real.arcsin (4/5)) = 3/5) ∧
  (Real.tan (Real.arcsin (4/5)) = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_cos_tan_arcsin_four_fifths_l3376_337611


namespace NUMINAMATH_CALUDE_bird_island_injured_parrots_l3376_337616

/-- The number of parrots on Bird Island -/
def total_parrots : ℕ := 105

/-- The fraction of parrots that are green -/
def green_fraction : ℚ := 5/7

/-- The percentage of green parrots that are injured -/
def injured_percentage : ℚ := 3/100

/-- The number of injured green parrots -/
def injured_green_parrots : ℕ := 2

theorem bird_island_injured_parrots :
  ⌊(total_parrots : ℚ) * green_fraction * injured_percentage⌋ = injured_green_parrots := by
  sorry

end NUMINAMATH_CALUDE_bird_island_injured_parrots_l3376_337616


namespace NUMINAMATH_CALUDE_angle_D_value_l3376_337681

-- Define the angles as real numbers
variable (A B C D : ℝ)

-- State the given conditions
axiom sum_A_B : A + B = 180
axiom C_eq_D : C = D
axiom A_value : A = 50

-- State the theorem to be proven
theorem angle_D_value : D = 25 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_value_l3376_337681


namespace NUMINAMATH_CALUDE_seashell_ratio_l3376_337631

theorem seashell_ratio (monday_shells : ℕ) (price_per_shell : ℚ) (total_revenue : ℚ) :
  monday_shells = 30 →
  price_per_shell = 6/5 →
  total_revenue = 54 →
  ∃ (tuesday_shells : ℕ), 
    (tuesday_shells : ℚ) / monday_shells = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_seashell_ratio_l3376_337631


namespace NUMINAMATH_CALUDE_binomial_1500_1_l3376_337666

theorem binomial_1500_1 : Nat.choose 1500 1 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1500_1_l3376_337666


namespace NUMINAMATH_CALUDE_simplify_sum_of_roots_l3376_337652

theorem simplify_sum_of_roots : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sum_of_roots_l3376_337652


namespace NUMINAMATH_CALUDE_f_neg_two_eq_three_l3376_337620

-- Define the function f
def f (x : ℝ) : ℝ := -2 * (x + 1) + 1

-- Theorem statement
theorem f_neg_two_eq_three : f (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_three_l3376_337620


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_specific_trapezoid_l3376_337689

/-- An isosceles trapezoid with given dimensions. -/
structure IsoscelesTrapezoid where
  height : ℝ
  base1 : ℝ
  base2 : ℝ

/-- The circumscribed circle of an isosceles trapezoid. -/
def circumscribedCircleArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the circumscribed circle for a specific isosceles trapezoid. -/
theorem circumscribed_circle_area_specific_trapezoid :
  let t : IsoscelesTrapezoid := { height := 14, base1 := 16, base2 := 12 }
  circumscribedCircleArea t = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_specific_trapezoid_l3376_337689


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3376_337646

theorem gcd_lcm_sum : Nat.gcd 48 70 + Nat.lcm 18 45 = 92 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3376_337646


namespace NUMINAMATH_CALUDE_two_lines_exist_l3376_337638

/-- A parabola defined by the equation y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The point P(2,4) -/
def P : ℝ × ℝ := (2, 4)

/-- A line that has exactly one common point with the parabola -/
def SingleIntersectionLine (l : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ l ∩ Parabola

/-- A line that passes through point P -/
def LineThroughP (l : Set (ℝ × ℝ)) : Prop :=
  P ∈ l

/-- The theorem stating that there are exactly two lines satisfying the conditions -/
theorem two_lines_exist : 
  ∃! (l1 l2 : Set (ℝ × ℝ)), 
    l1 ≠ l2 ∧ 
    LineThroughP l1 ∧ 
    LineThroughP l2 ∧ 
    SingleIntersectionLine l1 ∧ 
    SingleIntersectionLine l2 ∧
    ∀ l, LineThroughP l ∧ SingleIntersectionLine l → l = l1 ∨ l = l2 :=
sorry

end NUMINAMATH_CALUDE_two_lines_exist_l3376_337638


namespace NUMINAMATH_CALUDE_expand_expression_l3376_337608

theorem expand_expression (x : ℝ) : (x - 3) * (4 * x + 12) = 4 * x^2 - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3376_337608


namespace NUMINAMATH_CALUDE_min_weighings_eq_counterfeit_problem_weighings_l3376_337654

/-- Represents a coin collection with genuine and counterfeit coins. -/
structure CoinCollection where
  total : ℕ
  genuine : ℕ
  counterfeit : ℕ
  genuine_weight : ℝ
  counterfeit_weights : Finset ℝ
  h_total : total = genuine + counterfeit
  h_genuine_lt_counterfeit : genuine < counterfeit
  h_counterfeit_heavier : ∀ w ∈ counterfeit_weights, w > genuine_weight
  h_counterfeit_distinct : counterfeit_weights.card = counterfeit

/-- Represents a weighing on a balance scale. -/
def Weighing (c : CoinCollection) := Finset (Fin c.total)

/-- The result of a weighing is either balanced or unbalanced. -/
inductive WeighingResult
  | balanced
  | unbalanced

/-- Performs a weighing and returns the result. -/
def performWeighing (c : CoinCollection) (w : Weighing c) : WeighingResult :=
  sorry

/-- Theorem: The minimum number of weighings needed to guarantee finding a genuine coin is equal to the number of counterfeit coins. -/
theorem min_weighings_eq_counterfeit (c : CoinCollection) :
  (∃ n : ℕ, ∀ m : ℕ, (∃ (weighings : Fin m → Weighing c), 
    ∃ (i : Fin m), performWeighing c (weighings i) = WeighingResult.balanced) ↔ m ≥ n) ∧ 
  (∀ k : ℕ, k < c.counterfeit → 
    ∃ (weighings : Fin k → Weighing c), ∀ i : Fin k, 
      performWeighing c (weighings i) = WeighingResult.unbalanced) :=
  sorry

/-- The specific coin collection from the problem. -/
def problemCollection : CoinCollection where
  total := 100
  genuine := 30
  counterfeit := 70
  genuine_weight := 1
  counterfeit_weights := sorry
  h_total := rfl
  h_genuine_lt_counterfeit := by norm_num
  h_counterfeit_heavier := sorry
  h_counterfeit_distinct := sorry

/-- The main theorem: 70 weighings are needed for the problem collection. -/
theorem problem_weighings :
  (∃ n : ℕ, ∀ m : ℕ, (∃ (weighings : Fin m → Weighing problemCollection), 
    ∃ (i : Fin m), performWeighing problemCollection (weighings i) = WeighingResult.balanced) ↔ m ≥ n) ∧
  n = 70 :=
  sorry

end NUMINAMATH_CALUDE_min_weighings_eq_counterfeit_problem_weighings_l3376_337654


namespace NUMINAMATH_CALUDE_simplify_expression_l3376_337668

theorem simplify_expression : 10 * (15 / 8) * (-40 / 45) = -50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3376_337668


namespace NUMINAMATH_CALUDE_farm_horses_cows_l3376_337644

theorem farm_horses_cows (h c : ℕ) : 
  h = 4 * c →                             -- Initial ratio of horses to cows is 4:1
  (h - 15) / (c + 15) = 7 / 3 →           -- New ratio after transaction is 7:3
  (h - 15) - (c + 15) = 60 :=              -- Difference after transaction is 60
by
  sorry

end NUMINAMATH_CALUDE_farm_horses_cows_l3376_337644


namespace NUMINAMATH_CALUDE_only_one_correct_proposition_l3376_337663

-- Define a predicate for each proposition
def proposition1 (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (x + 1)) → (∃ a b, ∀ x, f x = a * Real.sin (b * x) + a * Real.cos (b * x))

def proposition2 : Prop :=
  ∃ x : ℝ, x^2 - x > 0

def proposition3 (A B : ℝ) : Prop :=
  Real.sin A > Real.sin B ↔ A > B

def proposition4 (f : ℝ → ℝ) : Prop :=
  (∃ x ∈ Set.Ioo 2015 2017, f x = 0) → f 2015 * f 2017 < 0

def proposition5 (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ (1 / x) * (1 / y) = -1

-- Theorem stating that only one proposition is correct
theorem only_one_correct_proposition :
  (¬ ∀ f, proposition1 f) ∧
  (¬ proposition2) ∧
  (∀ A B, proposition3 A B) ∧
  (¬ ∀ f, proposition4 f) ∧
  (¬ proposition5 Real.log) :=
sorry

end NUMINAMATH_CALUDE_only_one_correct_proposition_l3376_337663


namespace NUMINAMATH_CALUDE_cookie_ratio_l3376_337694

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The total number of cookies Meena baked -/
def total_cookies : ℕ := 5 * dozen

/-- The number of cookies Mr. Stone bought -/
def stone_cookies : ℕ := 2 * dozen

/-- The number of cookies Brock bought -/
def brock_cookies : ℕ := 7

/-- The number of cookies left -/
def cookies_left : ℕ := 15

/-- The number of cookies Katy bought -/
def katy_cookies : ℕ := total_cookies - stone_cookies - brock_cookies - cookies_left

theorem cookie_ratio : 
  (katy_cookies : ℚ) / brock_cookies = 2 := by sorry

end NUMINAMATH_CALUDE_cookie_ratio_l3376_337694


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l3376_337628

theorem three_digit_number_divisible_by_seven :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ (n / 100) % 10 = 5 ∧ n % 7 = 0 ∧ n = 553 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l3376_337628
