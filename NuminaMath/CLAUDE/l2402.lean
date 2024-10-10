import Mathlib

namespace no_four_tangents_for_different_radii_l2402_240249

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the number of common tangents between two circles --/
def commonTangents (c1 c2 : Circle) : ℕ := sorry

/-- Two circles with different radii cannot have exactly 4 common tangents --/
theorem no_four_tangents_for_different_radii (c1 c2 : Circle) 
  (h : c1.radius ≠ c2.radius) : commonTangents c1 c2 ≠ 4 := by sorry

end no_four_tangents_for_different_radii_l2402_240249


namespace intersection_of_A_and_B_l2402_240239

def A : Set ℝ := {x | x^2 - 4*x - 5 > 0}
def B : Set ℝ := {x | 4 - x^2 > 0}

theorem intersection_of_A_and_B :
  A ∩ B = {x | -2 < x ∧ x < -1} := by
  sorry

end intersection_of_A_and_B_l2402_240239


namespace divisibility_count_l2402_240243

theorem divisibility_count : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (1638 : ℤ) % (n^2 - 3) = 0) ∧ 
    (∀ n : ℕ, n > 0 ∧ (1638 : ℤ) % (n^2 - 3) = 0 → n ∈ S) ∧
    S.card = 4 :=
by sorry

end divisibility_count_l2402_240243


namespace probability_of_asian_card_l2402_240269

-- Define the set of cards
inductive Card : Type
| China : Card
| USA : Card
| UK : Card
| SouthKorea : Card

-- Define a function to check if a card corresponds to an Asian country
def isAsian : Card → Bool
| Card.China => true
| Card.SouthKorea => true
| _ => false

-- Define the total number of cards
def totalCards : ℕ := 4

-- Define the number of Asian countries
def asianCards : ℕ := 2

-- Theorem statement
theorem probability_of_asian_card :
  (asianCards : ℚ) / totalCards = 1 / 2 := by
  sorry

end probability_of_asian_card_l2402_240269


namespace special_function_value_at_neg_two_l2402_240293

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 4 * x * y

theorem special_function_value_at_neg_two
  (f : ℝ → ℝ) (h : special_function f) (h1 : f 1 = 2) :
  f (-2) = 8 := by
  sorry

end special_function_value_at_neg_two_l2402_240293


namespace video_game_price_l2402_240261

def lawn_price : ℕ := 15
def book_price : ℕ := 5
def lawns_mowed : ℕ := 35
def video_games_wanted : ℕ := 5
def books_bought : ℕ := 60

theorem video_game_price :
  (lawn_price * lawns_mowed - book_price * books_bought) / video_games_wanted = 45 := by
  sorry

end video_game_price_l2402_240261


namespace chris_money_before_birthday_l2402_240283

def grandmother_gift : ℕ := 25
def aunt_uncle_gift : ℕ := 20
def parents_gift : ℕ := 75
def total_money : ℕ := 279

theorem chris_money_before_birthday :
  total_money - (grandmother_gift + aunt_uncle_gift + parents_gift) = 159 := by
  sorry

end chris_money_before_birthday_l2402_240283


namespace roots_quadratic_equation_l2402_240276

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 + 2 * Real.sqrt 2 * m + 1 = 0) ∧ 
  (n^2 + 2 * Real.sqrt 2 * n + 1 = 0) → 
  Real.sqrt (m^2 + n^2 + 3*m*n) = 3 := by
  sorry

end roots_quadratic_equation_l2402_240276


namespace erroneous_product_theorem_l2402_240263

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem erroneous_product_theorem (a b : ℕ) (h1 : is_two_digit a) (h2 : reverse_digits a * b = 180) :
  a * b = 315 ∨ a * b = 810 := by
  sorry

end erroneous_product_theorem_l2402_240263


namespace magic_trick_basis_l2402_240214

/-- The set of valid dice face pairs -/
def DicePairs : Set (ℕ × ℕ) :=
  {p | 1 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ 6}

/-- The set of possible numbers of dice in the spectator's pocket -/
def PocketCounts : Set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 21}

/-- The statement of the magic trick's mathematical basis -/
theorem magic_trick_basis :
  ∃ f : DicePairs → PocketCounts, Function.Bijective f := by
  sorry

end magic_trick_basis_l2402_240214


namespace distribute_five_identical_books_to_three_students_l2402_240275

/-- The number of ways to distribute n identical objects to k recipients, 
    where each recipient receives exactly one object. -/
def distribute_identical (n k : ℕ) : ℕ :=
  if n = k then 1 else 0

/-- Theorem: There is only one way to distribute 5 identical books to 3 students, 
    with each student receiving one book. -/
theorem distribute_five_identical_books_to_three_students :
  distribute_identical 5 3 = 1 := by
  sorry

end distribute_five_identical_books_to_three_students_l2402_240275


namespace red_jellybeans_count_l2402_240205

/-- The number of red jellybeans in a jar -/
def num_red_jellybeans (total blue purple orange pink yellow : ℕ) : ℕ :=
  total - (blue + purple + orange + pink + yellow)

/-- Theorem stating the number of red jellybeans in the jar -/
theorem red_jellybeans_count :
  num_red_jellybeans 237 14 26 40 7 21 = 129 := by
  sorry

end red_jellybeans_count_l2402_240205


namespace janna_weekly_sleep_l2402_240257

/-- The number of hours Janna sleeps in a week -/
def total_sleep_hours (weekday_sleep : ℕ) (weekend_sleep : ℕ) (weekdays : ℕ) (weekend_days : ℕ) : ℕ :=
  weekday_sleep * weekdays + weekend_sleep * weekend_days

/-- Theorem stating that Janna sleeps 51 hours in a week -/
theorem janna_weekly_sleep :
  total_sleep_hours 7 8 5 2 = 51 := by
  sorry

end janna_weekly_sleep_l2402_240257


namespace unique_solution_2014_l2402_240271

theorem unique_solution_2014 (x : ℝ) (h : x > 0) :
  (x * 2014^(1/x) + (1/x) * 2014^x) / 2 = 2014 ↔ x = 1 := by
  sorry

end unique_solution_2014_l2402_240271


namespace salt_solution_volume_l2402_240227

/-- Proves that the initial volume of a solution is 80 gallons, given the conditions of the problem -/
theorem salt_solution_volume : 
  ∀ (V : ℝ), 
  (0.1 * V = 0.08 * (V + 20)) → 
  V = 80 := by
sorry

end salt_solution_volume_l2402_240227


namespace acme_cheaper_min_shirts_l2402_240299

def acme_cost (x : ℕ) : ℚ := 50 + 9 * x
def beta_cost (x : ℕ) : ℚ := 25 + 15 * x

theorem acme_cheaper_min_shirts : 
  ∀ n : ℕ, (∀ k : ℕ, k < n → acme_cost k ≥ beta_cost k) ∧ 
           (acme_cost n < beta_cost n) → n = 5 := by
  sorry

end acme_cheaper_min_shirts_l2402_240299


namespace max_digits_distinct_divisible_l2402_240290

/-- A function that checks if all digits in a natural number are different -/
def hasDistinctDigits (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is divisible by all of its digits -/
def isDivisibleByAllDigits (n : ℕ) : Prop := sorry

/-- A function that returns the number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the maximum number of digits in a natural number
    with distinct digits and divisible by all its digits is 7 -/
theorem max_digits_distinct_divisible :
  ∃ (n : ℕ), hasDistinctDigits n ∧ isDivisibleByAllDigits n ∧ numDigits n = 7 ∧
  ∀ (m : ℕ), hasDistinctDigits m → isDivisibleByAllDigits m → numDigits m ≤ 7 :=
sorry

end max_digits_distinct_divisible_l2402_240290


namespace sqrt_25_equals_5_l2402_240237

theorem sqrt_25_equals_5 : Real.sqrt 25 = 5 := by
  sorry

end sqrt_25_equals_5_l2402_240237


namespace yellow_beads_count_l2402_240270

theorem yellow_beads_count (blue_beads : ℕ) (total_parts : ℕ) (removed_per_part : ℕ) (final_per_part : ℕ) : 
  blue_beads = 23 →
  total_parts = 3 →
  removed_per_part = 10 →
  final_per_part = 6 →
  (∃ (yellow_beads : ℕ),
    let total_beads := blue_beads + yellow_beads
    let remaining_per_part := (total_beads / total_parts) - removed_per_part
    2 * remaining_per_part = final_per_part ∧
    yellow_beads = 16) :=
by
  sorry

#check yellow_beads_count

end yellow_beads_count_l2402_240270


namespace point_on_line_k_l2402_240219

/-- A line passing through the origin with slope 1/5 -/
def line_k (x y : ℝ) : Prop := y = (1/5) * x

theorem point_on_line_k (x : ℝ) : 
  line_k x 1 → x = 5 := by
  sorry

end point_on_line_k_l2402_240219


namespace factorization_exists_l2402_240215

theorem factorization_exists : ∃ (a b c : ℤ), ∀ x : ℝ,
  (x - a) * (x - 10) + 1 = (x + b) * (x + c) := by
  sorry

end factorization_exists_l2402_240215


namespace nabla_calculation_l2402_240282

def nabla (a b : ℕ) : ℕ := 3 + (Nat.factorial b) ^ a

theorem nabla_calculation : nabla (nabla 2 3) 4 = 3 + 24 ^ 39 := by
  sorry

end nabla_calculation_l2402_240282


namespace decimal_binary_equality_l2402_240246

-- Define a function to convert decimal to binary
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec toBinary (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

-- Define a function to convert binary to decimal
def binaryToDecimal (bits : List Nat) : Nat :=
  bits.foldl (fun acc bit => 2 * acc + bit) 0

-- Theorem statement
theorem decimal_binary_equality :
  (decimalToBinary 25 ≠ [1, 0, 1, 1, 0]) ∧
  (decimalToBinary 13 = [1, 1, 0, 1]) ∧
  (decimalToBinary 11 ≠ [1, 1, 0, 0]) ∧
  (decimalToBinary 10 ≠ [1, 0]) :=
by sorry

end decimal_binary_equality_l2402_240246


namespace plane_flight_time_l2402_240274

/-- Given a plane flying between two cities, prove that the return trip takes 84 minutes -/
theorem plane_flight_time (d : ℝ) (p : ℝ) (w : ℝ) :
  d > 0 ∧ p > 0 ∧ w > 0 ∧ p > w → -- Positive distance, plane speed, and wind speed
  d / (p - w) = 96 → -- Trip against wind takes 96 minutes
  d / (p + w) = d / p - 6 → -- Return trip is 6 minutes less than in still air
  d / (p + w) = 84 := by
  sorry

end plane_flight_time_l2402_240274


namespace count_negative_numbers_l2402_240250

def number_list : List ℚ := [-14, 7, 0, -2/3, -5/16]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 3 := by sorry

end count_negative_numbers_l2402_240250


namespace altitude_equation_correct_l2402_240222

-- Define the triangle vertices
def A : ℝ × ℝ := (-5, 3)
def B : ℝ × ℝ := (3, 7)
def C : ℝ × ℝ := (4, -1)

-- Define the vector BC
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Define the altitude equation
def altitudeEquation (x y : ℝ) : Prop := x - 8*y + 29 = 0

-- Theorem statement
theorem altitude_equation_correct :
  ∀ x y : ℝ, altitudeEquation x y ↔
  ((x - A.1, y - A.2) • BC = 0 ∧ ∃ t : ℝ, (x, y) = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2))) :=
by sorry

end altitude_equation_correct_l2402_240222


namespace lollipop_cost_theorem_lollipop_cost_correct_l2402_240217

/-- The cost of n lollipops given that 2 lollipops cost $2.40 and 6 lollipops cost $7.20 -/
def lollipop_cost (n : ℕ) : ℚ :=
  1.20 * n

/-- Theorem stating that the lollipop_cost function satisfies the given conditions -/
theorem lollipop_cost_theorem :
  lollipop_cost 2 = 2.40 ∧ lollipop_cost 6 = 7.20 :=
by sorry

/-- Theorem proving that the lollipop_cost function is correct for all non-negative integers -/
theorem lollipop_cost_correct (n : ℕ) :
  lollipop_cost n = 1.20 * n :=
by sorry

end lollipop_cost_theorem_lollipop_cost_correct_l2402_240217


namespace add_preserves_inequality_l2402_240273

theorem add_preserves_inequality (a b c : ℝ) (h : a > b) : a + c > b + c := by
  sorry

end add_preserves_inequality_l2402_240273


namespace trigonometric_identities_l2402_240297

theorem trigonometric_identities :
  (((Real.tan (10 * π / 180)) * (Real.tan (70 * π / 180))) /
   ((Real.tan (70 * π / 180)) - (Real.tan (10 * π / 180)) + (Real.tan (120 * π / 180))) = Real.sqrt 3 / 3) ∧
  ((2 * (Real.cos (40 * π / 180)) + (Real.cos (10 * π / 180)) * (1 + Real.sqrt 3 * (Real.tan (10 * π / 180)))) /
   (Real.sqrt (1 + Real.cos (10 * π / 180))) = 2) := by
  sorry

end trigonometric_identities_l2402_240297


namespace average_of_B_and_C_l2402_240213

theorem average_of_B_and_C (A B C : ℕ) : 
  A + B + C = 111 →
  (A + B) / 2 = 31 →
  (A + C) / 2 = 37 →
  (B + C) / 2 = 43 := by
sorry

end average_of_B_and_C_l2402_240213


namespace problem_solution_l2402_240251

theorem problem_solution (a b x : ℝ) 
  (h1 : a * (x + 2) + b * (x + 2) = 60) 
  (h2 : a + b = 12) : 
  x = 3 := by
sorry

end problem_solution_l2402_240251


namespace opposite_of_2023_l2402_240233

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- The theorem states that the opposite of 2023 is -2023. -/
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end opposite_of_2023_l2402_240233


namespace max_min_sum_f_l2402_240220

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + x) / (x^2 + 1)

theorem max_min_sum_f :
  (∃ M m : ℝ, (∀ x : ℝ, f x ≤ M ∧ m ≤ f x) ∧ (∃ x₁ x₂ : ℝ, f x₁ = M ∧ f x₂ = m) ∧ M + m = 2) :=
sorry

end max_min_sum_f_l2402_240220


namespace ram_exam_result_l2402_240223

/-- The percentage of marks Ram got in his exam -/
def ram_percentage (marks_obtained : ℕ) (total_marks : ℕ) : ℚ :=
  (marks_obtained : ℚ) / (total_marks : ℚ) * 100

/-- Theorem stating that Ram's percentage is 90% -/
theorem ram_exam_result : ram_percentage 450 500 = 90 := by
  sorry

end ram_exam_result_l2402_240223


namespace solution_sum_l2402_240209

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

theorem solution_sum (a b : ℝ) :
  a > b →
  f (a^2) * f a = 0.72 →
  f (b^2) * f b = 0.72 →
  19 * a + 7 * b = 134 := by
sorry

end solution_sum_l2402_240209


namespace eight_vases_needed_l2402_240206

/-- Represents the number of flowers of each type -/
structure FlowerCounts where
  roses : ℕ
  tulips : ℕ
  lilies : ℕ

/-- Represents the capacity of a vase for each flower type -/
structure VaseCapacity where
  roses : ℕ
  tulips : ℕ
  lilies : ℕ

/-- Calculates the minimum number of vases needed -/
def minVasesNeeded (flowers : FlowerCounts) (capacity : VaseCapacity) : ℕ :=
  sorry

/-- Theorem stating that 8 vases are needed for the given flower counts -/
theorem eight_vases_needed :
  let flowers := FlowerCounts.mk 20 15 5
  let capacity := VaseCapacity.mk 6 8 4
  minVasesNeeded flowers capacity = 8 := by
  sorry

end eight_vases_needed_l2402_240206


namespace green_eyes_count_l2402_240244

/-- The number of students with green eyes in Mrs. Jensen's preschool class -/
def green_eyes : ℕ := sorry

theorem green_eyes_count : green_eyes = 12 := by
  have total_students : ℕ := 40
  have red_hair : ℕ := 3 * green_eyes
  have both : ℕ := 8
  have neither : ℕ := 4

  have h1 : total_students = (green_eyes - both) + (red_hair - both) + both + neither := by sorry
  
  sorry

end green_eyes_count_l2402_240244


namespace inequality_always_holds_l2402_240200

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end inequality_always_holds_l2402_240200


namespace isosceles_triangle_largest_angle_l2402_240212

theorem isosceles_triangle_largest_angle (α β γ : Real) :
  -- The triangle is isosceles
  (α = β) →
  -- One of the angles opposite an equal side is 50°
  α = 50 →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 80°
  γ = 80 :=
by sorry

end isosceles_triangle_largest_angle_l2402_240212


namespace ring_with_finite_zero_divisors_is_finite_l2402_240289

/-- A ring with at least one non-zero zero divisor and finitely many zero divisors is finite. -/
theorem ring_with_finite_zero_divisors_is_finite (R : Type*) [Ring R]
  (h1 : ∃ (x y : R), x ≠ 0 ∧ y ≠ 0 ∧ x * y = 0)
  (h2 : Set.Finite {x : R | ∃ y, y ≠ 0 ∧ x * y = 0 ∨ y * x = 0}) :
  Set.Finite (Set.univ : Set R) := by
  sorry

end ring_with_finite_zero_divisors_is_finite_l2402_240289


namespace sum_of_f_92_and_neg_92_l2402_240235

/-- Given a polynomial function f(x) = ax^7 + bx^5 - cx^3 + dx + 3 where f(92) = 2,
    prove that f(92) + f(-92) = 6 -/
theorem sum_of_f_92_and_neg_92 (a b c d : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^7 + b * x^5 - c * x^3 + d * x + 3) 
  (h2 : f 92 = 2) : 
  f 92 + f (-92) = 6 := by
  sorry

end sum_of_f_92_and_neg_92_l2402_240235


namespace probability_at_least_twice_value_l2402_240208

def single_shot_probability : ℝ := 0.6
def number_of_shots : ℕ := 3

def probability_at_least_twice : ℝ :=
  (Nat.choose number_of_shots 2) * (single_shot_probability ^ 2) * (1 - single_shot_probability) +
  (Nat.choose number_of_shots 3) * (single_shot_probability ^ 3)

theorem probability_at_least_twice_value : 
  probability_at_least_twice = 81 / 125 := by
  sorry

end probability_at_least_twice_value_l2402_240208


namespace train_length_l2402_240241

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 24 → ∃ length : ℝ, 
  (abs (length - 199.92) < 0.01 ∧ length = speed * (1000 / 3600) * time) := by
  sorry

end train_length_l2402_240241


namespace assignment_count_l2402_240286

theorem assignment_count : 
  (∀ n : ℕ, n = 8 → ∀ k : ℕ, k = 4 → (k : ℕ) ^ n = 65536) := by sorry

end assignment_count_l2402_240286


namespace machine_production_l2402_240203

/-- The number of shirts a machine can make per minute -/
def shirts_per_minute : ℕ := 8

/-- The number of minutes the machine worked -/
def minutes_worked : ℕ := 2

/-- The number of shirts made by the machine -/
def shirts_made : ℕ := shirts_per_minute * minutes_worked

theorem machine_production :
  shirts_made = 16 := by sorry

end machine_production_l2402_240203


namespace intersection_of_A_and_B_l2402_240277

def set_A : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}
def set_B : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l2402_240277


namespace line_intercepts_sum_l2402_240266

/-- Given a line with equation y - 6 = -2(x - 3), 
    the sum of its x-intercept and y-intercept is 18 -/
theorem line_intercepts_sum : 
  ∀ (x y : ℝ), y - 6 = -2 * (x - 3) → 
  ∃ (x_int y_int : ℝ), 
    (y_int - 6 = -2 * (x_int - 3) ∧ y_int = 0) ∧
    (0 - 6 = -2 * (0 - 3) ∧ y_int = 0) ∧
    x_int + y_int = 18 := by
  sorry

end line_intercepts_sum_l2402_240266


namespace cheryl_prob_correct_l2402_240252

/-- Represents the number of marbles of each color in the box -/
def marbles_per_color : ℕ := 3

/-- Represents the number of colors of marbles in the box -/
def num_colors : ℕ := 4

/-- Represents the total number of marbles in the box -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- Represents the number of marbles each person draws -/
def marbles_drawn : ℕ := 3

/-- Represents the probability of Cheryl getting 3 marbles of the same color,
    given that Claudia did not draw 3 marbles of the same color -/
def cheryl_same_color_prob : ℚ := 55 / 1540

theorem cheryl_prob_correct :
  cheryl_same_color_prob =
    (num_colors - 1) * (Nat.choose total_marbles marbles_drawn) /
    (Nat.choose total_marbles marbles_drawn *
     (Nat.choose (total_marbles - marbles_drawn) marbles_drawn -
      num_colors * 1) * 1) :=
by sorry

end cheryl_prob_correct_l2402_240252


namespace arithmetic_progression_sum_l2402_240295

/-- 
Given an arithmetic progression with sum of n terms equal to 220,
common difference 3, first term an integer, and n > 1,
prove that the sum of the first 10 terms is 215.
-/
theorem arithmetic_progression_sum (n : ℕ) (a : ℤ) :
  n > 1 →
  (n : ℝ) * (a + (n - 1) * 3 / 2) = 220 →
  10 * (a + (10 - 1) * 3 / 2) = 215 :=
by sorry

end arithmetic_progression_sum_l2402_240295


namespace simple_interest_rate_l2402_240264

/-- Given a simple interest loan where:
    - The interest after 10 years is 1500
    - The principal amount is 1250
    Prove that the interest rate is 12% --/
theorem simple_interest_rate : 
  ∀ (rate : ℝ),
  (1250 * rate * 10 / 100 = 1500) →
  rate = 12 := by
  sorry

end simple_interest_rate_l2402_240264


namespace smallest_bound_for_cubic_coefficient_smallest_k_is_four_l2402_240298

-- Define the set of polynomials M
def M : Set (ℝ → ℝ) :=
  {P | ∃ (a b c d : ℝ), ∀ x, P x = a * x^3 + b * x^2 + c * x + d ∧ 
                         ∀ x ∈ Set.Icc (-1 : ℝ) 1, |P x| ≤ 1}

-- State the theorem
theorem smallest_bound_for_cubic_coefficient :
  ∃ k, (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) → |a| ≤ k) ∧
       (∀ k' < k, ∃ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| > k') :=
by
  -- The proof goes here
  sorry

-- State that the smallest k is 4
theorem smallest_k_is_four :
  ∃! k, (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) → |a| ≤ k) ∧
       (∀ k' < k, ∃ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| > k') ∧
       k = 4 :=
by
  -- The proof goes here
  sorry

end smallest_bound_for_cubic_coefficient_smallest_k_is_four_l2402_240298


namespace best_estimate_on_number_line_l2402_240294

theorem best_estimate_on_number_line (x : ℝ) (h1 : x < 0) (h2 : -2 < x) (h3 : x < -1) :
  let options := [1.3, -1.3, -2.7, 0.7, -0.7]
  (-1.3 : ℝ) = options.argmin (fun y => |x - y|) := by
  sorry

end best_estimate_on_number_line_l2402_240294


namespace bucket_fill_time_l2402_240240

/-- Given that two-thirds of a bucket is filled in 100 seconds,
    prove that the time taken to fill the bucket completely is 150 seconds. -/
theorem bucket_fill_time (fill_rate : ℝ) (h : fill_rate * (2/3) = 1/100) :
  (1 / fill_rate) = 150 := by
  sorry

end bucket_fill_time_l2402_240240


namespace inequality_implies_sign_conditions_l2402_240210

theorem inequality_implies_sign_conditions (a b : ℝ) 
  (h : (|abs a - (a + b)| : ℝ) < |a - abs (a + b)|) : 
  a < 0 ∧ b > 0 := by
  sorry

end inequality_implies_sign_conditions_l2402_240210


namespace students_speaking_both_languages_l2402_240218

/-- Theorem: In a class of 150 students, given that 55 speak English, 85 speak Telugu, 
    and 30 speak neither English nor Telugu, prove that 20 students speak both English and Telugu. -/
theorem students_speaking_both_languages (total : ℕ) (english : ℕ) (telugu : ℕ) (neither : ℕ) :
  total = 150 →
  english = 55 →
  telugu = 85 →
  neither = 30 →
  english + telugu - (total - neither) = 20 :=
by
  sorry


end students_speaking_both_languages_l2402_240218


namespace expression_value_l2402_240226

theorem expression_value (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : c - d = -3) : 
  (b - c) - (-d - a) = 7 := by
  sorry

end expression_value_l2402_240226


namespace initial_distance_calculation_l2402_240225

/-- Represents the scenario of two trucks traveling on the same route --/
structure TruckScenario where
  initial_distance : ℝ
  speed_x : ℝ
  speed_y : ℝ
  overtake_time : ℝ
  final_distance : ℝ

/-- Theorem stating the initial distance between trucks given the scenario conditions --/
theorem initial_distance_calculation (scenario : TruckScenario)
  (h1 : scenario.speed_x = 57)
  (h2 : scenario.speed_y = 63)
  (h3 : scenario.overtake_time = 3)
  (h4 : scenario.final_distance = 4)
  (h5 : scenario.speed_y > scenario.speed_x) :
  scenario.initial_distance = 14 := by
  sorry


end initial_distance_calculation_l2402_240225


namespace kamal_chemistry_marks_l2402_240272

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  biology : ℕ
  chemistry : ℕ

/-- Calculates the average marks for a student -/
def average (marks : StudentMarks) : ℚ :=
  (marks.english + marks.mathematics + marks.physics + marks.biology + marks.chemistry) / 5

/-- Theorem: Given Kamal's marks and average, his Chemistry marks must be 62 -/
theorem kamal_chemistry_marks :
  ∀ (kamal : StudentMarks),
    kamal.english = 66 →
    kamal.mathematics = 65 →
    kamal.physics = 77 →
    kamal.biology = 75 →
    average kamal = 69 →
    kamal.chemistry = 62 := by
  sorry

end kamal_chemistry_marks_l2402_240272


namespace smallest_angle_measure_l2402_240236

/-- A trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference

/-- The properties of an arithmetic trapezoid -/
def ArithmeticTrapezoid.valid (t : ArithmeticTrapezoid) : Prop :=
  -- Sum of interior angles is 360°
  t.a + (t.a + t.d) + (t.a + 2*t.d) + (t.a + 3*t.d) = 360 ∧
  -- Largest angle is 150°
  t.a + 3*t.d = 150

theorem smallest_angle_measure (t : ArithmeticTrapezoid) (h : t.valid) :
  t.a = 15 := by sorry

end smallest_angle_measure_l2402_240236


namespace beau_and_sons_ages_equality_l2402_240211

/-- Represents the problem of finding when Beau's age equaled the sum of his sons' ages --/
theorem beau_and_sons_ages_equality (beau_age_today : ℕ) (sons_age_today : ℕ) : 
  beau_age_today = 42 →
  sons_age_today = 16 →
  ∃ (years_ago : ℕ), 
    beau_age_today - years_ago = 3 * (sons_age_today - years_ago) ∧
    years_ago = 3 := by
sorry

end beau_and_sons_ages_equality_l2402_240211


namespace quadratic_roots_property_integer_values_k_l2402_240224

theorem quadratic_roots_property (k : ℝ) (x₁ x₂ : ℝ) : 
  (4 * k * x₁^2 - 4 * k * x₁ + k + 1 = 0 ∧ 
   4 * k * x₂^2 - 4 * k * x₂ + k + 1 = 0) → 
  (2 * x₁ - x₂) * (x₁ - 2 * x₂) ≠ -3/2 :=
sorry

theorem integer_values_k (k : ℤ) :
  (∃ x₁ x₂ : ℝ, 4 * (k : ℝ) * x₁^2 - 4 * (k : ℝ) * x₁ + (k : ℝ) + 1 = 0 ∧
                4 * (k : ℝ) * x₂^2 - 4 * (k : ℝ) * x₂ + (k : ℝ) + 1 = 0 ∧
                ∃ n : ℤ, (x₁ / x₂ + x₂ / x₁ - 2 : ℝ) = n) ↔
  k = -2 ∨ k = -3 ∨ k = -5 :=
sorry

end quadratic_roots_property_integer_values_k_l2402_240224


namespace same_terminal_side_l2402_240245

theorem same_terminal_side (k : ℤ) : 
  ∃ k : ℤ, -390 = k * 360 + 330 := by
  sorry

end same_terminal_side_l2402_240245


namespace triangle_coordinates_l2402_240281

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  P : Point
  Q : Point
  R : Point

/-- Predicate to check if a line segment is horizontal -/
def isHorizontal (p1 p2 : Point) : Prop :=
  p1.y = p2.y

/-- Predicate to check if a line segment is vertical -/
def isVertical (p1 p2 : Point) : Prop :=
  p1.x = p2.x

theorem triangle_coordinates (t : Triangle) 
  (h1 : isHorizontal t.P t.R)
  (h2 : isVertical t.P t.Q)
  (h3 : t.R.y = -2)
  (h4 : t.Q.x = -11) :
  t.P.x = -11 ∧ t.P.y = -2 := by
  sorry

end triangle_coordinates_l2402_240281


namespace cruise_ship_tourists_l2402_240267

theorem cruise_ship_tourists : ∃ (x : ℕ) (tourists : ℕ), 
  x > 1 ∧ 
  tourists = 12 * x + 1 ∧
  ∃ (y : ℕ), y ≤ 15 ∧ tourists = y * (x - 1) ∧
  tourists = 169 := by
  sorry

end cruise_ship_tourists_l2402_240267


namespace dart_score_is_75_l2402_240292

/-- The final score of three dart throws -/
def final_score (bullseye : ℕ) (half_bullseye : ℕ) (miss : ℕ) : ℕ :=
  bullseye + half_bullseye + miss

/-- Theorem stating that the final score is 75 points -/
theorem dart_score_is_75 :
  ∃ (bullseye half_bullseye miss : ℕ),
    bullseye = 50 ∧
    half_bullseye = bullseye / 2 ∧
    miss = 0 ∧
    final_score bullseye half_bullseye miss = 75 := by
  sorry

end dart_score_is_75_l2402_240292


namespace quadratic_roots_property_l2402_240242

-- Define a quadratic trinomial
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Theorem statement
theorem quadratic_roots_property (a b c : ℝ) (h : discriminant a b c ≥ 0) :
  ∃ (x : ℝ), ¬(∀ (y : ℝ), discriminant (a^2) (b^2) (c^2) ≥ 0) ∧
  (∀ (z : ℝ), discriminant (a^3) (b^3) (c^3) ≥ 0) := by sorry

end quadratic_roots_property_l2402_240242


namespace binomial_expectation_and_variance_l2402_240278

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_expectation_and_variance :
  ∃ (X : BinomialRV), X.n = 10 ∧ X.p = 0.6 ∧ expected_value X = 6 ∧ variance X = 2.4 := by
  sorry

end binomial_expectation_and_variance_l2402_240278


namespace cos_negative_seventy_nine_pi_sixths_l2402_240204

theorem cos_negative_seventy_nine_pi_sixths :
  Real.cos (-79 * Real.pi / 6) = -Real.sqrt 3 / 2 := by sorry

end cos_negative_seventy_nine_pi_sixths_l2402_240204


namespace greatest_prime_factor_of_sum_l2402_240234

theorem greatest_prime_factor_of_sum (p : ℕ) :
  (∃ (q : ℕ), Nat.Prime q ∧ q ∣ (5^7 + 6^6) ∧ q ≥ p) →
  p ≤ 211 :=
sorry

end greatest_prime_factor_of_sum_l2402_240234


namespace greatest_integer_less_than_negative_eight_thirds_l2402_240207

theorem greatest_integer_less_than_negative_eight_thirds :
  Int.floor (-8/3 : ℚ) = -3 :=
sorry

end greatest_integer_less_than_negative_eight_thirds_l2402_240207


namespace intersection_range_l2402_240280

/-- Hyperbola C centered at the origin with right focus at (2,0) and real axis length 2√3 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- Line l with equation y = kx + √2 -/
def line_l (k x : ℝ) (y : ℝ) : Prop := y = k * x + Real.sqrt 2

/-- Predicate to check if a point (x, y) is on the left branch of hyperbola C -/
def on_left_branch (x y : ℝ) : Prop := hyperbola_C x y ∧ x < 0

/-- Theorem stating the range of k for which line l intersects the left branch of hyperbola C at two points -/
theorem intersection_range (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    on_left_branch x₁ y₁ ∧ 
    on_left_branch x₂ y₂ ∧ 
    line_l k x₁ y₁ ∧ 
    line_l k x₂ y₂) ↔ 
  Real.sqrt 3 / 3 < k ∧ k < 1 :=
sorry

end intersection_range_l2402_240280


namespace symmetry_points_l2402_240285

/-- Given points M, N, P, and Q in a 2D plane, prove that Q has coordinates (b,a) -/
theorem symmetry_points (a b : ℝ) : 
  let M : ℝ × ℝ := (a, b)
  let N : ℝ × ℝ := (a, -b)  -- M symmetric to N w.r.t. x-axis
  let P : ℝ × ℝ := (-a, -b) -- P symmetric to N w.r.t. y-axis
  let Q : ℝ × ℝ := (b, a)   -- Q symmetric to P w.r.t. line x+y=0
  Q = (b, a) := by sorry

end symmetry_points_l2402_240285


namespace zero_in_interval_l2402_240221

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end zero_in_interval_l2402_240221


namespace right_triangle_perimeter_l2402_240247

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 15 →
  a^2 + b^2 = c^2 →
  a + b + c = 60 := by
sorry

end right_triangle_perimeter_l2402_240247


namespace angle_bisector_exists_l2402_240268

/-- A ruler with constant width and parallel edges -/
structure ConstantWidthRuler where
  width : ℝ
  width_positive : width > 0

/-- An angle in a plane -/
structure Angle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop

/-- A line in a plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a line bisects an angle -/
def bisects (l : Line) (a : Angle) : Prop :=
  sorry

/-- Predicate to check if a line can be constructed using a constant width ruler -/
def constructible_with_ruler (l : Line) (r : ConstantWidthRuler) : Prop :=
  sorry

/-- Theorem stating that for any angle, there exists a bisector constructible with a constant width ruler -/
theorem angle_bisector_exists (a : Angle) (r : ConstantWidthRuler) :
  ∃ l : Line, bisects l a ∧ constructible_with_ruler l r := by
  sorry

end angle_bisector_exists_l2402_240268


namespace negation_of_universal_proposition_l2402_240256

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end negation_of_universal_proposition_l2402_240256


namespace fraction_value_given_condition_l2402_240288

theorem fraction_value_given_condition (a b : ℝ) 
  (h : |a + 2| + Real.sqrt (b - 4) = 0) : a^2 / b = 1 := by
  sorry

end fraction_value_given_condition_l2402_240288


namespace two_digit_product_equals_concatenation_l2402_240254

def has_same_digits (a b : ℕ) : Prop :=
  (Nat.log 10 a).succ = (Nat.log 10 b).succ

def concatenate (a b : ℕ) : ℕ :=
  a * (10 ^ ((Nat.log 10 b).succ)) + b

theorem two_digit_product_equals_concatenation :
  ∀ A B : ℕ,
    A > 0 ∧ B > 0 →
    has_same_digits A B →
    2 * A * B = concatenate A B →
    (A = 3 ∧ B = 6) ∨ (A = 13 ∧ B = 52) :=
by sorry

end two_digit_product_equals_concatenation_l2402_240254


namespace arithmetic_progression_implies_linear_l2402_240231

/-- A function f: ℚ → ℚ satisfies the arithmetic progression property if
    for all rational numbers x < y < z < t in arithmetic progression,
    f(y) + f(z) = f(x) + f(t) -/
def ArithmeticProgressionProperty (f : ℚ → ℚ) : Prop :=
  ∀ (x y z t : ℚ), x < y ∧ y < z ∧ z < t ∧ (y - x = z - y) ∧ (z - y = t - z) →
    f y + f z = f x + f t

/-- The main theorem stating that any function satisfying the arithmetic progression property
    is a linear function -/
theorem arithmetic_progression_implies_linear
  (f : ℚ → ℚ) (h : ArithmeticProgressionProperty f) :
  ∃ (C : ℚ), ∀ (x : ℚ), f x = C * x := by
  sorry

end arithmetic_progression_implies_linear_l2402_240231


namespace pen_profit_calculation_l2402_240296

theorem pen_profit_calculation (total_pens : ℕ) (buy_price sell_price : ℚ) (target_profit : ℚ) :
  total_pens = 2000 →
  buy_price = 15/100 →
  sell_price = 30/100 →
  target_profit = 120 →
  ∃ (sold_pens : ℕ), 
    sold_pens ≤ total_pens ∧ 
    (↑sold_pens * sell_price) - (↑total_pens * buy_price) = target_profit ∧
    sold_pens = 1400 :=
by sorry

end pen_profit_calculation_l2402_240296


namespace evaluate_expressions_l2402_240232

-- Define the logarithm functions
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10
noncomputable def ln (x : ℝ) := Real.log x

-- Define the main theorem
theorem evaluate_expressions :
  (2 * Real.sqrt 3 * (12 ^ (1/6)) * (3 ^ (3/2)) = 6) ∧
  ((1/2) * lg 25 + lg 2 + ln (Real.sqrt (Real.exp 1)) - 
   (Real.log 27 / Real.log 2) * (Real.log 2 / Real.log 3) - 
   7 ^ (Real.log 3 / Real.log 7) = -9/2) := by
sorry

end evaluate_expressions_l2402_240232


namespace evaluate_g_l2402_240265

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 8

theorem evaluate_g : 3 * g 2 + 4 * g (-2) = 152 := by
  sorry

end evaluate_g_l2402_240265


namespace connie_blue_markers_l2402_240262

/-- Given that Connie has 2315 red markers and 3343 markers in total, 
    prove that she has 1028 blue markers. -/
theorem connie_blue_markers 
  (total_markers : ℕ) 
  (red_markers : ℕ) 
  (h1 : total_markers = 3343)
  (h2 : red_markers = 2315) :
  total_markers - red_markers = 1028 := by
  sorry

end connie_blue_markers_l2402_240262


namespace max_product_roots_quadratic_l2402_240291

/-- Given a quadratic equation 6x^2 - 12x + m = 0 with real roots,
    the maximum value of m that maximizes the product of the roots is 6. -/
theorem max_product_roots_quadratic :
  ∀ m : ℝ,
  (∃ x y : ℝ, 6 * x^2 - 12 * x + m = 0 ∧ 6 * y^2 - 12 * y + m = 0 ∧ x ≠ y) →
  (∀ k : ℝ, (∃ x y : ℝ, 6 * x^2 - 12 * x + k = 0 ∧ 6 * y^2 - 12 * y + k = 0 ∧ x ≠ y) →
    m / 6 ≥ k / 6) →
  m = 6 :=
by sorry

end max_product_roots_quadratic_l2402_240291


namespace measure_six_liters_possible_l2402_240230

/-- Represents the state of milk distribution among containers -/
structure MilkState :=
  (container : ℕ)
  (jug9 : ℕ)
  (jug5 : ℕ)
  (bucket10 : ℕ)

/-- Represents a pouring action between two containers -/
inductive PourAction
  | ContainerTo9
  | ContainerTo5
  | NineToContainer
  | NineTo10
  | NineTo5
  | FiveTo9
  | FiveTo10
  | FiveToContainer

/-- Applies a pouring action to a milk state -/
def applyAction (state : MilkState) (action : PourAction) : MilkState :=
  sorry

/-- Checks if the given sequence of actions results in 6 liters in the 10-liter bucket -/
def isValidSolution (actions : List PourAction) : Bool :=
  sorry

/-- Proves that it's possible to measure out 6 liters using given containers -/
theorem measure_six_liters_possible :
  ∃ (actions : List PourAction), isValidSolution actions = true :=
sorry

end measure_six_liters_possible_l2402_240230


namespace family_income_increase_l2402_240229

theorem family_income_increase (I : ℝ) (S M F G : ℝ) : 
  I > 0 →
  S = 0.05 * I →
  M = 0.15 * I →
  F = 0.25 * I →
  G = I - S - M - F →
  (2 * G - G) / I = 0.55 := by
sorry

end family_income_increase_l2402_240229


namespace vector_perpendicular_to_sum_l2402_240284

/-- Given vectors a and b in ℝ², prove that a is perpendicular to (a + b) -/
theorem vector_perpendicular_to_sum (a b : ℝ × ℝ) (ha : a = (2, -1)) (hb : b = (1, 7)) :
  a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0 := by
  sorry

#check vector_perpendicular_to_sum

end vector_perpendicular_to_sum_l2402_240284


namespace dave_apps_left_l2402_240259

/-- The number of apps Dave had left on his phone after adding and deleting apps. -/
def apps_left (initial_apps new_apps : ℕ) : ℕ :=
  initial_apps + new_apps - (new_apps + 1)

/-- Theorem stating that Dave had 14 apps left on his phone. -/
theorem dave_apps_left : apps_left 15 71 = 14 := by
  sorry

end dave_apps_left_l2402_240259


namespace reflection_line_sum_l2402_240258

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The midpoint of the original and reflected points lies on the line
    y = m * x + b ∧ 
    x = (2 + 10) / 2 ∧ 
    y = (3 + 7) / 2 ∧
    -- The line is perpendicular to the line segment between the original and reflected points
    m * ((10 - 2) / (7 - 3)) = -1) → 
  m + b = 15 := by
sorry

end reflection_line_sum_l2402_240258


namespace caterpillar_length_difference_l2402_240248

theorem caterpillar_length_difference :
  let green_length : ℝ := 3
  let orange_length : ℝ := 1.17
  green_length - orange_length = 1.83 := by
sorry

end caterpillar_length_difference_l2402_240248


namespace unique_top_coloring_l2402_240201

/-- Represents the colors used for the cube corners -/
inductive Color
  | Red
  | Green
  | Blue
  | Purple

/-- Represents a corner of the cube -/
structure Corner where
  position : Fin 8
  color : Color

/-- Represents a cube with colored corners -/
structure ColoredCube where
  corners : Fin 8 → Corner

/-- Checks if all corners on a face have different colors -/
def faceHasDifferentColors (cube : ColoredCube) (face : Fin 6) : Prop := sorry

/-- Checks if the bottom four corners of the cube are colored with four different colors -/
def bottomCornersAreDifferent (cube : ColoredCube) : Prop := sorry

/-- The main theorem stating that there is only one way to color the top corners -/
theorem unique_top_coloring (cube : ColoredCube) : 
  bottomCornersAreDifferent cube →
  (∀ face, faceHasDifferentColors cube face) →
  ∃! topColoring : Fin 4 → Color, 
    ∀ i : Fin 4, (cube.corners (i + 4)).color = topColoring i :=
sorry

end unique_top_coloring_l2402_240201


namespace water_speed_l2402_240287

/-- The speed of water given a swimmer's speed in still water and time taken to swim against the current -/
theorem water_speed (swim_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : swim_speed = 6)
  (h2 : distance = 14) (h3 : time = 3.5) :
  ∃ (water_speed : ℝ), water_speed = 2 ∧ distance = (swim_speed - water_speed) * time := by
  sorry

end water_speed_l2402_240287


namespace min_value_of_expression_l2402_240255

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geometric_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  (∀ x y : ℝ, x > 0 → y > 0 → 
    Real.sqrt 2 = Real.sqrt (4^x * 2^y) → 1/x + 2/y ≥ 1/a + 2/b) →
  1/a + 2/b = 8 :=
sorry

end min_value_of_expression_l2402_240255


namespace quadratic_rewrite_l2402_240228

theorem quadratic_rewrite :
  ∃ (a b c : ℤ), a > 0 ∧
  (∀ x, 64 * x^2 + 80 * x - 72 = 0 ↔ (a * x + b)^2 = c) ∧
  a + b + c = 110 := by
  sorry

end quadratic_rewrite_l2402_240228


namespace special_function_property_l2402_240279

/-- A continuous function f: ℝ → ℝ satisfying f(x) · f(f(x)) = 1 for all real x, and f(1000) = 999 -/
def special_function (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ 
  (∀ x : ℝ, f x * f (f x) = 1) ∧
  f 1000 = 999

theorem special_function_property (f : ℝ → ℝ) (h : special_function f) : 
  f 500 = 1 / 500 := by
  sorry

end special_function_property_l2402_240279


namespace sheetrock_length_l2402_240202

/-- Represents the properties of a rectangular sheetrock -/
structure Sheetrock where
  width : ℝ
  area : ℝ

/-- Theorem stating that a sheetrock with width 5 and area 30 has length 6 -/
theorem sheetrock_length (s : Sheetrock) (h1 : s.width = 5) (h2 : s.area = 30) :
  s.area / s.width = 6 := by
  sorry


end sheetrock_length_l2402_240202


namespace smallest_prime_with_digit_sum_23_l2402_240238

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 887 :=
sorry

end smallest_prime_with_digit_sum_23_l2402_240238


namespace solve_linear_equation_l2402_240216

theorem solve_linear_equation (x : ℝ) :
  2*x - 3*x + 5*x = 80 → x = 20 := by
  sorry

end solve_linear_equation_l2402_240216


namespace facebook_bonus_percentage_l2402_240260

/-- Represents the Facebook employee bonus problem -/
theorem facebook_bonus_percentage (total_employees : ℕ) 
  (annual_earnings : ℝ) (non_mother_women : ℕ) (bonus_per_mother : ℝ) :
  total_employees = 3300 →
  annual_earnings = 5000000 →
  non_mother_women = 1200 →
  bonus_per_mother = 1250 →
  (((total_employees * 2 / 3 - non_mother_women) * bonus_per_mother) / annual_earnings) * 100 = 25 := by
  sorry


end facebook_bonus_percentage_l2402_240260


namespace xena_head_start_l2402_240253

/-- Xena's running speed in feet per second -/
def xena_speed : ℝ := 15

/-- Dragon's flying speed in feet per second -/
def dragon_speed : ℝ := 30

/-- Time Xena has to reach the cave in seconds -/
def time_to_cave : ℝ := 32

/-- Minimum safe distance between Xena and the dragon in feet -/
def safe_distance : ℝ := 120

/-- Theorem stating Xena's head start distance -/
theorem xena_head_start : 
  xena_speed * time_to_cave + 360 = dragon_speed * time_to_cave - safe_distance := by
  sorry

end xena_head_start_l2402_240253
