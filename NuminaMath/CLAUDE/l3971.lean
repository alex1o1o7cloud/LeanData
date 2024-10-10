import Mathlib

namespace mrs_hilt_pennies_l3971_397121

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes : ℕ) : ℕ :=
  pennies * coin_value "penny" + 
  nickels * coin_value "nickel" + 
  dimes * coin_value "dime"

theorem mrs_hilt_pennies : 
  ∃ (p : ℕ), 
    total_value p 2 2 - total_value 4 1 1 = 13 ∧ 
    p = 2 := by
  sorry

end mrs_hilt_pennies_l3971_397121


namespace factorial_ratio_l3971_397112

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end factorial_ratio_l3971_397112


namespace perpendicular_radii_intercept_l3971_397180

/-- Given a circle and a line intersecting it, if the radii to the intersection points are perpendicular, then the y-intercept of the line has specific values. -/
theorem perpendicular_radii_intercept (b : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*x = 0}
  let line := {(x, y) : ℝ × ℝ | y = x + b}
  let C := (2, 0)
  ∃ (M N : ℝ × ℝ), 
    M ∈ circle ∧ M ∈ line ∧
    N ∈ circle ∧ N ∈ line ∧
    M ≠ N ∧
    (M.1 - C.1) * (N.1 - C.1) + (M.2 - C.2) * (N.2 - C.2) = 0 →
    b = 0 ∨ b = -4 := by
  sorry

end perpendicular_radii_intercept_l3971_397180


namespace no_real_roots_quadratic_l3971_397197

theorem no_real_roots_quadratic (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + a ≠ 0) ↔ a > 9/4 := by
sorry

end no_real_roots_quadratic_l3971_397197


namespace existence_of_m_l3971_397196

theorem existence_of_m : ∃ m : ℝ, m ≤ 3 ∧ ∀ x : ℝ, |x - 1| ≤ m → -2 ≤ x ∧ x ≤ 10 := by
  sorry

end existence_of_m_l3971_397196


namespace intersection_of_A_and_B_l3971_397145

def set_A : Set ℝ := {x | x^2 = 1}
def set_B : Set ℝ := {x | x^2 - 2*x - 3 = 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {-1} := by sorry

end intersection_of_A_and_B_l3971_397145


namespace maries_school_students_maries_school_students_proof_l3971_397186

theorem maries_school_students : ℕ → ℕ → Prop :=
  fun m c =>
    m = 4 * c ∧ m + c = 2500 → m = 2000

-- The proof is omitted
theorem maries_school_students_proof : maries_school_students 2000 500 := by
  sorry

end maries_school_students_maries_school_students_proof_l3971_397186


namespace new_students_count_l3971_397152

/-- Calculates the number of new students who came to school during the year. -/
def new_students (initial : ℕ) (left : ℕ) (final : ℕ) : ℕ :=
  final - (initial - left)

/-- Proves that the number of new students who came to school during the year is 42. -/
theorem new_students_count :
  new_students 4 3 43 = 42 := by
  sorry

end new_students_count_l3971_397152


namespace work_completion_time_l3971_397109

/-- 
If two workers can complete a job together in a certain time, 
and one worker can complete it alone in a known time, 
we can determine how long it takes the other worker to complete the job alone.
-/
theorem work_completion_time 
  (total_work : ℝ) 
  (time_together time_a time_b : ℝ) 
  (h1 : time_together > 0)
  (h2 : time_a > 0)
  (h3 : time_b > 0)
  (h4 : total_work / time_together = total_work / time_a + total_work / time_b)
  (h5 : time_together = 5)
  (h6 : time_a = 10) :
  time_b = 10 := by
sorry

end work_completion_time_l3971_397109


namespace code_deciphering_probability_l3971_397147

theorem code_deciphering_probability 
  (prob_A : ℚ) 
  (prob_B : ℚ) 
  (h_A : prob_A = 2 / 3) 
  (h_B : prob_B = 3 / 5) : 
  1 - (1 - prob_A) * (1 - prob_B) = 13 / 15 := by
  sorry

end code_deciphering_probability_l3971_397147


namespace potato_bag_weight_l3971_397176

theorem potato_bag_weight (current_weight : ℝ) (h : current_weight = 12) :
  ∃ (original_weight : ℝ), original_weight / 2 = current_weight ∧ original_weight = 24 := by
  sorry

end potato_bag_weight_l3971_397176


namespace walnut_chestnut_cost_l3971_397189

/-- The total cost of buying walnuts and chestnuts -/
def total_cost (m n : ℝ) : ℝ :=
  2 * m + 3 * n

/-- Theorem: The total cost of buying 2 kg of walnuts at m yuan/kg and 3 kg of chestnuts at n yuan/kg is (2m + 3n) yuan -/
theorem walnut_chestnut_cost (m n : ℝ) :
  total_cost m n = 2 * m + 3 * n :=
by sorry

end walnut_chestnut_cost_l3971_397189


namespace max_value_theorem_max_value_achievable_l3971_397164

theorem max_value_theorem (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 2) ≤ Real.sqrt 29 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 2) = Real.sqrt 29 :=
by sorry

end max_value_theorem_max_value_achievable_l3971_397164


namespace unique_prime_solution_l3971_397105

theorem unique_prime_solution :
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (7 * p * q^2 + p = q^3 + 43 * p^3 + 1) → 
    (p = 2 ∧ q = 7) := by
  sorry

end unique_prime_solution_l3971_397105


namespace power_sum_of_i_l3971_397132

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^11 + i^111 = -2 * i :=
  sorry

end power_sum_of_i_l3971_397132


namespace no_real_roots_l3971_397173

theorem no_real_roots (k : ℝ) (h : 12 - 3 * k < 0) : 
  ∀ x : ℝ, x^2 + 4*x + k ≠ 0 := by
sorry

end no_real_roots_l3971_397173


namespace polynomial_divisibility_l3971_397184

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℝ, (x + 3) * (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) → 
  p = -31 ∧ q = -71 := by
sorry

end polynomial_divisibility_l3971_397184


namespace sum_of_digits_up_to_999_l3971_397125

/-- Sum of digits function for a single number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of all digits in numbers from 0 to n -/
def sumOfAllDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all digits in decimal representations of integers from 0 to 999 is 13500 -/
theorem sum_of_digits_up_to_999 : sumOfAllDigits 999 = 13500 := by sorry

end sum_of_digits_up_to_999_l3971_397125


namespace grade_assignments_l3971_397159

/-- The number of students in the class -/
def num_students : ℕ := 8

/-- The number of distinct grades available -/
def num_grades : ℕ := 4

/-- Theorem: The number of ways to assign grades to students -/
theorem grade_assignments :
  num_grades ^ num_students = 65536 := by sorry

end grade_assignments_l3971_397159


namespace similar_triangles_leg_length_l3971_397179

theorem similar_triangles_leg_length (x : ℝ) : x > 0 →
  (12 : ℝ) / x = 9 / 7 → x = 84 / 9 := by sorry

end similar_triangles_leg_length_l3971_397179


namespace brothers_age_fraction_l3971_397133

theorem brothers_age_fraction :
  let younger_age : ℕ := 27
  let total_age : ℕ := 46
  let older_age : ℕ := total_age - younger_age
  ∃ f : ℚ, younger_age = f * older_age + 10 ∧ f = 17 / 19 := by
  sorry

end brothers_age_fraction_l3971_397133


namespace kolya_purchase_l3971_397137

/-- Represents the cost of an item in kopecks -/
def item_cost (rubles : ℕ) : ℕ := 100 * rubles + 99

/-- Represents the total purchase cost in kopecks -/
def total_cost : ℕ := 200 * 100 + 83

/-- Predicate to check if a given number of items is a valid solution -/
def is_valid_solution (n : ℕ) : Prop :=
  ∃ (rubles : ℕ), n * (item_cost rubles) = total_cost

theorem kolya_purchase :
  ∀ n : ℕ, is_valid_solution n ↔ n = 17 ∨ n = 117 :=
sorry

end kolya_purchase_l3971_397137


namespace complex_simplification_and_multiplication_l3971_397191

theorem complex_simplification_and_multiplication :
  ((-5 + 3 * Complex.I) - (2 - 7 * Complex.I)) * (1 + 2 * Complex.I) = -27 - 4 * Complex.I :=
by sorry

end complex_simplification_and_multiplication_l3971_397191


namespace dave_total_rides_l3971_397166

/-- The number of rides Dave took on the first day -/
def first_day_rides : ℕ := 4

/-- The number of rides Dave took on the second day -/
def second_day_rides : ℕ := 3

/-- The total number of rides Dave took over two days -/
def total_rides : ℕ := first_day_rides + second_day_rides

theorem dave_total_rides : total_rides = 7 := by
  sorry

end dave_total_rides_l3971_397166


namespace max_children_to_movies_l3971_397126

def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 3
def total_budget : ℕ := 35

theorem max_children_to_movies :
  (total_budget - adult_ticket_cost) / child_ticket_cost = 9 :=
sorry

end max_children_to_movies_l3971_397126


namespace roots_quadratic_equation_l3971_397113

theorem roots_quadratic_equation (m : ℝ) (a b : ℝ) (s t : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a^2 + 1/b^2)^2 - s*(a^2 + 1/b^2) + t = 0) →
  ((b^2 + 1/a^2)^2 - s*(b^2 + 1/a^2) + t = 0) →
  t = 100/9 := by
sorry

end roots_quadratic_equation_l3971_397113


namespace tangent_circle_radius_l3971_397174

/-- An isosceles right triangle with hypotenuse 68 and leg 48 -/
structure IsoscelesRightTriangle where
  hypotenuse : ℝ
  leg : ℝ
  hypotenuse_eq : hypotenuse = 68
  leg_eq : leg = 48

/-- A circle inscribed in the right angle of the triangle -/
structure InscribedCircle where
  radius : ℝ
  radius_eq : radius = 12

/-- A circle externally tangent to the inscribed circle and inscribed in the remaining space -/
structure TangentCircle where
  radius : ℝ

/-- The main theorem stating that the radius of the tangent circle is 8 -/
theorem tangent_circle_radius 
  (triangle : IsoscelesRightTriangle) 
  (inscribed : InscribedCircle) 
  (tangent : TangentCircle) : tangent.radius = 8 := by
  sorry


end tangent_circle_radius_l3971_397174


namespace omega_cube_root_unity_l3971_397123

theorem omega_cube_root_unity (ω : ℂ) : 
  ω = -1/2 + (Complex.I * Real.sqrt 3)/2 → ω^2 + ω + 1 = 0 := by
  sorry

end omega_cube_root_unity_l3971_397123


namespace castle_provisions_theorem_l3971_397120

/-- Represents the number of days provisions last given initial conditions and a change in population -/
def days_until_food_runs_out (initial_people : ℕ) (initial_days : ℕ) (days_passed : ℕ) (people_left : ℕ) : ℕ :=
  let remaining_days := initial_days - days_passed
  let new_duration := (remaining_days * initial_people) / people_left
  new_duration

/-- Theorem stating that under given conditions, food lasts for 90 more days after population change -/
theorem castle_provisions_theorem (initial_people : ℕ) (initial_days : ℕ) 
  (days_passed : ℕ) (people_left : ℕ) :
  initial_people = 300 ∧ initial_days = 90 ∧ days_passed = 30 ∧ people_left = 200 →
  days_until_food_runs_out initial_people initial_days days_passed people_left = 90 :=
by
  sorry

#eval days_until_food_runs_out 300 90 30 200

end castle_provisions_theorem_l3971_397120


namespace four_false_statements_l3971_397181

/-- Represents a statement on the card -/
inductive Statement
| one
| two
| three
| four
| all

/-- The truth value of a statement -/
def isFalse : Statement → Bool
| Statement.one => true
| Statement.two => true
| Statement.three => true
| Statement.four => false
| Statement.all => true

/-- The claim made by each statement -/
def claim : Statement → Nat
| Statement.one => 1
| Statement.two => 2
| Statement.three => 3
| Statement.four => 4
| Statement.all => 5

/-- The total number of false statements -/
def totalFalse : Nat := 
  (Statement.one :: Statement.two :: Statement.three :: Statement.four :: Statement.all :: []).filter isFalse |>.length

/-- Theorem stating that exactly 4 statements are false -/
theorem four_false_statements : totalFalse = 4 ∧ 
  ∀ s : Statement, isFalse s = true ↔ claim s ≠ totalFalse :=
  sorry


end four_false_statements_l3971_397181


namespace equal_roots_quadratic_l3971_397195

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - x + m = 0 ∧ (∀ y : ℝ, y^2 - y + m = 0 → y = x)) → m = 1/4 := by
  sorry

end equal_roots_quadratic_l3971_397195


namespace ratio_of_40_to_8_l3971_397163

theorem ratio_of_40_to_8 (certain_number : ℚ) (h : certain_number = 40) : 
  certain_number / 8 = 5 := by
  sorry

end ratio_of_40_to_8_l3971_397163


namespace divisibility_condition_l3971_397102

theorem divisibility_condition (x y : ℕ+) :
  (xy^2 + y + 7 ∣ x^2*y + x + y) ↔ 
  (∃ t : ℕ+, x = 7*t^2 ∧ y = 7*t) ∨ (x = 11 ∧ y = 1) ∨ (x = 49 ∧ y = 1) :=
by sorry

end divisibility_condition_l3971_397102


namespace smith_B_students_l3971_397122

/-- The number of students who received a "B" in Mrs. Smith's class -/
def students_with_B_smith (
  jacob_total : ℕ
  ) (jacob_B : ℕ
  ) (smith_total : ℕ
  ) : ℕ :=
  (smith_total * jacob_B) / jacob_total

theorem smith_B_students (
  jacob_total : ℕ
  ) (jacob_B : ℕ
  ) (smith_total : ℕ
  ) (h1 : jacob_total = 20
  ) (h2 : jacob_B = 8
  ) (h3 : smith_total = 30
  ) : students_with_B_smith jacob_total jacob_B smith_total = 12 := by
  sorry

end smith_B_students_l3971_397122


namespace two_ones_in_twelve_dice_l3971_397100

def probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem two_ones_in_twelve_dice :
  let n : ℕ := 12
  let k : ℕ := 2
  let p : ℚ := 1/6
  probability_two_ones n k p = 66 * (1/36) * (9765625/60466176) :=
sorry

end two_ones_in_twelve_dice_l3971_397100


namespace coin_collection_problem_l3971_397103

/-- Represents the types of coins in the collection --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- The value of each coin in cents --/
def coinValue : Coin → ℕ
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A collection of coins --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins in the collection --/
def CoinCollection.totalCoins (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes + c.quarters

/-- The total value of the collection in cents --/
def CoinCollection.totalValue (c : CoinCollection) : ℕ :=
  c.pennies * coinValue Coin.Penny +
  c.nickels * coinValue Coin.Nickel +
  c.dimes * coinValue Coin.Dime +
  c.quarters * coinValue Coin.Quarter

theorem coin_collection_problem :
  ∀ c : CoinCollection,
    c.totalCoins = 10 ∧
    c.totalValue = 110 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    c.quarters ≥ 2
    →
    c.dimes = 5 :=
by sorry

end coin_collection_problem_l3971_397103


namespace max_sum_constrained_length_l3971_397170

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer -/
def length (n : ℕ) : ℕ := sorry

/-- The theorem states that given the conditions, the maximum value of x + 3y is 49156 -/
theorem max_sum_constrained_length (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h_length_sum : length x + length y ≤ 16) :
  x + 3 * y ≤ 49156 := by
  sorry

end max_sum_constrained_length_l3971_397170


namespace base5_123_equals_38_l3971_397178

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 5^2 + tens * 5^1 + ones * 5^0

/-- Theorem: The base-5 number 123₍₅₎ is equal to the decimal number 38 --/
theorem base5_123_equals_38 : base5ToDecimal 1 2 3 = 38 := by
  sorry

end base5_123_equals_38_l3971_397178


namespace exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l3971_397117

/-- The measure of an exterior angle formed by a regular pentagon and a regular octagon sharing a side -/
theorem exterior_angle_pentagon_octagon : ℝ :=
  let pentagon_interior_angle : ℝ := (180 * (5 - 2)) / 5
  let octagon_interior_angle : ℝ := (180 * (8 - 2)) / 8
  360 - (pentagon_interior_angle + octagon_interior_angle)

/-- The exterior angle formed by a regular pentagon and a regular octagon sharing a side is 117 degrees -/
theorem exterior_angle_pentagon_octagon_is_117 :
  exterior_angle_pentagon_octagon = 117 := by sorry

end exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l3971_397117


namespace fraction_meaningful_condition_l3971_397157

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by sorry

end fraction_meaningful_condition_l3971_397157


namespace line_point_k_value_l3971_397153

/-- Given three points on a line, calculate the value of k -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 7 = m * 3 + b ∧ k = m * 5 + b ∧ 15 = m * 11 + b) → k = 9 := by
  sorry

end line_point_k_value_l3971_397153


namespace least_non_lucky_multiple_of_seven_l3971_397135

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n % sum_of_digits n = 0

def is_multiple_of_seven (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

theorem least_non_lucky_multiple_of_seven :
  (14 % sum_of_digits 14 ≠ 0) ∧
  is_multiple_of_seven 14 ∧
  ∀ n : ℕ, 0 < n ∧ n < 14 ∧ is_multiple_of_seven n → is_lucky n :=
by sorry

end least_non_lucky_multiple_of_seven_l3971_397135


namespace faye_bought_30_songs_l3971_397172

/-- The number of songs Faye bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Proof that Faye bought 30 songs -/
theorem faye_bought_30_songs :
  total_songs 2 3 6 = 30 := by
  sorry

end faye_bought_30_songs_l3971_397172


namespace no_square_with_two_or_three_ones_l3971_397131

/-- Represents a number in base-10 using only 0 and 1 digits -/
def IsBaseOneZero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- Counts the number of ones in the base-10 representation of a number -/
def CountOnes (n : ℕ) : ℕ :=
  (n.digits 10).filter (· = 1) |>.length

/-- Main theorem: No square number exists with only 0 and 1 digits and exactly 2 or 3 ones -/
theorem no_square_with_two_or_three_ones :
  ¬ ∃ n : ℕ, IsBaseOneZero (n^2) ∧ (CountOnes (n^2) = 2 ∨ CountOnes (n^2) = 3) :=
sorry

end no_square_with_two_or_three_ones_l3971_397131


namespace not_divisible_by_67_l3971_397106

theorem not_divisible_by_67 (x y : ℕ) 
  (h1 : ¬ 67 ∣ x) 
  (h2 : ¬ 67 ∣ y) 
  (h3 : 67 ∣ (7 * x + 32 * y)) : 
  ¬ 67 ∣ (10 * x + 17 * y + 1) := by
sorry

end not_divisible_by_67_l3971_397106


namespace greatest_distance_C_D_l3971_397187

def C : Set ℂ := {z : ℂ | z^3 = 1}

def D : Set ℂ := {z : ℂ | z^3 - 27*z^2 + 27*z - 1 = 0}

theorem greatest_distance_C_D : 
  ∃ (c : ℂ) (d : ℂ), c ∈ C ∧ d ∈ D ∧ 
    ∀ (c' : ℂ) (d' : ℂ), c' ∈ C → d' ∈ D → 
      Complex.abs (c - d) ≥ Complex.abs (c' - d') ∧
      Complex.abs (c - d) = Real.sqrt (184.5 + 60 * Real.sqrt 3) :=
sorry

end greatest_distance_C_D_l3971_397187


namespace necessary_not_sufficient_condition_l3971_397101

theorem necessary_not_sufficient_condition : 
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) := by
  sorry

end necessary_not_sufficient_condition_l3971_397101


namespace eight_circle_times_three_l3971_397115

-- Define the new operation ⨳
def circle_times (a b : ℤ) : ℤ := 4 * a + 6 * b

-- The theorem to prove
theorem eight_circle_times_three : circle_times 8 3 = 50 := by
  sorry

end eight_circle_times_three_l3971_397115


namespace trapezoid_perimeter_is_34_l3971_397107

/-- A trapezoid with specific side lengths -/
structure Trapezoid :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)
  (h_AB_eq_CD : AB = CD)
  (h_AB : AB = 8)
  (h_CD : CD = 16)
  (h_BC_eq_DA : BC = DA)
  (h_BC : BC = 5)

/-- The perimeter of a trapezoid is the sum of its sides -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.BC + t.CD + t.DA

/-- Theorem: The perimeter of the specified trapezoid is 34 -/
theorem trapezoid_perimeter_is_34 (t : Trapezoid) : perimeter t = 34 := by
  sorry

end trapezoid_perimeter_is_34_l3971_397107


namespace intersection_A_complement_B_l3971_397140

def U : Finset Nat := {1,2,3,4,5,6}
def A : Finset Nat := {1,3,5}
def B : Finset Nat := {1,4}

theorem intersection_A_complement_B : A ∩ (U \ B) = {3,5} := by sorry

end intersection_A_complement_B_l3971_397140


namespace seating_arrangements_l3971_397127

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 10 people in a row. -/
def totalArrangements : ℕ := factorial 10

/-- The number of arrangements with 3 specific people in consecutive seats. -/
def threeConsecutive : ℕ := factorial 8 * factorial 3

/-- The number of arrangements with 2 specific people next to each other. -/
def twoTogether : ℕ := factorial 9 * factorial 2

/-- The number of arrangements satisfying both conditions. -/
def bothConditions : ℕ := factorial 7 * factorial 3 * factorial 2

/-- The number of valid seating arrangements. -/
def validArrangements : ℕ := totalArrangements - threeConsecutive - twoTogether + bothConditions

theorem seating_arrangements :
  validArrangements = 2685600 :=
sorry

end seating_arrangements_l3971_397127


namespace quadratic_equation_roots_l3971_397110

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*m*x + 3*m^2
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (m > 0 → (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ |x₁ - x₂| = 2) → m = 1) :=
by sorry

end quadratic_equation_roots_l3971_397110


namespace negation_forall_squared_gt_neg_one_negation_exists_squared_leq_nine_abs_gt_not_necessary_for_gt_m_lt_zero_iff_one_positive_one_negative_root_l3971_397116

-- Statement 1
theorem negation_forall_squared_gt_neg_one :
  (¬ ∀ x : ℝ, x^2 > -1) ↔ (∃ x : ℝ, x^2 ≤ -1) := by sorry

-- Statement 2
theorem negation_exists_squared_leq_nine :
  (¬ ∃ x : ℝ, x > -3 ∧ x^2 ≤ 9) ↔ (∀ x : ℝ, x > -3 → x^2 > 9) := by sorry

-- Statement 3
theorem abs_gt_not_necessary_for_gt :
  ∃ x y : ℝ, (abs x > abs y) ∧ (x ≤ y) := by sorry

-- Statement 4
theorem m_lt_zero_iff_one_positive_one_negative_root :
  ∀ m : ℝ, (m < 0) ↔ 
    (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0 ∧ 
      (∀ z : ℝ, z^2 - 2*z + m = 0 → z = x ∨ z = y)) := by sorry

end negation_forall_squared_gt_neg_one_negation_exists_squared_leq_nine_abs_gt_not_necessary_for_gt_m_lt_zero_iff_one_positive_one_negative_root_l3971_397116


namespace production_problem_l3971_397143

def initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) : ℕ :=
  ((n + 1) * new_average - today_production) / n

theorem production_problem :
  let n : ℕ := 3
  let today_production : ℕ := 90
  let new_average : ℕ := 75
  initial_average_production n today_production new_average = 70 := by
  sorry

end production_problem_l3971_397143


namespace friend_lunch_cost_l3971_397149

/-- Proves that given a total lunch cost of $17 and one person spending $3 more than the other,
    the person who spent more paid $10. -/
theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 17 → difference = 3 → friend_cost = total / 2 + difference / 2 → friend_cost = 10 := by
  sorry

end friend_lunch_cost_l3971_397149


namespace perpendicular_slope_is_five_thirds_l3971_397151

/-- The slope of a line perpendicular to the line containing points (3, 5) and (-2, 8) is 5/3 -/
theorem perpendicular_slope_is_five_thirds :
  let point1 : ℝ × ℝ := (3, 5)
  let point2 : ℝ × ℝ := (-2, 8)
  let slope_original := (point2.2 - point1.2) / (point2.1 - point1.1)
  let slope_perpendicular := -1 / slope_original
  slope_perpendicular = 5/3 := by
sorry

end perpendicular_slope_is_five_thirds_l3971_397151


namespace weight_of_new_person_l3971_397171

theorem weight_of_new_person (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 40 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 60 :=
by sorry

end weight_of_new_person_l3971_397171


namespace max_m_value_inequality_solution_l3971_397146

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for the maximum value of m
theorem max_m_value : 
  (∃ m : ℝ, ∀ x : ℝ, f x - m ≥ 0 ∧ ¬∃ m' : ℝ, m' > m ∧ ∀ x : ℝ, f x - m' ≥ 0) → 
  (∃ m : ℝ, m = 3 ∧ ∀ x : ℝ, f x - m ≥ 0 ∧ ¬∃ m' : ℝ, m' > m ∧ ∀ x : ℝ, f x - m' ≥ 0) :=
sorry

-- Theorem for the solution of the inequality
theorem inequality_solution :
  {x : ℝ | |x - 3| - 2*x ≤ 4} = {x : ℝ | x ≥ -1/3} :=
sorry

end max_m_value_inequality_solution_l3971_397146


namespace arithmetic_sequence_15th_term_l3971_397119

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_15th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_3 : a 3 = 4) 
  (h_9 : a 9 = 10) : 
  a 15 = 16 := by
sorry

end arithmetic_sequence_15th_term_l3971_397119


namespace trigonometric_equation_solution_l3971_397144

theorem trigonometric_equation_solution (x : ℝ) :
  (∃ (k : ℤ), x = 2 * π * k / 3) ∨ 
  (∃ (n : ℤ), x = π * (4 * n + 1) / 6) ↔ 
  (Real.cos (3 * x / 2) ≠ 0 ∧ 
   Real.sin ((3 * x - 7 * π) / 2) * Real.cos ((π - 3 * x) / 2) = 
   Real.arccos (3 * x / 2)) :=
by sorry

end trigonometric_equation_solution_l3971_397144


namespace intersecting_lines_k_value_l3971_397111

/-- Given two lines that intersect at a specific point, prove the value of k -/
theorem intersecting_lines_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + 5) →  -- Line p equation
  (∀ x y : ℝ, y = k * x + 3) →  -- Line q equation
  -7 = 3 * (-4) + 5 →           -- Point (-4, -7) satisfies line p equation
  -7 = k * (-4) + 3 →           -- Point (-4, -7) satisfies line q equation
  k = 2.5 := by
sorry

end intersecting_lines_k_value_l3971_397111


namespace inequality_proof_l3971_397192

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) :
  1 + a + b > 3 * Real.sqrt (a * b) := by
  sorry

end inequality_proof_l3971_397192


namespace two_digit_number_problem_l3971_397129

theorem two_digit_number_problem (x : ℕ) : x ≥ 10 ∧ x ≤ 99 → 500 + x = 9 * x - 12 → x = 64 := by
  sorry

end two_digit_number_problem_l3971_397129


namespace cookie_averages_l3971_397160

def brand_x_packages : List ℕ := [6, 8, 9, 11, 13]
def brand_y_packages : List ℕ := [14, 15, 18, 20]

theorem cookie_averages :
  let x_total := brand_x_packages.sum
  let y_total := brand_y_packages.sum
  let x_avg : ℚ := x_total / brand_x_packages.length
  let y_avg : ℚ := y_total / brand_y_packages.length
  x_avg = 47 / 5 ∧ y_avg = 67 / 4 := by
  sorry

end cookie_averages_l3971_397160


namespace box_width_calculation_l3971_397158

theorem box_width_calculation (length depth : ℕ) (total_cubes : ℕ) (width : ℕ) : 
  length = 49 → 
  depth = 14 → 
  total_cubes = 84 → 
  (∃ (cube_side : ℕ), 
    cube_side > 0 ∧ 
    length % cube_side = 0 ∧ 
    depth % cube_side = 0 ∧ 
    width % cube_side = 0 ∧
    (length / cube_side) * (depth / cube_side) * (width / cube_side) = total_cubes) →
  width = 42 := by
sorry

end box_width_calculation_l3971_397158


namespace rectangle_area_l3971_397148

/-- The area of a rectangle with vertices at (-3, 6), (1, 1), and (1, -6), 
    where (1, -6) is 7 units away from (1, 1), is equal to 7√41. -/
theorem rectangle_area : 
  let v1 : ℝ × ℝ := (-3, 6)
  let v2 : ℝ × ℝ := (1, 1)
  let v3 : ℝ × ℝ := (1, -6)
  let side1 := Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2)
  let side2 := |v2.2 - v3.2|
  side2 = 7 →
  side1 * side2 = 7 * Real.sqrt 41 :=
by sorry

end rectangle_area_l3971_397148


namespace debate_team_girls_l3971_397130

theorem debate_team_girls (boys : ℕ) (groups : ℕ) (group_size : ℕ) (total : ℕ) :
  boys = 28 →
  groups = 8 →
  group_size = 4 →
  total = groups * group_size →
  total - boys = 4 :=
by
  sorry

end debate_team_girls_l3971_397130


namespace part_one_part_two_part_three_l3971_397198

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1) * x + 1

/-- Theorem corresponding to part (1) of the problem -/
theorem part_one (a : ℝ) : 
  (∀ x, f a x ≥ 0) ↔ -3 ≤ a ∧ a ≤ 1 := by sorry

/-- Theorem corresponding to part (2) of the problem -/
theorem part_two (a b : ℝ) :
  (∃ b, ∀ x, f a x < 0 ↔ b < x ∧ x < 2) ↔ a = 3/2 ∧ b = 1/2 := by sorry

/-- Theorem corresponding to part (3) of the problem -/
theorem part_three (a : ℝ) :
  ((∀ x, f a x ≤ 0) ∧ (∀ x, 0 ≤ x ∧ x ≤ 1 → f a x > 0)) ↔ a < 1 := by sorry

end part_one_part_two_part_three_l3971_397198


namespace next_roll_for_average_three_l3971_397138

def rolls : List Nat := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

theorem next_roll_for_average_three :
  let n : Nat := rolls.length
  let sum : Nat := rolls.sum
  let target_average : Rat := 3
  let next_roll : Nat := 2
  (sum + next_roll : Rat) / (n + 1) = target_average := by sorry

end next_roll_for_average_three_l3971_397138


namespace factors_imply_value_l3971_397161

/-- The polynomial p(x) = 3x^3 - mx + n -/
def p (m n : ℝ) (x : ℝ) : ℝ := 3 * x^3 - m * x + n

theorem factors_imply_value (m n : ℝ) 
  (h1 : p m n 3 = 0)  -- x-3 is a factor
  (h2 : p m n (-4) = 0)  -- x+4 is a factor
  : |3*m - 2*n| = 33 := by
  sorry

end factors_imply_value_l3971_397161


namespace cookie_problem_l3971_397134

/-- Cookie problem statement -/
theorem cookie_problem (alyssa_cookies aiyanna_cookies brady_cookies : ℕ) 
  (h1 : alyssa_cookies = 1523)
  (h2 : aiyanna_cookies = 3720)
  (h3 : brady_cookies = 2265) :
  (aiyanna_cookies - alyssa_cookies = 2197) ∧ 
  (aiyanna_cookies - brady_cookies = 1455) ∧ 
  (brady_cookies - alyssa_cookies = 742) := by
  sorry

end cookie_problem_l3971_397134


namespace stations_between_hyderabad_and_bangalore_l3971_397150

theorem stations_between_hyderabad_and_bangalore : 
  ∃ (n : ℕ), n > 2 ∧ (n * (n - 1)) / 2 = 306 ∧ n - 2 = 25 := by
  sorry

end stations_between_hyderabad_and_bangalore_l3971_397150


namespace alpha_value_l3971_397185

theorem alpha_value (α β : ℂ) 
  (h1 : (α + 2*β).re > 0)
  (h2 : (Complex.I * (α - 3*β)).re > 0)
  (h3 : β = 2 + 3*Complex.I) : 
  α = 6 - 6*Complex.I := by sorry

end alpha_value_l3971_397185


namespace inequality_holds_l3971_397183

/-- A quadratic function with the given symmetry property -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The symmetry property of f -/
axiom symmetry_at_3 (b c : ℝ) : ∀ t : ℝ, f b c (3 + t) = f b c (3 - t)

/-- The main theorem stating the inequality -/
theorem inequality_holds (b c : ℝ) : f b c 3 < f b c 1 ∧ f b c 1 < f b c 6 := by
  sorry

end inequality_holds_l3971_397183


namespace solution_sets_equivalence_l3971_397167

open Set

-- Define the solution set of the first inequality
def solution_set_1 : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 3}

-- Define the coefficients a, b, c based on the given conditions
def a : ℝ := -1  -- Assume a = -1 for simplicity, since we know a < 0
def b : ℝ := -2 * a
def c : ℝ := -3 * a

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := {x : ℝ | -1/3 < x ∧ x < 1}

-- Theorem statement
theorem solution_sets_equivalence : 
  (∀ x : ℝ, x ∈ solution_set_1 ↔ a * x^2 + b * x + c ≤ 0) →
  (∀ x : ℝ, x ∈ solution_set_2 ↔ c * x^2 - b * x + a < 0) := by
  sorry

end solution_sets_equivalence_l3971_397167


namespace sin_transformation_l3971_397162

theorem sin_transformation (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.sin (x - π / 3) := by
  sorry

end sin_transformation_l3971_397162


namespace shopping_mall_entrances_exits_l3971_397139

theorem shopping_mall_entrances_exits (n : ℕ) (h : n = 4) :
  (n * (n - 1) : ℕ) = 12 := by
  sorry

end shopping_mall_entrances_exits_l3971_397139


namespace fraction_sum_zero_l3971_397199

theorem fraction_sum_zero (a b : ℝ) (h : a ≠ b) : 
  1 / (a - b) + 1 / (b - a) = 0 := by
sorry

end fraction_sum_zero_l3971_397199


namespace volleyball_lineup_combinations_l3971_397193

def team_size : ℕ := 12
def starting_lineup_size : ℕ := 6
def non_libero_positions : ℕ := 5

theorem volleyball_lineup_combinations :
  (team_size) * (Nat.choose (team_size - 1) non_libero_positions) = 5544 :=
sorry

end volleyball_lineup_combinations_l3971_397193


namespace expression_evaluation_l3971_397128

theorem expression_evaluation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

#eval 14 + 19 + 29

end expression_evaluation_l3971_397128


namespace garage_sale_pricing_l3971_397155

theorem garage_sale_pricing (total_items : ℕ) (n : ℕ) : 
  total_items = 34 →
  n = (total_items - 20) →
  n = 14 :=
by
  sorry

end garage_sale_pricing_l3971_397155


namespace coordinates_of_M_l3971_397156

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Check if a point is on the angle bisector of the first and third quadrants -/
def isOnAngleBisector (p : Point) : Prop := p.x = p.y

/-- Given point M with coordinates (2-m, 1+2m) -/
def M (m : ℝ) : Point := ⟨2 - m, 1 + 2*m⟩

theorem coordinates_of_M (m : ℝ) :
  (distanceToYAxis (M m) = 3 → (M m = ⟨3, -1⟩ ∨ M m = ⟨-3, 11⟩)) ∧
  (isOnAngleBisector (M m) → M m = ⟨5/3, 5/3⟩) := by
  sorry

end coordinates_of_M_l3971_397156


namespace sum_of_coordinates_reflection_l3971_397175

/-- Given a point C with coordinates (3, y) and its reflection D over the x-axis,
    the sum of all coordinates of C and D is 6. -/
theorem sum_of_coordinates_reflection (y : ℝ) : 
  let C : ℝ × ℝ := (3, y)
  let D : ℝ × ℝ := (3, -y)  -- reflection of C over x-axis
  C.1 + C.2 + D.1 + D.2 = 6 :=
by sorry

end sum_of_coordinates_reflection_l3971_397175


namespace find_n_l3971_397194

theorem find_n : ∃ n : ℤ, 3^4 - 13 = 4^3 + n ∧ n = 4 := by
  sorry

end find_n_l3971_397194


namespace tea_house_payment_l3971_397168

theorem tea_house_payment (t k b : ℕ+) (h : 11 ∣ (3 * t + 4 * k + 5 * b)) :
  11 ∣ (9 * t + k + 4 * b) := by
  sorry

end tea_house_payment_l3971_397168


namespace angle_and_complement_differ_by_20_l3971_397114

theorem angle_and_complement_differ_by_20 (α : ℝ) : 
  α - (90 - α) = 20 → α = 55 := by
  sorry

end angle_and_complement_differ_by_20_l3971_397114


namespace solution_theorem_l3971_397188

-- Define the function f(x) = x^2023 + x
def f (x : ℝ) := x^2023 + x

-- State the theorem
theorem solution_theorem (x y : ℝ) :
  (3*x + y)^2023 + x^2023 + 4*x + y = 0 → 4*x + y = 0 := by
  sorry

end solution_theorem_l3971_397188


namespace fold_cut_unfold_result_l3971_397142

/-- Represents a square sheet of paper with two sides --/
structure Sheet :=
  (side_length : ℝ)
  (white_side : Bool)
  (gray_side : Bool)

/-- Represents a fold on the sheet --/
inductive Fold
  | Vertical
  | Horizontal

/-- Represents a cut on the folded sheet --/
structure Cut :=
  (size : ℝ)

/-- The result of unfolding the sheet after folding and cutting --/
structure UnfoldedResult :=
  (num_cutouts : ℕ)
  (symmetric : Bool)

/-- Function to fold the sheet --/
def fold_sheet (s : Sheet) (f : Fold) : Sheet :=
  sorry

/-- Function to cut the folded sheet --/
def cut_sheet (s : Sheet) (c : Cut) : Sheet :=
  sorry

/-- Function to unfold the sheet --/
def unfold_sheet (s : Sheet) : UnfoldedResult :=
  sorry

/-- Theorem stating the result of folding twice, cutting, and unfolding --/
theorem fold_cut_unfold_result (s : Sheet) (f1 f2 : Fold) (c : Cut) :
  let folded := fold_sheet (fold_sheet s f1) f2
  let cut := cut_sheet folded c
  let result := unfold_sheet cut
  result.num_cutouts = 4 ∧ result.symmetric = true :=
sorry

end fold_cut_unfold_result_l3971_397142


namespace cricket_run_rate_l3971_397154

/-- Calculates the required run rate for the remaining overs in a cricket game. -/
def required_run_rate (total_overs : ℕ) (initial_overs : ℕ) (initial_run_rate : ℚ) (target_runs : ℕ) : ℚ :=
  let remaining_overs := total_overs - initial_overs
  let initial_runs := initial_run_rate * initial_overs
  let remaining_runs := target_runs - initial_runs
  remaining_runs / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario. -/
theorem cricket_run_rate : required_run_rate 60 10 (32/10) 282 = 5 := by
  sorry


end cricket_run_rate_l3971_397154


namespace modular_arithmetic_problem_l3971_397182

theorem modular_arithmetic_problem (m : ℕ) : 
  13^5 % 7 = m → 0 ≤ m → m < 7 → m = 0 := by
  sorry

end modular_arithmetic_problem_l3971_397182


namespace exponent_division_l3971_397190

theorem exponent_division (a : ℝ) (m n : ℕ) (h : m > n) :
  a^m / a^n = a^(m - n) := by sorry

end exponent_division_l3971_397190


namespace expression_factorization_l3971_397169

variable (x : ℝ)

theorem expression_factorization :
  (12 * x^3 + 27 * x^2 + 90 * x - 9) - (-3 * x^3 + 9 * x^2 - 15 * x - 9) =
  3 * x * (5 * x^2 + 6 * x + 35) := by sorry

end expression_factorization_l3971_397169


namespace bobby_adult_jumps_per_second_l3971_397104

/-- Bobby's jumping ability as a child and adult -/
def bobby_jumping (child_jumps_per_minute : ℕ) (additional_jumps_per_minute : ℕ) : Prop :=
  let adult_jumps_per_minute := child_jumps_per_minute + additional_jumps_per_minute
  let adult_jumps_per_second := adult_jumps_per_minute / 60
  adult_jumps_per_second = 1

/-- Theorem: Bobby can jump 1 time per second as an adult -/
theorem bobby_adult_jumps_per_second :
  bobby_jumping 30 30 := by
  sorry

end bobby_adult_jumps_per_second_l3971_397104


namespace population_growth_rate_l3971_397177

/-- Given that a population increases by 90 persons in 30 minutes,
    prove that it takes 20 seconds for one person to be added. -/
theorem population_growth_rate (increase : ℕ) (time_minutes : ℕ) (time_seconds : ℕ) :
  increase = 90 →
  time_minutes = 30 →
  time_seconds = time_minutes * 60 →
  time_seconds / increase = 20 := by
  sorry

end population_growth_rate_l3971_397177


namespace total_production_is_29621_l3971_397118

/-- Represents the production numbers for a specific region -/
structure RegionProduction where
  sedans : Nat
  suvs : Nat
  pickups : Nat

/-- Calculates the total production for a region -/
def total_region_production (r : RegionProduction) : Nat :=
  r.sedans + r.suvs + r.pickups

/-- Represents the production data for all regions -/
structure GlobalProduction where
  north_america : RegionProduction
  europe : RegionProduction
  asia : RegionProduction
  south_america : RegionProduction

/-- Calculates the total global production -/
def total_global_production (g : GlobalProduction) : Nat :=
  total_region_production g.north_america +
  total_region_production g.europe +
  total_region_production g.asia +
  total_region_production g.south_america

/-- The production data for the 5-month period -/
def production_data : GlobalProduction := {
  north_america := { sedans := 3884, suvs := 2943, pickups := 1568 }
  europe := { sedans := 2871, suvs := 2145, pickups := 643 }
  asia := { sedans := 5273, suvs := 3881, pickups := 2338 }
  south_america := { sedans := 1945, suvs := 1365, pickups := 765 }
}

/-- Theorem stating that the total global production equals 29621 -/
theorem total_production_is_29621 :
  total_global_production production_data = 29621 := by
  sorry

end total_production_is_29621_l3971_397118


namespace alcohol_distribution_correct_l3971_397141

/-- Represents a container with an alcohol solution -/
structure Container where
  volume : ℝ
  concentration : ℝ

/-- Calculates the amount of pure alcohol needed to achieve the desired concentration -/
def pureAlcoholNeeded (c : Container) (desiredConcentration : ℝ) : ℝ :=
  c.volume * desiredConcentration - c.volume * c.concentration

/-- Theorem: The calculated amounts of pure alcohol will result in 50% solutions -/
theorem alcohol_distribution_correct 
  (containerA containerB containerC : Container)
  (pureAlcoholA pureAlcoholB pureAlcoholC : ℝ)
  (h1 : containerA = { volume := 8, concentration := 0.25 })
  (h2 : containerB = { volume := 10, concentration := 0.40 })
  (h3 : containerC = { volume := 6, concentration := 0.30 })
  (h4 : pureAlcoholA = pureAlcoholNeeded containerA 0.5)
  (h5 : pureAlcoholB = pureAlcoholNeeded containerB 0.5)
  (h6 : pureAlcoholC = pureAlcoholNeeded containerC 0.5)
  (h7 : pureAlcoholA + pureAlcoholB + pureAlcoholC ≤ 12) :
  pureAlcoholA = 2 ∧ 
  pureAlcoholB = 1 ∧ 
  pureAlcoholC = 1.2 ∧
  (containerA.volume * containerA.concentration + pureAlcoholA) / (containerA.volume + pureAlcoholA) = 0.5 ∧
  (containerB.volume * containerB.concentration + pureAlcoholB) / (containerB.volume + pureAlcoholB) = 0.5 ∧
  (containerC.volume * containerC.concentration + pureAlcoholC) / (containerC.volume + pureAlcoholC) = 0.5 := by
  sorry

end alcohol_distribution_correct_l3971_397141


namespace max_value_of_g_l3971_397108

def S : Set Int := {-3, -2, 1, 2, 3, 4}

def g (a b : Int) : ℚ := -((a - b)^2 : ℚ) / 4

theorem max_value_of_g :
  ∃ (max : ℚ), max = -1/4 ∧
  ∀ (a b : Int), a ∈ S → b ∈ S → a ≠ b → g a b ≤ max ∧
  ∃ (a₀ b₀ : Int), a₀ ∈ S ∧ b₀ ∈ S ∧ a₀ ≠ b₀ ∧ g a₀ b₀ = max :=
sorry

end max_value_of_g_l3971_397108


namespace road_repair_hours_proof_l3971_397124

/-- The number of hours the first group works per day to repair a road -/
def hours_per_day : ℕ := 5

/-- The number of people in the first group -/
def people_group1 : ℕ := 39

/-- The number of days the first group works -/
def days_group1 : ℕ := 12

/-- The number of people in the second group -/
def people_group2 : ℕ := 30

/-- The number of days the second group works -/
def days_group2 : ℕ := 13

/-- The number of hours per day the second group works -/
def hours_group2 : ℕ := 6

theorem road_repair_hours_proof :
  people_group1 * days_group1 * hours_per_day = people_group2 * days_group2 * hours_group2 :=
by sorry

end road_repair_hours_proof_l3971_397124


namespace collinear_vector_combinations_l3971_397136

/-- Given two non-zero vectors in a real vector space that are not collinear,
    if a linear combination of these vectors with scalar k is collinear with
    another linear combination of the same vectors where k's role is swapped,
    then k must be either 1 or -1. -/
theorem collinear_vector_combinations (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (e₁ e₂ : V) (k : ℝ) 
  (h_nonzero₁ : e₁ ≠ 0)
  (h_nonzero₂ : e₂ ≠ 0)
  (h_not_collinear : ¬ ∃ (c : ℝ), e₁ = c • e₂)
  (h_collinear : ∃ (t : ℝ), k • e₁ + e₂ = t • (e₁ + k • e₂)) :
  k = 1 ∨ k = -1 :=
sorry

end collinear_vector_combinations_l3971_397136


namespace scheme_probability_l3971_397165

theorem scheme_probability (p_both : ℝ) (h1 : p_both = 0.3) :
  1 - (1 - p_both) * (1 - p_both) = 0.51 := by
sorry

end scheme_probability_l3971_397165
