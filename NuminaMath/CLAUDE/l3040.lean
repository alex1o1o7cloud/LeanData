import Mathlib

namespace triangle_problem_l3040_304068

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- The dot product of two vectors -/
def dotProduct (v w : ℝ × ℝ) : ℝ := sorry

theorem triangle_problem (t : Triangle) 
  (h_area : area t = 30)
  (h_cos : Real.cos t.A = 12/13) : 
  ∃ (ab ac : ℝ × ℝ), 
    dotProduct ab ac = 144 ∧ 
    (t.c - t.b = 1 → t.a = 5) := by
  sorry

end triangle_problem_l3040_304068


namespace at_least_one_equation_has_two_distinct_roots_l3040_304037

theorem at_least_one_equation_has_two_distinct_roots
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  ¬(4*b^2 - 4*a*c ≤ 0 ∧ 4*c^2 - 4*a*b ≤ 0 ∧ 4*a^2 - 4*b*c ≤ 0) :=
by sorry

end at_least_one_equation_has_two_distinct_roots_l3040_304037


namespace equilateral_triangle_area_sum_l3040_304067

theorem equilateral_triangle_area_sum : 
  let triangle1_side : ℝ := 2
  let triangle2_side : ℝ := 3
  let new_triangle_side : ℝ := Real.sqrt 13
  let area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2
  area new_triangle_side = area triangle1_side + area triangle2_side :=
by sorry

end equilateral_triangle_area_sum_l3040_304067


namespace pig_purchase_equation_l3040_304012

/-- Represents a group purchase of pigs -/
structure PigPurchase where
  numPeople : ℕ
  excessAmount : ℕ
  exactAmount : ℕ

/-- The equation for the pig purchase problem is correct -/
theorem pig_purchase_equation (p : PigPurchase) 
  (h1 : p.numPeople * p.excessAmount - p.numPeople * p.exactAmount = p.excessAmount) 
  (h2 : p.excessAmount = 100) 
  (h3 : p.exactAmount = 90) : 
  100 * p.numPeople - 90 * p.numPeople = 100 := by
  sorry

end pig_purchase_equation_l3040_304012


namespace smallest_y_with_remainders_l3040_304024

theorem smallest_y_with_remainders : ∃! y : ℕ, 
  y > 0 ∧ 
  y % 6 = 5 ∧ 
  y % 7 = 6 ∧ 
  y % 8 = 7 ∧
  ∀ z : ℕ, z > 0 ∧ z % 6 = 5 ∧ z % 7 = 6 ∧ z % 8 = 7 → y ≤ z :=
by
  -- The proof goes here
  sorry

end smallest_y_with_remainders_l3040_304024


namespace survey_respondents_l3040_304007

/-- The number of people who preferred brand X -/
def X : ℕ := 60

/-- The number of people who preferred brand Y -/
def Y : ℕ := X / 3

/-- The number of people who preferred brand Z -/
def Z : ℕ := X * 3 / 2

/-- The total number of respondents to the survey -/
def total_respondents : ℕ := X + Y + Z

/-- Theorem stating that the total number of respondents is 170 -/
theorem survey_respondents : total_respondents = 170 := by
  sorry

end survey_respondents_l3040_304007


namespace negative_64_to_four_thirds_equals_256_l3040_304088

theorem negative_64_to_four_thirds_equals_256 : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end negative_64_to_four_thirds_equals_256_l3040_304088


namespace line_passes_through_fixed_point_l3040_304074

/-- The line equation passing through a fixed point for all values of parameter a -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x - y + 2 * a + 1 = 0

/-- Theorem stating that the line passes through the point (-2, 3) for all values of a -/
theorem line_passes_through_fixed_point :
  ∀ a : ℝ, line_equation a (-2) 3 := by
sorry

end line_passes_through_fixed_point_l3040_304074


namespace lcm_hcf_problem_l3040_304080

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 210 →
  B = 330 := by
sorry

end lcm_hcf_problem_l3040_304080


namespace bd_length_l3040_304046

-- Define the triangle ABC
structure Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :=
  (A B C : α)

-- Define the properties of the triangle
def IsoscelesTriangle {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : Prop :=
  ‖t.A - t.C‖ = ‖t.B - t.C‖

-- Define point D on AB
def PointOnLine {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (A B D : α) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • A + t • B

-- Main theorem
theorem bd_length {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) (D : α) :
  IsoscelesTriangle t →
  PointOnLine t.A t.B D →
  ‖t.A - t.C‖ = 10 →
  ‖t.A - D‖ = 12 →
  ‖t.C - D‖ = 4 →
  ‖t.B - D‖ = 7 := by
  sorry


end bd_length_l3040_304046


namespace no_odd_three_digit_div_five_without_five_l3040_304004

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def divisible_by_five (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

def does_not_contain_five (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 5

theorem no_odd_three_digit_div_five_without_five :
  {n : ℕ | is_odd n ∧ is_three_digit n ∧ divisible_by_five n ∧ does_not_contain_five n} = ∅ :=
sorry

end no_odd_three_digit_div_five_without_five_l3040_304004


namespace next_year_day_l3040_304039

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  numDays : Nat
  firstDay : DayOfWeek
  numSaturdays : Nat

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

theorem next_year_day (y : Year) (h1 : y.numDays = 366) (h2 : y.numSaturdays = 53) :
  nextDay y.firstDay = DayOfWeek.Monday := by
  sorry

end next_year_day_l3040_304039


namespace smallest_class_size_class_with_25_students_exists_l3040_304041

/-- Represents a class of students who took a history test -/
structure HistoryClass where
  /-- The total number of students in the class -/
  num_students : ℕ
  /-- The total score of all students -/
  total_score : ℕ
  /-- The number of students who scored 120 points -/
  perfect_scores : ℕ
  /-- The number of students who scored 115 points -/
  near_perfect_scores : ℕ

/-- The properties of the history class based on the given problem -/
def valid_history_class (c : HistoryClass) : Prop :=
  c.perfect_scores = 8 ∧
  c.near_perfect_scores = 3 ∧
  c.total_score = c.num_students * 92 ∧
  c.total_score ≥ c.perfect_scores * 120 + c.near_perfect_scores * 115 + (c.num_students - c.perfect_scores - c.near_perfect_scores) * 70

/-- The theorem stating that the smallest possible number of students in the class is 25 -/
theorem smallest_class_size (c : HistoryClass) (h : valid_history_class c) : c.num_students ≥ 25 := by
  sorry

/-- The theorem stating that a class with 25 students satisfying all conditions exists -/
theorem class_with_25_students_exists : ∃ c : HistoryClass, valid_history_class c ∧ c.num_students = 25 := by
  sorry

end smallest_class_size_class_with_25_students_exists_l3040_304041


namespace farm_animals_l3040_304063

/-- The number of animals in a farm with ducks and dogs -/
def total_animals (num_ducks : ℕ) (total_legs : ℕ) : ℕ :=
  num_ducks + (total_legs - 2 * num_ducks) / 4

/-- Theorem: Given the conditions, there are 11 animals in total -/
theorem farm_animals : total_animals 6 32 = 11 := by
  sorry

end farm_animals_l3040_304063


namespace storks_on_fence_l3040_304006

theorem storks_on_fence (initial_birds : ℕ) (new_birds : ℕ) (total_birds : ℕ) :
  initial_birds = 4 →
  new_birds = 6 →
  total_birds = 10 →
  initial_birds + new_birds = total_birds →
  ∃ storks : ℕ, storks = 0 :=
by
  sorry

end storks_on_fence_l3040_304006


namespace sum_of_integers_l3040_304008

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 130)
  (h2 : x * y = 45) : 
  ∃ (ε : ℝ), abs ((x : ℝ) + y - 15) < ε ∧ ε > 0 := by
sorry

end sum_of_integers_l3040_304008


namespace power_of_negative_square_l3040_304044

theorem power_of_negative_square (x : ℝ) : (-2 * x^2)^3 = -8 * x^6 := by
  sorry

end power_of_negative_square_l3040_304044


namespace count_non_divisors_is_33_l3040_304030

/-- g(n) is the product of the proper positive integer divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- The number of integers n between 2 and 100 (inclusive) that do not divide g(n) -/
def count_non_divisors : ℕ := sorry

/-- Theorem stating that the count of non-divisors is 33 -/
theorem count_non_divisors_is_33 : count_non_divisors = 33 := by sorry

end count_non_divisors_is_33_l3040_304030


namespace f_of_f_one_eq_two_l3040_304062

def f (x : ℝ) : ℝ := 4 * x^2 - 6 * x + 2

theorem f_of_f_one_eq_two : f (f 1) = 2 := by sorry

end f_of_f_one_eq_two_l3040_304062


namespace proposition_equivalence_l3040_304091

theorem proposition_equivalence (p q : Prop) : (p ∨ q) ↔ ¬(¬p ∧ ¬q) := by
  sorry

end proposition_equivalence_l3040_304091


namespace tank_solution_volume_l3040_304031

theorem tank_solution_volume 
  (V : ℝ) 
  (h1 : V > 0) 
  (h2 : 0.05 * V / (V - 5500) = 1 / 9) : 
  V = 10000 := by
sorry

end tank_solution_volume_l3040_304031


namespace prob_one_one_ten_dice_l3040_304059

/-- The probability of rolling exactly one 1 out of 10 standard 6-sided dice -/
def prob_one_one (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  ↑(n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of rolling exactly one 1 out of 10 standard 6-sided dice
    is equal to (10 * 5^9) / 6^10 -/
theorem prob_one_one_ten_dice :
  prob_one_one 10 1 (1/6) = (10 * 5^9) / 6^10 := by
  sorry

end prob_one_one_ten_dice_l3040_304059


namespace area_between_circle_and_squares_l3040_304098

theorem area_between_circle_and_squares :
  let outer_square_side : ℝ := 2
  let circle_radius : ℝ := 1/2
  let inner_square_side : ℝ := 1.8
  let outer_square_area : ℝ := outer_square_side^2
  let inner_square_area : ℝ := inner_square_side^2
  let circle_area : ℝ := π * circle_radius^2
  let area_between : ℝ := outer_square_area - inner_square_area - (outer_square_area - circle_area)
  area_between = 0.76 := by sorry

end area_between_circle_and_squares_l3040_304098


namespace avg_calculation_l3040_304000

/-- Calculates the average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Calculates the average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- The main theorem to prove -/
theorem avg_calculation : avg3 (avg3 2 2 0) (avg2 1 2) 1 = 23 / 18 := by
  sorry

end avg_calculation_l3040_304000


namespace difference_of_squares_153_147_l3040_304033

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end difference_of_squares_153_147_l3040_304033


namespace factorial_division_l3040_304032

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 5 = 30240 := by
  sorry

end factorial_division_l3040_304032


namespace mixed_fraction_division_subtraction_l3040_304027

theorem mixed_fraction_division_subtraction :
  (1 + 5/6) / (2 + 3/4) - 1/2 = 1/6 := by
  sorry

end mixed_fraction_division_subtraction_l3040_304027


namespace june1st_is_tuesday_l3040_304009

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific year -/
structure Year where
  febHasFiveSundays : Bool
  febHas29Days : Bool
  feb1stIsSunday : Bool

/-- Function to calculate the day of the week for June 1st -/
def june1stDayOfWeek (y : Year) : DayOfWeek :=
  sorry

/-- Theorem stating that June 1st is a Tuesday in the specified year -/
theorem june1st_is_tuesday (y : Year) 
  (h1 : y.febHasFiveSundays = true) 
  (h2 : y.febHas29Days = true) 
  (h3 : y.feb1stIsSunday = true) : 
  june1stDayOfWeek y = DayOfWeek.Tuesday :=
  sorry

end june1st_is_tuesday_l3040_304009


namespace distance_to_incenter_l3040_304066

/-- An isosceles right triangle with side length 6√2 -/
structure IsoscelesRightTriangle where
  /-- The length of the equal sides -/
  side_length : ℝ
  /-- The side length is 6√2 -/
  side_length_eq : side_length = 6 * Real.sqrt 2

/-- The incenter of a triangle -/
def incenter (t : IsoscelesRightTriangle) : ℝ × ℝ := sorry

/-- The distance from a vertex to a point -/
def distance_to_point (v : ℝ × ℝ) (p : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the distance from the right angle vertex to the incenter -/
theorem distance_to_incenter (t : IsoscelesRightTriangle) : 
  distance_to_point (0, t.side_length) (incenter t) = 6 - 3 * Real.sqrt 2 := by sorry

end distance_to_incenter_l3040_304066


namespace greatest_integer_fraction_l3040_304065

theorem greatest_integer_fraction (x : ℤ) :
  x ≠ 3 →
  (∃ y : ℤ, (x^2 + 2*x + 5) = (x - 3) * y) →
  x ≤ 23 :=
by sorry

end greatest_integer_fraction_l3040_304065


namespace rectangle_area_solution_l3040_304085

/-- A rectangle with dimensions (2x - 3) by (3x + 4) has an area of 14x - 12. -/
theorem rectangle_area_solution (x : ℝ) : 
  (2 * x - 3 > 0) → 
  (3 * x + 4 > 0) → 
  (2 * x - 3) * (3 * x + 4) = 14 * x - 12 → 
  x = 4 :=
by sorry

end rectangle_area_solution_l3040_304085


namespace boys_candies_order_independent_l3040_304014

/-- Represents a child's gender -/
inductive Gender
| Boy
| Girl

/-- Represents a child with their gender -/
structure Child where
  gender : Gender

/-- Represents the state of the candy distribution process -/
structure CandyState where
  remaining_candies : ℕ
  remaining_children : List Child

/-- Represents the result of a candy distribution process -/
structure DistributionResult where
  boys_candies : ℕ
  girls_candies : ℕ

/-- Function to distribute candies according to the rules -/
def distributeCandies (initial_state : CandyState) : DistributionResult :=
  sorry

/-- Theorem stating that the number of candies taken by boys is independent of the order -/
theorem boys_candies_order_independent
  (children : List Child)
  (perm : List Child)
  (h : perm.Perm children) :
  (distributeCandies { remaining_candies := 2021, remaining_children := children }).boys_candies =
  (distributeCandies { remaining_candies := 2021, remaining_children := perm }).boys_candies :=
  sorry

end boys_candies_order_independent_l3040_304014


namespace apples_picked_l3040_304071

theorem apples_picked (initial_apples new_apples final_apples : ℕ) 
  (h1 : initial_apples = 11)
  (h2 : new_apples = 2)
  (h3 : final_apples = 6) :
  initial_apples - (initial_apples - new_apples - final_apples) = 7 := by
  sorry

end apples_picked_l3040_304071


namespace sum_of_special_numbers_l3040_304099

theorem sum_of_special_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 50 * (x + y)) (h2 : x * y = 75 * (x - y)) :
  x + y = 360 := by
sorry

end sum_of_special_numbers_l3040_304099


namespace max_nine_letter_palindromes_l3040_304036

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the palindromes we're considering -/
def palindrome_length : ℕ := 9

/-- A palindrome is a word that reads the same forward and backward -/
def is_palindrome (word : List Char) : Prop :=
  word = word.reverse

/-- The maximum number of 9-letter palindromes using the English alphabet -/
theorem max_nine_letter_palindromes :
  (alphabet_size ^ ((palindrome_length - 1) / 2 + 1) : ℕ) = 11881376 :=
sorry

end max_nine_letter_palindromes_l3040_304036


namespace system_solution_l3040_304038

theorem system_solution (x y k : ℝ) : 
  x - y = k - 3 →
  3 * x + 5 * y = 2 * k + 8 →
  x + y = 2 →
  k = 1 := by
sorry

end system_solution_l3040_304038


namespace oil_measurement_l3040_304082

theorem oil_measurement (initial_oil : ℚ) (added_oil : ℚ) (total_oil : ℚ) :
  initial_oil = 17/100 →
  added_oil = 67/100 →
  total_oil = initial_oil + added_oil →
  total_oil = 84/100 := by
sorry

end oil_measurement_l3040_304082


namespace smallest_absolute_value_l3040_304026

theorem smallest_absolute_value : ∀ x : ℝ, |0| ≤ |x| := by
  sorry

end smallest_absolute_value_l3040_304026


namespace even_numbers_average_21_l3040_304013

/-- The sum of the first n even numbers -/
def sumFirstEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The average of the first n even numbers -/
def averageFirstEvenNumbers (n : ℕ) : ℚ := (sumFirstEvenNumbers n : ℚ) / n

theorem even_numbers_average_21 :
  ∃ n : ℕ, n > 0 ∧ averageFirstEvenNumbers n = 21 :=
sorry

end even_numbers_average_21_l3040_304013


namespace german_team_goals_l3040_304018

def journalist1 (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2 (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3 (x : ℕ) : Prop := x % 2 = 1

def twoCorrect (x : ℕ) : Prop :=
  (journalist1 x ∧ journalist2 x ∧ ¬journalist3 x) ∨
  (journalist1 x ∧ ¬journalist2 x ∧ journalist3 x) ∨
  (¬journalist1 x ∧ journalist2 x ∧ journalist3 x)

theorem german_team_goals :
  ∀ x : ℕ, twoCorrect x ↔ x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
by sorry

end german_team_goals_l3040_304018


namespace cost_increase_operation_l3040_304021

/-- Represents the cost function -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

/-- Theorem: If the new cost after an operation on b is 1600% of the original cost,
    then the operation performed on b is multiplication by 2 -/
theorem cost_increase_operation (t : ℝ) (b₀ b₁ : ℝ) (h : t > 0) :
  cost t b₁ = 16 * cost t b₀ → b₁ = 2 * b₀ :=
by sorry

end cost_increase_operation_l3040_304021


namespace incorrect_option_l3040_304015

/-- Represents the area of a rectangle with length 8 and width a -/
def option_a (a : ℝ) : ℝ := 8 * a

/-- Represents the selling price after a discount of 8% on an item priced a -/
def option_b (a : ℝ) : ℝ := 0.92 * a

/-- Represents the cost of 8 notebooks priced a each -/
def option_c (a : ℝ) : ℝ := 8 * a

/-- Represents the distance traveled at speed a for 8 hours -/
def option_d (a : ℝ) : ℝ := 8 * a

theorem incorrect_option (a : ℝ) : 
  option_a a = 8 * a ∧ 
  option_b a ≠ 8 * a ∧ 
  option_c a = 8 * a ∧ 
  option_d a = 8 * a :=
sorry

end incorrect_option_l3040_304015


namespace parabola_line_intersection_slopes_l3040_304056

/-- Given a parabola y^2 = 2px and a line intersecting it at points A and B, 
    if the slope of OA is 2 and the slope of AB is 6, then the slope of OB is -3. -/
theorem parabola_line_intersection_slopes (p : ℝ) (y₁ y₂ : ℝ) : 
  let A := (y₁^2 / (2*p), y₁)
  let B := (y₂^2 / (2*p), y₂)
  let k_OA := y₁ / (y₁^2 / (2*p))
  let k_AB := (y₂ - y₁) / (y₂^2 / (2*p) - y₁^2 / (2*p))
  let k_OB := y₂ / (y₂^2 / (2*p))
  k_OA = 2 ∧ k_AB = 6 → k_OB = -3 :=
by sorry

end parabola_line_intersection_slopes_l3040_304056


namespace debby_yoyo_tickets_debby_yoyo_tickets_proof_l3040_304094

/-- Theorem: Debby's yoyo ticket expenditure --/
theorem debby_yoyo_tickets : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun hat_tickets stuffed_animal_tickets total_tickets yoyo_tickets =>
    hat_tickets = 2 ∧ 
    stuffed_animal_tickets = 10 ∧ 
    total_tickets = 14 ∧ 
    yoyo_tickets + hat_tickets + stuffed_animal_tickets = total_tickets →
    yoyo_tickets = 2

/-- Proof of the theorem --/
theorem debby_yoyo_tickets_proof : 
  debby_yoyo_tickets 2 10 14 2 := by
  sorry

end debby_yoyo_tickets_debby_yoyo_tickets_proof_l3040_304094


namespace combination_sum_equals_55_l3040_304019

-- Define the combination function
def combination (n r : ℕ) : ℕ :=
  if r ≤ n then
    Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))
  else
    0

-- State the theorem
theorem combination_sum_equals_55 :
  combination 10 9 + combination 10 8 = 55 :=
sorry

end combination_sum_equals_55_l3040_304019


namespace complex_expression_equality_l3040_304042

/-- Given complex numbers x and y, prove that 3x + 4y = 17 + 2i -/
theorem complex_expression_equality (x y : ℂ) (hx : x = 3 + 2*I) (hy : y = 2 - I) :
  3*x + 4*y = 17 + 2*I := by sorry

end complex_expression_equality_l3040_304042


namespace space_shuttle_speed_km_per_second_l3040_304002

def orbit_speed_km_per_hour : ℝ := 43200

theorem space_shuttle_speed_km_per_second :
  orbit_speed_km_per_hour / 3600 = 12 := by
  sorry

end space_shuttle_speed_km_per_second_l3040_304002


namespace polynomial_identity_l3040_304096

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end polynomial_identity_l3040_304096


namespace problem_statement_l3040_304051

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : 
  (x - 1)^2 + 16/((x - 1)^2) = 8 := by
  sorry

end problem_statement_l3040_304051


namespace simplify_expression_expand_expression_l3040_304017

-- Problem 1
theorem simplify_expression (a : ℝ) : (2 * a^2)^3 + (-3 * a^3)^2 = 17 * a^6 := by
  sorry

-- Problem 2
theorem expand_expression (x y : ℝ) : (x + 3*y) * (x - y) = x^2 + 2*x*y - 3*y^2 := by
  sorry

end simplify_expression_expand_expression_l3040_304017


namespace transportation_budget_degrees_l3040_304073

theorem transportation_budget_degrees (salaries research_dev utilities equipment supplies : ℝ)
  (h1 : salaries = 60)
  (h2 : research_dev = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : supplies = 2)
  (h6 : salaries + research_dev + utilities + equipment + supplies < 100) :
  let transportation := 100 - (salaries + research_dev + utilities + equipment + supplies)
  360 * (transportation / 100) = 72 := by
  sorry

end transportation_budget_degrees_l3040_304073


namespace smallest_factor_of_4896_l3040_304011

theorem smallest_factor_of_4896 : 
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧ 
    10 ≤ b ∧ b < 100 ∧ 
    a * b = 4896 ∧ 
    (∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 4896 → min x y ≥ 32) :=
by sorry

end smallest_factor_of_4896_l3040_304011


namespace ice_cream_cone_types_l3040_304077

theorem ice_cream_cone_types (num_flavors : ℕ) (num_combinations : ℕ) (h1 : num_flavors = 4) (h2 : num_combinations = 8) :
  num_combinations / num_flavors = 2 := by
sorry

end ice_cream_cone_types_l3040_304077


namespace overhead_percentage_example_l3040_304025

/-- Given the purchase price, markup, and net profit of an article, 
    calculate the percentage of cost for overhead. -/
def overhead_percentage (purchase_price markup net_profit : ℚ) : ℚ :=
  let overhead := markup - net_profit
  (overhead / purchase_price) * 100

/-- Theorem stating that for the given values, the overhead percentage is 58.33% -/
theorem overhead_percentage_example : 
  overhead_percentage 48 40 12 = 58.33 := by
  sorry

end overhead_percentage_example_l3040_304025


namespace total_dolls_count_l3040_304070

/-- The number of dolls owned by the grandmother -/
def grandmother_dolls : ℕ := 50

/-- The number of dolls owned by the sister -/
def sister_dolls : ℕ := grandmother_dolls + 2

/-- The number of dolls owned by Rene -/
def rene_dolls : ℕ := 3 * sister_dolls

/-- The total number of dolls owned by Rene, her sister, and their grandmother -/
def total_dolls : ℕ := grandmother_dolls + sister_dolls + rene_dolls

theorem total_dolls_count : total_dolls = 258 := by
  sorry

end total_dolls_count_l3040_304070


namespace nonagon_perimeter_l3040_304060

/-- A regular nonagon is a polygon with 9 sides of equal length and equal angles -/
structure RegularNonagon where
  sideLength : ℝ
  numSides : ℕ
  numSides_eq : numSides = 9

/-- The perimeter of a regular nonagon is the product of its number of sides and side length -/
def perimeter (n : RegularNonagon) : ℝ := n.numSides * n.sideLength

/-- Theorem: The perimeter of a regular nonagon with side length 2 cm is 18 cm -/
theorem nonagon_perimeter :
  ∀ (n : RegularNonagon), n.sideLength = 2 → perimeter n = 18 := by
  sorry

end nonagon_perimeter_l3040_304060


namespace polynomial_expansion_l3040_304001

theorem polynomial_expansion (s : ℝ) :
  (3 * s^3 - 4 * s^2 + 5 * s - 2) * (2 * s^2 - 3 * s + 4) =
  6 * s^5 - 17 * s^4 + 34 * s^3 - 35 * s^2 + 26 * s - 8 := by
  sorry

end polynomial_expansion_l3040_304001


namespace custom_mult_example_l3040_304078

/-- Custom multiplication operation for rational numbers -/
def custom_mult (a b : ℚ) : ℚ := a * b + b ^ 2

/-- Theorem stating that 4 * (-2) = -4 using the custom multiplication -/
theorem custom_mult_example : custom_mult 4 (-2) = -4 := by sorry

end custom_mult_example_l3040_304078


namespace necessary_but_not_sufficient_condition_l3040_304023

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - x < 0.1) → (-0.1 < x ∧ x < 1.1) ∧
  ¬(∀ x : ℝ, (-0.1 < x ∧ x < 1.1) → (x^2 - x < 0.1)) :=
by sorry

end necessary_but_not_sufficient_condition_l3040_304023


namespace triangle_isosceles_if_bcosC_eq_CcosB_l3040_304052

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the cosine of an angle in a triangle
def cos_angle (t : Triangle) (angle : Fin 3) : ℝ :=
  sorry

-- Define the length of a side in a triangle
def side_length (t : Triangle) (side : Fin 3) : ℝ :=
  sorry

-- Define an isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  sorry

-- Theorem statement
theorem triangle_isosceles_if_bcosC_eq_CcosB (t : Triangle) :
  side_length t 1 * cos_angle t 2 = side_length t 2 * cos_angle t 1 →
  is_isosceles t :=
sorry

end triangle_isosceles_if_bcosC_eq_CcosB_l3040_304052


namespace tree_growth_theorem_l3040_304045

/-- Calculates the height of a tree after a given number of months -/
def treeHeight (initialHeight : ℝ) (growthRate : ℝ) (weeksPerMonth : ℕ) (months : ℕ) : ℝ :=
  initialHeight + growthRate * (months * weeksPerMonth : ℝ)

/-- Theorem: A tree with initial height 10 feet, growing 2 feet per week, 
    will be 42 feet tall after 4 months (with 4 weeks per month) -/
theorem tree_growth_theorem : 
  treeHeight 10 2 4 4 = 42 := by
  sorry

end tree_growth_theorem_l3040_304045


namespace three_books_purchase_ways_l3040_304090

/-- The number of ways to purchase books given the conditions -/
def purchase_ways (n : ℕ) : ℕ := 2^n - 1

/-- Theorem: There are 7 ways to purchase when there are 3 books -/
theorem three_books_purchase_ways :
  purchase_ways 3 = 7 := by
  sorry

end three_books_purchase_ways_l3040_304090


namespace solve_for_C_l3040_304084

theorem solve_for_C : ∃ C : ℝ, (4 * C + 5 = 37) ∧ (C = 8) := by
  sorry

end solve_for_C_l3040_304084


namespace sector_central_angle_l3040_304093

/-- Given a sector with radius R and area 2R^2, 
    the radian measure of its central angle is 4. -/
theorem sector_central_angle (R : ℝ) (h : R > 0) :
  let area := 2 * R^2
  let angle := (2 * area) / R^2
  angle = 4 := by sorry

end sector_central_angle_l3040_304093


namespace angle_terminal_side_value_l3040_304048

theorem angle_terminal_side_value (k : ℝ) (θ : ℝ) (h : k < 0) :
  (∃ (r : ℝ), r > 0 ∧ -4 * k = r * Real.cos θ ∧ 3 * k = r * Real.sin θ) →
  2 * Real.sin θ + Real.cos θ = -2/5 := by
  sorry

end angle_terminal_side_value_l3040_304048


namespace inverse_f_of_3_l3040_304020

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inverse_f_of_3 :
  ∃ (y : ℝ), y < 0 ∧ f y = 3 ∧ ∀ (z : ℝ), z < 0 ∧ f z = 3 → z = y :=
by sorry

end inverse_f_of_3_l3040_304020


namespace complex_fraction_simplification_l3040_304083

theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (1 - i) / (1 + i) = -i :=
by sorry

end complex_fraction_simplification_l3040_304083


namespace quadratic_coefficient_sum_l3040_304097

theorem quadratic_coefficient_sum (k : ℤ) : 
  (∃ x y : ℤ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + k*x + 25 = 0 ∧ y^2 + k*y + 25 = 0) → 
  k = 26 := by
sorry

end quadratic_coefficient_sum_l3040_304097


namespace packets_needed_l3040_304035

/-- Calculates the total number of packets needed for seedlings --/
def total_packets (oak_seedlings maple_seedlings pine_seedlings : ℕ) 
                  (oak_per_packet maple_per_packet pine_per_packet : ℕ) : ℕ :=
  (oak_seedlings / oak_per_packet) + 
  (maple_seedlings / maple_per_packet) + 
  (pine_seedlings / pine_per_packet)

/-- Theorem stating that the total number of packets needed is 395 --/
theorem packets_needed : 
  total_packets 420 825 2040 7 5 12 = 395 := by
  sorry

#eval total_packets 420 825 2040 7 5 12

end packets_needed_l3040_304035


namespace train_speed_l3040_304054

/-- Proves that a train crossing a bridge has a specific speed -/
theorem train_speed (train_length bridge_length : Real) (crossing_time : Real) :
  train_length = 110 ∧
  bridge_length = 112 ∧
  crossing_time = 11.099112071034318 →
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
sorry

end train_speed_l3040_304054


namespace inequality_proof_l3040_304061

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a * b^2 * c^3 * d^4 ≤ ((a + 2*b + 3*c + 4*d) / 10)^10 := by
  sorry

end inequality_proof_l3040_304061


namespace range_of_a_l3040_304075

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x + 1| ≤ 3 * a - a^2) → 1 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l3040_304075


namespace line_equation_l3040_304092

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m * x + b

-- Define the center of the circle
def center : ℝ × ℝ := (3, 5)

-- Define that line l passes through the center
def line_through_center (l : ℝ → ℝ → Prop) : Prop :=
  l center.1 center.2

-- Define points A and B on the circle and line
def point_on_circle_and_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  circle_C p.1 p.2 ∧ l p.1 p.2

-- Define point P on y-axis and line
def point_on_y_axis_and_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  p.1 = 0 ∧ l p.1 p.2

-- Define A as midpoint of BP
def A_midpoint_BP (A B P : ℝ × ℝ) : Prop :=
  A.1 = (B.1 + P.1) / 2 ∧ A.2 = (B.2 + P.2) / 2

-- Theorem statement
theorem line_equation :
  ∀ (A B P : ℝ × ℝ) (l : ℝ → ℝ → Prop),
    line_l = l →
    line_through_center l →
    point_on_circle_and_line A l →
    point_on_circle_and_line B l →
    point_on_y_axis_and_line P l →
    A_midpoint_BP A B P →
    (∀ x y, l x y ↔ (2*x - y - 1 = 0 ∨ 2*x + y - 11 = 0)) :=
by sorry

end line_equation_l3040_304092


namespace min_sum_of_chord_lengths_l3040_304057

/-- Parabola defined by y² = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Line passing through the focus with slope k -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - Focus.1)}

/-- Length of the chord formed by a line with slope k on the parabola -/
noncomputable def ChordLength (k : ℝ) : ℝ :=
  4 + 4 / k^2

/-- Sum of chord lengths for two lines with slopes k₁ and k₂ -/
noncomputable def SumOfChordLengths (k₁ k₂ : ℝ) : ℝ :=
  ChordLength k₁ + ChordLength k₂

/-- Theorem stating the minimum value of the sum of chord lengths -/
theorem min_sum_of_chord_lengths :
  ∀ k₁ k₂ : ℝ, k₁^2 + k₂^2 = 1 →
  24 ≤ SumOfChordLengths k₁ k₂ ∧
  (∃ k₁' k₂' : ℝ, k₁'^2 + k₂'^2 = 1 ∧ SumOfChordLengths k₁' k₂' = 24) :=
sorry

end min_sum_of_chord_lengths_l3040_304057


namespace min_c_value_l3040_304058

/-- Given positive integers a, b, c satisfying a < b < c, and a system of equations
    with exactly one solution, prove that the minimum value of c is 2002. -/
theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hab : a < b) (hbc : b < c)
    (h_unique : ∃! (x y : ℝ), 2 * x + y = 2004 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 2002 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 2002 ∧
    ∃! (x y : ℝ), 2 * x + y = 2004 ∧ y = |x - a'| + |x - b'| + |x - 2002| :=
by sorry

end min_c_value_l3040_304058


namespace only_2012_is_ternary_l3040_304072

def is_ternary (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 3 → d < 3

theorem only_2012_is_ternary :
  is_ternary 2012 ∧
  ¬is_ternary 2013 ∧
  ¬is_ternary 2014 ∧
  ¬is_ternary 2015 :=
by sorry

end only_2012_is_ternary_l3040_304072


namespace larger_number_proof_l3040_304079

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1345)
  (h2 : L = 6 * S + 15) : 
  L = 1611 := by
sorry

end larger_number_proof_l3040_304079


namespace sweetsies_remainder_l3040_304029

theorem sweetsies_remainder (m : ℕ) (h : m % 7 = 5) : (2 * m) % 7 = 3 := by
  sorry

end sweetsies_remainder_l3040_304029


namespace quadratic_inequality_solution_l3040_304049

theorem quadratic_inequality_solution (a : ℝ) :
  let solution_set := {x : ℝ | x^2 - a*x - 2*a^2 < 0}
  if a > 0 then
    solution_set = {x : ℝ | -a < x ∧ x < 2*a}
  else if a < 0 then
    solution_set = {x : ℝ | 2*a < x ∧ x < -a}
  else
    solution_set = ∅ :=
by sorry

end quadratic_inequality_solution_l3040_304049


namespace ellipse_max_dot_product_l3040_304022

/-- Definition of the ellipse M -/
def ellipse_M (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the circle N -/
def circle_N (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 1

/-- The ellipse passes through the point (2, √6) -/
def ellipse_point (a b : ℝ) : Prop :=
  ellipse_M a b 2 (Real.sqrt 6)

/-- The eccentricity of the ellipse is √2/2 -/
def ellipse_eccentricity (a b : ℝ) : Prop :=
  (Real.sqrt (a^2 - b^2)) / a = Real.sqrt 2 / 2

/-- Definition of the dot product PA · PB -/
def dot_product (x y : ℝ) : ℝ :=
  x^2 + y^2 - 4*y + 3

/-- Main theorem -/
theorem ellipse_max_dot_product (a b : ℝ) :
  ellipse_M a b 2 (Real.sqrt 6) →
  ellipse_eccentricity a b →
  (∀ x y : ℝ, ellipse_M a b x y → dot_product x y ≤ 23) ∧
  (∃ x y : ℝ, ellipse_M a b x y ∧ dot_product x y = 23) :=
sorry

end ellipse_max_dot_product_l3040_304022


namespace haley_current_height_l3040_304050

/-- Haley's growth rate in inches per year -/
def growth_rate : ℝ := 3

/-- Number of years in the future -/
def years : ℝ := 10

/-- Haley's height after 10 years in inches -/
def future_height : ℝ := 50

/-- Haley's current height in inches -/
def current_height : ℝ := future_height - growth_rate * years

theorem haley_current_height : current_height = 20 := by
  sorry

end haley_current_height_l3040_304050


namespace a_square_property_l3040_304055

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 14 * a (n + 1) - a n

theorem a_square_property : ∃ k : ℕ → ℤ, ∀ n : ℕ, 2 * a n - 1 = k n ^ 2 := by
  sorry

end a_square_property_l3040_304055


namespace roots_equation_result_l3040_304076

theorem roots_equation_result (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 + x₁ - 4 = 0) 
  (h₂ : x₂^2 + x₂ - 4 = 0) 
  (h₃ : x₁ ≠ x₂) : 
  x₁^3 - 5*x₂^2 + 10 = -19 := by
sorry

end roots_equation_result_l3040_304076


namespace gambler_initial_games_gambler_initial_games_proof_l3040_304040

theorem gambler_initial_games : ℝ → Prop :=
  fun x =>
    let initial_win_rate : ℝ := 0.4
    let new_win_rate : ℝ := 0.8
    let additional_games : ℝ := 30
    let final_win_rate : ℝ := 0.6
    (initial_win_rate * x + new_win_rate * additional_games) / (x + additional_games) = final_win_rate →
    x = 30

theorem gambler_initial_games_proof : ∃ x : ℝ, gambler_initial_games x := by
  sorry

end gambler_initial_games_gambler_initial_games_proof_l3040_304040


namespace power_inequality_l3040_304086

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^5 + b^5 > a^2 * b^3 + a^3 * b^2 := by
  sorry

end power_inequality_l3040_304086


namespace tablet_diagonal_l3040_304016

/-- Given two square tablets, if one has a diagonal of 5 inches and the other has a screen area 5.5 square inches larger, then the diagonal of the larger tablet is 6 inches. -/
theorem tablet_diagonal (d : ℝ) : 
  (d ^ 2 / 2 = 25 / 2 + 5.5) → d = 6 := by
  sorry

end tablet_diagonal_l3040_304016


namespace merchant_profit_l3040_304003

theorem merchant_profit (cost : ℝ) (cost_positive : cost > 0) : 
  let marked_price := cost * 1.2
  let discounted_price := marked_price * 0.9
  let profit := discounted_price - cost
  let profit_percentage := (profit / cost) * 100
  profit_percentage = 8 := by sorry

end merchant_profit_l3040_304003


namespace ceiling_negative_three_point_seven_l3040_304095

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 :=
sorry

end ceiling_negative_three_point_seven_l3040_304095


namespace orange_pricing_theorem_l3040_304064

/-- Pricing tiers for oranges -/
def price_4 : ℕ := 15
def price_7 : ℕ := 25
def price_10 : ℕ := 32

/-- Number of groups purchased -/
def num_groups : ℕ := 3

/-- Total number of oranges purchased -/
def total_oranges : ℕ := 4 * num_groups + 7 * num_groups + 10 * num_groups

/-- Calculate the total cost in cents -/
def total_cost : ℕ := price_4 * num_groups + price_7 * num_groups + price_10 * num_groups

/-- Calculate the average cost per orange in cents (as a rational number) -/
def avg_cost_per_orange : ℚ := total_cost / total_oranges

theorem orange_pricing_theorem :
  total_oranges = 21 ∧ 
  total_cost = 216 ∧ 
  avg_cost_per_orange = 1029 / 100 := by
  sorry

end orange_pricing_theorem_l3040_304064


namespace perpendicular_line_equation_l3040_304069

/-- A line passing through (1,1) and perpendicular to x+2y-3=0 has the equation y=2x-1 -/
theorem perpendicular_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ p : ℝ × ℝ, p ∈ l ↔ p.1 + 2 * p.2 - 3 = 0) →  -- Definition of line l'
  ((1, 1) ∈ l) →  -- l passes through (1,1)
  (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → (p.1 - q.1) * (p.1 + 2 * p.2 - 3 - (q.1 + 2 * q.2 - 3)) = 0) →  -- l is perpendicular to l'
  (∀ p : ℝ × ℝ, p ∈ l ↔ p.2 = 2 * p.1 - 1) :=  -- l has equation y=2x-1
by sorry

end perpendicular_line_equation_l3040_304069


namespace wood_rope_measurement_l3040_304005

/-- Represents the relationship between the length of a piece of wood and a rope used to measure it. -/
theorem wood_rope_measurement (x y : ℝ) :
  y = x + 4.5 ∧ 0.5 * y = x - 1 →
  (y - x = 4.5 ∧ y / 2 - x = -1) :=
by sorry

end wood_rope_measurement_l3040_304005


namespace digit_symmetrical_equation_l3040_304034

theorem digit_symmetrical_equation (a b : ℤ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10*a + b) * (100*b + 10*(a + b) + a) = (100*a + 10*(a + b) + b) * (10*b + a) := by
  sorry

end digit_symmetrical_equation_l3040_304034


namespace regular_polygon_interior_angle_sum_l3040_304087

theorem regular_polygon_interior_angle_sum 
  (n : ℕ) 
  (h_regular : n ≥ 3) 
  (h_exterior : (360 : ℝ) / n = 40) : 
  (n - 2 : ℝ) * 180 = 1260 := by
  sorry

end regular_polygon_interior_angle_sum_l3040_304087


namespace exists_permutation_9_not_exists_permutation_11_exists_permutation_1996_l3040_304043

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Theorem for n = 9 -/
theorem exists_permutation_9 :
  ∃ f : Fin 9 → Fin 9, Function.Bijective f ∧
    ∀ k : Fin 9, isPerfectSquare ((k : ℕ) + 1 + (f k : ℕ)) :=
sorry

/-- Theorem for n = 11 -/
theorem not_exists_permutation_11 :
  ¬ ∃ f : Fin 11 → Fin 11, Function.Bijective f ∧
    ∀ k : Fin 11, isPerfectSquare ((k : ℕ) + 1 + (f k : ℕ)) :=
sorry

/-- Theorem for n = 1996 -/
theorem exists_permutation_1996 :
  ∃ f : Fin 1996 → Fin 1996, Function.Bijective f ∧
    ∀ k : Fin 1996, isPerfectSquare ((k : ℕ) + 1 + (f k : ℕ)) :=
sorry

end exists_permutation_9_not_exists_permutation_11_exists_permutation_1996_l3040_304043


namespace bottle_caps_count_l3040_304047

/-- The number of bottle caps in the box after removing some and adding others. -/
def final_bottle_caps (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that given the initial conditions, the final number of bottle caps is 137. -/
theorem bottle_caps_count : final_bottle_caps 144 63 56 = 137 := by
  sorry

end bottle_caps_count_l3040_304047


namespace inequality_solution_set_l3040_304053

theorem inequality_solution_set (x : ℝ) :
  (x - 1) * (x + 2) < 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end inequality_solution_set_l3040_304053


namespace factorization_problem_1_factorization_problem_2_l3040_304010

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) : 
  2*x^2*y - 8*x*y + 8*y = 2*y*(x-2)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (x : ℝ) :
  x^4 - 81 = (x^2 + 9)*(x - 3)*(x + 3) := by sorry

end factorization_problem_1_factorization_problem_2_l3040_304010


namespace cookie_problem_l3040_304089

theorem cookie_problem (glenn kenny chris : ℕ) : 
  glenn = 24 →
  glenn = 4 * kenny →
  chris = kenny / 2 →
  glenn + kenny + chris = 33 :=
by sorry

end cookie_problem_l3040_304089


namespace robert_nickel_chocolate_difference_l3040_304028

theorem robert_nickel_chocolate_difference :
  let robert_chocolates : ℕ := 9
  let nickel_chocolates : ℕ := 2
  robert_chocolates - nickel_chocolates = 7 := by
sorry

end robert_nickel_chocolate_difference_l3040_304028


namespace distinct_points_difference_l3040_304081

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop := y^2 + x^4 = 3 * x^2 * y + 2

-- Define the constant e
noncomputable def e : ℝ := Real.exp 1

-- Theorem statement
theorem distinct_points_difference (a b : ℝ) 
  (ha : graph_equation (Real.sqrt e) a)
  (hb : graph_equation (Real.sqrt e) b)
  (hab : a ≠ b) : 
  |a - b| = Real.sqrt (5 * e^2 + 8) := by sorry

end distinct_points_difference_l3040_304081
