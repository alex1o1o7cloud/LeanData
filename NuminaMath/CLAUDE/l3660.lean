import Mathlib

namespace ratio_evaluation_l3660_366038

theorem ratio_evaluation : (2^2003 * 3^2002) / 6^2002 = 2 := by
  sorry

end ratio_evaluation_l3660_366038


namespace last_remaining_number_l3660_366080

def josephus_variant (n : ℕ) : ℕ :=
  let rec aux (k m : ℕ) : ℕ :=
    if k ≤ 1 then m
    else
      let m' := (m + 1) % k
      aux (k - 1) (2 * m' + 1)
  aux n 0

theorem last_remaining_number :
  josephus_variant 150 = 73 := by sorry

end last_remaining_number_l3660_366080


namespace soap_boxes_in_carton_l3660_366098

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the maximum number of smaller boxes that can fit in a larger box -/
def maxBoxesFit (largeBox smallBox : BoxDimensions) : ℕ :=
  (boxVolume largeBox) / (boxVolume smallBox)

theorem soap_boxes_in_carton :
  let carton : BoxDimensions := ⟨25, 42, 60⟩
  let soapBox : BoxDimensions := ⟨7, 12, 5⟩
  maxBoxesFit carton soapBox = 150 := by
  sorry

end soap_boxes_in_carton_l3660_366098


namespace zero_most_frequent_units_digit_l3660_366067

-- Define the set of numbers
def numbers : Finset ℕ := Finset.range 9

-- Function to calculate the units digit of a sum
def unitsDigitOfSum (a b : ℕ) : ℕ := (a + b) % 10

-- Function to count occurrences of a specific units digit
def countOccurrences (digit : ℕ) : ℕ :=
  numbers.card * numbers.card

-- Theorem stating that 0 is the most frequent units digit
theorem zero_most_frequent_units_digit :
  ∀ d : ℕ, d ∈ Finset.range 10 → d ≠ 0 →
    countOccurrences 0 > countOccurrences d :=
sorry

end zero_most_frequent_units_digit_l3660_366067


namespace jamie_speed_equals_alex_speed_l3660_366072

/-- Given the cycling speeds of Alex, Sam, and Jamie, prove that Jamie's speed equals Alex's speed. -/
theorem jamie_speed_equals_alex_speed (alex_speed : ℝ) (sam_speed : ℝ) (jamie_speed : ℝ)
  (h1 : alex_speed = 6)
  (h2 : sam_speed = 3/4 * alex_speed)
  (h3 : jamie_speed = 4/3 * sam_speed) :
  jamie_speed = alex_speed :=
by sorry

end jamie_speed_equals_alex_speed_l3660_366072


namespace trajectory_equation_l3660_366049

/-- Given a fixed point A(1, 2) and a moving point P(x, y) in a Cartesian coordinate system,
    if OP · OA = 4, then the equation of the trajectory of P is x + 2y - 4 = 0. -/
theorem trajectory_equation (x y : ℝ) :
  let A : ℝ × ℝ := (1, 2)
  let P : ℝ × ℝ := (x, y)
  let O : ℝ × ℝ := (0, 0)
  (P.1 - O.1) * (A.1 - O.1) + (P.2 - O.2) * (A.2 - O.2) = 4 →
  x + 2 * y - 4 = 0 :=
by sorry

end trajectory_equation_l3660_366049


namespace factorization_equality_l3660_366018

theorem factorization_equality (x y : ℝ) : 4 * x^2 - 8 * x * y + 4 * y^2 = 4 * (x - y)^2 := by
  sorry

end factorization_equality_l3660_366018


namespace arithmetic_expression_equality_l3660_366087

theorem arithmetic_expression_equality : 4 * 6 + 8 * 3 - 28 / 2 = 34 := by
  sorry

end arithmetic_expression_equality_l3660_366087


namespace hyperbola_intersection_line_l3660_366020

theorem hyperbola_intersection_line (θ : Real) : 
  let ρ := λ θ : Real => 3 / (1 - 2 * Real.cos θ)
  let A := (ρ θ, θ)
  let B := (ρ (θ + π), θ + π)
  let distance := |ρ θ + ρ (θ + π)|
  distance = 6 → 
    θ = π/2 ∨ θ = π/4 ∨ θ = 3*π/4 := by
  sorry

end hyperbola_intersection_line_l3660_366020


namespace total_ways_to_place_balls_l3660_366090

/-- The number of ways to place four distinct colored balls into two boxes -/
def place_balls : ℕ :=
  let box1_with_1_ball := Nat.choose 4 1
  let box1_with_2_balls := Nat.choose 4 2
  box1_with_1_ball + box1_with_2_balls

/-- Theorem stating that there are 10 ways to place the balls -/
theorem total_ways_to_place_balls : place_balls = 10 := by
  sorry

end total_ways_to_place_balls_l3660_366090


namespace charlottes_age_l3660_366054

theorem charlottes_age (B E C : ℚ) 
  (h1 : B = 4 * C)
  (h2 : E = C + 5)
  (h3 : B = E) :
  C = 5 / 3 := by
  sorry

end charlottes_age_l3660_366054


namespace ourSystem_is_linear_l3660_366093

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  toFun : ℝ → ℝ → Prop := fun x y => a * x + b * y = c

/-- A system of two equations -/
structure SystemOfTwoEquations where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system we want to prove is linear -/
def ourSystem : SystemOfTwoEquations where
  eq1 := { a := 1, b := 1, c := 5 }
  eq2 := { a := 0, b := 1, c := 2 }

/-- Predicate to check if a system is linear -/
def isLinearSystem (system : SystemOfTwoEquations) : Prop :=
  system.eq1.a ≠ 0 ∨ system.eq1.b ≠ 0 ∧
  system.eq2.a ≠ 0 ∨ system.eq2.b ≠ 0

theorem ourSystem_is_linear : isLinearSystem ourSystem := by
  sorry

end ourSystem_is_linear_l3660_366093


namespace specific_stack_logs_l3660_366082

/-- Represents a triangular stack of logs. -/
structure LogStack where
  bottom_logs : ℕ  -- Number of logs in the bottom row
  decrement : ℕ    -- Number of logs decreased in each row
  top_logs : ℕ     -- Number of logs in the top row

/-- Calculates the number of rows in a log stack. -/
def num_rows (stack : LogStack) : ℕ :=
  (stack.bottom_logs - stack.top_logs) / stack.decrement + 1

/-- Calculates the total number of logs in a stack. -/
def total_logs (stack : LogStack) : ℕ :=
  let n := num_rows stack
  n * (stack.bottom_logs + stack.top_logs) / 2

/-- Theorem stating the total number of logs in the specific stack. -/
theorem specific_stack_logs :
  let stack : LogStack := ⟨15, 2, 1⟩
  total_logs stack = 64 := by
  sorry


end specific_stack_logs_l3660_366082


namespace termite_ridden_not_collapsing_l3660_366091

-- Define the streets
inductive Street
| Batman
| Robin
| Joker

-- Define the properties for each street
def termite_ridden_fraction (s : Street) : ℚ :=
  match s with
  | Street.Batman => 1/3
  | Street.Robin => 3/7
  | Street.Joker => 1/2

def collapsing_fraction (s : Street) : ℚ :=
  match s with
  | Street.Batman => 7/10
  | Street.Robin => 4/5
  | Street.Joker => 3/8

-- Theorem to prove
theorem termite_ridden_not_collapsing (s : Street) :
  (termite_ridden_fraction s) * (1 - collapsing_fraction s) =
    match s with
    | Street.Batman => 1/10
    | Street.Robin => 3/35
    | Street.Joker => 5/16
    := by sorry

end termite_ridden_not_collapsing_l3660_366091


namespace simplify_and_rationalize_denominator_l3660_366013

theorem simplify_and_rationalize_denominator :
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplify_and_rationalize_denominator_l3660_366013


namespace age_ratio_in_one_year_l3660_366073

/-- Mike's current age -/
def m : ℕ := sorry

/-- Sarah's current age -/
def s : ℕ := sorry

/-- The condition that 3 years ago, Mike was twice as old as Sarah -/
axiom three_years_ago : m - 3 = 2 * (s - 3)

/-- The condition that 5 years ago, Mike was three times as old as Sarah -/
axiom five_years_ago : m - 5 = 3 * (s - 5)

/-- The number of years until the ratio of their ages is 3:2 -/
def years_until_ratio : ℕ := sorry

/-- The theorem stating that the number of years until the ratio of their ages is 3:2 is 1 -/
theorem age_ratio_in_one_year : 
  years_until_ratio = 1 ∧ (m + years_until_ratio) / (s + years_until_ratio) = 3 / 2 := by sorry

end age_ratio_in_one_year_l3660_366073


namespace davids_trip_expenses_l3660_366069

theorem davids_trip_expenses (initial_amount spent_amount remaining_amount : ℕ) : 
  initial_amount = 1800 →
  remaining_amount = 500 →
  spent_amount = initial_amount - remaining_amount →
  spent_amount - remaining_amount = 800 := by
  sorry

end davids_trip_expenses_l3660_366069


namespace beach_house_rent_l3660_366086

/-- The total amount paid for rent by a group of people -/
def total_rent (num_people : ℕ) (rent_per_person : ℚ) : ℚ :=
  num_people * rent_per_person

/-- Proof that 7 people paying $70.00 each results in a total of $490.00 -/
theorem beach_house_rent :
  total_rent 7 70 = 490 := by
  sorry

end beach_house_rent_l3660_366086


namespace finishing_order_equals_starting_order_l3660_366039

/-- Represents an athlete in the race -/
inductive Athlete : Type
  | Grisha : Athlete
  | Sasha : Athlete
  | Lena : Athlete

/-- Represents the order of athletes -/
def AthleteOrder := List Athlete

/-- The starting order of the race -/
def startingOrder : AthleteOrder := [Athlete.Grisha, Athlete.Sasha, Athlete.Lena]

/-- The number of overtakes by each athlete -/
def overtakes : Athlete → Nat
  | Athlete.Grisha => 10
  | Athlete.Sasha => 4
  | Athlete.Lena => 6

/-- No three athletes were at the same position simultaneously -/
axiom no_triple_overtake : True

/-- All athletes finished at different times -/
axiom different_finish_times : True

/-- The finishing order of the race -/
def finishingOrder : AthleteOrder := sorry

/-- Theorem stating that the finishing order is the same as the starting order -/
theorem finishing_order_equals_starting_order : 
  finishingOrder = startingOrder := by sorry

end finishing_order_equals_starting_order_l3660_366039


namespace points_on_line_l3660_366014

/-- Given a line with equation x = 2y + 5, prove that for any real number n,
    the points (m, n) and (m + 1, n + 0.5) lie on this line, where m = 2n + 5. -/
theorem points_on_line (n : ℝ) : 
  let m : ℝ := 2 * n + 5
  let point1 : ℝ × ℝ := (m, n)
  let point2 : ℝ × ℝ := (m + 1, n + 0.5)
  (point1.1 = 2 * point1.2 + 5) ∧ (point2.1 = 2 * point2.2 + 5) :=
by
  sorry


end points_on_line_l3660_366014


namespace smallest_prime_perimeter_scalene_triangle_l3660_366081

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a scalene triangle -/
def isScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ 
  b = a + 2 ∧ c = b + 2 ∧
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1

theorem smallest_prime_perimeter_scalene_triangle : 
  ∃ (a b c : ℕ), 
    isScaleneTriangle a b c ∧ 
    areConsecutiveOddPrimes a b c ∧ 
    isPrime (a + b + c) ∧
    ∀ (x y z : ℕ), 
      isScaleneTriangle x y z → 
      areConsecutiveOddPrimes x y z → 
      isPrime (x + y + z) → 
      a + b + c ≤ x + y + z ∧
    a + b + c = 23 :=
sorry

end smallest_prime_perimeter_scalene_triangle_l3660_366081


namespace abs_x_minus_one_necessary_not_sufficient_l3660_366027

theorem abs_x_minus_one_necessary_not_sufficient :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x - 3) < 0)) := by
  sorry

end abs_x_minus_one_necessary_not_sufficient_l3660_366027


namespace board_meeting_arrangement_l3660_366043

/-- The number of ways to arrange 3 indistinguishable objects among 8 positions -/
def arrangement_count : ℕ := 56

/-- The total number of seats -/
def total_seats : ℕ := 10

/-- The number of stools (men) -/
def stool_count : ℕ := 5

/-- The number of rocking chairs (women) -/
def chair_count : ℕ := 5

/-- The number of positions to fill after fixing first and last seats -/
def remaining_positions : ℕ := total_seats - 2

/-- The number of remaining stools to place after fixing first and last seats -/
def remaining_stools : ℕ := stool_count - 2

theorem board_meeting_arrangement :
  arrangement_count = Nat.choose remaining_positions remaining_stools := by
  sorry

end board_meeting_arrangement_l3660_366043


namespace total_marks_for_exam_l3660_366036

/-- Calculates the total marks given the number of candidates and average score -/
def totalMarks (numCandidates : ℕ) (averageScore : ℚ) : ℚ :=
  numCandidates * averageScore

/-- Proves that for 250 candidates with an average score of 42, the total marks is 10500 -/
theorem total_marks_for_exam : totalMarks 250 42 = 10500 := by
  sorry

#eval totalMarks 250 42

end total_marks_for_exam_l3660_366036


namespace quadratic_equation_solutions_l3660_366083

theorem quadratic_equation_solutions (x : ℝ) :
  (x^2 + 2*x + 1 = 4) ↔ (x = 1 ∨ x = -3) := by
  sorry

end quadratic_equation_solutions_l3660_366083


namespace regular_polygon_sides_l3660_366085

theorem regular_polygon_sides (interior_angle : ℝ) (sum_except_one : ℝ) : 
  interior_angle = 160 → sum_except_one = 3600 → 
  (sum_except_one + interior_angle) / interior_angle = 24 := by
  sorry

end regular_polygon_sides_l3660_366085


namespace doubled_speed_cleaning_time_l3660_366009

def house_cleaning (bruce_rate anne_rate : ℝ) : Prop :=
  bruce_rate > 0 ∧ anne_rate > 0 ∧
  bruce_rate + anne_rate = 1 / 4 ∧
  anne_rate = 1 / 12

theorem doubled_speed_cleaning_time (bruce_rate anne_rate : ℝ) 
  (h : house_cleaning bruce_rate anne_rate) : 
  1 / (bruce_rate + 2 * anne_rate) = 3 := by
  sorry

end doubled_speed_cleaning_time_l3660_366009


namespace parabola_equation_l3660_366005

/-- Given a parabola in the form x² = 2py where p > 0, with axis of symmetry y = -1/2,
    prove that its equation is x² = 2y -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) (h2 : -p/2 = -1/2) :
  ∀ x y : ℝ, x^2 = 2*p*y ↔ x^2 = 2*y :=
sorry

end parabola_equation_l3660_366005


namespace work_problem_solution_l3660_366045

/-- Proves that given the conditions of the work problem, the daily wage of worker c is 115 --/
theorem work_problem_solution (a b c : ℕ) : 
  (a : ℚ) / 3 = (b : ℚ) / 4 ∧ (a : ℚ) / 3 = (c : ℚ) / 5 →  -- daily wages ratio
  6 * a + 9 * b + 4 * c = 1702 →                          -- total earnings
  c = 115 := by
  sorry

end work_problem_solution_l3660_366045


namespace unique_number_with_three_prime_divisors_l3660_366035

theorem unique_number_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    x = p * q * r ∧ 
    (∀ s : ℕ, Nat.Prime s ∧ s ∣ x → s = p ∨ s = q ∨ s = r)) →
  7 ∣ x →
  x = 728 :=
by sorry

end unique_number_with_three_prime_divisors_l3660_366035


namespace smallest_n_for_inequality_l3660_366051

theorem smallest_n_for_inequality : ∃ n : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ m : ℕ, m < n → ∃ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) ∧
  n = 4 :=
by sorry

end smallest_n_for_inequality_l3660_366051


namespace polynomial_remainder_theorem_l3660_366003

theorem polynomial_remainder_theorem (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^5 - 8*x^4 + 15*x^3 + 30*x^2 - 47*x + 20
  (f 2) = 104 := by sorry

end polynomial_remainder_theorem_l3660_366003


namespace doughnut_cost_l3660_366023

/-- The cost of one dozen doughnuts -/
def cost_one_dozen : ℝ := sorry

/-- The cost of two dozen doughnuts -/
def cost_two_dozen : ℝ := 14

theorem doughnut_cost : cost_one_dozen = 7 := by
  sorry

end doughnut_cost_l3660_366023


namespace probability_in_pascal_triangle_l3660_366002

/-- The number of rows in Pascal's Triangle we're considering --/
def n : ℕ := 20

/-- The total number of elements in the first n rows of Pascal's Triangle --/
def total_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of ones in the first n rows of Pascal's Triangle --/
def ones_count (n : ℕ) : ℕ := 2 * (n - 1) + 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle --/
def probability_of_one (n : ℕ) : ℚ :=
  (ones_count n : ℚ) / (total_elements n : ℚ)

theorem probability_in_pascal_triangle :
  probability_of_one n = 13 / 70 := by
  sorry

end probability_in_pascal_triangle_l3660_366002


namespace manuscript_typing_cost_l3660_366089

/-- The cost per page for revisions -/
def revision_cost : ℝ := 4

/-- The total number of pages in the manuscript -/
def total_pages : ℕ := 100

/-- The number of pages revised once -/
def pages_revised_once : ℕ := 35

/-- The number of pages revised twice -/
def pages_revised_twice : ℕ := 15

/-- The total cost of typing the manuscript -/
def total_cost : ℝ := 860

/-- The cost per page for the first time a page is typed -/
def first_time_cost : ℝ := 6

theorem manuscript_typing_cost :
  first_time_cost * total_pages +
  revision_cost * pages_revised_once +
  2 * revision_cost * pages_revised_twice = total_cost :=
by sorry

end manuscript_typing_cost_l3660_366089


namespace median_same_variance_decreases_l3660_366064

def original_data : List ℝ := [2, 2, 4, 4]
def new_data : List ℝ := [2, 2, 3, 4, 4]

def median (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem median_same_variance_decreases :
  median original_data = median new_data ∧
  variance new_data < variance original_data := by sorry

end median_same_variance_decreases_l3660_366064


namespace data_value_proof_l3660_366001

theorem data_value_proof (a b c : ℝ) 
  (h1 : a + b = c)
  (h2 : b = 3 * a)
  (h3 : a + b + c = 96) :
  a = 12 := by
  sorry

end data_value_proof_l3660_366001


namespace equation_root_implies_m_value_l3660_366075

theorem equation_root_implies_m_value (x m : ℝ) :
  (∃ x, (x - 1) / (x - 4) = m / (x - 4)) → m = 3 := by
  sorry

end equation_root_implies_m_value_l3660_366075


namespace orange_distribution_l3660_366007

/-- Given a number of oranges, pieces per orange, and number of friends,
    calculate the number of pieces each friend receives. -/
def pieces_per_friend (oranges : ℕ) (pieces_per_orange : ℕ) (friends : ℕ) : ℚ :=
  (oranges * pieces_per_orange : ℚ) / friends

/-- Theorem stating that given 80 oranges, each divided into 10 pieces,
    and 200 friends, each friend will receive 4 pieces. -/
theorem orange_distribution :
  pieces_per_friend 80 10 200 = 4 := by
  sorry

end orange_distribution_l3660_366007


namespace danny_bottle_caps_l3660_366041

theorem danny_bottle_caps (thrown_away : ℕ) (found : ℕ) (final : ℕ) :
  thrown_away = 60 →
  found = 58 →
  final = 67 →
  final = (thrown_away - found + final) →
  thrown_away - found + final = 69 :=
by sorry

end danny_bottle_caps_l3660_366041


namespace dinos_remaining_balance_l3660_366008

/-- Represents a gig with hours worked per month and hourly rate -/
structure Gig where
  hours : ℕ
  rate : ℕ

/-- Calculates the monthly earnings from a gig -/
def monthlyEarnings (g : Gig) : ℕ := g.hours * g.rate

/-- Represents Dino's gigs -/
def dinos_gigs : List Gig := [
  ⟨20, 10⟩,
  ⟨30, 20⟩,
  ⟨5, 40⟩,
  ⟨15, 25⟩,
  ⟨10, 30⟩
]

/-- Dino's monthly expenses for each month -/
def monthly_expenses : List ℕ := [500, 550, 520, 480]

/-- The number of months -/
def num_months : ℕ := 4

theorem dinos_remaining_balance :
  (dinos_gigs.map monthlyEarnings).sum * num_months -
  monthly_expenses.sum = 4650 := by sorry

end dinos_remaining_balance_l3660_366008


namespace perpendicular_lines_condition_l3660_366021

theorem perpendicular_lines_condition (m : ℝ) :
  (m = -1) ↔ (∀ x y : ℝ, mx + y - 3 = 0 → 2*x + m*(m-1)*y + 2 = 0 → m*2 + 1*m*(m-1) = 0) :=
by sorry

end perpendicular_lines_condition_l3660_366021


namespace range_of_alpha_minus_half_beta_l3660_366065

theorem range_of_alpha_minus_half_beta (α β : Real) 
  (h_α : 0 ≤ α ∧ α ≤ π/2) 
  (h_β : π/2 ≤ β ∧ β ≤ π) : 
  ∃ (x : Real), x = α - β/2 ∧ -π/2 ≤ x ∧ x ≤ π/4 :=
by sorry

end range_of_alpha_minus_half_beta_l3660_366065


namespace fly_distance_from_ceiling_l3660_366046

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the configuration of a room with a sloped ceiling -/
structure Room where
  p : Point3D -- Point where walls and ceiling meet
  slope : ℝ -- Slope of the ceiling (rise / run)

/-- Represents the position of a fly in the room -/
structure FlyPosition where
  distWall1 : ℝ -- Distance from first wall
  distWall2 : ℝ -- Distance from second wall
  distFromP : ℝ -- Distance from point P

/-- Calculates the distance of a fly from the sloped ceiling in a room -/
def distanceFromCeiling (r : Room) (f : FlyPosition) : ℝ :=
  sorry

/-- Theorem stating that the fly's distance from the ceiling is (3√60 - 8)/3 -/
theorem fly_distance_from_ceiling (r : Room) (f : FlyPosition) :
  r.p = Point3D.mk 0 0 0 →
  r.slope = 1/3 →
  f.distWall1 = 2 →
  f.distWall2 = 6 →
  f.distFromP = 10 →
  distanceFromCeiling r f = (3 * Real.sqrt 60 - 8) / 3 :=
sorry

end fly_distance_from_ceiling_l3660_366046


namespace quadratic_function_range_l3660_366088

theorem quadratic_function_range (x : ℝ) :
  let y := x^2 - 4*x + 3
  y < 0 ↔ 1 < x ∧ x < 3 := by sorry

end quadratic_function_range_l3660_366088


namespace fraction_simplification_l3660_366074

theorem fraction_simplification :
  (16 : ℚ) / 54 * 27 / 8 * 64 / 81 = 64 / 9 := by
  sorry

end fraction_simplification_l3660_366074


namespace range_of_f_l3660_366084

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 7

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -2} := by sorry

end range_of_f_l3660_366084


namespace triangle_side_sum_max_l3660_366017

theorem triangle_side_sum_max (a b c : ℝ) (C : ℝ) :
  C = π / 3 →
  c = Real.sqrt 3 →
  a > 0 →
  b > 0 →
  c > 0 →
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
  a + b ≤ 2 * Real.sqrt 3 :=
by sorry

end triangle_side_sum_max_l3660_366017


namespace water_consumption_calculation_l3660_366026

/-- Water billing system and consumption calculation -/
theorem water_consumption_calculation 
  (base_rate : ℝ) 
  (excess_rate : ℝ) 
  (sewage_rate : ℝ) 
  (base_volume : ℝ) 
  (total_bill : ℝ) 
  (h1 : base_rate = 1.8) 
  (h2 : excess_rate = 2.3) 
  (h3 : sewage_rate = 1) 
  (h4 : base_volume = 15) 
  (h5 : total_bill = 58.5) : 
  ∃ (consumption : ℝ), 
    consumption = 20 ∧ 
    total_bill = 
      base_rate * min consumption base_volume + 
      excess_rate * max (consumption - base_volume) 0 + 
      sewage_rate * consumption :=
sorry

end water_consumption_calculation_l3660_366026


namespace tangent_line_at_2_2_tangent_lines_through_origin_l3660_366066

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x + 1

-- Theorem for part (I)
theorem tangent_line_at_2_2 :
  ∃ (m b : ℝ), (∀ x y, y = m*x + b ↔ 5*x - y - 8 = 0) ∧
               f 2 = 2 ∧
               f' 2 = m :=
sorry

-- Theorem for part (II)
theorem tangent_lines_through_origin :
  ∃ (x₁ x₂ : ℝ),
    (f x₁ = 0 ∧ f' x₁ = 1 ∧ x₁ ≠ x₂) ∧
    (f x₂ = 0 ∧ f' x₂ = 0) :=
sorry

end tangent_line_at_2_2_tangent_lines_through_origin_l3660_366066


namespace negation_of_universal_proposition_l3660_366047

theorem negation_of_universal_proposition 
  (f : ℝ → ℝ) (m : ℝ) : 
  (¬ ∀ x, f x ≥ m) ↔ ∃ x, f x < m :=
by sorry

end negation_of_universal_proposition_l3660_366047


namespace min_value_reciprocal_sum_l3660_366052

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_reciprocal_sum_l3660_366052


namespace range_of_fraction_l3660_366006

theorem range_of_fraction (x y : ℝ) (h1 : 2*x + y = 8) (h2 : 2 ≤ x) (h3 : x ≤ 3) :
  3/2 ≤ (y+1)/(x-1) ∧ (y+1)/(x-1) ≤ 5 := by
sorry

end range_of_fraction_l3660_366006


namespace intersection_point_Q_l3660_366079

-- Define the circles
def circle1 (x y r : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = r^2
def circle2 (x y R : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = R^2

-- Define the intersection points
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (-2, -1)

-- Theorem statement
theorem intersection_point_Q :
  ∀ (r R : ℝ),
  (∃ (x y : ℝ), circle1 x y r ∧ circle2 x y R) →  -- Circles intersect
  circle1 P.1 P.2 r →                            -- P is on circle1
  circle2 P.1 P.2 R →                            -- P is on circle2
  circle1 Q.1 Q.2 r ∧ circle2 Q.1 Q.2 R          -- Q is on both circles
  := by sorry

end intersection_point_Q_l3660_366079


namespace sufficient_not_necessary_condition_l3660_366094

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 1 ≤ x^2 ∧ x^2 ≤ 16) ∧ 
  (∃ x : ℝ, 1 ≤ x^2 ∧ x^2 ≤ 16 ∧ ¬(1 ≤ x ∧ x ≤ 4)) :=
by sorry

end sufficient_not_necessary_condition_l3660_366094


namespace pentagon_to_squares_area_ratio_l3660_366040

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  a : Point
  side : ℝ

/-- Calculate the area of a square -/
def squareArea (s : Square) : ℝ := s.side * s.side

/-- Calculate the area of a pentagon given its vertices -/
def pentagonArea (a b c d e : Point) : ℝ := sorry

/-- Main theorem: The ratio of the pentagon area to the sum of square areas is 5/12 -/
theorem pentagon_to_squares_area_ratio 
  (squareABCD squareEFGH squareKLMO : Square)
  (a b c d e f g h k l m o : Point) :
  squareABCD.side = 1 →
  squareEFGH.side = 2 →
  squareKLMO.side = 1 →
  b.x = h.x ∧ b.y = e.y → -- AB aligns with HE
  g.x = o.x ∧ m.y = k.y → -- GM aligns with OK
  d.x = (h.x + e.x) / 2 ∧ d.y = h.y → -- D is midpoint of HE
  c.x = h.x + (2/3) * (g.x - h.x) ∧ c.y = h.y → -- C is one-third along HG from H
  (pentagonArea a m k c b) / (squareArea squareABCD + squareArea squareEFGH + squareArea squareKLMO) = 5/12 := by
  sorry


end pentagon_to_squares_area_ratio_l3660_366040


namespace profit_percentage_l3660_366028

theorem profit_percentage (selling_price cost_price profit : ℝ) : 
  cost_price = 0.75 * selling_price →
  profit = selling_price - cost_price →
  (profit / cost_price) * 100 = 100/3 :=
by sorry

end profit_percentage_l3660_366028


namespace set_intersection_subset_l3660_366050

theorem set_intersection_subset (a : ℝ) : 
  let A := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 22}
  (A.Nonempty ∧ B.Nonempty) → (A ⊆ A ∩ B) ↔ (6 ≤ a ∧ a ≤ 9) :=
by sorry

end set_intersection_subset_l3660_366050


namespace sqrt_sum_max_value_l3660_366053

theorem sqrt_sum_max_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x y : ℝ), 0 < x → 0 < y → x + y = 2 → Real.sqrt x + Real.sqrt y ≤ max :=
sorry

end sqrt_sum_max_value_l3660_366053


namespace scientific_notation_of_1040000000_l3660_366010

theorem scientific_notation_of_1040000000 :
  (1040000000 : ℝ) = 1.04 * (10 : ℝ)^9 := by sorry

end scientific_notation_of_1040000000_l3660_366010


namespace quadratic_factorization_l3660_366004

theorem quadratic_factorization (b : ℤ) : 
  (∃ (m n p q : ℤ), 15 * x^2 + b * x + 30 = (m * x + n) * (p * x + q)) ↔ b = 43 :=
sorry

end quadratic_factorization_l3660_366004


namespace opposite_roots_quadratic_l3660_366070

theorem opposite_roots_quadratic (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (k^2 - 4)*x₁ + k + 1 = 0 ∧ 
    x₂^2 + (k^2 - 4)*x₂ + k + 1 = 0 ∧
    x₁ = -x₂) → 
  k = -2 := by
sorry

end opposite_roots_quadratic_l3660_366070


namespace equation_solutions_l3660_366000

open Complex

-- Define the set of solutions
def solutions : Set ℂ :=
  {2, -2, 1 + Complex.I * Real.sqrt 3, 1 - Complex.I * Real.sqrt 3,
   -1 + Complex.I * Real.sqrt 3, -1 - Complex.I * Real.sqrt 3}

-- State the theorem
theorem equation_solutions :
  {x : ℂ | x^6 - 64 = 0} = solutions :=
by sorry

end equation_solutions_l3660_366000


namespace derivative_of_fraction_l3660_366095

theorem derivative_of_fraction (x : ℝ) :
  let y : ℝ → ℝ := λ x => (1 - Real.cos (2 * x)) / (1 + Real.cos (2 * x))
  HasDerivAt y (4 * Real.sin (2 * x) / (1 + Real.cos (2 * x))^2) x :=
by
  sorry

end derivative_of_fraction_l3660_366095


namespace number_of_rattlesnakes_l3660_366012

theorem number_of_rattlesnakes (P B R V : ℕ) : 
  P + B + R + V = 420 →
  P = (3 * B) / 2 →
  V = 8 →
  P + R = 315 →
  R = 162 := by
sorry

end number_of_rattlesnakes_l3660_366012


namespace smaller_solution_quadratic_equation_l3660_366033

theorem smaller_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 15*x - 56
  let sol₁ : ℝ := (15 - Real.sqrt 449) / 2
  let sol₂ : ℝ := (15 + Real.sqrt 449) / 2
  f sol₁ = 0 ∧ f sol₂ = 0 ∧ sol₁ < sol₂ ∧ 
  ∀ x : ℝ, f x = 0 → x = sol₁ ∨ x = sol₂ :=
by sorry

end smaller_solution_quadratic_equation_l3660_366033


namespace chocolates_remaining_day5_l3660_366077

/-- Calculates the number of chocolates remaining after 4 days of consumption -/
def chocolates_remaining (initial : ℕ) (day1 : ℕ) : ℕ :=
  let day2 := 2 * day1 - 3
  let day3 := day1 - 2
  let day4 := day3 - 1
  initial - (day1 + day2 + day3 + day4)

/-- Theorem stating that given the initial conditions, 12 chocolates remain on Day 5 -/
theorem chocolates_remaining_day5 :
  chocolates_remaining 24 4 = 12 := by
  sorry

#eval chocolates_remaining 24 4

end chocolates_remaining_day5_l3660_366077


namespace ellipse_properties_l3660_366011

/-- Properties of the ellipse 9x^2 + y^2 = 81 -/
theorem ellipse_properties :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 81}
  ∃ (major_axis minor_axis eccentricity : ℝ) 
    (foci_y vertex_y vertex_x : ℝ),
    -- Length of major axis
    major_axis = 18 ∧
    -- Length of minor axis
    minor_axis = 6 ∧
    -- Eccentricity
    eccentricity = 2 * Real.sqrt 2 / 3 ∧
    -- Foci coordinates
    foci_y = 6 * Real.sqrt 2 ∧
    (0, foci_y) ∈ ellipse ∧ (0, -foci_y) ∈ ellipse ∧
    -- Vertex coordinates
    vertex_y = 9 ∧ vertex_x = 3 ∧
    (0, vertex_y) ∈ ellipse ∧ (0, -vertex_y) ∈ ellipse ∧
    (vertex_x, 0) ∈ ellipse ∧ (-vertex_x, 0) ∈ ellipse :=
by
  sorry

end ellipse_properties_l3660_366011


namespace square_of_complex_l3660_366099

theorem square_of_complex (z : ℂ) (i : ℂ) : z = 5 + 2 * i → i ^ 2 = -1 → z ^ 2 = 21 + 20 * i := by
  sorry

end square_of_complex_l3660_366099


namespace expression_evaluation_l3660_366071

theorem expression_evaluation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  let f := (((x + 2)^2 * (x^2 - 2*x + 4)^2) / (x^3 + 8)^2)^2 *
            (((x - 2)^2 * (x^2 + 2*x + 4)^2) / (x^3 - 8)^2)^2
  f = 1 := by
  sorry

end expression_evaluation_l3660_366071


namespace value_of_p_l3660_366037

theorem value_of_p (n : ℝ) (p : ℝ) : 
  n = 9/4 → p = 4*n*(1/2^2009)^(Real.log 1) → p = 9 := by sorry

end value_of_p_l3660_366037


namespace joan_seashells_l3660_366022

theorem joan_seashells (total : ℝ) (percentage : ℝ) (remaining : ℝ) : 
  total = 79.5 → 
  percentage = 45 → 
  remaining = total - (percentage / 100) * total → 
  remaining = 43.725 := by
sorry

end joan_seashells_l3660_366022


namespace angle_between_vectors_l3660_366019

def a : ℝ × ℝ := (1, 1)

theorem angle_between_vectors (b : ℝ × ℝ) 
  (h : (4 * a.1, 4 * a.2) + b = (4, 2)) : 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 3 * π / 4 := by
  sorry

end angle_between_vectors_l3660_366019


namespace rationalize_denominator_l3660_366058

theorem rationalize_denominator :
  Real.sqrt (5 / (2 + Real.sqrt 2)) = Real.sqrt 5 - Real.sqrt 10 / 2 := by
  sorry

end rationalize_denominator_l3660_366058


namespace watermelon_ratio_l3660_366034

theorem watermelon_ratio (michael_weight john_weight : ℚ) : 
  michael_weight = 8 →
  john_weight = 12 →
  john_weight / (3 * michael_weight) = 1 / 2 := by
  sorry

end watermelon_ratio_l3660_366034


namespace product_closed_in_P_l3660_366044

/-- The set of perfect squares -/
def P : Set ℕ := {n : ℕ | ∃ m : ℕ, m > 0 ∧ n = m^2}

/-- The theorem stating that the product of two elements in P is also in P -/
theorem product_closed_in_P (a b : ℕ) (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P := by
  sorry

#check product_closed_in_P

end product_closed_in_P_l3660_366044


namespace lemonade_lemons_l3660_366068

/-- Given that each glass of lemonade requires 2 lemons and Jane can make 9 glasses,
    prove that the total number of lemons is 18. -/
theorem lemonade_lemons :
  ∀ (lemons_per_glass : ℕ) (glasses : ℕ) (total_lemons : ℕ),
    lemons_per_glass = 2 →
    glasses = 9 →
    total_lemons = lemons_per_glass * glasses →
    total_lemons = 18 := by
  sorry

end lemonade_lemons_l3660_366068


namespace roots_on_circle_l3660_366096

theorem roots_on_circle : ∃ (r : ℝ), r = 1 / Real.sqrt 3 ∧
  ∀ (z : ℂ), (z - 1)^3 = 8*z^3 → Complex.abs (z + 1/3) = r := by
  sorry

end roots_on_circle_l3660_366096


namespace triangle_angle_45_l3660_366057

/-- Given a triangle with sides a, b, c, perimeter 2s, and area T,
    if T + (ab/2) = s(s-c), then the angle opposite side c is 45°. -/
theorem triangle_angle_45 (a b c s T : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_perimeter : a + b + c = 2 * s) (h_area : T > 0)
    (h_equation : T + (a * b / 2) = s * (s - c)) :
    let γ := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
    γ = π / 4 := by
  sorry

end triangle_angle_45_l3660_366057


namespace harriet_trip_time_l3660_366024

theorem harriet_trip_time (total_time : ℝ) (outbound_speed return_speed : ℝ) 
  (h1 : total_time = 5)
  (h2 : outbound_speed = 90)
  (h3 : return_speed = 160) :
  (total_time * outbound_speed * return_speed) / (outbound_speed + return_speed) / outbound_speed * 60 = 192 := by
  sorry

end harriet_trip_time_l3660_366024


namespace fourth_difference_zero_third_nonzero_l3660_366030

def u (n : ℕ) : ℤ := n^3 + n

def Δ' (u : ℕ → ℤ) (n : ℕ) : ℤ := u (n + 1) - u n

def Δ (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0 => u
  | k + 1 => Δ' (Δ k u)

theorem fourth_difference_zero_third_nonzero :
  (∀ n, Δ 4 u n = 0) ∧ (∃ n, Δ 3 u n ≠ 0) :=
sorry

end fourth_difference_zero_third_nonzero_l3660_366030


namespace students_under_three_l3660_366059

/-- Represents the number of students in different age groups in a nursery school -/
structure NurserySchool where
  total : ℕ
  fourAndOlder : ℕ
  underThree : ℕ
  notBetweenThreeAndFour : ℕ

/-- Theorem stating the number of students under three years old in the nursery school -/
theorem students_under_three (school : NurserySchool) 
  (h1 : school.total = 300)
  (h2 : school.fourAndOlder = school.total / 10)
  (h3 : school.notBetweenThreeAndFour = 50)
  (h4 : school.notBetweenThreeAndFour = school.fourAndOlder + school.underThree) :
  school.underThree = 20 := by
  sorry

end students_under_three_l3660_366059


namespace quadratic_form_ratio_l3660_366078

theorem quadratic_form_ratio (j : ℝ) :
  ∃ (c p q : ℝ), 
    (∀ j, 8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ 
    q / p = -151 / 3 := by
  sorry

end quadratic_form_ratio_l3660_366078


namespace ABD_collinear_l3660_366055

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (a b : V)

-- Define the points
variable (A B C D : V)

-- Define the vector relationships
axiom AB_def : B - A = a + 2 • b
axiom BC_def : C - B = -5 • a + 6 • b
axiom CD_def : D - C = 7 • a - 2 • b

-- Theorem to prove
theorem ABD_collinear : ∃ (t : ℝ), D - A = t • (B - A) := by
  sorry

end ABD_collinear_l3660_366055


namespace expand_polynomial_l3660_366061

theorem expand_polynomial (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4*x + 4) = x^4 + 4*x^3 - 16*x - 16 := by
  sorry

end expand_polynomial_l3660_366061


namespace aunt_may_milk_problem_l3660_366097

theorem aunt_may_milk_problem (morning_milk evening_milk sold_milk leftover_milk : ℕ) 
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk - sold_milk + leftover_milk = 148 :=
by sorry

end aunt_may_milk_problem_l3660_366097


namespace opposite_of_negative_fraction_l3660_366056

theorem opposite_of_negative_fraction :
  -(-(4/5 : ℚ)) = 4/5 := by sorry

end opposite_of_negative_fraction_l3660_366056


namespace line_equation_l3660_366063

/-- Circle C with equation x^2 + (y-1)^2 = 5 -/
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

/-- Line l with equation mx - y + 1 - m = 0 -/
def line_l (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

/-- Point P(1,1) -/
def point_P : ℝ × ℝ := (1, 1)

/-- Chord AB of circle C -/
def chord_AB (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  circle_C x₁ y₁ ∧ circle_C x₂ y₂

/-- Point P divides chord AB with ratio AP:PB = 1:2 -/
def divides_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  chord_AB x₁ y₁ x₂ y₂ ∧ 2*(1 - x₁) = x₂ - 1 ∧ 2*(1 - y₁) = y₂ - 1

theorem line_equation (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, line_l m 1 1 ∧ divides_chord x₁ y₁ x₂ y₂) →
  (line_l 1 x y ∨ line_l (-1) x y) :=
sorry

end line_equation_l3660_366063


namespace range_of_a_for_circle_condition_l3660_366025

/-- The range of 'a' for which there exists a point M on the circle (x-a)^2 + (y-a+2)^2 = 1
    such that |MA| = 2|MO|, where A is (0, -3) and O is the origin. -/
theorem range_of_a_for_circle_condition (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - a + 2)^2 = 1 ∧ 
    (x^2 + (y + 3)^2) = 4 * (x^2 + y^2)) ↔ 
  0 ≤ a ∧ a ≤ 3 :=
by sorry


end range_of_a_for_circle_condition_l3660_366025


namespace wang_liang_is_president_l3660_366029

-- Define the students and positions
inductive Student : Type
| ZhangQiang : Student
| LiMing : Student
| WangLiang : Student

inductive Position : Type
| President : Position
| LifeDelegate : Position
| StudyDelegate : Position

-- Define the council as a function from Position to Student
def Council := Position → Student

-- Define the predictions
def PredictionA (c : Council) : Prop :=
  c Position.President = Student.ZhangQiang ∧ c Position.LifeDelegate = Student.LiMing

def PredictionB (c : Council) : Prop :=
  c Position.President = Student.WangLiang ∧ c Position.LifeDelegate = Student.ZhangQiang

def PredictionC (c : Council) : Prop :=
  c Position.President = Student.LiMing ∧ c Position.StudyDelegate = Student.ZhangQiang

-- Define the condition that each prediction is half correct
def HalfCorrectPredictions (c : Council) : Prop :=
  (PredictionA c = true) = (PredictionA c = false) ∧
  (PredictionB c = true) = (PredictionB c = false) ∧
  (PredictionC c = true) = (PredictionC c = false)

-- Theorem statement
theorem wang_liang_is_president :
  ∀ c : Council, HalfCorrectPredictions c → c Position.President = Student.WangLiang :=
by
  sorry

end wang_liang_is_president_l3660_366029


namespace lizard_feature_difference_l3660_366016

/-- Represents a three-eyed lizard with wrinkles and spots -/
structure Lizard :=
  (eyes : ℕ)
  (wrinkle_factor : ℕ)
  (spot_factor : ℕ)

/-- Calculate the total number of features (eyes, wrinkles, and spots) for a lizard -/
def total_features (l : Lizard) : ℕ :=
  l.eyes + (l.eyes * l.wrinkle_factor) + (l.eyes * l.wrinkle_factor * l.spot_factor)

/-- The main theorem about the difference between total features and eyes for two lizards -/
theorem lizard_feature_difference (jan_lizard cousin_lizard : Lizard)
  (h1 : jan_lizard.eyes = 3)
  (h2 : jan_lizard.wrinkle_factor = 3)
  (h3 : jan_lizard.spot_factor = 7)
  (h4 : cousin_lizard.eyes = 3)
  (h5 : cousin_lizard.wrinkle_factor = 2)
  (h6 : cousin_lizard.spot_factor = 5) :
  (total_features jan_lizard + total_features cousin_lizard) - (jan_lizard.eyes + cousin_lizard.eyes) = 102 :=
sorry

end lizard_feature_difference_l3660_366016


namespace simplify_power_expression_l3660_366060

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^5 = 243 * x^20 := by
  sorry

end simplify_power_expression_l3660_366060


namespace solve_x_equation_l3660_366076

theorem solve_x_equation : ∃ x : ℝ, (0.6 * x = x / 3 + 110) ∧ x = 412.5 := by
  sorry

end solve_x_equation_l3660_366076


namespace division_problem_l3660_366032

theorem division_problem (x y : ℕ+) (h1 : x = 7 * y + 3) (h2 : 2 * x = 18 * y + 2) : 
  11 * y - x = 1 := by
  sorry

end division_problem_l3660_366032


namespace halfway_between_one_third_and_one_eighth_l3660_366092

theorem halfway_between_one_third_and_one_eighth :
  (1 / 3 : ℚ) / 2 + (1 / 8 : ℚ) / 2 = 11 / 48 := by
  sorry

end halfway_between_one_third_and_one_eighth_l3660_366092


namespace max_value_sqrt_sum_l3660_366042

theorem max_value_sqrt_sum (x : ℝ) (h : -25 ≤ x ∧ x ≤ 25) :
  Real.sqrt (25 + x) + Real.sqrt (25 - x) ≤ 10 ∧
  (Real.sqrt (25 + x) + Real.sqrt (25 - x) = 10 ↔ x = 0) :=
by sorry

end max_value_sqrt_sum_l3660_366042


namespace election_winner_votes_l3660_366062

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (58 : ℚ) / 100 * total_votes - (42 : ℚ) / 100 * total_votes = 288) :
  ⌊(58 : ℚ) / 100 * total_votes⌋ = 1044 := by
  sorry

end election_winner_votes_l3660_366062


namespace triangle_area_l3660_366048

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

theorem triangle_area : 
  let area := (1/2) * |a.1 * b.2 - a.2 * b.1|
  area = 9/2 := by sorry

end triangle_area_l3660_366048


namespace max_b_cubic_function_max_b_value_l3660_366031

/-- A cubic function f(x) = ax³ + bx + c -/
def cubic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + c

/-- The maximum possible value of b in a cubic function f(x) = ax³ + bx + c
    where 0 ≤ f(x) ≤ 1 for all x in [0, 1] -/
theorem max_b_cubic_function :
  ∃ (b_max : ℝ),
    ∀ (a b c : ℝ),
      (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a b c x ∧ cubic_function a b c x ≤ 1) →
      b ≤ b_max ∧
      ∃ (a' c' : ℝ), ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a' b_max c' x ∧ cubic_function a' b_max c' x ≤ 1 :=
by
  sorry

/-- The maximum possible value of b is 3√3/2 -/
theorem max_b_value : 
  ∃ (b_max : ℝ),
    b_max = 3 * Real.sqrt 3 / 2 ∧
    ∀ (a b c : ℝ),
      (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a b c x ∧ cubic_function a b c x ≤ 1) →
      b ≤ b_max ∧
      ∃ (a' c' : ℝ), ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ cubic_function a' b_max c' x ∧ cubic_function a' b_max c' x ≤ 1 :=
by
  sorry

end max_b_cubic_function_max_b_value_l3660_366031


namespace smallest_fraction_l3660_366015

theorem smallest_fraction (x : ℝ) (h : x = 9) : 
  min (8/x) (min (8/(x+2)) (min (8/(x-2)) (min (x/8) ((x^2+1)/8)))) = 8/(x+2) := by
  sorry

end smallest_fraction_l3660_366015
