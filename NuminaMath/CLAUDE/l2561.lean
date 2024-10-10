import Mathlib

namespace expression_evaluation_l2561_256179

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(3*x)) / (y^(2*y) * x^(3*x)) = (x/y)^(2*y - 3*x) := by
  sorry

end expression_evaluation_l2561_256179


namespace continued_fraction_value_l2561_256127

def continued_fraction (a b c d : ℚ) : ℚ :=
  -1 / (a - 1 / (b - 1 / (c - 1 / d)))

theorem continued_fraction_value : continued_fraction 2 2 2 2 = 4/5 := by
  sorry

end continued_fraction_value_l2561_256127


namespace sequence_sum_equals_321_64_l2561_256194

def sequence_term (n : ℕ) : ℚ := (2^n - 1) / 2^n

def sum_of_terms (n : ℕ) : ℚ := n - 1 + 1 / 2^(n+1)

theorem sequence_sum_equals_321_64 :
  ∃ n : ℕ, sum_of_terms n = 321 / 64 ∧ n = 6 := by sorry

end sequence_sum_equals_321_64_l2561_256194


namespace stratified_sampling_theorem_l2561_256142

/-- Represents a high school with stratified sampling -/
structure HighSchool where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  total_sample : ℕ
  first_year_sample : ℕ
  third_year_sample : ℕ

/-- The total number of students in the high school -/
def total_students (hs : HighSchool) : ℕ :=
  hs.first_year + hs.second_year + hs.third_year

/-- The sampling ratio for the second year -/
def sampling_ratio (hs : HighSchool) : ℚ :=
  (hs.total_sample - hs.first_year_sample - hs.third_year_sample : ℚ) / hs.second_year

theorem stratified_sampling_theorem (hs : HighSchool) 
  (h1 : hs.second_year = 900)
  (h2 : hs.total_sample = 370)
  (h3 : hs.first_year_sample = 120)
  (h4 : hs.third_year_sample = 100)
  (h5 : sampling_ratio hs = 1 / 6) :
  total_students hs = 2220 := by
  sorry

#check stratified_sampling_theorem

end stratified_sampling_theorem_l2561_256142


namespace reflected_light_equation_l2561_256138

/-- Given points A, B, and P in a plane, and a line l passing through P parallel to AB,
    prove that the equation of the reflected light line from B to A via l is 11x + 27y + 74 = 0 -/
theorem reflected_light_equation (A B P : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  A = (8, -6) →
  B = (2, 2) →
  P = (2, -3) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 4*x + 3*y + 1 = 0) →
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ y - P.2 = k * (x - P.1)) →
  (∃ (x₀ y₀ : ℝ), (x₀, y₀) ∈ l ∧ 
    ((y₀ - B.2) / (x₀ - B.1) = -(x₀ - A.1) / (y₀ - A.2))) →
  ∃ (x y : ℝ), 11*x + 27*y + 74 = 0 ↔ 
    (y - A.2) / (x - A.1) = (A.2 - y₀) / (A.1 - x₀) :=
by sorry

end reflected_light_equation_l2561_256138


namespace arrangements_count_l2561_256156

/-- The number of ways to arrange 5 distinct objects in a row, 
    where two specific objects are not allowed to be adjacent -/
def arrangements_with_restriction : ℕ := 72

/-- Theorem stating that the number of arrangements with the given restriction is 72 -/
theorem arrangements_count : arrangements_with_restriction = 72 := by
  sorry

end arrangements_count_l2561_256156


namespace point_on_line_l2561_256161

/-- Given a line y = mx + b where m is the slope and b is the y-intercept,
    if m + b = 3, then the point (1, 3) lies on this line. -/
theorem point_on_line (m b : ℝ) (h : m + b = 3) :
  let f : ℝ → ℝ := fun x ↦ m * x + b
  f 1 = 3 := by
  sorry

end point_on_line_l2561_256161


namespace hyperbola_a_plus_h_l2561_256163

/-- Given a hyperbola with the following properties:
  * Asymptotes: y = 3x + 2 and y = -3x + 8
  * Passes through the point (2, 10)
  * Standard form: (y-k)^2/a^2 - (x-h)^2/b^2 = 1
  * a, b > 0
  Prove that a + h = 6√2 + 1 -/
theorem hyperbola_a_plus_h (a b h k : ℝ) : 
  (∀ x y : ℝ, (y = 3*x + 2 ∨ y = -3*x + 8) → 
    ((y - k)^2 / a^2 - (x - h)^2 / b^2 = 1)) →
  ((10 - k)^2 / a^2 - (2 - h)^2 / b^2 = 1) →
  (a > 0 ∧ b > 0) →
  a + h = 6 * Real.sqrt 2 + 1 := by
  sorry

end hyperbola_a_plus_h_l2561_256163


namespace root_product_equals_27_l2561_256190

theorem root_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end root_product_equals_27_l2561_256190


namespace grandmas_age_l2561_256155

theorem grandmas_age :
  ∀ x : ℕ, (x : ℝ) - (x : ℝ) / 7 = 84 → x = 98 := by
  sorry

end grandmas_age_l2561_256155


namespace bug_position_after_2012_jumps_l2561_256109

/-- Represents the five points on the circle -/
inductive Point
| one
| two
| three
| four
| five

/-- Determines if a point is even -/
def Point.isEven : Point → Bool
  | .two => true
  | .four => true
  | _ => false

/-- Calculates the next point after a jump -/
def nextPoint (p : Point) : Point :=
  match p with
  | .one => .three
  | .two => .five
  | .three => .five
  | .four => .two
  | .five => .two

/-- Calculates the point after n jumps -/
def jumpNTimes (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpNTimes start n)

theorem bug_position_after_2012_jumps :
  jumpNTimes Point.five 2012 = Point.two := by
  sorry


end bug_position_after_2012_jumps_l2561_256109


namespace curly_bracket_calculation_l2561_256189

-- Define the ceiling function for rational numbers
def ceiling (a : ℚ) : ℤ := Int.ceil a

-- Define the curly bracket notation
def curly_bracket (a : ℚ) : ℤ := ceiling a

-- Theorem statement
theorem curly_bracket_calculation :
  (curly_bracket (-6 + 5/6) : ℚ) - 
  (curly_bracket 5 : ℚ) * (curly_bracket (-1 - 3/4) : ℚ) / (curly_bracket (59/10) : ℚ) = -5 := by
  sorry


end curly_bracket_calculation_l2561_256189


namespace perimeter_of_square_III_is_four_l2561_256176

/-- Given three squares I, II, and III, prove that the perimeter of square III is 4 -/
theorem perimeter_of_square_III_is_four :
  ∀ (side_I side_II side_III : ℝ),
  side_I * 4 = 20 →
  side_II * 4 = 16 →
  side_III = side_I - side_II →
  side_III * 4 = 4 := by
sorry

end perimeter_of_square_III_is_four_l2561_256176


namespace total_is_27_l2561_256159

def purchase1 : ℚ := 2.47
def purchase2 : ℚ := 7.51
def purchase3 : ℚ := 11.56
def purchase4 : ℚ := 4.98

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - x.floor < 0.5 then x.floor else x.ceil

def total_rounded : ℤ := 
  round_to_nearest_dollar (purchase1 + purchase2 + purchase3 + purchase4)

theorem total_is_27 : total_rounded = 27 := by
  sorry

end total_is_27_l2561_256159


namespace same_first_last_digit_exists_l2561_256150

-- Define a function to get the first digit of a natural number
def firstDigit (n : ℕ) : ℕ :=
  if n < 10 then n else firstDigit (n / 10)

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ :=
  n % 10

-- Theorem statement
theorem same_first_last_digit_exists (n : ℕ) (h : n > 0 ∧ n % 10 ≠ 0) :
  ∃ k : ℕ, k > 0 ∧ firstDigit (n^k) = lastDigit (n^k) :=
sorry

end same_first_last_digit_exists_l2561_256150


namespace sons_age_l2561_256126

theorem sons_age (father son : ℕ) : 
  (father + 6 + (son + 6) = 68) →  -- After 6 years, sum of ages is 68
  (father = 6 * son) →             -- Father's age is 6 times son's age
  son = 8 :=                       -- Son's age is 8
by sorry

end sons_age_l2561_256126


namespace parallel_lines_slope_l2561_256108

/-- Given two lines l₁ and l₂ in the real plane, prove that if l₁ with equation x + 2y - 1 = 0 
    is parallel to l₂ with equation mx - y = 0, then m = -1/2. -/
theorem parallel_lines_slope (m : ℝ) : 
  (∀ x y : ℝ, x + 2*y - 1 = 0 ↔ m*x - y = 0) → m = -1/2 := by
  sorry

end parallel_lines_slope_l2561_256108


namespace bike_route_length_l2561_256162

/-- Represents a rectangular bike route in a park -/
structure BikeRoute where
  upper_horizontal : List Float
  left_vertical : List Float

/-- Calculates the total length of the bike route -/
def total_length (route : BikeRoute) : Float :=
  2 * (route.upper_horizontal.sum + route.left_vertical.sum)

/-- Theorem stating the total length of the specific bike route -/
theorem bike_route_length :
  let route : BikeRoute := {
    upper_horizontal := [4, 7, 2],
    left_vertical := [6, 7]
  }
  total_length route = 52 := by sorry

end bike_route_length_l2561_256162


namespace b_fourth_plus_inverse_l2561_256103

theorem b_fourth_plus_inverse (b : ℝ) (h : (b + 1/b)^2 = 5) : b^4 + 1/b^4 = 7 := by
  sorry

end b_fourth_plus_inverse_l2561_256103


namespace smallest_solution_absolute_value_equation_l2561_256115

theorem smallest_solution_absolute_value_equation :
  ∃ (x : ℝ), x * |x| = 2 * x + 1 ∧ 
  ∀ (y : ℝ), y * |y| = 2 * y + 1 → x ≤ y :=
by sorry

end smallest_solution_absolute_value_equation_l2561_256115


namespace homogeneous_polynomial_on_circle_l2561_256113

-- Define a homogeneous polynomial
def IsHomogeneous (P : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (c x y : ℝ), P (c * x) (c * y) = c^n * P x y

-- Define the theorem
theorem homogeneous_polynomial_on_circle (P : ℝ → ℝ → ℝ) (n : ℕ) :
  IsHomogeneous P n →
  (∀ t : ℝ, P (Real.sin t) (Real.cos t) = 1) →
  n > 0 →
  ∃ k : ℕ, k > 0 ∧ ∀ x y : ℝ, P x y = (x^2 + y^2)^k :=
by sorry

end homogeneous_polynomial_on_circle_l2561_256113


namespace gcd_problem_l2561_256168

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 1193 * k ∧ Odd k) :
  Int.gcd (2 * b^2 + 31 * b + 73) (b + 17) = 1 := by
  sorry

end gcd_problem_l2561_256168


namespace ellipse_equation_and_intersection_range_l2561_256130

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the line x - y + 2√2 = 0
def Line := {p : ℝ × ℝ | p.1 - p.2 + 2 * Real.sqrt 2 = 0}

theorem ellipse_equation_and_intersection_range :
  ∃ (a b c : ℝ),
    -- Conditions
    (0, -1) ∈ Ellipse a b ∧  -- One vertex at (0, -1)
    (c, 0) ∈ Ellipse a b ∧   -- Right focus on x-axis
    (∀ (x y : ℝ), (x, y) ∈ Line → ((x - c)^2 + y^2).sqrt = 3) ∧  -- Distance from right focus to line is 3
    -- Conclusions
    (Ellipse a b = Ellipse (Real.sqrt 3) 1) ∧  -- Equation of ellipse
    (∀ m : ℝ, (∃ (p q : ℝ × ℝ), p ≠ q ∧ p ∈ Ellipse (Real.sqrt 3) 1 ∧ q ∈ Ellipse (Real.sqrt 3) 1 ∧
                p.2 = p.1 + m ∧ q.2 = q.1 + m) ↔ -2 < m ∧ m < 2)  -- Intersection range
    := by sorry

end ellipse_equation_and_intersection_range_l2561_256130


namespace min_value_theorem_l2561_256123

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = 1 → (a + 1) * (b + 1) / (a * b) ≥ (x + 1) * (y + 1) / (x * y)) ∧
  (x + 1) * (y + 1) / (x * y) = 8 + 4 * Real.sqrt 3 :=
by sorry

end min_value_theorem_l2561_256123


namespace negative_cube_squared_l2561_256199

theorem negative_cube_squared (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end negative_cube_squared_l2561_256199


namespace vacation_cost_l2561_256171

theorem vacation_cost (cost : ℝ) : 
  (cost / 3 - cost / 4 = 30) → cost = 360 := by
sorry

end vacation_cost_l2561_256171


namespace fraction_equality_l2561_256157

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 4 * y) = 3) : 
  (2 * x + 4 * y) / (4 * x - 2 * y) = 9 / 13 := by
  sorry

end fraction_equality_l2561_256157


namespace tank_emptied_in_three_minutes_l2561_256110

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initial_level : ℚ
  fill_rate : ℚ
  empty_rate : ℚ

/-- Calculates the time to empty or fill the tank when both pipes are open -/
def time_to_empty_or_fill (tank : WaterTank) : ℚ :=
  tank.initial_level / (tank.empty_rate - tank.fill_rate)

/-- Theorem stating that the tank will be emptied in 3 minutes under given conditions -/
theorem tank_emptied_in_three_minutes :
  let tank : WaterTank := {
    initial_level := 1/5,
    fill_rate := 1/10,
    empty_rate := 1/6
  }
  time_to_empty_or_fill tank = 3 := by
  sorry

#eval time_to_empty_or_fill {
  initial_level := 1/5,
  fill_rate := 1/10,
  empty_rate := 1/6
}

end tank_emptied_in_three_minutes_l2561_256110


namespace average_parking_cost_for_9_hours_l2561_256121

/-- Calculates the average cost per hour for parking given the following conditions:
  * Base cost for up to 2 hours
  * Additional cost per hour after 2 hours
  * Total number of hours parked
-/
def averageParkingCost (baseCost hourlyRate : ℚ) (totalHours : ℕ) : ℚ :=
  let totalCost := baseCost + hourlyRate * (totalHours - 2)
  totalCost / totalHours

/-- Theorem stating that the average parking cost for 9 hours is $3.03 -/
theorem average_parking_cost_for_9_hours :
  averageParkingCost 15 (7/4) 9 = 303/100 := by
  sorry

#eval averageParkingCost 15 (7/4) 9

end average_parking_cost_for_9_hours_l2561_256121


namespace worker_schedule_theorem_l2561_256119

/-- Represents a worker's daily schedule and pay --/
structure WorkerSchedule where
  baseHours : ℝ
  basePay : ℝ
  bonusPay : ℝ
  bonusHours : ℝ
  bonusHourlyRate : ℝ

/-- Theorem stating the conditions and conclusion about the worker's schedule --/
theorem worker_schedule_theorem (w : WorkerSchedule) 
  (h1 : w.basePay = 80)
  (h2 : w.bonusPay = 20)
  (h3 : w.bonusHours = 2)
  (h4 : w.bonusHourlyRate = 10)
  (h5 : w.bonusHourlyRate * (w.baseHours + w.bonusHours) = w.basePay + w.bonusPay) :
  w.baseHours = 8 := by
  sorry

#check worker_schedule_theorem

end worker_schedule_theorem_l2561_256119


namespace value_of_a_l2561_256141

theorem value_of_a (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 / b = 2) (h2 : b^2 / c = 4) (h3 : c^2 / a = 6) :
  a = (384 : ℝ)^(1/7) := by
sorry

end value_of_a_l2561_256141


namespace beatrice_book_count_l2561_256148

/-- The cost of each of the first 5 books -/
def initial_book_cost : ℕ := 20

/-- The number of books at the initial price -/
def initial_book_count : ℕ := 5

/-- The discount applied to each book after the initial count -/
def discount : ℕ := 2

/-- The total amount Beatrice paid -/
def total_paid : ℕ := 370

/-- Function to calculate the total cost for a given number of books -/
def total_cost (num_books : ℕ) : ℕ :=
  if num_books ≤ initial_book_count then
    num_books * initial_book_cost
  else
    initial_book_count * initial_book_cost +
    (num_books - initial_book_count) * (initial_book_cost - discount)

/-- Theorem stating that Beatrice bought 20 books -/
theorem beatrice_book_count : ∃ (n : ℕ), n = 20 ∧ total_cost n = total_paid := by
  sorry

end beatrice_book_count_l2561_256148


namespace horse_cost_problem_l2561_256105

theorem horse_cost_problem (selling_price : ℕ) (cost : ℕ) : 
  selling_price = 56 →
  selling_price = cost + (cost * cost) / 100 →
  cost = 40 := by
sorry

end horse_cost_problem_l2561_256105


namespace division_multiplication_equality_l2561_256196

theorem division_multiplication_equality : (0.24 / 0.006) * 2 = 80 := by
  sorry

end division_multiplication_equality_l2561_256196


namespace y_worked_days_proof_l2561_256178

/-- The number of days x needs to finish the entire work -/
def x_total_days : ℝ := 24

/-- The number of days y needs to finish the entire work -/
def y_total_days : ℝ := 16

/-- The number of days x needs to finish the remaining work after y leaves -/
def x_remaining_days : ℝ := 9

/-- The number of days y worked before leaving the job -/
def y_worked_days : ℝ := 10

theorem y_worked_days_proof :
  y_worked_days * (1 / y_total_days) + x_remaining_days * (1 / x_total_days) = 1 := by
  sorry

end y_worked_days_proof_l2561_256178


namespace total_amount_pens_pencils_l2561_256154

/-- The total amount spent on pens and pencils -/
def total_amount (num_pens : ℕ) (num_pencils : ℕ) (price_pen : ℚ) (price_pencil : ℚ) : ℚ :=
  num_pens * price_pen + num_pencils * price_pencil

/-- Theorem stating the total amount spent on pens and pencils -/
theorem total_amount_pens_pencils :
  total_amount 30 75 12 2 = 510 := by
  sorry

#eval total_amount 30 75 12 2

end total_amount_pens_pencils_l2561_256154


namespace notebook_word_count_l2561_256184

theorem notebook_word_count (total_pages : Nat) (max_words_per_page : Nat) 
  (h1 : total_pages = 150)
  (h2 : max_words_per_page = 90)
  (h3 : ∃ (words_per_page : Nat), words_per_page ≤ max_words_per_page ∧ 
        (total_pages * words_per_page) % 221 = 210) :
  ∃ (words_per_page : Nat), words_per_page = 90 ∧ 
    words_per_page ≤ max_words_per_page ∧ 
    (total_pages * words_per_page) % 221 = 210 := by
  sorry

end notebook_word_count_l2561_256184


namespace triangle_area_l2561_256111

/-- The area of a triangle with sides a = 4, b = 5, and angle C = 60° is 5√3 -/
theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 4) (h2 : b = 5) (h3 : C = π / 3) :
  (1 / 2) * a * b * Real.sin C = 5 * Real.sqrt 3 := by
  sorry

end triangle_area_l2561_256111


namespace sum_of_even_integers_102_to_200_l2561_256106

theorem sum_of_even_integers_102_to_200 :
  let first_term : ℕ := 102
  let last_term : ℕ := 200
  let num_terms : ℕ := 50
  (num_terms : ℚ) / 2 * (first_term + last_term) = 7550 := by
  sorry

end sum_of_even_integers_102_to_200_l2561_256106


namespace complement_of_40_degree_angle_l2561_256166

/-- Given an angle A of 40 degrees, its complement is 50 degrees. -/
theorem complement_of_40_degree_angle (A : ℝ) : 
  A = 40 → (90 - A) = 50 := by
  sorry

end complement_of_40_degree_angle_l2561_256166


namespace smallest_S_for_equal_probability_l2561_256188

-- Define the number of sides on a standard die
def standardDieSides : ℕ := 6

-- Define the target sum
def targetSum : ℕ := 2000

-- Define the function to calculate the minimum number of dice needed to reach the target sum
def minDiceNeeded (target : ℕ) (sides : ℕ) : ℕ :=
  (target + sides - 1) / sides

-- Define the function to calculate S given n dice
def calculateS (n : ℕ) (target : ℕ) : ℕ :=
  7 * n - target

-- Theorem statement
theorem smallest_S_for_equal_probability :
  let n := minDiceNeeded targetSum standardDieSides
  calculateS n targetSum = 338 := by sorry

end smallest_S_for_equal_probability_l2561_256188


namespace agrey_caught_more_l2561_256151

def fishing_problem (leo_fish agrey_fish total_fish : ℕ) : Prop :=
  leo_fish + agrey_fish = total_fish ∧ agrey_fish > leo_fish

theorem agrey_caught_more (leo_fish total_fish : ℕ) 
  (h : fishing_problem leo_fish (total_fish - leo_fish) total_fish) 
  (h_leo : leo_fish = 40) 
  (h_total : total_fish = 100) : 
  (total_fish - leo_fish) - leo_fish = 20 := by
  sorry

end agrey_caught_more_l2561_256151


namespace real_part_of_complex_square_l2561_256143

theorem real_part_of_complex_square : Complex.re ((1 + 2 * Complex.I) ^ 2) = -3 := by
  sorry

end real_part_of_complex_square_l2561_256143


namespace distance_traveled_l2561_256147

/-- Given a speed of 65 km/hr and a time of 3 hr, the distance traveled is 195 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 65) (h2 : time = 3) :
  speed * time = 195 :=
by sorry

end distance_traveled_l2561_256147


namespace sum_of_even_is_even_l2561_256164

theorem sum_of_even_is_even (a b : ℤ) (ha : Even a) (hb : Even b) : Even (a + b) := by
  sorry

end sum_of_even_is_even_l2561_256164


namespace circle_square_intersection_l2561_256153

theorem circle_square_intersection (r : ℝ) (s : ℝ) (x : ℝ) :
  r = 2 →
  s = 2 →
  (π * r^2 - (s^2 - (π * r^2 - 2 * r * x + x^2))) = 2 →
  x = π / 3 + Real.sqrt 3 / 2 - 1 := by
  sorry

end circle_square_intersection_l2561_256153


namespace multiply_and_simplify_l2561_256181

theorem multiply_and_simplify (x : ℝ) : (x^4 + 16*x^2 + 256) * (x^2 - 16) = x^4 + 32*x^2 := by
  sorry

end multiply_and_simplify_l2561_256181


namespace m_range_l2561_256120

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being decreasing on [-1,1]
def is_decreasing_on (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 1 → f x > f y

-- State the theorem
theorem m_range (f : ℝ → ℝ) (m : ℝ) 
  (h1 : is_decreasing_on f) 
  (h2 : f (m - 1) > f (2*m - 1)) : 
  0 < m ∧ m ≤ 1 := by sorry

end m_range_l2561_256120


namespace checkers_rectangle_exists_l2561_256195

/-- Represents the color of a checker -/
inductive Color
| White
| Black

/-- Represents a 3x7 grid of checkers -/
def CheckerGrid := Fin 3 → Fin 7 → Color

/-- Checks if four positions form a rectangle in the grid -/
def IsRectangle (a b c d : Fin 3 × Fin 7) : Prop :=
  (a.1 = b.1 ∧ c.1 = d.1 ∧ a.2 ≠ c.2) ∨
  (a.1 = c.1 ∧ b.1 = d.1 ∧ a.2 ≠ b.2)

/-- The main theorem -/
theorem checkers_rectangle_exists (grid : CheckerGrid) :
  ∃ (color : Color) (a b c d : Fin 3 × Fin 7),
    IsRectangle a b c d ∧
    grid a.1 a.2 = color ∧
    grid b.1 b.2 = color ∧
    grid c.1 c.2 = color ∧
    grid d.1 d.2 = color :=
sorry

end checkers_rectangle_exists_l2561_256195


namespace didi_fundraising_price_per_slice_l2561_256100

/-- Proves that the price per slice is $1 given the conditions of Didi's fundraising event --/
theorem didi_fundraising_price_per_slice :
  ∀ (price_per_slice : ℚ),
    (10 : ℕ) * (8 : ℕ) * price_per_slice +  -- Revenue from slice sales
    (10 : ℕ) * (8 : ℕ) * (1/2 : ℚ) +        -- Donation from first business owner
    (10 : ℕ) * (8 : ℕ) * (1/4 : ℚ) = 140    -- Donation from second business owner
    → price_per_slice = 1 := by
  sorry

end didi_fundraising_price_per_slice_l2561_256100


namespace max_value_of_expression_l2561_256180

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  let S := (a^2 - a*b + b^2) * (b^2 - b*c + c^2) * (c^2 - c*a + a^2)
  ∃ (max_value : ℝ), max_value = 12 ∧ S ≤ max_value :=
sorry

end max_value_of_expression_l2561_256180


namespace fraction_equality_l2561_256112

theorem fraction_equality (x : ℝ) (f : ℝ) (h1 : x > 0) (h2 : x = 0.4166666666666667) 
  (h3 : f * x = (25/216) * (1/x)) : f = 2/3 := by
  sorry

end fraction_equality_l2561_256112


namespace remainder_sum_mod_60_l2561_256135

theorem remainder_sum_mod_60 (c d : ℤ) 
  (h1 : c % 120 = 114)
  (h2 : d % 180 = 174) : 
  (c + d) % 60 = 48 := by
sorry

end remainder_sum_mod_60_l2561_256135


namespace three_distinct_roots_l2561_256145

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem three_distinct_roots 
  (a b c x₁ x₂ : ℝ) 
  (h1 : ∃ x₁ x₂, x₁ ≠ x₂ ∧ (3*x₁^2 + 2*a*x₁ + b = 0) ∧ (3*x₂^2 + 2*a*x₂ + b = 0)) 
  (h2 : f a b c x₁ = x₁) 
  (h3 : x₁ < x₂) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, 3*(f a b c x)^2 + 2*a*(f a b c x) + b = 0 :=
sorry

end three_distinct_roots_l2561_256145


namespace rachels_winter_clothing_l2561_256173

theorem rachels_winter_clothing (boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) 
  (h1 : boxes = 7)
  (h2 : scarves_per_box = 3)
  (h3 : mittens_per_box = 4) :
  boxes * (scarves_per_box + mittens_per_box) = 49 :=
by sorry

end rachels_winter_clothing_l2561_256173


namespace continuous_function_property_P_l2561_256160

open Function Set Real

theorem continuous_function_property_P 
  (f : ℝ → ℝ) 
  (hf_cont : Continuous f) 
  (hf_dom : ∀ x, x ∈ (Set.Ioc 0 1) → f x ≠ 0) 
  (hf_eq : f 0 = f 1) :
  ∀ k : ℕ, k ≥ 2 → ∃ x₀ ∈ Set.Icc 0 (1 - 1/k), f x₀ = f (x₀ + 1/k) :=
sorry

end continuous_function_property_P_l2561_256160


namespace parallel_vectors_k_l2561_256187

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![0, 1]
def vector_c (k : ℝ) : Fin 2 → ℝ := ![-2, k]

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, ∀ i, v i = t * w i

theorem parallel_vectors_k (k : ℝ) :
  parallel (λ i => vector_a i + 2 * vector_b i) (vector_c k) → k = -8 := by
  sorry

end parallel_vectors_k_l2561_256187


namespace function_properties_l2561_256131

open Function

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivatives of f and g
variable (f' g' : ℝ → ℝ)

-- State the conditions
variable (h1 : ∀ x, HasDerivAt f (f' x) x)
variable (h2 : ∀ x, HasDerivAt g (g' x) x)
variable (h3 : ∀ x, f x = g ((x + 1) / 2) + x)
variable (h4 : Even f)
variable (h5 : Odd (fun x ↦ g' (x + 1)))

-- State the theorem
theorem function_properties :
  f' 1 = 1 ∧ g' (3/2) = 2 ∧ g' 2 = 4 := by
  sorry

end function_properties_l2561_256131


namespace total_children_l2561_256175

/-- The number of happy children -/
def happy_children : ℕ := 30

/-- The number of sad children -/
def sad_children : ℕ := 10

/-- The number of children who are neither happy nor sad -/
def neutral_children : ℕ := 20

/-- The number of boys -/
def boys : ℕ := 17

/-- The number of girls -/
def girls : ℕ := 43

/-- The number of happy boys -/
def happy_boys : ℕ := 6

/-- The number of sad girls -/
def sad_girls : ℕ := 4

/-- The number of boys who are neither happy nor sad -/
def neutral_boys : ℕ := 5

/-- Theorem stating that the total number of children is 60 -/
theorem total_children : boys + girls = 60 := by
  sorry

end total_children_l2561_256175


namespace factors_of_12_correct_ratio_exists_in_factors_l2561_256191

def is_factor (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

def factors_of_12 : Set ℕ := {1, 2, 3, 4, 6, 12}

theorem factors_of_12_correct :
  ∀ n : ℕ, n ∈ factors_of_12 ↔ is_factor 12 n := by sorry

theorem ratio_exists_in_factors :
  ∃ a b c d : ℕ, a ∈ factors_of_12 ∧ b ∈ factors_of_12 ∧ c ∈ factors_of_12 ∧ d ∈ factors_of_12 ∧
  a * d = b * c ∧ a ≠ 0 ∧ b ≠ 0 := by sorry

end factors_of_12_correct_ratio_exists_in_factors_l2561_256191


namespace rebecca_earrings_l2561_256165

theorem rebecca_earrings (magnets : ℕ) : 
  magnets > 0 → 
  (4 * (3 * (magnets / 2))) = 24 → 
  magnets = 4 :=
by
  sorry

end rebecca_earrings_l2561_256165


namespace triangle_trigonometric_identity_l2561_256167

-- Define the triangle PQR
def Triangle (P Q R : ℝ) : Prop := 
  ∃ (pq pr qr : ℝ), pq = 7 ∧ pr = 8 ∧ qr = 5 ∧ 
  pq + pr > qr ∧ pq + qr > pr ∧ pr + qr > pq

-- State the theorem
theorem triangle_trigonometric_identity (P Q R : ℝ) 
  (h : Triangle P Q R) : 
  (Real.cos ((P - Q) / 2) / Real.sin (R / 2)) - 
  (Real.sin ((P - Q) / 2) / Real.cos (R / 2)) = 7 / 4 := by
  sorry

end triangle_trigonometric_identity_l2561_256167


namespace difference_of_ones_and_zeros_237_l2561_256185

def base_2_representation (n : Nat) : List Nat :=
  sorry

def count_zeros (l : List Nat) : Nat :=
  sorry

def count_ones (l : List Nat) : Nat :=
  sorry

theorem difference_of_ones_and_zeros_237 :
  let binary_237 := base_2_representation 237
  let x := count_zeros binary_237
  let y := count_ones binary_237
  y - x = 6 := by sorry

end difference_of_ones_and_zeros_237_l2561_256185


namespace reyansh_farm_cows_l2561_256116

/-- Represents the number of cows on Mr. Reyansh's farm -/
def num_cows : ℕ := sorry

/-- Represents the daily water consumption of one cow in liters -/
def cow_water_daily : ℕ := 80

/-- Represents the number of sheep on Mr. Reyansh's farm -/
def num_sheep : ℕ := 10 * num_cows

/-- Represents the daily water consumption of one sheep in liters -/
def sheep_water_daily : ℕ := cow_water_daily / 4

/-- Represents the total water consumption for all animals in a week in liters -/
def total_water_weekly : ℕ := 78400

/-- Theorem stating that the number of cows on Mr. Reyansh's farm is 40 -/
theorem reyansh_farm_cows :
  num_cows = 40 :=
by sorry

end reyansh_farm_cows_l2561_256116


namespace unique_numbers_problem_l2561_256182

theorem unique_numbers_problem (a b : ℕ) : 
  a ≠ b → 
  a > 11 → 
  b > 11 → 
  (∃ (s : ℕ), s = a + b) → 
  (a % 2 = 0 ∨ b % 2 = 0) →
  (∀ (x y : ℕ), x ≠ y → x > 11 → y > 11 → x + y = a + b → 
    (x % 2 = 0 ∨ y % 2 = 0) → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) →
  ((a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12)) :=
by sorry

end unique_numbers_problem_l2561_256182


namespace base_5_sum_l2561_256104

/-- Represents a digit in base 5 -/
def Base5Digit := { n : ℕ // n > 0 ∧ n < 5 }

/-- Converts a three-digit number in base 5 to its decimal representation -/
def toDecimal (a b c : Base5Digit) : ℕ := 25 * a.val + 5 * b.val + c.val

theorem base_5_sum (A B C : Base5Digit) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : toDecimal A B C + toDecimal B C A + toDecimal C A B = 25 * 31 * A.val) :
  B.val + C.val = 4 :=
sorry

end base_5_sum_l2561_256104


namespace no_integer_solution_l2561_256169

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 23 * x^2 - 92 * y^2 = 3128 := by
  sorry

end no_integer_solution_l2561_256169


namespace ellipse_constant_product_l2561_256134

def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

def line_through_point (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

def dot_product (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * x2 + y1 * y2

theorem ellipse_constant_product :
  ∀ k : ℝ,
  ∃ x1 y1 x2 y2 : ℝ,
  ellipse x1 y1 ∧ ellipse x2 y2 ∧
  line_through_point k 1 x1 y1 ∧
  line_through_point k 1 x2 y2 ∧
  x1 ≠ x2 →
  dot_product (17/8 - x1) (-y1) (17/8 - x2) (-y2) = 33/64 :=
sorry

end ellipse_constant_product_l2561_256134


namespace meeting_point_difference_l2561_256102

/-- The distance between points R and S in miles -/
def total_distance : ℕ := 80

/-- The constant speed of the man starting from R in miles per hour -/
def speed_R : ℕ := 5

/-- The initial speed of the man starting from S in miles per hour -/
def initial_speed_S : ℕ := 4

/-- The hourly increase in speed for the man starting from S in miles per hour -/
def speed_increase_S : ℕ := 1

/-- The number of hours it takes for the men to meet -/
def meeting_time : ℕ := 8

/-- The distance traveled by the man starting from R -/
def distance_R : ℕ := speed_R * meeting_time

/-- The distance traveled by the man starting from S -/
def distance_S : ℕ := initial_speed_S * meeting_time + (meeting_time - 1) * meeting_time / 2

/-- The difference in distances traveled by the two men -/
def x : ℤ := distance_S - distance_R

theorem meeting_point_difference : x = 20 := by
  sorry

end meeting_point_difference_l2561_256102


namespace simplify_first_expression_simplify_second_expression_simplify_third_expression_l2561_256114

-- First expression
theorem simplify_first_expression (a b : ℝ) (h : (a - b)^2 + a*b ≠ 0) :
  (a^3 + b^3) / ((a - b)^2 + a*b) = a + b := by sorry

-- Second expression
theorem simplify_second_expression (x a : ℝ) (h : x^2 - 4*a^2 ≠ 0) :
  (x^2 - 4*a*x + 4*a^2) / (x^2 - 4*a^2) = (x - 2*a) / (x + 2*a) := by sorry

-- Third expression
theorem simplify_third_expression (x y : ℝ) (h : x*y - 2*x ≠ 0) :
  (x*y - 2*x - 3*y + 6) / (x*y - 2*x) = (x - 3) / x := by sorry

end simplify_first_expression_simplify_second_expression_simplify_third_expression_l2561_256114


namespace third_term_zero_l2561_256193

/-- 
Given two geometric progressions with first terms u₁ and v₁ and common ratios q and p respectively,
if the sum of their first terms is 0 and the sum of their second terms is 0,
then the sum of their third terms is also 0.
-/
theorem third_term_zero (u₁ v₁ q p : ℝ) 
  (h1 : u₁ + v₁ = 0) 
  (h2 : u₁ * q + v₁ * p = 0) : 
  u₁ * q^2 + v₁ * p^2 = 0 := by
  sorry

end third_term_zero_l2561_256193


namespace equation_solution_l2561_256144

theorem equation_solution :
  ∃ x : ℚ, 
    (((x / 128 + (1 + 2 / 7)) / (5 - 4 * (2 / 21) * 0.75)) / 
    ((1 / 3 + 5 / 7 * 1.4) / ((4 - 2 * (2 / 3)) * 3)) = 4.5) ∧ 
    x = 1440 / 7 := by
  sorry

end equation_solution_l2561_256144


namespace ad_duration_l2561_256152

theorem ad_duration (num_ads : ℕ) (cost_per_minute : ℕ) (total_cost : ℕ) 
  (h1 : num_ads = 5)
  (h2 : cost_per_minute = 4000)
  (h3 : total_cost = 60000) :
  (total_cost / cost_per_minute) / num_ads = 3 := by
  sorry

end ad_duration_l2561_256152


namespace prob_at_least_one_odd_prob_outside_or_on_circle_l2561_256129

-- Define the sample space for a single die roll
def Die : Type := Fin 6

-- Define the sample space for two die rolls
def TwoRolls : Type := Die × Die

-- Define the probability measure
def P : Set TwoRolls → ℚ := sorry

-- Define the event of at least one odd number
def AtLeastOneOdd : Set TwoRolls := sorry

-- Define the event of the point lying outside or on the circle
def OutsideOrOnCircle : Set TwoRolls := sorry

-- Theorem for the first probability
theorem prob_at_least_one_odd : P AtLeastOneOdd = 3/4 := sorry

-- Theorem for the second probability
theorem prob_outside_or_on_circle : P OutsideOrOnCircle = 7/9 := sorry

end prob_at_least_one_odd_prob_outside_or_on_circle_l2561_256129


namespace equal_area_rectangles_width_l2561_256118

/-- Given two rectangles with equal area, where one rectangle has dimensions 4 inches by 30 inches,
    and the other has a length of 5 inches, prove that the width of the second rectangle is 24 inches. -/
theorem equal_area_rectangles_width (area : ℝ) (length1 width1 length2 width2 : ℝ) : 
  area = length1 * width1 → -- Area of the first rectangle
  area = length2 * width2 → -- Area of the second rectangle
  length1 = 4 →             -- Length of the first rectangle
  width1 = 30 →             -- Width of the first rectangle
  length2 = 5 →             -- Length of the second rectangle
  width2 = 24 :=            -- Width of the second rectangle (to be proved)
by sorry

end equal_area_rectangles_width_l2561_256118


namespace factorization_equality_l2561_256149

theorem factorization_equality (x : ℝ) : 
  x^2 * (x + 3) + 2 * (x + 3) - 5 * (x + 3) = (x + 3) * (x^2 - 3) := by
  sorry

end factorization_equality_l2561_256149


namespace trihedral_dihedral_planar_equality_l2561_256101

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real

/-- Represents a dihedral angle -/
def DihedralAngle : Type := Real

/-- 
Given a trihedral angle, there exists a planar angle equal to 
the dihedral angle opposite to one of its plane angles.
-/
theorem trihedral_dihedral_planar_equality 
  (t : TrihedralAngle) : 
  ∃ (planar_angle : Real) (dihedral : DihedralAngle), 
    planar_angle = dihedral := by
  sorry


end trihedral_dihedral_planar_equality_l2561_256101


namespace candy_mixture_price_prove_candy_mixture_price_l2561_256158

/-- Given two types of candies with equal amounts, priced at 2 and 3 rubles per kilogram respectively,
    the price of their mixture is 2.4 rubles per kilogram. -/
theorem candy_mixture_price : ℝ → Prop :=
  fun (s : ℝ) ↦
    let candy1_weight := s / 2
    let candy2_weight := s / 3
    let total_weight := candy1_weight + candy2_weight
    let total_cost := 2 * candy1_weight + 3 * candy2_weight
    let mixture_price := total_cost / total_weight
    mixture_price = 2.4

/-- Proof of the candy mixture price theorem -/
theorem prove_candy_mixture_price : ∃ s : ℝ, candy_mixture_price s := by
  sorry

end candy_mixture_price_prove_candy_mixture_price_l2561_256158


namespace elective_course_arrangements_l2561_256136

def slots : ℕ := 6
def courses : ℕ := 3

theorem elective_course_arrangements : 
  (slots.factorial) / ((slots - courses).factorial) = 120 := by
  sorry

end elective_course_arrangements_l2561_256136


namespace unbroken_seashells_l2561_256125

/-- Given that Tom found 7 seashells in total and 4 of them were broken,
    prove that the number of unbroken seashells is 3. -/
theorem unbroken_seashells (total : ℕ) (broken : ℕ) 
  (h1 : total = 7) 
  (h2 : broken = 4) : 
  total - broken = 3 := by
  sorry

end unbroken_seashells_l2561_256125


namespace coordinate_sum_of_A_l2561_256140

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the coordinate plane -/
structure Line where
  m : ℝ
  b : ℝ

/-- The theorem statement -/
theorem coordinate_sum_of_A (A B C : Point) (l₁ l₂ l₃ : Line) (a b : ℝ) :
  B.y = 0 →  -- B is on Ox axis
  C.x = 0 →  -- C is on Oy axis
  (l₁.m = a ∧ l₁.b = 4) ∨ (l₁.m = 2 ∧ l₁.b = b) ∨ (l₁.m = a/2 ∧ l₁.b = 8) →  -- l₁ is one of the given lines
  (l₂.m = a ∧ l₂.b = 4) ∨ (l₂.m = 2 ∧ l₂.b = b) ∨ (l₂.m = a/2 ∧ l₂.b = 8) →  -- l₂ is one of the given lines
  (l₃.m = a ∧ l₃.b = 4) ∨ (l₃.m = 2 ∧ l₃.b = b) ∨ (l₃.m = a/2 ∧ l₃.b = 8) →  -- l₃ is one of the given lines
  l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃ →  -- All lines are different
  (A.y = l₁.m * A.x + l₁.b) ∧ (B.y = l₁.m * B.x + l₁.b) →  -- A and B are on l₁
  (B.y = l₂.m * B.x + l₂.b) ∧ (C.y = l₂.m * C.x + l₂.b) →  -- B and C are on l₂
  (A.y = l₃.m * A.x + l₃.b) ∧ (C.y = l₃.m * C.x + l₃.b) →  -- A and C are on l₃
  A.x + A.y = 13 ∨ A.x + A.y = 20 := by
sorry

end coordinate_sum_of_A_l2561_256140


namespace probability_three_non_defective_pencils_l2561_256128

theorem probability_three_non_defective_pencils :
  let total_pencils : ℕ := 10
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations : ℕ := Nat.choose total_pencils selected_pencils
  let non_defective_combinations : ℕ := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 7 / 15 :=
by sorry

end probability_three_non_defective_pencils_l2561_256128


namespace inequality_one_inequality_two_inequality_three_l2561_256183

-- 1. 3x + 1 ≥ -2 if and only if x ≥ -1
theorem inequality_one (x : ℝ) : 3 * x + 1 ≥ -2 ↔ x ≥ -1 := by sorry

-- 2. (y ≥ 1 and -2y ≥ -2) if and only if y = 1
theorem inequality_two (y : ℝ) : (y ≥ 1 ∧ -2 * y ≥ -2) ↔ y = 1 := by sorry

-- 3. y²(x² + 1) - 1 ≤ x² if and only if -1 ≤ y ≤ 1
theorem inequality_three (x y : ℝ) : y^2 * (x^2 + 1) - 1 ≤ x^2 ↔ -1 ≤ y ∧ y ≤ 1 := by sorry

end inequality_one_inequality_two_inequality_three_l2561_256183


namespace julie_work_hours_l2561_256172

/-- Given Julie's work conditions, prove she needs to work 18 hours per week during school year --/
theorem julie_work_hours : 
  ∀ (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
    (school_weeks : ℕ) (school_earnings : ℕ),
  summer_weeks = 10 →
  summer_hours_per_week = 60 →
  summer_earnings = 7500 →
  school_weeks = 40 →
  school_earnings = 9000 →
  (school_earnings * summer_weeks * summer_hours_per_week) / 
    (summer_earnings * school_weeks) = 18 := by
sorry

end julie_work_hours_l2561_256172


namespace company_valuation_l2561_256107

theorem company_valuation (P A B : ℝ) 
  (h1 : P = 1.5 * A) 
  (h2 : P = 2 * B) : 
  P / (A + B) = 6 / 7 := by
  sorry

end company_valuation_l2561_256107


namespace smallest_multiplier_for_cube_l2561_256198

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_multiplier_for_cube (a : ℕ) : 
  (a > 0 ∧ is_cube (5880 * a) ∧ ∀ b : ℕ, 0 < b ∧ b < a → ¬is_cube (5880 * b)) ↔ a = 1575 :=
sorry

end smallest_multiplier_for_cube_l2561_256198


namespace matrix_product_l2561_256192

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 2; 3, 4]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![4, 3; 2, 1]

theorem matrix_product :
  A * B = !![8, 5; 20, 13] := by sorry

end matrix_product_l2561_256192


namespace videotape_boxes_needed_l2561_256177

/-- Represents the duration of a program -/
structure Duration :=
  (value : ℝ)

/-- Represents a box of videotape -/
structure Box :=
  (capacity : ℝ)

/-- Represents the content to be recorded -/
structure Content :=
  (tvEpisodes : ℕ)
  (skits : ℕ)
  (songs : ℕ)

def Box.canRecord (b : Box) (d1 d2 : Duration) (n1 n2 : ℕ) : Prop :=
  n1 * d1.value + n2 * d2.value ≤ b.capacity

theorem videotape_boxes_needed 
  (tvDuration skitDuration songDuration : Duration)
  (box : Box)
  (content : Content)
  (h1 : box.canRecord tvDuration skitDuration 2 1)
  (h2 : box.canRecord skitDuration songDuration 2 3)
  (h3 : skitDuration.value > songDuration.value)
  (h4 : content.tvEpisodes = 7 ∧ content.skits = 11 ∧ content.songs = 20) :
  (∃ n : ℕ, n = 8 ∨ n = 9) ∧ 
  (∀ m : ℕ, m < 8 → 
    m * box.capacity < 
      content.tvEpisodes * tvDuration.value + 
      content.skits * skitDuration.value + 
      content.songs * songDuration.value) :=
sorry

end videotape_boxes_needed_l2561_256177


namespace same_color_config_prob_is_correct_l2561_256197

def total_candies : ℕ := 40
def red_candies : ℕ := 15
def blue_candies : ℕ := 15
def green_candies : ℕ := 10

def same_color_config_prob : ℚ :=
  let prob_both_red := (red_candies * (red_candies - 1) * (red_candies - 2) * (red_candies - 3)) / 
                       (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  let prob_both_blue := (blue_candies * (blue_candies - 1) * (blue_candies - 2) * (blue_candies - 3)) / 
                        (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  let prob_both_green := (green_candies * (green_candies - 1) * (green_candies - 2) * (green_candies - 3)) / 
                         (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  let prob_both_red_blue := (red_candies * blue_candies * (red_candies - 1) * (blue_candies - 1)) / 
                            (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  2 * prob_both_red + 2 * prob_both_blue + prob_both_green + 2 * prob_both_red_blue

theorem same_color_config_prob_is_correct : same_color_config_prob = 579 / 8686 := by
  sorry

end same_color_config_prob_is_correct_l2561_256197


namespace pentagon_angle_Q_measure_l2561_256122

-- Define the sum of angles in a pentagon
def pentagon_angle_sum : ℝ := 540

-- Define the known angles
def angle1 : ℝ := 130
def angle2 : ℝ := 90
def angle3 : ℝ := 110
def angle4 : ℝ := 115

-- Define the relation between Q and R
def Q_R_relation (Q R : ℝ) : Prop := Q = 2 * R

-- Theorem statement
theorem pentagon_angle_Q_measure :
  ∀ Q R : ℝ,
  Q_R_relation Q R →
  angle1 + angle2 + angle3 + angle4 + Q + R = pentagon_angle_sum →
  Q = 63.33 := by
  sorry


end pentagon_angle_Q_measure_l2561_256122


namespace sequence_distinct_terms_l2561_256133

theorem sequence_distinct_terms (n m : ℕ) (hn : n ≥ 1) (hm : m ≥ 1) (hnm : n ≠ m) :
  n / (n + 1 : ℚ) ≠ m / (m + 1 : ℚ) := by
  sorry

end sequence_distinct_terms_l2561_256133


namespace trig_identity_l2561_256124

theorem trig_identity (θ : ℝ) (h : θ ≠ 0) (h' : θ ≠ π/2) : 
  (Real.sin θ + 1 / Real.sin θ)^2 + (Real.cos θ + 1 / Real.cos θ)^2 = 
  6 + 2 * ((Real.sin θ / Real.cos θ)^2 + (Real.cos θ / Real.sin θ)^2) := by
sorry

end trig_identity_l2561_256124


namespace binary_arithmetic_equality_l2561_256117

theorem binary_arithmetic_equality : 
  (0b10110 : Nat) + 0b1101 - 0b11100 + 0b11101 + 0b101 = 0b101101 := by
  sorry

end binary_arithmetic_equality_l2561_256117


namespace ellipse_and_line_equation_l2561_256139

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal distance
def focal_distance (c : ℝ) : Prop := c = 3

-- Define the perimeter of triangle ABF₂
def perimeter_ABF₂ (a : ℝ) : Prop := 4 * a = 12 * Real.sqrt 2

-- Define points P and Q on the ellipse
def point_on_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the midpoint of PQ
def midpoint_PQ (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 2 ∧ (y₁ + y₂) / 2 = 1

-- Define the theorem
theorem ellipse_and_line_equation 
  (a b c : ℝ) 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : focal_distance c) 
  (h₄ : perimeter_ABF₂ a) 
  (h₅ : point_on_ellipse x₁ y₁ a b) 
  (h₆ : point_on_ellipse x₂ y₂ a b) 
  (h₇ : x₁ ≠ x₂ ∨ y₁ ≠ y₂) 
  (h₈ : midpoint_PQ x₁ y₁ x₂ y₂) : 
  (ellipse_C 3 (Real.sqrt 2) = ellipse_C 3 3) ∧ 
  (∀ (x y : ℝ), y = -(x - 2) + 1 ↔ x + y = 3) := by
  sorry

end ellipse_and_line_equation_l2561_256139


namespace point_not_on_graph_l2561_256170

/-- A linear function y = (k+1)x + 3 where k > -1 -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k + 1) * x + 3

/-- The constraint on k -/
def k_constraint (k : ℝ) : Prop := k > -1

theorem point_not_on_graph (k : ℝ) (h : k_constraint k) :
  ¬ (linear_function k 5 = -1) := by
  sorry

end point_not_on_graph_l2561_256170


namespace min_non_parallel_lines_l2561_256186

/-- A type representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A type representing a line in a plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Predicate to check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Function to create a line passing through two points -/
def line_through_points (p q : Point) : Line :=
  { a := q.y - p.y,
    b := p.x - q.x,
    c := p.x * q.y - q.x * p.y }

/-- The main theorem -/
theorem min_non_parallel_lines (n : ℕ) (points : Fin n → Point) 
  (h_n : n ≥ 3)
  (h_not_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) :
  ∃ (lines : Fin n → Line),
    (∀ i j, i ≠ j → ¬parallel (lines i) (lines j)) ∧
    (∀ lines' : Fin n' → Line, n' < n →
      ¬(∀ i j, i ≠ j → ¬parallel (lines' i) (lines' j))) :=
sorry

end min_non_parallel_lines_l2561_256186


namespace sum_of_squares_l2561_256174

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 2*b = 8)
  (eq2 : b^2 + 4*c + 1 = -6)
  (eq3 : c^2 + 6*a = -15) :
  a^2 + b^2 + c^2 = 14 := by
  sorry

end sum_of_squares_l2561_256174


namespace condition_p_sufficient_not_necessary_for_q_l2561_256132

theorem condition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (|x| = x) → (x^2 + x ≥ 0)) ∧
  (∃ x : ℝ, (x^2 + x ≥ 0) ∧ (|x| ≠ x)) :=
by sorry

end condition_p_sufficient_not_necessary_for_q_l2561_256132


namespace class_size_l2561_256146

/-- The number of students in Yuna's class -/
def total_students : ℕ := 33

/-- The number of students who like Korean -/
def korean_students : ℕ := 28

/-- The number of students who like math -/
def math_students : ℕ := 27

/-- The number of students who like both Korean and math -/
def both_subjects : ℕ := 22

/-- There is no student who does not like both Korean and math -/
axiom no_neither : total_students = korean_students + math_students - both_subjects

theorem class_size : total_students = 33 :=
sorry

end class_size_l2561_256146


namespace count_divisible_by_3_or_5_is_28_l2561_256137

/-- The count of numbers from 1 to 60 that are divisible by either 3 or 5 or both -/
def count_divisible_by_3_or_5 : ℕ :=
  let n := 60
  let divisible_by_3 := n / 3
  let divisible_by_5 := n / 5
  let divisible_by_15 := n / 15
  divisible_by_3 + divisible_by_5 - divisible_by_15

theorem count_divisible_by_3_or_5_is_28 : count_divisible_by_3_or_5 = 28 := by
  sorry

end count_divisible_by_3_or_5_is_28_l2561_256137
