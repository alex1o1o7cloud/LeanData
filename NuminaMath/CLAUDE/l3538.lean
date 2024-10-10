import Mathlib

namespace log_equation_solution_l3538_353821

theorem log_equation_solution (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 4 → x = 2401 := by
  sorry

end log_equation_solution_l3538_353821


namespace jill_study_time_difference_l3538_353840

/-- Represents the study time in minutes for each day -/
def StudyTime := Fin 3 → ℕ

theorem jill_study_time_difference (study : StudyTime) : 
  (study 0 = 120) →  -- First day study time in minutes
  (study 1 = 2 * study 0) →  -- Second day is double the first day
  (study 0 + study 1 + study 2 = 540) →  -- Total study time over 3 days
  (study 1 - study 2 = 60) :=  -- Difference between second and third day
by
  sorry

end jill_study_time_difference_l3538_353840


namespace inequality_proof_l3538_353852

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  x^2 * y^2 + y^2 * z^2 + z^2 * x^2 + x^2 * y^2 * z^2 ≤ 1/16 := by
sorry

end inequality_proof_l3538_353852


namespace g_equals_2x_minus_1_l3538_353806

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the property of g in relation to f
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) = f x

-- Theorem statement
theorem g_equals_2x_minus_1 (g : ℝ → ℝ) (h : g_property g) :
  ∀ x, g x = 2 * x - 1 := by
  sorry

end g_equals_2x_minus_1_l3538_353806


namespace allowance_calculation_l3538_353898

def initial_amount : ℕ := 10
def total_amount : ℕ := 18

theorem allowance_calculation :
  total_amount - initial_amount = 8 := by sorry

end allowance_calculation_l3538_353898


namespace complement_A_intersect_B_l3538_353808

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {2,5,8}
def B : Set ℕ := {1,3,5,7}

theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {1,3,7} :=
by sorry

end complement_A_intersect_B_l3538_353808


namespace m_minus_n_equals_negative_interval_l3538_353872

-- Define the sets M and N
def M : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

-- Define the set difference operation
def setDifference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- State the theorem
theorem m_minus_n_equals_negative_interval :
  setDifference M N = {x | -3 ≤ x ∧ x < 0} := by sorry

end m_minus_n_equals_negative_interval_l3538_353872


namespace problem_statement_l3538_353877

-- Define the logarithm function
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define symmetry about a point
def symmetric_about (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

-- Define symmetry about the origin
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem problem_statement :
  (∀ a : ℝ, a > 0 → a ≠ 1 → log_base a (a * (-1) + 2 * a) = 1) ∧
  (∃ f : ℝ → ℝ, symmetric_about_origin (fun x ↦ f (x - 3)) ∧
    ¬ symmetric_about f (3, 0)) := by
  sorry

end problem_statement_l3538_353877


namespace linear_decreasing_slope_l3538_353860

/-- A function that represents a linear equation with slope (m-3) and y-intercept 4 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x + 4

/-- The property that the function decreases as x increases -/
def decreasing (m : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → f m x₁ > f m x₂

theorem linear_decreasing_slope (m : ℝ) : decreasing m → m < 3 := by
  sorry

end linear_decreasing_slope_l3538_353860


namespace max_value_of_expression_l3538_353833

theorem max_value_of_expression (x : ℝ) : (2*x^2 + 8*x + 16) / (2*x^2 + 8*x + 6) ≤ 6 := by
  sorry

end max_value_of_expression_l3538_353833


namespace student_distribution_theorem_l3538_353874

/-- Represents the number of classes available --/
def num_classes : ℕ := 3

/-- Represents the number of students requesting to change classes --/
def num_students : ℕ := 4

/-- Represents the maximum number of additional students a class can accept --/
def max_per_class : ℕ := 2

/-- Calculates the number of ways to distribute students among classes --/
def distribution_ways : ℕ := 54

/-- Theorem stating the number of ways to distribute students --/
theorem student_distribution_theorem :
  (num_classes = 3) →
  (num_students = 4) →
  (max_per_class = 2) →
  distribution_ways = 54 :=
by
  sorry

#check student_distribution_theorem

end student_distribution_theorem_l3538_353874


namespace certain_number_problem_l3538_353858

theorem certain_number_problem (x : ℤ) : 
  ((7 * (x + 5)) / 5 : ℚ) - 5 = 33 ↔ x = 22 :=
by sorry

end certain_number_problem_l3538_353858


namespace line_arrangements_l3538_353876

/-- The number of different arrangements for 3 boys and 4 girls standing in a line under various conditions -/
theorem line_arrangements (n : ℕ) (boys : ℕ) (girls : ℕ) : 
  boys = 3 → girls = 4 → n = boys + girls →
  (∃ (arrangements_1 arrangements_2 arrangements_3 arrangements_4 : ℕ),
    /- Condition 1: Person A and B must stand at the two ends -/
    arrangements_1 = 240 ∧
    /- Condition 2: Person A cannot stand at the left end, and person B cannot stand at the right end -/
    arrangements_2 = 3720 ∧
    /- Condition 3: Person A and B must stand next to each other -/
    arrangements_3 = 1440 ∧
    /- Condition 4: The 3 boys are arranged from left to right in descending order of height -/
    arrangements_4 = 840) :=
by sorry

end line_arrangements_l3538_353876


namespace binomial_coefficient_20_19_l3538_353800

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_coefficient_20_19_l3538_353800


namespace seat_39_is_51_l3538_353844

/-- Calculates the seat number for the nth person in a circular seating arrangement --/
def seatNumber (n : ℕ) (totalSeats : ℕ) : ℕ :=
  if n = 1 then 1
  else
    let binaryRep := (n - 1).digits 2
    let seatCalc := binaryRep.foldl (fun acc (b : ℕ) => (2 * acc + b) % totalSeats) 1
    if seatCalc = 0 then totalSeats else seatCalc

/-- The theorem stating that the 39th person sits on seat 51 in a 128-seat arrangement --/
theorem seat_39_is_51 : seatNumber 39 128 = 51 := by
  sorry

/-- Verifies the seating arrangement for the first few people --/
example : List.map (fun n => seatNumber n 128) [1, 2, 3, 4, 5] = [1, 65, 33, 97, 17] := by
  sorry

end seat_39_is_51_l3538_353844


namespace traffic_light_change_probability_l3538_353816

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_duration : ℕ
  change_interval : ℕ

/-- Calculates the probability of observing a color change -/
def probability_of_change (cycle : TrafficLightCycle) : ℚ :=
  cycle.change_interval / cycle.total_duration

theorem traffic_light_change_probability :
  let cycle : TrafficLightCycle := ⟨93, 15⟩
  probability_of_change cycle = 5 / 31 := by
  sorry

end traffic_light_change_probability_l3538_353816


namespace intersection_M_N_l3538_353892

def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 6 > 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | -4 ≤ x ∧ x ≤ -2 ∨ 3 < x ∧ x ≤ 7} := by
  sorry

end intersection_M_N_l3538_353892


namespace polygon_area_bound_l3538_353893

/-- A polygon in 2D space --/
structure Polygon where
  -- We don't need to define the exact structure of the polygon,
  -- just its projections and area
  proj_ox : ℝ
  proj_bisector13 : ℝ
  proj_oy : ℝ
  proj_bisector24 : ℝ
  area : ℝ

/-- Theorem stating that the area of a polygon with given projections is bounded --/
theorem polygon_area_bound (p : Polygon)
  (h1 : p.proj_ox = 4)
  (h2 : p.proj_bisector13 = 3 * Real.sqrt 2)
  (h3 : p.proj_oy = 5)
  (h4 : p.proj_bisector24 = 4 * Real.sqrt 2)
  : p.area ≤ 17.5 := by
  sorry

end polygon_area_bound_l3538_353893


namespace cubic_derivative_value_l3538_353897

theorem cubic_derivative_value (f : ℝ → ℝ) (x₀ : ℝ) 
  (h1 : ∀ x, f x = x^3)
  (h2 : deriv f x₀ = 3) :
  x₀ = 1 ∨ x₀ = -1 := by
  sorry

end cubic_derivative_value_l3538_353897


namespace factorization_sum_l3538_353826

theorem factorization_sum (a b c d e f g h j k : ℤ) :
  (∃ (x y : ℝ), 27 * x^6 - 512 * y^6 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2)) →
  a + b + c + d + e + f + g + h + j + k = 152 := by
sorry

end factorization_sum_l3538_353826


namespace pascal_row8_sum_and_difference_l3538_353866

/-- Pascal's Triangle sum for a given row -/
def pascal_sum (n : ℕ) : ℕ := 2^n

theorem pascal_row8_sum_and_difference :
  (pascal_sum 8 = 256) ∧
  (pascal_sum 8 - pascal_sum 7 = 128) := by
  sorry

end pascal_row8_sum_and_difference_l3538_353866


namespace sum_of_digits_of_squared_repeated_ones_l3538_353880

/-- The number formed by repeating the digit '1' eight times -/
def repeated_ones : ℕ := 11111111

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_squared_repeated_ones : sum_of_digits (repeated_ones ^ 2) = 64 := by
  sorry

end sum_of_digits_of_squared_repeated_ones_l3538_353880


namespace b_third_place_four_times_l3538_353812

-- Define the structure for a contestant
structure Contestant where
  name : String
  firstPlace : Nat
  secondPlace : Nat
  thirdPlace : Nat

-- Define the competition parameters
def numCompetitions : Nat := 6
def firstPlaceScore : Nat := 5
def secondPlaceScore : Nat := 2
def thirdPlaceScore : Nat := 1

-- Define the contestants
def contestantA : Contestant := ⟨"A", 4, 1, 1⟩
def contestantB : Contestant := ⟨"B", 1, 0, 4⟩
def contestantC : Contestant := ⟨"C", 0, 3, 2⟩

-- Define the score calculation function
def calculateScore (c : Contestant) : Nat :=
  c.firstPlace * firstPlaceScore + c.secondPlace * secondPlaceScore + c.thirdPlace * thirdPlaceScore

-- Theorem to prove
theorem b_third_place_four_times :
  (calculateScore contestantA = 26) ∧
  (calculateScore contestantB = 11) ∧
  (calculateScore contestantC = 11) ∧
  (contestantB.firstPlace = 1) ∧
  (contestantA.firstPlace + contestantB.firstPlace + contestantC.firstPlace +
   contestantA.secondPlace + contestantB.secondPlace + contestantC.secondPlace +
   contestantA.thirdPlace + contestantB.thirdPlace + contestantC.thirdPlace = numCompetitions) →
  contestantB.thirdPlace = 4 := by
  sorry


end b_third_place_four_times_l3538_353812


namespace power_evaluation_l3538_353894

theorem power_evaluation (a b : ℕ) (h : 360 = 2^a * 3^2 * 5^b) 
  (h2 : ∀ k > a, ¬ 2^k ∣ 360) (h5 : ∀ k > b, ¬ 5^k ∣ 360) : 
  (2/3 : ℚ)^(b-a) = 9/4 := by
  sorry

end power_evaluation_l3538_353894


namespace min_odd_in_A_P_l3538_353853

/-- A polynomial of degree 8 -/
def Polynomial8 : Type := ℝ → ℝ

/-- The set A_P for a polynomial P -/
def A_P (P : Polynomial8) (c : ℝ) : Set ℝ := {x | P x = c}

/-- Theorem: If 8 is in A_P, then A_P contains at least one odd number -/
theorem min_odd_in_A_P (P : Polynomial8) (c : ℝ) (h : 8 ∈ A_P P c) :
  ∃ (x : ℝ), x ∈ A_P P c ∧ ∃ (n : ℤ), x = 2 * n + 1 :=
sorry

end min_odd_in_A_P_l3538_353853


namespace middle_letter_value_is_eight_l3538_353846

/-- Represents a three-letter word in Scrabble -/
structure ScrabbleWord where
  first_letter_value : ℕ
  middle_letter_value : ℕ
  last_letter_value : ℕ

/-- Calculates the total value of a ScrabbleWord before tripling -/
def word_value (word : ScrabbleWord) : ℕ :=
  word.first_letter_value + word.middle_letter_value + word.last_letter_value

/-- Theorem: Given the conditions, the middle letter's value is 8 -/
theorem middle_letter_value_is_eight 
  (word : ScrabbleWord)
  (h1 : word.first_letter_value = 1)
  (h2 : word.last_letter_value = 1)
  (h3 : 3 * (word_value word) = 30) :
  word.middle_letter_value = 8 := by
  sorry


end middle_letter_value_is_eight_l3538_353846


namespace system_of_equations_solution_system_of_inequalities_solution_l3538_353861

-- System of equations
theorem system_of_equations_solution (x y : ℝ) :
  (3 * x + 4 * y = 2) ∧ (2 * x - y = 5) ↔ (x = 2 ∧ y = -1) := by sorry

-- System of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (x - 3 * (x - 1) < 7) ∧ (x - 2 ≤ (2 * x - 3) / 3) ↔ (-2 < x ∧ x ≤ 3) := by sorry

end system_of_equations_solution_system_of_inequalities_solution_l3538_353861


namespace solution_of_system_l3538_353879

def system_of_equations (x y z : ℝ) : Prop :=
  1 / x = y + z ∧ 1 / y = z + x ∧ 1 / z = x + y

theorem solution_of_system :
  ∃ (x y z : ℝ), system_of_equations x y z ∧
    ((x = Real.sqrt 2 / 2 ∧ y = Real.sqrt 2 / 2 ∧ z = Real.sqrt 2 / 2) ∨
     (x = -Real.sqrt 2 / 2 ∧ y = -Real.sqrt 2 / 2 ∧ z = -Real.sqrt 2 / 2)) :=
by sorry

end solution_of_system_l3538_353879


namespace problem_solution_l3538_353823

theorem problem_solution (a b : ℝ) (h_distinct : a ≠ b) (h_sum_squares : a^2 + b^2 = 5) :
  (ab = 2 → a + b = 3 ∨ a + b = -3) ∧
  (a^2 - 2*a = b^2 - 2*b → a + b = 2 ∧ a^2 - 2*a = (1/2 : ℝ)) :=
by sorry

end problem_solution_l3538_353823


namespace inequality_proof_l3538_353837

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end inequality_proof_l3538_353837


namespace total_selling_price_l3538_353891

/-- Calculate the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
theorem total_selling_price
  (quantity : ℕ)
  (profit_per_meter : ℚ)
  (cost_price_per_meter : ℚ)
  (h1 : quantity = 92)
  (h2 : profit_per_meter = 24)
  (h3 : cost_price_per_meter = 83.5)
  : (quantity : ℚ) * (cost_price_per_meter + profit_per_meter) = 9890 := by
  sorry

end total_selling_price_l3538_353891


namespace pure_imaginary_modulus_l3538_353868

theorem pure_imaginary_modulus (b : ℝ) : 
  (∃ y : ℝ, (1 + b * Complex.I) * (2 - Complex.I) = y * Complex.I) → 
  Complex.abs (1 + b * Complex.I) = Real.sqrt 5 := by
sorry

end pure_imaginary_modulus_l3538_353868


namespace sum_of_digits_of_X_squared_sum_of_digits_of_111111111_squared_l3538_353804

/-- 
Given a natural number n, we define X as the number consisting of n ones.
For example, if n = 3, then X = 111.
-/
def X (n : ℕ) : ℕ := (10^n - 1) / 9

/-- 
The sum of digits function for a natural number.
-/
def sumOfDigits (m : ℕ) : ℕ := sorry

/-- 
Theorem: For a number X consisting of n ones, the sum of the digits of X^2 is equal to n^2.
-/
theorem sum_of_digits_of_X_squared (n : ℕ) : 
  sumOfDigits ((X n)^2) = n^2 := by sorry

/-- 
Corollary: For the specific case where n = 9 (corresponding to 111111111), 
the sum of the digits of X^2 is 81.
-/
theorem sum_of_digits_of_111111111_squared : 
  sumOfDigits ((X 9)^2) = 81 := by sorry

end sum_of_digits_of_X_squared_sum_of_digits_of_111111111_squared_l3538_353804


namespace integral_sin_cos_identity_l3538_353828

theorem integral_sin_cos_identity : 
  ∫ x in (0)..(π / 2), (Real.sin (Real.sin x))^2 + (Real.cos (Real.cos x))^2 = π / 4 := by
  sorry

end integral_sin_cos_identity_l3538_353828


namespace adult_ticket_cost_l3538_353815

theorem adult_ticket_cost 
  (total_spent : ℕ) 
  (family_size : ℕ) 
  (child_ticket_cost : ℕ) 
  (adult_tickets : ℕ) 
  (h1 : total_spent = 119)
  (h2 : family_size = 7)
  (h3 : child_ticket_cost = 14)
  (h4 : adult_tickets = 4) :
  ∃ (adult_ticket_cost : ℕ), 
    adult_ticket_cost * adult_tickets + 
    child_ticket_cost * (family_size - adult_tickets) = total_spent ∧ 
    adult_ticket_cost = 14 :=
by sorry

end adult_ticket_cost_l3538_353815


namespace pens_distribution_ways_l3538_353801

/-- The number of ways to distribute n identical objects among k recipients,
    where each recipient must receive at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The number of ways to distribute 9 pens among 3 friends,
    where each friend must receive at least one pen. -/
def distribute_pens : ℕ := distribute 9 3

theorem pens_distribution_ways : distribute_pens = 28 := by sorry

end pens_distribution_ways_l3538_353801


namespace intersection_of_lines_l3538_353848

theorem intersection_of_lines (x y : ℚ) : 
  x = 155 / 67 ∧ y = 5 / 67 ↔ 
  11 * x - 5 * y = 40 ∧ 9 * x + 2 * y = 15 :=
by sorry

end intersection_of_lines_l3538_353848


namespace tenth_black_ball_probability_l3538_353834

/-- Represents the probability of drawing a black ball on the tenth draw from a box of colored balls. -/
def probability_tenth_black_ball (total_balls : ℕ) (black_balls : ℕ) : ℚ :=
  black_balls / total_balls

/-- Theorem stating that the probability of drawing a black ball on the tenth draw
    from a box with specific numbers of colored balls is 4/30. -/
theorem tenth_black_ball_probability :
  let red_balls : ℕ := 7
  let black_balls : ℕ := 4
  let yellow_balls : ℕ := 5
  let green_balls : ℕ := 6
  let white_balls : ℕ := 8
  let total_balls : ℕ := red_balls + black_balls + yellow_balls + green_balls + white_balls
  probability_tenth_black_ball total_balls black_balls = 4 / 30 :=
by
  sorry

end tenth_black_ball_probability_l3538_353834


namespace vector_operation_proof_l3538_353886

def vector1 : ℝ × ℝ := (4, -5)
def vector2 : ℝ × ℝ := (-2, 8)

theorem vector_operation_proof :
  2 • (vector1 + vector2) = (4, 6) := by
  sorry

end vector_operation_proof_l3538_353886


namespace unique_four_digit_number_l3538_353896

/-- Represents a four-digit number as a tuple of four natural numbers -/
def FourDigitNumber := (ℕ × ℕ × ℕ × ℕ)

/-- Checks if a given FourDigitNumber satisfies all the required conditions -/
def satisfiesConditions (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  a ≠ 0 ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
  a + b = c + d ∧
  b + d = 2 * (a + c) ∧
  a + d = c ∧
  b + c - a = 3 * d

/-- Theorem stating that there exists a unique four-digit number satisfying all conditions -/
theorem unique_four_digit_number : ∃! n : FourDigitNumber, satisfiesConditions n := by
  sorry


end unique_four_digit_number_l3538_353896


namespace jenny_cat_expenditure_first_year_l3538_353802

/-- Calculates Jenny's expenditure on a cat for the first year -/
def jennys_cat_expenditure (adoption_fee : ℕ) (vet_costs : ℕ) (monthly_food_cost : ℕ) (jenny_toy_costs : ℕ) : ℕ :=
  let shared_costs := adoption_fee + vet_costs
  let jenny_shared_costs := shared_costs / 2
  let annual_food_cost := monthly_food_cost * 12
  let jenny_food_cost := annual_food_cost / 2
  jenny_shared_costs + jenny_food_cost + jenny_toy_costs

/-- Theorem stating Jenny's total expenditure on the cat in the first year -/
theorem jenny_cat_expenditure_first_year : 
  jennys_cat_expenditure 50 500 25 200 = 625 := by
  sorry

end jenny_cat_expenditure_first_year_l3538_353802


namespace triangle_exists_l3538_353881

/-- A triangle with vertices in ℝ² --/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- The area of a triangle --/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- The altitudes of a triangle --/
def Triangle.altitudes (t : Triangle) : List ℝ := sorry

/-- Theorem: There exists a triangle with all altitudes less than 1 and area greater than or equal to 10 --/
theorem triangle_exists : ∃ t : Triangle, (∀ h ∈ t.altitudes, h < 1) ∧ t.area ≥ 10 := by
  sorry

end triangle_exists_l3538_353881


namespace triangle_area_l3538_353887

/-- Given a triangle with perimeter 36 and inradius 2.5, its area is 45 -/
theorem triangle_area (p : ℝ) (r : ℝ) (area : ℝ) 
    (h1 : p = 36) (h2 : r = 2.5) (h3 : area = r * p / 2) : area = 45 := by
  sorry

end triangle_area_l3538_353887


namespace square_corners_l3538_353839

theorem square_corners (S : ℤ) : ∃ (A B C D : ℤ),
  A + B + 9 = S ∧
  B + C + 6 = S ∧
  D + C + 12 = S ∧
  D + A + 15 = S ∧
  A + C + 17 = S ∧
  A + B + C + D = 123 ∧
  A = 26 ∧ B = 37 ∧ C = 29 ∧ D = 31 := by
  sorry

end square_corners_l3538_353839


namespace stream_speed_l3538_353883

/-- Proves that given a man's downstream speed of 18 km/h, upstream speed of 6 km/h,
    and still water speed of 12 km/h, the speed of the stream is 6 km/h. -/
theorem stream_speed (v_downstream v_upstream v_stillwater : ℝ)
    (h_downstream : v_downstream = 18)
    (h_upstream : v_upstream = 6)
    (h_stillwater : v_stillwater = 12)
    (h_downstream_eq : v_downstream = v_stillwater + (v_downstream - v_upstream) / 2)
    (h_upstream_eq : v_upstream = v_stillwater - (v_downstream - v_upstream) / 2) :
    (v_downstream - v_upstream) / 2 = 6 :=
by sorry

end stream_speed_l3538_353883


namespace problem_1_problem_2_problem_3_l3538_353820

/-- The function f(x) = x^2 + 2ax - a + 2 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - a + 2

/-- Statement 1: For any x ∈ ℝ, f(x) ≥ 0 if and only if a ∈ [-2,1] --/
theorem problem_1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 0) ↔ a ∈ Set.Icc (-2) 1 := by sorry

/-- Statement 2: For any x ∈ [-1,1], f(x) ≥ 0 if and only if a ∈ [-3,1] --/
theorem problem_2 (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 0) ↔ a ∈ Set.Icc (-3) 1 := by sorry

/-- Statement 3: For any a ∈ [-1,1], x^2 + 2ax - a + 2 > 0 if and only if x ≠ -1 --/
theorem problem_3 (x : ℝ) :
  (∀ a ∈ Set.Icc (-1) 1, f a x > 0) ↔ x ≠ -1 := by sorry

end problem_1_problem_2_problem_3_l3538_353820


namespace cube_root_of_64_l3538_353867

theorem cube_root_of_64 : (64 : ℝ) ^ (1/3 : ℝ) = 4 := by sorry

end cube_root_of_64_l3538_353867


namespace prob_same_color_left_right_is_31_138_l3538_353805

def total_pairs : ℕ := 12
def blue_pairs : ℕ := 7
def red_pairs : ℕ := 3
def green_pairs : ℕ := 2

def total_shoes : ℕ := total_pairs * 2

def prob_same_color_left_right : ℚ :=
  (blue_pairs * total_pairs + red_pairs * total_pairs + green_pairs * total_pairs) / 
  (total_shoes * (total_shoes - 1))

theorem prob_same_color_left_right_is_31_138 : 
  prob_same_color_left_right = 31 / 138 := by
  sorry

end prob_same_color_left_right_is_31_138_l3538_353805


namespace parabola_focus_l3538_353835

/-- The focus of a parabola with equation y^2 = -6x has coordinates (-3/2, 0) -/
theorem parabola_focus (x y : ℝ) :
  y^2 = -6*x → (x + 3/2)^2 + y^2 = (3/2)^2 := by
  sorry

end parabola_focus_l3538_353835


namespace perpendicular_diameter_bisects_chord_equal_central_angles_equal_arcs_equal_chords_equal_arcs_equal_arcs_equal_central_angles_l3538_353854

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a chord
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

-- Define an arc
structure Arc (c : Circle) where
  startPoint : ℝ × ℝ
  endPoint : ℝ × ℝ

-- Define a central angle
def CentralAngle (c : Circle) (a : Arc c) : ℝ := sorry

-- Define the length of a chord
def chordLength (c : Circle) (ch : Chord c) : ℝ := sorry

-- Define the length of an arc
def arcLength (c : Circle) (a : Arc c) : ℝ := sorry

-- Define a diameter
def Diameter (c : Circle) := Chord c

-- Define perpendicularity between a diameter and a chord
def isPerpendicular (d : Diameter c) (ch : Chord c) : Prop := sorry

-- Define bisection of a chord
def bisectsChord (d : Diameter c) (ch : Chord c) : Prop := sorry

-- Theorem 1: A diameter perpendicular to a chord bisects the chord
theorem perpendicular_diameter_bisects_chord (c : Circle) (d : Diameter c) (ch : Chord c) :
  isPerpendicular d ch → bisectsChord d ch := sorry

-- Theorem 2: Equal central angles correspond to equal arcs
theorem equal_central_angles_equal_arcs (c : Circle) (a1 a2 : Arc c) :
  CentralAngle c a1 = CentralAngle c a2 → arcLength c a1 = arcLength c a2 := sorry

-- Theorem 3: Equal chords correspond to equal arcs
theorem equal_chords_equal_arcs (c : Circle) (ch1 ch2 : Chord c) (a1 a2 : Arc c) :
  chordLength c ch1 = chordLength c ch2 → arcLength c a1 = arcLength c a2 := sorry

-- Theorem 4: Equal arcs correspond to equal central angles
theorem equal_arcs_equal_central_angles (c : Circle) (a1 a2 : Arc c) :
  arcLength c a1 = arcLength c a2 → CentralAngle c a1 = CentralAngle c a2 := sorry

end perpendicular_diameter_bisects_chord_equal_central_angles_equal_arcs_equal_chords_equal_arcs_equal_arcs_equal_central_angles_l3538_353854


namespace max_a_when_f_has_minimum_l3538_353869

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then -a * x + 1 else (x - 2)^2

/-- Proposition: If f(x) has a minimum value, then the maximum value of a is 1 -/
theorem max_a_when_f_has_minimum (a : ℝ) :
  (∃ m : ℝ, ∀ x : ℝ, f a x ≥ m) → a ≤ 1 :=
by sorry

end max_a_when_f_has_minimum_l3538_353869


namespace pure_imaginary_condition_l3538_353819

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (2 - Complex.I) * (a + 2 * Complex.I) = b * Complex.I) → a = -1 := by
  sorry

end pure_imaginary_condition_l3538_353819


namespace eds_pets_l3538_353873

theorem eds_pets (dogs cats : ℕ) (h1 : dogs = 2) (h2 : cats = 3) : 
  let fish := 2 * (dogs + cats)
  dogs + cats + fish = 15 := by
  sorry

end eds_pets_l3538_353873


namespace equation_solutions_l3538_353885

theorem equation_solutions :
  ∀ (x n : ℕ+) (p : ℕ), 
    Prime p → 
    (x^3 + 3*x + 14 = 2*p^(n : ℕ)) → 
    ((x = 1 ∧ n = 2 ∧ p = 3) ∨ (x = 3 ∧ n = 2 ∧ p = 5)) :=
by sorry

end equation_solutions_l3538_353885


namespace gate_ticket_price_l3538_353850

/-- The price of plane tickets bought at the gate -/
def gate_price : ℝ := 200

/-- The number of people who pre-bought tickets -/
def pre_bought_count : ℕ := 20

/-- The price of pre-bought tickets -/
def pre_bought_price : ℝ := 155

/-- The number of people who bought tickets at the gate -/
def gate_count : ℕ := 30

/-- The additional amount paid in total by those who bought at the gate -/
def additional_gate_cost : ℝ := 2900

theorem gate_ticket_price :
  gate_price * gate_count = pre_bought_price * pre_bought_count + additional_gate_cost :=
by sorry

end gate_ticket_price_l3538_353850


namespace specific_triangle_area_l3538_353855

/-- Represents a triangle with given properties -/
structure Triangle where
  base : ℝ
  side : ℝ
  median : ℝ

/-- Calculates the area of a triangle given its properties -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating that a triangle with base 30, side 14, and median 13 has an area of 168 -/
theorem specific_triangle_area :
  let t : Triangle := { base := 30, side := 14, median := 13 }
  triangleArea t = 168 := by
  sorry

end specific_triangle_area_l3538_353855


namespace f_value_at_pi_sixth_f_monotone_increasing_intervals_l3538_353878

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.cos x) ^ 2

theorem f_value_at_pi_sixth : f (π / 6) = 0 := by sorry

theorem f_monotone_increasing_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (-(π / 6) + k * π) ((π / 3) + k * π)) := by sorry

end f_value_at_pi_sixth_f_monotone_increasing_intervals_l3538_353878


namespace line_angle_slope_relation_l3538_353865

/-- Given two lines L₁ and L₂ in the xy-plane, prove that mn = 1/3 under specific conditions. -/
theorem line_angle_slope_relation (m n : ℝ) : 
  -- L₁ has equation y = 3mx
  -- L₂ has equation y = nx
  -- L₁ makes three times as large of an angle with the horizontal as L₂
  -- L₁ has 3 times the slope of L₂
  (∃ (θ₁ θ₂ : ℝ), θ₁ = 3 * θ₂ ∧ Real.tan θ₁ = 3 * m ∧ Real.tan θ₂ = n) →
  -- L₁ has 3 times the slope of L₂
  3 * m = n →
  -- L₁ is not vertical
  m ≠ 0 →
  -- Conclusion: mn = 1/3
  m * n = 1 / 3 := by
  sorry

end line_angle_slope_relation_l3538_353865


namespace sequence_properties_l3538_353845

theorem sequence_properties (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = 2*n) :
  (a 2 = 5 ∧ a 3 = 11 ∧ a 4 = 19) ∧
  (∀ n : ℕ, a n = n^2 + n - 1) := by
sorry

end sequence_properties_l3538_353845


namespace smallest_rectangle_cover_l3538_353813

/-- The width of the rectangle -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the smallest square that can be covered by whole rectangles -/
def square_side : ℕ := lcm rectangle_width rectangle_height

/-- The area of the square -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem smallest_rectangle_cover :
  num_rectangles = 12 ∧
  ∀ n : ℕ, n < num_rectangles → 
    ¬ (∃ s : ℕ, s * s = n * rectangle_area) :=
sorry

end smallest_rectangle_cover_l3538_353813


namespace coin_division_problem_l3538_353842

theorem coin_division_problem (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(m % 7 = 3 ∧ m % 4 = 2)) →
  n % 7 = 3 →
  n % 4 = 2 →
  n % 8 = 2 :=
by sorry

end coin_division_problem_l3538_353842


namespace quadratic_equation_solution_l3538_353836

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 - 4*x₁ + 3 = 0) ∧ 
  (x₂^2 - 4*x₂ + 3 = 0) ∧ 
  x₁ = 3 ∧ 
  x₂ = 1 :=
by
  sorry

end quadratic_equation_solution_l3538_353836


namespace max_parts_three_planes_is_eight_l3538_353825

/-- The maximum number of parts that three planes can divide 3D space into -/
def max_parts_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of parts that three planes can divide 3D space into is 8 -/
theorem max_parts_three_planes_is_eight :
  max_parts_three_planes = 8 := by
  sorry

end max_parts_three_planes_is_eight_l3538_353825


namespace second_number_proof_l3538_353870

theorem second_number_proof (x : ℕ) : 
  (∃ k m : ℕ, 1657 = 127 * k + 6 ∧ x = 127 * m + 5 ∧ 
   ∀ d : ℕ, d > 127 → (1657 % d ≠ 6 ∨ x % d ≠ 5)) → 
  x = 1529 := by
sorry

end second_number_proof_l3538_353870


namespace inequality_problem_l3538_353809

theorem inequality_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 1) :
  (∃ (max_val : ℝ), a = 2 → (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/2 + 1/x + 1/y = 1 →
    1/(x + y) ≤ max_val) ∧ max_val = 1/8) ∧
  1/(a + b) + 1/(b + c) + 1/(a + c) ≤ 1/2 :=
by sorry

end inequality_problem_l3538_353809


namespace percent_relation_l3538_353829

theorem percent_relation (x y : ℝ) (h : (1/2) * (x - y) = (2/5) * (x + y)) :
  y = (1/9) * x := by
  sorry

end percent_relation_l3538_353829


namespace octal_difference_multiple_of_seven_fifty_six_possible_difference_l3538_353807

-- Define a two-digit number in base 8
def octal_number (tens units : Nat) : Nat :=
  8 * tens + units

-- Define the reversed number
def reversed_octal_number (tens units : Nat) : Nat :=
  8 * units + tens

-- Define the difference between the original and reversed number
def octal_difference (tens units : Nat) : Int :=
  (octal_number tens units : Int) - (reversed_octal_number tens units : Int)

-- Theorem stating that the difference is always a multiple of 7
theorem octal_difference_multiple_of_seven (tens units : Nat) :
  ∃ k : Int, octal_difference tens units = 7 * k :=
sorry

-- Theorem stating that 56 is a possible difference
theorem fifty_six_possible_difference :
  ∃ tens units : Nat, octal_difference tens units = 56 :=
sorry

end octal_difference_multiple_of_seven_fifty_six_possible_difference_l3538_353807


namespace journey_time_proof_l3538_353841

/-- Proves that a journey of 224 km, divided into two equal halves with different speeds, takes 10 hours -/
theorem journey_time_proof (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 224)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 10 := by
  sorry

end journey_time_proof_l3538_353841


namespace tire_comparison_l3538_353889

def type_A : List ℕ := [94, 96, 99, 99, 105, 107]
def type_B : List ℕ := [95, 95, 98, 99, 104, 109]

def mode (l : List ℕ) : ℕ := sorry
def range (l : List ℕ) : ℕ := sorry
def mean (l : List ℕ) : ℚ := sorry
def variance (l : List ℕ) : ℚ := sorry

theorem tire_comparison :
  (mode type_A > mode type_B) ∧
  (range type_A < range type_B) ∧
  (mean type_A = mean type_B) ∧
  (variance type_A < variance type_B) := by sorry

end tire_comparison_l3538_353889


namespace min_value_expression_lower_bound_achievable_l3538_353849

theorem min_value_expression (x y : ℝ) : 4 * x^2 + 4 * x * Real.sin y - Real.cos y^2 ≥ -1 := by
  sorry

theorem lower_bound_achievable : ∃ x y : ℝ, 4 * x^2 + 4 * x * Real.sin y - Real.cos y^2 = -1 := by
  sorry

end min_value_expression_lower_bound_achievable_l3538_353849


namespace total_money_proof_l3538_353817

/-- Represents the ratio of money shares for Jonah, Kira, and Liam respectively -/
def money_ratio : Fin 3 → ℕ
| 0 => 2  -- Jonah's ratio
| 1 => 3  -- Kira's ratio
| 2 => 8  -- Liam's ratio

/-- Kira's share of the money -/
def kiras_share : ℕ := 45

/-- The total amount of money shared -/
def total_money : ℕ := 195

/-- Theorem stating that given the conditions, the total amount of money shared is $195 -/
theorem total_money_proof :
  (∃ (multiplier : ℚ), 
    (multiplier * money_ratio 1 = kiras_share) ∧ 
    (multiplier * (money_ratio 0 + money_ratio 1 + money_ratio 2) = total_money)) :=
by sorry

end total_money_proof_l3538_353817


namespace max_value_of_f_l3538_353871

def f (x a : ℝ) : ℝ := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 1) :=
sorry

end max_value_of_f_l3538_353871


namespace some_number_value_l3538_353814

theorem some_number_value (x : ℝ) : 
  7^8 - 6/x + 9^3 + 3 + 12 = 95 → x = 1 / 960908.333 :=
by sorry

end some_number_value_l3538_353814


namespace fraction_sum_equality_l3538_353811

theorem fraction_sum_equality : (18 : ℚ) / 42 - 2 / 9 + 1 / 14 = 5 / 18 := by
  sorry

end fraction_sum_equality_l3538_353811


namespace unique_two_digit_integer_l3538_353803

theorem unique_two_digit_integer (t : ℕ) : 
  (t ≥ 10 ∧ t ≤ 99) ∧ (13 * t) % 100 = 47 ↔ t = 19 :=
by sorry

end unique_two_digit_integer_l3538_353803


namespace qiqi_mistake_xiaoming_jiajia_relation_l3538_353882

-- Define the polynomials A and B
def A (x : ℝ) : ℝ := -x^2 + 4*x
def B (x : ℝ) : ℝ := 2*x^2 + 5*x - 4

-- Define Jiajia's correct answer
def correct_answer : ℝ := -18

-- Define Qiqi's mistaken coefficient
def qiqi_coefficient : ℝ := 3

-- Define the value of x
def x_value : ℝ := -2

-- Theorem 1: Qiqi's mistaken coefficient
theorem qiqi_mistake :
  A x_value + (2*x_value^2 + qiqi_coefficient*x_value - 4) = correct_answer + 16 :=
sorry

-- Theorem 2: Relationship between Xiaoming's and Jiajia's results
theorem xiaoming_jiajia_relation :
  A (-x_value) + B (-x_value) = -(A x_value + B x_value) :=
sorry

end qiqi_mistake_xiaoming_jiajia_relation_l3538_353882


namespace jill_marathon_time_l3538_353890

/-- The length of a marathon in kilometers -/
def marathon_length : ℝ := 40

/-- Jack's marathon time in hours -/
def jack_time : ℝ := 4.5

/-- The ratio of Jack's speed to Jill's speed -/
def speed_ratio : ℝ := 0.888888888888889

/-- Jill's marathon time in hours -/
def jill_time : ℝ := 4

theorem jill_marathon_time :
  marathon_length / (marathon_length / jack_time * (1 / speed_ratio)) = jill_time := by
  sorry

end jill_marathon_time_l3538_353890


namespace xy_expression_value_l3538_353838

theorem xy_expression_value (x y m : ℝ) 
  (eq1 : x + y + m = 6) 
  (eq2 : 3 * x - y + m = 4) : 
  -2 * x * y + 1 = 3/2 := by
  sorry

end xy_expression_value_l3538_353838


namespace quadratic_inequality_integer_solution_l3538_353810

theorem quadratic_inequality_integer_solution (a : ℤ) : 
  (∀ x : ℝ, x^2 + 2*↑a*x + 1 > 0) → a = 0 := by
  sorry

end quadratic_inequality_integer_solution_l3538_353810


namespace sales_job_base_salary_l3538_353859

/-- The base salary of a sales job, given the following conditions:
  - The original salary was $75,000 per year
  - The new job pays a base salary plus 15% commission
  - Each sale is worth $750
  - 266.67 sales per year are needed to not lose money
-/
theorem sales_job_base_salary :
  ∀ (original_salary : ℝ) (commission_rate : ℝ) (sale_value : ℝ) (sales_needed : ℝ),
    original_salary = 75000 →
    commission_rate = 0.15 →
    sale_value = 750 →
    sales_needed = 266.67 →
    ∃ (base_salary : ℝ),
      base_salary + sales_needed * commission_rate * sale_value = original_salary ∧
      base_salary = 45000 :=
by sorry

end sales_job_base_salary_l3538_353859


namespace range_of_a_l3538_353830

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → -1 < a ∧ a < 3 := by
  sorry

end range_of_a_l3538_353830


namespace calculation_problem_l3538_353832

theorem calculation_problem (n : ℝ) : n = -6.4 ↔ 10 * 1.8 - (n * 1.5 / 0.3) = 50 := by
  sorry

end calculation_problem_l3538_353832


namespace recurring_decimal_sum_l3538_353818

theorem recurring_decimal_sum : 
  (2 : ℚ) / 3 + 7 / 9 = 13 / 9 := by sorry

end recurring_decimal_sum_l3538_353818


namespace roulette_probability_l3538_353864

theorem roulette_probability (p_X p_Y p_Z p_W : ℚ) : 
  p_X = 1/4 → p_Y = 1/3 → p_W = 1/6 → p_X + p_Y + p_Z + p_W = 1 → p_Z = 1/4 := by
  sorry

end roulette_probability_l3538_353864


namespace consecutive_even_integers_sum_68_l3538_353843

theorem consecutive_even_integers_sum_68 :
  ∃ (x y z w : ℕ+), 
    (x : ℤ) + y + z + w = 68 ∧
    y = x + 2 ∧ z = y + 2 ∧ w = z + 2 ∧
    Even x ∧ Even y ∧ Even z ∧ Even w :=
by sorry

end consecutive_even_integers_sum_68_l3538_353843


namespace blue_cube_problem_l3538_353847

theorem blue_cube_problem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end blue_cube_problem_l3538_353847


namespace exactly_three_true_l3538_353895

-- Define the propositions
def prop1 : Prop := ∀ p q : Prop, (p ∧ q → False) → (p → False) ∧ (q → False)

def prop2 : Prop := (¬∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0)

def prop3 : Prop := 
  let p : ℝ → Prop := λ x ↦ x ≤ 1
  let q : ℝ → Prop := λ x ↦ 1 / x < 1
  (∀ x : ℝ, ¬(p x) → q x) ∧ ¬(∀ x : ℝ, q x → ¬(p x))

noncomputable def prop4 : Prop :=
  let X : ℝ → ℝ := λ _ ↦ 0  -- Placeholder for normal distribution
  ∀ C : ℝ, (∀ x : ℝ, (X x > C + 1) ↔ (X x < C - 1)) → C = 3

-- The main theorem
theorem exactly_three_true : 
  (¬prop1 ∧ prop2 ∧ prop3 ∧ prop4) ∨
  (prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4) ∨
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) ∨
  (prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4) :=
sorry

end exactly_three_true_l3538_353895


namespace job_completion_time_l3538_353831

theorem job_completion_time (x : ℝ) : 
  x > 0 → 
  (1 / x + 1 / 30 = 1 / 10) → 
  x = 15 := by
sorry

end job_completion_time_l3538_353831


namespace cab_base_price_l3538_353827

/-- Represents the base price of a cab ride -/
def base_price : ℝ := sorry

/-- Represents the per-mile charge of a cab ride -/
def per_mile_charge : ℝ := 4

/-- Represents the total distance traveled in miles -/
def distance : ℝ := 5

/-- Represents the total cost of the cab ride -/
def total_cost : ℝ := 23

/-- Theorem stating that the base price of the cab ride is $3 -/
theorem cab_base_price : base_price = 3 := by
  sorry

end cab_base_price_l3538_353827


namespace negation_of_universal_proposition_l3538_353862

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

theorem negation_of_universal_proposition :
  (¬ ∀ n ∈ M, n > 1) ↔ (∃ n ∈ M, n ≤ 1) := by sorry

end negation_of_universal_proposition_l3538_353862


namespace eggs_given_by_marie_l3538_353884

/-- Given that Joyce initially had 8 eggs and ended up with 14 eggs in total,
    prove that Marie gave Joyce 6 eggs. -/
theorem eggs_given_by_marie 
  (initial_eggs : ℕ) 
  (total_eggs : ℕ) 
  (h1 : initial_eggs = 8) 
  (h2 : total_eggs = 14) : 
  total_eggs - initial_eggs = 6 := by
  sorry

end eggs_given_by_marie_l3538_353884


namespace max_value_log_expression_l3538_353824

open Real

theorem max_value_log_expression (x : ℝ) (h : x > -1) :
  ∃ M, M = -2 ∧ 
  (log (x + 1 / (x + 1) + 3) / log (1/2) ≤ M) ∧
  ∃ x₀, x₀ > -1 ∧ log (x₀ + 1 / (x₀ + 1) + 3) / log (1/2) = M :=
by sorry

end max_value_log_expression_l3538_353824


namespace club_president_secretary_choices_l3538_353822

/-- A club with boys and girls -/
structure Club where
  total : ℕ
  boys : ℕ
  girls : ℕ

/-- The number of ways to choose a president (boy) and secretary (girl) from a club -/
def choosePresidentAndSecretary (c : Club) : ℕ :=
  c.boys * c.girls

/-- Theorem stating that for a club with 30 members (18 boys and 12 girls),
    the number of ways to choose a president and secretary is 216 -/
theorem club_president_secretary_choices :
  let c : Club := { total := 30, boys := 18, girls := 12 }
  choosePresidentAndSecretary c = 216 := by
  sorry

end club_president_secretary_choices_l3538_353822


namespace toy_truck_cost_l3538_353856

/-- The amount spent on toy trucks, given the total spent on toys and the costs of toy cars and skateboard. -/
theorem toy_truck_cost (total_toys : ℚ) (toy_cars : ℚ) (skateboard : ℚ) 
  (h1 : total_toys = 25.62)
  (h2 : toy_cars = 14.88)
  (h3 : skateboard = 4.88) :
  total_toys - (toy_cars + skateboard) = 5.86 := by
  sorry

end toy_truck_cost_l3538_353856


namespace center_square_side_length_l3538_353899

theorem center_square_side_length 
  (large_square_side : ℝ) 
  (l_shape_area_fraction : ℝ) 
  (num_l_shapes : ℕ) : 
  large_square_side = 120 →
  l_shape_area_fraction = 1/5 →
  num_l_shapes = 4 →
  ∃ (center_square_side : ℝ),
    center_square_side = 60 ∧
    center_square_side^2 = large_square_side^2 - num_l_shapes * l_shape_area_fraction * large_square_side^2 :=
by sorry

end center_square_side_length_l3538_353899


namespace popsicle_sticks_count_l3538_353863

theorem popsicle_sticks_count 
  (num_groups : ℕ) 
  (sticks_per_group : ℕ) 
  (sticks_left : ℕ) 
  (h1 : num_groups = 10)
  (h2 : sticks_per_group = 15)
  (h3 : sticks_left = 20) :
  num_groups * sticks_per_group + sticks_left = 170 := by
  sorry

end popsicle_sticks_count_l3538_353863


namespace equal_area_rectangles_l3538_353851

/-- Given two rectangles with equal areas, where one rectangle has dimensions 12 inches by W inches,
    and the other has dimensions 9 inches by 20 inches, prove that W equals 15 inches. -/
theorem equal_area_rectangles (W : ℝ) :
  (12 * W = 9 * 20) → W = 15 := by sorry

end equal_area_rectangles_l3538_353851


namespace circle_area_l3538_353857

theorem circle_area (c : ℝ) (h : c = 18 * Real.pi) :
  ∃ r : ℝ, c = 2 * Real.pi * r ∧ Real.pi * r^2 = 81 * Real.pi := by
  sorry

end circle_area_l3538_353857


namespace final_price_is_135_l3538_353875

/-- The original price of the dress -/
def original_price : ℝ := 250

/-- The first discount rate -/
def first_discount_rate : ℝ := 0.4

/-- The additional holiday discount rate -/
def holiday_discount_rate : ℝ := 0.1

/-- The price after the first discount -/
def price_after_first_discount : ℝ := original_price * (1 - first_discount_rate)

/-- The final price after both discounts -/
def final_price : ℝ := price_after_first_discount * (1 - holiday_discount_rate)

/-- Theorem stating that the final price is $135 -/
theorem final_price_is_135 : final_price = 135 := by sorry

end final_price_is_135_l3538_353875


namespace sally_has_88_cards_l3538_353888

/-- The number of Pokemon cards Sally has after receiving a gift and making a purchase -/
def sallys_cards (initial : ℕ) (gift : ℕ) (purchase : ℕ) : ℕ :=
  initial + gift + purchase

/-- Theorem: Sally has 88 Pokemon cards after starting with 27, receiving 41 as a gift, and buying 20 -/
theorem sally_has_88_cards : sallys_cards 27 41 20 = 88 := by
  sorry

end sally_has_88_cards_l3538_353888
