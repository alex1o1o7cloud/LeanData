import Mathlib

namespace three_tangents_range_l3611_361119

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- Predicate to check if a point (x, y) is on the curve y = f(x) --/
def on_curve (x y : ℝ) : Prop := y = f x

/-- Predicate to check if a line through (1, m) is tangent to the curve at some point --/
def is_tangent (m t : ℝ) : Prop := 
  ∃ x : ℝ, on_curve x (f x) ∧ (m - f 1) = f' x * (1 - x)

/-- The main theorem --/
theorem three_tangents_range (m : ℝ) : 
  (∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    is_tangent m t1 ∧ is_tangent m t2 ∧ is_tangent m t3) → 
  m > -3 ∧ m < -2 :=
sorry

end three_tangents_range_l3611_361119


namespace dihedral_angle_range_in_regular_prism_l3611_361101

theorem dihedral_angle_range_in_regular_prism (n : ℕ) (h : n > 2) :
  ∃ θ : ℝ, ((n - 2 : ℝ) / n) * π < θ ∧ θ < π :=
sorry

end dihedral_angle_range_in_regular_prism_l3611_361101


namespace hit_first_third_fifth_probability_hit_exactly_three_probability_l3611_361190

-- Define the probability of hitting the target
def hit_probability : ℚ := 3/5

-- Define the number of shots
def num_shots : ℕ := 5

-- Theorem for the first part of the problem
theorem hit_first_third_fifth_probability :
  (hit_probability * (1 - hit_probability) * hit_probability * (1 - hit_probability) * hit_probability : ℚ) = 108/3125 :=
sorry

-- Theorem for the second part of the problem
theorem hit_exactly_three_probability :
  (Nat.choose num_shots 3 : ℚ) * hit_probability^3 * (1 - hit_probability)^2 = 216/625 :=
sorry

end hit_first_third_fifth_probability_hit_exactly_three_probability_l3611_361190


namespace meaningful_expression_l3611_361186

/-- The expression x + 1/(x-2) is meaningful for all real x except 2 -/
theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = x + 1 / (x - 2)) ↔ x ≠ 2 :=
sorry

end meaningful_expression_l3611_361186


namespace linear_function_increasing_l3611_361139

/-- Given a linear function f(x) = 2x - 1, prove that for any two points
    (x₁, y₁) and (x₂, y₂) on its graph, if x₁ > x₂, then y₁ > y₂ -/
theorem linear_function_increasing (x₁ x₂ y₁ y₂ : ℝ) 
    (h1 : y₁ = 2 * x₁ - 1)
    (h2 : y₂ = 2 * x₂ - 1)
    (h3 : x₁ > x₂) : 
  y₁ > y₂ := by
  sorry

end linear_function_increasing_l3611_361139


namespace no_solution_for_inequality_l3611_361124

theorem no_solution_for_inequality (a b : ℝ) (h : |a - b| > 2) :
  ¬∃ x : ℝ, |x - a| + |x - b| ≤ 2 :=
by sorry

end no_solution_for_inequality_l3611_361124


namespace sum_of_root_products_l3611_361115

theorem sum_of_root_products (a b c d : ℂ) : 
  (2 * a^4 - 6 * a^3 + 14 * a^2 - 13 * a + 8 = 0) →
  (2 * b^4 - 6 * b^3 + 14 * b^2 - 13 * b + 8 = 0) →
  (2 * c^4 - 6 * c^3 + 14 * c^2 - 13 * c + 8 = 0) →
  (2 * d^4 - 6 * d^3 + 14 * d^2 - 13 * d + 8 = 0) →
  a * b + a * c + a * d + b * c + b * d + c * d = -7 := by
sorry

end sum_of_root_products_l3611_361115


namespace geometric_to_arithmetic_progression_l3611_361175

theorem geometric_to_arithmetic_progression :
  ∀ (a q : ℝ),
    a > 0 → q > 0 →
    a + a * q + a * q^2 = 105 →
    ∃ d : ℝ, a * q - a = (a * q^2 - 15) - a * q →
    (a = 15 ∧ q = 2) ∨ (a = 60 ∧ q = 1/2) :=
by sorry

end geometric_to_arithmetic_progression_l3611_361175


namespace line_intersects_circle_l3611_361105

/-- The line y - 1 = k(x - 1) always intersects the circle x² + y² - 2y = 0 for any real number k. -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  (y - 1 = k * (x - 1)) ∧ (x^2 + y^2 - 2*y = 0) :=
sorry

end line_intersects_circle_l3611_361105


namespace car_trip_speed_l3611_361109

/-- Given a car trip with the following properties:
  1. The car averages a certain speed for the first 6 hours.
  2. The car averages 46 miles per hour for each additional hour.
  3. The average speed for the entire trip is 34 miles per hour.
  4. The trip is 8 hours long.
  Prove that the average speed for the first 6 hours of the trip is 30 miles per hour. -/
theorem car_trip_speed (initial_speed : ℝ) : initial_speed = 30 := by
  sorry

end car_trip_speed_l3611_361109


namespace toms_profit_is_21988_l3611_361112

/-- Calculates Tom's profit from making the world's largest dough ball -/
def toms_profit (flour_needed : ℕ) (flour_bag_size : ℕ) (flour_bag_cost : ℕ)
                (salt_needed : ℕ) (salt_cost : ℚ)
                (sugar_needed : ℕ) (sugar_cost : ℚ)
                (butter_needed : ℕ) (butter_cost : ℕ)
                (chef_cost : ℕ) (promotion_cost : ℕ)
                (ticket_price : ℕ) (tickets_sold : ℕ) : ℤ :=
  let flour_cost := (flour_needed / flour_bag_size) * flour_bag_cost
  let salt_cost_total := (salt_needed : ℚ) * salt_cost
  let sugar_cost_total := (sugar_needed : ℚ) * sugar_cost
  let butter_cost_total := butter_needed * butter_cost
  let total_cost := flour_cost + salt_cost_total.ceil + sugar_cost_total.ceil + 
                    butter_cost_total + chef_cost + promotion_cost
  let revenue := ticket_price * tickets_sold
  revenue - total_cost

/-- Tom's profit from making the world's largest dough ball is $21988 -/
theorem toms_profit_is_21988 : 
  toms_profit 500 50 20 10 (2/10) 20 (1/2) 50 2 700 1000 20 1200 = 21988 := by
  sorry

end toms_profit_is_21988_l3611_361112


namespace four_bottles_left_l3611_361187

/-- The number of bottles left after a given number of days for a person who drinks half a bottle per day -/
def bottles_left (initial_bottles : ℕ) (days : ℕ) : ℕ :=
  initial_bottles - (days / 2)

/-- Theorem stating that 4 bottles will be left after 28 days, starting with 18 bottles -/
theorem four_bottles_left : bottles_left 18 28 = 4 := by
  sorry

end four_bottles_left_l3611_361187


namespace fourth_term_of_sequence_l3611_361108

theorem fourth_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = 4 * n^2) →
  a 4 = 28 :=
sorry

end fourth_term_of_sequence_l3611_361108


namespace range_of_a_l3611_361173

-- Define the set M
def M (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 4*x + 4*a < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (2 ∉ M a) ↔ a ∈ Set.Ici 1 := by
  sorry

end range_of_a_l3611_361173


namespace train_meeting_distance_l3611_361161

/-- Proves that Train A travels 75 miles before meeting Train B -/
theorem train_meeting_distance (route_length : ℝ) (time_a : ℝ) (time_b : ℝ) 
  (h1 : route_length = 200)
  (h2 : time_a = 10)
  (h3 : time_b = 6)
  : (route_length / time_a) * (route_length / (route_length / time_a + route_length / time_b)) = 75 := by
  sorry

end train_meeting_distance_l3611_361161


namespace constant_term_value_l3611_361165

theorem constant_term_value (y : ℝ) (d : ℝ) :
  y = 2 → (5 * y^2 - 8 * y + 55 = d ↔ d = 59) := by
  sorry

end constant_term_value_l3611_361165


namespace fraction_division_result_l3611_361163

theorem fraction_division_result : (3 / 8) / (5 / 9) = 27 / 40 := by sorry

end fraction_division_result_l3611_361163


namespace employee_payments_correct_l3611_361107

def video_recorder_price (wholesale : ℝ) (markup : ℝ) : ℝ :=
  wholesale * (1 + markup)

def employee_payment (retail : ℝ) (discount : ℝ) : ℝ :=
  retail * (1 - discount)

theorem employee_payments_correct :
  let wholesale_A := 200
  let wholesale_B := 250
  let wholesale_C := 300
  let markup_A := 0.20
  let markup_B := 0.25
  let markup_C := 0.30
  let discount_X := 0.15
  let discount_Y := 0.18
  let discount_Z := 0.20
  
  let retail_A := video_recorder_price wholesale_A markup_A
  let retail_B := video_recorder_price wholesale_B markup_B
  let retail_C := video_recorder_price wholesale_C markup_C
  
  let payment_X := employee_payment retail_A discount_X
  let payment_Y := employee_payment retail_B discount_Y
  let payment_Z := employee_payment retail_C discount_Z
  
  payment_X = 204 ∧ payment_Y = 256.25 ∧ payment_Z = 312 :=
by sorry

end employee_payments_correct_l3611_361107


namespace complex_parts_of_one_plus_sqrt_three_i_l3611_361193

theorem complex_parts_of_one_plus_sqrt_three_i :
  let z : ℂ := Complex.I * (1 + Real.sqrt 3)
  Complex.re z = 0 ∧ Complex.im z = 1 + Real.sqrt 3 := by
sorry

end complex_parts_of_one_plus_sqrt_three_i_l3611_361193


namespace solution_set_quadratic_inequality_l3611_361134

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 2*x > 0} = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end solution_set_quadratic_inequality_l3611_361134


namespace range_of_a_l3611_361126

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l3611_361126


namespace company_earnings_difference_l3611_361138

/-- Represents a company selling bottled milk -/
structure Company where
  price : ℝ  -- Price of a big bottle
  sold : ℕ   -- Number of big bottles sold

/-- Calculates the earnings of a company -/
def earnings (c : Company) : ℝ := c.price * c.sold

/-- The problem statement -/
theorem company_earnings_difference 
  (company_a company_b : Company)
  (ha : company_a.price = 4)
  (hb : company_b.price = 3.5)
  (sa : company_a.sold = 300)
  (sb : company_b.sold = 350) :
  earnings company_b - earnings company_a = 25 := by
  sorry

end company_earnings_difference_l3611_361138


namespace complement_of_union_is_empty_l3611_361121

universe u

def U : Set Char := {'a', 'b', 'c', 'd', 'e'}
def N : Set Char := {'b', 'd', 'e'}
def M : Set Char := {'a', 'c', 'd'}

theorem complement_of_union_is_empty :
  (M ∪ N)ᶜ = ∅ :=
sorry

end complement_of_union_is_empty_l3611_361121


namespace complement_of_A_in_U_l3611_361137

def U : Set ℤ := {-1, 0, 1, 2}

def A : Set ℤ := {x ∈ U | x^2 < 1}

theorem complement_of_A_in_U : Set.compl A = {-1, 1, 2} := by sorry

end complement_of_A_in_U_l3611_361137


namespace geometric_sequence_fourth_term_l3611_361145

theorem geometric_sequence_fourth_term :
  ∀ x : ℚ,
  let a₁ := x
  let a₂ := 3*x + 3
  let a₃ := 5*x + 5
  let r := a₂ / a₁
  (a₂ = r * a₁) ∧ (a₃ = r * a₂) →
  let a₄ := r * a₃
  a₄ = -125/12 :=
by sorry

end geometric_sequence_fourth_term_l3611_361145


namespace fraction_simplification_l3611_361182

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (5 - 4 * x) / 3 = (-13 * x + 26) / 12 := by
  sorry

end fraction_simplification_l3611_361182


namespace common_difference_is_two_l3611_361147

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_two (seq : ArithmeticSequence) 
  (h : seq.S 4 / 4 - seq.S 2 / 2 = 2) : 
  commonDifference seq = 2 := by
  sorry

end common_difference_is_two_l3611_361147


namespace henry_tournament_points_l3611_361118

/-- Point system for the tic-tac-toe tournament --/
structure PointSystem where
  win_points : ℕ
  loss_points : ℕ
  draw_points : ℕ

/-- Results of Henry's tournament --/
structure TournamentResults where
  wins : ℕ
  losses : ℕ
  draws : ℕ

/-- Calculate the total points for a given point system and tournament results --/
def calculateTotalPoints (ps : PointSystem) (tr : TournamentResults) : ℕ :=
  ps.win_points * tr.wins + ps.loss_points * tr.losses + ps.draw_points * tr.draws

/-- Theorem: Henry's total points in the tournament --/
theorem henry_tournament_points :
  let ps : PointSystem := { win_points := 5, loss_points := 2, draw_points := 3 }
  let tr : TournamentResults := { wins := 2, losses := 2, draws := 10 }
  calculateTotalPoints ps tr = 44 := by
  sorry


end henry_tournament_points_l3611_361118


namespace problem_solution_l3611_361106

-- Define the absolute value function
def f (x : ℝ) : ℝ := abs x

-- Define the function g
def g (t : ℝ) (x : ℝ) : ℝ := 3 * f x - f (x - t)

theorem problem_solution :
  -- Part I
  (∀ a : ℝ, a > 0 → (∀ x : ℝ, 2 * f (x - 1) + f (2 * x - a) ≥ 1) →
    (a ∈ Set.Ioo 0 1 ∪ Set.Ici 3)) ∧
  -- Part II
  (∀ t : ℝ, t ≠ 0 →
    (∫ x, abs (g t x)) = 3 →
    t = 2 * Real.sqrt 2 ∨ t = -2 * Real.sqrt 2) :=
by sorry

end problem_solution_l3611_361106


namespace deck_size_proof_l3611_361100

theorem deck_size_proof (spades : ℕ) (prob_not_spade : ℚ) (total : ℕ) : 
  spades = 13 → 
  prob_not_spade = 3/4 → 
  (total - spades : ℚ) / total = prob_not_spade → 
  total = 52 := by
sorry

end deck_size_proof_l3611_361100


namespace isabellas_house_number_l3611_361174

/-- A predicate that checks if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- A predicate that checks if a natural number has 9 as one of its digits -/
def has_digit_nine (n : ℕ) : Prop := ∃ d, d ∈ n.digits 10 ∧ d = 9

theorem isabellas_house_number :
  ∃! n : ℕ, is_two_digit n ∧
           ¬ Nat.Prime n ∧
           Even n ∧
           n % 7 = 0 ∧
           has_digit_nine n ∧
           n % 10 = 8 := by sorry

end isabellas_house_number_l3611_361174


namespace difference_of_squares_101_99_l3611_361199

theorem difference_of_squares_101_99 : 101^2 - 99^2 = 400 := by
  sorry

end difference_of_squares_101_99_l3611_361199


namespace sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l3611_361142

def fibonacci_factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 24
  | 5 => 120
  | 6 => 720
  | 7 => 5040
  | 8 => 40320
  | 9 => 362880
  | _ => 0  -- For n ≥ 10, we only care about the last two digits, which are 00

def modified_fibonacci_series : List ℕ :=
  [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 55]

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

theorem sum_of_last_two_digits_of_modified_fibonacci_factorial_series :
  (modified_fibonacci_series.map (λ x => last_two_digits (fibonacci_factorial x))).sum % 10 = 5 := by
  sorry

end sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l3611_361142


namespace line_intersects_parabola_at_one_point_l3611_361150

/-- The value of k for which the line x = k intersects the parabola x = -3y² - 4y + 7 at exactly one point -/
def k : ℚ := 25/3

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3*y^2 - 4*y + 7

theorem line_intersects_parabola_at_one_point :
  ∃! y : ℝ, parabola y = k := by sorry

end line_intersects_parabola_at_one_point_l3611_361150


namespace simple_interest_rate_l3611_361128

/-- 
Given a principal amount P and a time period of 10 years, 
prove that the rate percent per annum R is 12% when the simple interest 
is 6/5 of the principal amount.
-/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  let R := (6 / 5) * 100 / 10
  let simple_interest := (P * R * 10) / 100
  simple_interest = (6 / 5) * P → R = 12 := by
  sorry

#check simple_interest_rate

end simple_interest_rate_l3611_361128


namespace count_decreasing_digit_numbers_l3611_361148

/-- A function that checks if a natural number has strictly decreasing digits. -/
def hasDecreasingDigits (n : ℕ) : Bool :=
  sorry

/-- The count of natural numbers with at least two digits and strictly decreasing digits. -/
def countDecreasingDigitNumbers : ℕ :=
  sorry

/-- Theorem stating that the count of natural numbers with at least two digits 
    and strictly decreasing digits is 1013. -/
theorem count_decreasing_digit_numbers :
  countDecreasingDigitNumbers = 1013 := by
  sorry

end count_decreasing_digit_numbers_l3611_361148


namespace greatest_multiple_of_four_under_cube_root_8000_l3611_361141

theorem greatest_multiple_of_four_under_cube_root_8000 :
  (∃ (x : ℕ), x > 0 ∧ 4 ∣ x ∧ x^3 < 8000 ∧ ∀ (y : ℕ), y > 0 → 4 ∣ y → y^3 < 8000 → y ≤ x) →
  (∃ (x : ℕ), x = 16 ∧ x > 0 ∧ 4 ∣ x ∧ x^3 < 8000 ∧ ∀ (y : ℕ), y > 0 → 4 ∣ y → y^3 < 8000 → y ≤ x) :=
by sorry

end greatest_multiple_of_four_under_cube_root_8000_l3611_361141


namespace class_selection_theorem_l3611_361113

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of boys in the class. -/
def num_boys : ℕ := 13

/-- The total number of girls in the class. -/
def num_girls : ℕ := 10

/-- The number of boys selected. -/
def boys_selected : ℕ := 2

/-- The number of girls selected. -/
def girls_selected : ℕ := 1

/-- The total number of possible combinations. -/
def total_combinations : ℕ := 780

theorem class_selection_theorem :
  choose num_boys boys_selected * choose num_girls girls_selected = total_combinations :=
sorry

end class_selection_theorem_l3611_361113


namespace cricket_player_average_increase_l3611_361129

/-- 
Theorem: Cricket Player's Average Increase

Given:
- A cricket player has played 10 innings
- The current average is 32 runs per innings
- The player needs to make 76 runs in the next innings

Prove: The increase in average is 4 runs per innings
-/
theorem cricket_player_average_increase 
  (innings : ℕ) 
  (current_average : ℚ) 
  (next_innings_runs : ℕ) 
  (h1 : innings = 10)
  (h2 : current_average = 32)
  (h3 : next_innings_runs = 76) : 
  (((innings : ℚ) * current_average + next_innings_runs) / (innings + 1) - current_average) = 4 := by
  sorry


end cricket_player_average_increase_l3611_361129


namespace girls_from_pine_l3611_361188

theorem girls_from_pine (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (maple_students : ℕ) (pine_students : ℕ) (maple_boys : ℕ)
  (h1 : total_students = 120)
  (h2 : total_boys = 70)
  (h3 : total_girls = 50)
  (h4 : maple_students = 50)
  (h5 : pine_students = 70)
  (h6 : maple_boys = 25)
  (h7 : total_students = total_boys + total_girls)
  (h8 : total_students = maple_students + pine_students)
  (h9 : maple_students = maple_boys + (total_girls - (pine_students - (total_boys - maple_boys)))) :
  pine_students - (total_boys - maple_boys) = 25 := by
  sorry

end girls_from_pine_l3611_361188


namespace increase_by_percentage_increase_80_by_150_percent_l3611_361111

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 := by sorry

end increase_by_percentage_increase_80_by_150_percent_l3611_361111


namespace inscribed_circle_area_l3611_361171

/-- A circle inscribed in a right triangle with specific properties -/
structure InscribedCircle (A B C X Y : ℝ × ℝ) :=
  (right_angle : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0)
  (ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6)
  (tangent_x : (X.1 - A.1) * (B.1 - A.1) + (X.2 - A.2) * (B.2 - A.2) = 0)
  (tangent_y : (Y.1 - A.1) * (C.1 - A.1) + (Y.2 - A.2) * (C.2 - A.2) = 0)
  (opposite_on_bc : ∃ (X' Y' : ℝ × ℝ), 
    (X'.1 - B.1) * (C.1 - B.1) + (X'.2 - B.2) * (C.2 - B.2) = 0 ∧
    (Y'.1 - B.1) * (C.1 - B.1) + (Y'.2 - B.2) * (C.2 - B.2) = 0 ∧
    (X'.1 - X.1)^2 + (X'.2 - X.2)^2 = (Y'.1 - Y.1)^2 + (Y'.2 - Y.2)^2)

/-- The area of the portion of the circle outside the triangle is π - 2 -/
theorem inscribed_circle_area (A B C X Y : ℝ × ℝ) 
  (h : InscribedCircle A B C X Y) : 
  ∃ (r : ℝ), r > 0 ∧ π * r^2 / 4 - r^2 / 2 = π - 2 := by
  sorry

end inscribed_circle_area_l3611_361171


namespace gain_percent_calculation_l3611_361158

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h : 50 * cost_price = 32 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 56.25 := by
  sorry

end gain_percent_calculation_l3611_361158


namespace andy_initial_candies_l3611_361164

/-- The number of candies each person has initially and after distribution --/
structure CandyDistribution where
  billy_initial : ℕ
  caleb_initial : ℕ
  andy_initial : ℕ
  father_bought : ℕ
  billy_received : ℕ
  caleb_received : ℕ
  andy_final_diff : ℕ

/-- Theorem stating that Andy initially took 9 candies --/
theorem andy_initial_candies (d : CandyDistribution) 
  (h1 : d.billy_initial = 6)
  (h2 : d.caleb_initial = 11)
  (h3 : d.father_bought = 36)
  (h4 : d.billy_received = 8)
  (h5 : d.caleb_received = 11)
  (h6 : d.andy_final_diff = 4)
  : d.andy_initial = 9 := by
  sorry


end andy_initial_candies_l3611_361164


namespace unique_invalid_triangle_l3611_361151

/-- Represents the ratio of altitudes of a triangle -/
structure AltitudeRatio where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a triangle with given side lengths satisfies the triangle inequality -/
def satisfiesTriangleInequality (x y z : ℚ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

/-- Converts altitude ratios to side length ratios -/
def toSideLengthRatio (ar : AltitudeRatio) : (ℚ × ℚ × ℚ) :=
  (1 / ar.a, 1 / ar.b, 1 / ar.c)

/-- Theorem stating that among the given altitude ratios, only 1:2:3 violates the triangle inequality -/
theorem unique_invalid_triangle (ar : AltitudeRatio) : 
  (ar = ⟨1, 1, 2⟩ ∨ ar = ⟨1, 2, 3⟩ ∨ ar = ⟨2, 3, 4⟩ ∨ ar = ⟨3, 4, 5⟩) →
  (¬satisfiesTriangleInequality (toSideLengthRatio ar).1 (toSideLengthRatio ar).2.1 (toSideLengthRatio ar).2.2 ↔ ar = ⟨1, 2, 3⟩) :=
sorry

end unique_invalid_triangle_l3611_361151


namespace total_skips_is_450_l3611_361180

/-- The number of times Bob can skip a rock -/
def bob_skips : ℕ := 12

/-- The number of times Jim can skip a rock -/
def jim_skips : ℕ := 15

/-- The number of times Sally can skip a rock -/
def sally_skips : ℕ := 18

/-- The number of rocks each person skipped -/
def rocks_skipped : ℕ := 10

/-- The total number of skips for all three people -/
def total_skips : ℕ := bob_skips * rocks_skipped + jim_skips * rocks_skipped + sally_skips * rocks_skipped

theorem total_skips_is_450 : total_skips = 450 := by
  sorry

end total_skips_is_450_l3611_361180


namespace quadratic_roots_large_difference_l3611_361194

theorem quadratic_roots_large_difference :
  ∃ (p q p' q' u v u' v' : ℝ),
    (u > v) ∧ (u' > v') ∧
    (u^2 + p*u + q = 0) ∧
    (v^2 + p*v + q = 0) ∧
    (u'^2 + p'*u' + q' = 0) ∧
    (v'^2 + p'*v' + q' = 0) ∧
    (|p' - p| < 0.01) ∧
    (|q' - q| < 0.01) ∧
    (|u' - u| > 10000) :=
by sorry

end quadratic_roots_large_difference_l3611_361194


namespace largest_integer_less_than_100_with_remainder_5_mod_6_l3611_361153

theorem largest_integer_less_than_100_with_remainder_5_mod_6 :
  ∀ n : ℕ, n < 100 ∧ n % 6 = 5 → n ≤ 99 :=
by
  sorry

end largest_integer_less_than_100_with_remainder_5_mod_6_l3611_361153


namespace symmetric_points_product_l3611_361122

/-- 
If point A (2008, y) and point B (x, -1) are symmetric about the origin,
then xy = -2008.
-/
theorem symmetric_points_product (x y : ℝ) : 
  (2008 = -x ∧ y = 1) → x * y = -2008 := by
  sorry

end symmetric_points_product_l3611_361122


namespace farm_ploughing_problem_l3611_361152

/-- Calculates the remaining area to be ploughed given the total area, planned and actual ploughing rates, and additional days worked. -/
def remaining_area (total_area planned_rate actual_rate extra_days : ℕ) : ℕ :=
  let planned_days := total_area / planned_rate
  let actual_days := planned_days + extra_days
  let ploughed_area := actual_rate * actual_days
  total_area - ploughed_area

/-- Theorem stating that given the specific conditions of the farm problem, the remaining area to be ploughed is 40 hectares. -/
theorem farm_ploughing_problem :
  remaining_area 720 120 85 2 = 40 := by
  sorry

end farm_ploughing_problem_l3611_361152


namespace additive_inverse_equation_l3611_361176

theorem additive_inverse_equation (x : ℝ) : (6 * x - 12 = -(4 + 2 * x)) → x = 1 := by
  sorry

end additive_inverse_equation_l3611_361176


namespace positive_real_inequalities_l3611_361157

/-- Given positive real numbers a and b, prove two inequalities based on given conditions -/
theorem positive_real_inequalities (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b = 1 → (1 + 1/a) * (1 + 1/b) ≥ 9) ∧
  (2*a + b = a*b → a + b ≥ 3 + 2*Real.sqrt 2) := by
  sorry

end positive_real_inequalities_l3611_361157


namespace sine_cosine_inequality_l3611_361183

theorem sine_cosine_inequality (x : ℝ) (h : Real.sin x + Real.cos x ≤ 0) :
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := by
  sorry

end sine_cosine_inequality_l3611_361183


namespace initial_men_is_ten_l3611_361103

/-- The initial number of men in the camp -/
def initial_men : ℕ := sorry

/-- The number of days the food lasts for the initial number of men -/
def initial_days : ℕ := 20

/-- The number of additional men that join the camp -/
def additional_men : ℕ := 30

/-- The number of days the food lasts after additional men join -/
def final_days : ℕ := 5

/-- The total amount of food available -/
def total_food : ℕ := initial_men * initial_days

/-- Theorem stating that the initial number of men is 10 -/
theorem initial_men_is_ten : initial_men = 10 := by
  have h1 : total_food = (initial_men + additional_men) * final_days := sorry
  sorry

end initial_men_is_ten_l3611_361103


namespace unique_solution_is_negation_f_is_bijective_l3611_361197

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (y + 1) * f x + f (x * f y + f (x + y)) = y

/-- The main theorem stating that f(x) = -x is the unique solution. -/
theorem unique_solution_is_negation (f : ℝ → ℝ) 
    (h : SatisfiesFunctionalEquation f) : 
    f = fun x ↦ -x := by
  sorry

/-- f is bijective -/
theorem f_is_bijective (f : ℝ → ℝ) 
    (h : SatisfiesFunctionalEquation f) : 
    Function.Bijective f := by
  sorry

end unique_solution_is_negation_f_is_bijective_l3611_361197


namespace triangle_circle_area_relation_l3611_361117

theorem triangle_circle_area_relation (a b c : ℝ) (A B C : ℝ) : 
  a = 13 ∧ b = 14 ∧ c = 15 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  C ≥ A ∧ C ≥ B →
  (a + b + c) / 2 = 21 →
  Real.sqrt (21 * (21 - a) * (21 - b) * (21 - c)) = 84 →
  A + B + 84 = C := by
sorry

end triangle_circle_area_relation_l3611_361117


namespace x_equation_value_l3611_361132

theorem x_equation_value (x : ℝ) (h : x + 1/x = 3) :
  x^10 - 6*x^6 + x^2 = -328*x^2 := by
sorry

end x_equation_value_l3611_361132


namespace number_with_percentage_increase_l3611_361192

theorem number_with_percentage_increase : ∃ x : ℝ, x + 0.35 * x = x + 150 := by
  sorry

end number_with_percentage_increase_l3611_361192


namespace insufficient_payment_l3611_361116

def egg_price : ℝ := 3
def pancake_price : ℝ := 2
def cocoa_price : ℝ := 2
def croissant_price : ℝ := 1
def tax_rate : ℝ := 0.07

def initial_order_cost : ℝ := 4 * egg_price + 3 * pancake_price + 5 * cocoa_price + 2 * croissant_price

def additional_order_cost : ℝ := 2 * 3 * pancake_price + 3 * cocoa_price

def total_cost_before_tax : ℝ := initial_order_cost + additional_order_cost

def total_cost_with_tax : ℝ := total_cost_before_tax * (1 + tax_rate)

def payment : ℝ := 50

theorem insufficient_payment : total_cost_with_tax > payment ∧ 
  total_cost_with_tax - payment = 1.36 := by sorry

end insufficient_payment_l3611_361116


namespace no_solution_in_interval_l3611_361191

theorem no_solution_in_interval : 
  ¬ ∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ (3 * x - 2) ≥ 3 * (12 - 3 * x) := by
  sorry

end no_solution_in_interval_l3611_361191


namespace ball_distribution_l3611_361131

theorem ball_distribution (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) :
  (Nat.choose (n + k - 1 - k) (k - 1)) = 10 := by
  sorry

end ball_distribution_l3611_361131


namespace simplify_expression_l3611_361135

theorem simplify_expression :
  (6 * 10^7) * (2 * 10^3)^2 / (4 * 10^4) = 6 * 10^9 := by sorry

end simplify_expression_l3611_361135


namespace social_gathering_attendance_l3611_361198

theorem social_gathering_attendance
  (num_men : ℕ)
  (women_per_man : ℕ)
  (men_per_woman : ℕ)
  (h_num_men : num_men = 15)
  (h_women_per_man : women_per_man = 4)
  (h_men_per_woman : men_per_woman = 3) :
  (num_men * women_per_man) / men_per_woman = 20 :=
by
  sorry

end social_gathering_attendance_l3611_361198


namespace spider_cylinder_ratio_l3611_361110

/-- In a cylindrical room, a spider can reach the opposite point on the floor
    by two paths of equal length. This theorem proves the ratio of the cylinder's
    height to its diameter given these conditions. -/
theorem spider_cylinder_ratio (m r : ℝ) (h_positive : m > 0 ∧ r > 0) :
  (m + 2*r = Real.sqrt (m^2 + (r*Real.pi)^2)) →
  m / (2*r) = (Real.pi^2 - 4) / 8 := by
  sorry

#check spider_cylinder_ratio

end spider_cylinder_ratio_l3611_361110


namespace luncheon_attendance_l3611_361154

/-- A luncheon problem -/
theorem luncheon_attendance (total_invited : ℕ) (tables_needed : ℕ) (capacity_per_table : ℕ)
  (h1 : total_invited = 45)
  (h2 : tables_needed = 5)
  (h3 : capacity_per_table = 2) :
  total_invited - (tables_needed * capacity_per_table) = 35 := by
  sorry

end luncheon_attendance_l3611_361154


namespace border_material_length_l3611_361169

/-- Given a circular table top with an area of 616 square inches,
    calculate the length of border material needed to cover the circumference
    plus an additional 3 inches, using π ≈ 22/7. -/
theorem border_material_length : 
  let table_area : ℝ := 616
  let π_approx : ℝ := 22 / 7
  let radius : ℝ := Real.sqrt (table_area / π_approx)
  let circumference : ℝ := 2 * π_approx * radius
  let border_length : ℝ := circumference + 3
  border_length = 91 := by
  sorry

end border_material_length_l3611_361169


namespace largest_base_not_18_l3611_361170

/-- Represents a number in a given base as a list of digits -/
def Digits := List Nat

/-- Calculates the sum of digits -/
def sum_of_digits (digits : Digits) : Nat :=
  digits.sum

/-- Converts a number to its representation in a given base -/
def to_base (n : Nat) (base : Nat) : Digits :=
  sorry

theorem largest_base_not_18 :
  ∃ (max_base : Nat),
    (sum_of_digits (to_base (12^3) 10) = 18) ∧
    (12^3 = 1728) ∧
    (∀ b > 10, to_base (12^3) b = to_base 1728 b) ∧
    (to_base (12^3) 9 = [1, 4, 6, 7]) ∧
    (to_base (12^3) 8 = [1, 3, 7, 6]) ∧
    (∀ b > max_base, sum_of_digits (to_base (12^3) b) = 18) ∧
    (sum_of_digits (to_base (12^3) max_base) ≠ 18) ∧
    max_base = 8 :=
  sorry

end largest_base_not_18_l3611_361170


namespace lucas_chocolate_theorem_l3611_361189

/-- Represents the number of pieces of chocolate candy Lucas makes for each student. -/
def pieces_per_student : ℕ := 4

/-- Represents the total number of pieces of chocolate candy Lucas made last Monday. -/
def total_pieces_last_monday : ℕ := 40

/-- Represents the number of students who will not be coming to class this upcoming Monday. -/
def absent_students : ℕ := 3

/-- Calculates the number of pieces of chocolate candy Lucas will make for his class on the upcoming Monday. -/
def pieces_for_upcoming_monday : ℕ :=
  ((total_pieces_last_monday / pieces_per_student) - absent_students) * pieces_per_student

/-- Theorem stating that Lucas will make 28 pieces of chocolate candy for his class on the upcoming Monday. -/
theorem lucas_chocolate_theorem :
  pieces_for_upcoming_monday = 28 := by sorry

end lucas_chocolate_theorem_l3611_361189


namespace currency_exchange_problem_l3611_361167

def exchange_rate : ℚ := 8 / 6

theorem currency_exchange_problem (d : ℕ) :
  (d : ℚ) * exchange_rate - 96 = d →
  d = 288 := by sorry

end currency_exchange_problem_l3611_361167


namespace perpendicular_line_through_point_l3611_361114

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y => y = -2 * x + 1
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y ↔ y = (1/2) * x - 3/2) →  -- Slope-intercept form of L1
  (L2 P.1 P.2) →                          -- L2 passes through P
  ((-2) * (1/2) = -1) →                   -- Slopes are negative reciprocals
  ∀ x y, L1 x y → L2 x y → (x - P.1) * (x - P.1) + (y - P.2) * (y - P.2) ≠ 0 →
    (x - P.1) * (x - 2) + (y - P.2) * (y - (-3)) = 0 -- Perpendicular condition
  := by sorry

end perpendicular_line_through_point_l3611_361114


namespace first_equation_is_double_root_second_equation_values_l3611_361146

/-- Definition of a double root equation -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 2 * x

/-- The equation x^2 - 3x + 2 = 0 is a double root equation -/
theorem first_equation_is_double_root : is_double_root_equation 1 (-3) 2 :=
sorry

/-- For ax^2 + bx - 6 = 0, if it's a double root equation with one root as 2,
    then a and b have specific values -/
theorem second_equation_values (a b : ℝ) :
  is_double_root_equation a b (-6) ∧ (∃ x : ℝ, a * x^2 + b * x - 6 = 0 ∧ x = 2) →
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) :=
sorry

end first_equation_is_double_root_second_equation_values_l3611_361146


namespace hyperbola_min_value_l3611_361143

theorem hyperbola_min_value (x y : ℝ) : 
  x^2 / 4 - y^2 = 1 → (∀ z w : ℝ, z^2 / 4 - w^2 = 1 → 3*x^2 - 2*y ≤ 3*z^2 - 2*w) ∧ (∃ a b : ℝ, a^2 / 4 - b^2 = 1 ∧ 3*a^2 - 2*b = 143/12) := by
  sorry

end hyperbola_min_value_l3611_361143


namespace corner_square_length_l3611_361195

/-- Given a rectangular sheet of dimensions 48 m x 36 m, if squares of side length x
    are cut from each corner to form an open box with volume 5120 m³, then x = 8 meters. -/
theorem corner_square_length (x : ℝ) : 
  x > 0 ∧ x < 24 ∧ x < 18 →
  (48 - 2*x) * (36 - 2*x) * x = 5120 →
  x = 8 := by
sorry

end corner_square_length_l3611_361195


namespace chocolate_box_problem_l3611_361162

theorem chocolate_box_problem (total : ℕ) (p_peanut : ℚ) : 
  total = 50 → p_peanut = 64/100 → 
  ∃ (caramels nougats truffles peanuts : ℕ),
    nougats = 2 * caramels ∧
    truffles = caramels + 6 ∧
    caramels + nougats + truffles + peanuts = total ∧
    p_peanut = peanuts / total ∧
    caramels = 3 := by
  sorry

end chocolate_box_problem_l3611_361162


namespace group_b_more_stable_l3611_361133

-- Define the structure for a group's statistics
structure GroupStats where
  mean : ℝ
  variance : ℝ

-- Define the stability comparison function
def more_stable (a b : GroupStats) : Prop :=
  a.variance < b.variance

-- Theorem statement
theorem group_b_more_stable (group_a group_b : GroupStats) 
  (h1 : group_a.mean = group_b.mean)
  (h2 : group_a.variance = 36)
  (h3 : group_b.variance = 30) :
  more_stable group_b group_a :=
by
  sorry

end group_b_more_stable_l3611_361133


namespace taxi_ride_distance_l3611_361155

/-- Calculates the distance of a taxi ride given the fare structure and total fare -/
theorem taxi_ride_distance 
  (initial_fare : ℚ) 
  (initial_distance : ℚ) 
  (additional_fare : ℚ) 
  (additional_distance : ℚ) 
  (total_fare : ℚ) 
  (h1 : initial_fare = 2)
  (h2 : initial_distance = 1/5)
  (h3 : additional_fare = 3/5)
  (h4 : additional_distance = 1/5)
  (h5 : total_fare = 127/5) : 
  ∃ (distance : ℚ), distance = 8 ∧ 
    total_fare = initial_fare + (distance - initial_distance) / additional_distance * additional_fare :=
by sorry

end taxi_ride_distance_l3611_361155


namespace total_bricks_used_l3611_361144

/-- The number of walls being built -/
def number_of_walls : ℕ := 4

/-- The number of bricks in a single row of a wall -/
def bricks_per_row : ℕ := 60

/-- The number of rows in each wall -/
def rows_per_wall : ℕ := 100

/-- Theorem stating the total number of bricks used for all walls -/
theorem total_bricks_used :
  number_of_walls * bricks_per_row * rows_per_wall = 24000 := by
  sorry

end total_bricks_used_l3611_361144


namespace alice_box_height_l3611_361125

/-- The height of the box Alice needs to reach the light bulb -/
def box_height (ceiling_height room_height alice_height alice_reach light_bulb_distance shelf_distance : ℝ) : ℝ :=
  ceiling_height - light_bulb_distance - (alice_height + alice_reach)

/-- Proof that Alice needs a 75 cm box to reach the light bulb -/
theorem alice_box_height :
  let ceiling_height : ℝ := 300  -- cm
  let room_height : ℝ := 300     -- cm
  let alice_height : ℝ := 160    -- cm
  let alice_reach : ℝ := 50      -- cm
  let light_bulb_distance : ℝ := 15  -- cm from ceiling
  let shelf_distance : ℝ := 10   -- cm below light bulb
  box_height ceiling_height room_height alice_height alice_reach light_bulb_distance shelf_distance = 75 := by
  sorry


end alice_box_height_l3611_361125


namespace monotonicity_when_a_eq_1_extreme_value_two_zero_points_l3611_361130

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2 + (2*a - 1) * x

-- State the theorems
theorem monotonicity_when_a_eq_1 :
  ∀ x y, 0 < x ∧ x < 1 ∧ 0 < y ∧ 1 < y → f 1 x < f 1 1 ∧ f 1 1 > f 1 y := by sorry

theorem extreme_value :
  ∀ a, a > 0 → ∃ x, x > 0 ∧ ∀ y, y > 0 → f a y ≤ f a x ∧ f a x = a * (Real.log a + a - 1) := by sorry

theorem two_zero_points :
  ∀ a, (∃ x y, 0 < x ∧ x < y ∧ f a x = 0 ∧ f a y = 0) ↔ a > 1 := by sorry

end

end monotonicity_when_a_eq_1_extreme_value_two_zero_points_l3611_361130


namespace factorize_quadratic_factorize_cubic_factorize_quartic_l3611_361160

-- Problem 1
theorem factorize_quadratic (m : ℝ) : m^2 + 4*m + 4 = (m + 2)^2 := by sorry

-- Problem 2
theorem factorize_cubic (a b : ℝ) : a^2*b - 4*a*b^2 + 3*b^3 = b*(a-b)*(a-3*b) := by sorry

-- Problem 3
theorem factorize_quartic (x y : ℝ) : (x^2 + y^2)^2 - 4*x^2*y^2 = (x + y)^2 * (x - y)^2 := by sorry

end factorize_quadratic_factorize_cubic_factorize_quartic_l3611_361160


namespace simplify_expression_l3611_361156

theorem simplify_expression : (5 * 10^10) / (2 * 10^4 * 10^2) = 25000 := by
  sorry

end simplify_expression_l3611_361156


namespace rent_increase_effect_rent_problem_l3611_361179

theorem rent_increase_effect (num_friends : ℕ) (initial_avg_rent : ℚ) 
  (increased_rent : ℚ) (increase_percentage : ℚ) : ℚ :=
  let total_initial_rent := num_friends * initial_avg_rent
  let rent_increase := increased_rent * increase_percentage
  let new_total_rent := total_initial_rent + rent_increase
  let new_avg_rent := new_total_rent / num_friends
  new_avg_rent

theorem rent_problem :
  rent_increase_effect 4 800 1600 (1/5) = 880 := by sorry

end rent_increase_effect_rent_problem_l3611_361179


namespace line_equation_through_two_points_l3611_361120

/-- Given two points A and B on the line 2x + 3y = 4, 
    prove that this is the equation of the line passing through these points. -/
theorem line_equation_through_two_points 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 2 * x₁ + 3 * y₁ = 4) 
  (h₂ : 2 * x₂ + 3 * y₂ = 4) : 
  ∀ (x y : ℝ), (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) → 2 * x + 3 * y = 4 := by
sorry

end line_equation_through_two_points_l3611_361120


namespace edward_book_purchase_l3611_361172

/-- Given that Edward spent $6 on books and each book cost $3, prove that he bought 2 books. -/
theorem edward_book_purchase (total_spent : ℕ) (cost_per_book : ℕ) (h1 : total_spent = 6) (h2 : cost_per_book = 3) :
  total_spent / cost_per_book = 2 := by
  sorry

end edward_book_purchase_l3611_361172


namespace cone_lateral_surface_area_l3611_361196

theorem cone_lateral_surface_area (slant_height height : Real) 
  (h1 : slant_height = 15)
  (h2 : height = 9) :
  let radius := Real.sqrt (slant_height^2 - height^2)
  π * radius * slant_height = 180 * π := by
  sorry

end cone_lateral_surface_area_l3611_361196


namespace factorization_of_5a_cubed_minus_125a_l3611_361140

theorem factorization_of_5a_cubed_minus_125a (a : ℝ) :
  5 * a^3 - 125 * a = 5 * a * (a + 5) * (a - 5) := by
  sorry

end factorization_of_5a_cubed_minus_125a_l3611_361140


namespace apple_distribution_l3611_361136

theorem apple_distribution (total_apples : ℕ) (total_bags : ℕ) (x : ℕ) :
  total_apples = 109 →
  total_bags = 20 →
  (∃ k : ℕ, k * x + (total_bags - k) * 3 = total_apples ∧ 0 < k ∧ k ≤ total_bags) →
  (x = 10 ∨ x = 52) :=
by sorry

end apple_distribution_l3611_361136


namespace fraction_evaluation_l3611_361123

theorem fraction_evaluation : 
  (1 / 5 + 1 / 3) / (3 / 7 - 1 / 4) = 224 / 75 := by
  sorry

end fraction_evaluation_l3611_361123


namespace framed_painting_ratio_l3611_361159

theorem framed_painting_ratio : 
  let painting_width : ℝ := 18
  let painting_height : ℝ := 24
  let frame_side_width : ℝ := 3  -- This is derived from solving the equation in the solution
  let frame_top_bottom_width : ℝ := 2 * frame_side_width
  let framed_width : ℝ := painting_width + 2 * frame_side_width
  let framed_height : ℝ := painting_height + 2 * frame_top_bottom_width
  let frame_area : ℝ := framed_width * framed_height - painting_width * painting_height
  frame_area = painting_width * painting_height →
  (min framed_width framed_height) / (max framed_width framed_height) = 2 / 3 :=
by
  sorry

end framed_painting_ratio_l3611_361159


namespace shaded_area_fraction_l3611_361181

/-- RegularOctagon represents a regular octagon with center O and vertices A to H -/
structure RegularOctagon where
  O : Point
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  G : Point
  H : Point

/-- Given a regular octagon, returns the area of the specified region -/
def shaded_area (octagon : RegularOctagon) : ℝ :=
  sorry

/-- The total area of the regular octagon -/
def total_area (octagon : RegularOctagon) : ℝ :=
  sorry

/-- Theorem stating that the shaded area is 5/8 of the total area -/
theorem shaded_area_fraction (octagon : RegularOctagon) :
  shaded_area octagon / total_area octagon = 5/8 := by
  sorry

end shaded_area_fraction_l3611_361181


namespace one_third_minus_0_3333_l3611_361177

-- Define 0.3333 as a rational number
def decimal_0_3333 : ℚ := 3333 / 10000

-- State the theorem
theorem one_third_minus_0_3333 : (1 : ℚ) / 3 - decimal_0_3333 = 1 / 10000 := by
  sorry

end one_third_minus_0_3333_l3611_361177


namespace calculate_expression_l3611_361185

theorem calculate_expression : (1/3)⁻¹ + (2023 - Real.pi)^0 - Real.sqrt 12 * Real.sin (π/3) = 1 := by
  sorry

end calculate_expression_l3611_361185


namespace blue_highlighters_count_l3611_361127

/-- Given the number of highlighters in a teacher's desk, calculate the number of blue highlighters. -/
theorem blue_highlighters_count 
  (total : ℕ) 
  (pink : ℕ) 
  (yellow : ℕ) 
  (h1 : total = 11) 
  (h2 : pink = 4) 
  (h3 : yellow = 2) : 
  total - pink - yellow = 5 := by
  sorry

#check blue_highlighters_count

end blue_highlighters_count_l3611_361127


namespace seashells_sum_l3611_361168

/-- The number of seashells found by Mary and Keith -/
def total_seashells (mary_seashells keith_seashells : ℕ) : ℕ :=
  mary_seashells + keith_seashells

/-- Theorem stating that the total number of seashells is the sum of Mary's and Keith's seashells -/
theorem seashells_sum (mary_seashells keith_seashells : ℕ) :
  total_seashells mary_seashells keith_seashells = mary_seashells + keith_seashells :=
by sorry

end seashells_sum_l3611_361168


namespace quadratic_properties_l3611_361178

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) :
  f a b c (-2) = -11 →
  f a b c (-1) = 9 →
  f a b c 0 = 21 →
  f a b c 3 = 9 →
  (∃ (x_max : ℝ), x_max = 0 ∧ ∀ x, f a b c x ≤ f a b c x_max) ∧
  (∃ (x_sym : ℝ), x_sym = 1 ∧ ∀ x, f a b c (x_sym - x) = f a b c (x_sym + x)) ∧
  (∃ (x : ℝ), 3 < x ∧ x < 4 ∧ f a b c x = 0) ∧
  (∀ x, f a b c x > 21 ↔ 0 < x ∧ x < 2) :=
by sorry

end quadratic_properties_l3611_361178


namespace max_term_binomial_expansion_l3611_361166

theorem max_term_binomial_expansion :
  let n : ℕ := 213
  let x : ℝ := Real.sqrt 5
  let term (k : ℕ) := (n.choose k) * x^k
  ∃ k_max : ℕ, k_max = 147 ∧ ∀ k : ℕ, k ≤ n → term k ≤ term k_max :=
by sorry

end max_term_binomial_expansion_l3611_361166


namespace odot_inequality_iff_l3611_361149

-- Define the ⊙ operation
def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem odot_inequality_iff (x : ℝ) : odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end odot_inequality_iff_l3611_361149


namespace paper_boats_problem_l3611_361184

theorem paper_boats_problem (initial_boats : ℕ) : 
  (initial_boats : ℝ) * 0.8 - 2 = 22 → initial_boats = 30 := by
  sorry

end paper_boats_problem_l3611_361184


namespace neighborhood_vehicles_l3611_361104

theorem neighborhood_vehicles (total : Nat) (both : Nat) (car : Nat) (bike_only : Nat)
  (h1 : total = 90)
  (h2 : both = 16)
  (h3 : car = 44)
  (h4 : bike_only = 35) :
  total - (car + bike_only) = 11 := by
  sorry

end neighborhood_vehicles_l3611_361104


namespace unique_intersection_point_l3611_361102

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-3, -3) := by
  sorry

end unique_intersection_point_l3611_361102
