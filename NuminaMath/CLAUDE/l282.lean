import Mathlib

namespace xy_term_vanishes_l282_28227

/-- The polynomial in question -/
def polynomial (k x y : ℝ) : ℝ := x^2 + (k-1)*x*y - 3*y^2 - 2*x*y - 5

/-- The coefficient of xy in the polynomial -/
def xy_coefficient (k : ℝ) : ℝ := k - 3

theorem xy_term_vanishes (k : ℝ) :
  xy_coefficient k = 0 ↔ k = 3 := by sorry

end xy_term_vanishes_l282_28227


namespace different_color_probability_is_two_thirds_l282_28273

/-- The number of possible colors for the shorts -/
def shorts_colors : ℕ := 2

/-- The number of possible colors for the jersey -/
def jersey_colors : ℕ := 3

/-- The total number of possible color combinations -/
def total_combinations : ℕ := shorts_colors * jersey_colors

/-- The number of combinations where the shorts and jersey colors are different -/
def different_color_combinations : ℕ := shorts_colors * (jersey_colors - 1)

/-- The probability that the shorts will be a different color than the jersey -/
def different_color_probability : ℚ := different_color_combinations / total_combinations

theorem different_color_probability_is_two_thirds :
  different_color_probability = 2 / 3 := by sorry

end different_color_probability_is_two_thirds_l282_28273


namespace zack_traveled_to_18_countries_l282_28203

-- Define the number of countries each person traveled to
def george_countries : ℕ := 6
def joseph_countries : ℕ := george_countries / 2
def patrick_countries : ℕ := joseph_countries * 3
def zack_countries : ℕ := patrick_countries * 2

-- Theorem statement
theorem zack_traveled_to_18_countries :
  zack_countries = 18 :=
by
  sorry

end zack_traveled_to_18_countries_l282_28203


namespace special_offer_cost_l282_28263

/-- Represents the cost of a T-shirt in pence -/
def TShirtCost : ℕ := 1650

/-- Represents the savings per T-shirt in pence -/
def SavingsPerShirt : ℕ := 550

/-- Represents the number of T-shirts in the offer -/
def NumShirts : ℕ := 3

/-- Represents the number of T-shirts paid for in the offer -/
def PaidShirts : ℕ := 2

theorem special_offer_cost :
  PaidShirts * TShirtCost = 3300 := by sorry

end special_offer_cost_l282_28263


namespace fitted_bowling_ball_volume_l282_28212

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 40
  let hole1_diameter : ℝ := 2
  let hole2_diameter : ℝ := 4
  let hole3_diameter : ℝ := 4
  let hole_depth : ℝ := 10
  let sphere_volume := (4/3) * π * (sphere_diameter/2)^3
  let hole1_volume := π * (hole1_diameter/2)^2 * hole_depth
  let hole2_volume := π * (hole2_diameter/2)^2 * hole_depth
  let hole3_volume := π * (hole3_diameter/2)^2 * hole_depth
  sphere_volume - (hole1_volume + hole2_volume + hole3_volume) = (31710/3) * π :=
by sorry

end fitted_bowling_ball_volume_l282_28212


namespace jerusha_earnings_l282_28293

theorem jerusha_earnings (L : ℝ) : 
  L + 4 * L = 85 → 4 * L = 68 := by
  sorry

end jerusha_earnings_l282_28293


namespace quadratic_polynomial_roots_l282_28220

theorem quadratic_polynomial_roots (x₁ x₂ : ℝ) (h_sum : x₁ + x₂ = 8) (h_product : x₁ * x₂ = 16) :
  x₁ * x₂ = 16 ∧ x₁ + x₂ = 8 ↔ x₁^2 - 8*x₁ + 16 = 0 ∧ x₂^2 - 8*x₂ + 16 = 0 :=
by sorry

end quadratic_polynomial_roots_l282_28220


namespace weekday_classes_count_l282_28225

/-- Represents the Diving Club's class schedule --/
structure DivingClub where
  weekdayClasses : ℕ
  weekendClassesPerDay : ℕ
  peoplePerClass : ℕ
  totalWeeks : ℕ
  totalPeople : ℕ

/-- Calculates the total number of people who can take classes --/
def totalCapacity (club : DivingClub) : ℕ :=
  (club.weekdayClasses * club.totalWeeks + 
   club.weekendClassesPerDay * 2 * club.totalWeeks) * club.peoplePerClass

/-- Theorem stating the number of weekday classes --/
theorem weekday_classes_count (club : DivingClub) 
  (h1 : club.weekendClassesPerDay = 4)
  (h2 : club.peoplePerClass = 5)
  (h3 : club.totalWeeks = 3)
  (h4 : club.totalPeople = 270)
  (h5 : totalCapacity club = club.totalPeople) :
  club.weekdayClasses = 10 := by
  sorry

#check weekday_classes_count

end weekday_classes_count_l282_28225


namespace onion_problem_l282_28244

theorem onion_problem (initial : ℕ) (removed : ℕ) : 
  initial + 4 - removed + 9 = initial + 8 → removed = 5 := by
  sorry

end onion_problem_l282_28244


namespace min_value_quadratic_form_l282_28265

theorem min_value_quadratic_form (x y z : ℝ) :
  x^2 + x*y + y^2 + y*z + z^2 ≥ 0 ∧
  (x^2 + x*y + y^2 + y*z + z^2 = 0 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end min_value_quadratic_form_l282_28265


namespace bridget_apples_theorem_l282_28289

/-- The number of apples Bridget originally bought -/
def original_apples : ℕ := 14

/-- The number of apples Bridget gives to Ann -/
def apples_to_ann : ℕ := original_apples / 2

/-- The number of apples Bridget gives to Cassie -/
def apples_to_cassie : ℕ := 5

/-- The number of apples Bridget keeps for herself -/
def apples_for_bridget : ℕ := 2

theorem bridget_apples_theorem :
  original_apples = apples_to_ann * 2 ∧
  original_apples = apples_to_ann + apples_to_cassie + apples_for_bridget :=
by sorry

end bridget_apples_theorem_l282_28289


namespace remainder_theorem_l282_28240

theorem remainder_theorem (r : ℝ) : (r^13 + 1) % (r - 1) = 2 := by sorry

end remainder_theorem_l282_28240


namespace ribbon_boxes_theorem_l282_28210

theorem ribbon_boxes_theorem (total_ribbon : ℝ) (ribbon_per_box : ℝ) (leftover : ℝ) :
  total_ribbon = 12.5 ∧ 
  ribbon_per_box = 1.75 ∧ 
  leftover = 0.3 → 
  ⌊total_ribbon / (ribbon_per_box + leftover)⌋ = 6 :=
by sorry

end ribbon_boxes_theorem_l282_28210


namespace system_solution_l282_28211

theorem system_solution :
  ∃ (A B C D : ℚ),
    A = 1/42 ∧
    B = 1/7 ∧
    C = 1/3 ∧
    D = 1/2 ∧
    A = B * C * D ∧
    A + B = C * D ∧
    A + B + C = D ∧
    A + B + C + D = 1 := by
  sorry

end system_solution_l282_28211


namespace rectangle_width_equality_l282_28277

/-- Given two rectangles of equal area, where one rectangle measures 5 inches by 24 inches
    and the other rectangle is 4 inches long, prove that the width of the second rectangle
    is 30 inches. -/
theorem rectangle_width_equality (area carol_length carol_width jordan_length : ℝ)
    (h1 : area = carol_length * carol_width)
    (h2 : carol_length = 5)
    (h3 : carol_width = 24)
    (h4 : jordan_length = 4)
    (h5 : area = jordan_length * (area / jordan_length)) :
    area / jordan_length = 30 := by
  sorry

end rectangle_width_equality_l282_28277


namespace cost_price_per_metre_l282_28208

/-- The cost price of one metre of cloth given the selling price, quantity, and profit per metre -/
theorem cost_price_per_metre 
  (total_metres : ℕ) 
  (total_selling_price : ℚ) 
  (profit_per_metre : ℚ) 
  (h1 : total_metres = 85)
  (h2 : total_selling_price = 8925)
  (h3 : profit_per_metre = 15) :
  (total_selling_price - total_metres * profit_per_metre) / total_metres = 90 := by
  sorry

end cost_price_per_metre_l282_28208


namespace investment_growth_equation_l282_28286

/-- Represents the average growth rate equation for a two-year investment period -/
theorem investment_growth_equation (initial_investment : ℝ) (final_investment : ℝ) (x : ℝ) :
  initial_investment = 20000 →
  final_investment = 25000 →
  20 * (1 + x)^2 = 25 :=
by sorry

end investment_growth_equation_l282_28286


namespace student_failed_marks_l282_28271

def total_marks : ℕ := 300
def passing_percentage : ℚ := 60 / 100
def student_marks : ℕ := 160

theorem student_failed_marks :
  (passing_percentage * total_marks : ℚ).ceil - student_marks = 20 := by
  sorry

end student_failed_marks_l282_28271


namespace inequality_solution_range_l282_28238

theorem inequality_solution_range (m : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - 1| < m) → m > 2 :=
by sorry

end inequality_solution_range_l282_28238


namespace parabola_directrix_l282_28205

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form x^2 = 2py -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem: Given a parabola x^2 = 2py (p > 0) intersected by a line with slope 1 at points A and B,
    if the x-coordinate of the midpoint of AB is 2, then the equation of the directrix is y = -1 -/
theorem parabola_directrix (par : Parabola) (A B : Point) :
  (A.x^2 = 2 * par.p * A.y) →
  (B.x^2 = 2 * par.p * B.y) →
  (B.y - A.y = B.x - A.x) →
  ((A.x + B.x) / 2 = 2) →
  (∀ (x y : ℝ), y = -1 ↔ y = -par.p / 2) :=
by sorry

end parabola_directrix_l282_28205


namespace prime_pythagorean_triple_l282_28252

theorem prime_pythagorean_triple (p m n : ℕ) 
  (hp : Nat.Prime p) 
  (hm : m > 0) 
  (hn : n > 0) 
  (heq : p^2 + m^2 = n^2) : 
  m > p := by
  sorry

end prime_pythagorean_triple_l282_28252


namespace triangle_area_l282_28202

/-- The area of the triangle formed by the x-axis, y-axis, and the line 3x + ay = 12 is 3/2 square units. -/
theorem triangle_area (a : ℝ) : 
  let x_intercept : ℝ := 12 / 3
  let y_intercept : ℝ := 12 / a
  let triangle_area : ℝ := (1 / 2) * x_intercept * y_intercept
  triangle_area = 3 / 2 :=
by sorry

end triangle_area_l282_28202


namespace arithmetic_calculation_l282_28213

theorem arithmetic_calculation : (180 / 6) * 2 + 5 = 65 := by
  sorry

end arithmetic_calculation_l282_28213


namespace geometric_sequence_sum_l282_28207

/-- Given a geometric sequence {a_n} with common ratio q > 0,
    where a_2 = 1 and a_{n+2} + a_{n+1} = 6a_n,
    prove that the sum of the first four terms (S_4) is equal to 15/2. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_q_pos : q > 0)
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_a2 : a 2 = 1)
  (h_relation : ∀ n, a (n + 2) + a (n + 1) = 6 * a n) :
  a 1 + a 2 + a 3 + a 4 = 15 / 2 := by
  sorry

end geometric_sequence_sum_l282_28207


namespace water_drinking_time_l282_28294

/-- Proves that given a goal of drinking 3 liters of water and drinking 500 milliliters every 2 hours, it will take 12 hours to reach the goal. -/
theorem water_drinking_time (goal : ℕ) (intake : ℕ) (frequency : ℕ) (h1 : goal = 3) (h2 : intake = 500) (h3 : frequency = 2) : 
  (goal * 1000) / intake * frequency = 12 := by
  sorry

end water_drinking_time_l282_28294


namespace coffee_stock_problem_l282_28228

/-- Represents the coffee stock problem --/
theorem coffee_stock_problem 
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (new_decaf_percent : ℝ)
  (final_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.4)
  (h3 : new_decaf_percent = 0.6)
  (h4 : final_decaf_percent = 0.44)
  : ∃ (additional_coffee : ℝ),
    additional_coffee = 100 ∧
    (initial_stock * initial_decaf_percent + additional_coffee * new_decaf_percent) / (initial_stock + additional_coffee) = final_decaf_percent :=
by sorry


end coffee_stock_problem_l282_28228


namespace part_one_part_two_l282_28224

-- Define the conditions P and Q
def P (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def Q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem part_one (x : ℝ) (h : P x 1 ∧ Q x) : 2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) 
  (h : ∀ x, ¬(P x a) → ¬(Q x)) 
  (h_not_nec : ∃ x, ¬(P x a) ∧ Q x) : 
  1 < a ∧ a ≤ 2 := by sorry

end part_one_part_two_l282_28224


namespace tan_identity_l282_28249

theorem tan_identity (α β γ n : Real) 
  (h : Real.sin (2 * (α + γ)) = n * Real.sin (2 * β)) : 
  Real.tan (α + β + γ) = (n + 1) / (n - 1) * Real.tan (α - β + γ) := by
  sorry

end tan_identity_l282_28249


namespace sara_movie_purchase_cost_l282_28285

/-- The amount Sara spent on movie theater tickets -/
def theater_ticket_cost : ℚ := 10.62

/-- The number of movie theater tickets Sara bought -/
def number_of_tickets : ℕ := 2

/-- The cost of renting a movie -/
def rental_cost : ℚ := 1.59

/-- The total amount Sara spent on movies -/
def total_spent : ℚ := 36.78

/-- Theorem: Given the conditions, Sara spent $13.95 on buying the movie -/
theorem sara_movie_purchase_cost :
  total_spent - (theater_ticket_cost * number_of_tickets + rental_cost) = 13.95 := by
  sorry

end sara_movie_purchase_cost_l282_28285


namespace expression_simplification_and_evaluation_l282_28258

theorem expression_simplification_and_evaluation :
  ∀ a : ℤ, -Real.sqrt 2 < (a : ℝ) ∧ (a : ℝ) < Real.sqrt 5 →
  (a ≠ -1 ∧ a ≠ 2) →
  ((a - 1 - 3 / (a + 1)) / ((a^2 - 4*a + 4) / (a + 1)) = (a + 2) / (a - 2)) ∧
  ((a + 2) / (a - 2) = -1 ∨ (a + 2) / (a - 2) = -3) :=
by sorry

end expression_simplification_and_evaluation_l282_28258


namespace inequality_range_l282_28226

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x + 3| > a) → a < 5 := by
  sorry

end inequality_range_l282_28226


namespace water_addition_changes_ratio_l282_28248

/-- Given a mixture of alcohol and water, prove that adding 2 liters of water
    changes the ratio from 4:3 to 4:5 when the initial amount of alcohol is 4 liters. -/
theorem water_addition_changes_ratio :
  let initial_alcohol : ℝ := 4
  let initial_water : ℝ := 3
  let water_added : ℝ := 2
  let final_water : ℝ := initial_water + water_added
  let initial_ratio : ℝ := initial_alcohol / initial_water
  let final_ratio : ℝ := initial_alcohol / final_water
  initial_ratio = 4/3 ∧ final_ratio = 4/5 := by
  sorry

#check water_addition_changes_ratio

end water_addition_changes_ratio_l282_28248


namespace meeting_probability_for_seven_steps_l282_28242

/-- Represents a position on the coordinate plane -/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents the possible movements for an object -/
inductive Movement
  | Right
  | Up
  | Left
  | Down

/-- Represents an object on the coordinate plane -/
structure Object where
  position : Position
  allowedMovements : List Movement

/-- Calculates the number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Calculates the number of intersection paths for given number of steps -/
def intersectionPaths (steps : ℕ) : ℕ := sorry

/-- The probability of two objects meeting given their initial positions and movement constraints -/
def meetingProbability (obj1 obj2 : Object) (steps : ℕ) : ℚ := sorry

theorem meeting_probability_for_seven_steps :
  let c : Object := ⟨⟨1, 1⟩, [Movement.Right, Movement.Up]⟩
  let d : Object := ⟨⟨6, 7⟩, [Movement.Left, Movement.Down]⟩
  meetingProbability c d 7 = 1715 / 16384 := by sorry

end meeting_probability_for_seven_steps_l282_28242


namespace ken_to_don_ratio_l282_28274

-- Define the painting rates
def don_rate : ℕ := 3
def ken_rate : ℕ := don_rate + 2
def laura_rate : ℕ := 2 * ken_rate
def kim_rate : ℕ := laura_rate - 3

-- Define the total tiles painted in 15 minutes
def total_tiles : ℕ := 375

-- Theorem statement
theorem ken_to_don_ratio : 
  15 * (don_rate + ken_rate + laura_rate + kim_rate) = total_tiles →
  ken_rate / don_rate = 5 / 3 := by
sorry

end ken_to_don_ratio_l282_28274


namespace smallest_cross_family_bound_l282_28230

/-- A family of subsets A of a finite set X is a cross family if for every subset B of X,
    B is comparable with at least one subset in A. -/
def IsCrossFamily (X : Finset α) (A : Finset (Finset α)) : Prop :=
  ∀ B : Finset α, B ⊆ X → ∃ A' ∈ A, A' ⊆ B ∨ B ⊆ A'

/-- A is the smallest cross family if no proper subfamily of A is a cross family. -/
def IsSmallestCrossFamily (X : Finset α) (A : Finset (Finset α)) : Prop :=
  IsCrossFamily X A ∧ ∀ A' ⊂ A, ¬IsCrossFamily X A'

theorem smallest_cross_family_bound {α : Type*} [DecidableEq α] (X : Finset α) (A : Finset (Finset α)) :
  IsSmallestCrossFamily X A → A.card ≤ Nat.choose X.card (X.card / 2) := by
  sorry

end smallest_cross_family_bound_l282_28230


namespace digit_sum_l282_28236

theorem digit_sum (w x y z : ℕ) : 
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10 →
  y + w = 11 →
  x + y + 1 = 10 →
  w + z + 1 = 11 →
  w + x + y + z = 20 :=
by sorry

end digit_sum_l282_28236


namespace no_solution_iff_a_leq_8_l282_28206

theorem no_solution_iff_a_leq_8 :
  ∀ a : ℝ, (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) ↔ a ≤ 8 :=
by sorry

end no_solution_iff_a_leq_8_l282_28206


namespace copperfield_numbers_l282_28232

theorem copperfield_numbers :
  ∃ (x₁ x₂ x₃ : ℕ) (k₁ k₂ k₃ : ℕ+),
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    x₁ * 3^(k₁.val) = x₁ + 2500 * k₁.val ∧
    x₂ * 3^(k₂.val) = x₂ + 2500 * k₂.val ∧
    x₃ * 3^(k₃.val) = x₃ + 2500 * k₃.val :=
by sorry

end copperfield_numbers_l282_28232


namespace sufficient_not_necessary_l282_28270

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ ¬(a > 0)) := by
sorry

end sufficient_not_necessary_l282_28270


namespace divisor_problem_l282_28204

theorem divisor_problem (n : ℤ) : ∃ (d : ℤ), d = 22 ∧ ∃ (k : ℤ), n = k * d + 12 ∧ ∃ (m : ℤ), 2 * n = 11 * m + 2 :=
by sorry

end divisor_problem_l282_28204


namespace inequality_system_solution_l282_28256

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x - a > 2 ∧ b - 2*x > 0) ↔ (-1 < x ∧ x < 1)) →
  (a + b)^2021 = -1 := by
  sorry

end inequality_system_solution_l282_28256


namespace derivative_at_pi_sixth_l282_28250

theorem derivative_at_pi_sixth (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = Real.cos x - Real.sin x) →
  (∀ x, HasDerivAt f (f' x) x) →
  f' (π/6) = -(1 + Real.sqrt 3) / 2 := by
  sorry

end derivative_at_pi_sixth_l282_28250


namespace divisors_of_2160_l282_28281

def n : ℕ := 2160

-- Define the prime factorization of n
axiom n_factorization : n = 2^4 * 3^3 * 5

-- Define the number of positive divisors
def num_divisors (m : ℕ) : ℕ := sorry

-- Define the sum of positive divisors
def sum_divisors (m : ℕ) : ℕ := sorry

theorem divisors_of_2160 :
  (num_divisors n = 40) ∧ (sum_divisors n = 7440) := by sorry

end divisors_of_2160_l282_28281


namespace round_trip_time_ratio_l282_28290

/-- Proves that for a round trip with given average speeds, the ratio of return to outbound journey times is 3:2 -/
theorem round_trip_time_ratio 
  (distance : ℝ) 
  (speed_to_destination : ℝ) 
  (average_speed_round_trip : ℝ) 
  (h1 : speed_to_destination = 54) 
  (h2 : average_speed_round_trip = 36) 
  (h3 : distance > 0) 
  (h4 : speed_to_destination > 0) 
  (h5 : average_speed_round_trip > 0) : 
  (distance / average_speed_round_trip - distance / speed_to_destination) / (distance / speed_to_destination) = 3 / 2 :=
by sorry

end round_trip_time_ratio_l282_28290


namespace rest_area_distance_l282_28298

theorem rest_area_distance (d : ℝ) : 
  (¬ (d ≥ 8)) →  -- David's statement is false
  (¬ (d ≤ 7)) →  -- Ellen's statement is false
  (¬ (d ≤ 6)) →  -- Frank's statement is false
  (7 < d ∧ d < 8) := by
sorry

end rest_area_distance_l282_28298


namespace standard_deviation_of_commute_times_l282_28243

def commute_times : List ℝ := [12, 8, 10, 11, 9]

theorem standard_deviation_of_commute_times :
  let n : ℕ := commute_times.length
  let mean : ℝ := (commute_times.sum) / n
  let variance : ℝ := (commute_times.map (λ x => (x - mean)^2)).sum / n
  Real.sqrt variance = Real.sqrt 2 := by sorry

end standard_deviation_of_commute_times_l282_28243


namespace third_term_binomial_expansion_l282_28237

theorem third_term_binomial_expansion (x : ℝ) : 
  let n : ℕ := 4
  let a : ℝ := x
  let b : ℝ := 2
  let r : ℕ := 2
  let binomial_coeff := Nat.choose n r
  let power_term := a^(n - r) * b^r
  binomial_coeff * power_term = 24 * x^2 := by
sorry


end third_term_binomial_expansion_l282_28237


namespace first_chapter_pages_l282_28216

/-- Represents a book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- Theorem stating the number of pages in the first chapter of the book -/
theorem first_chapter_pages (b : Book) 
  (h1 : b.chapter2_pages = 11) 
  (h2 : b.chapter1_pages = b.chapter2_pages + 37) : 
  b.chapter1_pages = 48 := by
  sorry

end first_chapter_pages_l282_28216


namespace third_column_sum_l282_28282

/-- Represents a 3x3 grid of numbers -/
def Grid := Matrix (Fin 3) (Fin 3) ℤ

/-- The sum of a row in the grid -/
def row_sum (g : Grid) (i : Fin 3) : ℤ :=
  (g i 0) + (g i 1) + (g i 2)

/-- The sum of a column in the grid -/
def col_sum (g : Grid) (j : Fin 3) : ℤ :=
  (g 0 j) + (g 1 j) + (g 2 j)

/-- The theorem statement -/
theorem third_column_sum (g : Grid) 
  (h1 : row_sum g 0 = 24)
  (h2 : row_sum g 1 = 26)
  (h3 : row_sum g 2 = 40)
  (h4 : col_sum g 0 = 27)
  (h5 : col_sum g 1 = 20) :
  col_sum g 2 = 43 := by
  sorry


end third_column_sum_l282_28282


namespace ratio_problem_l282_28262

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 6) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := by
  sorry

end ratio_problem_l282_28262


namespace product_of_sum_and_sum_of_squares_l282_28283

theorem product_of_sum_and_sum_of_squares (a b : ℝ) 
  (sum_of_squares : a^2 + b^2 = 26) 
  (sum : a + b = 7) : 
  a * b = 23 / 2 := by
sorry

end product_of_sum_and_sum_of_squares_l282_28283


namespace fruit_pie_theorem_l282_28292

/-- Represents the number of fruits needed for different types of pies -/
structure FruitRequirement where
  apples : ℕ
  pears : ℕ
  peaches : ℕ

/-- Calculates the total fruits needed for a given number of pies -/
def total_fruits (req : FruitRequirement) (num_pies : ℕ) : FruitRequirement :=
  { apples := req.apples * num_pies
  , pears := req.pears * num_pies
  , peaches := req.peaches * num_pies }

/-- Adds two FruitRequirement structures -/
def add_requirements (a b : FruitRequirement) : FruitRequirement :=
  { apples := a.apples + b.apples
  , pears := a.pears + b.pears
  , peaches := a.peaches + b.peaches }

theorem fruit_pie_theorem :
  let fruit_pie_req : FruitRequirement := { apples := 4, pears := 3, peaches := 0 }
  let apple_peach_pie_req : FruitRequirement := { apples := 6, pears := 0, peaches := 2 }
  let fruit_pies := 357
  let apple_peach_pies := 712
  let total_req := add_requirements (total_fruits fruit_pie_req fruit_pies) (total_fruits apple_peach_pie_req apple_peach_pies)
  total_req.apples = 5700 ∧ total_req.pears = 1071 ∧ total_req.peaches = 1424 := by
  sorry

end fruit_pie_theorem_l282_28292


namespace chef_potato_problem_chef_potato_solution_l282_28219

theorem chef_potato_problem (already_cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) : ℕ :=
  let remaining_potatoes := remaining_cooking_time / cooking_time_per_potato
  let total_potatoes := already_cooked + remaining_potatoes
  total_potatoes

#check chef_potato_problem 8 9 63

theorem chef_potato_solution :
  chef_potato_problem 8 9 63 = 15 := by
  sorry

end chef_potato_problem_chef_potato_solution_l282_28219


namespace rachel_milk_consumption_l282_28288

theorem rachel_milk_consumption 
  (bottle1 : ℚ) (bottle2 : ℚ) (rachel_fraction : ℚ) :
  bottle1 = 3/8 →
  bottle2 = 1/4 →
  rachel_fraction = 3/4 →
  rachel_fraction * (bottle1 + bottle2) = 15/32 := by
  sorry

end rachel_milk_consumption_l282_28288


namespace trapezoid_bases_l282_28276

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- Diameter of the inscribed circle -/
  diameter : ℝ
  /-- Length of the leg (non-parallel side) -/
  leg : ℝ
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- Length of the shorter base -/
  shorterBase : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The trapezoid is circumscribed around the circle -/
  isCircumscribed : True

/-- Theorem stating the lengths of the bases for the given trapezoid -/
theorem trapezoid_bases (t : IsoscelesTrapezoid) 
    (h1 : t.diameter = 15)
    (h2 : t.leg = 17) :
    t.longerBase = 25 ∧ t.shorterBase = 9 := by
  sorry

end trapezoid_bases_l282_28276


namespace trajectory_equation_l282_28275

/-- The ellipse on which points M and N lie -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The condition for the product of slopes of OM and ON -/
def slope_product (a b : ℝ) (m_slope n_slope : ℝ) : Prop :=
  m_slope * n_slope = b^2 / a^2

/-- The trajectory equation for point P -/
def trajectory (a b m n : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = m^2 + n^2

/-- The main theorem -/
theorem trajectory_equation (a b : ℝ) (m n : ℕ+) (x y : ℝ) :
  a > b ∧ b > 0 →
  ∃ (mx my nx ny : ℝ),
    ellipse a b mx my ∧
    ellipse a b nx ny ∧
    ∃ (m_slope n_slope : ℝ),
      slope_product a b m_slope n_slope →
      x = m * mx + n * nx ∧
      y = m * my + n * ny →
      trajectory a b m n x y :=
by sorry

end trajectory_equation_l282_28275


namespace total_drivers_l282_28268

theorem total_drivers (N : ℕ) 
  (drivers_A : ℕ) 
  (sample_A sample_B sample_C sample_D : ℕ) :
  drivers_A = 96 →
  sample_A = 8 →
  sample_B = 23 →
  sample_C = 27 →
  sample_D = 43 →
  (sample_A : ℚ) / drivers_A = (sample_A + sample_B + sample_C + sample_D : ℚ) / N →
  N = 1212 :=
by sorry

end total_drivers_l282_28268


namespace boundary_is_pentagon_l282_28259

/-- The set S of points (x, y) satisfying the given conditions -/
def S (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               a / 2 ≤ x ∧ x ≤ 2 * a ∧
               a / 2 ≤ y ∧ y ≤ 2 * a ∧
               x + y ≥ 3 * a ∧
               x + a ≥ y ∧
               y + a ≥ x}

/-- The boundary of set S -/
def boundary (a : ℝ) : Set (ℝ × ℝ) :=
  frontier (S a)

/-- The number of sides of the polygon formed by the boundary of S -/
def numSides (a : ℝ) : ℕ :=
  sorry

theorem boundary_is_pentagon (a : ℝ) (h : a > 0) : numSides a = 5 :=
  sorry

end boundary_is_pentagon_l282_28259


namespace different_sum_of_digits_l282_28234

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Statement: For any natural number N, the sum of digits of N(N-1) is not equal to the sum of digits of (N+1)² -/
theorem different_sum_of_digits (N : ℕ) : 
  sum_of_digits (N * (N - 1)) ≠ sum_of_digits ((N + 1) ^ 2) := by
  sorry

end different_sum_of_digits_l282_28234


namespace village_lasts_five_weeks_l282_28253

/-- The number of weeks a village lasts given supernatural predators -/
def village_duration (village_population : ℕ) 
  (lead_vampire_drain : ℕ) (vampire_group_size : ℕ) (vampire_group_drain : ℕ)
  (alpha_werewolf_eat : ℕ) (werewolf_pack_size : ℕ) (werewolf_pack_eat : ℕ)
  (ghost_feed : ℕ) : ℕ :=
  let total_consumed_per_week := 
    lead_vampire_drain + 
    (vampire_group_size * vampire_group_drain) + 
    alpha_werewolf_eat + 
    (werewolf_pack_size * werewolf_pack_eat) + 
    ghost_feed
  village_population / total_consumed_per_week

theorem village_lasts_five_weeks :
  village_duration 200 5 3 5 7 2 5 2 = 5 := by
  sorry

end village_lasts_five_weeks_l282_28253


namespace functional_eq_solution_l282_28284

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - y) = 2009 * f x * f y

/-- The main theorem -/
theorem functional_eq_solution (f : ℝ → ℝ) 
  (h1 : FunctionalEq f) 
  (h2 : ∀ x : ℝ, f x ≠ 0) : 
  f (Real.sqrt 2009) = 1 / 2009 := by
  sorry

end functional_eq_solution_l282_28284


namespace magic_square_base_5_l282_28246

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e f g h i : ℕ)

/-- Converts a number from base 5 to base 10 -/
def toBase10 (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 10 => 5
  | 11 => 6
  | 12 => 7
  | 13 => 8
  | 14 => 9
  | _ => 0  -- For simplicity, we only define the conversion for numbers used in the square

/-- Checks if the given square is magic in base 5 -/
def isMagicSquare (s : MagicSquare) : Prop :=
  let a' := toBase10 s.a
  let b' := toBase10 s.b
  let c' := toBase10 s.c
  let d' := toBase10 s.d
  let e' := toBase10 s.e
  let f' := toBase10 s.f
  let g' := toBase10 s.g
  let h' := toBase10 s.h
  let i' := toBase10 s.i
  -- Row sums are equal
  (a' + b' + c' = d' + e' + f') ∧
  (d' + e' + f' = g' + h' + i') ∧
  -- Column sums are equal
  (a' + d' + g' = b' + e' + h') ∧
  (b' + e' + h' = c' + f' + i') ∧
  -- Diagonal sums are equal
  (a' + e' + i' = c' + e' + g')

theorem magic_square_base_5 : 
  isMagicSquare ⟨13, 1, 11, 3, 10, 12, 4, 14, 2⟩ := by
  sorry


end magic_square_base_5_l282_28246


namespace largest_coefficient_binomial_expansion_l282_28260

theorem largest_coefficient_binomial_expansion :
  ∀ n : ℕ, 
    n ≤ 11 → 
    (Nat.choose 11 n : ℚ) ≤ (Nat.choose 11 6 : ℚ) ∧
    (Nat.choose 11 6 : ℚ) = (Nat.choose 11 5 : ℚ) ∧
    (∀ k : ℕ, k < 5 → (Nat.choose 11 k : ℚ) < (Nat.choose 11 6 : ℚ)) :=
by
  sorry

#check largest_coefficient_binomial_expansion

end largest_coefficient_binomial_expansion_l282_28260


namespace calculation_proof_l282_28278

theorem calculation_proof : (30 / (7 + 2 - 6)) * 7 = 70 := by
  sorry

end calculation_proof_l282_28278


namespace product_increased_equals_nineteen_l282_28241

theorem product_increased_equals_nineteen (x : ℝ) : 5 * x + 4 = 19 ↔ x = 3 := by
  sorry

end product_increased_equals_nineteen_l282_28241


namespace problem_solution_l282_28287

theorem problem_solution (x y z : ℝ) 
  (h1 : |x| + x + y = 12)
  (h2 : x + |y| - y = 10)
  (h3 : x - y + z = 5) :
  x + y + z = 9/5 := by
sorry

end problem_solution_l282_28287


namespace shoes_per_person_l282_28297

theorem shoes_per_person (num_pairs : ℕ) (num_people : ℕ) : 
  num_pairs = 36 → num_people = 36 → (num_pairs * 2) / num_people = 2 := by
  sorry

end shoes_per_person_l282_28297


namespace cylinder_surface_area_l282_28200

/-- The total surface area of a cylinder with height 12 and radius 4 is 128π. -/
theorem cylinder_surface_area :
  let h : ℝ := 12
  let r : ℝ := 4
  let circle_area := π * r^2
  let lateral_area := 2 * π * r * h
  circle_area * 2 + lateral_area = 128 * π :=
by sorry

end cylinder_surface_area_l282_28200


namespace unique_solution_cube_equation_l282_28267

theorem unique_solution_cube_equation : 
  ∃! (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 ∧ x = 27 := by
sorry

end unique_solution_cube_equation_l282_28267


namespace grid_division_l282_28215

/-- Represents a cell in the grid -/
inductive Cell
| Shaded
| Unshaded

/-- Represents the 6x6 grid -/
def Grid := Matrix (Fin 6) (Fin 6) Cell

/-- Counts the number of shaded cells in a given region of the grid -/
def count_shaded (g : Grid) (start_row end_row start_col end_col : Fin 6) : Nat :=
  sorry

/-- Checks if a given 3x3 region of the grid contains exactly 3 shaded cells -/
def is_valid_part (g : Grid) (start_row start_col : Fin 6) : Prop :=
  count_shaded g start_row (start_row + 2) start_col (start_col + 2) = 3

/-- The main theorem to be proved -/
theorem grid_division (g : Grid) 
  (h1 : count_shaded g 0 5 0 5 = 12) : 
  (is_valid_part g 0 0) ∧ 
  (is_valid_part g 0 3) ∧ 
  (is_valid_part g 3 0) ∧ 
  (is_valid_part g 3 3) :=
sorry

end grid_division_l282_28215


namespace tangent_line_at_point_one_zero_l282_28255

/-- The equation of the tangent line to y = x^3 - 2x + 1 at (1, 0) is y = x - 1 -/
theorem tangent_line_at_point_one_zero (x y : ℝ) :
  let f : ℝ → ℝ := λ t => t^3 - 2*t + 1
  let f' : ℝ → ℝ := λ t => 3*t^2 - 2
  let tangent_line : ℝ → ℝ := λ t => t - 1
  f 1 = 0 ∧ f' 1 = (tangent_line 1 - tangent_line 0) → 
  ∀ t, tangent_line t = f 1 + f' 1 * (t - 1) :=
by sorry

end tangent_line_at_point_one_zero_l282_28255


namespace bench_cost_is_150_l282_28235

/-- The cost of a bench and garden table, where the table costs twice as much as the bench. -/
def BenchAndTableCost (bench_cost : ℝ) : ℝ := bench_cost + 2 * bench_cost

/-- Theorem stating that the bench costs 150 dollars given the conditions. -/
theorem bench_cost_is_150 :
  ∃ (bench_cost : ℝ), BenchAndTableCost bench_cost = 450 ∧ bench_cost = 150 :=
by
  sorry

end bench_cost_is_150_l282_28235


namespace no_lattice_equilateral_triangle_l282_28295

-- Define a lattice point as a point with integer coordinates
def LatticePoint (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

-- Define an equilateral triangle
def Equilateral (a b c : ℝ × ℝ) : Prop :=
  let d := (fun (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  d a b = d b c ∧ d b c = d c a

-- Theorem statement
theorem no_lattice_equilateral_triangle :
  ¬ ∃ (a b c : ℝ × ℝ), LatticePoint a ∧ LatticePoint b ∧ LatticePoint c ∧ Equilateral a b c :=
sorry

end no_lattice_equilateral_triangle_l282_28295


namespace percentage_to_new_school_l282_28223

theorem percentage_to_new_school (total_students : ℕ) 
  (percent_to_A : ℚ) (percent_to_B : ℚ) 
  (percent_A_to_C : ℚ) (percent_B_to_C : ℚ) :
  percent_to_A = 60 / 100 →
  percent_to_B = 40 / 100 →
  percent_A_to_C = 30 / 100 →
  percent_B_to_C = 40 / 100 →
  let students_A := (percent_to_A * total_students).floor
  let students_B := (percent_to_B * total_students).floor
  let students_A_to_C := (percent_A_to_C * students_A).floor
  let students_B_to_C := (percent_B_to_C * students_B).floor
  let total_to_C := students_A_to_C + students_B_to_C
  ((total_to_C : ℚ) / total_students * 100).floor = 34 := by
sorry

end percentage_to_new_school_l282_28223


namespace function_value_l282_28231

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt (a * x - 1) else -x^2 - 4*x

theorem function_value (a : ℝ) : f a (f a (-2)) = 3 → a = 5/2 := by
  sorry

end function_value_l282_28231


namespace max_sum_of_factors_l282_28291

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 24 → 
  Even heart → 
  (∀ h c : ℕ, h * c = 24 → Even h → heart + club ≥ h + c) →
  heart + club = 14 := by
sorry

end max_sum_of_factors_l282_28291


namespace night_heads_count_l282_28239

/-- Represents the number of animals of each type -/
structure AnimalCounts where
  chickens : ℕ
  rabbits : ℕ
  geese : ℕ

/-- Calculates the total number of legs during the day -/
def totalDayLegs (counts : AnimalCounts) : ℕ :=
  2 * counts.chickens + 4 * counts.rabbits + 2 * counts.geese

/-- Calculates the total number of heads -/
def totalHeads (counts : AnimalCounts) : ℕ :=
  counts.chickens + counts.rabbits + counts.geese

/-- Calculates the total number of legs at night -/
def totalNightLegs (counts : AnimalCounts) : ℕ :=
  2 * counts.chickens + 4 * counts.rabbits + counts.geese

/-- The main theorem to prove -/
theorem night_heads_count (counts : AnimalCounts) 
  (h1 : totalDayLegs counts = 56)
  (h2 : totalDayLegs counts - totalHeads counts = totalNightLegs counts - totalHeads counts) :
  totalHeads counts = 14 := by
  sorry


end night_heads_count_l282_28239


namespace interest_rate_calculation_l282_28229

theorem interest_rate_calculation (initial_charge : ℝ) (final_amount : ℝ) (time : ℝ) :
  initial_charge = 75 →
  final_amount = 80.25 →
  time = 1 →
  (final_amount - initial_charge) / (initial_charge * time) = 0.07 :=
by
  sorry

end interest_rate_calculation_l282_28229


namespace new_boat_travel_distance_l282_28264

/-- Calculates the distance traveled by a new boat given the speed increase and the distance traveled by an old boat -/
def new_boat_distance (speed_increase : ℝ) (old_distance : ℝ) : ℝ :=
  old_distance * (1 + speed_increase)

/-- Theorem: Given a new boat traveling 30% faster than an old boat, and the old boat traveling 150 miles,
    the new boat will travel 195 miles in the same time -/
theorem new_boat_travel_distance :
  new_boat_distance 0.3 150 = 195 := by
  sorry

#eval new_boat_distance 0.3 150

end new_boat_travel_distance_l282_28264


namespace subset_condition_l282_28218

def A (x : ℝ) : Prop := |2 * x - 1| < 1

def B (a x : ℝ) : Prop := x^2 - 2*a*x + a^2 - 1 > 0

theorem subset_condition (a : ℝ) :
  (∀ x, A x → B a x) ↔ (a ≤ -1 ∨ a ≥ 2) :=
sorry

end subset_condition_l282_28218


namespace line_circle_intersection_l282_28251

theorem line_circle_intersection (k : ℝ) : 
  ∃ x y : ℝ, y = k * (x + 1/2) ∧ x^2 + y^2 = 1 := by
sorry

end line_circle_intersection_l282_28251


namespace regular_polygon_properties_l282_28299

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides
    and the sum of its interior angles is 3240 degrees. -/
theorem regular_polygon_properties (n : ℕ) (exterior_angle : ℝ) :
  exterior_angle = 18 →
  (360 : ℝ) / exterior_angle = n →
  n = 20 ∧
  180 * (n - 2) = 3240 := by
  sorry

end regular_polygon_properties_l282_28299


namespace polar_rectangular_equivalence_l282_28257

theorem polar_rectangular_equivalence (ρ θ x y : ℝ) :
  y = ρ * Real.sin θ ∧ x = ρ * Real.cos θ →
  (y^2 = 12 * x ↔ ρ * Real.sin θ^2 = 12 * Real.cos θ) :=
by sorry

end polar_rectangular_equivalence_l282_28257


namespace coin_value_difference_l282_28266

-- Define the coin types
inductive Coin
| Penny
| Nickel
| Dime

-- Define the function to calculate the value of a coin in cents
def coinValue : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10

-- Define the total number of coins
def totalCoins : Nat := 3000

-- Define the theorem
theorem coin_value_difference :
  ∃ (p n d : Nat),
    p + n + d = totalCoins ∧
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧
    (∀ (p' n' d' : Nat),
      p' + n' + d' = totalCoins →
      p' ≥ 1 → n' ≥ 1 → d' ≥ 1 →
      coinValue Coin.Penny * p' + coinValue Coin.Nickel * n' + coinValue Coin.Dime * d' ≤
      coinValue Coin.Penny * p + coinValue Coin.Nickel * n + coinValue Coin.Dime * d) ∧
    (∀ (p' n' d' : Nat),
      p' + n' + d' = totalCoins →
      p' ≥ 1 → n' ≥ 1 → d' ≥ 1 →
      coinValue Coin.Penny * p + coinValue Coin.Nickel * n + coinValue Coin.Dime * d -
      (coinValue Coin.Penny * p' + coinValue Coin.Nickel * n' + coinValue Coin.Dime * d') = 26973) :=
by sorry


end coin_value_difference_l282_28266


namespace segment_length_ratio_l282_28279

/-- Given two line segments with points placed at equal intervals, 
    prove that the longer segment is 101 times the length of the shorter segment. -/
theorem segment_length_ratio 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_points_a : ∃ (d : ℝ), a = 99 * d ∧ d > 0) 
  (h_points_b : ∃ (d : ℝ), b = 9999 * d ∧ d > 0) 
  (h_same_interval : ∀ (d1 d2 : ℝ), (a = 99 * d1 ∧ d1 > 0) → (b = 9999 * d2 ∧ d2 > 0) → d1 = d2) :
  b = 101 * a := by
  sorry

end segment_length_ratio_l282_28279


namespace fred_final_balloons_l282_28209

def fred_balloons : ℕ → Prop
| n => ∃ (initial given received distributed : ℕ),
  initial = 1457 ∧
  given = 341 ∧
  received = 225 ∧
  distributed = ((initial - given + received) / 2) ∧
  n = initial - given + received - distributed

theorem fred_final_balloons : fred_balloons 671 := by
  sorry

end fred_final_balloons_l282_28209


namespace smallest_multiplier_for_perfect_cube_l282_28272

def y : ℕ := 2^63^74^95^86^47^5

theorem smallest_multiplier_for_perfect_cube (n : ℕ) :
  (∀ m : ℕ, 0 < m ∧ m < 18 → ¬ ∃ k : ℕ, y * m = k^3) ∧
  ∃ k : ℕ, y * 18 = k^3 :=
sorry

end smallest_multiplier_for_perfect_cube_l282_28272


namespace perfect_square_difference_l282_28261

theorem perfect_square_difference (x y : ℕ) (h : x > 0 ∧ y > 0) 
  (eq : 3 * x^2 + x = 4 * y^2 + y) : 
  ∃ (k : ℕ), x - y = k^2 := by
sorry

end perfect_square_difference_l282_28261


namespace rachel_and_sarah_return_trip_money_l282_28296

theorem rachel_and_sarah_return_trip_money :
  let initial_amount : ℚ := 50
  let gasoline_cost : ℚ := 8
  let lunch_cost : ℚ := 15.65
  let gift_cost_per_person : ℚ := 5
  let grandma_gift_per_person : ℚ := 10
  let num_people : ℕ := 2

  let total_spent : ℚ := gasoline_cost + lunch_cost + (gift_cost_per_person * num_people)
  let total_received_from_grandma : ℚ := grandma_gift_per_person * num_people
  let remaining_amount : ℚ := initial_amount - total_spent + total_received_from_grandma

  remaining_amount = 36.35 :=
by
  sorry

end rachel_and_sarah_return_trip_money_l282_28296


namespace food_distribution_l282_28280

/-- Represents the number of days the initial food supply lasts -/
def initial_days : ℕ := 22

/-- Represents the number of days that pass before additional men join -/
def days_passed : ℕ := 2

/-- Represents the number of additional men who join -/
def additional_men : ℕ := 2280

/-- Represents the number of days the food lasts after additional men join -/
def remaining_days : ℕ := 5

/-- Represents the initial number of men -/
def initial_men : ℕ := 760

theorem food_distribution (M : ℕ) :
  M * (initial_days - days_passed) = (M + additional_men) * remaining_days →
  M = initial_men := by sorry

end food_distribution_l282_28280


namespace coordinates_wrt_origin_l282_28254

-- Define a point in a 2D Cartesian coordinate system
def Point := ℝ × ℝ

-- Define the given point
def givenPoint : Point := (-2, 3)

-- Theorem stating that the coordinates of the given point with respect to the origin are (-2, 3)
theorem coordinates_wrt_origin (p : Point) (h : p = givenPoint) : p = (-2, 3) := by
  sorry

end coordinates_wrt_origin_l282_28254


namespace sqrt_real_range_l282_28247

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y^2 = x - 3) → x ≥ 3 := by
  sorry

end sqrt_real_range_l282_28247


namespace fraction_simplification_l282_28221

theorem fraction_simplification (m : ℝ) (hm : m ≠ 0) (hm1 : m ≠ 1) (hm2 : m ≠ -1) :
  ((m - 1) / m) / ((m^2 - 1) / m^2) = m / (m + 1) := by
sorry

end fraction_simplification_l282_28221


namespace sqrt_two_plus_sqrt_three_gt_sqrt_five_l282_28269

theorem sqrt_two_plus_sqrt_three_gt_sqrt_five :
  Real.sqrt 2 + Real.sqrt 3 > Real.sqrt 5 := by
  sorry

end sqrt_two_plus_sqrt_three_gt_sqrt_five_l282_28269


namespace square_ratio_l282_28214

theorem square_ratio (n m : ℝ) :
  (∃ a : ℝ, 9 * x^2 + n * x + 1 = (3 * x + a)^2) →
  (∃ b : ℝ, 4 * y^2 + 12 * y + m = (2 * y + b)^2) →
  n > 0 →
  n / m = 2 / 3 := by
sorry

end square_ratio_l282_28214


namespace base5_20314_equals_1334_l282_28222

def base5_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ (digits.length - 1 - i))) 0

theorem base5_20314_equals_1334 :
  base5_to_base10 [2, 0, 3, 1, 4] = 1334 := by
  sorry

end base5_20314_equals_1334_l282_28222


namespace chessboard_coverage_l282_28233

/-- An L-shaped tetromino covers exactly 4 squares. -/
def LTetromino : ℕ := 4

/-- Represents an m × n chessboard. -/
structure Chessboard where
  m : ℕ
  n : ℕ

/-- Predicate to check if a number is divisible by 8. -/
def divisible_by_eight (x : ℕ) : Prop := ∃ k, x = 8 * k

/-- Predicate to check if a chessboard can be covered by L-shaped tetrominoes. -/
def can_cover (board : Chessboard) : Prop :=
  divisible_by_eight (board.m * board.n) ∧ board.m ≠ 1 ∧ board.n ≠ 1

theorem chessboard_coverage (board : Chessboard) :
  (∃ (tiles : ℕ), board.m * board.n = tiles * LTetromino) ↔ can_cover board :=
sorry

end chessboard_coverage_l282_28233


namespace exam_score_proof_l282_28217

/-- Given an exam with mean score 76, prove that the score 2 standard deviations
    below the mean is 60, knowing that 100 is 3 standard deviations above the mean. -/
theorem exam_score_proof (mean : ℝ) (score_above : ℝ) (std_dev_above : ℝ) (std_dev_below : ℝ) :
  mean = 76 →
  score_above = 100 →
  std_dev_above = 3 →
  std_dev_below = 2 →
  score_above = mean + std_dev_above * ((score_above - mean) / std_dev_above) →
  mean - std_dev_below * ((score_above - mean) / std_dev_above) = 60 :=
by sorry

end exam_score_proof_l282_28217


namespace remainder_of_60_div_18_l282_28201

theorem remainder_of_60_div_18 : ∃ q : ℕ, 60 = 18 * q + 6 := by
  sorry

#check remainder_of_60_div_18

end remainder_of_60_div_18_l282_28201


namespace jerry_color_cartridges_l282_28245

/-- Represents the cost of a color cartridge in dollars -/
def color_cartridge_cost : ℕ := 32

/-- Represents the cost of a black-and-white cartridge in dollars -/
def bw_cartridge_cost : ℕ := 27

/-- Represents the total amount Jerry pays in dollars -/
def total_cost : ℕ := 123

/-- Represents the number of black-and-white cartridges Jerry needs -/
def bw_cartridges : ℕ := 1

theorem jerry_color_cartridges :
  ∃ (c : ℕ), c * color_cartridge_cost + bw_cartridges * bw_cartridge_cost = total_cost ∧ c = 3 := by
  sorry

end jerry_color_cartridges_l282_28245
