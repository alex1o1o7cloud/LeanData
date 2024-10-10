import Mathlib

namespace factorization_1_factorization_2_l401_40197

-- Part 1
theorem factorization_1 (x y : ℝ) : -x^5 * y^3 + x^3 * y^5 = -x^3 * y^3 * (x + y) * (x - y) := by sorry

-- Part 2
theorem factorization_2 (a : ℝ) : (a^2 + 1)^2 - 4 * a^2 = (a + 1)^2 * (a - 1)^2 := by sorry

end factorization_1_factorization_2_l401_40197


namespace kangaroo_distance_after_four_hops_l401_40184

/-- The distance traveled by a kangaroo after a certain number of hops,
    where each hop is 1/4 of the remaining distance to the target. -/
def kangaroo_distance (target : ℚ) (hops : ℕ) : ℚ :=
  target * (1 - (3/4)^hops)

/-- Theorem: A kangaroo starting at 0 aiming for 2, hopping 1/4 of the remaining
    distance each time, will travel 175/128 units after 4 hops. -/
theorem kangaroo_distance_after_four_hops :
  kangaroo_distance 2 4 = 175 / 128 := by
  sorry

#eval kangaroo_distance 2 4

end kangaroo_distance_after_four_hops_l401_40184


namespace jake_reading_theorem_l401_40167

def read_pattern (first_day : ℕ) : ℕ → ℕ
  | 1 => first_day
  | 2 => first_day - 20
  | 3 => 2 * (first_day - 20)
  | 4 => first_day / 2
  | _ => 0

def total_pages_read (first_day : ℕ) : ℕ :=
  (read_pattern first_day 1) + (read_pattern first_day 2) + 
  (read_pattern first_day 3) + (read_pattern first_day 4)

theorem jake_reading_theorem (book_chapters book_pages : ℕ) 
  (h1 : book_chapters = 8) (h2 : book_pages = 130) (h3 : read_pattern 37 1 = 37) :
  total_pages_read 37 = 106 := by sorry

end jake_reading_theorem_l401_40167


namespace probability_three_defective_shipment_l401_40177

/-- The probability of selecting three defective smartphones from a shipment --/
def probability_three_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total *
  ((defective - 1) : ℚ) / (total - 1) *
  ((defective - 2) : ℚ) / (total - 2)

/-- Theorem stating the approximate probability of selecting three defective smartphones --/
theorem probability_three_defective_shipment :
  let total := 500
  let defective := 85
  abs (probability_three_defective total defective - 0.0047) < 0.0001 := by
  sorry

end probability_three_defective_shipment_l401_40177


namespace quadratic_equation_k_value_l401_40179

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : Prop := k * x^2 + x - 3 = 0

-- Define the condition for distinct real roots
def has_distinct_real_roots (k : ℝ) : Prop := k > -1/12 ∧ k ≠ 0

-- Define the condition for the roots
def roots_condition (x₁ x₂ : ℝ) : Prop := (x₁ + x₂)^2 + x₁ * x₂ = 4

-- Theorem statement
theorem quadratic_equation_k_value :
  ∀ k : ℝ, 
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    quadratic_equation k x₁ ∧ 
    quadratic_equation k x₂ ∧
    has_distinct_real_roots k ∧
    roots_condition x₁ x₂) →
  k = 1/4 :=
sorry

end quadratic_equation_k_value_l401_40179


namespace cricket_player_innings_l401_40153

theorem cricket_player_innings 
  (average : ℝ) 
  (next_innings_runs : ℝ) 
  (average_increase : ℝ) 
  (h1 : average = 33) 
  (h2 : next_innings_runs = 77) 
  (h3 : average_increase = 4) :
  ∃ n : ℕ, 
    (n : ℝ) * average + next_innings_runs = (n + 1) * (average + average_increase) ∧ 
    n = 10 :=
by sorry

end cricket_player_innings_l401_40153


namespace range_of_fraction_l401_40195

theorem range_of_fraction (x y : ℝ) (hx : 1 < x ∧ x < 6) (hy : 2 < y ∧ y < 8) :
  1/8 < x/y ∧ x/y < 3 := by
  sorry

end range_of_fraction_l401_40195


namespace smallest_n_satisfying_conditions_l401_40117

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  (n = 626) ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (Real.sqrt (m : ℝ) - Real.sqrt ((m - 1) : ℝ) ≥ 0.02 ∨ 
     Real.sin (Real.pi / Real.sqrt (m : ℝ)) ≤ 0.5)) ∧
  (Real.sqrt (n : ℝ) - Real.sqrt ((n - 1) : ℝ) < 0.02) ∧
  (Real.sin (Real.pi / Real.sqrt (n : ℝ)) > 0.5) :=
by sorry

end smallest_n_satisfying_conditions_l401_40117


namespace parallel_vectors_m_value_l401_40145

-- Define the vectors a and b
def a (m : ℝ) : Fin 2 → ℝ := λ i => match i with
  | 0 => 2
  | 1 => m

def b (m : ℝ) : Fin 2 → ℝ := λ i => match i with
  | 0 => m
  | 1 => 2

-- Define the parallelism condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

-- State the theorem
theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (a m) (b m) → m = 2 ∨ m = -2 := by
  sorry

end parallel_vectors_m_value_l401_40145


namespace count_numbers_with_remainder_l401_40126

theorem count_numbers_with_remainder (n : ℕ) : 
  (Finset.filter (fun N => N > 17 ∧ 2017 % N = 17) (Finset.range (2017 + 1))).card = 13 := by
  sorry

end count_numbers_with_remainder_l401_40126


namespace number_of_petri_dishes_l401_40156

/-- The number of petri dishes in a lab, given the total number of germs and germs per dish -/
theorem number_of_petri_dishes 
  (total_germs : ℝ) 
  (germs_per_dish : ℝ) 
  (h1 : total_germs = 0.036 * 10^5)
  (h2 : germs_per_dish = 47.99999999999999)
  : ℤ :=
75

#check number_of_petri_dishes

end number_of_petri_dishes_l401_40156


namespace logarithm_sum_simplification_l401_40154

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 5 / Real.log 30 + 1) = 2 :=
by sorry

end logarithm_sum_simplification_l401_40154


namespace square_of_two_plus_i_l401_40106

theorem square_of_two_plus_i : (2 + Complex.I) ^ 2 = 3 + 4 * Complex.I := by
  sorry

end square_of_two_plus_i_l401_40106


namespace triangle_abc_properties_l401_40120

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  B = 150 * π / 180 →
  a = Real.sqrt 3 * c →
  b = 2 * Real.sqrt 7 →
  Real.sin A + Real.sqrt 3 * Real.sin C = Real.sqrt 2 / 2 →
  (∃ (S : Real), S = a * b * Real.sin C / 2 ∧ S = Real.sqrt 3) ∧
  C = 15 * π / 180 := by
  sorry

end triangle_abc_properties_l401_40120


namespace miltons_zoology_books_l401_40160

theorem miltons_zoology_books :
  ∀ (z b : ℕ), b = 4 * z → z + b = 80 → z = 16 := by sorry

end miltons_zoology_books_l401_40160


namespace set_intersection_equality_l401_40155

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2^x - 1)}

-- Define set B
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

-- Define the intersection set
def intersection_set : Set ℝ := {x | 0 ≤ x ∧ x < 2}

-- Theorem statement
theorem set_intersection_equality : A ∩ B = intersection_set := by sorry

end set_intersection_equality_l401_40155


namespace isosceles_triangle_area_l401_40133

theorem isosceles_triangle_area : 
  ∀ a b : ℕ,
  a > 0 ∧ b > 0 →
  2 * a + b = 12 →
  (a + a > b ∧ a + b > a) →
  (∃ (s : ℝ), s * s = (a * a : ℝ) - (b * b / 4 : ℝ)) →
  (a * s / 2 : ℝ) = 4 * Real.sqrt 3 :=
by sorry

end isosceles_triangle_area_l401_40133


namespace min_boxes_for_cube_l401_40175

/-- The width of the box in centimeters -/
def box_width : ℕ := 8

/-- The length of the box in centimeters -/
def box_length : ℕ := 12

/-- The height of the box in centimeters -/
def box_height : ℕ := 30

/-- The volume of a single box in cubic centimeters -/
def box_volume : ℕ := box_width * box_length * box_height

/-- The side length of the smallest cube that can be formed -/
def cube_side : ℕ := Nat.lcm (Nat.lcm box_width box_length) box_height

/-- The volume of the smallest cube that can be formed -/
def cube_volume : ℕ := cube_side ^ 3

/-- The theorem stating the minimum number of boxes needed to form a cube -/
theorem min_boxes_for_cube : cube_volume / box_volume = 600 := by
  sorry

end min_boxes_for_cube_l401_40175


namespace rock_sale_price_per_pound_l401_40171

theorem rock_sale_price_per_pound 
  (average_weight : ℝ) 
  (num_rocks : ℕ) 
  (total_sale : ℝ) 
  (h1 : average_weight = 1.5)
  (h2 : num_rocks = 10)
  (h3 : total_sale = 60) :
  total_sale / (average_weight * num_rocks) = 4 := by
sorry

end rock_sale_price_per_pound_l401_40171


namespace class_age_problem_l401_40140

/-- Proves that if the average age of 6 people remains 19 years after adding a 1-year-old person,
    then the original average was calculated 1 year ago. -/
theorem class_age_problem (initial_total_age : ℕ) (years_passed : ℕ) : 
  initial_total_age / 6 = 19 →
  (initial_total_age + 6 * years_passed + 1) / 7 = 19 →
  years_passed = 1 := by
sorry

end class_age_problem_l401_40140


namespace salary_approximation_l401_40196

/-- The salary of a man who spends specific fractions on expenses and has a remainder --/
def salary (food_fraction : ℚ) (rent_fraction : ℚ) (clothes_fraction : ℚ) (remainder : ℚ) : ℚ :=
  remainder / (1 - food_fraction - rent_fraction - clothes_fraction)

/-- Theorem stating the approximate salary of a man with given expenses and remainder --/
theorem salary_approximation :
  let s := salary (1/3) (1/4) (1/5) 1760
  ⌊s⌋ = 8123 := by sorry

end salary_approximation_l401_40196


namespace tensor_result_l401_40143

-- Define the sets P and Q
def P : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def Q : Set ℝ := {x | x > 1}

-- Define the ⊗ operation
def tensorOp (P Q : Set ℝ) : Set ℝ := {x | x ∈ P ∪ Q ∧ x ∉ P ∩ Q}

-- Theorem statement
theorem tensor_result : tensorOp P Q = {x | (0 ≤ x ∧ x ≤ 1) ∨ (x > 2)} := by
  sorry

end tensor_result_l401_40143


namespace range_of_a_l401_40100

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then (a + 1) * x - 2 * a else Real.log x / Real.log 3

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a ∈ Set.Ioi (-1) :=
by sorry

end range_of_a_l401_40100


namespace absolute_value_trigonometry_and_reciprocal_quadratic_equation_solution_l401_40134

-- Problem 1
theorem absolute_value_trigonometry_and_reciprocal :
  |(-3)| - 4 * Real.sin (π / 6) + (1 / 3)⁻¹ = 4 := by sorry

-- Problem 2
theorem quadratic_equation_solution :
  ∀ x : ℝ, 2 * x - 6 = x^2 - 9 ↔ x = -1 ∨ x = 3 := by sorry

end absolute_value_trigonometry_and_reciprocal_quadratic_equation_solution_l401_40134


namespace number_equality_l401_40147

theorem number_equality (x : ℚ) (h : (30 / 100) * x = (40 / 100) * 50) : x = 200 / 3 := by
  sorry

end number_equality_l401_40147


namespace fifth_over_eight_fourth_power_l401_40163

theorem fifth_over_eight_fourth_power : (5 / 8 : ℚ) ^ 4 = 625 / 4096 := by
  sorry

end fifth_over_eight_fourth_power_l401_40163


namespace transistor_count_2002_l401_40186

def moores_law (initial_year final_year : ℕ) (initial_transistors : ℕ) : ℕ :=
  initial_transistors * 2^((final_year - initial_year) / 2)

theorem transistor_count_2002 :
  moores_law 1988 2002 500000 = 64000000 := by
  sorry

end transistor_count_2002_l401_40186


namespace max_area_right_triangle_l401_40176

/-- The maximum area of a right-angled triangle with perimeter 2 is 3 - 2√2 -/
theorem max_area_right_triangle (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_perimeter : a + b + c = 2) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  (1/2) * a * b ≤ 3 - 2 * Real.sqrt 2 :=
sorry

end max_area_right_triangle_l401_40176


namespace compass_leg_swap_impossible_l401_40198

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents the state of the compass -/
structure CompassState where
  leg1 : GridPoint
  leg2 : GridPoint

/-- The squared distance between two grid points -/
def squaredDistance (p1 p2 : GridPoint) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- A valid move of the compass -/
def isValidMove (start finish : CompassState) : Prop :=
  (start.leg1 = finish.leg1 ∧ squaredDistance start.leg1 start.leg2 = squaredDistance start.leg1 finish.leg2) ∨
  (start.leg2 = finish.leg2 ∧ squaredDistance start.leg1 start.leg2 = squaredDistance finish.leg1 start.leg2)

/-- A sequence of valid moves -/
def isValidMoveSequence : List CompassState → Prop
  | [] => True
  | [_] => True
  | s1 :: s2 :: rest => isValidMove s1 s2 ∧ isValidMoveSequence (s2 :: rest)

/-- The main theorem stating it's impossible to swap compass legs -/
theorem compass_leg_swap_impossible (start finish : CompassState) (moves : List CompassState) :
  isValidMoveSequence (start :: moves ++ [finish]) →
  squaredDistance start.leg1 start.leg2 = squaredDistance finish.leg1 finish.leg2 →
  ¬(start.leg1 = finish.leg2 ∧ start.leg2 = finish.leg1) :=
sorry

end compass_leg_swap_impossible_l401_40198


namespace remainder_4039_div_31_l401_40118

theorem remainder_4039_div_31 : 4039 % 31 = 9 := by
  sorry

end remainder_4039_div_31_l401_40118


namespace jenny_donut_order_l401_40144

def donut_combinations (total_donuts : ℕ) (kinds : ℕ) (min_per_kind : ℕ) : ℕ :=
  let two_kinds := Nat.choose kinds 2 * Nat.choose (total_donuts - 2 * min_per_kind + 2 - 1) (2 - 1)
  let three_kinds := Nat.choose kinds 3 * Nat.choose (total_donuts - 3 * min_per_kind + 3 - 1) (3 - 1)
  two_kinds + three_kinds

theorem jenny_donut_order : donut_combinations 8 5 2 = 110 := by
  sorry

end jenny_donut_order_l401_40144


namespace eight_mile_taxi_ride_cost_l401_40164

/-- Calculates the cost of a taxi ride given the base fare, cost per mile, and total miles traveled. -/
def taxiRideCost (baseFare : ℝ) (costPerMile : ℝ) (miles : ℝ) : ℝ :=
  baseFare + costPerMile * miles

/-- Theorem stating that an 8-mile taxi ride with a $2.00 base fare and $0.30 per mile costs $4.40. -/
theorem eight_mile_taxi_ride_cost :
  taxiRideCost 2.00 0.30 8 = 4.40 := by
  sorry

end eight_mile_taxi_ride_cost_l401_40164


namespace additional_flowers_grown_l401_40138

theorem additional_flowers_grown 
  (initial_flowers : ℕ) 
  (dead_flowers : ℕ) 
  (final_flowers : ℕ) : 
  final_flowers > initial_flowers → 
  final_flowers - initial_flowers = 
    final_flowers - initial_flowers + dead_flowers - dead_flowers :=
by
  sorry

#check additional_flowers_grown

end additional_flowers_grown_l401_40138


namespace find_set_B_l401_40170

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

theorem find_set_B (A B : Set Nat) 
  (h1 : (U \ (A ∪ B)) = {1, 3})
  (h2 : (A ∩ (U \ B)) = {2, 5}) :
  B = {4, 6, 7} := by
  sorry

end find_set_B_l401_40170


namespace problem_solution_l401_40190

theorem problem_solution (a : ℝ) (h : a/3 - 3/a = 4) :
  (a^8 - 6561) / (81 * a^4) * (3 * a) / (a^2 + 9) = 72 := by
  sorry

end problem_solution_l401_40190


namespace max_value_of_f_l401_40188

-- Define the function f(x) = x(4 - x)
def f (x : ℝ) := x * (4 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 4 ∧ ∀ x, 0 < x ∧ x < 4 → f x ≤ M :=
by sorry

end max_value_of_f_l401_40188


namespace probability_two_red_balls_l401_40130

/-- The probability of picking two red balls from a bag with 4 red, 3 blue, and 2 green balls is 1/6 -/
theorem probability_two_red_balls (total_balls : Nat) (red_balls : Nat) (blue_balls : Nat) (green_balls : Nat)
  (h1 : total_balls = red_balls + blue_balls + green_balls)
  (h2 : red_balls = 4)
  (h3 : blue_balls = 3)
  (h4 : green_balls = 2) :
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 1 / 6 := by
  sorry

end probability_two_red_balls_l401_40130


namespace equation_solution_l401_40103

theorem equation_solution : 
  ∃! x : ℝ, Real.sqrt x + 3 * Real.sqrt (x^2 + 8*x) + Real.sqrt (x + 8) = 40 - 3*x := by
  sorry

end equation_solution_l401_40103


namespace macaron_fraction_l401_40122

theorem macaron_fraction (mitch joshua miles renz : ℕ) (total_kids : ℕ) :
  mitch = 20 →
  joshua = mitch + 6 →
  2 * joshua = miles →
  total_kids = 68 →
  2 * total_kids = mitch + joshua + miles + renz →
  renz + 1 = miles * 19 / 26 :=
by sorry

end macaron_fraction_l401_40122


namespace find_S_l401_40132

theorem find_S (R S : ℕ) (h1 : 111111111111 - 222222 = (R + S)^2) (h2 : S > 0) : S = 333332 := by
  sorry

end find_S_l401_40132


namespace ios_department_larger_l401_40150

theorem ios_department_larger (n m : ℕ) : 
  (7 * n + 15 * m = 15 * n + 9 * m) → m > n := by
  sorry

end ios_department_larger_l401_40150


namespace angle_in_fourth_quadrant_l401_40102

-- Define the angle in degrees
def angle : ℤ := -3290

-- Function to normalize an angle to the range [0, 360)
def normalizeAngle (a : ℤ) : ℤ :=
  a % 360

-- Function to determine the quadrant of a normalized angle
def quadrant (a : ℤ) : ℕ :=
  if 0 ≤ a ∧ a < 90 then 1
  else if 90 ≤ a ∧ a < 180 then 2
  else if 180 ≤ a ∧ a < 270 then 3
  else 4

-- Theorem statement
theorem angle_in_fourth_quadrant :
  quadrant (normalizeAngle angle) = 4 := by
  sorry

end angle_in_fourth_quadrant_l401_40102


namespace geometric_sequence_sum_product_l401_40114

/-- A geometric sequence with common ratio r -/
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum_product (a : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  a 4 + a 8 = -2 →
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
sorry

end geometric_sequence_sum_product_l401_40114


namespace square_sum_difference_l401_40112

theorem square_sum_difference : 3^2 + 7^2 - 5^2 = 33 := by
  sorry

end square_sum_difference_l401_40112


namespace stream_speed_l401_40104

theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 39 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 13 := by
sorry

end stream_speed_l401_40104


namespace greatest_common_remainder_l401_40125

theorem greatest_common_remainder (a b c : ℕ) (h : a = 25 ∧ b = 57 ∧ c = 105) :
  ∃ (k : ℕ), k > 0 ∧ 
    (∃ (r : ℕ), a % k = r ∧ b % k = r ∧ c % k = r) ∧
    (∀ (m : ℕ), m > k → ¬(∃ (s : ℕ), a % m = s ∧ b % m = s ∧ c % m = s)) ∧
  k = 16 := by
sorry

end greatest_common_remainder_l401_40125


namespace factor_4t_squared_minus_64_l401_40148

theorem factor_4t_squared_minus_64 (t : ℝ) : 4 * t^2 - 64 = 4 * (t - 4) * (t + 4) := by
  sorry

end factor_4t_squared_minus_64_l401_40148


namespace square_perimeter_l401_40129

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 588 → perimeter = 56 * Real.sqrt 3 := by
  sorry

end square_perimeter_l401_40129


namespace cookies_remaining_l401_40173

theorem cookies_remaining (initial : ℕ) (given : ℕ) (eaten : ℕ) : 
  initial = 36 → given = 14 → eaten = 10 → initial - (given + eaten) = 12 := by
  sorry

end cookies_remaining_l401_40173


namespace largest_prime_mersenne_under_500_l401_40137

def mersenne_number (n : ℕ) : ℕ := 2^n - 1

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem largest_prime_mersenne_under_500 :
  ∀ n : ℕ, is_power_of_two n → 
    mersenne_number n < 500 → 
    Nat.Prime (mersenne_number n) → 
    mersenne_number n ≤ 3 :=
sorry

end largest_prime_mersenne_under_500_l401_40137


namespace parallel_line_through_point_l401_40146

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  let l1 : Line := { a := 2, b := -1, c := -1 }  -- 2x - y - 1 = 0
  let l2 : Line := { a := 2, b := -1, c := 0 }   -- 2x - y = 0
  parallel l1 l2 ∧ point_on_line 1 2 l2 := by
  sorry

end parallel_line_through_point_l401_40146


namespace largest_prime_factor_of_sum_of_divisors_180_l401_40110

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sum_of_divisors 180 ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ sum_of_divisors 180 → q ≤ p ∧ p = 13 := by
  sorry

end largest_prime_factor_of_sum_of_divisors_180_l401_40110


namespace modified_cube_vertices_l401_40169

/-- Calculates the number of vertices in a modified cube -/
def modifiedCubeVertices (initialSideLength : ℕ) (removedSideLength : ℕ) : ℕ :=
  8 * (3 * 4 - 3)

/-- Theorem stating that a cube of side length 5 with smaller cubes of side length 2 
    removed from each corner has 64 vertices -/
theorem modified_cube_vertices :
  modifiedCubeVertices 5 2 = 64 := by
  sorry

end modified_cube_vertices_l401_40169


namespace unique_data_set_l401_40136

def mean (xs : Fin 4 → ℕ+) : ℚ :=
  (xs 0 + xs 1 + xs 2 + xs 3 : ℚ) / 4

def median (xs : Fin 4 → ℕ+) : ℚ :=
  (xs 1 + xs 2 : ℚ) / 2

def variance (xs : Fin 4 → ℕ+) (μ : ℚ) : ℚ :=
  ((xs 0 - μ)^2 + (xs 1 - μ)^2 + (xs 2 - μ)^2 + (xs 3 - μ)^2) / 4

def stdDev (xs : Fin 4 → ℕ+) (μ : ℚ) : ℚ :=
  (variance xs μ).sqrt

theorem unique_data_set (xs : Fin 4 → ℕ+) 
    (h_ordered : ∀ i j : Fin 4, i ≤ j → xs i ≤ xs j)
    (h_mean : mean xs = 2)
    (h_median : median xs = 2)
    (h_stddev : stdDev xs 2 = 1) :
    xs 0 = 1 ∧ xs 1 = 1 ∧ xs 2 = 3 ∧ xs 3 = 3 := by
  sorry

#check unique_data_set

end unique_data_set_l401_40136


namespace original_triangle_area_l401_40139

/-- Given a triangle whose dimensions are quadrupled to form a new triangle with an area of 256 square feet,
    prove that the area of the original triangle is 16 square feet. -/
theorem original_triangle_area (original : ℝ) (new : ℝ) : 
  new = 256 → -- area of the new triangle
  new = original * 16 → -- relationship between new and original areas
  original = 16 := by
sorry

end original_triangle_area_l401_40139


namespace two_numbers_sum_and_lcm_l401_40119

theorem two_numbers_sum_and_lcm : ∃ (x y : ℕ), 
  x + y = 316 ∧ 
  Nat.lcm x y = 4560 ∧ 
  x = 199 ∧ 
  y = 117 := by
sorry

end two_numbers_sum_and_lcm_l401_40119


namespace polygon_diagonals_l401_40193

theorem polygon_diagonals (n : ℕ) (h : (n - 2) * 180 + 360 = 1800) : n - 3 = 7 := by
  sorry

end polygon_diagonals_l401_40193


namespace base_eight_digits_of_1728_l401_40159

theorem base_eight_digits_of_1728 : ∃ n : ℕ, n > 0 ∧ 8^(n-1) ≤ 1728 ∧ 1728 < 8^n ∧ n = 4 := by
  sorry

end base_eight_digits_of_1728_l401_40159


namespace waiter_tables_l401_40162

theorem waiter_tables (total_customers : ℕ) (people_per_table : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) :
  total_customers = 90 →
  people_per_table = women_per_table + men_per_table →
  women_per_table = 7 →
  men_per_table = 3 →
  total_customers / people_per_table = 9 :=
by
  sorry

end waiter_tables_l401_40162


namespace quadratic_form_sum_l401_40182

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ),
  (5 * x^2 - 45 * x - 500 = a * (x + b)^2 + c) ∧ (a + b + c = -605.75) := by
  sorry

end quadratic_form_sum_l401_40182


namespace minimize_sum_of_squares_l401_40116

theorem minimize_sum_of_squares (s : ℝ) (hs : s > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = s ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = s → x^2 + y^2 ≤ a^2 + b^2 ∧
  x^2 + y^2 = s^2 / 2 ∧ x = s / 2 ∧ y = s / 2 := by
  sorry

end minimize_sum_of_squares_l401_40116


namespace expression_evaluation_l401_40149

theorem expression_evaluation : 
  (1/8)^(1/3) - Real.log 2 / Real.log 3 * Real.log 27 / Real.log 4 + 
  (Real.log (Real.sqrt 2) / Real.log 10 + Real.log (Real.sqrt 5) / Real.log 10) = -1/2 := by
  sorry

end expression_evaluation_l401_40149


namespace masha_creates_more_words_l401_40128

/-- Represents a word as a list of characters -/
def Word := List Char

/-- Counts the number of distinct words formed by removing exactly two letters from a given word -/
def countDistinctWordsRemovingTwo (w : Word) : Nat :=
  sorry

/-- The word "ИНТЕГРИРОВАНИЕ" -/
def integrirovanie : Word :=
  ['И', 'Н', 'Т', 'Е', 'Г', 'Р', 'И', 'Р', 'О', 'В', 'А', 'Н', 'И', 'Е']

/-- The word "СУПЕРКОМПЬЮТЕР" -/
def superkomputer : Word :=
  ['С', 'У', 'П', 'Е', 'Р', 'К', 'О', 'М', 'П', 'Ь', 'Ю', 'Т', 'Е', 'Р']

theorem masha_creates_more_words :
  countDistinctWordsRemovingTwo superkomputer > countDistinctWordsRemovingTwo integrirovanie :=
sorry

end masha_creates_more_words_l401_40128


namespace rectangle_area_diagonal_l401_40111

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end rectangle_area_diagonal_l401_40111


namespace fixed_point_coordinates_l401_40109

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation y - 2 = k(x + 1) -/
def lineEquation (k : ℝ) (p : Point) : Prop :=
  p.y - 2 = k * (p.x + 1)

/-- The fixed point M satisfies the line equation for all k -/
def isFixedPoint (M : Point) : Prop :=
  ∀ k : ℝ, lineEquation k M

theorem fixed_point_coordinates :
  ∀ M : Point, isFixedPoint M → M = Point.mk (-1) 2 := by
  sorry

end fixed_point_coordinates_l401_40109


namespace line_slope_problem_l401_40123

/-- Given m > 0 and points (m,1) and (2,√m) on a line with slope 2m, prove m = 4 -/
theorem line_slope_problem (m : ℝ) (h1 : m > 0) : 
  (2 * m = (Real.sqrt m - 1) / (2 - m)) → m = 4 := by
  sorry

end line_slope_problem_l401_40123


namespace non_collinear_implies_nonzero_l401_40189

-- Define the vector type
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the collinearity relation
def collinear (a b : V) : Prop := ∃ (k : ℝ), a = k • b

-- State the theorem
theorem non_collinear_implies_nonzero (a b : V) : 
  ¬(collinear a b) → a ≠ 0 ∧ b ≠ 0 := by
  sorry

end non_collinear_implies_nonzero_l401_40189


namespace min_employees_for_pollution_monitoring_l401_40191

/-- Calculates the minimum number of employees needed given the number of employees
    who can monitor different types of pollution. -/
def minimum_employees (water : ℕ) (air : ℕ) (soil : ℕ) 
                      (water_air : ℕ) (air_soil : ℕ) (water_soil : ℕ) 
                      (all_three : ℕ) : ℕ :=
  water + air + soil - water_air - air_soil - water_soil + all_three

/-- Theorem stating that given the specific numbers from the problem,
    the minimum number of employees needed is 165. -/
theorem min_employees_for_pollution_monitoring : 
  minimum_employees 95 80 45 30 20 15 10 = 165 := by
  sorry

#eval minimum_employees 95 80 45 30 20 15 10

end min_employees_for_pollution_monitoring_l401_40191


namespace strictly_increasing_inverse_sum_identity_l401_40178

theorem strictly_increasing_inverse_sum_identity 
  (f : ℝ → ℝ) 
  (h_incr : ∀ x y, x < y → f x < f y) 
  (h_inv : Function.Bijective f) 
  (h_sum : ∀ x, f x + (Function.invFun f) x = 2 * x) : 
  ∃ b : ℝ, ∀ x, f x = x + b :=
sorry

end strictly_increasing_inverse_sum_identity_l401_40178


namespace merry_go_round_revolutions_merry_go_round_specific_case_l401_40168

/-- Given two circular paths with different radii, prove that the number of revolutions
    needed to cover the same distance is inversely proportional to their radii. -/
theorem merry_go_round_revolutions (r1 r2 n1 : ℝ) (hr1 : r1 > 0) (hr2 : r2 > 0) (hn1 : n1 > 0) :
  let n2 := (r1 * n1) / r2
  2 * Real.pi * r1 * n1 = 2 * Real.pi * r2 * n2 := by sorry

/-- Prove that for the specific case of r1 = 30, r2 = 10, and n1 = 36, 
    the number of revolutions n2 for the second path is 108. -/
theorem merry_go_round_specific_case :
  let r1 : ℝ := 30
  let r2 : ℝ := 10
  let n1 : ℝ := 36
  let n2 := (r1 * n1) / r2
  n2 = 108 := by sorry

end merry_go_round_revolutions_merry_go_round_specific_case_l401_40168


namespace mame_on_top_probability_l401_40187

/-- Represents a piece of paper with 8 quadrants -/
structure Paper :=
  (quadrants : Fin 8)

/-- The probability of a specific quadrant being on top -/
def probability_on_top (p : Paper) : ℚ :=
  1 / 8

/-- The quadrant where "MAME" is written -/
def mame_quadrant : Fin 8 := 0

/-- Theorem: The probability of "MAME" being on top is 1/8 -/
theorem mame_on_top_probability :
  probability_on_top {quadrants := mame_quadrant} = 1 / 8 := by
  sorry


end mame_on_top_probability_l401_40187


namespace square_sum_from_product_and_sum_l401_40183

theorem square_sum_from_product_and_sum (p q : ℝ) 
  (h1 : p * q = 9) 
  (h2 : p + q = 6) : 
  p^2 + q^2 = 18 := by
sorry

end square_sum_from_product_and_sum_l401_40183


namespace solve_for_t_l401_40115

theorem solve_for_t (s t : ℝ) (eq1 : 7 * s + 3 * t = 84) (eq2 : s = t - 3) : t = 10.5 := by
  sorry

end solve_for_t_l401_40115


namespace express_y_in_terms_of_x_l401_40165

theorem express_y_in_terms_of_x (x y : ℝ) : 2 * x - y = 4 → y = 2 * x - 4 := by
  sorry

end express_y_in_terms_of_x_l401_40165


namespace line_passes_through_I_III_IV_l401_40180

-- Define the line
def line (x : ℝ) : ℝ := 5 * x - 2

-- Define the quadrants
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem line_passes_through_I_III_IV :
  (∃ x y : ℝ, y = line x ∧ in_quadrant_I x y) ∧
  (∃ x y : ℝ, y = line x ∧ in_quadrant_III x y) ∧
  (∃ x y : ℝ, y = line x ∧ in_quadrant_IV x y) ∧
  ¬(∃ x y : ℝ, y = line x ∧ in_quadrant_II x y) :=
sorry

end line_passes_through_I_III_IV_l401_40180


namespace product_simplification_l401_40157

theorem product_simplification (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end product_simplification_l401_40157


namespace inequality_proof_l401_40124

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1 / b - c) * (b + 1 / c - a) + 
  (b + 1 / c - a) * (c + 1 / a - b) + 
  (c + 1 / a - b) * (a + 1 / b - c) ≥ 3 := by
sorry

end inequality_proof_l401_40124


namespace union_of_A_and_B_l401_40174

-- Define the sets A and B
def A : Set ℝ := {x | 3 - x > 0 ∧ x + 2 > 0}
def B : Set ℝ := {m | 3 > 2 * m - 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 3} := by sorry

end union_of_A_and_B_l401_40174


namespace player_b_always_wins_l401_40166

/-- Represents a player's move in the game -/
structure Move where
  value : ℕ

/-- Represents the state of the game after each round -/
structure GameState where
  round : ℕ
  player_a_move : Move
  player_b_move : Move
  player_a_score : ℕ
  player_b_score : ℕ

/-- The game setup with n rounds and increment d -/
structure GameSetup where
  n : ℕ
  d : ℕ
  h1 : n > 1
  h2 : d ≥ 1

/-- A strategy for player B -/
def PlayerBStrategy (setup : GameSetup) : GameState → Move := sorry

/-- Checks if a move is valid according to the game rules -/
def isValidMove (setup : GameSetup) (prev : GameState) (curr : Move) : Prop := sorry

/-- Calculates the score for a round -/
def calculateScore (a_move : Move) (b_move : Move) : ℕ × ℕ := sorry

/-- Simulates the game for n rounds -/
def playGame (setup : GameSetup) (strategy : GameState → Move) : GameState := sorry

/-- Theorem: Player B always has a winning strategy -/
theorem player_b_always_wins (setup : GameSetup) :
  ∃ (strategy : GameState → Move),
    (playGame setup strategy).player_b_score ≥ (playGame setup strategy).player_a_score := by
  sorry

end player_b_always_wins_l401_40166


namespace S_infinite_l401_40107

/-- Sum of positive integer divisors of n -/
def d (n : ℕ) : ℕ := sorry

/-- Euler's totient function: count of integers in [0,n] coprime with n -/
def φ (n : ℕ) : ℕ := sorry

/-- The set of integers n for which d(n) * φ(n) is a perfect square -/
def S : Set ℕ := {n : ℕ | ∃ k : ℕ, d n * φ n = k^2}

/-- The main theorem: S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end S_infinite_l401_40107


namespace incorrect_vs_correct_operations_l401_40151

theorem incorrect_vs_correct_operations (x : ℝ) :
  (x / 8 - 12 = 18) → (x * 8 * 12 = 23040) := by
  sorry

end incorrect_vs_correct_operations_l401_40151


namespace man_upstream_speed_l401_40113

/-- Calculates the upstream speed of a man given his downstream speed and the stream speed. -/
def upstream_speed (downstream_speed stream_speed : ℝ) : ℝ :=
  downstream_speed - 2 * stream_speed

/-- Theorem stating that given a downstream speed of 14 km/h and a stream speed of 3 km/h, 
    the upstream speed is 8 km/h. -/
theorem man_upstream_speed : 
  upstream_speed 14 3 = 8 := by
  sorry

end man_upstream_speed_l401_40113


namespace power_of_i_product_l401_40194

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_product : i^45 * i^105 = -1 := by sorry

end power_of_i_product_l401_40194


namespace longest_side_of_triangle_with_incircle_l401_40108

/-- A triangle with an incircle -/
structure TriangleWithIncircle where
  /-- The radius of the incircle -/
  r : ℝ
  /-- The length of the segment of the side divided by the tangent point -/
  a : ℝ
  /-- The length of the other segment of the side divided by the tangent point -/
  b : ℝ
  /-- The length of the longest side of the triangle -/
  longest_side : ℝ

/-- Theorem: In a triangle with an incircle of radius 5 units, where the incircle is tangent
    to one side at a point dividing it into segments of 9 and 5 units, the length of the
    longest side is 18 units. -/
theorem longest_side_of_triangle_with_incircle
  (t : TriangleWithIncircle)
  (h1 : t.r = 5)
  (h2 : t.a = 9)
  (h3 : t.b = 5) :
  t.longest_side = 18 := by
  sorry

end longest_side_of_triangle_with_incircle_l401_40108


namespace product_of_roots_l401_40105

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 26 → ∃ y : ℝ, (x + 3) * (x - 4) = 26 ∧ (y + 3) * (y - 4) = 26 ∧ x * y = -38 := by
  sorry

end product_of_roots_l401_40105


namespace card_count_l401_40127

theorem card_count (black red spades diamonds hearts clubs : ℕ) : 
  black = 7 →
  red = 6 →
  diamonds = 2 * spades →
  hearts = 2 * diamonds →
  clubs = 6 →
  black = spades + clubs →
  red = diamonds + hearts →
  spades + diamonds + hearts + clubs = 13 :=
by
  sorry

end card_count_l401_40127


namespace ellipse1_passes_through_points_ellipse2_passes_through_point_ellipse3_passes_through_point_ellipse2_axis_ratio_ellipse3_axis_ratio_l401_40185

-- Define the ellipse equations
def ellipse1 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 3 = 1
def ellipse2 (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1
def ellipse3 (x y : ℝ) : Prop := y^2 / 81 + x^2 / 9 = 1

-- Theorem for the first ellipse
theorem ellipse1_passes_through_points :
  ellipse1 (Real.sqrt 6) 1 ∧ ellipse1 (-Real.sqrt 3) (-Real.sqrt 2) := by sorry

-- Theorems for the second and third ellipses
theorem ellipse2_passes_through_point : ellipse2 3 0 := by sorry
theorem ellipse3_passes_through_point : ellipse3 3 0 := by sorry

theorem ellipse2_axis_ratio :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a = 3 * b ∧
  ∀ (x y : ℝ), ellipse2 x y ↔ x^2 / a^2 + y^2 / b^2 = 1 := by sorry

theorem ellipse3_axis_ratio :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a = 3 * b ∧
  ∀ (x y : ℝ), ellipse3 x y ↔ y^2 / a^2 + x^2 / b^2 = 1 := by sorry

end ellipse1_passes_through_points_ellipse2_passes_through_point_ellipse3_passes_through_point_ellipse2_axis_ratio_ellipse3_axis_ratio_l401_40185


namespace three_independent_events_probability_l401_40142

/-- Given three independent events with equal probability, 
    prove that the probability of all three events occurring simultaneously 
    is the cube of the individual probability -/
theorem three_independent_events_probability 
  (p : ℝ) 
  (h1 : 0 ≤ p) 
  (h2 : p ≤ 1) 
  (h3 : p = 1/3) : 
  p * p * p = 1/27 := by
  sorry

end three_independent_events_probability_l401_40142


namespace shooter_score_problem_l401_40161

/-- A shooter's competition score problem -/
theorem shooter_score_problem 
  (first_six_shots : ℕ) 
  (record : ℕ) 
  (h1 : first_six_shots = 52) 
  (h2 : record = 89) 
  (h3 : ∀ shot, shot ∈ Set.Icc 1 10) :
  /- (1) Minimum score on 7th shot to break record -/
  (∃ x : ℕ, x ≥ 8 ∧ first_six_shots + x + 30 > record) ∧
  /- (2) Number of 10s needed in last 3 shots if 7th shot is 8 -/
  (first_six_shots + 8 + 30 > record) ∧
  /- (3) Necessity of at least one 10 in last 3 shots if 7th shot is 10 -/
  (∃ x y z : ℕ, x ∈ Set.Icc 1 10 ∧ y ∈ Set.Icc 1 10 ∧ z ∈ Set.Icc 1 10 ∧
    first_six_shots + 10 + x + y + z > record ∧ (x = 10 ∨ y = 10 ∨ z = 10)) := by
  sorry


end shooter_score_problem_l401_40161


namespace barium_oxide_weight_l401_40172

/-- The atomic weight of Barium in g/mol -/
def barium_weight : ℝ := 137.33

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (ba_count o_count : ℕ) : ℝ :=
  ba_count * barium_weight + o_count * oxygen_weight

/-- Theorem: The molecular weight of a compound with 1 Barium and 1 Oxygen atom is 153.33 g/mol -/
theorem barium_oxide_weight : molecular_weight 1 1 = 153.33 := by
  sorry

end barium_oxide_weight_l401_40172


namespace thirty_five_power_pq_l401_40121

theorem thirty_five_power_pq (p q : ℤ) (A B : ℝ) (hA : A = 5^p) (hB : B = 7^q) :
  A^q * B^p = 35^(p*q) := by
  sorry

end thirty_five_power_pq_l401_40121


namespace arithmetic_mean_problem_l401_40199

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 74 → a = 28 := by
  sorry

end arithmetic_mean_problem_l401_40199


namespace periodicity_2pi_l401_40181

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * Real.cos y

/-- The periodicity theorem -/
theorem periodicity_2pi (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (x + 2 * Real.pi) = f x :=
by sorry

end periodicity_2pi_l401_40181


namespace root_sum_reciprocal_l401_40158

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 30*x^2 + 105*x - 114 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 30*s^2 + 105*s - 114) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1/A + 1/B + 1/C = 300 := by
sorry

end root_sum_reciprocal_l401_40158


namespace hyperbola_circle_tangent_radius_l401_40141

/-- The radius of a circle that is tangent to the asymptotes of a specific hyperbola -/
theorem hyperbola_circle_tangent_radius : ∀ (r : ℝ), r > 0 →
  (∀ (x y : ℝ), x^2 / 9 - y^2 / 4 = 1 →
    (∃ (t : ℝ), (x - 3)^2 + y^2 = r^2 ∧
      (y = (2/3) * x ∨ y = -(2/3) * x) ∧
      (∀ (x' y' : ℝ), (x' - 3)^2 + y'^2 < r^2 →
        y' ≠ (2/3) * x' ∧ y' ≠ -(2/3) * x'))) →
  r = 6 * Real.sqrt 13 / 13 := by
sorry

end hyperbola_circle_tangent_radius_l401_40141


namespace somu_father_age_ratio_l401_40135

/-- Proves that the ratio of Somu's age to his father's age is 1:3 given the conditions -/
theorem somu_father_age_ratio :
  ∀ (somu_age father_age : ℕ),
  somu_age = 18 →
  somu_age - 9 = (father_age - 9) / 5 →
  ∃ (k : ℕ), k > 0 ∧ somu_age * 3 = father_age * k ∧ k = 1 :=
by sorry

end somu_father_age_ratio_l401_40135


namespace matching_instrument_probability_l401_40152

/-- The probability of selecting a matching cello-viola pair -/
theorem matching_instrument_probability
  (total_cellos : ℕ)
  (total_violas : ℕ)
  (matching_pairs : ℕ)
  (h1 : total_cellos = 800)
  (h2 : total_violas = 600)
  (h3 : matching_pairs = 100) :
  (matching_pairs : ℚ) / (total_cellos * total_violas) = 1 / 4800 :=
by sorry

end matching_instrument_probability_l401_40152


namespace binomial_expansion_coefficient_relation_l401_40192

theorem binomial_expansion_coefficient_relation (n : ℕ) : 
  (2 * n * (n - 1) = 7 * (2 * n)) → n = 8 := by
  sorry

end binomial_expansion_coefficient_relation_l401_40192


namespace tangent_line_slope_l401_40131

/-- The curve function f(x) = x³ - 3x² + 2x --/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, (k * x₀ = f x₀) ∧ (∀ x : ℝ, k * x ≤ f x) ∧ (k = f' x₀)) →
  (k = 2 ∨ k = -1/4) :=
sorry

end tangent_line_slope_l401_40131


namespace inequality_proof_l401_40101

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : n ≥ 2) :
  (1 + x)^n + (1 - x)^n < 2^n := by
  sorry

end inequality_proof_l401_40101
