import Mathlib

namespace exists_special_function_l2369_236903

/-- A function from pairs of positive integers to positive integers -/
def PositiveIntegerFunction := ℕ+ → ℕ+ → ℕ+

/-- Predicate for a function being a polynomial in one variable when the other is fixed -/
def IsPolynomialInOneVariable (f : PositiveIntegerFunction) : Prop :=
  (∀ x : ℕ+, ∃ Px : ℕ+ → ℕ+, ∀ y : ℕ+, f x y = Px y) ∧
  (∀ y : ℕ+, ∃ Qy : ℕ+ → ℕ+, ∀ x : ℕ+, f x y = Qy x)

/-- Predicate for a function not being a polynomial in both variables -/
def IsNotPolynomialInBothVariables (f : PositiveIntegerFunction) : Prop :=
  ¬∃ P : ℕ+ → ℕ+ → ℕ+, ∀ x y : ℕ+, f x y = P x y

/-- The main theorem stating the existence of a function with the required properties -/
theorem exists_special_function : 
  ∃ f : PositiveIntegerFunction, 
    IsPolynomialInOneVariable f ∧ IsNotPolynomialInBothVariables f := by
  sorry

end exists_special_function_l2369_236903


namespace line_inclination_angle_l2369_236905

/-- The inclination angle of a line is the angle between the positive x-axis and the line, measured counterclockwise. -/
def inclination_angle (a b c : ℝ) : ℝ := sorry

/-- A line is defined by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem line_inclination_angle :
  let l : Line := { a := 1, b := 1, c := 1 }
  inclination_angle l.a l.b l.c = 135 * π / 180 := by sorry

end line_inclination_angle_l2369_236905


namespace necessary_not_sufficient_negation_l2369_236954

theorem necessary_not_sufficient_negation (p q : Prop) 
  (h1 : q → p)  -- p is necessary for q
  (h2 : ¬(p → q))  -- p is not sufficient for q
  : (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
  sorry

end necessary_not_sufficient_negation_l2369_236954


namespace sin_2alpha_minus_pi_sixth_l2369_236908

theorem sin_2alpha_minus_pi_sixth (α : ℝ) 
  (h : Real.cos (α + π / 6) = Real.sqrt 3 / 3) : 
  Real.sin (2 * α - π / 6) = 1 / 3 := by
  sorry

end sin_2alpha_minus_pi_sixth_l2369_236908


namespace correct_product_l2369_236975

theorem correct_product (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ b ≥ 10 ∧ b < 100 →
  (a * (10 * (b % 10) + (b / 10)) = 143) →
  a * b = 341 :=
by
  sorry

end correct_product_l2369_236975


namespace parallel_vectors_imply_m_half_l2369_236906

/-- Given two non-zero parallel vectors (m^2-1, m+1) and (1, -2), prove that m = 1/2 -/
theorem parallel_vectors_imply_m_half (m : ℝ) : 
  (m^2 - 1 ≠ 0 ∨ m + 1 ≠ 0) →  -- Vector a is non-zero
  (∃ (k : ℝ), k ≠ 0 ∧ k * (1 : ℝ) = m^2 - 1 ∧ k * (-2 : ℝ) = m + 1) →  -- Vectors are parallel
  m = 1/2 := by
  sorry

end parallel_vectors_imply_m_half_l2369_236906


namespace initial_speed_is_850_l2369_236940

/-- Represents the airplane's journey with given conditions -/
structure AirplaneJourney where
  totalDistance : ℝ
  distanceBeforeLanding : ℝ
  landingDuration : ℝ
  speedReduction : ℝ
  totalTime : ℝ

/-- Calculates the initial speed of the airplane given the journey parameters -/
def initialSpeed (journey : AirplaneJourney) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the initial speed is 850 km/h for the given conditions -/
theorem initial_speed_is_850 :
  let journey : AirplaneJourney := {
    totalDistance := 2900
    distanceBeforeLanding := 1700
    landingDuration := 1.5
    speedReduction := 50
    totalTime := 5
  }
  initialSpeed journey = 850 := by
  sorry

end initial_speed_is_850_l2369_236940


namespace inequality_solution_l2369_236981

theorem inequality_solution (x : ℝ) : 
  (∃ a : ℝ, a ∈ Set.Icc (-1) 2 ∧ (2 - a) * x^3 + (1 - 2*a) * x^2 - 6*x + 5 + 4*a - a^2 < 0) ↔ 
  (x < -2 ∨ (0 < x ∧ x < 1) ∨ 1 < x) := by
sorry

end inequality_solution_l2369_236981


namespace car_speed_problem_l2369_236948

/-- Theorem: Given two cars starting from opposite ends of a 60-mile highway at the same time,
    with one car traveling at speed x mph and the other at 17 mph, if they meet after 2 hours,
    then the speed x of the first car is 13 mph. -/
theorem car_speed_problem (x : ℝ) :
  (x > 0) →  -- Assuming positive speed for the first car
  (2 * x + 2 * 17 = 60) →  -- Distance traveled by both cars equals highway length
  x = 13 := by
  sorry

end car_speed_problem_l2369_236948


namespace batsman_matches_l2369_236960

theorem batsman_matches (x : ℕ) 
  (h1 : x > 0)
  (h2 : (30 * x + 15 * 10) / (x + 10) = 25) : 
  x = 20 := by
sorry

end batsman_matches_l2369_236960


namespace area_between_semicircles_l2369_236910

/-- Given a semicircle with diameter D, which is divided into two parts,
    and semicircles constructed on each part inside the given semicircle,
    the area enclosed between the three semicircles is equal to πCD²/4,
    where CD is the length of the perpendicular from the division point to the semicircle. -/
theorem area_between_semicircles (D r : ℝ) (h : 0 < r ∧ r < D) : 
  let R := D / 2
  let area := π * r * (R - r)
  let CD := Real.sqrt (2 * r * (D - r))
  area = π * CD^2 / 4 := by sorry

end area_between_semicircles_l2369_236910


namespace max_sum_of_square_roots_l2369_236964

theorem max_sum_of_square_roots (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end max_sum_of_square_roots_l2369_236964


namespace physics_majors_consecutive_probability_l2369_236937

/-- The number of people sitting at the round table -/
def total_people : ℕ := 10

/-- The number of physics majors -/
def physics_majors : ℕ := 3

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 2

/-- The number of biology majors -/
def biology_majors : ℕ := 1

/-- The probability of all physics majors sitting in consecutive seats -/
def consecutive_physics_probability : ℚ := 1 / 24

theorem physics_majors_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let valid_arrangements := 3 * Nat.factorial (total_people - physics_majors)
  consecutive_physics_probability = (valid_arrangements : ℚ) / total_arrangements :=
sorry

end physics_majors_consecutive_probability_l2369_236937


namespace cube_remainder_sum_quotient_l2369_236949

def cube_rem_16 (n : ℕ) : ℕ := (n^3) % 16

def distinct_remainders : Finset ℕ :=
  (Finset.range 15).image cube_rem_16

theorem cube_remainder_sum_quotient :
  (Finset.sum distinct_remainders id) / 16 = 2 := by
  sorry

end cube_remainder_sum_quotient_l2369_236949


namespace hexagon_angle_measure_l2369_236945

theorem hexagon_angle_measure (a b c d e : ℝ) (h1 : a = 134) (h2 : b = 98) (h3 : c = 120) (h4 : d = 110) (h5 : e = 96) :
  720 - (a + b + c + d + e) = 162 := by
  sorry

end hexagon_angle_measure_l2369_236945


namespace exponential_equation_solution_l2369_236987

theorem exponential_equation_solution :
  ∃ x : ℝ, (4 : ℝ)^x * (4 : ℝ)^x * (4 : ℝ)^x = (16 : ℝ)^5 ∧ x = 10/3 := by
  sorry

end exponential_equation_solution_l2369_236987


namespace fraction_equation_solution_l2369_236933

theorem fraction_equation_solution :
  ∃ x : ℚ, (5 * x + 3) / (7 * x - 4) = 4128 / 4386 ∧ x = 115 / 27 := by
  sorry

end fraction_equation_solution_l2369_236933


namespace minimal_kamber_group_common_meal_l2369_236927

/-- The number of citizens in the city -/
def num_citizens : ℕ := 2017

/-- The number of meal types available -/
def num_meals : ℕ := 25

/-- A citizen is represented by a natural number -/
def Citizen := Fin num_citizens

/-- A meal is represented by a natural number -/
def Meal := Fin num_meals

/-- Predicate indicating whether a citizen likes a meal -/
def likes (c : Citizen) (m : Meal) : Prop := sorry

/-- A set of citizens is a suitable list if each meal is liked by at least one person in the set -/
def is_suitable_list (s : Set Citizen) : Prop :=
  ∀ m : Meal, ∃ c ∈ s, likes c m

/-- A set of citizens is a kamber group if it contains at least one person from each suitable list -/
def is_kamber_group (k : Set Citizen) : Prop :=
  ∀ s : Set Citizen, is_suitable_list s → (∃ c ∈ k, c ∈ s)

/-- A kamber group is minimal if no proper subset is also a kamber group -/
def is_minimal_kamber_group (k : Set Citizen) : Prop :=
  is_kamber_group k ∧ ∀ k' ⊂ k, ¬is_kamber_group k'

theorem minimal_kamber_group_common_meal (k : Set Citizen) 
  (h : is_minimal_kamber_group k) : 
  ∃ m : Meal, ∀ c ∈ k, likes c m := by sorry


end minimal_kamber_group_common_meal_l2369_236927


namespace scientific_notation_of_nanometers_l2369_236902

theorem scientific_notation_of_nanometers : 
  ∃ (a : ℝ) (n : ℤ), 0.000000007 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7 ∧ n = -9 :=
by sorry

end scientific_notation_of_nanometers_l2369_236902


namespace ten_streets_intersections_l2369_236935

/-- The number of intersections created by n non-parallel straight streets -/
def intersections (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- Theorem: 10 non-parallel straight streets create 45 intersections -/
theorem ten_streets_intersections :
  intersections 10 = 45 := by
  sorry

end ten_streets_intersections_l2369_236935


namespace arithmetic_sequence_first_term_l2369_236900

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a₃_eq : a 3 = -2
  aₙ_eq : ∃ n : ℕ, a n = 3/2
  Sₙ_eq : ∃ n : ℕ, (n : ℚ) * (a 1 + a n) / 2 = -15/2

/-- The first term of the arithmetic sequence is either -3 or -19/6 -/
theorem arithmetic_sequence_first_term (seq : ArithmeticSequence) :
  seq.a 1 = -3 ∨ seq.a 1 = -19/6 := by
  sorry

end arithmetic_sequence_first_term_l2369_236900


namespace pizza_toppings_l2369_236965

theorem pizza_toppings (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24)
  (h_pepperoni : pepperoni_slices = 15)
  (h_mushroom : mushroom_slices = 15)
  (h_at_least_one : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ slice ∈ Finset.range mushroom_slices)) :
  (Finset.range pepperoni_slices ∩ Finset.range mushroom_slices).card = 6 := by
sorry

end pizza_toppings_l2369_236965


namespace least_faces_combined_l2369_236984

/-- Represents a fair die with faces numbered from 1 to n -/
structure Die (n : ℕ) where
  faces : Fin n → ℕ
  is_fair : ∀ i : Fin n, faces i = i.val + 1

/-- Represents a pair of dice -/
structure DicePair (a b : ℕ) where
  die1 : Die a
  die2 : Die b
  die2_numbering : ∀ i : Fin b, die2.faces i = 2 * i.val + 2

/-- Probability of rolling a specific sum with a pair of dice -/
def prob_sum (d : DicePair a b) (sum : ℕ) : ℚ :=
  (Fintype.card {(i, j) : Fin a × Fin b | d.die1.faces i + d.die2.faces j = sum} : ℚ) / (a * b)

/-- The main theorem stating the least possible number of faces on two dice combined -/
theorem least_faces_combined (a b : ℕ) (d : DicePair a b) :
  (prob_sum d 8 = 2 * prob_sum d 12) →
  (prob_sum d 13 = 2 * prob_sum d 8) →
  a + b ≥ 11 :=
sorry

end least_faces_combined_l2369_236984


namespace expected_red_pairs_51_17_l2369_236918

/-- The expected number of red adjacent pairs in a circular arrangement of cards -/
def expected_red_pairs (total_cards : ℕ) (red_cards : ℕ) : ℚ :=
  (total_cards : ℚ) * (red_cards : ℚ) / (total_cards : ℚ) * ((red_cards - 1) : ℚ) / ((total_cards - 1) : ℚ)

/-- Theorem: Expected number of red adjacent pairs in a specific card arrangement -/
theorem expected_red_pairs_51_17 :
  expected_red_pairs 51 17 = 464 / 85 := by
  sorry

end expected_red_pairs_51_17_l2369_236918


namespace num_dogs_correct_l2369_236993

/-- The number of dogs Ella owns -/
def num_dogs : ℕ := 2

/-- The amount of food each dog eats per day (in scoops) -/
def food_per_dog : ℚ := 1/8

/-- The total amount of food eaten by all dogs per day (in scoops) -/
def total_food : ℚ := 1/4

/-- Theorem stating that the number of dogs is correct given the food consumption -/
theorem num_dogs_correct : (num_dogs : ℚ) * food_per_dog = total_food := by sorry

end num_dogs_correct_l2369_236993


namespace football_exercise_calories_l2369_236991

/-- Calculates the total calories burned during a stair-climbing exercise. -/
def total_calories_burned (round_trips : ℕ) (stairs_one_way : ℕ) (calories_per_stair : ℕ) : ℕ :=
  round_trips * (2 * stairs_one_way) * calories_per_stair

/-- Proves that given the specific conditions, the total calories burned is 16200. -/
theorem football_exercise_calories : 
  total_calories_burned 60 45 3 = 16200 := by
  sorry

end football_exercise_calories_l2369_236991


namespace rabbit_travel_time_l2369_236977

def rabbit_speed : ℝ := 10  -- miles per hour
def distance : ℝ := 3  -- miles

theorem rabbit_travel_time : 
  (distance / rabbit_speed) * 60 = 18 := by sorry

end rabbit_travel_time_l2369_236977


namespace initial_train_distance_l2369_236967

/-- Calculates the initial distance between two trains given their lengths, speeds, and time to meet. -/
theorem initial_train_distance
  (length1 : ℝ)
  (length2 : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (time : ℝ)
  (h1 : length1 = 100)
  (h2 : length2 = 200)
  (h3 : speed1 = 54)
  (h4 : speed2 = 72)
  (h5 : time = 1.999840012798976) :
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  let distance_covered := relative_speed * time * 3600
  distance_covered - (length1 + length2) = 251680.84161264498 :=
by sorry

end initial_train_distance_l2369_236967


namespace selenes_purchase_cost_l2369_236928

/-- The total cost of Selene's purchase after discount -/
def total_cost_after_discount (camera_price : ℝ) (frame_price : ℝ) (num_cameras : ℕ) (num_frames : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_before_discount := camera_price * num_cameras + frame_price * num_frames
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

/-- Theorem stating that Selene's total payment is $551 -/
theorem selenes_purchase_cost : 
  total_cost_after_discount 110 120 2 3 0.05 = 551 := by
  sorry

#eval total_cost_after_discount 110 120 2 3 0.05

end selenes_purchase_cost_l2369_236928


namespace max_boat_distance_xiaohu_max_distance_l2369_236947

/-- Calculates the maximum distance a boat can travel in a river with given conditions --/
theorem max_boat_distance (total_time : ℝ) (boat_speed : ℝ) (current_speed : ℝ) 
  (paddle_time : ℝ) (break_time : ℝ) : ℝ :=
  let total_minutes : ℝ := total_time * 60
  let cycle_time : ℝ := paddle_time + break_time
  let num_cycles : ℝ := total_minutes / cycle_time
  let total_break_time : ℝ := num_cycles * break_time
  let effective_paddle_time : ℝ := total_minutes - total_break_time - total_break_time
  let upstream_speed : ℝ := boat_speed - current_speed
  let downstream_speed : ℝ := boat_speed + current_speed
  let downstream_ratio : ℝ := downstream_speed / (upstream_speed + downstream_speed)
  let downstream_paddle_time : ℝ := downstream_ratio * effective_paddle_time
  let downstream_distance : ℝ := downstream_speed * (downstream_paddle_time / 60)
  let drift_distance : ℝ := current_speed * (break_time / 60)
  downstream_distance + drift_distance

/-- Proves that the maximum distance Xiaohu can be from the rental place is 1.375 km --/
theorem xiaohu_max_distance : 
  max_boat_distance 2 3 1.5 30 10 = 1.375 := by sorry

end max_boat_distance_xiaohu_max_distance_l2369_236947


namespace condition_sufficiency_not_necessity_l2369_236939

theorem condition_sufficiency_not_necessity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 < 1 → a * b + 1 > a + b) ∧ 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b + 1 > a + b ∧ a^2 + b^2 ≥ 1) := by
sorry

end condition_sufficiency_not_necessity_l2369_236939


namespace at_most_one_zero_point_l2369_236989

/-- A decreasing function on a closed interval has at most one zero point -/
theorem at_most_one_zero_point 
  {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b) (h_decr : ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x) :
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 ∨ ∀ x, a ≤ x → x ≤ b → f x ≠ 0 :=
sorry

end at_most_one_zero_point_l2369_236989


namespace cubic_function_property_l2369_236925

/-- Given a function f(x) = ax³ + bx + 1, prove that if f(a) = 8, then f(-a) = -6 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + b * x + 1
  f a = 8 → f (-a) = -6 := by
sorry

end cubic_function_property_l2369_236925


namespace cube_volume_ratio_l2369_236976

-- Define the edge lengths
def edge_length_small : ℚ := 4
def edge_length_large : ℚ := 24  -- 2 feet = 24 inches

-- Define the volume ratio
def volume_ratio : ℚ := (edge_length_small / edge_length_large) ^ 3

-- Theorem statement
theorem cube_volume_ratio :
  volume_ratio = 1 / 216 :=
by sorry

end cube_volume_ratio_l2369_236976


namespace student_multiplication_problem_l2369_236952

theorem student_multiplication_problem (x : ℝ) : 30 * x - 138 = 102 → x = 8 := by
  sorry

end student_multiplication_problem_l2369_236952


namespace expression_value_theorem_l2369_236990

theorem expression_value_theorem (a b c d m : ℝ) :
  (a = -b) →
  (c * d = 1) →
  (|m| = 5) →
  (2 * (a + b) - 3 * c * d + m = 2 ∨ 2 * (a + b) - 3 * c * d + m = -8) :=
by sorry

end expression_value_theorem_l2369_236990


namespace xyz_value_l2369_236957

theorem xyz_value (a b c x y z : ℂ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq1 : a = (b + c) / (x + 1))
  (eq2 : b = (a + c) / (y + 1))
  (eq3 : c = (a + b) / (z + 1))
  (sum_prod : x * y + x * z + y * z = 9)
  (sum : x + y + z = 5) :
  x * y * z = 13 := by
  sorry

end xyz_value_l2369_236957


namespace min_nSn_arithmetic_sequence_l2369_236997

/-- Arithmetic sequence sum function -/
def S (a₁ d : ℚ) (n : ℕ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Product of n and S_n -/
def nSn (a₁ d : ℚ) (n : ℕ) : ℚ := n * S a₁ d n

theorem min_nSn_arithmetic_sequence :
  ∃ (a₁ d : ℚ),
    S a₁ d 10 = 0 ∧
    S a₁ d 15 = 25 ∧
    (∀ (n : ℕ), n > 0 → nSn a₁ d n ≥ -48) ∧
    (∃ (n : ℕ), n > 0 ∧ nSn a₁ d n = -48) := by
  sorry

end min_nSn_arithmetic_sequence_l2369_236997


namespace terminating_decimal_expansion_of_7_625_l2369_236986

theorem terminating_decimal_expansion_of_7_625 :
  (7 : ℚ) / 625 = (112 : ℚ) / 10000 := by sorry

end terminating_decimal_expansion_of_7_625_l2369_236986


namespace rectangle_to_square_l2369_236961

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Theorem: A 12 × 3 rectangle can be cut into three equal parts that form a 6 × 6 square -/
theorem rectangle_to_square (rect : Rectangle) (sq : Square) : 
  rect.width = 12 ∧ rect.height = 3 ∧ sq.side = 6 →
  ∃ (part_width part_height : ℝ),
    part_width * part_height = rect.width * rect.height / 3 ∧
    3 * part_width = sq.side ∧
    part_height = sq.side :=
by sorry

end rectangle_to_square_l2369_236961


namespace twentyFiveCentCoins_l2369_236929

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  five : ℕ
  ten : ℕ
  twentyFive : ℕ

/-- Calculates the total number of coins -/
def totalCoins (c : CoinCounts) : ℕ := c.five + c.ten + c.twentyFive

/-- Calculates the number of different values that can be obtained -/
def differentValues (c : CoinCounts) : ℕ :=
  74 - 4 * c.five - 3 * c.ten

/-- Main theorem -/
theorem twentyFiveCentCoins (c : CoinCounts) :
  totalCoins c = 15 ∧ differentValues c = 30 → c.twentyFive = 2 := by
  sorry

end twentyFiveCentCoins_l2369_236929


namespace car_speed_problem_l2369_236979

/-- Two cars start from the same point and travel in opposite directions. -/
structure TwoCars where
  car1_speed : ℝ
  car2_speed : ℝ
  travel_time : ℝ
  total_distance : ℝ

/-- The theorem states that given the conditions of the problem, 
    the speed of the second car is 50 mph. -/
theorem car_speed_problem (cars : TwoCars) 
  (h1 : cars.car1_speed = 40)
  (h2 : cars.travel_time = 5)
  (h3 : cars.total_distance = 450) :
  cars.car2_speed = 50 := by
  sorry

end car_speed_problem_l2369_236979


namespace Q_proper_subset_P_l2369_236998

def P : Set ℝ := {x : ℝ | x ≥ 1}
def Q : Set ℝ := {2, 3}

theorem Q_proper_subset_P : Q ⊂ P := by sorry

end Q_proper_subset_P_l2369_236998


namespace two_person_subcommittees_l2369_236913

/-- The number of combinations of n items taken k at a time -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the original committee -/
def committeeSize : ℕ := 8

/-- The size of the sub-committee -/
def subCommitteeSize : ℕ := 2

/-- The number of different two-person sub-committees that can be selected from a committee of eight people -/
theorem two_person_subcommittees : choose committeeSize subCommitteeSize = 28 := by
  sorry

end two_person_subcommittees_l2369_236913


namespace min_queries_for_100_sets_l2369_236992

/-- Represents a query operation on two sets -/
inductive Query
  | intersect : ℕ → ℕ → Query
  | union : ℕ → ℕ → Query

/-- The result of a query operation -/
def QueryResult := Set ℕ

/-- A function that performs a query on two sets -/
def performQuery : Query → (ℕ → Set ℕ) → QueryResult := sorry

/-- A strategy is a sequence of queries -/
def Strategy := List Query

/-- Checks if a strategy determines all sets -/
def determinesAllSets (s : Strategy) (n : ℕ) : Prop := sorry

/-- The main theorem: 100 queries are necessary and sufficient -/
theorem min_queries_for_100_sets :
  (∃ (s : Strategy), s.length = 100 ∧ determinesAllSets s 100) ∧
  (∀ (s : Strategy), s.length < 100 → ¬determinesAllSets s 100) := by sorry

end min_queries_for_100_sets_l2369_236992


namespace fraction_to_decimal_subtraction_l2369_236956

theorem fraction_to_decimal_subtraction : (3 : ℚ) / 40 - 0.005 = 0.070 := by
  sorry

end fraction_to_decimal_subtraction_l2369_236956


namespace sum_of_digits_in_repeating_decimal_l2369_236923

/-- The repeating decimal representation of 3/11 -/
def repeating_decimal : ℚ := 3 / 11

/-- The first digit in the repeating part of the decimal -/
def a : ℕ := 2

/-- The second digit in the repeating part of the decimal -/
def b : ℕ := 7

/-- Theorem stating that the sum of a and b is 9 -/
theorem sum_of_digits_in_repeating_decimal : a + b = 9 := by sorry

end sum_of_digits_in_repeating_decimal_l2369_236923


namespace simplest_fraction_sum_l2369_236912

theorem simplest_fraction_sum (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  (a : ℚ) / b = 0.84375 ∧
  ∀ (c d : ℕ), c > 0 → d > 0 → (c : ℚ) / d = 0.84375 → a ≤ c ∧ b ≤ d →
  a + b = 59 := by
sorry

end simplest_fraction_sum_l2369_236912


namespace coinciding_rest_days_count_l2369_236931

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 7

/-- Barb's schedule cycle length -/
def barb_cycle : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 500

/-- Al's rest days in his cycle -/
def al_rest_days : Finset ℕ := {6, 7}

/-- Barb's rest day in her cycle -/
def barb_rest_day : ℕ := 5

/-- The number of days both Al and Barb rest in the same 35-day period -/
def coinciding_rest_days_per_cycle : ℕ := 1

theorem coinciding_rest_days_count : 
  (total_days / (al_cycle * barb_cycle)) * coinciding_rest_days_per_cycle = 14 := by
  sorry

end coinciding_rest_days_count_l2369_236931


namespace time_reduction_fraction_l2369_236911

theorem time_reduction_fraction (actual_speed : ℝ) (speed_increase : ℝ) : 
  actual_speed = 36.000000000000014 →
  speed_increase = 18 →
  (actual_speed / (actual_speed + speed_increase)) = 2/3 :=
by sorry

end time_reduction_fraction_l2369_236911


namespace hyperbola_eccentricity_l2369_236973

/-- A hyperbola with given asymptotes -/
structure Hyperbola where
  /-- The asymptotes of the hyperbola are x ± 2y = 0 -/
  asymptotes : ∀ (x y : ℝ), x = 2*y ∨ x = -2*y

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := 
  sorry

/-- Theorem: The eccentricity of a hyperbola with asymptotes x ± 2y = 0 is either √5 or √5/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  eccentricity h = Real.sqrt 5 ∨ eccentricity h = (Real.sqrt 5) / 2 := by
  sorry

end hyperbola_eccentricity_l2369_236973


namespace mark_and_carolyn_money_sum_l2369_236971

theorem mark_and_carolyn_money_sum : 
  let mark_money : ℚ := 3 / 4
  let carolyn_money : ℚ := 3 / 10
  mark_money + carolyn_money = 21 / 20 := by
sorry

end mark_and_carolyn_money_sum_l2369_236971


namespace chord_length_is_16_l2369_236924

/-- Represents a line in polar form -/
structure PolarLine where
  equation : ℝ → ℝ → ℝ

/-- Represents a circle in parametric form -/
structure ParametricCircle where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Calculates the length of a chord on a circle cut by a line -/
noncomputable def chordLength (l : PolarLine) (c : ParametricCircle) : ℝ :=
  sorry

/-- The main theorem stating that the chord length is 16 -/
theorem chord_length_is_16 :
  let l : PolarLine := { equation := λ ρ θ => ρ * Real.sin (θ - Real.pi / 3) - 6 }
  let c : ParametricCircle := { x := λ θ => 10 * Real.cos θ, y := λ θ => 10 * Real.sin θ }
  chordLength l c = 16 := by
  sorry

end chord_length_is_16_l2369_236924


namespace sqrt_equation_solution_l2369_236966

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 - 5 * x) = 8 → x = -12 := by sorry

end sqrt_equation_solution_l2369_236966


namespace initial_overs_is_ten_l2369_236951

/-- Represents a cricket game scenario -/
structure CricketGame where
  targetScore : ℕ
  initialRunRate : ℚ
  remainingOvers : ℕ
  requiredRunRate : ℚ

/-- Calculates the number of overs played initially -/
def initialOvers (game : CricketGame) : ℚ :=
  (game.targetScore - game.requiredRunRate * game.remainingOvers) / (game.initialRunRate - game.requiredRunRate)

/-- Theorem stating that the number of overs played initially is 10 -/
theorem initial_overs_is_ten (game : CricketGame) 
  (h1 : game.targetScore = 282)
  (h2 : game.initialRunRate = 4.8)
  (h3 : game.remainingOvers = 40)
  (h4 : game.requiredRunRate = 5.85) :
  initialOvers game = 10 := by
  sorry

end initial_overs_is_ten_l2369_236951


namespace memory_efficiency_improvement_l2369_236919

theorem memory_efficiency_improvement (x : ℝ) (h : x > 0) :
  (100 / x) - (100 / (1.2 * x)) = 5 / 12 ↔
  (100 / x) - (100 / ((1 + 0.2) * x)) = 5 / 12 :=
by sorry

end memory_efficiency_improvement_l2369_236919


namespace corn_height_after_three_weeks_l2369_236930

/-- The height of corn plants after three weeks of growth -/
def cornHeight (firstWeekGrowth : ℕ) : ℕ :=
  let secondWeekGrowth := 2 * firstWeekGrowth
  let thirdWeekGrowth := 4 * secondWeekGrowth
  firstWeekGrowth + secondWeekGrowth + thirdWeekGrowth

/-- Theorem stating that the corn plants grow to 22 inches after three weeks -/
theorem corn_height_after_three_weeks :
  cornHeight 2 = 22 := by
  sorry


end corn_height_after_three_weeks_l2369_236930


namespace circle_condition_m_set_l2369_236901

/-- A set in R² represents a circle if it can be expressed as 
    {(x, y) | (x - h)² + (y - k)² = r²} for some h, k, and r > 0 -/
def IsCircle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ h k r, r > 0 ∧ S = {p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - k)^2 = r^2}

/-- The set of points (x, y) satisfying the given equation -/
def S (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*m*p.1 - 2*m*p.2 + 2*m^2 + m - 1 = 0}

theorem circle_condition (m : ℝ) : IsCircle (S m) → m < 1 := by
  sorry

theorem m_set : {m : ℝ | IsCircle (S m)} = {m : ℝ | m < 1} := by
  sorry

end circle_condition_m_set_l2369_236901


namespace sum_of_roots_of_sum_l2369_236955

/-- Given two quadratic polynomials with the same leading coefficient, 
    if the sum of their four roots is p and their sum has two roots, 
    then the sum of the roots of their sum is p/2 -/
theorem sum_of_roots_of_sum (f g : ℝ → ℝ) (a b₁ b₂ c₁ c₂ p : ℝ) :
  (∀ x, f x = a * x^2 + b₁ * x + c₁) →
  (∀ x, g x = a * x^2 + b₂ * x + c₂) →
  (-b₁ / a - b₂ / a = p) →
  (∃ x y, ∀ z, f z + g z = 2 * a * (z - x) * (z - y)) →
  -(b₁ + b₂) / (2 * a) = p / 2 := by
sorry

end sum_of_roots_of_sum_l2369_236955


namespace rectangle_tiling_l2369_236932

theorem rectangle_tiling (n : ℕ) : 
  (∃ (a b : ℕ), 3 * a + 2 * b + 3 * n = 63 ∧ b + n = 20) ↔ 
  n ∈ ({2, 5, 8, 11, 14, 17, 20} : Set ℕ) := by
  sorry

end rectangle_tiling_l2369_236932


namespace no_natural_power_pair_l2369_236963

theorem no_natural_power_pair : ¬∃ (x y : ℕ), 
  (∃ (k : ℕ), x^2 + x + 1 = y^k) ∧ 
  (∃ (m : ℕ), y^2 + y + 1 = x^m) := by
  sorry

end no_natural_power_pair_l2369_236963


namespace f_composition_of_three_l2369_236922

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem f_composition_of_three : f (f (f (f 3))) = 24 := by
  sorry

end f_composition_of_three_l2369_236922


namespace olivias_albums_l2369_236904

def number_of_albums (pictures_from_phone : ℕ) (pictures_from_camera : ℕ) (pictures_per_album : ℕ) : ℕ :=
  (pictures_from_phone + pictures_from_camera) / pictures_per_album

theorem olivias_albums :
  let pictures_from_phone : ℕ := 5
  let pictures_from_camera : ℕ := 35
  let pictures_per_album : ℕ := 5
  number_of_albums pictures_from_phone pictures_from_camera pictures_per_album = 8 := by
  sorry

end olivias_albums_l2369_236904


namespace expression_is_square_difference_l2369_236936

/-- The square difference formula -/
def square_difference (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- The expression to be checked -/
def expression (x y : ℝ) : ℝ := (-x + y) * (x + y)

/-- Theorem stating that the expression can be calculated using the square difference formula -/
theorem expression_is_square_difference (x y : ℝ) :
  ∃ a b : ℝ, expression x y = -square_difference a b :=
sorry

end expression_is_square_difference_l2369_236936


namespace field_area_is_fifty_l2369_236959

/-- Represents a rectangular field with specific fencing conditions -/
structure FencedField where
  length : ℝ
  width : ℝ
  uncovered_side : ℝ
  fencing_length : ℝ

/-- The area of the field is 50 square feet given the specified conditions -/
theorem field_area_is_fifty (field : FencedField)
  (h1 : field.uncovered_side = 20)
  (h2 : field.fencing_length = 25)
  (h3 : field.length = field.uncovered_side)
  (h4 : field.fencing_length = field.length + 2 * field.width) :
  field.length * field.width = 50 := by
  sorry

#check field_area_is_fifty

end field_area_is_fifty_l2369_236959


namespace four_times_hash_58_l2369_236969

-- Define the function #
def hash (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem four_times_hash_58 : hash (hash (hash (hash 58))) = 11.8688 := by
  sorry

end four_times_hash_58_l2369_236969


namespace total_rabbits_l2369_236996

theorem total_rabbits (initial additional : ℕ) : 
  initial + additional = (initial + additional) :=
by sorry

end total_rabbits_l2369_236996


namespace smallest_non_representable_l2369_236994

def representable (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_non_representable : ∀ k < 11, representable k ∧ ¬ representable 11 :=
sorry

end smallest_non_representable_l2369_236994


namespace f_properties_l2369_236909

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x

theorem f_properties :
  (f (25 * Real.pi / 6) = 0) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 - Real.sqrt 3 / 2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1 - Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -Real.sqrt 3) :=
by sorry

end f_properties_l2369_236909


namespace min_sum_with_prime_hcfs_l2369_236974

/-- Given three positive integers with pairwise HCFs being distinct primes, 
    their sum is at least 31 -/
theorem min_sum_with_prime_hcfs (Q R S : ℕ+) 
  (hQR : ∃ (p : ℕ), Nat.Prime p ∧ Nat.gcd Q.val R.val = p)
  (hQS : ∃ (q : ℕ), Nat.Prime q ∧ Nat.gcd Q.val S.val = q)
  (hRS : ∃ (r : ℕ), Nat.Prime r ∧ Nat.gcd R.val S.val = r)
  (h_distinct : ∀ (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
    Nat.gcd Q.val R.val = p ∧ Nat.gcd Q.val S.val = q ∧ Nat.gcd R.val S.val = r →
    p ≠ q ∧ q ≠ r ∧ p ≠ r) :
  Q.val + R.val + S.val ≥ 31 := by
  sorry

#check min_sum_with_prime_hcfs

end min_sum_with_prime_hcfs_l2369_236974


namespace degree_three_polynomial_l2369_236920

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4
def g (x : ℝ) : ℝ := 3 - 2*x - 6*x^3 + 7*x^4

-- Define the combined polynomial h
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

-- Theorem statement
theorem degree_three_polynomial (c : ℝ) :
  c = -5/7 → (∀ x, h c x = 1 + (-12 - 2*c)*x + (3*x^2) + (-4 - 6*c)*x^3) :=
by sorry

end degree_three_polynomial_l2369_236920


namespace continued_fraction_evaluation_l2369_236953

theorem continued_fraction_evaluation :
  1 + 2 / (3 + 4 / (5 + 6 / 7)) = 233 / 151 := by
  sorry

end continued_fraction_evaluation_l2369_236953


namespace expression_value_at_nine_l2369_236926

theorem expression_value_at_nine :
  let x : ℝ := 9
  (x^6 - 27*x^3 + 729) / (x^3 - 27) = 702 := by sorry

end expression_value_at_nine_l2369_236926


namespace sin_120_degrees_l2369_236980

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_120_degrees_l2369_236980


namespace square_minus_one_roots_l2369_236944

theorem square_minus_one_roots (x : ℝ) : x^2 - 1 = 0 → x = -1 ∨ x = 1 := by
  sorry

end square_minus_one_roots_l2369_236944


namespace solution_exists_l2369_236958

-- Define the function f
def f (x : ℝ) : ℝ := (40 * x + (40 * x + 24) ^ (1/4)) ^ (1/4)

-- State the theorem
theorem solution_exists : ∃ x : ℝ, f x = 24 := by
  use 8293.8
  sorry

end solution_exists_l2369_236958


namespace max_volume_box_l2369_236982

/-- The volume function of the box -/
def V (x : ℝ) : ℝ := (48 - 2*x)^2 * x

/-- The domain of x -/
def valid_x (x : ℝ) : Prop := 0 < x ∧ x < 24

theorem max_volume_box :
  ∃ (x_max : ℝ), valid_x x_max ∧
  (∀ x, valid_x x → V x ≤ V x_max) ∧
  x_max = 8 ∧ V x_max = 8192 := by
sorry

end max_volume_box_l2369_236982


namespace product_equality_l2369_236941

theorem product_equality (a b c : ℝ) 
  (h : ∀ x y z : ℝ, x * y * z = Real.sqrt ((x + 2) * (y + 3)) / (z + 1)) : 
  6 * 15 * 5 = 2 := by
  sorry

end product_equality_l2369_236941


namespace sqrt_3_times_sqrt_12_l2369_236916

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l2369_236916


namespace salt_mixture_price_l2369_236968

theorem salt_mixture_price (initial_salt_weight : ℝ) (initial_salt_price : ℝ) 
  (new_salt_weight : ℝ) (selling_price : ℝ) (profit_percentage : ℝ) :
  initial_salt_weight = 40 ∧ 
  initial_salt_price = 0.35 ∧
  new_salt_weight = 5 ∧
  selling_price = 0.48 ∧
  profit_percentage = 0.2 →
  ∃ (new_salt_price : ℝ),
    new_salt_price = 0.80 ∧
    (initial_salt_weight * initial_salt_price + new_salt_weight * new_salt_price) * 
      (1 + profit_percentage) = 
    (initial_salt_weight + new_salt_weight) * selling_price :=
by sorry

end salt_mixture_price_l2369_236968


namespace count_distinct_digits_eq_2688_l2369_236978

/-- The number of integers between 1000 and 9999 with four distinct digits, none of which is '5' -/
def count_distinct_digits : ℕ :=
  let first_digit := 8  -- 9 digits excluding 5
  let second_digit := 8 -- 9 digits excluding 5 and the first digit
  let third_digit := 7  -- 8 digits excluding 5 and the first two digits
  let fourth_digit := 6 -- 7 digits excluding 5 and the first three digits
  first_digit * second_digit * third_digit * fourth_digit

theorem count_distinct_digits_eq_2688 : count_distinct_digits = 2688 := by
  sorry

end count_distinct_digits_eq_2688_l2369_236978


namespace quadratic_root_implies_a_value_l2369_236970

theorem quadratic_root_implies_a_value (x a : ℝ) : 
  x = 1 → x^2 + a*x - 2 = 0 → a = 1 := by sorry

end quadratic_root_implies_a_value_l2369_236970


namespace max_value_ln_x_over_x_l2369_236917

/-- The function f(x) = ln(x) / x attains its maximum value of 1/e for x > 0 -/
theorem max_value_ln_x_over_x :
  ∃ (c : ℝ), c > 0 ∧ 
    (∀ x > 0, (Real.log x) / x ≤ (Real.log c) / c) ∧
    (Real.log c) / c = 1 / Real.exp 1 := by
  sorry

end max_value_ln_x_over_x_l2369_236917


namespace odd_function_implies_m_equals_one_inequality_implies_a_range_l2369_236999

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := exp x - m / exp x

theorem odd_function_implies_m_equals_one (m : ℝ) :
  (∀ x, f m x = -f m (-x)) → m = 1 := by sorry

theorem inequality_implies_a_range (m : ℝ) :
  m = 1 →
  (∀ a : ℝ, f m (a - 1) + f m (2 * a^2) ≤ 0 → -1 ≤ a ∧ a ≤ 1/2) := by sorry

end odd_function_implies_m_equals_one_inequality_implies_a_range_l2369_236999


namespace trajectory_of_M_center_l2369_236995

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the properties of circle M
def M_externally_tangent_C₁ (M : ℝ × ℝ) : Prop :=
  ∃ r > 0, ∀ x y, C₁ x y → (x - M.1)^2 + (y - M.2)^2 = (r + 1)^2

def M_internally_tangent_C₂ (M : ℝ × ℝ) : Prop :=
  ∃ r > 0, ∀ x y, C₂ x y → (x - M.1)^2 + (y - M.2)^2 = (5 - r)^2

-- Theorem statement
theorem trajectory_of_M_center :
  ∀ M : ℝ × ℝ,
  M_externally_tangent_C₁ M →
  M_internally_tangent_C₂ M →
  M.1^2 / 9 + M.2^2 / 8 = 1 :=
sorry

end trajectory_of_M_center_l2369_236995


namespace amare_dresses_l2369_236988

/-- The number of dresses Amare needs to make -/
def number_of_dresses : ℕ := 4

/-- The amount of fabric required for one dress in yards -/
def fabric_per_dress : ℚ := 5.5

/-- The amount of fabric Amare has in feet -/
def fabric_amare_has : ℕ := 7

/-- The amount of fabric Amare still needs in feet -/
def fabric_amare_needs : ℕ := 59

/-- The number of feet in a yard -/
def feet_per_yard : ℕ := 3

theorem amare_dresses :
  number_of_dresses = 
    (((fabric_amare_has + fabric_amare_needs : ℚ) / feet_per_yard) / fabric_per_dress).floor :=
by sorry

end amare_dresses_l2369_236988


namespace quadratic_properties_l2369_236943

def f (x : ℝ) := 2 * x^2 - 4 * x + 3

theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (f 0 > 0) ∧ 
  (∀ x : ℝ, f x ≠ 0) ∧ 
  (∀ x y : ℝ, x < y → x < 1 → f x > f y) := by
  sorry

end quadratic_properties_l2369_236943


namespace distance_sum_property_l2369_236946

/-- Linear mapping between two line segments -/
structure LinearSegmentMap (AB A'B' : ℝ) where
  scale : ℝ
  map_points : ℝ → ℝ
  map_property : ∀ x, map_points x = scale * x

/-- Representation of a point on a line segment -/
structure SegmentPoint (total_length : ℝ) where
  position : ℝ
  valid_position : 0 ≤ position ∧ position ≤ total_length

theorem distance_sum_property 
  (AB A'B' : ℝ) 
  (h_AB_pos : AB > 0)
  (h_A'B'_pos : A'B' > 0)
  (h_linear_map : LinearSegmentMap AB A'B')
  (D : SegmentPoint AB)
  (D' : SegmentPoint A'B')
  (h_D_midpoint : D.position = AB / 2)
  (h_D'_third : D'.position = A'B' / 3)
  (P : SegmentPoint AB)
  (P' : SegmentPoint A'B')
  (h_P'_mapped : P'.position = h_linear_map.map_points P.position)
  (h_AB_length : AB = 3)
  (h_A'B'_length : A'B' = 6)
  (a : ℝ)
  (h_x_eq_a : |P.position - D.position| = a) :
  |P.position - D.position| + |P'.position - D'.position| = 3 * a :=
sorry

end distance_sum_property_l2369_236946


namespace friends_eating_pizza_l2369_236942

/-- The number of friends eating pizza with Ron -/
def num_friends : ℕ := 2

/-- The number of slices in the pizza -/
def total_slices : ℕ := 12

/-- The number of slices each person ate -/
def slices_per_person : ℕ := 4

/-- The total number of people eating, including Ron -/
def total_people : ℕ := total_slices / slices_per_person

theorem friends_eating_pizza : 
  num_friends = total_people - 1 :=
sorry

end friends_eating_pizza_l2369_236942


namespace sum_of_prime_factors_l2369_236938

def original_number : ℕ := 8679921
def divisor : ℕ := 330

theorem sum_of_prime_factors : 
  ∃ (n : ℕ), 
    n ≥ original_number ∧ 
    n % divisor = 0 ∧
    (∀ m : ℕ, m ≥ original_number ∧ m % divisor = 0 → m ≥ n) ∧
    (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id = 284) :=
sorry

end sum_of_prime_factors_l2369_236938


namespace remainder_theorem_f_of_one_eq_four_remainder_is_four_l2369_236972

-- Define the polynomial f(x) = x^15 + 3
def f (x : ℝ) : ℝ := x^15 + 3

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x - 1) * q x + f 1 := by
  sorry

-- Prove that f(1) = 4
theorem f_of_one_eq_four : f 1 = 4 := by
  sorry

-- Main theorem: The remainder when x^15 + 3 is divided by x-1 is 4
theorem remainder_is_four :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x - 1) * q x + 4 := by
  sorry

end remainder_theorem_f_of_one_eq_four_remainder_is_four_l2369_236972


namespace shearer_payment_l2369_236914

/-- Given the following conditions:
  - The number of sheep is 200
  - Each sheep produces 10 pounds of wool
  - The price of wool is $20 per pound
  - The profit is $38000
  Prove that the amount paid to the shearer is $2000 -/
theorem shearer_payment (num_sheep : ℕ) (wool_per_sheep : ℕ) (wool_price : ℕ) (profit : ℕ) :
  num_sheep = 200 →
  wool_per_sheep = 10 →
  wool_price = 20 →
  profit = 38000 →
  num_sheep * wool_per_sheep * wool_price - profit = 2000 := by
  sorry

end shearer_payment_l2369_236914


namespace third_podcast_length_l2369_236915

/-- Given a 6-hour drive and four podcasts, prove that the third podcast must be 105 minutes long to fill the entire drive time. -/
theorem third_podcast_length :
  let total_drive_time : ℕ := 6 * 60
  let first_podcast : ℕ := 45
  let second_podcast : ℕ := first_podcast * 2
  let fourth_podcast : ℕ := 60
  let next_podcast : ℕ := 60
  ∃ (third_podcast : ℕ),
    third_podcast = 105 ∧
    total_drive_time = first_podcast + second_podcast + third_podcast + fourth_podcast + next_podcast :=
by sorry

end third_podcast_length_l2369_236915


namespace function_transformation_l2369_236921

/-- Given a function f such that f(x-1) = x^2 + 4x - 5 for all x,
    prove that f(x) = x^2 + 6x for all x. -/
theorem function_transformation (f : ℝ → ℝ) 
    (h : ∀ x, f (x - 1) = x^2 + 4*x - 5) : 
    ∀ x, f x = x^2 + 6*x := by
  sorry

end function_transformation_l2369_236921


namespace car_downhill_speed_l2369_236985

/-- Proves that given specific conditions about a car's journey, the downhill speed is 60 km/hr -/
theorem car_downhill_speed 
  (uphill_speed : ℝ) 
  (uphill_distance : ℝ) 
  (downhill_distance : ℝ) 
  (average_speed : ℝ) 
  (h1 : uphill_speed = 30) 
  (h2 : uphill_distance = 100) 
  (h3 : downhill_distance = 50) 
  (h4 : average_speed = 36) : 
  ∃ downhill_speed : ℝ, 
    downhill_speed = 60 ∧ 
    average_speed = (uphill_distance + downhill_distance) / 
      (uphill_distance / uphill_speed + downhill_distance / downhill_speed) := by
  sorry

#check car_downhill_speed

end car_downhill_speed_l2369_236985


namespace percent_increase_proof_l2369_236950

def initial_cost : ℝ := 120000
def final_cost : ℝ := 192000

theorem percent_increase_proof :
  (final_cost - initial_cost) / initial_cost * 100 = 60 := by
  sorry

end percent_increase_proof_l2369_236950


namespace x_inequality_l2369_236934

theorem x_inequality (x : ℝ) : (x < 0 ∧ x < 1 / (4 * x)) ↔ (-1/2 < x ∧ x < 0) :=
sorry

end x_inequality_l2369_236934


namespace stratified_sampling_sample_size_l2369_236907

theorem stratified_sampling_sample_size 
  (total_population : ℕ) 
  (selection_probability : ℝ) 
  (sample_size : ℕ) :
  total_population = 1200 →
  selection_probability = 0.4 →
  (sample_size : ℝ) / total_population = selection_probability →
  sample_size = 480 := by
sorry

end stratified_sampling_sample_size_l2369_236907


namespace candy_bar_fundraiser_l2369_236962

theorem candy_bar_fundraiser (cost_per_bar : ℝ) (avg_sold_per_member : ℝ) (total_earnings : ℝ)
  (h1 : cost_per_bar = 0.5)
  (h2 : avg_sold_per_member = 8)
  (h3 : total_earnings = 80) :
  (total_earnings / cost_per_bar) / avg_sold_per_member = 20 := by
  sorry

end candy_bar_fundraiser_l2369_236962


namespace labourer_income_l2369_236983

/-- Proves that the monthly income of a labourer is 69 given the described conditions -/
theorem labourer_income (
  avg_expenditure_6months : ℝ)
  (reduced_monthly_expense : ℝ)
  (savings : ℝ)
  (h1 : avg_expenditure_6months = 70)
  (h2 : reduced_monthly_expense = 60)
  (h3 : savings = 30)
  : ∃ (monthly_income : ℝ),
    monthly_income = 69 ∧
    6 * monthly_income < 6 * avg_expenditure_6months ∧
    4 * monthly_income = 4 * reduced_monthly_expense + (6 * avg_expenditure_6months - 6 * monthly_income) + savings :=
by
  sorry

end labourer_income_l2369_236983
