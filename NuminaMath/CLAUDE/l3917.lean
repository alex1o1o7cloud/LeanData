import Mathlib

namespace NUMINAMATH_CALUDE_chocolate_squares_l3917_391764

theorem chocolate_squares (jenny_squares mike_squares : ℕ) : 
  jenny_squares = 65 → 
  jenny_squares = 3 * mike_squares + 5 → 
  mike_squares = 20 := by
sorry

end NUMINAMATH_CALUDE_chocolate_squares_l3917_391764


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3917_391757

theorem exponent_multiplication (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3917_391757


namespace NUMINAMATH_CALUDE_remaining_candies_formula_l3917_391710

/-- Represents the remaining number of candies after the first night -/
def remaining_candies (K S m n : ℕ) : ℚ :=
  (K + S : ℚ) * (1 - m / n)

/-- Theorem stating that the remaining number of candies after the first night
    is equal to (K + S) * (1 - m/n) -/
theorem remaining_candies_formula (K S m n : ℕ) (h : n ≠ 0) :
  remaining_candies K S m n = (K + S : ℚ) * (1 - m / n) :=
by sorry

end NUMINAMATH_CALUDE_remaining_candies_formula_l3917_391710


namespace NUMINAMATH_CALUDE_max_time_proof_l3917_391725

/-- The number of digits in the lock combination -/
def num_digits : ℕ := 3

/-- The number of possible values for each digit (0 to 8, inclusive) -/
def digits_range : ℕ := 9

/-- The time in seconds required for each trial -/
def time_per_trial : ℕ := 3

/-- Calculates the maximum time in seconds required to try all combinations -/
def max_time_seconds : ℕ := digits_range ^ num_digits * time_per_trial

/-- Theorem: The maximum time required to try all combinations is 2187 seconds -/
theorem max_time_proof : max_time_seconds = 2187 := by
  sorry

end NUMINAMATH_CALUDE_max_time_proof_l3917_391725


namespace NUMINAMATH_CALUDE_work_completion_time_l3917_391747

/-- If a group can complete a task in 12 days, then twice that group can complete half the task in 3 days. -/
theorem work_completion_time 
  (people : ℕ) 
  (work : ℝ) 
  (h : people > 0) 
  (completion_time : ℝ := 12) 
  (h_completion : work = people * completion_time) : 
  work / 2 = (2 * people) * 3 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3917_391747


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3917_391784

/-- A quadratic function satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : ∀ x, a*(x-1)^2 + b*(x-1) = a*(3-x)^2 + b*(3-x)
  h3 : ∃! x, a*x^2 + b*x = 2*x

/-- The main theorem about the quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  (∀ x, f.a*x^2 + f.b*x = -x^2 + 2*x) ∧
  ∃ m n, m < n ∧
    (∀ x, f.a*x^2 + f.b*x ∈ Set.Icc m n ↔ x ∈ Set.Icc (-1) 0) ∧
    (∀ y, y ∈ Set.Icc (4*(-1)) (4*0) ↔ ∃ x, f.a*x^2 + f.b*x = y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3917_391784


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_twenty_thirds_l3917_391712

theorem sum_abcd_equals_negative_twenty_thirds 
  (y a b c d : ℚ) 
  (h1 : y = a + 2)
  (h2 : y = b + 4)
  (h3 : y = c + 6)
  (h4 : y = d + 8)
  (h5 : y = a + b + c + d + 10) :
  a + b + c + d = -20 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_twenty_thirds_l3917_391712


namespace NUMINAMATH_CALUDE_positive_solution_range_l3917_391702

theorem positive_solution_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ a / (x + 3) = 1 / 2 ∧ x = a) → a > 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_range_l3917_391702


namespace NUMINAMATH_CALUDE_vector_equation_result_l3917_391737

def a : ℝ × ℝ := (2, -3)
def b : ℝ × ℝ := (1, 2)
def c : ℝ × ℝ := (9, 4)

theorem vector_equation_result (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : c = (m * a.1 + n * b.1, m * a.2 + n * b.2)) : 
  1/m + 1/n = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_result_l3917_391737


namespace NUMINAMATH_CALUDE_arthur_reading_challenge_l3917_391760

/-- Arthur's summer reading challenge -/
theorem arthur_reading_challenge 
  (total_goal : ℕ) 
  (book1_pages : ℕ) 
  (book1_read_percent : ℚ) 
  (book2_pages : ℕ) 
  (book2_read_fraction : ℚ) 
  (h1 : total_goal = 800)
  (h2 : book1_pages = 500)
  (h3 : book1_read_percent = 80 / 100)
  (h4 : book2_pages = 1000)
  (h5 : book2_read_fraction = 1 / 5)
  : ℕ := by
  sorry

#check arthur_reading_challenge

end NUMINAMATH_CALUDE_arthur_reading_challenge_l3917_391760


namespace NUMINAMATH_CALUDE_first_month_bill_l3917_391743

/-- Represents the telephone bill structure -/
structure TelephoneBill where
  callCharge : ℝ
  internetCharge : ℝ
  totalCharge : ℝ
  totalCharge_eq : totalCharge = callCharge + internetCharge

/-- The telephone bill problem -/
theorem first_month_bill (
  firstMonth secondMonth : TelephoneBill
) (h1 : firstMonth.totalCharge = 46)
  (h2 : secondMonth.totalCharge = 76)
  (h3 : secondMonth.callCharge = 2 * firstMonth.callCharge)
  (h4 : firstMonth.internetCharge = secondMonth.internetCharge) :
  firstMonth.totalCharge = 46 := by
sorry

end NUMINAMATH_CALUDE_first_month_bill_l3917_391743


namespace NUMINAMATH_CALUDE_min_value_3a_plus_b_min_value_exists_min_value_equality_l3917_391723

theorem min_value_3a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = a*b) :
  ∀ x y, x > 0 → y > 0 → x + 2*y = x*y → 3*a + b ≤ 3*x + y :=
by sorry

theorem min_value_exists (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = a*b) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = x*y ∧ 3*x + y = 7 + 2*Real.sqrt 6 :=
by sorry

theorem min_value_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = a*b) :
  3*a + b ≥ 7 + 2*Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3a_plus_b_min_value_exists_min_value_equality_l3917_391723


namespace NUMINAMATH_CALUDE_joan_balloons_l3917_391717

/-- The number of blue balloons Joan has after gaining more -/
def total_balloons (initial : ℕ) (gained : ℕ) : ℕ :=
  initial + gained

theorem joan_balloons :
  total_balloons 9 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l3917_391717


namespace NUMINAMATH_CALUDE_inverse_variation_solution_l3917_391744

/-- Represents the inverse variation relationship between 5y and x^3 -/
def inverse_variation (x y : ℝ) : Prop :=
  ∃ k : ℝ, 5 * y = k / (x^3)

theorem inverse_variation_solution :
  ∀ f : ℝ → ℝ,
  (∀ x, inverse_variation x (f x)) →  -- Condition: 5y varies inversely as the cube of x
  f 2 = 4 →                           -- Condition: When y = 4, x = 2
  f 4 = 1/2                           -- Conclusion: y = 1/2 when x = 4
:= by sorry

end NUMINAMATH_CALUDE_inverse_variation_solution_l3917_391744


namespace NUMINAMATH_CALUDE_map_distance_l3917_391706

/-- Given a map scale where 0.6 cm represents 6.6 km, and an actual distance of 885.5 km
    between two points, the distance between these points on the map is 80.5 cm. -/
theorem map_distance (scale_map : Real) (scale_actual : Real) (actual_distance : Real) :
  scale_map = 0.6 ∧ scale_actual = 6.6 ∧ actual_distance = 885.5 →
  (actual_distance / (scale_actual / scale_map)) = 80.5 := by
  sorry

#check map_distance

end NUMINAMATH_CALUDE_map_distance_l3917_391706


namespace NUMINAMATH_CALUDE_blackboard_numbers_l3917_391790

/-- Represents the state of the blackboard after n steps -/
def BlackboardState (n : ℕ) : Type := List ℕ

/-- The rule for updating the blackboard -/
def updateBlackboard (state : BlackboardState n) : BlackboardState (n + 1) :=
  sorry

/-- The number of numbers on the blackboard after n steps -/
def f (n : ℕ) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem blackboard_numbers (n : ℕ) : 
  f n = (1 / 2 : ℚ) * Nat.choose (2 * n + 2) (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l3917_391790


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l3917_391700

/-- Represents a repeating decimal with a single digit repeating part -/
def SingleDigitRepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with a two-digit repeating part -/
def TwoDigitRepeatingDecimal (n : ℕ) : ℚ := n / 99

theorem repeating_decimal_difference (h1 : 0 < 99) (h2 : 0 < 9) :
  99 * (TwoDigitRepeatingDecimal 49 - SingleDigitRepeatingDecimal 4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l3917_391700


namespace NUMINAMATH_CALUDE_train_length_problem_l3917_391789

/-- The length of Train 2 given the following conditions:
    - Train 1 length is 290 meters
    - Train 1 speed is 120 km/h
    - Train 2 speed is 80 km/h
    - Trains are running in opposite directions
    - Time to cross each other is 9 seconds
-/
theorem train_length_problem (train1_length : ℝ) (train1_speed : ℝ) (train2_speed : ℝ) (crossing_time : ℝ) :
  train1_length = 290 →
  train1_speed = 120 →
  train2_speed = 80 →
  crossing_time = 9 →
  ∃ train2_length : ℝ,
    (train1_length + train2_length) / crossing_time = (train1_speed + train2_speed) * (1000 / 3600) ∧
    abs (train2_length - 209.95) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l3917_391789


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l3917_391740

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 4*y + 12 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-2, 3)

-- Define a line in slope-intercept form
def line (k b : ℝ) (x y : ℝ) : Prop :=
  y = k * x + b

-- Define the property of being tangent to the circle
def is_tangent_to_circle (k b : ℝ) : Prop :=
  ∃ (x y : ℝ), line k b x y ∧ circle_C x y

-- Define the property of passing through the reflection of A
def passes_through_reflection (k b : ℝ) : Prop :=
  line k b (-2) (-3)

-- State the theorem
theorem reflected_ray_equation :
  ∃ (k b : ℝ), 
    is_tangent_to_circle k b ∧
    passes_through_reflection k b ∧
    ((k = 4/3 ∧ b = -1/3) ∨ (k = 3/4 ∧ b = -3/2)) :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l3917_391740


namespace NUMINAMATH_CALUDE_similar_not_congruent_l3917_391797

/-- Two triangles with sides a1, b1, c1 and a2, b2, c2 respectively -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of similar triangles -/
def similar (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

/-- Definition of congruent triangles -/
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

/-- Theorem: There exist two triangles with 3 equal angles (similar) 
    and 2 equal sides that are not congruent -/
theorem similar_not_congruent : ∃ (t1 t2 : Triangle), 
  similar t1 t2 ∧ t1.c = t2.c ∧ t1.a = t2.a ∧ ¬congruent t1 t2 := by
  sorry


end NUMINAMATH_CALUDE_similar_not_congruent_l3917_391797


namespace NUMINAMATH_CALUDE_stating_num_elective_ways_l3917_391761

/-- Represents the number of elective courses -/
def num_courses : ℕ := 4

/-- Represents the number of academic years -/
def num_years : ℕ := 3

/-- Represents the maximum number of courses a student can take per year -/
def max_courses_per_year : ℕ := 3

/-- 
Calculates the number of ways to distribute distinct courses over years
-/
def distribute_courses : ℕ := sorry

/-- 
Theorem stating that the number of ways to distribute the courses is 78
-/
theorem num_elective_ways : distribute_courses = 78 := by sorry

end NUMINAMATH_CALUDE_stating_num_elective_ways_l3917_391761


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3917_391765

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 1 = 0

-- Define the given line
def givenLine (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- Define the result circle
def resultCircle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

-- Theorem statement
theorem circle_equation_proof :
  ∀ x y : ℝ, 
    (∃ c : ℝ, c > 0 ∧ 
      (∀ x' y' : ℝ, givenCircle x' y' ↔ (x' - 1)^2 + (y' + 2)^2 = c)) ∧ 
    (∃ x₀ y₀ : ℝ, givenLine x₀ y₀ ∧ resultCircle x₀ y₀) ∧
    (∀ x' y' : ℝ, givenLine x' y' → ¬(resultCircle x' y' ∧ ¬(x' = x₀ ∧ y' = y₀))) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3917_391765


namespace NUMINAMATH_CALUDE_video_game_lives_calculation_l3917_391705

/-- Calculate the total number of lives for remaining players in a video game --/
theorem video_game_lives_calculation (initial_players : ℕ) (initial_lives : ℕ) 
  (quit_players : ℕ) (powerup_players : ℕ) (penalty_players : ℕ) 
  (powerup_lives : ℕ) (penalty_lives : ℕ) : 
  initial_players = 15 →
  initial_lives = 10 →
  quit_players = 5 →
  powerup_players = 4 →
  penalty_players = 6 →
  powerup_lives = 3 →
  penalty_lives = 2 →
  (initial_players - quit_players) * initial_lives + 
    powerup_players * powerup_lives - penalty_players * penalty_lives = 100 := by
  sorry


end NUMINAMATH_CALUDE_video_game_lives_calculation_l3917_391705


namespace NUMINAMATH_CALUDE_triangle_vector_relation_l3917_391714

-- Define the triangle ABC and vectors a and b
variable (A B C : EuclideanSpace ℝ (Fin 2))
variable (a b : EuclideanSpace ℝ (Fin 2))

-- Define points P and Q
variable (P : EuclideanSpace ℝ (Fin 2))
variable (Q : EuclideanSpace ℝ (Fin 2))

-- State the theorem
theorem triangle_vector_relation
  (h1 : B - A = a)
  (h2 : C - A = b)
  (h3 : P - A = (1/3) • (B - A))
  (h4 : Q - B = (1/3) • (C - B)) :
  Q - P = (1/3) • a + (1/3) • b := by sorry

end NUMINAMATH_CALUDE_triangle_vector_relation_l3917_391714


namespace NUMINAMATH_CALUDE_relationship_order_l3917_391767

theorem relationship_order (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a := by
  sorry

end NUMINAMATH_CALUDE_relationship_order_l3917_391767


namespace NUMINAMATH_CALUDE_erik_money_left_l3917_391724

/-- The amount of money Erik started with -/
def initial_money : ℕ := 86

/-- The number of loaves of bread Erik bought -/
def bread_quantity : ℕ := 3

/-- The cost of each loaf of bread -/
def bread_cost : ℕ := 3

/-- The number of cartons of orange juice Erik bought -/
def juice_quantity : ℕ := 3

/-- The cost of each carton of orange juice -/
def juice_cost : ℕ := 6

/-- The theorem stating how much money Erik has left -/
theorem erik_money_left : 
  initial_money - (bread_quantity * bread_cost + juice_quantity * juice_cost) = 59 := by
  sorry

end NUMINAMATH_CALUDE_erik_money_left_l3917_391724


namespace NUMINAMATH_CALUDE_range_of_a_l3917_391795

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ {x : ℝ | -4*x + 4*a < 0} → x ≠ 2) → 
  a ∈ {x : ℝ | x ≥ 2} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3917_391795


namespace NUMINAMATH_CALUDE_difference_of_squares_51_50_l3917_391776

-- Define the function for squaring a number
def square (n : ℕ) : ℕ := n * n

-- State the theorem
theorem difference_of_squares_51_50 : square 51 - square 50 = 101 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_51_50_l3917_391776


namespace NUMINAMATH_CALUDE_unique_sums_count_l3917_391755

def bag_C : Finset ℕ := {1, 3, 7, 9}
def bag_D : Finset ℕ := {4, 6, 8}

theorem unique_sums_count : 
  Finset.card ((bag_C.product bag_D).image (fun p => p.1 + p.2)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l3917_391755


namespace NUMINAMATH_CALUDE_simplify_expression_l3917_391788

theorem simplify_expression : 2^3 * 2^2 * 3^3 * 3^2 = 6^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3917_391788


namespace NUMINAMATH_CALUDE_sixtieth_number_is_sixteen_l3917_391715

/-- Defines the number of elements in each row of the sequence -/
def elementsInRow (n : ℕ) : ℕ := 2 * n

/-- Defines the value of elements in each row of the sequence -/
def valueInRow (n : ℕ) : ℕ := 2 * n

/-- Calculates the cumulative sum of elements up to and including row n -/
def cumulativeSum (n : ℕ) : ℕ :=
  (List.range n).map elementsInRow |>.sum

/-- Finds the row number for a given position in the sequence -/
def findRow (position : ℕ) : ℕ :=
  (List.range position).find? (fun n => cumulativeSum (n + 1) ≥ position)
    |>.getD 0

/-- The main theorem stating that the 60th number in the sequence is 16 -/
theorem sixtieth_number_is_sixteen :
  valueInRow (findRow 60 + 1) = 16 := by
  sorry

#eval valueInRow (findRow 60 + 1)

end NUMINAMATH_CALUDE_sixtieth_number_is_sixteen_l3917_391715


namespace NUMINAMATH_CALUDE_arrange_programs_count_l3917_391711

/-- The number of ways to arrange n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 5 programs with 2 consecutive -/
def arrange_programs : ℕ :=
  2 * permutations 4

theorem arrange_programs_count : arrange_programs = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrange_programs_count_l3917_391711


namespace NUMINAMATH_CALUDE_load_truck_time_proof_l3917_391722

/-- The time taken for three workers to load one truck simultaneously -/
def time_to_load_truck (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating the time taken for the given workers to load one truck -/
theorem load_truck_time_proof :
  let rate1 : ℚ := 1 / 5
  let rate2 : ℚ := 1 / 4
  let rate3 : ℚ := 1 / 6
  time_to_load_truck rate1 rate2 rate3 = 60 / 37 := by
  sorry

#eval time_to_load_truck (1/5) (1/4) (1/6)

end NUMINAMATH_CALUDE_load_truck_time_proof_l3917_391722


namespace NUMINAMATH_CALUDE_hexagon_area_in_circle_l3917_391758

/-- The area of a regular hexagon inscribed in a circle -/
theorem hexagon_area_in_circle (circle_area : ℝ) (hexagon_area : ℝ) : 
  circle_area = 400 * Real.pi →
  hexagon_area = 600 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_area_in_circle_l3917_391758


namespace NUMINAMATH_CALUDE_min_plates_min_plates_achieved_l3917_391771

theorem min_plates (m n : ℕ) : 
  2 * m + n ≥ 15 ∧ 
  m + 2 * n ≥ 18 ∧ 
  m + 3 * n ≥ 27 →
  m + n ≥ 12 :=
by
  sorry

theorem min_plates_achieved : 
  ∃ (m n : ℕ), 
    2 * m + n ≥ 15 ∧ 
    m + 2 * n ≥ 18 ∧ 
    m + 3 * n ≥ 27 ∧
    m + n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_min_plates_min_plates_achieved_l3917_391771


namespace NUMINAMATH_CALUDE_royalties_sales_ratio_decrease_l3917_391768

/-- Calculate the percentage decrease in the ratio of royalties to sales --/
theorem royalties_sales_ratio_decrease (first_royalties second_royalties : ℝ)
  (first_sales second_sales : ℝ) :
  first_royalties = 6 →
  first_sales = 20 →
  second_royalties = 9 →
  second_sales = 108 →
  let first_ratio := first_royalties / first_sales
  let second_ratio := second_royalties / second_sales
  let percentage_decrease := (first_ratio - second_ratio) / first_ratio * 100
  abs (percentage_decrease - 72.23) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_royalties_sales_ratio_decrease_l3917_391768


namespace NUMINAMATH_CALUDE_root_sum_l3917_391759

-- Define the complex number 2i-3
def z : ℂ := -3 + 2*Complex.I

-- Define the quadratic equation
def quadratic (p q : ℝ) (x : ℂ) : ℂ := 2*x^2 + p*x + q

-- State the theorem
theorem root_sum (p q : ℝ) : 
  quadratic p q z = 0 → p + q = 38 := by sorry

end NUMINAMATH_CALUDE_root_sum_l3917_391759


namespace NUMINAMATH_CALUDE_line_point_z_coordinate_l3917_391794

/-- Given a line passing through two points in 3D space, find the z-coordinate of a point on the line with a specific x-coordinate. -/
theorem line_point_z_coordinate 
  (p1 : ℝ × ℝ × ℝ) 
  (p2 : ℝ × ℝ × ℝ) 
  (x : ℝ) : 
  p1 = (1, 3, 2) → 
  p2 = (4, 2, -1) → 
  x = 3 → 
  ∃ (y z : ℝ), (∃ (t : ℝ), 
    (1 + 3*t, 3 - t, 2 - 3*t) = (x, y, z) ∧ 
    z = 0) := by
  sorry

#check line_point_z_coordinate

end NUMINAMATH_CALUDE_line_point_z_coordinate_l3917_391794


namespace NUMINAMATH_CALUDE_right_triangle_side_lengths_l3917_391731

/-- Represents a right-angled triangle with side lengths a, b, and c (hypotenuse) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2

theorem right_triangle_side_lengths
  (t : RightTriangle)
  (leg_a : t.a = 10)
  (sum_squares : t.a^2 + t.b^2 + t.c^2 = 2050) :
  t.b = Real.sqrt 925 ∧ t.c = Real.sqrt 1025 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_side_lengths_l3917_391731


namespace NUMINAMATH_CALUDE_convention_handshakes_specific_l3917_391775

/-- Calculates the number of handshakes in a convention with representatives from multiple companies. -/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Proves that in a convention with 5 representatives from each of 5 companies, 
    where representatives only shake hands with people from other companies, 
    the total number of handshakes is 250. -/
theorem convention_handshakes_specific : convention_handshakes 5 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_specific_l3917_391775


namespace NUMINAMATH_CALUDE_exists_subset_with_unique_sum_representation_l3917_391701

-- Define the property for the subset X
def has_unique_sum_representation (X : Set ℤ) : Prop :=
  ∀ n : ℤ, ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n

-- Theorem statement
theorem exists_subset_with_unique_sum_representation :
  ∃ X : Set ℤ, has_unique_sum_representation X :=
sorry

end NUMINAMATH_CALUDE_exists_subset_with_unique_sum_representation_l3917_391701


namespace NUMINAMATH_CALUDE_no_2016_subsequence_l3917_391751

-- Define the sequence
def seq : ℕ → ℕ
  | 0 => 2
  | 1 => 0
  | 2 => 1
  | 3 => 7
  | 4 => 0
  | n + 5 => (seq n + seq (n + 1) + seq (n + 2) + seq (n + 3)) % 10

-- Define a function to check if a subsequence appears at a given position
def subsequenceAt (start : ℕ) (subseq : List ℕ) : Prop :=
  ∀ i, i < subseq.length → seq (start + i) = subseq.get ⟨i, by sorry⟩

-- Theorem statement
theorem no_2016_subsequence :
  ¬ ∃ start : ℕ, start ≥ 4 ∧ subsequenceAt start [2, 0, 1, 6] :=
by sorry

end NUMINAMATH_CALUDE_no_2016_subsequence_l3917_391751


namespace NUMINAMATH_CALUDE_complement_not_always_greater_l3917_391773

def complement (θ : ℝ) : ℝ := 90 - θ

theorem complement_not_always_greater : ∃ θ : ℝ, complement θ ≤ θ := by
  sorry

end NUMINAMATH_CALUDE_complement_not_always_greater_l3917_391773


namespace NUMINAMATH_CALUDE_largest_band_size_l3917_391774

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- The total number of members in a formation --/
def totalMembers (f : BandFormation) : ℕ := f.rows * f.membersPerRow

/-- The condition that the band has less than 150 members --/
def lessThan150 (f : BandFormation) : Prop := totalMembers f < 150

/-- The condition that there are 3 members left over in the original formation --/
def hasThreeLeftOver (f : BandFormation) (totalBandMembers : ℕ) : Prop :=
  totalMembers f + 3 = totalBandMembers

/-- The new formation with 2 more members per row and 3 fewer rows --/
def newFormation (f : BandFormation) : BandFormation :=
  { rows := f.rows - 3, membersPerRow := f.membersPerRow + 2 }

/-- The condition that the new formation fits all members exactly --/
def newFormationFitsExactly (f : BandFormation) (totalBandMembers : ℕ) : Prop :=
  totalMembers (newFormation f) = totalBandMembers

/-- The theorem stating that the largest possible number of band members is 108 --/
theorem largest_band_size :
  ∃ (f : BandFormation) (totalBandMembers : ℕ),
    lessThan150 f ∧
    hasThreeLeftOver f totalBandMembers ∧
    newFormationFitsExactly f totalBandMembers ∧
    totalBandMembers = 108 ∧
    (∀ (g : BandFormation) (m : ℕ),
      lessThan150 g →
      hasThreeLeftOver g m →
      newFormationFitsExactly g m →
      m ≤ 108) :=
  sorry


end NUMINAMATH_CALUDE_largest_band_size_l3917_391774


namespace NUMINAMATH_CALUDE_sum_transformed_sequence_formula_l3917_391742

/-- Given a sequence {aₙ} where the sum of its first n terms Sₙ satisfies 3Sₙ = 4^(n+1) - 4,
    this function computes the sum of the first n terms of the sequence {(3n-2)aₙ}. -/
def sumTransformedSequence (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, 3 * S n = 4^(n+1) - 4) : ℝ :=
  4 + (n - 1 : ℝ) * 4^(n+1)

/-- Theorem stating that the sum of the first n terms of {(3n-2)aₙ} is 4 + (n-1) * 4^(n+1),
    given that the sum of the first n terms of {aₙ} satisfies 3Sₙ = 4^(n+1) - 4. -/
theorem sum_transformed_sequence_formula (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, 3 * S n = 4^(n+1) - 4) :
  sumTransformedSequence n S h = 4 + (n - 1 : ℝ) * 4^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_sum_transformed_sequence_formula_l3917_391742


namespace NUMINAMATH_CALUDE_b_work_alone_days_l3917_391735

/-- The number of days A takes to finish the work alone -/
def A_days : ℝ := 5

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B takes to finish the remaining work after A leaves -/
def B_remaining_days : ℝ := 7

/-- The number of days B takes to finish the work alone -/
def B_days : ℝ := 15

/-- Theorem stating that given the conditions, B can finish the work alone in 15 days -/
theorem b_work_alone_days :
  (together_days * (1 / A_days + 1 / B_days) + B_remaining_days * (1 / B_days) = 1) :=
sorry

end NUMINAMATH_CALUDE_b_work_alone_days_l3917_391735


namespace NUMINAMATH_CALUDE_duck_profit_is_170_l3917_391778

/-- Calculates the profit from selling ducks given the specified conditions -/
def duck_profit : ℕ :=
  let first_group_weight := 10 * 3
  let second_group_weight := 10 * 4
  let third_group_weight := 10 * 5
  let total_weight := first_group_weight + second_group_weight + third_group_weight

  let first_group_cost := 10 * 9
  let second_group_cost := 10 * 10
  let third_group_cost := 10 * 12
  let total_cost := first_group_cost + second_group_cost + third_group_cost

  let selling_price_per_pound := 5
  let total_selling_price := total_weight * selling_price_per_pound
  let discount_rate := 20
  let discount_amount := total_selling_price * discount_rate / 100
  let final_selling_price := total_selling_price - discount_amount

  final_selling_price - total_cost

theorem duck_profit_is_170 : duck_profit = 170 := by
  sorry

end NUMINAMATH_CALUDE_duck_profit_is_170_l3917_391778


namespace NUMINAMATH_CALUDE_contest_score_difference_l3917_391749

def score_65_percent : ℝ := 0.15
def score_85_percent : ℝ := 0.20
def score_95_percent : ℝ := 0.40
def score_110_percent : ℝ := 1 - (score_65_percent + score_85_percent + score_95_percent)

def score_65 : ℝ := 65
def score_85 : ℝ := 85
def score_95 : ℝ := 95
def score_110 : ℝ := 110

def mean_score : ℝ := 
  score_65_percent * score_65 + 
  score_85_percent * score_85 + 
  score_95_percent * score_95 + 
  score_110_percent * score_110

def median_score : ℝ := score_95

theorem contest_score_difference : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.25 ∧ |median_score - mean_score - 3| < ε :=
sorry

end NUMINAMATH_CALUDE_contest_score_difference_l3917_391749


namespace NUMINAMATH_CALUDE_max_garden_area_l3917_391781

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.width * d.length

/-- Calculates the perimeter of a rectangular garden (excluding the side adjacent to the house) -/
def gardenPerimeter (d : GardenDimensions) : ℝ := d.length + 2 * d.width

/-- Theorem stating the maximum area of the garden under given constraints -/
theorem max_garden_area :
  ∃ (d : GardenDimensions),
    d.length = 2 * d.width ∧
    gardenPerimeter d = 480 ∧
    ∀ (d' : GardenDimensions),
      d'.length = 2 * d'.width →
      gardenPerimeter d' = 480 →
      gardenArea d' ≤ 28800 :=
by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l3917_391781


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3917_391753

theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) 
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ 3 < x ∧ x < 6) :
  ∀ x, c*x^2 + b*x + a < 0 ↔ x < 1/6 ∨ x > 1/3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3917_391753


namespace NUMINAMATH_CALUDE_angle_measure_when_sine_is_half_l3917_391719

/-- If ∠A is an acute angle in a triangle and sin A = 1/2, then ∠A = 30°. -/
theorem angle_measure_when_sine_is_half (A : Real) (h_acute : 0 < A ∧ A < π / 2) 
  (h_sin : Real.sin A = 1 / 2) : A = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_when_sine_is_half_l3917_391719


namespace NUMINAMATH_CALUDE_hidden_sum_is_55_l3917_391785

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The total number of dice -/
def num_dice : ℕ := 4

/-- The list of visible numbers on the dice -/
def visible_numbers : List ℕ := [1, 2, 3, 3, 4, 5, 5, 6]

/-- The sum of all numbers on all dice -/
def total_sum : ℕ := die_sum * num_dice

/-- The sum of visible numbers -/
def visible_sum : ℕ := visible_numbers.sum

theorem hidden_sum_is_55 : total_sum - visible_sum = 55 := by
  sorry

end NUMINAMATH_CALUDE_hidden_sum_is_55_l3917_391785


namespace NUMINAMATH_CALUDE_remainder_polynomial_l3917_391798

theorem remainder_polynomial (p : ℝ → ℝ) (h1 : p 2 = 4) (h2 : p 4 = 8) :
  ∃ (q r : ℝ → ℝ), (∀ x, p x = q x * (x - 2) * (x - 4) + r x) ∧
                    (∀ x, r x = 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_remainder_polynomial_l3917_391798


namespace NUMINAMATH_CALUDE_b_formula_l3917_391791

/-- Sequence a_n defined recursively --/
def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 / (a n + 1)

/-- Sequence b_n defined in terms of a_n --/
def b (n : ℕ) : ℚ := |((a n + 2) / (a n - 1))|

/-- The main theorem to be proved --/
theorem b_formula (n : ℕ) : b n = 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_b_formula_l3917_391791


namespace NUMINAMATH_CALUDE_digit_placement_ways_l3917_391782

/-- The number of corner boxes in a 3x3 grid -/
def num_corners : ℕ := 4

/-- The total number of boxes in a 3x3 grid -/
def total_boxes : ℕ := 9

/-- The number of digits to be placed -/
def num_digits : ℕ := 4

/-- The number of ways to place digits 1, 2, 3, and 4 in a 3x3 grid -/
def num_ways : ℕ := num_corners * (total_boxes - 1) * (total_boxes - 2) * (total_boxes - 3)

theorem digit_placement_ways :
  num_ways = 1344 :=
sorry

end NUMINAMATH_CALUDE_digit_placement_ways_l3917_391782


namespace NUMINAMATH_CALUDE_square_difference_pattern_l3917_391720

theorem square_difference_pattern (n : ℕ+) : (n + 2)^2 - n^2 = 4 * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l3917_391720


namespace NUMINAMATH_CALUDE_coin_bag_total_l3917_391730

theorem coin_bag_total (p : ℕ) : ∃ (p : ℕ), 
  (0.01 * p + 0.05 * (3 * p) + 0.10 * (4 * 3 * p) : ℚ) = 408 := by
  sorry

end NUMINAMATH_CALUDE_coin_bag_total_l3917_391730


namespace NUMINAMATH_CALUDE_complex_modulus_range_l3917_391726

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z.re = a) (h4 : z.im = 1) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l3917_391726


namespace NUMINAMATH_CALUDE_exists_multiple_in_ascending_sequence_l3917_391713

/-- Definition of an ascending sequence -/
def IsAscending (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1) ∧ a (2 * n) = 2 * a n

/-- Theorem: For any ascending sequence of positive integers and prime p > a₁,
    there exists a term in the sequence divisible by p -/
theorem exists_multiple_in_ascending_sequence
    (a : ℕ → ℕ)
    (h_ascending : IsAscending a)
    (h_positive : ∀ n, a n > 0)
    (p : ℕ)
    (h_prime : Nat.Prime p)
    (h_p_gt_a1 : p > a 1) :
    ∃ n, p ∣ a n := by
  sorry

end NUMINAMATH_CALUDE_exists_multiple_in_ascending_sequence_l3917_391713


namespace NUMINAMATH_CALUDE_apples_given_correct_l3917_391709

/-- The number of apples the farmer originally had -/
def original_apples : ℕ := 127

/-- The number of apples the farmer now has -/
def current_apples : ℕ := 39

/-- The number of apples given to the neighbor -/
def apples_given : ℕ := original_apples - current_apples

theorem apples_given_correct : apples_given = 88 := by sorry

end NUMINAMATH_CALUDE_apples_given_correct_l3917_391709


namespace NUMINAMATH_CALUDE_division_problem_l3917_391748

theorem division_problem (n : ℕ) : 
  (n / 20 = 9) ∧ (n % 20 = 1) → n = 181 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3917_391748


namespace NUMINAMATH_CALUDE_min_value_theorem_l3917_391718

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  ∃ (min : ℝ), min = 9/2 ∧ ∀ x, x = 1/(2*a) + 2/(b-1) → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3917_391718


namespace NUMINAMATH_CALUDE_concrete_density_l3917_391796

/-- Concrete density problem -/
theorem concrete_density (num_homes : ℕ) (length width height : ℝ) (cost_per_pound : ℝ) (total_cost : ℝ)
  (h1 : num_homes = 3)
  (h2 : length = 100)
  (h3 : width = 100)
  (h4 : height = 0.5)
  (h5 : cost_per_pound = 0.02)
  (h6 : total_cost = 45000) :
  (total_cost / cost_per_pound) / (num_homes * length * width * height) = 150 := by
  sorry

end NUMINAMATH_CALUDE_concrete_density_l3917_391796


namespace NUMINAMATH_CALUDE_equation_solutions_l3917_391739

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 169 ↔ x = 15 ∨ x = -11) ∧
  (∀ x : ℝ, 3*(x - 3)^3 - 24 = 0 ↔ x = 5) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3917_391739


namespace NUMINAMATH_CALUDE_angle_PDO_is_45_degrees_l3917_391754

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its side length -/
structure Square where
  sideLength : ℝ

/-- Represents the configuration described in the problem -/
structure Configuration where
  outerSquare : Square
  L : Point
  P : Point
  O : Point

/-- Predicate to check if a point is on the diagonal of a square -/
def isOnDiagonal (s : Square) (p : Point) : Prop :=
  p.x = p.y ∧ 0 ≤ p.x ∧ p.x ≤ s.sideLength

/-- Predicate to check if a point is on the side of a square -/
def isOnSide (s : Square) (p : Point) : Prop :=
  p.y = 0 ∧ 0 ≤ p.x ∧ p.x ≤ s.sideLength

/-- Calculate the angle between three points in degrees -/
def angleBetween (p1 p2 p3 : Point) : ℝ := sorry

/-- The main theorem -/
theorem angle_PDO_is_45_degrees (c : Configuration) 
  (h1 : isOnDiagonal c.outerSquare c.L)
  (h2 : isOnSide c.outerSquare c.P)
  (h3 : c.O.x = (c.L.x + c.outerSquare.sideLength) / 2)
  (h4 : c.O.y = (c.L.y + c.outerSquare.sideLength) / 2) :
  angleBetween c.P (Point.mk 0 c.outerSquare.sideLength) c.O = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_PDO_is_45_degrees_l3917_391754


namespace NUMINAMATH_CALUDE_point_coordinates_l3917_391792

-- Define a point in the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the second quadrant
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

-- Define the distance from a point to the x-axis
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

-- Define the distance from a point to the y-axis
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

-- Theorem statement
theorem point_coordinates (p : Point) 
  (h1 : secondQuadrant p)
  (h2 : distanceToXAxis p = 4)
  (h3 : distanceToYAxis p = 5) :
  p.x = -5 ∧ p.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3917_391792


namespace NUMINAMATH_CALUDE_factorial_not_ending_1976_zeros_l3917_391728

theorem factorial_not_ending_1976_zeros (n : ℕ) : ∃ k : ℕ, n! % (10^k) ≠ 1976 * (10^k) :=
sorry

end NUMINAMATH_CALUDE_factorial_not_ending_1976_zeros_l3917_391728


namespace NUMINAMATH_CALUDE_fraction_value_l3917_391745

theorem fraction_value (a b c : Int) (h1 : a = 5) (h2 : b = -3) (h3 : c = 2) :
  3 / (a + b + c : ℚ) = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l3917_391745


namespace NUMINAMATH_CALUDE_crop_ratio_l3917_391780

theorem crop_ratio (corn_rows : ℕ) (potato_rows : ℕ) (corn_per_row : ℕ) (potatoes_per_row : ℕ) (remaining_crops : ℕ) : 
  corn_rows = 10 →
  potato_rows = 5 →
  corn_per_row = 9 →
  potatoes_per_row = 30 →
  remaining_crops = 120 →
  (remaining_crops : ℚ) / ((corn_rows * corn_per_row + potato_rows * potatoes_per_row) : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_crop_ratio_l3917_391780


namespace NUMINAMATH_CALUDE_scientific_notation_of_280000_l3917_391756

theorem scientific_notation_of_280000 :
  280000 = 2.8 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_280000_l3917_391756


namespace NUMINAMATH_CALUDE_selection_with_at_least_one_boy_l3917_391736

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of boys in the group -/
def num_boys : ℕ := 8

/-- The number of girls in the group -/
def num_girls : ℕ := 6

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of people to be selected -/
def selection_size : ℕ := 3

theorem selection_with_at_least_one_boy :
  choose total_people selection_size - choose num_girls selection_size = 344 := by
  sorry

end NUMINAMATH_CALUDE_selection_with_at_least_one_boy_l3917_391736


namespace NUMINAMATH_CALUDE_platform_length_l3917_391741

/-- Given a train that passes a pole and a platform, prove the length of the platform. -/
theorem platform_length
  (train_length : ℝ)
  (pole_time : ℝ)
  (platform_time : ℝ)
  (h1 : train_length = 120)
  (h2 : pole_time = 11)
  (h3 : platform_time = 22) :
  (train_length * platform_time / pole_time) - train_length = 120 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3917_391741


namespace NUMINAMATH_CALUDE_clapping_groups_l3917_391707

def number_of_people : ℕ := 4043
def claps_per_hand : ℕ := 2021

def valid_groups (n k : ℕ) : ℕ := Nat.choose n k

def invalid_groups (n m : ℕ) : ℕ := n * Nat.choose m 2

theorem clapping_groups :
  valid_groups number_of_people 3 - invalid_groups number_of_people claps_per_hand =
  valid_groups number_of_people 3 - number_of_people * valid_groups claps_per_hand 2 :=
by sorry

end NUMINAMATH_CALUDE_clapping_groups_l3917_391707


namespace NUMINAMATH_CALUDE_magnitude_of_vector_AB_l3917_391793

theorem magnitude_of_vector_AB (OA OB : ℝ × ℝ) : 
  OA = (Real.cos (15 * π / 180), Real.sin (15 * π / 180)) →
  OB = (Real.cos (75 * π / 180), Real.sin (75 * π / 180)) →
  Real.sqrt ((OB.1 - OA.1)^2 + (OB.2 - OA.2)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_AB_l3917_391793


namespace NUMINAMATH_CALUDE_power_of_four_remainder_l3917_391746

theorem power_of_four_remainder (a : ℕ+) (p : ℕ) :
  p = 4^(a : ℕ) → p % 10 = 6 → ∃ k : ℕ, (a : ℕ) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_remainder_l3917_391746


namespace NUMINAMATH_CALUDE_circular_arrangement_size_l3917_391704

/-- A circular arrangement of people with the property that the 7th person is directly opposite the 18th person -/
structure CircularArrangement where
  n : ℕ  -- Total number of people
  seventh_opposite_eighteenth : n ≥ 18 ∧ (18 - 7) * 2 + 2 = n

/-- The theorem stating that in a circular arrangement where the 7th person is directly opposite the 18th person, the total number of people is 24 -/
theorem circular_arrangement_size (c : CircularArrangement) : c.n = 24 := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_size_l3917_391704


namespace NUMINAMATH_CALUDE_fundraiser_result_l3917_391783

/-- Represents the fundraiser scenario with students bringing brownies, cookies, and donuts. -/
structure Fundraiser where
  brownie_students : ℕ
  brownies_per_student : ℕ
  cookie_students : ℕ
  cookies_per_student : ℕ
  donut_students : ℕ
  donuts_per_student : ℕ
  price_per_item : ℚ

/-- Calculates the total amount of money raised in the fundraiser. -/
def total_money_raised (f : Fundraiser) : ℚ :=
  ((f.brownie_students * f.brownies_per_student +
    f.cookie_students * f.cookies_per_student +
    f.donut_students * f.donuts_per_student) : ℚ) * f.price_per_item

/-- Theorem stating that the fundraiser with given conditions raises $2040.00. -/
theorem fundraiser_result : 
  let f : Fundraiser := {
    brownie_students := 30,
    brownies_per_student := 12,
    cookie_students := 20,
    cookies_per_student := 24,
    donut_students := 15,
    donuts_per_student := 12,
    price_per_item := 2
  }
  total_money_raised f = 2040 := by
  sorry


end NUMINAMATH_CALUDE_fundraiser_result_l3917_391783


namespace NUMINAMATH_CALUDE_problem_statement_l3917_391786

theorem problem_statement (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x + 2*y + 3*z = 1) : 
  (∃ (m : ℝ), m = 6 + 2*Real.sqrt 2 + 2*Real.sqrt 3 + 2*Real.sqrt 6 ∧ 
   (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + 2*b + 3*c = 1 → 
    1/a + 1/b + 1/c ≥ m) ∧
   1/x + 1/y + 1/z = m) ∧ 
  x^2 + y^2 + z^2 ≥ 1/14 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3917_391786


namespace NUMINAMATH_CALUDE_pentagonal_cross_section_exists_regular_pentagonal_cross_section_impossible_l3917_391721

/-- A cube in 3D space -/
structure Cube :=
  (side : ℝ)
  (side_pos : side > 0)

/-- A plane in 3D space -/
structure Plane :=
  (normal : ℝ × ℝ × ℝ)
  (point : ℝ × ℝ × ℝ)

/-- A pentagon in 2D space -/
structure Pentagon :=
  (vertices : Finset (ℝ × ℝ))
  (is_pentagon : vertices.card = 5)

/-- A regular pentagon in 2D space -/
structure RegularPentagon extends Pentagon :=
  (is_regular : ∀ (v1 v2 : ℝ × ℝ), v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 → 
    ∃ (rotation : ℝ × ℝ → ℝ × ℝ), rotation v1 = v2 ∧ rotation '' vertices = vertices)

/-- The cross-section formed by intersecting a cube with a plane -/
def crossSection (c : Cube) (p : Plane) : Set (ℝ × ℝ) :=
  sorry

theorem pentagonal_cross_section_exists (c : Cube) : 
  ∃ (p : Plane), ∃ (pent : Pentagon), crossSection c p = ↑pent.vertices :=
sorry

theorem regular_pentagonal_cross_section_impossible (c : Cube) : 
  ¬∃ (p : Plane), ∃ (reg_pent : RegularPentagon), crossSection c p = ↑reg_pent.vertices :=
sorry

end NUMINAMATH_CALUDE_pentagonal_cross_section_exists_regular_pentagonal_cross_section_impossible_l3917_391721


namespace NUMINAMATH_CALUDE_line_x_coordinate_l3917_391777

/-- Given a line passing through (x, -4) and (10, 3) with x-intercept 4, 
    prove that x = -4 -/
theorem line_x_coordinate (x : ℝ) : 
  (∃ (m b : ℝ), (∀ (t : ℝ), -4 = m * x + b) ∧ 
                 (3 = m * 10 + b) ∧ 
                 (0 = m * 4 + b)) →
  x = -4 := by
  sorry

end NUMINAMATH_CALUDE_line_x_coordinate_l3917_391777


namespace NUMINAMATH_CALUDE_pareto_principle_implies_key_parts_l3917_391770

/-- Represents the Pareto Principle applied to business management -/
structure ParetoPrinciple where
  core_business : ℝ
  total_business : ℝ
  core_result : ℝ
  total_result : ℝ
  efficiency_improvement : Bool
  core_business_ratio : core_business / total_business = 0.2
  result_ratio : core_result / total_result = 0.8
  focus_on_core : efficiency_improvement = true

/-- The conclusion drawn from the Pareto Principle -/
def emphasis_on_key_parts : Prop := True

/-- Theorem stating that the Pareto Principle implies emphasis on key parts -/
theorem pareto_principle_implies_key_parts (p : ParetoPrinciple) : 
  emphasis_on_key_parts :=
sorry

end NUMINAMATH_CALUDE_pareto_principle_implies_key_parts_l3917_391770


namespace NUMINAMATH_CALUDE_smallest_square_factor_l3917_391799

theorem smallest_square_factor (n : ℕ) (hn : n = 4410) :
  (∃ (y : ℕ), y > 0 ∧ ∃ (k : ℕ), n * y = k^2) ∧
  (∀ (z : ℕ), z > 0 → (∃ (k : ℕ), n * z = k^2) → z ≥ 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_factor_l3917_391799


namespace NUMINAMATH_CALUDE_problem_solution_l3917_391716

theorem problem_solution (x y z : ℚ) (w : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 4) * z → 
  z = 80 → 
  x = 20 / 3 ∧ w = x + y + z ∧ w = 320 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3917_391716


namespace NUMINAMATH_CALUDE_frustum_sphere_equal_volume_l3917_391703

/-- Given a frustum of a cone with small radius 2 inches, large radius 3 inches,
    and height 5 inches, the radius of a sphere with the same volume is ∛(95/4) inches. -/
theorem frustum_sphere_equal_volume :
  let r₁ : ℝ := 2  -- small radius of frustum
  let r₂ : ℝ := 3  -- large radius of frustum
  let h : ℝ := 5   -- height of frustum
  let V_frustum := (1/3) * π * h * (r₁^2 + r₁*r₂ + r₂^2)
  let r_sphere := (95/4)^(1/3 : ℝ)
  let V_sphere := (4/3) * π * r_sphere^3
  V_frustum = V_sphere := by sorry

end NUMINAMATH_CALUDE_frustum_sphere_equal_volume_l3917_391703


namespace NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l3917_391787

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  a : ℝ  -- Length of edge AB
  b : ℝ  -- Length of edge CD
  d : ℝ  -- Distance between lines AB and CD
  w : ℝ  -- Angle between lines AB and CD
  k : ℝ  -- Ratio of distances from plane π to AB and CD
  h_a : a > 0
  h_b : b > 0
  h_d : d > 0
  h_w : 0 < w ∧ w < π
  h_k : k > 0

/-- Calculates the volume ratio of the two parts of the tetrahedron divided by plane π -/
noncomputable def volumeRatio (t : Tetrahedron) : ℝ :=
  (t.k^3 + 3*t.k^2) / (3*t.k + 1)

/-- Theorem stating the volume ratio of the two parts of the tetrahedron -/
theorem tetrahedron_volume_ratio (t : Tetrahedron) :
  ∃ (v1 v2 : ℝ), v1 > 0 ∧ v2 > 0 ∧ v1 / v2 = volumeRatio t :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l3917_391787


namespace NUMINAMATH_CALUDE_solution_l3917_391752

def problem (m n : ℕ) : Prop :=
  m + n = 80 ∧ 
  Nat.gcd m n = 6 ∧ 
  Nat.lcm m n = 210

theorem solution (m n : ℕ) (h : problem m n) : 
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 15.75 := by
  sorry

end NUMINAMATH_CALUDE_solution_l3917_391752


namespace NUMINAMATH_CALUDE_lcm_of_12_and_18_l3917_391738

theorem lcm_of_12_and_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_and_18_l3917_391738


namespace NUMINAMATH_CALUDE_line_minimum_sum_l3917_391732

theorem line_minimum_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : (1 : ℝ) / a + (1 : ℝ) / b = 1) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ (1 : ℝ) / x + (1 : ℝ) / y = 1 → a + b ≤ x + y) ∧ a + b = 4 :=
by sorry

end NUMINAMATH_CALUDE_line_minimum_sum_l3917_391732


namespace NUMINAMATH_CALUDE_club_members_remainder_l3917_391779

theorem club_members_remainder (N : ℕ) : 
  50 < N → N < 80 → 
  N % 5 = 0 → (N % 8 = 0 ∨ N % 7 = 0) → 
  N % 9 = 6 ∨ N % 9 = 7 := by
sorry

end NUMINAMATH_CALUDE_club_members_remainder_l3917_391779


namespace NUMINAMATH_CALUDE_d_properties_l3917_391750

/-- Given a nonnegative integer c, define sequences a_n and d_n -/
def a (c n : ℕ) : ℕ := n^2 + c

def d (c n : ℕ) : ℕ := Nat.gcd (a c n) (a c (n + 1))

/-- Theorem stating the properties of d_n for different values of c -/
theorem d_properties (c : ℕ) :
  (∀ n : ℕ, n ≥ 1 → c = 0 → d c n = 1) ∧
  (∀ n : ℕ, n ≥ 1 → c = 1 → d c n = 1 ∨ d c n = 5) ∧
  (∀ n : ℕ, n ≥ 1 → d c n ≤ 4 * c + 1) :=
sorry

end NUMINAMATH_CALUDE_d_properties_l3917_391750


namespace NUMINAMATH_CALUDE_fair_selection_condition_l3917_391762

/-- Fairness condition for ball selection --/
def is_fair_selection (b c : ℕ) : Prop :=
  (b - c)^2 = b + c

/-- The probability of selecting same color balls --/
def prob_same_color (b c : ℕ) : ℚ :=
  (b * (b - 1) + c * (c - 1)) / ((b + c) * (b + c - 1))

/-- The probability of selecting different color balls --/
def prob_diff_color (b c : ℕ) : ℚ :=
  (2 * b * c) / ((b + c) * (b + c - 1))

/-- Theorem stating the fairness condition for ball selection --/
theorem fair_selection_condition (b c : ℕ) :
  prob_same_color b c = prob_diff_color b c ↔ is_fair_selection b c :=
sorry

end NUMINAMATH_CALUDE_fair_selection_condition_l3917_391762


namespace NUMINAMATH_CALUDE_problem_solution_l3917_391769

-- Define the variables
variable (a b c : ℝ)

-- Define the conditions
def condition1 : Prop := (5 * a + 2) ^ (1/3 : ℝ) = 3
def condition2 : Prop := (3 * a + b - 1) ^ (1/2 : ℝ) = 4
def condition3 : Prop := c = ⌊Real.sqrt 13⌋

-- Define the theorem
theorem problem_solution (h1 : condition1 a) (h2 : condition2 a b) (h3 : condition3 c) :
  a = 5 ∧ b = 2 ∧ c = 3 ∧ (3 * a - b + c) ^ (1/2 : ℝ) = 4 ∨ (3 * a - b + c) ^ (1/2 : ℝ) = -4 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3917_391769


namespace NUMINAMATH_CALUDE_no_isosceles_triangle_l3917_391772

/-- The set of stick lengths -/
def stickLengths : Set ℝ :=
  {x : ℝ | ∃ n : ℕ, n < 100 ∧ x = (0.9 : ℝ) ^ n}

/-- Definition of an isosceles triangle formed by three sticks -/
def isIsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ a + b > c ∧ b + c > a ∧ a + c > b

/-- Theorem stating the impossibility of forming an isosceles triangle -/
theorem no_isosceles_triangle :
  ¬ ∃ a b c : ℝ, a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧
    isIsoscelesTriangle a b c :=
sorry

end NUMINAMATH_CALUDE_no_isosceles_triangle_l3917_391772


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l3917_391766

theorem smallest_four_digit_multiple_of_17 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → n ≥ 1013 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l3917_391766


namespace NUMINAMATH_CALUDE_die_roll_probabilities_l3917_391727

-- Define the sample space for rolling a fair six-sided die twice
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events
def A : Set Ω := {ω | ω.1 + ω.2 = 4}
def B : Set Ω := {ω | ω.2 % 2 = 0}
def C : Set Ω := {ω | ω.1 = ω.2}
def D : Set Ω := {ω | ω.1 % 2 = 1 ∨ ω.2 % 2 = 1}

-- Theorem statement
theorem die_roll_probabilities :
  P D = 3/4 ∧
  P (B ∩ D) = 1/4 ∧
  P (B ∩ C) = P B * P C := by sorry

end NUMINAMATH_CALUDE_die_roll_probabilities_l3917_391727


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l3917_391729

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit in one dimension -/
def maxTilesInDimension (floorDim tileADim tileBDim : ℕ) : ℕ :=
  max (floorDim / tileADim) (floorDim / tileBDim)

/-- Calculates the total number of tiles for a given orientation -/
def totalTiles (floor tile : Dimensions) : ℕ :=
  (maxTilesInDimension floor.length tile.length tile.width) *
  (maxTilesInDimension floor.width tile.length tile.width)

/-- The main theorem stating the maximum number of tiles that can be accommodated -/
theorem max_tiles_on_floor :
  let floor := Dimensions.mk 180 120
  let tile := Dimensions.mk 25 16
  (max (totalTiles floor tile) (totalTiles floor (Dimensions.mk tile.width tile.length))) = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l3917_391729


namespace NUMINAMATH_CALUDE_polygon_150_sides_diagonals_l3917_391763

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 150 sides has 11025 diagonals -/
theorem polygon_150_sides_diagonals : num_diagonals 150 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_polygon_150_sides_diagonals_l3917_391763


namespace NUMINAMATH_CALUDE_train_speed_is_25_l3917_391708

-- Define the train and its properties
structure Train :=
  (speed : ℝ)
  (length : ℝ)

-- Define the tunnels
def tunnel1_length : ℝ := 85
def tunnel2_length : ℝ := 160
def tunnel1_time : ℝ := 5
def tunnel2_time : ℝ := 8

-- Theorem statement
theorem train_speed_is_25 (t : Train) :
  (tunnel1_length + t.length) / tunnel1_time = t.speed →
  (tunnel2_length + t.length) / tunnel2_time = t.speed →
  t.speed = 25 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_is_25_l3917_391708


namespace NUMINAMATH_CALUDE_average_fee_is_4_6_l3917_391733

/-- Represents the delivery statistics for a delivery person in December -/
structure DeliveryStats where
  short_distance_percent : ℝ  -- Percentage of deliveries ≤ 3 km
  long_distance_percent : ℝ   -- Percentage of deliveries > 3 km
  short_distance_fee : ℝ      -- Fee for deliveries ≤ 3 km
  long_distance_fee : ℝ       -- Fee for deliveries > 3 km

/-- Calculates the average delivery fee per order -/
def average_delivery_fee (stats : DeliveryStats) : ℝ :=
  stats.short_distance_percent * stats.short_distance_fee +
  stats.long_distance_percent * stats.long_distance_fee

/-- Theorem stating that the average delivery fee is 4.6 yuan for the given statistics -/
theorem average_fee_is_4_6 (stats : DeliveryStats) 
  (h1 : stats.short_distance_percent = 0.7)
  (h2 : stats.long_distance_percent = 0.3)
  (h3 : stats.short_distance_fee = 4)
  (h4 : stats.long_distance_fee = 6) :
  average_delivery_fee stats = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_average_fee_is_4_6_l3917_391733


namespace NUMINAMATH_CALUDE_integral_equals_ten_l3917_391734

theorem integral_equals_ten (k : ℝ) : 
  (∫ x in (0 : ℝ)..2, 3 * x^2 + k) = 10 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_ten_l3917_391734
