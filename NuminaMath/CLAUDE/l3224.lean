import Mathlib

namespace ratio_a_to_b_l3224_322403

theorem ratio_a_to_b (a b : ℝ) (h : (3 * a + 2 * b) / (3 * a - 2 * b) = 3) : a / b = 4 / 3 := by
  sorry

end ratio_a_to_b_l3224_322403


namespace brians_age_in_eight_years_l3224_322461

/-- Given that Christian is twice as old as Brian and Christian will be 72 years old in eight years,
    prove that Brian will be 40 years old in eight years. -/
theorem brians_age_in_eight_years (christian_age : ℕ) (brian_age : ℕ) : 
  christian_age = 2 * brian_age →
  christian_age + 8 = 72 →
  brian_age + 8 = 40 := by
  sorry

end brians_age_in_eight_years_l3224_322461


namespace line_y_intercept_l3224_322434

/-- A line with slope 3 and x-intercept (7,0) has y-intercept (0, -21) -/
theorem line_y_intercept (m : ℝ) (x₀ : ℝ) (y : ℝ → ℝ) :
  m = 3 →
  x₀ = 7 →
  y 0 = 0 →
  (∀ x, y x = m * (x - x₀)) →
  y 0 = -21 :=
by sorry

end line_y_intercept_l3224_322434


namespace honey_water_percentage_l3224_322466

/-- Given that 1.5 kg of flower-nectar yields 1 kg of honey and nectar contains 50% water,
    prove that the resulting honey contains 25% water. -/
theorem honey_water_percentage :
  ∀ (nectar_mass honey_mass water_percentage_nectar : ℝ),
    nectar_mass = 1.5 →
    honey_mass = 1 →
    water_percentage_nectar = 50 →
    (honey_mass - (nectar_mass * (1 - water_percentage_nectar / 100))) / honey_mass * 100 = 25 := by
  sorry

end honey_water_percentage_l3224_322466


namespace other_solution_quadratic_equation_l3224_322495

theorem other_solution_quadratic_equation :
  let f (x : ℚ) := 42 * x^2 + 2 * x + 31 - (73 * x + 4)
  (f (3/7) = 0) → (f (3/2) = 0) := by sorry

end other_solution_quadratic_equation_l3224_322495


namespace retail_price_increase_l3224_322471

theorem retail_price_increase (wholesale_cost employee_paid : ℝ) (employee_discount : ℝ) :
  wholesale_cost = 200 →
  employee_discount = 0.15 →
  employee_paid = 204 →
  ∃ (retail_price_increase : ℝ),
    retail_price_increase = 0.20 ∧
    employee_paid = wholesale_cost * (1 + retail_price_increase) * (1 - employee_discount) :=
by
  sorry

end retail_price_increase_l3224_322471


namespace student_rabbit_difference_l3224_322493

/-- Given 4 classrooms, each with 18 students and 2 rabbits, prove that the difference
    between the total number of students and rabbits is 64. -/
theorem student_rabbit_difference (num_classrooms : ℕ) (students_per_class : ℕ) (rabbits_per_class : ℕ)
    (h1 : num_classrooms = 4)
    (h2 : students_per_class = 18)
    (h3 : rabbits_per_class = 2) :
    num_classrooms * students_per_class - num_classrooms * rabbits_per_class = 64 := by
  sorry

end student_rabbit_difference_l3224_322493


namespace distinct_sums_count_l3224_322464

/-- Represents the number of coins of each denomination -/
structure CoinSet :=
  (one_yuan : ℕ)
  (half_yuan : ℕ)

/-- Represents a selection of coins -/
structure CoinSelection :=
  (one_yuan : ℕ)
  (half_yuan : ℕ)

/-- Calculates the sum of face values for a given coin selection -/
def sumFaceValues (selection : CoinSelection) : ℚ :=
  selection.one_yuan + selection.half_yuan / 2

/-- Generates all possible coin selections given a coin set and total number of coins to select -/
def possibleSelections (coins : CoinSet) (total : ℕ) : List CoinSelection :=
  sorry

/-- Calculates the number of distinct sums from all possible selections -/
def distinctSums (coins : CoinSet) (total : ℕ) : ℕ :=
  (possibleSelections coins total).map sumFaceValues |> List.eraseDups |> List.length

/-- The main theorem stating that there are exactly 7 distinct sums when selecting 6 coins from 5 one-yuan and 6 half-yuan coins -/
theorem distinct_sums_count :
  distinctSums (CoinSet.mk 5 6) 6 = 7 := by sorry

end distinct_sums_count_l3224_322464


namespace maria_remaining_towels_l3224_322431

def green_towels : ℕ := 35
def white_towels : ℕ := 21
def towels_given_to_mother : ℕ := 34

theorem maria_remaining_towels :
  green_towels + white_towels - towels_given_to_mother = 22 :=
by sorry

end maria_remaining_towels_l3224_322431


namespace remainder_8423_div_9_l3224_322467

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The digital root of a natural number (iterative sum of digits until a single digit is reached) -/
def digital_root (n : ℕ) : ℕ := sorry

theorem remainder_8423_div_9 : 8423 % 9 = 8 := by sorry

end remainder_8423_div_9_l3224_322467


namespace infinite_lcm_greater_than_ck_l3224_322401

theorem infinite_lcm_greater_than_ck 
  (a : ℕ → ℕ) 
  (c : ℝ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_positive : ∀ n, a n > 0) 
  (h_c : 0 < c ∧ c < 1.5) : 
  ∀ N, ∃ k > N, Nat.lcm (a k) (a (k + 1)) > ⌊c * k⌋ := by
  sorry

end infinite_lcm_greater_than_ck_l3224_322401


namespace function_inequality_implies_a_bound_l3224_322415

open Real

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ > a, log x₁ + 1/x₁ ≥ x₂ + 1/(x₂ - a)) → 
  a ≤ -1 := by
  sorry

end function_inequality_implies_a_bound_l3224_322415


namespace equal_roots_condition_l3224_322406

theorem equal_roots_condition (m : ℝ) : 
  (∃ (x : ℝ), (x * (x - 3) - (m + 2)) / ((x - 2) * (m - 2)) = x / m) ∧ 
  (∀ (x y : ℝ), (x * (x - 3) - (m + 2)) / ((x - 2) * (m - 2)) = x / m ∧ 
                 (y * (y - 3) - (m + 2)) / ((y - 2) * (m - 2)) = y / m 
                 → x = y) ↔ 
  m = (-7 + Real.sqrt 2) / 2 ∨ m = (-7 - Real.sqrt 2) / 2 :=
sorry

end equal_roots_condition_l3224_322406


namespace additional_amount_needed_for_free_shipping_l3224_322460

def free_shipping_threshold : ℝ := 50.00

def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00

def first_two_discount : ℝ := 0.25
def total_discount : ℝ := 0.10

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_price : ℝ :=
  discounted_price book1_price first_two_discount +
  discounted_price book2_price first_two_discount +
  book3_price + book4_price

def final_price : ℝ :=
  discounted_price total_price total_discount

theorem additional_amount_needed_for_free_shipping :
  free_shipping_threshold - final_price = 13.10 := by
  sorry

end additional_amount_needed_for_free_shipping_l3224_322460


namespace expression_bounds_l3224_322427

theorem expression_bounds (x y z w : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ 
    Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ∧
  Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ≤ 4 :=
by sorry

end expression_bounds_l3224_322427


namespace min_fence_posts_for_field_l3224_322432

/-- Calculates the number of fence posts needed for a rectangular field -/
def fence_posts (length width post_spacing_long post_spacing_short : ℕ) : ℕ :=
  let long_side_posts := length / post_spacing_long + 1
  let short_side_posts := width / post_spacing_short + 1
  long_side_posts + 2 * (short_side_posts - 1)

/-- Theorem stating the minimum number of fence posts required for the given field -/
theorem min_fence_posts_for_field : 
  fence_posts 150 50 15 10 = 21 :=
by sorry

end min_fence_posts_for_field_l3224_322432


namespace bracket_difference_l3224_322483

theorem bracket_difference (a b c : ℝ) : (a - (b - c)) - ((a - b) - c) = 2 * c := by
  sorry

end bracket_difference_l3224_322483


namespace product_equality_l3224_322473

theorem product_equality (h : 213 * 16 = 3408) : 0.16 * 2.13 = 0.3408 := by
  sorry

end product_equality_l3224_322473


namespace shortest_distance_between_circles_l3224_322453

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles :
  let circle1 := fun (x y : ℝ) => x^2 - 6*x + y^2 - 8*y - 15 = 0
  let circle2 := fun (x y : ℝ) => x^2 + 10*x + y^2 + 12*y + 21 = 0
  ∃ d : ℝ, d = 2 * Real.sqrt 41 - Real.sqrt 97 ∧
    ∀ p q : ℝ × ℝ, circle1 p.1 p.2 → circle2 q.1 q.2 →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

end shortest_distance_between_circles_l3224_322453


namespace triangle_problem_l3224_322436

theorem triangle_problem (A B C : Real) (a b c : Real) :
  0 < A → A < π / 2 →
  (1 / 2) * b * c * Real.sin A = (Real.sqrt 3 / 4) * b * c →
  c / b = 1 / 2 + Real.sqrt 3 →
  A = π / 3 ∧ Real.tan B = 1 / 2 := by
  sorry

end triangle_problem_l3224_322436


namespace contrapositive_odd_product_l3224_322405

theorem contrapositive_odd_product (a b : ℤ) :
  (¬(Odd (a * b)) → ¬(Odd a ∧ Odd b)) ↔
  ((Odd a ∧ Odd b) → Odd (a * b)) := by sorry

end contrapositive_odd_product_l3224_322405


namespace min_value_abs_sum_l3224_322465

theorem min_value_abs_sum (x : ℝ) (h : 0 ≤ x ∧ x ≤ 10) :
  |x - 4| + |x + 2| + |x - 5| + |3*x - 1| + |2*x + 6| ≥ 17.333 := by
  sorry

end min_value_abs_sum_l3224_322465


namespace jeans_final_price_l3224_322496

/-- Calculates the final price of jeans after summer and Wednesday discounts --/
theorem jeans_final_price (original_price : ℝ) (summer_discount_percent : ℝ) (wednesday_discount : ℝ) :
  original_price = 49 →
  summer_discount_percent = 50 →
  wednesday_discount = 10 →
  original_price * (1 - summer_discount_percent / 100) - wednesday_discount = 14.5 := by
  sorry

#check jeans_final_price

end jeans_final_price_l3224_322496


namespace other_x_intercept_is_seven_l3224_322448

/-- A quadratic function with vertex (4, -3) and one x-intercept at (1, 0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 4
  vertex_y : ℝ := -3
  intercept_x : ℝ := 1

/-- The x-coordinate of the other x-intercept of the quadratic function -/
def other_x_intercept (f : QuadraticFunction) : ℝ := 7

/-- Theorem stating that the x-coordinate of the other x-intercept is 7 -/
theorem other_x_intercept_is_seven (f : QuadraticFunction) :
  other_x_intercept f = 7 := by sorry

end other_x_intercept_is_seven_l3224_322448


namespace first_month_sale_is_3435_l3224_322409

/-- Calculates the sale in the first month given the sales for the next 5 months and the average sale -/
def first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale2 + sale3 + sale4 + sale5 + sale6)

/-- The sale in the first month is 3435 given the conditions of the problem -/
theorem first_month_sale_is_3435 :
  first_month_sale 3927 3855 4230 3562 1991 3500 = 3435 := by
sorry

#eval first_month_sale 3927 3855 4230 3562 1991 3500

end first_month_sale_is_3435_l3224_322409


namespace paths_count_l3224_322435

/-- The number of distinct paths from (0, n) to (m, m) on a plane,
    where only moves of 1 unit up or 1 unit left are allowed. -/
def numPaths (n m : ℕ) : ℕ :=
  Nat.choose n m

/-- Theorem stating that the number of distinct paths from (0, n) to (m, m)
    is equal to (n choose m) -/
theorem paths_count (n m : ℕ) (h : m ≤ n) :
  numPaths n m = Nat.choose n m := by
  sorry

end paths_count_l3224_322435


namespace sqrt_seven_to_sixth_l3224_322425

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_sixth_l3224_322425


namespace stating_average_enter_exit_time_l3224_322417

/-- Represents the speed of the car in miles per minute -/
def car_speed : ℚ := 5/4

/-- Represents the speed of the storm in miles per minute -/
def storm_speed : ℚ := 1/2

/-- Represents the radius of the storm in miles -/
def storm_radius : ℚ := 51

/-- Represents the initial y-coordinate of the storm center in miles -/
def initial_storm_y : ℚ := 110

/-- 
Theorem stating that the average time at which the car enters and exits the storm is 880/29 minutes
-/
theorem average_enter_exit_time : 
  let car_pos (t : ℚ) := (car_speed * t, 0)
  let storm_center (t : ℚ) := (0, initial_storm_y - storm_speed * t)
  let distance (t : ℚ) := 
    ((car_pos t).1 - (storm_center t).1)^2 + ((car_pos t).2 - (storm_center t).2)^2
  ∃ t₁ t₂,
    distance t₁ = storm_radius^2 ∧ 
    distance t₂ = storm_radius^2 ∧ 
    t₁ < t₂ ∧
    (t₁ + t₂) / 2 = 880 / 29 :=
sorry

end stating_average_enter_exit_time_l3224_322417


namespace triathlon_bike_speed_l3224_322418

def triathlon_speed (swim_distance : ℚ) (bike_distance : ℚ) (run_distance : ℚ)
                    (swim_speed : ℚ) (run_speed : ℚ) (total_time : ℚ) : ℚ :=
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let bike_time := total_time - swim_time - run_time
  bike_distance / bike_time

theorem triathlon_bike_speed :
  triathlon_speed (1/2) 10 2 (3/2) 5 (3/2) = 13 := by
  sorry

end triathlon_bike_speed_l3224_322418


namespace expression_simplification_l3224_322420

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - 3 / (x + 2)) / ((x^2 - 1) / (x + 2)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l3224_322420


namespace second_price_reduction_l3224_322442

theorem second_price_reduction (P : ℝ) (x : ℝ) (h1 : P > 0) :
  (P - 0.25 * P) * (1 - x / 100) = P * (1 - 0.7) → x = 60 := by
  sorry

end second_price_reduction_l3224_322442


namespace solution_value_l3224_322440

theorem solution_value (x y a : ℝ) : 
  x = 1 ∧ y = 1 ∧ 2*x - a*y = 3 → a = -1 := by
  sorry

end solution_value_l3224_322440


namespace quadratic_equation_roots_l3224_322422

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k - 2) * x₁^2 - 2 * x₁ + (1/2) = 0 ∧ 
    (k - 2) * x₂^2 - 2 * x₂ + (1/2) = 0) ↔ 
  (k < 4 ∧ k ≠ 2) :=
by sorry

end quadratic_equation_roots_l3224_322422


namespace computer_price_increase_l3224_322490

theorem computer_price_increase (c : ℝ) : 
  c + c * 0.3 = 351 → c + 351 = 621 :=
by sorry

end computer_price_increase_l3224_322490


namespace sum_of_3rd_4th_5th_terms_l3224_322413

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem sum_of_3rd_4th_5th_terms
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_common_ratio : q = 2)
  (h_sum_first_3 : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
sorry

end sum_of_3rd_4th_5th_terms_l3224_322413


namespace rectangle_to_square_length_l3224_322414

theorem rectangle_to_square_length (width : ℝ) (height : ℝ) (y : ℝ) :
  width = 10 →
  height = 20 →
  (width * height = y * y * 16) →
  y = 5 * Real.sqrt 2 / 2 :=
by
  sorry

end rectangle_to_square_length_l3224_322414


namespace abs_3_minus_4i_l3224_322424

theorem abs_3_minus_4i : Complex.abs (3 - 4 * Complex.I) = 5 := by
  sorry

end abs_3_minus_4i_l3224_322424


namespace multiplication_problem_l3224_322469

theorem multiplication_problem : ∃ x : ℕ, 582964 * x = 58293485180 ∧ x = 100000 := by
  sorry

end multiplication_problem_l3224_322469


namespace first_grade_sample_size_l3224_322428

/-- Given a total sample size and ratios for three groups, 
    calculate the number of samples for the first group -/
def stratifiedSampleSize (totalSample : ℕ) (ratio1 ratio2 ratio3 : ℕ) : ℕ :=
  (ratio1 * totalSample) / (ratio1 + ratio2 + ratio3)

/-- Theorem: For a total sample of 80 and ratios 4:3:3, 
    the first group's sample size is 32 -/
theorem first_grade_sample_size :
  stratifiedSampleSize 80 4 3 3 = 32 := by
  sorry

end first_grade_sample_size_l3224_322428


namespace parabola_directrix_l3224_322486

/-- The equation of the directrix of a parabola with equation x² = 4y and focus at (0, 1) -/
theorem parabola_directrix : ∃ (l : ℝ → ℝ), 
  (∀ x y : ℝ, x^2 = 4*y → (∀ t : ℝ, (x - 0)^2 + (y - 1)^2 = (x - t)^2 + (y - l t)^2)) → 
  (∀ t : ℝ, l t = -1) :=
sorry

end parabola_directrix_l3224_322486


namespace solution_implies_m_equals_one_l3224_322480

theorem solution_implies_m_equals_one (x y m : ℝ) : 
  x = 2 → y = -1 → m * x - y = 3 → m = 1 := by
  sorry

end solution_implies_m_equals_one_l3224_322480


namespace hundred_hours_before_seven_am_l3224_322470

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the time a given number of hours before a specified time -/
def timeBefore (t : TimeOfDay) (h : Nat) : TimeOfDay :=
  sorry

/-- Theorem: 100 hours before 7:00 a.m. is 3:00 a.m. -/
theorem hundred_hours_before_seven_am :
  let start_time : TimeOfDay := ⟨7, 0, by sorry⟩
  let end_time : TimeOfDay := ⟨3, 0, by sorry⟩
  timeBefore start_time 100 = end_time := by
  sorry

end hundred_hours_before_seven_am_l3224_322470


namespace tiger_tree_trunk_time_l3224_322439

/-- The time taken for a tiger to run above a fallen tree trunk -/
theorem tiger_tree_trunk_time (tiger_length : ℝ) (tree_trunk_length : ℝ) (time_to_pass_point : ℝ) : 
  tiger_length = 5 →
  tree_trunk_length = 20 →
  time_to_pass_point = 1 →
  (tiger_length + tree_trunk_length) / (tiger_length / time_to_pass_point) = 5 := by
  sorry

end tiger_tree_trunk_time_l3224_322439


namespace arithmetic_sequence_ninth_term_l3224_322426

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℤ)
  (h_arith : ArithmeticSequence a)
  (h_diff : a 3 - a 2 = -2)
  (h_seventh : a 7 = -2) :
  a 9 = -6 := by
  sorry

end arithmetic_sequence_ninth_term_l3224_322426


namespace green_pill_cost_proof_l3224_322482

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℝ := 21

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℝ := green_pill_cost - 3

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 819

theorem green_pill_cost_proof :
  green_pill_cost = 21 ∧
  pink_pill_cost = green_pill_cost - 3 ∧
  treatment_days = 21 ∧
  total_cost = 819 ∧
  treatment_days * (green_pill_cost + pink_pill_cost) = total_cost :=
by sorry

end green_pill_cost_proof_l3224_322482


namespace problem_solution_l3224_322402

theorem problem_solution (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^2 + 1/a^2 + a^3 + 1/a^3 = 3 + 2 * Real.sqrt 5 := by
  sorry

end problem_solution_l3224_322402


namespace school_prizes_l3224_322433

theorem school_prizes (total_money : ℝ) (pen_cost notebook_cost : ℝ) 
  (h1 : total_money = 60 * (pen_cost + 2 * notebook_cost))
  (h2 : total_money = 50 * (pen_cost + 3 * notebook_cost)) :
  (total_money / pen_cost : ℝ) = 100 := by
  sorry

end school_prizes_l3224_322433


namespace no_fraternity_member_is_club_member_l3224_322429

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (student : U → Prop)
variable (club_member : U → Prop)
variable (fraternity_member : U → Prop)
variable (honest : U → Prop)

-- State the theorem
theorem no_fraternity_member_is_club_member
  (h1 : ∀ x, club_member x → student x)
  (h2 : ∀ x, club_member x → ¬honest x)
  (h3 : ∀ x, fraternity_member x → honest x) :
  ∀ x, fraternity_member x → ¬club_member x :=
by
  sorry


end no_fraternity_member_is_club_member_l3224_322429


namespace least_cube_divisible_by_168_l3224_322449

theorem least_cube_divisible_by_168 :
  ∀ k : ℕ, k > 0 → k^3 % 168 = 0 → k ≥ 42 :=
by
  sorry

end least_cube_divisible_by_168_l3224_322449


namespace arithmetic_sequence_2009_l3224_322485

/-- Given an arithmetic sequence {a_n} with common difference d and a_k, 
    this function returns a_n -/
def arithmeticSequence (d : ℤ) (k : ℕ) (a_k : ℤ) (n : ℕ) : ℤ :=
  a_k + d * (n - k)

theorem arithmetic_sequence_2009 :
  let d := 2
  let k := 2007
  let a_k := 2007
  let n := 2009
  arithmeticSequence d k a_k n = 2011 := by
  sorry

end arithmetic_sequence_2009_l3224_322485


namespace exactly_two_out_of_three_germinate_l3224_322491

def seed_germination_probability : ℚ := 3/5

def exactly_two_out_of_three_probability : ℚ :=
  3 * seed_germination_probability^2 * (1 - seed_germination_probability)

theorem exactly_two_out_of_three_germinate :
  exactly_two_out_of_three_probability = 54/125 := by
  sorry

end exactly_two_out_of_three_germinate_l3224_322491


namespace arithmetic_sequence_2015th_term_l3224_322438

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_2015th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a5 : a 5 = 6) :
  a 2015 = 2016 := by
  sorry

end arithmetic_sequence_2015th_term_l3224_322438


namespace sqrt_expression_equals_seven_halves_l3224_322445

theorem sqrt_expression_equals_seven_halves :
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 / Real.sqrt 24 = 7/2 := by
  sorry

end sqrt_expression_equals_seven_halves_l3224_322445


namespace square_with_semicircular_arcs_perimeter_l3224_322441

/-- The perimeter of a region bounded by semicircular arcs constructed on the sides of a square -/
theorem square_with_semicircular_arcs_perimeter (side_length : Real) : 
  side_length = 4 / Real.pi → 
  (4 : Real) * Real.pi * (side_length / 2) = 8 := by
  sorry

end square_with_semicircular_arcs_perimeter_l3224_322441


namespace count_students_without_A_l3224_322408

/-- The number of students who did not receive an A in any subject -/
def students_without_A (total_students : ℕ) (history_A : ℕ) (math_A : ℕ) (science_A : ℕ) 
  (math_history_A : ℕ) (history_science_A : ℕ) (science_math_A : ℕ) (all_subjects_A : ℕ) : ℕ :=
  total_students - (history_A + math_A + science_A - math_history_A - history_science_A - science_math_A + all_subjects_A)

theorem count_students_without_A :
  students_without_A 50 9 15 12 5 3 4 1 = 28 := by
  sorry

end count_students_without_A_l3224_322408


namespace remaining_work_time_l3224_322484

theorem remaining_work_time (a_rate b_rate : ℚ) (b_work_days : ℕ) : 
  a_rate = 1 / 12 →
  b_rate = 1 / 15 →
  b_work_days = 10 →
  (1 - b_rate * b_work_days) / a_rate = 4 :=
by
  sorry

end remaining_work_time_l3224_322484


namespace erica_earnings_l3224_322412

def fish_price : ℕ := 20
def past_four_months_catch : ℕ := 80
def monthly_maintenance : ℕ := 50
def fuel_cost_per_kg : ℕ := 2
def num_months : ℕ := 5

def total_catch : ℕ := past_four_months_catch * 3

def total_income : ℕ := total_catch * fish_price

def total_maintenance_cost : ℕ := monthly_maintenance * num_months

def total_fuel_cost : ℕ := fuel_cost_per_kg * total_catch

def total_cost : ℕ := total_maintenance_cost + total_fuel_cost

def net_income : ℤ := total_income - total_cost

theorem erica_earnings : net_income = 4070 := by
  sorry

end erica_earnings_l3224_322412


namespace coefficient_x_squared_eq_five_l3224_322468

/-- The coefficient of x^2 in the expansion of (1/x^2 + x)^5 -/
def coefficient_x_squared : ℕ :=
  (Nat.choose 5 1)

theorem coefficient_x_squared_eq_five : coefficient_x_squared = 5 := by
  sorry

end coefficient_x_squared_eq_five_l3224_322468


namespace exactly_six_solutions_l3224_322451

/-- The number of ordered pairs of positive integers satisfying 3/m + 6/n = 1 -/
def solution_count : ℕ := 6

/-- Predicate for ordered pairs (m,n) satisfying the equation -/
def satisfies_equation (m n : ℕ+) : Prop :=
  (3 : ℚ) / m.val + (6 : ℚ) / n.val = 1

/-- The theorem stating that there are exactly 6 solutions -/
theorem exactly_six_solutions :
  ∃! (s : Finset (ℕ+ × ℕ+)), 
    s.card = solution_count ∧ 
    (∀ p ∈ s, satisfies_equation p.1 p.2) ∧
    (∀ m n : ℕ+, satisfies_equation m n → (m, n) ∈ s) :=
  sorry

end exactly_six_solutions_l3224_322451


namespace tan_product_simplification_l3224_322487

theorem tan_product_simplification :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end tan_product_simplification_l3224_322487


namespace unique_polygon_diagonals_l3224_322498

/-- The number of diagonals in a convex polygon with k sides -/
def numDiagonals (k : ℕ) : ℚ := (k * (k - 3)) / 2

/-- The condition for the number of diagonals in the two polygons -/
def diagonalCondition (n : ℕ) : Prop :=
  numDiagonals (3 * n + 2) = (1 - 0.615) * numDiagonals (5 * n - 2)

theorem unique_polygon_diagonals : ∃! (n : ℕ), n > 0 ∧ diagonalCondition n :=
  sorry

end unique_polygon_diagonals_l3224_322498


namespace library_books_remaining_l3224_322488

theorem library_books_remaining (initial_books : ℕ) (given_away : ℕ) (donated : ℕ) : 
  initial_books = 125 → given_away = 42 → donated = 31 → 
  initial_books - given_away - donated = 52 := by
sorry

end library_books_remaining_l3224_322488


namespace complex_fraction_simplification_l3224_322411

theorem complex_fraction_simplification :
  (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I :=
by sorry

end complex_fraction_simplification_l3224_322411


namespace buffet_combinations_l3224_322478

def num_meat_options : ℕ := 4
def num_vegetable_options : ℕ := 5
def num_vegetables_to_choose : ℕ := 3
def num_dessert_options : ℕ := 4
def num_desserts_to_choose : ℕ := 2

def choose (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem buffet_combinations :
  (num_meat_options) *
  (choose num_vegetable_options num_vegetables_to_choose) *
  (choose num_dessert_options num_desserts_to_choose) = 240 := by
  sorry

end buffet_combinations_l3224_322478


namespace bryden_receives_correct_amount_l3224_322499

/-- The amount Bryden receives for his state quarters -/
def bryden_amount (num_quarters : ℕ) (face_value : ℚ) (percentage : ℕ) (bonus_per_five : ℚ) : ℚ :=
  let base_amount := (num_quarters : ℚ) * face_value * (percentage : ℚ) / 100
  let num_bonuses := num_quarters / 5
  base_amount + (num_bonuses : ℚ) * bonus_per_five

/-- Theorem stating that Bryden will receive $45.75 for his seven state quarters -/
theorem bryden_receives_correct_amount :
  bryden_amount 7 0.25 2500 2 = 45.75 := by
  sorry

end bryden_receives_correct_amount_l3224_322499


namespace sum_of_two_squares_l3224_322430

theorem sum_of_two_squares (x y : ℝ) : 2 * x^2 + 2 * y^2 = (x + y)^2 + (x - y)^2 := by
  sorry

end sum_of_two_squares_l3224_322430


namespace indefinite_integral_proof_l3224_322416

theorem indefinite_integral_proof (x : Real) :
  let f := fun x => -(1 / (x - Real.sin x))
  let g := fun x => (1 - Real.cos x) / (x - Real.sin x)^2
  deriv f x = g x :=
by sorry

end indefinite_integral_proof_l3224_322416


namespace benny_spent_85_dollars_l3224_322400

def baseball_gear_total (glove_price baseball_price bat_price helmet_price gloves_price : ℕ) : ℕ :=
  glove_price + baseball_price + bat_price + helmet_price + gloves_price

theorem benny_spent_85_dollars : 
  baseball_gear_total 25 5 30 15 10 = 85 := by
  sorry

end benny_spent_85_dollars_l3224_322400


namespace ruble_combinations_l3224_322444

theorem ruble_combinations : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 5 * p.1 + 3 * p.2 = 78) 
    (Finset.product (Finset.range 79) (Finset.range 79))).card ∧ n = 5 := by
  sorry

end ruble_combinations_l3224_322444


namespace age_difference_l3224_322462

/-- Given three people A, B, and C, where the total age of A and B is 18 years more than
    the total age of B and C, prove that C is 18 years younger than A. -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 18) : A = C + 18 := by
  sorry

end age_difference_l3224_322462


namespace y_relationship_l3224_322481

/-- A linear function with slope -2 and y-intercept 5 -/
def f (x : ℝ) : ℝ := -2 * x + 5

/-- Theorem stating the relationship between y-values for specific x-values in the linear function f -/
theorem y_relationship (x₁ y₁ y₂ y₃ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f (x₁ - 2) = y₂) 
  (h3 : f (x₁ + 3) = y₃) : 
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end y_relationship_l3224_322481


namespace bagel_store_spending_l3224_322410

theorem bagel_store_spending :
  ∀ (B D : ℝ),
  D = (7/10) * B →
  B = D + 15 →
  B + D = 85 :=
by
  sorry

end bagel_store_spending_l3224_322410


namespace calculation_problem_1_calculation_problem_2_l3224_322476

-- Question 1
theorem calculation_problem_1 :
  (-1/4)⁻¹ - |Real.sqrt 3 - 1| + 3 * Real.tan (30 * π / 180) + (2017 - π) = -2 := by sorry

-- Question 2
theorem calculation_problem_2 (x : ℝ) (h : x = 2) :
  (2 * x^2) / (x^2 - 2*x + 1) / ((2*x + 1) / (x + 1) + 1 / (x - 1)) = 3 := by sorry

end calculation_problem_1_calculation_problem_2_l3224_322476


namespace arithmetic_square_root_when_negative_root_is_five_l3224_322463

theorem arithmetic_square_root_when_negative_root_is_five (x : ℝ) : 
  ((-5 : ℝ)^2 = x) → Real.sqrt x = 5 := by
  sorry

end arithmetic_square_root_when_negative_root_is_five_l3224_322463


namespace difference_of_two_numbers_l3224_322472

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) :
  |x - y| = 4 := by sorry

end difference_of_two_numbers_l3224_322472


namespace cube_sum_and_reciprocal_l3224_322494

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end cube_sum_and_reciprocal_l3224_322494


namespace son_age_l3224_322489

theorem son_age (son_age father_age : ℕ) : 
  father_age = son_age + 20 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
sorry

end son_age_l3224_322489


namespace bugs_meet_on_bc_l3224_322459

/-- Triangle with side lengths -/
structure Triangle where
  ab : ℝ
  bc : ℝ
  ac : ℝ

/-- Bug with starting position and speed -/
structure Bug where
  start : ℕ  -- 0 for A, 1 for B, 2 for C
  speed : ℝ
  clockwise : Bool

/-- The point where bugs meet -/
def MeetingPoint (t : Triangle) (bugA bugC : Bug) : ℝ := sorry

theorem bugs_meet_on_bc (t : Triangle) (bugA bugC : Bug) :
  t.ab = 5 ∧ t.bc = 6 ∧ t.ac = 7 ∧
  bugA.start = 0 ∧ bugA.speed = 1 ∧ bugA.clockwise = true ∧
  bugC.start = 2 ∧ bugC.speed = 2 ∧ bugC.clockwise = false →
  MeetingPoint t bugA bugC = 1 := by sorry

end bugs_meet_on_bc_l3224_322459


namespace max_min_product_l3224_322474

theorem max_min_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 12) (prod_sum_eq : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 9 ∧ 
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 9 :=
sorry

end max_min_product_l3224_322474


namespace exists_top_choice_l3224_322419

/- Define the type for houses and people -/
variable {α : Type*} [Finite α]

/- Define the preference relation -/
def Prefers (p : α → α → Prop) : Prop :=
  ∀ x y z, p x y ∧ p y z → p x z

/- Define the assignment function -/
def Assignment (f : α → α) : Prop :=
  Function.Bijective f

/- Define the stability condition -/
def Stable (f : α → α) (p : α → α → Prop) : Prop :=
  ∀ g : α → α, Assignment g →
    ∃ x, p x (f x) ∧ ¬p x (g x)

/- State the theorem -/
theorem exists_top_choice
  (f : α → α)
  (p : α → α → Prop)
  (h_assign : Assignment f)
  (h_prefers : Prefers p)
  (h_stable : Stable f p) :
  ∃ x, ∀ y, p x (f x) ∧ (p x y → y = f x) :=
sorry

end exists_top_choice_l3224_322419


namespace unique_number_with_conditions_l3224_322452

theorem unique_number_with_conditions : ∃! n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ 
  2 ∣ n ∧
  3 ∣ (n + 1) ∧
  4 ∣ (n + 2) ∧
  5 ∣ (n + 3) ∧
  n = 62 := by
sorry

end unique_number_with_conditions_l3224_322452


namespace product_from_lcm_gcd_l3224_322450

theorem product_from_lcm_gcd (x y : ℕ+) 
  (h_lcm : Nat.lcm x y = 48) 
  (h_gcd : Nat.gcd x y = 8) : 
  x * y = 384 := by
sorry

end product_from_lcm_gcd_l3224_322450


namespace bat_lifespan_solution_l3224_322479

def bat_lifespan_problem (bat_lifespan : ℕ) : Prop :=
  let hamster_lifespan := bat_lifespan - 6
  let frog_lifespan := 4 * hamster_lifespan
  bat_lifespan + hamster_lifespan + frog_lifespan = 30

theorem bat_lifespan_solution :
  ∃ (bat_lifespan : ℕ), bat_lifespan_problem bat_lifespan ∧ bat_lifespan = 10 := by
  sorry

end bat_lifespan_solution_l3224_322479


namespace sin_cos_sum_identity_l3224_322497

theorem sin_cos_sum_identity (x y : ℝ) :
  Real.sin (x + y) * Real.cos (2 * y) + Real.cos (x + y) * Real.sin (2 * y) = Real.sin (x + 3 * y) := by
  sorry

end sin_cos_sum_identity_l3224_322497


namespace beth_coin_sale_l3224_322423

/-- Given Beth's initial gold coins and Carl's gift, prove the number of coins Beth sold when she sold half her total. -/
theorem beth_coin_sale (initial_coins : ℕ) (gift_coins : ℕ) : 
  initial_coins = 125 → gift_coins = 35 → (initial_coins + gift_coins) / 2 = 80 := by
sorry

end beth_coin_sale_l3224_322423


namespace regular_polygon_sides_l3224_322492

theorem regular_polygon_sides (interior_angle : ℝ) : 
  interior_angle = 140 → (360 / (180 - interior_angle) : ℝ) = 9 := by
  sorry

end regular_polygon_sides_l3224_322492


namespace least_positive_integer_with_remainders_l3224_322458

theorem least_positive_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 5 = 4 ∧ 
  n % 7 = 6 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 5 = 4 ∧ m % 7 = 6 → n ≤ m :=
by
  use 69
  sorry

end least_positive_integer_with_remainders_l3224_322458


namespace quadratic_roots_imply_a_less_than_one_l3224_322446

theorem quadratic_roots_imply_a_less_than_one (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0) → a < 1 := by
  sorry

end quadratic_roots_imply_a_less_than_one_l3224_322446


namespace line_parallel_to_intersection_of_parallel_planes_l3224_322421

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection_plane_plane : Plane → Plane → Line)

theorem line_parallel_to_intersection_of_parallel_planes 
  (a : Line) (α β : Plane) (b : Line) :
  parallel_line_plane a α →
  parallel_line_plane a β →
  intersection_plane_plane α β = b →
  parallel_line_line a b := by
sorry

end line_parallel_to_intersection_of_parallel_planes_l3224_322421


namespace trishSellPriceIs150Cents_l3224_322407

/-- The price at which Trish sells each stuffed animal -/
def trishSellPrice (barbaraStuffedAnimals : ℕ) (barbaraSellPrice : ℚ) (totalDonation : ℚ) : ℚ :=
  let trishStuffedAnimals := 2 * barbaraStuffedAnimals
  let barbaraContribution := barbaraStuffedAnimals * barbaraSellPrice
  let trishContribution := totalDonation - barbaraContribution
  trishContribution / trishStuffedAnimals

theorem trishSellPriceIs150Cents 
  (barbaraStuffedAnimals : ℕ) 
  (barbaraSellPrice : ℚ) 
  (totalDonation : ℚ) 
  (h1 : barbaraStuffedAnimals = 9)
  (h2 : barbaraSellPrice = 2)
  (h3 : totalDonation = 45) :
  trishSellPrice barbaraStuffedAnimals barbaraSellPrice totalDonation = 3/2 := by
  sorry

#eval trishSellPrice 9 2 45

end trishSellPriceIs150Cents_l3224_322407


namespace tan_beta_value_l3224_322447

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := by sorry

end tan_beta_value_l3224_322447


namespace bob_second_week_hours_l3224_322404

/-- Calculates the total pay for a given number of hours worked --/
def calculatePay (hours : ℕ) : ℕ :=
  if hours ≤ 40 then
    hours * 5
  else
    40 * 5 + (hours - 40) * 6

theorem bob_second_week_hours :
  ∃ (second_week_hours : ℕ),
    calculatePay 44 + calculatePay second_week_hours = 472 ∧
    second_week_hours = 48 := by
  sorry

end bob_second_week_hours_l3224_322404


namespace player_a_wins_two_player_player_b_wins_three_player_l3224_322457

/-- Represents a player in the Lazy Checkers game -/
inductive Player : Type
| A : Player
| B : Player
| C : Player

/-- Represents a position on the 5x5 board -/
structure Position :=
(row : Fin 5)
(col : Fin 5)

/-- Represents the state of the Lazy Checkers game -/
structure GameState :=
(board : Position → Option Player)
(current_player : Player)

/-- Represents a winning strategy for a player -/
def WinningStrategy (p : Player) : Type :=
GameState → Position

/-- The rules of Lazy Checkers ensure a valid game state -/
def ValidGameState (state : GameState) : Prop :=
sorry

/-- Theorem: In a two-player Lazy Checkers game, Player A has a winning strategy -/
theorem player_a_wins_two_player :
  ∃ (strategy : WinningStrategy Player.A),
    ∀ (initial_state : GameState),
      ValidGameState initial_state →
      initial_state.current_player = Player.A →
      -- The strategy leads to a win for Player A
      sorry :=
sorry

/-- Theorem: In a three-player Lazy Checkers game, Player B has a winning strategy when cooperating with Player C -/
theorem player_b_wins_three_player :
  ∃ (strategy_b : WinningStrategy Player.B) (strategy_c : WinningStrategy Player.C),
    ∀ (initial_state : GameState),
      ValidGameState initial_state →
      initial_state.current_player = Player.A →
      -- The strategies lead to a win for Player B
      sorry :=
sorry

end player_a_wins_two_player_player_b_wins_three_player_l3224_322457


namespace max_m_value_l3224_322443

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x + x^2 - m*x + Real.exp (2 - x)

theorem max_m_value :
  ∃ (m_max : ℝ), m_max = 3 ∧ 
  (∀ (m : ℝ), (∀ (x : ℝ), x > 0 → f m x ≥ 0) → m ≤ m_max) ∧
  (∀ (x : ℝ), x > 0 → f m_max x ≥ 0) :=
sorry

end max_m_value_l3224_322443


namespace parallelogram_area_l3224_322475

/-- The area of a parallelogram with one angle of 100 degrees and two consecutive sides of lengths 10 and 20 is approximately 196.96 -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h₁ : a = 10) (h₂ : b = 20) (h₃ : θ = 100 * π / 180) :
  abs (a * b * Real.sin θ - 196.96) < 0.01 := by
  sorry

end parallelogram_area_l3224_322475


namespace max_satisfying_all_is_50_l3224_322456

/-- Represents the youth summer village population --/
structure Village where
  total : ℕ
  notWorking : ℕ
  withFamily : ℕ
  singingInShower : ℕ

/-- The conditions of the problem --/
def problemVillage : Village :=
  { total := 100
  , notWorking := 50
  , withFamily := 25
  , singingInShower := 75 }

/-- The maximum number of people satisfying all conditions --/
def maxSatisfyingAll (v : Village) : ℕ :=
  min (v.total - v.notWorking) (min (v.total - v.withFamily) v.singingInShower)

/-- Theorem stating the maximum number of people satisfying all conditions --/
theorem max_satisfying_all_is_50 :
  maxSatisfyingAll problemVillage = 50 := by sorry

end max_satisfying_all_is_50_l3224_322456


namespace subset_implies_a_equals_one_l3224_322455

-- Define sets M and N
def M (a : ℝ) : Set ℝ := {3, 2*a}
def N (a : ℝ) : Set ℝ := {a+1, 3}

-- State the theorem
theorem subset_implies_a_equals_one :
  ∀ a : ℝ, M a ⊆ N a → a = 1 := by
  sorry

end subset_implies_a_equals_one_l3224_322455


namespace quarters_for_mowing_lawns_l3224_322454

def penny_value : ℚ := 1 / 100
def quarter_value : ℚ := 25 / 100

def pennies : ℕ := 9
def total_amount : ℚ := 184 / 100

theorem quarters_for_mowing_lawns :
  (total_amount - pennies * penny_value) / quarter_value = 7 := by sorry

end quarters_for_mowing_lawns_l3224_322454


namespace last_three_digits_of_7_to_80_l3224_322437

theorem last_three_digits_of_7_to_80 : 7^80 ≡ 961 [ZMOD 1000] := by sorry

end last_three_digits_of_7_to_80_l3224_322437


namespace pradeep_marks_l3224_322477

theorem pradeep_marks (total_marks : ℕ) (pass_percentage : ℚ) (fail_margin : ℕ) : 
  total_marks = 840 → 
  pass_percentage = 1/4 → 
  (total_marks * pass_percentage).floor - fail_margin = 185 :=
by sorry

end pradeep_marks_l3224_322477
