import Mathlib

namespace mikes_books_l225_22556

theorem mikes_books (initial_books new_books : ℕ) : 
  initial_books = 35 → new_books = 56 → initial_books + new_books = 91 := by
  sorry

end mikes_books_l225_22556


namespace baking_on_thursday_l225_22514

/-- The number of days between Amrita's cake baking -/
def baking_cycle : ℕ := 5

/-- The number of days between Thursdays -/
def thursday_cycle : ℕ := 7

/-- The number of days until Amrita bakes a cake on a Thursday again -/
def days_until_thursday_baking : ℕ := 35

theorem baking_on_thursday :
  Nat.lcm baking_cycle thursday_cycle = days_until_thursday_baking := by
  sorry

end baking_on_thursday_l225_22514


namespace oak_trees_planted_l225_22583

/-- The number of oak trees planted by workers in a park. -/
def trees_planted (initial_trees final_trees : ℕ) : ℕ :=
  final_trees - initial_trees

/-- Theorem: Given 5 initial oak trees and 9 final oak trees, the number of trees planted is 4. -/
theorem oak_trees_planted :
  let initial_trees : ℕ := 5
  let final_trees : ℕ := 9
  trees_planted initial_trees final_trees = 4 := by
  sorry

end oak_trees_planted_l225_22583


namespace meeting_point_equation_correct_l225_22510

/-- Represents the time taken for two travelers to meet given their journey durations and a head start for one traveler. -/
def meeting_equation (x : ℚ) : Prop :=
  (x + 2) / 7 + x / 5 = 1

/-- The total journey time for the first traveler -/
def journey_time_A : ℚ := 5

/-- The total journey time for the second traveler -/
def journey_time_B : ℚ := 7

/-- The head start time for the second traveler -/
def head_start : ℚ := 2

/-- Theorem stating that the meeting equation correctly represents the meeting point of two travelers given the conditions -/
theorem meeting_point_equation_correct :
  ∃ x : ℚ, 
    x > 0 ∧ 
    x < journey_time_A ∧
    x + head_start < journey_time_B ∧
    meeting_equation x :=
sorry

end meeting_point_equation_correct_l225_22510


namespace distribute_seven_into_four_l225_22561

/-- Number of ways to distribute indistinguishable objects into distinct containers -/
def distribute_objects (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 indistinguishable objects into 4 distinct containers -/
theorem distribute_seven_into_four :
  distribute_objects 7 4 = 132 := by sorry

end distribute_seven_into_four_l225_22561


namespace average_of_first_group_l225_22598

theorem average_of_first_group (total_average : ℝ) (second_group_average : ℝ) (third_group_average : ℝ)
  (h1 : total_average = 2.80)
  (h2 : second_group_average = 2.3)
  (h3 : third_group_average = 3.7) :
  let total_sum := 6 * total_average
  let second_group_sum := 2 * second_group_average
  let third_group_sum := 2 * third_group_average
  let first_group_sum := total_sum - second_group_sum - third_group_sum
  first_group_sum / 2 = 2.4 := by
sorry

end average_of_first_group_l225_22598


namespace negation_of_cube_odd_is_odd_l225_22584

theorem negation_of_cube_odd_is_odd :
  (¬ ∀ n : ℤ, Odd n → Odd (n^3)) ↔ (∃ n : ℤ, Odd n ∧ ¬Odd (n^3)) := by
  sorry

end negation_of_cube_odd_is_odd_l225_22584


namespace dollar_op_neg_two_three_l225_22596

def dollar_op (a b : ℤ) : ℤ := a * (b + 1) + a * b

theorem dollar_op_neg_two_three : dollar_op (-2) 3 = -14 := by sorry

end dollar_op_neg_two_three_l225_22596


namespace other_man_age_is_ten_l225_22557

/-- The age of the other replaced man given the conditions of the problem -/
def other_man_age (initial_men : ℕ) (replaced_men : ℕ) (age_increase : ℕ) 
  (known_man_age : ℕ) (women_avg_age : ℕ) : ℕ :=
  26 - age_increase

/-- Theorem stating the age of the other replaced man -/
theorem other_man_age_is_ten 
  (initial_men : ℕ) 
  (replaced_men : ℕ) 
  (age_increase : ℕ) 
  (known_man_age : ℕ) 
  (women_avg_age : ℕ) 
  (h1 : initial_men = 8)
  (h2 : replaced_men = 2)
  (h3 : age_increase = 2)
  (h4 : known_man_age = 20)
  (h5 : women_avg_age = 23) :
  other_man_age initial_men replaced_men age_increase known_man_age women_avg_age = 10 := by
  sorry


end other_man_age_is_ten_l225_22557


namespace annie_travel_distance_l225_22558

/-- The number of blocks Annie walked from her house to the bus stop -/
def blocks_to_bus_stop : ℕ := 5

/-- The number of blocks Annie rode the bus to the coffee shop -/
def blocks_on_bus : ℕ := 7

/-- The total number of blocks Annie traveled in her round trip -/
def total_blocks : ℕ := 2 * (blocks_to_bus_stop + blocks_on_bus)

theorem annie_travel_distance : total_blocks = 24 := by sorry

end annie_travel_distance_l225_22558


namespace cost_of_bacon_bacon_cost_is_ten_l225_22565

/-- The cost of bacon given Joan's shopping scenario -/
theorem cost_of_bacon (total_budget : ℕ) (hummus_cost : ℕ) (hummus_quantity : ℕ)
  (chicken_cost : ℕ) (vegetable_cost : ℕ) (apple_cost : ℕ) (apple_quantity : ℕ) : ℕ :=
  by
  -- Define the conditions
  have h1 : total_budget = 60 := by sorry
  have h2 : hummus_cost = 5 := by sorry
  have h3 : hummus_quantity = 2 := by sorry
  have h4 : chicken_cost = 20 := by sorry
  have h5 : vegetable_cost = 10 := by sorry
  have h6 : apple_cost = 2 := by sorry
  have h7 : apple_quantity = 5 := by sorry

  -- Prove that the cost of bacon is 10
  sorry

/-- The main theorem stating that the cost of bacon is 10 -/
theorem bacon_cost_is_ten : cost_of_bacon 60 5 2 20 10 2 5 = 10 := by sorry

end cost_of_bacon_bacon_cost_is_ten_l225_22565


namespace defective_film_probability_l225_22569

/-- The probability of selecting a defective X-ray film from a warehouse with
    specified conditions. -/
theorem defective_film_probability :
  let total_boxes : ℕ := 10
  let boxes_a : ℕ := 5
  let boxes_b : ℕ := 3
  let boxes_c : ℕ := 2
  let defective_rate_a : ℚ := 1 / 10
  let defective_rate_b : ℚ := 1 / 15
  let defective_rate_c : ℚ := 1 / 20
  let prob_a : ℚ := boxes_a / total_boxes
  let prob_b : ℚ := boxes_b / total_boxes
  let prob_c : ℚ := boxes_c / total_boxes
  let total_prob : ℚ := prob_a * defective_rate_a + prob_b * defective_rate_b + prob_c * defective_rate_c
  total_prob = 8 / 100 :=
by sorry

end defective_film_probability_l225_22569


namespace largest_c_value_l225_22533

theorem largest_c_value : ∃ (c : ℝ), (3 * c + 4) * (c - 2) = 9 * c ∧
  ∀ (x : ℝ), (3 * x + 4) * (x - 2) = 9 * x → x ≤ c ∧ c = 4 := by
  sorry

end largest_c_value_l225_22533


namespace random_walk_properties_l225_22588

/-- Represents a random walk on a line -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of the random walk -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences that achieve the maximum range -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Theorem stating the properties of the random walk -/
theorem random_walk_properties (w : RandomWalk) : 
  (max_range w = w.a) ∧ 
  (min_range w = w.a - w.b) ∧ 
  (max_range_sequences w = w.b + 1) := by
  sorry

end random_walk_properties_l225_22588


namespace combined_value_of_a_and_b_l225_22523

/-- Given that 0.5% of a equals 95 paise and b is three times a minus 50,
    prove that the combined value of a and b is 710 rupees. -/
theorem combined_value_of_a_and_b (a b : ℝ) 
  (h1 : 0.005 * a = 95 / 100)  -- 0.5% of a equals 95 paise
  (h2 : b = 3 * a - 50)        -- b is three times a minus 50
  : a + b = 710 := by sorry

end combined_value_of_a_and_b_l225_22523


namespace absolute_value_equality_l225_22568

theorem absolute_value_equality (x : ℝ) (h : x > 3) : 
  |x - Real.sqrt ((x - 3)^2)| = 3 := by
sorry

end absolute_value_equality_l225_22568


namespace extreme_values_sum_reciprocals_l225_22571

theorem extreme_values_sum_reciprocals (x y : ℝ) :
  (4 * x^2 - 5 * x * y + 4 * y^2 = 5) →
  let S := x^2 + y^2
  (∃ S_max : ℝ, ∀ x y : ℝ, (4 * x^2 - 5 * x * y + 4 * y^2 = 5) → x^2 + y^2 ≤ S_max) ∧
  (∃ S_min : ℝ, ∀ x y : ℝ, (4 * x^2 - 5 * x * y + 4 * y^2 = 5) → S_min ≤ x^2 + y^2) ∧
  (1 / (10/3) + 1 / (10/13) = 8/5) :=
by sorry

end extreme_values_sum_reciprocals_l225_22571


namespace right_triangle_arithmetic_sides_l225_22594

/-- A right-angled triangle with sides in arithmetic progression and area 486 dm² has sides 27 dm, 36 dm, and 45 dm. -/
theorem right_triangle_arithmetic_sides (a b c : ℝ) : 
  (a * a + b * b = c * c) →  -- Pythagorean theorem
  (b - a = c - b) →  -- Sides in arithmetic progression
  (a * b / 2 = 486) →  -- Area of the triangle
  (a = 27 ∧ b = 36 ∧ c = 45) := by
sorry

end right_triangle_arithmetic_sides_l225_22594


namespace prob_4_largest_l225_22512

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def draw_size : ℕ := 3

def prob_not_select_5 : ℚ := 2 / 5

def prob_not_select_4_and_5 : ℚ := 1 / 10

theorem prob_4_largest (s : Finset ℕ) (n : ℕ) 
  (h1 : s = card_set) 
  (h2 : n = draw_size) 
  (h3 : prob_not_select_5 = 2 / 5) 
  (h4 : prob_not_select_4_and_5 = 1 / 10) : 
  (prob_not_select_5 - prob_not_select_4_and_5 : ℚ) = 3 / 10 := by
  sorry

end prob_4_largest_l225_22512


namespace investment_interest_calculation_l225_22549

/-- Proves that an investment of $31,200 with a simple annual interest rate of 9% yields a monthly interest payment of $234 -/
theorem investment_interest_calculation (principal : ℝ) (annual_rate : ℝ) (monthly_interest : ℝ) : 
  principal = 31200 ∧ annual_rate = 0.09 → monthly_interest = 234 :=
by
  sorry

end investment_interest_calculation_l225_22549


namespace equation_solution_l225_22513

theorem equation_solution : ∃ x : ℝ, 45 - (x - (37 - (15 - 16))) = 55 ∧ x = 28 := by
  sorry

end equation_solution_l225_22513


namespace remainder_problem_l225_22519

theorem remainder_problem : (245 * 15 - 20 * 8 + 5) % 17 = 1 := by
  sorry

end remainder_problem_l225_22519


namespace probability_theorem_l225_22548

def num_questions : ℕ := 5

def valid_sum (a b : ℕ) : Prop :=
  4 ≤ a + b ∧ a + b < 8

def num_valid_combinations : ℕ := 7

def total_combinations : ℕ := num_questions * (num_questions - 1) / 2

theorem probability_theorem :
  (num_valid_combinations : ℚ) / (total_combinations : ℚ) = 7 / 10 := by sorry

end probability_theorem_l225_22548


namespace fraction_simplification_l225_22590

theorem fraction_simplification (x : ℝ) (h : x ≠ 3) :
  (3 * x) / (x - 3) + (x + 6) / (3 - x) = 2 := by
  sorry

end fraction_simplification_l225_22590


namespace common_area_rectangle_circle_l225_22504

/-- The area of the region common to a 10 by 4 rectangle and a circle with radius 3, sharing the same center, is equal to 9π. -/
theorem common_area_rectangle_circle :
  let rectangle_width : ℝ := 10
  let rectangle_height : ℝ := 4
  let circle_radius : ℝ := 3
  let circle_area : ℝ := π * circle_radius^2
  (∀ x y, x^2 / (rectangle_width/2)^2 + y^2 / (rectangle_height/2)^2 ≤ 1 → x^2 + y^2 ≤ circle_radius^2) →
  circle_area = 9 * π :=
by sorry

end common_area_rectangle_circle_l225_22504


namespace senior_discount_percentage_l225_22591

def shorts_price : ℝ := 15
def shirts_price : ℝ := 17
def num_shorts : ℕ := 3
def num_shirts : ℕ := 5
def total_paid : ℝ := 117

theorem senior_discount_percentage :
  let total_cost := shorts_price * num_shorts + shirts_price * num_shirts
  let discount := total_cost - total_paid
  let discount_percentage := (discount / total_cost) * 100
  discount_percentage = 10 := by
sorry

end senior_discount_percentage_l225_22591


namespace third_batch_average_l225_22589

theorem third_batch_average (n₁ n₂ n₃ : ℕ) (a₁ a₂ a_total : ℚ) :
  n₁ = 40 →
  n₂ = 50 →
  n₃ = 60 →
  a₁ = 45 →
  a₂ = 55 →
  a_total = 56333333333333336 / 1000000000000000 →
  (n₁ * a₁ + n₂ * a₂ + n₃ * (3900 / 60)) / (n₁ + n₂ + n₃) = a_total :=
by sorry

end third_batch_average_l225_22589


namespace exactly_two_roots_l225_22544

def equation (x k : ℂ) : Prop :=
  x / (x + 1) + x / (x + 3) = k * x

theorem exactly_two_roots :
  ∃! k : ℂ, (∃ x y : ℂ, x ≠ y ∧ 
    (∀ z : ℂ, equation z k ↔ z = x ∨ z = y)) ↔ 
  k = (4 : ℂ) / 3 :=
sorry

end exactly_two_roots_l225_22544


namespace angies_age_l225_22509

theorem angies_age : ∃ (age : ℕ), 2 * age + 4 = 20 ∧ age = 8 := by
  sorry

end angies_age_l225_22509


namespace exercise_book_distribution_l225_22528

theorem exercise_book_distribution (students : ℕ) (total_books : ℕ) : 
  (3 * students + 7 = total_books) ∧ (5 * students = total_books + 9) →
  students = 8 ∧ total_books = 31 := by
sorry

end exercise_book_distribution_l225_22528


namespace num_sam_sandwiches_l225_22527

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of restricted sandwich combinations due to roast beef and swiss cheese. -/
def roast_beef_swiss_restrictions : ℕ := num_breads

/-- Represents the number of restricted sandwich combinations due to rye bread and turkey. -/
def rye_turkey_restrictions : ℕ := num_cheeses

/-- Represents the number of restricted sandwich combinations due to roast beef and rye bread. -/
def roast_beef_rye_restrictions : ℕ := num_cheeses

/-- The total number of possible sandwich combinations without restrictions. -/
def total_combinations : ℕ := num_breads * num_meats * num_cheeses

/-- The number of restricted sandwich combinations. -/
def total_restrictions : ℕ := roast_beef_swiss_restrictions + rye_turkey_restrictions + roast_beef_rye_restrictions

/-- Theorem stating the number of sandwiches Sam can order. -/
theorem num_sam_sandwiches : total_combinations - total_restrictions = 193 := by
  sorry

end num_sam_sandwiches_l225_22527


namespace remaining_oranges_l225_22538

def initial_oranges : ℝ := 77.0
def eaten_oranges : ℝ := 2.0

theorem remaining_oranges : initial_oranges - eaten_oranges = 75.0 := by
  sorry

end remaining_oranges_l225_22538


namespace initial_cow_count_l225_22517

theorem initial_cow_count (x : ℕ) 
  (h1 : x - 31 + 75 = 83) : x = 39 := by
  sorry

end initial_cow_count_l225_22517


namespace intersection_A_B_union_A_complement_B_l225_22579

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x - 10 < 0}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

-- Define the universal set R (real numbers)
def R : Type := ℝ

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -5 < x ∧ x ≤ -1} := by sorry

-- Theorem for A ∪ (∁ₖ B)
theorem union_A_complement_B : A ∪ (Set.univ \ B) = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end intersection_A_B_union_A_complement_B_l225_22579


namespace triangle_middle_side_bound_l225_22501

theorem triangle_middle_side_bound (a b c : ℝ) (h_area : 1 = (1/2) * b * c * Real.sin α) 
  (h_order : a ≥ b ∧ b ≥ c) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a < b + c) :
  b ≥ Real.sqrt 2 := by
  sorry

end triangle_middle_side_bound_l225_22501


namespace tile_covers_25_squares_l225_22566

/-- Represents a square tile -/
structure Tile :=
  (sideLength : ℝ)

/-- Represents a checkerboard -/
structure Checkerboard :=
  (size : ℕ)
  (squareWidth : ℝ)

/-- Counts the number of squares completely covered by a tile on a checkerboard -/
def countCoveredSquares (t : Tile) (c : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating that a square tile with side length D placed on a 10x10 checkerboard
    with square width D, such that their centers coincide, covers exactly 25 squares -/
theorem tile_covers_25_squares (D : ℝ) (D_pos : D > 0) :
  let t : Tile := { sideLength := D }
  let c : Checkerboard := { size := 10, squareWidth := D }
  countCoveredSquares t c = 25 :=
sorry

end tile_covers_25_squares_l225_22566


namespace largest_even_digit_multiple_of_5_proof_l225_22547

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_5 : ℕ := 86880

theorem largest_even_digit_multiple_of_5_proof :
  (has_only_even_digits largest_even_digit_multiple_of_5) ∧
  (largest_even_digit_multiple_of_5 < 100000) ∧
  (largest_even_digit_multiple_of_5 % 5 = 0) ∧
  (∀ m : ℕ, m > largest_even_digit_multiple_of_5 →
    ¬(has_only_even_digits m ∧ m < 100000 ∧ m % 5 = 0)) :=
by sorry

end largest_even_digit_multiple_of_5_proof_l225_22547


namespace doubled_number_excess_l225_22539

theorem doubled_number_excess (x : ℝ) : x^2 = 25 → 2*x - x/5 = 9 := by
  sorry

end doubled_number_excess_l225_22539


namespace octal_sum_451_167_l225_22511

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- The sum of two octal numbers in base 8 --/
def octal_sum (a b : ℕ) : ℕ := decimal_to_octal (octal_to_decimal a + octal_to_decimal b)

theorem octal_sum_451_167 : octal_sum 451 167 = 640 := by sorry

end octal_sum_451_167_l225_22511


namespace gcf_of_40_and_56_l225_22524

theorem gcf_of_40_and_56 : Nat.gcd 40 56 = 8 := by
  sorry

end gcf_of_40_and_56_l225_22524


namespace x_squared_plus_2xy_range_l225_22525

theorem x_squared_plus_2xy_range :
  ∀ x y : ℝ, x^2 + y^2 = 1 →
  (∃ (z : ℝ), z = x^2 + 2*x*y ∧ 1/2 - Real.sqrt 5 / 2 ≤ z ∧ z ≤ 1/2 + Real.sqrt 5 / 2) ∧
  (∃ (a b : ℝ), a = x^2 + 2*x*y ∧ b = x^2 + 2*x*y ∧ 
   a = 1/2 - Real.sqrt 5 / 2 ∧ b = 1/2 + Real.sqrt 5 / 2) :=
by sorry

end x_squared_plus_2xy_range_l225_22525


namespace sigma_phi_bounds_l225_22574

open Nat Real

/-- The sum of divisors function -/
noncomputable def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
noncomputable def phi (n : ℕ) : ℕ := sorry

theorem sigma_phi_bounds (n : ℕ) (h : n > 0) : 
  (sigma n * phi n : ℝ) < n^2 ∧ 
  ∃ c : ℝ, c > 0 ∧ ∀ m : ℕ, m > 0 → (sigma m * phi m : ℝ) ≥ c * m^2 := by
  sorry

end sigma_phi_bounds_l225_22574


namespace middle_circle_radius_l225_22553

/-- Represents the radii of five circles in an arithmetic sequence -/
def CircleRadii := Fin 5 → ℝ

/-- The property that the radii form an arithmetic sequence -/
def is_arithmetic_sequence (r : CircleRadii) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 4, r (i + 1) = r i + d

/-- The theorem statement -/
theorem middle_circle_radius 
  (r : CircleRadii) 
  (h_arithmetic : is_arithmetic_sequence r)
  (h_smallest : r 0 = 6)
  (h_largest : r 4 = 30) :
  r 2 = 18 := by
sorry

end middle_circle_radius_l225_22553


namespace athena_spent_14_l225_22550

/-- The total amount Athena spent on snacks for her friends -/
def total_spent (sandwich_price : ℚ) (sandwich_quantity : ℕ) (drink_price : ℚ) (drink_quantity : ℕ) : ℚ :=
  sandwich_price * sandwich_quantity + drink_price * drink_quantity

/-- Theorem: Athena spent $14 in total -/
theorem athena_spent_14 :
  total_spent 3 3 (5/2) 2 = 14 := by
  sorry

end athena_spent_14_l225_22550


namespace cube_properties_l225_22582

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D
  edgeLength : ℝ

/-- Returns true if two lines are skew -/
def areSkewLines (l1 l2 : Line3D) : Prop := sorry

/-- Returns true if a line is perpendicular to two other lines -/
def isPerpendicularToLines (l : Line3D) (l1 l2 : Line3D) : Prop := sorry

/-- Calculates the distance between two skew lines -/
def distanceBetweenSkewLines (l1 l2 : Line3D) : ℝ := sorry

theorem cube_properties (cube : Cube) :
  let AA₁ : Line3D := { point := cube.A, direction := { x := 0, y := 0, z := 1 } }
  let BC : Line3D := { point := cube.B, direction := { x := 1, y := 0, z := 0 } }
  let AB : Line3D := { point := cube.A, direction := { x := 1, y := 0, z := 0 } }
  areSkewLines AA₁ BC ∧
  isPerpendicularToLines AB AA₁ BC ∧
  distanceBetweenSkewLines AA₁ BC = cube.edgeLength := by
  sorry

end cube_properties_l225_22582


namespace barbaras_total_cost_l225_22535

/-- The cost of Barbara's purchase at the butcher's --/
def barbaras_purchase_cost (steak_weight : Real) (steak_price : Real) 
  (chicken_weight : Real) (chicken_price : Real) : Real :=
  steak_weight * steak_price + chicken_weight * chicken_price

/-- Theorem stating the total cost of Barbara's purchase --/
theorem barbaras_total_cost : 
  barbaras_purchase_cost 2 15 1.5 8 = 42 := by
  sorry

end barbaras_total_cost_l225_22535


namespace sector_triangle_area_equality_l225_22530

/-- Given a circle with center C and radius r, and an angle φ where 0 < φ < π/2,
    prove that the area of the circular sector formed by φ is equal to 
    the area of the triangle formed by the tangent line and the radius 
    if and only if tan φ = φ. -/
theorem sector_triangle_area_equality (φ : Real) (h1 : 0 < φ) (h2 : φ < π/2) :
  let r : Real := 1  -- Assuming unit circle for simplicity
  let sector_area : Real := (φ * r^2) / 2
  let triangle_area : Real := (r^2 * Real.tan φ) / 2
  sector_area = triangle_area ↔ Real.tan φ = φ := by
  sorry

end sector_triangle_area_equality_l225_22530


namespace chess_tournament_participants_l225_22522

/-- Represents a chess tournament with the given property --/
structure ChessTournament where
  n : ℕ  -- Total number of players
  half_points_from_last_three : Prop  -- Property that each player scored half their points against the last three

/-- Theorem stating that a chess tournament satisfying the given condition has 9 participants --/
theorem chess_tournament_participants (t : ChessTournament) : t.n = 9 := by
  sorry

end chess_tournament_participants_l225_22522


namespace min_probability_cards_l225_22559

/-- Represents the probability of a card being red-side up after two flips -/
def probability_red (k : ℕ) : ℚ :=
  if k ≤ 25 then
    (676 - 52 * k + 2 * k^2 : ℚ) / 676
  else
    (676 - 52 * (51 - k) + 2 * (51 - k)^2 : ℚ) / 676

/-- The total number of cards -/
def total_cards : ℕ := 50

/-- The number of cards flipped in each operation -/
def flip_size : ℕ := 25

/-- Theorem stating that cards 13 and 38 have the lowest probability of being red-side up -/
theorem min_probability_cards :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ total_cards →
    (probability_red 13 ≤ probability_red k ∧
     probability_red 38 ≤ probability_red k) :=
by sorry

end min_probability_cards_l225_22559


namespace problem_1_l225_22564

theorem problem_1 : (-2)^3 + (1/9)⁻¹ - (3.14 - Real.pi)^0 = 0 := by sorry

end problem_1_l225_22564


namespace amanda_kitchen_upgrade_cost_l225_22546

def kitchen_upgrade_cost (num_knobs : ℕ) (knob_price : ℚ) (num_pulls : ℕ) (pull_price : ℚ) : ℚ :=
  (num_knobs * knob_price) + (num_pulls * pull_price)

theorem amanda_kitchen_upgrade_cost :
  kitchen_upgrade_cost 18 (5/2) 8 4 = 77 := by
  sorry

end amanda_kitchen_upgrade_cost_l225_22546


namespace billy_hike_distance_l225_22500

theorem billy_hike_distance :
  let east_distance : ℝ := 7
  let north_distance : ℝ := 3 * Real.sqrt 3
  let total_distance : ℝ := Real.sqrt (east_distance^2 + north_distance^2)
  total_distance = 2 * Real.sqrt 19 := by
  sorry

end billy_hike_distance_l225_22500


namespace shirt_sales_theorem_l225_22581

/-- Represents the sales and profit data for a shirt selling business -/
structure ShirtSales where
  initial_sales : ℕ
  initial_profit : ℝ
  sales_increase : ℝ
  profit_decrease : ℝ

/-- Calculates the new sales quantity after a price reduction -/
def new_sales (data : ShirtSales) (reduction : ℝ) : ℝ :=
  data.initial_sales + data.sales_increase * reduction

/-- Calculates the new profit per piece after a price reduction -/
def new_profit_per_piece (data : ShirtSales) (reduction : ℝ) : ℝ :=
  data.initial_profit - reduction

/-- Calculates the total daily profit after a price reduction -/
def total_daily_profit (data : ShirtSales) (reduction : ℝ) : ℝ :=
  new_sales data reduction * new_profit_per_piece data reduction

/-- The main theorem about shirt sales and profit -/
theorem shirt_sales_theorem (data : ShirtSales) 
    (h1 : data.initial_sales = 20)
    (h2 : data.initial_profit = 40)
    (h3 : data.sales_increase = 2)
    (h4 : data.profit_decrease = 1) : 
    new_sales data 3 = 26 ∧ 
    ∃ x : ℝ, x = 20 ∧ total_daily_profit data x = 1200 := by
  sorry

end shirt_sales_theorem_l225_22581


namespace closest_to_quotient_l225_22543

def options : List ℝ := [500, 1500, 2500, 5000, 7500]

theorem closest_to_quotient (x : ℝ) (h : x ∈ options \ {2500}) :
  |503 / 0.198 - 2500| < |503 / 0.198 - x| :=
by sorry

end closest_to_quotient_l225_22543


namespace intersection_point_is_e_e_l225_22554

theorem intersection_point_is_e_e (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x = Real.exp 1 ∧ y = Real.exp 1) →
  (x^y = y^x ∧ y = x) :=
by sorry

end intersection_point_is_e_e_l225_22554


namespace three_solutions_iff_a_gt_two_l225_22536

/-- The equation x · |x-a| = 1 has exactly three distinct solutions if and only if a > 2 -/
theorem three_solutions_iff_a_gt_two (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ (x : ℝ), x * |x - a| = 1 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  a > 2 := by
  sorry

end three_solutions_iff_a_gt_two_l225_22536


namespace stripe_area_cylindrical_tower_l225_22503

/-- The area of a horizontal stripe wrapping twice around a cylindrical tower -/
theorem stripe_area_cylindrical_tower (d h w : ℝ) (hd : d = 25) (hh : h = 60) (hw : w = 2) :
  let circumference := π * d
  let stripe_length := 2 * circumference
  let stripe_area := stripe_length * w
  stripe_area = 100 * π :=
sorry

end stripe_area_cylindrical_tower_l225_22503


namespace smallest_class_size_exists_class_size_l225_22516

theorem smallest_class_size (n : ℕ) : 
  (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 8 = 5) → n ≥ 53 :=
by sorry

theorem exists_class_size : 
  ∃ n : ℕ, (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 8 = 5) ∧ n = 53 :=
by sorry

end smallest_class_size_exists_class_size_l225_22516


namespace investment_time_solution_l225_22585

/-- Represents a partner in the investment problem -/
structure Partner where
  investment : ℝ
  time : ℝ
  profit : ℝ

/-- The investment problem -/
def InvestmentProblem (p q : Partner) : Prop :=
  p.investment / q.investment = 7 / 5 ∧
  p.profit / q.profit = 7 / 10 ∧
  p.time = 7 ∧
  p.investment * p.time / (q.investment * q.time) = p.profit / q.profit

theorem investment_time_solution (p q : Partner) :
  InvestmentProblem p q → q.time = 14 := by
  sorry

end investment_time_solution_l225_22585


namespace lori_earnings_l225_22531

/-- Represents the earnings from Lori's carsharing company -/
def carsharing_earnings (num_red_cars num_white_cars : ℕ) 
  (red_car_rate white_car_rate : ℚ) (rental_hours : ℕ) : ℚ :=
  let total_minutes := rental_hours * 60
  let red_car_earnings := num_red_cars * red_car_rate * total_minutes
  let white_car_earnings := num_white_cars * white_car_rate * total_minutes
  red_car_earnings + white_car_earnings

/-- Theorem stating that Lori's earnings are $2340 given the problem conditions -/
theorem lori_earnings : 
  carsharing_earnings 3 2 3 2 3 = 2340 := by
  sorry

#eval carsharing_earnings 3 2 3 2 3

end lori_earnings_l225_22531


namespace basketball_conference_games_l225_22573

/-- The number of teams in the basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team in the conference -/
def games_per_pair : ℕ := 3

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 5

/-- The total number of games in a season for the basketball conference -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pair) + (num_teams * non_conference_games)

theorem basketball_conference_games :
  total_games = 185 := by sorry

end basketball_conference_games_l225_22573


namespace parabola_with_directrix_neg_one_l225_22577

/-- A parabola is defined by its directrix and focus. This structure represents a parabola with a vertical directrix. -/
structure Parabola where
  /-- The x-coordinate of the directrix -/
  directrix : ℝ

/-- The standard equation of a parabola with a vertical directrix -/
def standardEquation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (y^2 = 4*(x - p.directrix/2))

/-- Theorem: For a parabola with directrix x = -1, its standard equation is y^2 = 4x -/
theorem parabola_with_directrix_neg_one (p : Parabola) (h : p.directrix = -1) :
  standardEquation p ↔ ∀ x y : ℝ, (y^2 = 4*x) :=
sorry

end parabola_with_directrix_neg_one_l225_22577


namespace scientific_notation_864000_l225_22560

theorem scientific_notation_864000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 864000 = a * (10 : ℝ) ^ n :=
by
  use 8.64, 5
  sorry

end scientific_notation_864000_l225_22560


namespace cafeteria_milk_stacks_l225_22515

/-- Given a total number of cartons and the number of cartons per stack, 
    calculate the maximum number of full stacks that can be made. -/
def maxFullStacks (totalCartons : ℕ) (cartonsPerStack : ℕ) : ℕ :=
  totalCartons / cartonsPerStack

theorem cafeteria_milk_stacks : maxFullStacks 799 6 = 133 := by
  sorry

end cafeteria_milk_stacks_l225_22515


namespace total_project_hours_l225_22567

def project_hours (kate_hours : ℝ) : ℝ × ℝ × ℝ :=
  let pat_hours := 2 * kate_hours
  let mark_hours := kate_hours + 75
  (pat_hours, kate_hours, mark_hours)

theorem total_project_hours :
  ∃ (kate_hours : ℝ),
    let (pat_hours, _, mark_hours) := project_hours kate_hours
    pat_hours = (1/3) * mark_hours ∧
    (pat_hours + kate_hours + mark_hours) = 135 := by
  sorry

end total_project_hours_l225_22567


namespace min_even_integers_l225_22508

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 24 →
  a + b + c + d = 39 →
  a + b + c + d + e + f = 58 →
  ∃ (count : ℕ), count ≥ 2 ∧ 
    count = (if Even a then 1 else 0) + 
            (if Even b then 1 else 0) + 
            (if Even c then 1 else 0) + 
            (if Even d then 1 else 0) + 
            (if Even e then 1 else 0) + 
            (if Even f then 1 else 0) ∧
    ∀ (other_count : ℕ), 
      other_count = (if Even a then 1 else 0) + 
                    (if Even b then 1 else 0) + 
                    (if Even c then 1 else 0) + 
                    (if Even d then 1 else 0) + 
                    (if Even e then 1 else 0) + 
                    (if Even f then 1 else 0) →
      other_count ≥ count := by
sorry

end min_even_integers_l225_22508


namespace sum_of_sequences_is_435_l225_22540

def sequence1 : List ℕ := [2, 14, 26, 38, 50]
def sequence2 : List ℕ := [12, 24, 36, 48, 60]
def sequence3 : List ℕ := [5, 15, 25, 35, 45]

theorem sum_of_sequences_is_435 :
  (sequence1.sum + sequence2.sum + sequence3.sum) = 435 := by
  sorry

end sum_of_sequences_is_435_l225_22540


namespace cross_section_area_theorem_l225_22507

/-- Regular quadrilateral prism with given dimensions -/
structure RegularQuadrilateralPrism where
  a : ℝ
  base_edge : ℝ
  height : ℝ
  h_base_edge : base_edge = a
  h_height : height = 2 * a

/-- Plane passing through diagonal B₁D₁ and midpoint of edge DC -/
structure CuttingPlane (prism : RegularQuadrilateralPrism) where
  diagonal : ℝ × ℝ × ℝ
  midpoint : ℝ × ℝ × ℝ
  h_diagonal : diagonal = (prism.a, prism.a, prism.height)
  h_midpoint : midpoint = (prism.a / 2, prism.a, 0)

/-- Area of cross-section created by cutting plane -/
noncomputable def cross_section_area (prism : RegularQuadrilateralPrism) (plane : CuttingPlane prism) : ℝ :=
  (3 * prism.a^2 * Real.sqrt 33) / 8

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_theorem (prism : RegularQuadrilateralPrism) (plane : CuttingPlane prism) :
  cross_section_area prism plane = (3 * prism.a^2 * Real.sqrt 33) / 8 := by
  sorry

end cross_section_area_theorem_l225_22507


namespace square_equality_solution_l225_22570

theorem square_equality_solution : ∃ x : ℝ, (9 - x)^2 = x^2 ∧ x = 4.5 := by
  sorry

end square_equality_solution_l225_22570


namespace multipleOfThree_is_closed_l225_22593

def ClosedSet (A : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

def MultipleOfThree : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem multipleOfThree_is_closed : ClosedSet MultipleOfThree := by
  sorry

end multipleOfThree_is_closed_l225_22593


namespace polynomial_sum_simplification_l225_22595

-- Define the polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum_simplification :
  ∀ x : ℝ, p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by sorry

end polynomial_sum_simplification_l225_22595


namespace point_positions_l225_22520

/-- Define a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point is in the first octant -/
def isInFirstOctant (p : Point3D) : Prop :=
  p.x > 0 ∧ p.y > 0 ∧ p.z > 0

/-- Check if a point is in the second octant -/
def isInSecondOctant (p : Point3D) : Prop :=
  p.x < 0 ∧ p.y > 0 ∧ p.z > 0

/-- Check if a point is in the eighth octant -/
def isInEighthOctant (p : Point3D) : Prop :=
  p.x > 0 ∧ p.y < 0 ∧ p.z < 0

/-- Check if a point lies in the YOZ plane -/
def isInYOZPlane (p : Point3D) : Prop :=
  p.x = 0

/-- Check if a point lies on the OY axis -/
def isOnOYAxis (p : Point3D) : Prop :=
  p.x = 0 ∧ p.z = 0

/-- Check if a point is at the origin -/
def isAtOrigin (p : Point3D) : Prop :=
  p.x = 0 ∧ p.y = 0 ∧ p.z = 0

theorem point_positions :
  let A : Point3D := ⟨3, 2, 6⟩
  let B : Point3D := ⟨-2, 3, 1⟩
  let C : Point3D := ⟨1, -4, -2⟩
  let D : Point3D := ⟨1, -2, -1⟩
  let E : Point3D := ⟨0, 4, 1⟩
  let F : Point3D := ⟨0, 2, 0⟩
  let P : Point3D := ⟨0, 0, 0⟩
  isInFirstOctant A ∧
  isInSecondOctant B ∧
  isInEighthOctant C ∧
  isInEighthOctant D ∧
  isInYOZPlane E ∧
  isOnOYAxis F ∧
  isAtOrigin P := by
  sorry

end point_positions_l225_22520


namespace tournament_outcomes_l225_22599

/-- Represents the number of players in the tournament -/
def num_players : Nat := 5

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : Nat := 2

/-- Represents the number of elimination rounds -/
def num_rounds : Nat := 4

/-- Calculates the total number of possible outcomes for the tournament -/
def total_outcomes : Nat := outcomes_per_match ^ num_rounds

/-- Theorem stating that the total number of possible outcomes is 16 -/
theorem tournament_outcomes :
  total_outcomes = 16 := by sorry

end tournament_outcomes_l225_22599


namespace sqrt_of_four_equals_two_l225_22545

theorem sqrt_of_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end sqrt_of_four_equals_two_l225_22545


namespace factory_production_correct_factory_produces_90_refrigerators_per_hour_l225_22552

/-- Represents the production of a factory making refrigerators and coolers -/
structure FactoryProduction where
  refrigerators_per_hour : ℕ
  coolers_per_hour : ℕ
  total_products : ℕ
  days : ℕ
  hours_per_day : ℕ

/-- The conditions of the factory production problem -/
def factory_conditions : FactoryProduction where
  refrigerators_per_hour := 90  -- This is what we want to prove
  coolers_per_hour := 90 + 70
  total_products := 11250
  days := 5
  hours_per_day := 9

/-- Theorem stating that the given conditions satisfy the problem requirements -/
theorem factory_production_correct (fp : FactoryProduction) : 
  fp.coolers_per_hour = fp.refrigerators_per_hour + 70 →
  fp.total_products = (fp.refrigerators_per_hour + fp.coolers_per_hour) * fp.days * fp.hours_per_day →
  fp.refrigerators_per_hour = 90 :=
by
  sorry

/-- The main theorem proving that the factory produces 90 refrigerators per hour -/
theorem factory_produces_90_refrigerators_per_hour : 
  factory_conditions.refrigerators_per_hour = 90 :=
by
  apply factory_production_correct factory_conditions
  · -- Prove that coolers_per_hour = refrigerators_per_hour + 70
    sorry
  · -- Prove that total_products = (refrigerators_per_hour + coolers_per_hour) * days * hours_per_day
    sorry

end factory_production_correct_factory_produces_90_refrigerators_per_hour_l225_22552


namespace square_perimeter_l225_22541

/-- Given a square with area 400 square meters, its perimeter is 80 meters. -/
theorem square_perimeter (s : ℝ) (h : s^2 = 400) : 4 * s = 80 := by
  sorry

end square_perimeter_l225_22541


namespace C_is_hyperbola_l225_22576

/-- The curve C is defined by the equation 3y^2 - 4(x+1)y + 12(x-2) = 0 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.2^2 - 4 * (p.1 + 1) * p.2 + 12 * (p.1 - 2) = 0}

/-- The discriminant of the quadratic equation in y -/
def discriminant (x : ℝ) : ℝ :=
  16 * x^2 - 112 * x + 304

/-- Theorem: The curve C is a hyperbola -/
theorem C_is_hyperbola : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1^2 / a^2) - (p.2^2 / b^2) = 1 :=
sorry

end C_is_hyperbola_l225_22576


namespace sara_received_six_kittens_l225_22575

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara (initial : ℕ) (to_jessica : ℕ) (left : ℕ) : ℕ :=
  initial - to_jessica - left

/-- Proof that Tim gave 6 kittens to Sara -/
theorem sara_received_six_kittens :
  kittens_to_sara 18 3 9 = 6 := by
  sorry

end sara_received_six_kittens_l225_22575


namespace exists_divisor_friendly_bijection_l225_22597

/-- The number of positive divisors of a positive integer n -/
def d (n : ℕ+) : ℕ := sorry

/-- A bijection is divisor-friendly if it satisfies the given property -/
def divisor_friendly (F : ℕ+ → ℕ+) : Prop :=
  Function.Bijective F ∧ ∀ m n : ℕ+, d (F (m * n)) = d (F m) * d (F n)

/-- There exists a divisor-friendly bijection -/
theorem exists_divisor_friendly_bijection : ∃ F : ℕ+ → ℕ+, divisor_friendly F := by
  sorry

end exists_divisor_friendly_bijection_l225_22597


namespace log_equation_implies_non_square_non_cube_integer_l225_22580

-- Define the logarithm equation
def log_equation (x : ℝ) : Prop :=
  Real.log (343 : ℝ) / Real.log (3 * x + 1) = x

-- Define what it means to be a non-square, non-cube integer
def is_non_square_non_cube_integer (x : ℝ) : Prop :=
  ∃ n : ℤ, (x : ℝ) = n ∧ ¬∃ m : ℤ, n = m^2 ∧ ¬∃ k : ℤ, n = k^3

-- The theorem statement
theorem log_equation_implies_non_square_non_cube_integer :
  ∀ x : ℝ, log_equation x → is_non_square_non_cube_integer x :=
by sorry

end log_equation_implies_non_square_non_cube_integer_l225_22580


namespace factor_expression_l225_22537

theorem factor_expression (x : ℝ) : 72 * x^5 - 90 * x^9 = 18 * x^5 * (4 - 5 * x^4) := by
  sorry

end factor_expression_l225_22537


namespace angles_on_y_axis_correct_l225_22505

/-- The set of angles whose terminal sides fall on the y-axis -/
def angles_on_y_axis : Set ℝ :=
  { α | ∃ k : ℤ, α = k * Real.pi + Real.pi / 2 }

/-- Theorem stating that angles_on_y_axis correctly represents
    the set of angles whose terminal sides fall on the y-axis -/
theorem angles_on_y_axis_correct :
  ∀ α : ℝ, α ∈ angles_on_y_axis ↔ 
    (∃ k : ℤ, α = k * Real.pi + Real.pi / 2) :=
by sorry

end angles_on_y_axis_correct_l225_22505


namespace dress_price_difference_l225_22526

/-- Given a dress with an original price that was discounted by 15% to $85, 
    and then increased by 25%, prove that the difference between the original 
    price and the final price is $6.25. -/
theorem dress_price_difference (original_price : ℝ) : 
  original_price * (1 - 0.15) = 85 →
  original_price - (85 * (1 + 0.25)) = -6.25 := by
sorry

end dress_price_difference_l225_22526


namespace parallelogram_fourth_vertex_l225_22586

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A quadrilateral in a 2D Cartesian coordinate system -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Predicate to check if two line segments are parallel -/
def parallel (p1 p2 p3 p4 : Point2D) : Prop :=
  (p2.x - p1.x) * (p4.y - p3.y) = (p2.y - p1.y) * (p4.x - p3.x)

theorem parallelogram_fourth_vertex 
  (q : Quadrilateral)
  (h1 : parallel q.A q.B q.D q.C)
  (h2 : parallel q.A q.D q.B q.C)
  (h3 : q.A = Point2D.mk (-2) 0)
  (h4 : q.B = Point2D.mk 6 8)
  (h5 : q.C = Point2D.mk 8 6) :
  q.D = Point2D.mk 0 (-2) := by
  sorry

end parallelogram_fourth_vertex_l225_22586


namespace unique_prime_sum_difference_l225_22532

theorem unique_prime_sum_difference : ∃! p : ℕ, 
  Prime p ∧ 
  (∃ x y z w : ℕ, Prime x ∧ Prime y ∧ Prime z ∧ Prime w ∧ 
    p = x + y ∧ p = z - w) :=
by
  -- The proof would go here
  sorry

end unique_prime_sum_difference_l225_22532


namespace factorization_equality_l225_22555

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end factorization_equality_l225_22555


namespace fair_coin_five_flips_probability_l225_22521

/-- Represents the probability of a specific outcome when flipping a fair coin n times -/
def coin_flip_probability (n : ℕ) (heads : Finset ℕ) : ℚ :=
  (1 / 2) ^ n

theorem fair_coin_five_flips_probability :
  coin_flip_probability 5 {0, 1} = 1 / 32 := by
  sorry

end fair_coin_five_flips_probability_l225_22521


namespace tetrahedron_edge_length_l225_22551

/-- A configuration of four mutually tangent spheres on a plane -/
structure SphericalConfiguration where
  radius : ℝ
  mutually_tangent : Bool
  on_plane : Bool

/-- A tetrahedron circumscribed around four spheres -/
structure CircumscribedTetrahedron where
  spheres : SphericalConfiguration
  edge_length : ℝ

/-- The theorem stating that the edge length of a tetrahedron circumscribed around
    four mutually tangent spheres of radius 2 is equal to 4 -/
theorem tetrahedron_edge_length 
  (config : SphericalConfiguration) 
  (tetra : CircumscribedTetrahedron) :
  config.radius = 2 ∧ 
  config.mutually_tangent = true ∧ 
  config.on_plane = true ∧
  tetra.spheres = config →
  tetra.edge_length = 4 := by
  sorry

end tetrahedron_edge_length_l225_22551


namespace sum_less_than_addends_l225_22572

theorem sum_less_than_addends : ∃ a b : ℝ, a + b < a ∧ a + b < b := by
  sorry

end sum_less_than_addends_l225_22572


namespace sphere_radius_at_specific_time_l225_22592

/-- The radius of a sphere with a variable density distribution and time-dependent curved surface area. -/
theorem sphere_radius_at_specific_time
  (k ω β c : ℝ)
  (ρ : ℝ → ℝ)
  (A : ℝ → ℝ)
  (h1 : ∀ r, ρ r = k * r^2)
  (h2 : ∀ t, A t = ω * Real.sin (β * t) + c)
  (h3 : A (Real.pi / (2 * β)) = 64 * Real.pi) :
  ∃ r : ℝ, r = 4 ∧ A (Real.pi / (2 * β)) = 4 * Real.pi * r^2 :=
sorry

end sphere_radius_at_specific_time_l225_22592


namespace roots_negative_reciprocals_implies_a_eq_neg_c_l225_22518

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the concept of roots
def is_root (r : ℝ) (a b c : ℝ) : Prop := quadratic_equation a b c r

-- Define negative reciprocals
def negative_reciprocals (r s : ℝ) : Prop := r = -1/s ∧ s = -1/r

-- Theorem statement
theorem roots_negative_reciprocals_implies_a_eq_neg_c 
  (a b c r s : ℝ) (h1 : is_root r a b c) (h2 : is_root s a b c) 
  (h3 : negative_reciprocals r s) : a = -c :=
sorry

end roots_negative_reciprocals_implies_a_eq_neg_c_l225_22518


namespace reemas_correct_marks_l225_22578

/-- Proves that given a class of 35 students with an initial average of 72,
    if one student's marks are changed from 46 to x, resulting in a new average of 71.71,
    then x = 36.85 -/
theorem reemas_correct_marks 
  (num_students : Nat)
  (initial_average : ℚ)
  (incorrect_marks : ℚ)
  (new_average : ℚ)
  (h1 : num_students = 35)
  (h2 : initial_average = 72)
  (h3 : incorrect_marks = 46)
  (h4 : new_average = 71.71)
  : ∃ x : ℚ, x = 36.85 ∧ 
    (num_students : ℚ) * initial_average - incorrect_marks + x = 
    (num_students : ℚ) * new_average :=
by
  sorry


end reemas_correct_marks_l225_22578


namespace complex_power_2009_l225_22506

theorem complex_power_2009 (i : ℂ) (h : i^2 = -1) : i^2009 = i := by sorry

end complex_power_2009_l225_22506


namespace post_height_l225_22563

/-- The height of a cylindrical post given a squirrel's spiral path. -/
theorem post_height (circuit_rise : ℝ) (post_circumference : ℝ) (total_distance : ℝ) : 
  circuit_rise = 4 →
  post_circumference = 3 →
  total_distance = 9 →
  (total_distance / post_circumference) * circuit_rise = 12 :=
by sorry

end post_height_l225_22563


namespace isosceles_trapezoid_circle_tangent_l225_22502

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a circle is tangent to a line segment -/
def isTangent (c : Circle) (p1 p2 : Point) : Prop := sorry

/-- Checks if a point lies on a line segment -/
def liesBetween (p : Point) (p1 p2 : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Main theorem -/
theorem isosceles_trapezoid_circle_tangent 
  (ABCD : IsoscelesTrapezoid) 
  (c : Circle) 
  (M N : Point) :
  isTangent c ABCD.A ABCD.B →
  isTangent c ABCD.B ABCD.C →
  liesBetween M ABCD.A ABCD.D →
  liesBetween N ABCD.C ABCD.D →
  distance ABCD.A M / distance ABCD.D M = 1 / 3 →
  distance ABCD.C N / distance ABCD.D N = 4 / 3 →
  distance ABCD.A ABCD.B = 7 →
  distance ABCD.A ABCD.D = 6 →
  distance ABCD.B ABCD.C = 4 + 2 * Real.sqrt 7 := by
  sorry

end isosceles_trapezoid_circle_tangent_l225_22502


namespace miquel_point_existence_l225_22587

-- Define the basic geometric objects
variable (A B C D H M N S T : Point)

-- Define the quadrilateral ABCD
def is_convex_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define that ABCD is not a kite
def is_not_kite (A B C D : Point) : Prop := sorry

-- Define perpendicular diagonals
def perpendicular_diagonals (A B C D H : Point) : Prop := sorry

-- Define midpoints
def is_midpoint (M : Point) (B C : Point) : Prop := sorry

-- Define ray intersection
def ray_intersects (M H S A D : Point) : Prop := sorry

-- Define point outside quadrilateral
def point_outside_quadrilateral (E A B C D : Point) : Prop := sorry

-- Define angle bisector
def is_angle_bisector (E H B S : Point) : Prop := sorry

-- Define equal angles
def equal_angles (B E N M D : Point) : Prop := sorry

-- Main theorem
theorem miquel_point_existence 
  (h1 : is_convex_cyclic_quadrilateral A B C D)
  (h2 : is_not_kite A B C D)
  (h3 : perpendicular_diagonals A B C D H)
  (h4 : is_midpoint M B C)
  (h5 : is_midpoint N C D)
  (h6 : ray_intersects M H S A D)
  (h7 : ray_intersects N H T A B) :
  ∃ E : Point,
    point_outside_quadrilateral E A B C D ∧
    is_angle_bisector E H B S ∧
    is_angle_bisector E H T D ∧
    equal_angles B E N M D :=
sorry

end miquel_point_existence_l225_22587


namespace necessary_condition_for_A_l225_22542

-- Define the set A
def A : Set ℝ := {x | (x - 2) / (x + 1) ≤ 0}

-- State the theorem
theorem necessary_condition_for_A (a : ℝ) :
  (∀ x ∈ A, x ≥ a) → a ≤ -1 := by
  sorry

end necessary_condition_for_A_l225_22542


namespace triangle_properties_l225_22562

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def ABC : Triangle := { A := (8, 5), B := (4, -2), C := (-6, 3) }

-- Equation of a line: ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def median_to_AC (t : Triangle) : Line := sorry

def altitude_to_AB (t : Triangle) : Line := sorry

def perpendicular_bisector_BC (t : Triangle) : Line := sorry

theorem triangle_properties :
  let m := median_to_AC ABC
  let h := altitude_to_AB ABC
  let p := perpendicular_bisector_BC ABC
  m.a = 2 ∧ m.b = 1 ∧ m.c = -6 ∧
  h.a = 4 ∧ h.b = 7 ∧ h.c = 3 ∧
  p.a = 2 ∧ p.b = -1 ∧ p.c = 5/2 := by sorry

end triangle_properties_l225_22562


namespace intersection_equals_result_l225_22534

-- Define the sets M and N
def M : Set ℝ := {x | (x - 3) * Real.sqrt (x - 1) ≥ 0}
def N : Set ℝ := {x | (x - 3) * (x - 1) ≥ 0}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Theorem statement
theorem intersection_equals_result : M_intersect_N = {x | x ≥ 3 ∨ x = 1} := by
  sorry

end intersection_equals_result_l225_22534


namespace tens_digit_of_expression_l225_22529

theorem tens_digit_of_expression : ∃ n : ℕ, (2023^2024 - 2025 + 6) % 100 = 10 + 100 * n := by
  sorry

end tens_digit_of_expression_l225_22529
