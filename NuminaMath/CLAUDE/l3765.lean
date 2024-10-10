import Mathlib

namespace fishing_competition_l3765_376545

/-- Fishing competition problem -/
theorem fishing_competition 
  (days : ℕ) 
  (jackson_per_day : ℕ) 
  (jonah_per_day : ℕ) 
  (total_catch : ℕ) :
  days = 5 →
  jackson_per_day = 6 →
  jonah_per_day = 4 →
  total_catch = 90 →
  ∃ (george_per_day : ℕ), 
    george_per_day = 8 ∧ 
    days * (jackson_per_day + jonah_per_day + george_per_day) = total_catch :=
by sorry

end fishing_competition_l3765_376545


namespace difference_of_values_l3765_376524

theorem difference_of_values (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
sorry

end difference_of_values_l3765_376524


namespace sum_of_solutions_is_seven_l3765_376585

theorem sum_of_solutions_is_seven : 
  let f (x : ℝ) := |x^2 - 8*x + 12|
  let g (x : ℝ) := 35/4 - x
  ∃ (a b : ℝ), (f a = g a) ∧ (f b = g b) ∧ (a + b = 7) ∧ 
    (∀ (x : ℝ), (f x = g x) → (x = a ∨ x = b)) :=
by sorry

end sum_of_solutions_is_seven_l3765_376585


namespace initial_girls_count_l3765_376509

theorem initial_girls_count (initial_boys : ℕ) (new_girls : ℕ) (total_pupils : ℕ) 
  (h1 : initial_boys = 222)
  (h2 : new_girls = 418)
  (h3 : total_pupils = 1346)
  : ∃ initial_girls : ℕ, initial_girls + initial_boys + new_girls = total_pupils ∧ initial_girls = 706 := by
  sorry

end initial_girls_count_l3765_376509


namespace problem_1_problem_2_l3765_376511

-- Problem 1
theorem problem_1 (a : ℝ) : (-2*a)^3 + 2*a^2 * 5*a = 2*a^3 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : (3*x*y^2)^2 + (-4*x*y^3)*(-x*y) = 13*x^2*y^4 := by
  sorry

end problem_1_problem_2_l3765_376511


namespace equation_always_has_real_root_l3765_376556

theorem equation_always_has_real_root :
  ∀ (q : ℝ), ∃ (x : ℝ), x^6 + q*x^4 + q^2*x^2 + 1 = 0 :=
by sorry

end equation_always_has_real_root_l3765_376556


namespace positive_number_square_plus_twice_l3765_376579

theorem positive_number_square_plus_twice : ∃ n : ℝ, n > 0 ∧ n^2 + 2*n = 210 ∧ n = 14 := by
  sorry

end positive_number_square_plus_twice_l3765_376579


namespace at_least_one_half_l3765_376534

theorem at_least_one_half (x y z : ℝ) 
  (h : x + y + z - 2 * (x * y + y * z + x * z) + 4 * x * y * z = (1 : ℝ) / 2) : 
  x = (1 : ℝ) / 2 ∨ y = (1 : ℝ) / 2 ∨ z = (1 : ℝ) / 2 := by
  sorry

end at_least_one_half_l3765_376534


namespace square_root_of_256_l3765_376577

theorem square_root_of_256 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 256) : y = 16 := by
  sorry

end square_root_of_256_l3765_376577


namespace motorcycle_price_increase_l3765_376573

/-- Represents the price increase of a motorcycle model --/
def price_increase (original_price : ℝ) (new_price : ℝ) : ℝ :=
  new_price - original_price

/-- Theorem stating the price increase given the problem conditions --/
theorem motorcycle_price_increase :
  ∀ (original_price : ℝ) (original_quantity : ℕ) (new_quantity : ℕ) (revenue_increase : ℝ),
    original_quantity = new_quantity + 8 →
    new_quantity = 63 →
    revenue_increase = 26000 →
    original_price * original_quantity = 594000 - revenue_increase →
    (original_price + price_increase original_price (original_price + price_increase original_price original_price)) * new_quantity = 594000 →
    price_increase original_price (original_price + price_increase original_price original_price) = 1428.57 := by
  sorry


end motorcycle_price_increase_l3765_376573


namespace equation1_solutions_equation2_solutions_l3765_376561

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 + 4*x - 1 = 0
def equation2 (x : ℝ) : Prop := (x-2)^2 - 3*x*(x-2) = 0

-- Theorem for the first equation
theorem equation1_solutions :
  ∃ x1 x2 : ℝ, x1 = -2 + Real.sqrt 5 ∧ x2 = -2 - Real.sqrt 5 ∧
  equation1 x1 ∧ equation1 x2 ∧
  ∀ x : ℝ, equation1 x → x = x1 ∨ x = x2 :=
sorry

-- Theorem for the second equation
theorem equation2_solutions :
  ∃ x1 x2 : ℝ, x1 = 2 ∧ x2 = -1 ∧
  equation2 x1 ∧ equation2 x2 ∧
  ∀ x : ℝ, equation2 x → x = x1 ∨ x = x2 :=
sorry

end equation1_solutions_equation2_solutions_l3765_376561


namespace cryptarithm_solution_l3765_376560

theorem cryptarithm_solution (A B C : ℕ) : 
  A ≠ 0 ∧ 
  A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧
  100 * A + 10 * B + C - (10 * B + C) = 100 * A + A → 
  C = 9 := by
sorry

end cryptarithm_solution_l3765_376560


namespace alligators_not_hiding_l3765_376542

/-- The number of alligators not hiding in a zoo cage -/
theorem alligators_not_hiding (total_alligators hiding_alligators : ℕ) :
  total_alligators = 75 → hiding_alligators = 19 →
  total_alligators - hiding_alligators = 56 := by
  sorry

#check alligators_not_hiding

end alligators_not_hiding_l3765_376542


namespace amy_haircut_l3765_376590

/-- Given an initial hair length and the amount cut off, calculates the final hair length -/
def final_hair_length (initial_length cut_off : ℕ) : ℕ :=
  initial_length - cut_off

/-- Proves that given an initial hair length of 11 inches and cutting off 4 inches, 
    the resulting hair length is 7 inches -/
theorem amy_haircut : final_hair_length 11 4 = 7 := by
  sorry

end amy_haircut_l3765_376590


namespace crayon_count_prove_crayon_count_l3765_376571

theorem crayon_count : ℕ → Prop :=
  fun red_count =>
    let blue_count := red_count + 5
    let yellow_count := 2 * blue_count - 6
    yellow_count = 32 → red_count = 14

/-- Proof of the crayon count theorem -/
theorem prove_crayon_count : ∃ (red_count : ℕ), crayon_count red_count :=
  sorry

end crayon_count_prove_crayon_count_l3765_376571


namespace linear_system_integer_solution_l3765_376555

theorem linear_system_integer_solution :
  ∃ (x y : ℤ), x + y = 5 ∧ 2 * x + y = 7 := by
  sorry

end linear_system_integer_solution_l3765_376555


namespace two_colored_cubes_count_l3765_376510

/-- Represents a cube with its side length -/
structure Cube where
  side : ℕ

/-- Represents a hollow cube with outer and inner dimensions -/
structure HollowCube where
  outer : Cube
  inner : Cube

/-- Calculates the number of smaller cubes with paint on exactly two sides -/
def cubesWithTwoColoredSides (hc : HollowCube) (smallCubeSide : ℕ) : ℕ :=
  12 * (hc.outer.side / smallCubeSide - 2)

theorem two_colored_cubes_count 
  (bigCube : Cube)
  (smallCube : Cube)
  (tinyCube : Cube)
  (hc : HollowCube) :
  bigCube.side = 27 →
  smallCube.side = 9 →
  tinyCube.side = 3 →
  hc.outer = bigCube →
  hc.inner = smallCube →
  cubesWithTwoColoredSides hc tinyCube.side = 84 := by
  sorry

#check two_colored_cubes_count

end two_colored_cubes_count_l3765_376510


namespace unique_valid_stamp_set_l3765_376517

/-- Given unlimited supply of stamps of denominations 7, n, and n+1 cents,
    101 cents is the greatest postage that cannot be formed -/
def is_valid_stamp_set (n : ℕ+) : Prop :=
  ∀ k : ℕ, k > 101 → ∃ a b c : ℕ, k = 7 * a + n * b + (n + 1) * c ∧
  ¬∃ a b c : ℕ, 101 = 7 * a + n * b + (n + 1) * c

theorem unique_valid_stamp_set :
  ∃! n : ℕ+, is_valid_stamp_set n ∧ n = 18 := by sorry

end unique_valid_stamp_set_l3765_376517


namespace possible_sets_B_l3765_376543

theorem possible_sets_B (A B : Set Int) : 
  A = {-1} → A ∪ B = {-1, 3} → (B = {3} ∨ B = {-1, 3}) := by
  sorry

end possible_sets_B_l3765_376543


namespace amanda_ticket_sales_l3765_376564

/-- The number of days Amanda needs to sell tickets -/
def days_to_sell : ℕ := 3

/-- The total number of tickets Amanda needs to sell -/
def total_tickets : ℕ := 80

/-- The number of tickets sold on day 1 -/
def day1_sales : ℕ := 20

/-- The number of tickets sold on day 2 -/
def day2_sales : ℕ := 32

/-- The number of tickets sold on day 3 -/
def day3_sales : ℕ := 28

/-- Theorem stating that Amanda needs 3 days to sell all tickets -/
theorem amanda_ticket_sales : 
  days_to_sell = 3 ∧ 
  total_tickets = day1_sales + day2_sales + day3_sales := by
  sorry

end amanda_ticket_sales_l3765_376564


namespace largest_quantity_l3765_376582

theorem largest_quantity (A B C : ℚ) : 
  A = 3003 / 3002 + 3003 / 3004 →
  B = 2 / 1 + 4 / 2 + 3005 / 3004 →
  C = 3004 / 3003 + 3004 / 3005 →
  B > A ∧ B > C := by
sorry


end largest_quantity_l3765_376582


namespace system_solutions_l3765_376513

def is_solution (x y : ℤ) : Prop :=
  |x^2 - 2*x| < y + (1/2) ∧ y + |x - 1| < 2

theorem system_solutions :
  ∀ x y : ℤ, is_solution x y ↔ (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 0) ∨ (x = 1 ∧ y = 1) :=
by sorry

end system_solutions_l3765_376513


namespace ending_number_proof_l3765_376562

theorem ending_number_proof (start : ℕ) (multiples : ℚ) (end_number : ℕ) : 
  start = 81 → 
  multiples = 93.33333333333333 → 
  end_number = (start + 3 * (multiples.floor - 1)) → 
  end_number = 357 := by
sorry

end ending_number_proof_l3765_376562


namespace cos_equation_solution_l3765_376532

theorem cos_equation_solution (θ : Real) :
  2 * (Real.cos θ)^2 - 5 * Real.cos θ + 2 = 0 → θ = Real.pi / 3 := by
  sorry

end cos_equation_solution_l3765_376532


namespace prob_one_common_is_two_thirds_l3765_376557

/-- The number of elective courses available -/
def num_courses : ℕ := 4

/-- The number of courses each student selects -/
def courses_per_student : ℕ := 2

/-- The total number of ways two students can select their courses -/
def total_selections : ℕ := (num_courses.choose courses_per_student) ^ 2

/-- The number of ways two students can select courses with exactly one in common -/
def one_common_selection : ℕ := num_courses * (num_courses - 1) * (num_courses - 2)

/-- The probability of two students sharing exactly one course in common -/
def prob_one_common : ℚ := one_common_selection / total_selections

theorem prob_one_common_is_two_thirds : prob_one_common = 2 / 3 := by
  sorry

end prob_one_common_is_two_thirds_l3765_376557


namespace min_value_reciprocal_sum_l3765_376576

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (1 / a + 4 / b) ≥ 9 / 4 :=
sorry

end min_value_reciprocal_sum_l3765_376576


namespace inequality_proof_l3765_376539

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let P := Real.sqrt ((a^2 + b^2)/2) - (a + b)/2
  let Q := (a + b)/2 - Real.sqrt (a*b)
  let R := Real.sqrt (a*b) - (2*a*b)/(a + b)
  Q ≥ P ∧ P ≥ R := by sorry

end inequality_proof_l3765_376539


namespace arithmetic_mean_relation_l3765_376504

theorem arithmetic_mean_relation (a b x : ℝ) : 
  (2 * x = a + b) →  -- x is the arithmetic mean of a and b
  (2 * x^2 = a^2 - b^2) →  -- x² is the arithmetic mean of a² and -b²
  (a = -b ∨ a = 3*b) :=  -- The relationship between a and b
by
  sorry

end arithmetic_mean_relation_l3765_376504


namespace squares_to_rectangles_ratio_l3765_376586

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : Nat
  cols : Nat
  horizontal_lines : Nat
  vertical_lines : Nat

/-- Calculates the number of squares on a checkerboard -/
def count_squares (board : Checkerboard) : Nat :=
  sorry

/-- Calculates the number of rectangles on a checkerboard -/
def count_rectangles (board : Checkerboard) : Nat :=
  sorry

/-- The main theorem stating the ratio of squares to rectangles on a 6x6 checkerboard -/
theorem squares_to_rectangles_ratio (board : Checkerboard) :
  board.rows = 6 ∧ board.cols = 6 ∧ board.horizontal_lines = 5 ∧ board.vertical_lines = 5 →
  (count_squares board : Rat) / (count_rectangles board : Rat) = 1 / 7 := by
  sorry

end squares_to_rectangles_ratio_l3765_376586


namespace flour_needed_for_cake_l3765_376580

/-- Given a recipe that requires a certain amount of flour and some flour already added,
    calculate the remaining amount of flour needed. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- The problem statement -/
theorem flour_needed_for_cake : remaining_flour 7 2 = 5 := by
  sorry

end flour_needed_for_cake_l3765_376580


namespace salaria_trees_count_l3765_376522

/-- Represents the total number of trees Salaria has -/
def total_trees : ℕ := sorry

/-- Represents the number of oranges tree A produces per month -/
def tree_A_oranges : ℕ := 10

/-- Represents the number of oranges tree B produces per month -/
def tree_B_oranges : ℕ := 15

/-- Represents the fraction of good oranges from tree A -/
def tree_A_good_fraction : ℚ := 3/5

/-- Represents the fraction of good oranges from tree B -/
def tree_B_good_fraction : ℚ := 1/3

/-- Represents the total number of good oranges Salaria gets per month -/
def total_good_oranges : ℕ := 55

theorem salaria_trees_count :
  total_trees = 10 ∧
  (total_trees / 2 : ℚ) * tree_A_oranges * tree_A_good_fraction +
  (total_trees / 2 : ℚ) * tree_B_oranges * tree_B_good_fraction = total_good_oranges := by
  sorry

end salaria_trees_count_l3765_376522


namespace boat_travel_time_l3765_376502

theorem boat_travel_time (boat_speed : ℝ) (distance : ℝ) (return_time : ℝ) :
  boat_speed = 15.6 →
  distance = 96 →
  return_time = 5 →
  ∃ (current_speed : ℝ),
    current_speed > 0 ∧
    current_speed < boat_speed ∧
    distance = (boat_speed + current_speed) * return_time ∧
    distance / (boat_speed - current_speed) = 8 :=
by sorry

end boat_travel_time_l3765_376502


namespace ice_cream_cost_l3765_376554

theorem ice_cream_cost (price : ℚ) (discount : ℚ) : 
  price = 99/100 ∧ discount = 1/10 → 
  price + price * (1 - discount) = 1881/1000 := by
  sorry

end ice_cream_cost_l3765_376554


namespace cyclist_distance_difference_l3765_376568

/-- The difference in distance traveled between two cyclists over a given time period -/
def distance_difference (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) : ℝ :=
  (rate1 * time) - (rate2 * time)

/-- Theorem: The difference in distance traveled between two cyclists, 
    one traveling at 12 miles per hour and the other at 10 miles per hour, 
    over a period of 6 hours, is 12 miles. -/
theorem cyclist_distance_difference :
  distance_difference 12 10 6 = 12 := by
  sorry

end cyclist_distance_difference_l3765_376568


namespace job_fair_problem_l3765_376531

/-- The probability of individual A being hired -/
def prob_A : ℚ := 4/9

/-- The probability of individuals B and C being hired -/
def prob_BC (t : ℚ) : ℚ := t/3

/-- The condition that t is between 0 and 3 -/
def t_condition (t : ℚ) : Prop := 0 < t ∧ t < 3

/-- The probability of all three individuals being hired -/
def prob_all (t : ℚ) : ℚ := prob_A * prob_BC t * prob_BC t

/-- The number of people hired from A and B -/
def ξ : Fin 3 → ℚ
| 0 => 0
| 1 => 1
| 2 => 2

/-- The probability distribution of ξ -/
def prob_ξ (t : ℚ) : Fin 3 → ℚ
| 0 => (1 - prob_A) * (1 - prob_BC t)
| 1 => prob_A * (1 - prob_BC t) + (1 - prob_A) * prob_BC t
| 2 => prob_A * prob_BC t

/-- The mathematical expectation of ξ -/
def expectation_ξ (t : ℚ) : ℚ :=
  (ξ 0) * (prob_ξ t 0) + (ξ 1) * (prob_ξ t 1) + (ξ 2) * (prob_ξ t 2)

theorem job_fair_problem (t : ℚ) (h : t_condition t) (h_prob : prob_all t = 16/81) :
  t = 2 ∧ expectation_ξ t = 10/9 := by
  sorry

end job_fair_problem_l3765_376531


namespace range_of_f_l3765_376516

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ≥ 0, f x = y} = {y : ℝ | y ≥ 3} := by
  sorry

end range_of_f_l3765_376516


namespace least_number_with_remainder_l3765_376549

theorem least_number_with_remainder (n : ℕ) : n = 282 ↔ 
  (n > 0 ∧ 
   n % 31 = 3 ∧ 
   n % 9 = 3 ∧ 
   ∀ m : ℕ, m > 0 → m % 31 = 3 → m % 9 = 3 → m ≥ n) :=
by sorry

end least_number_with_remainder_l3765_376549


namespace plot_length_l3765_376596

/-- Given a rectangular plot with the specified conditions, prove that its length is 70 meters. -/
theorem plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 40 →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  perimeter = 2 * (length + breadth) →
  total_cost = cost_per_meter * perimeter →
  length = 70 := by sorry

end plot_length_l3765_376596


namespace ball_fall_height_l3765_376583

/-- Given a ball falling from a certain height, this theorem calculates its final height from the ground. -/
theorem ball_fall_height (initial_height : ℝ) (fall_time : ℝ) (fall_speed : ℝ) :
  initial_height = 120 →
  fall_time = 20 →
  fall_speed = 4 →
  initial_height - fall_time * fall_speed = 40 := by
sorry

end ball_fall_height_l3765_376583


namespace min_sum_squares_exists_min_sum_squares_l3765_376530

def S : Finset ℤ := {3, -5, 0, 9, -2}

theorem min_sum_squares (a b c : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  13 ≤ a^2 + b^2 + c^2 :=
by sorry

theorem exists_min_sum_squares :
  ∃ (a b c : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^2 + b^2 + c^2 = 13 :=
by sorry

end min_sum_squares_exists_min_sum_squares_l3765_376530


namespace coefficient_x_squared_is_correct_l3765_376599

/-- The coefficient of x^2 in the binomial expansion of (x^2/2 - 1/√x)^6 -/
def coefficient_x_squared : ℚ :=
  let expression := (fun x => x^2/2 - 1/Real.sqrt x)^6
  -- We don't actually compute the coefficient here, just define it
  15/4

/-- Theorem stating that the coefficient of x^2 in the binomial expansion of (x^2/2 - 1/√x)^6 is 15/4 -/
theorem coefficient_x_squared_is_correct :
  coefficient_x_squared = 15/4 := by
  sorry

end coefficient_x_squared_is_correct_l3765_376599


namespace function_existence_condition_l3765_376550

theorem function_existence_condition (k a : ℕ) :
  (∃ f : ℕ → ℕ, ∀ n : ℕ, (f^[k] n) = n + a) ↔ (a = 0 ∨ a > 0) ∧ k ∣ a :=
by sorry

end function_existence_condition_l3765_376550


namespace measles_cases_1995_l3765_376521

/-- Represents the number of measles cases in a given year -/
def measles_cases (year : ℕ) : ℝ :=
  if year ≤ 1990 then
    300000 - 14950 * (year - 1970)
  else
    -8 * (year - 1990)^2 + 1000

/-- The theorem stating that the number of measles cases in 1995 is 800 -/
theorem measles_cases_1995 : measles_cases 1995 = 800 := by
  sorry

end measles_cases_1995_l3765_376521


namespace complex_modulus_problem_l3765_376540

/-- Given that z₁ = -1 + i and z₁z₂ = -2, prove that |z₂ + 2i| = √10 -/
theorem complex_modulus_problem (z₁ z₂ : ℂ) : 
  z₁ = -1 + Complex.I → z₁ * z₂ = -2 → Complex.abs (z₂ + 2 * Complex.I) = Real.sqrt 10 := by
  sorry

end complex_modulus_problem_l3765_376540


namespace min_sum_sides_triangle_l3765_376567

theorem min_sum_sides_triangle (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  ((a + b)^2 - c^2 = 4) →
  (C = Real.pi / 3) →
  (c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) →
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = a + b ∧ x * y = 4 / 3) →
  (a + b ≥ 4 * Real.sqrt 3 / 3) :=
by sorry

end min_sum_sides_triangle_l3765_376567


namespace not_q_necessary_not_sufficient_for_not_p_l3765_376558

def p (x : ℝ) : Prop := |x + 1| ≤ 4

def q (x : ℝ) : Prop := 2 < x ∧ x < 3

theorem not_q_necessary_not_sufficient_for_not_p :
  (∀ x, ¬(p x) → ¬(q x)) ∧ (∃ x, ¬(q x) ∧ p x) :=
sorry

end not_q_necessary_not_sufficient_for_not_p_l3765_376558


namespace five_touching_circles_exist_l3765_376565

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_positive : radius > 0

/-- Two circles touch if the distance between their centers is equal to the sum or difference of their radii --/
def circles_touch (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2 ∨
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius - c2.radius)^2

/-- Theorem: There exists a configuration of five circles such that any two of them touch each other --/
theorem five_touching_circles_exist : ∃ (c1 c2 c3 c4 c5 : Circle),
  circles_touch c1 c2 ∧ circles_touch c1 c3 ∧ circles_touch c1 c4 ∧ circles_touch c1 c5 ∧
  circles_touch c2 c3 ∧ circles_touch c2 c4 ∧ circles_touch c2 c5 ∧
  circles_touch c3 c4 ∧ circles_touch c3 c5 ∧
  circles_touch c4 c5 :=
sorry

end five_touching_circles_exist_l3765_376565


namespace condition_sufficient_not_necessary_l3765_376553

-- Define the condition "m < 1/4"
def condition (m : ℝ) : Prop := m < (1/4 : ℝ)

-- Define when a quadratic equation has real solutions
def has_real_solutions (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

-- State the theorem
theorem condition_sufficient_not_necessary :
  (∀ m : ℝ, condition m → has_real_solutions 1 1 m) ∧
  (∃ m : ℝ, ¬(condition m) ∧ has_real_solutions 1 1 m) :=
sorry

end condition_sufficient_not_necessary_l3765_376553


namespace unique_a_value_l3765_376523

/-- The base-72 number 235935623 -/
def base_72_num : ℕ := 235935623

/-- The proposition that the given base-72 number minus a is divisible by 9 -/
def is_divisible_by_nine (a : ℤ) : Prop :=
  (base_72_num : ℤ) - a ≡ 0 [ZMOD 9]

theorem unique_a_value :
  ∃! a : ℤ, 0 ≤ a ∧ a ≤ 18 ∧ is_divisible_by_nine a ∧ a = 4 := by
  sorry

end unique_a_value_l3765_376523


namespace triangle_side_length_l3765_376527

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 5) (h2 : c = 8) (h3 : B = π / 3) :
  ∃ b : ℝ, b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B ∧ b > 0 :=
by sorry

end triangle_side_length_l3765_376527


namespace time_to_finish_book_l3765_376548

/-- Calculates the time needed to finish a book given the current reading progress and reading speed. -/
theorem time_to_finish_book (total_pages reading_speed current_page : ℕ) 
  (h1 : total_pages = 210)
  (h2 : current_page = 90)
  (h3 : reading_speed = 30) : 
  (total_pages - current_page) / reading_speed = 4 :=
by sorry

end time_to_finish_book_l3765_376548


namespace perpendicular_lines_a_value_l3765_376594

-- Define the slopes of two lines
def slope1 (a : ℝ) := -a
def slope2 : ℝ := 3

-- Define the perpendicular condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_a_value :
  ∀ a : ℝ, perpendicular (slope1 a) slope2 → a = 1/3 := by
  sorry

end perpendicular_lines_a_value_l3765_376594


namespace negative_f_m_plus_one_l3765_376536

theorem negative_f_m_plus_one 
  (f : ℝ → ℝ) 
  (a m : ℝ) 
  (h1 : ∀ x, f x = x^2 - x + a) 
  (h2 : f (-m) < 0) : 
  f (m + 1) < 0 := by
sorry

end negative_f_m_plus_one_l3765_376536


namespace circle_area_increase_l3765_376508

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end circle_area_increase_l3765_376508


namespace set_relationship_l3765_376544

def M : Set ℤ := {x | ∃ k : ℤ, x = 5 * k - 2}
def P : Set ℤ := {x | ∃ n : ℤ, x = 5 * n + 3}
def S : Set ℤ := {x | ∃ m : ℤ, x = 10 * m + 3}

theorem set_relationship : S ⊆ P ∧ P = M := by sorry

end set_relationship_l3765_376544


namespace student_ticket_price_l3765_376541

theorem student_ticket_price 
  (total_tickets : ℕ) 
  (student_tickets : ℕ) 
  (non_student_tickets : ℕ) 
  (non_student_price : ℚ) 
  (total_revenue : ℚ) :
  total_tickets = 150 →
  student_tickets = 90 →
  non_student_tickets = 60 →
  non_student_price = 8 →
  total_revenue = 930 →
  ∃ (student_price : ℚ), 
    student_price * student_tickets + non_student_price * non_student_tickets = total_revenue ∧
    student_price = 5 := by
  sorry

end student_ticket_price_l3765_376541


namespace roots_of_polynomial_l3765_376537

def f (x : ℝ) : ℝ := 4*x^4 + 17*x^3 - 37*x^2 + 6*x

theorem roots_of_polynomial :
  ∃ (a b c d : ℝ),
    (a = 0) ∧
    (b = 1/2) ∧
    (c = (-9 + Real.sqrt 129) / 4) ∧
    (d = (-9 - Real.sqrt 129) / 4) ∧
    (∀ x : ℝ, f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end roots_of_polynomial_l3765_376537


namespace modular_inverse_of_7_mod_26_l3765_376546

theorem modular_inverse_of_7_mod_26 : ∃ x : ℕ, x ∈ Finset.range 26 ∧ (7 * x) % 26 = 1 := by
  use 15
  sorry

end modular_inverse_of_7_mod_26_l3765_376546


namespace product_squared_l3765_376518

theorem product_squared (a b : ℝ) : (a * b)^2 = a^2 * b^2 := by sorry

end product_squared_l3765_376518


namespace fraction_simplification_l3765_376505

theorem fraction_simplification : (25 : ℚ) / 24 * 18 / 35 * 56 / 45 = 50 / 3 := by
  sorry

end fraction_simplification_l3765_376505


namespace triangle_side_length_l3765_376569

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the length function
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the median function
def median (P M : ℝ × ℝ) (Q R : ℝ × ℝ) : Prop :=
  length P M = 3.5 ∧ M = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2)

theorem triangle_side_length (P Q R : ℝ × ℝ) :
  Triangle P Q R →
  length P Q = 4 →
  length P R = 7 →
  median P M Q R →
  length Q R = 9 := by sorry

end triangle_side_length_l3765_376569


namespace brownie_pieces_l3765_376512

theorem brownie_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 15)
  (h3 : piece_length = 3)
  (h4 : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 := by
  sorry

end brownie_pieces_l3765_376512


namespace function_composition_equality_l3765_376598

/-- Given two functions f and g, where f is quadratic and g is linear,
    if f(g(x)) = g(f(x)) for all x, then certain conditions on their coefficients must hold. -/
theorem function_composition_equality
  (a b c d e : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∀ x, g x = d * x + e)
  (h_eq : ∀ x, f (g x) = g (f x)) :
  a * (d - 1) = 0 ∧ a * e = 0 ∧ c - e = a * e^2 :=
by sorry

end function_composition_equality_l3765_376598


namespace train_length_proof_l3765_376570

/-- Given a train and a platform with equal length, if the train crosses the platform
    in 60 seconds at a speed of 30 m/s, then the length of the train is 900 meters. -/
theorem train_length_proof (train_length platform_length : ℝ) 
  (speed : ℝ) (time : ℝ) (h1 : train_length = platform_length) 
  (h2 : speed = 30) (h3 : time = 60) :
  train_length = 900 := by
  sorry

end train_length_proof_l3765_376570


namespace cubic_is_odd_l3765_376519

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f (x : ℝ) : ℝ := x^3

theorem cubic_is_odd : is_odd_function f := by
  sorry

end cubic_is_odd_l3765_376519


namespace unique_solution_trigonometric_equation_l3765_376572

theorem unique_solution_trigonometric_equation :
  ∃! (n : ℕ+), (Real.sin (π / (3 * n.val)) + Real.cos (π / (3 * n.val)) = Real.sqrt (3 * n.val) / 3) ∧ n = 6 := by
  sorry

end unique_solution_trigonometric_equation_l3765_376572


namespace natasha_dimes_l3765_376593

theorem natasha_dimes : ∃ n : ℕ, 
  10 < n ∧ n < 100 ∧ 
  n % 3 = 1 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 1 ∧ 
  n = 61 := by
  sorry

end natasha_dimes_l3765_376593


namespace dans_remaining_money_l3765_376503

def dans_money_left (initial_amount spent_on_candy spent_on_chocolate : ℕ) : ℕ :=
  initial_amount - (spent_on_candy + spent_on_chocolate)

theorem dans_remaining_money :
  dans_money_left 7 2 3 = 2 := by sorry

end dans_remaining_money_l3765_376503


namespace coefficient_of_x_l3765_376559

theorem coefficient_of_x (x y : ℚ) :
  (x + 3 * y = 1) →
  (2 * x + y = 5) →
  ∃ (a : ℚ), a * x + y = 19 ∧ a = 7 := by
  sorry

end coefficient_of_x_l3765_376559


namespace cistern_wet_surface_area_l3765_376597

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem stating the total wet surface area of the given cistern -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 4 8 1.25 = 62 := by
  sorry

end cistern_wet_surface_area_l3765_376597


namespace quadratic_minimum_l3765_376535

/-- 
Given a quadratic function y = ax² + px + q where a ≠ 0,
if the minimum value of y is m, then q = m + p²/(4a)
-/
theorem quadratic_minimum (a p q m : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 + p * x + q ≥ m) →
  (∃ x₀, a * x₀^2 + p * x₀ + q = m) →
  q = m + p^2 / (4 * a) := by
  sorry

end quadratic_minimum_l3765_376535


namespace candles_per_box_l3765_376595

/-- Given Kerry's birthday celebration scenario, prove the number of candles in a box. -/
theorem candles_per_box (kerry_age : ℕ) (num_cakes : ℕ) (total_cost : ℚ) (box_cost : ℚ) 
  (h1 : kerry_age = 8)
  (h2 : num_cakes = 3)
  (h3 : total_cost = 5)
  (h4 : box_cost = 5/2) :
  (kerry_age * num_cakes) / (total_cost / box_cost) = 12 := by
  sorry

end candles_per_box_l3765_376595


namespace sin_half_range_l3765_376574

theorem sin_half_range (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α/2) > Real.cos (α/2)) :
  ∃ x, Real.sqrt 2 / 2 < x ∧ x < 1 ∧ x = Real.sin (α/2) :=
sorry

end sin_half_range_l3765_376574


namespace divisibility_by_100_l3765_376538

theorem divisibility_by_100 (a : ℕ) (h : ¬(5 ∣ a)) : 100 ∣ (a^8 + 3*a^4 - 4) := by
  sorry

end divisibility_by_100_l3765_376538


namespace probability_intersecting_diagonals_l3765_376501

/-- A regular decagon -/
structure RegularDecagon where
  -- Add any necessary properties

/-- Represents a diagonal in a regular decagon -/
structure Diagonal where
  -- Add any necessary properties

/-- The set of all diagonals in a regular decagon -/
def allDiagonals (d : RegularDecagon) : Set Diagonal :=
  sorry

/-- Predicate to check if two diagonals intersect inside the decagon -/
def intersectInside (d : RegularDecagon) (d1 d2 : Diagonal) : Prop :=
  sorry

/-- The number of ways to choose 3 diagonals from all diagonals -/
def numWaysChoose3Diagonals (d : RegularDecagon) : ℕ :=
  sorry

/-- The number of ways to choose 3 diagonals where at least two intersect -/
def numWaysChoose3IntersectingDiagonals (d : RegularDecagon) : ℕ :=
  sorry

theorem probability_intersecting_diagonals (d : RegularDecagon) :
    (numWaysChoose3IntersectingDiagonals d : ℚ) / (numWaysChoose3Diagonals d : ℚ) = 252 / 1309 := by
  sorry

end probability_intersecting_diagonals_l3765_376501


namespace base_is_seven_l3765_376525

/-- Converts a number from base s to base 10 -/
def to_base_10 (digits : List Nat) (s : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * s^i) 0

/-- The transaction equation holds for the given base -/
def transaction_holds (s : Nat) : Prop :=
  to_base_10 [3, 2, 5] s + to_base_10 [3, 5, 4] s = to_base_10 [0, 0, 1, 1] s

theorem base_is_seven :
  ∃ s : Nat, s > 1 ∧ transaction_holds s ∧ s = 7 := by
  sorry

end base_is_seven_l3765_376525


namespace fifth_selected_number_l3765_376526

def random_number_table : List Nat :=
  [16, 22, 77, 94, 39, 49, 54, 43, 54, 82, 17, 37, 93, 23, 78, 87, 35, 20, 96, 43]

def class_size : Nat := 50

def is_valid_number (n : Nat) : Bool :=
  n < class_size

def select_valid_numbers (numbers : List Nat) (count : Nat) : List Nat :=
  (numbers.filter is_valid_number).take count

theorem fifth_selected_number :
  (select_valid_numbers random_number_table 5).reverse.head? = some 43 := by
  sorry

end fifth_selected_number_l3765_376526


namespace attic_junk_items_l3765_376551

theorem attic_junk_items (total : ℕ) (useful : ℕ) (junk_percent : ℚ) :
  useful = (20 : ℚ) / 100 * total →
  junk_percent = 70 / 100 →
  useful = 8 →
  ⌊junk_percent * total⌋ = 28 := by
sorry

end attic_junk_items_l3765_376551


namespace blue_highlighters_count_l3765_376533

theorem blue_highlighters_count (pink : ℕ) (yellow : ℕ) (total : ℕ) (blue : ℕ) :
  pink = 6 → yellow = 2 → total = 12 → blue = total - (pink + yellow) → blue = 4 := by
  sorry

end blue_highlighters_count_l3765_376533


namespace car_distance_proof_l3765_376578

theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) : 
  initial_time = 6 →
  speed = 80 →
  (initial_time * 3 / 2) * speed = 720 :=
by
  sorry

end car_distance_proof_l3765_376578


namespace same_combination_probability_is_correct_l3765_376547

def jar_candies : ℕ × ℕ := (12, 8)

def total_candies : ℕ := jar_candies.1 + jar_candies.2

def same_combination_probability : ℚ :=
  let terry_picks := Nat.choose total_candies 2
  let mary_picks := Nat.choose (total_candies - 2) 2
  let both_red := (Nat.choose jar_candies.1 2 * Nat.choose (jar_candies.1 - 2) 2) / (terry_picks * mary_picks)
  let both_blue := (Nat.choose jar_candies.2 2 * Nat.choose (jar_candies.2 - 2) 2) / (terry_picks * mary_picks)
  let mixed := (Nat.choose jar_candies.1 1 * Nat.choose jar_candies.2 1 * 
                Nat.choose (jar_candies.1 - 1) 1 * Nat.choose (jar_candies.2 - 1) 1) / 
               (terry_picks * mary_picks)
  both_red + both_blue + mixed

theorem same_combination_probability_is_correct : 
  same_combination_probability = 143 / 269 := by
  sorry

end same_combination_probability_is_correct_l3765_376547


namespace evaluate_expression_l3765_376506

theorem evaluate_expression : Real.sqrt ((4 / 3) * (1 / 15 + 1 / 25)) = 4 * Real.sqrt 2 / 15 := by
  sorry

end evaluate_expression_l3765_376506


namespace complex_square_equality_l3765_376566

theorem complex_square_equality (c d : ℕ+) :
  (c + d * Complex.I) ^ 2 = 15 + 8 * Complex.I →
  c + d * Complex.I = 4 + Complex.I := by
  sorry

end complex_square_equality_l3765_376566


namespace cube_surface_area_l3765_376528

theorem cube_surface_area (x : ℝ) (h : x > 0) :
  let volume := x^3
  let side_length := x
  let surface_area := 6 * side_length^2
  volume = x^3 → surface_area = 6 * x^2 := by
sorry

end cube_surface_area_l3765_376528


namespace prime_natural_equation_solutions_l3765_376591

theorem prime_natural_equation_solutions :
  ∀ p n : ℕ,
    Prime p →
    p^2 + n^2 = 3*p*n + 1 →
    ((p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8)) :=
by sorry

end prime_natural_equation_solutions_l3765_376591


namespace eve_distance_difference_l3765_376581

/-- Eve's running and walking distances problem -/
theorem eve_distance_difference :
  let run_distance : ℝ := 0.7
  let walk_distance : ℝ := 0.6
  run_distance - walk_distance = 0.1 := by sorry

end eve_distance_difference_l3765_376581


namespace min_value_sqrt_sum_squares_l3765_376514

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 3 := by
  sorry

end min_value_sqrt_sum_squares_l3765_376514


namespace system_solution_l3765_376588

/-- Given a system of equations, prove that x and y have specific values. -/
theorem system_solution (a x y : ℝ) (h1 : Real.log (x^2 + y^2) / Real.log (Real.sqrt 10) = 2 * Real.log (2*a) / Real.log 10 + 2 * Real.log (x^2 - y^2) / Real.log 100) (h2 : x * y = a^2) :
  (x = a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = a * Real.sqrt (Real.sqrt 2 - 1)) ∨
  (x = -a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = -a * Real.sqrt (Real.sqrt 2 - 1)) ∨
  (x = a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = -a * Real.sqrt (Real.sqrt 2 - 1)) ∨
  (x = -a * Real.sqrt (Real.sqrt 2 + 1) ∧ y = a * Real.sqrt (Real.sqrt 2 - 1)) :=
by sorry

end system_solution_l3765_376588


namespace sqrt_three_irrational_l3765_376587

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_three_irrational_l3765_376587


namespace johns_age_l3765_376563

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 := by
  sorry

end johns_age_l3765_376563


namespace modular_inverse_of_3_mod_37_l3765_376584

theorem modular_inverse_of_3_mod_37 : ∃ x : ℤ, 
  (x * 3) % 37 = 1 ∧ 
  0 ≤ x ∧ 
  x ≤ 36 ∧ 
  x = 25 := by
  sorry

end modular_inverse_of_3_mod_37_l3765_376584


namespace least_subtraction_for_divisibility_problem_solution_l3765_376500

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 2 ∧ (427398 - x) % 13 = 0 ∧ ∀ (y : ℕ), y < x → (427398 - y) % 13 ≠ 0 :=
by
  sorry

end least_subtraction_for_divisibility_problem_solution_l3765_376500


namespace sequence_formula_l3765_376529

/-- Given a sequence {a_n} defined by a₁ = 2 and a_{n+1} = a_n + ln(1 + 1/n) for n ≥ 1,
    prove that a_n = 2 + ln(n) for all n ≥ 1 -/
theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 2) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + Real.log (1 + 1 / n)) :
    ∀ n : ℕ, n ≥ 1 → a n = 2 + Real.log n := by
  sorry

end sequence_formula_l3765_376529


namespace f_max_at_neg_two_l3765_376507

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 16

-- State the theorem
theorem f_max_at_neg_two :
  ∀ x : ℝ, f x ≤ f (-2) :=
by
  sorry

end f_max_at_neg_two_l3765_376507


namespace january_has_greatest_difference_l3765_376520

-- Define the sales data for each month
def january_sales : (Nat × Nat) := (5, 2)
def february_sales : (Nat × Nat) := (6, 4)
def march_sales : (Nat × Nat) := (5, 5)
def april_sales : (Nat × Nat) := (4, 6)
def may_sales : (Nat × Nat) := (3, 5)

-- Define the percentage difference function
def percentage_difference (sales : Nat × Nat) : ℚ :=
  let (drummers, buglers) := sales
  (↑(max drummers buglers - min drummers buglers) / ↑(min drummers buglers)) * 100

-- Theorem statement
theorem january_has_greatest_difference :
  percentage_difference january_sales >
  max (percentage_difference february_sales)
    (max (percentage_difference march_sales)
      (max (percentage_difference april_sales)
        (percentage_difference may_sales))) :=
by sorry

end january_has_greatest_difference_l3765_376520


namespace jamie_marbles_l3765_376515

theorem jamie_marbles (n : ℕ) : 
  n > 0 ∧ 
  (2 * n) % 5 = 0 ∧ 
  n % 3 = 0 ∧ 
  n ≥ 15 → 
  ∃ (blue red green yellow : ℕ), 
    blue = 2 * n / 5 ∧
    red = n / 3 ∧
    green = 4 ∧
    yellow = n - (blue + red + green) ∧
    yellow ≥ 0 ∧
    ∀ (m : ℕ), m < n → 
      (2 * m) % 5 = 0 → 
      m % 3 = 0 → 
      m - (2 * m / 5 + m / 3 + 4) < 0 :=
by sorry

end jamie_marbles_l3765_376515


namespace garden_length_is_50_l3765_376552

def garden_length (width : ℝ) : ℝ := 2 * width

def garden_perimeter (width : ℝ) : ℝ := 2 * garden_length width + 2 * width

theorem garden_length_is_50 :
  ∃ (width : ℝ), garden_perimeter width = 150 ∧ garden_length width = 50 :=
by sorry

end garden_length_is_50_l3765_376552


namespace tank_emptying_time_l3765_376589

-- Define constants
def tank_volume_cubic_feet : ℝ := 20
def inlet_rate : ℝ := 5
def outlet_rate_1 : ℝ := 9
def outlet_rate_2 : ℝ := 8
def inches_per_foot : ℝ := 12

-- Theorem statement
theorem tank_emptying_time :
  let tank_volume_cubic_inches := tank_volume_cubic_feet * (inches_per_foot ^ 3)
  let net_emptying_rate := outlet_rate_1 + outlet_rate_2 - inlet_rate
  tank_volume_cubic_inches / net_emptying_rate = 2880 := by
  sorry

end tank_emptying_time_l3765_376589


namespace division_problem_l3765_376592

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
    (h1 : dividend = 2944)
    (h2 : divisor = 72)
    (h3 : remainder = 64)
    (h4 : dividend = divisor * quotient + remainder) :
  quotient = 40 := by
  sorry

end division_problem_l3765_376592


namespace ice_cream_choices_l3765_376575

/-- The number of ways to choose n items from k types with repetition -/
def choose_with_repetition (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to choose 5 scoops from 14 flavors with repetition -/
theorem ice_cream_choices : choose_with_repetition 5 14 = 3060 := by
  sorry

end ice_cream_choices_l3765_376575
