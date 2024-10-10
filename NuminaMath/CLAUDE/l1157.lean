import Mathlib

namespace arithmetic_sequence_property_l1157_115767

theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + d) → a 3 + a 7 = 2 * a 5 := by
  sorry

end arithmetic_sequence_property_l1157_115767


namespace isabel_savings_l1157_115715

def initial_amount : ℚ := 204
def toy_fraction : ℚ := 1/2
def book_fraction : ℚ := 1/2

theorem isabel_savings : 
  initial_amount * (1 - toy_fraction) * (1 - book_fraction) = 51 := by
  sorry

end isabel_savings_l1157_115715


namespace solve_lawn_mowing_problem_l1157_115710

/-- Edward's lawn mowing business finances -/
def lawn_mowing_problem (spring_earnings summer_earnings final_amount : ℕ) : Prop :=
  let total_earnings := spring_earnings + summer_earnings
  let supplies_cost := total_earnings - final_amount
  supplies_cost = total_earnings - final_amount

theorem solve_lawn_mowing_problem :
  lawn_mowing_problem 2 27 24 = true :=
by sorry

end solve_lawn_mowing_problem_l1157_115710


namespace remainder_calculation_l1157_115742

theorem remainder_calculation (a b r : ℕ) 
  (h1 : a - b = 1200)
  (h2 : a = 1495)
  (h3 : a = 5 * b + r)
  (h4 : r < b) : 
  r = 20 := by
sorry

end remainder_calculation_l1157_115742


namespace circle_center_sum_l1157_115756

/-- Given a circle with equation x^2 + y^2 - 6x + 8y + 9 = 0, 
    prove that the sum of the coordinates of its center is -1 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 8*y + 9 = 0 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9)) →
  h + k = -1 := by
sorry

end circle_center_sum_l1157_115756


namespace subset_implies_a_equals_three_l1157_115714

def A : Set ℝ := {2, 3}
def B (a : ℝ) : Set ℝ := {1, 2, a}

theorem subset_implies_a_equals_three (h : A ⊆ B a) : a = 3 := by
  sorry

end subset_implies_a_equals_three_l1157_115714


namespace jacket_price_proof_l1157_115790

theorem jacket_price_proof (S P : ℝ) (h1 : S = P + 0.4 * S) 
  (h2 : 0.8 * S - P = 18) : P = 54 := by
  sorry

end jacket_price_proof_l1157_115790


namespace cookie_radius_is_8_l1157_115794

/-- The equation of the cookie's boundary -/
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 21 = 4*x + 18*y

/-- The radius of the cookie -/
def cookie_radius : ℝ := 8

/-- Theorem stating that the radius of the cookie defined by the equation is 8 -/
theorem cookie_radius_is_8 :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    cookie_equation x y ↔ (x - h)^2 + (y - k)^2 = cookie_radius^2 :=
sorry

end cookie_radius_is_8_l1157_115794


namespace sallys_nickels_l1157_115752

theorem sallys_nickels (x : ℕ) : x + 9 + 2 = 18 → x = 7 := by
  sorry

end sallys_nickels_l1157_115752


namespace abs_eq_sum_implies_zero_l1157_115726

theorem abs_eq_sum_implies_zero (x y : ℝ) :
  |x - y^2| = x + y^2 → x = 0 ∧ y = 0 := by
  sorry

end abs_eq_sum_implies_zero_l1157_115726


namespace goat_average_price_l1157_115727

/-- The average price of a goat given the total cost of cows and goats, and the average price of a cow -/
theorem goat_average_price
  (total_cost : ℕ)
  (num_cows : ℕ)
  (num_goats : ℕ)
  (cow_avg_price : ℕ)
  (h1 : total_cost = 1400)
  (h2 : num_cows = 2)
  (h3 : num_goats = 8)
  (h4 : cow_avg_price = 460) :
  (total_cost - num_cows * cow_avg_price) / num_goats = 60 := by
  sorry

end goat_average_price_l1157_115727


namespace combined_area_square_triangle_l1157_115775

/-- The combined area of a square with diagonal 30 m and an equilateral triangle sharing that diagonal as its side is 450 m² + 225√3 m². -/
theorem combined_area_square_triangle (diagonal : ℝ) (h_diagonal : diagonal = 30) :
  let square_side := diagonal / Real.sqrt 2
  let square_area := square_side ^ 2
  let triangle_area := (Real.sqrt 3 / 4) * diagonal ^ 2
  square_area + triangle_area = 450 + 225 * Real.sqrt 3 := by
sorry

end combined_area_square_triangle_l1157_115775


namespace mutually_exclusive_not_opposite_l1157_115762

structure PencilCase where
  pencils : ℕ
  pens : ℕ

def case : PencilCase := { pencils := 2, pens := 2 }

def select_two (pc : PencilCase) : ℕ := 2

def exactly_one_pen (pc : PencilCase) : Prop :=
  ∃ (x : ℕ), x = 1 ∧ x ≤ pc.pens

def exactly_two_pencils (pc : PencilCase) : Prop :=
  ∃ (x : ℕ), x = 2 ∧ x ≤ pc.pencils

theorem mutually_exclusive_not_opposite :
  (exactly_one_pen case ∧ exactly_two_pencils case → False) ∧
  ¬(exactly_one_pen case ↔ ¬exactly_two_pencils case) :=
by sorry

end mutually_exclusive_not_opposite_l1157_115762


namespace complex_modulus_problem_l1157_115786

theorem complex_modulus_problem (x y : ℝ) (h : (x + Complex.I) * x = 4 + 2 * y * Complex.I) :
  Complex.abs ((x + 4 * y * Complex.I) / (1 + Complex.I)) = Real.sqrt 10 := by
  sorry

end complex_modulus_problem_l1157_115786


namespace race_finish_orders_l1157_115721

theorem race_finish_orders (n : Nat) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end race_finish_orders_l1157_115721


namespace smallest_sum_of_three_l1157_115738

def S : Finset Int := {10, 30, -12, 15, -8}

theorem smallest_sum_of_three (s : Finset Int) (h : s = S) :
  (Finset.powersetCard 3 s).toList.map (fun t => t.toList.sum)
    |>.minimum?
    |>.map (fun x => x = -10)
    |>.getD False :=
  sorry

end smallest_sum_of_three_l1157_115738


namespace daisy_shop_total_sales_l1157_115768

def daisy_shop_sales (day1 : ℕ) (day2_increase : ℕ) (day3_decrease : ℕ) (day4 : ℕ) : ℕ :=
  let day2 := day1 + day2_increase
  let day3 := 2 * day2 - day3_decrease
  day1 + day2 + day3 + day4

theorem daisy_shop_total_sales :
  daisy_shop_sales 45 20 10 120 = 350 := by
  sorry

end daisy_shop_total_sales_l1157_115768


namespace percent_application_l1157_115744

theorem percent_application (x : ℝ) : x * 0.0002 = 2.4712 → x = 12356 := by sorry

end percent_application_l1157_115744


namespace wire_length_problem_l1157_115712

theorem wire_length_problem (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 20 →
  longer_piece = 2/4 * shorter_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 60 := by
sorry

end wire_length_problem_l1157_115712


namespace smallest_shift_for_scaled_periodic_function_l1157_115758

-- Define a periodic function with period 20
def isPeriodic20 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 20) = f x

-- Define the property we want to prove
def smallestShift (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f ((x - a) / 5) = f (x / 5)) ∧
  (∀ b, 0 < b → b < a → ∃ x, f ((x - b) / 5) ≠ f (x / 5))

-- Theorem statement
theorem smallest_shift_for_scaled_periodic_function (f : ℝ → ℝ) (h : isPeriodic20 f) :
  smallestShift f 100 := by
  sorry

end smallest_shift_for_scaled_periodic_function_l1157_115758


namespace batsman_average_is_60_l1157_115736

/-- Represents a batsman's performance statistics -/
structure BatsmanStats where
  total_innings : ℕ
  highest_score : ℕ
  score_difference : ℕ
  avg_excluding_extremes : ℕ

/-- Calculates the overall batting average -/
def overall_average (stats : BatsmanStats) : ℚ :=
  let lowest_score := stats.highest_score - stats.score_difference
  let total_runs := (stats.total_innings - 2) * stats.avg_excluding_extremes + stats.highest_score + lowest_score
  total_runs / stats.total_innings

/-- Theorem stating the overall batting average is 60 runs given the specific conditions -/
theorem batsman_average_is_60 (stats : BatsmanStats) 
  (h_innings : stats.total_innings = 46)
  (h_highest : stats.highest_score = 199)
  (h_diff : stats.score_difference = 190)
  (h_avg : stats.avg_excluding_extremes = 58) :
  overall_average stats = 60 := by
  sorry


end batsman_average_is_60_l1157_115736


namespace functional_equation_solution_l1157_115718

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = x * f(1) -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f x = x * f 1 := by
  sorry

end functional_equation_solution_l1157_115718


namespace halloween_goodie_bags_l1157_115737

/-- Calculates the minimum cost for buying a given number of items,
    where packs of 5 cost $3 and individual items cost $1 each. -/
def minCost (n : ℕ) : ℕ :=
  (n / 5) * 3 + (n % 5)

/-- The Halloween goodie bag problem -/
theorem halloween_goodie_bags :
  let vampireBags := 11
  let pumpkinBags := 14
  let totalBags := vampireBags + pumpkinBags
  totalBags = 25 →
  minCost vampireBags + minCost pumpkinBags = 17 := by
sorry

end halloween_goodie_bags_l1157_115737


namespace triangle_properties_l1157_115781

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- State the theorem
theorem triangle_properties (t : Triangle)
  (h1 : t.A + t.B + t.C = π)
  (h2 : t.S > 0)
  (h3 : Real.tan (t.A / 2) * Real.tan (t.B / 2) + Real.sqrt 3 * (Real.tan (t.A / 2) + Real.tan (t.B / 2)) = 1) :
  t.C = 2 * π / 3 ∧ t.c^2 ≥ 4 * Real.sqrt 3 * t.S := by
  sorry

end triangle_properties_l1157_115781


namespace min_students_forgot_all_items_l1157_115747

theorem min_students_forgot_all_items (total : ℕ) (forgot_gloves : ℕ) (forgot_scarves : ℕ) (forgot_hats : ℕ) 
  (h1 : total = 60)
  (h2 : forgot_gloves = 55)
  (h3 : forgot_scarves = 52)
  (h4 : forgot_hats = 50) :
  total - ((total - forgot_gloves) + (total - forgot_scarves) + (total - forgot_hats)) = 37 := by
  sorry

end min_students_forgot_all_items_l1157_115747


namespace unique_solution_quadratic_linear_l1157_115711

theorem unique_solution_quadratic_linear (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 ∧ p.2 = 2*p.1 - k) ↔ k = 1 :=
sorry

end unique_solution_quadratic_linear_l1157_115711


namespace comb_cost_is_one_l1157_115797

/-- The cost of one set of barrettes in dollars -/
def barrette_cost : ℝ := 3

/-- The cost of one comb in dollars -/
def comb_cost : ℝ := 1

/-- Kristine's total purchase cost in dollars -/
def kristine_cost : ℝ := barrette_cost + comb_cost

/-- Crystal's total purchase cost in dollars -/
def crystal_cost : ℝ := 3 * barrette_cost + comb_cost

/-- The total amount spent by both girls in dollars -/
def total_spent : ℝ := 14

theorem comb_cost_is_one :
  kristine_cost + crystal_cost = total_spent → comb_cost = 1 := by
  sorry

end comb_cost_is_one_l1157_115797


namespace sequence_100th_term_l1157_115739

theorem sequence_100th_term (a : ℕ → ℕ) (h1 : a 1 = 2) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) : 
  a 100 = 9902 := by
  sorry

end sequence_100th_term_l1157_115739


namespace number_plus_five_equals_500_l1157_115741

theorem number_plus_five_equals_500 : ∃ x : ℤ, x + 5 = 500 ∧ x = 495 := by
  sorry

end number_plus_five_equals_500_l1157_115741


namespace logarithm_expression_equality_l1157_115779

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equality :
  lg (4 * Real.sqrt 2 / 7) - lg (2 / 3) + lg (7 * Real.sqrt 5) = lg 6 + 1/2 := by
  sorry

end logarithm_expression_equality_l1157_115779


namespace perfect_square_trinomial_condition_l1157_115772

/-- A trinomial of the form ax^2 + bx + c is a perfect square if and only if
    there exist real numbers p and q such that ax^2 + bx + c = (px + q)^2 -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_condition (k : ℝ) :
  IsPerfectSquareTrinomial 1 (-k) 9 → k = 6 ∨ k = -6 := by
  sorry

end perfect_square_trinomial_condition_l1157_115772


namespace triangle_area_theorem_l1157_115707

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) :
  (1/2) * x * (3*x) = 96 → x = 8 := by sorry

end triangle_area_theorem_l1157_115707


namespace function_increasing_iff_a_geq_neg_three_l1157_115720

/-- The function f(x) = x^2 + 2(a-1)x + 2 is increasing on [4, +∞) if and only if a ≥ -3 -/
theorem function_increasing_iff_a_geq_neg_three (a : ℝ) :
  (∀ x ≥ 4, Monotone (fun x => x^2 + 2*(a-1)*x + 2)) ↔ a ≥ -3 := by
  sorry

end function_increasing_iff_a_geq_neg_three_l1157_115720


namespace journey_time_calculation_l1157_115784

/-- Proves that given a journey of 240 km completed in 5 hours, 
    where the first part is traveled at 40 kmph and the second part at 60 kmph, 
    the time spent on the first part of the journey is 3 hours. -/
theorem journey_time_calculation (total_distance : ℝ) (total_time : ℝ) 
    (speed_first_part : ℝ) (speed_second_part : ℝ) 
    (h1 : total_distance = 240)
    (h2 : total_time = 5)
    (h3 : speed_first_part = 40)
    (h4 : speed_second_part = 60) :
    ∃ (first_part_time : ℝ), 
      first_part_time * speed_first_part + 
      (total_time - first_part_time) * speed_second_part = total_distance ∧
      first_part_time = 3 :=
by sorry

end journey_time_calculation_l1157_115784


namespace orange_cost_l1157_115728

theorem orange_cost (num_bananas : ℕ) (num_oranges : ℕ) (banana_cost : ℚ) (total_cost : ℚ) :
  num_bananas = 5 →
  num_oranges = 10 →
  banana_cost = 2 →
  total_cost = 25 →
  (total_cost - num_bananas * banana_cost) / num_oranges = 1.5 := by
  sorry

end orange_cost_l1157_115728


namespace mistaken_subtraction_l1157_115701

/-- Given a two-digit number where the units digit is 9, 
    if subtracting 57 from the number with the units digit mistaken as 6 results in 39,
    then the original number is 99. -/
theorem mistaken_subtraction (x : ℕ) : 
  x < 10 →  -- Ensure x is a single digit (tens place)
  (10 * x + 6) - 57 = 39 → 
  10 * x + 9 = 99 :=
by sorry

end mistaken_subtraction_l1157_115701


namespace binary_equals_base_4_l1157_115770

-- Define the binary number
def binary_num : List Bool := [true, false, true, false, true, true, true, false, true]

-- Define the base 4 number
def base_4_num : List Nat := [1, 1, 3, 1]

-- Function to convert binary to decimal
def binary_to_decimal (bin : List Bool) : Nat :=
  bin.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Function to convert base 4 to decimal
def base_4_to_decimal (b4 : List Nat) : Nat :=
  b4.reverse.enum.foldl (fun acc (i, d) => acc + d * 4^i) 0

-- Theorem statement
theorem binary_equals_base_4 :
  binary_to_decimal binary_num = base_4_to_decimal base_4_num := by
  sorry

end binary_equals_base_4_l1157_115770


namespace second_to_third_ratio_l1157_115735

/-- Given three numbers where their sum is 500, the first number is 200, and the third number is 100,
    the ratio of the second number to the third number is 2:1. -/
theorem second_to_third_ratio (a b c : ℚ) : 
  a + b + c = 500 → a = 200 → c = 100 → b / c = 2 := by
  sorry

end second_to_third_ratio_l1157_115735


namespace vector_sum_magnitude_l1157_115792

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = π/3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 1) : 
  Real.sqrt (((a.1 + 2*b.1) ^ 2) + ((a.2 + 2*b.2) ^ 2)) = 2 * Real.sqrt 3 := by
  sorry

end vector_sum_magnitude_l1157_115792


namespace pizza_toppings_combinations_l1157_115740

theorem pizza_toppings_combinations (n m : ℕ) (h1 : n = 7) (h2 : m = 4) : 
  Nat.choose n m = 35 := by
  sorry

end pizza_toppings_combinations_l1157_115740


namespace new_apples_grown_l1157_115791

/-- Given a tree with apples, calculate the number of new apples grown -/
theorem new_apples_grown
  (initial_apples : ℕ)
  (picked_apples : ℕ)
  (current_apples : ℕ)
  (h1 : initial_apples = 4)
  (h2 : picked_apples = 2)
  (h3 : current_apples = 5)
  (h4 : picked_apples ≤ initial_apples) :
  current_apples - (initial_apples - picked_apples) = 3 :=
by sorry

end new_apples_grown_l1157_115791


namespace union_of_A_and_B_l1157_115763

def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end union_of_A_and_B_l1157_115763


namespace special_function_is_x_plus_one_l1157_115789

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 0 = 1 ∧ ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2

/-- Theorem stating that the special function is x + 1 -/
theorem special_function_is_x_plus_one (f : ℝ → ℝ) (hf : special_function f) :
  ∀ x : ℝ, f x = x + 1 := by
  sorry

end special_function_is_x_plus_one_l1157_115789


namespace drivers_days_off_l1157_115780

/-- Proves that drivers get 5 days off per month given the specified conditions -/
theorem drivers_days_off 
  (num_drivers : ℕ) 
  (days_in_month : ℕ) 
  (total_cars : ℕ) 
  (maintenance_percentage : ℚ) 
  (h1 : num_drivers = 54)
  (h2 : days_in_month = 30)
  (h3 : total_cars = 60)
  (h4 : maintenance_percentage = 1/4) : 
  (days_in_month : ℚ) - (total_cars * (1 - maintenance_percentage) * days_in_month) / num_drivers = 5 := by
  sorry

end drivers_days_off_l1157_115780


namespace paving_cost_calculation_l1157_115724

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with given dimensions and rate -/
theorem paving_cost_calculation (length width rate : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : rate = 1200) :
  paving_cost length width rate = 24750 := by
  sorry

end paving_cost_calculation_l1157_115724


namespace translation_problem_l1157_115766

-- Define a translation of the complex plane
def translation (w : ℂ) : ℂ → ℂ := λ z ↦ z + w

-- Theorem statement
theorem translation_problem (w : ℂ) 
  (h : translation w (1 + 2*I) = 3 + 6*I) : 
  translation w (2 + 3*I) = 4 + 7*I := by
  sorry

end translation_problem_l1157_115766


namespace chair_distribution_count_l1157_115793

/-- The number of ways to distribute n identical objects into two groups,
    where one group must have at least a objects and the other group
    must have at least b objects. -/
def distribution_count (n a b : ℕ) : ℕ :=
  (n - a - b + 1).max 0

/-- Theorem: There are 5 ways to distribute 8 identical chairs into two groups,
    where one group (circle) must have at least 2 chairs and the other group (stack)
    must have at least 1 chair. -/
theorem chair_distribution_count : distribution_count 8 2 1 = 5 := by
  sorry

end chair_distribution_count_l1157_115793


namespace john_finish_time_l1157_115777

-- Define the start time of the first task
def start_time : Nat := 14 * 60 + 30  -- 2:30 PM in minutes since midnight

-- Define the end time of the second task
def end_second_task : Nat := 16 * 60 + 20  -- 4:20 PM in minutes since midnight

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem statement
theorem john_finish_time :
  let task_duration := (end_second_task - start_time) / 2
  let finish_time := end_second_task + 2 * task_duration
  finish_time = 18 * 60 + 10  -- 6:10 PM in minutes since midnight
  := by sorry

end john_finish_time_l1157_115777


namespace pizza_toppings_l1157_115708

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 20)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 11 := by
  sorry

end pizza_toppings_l1157_115708


namespace a_minus_b_equals_thirteen_l1157_115706

theorem a_minus_b_equals_thirteen (a b : ℝ) 
  (ha : |a| = 8)
  (hb : |b| = 5)
  (ha_pos : a > 0)
  (hb_neg : b < 0) : 
  a - b = 13 := by
sorry

end a_minus_b_equals_thirteen_l1157_115706


namespace symmetric_points_sum_l1157_115725

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- Given points A(1,a) and B(b,2) are symmetric with respect to the origin,
    prove that a + b = -3 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin 1 a b 2) : a + b = -3 := by
  sorry

end symmetric_points_sum_l1157_115725


namespace quadratic_coefficients_divisible_by_three_l1157_115757

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial (a b c : ℤ) : ℤ → ℤ := λ x ↦ a * x^2 + b * x + c

/-- The property that a polynomial is divisible by 3 for all integer inputs -/
def DivisibleByThreeForAllIntegers (P : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, P x = 3 * k

theorem quadratic_coefficients_divisible_by_three
  (a b c : ℤ)
  (h : DivisibleByThreeForAllIntegers (QuadraticPolynomial a b c)) :
  (∃ k₁ k₂ k₃ : ℤ, a = 3 * k₁ ∧ b = 3 * k₂ ∧ c = 3 * k₃) :=
sorry

end quadratic_coefficients_divisible_by_three_l1157_115757


namespace truncated_cone_radius_l1157_115700

/-- Represents a cone with its base radius -/
structure Cone :=
  (radius : ℝ)

/-- Represents a truncated cone with its smaller base radius -/
structure TruncatedCone :=
  (smallerRadius : ℝ)

/-- 
  Given three touching cones and a truncated cone sharing a common generatrix with each,
  the radius of the smaller base of the truncated cone is 6.
-/
theorem truncated_cone_radius 
  (cone1 cone2 cone3 : Cone) 
  (truncCone : TruncatedCone) 
  (h1 : cone1.radius = 23) 
  (h2 : cone2.radius = 46) 
  (h3 : cone3.radius = 69) 
  (h4 : ∃ (x y : ℝ), 
    (x^2 + y^2 = (cone1.radius + truncCone.smallerRadius)^2) ∧ 
    ((x - (cone1.radius + cone2.radius))^2 + y^2 = (cone2.radius + truncCone.smallerRadius)^2) ∧
    (x^2 + (y - (cone1.radius + cone3.radius))^2 = (cone3.radius + truncCone.smallerRadius)^2)) :
  truncCone.smallerRadius = 6 := by
  sorry

end truncated_cone_radius_l1157_115700


namespace problem_solution_l1157_115704

theorem problem_solution (y : ℝ) (h : y + Real.sqrt (y^2 - 4) + 1 / (y - Real.sqrt (y^2 - 4)) = 24) :
  y^2 + Real.sqrt (y^4 - 4) + 1 / (y^2 + Real.sqrt (y^4 - 4)) = 1369/36 := by
  sorry

end problem_solution_l1157_115704


namespace polynomial_expansion_l1157_115734

theorem polynomial_expansion (x : ℝ) : 
  (2*x^2 + 3*x + 7)*(x - 2) - (x - 2)*(x^2 - 4*x + 9) + (4*x^2 - 3*x + 1)*(x - 2)*(x - 5) = 
  5*x^3 - 26*x^2 + 35*x - 6 := by
sorry

end polynomial_expansion_l1157_115734


namespace prob_full_house_is_one_third_l1157_115719

/-- Represents the outcome of rolling five six-sided dice -/
structure DiceRoll where
  pairs : Fin 6 × Fin 6
  odd : Fin 6

/-- The probability of getting a full house after rerolling the odd die -/
def prob_full_house_after_reroll (roll : DiceRoll) : ℚ :=
  2 / 6

/-- Theorem stating the probability of getting a full house after rerolling the odd die -/
theorem prob_full_house_is_one_third (roll : DiceRoll) :
  prob_full_house_after_reroll roll = 1 / 3 := by
  sorry

#check prob_full_house_is_one_third

end prob_full_house_is_one_third_l1157_115719


namespace max_gcd_of_sequence_l1157_115760

theorem max_gcd_of_sequence (n : ℕ+) : 
  Nat.gcd (99 + n^2) (99 + (n + 1)^2) = 1 := by
  sorry

end max_gcd_of_sequence_l1157_115760


namespace oil_production_scientific_notation_l1157_115754

theorem oil_production_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 45000000 = a * (10 : ℝ) ^ n ∧ a = 4.5 ∧ n = 8 := by
  sorry

end oil_production_scientific_notation_l1157_115754


namespace unique_quadratic_solution_l1157_115753

theorem unique_quadratic_solution (a c : ℤ) : 
  (∃! x : ℝ, a * x^2 + 36 * x + c = 0) →
  a + c = 37 →
  a < c →
  (a = 12 ∧ c = 25) :=
by sorry

end unique_quadratic_solution_l1157_115753


namespace train_crossing_time_l1157_115776

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 200 ∧ train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 5 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1157_115776


namespace handshakes_in_specific_gathering_l1157_115787

/-- Represents a gathering of people with specific knowledge relationships -/
structure Gathering where
  total : Nat
  group1 : Nat
  group2 : Nat
  group2_with_connections : Nat
  group2_without_connections : Nat

/-- Calculates the number of handshakes in the gathering -/
def count_handshakes (g : Gathering) : Nat :=
  let group2_no_connections_handshakes := g.group2_without_connections * (g.total - 1)
  let group2_with_connections_handshakes := g.group2_with_connections * (g.total - 11)
  (group2_no_connections_handshakes + group2_with_connections_handshakes) / 2

/-- Theorem stating the number of handshakes in the specific gathering -/
theorem handshakes_in_specific_gathering :
  let g : Gathering := {
    total := 40,
    group1 := 25,
    group2 := 15,
    group2_with_connections := 5,
    group2_without_connections := 10
  }
  count_handshakes g = 305 := by
  sorry

#eval count_handshakes {
  total := 40,
  group1 := 25,
  group2 := 15,
  group2_with_connections := 5,
  group2_without_connections := 10
}

end handshakes_in_specific_gathering_l1157_115787


namespace point_on_x_axis_l1157_115717

/-- A point in the 2D coordinate plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The x-axis in the 2D coordinate plane -/
def xAxis : Set Point2D := {p : Point2D | p.y = 0}

/-- Theorem: A point P(x,0) lies on the x-axis -/
theorem point_on_x_axis (x : ℝ) : 
  Point2D.mk x 0 ∈ xAxis := by sorry

end point_on_x_axis_l1157_115717


namespace four_Y_three_l1157_115798

def Y (a b : ℝ) : ℝ := (a - b)^3 + 5

theorem four_Y_three : Y 4 3 = 6 := by sorry

end four_Y_three_l1157_115798


namespace min_b_value_l1157_115748

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/4) * x + 3/(4*x) - 1

def g (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 4

theorem min_b_value (b : ℝ) :
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g b x₂) →
  b ≥ 17/8 := by
  sorry

end min_b_value_l1157_115748


namespace intersection_area_formula_l1157_115782

/-- Regular octahedron with side length s -/
structure RegularOctahedron where
  s : ℝ
  s_pos : 0 < s

/-- Plane parallel to two opposite faces of the octahedron -/
structure ParallelPlane where
  distance_ratio : ℝ
  is_one_third : distance_ratio = 1/3

/-- The intersection of the plane and the octahedron forms a polygon -/
def intersection_polygon (o : RegularOctahedron) (p : ParallelPlane) : Set (ℝ × ℝ) := sorry

/-- The area of the intersection polygon -/
def intersection_area (o : RegularOctahedron) (p : ParallelPlane) : ℝ := sorry

/-- Theorem: The area of the intersection polygon is √3 * s^2 / 6 -/
theorem intersection_area_formula (o : RegularOctahedron) (p : ParallelPlane) :
  intersection_area o p = (Real.sqrt 3 * o.s^2) / 6 := by sorry

end intersection_area_formula_l1157_115782


namespace arc_length_quarter_circle_l1157_115773

/-- Given a circle with circumference 120 feet and a central angle of 90°, 
    the length of the corresponding arc is 30 feet. -/
theorem arc_length_quarter_circle (D : Real) (EF : Real) (EOF : Real) : 
  D = 120 → EOF = 90 → EF = 30 := by
  sorry

end arc_length_quarter_circle_l1157_115773


namespace product_expansion_sum_l1157_115745

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (5*x^2 - 3*x + 2)*(9 - 3*x) = a*x^3 + b*x^2 + c*x + d) →
  27*a + 9*b + 3*c + d = 0 := by
sorry

end product_expansion_sum_l1157_115745


namespace quadratic_inequality_range_l1157_115731

open Real

def quadratic_inequality (k : ℝ) (x : ℝ) : Prop :=
  2 * k * x^2 + k * x - 3/8 < 0

theorem quadratic_inequality_range :
  ∀ k : ℝ, (∀ x : ℝ, quadratic_inequality k x) ↔ k ∈ Set.Ioo (-3/2) 0 :=
sorry

end quadratic_inequality_range_l1157_115731


namespace min_value_rational_function_l1157_115743

theorem min_value_rational_function (x : ℝ) (h : x > 6) :
  (x^2 + 12*x) / (x - 6) ≥ 30 ∧
  ((x^2 + 12*x) / (x - 6) = 30 ↔ x = 12) :=
by sorry

end min_value_rational_function_l1157_115743


namespace expression_not_constant_l1157_115761

theorem expression_not_constant : 
  ∀ x y : ℝ, x ≠ 3 → x ≠ -2 → y ≠ 3 → y ≠ -2 → x ≠ y → 
  (3*x^2 + 2*x - 5) / ((x-3)*(x+2)) - (5*x - 7) / ((x-3)*(x+2)) ≠ 
  (3*y^2 + 2*y - 5) / ((y-3)*(y+2)) - (5*y - 7) / ((y-3)*(y+2)) := by
  sorry

end expression_not_constant_l1157_115761


namespace matrix_power_2023_l1157_115729

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 2, 1]

theorem matrix_power_2023 : 
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end matrix_power_2023_l1157_115729


namespace possible_values_of_a_l1157_115764

-- Define the sets A and B
def A : Set ℝ := {0, 1}
def B (a : ℝ) : Set ℝ := {x | a * x^2 + x - 1 = 0}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : A ⊇ B a → a = 0 ∨ a < -1/4 := by
  sorry

end possible_values_of_a_l1157_115764


namespace family_road_trip_l1157_115795

/-- A theorem about a family's road trip with constant speed -/
theorem family_road_trip 
  (total_time : ℝ) 
  (first_part_distance : ℝ) 
  (first_part_time : ℝ) 
  (h1 : total_time = 4) 
  (h2 : first_part_distance = 100) 
  (h3 : first_part_time = 1) :
  let speed := first_part_distance / first_part_time
  let remaining_time := total_time - first_part_time
  remaining_time * speed = 300 := by
  sorry

#check family_road_trip

end family_road_trip_l1157_115795


namespace largest_multiple_of_11_under_100_l1157_115723

theorem largest_multiple_of_11_under_100 : ∃ n : ℕ, n * 11 = 99 ∧ n * 11 < 100 ∧ ∀ m : ℕ, m * 11 < 100 → m * 11 ≤ 99 := by
  sorry

end largest_multiple_of_11_under_100_l1157_115723


namespace hyperbola_equation_l1157_115703

theorem hyperbola_equation (a b p x₀ : ℝ) : 
  a > 0 → b > 0 → p > 0 →
  (b / a = 2) →
  (p / 2 = 4 / 3) →
  (x₀ = 3) →
  (16 = 2 * p * x₀) →
  (9 / a^2 - 16 / b^2 = 1) →
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 5 - y^2 / 20 = 1) :=
by sorry

end hyperbola_equation_l1157_115703


namespace f_mapping_result_l1157_115771

def A : Set (ℝ × ℝ) := Set.univ

def B : Set (ℝ × ℝ) := Set.univ

def f : (ℝ × ℝ) → (ℝ × ℝ) := λ (x, y) ↦ (x - y, x + y)

theorem f_mapping_result : f (-1, 2) = (-3, 1) := by
  sorry

end f_mapping_result_l1157_115771


namespace ellipse_dot_product_range_l1157_115796

/-- The ellipse in the first quadrant -/
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1 ∧ x > 0 ∧ y > 0

/-- The dot product of OP and PF -/
def dot_product (x y : ℝ) : ℝ := x*(3-x) - y^2

theorem ellipse_dot_product_range :
  ∀ x y : ℝ, ellipse x y → -16 < dot_product x y ∧ dot_product x y ≤ -39/4 := by
  sorry

end ellipse_dot_product_range_l1157_115796


namespace tan_product_30_15_l1157_115769

theorem tan_product_30_15 :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end tan_product_30_15_l1157_115769


namespace range_of_positives_in_K_l1157_115751

/-- Definition of the list K -/
def list_K : List ℤ := List.range 40 |>.map (fun i => -25 + 3 * i)

/-- The range of positive integers in list K -/
def positive_range (L : List ℤ) : ℤ :=
  let positives := L.filter (· > 0)
  positives.maximum.getD 0 - positives.minimum.getD 0

/-- Theorem: The range of positive integers in list K is 90 -/
theorem range_of_positives_in_K : positive_range list_K = 90 := by
  sorry

end range_of_positives_in_K_l1157_115751


namespace thabo_hardcover_books_l1157_115702

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (bc : BookCollection) : Prop :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction = 500 ∧
  bc.paperback_nonfiction = bc.hardcover_nonfiction + 30 ∧
  bc.paperback_fiction = 3 * bc.paperback_nonfiction

theorem thabo_hardcover_books (bc : BookCollection) 
  (h : is_valid_collection bc) : bc.hardcover_nonfiction = 76 := by
  sorry

end thabo_hardcover_books_l1157_115702


namespace sara_balloons_l1157_115788

/-- The number of red balloons Sara has left after giving some away -/
def balloons_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Sara is left with 7 red balloons -/
theorem sara_balloons : balloons_left 31 24 = 7 := by
  sorry

end sara_balloons_l1157_115788


namespace geometric_series_sum_l1157_115759

def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a : ℚ := 1
  let r : ℚ := 1/4
  let n : ℕ := 5
  geometricSum a r n = 341/256 := by
sorry

end geometric_series_sum_l1157_115759


namespace randys_brother_biscuits_l1157_115799

/-- The number of biscuits Randy's brother ate -/
def biscuits_eaten (initial : ℕ) (from_father : ℕ) (from_mother : ℕ) (remaining : ℕ) : ℕ :=
  initial + from_father + from_mother - remaining

/-- Theorem stating the number of biscuits Randy's brother ate -/
theorem randys_brother_biscuits :
  biscuits_eaten 32 13 15 40 = 20 := by
  sorry

end randys_brother_biscuits_l1157_115799


namespace cells_covered_by_two_squares_l1157_115713

/-- Represents a square on a graph paper --/
structure Square where
  size : ℕ
  position : ℕ × ℕ

/-- Represents the configuration of squares on the graph paper --/
def SquareConfiguration := List Square

/-- Counts the number of cells covered by exactly two squares in a given configuration --/
def countCellsCoveredByTwoSquares (config : SquareConfiguration) : ℕ :=
  sorry

/-- The specific configuration of squares from the problem --/
def problemConfiguration : SquareConfiguration :=
  [{ size := 5, position := (0, 0) },
   { size := 5, position := (3, 0) },
   { size := 5, position := (3, 3) }]

theorem cells_covered_by_two_squares :
  countCellsCoveredByTwoSquares problemConfiguration = 13 := by
  sorry

end cells_covered_by_two_squares_l1157_115713


namespace expression_value_l1157_115730

theorem expression_value : 
  |1 - Real.sqrt 3| - 2 * Real.sin (π / 3) + (π - 2023) ^ 0 = 0 := by sorry

end expression_value_l1157_115730


namespace pauls_cousin_score_l1157_115778

/-- Given Paul's score and the total score of Paul and his cousin, 
    calculate Paul's cousin's score. -/
theorem pauls_cousin_score (paul_score total_score : ℕ) 
  (h1 : paul_score = 3103)
  (h2 : total_score = 5816) :
  total_score - paul_score = 2713 := by
  sorry

end pauls_cousin_score_l1157_115778


namespace problem_statement_l1157_115722

theorem problem_statement (a b k : ℕ+) (h : (a.val^2 - 1 - b.val^2) / (a.val * b.val - 1) = k.val) : k = 5 := by
  sorry

end problem_statement_l1157_115722


namespace system_solution_l1157_115709

theorem system_solution :
  ∃ (k m : ℚ),
    (3 * k - 4) / (k + 7) = 2/5 ∧
    2 * m + 5 * k = 14 ∧
    k = 34/13 ∧
    m = 6/13 := by
  sorry

end system_solution_l1157_115709


namespace proposition_B_is_false_l1157_115750

-- Define propositions as boolean variables
variable (p q : Prop)

-- Define the proposition B
def proposition_B (p q : Prop) : Prop :=
  (¬p ∧ ¬q) → (¬p ∧ ¬q)

-- Theorem stating that proposition B is false
theorem proposition_B_is_false :
  ∃ p q : Prop, ¬(proposition_B p q) :=
sorry

end proposition_B_is_false_l1157_115750


namespace max_hearts_desire_desire_fulfilled_l1157_115716

/-- Represents a four-digit natural number M = 1000a + 100b + 10c + d -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h1 : 1 ≤ a ∧ a ≤ 9
  h2 : 1 ≤ b ∧ b ≤ 9
  h3 : 1 ≤ c ∧ c ≤ 9
  h4 : 1 ≤ d ∧ d ≤ 9
  h5 : c > d

/-- Calculates the value of M given its digits -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Checks if a number is a "heart's desire" and "desire fulfilled" number -/
def isHeartsDesireAndDesireFulfilled (n : FourDigitNumber) : Prop :=
  (10 * n.b + n.c) / (n.a + n.d) = 11

/-- Calculates F(M) -/
def F (n : FourDigitNumber) : Nat :=
  10 * (n.a + n.b) + 3 * n.c

/-- Main theorem statement -/
theorem max_hearts_desire_desire_fulfilled :
  ∃ (M : FourDigitNumber),
    isHeartsDesireAndDesireFulfilled M ∧
    F M % 7 = 0 ∧
    M.value = 5883 ∧
    (∀ (N : FourDigitNumber),
      isHeartsDesireAndDesireFulfilled N ∧
      F N % 7 = 0 →
      N.value ≤ M.value) := by
  sorry

end max_hearts_desire_desire_fulfilled_l1157_115716


namespace no_real_roots_l1157_115785

theorem no_real_roots (A B : ℝ) : 
  (∀ x y : ℝ, x^2 + x*y + y = A ∧ y / (y - x) = B → False) ↔ A = 2 ∧ B = 2/3 :=
by sorry

end no_real_roots_l1157_115785


namespace complex_equation_sum_of_squares_l1157_115755

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the theorem
theorem complex_equation_sum_of_squares 
  (a b : ℝ) 
  (h : (a - 2 * i) * i = b - i) : 
  a^2 + b^2 = 5 := by
  sorry

end complex_equation_sum_of_squares_l1157_115755


namespace store_profit_theorem_l1157_115746

/-- Represents the selling price and number of items sold -/
structure SaleInfo where
  price : ℝ
  quantity : ℝ

/-- The profit function given the cost, price, and quantity -/
def profit (cost : ℝ) (info : SaleInfo) : ℝ :=
  (info.price - cost) * info.quantity

/-- The demand function given the base price, base quantity, and price sensitivity -/
def demand (basePrice baseQuantity priceSensitivity : ℝ) (price : ℝ) : ℝ :=
  baseQuantity - priceSensitivity * (price - basePrice)

theorem store_profit_theorem (cost basePrice baseQuantity priceSensitivity targetProfit : ℝ) :
  cost = 40 ∧
  basePrice = 50 ∧
  baseQuantity = 150 ∧
  priceSensitivity = 5 ∧
  targetProfit = 1500 →
  ∃ (info1 info2 : SaleInfo),
    info1.price = 50 ∧
    info1.quantity = 150 ∧
    info2.price = 70 ∧
    info2.quantity = 50 ∧
    profit cost info1 = targetProfit ∧
    profit cost info2 = targetProfit ∧
    info1.quantity = demand basePrice baseQuantity priceSensitivity info1.price ∧
    info2.quantity = demand basePrice baseQuantity priceSensitivity info2.price ∧
    ∀ (info : SaleInfo),
      profit cost info = targetProfit ∧
      info.quantity = demand basePrice baseQuantity priceSensitivity info.price →
      (info = info1 ∨ info = info2) := by
  sorry


end store_profit_theorem_l1157_115746


namespace dvd_rental_cost_l1157_115783

theorem dvd_rental_cost (total_cost : ℝ) (num_dvds : ℕ) (cost_per_dvd : ℝ) 
  (h1 : total_cost = 4.8)
  (h2 : num_dvds = 4)
  (h3 : cost_per_dvd = total_cost / num_dvds) :
  cost_per_dvd = 1.2 := by
  sorry

end dvd_rental_cost_l1157_115783


namespace total_birds_on_fence_l1157_115705

def initial_birds : ℕ := 4
def additional_birds : ℕ := 6

theorem total_birds_on_fence :
  initial_birds + additional_birds = 10 := by sorry

end total_birds_on_fence_l1157_115705


namespace sundae_price_l1157_115749

/-- Given a caterer's order of ice-cream bars and sundaes, calculate the price of each sundae. -/
theorem sundae_price
  (ice_cream_bars : ℕ)
  (sundaes : ℕ)
  (total_price : ℚ)
  (ice_cream_bar_price : ℚ)
  (h1 : ice_cream_bars = 125)
  (h2 : sundaes = 125)
  (h3 : total_price = 250)
  (h4 : ice_cream_bar_price = 0.6) :
  (total_price - ice_cream_bars * ice_cream_bar_price) / sundaes = 1.4 := by
  sorry

#check sundae_price

end sundae_price_l1157_115749


namespace abs_neg_2022_l1157_115732

theorem abs_neg_2022 : |(-2022 : ℤ)| = 2022 := by
  sorry

end abs_neg_2022_l1157_115732


namespace triangle_area_is_four_l1157_115774

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- The area of a triangle given two sides and the sine of the included angle. -/
def triangleArea (s1 s2 sinAngle : ℝ) : ℝ :=
  0.5 * s1 * s2 * sinAngle

/-- The theorem stating that the area of the given triangle is 4. -/
theorem triangle_area_is_four (t : Triangle) 
    (ha : t.a = 2)
    (hc : t.c = 5)
    (hcosB : Real.cos t.angleB = 3/5) : 
    triangleArea t.a t.c (Real.sin t.angleB) = 4 := by
  sorry

end triangle_area_is_four_l1157_115774


namespace tv_sales_effect_l1157_115733

theorem tv_sales_effect (price_reduction : Real) (sales_increase : Real) :
  price_reduction = 0.18 →
  sales_increase = 0.72 →
  let new_price_factor := 1 - price_reduction
  let new_sales_factor := 1 + sales_increase
  let net_effect := new_price_factor * new_sales_factor - 1
  net_effect * 100 = 41.04 := by
  sorry

end tv_sales_effect_l1157_115733


namespace simplify_fraction_product_l1157_115765

theorem simplify_fraction_product : 8 * (15 / 14) * (-49 / 45) = -28 / 3 := by
  sorry

end simplify_fraction_product_l1157_115765
