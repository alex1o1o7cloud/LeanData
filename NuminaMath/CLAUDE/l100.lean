import Mathlib

namespace davids_biology_marks_l100_10048

/-- Given David's marks in four subjects and his average marks across five subjects,
    proves that his marks in Biology are 90. -/
theorem davids_biology_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (average : ℚ)
  (h1 : english = 74)
  (h2 : mathematics = 65)
  (h3 : physics = 82)
  (h4 : chemistry = 67)
  (h5 : average = 75.6)
  (h6 : average = (english + mathematics + physics + chemistry + biology) / 5) :
  biology = 90 :=
by
  sorry


end davids_biology_marks_l100_10048


namespace sqrt5_irrational_l100_10023

theorem sqrt5_irrational : Irrational (Real.sqrt 5) := by
  sorry

end sqrt5_irrational_l100_10023


namespace solve_equation_l100_10064

theorem solve_equation (x : ℝ) : 3639 + 11.95 - x = 3054 → x = 596.95 := by
  sorry

end solve_equation_l100_10064


namespace debby_text_messages_l100_10008

theorem debby_text_messages 
  (total_messages : ℕ) 
  (before_noon_messages : ℕ) 
  (h1 : total_messages = 39) 
  (h2 : before_noon_messages = 21) : 
  total_messages - before_noon_messages = 18 := by
sorry

end debby_text_messages_l100_10008


namespace union_complement_equality_l100_10016

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3}

-- Define set M
def M : Set Nat := {0, 1, 2}

-- Define set N
def N : Set Nat := {0, 2, 3}

-- Theorem statement
theorem union_complement_equality : M ∪ (U \ N) = {0, 1, 2} := by sorry

end union_complement_equality_l100_10016


namespace smallest_winning_number_l100_10070

def B (x : ℕ) : ℕ := 3 * x

def S (x : ℕ) : ℕ := x + 100

def game_sequence (N : ℕ) : ℕ := B (S (B (S (B N))))

theorem smallest_winning_number :
  ∀ N : ℕ, 0 ≤ N ∧ N ≤ 1999 →
    (∀ M : ℕ, 0 ≤ M ∧ M < N → S (B (S (B M))) ≤ 2000) ∧
    2000 < game_sequence N ∧
    S (B (S (B N))) ≤ 2000 →
    N = 26 :=
sorry

end smallest_winning_number_l100_10070


namespace rational_abs_four_and_self_reciprocal_l100_10086

theorem rational_abs_four_and_self_reciprocal :
  (∀ x : ℚ, |x| = 4 ↔ x = -4 ∨ x = 4) ∧
  (∀ x : ℝ, x⁻¹ = x ↔ x = -1 ∨ x = 1) := by sorry

end rational_abs_four_and_self_reciprocal_l100_10086


namespace sequence_problem_l100_10094

theorem sequence_problem (a b : ℝ) : 
  (∃ r : ℝ, 10 * r = a ∧ a * r = 1/2) →  -- geometric sequence condition
  (∃ d : ℝ, a + d = 5 ∧ 5 + d = b) →    -- arithmetic sequence condition
  a = Real.sqrt 5 ∧ b = 10 - Real.sqrt 5 := by
sorry


end sequence_problem_l100_10094


namespace tammy_mountain_climb_l100_10045

/-- Tammy's mountain climbing problem -/
theorem tammy_mountain_climb 
  (total_time : ℝ) 
  (second_day_speed : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) :
  total_time = 14 →
  second_day_speed = 4 →
  speed_difference = 0.5 →
  time_difference = 2 →
  ∃ (first_day_time second_day_time : ℝ),
    first_day_time + second_day_time = total_time ∧
    second_day_time = first_day_time - time_difference ∧
    ∃ (first_day_speed : ℝ),
      first_day_speed = second_day_speed - speed_difference ∧
      first_day_speed * first_day_time + second_day_speed * second_day_time = 52 :=
by sorry

end tammy_mountain_climb_l100_10045


namespace number_of_products_l100_10066

/-- Prove that the number of products is 20 given the fixed cost, marginal cost, and total cost. -/
theorem number_of_products (fixed_cost marginal_cost total_cost : ℚ)
  (h1 : fixed_cost = 12000)
  (h2 : marginal_cost = 200)
  (h3 : total_cost = 16000)
  (h4 : total_cost = fixed_cost + marginal_cost * n) :
  n = 20 := by
  sorry

end number_of_products_l100_10066


namespace los_angeles_women_ratio_l100_10021

/-- The ratio of women to the total population in Los Angeles -/
def women_ratio (total_population women_in_retail : ℕ) (retail_fraction : ℚ) : ℚ :=
  (women_in_retail / retail_fraction) / total_population

/-- Proof that the ratio of women to the total population in Los Angeles is 1/2 -/
theorem los_angeles_women_ratio :
  women_ratio 6000000 1000000 (1/3) = 1/2 := by
  sorry

end los_angeles_women_ratio_l100_10021


namespace correct_equation_l100_10039

/-- Represents a bookstore's novel purchases -/
structure NovelPurchases where
  first_cost : ℝ
  second_cost : ℝ
  quantity_difference : ℕ
  first_quantity : ℕ

/-- The equation representing equal cost per copy for both purchases -/
def equal_cost_equation (p : NovelPurchases) : Prop :=
  p.first_cost / p.first_quantity = p.second_cost / (p.first_quantity + p.quantity_difference)

/-- Theorem stating that the given equation correctly represents the situation -/
theorem correct_equation (p : NovelPurchases) 
  (h1 : p.first_cost = 2000)
  (h2 : p.second_cost = 3000)
  (h3 : p.quantity_difference = 50) :
  equal_cost_equation p ↔ p.first_cost / p.first_quantity = p.second_cost / (p.first_quantity + p.quantity_difference) :=
sorry

end correct_equation_l100_10039


namespace selling_price_range_l100_10078

/-- Represents the daily sales revenue as a function of the selling price --/
def revenue (x : ℝ) : ℝ := x * (45 - 3 * (x - 15))

/-- The minimum selling price in yuan --/
def min_price : ℝ := 15

/-- The theorem stating the range of selling prices that generate over 600 yuan in daily revenue --/
theorem selling_price_range :
  {x : ℝ | revenue x > 600 ∧ x ≥ min_price} = Set.Icc 15 20 := by sorry

end selling_price_range_l100_10078


namespace complex_equation_solution_l100_10069

theorem complex_equation_solution (z : ℂ) 
  (h : 10 * Complex.normSq z = 2 * Complex.normSq (z + 3) + Complex.normSq (z^2 + 16) + 40) : 
  z + 9 / z = -3 / 17 := by
  sorry

end complex_equation_solution_l100_10069


namespace postcard_width_is_six_l100_10024

/-- Represents a rectangular postcard -/
structure Postcard where
  width : ℝ
  height : ℝ

/-- The perimeter of a rectangular postcard -/
def perimeter (p : Postcard) : ℝ := 2 * (p.width + p.height)

theorem postcard_width_is_six :
  ∀ p : Postcard,
  p.height = 4 →
  perimeter p = 20 →
  p.width = 6 := by
sorry

end postcard_width_is_six_l100_10024


namespace poster_board_side_length_l100_10080

/-- Prove that a square poster board that can fit 24 rectangular cards
    measuring 2 inches by 3 inches has a side length of 1 foot. -/
theorem poster_board_side_length :
  ∀ (side_length : ℝ),
  (side_length * side_length = 24 * 2 * 3) →
  (side_length / 12 = 1) :=
by
  sorry

end poster_board_side_length_l100_10080


namespace carpool_arrangement_count_l100_10090

def num_students : ℕ := 8
def num_grades : ℕ := 4
def students_per_grade : ℕ := 2
def car_capacity : ℕ := 4

def has_twin_sisters : Prop := true

theorem carpool_arrangement_count : ℕ := by
  sorry

end carpool_arrangement_count_l100_10090


namespace min_value_of_expression_lower_bound_achievable_l100_10074

theorem min_value_of_expression (x : ℝ) (h : x > 1) :
  2*x + 7/(x-1) ≥ 2*Real.sqrt 14 + 2 :=
sorry

theorem lower_bound_achievable :
  ∃ x > 1, 2*x + 7/(x-1) = 2*Real.sqrt 14 + 2 :=
sorry

end min_value_of_expression_lower_bound_achievable_l100_10074


namespace circle_radius_condition_l100_10035

theorem circle_radius_condition (x y c : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 - 6*y + c = 0 → (x + 4)^2 + (y - 3)^2 = 25) → c = 0 := by
  sorry

end circle_radius_condition_l100_10035


namespace train_speed_l100_10020

/-- A train journey with two segments and a given average speed -/
structure TrainJourney where
  x : ℝ  -- distance of the first segment
  V : ℝ  -- speed of the train in the first segment
  avg_speed : ℝ  -- average speed for the entire journey

/-- The train journey satisfies the given conditions -/
def valid_journey (j : TrainJourney) : Prop :=
  j.x > 0 ∧ j.V > 0 ∧ j.avg_speed = 16 ∧
  (j.x / j.V + (2 * j.x) / 20) = (3 * j.x) / j.avg_speed

theorem train_speed (j : TrainJourney) (h : valid_journey j) : j.V = 40 / 7 := by
  sorry

end train_speed_l100_10020


namespace ceiling_squared_negative_fraction_l100_10053

theorem ceiling_squared_negative_fraction :
  ⌈(-7/4)^2⌉ = 4 := by sorry

end ceiling_squared_negative_fraction_l100_10053


namespace min_packs_for_120_cans_l100_10005

/-- Represents the number of cans in each pack size -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans in a given pack size -/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | PackSize.small => 8
  | PackSize.medium => 16
  | PackSize.large => 32

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a pack combination -/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- Theorem: The minimum number of packs to buy 120 cans is 5 -/
theorem min_packs_for_120_cans :
  ∃ (c : PackCombination),
    totalCans c = 120 ∧
    totalPacks c = 5 ∧
    (∀ (c' : PackCombination), totalCans c' = 120 → totalPacks c' ≥ 5) :=
by
  sorry

end min_packs_for_120_cans_l100_10005


namespace cubic_root_sum_cubes_l100_10030

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (6 * a^3 - 803 * a + 1606 = 0) → 
  (6 * b^3 - 803 * b + 1606 = 0) → 
  (6 * c^3 - 803 * c + 1606 = 0) → 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 803 := by
sorry

end cubic_root_sum_cubes_l100_10030


namespace reciprocal_equation_l100_10042

theorem reciprocal_equation (x : ℝ) : 
  (((5 * x - 1) / 6 - 2)⁻¹ = 3) → x = 3 := by
  sorry

end reciprocal_equation_l100_10042


namespace diophantine_equation_solution_l100_10092

theorem diophantine_equation_solution (x y : ℕ) (h : 65 * x - 43 * y = 2) :
  ∃ t : ℤ, t ≤ 0 ∧ x = 4 - 43 * t ∧ y = 6 - 65 * t := by
  sorry

end diophantine_equation_solution_l100_10092


namespace pen_ratio_problem_l100_10027

theorem pen_ratio_problem (blue_pens green_pens : ℕ) : 
  (blue_pens : ℚ) / green_pens = 4 / 3 →
  blue_pens = 16 →
  green_pens = 12 := by
sorry

end pen_ratio_problem_l100_10027


namespace arithmetic_expression_equality_l100_10038

theorem arithmetic_expression_equality : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end arithmetic_expression_equality_l100_10038


namespace michael_spending_l100_10015

def fair_spending (initial_amount snack_cost : ℕ) : ℕ :=
  let game_cost := 3 * snack_cost
  let total_spent := snack_cost + game_cost
  initial_amount - total_spent

theorem michael_spending :
  fair_spending 80 20 = 0 := by
  sorry

end michael_spending_l100_10015


namespace switch_pairs_relation_l100_10001

/-- Represents a row in the sequence --/
structure Row where
  switchPairs : ℕ
  oddBlocks : ℕ

/-- The relationship between switch pairs and odd blocks in a row --/
axiom switch_pairs_odd_blocks (r : Row) : r.switchPairs = 2 * r.oddBlocks

/-- The existence of at least one switch pair above each odd block --/
axiom switch_pair_above_odd_block (rn : Row) (rn_minus_1 : Row) :
  rn.oddBlocks ≤ rn_minus_1.switchPairs

/-- Theorem: The number of switch pairs in row n is at most twice 
    the number of switch pairs in row n-1 --/
theorem switch_pairs_relation (rn : Row) (rn_minus_1 : Row) :
  rn.switchPairs ≤ 2 * rn_minus_1.switchPairs := by
  sorry

end switch_pairs_relation_l100_10001


namespace sally_quarters_remaining_l100_10049

def initial_quarters : ℕ := 760
def first_purchase : ℕ := 418
def second_purchase : ℕ := 192

theorem sally_quarters_remaining :
  initial_quarters - first_purchase - second_purchase = 150 := by sorry

end sally_quarters_remaining_l100_10049


namespace random_events_l100_10051

-- Define a type for the events
inductive Event
  | addition
  | subtraction
  | multiplication
  | division

-- Define a function to check if an event is random
def is_random (e : Event) : Prop :=
  match e with
  | Event.addition => ∃ (a b : ℝ), a * b < 0 ∧ a + b < 0
  | Event.subtraction => ∃ (a b : ℝ), a * b < 0 ∧ a - b > 0
  | Event.multiplication => false
  | Event.division => true

-- Theorem stating which events are random
theorem random_events :
  (is_random Event.addition) ∧
  (is_random Event.subtraction) ∧
  (¬ is_random Event.multiplication) ∧
  (¬ is_random Event.division) := by
  sorry

end random_events_l100_10051


namespace sum_of_coefficients_is_three_l100_10043

/-- Given two linear functions f and g defined by real parameters A and B,
    proves that if f(g(x)) - g(f(x)) = 2(B - A) and A ≠ B, then A + B = 3. -/
theorem sum_of_coefficients_is_three
  (A B : ℝ)
  (hne : A ≠ B)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x + B)
  (hg : ∀ x, g x = B * x + A)
  (h : ∀ x, f (g x) - g (f x) = 2 * (B - A)) :
  A + B = 3 := by
sorry

end sum_of_coefficients_is_three_l100_10043


namespace closest_fraction_l100_10059

def medals_won : ℚ := 35 / 225

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (x : ℚ), x ∈ options ∧ 
  ∀ (y : ℚ), y ∈ options → |x - medals_won| ≤ |y - medals_won| ∧
  x = 1/6 :=
sorry

end closest_fraction_l100_10059


namespace milk_price_increase_percentage_l100_10072

def lowest_price : ℝ := 16
def highest_price : ℝ := 22

theorem milk_price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 37.5 := by
  sorry

end milk_price_increase_percentage_l100_10072


namespace ellipse_eccentricity_a_values_l100_10075

theorem ellipse_eccentricity_a_values (a : ℝ) :
  a > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 6 = 1) →
  (let e := Real.sqrt 6 / 6
   ∃ b : ℝ, e^2 = 1 - (min a (Real.sqrt 6))^2 / (max a (Real.sqrt 6))^2) →
  a = 6 * Real.sqrt 5 / 5 ∨ a = Real.sqrt 5 := by
  sorry

end ellipse_eccentricity_a_values_l100_10075


namespace quadratic_roots_condition_l100_10032

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 1) * x^2 + 4 * x + 1 = 0 ∧ 
   (k - 1) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 5 ∧ k ≠ 1) :=
sorry

end quadratic_roots_condition_l100_10032


namespace max_value_constraint_l100_10022

theorem max_value_constraint (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  2*x*y*Real.sqrt 6 + 9*y*z ≤ Real.sqrt 87 := by
  sorry

end max_value_constraint_l100_10022


namespace cookie_difference_l100_10029

theorem cookie_difference (alyssa_cookies aiyanna_cookies : ℕ) 
  (h1 : alyssa_cookies = 129) (h2 : aiyanna_cookies = 140) : 
  aiyanna_cookies - alyssa_cookies = 11 := by
  sorry

end cookie_difference_l100_10029


namespace student_count_l100_10055

theorem student_count (ratio : ℝ) (teachers : ℕ) (h1 : ratio = 27.5) (h2 : teachers = 42) :
  ↑teachers * ratio = 1155 := by
  sorry

end student_count_l100_10055


namespace max_value_x_plus_y_l100_10019

theorem max_value_x_plus_y :
  ∃ (x y : ℝ),
    (2 * Real.sin x - 1) * (2 * Real.cos y - Real.sqrt 3) = 0 ∧
    x ∈ Set.Icc 0 (3 * Real.pi / 2) ∧
    y ∈ Set.Icc Real.pi (2 * Real.pi) ∧
    ∀ (x' y' : ℝ),
      (2 * Real.sin x' - 1) * (2 * Real.cos y' - Real.sqrt 3) = 0 →
      x' ∈ Set.Icc 0 (3 * Real.pi / 2) →
      y' ∈ Set.Icc Real.pi (2 * Real.pi) →
      x + y ≥ x' + y' ∧
    x + y = 8 * Real.pi / 3 :=
by sorry

end max_value_x_plus_y_l100_10019


namespace binomial_512_512_l100_10093

theorem binomial_512_512 : Nat.choose 512 512 = 1 := by
  sorry

end binomial_512_512_l100_10093


namespace geometric_sequence_properties_l100_10062

/-- A geometric sequence with common ratio q -/
def geometric_sequence (q : ℝ) : ℕ → ℝ := fun n ↦ q ^ (n - 1)

theorem geometric_sequence_properties (q : ℝ) (h_q : 0 < q ∧ q < 1) :
  let a := geometric_sequence q
  (∀ n : ℕ, a (n + 1) < a n) ∧
  (∃ k : ℕ+, a (k + 1) = (a k + a (k + 2)) / 2 → q = (1 - Real.sqrt 5) / 2) :=
by sorry

end geometric_sequence_properties_l100_10062


namespace f_explicit_function_l100_10028

-- Define the function f
def f : ℝ → ℝ := fun x => x^2 - 1

-- State the theorem
theorem f_explicit_function (x : ℝ) (h : x ≥ 0) : 
  f (Real.sqrt x + 1) = x + 2 * Real.sqrt x ↔ (∀ y ≥ 1, f y = y^2 - 1) := by
  sorry

end f_explicit_function_l100_10028


namespace max_real_axis_length_l100_10031

/-- Represents a hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  /-- The asymptotes of the hyperbola are of the form 2x ± y = 0 -/
  asymptotes : Unit
  /-- The hyperbola passes through the intersection of two lines -/
  intersection_point : ℝ × ℝ
  /-- The parameter t determines the intersection point -/
  t : ℝ
  /-- The intersection point satisfies the equations of both lines -/
  satisfies_line1 : intersection_point.1 + intersection_point.2 = 3
  satisfies_line2 : 2 * intersection_point.1 - intersection_point.2 = -3 * t
  /-- The parameter t is within the specified range -/
  t_range : -2 ≤ t ∧ t ≤ 5

/-- The length of the real axis of the hyperbola -/
def real_axis_length (h : Hyperbola) : ℝ := sorry

/-- Theorem stating the maximum possible length of the real axis -/
theorem max_real_axis_length (h : Hyperbola) : 
  real_axis_length h ≤ 4 * Real.sqrt 3 := by sorry

end max_real_axis_length_l100_10031


namespace speaking_orders_eq_552_l100_10085

/-- The number of students in the class -/
def total_students : ℕ := 7

/-- The number of students to be selected for speaking -/
def speakers : ℕ := 4

/-- Function to calculate the number of different speaking orders -/
def speaking_orders : ℕ :=
  let only_one_ab := 2 * (total_students - 2).choose (speakers - 1) * (speakers).factorial
  let both_ab := (total_students - 3).choose (speakers - 2) * 2 * 6
  only_one_ab + both_ab

/-- Theorem stating that the number of different speaking orders is 552 -/
theorem speaking_orders_eq_552 : speaking_orders = 552 := by
  sorry

end speaking_orders_eq_552_l100_10085


namespace perpendicular_line_correct_l100_10046

/-- The slope of the given line x - 2y + 3 = 0 -/
def m₁ : ℚ := 1 / 2

/-- The point P through which the perpendicular line passes -/
def P : ℚ × ℚ := (-1, 3)

/-- The equation of the perpendicular line in the form ax + by + c = 0 -/
def perpendicular_line (x y : ℚ) : Prop := 2 * x + y - 1 = 0

theorem perpendicular_line_correct :
  /- The line passes through point P -/
  perpendicular_line P.1 P.2 ∧
  /- The line is perpendicular to x - 2y + 3 = 0 -/
  (∃ m₂ : ℚ, m₂ * m₁ = -1 ∧
    ∀ x y : ℚ, perpendicular_line x y ↔ y - P.2 = m₂ * (x - P.1)) :=
sorry

end perpendicular_line_correct_l100_10046


namespace intersection_of_A_and_B_l100_10009

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end intersection_of_A_and_B_l100_10009


namespace brian_white_stones_l100_10097

/-- Represents Brian's stone collection -/
structure StoneCollection where
  white : ℕ
  black : ℕ
  grey : ℕ
  green : ℕ
  red : ℕ
  blue : ℕ

/-- Conditions of Brian's stone collection -/
def BrianCollection : StoneCollection → Prop := fun c =>
  c.white + c.black = 100 ∧
  c.grey + c.green = 100 ∧
  c.red + c.blue = 130 ∧
  c.white + c.black + c.grey + c.green + c.red + c.blue = 330 ∧
  c.white > c.black ∧
  c.white = c.grey ∧
  c.black = c.green ∧
  3 * c.blue = 2 * c.red ∧
  2 * (c.white + c.grey) = c.red

theorem brian_white_stones (c : StoneCollection) 
  (h : BrianCollection c) : c.white = 78 := by
  sorry

end brian_white_stones_l100_10097


namespace product_of_odd_is_even_correct_propositions_count_l100_10061

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The product of two odd functions is even -/
theorem product_of_odd_is_even (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsOdd g) :
    IsEven (fun x ↦ f x * g x) := by
  sorry

/-- There are exactly two correct propositions among the original, converse, negation, and contrapositive -/
theorem correct_propositions_count : ℕ := by
  sorry

end product_of_odd_is_even_correct_propositions_count_l100_10061


namespace volume_difference_rectangular_prisms_volume_difference_specific_bowls_l100_10050

/-- The volume difference between two rectangular prisms with the same width and length
    but different heights is equal to the product of the width, length, and the difference in heights. -/
theorem volume_difference_rectangular_prisms
  (w : ℝ) (l : ℝ) (h₁ : ℝ) (h₂ : ℝ)
  (hw : w > 0) (hl : l > 0) (hh₁ : h₁ > 0) (hh₂ : h₂ > 0) :
  w * l * h₁ - w * l * h₂ = w * l * (h₁ - h₂) :=
by sorry

/-- The volume difference between two specific bowls -/
theorem volume_difference_specific_bowls :
  (16 : ℝ) * 14 * 9 - (16 : ℝ) * 14 * 4 = 1120 :=
by sorry

end volume_difference_rectangular_prisms_volume_difference_specific_bowls_l100_10050


namespace arithmetic_sequence_ratio_l100_10091

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms
def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (a 1 + a n) * n / 2

-- Theorem statement
theorem arithmetic_sequence_ratio 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_a3 : a 3 = 9) 
  (h_a5 : a 5 = 5) : 
  S a 9 / S a 5 = 1/2 := by
sorry

end arithmetic_sequence_ratio_l100_10091


namespace max_value_sqrt_sum_l100_10011

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 8) :
  Real.sqrt (4 * x + 1) + Real.sqrt (4 * y + 1) + Real.sqrt (4 * z + 1) ≤ 3 * Real.sqrt (35 / 3) ∧
  ∃ x y z, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 8 ∧
    Real.sqrt (4 * x + 1) + Real.sqrt (4 * y + 1) + Real.sqrt (4 * z + 1) = 3 * Real.sqrt (35 / 3) :=
by
  sorry

end max_value_sqrt_sum_l100_10011


namespace day_relationship_l100_10098

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to determine the day of the week for a given day number -/
def dayOfWeek (dayNumber : ℕ) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

/-- Theorem stating the relationship between days in different years -/
theorem day_relationship (N : ℕ) :
  dayOfWeek 290 = DayOfWeek.Wednesday →
  dayOfWeek 210 = DayOfWeek.Wednesday →
  dayOfWeek 110 = DayOfWeek.Wednesday :=
by
  sorry

end day_relationship_l100_10098


namespace no_valid_domino_placement_without_2x2_square_l100_10041

/-- Represents a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Represents a domino placement on the chessboard -/
def DominoPlacement := List (Fin 8 × Fin 8 × Bool)

/-- Checks if a domino placement is valid (covers the entire board without overlaps) -/
def isValidPlacement (board : Chessboard) (placement : DominoPlacement) : Prop :=
  sorry

/-- Checks if a domino placement forms a 2x2 square -/
def forms2x2Square (placement : DominoPlacement) : Prop :=
  sorry

/-- The main theorem: it's impossible to cover an 8x8 chessboard with 2x1 dominoes
    without forming a 2x2 square -/
theorem no_valid_domino_placement_without_2x2_square :
  ¬ ∃ (board : Chessboard) (placement : DominoPlacement),
    isValidPlacement board placement ∧ ¬ forms2x2Square placement :=
  sorry

end no_valid_domino_placement_without_2x2_square_l100_10041


namespace smallest_number_divisible_by_all_l100_10073

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 11) % 29 = 0 ∧
  (n + 11) % 53 = 0 ∧
  (n + 11) % 37 = 0 ∧
  (n + 11) % 41 = 0 ∧
  (n + 11) % 47 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 109871748 ∧
  ∀ m : ℕ, m < 109871748 → ¬is_divisible_by_all m :=
by sorry

end smallest_number_divisible_by_all_l100_10073


namespace certain_number_proof_l100_10044

def smallest_number : ℕ := 3153
def increase : ℕ := 3
def divisor1 : ℕ := 70
def divisor2 : ℕ := 25
def divisor3 : ℕ := 21

theorem certain_number_proof :
  ∃ (n : ℕ), n > 0 ∧
  (smallest_number + increase) % n = 0 ∧
  n % divisor1 = 0 ∧
  n % divisor2 = 0 ∧
  n % divisor3 = 0 ∧
  ∀ (m : ℕ), m > 0 →
    (smallest_number + increase) % m = 0 →
    m % divisor1 = 0 →
    m % divisor2 = 0 →
    m % divisor3 = 0 →
    n ≤ m :=
by
  sorry

end certain_number_proof_l100_10044


namespace max_value_f_l100_10000

theorem max_value_f (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  let f := fun (a b c : ℝ) => (1 - b*c + c) * (1 - a*c + a) * (1 - a*b + b)
  (∀ a b c, a > 0 → b > 0 → c > 0 → a * b * c = 1 → f a b c ≤ 1) ∧
  f x y z = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
by sorry

end max_value_f_l100_10000


namespace determinant_max_value_l100_10079

open Real

theorem determinant_max_value :
  let det (θ : ℝ) := 
    let a11 := 1
    let a12 := 1
    let a13 := 1
    let a21 := 1
    let a22 := 1 + sin θ ^ 2
    let a23 := 1
    let a31 := 1 + cos θ ^ 2
    let a32 := 1
    let a33 := 1
    a11 * (a22 * a33 - a23 * a32) - 
    a12 * (a21 * a33 - a23 * a31) + 
    a13 * (a21 * a32 - a22 * a31)
  ∀ θ : ℝ, det θ ≤ 1 ∧ ∃ θ₀ : ℝ, det θ₀ = 1 :=
by sorry

end determinant_max_value_l100_10079


namespace functional_equation_solution_l100_10010

/-- A function f: ℝ⁺ → ℝ⁺ satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (y * f x) * (x + y) = x^2 * (f x + f y)

/-- The theorem stating that the only function satisfying the equation is f(x) = 1/x -/
theorem functional_equation_solution (f : ℝ → ℝ) :
  FunctionalEquation f → ∀ x, x > 0 → f x = 1 / x := by
  sorry


end functional_equation_solution_l100_10010


namespace cos_product_from_sum_relations_l100_10056

theorem cos_product_from_sum_relations (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 0.6) 
  (h2 : Real.cos x + Real.cos y = 0.8) : 
  Real.cos x * Real.cos y = -11/100 := by
sorry

end cos_product_from_sum_relations_l100_10056


namespace simplify_expression_log_equation_result_l100_10067

-- Part 1
theorem simplify_expression (x : ℝ) (h : x > 0) :
  (x - 1) / (x^(2/3) + x^(1/3) + 1) + (x + 1) / (x^(1/3) + 1) - (x - x^(1/3)) / (x^(1/3) - 1) = -x^(1/3) :=
sorry

-- Part 2
theorem log_equation_result (x : ℝ) (h1 : x > 0) (h2 : 3*x - 2 > 0) (h3 : 3*x + 2 > 0)
  (h4 : 2 * Real.log (3*x - 2) = Real.log x + Real.log (3*x + 2)) :
  Real.log (Real.sqrt (2 * Real.sqrt (2 * Real.sqrt 2))) / Real.log (Real.sqrt x) = 7/4 :=
sorry

end simplify_expression_log_equation_result_l100_10067


namespace linear_function_constraint_l100_10076

/-- Given a linear function y = x - k, if for all x < 3, y < 2k, then k ≥ 1 -/
theorem linear_function_constraint (k : ℝ) : 
  (∀ x : ℝ, x < 3 → x - k < 2 * k) → k ≥ 1 := by
  sorry

end linear_function_constraint_l100_10076


namespace variance_or_std_dev_measures_stability_l100_10036

-- Define a type for exam scores
def ExamScore := ℝ

-- Define a type for a set of exam scores
def ExamScores := List ExamScore

-- Define a function to calculate variance
noncomputable def variance (scores : ExamScores) : ℝ := sorry

-- Define a function to calculate standard deviation
noncomputable def standardDeviation (scores : ExamScores) : ℝ := sorry

-- Define a measure of stability
noncomputable def stabilityMeasure (scores : ExamScores) : ℝ := sorry

-- Theorem stating that variance or standard deviation is the most appropriate measure of stability
theorem variance_or_std_dev_measures_stability (scores : ExamScores) :
  (stabilityMeasure scores = variance scores) ∨ (stabilityMeasure scores = standardDeviation scores) :=
sorry

end variance_or_std_dev_measures_stability_l100_10036


namespace largest_n_satisfying_inequality_l100_10096

theorem largest_n_satisfying_inequality :
  ∀ n : ℕ, n ≤ 9 ↔ (1 / 4 : ℚ) + (n / 8 : ℚ) < (3 / 2 : ℚ) :=
sorry

end largest_n_satisfying_inequality_l100_10096


namespace triangle_properties_l100_10017

theorem triangle_properties (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ratio : ∃ (k : ℝ), a = 2*k ∧ b = 5*k ∧ c = 6*k) 
  (h_area : (1/2) * a * c * Real.sqrt (1 - ((a^2 + c^2 - b^2) / (2*a*c))^2) = 3 * Real.sqrt 39 / 4) :
  ((a^2 + c^2 - b^2) / (2*a*c) = 5/8) ∧ (a + b + c = 13) := by
  sorry

end triangle_properties_l100_10017


namespace hydrogen_moles_formed_l100_10063

/-- Represents a chemical element --/
structure Element where
  name : String
  atomic_mass : Float

/-- Represents a chemical compound --/
structure Compound where
  formula : String
  elements : List (Element × Nat)

/-- Represents a chemical reaction --/
structure Reaction where
  reactants : List (Compound × Float)
  products : List (Compound × Float)

/-- Calculate the molar mass of a compound --/
def molar_mass (c : Compound) : Float :=
  c.elements.foldl (fun acc (elem, count) => acc + elem.atomic_mass * count.toFloat) 0

/-- Calculate the number of moles given mass and molar mass --/
def moles (mass : Float) (molar_mass : Float) : Float :=
  mass / molar_mass

/-- The main theorem --/
theorem hydrogen_moles_formed
  (carbon : Element)
  (hydrogen : Element)
  (benzene : Compound)
  (methane : Compound)
  (toluene : Compound)
  (h2 : Compound)
  (reaction : Reaction)
  (benzene_mass : Float) :
  carbon.atomic_mass = 12.01 →
  hydrogen.atomic_mass = 1.008 →
  benzene.elements = [(carbon, 6), (hydrogen, 6)] →
  methane.elements = [(carbon, 1), (hydrogen, 4)] →
  toluene.elements = [(carbon, 7), (hydrogen, 8)] →
  h2.elements = [(hydrogen, 2)] →
  reaction.reactants = [(benzene, 1), (methane, 1)] →
  reaction.products = [(toluene, 1), (h2, 1)] →
  benzene_mass = 156 →
  moles benzene_mass (molar_mass benzene) = 2 →
  moles benzene_mass (molar_mass benzene) = moles 2 (molar_mass h2) :=
by sorry

end hydrogen_moles_formed_l100_10063


namespace team_formation_ways_l100_10003

/-- Represents the number of people who know a specific pair of subjects -/
structure SubjectKnowledge where
  math_physics : Nat
  physics_chemistry : Nat
  chemistry_math : Nat
  physics_biology : Nat

/-- Calculates the total number of people -/
def total_people (sk : SubjectKnowledge) : Nat :=
  sk.math_physics + sk.physics_chemistry + sk.chemistry_math + sk.physics_biology

/-- Calculates the number of ways to choose 3 people from n people -/
def choose_3_from_n (n : Nat) : Nat :=
  n * (n - 1) * (n - 2) / 6

/-- Calculates the number of invalid selections (all 3 from the same group) -/
def invalid_selections (sk : SubjectKnowledge) : Nat :=
  choose_3_from_n sk.math_physics +
  choose_3_from_n sk.physics_chemistry +
  choose_3_from_n sk.chemistry_math +
  choose_3_from_n sk.physics_biology

/-- The main theorem to prove -/
theorem team_formation_ways (sk : SubjectKnowledge) 
  (h1 : sk.math_physics = 7)
  (h2 : sk.physics_chemistry = 6)
  (h3 : sk.chemistry_math = 3)
  (h4 : sk.physics_biology = 4) :
  choose_3_from_n (total_people sk) - invalid_selections sk = 1080 := by
  sorry

end team_formation_ways_l100_10003


namespace M_intersect_N_eq_unit_interval_l100_10088

-- Define the sets M and N
def M : Set ℝ := {x | x^2 ≤ 1}
def N : Set (ℝ × ℝ) := {p | p.2 ∈ M ∧ p.1 = p.2^2}

-- State the theorem
theorem M_intersect_N_eq_unit_interval :
  (M ∩ (N.image Prod.snd)) = Set.Icc 0 1 := by
  sorry

end M_intersect_N_eq_unit_interval_l100_10088


namespace range_of_b_l100_10033

theorem range_of_b (a b c : ℝ) (sum_eq : a + b + c = 9) (prod_eq : a * b + b * c + c * a = 24) :
  1 ≤ b ∧ b ≤ 5 := by
sorry

end range_of_b_l100_10033


namespace fraction_sum_equals_one_l100_10025

theorem fraction_sum_equals_one (a : ℝ) (h : a ≠ -2) :
  (a + 1) / (a + 2) + 1 / (a + 2) = 1 := by
  sorry

end fraction_sum_equals_one_l100_10025


namespace min_sum_at_6_l100_10082

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = -14
  sum_of_5th_6th : a 5 + a 6 = -4
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_of_first_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: The sum of the first n terms takes its minimum value when n = 6 -/
theorem min_sum_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≠ 0 → sum_of_first_n_terms seq 6 ≤ sum_of_first_n_terms seq n := by
  sorry

end min_sum_at_6_l100_10082


namespace concert_guests_combinations_l100_10089

theorem concert_guests_combinations : Nat.choose 10 5 = 252 := by
  sorry

end concert_guests_combinations_l100_10089


namespace yellow_candy_probability_l100_10006

theorem yellow_candy_probability (p_red p_orange p_yellow : ℝ) : 
  p_red = 0.25 →
  p_orange = 0.35 →
  p_red + p_orange + p_yellow = 1 →
  p_yellow = 0.4 := by
sorry

end yellow_candy_probability_l100_10006


namespace special_function_zero_l100_10095

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, |f a - f b| ≤ |a - b|) ∧ (f (f (f 0)) = 0)

/-- Theorem: If f is a special function, then f(0) = 0 -/
theorem special_function_zero (f : ℝ → ℝ) (h : special_function f) : f 0 = 0 := by
  sorry

end special_function_zero_l100_10095


namespace arithmetic_sequence_problem_l100_10081

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h1 : seq.S 2016 = 2016)
    (h2 : seq.S 2016 / 2016 - seq.S 16 / 16 = 2000) :
  seq.a 1 = -2014 := by
  sorry


end arithmetic_sequence_problem_l100_10081


namespace other_x_intercept_l100_10052

/-- Given a quadratic function with vertex (5, -3) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = -3 + a * (x - 5)^2) →  -- vertex form
  (a * 1^2 + b * 1 + c = 0) →                        -- x-intercept at (1, 0)
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9       -- other x-intercept at 9
  := by sorry

end other_x_intercept_l100_10052


namespace smallest_n_after_tax_l100_10034

theorem smallest_n_after_tax : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), m > 0 ∧ (104 * m = 100 * 100 * n)) ∧ 
  (∀ (k : ℕ), k > 0 → k < n → ¬∃ (j : ℕ), j > 0 ∧ (104 * j = 100 * 100 * k)) ∧ n = 13 := by
  sorry

end smallest_n_after_tax_l100_10034


namespace intersection_of_A_and_B_l100_10047

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) * (x + 1) ≥ 0}
def B : Set ℝ := {x | x < -4/5}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x ≤ -1} := by sorry

end intersection_of_A_and_B_l100_10047


namespace lcm_gcd_sum_theorem_l100_10013

theorem lcm_gcd_sum_theorem : 
  (Nat.lcm 12 18 * Nat.gcd 12 18) + (Nat.lcm 10 15 * Nat.gcd 10 15) = 366 := by
  sorry

end lcm_gcd_sum_theorem_l100_10013


namespace product_of_three_numbers_l100_10014

theorem product_of_three_numbers (x y z : ℝ) 
  (sum_eq_20 : x + y + z = 20)
  (first_eq_four_times_sum_others : x = 4 * (y + z))
  (second_eq_seven_times_third : y = 7 * z) : 
  x * y * z = 28 := by
  sorry

end product_of_three_numbers_l100_10014


namespace tens_digit_of_subtraction_l100_10057

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hun_less_than_tens : hundreds = tens - 3
  tens_double_units : tens = 2 * units
  is_three_digit : hundreds ≥ 1 ∧ hundreds ≤ 9

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

def ThreeDigitNumber.reversed (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

theorem tens_digit_of_subtraction (n : ThreeDigitNumber) :
  (n.toNat - n.reversed) / 10 % 10 = 9 := by
  sorry

end tens_digit_of_subtraction_l100_10057


namespace max_colored_cells_1000_cube_l100_10068

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  sideLength : n > 0

/-- Represents the maximum number of cells that can be colored on a cube's surface -/
def maxColoredCells (c : Cube n) : ℕ :=
  6 * n^2 - 2 * n^2

theorem max_colored_cells_1000_cube :
  ∀ (c : Cube 1000), maxColoredCells c = 2998000 :=
sorry

end max_colored_cells_1000_cube_l100_10068


namespace geometric_sequence_formula_arithmetic_sequence_sum_l100_10040

-- Define the geometric sequence {a_n}
def a (n : ℕ) : ℝ := sorry

-- Define the arithmetic sequence {b_n}
def b (n : ℕ) : ℝ := sorry

-- Define the sum of the first n terms of {b_n}
def S (n : ℕ) : ℝ := sorry

theorem geometric_sequence_formula :
  (a 2 = 6) →
  (a 2 + a 3 = 24) →
  ∀ n : ℕ, a n = 2 * 3^(n - 1) := by sorry

theorem arithmetic_sequence_sum :
  (b 1 = a 1) →
  (b 3 = -10) →
  ∀ n : ℕ, S n = -3 * n^2 + 5 * n := by sorry

end geometric_sequence_formula_arithmetic_sequence_sum_l100_10040


namespace candy_mixture_problem_l100_10054

/-- Candy mixture problem -/
theorem candy_mixture_problem (x : ℝ) :
  (64 * 2 + x * 3 = (64 + x) * 2.2) →
  (64 + x = 80) := by
  sorry

end candy_mixture_problem_l100_10054


namespace only_valid_N_l100_10087

theorem only_valid_N : 
  {N : ℕ+ | (∃ a b : ℕ, N = 2^a * 5^b) ∧ 
            (∃ k : ℕ, N + 25 = k^2)} = 
  {200, 2000} := by sorry

end only_valid_N_l100_10087


namespace valid_draws_eq_189_l100_10083

def total_cards : ℕ := 12
def cards_per_color : ℕ := 3
def num_colors : ℕ := 4
def cards_to_draw : ℕ := 3

def valid_draws : ℕ := Nat.choose total_cards cards_to_draw - 
                        (num_colors * Nat.choose cards_per_color cards_to_draw) - 
                        (Nat.choose cards_per_color 2 * Nat.choose (total_cards - cards_per_color) 1)

theorem valid_draws_eq_189 : valid_draws = 189 := by sorry

end valid_draws_eq_189_l100_10083


namespace garden_feet_is_117_l100_10065

/-- The number of feet in a garden with various animals --/
def garden_feet : ℕ :=
  let normal_dog_count : ℕ := 5
  let normal_cat_count : ℕ := 3
  let normal_bird_count : ℕ := 6
  let duck_count : ℕ := 2
  let insect_count : ℕ := 10
  let three_legged_dog_count : ℕ := 1
  let three_legged_cat_count : ℕ := 1
  let three_legged_bird_count : ℕ := 1
  let dog_legs : ℕ := normal_dog_count * 4 + three_legged_dog_count * 3
  let cat_legs : ℕ := normal_cat_count * 4 + three_legged_cat_count * 3
  let bird_legs : ℕ := normal_bird_count * 2 + three_legged_bird_count * 3
  let duck_legs : ℕ := duck_count * 2
  let insect_legs : ℕ := insect_count * 6
  dog_legs + cat_legs + bird_legs + duck_legs + insect_legs

theorem garden_feet_is_117 : garden_feet = 117 := by
  sorry

end garden_feet_is_117_l100_10065


namespace percent_employed_females_l100_10077

/-- Given a town where 60% of the population are employed and 42% of the population are employed males,
    prove that 30% of the employed people are females. -/
theorem percent_employed_females (town : Type) 
  (total_population : ℕ) 
  (employed : ℕ) 
  (employed_males : ℕ) 
  (h1 : employed = (60 : ℚ) / 100 * total_population) 
  (h2 : employed_males = (42 : ℚ) / 100 * total_population) : 
  (employed - employed_males : ℚ) / employed = 30 / 100 := by
sorry

end percent_employed_females_l100_10077


namespace perimeter_to_hypotenuse_ratio_l100_10012

/-- Right triangle ABC with altitude CD to hypotenuse AB and circle ω with CD as diameter -/
structure RightTriangleWithCircle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point D on hypotenuse AB -/
  D : ℝ × ℝ
  /-- Center of circle ω -/
  O : ℝ × ℝ
  /-- Point I outside the triangle -/
  I : ℝ × ℝ
  /-- ABC is a right triangle with right angle at C -/
  is_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  /-- AC = 15 -/
  ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 15
  /-- BC = 20 -/
  bc_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 20
  /-- CD is perpendicular to AB -/
  cd_perpendicular : (D.1 - C.1) * (B.1 - A.1) + (D.2 - C.2) * (B.2 - A.2) = 0
  /-- D is on AB -/
  d_on_ab : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  /-- O is the midpoint of CD -/
  o_midpoint : O = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  /-- AI is tangent to circle ω -/
  ai_tangent : Real.sqrt ((I.1 - A.1)^2 + (I.2 - A.2)^2) * Real.sqrt ((I.1 - O.1)^2 + (I.2 - O.2)^2) = (I.1 - A.1) * (I.1 - O.1) + (I.2 - A.2) * (I.2 - O.2)
  /-- BI is tangent to circle ω -/
  bi_tangent : Real.sqrt ((I.1 - B.1)^2 + (I.2 - B.2)^2) * Real.sqrt ((I.1 - O.1)^2 + (I.2 - O.2)^2) = (I.1 - B.1) * (I.1 - O.1) + (I.2 - B.2) * (I.2 - O.2)

/-- The ratio of the perimeter of triangle ABI to the length of AB is 5/2 -/
theorem perimeter_to_hypotenuse_ratio (t : RightTriangleWithCircle) :
  let ab_length := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let ai_length := Real.sqrt ((t.I.1 - t.A.1)^2 + (t.I.2 - t.A.2)^2)
  let bi_length := Real.sqrt ((t.I.1 - t.B.1)^2 + (t.I.2 - t.B.2)^2)
  (ai_length + bi_length + ab_length) / ab_length = 5/2 := by
  sorry

end perimeter_to_hypotenuse_ratio_l100_10012


namespace friday_snowfall_l100_10037

-- Define the snowfall amounts
def total_snowfall : Float := 0.89
def wednesday_snowfall : Float := 0.33
def thursday_snowfall : Float := 0.33

-- Define the theorem
theorem friday_snowfall :
  total_snowfall - (wednesday_snowfall + thursday_snowfall) = 0.23 := by
  sorry

end friday_snowfall_l100_10037


namespace gina_initial_amount_l100_10099

def initial_amount (remaining : ℚ) (fraction_given : ℚ) : ℚ :=
  remaining / (1 - fraction_given)

theorem gina_initial_amount :
  let fraction_to_mom : ℚ := 1/4
  let fraction_for_clothes : ℚ := 1/8
  let fraction_to_charity : ℚ := 1/5
  let total_fraction_given := fraction_to_mom + fraction_for_clothes + fraction_to_charity
  let remaining_amount : ℚ := 170
  initial_amount remaining_amount total_fraction_given = 400 := by
  sorry

end gina_initial_amount_l100_10099


namespace triangle_side_length_l100_10084

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  a = Real.sqrt 3 →
  b = 1 →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  c = 2 := by
sorry

end triangle_side_length_l100_10084


namespace expression_change_l100_10007

/-- The change in the expression 2x^2 + 5 when x changes by ±b -/
theorem expression_change (x b : ℝ) (h : b > 0) :
  let f : ℝ → ℝ := λ t => 2 * t^2 + 5
  abs (f (x + b) - f x) = 2 * b * (2 * x + b) ∧
  abs (f (x - b) - f x) = 2 * b * (2 * x + b) :=
by sorry

end expression_change_l100_10007


namespace greatest_two_digit_multiple_of_11_l100_10060

theorem greatest_two_digit_multiple_of_11 : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 11 ∣ n → n ≤ 99 :=
by
  sorry

end greatest_two_digit_multiple_of_11_l100_10060


namespace johns_remaining_money_l100_10002

/-- Given John's initial money and his purchases, calculate the remaining amount --/
theorem johns_remaining_money (initial : ℕ) (roast : ℕ) (vegetables : ℕ) :
  initial = 100 ∧ roast = 17 ∧ vegetables = 11 →
  initial - (roast + vegetables) = 72 := by
  sorry

end johns_remaining_money_l100_10002


namespace set_01_proper_subset_N_l100_10058

-- Define the set of natural numbers
def N : Set ℕ := Set.univ

-- Define the set {0,1}
def set_01 : Set ℕ := {0, 1}

-- Theorem to prove
theorem set_01_proper_subset_N : set_01 ⊂ N := by sorry

end set_01_proper_subset_N_l100_10058


namespace allison_wins_probability_l100_10026

-- Define the faces of each cube
def allison_cube : Finset Nat := {4}
def charlie_cube : Finset Nat := {1, 2, 3, 4, 5, 6}
def eve_cube : Finset Nat := {3, 3, 4, 4, 4, 5}

-- Define the probability of rolling each face
def prob_roll (cube : Finset Nat) (face : Nat) : ℚ :=
  (cube.filter (· = face)).card / cube.card

-- Define the event of rolling less than 4
def roll_less_than_4 (cube : Finset Nat) : ℚ :=
  (cube.filter (· < 4)).card / cube.card

-- Theorem statement
theorem allison_wins_probability :
  prob_roll allison_cube 4 * roll_less_than_4 charlie_cube * roll_less_than_4 eve_cube = 1/6 := by
  sorry

end allison_wins_probability_l100_10026


namespace sphere_volume_from_surface_area_l100_10018

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), r > 0 → 4 * π * r^2 = 9 * π → (4 / 3) * π * r^3 = 36 * π := by
  sorry

end sphere_volume_from_surface_area_l100_10018


namespace square_side_length_l100_10004

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 9 / 16) (h2 : side * side = area) : side = 3 / 4 := by
  sorry

end square_side_length_l100_10004


namespace exam_score_ratio_l100_10071

theorem exam_score_ratio (total_questions : ℕ) (lowella_percentage : ℚ) 
  (pamela_additional_percentage : ℚ) (mandy_score : ℕ) : 
  total_questions = 100 →
  lowella_percentage = 35 / 100 →
  pamela_additional_percentage = 20 / 100 →
  mandy_score = 84 →
  ∃ (k : ℚ), k * (lowella_percentage * total_questions + 
    pamela_additional_percentage * (lowella_percentage * total_questions)) = mandy_score ∧ 
  k = 2 := by
  sorry

end exam_score_ratio_l100_10071
