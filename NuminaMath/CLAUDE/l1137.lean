import Mathlib

namespace diamond_three_two_l1137_113719

def diamond (a b : ℝ) : ℝ := a * b^3 - b^2 + 1

theorem diamond_three_two : diamond 3 2 = 21 := by
  sorry

end diamond_three_two_l1137_113719


namespace lunch_expense_calculation_l1137_113735

theorem lunch_expense_calculation (initial_money : ℝ) (gasoline_expense : ℝ) (gift_expense_per_person : ℝ) (grandma_gift_per_person : ℝ) (return_trip_money : ℝ) :
  initial_money = 50 →
  gasoline_expense = 8 →
  gift_expense_per_person = 5 →
  grandma_gift_per_person = 10 →
  return_trip_money = 36.35 →
  let total_money := initial_money + 2 * grandma_gift_per_person
  let total_expense := gasoline_expense + 2 * gift_expense_per_person
  let lunch_expense := total_money - total_expense - return_trip_money
  lunch_expense = 15.65 := by
sorry

end lunch_expense_calculation_l1137_113735


namespace sin_A_range_l1137_113713

theorem sin_A_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- positive angles
  A + B + C = π ∧ -- sum of angles in a triangle
  C = π / 3 ∧ 
  a = 6 ∧ 
  1 ≤ b ∧ b ≤ 4 ∧
  a / (Real.sin A) = b / (Real.sin B) ∧ -- sine rule
  a / (Real.sin A) = c / (Real.sin C) ∧ -- sine rule
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) -- cosine rule
  →
  3 * Real.sqrt 93 / 31 ≤ Real.sin A ∧ Real.sin A ≤ 1 :=
by sorry

end sin_A_range_l1137_113713


namespace exists_double_application_square_l1137_113746

theorem exists_double_application_square : 
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n^2 := by sorry

end exists_double_application_square_l1137_113746


namespace trajectory_is_circle_l1137_113763

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

-- Define the points
variable (F₁ F₂ P Q : E)

-- Define the ellipse
def is_on_ellipse (P : E) (F₁ F₂ : E) (a : ℝ) : Prop :=
  dist P F₁ + dist P F₂ = 2 * a

-- Define the condition for Q
def extends_to_Q (P Q : E) (F₁ F₂ : E) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q = F₁ + t • (P - F₁) ∧ dist P Q = dist P F₂

-- Theorem statement
theorem trajectory_is_circle 
  (a : ℝ) 
  (h_ellipse : is_on_ellipse P F₁ F₂ a) 
  (h_extends : extends_to_Q P Q F₁ F₂) :
  ∃ (center : E) (radius : ℝ), dist Q center = radius :=
sorry

end trajectory_is_circle_l1137_113763


namespace periodic_odd_function_sum_l1137_113791

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def MinimumPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  HasPeriod f p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬HasPeriod f q

theorem periodic_odd_function_sum (f : ℝ → ℝ) :
  IsOdd f →
  MinimumPositivePeriod f 3 →
  (∀ x, f x = Real.log (1 - x)) →
  f 2010 + f 2011 = 1 := by
  sorry

end periodic_odd_function_sum_l1137_113791


namespace power_function_through_point_l1137_113708

theorem power_function_through_point (a k : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^k = ((a - 1):ℝ) * x^k) → 
  (a - 1) * (Real.sqrt 2)^k = 2 → 
  a + k = 4 := by
sorry

end power_function_through_point_l1137_113708


namespace sum_of_first_four_terms_l1137_113779

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_four_terms
  (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_5th : a 5 = 11)
  (h_6th : a 6 = 17)
  (h_7th : a 7 = 23) :
  a 1 + a 2 + a 3 + a 4 = -16 :=
sorry

end sum_of_first_four_terms_l1137_113779


namespace basketball_free_throws_l1137_113778

theorem basketball_free_throws 
  (two_point_shots three_point_shots free_throws : ℕ) :
  (3 * three_point_shots = 2 * two_point_shots) →
  (three_point_shots = two_point_shots - 2) →
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 68) →
  free_throws = 44 := by
sorry

end basketball_free_throws_l1137_113778


namespace triangle_perimeter_impossibility_l1137_113787

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 24) (h2 : b = 18) : 
  (a + b + x = 87) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
by sorry

end triangle_perimeter_impossibility_l1137_113787


namespace goods_train_length_l1137_113789

/-- The length of a goods train passing a man on another train --/
theorem goods_train_length
  (man_train_speed : ℝ)
  (goods_train_speed : ℝ)
  (passing_time : ℝ)
  (h1 : man_train_speed = 40)
  (h2 : goods_train_speed = 72)
  (h3 : passing_time = 9) :
  (man_train_speed + goods_train_speed) * passing_time * 1000 / 3600 = 280 := by
  sorry

end goods_train_length_l1137_113789


namespace butter_amount_is_480_l1137_113748

/-- Represents the ingredients in a recipe --/
structure Ingredients where
  flour : ℝ
  butter : ℝ
  sugar : ℝ

/-- Represents the ratios of ingredients in a recipe --/
structure Ratio where
  flour : ℝ
  butter : ℝ
  sugar : ℝ

/-- Calculates the total ingredients after mixing two recipes and adding extra flour --/
def mixRecipes (cake : Ingredients) (cream : Ingredients) (extraFlour : ℝ) : Ingredients :=
  { flour := cake.flour + extraFlour
  , butter := cake.butter + cream.butter
  , sugar := cake.sugar + cream.sugar }

/-- Checks if the given ingredients satisfy the required ratio --/
def satisfiesRatio (ingredients : Ingredients) (ratio : Ratio) : Prop :=
  ingredients.flour / ratio.flour = ingredients.butter / ratio.butter ∧
  ingredients.flour / ratio.flour = ingredients.sugar / ratio.sugar

/-- Main theorem: The amount of butter used is 480 grams --/
theorem butter_amount_is_480 
  (cake_ratio : Ratio)
  (cream_ratio : Ratio)
  (cookie_ratio : Ratio)
  (cake : Ingredients)
  (cream : Ingredients)
  (h1 : satisfiesRatio cake cake_ratio)
  (h2 : satisfiesRatio cream cream_ratio)
  (h3 : cake_ratio = { flour := 3, butter := 2, sugar := 1 })
  (h4 : cream_ratio = { flour := 0, butter := 2, sugar := 3 })
  (h5 : cookie_ratio = { flour := 5, butter := 3, sugar := 2 })
  (h6 : satisfiesRatio (mixRecipes cake cream 200) cookie_ratio) :
  cake.butter + cream.butter = 480 := by
  sorry


end butter_amount_is_480_l1137_113748


namespace subset_proportion_bound_l1137_113798

theorem subset_proportion_bound 
  (total : ℕ) 
  (subset : ℕ) 
  (event1_total : ℕ) 
  (event1_subset : ℕ) 
  (event2_total : ℕ) 
  (event2_subset : ℕ) 
  (h1 : event1_subset < 2 * event1_total / 5)
  (h2 : event2_subset < 2 * event2_total / 5)
  (h3 : event1_subset + event2_subset ≥ subset)
  (h4 : event1_total + event2_total ≥ total) :
  subset < 4 * total / 7 := by
sorry

end subset_proportion_bound_l1137_113798


namespace andy_math_problem_l1137_113718

theorem andy_math_problem (last_problem : ℕ) (total_solved : ℕ) (start_problem : ℕ) :
  last_problem = 125 →
  total_solved = 56 →
  start_problem = last_problem - total_solved + 1 →
  start_problem = 70 := by
sorry

end andy_math_problem_l1137_113718


namespace cookie_count_l1137_113710

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the number of full smaller rectangles that can fit into a larger rectangle -/
def fullRectanglesFit (large : Dimensions) (small : Dimensions) : ℕ :=
  (large.length / small.length) * (large.width / small.width)

theorem cookie_count :
  let sheet := Dimensions.mk 30 24
  let cookie := Dimensions.mk 3 4
  fullRectanglesFit sheet cookie = 60 := by
  sorry

#eval fullRectanglesFit (Dimensions.mk 30 24) (Dimensions.mk 3 4)

end cookie_count_l1137_113710


namespace probability_of_y_selection_l1137_113781

theorem probability_of_y_selection (p_x p_both : ℝ) (h1 : p_x = 1/7) 
  (h2 : p_both = 0.031746031746031744) : 
  ∃ p_y : ℝ, p_y = 0.2222222222222222 ∧ p_both = p_x * p_y :=
sorry

end probability_of_y_selection_l1137_113781


namespace base9_multiplication_l1137_113707

/-- Converts a base 9 number represented as a list of digits to its decimal equivalent -/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 9 * acc + d) 0

/-- Converts a decimal number to its base 9 representation as a list of digits -/
def decimalToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- The main theorem statement -/
theorem base9_multiplication :
  let a := base9ToDecimal [3, 2, 7]
  let b := base9ToDecimal [1, 2]
  decimalToBase9 (a * b) = [4, 0, 4, 5] := by
  sorry


end base9_multiplication_l1137_113707


namespace parallel_iff_a_eq_two_l1137_113750

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_iff_same_slope {a b c d : ℝ} (l1 l2 : ℝ → ℝ → Prop) :
  (∀ x y, l1 x y ↔ a * x + b * y = 0) →
  (∀ x y, l2 x y ↔ c * x + d * y = 1) →
  (∀ x y, l1 x y → l2 x y) ↔ a / b = c / d

/-- The line ax + 2y = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y = 0

/-- The line x + y = 1 -/
def line2 (x y : ℝ) : Prop := x + y = 1

/-- Theorem: a = 2 is both sufficient and necessary for line1 to be parallel to line2 -/
theorem parallel_iff_a_eq_two (a : ℝ) :
  (∀ x y, line1 a x y → line2 x y) ↔ a = 2 :=
sorry

end parallel_iff_a_eq_two_l1137_113750


namespace poster_purchase_l1137_113753

theorem poster_purchase (regular_price : ℕ) (budget : ℕ) : 
  budget = 24 * regular_price → 
  (∃ (num_posters : ℕ), 
    num_posters * regular_price + (num_posters / 2) * (regular_price / 2) = budget ∧ 
    num_posters = 32) :=
by sorry

end poster_purchase_l1137_113753


namespace soda_price_ratio_l1137_113760

/-- The ratio of unit prices between two soda brands -/
theorem soda_price_ratio (v : ℝ) (p : ℝ) (h1 : v > 0) (h2 : p > 0) : 
  (0.85 * p) / (1.25 * v) / (p / v) = 17 / 25 := by
  sorry

end soda_price_ratio_l1137_113760


namespace john_spent_15_dollars_l1137_113796

def price_per_dozen : ℕ := 5
def rolls_bought : ℕ := 36

theorem john_spent_15_dollars : 
  (rolls_bought / 12) * price_per_dozen = 15 := by
  sorry

end john_spent_15_dollars_l1137_113796


namespace valentine_card_cost_l1137_113730

def total_students : ℕ := 30
def valentine_percentage : ℚ := 60 / 100
def initial_money : ℚ := 40
def spending_percentage : ℚ := 90 / 100

theorem valentine_card_cost :
  let students_receiving := total_students * valentine_percentage
  let money_spent := initial_money * spending_percentage
  let cost_per_card := money_spent / students_receiving
  cost_per_card = 2 := by sorry

end valentine_card_cost_l1137_113730


namespace three_fifths_of_ten_times_seven_minus_three_l1137_113783

theorem three_fifths_of_ten_times_seven_minus_three (x : ℚ) : x = 40.2 → x = (3 / 5) * ((10 * 7) - 3) := by
  sorry

end three_fifths_of_ten_times_seven_minus_three_l1137_113783


namespace train_passing_time_l1137_113729

/-- Given a train of length 420 meters traveling at 63 km/hr,
    prove that it takes 24 seconds to pass a stationary point. -/
theorem train_passing_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 420 ∧ train_speed_kmh = 63 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 24 := by
  sorry

end train_passing_time_l1137_113729


namespace flower_pots_height_l1137_113747

/-- Calculates the total vertical distance of stacked flower pots --/
def total_vertical_distance (top_diameter : ℕ) (bottom_diameter : ℕ) (thickness : ℕ) : ℕ :=
  let num_pots := (top_diameter - bottom_diameter) / 2 + 1
  let inner_sum := num_pots * (top_diameter - thickness + bottom_diameter - thickness) / 2
  inner_sum + 2 * thickness

/-- Theorem stating the total vertical distance of the flower pots --/
theorem flower_pots_height : total_vertical_distance 16 4 1 = 65 := by
  sorry

end flower_pots_height_l1137_113747


namespace parabola_b_value_l1137_113734

/-- A parabola passing through two points -/
structure Parabola where
  a : ℝ
  b : ℝ
  passes_through_1_2 : 2 = 1^2 + a * 1 + b
  passes_through_3_2 : 2 = 3^2 + a * 3 + b

/-- The value of b for the parabola passing through (1,2) and (3,2) is 5 -/
theorem parabola_b_value (p : Parabola) : p.b = 5 := by
  sorry

end parabola_b_value_l1137_113734


namespace ten_day_search_cost_l1137_113769

/-- Tom's charging scheme for item search -/
def search_cost (days : ℕ) : ℕ :=
  let initial_rate := 100
  let discounted_rate := 60
  let initial_period := 5
  if days ≤ initial_period then
    days * initial_rate
  else
    initial_period * initial_rate + (days - initial_period) * discounted_rate

/-- The theorem stating the total cost for a 10-day search -/
theorem ten_day_search_cost : search_cost 10 = 800 := by
  sorry

end ten_day_search_cost_l1137_113769


namespace parabola_point_relation_l1137_113744

theorem parabola_point_relation (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = x₁^2 - 4*x₁ + 3 →
  y₂ = x₂^2 - 4*x₂ + 3 →
  x₁ > x₂ →
  x₂ > 2 →
  y₁ > y₂ := by
sorry

end parabola_point_relation_l1137_113744


namespace expected_sum_of_three_marbles_l1137_113774

def marbles : Finset ℕ := Finset.range 7

def draw_size : ℕ := 3

theorem expected_sum_of_three_marbles :
  let all_draws := marbles.powerset.filter (λ s => s.card = draw_size)
  let sum_of_draws := all_draws.sum (λ s => s.sum id)
  let num_of_draws := all_draws.card
  (sum_of_draws : ℚ) / num_of_draws = 12 := by sorry

end expected_sum_of_three_marbles_l1137_113774


namespace train_speed_l1137_113741

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length time : ℝ) (h1 : length = 560) (h2 : time = 16) :
  length / time = 35 := by
  sorry

end train_speed_l1137_113741


namespace gcd_problem_l1137_113727

-- Define the operation * as the greatest common divisor
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

-- State the theorem
theorem gcd_problem : gcd_op (gcd_op 20 16) (gcd_op 18 24) = 2 := by sorry

end gcd_problem_l1137_113727


namespace value_of_z_l1137_113761

theorem value_of_z (x y z : ℝ) : 
  y = 3 * x - 5 → 
  z = 3 * x + 3 → 
  y = 1 → 
  z = 9 := by
sorry

end value_of_z_l1137_113761


namespace C_is_largest_l1137_113784

-- Define A, B, and C
def A : ℚ := 2010/2009 + 2010/2011
def B : ℚ := (2010/2011) * (2012/2011)
def C : ℚ := 2011/2010 + 2011/2012 + 1/10000

-- Theorem statement
theorem C_is_largest : C > A ∧ C > B := by
  sorry

end C_is_largest_l1137_113784


namespace total_wheels_is_142_l1137_113742

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of wheels on a unicycle -/
def unicycle_wheels : ℕ := 1

/-- The number of wheels on a four-wheeler -/
def four_wheeler_wheels : ℕ := 4

/-- The number of bicycles in Storage Area A -/
def bicycles_A : ℕ := 16

/-- The number of tricycles in Storage Area A -/
def tricycles_A : ℕ := 7

/-- The number of unicycles in Storage Area A -/
def unicycles_A : ℕ := 10

/-- The number of four-wheelers in Storage Area A -/
def four_wheelers_A : ℕ := 5

/-- The number of bicycles in Storage Area B -/
def bicycles_B : ℕ := 12

/-- The number of tricycles in Storage Area B -/
def tricycles_B : ℕ := 5

/-- The number of unicycles in Storage Area B -/
def unicycles_B : ℕ := 8

/-- The number of four-wheelers in Storage Area B -/
def four_wheelers_B : ℕ := 3

/-- The total number of wheels in both storage areas -/
def total_wheels : ℕ := 
  (bicycles_A * bicycle_wheels + tricycles_A * tricycle_wheels + unicycles_A * unicycle_wheels + four_wheelers_A * four_wheeler_wheels) +
  (bicycles_B * bicycle_wheels + tricycles_B * tricycle_wheels + unicycles_B * unicycle_wheels + four_wheelers_B * four_wheeler_wheels)

theorem total_wheels_is_142 : total_wheels = 142 := by
  sorry

end total_wheels_is_142_l1137_113742


namespace equal_money_distribution_l1137_113714

theorem equal_money_distribution (younger_money : ℝ) (h : younger_money > 0) :
  let elder_money := 1.25 * younger_money
  let transfer_amount := 0.1 * elder_money
  elder_money - transfer_amount = younger_money + transfer_amount := by
  sorry

end equal_money_distribution_l1137_113714


namespace house_to_school_distance_house_to_school_distance_is_60_l1137_113732

/-- The distance between a house and a school, given travel times at different speeds -/
theorem house_to_school_distance : ℝ :=
  let speed_slow : ℝ := 10  -- km/hr
  let speed_fast : ℝ := 20  -- km/hr
  let time_late : ℝ := 2    -- hours
  let time_early : ℝ := 1   -- hours
  let distance : ℝ := 60    -- km

  have h1 : distance = speed_slow * (distance / speed_slow + time_late) := by sorry
  have h2 : distance = speed_fast * (distance / speed_fast - time_early) := by sorry

  distance

/-- The proof that the distance is indeed 60 km -/
theorem house_to_school_distance_is_60 : house_to_school_distance = 60 := by sorry

end house_to_school_distance_house_to_school_distance_is_60_l1137_113732


namespace square_root_problem_l1137_113720

theorem square_root_problem (x y z : ℝ) 
  (h1 : Real.sqrt (2 * x + 1) = 0)
  (h2 : Real.sqrt y = 4)
  (h3 : z^3 = -27) :
  {r : ℝ | r^2 = 2*x + y + z} = {2 * Real.sqrt 3, -2 * Real.sqrt 3} := by
sorry

end square_root_problem_l1137_113720


namespace rider_distance_l1137_113745

/-- The distance traveled by a rider moving back and forth along a moving caravan -/
theorem rider_distance (caravan_length caravan_distance : ℝ) 
  (h_length : caravan_length = 1)
  (h_distance : caravan_distance = 1) : 
  ∃ (rider_speed : ℝ), 
    rider_speed > 0 ∧ 
    (1 / (rider_speed - 1) + 1 / (rider_speed + 1) = 1) ∧
    rider_speed * caravan_distance = 1 + Real.sqrt 2 :=
by sorry

end rider_distance_l1137_113745


namespace print_325_pages_time_l1137_113749

/-- Calculates the time required to print a given number of pages with a printer that has a specific print rate and delay after every 100 pages. -/
def print_time (total_pages : ℕ) (pages_per_minute : ℕ) (delay_minutes : ℕ) : ℕ :=
  let print_time := total_pages / pages_per_minute
  let num_delays := total_pages / 100
  print_time + num_delays * delay_minutes

/-- Theorem stating that printing 325 pages takes 16 minutes with the given conditions. -/
theorem print_325_pages_time :
  print_time 325 25 1 = 16 :=
by sorry

end print_325_pages_time_l1137_113749


namespace f_has_root_in_interval_l1137_113712

/-- The function f(x) = ln(2x) - 1 has a root in the interval (1, 2) -/
theorem f_has_root_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ Real.log (2 * x) - 1 = 0 :=
by
  sorry

end f_has_root_in_interval_l1137_113712


namespace prism_volume_l1137_113702

/-- The volume of a right rectangular prism with face areas 30, 50, and 75 square centimeters is 335 cubic centimeters. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 335 := by
  sorry

end prism_volume_l1137_113702


namespace find_a_solution_set_g_solution_set_h_l1137_113722

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := (1 - a) * x^2 - 4 * x + 6

-- Define the solution set condition
def solution_set_condition (a : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ -3 < x ∧ x < 1

-- Theorem 1
theorem find_a (a : ℝ) (h : solution_set_condition a) : a = 3 :=
sorry

-- Define the quadratic function for part 2
def g (x : ℝ) : ℝ := 2 * x^2 - x - 3

-- Theorem 2
theorem solution_set_g :
  ∀ x, g x > 0 ↔ x < -1 ∨ x > 3/2 :=
sorry

-- Define the quadratic function for part 3
def h (b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + b * x + 3

-- Theorem 3
theorem solution_set_h (b : ℝ) :
  (∀ x, h b x ≥ 0) ↔ -6 ≤ b ∧ b ≤ 6 :=
sorry

end find_a_solution_set_g_solution_set_h_l1137_113722


namespace equation_is_quadratic_l1137_113704

/-- Definition of a quadratic equation in x -/
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x - 6

/-- Theorem: The given equation is a quadratic equation in x -/
theorem equation_is_quadratic : is_quadratic_in_x f := by
  sorry

end equation_is_quadratic_l1137_113704


namespace remainder_problem_l1137_113757

theorem remainder_problem (x : ℤ) : x % 63 = 27 → x % 8 = 3 := by
  sorry

end remainder_problem_l1137_113757


namespace remainder_x7_plus_2_div_x_plus_1_l1137_113775

theorem remainder_x7_plus_2_div_x_plus_1 :
  ∃ q : Polynomial ℤ, (X ^ 7 + 2 : Polynomial ℤ) = (X + 1) * q + 1 :=
sorry

end remainder_x7_plus_2_div_x_plus_1_l1137_113775


namespace sine_graph_shift_l1137_113768

theorem sine_graph_shift (x : ℝ) :
  Real.sin (2 * (x + π / 4) + π / 6) = Real.sin (2 * x + 2 * π / 3) := by
  sorry

#check sine_graph_shift

end sine_graph_shift_l1137_113768


namespace first_same_side_after_104_minutes_l1137_113721

/-- Represents a person walking around a pentagonal square -/
structure Walker where
  start_point : Fin 5
  speed : ℝ

/-- The time when two walkers are first on the same side of a pentagonal square -/
def first_same_side_time (perimeter : ℝ) (walker_a walker_b : Walker) : ℝ :=
  sorry

/-- The main theorem -/
theorem first_same_side_after_104_minutes :
  let perimeter : ℝ := 2000
  let walker_a : Walker := { start_point := 0, speed := 50 }
  let walker_b : Walker := { start_point := 2, speed := 46 }
  first_same_side_time perimeter walker_a walker_b = 104 := by
  sorry

end first_same_side_after_104_minutes_l1137_113721


namespace inequality_and_equality_cases_l1137_113731

theorem inequality_and_equality_cases (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 ∧
  ((a - b) * (b - c) * (a - c) = 2 ↔ 
    ((a = 2 ∧ b = 1 ∧ c = 0) ∨ 
     (a = 1 ∧ b = 0 ∧ c = 2) ∨ 
     (a = 0 ∧ b = 2 ∧ c = 1))) :=
by sorry

end inequality_and_equality_cases_l1137_113731


namespace curvilinear_triangle_area_half_triangle_area_l1137_113743

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle type
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Function to calculate the area of a curvilinear triangle formed by three circles
def curvilinearTriangleArea (c1 c2 c3 : Circle) : ℝ := sorry

-- Theorem statement
theorem curvilinear_triangle_area_half_triangle_area 
  (c1 c2 c3 : Circle) 
  (t : Triangle) 
  (h1 : c1.radius = c2.radius ∧ c2.radius = c3.radius)
  (h2 : c1.center = t.a ∧ c2.center = t.b ∧ c3.center = t.c) :
  curvilinearTriangleArea c1 c2 c3 = (1/2) * triangleArea t := by sorry

end curvilinear_triangle_area_half_triangle_area_l1137_113743


namespace first_share_rate_is_9_percent_l1137_113771

-- Define the total investment
def total_investment : ℝ := 10000

-- Define the interest rate of the second share
def second_share_rate : ℝ := 0.11

-- Define the total interest rate after one year
def total_interest_rate : ℝ := 0.0975

-- Define the amount invested in the second share
def second_share_investment : ℝ := 3750

-- Define the amount invested in the first share
def first_share_investment : ℝ := total_investment - second_share_investment

-- Theorem: The interest rate of the first share is 9%
theorem first_share_rate_is_9_percent :
  ∃ r : ℝ, r = 0.09 ∧
  r * first_share_investment + second_share_rate * second_share_investment =
  total_interest_rate * total_investment :=
by sorry

end first_share_rate_is_9_percent_l1137_113771


namespace inequality_proof_l1137_113790

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end inequality_proof_l1137_113790


namespace axis_of_symmetry_implies_r_equals_s_l1137_113785

/-- Represents a rational function of the form (px + q) / (rx + s) -/
structure RationalFunction (α : Type) [Field α] where
  p : α
  q : α
  r : α
  s : α
  p_nonzero : p ≠ 0
  q_nonzero : q ≠ 0
  r_nonzero : r ≠ 0
  s_nonzero : s ≠ 0

/-- Defines the property of y = -x being an axis of symmetry for a given rational function -/
def isAxisOfSymmetry {α : Type} [Field α] (f : RationalFunction α) : Prop :=
  ∀ (x y : α), y = (f.p * x + f.q) / (f.r * x + f.s) → (-x) = (f.p * (-y) + f.q) / (f.r * (-y) + f.s)

/-- Theorem stating that if y = -x is an axis of symmetry for the rational function,
    then r - s = 0 -/
theorem axis_of_symmetry_implies_r_equals_s {α : Type} [Field α] (f : RationalFunction α) :
  isAxisOfSymmetry f → f.r = f.s :=
sorry

end axis_of_symmetry_implies_r_equals_s_l1137_113785


namespace clocks_chime_together_l1137_113770

def clock1_interval : ℕ := 15
def clock2_interval : ℕ := 25

theorem clocks_chime_together : Nat.lcm clock1_interval clock2_interval = 75 := by
  sorry

end clocks_chime_together_l1137_113770


namespace dart_board_probability_l1137_113709

/-- The probability of a dart landing in the center square of a regular hexagon dart board -/
theorem dart_board_probability (s : ℝ) (h : s > 0) : 
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let center_square_area := s^2 / 3
  center_square_area / hexagon_area = 2 * Real.sqrt 3 / 27 := by
  sorry

end dart_board_probability_l1137_113709


namespace quadratic_factorization_l1137_113700

theorem quadratic_factorization (a b : ℤ) :
  (∀ x, 25 * x^2 - 195 * x - 198 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -420 := by
sorry

end quadratic_factorization_l1137_113700


namespace odd_function_extension_l1137_113788

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_extension :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = x * (1 - x)) →  -- f(x) = x(1-x) for x > 0
  (∀ x < 0, f x = x * (1 + x)) :=  -- f(x) = x(1+x) for x < 0
by sorry

end odd_function_extension_l1137_113788


namespace z₂_in_fourth_quadrant_z₂_equals_z₁_times_ni_l1137_113724

-- Define complex numbers z₁ and z₂
def z₁ (m : ℝ) : ℂ := m + Complex.I
def z₂ (m : ℝ) : ℂ := m + (m - 2) * Complex.I

-- Theorem 1: If z₂ is in the fourth quadrant, then 0 < m < 2
theorem z₂_in_fourth_quadrant (m : ℝ) :
  (z₂ m).re > 0 ∧ (z₂ m).im < 0 → 0 < m ∧ m < 2 := by sorry

-- Theorem 2: If z₂ = z₁ · ni, then (m = 1 and n = -1) or (m = -2 and n = 2)
theorem z₂_equals_z₁_times_ni (m n : ℝ) :
  z₂ m = z₁ m * (n * Complex.I) →
  (m = 1 ∧ n = -1) ∨ (m = -2 ∧ n = 2) := by sorry

end z₂_in_fourth_quadrant_z₂_equals_z₁_times_ni_l1137_113724


namespace vertices_form_parabola_l1137_113737

/-- A parabola in the family of parabolas described by y = x^2 + 2ax + a for all real a -/
structure Parabola where
  a : ℝ

/-- The vertex of a parabola in the family -/
def vertex (p : Parabola) : ℝ × ℝ :=
  (-p.a, p.a - p.a^2)

/-- The set of all vertices of parabolas in the family -/
def vertex_set : Set (ℝ × ℝ) :=
  {v | ∃ p : Parabola, v = vertex p}

/-- The equation of the curve on which the vertices lie -/
def vertex_curve (x y : ℝ) : Prop :=
  y = -x^2 - x

theorem vertices_form_parabola :
  ∀ v ∈ vertex_set, vertex_curve v.1 v.2 := by
  sorry

#check vertices_form_parabola

end vertices_form_parabola_l1137_113737


namespace fgh_supermarket_count_l1137_113782

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 41

/-- The difference between the number of US and Canadian supermarkets -/
def difference : ℕ := 22

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - difference

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := us_supermarkets + canada_supermarkets

theorem fgh_supermarket_count : total_supermarkets = 60 := by
  sorry

end fgh_supermarket_count_l1137_113782


namespace smallest_angle_in_triangle_l1137_113733

theorem smallest_angle_in_triangle (x y z : ℝ) : 
  x + y + z = 180 →  -- Sum of angles in a triangle is 180°
  x + y = 45 →       -- Sum of two angles is 45°
  y = x - 5 →        -- One angle is 5° less than the other
  x > 0 ∧ y > 0 ∧ z > 0 →  -- All angles are positive
  min x (min y z) = 20 :=  -- The smallest angle is 20°
by sorry

end smallest_angle_in_triangle_l1137_113733


namespace nearest_integer_to_power_l1137_113705

theorem nearest_integer_to_power : 
  ∃ n : ℤ, n = 7414 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - (m : ℝ)| :=
by sorry

end nearest_integer_to_power_l1137_113705


namespace YZ_squared_equals_33_l1137_113764

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB BC CA : ℝ)
  (AB_pos : AB > 0)
  (BC_pos : BC > 0)
  (CA_pos : CA > 0)

/-- Circumcircle of a triangle -/
def Circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Incircle of a triangle -/
def Incircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Circle tangent to circumcircle and two sides of the triangle -/
def TangentCircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Intersection point of TangentCircle and Circumcircle -/
def X (t : Triangle) : ℝ × ℝ := sorry

/-- Points Y and Z on the circumcircle such that XY and YZ are tangent to the incircle -/
def Y (t : Triangle) : ℝ × ℝ := sorry
def Z (t : Triangle) : ℝ × ℝ := sorry

/-- Square of the distance between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ := sorry

theorem YZ_squared_equals_33 (t : Triangle) 
  (h1 : t.AB = 4) 
  (h2 : t.BC = 5) 
  (h3 : t.CA = 6) : 
  dist_squared (Y t) (Z t) = 33 := by sorry

end YZ_squared_equals_33_l1137_113764


namespace sugar_water_inequality_l1137_113726

theorem sugar_water_inequality (a b c d : ℝ) : 
  a > b ∧ b > 0 ∧ c > d ∧ d > 0 → 
  (b + d) / (a + d) < (b + c) / (a + c) ∧
  ∀ (a b : ℝ), a > 0 ∧ b > 0 → 
  a / (1 + a + b) + b / (1 + a + b) < a / (1 + a) + b / (1 + b) := by
sorry

end sugar_water_inequality_l1137_113726


namespace cycle_original_price_l1137_113767

/-- Given a cycle sold for Rs. 1080 with a gain of 8%, prove that the original price was Rs. 1000 -/
theorem cycle_original_price (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 1080)
  (h2 : gain_percentage = 8) :
  let original_price := selling_price / (1 + gain_percentage / 100)
  original_price = 1000 := by sorry

end cycle_original_price_l1137_113767


namespace linear_equation_implies_specific_value_l1137_113766

/-- 
If $2x^{2a-b}-y^{a+b-1}=3$ is a linear equation in $x$ and $y$, 
then $(a-2b)^{2023} = -1$.
-/
theorem linear_equation_implies_specific_value (a b : ℝ) : 
  (∀ x y, ∃ k₁ k₂ c : ℝ, 2 * x^(2*a-b) - y^(a+b-1) = k₁ * x + k₂ * y + c) → 
  (a - 2*b)^2023 = -1 := by
  sorry

end linear_equation_implies_specific_value_l1137_113766


namespace symmetric_sine_extreme_value_l1137_113795

/-- Given a function f(x) = 2sin(ωx + φ) that satisfies f(π/4 + x) = f(π/4 - x) for all x,
    prove that f(π/4) equals either 2 or -2. -/
theorem symmetric_sine_extreme_value 
  (ω φ : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * Real.sin (ω * x + φ)) 
  (h2 : ∀ x, f (π/4 + x) = f (π/4 - x)) : 
  f (π/4) = 2 ∨ f (π/4) = -2 :=
sorry

end symmetric_sine_extreme_value_l1137_113795


namespace continuous_n_times_iff_odd_l1137_113736

/-- A function that takes every real value exactly n times. -/
def ExactlyNTimes (f : ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ∧ (∃ (S : Finset ℝ), S.card = n ∧ ∀ x : ℝ, f x = y ↔ x ∈ S)

/-- Main theorem: A continuous function that takes every real value exactly n times exists if and only if n is odd. -/
theorem continuous_n_times_iff_odd (n : ℕ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ ExactlyNTimes f n) ↔ Odd n :=
sorry


end continuous_n_times_iff_odd_l1137_113736


namespace imaginary_part_of_z_l1137_113755

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) (h1 : i ^ 2 = -1) (h2 : (1 + i) * z = i) : 
  z.im = 1 / 2 := by sorry

end imaginary_part_of_z_l1137_113755


namespace complex_equality_implies_modulus_l1137_113740

theorem complex_equality_implies_modulus (x y : ℝ) :
  (1 : ℂ) + x * Complex.I = (2 - y : ℂ) - 3 * Complex.I →
  Complex.abs (x + y * Complex.I) = Real.sqrt 10 := by
  sorry

end complex_equality_implies_modulus_l1137_113740


namespace rectangle_division_l1137_113776

theorem rectangle_division (a b : ℝ) (h1 : a + b = 50) (h2 : 7 * b + 10 * a = 434) :
  2 * (a / 8 + b / 11) = 11 := by
  sorry

end rectangle_division_l1137_113776


namespace right_triangle_9_40_41_l1137_113765

theorem right_triangle_9_40_41 : 
  ∀ (a b c : ℝ), a = 9 ∧ b = 40 ∧ c = 41 → a^2 + b^2 = c^2 :=
by
  sorry

#check right_triangle_9_40_41

end right_triangle_9_40_41_l1137_113765


namespace machine_selling_price_l1137_113777

/-- Calculates the selling price of a machine given its costs and profit percentage -/
def calculate_selling_price (purchase_price repair_cost transportation_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transportation_charges
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 28500 Rs -/
theorem machine_selling_price :
  calculate_selling_price 13000 5000 1000 50 = 28500 := by
  sorry

#eval calculate_selling_price 13000 5000 1000 50

end machine_selling_price_l1137_113777


namespace songs_on_mp3_player_l1137_113756

theorem songs_on_mp3_player (initial : ℕ) (deleted : ℕ) (added : ℕ) :
  initial ≥ deleted →
  (initial - deleted + added : ℕ) = initial - deleted + added :=
by sorry

end songs_on_mp3_player_l1137_113756


namespace binomial_8_choose_5_l1137_113793

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_8_choose_5_l1137_113793


namespace line_point_k_value_l1137_113772

/-- A line contains the points (6,8), (-2,k), and (-10,4). Prove that k = 6. -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 8 = m * 6 + b ∧ k = m * (-2) + b ∧ 4 = m * (-10) + b) → k = 6 := by
sorry

end line_point_k_value_l1137_113772


namespace value_range_of_f_l1137_113717

-- Define the function
def f (x : ℝ) : ℝ := |x - 3| + 1

-- Define the domain
def domain : Set ℝ := Set.Icc 0 9

-- Theorem statement
theorem value_range_of_f :
  Set.Icc 1 7 = (Set.image f domain) := by sorry

end value_range_of_f_l1137_113717


namespace fill_large_bottle_l1137_113799

/-- The volume of shampoo in milliliters that a medium-sized bottle holds -/
def medium_bottle_volume : ℕ := 150

/-- The volume of shampoo in milliliters that a large bottle holds -/
def large_bottle_volume : ℕ := 1200

/-- The number of medium-sized bottles needed to fill a large bottle -/
def bottles_needed : ℕ := large_bottle_volume / medium_bottle_volume

theorem fill_large_bottle : bottles_needed = 8 := by
  sorry

end fill_large_bottle_l1137_113799


namespace absolute_value_sum_difference_l1137_113762

theorem absolute_value_sum_difference (a b : ℝ) 
  (ha : |a| = 4) (hb : |b| = 3) : 
  ((a * b < 0 → |a + b| = 1) ∧ (a * b > 0 → |a - b| = 1)) := by
  sorry

end absolute_value_sum_difference_l1137_113762


namespace x_and_a_ranges_l1137_113701

theorem x_and_a_ranges (x m a : ℝ) 
  (h1 : x^2 - 4*a*x + 3*a^2 < 0)
  (h2 : x = (1/2)^(m-1))
  (h3 : 1 < m ∧ m < 2) :
  (a = 1/4 → 1/2 < x ∧ x < 3/4) ∧
  (1/3 ≤ a ∧ a ≤ 1/2) :=
sorry

end x_and_a_ranges_l1137_113701


namespace largest_constant_for_good_array_l1137_113794

def isGoodArray (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a < b ∧
  Nat.lcm a b + Nat.lcm (a + 2) (b + 2) = 2 * Nat.lcm (a + 1) (b + 1)

theorem largest_constant_for_good_array :
  (∃ c : ℚ, c > 0 ∧
    (∀ a b : ℕ, isGoodArray a b → b > c * a^3) ∧
    (∀ ε > 0, ∃ a b : ℕ, isGoodArray a b ∧ b ≤ (c + ε) * a^3)) ∧
  (let c := (1/2 : ℚ); 
   (∀ a b : ℕ, isGoodArray a b → b > c * a^3) ∧
   (∀ ε > 0, ∃ a b : ℕ, isGoodArray a b ∧ b ≤ (c + ε) * a^3)) :=
by sorry

end largest_constant_for_good_array_l1137_113794


namespace right_trapezoid_area_l1137_113715

/-- The area of a right trapezoid with specific dimensions -/
theorem right_trapezoid_area (upper_base lower_base height : ℝ) 
  (h1 : upper_base = 25)
  (h2 : lower_base - 15 = height)
  (h3 : height > 0) : 
  (upper_base + lower_base) * height / 2 = 175 := by
  sorry

end right_trapezoid_area_l1137_113715


namespace podium_cube_count_theorem_l1137_113716

/-- Represents a three-step podium made of wooden cubes -/
structure Podium where
  total_cubes : ℕ
  no_white_faces : ℕ
  one_white_face : ℕ
  two_white_faces : ℕ
  three_white_faces : ℕ

/-- The podium is valid if it satisfies the conditions of the problem -/
def is_valid_podium (p : Podium) : Prop :=
  p.total_cubes = 144 ∧
  p.no_white_faces = 40 ∧
  p.one_white_face = 64 ∧
  p.two_white_faces = 32 ∧
  p.three_white_faces = 8

/-- Theorem stating that the sum of cubes with 0, 1, 2, and 3 white faces
    equals the total number of cubes, implying no cubes with 4, 5, or 6 white faces -/
theorem podium_cube_count_theorem (p : Podium) (h : is_valid_podium p) :
  p.no_white_faces + p.one_white_face + p.two_white_faces + p.three_white_faces = p.total_cubes :=
by sorry


end podium_cube_count_theorem_l1137_113716


namespace subscription_period_l1137_113754

/-- Proves that the subscription period is 18 months given the promotion conditions -/
theorem subscription_period (normal_price : ℚ) (discount_per_issue : ℚ) (total_discount : ℚ) :
  normal_price = 34 →
  discount_per_issue = 0.25 →
  total_discount = 9 →
  ∃ (period : ℕ), period * 2 * discount_per_issue = total_discount ∧ period = 18 :=
by sorry

end subscription_period_l1137_113754


namespace rain_probability_l1137_113706

theorem rain_probability (p_monday p_tuesday p_no_rain : ℝ) 
  (h1 : p_monday = 0.62)
  (h2 : p_tuesday = 0.54)
  (h3 : p_no_rain = 0.28)
  : p_monday + p_tuesday - (1 - p_no_rain) = 0.44 := by
  sorry

end rain_probability_l1137_113706


namespace absolute_value_inequality_solution_set_l1137_113758

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| ≤ 1} = Set.Icc 0 1 := by sorry

end absolute_value_inequality_solution_set_l1137_113758


namespace range_of_x0_l1137_113728

/-- The circle C: x^2 + y^2 = 3 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 3

/-- The line l: x + 3y - 6 = 0 -/
def Line (x y : ℝ) : Prop := x + 3*y - 6 = 0

/-- The angle between two vectors is 60 degrees -/
def AngleSixtyDegrees (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1*x2 + y1*y2) / (Real.sqrt (x1^2 + y1^2) * Real.sqrt (x2^2 + y2^2)) = 1/2

theorem range_of_x0 (x0 y0 : ℝ) :
  Line x0 y0 →
  (∃ x y, Circle x y ∧ AngleSixtyDegrees x0 y0 x y) →
  0 ≤ x0 ∧ x0 ≤ 6/5 := by sorry

end range_of_x0_l1137_113728


namespace complementary_angle_triple_l1137_113739

theorem complementary_angle_triple (x y : ℝ) : 
  x + y = 90 ∧ x = 3 * y → x = 67.5 := by
  sorry

end complementary_angle_triple_l1137_113739


namespace nicole_fish_tanks_water_needed_l1137_113792

theorem nicole_fish_tanks_water_needed :
  -- Define the number of tanks
  let total_tanks : ℕ := 4
  let first_group_tanks : ℕ := 2
  let second_group_tanks : ℕ := total_tanks - first_group_tanks

  -- Define water needed for each group
  let first_group_water : ℕ := 8
  let second_group_water : ℕ := first_group_water - 2

  -- Define the number of weeks
  let weeks : ℕ := 4

  -- Calculate total water needed per week
  let water_per_week : ℕ := first_group_tanks * first_group_water + second_group_tanks * second_group_water

  -- Calculate total water needed for four weeks
  let total_water : ℕ := water_per_week * weeks

  -- Prove that the total water needed is 112 gallons
  total_water = 112 := by sorry

end nicole_fish_tanks_water_needed_l1137_113792


namespace total_sprockets_produced_l1137_113797

-- Define the production rates and time difference
def machine_x_rate : ℝ := 5.999999999999999
def machine_b_rate : ℝ := machine_x_rate * 1.1
def time_difference : ℝ := 10

-- Define the theorem
theorem total_sprockets_produced :
  ∃ (time_b : ℝ),
    time_b > 0 ∧
    (machine_x_rate * (time_b + time_difference) = machine_b_rate * time_b) ∧
    (machine_x_rate * (time_b + time_difference) + machine_b_rate * time_b = 1320) := by
  sorry


end total_sprockets_produced_l1137_113797


namespace inequality_proof_l1137_113703

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end inequality_proof_l1137_113703


namespace tangent_inclination_range_l1137_113711

open Real

theorem tangent_inclination_range (x : ℝ) : 
  let y := sin x
  let slope := cos x
  let θ := arctan slope
  0 ≤ θ ∧ θ < π ∧ (θ ≤ π/4 ∨ 3*π/4 ≤ θ) := by sorry

end tangent_inclination_range_l1137_113711


namespace eel_cost_l1137_113780

theorem eel_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : E = 180 := by
  sorry

end eel_cost_l1137_113780


namespace largest_good_number_all_greater_bad_smallest_bad_number_all_lesser_good_l1137_113752

/-- Definition of a good number -/
def is_good (M : ℕ) : Prop :=
  ∃ a b c d : ℤ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

/-- 576 is a good number -/
theorem largest_good_number : is_good 576 := by sorry

/-- All numbers greater than 576 are bad numbers -/
theorem all_greater_bad (M : ℕ) : M > 576 → ¬ is_good M := by sorry

/-- 443 is a bad number -/
theorem smallest_bad_number : ¬ is_good 443 := by sorry

/-- All numbers less than 443 are good numbers -/
theorem all_lesser_good (M : ℕ) : M < 443 → is_good M := by sorry

end largest_good_number_all_greater_bad_smallest_bad_number_all_lesser_good_l1137_113752


namespace simplify_expression_1_l1137_113773

theorem simplify_expression_1 (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (x^2 / (-2*y)) * (6*x*y^2 / x^4) = -3*y/x :=
sorry

end simplify_expression_1_l1137_113773


namespace fermat_number_large_prime_factor_l1137_113723

/-- Fermat number -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Theorem: For n ≥ 3, F_n has a prime factor greater than 2^(n+2)(n+1) -/
theorem fermat_number_large_prime_factor (n : ℕ) (h : n ≥ 3) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ F n ∧ p > 2^(n+2) * (n+1) := by
  sorry

end fermat_number_large_prime_factor_l1137_113723


namespace intersection_of_A_and_B_l1137_113786

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l1137_113786


namespace mean_interior_angle_quadrilateral_l1137_113738

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- Formula for the sum of interior angles of a polygon -/
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The mean value of interior angles of a quadrilateral -/
theorem mean_interior_angle_quadrilateral :
  (sum_of_interior_angles quadrilateral_sides) / quadrilateral_sides = 90 := by
  sorry

end mean_interior_angle_quadrilateral_l1137_113738


namespace max_sum_given_sum_of_squares_and_product_l1137_113751

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 130 → xy = 45 → x + y ≤ 10 * Real.sqrt 2.2 := by
  sorry

end max_sum_given_sum_of_squares_and_product_l1137_113751


namespace no_consecutive_red_probability_l1137_113759

def num_lights : ℕ := 8
def red_prob : ℝ := 0.4
def green_prob : ℝ := 1 - red_prob

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def prob_no_consecutive_red : ℝ :=
  (green_prob ^ num_lights) * (binomial (num_lights + 1 - 0) 0) +
  (green_prob ^ 7) * red_prob * (binomial (num_lights + 1 - 1) 1) +
  (green_prob ^ 6) * (red_prob ^ 2) * (binomial (num_lights + 1 - 2) 2) +
  (green_prob ^ 5) * (red_prob ^ 3) * (binomial (num_lights + 1 - 3) 3) +
  (green_prob ^ 4) * (red_prob ^ 4) * (binomial (num_lights + 1 - 4) 4)

theorem no_consecutive_red_probability :
  prob_no_consecutive_red = 0.3499456 := by
  sorry

end no_consecutive_red_probability_l1137_113759


namespace sector_circumference_l1137_113725

/-- The circumference of a sector with central angle 60° and radius 15 cm is 5(6 + π) cm. -/
theorem sector_circumference :
  let θ : ℝ := 60  -- Central angle in degrees
  let r : ℝ := 15  -- Radius in cm
  let arc_length : ℝ := (θ / 360) * (2 * π * r)
  let circumference : ℝ := arc_length + 2 * r
  circumference = 5 * (6 + π) := by sorry

end sector_circumference_l1137_113725
