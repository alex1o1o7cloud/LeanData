import Mathlib

namespace birthday_cake_division_l4005_400596

/-- Calculates the weight of cake each of Juelz's sisters received after the birthday party -/
theorem birthday_cake_division (total_pieces : ℕ) (square_pieces : ℕ) (triangle_pieces : ℕ)
  (square_weight : ℕ) (triangle_weight : ℕ) (square_eaten_percent : ℚ) 
  (triangle_eaten_percent : ℚ) (forest_family_percent : ℚ) (friends_percent : ℚ) 
  (num_sisters : ℕ) :
  total_pieces = square_pieces + triangle_pieces →
  square_pieces = 160 →
  triangle_pieces = 80 →
  square_weight = 25 →
  triangle_weight = 20 →
  square_eaten_percent = 60 / 100 →
  triangle_eaten_percent = 40 / 100 →
  forest_family_percent = 30 / 100 →
  friends_percent = 25 / 100 →
  num_sisters = 3 →
  ∃ (sisters_share : ℕ), sisters_share = 448 ∧
    sisters_share = 
      ((1 - friends_percent) * 
       ((1 - forest_family_percent) * 
        ((square_pieces * (1 - square_eaten_percent) * square_weight) + 
         (triangle_pieces * (1 - triangle_eaten_percent) * triangle_weight)))) / num_sisters :=
by sorry

end birthday_cake_division_l4005_400596


namespace completing_square_equivalence_l4005_400561

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 :=
by sorry

end completing_square_equivalence_l4005_400561


namespace power_of_two_equality_l4005_400512

theorem power_of_two_equality : (16^3) * (4^4) * (32^2) = 2^30 := by
  sorry

end power_of_two_equality_l4005_400512


namespace first_term_of_geometric_series_l4005_400526

/-- Given an infinite geometric series with common ratio -1/3 and sum 18,
    the first term of the series is 24. -/
theorem first_term_of_geometric_series :
  ∀ (a : ℝ), 
    (∃ (S : ℝ), S = 18 ∧ S = a / (1 - (-1/3))) →
    a = 24 := by
  sorry

end first_term_of_geometric_series_l4005_400526


namespace total_brownies_l4005_400580

/-- The number of brownies Tina ate per day -/
def tina_daily : ℕ := 2

/-- The number of days Tina ate brownies -/
def days : ℕ := 5

/-- The number of brownies Tina's husband ate per day -/
def husband_daily : ℕ := 1

/-- The number of brownies shared with guests -/
def shared : ℕ := 4

/-- The number of brownies left -/
def left : ℕ := 5

/-- Theorem stating the total number of brownie pieces -/
theorem total_brownies : 
  tina_daily * days + husband_daily * days + shared + left = 24 := by
  sorry

end total_brownies_l4005_400580


namespace max_garden_area_l4005_400567

/-- Represents a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- The perimeter of the garden -/
def Garden.perimeter (g : Garden) : ℝ := 2 * (g.length + g.width)

/-- The area of the garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the maximum area of a rectangular garden with given constraints -/
theorem max_garden_area (g : Garden) 
  (h_perimeter : g.perimeter = 400) 
  (h_min_length : g.length ≥ 100) : 
  g.area ≤ 10000 ∧ (g.area = 10000 ↔ g.length = 100 ∧ g.width = 100) := by
  sorry

#check max_garden_area

end max_garden_area_l4005_400567


namespace square_root_sum_implies_product_l4005_400506

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (7 + x) + Real.sqrt (25 - x) = 8) →
  ((7 + x) * (25 - x) = 256) := by
sorry

end square_root_sum_implies_product_l4005_400506


namespace union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l4005_400507

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 < x ≤ 8}
theorem union_A_B : A ∪ B = {x | 1 < x ∧ x ≤ 8} := by sorry

-- Theorem 2: (∁ᵤA) ∩ B = {x | 1 < x < 2}
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = {x | 1 < x ∧ x < 2} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a < 8
theorem intersection_A_C_nonempty (a : ℝ) : (A ∩ C a).Nonempty → a < 8 := by sorry

end union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l4005_400507


namespace solve_equation_l4005_400501

theorem solve_equation : ∃ x : ℝ, 5 * (x - 9) = 6 * (3 - 3 * x) + 6 ∧ x = 3 := by
  sorry

end solve_equation_l4005_400501


namespace eighteen_wheel_truck_toll_l4005_400569

/-- Calculates the toll for a truck based on the number of axles -/
def toll (axles : ℕ) : ℚ :=
  1.50 + 0.50 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels -/
def axles_count (wheels : ℕ) : ℕ :=
  wheels / 2

theorem eighteen_wheel_truck_toll :
  let wheels : ℕ := 18
  let axles : ℕ := axles_count wheels
  toll axles = 5 := by sorry

end eighteen_wheel_truck_toll_l4005_400569


namespace pears_picked_total_l4005_400509

/-- The number of pears Sara picked -/
def sara_pears : ℕ := 45

/-- The number of pears Sally picked -/
def sally_pears : ℕ := 11

/-- The total number of pears picked -/
def total_pears : ℕ := sara_pears + sally_pears

theorem pears_picked_total : total_pears = 56 := by
  sorry

end pears_picked_total_l4005_400509


namespace cubic_difference_l4005_400578

theorem cubic_difference (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 50) :
  a^3 - b^3 = 342 := by
  sorry

end cubic_difference_l4005_400578


namespace sum_of_p_x_coordinates_l4005_400556

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (a b c : Point) : ℝ := sorry

/-- Theorem: Sum of possible x-coordinates of P -/
theorem sum_of_p_x_coordinates : ∃ (P₁ P₂ P₃ P₄ : Point),
  let Q : Point := ⟨0, 0⟩
  let R : Point := ⟨368, 0⟩
  let S₁ : Point := ⟨901, 501⟩
  let S₂ : Point := ⟨912, 514⟩
  triangleArea P₁ Q R = 4128 ∧
  triangleArea P₂ Q R = 4128 ∧
  triangleArea P₃ Q R = 4128 ∧
  triangleArea P₄ Q R = 4128 ∧
  (triangleArea P₁ R S₁ = 12384 ∨ triangleArea P₁ R S₂ = 12384) ∧
  (triangleArea P₂ R S₁ = 12384 ∨ triangleArea P₂ R S₂ = 12384) ∧
  (triangleArea P₃ R S₁ = 12384 ∨ triangleArea P₃ R S₂ = 12384) ∧
  (triangleArea P₄ R S₁ = 12384 ∨ triangleArea P₄ R S₂ = 12384) ∧
  P₁.x + P₂.x + P₃.x + P₄.x = 4000 :=
sorry

end sum_of_p_x_coordinates_l4005_400556


namespace train_crossing_time_l4005_400517

/-- Given a train and a platform with specific dimensions and time to pass,
    calculate the time it takes for the train to cross a tree. -/
theorem train_crossing_time (train_length platform_length time_to_pass_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 1100)
  (h3 : time_to_pass_platform = 230) :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / time_to_pass_platform
  train_length / train_speed = 120 := by
  sorry

end train_crossing_time_l4005_400517


namespace inequality_and_equality_condition_l4005_400538

theorem inequality_and_equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b + b / c + c / a + 1 ≥ (2 * Real.sqrt 2 / 3) * (Real.sqrt ((a + b) / c) + Real.sqrt ((b + c) / a) + Real.sqrt ((c + a) / b))) ∧
  (a / b + b / c + c / a + 1 = (2 * Real.sqrt 2 / 3) * (Real.sqrt ((a + b) / c) + Real.sqrt ((b + c) / a) + Real.sqrt ((c + a) / b)) ↔ a = b ∧ b = c) :=
by sorry

end inequality_and_equality_condition_l4005_400538


namespace square_sum_from_difference_and_sum_squares_l4005_400568

theorem square_sum_from_difference_and_sum_squares 
  (m n : ℝ) 
  (h1 : (m - n)^2 = 8) 
  (h2 : (m + n)^2 = 2) : 
  m^2 + n^2 = 5 := by
sorry

end square_sum_from_difference_and_sum_squares_l4005_400568


namespace root_preservation_l4005_400546

-- Define the polynomial p(x) = x³ - 5x + 3
def p (x : ℚ) : ℚ := x^3 - 5*x + 3

-- Define a type for polynomials with rational coefficients
def RationalPolynomial := ℚ → ℚ

-- Theorem statement
theorem root_preservation 
  (α : ℚ) 
  (f : RationalPolynomial) 
  (h1 : p α = 0) 
  (h2 : p (f α) = 0) : 
  p (f (f α)) = 0 := by
  sorry

end root_preservation_l4005_400546


namespace zinc_weight_in_mixture_l4005_400597

/-- Given a mixture of zinc and copper in the ratio 9:11 with a total weight of 78 kg,
    the weight of zinc in the mixture is 35.1 kg. -/
theorem zinc_weight_in_mixture (zinc_ratio : ℚ) (copper_ratio : ℚ) (total_weight : ℚ) :
  zinc_ratio = 9 →
  copper_ratio = 11 →
  total_weight = 78 →
  (zinc_ratio / (zinc_ratio + copper_ratio)) * total_weight = 35.1 := by
  sorry

#check zinc_weight_in_mixture

end zinc_weight_in_mixture_l4005_400597


namespace min_argument_on_semicircle_l4005_400504

open Complex

noncomputable def min_argument : ℝ := Real.pi - Real.arctan (5 * Real.sqrt 6 / 12)

theorem min_argument_on_semicircle :
  ∀ z : ℂ, (abs z = 1 ∧ im z > 0) →
  arg ((z - 2) / (z + 3)) ≥ min_argument :=
by sorry

end min_argument_on_semicircle_l4005_400504


namespace negation_of_absolute_value_not_three_l4005_400515

theorem negation_of_absolute_value_not_three :
  (¬ ∀ x : ℤ, abs x ≠ 3) ↔ (∃ x : ℤ, abs x = 3) := by sorry

end negation_of_absolute_value_not_three_l4005_400515


namespace halloween_candy_weight_l4005_400554

/-- The combined weight of candy for Frank and Gwen -/
theorem halloween_candy_weight (frank_candy : ℕ) (gwen_candy : ℕ) 
  (h1 : frank_candy = 10) (h2 : gwen_candy = 7) : 
  frank_candy + gwen_candy = 17 := by
  sorry

end halloween_candy_weight_l4005_400554


namespace total_ingredients_for_batches_l4005_400523

/-- The amount of flour needed for one batch of cookies (in cups) -/
def flour_per_batch : ℝ := 4

/-- The amount of sugar needed for one batch of cookies (in cups) -/
def sugar_per_batch : ℝ := 1.5

/-- The number of batches we want to make -/
def num_batches : ℕ := 8

/-- Theorem: The total amount of flour and sugar combined needed for 8 batches is 44 cups -/
theorem total_ingredients_for_batches : 
  (flour_per_batch + sugar_per_batch) * num_batches = 44 := by
  sorry

end total_ingredients_for_batches_l4005_400523


namespace solution_set_part1_min_value_part2_l4005_400550

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 2 x ≥ 1 - 2*x} = {x : ℝ | x ≥ -1} := by sorry

-- Part 2
theorem min_value_part2 (a m n : ℝ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) 
  (h4 : m^2 * n = a) (h5 : ∀ x, f a x + |x - 1| ≥ 3) :
  ∃ (x : ℝ), m + n ≥ x ∧ x = 3 := by sorry

end solution_set_part1_min_value_part2_l4005_400550


namespace middle_number_proof_l4005_400529

theorem middle_number_proof (x y z : ℕ) 
  (sum_xy : x + y = 20)
  (sum_xz : x + z = 26)
  (sum_yz : y + z = 30) :
  y = 12 := by
  sorry

end middle_number_proof_l4005_400529


namespace absolute_value_inequality_solution_set_l4005_400575

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2 ∨ x > 0} := by sorry

end absolute_value_inequality_solution_set_l4005_400575


namespace sequence_sum_2000_l4005_400545

def sequence_sum (n : ℕ) : ℤ :=
  let group_sum := -1
  let num_groups := n / 6
  num_groups * group_sum

theorem sequence_sum_2000 :
  sequence_sum 2000 = -334 :=
sorry

end sequence_sum_2000_l4005_400545


namespace shirt_sale_problem_l4005_400524

/-- Shirt sale problem -/
theorem shirt_sale_problem 
  (total_shirts : ℕ) 
  (total_cost : ℕ) 
  (black_wholesale : ℕ) 
  (black_retail : ℕ) 
  (white_wholesale : ℕ) 
  (white_retail : ℕ) 
  (h1 : total_shirts = 200)
  (h2 : total_cost = 3500)
  (h3 : black_wholesale = 25)
  (h4 : black_retail = 50)
  (h5 : white_wholesale = 15)
  (h6 : white_retail = 35) :
  ∃ (black_count white_count : ℕ),
    black_count + white_count = total_shirts ∧
    black_count * black_wholesale + white_count * white_wholesale = total_cost ∧
    black_count = 50 ∧
    white_count = 150 ∧
    (black_count * (black_retail - black_wholesale) + 
     white_count * (white_retail - white_wholesale)) = 4250 :=
by sorry

end shirt_sale_problem_l4005_400524


namespace matching_times_correct_l4005_400581

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Calculates the total minutes elapsed since 00:00 -/
def totalMinutes (t : Time) : Nat :=
  t.hours * 60 + t.minutes

/-- Calculates the charge of the mortar at a given time -/
def charge (t : Time) : Nat :=
  100 - (totalMinutes t) / 6

/-- The list of times when the charge equals the number of minutes -/
def matchingTimes : List Time := [
  ⟨4, 52, by sorry, by sorry⟩,
  ⟨5, 43, by sorry, by sorry⟩,
  ⟨6, 35, by sorry, by sorry⟩,
  ⟨7, 26, by sorry, by sorry⟩,
  ⟨9, 9, by sorry, by sorry⟩
]

/-- Theorem stating that the matching times are correct -/
theorem matching_times_correct :
  ∀ t ∈ matchingTimes, charge t = t.minutes :=
by sorry

end matching_times_correct_l4005_400581


namespace unique_integer_product_of_digits_l4005_400589

/-- Given a positive integer n, returns the product of its digits -/
def productOfDigits (n : ℕ+) : ℕ := sorry

/-- Theorem: The only positive integer n whose product of digits equals n^2 - 15n - 27 is 17 -/
theorem unique_integer_product_of_digits : 
  ∃! (n : ℕ+), productOfDigits n = n^2 - 15*n - 27 ∧ n = 17 := by sorry

end unique_integer_product_of_digits_l4005_400589


namespace apples_left_is_ten_l4005_400587

/-- The number of apples left in the cafeteria -/
def apples_left : ℕ := sorry

/-- The initial number of apples -/
def initial_apples : ℕ := 50

/-- The initial number of oranges -/
def initial_oranges : ℕ := 40

/-- The cost of an apple in cents -/
def apple_cost : ℕ := 80

/-- The cost of an orange in cents -/
def orange_cost : ℕ := 50

/-- The total earnings from apples and oranges in cents -/
def total_earnings : ℕ := 4900

/-- The number of oranges left -/
def oranges_left : ℕ := 6

/-- Theorem stating that the number of apples left is 10 -/
theorem apples_left_is_ten :
  apples_left = 10 ∧
  initial_apples * apple_cost - apples_left * apple_cost +
  (initial_oranges - oranges_left) * orange_cost = total_earnings :=
sorry

end apples_left_is_ten_l4005_400587


namespace julie_reading_ratio_l4005_400560

theorem julie_reading_ratio : 
  ∀ (total_pages pages_yesterday pages_tomorrow : ℕ) (pages_today : ℕ),
    total_pages = 120 →
    pages_yesterday = 12 →
    pages_tomorrow = 42 →
    2 * pages_tomorrow = total_pages - pages_yesterday - pages_today →
    pages_today / pages_yesterday = 2 := by
  sorry

end julie_reading_ratio_l4005_400560


namespace difference_of_squares_65_35_l4005_400593

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end difference_of_squares_65_35_l4005_400593


namespace division_subtraction_equality_l4005_400571

theorem division_subtraction_equality : 144 / (12 / 3) - 5 = 31 := by
  sorry

end division_subtraction_equality_l4005_400571


namespace inequality_proof_l4005_400565

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  1 / (a^2 + b^2) ≤ 1/8 := by
  sorry

end inequality_proof_l4005_400565


namespace smallest_n_congruence_l4005_400562

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 29 * n ≡ 5678 [ZMOD 11] ∧ ∀ m : ℕ, (0 < m ∧ m < n) → ¬(29 * m ≡ 5678 [ZMOD 11])) ↔ 
  n = 9 := by
sorry

end smallest_n_congruence_l4005_400562


namespace integer_solutions_quadratic_equation_l4005_400503

theorem integer_solutions_quadratic_equation :
  ∀ a b : ℤ, 7 * a + 14 * b = 5 * a^2 + 5 * a * b + 5 * b^2 ↔
    (a = -1 ∧ b = 3) ∨ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 2) := by
  sorry

end integer_solutions_quadratic_equation_l4005_400503


namespace total_dogs_count_l4005_400563

/-- The number of boxes containing stuffed toy dogs -/
def num_boxes : ℕ := 7

/-- The number of dogs in each box -/
def dogs_per_box : ℕ := 4

/-- The total number of dogs -/
def total_dogs : ℕ := num_boxes * dogs_per_box

theorem total_dogs_count : total_dogs = 28 := by
  sorry

end total_dogs_count_l4005_400563


namespace basin_fill_time_l4005_400583

def right_eye_rate : ℚ := 1 / 48
def left_eye_rate : ℚ := 1 / 72
def right_foot_rate : ℚ := 1 / 96
def throat_rate : ℚ := 1 / 6

def combined_rate : ℚ := right_eye_rate + left_eye_rate + right_foot_rate + throat_rate

theorem basin_fill_time :
  (1 : ℚ) / combined_rate = 288 / 61 := by sorry

end basin_fill_time_l4005_400583


namespace truck_distance_l4005_400572

/-- Proves the distance traveled by a truck in yards over 5 minutes -/
theorem truck_distance (b t : ℝ) (h1 : t > 0) : 
  let feet_per_t_seconds : ℝ := b / 4
  let feet_in_yard : ℝ := 2
  let minutes : ℝ := 5
  let seconds_in_minute : ℝ := 60
  let yards_traveled : ℝ := (feet_per_t_seconds * (minutes * seconds_in_minute) / t) / feet_in_yard
  yards_traveled = 37.5 * b / t := by
sorry

end truck_distance_l4005_400572


namespace final_produce_theorem_l4005_400531

/-- Represents the quantity of produce -/
structure Produce where
  potatoes : ℕ
  cantaloupes : ℕ
  cucumbers : ℕ

/-- Calculates the final quantity of produce after various events -/
def finalProduce (initial : Produce) : Produce :=
  let potatoesAfterRabbits := initial.potatoes - initial.potatoes / 2
  let cantaloupesAfterSquirrels := initial.cantaloupes - initial.cantaloupes / 4
  let cantaloupesAfterGift := cantaloupesAfterSquirrels + initial.cantaloupes / 2
  let cucumbersAfterRabbits := initial.cucumbers - 2
  let cucumbersAfterHarvest := cucumbersAfterRabbits - (cucumbersAfterRabbits * 3) / 4
  { potatoes := potatoesAfterRabbits,
    cantaloupes := cantaloupesAfterGift,
    cucumbers := cucumbersAfterHarvest }

theorem final_produce_theorem (initial : Produce) :
  initial.potatoes = 7 ∧ initial.cantaloupes = 4 ∧ initial.cucumbers = 5 →
  finalProduce initial = { potatoes := 4, cantaloupes := 5, cucumbers := 1 } :=
by sorry

end final_produce_theorem_l4005_400531


namespace circle_condition_l4005_400594

/-- The equation x^2 + y^2 - 2x + 6y + m = 0 represents a circle if and only if m < 10 -/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + 6*y + m = 0 ∧ 
   ∃ (h k r : ℝ), ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - 2*x + 6*y + m = 0) 
  ↔ m < 10 :=
sorry

end circle_condition_l4005_400594


namespace increasing_sequence_range_l4005_400539

def a_n (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 7 then (4 - a) * n - 10 else a^(n - 6)

theorem increasing_sequence_range (a : ℝ) :
  (∀ n m : ℕ, n < m → a_n a n < a_n a m) →
  2 < a ∧ a < 4 :=
sorry

end increasing_sequence_range_l4005_400539


namespace tylers_age_l4005_400528

theorem tylers_age (clay jessica alex tyler : ℕ) : 
  tyler = 3 * clay + 1 →
  jessica = 2 * tyler - 4 →
  alex = (clay + jessica) / 2 →
  clay + jessica + alex + tyler = 52 →
  tyler = 13 := by
sorry

end tylers_age_l4005_400528


namespace profit_at_55_profit_price_relationship_optimal_price_l4005_400559

-- Define the constants and variables
def sales_cost : ℝ := 40
def initial_price : ℝ := 50
def initial_volume : ℝ := 500
def volume_decrease_rate : ℝ := 10

-- Define the sales volume function
def sales_volume (price : ℝ) : ℝ :=
  initial_volume - volume_decrease_rate * (price - initial_price)

-- Define the profit function
def profit (price : ℝ) : ℝ :=
  (price - sales_cost) * sales_volume price

-- Theorem 1: Monthly sales profit at $55 per kilogram
theorem profit_at_55 :
  profit 55 = 6750 := by sorry

-- Theorem 2: Relationship between profit and price
theorem profit_price_relationship (price : ℝ) :
  profit price = -10 * price^2 + 1400 * price - 40000 := by sorry

-- Theorem 3: Optimal price for $8000 profit without exceeding $10000 cost
theorem optimal_price :
  ∃ (price : ℝ),
    profit price = 8000 ∧
    sales_volume price * sales_cost ≤ 10000 ∧
    price = 80 := by sorry

end profit_at_55_profit_price_relationship_optimal_price_l4005_400559


namespace max_polyline_length_6x10_l4005_400590

/-- Represents a checkered field with rows and columns -/
structure CheckeredField where
  rows : Nat
  columns : Nat

/-- Represents a polyline on a checkered field -/
structure Polyline where
  field : CheckeredField
  length : Nat
  closed : Bool
  nonSelfIntersecting : Bool

/-- The maximum length of a closed, non-self-intersecting polyline on a given field -/
def maxPolylineLength (field : CheckeredField) : Nat :=
  sorry

/-- Theorem: The maximum length of a closed, non-self-intersecting polyline
    on a 6 × 10 checkered field is 76 -/
theorem max_polyline_length_6x10 :
  let field := CheckeredField.mk 6 10
  maxPolylineLength field = 76 := by
  sorry

end max_polyline_length_6x10_l4005_400590


namespace permutations_with_non_adjacent_yellow_eq_11760_l4005_400558

/-- The number of permutations of 3 green, 2 red, 2 white, and 3 yellow balls
    where no two yellow balls are adjacent -/
def permutations_with_non_adjacent_yellow : ℕ :=
  let green : ℕ := 3
  let red : ℕ := 2
  let white : ℕ := 2
  let yellow : ℕ := 3
  let non_yellow : ℕ := green + red + white
  let gaps : ℕ := non_yellow + 1
  (Nat.factorial non_yellow / (Nat.factorial green * Nat.factorial red * Nat.factorial white)) *
  (Nat.choose gaps yellow)

theorem permutations_with_non_adjacent_yellow_eq_11760 :
  permutations_with_non_adjacent_yellow = 11760 := by
  sorry

end permutations_with_non_adjacent_yellow_eq_11760_l4005_400558


namespace renata_lottery_winnings_l4005_400516

/-- Represents the financial transactions of Renata --/
structure RenataMoney where
  initial : ℕ
  donation : ℕ
  charityWin : ℕ
  waterCost : ℕ
  lotteryCost : ℕ
  final : ℕ

/-- Calculates the lottery winnings based on Renata's transactions --/
def lotteryWinnings (r : RenataMoney) : ℕ :=
  r.final + r.donation + r.waterCost + r.lotteryCost - r.initial - r.charityWin

/-- Theorem stating that Renata's lottery winnings were $2 --/
theorem renata_lottery_winnings :
  let r : RenataMoney := {
    initial := 10,
    donation := 4,
    charityWin := 90,
    waterCost := 1,
    lotteryCost := 1,
    final := 94
  }
  lotteryWinnings r = 2 := by sorry

end renata_lottery_winnings_l4005_400516


namespace percentage_increase_l4005_400544

theorem percentage_increase (original : ℝ) (new : ℝ) : 
  original = 80 → new = 96 → (new - original) / original * 100 = 20 := by
  sorry

end percentage_increase_l4005_400544


namespace multiple_calculation_l4005_400557

theorem multiple_calculation (n a : ℕ) (m : ℚ) 
  (h1 : n = 16) 
  (h2 : a = 12) 
  (h3 : m * n - a = 20) : 
  m = 2 := by
  sorry

end multiple_calculation_l4005_400557


namespace six_chairs_three_people_l4005_400543

/-- The number of ways to arrange n people among m chairs in a row, with no two people adjacent -/
def nonadjacentArrangements (m n : ℕ) : ℕ :=
  if m ≤ n then 0
  else Nat.descFactorial (m - n + 1) n

theorem six_chairs_three_people :
  nonadjacentArrangements 6 3 = 24 := by
  sorry

end six_chairs_three_people_l4005_400543


namespace pizza_varieties_theorem_four_topping_combinations_l4005_400588

/-- Represents the number of base pizza flavors -/
def num_flavors : Nat := 4

/-- Represents the number of topping combinations -/
def num_topping_combinations : Nat := 4

/-- Represents the total number of pizza varieties -/
def total_varieties : Nat := 16

/-- Theorem stating that the number of pizza varieties is the product of 
    the number of flavors and the number of topping combinations -/
theorem pizza_varieties_theorem :
  num_flavors * num_topping_combinations = total_varieties := by
  sorry

/-- Definition of the possible topping combinations -/
inductive ToppingCombination
  | None
  | ExtraCheese
  | Mushrooms
  | ExtraCheeseAndMushrooms

/-- Theorem stating that there are exactly 4 topping combinations -/
theorem four_topping_combinations :
  (ToppingCombination.None :: ToppingCombination.ExtraCheese :: 
   ToppingCombination.Mushrooms :: ToppingCombination.ExtraCheeseAndMushrooms :: []).length = 
  num_topping_combinations := by
  sorry

end pizza_varieties_theorem_four_topping_combinations_l4005_400588


namespace infinite_geometric_sum_l4005_400536

/-- The sum of an infinite geometric sequence with first term 1 and common ratio -1/2 is 2/3 -/
theorem infinite_geometric_sum : 
  ∀ (a : ℕ → ℚ), 
  (a 0 = 1) → 
  (∀ n : ℕ, a (n + 1) = a n * (-1/2)) → 
  (∑' n, a n) = 2/3 :=
by sorry

end infinite_geometric_sum_l4005_400536


namespace extreme_value_implies_a_range_l4005_400540

/-- The function f(x) defined in terms of a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*x + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*a*x + 3

/-- Theorem: If f(x) has at least one extreme value point in (2, 3), then 5/4 < a < 5/3 -/
theorem extreme_value_implies_a_range (a : ℝ) :
  (∃ x : ℝ, 2 < x ∧ x < 3 ∧ f_deriv a x = 0) →
  (5/4 : ℝ) < a ∧ a < (5/3 : ℝ) :=
by
  sorry


end extreme_value_implies_a_range_l4005_400540


namespace problem_statement_l4005_400584

theorem problem_statement (m n : ℝ) (h1 : m ≠ n) (h2 : m^2 = n + 2) (h3 : n^2 = m + 2) :
  4 * m * n - m^3 - n^3 = 0 := by
sorry

end problem_statement_l4005_400584


namespace range_of_a_l4005_400573

def P : Set ℝ := {x : ℝ | x^2 ≤ 4}
def M (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) : P ∪ M a = P → a ∈ Set.Icc (-2) 2 := by
  sorry

end range_of_a_l4005_400573


namespace average_of_six_numbers_l4005_400502

theorem average_of_six_numbers (numbers : Fin 6 → ℝ) 
  (h1 : (numbers 0 + numbers 1) / 2 = 1.1)
  (h2 : (numbers 2 + numbers 3) / 2 = 1.4)
  (h3 : (numbers 4 + numbers 5) / 2 = 5) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 2.5 := by
  sorry

end average_of_six_numbers_l4005_400502


namespace percentage_of_whole_l4005_400521

theorem percentage_of_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = 40.25 ↔ part = 193.2 ∧ whole = 480 :=
sorry

end percentage_of_whole_l4005_400521


namespace isabel_songs_proof_l4005_400530

/-- The number of songs Isabel bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Proof that Isabel bought 72 songs -/
theorem isabel_songs_proof :
  total_songs 6 2 9 = 72 := by
  sorry

end isabel_songs_proof_l4005_400530


namespace possible_values_of_a_l4005_400532

theorem possible_values_of_a (a : ℝ) : 
  let P : Set ℝ := {-1, 2*a+1, a^2-1}
  0 ∈ P → a = -1/2 ∨ a = 1 :=
by sorry

end possible_values_of_a_l4005_400532


namespace james_toys_l4005_400585

theorem james_toys (toy_cars : ℕ) (toy_soldiers : ℕ) : 
  toy_cars = 20 → 
  toy_soldiers = 2 * toy_cars → 
  toy_cars + toy_soldiers = 60 := by
  sorry

end james_toys_l4005_400585


namespace p_current_age_is_fifteen_l4005_400522

/-- Given the age ratios of two people P and Q at different times, 
    prove that P's current age is 15 years. -/
theorem p_current_age_is_fifteen :
  ∀ (p q : ℕ),
  (p - 3) / (q - 3) = 4 / 3 →
  (p + 6) / (q + 6) = 7 / 6 →
  p = 15 := by
  sorry

end p_current_age_is_fifteen_l4005_400522


namespace max_servings_is_56_l4005_400541

/-- Represents the ingredients required for one serving of salad -/
structure ServingRequirement where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Represents the available ingredients in the restaurant's warehouse -/
structure AvailableIngredients where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Calculates the maximum number of servings that can be made -/
def maxServings (req : ServingRequirement) (avail : AvailableIngredients) : ℕ :=
  min (avail.cucumbers / req.cucumbers)
      (min (avail.tomatoes / req.tomatoes)
           (min (avail.brynza / req.brynza)
                (avail.peppers / req.peppers)))

/-- Theorem stating that the maximum number of servings is 56 -/
theorem max_servings_is_56 :
  let req := ServingRequirement.mk 2 2 75 1
  let avail := AvailableIngredients.mk 117 116 4200 60
  maxServings req avail = 56 := by sorry

end max_servings_is_56_l4005_400541


namespace angle_measure_l4005_400566

theorem angle_measure (θ φ : ℝ) : 
  (90 - θ) = 0.4 * (180 - θ) →  -- complement is 40% of supplement
  φ = 180 - θ →                 -- θ and φ form a linear pair
  φ = 2 * θ →                   -- φ is twice the size of θ
  θ = 30 := by
sorry

end angle_measure_l4005_400566


namespace last_two_digits_sum_l4005_400592

theorem last_two_digits_sum : (13^27 + 17^27) % 100 = 90 := by
  sorry

end last_two_digits_sum_l4005_400592


namespace boat_speed_in_still_water_l4005_400599

/-- 
Given a boat that travels at different speeds with and against a stream,
this theorem proves that its speed in still water is 6 km/hr.
-/
theorem boat_speed_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 7) 
  (h2 : speed_against_stream = 5) : 
  (speed_with_stream + speed_against_stream) / 2 = 6 := by
sorry


end boat_speed_in_still_water_l4005_400599


namespace absolute_value_inequality_solution_set_l4005_400577

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 500| ≤ 5} = {x : ℝ | 495 ≤ x ∧ x ≤ 505} := by
  sorry

end absolute_value_inequality_solution_set_l4005_400577


namespace diagonal_passes_900_cubes_l4005_400553

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: An internal diagonal of a 200 × 400 × 500 rectangular solid
    passes through 900 cubes -/
theorem diagonal_passes_900_cubes :
  cubes_passed 200 400 500 = 900 := by
  sorry

end diagonal_passes_900_cubes_l4005_400553


namespace initial_girls_count_initial_girls_count_proof_l4005_400511

theorem initial_girls_count : ℕ → ℕ → Prop :=
  fun b g =>
    (3 * (g - 20) = b) →
    (4 * (b - 60) = g - 20) →
    g = 42

-- The proof is omitted
theorem initial_girls_count_proof : ∃ b g : ℕ, initial_girls_count b g := by sorry

end initial_girls_count_initial_girls_count_proof_l4005_400511


namespace no_perfect_square_133_base_n_l4005_400519

/-- Represents a number in base n -/
def base_n (digits : List Nat) (n : Nat) : Nat :=
  digits.foldr (fun d acc => d + n * acc) 0

/-- Checks if a number is a perfect square -/
def is_perfect_square (m : Nat) : Prop :=
  ∃ k : Nat, k * k = m

theorem no_perfect_square_133_base_n :
  ¬∃ n : Nat, 5 ≤ n ∧ n ≤ 15 ∧ is_perfect_square (base_n [1, 3, 3] n) := by
  sorry

end no_perfect_square_133_base_n_l4005_400519


namespace amandas_family_painting_l4005_400555

/-- The number of walls each person should paint in Amanda's family house painting problem -/
theorem amandas_family_painting (
  total_rooms : ℕ)
  (rooms_with_four_walls : ℕ)
  (rooms_with_five_walls : ℕ)
  (family_size : ℕ)
  (h1 : total_rooms = rooms_with_four_walls + rooms_with_five_walls)
  (h2 : total_rooms = 9)
  (h3 : rooms_with_four_walls = 5)
  (h4 : rooms_with_five_walls = 4)
  (h5 : family_size = 5)
  : (4 * rooms_with_four_walls + 5 * rooms_with_five_walls) / family_size = 8 := by
  sorry

end amandas_family_painting_l4005_400555


namespace dance_students_l4005_400520

/-- Represents the number of students taking each elective in a school -/
structure SchoolElectives where
  total : ℕ
  art : ℕ
  music : ℕ
  dance : ℕ

/-- The properties of the school electives -/
def valid_electives (s : SchoolElectives) : Prop :=
  s.total = 400 ∧
  s.art = 200 ∧
  s.music = s.total / 5 ∧
  s.total = s.art + s.music + s.dance

/-- Theorem stating that the number of students taking dance is 120 -/
theorem dance_students (s : SchoolElectives) (h : valid_electives s) : s.dance = 120 := by
  sorry

end dance_students_l4005_400520


namespace football_team_members_l4005_400510

/-- The total number of members in a football team after new members join -/
def total_members (initial : ℕ) (new : ℕ) : ℕ :=
  initial + new

/-- Theorem stating that the total number of members in the football team is 59 -/
theorem football_team_members :
  total_members 42 17 = 59 := by sorry

end football_team_members_l4005_400510


namespace ceiling_negative_five_thirds_squared_l4005_400527

theorem ceiling_negative_five_thirds_squared : ⌈(-5/3)^2⌉ = 3 := by
  sorry

end ceiling_negative_five_thirds_squared_l4005_400527


namespace gcd_count_equals_fourteen_l4005_400549

theorem gcd_count_equals_fourteen : 
  (Finset.filter (fun n : ℕ => Nat.gcd 21 n = 7) (Finset.range 150)).card = 14 := by
  sorry

end gcd_count_equals_fourteen_l4005_400549


namespace master_craftsman_production_l4005_400518

/-- The number of parts manufactured by a master craftsman during a shift -/
def total_parts : ℕ := 210

/-- The number of parts manufactured in the first hour -/
def first_hour_parts : ℕ := 35

/-- The increase in production rate (parts per hour) -/
def rate_increase : ℕ := 15

/-- The time saved by increasing the production rate (in hours) -/
def time_saved : ℚ := 1.5

theorem master_craftsman_production :
  ∃ (N : ℕ),
    (N : ℚ) / first_hour_parts - (N : ℚ) / (first_hour_parts + rate_increase) = time_saved ∧
    total_parts = first_hour_parts + N :=
  sorry

end master_craftsman_production_l4005_400518


namespace polygon_sides_l4005_400586

theorem polygon_sides (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 := by
  sorry

end polygon_sides_l4005_400586


namespace u_equals_fib_l4005_400514

/-- Array I as defined in the problem -/
def array_I (n : ℕ) : Fin n → Fin 3 → ℕ :=
  λ i j => match j with
    | 0 => i + 1
    | 1 => i + 2
    | 2 => i + 3

/-- Number of SDRs for array I -/
def u (n : ℕ) : ℕ := sorry

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Theorem stating that u_n is equal to the (n+1)th Fibonacci number for n ≥ 2 -/
theorem u_equals_fib (n : ℕ) (h : n ≥ 2) : u n = fib (n + 1) := by
  sorry

end u_equals_fib_l4005_400514


namespace max_consecutive_integers_sum_55_l4005_400513

/-- The sum of n consecutive positive integers starting from a -/
def consecutiveSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Predicate to check if a sequence of n consecutive integers starting from a sums to 55 -/
def isValidSequence (a n : ℕ) : Prop :=
  a > 0 ∧ consecutiveSum a n = 55

theorem max_consecutive_integers_sum_55 :
  (∃ a : ℕ, isValidSequence a 10) ∧
  (∀ n : ℕ, n > 10 → ¬∃ a : ℕ, isValidSequence a n) :=
sorry

end max_consecutive_integers_sum_55_l4005_400513


namespace trees_planted_l4005_400533

theorem trees_planted (initial_trees final_trees : ℕ) (h1 : initial_trees = 13) (h2 : final_trees = 25) :
  final_trees - initial_trees = 12 := by
  sorry

end trees_planted_l4005_400533


namespace negative_fraction_comparison_l4005_400564

theorem negative_fraction_comparison : -5/4 > -4/3 := by
  sorry

end negative_fraction_comparison_l4005_400564


namespace one_real_root_condition_l4005_400576

/-- Given the equation lg(kx) = 2lg(x+1), this theorem states the condition for k
    such that the equation has only one real root. -/
theorem one_real_root_condition (k : ℝ) : 
  (∃! x : ℝ, Real.log (k * x) = 2 * Real.log (x + 1)) ↔ (k < 0 ∨ k = 4) :=
by sorry

end one_real_root_condition_l4005_400576


namespace conic_eccentricity_l4005_400542

-- Define the geometric sequence
def is_geometric_sequence (a : ℝ) : Prop := a * a = 81

-- Define the conic section
def conic_section (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 / a = 1

-- Define the eccentricity
def eccentricity (e : ℝ) (a : ℝ) : Prop :=
  (e = Real.sqrt 10 ∧ a = -9) ∨ (e = 2 * Real.sqrt 2 / 3 ∧ a = 9)

-- Theorem statement
theorem conic_eccentricity (a : ℝ) (e : ℝ) :
  is_geometric_sequence a →
  (∃ x y, conic_section a x y) →
  eccentricity e a :=
sorry

end conic_eccentricity_l4005_400542


namespace area_outside_inscribed_square_l4005_400574

def square_side_length : ℝ := 2

theorem area_outside_inscribed_square (square_side : ℝ) (h : square_side = square_side_length) :
  let circle_radius : ℝ := square_side * Real.sqrt 2 / 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let square_area : ℝ := square_side ^ 2
  circle_area - square_area = 2 * π - 4 := by
sorry

end area_outside_inscribed_square_l4005_400574


namespace matrix_equation_solution_l4005_400582

theorem matrix_equation_solution :
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  N ^ 2 - 3 • N + 2 • N = !![6, 12; 3, 6] := by
  sorry

end matrix_equation_solution_l4005_400582


namespace bobs_deli_cost_l4005_400595

/-- The total cost for a customer at Bob's Deli -/
def total_cost (sandwich_price soda_price : ℕ) (sandwich_quantity soda_quantity : ℕ) (discount_threshold discount_amount : ℕ) : ℕ :=
  let initial_total := sandwich_price * sandwich_quantity + soda_price * soda_quantity
  if initial_total > discount_threshold then
    initial_total - discount_amount
  else
    initial_total

/-- The theorem stating that the customer will pay $55 in total -/
theorem bobs_deli_cost : total_cost 5 3 7 10 50 10 = 55 := by
  sorry

end bobs_deli_cost_l4005_400595


namespace nine_times_reverse_is_9801_l4005_400525

def reverse_number (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

theorem nine_times_reverse_is_9801 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (n = 9 * reverse_number n) → n = 9801 :=
by sorry

end nine_times_reverse_is_9801_l4005_400525


namespace cubic_roots_eighth_power_sum_l4005_400534

theorem cubic_roots_eighth_power_sum (r s : ℂ) : 
  (r^3 - r^2 * Real.sqrt 5 - r + 1 = 0) → 
  (s^3 - s^2 * Real.sqrt 5 - s + 1 = 0) → 
  r^8 + s^8 = 47 := by
sorry

end cubic_roots_eighth_power_sum_l4005_400534


namespace sqrt_18_times_sqrt_72_l4005_400579

theorem sqrt_18_times_sqrt_72 : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end sqrt_18_times_sqrt_72_l4005_400579


namespace base_four_representation_of_256_l4005_400548

theorem base_four_representation_of_256 :
  (256 : ℕ).digits 4 = [0, 0, 0, 0, 1] :=
sorry

end base_four_representation_of_256_l4005_400548


namespace last_four_digits_of_5_pow_2016_l4005_400570

def last_four_digits (n : ℕ) : ℕ := n % 10000

def power_five_pattern : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2016 :
  last_four_digits (5^2016) = 0625 :=
sorry

end last_four_digits_of_5_pow_2016_l4005_400570


namespace horse_speed_around_square_field_l4005_400535

/-- Given a square field with area 900 km² and a horse that takes 10 hours to run around it,
    prove that the horse's speed is 12 km/h. -/
theorem horse_speed_around_square_field : 
  let field_area : ℝ := 900
  let time_to_run_around : ℝ := 10
  let horse_speed : ℝ := 4 * Real.sqrt field_area / time_to_run_around
  horse_speed = 12 := by
  sorry

end horse_speed_around_square_field_l4005_400535


namespace equation_solution_approximation_l4005_400508

theorem equation_solution_approximation : ∃ x : ℝ, 
  (2.5 * ((x * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2000.0000000000002) ∧ 
  (abs (x - 3.6) < 0.0000000000000005) := by
  sorry

end equation_solution_approximation_l4005_400508


namespace water_storage_solution_l4005_400598

/-- Represents the water storage problem with barrels and casks. -/
def WaterStorage (cask_capacity : ℕ) (barrel_count : ℕ) : Prop :=
  let barrel_capacity := 2 * cask_capacity + 3
  barrel_count * barrel_capacity = 172

/-- Theorem stating that given the problem conditions, the total water storage is 172 gallons. -/
theorem water_storage_solution :
  WaterStorage 20 4 := by
  sorry

end water_storage_solution_l4005_400598


namespace worker_speed_ratio_l4005_400547

/-- Given two workers a and b, where a is k times as fast as b, prove that k = 3 
    under the given conditions. -/
theorem worker_speed_ratio (k : ℝ) : 
  (∃ (rate_b : ℝ), 
    (k * rate_b + rate_b = 1 / 30) ∧ 
    (k * rate_b = 1 / 40)) → 
  k = 3 := by
  sorry

end worker_speed_ratio_l4005_400547


namespace arithmetic_sequence_sum_l4005_400552

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) := by
  sorry

end arithmetic_sequence_sum_l4005_400552


namespace geometric_sequence_problem_l4005_400505

theorem geometric_sequence_problem :
  ∀ (a b c d : ℝ),
    (a / b = b / c) →                   -- geometric sequence condition
    (c / d = b / c) →                   -- geometric sequence condition
    (a - b = 6) →                       -- difference between first and second
    (c - d = 5) →                       -- difference between third and fourth
    (a^2 + b^2 + c^2 + d^2 = 793) →     -- sum of squares
    ((a = 18 ∧ b = 12 ∧ c = 15 ∧ d = 10) ∨
     (a = -12 ∧ b = -18 ∧ c = -10 ∧ d = -15)) :=
by sorry

end geometric_sequence_problem_l4005_400505


namespace student_in_all_clubs_l4005_400551

theorem student_in_all_clubs (n : ℕ) (F G C : Finset (Fin n)) :
  n = 30 →
  F.card = 22 →
  G.card = 21 →
  C.card = 18 →
  ∃ s, s ∈ F ∩ G ∩ C :=
by
  sorry

end student_in_all_clubs_l4005_400551


namespace percent_difference_l4005_400500

theorem percent_difference (N M : ℝ) (h : N > 0) : 
  let N' := 1.5 * N
  100 - (M / N') * 100 = 100 - (200 * M) / (3 * N) :=
by sorry

end percent_difference_l4005_400500


namespace prop_range_m_l4005_400537

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- State the theorem
theorem prop_range_m : 
  ∀ m : ℝ, ¬(p m ∨ q m) → m ≥ 2 :=
by sorry

end prop_range_m_l4005_400537


namespace set_operation_result_l4005_400591

-- Define the sets A, B, and C
def A : Set ℕ := {0, 1, 2, 4, 5, 7}
def B : Set ℕ := {1, 3, 6, 8, 9}
def C : Set ℕ := {3, 7, 8}

-- State the theorem
theorem set_operation_result : (A ∪ B) ∩ C = {3, 7, 8} := by
  sorry

end set_operation_result_l4005_400591
