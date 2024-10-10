import Mathlib

namespace used_car_selections_l3805_380502

/-- Proves that given 16 cars, 24 clients, and each client selecting 2 cars, 
    each car must be selected 3 times. -/
theorem used_car_selections (cars : ℕ) (clients : ℕ) (selections_per_client : ℕ) 
    (h1 : cars = 16) 
    (h2 : clients = 24) 
    (h3 : selections_per_client = 2) : 
  (clients * selections_per_client) / cars = 3 := by
  sorry

#check used_car_selections

end used_car_selections_l3805_380502


namespace function_properties_l3805_380526

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + Real.cos x + a

theorem function_properties (a : ℝ) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f a (x + T) = f a x ∧ 
   ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f a (x + T') = f a x) → T ≤ T') ∧
  (∃ (M : ℝ), M = 3 ∧ (∀ (x : ℝ), f a x ≤ M) → a = 1) ∧
  (∀ (k : ℤ), ∀ (x y : ℝ), 
    2 * k * Real.pi - 2 * Real.pi / 3 ≤ x ∧ 
    x < y ∧ 
    y ≤ 2 * k * Real.pi + Real.pi / 3 → 
    f a x < f a y) := by
  sorry

end function_properties_l3805_380526


namespace intersection_of_A_and_B_l3805_380559

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end intersection_of_A_and_B_l3805_380559


namespace largest_garden_difference_l3805_380517

/-- Represents a rectangular garden with length and width in feet. -/
structure Garden where
  length : ℕ
  width : ℕ

/-- Calculates the area of a garden in square feet. -/
def gardenArea (g : Garden) : ℕ := g.length * g.width

/-- Alice's garden -/
def aliceGarden : Garden := { length := 30, width := 50 }

/-- Bob's garden -/
def bobGarden : Garden := { length := 35, width := 45 }

/-- Candace's garden -/
def candaceGarden : Garden := { length := 40, width := 40 }

theorem largest_garden_difference :
  let gardens := [aliceGarden, bobGarden, candaceGarden]
  let areas := gardens.map gardenArea
  let maxArea := areas.maximum?
  let minArea := areas.minimum?
  ∀ max min, maxArea = some max → minArea = some min →
    max - min = 100 := by sorry

end largest_garden_difference_l3805_380517


namespace total_birds_caught_l3805_380539

-- Define the number of birds hunted during the day
def day_hunt : ℕ := 15

-- Define the success rate during the day
def day_success_rate : ℚ := 3/5

-- Define the number of birds hunted at night
def night_hunt : ℕ := 25

-- Define the success rate at night
def night_success_rate : ℚ := 4/5

-- Define the relationship between day and night catches
def night_day_ratio : ℕ := 2

-- Theorem statement
theorem total_birds_caught :
  ⌊(day_hunt : ℚ) * day_success_rate⌋ +
  ⌊(night_hunt : ℚ) * night_success_rate⌋ = 29 := by
  sorry


end total_birds_caught_l3805_380539


namespace simplify_expression_l3805_380571

theorem simplify_expression (k : ℝ) (h : k ≠ 0) :
  ∃ (a b c : ℤ), (8 * k + 3 + 6 * k^2) + (5 * k^2 + 4 * k + 7) = a * k^2 + b * k + c ∧ a + b + c = 33 :=
by sorry

end simplify_expression_l3805_380571


namespace cody_dumplings_l3805_380561

theorem cody_dumplings (A B : ℕ) (P1 Q1 Q2 P2 : ℚ) : 
  A = 14 → 
  B = 20 → 
  P1 = 1/2 → 
  Q1 = 1/4 → 
  Q2 = 2/5 → 
  P2 = 3/20 → 
  ∃ (remaining : ℕ), remaining = 16 ∧ 
    remaining = A - Int.floor (P1 * A) - Int.floor (Q1 * (A - Int.floor (P1 * A))) + 
                B - Int.floor (Q2 * B) - 
                Int.floor (P2 * (A - Int.floor (P1 * A) - Int.floor (Q1 * (A - Int.floor (P1 * A))) + 
                                 B - Int.floor (Q2 * B))) :=
by sorry

end cody_dumplings_l3805_380561


namespace max_soap_boxes_in_carton_l3805_380563

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Represents a carton with base and top dimensions -/
structure Carton where
  base : Dimensions
  top : Dimensions
  height : ℕ

/-- Represents a soap box with its dimensions and weight -/
structure SoapBox where
  dimensions : Dimensions
  weight : ℕ

def carton : Carton := {
  base := { width := 25, length := 42, height := 0 },
  top := { width := 20, length := 35, height := 0 },
  height := 60
}

def soapBox : SoapBox := {
  dimensions := { width := 7, length := 6, height := 10 },
  weight := 3
}

def maxWeight : ℕ := 150

theorem max_soap_boxes_in_carton :
  let spaceConstraint := (carton.top.width / soapBox.dimensions.width) *
                         (carton.top.length / soapBox.dimensions.length) *
                         (carton.height / soapBox.dimensions.height)
  let weightConstraint := maxWeight / soapBox.weight
  min spaceConstraint weightConstraint = 50 := by
  sorry

end max_soap_boxes_in_carton_l3805_380563


namespace equal_cost_at_45_l3805_380511

/-- Represents the number of students in a class -/
def num_students : ℕ := 45

/-- Represents the original ticket price in yuan -/
def ticket_price : ℕ := 30

/-- Calculates the cost of Option 1 (20% discount for all) -/
def option1_cost (n : ℕ) : ℚ :=
  n * ticket_price * (4 / 5)

/-- Calculates the cost of Option 2 (10% discount and 5 free tickets) -/
def option2_cost (n : ℕ) : ℚ :=
  (n - 5) * ticket_price * (9 / 10)

/-- Theorem stating that for 45 students, the costs of both options are equal -/
theorem equal_cost_at_45 :
  option1_cost num_students = option2_cost num_students :=
by sorry

end equal_cost_at_45_l3805_380511


namespace divisible_pair_count_l3805_380585

/-- Given a set of 2117 cards numbered from 1 to 2117, this function calculates
    the number of ways to choose two cards such that their sum is divisible by 100. -/
def count_divisible_pairs : ℕ := 
  let card_count := 2117
  sorry

/-- Theorem stating that the number of ways to choose two cards with a sum
    divisible by 100 from a set of 2117 cards numbered 1 to 2117 is 23058. -/
theorem divisible_pair_count : count_divisible_pairs = 23058 := by
  sorry

end divisible_pair_count_l3805_380585


namespace negation_of_existential_proposition_l3805_380577

theorem negation_of_existential_proposition :
  (¬ (∃ x : ℝ, 2 * x - 3 > 1)) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) := by
  sorry

end negation_of_existential_proposition_l3805_380577


namespace sum_of_a_and_b_l3805_380573

/-- The smallest positive integer a such that 450 * a is a perfect square -/
def a : ℕ := 2

/-- The smallest positive integer b such that 450 * b is a perfect cube -/
def b : ℕ := 60

/-- 450 * a is a perfect square -/
axiom h1 : ∃ n : ℕ, 450 * a = n^2

/-- 450 * b is a perfect cube -/
axiom h2 : ∃ n : ℕ, 450 * b = n^3

/-- a is the smallest positive integer satisfying the square condition -/
axiom h3 : ∀ x : ℕ, 0 < x → x < a → ¬∃ n : ℕ, 450 * x = n^2

/-- b is the smallest positive integer satisfying the cube condition -/
axiom h4 : ∀ x : ℕ, 0 < x → x < b → ¬∃ n : ℕ, 450 * x = n^3

theorem sum_of_a_and_b : a + b = 62 := by
  sorry

end sum_of_a_and_b_l3805_380573


namespace no_three_digit_divisible_by_15_ending_in_7_l3805_380566

theorem no_three_digit_divisible_by_15_ending_in_7 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 7 → ¬(n % 15 = 0) :=
by sorry

end no_three_digit_divisible_by_15_ending_in_7_l3805_380566


namespace magic_deck_cost_l3805_380551

/-- Calculates the cost per deck given the initial number of decks, remaining decks, and total earnings -/
def cost_per_deck (initial_decks : ℕ) (remaining_decks : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (initial_decks - remaining_decks)

/-- Proves that the cost per deck is 7 dollars given the problem conditions -/
theorem magic_deck_cost :
  let initial_decks : ℕ := 16
  let remaining_decks : ℕ := 8
  let total_earnings : ℕ := 56
  cost_per_deck initial_decks remaining_decks total_earnings = 7 := by
  sorry


end magic_deck_cost_l3805_380551


namespace card_drawing_probabilities_l3805_380508

def num_cards : ℕ := 5
def num_odd_cards : ℕ := 3
def num_even_cards : ℕ := 2

def prob_not_both_odd_or_even : ℚ := 3 / 5

def prob_two_even_in_three_draws : ℚ := 36 / 125

theorem card_drawing_probabilities :
  (prob_not_both_odd_or_even = (num_odd_cards * num_even_cards : ℚ) / (num_cards.choose 2)) ∧
  (prob_two_even_in_three_draws = 3 * (num_even_cards / num_cards)^2 * (1 - num_even_cards / num_cards)) :=
by sorry

end card_drawing_probabilities_l3805_380508


namespace system_one_solution_system_two_solution_l3805_380509

-- System (1)
theorem system_one_solution (x y : ℚ) :
  (3 * x + 4 * y = 16) ∧ (5 * x - 8 * y = 34) → x = 6 ∧ y = -1/2 := by sorry

-- System (2)
theorem system_two_solution (x y : ℚ) :
  ((x - 1) / 2 + (y + 1) / 3 = 1) ∧ (x + y = 4) → x = -1 ∧ y = 5 := by sorry

end system_one_solution_system_two_solution_l3805_380509


namespace solve_equation_l3805_380575

theorem solve_equation : ∃ x : ℝ, (5*x + 9*x = 350 - 10*(x - 4)) ∧ x = 16.25 := by
  sorry

end solve_equation_l3805_380575


namespace office_salary_problem_l3805_380562

/-- Represents the average salary of non-officers in Rs/month -/
def average_salary_non_officers : ℝ := 110

theorem office_salary_problem (total_employees : ℕ) (officers : ℕ) (non_officers : ℕ)
  (avg_salary_all : ℝ) (avg_salary_officers : ℝ) :
  total_employees = officers + non_officers →
  total_employees = 495 →
  officers = 15 →
  non_officers = 480 →
  avg_salary_all = 120 →
  avg_salary_officers = 440 →
  average_salary_non_officers = 
    (total_employees * avg_salary_all - officers * avg_salary_officers) / non_officers :=
by sorry

end office_salary_problem_l3805_380562


namespace remaining_amount_is_10_95_l3805_380588

def initial_amount : ℝ := 60

def frame_price : ℝ := 15
def frame_discount : ℝ := 0.1

def wheel_price : ℝ := 25
def wheel_discount : ℝ := 0.05

def seat_price : ℝ := 8
def seat_discount : ℝ := 0.15

def tape_price : ℝ := 5
def tape_discount : ℝ := 0

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_cost : ℝ :=
  discounted_price frame_price frame_discount +
  discounted_price wheel_price wheel_discount +
  discounted_price seat_price seat_discount +
  discounted_price tape_price tape_discount

theorem remaining_amount_is_10_95 :
  initial_amount - total_cost = 10.95 :=
by sorry

end remaining_amount_is_10_95_l3805_380588


namespace belinda_age_difference_l3805_380570

/-- Given the ages of Tony and Belinda, prove that Belinda's age is 8 years more than twice Tony's age. -/
theorem belinda_age_difference (tony_age belinda_age : ℕ) : 
  tony_age = 16 →
  belinda_age = 40 →
  tony_age + belinda_age = 56 →
  belinda_age > 2 * tony_age →
  belinda_age - 2 * tony_age = 8 := by
sorry

end belinda_age_difference_l3805_380570


namespace unique_factorial_sum_l3805_380590

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Sum of factorials of digits -/
def sum_factorial_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  factorial hundreds + factorial tens + factorial ones

/-- Theorem stating that 145 is the only three-digit number equal to the sum of factorials of its digits -/
theorem unique_factorial_sum :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 → (n = sum_factorial_digits n ↔ n = 145) := by
  sorry

#eval sum_factorial_digits 145  -- Should output 145

end unique_factorial_sum_l3805_380590


namespace closest_sum_to_zero_l3805_380516

def S : Finset Int := {5, 19, -6, 0, -4}

theorem closest_sum_to_zero (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∀ x y z, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
  |a + b + c| ≤ |x + y + z| ∧ (∃ p q r, p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ |p + q + r| = 1) :=
by sorry

end closest_sum_to_zero_l3805_380516


namespace gregory_age_l3805_380564

/-- Represents the ages of Dmitry and Gregory at different points in time -/
structure Ages where
  gregory_past : ℕ
  dmitry_past : ℕ
  gregory_current : ℕ
  dmitry_current : ℕ
  gregory_future : ℕ
  dmitry_future : ℕ

/-- The conditions of the problem -/
def age_conditions (a : Ages) : Prop :=
  a.dmitry_current = 3 * a.gregory_past ∧
  a.gregory_current = a.dmitry_past ∧
  a.gregory_future = a.dmitry_current ∧
  a.gregory_future + a.dmitry_future = 49 ∧
  a.dmitry_future - a.gregory_future = a.dmitry_current - a.gregory_current

theorem gregory_age (a : Ages) : age_conditions a → a.gregory_current = 14 := by
  sorry

#check gregory_age

end gregory_age_l3805_380564


namespace can_transport_goods_l3805_380555

/-- Represents the total weight of goods in tonnes -/
def total_weight : ℝ := 13.5

/-- Represents the maximum weight of goods in a single box in tonnes -/
def max_box_weight : ℝ := 0.35

/-- Represents the number of available trucks -/
def num_trucks : ℕ := 11

/-- Represents the load capacity of each truck in tonnes -/
def truck_capacity : ℝ := 1.5

/-- Theorem stating that the given number of trucks can transport all goods in a single trip -/
theorem can_transport_goods : 
  (num_trucks : ℝ) * truck_capacity ≥ total_weight := by sorry

end can_transport_goods_l3805_380555


namespace min_value_xyz_l3805_380527

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 18) :
  x + 3 * y + 6 * z ≥ 3 * (2 * Real.sqrt 6 + 1) :=
sorry

end min_value_xyz_l3805_380527


namespace rectangle_ratio_theorem_l3805_380586

theorem rectangle_ratio_theorem (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∃ (k l m n : ℕ), k * a + l * b = a * Real.sqrt 30 ∧
                     m * a + n * b = b * Real.sqrt 30 ∧
                     k * n = l * m ∧ l * m = 30) →
  (a / b = Real.sqrt 30 ∨
   a / b = Real.sqrt 30 / 2 ∨
   a / b = Real.sqrt 30 / 3 ∨
   a / b = Real.sqrt 30 / 5 ∨
   a / b = Real.sqrt 30 / 6 ∨
   a / b = Real.sqrt 30 / 10 ∨
   a / b = Real.sqrt 30 / 15 ∨
   a / b = Real.sqrt 30 / 30) :=
by sorry

end rectangle_ratio_theorem_l3805_380586


namespace tank_capacity_l3805_380568

/-- Proves that the capacity of a tank filled by two buckets of 4 and 3 liters,
    where the 3-liter bucket is used 4 times more, is 48 liters. -/
theorem tank_capacity (x : ℕ) : 
  (4 * x = 3 * (x + 4)) → (4 * x = 48) :=
by sorry

end tank_capacity_l3805_380568


namespace geometric_sequence_equality_l3805_380552

theorem geometric_sequence_equality (a b c d : ℝ) :
  (a / b = c / d) ↔ (a * d = b * c) :=
by sorry

end geometric_sequence_equality_l3805_380552


namespace meaningful_range_l3805_380599

def is_meaningful (x : ℝ) : Prop :=
  1 - x ≥ 0 ∧ 2 + x ≠ 0

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≤ 1 ∧ x ≠ -2 := by
sorry

end meaningful_range_l3805_380599


namespace jump_rope_results_l3805_380578

def passing_score : ℕ := 140

def scores : List ℤ := [-25, 17, 23, 0, -39, -11, 9, 34]

def score_difference (scores : List ℤ) : ℕ :=
  (scores.maximum?.getD 0 - scores.minimum?.getD 0).toNat

def average_score (scores : List ℤ) : ℚ :=
  passing_score + (scores.sum : ℚ) / scores.length

def calculate_points (score : ℤ) : ℤ :=
  if score > 0 then 2 * score else -score

def total_score (scores : List ℤ) : ℤ :=
  scores.map calculate_points |>.sum

theorem jump_rope_results :
  score_difference scores = 73 ∧
  average_score scores = 141 ∧
  total_score scores = 91 := by
  sorry

end jump_rope_results_l3805_380578


namespace max_students_distribution_l3805_380580

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 640) (h2 : pencils = 520) :
  (∃ (students : ℕ), students > 0 ∧ pens % students = 0 ∧ pencils % students = 0) ∧
  (∀ (n : ℕ), n > 0 ∧ pens % n = 0 ∧ pencils % n = 0 → n ≤ 40) ∧
  (pens % 40 = 0 ∧ pencils % 40 = 0) :=
by sorry

end max_students_distribution_l3805_380580


namespace simplify_expression_l3805_380546

theorem simplify_expression (x : ℝ) (h : x^2 ≥ 49) :
  (7 - Real.sqrt (x^2 - 49))^2 = x^2 - 14 * Real.sqrt (x^2 - 49) := by
  sorry

end simplify_expression_l3805_380546


namespace calculation_1_calculation_2_calculation_3_calculation_4_calculation_5_calculation_6_l3805_380594

theorem calculation_1 : 320 + 16 * 27 = 752 := by sorry

theorem calculation_2 : 1500 - 125 * 8 = 500 := by sorry

theorem calculation_3 : 22 * 22 - 84 = 400 := by sorry

theorem calculation_4 : 25 * 8 * 9 = 1800 := by sorry

theorem calculation_5 : (25 + 38) * 15 = 945 := by sorry

theorem calculation_6 : (62 + 12) * 38 = 2812 := by sorry

end calculation_1_calculation_2_calculation_3_calculation_4_calculation_5_calculation_6_l3805_380594


namespace employed_female_parttime_ratio_is_60_percent_l3805_380519

/-- Represents the population statistics of Town P -/
structure TownP where
  total_population : ℝ
  employed_percentage : ℝ
  employed_female_percentage : ℝ
  employed_female_parttime_percentage : ℝ
  employed_male_percentage : ℝ

/-- Calculates the percentage of employed females who are part-time workers in Town P -/
def employed_female_parttime_ratio (town : TownP) : ℝ :=
  town.employed_female_parttime_percentage

/-- Theorem stating that 60% of employed females in Town P are part-time workers -/
theorem employed_female_parttime_ratio_is_60_percent (town : TownP) 
  (h1 : town.employed_percentage = 0.6)
  (h2 : town.employed_female_percentage = 0.4)
  (h3 : town.employed_female_parttime_percentage = 0.6)
  (h4 : town.employed_male_percentage = 0.48) :
  employed_female_parttime_ratio town = 0.6 := by
  sorry

#check employed_female_parttime_ratio_is_60_percent

end employed_female_parttime_ratio_is_60_percent_l3805_380519


namespace sin_beta_value_l3805_380532

theorem sin_beta_value (α β : Real) (h_acute : 0 < α ∧ α < π / 2)
  (h1 : 2 * Real.tan (π - α) - 3 * Real.cos (π / 2 + β) + 5 = 0)
  (h2 : Real.tan (π + α) + 6 * Real.sin (π + β) = 1) :
  Real.sin β = 1 / 3 := by
  sorry

end sin_beta_value_l3805_380532


namespace square_root_reverses_squaring_l3805_380581

theorem square_root_reverses_squaring (x : ℝ) (hx : x = 25) : 
  Real.sqrt (x ^ 2) = x := by sorry

end square_root_reverses_squaring_l3805_380581


namespace missing_number_is_33_l3805_380530

def known_numbers : List ℝ := [1, 22, 24, 25, 26, 27, 2]

theorem missing_number_is_33 :
  ∃ x : ℝ, (known_numbers.sum + x) / 8 = 20 ∧ x = 33 := by
  sorry

end missing_number_is_33_l3805_380530


namespace perfume_price_problem_l3805_380549

/-- Proves that given the conditions of the perfume price changes, the original price must be $1200 -/
theorem perfume_price_problem (P : ℝ) : 
  (P * 1.10 * 0.85 = P - 78) → P = 1200 := by
  sorry

end perfume_price_problem_l3805_380549


namespace total_triangles_is_seventeen_l3805_380547

/-- Represents a 2x2 square grid where each square is divided diagonally into two right-angled triangles -/
structure DiagonallyDividedGrid :=
  (size : Nat)
  (is_two_by_two : size = 2)
  (diagonally_divided : Bool)

/-- Counts the total number of triangles in the grid, including all possible combinations -/
def count_triangles (grid : DiagonallyDividedGrid) : Nat :=
  sorry

/-- Theorem stating that the total number of triangles in the described grid is 17 -/
theorem total_triangles_is_seventeen (grid : DiagonallyDividedGrid) 
  (h1 : grid.diagonally_divided = true) : 
  count_triangles grid = 17 := by
  sorry

end total_triangles_is_seventeen_l3805_380547


namespace perimeter_of_specific_rectangle_l3805_380544

/-- A figure that can be completed to form a rectangle -/
structure CompletableRectangle where
  length : ℝ
  width : ℝ

/-- The perimeter of a CompletableRectangle -/
def perimeter (r : CompletableRectangle) : ℝ :=
  2 * (r.length + r.width)

theorem perimeter_of_specific_rectangle :
  ∃ (r : CompletableRectangle), r.length = 6 ∧ r.width = 5 ∧ perimeter r = 22 :=
by sorry

end perimeter_of_specific_rectangle_l3805_380544


namespace work_equivalence_first_group_size_correct_l3805_380506

/-- The number of hours it takes the first group to complete the work -/
def first_group_hours : ℕ := 20

/-- The number of men in the second group -/
def second_group_men : ℕ := 15

/-- The number of hours it takes the second group to complete the work -/
def second_group_hours : ℕ := 48

/-- The number of men in the first group -/
def first_group_men : ℕ := 36

theorem work_equivalence :
  first_group_men * first_group_hours = second_group_men * second_group_hours :=
by sorry

/-- Proves that the number of men in the first group is correct -/
theorem first_group_size_correct :
  first_group_men = (second_group_men * second_group_hours) / first_group_hours :=
by sorry

end work_equivalence_first_group_size_correct_l3805_380506


namespace parallelogram_area_minimum_l3805_380576

theorem parallelogram_area_minimum (z : ℂ) : 
  (∃ (area : ℝ), area = (36:ℝ)/(37:ℝ) ∧ 
    area = 2 * Complex.abs (z * Complex.I * (1/z - z))) →
  (Complex.re z > 0) →
  (Complex.im z < 0) →
  (∃ (d : ℝ), d = Complex.abs (z + 1/z) ∧
    ∀ (w : ℂ), (Complex.re w > 0) → (Complex.im w < 0) →
    (∃ (area : ℝ), area = (36:ℝ)/(37:ℝ) ∧ 
      area = 2 * Complex.abs (w * Complex.I * (1/w - w))) →
    d ≤ Complex.abs (w + 1/w)) →
  (Complex.abs (z + 1/z))^2 = (12:ℝ)/(37:ℝ) :=
by sorry

end parallelogram_area_minimum_l3805_380576


namespace muffin_banana_price_ratio_l3805_380560

/-- The price ratio of a muffin to a banana -/
def price_ratio (muffin_price banana_price : ℚ) : ℚ :=
  muffin_price / banana_price

/-- Susie's total cost for 4 muffins and 5 bananas -/
def susie_cost (muffin_price banana_price : ℚ) : ℚ :=
  4 * muffin_price + 5 * banana_price

/-- Calvin's total cost for 2 muffins and 12 bananas -/
def calvin_cost (muffin_price banana_price : ℚ) : ℚ :=
  2 * muffin_price + 12 * banana_price

theorem muffin_banana_price_ratio :
  ∀ (muffin_price banana_price : ℚ),
    muffin_price > 0 →
    banana_price > 0 →
    calvin_cost muffin_price banana_price = 3 * susie_cost muffin_price banana_price →
    price_ratio muffin_price banana_price = 3 / 10 := by
  sorry


end muffin_banana_price_ratio_l3805_380560


namespace min_distance_circle_point_l3805_380583

noncomputable section

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 + 1)^2 = 4}

-- Define point Q
def point_Q : ℝ × ℝ := (Real.sqrt 2, -Real.sqrt 2)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_circle_point :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 ∧
  ∀ (P : ℝ × ℝ), P ∈ circle_C →
  distance P point_Q ≥ min_dist :=
sorry

end

end min_distance_circle_point_l3805_380583


namespace even_decreasing_nonpos_ordering_l3805_380514

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def decreasing_on_nonpos (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 0 → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem even_decreasing_nonpos_ordering (f : ℝ → ℝ) 
  (h_even : is_even f) (h_decr : decreasing_on_nonpos f) : 
  f 1 < f (-2) ∧ f (-2) < f (-3) := by
  sorry

end even_decreasing_nonpos_ordering_l3805_380514


namespace restaurant_service_charge_l3805_380545

theorem restaurant_service_charge (total_paid : ℝ) (service_charge_rate : ℝ) 
  (h1 : service_charge_rate = 0.04)
  (h2 : total_paid = 468) :
  ∃ (original_amount : ℝ), 
    original_amount * (1 + service_charge_rate) = total_paid ∧ 
    original_amount = 450 := by
  sorry

end restaurant_service_charge_l3805_380545


namespace initial_amount_at_racetrack_l3805_380558

/-- Represents the sequence of bets and their outcomes at the racetrack --/
def racetrack_bets (initial_amount : ℝ) : ℝ :=
  let after_first := initial_amount * 2
  let after_second := after_first - 60
  let after_third := after_second * 2
  let after_fourth := after_third - 60
  let after_fifth := after_fourth * 2
  after_fifth - 60

/-- Theorem stating that the initial amount at the racetrack was 52.5 francs --/
theorem initial_amount_at_racetrack : 
  ∃ (x : ℝ), x > 0 ∧ racetrack_bets x = 0 ∧ x = 52.5 :=
sorry

end initial_amount_at_racetrack_l3805_380558


namespace ninth_minus_eighth_square_tiles_l3805_380550

/-- The side length of the nth square in the sequence -/
def L (n : ℕ) : ℕ := 2 * n + 1

/-- The number of tiles in the nth square -/
def tiles (n : ℕ) : ℕ := (L n) ^ 2

theorem ninth_minus_eighth_square_tiles : tiles 9 - tiles 8 = 72 := by
  sorry

end ninth_minus_eighth_square_tiles_l3805_380550


namespace pencil_buyers_difference_l3805_380596

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The total amount spent by eighth graders in cents -/
def eighth_grade_total : ℕ := 162

/-- The total amount spent by fifth graders in cents -/
def fifth_grade_total : ℕ := 216

/-- The cost of each pencil in cents -/
def pencil_cost : ℕ := 18

theorem pencil_buyers_difference : 
  ∃ (eighth_buyers fifth_buyers : ℕ),
    eighth_grade_total = eighth_buyers * pencil_cost ∧
    fifth_grade_total = fifth_buyers * pencil_cost ∧
    fifth_buyers - eighth_buyers = 3 :=
by sorry

end pencil_buyers_difference_l3805_380596


namespace fourth_derivative_y_l3805_380522

noncomputable def y (x : ℝ) : ℝ := (x^2 + 3) * Real.log (x - 3)

theorem fourth_derivative_y (x : ℝ) (h : x ≠ 3) :
  (deriv^[4] y) x = (-2 * x^2 + 24 * x - 126) / (x - 3)^4 := by
  sorry

end fourth_derivative_y_l3805_380522


namespace chicken_cage_problem_l3805_380537

theorem chicken_cage_problem :
  ∃ (chickens cages : ℕ),
    chickens = 25 ∧ cages = 6 ∧
    (4 * cages + 1 = chickens) ∧
    (5 * (cages - 1) = chickens) :=
by sorry

end chicken_cage_problem_l3805_380537


namespace sqrt_21_is_11th_term_l3805_380584

theorem sqrt_21_is_11th_term (a : ℕ → ℝ) :
  (∀ n, a n = Real.sqrt (2 * n - 1)) →
  a 11 = Real.sqrt 21 :=
by
  sorry

end sqrt_21_is_11th_term_l3805_380584


namespace orange_face_probability_l3805_380598

/-- Represents a die with a specific number of sides and orange faces. -/
structure Die where
  totalSides : ℕ
  orangeFaces : ℕ
  orangeFaces_le_totalSides : orangeFaces ≤ totalSides

/-- Calculates the probability of rolling an orange face on a given die. -/
def probabilityOrangeFace (d : Die) : ℚ :=
  d.orangeFaces / d.totalSides

/-- The specific 10-sided die with 4 orange faces. -/
def tenSidedDie : Die where
  totalSides := 10
  orangeFaces := 4
  orangeFaces_le_totalSides := by norm_num

/-- Theorem stating that the probability of rolling an orange face on the 10-sided die is 2/5. -/
theorem orange_face_probability :
  probabilityOrangeFace tenSidedDie = 2 / 5 := by
  sorry

end orange_face_probability_l3805_380598


namespace exists_valid_divided_rectangle_area_of_valid_divided_rectangle_l3805_380567

/-- Represents a rectangle divided into four smaller rectangles --/
structure DividedRectangle where
  a : ℝ  -- vertical side of the original rectangle
  b : ℝ  -- horizontal side of the original rectangle
  x : ℝ  -- side length of the square

/-- Conditions for the divided rectangle --/
def validDividedRectangle (r : DividedRectangle) : Prop :=
  r.a > 0 ∧ r.b > 0 ∧ r.x > 0 ∧
  r.a + r.x = 10 ∧  -- perimeter of adjacent rectangle is 20
  r.b + r.x = 8     -- perimeter of adjacent rectangle is 16

/-- Theorem stating the existence of a valid divided rectangle --/
theorem exists_valid_divided_rectangle :
  ∃ (r : DividedRectangle), validDividedRectangle r :=
sorry

/-- Function to calculate the area of the original rectangle --/
def area (r : DividedRectangle) : ℝ := r.a * r.b

/-- Theorem to find the area of the valid divided rectangle --/
theorem area_of_valid_divided_rectangle :
  ∃ (r : DividedRectangle), validDividedRectangle r ∧ 
  ∃ (A : ℝ), area r = A :=
sorry

end exists_valid_divided_rectangle_area_of_valid_divided_rectangle_l3805_380567


namespace mike_total_score_l3805_380591

/-- Given that Mike played six games of basketball and scored four points in each game,
    prove that his total score is 24 points. -/
theorem mike_total_score :
  let games_played : ℕ := 6
  let points_per_game : ℕ := 4
  let total_score := games_played * points_per_game
  total_score = 24 := by sorry

end mike_total_score_l3805_380591


namespace arithmetic_geometric_progression_existence_l3805_380593

theorem arithmetic_geometric_progression_existence :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ (d : ℝ), b = a + d ∧ c = b + d) ∧
  (∃ (r : ℝ), (a = b ∧ b = c * r) ∨ (a = b * r ∧ b = c) ∨ (a = c ∧ c = b * r)) :=
by sorry

end arithmetic_geometric_progression_existence_l3805_380593


namespace expansion_properties_l3805_380540

theorem expansion_properties (n : ℕ) : 
  (∃ a b : ℚ, 
    (1 : ℚ) = a ∧ 
    (n : ℚ) * (1 / 2 : ℚ) = a + b ∧ 
    (n * (n - 1) / 2 : ℚ) * (1 / 4 : ℚ) = a + 2 * b) → 
  n = 8 ∧ (2 : ℕ) ^ n = 256 :=
by sorry

end expansion_properties_l3805_380540


namespace tree_height_from_shadows_l3805_380512

/-- Given a person and a tree casting shadows, calculates the height of the tree -/
theorem tree_height_from_shadows 
  (h s S : ℝ) 
  (h_pos : h > 0) 
  (s_pos : s > 0) 
  (S_pos : S > 0) 
  (h_val : h = 1.5) 
  (s_val : s = 0.5) 
  (S_val : S = 10) : 
  h / s * S = 30 := by
sorry

end tree_height_from_shadows_l3805_380512


namespace max_cube_sum_on_sphere_l3805_380557

theorem max_cube_sum_on_sphere (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) :
  x^3 + y^3 + z^3 ≤ 27 ∧ ∃ x y z : ℝ, x^2 + y^2 + z^2 = 9 ∧ x^3 + y^3 + z^3 = 27 := by
  sorry

end max_cube_sum_on_sphere_l3805_380557


namespace greatest_common_remainder_l3805_380565

theorem greatest_common_remainder (a b c d : ℕ) (h1 : a % 2 = 0 ∧ a % 3 = 0 ∧ a % 5 = 0 ∧ a % 7 = 0 ∧ a % 11 = 0)
                                               (h2 : b % 2 = 0 ∧ b % 3 = 0 ∧ b % 5 = 0 ∧ b % 7 = 0 ∧ b % 11 = 0)
                                               (h3 : c % 2 = 0 ∧ c % 3 = 0 ∧ c % 5 = 0 ∧ c % 7 = 0 ∧ c % 11 = 0)
                                               (h4 : d % 2 = 0 ∧ d % 3 = 0 ∧ d % 5 = 0 ∧ d % 7 = 0 ∧ d % 11 = 0)
                                               (ha : a = 1260) (hb : b = 2310) (hc : c = 30030) (hd : d = 72930) :
  ∃! k : ℕ, k > 0 ∧ k ≤ 30 ∧ 
  ∃ r : ℕ, a % k = r ∧ b % k = r ∧ c % k = r ∧ d % k = r ∧
  ∀ m : ℕ, m > k → ¬(∃ s : ℕ, a % m = s ∧ b % m = s ∧ c % m = s ∧ d % m = s) :=
by sorry

end greatest_common_remainder_l3805_380565


namespace function_inequality_and_sum_product_l3805_380503

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 2|

-- State the theorem
theorem function_inequality_and_sum_product (m M a b : ℝ) :
  (∀ x, f x ≥ |m - 1|) →
  (-2 ≤ m ∧ m ≤ 4) ∧
  (M = 4 →
   a > 0 →
   b > 0 →
   a^2 + b^2 = M/2 →
   a + b ≥ 2*a*b) :=
by sorry

end function_inequality_and_sum_product_l3805_380503


namespace sara_marble_count_l3805_380523

/-- The number of black marbles Sara has after receiving a gift from Fred -/
def saras_marbles (initial : Float) (gift : Float) : Float :=
  initial + gift

/-- Theorem stating that Sara has 1025.0 black marbles after receiving Fred's gift -/
theorem sara_marble_count : saras_marbles 792.0 233.0 = 1025.0 := by
  sorry

end sara_marble_count_l3805_380523


namespace sachin_age_l3805_380531

/-- Proves that Sachin's age is 49 given the conditions -/
theorem sachin_age :
  ∀ (s r : ℕ),
  r = s + 14 →
  s * 9 = r * 7 →
  s = 49 :=
by
  sorry

end sachin_age_l3805_380531


namespace sin_cos_pi_12_star_l3805_380529

-- Define the * operation
def star (a b : ℝ) : ℝ := a^2 - a*b - b^2

-- State the theorem
theorem sin_cos_pi_12_star :
  star (Real.sin (π/12)) (Real.cos (π/12)) = -(1 + 2*Real.sqrt 3)/4 := by
  sorry

end sin_cos_pi_12_star_l3805_380529


namespace sara_marbles_l3805_380582

/-- Given that Sara has 10 marbles initially and loses 7 marbles, prove that she will have 3 marbles left. -/
theorem sara_marbles (initial_marbles : ℕ) (lost_marbles : ℕ) (h1 : initial_marbles = 10) (h2 : lost_marbles = 7) :
  initial_marbles - lost_marbles = 3 := by
  sorry

end sara_marbles_l3805_380582


namespace vanessa_birthday_money_l3805_380543

theorem vanessa_birthday_money (money : ℕ) : 
  (∃ k : ℕ, money = 9 * k + 1) ↔ 
  (∃ n : ℕ, money = 9 * n + 1 ∧ 
    ∀ m : ℕ, m < n → money ≥ 9 * m + 1) :=
by sorry

end vanessa_birthday_money_l3805_380543


namespace prob_no_adjacent_three_of_ten_l3805_380521

/-- The number of chairs in a row -/
def n : ℕ := 10

/-- The number of people choosing seats -/
def k : ℕ := 3

/-- The probability of k people choosing seats from n chairs such that none sit next to each other -/
def prob_no_adjacent (n k : ℕ) : ℚ :=
  sorry

/-- Theorem stating that the probability of 3 people choosing seats from 10 chairs 
    such that none sit next to each other is 1/3 -/
theorem prob_no_adjacent_three_of_ten : prob_no_adjacent n k = 1/3 := by
  sorry

end prob_no_adjacent_three_of_ten_l3805_380521


namespace largest_expression_l3805_380528

theorem largest_expression :
  let a := 3 + 1 + 2 + 8
  let b := 3 * 1 + 2 + 8
  let c := 3 + 1 * 2 + 8
  let d := 3 + 1 + 2 * 8
  let e := 3 * 1 * 2 * 8
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) := by
  sorry

end largest_expression_l3805_380528


namespace marbles_sharing_l3805_380554

theorem marbles_sharing (sienna_initial jordan_initial : ℕ)
  (h1 : sienna_initial = 150)
  (h2 : jordan_initial = 90)
  (shared : ℕ)
  (h3 : sienna_initial - shared = 3 * (jordan_initial + shared)) :
  shared = 30 := by
  sorry

end marbles_sharing_l3805_380554


namespace simplify_expression_l3805_380524

theorem simplify_expression (a b : ℝ) : a + (3*a - 3*b) - (a - 2*b) = 3*a - b := by
  sorry

end simplify_expression_l3805_380524


namespace greene_nursery_roses_l3805_380597

/-- The Greene Nursery flower counting problem -/
theorem greene_nursery_roses (total_flowers yellow_carnations white_roses : ℕ) 
  (h1 : total_flowers = 6284)
  (h2 : yellow_carnations = 3025)
  (h3 : white_roses = 1768) :
  total_flowers - yellow_carnations - white_roses = 1491 := by
  sorry

end greene_nursery_roses_l3805_380597


namespace no_special_numbers_l3805_380538

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_special_numbers :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ n % 3 = 0 ∧ Even n :=
by sorry

end no_special_numbers_l3805_380538


namespace distance_on_number_line_l3805_380520

theorem distance_on_number_line : 
  let point_a : ℝ := 3
  let point_b : ℝ := -2
  |point_a - point_b| = 5 := by sorry

end distance_on_number_line_l3805_380520


namespace lipstick_ratio_l3805_380515

/-- Proves that the ratio of students wearing blue lipstick to those wearing red lipstick is 1:5 -/
theorem lipstick_ratio (total_students : ℕ) (blue_lipstick : ℕ) 
  (h1 : total_students = 200)
  (h2 : blue_lipstick = 5)
  (h3 : 2 * (total_students / 2) = total_students)  -- Half of students wore lipstick
  (h4 : 4 * (total_students / 2 / 4) = total_students / 2)  -- Quarter of lipstick wearers wore red
  (h5 : blue_lipstick = total_students / 2 / 4)  -- Same number wore blue as red
  : (blue_lipstick : ℚ) / (total_students / 2 / 4 : ℚ) = 1 / 5 := by
  sorry


end lipstick_ratio_l3805_380515


namespace result_circle_properties_l3805_380592

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

/-- The equation of the resulting circle -/
def resultCircle (x y : ℝ) : Prop := 9*x^2 + 9*y^2 - 14*x + 4*y = 0

/-- Theorem stating that the resulting circle passes through the intersection points of the given circles and the point (1, -1) -/
theorem result_circle_properties :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → resultCircle x y) ∧
  resultCircle 1 (-1) := by
  sorry

end result_circle_properties_l3805_380592


namespace four_digit_integer_problem_l3805_380579

theorem four_digit_integer_problem (n : ℕ) (a b c d : ℕ) :
  n = a * 1000 + b * 100 + c * 10 + d →
  a ≥ 1 →
  a ≤ 9 →
  b ≤ 9 →
  c ≤ 9 →
  d ≤ 9 →
  a + b + c + d = 17 →
  b + c = 10 →
  a - d = 3 →
  n % 13 = 0 →
  n = 5732 := by sorry

end four_digit_integer_problem_l3805_380579


namespace tuesday_texts_l3805_380513

/-- The number of texts sent by Sydney to each person on Monday -/
def monday_texts_per_person : ℕ := 5

/-- The total number of texts sent by Sydney over both days -/
def total_texts : ℕ := 40

/-- The number of people Sydney sent texts to -/
def num_people : ℕ := 2

/-- Theorem: The number of texts sent on Tuesday is 30 -/
theorem tuesday_texts :
  total_texts - (monday_texts_per_person * num_people) = 30 := by
  sorry

end tuesday_texts_l3805_380513


namespace pie_weight_l3805_380536

theorem pie_weight (total_weight : ℝ) (eaten_fraction : ℝ) (eaten_weight : ℝ) : 
  eaten_fraction = 1/6 →
  eaten_weight = 240 →
  total_weight = eaten_weight / eaten_fraction →
  (1 - eaten_fraction) * total_weight = 1200 :=
by
  sorry

end pie_weight_l3805_380536


namespace bingo_prize_distribution_l3805_380556

theorem bingo_prize_distribution (total_prize : ℚ) (first_winner_fraction : ℚ) 
  (num_subsequent_winners : ℕ) (each_subsequent_winner_prize : ℚ) :
  total_prize = 2400 →
  first_winner_fraction = 1/3 →
  num_subsequent_winners = 10 →
  each_subsequent_winner_prize = 160 →
  let remaining_prize := total_prize - first_winner_fraction * total_prize
  (each_subsequent_winner_prize / remaining_prize) = 1/10 := by
  sorry

end bingo_prize_distribution_l3805_380556


namespace optimal_plan_is_most_cost_effective_l3805_380569

/-- Represents a sewage treatment equipment purchase plan -/
structure PurchasePlan where
  modelA : ℕ
  modelB : ℕ

/-- Checks if a purchase plan is valid according to the given conditions -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.modelA + p.modelB = 10 ∧
  12 * p.modelA + 10 * p.modelB ≤ 105 ∧
  240 * p.modelA + 200 * p.modelB ≥ 2040

/-- Calculates the total cost of a purchase plan -/
def totalCost (p : PurchasePlan) : ℕ :=
  12 * p.modelA + 10 * p.modelB

/-- The optimal purchase plan -/
def optimalPlan : PurchasePlan :=
  { modelA := 1, modelB := 9 }

/-- Theorem stating that the optimal plan is the most cost-effective valid plan -/
theorem optimal_plan_is_most_cost_effective :
  isValidPlan optimalPlan ∧
  ∀ p : PurchasePlan, isValidPlan p → totalCost optimalPlan ≤ totalCost p :=
sorry

end optimal_plan_is_most_cost_effective_l3805_380569


namespace probability_of_exact_tails_l3805_380504

noncomputable def probability_of_tails : ℚ := 2/3
noncomputable def number_of_flips : ℕ := 10
noncomputable def number_of_tails : ℕ := 4

theorem probability_of_exact_tails :
  (Nat.choose number_of_flips number_of_tails) *
  (probability_of_tails ^ number_of_tails) *
  ((1 - probability_of_tails) ^ (number_of_flips - number_of_tails)) =
  3360/6561 :=
by sorry

end probability_of_exact_tails_l3805_380504


namespace cos_2alpha_eq_neg_four_fifths_l3805_380589

theorem cos_2alpha_eq_neg_four_fifths (α : Real) 
  (h : (Real.tan α + 1) / (Real.tan α - 1) = 2) : 
  Real.cos (2 * α) = -4/5 := by
  sorry

end cos_2alpha_eq_neg_four_fifths_l3805_380589


namespace planet_combinations_correct_l3805_380541

/-- The number of different combinations of planets that can be occupied. -/
def planetCombinations : ℕ :=
  let earthLike := 7
  let marsLike := 8
  let earthUnits := 3
  let marsUnits := 1
  let totalUnits := 21
  2941

/-- Theorem stating that the number of planet combinations is correct. -/
theorem planet_combinations_correct : planetCombinations = 2941 := by
  sorry

end planet_combinations_correct_l3805_380541


namespace sunzi_suanjing_congruence_l3805_380574

theorem sunzi_suanjing_congruence : ∃ (m : ℕ+), (3 ^ 20 : ℤ) ≡ 2013 [ZMOD m] := by
  sorry

end sunzi_suanjing_congruence_l3805_380574


namespace james_vehicle_count_l3805_380535

theorem james_vehicle_count :
  let trucks : ℕ := 12
  let buses : ℕ := 2
  let taxis : ℕ := 4
  let cars : ℕ := 30
  let truck_capacity : ℕ := 2
  let bus_capacity : ℕ := 15
  let taxi_capacity : ℕ := 2
  let motorbike_capacity : ℕ := 1
  let car_capacity : ℕ := 3
  let total_passengers : ℕ := 156
  let motorbikes : ℕ := total_passengers - 
    (trucks * truck_capacity + buses * bus_capacity + 
     taxis * taxi_capacity + cars * car_capacity)
  trucks + buses + taxis + motorbikes + cars = 52 := by
sorry

end james_vehicle_count_l3805_380535


namespace unique_two_digit_integer_l3805_380510

theorem unique_two_digit_integer (t : ℕ) : 
  (10 ≤ t ∧ t < 100) ∧ (13 * t) % 100 = 52 ↔ t = 4 := by
  sorry

end unique_two_digit_integer_l3805_380510


namespace probability_of_red_ball_in_bag_A_l3805_380572

theorem probability_of_red_ball_in_bag_A 
  (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) :
  let total_A := m + n
  let total_B := 7
  let prob_red_A := m / total_A
  let prob_white_A := n / total_A
  let prob_red_B_after_red := (3 + 1) / (total_B + 1)
  let prob_red_B_after_white := 3 / (total_B + 1)
  let total_prob_red := prob_red_A * prob_red_B_after_red + prob_white_A * prob_red_B_after_white
  total_prob_red = 15/32 → prob_red_A = 3/4 := by
sorry

end probability_of_red_ball_in_bag_A_l3805_380572


namespace correct_change_l3805_380500

/-- Calculates the change received when purchasing frames -/
def change_received (num_frames : ℕ) (frame_cost : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_frames * frame_cost)

/-- Proves that the change received is correct for the given problem -/
theorem correct_change : change_received 3 3 20 = 11 := by
  sorry

end correct_change_l3805_380500


namespace embankment_project_additional_days_l3805_380553

/-- Represents the embankment construction project -/
structure EmbankmentProject where
  initial_workers : ℕ
  initial_days : ℕ
  reassigned_workers : ℕ
  reassignment_day : ℕ
  productivity_factor : ℚ

/-- Calculates the additional days needed to complete the project -/
def additional_days_needed (project : EmbankmentProject) : ℚ :=
  let initial_rate : ℚ := 1 / (project.initial_workers * project.initial_days)
  let work_done_before_reassignment : ℚ := project.initial_workers * initial_rate * project.reassignment_day
  let remaining_work : ℚ := 1 - work_done_before_reassignment
  let remaining_workers : ℕ := project.initial_workers - project.reassigned_workers
  let new_rate : ℚ := initial_rate * project.productivity_factor
  let total_days : ℚ := project.reassignment_day + (remaining_work / (remaining_workers * new_rate))
  total_days - project.reassignment_day

/-- Theorem stating the additional days needed for the specific project -/
theorem embankment_project_additional_days :
  let project : EmbankmentProject := {
    initial_workers := 100,
    initial_days := 5,
    reassigned_workers := 40,
    reassignment_day := 2,
    productivity_factor := 3/4
  }
  additional_days_needed project = 53333 / 1000 := by sorry

end embankment_project_additional_days_l3805_380553


namespace roots_of_polynomial_l3805_380518

/-- The polynomial function we're considering -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

/-- Theorem stating that 1, -1, and 3 are the only roots of the polynomial -/
theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
sorry

end roots_of_polynomial_l3805_380518


namespace croissant_baking_time_l3805_380595

/-- Calculates the baking time for croissants given the process parameters -/
theorem croissant_baking_time 
  (num_folds : ℕ) 
  (fold_time : ℕ) 
  (rest_time : ℕ) 
  (mixing_time : ℕ) 
  (total_time : ℕ) 
  (h1 : num_folds = 4)
  (h2 : fold_time = 5)
  (h3 : rest_time = 75)
  (h4 : mixing_time = 10)
  (h5 : total_time = 360) :
  total_time - (mixing_time + num_folds * fold_time + num_folds * rest_time) = 30 := by
  sorry

#check croissant_baking_time

end croissant_baking_time_l3805_380595


namespace race_distance_l3805_380548

/-- Race conditions and proof of distance -/
theorem race_distance (d x y z : ℝ) 
  (h1 : d / x = (d - 25) / y)  -- X beats Y by 25 meters
  (h2 : d / y = (d - 15) / z)  -- Y beats Z by 15 meters
  (h3 : d / x = (d - 37) / z)  -- X beats Z by 37 meters
  (h4 : d > 0) : d = 125 := by
  sorry

end race_distance_l3805_380548


namespace bryan_bookshelves_l3805_380534

/-- Given a total number of books and books per bookshelf, calculates the number of bookshelves -/
def calculate_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books / books_per_shelf

/-- Theorem stating that with 504 total books and 56 books per shelf, there are 9 bookshelves -/
theorem bryan_bookshelves :
  calculate_bookshelves 504 56 = 9 := by
  sorry

end bryan_bookshelves_l3805_380534


namespace monotone_sin_range_l3805_380507

theorem monotone_sin_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, Monotone (fun x ↦ f x)) →
  (∀ x ∈ Set.Icc 0 a, f x = Real.sin (2 * x + π / 3)) →
  a > 0 →
  0 < a ∧ a ≤ π / 12 := by
  sorry

end monotone_sin_range_l3805_380507


namespace min_disks_is_fifteen_l3805_380505

/-- Represents the storage problem with given file sizes and quantities --/
structure StorageProblem where
  total_files : Nat
  disk_capacity : Real
  files_09mb : Nat
  files_08mb : Nat
  files_05mb : Nat
  h_total : total_files = files_09mb + files_08mb + files_05mb
  h_capacity : disk_capacity = 2

/-- Calculates the minimum number of disks required for the given storage problem --/
def min_disks_required (problem : StorageProblem) : Nat :=
  sorry

/-- The main theorem stating that the minimum number of disks required is 15 --/
theorem min_disks_is_fifteen :
  ∀ (problem : StorageProblem),
    problem.total_files = 35 →
    problem.files_09mb = 4 →
    problem.files_08mb = 15 →
    problem.files_05mb = 16 →
    min_disks_required problem = 15 :=
  sorry

end min_disks_is_fifteen_l3805_380505


namespace complex_equation_solution_l3805_380542

theorem complex_equation_solution (z : ℂ) : (3 - 4*I + z)*I = 2 + I → z = -2 + 2*I := by
  sorry

end complex_equation_solution_l3805_380542


namespace number_division_problem_l3805_380587

theorem number_division_problem : ∃ x : ℝ, x / 5 = 80 + x / 6 ∧ x = 2400 := by
  sorry

end number_division_problem_l3805_380587


namespace quadratic_solution_with_nested_root_l3805_380501

theorem quadratic_solution_with_nested_root (a b : ℤ) :
  (∃ x : ℝ, x^2 + a*x + b = 0 ∧ x = Real.sqrt (2010 + 2 * Real.sqrt 2009)) →
  a = 0 ∧ b = -2008 := by
sorry

end quadratic_solution_with_nested_root_l3805_380501


namespace polynomial_remainder_theorem_l3805_380533

/-- The degree of a polynomial -/
def degree (p : Polynomial ℂ) : ℕ := sorry

theorem polynomial_remainder_theorem :
  ∃ (Q R : Polynomial ℂ),
    (X : Polynomial ℂ)^2023 + 1 = (X^2 + X + 1) * Q + R ∧
    degree R < 2 ∧
    R = -X + 1 := by sorry

end polynomial_remainder_theorem_l3805_380533


namespace gcd_abcd_plus_dcba_eq_one_l3805_380525

def abcd_plus_dcba (a : ℤ) : ℤ :=
  let b := a^2 + 1
  let c := a^2 + 2
  let d := a^2 + 3
  2111 * a^2 + 1001 * a + 3333

theorem gcd_abcd_plus_dcba_eq_one :
  ∃ (a : ℤ), ∀ (x : ℤ), x ∣ abcd_plus_dcba a → x = 1 ∨ x = -1 := by sorry

end gcd_abcd_plus_dcba_eq_one_l3805_380525
