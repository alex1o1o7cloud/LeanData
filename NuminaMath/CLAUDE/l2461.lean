import Mathlib

namespace tria_currency_base_l2461_246125

/-- Converts a number from base r to base 10 -/
def toBase10 (digits : List Nat) (r : Nat) : Nat :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The problem statement -/
theorem tria_currency_base : ∃! r : Nat, r > 1 ∧
  toBase10 [5, 3, 2] r + toBase10 [2, 6, 0] r + toBase10 [2, 0, 8] r = toBase10 [1, 0, 0, 0] r :=
by
  sorry

end tria_currency_base_l2461_246125


namespace classroom_shirts_and_shorts_l2461_246138

theorem classroom_shirts_and_shorts (total_students : ℕ) 
  (h1 : total_students = 81)
  (h2 : ∃ striped_shirts : ℕ, striped_shirts = 2 * total_students / 3)
  (h3 : ∃ checkered_shirts : ℕ, checkered_shirts = total_students - striped_shirts)
  (h4 : ∃ shorts : ℕ, striped_shirts = shorts + 8) :
  ∃ difference : ℕ, shorts = checkered_shirts + difference ∧ difference = 19 := by
  sorry

end classroom_shirts_and_shorts_l2461_246138


namespace fraction_equality_l2461_246170

theorem fraction_equality (p r s u : ℝ) 
  (h1 : p / r = 4)
  (h2 : s / r = 8)
  (h3 : s / u = 1 / 4) :
  u / p = 8 := by
  sorry

end fraction_equality_l2461_246170


namespace range_of_a_l2461_246108

def f (x : ℝ) := x^2 - 4*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) a, f x ∈ Set.Icc (-4 : ℝ) 5) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) a, f x = -4) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) a, f x = 5) →
  a ∈ Set.Icc 2 5 :=
by sorry

end range_of_a_l2461_246108


namespace car_journey_average_speed_l2461_246164

-- Define the parameters of the journey
def distance_part1 : ℝ := 18  -- km
def time_part1 : ℝ := 24      -- minutes
def time_part2 : ℝ := 35      -- minutes
def speed_part2 : ℝ := 72     -- km/h

-- Define the theorem
theorem car_journey_average_speed :
  let total_distance := distance_part1 + speed_part2 * (time_part2 / 60)
  let total_time := time_part1 + time_part2
  let average_speed := total_distance / (total_time / 60)
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 0.005 ∧ |average_speed - 61.02| < ε :=
by sorry

end car_journey_average_speed_l2461_246164


namespace f_satisfies_equation_l2461_246174

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0.5 then 1 / (0.5 - x) else 0.5

theorem f_satisfies_equation :
  ∀ x : ℝ, f x + (0.5 + x) * f (1 - x) = 1 := by
  sorry

end f_satisfies_equation_l2461_246174


namespace library_donation_l2461_246179

/-- The number of books donated to the library --/
def books_donated (num_students : ℕ) (books_per_student : ℕ) (shortfall : ℕ) : ℕ :=
  num_students * books_per_student - shortfall

/-- Theorem stating the number of books donated to the library --/
theorem library_donation (num_students : ℕ) (books_per_student : ℕ) (shortfall : ℕ) :
  books_donated num_students books_per_student shortfall = 294 :=
by
  sorry

#eval books_donated 20 15 6

end library_donation_l2461_246179


namespace correct_package_cost_l2461_246193

def packageCost (P : ℕ) : ℕ :=
  15 + 5 * (P - 1) - 8 * (if P ≥ 5 then 1 else 0)

theorem correct_package_cost (P : ℕ) (h : P ≥ 1) :
  packageCost P = 15 + 5 * (P - 1) - 8 * (if P ≥ 5 then 1 else 0) :=
by sorry

end correct_package_cost_l2461_246193


namespace temperature_sum_l2461_246133

theorem temperature_sum (t1 t2 t3 k1 k2 k3 : ℚ) : 
  t1 = 5 / 9 * (k1 - 32) →
  t2 = 5 / 9 * (k2 - 32) →
  t3 = 5 / 9 * (k3 - 32) →
  t1 = 105 →
  t2 = 80 →
  t3 = 45 →
  k1 + k2 + k3 = 510 := by
sorry

end temperature_sum_l2461_246133


namespace folded_rectangle_perimeter_l2461_246186

/-- Given a rectangle with length 20 cm and width 12 cm, when folded along its diagonal,
    the perimeter of the resulting shaded region is 64 cm. -/
theorem folded_rectangle_perimeter :
  ∀ (length width : ℝ),
    length = 20 →
    width = 12 →
    let perimeter := (length + width) * 2
    perimeter = 64 := by
  sorry

end folded_rectangle_perimeter_l2461_246186


namespace sum_of_coefficients_l2461_246111

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (1 - 2*x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 510 := by
  sorry

end sum_of_coefficients_l2461_246111


namespace unique_positive_solution_l2461_246161

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 :=
by
  -- The unique positive solution is 5/3
  use 5/3
  constructor
  · constructor
    · -- Prove 5/3 > 0
      sorry
    · -- Prove 3 * (5/3)^2 + 7 * (5/3) - 20 = 0
      sorry
  · -- Prove uniqueness
    sorry

end unique_positive_solution_l2461_246161


namespace complex_equation_solution_l2461_246121

/-- Given (1 + 2i)a + b = 2i, where a and b are real numbers, prove that a = 1 and b = -1 -/
theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end complex_equation_solution_l2461_246121


namespace granddaughter_age_l2461_246198

/-- Represents a family with three generations -/
structure Family where
  betty_age : ℕ
  daughter_age : ℕ
  granddaughter_age : ℕ

/-- The age relationship in the family -/
def valid_family_ages (f : Family) : Prop :=
  f.betty_age = 60 ∧
  f.daughter_age = f.betty_age - (f.betty_age * 40 / 100) ∧
  f.granddaughter_age = f.daughter_age / 3

/-- Theorem stating the granddaughter's age in the family -/
theorem granddaughter_age (f : Family) (h : valid_family_ages f) : 
  f.granddaughter_age = 12 := by
  sorry

end granddaughter_age_l2461_246198


namespace laptop_final_price_l2461_246132

/-- Calculate the final price of a laptop given the original price, two discount rates, and a recycling fee rate. -/
theorem laptop_final_price
  (original_price : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (recycling_fee_rate : ℝ)
  (h1 : original_price = 1000)
  (h2 : discount1 = 0.1)
  (h3 : discount2 = 0.25)
  (h4 : recycling_fee_rate = 0.05) :
  let price_after_discount1 := original_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let recycling_fee := price_after_discount2 * recycling_fee_rate
  let final_price := price_after_discount2 + recycling_fee
  final_price = 708.75 :=
by
  sorry

end laptop_final_price_l2461_246132


namespace solution_set_characterization_l2461_246196

def solution_set (x y : ℝ) : Prop :=
  3 * x - 4 * y + 12 > 0 ∧ x + y - 2 < 0

theorem solution_set_characterization (x y : ℝ) :
  solution_set x y ↔ (3 * x - 4 * y + 12 > 0 ∧ x + y - 2 < 0) :=
by sorry

end solution_set_characterization_l2461_246196


namespace smallest_sum_of_digits_l2461_246146

def Digits : Finset ℕ := {4, 5, 6, 7, 8, 9, 10}

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ 
  a ≥ 100 ∧ a < 1000 ∧ 
  b ≥ 100 ∧ b < 1000 ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.range 10 \ {0}))) = 7 ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits ∧ (d ∈ (Finset.range 10 \ {0}) ∨ d = 10)) ((Finset.range 10 \ {0}) ∪ {10}))) = 6

theorem smallest_sum_of_digits : 
  ∀ a b : ℕ, is_valid_pair a b → a + b ≥ 1245 :=
by sorry

end smallest_sum_of_digits_l2461_246146


namespace find_b_value_l2461_246155

theorem find_b_value (a b : ℝ) (eq1 : 3 * a + 2 = 2) (eq2 : b - a = 2) : b = 2 := by
  sorry

end find_b_value_l2461_246155


namespace consecutive_integers_average_l2461_246131

theorem consecutive_integers_average (c d : ℝ) : 
  (c ≥ 1) →  -- Ensure c is positive
  (d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5)) / 6 = c + 5) := by
sorry

end consecutive_integers_average_l2461_246131


namespace tim_meditates_one_hour_per_day_l2461_246197

/-- Tim's weekly schedule -/
structure TimSchedule where
  reading_time_per_week : ℝ
  meditation_time_per_day : ℝ

/-- Tim's schedule satisfies the given conditions -/
def valid_schedule (s : TimSchedule) : Prop :=
  s.reading_time_per_week = 14 ∧
  s.reading_time_per_week = 2 * (7 * s.meditation_time_per_day)

/-- Theorem: Tim meditates 1 hour per day -/
theorem tim_meditates_one_hour_per_day (s : TimSchedule) (h : valid_schedule s) :
  s.meditation_time_per_day = 1 := by
  sorry

end tim_meditates_one_hour_per_day_l2461_246197


namespace birds_in_tree_l2461_246116

/-- The total number of birds in a tree after two groups join -/
def total_birds (initial : ℕ) (group1 : ℕ) (group2 : ℕ) : ℕ :=
  initial + group1 + group2

/-- Theorem stating that the total number of birds is 76 -/
theorem birds_in_tree : total_birds 24 37 15 = 76 := by
  sorry

end birds_in_tree_l2461_246116


namespace union_complement_equality_l2461_246139

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}

theorem union_complement_equality : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by
  sorry

end union_complement_equality_l2461_246139


namespace book_shelf_problem_l2461_246165

theorem book_shelf_problem (total_books : ℕ) (books_moved : ℕ) 
  (h1 : total_books = 180)
  (h2 : books_moved = 15)
  (h3 : ∃ (upper lower : ℕ), 
    upper + lower = total_books ∧ 
    (lower + books_moved) = 2 * (upper - books_moved)) :
  ∃ (original_upper original_lower : ℕ),
    original_upper = 75 ∧ 
    original_lower = 105 ∧
    original_upper + original_lower = total_books := by
  sorry

end book_shelf_problem_l2461_246165


namespace smallest_n_for_inequality_l2461_246148

theorem smallest_n_for_inequality :
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), (x^2 + y^2 + z^2)^2 > m * (x^4 + y^4 + z^4)) := by
  sorry

end smallest_n_for_inequality_l2461_246148


namespace unknown_number_solution_l2461_246118

theorem unknown_number_solution : 
  ∃ x : ℚ, (x + 23 / 89) * 89 = 4028 ∧ x = 45 := by sorry

end unknown_number_solution_l2461_246118


namespace bear_food_in_victors_l2461_246182

/-- The number of "Victors" worth of food a bear eats in 3 weeks -/
def victors_worth_of_food (bear_food_per_day : ℕ) (victor_weight : ℕ) (weeks : ℕ) : ℕ :=
  (bear_food_per_day * weeks * 7) / victor_weight

/-- Theorem stating that a bear eating 90 pounds of food per day would eat 15 "Victors" worth of food in 3 weeks, given that Victor weighs 126 pounds -/
theorem bear_food_in_victors : victors_worth_of_food 90 126 3 = 15 := by
  sorry

end bear_food_in_victors_l2461_246182


namespace problem_1_l2461_246171

theorem problem_1 (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 3) :
  a^2 + b^2 = 10 := by sorry

end problem_1_l2461_246171


namespace sufficient_not_necessary_condition_l2461_246124

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a ∈ Set.Iic 0 → ∃ y : ℝ, y^2 - y + a ≤ 0) ∧
  (∃ b : ℝ, b ∉ Set.Iic 0 ∧ ∃ z : ℝ, z^2 - z + b ≤ 0) := by
  sorry

end sufficient_not_necessary_condition_l2461_246124


namespace student_weight_l2461_246176

theorem student_weight (student sister brother : ℝ) 
  (h1 : student - 5 = 2 * sister)
  (h2 : student + sister + brother = 150)
  (h3 : brother = sister - 10) :
  student = 82.5 := by
sorry

end student_weight_l2461_246176


namespace remainder_99_101_div_9_l2461_246114

theorem remainder_99_101_div_9 : (99 * 101) % 9 = 0 := by
  sorry

end remainder_99_101_div_9_l2461_246114


namespace max_servings_emily_l2461_246167

/-- Represents the recipe requirements for 4 servings --/
structure Recipe where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ

/-- Represents Emily's available ingredients --/
structure Available where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ

/-- Calculates the maximum number of servings that can be made --/
def max_servings (recipe : Recipe) (available : Available) : ℕ :=
  min
    (available.bananas * 4 / recipe.bananas)
    (min
      (available.strawberries * 4 / recipe.strawberries)
      (available.yogurt * 4 / recipe.yogurt))

theorem max_servings_emily :
  let recipe := Recipe.mk 3 1 2
  let available := Available.mk 10 3 12
  max_servings recipe available = 12 := by
  sorry

end max_servings_emily_l2461_246167


namespace min_containers_correct_l2461_246119

/-- Calculates the minimum number of containers needed to transport boxes with weight restrictions. -/
def min_containers (total_boxes : ℕ) (main_box_weight : ℕ) (light_boxes : ℕ) (light_box_weight : ℕ) (max_container_weight : ℕ) : ℕ :=
  let total_weight := (total_boxes - light_boxes) * main_box_weight + light_boxes * light_box_weight
  let boxes_per_container := max_container_weight * 1000 / main_box_weight
  (total_boxes + boxes_per_container - 1) / boxes_per_container

theorem min_containers_correct :
  min_containers 90000 3300 5000 200 100 = 3000 := by
  sorry

end min_containers_correct_l2461_246119


namespace solve_for_q_l2461_246106

theorem solve_for_q (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -25 / 11 := by
sorry

end solve_for_q_l2461_246106


namespace apple_preference_percentage_l2461_246157

/-- Represents the frequencies of fruits in a survey --/
structure FruitSurvey where
  apples : ℕ
  bananas : ℕ
  cherries : ℕ
  oranges : ℕ
  grapes : ℕ

/-- Calculates the total number of responses in the survey --/
def totalResponses (survey : FruitSurvey) : ℕ :=
  survey.apples + survey.bananas + survey.cherries + survey.oranges + survey.grapes

/-- Calculates the percentage of respondents who preferred apples --/
def applePercentage (survey : FruitSurvey) : ℚ :=
  (survey.apples : ℚ) / (totalResponses survey : ℚ) * 100

/-- The given survey results --/
def givenSurvey : FruitSurvey :=
  { apples := 70
  , bananas := 50
  , cherries := 30
  , oranges := 50
  , grapes := 40 }

theorem apple_preference_percentage :
  applePercentage givenSurvey = 29 := by
  sorry

end apple_preference_percentage_l2461_246157


namespace fans_with_all_items_l2461_246180

def stadium_capacity : ℕ := 4800
def scarf_interval : ℕ := 80
def hat_interval : ℕ := 40
def whistle_interval : ℕ := 60

theorem fans_with_all_items :
  (stadium_capacity / (lcm scarf_interval (lcm hat_interval whistle_interval))) = 20 := by
  sorry

end fans_with_all_items_l2461_246180


namespace x0_value_l2461_246104

-- Define the function f
def f (x : ℝ) : ℝ := x^5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 5 * x^4

-- Theorem statement
theorem x0_value (x₀ : ℝ) (h : f' x₀ = 20) : x₀ = Real.sqrt 2 ∨ x₀ = -Real.sqrt 2 := by
  sorry

end x0_value_l2461_246104


namespace best_distribution_for_1_l2461_246101

/-- Represents a distribution of pearls among 4 people -/
def Distribution := Fin 4 → ℕ

/-- The total number of pearls to be distributed -/
def totalPearls : ℕ := 10

/-- A valid distribution must sum to the total number of pearls -/
def isValidDistribution (d : Distribution) : Prop :=
  (Finset.univ.sum d) = totalPearls

/-- A distribution passes if it has at least half of the votes -/
def passes (d : Distribution) : Prop :=
  2 * (Finset.filter (fun i => d i > 0) Finset.univ).card ≥ 4

/-- The best distribution for person 3 if 1 and 2 are eliminated -/
def bestFor3 : Distribution :=
  fun i => if i = 2 then 10 else 0

/-- The proposed best distribution for person 1 -/
def proposedBest : Distribution :=
  fun i => match i with
  | 0 => 9
  | 2 => 1
  | _ => 0

theorem best_distribution_for_1 :
  isValidDistribution proposedBest ∧
  passes proposedBest ∧
  ∀ d : Distribution, isValidDistribution d ∧ passes d → proposedBest 0 ≥ d 0 :=
by sorry

end best_distribution_for_1_l2461_246101


namespace smallest_four_digit_multiple_l2461_246199

theorem smallest_four_digit_multiple : ∃ (n : ℕ), 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  (n % 5 = 0) ∧
  (n % 11 = 7) ∧
  (n % 7 = 4) ∧
  (n % 9 = 4) ∧
  (∀ m : ℕ, 
    (m ≥ 1000 ∧ m < 10000) ∧ 
    (m % 5 = 0) ∧
    (m % 11 = 7) ∧
    (m % 7 = 4) ∧
    (m % 9 = 4) →
    n ≤ m) ∧
  n = 2020 := by
  sorry

end smallest_four_digit_multiple_l2461_246199


namespace school_boys_count_l2461_246110

theorem school_boys_count :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 64 →
  boys = 40 :=
by
  sorry

end school_boys_count_l2461_246110


namespace total_cost_is_14000_l2461_246115

/-- Represents the dimensions and costs of the roads on a rectangular lawn. -/
structure LawnRoads where
  lawn_length : ℝ
  lawn_width : ℝ
  road1_width : ℝ
  road1_cost_per_sqm : ℝ
  road2_width : ℝ
  road2_cost_per_sqm : ℝ
  hill_length : ℝ
  hill_cost_increase : ℝ

/-- Calculates the total cost of traveling both roads on the lawn. -/
def total_cost (lr : LawnRoads) : ℝ :=
  let road1_area := lr.lawn_length * lr.road1_width
  let road1_cost := road1_area * lr.road1_cost_per_sqm
  let hill_area := lr.hill_length * lr.road1_width
  let hill_additional_cost := hill_area * (lr.road1_cost_per_sqm * lr.hill_cost_increase)
  let road2_area := lr.lawn_width * lr.road2_width
  let road2_cost := road2_area * lr.road2_cost_per_sqm
  road1_cost + hill_additional_cost + road2_cost

/-- Theorem stating that the total cost of traveling both roads is 14000. -/
theorem total_cost_is_14000 (lr : LawnRoads) 
    (h1 : lr.lawn_length = 150)
    (h2 : lr.lawn_width = 80)
    (h3 : lr.road1_width = 12)
    (h4 : lr.road1_cost_per_sqm = 4)
    (h5 : lr.road2_width = 8)
    (h6 : lr.road2_cost_per_sqm = 5)
    (h7 : lr.hill_length = 60)
    (h8 : lr.hill_cost_increase = 0.25) :
    total_cost lr = 14000 := by
  sorry


end total_cost_is_14000_l2461_246115


namespace original_stations_count_l2461_246183

def number_of_ticket_types (k : ℕ) : ℕ := k * (k - 1) / 2

theorem original_stations_count 
  (m n : ℕ) 
  (h1 : n > 1) 
  (h2 : number_of_ticket_types (m + n) - number_of_ticket_types m = 58) : 
  m = 14 := by
sorry

end original_stations_count_l2461_246183


namespace negation_of_universal_proposition_l2461_246102

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end negation_of_universal_proposition_l2461_246102


namespace ball_cost_l2461_246145

theorem ball_cost (C : ℝ) : 
  (C / 2 + C / 6 + C / 12 + 5 = C) → C = 20 := by sorry

end ball_cost_l2461_246145


namespace sqrt_sum_diff_equals_fifteen_halves_l2461_246159

theorem sqrt_sum_diff_equals_fifteen_halves :
  Real.sqrt 9 + Real.sqrt 25 - Real.sqrt (1/4) = 15/2 := by
  sorry

end sqrt_sum_diff_equals_fifteen_halves_l2461_246159


namespace krishans_money_l2461_246137

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount,
    prove that Krishan's amount is 3774. -/
theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 637 →
  krishan = 3774 := by
sorry

end krishans_money_l2461_246137


namespace unit_circle_tangent_l2461_246153

theorem unit_circle_tangent (x y θ : ℝ) : 
  x^2 + y^2 = 1 →  -- Point (x, y) is on the unit circle
  x > 0 →          -- Point is in the first quadrant
  y > 0 →          -- Point is in the first quadrant
  x = Real.cos θ → -- θ is the angle from positive x-axis
  y = Real.sin θ → -- to the ray through (x, y)
  Real.arccos ((4*x + 3*y) / 5) = θ → -- Given condition
  Real.tan θ = 1/3 := by sorry

end unit_circle_tangent_l2461_246153


namespace geometric_sequence_sum_l2461_246100

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a n > 0) →  -- positive terms
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence
  a 1 = 3 →  -- first term
  a 1 + a 2 + a 3 = 21 →  -- sum of first three terms
  a 3 + a 4 + a 5 = 84 := by
sorry

end geometric_sequence_sum_l2461_246100


namespace small_pond_green_percentage_l2461_246150

def total_ducks : ℕ := 100
def small_pond_ducks : ℕ := 20
def large_pond_ducks : ℕ := 80
def large_pond_green_percentage : ℚ := 15 / 100
def total_green_percentage : ℚ := 16 / 100

theorem small_pond_green_percentage :
  ∃ x : ℚ,
    x * small_pond_ducks + large_pond_green_percentage * large_pond_ducks =
    total_green_percentage * total_ducks ∧
    x = 20 / 100 := by
  sorry

end small_pond_green_percentage_l2461_246150


namespace problem_solution_l2461_246188

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + 2 * |x - 1|

-- Define the function g
def g (x m : ℝ) : ℝ := |x + 1 + m| + 2 * |x|

theorem problem_solution :
  (∀ x : ℝ, m > 0 → 
    (m = 1 → (f x m ≤ 10 ↔ -3 ≤ x ∧ x ≤ 11/3))) ∧
  (∀ m : ℝ, (∀ x : ℝ, g x m ≥ 3) ↔ m ≥ 2) :=
sorry

end problem_solution_l2461_246188


namespace apples_after_sharing_l2461_246144

/-- The total number of apples Craig and Judy have after sharing -/
def total_apples_after_sharing (craig_initial : ℕ) (judy_initial : ℕ) (craig_shared : ℕ) (judy_shared : ℕ) : ℕ :=
  (craig_initial - craig_shared) + (judy_initial - judy_shared)

/-- Theorem: Given the initial and shared apple counts, Craig and Judy have 19 apples together after sharing -/
theorem apples_after_sharing :
  total_apples_after_sharing 20 11 7 5 = 19 := by
  sorry

end apples_after_sharing_l2461_246144


namespace largest_common_term_l2461_246105

def is_in_sequence (start : ℤ) (diff : ℤ) (n : ℤ) : Prop :=
  ∃ k : ℤ, n = start + k * diff

theorem largest_common_term : ∃ n : ℤ,
  (1 ≤ n ∧ n ≤ 100) ∧
  (is_in_sequence 2 5 n) ∧
  (is_in_sequence 3 8 n) ∧
  (∀ m : ℤ, (1 ≤ m ∧ m ≤ 100) → 
    (is_in_sequence 2 5 m) → 
    (is_in_sequence 3 8 m) → 
    m ≤ n) ∧
  n = 67 := by
  sorry

end largest_common_term_l2461_246105


namespace democrat_ratio_l2461_246136

theorem democrat_ratio (total : ℕ) (female_dem : ℕ) :
  total = 840 →
  female_dem = 140 →
  (∃ (female male : ℕ),
    female + male = total ∧
    2 * female_dem = female ∧
    4 * female_dem = male) →
  3 * (2 * female_dem) = total :=
by
  sorry

end democrat_ratio_l2461_246136


namespace bicycle_trip_time_l2461_246190

theorem bicycle_trip_time (distance : Real) (outbound_speed return_speed : Real) :
  distance = 28.8 ∧ outbound_speed = 16 ∧ return_speed = 24 →
  distance / outbound_speed + distance / return_speed = 3 := by
  sorry

end bicycle_trip_time_l2461_246190


namespace difference_of_squares_l2461_246107

theorem difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end difference_of_squares_l2461_246107


namespace david_presents_l2461_246191

/-- The total number of presents David received -/
def total_presents (christmas_presents : ℕ) (birthday_presents : ℕ) : ℕ :=
  christmas_presents + birthday_presents

/-- Theorem: Given the conditions, David received 90 presents in total -/
theorem david_presents : 
  ∀ (christmas_presents birthday_presents : ℕ),
  christmas_presents = 60 →
  christmas_presents = 2 * birthday_presents →
  total_presents christmas_presents birthday_presents = 90 := by
  sorry

end david_presents_l2461_246191


namespace cherry_tree_leaves_l2461_246173

theorem cherry_tree_leaves (original_plan : ℕ) (actual_multiplier : ℕ) (leaves_per_tree : ℕ) : 
  original_plan = 7 → 
  actual_multiplier = 2 → 
  leaves_per_tree = 100 → 
  (original_plan * actual_multiplier * leaves_per_tree) = 1400 := by
sorry

end cherry_tree_leaves_l2461_246173


namespace extreme_values_l2461_246147

/-- A quadratic function passing through four points with specific properties. -/
structure QuadraticFunction where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  y₄ : ℝ
  h₁ : y₂ < y₃
  h₂ : y₃ = y₄

/-- Theorem stating that y₁ is the smallest and y₃ is the largest among y₁, y₂, and y₃ -/
theorem extreme_values (f : QuadraticFunction) : 
  f.y₁ ≤ f.y₂ ∧ f.y₁ ≤ f.y₃ ∧ f.y₂ < f.y₃ := by
  sorry

#check extreme_values

end extreme_values_l2461_246147


namespace expression_value_l2461_246158

theorem expression_value : 
  2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 4000 := by
  sorry

end expression_value_l2461_246158


namespace parabola_range_theorem_l2461_246194

/-- Represents a quadratic function of the form f(x) = x^2 + bx + c -/
def QuadraticFunction (b c : ℝ) := λ x : ℝ => x^2 + b*x + c

theorem parabola_range_theorem (b c : ℝ) :
  (QuadraticFunction b c (-1) = 0) →
  (QuadraticFunction b c 3 = 0) →
  (∀ x : ℝ, QuadraticFunction b c x > -3 ↔ (x < 0 ∨ x > 2)) :=
by sorry

end parabola_range_theorem_l2461_246194


namespace expand_product_l2461_246166

theorem expand_product (x : ℝ) : (x + 3) * (x + 4) = x^2 + 7*x + 12 := by
  sorry

end expand_product_l2461_246166


namespace max_sum_of_factors_of_60_l2461_246140

theorem max_sum_of_factors_of_60 : 
  ∃ (a b : ℕ), a * b = 60 ∧ 
  ∀ (x y : ℕ), x * y = 60 → x + y ≤ a + b ∧ a + b = 61 := by
sorry

end max_sum_of_factors_of_60_l2461_246140


namespace quadratic_distinct_roots_l2461_246184

theorem quadratic_distinct_roots (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*(m-2)*x + m^2 = 0 → (∃ y : ℝ, x ≠ y ∧ y^2 - 2*(m-2)*y + m^2 = 0)) →
  m < 1 :=
by sorry

end quadratic_distinct_roots_l2461_246184


namespace sum_f_2015_is_zero_l2461_246126

-- Define the properties of the function f
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (2 - x) = f x

def sum_f (f : ℝ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_f f n + f (n + 1)

theorem sum_f_2015_is_zero 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_sym : is_symmetric_about_one f) 
  (h_f_neg_one : f (-1) = 1) : 
  sum_f f 2015 = 0 := by
  sorry

end sum_f_2015_is_zero_l2461_246126


namespace symmetric_point_on_x_axis_l2461_246135

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the form y = kx -/
structure Line where
  k : ℝ

/-- Predicate to check if a point is on the x-axis -/
def onXAxis (p : Point) : Prop :=
  p.y = 0

/-- Predicate to check if two points are symmetric about a line -/
def areSymmetric (p1 p2 : Point) (l : Line) : Prop :=
  (p1.y + p2.y) / 2 = l.k * ((p1.x + p2.x) / 2) ∧
  (p1.y - p2.y) / (p1.x - p2.x) * l.k = -1

theorem symmetric_point_on_x_axis (A : Point) (l : Line) :
  A.x = 3 ∧ A.y = 5 →
  ∃ (B : Point), areSymmetric A B l ∧ onXAxis B →
  l.k = (-3 + Real.sqrt 34) / 5 ∨ l.k = (-3 - Real.sqrt 34) / 5 := by
  sorry

end symmetric_point_on_x_axis_l2461_246135


namespace buns_eaten_proof_l2461_246142

/-- Represents the number of buns eaten by Zhenya -/
def zhenya_buns : ℕ := 40

/-- Represents the number of buns eaten by Sasha -/
def sasha_buns : ℕ := 30

/-- The total number of buns eaten -/
def total_buns : ℕ := 70

/-- The total eating time in minutes -/
def total_time : ℕ := 180

/-- Zhenya's eating rate in buns per minute -/
def zhenya_rate : ℚ := 1/2

/-- Sasha's eating rate in buns per minute -/
def sasha_rate : ℚ := 3/10

theorem buns_eaten_proof :
  zhenya_buns + sasha_buns = total_buns ∧
  zhenya_rate * total_time = zhenya_buns ∧
  sasha_rate * total_time = sasha_buns :=
by sorry

#check buns_eaten_proof

end buns_eaten_proof_l2461_246142


namespace product_of_roots_l2461_246152

theorem product_of_roots (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016)
  (h₄ : x₄^3 - 3*x₄*y₄^2 = 2017 ∧ y₄^3 - 3*x₄^2*y₄ = 2016)
  (h₅ : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0 ∧ y₄ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) * (1 - x₄/y₄) = -1/1008 := by
sorry

end product_of_roots_l2461_246152


namespace laptop_price_difference_l2461_246143

/-- The price difference between two stores for Laptop Y -/
theorem laptop_price_difference
  (list_price : ℝ)
  (gadget_gurus_discount_percent : ℝ)
  (tech_trends_discount_amount : ℝ)
  (h1 : list_price = 300)
  (h2 : gadget_gurus_discount_percent = 0.15)
  (h3 : tech_trends_discount_amount = 45) :
  list_price * (1 - gadget_gurus_discount_percent) = list_price - tech_trends_discount_amount :=
by sorry

end laptop_price_difference_l2461_246143


namespace fraction_equality_l2461_246134

theorem fraction_equality (x y : ℝ) (h : (x + y) / (1 - x * y) = Real.sqrt 5) :
  |1 - x * y| / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2)) = Real.sqrt 6 / 6 := by
  sorry

end fraction_equality_l2461_246134


namespace smallest_integers_for_720_square_and_cube_l2461_246195

theorem smallest_integers_for_720_square_and_cube (a b : ℕ+) : 
  (∀ x : ℕ+, x < a → ¬∃ y : ℕ, 720 * x = y * y) ∧
  (∀ x : ℕ+, x < b → ¬∃ y : ℕ, 720 * x = y * y * y) ∧
  (∃ y : ℕ, 720 * a = y * y) ∧
  (∃ y : ℕ, 720 * b = y * y * y) →
  a + b = 305 := by
sorry

end smallest_integers_for_720_square_and_cube_l2461_246195


namespace min_production_volume_correct_l2461_246187

/-- The minimum production volume to avoid a loss -/
def min_production_volume : ℕ := 150

/-- The total cost function -/
def total_cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The selling price per unit -/
def selling_price : ℝ := 25

/-- Theorem stating the minimum production volume to avoid a loss -/
theorem min_production_volume_correct :
  ∀ x : ℕ, 0 < x → x < 240 →
  (selling_price * x ≥ total_cost x ↔ x ≥ min_production_volume) := by
  sorry

end min_production_volume_correct_l2461_246187


namespace x_neg_one_is_local_minimum_l2461_246169

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 1

theorem x_neg_one_is_local_minimum :
  ∃ δ > 0, ∀ x : ℝ, x ≠ -1 ∧ |x - (-1)| < δ → f x ≥ f (-1) := by
  sorry

end x_neg_one_is_local_minimum_l2461_246169


namespace solution_ranges_l2461_246123

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2*m - 2

-- Define the conditions
def has_solution_in_closed_interval (m : ℝ) : Prop :=
  ∃ x, x ∈ Set.Icc 0 (3/2) ∧ quadratic m x = 0

def has_solution_in_open_interval (m : ℝ) : Prop :=
  ∃ x, x ∈ Set.Ioo 0 (3/2) ∧ quadratic m x = 0

def has_exactly_one_solution_in_open_interval (m : ℝ) : Prop :=
  ∃! x, x ∈ Set.Ioo 0 (3/2) ∧ quadratic m x = 0

def has_two_solutions_in_closed_interval (m : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ x ∈ Set.Icc 0 (3/2) ∧ y ∈ Set.Icc 0 (3/2) ∧ quadratic m x = 0 ∧ quadratic m y = 0

-- Theorem statements
theorem solution_ranges :
  (∀ m, has_solution_in_closed_interval m ↔ m ∈ Set.Icc (-1/2) (4 - 2*Real.sqrt 2)) ∧
  (∀ m, has_solution_in_open_interval m ↔ m ∈ Set.Ico (-1/2) (4 - 2*Real.sqrt 2)) ∧
  (∀ m, has_exactly_one_solution_in_open_interval m ↔ m ∈ Set.Ioc (-1/2) 1) ∧
  (∀ m, has_two_solutions_in_closed_interval m ↔ m ∈ Set.Ioo 1 (4 - 2*Real.sqrt 2)) :=
sorry

end solution_ranges_l2461_246123


namespace road_trip_gas_cost_l2461_246113

/-- Calculates the total cost of filling up a car's gas tank at multiple stations -/
theorem road_trip_gas_cost (tank_capacity : ℝ) (prices : List ℝ) : 
  tank_capacity = 12 ∧ 
  prices = [3, 3.5, 4, 4.5] →
  (prices.map (λ price => tank_capacity * price)).sum = 180 := by
  sorry

end road_trip_gas_cost_l2461_246113


namespace y₁_greater_than_y₂_l2461_246177

-- Define the line
def line (x : ℝ) (b : ℝ) : ℝ := -2023 * x + b

-- Define the points A and B
def point_A (y₁ : ℝ) : ℝ × ℝ := (-2, y₁)
def point_B (y₂ : ℝ) : ℝ × ℝ := (-1, y₂)

-- Theorem statement
theorem y₁_greater_than_y₂ (b y₁ y₂ : ℝ) 
  (h₁ : line (-2) b = y₁)
  (h₂ : line (-1) b = y₂) :
  y₁ > y₂ := by
  sorry

end y₁_greater_than_y₂_l2461_246177


namespace subset_union_implies_complement_superset_l2461_246141

universe u

theorem subset_union_implies_complement_superset
  {U : Type u} [CompleteLattice U]
  (M N : Set U) (h : M ∪ N = N) :
  (M : Set U)ᶜ ⊇ (N : Set U)ᶜ :=
by sorry

end subset_union_implies_complement_superset_l2461_246141


namespace fraction_modification_l2461_246175

theorem fraction_modification (a : ℕ) : (29 - a : ℚ) / (43 + a) = 3/5 → a = 2 := by
  sorry

end fraction_modification_l2461_246175


namespace reflection_line_sum_l2461_246185

/-- Given a line y = mx + b, if the reflection of point (1, -2) across this line is (7, 4), then m + b = 4 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The midpoint of the segment is on the line
    y = m * x + b ∧ 
    -- The midpoint coordinates
    x = (1 + 7) / 2 ∧ 
    y = (-2 + 4) / 2 ∧ 
    -- The line is perpendicular to the segment
    m * ((7 - 1) / (4 - (-2))) = -1) → 
  m + b = 4 := by
sorry

end reflection_line_sum_l2461_246185


namespace fraction_equality_l2461_246112

theorem fraction_equality (a b : ℝ) (h : 1/a - 1/b = 4) : 
  (a - 2*a*b - b) / (2*a + 7*a*b - 2*b) = 6 := by
  sorry

end fraction_equality_l2461_246112


namespace special_quadrilateral_not_necessarily_square_l2461_246178

/-- A quadrilateral with perpendicular and bisecting diagonals -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perpendicular_diagonals : Bool
  /-- The quadrilateral has bisecting diagonals -/
  bisecting_diagonals : Bool

/-- Definition of a square -/
structure Square where
  /-- All sides of the square are equal -/
  equal_sides : Bool
  /-- All angles of the square are right angles -/
  right_angles : Bool

/-- Theorem stating that a quadrilateral with perpendicular and bisecting diagonals is not necessarily a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ (q : SpecialQuadrilateral), q.perpendicular_diagonals ∧ q.bisecting_diagonals ∧
  ∃ (s : Square), ¬(q.perpendicular_diagonals → s.equal_sides ∧ s.right_angles) :=
sorry

end special_quadrilateral_not_necessarily_square_l2461_246178


namespace rectangle_area_with_circles_l2461_246160

/-- The area of a rectangle with specific circle arrangement -/
theorem rectangle_area_with_circles (d : ℝ) (w l : ℝ) : 
  d = 6 →                    -- diameter of each circle
  w = 3 * d →                -- width equals total diameter of three circles
  l = 2 * w →                -- length is twice the width
  w * l = 648 := by           -- area of the rectangle
  sorry

#check rectangle_area_with_circles

end rectangle_area_with_circles_l2461_246160


namespace arithmetic_sequence_length_specific_l2461_246109

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (start : Int) (diff : Int) (endInclusive : Int) : Nat :=
  if start > endInclusive then 0
  else ((endInclusive - start) / diff).toNat + 1

theorem arithmetic_sequence_length_specific :
  arithmeticSequenceLength (-48) 7 119 = 24 := by
  sorry

end arithmetic_sequence_length_specific_l2461_246109


namespace M_greater_than_N_l2461_246103

theorem M_greater_than_N (a : ℝ) : 2*a*(a-2) + 7 > (a-2)*(a-3) := by
  sorry

end M_greater_than_N_l2461_246103


namespace balloon_count_theorem_l2461_246189

/-- Represents the number of balloons each person has --/
structure BalloonCount where
  allan : ℕ
  jake : ℕ
  sarah : ℕ

/-- Initial balloon count --/
def initial : BalloonCount :=
  { allan := 5, jake := 4, sarah := 0 }

/-- Sarah buys balloons at the park --/
def sarah_buys (bc : BalloonCount) (n : ℕ) : BalloonCount :=
  { bc with sarah := bc.sarah + n }

/-- Allan buys balloons at the park --/
def allan_buys (bc : BalloonCount) (n : ℕ) : BalloonCount :=
  { bc with allan := bc.allan + n }

/-- Allan gives balloons to Jake --/
def allan_gives_to_jake (bc : BalloonCount) (n : ℕ) : BalloonCount :=
  { bc with allan := bc.allan - n, jake := bc.jake + n }

/-- The final balloon count after all actions --/
def final : BalloonCount :=
  allan_gives_to_jake (allan_buys (sarah_buys initial 7) 3) 2

theorem balloon_count_theorem :
  final = { allan := 6, jake := 6, sarah := 7 } := by
  sorry

end balloon_count_theorem_l2461_246189


namespace power_of_power_l2461_246130

theorem power_of_power (a : ℝ) : (a^2)^10 = a^20 := by sorry

end power_of_power_l2461_246130


namespace vector_addition_l2461_246129

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![3, 0]

-- State the theorem
theorem vector_addition :
  (a + b) = ![4, 2] := by sorry

end vector_addition_l2461_246129


namespace range_of_a_for_inequality_l2461_246117

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 4 := by
  sorry

end range_of_a_for_inequality_l2461_246117


namespace rotate_A_180_origin_l2461_246163

def rotate_180_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), -(p.2))

theorem rotate_A_180_origin :
  let A : ℝ × ℝ := (1, 2)
  rotate_180_origin A = (-1, -2) := by sorry

end rotate_A_180_origin_l2461_246163


namespace acute_triangle_angle_b_l2461_246151

theorem acute_triangle_angle_b (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- Acute triangle condition
  A + B + C = π →  -- Sum of angles in a triangle
  Real.sqrt 3 * a = 2 * b * Real.sin B * Real.cos C + 2 * b * Real.sin C * Real.cos B →
  B = π/3 := by
sorry

end acute_triangle_angle_b_l2461_246151


namespace length_of_AB_l2461_246181

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x + y - 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 9 = 0

-- Define that l is the axis of symmetry of C
def is_axis_of_symmetry (k : ℝ) : Prop := 
  ∀ x y : ℝ, line_l k x y → (circle_C x y ↔ circle_C (2*3-x) (2*(-1)-y))

-- Define point A
def point_A (k : ℝ) : ℝ × ℝ := (0, k)

-- Define that there exists a point B on circle C such that AB is tangent to C
def exists_tangent_point (k : ℝ) : Prop :=
  ∃ B : ℝ × ℝ, circle_C B.1 B.2 ∧ 
    ((B.1 - 0) * (B.2 - k) = 1 ∨ (B.1 - 0) * (B.2 - k) = -1)

-- Theorem statement
theorem length_of_AB (k : ℝ) :
  is_axis_of_symmetry k →
  exists_tangent_point k →
  ∃ B : ℝ × ℝ, circle_C B.1 B.2 ∧ 
    ((B.1 - 0) * (B.2 - k) = 1 ∨ (B.1 - 0) * (B.2 - k) = -1) ∧
    Real.sqrt ((B.1 - 0)^2 + (B.2 - k)^2) = 2 * Real.sqrt 3 :=
sorry

end length_of_AB_l2461_246181


namespace problem_statement_l2461_246192

theorem problem_statement (x n : ℕ) : 
  x = 5^n - 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2^(Nat.log2 x) * p * q * 11) →
  x = 3124 := by
sorry

end problem_statement_l2461_246192


namespace museum_pictures_l2461_246149

theorem museum_pictures (zoo_pics : ℕ) (deleted_pics : ℕ) (remaining_pics : ℕ) :
  zoo_pics = 41 →
  deleted_pics = 15 →
  remaining_pics = 55 →
  ∃ museum_pics : ℕ, zoo_pics + museum_pics = remaining_pics + deleted_pics ∧ museum_pics = 29 :=
by sorry

end museum_pictures_l2461_246149


namespace circle_radius_condition_l2461_246122

theorem circle_radius_condition (c : ℝ) : 
  (∀ x y : ℝ, x^2 - 4*x + y^2 + 2*y + c = 0 ↔ (x - 2)^2 + (y + 1)^2 = 5^2) → 
  c = -20 :=
by sorry

end circle_radius_condition_l2461_246122


namespace fraction_product_cube_main_problem_l2461_246127

theorem fraction_product_cube (a b c d e f : ℚ) :
  (a / b) ^ 3 * (c / d) ^ 3 * (e / f) ^ 3 = ((a * c * e) / (b * d * f)) ^ 3 :=
sorry

theorem main_problem : 
  (8 / 9) ^ 3 * (1 / 3) ^ 3 * (2 / 5) ^ 3 = 4096 / 2460375 :=
sorry

end fraction_product_cube_main_problem_l2461_246127


namespace complex_projective_transformation_properties_l2461_246172

-- Define a complex projective transformation
noncomputable def ComplexProjectiveTransformation := ℂ → ℂ

-- State the theorem
theorem complex_projective_transformation_properties
  (f : ComplexProjectiveTransformation) :
  (∃ (a b c d : ℂ), ∀ z, f z = (a * z + b) / (c * z + d)) ∧
  (∃! (p q : ℂ), f p = p ∧ f q = q) :=
sorry

end complex_projective_transformation_properties_l2461_246172


namespace baseball_card_value_decrease_l2461_246128

theorem baseball_card_value_decrease :
  ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - 0.2)
  let value_after_second_year := value_after_first_year * (1 - 0.2)
  let total_decrease := initial_value - value_after_second_year
  let percent_decrease := (total_decrease / initial_value) * 100
  percent_decrease = 36 := by
sorry

end baseball_card_value_decrease_l2461_246128


namespace valid_arrangements_count_l2461_246154

/-- Represents the four types of crops -/
inductive Crop
| Corn
| Wheat
| Soybeans
| Potatoes

/-- Represents a position in the 3x3 grid -/
structure Position :=
  (row : Fin 3)
  (col : Fin 3)

/-- Checks if two positions are adjacent -/
def are_adjacent (p1 p2 : Position) : Bool :=
  (p1.row = p2.row ∧ (p1.col.val + 1 = p2.col.val ∨ p2.col.val + 1 = p1.col.val)) ∨
  (p1.col = p2.col ∧ (p1.row.val + 1 = p2.row.val ∨ p2.row.val + 1 = p1.row.val))

/-- Represents a planting arrangement -/
def Arrangement := Position → Crop

/-- Checks if an arrangement is valid according to the rules -/
def is_valid_arrangement (arr : Arrangement) : Prop :=
  ∀ p1 p2 : Position,
    are_adjacent p1 p2 →
      (arr p1 ≠ arr p2) ∧
      ¬(arr p1 = Crop.Corn ∧ arr p2 = Crop.Wheat) ∧
      ¬(arr p1 = Crop.Wheat ∧ arr p2 = Crop.Corn)

/-- The main theorem to be proved -/
theorem valid_arrangements_count :
  ∃ (arrangements : Finset Arrangement),
    (∀ arr ∈ arrangements, is_valid_arrangement arr) ∧
    (∀ arr, is_valid_arrangement arr → arr ∈ arrangements) ∧
    arrangements.card = 16 := by
  sorry

end valid_arrangements_count_l2461_246154


namespace A_power_100_l2461_246168

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, 1; -9, -2]

theorem A_power_100 : A ^ 100 = !![301, 100; -900, -299] := by sorry

end A_power_100_l2461_246168


namespace simplify_sqrt_expression_l2461_246156

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 288) + (Real.sqrt 245 / Real.sqrt 96) = (30 + 7 * Real.sqrt 30) / 24 := by
  sorry

end simplify_sqrt_expression_l2461_246156


namespace puppy_food_bags_puppy_food_bags_proof_l2461_246162

/-- Calculates the number of bags of special dog food needed for a puppy's first year --/
theorem puppy_food_bags : ℕ :=
  let days_in_year : ℕ := 365
  let first_period : ℕ := 60
  let second_period : ℕ := days_in_year - first_period
  let first_period_consumption : ℕ := first_period * 2
  let second_period_consumption : ℕ := second_period * 4
  let total_consumption : ℕ := first_period_consumption + second_period_consumption
  let ounces_per_pound : ℕ := 16
  let pounds_per_bag : ℕ := 5
  let ounces_per_bag : ℕ := ounces_per_pound * pounds_per_bag
  let bags_needed : ℕ := (total_consumption + ounces_per_bag - 1) / ounces_per_bag
  17

/-- Proof that the number of bags needed is 17 --/
theorem puppy_food_bags_proof : puppy_food_bags = 17 := by
  sorry

end puppy_food_bags_puppy_food_bags_proof_l2461_246162


namespace land_division_l2461_246120

theorem land_division (total_land : ℝ) (num_siblings : ℕ) (jose_share : ℝ) : 
  total_land = 20000 ∧ num_siblings = 4 → 
  jose_share = total_land / (num_siblings + 1) ∧ 
  jose_share = 4000 := by
sorry

end land_division_l2461_246120
