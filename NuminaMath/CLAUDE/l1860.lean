import Mathlib

namespace contrapositive_equivalence_l1860_186072

theorem contrapositive_equivalence (x : ℝ) :
  (x > 1 → x^2 > 1) ↔ (x^2 ≤ 1 → x ≤ 1) := by sorry

end contrapositive_equivalence_l1860_186072


namespace jennifer_run_time_l1860_186037

/-- 
Given:
- Jennifer ran 3 miles in 1/3 of the time it took Mark to run 5 miles
- Mark took 45 minutes to run 5 miles

Prove that Jennifer would take 35 minutes to run 7 miles at the same rate.
-/
theorem jennifer_run_time 
  (mark_distance : ℝ) 
  (mark_time : ℝ) 
  (jennifer_distance : ℝ) 
  (jennifer_time_ratio : ℝ) 
  (jennifer_new_distance : ℝ)
  (h1 : mark_distance = 5)
  (h2 : mark_time = 45)
  (h3 : jennifer_distance = 3)
  (h4 : jennifer_time_ratio = 1/3)
  (h5 : jennifer_new_distance = 7)
  : (jennifer_new_distance / jennifer_distance) * (jennifer_time_ratio * mark_time) = 35 := by
  sorry

#check jennifer_run_time

end jennifer_run_time_l1860_186037


namespace used_car_clients_l1860_186036

theorem used_car_clients (num_cars : ℕ) (selections_per_car : ℕ) (cars_per_client : ℕ)
  (h_num_cars : num_cars = 16)
  (h_selections_per_car : selections_per_car = 3)
  (h_cars_per_client : cars_per_client = 2) :
  (num_cars * selections_per_car) / cars_per_client = 24 := by
  sorry

end used_car_clients_l1860_186036


namespace unique_satisfying_function_l1860_186067

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, (f n + m) ∣ (n^2 + f n * f m)

theorem unique_satisfying_function :
  ∃! f : ℕ → ℕ, satisfies_condition f ∧ ∀ n : ℕ, f n = n :=
by sorry

end unique_satisfying_function_l1860_186067


namespace reciprocal_product_theorem_l1860_186087

theorem reciprocal_product_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a + b = 3 * a * b) : 
  (1 / a) * (1 / b) = 9 / 4 := by
sorry

end reciprocal_product_theorem_l1860_186087


namespace division_remainder_l1860_186096

theorem division_remainder (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 13 →
  divisor = 7 →
  quotient = 1 →
  remainder = 6 := by
sorry

end division_remainder_l1860_186096


namespace cost_of_grapes_and_pineapple_l1860_186047

/-- Represents the price of fruits and their combinations -/
structure FruitPrices where
  f : ℚ  -- price of one piece of fruit
  g : ℚ  -- price of a bunch of grapes
  p : ℚ  -- price of a pineapple
  φ : ℚ  -- price of a pack of figs

/-- The conditions given in the problem -/
def satisfiesConditions (prices : FruitPrices) : Prop :=
  3 * prices.f + 2 * prices.g + prices.p + prices.φ = 36 ∧
  prices.φ = 3 * prices.f ∧
  prices.p = prices.f + prices.g

/-- The theorem to be proved -/
theorem cost_of_grapes_and_pineapple (prices : FruitPrices) 
  (h : satisfiesConditions prices) : 
  2 * prices.g + prices.p = (15 * prices.g + 36) / 7 := by
  sorry

end cost_of_grapes_and_pineapple_l1860_186047


namespace train_speed_l1860_186025

/-- The speed of a train given its length, time to cross a moving person, and the person's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (person_speed : ℝ) :
  train_length = 500 →
  crossing_time = 29.997600191984642 →
  person_speed = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63) < 0.1 := by
  sorry

#check train_speed

end train_speed_l1860_186025


namespace eliminate_y_condition_l1860_186095

/-- Represents a system of two linear equations in two variables -/
structure LinearSystem (α : Type*) [Field α] :=
  (a₁ b₁ c₁ : α)
  (a₂ b₂ c₂ : α)

/-- Checks if y can be directly eliminated when subtracting the second equation from the first -/
def canEliminateY {α : Type*} [Field α] (sys : LinearSystem α) : Prop :=
  sys.b₁ + sys.b₂ = 0

/-- The specific linear system from the problem -/
def problemSystem (α : Type*) [Field α] (m n : α) : LinearSystem α :=
  { a₁ := 6, b₁ := m, c₁ := 3,
    a₂ := 2, b₂ := -n, c₂ := -6 }

theorem eliminate_y_condition (α : Type*) [Field α] (m n : α) :
  canEliminateY (problemSystem α m n) ↔ m + n = 0 :=
sorry

end eliminate_y_condition_l1860_186095


namespace triangle_area_is_correct_l1860_186057

/-- The area of a triangular region bounded by the coordinate axes and the line 3x + y = 9 -/
def triangleArea : ℝ := 13.5

/-- The equation of the bounding line -/
def boundingLine (x y : ℝ) : Prop := 3 * x + y = 9

theorem triangle_area_is_correct : 
  triangleArea = 13.5 := by sorry

end triangle_area_is_correct_l1860_186057


namespace arithmetic_sequence_sum_l1860_186001

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmeticSequence a →
  (a 2 = 0) →
  (S 3 + S 4 = 6) →
  (a 5 + a 6 = 21) :=
by
  sorry

end arithmetic_sequence_sum_l1860_186001


namespace increase_by_percentage_l1860_186027

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 200 →
  percentage = 25 →
  final = initial * (1 + percentage / 100) →
  final = 250 := by
  sorry

end increase_by_percentage_l1860_186027


namespace bakers_friend_cakes_l1860_186039

/-- Given that Baker made 155 cakes initially and now has 15 cakes remaining,
    prove that Baker's friend bought 140 cakes. -/
theorem bakers_friend_cakes :
  let initial_cakes : ℕ := 155
  let remaining_cakes : ℕ := 15
  let friend_bought : ℕ := initial_cakes - remaining_cakes
  friend_bought = 140 := by sorry

end bakers_friend_cakes_l1860_186039


namespace symmetry_yoz_proof_l1860_186019

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the yOz plane -/
def symmetryYOZ (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

/-- The original point -/
def originalPoint : Point3D :=
  ⟨1, -2, 3⟩

theorem symmetry_yoz_proof :
  symmetryYOZ originalPoint = Point3D.mk (-1) (-2) 3 := by
  sorry

end symmetry_yoz_proof_l1860_186019


namespace valid_reasoning_methods_l1860_186003

-- Define the set of reasoning methods
inductive ReasoningMethod
| Method1
| Method2
| Method3
| Method4

-- Define a predicate for valid analogical reasoning
def is_valid_analogical_reasoning (m : ReasoningMethod) : Prop :=
  m = ReasoningMethod.Method1

-- Define a predicate for valid inductive reasoning
def is_valid_inductive_reasoning (m : ReasoningMethod) : Prop :=
  m = ReasoningMethod.Method2 ∨ m = ReasoningMethod.Method4

-- Define a predicate for valid reasoning
def is_valid_reasoning (m : ReasoningMethod) : Prop :=
  is_valid_analogical_reasoning m ∨ is_valid_inductive_reasoning m

-- Theorem statement
theorem valid_reasoning_methods :
  {m : ReasoningMethod | is_valid_reasoning m} =
  {ReasoningMethod.Method1, ReasoningMethod.Method2, ReasoningMethod.Method4} :=
by sorry

end valid_reasoning_methods_l1860_186003


namespace heximal_binary_equality_l1860_186016

/-- Converts a heximal (base-6) number to decimal --/
def heximal_to_decimal (a b c d : ℕ) : ℕ :=
  a * 6^3 + b * 6^2 + c * 6^1 + d * 6^0

/-- Converts a binary number to decimal --/
def binary_to_decimal (a b c d e f g h : ℕ) : ℕ :=
  a * 2^7 + b * 2^6 + c * 2^5 + d * 2^4 + e * 2^3 + f * 2^2 + g * 2^1 + h * 2^0

/-- The theorem stating that k = 3 is the unique solution --/
theorem heximal_binary_equality :
  ∃! k : ℕ, k > 0 ∧ heximal_to_decimal 1 0 k 5 = binary_to_decimal 1 1 1 0 1 1 1 1 :=
by sorry

end heximal_binary_equality_l1860_186016


namespace gus_total_eggs_l1860_186054

/-- The number of eggs Gus ate for breakfast -/
def breakfast_eggs : ℕ := 2

/-- The number of eggs Gus ate for lunch -/
def lunch_eggs : ℕ := 3

/-- The number of eggs Gus ate for dinner -/
def dinner_eggs : ℕ := 1

/-- The total number of eggs Gus ate -/
def total_eggs : ℕ := breakfast_eggs + lunch_eggs + dinner_eggs

theorem gus_total_eggs : total_eggs = 6 := by
  sorry

end gus_total_eggs_l1860_186054


namespace yellow_red_difference_l1860_186059

/-- The number of houses Isabella has -/
structure Houses where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Conditions for Isabella's houses -/
def isabellaHouses (h : Houses) : Prop :=
  h.green = 3 * h.yellow ∧
  h.green = 90 ∧
  h.green + h.red = 160

/-- Theorem: Isabella has 40 fewer yellow houses than red houses -/
theorem yellow_red_difference (h : Houses) (hcond : isabellaHouses h) :
  h.red - h.yellow = 40 := by
  sorry

end yellow_red_difference_l1860_186059


namespace rosa_flowers_total_l1860_186071

theorem rosa_flowers_total (initial : ℝ) (gift : ℝ) (total : ℝ) 
    (h1 : initial = 67.5) 
    (h2 : gift = 90.75) 
    (h3 : total = initial + gift) : 
  total = 158.25 := by
sorry

end rosa_flowers_total_l1860_186071


namespace negation_implication_true_l1860_186090

theorem negation_implication_true (a b c : ℝ) : 
  ¬(a > b → a * c^2 > b * c^2) :=
by sorry

end negation_implication_true_l1860_186090


namespace swordtails_count_l1860_186089

/-- The number of Goldfish Layla has -/
def num_goldfish : ℕ := 2

/-- The amount of food each Goldfish gets (in teaspoons) -/
def goldfish_food : ℚ := 1

/-- The number of Guppies Layla has -/
def num_guppies : ℕ := 8

/-- The amount of food each Guppy gets (in teaspoons) -/
def guppy_food : ℚ := 1/2

/-- The amount of food each Swordtail gets (in teaspoons) -/
def swordtail_food : ℚ := 2

/-- The total amount of food given to all fish (in teaspoons) -/
def total_food : ℚ := 12

/-- The number of Swordtails Layla has -/
def num_swordtails : ℕ := 3

theorem swordtails_count : 
  (num_goldfish : ℚ) * goldfish_food + 
  (num_guppies : ℚ) * guppy_food + 
  (num_swordtails : ℚ) * swordtail_food = total_food :=
sorry

end swordtails_count_l1860_186089


namespace amy_and_noah_total_l1860_186038

/-- The number of books each person has -/
structure BookCounts where
  maddie : ℕ
  luisa : ℕ
  amy : ℕ
  noah : ℕ

/-- The conditions of the problem -/
def book_problem (bc : BookCounts) : Prop :=
  bc.maddie = 2^4 - 1 ∧
  bc.luisa = 18 ∧
  bc.amy + bc.luisa = bc.maddie + 9 ∧
  bc.noah = Int.sqrt (bc.amy^2) + 2 ∧
  (Int.sqrt (bc.amy^2))^2 = bc.amy^2

/-- The theorem to prove -/
theorem amy_and_noah_total (bc : BookCounts) :
  book_problem bc → bc.amy + bc.noah = 14 := by
  sorry

end amy_and_noah_total_l1860_186038


namespace perpendicular_lines_l1860_186075

theorem perpendicular_lines (b : ℝ) : 
  let v1 : Fin 2 → ℝ := ![4, -9]
  let v2 : Fin 2 → ℝ := ![b, 3]
  (∀ i : Fin 2, v1 i * v2 i = 0) → b = 27/4 := by
sorry

end perpendicular_lines_l1860_186075


namespace female_democrat_ratio_l1860_186081

theorem female_democrat_ratio (total_participants male_participants female_participants : ℕ)
  (female_democrats male_democrats : ℕ) :
  total_participants = 750 →
  total_participants = male_participants + female_participants →
  male_democrats = male_participants / 4 →
  female_democrats = 125 →
  male_democrats + female_democrats = total_participants / 3 →
  2 * female_democrats = female_participants :=
by
  sorry

end female_democrat_ratio_l1860_186081


namespace min_translation_for_even_function_l1860_186091

theorem min_translation_for_even_function (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.sin (3 * x + π / 4)) →
  m > 0 →
  (∀ x, Real.sin (3 * (x + m) + π / 4) = Real.sin (3 * (-x + m) + π / 4)) →
  m ≥ π / 12 :=
by sorry

end min_translation_for_even_function_l1860_186091


namespace cookies_left_l1860_186031

-- Define the number of cookies in a dozen
def cookies_per_dozen : ℕ := 12

-- Define the number of dozens John buys
def dozens_bought : ℕ := 2

-- Define the number of cookies John eats
def cookies_eaten : ℕ := 3

-- Theorem statement
theorem cookies_left : 
  dozens_bought * cookies_per_dozen - cookies_eaten = 21 := by
sorry

end cookies_left_l1860_186031


namespace quadratic_polynomial_property_l1860_186026

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Evaluation of a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℚ) : ℚ :=
  p.a * x^2 + p.b * x + p.c

/-- Condition for divisibility by (x - 2)(x + 2)(x - 9) -/
def isDivisibleByFactors (p : QuadraticPolynomial) : Prop :=
  (p.eval 2)^3 = 2 ∧ (p.eval (-2))^3 = -2 ∧ (p.eval 9)^3 = 9

theorem quadratic_polynomial_property (p : QuadraticPolynomial) 
  (h : isDivisibleByFactors p) : p.eval 14 = -230/11 := by
  sorry

end quadratic_polynomial_property_l1860_186026


namespace rectangle_perimeter_l1860_186012

/-- The perimeter of a rectangular field with length 7/5 times its width and width of 75 meters is 360 meters. -/
theorem rectangle_perimeter (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width = 75 →
  length = (7/5) * width →
  perimeter = 2 * (length + width) →
  perimeter = 360 := by
  sorry

end rectangle_perimeter_l1860_186012


namespace geometric_sequence_ratio_l1860_186083

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The given arithmetic sequence condition -/
def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop :=
  2 * ((1/2) * a 3) = 3 * a 1 + 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  arithmetic_sequence_condition a →
  (a 2014 - a 2015) / (a 2016 - a 2017) = 1/9 :=
by sorry

end geometric_sequence_ratio_l1860_186083


namespace percentage_students_owning_only_cats_l1860_186060

/-- Proves that the percentage of students owning only cats is 10% -/
theorem percentage_students_owning_only_cats
  (total_students : ℕ)
  (students_with_dogs : ℕ)
  (students_with_cats : ℕ)
  (students_with_both : ℕ)
  (h1 : total_students = 500)
  (h2 : students_with_dogs = 200)
  (h3 : students_with_cats = 100)
  (h4 : students_with_both = 50) :
  (students_with_cats - students_with_both) / total_students = 1 / 10 := by
  sorry


end percentage_students_owning_only_cats_l1860_186060


namespace weight_replacement_l1860_186045

theorem weight_replacement (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) : 
  initial_count = 5 → 
  replaced_weight = 65 → 
  avg_increase = 1.5 → 
  (initial_count : ℝ) * avg_increase + replaced_weight = 72.5 := by
  sorry

end weight_replacement_l1860_186045


namespace arcade_tickets_difference_l1860_186073

theorem arcade_tickets_difference (tickets_won tickets_left : ℕ) 
  (h1 : tickets_won = 48) 
  (h2 : tickets_left = 32) : 
  tickets_won - tickets_left = 16 := by
  sorry

end arcade_tickets_difference_l1860_186073


namespace chairs_built_in_ten_days_l1860_186049

/-- Calculates the number of chairs a worker can build in a given number of days -/
def chairs_built (shift_hours : ℕ) (build_time : ℕ) (days : ℕ) : ℕ :=
  let chairs_per_shift := min 1 (shift_hours / build_time)
  chairs_per_shift * days

/-- Theorem stating that a worker who works 8-hour shifts and takes 5 hours to build 1 chair
    can build 10 chairs in 10 days -/
theorem chairs_built_in_ten_days :
  chairs_built 8 5 10 = 10 := by
  sorry

end chairs_built_in_ten_days_l1860_186049


namespace lemonade_water_calculation_l1860_186035

/-- The amount of water needed to make lemonade with a given ratio and total volume -/
def water_needed (water_ratio : ℚ) (juice_ratio : ℚ) (total_gallons : ℚ) (liters_per_gallon : ℚ) : ℚ :=
  (water_ratio / (water_ratio + juice_ratio)) * (total_gallons * liters_per_gallon)

/-- Theorem stating the amount of water needed for the lemonade recipe -/
theorem lemonade_water_calculation :
  let water_ratio : ℚ := 8
  let juice_ratio : ℚ := 2
  let total_gallons : ℚ := 2
  let liters_per_gallon : ℚ := 3785/1000
  water_needed water_ratio juice_ratio total_gallons liters_per_gallon = 6056/1000 := by
  sorry

end lemonade_water_calculation_l1860_186035


namespace abs_plus_power_minus_sqrt_inequality_system_solution_l1860_186030

-- Part 1
theorem abs_plus_power_minus_sqrt : |-2| + (1 + Real.sqrt 3)^0 - Real.sqrt 9 = 0 := by
  sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (2 * x + 1 > 3 * (x - 1) ∧ x + (x - 1) / 3 < 1) ↔ x < 1 := by
  sorry

end abs_plus_power_minus_sqrt_inequality_system_solution_l1860_186030


namespace min_positive_temperatures_l1860_186050

theorem min_positive_temperatures (n : ℕ) (pos_products neg_products : ℕ) : 
  n * (n - 1) = pos_products + neg_products →
  pos_products = 68 →
  neg_products = 64 →
  ∃ k : ℕ, k ≤ n ∧ k * (k - 1) = pos_products ∧ k ≥ 4 ∧ 
  ∀ m : ℕ, m < k → m * (m - 1) ≠ pos_products :=
by sorry

end min_positive_temperatures_l1860_186050


namespace max_value_expression_l1860_186017

def S : Set ℕ := {0, 1, 2, 3}

theorem max_value_expression (a b c d : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) :
  (∀ x y z w : ℕ, x ∈ S → y ∈ S → z ∈ S → w ∈ S →
    z * (x^y + 1) - w ≤ c * (a^b + 1) - d) →
  c * (a^b + 1) - d = 30 :=
sorry

end max_value_expression_l1860_186017


namespace ball_probabilities_l1860_186033

/-- The total number of balls in the box -/
def total_balls : ℕ := 12

/-- The number of red balls in the box -/
def red_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 4

/-- The number of white balls in the box -/
def white_balls : ℕ := 2

/-- The number of green balls in the box -/
def green_balls : ℕ := 1

/-- The probability of drawing a red or black ball -/
def prob_red_or_black : ℚ := (red_balls + black_balls) / total_balls

/-- The probability of drawing a red, black, or white ball -/
def prob_red_black_or_white : ℚ := (red_balls + black_balls + white_balls) / total_balls

theorem ball_probabilities :
  prob_red_or_black = 3/4 ∧ prob_red_black_or_white = 11/12 :=
by sorry

end ball_probabilities_l1860_186033


namespace factorization_1_factorization_2_l1860_186023

-- First expression
theorem factorization_1 (a b : ℝ) :
  -6 * a * b + 3 * a^2 + 3 * b^2 = 3 * (a - b)^2 := by sorry

-- Second expression
theorem factorization_2 (x y m : ℝ) :
  y^2 * (2 - m) + x^2 * (m - 2) = (m - 2) * (x + y) * (x - y) := by sorry

end factorization_1_factorization_2_l1860_186023


namespace cyclic_inequality_l1860_186065

def cyclic_system (n : ℕ) (p q : ℝ) (x y z : ℝ) : Prop :=
  y = x^n + p*x + q ∧ z = y^n + p*y + q ∧ x = z^n + p*z + q

theorem cyclic_inequality (n : ℕ) (p q : ℝ) (x y z : ℝ) 
  (h_sys : cyclic_system n p q x y z) 
  (h_n : n = 2 ∨ n = 2010) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) :
  x^2 * y + y^2 * z + z^2 * x ≥ x^2 * z + y^2 * x + z^2 * y := by
  sorry

end cyclic_inequality_l1860_186065


namespace scatter_plot_always_possible_l1860_186088

/-- Represents statistical data for two variables -/
structure StatisticalData where
  variable1 : List ℝ
  variable2 : List ℝ
  length_eq : variable1.length = variable2.length

/-- Represents a scatter plot -/
structure ScatterPlot where
  points : List (ℝ × ℝ)

/-- Given statistical data for two variables, it is always possible to create a scatter plot -/
theorem scatter_plot_always_possible (data : StatisticalData) : 
  ∃ (plot : ScatterPlot), true := by sorry

end scatter_plot_always_possible_l1860_186088


namespace min_b_value_l1860_186015

/-- Given positive integers x, y, z in ratio 3:4:7 and y = 15b - 5, 
    prove the minimum positive integer b is 3 -/
theorem min_b_value (x y z b : ℕ+) : 
  (∃ k : ℕ+, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  y = 15 * b - 5 →
  (∀ b' : ℕ+, b' < b → 
    ¬∃ x' y' z' : ℕ+, (∃ k : ℕ+, x' = 3 * k ∧ y' = 4 * k ∧ z' = 7 * k) ∧ 
    y' = 15 * b' - 5) →
  b = 3 := by
  sorry

#check min_b_value

end min_b_value_l1860_186015


namespace logarithm_sum_simplification_l1860_186042

theorem logarithm_sum_simplification : 
  1 / (Real.log 3 / Real.log 20 + 1) + 
  1 / (Real.log 5 / Real.log 12 + 1) + 
  1 / (Real.log 7 / Real.log 8 + 1) = 2 := by sorry

end logarithm_sum_simplification_l1860_186042


namespace tim_sweets_multiple_of_four_l1860_186021

/-- The number of grape-flavored sweets Peter has -/
def peter_sweets : ℕ := 44

/-- The largest possible number of sweets in each tray without remainder -/
def tray_size : ℕ := 4

/-- The number of orange-flavored sweets Tim has -/
def tim_sweets : ℕ := sorry

theorem tim_sweets_multiple_of_four :
  ∃ k : ℕ, tim_sweets = k * tray_size ∧ peter_sweets % tray_size = 0 :=
by sorry

end tim_sweets_multiple_of_four_l1860_186021


namespace expected_remaining_balls_l1860_186074

/-- Represents the number of red balls initially in the bag -/
def redBalls : ℕ := 100

/-- Represents the number of blue balls initially in the bag -/
def blueBalls : ℕ := 100

/-- Represents the total number of balls initially in the bag -/
def totalBalls : ℕ := redBalls + blueBalls

/-- Represents the process of drawing balls without replacement until all red balls are drawn -/
def drawUntilAllRed (red blue : ℕ) : ℝ := sorry

/-- Theorem stating the expected number of remaining balls after drawing all red balls -/
theorem expected_remaining_balls :
  drawUntilAllRed redBalls blueBalls = blueBalls / (totalBalls : ℝ) := by sorry

end expected_remaining_balls_l1860_186074


namespace smallest_tree_height_l1860_186052

/-- Proves that the height of the smallest tree is 12 feet given the conditions of the problem -/
theorem smallest_tree_height (tallest middle smallest : ℝ) : 
  tallest = 108 ∧ 
  middle = tallest / 2 - 6 ∧ 
  smallest = middle / 4 → 
  smallest = 12 := by
  sorry

end smallest_tree_height_l1860_186052


namespace solution_set_equality_l1860_186007

-- Define the inequality function
def f (x : ℝ) := x^2 + 2*x - 3

-- State the theorem
theorem solution_set_equality :
  {x : ℝ | f x < 0} = Set.Ioo (-3 : ℝ) (1 : ℝ) :=
sorry

end solution_set_equality_l1860_186007


namespace range_of_a_l1860_186008

/-- The function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The set A = {x | f(x) ≤ 0} -/
def A (a b : ℝ) : Set ℝ := {x | f a b x ≤ 0}

/-- The set B = {x | f(f(x)) ≤ 5/4} -/
def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) ≤ 5/4}

/-- Theorem: Given f(x) = x^2 + ax + b, A = {x | f(x) ≤ 0}, B = {x | f(f(x)) ≤ 5/4},
    and A = B ≠ ∅, the range of a is [√5, 5] -/
theorem range_of_a (a b : ℝ) : 
  A a b = B a b ∧ A a b ≠ ∅ → a ∈ Set.Icc (Real.sqrt 5) 5 := by
  sorry

end range_of_a_l1860_186008


namespace min_value_cyclic_fraction_l1860_186051

theorem min_value_cyclic_fraction (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / b + b / c + c / d + d / a ≥ 4 ∧
  (a / b + b / c + c / d + d / a = 4 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end min_value_cyclic_fraction_l1860_186051


namespace subtract_from_percentage_l1860_186058

theorem subtract_from_percentage (number : ℝ) (percentage : ℝ) (subtrahend : ℝ) : 
  number = 200 → percentage = 40 → subtrahend = 30 →
  percentage / 100 * number - subtrahend = 50 := by
sorry

end subtract_from_percentage_l1860_186058


namespace price_change_calculation_l1860_186041

theorem price_change_calculation (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_decrease := 0.8 * initial_price
  let final_price := 1.04 * initial_price
  ∃ x : ℝ, price_after_decrease * (1 + x / 100) = final_price ∧ x = 30 :=
by sorry

end price_change_calculation_l1860_186041


namespace power_seven_mod_nine_l1860_186040

theorem power_seven_mod_nine : 7^123 % 9 = 1 := by
  sorry

end power_seven_mod_nine_l1860_186040


namespace regression_line_prediction_l1860_186063

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y-value for a given x on the regression line -/
def RegressionLine.predict (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

theorem regression_line_prediction 
  (slope : ℝ) 
  (center_x center_y : ℝ) 
  (h_slope : slope = 1.23) 
  (h_center : center_y = slope * center_x + intercept) 
  (h_center_x : center_x = 4) 
  (h_center_y : center_y = 5) :
  let line : RegressionLine := {
    slope := slope,
    intercept := center_y - slope * center_x
  }
  line.predict 2 = 2.54 := by
  sorry

end regression_line_prediction_l1860_186063


namespace diamond_property_false_l1860_186080

def diamond (x y : ℝ) : ℝ := 2 * |x - y| + 1

theorem diamond_property_false :
  ¬ ∀ x y : ℝ, 3 * (diamond x y) = 3 * (diamond (2*x) (2*y)) :=
sorry

end diamond_property_false_l1860_186080


namespace stock_z_shares_l1860_186020

/-- Represents the number of shares for each stock --/
structure ShareDistribution where
  v : ℕ
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the range of shares in a distribution --/
def calculateRange (shares : ShareDistribution) : ℕ :=
  max shares.v (max shares.w (max shares.x (max shares.y shares.z))) -
  min shares.v (min shares.w (min shares.x (min shares.y shares.z)))

/-- Theorem stating that the number of shares of stock z is 47 --/
theorem stock_z_shares : ∃ (initial : ShareDistribution),
  initial.v = 68 ∧
  initial.w = 112 ∧
  initial.x = 56 ∧
  initial.y = 94 ∧
  let final : ShareDistribution := {
    v := initial.v,
    w := initial.w,
    x := initial.x - 20,
    y := initial.y + 23,
    z := initial.z
  }
  calculateRange final - calculateRange initial = 14 →
  initial.z = 47 := by
  sorry

end stock_z_shares_l1860_186020


namespace two_digit_integers_count_l1860_186079

def available_digits : Finset Nat := {1, 2, 3, 8, 9}

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

def count_two_digit_integers (digits : Finset Nat) : Nat :=
  (digits.filter (λ d ↦ d ≤ 9)).card * (digits.filter (λ d ↦ d ≤ 9)).card

theorem two_digit_integers_count :
  count_two_digit_integers available_digits = 25 := by sorry

end two_digit_integers_count_l1860_186079


namespace average_calls_is_40_l1860_186082

/-- Represents the number of calls answered each day for a week --/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the average number of calls per day --/
def averageCalls (w : WeekCalls) : ℚ :=
  (w.monday + w.tuesday + w.wednesday + w.thursday + w.friday) / 5

/-- Theorem stating that for the given week of calls, the average is 40 --/
theorem average_calls_is_40 (w : WeekCalls) 
    (h1 : w.monday = 35)
    (h2 : w.tuesday = 46)
    (h3 : w.wednesday = 27)
    (h4 : w.thursday = 61)
    (h5 : w.friday = 31) :
    averageCalls w = 40 := by
  sorry

end average_calls_is_40_l1860_186082


namespace arithmetic_calculation_l1860_186053

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 3 + (5 - 3)^2 = 65 := by
  sorry

end arithmetic_calculation_l1860_186053


namespace factorization_cube_minus_linear_l1860_186046

theorem factorization_cube_minus_linear (a b : ℝ) : a^3 * b - a * b = a * b * (a + 1) * (a - 1) := by
  sorry

end factorization_cube_minus_linear_l1860_186046


namespace line_passes_through_other_lattice_points_l1860_186032

theorem line_passes_through_other_lattice_points :
  ∃ (x y : ℤ), x ≠ 0 ∧ x ≠ 5 ∧ y ≠ 0 ∧ y ≠ 3 ∧ 5 * y = 3 * x := by
  sorry

end line_passes_through_other_lattice_points_l1860_186032


namespace smallest_divisible_by_3_5_7_13_greater_than_1000_l1860_186086

theorem smallest_divisible_by_3_5_7_13_greater_than_1000 : ∃ n : ℕ,
  n > 1000 ∧
  n % 3 = 0 ∧
  n % 5 = 0 ∧
  n % 7 = 0 ∧
  n % 13 = 0 ∧
  (∀ m : ℕ, m > 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 ∧ m % 13 = 0 → m ≥ n) ∧
  n = 1365 :=
by
  sorry

end smallest_divisible_by_3_5_7_13_greater_than_1000_l1860_186086


namespace solution_set_when_m_is_neg_three_m_range_when_subset_condition_holds_l1860_186024

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |2*x + m|

-- Part I
theorem solution_set_when_m_is_neg_three :
  {x : ℝ | f x (-3) ≤ 6} = {x : ℝ | -4/3 ≤ x ∧ x ≤ 8/3} := by sorry

-- Part II
theorem m_range_when_subset_condition_holds :
  (∀ x ∈ Set.Icc (-1 : ℝ) (1/2 : ℝ), f x m ≤ |2*x - 4|) →
  m ∈ Set.Icc (-5/2 : ℝ) (1/2 : ℝ) := by sorry

end solution_set_when_m_is_neg_three_m_range_when_subset_condition_holds_l1860_186024


namespace tuesday_rainfall_l1860_186093

/-- Rainfall recorded over three days -/
def total_rainfall : ℝ := 0.67

/-- Rainfall recorded on Monday -/
def monday_rainfall : ℝ := 0.17

/-- Rainfall recorded on Wednesday -/
def wednesday_rainfall : ℝ := 0.08

/-- Theorem stating that the rainfall on Tuesday is 0.42 cm -/
theorem tuesday_rainfall : 
  total_rainfall - (monday_rainfall + wednesday_rainfall) = 0.42 := by sorry

end tuesday_rainfall_l1860_186093


namespace shirley_eggs_theorem_l1860_186002

/-- The number of eggs Shirley started with -/
def initial_eggs : ℕ := 98

/-- The number of eggs Shirley bought -/
def bought_eggs : ℕ := 8

/-- The total number of eggs Shirley ended with -/
def final_eggs : ℕ := 106

/-- Theorem stating that the initial number of eggs plus the bought eggs equals the final number of eggs -/
theorem shirley_eggs_theorem : initial_eggs + bought_eggs = final_eggs := by
  sorry

end shirley_eggs_theorem_l1860_186002


namespace sequence_value_l1860_186000

theorem sequence_value (a : ℕ → ℕ) :
  a 1 = 0 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) →
  a 2012 = 2011 * 2012 :=
by
  sorry

end sequence_value_l1860_186000


namespace sum_xyz_equals_2014_l1860_186004

theorem sum_xyz_equals_2014 (x y z : ℝ) : 
  Real.sqrt (x - 3) + Real.sqrt (3 - x) + abs (x - y + 2010) + z^2 + 4*z + 4 = 0 → 
  x + y + z = 2014 := by
  sorry

end sum_xyz_equals_2014_l1860_186004


namespace g_behavior_at_infinity_l1860_186069

def g (x : ℝ) : ℝ := -3 * x^3 - 2 * x^2 + x + 10

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) :=
by sorry

end g_behavior_at_infinity_l1860_186069


namespace triangle_property_l1860_186010

/-- Given a triangle ABC with sides a, b, and c satisfying the equation
    a^2 + b^2 + c^2 + 50 = 6a + 8b + 10c, prove that it is a right-angled
    triangle with area 6. -/
theorem triangle_property (a b c : ℝ) (h : a^2 + b^2 + c^2 + 50 = 6*a + 8*b + 10*c) :
  (a = 3 ∧ b = 4 ∧ c = 5) ∧ a^2 + b^2 = c^2 ∧ (1/2) * a * b = 6 := by
  sorry

end triangle_property_l1860_186010


namespace finite_divisor_property_l1860_186028

/-- A number is a finite decimal if it can be expressed as a/b where b is of the form 2^u * 5^v -/
def IsFiniteDecimal (q : ℚ) : Prop :=
  ∃ (a b : ℕ) (u v : ℕ), q = a / b ∧ b = 2^u * 5^v

/-- A natural number n has the property that all its divisors result in finite decimals -/
def HasFiniteDivisors (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k → k < n → IsFiniteDecimal (n / k)

/-- The theorem stating that only 2, 3, and 6 have the finite divisor property -/
theorem finite_divisor_property :
  ∀ n : ℕ, HasFiniteDivisors n ↔ n = 2 ∨ n = 3 ∨ n = 6 :=
sorry

end finite_divisor_property_l1860_186028


namespace merchant_markup_l1860_186085

theorem merchant_markup (C : ℝ) (x : ℝ) : 
  (C * (1 + x / 100) * 0.7 = C * 1.225) → x = 75 := by
  sorry

end merchant_markup_l1860_186085


namespace problem_solution_l1860_186018

theorem problem_solution (a b c d e : ℝ) 
  (h1 : |2 + a| + |b - 3| = 0)
  (h2 : c ≠ 0)
  (h3 : 1 / c = -d)
  (h4 : e = -5) :
  -a^b + 1/c - e + d = 13 := by
  sorry

end problem_solution_l1860_186018


namespace race_track_distance_squared_l1860_186098

theorem race_track_distance_squared (inner_radius outer_radius : ℝ) 
  (h_inner : inner_radius = 11) 
  (h_outer : outer_radius = 12) 
  (separation_angle : ℝ) 
  (h_angle : separation_angle = 30 * π / 180) : 
  (inner_radius^2 + outer_radius^2 - 2 * inner_radius * outer_radius * Real.cos separation_angle) 
  = 265 - 132 * Real.sqrt 3 := by
  sorry

end race_track_distance_squared_l1860_186098


namespace compare_negative_fractions_l1860_186005

theorem compare_negative_fractions : -3/4 > -4/5 := by
  sorry

end compare_negative_fractions_l1860_186005


namespace initial_machines_count_l1860_186068

/-- Given a number of machines working at a constant rate, this theorem proves
    the number of machines initially working based on their production output. -/
theorem initial_machines_count (x : ℝ) (N : ℕ) : 
  (N : ℝ) * x / 4 = 20 * 3 * x / 6 → N = 10 := by
  sorry

end initial_machines_count_l1860_186068


namespace sqrt2_minus1_cf_infinite_sqrt3_minus1_cf_infinite_sqrt2_minus1_4th_convergent_error_sqrt3_minus1_4th_convergent_error_l1860_186062

-- Define the continued fraction representation for √2 - 1
def sqrt2_minus1_cf (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | n+1 => 1 / (2 + sqrt2_minus1_cf n)

-- Define the continued fraction representation for √3 - 1
def sqrt3_minus1_cf (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n+2 => 1 / (1 + 1 / (2 + sqrt3_minus1_cf n))

-- Define the fourth convergent of √2 - 1
def sqrt2_minus1_4th_convergent : ℚ := 12 / 29

-- Define the fourth convergent of √3 - 1
def sqrt3_minus1_4th_convergent : ℚ := 8 / 11

theorem sqrt2_minus1_cf_infinite :
  ∀ n : ℕ, sqrt2_minus1_cf n ≠ sqrt2_minus1_cf (n+1) :=
sorry

theorem sqrt3_minus1_cf_infinite :
  ∀ n : ℕ, sqrt3_minus1_cf n ≠ sqrt3_minus1_cf (n+1) :=
sorry

theorem sqrt2_minus1_4th_convergent_error :
  |Real.sqrt 2 - 1 - sqrt2_minus1_4th_convergent| < 1 / 2000 :=
sorry

theorem sqrt3_minus1_4th_convergent_error :
  |Real.sqrt 3 - 1 - sqrt3_minus1_4th_convergent| < 1 / 209 :=
sorry

end sqrt2_minus1_cf_infinite_sqrt3_minus1_cf_infinite_sqrt2_minus1_4th_convergent_error_sqrt3_minus1_4th_convergent_error_l1860_186062


namespace solve_equation_l1860_186094

theorem solve_equation (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 := by
  sorry

end solve_equation_l1860_186094


namespace analogical_reasoning_is_specific_to_specific_l1860_186077

-- Define the types of reasoning
inductive ReasoningType
  | Reasonable
  | Inductive
  | Deductive
  | Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
  | GeneralToSpecific
  | SpecificToGeneral
  | SpecificToSpecific

-- Define the property of a reasoning type
def reasoning_direction (r : ReasoningType) : ReasoningDirection :=
  match r with
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific
  | _ => ReasoningDirection.GeneralToSpecific -- Default for other types, not relevant for this problem

-- Theorem statement
theorem analogical_reasoning_is_specific_to_specific :
  reasoning_direction ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific :=
by sorry

end analogical_reasoning_is_specific_to_specific_l1860_186077


namespace intersection_A_B_l1860_186099

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 1}
def B : Set ℝ := {-2, -1, 0, 1}

theorem intersection_A_B : A ∩ B = {-2, -1, 0} := by sorry

end intersection_A_B_l1860_186099


namespace age_ratio_theorem_l1860_186064

/-- Represents the ages of Tom and Jerry -/
structure Ages where
  tom : ℕ
  jerry : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  (ages.tom - 3 = 4 * (ages.jerry - 3)) ∧ 
  (ages.tom - 8 = 5 * (ages.jerry - 8))

/-- The future age ratio condition -/
def future_ratio (ages : Ages) (years : ℕ) : Prop :=
  3 * (ages.jerry + years) = ages.tom + years

/-- The main theorem to prove -/
theorem age_ratio_theorem : 
  ∃ (ages : Ages), age_conditions ages → future_ratio ages 7 :=
sorry

end age_ratio_theorem_l1860_186064


namespace eagle_pairs_count_l1860_186078

/-- The number of nesting pairs of bald eagles in 1963 -/
def pairs_1963 : ℕ := 417

/-- The increase in nesting pairs since 1963 -/
def increase : ℕ := 6649

/-- The current number of nesting pairs of bald eagles in the lower 48 states -/
def current_pairs : ℕ := pairs_1963 + increase

theorem eagle_pairs_count : current_pairs = 7066 := by
  sorry

end eagle_pairs_count_l1860_186078


namespace min_value_theorem_l1860_186044

theorem min_value_theorem (a b : ℝ) (h1 : 2*a + 3*b = 6) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2*x + 3*y = 6 → 2/x + 3/y ≥ 25/6) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + 3*y = 6 ∧ 2/x + 3/y = 25/6) :=
by sorry

end min_value_theorem_l1860_186044


namespace arithmetic_sequence_sum_l1860_186055

def isArithmeticSequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  isArithmeticSequence a →
  isArithmeticSequence b →
  a 1 = 25 →
  b 1 = 125 →
  a 2 + b 2 = 150 →
  (a + b) 2006 = 150 := by
  sorry

end arithmetic_sequence_sum_l1860_186055


namespace point_coordinates_l1860_186006

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point M in the second quadrant, 5 units away from the x-axis
    and 3 units away from the y-axis, has coordinates (-3, 5) -/
theorem point_coordinates (M : Point) 
  (h1 : SecondQuadrant M) 
  (h2 : DistanceToXAxis M = 5) 
  (h3 : DistanceToYAxis M = 3) : 
  M.x = -3 ∧ M.y = 5 := by
  sorry

end point_coordinates_l1860_186006


namespace sequence_inequality_l1860_186076

theorem sequence_inequality (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (a 1 + a (n + 1)) / a n > 1 + 1 / n :=
sorry

end sequence_inequality_l1860_186076


namespace symmetry_about_x_axis_and_origin_l1860_186043

/-- Given point A (2, -3) and point B symmetrical to A about the x-axis,
    prove that the coordinates of point C, which is symmetrical to point B about the origin,
    are (-2, -3). -/
theorem symmetry_about_x_axis_and_origin :
  let A : ℝ × ℝ := (2, -3)
  let B : ℝ × ℝ := (A.1, -A.2)  -- B is symmetrical to A about the x-axis
  let C : ℝ × ℝ := (-B.1, -B.2) -- C is symmetrical to B about the origin
  C = (-2, -3) := by sorry

end symmetry_about_x_axis_and_origin_l1860_186043


namespace sunzi_wood_measurement_l1860_186014

/-- Represents the problem from "The Mathematical Classic of Sunzi" --/
theorem sunzi_wood_measurement (x y : ℝ) :
  y = x + 4.5 ∧ 0.5 * y = x - 1 →
  (y - x = 4.5 ∧ y / 2 - x = 1) :=
by sorry

end sunzi_wood_measurement_l1860_186014


namespace max_sum_with_constraint_l1860_186013

theorem max_sum_with_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 2) :
  a + b ≤ 3/2 + Real.sqrt 2 := by
  sorry

end max_sum_with_constraint_l1860_186013


namespace quadratic_equation_solution_l1860_186029

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 6*x + 8 = 0 → x ≠ 4 → x = 2 := by
  sorry

end quadratic_equation_solution_l1860_186029


namespace three_pi_irrational_l1860_186056

/-- π is an irrational number -/
axiom pi_irrational : Irrational Real.pi

/-- The product of an irrational number and a non-zero rational number is irrational -/
axiom irrational_mul_rational {x : ℝ} (hx : Irrational x) {q : ℚ} (hq : q ≠ 0) :
  Irrational (x * ↑q)

/-- 3π is an irrational number -/
theorem three_pi_irrational : Irrational (3 * Real.pi) := by sorry

end three_pi_irrational_l1860_186056


namespace inequality_solution_sets_l1860_186034

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := a * x^2 + (1 - a) * x - 1 > 0

-- Define the solution set for a = 2
def solution_set_a2 : Set ℝ := {x | x < -1/2 ∨ x > 1}

-- Define the solution set for a > -1
def solution_set_a_gt_neg1 (a : ℝ) : Set ℝ :=
  if a = 0 then
    {x | x > 1}
  else if a > 0 then
    {x | x < -1/a ∨ x > 1}
  else
    {x | 1 < x ∧ x < -1/a}

theorem inequality_solution_sets :
  (∀ x, x ∈ solution_set_a2 ↔ inequality 2 x) ∧
  (∀ a, a > -1 → ∀ x, x ∈ solution_set_a_gt_neg1 a ↔ inequality a x) :=
sorry

end inequality_solution_sets_l1860_186034


namespace arithmetic_sequence_from_equation_l1860_186022

theorem arithmetic_sequence_from_equation (a b c : ℝ) :
  (2*b - a)^2 + (2*b - c)^2 = 2*(2*b^2 - a*c) →
  b = (a + c) / 2 :=
by sorry

end arithmetic_sequence_from_equation_l1860_186022


namespace circle_perimeter_special_radius_l1860_186097

/-- The perimeter of a circle with radius 4 / π cm is 8 cm. -/
theorem circle_perimeter_special_radius :
  let r : ℝ := 4 / Real.pi
  2 * Real.pi * r = 8 := by sorry

end circle_perimeter_special_radius_l1860_186097


namespace edward_remaining_money_l1860_186009

/-- Given Edward's initial amount and the amount he spent, calculate how much he has left. -/
theorem edward_remaining_money (initial : ℕ) (spent : ℕ) (remaining : ℕ) 
  (h1 : initial = 19) 
  (h2 : spent = 13) 
  (h3 : remaining = initial - spent) : 
  remaining = 6 := by
  sorry

end edward_remaining_money_l1860_186009


namespace sum_equals_three_fourths_l1860_186066

theorem sum_equals_three_fourths : 
  let original_sum := (1/3 : ℚ) + 1/6 + 1/9 + 1/12 + 1/15 + 1/18 + 1/21
  let removed_terms := (1/12 : ℚ) + 1/21
  original_sum - removed_terms = 3/4 := by
sorry

end sum_equals_three_fourths_l1860_186066


namespace harmonic_point_3_m_harmonic_point_hyperbola_l1860_186011

-- Definition of a harmonic point
def is_harmonic_point (x y t : ℝ) : Prop :=
  x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y

-- Theorem for part 1
theorem harmonic_point_3_m (m : ℝ) :
  is_harmonic_point 3 m (3^2 - 4*m) → m = -7 :=
sorry

-- Theorem for part 2
theorem harmonic_point_hyperbola (k : ℝ) :
  (∃ x : ℝ, -3 < x ∧ x < -1 ∧ is_harmonic_point x (k/x) (x^2 - 4*(k/x))) →
  3 < k ∧ k < 4 :=
sorry

end harmonic_point_3_m_harmonic_point_hyperbola_l1860_186011


namespace original_number_problem_l1860_186092

theorem original_number_problem (x : ℚ) : 2 * x + 5 = x / 2 + 20 → x = 10 := by
  sorry

end original_number_problem_l1860_186092


namespace toy_average_price_l1860_186048

theorem toy_average_price (n : ℕ) (dhoni_avg : ℚ) (david_price : ℚ) : 
  n = 5 → dhoni_avg = 10 → david_price = 16 → 
  (n * dhoni_avg + david_price) / (n + 1) = 11 := by sorry

end toy_average_price_l1860_186048


namespace b_2023_value_l1860_186070

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (b : ℕ → ℝ) : Prop :=
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

theorem b_2023_value (b : ℕ → ℝ) :
  RecurrenceSequence b →
  b 1 = 2 + Real.sqrt 5 →
  b 2010 = 12 + Real.sqrt 5 →
  b 2023 = (4 + 10 * Real.sqrt 5) / 3 := by
  sorry

end b_2023_value_l1860_186070


namespace ratio_problem_l1860_186084

/-- Given two numbers with a 20:1 ratio where the first number is 200, 
    the second number is 10. -/
theorem ratio_problem (a b : ℝ) : 
  (a / b = 20) → (a = 200) → (b = 10) := by
  sorry

end ratio_problem_l1860_186084


namespace no_multiple_of_five_l1860_186061

theorem no_multiple_of_five (n : ℕ) : 
  2 ≤ n → n ≤ 100 → ¬(5 ∣ (2 + 5*n + n^2 + 5*n^3 + 2*n^4)) :=
by sorry

end no_multiple_of_five_l1860_186061
