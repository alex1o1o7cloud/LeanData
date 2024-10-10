import Mathlib

namespace complex_fourth_power_equality_implies_ratio_one_l2846_284649

theorem complex_fourth_power_equality_implies_ratio_one 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4 → b / a = 1 := by
  sorry

end complex_fourth_power_equality_implies_ratio_one_l2846_284649


namespace third_player_win_probability_probability_third_player_wins_l2846_284647

/-- The probability of winning for the third player in a four-player 
    coin flipping game where players take turns and the first to flip 
    heads wins. -/
theorem third_player_win_probability : ℝ :=
  2 / 63

/-- The game ends when a player flips heads -/
axiom game_ends_on_heads : Prop

/-- There are four players taking turns -/
axiom four_players : Prop

/-- Players take turns in order -/
axiom turns_in_order : Prop

/-- Each flip has a 1/2 probability of heads -/
axiom fair_coin : Prop

theorem probability_third_player_wins : 
  game_ends_on_heads → four_players → turns_in_order → fair_coin →
  third_player_win_probability = 2 / 63 :=
sorry

end third_player_win_probability_probability_third_player_wins_l2846_284647


namespace total_fruits_is_137_l2846_284645

/-- The number of fruits picked by George, Amelia, and Olivia -/
def total_fruits (george_oranges : ℕ) (amelia_apples : ℕ) (amelia_orange_diff : ℕ) 
  (george_apple_diff : ℕ) (olivia_time : ℕ) (olivia_orange_rate : ℕ) (olivia_apple_rate : ℕ) 
  (olivia_time_unit : ℕ) : ℕ :=
  let george_apples := amelia_apples + george_apple_diff
  let amelia_oranges := george_oranges - amelia_orange_diff
  let olivia_cycles := olivia_time / olivia_time_unit
  let olivia_oranges := olivia_orange_rate * olivia_cycles
  let olivia_apples := olivia_apple_rate * olivia_cycles
  george_oranges + george_apples + amelia_oranges + amelia_apples + olivia_oranges + olivia_apples

/-- Theorem stating that the total number of fruits picked is 137 -/
theorem total_fruits_is_137 : 
  total_fruits 45 15 18 5 30 3 2 5 = 137 := by
  sorry


end total_fruits_is_137_l2846_284645


namespace first_agency_less_expensive_l2846_284661

/-- The number of miles at which the first agency becomes less expensive than the second -/
def miles_threshold : ℝ := 25

/-- The daily rate for the first agency -/
def daily_rate_1 : ℝ := 20.25

/-- The per-mile rate for the first agency -/
def mile_rate_1 : ℝ := 0.14

/-- The daily rate for the second agency -/
def daily_rate_2 : ℝ := 18.25

/-- The per-mile rate for the second agency -/
def mile_rate_2 : ℝ := 0.22

/-- Theorem stating that the first agency is less expensive when miles driven exceed the threshold -/
theorem first_agency_less_expensive (miles : ℝ) (days : ℝ) 
  (h : miles > miles_threshold) : 
  daily_rate_1 * days + mile_rate_1 * miles < daily_rate_2 * days + mile_rate_2 * miles :=
by
  sorry


end first_agency_less_expensive_l2846_284661


namespace bug_flower_consumption_l2846_284629

theorem bug_flower_consumption (total_bugs : ℕ) (total_flowers : ℕ) (flowers_per_bug : ℕ) :
  total_bugs = 3 →
  total_flowers = 6 →
  total_flowers = total_bugs * flowers_per_bug →
  flowers_per_bug = 2 := by
  sorry

end bug_flower_consumption_l2846_284629


namespace min_value_trig_expression_l2846_284679

theorem min_value_trig_expression (A : Real) (h : 0 < A ∧ A < Real.pi / 2) :
  Real.sqrt (Real.sin A ^ 4 + 1) + Real.sqrt (Real.cos A ^ 4 + 4) ≥ Real.sqrt 10 := by
  sorry

end min_value_trig_expression_l2846_284679


namespace intersection_point_determines_m_l2846_284668

-- Define the two lines
def line1 (x y m : ℝ) : Prop := 3 * x - 2 * y = m
def line2 (x y : ℝ) : Prop := -x - 2 * y = -10

-- Define the intersection point
def intersection (x y : ℝ) : Prop := line1 x y 6 ∧ line2 x y

-- Theorem statement
theorem intersection_point_determines_m :
  ∃ y : ℝ, intersection 4 y → (∀ m : ℝ, line1 4 y m → m = 6) := by sorry

end intersection_point_determines_m_l2846_284668


namespace reseating_ways_l2846_284652

/-- Represents the number of ways n women can be reseated under the given rules -/
def S : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 3
  | n + 3 => S (n + 2) + S (n + 1) + S n

/-- The number of women -/
def num_women : ℕ := 12

/-- The theorem stating that the number of ways 12 women can be reseated is 927 -/
theorem reseating_ways : S num_women = 927 := by
  sorry

end reseating_ways_l2846_284652


namespace find_number_l2846_284637

theorem find_number (x : ℝ) : ((x - 1.9) * 1.5 + 32) / 2.5 = 20 → x = 13.9 := by
  sorry

end find_number_l2846_284637


namespace problem_1_l2846_284684

theorem problem_1 (x : ℝ) (h1 : x ≠ 0) (h2 : x^2 - Real.sqrt 5 * x - x - 1 = 0) :
  x^2 + 1/x^2 = 8 + 2 * Real.sqrt 5 := by
  sorry

end problem_1_l2846_284684


namespace root_line_tangent_to_discriminant_parabola_l2846_284633

/-- The discriminant parabola in the Opq plane -/
def discriminant_parabola (p q : ℝ) : Prop := p^2 - 4*q = 0

/-- The root line for a given real number a in the Opq plane -/
def root_line (a p q : ℝ) : Prop := a^2 + a*p + q = 0

/-- A line is tangent to the discriminant parabola -/
def is_tangent_line (p q : ℝ → ℝ) : Prop :=
  ∃ (x : ℝ), discriminant_parabola (p x) (q x) ∧
    ∀ (y : ℝ), y ≠ x → ¬discriminant_parabola (p y) (q y)

theorem root_line_tangent_to_discriminant_parabola :
  (∀ a : ℝ, ∃ p q : ℝ → ℝ, is_tangent_line p q ∧ ∀ x : ℝ, root_line a (p x) (q x)) ∧
  (∀ p q : ℝ → ℝ, is_tangent_line p q → ∃ a : ℝ, ∀ x : ℝ, root_line a (p x) (q x)) :=
sorry

end root_line_tangent_to_discriminant_parabola_l2846_284633


namespace absolute_value_equation_solution_l2846_284667

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 24| + |x - 30| = |3*x - 72| :=
by
  -- The unique solution is x = 26
  use 26
  sorry

end absolute_value_equation_solution_l2846_284667


namespace susan_bob_cat_difference_l2846_284615

/-- The number of cats Susan has initially -/
def susan_initial_cats : ℕ := 21

/-- The number of cats Bob has -/
def bob_cats : ℕ := 3

/-- The number of cats Susan gives to Robert -/
def cats_given_to_robert : ℕ := 4

/-- Theorem stating the difference between Susan's remaining cats and Bob's cats -/
theorem susan_bob_cat_difference : 
  susan_initial_cats - cats_given_to_robert - bob_cats = 14 := by
  sorry

end susan_bob_cat_difference_l2846_284615


namespace difference_of_place_values_l2846_284688

def numeral : ℕ := 7669

def place_value (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (10 ^ position)

theorem difference_of_place_values : 
  place_value 6 2 - place_value 6 1 = 540 := by
  sorry

end difference_of_place_values_l2846_284688


namespace truncated_cube_properties_l2846_284677

/-- A space-filling cube arrangement --/
structure CubeArrangement where
  /-- The space is filled with equal cubes --/
  space_filled : Bool
  /-- Eight cubes converge at each vertex --/
  eight_cubes_at_vertex : Bool

/-- A truncated cube in the arrangement --/
structure TruncatedCube where
  /-- The number of faces after truncation --/
  num_faces : Nat
  /-- The number of octagonal faces --/
  num_octagonal_faces : Nat
  /-- The number of triangular faces --/
  num_triangular_faces : Nat

/-- The result of truncating and joining cubes in the arrangement --/
def truncate_and_join (arr : CubeArrangement) : TruncatedCube × Rational :=
  sorry

/-- Theorem stating the properties of truncated cubes and space occupation --/
theorem truncated_cube_properties (arr : CubeArrangement) :
  arr.space_filled ∧ arr.eight_cubes_at_vertex →
  let (truncated_cube, octahedra_space) := truncate_and_join arr
  truncated_cube.num_faces = 14 ∧
  truncated_cube.num_octagonal_faces = 6 ∧
  truncated_cube.num_triangular_faces = 8 ∧
  octahedra_space = 5/6 :=
  sorry

end truncated_cube_properties_l2846_284677


namespace quadratic_expression_evaluation_l2846_284617

theorem quadratic_expression_evaluation :
  let x : ℝ := 2
  (x^2 - 3*x + 2) = 0 := by
  sorry

end quadratic_expression_evaluation_l2846_284617


namespace alex_problem_count_l2846_284650

/-- Given that Alex has written 61 problems out of 187 total problems,
    this theorem proves that he needs to write 65 more problems
    to have written half of the total problems. -/
theorem alex_problem_count (alex_initial : ℕ) (total_initial : ℕ)
    (h1 : alex_initial = 61)
    (h2 : total_initial = 187) :
    ∃ x : ℕ, 2 * (alex_initial + x) = total_initial + x ∧ x = 65 := by
  sorry

end alex_problem_count_l2846_284650


namespace percentage_with_repeat_approx_l2846_284699

/-- The count of all possible five-digit numbers -/
def total_five_digit_numbers : ℕ := 90000

/-- The count of five-digit numbers without repeated digits -/
def no_repeat_numbers : ℕ := 27216

/-- The count of five-digit numbers with at least one repeated digit -/
def repeat_numbers : ℕ := total_five_digit_numbers - no_repeat_numbers

/-- The percentage of five-digit numbers with at least one repeated digit -/
def percentage_with_repeat : ℚ :=
  (repeat_numbers : ℚ) / (total_five_digit_numbers : ℚ) * 100

theorem percentage_with_repeat_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 10 ∧ 
  abs (percentage_with_repeat - 698 / 10) < ε :=
sorry

end percentage_with_repeat_approx_l2846_284699


namespace cost_increase_percentage_l2846_284675

theorem cost_increase_percentage 
  (initial_cost_eggs initial_cost_apples : ℝ)
  (h_equal_initial_cost : initial_cost_eggs = initial_cost_apples)
  (egg_price_decrease : ℝ := 0.02)
  (apple_price_increase : ℝ := 0.10) :
  let new_cost_eggs := initial_cost_eggs * (1 - egg_price_decrease)
  let new_cost_apples := initial_cost_apples * (1 + apple_price_increase)
  let total_initial_cost := initial_cost_eggs + initial_cost_apples
  let total_new_cost := new_cost_eggs + new_cost_apples
  (total_new_cost - total_initial_cost) / total_initial_cost = 0.04 := by
  sorry

end cost_increase_percentage_l2846_284675


namespace book_sale_loss_percentage_l2846_284662

/-- Given two books with a total cost of 600, where one is sold at a loss and the other at a 19% gain,
    both sold at the same price, and the cost of the book sold at a loss is 350,
    prove that the loss percentage on the first book is 15%. -/
theorem book_sale_loss_percentage : 
  ∀ (total_cost cost_book1 cost_book2 selling_price gain_percentage : ℚ),
  total_cost = 600 →
  cost_book1 = 350 →
  cost_book2 = total_cost - cost_book1 →
  gain_percentage = 19 →
  selling_price = cost_book2 * (1 + gain_percentage / 100) →
  (cost_book1 - selling_price) / cost_book1 * 100 = 15 := by
  sorry

end book_sale_loss_percentage_l2846_284662


namespace max_negative_integers_l2846_284673

theorem max_negative_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  (∃ neg_count : ℕ, neg_count ≤ 3 ∧
    neg_count = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) +
                (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) +
                (if e < 0 then 1 else 0) + (if f < 0 then 1 else 0)) ∧
  ¬(∃ neg_count : ℕ, neg_count > 3 ∧
    neg_count = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) +
                (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) +
                (if e < 0 then 1 else 0) + (if f < 0 then 1 else 0)) :=
by sorry

end max_negative_integers_l2846_284673


namespace at_least_one_not_less_than_two_l2846_284676

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 := by
  sorry

end at_least_one_not_less_than_two_l2846_284676


namespace max_value_of_sum_products_l2846_284621

theorem max_value_of_sum_products (a b c d : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) (nonneg_d : d ≥ 0)
  (sum_constraint : a + b + c + d = 120) :
  ab + bc + cd ≤ 3600 :=
sorry

end max_value_of_sum_products_l2846_284621


namespace translation_proof_l2846_284643

-- Define a translation of the complex plane
def translation (z w : ℂ) := z + w

-- Theorem statement
theorem translation_proof (t : ℂ → ℂ) :
  (∃ w : ℂ, ∀ z : ℂ, t z = translation z w) →
  (t (1 + 3*I) = 4 + 7*I) →
  (t (2 - I) = 5 + 3*I) :=
by sorry

end translation_proof_l2846_284643


namespace village_population_growth_l2846_284695

theorem village_population_growth (c d : ℕ) : 
  c^3 + 180 = d^3 + 10 →                     -- Population condition for 2001
  (d + 1)^3 = d^3 + 180 →                    -- Population condition for 2011
  (((d + 1)^3 - c^3) * 100) / c^3 = 101 :=   -- Percent growth over 20 years
by
  sorry

end village_population_growth_l2846_284695


namespace probability_B_given_A_l2846_284630

/-- Represents the number of people in the research study group -/
def group_size : ℕ := 6

/-- Represents the number of halls in the exhibition -/
def num_halls : ℕ := 3

/-- Represents the event A: In the first hour, each hall has exactly 2 people -/
def event_A : Prop := True

/-- Represents the event B: In the second hour, there are exactly 2 people in Hall A -/
def event_B : Prop := True

/-- Represents the number of ways event B can occur given event A has occurred -/
def ways_B_given_A : ℕ := 3

/-- Represents the total number of possible distributions in the second hour -/
def total_distributions : ℕ := 8

/-- The probability of event B given event A -/
def P_B_given_A : ℚ := ways_B_given_A / total_distributions

theorem probability_B_given_A : 
  P_B_given_A = 3 / 8 :=
sorry

end probability_B_given_A_l2846_284630


namespace symmetry_composition_is_translation_l2846_284639

/-- Central symmetry with respect to a point -/
def central_symmetry {V : Type*} [AddCommGroup V] (center : V) (point : V) : V :=
  2 • center - point

/-- Composition of two central symmetries -/
def compose_symmetries {V : Type*} [AddCommGroup V] (O₁ O₂ : V) (point : V) : V :=
  central_symmetry O₂ (central_symmetry O₁ point)

/-- Translation by a vector -/
def translate {V : Type*} [AddCommGroup V] (v : V) (point : V) : V :=
  point + v

theorem symmetry_composition_is_translation {V : Type*} [AddCommGroup V] (O₁ O₂ : V) :
  ∀ (point : V), compose_symmetries O₁ O₂ point = translate (2 • (O₂ - O₁)) point := by
  sorry

end symmetry_composition_is_translation_l2846_284639


namespace packing_problem_l2846_284606

theorem packing_problem :
  ∃! n : ℕ, 500 ≤ n ∧ n ≤ 600 ∧ n % 20 = 13 ∧ n % 27 = 20 ∧ n = 533 := by
  sorry

end packing_problem_l2846_284606


namespace sum_opposite_and_sqrt_81_l2846_284653

-- Define the function for the sum
def sum_opposite_and_sqrt : ℝ → Set ℝ :=
  λ x => {2 + Real.sqrt 2, Real.sqrt 2 - 4}

-- State the theorem
theorem sum_opposite_and_sqrt_81 :
  sum_opposite_and_sqrt (Real.sqrt 81) = {2 + Real.sqrt 2, Real.sqrt 2 - 4} :=
by sorry

end sum_opposite_and_sqrt_81_l2846_284653


namespace teacher_engineer_ratio_l2846_284660

theorem teacher_engineer_ratio (t e : ℕ) (t_pos : t > 0) (e_pos : e > 0) :
  (40 * t + 55 * e) / (t + e) = 45 →
  t / e = 2 := by
sorry

end teacher_engineer_ratio_l2846_284660


namespace total_potatoes_eq_sum_l2846_284623

/-- The number of potatoes mother bought -/
def total_potatoes : ℕ := sorry

/-- The number of potatoes used for salads -/
def salad_potatoes : ℕ := 15

/-- The number of potatoes used for mashed potatoes -/
def mashed_potatoes : ℕ := 24

/-- The number of leftover potatoes -/
def leftover_potatoes : ℕ := 13

/-- Theorem stating that the total number of potatoes is equal to the sum of
    potatoes used for salads, mashed potatoes, and leftover potatoes -/
theorem total_potatoes_eq_sum :
  total_potatoes = salad_potatoes + mashed_potatoes + leftover_potatoes := by sorry

end total_potatoes_eq_sum_l2846_284623


namespace systematic_sampling_selection_l2846_284674

/-- Represents a student number in the range [1, 1000] -/
def StudentNumber := Fin 1000

/-- The total number of students -/
def totalStudents : ℕ := 1000

/-- The number of students to be selected -/
def sampleSize : ℕ := 200

/-- The sample interval for systematic sampling -/
def sampleInterval : ℕ := totalStudents / sampleSize

/-- Predicate to check if a student number is selected in the systematic sampling -/
def isSelected (n : StudentNumber) : Prop :=
  n.val % sampleInterval = 122 % sampleInterval

theorem systematic_sampling_selection :
  isSelected ⟨121, by norm_num⟩ → isSelected ⟨926, by norm_num⟩ := by
  sorry

end systematic_sampling_selection_l2846_284674


namespace complex_magnitude_problem_l2846_284642

theorem complex_magnitude_problem (w z : ℂ) :
  w * z = 15 - 20 * I ∧ Complex.abs w = 5 → Complex.abs z = 5 := by
  sorry

end complex_magnitude_problem_l2846_284642


namespace quadratic_equation_solutions_quadratic_equation_with_square_l2846_284659

theorem quadratic_equation_solutions :
  (∃ x : ℝ, x^2 - 4*x - 8 = 0) ↔ 
  (∃ x : ℝ, x = 2 + 2*Real.sqrt 3 ∨ x = 2 - 2*Real.sqrt 3) :=
sorry

theorem quadratic_equation_with_square :
  (∃ x : ℝ, (x - 2)^2 = 2*x - 4) ↔ 
  (∃ x : ℝ, x = 2 ∨ x = 4) :=
sorry

end quadratic_equation_solutions_quadratic_equation_with_square_l2846_284659


namespace quadratic_sum_l2846_284614

/-- A quadratic function f(x) = ax^2 + bx + c with roots at -3 and 5, and minimum value 36 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x ≥ 36) ∧
  QuadraticFunction a b c (-3) = 0 ∧
  QuadraticFunction a b c 5 = 0 →
  a + b + c = 36 :=
by sorry

end quadratic_sum_l2846_284614


namespace max_base_eight_digit_sum_l2846_284625

/-- Represents a positive integer in base 8 --/
def BaseEightRepresentation := List Nat

/-- Converts a natural number to its base-eight representation --/
def toBaseEight (n : Nat) : BaseEightRepresentation :=
  sorry

/-- Calculates the sum of digits in a base-eight representation --/
def digitSum (rep : BaseEightRepresentation) : Nat :=
  sorry

/-- Theorem stating the maximum digit sum for numbers less than 1729 in base 8 --/
theorem max_base_eight_digit_sum :
  (∃ (n : Nat), n < 1729 ∧ 
    digitSum (toBaseEight n) = 19 ∧ 
    ∀ (m : Nat), m < 1729 → digitSum (toBaseEight m) ≤ 19) :=
  sorry

end max_base_eight_digit_sum_l2846_284625


namespace apartment_building_floors_l2846_284671

/-- Represents an apartment building with the given specifications. -/
structure ApartmentBuilding where
  floors : ℕ
  apartments_per_floor : ℕ
  people_per_apartment : ℕ
  total_people : ℕ

/-- Calculates the number of people on a full floor. -/
def people_on_full_floor (building : ApartmentBuilding) : ℕ :=
  building.apartments_per_floor * building.people_per_apartment

/-- Calculates the number of people on a half-capacity floor. -/
def people_on_half_capacity_floor (building : ApartmentBuilding) : ℕ :=
  (building.apartments_per_floor / 2) * building.people_per_apartment

/-- Theorem stating that given the conditions, the apartment building has 12 floors. -/
theorem apartment_building_floors
  (building : ApartmentBuilding)
  (h1 : building.apartments_per_floor = 10)
  (h2 : building.people_per_apartment = 4)
  (h3 : building.total_people = 360)
  (h4 : building.total_people = 
    (building.floors / 2 * people_on_full_floor building) + 
    (building.floors / 2 * people_on_half_capacity_floor building)) :
  building.floors = 12 := by
  sorry


end apartment_building_floors_l2846_284671


namespace arrangement_count_is_correct_l2846_284670

/-- The number of ways to arrange 3 individuals on 7 steps -/
def arrangement_count : ℕ := 336

/-- The number of steps -/
def num_steps : ℕ := 7

/-- The number of individuals -/
def num_individuals : ℕ := 3

/-- The maximum number of people allowed on each step -/
def max_per_step : ℕ := 2

/-- Theorem stating that the arrangement count is correct -/
theorem arrangement_count_is_correct :
  arrangement_count = 
    (num_steps.choose num_individuals * num_individuals.factorial) + 
    (num_individuals.choose 2 * num_steps.choose 2) := by
  sorry

end arrangement_count_is_correct_l2846_284670


namespace sufficient_not_necessary_l2846_284612

theorem sufficient_not_necessary (x y : ℝ) : 
  (∀ x y : ℝ, x + y ≠ 8 → (x ≠ 2 ∨ y ≠ 6)) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 6) ∧ x + y = 8) :=
by sorry

end sufficient_not_necessary_l2846_284612


namespace quadratic_points_ordering_l2846_284603

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the points
def P₁ : ℝ × ℝ := (-1, f (-1))
def P₂ : ℝ × ℝ := (2, f 2)
def P₃ : ℝ × ℝ := (5, f 5)

-- Theorem statement
theorem quadratic_points_ordering :
  P₂.2 > P₁.2 ∧ P₁.2 > P₃.2 := by sorry

end quadratic_points_ordering_l2846_284603


namespace probability_zero_or_one_excellent_equals_formula_l2846_284696

def total_people : ℕ := 12
def excellent_students : ℕ := 5
def selected_people : ℕ := 5

def probability_zero_or_one_excellent : ℚ :=
  (Nat.choose (total_people - excellent_students) selected_people +
   Nat.choose excellent_students 1 * Nat.choose (total_people - excellent_students) (selected_people - 1)) /
  Nat.choose total_people selected_people

theorem probability_zero_or_one_excellent_equals_formula :
  probability_zero_or_one_excellent = 
  (Nat.choose (total_people - excellent_students) selected_people +
   Nat.choose excellent_students 1 * Nat.choose (total_people - excellent_students) (selected_people - 1)) /
  Nat.choose total_people selected_people :=
by sorry

end probability_zero_or_one_excellent_equals_formula_l2846_284696


namespace integer_solutions_yk_eq_x2_plus_x_l2846_284687

theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (h : k > 1) :
  ∀ x y : ℤ, y^k = x^2 + x ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) := by
  sorry

end integer_solutions_yk_eq_x2_plus_x_l2846_284687


namespace product_ratio_theorem_l2846_284655

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 3/4 := by
  sorry

end product_ratio_theorem_l2846_284655


namespace seventh_data_entry_is_18_l2846_284683

-- Define the given conditions
def total_results : ℕ := 15
def total_average : ℚ := 60
def first_set_count : ℕ := 7
def first_set_average : ℚ := 56
def second_set_count : ℕ := 6
def second_set_average : ℚ := 63
def last_set_count : ℕ := 6
def last_set_average : ℚ := 66

-- Theorem to prove
theorem seventh_data_entry_is_18 :
  ∃ (x : ℚ),
    x = 18 ∧
    total_average * total_results =
      first_set_average * first_set_count +
      second_set_average * second_set_count +
      x +
      (last_set_average * last_set_count - second_set_average * second_set_count - x) :=
by sorry

end seventh_data_entry_is_18_l2846_284683


namespace circle_tape_length_16_strips_l2846_284635

/-- The total length of a circle-shaped tape made from overlapping strips -/
def circle_tape_length (num_strips : ℕ) (strip_length : ℝ) (overlap_length : ℝ) : ℝ :=
  num_strips * strip_length - num_strips * overlap_length

/-- Theorem: The length of a circle-shaped tape made from 16 strips of 10.4 cm
    with 3.5 cm overlaps is 110.4 cm -/
theorem circle_tape_length_16_strips :
  circle_tape_length 16 10.4 3.5 = 110.4 := by
  sorry

#eval circle_tape_length 16 10.4 3.5

end circle_tape_length_16_strips_l2846_284635


namespace false_proposition_l2846_284609

def p1 : Prop := ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0

def p2 : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

theorem false_proposition : ¬((¬p1) ∧ (¬p2)) := by
  sorry

end false_proposition_l2846_284609


namespace tangent_line_minimum_value_l2846_284631

theorem tangent_line_minimum_value (k b : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧
    k = 1 / (2 * Real.sqrt x₀) ∧
    b = Real.sqrt x₀ / 2 + 1 ∧
    k * x₀ + b = Real.sqrt x₀ + 1) →
  k^2 + b^2 - 2*b ≥ -1/2 :=
by sorry

end tangent_line_minimum_value_l2846_284631


namespace solve_for_t_l2846_284628

theorem solve_for_t (a b d x y t : ℕ) 
  (h1 : a + b = x)
  (h2 : x + d = t)
  (h3 : t + a = y)
  (h4 : b + d + y = 16)
  (ha : a > 0)
  (hb : b > 0)
  (hd : d > 0)
  (hx : x > 0)
  (hy : y > 0)
  (ht : t > 0) :
  t = 8 := by
sorry


end solve_for_t_l2846_284628


namespace rectangle_tiling_divisibility_l2846_284613

/-- An L-shaped piece made of 4 unit squares -/
structure LPiece :=
  (squares : Fin 4 → (Nat × Nat))

/-- A tiling of an m × n rectangle with L-shaped pieces -/
def Tiling (m n : Nat) := List LPiece

/-- Predicate to check if a tiling is valid for an m × n rectangle -/
def IsValidTiling (t : Tiling m n) : Prop := sorry

theorem rectangle_tiling_divisibility (m n : Nat) (t : Tiling m n) :
  IsValidTiling t → (m * n) % 8 = 0 := by sorry

end rectangle_tiling_divisibility_l2846_284613


namespace quinary_to_octal_conversion_polynomial_evaluation_l2846_284627

-- Define the polynomial f(x)
def f (x : ℕ) : ℕ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

-- Define the quinary to decimal conversion function
def quinary_to_decimal (q : ℕ) : ℕ :=
  (q / 1000) * 5^3 + ((q / 100) % 10) * 5^2 + ((q / 10) % 10) * 5^1 + (q % 10)

-- Define the decimal to octal conversion function
def decimal_to_octal (d : ℕ) : ℕ :=
  (d / 64) * 100 + ((d / 8) % 8) * 10 + (d % 8)

theorem quinary_to_octal_conversion :
  decimal_to_octal (quinary_to_decimal 1234) = 302 := by sorry

theorem polynomial_evaluation :
  f 3 = 21324 := by sorry

end quinary_to_octal_conversion_polynomial_evaluation_l2846_284627


namespace abs_purely_imaginary_complex_l2846_284697

/-- Given a complex number z = (a + i) / (1 + i) where a is real,
    if z is purely imaginary, then its absolute value is 1. -/
theorem abs_purely_imaginary_complex (a : ℝ) :
  let z : ℂ := (a + Complex.I) / (1 + Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 1 := by
  sorry

end abs_purely_imaginary_complex_l2846_284697


namespace target_line_correct_l2846_284666

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel_lines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The given line 2x - y + 1 = 0 -/
def given_line : Line2D :=
  { a := 2, b := -1, c := 1 }

/-- Point A (-1, 0) -/
def point_A : Point2D :=
  { x := -1, y := 0 }

/-- The line we need to prove -/
def target_line : Line2D :=
  { a := 2, b := -1, c := 2 }

theorem target_line_correct :
  point_on_line point_A target_line ∧
  parallel_lines target_line given_line := by
  sorry

end target_line_correct_l2846_284666


namespace max_projection_area_specific_tetrahedron_l2846_284651

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  /-- Two adjacent faces are equilateral triangles -/
  adjacent_faces_equilateral : Bool
  /-- Side length of the equilateral triangular faces -/
  side_length : ℝ
  /-- Dihedral angle between the two adjacent equilateral faces -/
  dihedral_angle : ℝ

/-- Calculates the maximum projection area of the tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating the maximum projection area for a specific tetrahedron -/
theorem max_projection_area_specific_tetrahedron :
  ∀ t : Tetrahedron,
    t.adjacent_faces_equilateral = true →
    t.side_length = 1 →
    t.dihedral_angle = π / 3 →
    max_projection_area t = Real.sqrt 3 / 4 :=
  sorry

end max_projection_area_specific_tetrahedron_l2846_284651


namespace domain_intersection_and_union_range_of_p_l2846_284618

def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}
def C (p : ℝ) : Set ℝ := {x | 4*x + p < 0}

theorem domain_intersection_and_union :
  (A ∩ B = {x | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3}) ∧
  (A ∪ B = Set.univ) :=
sorry

theorem range_of_p (p : ℝ) :
  (C p ⊆ A) → p ≥ 4 :=
sorry

end domain_intersection_and_union_range_of_p_l2846_284618


namespace gecko_eats_hundred_crickets_l2846_284672

/-- The number of crickets a gecko eats over three days -/
def gecko_crickets : ℕ → Prop
| C => 
  -- Day 1: 30% of total
  let day1 : ℚ := 0.3 * C
  -- Day 2: 6 less than day 1
  let day2 : ℚ := day1 - 6
  -- Day 3: 34 crickets
  let day3 : ℕ := 34
  -- Total crickets eaten equals sum of three days
  C = day1.ceil + day2.ceil + day3

theorem gecko_eats_hundred_crickets : 
  ∃ C : ℕ, gecko_crickets C ∧ C = 100 := by sorry

end gecko_eats_hundred_crickets_l2846_284672


namespace sum_of_repeating_decimals_l2846_284658

def repeating_decimal_3 : ℚ := 1 / 3
def repeating_decimal_56 : ℚ := 56 / 99

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_56 = 89 / 99 := by
  sorry

end sum_of_repeating_decimals_l2846_284658


namespace problem_solution_l2846_284681

theorem problem_solution : 
  ((-1 : ℝ) ^ 2023 + Real.sqrt 9 - 2022 ^ 0 = 1) ∧ 
  ((Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) + Real.sqrt 8 = 2 + 2 * Real.sqrt 2) := by
  sorry

end problem_solution_l2846_284681


namespace maurice_cookout_invites_l2846_284686

/-- The number of people Maurice can invite to the cookout --/
def people_invited : ℕ := by sorry

theorem maurice_cookout_invites :
  let packages : ℕ := 4
  let pounds_per_package : ℕ := 5
  let pounds_per_burger : ℕ := 2
  let total_pounds : ℕ := packages * pounds_per_package
  let total_burgers : ℕ := total_pounds / pounds_per_burger
  people_invited = total_burgers - 1 := by sorry

end maurice_cookout_invites_l2846_284686


namespace special_function_property_l2846_284691

/-- A function that is monotonically increasing on [0,2] and f(x+2) is even -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x ≤ f y) ∧
  (∀ x, f (x + 2) = f (-x + 2))

theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  f (7/2) < f 1 ∧ f 1 < f (5/2) := by
  sorry

end special_function_property_l2846_284691


namespace calen_lost_pencils_l2846_284619

theorem calen_lost_pencils (p_candy p_caleb p_calen_original p_calen_after_loss : ℕ) :
  p_candy = 9 →
  p_caleb = 2 * p_candy - 3 →
  p_calen_original = p_caleb + 5 →
  p_calen_after_loss = 10 →
  p_calen_original - p_calen_after_loss = 10 := by
sorry

end calen_lost_pencils_l2846_284619


namespace roberto_cost_per_dozen_approx_l2846_284605

/-- Represents the chicken and egg scenario for Roberto --/
structure ChickenScenario where
  num_chickens : ℕ
  chicken_cost : ℚ
  weekly_feed_cost : ℚ
  eggs_per_chicken_per_week : ℕ
  break_even_weeks : ℕ

/-- Calculates the cost per dozen eggs given a ChickenScenario --/
def cost_per_dozen (scenario : ChickenScenario) : ℚ :=
  let total_cost := scenario.num_chickens * scenario.chicken_cost + 
                    scenario.weekly_feed_cost * scenario.break_even_weeks
  let total_eggs := scenario.num_chickens * scenario.eggs_per_chicken_per_week * 
                    scenario.break_even_weeks
  let total_dozens := total_eggs / 12
  total_cost / total_dozens

/-- Roberto's specific scenario --/
def roberto_scenario : ChickenScenario :=
  { num_chickens := 4
  , chicken_cost := 20
  , weekly_feed_cost := 1
  , eggs_per_chicken_per_week := 3
  , break_even_weeks := 81 }

/-- Theorem stating that Roberto's cost per dozen eggs is approximately $1.99 --/
theorem roberto_cost_per_dozen_approx :
  abs (cost_per_dozen roberto_scenario - 1.99) < 0.01 := by
  sorry


end roberto_cost_per_dozen_approx_l2846_284605


namespace age_problem_l2846_284607

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = b / 2 →
  a + b + c + d = 44 →
  b = 14 := by
sorry

end age_problem_l2846_284607


namespace factorial_equation_solution_l2846_284644

theorem factorial_equation_solution :
  ∃ (n : ℕ), (4 * 3 * 2 * 1) / (Nat.factorial (4 - n)) = 24 ∧ n = 3 := by
sorry

end factorial_equation_solution_l2846_284644


namespace sequence_problem_l2846_284690

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b / a = c / b

theorem sequence_problem (x y : ℝ) :
  is_arithmetic_sequence (2 * x) 1 (y - 1) →
  is_geometric_sequence (y + 3) (|x + 1| + |x - 1|) (Real.cos (Real.arcsin (Real.sqrt (1 - x^2)))) →
  (x + 1) * (y + 1) = 4 ∨ (x + 1) * (y + 1) = 2 * (Real.sqrt 17 - 3) := by
  sorry

end sequence_problem_l2846_284690


namespace company_women_count_l2846_284692

theorem company_women_count (total_workers : ℕ) 
  (h1 : total_workers / 3 = total_workers - (2 * total_workers / 3))  -- One-third don't have retirement plan
  (h2 : (total_workers / 3) / 5 = total_workers / 15)  -- 20% of workers without plan are women
  (h3 : (2 * total_workers / 3) * 2 / 5 = (2 * total_workers / 3) - ((2 * total_workers / 3) * 3 / 5))  -- 40% of workers with plan are men
  (h4 : 144 = (2 * total_workers / 3) * 2 / 5)  -- 144 men in the company
  : (total_workers / 15 + (2 * total_workers / 3) * 3 / 5 = 252) := by
  sorry

end company_women_count_l2846_284692


namespace natalia_comics_count_l2846_284611

/-- The number of novels Natalia has -/
def novels : ℕ := 145

/-- The number of documentaries Natalia has -/
def documentaries : ℕ := 419

/-- The number of albums Natalia has -/
def albums : ℕ := 209

/-- The number of items each crate can hold -/
def items_per_crate : ℕ := 9

/-- The number of crates Natalia will use -/
def num_crates : ℕ := 116

/-- The number of comics Natalia has -/
def comics : ℕ := 271

theorem natalia_comics_count : 
  novels + documentaries + albums + comics = num_crates * items_per_crate := by
  sorry

end natalia_comics_count_l2846_284611


namespace isosceles_triangle_yw_length_l2846_284648

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the isosceles property
def isIsosceles (t : Triangle) : Prop :=
  dist t.X t.Z = dist t.Y t.Z

-- Define the point W on XZ
def W (t : Triangle) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem isosceles_triangle_yw_length 
  (t : Triangle) 
  (h1 : isIsosceles t) 
  (h2 : dist t.X t.Y = 3) 
  (h3 : dist t.X t.Z = 5) 
  (h4 : dist t.Y t.Z = 5) 
  (h5 : dist (W t) t.Z = 2) : 
  dist (W t) t.Y = Real.sqrt 18.5 := 
sorry

end isosceles_triangle_yw_length_l2846_284648


namespace set_operation_equality_l2846_284669

def U : Finset Nat := {1,2,3,4,5}
def M : Finset Nat := {3,4,5}
def N : Finset Nat := {1,2,5}

theorem set_operation_equality : 
  (U \ M) ∩ N = {1,2} := by sorry

end set_operation_equality_l2846_284669


namespace somu_age_problem_l2846_284693

/-- Somu's age problem -/
theorem somu_age_problem (somu_age father_age : ℕ) : 
  somu_age = father_age / 3 →
  somu_age - 7 = (father_age - 7) / 5 →
  somu_age = 14 := by
  sorry

end somu_age_problem_l2846_284693


namespace characterization_theorem_l2846_284640

/-- A function that checks if a number is prime -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

/-- A function that checks if a number is a square of a prime -/
def is_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ n = p * p

/-- The main theorem statement -/
theorem characterization_theorem (n : ℕ) (h : n ≥ 2) :
  (∀ d : ℕ, d ≥ 2 → d ∣ n → (d - 1) ∣ (n - 1)) ↔ (is_prime n ∨ is_prime_square n) :=
sorry

end characterization_theorem_l2846_284640


namespace divisible_by_eleven_iff_d_equals_three_l2846_284616

/-- A function that constructs a six-digit number from its digits -/
def sixDigitNumber (a b c d e f : ℕ) : ℕ := 
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

/-- Proposition: The six-digit number 54321d is divisible by 11 if and only if d = 3 -/
theorem divisible_by_eleven_iff_d_equals_three : 
  ∀ d : ℕ, d < 10 → (sixDigitNumber 5 4 3 2 1 d) % 11 = 0 ↔ d = 3 :=
by sorry

end divisible_by_eleven_iff_d_equals_three_l2846_284616


namespace quarter_orbit_distance_l2846_284622

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ  -- Distance of nearest point from focus
  apogee : ℝ   -- Distance of farthest point from focus

/-- Calculates the distance from a point on the orbit to the focus -/
def distance_to_focus (orbit : EllipticalOrbit) (fraction : ℝ) : ℝ :=
  sorry

theorem quarter_orbit_distance (orbit : EllipticalOrbit) 
  (h1 : orbit.perigee = 3)
  (h2 : orbit.apogee = 15) :
  distance_to_focus orbit 0.25 = 13.5 := by
  sorry

end quarter_orbit_distance_l2846_284622


namespace sum_a_b_equals_one_l2846_284608

theorem sum_a_b_equals_one (a b : ℝ) : 
  Real.sqrt (a - b - 3) + abs (2 * a - 4) = 0 → a + b = 1 := by
sorry

end sum_a_b_equals_one_l2846_284608


namespace quiz_ranking_l2846_284680

structure Student where
  name : String
  score : ℕ

def Hannah : Student := { name := "Hannah", score := 0 }
def Cassie : Student := { name := "Cassie", score := 0 }
def Bridget : Student := { name := "Bridget", score := 0 }

def is_not_highest (s : Student) (others : List Student) : Prop :=
  ∃ t ∈ others, t.score > s.score

def scored_better_than (s1 s2 : Student) : Prop :=
  s1.score > s2.score

def is_not_lowest (s : Student) (others : List Student) : Prop :=
  ∃ t ∈ others, s.score > t.score

theorem quiz_ranking :
  is_not_highest Hannah [Cassie, Bridget] →
  scored_better_than Bridget Cassie →
  is_not_lowest Cassie [Hannah, Bridget] →
  scored_better_than Bridget Cassie ∧ scored_better_than Cassie Hannah :=
by sorry

end quiz_ranking_l2846_284680


namespace blake_change_l2846_284663

/-- The amount Blake spent on oranges -/
def orange_cost : ℕ := 40

/-- The amount Blake spent on apples -/
def apple_cost : ℕ := 50

/-- The amount Blake spent on mangoes -/
def mango_cost : ℕ := 60

/-- The initial amount Blake had -/
def initial_amount : ℕ := 300

/-- The change Blake received after shopping -/
def change : ℕ := initial_amount - (orange_cost + apple_cost + mango_cost)

theorem blake_change : change = 150 := by sorry

end blake_change_l2846_284663


namespace final_cost_is_12_l2846_284646

def purchase1 : ℚ := 2.45
def purchase2 : ℚ := 7.60
def purchase3 : ℚ := 3.15
def discount_rate : ℚ := 0.1

def total_before_discount : ℚ := purchase1 + purchase2 + purchase3
def discount_amount : ℚ := total_before_discount * discount_rate
def total_after_discount : ℚ := total_before_discount - discount_amount

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - x.floor < 0.5 then x.floor else x.ceil

theorem final_cost_is_12 :
  round_to_nearest_dollar total_after_discount = 12 := by
  sorry

end final_cost_is_12_l2846_284646


namespace inequality_proof_l2846_284638

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_bound_x : x ≥ (1 : ℝ) / 2) (h_bound_y : y ≥ (1 : ℝ) / 2) (h_bound_z : z ≥ (1 : ℝ) / 2)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (1/x + 1/y - 1/z) * (1/x - 1/y + 1/z) ≥ 2 := by
  sorry

end inequality_proof_l2846_284638


namespace marble_sharing_l2846_284665

theorem marble_sharing (initial_marbles : ℝ) (initial_marbles_pos : initial_marbles > 0) :
  let remaining_after_lara := initial_marbles * (1 - 0.3)
  let remaining_after_max := remaining_after_lara * (1 - 0.15)
  let remaining_after_ben := remaining_after_max * (1 - 0.2)
  remaining_after_ben / initial_marbles = 0.476 := by
sorry

end marble_sharing_l2846_284665


namespace derivative_of_x_exp_x_l2846_284620

noncomputable def f (x : ℝ) := x * Real.exp x

theorem derivative_of_x_exp_x :
  deriv f = fun x ↦ (1 + x) * Real.exp x := by sorry

end derivative_of_x_exp_x_l2846_284620


namespace power_of_power_l2846_284641

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end power_of_power_l2846_284641


namespace total_courses_is_200_l2846_284600

/-- The number of college courses Max attended -/
def max_courses : ℕ := 40

/-- The number of college courses Sid attended relative to Max -/
def sid_multiplier : ℕ := 4

/-- The total number of college courses attended by Max and Sid -/
def total_courses : ℕ := max_courses + sid_multiplier * max_courses

/-- Theorem stating that the total number of courses attended by Max and Sid is 200 -/
theorem total_courses_is_200 : total_courses = 200 := by
  sorry

end total_courses_is_200_l2846_284600


namespace translation_of_line_segment_l2846_284602

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def translatePoint (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_of_line_segment (A B A' : Point) :
  A.x = -4 ∧ A.y = -1 ∧
  B.x = 1 ∧ B.y = 1 ∧
  A'.x = -2 ∧ A'.y = 2 →
  ∃ (t : Translation), translatePoint A t = A' ∧ translatePoint B t = { x := 3, y := 4 } := by
  sorry

end translation_of_line_segment_l2846_284602


namespace max_cookies_juan_l2846_284664

/-- Represents the ingredients required for baking cookies -/
structure Ingredients where
  milk : ℚ
  sugar : ℚ
  flour : ℚ

/-- Represents the storage capacity for ingredients -/
structure StorageCapacity where
  milk : ℚ
  sugar : ℚ
  flour : ℚ

/-- Calculate the maximum number of cookies that can be baked given the ingredients per cookie and storage capacity -/
def max_cookies (ingredients_per_cookie : Ingredients) (storage : StorageCapacity) : ℚ :=
  min (storage.milk / ingredients_per_cookie.milk)
      (min (storage.sugar / ingredients_per_cookie.sugar)
           (storage.flour / ingredients_per_cookie.flour))

/-- Theorem: The maximum number of cookies Juan can bake within storage constraints is 320 -/
theorem max_cookies_juan :
  let ingredients_per_40_cookies : Ingredients := { milk := 10, sugar := 5, flour := 15 }
  let ingredients_per_cookie : Ingredients := {
    milk := ingredients_per_40_cookies.milk / 40,
    sugar := ingredients_per_40_cookies.sugar / 40,
    flour := ingredients_per_40_cookies.flour / 40
  }
  let storage : StorageCapacity := { milk := 80, sugar := 200, flour := 220 }
  max_cookies ingredients_per_cookie storage = 320 := by sorry

end max_cookies_juan_l2846_284664


namespace remainder_123456789012_mod_252_l2846_284654

theorem remainder_123456789012_mod_252 : 
  123456789012 % 252 = 156 := by sorry

end remainder_123456789012_mod_252_l2846_284654


namespace simplify_expression_l2846_284626

theorem simplify_expression (a b : ℝ) :
  (-2 * a^2 * b)^3 / (-2 * a * b) * (1/3 * a^2 * b^3) = 4/3 * a^7 * b^5 := by
  sorry

end simplify_expression_l2846_284626


namespace trig_identity_l2846_284657

theorem trig_identity : 
  (Real.cos (20 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (25 * π / 180)^2 - Real.sin (25 * π / 180)^2) = 1/2 := by
  sorry

end trig_identity_l2846_284657


namespace circle_area_greater_than_five_times_triangle_area_l2846_284682

theorem circle_area_greater_than_five_times_triangle_area 
  (R r : ℝ) (S : ℝ) (h_R_positive : R > 0) (h_r_positive : r > 0) (h_S_positive : S > 0)
  (h_R_r : R ≥ 2 * r) -- Euler's inequality
  (h_S : S ≤ (3 * Real.sqrt 3 / 2) * R * r) -- Upper bound for triangle area
  : π * (R + r)^2 > 5 * S := by
  sorry

end circle_area_greater_than_five_times_triangle_area_l2846_284682


namespace jessicas_carrots_l2846_284685

theorem jessicas_carrots (joan_carrots : ℕ) (total_carrots : ℕ) 
  (h1 : joan_carrots = 29) 
  (h2 : total_carrots = 40) : 
  total_carrots - joan_carrots = 11 := by
  sorry

end jessicas_carrots_l2846_284685


namespace hyperbola_asymptote_slope_l2846_284624

/-- The positive slope of an asymptote of the hyperbola defined by 
    √((x-2)² + (y-3)²) - √((x-8)² + (y-3)²) = 4 -/
theorem hyperbola_asymptote_slope : ∃ (m : ℝ), m > 0 ∧ 
  (∀ (x y : ℝ), Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4 →
    m = Real.sqrt 5 / 2) :=
by sorry

end hyperbola_asymptote_slope_l2846_284624


namespace sixth_root_of_unity_product_l2846_284634

theorem sixth_root_of_unity_product (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 0 := by
  sorry

end sixth_root_of_unity_product_l2846_284634


namespace square_pyramid_sphere_ratio_l2846_284678

/-- Represents a pyramid with a square base -/
structure SquarePyramid where
  -- Length of the edge of the square base
  baseEdge : ℝ
  -- Height of the pyramid (perpendicular distance from apex to base)
  height : ℝ

/-- Calculates the ratio of surface areas of circumscribed to inscribed spheres for a square pyramid -/
def sphereAreaRatio (p : SquarePyramid) : ℝ :=
  -- This function would contain the actual calculation
  sorry

theorem square_pyramid_sphere_ratio :
  let p := SquarePyramid.mk 8 6
  sphereAreaRatio p = 41 / 4 := by
  sorry

end square_pyramid_sphere_ratio_l2846_284678


namespace reciprocals_not_arithmetic_sequence_l2846_284636

/-- If positive numbers a, b, c form an arithmetic sequence with non-zero common difference,
    then their reciprocals cannot form an arithmetic sequence. -/
theorem reciprocals_not_arithmetic_sequence (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (h_arith : ∃ d ≠ 0, b - a = d ∧ c - b = d) : 
    ¬∃ k : ℝ, (1 / b - 1 / a = k) ∧ (1 / c - 1 / b = k) := by
  sorry

end reciprocals_not_arithmetic_sequence_l2846_284636


namespace range_of_m_l2846_284604

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2) →
  (∀ x, g x = 2^x - m) →
  (∀ x₁ ∈ Set.Icc (-1) 3, ∃ x₂ ∈ Set.Icc 0 2, f x₁ ≥ g x₂) →
  m ≥ 1 := by
  sorry

end range_of_m_l2846_284604


namespace altitude_inradius_equality_l2846_284694

/-- Triangle ABC with side lengths a, b, c, altitudes h_a, h_b, h_c, and inradius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  r : ℝ

/-- The theorem states that the sum of altitudes equals 9 times the inradius 
    if and only if the triangle is equilateral -/
theorem altitude_inradius_equality (t : Triangle) : 
  t.h_a + t.h_b + t.h_c = 9 * t.r ↔ t.a = t.b ∧ t.b = t.c :=
sorry

end altitude_inradius_equality_l2846_284694


namespace complex_magnitude_l2846_284689

theorem complex_magnitude (z : ℂ) (h : z^4 = 80 - 96*I) : Complex.abs z = 5^(3/4) := by
  sorry

end complex_magnitude_l2846_284689


namespace simplify_trig_expression_l2846_284656

theorem simplify_trig_expression (x : ℝ) :
  (Real.sin x + Real.sin (3 * x)) / (1 + Real.cos x + Real.cos (3 * x)) =
  (4 * Real.sin x * Real.cos x ^ 2) / (1 - 4 * Real.cos x ^ 2) := by
  sorry

end simplify_trig_expression_l2846_284656


namespace arithmetic_sequence_sum_l2846_284601

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 2 = 10 →
  a 4 = a 3 + 2 →
  a 3 + a 4 = 18 := by
sorry

end arithmetic_sequence_sum_l2846_284601


namespace breakfast_cooking_time_l2846_284698

theorem breakfast_cooking_time (num_sausages num_eggs egg_time total_time : ℕ) 
  (h1 : num_sausages = 3)
  (h2 : num_eggs = 6)
  (h3 : egg_time = 4)
  (h4 : total_time = 39) :
  ∃ (sausage_time : ℕ), 
    sausage_time * num_sausages + egg_time * num_eggs = total_time ∧ 
    sausage_time = 5 := by
  sorry

end breakfast_cooking_time_l2846_284698


namespace derivative_x_cos_x_l2846_284632

theorem derivative_x_cos_x (x : ℝ) :
  deriv (fun x => x * Real.cos x) x = Real.cos x - x * Real.sin x := by
  sorry

end derivative_x_cos_x_l2846_284632


namespace juggling_balls_average_l2846_284610

/-- Represents a juggling sequence -/
def JugglingSequence (n : ℕ) := Fin n → ℕ

/-- The number of balls in a juggling sequence -/
def numberOfBalls (n : ℕ) (j : JugglingSequence n) : ℚ :=
  (Finset.sum Finset.univ (fun i => j i)) / n

theorem juggling_balls_average (n : ℕ) (j : JugglingSequence n) :
  numberOfBalls n j = (Finset.sum Finset.univ (fun i => j i)) / n :=
by sorry

end juggling_balls_average_l2846_284610
