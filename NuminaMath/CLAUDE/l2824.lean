import Mathlib

namespace odd_function_value_at_one_l2824_282486

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a-5)*x^2 + a*x

-- State the theorem
theorem odd_function_value_at_one :
  ∀ a : ℝ, (∀ x : ℝ, f a (-x) = -(f a x)) → f a 1 = 6 :=
by
  sorry

end odd_function_value_at_one_l2824_282486


namespace complex_equation_ratio_l2824_282405

theorem complex_equation_ratio (a b : ℝ) : 
  (Complex.mk a b) * (Complex.mk 1 1) = Complex.mk 7 (-3) → a / b = -2 / 5 := by
  sorry

end complex_equation_ratio_l2824_282405


namespace square_difference_equality_l2824_282408

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end square_difference_equality_l2824_282408


namespace puppies_adoption_time_l2824_282414

/-- The number of days required to adopt all puppies -/
def adoption_days (initial_puppies : ℕ) (additional_puppies : ℕ) (adoption_rate : ℕ) : ℕ :=
  (initial_puppies + additional_puppies) / adoption_rate

/-- Theorem: Given the initial conditions, it takes 9 days to adopt all puppies -/
theorem puppies_adoption_time :
  adoption_days 2 34 4 = 9 := by
  sorry

end puppies_adoption_time_l2824_282414


namespace rebecca_eggs_l2824_282433

/-- The number of eggs Rebecca has -/
def number_of_eggs : ℕ := 3 * 3

/-- The size of each group of eggs -/
def group_size : ℕ := 3

/-- The number of groups Rebecca created -/
def number_of_groups : ℕ := 3

theorem rebecca_eggs : 
  number_of_eggs = group_size * number_of_groups := by sorry

end rebecca_eggs_l2824_282433


namespace expression_evaluation_l2824_282496

theorem expression_evaluation : 
  let x : ℤ := -3
  7 * x^2 - 3 * (2 * x^2 - 1) - 4 = 8 := by sorry

end expression_evaluation_l2824_282496


namespace probability_of_red_in_C_l2824_282422

-- Define the initial configuration of balls in each box
def box_A : ℕ × ℕ := (2, 1)  -- (red, yellow)
def box_B : ℕ × ℕ := (1, 2)  -- (red, yellow)
def box_C : ℕ × ℕ := (1, 1)  -- (red, yellow)

-- Define the process of transferring balls
def transfer_process : (ℕ × ℕ) → (ℕ × ℕ) → (ℕ × ℕ) → ℚ := sorry

-- Theorem statement
theorem probability_of_red_in_C :
  transfer_process box_A box_B box_C = 17/36 := by sorry

end probability_of_red_in_C_l2824_282422


namespace jeff_donuts_per_day_l2824_282401

/-- The number of days Jeff makes donuts -/
def days : ℕ := 12

/-- The number of donuts Jeff eats per day -/
def jeff_eats_per_day : ℕ := 1

/-- The total number of donuts Chris eats -/
def chris_eats_total : ℕ := 8

/-- The number of donuts that fit in each box -/
def donuts_per_box : ℕ := 10

/-- The number of boxes Jeff can fill -/
def boxes_filled : ℕ := 10

/-- The number of donuts Jeff makes each day -/
def donuts_per_day : ℕ := 10

theorem jeff_donuts_per_day :
  ∃ (d : ℕ), 
    d * days - (jeff_eats_per_day * days) - chris_eats_total = boxes_filled * donuts_per_box ∧
    d = donuts_per_day :=
by sorry

end jeff_donuts_per_day_l2824_282401


namespace finish_book_in_three_days_l2824_282494

def pages_to_read_on_third_day (total_pages : ℕ) (pages_day1 : ℕ) (fewer_pages_day2 : ℕ) : ℕ :=
  total_pages - (pages_day1 + (pages_day1 - fewer_pages_day2))

theorem finish_book_in_three_days (total_pages : ℕ) (pages_day1 : ℕ) (fewer_pages_day2 : ℕ)
  (h1 : total_pages = 100)
  (h2 : pages_day1 = 35)
  (h3 : fewer_pages_day2 = 5) :
  pages_to_read_on_third_day total_pages pages_day1 fewer_pages_day2 = 35 := by
  sorry

end finish_book_in_three_days_l2824_282494


namespace max_xy_value_l2824_282427

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 112 := by
  sorry

end max_xy_value_l2824_282427


namespace two_numbers_sum_and_reverse_sum_l2824_282436

def reverse (n : ℕ) : ℕ :=
  let rec rev_aux (n acc : ℕ) : ℕ :=
    if n = 0 then acc
    else rev_aux (n / 10) (acc * 10 + n % 10)
  rev_aux n 0

theorem two_numbers_sum_and_reverse_sum :
  ∃ a b : ℕ,
    a + b = 2017 ∧
    reverse a + reverse b = 8947 ∧
    a = 1408 ∧
    b = 609 := by
  sorry

end two_numbers_sum_and_reverse_sum_l2824_282436


namespace distance_between_points_l2824_282488

theorem distance_between_points (speed_A speed_B speed_C : ℝ) (extra_time : ℝ) : 
  speed_A = 100 →
  speed_B = 90 →
  speed_C = 75 →
  extra_time = 3 →
  ∃ (distance : ℝ), 
    distance / (speed_A + speed_B) + extra_time = distance / speed_C ∧
    distance = 650 := by
  sorry

end distance_between_points_l2824_282488


namespace restaurant_bill_l2824_282437

/-- Given a total spent of $23 on an entree and a dessert, where the entree costs $5 more than the dessert, prove that the cost of the entree is $14. -/
theorem restaurant_bill (total : ℝ) (entree_cost : ℝ) (dessert_cost : ℝ) 
  (h1 : total = 23)
  (h2 : entree_cost = dessert_cost + 5)
  (h3 : total = entree_cost + dessert_cost) :
  entree_cost = 14 := by
  sorry

end restaurant_bill_l2824_282437


namespace inequality_proof_l2824_282424

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (a * c) ≤ a + b + c) ∧
  (a + b + c = 1 → (2 * a * b) / (a + b) + (2 * b * c) / (b + c) + (2 * c * a) / (c + a) ≤ 1) :=
by sorry

end inequality_proof_l2824_282424


namespace common_number_in_list_l2824_282461

theorem common_number_in_list (list : List ℝ) : 
  list.length = 9 →
  (list.take 5).sum / 5 = 7 →
  (list.drop 4).sum / 5 = 9 →
  list.sum / 9 = 73 / 9 →
  ∃ x ∈ list.take 5 ∩ list.drop 4, x = 7 :=
by sorry

end common_number_in_list_l2824_282461


namespace tom_bought_six_oranges_l2824_282443

/-- Represents the number of oranges Tom bought -/
def num_oranges : ℕ := 6

/-- Represents the number of apples Tom bought -/
def num_apples : ℕ := 7 - num_oranges

/-- The cost of an orange in cents -/
def orange_cost : ℕ := 90

/-- The cost of an apple in cents -/
def apple_cost : ℕ := 60

/-- The total number of fruits bought -/
def total_fruits : ℕ := 7

/-- The total cost in cents -/
def total_cost : ℕ := orange_cost * num_oranges + apple_cost * num_apples

theorem tom_bought_six_oranges :
  num_oranges + num_apples = total_fruits ∧
  total_cost % 100 = 0 ∧
  num_oranges = 6 := by
  sorry

end tom_bought_six_oranges_l2824_282443


namespace coefficient_x5_in_binomial_expansion_l2824_282447

theorem coefficient_x5_in_binomial_expansion :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k) * (1 ^ (8 - k)) * (1 ^ k)) = 256 ∧
  (Finset.range 9).sum (fun k => if k = 3 then (Nat.choose 8 k) else 0) = 56 :=
by sorry

end coefficient_x5_in_binomial_expansion_l2824_282447


namespace intersection_of_sets_l2824_282492

theorem intersection_of_sets :
  let A : Set ℤ := {-1, 2, 4}
  let B : Set ℤ := {0, 2, 6}
  A ∩ B = {2} := by
  sorry

end intersection_of_sets_l2824_282492


namespace equal_charge_at_120_minutes_l2824_282431

/-- United Telephone's base rate in dollars -/
def united_base : ℚ := 6

/-- United Telephone's per-minute rate in dollars -/
def united_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℚ := 12

/-- Atlantic Call's per-minute rate in dollars -/
def atlantic_per_minute : ℚ := 1/5

/-- The number of minutes at which both companies charge the same amount -/
def equal_charge_minutes : ℚ := 120

theorem equal_charge_at_120_minutes :
  united_base + united_per_minute * equal_charge_minutes =
  atlantic_base + atlantic_per_minute * equal_charge_minutes :=
sorry

end equal_charge_at_120_minutes_l2824_282431


namespace extra_domino_possible_l2824_282466

/-- Represents a 6x6 chessboard -/
def Chessboard := Fin 6 → Fin 6 → Bool

/-- A domino is a pair of adjacent squares on the chessboard -/
def Domino := (Fin 6 × Fin 6) × (Fin 6 × Fin 6)

/-- Checks if two squares are adjacent -/
def adjacent (s1 s2 : Fin 6 × Fin 6) : Prop :=
  (s1.1 = s2.1 ∧ s1.2.succ = s2.2) ∨
  (s1.1 = s2.1 ∧ s1.2 = s2.2.succ) ∨
  (s1.1.succ = s2.1 ∧ s1.2 = s2.2) ∨
  (s1.1 = s2.1.succ ∧ s1.2 = s2.2)

/-- Checks if a domino is valid (covers two adjacent squares) -/
def validDomino (d : Domino) : Prop :=
  adjacent d.1 d.2

/-- Checks if two dominoes overlap -/
def overlap (d1 d2 : Domino) : Prop :=
  d1.1 = d2.1 ∨ d1.1 = d2.2 ∨ d1.2 = d2.1 ∨ d1.2 = d2.2

/-- Represents a configuration of 11 dominoes on the chessboard -/
def Configuration := Fin 11 → Domino

/-- Checks if a configuration is valid (no overlaps) -/
def validConfiguration (config : Configuration) : Prop :=
  ∀ i j : Fin 11, i ≠ j → ¬(overlap (config i) (config j))

/-- Theorem: Given a valid configuration of 11 dominoes on a 6x6 chessboard,
    there always exists at least two adjacent empty squares -/
theorem extra_domino_possible (config : Configuration) 
  (h_valid : validConfiguration config) :
  ∃ s1 s2 : Fin 6 × Fin 6, adjacent s1 s2 ∧
    (∀ i : Fin 11, s1 ≠ (config i).1 ∧ s1 ≠ (config i).2 ∧
                   s2 ≠ (config i).1 ∧ s2 ≠ (config i).2) :=
  sorry


end extra_domino_possible_l2824_282466


namespace factor_tree_value_l2824_282416

theorem factor_tree_value (X Y Z F G : ℕ) : 
  X = Y * Z ∧
  Y = 7 * F ∧
  F = 2 * 5 ∧
  Z = 11 * G ∧
  G = 3 * 7 →
  X = 16170 := by sorry

end factor_tree_value_l2824_282416


namespace gcd_180_450_l2824_282409

theorem gcd_180_450 : Nat.gcd 180 450 = 90 := by
  sorry

end gcd_180_450_l2824_282409


namespace min_value_expression_l2824_282467

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((x^2 + y^2) * (3 * x^2 + y^2))) / (x * y) ≥ 1 + Real.sqrt 3 :=
by sorry

end min_value_expression_l2824_282467


namespace union_equals_A_A_subset_complement_B_l2824_282476

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 2}

-- Theorem 1
theorem union_equals_A (m : ℝ) : A ∪ B m = A → m = 1 := by
  sorry

-- Theorem 2
theorem A_subset_complement_B (m : ℝ) : A ⊆ (B m)ᶜ → m > 5 ∨ m < -3 := by
  sorry

end union_equals_A_A_subset_complement_B_l2824_282476


namespace sqrt_x_minus_one_meaningful_only_four_satisfies_l2824_282415

theorem sqrt_x_minus_one_meaningful (x : ℝ) : x - 1 ≥ 0 ↔ x ≥ 1 := by sorry

theorem only_four_satisfies :
  (4 - 1 ≥ 0) ∧ 
  ¬(-4 - 1 ≥ 0) ∧ 
  ¬(-1 - 1 ≥ 0) ∧ 
  ¬(0 - 1 ≥ 0) := by sorry

end sqrt_x_minus_one_meaningful_only_four_satisfies_l2824_282415


namespace teacher_age_l2824_282417

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) :
  num_students = 100 →
  student_avg_age = 17 →
  total_avg_age = 18 →
  (num_students : ℝ) * student_avg_age + (num_students + 1 : ℝ) * total_avg_age - (num_students : ℝ) * student_avg_age = 118 :=
by sorry

end teacher_age_l2824_282417


namespace equal_squares_sum_l2824_282472

theorem equal_squares_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 0 → a = b ∧ b = c := by
  sorry

end equal_squares_sum_l2824_282472


namespace area_PTW_approx_34_l2824_282439

-- Define the areas of triangles as functions of x
def area_PUW (x : ℝ) : ℝ := 4*x + 4
def area_SUW (x : ℝ) : ℝ := 2*x + 20
def area_SVW (x : ℝ) : ℝ := 5*x + 20
def area_SVR (x : ℝ) : ℝ := 5*x + 11
def area_QVR (x : ℝ) : ℝ := 8*x + 32
def area_QVW (x : ℝ) : ℝ := 8*x + 50

-- Define the equation for solving x
def solve_for_x (x : ℝ) : Prop :=
  (area_QVW x) / (area_SVW x) = (area_QVR x) / (area_SVR x)

-- Define the area of triangle PTW
noncomputable def area_PTW (x : ℝ) : ℝ := 
  sorry  -- The exact formula is not provided in the problem

-- State the theorem
theorem area_PTW_approx_34 :
  ∃ (x : ℝ), solve_for_x x ∧ 
  (∀ (y : ℝ), abs (area_PTW x - 34) ≤ abs (area_PTW x - y) ∨ y = 34) :=
sorry

end area_PTW_approx_34_l2824_282439


namespace unique_positive_number_l2824_282487

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 / x := by
  sorry

end unique_positive_number_l2824_282487


namespace factorization_of_5x_squared_minus_5_l2824_282480

theorem factorization_of_5x_squared_minus_5 (x : ℝ) : 5 * x^2 - 5 = 5 * (x + 1) * (x - 1) := by
  sorry

end factorization_of_5x_squared_minus_5_l2824_282480


namespace parentheses_equivalence_l2824_282491

theorem parentheses_equivalence (a b c : ℝ) : a + 2*b - 3*c = a + (2*b - 3*c) := by
  sorry

end parentheses_equivalence_l2824_282491


namespace complex_number_location_l2824_282464

theorem complex_number_location :
  ∀ (z : ℂ), (z * Complex.I = 1 - 2 * Complex.I) →
  (z = -2 - Complex.I ∧ z.re < 0 ∧ z.im < 0) := by
  sorry

end complex_number_location_l2824_282464


namespace danielas_age_l2824_282444

/-- Given the ages and relationships of several people, prove Daniela's age --/
theorem danielas_age (clara_age : ℕ) (daniela_age evelina_age fidel_age caitlin_age : ℕ) :
  clara_age = 60 →
  daniela_age = evelina_age - 8 →
  evelina_age = clara_age / 3 →
  fidel_age = 2 * caitlin_age →
  fidel_age = evelina_age - 6 →
  daniela_age = 12 := by
sorry


end danielas_age_l2824_282444


namespace pizza_toppings_l2824_282481

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 14)
  (h2 : pepperoni_slices = 8)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ slice ∈ Finset.range mushroom_slices)) :
  (pepperoni_slices + mushroom_slices - total_slices : ℕ) = 6 := by
  sorry

end pizza_toppings_l2824_282481


namespace additional_bags_needed_l2824_282489

/-- The number of people guaranteed to show up -/
def guaranteed_visitors : ℕ := 50

/-- The number of additional people who might show up -/
def potential_visitors : ℕ := 40

/-- The number of extravagant gift bags already made -/
def extravagant_bags : ℕ := 10

/-- The number of average gift bags already made -/
def average_bags : ℕ := 20

/-- The total number of visitors Carl is preparing for -/
def total_visitors : ℕ := guaranteed_visitors + potential_visitors

/-- The total number of gift bags already made -/
def existing_bags : ℕ := extravagant_bags + average_bags

/-- Theorem stating the number of additional bags Carl needs to make -/
theorem additional_bags_needed : total_visitors - existing_bags = 60 := by
  sorry

end additional_bags_needed_l2824_282489


namespace four_digit_kabulek_numbers_l2824_282495

def is_kabulek (n : ℕ) : Prop :=
  ∃ x y : ℕ,
    n = 100 * x + y ∧
    x < 100 ∧
    y < 100 ∧
    (x + y) ^ 2 = n

theorem four_digit_kabulek_numbers :
  ∀ n : ℕ,
    1000 ≤ n ∧ n < 10000 →
    is_kabulek n ↔ n = 2025 ∨ n = 3025 ∨ n = 9801 :=
by sorry

end four_digit_kabulek_numbers_l2824_282495


namespace tips_fraction_is_three_sevenths_l2824_282485

/-- Represents the waiter's income structure -/
structure WaiterIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the fraction of income from tips -/
def fractionFromTips (income : WaiterIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: The fraction of income from tips is 3/7 when tips are 3/4 of the salary -/
theorem tips_fraction_is_three_sevenths (income : WaiterIncome) 
    (h : income.tips = 3/4 * income.salary) : 
    fractionFromTips income = 3/7 := by
  sorry

end tips_fraction_is_three_sevenths_l2824_282485


namespace fraction_simplification_l2824_282452

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end fraction_simplification_l2824_282452


namespace hybrid_car_journey_length_l2824_282462

theorem hybrid_car_journey_length :
  ∀ (d : ℝ),
  d > 60 →
  (60 : ℝ) / d + (d - 60) / (0.04 * (d - 60)) = 50 →
  d = 120 := by
  sorry

end hybrid_car_journey_length_l2824_282462


namespace coles_return_speed_coles_return_speed_is_90_l2824_282438

/-- Calculates the average speed on the return trip given the conditions of Cole's journey. -/
theorem coles_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : ℝ :=
  let distance_to_work := speed_to_work * (time_to_work / 60)
  let time_to_return := total_time - (time_to_work / 60)
  distance_to_work / time_to_return

/-- Proves that Cole's average speed on the return trip is 90 km/h given the problem conditions. -/
theorem coles_return_speed_is_90 :
  coles_return_speed 30 2 90 = 90 := by
  sorry

end coles_return_speed_coles_return_speed_is_90_l2824_282438


namespace max_distance_point_to_circle_l2824_282475

/-- The maximum distance from a point to a circle -/
theorem max_distance_point_to_circle :
  let circle := {(x, y) : ℝ × ℝ | (x - 3)^2 + (y - 4)^2 = 25}
  let point := (2, 3)
  (⨆ p ∈ circle, Real.sqrt ((point.1 - p.1)^2 + (point.2 - p.2)^2)) = Real.sqrt 2 + 5 := by
  sorry

end max_distance_point_to_circle_l2824_282475


namespace boxer_weight_theorem_l2824_282425

def initial_weight : ℝ := 106

def weight_loss_rate_A1 : ℝ := 2
def weight_loss_rate_A2 : ℝ := 3
def weight_loss_duration_A1 : ℝ := 2
def weight_loss_duration_A2 : ℝ := 2

def weight_loss_rate_B : ℝ := 3
def weight_loss_duration_B : ℝ := 3

def weight_loss_rate_C : ℝ := 4
def weight_loss_duration_C : ℝ := 4

def final_weight_A : ℝ := initial_weight - (weight_loss_rate_A1 * weight_loss_duration_A1 + weight_loss_rate_A2 * weight_loss_duration_A2)
def final_weight_B : ℝ := initial_weight - (weight_loss_rate_B * weight_loss_duration_B)
def final_weight_C : ℝ := initial_weight - (weight_loss_rate_C * weight_loss_duration_C)

theorem boxer_weight_theorem :
  final_weight_A = 96 ∧
  final_weight_B = 97 ∧
  final_weight_C = 90 := by
  sorry

end boxer_weight_theorem_l2824_282425


namespace mias_socks_theorem_l2824_282410

/-- Represents the number of pairs of socks at each price point --/
structure SockInventory where
  one_dollar : ℕ
  two_dollar : ℕ
  three_dollar : ℕ
  four_dollar : ℕ

/-- Calculates the total number of pairs of socks --/
def total_pairs (s : SockInventory) : ℕ :=
  s.one_dollar + s.two_dollar + s.three_dollar + s.four_dollar

/-- Calculates the total cost of all socks --/
def total_cost (s : SockInventory) : ℕ :=
  s.one_dollar + 2 * s.two_dollar + 3 * s.three_dollar + 4 * s.four_dollar

/-- Checks if at least one pair of each type was bought --/
def at_least_one_each (s : SockInventory) : Prop :=
  s.one_dollar ≥ 1 ∧ s.two_dollar ≥ 1 ∧ s.three_dollar ≥ 1 ∧ s.four_dollar ≥ 1

theorem mias_socks_theorem (s : SockInventory) 
  (h1 : total_pairs s = 16)
  (h2 : total_cost s = 36)
  (h3 : at_least_one_each s) :
  s.one_dollar = 3 := by
  sorry

end mias_socks_theorem_l2824_282410


namespace project_budget_increase_l2824_282429

/-- Proves that the annual increase in budget for project Q is $50,000 -/
theorem project_budget_increase (initial_q initial_v decrease_v : ℕ) 
  (h1 : initial_q = 540000)
  (h2 : initial_v = 780000)
  (h3 : decrease_v = 10000)
  (h4 : ∃ (increase_q : ℕ), initial_q + 4 * increase_q = initial_v - 4 * decrease_v) :
  ∃ (increase_q : ℕ), increase_q = 50000 := by
sorry


end project_budget_increase_l2824_282429


namespace log_inequality_l2824_282473

theorem log_inequality : ∃ (a b : ℝ), 
  a = Real.log 0.8 / Real.log 0.7 ∧ 
  b = Real.log 0.9 / Real.log 1.1 ∧ 
  a > 0 ∧ 0 > b := by
  sorry

end log_inequality_l2824_282473


namespace sphere_dihedral_angle_segment_fraction_l2824_282411

/-- The fraction of the segment AB that lies outside two equal touching spheres inscribed in a dihedral angle -/
theorem sphere_dihedral_angle_segment_fraction (α : Real) : 
  α > 0 → α < π → 
  let f := (1 - (Real.cos (α / 2))^2) / (1 + (Real.cos (α / 2))^2)
  0 ≤ f ∧ f ≤ 1 := by sorry

end sphere_dihedral_angle_segment_fraction_l2824_282411


namespace faculty_reduction_percentage_l2824_282459

theorem faculty_reduction_percentage (original : ℕ) (reduced : ℕ) : 
  original = 260 → reduced = 195 → 
  (original - reduced : ℚ) / original * 100 = 25 := by
  sorry

end faculty_reduction_percentage_l2824_282459


namespace smallest_base_for_fourth_power_l2824_282445

theorem smallest_base_for_fourth_power (b : ℕ) : 
  b > 0 ∧ 
  (∃ (x : ℕ), 7 * b^2 + 7 * b + 7 = x^4) ∧
  (∀ (c : ℕ), 0 < c ∧ c < b → ¬∃ (y : ℕ), 7 * c^2 + 7 * c + 7 = y^4) → 
  b = 18 := by
sorry

end smallest_base_for_fourth_power_l2824_282445


namespace water_percentage_in_container_l2824_282421

/-- Proves that the percentage of a container's capacity filled with 8 liters of water is 20%,
    given that the total capacity of 40 such containers is 1600 liters. -/
theorem water_percentage_in_container (container_capacity : ℝ) : 
  (40 * container_capacity = 1600) → (8 / container_capacity * 100 = 20) := by
  sorry

end water_percentage_in_container_l2824_282421


namespace average_of_six_integers_l2824_282478

theorem average_of_six_integers (a b c d e f : ℤ) :
  a = 22 ∧ b = 23 ∧ c = 23 ∧ d = 25 ∧ e = 26 ∧ f = 31 →
  (a + b + c + d + e + f) / 6 = 25 := by
sorry

end average_of_six_integers_l2824_282478


namespace group_dynamics_index_difference_l2824_282426

theorem group_dynamics_index_difference :
  let n : ℕ := 35
  let k1 : ℕ := 15
  let k2 : ℕ := 5
  let k3 : ℕ := 8
  let l1 : ℕ := 6
  let l2 : ℕ := 10
  let index_females : ℚ := ((n - k1 + k2) / n : ℚ) * (1 + k3/10)
  let index_males : ℚ := ((n - (n - k1) + l1) / n : ℚ) * (1 + l2/10)
  index_females - index_males = 3/35 := by
  sorry

end group_dynamics_index_difference_l2824_282426


namespace hidden_cannonball_label_l2824_282448

structure CannonballPyramid where
  total_cannonballs : Nat
  labels : Finset Char
  label_count : Char → Nat
  visible_count : Char → Nat

def is_valid_pyramid (p : CannonballPyramid) : Prop :=
  p.total_cannonballs = 20 ∧
  p.labels = {'A', 'B', 'C', 'D', 'E'} ∧
  ∀ l ∈ p.labels, p.label_count l = 4 ∧
  ∀ l ∈ p.labels, p.visible_count l ≤ p.label_count l

theorem hidden_cannonball_label (p : CannonballPyramid) 
  (h_valid : is_valid_pyramid p)
  (h_visible : ∀ l ∈ p.labels, l ≠ 'D' → p.visible_count l = 4)
  (h_d_visible : p.visible_count 'D' = 3) :
  p.label_count 'D' - p.visible_count 'D' = 1 := by
sorry

end hidden_cannonball_label_l2824_282448


namespace range_of_a_l2824_282468

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → a ∈ Set.Ioo (-1 : ℝ) 3 := by
  sorry

end range_of_a_l2824_282468


namespace perfect_square_factorization_l2824_282404

theorem perfect_square_factorization (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end perfect_square_factorization_l2824_282404


namespace product_of_three_numbers_l2824_282419

theorem product_of_three_numbers (a b c : ℕ) : 
  (a * b * c = 224) ∧ 
  (a < b) ∧ (b < c) ∧ 
  (a * 2 = c) ∧
  (∀ x y z : ℕ, x * y * z = 224 ∧ x < y ∧ y < z ∧ x * 2 = z → x = a ∧ y = b ∧ z = c) :=
by sorry

end product_of_three_numbers_l2824_282419


namespace max_sum_on_integer_circle_l2824_282430

theorem max_sum_on_integer_circle : 
  ∀ x y : ℤ, x^2 + y^2 = 100 → (∀ a b : ℤ, a^2 + b^2 = 100 → x + y ≥ a + b) → x + y = 14 := by
  sorry

end max_sum_on_integer_circle_l2824_282430


namespace complex_magnitude_problem_l2824_282469

theorem complex_magnitude_problem (a : ℝ) (z : ℂ) : 
  z = (a * Complex.I) / (4 - 3 * Complex.I) → 
  Complex.abs z = 5 → 
  a = 25 ∨ a = -25 := by
  sorry

end complex_magnitude_problem_l2824_282469


namespace expression_equality_l2824_282432

theorem expression_equality : (2^1004 + 5^1005)^2 - (2^1004 - 5^1005)^2 = 20 * 10^1004 := by
  sorry

end expression_equality_l2824_282432


namespace combined_salaries_BCDE_l2824_282441

def salary_A : ℕ := 9000
def average_salary : ℕ := 8200
def num_employees : ℕ := 5

theorem combined_salaries_BCDE :
  salary_A + (num_employees - 1) * (average_salary * num_employees - salary_A) / (num_employees - 1) = average_salary * num_employees :=
by sorry

end combined_salaries_BCDE_l2824_282441


namespace find_some_number_l2824_282455

theorem find_some_number (x : ℝ) (some_number : ℝ) 
  (eq1 : x + some_number = 4) (eq2 : x = 3) : some_number = 1 := by
  sorry

end find_some_number_l2824_282455


namespace number_count_l2824_282493

theorem number_count (average : ℝ) (sum_of_three : ℝ) (average_of_two : ℝ) (n : ℕ) : 
  average = 20 →
  sum_of_three = 48 →
  average_of_two = 26 →
  (average * n : ℝ) = sum_of_three + 2 * average_of_two →
  n = 5 := by
sorry

end number_count_l2824_282493


namespace max_value_x_plus_2cos_x_l2824_282449

open Real

theorem max_value_x_plus_2cos_x (x : ℝ) :
  let f : ℝ → ℝ := λ x => x + 2 * cos x
  (∀ y ∈ Set.Icc 0 (π / 2), f (π / 6) ≥ f y) ∧
  (π / 6 ∈ Set.Icc 0 (π / 2)) :=
by sorry

end max_value_x_plus_2cos_x_l2824_282449


namespace shoe_color_probability_l2824_282428

theorem shoe_color_probability (n : ℕ) (h : n = 6) :
  let total_shoes := 2 * n
  let same_color_selections := n
  let total_selections := total_shoes.choose 2
  (same_color_selections : ℚ) / total_selections = 1 / 11 :=
by sorry

end shoe_color_probability_l2824_282428


namespace g_increasing_on_neg_l2824_282479

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions on f
variable (h1 : ∀ x y, x < y → f x < f y)  -- f is increasing
variable (h2 : ∀ x, f x < 0)  -- f(x) < 0 for all x

-- Define the function g
def g (x : ℝ) : ℝ := x^2 * f x

-- State the theorem
theorem g_increasing_on_neg : 
  ∀ x y, x < y ∧ y < 0 → g f x < g f y :=
sorry

end g_increasing_on_neg_l2824_282479


namespace point_on_y_axis_l2824_282477

theorem point_on_y_axis (x : ℝ) :
  (x^2 - 1 = 0) → 
  ((x^2 - 1, 2*x + 4) = (0, 6) ∨ (x^2 - 1, 2*x + 4) = (0, 2)) :=
by sorry

end point_on_y_axis_l2824_282477


namespace lisa_process_ends_at_39_l2824_282418

/-- The function that represents one step of Lisa's process -/
def f (x : ℕ) : ℕ :=
  (x / 10) + 4 * (x % 10)

/-- The sequence of numbers generated by Lisa's process -/
def lisa_sequence (x : ℕ) : ℕ → ℕ
  | 0 => x
  | n + 1 => f (lisa_sequence x n)

/-- The theorem stating that Lisa's process always ends at 39 when starting with 53^2022 - 1 -/
theorem lisa_process_ends_at_39 :
  ∃ n : ℕ, ∀ m : ℕ, m ≥ n → lisa_sequence (53^2022 - 1) m = 39 :=
sorry

end lisa_process_ends_at_39_l2824_282418


namespace composite_function_inverse_l2824_282484

theorem composite_function_inverse (a b : ℝ) : 
  let f (x : ℝ) := a * x + b
  let g (x : ℝ) := -2 * x^2 + 4 * x - 1
  let h := f ∘ g
  (∀ x, h.invFun x = 2 * x - 3) →
  2 * a - 3 * b = -91 / 32 := by
sorry

end composite_function_inverse_l2824_282484


namespace beijing_to_lanzhou_distance_l2824_282457

/-- The distance from Beijing to Lanzhou, given the distances from Beijing to Lhasa (via Lanzhou) and from Lanzhou to Lhasa. -/
theorem beijing_to_lanzhou_distance 
  (beijing_to_lhasa : ℕ) 
  (lanzhou_to_lhasa : ℕ) 
  (h1 : beijing_to_lhasa = 3985)
  (h2 : lanzhou_to_lhasa = 2054) :
  beijing_to_lhasa - lanzhou_to_lhasa = 1931 :=
by sorry

end beijing_to_lanzhou_distance_l2824_282457


namespace standard_deviation_from_variance_l2824_282420

theorem standard_deviation_from_variance (variance : ℝ) (std_dev : ℝ) :
  variance = 2 → std_dev = Real.sqrt variance → std_dev = Real.sqrt 2 := by
  sorry

end standard_deviation_from_variance_l2824_282420


namespace pigs_in_barn_l2824_282483

/-- The total number of pigs after more pigs join the barn -/
def total_pigs (initial : Float) (joined : Float) : Float :=
  initial + joined

/-- Theorem stating that given 64.0 initial pigs and 86.0 pigs joining, the total is 150.0 -/
theorem pigs_in_barn : total_pigs 64.0 86.0 = 150.0 := by
  sorry

end pigs_in_barn_l2824_282483


namespace concatenated_seven_digit_divisible_by_239_l2824_282423

/-- Represents a sequence of seven-digit numbers -/
def SevenDigitSequence := List Nat

/-- Concatenates a list of natural numbers -/
def concatenate (seq : SevenDigitSequence) : Nat :=
  seq.foldl (fun acc n => acc * 10000000 + n) 0

/-- The sequence of all seven-digit numbers -/
def allSevenDigitNumbers : SevenDigitSequence :=
  List.range 10000000

theorem concatenated_seven_digit_divisible_by_239 :
  ∃ k : ℕ, concatenate allSevenDigitNumbers = 239 * k :=
sorry

end concatenated_seven_digit_divisible_by_239_l2824_282423


namespace complex_fraction_equality_l2824_282471

theorem complex_fraction_equality : (1 + 3*Complex.I) / (Complex.I - 1) = 1 - 2*Complex.I := by
  sorry

end complex_fraction_equality_l2824_282471


namespace sports_store_sales_l2824_282435

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 10

/-- The number of customers in each car -/
def customers_per_car : ℕ := 5

/-- The number of sales made by the music store -/
def music_store_sales : ℕ := 30

/-- The total number of customers in the parking lot -/
def total_customers : ℕ := num_cars * customers_per_car

theorem sports_store_sales :
  total_customers - music_store_sales = 20 := by
  sorry

#check sports_store_sales

end sports_store_sales_l2824_282435


namespace complex_number_problem_l2824_282451

/-- Given a complex number z where z + 2i and z / (2 - i) are real numbers, 
    z = 4 - 2i and (z + ai)² is in the first quadrant when 2 < a < 6 -/
theorem complex_number_problem (z : ℂ) 
  (h1 : (z + 2*Complex.I).im = 0)
  (h2 : (z / (2 - Complex.I)).im = 0) :
  z = 4 - 2*Complex.I ∧ 
  ∀ a : ℝ, (z + a*Complex.I)^2 ∈ {w : ℂ | w.re > 0 ∧ w.im > 0} ↔ 2 < a ∧ a < 6 :=
by sorry

end complex_number_problem_l2824_282451


namespace c_value_is_one_l2824_282470

/-- The quadratic function f(x) = -x^2 + cx + 12 is positive only on (-∞, -3) ∪ (4, ∞) -/
def is_positive_on_intervals (c : ℝ) : Prop :=
  ∀ x : ℝ, (-x^2 + c*x + 12 > 0) ↔ (x < -3 ∨ x > 4)

/-- The value of c for which f(x) = -x^2 + cx + 12 is positive only on (-∞, -3) ∪ (4, ∞) is 1 -/
theorem c_value_is_one :
  ∃! c : ℝ, is_positive_on_intervals c ∧ c = 1 :=
by sorry

end c_value_is_one_l2824_282470


namespace sum_of_max_min_values_l2824_282497

/-- Given real numbers a and b satisfying the condition,
    prove that the sum of max and min values of a^2 + 2b^2 is 16/7 -/
theorem sum_of_max_min_values (a b : ℝ) 
  (h : (a - b/2)^2 = 1 - (7/4)*b^2) : 
  ∃ (t_max t_min : ℝ), 
    (∀ t, t = a^2 + 2*b^2 → t ≤ t_max ∧ t ≥ t_min) ∧
    t_max + t_min = 16/7 := by
  sorry

end sum_of_max_min_values_l2824_282497


namespace rotation_90_ccw_coordinates_l2824_282499

def rotate90CCW (x y : ℝ) : ℝ × ℝ := (-y, x)

theorem rotation_90_ccw_coordinates :
  let A : ℝ × ℝ := (3, 5)
  let A' : ℝ × ℝ := rotate90CCW A.1 A.2
  A' = (5, -3) := by sorry

end rotation_90_ccw_coordinates_l2824_282499


namespace binomial_expansion_property_l2824_282465

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the condition for the sum of the first three binomial coefficients
def first_three_sum_condition (n : ℕ) : Prop :=
  binomial n 0 + binomial n 1 + binomial n 2 = 79

-- Define the coefficient of the k-th term in the expansion
def coefficient (n k : ℕ) : ℚ := sorry

-- Define the property of having maximum coefficient
def has_max_coefficient (n k : ℕ) : Prop :=
  ∀ j, j ≠ k → coefficient n k ≥ coefficient n j

theorem binomial_expansion_property (n : ℕ) 
  (h : n > 0) 
  (h_sum : first_three_sum_condition n) :
  n = 12 ∧ has_max_coefficient n 10 := by sorry

end binomial_expansion_property_l2824_282465


namespace rectangle_area_diagonal_l2824_282463

/-- Proof that for a rectangle with length to width ratio of 5:2 and perimeter 42 cm, 
    the area A can be expressed as (10/29)d^2, where d is the diagonal of the rectangle. -/
theorem rectangle_area_diagonal (length width : ℝ) (d : ℝ) : 
  length / width = 5 / 2 →
  2 * (length + width) = 42 →
  d^2 = length^2 + width^2 →
  length * width = (10/29) * d^2 :=
by sorry

end rectangle_area_diagonal_l2824_282463


namespace rectangle_side_length_l2824_282456

/-- Given two rectangles A and B, with sides (a, b) and (c, d) respectively,
    where the ratio of corresponding sides is 3/4 and rectangle B has sides 4 and 8,
    prove that the side a of rectangle A is 3. -/
theorem rectangle_side_length (a b c d : ℝ) : 
  a / c = 3 / 4 →
  b / d = 3 / 4 →
  c = 4 →
  d = 8 →
  a = 3 := by
  sorry

end rectangle_side_length_l2824_282456


namespace geometric_sequence_a7_l2824_282458

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_a1 : a 1 = 2) (h_a4 : a 4 = 4) : a 7 = 8 := by
  sorry

end geometric_sequence_a7_l2824_282458


namespace perpendicular_and_equal_intercepts_l2824_282400

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def line3 (x y : ℝ) : Prop := 3 * x + 4 * y - 15 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (3, 2)

-- Define the perpendicular line l1
def l1 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0

-- Define the two possible lines with equal intercepts
def l2_case1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def l2_case2 (x y : ℝ) : Prop := x + y - 5 = 0

theorem perpendicular_and_equal_intercepts :
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = P) →
  (∀ x y : ℝ, l1 x y → (x, y) = P ∨ (3 * x + 4 * y ≠ 15)) ∧
  ((∀ x y : ℝ, l2_case1 x y → (x, y) = P) ∨ (∀ x y : ℝ, l2_case2 x y → (x, y) = P)) :=
by sorry

end perpendicular_and_equal_intercepts_l2824_282400


namespace f_equals_2x_plus_7_l2824_282403

-- Define the functions g and f
def g (x : ℝ) : ℝ := 2 * x + 3
def f (x : ℝ) : ℝ := g (x + 2)

-- State the theorem
theorem f_equals_2x_plus_7 : ∀ x : ℝ, f x = 2 * x + 7 := by
  sorry

end f_equals_2x_plus_7_l2824_282403


namespace sum_of_reciprocals_of_roots_l2824_282442

theorem sum_of_reciprocals_of_roots (p q : ℝ) : 
  p^2 - 17*p + 8 = 0 → q^2 - 17*q + 8 = 0 → 1/p + 1/q = 17/8 := by
  sorry

end sum_of_reciprocals_of_roots_l2824_282442


namespace wooden_planks_weight_l2824_282446

theorem wooden_planks_weight
  (crate_capacity : ℕ)
  (num_crates : ℕ)
  (num_nail_bags : ℕ)
  (nail_bag_weight : ℕ)
  (num_hammer_bags : ℕ)
  (hammer_bag_weight : ℕ)
  (num_plank_bags : ℕ)
  (weight_to_leave_out : ℕ)
  (h1 : crate_capacity = 20)
  (h2 : num_crates = 15)
  (h3 : num_nail_bags = 4)
  (h4 : nail_bag_weight = 5)
  (h5 : num_hammer_bags = 12)
  (h6 : hammer_bag_weight = 5)
  (h7 : num_plank_bags = 10)
  (h8 : weight_to_leave_out = 80) :
  (num_crates * crate_capacity - weight_to_leave_out
    - (num_nail_bags * nail_bag_weight + num_hammer_bags * hammer_bag_weight))
  / num_plank_bags = 14 := by
sorry

end wooden_planks_weight_l2824_282446


namespace log_product_equality_l2824_282454

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.log x^2 / Real.log y^5 * 
  Real.log y^3 / Real.log x^4 * 
  Real.log x^4 / Real.log y^3 * 
  Real.log y^5 / Real.log x^3 * 
  Real.log x^3 / Real.log y^4 = 
  (1 / 6) * (Real.log x / Real.log y) :=
sorry

end log_product_equality_l2824_282454


namespace circle_radius_from_tangents_l2824_282434

/-- A circle with two parallel tangents and a third tangent -/
structure CircleWithTangents where
  r : ℝ  -- radius of the circle
  xy : ℝ  -- length of tangent XY
  xpyp : ℝ  -- length of tangent X'Y'

/-- The theorem stating the relationship between the tangents and the radius -/
theorem circle_radius_from_tangents (c : CircleWithTangents) 
  (h1 : c.xy = 7)
  (h2 : c.xpyp = 12) :
  c.r = 4 * Real.sqrt 21 := by
  sorry

end circle_radius_from_tangents_l2824_282434


namespace hyperbola_equation_l2824_282407

/-- Given a hyperbola with the following properties:
  - Standard form equation: x²/a² - y²/b² = 1
  - a > 0 and b > 0
  - A focus at (2, 0)
  - Asymptotes: y = ±√3x
  Prove that the equation of the hyperbola is x² - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (focus : (2 : ℝ) = (a^2 + b^2).sqrt)
  (asymptote : b/a = Real.sqrt 3) :
  ∀ x y : ℝ, x^2 - y^2/3 = 1 ↔ x^2/a^2 - y^2/b^2 = 1 :=
sorry

end hyperbola_equation_l2824_282407


namespace evelyns_marbles_l2824_282453

theorem evelyns_marbles (initial_marbles : ℕ) : 
  initial_marbles + 9 = 104 → initial_marbles = 95 := by
  sorry

end evelyns_marbles_l2824_282453


namespace concert_ticket_price_l2824_282490

theorem concert_ticket_price (student_price : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (student_tickets : ℕ) :
  student_price = 9 →
  total_tickets = 2000 →
  total_revenue = 20960 →
  student_tickets = 520 →
  ∃ (non_student_price : ℕ),
    non_student_price * (total_tickets - student_tickets) + student_price * student_tickets = total_revenue ∧
    non_student_price = 11 :=
by
  sorry

end concert_ticket_price_l2824_282490


namespace smallest_n_for_perfect_square_product_l2824_282474

/-- The set of integers from 70 to 70 + n, inclusive -/
def numberSet (n : ℕ) : Set ℤ :=
  {x | 70 ≤ x ∧ x ≤ 70 + n}

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (x : ℤ) : Prop :=
  ∃ y : ℤ, x = y * y

/-- Predicate to check if there exist two different numbers in the set whose product is a perfect square -/
def hasPerfectSquareProduct (n : ℕ) : Prop :=
  ∃ a b : ℤ, a ∈ numberSet n ∧ b ∈ numberSet n ∧ a ≠ b ∧ isPerfectSquare (a * b)

theorem smallest_n_for_perfect_square_product : 
  (∀ m : ℕ, m < 28 → ¬hasPerfectSquareProduct m) ∧ hasPerfectSquareProduct 28 :=
sorry

end smallest_n_for_perfect_square_product_l2824_282474


namespace stewart_farm_sheep_count_l2824_282406

/-- The Stewart farm problem -/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
  sheep * 7 = horses * 6 →
  horses * 230 = 12880 →
  sheep = 48 :=
by
  sorry

end stewart_farm_sheep_count_l2824_282406


namespace correct_statements_count_l2824_282402

/-- Represents the correctness of a statement -/
inductive Correctness
| correct
| incorrect

/-- Evaluates the correctness of statement 1 -/
def statement1 : Correctness := Correctness.correct

/-- Evaluates the correctness of statement 2 -/
def statement2 : Correctness := Correctness.incorrect

/-- Evaluates the correctness of statement 3 -/
def statement3 : Correctness := Correctness.incorrect

/-- Evaluates the correctness of statement 4 -/
def statement4 : Correctness := Correctness.correct

/-- Counts the number of correct statements -/
def countCorrect (s1 s2 s3 s4 : Correctness) : Nat :=
  match s1, s2, s3, s4 with
  | Correctness.correct, Correctness.correct, Correctness.correct, Correctness.correct => 4
  | Correctness.correct, Correctness.correct, Correctness.correct, Correctness.incorrect => 3
  | Correctness.correct, Correctness.correct, Correctness.incorrect, Correctness.correct => 3
  | Correctness.correct, Correctness.correct, Correctness.incorrect, Correctness.incorrect => 2
  | Correctness.correct, Correctness.incorrect, Correctness.correct, Correctness.correct => 3
  | Correctness.correct, Correctness.incorrect, Correctness.correct, Correctness.incorrect => 2
  | Correctness.correct, Correctness.incorrect, Correctness.incorrect, Correctness.correct => 2
  | Correctness.correct, Correctness.incorrect, Correctness.incorrect, Correctness.incorrect => 1
  | Correctness.incorrect, Correctness.correct, Correctness.correct, Correctness.correct => 3
  | Correctness.incorrect, Correctness.correct, Correctness.correct, Correctness.incorrect => 2
  | Correctness.incorrect, Correctness.correct, Correctness.incorrect, Correctness.correct => 2
  | Correctness.incorrect, Correctness.correct, Correctness.incorrect, Correctness.incorrect => 1
  | Correctness.incorrect, Correctness.incorrect, Correctness.correct, Correctness.correct => 2
  | Correctness.incorrect, Correctness.incorrect, Correctness.correct, Correctness.incorrect => 1
  | Correctness.incorrect, Correctness.incorrect, Correctness.incorrect, Correctness.correct => 1
  | Correctness.incorrect, Correctness.incorrect, Correctness.incorrect, Correctness.incorrect => 0

theorem correct_statements_count :
  countCorrect statement1 statement2 statement3 statement4 = 2 := by
  sorry

end correct_statements_count_l2824_282402


namespace sin_symmetry_condition_l2824_282498

/-- A function f: ℝ → ℝ is symmetric about x = a if f(a + x) = f(a - x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem sin_symmetry_condition (φ : ℝ) :
  let f := fun x => Real.sin (x + φ)
  (f 0 = f π) ↔ SymmetricAbout f (π / 2) := by sorry

end sin_symmetry_condition_l2824_282498


namespace books_per_bookshelf_l2824_282482

theorem books_per_bookshelf (num_bookshelves : ℕ) (magazines_per_bookshelf : ℕ) (total_items : ℕ) : 
  num_bookshelves = 29 →
  magazines_per_bookshelf = 61 →
  total_items = 2436 →
  (total_items - num_bookshelves * magazines_per_bookshelf) / num_bookshelves = 23 := by
sorry

end books_per_bookshelf_l2824_282482


namespace area_triangle_AOC_l2824_282450

/-- Circle C with equation x^2 + y^2 - 4x - 6y + 12 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 12 = 0

/-- Point A with coordinates (3, 5) -/
def point_A : ℝ × ℝ := (3, 5)

/-- Origin O -/
def point_O : ℝ × ℝ := (0, 0)

/-- Point C is the center of the circle -/
def point_C : ℝ × ℝ := (2, 3)

/-- The area of triangle AOC is 1/2 -/
theorem area_triangle_AOC :
  let A := point_A
  let O := point_O
  let C := point_C
  (1/2 : ℝ) * ‖(A.1 - O.1, A.2 - O.2)‖ * ‖(C.1 - O.1, C.2 - O.2)‖ * 
    Real.sin (Real.arccos ((A.1 - O.1) * (C.1 - O.1) + (A.2 - O.2) * (C.2 - O.2)) / 
    (‖(A.1 - O.1, A.2 - O.2)‖ * ‖(C.1 - O.1, C.2 - O.2)‖)) = 1/2 := by
  sorry


end area_triangle_AOC_l2824_282450


namespace fuel_mixture_problem_l2824_282413

theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_A : ℝ) (ethanol_B : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 218 →
  ethanol_A = 0.12 →
  ethanol_B = 0.16 →
  total_ethanol = 30 →
  ∃ (V_A : ℝ), V_A = 122 ∧
    ∃ (V_B : ℝ), V_A + V_B = tank_capacity ∧
    ethanol_A * V_A + ethanol_B * V_B = total_ethanol :=
by sorry

end fuel_mixture_problem_l2824_282413


namespace imaginary_part_of_z_l2824_282460

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) : 
  z.im = (1 : ℝ) / 2 := by sorry

end imaginary_part_of_z_l2824_282460


namespace range_of_2x_minus_3_l2824_282412

theorem range_of_2x_minus_3 (x : ℝ) (h : -1 < 2*x + 3 ∧ 2*x + 3 < 1) :
  ∃! (n : ℤ), ∃ (y : ℝ), 2*y - 3 = ↑n ∧ -1 < 2*y + 3 ∧ 2*y + 3 < 1 :=
sorry

end range_of_2x_minus_3_l2824_282412


namespace walter_chores_l2824_282440

theorem walter_chores (total_days : ℕ) (total_earnings : ℕ) 
  (regular_pay : ℕ) (exceptional_pay : ℕ) :
  total_days = 15 →
  total_earnings = 47 →
  regular_pay = 3 →
  exceptional_pay = 4 →
  ∃ (regular_days exceptional_days : ℕ),
    regular_days + exceptional_days = total_days ∧
    regular_days * regular_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days = 2 :=
by sorry

end walter_chores_l2824_282440
