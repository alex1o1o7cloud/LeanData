import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l2457_245713

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y) :
  (2 * x - 1 / x) * (y - 1 / y) = -2 * x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2457_245713


namespace NUMINAMATH_CALUDE_expression_evaluation_l2457_245740

theorem expression_evaluation :
  let x : ℚ := -2/5
  let y : ℚ := 2
  2 * (x^2 * y - 2 * x * y) - 3 * (x^2 * y - 3 * x * y) + x^2 * y = -4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2457_245740


namespace NUMINAMATH_CALUDE_video_game_players_l2457_245764

/-- The number of players who joined a video game -/
def players_joined : ℕ := 5

theorem video_game_players :
  let initial_players : ℕ := 4
  let lives_per_player : ℕ := 3
  let total_lives : ℕ := 27
  players_joined = (total_lives - initial_players * lives_per_player) / lives_per_player :=
by
  sorry

#check video_game_players

end NUMINAMATH_CALUDE_video_game_players_l2457_245764


namespace NUMINAMATH_CALUDE_rachel_reading_homework_l2457_245789

/-- The number of pages of reading homework Rachel had to complete -/
def reading_pages : ℕ := 15 - 2 - 10

/-- The total number of pages Rachel had to complete -/
def total_pages : ℕ := 15

/-- The number of pages of math homework Rachel had to complete -/
def math_pages : ℕ := 2

/-- The number of pages of biology homework Rachel had to complete -/
def biology_pages : ℕ := 10

theorem rachel_reading_homework :
  reading_pages = 3 ∧
  total_pages = math_pages + reading_pages + biology_pages :=
sorry

end NUMINAMATH_CALUDE_rachel_reading_homework_l2457_245789


namespace NUMINAMATH_CALUDE_percentage_before_break_l2457_245792

/-- Given a total number of pages and the number of pages to read after a break,
    calculate the percentage of pages that must be read before the break. -/
theorem percentage_before_break (total_pages : ℕ) (pages_after_break : ℕ) 
    (h1 : total_pages = 30) (h2 : pages_after_break = 9) : 
    (((total_pages - pages_after_break : ℚ) / total_pages) * 100 = 70) := by
  sorry

end NUMINAMATH_CALUDE_percentage_before_break_l2457_245792


namespace NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l2457_245753

/-- The length of a bridge that a train can cross, given the train's length, speed, and time to cross. -/
theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time_s : ℝ) : ℝ :=
let train_speed_ms := train_speed_kmh * (1000 / 3600)
let total_distance := train_speed_ms * crossing_time_s
total_distance - train_length

/-- Proof that a bridge is 227 meters long given specific conditions -/
theorem specific_bridge_length : 
  bridge_length 148 45 30 = 227 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l2457_245753


namespace NUMINAMATH_CALUDE_bruce_bought_five_crayons_l2457_245719

/-- Calculates the number of packs of crayons Bruce bought given the conditions of the problem. -/
def bruces_crayons (crayonPrice bookPrice calculatorPrice bagPrice totalMoney : ℕ) 
  (numBooks numCalculators numBags : ℕ) : ℕ :=
  let bookCost := numBooks * bookPrice
  let calculatorCost := numCalculators * calculatorPrice
  let bagCost := numBags * bagPrice
  let remainingMoney := totalMoney - bookCost - calculatorCost - bagCost
  remainingMoney / crayonPrice

/-- Theorem stating that Bruce bought 5 packs of crayons given the conditions of the problem. -/
theorem bruce_bought_five_crayons : 
  bruces_crayons 5 5 5 10 200 10 3 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bruce_bought_five_crayons_l2457_245719


namespace NUMINAMATH_CALUDE_food_for_six_days_is_87_l2457_245799

/-- Represents the daily food consumption for Joy's foster dogs -/
def daily_food_consumption : ℚ :=
  -- Mom's food
  (1.5 * 3) +
  -- First two puppies
  (2 * (1/2 * 3)) +
  -- Next two puppies
  (2 * (3/4 * 2)) +
  -- Last puppy
  (1 * 4)

/-- The total amount of food needed for 6 days -/
def total_food_for_six_days : ℚ := daily_food_consumption * 6

/-- Theorem stating that the total food needed for 6 days is 87 cups -/
theorem food_for_six_days_is_87 : total_food_for_six_days = 87 := by sorry

end NUMINAMATH_CALUDE_food_for_six_days_is_87_l2457_245799


namespace NUMINAMATH_CALUDE_total_bad_produce_l2457_245710

-- Define the number of carrots and tomatoes picked by each person
def vanessa_carrots : ℕ := 17
def vanessa_tomatoes : ℕ := 12
def mom_carrots : ℕ := 14
def mom_tomatoes : ℕ := 22
def brother_carrots : ℕ := 6
def brother_tomatoes : ℕ := 8

-- Define the number of good carrots and tomatoes
def good_carrots : ℕ := 28
def good_tomatoes : ℕ := 35

-- Define the total number of carrots and tomatoes picked
def total_carrots : ℕ := vanessa_carrots + mom_carrots + brother_carrots
def total_tomatoes : ℕ := vanessa_tomatoes + mom_tomatoes + brother_tomatoes

-- Define the number of bad carrots and tomatoes
def bad_carrots : ℕ := total_carrots - good_carrots
def bad_tomatoes : ℕ := total_tomatoes - good_tomatoes

-- Theorem to prove
theorem total_bad_produce : bad_carrots + bad_tomatoes = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_bad_produce_l2457_245710


namespace NUMINAMATH_CALUDE_no_power_of_three_and_five_l2457_245791

def sequence_v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * sequence_v (n + 1) - sequence_v n

theorem no_power_of_three_and_five :
  ∀ n : ℕ, ¬∃ (a b : ℕ+), sequence_v n = 3^(a:ℕ) * 5^(b:ℕ) := by
  sorry

end NUMINAMATH_CALUDE_no_power_of_three_and_five_l2457_245791


namespace NUMINAMATH_CALUDE_min_board_sum_with_hundred_ones_l2457_245733

/-- Represents the state of the board --/
structure BoardState where
  ones : ℕ
  tens : ℕ
  twentyFives : ℕ

/-- Defines the allowed operations on the board --/
inductive Operation
  | replaceOneWithTen
  | replaceTenWithOneAndTwentyFive
  | replaceTwentyFiveWithTwoTens

/-- Applies an operation to the board state --/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.replaceOneWithTen => 
    { ones := state.ones - 1, tens := state.tens + 1, twentyFives := state.twentyFives }
  | Operation.replaceTenWithOneAndTwentyFive => 
    { ones := state.ones + 1, tens := state.tens - 1, twentyFives := state.twentyFives + 1 }
  | Operation.replaceTwentyFiveWithTwoTens => 
    { ones := state.ones, tens := state.tens + 2, twentyFives := state.twentyFives - 1 }

/-- Calculates the sum of all numbers on the board --/
def boardSum (state : BoardState) : ℕ :=
  state.ones + 10 * state.tens + 25 * state.twentyFives

/-- The main theorem to prove --/
theorem min_board_sum_with_hundred_ones : 
  ∃ (final : BoardState) (ops : List Operation),
    final.ones = 100 ∧
    (∀ (state : BoardState), 
      state.ones = 100 → boardSum state ≥ boardSum final) ∧
    boardSum final = 1370 := by
  sorry


end NUMINAMATH_CALUDE_min_board_sum_with_hundred_ones_l2457_245733


namespace NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l2457_245750

/-- The number of ways to distribute n distinct objects into two boxes,
    where box 1 contains at least k objects and box 2 contains at least m objects. -/
def distribute (n k m : ℕ) : ℕ :=
  (Finset.range (n - k - m + 1)).sum (λ i => Nat.choose n (k + i))

/-- Theorem: There are 25 ways to distribute 5 distinct objects into two boxes,
    where box 1 contains at least 1 object and box 2 contains at least 2 objects. -/
theorem distribute_five_balls_two_boxes : distribute 5 1 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l2457_245750


namespace NUMINAMATH_CALUDE_complement_union_is_empty_l2457_245724

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {2, 4, 5}

theorem complement_union_is_empty :
  (U \ (M ∪ N)) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_complement_union_is_empty_l2457_245724


namespace NUMINAMATH_CALUDE_max_value_function_l2457_245736

theorem max_value_function (t : ℝ) : (3^t - 4*t)*t / (9^t + t) ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_function_l2457_245736


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_value_l2457_245712

theorem quadratic_root_implies_u_value (u : ℝ) :
  ((-15 - Real.sqrt 145) / 6 : ℝ) ∈ {x : ℝ | 3 * x^2 + 15 * x + u = 0} →
  u = 20/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_value_l2457_245712


namespace NUMINAMATH_CALUDE_fraction_equality_l2457_245761

theorem fraction_equality (P Q : ℤ) : 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 4 ∧ x ≠ -4 → 
    P / (x^2 - 5*x) + Q / (x + 4) = (x^2 - 3*x + 8) / (x^3 - 5*x^2 + 4*x)) →
  (Q : ℚ) / (P : ℚ) = 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2457_245761


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l2457_245702

/-- The trajectory of the center of a moving circle -/
def trajectory_equation (x y : ℝ) : Prop :=
  (x - 5)^2 + (y + 7)^2 = 25

/-- The equation of the stationary circle -/
def stationary_circle (x y : ℝ) : Prop :=
  (x - 5)^2 + (y + 7)^2 = 16

/-- The radius of the moving circle -/
def moving_circle_radius : ℝ := 1

theorem trajectory_of_moving_circle :
  ∀ x y : ℝ,
  (∃ x₀ y₀ : ℝ, stationary_circle x₀ y₀ ∧ 
    ((x - x₀)^2 + (y - y₀)^2 = (moving_circle_radius + 4)^2)) →
  trajectory_equation x y :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l2457_245702


namespace NUMINAMATH_CALUDE_stating_num_distributions_eq_16_l2457_245759

/-- Represents the number of classes -/
def num_classes : ℕ := 4

/-- Represents the number of "Outstanding Class" spots -/
def num_outstanding_class : ℕ := 4

/-- Represents the number of "Outstanding Group Branch" spots -/
def num_outstanding_group : ℕ := 1

/-- Represents the total number of spots to be distributed -/
def total_spots : ℕ := num_outstanding_class + num_outstanding_group

/-- 
  Theorem stating that the number of ways to distribute the spots among classes,
  with each class receiving at least one spot, is equal to 16
-/
theorem num_distributions_eq_16 : 
  (Finset.univ.filter (fun f : Fin num_classes → Fin (total_spots + 1) => 
    (∀ i, f i > 0) ∧ (Finset.sum Finset.univ f = total_spots))).card = 16 := by
  sorry


end NUMINAMATH_CALUDE_stating_num_distributions_eq_16_l2457_245759


namespace NUMINAMATH_CALUDE_class_selection_combinations_l2457_245711

-- Define the number of total classes and classes to choose
def n : ℕ := 10
def r : ℕ := 4

-- Define the combination function
def combination (n r : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

-- State the theorem
theorem class_selection_combinations : combination n r = 210 := by
  sorry

end NUMINAMATH_CALUDE_class_selection_combinations_l2457_245711


namespace NUMINAMATH_CALUDE_delores_purchase_shortage_delores_specific_shortage_l2457_245747

/-- Calculates the amount Delores is short by after attempting to purchase a computer, printer, and table -/
theorem delores_purchase_shortage (initial_amount : ℝ) (computer_price : ℝ) (computer_discount : ℝ)
  (printer_price : ℝ) (printer_tax : ℝ) (table_price_euros : ℝ) (exchange_rate : ℝ) : ℝ :=
  let computer_cost := computer_price * (1 - computer_discount)
  let printer_cost := printer_price * (1 + printer_tax)
  let table_cost := table_price_euros * exchange_rate
  let total_cost := computer_cost + printer_cost + table_cost
  total_cost - initial_amount

/-- Proves that Delores is short by $605 given the specific conditions -/
theorem delores_specific_shortage : 
  delores_purchase_shortage 450 1000 0.3 100 0.15 200 1.2 = 605 := by
  sorry


end NUMINAMATH_CALUDE_delores_purchase_shortage_delores_specific_shortage_l2457_245747


namespace NUMINAMATH_CALUDE_shortest_chain_no_self_intersections_l2457_245772

/-- A polygonal chain in a plane -/
structure PolygonalChain (n : ℕ) where
  points : Fin n → ℝ × ℝ
  
/-- The length of a polygonal chain -/
def length (chain : PolygonalChain n) : ℝ := sorry

/-- A polygonal chain has self-intersections -/
def has_self_intersections (chain : PolygonalChain n) : Prop := sorry

/-- A polygonal chain is the shortest among all chains connecting the same points -/
def is_shortest (chain : PolygonalChain n) : Prop := 
  ∀ other : PolygonalChain n, chain.points = other.points → length chain ≤ length other

/-- The shortest polygonal chain connecting n points in a plane has no self-intersections -/
theorem shortest_chain_no_self_intersections (n : ℕ) (chain : PolygonalChain n) :
  is_shortest chain → ¬ has_self_intersections chain :=
sorry

end NUMINAMATH_CALUDE_shortest_chain_no_self_intersections_l2457_245772


namespace NUMINAMATH_CALUDE_fundraising_total_donation_l2457_245703

def total_donation (days : ℕ) (initial_donors : ℕ) (initial_donation : ℕ) : ℕ :=
  let rec donation_sum (d : ℕ) (donors : ℕ) (avg_donation : ℕ) (acc : ℕ) : ℕ :=
    if d = 0 then acc
    else donation_sum (d - 1) (donors * 2) (avg_donation + 5) (acc + donors * avg_donation)
  donation_sum days initial_donors initial_donation 0

theorem fundraising_total_donation :
  total_donation 5 10 10 = 8000 :=
by sorry

end NUMINAMATH_CALUDE_fundraising_total_donation_l2457_245703


namespace NUMINAMATH_CALUDE_work_rate_problem_l2457_245742

theorem work_rate_problem (A B C : ℚ) 
  (h1 : A + B = 1/8)
  (h2 : B + C = 1/12)
  (h3 : A + B + C = 1/6) :
  A + C = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_work_rate_problem_l2457_245742


namespace NUMINAMATH_CALUDE_select_students_l2457_245723

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem select_students (num_boys num_girls : ℕ) (boys_selected girls_selected : ℕ) : 
  num_boys = 11 → num_girls = 10 → boys_selected = 2 → girls_selected = 3 →
  (choose num_girls girls_selected) * (choose num_boys boys_selected) = 6600 := by
  sorry

end NUMINAMATH_CALUDE_select_students_l2457_245723


namespace NUMINAMATH_CALUDE_new_plant_characteristics_l2457_245738

/-- Represents a plant with genetic characteristics -/
structure Plant where
  ploidy : Nat
  has_homologous_chromosomes : Bool
  can_form_fertile_gametes : Bool
  homozygosity : Option Bool

/-- Represents the process of obtaining new plants from treated corn -/
def obtain_new_plants (original : Plant) (colchicine_treated : Bool) (anther_culture : Bool) : Plant :=
  sorry

/-- Theorem stating the characteristics of new plants obtained from treated corn -/
theorem new_plant_characteristics 
  (original : Plant)
  (h_original_diploid : original.ploidy = 2)
  (h_colchicine_treated : Bool)
  (h_anther_culture : Bool) :
  let new_plant := obtain_new_plants original h_colchicine_treated h_anther_culture
  new_plant.ploidy = 1 ∧ 
  new_plant.has_homologous_chromosomes = true ∧
  new_plant.can_form_fertile_gametes = true ∧
  new_plant.homozygosity = none :=
by sorry

end NUMINAMATH_CALUDE_new_plant_characteristics_l2457_245738


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l2457_245731

-- Define the number of available toppings
def n : ℕ := 9

-- Define the number of toppings to choose
def k : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem to prove
theorem pizza_toppings_combinations :
  combination n k = 84 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l2457_245731


namespace NUMINAMATH_CALUDE_distribution_difference_l2457_245781

theorem distribution_difference (total_amount : ℕ) (group1 : ℕ) (group2 : ℕ)
  (h1 : total_amount = 5040)
  (h2 : group1 = 14)
  (h3 : group2 = 18)
  (h4 : group1 < group2) :
  (total_amount / group1) - (total_amount / group2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_distribution_difference_l2457_245781


namespace NUMINAMATH_CALUDE_daniels_age_l2457_245744

theorem daniels_age (uncle_bob_age : ℕ) (elizabeth_age : ℕ) (daniel_age : ℕ) :
  uncle_bob_age = 60 →
  elizabeth_age = (2 * uncle_bob_age) / 3 →
  daniel_age = elizabeth_age - 10 →
  daniel_age = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_daniels_age_l2457_245744


namespace NUMINAMATH_CALUDE_range_of_a_when_p_and_q_false_l2457_245760

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 2

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃! x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (1/2) (3/2) → x^2 + 3*(a+1)*x + 2 ≤ 0

-- Theorem statement
theorem range_of_a_when_p_and_q_false :
  ∀ a : ℝ, ¬(p a ∧ q a) → a > -5/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_p_and_q_false_l2457_245760


namespace NUMINAMATH_CALUDE_weight_of_b_l2457_245706

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 41) :
  b = 27 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l2457_245706


namespace NUMINAMATH_CALUDE_vet_formula_portions_l2457_245709

/-- Calculates the total number of formula portions needed for puppies -/
def total_formula_portions (num_puppies : ℕ) (num_days : ℕ) (feedings_per_day : ℕ) : ℕ :=
  num_puppies * num_days * feedings_per_day

/-- Theorem: The vet gave Sandra 105 portions of formula for her puppies -/
theorem vet_formula_portions : total_formula_portions 7 5 3 = 105 := by
  sorry

end NUMINAMATH_CALUDE_vet_formula_portions_l2457_245709


namespace NUMINAMATH_CALUDE_abc_value_l2457_245777

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 30 * Real.sqrt 5)
  (hac : a * c = 45 * Real.sqrt 5)
  (hbc : b * c = 40 * Real.sqrt 5) :
  a * b * c = 300 * Real.sqrt 3 * (5 : ℝ) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_abc_value_l2457_245777


namespace NUMINAMATH_CALUDE_area_ratio_dodecagon_quadrilateral_l2457_245755

/-- A regular dodecagon -/
structure RegularDodecagon where
  -- We don't need to define the vertices explicitly
  area : ℝ

/-- A quadrilateral formed by connecting every third vertex of a regular dodecagon -/
structure Quadrilateral where
  area : ℝ

/-- The theorem stating the ratio of areas -/
theorem area_ratio_dodecagon_quadrilateral 
  (d : RegularDodecagon) 
  (q : Quadrilateral) : 
  q.area / d.area = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_dodecagon_quadrilateral_l2457_245755


namespace NUMINAMATH_CALUDE_equation_solution_l2457_245737

theorem equation_solution :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0 :=
by
  -- The unique value of x that satisfies the equation for all y is 3/2
  use 3/2
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2457_245737


namespace NUMINAMATH_CALUDE_proportional_relationship_l2457_245782

/-- Given that x is directly proportional to y^4, y is inversely proportional to z^2,
    and x = 4 when z = 3, prove that x = 3/192 when z = 6. -/
theorem proportional_relationship (x y z : ℝ) (k : ℝ) 
    (h1 : ∃ m : ℝ, x = m * y^4)
    (h2 : ∃ n : ℝ, y = n / z^2)
    (h3 : x = 4 ∧ z = 3 → x * z^8 = k)
    (h4 : x * z^8 = k) :
    z = 6 → x = 3 / 192 := by
  sorry

end NUMINAMATH_CALUDE_proportional_relationship_l2457_245782


namespace NUMINAMATH_CALUDE_factor_expression_l2457_245754

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2457_245754


namespace NUMINAMATH_CALUDE_mississippi_permutations_count_l2457_245774

def mississippi_permutations : ℕ :=
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)

theorem mississippi_permutations_count : mississippi_permutations = 34650 := by
  sorry

end NUMINAMATH_CALUDE_mississippi_permutations_count_l2457_245774


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l2457_245718

theorem inequality_holds_iff_m_in_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1 / (x + 1) + 4 / y = 1) :
  (∀ m : ℝ, x + y / 4 > m^2 - 5*m - 3) ↔ ∀ m : ℝ, -1 < m ∧ m < 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l2457_245718


namespace NUMINAMATH_CALUDE_largest_c_for_three_in_range_l2457_245765

/-- The function f(x) = x^2 - 7x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 7*x + c

/-- 3 is in the range of f -/
def three_in_range (c : ℝ) : Prop := ∃ x, f c x = 3

/-- The largest value of c such that 3 is in the range of f(x) = x^2 - 7x + c is 61/4 -/
theorem largest_c_for_three_in_range :
  (∃ c, three_in_range c ∧ ∀ c', three_in_range c' → c' ≤ c) ∧
  (∀ c, three_in_range c → c ≤ 61/4) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_three_in_range_l2457_245765


namespace NUMINAMATH_CALUDE_quadratic_trinomial_from_complete_square_l2457_245785

/-- 
Given a quadratic trinomial p(x) = Ax² + Bx + C, if its complete square form 
is x⁴ - 6x³ + 7x² + ax + b, then p(x) = x² - 3x - 1 or p(x) = -x² + 3x + 1.
-/
theorem quadratic_trinomial_from_complete_square (A B C a b : ℝ) :
  (∀ x, A * x^2 + B * x + C = x^4 - 6*x^3 + 7*x^2 + a*x + b) →
  ((A = 1 ∧ B = -3 ∧ C = -1) ∨ (A = -1 ∧ B = 3 ∧ C = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_from_complete_square_l2457_245785


namespace NUMINAMATH_CALUDE_baseball_cards_packs_l2457_245775

/-- The number of people who bought baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person bought -/
def cards_per_person : ℕ := 540

/-- The number of cards in each pack -/
def cards_per_pack : ℕ := 20

/-- The total number of packs of baseball cards for all people -/
def total_packs : ℕ := (num_people * cards_per_person) / cards_per_pack

theorem baseball_cards_packs : total_packs = 108 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_packs_l2457_245775


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l2457_245795

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ := 
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)

theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l2457_245795


namespace NUMINAMATH_CALUDE_fixed_fee_is_5_20_l2457_245784

/-- Represents a music streaming service with a fixed monthly fee and a per-song fee -/
structure StreamingService where
  fixedFee : ℝ
  perSongFee : ℝ

/-- Calculates the total bill for a given number of songs -/
def bill (s : StreamingService) (songs : ℕ) : ℝ :=
  s.fixedFee + s.perSongFee * songs

theorem fixed_fee_is_5_20 (s : StreamingService) :
  bill s 20 = 15.20 ∧ bill s 40 = 25.20 → s.fixedFee = 5.20 := by
  sorry

#check fixed_fee_is_5_20

end NUMINAMATH_CALUDE_fixed_fee_is_5_20_l2457_245784


namespace NUMINAMATH_CALUDE_fraction_comparison_l2457_245762

theorem fraction_comparison (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 3 →
  (5 * x + 2 > 8 - 3 * x) ↔ (x ∈ Set.Ioo (3/4 : ℝ) 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2457_245762


namespace NUMINAMATH_CALUDE_quadratic_function_sum_l2457_245779

/-- Given two quadratic functions f and g, prove that A + B = 0 under certain conditions -/
theorem quadratic_function_sum (A B : ℝ) (f g : ℝ → ℝ) : 
  A ≠ B →
  (∀ x, f x = A * x^2 + B) →
  (∀ x, g x = B * x^2 + A) →
  (∀ x, f (g x) - g (f x) = -A^2 + B^2) →
  A + B = 0 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_sum_l2457_245779


namespace NUMINAMATH_CALUDE_expression_simplification_l2457_245794

theorem expression_simplification (x y : ℝ) : 
  x^2*y - 3*x*y^2 + 2*y*x^2 - y^2*x = 3*x^2*y - 4*x*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2457_245794


namespace NUMINAMATH_CALUDE_largest_prime_factor_57_largest_prime_factor_57_is_19_l2457_245704

def numbers : List Nat := [57, 75, 91, 143, 169]

def largest_prime_factor (n : Nat) : Nat :=
  sorry

theorem largest_prime_factor_57 :
  ∀ n ∈ numbers, n ≠ 57 → largest_prime_factor n < largest_prime_factor 57 :=
  sorry

theorem largest_prime_factor_57_is_19 :
  largest_prime_factor 57 = 19 :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_57_largest_prime_factor_57_is_19_l2457_245704


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_iff_m_eq_7_or_neg_1_l2457_245739

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number k such that
    ax^2 + bx + c = (√a * x + k)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + k)^2

/-- The main theorem stating that m = 7 or m = -1 if and only if
    x^2 + 2(m-3)x + 16 is a perfect square trinomial -/
theorem perfect_square_trinomial_iff_m_eq_7_or_neg_1 :
  ∀ m : ℝ, (m = 7 ∨ m = -1) ↔ is_perfect_square_trinomial 1 (2*(m-3)) 16 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_iff_m_eq_7_or_neg_1_l2457_245739


namespace NUMINAMATH_CALUDE_age_problem_l2457_245715

/-- Given three people a, b, and c, where:
  * a is two years older than b
  * b is twice as old as c
  * The sum of their ages is 22
  Prove that b is 8 years old. -/
theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 22) : 
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l2457_245715


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l2457_245701

theorem infinite_geometric_series_ratio
  (a : ℝ)  -- first term
  (S : ℝ)  -- sum of the series
  (h1 : a = 328)
  (h2 : S = 2009)
  (h3 : S = a / (1 - r))  -- formula for sum of infinite geometric series
  : r = 41 / 49 :=
by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l2457_245701


namespace NUMINAMATH_CALUDE_largest_square_from_rectangle_l2457_245732

theorem largest_square_from_rectangle (width length : ℕ) 
  (h_width : width = 32) (h_length : length = 74) :
  ∃ (side : ℕ), side = Nat.gcd width length ∧ 
  side * (width / side) = width ∧ 
  side * (length / side) = length ∧
  ∀ (n : ℕ), n * (width / n) = width ∧ n * (length / n) = length → n ≤ side :=
sorry

end NUMINAMATH_CALUDE_largest_square_from_rectangle_l2457_245732


namespace NUMINAMATH_CALUDE_lucys_journey_l2457_245745

theorem lucys_journey (total : ℚ) 
  (h1 : total / 4 + 25 + total / 6 = total) : total = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_lucys_journey_l2457_245745


namespace NUMINAMATH_CALUDE_calculation_proof_l2457_245749

theorem calculation_proof : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2457_245749


namespace NUMINAMATH_CALUDE_occupancy_theorem_hundred_mathematicians_l2457_245768

/-- The number of ways k mathematicians can occupy k rooms under the given conditions -/
def occupancy_ways (k : ℕ) : ℕ :=
  2^(k - 1)

/-- Theorem stating that the number of ways k mathematicians can occupy k rooms is 2^(k-1) -/
theorem occupancy_theorem (k : ℕ) (h : k > 0) :
  occupancy_ways k = 2^(k - 1) :=
by sorry

/-- Corollary for the specific case of 100 mathematicians -/
theorem hundred_mathematicians :
  occupancy_ways 100 = 2^99 :=
by sorry

end NUMINAMATH_CALUDE_occupancy_theorem_hundred_mathematicians_l2457_245768


namespace NUMINAMATH_CALUDE_final_replacement_weight_is_140_l2457_245734

/-- The weight of the final replacement person in a series of replacements --/
def final_replacement_weight (initial_people : ℕ) (initial_weight : ℝ) 
  (first_increase : ℝ) (second_decrease : ℝ) (third_increase : ℝ) : ℝ :=
  let first_replacement := initial_weight + initial_people * first_increase
  let second_replacement := first_replacement - initial_people * second_decrease
  second_replacement + initial_people * third_increase - 
    (second_replacement - initial_people * second_decrease)

/-- Theorem stating the weight of the final replacement person --/
theorem final_replacement_weight_is_140 :
  final_replacement_weight 10 70 4 2 5 = 140 := by
  sorry


end NUMINAMATH_CALUDE_final_replacement_weight_is_140_l2457_245734


namespace NUMINAMATH_CALUDE_blanch_lunch_slices_l2457_245758

/-- The number of pizza slices Blanch ate during lunch -/
def lunch_slices (initial : ℕ) (breakfast : ℕ) (snack : ℕ) (dinner : ℕ) (remaining : ℕ) : ℕ :=
  initial - breakfast - snack - dinner - remaining

/-- Theorem stating that Blanch ate 2 slices during lunch -/
theorem blanch_lunch_slices :
  lunch_slices 15 4 2 5 2 = 2 := by sorry

end NUMINAMATH_CALUDE_blanch_lunch_slices_l2457_245758


namespace NUMINAMATH_CALUDE_probability_triangle_or_circle_l2457_245748

theorem probability_triangle_or_circle (total : ℕ) (triangles : ℕ) (circles : ℕ) (squares : ℕ)
  (h_total : total = 10)
  (h_triangles : triangles = 4)
  (h_circles : circles = 3)
  (h_squares : squares = 3)
  (h_sum : triangles + circles + squares = total) :
  (triangles + circles : ℚ) / total = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_triangle_or_circle_l2457_245748


namespace NUMINAMATH_CALUDE_jan_paid_288_dollars_l2457_245780

def roses_per_dozen : ℕ := 12
def dozen_bought : ℕ := 5
def cost_per_rose : ℚ := 6
def discount_percentage : ℚ := 80

def total_roses : ℕ := dozen_bought * roses_per_dozen

def full_price : ℚ := (total_roses : ℚ) * cost_per_rose

def discounted_price : ℚ := full_price * (discount_percentage / 100)

theorem jan_paid_288_dollars : discounted_price = 288 := by
  sorry

end NUMINAMATH_CALUDE_jan_paid_288_dollars_l2457_245780


namespace NUMINAMATH_CALUDE_twenty_four_shots_hit_ship_l2457_245729

/-- Represents a point on the grid -/
structure Point where
  x : Fin 10
  y : Fin 10

/-- Represents a 1x4 ship on the grid -/
structure Ship where
  start : Point
  horizontal : Bool

/-- The set of 24 shots -/
def shots : Set Point := sorry

/-- Predicate to check if a ship overlaps with a point -/
def shipOverlapsPoint (s : Ship) (p : Point) : Prop := sorry

theorem twenty_four_shots_hit_ship :
  ∀ s : Ship, ∃ p ∈ shots, shipOverlapsPoint s p := by sorry

end NUMINAMATH_CALUDE_twenty_four_shots_hit_ship_l2457_245729


namespace NUMINAMATH_CALUDE_certain_value_proof_l2457_245707

theorem certain_value_proof (x y : ℕ) : 
  x + y = 50 → x = 30 → y = 20 → 2 * (x - y) = 20 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l2457_245707


namespace NUMINAMATH_CALUDE_tangent_line_min_slope_l2457_245743

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 3

-- Theorem statement
theorem tangent_line_min_slope :
  ∃ (a b : ℝ), 
    (∀ x : ℝ, f' x ≥ f' a) ∧ 
    (∀ x : ℝ, f x = f a + f' a * (x - a)) ∧ 
    (b = -3 * a) ∧
    (∀ x : ℝ, f x = f a + b * (x - a)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_min_slope_l2457_245743


namespace NUMINAMATH_CALUDE_triangle_side_relation_l2457_245796

/-- Given a triangle ABC where the angles satisfy the equation 3α + 2β = 180°,
    prove that a^2 + bc = c^2, where a, b, and c are the lengths of the sides
    opposite to angles α, β, and γ respectively. -/
theorem triangle_side_relation (α β γ a b c : ℝ) : 
  0 < α ∧ 0 < β ∧ 0 < γ ∧ 
  α + β + γ = Real.pi ∧
  3 * α + 2 * β = Real.pi →
  a^2 + b * c = c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l2457_245796


namespace NUMINAMATH_CALUDE_regular_hexagon_diagonal_l2457_245788

theorem regular_hexagon_diagonal (side_length : ℝ) (h : side_length = 10) :
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 10 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_regular_hexagon_diagonal_l2457_245788


namespace NUMINAMATH_CALUDE_cylinder_height_ratio_l2457_245708

theorem cylinder_height_ratio (h : ℝ) (h_pos : h > 0) : 
  ∃ (H : ℝ), H = (14 / 15) * h ∧ 
  (7 / 8) * π * h = (3 / 5) * π * ((5 / 4) ^ 2) * H :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_ratio_l2457_245708


namespace NUMINAMATH_CALUDE_loan_amount_proof_l2457_245720

/-- Represents the interest rate as a decimal -/
def interest_rate : ℝ := 0.04

/-- Represents the loan duration in years -/
def years : ℕ := 2

/-- Calculates the compound interest amount after n years -/
def compound_interest (P : ℝ) : ℝ := P * (1 + interest_rate) ^ years

/-- Calculates the simple interest amount after n years -/
def simple_interest (P : ℝ) : ℝ := P * (1 + interest_rate * years)

/-- The difference between compound and simple interest -/
def interest_difference : ℝ := 10.40

theorem loan_amount_proof (P : ℝ) : 
  compound_interest P - simple_interest P = interest_difference → P = 6500 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_proof_l2457_245720


namespace NUMINAMATH_CALUDE_circle_center_l2457_245757

/-- The equation of a circle in the form x^2 - 6x + y^2 + 2y = 9 -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y = 9

/-- The center of a circle -/
def CircleCenter (h k : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x - h)^2 + (y - k)^2 = 19

/-- Theorem: The center of the circle with equation x^2 - 6x + y^2 + 2y = 9 is (3, -1) -/
theorem circle_center :
  CircleCenter 3 (-1) CircleEquation :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2457_245757


namespace NUMINAMATH_CALUDE_circle_theorem_sphere_theorem_l2457_245725

-- Define a circle and a sphere
def Circle : Type := Unit
def Sphere : Type := Unit

-- Define a point on a circle and a sphere
def PointOnCircle : Type := Unit
def PointOnSphere : Type := Unit

-- Define a semicircle and a hemisphere
def Semicircle : Type := Unit
def Hemisphere : Type := Unit

-- Define a function to check if a point is in a semicircle or hemisphere
def isIn : PointOnCircle → Semicircle → Prop := sorry
def isInHemisphere : PointOnSphere → Hemisphere → Prop := sorry

-- Theorem for the circle problem
theorem circle_theorem (c : Circle) (p1 p2 p3 p4 : PointOnCircle) :
  ∃ (s : Semicircle), (isIn p1 s ∧ isIn p2 s ∧ isIn p3 s) ∨
                      (isIn p1 s ∧ isIn p2 s ∧ isIn p4 s) ∨
                      (isIn p1 s ∧ isIn p3 s ∧ isIn p4 s) ∨
                      (isIn p2 s ∧ isIn p3 s ∧ isIn p4 s) :=
sorry

-- Theorem for the sphere problem
theorem sphere_theorem (s : Sphere) (p1 p2 p3 p4 p5 : PointOnSphere) :
  ∃ (h : Hemisphere), (isInHemisphere p1 h ∧ isInHemisphere p2 h ∧ isInHemisphere p3 h ∧ isInHemisphere p4 h) ∨
                      (isInHemisphere p1 h ∧ isInHemisphere p2 h ∧ isInHemisphere p3 h ∧ isInHemisphere p5 h) ∨
                      (isInHemisphere p1 h ∧ isInHemisphere p2 h ∧ isInHemisphere p4 h ∧ isInHemisphere p5 h) ∨
                      (isInHemisphere p1 h ∧ isInHemisphere p3 h ∧ isInHemisphere p4 h ∧ isInHemisphere p5 h) ∨
                      (isInHemisphere p2 h ∧ isInHemisphere p3 h ∧ isInHemisphere p4 h ∧ isInHemisphere p5 h) :=
sorry

end NUMINAMATH_CALUDE_circle_theorem_sphere_theorem_l2457_245725


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_l2457_245741

theorem right_triangle_acute_angle (α : ℝ) : 
  α > 0 ∧ α < 90 → -- α is an acute angle
  α + (α - 10) + 90 = 180 → -- sum of angles in the triangle
  α = 50 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_l2457_245741


namespace NUMINAMATH_CALUDE_tire_price_proof_l2457_245730

/-- The regular price of one tire -/
def regular_price : ℝ := 79

/-- The sale price of the fourth tire -/
def fourth_tire_price : ℝ := 3

/-- The total cost of four tires -/
def total_cost : ℝ := 240

theorem tire_price_proof :
  3 * regular_price + fourth_tire_price = total_cost :=
by sorry

end NUMINAMATH_CALUDE_tire_price_proof_l2457_245730


namespace NUMINAMATH_CALUDE_function_f_is_identity_l2457_245786

/-- A function satisfying the given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

/-- Theorem stating that the only function satisfying the conditions is the identity function -/
theorem function_f_is_identity (f : ℝ → ℝ) (hf : FunctionF f) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_f_is_identity_l2457_245786


namespace NUMINAMATH_CALUDE_volume_of_regular_triangular_truncated_pyramid_l2457_245797

/-- A regular triangular truncated pyramid -/
structure RegularTriangularTruncatedPyramid where
  /-- Height of the pyramid -/
  H : ℝ
  /-- Angle between lateral edge and base -/
  α : ℝ
  /-- H is positive -/
  H_pos : 0 < H
  /-- α is between 0 and π/2 -/
  α_range : 0 < α ∧ α < Real.pi / 2
  /-- H is the geometric mean between the sides of the bases -/
  H_is_geometric_mean : ∃ a b : ℝ, 0 < b ∧ b < a ∧ H^2 = a * b

/-- Volume of a regular triangular truncated pyramid -/
noncomputable def volume (p : RegularTriangularTruncatedPyramid) : ℝ :=
  (p.H^3 * Real.sqrt 3) / (4 * Real.sin p.α ^ 2)

/-- Theorem stating the volume of a regular triangular truncated pyramid -/
theorem volume_of_regular_triangular_truncated_pyramid (p : RegularTriangularTruncatedPyramid) :
  volume p = (p.H^3 * Real.sqrt 3) / (4 * Real.sin p.α ^ 2) := by sorry

end NUMINAMATH_CALUDE_volume_of_regular_triangular_truncated_pyramid_l2457_245797


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l2457_245778

theorem mixed_number_calculation : 
  (7 + 1/2 - (5 + 3/4)) * (3 + 1/6 + 2 + 1/8) = 9 + 25/96 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l2457_245778


namespace NUMINAMATH_CALUDE_tiling_cost_difference_l2457_245787

/-- Represents a tiling option with its cost per tile and labor cost per square foot -/
structure TilingOption where
  tileCost : ℕ
  laborCost : ℕ

/-- Calculates the total cost for a tiling option -/
def totalCost (option : TilingOption) (totalArea : ℕ) (tilesPerSqFt : ℕ) : ℕ :=
  option.tileCost * totalArea * tilesPerSqFt + option.laborCost * totalArea

theorem tiling_cost_difference :
  let turquoise := TilingOption.mk 13 6
  let purple := TilingOption.mk 11 8
  let orange := TilingOption.mk 15 5
  let totalArea := 5 * 8 + 7 * 8 + 6 * 9
  let tilesPerSqFt := 4
  let turquoiseCost := totalCost turquoise totalArea tilesPerSqFt
  let purpleCost := totalCost purple totalArea tilesPerSqFt
  let orangeCost := totalCost orange totalArea tilesPerSqFt
  max turquoiseCost (max purpleCost orangeCost) - min turquoiseCost (min purpleCost orangeCost) = 1950 := by
  sorry

end NUMINAMATH_CALUDE_tiling_cost_difference_l2457_245787


namespace NUMINAMATH_CALUDE_biased_die_expected_value_l2457_245766

/-- The expected value of winnings for a biased die roll -/
theorem biased_die_expected_value :
  let p_six : ℚ := 1/4  -- Probability of rolling a 6
  let p_other : ℚ := 3/4  -- Probability of rolling any other number
  let win_six : ℚ := 4  -- Winnings for rolling a 6
  let lose_other : ℚ := -1  -- Loss for rolling any other number
  p_six * win_six + p_other * lose_other = 1/4 := by
sorry

end NUMINAMATH_CALUDE_biased_die_expected_value_l2457_245766


namespace NUMINAMATH_CALUDE_jasmine_solution_concentration_l2457_245776

theorem jasmine_solution_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (added_jasmine : ℝ) 
  (added_water : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 100)
  (h2 : initial_concentration = 0.1)
  (h3 : added_jasmine = 5)
  (h4 : added_water = 67.5)
  (h5 : final_concentration = 0.08695652173913043) :
  let initial_jasmine := initial_volume * initial_concentration
  let total_jasmine := initial_jasmine + added_jasmine
  let final_volume := initial_volume + added_jasmine + added_water
  total_jasmine / final_volume = final_concentration :=
sorry

end NUMINAMATH_CALUDE_jasmine_solution_concentration_l2457_245776


namespace NUMINAMATH_CALUDE_stationery_ratio_is_three_to_one_l2457_245700

/-- The number of pieces of stationery Georgia has -/
def georgia_stationery : ℕ := 25

/-- The number of pieces of stationery Lorene has -/
def lorene_stationery : ℕ := georgia_stationery + 50

/-- The ratio of Lorene's stationery to Georgia's stationery -/
def stationery_ratio : ℚ := lorene_stationery / georgia_stationery

theorem stationery_ratio_is_three_to_one :
  stationery_ratio = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_stationery_ratio_is_three_to_one_l2457_245700


namespace NUMINAMATH_CALUDE_ship_length_l2457_245770

theorem ship_length (emily_step : ℝ) (ship_step : ℝ) :
  let emily_forward := 150
  let emily_backward := 70
  let wind_factor := 0.9
  let ship_length := 150 * emily_step - 150 * ship_step
  emily_backward * emily_step = ship_length - emily_backward * ship_step * wind_factor
  →
  ship_length = 19950 / 213 * emily_step :=
by sorry

end NUMINAMATH_CALUDE_ship_length_l2457_245770


namespace NUMINAMATH_CALUDE_problem_solution_l2457_245790

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 4*a| + |x|

-- Theorem statement
theorem problem_solution :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ a^2) →
  (-4 ≤ a ∧ a ≤ 4) ∧
  (∃ min_value : ℝ, min_value = 16/21 ∧
    ∀ x y z : ℝ, 4*x + 2*y + z = 4 →
      (x + y)^2 + y^2 + z^2 ≥ min_value) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2457_245790


namespace NUMINAMATH_CALUDE_distinct_subsets_remain_distinct_after_removal_l2457_245714

universe u

theorem distinct_subsets_remain_distinct_after_removal 
  {α : Type u} [DecidableEq α] (A : Finset α) (n : ℕ) 
  (subsets : Fin n → Finset α)
  (h_subset : ∀ i, (subsets i) ⊆ A)
  (h_distinct : ∀ i j, i ≠ j → subsets i ≠ subsets j) :
  ∃ a ∈ A, ∀ i j, i ≠ j → 
    (subsets i).erase a ≠ (subsets j).erase a :=
sorry

end NUMINAMATH_CALUDE_distinct_subsets_remain_distinct_after_removal_l2457_245714


namespace NUMINAMATH_CALUDE_james_car_rental_days_l2457_245746

/-- Calculates the number of days James rents his car per week -/
def days_rented_per_week (hourly_rate : ℕ) (hours_per_day : ℕ) (weekly_earnings : ℕ) : ℕ :=
  weekly_earnings / (hourly_rate * hours_per_day)

/-- Theorem stating that James rents his car for 4 days per week -/
theorem james_car_rental_days :
  days_rented_per_week 20 8 640 = 4 := by
  sorry

end NUMINAMATH_CALUDE_james_car_rental_days_l2457_245746


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l2457_245717

theorem quadratic_root_condition (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (∀ x : ℝ, x^2 + p*x + p - 1 = 0 ↔ x = x₁ ∨ x = x₂) ∧ 
    x₁^2 + x₁^3 = -(x₂^2 + x₂^3)) ↔ 
  p = 1 ∨ p = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l2457_245717


namespace NUMINAMATH_CALUDE_unique_valid_number_l2457_245705

def is_valid_number (n : ℕ) : Prop :=
  -- The number is six digits long
  100000 ≤ n ∧ n < 1000000 ∧
  -- It begins with digit 1
  n / 100000 = 1 ∧
  -- It ends with digit 7
  n % 10 = 7 ∧
  -- If the last digit is decreased by 1 and moved to the first place,
  -- the resulting number is five times the original number
  (6 * 100000 + n / 10) = 5 * n

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 142857 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2457_245705


namespace NUMINAMATH_CALUDE_max_y_over_x_on_circle_l2457_245726

theorem max_y_over_x_on_circle (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) : 
  ∃ (max : ℝ), (∀ (a b : ℝ), (a - 2)^2 + b^2 = 3 → b / a ≤ max) ∧ max = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_y_over_x_on_circle_l2457_245726


namespace NUMINAMATH_CALUDE_toms_profit_l2457_245751

/-- Calculate Tom's profit from the world's largest dough ball event -/
theorem toms_profit (flour_needed : ℕ) (flour_bag_size : ℕ) (flour_bag_cost : ℕ)
                    (salt_needed : ℕ) (salt_cost_per_pound : ℚ)
                    (promotion_cost : ℕ) (ticket_price : ℕ) (tickets_sold : ℕ) :
  flour_needed = 500 →
  flour_bag_size = 50 →
  flour_bag_cost = 20 →
  salt_needed = 10 →
  salt_cost_per_pound = 1/5 →
  promotion_cost = 1000 →
  ticket_price = 20 →
  tickets_sold = 500 →
  (tickets_sold * ticket_price : ℤ) - 
  (((flour_needed / flour_bag_size) * flour_bag_cost : ℕ) + 
   (salt_needed * salt_cost_per_pound).num + 
   promotion_cost : ℤ) = 8798 :=
by sorry

end NUMINAMATH_CALUDE_toms_profit_l2457_245751


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2457_245752

theorem quadratic_equation_roots (m : ℝ) :
  ((-1 : ℝ)^2 + m * (-1) - 5 = 0) →
  (m = -4 ∧ ∃ x₂ : ℝ, x₂ = 5 ∧ x₂^2 + m * x₂ - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2457_245752


namespace NUMINAMATH_CALUDE_high_school_nine_games_l2457_245727

/-- The number of teams in the league -/
def num_teams : ℕ := 9

/-- The number of non-league games each team plays -/
def non_league_games : ℕ := 6

/-- Calculate the total number of games in a season -/
def total_games : ℕ := 
  (num_teams * (num_teams - 1) / 2) + (num_teams * non_league_games)

/-- Theorem stating that the total number of games is 90 -/
theorem high_school_nine_games : total_games = 90 := by
  sorry

end NUMINAMATH_CALUDE_high_school_nine_games_l2457_245727


namespace NUMINAMATH_CALUDE_max_d_value_l2457_245728

/-- Represents a 6-digit number of the form 6d6,33f -/
def sixDigitNumber (d f : ℕ) : ℕ := 600000 + 10000*d + 3300 + f

/-- Predicate for d and f being single digits -/
def areSingleDigits (d f : ℕ) : Prop := d < 10 ∧ f < 10

/-- Predicate for the number being divisible by 33 -/
def isDivisibleBy33 (d f : ℕ) : Prop :=
  (sixDigitNumber d f) % 33 = 0

theorem max_d_value :
  ∃ (d : ℕ), 
    (∃ (f : ℕ), areSingleDigits d f ∧ isDivisibleBy33 d f) ∧
    (∀ (d' f' : ℕ), areSingleDigits d' f' → isDivisibleBy33 d' f' → d' ≤ d) ∧
    d = 1 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l2457_245728


namespace NUMINAMATH_CALUDE_shaded_area_is_36_l2457_245756

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square -/
structure Square :=
  (bottomLeft : Point)
  (sideLength : ℝ)

/-- Represents a right triangle -/
structure RightTriangle :=
  (bottomLeft : Point)
  (base : ℝ)
  (height : ℝ)

/-- Calculates the area of the shaded region -/
def shadedArea (square : Square) (triangle : RightTriangle) : ℝ :=
  sorry

/-- Theorem stating the area of the shaded region is 36 square units -/
theorem shaded_area_is_36 (square : Square) (triangle : RightTriangle) :
  square.bottomLeft = Point.mk 0 0 →
  square.sideLength = 12 →
  triangle.bottomLeft = Point.mk 12 0 →
  triangle.base = 12 →
  triangle.height = 12 →
  shadedArea square triangle = 36 :=
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_36_l2457_245756


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l2457_245763

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l2457_245763


namespace NUMINAMATH_CALUDE_min_distance_ellipse_to_Q_l2457_245793

/-- The ellipse with semi-major axis 4 and semi-minor axis 2 -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

/-- The point Q -/
def Q : ℝ × ℝ := (2, 0)

/-- The squared distance between two points -/
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem min_distance_ellipse_to_Q :
  ∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧
    (∀ (P' : ℝ × ℝ), ellipse P'.1 P'.2 →
      distance_squared P Q ≤ distance_squared P' Q) ∧
    distance_squared P Q = (2*Real.sqrt 6/3)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_to_Q_l2457_245793


namespace NUMINAMATH_CALUDE_impossible_relationships_l2457_245771

theorem impossible_relationships (a b : ℝ) (h : 1 / a = 1 / b) :
  ¬(0 < a ∧ a < b) ∧ ¬(b < a ∧ a < 0) := by
  sorry

end NUMINAMATH_CALUDE_impossible_relationships_l2457_245771


namespace NUMINAMATH_CALUDE_fraction_product_l2457_245769

theorem fraction_product : (2 : ℚ) / 9 * (-4 : ℚ) / 5 = (-8 : ℚ) / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l2457_245769


namespace NUMINAMATH_CALUDE_sandy_fish_count_l2457_245783

def final_fish_count (initial : ℕ) (bought : ℕ) (given_away : ℕ) (babies : ℕ) : ℕ :=
  initial + bought - given_away + babies

theorem sandy_fish_count :
  final_fish_count 26 6 10 15 = 37 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fish_count_l2457_245783


namespace NUMINAMATH_CALUDE_olivias_quarters_l2457_245773

theorem olivias_quarters (spent : ℕ) (left : ℕ) : spent = 4 → left = 7 → spent + left = 11 := by
  sorry

end NUMINAMATH_CALUDE_olivias_quarters_l2457_245773


namespace NUMINAMATH_CALUDE_laundry_charge_calculation_l2457_245716

/-- The amount charged per kilo of laundry -/
def charge_per_kilo : ℝ := sorry

/-- The number of kilos washed two days ago -/
def kilos_two_days_ago : ℝ := 5

/-- The number of kilos washed yesterday -/
def kilos_yesterday : ℝ := kilos_two_days_ago + 5

/-- The number of kilos washed today -/
def kilos_today : ℝ := 2 * kilos_yesterday

/-- The total earnings for three days -/
def total_earnings : ℝ := 70

theorem laundry_charge_calculation :
  charge_per_kilo * (kilos_two_days_ago + kilos_yesterday + kilos_today) = total_earnings ∧
  charge_per_kilo = 2 := by sorry

end NUMINAMATH_CALUDE_laundry_charge_calculation_l2457_245716


namespace NUMINAMATH_CALUDE_billy_ate_nine_apples_on_wednesday_l2457_245722

/-- The number of apples Billy ate each day of the week --/
structure WeeklyApples where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  total : ℕ

/-- Billy's apple consumption for the week satisfies the given conditions --/
def satisfiesConditions (w : WeeklyApples) : Prop :=
  w.monday = 2 ∧
  w.tuesday = 2 * w.monday ∧
  w.thursday = 4 * w.friday ∧
  w.friday = w.monday / 2 ∧
  w.total = 20 ∧
  w.total = w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- The theorem stating that Billy ate 9 apples on Wednesday --/
theorem billy_ate_nine_apples_on_wednesday (w : WeeklyApples) 
  (h : satisfiesConditions w) : w.wednesday = 9 := by
  sorry

end NUMINAMATH_CALUDE_billy_ate_nine_apples_on_wednesday_l2457_245722


namespace NUMINAMATH_CALUDE_interest_rate_satisfies_conditions_interest_rate_unique_solution_l2457_245735

/-- The principal amount -/
def P : ℝ := 6800.000000000145

/-- The time period in years -/
def t : ℝ := 2

/-- The difference between compound interest and simple interest -/
def diff : ℝ := 17

/-- The interest rate as a percentage -/
def r : ℝ := 5

/-- Theorem stating that the given interest rate satisfies the conditions -/
theorem interest_rate_satisfies_conditions :
  P * (1 + r / 100) ^ t - P - (P * r * t / 100) = diff := by
  sorry

/-- Theorem stating that the given interest rate is the unique solution -/
theorem interest_rate_unique_solution :
  ∀ x : ℝ, P * (1 + x / 100) ^ t - P - (P * x * t / 100) = diff → x = r := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_satisfies_conditions_interest_rate_unique_solution_l2457_245735


namespace NUMINAMATH_CALUDE_difference_even_prime_sums_l2457_245767

def sumFirstNEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

def sumFirstNPrimes (n : ℕ) : ℕ := sorry

theorem difference_even_prime_sums : 
  sumFirstNEvenNumbers 3005 - sumFirstNPrimes 3005 = 9039030 - sumFirstNPrimes 3005 := by
  sorry

end NUMINAMATH_CALUDE_difference_even_prime_sums_l2457_245767


namespace NUMINAMATH_CALUDE_max_x5y_given_constraint_l2457_245798

theorem max_x5y_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * (x + 2 * y) = 9) :
  x^5 * y ≤ 54 ∧ ∃ x0 y0 : ℝ, x0 > 0 ∧ y0 > 0 ∧ x0 * (x0 + 2 * y0) = 9 ∧ x0^5 * y0 = 54 :=
sorry

end NUMINAMATH_CALUDE_max_x5y_given_constraint_l2457_245798


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l2457_245721

theorem modulo_eleven_residue : (368 + 3 * 78 + 8 * 242 + 6 * 22) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l2457_245721
