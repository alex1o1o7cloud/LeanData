import Mathlib

namespace solution_set_quadratic_inequality_l344_34436

theorem solution_set_quadratic_inequality :
  Set.Icc (-(1/2) : ℝ) 1 = {x : ℝ | 2 * x^2 - x ≤ 1} :=
by sorry

end solution_set_quadratic_inequality_l344_34436


namespace marcus_rachel_percentage_l344_34414

def marcus_score : ℕ := 5 * 3 + 10 * 2 + 8 * 1 + 2 * 4
def brian_score : ℕ := 6 * 3 + 8 * 2 + 9 * 1 + 1 * 4
def rachel_score : ℕ := 4 * 3 + 12 * 2 + 7 * 1 + 0 * 4
def team_total_score : ℕ := 150

theorem marcus_rachel_percentage :
  (marcus_score + rachel_score : ℚ) / team_total_score * 100 = 62.67 := by
  sorry

end marcus_rachel_percentage_l344_34414


namespace even_increasing_inequality_l344_34439

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

-- State the theorem
theorem even_increasing_inequality (h1 : is_even f) (h2 : increasing_on f 0 1) :
  f 0 < f (-0.5) ∧ f (-0.5) < f (-1) :=
sorry

end even_increasing_inequality_l344_34439


namespace fraction_is_one_ninth_l344_34438

/-- Represents a taxi trip with given parameters -/
structure TaxiTrip where
  initialFee : ℚ
  additionalChargePerFraction : ℚ
  totalDistance : ℚ
  totalCharge : ℚ

/-- Calculates the fraction of a mile for which the additional charge applies -/
def fractionOfMile (trip : TaxiTrip) : ℚ :=
  let additionalCharge := trip.totalCharge - trip.initialFee
  let numberOfFractions := additionalCharge / trip.additionalChargePerFraction
  trip.totalDistance / numberOfFractions

/-- Theorem stating that for the given trip parameters, the fraction of a mile
    for which the additional charge applies is 1/9 -/
theorem fraction_is_one_ninth :
  let trip := TaxiTrip.mk 2.25 0.15 3.6 3.60
  fractionOfMile trip = 1/9 := by
  sorry

end fraction_is_one_ninth_l344_34438


namespace happy_valley_kennel_arrangements_l344_34400

/-- The number of ways to arrange animals in cages. -/
def arrange_animals (num_chickens num_dogs num_cats : ℕ) : ℕ :=
  Nat.factorial 3 * Nat.factorial num_chickens * Nat.factorial num_dogs * Nat.factorial num_cats

/-- The theorem stating the number of arrangements for the given problem. -/
theorem happy_valley_kennel_arrangements :
  arrange_animals 3 3 4 = 5184 :=
by sorry

end happy_valley_kennel_arrangements_l344_34400


namespace stating_bus_passenger_count_l344_34481

/-- 
Calculates the final number of passengers on a bus given the initial number
and the changes at various stops.
-/
def final_passengers (initial : ℕ) (first_stop_on : ℕ) (other_stops_off : ℕ) (other_stops_on : ℕ) : ℕ :=
  initial + first_stop_on - other_stops_off + other_stops_on

/-- 
Theorem stating that given the specific passenger changes described in the problem,
the final number of passengers on the bus is 49.
-/
theorem bus_passenger_count : final_passengers 50 16 22 5 = 49 := by
  sorry

end stating_bus_passenger_count_l344_34481


namespace cuboid_length_calculation_l344_34444

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboid_surface_area (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: A cuboid with surface area 720, breadth 6, and height 10 has length 18.75 -/
theorem cuboid_length_calculation (l : ℝ) :
  cuboid_surface_area l 6 10 = 720 → l = 18.75 := by
  sorry

end cuboid_length_calculation_l344_34444


namespace rectangular_garden_width_l344_34429

theorem rectangular_garden_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 768 →
  width = 16 := by
sorry

end rectangular_garden_width_l344_34429


namespace entertainment_expense_calculation_l344_34426

def entertainment_expense (initial_amount : ℝ) (food_percentage : ℝ) (phone_percentage : ℝ) (final_amount : ℝ) : ℝ :=
  let food_expense := initial_amount * food_percentage
  let after_food := initial_amount - food_expense
  let phone_expense := after_food * phone_percentage
  let after_phone := after_food - phone_expense
  after_phone - final_amount

theorem entertainment_expense_calculation :
  entertainment_expense 200 0.60 0.25 40 = 20 := by
  sorry

end entertainment_expense_calculation_l344_34426


namespace chocolate_distribution_l344_34424

theorem chocolate_distribution (total_bars : ℕ) (num_people : ℕ) 
  (h1 : total_bars = 60) 
  (h2 : num_people = 5) : 
  let bars_per_person := total_bars / num_people
  let person1_final := bars_per_person - bars_per_person / 2
  let person2_final := bars_per_person + 2
  let person3_final := bars_per_person - 2
  let person4_final := bars_per_person
  person2_final + person3_final + person4_final = 36 := by
sorry

end chocolate_distribution_l344_34424


namespace triangle_area_l344_34460

theorem triangle_area (a b c : ℝ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) :
  (1 / 2) * a * b = 120 :=
by
  sorry

end triangle_area_l344_34460


namespace smallest_four_digit_in_pascal_l344_34449

/-- Pascal's triangle contains every positive integer -/
axiom pascal_contains_all_positive : ∀ n : ℕ, n > 0 → ∃ (row k : ℕ), Nat.choose row k = n

/-- The binomial coefficient function -/
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

/-- The smallest four-digit number -/
def smallest_four_digit : ℕ := 1000

/-- Theorem: 1000 is the smallest four-digit number in Pascal's triangle -/
theorem smallest_four_digit_in_pascal : 
  ∃ (row k : ℕ), binomial_coeff row k = smallest_four_digit ∧ 
  (∀ (r s : ℕ), binomial_coeff r s < smallest_four_digit → binomial_coeff r s < 1000) :=
sorry

end smallest_four_digit_in_pascal_l344_34449


namespace unfair_coin_expected_value_l344_34442

def coin_flip_expected_value (p_heads : ℚ) (p_tails : ℚ) (gain_heads : ℚ) (loss_tails : ℚ) : ℚ :=
  p_heads * gain_heads + p_tails * (-loss_tails)

theorem unfair_coin_expected_value :
  let p_heads : ℚ := 3/5
  let p_tails : ℚ := 2/5
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := 6
  coin_flip_expected_value p_heads p_tails gain_heads loss_tails = 3/5 :=
by
  sorry

end unfair_coin_expected_value_l344_34442


namespace chinese_paper_probability_l344_34480

/-- The number of Chinese exam papers in the bag -/
def chinese_papers : ℕ := 2

/-- The number of Tibetan exam papers in the bag -/
def tibetan_papers : ℕ := 3

/-- The number of English exam papers in the bag -/
def english_papers : ℕ := 1

/-- The total number of exam papers in the bag -/
def total_papers : ℕ := chinese_papers + tibetan_papers + english_papers

/-- The probability of drawing a Chinese exam paper -/
def prob_chinese : ℚ := chinese_papers / total_papers

theorem chinese_paper_probability : prob_chinese = 1 / 3 := by sorry

end chinese_paper_probability_l344_34480


namespace age_difference_james_jessica_prove_age_difference_l344_34486

/-- Given the ages and relationships of Justin, Jessica, and James, prove that James is 7 years older than Jessica. -/
theorem age_difference_james_jessica : ℕ → Prop :=
  fun age_difference =>
    ∀ (justin_age jessica_age james_age : ℕ),
      justin_age = 26 →
      jessica_age = justin_age + 6 →
      james_age > jessica_age →
      james_age + 5 = 44 →
      james_age - jessica_age = age_difference →
      age_difference = 7

/-- Proof of the theorem -/
theorem prove_age_difference : ∃ (age_difference : ℕ), age_difference_james_jessica age_difference := by
  sorry

end age_difference_james_jessica_prove_age_difference_l344_34486


namespace g_composition_result_l344_34468

-- Define the complex function g
noncomputable def g (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^3 else -z^3

-- State the theorem
theorem g_composition_result :
  g (g (g (g (1 + Complex.I)))) = -8192 - 45056 * Complex.I := by
  sorry

end g_composition_result_l344_34468


namespace intersection_equals_interval_l344_34484

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- Define the interval (1, 2]
def interval_one_two : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_equals_interval : M ∩ N = interval_one_two := by
  sorry

end intersection_equals_interval_l344_34484


namespace line_passes_through_fixed_point_k_range_below_x_axis_k_values_for_unit_area_l344_34413

-- Define the line equation
def line_equation (k x : ℝ) : ℝ := k * x + k - 1

-- Part 1: Prove that the line passes through (-1, -1) for all k
theorem line_passes_through_fixed_point (k : ℝ) :
  line_equation k (-1) = -1 := by sorry

-- Part 2: Prove the range of k when the line is below x-axis for -4 < x < 4
theorem k_range_below_x_axis (k : ℝ) :
  (∀ x, -4 < x ∧ x < 4 → line_equation k x < 0) ↔ -1/3 ≤ k ∧ k ≤ 1/5 := by sorry

-- Part 3: Prove the values of k when the triangle area is 1
theorem k_values_for_unit_area (k : ℝ) :
  (∃ x y, x > 0 ∧ y > 0 ∧ line_equation k x = 0 ∧ line_equation k 0 = y ∧ x * y / 2 = 1) ↔
  (k = 2 + Real.sqrt 3 ∨ k = 2 - Real.sqrt 3) := by sorry

end line_passes_through_fixed_point_k_range_below_x_axis_k_values_for_unit_area_l344_34413


namespace initial_storks_count_storks_on_fence_l344_34433

/-- Given a fence with birds and storks, prove the initial number of storks. -/
theorem initial_storks_count (initial_birds : ℕ) (additional_birds : ℕ) (stork_bird_difference : ℕ) : ℕ :=
  let final_birds := initial_birds + additional_birds
  let storks := final_birds + stork_bird_difference
  storks

/-- Prove that the number of storks initially on the fence is 6. -/
theorem storks_on_fence :
  initial_storks_count 2 3 1 = 6 := by
  sorry

end initial_storks_count_storks_on_fence_l344_34433


namespace expression_bounds_l344_34459

theorem expression_bounds (x y : ℝ) 
  (hx : -3 ≤ x ∧ x ≤ 3) (hy : -2 ≤ y ∧ y ≤ 2) : 
  -6 ≤ x * Real.sqrt (4 - y^2) + y * Real.sqrt (9 - x^2) ∧ 
  x * Real.sqrt (4 - y^2) + y * Real.sqrt (9 - x^2) ≤ 6 := by
  sorry

#check expression_bounds

end expression_bounds_l344_34459


namespace fruit_arrangement_count_l344_34410

def num_apples : ℕ := 4
def num_oranges : ℕ := 2
def num_bananas : ℕ := 2
def num_pears : ℕ := 1

def total_fruits : ℕ := num_apples + num_oranges + num_bananas + num_pears

theorem fruit_arrangement_count :
  (total_fruits.factorial) / (num_apples.factorial * num_oranges.factorial * num_bananas.factorial * num_pears.factorial) = 3780 := by
  sorry

end fruit_arrangement_count_l344_34410


namespace handshakes_for_seven_people_l344_34466

/-- The number of handshakes in a group of n people, where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- Theorem stating that the number of handshakes for 7 people is 21. -/
theorem handshakes_for_seven_people : handshakes 7 = 21 := by sorry

end handshakes_for_seven_people_l344_34466


namespace simple_interest_rate_l344_34479

/-- Simple interest calculation -/
theorem simple_interest_rate
  (principal : ℝ)
  (time : ℝ)
  (interest : ℝ)
  (h1 : principal = 10000)
  (h2 : time = 1)
  (h3 : interest = 900) :
  (interest / (principal * time)) * 100 = 9 := by
  sorry

end simple_interest_rate_l344_34479


namespace arithmetic_sequence_n_terms_l344_34470

def arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, i < j → j ≤ n → a j - a i = (j - i : ℝ) * (a 2 - a 1)

theorem arithmetic_sequence_n_terms
  (a : ℕ → ℝ) (n : ℕ)
  (h1 : arithmetic_sequence a n)
  (h2 : a 1 + a 2 + a 3 = 20)
  (h3 : a (n-2) + a (n-1) + a n = 130)
  (h4 : (Finset.range n).sum a = 200) :
  n = 8 := by
  sorry

end arithmetic_sequence_n_terms_l344_34470


namespace negation_of_universal_proposition_l344_34483

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 3 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l344_34483


namespace determinant_relations_l344_34415

theorem determinant_relations (a b c : ℤ) : ∃ (p₁ q₁ r₁ p₂ q₂ r₂ : ℤ),
  a = q₁ * r₂ - q₂ * r₁ ∧
  b = r₁ * p₂ - r₂ * p₁ ∧
  c = p₁ * q₂ - p₂ * q₁ := by
  sorry

end determinant_relations_l344_34415


namespace polynomial_identity_l344_34457

/-- The polynomial p(x) = x^2 - x + 1 -/
def p (x : ℂ) : ℂ := x^2 - x + 1

/-- α is a root of p(p(p(p(x)))) -/
def α : ℂ := sorry

theorem polynomial_identity :
  (p α - 1) * p α * p (p α) * p (p (p α)) = -1 := by sorry

end polynomial_identity_l344_34457


namespace no_integer_solution_cube_equation_l344_34412

theorem no_integer_solution_cube_equation :
  ¬ ∃ (x y z : ℤ), x^3 + y^3 = z^3 + 4 := by
sorry

end no_integer_solution_cube_equation_l344_34412


namespace inequality_proof_l344_34441

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l344_34441


namespace roberta_garage_sale_records_l344_34401

/-- The number of records Roberta bought at the garage sale -/
def records_bought_at_garage_sale (initial_records : ℕ) (gifted_records : ℕ) (days_per_record : ℕ) (total_listening_days : ℕ) : ℕ :=
  (total_listening_days / days_per_record) - (initial_records + gifted_records)

/-- Theorem stating that Roberta bought 30 records at the garage sale -/
theorem roberta_garage_sale_records : 
  records_bought_at_garage_sale 8 12 2 100 = 30 := by
  sorry

end roberta_garage_sale_records_l344_34401


namespace cashier_miscount_adjustment_l344_34497

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Calculates the error when miscounting one coin as another -/
def miscount_error (actual : String) (counted_as : String) : ℤ :=
  (coin_value counted_as : ℤ) - (coin_value actual : ℤ)

/-- Theorem: The net error and correct adjustment for x miscounted coins -/
theorem cashier_miscount_adjustment (x : ℕ) :
  let penny_as_nickel_error := miscount_error "penny" "nickel"
  let quarter_as_dime_error := miscount_error "quarter" "dime"
  let net_error := x * penny_as_nickel_error + x * quarter_as_dime_error
  let adjustment := -net_error
  (net_error = -11 * x) ∧ (adjustment = 11 * x) := by
  sorry

end cashier_miscount_adjustment_l344_34497


namespace cistern_filling_time_l344_34498

/-- Given a cistern with two taps, one that can fill it in 5 hours and another that can empty it in 6 hours,
    calculate the time it takes to fill the cistern when both taps are opened simultaneously. -/
theorem cistern_filling_time (fill_time empty_time : ℝ) (h_fill : fill_time = 5) (h_empty : empty_time = 6) :
  (fill_time * empty_time) / (empty_time - fill_time) = 30 := by
  sorry

#check cistern_filling_time

end cistern_filling_time_l344_34498


namespace player_A_wins_l344_34434

/-- Represents a card with a digit from 0 to 6 -/
inductive Card : Type
| zero | one | two | three | four | five | six

/-- Represents a player in the game -/
inductive Player : Type
| A | B

/-- Represents the state of the game -/
structure GameState :=
(remaining_cards : List Card)
(player_A_cards : List Card)
(player_B_cards : List Card)
(current_player : Player)

/-- Checks if a list of cards can form a number divisible by 17 -/
def can_form_divisible_by_17 (cards : List Card) : Bool :=
  sorry

/-- Determines the winner of the game given optimal play -/
def optimal_play_winner (initial_state : GameState) : Player :=
  sorry

/-- The main theorem stating that Player A wins with optimal play -/
theorem player_A_wins :
  ∀ (initial_state : GameState),
    initial_state.remaining_cards = [Card.zero, Card.one, Card.two, Card.three, Card.four, Card.five, Card.six] →
    initial_state.player_A_cards = [] →
    initial_state.player_B_cards = [] →
    initial_state.current_player = Player.A →
    optimal_play_winner initial_state = Player.A :=
  sorry

end player_A_wins_l344_34434


namespace frequency_not_exceeding_15_minutes_l344_34453

def duration_intervals : List (Real × Real) := [(0, 5), (5, 10), (10, 15), (15, 20)]
def frequencies : List Nat := [20, 16, 9, 5]

def total_calls : Nat := frequencies.sum

def calls_not_exceeding_15 : Nat := (frequencies.take 3).sum

theorem frequency_not_exceeding_15_minutes : 
  (calls_not_exceeding_15 : Real) / total_calls = 0.9 := by
  sorry

end frequency_not_exceeding_15_minutes_l344_34453


namespace prob_at_least_8_stay_correct_l344_34492

def total_people : ℕ := 10
def certain_people : ℕ := 5
def uncertain_people : ℕ := 5
def uncertain_stay_prob : ℚ := 3/7

def prob_at_least_8_stay : ℚ := 4563/16807

theorem prob_at_least_8_stay_correct :
  let prob_8_stay := (uncertain_people.choose 3) * (uncertain_stay_prob^3 * (1 - uncertain_stay_prob)^2)
  let prob_10_stay := uncertain_stay_prob^uncertain_people
  prob_at_least_8_stay = prob_8_stay + prob_10_stay :=
sorry

end prob_at_least_8_stay_correct_l344_34492


namespace inequality_solution_set_l344_34448

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo 0 2 : Set ℝ) = {x | |2*x - 1| < |x| + 1} :=
by sorry

end inequality_solution_set_l344_34448


namespace coloring_books_per_shelf_l344_34420

theorem coloring_books_per_shelf
  (initial_stock : ℕ)
  (books_sold : ℕ)
  (num_shelves : ℕ)
  (h1 : initial_stock = 86)
  (h2 : books_sold = 37)
  (h3 : num_shelves = 7)
  : (initial_stock - books_sold) / num_shelves = 7 :=
by
  sorry

end coloring_books_per_shelf_l344_34420


namespace west_notation_l344_34495

-- Define a type for distance with direction
inductive DirectedDistance
  | east (km : ℝ)
  | west (km : ℝ)

-- Define a function to convert DirectedDistance to a signed real number
def directedDistanceToSigned : DirectedDistance → ℝ
  | DirectedDistance.east km => km
  | DirectedDistance.west km => -km

-- State the theorem
theorem west_notation (d : ℝ) :
  directedDistanceToSigned (DirectedDistance.east 3) = 3 →
  directedDistanceToSigned (DirectedDistance.west 2) = -2 :=
by
  sorry

end west_notation_l344_34495


namespace final_alloy_mass_l344_34496

/-- Given two alloys with different copper percentages and their masses,
    prove that the total mass of the final alloy is the sum of the masses of the component alloys. -/
theorem final_alloy_mass
  (alloy1_copper_percent : ℚ)
  (alloy2_copper_percent : ℚ)
  (final_alloy_copper_percent : ℚ)
  (alloy1_mass : ℚ)
  (alloy2_mass : ℚ)
  (h1 : alloy1_copper_percent = 25 / 100)
  (h2 : alloy2_copper_percent = 50 / 100)
  (h3 : final_alloy_copper_percent = 45 / 100)
  (h4 : alloy1_mass = 200)
  (h5 : alloy2_mass = 800) :
  alloy1_mass + alloy2_mass = 1000 := by
  sorry

end final_alloy_mass_l344_34496


namespace brothers_book_pages_l344_34494

/-- Represents the number of pages read by a person in a week --/
structure WeeklyReading where
  total_pages : ℕ
  books_per_week : ℕ
  days_to_finish : ℕ

/-- Calculates the average pages read per day --/
def average_pages_per_day (r : WeeklyReading) : ℕ :=
  r.total_pages / r.days_to_finish

theorem brothers_book_pages 
  (ryan : WeeklyReading)
  (ryan_brother : WeeklyReading)
  (h1 : ryan.total_pages = 2100)
  (h2 : ryan.books_per_week = 5)
  (h3 : ryan.days_to_finish = 7)
  (h4 : ryan_brother.books_per_week = 7)
  (h5 : ryan_brother.days_to_finish = 7)
  (h6 : average_pages_per_day ryan = average_pages_per_day ryan_brother + 100) :
  ryan_brother.total_pages / ryan_brother.books_per_week = 200 :=
by sorry

end brothers_book_pages_l344_34494


namespace circular_table_arrangements_l344_34450

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem circular_table_arrangements (num_mathletes : ℕ) (num_coaches : ℕ) : 
  num_mathletes = 4 → num_coaches = 2 → 
  (factorial num_mathletes * 2) / 2 = 24 := by
  sorry

#check circular_table_arrangements

end circular_table_arrangements_l344_34450


namespace souvenir_purchase_theorem_l344_34485

/-- Represents a purchasing plan for souvenirs -/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a purchase plan is valid according to the given constraints -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.typeA + p.typeB = 60 ∧
  p.typeB ≤ 2 * p.typeA ∧
  100 * p.typeA + 60 * p.typeB ≤ 4500

/-- Calculates the cost of a purchase plan -/
def planCost (p : PurchasePlan) : ℕ :=
  100 * p.typeA + 60 * p.typeB

/-- The main theorem encompassing all parts of the problem -/
theorem souvenir_purchase_theorem :
  (∃! p : PurchasePlan, p.typeA + p.typeB = 60 ∧ planCost p = 4600) ∧
  (∃! plans : List PurchasePlan, plans.length = 3 ∧ 
    ∀ p ∈ plans, isValidPlan p ∧
    ∀ p, isValidPlan p → p ∈ plans) ∧
  (∃ p : PurchasePlan, isValidPlan p ∧
    ∀ q, isValidPlan q → planCost p ≤ planCost q ∧
    planCost p = 4400) := by
  sorry

#check souvenir_purchase_theorem

end souvenir_purchase_theorem_l344_34485


namespace sufficient_not_necessary_l344_34461

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

-- Define monotonicity on an interval
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

-- State the theorem
theorem sufficient_not_necessary (a : ℝ) :
  (a ≥ 2 → monotonic_on (f a) 1 2) ∧
  (∃ b : ℝ, b < 2 ∧ monotonic_on (f b) 1 2) :=
sorry

end sufficient_not_necessary_l344_34461


namespace center_is_eight_l344_34458

-- Define the type for our 3x3 grid
def Grid := Fin 3 → Fin 3 → Nat

-- Define what it means for two positions to share an edge
def sharesEdge (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

-- Define the property of consecutive numbers sharing an edge
def consecutiveShareEdge (g : Grid) : Prop :=
  ∀ (i j : Fin 3 × Fin 3), 
    g i.1 i.2 + 1 = g j.1 j.2 → sharesEdge i j

-- Define the sum of corner numbers
def cornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

-- Define the theorem
theorem center_is_eight (g : Grid) 
  (all_numbers : ∀ n : Fin 9, ∃ (i j : Fin 3), g i j = n.val + 1)
  (consec_edge : consecutiveShareEdge g)
  (corner_sum_20 : cornerSum g = 20) :
  g 1 1 = 8 := by
  sorry

end center_is_eight_l344_34458


namespace female_listeners_l344_34422

/-- Given a radio station survey with total listeners and male listeners,
    prove the number of female listeners. -/
theorem female_listeners (total_listeners male_listeners : ℕ) :
  total_listeners = 130 →
  male_listeners = 62 →
  total_listeners - male_listeners = 68 := by
  sorry

end female_listeners_l344_34422


namespace det_A_eq_11_l344_34421

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -2; 3, 1]

theorem det_A_eq_11 : A.det = 11 := by
  sorry

end det_A_eq_11_l344_34421


namespace point_outside_circle_l344_34471

/-- Given a circle with center O and radius 3, and a point P outside the circle,
    prove that the distance between O and P is greater than 3. -/
theorem point_outside_circle (O P : ℝ × ℝ) (r : ℝ) : 
  r = 3 →  -- The radius of the circle is 3
  (∀ Q : ℝ × ℝ, dist O Q = r → dist O P > dist O Q) →  -- P is outside the circle
  dist O P > 3  -- The distance between O and P is greater than 3
:= by sorry

end point_outside_circle_l344_34471


namespace matthew_ate_six_l344_34465

/-- The number of egg rolls eaten by Matthew, Patrick, and Alvin. -/
structure EggRolls where
  matthew : ℕ
  patrick : ℕ
  alvin : ℕ

/-- The conditions of the egg roll problem. -/
def egg_roll_conditions (e : EggRolls) : Prop :=
  e.matthew = 3 * e.patrick ∧
  e.patrick = e.alvin / 2 ∧
  e.alvin = 4

/-- The theorem stating that Matthew ate 6 egg rolls. -/
theorem matthew_ate_six (e : EggRolls) (h : egg_roll_conditions e) : e.matthew = 6 := by
  sorry

end matthew_ate_six_l344_34465


namespace product_of_exponents_l344_34408

theorem product_of_exponents (p r s : ℕ) : 
  4^p + 4^3 = 320 → 
  3^r + 27 = 54 → 
  2^5 + 7^s = 375 → 
  p * r * s = 36 := by
  sorry

end product_of_exponents_l344_34408


namespace function_relation_l344_34447

/-- Given functions f, g, and h from ℝ to ℝ satisfying certain conditions,
    prove that h can be expressed in terms of f and g. -/
theorem function_relation (f g h : ℝ → ℝ) 
    (hf : ∀ x, f x = (h (x + 1) + h (x - 1)) / 2)
    (hg : ∀ x, g x = (h (x + 4) + h (x - 4)) / 2) :
    ∀ x, h x = g x - f (x - 3) + f (x - 1) + f (x + 1) - f (x + 3) := by
  sorry

end function_relation_l344_34447


namespace product_of_solutions_l344_34489

theorem product_of_solutions (x : ℝ) : 
  (45 = -x^2 - 4*x) → (∃ α β : ℝ, α * β = -45 ∧ 45 = -α^2 - 4*α ∧ 45 = -β^2 - 4*β) :=
sorry

end product_of_solutions_l344_34489


namespace computer_accessories_cost_l344_34467

/-- Proves that the amount spent on computer accessories is $12 -/
theorem computer_accessories_cost (initial_amount : ℕ) (snack_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 48 →
  snack_cost = 8 →
  remaining_amount = initial_amount / 2 + 4 →
  initial_amount - (remaining_amount + snack_cost) = 12 :=
by
  sorry

end computer_accessories_cost_l344_34467


namespace necessary_not_sufficient_negation_l344_34417

theorem necessary_not_sufficient_negation (p q : Prop) :
  (q → p) ∧ ¬(p → q) → (¬p → ¬q) ∧ ¬(¬q → ¬p) := by
  sorry

end necessary_not_sufficient_negation_l344_34417


namespace lucy_earnings_l344_34411

/-- Calculates the earnings for a single 6-hour cycle -/
def cycle_earnings : ℕ := 1 + 2 + 3 + 4 + 5 + 6

/-- Calculates the earnings for the remaining hours after complete cycles -/
def remaining_earnings (hours : ℕ) : ℕ :=
  match hours with
  | 0 => 0
  | 1 => 1
  | 2 => 1 + 2
  | _ => 1 + 2 + 3

/-- Calculates the total earnings for a given number of hours -/
def total_earnings (hours : ℕ) : ℕ :=
  let complete_cycles := hours / 6
  let remaining_hours := hours % 6
  complete_cycles * cycle_earnings + remaining_earnings remaining_hours

/-- The theorem stating that Lucy's earnings for 45 hours of work is $153 -/
theorem lucy_earnings : total_earnings 45 = 153 := by
  sorry

end lucy_earnings_l344_34411


namespace exponent_division_l344_34499

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^5 / a^3 = a^2 := by
  sorry

end exponent_division_l344_34499


namespace xy_equation_solutions_l344_34491

theorem xy_equation_solutions (x y : ℤ) : x + y = x * y ↔ (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) := by
  sorry

end xy_equation_solutions_l344_34491


namespace johannes_earnings_today_l344_34443

/-- Represents the earnings and sales of a vegetable shop owner over three days -/
structure VegetableShopEarnings where
  cabbage_price : ℝ
  wednesday_earnings : ℝ
  friday_earnings : ℝ
  total_cabbage_sold : ℝ

/-- Calculates the earnings for today given the total earnings and previous days' earnings -/
def earnings_today (shop : VegetableShopEarnings) : ℝ :=
  shop.cabbage_price * shop.total_cabbage_sold - (shop.wednesday_earnings + shop.friday_earnings)

/-- Theorem stating that given the specific conditions, Johannes earned $42 today -/
theorem johannes_earnings_today :
  let shop : VegetableShopEarnings := {
    cabbage_price := 2,
    wednesday_earnings := 30,
    friday_earnings := 24,
    total_cabbage_sold := 48
  }
  earnings_today shop = 42 := by sorry

end johannes_earnings_today_l344_34443


namespace train_length_calculation_l344_34445

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length_calculation (speed : Real) (time : Real) : 
  speed = 144 ∧ time = 1.24990000799936 → 
  ∃ (length : Real), abs (length - 50) < 0.01 ∧ length = speed * time * (5 / 18) := by
  sorry

end train_length_calculation_l344_34445


namespace group_size_correct_l344_34428

/-- The number of members in the group -/
def n : ℕ := 93

/-- The total collection in paise -/
def total_paise : ℕ := 8649

/-- Theorem stating that n is the correct number of members -/
theorem group_size_correct : n * n = total_paise := by sorry

end group_size_correct_l344_34428


namespace exists_k_not_equal_f_diff_l344_34462

/-- f(n) is the largest integer k such that 2^k divides n -/
def f (n : ℕ) : ℕ := Nat.log2 (n.gcd (2^n))

/-- Theorem statement -/
theorem exists_k_not_equal_f_diff (n : ℕ) (h : n ≥ 2) (a : Fin n → ℕ)
  (h_sorted : ∀ i j, i < j → a i < a j) :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧
    ∀ i j : Fin n, j ≤ i → f (a i - a j) ≠ k :=
  sorry

end exists_k_not_equal_f_diff_l344_34462


namespace no_integer_square_diff_222_l344_34451

theorem no_integer_square_diff_222 : ¬ ∃ (a b : ℤ), a^2 - b^2 = 222 := by
  sorry

end no_integer_square_diff_222_l344_34451


namespace no_rain_probability_l344_34406

theorem no_rain_probability (p : ℚ) (h : p = 2/3) : (1 - p)^4 = 1/81 := by
  sorry

end no_rain_probability_l344_34406


namespace complement_union_theorem_l344_34493

def U : Set Nat := {0, 1, 2, 3, 5, 6, 8}
def A : Set Nat := {1, 5, 8}
def B : Set Nat := {2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end complement_union_theorem_l344_34493


namespace sum_first_105_remainder_l344_34456

theorem sum_first_105_remainder (n : Nat) (sum : Nat → Nat) : 
  n = 105 → 
  (∀ k, sum k = k * (k + 1) / 2) → 
  sum n % 1000 = 565 := by
sorry

end sum_first_105_remainder_l344_34456


namespace arctg_difference_bound_l344_34403

theorem arctg_difference_bound (a b : ℝ) : 
  |Real.arctan a - Real.arctan b| ≤ |b - a| := by sorry

end arctg_difference_bound_l344_34403


namespace ellipse_k_values_l344_34463

def ellipse_equation (x y k : ℝ) : Prop := x^2/5 + y^2/k = 1

def eccentricity (e : ℝ) : Prop := e = Real.sqrt 10 / 5

theorem ellipse_k_values (k : ℝ) :
  (∃ x y, ellipse_equation x y k) ∧ eccentricity (Real.sqrt 10 / 5) →
  k = 3 ∨ k = 25/3 := by
  sorry

end ellipse_k_values_l344_34463


namespace sequence_term_expression_l344_34440

def S (n : ℕ) : ℤ := 2 * n^2 - n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 2 else 4 * n - 3

theorem sequence_term_expression (n : ℕ) :
  n ≥ 1 → a n = S n - S (n - 1) :=
sorry

end sequence_term_expression_l344_34440


namespace dogs_food_average_l344_34464

theorem dogs_food_average (num_dogs : ℕ) (dog1_food : ℝ) (dog2_food : ℝ) (dog3_food : ℝ) :
  num_dogs = 3 →
  dog1_food = 13 →
  dog2_food = 2 * dog1_food →
  dog3_food = 6 →
  (dog1_food + dog2_food + dog3_food) / num_dogs = 15 := by
sorry

end dogs_food_average_l344_34464


namespace closest_estimate_l344_34472

-- Define the constants from the problem
def cars_observed : ℕ := 8
def observation_time : ℕ := 20
def delay_time : ℕ := 15
def total_time : ℕ := 210  -- 3 minutes and 30 seconds in seconds

-- Define the function to calculate the estimated number of cars
def estimate_cars : ℚ :=
  let rate : ℚ := cars_observed / observation_time
  let missed_cars : ℚ := rate * delay_time
  let observed_cars : ℚ := rate * (total_time - delay_time)
  missed_cars + observed_cars

-- Define the given options
def options : List ℕ := [120, 150, 210, 240, 280]

-- Theorem statement
theorem closest_estimate :
  ∃ (n : ℕ), n ∈ options ∧ 
  ∀ (m : ℕ), m ∈ options → |n - estimate_cars| ≤ |m - estimate_cars| ∧
  n = 120 := by
  sorry

end closest_estimate_l344_34472


namespace taco_truck_revenue_is_66_l344_34490

/-- Calculates the total revenue of a taco truck during lunch rush -/
def taco_truck_revenue (soft_taco_price hard_taco_price : ℕ)
  (family_soft_tacos family_hard_tacos : ℕ)
  (additional_customers : ℕ) : ℕ :=
  let total_soft_tacos := family_soft_tacos + 2 * additional_customers
  let soft_taco_revenue := soft_taco_price * total_soft_tacos
  let hard_taco_revenue := hard_taco_price * family_hard_tacos
  soft_taco_revenue + hard_taco_revenue

/-- The total revenue of the taco truck during lunch rush is $66 -/
theorem taco_truck_revenue_is_66 :
  taco_truck_revenue 2 5 3 4 10 = 66 := by
  sorry

end taco_truck_revenue_is_66_l344_34490


namespace circles_tangent_m_value_l344_34475

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

-- Define the tangency condition
def are_tangent (C₁ C₂ : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y ∧
  ∀ (x' y' : ℝ), C₁ x' y' ∧ C₂ x' y' → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_tangent_m_value :
  are_tangent C₁ (C₂ · · 9) → ∀ m : ℝ, are_tangent C₁ (C₂ · · m) → m = 9 :=
by sorry

end circles_tangent_m_value_l344_34475


namespace volume_of_rotated_region_l344_34416

/-- A region composed of unit squares resting along the x-axis and y-axis -/
structure Region :=
  (squares : ℕ)
  (along_x_axis : Bool)
  (along_y_axis : Bool)

/-- The volume of a solid formed by rotating a region about the y-axis -/
noncomputable def rotated_volume (r : Region) : ℝ :=
  sorry

/-- The problem statement -/
theorem volume_of_rotated_region :
  ∃ (r : Region),
    r.squares = 16 ∧
    r.along_x_axis = true ∧
    r.along_y_axis = true ∧
    rotated_volume r = 37 * Real.pi :=
  sorry

end volume_of_rotated_region_l344_34416


namespace sum_M_N_equals_two_l344_34455

/-- Definition of M -/
def M : ℚ := 1^5 + 2^4 * 3^3 - 4^2 / 5^1

/-- Definition of N -/
def N : ℚ := 1^5 - 2^4 * 3^3 + 4^2 / 5^1

/-- Theorem: The sum of M and N is equal to 2 -/
theorem sum_M_N_equals_two : M + N = 2 := by
  sorry

end sum_M_N_equals_two_l344_34455


namespace inequality_holds_l344_34454

theorem inequality_holds (p q : ℝ) (h_p : 0 < p) (h_p_upper : p < 2) (h_q : 0 < q) :
  (4 * (p * q^2 + 2 * p^2 * q + 2 * q^2 + 2 * p * q)) / (p + q) > 3 * p^2 * q :=
by sorry

end inequality_holds_l344_34454


namespace arithmetic_mean_problem_l344_34423

theorem arithmetic_mean_problem (a b c d : ℝ) : 
  (a + b + c + d + 106) / 5 = 92 →
  (a + b + c + d) / 4 = 88.5 :=
by sorry

end arithmetic_mean_problem_l344_34423


namespace chosen_number_proof_l344_34482

theorem chosen_number_proof (x : ℝ) : (x / 4) - 175 = 10 → x = 740 := by
  sorry

end chosen_number_proof_l344_34482


namespace cryptarithm_solution_l344_34469

/-- Represents a digit (1-9) -/
def Digit := { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

/-- Converts a two-digit number to its decimal representation -/
def twoDigitToNum (a b : Digit) : ℕ := 10 * a.val + b.val

/-- Converts a three-digit number with all digits the same to its decimal representation -/
def threeDigitSameToNum (c : Digit) : ℕ := 100 * c.val + 10 * c.val + c.val

theorem cryptarithm_solution :
  ∃! (a b c : Digit),
    a.val ≠ b.val ∧ b.val ≠ c.val ∧ a.val ≠ c.val ∧
    twoDigitToNum a b + a.val * threeDigitSameToNum c = 247 ∧
    a.val = 2 ∧ b.val = 5 ∧ c.val = 1 := by
  sorry

end cryptarithm_solution_l344_34469


namespace min_abs_z_l344_34407

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 2*I) + Complex.abs (z - (3 + 2*I)) = 7) :
  ∃ (z_min : ℂ), ∀ (w : ℂ), Complex.abs (w - 2*I) + Complex.abs (w - (3 + 2*I)) = 7 →
    Complex.abs z_min ≤ Complex.abs w ∧ Complex.abs z_min = 2 :=
sorry

end min_abs_z_l344_34407


namespace max_value_on_ellipse_l344_34473

theorem max_value_on_ellipse :
  ∀ x y : ℝ, (x^2 / 4 + y^2 / 9 = 1) → (2*x - y ≤ 5) ∧ ∃ x₀ y₀ : ℝ, (x₀^2 / 4 + y₀^2 / 9 = 1) ∧ (2*x₀ - y₀ = 5) :=
by sorry

end max_value_on_ellipse_l344_34473


namespace root_sum_squared_plus_three_times_root_l344_34437

theorem root_sum_squared_plus_three_times_root : ∀ (α β : ℝ), 
  (α^2 + 2*α - 2025 = 0) → 
  (β^2 + 2*β - 2025 = 0) → 
  (α^2 + 3*α + β = 2023) := by
  sorry

end root_sum_squared_plus_three_times_root_l344_34437


namespace digits_of_2_to_70_l344_34404

theorem digits_of_2_to_70 : ∃ n : ℕ, n = 22 ∧ (2^70 : ℕ) < 10^n ∧ 10^(n-1) ≤ 2^70 :=
  sorry

end digits_of_2_to_70_l344_34404


namespace boat_speed_in_still_water_l344_34476

/-- Proves that the speed of a boat in still water is 24 km/hr, given the speed of the stream and downstream travel information. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 168)
  (h3 : downstream_time = 6)
  : ∃ (boat_speed : ℝ), boat_speed = 24 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
sorry

end boat_speed_in_still_water_l344_34476


namespace range_of_T_l344_34487

theorem range_of_T (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x + y + z = 30) (h5 : 3 * x + y - z = 50) :
  let T := 5 * x + 4 * y + 2 * z
  ∃ (T_min T_max : ℝ), T_min = 120 ∧ T_max = 130 ∧ T_min ≤ T ∧ T ≤ T_max :=
by sorry

end range_of_T_l344_34487


namespace point_quadrant_theorem_l344_34431

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Definition of the third quadrant -/
def in_third_quadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A(x, y-2) is in the fourth quadrant, then B(y-2, -x) is in the third quadrant -/
theorem point_quadrant_theorem (x y : ℝ) :
  let A : Point2D := ⟨x, y - 2⟩
  let B : Point2D := ⟨y - 2, -x⟩
  in_fourth_quadrant A → in_third_quadrant B := by
  sorry

end point_quadrant_theorem_l344_34431


namespace sphere_surface_area_l344_34478

theorem sphere_surface_area (V : ℝ) (r : ℝ) (S : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  S = 4 * Real.pi * r^2 → 
  S = 36 * Real.pi * (4^(1/3)) :=
by sorry

end sphere_surface_area_l344_34478


namespace production_difference_formula_l344_34452

/-- The number of widgets David produces per hour on Monday -/
def w (t : ℝ) : ℝ := 2 * t

/-- The number of hours David works on Monday -/
def monday_hours (t : ℝ) : ℝ := t

/-- The number of hours David works on Tuesday -/
def tuesday_hours (t : ℝ) : ℝ := t - 1

/-- The number of widgets David produces per hour on Tuesday -/
def tuesday_rate (t : ℝ) : ℝ := w t + 5

/-- The difference in widget production between Monday and Tuesday -/
def production_difference (t : ℝ) : ℝ :=
  w t * monday_hours t - tuesday_rate t * tuesday_hours t

theorem production_difference_formula (t : ℝ) :
  production_difference t = -3 * t + 5 := by
  sorry

end production_difference_formula_l344_34452


namespace equation_solution_count_l344_34446

theorem equation_solution_count : ∃ (s : Finset ℕ),
  (∀ c ∈ s, c ≤ 1000) ∧ 
  (∀ c ∈ s, ∃ x : ℝ, 7 * ⌊x⌋ + 2 * ⌈x⌉ = c) ∧
  (∀ c ≤ 1000, c ∉ s → ¬∃ x : ℝ, 7 * ⌊x⌋ + 2 * ⌈x⌉ = c) ∧
  s.card = 223 :=
by sorry

end equation_solution_count_l344_34446


namespace geometry_theorem_l344_34430

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane)
  (h_distinct_lines : m ≠ n)
  (h_distinct_planes : α ≠ β) :
  (∀ (m : Line) (β α : Plane), 
    subset m β → plane_parallel α β → parallel m α) ∧
  (∀ (m n : Line) (α β : Plane),
    perpendicular m α → perpendicular n β → plane_parallel α β → line_parallel m n) :=
by sorry

end geometry_theorem_l344_34430


namespace disjunction_is_true_l344_34409

-- Define the propositions p and q
def p : Prop := ∀ a b : ℝ, a > |b| → a^2 > b^2
def q : Prop := ∀ x : ℝ, x^2 = 4 → x = 2

-- State the theorem
theorem disjunction_is_true (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end disjunction_is_true_l344_34409


namespace triangle_properties_l344_34419

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  a / Real.sin A = 2 * Real.sqrt 3 ∧
  a * Real.sin C + Real.sqrt 3 * c * Real.cos A = 0 →
  a = 3 ∧
  (b + c = Real.sqrt 11 →
    1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) := by
  sorry

end triangle_properties_l344_34419


namespace equation_solutions_l344_34435

theorem equation_solutions :
  (∀ x : ℝ, (5 - 2*x)^2 - 16 = 0 ↔ (x = 1/2 ∨ x = 9/2)) ∧
  (∀ x : ℝ, 2*(x - 3) = x^2 - 9 ↔ (x = 3 ∨ x = -1)) := by
  sorry

end equation_solutions_l344_34435


namespace divisor_sum_condition_l344_34418

theorem divisor_sum_condition (n : ℕ+) :
  (∃ (a b c : ℕ+), a + b + c = n ∧ a ∣ b ∧ b ∣ c ∧ a < b ∧ b < c) ↔ 
  n ∉ ({1, 2, 3, 4, 5, 6, 8, 12, 24} : Set ℕ+) :=
by sorry

end divisor_sum_condition_l344_34418


namespace angle_A_value_perimeter_range_l344_34425

-- Define the triangle
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : a * Real.cos C + Real.sqrt 3 * Real.sin C - b - c = 0
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom angle_sum : A + B + C = Real.pi
axiom law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Part 1: Prove that A = π/3
theorem angle_A_value : A = Real.pi / 3 := by sorry

-- Part 2: Prove the perimeter range
theorem perimeter_range (h_acute : A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2) (h_c : c = 3) :
  (3 * Real.sqrt 3 + 9) / 2 < a + b + c ∧ a + b + c < 9 + 3 * Real.sqrt 3 := by sorry

end angle_A_value_perimeter_range_l344_34425


namespace reading_program_classes_l344_34488

/-- The number of classes in a school with a specific reading program. -/
def number_of_classes (s : ℕ) : ℕ :=
  if s = 0 then 0 else 1

theorem reading_program_classes (s : ℕ) (h : s > 0) :
  let books_per_student_per_year := 4 * 12
  let total_books_read := 48
  number_of_classes s = 1 ∧ s * books_per_student_per_year = total_books_read :=
by sorry

end reading_program_classes_l344_34488


namespace expected_pine_saplings_l344_34405

/-- Represents the number of pine saplings in a stratified sample. -/
def pine_saplings_in_sample (total_saplings : ℕ) (pine_saplings : ℕ) (sample_size : ℕ) : ℚ :=
  (pine_saplings : ℚ) / (total_saplings : ℚ) * (sample_size : ℚ)

/-- Theorem stating the expected number of pine saplings in the sample. -/
theorem expected_pine_saplings :
  pine_saplings_in_sample 30000 4000 150 = 20 := by
  sorry

end expected_pine_saplings_l344_34405


namespace interior_angles_integral_count_l344_34427

theorem interior_angles_integral_count : 
  (Finset.filter (fun n : ℕ => n > 2 ∧ (n - 2) * 180 % n = 0) (Finset.range 361)).card = 22 := by
  sorry

end interior_angles_integral_count_l344_34427


namespace two_numbers_problem_l344_34402

theorem two_numbers_problem (a b : ℝ) (h1 : a + b = 40) (h2 : a * b = 375) :
  |a - b| = 10 := by sorry

end two_numbers_problem_l344_34402


namespace pots_per_vertical_stack_l344_34474

theorem pots_per_vertical_stack (total_pots : ℕ) (num_shelves : ℕ) (sets_per_shelf : ℕ) : 
  total_pots = 60 → num_shelves = 4 → sets_per_shelf = 3 → 
  (total_pots / (num_shelves * sets_per_shelf) : ℕ) = 5 := by
  sorry

end pots_per_vertical_stack_l344_34474


namespace opposite_numbers_product_l344_34477

theorem opposite_numbers_product (x y : ℝ) : 
  (|x - 3| + |y + 1| = 0) → xy = -3 := by
sorry

end opposite_numbers_product_l344_34477


namespace inequality_proof_l344_34432

theorem inequality_proof (x : ℝ) : 
  -7 < x ∧ x < -0.775 → (x + Real.sqrt 3) / (x + 10) > (3*x + 2*Real.sqrt 3) / (2*x + 14) := by
  sorry

end inequality_proof_l344_34432
