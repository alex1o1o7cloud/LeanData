import Mathlib

namespace NUMINAMATH_CALUDE_garrison_provisions_theorem_l1426_142640

/-- Calculates the number of days provisions will last after reinforcement arrives -/
def daysProvisionsLast (initialMen : ℕ) (initialDays : ℕ) (reinforcementMen : ℕ) (daysPassed : ℕ) : ℕ :=
  let totalProvisions := initialMen * initialDays
  let remainingProvisions := totalProvisions - (initialMen * daysPassed)
  let totalMenAfterReinforcement := initialMen + reinforcementMen
  remainingProvisions / totalMenAfterReinforcement

/-- Theorem stating that given the specific conditions, provisions will last 10 more days -/
theorem garrison_provisions_theorem :
  daysProvisionsLast 2000 40 2000 20 = 10 := by
  sorry

#eval daysProvisionsLast 2000 40 2000 20

end NUMINAMATH_CALUDE_garrison_provisions_theorem_l1426_142640


namespace NUMINAMATH_CALUDE_roses_in_vase_l1426_142656

/-- The number of roses in a vase after adding more roses -/
def total_roses (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of roses is 18 given the initial and added amounts -/
theorem roses_in_vase : total_roses 10 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l1426_142656


namespace NUMINAMATH_CALUDE_king_hearts_diamonds_probability_l1426_142610

/-- The number of cards in a double deck -/
def total_cards : ℕ := 104

/-- The number of King of Hearts and King of Diamonds cards in a double deck -/
def target_cards : ℕ := 4

/-- The probability of drawing a King of Hearts or King of Diamonds from a shuffled double deck -/
def probability : ℚ := target_cards / total_cards

theorem king_hearts_diamonds_probability :
  probability = 1 / 26 := by sorry

end NUMINAMATH_CALUDE_king_hearts_diamonds_probability_l1426_142610


namespace NUMINAMATH_CALUDE_arun_speed_doubling_l1426_142651

/-- Proves that Arun takes 1 hour less than Anil when he doubles his speed -/
theorem arun_speed_doubling (distance : ℝ) (arun_speed : ℝ) (anil_time : ℝ) :
  distance = 30 →
  arun_speed = 5 →
  distance / arun_speed = anil_time + 2 →
  distance / (2 * arun_speed) = anil_time - 1 := by
  sorry

#check arun_speed_doubling

end NUMINAMATH_CALUDE_arun_speed_doubling_l1426_142651


namespace NUMINAMATH_CALUDE_solution_count_equals_divisors_of_square_l1426_142688

/-- 
Given a positive integer n, count_solutions n returns the number of
ordered pairs (x, y) of positive integers satisfying xy/(x+y) = n
-/
def count_solutions (n : ℕ+) : ℕ :=
  sorry

/--
Given a positive integer n, num_divisors_square n returns the number of
positive divisors of n²
-/
def num_divisors_square (n : ℕ+) : ℕ :=
  sorry

/--
For any positive integer n, the number of ordered pairs (x, y) of
positive integers satisfying xy/(x+y) = n is equal to the number of
positive divisors of n²
-/
theorem solution_count_equals_divisors_of_square (n : ℕ+) :
  count_solutions n = num_divisors_square n :=
by sorry

end NUMINAMATH_CALUDE_solution_count_equals_divisors_of_square_l1426_142688


namespace NUMINAMATH_CALUDE_point_not_in_second_quadrant_l1426_142622

theorem point_not_in_second_quadrant (m : ℝ) : 
  let P : ℝ × ℝ := (m + 1, m)
  ¬ (P.1 < 0 ∧ P.2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_second_quadrant_l1426_142622


namespace NUMINAMATH_CALUDE_centipede_dressing_sequences_l1426_142684

/-- The number of legs a centipede has -/
def num_legs : ℕ := 10

/-- The total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- The number of valid sequences for a centipede to wear its socks and shoes -/
def valid_sequences : ℕ := Nat.factorial total_items / (2^ num_legs)

/-- Theorem stating the number of valid sequences for a centipede to wear its socks and shoes -/
theorem centipede_dressing_sequences :
  valid_sequences = Nat.factorial total_items / (2 ^ num_legs) :=
by sorry

end NUMINAMATH_CALUDE_centipede_dressing_sequences_l1426_142684


namespace NUMINAMATH_CALUDE_power_of_sum_equals_225_l1426_142680

theorem power_of_sum_equals_225 : (3^2 + 6)^(4/2) = 225 := by sorry

end NUMINAMATH_CALUDE_power_of_sum_equals_225_l1426_142680


namespace NUMINAMATH_CALUDE_percent_employed_females_l1426_142666

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

end NUMINAMATH_CALUDE_percent_employed_females_l1426_142666


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1426_142632

theorem sum_of_solutions_is_zero (y : ℝ) (h1 : y = 8) (h2 : ∃ x : ℝ, x^2 + y^2 = 225) :
  ∃ x₁ x₂ : ℝ, x₁^2 + y^2 = 225 ∧ x₂^2 + y^2 = 225 ∧ x₁ + x₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1426_142632


namespace NUMINAMATH_CALUDE_profit_difference_theorem_l1426_142676

/-- Represents the profit distribution for a business partnership --/
structure ProfitDistribution where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  b_profit : ℕ

/-- Calculates the difference between profit shares of A and C --/
def profit_difference (pd : ProfitDistribution) : ℕ :=
  let total_ratio := pd.a_investment + pd.b_investment + pd.c_investment
  let unit_profit := pd.b_profit * total_ratio / pd.b_investment
  let a_profit := unit_profit * pd.a_investment / total_ratio
  let c_profit := unit_profit * pd.c_investment / total_ratio
  c_profit - a_profit

/-- Theorem stating the difference in profit shares --/
theorem profit_difference_theorem (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 8000)
  (h2 : pd.b_investment = 10000)
  (h3 : pd.c_investment = 12000)
  (h4 : pd.b_profit = 3000) :
  profit_difference pd = 1200 := by
  sorry

#eval profit_difference ⟨8000, 10000, 12000, 3000⟩

end NUMINAMATH_CALUDE_profit_difference_theorem_l1426_142676


namespace NUMINAMATH_CALUDE_percentage_difference_l1426_142699

theorem percentage_difference (A B C y : ℝ) : 
  C > A ∧ A > B ∧ B > 0 → 
  C = 2 * B → 
  A = C * (1 - y / 100) → 
  y = 100 - 50 * (A / B) :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1426_142699


namespace NUMINAMATH_CALUDE_conference_attendees_l1426_142620

theorem conference_attendees (total : ℕ) (creators : ℕ) (editors : ℕ) (y : ℕ) :
  total = 200 →
  creators = 80 →
  editors = 65 →
  total = creators + editors - y + 3 * y →
  y ≤ 27 ∧ ∃ (y : ℕ), y = 27 ∧ total = creators + editors - y + 3 * y :=
by sorry

end NUMINAMATH_CALUDE_conference_attendees_l1426_142620


namespace NUMINAMATH_CALUDE_cylinder_radius_l1426_142659

/-- 
Given a right circular cylinder with height h and diagonal d (measured from the center of the
circular base to the top edge of the cylinder), this theorem proves that when h = 12 and d = 13,
the radius r of the cylinder is 5.
-/
theorem cylinder_radius (h d : ℝ) (h_pos : h > 0) (d_pos : d > 0) 
  (h_val : h = 12) (d_val : d = 13) : ∃ r : ℝ, r > 0 ∧ r = 5 ∧ r^2 + h^2 = d^2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_l1426_142659


namespace NUMINAMATH_CALUDE_dvd_price_proof_l1426_142604

/-- The price Mike paid for the DVD at the store -/
def mike_price : ℝ := 5

/-- The price Steve paid for the DVD online -/
def steve_online_price (p : ℝ) : ℝ := 2 * p

/-- The shipping cost Steve paid -/
def steve_shipping_cost (p : ℝ) : ℝ := 0.8 * steve_online_price p

/-- The total amount Steve paid -/
def steve_total_cost (p : ℝ) : ℝ := steve_online_price p + steve_shipping_cost p

theorem dvd_price_proof :
  steve_total_cost mike_price = 18 :=
sorry

end NUMINAMATH_CALUDE_dvd_price_proof_l1426_142604


namespace NUMINAMATH_CALUDE_polygon_sides_l1426_142658

theorem polygon_sides (n : ℕ) : n > 2 →
  (n - 2) * 180 = 3 * 360 - 180 →
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l1426_142658


namespace NUMINAMATH_CALUDE_petya_final_vote_percentage_l1426_142643

theorem petya_final_vote_percentage 
  (x : ℝ) -- Total votes by noon
  (y : ℝ) -- Votes cast after noon
  (h1 : 0.45 * x = 0.27 * (x + y)) -- Vasya's final vote count
  (h2 : y = (2/3) * x) -- Relationship between x and y
  : (0.25 * x + y) / (x + y) = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_petya_final_vote_percentage_l1426_142643


namespace NUMINAMATH_CALUDE_nancy_money_l1426_142608

def five_dollar_bills : ℕ := 9
def ten_dollar_bills : ℕ := 4
def one_dollar_bills : ℕ := 7

def total_money : ℕ := five_dollar_bills * 5 + ten_dollar_bills * 10 + one_dollar_bills * 1

theorem nancy_money : total_money = 92 := by
  sorry

end NUMINAMATH_CALUDE_nancy_money_l1426_142608


namespace NUMINAMATH_CALUDE_class_size_proof_l1426_142697

theorem class_size_proof (total : ℕ) 
  (h1 : (3 : ℚ) / 5 * total + (1 : ℚ) / 5 * total + 10 = total) : total = 50 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l1426_142697


namespace NUMINAMATH_CALUDE_max_gross_profit_l1426_142606

/-- The gross profit function L(p) for a store selling goods --/
def L (p : ℝ) : ℝ := (8300 - 170*p - p^2)*(p - 20)

/-- The statement that L(p) achieves its maximum at p = 30 with a value of 23000 --/
theorem max_gross_profit :
  ∃ (p : ℝ), p > 0 ∧ L p = 23000 ∧ ∀ (q : ℝ), q > 0 → L q ≤ L p :=
sorry

end NUMINAMATH_CALUDE_max_gross_profit_l1426_142606


namespace NUMINAMATH_CALUDE_marbles_redistribution_l1426_142639

/-- The number of marbles Tyrone initially had -/
def tyrone_initial : ℕ := 120

/-- The number of marbles Eric initially had -/
def eric_initial : ℕ := 18

/-- The ratio of Tyrone's marbles to Eric's after redistribution -/
def final_ratio : ℕ := 3

/-- The number of marbles Tyrone gave to Eric -/
def marbles_given : ℚ := 16.5

theorem marbles_redistribution :
  let tyrone_final := tyrone_initial - marbles_given
  let eric_final := eric_initial + marbles_given
  tyrone_final = final_ratio * eric_final := by sorry

end NUMINAMATH_CALUDE_marbles_redistribution_l1426_142639


namespace NUMINAMATH_CALUDE_pipe_A_fill_time_l1426_142649

/-- The time it takes for pipe A to fill the cistern -/
def fill_time_A : ℝ := 10

/-- The time it takes for pipe B to empty the cistern -/
def empty_time_B : ℝ := 12

/-- The time it takes to fill the cistern with both pipes open -/
def fill_time_both : ℝ := 60

/-- Theorem stating that the fill time for pipe A is correct -/
theorem pipe_A_fill_time :
  fill_time_A = 10 ∧
  (1 / fill_time_A - 1 / empty_time_B = 1 / fill_time_both) :=
sorry

end NUMINAMATH_CALUDE_pipe_A_fill_time_l1426_142649


namespace NUMINAMATH_CALUDE_jimmy_earnings_theorem_l1426_142629

/-- Calculates Jimmy's total earnings from selling all his action figures --/
def jimmy_total_earnings : ℕ := by
  -- Define the number of each type of action figure
  let num_type_a : ℕ := 5
  let num_type_b : ℕ := 4
  let num_type_c : ℕ := 3

  -- Define the original value of each type of action figure
  let value_type_a : ℕ := 20
  let value_type_b : ℕ := 30
  let value_type_c : ℕ := 40

  -- Define the discount for each type of action figure
  let discount_type_a : ℕ := 7
  let discount_type_b : ℕ := 10
  let discount_type_c : ℕ := 12

  -- Calculate the selling price for each type of action figure
  let sell_price_a := value_type_a - discount_type_a
  let sell_price_b := value_type_b - discount_type_b
  let sell_price_c := value_type_c - discount_type_c

  -- Calculate the total earnings
  let total := num_type_a * sell_price_a + num_type_b * sell_price_b + num_type_c * sell_price_c

  exact total

/-- Theorem stating that Jimmy's total earnings is 229 --/
theorem jimmy_earnings_theorem : jimmy_total_earnings = 229 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_earnings_theorem_l1426_142629


namespace NUMINAMATH_CALUDE_cube_root_of_product_l1426_142681

theorem cube_root_of_product (a b c : ℕ) : 
  (2^9 * 5^3 * 7^3 : ℝ)^(1/3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l1426_142681


namespace NUMINAMATH_CALUDE_all_expressions_zero_l1426_142673

-- Define a 2D vector type
def Vector2D := ℝ × ℝ

-- Define vector addition
def add_vectors (v1 v2 : Vector2D) : Vector2D :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Define vector subtraction
def sub_vectors (v1 v2 : Vector2D) : Vector2D :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Define the zero vector
def zero_vector : Vector2D := (0, 0)

-- Define variables for each point
variable (A B C D E F O P Q : Vector2D)

-- Define the theorem
theorem all_expressions_zero : 
  (add_vectors (add_vectors (sub_vectors B A) (sub_vectors C B)) (sub_vectors A C) = zero_vector) ∧
  (add_vectors (sub_vectors (sub_vectors (sub_vectors B A) (sub_vectors C A)) (sub_vectors D B)) (sub_vectors D C) = zero_vector) ∧
  (sub_vectors (add_vectors (add_vectors (sub_vectors Q F) (sub_vectors P Q)) (sub_vectors F E)) (sub_vectors P E) = zero_vector) ∧
  (add_vectors (sub_vectors (sub_vectors A O) (sub_vectors B O)) (sub_vectors B A) = zero_vector) := by
  sorry

end NUMINAMATH_CALUDE_all_expressions_zero_l1426_142673


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l1426_142626

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.sin (40 * π / 180) =
  (Real.sin (50 * π / 180) * (Real.cos (60 * π / 180) * Real.cos (70 * π / 180) + Real.cos (20 * π / 180) * Real.cos (30 * π / 180))) /
  (Real.sin (40 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l1426_142626


namespace NUMINAMATH_CALUDE_rectangle_area_l1426_142615

/-- Given three similar rectangles where ABCD is the largest, prove its area --/
theorem rectangle_area (width height : ℝ) (h1 : width = 15) (h2 : height = width * Real.sqrt 6) :
  width * height = 75 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1426_142615


namespace NUMINAMATH_CALUDE_house_transactions_result_l1426_142619

/-- Represents the state of cash and house ownership for both Mr. A and Mr. B -/
structure State where
  a_cash : Int
  b_cash : Int
  a_has_house : Bool

/-- Represents a transaction between Mr. A and Mr. B -/
inductive Transaction
  | sell_to_b (price : Int)
  | buy_from_b (price : Int)

def initial_state : State := {
  a_cash := 12000,
  b_cash := 13000,
  a_has_house := true
}

def apply_transaction (s : State) (t : Transaction) : State :=
  match t with
  | Transaction.sell_to_b price =>
      { a_cash := s.a_cash + price,
        b_cash := s.b_cash - price,
        a_has_house := false }
  | Transaction.buy_from_b price =>
      { a_cash := s.a_cash - price,
        b_cash := s.b_cash + price,
        a_has_house := true }

def transactions : List Transaction := [
  Transaction.sell_to_b 14000,
  Transaction.buy_from_b 11000,
  Transaction.sell_to_b 15000
]

def final_state : State :=
  transactions.foldl apply_transaction initial_state

theorem house_transactions_result :
  final_state.a_cash = 30000 ∧ final_state.b_cash = -5000 := by
  sorry

end NUMINAMATH_CALUDE_house_transactions_result_l1426_142619


namespace NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l1426_142633

def charlie_cycle : ℕ := 6
def dana_cycle : ℕ := 7
def total_days : ℕ := 1000

def coinciding_rest_days (c_cycle d_cycle total : ℕ) : ℕ :=
  let lcm := Nat.lcm c_cycle d_cycle
  let full_cycles := total / lcm
  let c_rest_days := 2
  let d_rest_days := 2
  let coinciding_days_per_cycle := 4  -- This should be proven, not assumed
  full_cycles * coinciding_days_per_cycle

theorem coinciding_rest_days_theorem :
  coinciding_rest_days charlie_cycle dana_cycle total_days = 92 := by
  sorry

#eval coinciding_rest_days charlie_cycle dana_cycle total_days

end NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l1426_142633


namespace NUMINAMATH_CALUDE_min_value_x2_minus_xy_plus_y2_l1426_142609

theorem min_value_x2_minus_xy_plus_y2 :
  ∀ x y : ℝ, x^2 - x*y + y^2 ≥ 0 ∧ (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_x2_minus_xy_plus_y2_l1426_142609


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l1426_142682

theorem parallelepiped_volume (j : ℝ) :
  j > 0 →
  (abs (3 * (j^2 - 9) - 2 * (4*j - 15) + 2 * (12 - 5*j)) = 36) →
  j = (9 + Real.sqrt 585) / 6 := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l1426_142682


namespace NUMINAMATH_CALUDE_glove_selection_theorem_l1426_142685

/-- The number of pairs of gloves -/
def num_pairs : ℕ := 5

/-- The number of gloves to be selected -/
def num_selected : ℕ := 3

/-- The number of ways to select 3 gloves of different colors from 5 pairs of gloves -/
def num_selections : ℕ := 80

/-- Theorem stating that the number of ways to select 3 gloves of different colors
    from 5 pairs of gloves is equal to 80 -/
theorem glove_selection_theorem :
  (num_pairs.choose num_selected) * (2^num_selected) = num_selections :=
sorry

end NUMINAMATH_CALUDE_glove_selection_theorem_l1426_142685


namespace NUMINAMATH_CALUDE_negative_x_implies_positive_expression_l1426_142641

theorem negative_x_implies_positive_expression (x : ℝ) (h : x < 0) : -3 * x⁻¹ > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_implies_positive_expression_l1426_142641


namespace NUMINAMATH_CALUDE_jackfruit_division_l1426_142602

/-- Represents the fair division of jackfruits between Renato and Leandro -/
def fair_division (renato_watermelons leandro_watermelons marcelo_jackfruits : ℕ) 
  (renato_jackfruits leandro_jackfruits : ℕ) : Prop :=
  renato_watermelons = 30 ∧
  leandro_watermelons = 18 ∧
  marcelo_jackfruits = 24 ∧
  renato_jackfruits + leandro_jackfruits = marcelo_jackfruits ∧
  (renato_watermelons + leandro_watermelons) / 3 = 16 ∧
  renato_jackfruits * 2 = renato_watermelons ∧
  leandro_jackfruits * 2 = leandro_watermelons

theorem jackfruit_division :
  ∃ (renato_jackfruits leandro_jackfruits : ℕ),
    fair_division 30 18 24 renato_jackfruits leandro_jackfruits ∧
    renato_jackfruits = 15 ∧
    leandro_jackfruits = 9 := by
  sorry

end NUMINAMATH_CALUDE_jackfruit_division_l1426_142602


namespace NUMINAMATH_CALUDE_monomial_sum_equation_solution_l1426_142695

theorem monomial_sum_equation_solution :
  ∀ (a b : ℝ) (m n : ℕ),
  (∃ (k : ℝ), ∀ (a b : ℝ), (1/3 * a^m * b^3) + (-2 * a^2 * b^n) = k * a^m * b^n) →
  (∃ (x : ℝ), (x - 7) / n - (1 + x) / m = 1) →
  (∃ (x : ℝ), (x - 7) / n - (1 + x) / m = 1 ∧ x = -23) :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_equation_solution_l1426_142695


namespace NUMINAMATH_CALUDE_prove_depletion_rate_l1426_142691

-- Define the initial value of the machine
def initial_value : ℝ := 2500

-- Define the value of the machine after 2 years
def value_after_2_years : ℝ := 2256.25

-- Define the number of years
def years : ℝ := 2

-- Define the depletion rate
def depletion_rate : ℝ := 0.05

-- Theorem to prove that the given depletion rate is correct
theorem prove_depletion_rate : 
  value_after_2_years = initial_value * (1 - depletion_rate) ^ years := by
  sorry


end NUMINAMATH_CALUDE_prove_depletion_rate_l1426_142691


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1426_142665

def lunch_meat : ℕ := 12
def cheese : ℕ := 8

theorem sandwich_combinations : 
  (lunch_meat.choose 1) * (cheese.choose 2) = 336 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1426_142665


namespace NUMINAMATH_CALUDE_butterflies_fraction_l1426_142624

theorem butterflies_fraction (initial : ℕ) (remaining : ℕ) : 
  initial = 9 → remaining = 6 → (initial - remaining : ℚ) / initial = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_fraction_l1426_142624


namespace NUMINAMATH_CALUDE_interior_angle_regular_hexagon_l1426_142654

/-- The measure of an interior angle of a regular hexagon -/
def interior_angle_measure_hexagon : ℝ := 120

/-- A regular hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The measure of each interior angle in a regular hexagon is 120° -/
theorem interior_angle_regular_hexagon :
  interior_angle_measure_hexagon = (((hexagon_sides - 2 : ℕ) * 180) / hexagon_sides : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_interior_angle_regular_hexagon_l1426_142654


namespace NUMINAMATH_CALUDE_horner_method_v4_l1426_142655

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := x * v0 - 12
  let v2 := x * v1 + 60
  let v3 := x * v2 - 160
  x * v3 + 240

theorem horner_method_v4 :
  horner_v4 2 = 80 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v4_l1426_142655


namespace NUMINAMATH_CALUDE_license_plate_count_l1426_142628

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 3

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 3

/-- The number of possible positions for the letter block -/
def letter_block_positions : ℕ := 4

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * (num_digits ^ digits_in_plate) * (num_letters ^ letters_in_plate)

theorem license_plate_count : total_license_plates = 70304000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1426_142628


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_l1426_142679

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x^2)

theorem derivative_f_at_pi : 
  deriv f π = -1 / (π^2) := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_l1426_142679


namespace NUMINAMATH_CALUDE_book_distribution_l1426_142635

theorem book_distribution (x : ℕ) : 
  (∀ (total_books : ℕ), total_books = 9 * x + 7 → 
    (∀ (student : ℕ), student < x → ∃ (books : ℕ), books = 9)) ∧ 
  (∀ (total_books : ℕ), total_books ≤ 11 * x - 1 → 
    (∃ (student : ℕ), student < x ∧ ∀ (books : ℕ), books < 11)) →
  9 * x + 7 < 11 * x := by
sorry

end NUMINAMATH_CALUDE_book_distribution_l1426_142635


namespace NUMINAMATH_CALUDE_sum_of_two_positive_integers_greater_than_one_l1426_142694

theorem sum_of_two_positive_integers_greater_than_one (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b > 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_two_positive_integers_greater_than_one_l1426_142694


namespace NUMINAMATH_CALUDE_seventh_person_weight_l1426_142653

def elevator_problem (initial_people : ℕ) (initial_avg_weight : ℚ) (new_avg_weight : ℚ) : ℚ :=
  let total_initial_weight := initial_people * initial_avg_weight
  let total_new_weight := (initial_people + 1) * new_avg_weight
  total_new_weight - total_initial_weight

theorem seventh_person_weight :
  elevator_problem 6 160 151 = 97 := by sorry

end NUMINAMATH_CALUDE_seventh_person_weight_l1426_142653


namespace NUMINAMATH_CALUDE_equation_solution_l1426_142646

theorem equation_solution (x : ℝ) : (x + 1) ^ (x + 3) = 1 ↔ x = -3 ∨ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1426_142646


namespace NUMINAMATH_CALUDE_fourth_student_guess_l1426_142607

def jellybean_guess (first_guess : ℕ) : ℕ :=
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let average := (first_guess + second_guess + third_guess) / 3
  average + 25

theorem fourth_student_guess :
  jellybean_guess 100 = 525 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_guess_l1426_142607


namespace NUMINAMATH_CALUDE_jason_music_store_expense_l1426_142605

/-- The amount Jason spent at the music store -/
def jason_total_spent (flute_cost music_stand_cost song_book_cost : ℝ) : ℝ :=
  flute_cost + music_stand_cost + song_book_cost

/-- Theorem: Jason spent $158.35 at the music store -/
theorem jason_music_store_expense :
  jason_total_spent 142.46 8.89 7 = 158.35 := by
  sorry

end NUMINAMATH_CALUDE_jason_music_store_expense_l1426_142605


namespace NUMINAMATH_CALUDE_john_pill_schedule_l1426_142657

/-- The number of pills John takes per week -/
def pills_per_week : ℕ := 28

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of hours between each pill John takes -/
def hours_between_pills : ℚ :=
  (days_per_week * hours_per_day) / pills_per_week

theorem john_pill_schedule :
  hours_between_pills = 6 := by sorry

end NUMINAMATH_CALUDE_john_pill_schedule_l1426_142657


namespace NUMINAMATH_CALUDE_triangle_lines_theorem_l1426_142634

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  medianCM : ℝ → ℝ → ℝ
  altitudeBH : ℝ → ℝ → ℝ

/-- The given triangle satisfies the problem conditions -/
def given_triangle : Triangle where
  A := (5, 1)
  medianCM := λ x y => 2*x - y - 5
  altitudeBH := λ x y => x - 2*y - 5

/-- The equation of line BC -/
def line_BC (t : Triangle) : ℝ → ℝ → ℝ :=
  λ x y => 6*x - 5*y - 9

/-- The equation of the line symmetric to BC with respect to CM -/
def symmetric_line_BC (t : Triangle) : ℝ → ℝ → ℝ :=
  λ x y => 38*x - 9*y - 125

/-- Main theorem proving the equations of lines -/
theorem triangle_lines_theorem (t : Triangle) (h : t = given_triangle) :
  (line_BC t = λ x y => 6*x - 5*y - 9) ∧
  (symmetric_line_BC t = λ x y => 38*x - 9*y - 125) := by
  sorry

end NUMINAMATH_CALUDE_triangle_lines_theorem_l1426_142634


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1426_142621

theorem abs_sum_inequality (a b c d : ℝ) 
  (sum_pos : a + b + c + d > 0)
  (a_gt_c : a > c)
  (b_gt_d : b > d) :
  |a + b| > |c + d| := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1426_142621


namespace NUMINAMATH_CALUDE_max_expression_value_l1426_142636

def is_valid_assignment (O L I M P A D : ℕ) : Prop :=
  O ≠ L ∧ O ≠ I ∧ O ≠ M ∧ O ≠ P ∧ O ≠ A ∧ O ≠ D ∧
  L ≠ I ∧ L ≠ M ∧ L ≠ P ∧ L ≠ A ∧ L ≠ D ∧
  I ≠ M ∧ I ≠ P ∧ I ≠ A ∧ I ≠ D ∧
  M ≠ P ∧ M ≠ A ∧ M ≠ D ∧
  P ≠ A ∧ P ≠ D ∧
  A ≠ D ∧
  O < 10 ∧ L < 10 ∧ I < 10 ∧ M < 10 ∧ P < 10 ∧ A < 10 ∧ D < 10 ∧
  O ≠ 0 ∧ I ≠ 0

def expression_value (O L I M P A D : ℕ) : ℤ :=
  (10 * O + L) + (10 * I + M) - P + (10 * I + A) - (10 * D + A)

theorem max_expression_value :
  ∀ O L I M P A D : ℕ,
    is_valid_assignment O L I M P A D →
    expression_value O L I M P A D ≤ 263 :=
sorry

end NUMINAMATH_CALUDE_max_expression_value_l1426_142636


namespace NUMINAMATH_CALUDE_class_size_l1426_142644

theorem class_size (average_age : ℝ) (new_average : ℝ) (student_leave_age : ℝ) (teacher_age : ℝ)
  (h1 : average_age = 10)
  (h2 : new_average = 11)
  (h3 : student_leave_age = 11)
  (h4 : teacher_age = 41) :
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * average_age = (n : ℝ) * new_average - teacher_age + student_leave_age :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l1426_142644


namespace NUMINAMATH_CALUDE_project_hours_difference_l1426_142625

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 144) 
  (kate_hours pat_hours mark_hours : ℕ) 
  (h_pat_kate : pat_hours = 2 * kate_hours)
  (h_pat_mark : pat_hours * 3 = mark_hours)
  (h_sum : kate_hours + pat_hours + mark_hours = total_hours) :
  mark_hours - kate_hours = 80 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l1426_142625


namespace NUMINAMATH_CALUDE_garden_feet_is_117_l1426_142638

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

end NUMINAMATH_CALUDE_garden_feet_is_117_l1426_142638


namespace NUMINAMATH_CALUDE_simplify_expression_log_equation_result_l1426_142687

-- Part 1
theorem simplify_expression (x : ℝ) (h : x > 0) :
  (x - 1) / (x^(2/3) + x^(1/3) + 1) + (x + 1) / (x^(1/3) + 1) - (x - x^(1/3)) / (x^(1/3) - 1) = -x^(1/3) :=
sorry

-- Part 2
theorem log_equation_result (x : ℝ) (h1 : x > 0) (h2 : 3*x - 2 > 0) (h3 : 3*x + 2 > 0)
  (h4 : 2 * Real.log (3*x - 2) = Real.log x + Real.log (3*x + 2)) :
  Real.log (Real.sqrt (2 * Real.sqrt (2 * Real.sqrt 2))) / Real.log (Real.sqrt x) = 7/4 :=
sorry

end NUMINAMATH_CALUDE_simplify_expression_log_equation_result_l1426_142687


namespace NUMINAMATH_CALUDE_number_count_proof_l1426_142692

theorem number_count_proof (total_avg : ℝ) (group1_avg : ℝ) (group2_avg : ℝ) (group3_avg : ℝ) :
  total_avg = 3.95 →
  group1_avg = 4.2 →
  group2_avg = 3.85 →
  group3_avg = 3.8000000000000007 →
  (2 * group1_avg + 2 * group2_avg + 2 * group3_avg) / total_avg = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_number_count_proof_l1426_142692


namespace NUMINAMATH_CALUDE_floor_sum_of_positive_reals_l1426_142668

theorem floor_sum_of_positive_reals (u v w x : ℝ) 
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x)
  (h1 : u^2 + v^2 = 3005) (h2 : w^2 + x^2 = 3005)
  (h3 : u * w = 1729) (h4 : v * x = 1729) : 
  ⌊u + v + w + x⌋ = 155 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_of_positive_reals_l1426_142668


namespace NUMINAMATH_CALUDE_even_quadratic_function_l1426_142678

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem even_quadratic_function (a b c : ℝ) :
  (∀ x, (2 * a - 3 ≤ x ∧ x ≤ 1) → f a b c x = f a b c (-x)) →
  a = 1 ∧ b = 0 ∧ ∃ c : ℝ, True :=
by sorry

end NUMINAMATH_CALUDE_even_quadratic_function_l1426_142678


namespace NUMINAMATH_CALUDE_georgie_guacamole_servings_l1426_142693

/-- The number of avocados needed for one serving of guacamole -/
def avocados_per_serving : ℕ := 3

/-- The number of avocados Georgie initially has -/
def initial_avocados : ℕ := 5

/-- The number of avocados Georgie's sister buys -/
def sister_bought_avocados : ℕ := 4

/-- The total number of avocados Georgie has -/
def total_avocados : ℕ := initial_avocados + sister_bought_avocados

/-- The number of servings of guacamole Georgie can make -/
def servings_of_guacamole : ℕ := total_avocados / avocados_per_serving

theorem georgie_guacamole_servings : servings_of_guacamole = 3 := by
  sorry

end NUMINAMATH_CALUDE_georgie_guacamole_servings_l1426_142693


namespace NUMINAMATH_CALUDE_arithmetic_mean_property_l1426_142650

def consecutive_digits_set : List Nat := [1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789]

def arithmetic_mean (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

def digits (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 10) ((m % 10) :: acc)
    go n []

theorem arithmetic_mean_property :
  let M : Rat := arithmetic_mean consecutive_digits_set
  (M = 137174210) ∧
  (∀ d : Nat, d < 10 → (d ≠ 5 ↔ d ∈ digits M.num.toNat)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_property_l1426_142650


namespace NUMINAMATH_CALUDE_orange_balls_count_l1426_142611

theorem orange_balls_count :
  let total_balls : ℕ := 100
  let red_balls : ℕ := 30
  let blue_balls : ℕ := 20
  let yellow_balls : ℕ := 10
  let green_balls : ℕ := 5
  let pink_balls : ℕ := 2 * green_balls
  let orange_balls : ℕ := 3 * pink_balls
  red_balls + blue_balls + yellow_balls + green_balls + pink_balls + orange_balls = total_balls →
  orange_balls = 30 :=
by sorry

end NUMINAMATH_CALUDE_orange_balls_count_l1426_142611


namespace NUMINAMATH_CALUDE_profit_at_twenty_reduction_max_profit_at_fifteen_reduction_l1426_142618

-- Define the profit function
def profit_function (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

-- Theorem for part 1
theorem profit_at_twenty_reduction (x : ℝ) :
  x = 20 → profit_function x = 1200 := by sorry

-- Theorem for part 2
theorem max_profit_at_fifteen_reduction :
  ∃ (x : ℝ), x = 15 ∧ 
  profit_function x = 1250 ∧ 
  ∀ (y : ℝ), profit_function y ≤ profit_function x := by sorry

end NUMINAMATH_CALUDE_profit_at_twenty_reduction_max_profit_at_fifteen_reduction_l1426_142618


namespace NUMINAMATH_CALUDE_line_equation_correct_l1426_142612

-- Define the line passing through two points
def line_through_points (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

-- Define the equation of the line we want to prove
def line_equation (x y : ℝ) : Prop :=
  -x + y - 2 = 0

-- Theorem statement
theorem line_equation_correct :
  ∀ (x y : ℝ), line_through_points 3 2 1 4 x y ↔ line_equation x y :=
by sorry

end NUMINAMATH_CALUDE_line_equation_correct_l1426_142612


namespace NUMINAMATH_CALUDE_inequality_proof_l1426_142647

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt ((z + x) * (z + y)) - z ≥ Real.sqrt (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1426_142647


namespace NUMINAMATH_CALUDE_y_relationship_l1426_142683

theorem y_relationship : ∀ y₁ y₂ y₃ : ℝ,
  y₁ = (0.5 : ℝ) ^ (1/4 : ℝ) →
  y₂ = (0.6 : ℝ) ^ (1/4 : ℝ) →
  y₃ = (0.6 : ℝ) ^ (1/5 : ℝ) →
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_relationship_l1426_142683


namespace NUMINAMATH_CALUDE_shopkeeper_visits_l1426_142642

theorem shopkeeper_visits (initial_amount : ℚ) (spent_per_shop : ℚ) : initial_amount = 8.75 ∧ spent_per_shop = 10 →
  ∃ n : ℕ, n = 3 ∧ 2^n * initial_amount - spent_per_shop * (2^n - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_visits_l1426_142642


namespace NUMINAMATH_CALUDE_selling_price_range_l1426_142667

/-- Represents the daily sales revenue as a function of the selling price --/
def revenue (x : ℝ) : ℝ := x * (45 - 3 * (x - 15))

/-- The minimum selling price in yuan --/
def min_price : ℝ := 15

/-- The theorem stating the range of selling prices that generate over 600 yuan in daily revenue --/
theorem selling_price_range :
  {x : ℝ | revenue x > 600 ∧ x ≥ min_price} = Set.Icc 15 20 := by sorry

end NUMINAMATH_CALUDE_selling_price_range_l1426_142667


namespace NUMINAMATH_CALUDE_winning_number_correct_l1426_142603

/-- The number of callers needed to win all three prizes -/
def winning_number : ℕ := 1125

/-- The maximum allowed number of callers -/
def max_callers : ℕ := 2000

/-- Checks if a number is divisible by another number -/
def is_divisible (a b : ℕ) : Prop := b ∣ a

/-- Checks if a number is not divisible by 10 -/
def not_multiple_of_ten (n : ℕ) : Prop := ¬(is_divisible n 10)

/-- Theorem stating the winning number is correct -/
theorem winning_number_correct :
  (is_divisible winning_number 100) ∧ 
  (is_divisible winning_number 40) ∧ 
  (is_divisible winning_number 250) ∧
  (not_multiple_of_ten winning_number) ∧
  (∀ n : ℕ, n < winning_number → 
    ¬(is_divisible n 100 ∧ is_divisible n 40 ∧ is_divisible n 250 ∧ not_multiple_of_ten n)) ∧
  (winning_number ≤ max_callers) :=
sorry

end NUMINAMATH_CALUDE_winning_number_correct_l1426_142603


namespace NUMINAMATH_CALUDE_function_relationship_l1426_142660

/-- Given that y-m is directly proportional to 3x+6, where m is a constant,
    and that when x=2, y=4 and when x=3, y=7,
    prove that the function relationship between y and x is y = 3x - 2 -/
theorem function_relationship (m : ℝ) (k : ℝ) :
  (∀ x y, y - m = k * (3 * x + 6)) →
  (4 - m = k * (3 * 2 + 6)) →
  (7 - m = k * (3 * 3 + 6)) →
  ∀ x y, y = 3 * x - 2 := by
sorry


end NUMINAMATH_CALUDE_function_relationship_l1426_142660


namespace NUMINAMATH_CALUDE_cos_three_halves_lt_sin_one_tenth_l1426_142613

theorem cos_three_halves_lt_sin_one_tenth :
  Real.cos (3/2) < Real.sin (1/10) := by
  sorry

end NUMINAMATH_CALUDE_cos_three_halves_lt_sin_one_tenth_l1426_142613


namespace NUMINAMATH_CALUDE_new_jasmine_concentration_l1426_142663

/-- Calculates the new jasmine concentration after adding pure jasmine and water to a solution -/
theorem new_jasmine_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 80)
  (h2 : initial_concentration = 0.1)
  (h3 : added_jasmine = 5)
  (h4 : added_water = 15) :
  let initial_jasmine := initial_volume * initial_concentration
  let new_jasmine := initial_jasmine + added_jasmine
  let new_volume := initial_volume + added_jasmine + added_water
  new_jasmine / new_volume = 0.13 :=
by sorry

end NUMINAMATH_CALUDE_new_jasmine_concentration_l1426_142663


namespace NUMINAMATH_CALUDE_fraction_simplification_l1426_142627

theorem fraction_simplification : (5 * 8) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1426_142627


namespace NUMINAMATH_CALUDE_shortest_distance_correct_l1426_142623

/-- Represents the lengths of six lines meeting at a point -/
structure SixLines where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e ∧ e ≥ f
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0

/-- The shortest distance to draw all lines without lifting the pencil -/
def shortestDistance (lines : SixLines) : ℝ :=
  lines.a + 2 * (lines.b + lines.c + lines.d + lines.e + lines.f)

/-- Theorem stating that the shortest distance formula is correct -/
theorem shortest_distance_correct (lines : SixLines) :
  shortestDistance lines = lines.a + 2 * (lines.b + lines.c + lines.d + lines.e + lines.f) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_correct_l1426_142623


namespace NUMINAMATH_CALUDE_power_sum_equality_two_variables_power_sum_equality_three_variables_l1426_142675

-- Part (a)
theorem power_sum_equality_two_variables (x y u v : ℝ) (h1 : x + y = u + v) (h2 : x^2 + y^2 = u^2 + v^2) :
  ∀ n : ℕ, x^n + y^n = u^n + v^n := by sorry

-- Part (b)
theorem power_sum_equality_three_variables (x y z u v t : ℝ) 
  (h1 : x + y + z = u + v + t) 
  (h2 : x^2 + y^2 + z^2 = u^2 + v^2 + t^2) 
  (h3 : x^3 + y^3 + z^3 = u^3 + v^3 + t^3) :
  ∀ n : ℕ, x^n + y^n + z^n = u^n + v^n + t^n := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_two_variables_power_sum_equality_three_variables_l1426_142675


namespace NUMINAMATH_CALUDE_pauls_crayons_and_erasers_l1426_142601

theorem pauls_crayons_and_erasers 
  (initial_crayons : ℕ) 
  (initial_erasers : ℕ) 
  (final_crayons : ℕ) 
  (h1 : initial_crayons = 531)
  (h2 : initial_erasers = 38)
  (h3 : final_crayons = 391)
  (h4 : initial_erasers = final_erasers) :
  initial_crayons - final_crayons - initial_erasers = 102 :=
by sorry

end NUMINAMATH_CALUDE_pauls_crayons_and_erasers_l1426_142601


namespace NUMINAMATH_CALUDE_expression_simplification_l1426_142690

theorem expression_simplification (a b x y : ℝ) (h : b*x + a*y ≠ 0) :
  (b*x*(a^2*x^2 + 2*a^2*y^2 + b^2*y^2) + a*y*(a^2*x^2 + 2*b^2*x^2 + b^2*y^2)) / (b*x + a*y)
  = (a*x + b*y)^2 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l1426_142690


namespace NUMINAMATH_CALUDE_not_right_triangle_11_12_15_l1426_142661

/-- A function that checks if three numbers can form a right triangle -/
def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- Theorem stating that 11, 12, and 15 cannot form a right triangle -/
theorem not_right_triangle_11_12_15 : ¬ isRightTriangle 11 12 15 := by
  sorry

#check not_right_triangle_11_12_15

end NUMINAMATH_CALUDE_not_right_triangle_11_12_15_l1426_142661


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1426_142648

theorem cube_equation_solution (a d : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * d) : d = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1426_142648


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1426_142662

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (∀ k : ℕ, k * (k + 1) < 500 → k ≤ n) → 
  n * (n + 1) < 500 → 
  n + (n + 1) = 43 :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1426_142662


namespace NUMINAMATH_CALUDE_expression_equivalence_l1426_142600

theorem expression_equivalence : 
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) * 
  (5^32 + 7^32) * (5^64 + 7^64) * (5^128 + 7^128) = 7^256 - 5^256 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l1426_142600


namespace NUMINAMATH_CALUDE_square_greater_not_sufficient_nor_necessary_l1426_142671

theorem square_greater_not_sufficient_nor_necessary :
  ∃ (a b : ℝ), a^2 > b^2 ∧ ¬(a > b) ∧
  ∃ (c d : ℝ), c > d ∧ ¬(c^2 > d^2) := by
  sorry

end NUMINAMATH_CALUDE_square_greater_not_sufficient_nor_necessary_l1426_142671


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_sum_4_l1426_142670

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Check if a year is after 2020 and has sum of digits equal to 4 -/
def isValidYear (year : ℕ) : Prop :=
  year > 2020 ∧ sumOfDigits year = 4

/-- 2030 is the first year after 2020 with sum of digits equal to 4 -/
theorem first_year_after_2020_with_sum_4 :
  (∀ y : ℕ, y > 2020 ∧ y < 2030 → sumOfDigits y ≠ 4) ∧
  sumOfDigits 2030 = 4 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_sum_4_l1426_142670


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1426_142677

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 1 - 3 * Complex.I) : 
  z.im = -2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1426_142677


namespace NUMINAMATH_CALUDE_number_of_products_l1426_142686

/-- Prove that the number of products is 20 given the fixed cost, marginal cost, and total cost. -/
theorem number_of_products (fixed_cost marginal_cost total_cost : ℚ)
  (h1 : fixed_cost = 12000)
  (h2 : marginal_cost = 200)
  (h3 : total_cost = 16000)
  (h4 : total_cost = fixed_cost + marginal_cost * n) :
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_products_l1426_142686


namespace NUMINAMATH_CALUDE_abc_perfect_cube_l1426_142672

theorem abc_perfect_cube (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ∃ (n : ℤ), (a / b : ℚ) + (b / c : ℚ) + (c / a : ℚ) = n) : 
  ∃ (k : ℤ), a * b * c = k^3 := by
  sorry

end NUMINAMATH_CALUDE_abc_perfect_cube_l1426_142672


namespace NUMINAMATH_CALUDE_backyard_width_calculation_l1426_142630

/-- Given a rectangular backyard with a rectangular shed, calculate the width of the backyard -/
theorem backyard_width_calculation 
  (backyard_length : ℝ) 
  (shed_length shed_width : ℝ) 
  (sod_area : ℝ) :
  backyard_length = 20 →
  shed_length = 3 →
  shed_width = 5 →
  sod_area = 245 →
  ∃ (backyard_width : ℝ), 
    sod_area = backyard_length * backyard_width - shed_length * shed_width ∧ 
    backyard_width = 13 :=
by sorry

end NUMINAMATH_CALUDE_backyard_width_calculation_l1426_142630


namespace NUMINAMATH_CALUDE_final_marble_count_l1426_142616

def initial_marbles : ℝ := 87.0
def received_marbles : ℝ := 8.0

theorem final_marble_count :
  initial_marbles + received_marbles = 95.0 := by sorry

end NUMINAMATH_CALUDE_final_marble_count_l1426_142616


namespace NUMINAMATH_CALUDE_nero_speed_l1426_142674

/-- Given a trail that takes Jerome 6 hours to run at 4 MPH, and Nero 3 hours to run,
    prove that Nero's speed is 8 MPH. -/
theorem nero_speed (jerome_time : ℝ) (nero_time : ℝ) (jerome_speed : ℝ) :
  jerome_time = 6 →
  nero_time = 3 →
  jerome_speed = 4 →
  jerome_time * jerome_speed = nero_time * (jerome_time * jerome_speed / nero_time) :=
by sorry

end NUMINAMATH_CALUDE_nero_speed_l1426_142674


namespace NUMINAMATH_CALUDE_devin_teaching_years_l1426_142652

def total_years_taught (calculus algebra statistics geometry discrete_math : ℕ) : ℕ :=
  calculus + algebra + statistics + geometry + discrete_math

theorem devin_teaching_years :
  ∀ (calculus algebra statistics geometry discrete_math : ℕ),
    calculus = 4 →
    algebra = 2 * calculus →
    statistics = 5 * algebra →
    geometry = 3 * statistics →
    discrete_math = geometry / 2 →
    total_years_taught calculus algebra statistics geometry discrete_math = 232 := by
  sorry

end NUMINAMATH_CALUDE_devin_teaching_years_l1426_142652


namespace NUMINAMATH_CALUDE_social_science_papers_selected_l1426_142698

/-- Proves the number of social science papers selected in stratified sampling -/
theorem social_science_papers_selected
  (total_papers : ℕ)
  (social_science_papers : ℕ)
  (selected_papers : ℕ)
  (h1 : total_papers = 153)
  (h2 : social_science_papers = 54)
  (h3 : selected_papers = 51)
  : (social_science_papers * selected_papers) / total_papers = 18 := by
  sorry

end NUMINAMATH_CALUDE_social_science_papers_selected_l1426_142698


namespace NUMINAMATH_CALUDE_complex_sum_equals_i_l1426_142631

theorem complex_sum_equals_i : Complex.I + 1 + Complex.I^2 = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_sum_equals_i_l1426_142631


namespace NUMINAMATH_CALUDE_smallest_equal_partition_is_seven_l1426_142645

/-- The sum of squares from 1 to n -/
def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Checks if there exists a subset of squares that sum to half the total sum -/
def existsEqualPartition (n : ℕ) : Prop :=
  ∃ (subset : Finset ℕ), subset ⊆ Finset.range n ∧ 
    subset.sum (λ i => (i + 1)^2) = sumOfSquares n / 2

/-- The smallest n for which an equal partition exists -/
def smallestEqualPartition : ℕ := 7

theorem smallest_equal_partition_is_seven :
  (smallestEqualPartition = 7) ∧ 
  (existsEqualPartition 7) ∧ 
  (∀ k < 7, ¬ existsEqualPartition k) :=
sorry

end NUMINAMATH_CALUDE_smallest_equal_partition_is_seven_l1426_142645


namespace NUMINAMATH_CALUDE_solve_equation_l1426_142637

theorem solve_equation (x : ℝ) : 3639 + 11.95 - x = 3054 → x = 596.95 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1426_142637


namespace NUMINAMATH_CALUDE_december_savings_l1426_142689

def savings_plan (initial_amount : ℕ) (months : ℕ) : ℕ :=
  (initial_amount : ℕ) * (3 ^ (months - 1))

theorem december_savings :
  savings_plan 10 12 = 1771470 := by
  sorry

end NUMINAMATH_CALUDE_december_savings_l1426_142689


namespace NUMINAMATH_CALUDE_vertical_line_slope_angle_l1426_142669

-- Define the line x + 2 = 0
def vertical_line (x : ℝ) : Prop := x + 2 = 0

-- Define the slope angle of a line
def slope_angle (line : ℝ → Prop) : ℝ := sorry

-- Theorem: The slope angle of the line x + 2 = 0 is π/2
theorem vertical_line_slope_angle :
  slope_angle vertical_line = π / 2 := by sorry

end NUMINAMATH_CALUDE_vertical_line_slope_angle_l1426_142669


namespace NUMINAMATH_CALUDE_stone_slab_length_l1426_142614

/-- Given a floor covered by square stone slabs, this theorem calculates the length of each slab. -/
theorem stone_slab_length
  (num_slabs : ℕ)
  (total_area : ℝ)
  (h_num_slabs : num_slabs = 30)
  (h_total_area : total_area = 50.7) :
  ∃ (slab_length : ℝ),
    slab_length = 130 ∧
    num_slabs * (slab_length / 100)^2 = total_area :=
by sorry

end NUMINAMATH_CALUDE_stone_slab_length_l1426_142614


namespace NUMINAMATH_CALUDE_score_order_l1426_142696

theorem score_order (a b c d : ℝ) 
  (sum_eq : b + d = a + c)
  (swap_gt : a + b > c + d)
  (d_gt_sum : d > b + c)
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :
  a > d ∧ d > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_score_order_l1426_142696


namespace NUMINAMATH_CALUDE_courageous_iff_coprime_l1426_142617

/-- A function is courageous if it and its 100 shifts are all bijections -/
def IsCourageous (n : ℕ) (g : ZMod n → ZMod n) : Prop :=
  Function.Bijective g ∧
  ∀ k : Fin 101, Function.Bijective (λ x => g x + k * x)

/-- The main theorem: existence of a courageous function is equivalent to n being coprime to 101! -/
theorem courageous_iff_coprime (n : ℕ) :
  (∃ g : ZMod n → ZMod n, IsCourageous n g) ↔ Nat.Coprime n (Nat.factorial 101) := by
  sorry

end NUMINAMATH_CALUDE_courageous_iff_coprime_l1426_142617


namespace NUMINAMATH_CALUDE_may_has_greatest_percentage_difference_l1426_142664

/-- Represents the sales data for a single month --/
structure MonthSales where
  drummers : ℕ
  bugles : ℕ
  flutes : ℕ

/-- Calculates the percentage difference for a given month's sales --/
def percentageDifference (sales : MonthSales) : ℚ :=
  let max := max sales.drummers (max sales.bugles sales.flutes)
  let min := min sales.drummers (min sales.bugles sales.flutes)
  (max - min : ℚ) / min * 100

/-- Sales data for each month --/
def januarySales : MonthSales := ⟨5, 4, 6⟩
def februarySales : MonthSales := ⟨6, 5, 6⟩
def marchSales : MonthSales := ⟨6, 6, 6⟩
def aprilSales : MonthSales := ⟨7, 5, 8⟩
def maySales : MonthSales := ⟨3, 5, 4⟩

/-- Theorem: May has the greatest percentage difference in sales --/
theorem may_has_greatest_percentage_difference :
  percentageDifference maySales > percentageDifference januarySales ∧
  percentageDifference maySales > percentageDifference februarySales ∧
  percentageDifference maySales > percentageDifference marchSales ∧
  percentageDifference maySales > percentageDifference aprilSales :=
by sorry


end NUMINAMATH_CALUDE_may_has_greatest_percentage_difference_l1426_142664
