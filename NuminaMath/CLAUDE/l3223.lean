import Mathlib

namespace inverse_sum_product_l3223_322351

theorem inverse_sum_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : 3*x + y/3 + z ≠ 0) : 
  (3*x + y/3 + z)⁻¹ * ((3*x)⁻¹ + (y/3)⁻¹ + z⁻¹) = (x*y*z)⁻¹ :=
by sorry

end inverse_sum_product_l3223_322351


namespace corn_acreage_l3223_322360

theorem corn_acreage (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : beans_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) : 
  (total_land * corn_ratio) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end corn_acreage_l3223_322360


namespace x_plus_x_squared_l3223_322333

theorem x_plus_x_squared (x : ℕ) (h : x = 3) : x + (x * x) = 12 := by
  sorry

end x_plus_x_squared_l3223_322333


namespace linear_equation_solution_l3223_322397

theorem linear_equation_solution :
  ∀ x : ℝ, x - 2 = 0 ↔ x = 2 := by sorry

end linear_equation_solution_l3223_322397


namespace unique_three_digit_number_twelve_times_sum_of_digits_l3223_322388

theorem unique_three_digit_number_twelve_times_sum_of_digits : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    n = 12 * (n / 100 + (n / 10 % 10) + (n % 10)) := by
  sorry

end unique_three_digit_number_twelve_times_sum_of_digits_l3223_322388


namespace tom_final_coin_value_l3223_322369

/-- Represents the types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- Calculates the total value of a collection of coins --/
def totalValue (coins : List (Coin × ℕ)) : ℕ :=
  coins.foldl (fun acc (c, n) => acc + n * coinValue c) 0

/-- Tom's initial coins --/
def initialCoins : List (Coin × ℕ) :=
  [(Coin.Penny, 27), (Coin.Dime, 15), (Coin.Quarter, 9), (Coin.HalfDollar, 2)]

/-- Coins given by dad --/
def coinsFromDad : List (Coin × ℕ) :=
  [(Coin.Dime, 33), (Coin.Nickel, 49), (Coin.Quarter, 7), (Coin.HalfDollar, 4)]

/-- Coins spent by Tom --/
def spentCoins : List (Coin × ℕ) :=
  [(Coin.Dime, 11), (Coin.Quarter, 5)]

/-- Number of half dollars exchanged for quarters --/
def exchangedHalfDollars : ℕ := 5

/-- Theorem stating the final value of Tom's coins --/
theorem tom_final_coin_value :
  totalValue initialCoins +
  totalValue coinsFromDad -
  totalValue spentCoins +
  exchangedHalfDollars * 2 * coinValue Coin.Quarter =
  1702 := by sorry

end tom_final_coin_value_l3223_322369


namespace geometric_sequence_k_value_l3223_322349

/-- Given a geometric sequence {a_n} with a₂ = 3, a₃ = 9, and a_k = 243, prove that k = 6 -/
theorem geometric_sequence_k_value (a : ℕ → ℝ) (k : ℕ) :
  (∀ n : ℕ, a (n + 1) / a n = a 3 / a 2) →  -- geometric sequence condition
  a 2 = 3 →
  a 3 = 9 →
  a k = 243 →
  k = 6 := by
sorry

end geometric_sequence_k_value_l3223_322349


namespace equation_solution_l3223_322347

theorem equation_solution : ∃ (x : ℚ), 5*x - 3*x = 420 - 10*(x + 2) ∧ x = 100/3 := by
  sorry

end equation_solution_l3223_322347


namespace hayden_evening_snack_l3223_322307

/-- Calculates the amount of nuts in one serving given the bag cost, weight, coupon value, and cost per serving after coupon. -/
def nuts_per_serving (bag_cost : ℚ) (bag_weight : ℚ) (coupon : ℚ) (serving_cost : ℚ) : ℚ :=
  let cost_after_coupon := bag_cost - coupon
  let num_servings := cost_after_coupon / serving_cost
  bag_weight / num_servings

/-- Theorem stating that under the given conditions, the amount of nuts in one serving is 1 oz. -/
theorem hayden_evening_snack :
  nuts_per_serving 25 40 5 (1/2) = 1 := by sorry

end hayden_evening_snack_l3223_322307


namespace base4_sum_234_73_l3223_322317

/-- Converts a number from base 4 to base 10 -/
def base4_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10_to_base4 (n : ℕ) : ℕ := sorry

/-- The sum of two numbers in base 4 -/
def base4_sum (a b : ℕ) : ℕ :=
  base10_to_base4 (base4_to_base10 a + base4_to_base10 b)

theorem base4_sum_234_73 : base4_sum 234 73 = 10303 := by sorry

end base4_sum_234_73_l3223_322317


namespace count_equal_to_one_l3223_322330

theorem count_equal_to_one : 
  let numbers := [(-1)^2, (-1)^3, -(1^2), |(-1)|, -(-1), 1/(-1)]
  (numbers.filter (λ x => x = 1)).length = 3 := by
sorry

end count_equal_to_one_l3223_322330


namespace friendship_theorem_l3223_322324

/-- A graph representing friendships in a class -/
structure FriendshipGraph where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  symmetric : ∀ {i j}, (i, j) ∈ edges → (j, i) ∈ edges
  no_self_loops : ∀ i, (i, i) ∉ edges

/-- The degree of a vertex in the graph -/
def degree (G : FriendshipGraph) (v : ℕ) : ℕ :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- A clique in the graph -/
def is_clique (G : FriendshipGraph) (S : Finset ℕ) : Prop :=
  ∀ i j, i ∈ S → j ∈ S → i ≠ j → (i, j) ∈ G.edges

theorem friendship_theorem (G : FriendshipGraph) 
  (h1 : G.vertices.card = 20)
  (h2 : ∀ v ∈ G.vertices, degree G v ≥ 14) :
  ∃ S : Finset ℕ, S.card = 4 ∧ is_clique G S :=
sorry

end friendship_theorem_l3223_322324


namespace blue_cards_count_l3223_322357

theorem blue_cards_count (red_cards : ℕ) (blue_prob : ℚ) (blue_cards : ℕ) : 
  red_cards = 8 →
  blue_prob = 6/10 →
  (blue_cards : ℚ) / (blue_cards + red_cards) = blue_prob →
  blue_cards = 12 := by
sorry

end blue_cards_count_l3223_322357


namespace recurrence_sequence_is_natural_l3223_322392

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ+ → ℚ) : Prop :=
  a 2 = 2 ∧ ∀ n : ℕ+, (n - 1) * a (n + 1) - n * a n + 1 = 0

/-- The theorem stating that the sequence is equal to the natural numbers -/
theorem recurrence_sequence_is_natural (a : ℕ+ → ℚ) (h : RecurrenceSequence a) :
    ∀ n : ℕ+, a n = n := by
  sorry

end recurrence_sequence_is_natural_l3223_322392


namespace gcd_of_72_120_168_l3223_322310

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by sorry

end gcd_of_72_120_168_l3223_322310


namespace ed_conch_shells_ed_conch_shells_eq_8_l3223_322316

theorem ed_conch_shells (initial_shells : ℕ) (ed_limpet : ℕ) (ed_oyster : ℕ) (jacob_extra : ℕ) (total_shells : ℕ) : ℕ :=
  let ed_known := ed_limpet + ed_oyster
  let jacob_shells := ed_known + jacob_extra
  let known_shells := initial_shells + ed_known + jacob_shells
  total_shells - known_shells

theorem ed_conch_shells_eq_8 : 
  ed_conch_shells 2 7 2 2 30 = 8 := by sorry

end ed_conch_shells_ed_conch_shells_eq_8_l3223_322316


namespace expression_evaluation_l3223_322362

theorem expression_evaluation : (255^2 - 231^2 - (231^2 - 207^2)) / 24 = 48 := by sorry

end expression_evaluation_l3223_322362


namespace sum_of_roots_equals_fourteen_thirds_l3223_322334

-- Define the polynomial
def f (x : ℝ) : ℝ := (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7)

-- Theorem statement
theorem sum_of_roots_equals_fourteen_thirds :
  ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = 14/3 :=
sorry

end sum_of_roots_equals_fourteen_thirds_l3223_322334


namespace T_properties_l3223_322336

-- Define the operation T
def T (a b x y : ℚ) : ℚ := a * x * y + b * x - 4

-- State the theorem
theorem T_properties (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : T a b 2 1 = 2) (h2 : T a b (-1) 2 = -8) :
  (a = 1 ∧ b = 2) ∧ 
  (∀ m n, n ≠ -2 → T a b m n = 0 → m = 4 / (n + 2)) ∧
  (∀ k x y, (∀ k', T a b (k' * x) y = T a b (k * x) y) → y = -2) ∧
  (∀ x y : ℚ, (∀ k, T a b (k * x) y = T a b (k * y) x) → k = 0) :=
by sorry

end T_properties_l3223_322336


namespace sum_lent_calculation_l3223_322359

/-- Calculates the sum lent given the interest rate, time period, and interest amount -/
theorem sum_lent_calculation (interest_rate : ℚ) (years : ℕ) (interest_difference : ℚ) : 
  interest_rate = 5 / 100 →
  years = 8 →
  interest_difference = 360 →
  (1 - years * interest_rate) * 600 = interest_difference :=
by
  sorry

end sum_lent_calculation_l3223_322359


namespace arithmetic_sequence_property_l3223_322302

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₄ = -4 and a₈ = 4, a₁₂ = 12 -/
theorem arithmetic_sequence_property (a : ℕ → ℤ) 
  (h_arith : ArithmeticSequence a) 
  (h_a4 : a 4 = -4) 
  (h_a8 : a 8 = 4) : 
  a 12 = 12 := by
sorry

end arithmetic_sequence_property_l3223_322302


namespace geometric_sequence_general_term_l3223_322365

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : |a 1| = 1)
  (h_a5_a2 : a 5 = -8 * a 2)
  (h_a5_gt_a2 : a 5 > a 2) :
  ∃ r : ℝ, r = -2 ∧ ∀ n : ℕ, a n = r^(n-1) :=
sorry

end geometric_sequence_general_term_l3223_322365


namespace game_result_l3223_322308

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 9
  else if n % 2 = 0 then 3
  else 1

def allie_rolls : List ℕ := [6, 3, 4]
def betty_rolls : List ℕ := [1, 2, 5, 6]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : 
  total_points allie_rolls * total_points betty_rolls = 294 := by
  sorry

end game_result_l3223_322308


namespace triangle_interior_angle_ratio_l3223_322378

theorem triangle_interior_angle_ratio 
  (α β γ : ℝ) 
  (h1 : 2 * α + 3 * β = 4 * γ) 
  (h2 : α = 4 * β - γ) :
  ∃ (k : ℝ), k > 0 ∧ 
    2 * k = 180 - α ∧
    9 * k = 180 - β ∧
    4 * k = 180 - γ := by
sorry

end triangle_interior_angle_ratio_l3223_322378


namespace min_cost_rectangular_container_l3223_322379

/-- Represents the cost function for a rectangular container -/
def cost_function (a b : ℝ) : ℝ := 20 * a * b + 10 * 2 * (a + b)

/-- Theorem stating the minimum cost for the rectangular container -/
theorem min_cost_rectangular_container :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = 4 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y = 4 → cost_function a b ≤ cost_function x y) ∧
  cost_function a b = 160 :=
sorry

end min_cost_rectangular_container_l3223_322379


namespace xiao_he_purchase_cost_l3223_322385

/-- The total cost of Xiao He's purchase -/
def total_cost (notebook_price pen_price : ℝ) : ℝ :=
  4 * notebook_price + 10 * pen_price

/-- Theorem: The total cost of Xiao He's purchase is 4a + 10b -/
theorem xiao_he_purchase_cost (a b : ℝ) :
  total_cost a b = 4 * a + 10 * b := by
  sorry

end xiao_he_purchase_cost_l3223_322385


namespace pond_to_field_area_ratio_l3223_322374

theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    field_length = 2 * field_width →
    field_length = 16 →
    pond_side = 4 →
    (pond_side^2) / (field_length * field_width) = 1/8 :=
by
  sorry

end pond_to_field_area_ratio_l3223_322374


namespace area_of_problem_shape_l3223_322366

/-- A composite shape with right-angled corners -/
structure CompositeShape :=
  (height1 : ℕ)
  (width1 : ℕ)
  (height2 : ℕ)
  (width2 : ℕ)
  (height3 : ℕ)
  (width3 : ℕ)

/-- Calculate the area of the composite shape -/
def area (shape : CompositeShape) : ℕ :=
  shape.height1 * shape.width1 +
  shape.height2 * shape.width2 +
  shape.height3 * shape.width3

/-- The specific shape from the problem -/
def problem_shape : CompositeShape :=
  { height1 := 8
  , width1 := 4
  , height2 := 6
  , width2 := 4
  , height3 := 5
  , width3 := 3 }

theorem area_of_problem_shape :
  area problem_shape = 71 :=
by sorry

end area_of_problem_shape_l3223_322366


namespace tower_height_ratio_l3223_322375

theorem tower_height_ratio :
  ∀ (grace_height clyde_height : ℕ),
    grace_height = 40 →
    grace_height = clyde_height + 35 →
    (grace_height : ℚ) / (clyde_height : ℚ) = 8 := by
  sorry

end tower_height_ratio_l3223_322375


namespace complement_of_sqrt_range_l3223_322395

-- Define the universal set U as ℝ
def U := ℝ

-- Define the set A as the range of y = x^(1/2)
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt x}

-- State the theorem
theorem complement_of_sqrt_range :
  Set.compl A = Set.Iio (0 : ℝ) := by sorry

end complement_of_sqrt_range_l3223_322395


namespace max_value_of_a_l3223_322394

/-- Given that "x^2 + 2x - 3 > 0" is a necessary but not sufficient condition for "x < a",
    prove that the maximum value of a is -3. -/
theorem max_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x < a → x^2 + 2*x - 3 > 0) ∧ 
  (∃ x : ℝ, x^2 + 2*x - 3 > 0 ∧ x ≥ a) →
  a ≤ -3 ∧ ∀ b : ℝ, b > -3 → ¬((∀ x : ℝ, x < b → x^2 + 2*x - 3 > 0) ∧ 
                               (∃ x : ℝ, x^2 + 2*x - 3 > 0 ∧ x ≥ b)) :=
by sorry

end max_value_of_a_l3223_322394


namespace abs_three_implies_plus_minus_three_l3223_322306

theorem abs_three_implies_plus_minus_three (a : ℝ) : 
  |a| = 3 → (a = 3 ∨ a = -3) := by sorry

end abs_three_implies_plus_minus_three_l3223_322306


namespace complex_power_problem_l3223_322391

theorem complex_power_problem : ((1 - Complex.I) / (1 + Complex.I)) ^ 10 = -1 := by
  sorry

end complex_power_problem_l3223_322391


namespace invalid_inequality_l3223_322341

theorem invalid_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬(1 / (a - b) > 1 / a) := by
  sorry

end invalid_inequality_l3223_322341


namespace regular_octagon_side_length_l3223_322398

/-- A regular octagon with a perimeter of 23.6 cm has sides of length 2.95 cm. -/
theorem regular_octagon_side_length : 
  ∀ (perimeter side_length : ℝ),
  perimeter = 23.6 →
  perimeter = 8 * side_length →
  side_length = 2.95 := by
sorry

end regular_octagon_side_length_l3223_322398


namespace equation_solutions_l3223_322368

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧
    x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -4 ∧ x₂ = 1 ∧
    (x₁ + 4)^2 = 5*(x₁ + 4) ∧ (x₂ + 4)^2 = 5*(x₂ + 4)) :=
by sorry

end equation_solutions_l3223_322368


namespace distance_between_stations_l3223_322315

/-- The distance between two stations given the conditions of two trains meeting --/
theorem distance_between_stations
  (speed_train1 : ℝ)
  (speed_train2 : ℝ)
  (extra_distance : ℝ)
  (h1 : speed_train1 = 20)
  (h2 : speed_train2 = 25)
  (h3 : extra_distance = 70)
  (h4 : speed_train1 > 0)
  (h5 : speed_train2 > 0) :
  ∃ (time : ℝ),
    time > 0 ∧
    speed_train1 * time + speed_train2 * time = speed_train1 * time + extra_distance ∧
    speed_train1 * time + speed_train2 * time = 630 :=
by sorry


end distance_between_stations_l3223_322315


namespace one_element_condition_at_most_one_element_condition_l3223_322346

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

-- Theorem 1
theorem one_element_condition (a : ℝ) :
  (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = 1) := by sorry

-- Theorem 2
theorem at_most_one_element_condition (a : ℝ) :
  (∃ x, x ∈ A a → ∀ y, y ∈ A a → x = y) ↔ (a ≥ 1 ∨ a = 0) := by sorry

end one_element_condition_at_most_one_element_condition_l3223_322346


namespace bracelet_arrangement_l3223_322312

/-- The number of unique arrangements of beads on a bracelet -/
def uniqueArrangements (n : ℕ) : ℕ := sorry

/-- Two specific beads are always adjacent -/
def adjacentBeads : Prop := sorry

/-- Rotations and reflections of the same arrangement are considered identical -/
def symmetryEquivalence : Prop := sorry

theorem bracelet_arrangement :
  uniqueArrangements 8 = 720 ∧ adjacentBeads ∧ symmetryEquivalence :=
sorry

end bracelet_arrangement_l3223_322312


namespace extreme_point_of_f_l3223_322305

/-- The function f(x) = 2x^2 - 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4 * x

theorem extreme_point_of_f :
  ∃! x : ℝ, ∀ y : ℝ, f y ≥ f x :=
sorry

end extreme_point_of_f_l3223_322305


namespace seven_ways_to_make_eight_cents_l3223_322387

/-- Represents the number of ways to make a certain amount with given coins -/
def num_ways_to_make_amount (one_cent : ℕ) (two_cent : ℕ) (five_cent : ℕ) (target : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 7 ways to make 8 cents with the given coins -/
theorem seven_ways_to_make_eight_cents :
  num_ways_to_make_amount 8 4 1 8 = 7 := by
  sorry

end seven_ways_to_make_eight_cents_l3223_322387


namespace max_angle_cone_from_semicircle_l3223_322343

/-- The maximum angle between generatrices of a cone formed by a semicircle -/
theorem max_angle_cone_from_semicircle :
  ∀ (r : ℝ),
  r > 0 →
  let semicircle_arc_length := r * Real.pi
  let base_circumference := 2 * r * Real.pi / 2
  semicircle_arc_length = base_circumference →
  ∃ (θ : ℝ),
  θ = 60 * (Real.pi / 180) ∧
  ∀ (α : ℝ),
  (α ≥ 0 ∧ α ≤ θ) →
  ∃ (g₁ g₂ : ℝ × ℝ),
  (g₁.1 - g₂.1)^2 + (g₁.2 - g₂.2)^2 ≤ r^2 ∧
  Real.arccos ((g₁.1 * g₂.1 + g₁.2 * g₂.2) / r^2) = α :=
by sorry

end max_angle_cone_from_semicircle_l3223_322343


namespace sandwich_jam_cost_l3223_322399

theorem sandwich_jam_cost 
  (N B J : ℕ) 
  (h1 : N > 1) 
  (h2 : B > 0) 
  (h3 : J > 0) 
  (h4 : N * (4 * B + 5 * J + 20) = 414) : 
  N * 5 * J = 225 := by
  sorry

end sandwich_jam_cost_l3223_322399


namespace solution_set_when_a_is_one_a_value_when_minimum_is_four_l3223_322361

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |a*x - 5|

-- Part 1
theorem solution_set_when_a_is_one :
  let a : ℝ := 1
  {x : ℝ | f a x ≥ 9} = {x : ℝ | x ≤ -1 ∨ x > 5} := by sorry

-- Part 2
theorem a_value_when_minimum_is_four :
  ∃ (a : ℝ), 0 < a ∧ a < 5 ∧ 
  (∀ x : ℝ, f a x ≥ 4) ∧
  (∃ x : ℝ, f a x = 4) →
  a = 2 := by sorry

end solution_set_when_a_is_one_a_value_when_minimum_is_four_l3223_322361


namespace smallest_solution_of_equation_l3223_322327

theorem smallest_solution_of_equation (x : ℝ) :
  (3 * x^2 + 33 * x - 90 = x * (x + 18)) →
  x ≥ -10.5 :=
by sorry

end smallest_solution_of_equation_l3223_322327


namespace equation_satisfied_at_eight_l3223_322344

def f (x : ℝ) : ℝ := 2 * x - 3

theorem equation_satisfied_at_eight :
  ∃ x : ℝ, 2 * (f x) - 21 = f (x - 4) ∧ x = 8 := by
  sorry

end equation_satisfied_at_eight_l3223_322344


namespace complex_power_difference_l3223_322352

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference : i^2 = -1 → (1 + i)^18 - (1 - i)^18 = 1024 * i := by
  sorry

end complex_power_difference_l3223_322352


namespace dynaco_shares_sold_is_150_l3223_322370

/-- Represents the stock portfolio problem --/
structure StockPortfolio where
  microtron_price : ℝ
  dynaco_price : ℝ
  total_shares : ℕ
  average_price : ℝ

/-- Calculates the number of Dynaco shares sold --/
def dynaco_shares_sold (portfolio : StockPortfolio) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, 150 Dynaco shares were sold --/
theorem dynaco_shares_sold_is_150 : 
  let portfolio := StockPortfolio.mk 36 44 300 40
  dynaco_shares_sold portfolio = 150 := by
  sorry

end dynaco_shares_sold_is_150_l3223_322370


namespace consecutive_pages_sum_l3223_322345

theorem consecutive_pages_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20736 → n + (n + 1) = 287 := by
  sorry

end consecutive_pages_sum_l3223_322345


namespace half_area_closest_to_longest_side_l3223_322389

/-- Represents a trapezoid field with specific measurements -/
structure TrapezoidField where
  short_base : ℝ
  long_base : ℝ
  slant_side : ℝ
  slant_angle : ℝ

/-- The fraction of the area closer to the longest side of the trapezoid field -/
def fraction_closest_to_longest_side (field : TrapezoidField) : ℝ :=
  sorry

/-- Theorem stating that for a specific trapezoid field, the fraction of area closest to the longest side is 1/2 -/
theorem half_area_closest_to_longest_side :
  let field : TrapezoidField := {
    short_base := 80,
    long_base := 160,
    slant_side := 120,
    slant_angle := π / 4
  }
  fraction_closest_to_longest_side field = 1 / 2 := by
  sorry

end half_area_closest_to_longest_side_l3223_322389


namespace sams_age_l3223_322380

theorem sams_age (billy joe sam : ℕ) 
  (h1 : billy = 2 * joe) 
  (h2 : billy + joe = 60) 
  (h3 : sam = (billy + joe) / 2) : 
  sam = 30 := by
sorry

end sams_age_l3223_322380


namespace mailing_cost_correct_l3223_322356

/-- The cost function for mailing a document -/
def mailing_cost (P : ℕ) : ℕ :=
  if P ≤ 5 then
    15 + 5 * (P - 1)
  else
    15 + 5 * (P - 1) + 2

/-- Theorem stating the correctness of the mailing cost function -/
theorem mailing_cost_correct (P : ℕ) :
  mailing_cost P =
    if P ≤ 5 then
      15 + 5 * (P - 1)
    else
      15 + 5 * (P - 1) + 2 :=
by
  sorry

/-- Lemma: The cost for the first kilogram is 15 cents -/
lemma first_kg_cost (P : ℕ) (h : P > 0) : mailing_cost P ≥ 15 :=
by
  sorry

/-- Lemma: Each subsequent kilogram costs 5 cents -/
lemma subsequent_kg_cost (P : ℕ) (h : P > 1) :
  mailing_cost P - mailing_cost (P - 1) = 5 :=
by
  sorry

/-- Lemma: Additional handling fee of 2 cents for documents over 5 kg -/
lemma handling_fee (P : ℕ) (h : P > 5) :
  mailing_cost P - mailing_cost 5 = 5 * (P - 5) + 2 :=
by
  sorry

end mailing_cost_correct_l3223_322356


namespace wrapping_paper_area_theorem_l3223_322321

/-- The area of wrapping paper required for a rectangular box -/
def wrapping_paper_area (l w h : ℝ) : ℝ := l * w + 2 * l * h + 2 * w * h + 4 * h^2

/-- Theorem stating the area of wrapping paper required for a rectangular box -/
theorem wrapping_paper_area_theorem (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  let box_volume := l * w * h
  let paper_length := l + 2 * h
  let paper_width := w + 2 * h
  let paper_area := paper_length * paper_width
  paper_area = wrapping_paper_area l w h :=
by sorry

end wrapping_paper_area_theorem_l3223_322321


namespace percentage_difference_l3223_322386

theorem percentage_difference (z y x : ℝ) (total : ℝ) : 
  y = 1.2 * z →
  z = 300 →
  total = 1110 →
  x = total - y - z →
  (x - y) / y * 100 = 25 :=
by sorry

end percentage_difference_l3223_322386


namespace dividend_calculation_l3223_322328

theorem dividend_calculation (remainder quotient divisor : ℕ) 
  (h_remainder : remainder = 19)
  (h_quotient : quotient = 61)
  (h_divisor : divisor = 8) :
  divisor * quotient + remainder = 507 := by
  sorry

end dividend_calculation_l3223_322328


namespace randys_trip_l3223_322303

theorem randys_trip (x : ℚ) 
  (h1 : x / 4 + 30 + x / 6 = x) : x = 360 / 7 := by
  sorry

end randys_trip_l3223_322303


namespace bob_has_winning_strategy_l3223_322325

/-- Represents the state of the game board -/
structure GameState where
  value : Nat

/-- Represents a player's move -/
inductive Move
  | Bob (a : Nat)
  | Alice (k : Nat)

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Bob a => ⟨state.value - a^2⟩
  | Move.Alice k => ⟨state.value^k⟩

/-- Defines a winning sequence of moves for Bob -/
def WinningSequence (initialState : GameState) (moves : List Move) : Prop :=
  moves.foldl applyMove initialState = ⟨0⟩

/-- The main theorem stating Bob's winning strategy exists -/
theorem bob_has_winning_strategy :
  ∀ (initialState : GameState), initialState.value > 0 →
  ∃ (moves : List Move), WinningSequence initialState moves :=
sorry


end bob_has_winning_strategy_l3223_322325


namespace milk_quality_theorem_l3223_322314

/-- The probability of a single bottle of milk being qualified -/
def p_qualified : ℝ := 0.8

/-- The number of bottles bought -/
def n_bottles : ℕ := 2

/-- The number of days considered -/
def n_days : ℕ := 3

/-- The probability that all bought bottles are qualified -/
def prob_all_qualified : ℝ := p_qualified ^ n_bottles

/-- The probability of drinking unqualified milk in a day -/
def p_unqualified_day : ℝ := 1 - p_qualified ^ n_bottles

/-- The expected number of days drinking unqualified milk -/
def expected_unqualified_days : ℝ := n_days * p_unqualified_day

theorem milk_quality_theorem :
  prob_all_qualified = 0.64 ∧ expected_unqualified_days = 1.08 := by
  sorry

end milk_quality_theorem_l3223_322314


namespace simplify_nested_expression_l3223_322313

theorem simplify_nested_expression (x : ℝ) : 2 - (3 - (2 - (5 - (3 - x)))) = -1 - x := by
  sorry

end simplify_nested_expression_l3223_322313


namespace cos_plus_one_is_pseudo_even_l3223_322371

-- Define the concept of a pseudo-even function
def isPseudoEven (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = f (2 * a - x)

-- State the theorem
theorem cos_plus_one_is_pseudo_even :
  isPseudoEven (λ x => Real.cos (x + 1)) := by
  sorry

end cos_plus_one_is_pseudo_even_l3223_322371


namespace negation_equivalence_l3223_322301

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ x^2 > 3) ↔ (∀ x : ℝ, x ≥ 0 → x^2 ≤ 3) :=
by sorry

end negation_equivalence_l3223_322301


namespace justice_ferns_l3223_322329

/-- Given the number of palms and succulents Justice has, the total number of plants she wants,
    and the number of additional plants she needs, prove that Justice has 3 ferns. -/
theorem justice_ferns (palms_and_succulents : ℕ) (desired_total : ℕ) (additional_needed : ℕ)
  (h1 : palms_and_succulents = 12)
  (h2 : desired_total = 24)
  (h3 : additional_needed = 9) :
  desired_total - additional_needed - palms_and_succulents = 3 :=
by sorry

end justice_ferns_l3223_322329


namespace regular_polygon_radius_l3223_322355

/-- A regular polygon with the given properties --/
structure RegularPolygon where
  -- Number of sides
  n : ℕ
  -- Side length
  s : ℝ
  -- Radius
  r : ℝ
  -- Sum of interior angles is twice the sum of exterior angles
  interior_sum_twice_exterior : (n - 2) * 180 = 2 * 360
  -- Side length is 2
  side_length_is_two : s = 2

/-- The radius of the regular polygon with the given properties is 2 --/
theorem regular_polygon_radius (p : RegularPolygon) : p.r = 2 := by
  sorry

end regular_polygon_radius_l3223_322355


namespace min_blue_beads_l3223_322319

/-- Represents a necklace with red and blue beads. -/
structure Necklace :=
  (red_beads : ℕ)
  (blue_beads : ℕ)

/-- Checks if a necklace satisfies the condition that any segment
    containing 10 red beads also contains at least 7 blue beads. -/
def satisfies_condition (n : Necklace) : Prop :=
  ∀ (segment : List (Bool)), 
    segment.length ≤ n.red_beads + n.blue_beads →
    (segment.filter id).length = 10 →
    (segment.filter not).length ≥ 7

/-- The main theorem: The minimum number of blue beads in a necklace
    with 100 red beads that satisfies the given condition is 78. -/
theorem min_blue_beads :
  ∃ (n : Necklace), 
    n.red_beads = 100 ∧ 
    satisfies_condition n ∧
    n.blue_beads = 78 ∧
    (∀ (m : Necklace), m.red_beads = 100 → satisfies_condition m → m.blue_beads ≥ 78) :=
by sorry

end min_blue_beads_l3223_322319


namespace ice_cream_sundaes_l3223_322339

theorem ice_cream_sundaes (n : ℕ) (k : ℕ) : n = 8 ∧ k = 2 → Nat.choose n k = 28 := by
  sorry

end ice_cream_sundaes_l3223_322339


namespace books_bought_at_yard_sale_l3223_322332

def initial_books : ℕ := 35
def final_books : ℕ := 91

theorem books_bought_at_yard_sale :
  final_books - initial_books = 56 :=
by sorry

end books_bought_at_yard_sale_l3223_322332


namespace smallest_possible_d_l3223_322322

theorem smallest_possible_d : ∃ d : ℝ,
  (∀ d' : ℝ, d' ≥ 0 → (4 * Real.sqrt 3) ^ 2 + (d' - 2) ^ 2 = (4 * d') ^ 2 → d ≤ d') ∧
  (4 * Real.sqrt 3) ^ 2 + (d - 2) ^ 2 = (4 * d) ^ 2 ∧
  d = 26 / 15 := by
  sorry

end smallest_possible_d_l3223_322322


namespace condition_neither_sufficient_nor_necessary_l3223_322363

/-- The function f(x) = x^3 - x + a --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x + a

/-- The condition a^2 - a = 0 --/
def condition (a : ℝ) : Prop := a^2 - a = 0

/-- f is an increasing function --/
def is_increasing (a : ℝ) : Prop := ∀ x y, x < y → f a x < f a y

theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ a, condition a → is_increasing a) ∧
  ¬(∀ a, is_increasing a → condition a) :=
sorry

end condition_neither_sufficient_nor_necessary_l3223_322363


namespace triangle_inequality_l3223_322337

/-- Checks if three lengths can form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ¬(is_valid_triangle a b 15) ∧ is_valid_triangle a b 13 :=
by
  sorry

#check triangle_inequality 8 7

end triangle_inequality_l3223_322337


namespace cube_packing_surface_area_l3223_322372

/-- A rectangular box that can fit cubic products. -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The surface area of a box in square centimeters. -/
def surfaceArea (b : Box) : ℕ :=
  2 * (b.length * b.width + b.length * b.height + b.width * b.height)

/-- The volume of a box in cubic centimeters. -/
def volume (b : Box) : ℕ :=
  b.length * b.width * b.height

theorem cube_packing_surface_area :
  ∃ (b : Box), volume b = 12 ∧ (surfaceArea b = 40 ∨ surfaceArea b = 38 ∨ surfaceArea b = 32) := by
  sorry


end cube_packing_surface_area_l3223_322372


namespace angle_with_complement_one_third_of_supplement_l3223_322381

theorem angle_with_complement_one_third_of_supplement (x : Real) : 
  (90 - x = (1 / 3) * (180 - x)) → x = 45 := by
  sorry

end angle_with_complement_one_third_of_supplement_l3223_322381


namespace fraction_sum_equality_l3223_322377

theorem fraction_sum_equality (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2*x - 5) / (x^2 - 1) + 3 / (1 - x) = -(x + 8) / (x^2 - 1) := by
  sorry

end fraction_sum_equality_l3223_322377


namespace point_in_second_quadrant_l3223_322320

/-- A point P(x, y) is in the second quadrant if and only if x < 0 and y > 0 -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P is a - 2 -/
def x_coordinate (a : ℝ) : ℝ := a - 2

/-- The y-coordinate of point P is 2 -/
def y_coordinate : ℝ := 2

/-- Theorem: For a point P(a-2, 2) to be in the second quadrant, a must be less than 2 -/
theorem point_in_second_quadrant (a : ℝ) : 
  second_quadrant (x_coordinate a) y_coordinate ↔ a < 2 := by
  sorry

end point_in_second_quadrant_l3223_322320


namespace max_additional_plates_l3223_322323

/-- Represents the number of letters in each set for license plates -/
structure LicensePlateSets :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Calculates the total number of unique license plates -/
def totalPlates (sets : LicensePlateSets) : ℕ :=
  sets.first * sets.second * sets.third

/-- The initial configuration of letter sets -/
def initialSets : LicensePlateSets :=
  ⟨5, 3, 4⟩

/-- The number of new letters to be added -/
def newLetters : ℕ := 2

/-- Theorem: The maximum number of additional unique license plates is 30 -/
theorem max_additional_plates :
  ∃ (newSets : LicensePlateSets),
    (newSets.first + newSets.second + newSets.third = initialSets.first + initialSets.second + initialSets.third + newLetters) ∧
    (∀ (otherSets : LicensePlateSets),
      (otherSets.first + otherSets.second + otherSets.third = initialSets.first + initialSets.second + initialSets.third + newLetters) →
      (totalPlates newSets - totalPlates initialSets ≥ totalPlates otherSets - totalPlates initialSets)) ∧
    (totalPlates newSets - totalPlates initialSets = 30) :=
  sorry

end max_additional_plates_l3223_322323


namespace binomial_coefficient_ratio_l3223_322350

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 3 / 4 ∧ 
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 4 / 5 → 
  n + k = 55 := by sorry

end binomial_coefficient_ratio_l3223_322350


namespace exists_function_satisfying_condition_l3223_322364

theorem exists_function_satisfying_condition : 
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2*x) = |x + 1| := by
  sorry

end exists_function_satisfying_condition_l3223_322364


namespace purely_imaginary_complex_number_l3223_322353

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := m + 1 + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = -1 := by
  sorry

end purely_imaginary_complex_number_l3223_322353


namespace coin_flip_sequences_l3223_322376

theorem coin_flip_sequences (n : ℕ) : n = 10 → (2 : ℕ) ^ n = 1024 := by
  sorry

end coin_flip_sequences_l3223_322376


namespace children_with_cats_l3223_322342

/-- Represents the number of children in each category -/
structure KindergartenPets where
  total : ℕ
  onlyDogs : ℕ
  bothPets : ℕ
  onlyCats : ℕ

/-- The conditions of the kindergarten pet situation -/
def kindergartenConditions : KindergartenPets where
  total := 30
  onlyDogs := 18
  bothPets := 6
  onlyCats := 30 - 18 - 6

theorem children_with_cats (k : KindergartenPets) 
  (h1 : k.total = 30)
  (h2 : k.onlyDogs = 18)
  (h3 : k.bothPets = 6)
  (h4 : k.total = k.onlyDogs + k.onlyCats + k.bothPets) :
  k.onlyCats + k.bothPets = 12 := by
  sorry

#eval kindergartenConditions.onlyCats + kindergartenConditions.bothPets

end children_with_cats_l3223_322342


namespace total_squares_count_l3223_322309

/-- Represents a point in the grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a square in the grid --/
structure GridSquare where
  topLeft : GridPoint
  size : ℕ

/-- The set of all points in the grid, including additional points --/
def gridPoints : Set GridPoint := sorry

/-- Checks if a square is valid within the given grid --/
def isValidSquare (square : GridSquare) : Prop := sorry

/-- Counts the number of valid squares of a given size --/
def countValidSquares (size : ℕ) : ℕ := sorry

/-- The main theorem to prove --/
theorem total_squares_count :
  (countValidSquares 1) + (countValidSquares 2) = 59 := by sorry

end total_squares_count_l3223_322309


namespace perfect_square_from_sqrt_l3223_322338

theorem perfect_square_from_sqrt (n : ℤ) :
  ∃ (m : ℤ), m = 2 + 2 * Real.sqrt (28 * n^2 + 1) → ∃ (k : ℤ), m = k^2 := by
  sorry

end perfect_square_from_sqrt_l3223_322338


namespace investment_return_is_25_percent_l3223_322340

/-- Calculates the percentage return on investment for a given dividend rate, face value, and purchase price of shares. -/
def percentage_return_on_investment (dividend_rate : ℚ) (face_value : ℚ) (purchase_price : ℚ) : ℚ :=
  (dividend_rate * face_value / purchase_price) * 100

/-- Theorem stating that for the given conditions, the percentage return on investment is 25%. -/
theorem investment_return_is_25_percent :
  let dividend_rate : ℚ := 125 / 1000
  let face_value : ℚ := 60
  let purchase_price : ℚ := 30
  percentage_return_on_investment dividend_rate face_value purchase_price = 25 := by
sorry

#eval percentage_return_on_investment (125/1000) 60 30

end investment_return_is_25_percent_l3223_322340


namespace no_solution_for_digit_difference_l3223_322367

theorem no_solution_for_digit_difference : 
  ¬ ∃ (x : ℕ), x < 10 ∧ 
    (max (max (max x 3) 1) 4 * 1000 + 
     max (max (min x 3) 1) 4 * 100 + 
     min (min (max x 3) 1) 4 * 10 + 
     min (min (min x 3) 1) 4) - 
    (min (min (min x 3) 1) 4 * 1000 + 
     min (min (max x 3) 1) 4 * 100 + 
     max (max (min x 3) 1) 4 * 10 + 
     max (max (max x 3) 1) 4) = 4086 := by
  sorry

end no_solution_for_digit_difference_l3223_322367


namespace increasing_geometric_sequence_exists_l3223_322373

theorem increasing_geometric_sequence_exists : ∃ (a : ℕ → ℝ), 
  (∀ n : ℕ, a (n + 1) > a n) ∧  -- increasing
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) ∧  -- geometric
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 4 ∧  -- first three terms
  a 2 + a 3 = 6 * a 1  -- given condition
:= by sorry

end increasing_geometric_sequence_exists_l3223_322373


namespace quadratic_inequality_set_l3223_322383

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 6 * m * x + 5 * m + 1

-- State the theorem
theorem quadratic_inequality_set :
  {m : ℝ | ∀ x : ℝ, f m x > 0} = {m : ℝ | 0 ≤ m ∧ m < 1/4} := by sorry

end quadratic_inequality_set_l3223_322383


namespace arithmetic_to_geometric_sequence_l3223_322318

/-- Given three numbers in an arithmetic sequence with a ratio of 3:4:5,
    if increasing the smallest number by 1 forms a geometric sequence,
    then the original three numbers are 15, 20, and 25. -/
theorem arithmetic_to_geometric_sequence (a d : ℝ) : 
  (a - d : ℝ) / 3 = a / 4 ∧ a / 4 = (a + d) / 5 →  -- arithmetic sequence with ratio 3:4:5
  ∃ r : ℝ, (a - d + 1) / a = a / (a + d) ∧ a / (a + d) = r →  -- geometric sequence after increasing smallest by 1
  a - d = 15 ∧ a = 20 ∧ a + d = 25 := by  -- original numbers are 15, 20, 25
sorry

end arithmetic_to_geometric_sequence_l3223_322318


namespace smallest_percent_increase_l3223_322354

-- Define the values for the relevant questions
def value : Nat → ℕ
| 1 => 100
| 2 => 300
| 3 => 400
| 4 => 700
| 12 => 180000
| 13 => 360000
| 14 => 720000
| 15 => 1440000
| _ => 0  -- Default case, not used in our problem

-- Define the percent increase function
def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

-- Theorem statement
theorem smallest_percent_increase :
  let increase_1_2 := percent_increase (value 1) (value 2)
  let increase_2_3 := percent_increase (value 2) (value 3)
  let increase_3_4 := percent_increase (value 3) (value 4)
  let increase_12_13 := percent_increase (value 12) (value 13)
  let increase_14_15 := percent_increase (value 14) (value 15)
  increase_2_3 < increase_1_2 ∧
  increase_2_3 < increase_3_4 ∧
  increase_2_3 < increase_12_13 ∧
  increase_2_3 < increase_14_15 := by
  sorry

end smallest_percent_increase_l3223_322354


namespace sticker_distribution_l3223_322304

theorem sticker_distribution (n k : ℕ) (hn : n = 10) (hk : k = 5) :
  Nat.choose (n + k - 1) (k - 1) = 1001 := by
  sorry

end sticker_distribution_l3223_322304


namespace circle_area_and_diameter_l3223_322331

/-- For a circle with circumference 36 cm, prove its area and diameter -/
theorem circle_area_and_diameter (C : ℝ) (h : C = 36) :
  ∃ (A d : ℝ),
    A = 324 / Real.pi ∧
    d = 36 / Real.pi ∧
    C = Real.pi * d ∧
    A = Real.pi * (d / 2)^2 := by
sorry


end circle_area_and_diameter_l3223_322331


namespace greatest_ratio_bound_l3223_322311

theorem greatest_ratio_bound (x y z u : ℕ+) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (h3 : x ≥ y) :
  (x : ℝ) / y ≤ 3 + 2 * Real.sqrt 2 ∧ ∃ (x' y' z' u' : ℕ+), 
    x' + y' = z' + u' ∧ 
    2 * x' * y' = z' * u' ∧ 
    x' ≥ y' ∧ 
    (x' : ℝ) / y' = 3 + 2 * Real.sqrt 2 :=
sorry

end greatest_ratio_bound_l3223_322311


namespace least_cans_proof_l3223_322390

/-- The number of liters of Maaza -/
def maaza_liters : ℕ := 50

/-- The number of liters of Pepsi -/
def pepsi_liters : ℕ := 144

/-- The number of liters of Sprite -/
def sprite_liters : ℕ := 368

/-- The least number of cans required to pack all drinks -/
def least_cans : ℕ := 281

/-- Theorem stating that the least number of cans required is 281 -/
theorem least_cans_proof :
  ∃ (can_size : ℕ), can_size > 0 ∧
  maaza_liters % can_size = 0 ∧
  pepsi_liters % can_size = 0 ∧
  sprite_liters % can_size = 0 ∧
  least_cans = maaza_liters / can_size + pepsi_liters / can_size + sprite_liters / can_size ∧
  ∀ (other_size : ℕ), other_size > 0 →
    maaza_liters % other_size = 0 →
    pepsi_liters % other_size = 0 →
    sprite_liters % other_size = 0 →
    least_cans ≤ maaza_liters / other_size + pepsi_liters / other_size + sprite_liters / other_size :=
by
  sorry

end least_cans_proof_l3223_322390


namespace expression_evaluation_l3223_322348

theorem expression_evaluation :
  (45 + 15)^2 - (45^2 + 15^2 + 2 * 45 * 5) = 900 := by
  sorry

end expression_evaluation_l3223_322348


namespace nine_pointed_star_angle_sum_l3223_322358

/-- A star polygon with n points, skipping k points between connections -/
structure StarPolygon where
  n : ℕ  -- number of points
  k : ℕ  -- number of points skipped

/-- The sum of angles at the tips of a star polygon -/
def sumOfTipAngles (star : StarPolygon) : ℝ :=
  sorry

/-- Theorem: The sum of angles at the tips of a 9-pointed star, skipping 3 points, is 720° -/
theorem nine_pointed_star_angle_sum :
  let star : StarPolygon := { n := 9, k := 3 }
  sumOfTipAngles star = 720 := by
  sorry

end nine_pointed_star_angle_sum_l3223_322358


namespace parabola_c_is_negative_eighteen_l3223_322335

/-- A parabola passing through two given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  pass_through_point1 : 2 * 2^2 + b * 2 + c = 6
  pass_through_point2 : 2 * (-3)^2 + b * (-3) + c = -24

/-- The value of c for the parabola -/
def parabola_c_value (p : Parabola) : ℝ := -18

/-- Theorem stating that the value of c for the parabola is -18 -/
theorem parabola_c_is_negative_eighteen (p : Parabola) : 
  parabola_c_value p = p.c := by sorry

end parabola_c_is_negative_eighteen_l3223_322335


namespace oxide_other_element_weight_l3223_322382

/-- The atomic weight of the other element in a calcium oxide -/
def atomic_weight_other_element (molecular_weight : ℝ) (calcium_weight : ℝ) : ℝ :=
  molecular_weight - calcium_weight

/-- Theorem stating that the atomic weight of the other element in the oxide is 16 -/
theorem oxide_other_element_weight :
  let molecular_weight : ℝ := 56
  let calcium_weight : ℝ := 40
  atomic_weight_other_element molecular_weight calcium_weight = 16 := by
  sorry

end oxide_other_element_weight_l3223_322382


namespace inverse_implies_negation_l3223_322396

theorem inverse_implies_negation (P : Prop) :
  (¬P → False) → (¬P) :=
by
  sorry

end inverse_implies_negation_l3223_322396


namespace tan_alpha_plus_pi_third_l3223_322326

theorem tan_alpha_plus_pi_third (α β : Real) 
  (h1 : Real.tan (α + β + π/6) = 1/2) 
  (h2 : Real.tan (β - π/6) = -1/3) : 
  Real.tan (α + π/3) = 1 := by
  sorry

end tan_alpha_plus_pi_third_l3223_322326


namespace siblings_ages_l3223_322393

/-- Represents the ages of the siblings -/
structure SiblingAges where
  richard : ℕ
  david : ℕ
  scott : ℕ
  emily : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : SiblingAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.david = ages.scott + 8 ∧
  ages.emily = ages.richard - 5 ∧
  ages.richard + 8 = 2 * (ages.scott + 8)

/-- The theorem to be proved -/
theorem siblings_ages : 
  ∃ (ages : SiblingAges), satisfies_conditions ages ∧ 
    ages.richard = 20 ∧ ages.david = 14 ∧ ages.scott = 6 ∧ ages.emily = 15 := by
  sorry

end siblings_ages_l3223_322393


namespace min_value_and_existence_l3223_322384

/-- The circle C defined by x^2 + y^2 = x + y where x, y > 0 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = p.1 + p.2 ∧ p.1 > 0 ∧ p.2 > 0}

theorem min_value_and_existence : 
  (∀ p ∈ C, 1 / p.1 + 1 / p.2 ≥ 2) ∧ 
  (∃ p ∈ C, (p.1 + 1) * (p.2 + 1) = 4) := by
sorry

end min_value_and_existence_l3223_322384


namespace journey_end_day_l3223_322300

/-- Represents days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the next day of the week -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

/-- Calculates the arrival day given a starting day and journey duration in hours -/
def arrivalDay (startDay : Day) (journeyHours : Nat) : Day :=
  let daysPassed := journeyHours / 24
  (List.range daysPassed).foldl (fun d _ => nextDay d) startDay

/-- Theorem: A 28-hour journey starting on Tuesday will end on Wednesday -/
theorem journey_end_day :
  arrivalDay Day.Tuesday 28 = Day.Wednesday := by
  sorry


end journey_end_day_l3223_322300
