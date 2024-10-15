import Mathlib

namespace NUMINAMATH_CALUDE_angle_b_is_sixty_degrees_triangle_is_equilateral_l3341_334144

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Add necessary conditions
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  area_formula : S = (1/2) * a * c * Real.sin B

-- Theorem 1
theorem angle_b_is_sixty_degrees (t : Triangle) 
  (h : t.a^2 + t.c^2 = t.b^2 + t.a * t.c) : 
  t.B = π/3 := by sorry

-- Theorem 2
theorem triangle_is_equilateral (t : Triangle)
  (h1 : t.a^2 + t.c^2 = t.b^2 + t.a * t.c)
  (h2 : t.b = 2)
  (h3 : t.S = Real.sqrt 3) :
  t.a = t.b ∧ t.b = t.c := by sorry

end NUMINAMATH_CALUDE_angle_b_is_sixty_degrees_triangle_is_equilateral_l3341_334144


namespace NUMINAMATH_CALUDE_min_value_theorem_l3341_334193

theorem min_value_theorem (x y : ℝ) (h1 : x > 2) (h2 : y > 2) (h3 : x + y = 6) :
  ((x - 1)^2 / (y - 2)) + ((y - 1)^2 / (x - 2)) ≥ 8 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 2 ∧ y₀ > 2 ∧ x₀ + y₀ = 6 ∧
    ((x₀ - 1)^2 / (y₀ - 2)) + ((y₀ - 1)^2 / (x₀ - 2)) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3341_334193


namespace NUMINAMATH_CALUDE_base6_divisibility_by_19_l3341_334108

/-- Converts a base 6 number to decimal --/
def base6ToDecimal (a b c d : ℕ) : ℕ := a * 6^3 + b * 6^2 + c * 6 + d

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

theorem base6_divisibility_by_19 (y : ℕ) (h : y < 6) :
  isDivisibleBy19 (base6ToDecimal 2 5 y 3) ↔ y = 2 := by sorry

end NUMINAMATH_CALUDE_base6_divisibility_by_19_l3341_334108


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3341_334142

theorem triangle_angle_problem (A B C x : ℝ) : 
  A = 40 ∧ B = 3*x ∧ C = 2*x ∧ A + B + C = 180 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3341_334142


namespace NUMINAMATH_CALUDE_scientific_notation_of_seven_nm_l3341_334178

-- Define the value of 7nm in meters
def seven_nm : ℝ := 0.000000007

-- Theorem to prove the scientific notation
theorem scientific_notation_of_seven_nm :
  ∃ (a : ℝ) (n : ℤ), seven_nm = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7 ∧ n = -9 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_seven_nm_l3341_334178


namespace NUMINAMATH_CALUDE_gum_purchase_cost_l3341_334107

/-- Calculates the total cost in dollars for buying gum with a discount -/
def total_cost_with_discount (price_per_piece : ℚ) (num_pieces : ℕ) (discount_rate : ℚ) : ℚ :=
  let total_cost_cents := price_per_piece * num_pieces
  let discount_amount := discount_rate * total_cost_cents
  let final_cost_cents := total_cost_cents - discount_amount
  final_cost_cents / 100

/-- Theorem: The total cost of buying 1500 pieces of gum at 2 cents each with a 10% discount is $27 -/
theorem gum_purchase_cost :
  total_cost_with_discount 2 1500 (10/100) = 27 := by
  sorry


end NUMINAMATH_CALUDE_gum_purchase_cost_l3341_334107


namespace NUMINAMATH_CALUDE_diamond_digit_value_l3341_334151

/-- Given that ◇6₉ = ◇3₁₀, where ◇ represents a digit, prove that ◇ = 3 -/
theorem diamond_digit_value :
  ∀ (diamond : ℕ),
  diamond < 10 →
  diamond * 9 + 6 = diamond * 10 + 3 →
  diamond = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_diamond_digit_value_l3341_334151


namespace NUMINAMATH_CALUDE_roots_sum_abs_l3341_334169

theorem roots_sum_abs (d e f n : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + n = (x - d) * (x - e) * (x - f)) →
  abs d + abs e + abs f = 98 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_abs_l3341_334169


namespace NUMINAMATH_CALUDE_smallest_mersenne_prime_above_30_l3341_334111

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def mersenne_prime (p : ℕ) : Prop :=
  is_prime p ∧ ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem smallest_mersenne_prime_above_30 :
  ∃ p : ℕ, mersenne_prime p ∧ p > 30 ∧ 
  ∀ q : ℕ, mersenne_prime q → q > 30 → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_mersenne_prime_above_30_l3341_334111


namespace NUMINAMATH_CALUDE_dog_cord_length_l3341_334163

theorem dog_cord_length (diameter : ℝ) (h : diameter = 30) : 
  diameter / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_cord_length_l3341_334163


namespace NUMINAMATH_CALUDE_min_value_theorem_l3341_334162

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 8/n = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 8/y = 4 → 8*m + n ≤ 8*x + y :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3341_334162


namespace NUMINAMATH_CALUDE_candies_per_block_l3341_334116

theorem candies_per_block (candies_per_house : ℕ) (houses_per_block : ℕ) : 
  candies_per_house = 7 → houses_per_block = 5 → candies_per_house * houses_per_block = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_candies_per_block_l3341_334116


namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l3341_334131

def candidate1_votes : ℕ := 1136
def candidate2_votes : ℕ := 8236
def candidate3_votes : ℕ := 11628

def total_votes : ℕ := candidate1_votes + candidate2_votes + candidate3_votes

def winning_votes : ℕ := max candidate1_votes (max candidate2_votes candidate3_votes)

theorem winning_candidate_percentage :
  (winning_votes : ℚ) / (total_votes : ℚ) * 100 = 58.14 := by
  sorry

end NUMINAMATH_CALUDE_winning_candidate_percentage_l3341_334131


namespace NUMINAMATH_CALUDE_car_journey_time_proof_l3341_334100

/-- Represents the speed and distance of a car's journey -/
structure CarJourney where
  speed : ℝ
  distance : ℝ
  time : ℝ

/-- Given two car journeys, proves that the time taken by the second car
    is 4/3 hours under specific conditions -/
theorem car_journey_time_proof
  (m n : CarJourney)
  (h1 : m.time = 4)
  (h2 : n.speed = 3 * m.speed)
  (h3 : n.distance = 3 * m.distance)
  (h4 : m.distance = m.speed * m.time)
  (h5 : n.distance = n.speed * n.time) :
  n.time = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_car_journey_time_proof_l3341_334100


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l3341_334138

theorem fraction_zero_solution (x : ℝ) : 
  (x + 2) / (2 * x - 4) = 0 ↔ x = -2 ∧ 2 * x - 4 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_solution_l3341_334138


namespace NUMINAMATH_CALUDE_profit_maximized_at_25_l3341_334141

/-- Profit function for the commodity -/
def profit (x : ℤ) : ℤ := (x - 20) * (30 - x)

/-- Theorem stating that profit is maximized at x = 25 -/
theorem profit_maximized_at_25 :
  ∀ x : ℤ, 20 ≤ x → x ≤ 30 → profit x ≤ profit 25 :=
by
  sorry

#check profit_maximized_at_25

end NUMINAMATH_CALUDE_profit_maximized_at_25_l3341_334141


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3341_334184

theorem arithmetic_computation : (-12 * 6) - (-4 * -8) + (-15 * -3) - (36 / -2) = -77 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3341_334184


namespace NUMINAMATH_CALUDE_streetlight_problem_l3341_334149

/-- The number of streetlights -/
def total_streetlights : ℕ := 12

/-- The number of streetlights that need to be turned off -/
def lights_to_turn_off : ℕ := 4

/-- The number of available positions to turn off lights, considering the constraints -/
def available_positions : ℕ := total_streetlights - 5

/-- The number of ways to choose 4 non-adjacent positions from 7 available positions -/
def ways_to_turn_off_lights : ℕ := Nat.choose available_positions lights_to_turn_off

theorem streetlight_problem :
  ways_to_turn_off_lights = 35 :=
sorry

end NUMINAMATH_CALUDE_streetlight_problem_l3341_334149


namespace NUMINAMATH_CALUDE_chord_relations_l3341_334175

/-- Represents a chord in a unit circle -/
structure Chord where
  length : ℝ

/-- Represents the configuration of chords in the unit circle -/
structure CircleChords where
  MP : Chord
  PQ : Chord
  NR : Chord
  MN : Chord

/-- The given configuration of chords satisfying the problem conditions -/
def given_chords : CircleChords :=
  { MP := ⟨1⟩
  , PQ := ⟨1⟩
  , NR := ⟨2⟩
  , MN := ⟨3⟩ }

theorem chord_relations (c : CircleChords) (h : c = given_chords) :
  (c.MN.length - c.NR.length = 1) ∧
  (c.MN.length * c.NR.length = 6) ∧
  (c.MN.length ^ 2 - c.NR.length ^ 2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_chord_relations_l3341_334175


namespace NUMINAMATH_CALUDE_largest_movable_n_l3341_334147

/-- Represents the rules for moving cards between boxes -/
structure CardMoveRules where
  /-- A card can be placed in an empty box -/
  place_in_empty : Bool
  /-- A card can be placed on top of a card with a number one greater than its own -/
  place_on_greater : Bool

/-- Represents the configuration of card boxes -/
structure BoxConfiguration where
  /-- Number of blue boxes -/
  k : Nat
  /-- Number of cards (2n) -/
  card_count : Nat
  /-- Rules for moving cards -/
  move_rules : CardMoveRules

/-- Determines if all cards can be moved to blue boxes given a configuration -/
def can_move_all_cards (config : BoxConfiguration) : Prop :=
  ∃ (final_state : List (List Nat)), 
    final_state.length = config.k ∧ 
    final_state.all (λ box => box.length > 0) ∧
    final_state.join.toFinset = Finset.range config.card_count

/-- The main theorem stating the largest possible n for which all cards can be moved -/
theorem largest_movable_n (k : Nat) (h : k > 1) :
  ∀ n : Nat, (
    let config := BoxConfiguration.mk k (2 * n) 
      { place_in_empty := true, place_on_greater := true }
    can_move_all_cards config ↔ n ≤ k - 1
  ) := by sorry

end NUMINAMATH_CALUDE_largest_movable_n_l3341_334147


namespace NUMINAMATH_CALUDE_weight_difference_l3341_334118

/-- Given the weights of Mildred and Carol, prove the difference in their weights. -/
theorem weight_difference (mildred_weight carol_weight : ℕ) 
  (h1 : mildred_weight = 59) 
  (h2 : carol_weight = 9) : 
  mildred_weight - carol_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3341_334118


namespace NUMINAMATH_CALUDE_complex_product_zero_l3341_334102

theorem complex_product_zero (z : ℂ) (h : z^2 + 1 = 0) :
  (z^4 + Complex.I) * (z^4 - Complex.I) = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_product_zero_l3341_334102


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3341_334157

-- Define the quadratic function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties (b c : ℝ) :
  (∀ α : ℝ, f b c (Real.sin α) ≥ 0) →
  (∀ β : ℝ, f b c (2 + Real.cos β) ≤ 0) →
  (∃ M : ℝ, M = 8 ∧ ∀ α : ℝ, f b c (Real.sin α) ≤ M) →
  b = -4 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3341_334157


namespace NUMINAMATH_CALUDE_original_fraction_problem_l3341_334168

theorem original_fraction_problem (N D : ℚ) :
  (N > 0) →
  (D > 0) →
  ((1.4 * N) / (0.5 * D) = 4/5) →
  (N / D = 2/7) :=
by sorry

end NUMINAMATH_CALUDE_original_fraction_problem_l3341_334168


namespace NUMINAMATH_CALUDE_g_equality_l3341_334197

-- Define the function g
def g : ℝ → ℝ := fun x ↦ -4 * x^4 + 2 * x^3 - 5 * x^2 + x + 4

-- State the theorem
theorem g_equality (x : ℝ) : 4 * x^4 + 2 * x^2 - x + g x = 2 * x^3 - 3 * x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_g_equality_l3341_334197


namespace NUMINAMATH_CALUDE_cat_food_sale_l3341_334106

theorem cat_food_sale (total_customers : Nat) (first_group : Nat) (middle_group : Nat) (last_group : Nat)
  (first_group_cases : Nat) (last_group_cases : Nat) (total_cases : Nat)
  (h1 : total_customers = first_group + middle_group + last_group)
  (h2 : total_customers = 20)
  (h3 : first_group = 8)
  (h4 : middle_group = 4)
  (h5 : last_group = 8)
  (h6 : first_group_cases = 3)
  (h7 : last_group_cases = 1)
  (h8 : total_cases = 40)
  (h9 : total_cases = first_group * first_group_cases + middle_group * x + last_group * last_group_cases)
  : x = 2 := by
  sorry

#check cat_food_sale

end NUMINAMATH_CALUDE_cat_food_sale_l3341_334106


namespace NUMINAMATH_CALUDE_total_balloons_count_l3341_334134

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 60

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 85

/-- The number of blue balloons Alex has -/
def alex_balloons : ℕ := 37

/-- The total number of blue balloons -/
def total_balloons : ℕ := joan_balloons + melanie_balloons + alex_balloons

theorem total_balloons_count : total_balloons = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_count_l3341_334134


namespace NUMINAMATH_CALUDE_second_smallest_odd_number_l3341_334159

/-- Given a sequence of four consecutive odd numbers whose sum is 112,
    the second smallest number in this sequence is 27. -/
theorem second_smallest_odd_number : ∀ (a b c d : ℤ),
  (∃ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5 ∧ d = 2*n + 7) →  -- consecutive odd numbers
  (a + b + c + d = 112) →                                            -- sum is 112
  b = 27                                                             -- second smallest is 27
:= by sorry

end NUMINAMATH_CALUDE_second_smallest_odd_number_l3341_334159


namespace NUMINAMATH_CALUDE_sales_volume_correct_profit_10000_prices_max_profit_under_constraints_l3341_334135

/-- Toy sales model -/
structure ToySalesModel where
  purchase_price : ℝ
  initial_price : ℝ
  initial_volume : ℝ
  price_sensitivity : ℝ
  min_price : ℝ
  min_volume : ℝ

/-- Given toy sales model -/
def given_model : ToySalesModel :=
  { purchase_price := 30
  , initial_price := 40
  , initial_volume := 600
  , price_sensitivity := 10
  , min_price := 44
  , min_volume := 540 }

/-- Sales volume as a function of price -/
def sales_volume (model : ToySalesModel) (x : ℝ) : ℝ :=
  model.initial_volume - model.price_sensitivity * (x - model.initial_price)

/-- Profit as a function of price -/
def profit (model : ToySalesModel) (x : ℝ) : ℝ :=
  (x - model.purchase_price) * (sales_volume model x)

/-- Theorem stating the correctness of the sales volume function -/
theorem sales_volume_correct (x : ℝ) (h : x > given_model.initial_price) :
  sales_volume given_model x = 1000 - 10 * x := by sorry

/-- Theorem stating the selling prices for a profit of 10,000 yuan -/
theorem profit_10000_prices :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ profit given_model x₁ = 10000 ∧ profit given_model x₂ = 10000 ∧
  (x₁ = 50 ∨ x₁ = 80) ∧ (x₂ = 50 ∨ x₂ = 80) := by sorry

/-- Theorem stating the maximum profit under constraints -/
theorem max_profit_under_constraints :
  ∃ x : ℝ, x ≥ given_model.min_price ∧ 
    sales_volume given_model x ≥ given_model.min_volume ∧
    ∀ y : ℝ, y ≥ given_model.min_price → 
      sales_volume given_model y ≥ given_model.min_volume →
      profit given_model x ≥ profit given_model y ∧
      profit given_model x = 8640 := by sorry

end NUMINAMATH_CALUDE_sales_volume_correct_profit_10000_prices_max_profit_under_constraints_l3341_334135


namespace NUMINAMATH_CALUDE_ellipse_b_value_l3341_334148

/-- Define an ellipse with foci F1 and F2, and a point P on the ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  P : ℝ × ℝ
  h1 : a > b
  h2 : b > 0
  h3 : P.1^2 / a^2 + P.2^2 / b^2 = 1  -- P is on the ellipse

/-- The dot product of PF1 and PF2 is zero -/
def orthogonal_foci (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0

/-- The area of triangle PF1F2 is 9 -/
def triangle_area (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  abs (PF1.1 * PF2.2 - PF1.2 * PF2.1) / 2 = 9

/-- Main theorem: If the foci are orthogonal from P and the triangle area is 9, then b = 3 -/
theorem ellipse_b_value (e : Ellipse) 
  (h_orth : orthogonal_foci e) (h_area : triangle_area e) : e.b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_b_value_l3341_334148


namespace NUMINAMATH_CALUDE_book_sale_profit_l3341_334130

/-- Prove that given two books with a total cost of 480, where the first book costs 280 and is sold
    at a 15% loss, and both books are sold at the same price, the gain percentage on the second book
    is 19%. -/
theorem book_sale_profit (total_cost : ℝ) (cost_book1 : ℝ) (loss_percentage : ℝ) 
  (h1 : total_cost = 480)
  (h2 : cost_book1 = 280)
  (h3 : loss_percentage = 15)
  (h4 : ∃ (sell_price : ℝ), sell_price = cost_book1 * (1 - loss_percentage / 100) ∧ 
        sell_price = (total_cost - cost_book1) * (1 + x / 100)) :
  x = 19 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_profit_l3341_334130


namespace NUMINAMATH_CALUDE_fifth_term_of_8998_sequence_l3341_334119

-- Define the sequence generation function
def generateSequence (n : ℕ) : List ℕ :=
  -- Implementation of the sequence generation rules
  sorry

-- Define a function to get the nth term of a sequence
def getNthTerm (sequence : List ℕ) (n : ℕ) : ℕ :=
  -- Implementation to get the nth term
  sorry

-- Theorem statement
theorem fifth_term_of_8998_sequence :
  getNthTerm (generateSequence 8998) 5 = 4625 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_8998_sequence_l3341_334119


namespace NUMINAMATH_CALUDE_parabola_focus_l3341_334187

/-- A parabola is defined by its equation relating x and y coordinates -/
structure Parabola where
  equation : ℝ → ℝ

/-- The focus of a parabola is a point (x, y) -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Predicate to check if a given point is the focus of a parabola -/
def is_focus (p : Parabola) (f : Focus) : Prop :=
  ∀ (y : ℝ), 
    let x := p.equation y
    (x - f.x)^2 + y^2 = (x - (f.x - 3))^2

/-- Theorem stating that (-3, 0) is the focus of the parabola x = -1/12 * y^2 -/
theorem parabola_focus :
  let p : Parabola := ⟨λ y => -1/12 * y^2⟩
  let f : Focus := ⟨-3, 0⟩
  is_focus p f := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l3341_334187


namespace NUMINAMATH_CALUDE_no_2013_numbers_exist_l3341_334173

theorem no_2013_numbers_exist : ¬ ∃ (S : Finset ℕ), 
  (S.card = 2013) ∧ 
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y) ∧
  (∀ a ∈ S, (S.sum id - a) ≥ a^2) :=
by sorry

end NUMINAMATH_CALUDE_no_2013_numbers_exist_l3341_334173


namespace NUMINAMATH_CALUDE_trader_loss_percentage_l3341_334192

theorem trader_loss_percentage (cost_price : ℝ) (cost_price_pos : cost_price > 0) : 
  let marked_price := cost_price * 1.1
  let selling_price := marked_price * 0.9
  let loss := cost_price - selling_price
  loss / cost_price = 0.01 := by
sorry

end NUMINAMATH_CALUDE_trader_loss_percentage_l3341_334192


namespace NUMINAMATH_CALUDE_swing_wait_time_l3341_334181

/-- Proves that the wait time for swings is 4.75 minutes given the problem conditions -/
theorem swing_wait_time :
  let kids_for_swings : ℕ := 3
  let kids_for_slide : ℕ := 6
  let slide_wait_time : ℝ := 15
  let wait_time_difference : ℝ := 270
  let swing_wait_time : ℝ := (slide_wait_time + wait_time_difference) / 60
  swing_wait_time = 4.75 := by
sorry

#eval (15 + 270) / 60  -- Should output 4.75

end NUMINAMATH_CALUDE_swing_wait_time_l3341_334181


namespace NUMINAMATH_CALUDE_intersection_y_coordinate_l3341_334172

/-- The y-coordinate of the intersection point between a line and a parabola -/
theorem intersection_y_coordinate (x : ℝ) : 
  x > 0 ∧ 
  (x - 1)^2 + 1 = -2*x + 11 → 
  -2*x + 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_coordinate_l3341_334172


namespace NUMINAMATH_CALUDE_r_squared_equals_one_for_linear_plot_l3341_334124

/-- A scatter plot where all points lie on a straight line -/
structure LinearScatterPlot where
  /-- The slope of the line on which all points lie -/
  slope : ℝ
  /-- All points in the scatter plot lie on a straight line -/
  all_points_on_line : Bool

/-- The coefficient of determination (R²) for a scatter plot -/
def r_squared (plot : LinearScatterPlot) : ℝ :=
  sorry

/-- Theorem: If all points in a scatter plot lie on a straight line with a slope of 2,
    then R² equals 1 -/
theorem r_squared_equals_one_for_linear_plot (plot : LinearScatterPlot)
    (h1 : plot.slope = 2)
    (h2 : plot.all_points_on_line = true) :
    r_squared plot = 1 := by
  sorry

end NUMINAMATH_CALUDE_r_squared_equals_one_for_linear_plot_l3341_334124


namespace NUMINAMATH_CALUDE_lesser_fraction_l3341_334191

theorem lesser_fraction (x y : ℚ) (h_sum : x + y = 3/4) (h_product : x * y = 1/8) : 
  min x y = 1/4 := by sorry

end NUMINAMATH_CALUDE_lesser_fraction_l3341_334191


namespace NUMINAMATH_CALUDE_g_1003_fixed_point_l3341_334101

def g₁ (x : ℚ) : ℚ := 1/2 - 4/(4*x+2)

def g (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | 1 => g₁ x
  | n+1 => g₁ (g n x)

theorem g_1003_fixed_point :
  g 1003 (11/2) = 11/2 - 4 := by sorry

end NUMINAMATH_CALUDE_g_1003_fixed_point_l3341_334101


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3341_334179

theorem intersection_of_lines :
  ∃! (x y : ℚ), (6 * x - 5 * y = 10) ∧ (8 * x + 2 * y = 20) ∧ (x = 30 / 13) ∧ (y = 10 / 13) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3341_334179


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_rectangular_prism_diagonal_h12_l3341_334155

/-- Theorem: Diagonal of a rectangular prism with specific dimensions --/
theorem rectangular_prism_diagonal (h : ℝ) (l : ℝ) (w : ℝ) : 
  h = 12 → l = 2 * h → w = l / 2 → 
  Real.sqrt (l^2 + w^2 + h^2) = 12 * Real.sqrt 6 := by
  sorry

/-- Corollary: Specific case with h = 12 --/
theorem rectangular_prism_diagonal_h12 : 
  ∃ (h l w : ℝ), h = 12 ∧ l = 2 * h ∧ w = l / 2 ∧ 
  Real.sqrt (l^2 + w^2 + h^2) = 12 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_rectangular_prism_diagonal_h12_l3341_334155


namespace NUMINAMATH_CALUDE_zhang_apple_sales_l3341_334127

/-- Represents the number of apples Zhang needs to sell to earn a specific profit -/
def apples_to_sell (buy_price : ℚ) (sell_price : ℚ) (target_profit : ℚ) : ℚ :=
  target_profit / (sell_price - buy_price)

/-- Theorem stating the number of apples Zhang needs to sell to earn 15 yuan -/
theorem zhang_apple_sales : 
  let buy_price : ℚ := 1 / 4  -- 4 apples for 1 yuan
  let sell_price : ℚ := 2 / 5 -- 5 apples for 2 yuan
  let target_profit : ℚ := 15
  apples_to_sell buy_price sell_price target_profit = 100 :=
by
  sorry

#eval apples_to_sell (1/4) (2/5) 15

end NUMINAMATH_CALUDE_zhang_apple_sales_l3341_334127


namespace NUMINAMATH_CALUDE_power_multiplication_l3341_334146

theorem power_multiplication (a : ℝ) : a^3 * a^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3341_334146


namespace NUMINAMATH_CALUDE_min_value_problem_l3341_334153

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := |x + a| + |x - b| + c

-- State the theorem
theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, f a b c x ≥ 4) 
  (hmin_exists : ∃ x, f a b c x = 4) : 
  (a + b + c = 4) ∧ 
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) ∧
  (∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 = 8/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3341_334153


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3341_334152

/-- The maximum value of the quadratic function f(x) = -2x^2 + 4x - 18 is -16 -/
theorem quadratic_max_value :
  let f : ℝ → ℝ := fun x ↦ -2 * x^2 + 4 * x - 18
  ∃ M : ℝ, M = -16 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3341_334152


namespace NUMINAMATH_CALUDE_quadratic_expansion_l3341_334182

theorem quadratic_expansion (m n : ℝ) :
  (∀ x : ℝ, (x + 4) * (x - 2) = x^2 + m*x + n) →
  m = 2 ∧ n = -8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expansion_l3341_334182


namespace NUMINAMATH_CALUDE_history_not_statistics_l3341_334110

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ) :
  total = 90 →
  history = 36 →
  statistics = 30 →
  history_or_statistics = 59 →
  history - (history + statistics - history_or_statistics) = 29 := by
  sorry

end NUMINAMATH_CALUDE_history_not_statistics_l3341_334110


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3341_334186

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 7) * (x^2 + 6*x + 8) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3341_334186


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_product_l3341_334199

theorem pure_imaginary_complex_product (a : ℝ) :
  let z : ℂ := (1 - 2*I) * (a - I) * I
  (z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_product_l3341_334199


namespace NUMINAMATH_CALUDE_mixed_committee_probability_l3341_334125

def total_members : ℕ := 30
def num_boys : ℕ := 13
def num_girls : ℕ := 17
def committee_size : ℕ := 6

def probability_mixed_committee : ℚ :=
  1 - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size) / Nat.choose total_members committee_size

theorem mixed_committee_probability :
  probability_mixed_committee = 579683 / 593775 :=
by sorry

end NUMINAMATH_CALUDE_mixed_committee_probability_l3341_334125


namespace NUMINAMATH_CALUDE_intersection_sum_l3341_334188

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x - 4
def g (x y : ℝ) : Prop := x + 5*y = 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = p.2 ∧ g p.1 p.2}

-- Theorem statement
theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    x₁ + x₂ + x₃ = 5 ∧
    y₁ + y₂ + y₃ = 2 :=
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l3341_334188


namespace NUMINAMATH_CALUDE_permutation_residue_systems_l3341_334160

theorem permutation_residue_systems (n : ℕ) : 
  (∃ p : Fin n → Fin n, Function.Bijective p ∧ 
    (∀ (i : Fin n), ∃ (j : Fin n), (p j + j : ℕ) % n = i) ∧
    (∀ (i : Fin n), ∃ (j : Fin n), (p j - j : ℤ) % n = i)) ↔ 
  (n % 6 = 1 ∨ n % 6 = 5) :=
sorry

end NUMINAMATH_CALUDE_permutation_residue_systems_l3341_334160


namespace NUMINAMATH_CALUDE_relationship_abc_l3341_334143

theorem relationship_abc (a b c : ℝ) 
  (eq1 : b + c = 6 - 4*a + 3*a^2)
  (eq2 : c - b = 4 - 4*a + a^2) : 
  a < b ∧ b ≤ c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3341_334143


namespace NUMINAMATH_CALUDE_ellipse_constants_l3341_334167

/-- An ellipse with given foci and a point on its curve -/
structure Ellipse where
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  point : ℝ × ℝ

/-- The standard form constants of an ellipse -/
structure EllipseConstants where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

/-- Theorem: For an ellipse with foci at (1, 3) and (1, 7) passing through (12, 0),
    the constants in the standard form equation are as given -/
theorem ellipse_constants (e : Ellipse) 
    (h_focus1 : e.focus1 = (1, 3))
    (h_focus2 : e.focus2 = (1, 7))
    (h_point : e.point = (12, 0)) :
    ∃ (c : EllipseConstants), 
      c.a = (Real.sqrt 130 + Real.sqrt 170) / 2 ∧
      c.b = Real.sqrt (((Real.sqrt 130 + Real.sqrt 170) / 2)^2 - 4^2) ∧
      c.h = 1 ∧
      c.k = 5 ∧
      c.a > 0 ∧
      c.b > 0 ∧
      (e.point.1 - c.h)^2 / c.a^2 + (e.point.2 - c.k)^2 / c.b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_constants_l3341_334167


namespace NUMINAMATH_CALUDE_prob_at_least_9_is_0_7_l3341_334139

/-- A shooter has probabilities of scoring different points in one shot. -/
structure Shooter where
  prob_10 : ℝ  -- Probability of scoring 10 points
  prob_9 : ℝ   -- Probability of scoring 9 points
  prob_8_or_less : ℝ  -- Probability of scoring 8 or fewer points
  sum_to_one : prob_10 + prob_9 + prob_8_or_less = 1  -- Probabilities sum to 1

/-- The probability of scoring at least 9 points is the sum of probabilities of scoring 10 and 9 points. -/
def prob_at_least_9 (s : Shooter) : ℝ := s.prob_10 + s.prob_9

/-- Given the probabilities for a shooter, prove that the probability of scoring at least 9 points is 0.7. -/
theorem prob_at_least_9_is_0_7 (s : Shooter) 
    (h1 : s.prob_10 = 0.4) 
    (h2 : s.prob_9 = 0.3) 
    (h3 : s.prob_8_or_less = 0.3) : 
  prob_at_least_9 s = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_9_is_0_7_l3341_334139


namespace NUMINAMATH_CALUDE_race_head_start_l3341_334170

theorem race_head_start (Va Vb L H : ℝ) :
  Va = (22 / 19) * Vb →
  L / Va = (L - H) / Vb →
  H = (3 / 22) * L :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l3341_334170


namespace NUMINAMATH_CALUDE_tangent_directrix_parabola_circle_l3341_334120

/-- Given a circle and a parabola with a tangent directrix, prove the value of m -/
theorem tangent_directrix_parabola_circle (m : ℝ) : 
  m > 0 → 
  (∃ (x y : ℝ), x^2 + y^2 = 1/4) →
  (∃ (x y : ℝ), y = m * x^2) →
  (∃ (d : ℝ), d = 1/(4*m) ∧ d = 1/2) →
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_tangent_directrix_parabola_circle_l3341_334120


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l3341_334136

theorem prime_square_mod_twelve (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l3341_334136


namespace NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l3341_334122

def f (x : ℝ) := (20 * x + (20 * x + 13) ^ (1/3)) ^ (1/3)

theorem unique_solution_cube_root_equation :
  ∃! x : ℝ, f x = 13 ∧ x = 546/5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l3341_334122


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3341_334165

theorem chess_tournament_games (n : Nat) (h : n = 5) : 
  n * (n - 1) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3341_334165


namespace NUMINAMATH_CALUDE_grade_distribution_l3341_334195

theorem grade_distribution (n : ℕ) : 
  ∃ (a b c m : ℕ),
    (2 * m + 3 = n) ∧  -- Total students
    (b = a + 2) ∧      -- B grades
    (c = 2 * b) ∧      -- C grades
    (4 * a + 6 ≠ n)    -- Total A, B, C grades ≠ Total students
  := by sorry

end NUMINAMATH_CALUDE_grade_distribution_l3341_334195


namespace NUMINAMATH_CALUDE_circle_radius_l3341_334140

/-- The radius of the circle defined by x^2 + y^2 + 2x + 6y = 0 is √10 -/
theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 + 2*x + 6*y = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3341_334140


namespace NUMINAMATH_CALUDE_ratio_problem_l3341_334176

theorem ratio_problem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) :
  (3 * a + 2 * b) / (b + 4 * c) = 3 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3341_334176


namespace NUMINAMATH_CALUDE_cost_per_crayon_is_two_l3341_334166

/-- The number of crayons in half a dozen -/
def half_dozen : ℕ := 6

/-- The number of half dozens Jamal bought -/
def number_of_half_dozens : ℕ := 4

/-- The total cost of the crayons in dollars -/
def total_cost : ℕ := 48

/-- The total number of crayons Jamal bought -/
def total_crayons : ℕ := number_of_half_dozens * half_dozen

/-- The cost per crayon in dollars -/
def cost_per_crayon : ℚ := total_cost / total_crayons

theorem cost_per_crayon_is_two : cost_per_crayon = 2 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_crayon_is_two_l3341_334166


namespace NUMINAMATH_CALUDE_lcm_problem_l3341_334132

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by sorry

end NUMINAMATH_CALUDE_lcm_problem_l3341_334132


namespace NUMINAMATH_CALUDE_spending_problem_l3341_334190

theorem spending_problem (initial_amount : ℚ) : 
  (2 / 5 : ℚ) * initial_amount = 600 → initial_amount = 1500 := by
  sorry

end NUMINAMATH_CALUDE_spending_problem_l3341_334190


namespace NUMINAMATH_CALUDE_annie_passes_bonnie_at_six_laps_l3341_334194

/-- Represents the track and runners' properties -/
structure RaceSetup where
  trackLength : ℝ
  annieSpeedFactor : ℝ
  bonnieAcceleration : ℝ

/-- Calculates the number of laps Annie runs when she first passes Bonnie -/
def lapsWhenAnniePasses (setup : RaceSetup) (bonnieInitialSpeed : ℝ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem annie_passes_bonnie_at_six_laps (setup : RaceSetup) (bonnieInitialSpeed : ℝ) 
    (h1 : setup.trackLength = 300)
    (h2 : setup.annieSpeedFactor = 1.2)
    (h3 : setup.bonnieAcceleration = 0.1) :
  lapsWhenAnniePasses setup bonnieInitialSpeed = 6 := by
  sorry

end NUMINAMATH_CALUDE_annie_passes_bonnie_at_six_laps_l3341_334194


namespace NUMINAMATH_CALUDE_f_even_implies_a_zero_f_min_value_when_a_zero_f_never_odd_l3341_334117

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

-- Part I: If f is even, then a = 0
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 :=
sorry

-- Part II: When a = 0, the minimum value of f is 1
theorem f_min_value_when_a_zero :
  ∀ x, f 0 x ≥ 1 :=
sorry

-- Part III: f can never be an odd function for any real a
theorem f_never_odd (a : ℝ) :
  ¬(∀ x, f a x = -(f a (-x))) :=
sorry

end NUMINAMATH_CALUDE_f_even_implies_a_zero_f_min_value_when_a_zero_f_never_odd_l3341_334117


namespace NUMINAMATH_CALUDE_andre_purchase_total_l3341_334171

/-- Calculates the discounted price given the original price and discount percentage. -/
def discountedPrice (originalPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  originalPrice * (1 - discountPercentage / 100)

/-- Calculates the total price for multiple items with the same price. -/
def totalPrice (itemPrice : ℚ) (quantity : ℕ) : ℚ :=
  itemPrice * quantity

theorem andre_purchase_total : 
  let treadmillOriginalPrice : ℚ := 1350
  let treadmillDiscount : ℚ := 30
  let plateOriginalPrice : ℚ := 60
  let plateQuantity : ℕ := 2
  let plateDiscount : ℚ := 15
  
  discountedPrice treadmillOriginalPrice treadmillDiscount + 
  discountedPrice (totalPrice plateOriginalPrice plateQuantity) plateDiscount = 1047 := by
sorry

end NUMINAMATH_CALUDE_andre_purchase_total_l3341_334171


namespace NUMINAMATH_CALUDE_parabola_theorem_l3341_334105

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  h : a ≠ 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  m : ℝ
  b : ℝ

/-- Definition of the parabola E: x^2 = 4y -/
def parabolaE : Parabola := ⟨4, by norm_num⟩

/-- Focus of the parabola -/
def focusF : Point := ⟨0, 1⟩

/-- Origin point -/
def originO : Point := ⟨0, 0⟩

/-- Function to calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Function to check if two lines are parallel -/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Theorem statement -/
theorem parabola_theorem (l : Line) (A B : Point) 
  (h1 : A.x^2 = 4 * A.y) -- A is on the parabola
  (h2 : B.x^2 = 4 * B.y) -- B is on the parabola
  (h3 : focusF.y = l.m * focusF.x + l.b) -- l passes through F
  (h4 : A.y = l.m * A.x + l.b) -- A is on l
  (h5 : B.y = l.m * B.x + l.b) -- B is on l
  :
  (∃ (minArea : ℝ), minArea = 2 ∧ 
    ∀ (A' B' : Point), A'.x^2 = 4 * A'.y → B'.x^2 = 4 * B'.y → 
    A'.y = l.m * A'.x + l.b → B'.y = l.m * B'.x + l.b →
    triangleArea originO A' B' ≥ minArea) ∧ 
  (∃ (C : Point) (lAO lBC : Line), 
    C.y = -1 ∧ -- C is on the directrix
    A.y = lAO.m * A.x + lAO.b ∧ -- AO line
    originO.y = lAO.m * originO.x + lAO.b ∧
    C.y = lAO.m * C.x + lAO.b ∧
    B.x = C.x ∧ -- BC is vertical
    isParallel lBC ⟨0, 1⟩) -- BC is parallel to y-axis
  := by sorry

end NUMINAMATH_CALUDE_parabola_theorem_l3341_334105


namespace NUMINAMATH_CALUDE_average_age_problem_l3341_334104

theorem average_age_problem (a b c : ℕ) : 
  (a + c) / 2 = 29 →
  b = 17 →
  (a + b + c) / 3 = 25 := by
sorry

end NUMINAMATH_CALUDE_average_age_problem_l3341_334104


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3341_334115

/-- An arithmetic sequence with the given property -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the common difference of the specific arithmetic sequence -/
theorem arithmetic_sequence_difference (a : ℕ → ℝ) 
    (h : ArithmeticSequence a) (h2015 : a 2015 = a 2013 + 6) : 
    ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3341_334115


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3341_334180

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 - y^2 = a^2

-- Define the semi-latus rectum of a parabola
def semi_latus_rectum (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

theorem hyperbola_real_axis_length 
  (a p x1 y1 x2 y2 : ℝ) 
  (h1 : hyperbola a x1 y1)
  (h2 : hyperbola a x2 y2)
  (h3 : semi_latus_rectum p x1)
  (h4 : semi_latus_rectum p x2)
  (h5 : distance x1 y1 x2 y2 = 4 * (3^(1/2))) :
  2 * a = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3341_334180


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l3341_334133

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n = 15625) ∧ 
  (∀ m : ℕ, m < n → m < 10000 ∨ m > 99999 ∨ ¬∃ a : ℕ, m = a^2 ∨ ¬∃ b : ℕ, m = b^3) ∧
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n = b^3) :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l3341_334133


namespace NUMINAMATH_CALUDE_vector_equation_l3341_334183

theorem vector_equation (a b c : ℝ × ℝ) : 
  a = (1, 1) → b = (1, -1) → c = (-1, -2) → 
  c = (-3/2 : ℝ) • a + (1/2 : ℝ) • b :=
by sorry

end NUMINAMATH_CALUDE_vector_equation_l3341_334183


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l3341_334185

theorem fifteenth_student_age
  (n : ℕ)
  (total_students : n = 15)
  (avg_age : ℝ)
  (total_avg : avg_age = 15)
  (group1_size group2_size : ℕ)
  (group1_avg group2_avg : ℝ)
  (group_sizes : group1_size = 7 ∧ group2_size = 7)
  (group_avgs : group1_avg = 14 ∧ group2_avg = 16)
  : ∃ (fifteenth_age : ℝ), fifteenth_age = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l3341_334185


namespace NUMINAMATH_CALUDE_train_stoppage_time_l3341_334121

/-- Calculates the stoppage time per hour for a train given its speeds with and without stoppages. -/
theorem train_stoppage_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 48) 
  (h2 : speed_with_stops = 40) : 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_stoppage_time_l3341_334121


namespace NUMINAMATH_CALUDE_total_spent_l3341_334156

def lunch_cost : ℝ := 60.50
def tip_percentage : ℝ := 20

theorem total_spent (lunch_cost : ℝ) (tip_percentage : ℝ) : 
  lunch_cost * (1 + tip_percentage / 100) = 72.60 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_l3341_334156


namespace NUMINAMATH_CALUDE_x_sixth_plus_inverse_l3341_334174

theorem x_sixth_plus_inverse (x : ℝ) (h : x + 1/x = 7) : x^6 + 1/x^6 = 103682 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_plus_inverse_l3341_334174


namespace NUMINAMATH_CALUDE_pass_in_later_rounds_l3341_334189

/-- Represents the probability of correctly answering each question -/
structure QuestionProbabilities where
  A : ℚ
  B : ℚ
  C : ℚ

/-- Represents the interview process -/
def Interview (probs : QuestionProbabilities) : Prop :=
  probs.A = 1/2 ∧ probs.B = 1/3 ∧ probs.C = 1/4

/-- The probability of passing the interview in the second or third round -/
def PassInLaterRounds (probs : QuestionProbabilities) : ℚ :=
  7/18

/-- Theorem stating the probability of passing in later rounds -/
theorem pass_in_later_rounds (probs : QuestionProbabilities) 
  (h : Interview probs) : 
  PassInLaterRounds probs = 7/18 := by
  sorry


end NUMINAMATH_CALUDE_pass_in_later_rounds_l3341_334189


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_315_l3341_334103

theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_315_l3341_334103


namespace NUMINAMATH_CALUDE_race_track_width_l3341_334128

/-- The width of a circular race track given its inner circumference and outer radius -/
theorem race_track_width (inner_circumference outer_radius : ℝ) :
  inner_circumference = 880 →
  outer_radius = 165.0563499208679 →
  ∃ width : ℝ, (abs (width - 25.049) < 0.001 ∧
    width = outer_radius - inner_circumference / (2 * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_race_track_width_l3341_334128


namespace NUMINAMATH_CALUDE_exists_ten_digit_number_with_composite_subnumbers_l3341_334196

/-- A ten-digit number composed of ten different digits. -/
def TenDigitNumber := Fin 10 → Fin 10

/-- Checks if a number is composite. -/
def IsComposite (n : ℕ) : Prop := ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- Checks if a four-digit number is composite. -/
def IsFourDigitComposite (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000 ∧ IsComposite n

/-- Generates all four-digit numbers from a ten-digit number by removing six digits. -/
def FourDigitSubnumbers (n : TenDigitNumber) : Set ℕ :=
  {m | ∃ (i j k l : Fin 10), i < j ∧ j < k ∧ k < l ∧
    m = n i * 1000 + n j * 100 + n k * 10 + n l}

/-- The main theorem stating that there exists a ten-digit number with the required property. -/
theorem exists_ten_digit_number_with_composite_subnumbers :
  ∃ (n : TenDigitNumber),
    (∀ i j, i ≠ j → n i ≠ n j) ∧
    (∀ m ∈ FourDigitSubnumbers n, IsFourDigitComposite m) := by
  sorry

end NUMINAMATH_CALUDE_exists_ten_digit_number_with_composite_subnumbers_l3341_334196


namespace NUMINAMATH_CALUDE_problem_proof_l3341_334123

theorem problem_proof : (1 / (2 - Real.sqrt 3)) - 1 - 2 * (Real.sqrt 3 / 2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_problem_proof_l3341_334123


namespace NUMINAMATH_CALUDE_max_table_sum_l3341_334137

def numbers : List ℕ := [3, 5, 7, 11, 17, 19]

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ 
  d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧
  (a = b ∧ b = c) ∨ (d = e ∧ e = f)

def table_sum (a b c d e f : ℕ) : ℕ :=
  a*d + a*e + a*f + b*d + b*e + b*f + c*d + c*e + c*f

theorem max_table_sum :
  ∀ a b c d e f : ℕ,
    is_valid_arrangement a b c d e f →
    table_sum a b c d e f ≤ 1995 ∧
    (∃ a b c d e f : ℕ, 
      is_valid_arrangement a b c d e f ∧ 
      table_sum a b c d e f = 1995 ∧
      (a = 19 ∧ b = 19 ∧ c = 19) ∨ (d = 19 ∧ e = 19 ∧ f = 19)) := by
  sorry

end NUMINAMATH_CALUDE_max_table_sum_l3341_334137


namespace NUMINAMATH_CALUDE_inequality_implication_l3341_334154

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3341_334154


namespace NUMINAMATH_CALUDE_basketball_game_difference_l3341_334114

/-- Given a ratio of boys to girls and the number of girls, 
    calculate the difference between the number of boys and girls -/
def boys_girls_difference (boys_ratio : ℕ) (girls_ratio : ℕ) (num_girls : ℕ) : ℕ :=
  let num_boys := (num_girls / girls_ratio) * boys_ratio
  num_boys - num_girls

/-- Theorem stating that with a ratio of 8:5 boys to girls and 30 girls, 
    there are 18 more boys than girls -/
theorem basketball_game_difference : boys_girls_difference 8 5 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_difference_l3341_334114


namespace NUMINAMATH_CALUDE_abs_sum_gt_abs_diff_when_product_positive_l3341_334158

theorem abs_sum_gt_abs_diff_when_product_positive (a b : ℝ) (h : a * b > 0) :
  |a + b| > |a - b| := by sorry

end NUMINAMATH_CALUDE_abs_sum_gt_abs_diff_when_product_positive_l3341_334158


namespace NUMINAMATH_CALUDE_dans_initial_green_marbles_count_l3341_334109

def dans_initial_green_marbles : ℕ := sorry

def mikes_taken_marbles : ℕ := 23

def dans_remaining_green_marbles : ℕ := 9

theorem dans_initial_green_marbles_count : 
  dans_initial_green_marbles = dans_remaining_green_marbles + mikes_taken_marbles := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_green_marbles_count_l3341_334109


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l3341_334113

theorem pie_crust_flour_calculation (total_flour : ℚ) (original_crusts new_crusts : ℕ) :
  total_flour > 0 →
  original_crusts > 0 →
  new_crusts > 0 →
  (total_flour / original_crusts) * new_crusts = total_flour →
  total_flour / new_crusts = 1 / 5 := by
  sorry

#check pie_crust_flour_calculation (5 : ℚ) 40 25

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l3341_334113


namespace NUMINAMATH_CALUDE_shortest_distance_to_circle_l3341_334112

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 24*x + y^2 + 10*y + 160 = 0

/-- The shortest distance from the origin to the circle -/
def shortest_distance : ℝ := 10

theorem shortest_distance_to_circle :
  ∀ p : ℝ × ℝ, circle_equation p.1 p.2 →
  ∃ q : ℝ × ℝ, circle_equation q.1 q.2 ∧
  ∀ r : ℝ × ℝ, circle_equation r.1 r.2 →
  Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) ≤ Real.sqrt ((r.1 - 0)^2 + (r.2 - 0)^2) ∧
  Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) = shortest_distance := by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_to_circle_l3341_334112


namespace NUMINAMATH_CALUDE_fraction_of_states_1790_to_1799_l3341_334129

theorem fraction_of_states_1790_to_1799 (total_states : ℕ) (states_1790_to_1799 : ℕ) : 
  total_states = 30 → states_1790_to_1799 = 9 → (states_1790_to_1799 : ℚ) / total_states = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_states_1790_to_1799_l3341_334129


namespace NUMINAMATH_CALUDE_largest_number_with_quotient_30_l3341_334164

theorem largest_number_with_quotient_30 : 
  ∀ n : ℕ, n ≤ 960 ∧ n / 31 = 30 → n = 960 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_quotient_30_l3341_334164


namespace NUMINAMATH_CALUDE_stadium_seats_l3341_334177

/-- Represents the number of seats in the little league stadium -/
def total_seats : ℕ := sorry

/-- Represents the number of people who came to the game -/
def people_at_game : ℕ := 47

/-- Represents the number of people holding banners -/
def people_with_banners : ℕ := 38

/-- Represents the number of empty seats -/
def empty_seats : ℕ := 45

/-- Theorem stating that the total number of seats is equal to the sum of people at the game and empty seats -/
theorem stadium_seats : total_seats = people_at_game + empty_seats := by sorry

end NUMINAMATH_CALUDE_stadium_seats_l3341_334177


namespace NUMINAMATH_CALUDE_group_earnings_l3341_334145

/-- Represents the wage of a man in rupees -/
def man_wage : ℕ := 6

/-- Represents the number of men in the group -/
def num_men : ℕ := 5

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 8

/-- Represents the number of women in the group (unknown) -/
def num_women : ℕ := sorry

/-- The total amount earned by the group -/
def total_amount : ℕ := 3 * (num_men * man_wage)

theorem group_earnings : 
  total_amount = 90 := by sorry

end NUMINAMATH_CALUDE_group_earnings_l3341_334145


namespace NUMINAMATH_CALUDE_f_divisible_by_factorial_l3341_334150

def f : ℕ → ℕ → ℕ
  | 0, 0 => 1
  | 0, _ => 0
  | _, 0 => 0
  | n+1, k+1 => (n+1) * (f (n+1) k + f n k)

theorem f_divisible_by_factorial (n k : ℕ) : 
  ∃ m : ℤ, f n k = n! * m := by sorry

end NUMINAMATH_CALUDE_f_divisible_by_factorial_l3341_334150


namespace NUMINAMATH_CALUDE_ginger_garden_work_hours_l3341_334126

/-- Calculates the number of hours Ginger worked in her garden --/
def hours_worked (bottle_capacity : ℕ) (bottles_for_plants : ℕ) (total_water_used : ℕ) : ℕ :=
  (total_water_used - bottles_for_plants * bottle_capacity) / bottle_capacity

/-- Proves that Ginger worked 8 hours in her garden given the problem conditions --/
theorem ginger_garden_work_hours :
  let bottle_capacity : ℕ := 2
  let bottles_for_plants : ℕ := 5
  let total_water_used : ℕ := 26
  hours_worked bottle_capacity bottles_for_plants total_water_used = 8 := by
  sorry


end NUMINAMATH_CALUDE_ginger_garden_work_hours_l3341_334126


namespace NUMINAMATH_CALUDE_consecutive_integers_base_sum_l3341_334161

/-- Represents a number in a given base -/
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem consecutive_integers_base_sum (C D : Nat) : 
  C.succ = D →
  C < D →
  to_base_10 [2, 3, 1] C + to_base_10 [5, 6] D = to_base_10 [1, 0, 5] (C + D) →
  C + D = 7 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_base_sum_l3341_334161


namespace NUMINAMATH_CALUDE_bananas_in_jar_l3341_334198

/-- The number of bananas originally in the jar -/
def original_bananas : ℕ := 46

/-- The number of bananas removed from the jar -/
def removed_bananas : ℕ := 5

/-- The number of bananas left in the jar after removal -/
def remaining_bananas : ℕ := 41

/-- Theorem stating that the original number of bananas is equal to the sum of removed and remaining bananas -/
theorem bananas_in_jar : original_bananas = removed_bananas + remaining_bananas := by
  sorry

end NUMINAMATH_CALUDE_bananas_in_jar_l3341_334198
