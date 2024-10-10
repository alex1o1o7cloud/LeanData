import Mathlib

namespace machine_profit_percentage_l1443_144319

/-- Calculates the profit percentage given the purchase price, repair cost, transportation charges, and selling price of a machine. -/
def profit_percentage (purchase_price repair_cost transport_charges selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost + transport_charges
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem: The profit percentage for the given machine transaction is 50%. -/
theorem machine_profit_percentage :
  profit_percentage 13000 5000 1000 28500 = 50 := by
  sorry

end machine_profit_percentage_l1443_144319


namespace no_quadratic_with_discriminant_23_l1443_144367

theorem no_quadratic_with_discriminant_23 :
  ¬ ∃ (a b c : ℤ), b^2 - 4*a*c = 23 := by
  sorry

end no_quadratic_with_discriminant_23_l1443_144367


namespace remainder_sum_l1443_144385

theorem remainder_sum (n : ℤ) : n % 20 = 11 → (n % 4 + n % 5 = 4) := by
  sorry

end remainder_sum_l1443_144385


namespace binomial_expansion_coefficient_l1443_144376

theorem binomial_expansion_coefficient (x : ℝ) : 
  let expansion := (x^2 - 2/x)^7
  ∃ (terms : List ℝ), 
    expansion = (terms.map (λ t => t * x^5)).sum ∧ 
    (terms.filter (λ t => t ≠ 0)).length = 1 ∧
    (terms.filter (λ t => t ≠ 0)).head! = -280 :=
by sorry

end binomial_expansion_coefficient_l1443_144376


namespace sqrt_product_simplification_l1443_144321

theorem sqrt_product_simplification (p : ℝ) :
  Real.sqrt (8 * p^2) * Real.sqrt (12 * p^3) * Real.sqrt (18 * p^5) = 24 * p^5 * Real.sqrt 3 := by
  sorry

end sqrt_product_simplification_l1443_144321


namespace subcommittee_count_l1443_144307

/-- The number of members in the planning committee -/
def total_members : ℕ := 12

/-- The number of teachers in the planning committee -/
def teacher_count : ℕ := 5

/-- The size of the subcommittee to be formed -/
def subcommittee_size : ℕ := 4

/-- The minimum number of teachers required in the subcommittee -/
def min_teachers : ℕ := 2

/-- Calculates the number of valid subcommittees -/
def valid_subcommittees : ℕ := 285

theorem subcommittee_count :
  (Nat.choose total_members subcommittee_size) -
  (Nat.choose (total_members - teacher_count) subcommittee_size) -
  (Nat.choose teacher_count 1 * Nat.choose (total_members - teacher_count) (subcommittee_size - 1)) =
  valid_subcommittees :=
sorry

end subcommittee_count_l1443_144307


namespace carpet_fits_both_rooms_l1443_144313

-- Define the carpet and room dimensions
def carpet_width : ℝ := 25
def carpet_length : ℝ := 50
def room1_width : ℝ := 38
def room1_length : ℝ := 55
def room2_width : ℝ := 50
def room2_length : ℝ := 55

-- Define a function to check if the carpet fits in a room
def carpet_fits_room (carpet_w carpet_l room_w room_l : ℝ) : Prop :=
  carpet_w^2 + carpet_l^2 = room_w^2 + room_l^2

-- Theorem statement
theorem carpet_fits_both_rooms :
  carpet_fits_room carpet_width carpet_length room1_width room1_length ∧
  carpet_fits_room carpet_width carpet_length room2_width room2_length :=
by sorry

end carpet_fits_both_rooms_l1443_144313


namespace jacks_change_jacks_change_is_five_l1443_144324

/-- Given Jack's sandwich order and payment, calculate his change -/
theorem jacks_change (num_sandwiches : ℕ) (price_per_sandwich : ℕ) (payment : ℕ) : ℕ :=
  let total_cost := num_sandwiches * price_per_sandwich
  payment - total_cost

/-- Prove that Jack's change is $5 given the problem conditions -/
theorem jacks_change_is_five : 
  jacks_change 3 5 20 = 5 := by
  sorry

end jacks_change_jacks_change_is_five_l1443_144324


namespace price_reduction_l1443_144329

theorem price_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_reduction := 1 - 0.08
  let second_reduction := 1 - 0.10
  let final_price := original_price * first_reduction * second_reduction
  final_price / original_price = 0.828 := by
sorry

end price_reduction_l1443_144329


namespace parallel_lines_c_value_l1443_144325

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 7x + 3 and y = (3c)x + 5 are parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = 7 * x + 3 ↔ y = (3 * c) * x + 5) → c = 7 / 3 :=
by sorry

end parallel_lines_c_value_l1443_144325


namespace hyperbola_eccentricity_l1443_144340

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_equation : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptote : ∀ x, ∃ y, y = Real.sqrt 3 / 3 * x ∨ y = -Real.sqrt 3 / 3 * x) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 * Real.sqrt 3 / 3 := by
  sorry

end hyperbola_eccentricity_l1443_144340


namespace equation_solutions_l1443_144364

def satisfies_equation (x y z : ℕ) : Prop :=
  x^2 + y^2 = 9 + z^2 - 2*x*y

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(0,5,4), (1,4,4), (2,3,4), (3,2,4), (4,1,4), (5,0,4), (0,3,0), (1,2,0), (2,1,0), (3,0,0)}

theorem equation_solutions :
  ∀ x y z : ℕ, satisfies_equation x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end equation_solutions_l1443_144364


namespace polynomial_with_three_equal_roots_l1443_144337

theorem polynomial_with_three_equal_roots (a b : ℤ) : 
  (∃ r : ℤ, (∀ x : ℝ, x^4 + x^3 - 18*x^2 + a*x + b = 0 ↔ 
    (x = r ∨ x = r ∨ x = r ∨ x = ((-1 : ℝ) - 3*r)))) → 
  (a = -52 ∧ b = -40) := by
sorry

end polynomial_with_three_equal_roots_l1443_144337


namespace inequality_proof_l1443_144354

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * y ∧ y = z) :=
by sorry

end inequality_proof_l1443_144354


namespace joint_purchase_savings_l1443_144396

/-- Represents the store's tile offer structure -/
structure TileOffer where
  regularPrice : ℕ  -- Regular price per tile
  buyQuantity : ℕ   -- Number of tiles to buy
  freeQuantity : ℕ  -- Number of free tiles given

/-- Calculates the cost of purchasing a given number of tiles under the offer -/
def calculateCost (offer : TileOffer) (tilesNeeded : ℕ) : ℕ :=
  let fullSets := tilesNeeded / (offer.buyQuantity + offer.freeQuantity)
  let remainingTiles := tilesNeeded % (offer.buyQuantity + offer.freeQuantity)
  fullSets * offer.buyQuantity * offer.regularPrice + remainingTiles * offer.regularPrice

/-- Theorem stating the savings when Dave and Doug purchase together -/
theorem joint_purchase_savings (offer : TileOffer) (daveTiles dougTiles : ℕ) :
  offer.regularPrice = 150 ∧ 
  offer.buyQuantity = 9 ∧ 
  offer.freeQuantity = 2 ∧
  daveTiles = 11 ∧
  dougTiles = 13 →
  calculateCost offer daveTiles + calculateCost offer dougTiles - 
  calculateCost offer (daveTiles + dougTiles) = 600 := by
  sorry

end joint_purchase_savings_l1443_144396


namespace probability_calculation_l1443_144323

def total_silverware : ℕ := 8 + 7 + 5

def probability_2forks_1spoon_1knife (forks spoons knives total : ℕ) : ℚ :=
  let favorable_outcomes := Nat.choose forks 2 * Nat.choose spoons 1 * Nat.choose knives 1
  let total_outcomes := Nat.choose total 4
  favorable_outcomes / total_outcomes

theorem probability_calculation :
  probability_2forks_1spoon_1knife 8 7 5 total_silverware = 196 / 969 := by
  sorry

end probability_calculation_l1443_144323


namespace cube_root_sqrt_64_l1443_144371

theorem cube_root_sqrt_64 : 
  {x : ℝ | x^3 = Real.sqrt 64} = {2, -2} := by sorry

end cube_root_sqrt_64_l1443_144371


namespace quadratic_ratio_l1443_144379

/-- 
Given a quadratic expression x^2 + 1440x + 1600, which can be written in the form (x + d)^2 + e,
prove that e/d = -718.
-/
theorem quadratic_ratio (d e : ℝ) : 
  (∀ x, x^2 + 1440*x + 1600 = (x + d)^2 + e) → e/d = -718 := by
  sorry

end quadratic_ratio_l1443_144379


namespace ratio_equality_l1443_144353

theorem ratio_equality (x : ℝ) :
  (0.75 / x = 5 / 8) → x = 1.2 := by
  sorry

end ratio_equality_l1443_144353


namespace shortest_side_of_special_triangle_l1443_144334

theorem shortest_side_of_special_triangle :
  ∀ (a b c : ℕ),
    a = 17 →
    a + b + c = 50 →
    (∃ A : ℕ, A^2 = (a + b + c) * (b + c - a) * (a + c - b) * (a + b - c) / 16) →
    b ≥ 13 ∧ c ≥ 13 :=
by sorry

end shortest_side_of_special_triangle_l1443_144334


namespace league_matches_count_l1443_144301

/-- The number of teams in the league -/
def num_teams : ℕ := 14

/-- The number of matches played in the league -/
def total_matches : ℕ := num_teams * (num_teams - 1)

/-- Theorem stating that the total number of matches in the league is 182 -/
theorem league_matches_count :
  total_matches = 182 :=
sorry

end league_matches_count_l1443_144301


namespace rancher_feed_corn_cost_l1443_144373

/-- Represents the rancher's farm and animals -/
structure Farm where
  sheep : ℕ
  cattle : ℕ
  pasture_acres : ℕ
  cow_grass_consumption : ℕ
  sheep_grass_consumption : ℕ
  feed_corn_cost : ℕ
  feed_corn_cow_duration : ℕ
  feed_corn_sheep_duration : ℕ

/-- Calculates the yearly cost of feed corn for the farm -/
def yearly_feed_corn_cost (f : Farm) : ℕ :=
  let total_monthly_grass_consumption := f.cattle * f.cow_grass_consumption + f.sheep * f.sheep_grass_consumption
  let grazing_months := f.pasture_acres / total_monthly_grass_consumption
  let feed_corn_months := 12 - grazing_months
  let monthly_feed_corn_bags := f.cattle + f.sheep / f.feed_corn_sheep_duration
  let total_feed_corn_bags := monthly_feed_corn_bags * feed_corn_months
  total_feed_corn_bags * f.feed_corn_cost

/-- The main theorem stating the yearly cost of feed corn for the given farm -/
theorem rancher_feed_corn_cost :
  let farm := Farm.mk 8 5 144 2 1 10 1 2
  yearly_feed_corn_cost farm = 360 := by
  sorry

end rancher_feed_corn_cost_l1443_144373


namespace square_sum_from_difference_and_product_l1443_144348

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) (h2 : x * y = 16) : x^2 + y^2 = 356 := by
  sorry

end square_sum_from_difference_and_product_l1443_144348


namespace perfect_square_condition_l1443_144343

/-- If x^2 + 6x + k^2 is a perfect square polynomial, then k = ± 3 -/
theorem perfect_square_condition (k : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, x^2 + 6*x + k^2 = (p x)^2) → 
  k = 3 ∨ k = -3 := by
sorry

end perfect_square_condition_l1443_144343


namespace find_k_l1443_144388

theorem find_k (k : ℝ) (h : 64 / k = 8) : k = 8 := by
  sorry

end find_k_l1443_144388


namespace sum_of_digits_Y_squared_l1443_144394

/-- The number of digits in 222222222 -/
def n : ℕ := 9

/-- The digit repeated in 222222222 -/
def d : ℕ := 2

/-- The number 222222222 -/
def Y : ℕ := d * (10^n - 1) / 9

/-- Sum of digits function -/
def sum_of_digits (x : ℕ) : ℕ := sorry

/-- Theorem: The sum of the digits of the square of 222222222 is 162 -/
theorem sum_of_digits_Y_squared : sum_of_digits (Y^2) = 162 := by sorry

end sum_of_digits_Y_squared_l1443_144394


namespace age_difference_l1443_144349

/-- Given three people a, b, and c, with their ages satisfying certain conditions,
    prove that a is 2 years older than b. -/
theorem age_difference (a b c : ℕ) : 
  b = 28 →                  -- b is 28 years old
  b = 2 * c →               -- b is twice as old as c
  a + b + c = 72 →          -- The total of the ages of a, b, and c is 72
  a = b + 2 :=              -- a is 2 years older than b
by
  sorry

end age_difference_l1443_144349


namespace flashlight_visibility_difference_l1443_144327

/-- Flashlight visibility problem -/
theorem flashlight_visibility_difference (veronica_visibility : ℝ) :
  veronica_visibility = 1000 →
  let freddie_visibility := 3 * veronica_visibility
  let velma_visibility := 5 * freddie_visibility - 2000
  let daphne_visibility := (veronica_visibility + freddie_visibility + velma_visibility) / 3
  let total_visibility := veronica_visibility + freddie_visibility + velma_visibility + daphne_visibility
  total_visibility = 40000 →
  ∃ ε > 0, |velma_visibility - daphne_visibility - 7666.67| < ε :=
by
  sorry

end flashlight_visibility_difference_l1443_144327


namespace matrix_determinant_equals_four_l1443_144347

def A (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![3*y, 2], ![3, y]]

theorem matrix_determinant_equals_four (y : ℝ) :
  Matrix.det (A y) = 4 ↔ y = Real.sqrt (10/3) ∨ y = -Real.sqrt (10/3) := by
  sorry

end matrix_determinant_equals_four_l1443_144347


namespace max_value_of_f_l1443_144351

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - 4 * x^3

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ ∀ x, x ∈ Set.Icc 0 1 → f x ≤ f c ∧ f c = 1 :=
sorry

end max_value_of_f_l1443_144351


namespace cube_plus_inverse_cube_l1443_144305

theorem cube_plus_inverse_cube (a : ℝ) (h : (a + 1 / (3 * a))^2 = 3) : 
  27 * a^3 + 1 / a^3 = 54 * Real.sqrt 3 ∨ 27 * a^3 + 1 / a^3 = -54 * Real.sqrt 3 :=
by sorry

end cube_plus_inverse_cube_l1443_144305


namespace existence_of_good_subset_l1443_144356

def M : Finset ℕ := Finset.range 20

def is_valid_function (f : Finset ℕ → ℕ) : Prop :=
  ∀ S : Finset ℕ, S ⊆ M → S.card = 9 → f S ∈ M

theorem existence_of_good_subset (f : Finset ℕ → ℕ) (h : is_valid_function f) :
  ∃ T : Finset ℕ, T ⊆ M ∧ T.card = 10 ∧ ∀ k ∈ T, f (T \ {k}) ≠ k := by sorry

end existence_of_good_subset_l1443_144356


namespace square_root_of_1024_l1443_144355

theorem square_root_of_1024 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 1024) : x = 32 := by
  sorry

end square_root_of_1024_l1443_144355


namespace combined_stock_cost_value_l1443_144331

/-- Calculate the final cost of a stock given its initial parameters -/
def calculate_stock_cost (initial_price discount brokerage tax_rate transaction_fee : ℚ) : ℚ :=
  let discounted_price := initial_price * (1 - discount)
  let brokerage_fee := discounted_price * brokerage
  let net_purchase_price := discounted_price + brokerage_fee
  let tax := net_purchase_price * tax_rate
  net_purchase_price + tax + transaction_fee

/-- The combined cost of three stocks with given parameters -/
def combined_stock_cost : ℚ :=
  calculate_stock_cost 100 (4/100) (1/500) (12/100) 2 +
  calculate_stock_cost 200 (6/100) (1/400) (10/100) 3 +
  calculate_stock_cost 150 (3/100) (1/200) (15/100) 1

/-- Theorem stating the combined cost of the three stocks -/
theorem combined_stock_cost_value : 
  combined_stock_cost = 489213665/1000000 := by sorry

end combined_stock_cost_value_l1443_144331


namespace max_rope_length_l1443_144391

theorem max_rope_length (a b c d : ℕ) 
  (ha : a = 48) (hb : b = 72) (hc : c = 108) (hd : d = 120) : 
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 12 := by
  sorry

end max_rope_length_l1443_144391


namespace banana_permutations_l1443_144370

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multinomial_coefficient (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / (ks.map factorial).prod

theorem banana_permutations :
  let total_letters : ℕ := 6
  let b_count : ℕ := 1
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  multinomial_coefficient total_letters [b_count, a_count, n_count] = 60 := by
  sorry

end banana_permutations_l1443_144370


namespace fraction_equivalence_l1443_144366

theorem fraction_equivalence : 
  let n : ℚ := 13/2
  (4 + n) / (7 + n) = 7/9 := by sorry

end fraction_equivalence_l1443_144366


namespace factorization_proof_l1443_144352

theorem factorization_proof (a b x y m : ℝ) : 
  (3 * a^2 - 6 * a * b + 3 * b^2 = 3 * (a - b)^2) ∧ 
  (x^2 * (m - 2) + y^2 * (2 - m) = (m - 2) * (x + y) * (x - y)) := by
  sorry

end factorization_proof_l1443_144352


namespace arithmetic_sequence_ratio_l1443_144395

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : ∀ n, S seq n / S seq (2 * n) = (n + 1 : ℚ) / (4 * n + 2)) :
  seq.a 3 / seq.a 5 = 3 / 5 := by
  sorry

end arithmetic_sequence_ratio_l1443_144395


namespace sequence_inequality_l1443_144361

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, a n ≥ 0)
  (h_ineq : ∀ m n, a (m + n) ≤ a m + a n) :
  ∀ m n, m > 0 → n ≥ m → a n ≤ m * a 1 + (n / m - 1) * a m :=
by sorry

end sequence_inequality_l1443_144361


namespace one_in_A_l1443_144308

def A : Set ℝ := {x : ℝ | x ≥ -1}

theorem one_in_A : (1 : ℝ) ∈ A := by
  sorry

end one_in_A_l1443_144308


namespace no_quadratic_trinomials_satisfying_equation_l1443_144393

/-- A quadratic trinomial -/
structure QuadraticTrinomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluate a quadratic trinomial at a given value -/
def QuadraticTrinomial.eval (p : QuadraticTrinomial) (x : ℤ) : ℤ :=
  p.a * x^2 + p.b * x + p.c

/-- Statement: There do not exist quadratic trinomials P, Q, R such that
    for all integers x and y, there exists an integer z satisfying P(x) + Q(y) = R(z) -/
theorem no_quadratic_trinomials_satisfying_equation :
  ¬∃ (P Q R : QuadraticTrinomial), ∀ (x y : ℤ), ∃ (z : ℤ),
    P.eval x + Q.eval y = R.eval z := by
  sorry

end no_quadratic_trinomials_satisfying_equation_l1443_144393


namespace marble_probability_l1443_144311

theorem marble_probability (green yellow white : ℕ) 
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_white : white = 6) :
  (green + yellow : ℚ) / (green + yellow + white) = 7 / 13 :=
by sorry

end marble_probability_l1443_144311


namespace sum_of_intersection_points_l1443_144346

/-- A type representing a line in a plane -/
structure Line :=
  (id : ℕ)

/-- A type representing an intersection point of two lines -/
structure IntersectionPoint :=
  (line1 : Line)
  (line2 : Line)

/-- A configuration of lines in a plane -/
structure LineConfiguration :=
  (lines : Finset Line)
  (intersections : Finset IntersectionPoint)
  (distinct_lines : lines.card = 5)
  (no_triple_intersections : ∀ p q r : Line, p ∈ lines → q ∈ lines → r ∈ lines → 
    p ≠ q → q ≠ r → p ≠ r → 
    ¬∃ i : IntersectionPoint, i ∈ intersections ∧ 
      (i.line1 = p ∧ i.line2 = q) ∧
      (i.line1 = q ∧ i.line2 = r) ∧
      (i.line1 = p ∧ i.line2 = r))

/-- The theorem to be proved -/
theorem sum_of_intersection_points (config : LineConfiguration) :
  (Finset.range 11).sum (λ n => n * (Finset.filter (λ c : LineConfiguration => c.intersections.card = n) {config}).card) = 54 :=
sorry

end sum_of_intersection_points_l1443_144346


namespace triangle_inequality_l1443_144381

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  |a/b + b/c + c/a - b/a - c/b - a/c| < 1 := by
sorry

end triangle_inequality_l1443_144381


namespace mia_sixth_game_shots_l1443_144302

-- Define the initial conditions
def initial_shots : ℕ := 50
def initial_made : ℕ := 20
def new_shots : ℕ := 15

-- Define the function to calculate the new shooting average
def new_average (x : ℕ) : ℚ :=
  (initial_made + x : ℚ) / (initial_shots + new_shots : ℚ)

-- Theorem statement
theorem mia_sixth_game_shots :
  ∃ x : ℕ, x ≤ new_shots ∧ new_average x = 45 / 100 :=
by
  -- The proof goes here
  sorry

end mia_sixth_game_shots_l1443_144302


namespace product_of_sum_of_squares_l1443_144377

theorem product_of_sum_of_squares (x₁ y₁ x₂ y₂ : ℝ) :
  ∃ u v : ℝ, (x₁^2 + y₁^2) * (x₂^2 + y₂^2) = u^2 + v^2 := by
  sorry

end product_of_sum_of_squares_l1443_144377


namespace shopkeeper_loss_percent_l1443_144306

theorem shopkeeper_loss_percent
  (initial_value : ℝ)
  (profit_margin : ℝ)
  (theft_percentage : ℝ)
  (h_profit : profit_margin = 0.1)
  (h_theft : theft_percentage = 0.6)
  (h_initial_positive : initial_value > 0) :
  let selling_price := initial_value * (1 + profit_margin)
  let remaining_value := initial_value * (1 - theft_percentage)
  let remaining_selling_price := selling_price * (1 - theft_percentage)
  let loss := initial_value - remaining_selling_price
  let loss_percent := (loss / initial_value) * 100
  loss_percent = 56 := by
sorry

end shopkeeper_loss_percent_l1443_144306


namespace probability_theorem_l1443_144303

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ m : ℕ, a * b + a + b = 7 * m - 2

def count_valid_pairs : ℕ := Nat.choose 100 2

def count_satisfying_pairs : ℕ := 1295

theorem probability_theorem :
  (count_satisfying_pairs : ℚ) / count_valid_pairs = 259 / 990 :=
sorry

end probability_theorem_l1443_144303


namespace women_equal_to_five_men_l1443_144300

/-- Represents the amount of work one person can do in a day -/
structure WorkPerDay (α : Type) where
  amount : ℝ

/-- Represents the total amount of work for a job -/
def Job : Type := ℝ

variable (men_work : WorkPerDay Unit) (women_work : WorkPerDay Unit)

/-- The amount of work 5 men do in a day equals the amount of work x women do in a day -/
def men_women_equal (x : ℝ) : Prop :=
  5 * men_work.amount = x * women_work.amount

/-- 3 men and 5 women finish the job in 10 days -/
def job_condition1 (job : Job) : Prop :=
  (3 * men_work.amount + 5 * women_work.amount) * 10 = job

/-- 7 women finish the job in 14 days -/
def job_condition2 (job : Job) : Prop :=
  7 * women_work.amount * 14 = job

/-- The main theorem: prove that 8 women do the same amount of work in a day as 5 men -/
theorem women_equal_to_five_men
  (job : Job)
  (h1 : job_condition1 men_work women_work job)
  (h2 : job_condition2 women_work job) :
  men_women_equal men_work women_work 8 := by
  sorry


end women_equal_to_five_men_l1443_144300


namespace percentage_failed_both_subjects_l1443_144390

theorem percentage_failed_both_subjects
  (failed_hindi : Real)
  (failed_english : Real)
  (passed_both : Real)
  (h1 : failed_hindi = 30)
  (h2 : failed_english = 42)
  (h3 : passed_both = 56) :
  100 - passed_both = failed_hindi + failed_english - (failed_hindi + failed_english - (100 - passed_both)) :=
by sorry

end percentage_failed_both_subjects_l1443_144390


namespace square_of_negative_cube_l1443_144312

theorem square_of_negative_cube (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end square_of_negative_cube_l1443_144312


namespace exponential_equation_solution_l1443_144386

theorem exponential_equation_solution :
  ∃ x : ℝ, (3 : ℝ) ^ (x - 2) = 9 ^ (x + 1) ∧ x = -4 :=
by
  sorry

end exponential_equation_solution_l1443_144386


namespace gain_percent_calculation_l1443_144357

theorem gain_percent_calculation (cost_price selling_price : ℝ) : 
  cost_price = 600 → 
  selling_price = 1080 → 
  (selling_price - cost_price) / cost_price * 100 = 80 := by
sorry

end gain_percent_calculation_l1443_144357


namespace cafeteria_seating_capacity_l1443_144368

theorem cafeteria_seating_capacity
  (total_tables : ℕ)
  (occupied_ratio : ℚ)
  (occupied_seats : ℕ)
  (h1 : total_tables = 15)
  (h2 : occupied_ratio = 9/10)
  (h3 : occupied_seats = 135) :
  (occupied_seats / occupied_ratio) / total_tables = 10 := by
  sorry

end cafeteria_seating_capacity_l1443_144368


namespace scenario_I_correct_scenario_II_correct_scenario_III_correct_scenario_IV_correct_l1443_144363

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 2

-- Define the total number of people
def total_people : ℕ := num_boys + num_girls

-- Define the arrangement functions for each scenario
def arrangements_I : ℕ := sorry

def arrangements_II : ℕ := sorry

def arrangements_III : ℕ := sorry

def arrangements_IV : ℕ := sorry

-- Theorem for scenario I
theorem scenario_I_correct : 
  arrangements_I = 48 := by sorry

-- Theorem for scenario II
theorem scenario_II_correct : 
  arrangements_II = 36 := by sorry

-- Theorem for scenario III
theorem scenario_III_correct : 
  arrangements_III = 60 := by sorry

-- Theorem for scenario IV
theorem scenario_IV_correct : 
  arrangements_IV = 78 := by sorry

end scenario_I_correct_scenario_II_correct_scenario_III_correct_scenario_IV_correct_l1443_144363


namespace sqrt_seven_fraction_l1443_144330

theorem sqrt_seven_fraction (p q : ℝ) (hp : p > 0) (hq : q > 0) (h : Real.sqrt 7 = p / q) :
  Real.sqrt 7 = (7 * q - 2 * p) / (p - 2 * q) ∧ p - 2 * q > 0 ∧ p - 2 * q < q := by
  sorry

end sqrt_seven_fraction_l1443_144330


namespace min_value_and_k_range_l1443_144372

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 2 / x

noncomputable def σ (x : ℝ) : ℝ := log x + exp x / x - x

noncomputable def g (x : ℝ) : ℝ := x^2 - x * log x + 2

theorem min_value_and_k_range :
  (∃ (x : ℝ), ∀ (y : ℝ), y > 0 → σ y ≥ σ x) ∧
  σ (1 : ℝ) = exp 1 - 1 ∧
  ∀ (k : ℝ), (∃ (a b : ℝ), 1/2 ≤ a ∧ a < b ∧
    (∀ (x : ℝ), a ≤ x ∧ x ≤ b → k * (a + 2) ≤ g x ∧ g x ≤ k * (b + 2))) →
    1 < k ∧ k ≤ (9 + 2 * log 2) / 10 :=
by sorry

end min_value_and_k_range_l1443_144372


namespace rooster_weight_problem_l1443_144328

theorem rooster_weight_problem (price_per_kg : ℝ) (weight_rooster1 : ℝ) (total_earnings : ℝ) :
  price_per_kg = 0.5 →
  weight_rooster1 = 30 →
  total_earnings = 35 →
  ∃ weight_rooster2 : ℝ,
    weight_rooster2 = 40 ∧
    total_earnings = price_per_kg * (weight_rooster1 + weight_rooster2) :=
by sorry

end rooster_weight_problem_l1443_144328


namespace percentage_failed_hindi_l1443_144309

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ) 
  (h1 : failed_english = 45)
  (h2 : failed_both = 20)
  (h3 : passed_both = 40) :
  ∃ (failed_hindi : ℝ), failed_hindi = 35 := by
sorry

end percentage_failed_hindi_l1443_144309


namespace city_fuel_efficiency_l1443_144304

/-- Fuel efficiency of a car on highway and in city -/
structure CarFuelEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city
  tank : ℝ     -- Tank capacity in gallons
  h_positive : highway > 0
  c_positive : city > 0
  t_positive : tank > 0
  city_less : city = highway - 6

/-- Theorem stating the car's fuel efficiency in the city is 18 mpg -/
theorem city_fuel_efficiency 
  (car : CarFuelEfficiency)
  (h_highway : car.highway * car.tank = 448)
  (h_city : car.city * car.tank = 336) :
  car.city = 18 := by
  sorry

end city_fuel_efficiency_l1443_144304


namespace secant_length_l1443_144360

/-- Given a circle with center O and radius r, and a point A outside the circle,
    this theorem proves the length of a secant line from A with internal segment length d. -/
theorem secant_length (O A : Point) (r d a : ℝ) (h1 : r > 0) (h2 : d > 0) (h3 : a > r) :
  ∃ x : ℝ, x = d / 2 + Real.sqrt ((d / 2) ^ 2 + a * (a + 2 * r)) ∨
           x = d / 2 - Real.sqrt ((d / 2) ^ 2 + a * (a + 2 * r)) :=
by sorry

/-- Point type (placeholder) -/
def Point : Type := sorry

end secant_length_l1443_144360


namespace rectangular_solid_surface_area_l1443_144378

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The volume of a rectangular solid is the product of its three edge lengths. -/
def volume (a b c : ℕ) : ℕ := a * b * c

/-- The surface area of a rectangular solid is twice the sum of the areas of its three distinct faces. -/
def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

/-- Theorem: For a rectangular solid with prime edge lengths and a volume of 1001 cubic units, 
    the total surface area is 622 square units. -/
theorem rectangular_solid_surface_area :
  ∀ a b c : ℕ,
  is_prime a ∧ is_prime b ∧ is_prime c →
  volume a b c = 1001 →
  surface_area a b c = 622 :=
by sorry

end rectangular_solid_surface_area_l1443_144378


namespace right_triangle_cosine_l1443_144399

theorem right_triangle_cosine (a b c : ℝ) (h1 : a = 9) (h2 : c = 15) (h3 : a^2 + b^2 = c^2) :
  (a / c) = (3 : ℝ) / 5 :=
by sorry

end right_triangle_cosine_l1443_144399


namespace fewer_bees_than_flowers_l1443_144359

theorem fewer_bees_than_flowers :
  let num_flowers : ℕ := 5
  let num_bees : ℕ := 3
  num_flowers - num_bees = 2 := by
sorry

end fewer_bees_than_flowers_l1443_144359


namespace complex_arithmetic_expression_equals_132_l1443_144322

theorem complex_arithmetic_expression_equals_132 :
  10 * 9 * 8 + 7 * 6 * 5 + 6 * 5 * 4 + 3 * 2 * 1 - 9 * 8 * 7 - 8 * 7 * 6 - 5 * 4 * 3 - 4 * 3 * 2 = 132 := by
  sorry

end complex_arithmetic_expression_equals_132_l1443_144322


namespace union_complement_equals_less_than_three_l1443_144362

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem union_complement_equals_less_than_three :
  A ∪ (univ \ B) = {x : ℝ | x < 3} := by sorry

end union_complement_equals_less_than_three_l1443_144362


namespace polynomial_simplification_l1443_144341

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 6 * x - 5) - (2 * x^3 + 2 * x^2 + 4 * x - 15) =
  x^3 + 2 * x^2 + 2 * x + 10 := by
  sorry

end polynomial_simplification_l1443_144341


namespace geometric_sum_formula_l1443_144384

/-- Geometric sequence with first term 1 and common ratio 1/3 -/
def geometric_sequence (n : ℕ) : ℚ :=
  (1 / 3) ^ (n - 1)

/-- Sum of the first n terms of the geometric sequence -/
def geometric_sum (n : ℕ) : ℚ :=
  (3 - geometric_sequence n) / 2

/-- Theorem: The sum of the first n terms of the geometric sequence
    is equal to (3 - a_n) / 2 -/
theorem geometric_sum_formula (n : ℕ) :
  geometric_sum n = (3 - geometric_sequence n) / 2 := by
  sorry

end geometric_sum_formula_l1443_144384


namespace solve_coin_problem_l1443_144333

def coin_problem (total : ℕ) (coin1 : ℕ) (coin2 : ℕ) : Prop :=
  ∃ (max min : ℕ),
    (∃ (a : ℕ), a * coin1 = total ∧ a = max) ∧
    (∃ (b c : ℕ), b * coin1 + c * coin2 = total ∧ b + c = min) ∧
    max - min = 2

theorem solve_coin_problem :
  coin_problem 45 10 25 := by sorry

end solve_coin_problem_l1443_144333


namespace average_of_six_numbers_l1443_144326

theorem average_of_six_numbers (numbers : List ℕ) :
  numbers = [12, 412, 812, 1212, 1612, 2012] →
  (numbers.sum / numbers.length : ℚ) = 1012 := by
sorry

end average_of_six_numbers_l1443_144326


namespace intersection_midpoint_l1443_144350

theorem intersection_midpoint (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = A.1 - k ∧ A.1^2 = A.2) ∧ 
    (B.2 = B.1 - k ∧ B.1^2 = B.2) ∧ 
    A ≠ B ∧
    (A.2 + B.2) / 2 = 1) →
  k = -1/2 := by sorry

end intersection_midpoint_l1443_144350


namespace fraction_of_decimals_equals_300_l1443_144318

theorem fraction_of_decimals_equals_300 : (0.3 ^ 4) / (0.03 ^ 3) = 300 := by
  sorry

end fraction_of_decimals_equals_300_l1443_144318


namespace line_through_point_parallel_to_x_axis_l1443_144375

/-- A line parallel to the x-axis has a constant y-coordinate -/
def parallel_to_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f x₁ = f x₂

/-- The equation of a line passing through (4, 2) and parallel to the x-axis -/
def line_equation : ℝ → ℝ := λ x => 2

theorem line_through_point_parallel_to_x_axis :
  line_equation 4 = 2 ∧ parallel_to_x_axis line_equation := by
  sorry

#check line_through_point_parallel_to_x_axis

end line_through_point_parallel_to_x_axis_l1443_144375


namespace frog_climb_time_l1443_144310

/-- Represents the frog's climbing problem -/
structure FrogClimb where
  well_depth : ℝ
  climb_distance : ℝ
  slip_distance : ℝ
  slip_time_ratio : ℝ

/-- Calculates the time taken for the frog to climb the well -/
def climb_time (f : FrogClimb) : ℝ :=
  sorry

/-- Theorem stating that the frog takes 22 minutes to climb the well -/
theorem frog_climb_time :
  let f : FrogClimb := {
    well_depth := 12,
    climb_distance := 3,
    slip_distance := 1,
    slip_time_ratio := 1/3
  }
  climb_time f = 22 := by sorry

end frog_climb_time_l1443_144310


namespace right_square_prism_volume_l1443_144374

/-- Represents the dimensions of a rectangle --/
structure RectangleDimensions where
  length : ℝ
  width : ℝ

/-- Represents the volume of a right square prism --/
def prism_volume (base_side : ℝ) (height : ℝ) : ℝ :=
  base_side ^ 2 * height

/-- Theorem stating the possible volumes of the right square prism --/
theorem right_square_prism_volume 
  (lateral_surface : RectangleDimensions)
  (h_length : lateral_surface.length = 12)
  (h_width : lateral_surface.width = 8) :
  ∃ (v : ℝ), (v = prism_volume 3 8 ∨ v = prism_volume 2 12) ∧ 
             (v = 72 ∨ v = 48) := by
  sorry

end right_square_prism_volume_l1443_144374


namespace aarons_brothers_l1443_144380

theorem aarons_brothers (bennett_brothers : ℕ) (h1 : bennett_brothers = 6) 
  (h2 : bennett_brothers = 2 * aaron_brothers - 2) : aaron_brothers = 4 := by
  sorry

end aarons_brothers_l1443_144380


namespace seashell_ratio_l1443_144315

def seashells_day1 : ℕ := 5
def seashells_day2 : ℕ := 7
def total_seashells : ℕ := 36

def seashells_first_two_days : ℕ := seashells_day1 + seashells_day2
def seashells_day3 : ℕ := total_seashells - seashells_first_two_days

theorem seashell_ratio :
  seashells_day3 / seashells_first_two_days = 2 := by sorry

end seashell_ratio_l1443_144315


namespace smallest_n_for_314_fraction_l1443_144345

def is_relatively_prime (m n : ℕ) : Prop := Nat.gcd m n = 1

def contains_314 (q : ℚ) : Prop :=
  ∃ k : ℕ, (10^k * q - (10^k * q).floor) * 1000 ≥ 314 ∧
            (10^k * q - (10^k * q).floor) * 1000 < 315

theorem smallest_n_for_314_fraction :
  ∃ (m n : ℕ), 
    n = 159 ∧
    m < n ∧
    is_relatively_prime m n ∧
    contains_314 (m / n) ∧
    (∀ (m' n' : ℕ), n' < 159 → m' < n' → is_relatively_prime m' n' → ¬contains_314 (m' / n')) :=
sorry

end smallest_n_for_314_fraction_l1443_144345


namespace midpoint_distance_theorem_l1443_144317

theorem midpoint_distance_theorem (s : ℝ) : 
  let P : ℝ × ℝ := (s - 3, 2)
  let Q : ℝ × ℝ := (1, s + 2)
  let M : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  (M.1 - P.1)^2 + (M.2 - P.2)^2 = 3 * s^2 / 4 →
  s = -5 - 5 * Real.sqrt 2 ∨ s = -5 + 5 * Real.sqrt 2 :=
by sorry

end midpoint_distance_theorem_l1443_144317


namespace P_intersect_Q_equals_y_leq_2_l1443_144382

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x, y = -x + 2}

-- Theorem statement
theorem P_intersect_Q_equals_y_leq_2 : P ∩ Q = {y | y ≤ 2} := by sorry

end P_intersect_Q_equals_y_leq_2_l1443_144382


namespace inequality_solution_range_l1443_144336

theorem inequality_solution_range (b : ℝ) : 
  (∃ x y : ℤ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
   (∀ z : ℤ, z < 0 → (z - b > 0 ↔ z = x ∨ z = y))) → 
  -3 ≤ b ∧ b < -2 :=
sorry

end inequality_solution_range_l1443_144336


namespace card_selection_count_l1443_144387

/-- Represents a card with two sides -/
structure Card where
  red : Nat
  blue : Nat
  red_in_range : red ≥ 1 ∧ red ≤ 12
  blue_in_range : blue ≥ 1 ∧ blue ≤ 12

/-- The set of all possible cards -/
def all_cards : Finset Card :=
  sorry

/-- A card is a duplicate if both sides have the same number -/
def is_duplicate (c : Card) : Prop :=
  c.red = c.blue

/-- Two cards have no common numbers -/
def no_common_numbers (c1 c2 : Card) : Prop :=
  c1.red ≠ c2.red ∧ c1.red ≠ c2.blue ∧ c1.blue ≠ c2.red ∧ c1.blue ≠ c2.blue

/-- The set of valid card pairs -/
def valid_pairs : Finset (Card × Card) :=
  sorry

theorem card_selection_count :
  Finset.card valid_pairs = 1386 :=
sorry

end card_selection_count_l1443_144387


namespace emilys_necklaces_l1443_144338

/-- Emily's necklace-making problem -/
theorem emilys_necklaces (necklaces : ℕ) (beads_per_necklace : ℕ) (total_beads : ℕ) 
  (h1 : necklaces = 26)
  (h2 : beads_per_necklace = 2)
  (h3 : total_beads = 52)
  (h4 : necklaces * beads_per_necklace = total_beads) :
  necklaces = total_beads / beads_per_necklace :=
by sorry

end emilys_necklaces_l1443_144338


namespace hexagons_in_50th_ring_l1443_144383

/-- The number of hexagons in the nth ring of a hexagonal array -/
def hexagons_in_ring (n : ℕ) : ℕ := 6 * n

/-- Theorem: The number of hexagons in the 50th ring is 300 -/
theorem hexagons_in_50th_ring : hexagons_in_ring 50 = 300 := by
  sorry

end hexagons_in_50th_ring_l1443_144383


namespace houses_with_garage_count_l1443_144344

/-- Represents the number of houses with various features in a development --/
structure Development where
  total : ℕ
  withPool : ℕ
  withBoth : ℕ
  withNeither : ℕ

/-- Calculates the number of houses with a two-car garage --/
def housesWithGarage (d : Development) : ℕ :=
  d.total + d.withBoth - d.withPool - d.withNeither

/-- Theorem stating that in the given development, 75 houses have a two-car garage --/
theorem houses_with_garage_count (d : Development) 
  (h1 : d.total = 85)
  (h2 : d.withPool = 40)
  (h3 : d.withBoth = 35)
  (h4 : d.withNeither = 30) :
  housesWithGarage d = 75 := by
  sorry

#eval housesWithGarage ⟨85, 40, 35, 30⟩

end houses_with_garage_count_l1443_144344


namespace probability_at_least_one_unqualified_l1443_144314

/-- The number of products -/
def total_products : ℕ := 5

/-- The number of qualified products -/
def qualified_products : ℕ := 3

/-- The number of unqualified products -/
def unqualified_products : ℕ := 2

/-- The number of products inspected -/
def inspected_products : ℕ := 2

/-- The probability of selecting at least one unqualified product -/
def prob_at_least_one_unqualified : ℚ := 7/10

/-- Theorem stating the probability of selecting at least one unqualified product -/
theorem probability_at_least_one_unqualified :
  let total_ways := Nat.choose total_products inspected_products
  let qualified_ways := Nat.choose qualified_products inspected_products
  1 - (qualified_ways : ℚ) / (total_ways : ℚ) = prob_at_least_one_unqualified :=
by sorry

end probability_at_least_one_unqualified_l1443_144314


namespace sixth_term_is_three_l1443_144332

/-- An arithmetic sequence with 10 terms where the sum of even-numbered terms is 15 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n m, a (n + 1) - a n = a (m + 1) - a m) ∧ 
  (a 2 + a 4 + a 6 + a 8 + a 10 = 15)

/-- The 6th term of the arithmetic sequence is 3 -/
theorem sixth_term_is_three (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  a 6 = 3 := by sorry

end sixth_term_is_three_l1443_144332


namespace sqrt_a_3a_sqrt_a_l1443_144316

theorem sqrt_a_3a_sqrt_a (a : ℝ) (ha : a > 0) :
  Real.sqrt (a * 3 * a * Real.sqrt a) = a ^ (3/4) := by
  sorry

end sqrt_a_3a_sqrt_a_l1443_144316


namespace f_of_5_equals_92_l1443_144342

/-- Given a function f(x) = 2x^2 + y where f(2) = 50, prove that f(5) = 92 -/
theorem f_of_5_equals_92 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y) 
  (h2 : f 2 = 50) : 
  f 5 = 92 := by
  sorry

end f_of_5_equals_92_l1443_144342


namespace min_value_of_f_l1443_144392

/-- The quadratic function f(x) = (x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem: The minimum value of f(x) = (x-1)^2 + 2 is 2 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 :=
sorry


end min_value_of_f_l1443_144392


namespace complex_equation_solution_l1443_144398

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 3 + 4 * Complex.I) → z = 4 - 3 * Complex.I := by
  sorry

end complex_equation_solution_l1443_144398


namespace ball_arrangement_count_l1443_144365

def number_of_yellow_balls : ℕ := 4
def number_of_red_balls : ℕ := 3
def total_balls : ℕ := number_of_yellow_balls + number_of_red_balls

def arrangement_count : ℕ := Nat.choose total_balls number_of_yellow_balls

theorem ball_arrangement_count :
  arrangement_count = 35 :=
by sorry

end ball_arrangement_count_l1443_144365


namespace parabola_directrix_m_l1443_144389

/-- Given a parabola with equation y = mx² and directrix y = 1/8, prove that m = -2 -/
theorem parabola_directrix_m (m : ℝ) : 
  (∀ x y : ℝ, y = m * x^2) →  -- Parabola equation
  (∃ k : ℝ, k = 1/8 ∧ ∀ x : ℝ, k = -(1 / (4 * m))) →  -- Directrix equation
  m = -2 :=
by sorry

end parabola_directrix_m_l1443_144389


namespace ten_player_tournament_matches_l1443_144369

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n.choose 2

/-- A round-robin tournament with n players -/
structure RoundRobinTournament (n : ℕ) where
  players : Fin n → Type
  plays_once : ∀ (i j : Fin n), i ≠ j → Type

theorem ten_player_tournament_matches :
  ∀ (t : RoundRobinTournament 10),
  num_matches 10 = 45 := by
  sorry

#eval num_matches 10

end ten_player_tournament_matches_l1443_144369


namespace non_pen_pencil_sales_l1443_144335

/-- The percentage of June sales for pens -/
def pen_sales : ℝ := 42

/-- The percentage of June sales for pencils -/
def pencil_sales : ℝ := 27

/-- The total percentage of all sales -/
def total_sales : ℝ := 100

/-- Theorem: The combined percentage of June sales that were not pens or pencils is 31% -/
theorem non_pen_pencil_sales : 
  total_sales - (pen_sales + pencil_sales) = 31 := by sorry

end non_pen_pencil_sales_l1443_144335


namespace perfect_square_condition_l1443_144358

theorem perfect_square_condition (n : ℕ) : 
  ∃ m : ℕ, n^5 - n^4 - 2*n^3 + 2*n^2 + n - 1 = m^2 ↔ ∃ k : ℕ, n = k^2 + 1 :=
sorry

end perfect_square_condition_l1443_144358


namespace water_transfer_result_l1443_144320

/-- Represents the volume of water transferred on a given day -/
def transfer (day : ℕ) : ℚ :=
  if day % 2 = 1 then day else day + 1

/-- Calculates the sum of transfers for odd or even days up to n days -/
def sumTransfers (n : ℕ) (isOdd : Bool) : ℚ :=
  let start := if isOdd then 1 else 2
  let count := n / 2
  count * (2 * start + (count - 1) * 4) / 2

/-- The initial volume of water in each jar (in ml) -/
def initialVolume : ℚ := 1000

/-- The number of days for which transfers occur -/
def totalDays : ℕ := 200

/-- The final volume in Maria's jar after the transfers -/
def finalVolume : ℚ := initialVolume + sumTransfers totalDays true - sumTransfers totalDays false

theorem water_transfer_result :
  finalVolume = 900 := by sorry

end water_transfer_result_l1443_144320


namespace sqrt_product_sqrt_l1443_144397

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_product_sqrt_l1443_144397


namespace circle_larger_than_unit_circle_in_larger_square_circle_may_not_touch_diamond_l1443_144339

/-- A circle defined by two inequalities -/
def SpecialCircle (x y : ℝ) : Prop :=
  (abs x + abs y ≤ (3/2) * Real.sqrt (2 * (x^2 + y^2))) ∧
  (Real.sqrt (2 * (x^2 + y^2)) ≤ 3 * max (abs x) (abs y))

/-- The circle is larger than a standard unit circle -/
theorem circle_larger_than_unit : ∃ (x y : ℝ), SpecialCircle x y ∧ x^2 + y^2 > 1 := by sorry

/-- The circle is contained within a square larger than the standard unit square -/
theorem circle_in_larger_square : ∃ (s : ℝ), s > 1 ∧ ∀ (x y : ℝ), SpecialCircle x y → max (abs x) (abs y) ≤ s := by sorry

/-- The circle may not touch all points of a diamond inscribed in the square -/
theorem circle_may_not_touch_diamond : ∃ (x y : ℝ), abs x + abs y = 1 ∧ ¬(SpecialCircle x y) := by sorry

end circle_larger_than_unit_circle_in_larger_square_circle_may_not_touch_diamond_l1443_144339
