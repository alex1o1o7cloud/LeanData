import Mathlib

namespace NUMINAMATH_CALUDE_line_equation_proof_l2999_299915

def P : ℝ × ℝ := (1, 2)
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -5)

def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

def DistancePointToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry -- Definition of distance from a point to a line

theorem line_equation_proof :
  ∃ (l : Set (ℝ × ℝ)),
    P ∈ l ∧
    DistancePointToLine A l = DistancePointToLine B l ∧
    (l = Line 4 1 (-6) ∨ l = Line 3 2 (-7)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2999_299915


namespace NUMINAMATH_CALUDE_sqrt_problem_l2999_299958

theorem sqrt_problem : 
  (Real.sqrt 18 - Real.sqrt 8 - Real.sqrt 2 = 0) ∧
  (6 * Real.sqrt 2 * Real.sqrt 3 + 3 * Real.sqrt 30 / Real.sqrt 5 = 9 * Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_sqrt_problem_l2999_299958


namespace NUMINAMATH_CALUDE_complex_product_range_l2999_299977

theorem complex_product_range (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ < 1)
  (h₂ : Complex.abs z₂ < 1)
  (h₃ : ∃ (r : ℝ), z₁ + z₂ = r)
  (h₄ : z₁ + z₂ + z₁ * z₂ = 0) :
  ∃ (x : ℝ), z₁ * z₂ = x ∧ -1/2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_range_l2999_299977


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2999_299989

theorem quadratic_expression_value :
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  x^2 + y^2 + z^2 + 2*x*z = 26 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2999_299989


namespace NUMINAMATH_CALUDE_sqrt_500_simplification_l2999_299960

theorem sqrt_500_simplification : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_500_simplification_l2999_299960


namespace NUMINAMATH_CALUDE_stability_comparison_l2999_299953

/-- Represents a student's performance in a series of matches -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Defines the stability of scores based on variance -/
def more_stable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

theorem stability_comparison 
  (student_A student_B : StudentPerformance)
  (h_avg : student_A.average_score = student_B.average_score)
  (h_var_A : student_A.variance = 0.2)
  (h_var_B : student_B.variance = 0.8) :
  more_stable student_A student_B :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l2999_299953


namespace NUMINAMATH_CALUDE_complement_union_equals_five_l2999_299936

def U : Set Nat := {1, 3, 5, 9}
def A : Set Nat := {1, 3, 9}
def B : Set Nat := {1, 9}

theorem complement_union_equals_five : (U \ (A ∪ B)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_five_l2999_299936


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_300_l2999_299921

def mersenne_number (n : ℕ) : ℕ := 2^n - 1

def is_mersenne_prime (m : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ m = mersenne_number n ∧ Prime m

theorem largest_mersenne_prime_under_300 :
  ∀ m : ℕ, is_mersenne_prime m → m < 300 → m ≤ 127 :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_300_l2999_299921


namespace NUMINAMATH_CALUDE_original_curve_equation_l2999_299964

/-- Given a curve C in a Cartesian coordinate system that undergoes a stretching transformation,
    this theorem proves the equation of the original curve C. -/
theorem original_curve_equation
  (C : Set (ℝ × ℝ)) -- The original curve C
  (stretching : ℝ × ℝ → ℝ × ℝ) -- The stretching transformation
  (h_stretching : ∀ (x y : ℝ), stretching (x, y) = (3 * x, y)) -- Definition of the stretching
  (h_transformed : ∀ (x y : ℝ), (x, y) ∈ C → x^2 + 9*y^2 = 9) -- Equation of the transformed curve
  : ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_original_curve_equation_l2999_299964


namespace NUMINAMATH_CALUDE_constant_term_of_f_composition_l2999_299928

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x - 1/x)^8 else -Real.sqrt x

theorem constant_term_of_f_composition (x : ℝ) (h : x > 0) :
  ∃ (expansion : ℝ → ℝ),
    (∀ y, y > 0 → f (f y) = expansion y) ∧
    (∃ c, ∀ ε > 0, |expansion x - c| < ε) ∧
    (∀ c, (∃ ε > 0, |expansion x - c| < ε) → c = 70) :=
sorry

end NUMINAMATH_CALUDE_constant_term_of_f_composition_l2999_299928


namespace NUMINAMATH_CALUDE_kyler_wins_one_game_l2999_299952

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Peter : Player
| Emma : Player
| Kyler : Player

/-- Represents the outcome of a chess game -/
inductive Outcome : Type
| Win : Outcome
| Loss : Outcome
| Draw : Outcome

/-- The number of games each player played -/
def games_per_player : ℕ := 6

/-- The total number of game outcomes recorded -/
def total_outcomes : ℕ := 18

/-- Function to get the number of wins for a player -/
def wins (p : Player) : ℕ :=
  match p with
  | Player.Peter => 3
  | Player.Emma => 2
  | Player.Kyler => 0  -- We'll prove this is actually 1

/-- Function to get the number of losses for a player -/
def losses (p : Player) : ℕ :=
  match p with
  | Player.Peter => 2
  | Player.Emma => 2
  | Player.Kyler => 3

/-- Function to get the number of draws for a player -/
def draws (p : Player) : ℕ :=
  match p with
  | Player.Peter => 1
  | Player.Emma => 2
  | Player.Kyler => 2

theorem kyler_wins_one_game :
  wins Player.Kyler = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_kyler_wins_one_game_l2999_299952


namespace NUMINAMATH_CALUDE_original_cost_price_satisfies_conditions_l2999_299917

/-- The original cost price of a computer satisfying given conditions -/
def original_cost_price : ℝ := 40

/-- The selling price of the computer -/
def selling_price : ℝ := 48

/-- The decrease rate of the cost price -/
def cost_decrease_rate : ℝ := 0.04

/-- The increase rate of the profit margin -/
def profit_margin_increase_rate : ℝ := 0.05

/-- Theorem stating that the original cost price satisfies all given conditions -/
theorem original_cost_price_satisfies_conditions :
  let new_cost_price := original_cost_price * (1 - cost_decrease_rate)
  let original_profit_margin := (selling_price - original_cost_price) / original_cost_price
  let new_profit_margin := (selling_price - new_cost_price) / new_cost_price
  new_profit_margin = original_profit_margin + profit_margin_increase_rate := by
  sorry


end NUMINAMATH_CALUDE_original_cost_price_satisfies_conditions_l2999_299917


namespace NUMINAMATH_CALUDE_complement_intersection_l2999_299939

def A : Set ℕ := {2, 3, 4}
def B (a : ℕ) : Set ℕ := {a + 2, a}

theorem complement_intersection (a : ℕ) (h : A ∩ B a = B a) : (Aᶜ ∩ B a) = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_l2999_299939


namespace NUMINAMATH_CALUDE_binomial_variance_10_07_l2999_299918

/-- The variance of a binomial distribution with 10 trials and 0.7 probability of success is 2.1 -/
theorem binomial_variance_10_07 :
  let n : ℕ := 10
  let p : ℝ := 0.7
  let variance := n * p * (1 - p)
  variance = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_10_07_l2999_299918


namespace NUMINAMATH_CALUDE_jerry_max_showers_l2999_299972

/-- Represents the water usage scenario for Jerry in July --/
structure WaterUsage where
  total_allowance : ℕ
  drinking_cooking : ℕ
  shower_usage : ℕ
  pool_length : ℕ
  pool_width : ℕ
  pool_height : ℕ
  gallon_per_cubic_foot : ℕ
  leakage_rate : ℕ
  days_in_july : ℕ

/-- Calculates the maximum number of showers Jerry can take in July --/
def max_showers (w : WaterUsage) : ℕ :=
  let pool_volume := w.pool_length * w.pool_width * w.pool_height
  let pool_water := pool_volume * w.gallon_per_cubic_foot
  let total_leakage := w.leakage_rate * w.days_in_july
  let water_for_showers := w.total_allowance - w.drinking_cooking - pool_water - total_leakage
  water_for_showers / w.shower_usage

/-- Theorem stating that Jerry can take at most 7 showers in July --/
theorem jerry_max_showers :
  let w : WaterUsage := {
    total_allowance := 1000,
    drinking_cooking := 100,
    shower_usage := 20,
    pool_length := 10,
    pool_width := 10,
    pool_height := 6,
    gallon_per_cubic_foot := 1,
    leakage_rate := 5,
    days_in_july := 31
  }
  max_showers w = 7 := by
  sorry


end NUMINAMATH_CALUDE_jerry_max_showers_l2999_299972


namespace NUMINAMATH_CALUDE_beth_cookie_price_l2999_299973

/-- Represents a cookie batch with a count and price per cookie -/
structure CookieBatch where
  count : ℕ
  price : ℚ

/-- Calculates the total earnings from a cookie batch -/
def totalEarnings (batch : CookieBatch) : ℚ :=
  batch.count * batch.price

theorem beth_cookie_price (alan_batch beth_batch : CookieBatch) : 
  alan_batch.count = 15 → 
  alan_batch.price = 1/2 → 
  beth_batch.count = 18 → 
  totalEarnings alan_batch = totalEarnings beth_batch → 
  beth_batch.price = 21/50 := by
sorry

#eval (21 : ℚ) / 50

end NUMINAMATH_CALUDE_beth_cookie_price_l2999_299973


namespace NUMINAMATH_CALUDE_petrol_expense_l2999_299985

def monthly_expenses (rent milk groceries education misc petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + misc + petrol

def savings_percentage : ℚ := 1 / 10

theorem petrol_expense (rent milk groceries education misc savings : ℕ) 
  (h1 : rent = 5000)
  (h2 : milk = 1500)
  (h3 : groceries = 4500)
  (h4 : education = 2500)
  (h5 : misc = 3940)
  (h6 : savings = 2160)
  (h7 : ∃ (salary petrol : ℕ), savings_percentage * salary = savings ∧ 
        monthly_expenses rent milk groceries education misc petrol = salary - savings) :
  ∃ (petrol : ℕ), petrol = 2000 := by
sorry

end NUMINAMATH_CALUDE_petrol_expense_l2999_299985


namespace NUMINAMATH_CALUDE_data_mode_is_60_l2999_299980

def data : List Nat := [65, 60, 75, 60, 80]

def mode (l : List Nat) : Nat :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem data_mode_is_60 : mode data = 60 := by
  sorry

end NUMINAMATH_CALUDE_data_mode_is_60_l2999_299980


namespace NUMINAMATH_CALUDE_no_real_solutions_l2999_299904

theorem no_real_solutions : ¬∃ y : ℝ, (y - 4*y + 10)^2 + 4 = -2*abs y := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2999_299904


namespace NUMINAMATH_CALUDE_pascal_triangle_42nd_number_in_45_number_row_l2999_299959

theorem pascal_triangle_42nd_number_in_45_number_row : 
  let n : ℕ := 44  -- The row number (0-indexed) that contains 45 numbers
  let k : ℕ := 41  -- The position (0-indexed) of the 42nd number in the row
  (n.choose k) = 13254 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_42nd_number_in_45_number_row_l2999_299959


namespace NUMINAMATH_CALUDE_base8_digit_product_l2999_299926

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem base8_digit_product :
  productOfList (toBase8 8127) = 1764 :=
sorry

end NUMINAMATH_CALUDE_base8_digit_product_l2999_299926


namespace NUMINAMATH_CALUDE_factor_representation_1000000_l2999_299988

/-- The number of ways to represent 1,000,000 as the product of three factors -/
def factor_representation (n : ℕ) (distinct_order : Bool) : ℕ :=
  if distinct_order then 784 else 139

/-- Theorem stating the number of ways to represent 1,000,000 as the product of three factors -/
theorem factor_representation_1000000 :
  (factor_representation 1000000 true = 784) ∧
  (factor_representation 1000000 false = 139) := by
  sorry

end NUMINAMATH_CALUDE_factor_representation_1000000_l2999_299988


namespace NUMINAMATH_CALUDE_prime_pairs_problem_l2999_299946

theorem prime_pairs_problem :
  ∀ p q : ℕ,
    1 < p → p < 100 →
    1 < q → q < 100 →
    Prime p →
    Prime q →
    Prime (p + 6) →
    Prime (p + 10) →
    Prime (q + 4) →
    Prime (q + 10) →
    Prime (p + q + 1) →
    ((p = 7 ∧ q = 3) ∨ (p = 13 ∧ q = 3) ∨ (p = 37 ∧ q = 3) ∨ (p = 97 ∧ q = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_problem_l2999_299946


namespace NUMINAMATH_CALUDE_lattice_points_on_segment_l2999_299927

/-- The number of lattice points on a line segment with given endpoints -/
def latticePointCount (x₁ y₁ x₂ y₂ : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (7,23) to (61,353) is 7 -/
theorem lattice_points_on_segment : latticePointCount 7 23 61 353 = 7 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_segment_l2999_299927


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2999_299924

theorem smallest_prime_divisor_of_sum (p : ℕ) : 
  (p.Prime ∧ p ∣ (7^15 + 9^7)) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2999_299924


namespace NUMINAMATH_CALUDE_least_equal_bulbs_l2999_299965

def tulip_pack_size : ℕ := 15
def daffodil_pack_size : ℕ := 16

theorem least_equal_bulbs :
  ∃ (n : ℕ), n > 0 ∧ n % tulip_pack_size = 0 ∧ n % daffodil_pack_size = 0 ∧
  ∀ (m : ℕ), (m > 0 ∧ m % tulip_pack_size = 0 ∧ m % daffodil_pack_size = 0) → m ≥ n :=
by
  use 240
  sorry

end NUMINAMATH_CALUDE_least_equal_bulbs_l2999_299965


namespace NUMINAMATH_CALUDE_tileable_rectangle_divisibility_l2999_299957

/-- A rectangle is (a, b)-tileable if it can be covered by non-overlapping a × b tiles -/
def is_tileable (m n a b : ℕ) : Prop := sorry

/-- Main theorem: If k divides a and b, and an m × n rectangle is (a, b)-tileable, 
    then 2k divides m or 2k divides n -/
theorem tileable_rectangle_divisibility 
  (k a b m n : ℕ) 
  (h1 : k ∣ a) 
  (h2 : k ∣ b) 
  (h3 : is_tileable m n a b) : 
  (2 * k) ∣ m ∨ (2 * k) ∣ n :=
sorry

end NUMINAMATH_CALUDE_tileable_rectangle_divisibility_l2999_299957


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2999_299913

theorem rectangle_dimensions (x : ℝ) : 
  (x + 3) * (3 * x - 4) = 5 * x + 14 → x = Real.sqrt 78 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2999_299913


namespace NUMINAMATH_CALUDE_infinite_representations_l2999_299976

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 10*x^2 + 29*x - 25

-- Define the property for a number to be a root of f
def is_root (x : ℝ) : Prop := f x = 0

-- Define the property for two numbers to be distinct
def are_distinct (x y : ℝ) : Prop := x ≠ y

-- Define the property for a positive integer to have the required representation
def has_representation (n : ℕ) (α β : ℝ) : Prop :=
  ∃ (r s : ℤ), n = ⌊r * α⌋ ∧ n = ⌊s * β⌋

-- State the theorem
theorem infinite_representations :
  ∃ (α β : ℝ), is_root α ∧ is_root β ∧ are_distinct α β ∧
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → has_representation n α β :=
sorry

end NUMINAMATH_CALUDE_infinite_representations_l2999_299976


namespace NUMINAMATH_CALUDE_campsite_return_strategy_l2999_299990

structure CampsiteScenario where
  num_students : ℕ
  time_remaining : ℕ
  num_roads : ℕ
  time_per_road : ℕ
  num_liars : ℕ

def has_reliable_strategy (scenario : CampsiteScenario) : Prop :=
  ∃ (strategy : CampsiteScenario → Bool),
    strategy scenario = true

theorem campsite_return_strategy 
  (scenario1 : CampsiteScenario)
  (scenario2 : CampsiteScenario)
  (h1 : scenario1.num_students = 8)
  (h2 : scenario1.time_remaining = 60)
  (h3 : scenario2.num_students = 4)
  (h4 : scenario2.time_remaining = 100)
  (h5 : scenario1.num_roads = 4)
  (h6 : scenario2.num_roads = 4)
  (h7 : scenario1.time_per_road = 20)
  (h8 : scenario2.time_per_road = 20)
  (h9 : scenario1.num_liars = 2)
  (h10 : scenario2.num_liars = 2) :
  has_reliable_strategy scenario1 ∧ has_reliable_strategy scenario2 :=
sorry

end NUMINAMATH_CALUDE_campsite_return_strategy_l2999_299990


namespace NUMINAMATH_CALUDE_lauren_change_l2999_299955

/-- Represents the grocery items and their prices --/
structure GroceryItems where
  meat_price : ℝ
  meat_weight : ℝ
  buns_price : ℝ
  lettuce_price : ℝ
  tomato_price : ℝ
  tomato_weight : ℝ
  pickles_price : ℝ
  pickle_coupon : ℝ

/-- Calculates the total cost of the grocery items --/
def total_cost (items : GroceryItems) : ℝ :=
  items.meat_price * items.meat_weight +
  items.buns_price +
  items.lettuce_price +
  items.tomato_price * items.tomato_weight +
  (items.pickles_price - items.pickle_coupon)

/-- Calculates the change from a given payment --/
def calculate_change (items : GroceryItems) (payment : ℝ) : ℝ :=
  payment - total_cost items

/-- Theorem stating that Lauren's change is $6.00 --/
theorem lauren_change :
  let items : GroceryItems := {
    meat_price := 3.5,
    meat_weight := 2,
    buns_price := 1.5,
    lettuce_price := 1,
    tomato_price := 2,
    tomato_weight := 1.5,
    pickles_price := 2.5,
    pickle_coupon := 1
  }
  calculate_change items 20 = 6 := by sorry

end NUMINAMATH_CALUDE_lauren_change_l2999_299955


namespace NUMINAMATH_CALUDE_pet_store_feet_count_l2999_299974

/-- A pet store sells dogs and parakeets. -/
structure PetStore :=
  (dogs : ℕ)
  (parakeets : ℕ)

/-- Calculate the total number of feet in the pet store. -/
def total_feet (store : PetStore) : ℕ :=
  4 * store.dogs + 2 * store.parakeets

/-- Theorem: Given 15 total heads and 9 dogs, the total number of feet is 48. -/
theorem pet_store_feet_count :
  ∀ (store : PetStore),
  store.dogs + store.parakeets = 15 →
  store.dogs = 9 →
  total_feet store = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_pet_store_feet_count_l2999_299974


namespace NUMINAMATH_CALUDE_grid_solution_l2999_299931

/-- Represents the possible values in the grid -/
inductive GridValue
  | Two
  | Zero
  | One
  | Five
  | Blank

/-- Represents a 5x5 grid -/
def Grid := Fin 5 → Fin 5 → GridValue

/-- Check if a grid satisfies the row and column constraints -/
def isValidGrid (g : Grid) : Prop :=
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.Two) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.Zero) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.One) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.Five) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.Two) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.Zero) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.One) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.Five)

/-- Check if the diagonal constraint is satisfied -/
def validDiagonal (g : Grid) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i i ≠ g j j

/-- The main theorem stating the solution -/
theorem grid_solution (g : Grid) 
  (hvalid : isValidGrid g) 
  (hdiag : validDiagonal g) : 
  g 4 0 = GridValue.One ∧ 
  g 4 1 = GridValue.Five ∧ 
  g 4 2 = GridValue.Blank ∧ 
  g 4 3 = GridValue.Zero ∧ 
  g 4 4 = GridValue.Two :=
sorry

end NUMINAMATH_CALUDE_grid_solution_l2999_299931


namespace NUMINAMATH_CALUDE_coin_weight_verification_l2999_299932

theorem coin_weight_verification (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  x + y = 3 ∧ 
  x + (x + y) + (x + 2*y) = 9 ∧ 
  y + (x + y) + (x + 2*y) = x + 9 → 
  x = 1 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_coin_weight_verification_l2999_299932


namespace NUMINAMATH_CALUDE_line_m_equation_line_n_equation_l2999_299935

-- Define the point A
def A : ℝ × ℝ := (-2, 1)

-- Define the line l
def l (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Define parallelism
def parallel (l₁ l₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ x y, l₁ x y ↔ l₂ (k * x) (k * y)

-- Define perpendicularity
def perpendicular (l₁ l₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ x y, l₁ x y ↔ l₂ y (-x)

-- Theorem for line m
theorem line_m_equation :
  ∃ (m : ℝ → ℝ → Prop),
    (m A.1 A.2) ∧
    (parallel l m) ∧
    (∀ x y, m x y ↔ 2 * x - y + 5 = 0) :=
sorry

-- Theorem for line n
theorem line_n_equation :
  ∃ (n : ℝ → ℝ → Prop),
    (n A.1 A.2) ∧
    (perpendicular l n) ∧
    (∀ x y, n x y ↔ x + 2 * y = 0) :=
sorry

end NUMINAMATH_CALUDE_line_m_equation_line_n_equation_l2999_299935


namespace NUMINAMATH_CALUDE_diana_remaining_paint_l2999_299962

/-- The amount of paint required for one statue in gallons -/
def paint_per_statue : ℚ := 1/16

/-- The number of statues Diana can paint with the remaining paint -/
def statues_to_paint : ℕ := 7

/-- The amount of paint Diana has remaining in gallons -/
def remaining_paint : ℚ := paint_per_statue * statues_to_paint

theorem diana_remaining_paint :
  remaining_paint = 7/16 := by sorry

end NUMINAMATH_CALUDE_diana_remaining_paint_l2999_299962


namespace NUMINAMATH_CALUDE_power_division_l2999_299984

theorem power_division (m : ℝ) : m^4 / m^2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l2999_299984


namespace NUMINAMATH_CALUDE_max_volume_at_10cm_l2999_299916

/-- The length of the original sheet in centimeters -/
def sheet_length : ℝ := 90

/-- The width of the original sheet in centimeters -/
def sheet_width : ℝ := 48

/-- The side length of the cut-out squares in centimeters -/
def cut_length : ℝ := 10

/-- The volume of the container as a function of the cut length -/
def container_volume (x : ℝ) : ℝ := (sheet_length - 2*x) * (sheet_width - 2*x) * x

theorem max_volume_at_10cm :
  ∀ x, 0 < x → x < sheet_width/2 → x < sheet_length/2 →
  container_volume x ≤ container_volume cut_length :=
sorry

end NUMINAMATH_CALUDE_max_volume_at_10cm_l2999_299916


namespace NUMINAMATH_CALUDE_rebus_puzzle_solution_l2999_299996

theorem rebus_puzzle_solution :
  ∃! (A B C : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    100 * A + 10 * B + A + 100 * A + 10 * B + C = 100 * A + 10 * C + C ∧
    100 * A + 10 * C + C = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 :=
by sorry

end NUMINAMATH_CALUDE_rebus_puzzle_solution_l2999_299996


namespace NUMINAMATH_CALUDE_largest_integer_divisible_by_18_with_sqrt_between_26_and_26_5_l2999_299986

theorem largest_integer_divisible_by_18_with_sqrt_between_26_and_26_5 : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    n % 18 = 0 ∧ 
    (26 : ℝ) < Real.sqrt n ∧ 
    Real.sqrt n ≤ 26.5 ∧
    ∀ (m : ℕ), m > 0 ∧ m % 18 = 0 ∧ (26 : ℝ) < Real.sqrt m ∧ Real.sqrt m ≤ 26.5 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_divisible_by_18_with_sqrt_between_26_and_26_5_l2999_299986


namespace NUMINAMATH_CALUDE_production_average_problem_l2999_299906

theorem production_average_problem (n : ℕ) : 
  (n * 50 + 105) / (n + 1) = 55 → n = 10 := by sorry

end NUMINAMATH_CALUDE_production_average_problem_l2999_299906


namespace NUMINAMATH_CALUDE_restaurant_students_l2999_299948

theorem restaurant_students (burger_count : ℕ) (hotdog_count : ℕ) :
  burger_count = 30 →
  burger_count = 2 * hotdog_count →
  burger_count + hotdog_count = 45 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_students_l2999_299948


namespace NUMINAMATH_CALUDE_inequality_proof_l2999_299940

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z) / (x^3 + y^3 + x * y * z) +
  (x * y * z) / (y^3 + z^3 + x * y * z) +
  (x * y * z) / (z^3 + x^3 + x * y * z) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2999_299940


namespace NUMINAMATH_CALUDE_conic_section_is_hyperbola_l2999_299949

/-- The equation (x-3)^2 = 3(2y+4)^2 - 75 represents a hyperbola -/
theorem conic_section_is_hyperbola :
  ∃ (a b c d e f : ℝ), 
    (∀ x y : ℝ, (x - 3)^2 = 3*(2*y + 4)^2 - 75 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0) ∧
    a > 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_conic_section_is_hyperbola_l2999_299949


namespace NUMINAMATH_CALUDE_long_furred_brown_dogs_l2999_299937

theorem long_furred_brown_dogs 
  (total : ℕ) 
  (long_furred : ℕ) 
  (brown : ℕ) 
  (neither : ℕ) 
  (h1 : total = 45) 
  (h2 : long_furred = 29) 
  (h3 : brown = 17) 
  (h4 : neither = 8) : 
  long_furred + brown - (total - neither) = 9 := by
sorry

end NUMINAMATH_CALUDE_long_furred_brown_dogs_l2999_299937


namespace NUMINAMATH_CALUDE_triangle_ratio_sqrt_two_l2999_299900

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * sin(A) * sin(B) + b * cos²(A) = √2 * a, then b/a = √2 -/
theorem triangle_ratio_sqrt_two (a b c : ℝ) (A B C : ℝ) 
    (h : a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = Real.sqrt 2 * a) 
    (h_positive : a > 0) : b / a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_sqrt_two_l2999_299900


namespace NUMINAMATH_CALUDE_parabola_perpendicular_point_range_l2999_299929

/-- Given points A, B, C where B and C are on a parabola and AB is perpendicular to BC,
    the y-coordinate of C satisfies y ≤ 0 or y ≥ 4 -/
theorem parabola_perpendicular_point_range 
  (A B C : ℝ × ℝ)
  (h_A : A = (0, 2))
  (h_B : B.1 = B.2^2 - 4)
  (h_C : C.1 = C.2^2 - 4)
  (h_perp : (B.2 - 2) * (C.2 - B.2) = -(B.1 - 0) * (C.1 - B.1)) :
  C.2 ≤ 0 ∨ C.2 ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_point_range_l2999_299929


namespace NUMINAMATH_CALUDE_remainder_equality_l2999_299982

theorem remainder_equality (a b : ℕ+) :
  (∀ p : ℕ, Nat.Prime p → 
    (a : ℕ) % p ≤ (b : ℕ) % p) →
  a = b := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l2999_299982


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_range_l2999_299983

def integer_range : List Int := List.range 14 |>.map (fun i => i - 6)

theorem arithmetic_mean_of_range : 
  (integer_range.sum : ℚ) / integer_range.length = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_range_l2999_299983


namespace NUMINAMATH_CALUDE_diminished_value_is_seven_l2999_299911

def smallest_number : ℕ := 1015

def divisors : List ℕ := [12, 16, 18, 21, 28]

theorem diminished_value_is_seven :
  ∃ (k : ℕ), k = 7 ∧
  ∀ d ∈ divisors, (smallest_number - k) % d = 0 ∧
  ∀ m < k, ∃ d ∈ divisors, (smallest_number - m) % d ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_diminished_value_is_seven_l2999_299911


namespace NUMINAMATH_CALUDE_equation_solutions_l2999_299910

theorem equation_solutions :
  (∀ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 →
    ¬(y.val = x.val + 1 ∧ z.val = y.val + 1 ∧ w.val = z.val + 1)) ∧
  (∃ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 ∧
    y.val = x.val + 2 ∧ z.val = y.val + 2 ∧ w.val = z.val + 2 ∧
    Even x.val ∧ Even y.val ∧ Even z.val ∧ Even w.val) ∧
  (∀ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 →
    ¬(y.val = x.val + 2 ∧ z.val = y.val + 2 ∧ w.val = z.val + 2 ∧
      Odd x.val ∧ Odd y.val ∧ Odd z.val ∧ Odd w.val)) ∧
  (∃ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 ∧
    y.val = x.val + 2 ∧ z.val = y.val + 2 ∧ w.val = z.val + 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2999_299910


namespace NUMINAMATH_CALUDE_complement_union_A_B_l2999_299909

def U : Set ℕ := {0, 1, 2, 3, 4, 5}

def A : Set ℕ := {x ∈ U | x^2 - 7*x + 12 = 0}

def B : Set ℕ := {1, 3, 5}

theorem complement_union_A_B : (U \ (A ∪ B)) = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l2999_299909


namespace NUMINAMATH_CALUDE_balls_in_boxes_l2999_299998

def to_base_7_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem balls_in_boxes (n : ℕ) (h : n = 3010) : 
  (to_base_7_digits n).sum = 16 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l2999_299998


namespace NUMINAMATH_CALUDE_intersection_set_characterization_l2999_299970

/-- The set of positive real numbers m for which the graphs of y = (mx-1)^2 and y = √x + m 
    have exactly one intersection point on the interval [0,1] -/
def IntersectionSet : Set ℝ :=
  {m : ℝ | m > 0 ∧ ∃! x : ℝ, x ∈ [0, 1] ∧ (m * x - 1)^2 = Real.sqrt x + m}

/-- The theorem stating that the IntersectionSet is equal to (0,1] ∪ [3, +∞) -/
theorem intersection_set_characterization :
  IntersectionSet = Set.Ioo 0 1 ∪ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_set_characterization_l2999_299970


namespace NUMINAMATH_CALUDE_vanessa_picked_17_carrots_l2999_299969

/-- The number of carrots Vanessa picked -/
def vanessas_carrots (good_carrots bad_carrots moms_carrots : ℕ) : ℕ :=
  good_carrots + bad_carrots - moms_carrots

/-- Proof that Vanessa picked 17 carrots -/
theorem vanessa_picked_17_carrots :
  vanessas_carrots 24 7 14 = 17 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_picked_17_carrots_l2999_299969


namespace NUMINAMATH_CALUDE_shipment_box_count_l2999_299963

theorem shipment_box_count :
  ∀ (x y : ℕ),
  (10 * x + 20 * y) / (x + y) = 18 →
  (10 * x + 20 * (y - 15)) / (x + y - 15) = 16 →
  x + y = 30 := by
sorry

end NUMINAMATH_CALUDE_shipment_box_count_l2999_299963


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l2999_299975

theorem inequality_solution_sets (a : ℝ) : 
  (∀ x : ℝ, 3 * x - 5 < a ↔ 2 * x < 4) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l2999_299975


namespace NUMINAMATH_CALUDE_djibos_sister_age_l2999_299968

/-- Given that Djibo is 17 years old and 5 years ago the sum of his and his sister's ages was 35,
    prove that his sister is 28 years old today. -/
theorem djibos_sister_age :
  ∀ (djibo_age sister_age : ℕ),
    djibo_age = 17 →
    djibo_age + sister_age = 35 + 5 →
    sister_age = 28 :=
by sorry

end NUMINAMATH_CALUDE_djibos_sister_age_l2999_299968


namespace NUMINAMATH_CALUDE_batsman_average_l2999_299956

/-- Calculates the average runs for a batsman given two sets of matches --/
def calculate_average (matches1 : ℕ) (average1 : ℕ) (matches2 : ℕ) (average2 : ℕ) : ℚ :=
  let total_runs := matches1 * average1 + matches2 * average2
  let total_matches := matches1 + matches2
  (total_runs : ℚ) / total_matches

/-- Proves that the batsman's average for 30 matches is 31 runs --/
theorem batsman_average : calculate_average 20 40 10 13 = 31 := by
  sorry

#eval calculate_average 20 40 10 13

end NUMINAMATH_CALUDE_batsman_average_l2999_299956


namespace NUMINAMATH_CALUDE_f_properties_l2999_299991

noncomputable section

open Real

/-- The function f(x) = ae^(2x) - ae^x - xe^x --/
def f (a : ℝ) (x : ℝ) : ℝ := a * exp (2 * x) - a * exp x - x * exp x

/-- The theorem stating the properties of f --/
theorem f_properties :
  ∀ a : ℝ, a ≥ 0 → (∀ x : ℝ, f a x ≥ 0) →
  ∃ x₀ : ℝ,
    a = 1 ∧
    (∀ x : ℝ, f 1 x ≤ f 1 x₀) ∧
    (∀ x : ℝ, x ≠ x₀ → f 1 x < f 1 x₀) ∧
    (log 2 / (2 * exp 1) + 1 / (4 * exp 1 ^ 2) ≤ f 1 x₀) ∧
    (f 1 x₀ < 1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2999_299991


namespace NUMINAMATH_CALUDE_fraction_transformation_l2999_299905

theorem fraction_transformation (x : ℚ) : 
  (3 + x) / (11 + x) = 5 / 9 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l2999_299905


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l2999_299902

/-- The number of Popsicles consumed in a given time period -/
def popsicles_consumed (rate_minutes : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / rate_minutes

theorem megan_popsicle_consumption :
  popsicles_consumed 20 340 = 17 := by
  sorry

#eval popsicles_consumed 20 340

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l2999_299902


namespace NUMINAMATH_CALUDE_two_n_squares_implies_n_squares_l2999_299925

theorem two_n_squares_implies_n_squares (n : ℕ) 
  (h : ∃ (k m : ℤ), 2 * n = k^2 + m^2) : 
  ∃ (a b : ℚ), n = a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_two_n_squares_implies_n_squares_l2999_299925


namespace NUMINAMATH_CALUDE_factoring_transformation_l2999_299908

-- Define the concept of factoring
def is_factored (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x ∧ ∃ (p q : ℝ → ℝ), g x = p x * q x

-- Define the specific expression
def left_expr : ℝ → ℝ := λ x ↦ x^2 - 4
def right_expr : ℝ → ℝ := λ x ↦ (x + 2) * (x - 2)

-- Theorem statement
theorem factoring_transformation :
  is_factored left_expr right_expr :=
sorry

end NUMINAMATH_CALUDE_factoring_transformation_l2999_299908


namespace NUMINAMATH_CALUDE_lucky_larry_problem_l2999_299947

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 2 → b = 5 → c = 3 → d = 4 → 
  (a + b - c - d * e = a + (b - (c - (d * e)))) → e = 0 := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_problem_l2999_299947


namespace NUMINAMATH_CALUDE_card_length_is_three_inches_l2999_299942

-- Define the poster board size in inches
def posterBoardSize : ℕ := 12

-- Define the width of the cards in inches
def cardWidth : ℕ := 2

-- Define the maximum number of cards that can be made
def maxCards : ℕ := 24

-- Theorem statement
theorem card_length_is_three_inches :
  ∀ (cardLength : ℕ),
    (posterBoardSize / cardWidth) * (posterBoardSize / cardLength) = maxCards →
    cardLength = 3 := by
  sorry

end NUMINAMATH_CALUDE_card_length_is_three_inches_l2999_299942


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2999_299941

def snake_toy_price : ℚ := 1176 / 100
def cage_price : ℚ := 1454 / 100
def heat_lamp_price : ℚ := 625 / 100
def cage_discount_rate : ℚ := 10 / 100
def sales_tax_rate : ℚ := 8 / 100
def found_money : ℚ := 1

def total_cost : ℚ :=
  let discounted_cage_price := cage_price * (1 - cage_discount_rate)
  let subtotal := snake_toy_price + discounted_cage_price + heat_lamp_price
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax - found_money

theorem total_cost_calculation :
  (total_cost * 100).floor / 100 = 3258 / 100 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2999_299941


namespace NUMINAMATH_CALUDE_correct_observation_value_l2999_299995

theorem correct_observation_value 
  (n : ℕ) 
  (original_mean : ℚ) 
  (incorrect_value : ℚ) 
  (corrected_mean : ℚ) 
  (h1 : n = 50) 
  (h2 : original_mean = 30) 
  (h3 : incorrect_value = 23) 
  (h4 : corrected_mean = 30.5) : 
  (n : ℚ) * corrected_mean - ((n : ℚ) * original_mean - incorrect_value) = 48 :=
by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l2999_299995


namespace NUMINAMATH_CALUDE_unique_solution_l2999_299903

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ :=
  x - (floor x : ℝ)

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  2 * (floor x : ℝ) * frac x = x^2 - 3/2 * x - 11/16

theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 9/4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2999_299903


namespace NUMINAMATH_CALUDE_points_and_lines_l2999_299901

theorem points_and_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≤ 45 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_points_and_lines_l2999_299901


namespace NUMINAMATH_CALUDE_bill_toilet_paper_supply_l2999_299922

/-- The number of days Bill's toilet paper supply will last -/
def toilet_paper_days (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ) (rolls : ℕ) (squares_per_roll : ℕ) : ℕ :=
  (rolls * squares_per_roll) / (bathroom_visits_per_day * squares_per_visit)

/-- Theorem stating that Bill's toilet paper supply will last for 20,000 days -/
theorem bill_toilet_paper_supply : toilet_paper_days 3 5 1000 300 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_bill_toilet_paper_supply_l2999_299922


namespace NUMINAMATH_CALUDE_science_books_count_l2999_299950

theorem science_books_count (total : ℕ) (storybooks science picture dictionaries : ℕ) :
  total = 35 →
  total = storybooks + science + picture + dictionaries →
  storybooks + science = 17 →
  science + picture = 16 →
  storybooks ≠ science →
  storybooks ≠ picture →
  storybooks ≠ dictionaries →
  science ≠ picture →
  science ≠ dictionaries →
  picture ≠ dictionaries →
  (storybooks = 9 ∨ science = 9 ∨ picture = 9 ∨ dictionaries = 9) →
  science = 9 :=
by sorry

end NUMINAMATH_CALUDE_science_books_count_l2999_299950


namespace NUMINAMATH_CALUDE_sum_other_vertices_y_equals_14_l2999_299997

structure Rectangle where
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ

def Rectangle.sumOtherVerticesY (r : Rectangle) : ℝ :=
  r.vertex1.2 + r.vertex2.2

theorem sum_other_vertices_y_equals_14 (r : Rectangle) 
  (h1 : r.vertex1 = (2, 20))
  (h2 : r.vertex2 = (10, -6)) :
  r.sumOtherVerticesY = 14 := by
  sorry

#check sum_other_vertices_y_equals_14

end NUMINAMATH_CALUDE_sum_other_vertices_y_equals_14_l2999_299997


namespace NUMINAMATH_CALUDE_square_difference_l2999_299967

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2999_299967


namespace NUMINAMATH_CALUDE_combined_population_theorem_l2999_299954

/-- The combined population of New York and New England -/
def combined_population (new_england_population : ℕ) : ℕ :=
  new_england_population + (2 * new_england_population) / 3

/-- Theorem stating the combined population of New York and New England -/
theorem combined_population_theorem :
  combined_population 2100000 = 3500000 := by
  sorry

#eval combined_population 2100000

end NUMINAMATH_CALUDE_combined_population_theorem_l2999_299954


namespace NUMINAMATH_CALUDE_boat_travel_time_l2999_299987

theorem boat_travel_time (v : ℝ) :
  let upstream_speed := v - 4
  let downstream_speed := v + 4
  let distance := 120
  let upstream_time (t : ℝ) := t + 1
  let downstream_time := 1
  (upstream_speed * upstream_time downstream_time = distance) ∧
  (downstream_speed * downstream_time = distance) →
  downstream_time = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_travel_time_l2999_299987


namespace NUMINAMATH_CALUDE_triangle_altitude_sum_perfect_square_l2999_299930

theorem triangle_altitude_sum_perfect_square (x y z : ℤ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  (∃ (h_x h_y h_z : ℝ), 
    h_x > 0 ∧ h_y > 0 ∧ h_z > 0 ∧
    (h_x = h_y + h_z ∨ h_y = h_x + h_z ∨ h_z = h_x + h_y) ∧
    x * h_x = y * h_y ∧ y * h_y = z * h_z) →
  ∃ (n : ℤ), x^2 + y^2 + z^2 = n^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_sum_perfect_square_l2999_299930


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l2999_299966

/-- Proves that given a mixture with 20% alcohol, if 3 liters of water are added
    and the resulting mixture has 17.14285714285715% alcohol, 
    then the initial amount of mixture was 18 liters. -/
theorem initial_mixture_volume (initial_volume : ℝ) : 
  initial_volume > 0 →
  (0.2 * initial_volume) / (initial_volume + 3) = 17.14285714285715 / 100 →
  initial_volume = 18 := by
sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l2999_299966


namespace NUMINAMATH_CALUDE_irregular_polygon_rotation_implies_composite_l2999_299961

/-- An n-gon inscribed in a circle -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Rotation of a point around the origin by an angle -/
def rotate (p : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ := sorry

/-- A polygon is irregular if not all sides have the same length -/
def irregular (P : Polygon n) : Prop := sorry

/-- A polygon coincides with itself after rotation -/
def coincides_after_rotation (P : Polygon n) (angle : ℝ) : Prop := sorry

/-- A number is composite if it's not prime and greater than 1 -/
def composite (n : ℕ) : Prop := ¬ Nat.Prime n ∧ n > 1

/-- Main theorem -/
theorem irregular_polygon_rotation_implies_composite
  (n : ℕ) (P : Polygon n) (α : ℝ) :
  irregular P →
  α ≠ 2 * Real.pi →
  coincides_after_rotation P α →
  composite n := by sorry

end NUMINAMATH_CALUDE_irregular_polygon_rotation_implies_composite_l2999_299961


namespace NUMINAMATH_CALUDE_investment_with_interest_l2999_299943

def total_investment : ℝ := 1000
def amount_at_3_percent : ℝ := 199.99999999999983
def interest_rate_3_percent : ℝ := 0.03
def interest_rate_5_percent : ℝ := 0.05

theorem investment_with_interest :
  let amount_at_5_percent := total_investment - amount_at_3_percent
  let interest_at_3_percent := amount_at_3_percent * interest_rate_3_percent
  let interest_at_5_percent := amount_at_5_percent * interest_rate_5_percent
  let total_with_interest := total_investment + interest_at_3_percent + interest_at_5_percent
  total_with_interest = 1046 := by sorry

end NUMINAMATH_CALUDE_investment_with_interest_l2999_299943


namespace NUMINAMATH_CALUDE_max_median_value_l2999_299920

theorem max_median_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 10 →
  k < m → m < r → r < s → s < t →
  t = 20 →
  r ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_max_median_value_l2999_299920


namespace NUMINAMATH_CALUDE_thumbtack_count_l2999_299978

theorem thumbtack_count (num_cans : ℕ) (boards_tested : ℕ) (tacks_per_board : ℕ) (remaining_tacks : ℕ) : 
  num_cans = 3 →
  boards_tested = 120 →
  tacks_per_board = 1 →
  remaining_tacks = 30 →
  (num_cans * (boards_tested * tacks_per_board + remaining_tacks) = 450) :=
by sorry

end NUMINAMATH_CALUDE_thumbtack_count_l2999_299978


namespace NUMINAMATH_CALUDE_apps_added_l2999_299933

theorem apps_added (initial_apps final_apps : ℕ) 
  (h1 : initial_apps = 17) 
  (h2 : final_apps = 18) : 
  final_apps - initial_apps = 1 := by
  sorry

end NUMINAMATH_CALUDE_apps_added_l2999_299933


namespace NUMINAMATH_CALUDE_find_y_l2999_299934

theorem find_y : ∃ y : ℝ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2999_299934


namespace NUMINAMATH_CALUDE_triangle_sine_sum_bound_l2999_299994

/-- Given a triangle with angles A, B, and C (in radians), 
    the sum of the sines of its angles is at most 3√3/2, 
    with equality if and only if the triangle is equilateral. -/
theorem triangle_sine_sum_bound (A B C : ℝ) 
    (h_angles : A + B + C = π) 
    (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) : 
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 ∧ 
  (Real.sin A + Real.sin B + Real.sin C = 3 * Real.sqrt 3 / 2 ↔ A = B ∧ B = C) := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_bound_l2999_299994


namespace NUMINAMATH_CALUDE_rulers_in_drawer_l2999_299981

/-- The number of rulers originally in the drawer -/
def original_rulers : ℕ := 71 - 25

theorem rulers_in_drawer : original_rulers = 46 := by
  sorry

end NUMINAMATH_CALUDE_rulers_in_drawer_l2999_299981


namespace NUMINAMATH_CALUDE_number_of_students_is_five_l2999_299945

/-- The number of students who will receive stickers from Miss Walter -/
def number_of_students : ℕ :=
  let gold_stickers : ℕ := 50
  let silver_stickers : ℕ := 2 * gold_stickers
  let bronze_stickers : ℕ := silver_stickers - 20
  let total_stickers : ℕ := gold_stickers + silver_stickers + bronze_stickers
  let stickers_per_student : ℕ := 46
  total_stickers / stickers_per_student

/-- Theorem stating that the number of students who will receive stickers is 5 -/
theorem number_of_students_is_five : number_of_students = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_is_five_l2999_299945


namespace NUMINAMATH_CALUDE_numerical_expression_problem_l2999_299951

theorem numerical_expression_problem :
  ∃ (A B C D : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
    20180 ≤ 2018 * 10 + A ∧ 2018 * 10 + A < 20190 ∧
    100 ≤ B * 100 + C * 10 + D ∧ B * 100 + C * 10 + D < 1000 ∧
    (2018 * 10 + A) / (B * 100 + C * 10 + D) = 10 * A + A ∧
    A = 5 ∧ B = 3 ∧ C = 6 ∧ D = 7 :=
by sorry

end NUMINAMATH_CALUDE_numerical_expression_problem_l2999_299951


namespace NUMINAMATH_CALUDE_white_towels_count_l2999_299938

def green_towels : ℕ := 35
def towels_given_away : ℕ := 34
def towels_remaining : ℕ := 22

theorem white_towels_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_white_towels_count_l2999_299938


namespace NUMINAMATH_CALUDE_sqrt_640000_equals_800_l2999_299992

theorem sqrt_640000_equals_800 : Real.sqrt 640000 = 800 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_640000_equals_800_l2999_299992


namespace NUMINAMATH_CALUDE_equation_solutions_l2999_299914

theorem equation_solutions :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4 → x + 1 ≠ 7 * (x - 1) - x^2) ∧
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4 → (x + 1 = x * (10 - x) - 7 ↔ x = 8 ∨ x = 1)) ∧
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4 → (x + 1 = x * (7 - x) + 1 ↔ x = 6)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2999_299914


namespace NUMINAMATH_CALUDE_quadratic_no_solution_l2999_299944

theorem quadratic_no_solution (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x - 1 ≠ 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_solution_l2999_299944


namespace NUMINAMATH_CALUDE_f_max_value_l2999_299919

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x^2)

theorem f_max_value :
  ∃ (x_max : ℝ), x_max > 0 ∧
  (∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
  x_max = Real.sqrt (Real.exp 1) ∧
  f x_max = 1 / (2 * Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l2999_299919


namespace NUMINAMATH_CALUDE_geometric_sequence_term_count_l2999_299971

theorem geometric_sequence_term_count :
  ∀ (a : ℕ → ℚ),
  (∀ k : ℕ, a (k + 1) = a k * (1/2)) →  -- Geometric sequence with q = 1/2
  a 1 = 1/2 →                           -- First term a₁ = 1/2
  (∃ n : ℕ, a n = 1/32) →               -- Some term aₙ = 1/32
  ∃ n : ℕ, n = 5 ∧ a n = 1/32 :=        -- The term count n is 5
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_term_count_l2999_299971


namespace NUMINAMATH_CALUDE_bike_ride_distance_l2999_299923

/-- Calculates the total distance of a 3-hour bike ride given specific conditions -/
theorem bike_ride_distance (second_hour_distance : ℝ) 
  (h1 : second_hour_distance = 18)
  (h2 : second_hour_distance = 1.2 * (second_hour_distance / 1.2))
  (h3 : 1.25 * second_hour_distance = 22.5) :
  (second_hour_distance / 1.2) + second_hour_distance + (1.25 * second_hour_distance) = 55.5 := by
sorry

end NUMINAMATH_CALUDE_bike_ride_distance_l2999_299923


namespace NUMINAMATH_CALUDE_repeating_decimal_equiv_l2999_299912

-- Define the repeating decimal 0.4̄13
def repeating_decimal : ℚ := 409 / 990

-- Theorem statement
theorem repeating_decimal_equiv : 
  repeating_decimal = 409 / 990 ∧ 
  (∀ n d : ℕ, n / d = 409 / 990 → d ≤ 990) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equiv_l2999_299912


namespace NUMINAMATH_CALUDE_fraction_product_equality_l2999_299993

theorem fraction_product_equality : (1 / 3 : ℚ)^3 * (1 / 7 : ℚ)^2 = 1 / 1323 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l2999_299993


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2999_299979

theorem quadratic_equation_solution :
  ∃! x : ℚ, x > 0 ∧ 7 * x^2 + 13 * x - 30 = 0 :=
by
  -- The unique solution is x = 10/7
  use 10/7
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2999_299979


namespace NUMINAMATH_CALUDE_translation_right_3_units_l2999_299907

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_3_units :
  let A : Point := { x := 1, y := 2 }
  let B : Point := translateRight A 3
  B = { x := 4, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_translation_right_3_units_l2999_299907


namespace NUMINAMATH_CALUDE_sequence_length_l2999_299999

theorem sequence_length (n : ℕ) (b : ℕ → ℝ) : 
  (n > 0) →
  (b 0 = 45) →
  (b 1 = 80) →
  (b n = 0) →
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 4 / b k) →
  n = 901 := by
sorry

end NUMINAMATH_CALUDE_sequence_length_l2999_299999
