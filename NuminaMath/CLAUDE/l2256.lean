import Mathlib

namespace expression_evaluation_l2256_225683

theorem expression_evaluation (x y : ℚ) (hx : x = -2) (hy : y = 1) :
  (-2 * x + x + 3 * y) - 2 * (-x^2 - 2 * x + 1/2 * y) = 4 := by
  sorry

end expression_evaluation_l2256_225683


namespace big_stack_orders_l2256_225628

/-- The number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- The number of pancakes in a short stack -/
def short_stack : ℕ := 3

/-- The number of customers who ordered the short stack -/
def short_stack_orders : ℕ := 9

/-- The total number of pancakes needed -/
def total_pancakes : ℕ := 57

/-- Theorem stating that the number of customers who ordered the big stack is 6 -/
theorem big_stack_orders : ℕ := by
  sorry

end big_stack_orders_l2256_225628


namespace preceding_number_in_base_3_l2256_225687

def base_3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3^i)) 0

def decimal_to_base_3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
  aux n []

theorem preceding_number_in_base_3 (N : Nat) (h : base_3_to_decimal [2, 1, 0, 1] = N) :
  decimal_to_base_3 (N - 1) = [2, 1, 0, 0] :=
sorry

end preceding_number_in_base_3_l2256_225687


namespace log_sum_theorem_l2256_225608

theorem log_sum_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 4) 
  (h2 : x * y = 64) : 
  (x + y) / 2 = (64^(3/(5+Real.sqrt 3)) + 64^(1/(5+Real.sqrt 3))) / 2 := by
sorry

end log_sum_theorem_l2256_225608


namespace instantaneous_velocity_at_4_seconds_l2256_225654

-- Define the position function
def s (t : ℝ) : ℝ := t^2 - 2*t + 5

-- Define the velocity function as the derivative of the position function
def v (t : ℝ) : ℝ := 2*t - 2

-- Theorem statement
theorem instantaneous_velocity_at_4_seconds :
  v 4 = 6 :=
sorry

end instantaneous_velocity_at_4_seconds_l2256_225654


namespace bicycle_shop_optimal_plan_l2256_225618

/-- Represents the purchase plan for bicycles -/
structure BicyclePlan where
  modelA : ℕ
  modelB : ℕ

/-- The bicycle shop problem -/
theorem bicycle_shop_optimal_plan :
  ∀ (plan : BicyclePlan),
  plan.modelA + plan.modelB = 50 →
  plan.modelB ≥ plan.modelA →
  1000 * plan.modelA + 1600 * plan.modelB ≤ 68000 →
  ∃ (optimalPlan : BicyclePlan),
  optimalPlan.modelA = 25 ∧
  optimalPlan.modelB = 25 ∧
  ∀ (p : BicyclePlan),
  p.modelA + p.modelB = 50 →
  p.modelB ≥ p.modelA →
  1000 * p.modelA + 1600 * p.modelB ≤ 68000 →
  500 * p.modelA + 400 * p.modelB ≤ 500 * optimalPlan.modelA + 400 * optimalPlan.modelB ∧
  500 * optimalPlan.modelA + 400 * optimalPlan.modelB = 22500 :=
by
  sorry


end bicycle_shop_optimal_plan_l2256_225618


namespace buffy_whiskers_l2256_225685

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  juniper : ℕ

/-- Theorem stating the number of whiskers Buffy has -/
theorem buffy_whiskers (c : CatWhiskers) : 
  c.juniper = 12 →
  c.puffy = 3 * c.juniper →
  c.scruffy = 2 * c.puffy →
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3 →
  c.buffy = 40 := by
  sorry

#check buffy_whiskers

end buffy_whiskers_l2256_225685


namespace buckets_required_l2256_225657

/-- The number of buckets required to fill a tank with the original bucket size,
    given that 62.5 buckets are needed when the bucket capacity is reduced to two-fifths. -/
theorem buckets_required (original_buckets : ℝ) : 
  (62.5 * (2/5) * original_buckets = original_buckets) → original_buckets = 25 := by
  sorry

end buckets_required_l2256_225657


namespace sports_school_distribution_l2256_225648

theorem sports_school_distribution (total : ℕ) (skiing : ℕ) (speed_skating : ℕ) (hockey : ℕ) :
  total = 96 →
  speed_skating = (skiing * 4) / 5 →
  hockey = (skiing + speed_skating) / 3 →
  skiing + speed_skating + hockey = total →
  (skiing = 40 ∧ speed_skating = 32 ∧ hockey = 24) := by
  sorry

end sports_school_distribution_l2256_225648


namespace connect_four_shapes_l2256_225626

/-- The number of columns in the Connect Four board -/
def num_columns : ℕ := 7

/-- The number of rows in the Connect Four board -/
def num_rows : ℕ := 8

/-- The number of possible states for each column (0 to 8 checkers) -/
def states_per_column : ℕ := num_rows + 1

/-- The total number of shapes before accounting for symmetry -/
def total_shapes : ℕ := states_per_column ^ num_columns

/-- The number of symmetric shapes -/
def symmetric_shapes : ℕ := states_per_column ^ (num_columns / 2 + 1)

/-- The formula for the number of distinct shapes -/
def distinct_shapes (n : ℕ) : ℕ := 9 * (n * (n + 1) / 2)

/-- The theorem stating that the number of distinct shapes is equal to 9(1+2+...+729) -/
theorem connect_four_shapes :
  ∃ n : ℕ, distinct_shapes n = symmetric_shapes + (total_shapes - symmetric_shapes) / 2 ∧ n = 729 := by
  sorry

end connect_four_shapes_l2256_225626


namespace books_read_ratio_l2256_225619

theorem books_read_ratio : 
  let william_last_month : ℕ := 6
  let brad_this_month : ℕ := 8
  let william_this_month : ℕ := 2 * brad_this_month
  let william_total : ℕ := william_last_month + william_this_month
  let brad_total : ℕ := william_total - 4
  let brad_last_month : ℕ := brad_total - brad_this_month
  ∃ (a b : ℕ), a * william_last_month = b * brad_last_month ∧ a = 3 ∧ b = 5 :=
by
  sorry


end books_read_ratio_l2256_225619


namespace jogging_time_difference_fathers_jogging_time_saved_l2256_225612

/-- Calculates the time difference in minutes between jogging at varying speeds and a constant speed -/
theorem jogging_time_difference (distance : ℝ) (constant_speed : ℝ) 
  (speeds : List ℝ) : ℝ :=
  let varying_time := (speeds.map (λ s => distance / s)).sum
  let constant_time := speeds.length * (distance / constant_speed)
  (varying_time - constant_time) * 60

/-- Proves that the time difference for the given scenario is 3 minutes -/
theorem fathers_jogging_time_saved : 
  jogging_time_difference 3 5 [6, 5, 4, 5] = 3 := by
  sorry

end jogging_time_difference_fathers_jogging_time_saved_l2256_225612


namespace whiskers_cat_school_total_l2256_225605

/-- Represents the number of cats that can perform a specific trick or combination of tricks -/
structure CatTricks where
  jump : ℕ
  sit : ℕ
  playDead : ℕ
  fetch : ℕ
  jumpSit : ℕ
  sitPlayDead : ℕ
  playDeadFetch : ℕ
  fetchJump : ℕ
  jumpSitPlayDead : ℕ
  sitPlayDeadFetch : ℕ
  playDeadFetchJump : ℕ
  jumpFetchSit : ℕ
  allFour : ℕ
  none : ℕ

/-- Calculates the total number of cats in the Whisker's Cat School -/
def totalCats (tricks : CatTricks) : ℕ :=
  let exclusiveJump := tricks.jump - (tricks.jumpSit + tricks.fetchJump + tricks.jumpSitPlayDead + tricks.allFour)
  let exclusiveSit := tricks.sit - (tricks.jumpSit + tricks.sitPlayDead + tricks.jumpSitPlayDead + tricks.allFour)
  let exclusivePlayDead := tricks.playDead - (tricks.sitPlayDead + tricks.playDeadFetch + tricks.jumpSitPlayDead + tricks.allFour)
  let exclusiveFetch := tricks.fetch - (tricks.playDeadFetch + tricks.fetchJump + tricks.sitPlayDeadFetch + tricks.allFour)
  exclusiveJump + exclusiveSit + exclusivePlayDead + exclusiveFetch +
  tricks.jumpSit + tricks.sitPlayDead + tricks.playDeadFetch + tricks.fetchJump +
  tricks.jumpSitPlayDead + tricks.sitPlayDeadFetch + tricks.playDeadFetchJump + tricks.jumpFetchSit +
  tricks.allFour + tricks.none

/-- The specific number of cats for each trick or combination at the Whisker's Cat School -/
def whiskersCatSchool : CatTricks :=
  { jump := 60
  , sit := 40
  , playDead := 35
  , fetch := 45
  , jumpSit := 20
  , sitPlayDead := 15
  , playDeadFetch := 10
  , fetchJump := 18
  , jumpSitPlayDead := 5
  , sitPlayDeadFetch := 3
  , playDeadFetchJump := 7
  , jumpFetchSit := 10
  , allFour := 2
  , none := 12 }

/-- Theorem stating that the total number of cats at the Whisker's Cat School is 143 -/
theorem whiskers_cat_school_total : totalCats whiskersCatSchool = 143 := by
  sorry

end whiskers_cat_school_total_l2256_225605


namespace lcm_gcd_product_equality_l2256_225627

theorem lcm_gcd_product_equality (a b : ℕ) (h : Nat.lcm a b + Nat.gcd a b = a * b) : a = 2 ∧ b = 2 := by
  sorry

end lcm_gcd_product_equality_l2256_225627


namespace frame_uncovered_area_l2256_225689

/-- The area of a rectangular frame not covered by a photo -/
theorem frame_uncovered_area (frame_length frame_width photo_length photo_width : ℝ)
  (h1 : frame_length = 40)
  (h2 : frame_width = 32)
  (h3 : photo_length = 32)
  (h4 : photo_width = 28) :
  frame_length * frame_width - photo_length * photo_width = 384 := by
  sorry

end frame_uncovered_area_l2256_225689


namespace server_data_processing_l2256_225651

/-- Represents the data processing rate in megabytes per minute -/
def processing_rate : ℝ := 150

/-- Represents the time period in hours -/
def time_period : ℝ := 12

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℝ := 60

/-- Represents the number of megabytes in a gigabyte -/
def mb_per_gb : ℝ := 1000

/-- Theorem stating that the server processes 108 gigabytes in 12 hours -/
theorem server_data_processing :
  (processing_rate * time_period * minutes_per_hour) / mb_per_gb = 108 := by
  sorry

end server_data_processing_l2256_225651


namespace modulus_of_complex_fraction_l2256_225677

theorem modulus_of_complex_fraction : 
  let z : ℂ := (2 - I) / (1 + 2*I)
  Complex.abs z = 1 := by
sorry

end modulus_of_complex_fraction_l2256_225677


namespace compound_interest_calculation_l2256_225660

theorem compound_interest_calculation (principal : ℝ) (rate : ℝ) (time : ℕ) (final_amount : ℝ) : 
  principal = 8000 →
  rate = 0.05 →
  time = 2 →
  final_amount = 8820 →
  final_amount = principal * (1 + rate) ^ time :=
sorry

end compound_interest_calculation_l2256_225660


namespace chocolate_distribution_problem_l2256_225638

/-- The number of ways to distribute n chocolates among k people, 
    with each person receiving at least m chocolates -/
def distribute_chocolates (n k m : ℕ) : ℕ := 
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The problem statement -/
theorem chocolate_distribution_problem : 
  distribute_chocolates 30 3 3 = 253 := by sorry

end chocolate_distribution_problem_l2256_225638


namespace discount_calculation_l2256_225629

/-- The discount received when buying multiple parts with a given original price,
    number of parts, and final price paid. -/
def discount (original_price : ℕ) (num_parts : ℕ) (final_price : ℕ) : ℕ :=
  original_price * num_parts - final_price

/-- Theorem stating that the discount is $121 given the problem conditions. -/
theorem discount_calculation :
  let original_price : ℕ := 80
  let num_parts : ℕ := 7
  let final_price : ℕ := 439
  discount original_price num_parts final_price = 121 := by
  sorry

end discount_calculation_l2256_225629


namespace subtraction_of_fractions_l2256_225653

theorem subtraction_of_fractions : 
  (16 : ℚ) / 24 - (1 + 2 / 9) = -5 / 9 := by
  sorry

end subtraction_of_fractions_l2256_225653


namespace y_change_when_x_increases_l2256_225606

/-- Regression equation: y = 3 - 5x -/
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

/-- Theorem: When x increases by 1, y decreases by 5 -/
theorem y_change_when_x_increases (x : ℝ) :
  regression_equation (x + 1) = regression_equation x - 5 := by
  sorry

end y_change_when_x_increases_l2256_225606


namespace triangle_inequality_l2256_225658

theorem triangle_inequality (a b c p q r : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- triangle side lengths are positive
  a + b > c ∧ b + c > a ∧ c + a > b ∧  -- triangle inequality
  p + q + r = 0 →  -- sum of p, q, r is zero
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := by
sorry

end triangle_inequality_l2256_225658


namespace proposition_relations_l2256_225602

-- Define the original proposition
def p (a : ℝ) : Prop := a > 0 → a^2 ≠ 0

-- Define the converse
def converse (a : ℝ) : Prop := a^2 ≠ 0 → a > 0

-- Define the inverse
def inverse (a : ℝ) : Prop := ¬(a > 0) → a^2 = 0

-- Define the contrapositive
def contrapositive (a : ℝ) : Prop := a^2 = 0 → ¬(a > 0)

-- Define the negation
def negation : Prop := ∃ a : ℝ, a > 0 ∧ a^2 = 0

-- Theorem stating the truth values of each related proposition
theorem proposition_relations :
  (∃ a : ℝ, ¬(converse a)) ∧
  (∃ a : ℝ, ¬(inverse a)) ∧
  (∀ a : ℝ, contrapositive a) ∧
  ¬negation :=
sorry

end proposition_relations_l2256_225602


namespace sin_40_tan_10_minus_sqrt_3_l2256_225622

theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end sin_40_tan_10_minus_sqrt_3_l2256_225622


namespace georgia_carnation_cost_l2256_225613

/-- The cost of a single carnation in dollars -/
def single_carnation_cost : ℚ := 0.5

/-- The cost of a dozen carnations in dollars -/
def dozen_carnation_cost : ℚ := 4

/-- The number of teachers Georgia wants to send carnations to -/
def number_of_teachers : ℕ := 5

/-- The number of friends Georgia wants to buy carnations for -/
def number_of_friends : ℕ := 14

/-- The total cost of Georgia's carnation purchases -/
def total_cost : ℚ := 
  (number_of_teachers * dozen_carnation_cost) + 
  dozen_carnation_cost + 
  (2 * single_carnation_cost)

/-- Theorem stating that the total cost of Georgia's carnation purchases is $25.00 -/
theorem georgia_carnation_cost : total_cost = 25 := by
  sorry

end georgia_carnation_cost_l2256_225613


namespace weight_of_b_l2256_225690

/-- Given three weights a, b, and c, prove that b = 70 under the given conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 60 →
  (a + b) / 2 = 70 →
  (b + c) / 2 = 50 →
  b = 70 := by
sorry

end weight_of_b_l2256_225690


namespace isosceles_triangle_base_angle_l2256_225637

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b) ∨ (a = c) ∨ (b = c)
  sumAngles : a + b + c = 180

-- Define the condition of angle ratio
def hasAngleRatio (t : IsoscelesTriangle) : Prop :=
  (t.a = 2 * t.b) ∨ (t.a = 2 * t.c) ∨ (t.b = 2 * t.c) ∨
  (2 * t.a = t.b) ∨ (2 * t.a = t.c) ∨ (2 * t.b = t.c)

-- Theorem statement
theorem isosceles_triangle_base_angle 
  (t : IsoscelesTriangle) 
  (h : hasAngleRatio t) : 
  (t.a = 45 ∧ (t.b = 45 ∨ t.c = 45)) ∨
  (t.b = 45 ∧ (t.a = 45 ∨ t.c = 45)) ∨
  (t.c = 45 ∧ (t.a = 45 ∨ t.b = 45)) ∨
  (t.a = 72 ∧ (t.b = 72 ∨ t.c = 72)) ∨
  (t.b = 72 ∧ (t.a = 72 ∨ t.c = 72)) ∨
  (t.c = 72 ∧ (t.a = 72 ∨ t.b = 72)) :=
sorry

end isosceles_triangle_base_angle_l2256_225637


namespace decimal_to_binary_88_l2256_225682

theorem decimal_to_binary_88 : 
  (88 : ℕ).digits 2 = [0, 0, 0, 1, 1, 0, 1] :=
sorry

end decimal_to_binary_88_l2256_225682


namespace unique_two_digit_number_l2256_225616

/-- Product of digits of a two-digit number -/
def P (n : ℕ) : ℕ := (n / 10) * (n % 10)

/-- Sum of digits of a two-digit number -/
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

/-- A two-digit number is between 10 and 99, inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_two_digit_number :
  ∃! M : ℕ, is_two_digit M ∧ M = P M + S M + 1 :=
sorry

end unique_two_digit_number_l2256_225616


namespace crane_sling_diameter_l2256_225666

/-- Represents the problem of determining the smallest safe rope diameter for a crane sling. -/
theorem crane_sling_diameter
  (M : ℝ) -- Mass of the load in tons
  (n : ℕ) -- Number of slings
  (α : ℝ) -- Angle of each sling with vertical in radians
  (k : ℝ) -- Safety factor
  (q : ℝ) -- Maximum load per thread in N/mm²
  (g : ℝ) -- Free fall acceleration in m/s²
  (h : M = 20)
  (hn : n = 3)
  (hα : α = Real.pi / 6) -- 30° in radians
  (hk : k = 6)
  (hq : q = 1000)
  (hg : g = 10)
  : ∃ (D : ℕ), D = 26 ∧ 
    D = ⌈(4 * (k * M * g) / (n * q * π * Real.cos α))^(1/2) * 1000⌉ ∧
    ∀ (D' : ℕ), D' < D → 
      D' < ⌈(4 * (k * M * g) / (n * q * π * Real.cos α))^(1/2) * 1000⌉ :=
sorry

end crane_sling_diameter_l2256_225666


namespace twelfth_finger_number_l2256_225620

-- Define the function f
def f : ℕ → ℕ
| 4 => 7
| 7 => 8
| 8 => 3
| 3 => 5
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define a function to apply f n times
def apply_f_n_times (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (apply_f_n_times n x)

-- Theorem statement
theorem twelfth_finger_number : apply_f_n_times 11 4 = 7 := by
  sorry

end twelfth_finger_number_l2256_225620


namespace digit_doubling_theorem_l2256_225661

def sumOfDigits (n : ℕ) : ℕ := sorry

def doubleDigitSum (n : ℕ) : ℕ := 2 * (sumOfDigits n)

def eventuallyOneDigit (n : ℕ) : Prop :=
  ∃ k, ∃ m : ℕ, (m < 10) ∧ (Nat.iterate doubleDigitSum k n = m)

theorem digit_doubling_theorem :
  (∀ n : ℕ, n ≠ 18 → doubleDigitSum n ≠ n) ∧
  (doubleDigitSum 18 = 18) ∧
  (∀ n : ℕ, n ≠ 18 → eventuallyOneDigit n) := by sorry

end digit_doubling_theorem_l2256_225661


namespace complement_union_equality_l2256_225634

universe u

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {2, 4, 6}

theorem complement_union_equality : 
  (U \ M) ∪ (U \ N) = U := by sorry

end complement_union_equality_l2256_225634


namespace perimeter_after_adding_tiles_l2256_225686

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a configuration -/
def add_tiles (config : TileConfiguration) (new_tiles : ℕ) : TileConfiguration :=
  { tiles := config.tiles + new_tiles, perimeter := config.perimeter }

theorem perimeter_after_adding_tiles 
  (initial : TileConfiguration) 
  (added_tiles : ℕ) 
  (final : TileConfiguration) :
  initial.tiles = 10 →
  initial.perimeter = 16 →
  added_tiles = 4 →
  final = add_tiles initial added_tiles →
  final.perimeter = 18 :=
by
  sorry

#check perimeter_after_adding_tiles

end perimeter_after_adding_tiles_l2256_225686


namespace divisible_by_5040_l2256_225611

theorem divisible_by_5040 (n : ℤ) (h : n > 3) : 
  ∃ k : ℤ, n^7 - 14*n^5 + 49*n^3 - 36*n = 5040 * k := by
  sorry

end divisible_by_5040_l2256_225611


namespace leak_drain_time_l2256_225645

-- Define the pump fill rate
def pump_rate : ℚ := 1 / 2

-- Define the time it takes to fill the tank with the leak
def fill_time_with_leak : ℚ := 17 / 8

-- Define the leak rate
def leak_rate : ℚ := pump_rate - (1 / fill_time_with_leak)

-- Theorem to prove
theorem leak_drain_time (pump_rate : ℚ) (fill_time_with_leak : ℚ) (leak_rate : ℚ) :
  pump_rate = 1 / 2 →
  fill_time_with_leak = 17 / 8 →
  leak_rate = pump_rate - (1 / fill_time_with_leak) →
  (1 / leak_rate) = 34 := by
  sorry

end leak_drain_time_l2256_225645


namespace holly_chocolate_milk_container_size_l2256_225662

/-- Represents the amount of chocolate milk Holly has throughout the day -/
structure ChocolateMilk where
  initial : ℕ  -- Initial amount of chocolate milk
  breakfast : ℕ  -- Amount drunk at breakfast
  lunch : ℕ  -- Amount drunk at lunch
  dinner : ℕ  -- Amount drunk at dinner
  final : ℕ  -- Final amount of chocolate milk
  new_container : ℕ  -- Size of the new container bought at lunch

/-- Theorem stating the size of the new container Holly bought -/
theorem holly_chocolate_milk_container_size 
  (h : ChocolateMilk) 
  (h_initial : h.initial = 16)
  (h_breakfast : h.breakfast = 8)
  (h_lunch : h.lunch = 8)
  (h_dinner : h.dinner = 8)
  (h_final : h.final = 56)
  : h.new_container = 64 := by
  sorry

end holly_chocolate_milk_container_size_l2256_225662


namespace solve_for_q_l2256_225675

theorem solve_for_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end solve_for_q_l2256_225675


namespace expression_change_l2256_225601

theorem expression_change (x a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ y => y^2 - 5*y
  (f (x + a) - f x = 2*a*x + a^2 - 5*a) ∧ 
  (f (x - a) - f x = -2*a*x + a^2 + 5*a) :=
sorry

end expression_change_l2256_225601


namespace complex_magnitude_equation_l2256_225607

theorem complex_magnitude_equation (z : ℂ) :
  Complex.abs z * (3 * z + 2 * Complex.I) = 2 * (Complex.I * z - 6) →
  Complex.abs z = 2 := by
sorry

end complex_magnitude_equation_l2256_225607


namespace dot_product_range_l2256_225673

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4 ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8

-- Define points M and N on the hypotenuse
def OnHypotenuse (M N : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ (t s : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧
  M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) ∧
  N = (s * A.1 + (1 - s) * B.1, s * A.2 + (1 - s) * B.2)

-- Define the distance between M and N
def MNDistance (M N : ℝ × ℝ) : Prop :=
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 2

-- Define the dot product of CM and CN
def DotProduct (C M N : ℝ × ℝ) : ℝ :=
  (M.1 - C.1) * (N.1 - C.1) + (M.2 - C.2) * (N.2 - C.2)

theorem dot_product_range (A B C M N : ℝ × ℝ) :
  Triangle A B C →
  OnHypotenuse M N A B →
  MNDistance M N →
  (3/2 : ℝ) ≤ DotProduct C M N ∧ DotProduct C M N ≤ 2 := by sorry

end dot_product_range_l2256_225673


namespace larger_number_problem_l2256_225643

theorem larger_number_problem (x y : ℤ) : 
  x + y = 96 → y = x + 12 → y = 54 := by
  sorry

end larger_number_problem_l2256_225643


namespace difference_of_squares_l2256_225671

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by sorry

end difference_of_squares_l2256_225671


namespace binomial_sum_cubes_l2256_225684

theorem binomial_sum_cubes (x y : ℤ) :
  (x^4 + 9*x*y^3)^3 + (-3*x^3*y - 9*y^4)^3 = x^12 - 729*y^12 := by
  sorry

end binomial_sum_cubes_l2256_225684


namespace two_extreme_points_l2256_225665

noncomputable section

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2 - x + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

/-- Theorem stating the range of a for which f(x) has two extreme value points -/
theorem two_extreme_points (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
    f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0) ↔ 
  (0 < a ∧ a < Real.exp (-1)) :=
sorry

end

end two_extreme_points_l2256_225665


namespace parallel_line_through_point_l2256_225617

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point (P : Point) (l1 l2 : Line) :
  P.liesOn l2 →
  l2.isParallelTo l1 →
  l1 = Line.mk 3 (-4) 6 →
  P = Point.mk 4 (-1) →
  l2 = Line.mk 3 (-4) (-16) := by
  sorry

end parallel_line_through_point_l2256_225617


namespace semicircle_chord_length_l2256_225674

theorem semicircle_chord_length (d : ℝ) (h : d > 0) :
  let r := d / 2
  let remaining_area := π * r^2 / 2 - π * (d/4)^2
  remaining_area = 16 * π^3 →
  2 * Real.sqrt (r^2 - (d/4)^2) = 32 * Real.sqrt 2 := by
  sorry

end semicircle_chord_length_l2256_225674


namespace least_product_consecutive_primes_above_50_l2256_225603

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def consecutive_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ ∀ r : ℕ, is_prime r → (p < r → r < q) → r = p ∨ r = q

theorem least_product_consecutive_primes_above_50 :
  ∃ p q : ℕ, consecutive_primes p q ∧ p > 50 ∧ q > 50 ∧
  p * q = 3127 ∧
  ∀ a b : ℕ, consecutive_primes a b → a > 50 → b > 50 → a * b ≥ 3127 :=
sorry

end least_product_consecutive_primes_above_50_l2256_225603


namespace range_of_expression_l2256_225639

theorem range_of_expression (x y : ℝ) (h : x^2 - y^2 = 4) :
  ∃ (z : ℝ), z = (1/x^2) - (y/x) ∧ -1 ≤ z ∧ z ≤ 5/4 :=
by sorry

end range_of_expression_l2256_225639


namespace constant_value_l2256_225621

theorem constant_value (f : ℝ → ℝ) (c : ℝ) 
  (h1 : ∀ x : ℝ, f x + c * f (8 - x) = x) 
  (h2 : f 2 = 2) : 
  c = 3 := by sorry

end constant_value_l2256_225621


namespace simple_interest_principal_calculation_l2256_225691

/-- Proves that given a simple interest of 4025.25, an interest rate of 9% per annum, 
and a time period of 5 years, the principal sum is 8950. -/
theorem simple_interest_principal_calculation :
  let simple_interest : ℝ := 4025.25
  let rate : ℝ := 9
  let time : ℝ := 5
  let principal : ℝ := simple_interest / (rate * time / 100)
  principal = 8950 := by sorry

end simple_interest_principal_calculation_l2256_225691


namespace like_terms_exponent_sum_l2256_225670

theorem like_terms_exponent_sum (x y : ℝ) (m n : ℕ) :
  (∃ (k : ℝ), -x^6 * y^(2*m) = k * x^(n+2) * y^4) →
  n + m = 6 := by
  sorry

end like_terms_exponent_sum_l2256_225670


namespace third_number_proof_l2256_225697

theorem third_number_proof :
  ∃ x : ℝ, 12.1212 + 17.0005 - x = 20.011399999999995 ∧ x = 9.110300000000005 := by
  sorry

end third_number_proof_l2256_225697


namespace grid_path_problem_l2256_225636

/-- The number of paths on a grid from (0,0) to (m,n) with exactly k steps -/
def grid_paths (m n k : ℕ) : ℕ := Nat.choose k m

/-- The problem statement -/
theorem grid_path_problem :
  let m : ℕ := 7  -- width of the grid
  let n : ℕ := 5  -- height of the grid
  let k : ℕ := 10 -- total number of steps
  grid_paths m n k = 120 := by sorry

end grid_path_problem_l2256_225636


namespace vector_dot_product_l2256_225642

/-- Given vectors a, b, c in ℝ², if a is parallel to b, then b · c = 10 -/
theorem vector_dot_product (a b c : ℝ × ℝ) : 
  a = (-1, 2) → b.1 = 2 → c = (7, 1) → (∃ k : ℝ, b = k • a) → b.1 * c.1 + b.2 * c.2 = 10 := by
  sorry

end vector_dot_product_l2256_225642


namespace min_value_a_l2256_225604

theorem min_value_a (a : ℝ) (ha : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (a / x + 4 / y) ≥ 16) ↔ a ≥ 4 := by
  sorry

end min_value_a_l2256_225604


namespace star_example_l2256_225656

/-- Custom binary operation ※ -/
def star (a b : ℕ) : ℕ := a * b + a + b

/-- Theorem: (3※4)※1 = 39 -/
theorem star_example : star (star 3 4) 1 = 39 := by
  sorry

end star_example_l2256_225656


namespace total_votes_l2256_225676

theorem total_votes (votes_for votes_against total : ℕ) : 
  votes_for = votes_against + 66 →
  votes_against = (40 * total) / 100 →
  votes_for + votes_against = total →
  total = 330 := by
sorry

end total_votes_l2256_225676


namespace min_value_sum_reciprocals_l2256_225659

/-- Given a function y = log_a(x + 3) - 1 where a > 0 and a ≠ 1, 
    its graph always passes through a fixed point A.
    If point A lies on the line mx + ny + 2 = 0 where mn > 0,
    then the minimum value of 1/m + 2/n is 4. -/
theorem min_value_sum_reciprocals (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : m > 0) (h4 : n > 0) (h5 : m * n > 0) :
  let f : ℝ → ℝ := λ x => (Real.log x) / (Real.log a) - 1
  let A : ℝ × ℝ := (-2, -1)
  (f (A.1 + 3) = A.2) →
  (m * A.1 + n * A.2 + 2 = 0) →
  (∀ x y, f y = x → m * x + n * y + 2 = 0) →
  (1 / m + 2 / n) ≥ 4 ∧ ∃ m₀ n₀, 1 / m₀ + 2 / n₀ = 4 := by
  sorry


end min_value_sum_reciprocals_l2256_225659


namespace product_of_three_numbers_l2256_225609

theorem product_of_three_numbers (x y z n : ℝ) 
  (sum_eq : x + y + z = 255)
  (n_eq_8x : n = 8 * x)
  (n_eq_y_minus_11 : n = y - 11)
  (n_eq_z_plus_13 : n = z + 13) :
  x * y * z = 209805 := by
  sorry

end product_of_three_numbers_l2256_225609


namespace tangent_line_at_origin_l2256_225678

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a - 3)*x

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a - 3)

/-- Theorem stating the equation of the tangent line at the origin -/
theorem tangent_line_at_origin (a : ℝ) (h : ∀ x, f' a x = f' a (-x)) :
  ∃ m : ℝ, m = -3 ∧ ∀ x, f a x = m * x + f a 0 := by sorry

end tangent_line_at_origin_l2256_225678


namespace square_root_of_square_l2256_225672

theorem square_root_of_square (n : ℝ) (h : n = 36) : Real.sqrt (n ^ 2) = n := by
  sorry

end square_root_of_square_l2256_225672


namespace question_1_question_2_l2256_225692

-- Define the given conditions
def p (x : ℝ) : Prop := -x^2 + 2*x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0
def s (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0

-- Define the sufficient but not necessary conditions
def sufficient_not_necessary (P Q : ℝ → Prop) : Prop :=
  (∀ x, P x → Q x) ∧ ∃ x, Q x ∧ ¬(P x)

-- Theorem for the first question
theorem question_1 (m : ℝ) :
  m > 0 → (sufficient_not_necessary (p) (q m) → m ≥ 3) :=
sorry

-- Theorem for the second question
theorem question_2 (m : ℝ) :
  m > 0 → (sufficient_not_necessary (fun x => ¬(s x)) (fun x => ¬(q m x)) → False) :=
sorry

end question_1_question_2_l2256_225692


namespace strawberry_cakes_ordered_l2256_225644

/-- The number of strawberry cakes Leila ordered -/
def strawberry_cakes : ℕ := 
  let chocolate_cake_price : ℕ := 12
  let strawberry_cake_price : ℕ := 22
  let chocolate_cakes_ordered : ℕ := 3
  let total_payment : ℕ := 168
  (total_payment - chocolate_cake_price * chocolate_cakes_ordered) / strawberry_cake_price

theorem strawberry_cakes_ordered : strawberry_cakes = 6 := by
  sorry

end strawberry_cakes_ordered_l2256_225644


namespace quadratic_inequality_solution_set_l2256_225649

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x + 1) * (x - 2) > 0} = {x : ℝ | x < -1 ∨ x > 2} := by sorry

end quadratic_inequality_solution_set_l2256_225649


namespace win_sector_area_l2256_225610

/-- Given a circular spinner with radius 12 cm and probability of winning 1/4,
    the area of the WIN sector is 36π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (total_area : ℝ) (win_area : ℝ) :
  r = 12 →
  p = 1 / 4 →
  total_area = π * r^2 →
  win_area = p * total_area →
  win_area = 36 * π := by
sorry

end win_sector_area_l2256_225610


namespace no_solutions_condition_l2256_225694

theorem no_solutions_condition (b : ℝ) : 
  (∀ a x : ℝ, a > 1 → a^(2-2*x^2) + (b+4)*a^(1-x^2) + 3*b + 4 ≠ 0) ↔ 
  (b ∈ Set.Ioc (-4/3) 0 ∪ Set.Ici 4) := by
  sorry

end no_solutions_condition_l2256_225694


namespace geometric_mean_point_existence_l2256_225615

/-- In a triangle ABC, point D on BC exists such that AD is the geometric mean of BD and DC
    if and only if b + c ≤ a√2, where a = BC, b = AC, and c = AB. -/
theorem geometric_mean_point_existence (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) :
  (∃ (t : ℝ), 0 < t ∧ t < a ∧ 
    (b^2 * t * (a - t) = a * (a - t) * t)) ↔ b + c ≤ a * Real.sqrt 2 :=
by sorry

end geometric_mean_point_existence_l2256_225615


namespace intersection_A_complement_B_l2256_225669

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {x | x^2 - 3*x > 0}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Ioo 1 3 := by sorry

end intersection_A_complement_B_l2256_225669


namespace cos_300_degrees_l2256_225630

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by sorry

end cos_300_degrees_l2256_225630


namespace field_trip_fraction_l2256_225614

theorem field_trip_fraction (b : ℚ) (g : ℚ) : 
  g = 2 * b →  -- There are twice as many girls as boys
  (2 / 3 * g + 3 / 5 * b) ≠ 0 → -- Total students on trip is not zero
  (2 / 3 * g) / (2 / 3 * g + 3 / 5 * b) = 20 / 29 := by
  sorry

end field_trip_fraction_l2256_225614


namespace candy_bar_sales_proof_l2256_225693

/-- The number of candy bars sold on the first day -/
def first_day_sales : ℕ := 190

/-- The number of days Sol sells candy bars in a week -/
def days_per_week : ℕ := 6

/-- The cost of each candy bar in cents -/
def candy_bar_cost : ℕ := 10

/-- The increase in candy bar sales each day after the first day -/
def daily_increase : ℕ := 4

/-- The total earnings in cents for the week -/
def total_earnings : ℕ := 1200

theorem candy_bar_sales_proof :
  (first_day_sales * days_per_week + 
   (daily_increase * (days_per_week - 1) * days_per_week) / 2) * 
  candy_bar_cost = total_earnings :=
sorry

end candy_bar_sales_proof_l2256_225693


namespace tree_planting_problem_l2256_225646

theorem tree_planting_problem (total_trees : ℕ) (a : ℕ) (b : ℕ) : 
  total_trees = 2013 →
  a * b = total_trees →
  (a - 5) * (b + 2) < total_trees →
  (a - 5) * (b + 3) > total_trees →
  a = 61 := by
sorry

end tree_planting_problem_l2256_225646


namespace system_solution_exists_l2256_225600

theorem system_solution_exists (m : ℝ) :
  (∃ x y : ℝ, y = (m + 1) * x + 2 ∧ y = (3 * m - 2) * x + 5 ∧ y > 5) ↔ m ≠ 3/2 := by
  sorry

end system_solution_exists_l2256_225600


namespace license_plate_count_l2256_225698

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 5

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of positions where the letter block can be placed -/
def letter_block_positions : ℕ := digits_count + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * (num_digits ^ digits_count) * (num_letters ^ letters_count)

theorem license_plate_count : total_license_plates = 10584576000 := by
  sorry

end license_plate_count_l2256_225698


namespace marble_probability_l2256_225647

theorem marble_probability (total : ℕ) (red : ℕ) (white : ℕ) (green : ℕ)
  (h_total : total = 100)
  (h_red : red = 35)
  (h_white : white = 30)
  (h_green : green = 10) :
  (red + white + green : ℚ) / total = 3/4 := by
  sorry

end marble_probability_l2256_225647


namespace consecutive_numbers_sum_no_ten_consecutive_sum_2016_seven_consecutive_sum_2016_l2256_225623

theorem consecutive_numbers_sum (n : ℕ) (sum : ℕ) : Prop :=
  ∃ a : ℕ, (n * a + n * (n - 1) / 2 = sum)

theorem no_ten_consecutive_sum_2016 : ¬ consecutive_numbers_sum 10 2016 := by
  sorry

theorem seven_consecutive_sum_2016 : consecutive_numbers_sum 7 2016 := by
  sorry

end consecutive_numbers_sum_no_ten_consecutive_sum_2016_seven_consecutive_sum_2016_l2256_225623


namespace fraction_inequality_solution_set_l2256_225667

theorem fraction_inequality_solution_set (x : ℝ) : 
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
sorry

end fraction_inequality_solution_set_l2256_225667


namespace backyard_area_l2256_225695

theorem backyard_area (length width : ℝ) 
  (h1 : 40 * length = 1000) 
  (h2 : 8 * (2 * (length + width)) = 1000) : 
  length * width = 937.5 := by
sorry

end backyard_area_l2256_225695


namespace geometric_sequence_sum_l2256_225641

/-- Given a geometric sequence {a_n} where a_1 = 1 and 4a_2, 2a_3, a_4 form an arithmetic sequence,
    prove that the sum a_2 + a_3 + a_4 equals 14. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                            -- a_1 = 1
  4 * a 2 - 2 * a 3 = 2 * a 3 - a 4 →  -- arithmetic sequence condition
  a 2 + a 3 + a 4 = 14 := by
sorry

end geometric_sequence_sum_l2256_225641


namespace segment_division_ratio_l2256_225681

/-- Given a line segment AC and points B and D on it, this theorem proves
    that if B divides AC in a 2:1 ratio and D divides AB in a 3:2 ratio,
    then D divides AC in a 2:3 ratio. -/
theorem segment_division_ratio (A B C D : ℝ) :
  (B - A) / (C - B) = 2 / 1 →
  (D - A) / (B - D) = 3 / 2 →
  (D - A) / (C - D) = 2 / 3 := by
sorry

end segment_division_ratio_l2256_225681


namespace estimate_larger_than_original_l2256_225632

theorem estimate_larger_than_original 
  (u v δ γ : ℝ) 
  (hu_pos : u > 0) 
  (hv_pos : v > 0) 
  (huv : u > v) 
  (hδγ : δ > γ) 
  (hγ_pos : γ > 0) : 
  (u + δ) - (v - γ) > u - v := by
sorry

end estimate_larger_than_original_l2256_225632


namespace matrix_not_invertible_sum_l2256_225663

def matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x + y, x, y],
    ![x, y + z, y],
    ![y, x, x + z]]

theorem matrix_not_invertible_sum (x y z : ℝ) :
  ¬(IsUnit (Matrix.det (matrix x y z))) →
  x + y + z = 0 →
  x / (y + z) + y / (x + z) + z / (x + y) = -3 := by
  sorry

end matrix_not_invertible_sum_l2256_225663


namespace shopkeeper_profit_l2256_225631

/-- The number of markers bought by the shopkeeper -/
def total_markers : ℕ := 2000

/-- The cost price of each marker in dollars -/
def cost_price : ℚ := 3/10

/-- The selling price of each marker in dollars -/
def selling_price : ℚ := 11/20

/-- The target profit in dollars -/
def target_profit : ℚ := 150

/-- The number of markers that need to be sold to achieve the target profit -/
def markers_to_sell : ℕ := 1364

theorem shopkeeper_profit :
  (markers_to_sell : ℚ) * selling_price - (total_markers : ℚ) * cost_price = target_profit :=
sorry

end shopkeeper_profit_l2256_225631


namespace complex_equality_l2256_225688

theorem complex_equality (z : ℂ) : 
  z = -1 + I ↔ Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
               Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end complex_equality_l2256_225688


namespace students_in_davids_grade_l2256_225640

/-- Given that David is both the 75th best and 75th worst student in his grade,
    prove that there are 149 students in total. -/
theorem students_in_davids_grade (n : ℕ) 
  (h1 : n ≥ 75)  -- David's grade has at least 75 students
  (h2 : ∃ (better worse : ℕ), better = 74 ∧ worse = 74 ∧ n = better + worse + 1) 
  : n = 149 := by
  sorry

end students_in_davids_grade_l2256_225640


namespace negation_of_universal_proposition_l2256_225625

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) := by sorry

end negation_of_universal_proposition_l2256_225625


namespace exponential_linear_independence_l2256_225655

theorem exponential_linear_independence 
  (k₁ k₂ k₃ : ℝ) 
  (h₁ : k₁ ≠ k₂) 
  (h₂ : k₁ ≠ k₃) 
  (h₃ : k₂ ≠ k₃) :
  ∀ (α₁ α₂ α₃ : ℝ), 
  (∀ x : ℝ, α₁ * Real.exp (k₁ * x) + α₂ * Real.exp (k₂ * x) + α₃ * Real.exp (k₃ * x) = 0) → 
  α₁ = 0 ∧ α₂ = 0 ∧ α₃ = 0 := by
sorry

end exponential_linear_independence_l2256_225655


namespace lcm_of_8_24_36_54_l2256_225650

theorem lcm_of_8_24_36_54 : Nat.lcm 8 (Nat.lcm 24 (Nat.lcm 36 54)) = 216 := by
  sorry

end lcm_of_8_24_36_54_l2256_225650


namespace quadratic_roots_interlace_l2256_225699

theorem quadratic_roots_interlace (p1 p2 q1 q2 : ℝ) 
  (h : (q1 - q2)^2 + (p1 - p2)*(p1*q2 - p2*q1) < 0) :
  ∃ (α1 β1 α2 β2 : ℝ),
    (∀ x, x^2 + p1*x + q1 = (x - α1) * (x - β1)) ∧
    (∀ x, x^2 + p2*x + q2 = (x - α2) * (x - β2)) ∧
    ((α1 < α2 ∧ α2 < β1 ∧ β1 < β2) ∨ (α2 < α1 ∧ α1 < β2 ∧ β2 < β1)) :=
by sorry

end quadratic_roots_interlace_l2256_225699


namespace smallest_n_is_correct_l2256_225696

/-- The smallest positive integer n such that all roots of z^6 - z^3 + 1 = 0 are n-th roots of unity -/
def smallest_n : ℕ := 18

/-- The polynomial z^6 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^6 - z^3 + 1

theorem smallest_n_is_correct :
  smallest_n = 18 ∧
  (∀ z : ℂ, f z = 0 → z^smallest_n = 1) ∧
  (∀ m : ℕ, m < smallest_n → ∃ z : ℂ, f z = 0 ∧ z^m ≠ 1) :=
sorry

end smallest_n_is_correct_l2256_225696


namespace greatest_power_of_three_l2256_225652

def w : ℕ := (List.range 30).foldl (· * ·) 1

theorem greatest_power_of_three (p : ℕ) : 
  (3^p ∣ w) ∧ ∀ q, q > p → ¬(3^q ∣ w) ↔ p = 15 :=
sorry

end greatest_power_of_three_l2256_225652


namespace vector_problem_l2256_225680

/-- Given two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_problem (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (1/2, 1)
  let c : ℝ × ℝ := (a.1 + 2*b.1, a.2 + 2*b.2)
  let d : ℝ × ℝ := (2*a.1 - b.1, 2*a.2 - b.2)
  are_parallel c d →
  (c.1 - 2*d.1, c.2 - 2*d.2) = (-1, -2) :=
by sorry

end vector_problem_l2256_225680


namespace marys_sheep_ratio_l2256_225633

theorem marys_sheep_ratio (total_sheep : ℕ) (sister_fraction : ℚ) (remaining_sheep : ℕ) : 
  total_sheep = 400 →
  sister_fraction = 1/4 →
  remaining_sheep = 150 →
  let sheep_to_sister := total_sheep * sister_fraction
  let sheep_after_sister := total_sheep - sheep_to_sister
  let sheep_to_brother := sheep_after_sister - remaining_sheep
  (sheep_to_brother : ℚ) / sheep_after_sister = 1/2 := by
    sorry

end marys_sheep_ratio_l2256_225633


namespace kyle_pe_laps_l2256_225668

/-- The number of laps Kyle jogged during track practice -/
def track_laps : ℝ := 2.12

/-- The total number of laps Kyle jogged -/
def total_laps : ℝ := 3.25

/-- The number of laps Kyle jogged in P.E. class -/
def pe_laps : ℝ := total_laps - track_laps

theorem kyle_pe_laps : pe_laps = 1.13 := by
  sorry

end kyle_pe_laps_l2256_225668


namespace point_trajectory_l2256_225624

/-- The trajectory of a point with constant ratio between distances to axes -/
theorem point_trajectory (k : ℝ) (h : k ≠ 0) :
  ∀ x y : ℝ, x ≠ 0 →
  (|y| / |x| = k) ↔ (y = k * x ∨ y = -k * x) :=
by sorry

end point_trajectory_l2256_225624


namespace denis_neighbors_l2256_225664

-- Define the set of children
inductive Child : Type
| Anya : Child
| Borya : Child
| Vera : Child
| Gena : Child
| Denis : Child

-- Define the line as a function from position (1 to 5) to Child
def Line := Fin 5 → Child

-- Define the conditions
def is_valid_line (l : Line) : Prop :=
  -- Borya is at the beginning of the line
  l 1 = Child.Borya ∧
  -- Vera is next to Anya but not next to Gena
  (∃ i : Fin 4, (l i = Child.Vera ∧ l (i+1) = Child.Anya) ∨ (l (i+1) = Child.Vera ∧ l i = Child.Anya)) ∧
  (∀ i : Fin 4, ¬(l i = Child.Vera ∧ l (i+1) = Child.Gena) ∧ ¬(l (i+1) = Child.Vera ∧ l i = Child.Gena)) ∧
  -- Among Anya, Borya, and Gena, no two are standing next to each other
  (∀ i : Fin 4, ¬((l i = Child.Anya ∨ l i = Child.Borya ∨ l i = Child.Gena) ∧
                 (l (i+1) = Child.Anya ∨ l (i+1) = Child.Borya ∨ l (i+1) = Child.Gena)))

-- Theorem statement
theorem denis_neighbors (l : Line) (h : is_valid_line l) :
  (∃ i : Fin 4, (l i = Child.Anya ∧ l (i+1) = Child.Denis) ∨ (l (i+1) = Child.Anya ∧ l i = Child.Denis)) ∧
  (∃ j : Fin 4, (l j = Child.Gena ∧ l (j+1) = Child.Denis) ∨ (l (j+1) = Child.Gena ∧ l j = Child.Denis)) :=
by sorry

end denis_neighbors_l2256_225664


namespace range_of_a_l2256_225679

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l2256_225679


namespace ellipse_k_range_l2256_225635

/-- An ellipse represented by the equation x² + ky² = 2 with foci on the y-axis -/
structure Ellipse where
  k : ℝ
  eq : ∀ (x y : ℝ), x^2 + k * y^2 = 2
  foci_on_y_axis : True  -- This is a placeholder for the foci condition

/-- The range of k for an ellipse with the given properties is (0, 1) -/
theorem ellipse_k_range (e : Ellipse) : 0 < e.k ∧ e.k < 1 := by
  sorry

end ellipse_k_range_l2256_225635
