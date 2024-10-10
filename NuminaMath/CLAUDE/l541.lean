import Mathlib

namespace expression_value_l541_54190

theorem expression_value (a b m n x : ℝ) : 
  (a = -b) →                   -- a and b are opposite numbers
  (m * n = 1) →                -- m and n are reciprocal numbers
  (m - n ≠ 0) →                -- given condition
  (abs x = 2) →                -- absolute value of x is 2
  (-2 * m * n + (b + a) / (m - n) - x = -4 ∨ 
   -2 * m * n + (b + a) / (m - n) - x = 0) :=
by sorry

end expression_value_l541_54190


namespace equation_solution_l541_54112

theorem equation_solution : 
  ∀ x : ℝ, x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by sorry

end equation_solution_l541_54112


namespace caterer_order_conditions_caterer_order_underdetermined_l541_54129

/-- Represents the order of a caterer -/
structure CatererOrder where
  x : ℝ  -- number of ice-cream bars
  y : ℝ  -- number of sundaes
  z : ℝ  -- number of milkshakes
  m : ℝ  -- price of each milkshake

/-- Theorem stating the conditions of the caterer's order -/
theorem caterer_order_conditions (order : CatererOrder) : Prop :=
  0.60 * order.x + 1.20 * order.y + order.m * order.z = 425 ∧
  order.x + order.y + order.z = 350

/-- Theorem stating that the conditions do not uniquely determine the order -/
theorem caterer_order_underdetermined :
  ∃ (order1 order2 : CatererOrder),
    caterer_order_conditions order1 ∧
    caterer_order_conditions order2 ∧
    order1 ≠ order2 := by
  sorry

end caterer_order_conditions_caterer_order_underdetermined_l541_54129


namespace min_value_f_max_value_y_l541_54104

-- Problem 1
theorem min_value_f (x : ℝ) (hx : x > 0) :
  2 / x + 2 * x ≥ 4 ∧ (2 / x + 2 * x = 4 ↔ x = 1) :=
by sorry

-- Problem 2
theorem max_value_y (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) :
  x * (1 - 3 * x) ≤ 1/12 :=
by sorry

end min_value_f_max_value_y_l541_54104


namespace solve_strawberry_problem_l541_54153

def strawberry_problem (betty_strawberries : ℕ) (matthew_extra : ℕ) (jar_strawberries : ℕ) (total_money : ℕ) : Prop :=
  let matthew_strawberries := betty_strawberries + matthew_extra
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let num_jars := total_strawberries / jar_strawberries
  let price_per_jar := total_money / num_jars
  price_per_jar = 4

theorem solve_strawberry_problem :
  strawberry_problem 16 20 7 40 := by
  sorry

end solve_strawberry_problem_l541_54153


namespace smallest_lucky_number_unique_lucky_number_divisible_by_seven_l541_54175

/-- Definition of a lucky number -/
def is_lucky_number (M : ℕ) : Prop :=
  ∃ (A B : ℕ), 
    M % 10 ≠ 0 ∧
    M = A * B ∧
    A ≥ B ∧
    10 ≤ A ∧ A < 100 ∧
    10 ≤ B ∧ B < 100 ∧
    (A / 10) = (B / 10) ∧
    (A % 10) + (B % 10) = 6

/-- The smallest lucky number is 165 -/
theorem smallest_lucky_number : 
  is_lucky_number 165 ∧ ∀ M, is_lucky_number M → M ≥ 165 := by sorry

/-- There exists a unique lucky number M such that (A + B) / (A - B) is divisible by 7, and it equals 3968 -/
theorem unique_lucky_number_divisible_by_seven :
  ∃! M, is_lucky_number M ∧ 
    (∃ A B, M = A * B ∧ ((A + B) / (A - B)) % 7 = 0) ∧
    M = 3968 := by sorry

end smallest_lucky_number_unique_lucky_number_divisible_by_seven_l541_54175


namespace avery_egg_cartons_l541_54107

/-- Given the number of chickens, eggs per chicken, and number of filled cartons,
    calculate the number of eggs per carton. -/
def eggs_per_carton (num_chickens : ℕ) (eggs_per_chicken : ℕ) (num_cartons : ℕ) : ℕ :=
  (num_chickens * eggs_per_chicken) / num_cartons

/-- Prove that with 20 chickens laying 6 eggs each, filling 10 cartons results in 12 eggs per carton. -/
theorem avery_egg_cartons : eggs_per_carton 20 6 10 = 12 := by
  sorry

end avery_egg_cartons_l541_54107


namespace noa_score_l541_54181

/-- Proves that Noa scored 30 points given the conditions of the problem -/
theorem noa_score (noa_score : ℕ) (phillip_score : ℕ) : 
  phillip_score = 2 * noa_score →
  noa_score + phillip_score = 90 →
  noa_score = 30 := by
sorry

end noa_score_l541_54181


namespace roots_of_quadratic_l541_54159

theorem roots_of_quadratic (x y : ℝ) : 
  x + y = 10 ∧ |x - y| = 6 → 
  x^2 - 10*x + 16 = 0 ∧ y^2 - 10*y + 16 = 0 := by
sorry

end roots_of_quadratic_l541_54159


namespace zach_needs_six_more_l541_54115

/-- Calculates how much more money Zach needs to buy a bike --/
def money_needed_for_bike (bike_cost allowance lawn_pay babysit_rate current_savings babysit_hours : ℕ) : ℕ :=
  let total_earnings := allowance + lawn_pay + babysit_rate * babysit_hours
  let total_savings := current_savings + total_earnings
  if total_savings ≥ bike_cost then 0
  else bike_cost - total_savings

/-- Proves that Zach needs $6 more to buy the bike --/
theorem zach_needs_six_more : 
  money_needed_for_bike 100 5 10 7 65 2 = 6 := by
  sorry

end zach_needs_six_more_l541_54115


namespace lp_has_only_minimum_l541_54131

/-- A point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The feasible region defined by the constraints -/
def FeasibleRegion (p : Point) : Prop :=
  6 * p.x + 3 * p.y < 15 ∧ p.y ≤ p.x + 1 ∧ p.x - 5 * p.y ≤ 3

/-- The objective function -/
def ObjectiveFunction (p : Point) : ℝ :=
  3 * p.x + 5 * p.y

/-- The statement that the linear programming problem has only a minimum value and no maximum value -/
theorem lp_has_only_minimum :
  (∃ (p : Point), FeasibleRegion p ∧
    ∀ (q : Point), FeasibleRegion q → ObjectiveFunction p ≤ ObjectiveFunction q) ∧
  (¬ ∃ (p : Point), FeasibleRegion p ∧
    ∀ (q : Point), FeasibleRegion q → ObjectiveFunction q ≤ ObjectiveFunction p) :=
sorry

end lp_has_only_minimum_l541_54131


namespace expression_evaluation_l541_54162

/-- Proves that the given expression evaluates to 58.51045 -/
theorem expression_evaluation :
  (3.415 * 2.67) + (8.641 - 1.23) / (0.125 * 4.31) + 5.97^2 = 58.51045 := by
  sorry

end expression_evaluation_l541_54162


namespace leftmost_box_value_l541_54160

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i, i < n - 2 → a i + a (i + 1) + a (i + 2) = 2005

theorem leftmost_box_value (a : ℕ → ℕ) :
  sequence_sum a 9 →
  a 1 = 888 →
  a 2 = 999 →
  a 0 = 118 :=
by sorry

end leftmost_box_value_l541_54160


namespace inverse_variation_problem_l541_54172

theorem inverse_variation_problem (x y : ℝ) : 
  (x > 0) →
  (y > 0) →
  (∃ k : ℝ, ∀ x y, x^3 * y = k) →
  (2^3 * 8 = x^3 * 512) →
  (y = 512) →
  x = 1/2 := by
  sorry

end inverse_variation_problem_l541_54172


namespace swimmer_speed_in_still_water_l541_54196

/-- Represents the speed of a swimmer in still water and the speed of the stream -/
structure SwimmerSpeeds where
  manSpeed : ℝ
  streamSpeed : ℝ

/-- Calculates the effective speed given the swimmer's speed and stream speed -/
def effectiveSpeed (speeds : SwimmerSpeeds) (isDownstream : Bool) : ℝ :=
  if isDownstream then speeds.manSpeed + speeds.streamSpeed else speeds.manSpeed - speeds.streamSpeed

/-- Theorem stating the speed of the man in still water given the problem conditions -/
theorem swimmer_speed_in_still_water
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (time : ℝ)
  (h_downstream : downstream_distance = 28)
  (h_upstream : upstream_distance = 12)
  (h_time : time = 2)
  (speeds : SwimmerSpeeds)
  (h_downstream_speed : effectiveSpeed speeds true = downstream_distance / time)
  (h_upstream_speed : effectiveSpeed speeds false = upstream_distance / time) :
  speeds.manSpeed = 10 := by
sorry


end swimmer_speed_in_still_water_l541_54196


namespace definite_integral_equals_26_over_3_l541_54151

theorem definite_integral_equals_26_over_3 : 
  ∫ x in (0)..(2 * Real.arctan (1/2)), (1 + Real.sin x) / ((1 - Real.sin x)^2) = 26/3 := by sorry

end definite_integral_equals_26_over_3_l541_54151


namespace emily_income_l541_54191

/-- Represents the tax structure and Emily's tax payment --/
structure TaxSystem where
  q : ℝ  -- Base tax rate
  income : ℝ  -- Emily's annual income
  total_tax : ℝ  -- Total tax paid by Emily

/-- The tax system satisfies the given conditions --/
def valid_tax_system (ts : TaxSystem) : Prop :=
  ts.total_tax = 
    (0.01 * ts.q * 35000 + 
     0.01 * (ts.q + 3) * 15000 + 
     0.01 * (ts.q + 5) * (ts.income - 50000)) *
    (if ts.income > 50000 then 1 else 0) +
    (0.01 * ts.q * 35000 + 
     0.01 * (ts.q + 3) * (ts.income - 35000)) *
    (if ts.income > 35000 ∧ ts.income ≤ 50000 then 1 else 0) +
    (0.01 * ts.q * ts.income) *
    (if ts.income ≤ 35000 then 1 else 0)

/-- Emily's total tax is (q + 0.75)% of her income --/
def emily_tax_condition (ts : TaxSystem) : Prop :=
  ts.total_tax = 0.01 * (ts.q + 0.75) * ts.income

/-- Theorem: Emily's income is $48235 --/
theorem emily_income (ts : TaxSystem) 
  (h1 : valid_tax_system ts) 
  (h2 : emily_tax_condition ts) : 
  ts.income = 48235 :=
sorry

end emily_income_l541_54191


namespace alloy_composition_ratio_l541_54193

/-- Given two alloys A and B, with known compositions and mixture properties,
    prove that the ratio of tin to copper in alloy B is 1:4. -/
theorem alloy_composition_ratio :
  -- Define the masses of alloys
  ∀ (mass_A mass_B : ℝ),
  -- Define the ratio of lead to tin in alloy A
  ∀ (lead_ratio tin_ratio : ℝ),
  -- Define the total amount of tin in the mixture
  ∀ (total_tin : ℝ),
  -- Conditions
  mass_A = 60 →
  mass_B = 100 →
  lead_ratio = 3 →
  tin_ratio = 2 →
  total_tin = 44 →
  -- Calculate tin in alloy A
  let tin_A := (tin_ratio / (lead_ratio + tin_ratio)) * mass_A
  -- Calculate tin in alloy B
  let tin_B := total_tin - tin_A
  -- Calculate copper in alloy B
  let copper_B := mass_B - tin_B
  -- Prove the ratio
  tin_B / copper_B = 1 / 4 :=
by sorry

end alloy_composition_ratio_l541_54193


namespace no_solution_implies_m_leq_one_l541_54102

theorem no_solution_implies_m_leq_one :
  (∀ x : ℝ, ¬(2*x - 1 > 1 ∧ x < m)) → m ≤ 1 := by
  sorry

end no_solution_implies_m_leq_one_l541_54102


namespace a_range_l541_54173

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > a then x + 2 else x^2 + 5*x + 2

-- Define g(x) in terms of f(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2*x

-- Define a predicate for g having exactly three distinct zeros
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    g a x = 0 ∧ g a y = 0 ∧ g a z = 0 ∧
    ∀ w : ℝ, g a w = 0 → w = x ∨ w = y ∨ w = z

-- The main theorem
theorem a_range (a : ℝ) :
  has_three_distinct_zeros a ↔ a ∈ Set.Icc (-1 : ℝ) 2 ∧ a ≠ 2 :=
sorry

end a_range_l541_54173


namespace volume_of_specific_prism_l541_54120

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculate the volume of a prism formed by slicing a rectangular solid -/
def volumeOfSlicedPrism (solid : RectangularSolid) (plane : Plane3D) (p1 p2 p3 vertex : Point3D) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific prism -/
theorem volume_of_specific_prism :
  let solid := RectangularSolid.mk 4 3 3
  let p1 := Point3D.mk 0 0 3
  let p2 := Point3D.mk 4 0 3
  let p3 := Point3D.mk 0 3 1.5
  let vertex := Point3D.mk 4 3 0
  let plane := Plane3D.mk (-0.75) 0.75 1 (-3)
  volumeOfSlicedPrism solid plane p1 p2 p3 vertex = 13.5 := by
  sorry

end volume_of_specific_prism_l541_54120


namespace carols_birthday_invitations_l541_54122

/-- Given that Carol buys invitation packages with 2 invitations each and needs 5 packs,
    prove that she is inviting 10 friends. -/
theorem carols_birthday_invitations
  (invitations_per_pack : ℕ)
  (packs_needed : ℕ)
  (h1 : invitations_per_pack = 2)
  (h2 : packs_needed = 5) :
  invitations_per_pack * packs_needed = 10 := by
  sorry

end carols_birthday_invitations_l541_54122


namespace not_divisible_by_4_8_16_32_l541_54174

def x : ℕ := 80 + 112 + 144 + 176 + 304 + 368 + 3248 + 17

theorem not_divisible_by_4_8_16_32 : 
  ¬(∃ k : ℕ, x = 4 * k) ∧ 
  ¬(∃ k : ℕ, x = 8 * k) ∧ 
  ¬(∃ k : ℕ, x = 16 * k) ∧ 
  ¬(∃ k : ℕ, x = 32 * k) :=
by sorry

end not_divisible_by_4_8_16_32_l541_54174


namespace rice_purchase_problem_l541_54178

/-- The problem of determining the amount of rice bought given the prices and quantities of different grains --/
theorem rice_purchase_problem (rice_price corn_price beans_price : ℚ)
  (total_weight total_cost : ℚ) (beans_weight : ℚ) :
  rice_price = 75 / 100 →
  corn_price = 110 / 100 →
  beans_price = 55 / 100 →
  total_weight = 36 →
  total_cost = 2835 / 100 →
  beans_weight = 8 →
  ∃ (rice_weight : ℚ), 
    (rice_weight + (total_weight - rice_weight - beans_weight) + beans_weight = total_weight) ∧
    (rice_price * rice_weight + corn_price * (total_weight - rice_weight - beans_weight) + beans_price * beans_weight = total_cost) ∧
    (abs (rice_weight - 196 / 10) < 1 / 10) :=
by sorry

end rice_purchase_problem_l541_54178


namespace x_plus_inv_x_power_n_is_integer_l541_54100

theorem x_plus_inv_x_power_n_is_integer
  (x : ℝ) (h : ∃ (k : ℤ), x + 1 / x = k) :
  ∀ n : ℕ, ∃ (m : ℤ), x^n + 1 / x^n = m :=
by sorry

end x_plus_inv_x_power_n_is_integer_l541_54100


namespace rabbits_ate_23_pumpkins_l541_54169

/-- The number of pumpkins Sara initially grew -/
def initial_pumpkins : ℕ := 43

/-- The number of pumpkins Sara has left -/
def remaining_pumpkins : ℕ := 20

/-- The number of pumpkins eaten by rabbits -/
def eaten_pumpkins : ℕ := initial_pumpkins - remaining_pumpkins

theorem rabbits_ate_23_pumpkins : eaten_pumpkins = 23 := by
  sorry

end rabbits_ate_23_pumpkins_l541_54169


namespace field_area_l541_54119

/-- The area of a rectangular field with specific fencing conditions -/
theorem field_area (L W : ℝ) : 
  L = 20 →  -- One side is 20 feet
  2 * W + L = 41 →  -- Total fencing is 41 feet
  L * W = 210 :=  -- Area of the field
by
  sorry

end field_area_l541_54119


namespace stability_ratio_calculation_l541_54176

theorem stability_ratio_calculation (T H L : ℚ) : 
  T = 3 → H = 9 → L = (30 * T^3) / H^3 → L = 10/9 := by
  sorry

end stability_ratio_calculation_l541_54176


namespace fraction_simplification_l541_54126

theorem fraction_simplification (a b c : ℝ) :
  ((a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) / (a^2 - b^2 - c^2 - 2*b*c) = (a + b + c) / (a - b - c)) ∧
  ((a^2 - 3*a*b + a*c + 2*b^2 - 2*b*c) / (a^2 - b^2 + 2*b*c - c^2) = (a - 2*b) / (a + b - c)) :=
by sorry

end fraction_simplification_l541_54126


namespace investment_interest_rate_l541_54132

/-- Proves that given the specified investment conditions, the second year's interest rate is 6% -/
theorem investment_interest_rate (initial_investment : ℝ) (first_rate : ℝ) (second_rate : ℝ) (final_value : ℝ) :
  initial_investment = 15000 →
  first_rate = 0.08 →
  final_value = 17160 →
  initial_investment * (1 + first_rate) * (1 + second_rate) = final_value →
  second_rate = 0.06 := by
  sorry

end investment_interest_rate_l541_54132


namespace necessary_but_not_sufficient_l541_54156

open Real

theorem necessary_but_not_sufficient 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) : 
  (a > b ∧ b > ℯ → a^b < b^a) ∧
  ¬(a^b < b^a → a > b ∧ b > ℯ) :=
sorry

end necessary_but_not_sufficient_l541_54156


namespace infinitely_many_primes_4k_plus_3_l541_54164

theorem infinitely_many_primes_4k_plus_3 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4 * k + 3) →
  ∃ q, Nat.Prime q ∧ (∃ m, q = 4 * m + 3) ∧ q ∉ S :=
by sorry

end infinitely_many_primes_4k_plus_3_l541_54164


namespace sqrt_of_sixteen_l541_54136

theorem sqrt_of_sixteen : ∃ (x : ℝ), x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end sqrt_of_sixteen_l541_54136


namespace triangle_inequality_l541_54130

theorem triangle_inequality (a b c : ℝ) (x y z : ℝ) : 
  a ≥ b ∧ b ≥ c ∧ c > 0 ∧ 
  0 < x ∧ x < π ∧ 0 < y ∧ y < π ∧ 0 < z ∧ z < π ∧ x + y + z = π →
  b * c + c * a - a * b < b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ∧
  b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ≤ (1 / 2) * (a^2 + b^2 + c^2) :=
by sorry

end triangle_inequality_l541_54130


namespace sqrt_equation_solution_l541_54183

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 16 :=
by
  -- The unique solution is z = -251/4
  use -251/4
  sorry

end sqrt_equation_solution_l541_54183


namespace no_sequences_exist_l541_54135

theorem no_sequences_exist : ¬ ∃ (a b : ℕ → ℝ), 
  (∀ n : ℕ, (3/2) * Real.pi ≤ a n ∧ a n ≤ b n) ∧
  (∀ n : ℕ, ∀ x : ℝ, 0 < x ∧ x < 1 → Real.cos (a n * x) + Real.cos (b n * x) ≥ -1 / n) := by
  sorry

end no_sequences_exist_l541_54135


namespace pauls_peaches_l541_54145

/-- Given that Audrey has 26 peaches and the difference between Audrey's and Paul's peaches is 22,
    prove that Paul has 4 peaches. -/
theorem pauls_peaches (audrey_peaches : ℕ) (peach_difference : ℕ) 
    (h1 : audrey_peaches = 26)
    (h2 : peach_difference = 22)
    (h3 : audrey_peaches - paul_peaches = peach_difference) : 
    paul_peaches = 4 :=
by
  sorry

end pauls_peaches_l541_54145


namespace bacteria_growth_7_hours_l541_54138

/-- Calculates the number of bacteria after a given number of hours, 
    given an initial count and doubling time. -/
def bacteria_growth (initial_count : ℕ) (hours : ℕ) : ℕ :=
  initial_count * 2^hours

/-- Theorem stating that after 7 hours, starting with 10 bacteria, 
    the population will be 1280 bacteria. -/
theorem bacteria_growth_7_hours : 
  bacteria_growth 10 7 = 1280 := by
  sorry

end bacteria_growth_7_hours_l541_54138


namespace min_diff_y_x_l541_54144

theorem min_diff_y_x (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  Even x ∧ Odd y ∧ Odd z ∧
  (∀ w, w - x ≥ 9 → z ≤ w) →
  ∃ (d : ℤ), d = y - x ∧ (∀ d' : ℤ, d' = y - x → d ≤ d') ∧ d = 1 := by
sorry

end min_diff_y_x_l541_54144


namespace calculator_operations_l541_54150

def A (n : ℕ) : ℕ := 2 * n
def B (n : ℕ) : ℕ := n + 1

def applyKeys (n : ℕ) (keys : List (ℕ → ℕ)) : ℕ :=
  keys.foldl (fun acc f => f acc) n

theorem calculator_operations :
  (∃ keys : List (ℕ → ℕ), keys.length = 4 ∧ applyKeys 1 keys = 10) ∧
  (∃ keys : List (ℕ → ℕ), keys.length = 6 ∧ applyKeys 1 keys = 15) ∧
  (∃ keys : List (ℕ → ℕ), keys.length = 8 ∧ applyKeys 1 keys = 100) := by
  sorry

end calculator_operations_l541_54150


namespace unique_pair_existence_l541_54139

theorem unique_pair_existence :
  ∃! (c d : ℝ), 0 < c ∧ c < d ∧ d < π / 2 ∧
    Real.sin (Real.cos c) = c ∧ Real.cos (Real.sin d) = d := by
  sorry

end unique_pair_existence_l541_54139


namespace steves_speed_steves_speed_proof_l541_54186

/-- Calculates Steve's speed during John's final push in a race --/
theorem steves_speed (john_initial_behind : ℝ) (john_speed : ℝ) (john_time : ℝ) (john_final_ahead : ℝ) : ℝ :=
  let john_distance := john_speed * john_time
  let steve_distance := john_distance - (john_initial_behind + john_final_ahead)
  steve_distance / john_time

/-- Proves that Steve's speed during John's final push was 3.8 m/s --/
theorem steves_speed_proof :
  steves_speed 15 4.2 42.5 2 = 3.8 := by
  sorry

end steves_speed_steves_speed_proof_l541_54186


namespace new_person_weight_l541_54103

def initial_persons : ℕ := 10
def average_weight_increase : ℚ := 63/10
def replaced_person_weight : ℚ := 65

theorem new_person_weight :
  let total_weight_increase : ℚ := initial_persons * average_weight_increase
  let new_person_weight : ℚ := replaced_person_weight + total_weight_increase
  new_person_weight = 128 := by sorry

end new_person_weight_l541_54103


namespace system_solution_l541_54161

theorem system_solution : ∃ (x y z : ℚ), 
  (7 * x + 3 * y = z - 10) ∧ 
  (2 * x - 4 * y = 3 * z + 20) := by
  use 0, -50/13, -20/13
  sorry

end system_solution_l541_54161


namespace intersection_of_A_and_B_l541_54192

def A : Set (ℝ × ℝ) := {p | p.2 = 3 * p.1 - 2}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1 ^ 2}

theorem intersection_of_A_and_B :
  A ∩ B = {(1, 1), (2, 4)} := by sorry

end intersection_of_A_and_B_l541_54192


namespace polynomial_factorization_l541_54146

theorem polynomial_factorization (a x y : ℝ) : 3*a*x^2 + 6*a*x*y + 3*a*y^2 = 3*a*(x+y)^2 := by
  sorry

end polynomial_factorization_l541_54146


namespace complex_point_C_l541_54110

/-- Given points A, B, and C in the complex plane, prove that C corresponds to 4-2i -/
theorem complex_point_C (A B C : ℂ) : 
  A = 2 + I →
  B - A = 1 + 2*I →
  C - B = 3 - I →
  C = 4 - 2*I :=
by sorry

end complex_point_C_l541_54110


namespace triangle_properties_l541_54142

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_A : 0 < A
  pos_B : 0 < B
  pos_C : 0 < C
  sum_angles : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem triangle_properties (t : Triangle) :
  (t.c^2 + t.a*t.b = t.c*(t.a*Real.cos t.B - t.b*Real.cos t.A) + 2*t.b^2 → t.C = π/3) ∧
  (t.C = π/3 ∧ t.c = 2*Real.sqrt 3 → -2*Real.sqrt 3 < 4*Real.sin t.B - t.a ∧ 4*Real.sin t.B - t.a < 2*Real.sqrt 3) :=
by sorry

end triangle_properties_l541_54142


namespace tree_height_after_two_years_l541_54109

/-- The height of a tree after a given number of years, given that it triples its height every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem stating that a tree tripling its height yearly reaches 9 feet after 2 years if it's 81 feet after 4 years -/
theorem tree_height_after_two_years 
  (h : tree_height (tree_height h₀ 2) 2 = 81) : 
  tree_height h₀ 2 = 9 :=
by
  sorry

#check tree_height_after_two_years

end tree_height_after_two_years_l541_54109


namespace line_y_intercept_l541_54143

/-- Proves that for a line ax + y + 2 = 0 with an inclination angle of 3π/4, the y-intercept is -2 -/
theorem line_y_intercept (a : ℝ) : 
  (∀ x y : ℝ, a * x + y + 2 = 0) → 
  (Real.tan (3 * Real.pi / 4) = -a) → 
  (∃ x : ℝ, 0 * x + (-2) + 2 = 0) :=
by sorry

end line_y_intercept_l541_54143


namespace tickets_left_to_sell_l541_54182

theorem tickets_left_to_sell (total : ℕ) (first_week : ℕ) (second_week : ℕ) 
  (h1 : total = 90) 
  (h2 : first_week = 38) 
  (h3 : second_week = 17) :
  total - (first_week + second_week) = 35 := by
  sorry

end tickets_left_to_sell_l541_54182


namespace unique_two_digit_number_mod_4_17_l541_54111

theorem unique_two_digit_number_mod_4_17 : 
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 4 = 1 ∧ n % 17 = 1 :=
by sorry

end unique_two_digit_number_mod_4_17_l541_54111


namespace repeating_decimal_as_fraction_l541_54170

-- Define the repeating decimal 4.8̄
def repeating_decimal : ℚ := 4 + 8/9

-- Theorem to prove
theorem repeating_decimal_as_fraction : repeating_decimal = 44/9 := by
  sorry

end repeating_decimal_as_fraction_l541_54170


namespace lcm_eight_fifteen_l541_54185

theorem lcm_eight_fifteen : Nat.lcm 8 15 = 120 := by sorry

end lcm_eight_fifteen_l541_54185


namespace sum_exterior_angles_quadrilateral_l541_54157

/-- A quadrilateral is a polygon with four sides. -/
def Quadrilateral : Type := Unit  -- Placeholder definition

/-- The sum of exterior angles of a polygon. -/
def sum_exterior_angles (p : Type) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of a quadrilateral is 360 degrees. -/
theorem sum_exterior_angles_quadrilateral :
  sum_exterior_angles Quadrilateral = 360 := by sorry

end sum_exterior_angles_quadrilateral_l541_54157


namespace petyas_sum_l541_54134

theorem petyas_sum (n k : ℕ) : 
  (∀ i ∈ Finset.range (k + 1), Even (n + 2 * i)) →
  ((k + 1) * (n + k) = 30 * (n + 2 * k)) →
  ((k + 1) * (n + k) = 90 * n) →
  n = 44 ∧ k = 44 :=
by sorry

end petyas_sum_l541_54134


namespace lowest_possible_score_l541_54108

theorem lowest_possible_score (num_tests : Nat) (max_score : Nat) (avg_score : Nat) :
  num_tests = 4 →
  max_score = 100 →
  avg_score = 88 →
  ∃ (scores : Fin num_tests → Nat),
    (∀ i, scores i ≤ max_score) ∧
    (Finset.sum Finset.univ (λ i => scores i) = num_tests * avg_score) ∧
    (∃ i, scores i = 52) ∧
    (∀ i, scores i ≥ 52) :=
by
  sorry

#check lowest_possible_score

end lowest_possible_score_l541_54108


namespace geometric_sequence_seventh_term_l541_54141

theorem geometric_sequence_seventh_term 
  (a₁ : ℝ) 
  (a₃ : ℝ) 
  (h₁ : a₁ = 3) 
  (h₃ : a₃ = 1/9) : 
  let r := (a₃ / a₁) ^ (1/2)
  a₁ * r^6 = Real.sqrt 3 / 81 := by
sorry

end geometric_sequence_seventh_term_l541_54141


namespace exists_six_digit_number_without_identical_endings_l541_54117

theorem exists_six_digit_number_without_identical_endings : ∃ A : ℕ, 
  (100000 ≤ A ∧ A < 1000000) ∧ 
  ∀ k : ℕ, k ≤ 500000 → ∀ d : ℕ, d < 10 → 
    (k * A) % 1000000 ≠ d * 111111 := by
  sorry

end exists_six_digit_number_without_identical_endings_l541_54117


namespace inequality_proof_l541_54123

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2 * x * y) / (x + y) + Real.sqrt ((x^2 + y^2) / 2) ≥ (x + y) / 2 + Real.sqrt (x * y) := by
  sorry

end inequality_proof_l541_54123


namespace not_sufficient_nor_necessary_l541_54165

theorem not_sufficient_nor_necessary : 
  ¬(∀ x : ℝ, x < 0 → Real.log (x + 1) ≤ 0) ∧ 
  ¬(∀ x : ℝ, Real.log (x + 1) ≤ 0 → x < 0) := by
  sorry

end not_sufficient_nor_necessary_l541_54165


namespace probability_of_2500_is_6_125_l541_54140

/-- The number of outcomes on the wheel -/
def num_outcomes : ℕ := 5

/-- The number of spins -/
def num_spins : ℕ := 3

/-- The number of ways to achieve the desired sum -/
def num_successful_combinations : ℕ := 6

/-- The total number of possible outcomes after three spins -/
def total_possibilities : ℕ := num_outcomes ^ num_spins

/-- The probability of earning exactly $2500 in three spins -/
def probability_of_2500 : ℚ := num_successful_combinations / total_possibilities

theorem probability_of_2500_is_6_125 : probability_of_2500 = 6 / 125 := by
  sorry

end probability_of_2500_is_6_125_l541_54140


namespace camping_trip_percentage_l541_54105

theorem camping_trip_percentage (total_students : ℕ) 
  (march_trip_percentage : ℝ) (march_over_100_percentage : ℝ)
  (june_trip_percentage : ℝ) (june_over_100_percentage : ℝ)
  (over_100_march_percentage : ℝ) :
  march_trip_percentage = 0.2 →
  march_over_100_percentage = 0.35 →
  june_trip_percentage = 0.15 →
  june_over_100_percentage = 0.4 →
  over_100_march_percentage = 0.7 →
  (march_trip_percentage + june_trip_percentage) * total_students = 
    0.35 * total_students :=
by sorry

end camping_trip_percentage_l541_54105


namespace crayon_selection_ways_l541_54106

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of crayons in the box. -/
def total_crayons : ℕ := 15

/-- The number of crayons Karl selects. -/
def selected_crayons : ℕ := 5

/-- Theorem stating that selecting 5 crayons from 15 crayons can be done in 3003 ways. -/
theorem crayon_selection_ways : binomial total_crayons selected_crayons = 3003 := by
  sorry

end crayon_selection_ways_l541_54106


namespace average_speed_problem_l541_54113

/-- The average speed for an hour drive, given that driving twice as fast for 4 hours covers 528 miles. -/
theorem average_speed_problem (v : ℝ) : v > 0 → 2 * v * 4 = 528 → v = 66 := by
  sorry

end average_speed_problem_l541_54113


namespace cost_of_450_candies_l541_54179

/-- The cost of buying a specific number of chocolate candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) : ℚ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem stating the cost of 450 chocolate candies -/
theorem cost_of_450_candies :
  cost_of_candies 30 (7.5 : ℚ) 450 = 112.5 := by
  sorry

#eval cost_of_candies 30 (7.5 : ℚ) 450

end cost_of_450_candies_l541_54179


namespace hyperbola_t_squared_l541_54118

-- Define a hyperbola
structure Hyperbola where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ
  vertical : Bool

-- Define a function to check if a point is on the hyperbola
def on_hyperbola (h : Hyperbola) (p : ℝ × ℝ) : Prop :=
  if h.vertical then
    (p.2 - h.center.2)^2 / h.b^2 - (p.1 - h.center.1)^2 / h.a^2 = 1
  else
    (p.1 - h.center.1)^2 / h.a^2 - (p.2 - h.center.2)^2 / h.b^2 = 1

-- Theorem statement
theorem hyperbola_t_squared (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_point1 : on_hyperbola h (4, -3))
  (h_point2 : on_hyperbola h (0, -2))
  (h_point3 : ∃ t : ℝ, on_hyperbola h (2, t)) :
  ∃ t : ℝ, on_hyperbola h (2, t) ∧ t^2 = 21/4 := by
  sorry

end hyperbola_t_squared_l541_54118


namespace only_expr2_same_type_as_reference_l541_54137

-- Define the structure of a monomial expression
structure Monomial (α : Type*) :=
  (coeff : ℤ)
  (vars : List (α × ℕ))

-- Define a function to check if two monomials have the same type
def same_type {α : Type*} (m1 m2 : Monomial α) : Prop :=
  m1.vars = m2.vars

-- Define the reference monomial -3a²b
def reference : Monomial Char :=
  ⟨-3, [('a', 2), ('b', 1)]⟩

-- Define the given expressions
def expr1 : Monomial Char := ⟨-3, [('a', 1), ('b', 2)]⟩  -- -3ab²
def expr2 : Monomial Char := ⟨-1, [('b', 1), ('a', 2)]⟩  -- -ba²
def expr3 : Monomial Char := ⟨2, [('a', 1), ('b', 2)]⟩   -- 2ab²
def expr4 : Monomial Char := ⟨2, [('a', 3), ('b', 1)]⟩   -- 2a³b

-- Theorem to prove
theorem only_expr2_same_type_as_reference :
  (¬ same_type reference expr1) ∧
  (same_type reference expr2) ∧
  (¬ same_type reference expr3) ∧
  (¬ same_type reference expr4) :=
sorry

end only_expr2_same_type_as_reference_l541_54137


namespace sqrt5_and_sequences_l541_54121

-- Define p-arithmetic
structure PArithmetic where
  p : ℕ
  -- Add more properties as needed

-- Define the concept of "extracting √5"
def can_extract_sqrt5 (pa : PArithmetic) : Prop := sorry

-- Define a sequence type
def Sequence (α : Type) := ℕ → α

-- Define properties for Fibonacci and geometric sequences
def is_fibonacci {α : Type} [Add α] (seq : Sequence α) : Prop := 
  ∀ n, seq (n + 2) = seq (n + 1) + seq n

def is_geometric {α : Type} [Mul α] (seq : Sequence α) : Prop := 
  ∃ r, ∀ n, seq (n + 1) = r * seq n

-- Main theorem
theorem sqrt5_and_sequences (pa : PArithmetic) :
  (¬ can_extract_sqrt5 pa → 
    ¬ ∃ (seq : Sequence ℚ), is_fibonacci seq ∧ is_geometric seq) ∧
  (can_extract_sqrt5 pa → 
    (∃ (seq : Sequence ℚ), is_fibonacci seq ∧ is_geometric seq) ∧
    (∀ (fib : Sequence ℚ), is_fibonacci fib → 
      ∃ (seq1 seq2 : Sequence ℚ), 
        is_fibonacci seq1 ∧ is_geometric seq1 ∧
        is_fibonacci seq2 ∧ is_geometric seq2 ∧
        (∀ n, fib n = seq1 n + seq2 n))) :=
by sorry

end sqrt5_and_sequences_l541_54121


namespace updated_p_value_l541_54187

/-- Given the equation fp - w = 20000, where f = 10 and w = 10 + 250i, prove that p = 2001 + 25i -/
theorem updated_p_value (f w p : ℂ) : 
  f = 10 → w = 10 + 250 * Complex.I → f * p - w = 20000 → p = 2001 + 25 * Complex.I := by
  sorry

end updated_p_value_l541_54187


namespace subset_condition_1_subset_condition_2_l541_54147

-- Define the sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 7}
def B (a : ℝ) : Set ℝ := {x | 3 - 2*a ≤ x ∧ x ≤ 2*a - 5}

-- Theorem for part 1
theorem subset_condition_1 (a : ℝ) : A ⊆ B a ↔ a ≥ 6 := by sorry

-- Theorem for part 2
theorem subset_condition_2 (a : ℝ) : B a ⊆ A ↔ 2 ≤ a ∧ a ≤ 3 := by sorry

end subset_condition_1_subset_condition_2_l541_54147


namespace circle_tangent_to_parabola_directrix_l541_54168

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -1

-- Define a circle with center at the origin
def circle_at_origin (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Theorem statement
theorem circle_tangent_to_parabola_directrix :
  ∀ (x y r : ℝ),
  (∃ (x_d : ℝ), directrix x_d ∧ 
    (∀ (x_p y_p : ℝ), parabola x_p y_p → x_p ≥ x_d) ∧
    (∃ (x_t y_t : ℝ), parabola x_t y_t ∧ x_t = x_d ∧ 
      circle_at_origin x_t y_t r)) →
  circle_at_origin x y 1 :=
sorry

end circle_tangent_to_parabola_directrix_l541_54168


namespace no_natural_solution_l541_54128

theorem no_natural_solution (x y z : ℕ) : 
  (x : ℚ) / y + (y : ℚ) / z + (z : ℚ) / x ≠ 1 := by
  sorry

end no_natural_solution_l541_54128


namespace rhombus_common_area_l541_54167

/-- Represents a rhombus with given diagonal lengths -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculates the area of the common part of two rhombuses -/
def commonArea (r : Rhombus) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem: The area of the common part of two rhombuses is 9.6 cm² -/
theorem rhombus_common_area :
  let r := Rhombus.mk 4 6
  commonArea r = 9.6 := by
  sorry

end rhombus_common_area_l541_54167


namespace cube_decomposition_smallest_l541_54158

theorem cube_decomposition_smallest (m : ℕ) (h1 : m ≥ 2) : 
  (m^2 - m + 1 = 73) → m = 9 := by
  sorry

end cube_decomposition_smallest_l541_54158


namespace book_reading_time_l541_54198

theorem book_reading_time (pages_per_book : ℕ) (pages_per_day : ℕ) (days_to_finish : ℕ) : 
  pages_per_book = 249 → pages_per_day = 83 → days_to_finish = 3 →
  pages_per_book = days_to_finish * pages_per_day :=
by
  sorry

end book_reading_time_l541_54198


namespace frog_jump_probability_l541_54188

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the rectangular garden -/
structure Garden where
  bottomLeft : Point
  topRight : Point

/-- Represents the possible jump directions -/
inductive JumpDirection
  | Up
  | Down
  | Left
  | Right
  | NorthEast
  | NorthWest
  | SouthEast
  | SouthWest

/-- Represents the possible jump lengths -/
inductive JumpLength
  | One
  | Two

/-- Function to calculate the probability of ending on a horizontal side -/
def probabilityHorizontalEnd (garden : Garden) (start : Point) : ℝ :=
  sorry

/-- Theorem stating the probability of ending on a horizontal side is 0.4 -/
theorem frog_jump_probability (garden : Garden) (start : Point) :
  garden.bottomLeft = ⟨1, 1⟩ ∧
  garden.topRight = ⟨5, 6⟩ ∧
  start = ⟨2, 3⟩ →
  probabilityHorizontalEnd garden start = 0.4 :=
sorry


end frog_jump_probability_l541_54188


namespace expression_value_l541_54155

theorem expression_value (x y : ℝ) (h : x - 3*y = 4) :
  (x - 3*y)^2 + 2*x - 6*y - 10 = 14 := by
  sorry

end expression_value_l541_54155


namespace find_divisor_l541_54195

theorem find_divisor (dividend : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 76)
  (h2 : remainder = 8)
  (h3 : quotient = 4)
  : ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 17 :=
by sorry

end find_divisor_l541_54195


namespace ace_diamond_king_probability_l541_54171

-- Define the structure of a standard deck
def StandardDeck : Type := Fin 52

-- Define the properties of cards
def isAce : StandardDeck → Prop := sorry
def isDiamond : StandardDeck → Prop := sorry
def isKing : StandardDeck → Prop := sorry

-- Define the draw function
def draw : ℕ → (StandardDeck → Prop) → ℚ := sorry

-- Theorem statement
theorem ace_diamond_king_probability :
  draw 1 isAce * draw 2 isDiamond * draw 3 isKing = 1 / 663 := by sorry

end ace_diamond_king_probability_l541_54171


namespace existence_of_prime_q_l541_54189

theorem existence_of_prime_q (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, n > 0 → ¬(q ∣ (n^p - p)) :=
by sorry

end existence_of_prime_q_l541_54189


namespace total_gum_pieces_l541_54116

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 9

/-- The number of gum pieces in each package -/
def pieces_per_package : ℕ := 15

/-- Theorem: The total number of gum pieces Robin has is 135 -/
theorem total_gum_pieces : num_packages * pieces_per_package = 135 := by
  sorry

end total_gum_pieces_l541_54116


namespace min_width_rectangle_l541_54127

/-- Given a rectangular area with length 20 ft longer than the width,
    and an area of at least 150 sq. ft, the minimum possible width is 10 ft. -/
theorem min_width_rectangle (w : ℝ) (h1 : w > 0) : 
  w * (w + 20) ≥ 150 → w ≥ 10 :=
by sorry

end min_width_rectangle_l541_54127


namespace special_function_value_l541_54124

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2))

/-- The main theorem stating that f(2009) = 2 for any function satisfying SpecialFunction -/
theorem special_function_value :
    ∀ f : ℝ → ℝ, SpecialFunction f → f 2009 = 2 := by
  sorry

end special_function_value_l541_54124


namespace percentage_of_red_non_honda_cars_l541_54166

theorem percentage_of_red_non_honda_cars 
  (total_cars : ℕ) 
  (honda_cars : ℕ) 
  (honda_red_ratio : ℚ) 
  (total_red_ratio : ℚ) 
  (h1 : total_cars = 9000)
  (h2 : honda_cars = 5000)
  (h3 : honda_red_ratio = 90 / 100)
  (h4 : total_red_ratio = 60 / 100)
  : (↑(total_cars * total_red_ratio - honda_cars * honda_red_ratio) / ↑(total_cars - honda_cars) : ℚ) = 225 / 1000 := by
  sorry

end percentage_of_red_non_honda_cars_l541_54166


namespace distance_between_foci_l541_54163

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y + 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

-- Define the foci
def focus1 : ℝ × ℝ := (4, -5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem statement
theorem distance_between_foci :
  ∃ (x y : ℝ), ellipse_equation x y →
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 74 :=
sorry

end distance_between_foci_l541_54163


namespace no_solution_for_modified_problem_l541_54180

theorem no_solution_for_modified_problem (r : ℝ) : 
  ¬∃ (a h : ℝ), 
    (0 < r) ∧ 
    (0 < a) ∧ (a ≤ 2*r) ∧ 
    (0 < h) ∧ (h < 2*r) ∧ 
    (a + h = 2*Real.pi*r) := by
  sorry


end no_solution_for_modified_problem_l541_54180


namespace subtraction_result_l541_54133

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_valid_arrangement (a b c d e f g h i j : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧ is_valid_digit e ∧
  is_valid_digit f ∧ is_valid_digit g ∧ is_valid_digit h ∧ is_valid_digit i ∧ is_valid_digit j ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

theorem subtraction_result 
  (a b c d e f g h i j : ℕ) 
  (h1 : is_valid_arrangement a b c d e f g h i j)
  (h2 : a = 6)
  (h3 : b = 1) :
  61000 + c * 1000 + d * 100 + e * 10 + f - (g * 10000 + h * 1000 + i * 100 + j * 10 + a) = 59387 := by
  sorry

end subtraction_result_l541_54133


namespace preston_high_teachers_l541_54149

/-- Represents the number of students in Preston High School -/
def total_students : ℕ := 1500

/-- Represents the number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- Represents the number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- Represents the number of students in each class -/
def students_per_class : ℕ := 30

/-- Calculates the number of teachers at Preston High School -/
def number_of_teachers : ℕ :=
  (total_students * classes_per_student) / (students_per_class * classes_per_teacher)

/-- Theorem stating that the number of teachers at Preston High School is 60 -/
theorem preston_high_teachers :
  number_of_teachers = 60 := by sorry

end preston_high_teachers_l541_54149


namespace function_value_at_four_l541_54184

/-- Given a function f where f(2x) = 3x^2 + 1 for all x, prove that f(4) = 13 -/
theorem function_value_at_four (f : ℝ → ℝ) (h : ∀ x, f (2 * x) = 3 * x^2 + 1) : f 4 = 13 := by
  sorry

end function_value_at_four_l541_54184


namespace expand_product_l541_54114

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 := by
  sorry

end expand_product_l541_54114


namespace smallest_positive_integer_2016m_45000n_l541_54177

theorem smallest_positive_integer_2016m_45000n :
  ∃ (k : ℕ), k > 0 ∧
  (∃ (m n : ℤ), k = 2016 * m + 45000 * n) ∧
  (∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 2016 * x + 45000 * y) → j ≥ k) ∧
  k = 24 := by
sorry

end smallest_positive_integer_2016m_45000n_l541_54177


namespace stratified_sample_theorem_l541_54152

/-- Represents the number of male students in a stratified sample -/
def male_students_in_sample (total_male : ℕ) (total_female : ℕ) (sample_size : ℕ) : ℕ :=
  (total_male * sample_size) / (total_male + total_female)

/-- Theorem: In a school with 560 male students and 420 female students,
    a stratified sample of 140 students will contain 80 male students -/
theorem stratified_sample_theorem :
  male_students_in_sample 560 420 140 = 80 := by
  sorry

end stratified_sample_theorem_l541_54152


namespace solution_l541_54148

def problem (x y a : ℚ) : Prop :=
  (1 / 5) * x = (5 / 8) * y ∧
  y = 40 ∧
  x + a = 4 * y

theorem solution : ∃ x y a : ℚ, problem x y a ∧ a = 35 := by
  sorry

end solution_l541_54148


namespace sqrt_expression_equality_l541_54101

theorem sqrt_expression_equality : 
  (Real.sqrt 3 + Real.sqrt 2 - 1) * (Real.sqrt 3 - Real.sqrt 2 + 1) = 2 * Real.sqrt 2 := by
  sorry

end sqrt_expression_equality_l541_54101


namespace equal_volumes_l541_54154

-- Define the tetrahedrons
structure Tetrahedron :=
  (a b c d e f : ℝ)

-- Define the volumes of the tetrahedrons
def volume (t : Tetrahedron) : ℝ := sorry

-- Define the specific tetrahedrons
def ABCD : Tetrahedron :=
  { a := 13, b := 5, c := 12, d := 13, e := 6, f := 5 }

def EFGH : Tetrahedron :=
  { a := 13, b := 13, c := 8, d := 5, e := 12, f := 5 }

-- Theorem statement
theorem equal_volumes : volume ABCD = volume EFGH := by
  sorry

end equal_volumes_l541_54154


namespace series_sum_equals_half_l541_54194

/-- The sum of the series Σ(2^n / (3^(2^n) + 1)) from n = 0 to infinity is equal to 1/2 -/
theorem series_sum_equals_half :
  ∑' n : ℕ, (2 : ℝ)^n / (3^(2^n) + 1) = 1/2 := by sorry

end series_sum_equals_half_l541_54194


namespace cyclic_sum_inequality_l541_54199

theorem cyclic_sum_inequality (k : ℕ) (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + y + z = 1) : 
  (x^(k+2) / (x^(k+1) + y^k + z^k)) + 
  (y^(k+2) / (y^(k+1) + z^k + x^k)) + 
  (z^(k+2) / (z^(k+1) + x^k + y^k)) ≥ 1/7 := by
  sorry

end cyclic_sum_inequality_l541_54199


namespace original_number_is_35_l541_54197

-- Define a two-digit number type
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

-- Define functions to get tens and units digits
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

-- Define a function to swap digits
def swap_digits (n : TwoDigitNumber) : TwoDigitNumber :=
  ⟨10 * (units_digit n) + (tens_digit n), by sorry⟩

-- Theorem statement
theorem original_number_is_35 (n : TwoDigitNumber) 
  (h1 : tens_digit n + units_digit n = 8)
  (h2 : (swap_digits n).val = n.val + 18) : 
  n.val = 35 := by sorry

end original_number_is_35_l541_54197


namespace sum_of_four_consecutive_even_integers_l541_54125

theorem sum_of_four_consecutive_even_integers :
  ∀ a : ℤ,
  (∃ b c d : ℤ, 
    b = a + 2 ∧ 
    c = a + 4 ∧ 
    d = a + 6 ∧ 
    a + d = 136) →
  a + (a + 2) + (a + 4) + (a + 6) = 272 :=
by sorry

end sum_of_four_consecutive_even_integers_l541_54125
