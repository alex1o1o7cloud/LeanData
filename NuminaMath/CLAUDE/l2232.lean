import Mathlib

namespace NUMINAMATH_CALUDE_product_selection_proof_l2232_223299

def total_products : ℕ := 100
def qualified_products : ℕ := 98
def defective_products : ℕ := 2
def selected_products : ℕ := 3

theorem product_selection_proof :
  (Nat.choose total_products selected_products = 161700) ∧
  (Nat.choose defective_products 1 * Nat.choose qualified_products 2 = 9506) ∧
  (Nat.choose total_products selected_products - Nat.choose qualified_products selected_products = 9604) :=
by sorry

end NUMINAMATH_CALUDE_product_selection_proof_l2232_223299


namespace NUMINAMATH_CALUDE_platform_length_l2232_223254

/-- Given a train of length 300 m that crosses a platform in 39 seconds
    and a signal pole in 12 seconds, the length of the platform is 675 m. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 39)
  (h3 : pole_crossing_time = 12) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 675 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2232_223254


namespace NUMINAMATH_CALUDE_david_scott_age_difference_l2232_223234

/-- Represents the ages of the three sons -/
structure Ages where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.david > ages.scott ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 11 + 3

/-- The theorem stating that David is 8 years older than Scott -/
theorem david_scott_age_difference (ages : Ages) : 
  problem_conditions ages → ages.david - ages.scott = 8 := by
  sorry

end NUMINAMATH_CALUDE_david_scott_age_difference_l2232_223234


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2232_223247

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem states that for a geometric sequence satisfying given conditions, 
    the sum of the 5th and 6th terms equals 48. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_sum1 : a 1 + a 2 = 3)
  (h_sum2 : a 3 + a 4 = 12) :
  a 5 + a 6 = 48 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l2232_223247


namespace NUMINAMATH_CALUDE_subtract_fractions_l2232_223239

theorem subtract_fractions : (3 : ℚ) / 4 - (1 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l2232_223239


namespace NUMINAMATH_CALUDE_bake_sale_donation_l2232_223252

/-- Calculates the total donation to the homeless shelter from a bake sale fundraiser --/
theorem bake_sale_donation (total_earnings : ℕ) (ingredient_cost : ℕ) (personal_donation : ℕ) : 
  total_earnings = 400 →
  ingredient_cost = 100 →
  personal_donation = 10 →
  ((total_earnings - ingredient_cost) / 2 + personal_donation : ℕ) = 160 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_donation_l2232_223252


namespace NUMINAMATH_CALUDE_vector_collinearity_l2232_223237

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 5)
def c : ℝ → ℝ × ℝ := λ x => (x, 1)

-- Define collinearity
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem vector_collinearity (x : ℝ) :
  collinear (2 * a - b) (c x) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2232_223237


namespace NUMINAMATH_CALUDE_smallest_integer_y_l2232_223253

theorem smallest_integer_y : ∀ y : ℤ, (5 : ℚ) / 8 < (y - 3 : ℚ) / 19 → y ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l2232_223253


namespace NUMINAMATH_CALUDE_inequality_proof_l2232_223243

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) : 
  a * b ≤ 1/8 ∧ 1/a + 2/b ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2232_223243


namespace NUMINAMATH_CALUDE_find_B_value_l2232_223287

theorem find_B_value (A B : Nat) (h1 : A ≤ 9) (h2 : B ≤ 9) 
  (h3 : 32 + A * 100 + 70 + B = 705) : B = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_B_value_l2232_223287


namespace NUMINAMATH_CALUDE_apples_ratio_proof_l2232_223251

theorem apples_ratio_proof (jim_apples jane_apples jerry_apples : ℕ) 
  (h1 : jim_apples = 20)
  (h2 : jane_apples = 60)
  (h3 : jerry_apples = 40) :
  (jim_apples + jane_apples + jerry_apples) / 3 / jim_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apples_ratio_proof_l2232_223251


namespace NUMINAMATH_CALUDE_spade_nested_operation_l2232_223279

-- Define the spade operation
def spade (a b : ℤ) : ℤ := |a - b|

-- Theorem statement
theorem spade_nested_operation : spade 3 (spade 5 (spade 8 12)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_spade_nested_operation_l2232_223279


namespace NUMINAMATH_CALUDE_max_sum_of_counts_l2232_223285

/-- Represents the color of a card -/
inductive CardColor
  | White
  | Black
  | Red

/-- Represents a stack of cards -/
structure CardStack :=
  (cards : List CardColor)
  (white_count : Nat)
  (black_count : Nat)
  (red_count : Nat)

/-- Calculates the sum of counts for a given card stack -/
def calculate_sum (stack : CardStack) : Nat :=
  sorry

/-- Theorem stating the maximum possible sum of counts -/
theorem max_sum_of_counts (stack : CardStack) 
  (h1 : stack.cards.length = 300)
  (h2 : stack.white_count = 100)
  (h3 : stack.black_count = 100)
  (h4 : stack.red_count = 100) :
  (∀ s : CardStack, calculate_sum s ≤ calculate_sum stack) →
  calculate_sum stack = 20000 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_counts_l2232_223285


namespace NUMINAMATH_CALUDE_pacos_marble_purchase_l2232_223212

theorem pacos_marble_purchase : 
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_pacos_marble_purchase_l2232_223212


namespace NUMINAMATH_CALUDE_danielle_popsicle_sticks_l2232_223261

/-- Calculates the number of remaining popsicle sticks after making popsicles -/
def remaining_popsicle_sticks (total_money : ℕ) (mold_cost : ℕ) (stick_pack_cost : ℕ) 
  (stick_pack_size : ℕ) (juice_cost : ℕ) (popsicles_per_juice : ℕ) : ℕ :=
  let remaining_money := total_money - mold_cost - stick_pack_cost
  let juice_bottles := remaining_money / juice_cost
  let popsicles_made := juice_bottles * popsicles_per_juice
  stick_pack_size - popsicles_made

/-- Proves that Danielle will be left with 40 popsicle sticks -/
theorem danielle_popsicle_sticks : 
  remaining_popsicle_sticks 10 3 1 100 2 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_danielle_popsicle_sticks_l2232_223261


namespace NUMINAMATH_CALUDE_price_increase_percentage_l2232_223277

theorem price_increase_percentage (original_price : ℝ) (increase_rate : ℝ) : 
  original_price = 200 →
  increase_rate = 0.1 →
  let new_price := original_price * (1 + increase_rate)
  (new_price - original_price) / original_price = increase_rate :=
by sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l2232_223277


namespace NUMINAMATH_CALUDE_bunny_burrow_exits_l2232_223231

/-- Represents the total number of bunnies in the community -/
def total_bunnies : ℕ := 100

/-- Represents the number of bunnies in Group A -/
def group_a_bunnies : ℕ := 40

/-- Represents the number of bunnies in Group B -/
def group_b_bunnies : ℕ := 30

/-- Represents the number of bunnies in Group C -/
def group_c_bunnies : ℕ := 30

/-- Represents how many times a bunny in Group A comes out per minute -/
def group_a_frequency : ℚ := 3

/-- Represents how many times a bunny in Group B comes out per minute -/
def group_b_frequency : ℚ := 5 / 2

/-- Represents how many times a bunny in Group C comes out per minute -/
def group_c_frequency : ℚ := 8 / 5

/-- Represents the reduction factor in burrow-exiting behavior after environmental change -/
def reduction_factor : ℚ := 1 / 2

/-- Represents the number of weeks before environmental change -/
def weeks_before_change : ℕ := 1

/-- Represents the number of weeks after environmental change -/
def weeks_after_change : ℕ := 2

/-- Represents the total number of weeks -/
def total_weeks : ℕ := 3

/-- Represents the number of minutes in a day -/
def minutes_per_day : ℕ := 1440

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating that the combined number of times all bunnies come out during 3 weeks is 4,897,920 -/
theorem bunny_burrow_exits : 
  (group_a_bunnies * group_a_frequency + 
   group_b_bunnies * group_b_frequency + 
   group_c_bunnies * group_c_frequency) * 
  minutes_per_day * days_per_week * weeks_before_change +
  (group_a_bunnies * group_a_frequency + 
   group_b_bunnies * group_b_frequency + 
   group_c_bunnies * group_c_frequency) * 
  reduction_factor * 
  minutes_per_day * days_per_week * weeks_after_change = 4897920 := by
  sorry

end NUMINAMATH_CALUDE_bunny_burrow_exits_l2232_223231


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2232_223265

def tshirt_cost : ℝ := 9.95
def number_of_tshirts : ℕ := 20

theorem total_cost_calculation : 
  tshirt_cost * (number_of_tshirts : ℝ) = 199.00 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2232_223265


namespace NUMINAMATH_CALUDE_machine_production_rate_l2232_223291

/-- Given an industrial machine that made 8 shirts in 4 minutes today,
    prove that it can make 2 shirts per minute. -/
theorem machine_production_rate (shirts_today : ℕ) (minutes_today : ℕ) 
  (h1 : shirts_today = 8) (h2 : minutes_today = 4) :
  shirts_today / minutes_today = 2 := by
sorry

end NUMINAMATH_CALUDE_machine_production_rate_l2232_223291


namespace NUMINAMATH_CALUDE_circle_equation_theorem_l2232_223201

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the coefficients of a circle equation -/
structure CircleCoefficients where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Checks if a point lies on a circle with given coefficients -/
def pointOnCircle (p : Point) (c : CircleCoefficients) : Prop :=
  p.x^2 + p.y^2 + c.D * p.x + c.E * p.y + c.F = 0

/-- The theorem stating that the given equation represents the circle passing through the specified points -/
theorem circle_equation_theorem (p1 p2 p3 : Point) : 
  p1 = ⟨0, 0⟩ → 
  p2 = ⟨4, 0⟩ → 
  p3 = ⟨-1, 1⟩ → 
  ∃ (c : CircleCoefficients), 
    c.D = -4 ∧ c.E = -6 ∧ c.F = 0 ∧ 
    pointOnCircle p1 c ∧ 
    pointOnCircle p2 c ∧ 
    pointOnCircle p3 c :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_theorem_l2232_223201


namespace NUMINAMATH_CALUDE_largest_y_value_l2232_223292

theorem largest_y_value (y : ℝ) : 
  (y / 3 + 2 / (3 * y) = 1) → y ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_value_l2232_223292


namespace NUMINAMATH_CALUDE_clerks_count_l2232_223223

/-- Represents the grocery store employee structure and salaries --/
structure GroceryStore where
  manager_salary : ℕ
  clerk_salary : ℕ
  num_managers : ℕ
  total_salary : ℕ

/-- Calculates the number of clerks in the grocery store --/
def num_clerks (store : GroceryStore) : ℕ :=
  (store.total_salary - store.manager_salary * store.num_managers) / store.clerk_salary

/-- Theorem stating that the number of clerks is 3 given the conditions --/
theorem clerks_count (store : GroceryStore) 
    (h1 : store.manager_salary = 5)
    (h2 : store.clerk_salary = 2)
    (h3 : store.num_managers = 2)
    (h4 : store.total_salary = 16) : 
  num_clerks store = 3 := by
  sorry

end NUMINAMATH_CALUDE_clerks_count_l2232_223223


namespace NUMINAMATH_CALUDE_yellow_flags_count_l2232_223206

/-- Represents the number of yellow flags in a cycle -/
def yellow_per_cycle : ℕ := 2

/-- Represents the length of the repeating cycle -/
def cycle_length : ℕ := 5

/-- Represents the total number of flags we're considering -/
def total_flags : ℕ := 200

/-- Theorem: The number of yellow flags in the first 200 flags is 80 -/
theorem yellow_flags_count : 
  (total_flags / cycle_length) * yellow_per_cycle = 80 := by
sorry

end NUMINAMATH_CALUDE_yellow_flags_count_l2232_223206


namespace NUMINAMATH_CALUDE_marcos_dads_strawberries_strawberry_problem_l2232_223217

theorem marcos_dads_strawberries (initial_total : ℕ) (dads_extra : ℕ) (marcos_final : ℕ) : ℕ :=
  let dads_initial := initial_total - (marcos_final - dads_extra)
  dads_initial + dads_extra

theorem strawberry_problem : 
  marcos_dads_strawberries 22 30 36 = 46 := by
  sorry

end NUMINAMATH_CALUDE_marcos_dads_strawberries_strawberry_problem_l2232_223217


namespace NUMINAMATH_CALUDE_subtraction_error_correction_l2232_223200

theorem subtraction_error_correction (x y : ℕ) 
  (h1 : x - y = 8008)
  (h2 : x - 10 * y = 88) :
  x = 8888 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_error_correction_l2232_223200


namespace NUMINAMATH_CALUDE_vaccine_effective_l2232_223250

/-- Represents the contingency table for vaccine effectiveness study -/
structure VaccineStudy where
  total_mice : ℕ
  infected_mice : ℕ
  not_infected_mice : ℕ
  prob_infected_not_vaccinated : ℚ

/-- Calculates the chi-square statistic for the vaccine study -/
def chi_square (study : VaccineStudy) : ℚ :=
  let a := study.not_infected_mice - (study.total_mice / 2 - study.infected_mice * study.prob_infected_not_vaccinated)
  let b := study.not_infected_mice - a
  let c := study.total_mice / 2 - study.infected_mice * study.prob_infected_not_vaccinated
  let d := study.infected_mice - c
  let n := study.total_mice
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for 95% confidence in the chi-square test -/
def chi_square_critical : ℚ := 3841 / 1000

/-- Theorem stating that the vaccine is effective with 95% confidence -/
theorem vaccine_effective (study : VaccineStudy) 
  (h1 : study.total_mice = 200)
  (h2 : study.infected_mice = 100)
  (h3 : study.not_infected_mice = 100)
  (h4 : study.prob_infected_not_vaccinated = 3/5) :
  chi_square study > chi_square_critical := by
  sorry

end NUMINAMATH_CALUDE_vaccine_effective_l2232_223250


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_l2232_223244

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the intersection of planes
variable (intersection : Plane → Plane → Line)

-- Theorem statement
theorem line_parallel_to_intersection
  (a b : Line) (α β : Plane)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : parallel_line_plane a α)
  (h4 : parallel_line_plane a β)
  (h5 : intersection α β = b) :
  parallel_line_line a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_l2232_223244


namespace NUMINAMATH_CALUDE_p_recurrence_l2232_223209

/-- The probability of having a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℚ :=
  sorry

/-- The recurrence relation for p(n,k) -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end NUMINAMATH_CALUDE_p_recurrence_l2232_223209


namespace NUMINAMATH_CALUDE_min_bills_for_payment_l2232_223266

/-- Represents the number of bills of each denomination --/
structure Bills :=
  (tens : ℕ)
  (fives : ℕ)
  (ones : ℕ)

/-- Calculates the total value of the bills --/
def billsValue (b : Bills) : ℕ :=
  10 * b.tens + 5 * b.fives + b.ones

/-- Checks if a given number of bills is valid for the payment --/
def isValidPayment (b : Bills) (amount : ℕ) : Prop :=
  b.tens ≤ 13 ∧ b.fives ≤ 11 ∧ b.ones ≤ 17 ∧ billsValue b = amount

/-- Counts the total number of bills --/
def totalBills (b : Bills) : ℕ :=
  b.tens + b.fives + b.ones

/-- The main theorem stating that the minimum number of bills required is 16 --/
theorem min_bills_for_payment :
  ∃ (b : Bills), isValidPayment b 128 ∧
  ∀ (b' : Bills), isValidPayment b' 128 → totalBills b ≤ totalBills b' :=
by sorry

end NUMINAMATH_CALUDE_min_bills_for_payment_l2232_223266


namespace NUMINAMATH_CALUDE_prob_A_squared_zero_correct_l2232_223257

/-- Probability that A² = O for an n × n matrix A with exactly two 1's -/
def prob_A_squared_zero (n : ℕ) : ℚ :=
  if n < 2 then 0
  else (n - 1) * (n - 2) / (n * (n + 1))

/-- Theorem stating the probability that A² = O for the given conditions -/
theorem prob_A_squared_zero_correct (n : ℕ) (h : n ≥ 2) :
  prob_A_squared_zero n = (n - 1) * (n - 2) / (n * (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_prob_A_squared_zero_correct_l2232_223257


namespace NUMINAMATH_CALUDE_successive_discounts_l2232_223235

theorem successive_discounts (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  let price_after_discount1 := initial_price * (1 - discount1)
  let final_price := price_after_discount1 * (1 - discount2)
  discount1 = 0.1 ∧ discount2 = 0.2 →
  final_price / initial_price = 0.72 := by
sorry

end NUMINAMATH_CALUDE_successive_discounts_l2232_223235


namespace NUMINAMATH_CALUDE_transform_equation_l2232_223248

theorem transform_equation (x m : ℝ) : 
  (x^2 + 4*x = m) ∧ ((x + 2)^2 = 5) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_transform_equation_l2232_223248


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2232_223268

theorem quadratic_equation_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁^2 + k*x₁ + k - 1 = 0 ∧ x₂^2 + k*x₂ + k - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2232_223268


namespace NUMINAMATH_CALUDE_factor_expression_l2232_223283

theorem factor_expression (x : ℝ) : 5*x*(x+2) + 11*(x+2) = (x+2)*(5*x+11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2232_223283


namespace NUMINAMATH_CALUDE_jerry_age_l2232_223290

/-- Given that Mickey's age is 17 years and Mickey's age is 3 years less than 250% of Jerry's age,
    prove that Jerry's age is 8 years. -/
theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 17 → 
  mickey_age = (250 * jerry_age) / 100 - 3 → 
  jerry_age = 8 :=
by sorry

end NUMINAMATH_CALUDE_jerry_age_l2232_223290


namespace NUMINAMATH_CALUDE_wednesday_earnings_l2232_223226

/-- Represents the earnings from selling cabbage over three days -/
structure CabbageEarnings where
  wednesday : ℝ
  friday : ℝ
  today : ℝ
  total_kg : ℝ
  price_per_kg : ℝ

/-- Theorem stating that given the conditions, Johannes earned $30 on Wednesday -/
theorem wednesday_earnings (e : CabbageEarnings) 
  (h1 : e.friday = 24)
  (h2 : e.today = 42)
  (h3 : e.total_kg = 48)
  (h4 : e.price_per_kg = 2)
  (h5 : e.wednesday + e.friday + e.today = e.total_kg * e.price_per_kg) :
  e.wednesday = 30 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_earnings_l2232_223226


namespace NUMINAMATH_CALUDE_simplify_expression_l2232_223213

theorem simplify_expression : 
  (2 * (10^12)) / ((4 * (10^5)) - (1 * (10^4))) = 5.1282 * (10^6) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2232_223213


namespace NUMINAMATH_CALUDE_gcd_of_324_243_270_l2232_223210

theorem gcd_of_324_243_270 : Nat.gcd 324 (Nat.gcd 243 270) = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_324_243_270_l2232_223210


namespace NUMINAMATH_CALUDE_bryan_pushups_l2232_223228

/-- The number of push-up sets Bryan does -/
def total_sets : ℕ := 15

/-- The number of push-ups Bryan intends to do in each set -/
def pushups_per_set : ℕ := 18

/-- The number of push-ups Bryan doesn't do in the last set due to exhaustion -/
def missed_pushups : ℕ := 12

/-- The actual number of push-ups Bryan does in the last set -/
def last_set_pushups : ℕ := pushups_per_set - missed_pushups

/-- The total number of push-ups Bryan does -/
def total_pushups : ℕ := (total_sets - 1) * pushups_per_set + last_set_pushups

theorem bryan_pushups : total_pushups = 258 := by
  sorry

end NUMINAMATH_CALUDE_bryan_pushups_l2232_223228


namespace NUMINAMATH_CALUDE_max_min_powers_l2232_223295

theorem max_min_powers (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  let M := max (max (a^a) (a^b)) (max (b^a) (b^b))
  let m := min (min (a^a) (a^b)) (min (b^a) (b^b))
  M = b^a ∧ m = a^b := by
sorry

end NUMINAMATH_CALUDE_max_min_powers_l2232_223295


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l2232_223233

def is_valid_representation (n : ℕ) (a b : ℕ) : Prop :=
  a > 2 ∧ b > 2 ∧ n = 1 * a + 3 ∧ n = 3 * b + 1

theorem smallest_dual_base_representation :
  ∃ (n : ℕ), is_valid_representation n 7 3 ∧
  ∀ (m : ℕ) (a b : ℕ), is_valid_representation m a b → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l2232_223233


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l2232_223219

theorem complex_purely_imaginary (z : ℂ) : 
  (∃ y : ℝ, z = y * I) →  -- z is purely imaginary
  (∃ w : ℝ, (z + 2)^2 - 8*I = w * I) →  -- (z + 2)² - 8i is purely imaginary
  z = -2 * I :=  -- z = -2i
by sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l2232_223219


namespace NUMINAMATH_CALUDE_telephone_pole_height_l2232_223286

-- Define the problem parameters
def base_height : Real := 1
def cable_ground_distance : Real := 5
def leah_distance : Real := 4
def leah_height : Real := 1.8

-- Define the theorem
theorem telephone_pole_height :
  let total_ground_distance : Real := cable_ground_distance + base_height
  let remaining_distance : Real := total_ground_distance - leah_distance
  let pole_height : Real := (leah_height * total_ground_distance) / remaining_distance
  pole_height = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_telephone_pole_height_l2232_223286


namespace NUMINAMATH_CALUDE_smallest_value_l2232_223289

theorem smallest_value (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 < x^2 ∧ x^3 < 3*x ∧ x^3 < Real.sqrt x ∧ x^3 < 1/x := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_l2232_223289


namespace NUMINAMATH_CALUDE_bank_number_inconsistency_l2232_223203

-- Define the initial number of banks
def initial_banks : ℕ := 10

-- Define the splitting rule
def split_rule (n : ℕ) : ℕ := n + 7

-- Define the claimed number of banks
def claimed_banks : ℕ := 2023

-- Theorem stating the impossibility of reaching the claimed number of banks
theorem bank_number_inconsistency :
  ∀ n : ℕ, n % 7 = initial_banks % 7 → n ≠ claimed_banks :=
by
  sorry

end NUMINAMATH_CALUDE_bank_number_inconsistency_l2232_223203


namespace NUMINAMATH_CALUDE_card_collection_average_l2232_223241

-- Define the sum of integers from 1 to n
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of squares from 1 to n
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem card_collection_average (n : ℕ) : 
  (sum_of_squares n : ℚ) / (sum_to_n n : ℚ) = 5050 → n = 7575 := by
  sorry

end NUMINAMATH_CALUDE_card_collection_average_l2232_223241


namespace NUMINAMATH_CALUDE_radical_product_simplification_l2232_223262

theorem radical_product_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (98 * x) * Real.sqrt (18 * x) * Real.sqrt (50 * x) = 210 * x * Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l2232_223262


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2232_223293

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 - 7 * q^3 + 3 * q + 8) + (5 - 9 * q^3 + 4 * q^2 - 2 * q) =
  4 * q^4 - 16 * q^3 + 4 * q^2 + q + 13 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2232_223293


namespace NUMINAMATH_CALUDE_fraction_comparison_l2232_223281

theorem fraction_comparison (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 - y^2) / (x - y) > (x^2 + y^2) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2232_223281


namespace NUMINAMATH_CALUDE_min_distinct_lines_for_31_segments_l2232_223230

/-- Represents a non-self-intersecting open polyline on a plane -/
structure OpenPolyline where
  segments : ℕ
  non_self_intersecting : Bool
  consecutive_segments_not_collinear : Bool

/-- The minimum number of distinct lines needed to contain all segments of the polyline -/
def min_distinct_lines (p : OpenPolyline) : ℕ :=
  sorry

/-- Theorem stating the minimum number of distinct lines for a specific polyline -/
theorem min_distinct_lines_for_31_segments (p : OpenPolyline) :
  p.segments = 31 ∧ p.non_self_intersecting ∧ p.consecutive_segments_not_collinear →
  min_distinct_lines p = 9 :=
sorry

end NUMINAMATH_CALUDE_min_distinct_lines_for_31_segments_l2232_223230


namespace NUMINAMATH_CALUDE_cora_cookie_purchase_l2232_223264

/-- The number of cookies Cora purchased each day in April -/
def cookies_per_day : ℕ := 3

/-- The cost of each cookie in dollars -/
def cookie_cost : ℕ := 18

/-- The total amount Cora spent on cookies in April in dollars -/
def total_spent : ℕ := 1620

/-- The number of days in April -/
def days_in_april : ℕ := 30

/-- Theorem stating that Cora purchased 3 cookies each day in April -/
theorem cora_cookie_purchase :
  cookies_per_day * days_in_april * cookie_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_cora_cookie_purchase_l2232_223264


namespace NUMINAMATH_CALUDE_max_value_of_function_l2232_223274

theorem max_value_of_function (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  ∃ (max_y : ℝ), max_y = 5/2 ∧ 
  ∀ y : ℝ, y = 2^(2*x - 1) - 3 * 2^x + 5 → y ≤ max_y :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2232_223274


namespace NUMINAMATH_CALUDE_expression_equality_l2232_223296

theorem expression_equality : 784 + 2 * 28 * 7 + 49 = 1225 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l2232_223296


namespace NUMINAMATH_CALUDE_unattainable_y_value_l2232_223272

theorem unattainable_y_value (x : ℝ) (h : x ≠ -5/4) :
  ∀ y : ℝ, y = (2 - 3*x) / (4*x + 5) → y ≠ -3/4 := by
sorry

end NUMINAMATH_CALUDE_unattainable_y_value_l2232_223272


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2232_223204

theorem right_triangle_sides (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 156 →
  x * y / 2 = 1014 →
  x^2 + y^2 = z^2 →
  (x = 39 ∧ y = 52 ∧ z = 65) ∨ (x = 52 ∧ y = 39 ∧ z = 65) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l2232_223204


namespace NUMINAMATH_CALUDE_arithmetic_24_l2232_223211

def numbers : List ℕ := [8, 8, 8, 10]

inductive ArithExpr
  | Num : ℕ → ArithExpr
  | Add : ArithExpr → ArithExpr → ArithExpr
  | Sub : ArithExpr → ArithExpr → ArithExpr
  | Mul : ArithExpr → ArithExpr → ArithExpr
  | Div : ArithExpr → ArithExpr → ArithExpr

def eval : ArithExpr → ℕ
  | ArithExpr.Num n => n
  | ArithExpr.Add e1 e2 => eval e1 + eval e2
  | ArithExpr.Sub e1 e2 => eval e1 - eval e2
  | ArithExpr.Mul e1 e2 => eval e1 * eval e2
  | ArithExpr.Div e1 e2 => eval e1 / eval e2

def uses_all_numbers (expr : ArithExpr) (nums : List ℕ) : Prop := sorry

theorem arithmetic_24 : 
  ∃ (expr : ArithExpr), uses_all_numbers expr numbers ∧ eval expr = 24 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_24_l2232_223211


namespace NUMINAMATH_CALUDE_oplus_roots_l2232_223245

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a^2 - 5*a + 2*b

-- State the theorem
theorem oplus_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, oplus x 3 = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_oplus_roots_l2232_223245


namespace NUMINAMATH_CALUDE_two_heart_three_l2232_223276

/-- The ♥ operation defined as a ♥ b = ab³ - 2b + 3 -/
def heart (a b : ℝ) : ℝ := a * b^3 - 2*b + 3

/-- Theorem stating that 2 ♥ 3 = 51 -/
theorem two_heart_three : heart 2 3 = 51 := by
  sorry

end NUMINAMATH_CALUDE_two_heart_three_l2232_223276


namespace NUMINAMATH_CALUDE_jason_pepper_spray_l2232_223267

def total_animals (raccoons : ℕ) (squirrel_multiplier : ℕ) : ℕ :=
  raccoons + raccoons * squirrel_multiplier

theorem jason_pepper_spray :
  total_animals 12 6 = 84 :=
by sorry

end NUMINAMATH_CALUDE_jason_pepper_spray_l2232_223267


namespace NUMINAMATH_CALUDE_complement_of_equal_sets_l2232_223215

def U : Set Nat := {1, 3}
def A : Set Nat := {1, 3}

theorem complement_of_equal_sets :
  (U \ A : Set Nat) = ∅ :=
sorry

end NUMINAMATH_CALUDE_complement_of_equal_sets_l2232_223215


namespace NUMINAMATH_CALUDE_compressor_stations_theorem_l2232_223240

/-- Represents the distances between three compressor stations -/
structure CompressorStations where
  x : ℝ  -- distance between first and second stations
  y : ℝ  -- distance between second and third stations
  z : ℝ  -- direct distance between first and third stations
  a : ℝ  -- parameter

/-- Conditions for the compressor stations arrangement -/
def valid_arrangement (s : CompressorStations) : Prop :=
  s.x > 0 ∧ s.y > 0 ∧ s.z > 0 ∧
  s.x + s.y = 4 * s.z ∧
  s.z + s.y = s.x + s.a ∧
  s.x + s.z = 85 ∧
  s.x + s.y > s.z ∧
  s.x + s.z > s.y ∧
  s.y + s.z > s.x

theorem compressor_stations_theorem :
  ∀ s : CompressorStations,
  valid_arrangement s →
  (0 < s.a ∧ s.a < 68) ∧
  (s.a = 5 → s.x = 60 ∧ s.y = 40 ∧ s.z = 25) :=
sorry

end NUMINAMATH_CALUDE_compressor_stations_theorem_l2232_223240


namespace NUMINAMATH_CALUDE_dino_expenses_l2232_223225

/-- Calculates Dino's monthly expenses based on his work hours, hourly rates, and remaining money --/
theorem dino_expenses (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (remaining : ℕ) : 
  hours1 = 20 → hours2 = 30 → hours3 = 5 →
  rate1 = 10 → rate2 = 20 → rate3 = 40 →
  remaining = 500 →
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - remaining = 500 := by
  sorry

#check dino_expenses

end NUMINAMATH_CALUDE_dino_expenses_l2232_223225


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2232_223259

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8| + 3

theorem sum_of_max_min_g : 
  ∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → g x ≤ max) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ g x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → min ≤ g x) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ g x = min) ∧
    max + min = 18 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2232_223259


namespace NUMINAMATH_CALUDE_max_strips_from_sheet_l2232_223288

/-- Represents a rectangular sheet of paper --/
structure Sheet where
  length : ℕ
  width : ℕ

/-- Represents a rectangular strip of paper --/
structure Strip where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of strips that can be cut from a sheet --/
def maxStrips (sheet : Sheet) (strip : Strip) : ℕ :=
  max
    ((sheet.length / strip.length) * (sheet.width / strip.width))
    ((sheet.length / strip.width) * (sheet.width / strip.length))

theorem max_strips_from_sheet :
  let sheet := Sheet.mk 14 11
  let strip := Strip.mk 4 1
  maxStrips sheet strip = 33 := by sorry

end NUMINAMATH_CALUDE_max_strips_from_sheet_l2232_223288


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2232_223208

theorem cubic_root_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) → (0 < b ∧ b < 1) → (0 < c ∧ c < 1) →
  a ≠ b → b ≠ c → a ≠ c →
  20 * a^3 - 34 * a^2 + 15 * a - 1 = 0 →
  20 * b^3 - 34 * b^2 + 15 * b - 1 = 0 →
  20 * c^3 - 34 * c^2 + 15 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1.3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2232_223208


namespace NUMINAMATH_CALUDE_new_student_info_is_unique_l2232_223270

-- Define the possible values for each attribute
inductive Surname
  | Ji | Zhang | Chen | Huang
deriving Repr, DecidableEq

inductive Gender
  | Male | Female
deriving Repr, DecidableEq

inductive Specialty
  | Singing | Dancing | Drawing
deriving Repr, DecidableEq

-- Define a structure for student information
structure StudentInfo where
  surname : Surname
  gender : Gender
  totalScore : Nat
  specialty : Specialty
deriving Repr

-- Define the information provided by each classmate
def classmate_A : StudentInfo := ⟨Surname.Ji, Gender.Male, 260, Specialty.Singing⟩
def classmate_B : StudentInfo := ⟨Surname.Zhang, Gender.Female, 220, Specialty.Dancing⟩
def classmate_C : StudentInfo := ⟨Surname.Chen, Gender.Male, 260, Specialty.Singing⟩
def classmate_D : StudentInfo := ⟨Surname.Huang, Gender.Female, 220, Specialty.Drawing⟩
def classmate_E : StudentInfo := ⟨Surname.Zhang, Gender.Female, 240, Specialty.Singing⟩

-- Define the correct information
def correct_info : StudentInfo := ⟨Surname.Huang, Gender.Male, 240, Specialty.Dancing⟩

-- Define a function to check if a piece of information is correct
def is_correct_piece (info : StudentInfo) (correct : StudentInfo) : Bool :=
  info.surname = correct.surname ∨ 
  info.gender = correct.gender ∨ 
  info.totalScore = correct.totalScore ∨ 
  info.specialty = correct.specialty

-- Theorem statement
theorem new_student_info_is_unique :
  (is_correct_piece classmate_A correct_info) ∧
  (is_correct_piece classmate_B correct_info) ∧
  (is_correct_piece classmate_C correct_info) ∧
  (is_correct_piece classmate_D correct_info) ∧
  (is_correct_piece classmate_E correct_info) ∧
  (∀ info : StudentInfo, 
    info ≠ correct_info → 
    (¬(is_correct_piece classmate_A info) ∨
     ¬(is_correct_piece classmate_B info) ∨
     ¬(is_correct_piece classmate_C info) ∨
     ¬(is_correct_piece classmate_D info) ∨
     ¬(is_correct_piece classmate_E info))) :=
by sorry

end NUMINAMATH_CALUDE_new_student_info_is_unique_l2232_223270


namespace NUMINAMATH_CALUDE_books_total_is_54_l2232_223246

/-- The total number of books Darla, Katie, and Gary have -/
def total_books (darla_books katie_books gary_books : ℕ) : ℕ :=
  darla_books + katie_books + gary_books

/-- Theorem stating the total number of books is 54 -/
theorem books_total_is_54 :
  ∀ (darla_books katie_books gary_books : ℕ),
    darla_books = 6 →
    katie_books = darla_books / 2 →
    gary_books = 5 * (darla_books + katie_books) →
    total_books darla_books katie_books gary_books = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_books_total_is_54_l2232_223246


namespace NUMINAMATH_CALUDE_function_inequality_l2232_223269

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

theorem function_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_periodic : ∀ x, f (x + 4) = -f x)
  (h_decreasing : is_decreasing_on f 0 4) :
  f 13 < f 10 ∧ f 10 < f 15 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2232_223269


namespace NUMINAMATH_CALUDE_three_digit_sum_reverse_l2232_223218

theorem three_digit_sum_reverse : ∃ (a b c : ℕ), 
  (a ≥ 1 ∧ a ≤ 9) ∧ 
  (b ≥ 0 ∧ b ≤ 9) ∧ 
  (c ≥ 0 ∧ c ≤ 9) ∧
  (100 * a + 10 * b + c) + (100 * c + 10 * b + a) = 1777 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_sum_reverse_l2232_223218


namespace NUMINAMATH_CALUDE_no_adjacent_same_probability_correct_probability_between_zero_and_one_l2232_223284

def number_of_people : ℕ := 6
def die_sides : ℕ := 6

/-- The probability of no two adjacent people rolling the same number on a six-sided die 
    when six people are sitting around a circular table. -/
def no_adjacent_same_probability : ℚ :=
  625 / 1944

/-- Theorem stating that the calculated probability is correct. -/
theorem no_adjacent_same_probability_correct : 
  no_adjacent_same_probability = 625 / 1944 := by
  sorry

/-- Theorem stating that the probability is between 0 and 1. -/
theorem probability_between_zero_and_one :
  0 ≤ no_adjacent_same_probability ∧ no_adjacent_same_probability ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_same_probability_correct_probability_between_zero_and_one_l2232_223284


namespace NUMINAMATH_CALUDE_triangle_area_fraction_l2232_223298

/-- The area of a right triangle with vertices at (3,3), (5,5), and (3,5) on a 6 by 6 grid
    is 1/18 of the total area of the 6 by 6 square. -/
theorem triangle_area_fraction (grid_size : ℕ) (x1 y1 x2 y2 x3 y3 : ℕ) : 
  grid_size = 6 →
  x1 = 3 → y1 = 3 →
  x2 = 5 → y2 = 5 →
  x3 = 3 → y3 = 5 →
  (1 : ℚ) / 2 * (x2 - x1) * (y3 - y1) / (grid_size * grid_size) = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_fraction_l2232_223298


namespace NUMINAMATH_CALUDE_sum_equals_five_l2232_223258

/-- Definition of the star operation -/
def star (a b : ℕ) : ℤ := a^b - a*b

/-- Theorem statement -/
theorem sum_equals_five (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (h : star a b = 3) : a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_five_l2232_223258


namespace NUMINAMATH_CALUDE_opposite_numbers_expression_l2232_223216

theorem opposite_numbers_expression (a b c d : ℤ) : 
  (a + b = 0) →  -- a and b are opposite numbers
  (c = -1) →     -- c is the largest negative integer
  (d = 1) →      -- d is the smallest positive integer
  (a + b) * d + d - c = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_expression_l2232_223216


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_cube_minus_self_l2232_223273

def is_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = p^2

theorem largest_common_divisor_of_cube_minus_self (n : ℕ) (h : is_prime_square n) :
  (∀ d : ℕ, d > 30 → ¬(d ∣ (n^3 - n))) ∧
  (30 ∣ (n^3 - n)) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_cube_minus_self_l2232_223273


namespace NUMINAMATH_CALUDE_golden_ratio_geometric_mean_l2232_223280

/-- Golden ratio division of a line segment -/
structure GoldenRatioDivision (α : Type*) [LinearOrderedField α] where
  a : α -- length of the whole segment
  b : α -- length of the smaller segment
  h1 : 0 < b
  h2 : b < a
  h3 : (a - b) / b = b / a -- golden ratio condition

/-- Right triangle formed from a golden ratio division -/
def goldenRatioTriangle {α : Type*} [LinearOrderedField α] (d : GoldenRatioDivision α) :=
  { x : α // x^2 + d.b^2 = d.a^2 }

/-- The other leg of the golden ratio triangle is the geometric mean of the hypotenuse and the first leg -/
theorem golden_ratio_geometric_mean {α : Type*} [LinearOrderedField α] (d : GoldenRatioDivision α) :
  let t := goldenRatioTriangle d
  ∀ x : t, x.val * d.a = d.b * x.val :=
by sorry

end NUMINAMATH_CALUDE_golden_ratio_geometric_mean_l2232_223280


namespace NUMINAMATH_CALUDE_odot_calculation_l2232_223238

-- Define the ⊙ operation
def odot (a b : ℤ) : ℤ := a * b - (a + b)

-- State the theorem
theorem odot_calculation : odot 6 (odot 5 4) = 49 := by
  sorry

end NUMINAMATH_CALUDE_odot_calculation_l2232_223238


namespace NUMINAMATH_CALUDE_tamika_always_wins_l2232_223275

def tamika_set : Finset ℕ := {7, 11, 14}
def carlos_set : Finset ℕ := {2, 4, 7}

theorem tamika_always_wins :
  ∀ (a b : ℕ), a ∈ tamika_set → b ∈ tamika_set → a ≠ b →
    ∀ (c d : ℕ), c ∈ carlos_set → d ∈ carlos_set → c ≠ d →
      a * b > c + d :=
by sorry

end NUMINAMATH_CALUDE_tamika_always_wins_l2232_223275


namespace NUMINAMATH_CALUDE_grid_paths_7x6_l2232_223214

/-- The number of paths in a grid from (0,0) to (m,n) where each step is either right or up -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The dimensions of our grid -/
def gridWidth : ℕ := 7
def gridHeight : ℕ := 6

/-- The total number of steps in our grid -/
def totalSteps : ℕ := gridWidth + gridHeight

theorem grid_paths_7x6 : gridPaths gridWidth gridHeight = 1716 := by sorry

end NUMINAMATH_CALUDE_grid_paths_7x6_l2232_223214


namespace NUMINAMATH_CALUDE_f_properties_l2232_223232

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - a*b*c

-- State the theorem
theorem f_properties (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c)
  (h4 : f a b c a = 0) (h5 : f a b c b = 0) (h6 : f a b c c = 0) :
  (f a b c 0) * (f a b c 1) < 0 ∧ (f a b c 0) * (f a b c 3) > 0 := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l2232_223232


namespace NUMINAMATH_CALUDE_barney_towel_usage_l2232_223222

/-- The number of towels Barney owns -/
def total_towels : ℕ := 18

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of days Barney will not have clean towels -/
def days_without_clean_towels : ℕ := 5

/-- The number of towels Barney uses at a time -/
def towels_per_use : ℕ := 2

theorem barney_towel_usage :
  ∃ (x : ℕ),
    x = towels_per_use ∧
    total_towels - days_per_week * x = (days_per_week - days_without_clean_towels) * x :=
by sorry

end NUMINAMATH_CALUDE_barney_towel_usage_l2232_223222


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l2232_223207

theorem sqrt_inequality_solution_set (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 3 ∧ y < 2) ↔ x ∈ Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l2232_223207


namespace NUMINAMATH_CALUDE_sin_transformation_l2232_223242

theorem sin_transformation (x : ℝ) : 
  2 * Real.sin (3 * x + π / 6) = 2 * Real.sin ((x + π / 6) / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_transformation_l2232_223242


namespace NUMINAMATH_CALUDE_subtraction_inequality_l2232_223260

theorem subtraction_inequality (a b c : ℝ) (h : a > b) : c - a < c - b := by
  sorry

end NUMINAMATH_CALUDE_subtraction_inequality_l2232_223260


namespace NUMINAMATH_CALUDE_problem_statement_l2232_223227

theorem problem_statement : 
  let N := (Real.sqrt (Real.sqrt 6 + 3) + Real.sqrt (Real.sqrt 6 - 3)) / Real.sqrt (Real.sqrt 6 + 2) - Real.sqrt (4 - 2 * Real.sqrt 3)
  N = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2232_223227


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2232_223229

theorem necessary_but_not_sufficient_condition :
  ∀ x : ℝ, (2 * x^2 - 5 * x - 3 ≥ 0) → (x < 0 ∨ x > 2) ∧
  ∃ y : ℝ, (y < 0 ∨ y > 2) ∧ ¬(2 * y^2 - 5 * y - 3 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2232_223229


namespace NUMINAMATH_CALUDE_ratio_theorem_l2232_223271

theorem ratio_theorem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) : 
  (2 * a + 3 * b) / (b + 2 * c) = 11 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ratio_theorem_l2232_223271


namespace NUMINAMATH_CALUDE_stickers_per_sheet_l2232_223263

theorem stickers_per_sheet 
  (initial_stickers : ℕ) 
  (shared_stickers : ℕ) 
  (remaining_sheets : ℕ) 
  (h1 : initial_stickers = 150)
  (h2 : shared_stickers = 100)
  (h3 : remaining_sheets = 5)
  (h4 : initial_stickers ≥ shared_stickers)
  (h5 : remaining_sheets > 0) :
  (initial_stickers - shared_stickers) / remaining_sheets = 10 :=
by sorry

end NUMINAMATH_CALUDE_stickers_per_sheet_l2232_223263


namespace NUMINAMATH_CALUDE_max_difference_l2232_223205

theorem max_difference (a b : ℝ) (h1 : a < 0) 
  (h2 : ∀ x ∈ Set.Ioo a b, (3 * x^2 + a) * (2 * x + b) ≥ 0) :
  b - a ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_difference_l2232_223205


namespace NUMINAMATH_CALUDE_custom_mul_result_l2232_223220

/-- Custom multiplication operation -/
def custom_mul (a b c : ℚ) (x y : ℚ) : ℚ := a * x + b * y + c

theorem custom_mul_result (a b c : ℚ) :
  custom_mul a b c 1 2 = 9 →
  custom_mul a b c (-3) 3 = 6 →
  custom_mul a b c 0 1 = 2 →
  custom_mul a b c (-2) 5 = 18 := by
sorry

end NUMINAMATH_CALUDE_custom_mul_result_l2232_223220


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_large_rectangle_perimeter_proof_l2232_223255

theorem large_rectangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun square_perimeter small_rect_perimeter large_rect_perimeter =>
    square_perimeter = 24 ∧
    small_rect_perimeter = 16 ∧
    (∃ (square_side small_rect_width : ℝ),
      square_side = square_perimeter / 4 ∧
      small_rect_width = small_rect_perimeter / 2 - square_side ∧
      large_rect_perimeter = 2 * (3 * square_side + (square_side + small_rect_width))) →
    large_rect_perimeter = 52

theorem large_rectangle_perimeter_proof :
  large_rectangle_perimeter 24 16 52 :=
sorry

end NUMINAMATH_CALUDE_large_rectangle_perimeter_large_rectangle_perimeter_proof_l2232_223255


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2232_223249

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 10) →  -- Ensure n has at least two digits
  (n % 4 = 0) →     -- n is divisible by 4
  ((n % 100) / 10 + n % 10 = 16) →  -- Sum of last two digits is 16
  ((n % 100) / 10) * (n % 10) = 64 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2232_223249


namespace NUMINAMATH_CALUDE_black_cubes_removed_multiple_of_four_l2232_223224

/-- Represents a cube constructed from unit cubes of two colors -/
structure ColoredCube where
  edge_length : ℕ
  black_cubes : ℕ
  white_cubes : ℕ
  adjacent_different : Bool

/-- Represents the removal of unit cubes from a ColoredCube -/
structure CubeRemoval where
  cube : ColoredCube
  removed_cubes : ℕ
  rods_affected : ℕ
  cubes_per_rod : ℕ

/-- Theorem stating that the number of black cubes removed is a multiple of 4 -/
theorem black_cubes_removed_multiple_of_four (removal : CubeRemoval) : 
  removal.cube.edge_length = 10 ∧ 
  removal.cube.black_cubes = 500 ∧ 
  removal.cube.white_cubes = 500 ∧
  removal.cube.adjacent_different = true ∧
  removal.removed_cubes = 100 ∧
  removal.rods_affected = 300 ∧
  removal.cubes_per_rod = 1 →
  ∃ (k : ℕ), (removal.removed_cubes - k) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_black_cubes_removed_multiple_of_four_l2232_223224


namespace NUMINAMATH_CALUDE_negative_five_greater_than_negative_seventeen_l2232_223202

theorem negative_five_greater_than_negative_seventeen : -5 > -17 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_greater_than_negative_seventeen_l2232_223202


namespace NUMINAMATH_CALUDE_tank_inlet_rate_l2232_223294

/-- Given a tank with the following properties:
  * Capacity of 3600.000000000001 liters
  * Empties in 6 hours due to a leak
  * Empties in 8 hours when both the leak and inlet are open
  Prove that the rate at which the inlet pipe fills the tank is 150 liters per hour -/
theorem tank_inlet_rate (capacity : ℝ) (leak_time : ℝ) (combined_time : ℝ) :
  capacity = 3600.000000000001 →
  leak_time = 6 →
  combined_time = 8 →
  ∃ (inlet_rate : ℝ),
    inlet_rate = 150 ∧
    inlet_rate = (capacity / leak_time) - (capacity / combined_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_inlet_rate_l2232_223294


namespace NUMINAMATH_CALUDE_area_of_ABCD_l2232_223278

/-- Represents a rectangle with length and height -/
structure Rectangle where
  length : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.height

/-- Theorem: Area of rectangle ABCD -/
theorem area_of_ABCD (r1 r2 r3 : Rectangle) (ABCD : Rectangle) :
  area r1 + area r2 + area r3 = area ABCD →
  area r1 = 2 →
  ABCD.length = 5 →
  ABCD.height = 3 →
  area ABCD = 15 := by
  sorry

#check area_of_ABCD

end NUMINAMATH_CALUDE_area_of_ABCD_l2232_223278


namespace NUMINAMATH_CALUDE_factory_production_l2232_223297

/-- Calculates the number of toys produced per week in a factory -/
def toys_per_week (days_per_week : ℕ) (toys_per_day : ℕ) : ℕ :=
  days_per_week * toys_per_day

/-- Proves that the factory produces 5505 toys per week -/
theorem factory_production : toys_per_week 5 1101 = 5505 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l2232_223297


namespace NUMINAMATH_CALUDE_equation_solution_l2232_223221

theorem equation_solution : 
  ∃ (x : ℝ), x ≠ 2 ∧ (4*x^2 + 3*x + 2) / (x - 2) = 4*x + 5 → x = -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2232_223221


namespace NUMINAMATH_CALUDE_x_power_six_plus_reciprocal_l2232_223282

theorem x_power_six_plus_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^6 + 1/x^6 = 322 := by
  sorry

end NUMINAMATH_CALUDE_x_power_six_plus_reciprocal_l2232_223282


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l2232_223236

theorem sphere_radius_ratio (V_large V_small : ℝ) (h1 : V_large = 500 * Real.pi) (h2 : V_small = 40 * Real.pi) :
  (((3 * V_small) / (4 * Real.pi)) ^ (1/3)) / (((3 * V_large) / (4 * Real.pi)) ^ (1/3)) = (10 ^ (1/3)) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l2232_223236


namespace NUMINAMATH_CALUDE_queen_secondary_teachers_queen_secondary_teachers_count_l2232_223256

/-- Calculates the number of teachers required at Queen Secondary School -/
theorem queen_secondary_teachers (total_students : ℕ) (classes_per_student : ℕ) 
  (students_per_class : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  let total_class_instances := total_students * classes_per_student
  let unique_classes := total_class_instances / students_per_class
  unique_classes / classes_per_teacher

/-- Proves that the number of teachers at Queen Secondary School is 48 -/
theorem queen_secondary_teachers_count : 
  queen_secondary_teachers 1500 4 25 5 = 48 := by
  sorry

end NUMINAMATH_CALUDE_queen_secondary_teachers_queen_secondary_teachers_count_l2232_223256
