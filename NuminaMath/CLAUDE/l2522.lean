import Mathlib

namespace NUMINAMATH_CALUDE_buckingham_visitors_theorem_l2522_252248

/-- Represents the number of visitors to Buckingham Palace -/
structure BuckinghamVisitors where
  total_85_days : ℕ
  previous_day : ℕ

/-- Calculates the number of visitors on a specific day -/
def visitors_on_day (bv : BuckinghamVisitors) : ℕ :=
  bv.total_85_days - bv.previous_day

/-- Theorem statement for the Buckingham Palace visitor calculation -/
theorem buckingham_visitors_theorem (bv : BuckinghamVisitors) 
  (h1 : bv.total_85_days = 829)
  (h2 : bv.previous_day = 45) :
  visitors_on_day bv = 784 := by
  sorry

#eval visitors_on_day { total_85_days := 829, previous_day := 45 }

end NUMINAMATH_CALUDE_buckingham_visitors_theorem_l2522_252248


namespace NUMINAMATH_CALUDE_min_sum_of_product_l2522_252286

theorem min_sum_of_product (a b : ℤ) (h : a * b = 196) : 
  ∀ x y : ℤ, x * y = 196 → a + b ≤ x + y ∧ ∃ a b : ℤ, a * b = 196 ∧ a + b = -197 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l2522_252286


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2522_252270

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := ∃ (y : ℝ), x = y^2

-- Define the concept of simplest quadratic radical
def SimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧ 
  ∀ y : ℝ, (QuadraticRadical y ∧ y ≠ x) → (∃ z : ℝ, z ≠ 1 ∧ y = z * x)

-- Theorem statement
theorem simplest_quadratic_radical :
  SimplestQuadraticRadical (Real.sqrt 6) ∧
  ¬SimplestQuadraticRadical (Real.sqrt 12) ∧
  ¬SimplestQuadraticRadical (Real.sqrt (1/3)) ∧
  ¬SimplestQuadraticRadical (Real.sqrt 0.3) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2522_252270


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2522_252219

theorem complex_equation_solution (x y : ℂ) (hx : x ≠ 0) (hxy : x + 2*y ≠ 0) :
  (x + 2*y) / x = 2*y / (x + 2*y) →
  (x = -y + Complex.I * Real.sqrt 3 * y) ∨ (x = -y - Complex.I * Real.sqrt 3 * y) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2522_252219


namespace NUMINAMATH_CALUDE_exchange_ways_eq_six_l2522_252269

/-- The number of ways to exchange 100 yuan into 20 yuan and 10 yuan bills -/
def exchange_ways : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 20 * p.1 + 10 * p.2 = 100) (Finset.product (Finset.range 6) (Finset.range 11))).card

/-- Theorem stating that there are exactly 6 ways to exchange 100 yuan into 20 yuan and 10 yuan bills -/
theorem exchange_ways_eq_six : exchange_ways = 6 := by
  sorry

end NUMINAMATH_CALUDE_exchange_ways_eq_six_l2522_252269


namespace NUMINAMATH_CALUDE_optimal_procurement_plan_l2522_252257

/-- Represents a snowflake model type -/
inductive ModelType
| A
| B

/-- Represents the number of pipes needed for a model -/
structure PipeCount where
  long : ℕ
  short : ℕ

/-- Represents the store's inventory -/
structure Inventory where
  long : ℕ
  short : ℕ

/-- Represents a procurement plan -/
structure ProcurementPlan where
  modelA : ℕ
  modelB : ℕ

def pipe_price : ℚ := 1/2

def long_pipe_price : ℚ := 2 * pipe_price

def inventory : Inventory := ⟨267, 2130⟩

def budget : ℚ := 1280

def pipes_per_model (t : ModelType) : PipeCount :=
  match t with
  | ModelType.A => ⟨3, 21⟩
  | ModelType.B => ⟨3, 27⟩

def cost_of_plan (plan : ProcurementPlan) : ℚ :=
  let total_long := plan.modelA * (pipes_per_model ModelType.A).long + 
                    plan.modelB * (pipes_per_model ModelType.B).long
  let total_short := plan.modelA * (pipes_per_model ModelType.A).short + 
                     plan.modelB * (pipes_per_model ModelType.B).short - 
                     (total_long / 3)
  total_long * long_pipe_price + total_short * pipe_price

def is_valid_plan (plan : ProcurementPlan) : Prop :=
  let total_long := plan.modelA * (pipes_per_model ModelType.A).long + 
                    plan.modelB * (pipes_per_model ModelType.B).long
  let total_short := plan.modelA * (pipes_per_model ModelType.A).short + 
                     plan.modelB * (pipes_per_model ModelType.B).short
  total_long ≤ inventory.long ∧ 
  total_short ≤ inventory.short ∧ 
  cost_of_plan plan = budget

theorem optimal_procurement_plan :
  ∀ plan : ProcurementPlan,
    is_valid_plan plan →
    plan.modelA + plan.modelB ≤ 49 ∧
    (plan.modelA + plan.modelB = 49 → plan.modelA = 48 ∧ plan.modelB = 1) :=
sorry

end NUMINAMATH_CALUDE_optimal_procurement_plan_l2522_252257


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2522_252230

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : ℝ := x^2 + 2*k*x + k

-- Theorem statement
theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, quadratic_equation x k = 0) ↔ (k ≤ 0 ∨ k ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2522_252230


namespace NUMINAMATH_CALUDE_teresa_pencil_distribution_l2522_252249

/-- Given Teresa's pencil collection and distribution rules, prove each sibling gets 13 pencils -/
theorem teresa_pencil_distribution :
  let colored_pencils : ℕ := 14
  let black_pencils : ℕ := 35
  let total_pencils : ℕ := colored_pencils + black_pencils
  let pencils_to_keep : ℕ := 10
  let number_of_siblings : ℕ := 3
  let pencils_to_distribute : ℕ := total_pencils - pencils_to_keep
  pencils_to_distribute / number_of_siblings = 13 :=
by
  sorry

#eval (14 + 35 - 10) / 3  -- This should output 13

end NUMINAMATH_CALUDE_teresa_pencil_distribution_l2522_252249


namespace NUMINAMATH_CALUDE_custom_mul_identity_l2522_252294

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 4 * a * b

/-- Theorem stating that if a * x = x for all x, then a = 1/4 -/
theorem custom_mul_identity (a : ℝ) : 
  (∀ x, custom_mul a x = x) → a = (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_identity_l2522_252294


namespace NUMINAMATH_CALUDE_garden_potato_yield_l2522_252222

def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def feet_per_step : ℕ := 3
def potato_yield_per_sqft : ℚ := 3/4

def garden_length_feet : ℕ := garden_length_steps * feet_per_step
def garden_width_feet : ℕ := garden_width_steps * feet_per_step
def garden_area_sqft : ℕ := garden_length_feet * garden_width_feet

def expected_potato_yield : ℚ := garden_area_sqft * potato_yield_per_sqft

theorem garden_potato_yield :
  expected_potato_yield = 3037.5 := by sorry

end NUMINAMATH_CALUDE_garden_potato_yield_l2522_252222


namespace NUMINAMATH_CALUDE_p_properties_l2522_252223

-- Define the polynomial
def p (x y : ℝ) : ℝ := x^2 * y^3 - 3 * x * y^3 - 2

-- Define the degree of a monomial
def monomial_degree (m : ℕ × ℕ) : ℕ := m.1 + m.2

-- Define the degree of a polynomial
def polynomial_degree (p : ℝ → ℝ → ℝ) : ℕ :=
  -- This definition is a placeholder and needs to be implemented
  sorry

-- Define the number of terms in a polynomial
def number_of_terms (p : ℝ → ℝ → ℝ) : ℕ :=
  -- This definition is a placeholder and needs to be implemented
  sorry

-- Theorem stating the properties of the polynomial p
theorem p_properties :
  polynomial_degree p = 5 ∧ number_of_terms p = 3 := by
  sorry

end NUMINAMATH_CALUDE_p_properties_l2522_252223


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2522_252275

theorem inequality_equivalence (x y : ℝ) (h : x > 0) :
  (Real.sqrt (y - x) / x ≤ 1) ↔ (x ≤ y ∧ y ≤ x^2 + x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2522_252275


namespace NUMINAMATH_CALUDE_square_room_tiles_l2522_252239

theorem square_room_tiles (n : ℕ) : 
  n > 0 →  -- Ensure the room has a positive size
  (2 * n = 62) →  -- Total tiles on both diagonals
  n * n = 961  -- Total tiles in the room
  := by sorry

end NUMINAMATH_CALUDE_square_room_tiles_l2522_252239


namespace NUMINAMATH_CALUDE_cents_left_over_l2522_252233

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the number of pennies in the jar -/
def num_pennies : ℕ := 123

/-- Represents the number of nickels in the jar -/
def num_nickels : ℕ := 85

/-- Represents the number of dimes in the jar -/
def num_dimes : ℕ := 35

/-- Represents the number of quarters in the jar -/
def num_quarters : ℕ := 26

/-- Represents the cost of a double scoop in dollars -/
def double_scoop_cost : ℕ := 3

/-- Represents the number of family members -/
def num_family_members : ℕ := 5

/-- Theorem stating that the number of cents left over after the trip is 48 -/
theorem cents_left_over : 
  (num_pennies * penny_value + 
   num_nickels * nickel_value + 
   num_dimes * dime_value + 
   num_quarters * quarter_value) - 
  (double_scoop_cost * num_family_members * cents_per_dollar) = 48 := by
  sorry


end NUMINAMATH_CALUDE_cents_left_over_l2522_252233


namespace NUMINAMATH_CALUDE_distribute_six_to_four_l2522_252253

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinct objects into 4 distinct groups,
    where each group must contain at least one object, is 1560. -/
theorem distribute_six_to_four : distribute 6 4 = 1560 := by sorry

end NUMINAMATH_CALUDE_distribute_six_to_four_l2522_252253


namespace NUMINAMATH_CALUDE_smallest_divisor_power_l2522_252227

def polynomial (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_divisor_power : ∃! k : ℕ+, 
  (∀ z : ℂ, polynomial z ∣ (z^k.val - 1)) ∧ 
  (∀ m : ℕ+, m < k → ∃ z : ℂ, ¬(polynomial z ∣ (z^m.val - 1))) ∧
  k = 84 := by sorry

end NUMINAMATH_CALUDE_smallest_divisor_power_l2522_252227


namespace NUMINAMATH_CALUDE_junior_boy_girl_ratio_l2522_252209

/-- Represents the number of participants in each category -/
structure Participants where
  juniorBoys : ℕ
  seniorBoys : ℕ
  juniorGirls : ℕ
  seniorGirls : ℕ

/-- The ratio of boys to total participants is 55% -/
def boyRatio (p : Participants) : Prop :=
  (p.juniorBoys + p.seniorBoys : ℚ) / (p.juniorBoys + p.seniorBoys + p.juniorGirls + p.seniorGirls) = 55 / 100

/-- The ratio of junior boys to senior boys equals the ratio of all juniors to all seniors -/
def juniorSeniorRatio (p : Participants) : Prop :=
  (p.juniorBoys : ℚ) / p.seniorBoys = (p.juniorBoys + p.juniorGirls : ℚ) / (p.seniorBoys + p.seniorGirls)

/-- The main theorem: given the conditions, prove that the ratio of junior boys to junior girls is 11:9 -/
theorem junior_boy_girl_ratio (p : Participants) 
  (hBoyRatio : boyRatio p) (hJuniorSeniorRatio : juniorSeniorRatio p) : 
  (p.juniorBoys : ℚ) / p.juniorGirls = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_junior_boy_girl_ratio_l2522_252209


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2522_252218

theorem divisibility_theorem (k n : ℕ) (hk : k > 0) (hn : n > 0) :
  ∃ m : ℤ, (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4*k - 1) = m * (n^5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2522_252218


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2522_252200

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![3, -2]

theorem vector_magnitude_proof :
  Real.sqrt ((2 * a 0 - b 0)^2 + (2 * a 1 - b 1)^2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2522_252200


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2522_252271

theorem unique_quadratic_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + 2*(b + 1/b)*x + c = 0) → 
  c = 4 := by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2522_252271


namespace NUMINAMATH_CALUDE_triangle_classification_l2522_252297

theorem triangle_classification (a b : ℝ) (A B : ℝ) (h_positive : 0 < A ∧ A < π) 
  (h_eq : a * Real.cos A = b * Real.cos B) :
  A = B ∨ A + B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_classification_l2522_252297


namespace NUMINAMATH_CALUDE_land_area_calculation_l2522_252247

theorem land_area_calculation (average_yield total_area first_area first_yield second_yield : ℝ) : 
  average_yield = 675 →
  first_area = 5 →
  first_yield = 705 →
  second_yield = 650 →
  total_area * average_yield = first_area * first_yield + (total_area - first_area) * second_yield →
  total_area = 11 :=
by sorry

end NUMINAMATH_CALUDE_land_area_calculation_l2522_252247


namespace NUMINAMATH_CALUDE_sum_of_two_and_four_l2522_252237

theorem sum_of_two_and_four : 2 + 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_and_four_l2522_252237


namespace NUMINAMATH_CALUDE_cos_300_degrees_l2522_252280

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l2522_252280


namespace NUMINAMATH_CALUDE_f_expression_for_x_less_than_2_l2522_252240

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_expression_for_x_less_than_2
  (f : ℝ → ℝ)
  (h1 : is_even_function (λ x ↦ f (x + 2)))
  (h2 : ∀ x ≥ 2, f x = 3^x - 1) :
  ∀ x < 2, f x = 3^(4 - x) - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_expression_for_x_less_than_2_l2522_252240


namespace NUMINAMATH_CALUDE_overall_profit_l2522_252299

def refrigerator_cost : ℝ := 15000
def mobile_cost : ℝ := 8000
def refrigerator_loss_percent : ℝ := 0.03
def mobile_profit_percent : ℝ := 0.10

def refrigerator_selling_price : ℝ := refrigerator_cost * (1 - refrigerator_loss_percent)
def mobile_selling_price : ℝ := mobile_cost * (1 + mobile_profit_percent)

def total_cost : ℝ := refrigerator_cost + mobile_cost
def total_selling_price : ℝ := refrigerator_selling_price + mobile_selling_price

theorem overall_profit : total_selling_price - total_cost = 350 := by
  sorry

end NUMINAMATH_CALUDE_overall_profit_l2522_252299


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2522_252250

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 12) (h2 : x - y = 20) : y = -4 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2522_252250


namespace NUMINAMATH_CALUDE_value_of_A_l2522_252245

def round_down_tens (n : ℕ) : ℕ := n / 10 * 10

theorem value_of_A (A : ℕ) : 
  A < 10 → 
  round_down_tens (900 + 10 * A + 7) = 930 → 
  A = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_A_l2522_252245


namespace NUMINAMATH_CALUDE_bowl_game_score_l2522_252231

/-- Given the scores of Noa, Phillip, and Lucy in a bowl game, prove their total score. -/
theorem bowl_game_score (noa_score : ℕ) (phillip_score : ℕ) (lucy_score : ℕ) 
  (h1 : noa_score = 30)
  (h2 : phillip_score = 2 * noa_score)
  (h3 : lucy_score = (3 : ℕ) / 2 * phillip_score) :
  noa_score + phillip_score + lucy_score = 180 := by
  sorry

end NUMINAMATH_CALUDE_bowl_game_score_l2522_252231


namespace NUMINAMATH_CALUDE_first_digit_change_largest_l2522_252229

def original_number : ℚ := 0.12345678

def change_digit (n : ℚ) (position : ℕ) : ℚ :=
  n + (9 - (n * 10^position % 10)) / 10^position

theorem first_digit_change_largest :
  ∀ position : ℕ, position > 0 → 
    change_digit original_number 1 ≥ change_digit original_number position :=
by sorry

end NUMINAMATH_CALUDE_first_digit_change_largest_l2522_252229


namespace NUMINAMATH_CALUDE_wheel_distance_l2522_252259

/-- Given two wheels with different perimeters, prove that the distance traveled
    is 315 feet when the front wheel makes 10 more revolutions than the back wheel. -/
theorem wheel_distance (back_perimeter front_perimeter : ℝ) 
  (h1 : back_perimeter = 9)
  (h2 : front_perimeter = 7)
  (h3 : ∃ (back_revs front_revs : ℝ), 
    front_revs = back_revs + 10 ∧ 
    back_revs * back_perimeter = front_revs * front_perimeter) :
  ∃ (distance : ℝ), distance = 315 := by
  sorry

end NUMINAMATH_CALUDE_wheel_distance_l2522_252259


namespace NUMINAMATH_CALUDE_min_height_box_l2522_252238

/-- Represents a rectangular box with a square base -/
structure Box where
  base : ℝ
  height : ℝ

/-- Calculates the surface area of a box -/
def surfaceArea (b : Box) : ℝ :=
  2 * b.base^2 + 4 * b.base * b.height

/-- Theorem stating the minimum height of the box under given conditions -/
theorem min_height_box :
  ∀ (b : Box),
    b.height = b.base + 4 →
    surfaceArea b ≥ 150 →
    ∀ (b' : Box),
      b'.height = b'.base + 4 →
      surfaceArea b' ≥ 150 →
      b.height ≤ b'.height →
      b.height = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_height_box_l2522_252238


namespace NUMINAMATH_CALUDE_correct_street_loss_percentage_l2522_252279

/-- The percentage of marbles lost into the street -/
def street_loss_percentage : ℝ := 60

/-- The initial number of marbles -/
def initial_marbles : ℕ := 100

/-- The final number of marbles after losses -/
def final_marbles : ℕ := 20

/-- Theorem stating the correct percentage of marbles lost into the street -/
theorem correct_street_loss_percentage :
  street_loss_percentage = 60 ∧
  final_marbles = (initial_marbles - initial_marbles * street_loss_percentage / 100) / 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_street_loss_percentage_l2522_252279


namespace NUMINAMATH_CALUDE_f_properties_l2522_252255

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem f_properties (a : ℝ) (h : a < 0) :
  (∀ x : ℝ, x ≠ 0 → f a x + f a (-1/x) ≥ 2) ∧
  (∃ x : ℝ, f a x + f a (2*x) < 1/2 ↔ -1 < a ∧ a < 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2522_252255


namespace NUMINAMATH_CALUDE_jan_ian_distance_difference_l2522_252228

/-- Represents the driving scenario of Ian, Han, and Jan -/
structure DrivingScenario where
  t : ℝ  -- Ian's driving time
  s : ℝ  -- Ian's driving speed
  ian_distance : ℝ := t * s
  han_distance : ℝ := (t + 2) * (s + 10)
  jan_distance : ℝ := (t + 3) * (s + 15)

/-- The theorem stating the difference between Jan's and Ian's distances -/
theorem jan_ian_distance_difference (scenario : DrivingScenario) 
  (h : scenario.han_distance = scenario.ian_distance + 100) : 
  scenario.jan_distance - scenario.ian_distance = 165 := by
  sorry

#check jan_ian_distance_difference

end NUMINAMATH_CALUDE_jan_ian_distance_difference_l2522_252228


namespace NUMINAMATH_CALUDE_solution_sets_correct_l2522_252292

-- Define the solution sets for each inequality
def solution_set1 : Set ℝ := {x | x < -1 ∨ x > 3/2}
def solution_set2 : Set ℝ := {x | x < 2 ∨ x ≥ 5}

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := -2 * x^2 + x < -3
def inequality2 (x : ℝ) : Prop := (x + 1) / (x - 2) ≤ 2

-- Theorem stating that the solution sets are correct
theorem solution_sets_correct :
  (∀ x : ℝ, x ∈ solution_set1 ↔ inequality1 x) ∧
  (∀ x : ℝ, x ∈ solution_set2 ↔ inequality2 x) :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_correct_l2522_252292


namespace NUMINAMATH_CALUDE_job_completion_time_l2522_252246

theorem job_completion_time (efficiency_ratio : ℝ) (joint_completion_time : ℝ) :
  efficiency_ratio = (1 : ℝ) / 2 →
  joint_completion_time = 15 →
  ∃ (solo_completion_time : ℝ),
    solo_completion_time = (3 / 2) * joint_completion_time ∧
    solo_completion_time = 45 / 2 :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l2522_252246


namespace NUMINAMATH_CALUDE_unique_solution_l2522_252266

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Define the equation
def equation (x : ℕ) : Prop :=
  combination x 3 + combination x 2 = 12 * (x - 1)

-- State the theorem
theorem unique_solution :
  ∃! x : ℕ, x ≥ 3 ∧ equation x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2522_252266


namespace NUMINAMATH_CALUDE_function_domain_constraint_l2522_252205

theorem function_domain_constraint (f : ℝ → ℝ) (h : ∀ x, x ∈ (Set.Icc 0 1) → f x ≠ 0) :
  ∀ a : ℝ, (∀ x, x ∈ (Set.Icc 0 1) → (f (x - a) + f (x + a)) ≠ 0) ↔ a ∈ (Set.Icc (-1/2) (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_function_domain_constraint_l2522_252205


namespace NUMINAMATH_CALUDE_pigeon_increase_l2522_252282

theorem pigeon_increase (total : ℕ) (initial : ℕ) (h1 : total = 21) (h2 : initial = 15) :
  total - initial = 6 := by
  sorry

end NUMINAMATH_CALUDE_pigeon_increase_l2522_252282


namespace NUMINAMATH_CALUDE_sum_max_value_sum_max_x_product_max_value_product_max_x_l2522_252202

/-- Represents a point on an ellipse --/
structure EllipsePoint where
  x : ℝ
  y : ℝ
  a : ℝ
  b : ℝ
  h_ellipse : x^2 / a^2 + y^2 / b^2 = 1
  h_positive : a > 0 ∧ b > 0

/-- The sum of x and y coordinates has a maximum value --/
theorem sum_max_value (p : EllipsePoint) :
  ∃ m : ℝ, m = Real.sqrt (p.a^2 + p.b^2) ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b → q.x + q.y ≤ m :=
sorry

/-- The sum of x and y coordinates reaches its maximum when x has a specific value --/
theorem sum_max_x (p : EllipsePoint) :
  ∃ x_max : ℝ, x_max = p.a^2 / Real.sqrt (p.a^2 + p.b^2) ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b ∧ q.x + q.y = Real.sqrt (p.a^2 + p.b^2) →
      q.x = x_max :=
sorry

/-- The product of x and y coordinates has a maximum value --/
theorem product_max_value (p : EllipsePoint) :
  ∃ m : ℝ, m = p.a * p.b / 2 ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b → q.x * q.y ≤ m :=
sorry

/-- The product of x and y coordinates reaches its maximum when x has a specific value --/
theorem product_max_x (p : EllipsePoint) :
  ∃ x_max : ℝ, x_max = p.a * Real.sqrt 2 / 2 ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b ∧ q.x * q.y = p.a * p.b / 2 →
      q.x = x_max :=
sorry

end NUMINAMATH_CALUDE_sum_max_value_sum_max_x_product_max_value_product_max_x_l2522_252202


namespace NUMINAMATH_CALUDE_distance_between_centers_l2522_252220

/-- The distance between the centers of inscribed and circumscribed circles in a right triangle -/
theorem distance_between_centers (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let r := (a * b) / (2 * s)
  let x_i := r
  let y_i := r
  let x_o := c / 2
  let y_o := 0
  Real.sqrt ((x_o - x_i)^2 + (y_o - y_i)^2) = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_centers_l2522_252220


namespace NUMINAMATH_CALUDE_female_worker_ants_l2522_252293

theorem female_worker_ants (total_ants : ℕ) (worker_ratio : ℚ) (male_ratio : ℚ) : 
  total_ants = 110 →
  worker_ratio = 1/2 →
  male_ratio = 1/5 →
  ⌊(total_ants : ℚ) * worker_ratio * (1 - male_ratio)⌋ = 44 := by
sorry

end NUMINAMATH_CALUDE_female_worker_ants_l2522_252293


namespace NUMINAMATH_CALUDE_polynomial_shift_representation_l2522_252224

theorem polynomial_shift_representation (f : Polynomial ℝ) (x₀ : ℝ) :
  ∃! g : Polynomial ℝ, ∀ x, f.eval x = g.eval (x - x₀) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_shift_representation_l2522_252224


namespace NUMINAMATH_CALUDE_correct_calculation_l2522_252207

theorem correct_calculation (x : ℝ) : 3 * x = 135 → x / 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2522_252207


namespace NUMINAMATH_CALUDE_fruit_lovers_count_l2522_252213

/-- The number of people who like apple -/
def apple_lovers : ℕ := 40

/-- The number of people who like orange and mango but dislike apple -/
def orange_mango_lovers : ℕ := 7

/-- The number of people who like mango and apple and dislike orange -/
def apple_mango_lovers : ℕ := 10

/-- The total number of people who like apple -/
def total_apple_lovers : ℕ := 47

/-- The number of people who like all three fruits (apple, orange, and mango) -/
def all_fruit_lovers : ℕ := 3

theorem fruit_lovers_count : 
  apple_lovers + (apple_mango_lovers - all_fruit_lovers) + all_fruit_lovers = total_apple_lovers :=
by sorry

end NUMINAMATH_CALUDE_fruit_lovers_count_l2522_252213


namespace NUMINAMATH_CALUDE_cyclist_distance_l2522_252274

/-- Cyclist's travel problem -/
theorem cyclist_distance :
  ∀ (v t : ℝ),
  v > 0 →
  t > 0 →
  (v + 1) * (3/4 * t) = v * t →
  (v - 1) * (t + 3) = v * t →
  v * t = 18 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_distance_l2522_252274


namespace NUMINAMATH_CALUDE_quadratic_points_relation_l2522_252226

-- Define the quadratic function
def f (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Define the points A, B, and C
def A (m : ℝ) : ℝ × ℝ := (-4, f (-4) m)
def B (m : ℝ) : ℝ × ℝ := (0, f 0 m)
def C (m : ℝ) : ℝ × ℝ := (3, f 3 m)

-- State the theorem
theorem quadratic_points_relation (m : ℝ) :
  (B m).2 < (C m).2 ∧ (C m).2 < (A m).2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_relation_l2522_252226


namespace NUMINAMATH_CALUDE_regular_rate_is_three_l2522_252267

/-- Represents a worker's pay structure and hours worked -/
structure PayStructure where
  regularRate : ℝ
  overtimeHours : ℝ
  totalPay : ℝ

/-- Calculates the total pay for a given pay structure -/
def calculateTotalPay (p : PayStructure) : ℝ :=
  40 * p.regularRate + p.overtimeHours * (2 * p.regularRate)

/-- Theorem stating that given the conditions, the regular rate is $3 per hour -/
theorem regular_rate_is_three (p : PayStructure) 
  (h1 : p.overtimeHours = 8)
  (h2 : p.totalPay = 168)
  (h3 : calculateTotalPay p = p.totalPay) : 
  p.regularRate = 3 := by
  sorry


end NUMINAMATH_CALUDE_regular_rate_is_three_l2522_252267


namespace NUMINAMATH_CALUDE_course_selection_schemes_l2522_252276

/-- The number of elective courses in each category (physical education and art) -/
def n : ℕ := 4

/-- The total number of different course selection schemes -/
def total_schemes : ℕ := (n * n) + (n * (n - 1) * n) / 2

/-- Theorem stating that the total number of course selection schemes is 64 -/
theorem course_selection_schemes :
  total_schemes = 64 := by sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l2522_252276


namespace NUMINAMATH_CALUDE_hotel_room_charge_difference_l2522_252251

theorem hotel_room_charge_difference (G : ℝ) (h1 : G > 0) : 
  let R := 1.5000000000000002 * G
  let P := 0.6 * R
  (G - P) / G * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_hotel_room_charge_difference_l2522_252251


namespace NUMINAMATH_CALUDE_range_of_f_range_of_m_l2522_252243

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1| - 1
def g (x : ℝ) : ℝ := -|x + 1| - 4

-- Theorem 1: Range of x for which f(x) ≤ 1
theorem range_of_f (x : ℝ) : f x ≤ 1 ↔ x ∈ Set.Icc (-1) 3 := by sorry

-- Theorem 2: Range of m for which f(x) - g(x) ≥ m + 1 holds for all x
theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ∈ Set.Iic 4 := by sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_m_l2522_252243


namespace NUMINAMATH_CALUDE_intersection_slope_inequality_l2522_252256

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := 3/2 * x^2 - (6+a)*x + 2*a * f x

noncomputable def g (x : ℝ) : ℝ := f x / (deriv f x)

theorem intersection_slope_inequality (a k x₁ x₂ : ℝ) (h₁ : a > 0) (h₂ : x₁ < x₂) 
  (h₃ : ∃ y₁ y₂, (k * x₁ + y₁ = deriv g x₁) ∧ (k * x₂ + y₂ = deriv g x₂)) :
  x₁ < 1/k ∧ 1/k < x₂ := by sorry

end NUMINAMATH_CALUDE_intersection_slope_inequality_l2522_252256


namespace NUMINAMATH_CALUDE_parallelogram_angles_l2522_252290

theorem parallelogram_angles (A B C D : Real) : 
  -- ABCD is a parallelogram
  (A + C = 180) →
  (B + D = 180) →
  -- ∠B - ∠A = 30°
  (B - A = 30) →
  -- Prove that ∠A = 75°, ∠B = 105°, ∠C = 75°, and ∠D = 105°
  (A = 75 ∧ B = 105 ∧ C = 75 ∧ D = 105) := by
sorry

end NUMINAMATH_CALUDE_parallelogram_angles_l2522_252290


namespace NUMINAMATH_CALUDE_checkerboard_inner_probability_l2522_252235

/-- The size of one side of the square checkerboard -/
def boardSize : ℕ := 10

/-- The total number of squares on the checkerboard -/
def totalSquares : ℕ := boardSize * boardSize

/-- The number of squares on the perimeter of the checkerboard -/
def perimeterSquares : ℕ := 4 * boardSize - 4

/-- The number of squares not touching the outer edge -/
def innerSquares : ℕ := totalSquares - perimeterSquares

/-- The probability of choosing a square not touching the outer edge -/
def innerProbability : ℚ := innerSquares / totalSquares

theorem checkerboard_inner_probability :
  innerProbability = 16 / 25 := by sorry

end NUMINAMATH_CALUDE_checkerboard_inner_probability_l2522_252235


namespace NUMINAMATH_CALUDE_cos_18_cos_42_minus_cos_72_sin_42_l2522_252263

theorem cos_18_cos_42_minus_cos_72_sin_42 :
  Real.cos (18 * π / 180) * Real.cos (42 * π / 180) - 
  Real.cos (72 * π / 180) * Real.sin (42 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_cos_42_minus_cos_72_sin_42_l2522_252263


namespace NUMINAMATH_CALUDE_dog_turn_point_sum_cat_dog_problem_l2522_252211

/-- The point where the dog starts moving away from the cat -/
def dog_turn_point (cat_x cat_y : ℚ) (dog_line_slope dog_line_intercept : ℚ) : ℚ × ℚ :=
  sorry

/-- Theorem stating that the sum of coordinates of the turn point is 243/68 -/
theorem dog_turn_point_sum (cat_x cat_y : ℚ) (dog_line_slope dog_line_intercept : ℚ) :
  let (c, d) := dog_turn_point cat_x cat_y dog_line_slope dog_line_intercept
  c + d = 243/68 := by sorry

/-- Main theorem proving the specific case in the problem -/
theorem cat_dog_problem :
  let (c, d) := dog_turn_point 15 12 (-4) 15
  c + d = 243/68 := by sorry

end NUMINAMATH_CALUDE_dog_turn_point_sum_cat_dog_problem_l2522_252211


namespace NUMINAMATH_CALUDE_bus_assignment_count_l2522_252244

def num_buses : ℕ := 6
def num_destinations : ℕ := 4
def num_restricted_buses : ℕ := 2

def choose (n k : ℕ) : ℕ := Nat.choose n k

def arrange (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem bus_assignment_count : 
  choose num_destinations 1 * arrange (num_buses - num_restricted_buses) (num_destinations - 1) = 240 := by
  sorry

end NUMINAMATH_CALUDE_bus_assignment_count_l2522_252244


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l2522_252232

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 5)
  (sum_cubes : a^3 + b^3 + c^3 = 10) :
  a^4 + b^4 + c^4 = 68/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l2522_252232


namespace NUMINAMATH_CALUDE_triangle_is_acute_l2522_252215

theorem triangle_is_acute (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) :
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2 := by
  sorry

#check triangle_is_acute

end NUMINAMATH_CALUDE_triangle_is_acute_l2522_252215


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l2522_252262

theorem smallest_positive_difference (a b : ℤ) (h : 17 * a + 6 * b = 13) :
  ∃ (k : ℤ), k > 0 ∧ a - b = k ∧ ∀ (m : ℤ), m > 0 ∧ (∃ (x y : ℤ), 17 * x + 6 * y = 13 ∧ x - y = m) → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l2522_252262


namespace NUMINAMATH_CALUDE_function_properties_l2522_252242

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * log x - b * x^2 + 1

theorem function_properties (a b : ℝ) :
  (∀ x > 0, deriv (f a b) x = 3 → f a b 1 = 1/2) →
  (a = 4 ∧ b = 1/2) ∧
  (∀ x ∈ Set.Icc (1/ℯ) (ℯ^2), f 4 (1/2) x ≤ 4 * log 2 - 1) ∧
  (∃ x ∈ Set.Icc (1/ℯ) (ℯ^2), f 4 (1/2) x = 4 * log 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2522_252242


namespace NUMINAMATH_CALUDE_simplify_trig_expression_1_simplify_trig_expression_2_l2522_252204

-- Part 1
theorem simplify_trig_expression_1 :
  (Real.sin (35 * π / 180))^2 - (1/2) = -Real.cos (10 * π / 180) * Real.cos (80 * π / 180) := by
  sorry

-- Part 2
theorem simplify_trig_expression_2 (α : ℝ) :
  (1 / Real.tan (α/2) - Real.tan (α/2)) * ((1 - Real.cos (2*α)) / Real.sin (2*α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_1_simplify_trig_expression_2_l2522_252204


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l2522_252214

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_f_odd_and_decreasing_l2522_252214


namespace NUMINAMATH_CALUDE_emma_age_when_sister_is_56_l2522_252295

/-- Emma's current age -/
def emma_age : ℕ := 7

/-- Age difference between Emma and her sister -/
def age_difference : ℕ := 9

/-- Age of Emma's sister when the problem is solved -/
def sister_future_age : ℕ := 56

/-- Emma's age when her sister reaches the future age -/
def emma_future_age : ℕ := emma_age + (sister_future_age - (emma_age + age_difference))

theorem emma_age_when_sister_is_56 : emma_future_age = 47 := by
  sorry

end NUMINAMATH_CALUDE_emma_age_when_sister_is_56_l2522_252295


namespace NUMINAMATH_CALUDE_water_intake_increase_l2522_252289

theorem water_intake_increase (current : ℕ) (recommended : ℕ) : 
  current = 15 → recommended = 21 → 
  (((recommended - current) : ℚ) / current) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_water_intake_increase_l2522_252289


namespace NUMINAMATH_CALUDE_double_line_chart_capabilities_l2522_252284

/-- Represents a data set -/
structure DataSet where
  values : List ℝ

/-- Represents a double line chart -/
structure DoubleLineChart where
  dataset1 : DataSet
  dataset2 : DataSet

/-- Function to calculate changes in a dataset -/
def calculateChanges (ds : DataSet) : List ℝ := sorry

/-- Function to analyze differences between two datasets -/
def analyzeDifferences (ds1 ds2 : DataSet) : List ℝ := sorry

theorem double_line_chart_capabilities (dlc : DoubleLineChart) :
  (∃ (changes1 changes2 : List ℝ), 
     changes1 = calculateChanges dlc.dataset1 ∧ 
     changes2 = calculateChanges dlc.dataset2) ∧
  (∃ (differences : List ℝ), 
     differences = analyzeDifferences dlc.dataset1 dlc.dataset2) := by
  sorry

end NUMINAMATH_CALUDE_double_line_chart_capabilities_l2522_252284


namespace NUMINAMATH_CALUDE_inscribed_circle_tangent_triangle_area_l2522_252264

/-- Given a right triangle with hypotenuse c, area T, and an inscribed circle of radius ρ,
    the area of the triangle formed by the points where the inscribed circle touches the sides
    of the right triangle is equal to (ρ/c) * T. -/
theorem inscribed_circle_tangent_triangle_area
  (c T ρ : ℝ)
  (h_positive : c > 0 ∧ T > 0 ∧ ρ > 0)
  (h_right_triangle : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2)
  (h_area : T = (a + b + c) * ρ / 2)
  (h_inscribed : ρ = T / (a + b + c)) :
  (ρ / c) * T = (area_of_tangent_triangle : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_tangent_triangle_area_l2522_252264


namespace NUMINAMATH_CALUDE_integer_root_values_l2522_252225

def polynomial (a x : ℤ) : ℤ := x^3 - 2*x^2 + a*x + 8

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial a x = 0

theorem integer_root_values : 
  {a : ℤ | has_integer_root a} = {-49, -47, -22, -10, -7, 4, 9, 16} := by sorry

end NUMINAMATH_CALUDE_integer_root_values_l2522_252225


namespace NUMINAMATH_CALUDE_problem_statement_l2522_252217

/-- The set D of positive real pairs (x₁, x₂) that sum to k -/
def D (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = k}

theorem problem_statement (k : ℝ) (hk : k > 0) :
  (∀ p ∈ D k, 0 < p.1 * p.2 ∧ p.1 * p.2 ≤ k^2 / 4) ∧
  (k ≥ 1 → ∀ p ∈ D k, (1 / p.1 - p.1) * (1 / p.2 - p.2) ≤ (k / 2 - 2 / k)^2) ∧
  (∀ p ∈ D k, (1 / p.1 - p.1) * (1 / p.2 - p.2) ≥ (k / 2 - 2 / k)^2 ↔ 0 < k^2 ∧ k^2 ≤ 4 * Real.sqrt 5 - 8) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2522_252217


namespace NUMINAMATH_CALUDE_complex_square_problem_l2522_252287

theorem complex_square_problem (a b : ℝ) (i : ℂ) :
  i^2 = -1 →
  a + i = 2 - b*i →
  (a + b*i)^2 = 3 - 4*i := by
sorry

end NUMINAMATH_CALUDE_complex_square_problem_l2522_252287


namespace NUMINAMATH_CALUDE_strictly_decreasing_implies_inequality_odd_function_property_l2522_252241

-- Define a function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Statement 1
theorem strictly_decreasing_implies_inequality (h : ∀ x y, x < y → f x > f y) : f (-4) > f 4 := by
  sorry

-- Statement 2
theorem odd_function_property (h : ∀ x, f (-x) = -f x) : f (-4) + f 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_strictly_decreasing_implies_inequality_odd_function_property_l2522_252241


namespace NUMINAMATH_CALUDE_remaining_milk_average_price_l2522_252277

/-- Calculates the average price of remaining milk packets after returning some packets. -/
theorem remaining_milk_average_price
  (total_packets : ℕ)
  (initial_avg_price : ℚ)
  (returned_packets : ℕ)
  (returned_avg_price : ℚ)
  (h1 : total_packets = 5)
  (h2 : initial_avg_price = 20/100)
  (h3 : returned_packets = 2)
  (h4 : returned_avg_price = 32/100)
  : (total_packets * initial_avg_price - returned_packets * returned_avg_price) / (total_packets - returned_packets) = 12/100 := by
  sorry

end NUMINAMATH_CALUDE_remaining_milk_average_price_l2522_252277


namespace NUMINAMATH_CALUDE_absolute_value_equality_implies_inequality_l2522_252296

theorem absolute_value_equality_implies_inequality (m : ℝ) : 
  |m - 9| = 9 - m → m ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equality_implies_inequality_l2522_252296


namespace NUMINAMATH_CALUDE_units_digit_47_power_47_l2522_252283

theorem units_digit_47_power_47 : 47^47 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_47_power_47_l2522_252283


namespace NUMINAMATH_CALUDE_fault_line_movement_l2522_252285

/-- Represents the movement of a fault line over two years -/
structure FaultLineMovement where
  past_year : ℝ
  year_before : ℝ

/-- Calculates the total movement of a fault line over two years -/
def total_movement (f : FaultLineMovement) : ℝ :=
  f.past_year + f.year_before

/-- Theorem: The total movement of the fault line is 6.50 inches -/
theorem fault_line_movement :
  let f : FaultLineMovement := { past_year := 1.25, year_before := 5.25 }
  total_movement f = 6.50 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_movement_l2522_252285


namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l2522_252212

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l2522_252212


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_specific_root_condition_l2522_252216

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + 4*x + m - 1
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) → m ≤ 5 :=
by sorry

theorem specific_root_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + 4*x + m - 1
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 2*(x₁ + x₂) + x₁*x₂ + 10 = 0) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_specific_root_condition_l2522_252216


namespace NUMINAMATH_CALUDE_factorization_difference_l2522_252265

theorem factorization_difference (y : ℝ) (a b : ℤ) : 
  3 * y^2 - y - 24 = (3*y + a) * (y + b) → a - b = 11 := by
sorry

end NUMINAMATH_CALUDE_factorization_difference_l2522_252265


namespace NUMINAMATH_CALUDE_ab_sum_problem_l2522_252281

theorem ab_sum_problem (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (ha_upper : a < 15) (hb_upper : b < 15) 
  (h_eq : a + b + a * b = 119) : a + b = 18 ∨ a + b = 19 := by
  sorry

end NUMINAMATH_CALUDE_ab_sum_problem_l2522_252281


namespace NUMINAMATH_CALUDE_crayons_in_box_l2522_252208

def initial_crayons : ℕ := 7
def added_crayons : ℕ := 6

theorem crayons_in_box : initial_crayons + added_crayons = 13 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_box_l2522_252208


namespace NUMINAMATH_CALUDE_alex_zhu_same_section_probability_l2522_252236

def total_students : ℕ := 100
def selected_students : ℕ := 60
def num_sections : ℕ := 3
def students_per_section : ℕ := 20

theorem alex_zhu_same_section_probability :
  (3 : ℚ) * (Nat.choose 58 18) / (Nat.choose 60 20) = 19 / 165 := by
  sorry

end NUMINAMATH_CALUDE_alex_zhu_same_section_probability_l2522_252236


namespace NUMINAMATH_CALUDE_custom_operation_value_l2522_252273

/-- Custom operation * for non-zero integers -/
def star (a b : ℤ) : ℚ :=
  1 / a + 1 / b

theorem custom_operation_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 10) (prod_eq : a * b = 24) : 
  star a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_value_l2522_252273


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l2522_252260

/-- The distance between Stockholm and Uppsala on a map in centimeters -/
def map_distance : ℝ := 45

/-- The scale of the map, representing how many kilometers in reality one centimeter on the map represents -/
def map_scale : ℝ := 10

/-- The actual distance between Stockholm and Uppsala in kilometers -/
def actual_distance : ℝ := map_distance * map_scale

theorem stockholm_uppsala_distance :
  actual_distance = 450 :=
by sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l2522_252260


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2522_252291

/-- If (3sin(α) + 2cos(α)) / (2sin(α) - cos(α)) = 8/3, then tan(α + π/4) = -3 -/
theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h : (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8/3) : 
  Real.tan (α + π/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2522_252291


namespace NUMINAMATH_CALUDE_letterbox_strip_height_calculation_l2522_252234

/-- Represents a screen with width, height, and diagonal measurements -/
structure Screen where
  width : ℝ
  height : ℝ
  diagonal : ℝ

/-- Represents an aspect ratio as a pair of numbers -/
structure AspectRatio where
  horizontal : ℝ
  vertical : ℝ

/-- Calculates the height of letterbox strips when showing a movie on a TV -/
def letterboxStripHeight (tv : Screen) (movieRatio : AspectRatio) : ℝ :=
  sorry

theorem letterbox_strip_height_calculation 
  (tv : Screen)
  (movieRatio : AspectRatio)
  (h1 : tv.diagonal = 27)
  (h2 : tv.width / tv.height = 4 / 3)
  (h3 : movieRatio.horizontal / movieRatio.vertical = 2 / 1)
  (h4 : tv.width^2 + tv.height^2 = tv.diagonal^2) :
  letterboxStripHeight tv movieRatio = 2.7 := by
  sorry

end NUMINAMATH_CALUDE_letterbox_strip_height_calculation_l2522_252234


namespace NUMINAMATH_CALUDE_anita_blueberry_cartons_l2522_252268

/-- Represents the number of cartons of berries in Anita's berry cobbler problem -/
structure BerryCobbler where
  total : ℕ
  strawberries : ℕ
  to_buy : ℕ

/-- Calculates the number of blueberry cartons Anita has -/
def blueberry_cartons (bc : BerryCobbler) : ℕ :=
  bc.total - bc.strawberries - bc.to_buy

/-- Theorem stating that Anita has 9 cartons of blueberries -/
theorem anita_blueberry_cartons :
  ∀ (bc : BerryCobbler),
    bc.total = 26 → bc.strawberries = 10 → bc.to_buy = 7 →
    blueberry_cartons bc = 9 := by
  sorry

end NUMINAMATH_CALUDE_anita_blueberry_cartons_l2522_252268


namespace NUMINAMATH_CALUDE_percentage_free_lunch_l2522_252254

/-- Proves that 40% of students receive a free lunch given the specified conditions --/
theorem percentage_free_lunch (total_students : ℕ) (total_cost : ℚ) (paying_price : ℚ) :
  total_students = 50 →
  total_cost = 210 →
  paying_price = 7 →
  (∃ (paying_students : ℕ), paying_students * paying_price = total_cost) →
  (total_students - (total_cost / paying_price : ℚ)) / total_students = 2/5 := by
  sorry

#check percentage_free_lunch

end NUMINAMATH_CALUDE_percentage_free_lunch_l2522_252254


namespace NUMINAMATH_CALUDE_circumcenter_rational_l2522_252203

-- Define a triangle with rational coordinates
structure RationalTriangle where
  a : ℚ × ℚ
  b : ℚ × ℚ
  c : ℚ × ℚ

-- Define the center of the circumscribed circle
def circumcenter (t : RationalTriangle) : ℚ × ℚ :=
  sorry

-- Theorem statement
theorem circumcenter_rational (t : RationalTriangle) :
  ∃ (x y : ℚ), circumcenter t = (x, y) :=
sorry

end NUMINAMATH_CALUDE_circumcenter_rational_l2522_252203


namespace NUMINAMATH_CALUDE_min_shoeing_time_for_scenario_l2522_252221

/-- The minimum time needed for blacksmiths to shoe horses -/
def min_shoeing_time (blacksmiths horses : ℕ) (time_per_shoe : ℕ) : ℕ :=
  let total_shoes := horses * 4
  let total_time := total_shoes * time_per_shoe
  (total_time + blacksmiths - 1) / blacksmiths

/-- Theorem stating the minimum time needed for the given scenario -/
theorem min_shoeing_time_for_scenario :
  min_shoeing_time 48 60 5 = 25 := by
sorry

#eval min_shoeing_time 48 60 5

end NUMINAMATH_CALUDE_min_shoeing_time_for_scenario_l2522_252221


namespace NUMINAMATH_CALUDE_sherman_weekly_driving_time_l2522_252252

-- Define the daily commute time in minutes
def daily_commute : ℕ := 30 + 30

-- Define the number of workdays in a week
def workdays : ℕ := 5

-- Define the weekend driving time in hours
def weekend_driving : ℕ := 2 * 2

-- Theorem statement
theorem sherman_weekly_driving_time :
  (workdays * daily_commute) / 60 + weekend_driving = 9 := by
  sorry

end NUMINAMATH_CALUDE_sherman_weekly_driving_time_l2522_252252


namespace NUMINAMATH_CALUDE_money_division_l2522_252206

/-- Proves that the total amount of money divided among A, B, and C is 980,
    given the specified conditions. -/
theorem money_division (a b c : ℕ) : 
  b = 290 →            -- B's share is 290
  a = b + 40 →         -- A has 40 more than B
  c = a + 30 →         -- C has 30 more than A
  a + b + c = 980 :=   -- Total amount is 980
by
  sorry


end NUMINAMATH_CALUDE_money_division_l2522_252206


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2522_252272

/-- 
Given:
- Ingrid starts with n chocolates
- Jin receives 1/3 of Ingrid's chocolates
- Jin gives 8 chocolates to Brian
- Jin eats half of her remaining chocolates
- Jin ends up with 5 chocolates

Prove: n = 54
-/
theorem chocolate_distribution (n : ℕ) : 
  (n / 3 - 8) / 2 = 5 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2522_252272


namespace NUMINAMATH_CALUDE_remaining_bag_weight_l2522_252261

def bag_weights : List ℕ := [15, 16, 18, 19, 20, 31]

def is_valid_partition (partition : List ℕ × List ℕ) : Prop :=
  let (group1, group2) := partition
  group1.length + group2.length = 5 ∧
  group1.sum = 2 * group2.sum ∧
  (∀ w ∈ group1, w ∈ bag_weights) ∧
  (∀ w ∈ group2, w ∈ bag_weights) ∧
  (∀ w ∈ group1, w ∉ group2) ∧
  (∀ w ∈ group2, w ∉ group1)

theorem remaining_bag_weight :
  ∃ (partition : List ℕ × List ℕ), is_valid_partition partition →
  bag_weights.sum - (partition.1.sum + partition.2.sum) = 20 :=
sorry

end NUMINAMATH_CALUDE_remaining_bag_weight_l2522_252261


namespace NUMINAMATH_CALUDE_quadratic_from_means_l2522_252288

theorem quadratic_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 8) 
  (h_geometric : Real.sqrt (a * b) = 12) : 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = a ∨ x = b) := by
sorry

end NUMINAMATH_CALUDE_quadratic_from_means_l2522_252288


namespace NUMINAMATH_CALUDE_max_value_of_function_l2522_252298

theorem max_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  x * (1 - 2*x) ≤ 1/8 ∧ ∃ x₀, 0 < x₀ ∧ x₀ < 1/2 ∧ x₀ * (1 - 2*x₀) = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2522_252298


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l2522_252210

/-- Represents the distance on a map in centimeters -/
def map_distance : ℝ := 65

/-- Represents the scale factor of the map (km per cm) -/
def scale_factor : ℝ := 20

/-- Calculates the actual distance in kilometers given the map distance and scale factor -/
def actual_distance (map_dist : ℝ) (scale : ℝ) : ℝ := map_dist * scale

theorem stockholm_uppsala_distance :
  actual_distance map_distance scale_factor = 1300 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l2522_252210


namespace NUMINAMATH_CALUDE_no_intersection_implies_k_equals_one_l2522_252278

theorem no_intersection_implies_k_equals_one (k : ℕ+) :
  (∀ x y : ℝ, x^2 + y^2 = k^2 → x * y ≠ k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_implies_k_equals_one_l2522_252278


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2522_252258

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x ≥ 2 → x ≠ 1) ∧
  (∃ x : ℝ, x ≠ 1 ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2522_252258


namespace NUMINAMATH_CALUDE_special_polygon_sum_angles_l2522_252201

/-- A polygon where 3 diagonals can be drawn from one vertex -/
structure SpecialPolygon where
  /-- The number of diagonals that can be drawn from one vertex -/
  diagonals_from_vertex : ℕ
  /-- The condition that 3 diagonals can be drawn from one vertex -/
  diag_condition : diagonals_from_vertex = 3

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- Theorem: The sum of interior angles of a SpecialPolygon is 720° -/
theorem special_polygon_sum_angles (p : SpecialPolygon) : 
  sum_interior_angles (p.diagonals_from_vertex + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_sum_angles_l2522_252201
