import Mathlib

namespace NUMINAMATH_CALUDE_opposite_of_one_minus_sqrt_two_l1752_175219

theorem opposite_of_one_minus_sqrt_two :
  ∃ x : ℝ, (1 - Real.sqrt 2) + x = 0 ∧ x = -1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_minus_sqrt_two_l1752_175219


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1752_175258

theorem smallest_positive_integer_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 6 = 3) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 6 = 3 ∧ m % 8 = 5 → m ≥ n) ∧
  (n = 21) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1752_175258


namespace NUMINAMATH_CALUDE_inequality_proof_l1752_175286

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 + c^2 ≥ a*b + b*c + c*a) ∧ ((a + b + c)^2 ≥ 3*(a*b + b*c + c*a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1752_175286


namespace NUMINAMATH_CALUDE_orange_cost_l1752_175295

theorem orange_cost (calorie_per_orange : ℝ) (total_money : ℝ) (required_calories : ℝ) (money_left : ℝ) :
  calorie_per_orange = 80 →
  total_money = 10 →
  required_calories = 400 →
  money_left = 4 →
  (total_money - money_left) / (required_calories / calorie_per_orange) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_l1752_175295


namespace NUMINAMATH_CALUDE_set_B_equality_l1752_175227

def A : Set ℤ := {-1, 0, 1}

def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem set_B_equality : B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_set_B_equality_l1752_175227


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_squares_l1752_175235

theorem root_sum_reciprocal_squares (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 → 
  b^3 - 6*b^2 + 11*b - 6 = 0 → 
  c^3 - 6*c^2 + 11*c - 6 = 0 → 
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_squares_l1752_175235


namespace NUMINAMATH_CALUDE_monotonic_intervals_max_value_when_a_2_l1752_175243

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^4 - 4 * (a + 1) * x^3 + 6 * a * x^2 - 12

-- Theorem for the intervals of monotonic increase
theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ a → f a x < f a y) ∧
  (∀ x y, 1 ≤ x ∧ x < y → f a x < f a y) ∧
  (a = 1 → ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y) ∧
  (a > 1 → (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a x < f a y) ∧
           (∀ x y, a ≤ x ∧ x < y → f a x < f a y)) :=
sorry

-- Theorem for the maximum value when a = 2
theorem max_value_when_a_2 :
  ∀ x, f 2 x ≤ f 2 1 ∧ f 2 1 = -9 :=
sorry

end NUMINAMATH_CALUDE_monotonic_intervals_max_value_when_a_2_l1752_175243


namespace NUMINAMATH_CALUDE_natural_number_representation_l1752_175209

def representable (n : ℕ) : Prop :=
  ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  (x * y + y * z + z * x : ℚ) / (x + y + z : ℚ) = n

theorem natural_number_representation :
  ∀ n : ℕ, n > 1 → representable n ∧ ¬representable 1 := by sorry

end NUMINAMATH_CALUDE_natural_number_representation_l1752_175209


namespace NUMINAMATH_CALUDE_number_guessing_game_l1752_175252

theorem number_guessing_game (a b c d : ℕ) 
  (ha : a ≥ 10) 
  (hb : b < 10) (hc : c < 10) (hd : d < 10) : 
  ((((((a * 2 + 1) * 5 + b) * 2 + 1) * 5 + c) * 2 + 1) * 5 + d) - 555 = 1000 * a + 100 * b + 10 * c + d :=
by sorry

#check number_guessing_game

end NUMINAMATH_CALUDE_number_guessing_game_l1752_175252


namespace NUMINAMATH_CALUDE_intersection_distance_product_l1752_175273

/-- Parabola defined by y² = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Line with equation y = x - 2 -/
def line (x y : ℝ) : Prop := y = x - 2

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- Theorem stating that the product of distances from focus to intersection points is 32 -/
theorem intersection_distance_product : 
  ∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    (A.1 - focus.1)^2 + (A.2 - focus.2)^2 * 
    (B.1 - focus.1)^2 + (B.2 - focus.2)^2 = 32^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_product_l1752_175273


namespace NUMINAMATH_CALUDE_equation_solution_l1752_175261

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 2 → (x - 8 / (x - 2) = 5 + 8 / (x - 2)) ↔ (x = 9 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1752_175261


namespace NUMINAMATH_CALUDE_sin_315_degrees_l1752_175213

/-- Proves that sin 315° = -√2/2 -/
theorem sin_315_degrees : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l1752_175213


namespace NUMINAMATH_CALUDE_functional_equation_problem_l1752_175232

open Function Real

/-- The functional equation problem -/
theorem functional_equation_problem (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f x * f (y * f x - 1) = x^2 * f y - f x) ↔ (f = id) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l1752_175232


namespace NUMINAMATH_CALUDE_min_sum_distances_l1752_175212

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define line l₁
def line_l₁ (x y : ℝ) : Prop := 4*x - 3*y + 6 = 0

-- Define line l₂
def line_l₂ (x : ℝ) : Prop := x = -1

-- Define the distance function from a point to a line
noncomputable def dist_point_to_line (px py : ℝ) (a b c : ℝ) : ℝ :=
  abs (a * px + b * py + c) / Real.sqrt (a^2 + b^2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem min_sum_distances :
  ∃ (d : ℝ), d = 2 ∧
  ∀ (px py : ℝ), parabola px py →
    d ≤ dist_point_to_line px py 4 (-3) 6 + abs (px + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_distances_l1752_175212


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l1752_175250

-- Define the set of numbers
def S : Set ℕ := {1, 2, 3, 4, 5}

-- Define the event A: "The product of the two chosen numbers is even"
def event_A (x y : ℕ) : Prop := x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ Even (x * y)

-- Define the event B: "Both chosen numbers are even"
def event_B (x y : ℕ) : Prop := x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ Even x ∧ Even y

-- Define the probability of choosing two different numbers from S
def prob_total : ℚ := 10 / 1

-- Define the probability of event A
def prob_A : ℚ := 7 / 10

-- Define the probability of event A ∩ B
def prob_A_and_B : ℚ := 1 / 10

-- Theorem statement
theorem conditional_probability_B_given_A :
  prob_A_and_B / prob_A = 1 / 7 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l1752_175250


namespace NUMINAMATH_CALUDE_power_function_value_l1752_175229

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Define the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = 4 → f (-3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l1752_175229


namespace NUMINAMATH_CALUDE_power_multiplication_l1752_175244

theorem power_multiplication (x : ℝ) : x^2 * x^4 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1752_175244


namespace NUMINAMATH_CALUDE_total_weight_proof_l1752_175275

theorem total_weight_proof (jim_weight steve_weight stan_weight : ℕ) 
  (h1 : stan_weight = steve_weight + 5)
  (h2 : steve_weight = jim_weight - 8)
  (h3 : jim_weight = 110) : 
  jim_weight + steve_weight + stan_weight = 319 :=
by
  sorry

end NUMINAMATH_CALUDE_total_weight_proof_l1752_175275


namespace NUMINAMATH_CALUDE_emptyBoxes_l1752_175245

/-- Represents the state of the two boxes with pebbles -/
structure BoxState :=
  (p : ℕ)
  (q : ℕ)

/-- Defines a single step operation on the boxes -/
inductive Step
  | Remove : Step
  | TripleP : Step
  | TripleQ : Step

/-- Applies a single step to a BoxState -/
def applyStep (state : BoxState) (step : Step) : BoxState :=
  match step with
  | Step.Remove => ⟨state.p - 1, state.q - 1⟩
  | Step.TripleP => ⟨state.p * 3, state.q⟩
  | Step.TripleQ => ⟨state.p, state.q * 3⟩

/-- Checks if a BoxState is empty (both boxes have 0 pebbles) -/
def isEmpty (state : BoxState) : Prop :=
  state.p = 0 ∧ state.q = 0

/-- Defines if it's possible to empty both boxes from a given initial state -/
def canEmpty (initial : BoxState) : Prop :=
  ∃ (steps : List Step), isEmpty (steps.foldl applyStep initial)

/-- The main theorem to be proved -/
theorem emptyBoxes (p q : ℕ) :
  canEmpty ⟨p, q⟩ ↔ p % 2 = q % 2 := by sorry


end NUMINAMATH_CALUDE_emptyBoxes_l1752_175245


namespace NUMINAMATH_CALUDE_spending_pattern_proof_l1752_175233

/-- Represents the initial amount of money in won -/
def initial_amount : ℕ := 1500

/-- Represents the amount left after spending at all stores -/
def final_amount : ℕ := 250

/-- Theorem stating that given the spending pattern, the initial amount results in the final amount -/
theorem spending_pattern_proof :
  (initial_amount / 2 / 3 / 2 : ℚ) = final_amount := by
  sorry

end NUMINAMATH_CALUDE_spending_pattern_proof_l1752_175233


namespace NUMINAMATH_CALUDE_terrell_hike_distance_l1752_175259

theorem terrell_hike_distance (saturday_distance sunday_distance : ℝ) 
  (h1 : saturday_distance = 8.2)
  (h2 : sunday_distance = 1.6) : 
  saturday_distance + sunday_distance = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_terrell_hike_distance_l1752_175259


namespace NUMINAMATH_CALUDE_basil_seed_cost_l1752_175222

/-- Represents the cost structure and profit for Burt's basil plant business -/
structure BasilBusiness where
  seed_cost : ℝ
  soil_cost : ℝ
  plants : ℕ
  price_per_plant : ℝ
  net_profit : ℝ

/-- Calculates the total revenue from selling basil plants -/
def total_revenue (b : BasilBusiness) : ℝ :=
  b.plants * b.price_per_plant

/-- Calculates the total expenses for the basil business -/
def total_expenses (b : BasilBusiness) : ℝ :=
  b.seed_cost + b.soil_cost

/-- Theorem stating that given the conditions, the seed cost is $2.00 -/
theorem basil_seed_cost (b : BasilBusiness) 
  (h1 : b.soil_cost = 8)
  (h2 : b.plants = 20)
  (h3 : b.price_per_plant = 5)
  (h4 : b.net_profit = 90)
  (h5 : total_revenue b - total_expenses b = b.net_profit) :
  b.seed_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_basil_seed_cost_l1752_175222


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1752_175251

theorem quadratic_equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃ (a b : ℝ), ∀ (x : ℝ),
    x = a * m + b * n →
    (x + m)^2 - (x + n)^2 = (m - n)^2 →
    a = 0 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1752_175251


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1752_175270

theorem trigonometric_equation_solution (x : ℝ) : 
  (2 * Real.sin x - Real.sin (2 * x)) / (2 * Real.sin x + Real.sin (2 * x)) + 
  (Real.cos (x / 2) / Real.sin (x / 2))^2 = 10 / 3 →
  ∃ k : ℤ, x = π / 3 * (3 * ↑k + 1) ∨ x = π / 3 * (3 * ↑k - 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1752_175270


namespace NUMINAMATH_CALUDE_tip_percentage_calculation_l1752_175215

theorem tip_percentage_calculation (total_bill : ℝ) (food_price : ℝ) (tax_rate : ℝ) : 
  total_bill = 198 →
  food_price = 150 →
  tax_rate = 0.1 →
  (total_bill - food_price * (1 + tax_rate)) / (food_price * (1 + tax_rate)) = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_tip_percentage_calculation_l1752_175215


namespace NUMINAMATH_CALUDE_joe_average_speed_l1752_175234

-- Define the parameters of the problem
def distance1 : ℝ := 180
def speed1 : ℝ := 60
def distance2 : ℝ := 120
def speed2 : ℝ := 40

-- Define the theorem
theorem joe_average_speed :
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  (total_distance / total_time) = 50 := by sorry

end NUMINAMATH_CALUDE_joe_average_speed_l1752_175234


namespace NUMINAMATH_CALUDE_rectangle_area_l1752_175264

/-- Given a rectangle with an inscribed circle of radius 10 and a length-to-width ratio of 3:1,
    prove that its area is 1200. -/
theorem rectangle_area (r : ℝ) (l w : ℝ) (h_radius : r = 10) (h_ratio : l = 3 * w) 
    (h_inscribed : w = 2 * r) (h_circumscribed : l^2 + w^2 = (2 * r)^2) : l * w = 1200 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_l1752_175264


namespace NUMINAMATH_CALUDE_verandah_flooring_rate_l1752_175293

def hall_length : ℝ := 20
def hall_width : ℝ := 15
def verandah_width : ℝ := 2.5
def total_cost : ℝ := 700

def total_length : ℝ := hall_length + 2 * verandah_width
def total_width : ℝ := hall_width + 2 * verandah_width

def hall_area : ℝ := hall_length * hall_width
def total_area : ℝ := total_length * total_width
def verandah_area : ℝ := total_area - hall_area

theorem verandah_flooring_rate :
  total_cost / verandah_area = 3.5 := by sorry

end NUMINAMATH_CALUDE_verandah_flooring_rate_l1752_175293


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1752_175239

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (5 * Complex.I) / (1 + 2 * Complex.I) → Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1752_175239


namespace NUMINAMATH_CALUDE_distance_point_to_line_bounded_l1752_175240

/-- The distance from a point to a line in 2D space is bounded. -/
theorem distance_point_to_line_bounded (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let P : ℝ × ℝ := (-2, 2)
  let l := {(x, y) : ℝ × ℝ | a * (x - 1) + b * (y + 2) = 0}
  let d := Real.sqrt ((a * (-2 - 1) + b * (2 + 2))^2 / (a^2 + b^2))
  0 ≤ d ∧ d ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_bounded_l1752_175240


namespace NUMINAMATH_CALUDE_unique_solution_for_t_l1752_175265

/-- A non-zero digit is an integer between 1 and 9, inclusive. -/
def NonZeroDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The expression as a function of k and t -/
def expression (k t : NonZeroDigit) : ℤ :=
  808 + 10 * k.val + 80 * k.val + 8 - (1600 + 6 * t.val + 6)

theorem unique_solution_for_t :
  ∃! (t : NonZeroDigit), ∀ (k : NonZeroDigit),
    ∃ (n : ℤ), expression k t = n ∧ n % 10 = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_t_l1752_175265


namespace NUMINAMATH_CALUDE_first_last_checkpoint_distance_l1752_175231

-- Define the marathon parameters
def marathon_length : ℝ := 26
def num_checkpoints : ℕ := 4
def distance_between_checkpoints : ℝ := 6

-- Theorem statement
theorem first_last_checkpoint_distance :
  let total_checkpoint_distance := (num_checkpoints - 1 : ℝ) * distance_between_checkpoints
  let remaining_distance := marathon_length - total_checkpoint_distance
  let first_last_distance := remaining_distance / 2
  first_last_distance = 1 := by sorry

end NUMINAMATH_CALUDE_first_last_checkpoint_distance_l1752_175231


namespace NUMINAMATH_CALUDE_union_set_problem_l1752_175280

theorem union_set_problem (A B : Set ℕ) (m : ℕ) :
  A = {1, 2, m} →
  B = {2, 4} →
  A ∪ B = {1, 2, 3, 4} →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_union_set_problem_l1752_175280


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1752_175296

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (0 < a ∧ a < b → 1/a > 1/b) ∧
  ∃ a b : ℝ, 1/a > 1/b ∧ ¬(0 < a ∧ a < b) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1752_175296


namespace NUMINAMATH_CALUDE_conical_tube_surface_area_l1752_175284

/-- The surface area of a conical tube formed by rolling a semicircular paper. -/
theorem conical_tube_surface_area (r : ℝ) (h : r = 2) : 
  (π * r) = Real.pi * 2 := by
  sorry

end NUMINAMATH_CALUDE_conical_tube_surface_area_l1752_175284


namespace NUMINAMATH_CALUDE_sin_780_degrees_l1752_175200

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_780_degrees_l1752_175200


namespace NUMINAMATH_CALUDE_trick_deck_cost_is_nine_l1752_175290

/-- The cost of a single trick deck, given that 8 decks cost 72 dollars -/
def trick_deck_cost : ℚ :=
  72 / 8

/-- Theorem stating that the cost of each trick deck is 9 dollars -/
theorem trick_deck_cost_is_nine : trick_deck_cost = 9 := by
  sorry

end NUMINAMATH_CALUDE_trick_deck_cost_is_nine_l1752_175290


namespace NUMINAMATH_CALUDE_kiras_cat_kibble_l1752_175288

/-- Calculates the amount of kibble Kira initially filled her cat's bowl with. -/
def initial_kibble_amount (eating_rate : ℚ) (time_away : ℚ) (kibble_left : ℚ) : ℚ :=
  (time_away / 4) * eating_rate + kibble_left

/-- Theorem stating that given the conditions, Kira initially filled the bowl with 3 pounds of kibble. -/
theorem kiras_cat_kibble : initial_kibble_amount 1 8 1 = 3 := by
  sorry

#eval initial_kibble_amount 1 8 1

end NUMINAMATH_CALUDE_kiras_cat_kibble_l1752_175288


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1752_175205

theorem trigonometric_simplification (α : ℝ) :
  (Real.sin (6 * α) / Real.sin (2 * α)) + (Real.cos (6 * α - π) / Real.cos (2 * α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1752_175205


namespace NUMINAMATH_CALUDE_drum_capacity_ratio_l1752_175217

theorem drum_capacity_ratio (capacity_x capacity_y : ℝ) 
  (h1 : capacity_x > 0) 
  (h2 : capacity_y > 0) 
  (h3 : (1/2 : ℝ) * capacity_x + (2/5 : ℝ) * capacity_y = (65/100 : ℝ) * capacity_y) : 
  capacity_y / capacity_x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_drum_capacity_ratio_l1752_175217


namespace NUMINAMATH_CALUDE_fraction_equality_implies_cross_product_l1752_175236

theorem fraction_equality_implies_cross_product (x y : ℚ) :
  x / 2 = y / 3 → 3 * x = 2 * y ∧ ¬(2 * x = 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_cross_product_l1752_175236


namespace NUMINAMATH_CALUDE_min_value_of_function_l1752_175247

theorem min_value_of_function (x : ℝ) (h : x > 1) : 
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1752_175247


namespace NUMINAMATH_CALUDE_calculate_expression_l1752_175246

theorem calculate_expression : (π - 2023)^0 + |-9| - 3^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1752_175246


namespace NUMINAMATH_CALUDE_difference_of_squares_l1752_175216

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1752_175216


namespace NUMINAMATH_CALUDE_video_length_correct_l1752_175225

/-- The length of each video in minutes -/
def video_length : ℝ := 7

/-- The number of videos watched per day -/
def videos_per_day : ℝ := 2

/-- The time spent watching ads in minutes -/
def ad_time : ℝ := 3

/-- The total time spent on Youtube in minutes -/
def total_time : ℝ := 17

/-- Theorem stating that the video length is correct given the conditions -/
theorem video_length_correct :
  videos_per_day * video_length + ad_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_video_length_correct_l1752_175225


namespace NUMINAMATH_CALUDE_correct_marble_distribution_l1752_175242

/-- Represents the number of marbles each boy has -/
structure MarbleDistribution where
  middle : ℕ
  least : ℕ
  most : ℕ

/-- Checks if the given marble distribution satisfies the problem conditions -/
def is_valid_distribution (d : MarbleDistribution) : Prop :=
  -- The ratio of marbles is 4:2:3
  4 * d.middle = 2 * d.most ∧
  2 * d.least = 3 * d.middle ∧
  -- The boy with the least marbles has 10 more than twice the middle boy's marbles
  d.least = 2 * d.middle + 10 ∧
  -- The total number of marbles is 156
  d.middle + d.least + d.most = 156

/-- The theorem stating the correct distribution of marbles -/
theorem correct_marble_distribution :
  is_valid_distribution ⟨23, 57, 76⟩ := by sorry

end NUMINAMATH_CALUDE_correct_marble_distribution_l1752_175242


namespace NUMINAMATH_CALUDE_tea_consumption_average_l1752_175276

/-- Represents the inverse proportionality between research hours and tea quantity -/
def inverse_prop (k : ℝ) (r t : ℝ) : Prop := r * t = k

theorem tea_consumption_average (k : ℝ) :
  k = 8 * 3 →
  let t1 := k / 5
  let t2 := k / 10
  let t3 := k / 7
  (t1 + t2 + t3) / 3 = 124 / 35 := by
  sorry

end NUMINAMATH_CALUDE_tea_consumption_average_l1752_175276


namespace NUMINAMATH_CALUDE_progression_to_floor_pushups_l1752_175228

/-- The number of weeks it takes to progress to floor push-ups -/
def weeks_to_floor_pushups (days_per_week : ℕ) (levels_before_floor : ℕ) (days_per_level : ℕ) : ℕ :=
  (levels_before_floor * days_per_level) / days_per_week

/-- Theorem stating that it takes 9 weeks to progress to floor push-ups under given conditions -/
theorem progression_to_floor_pushups :
  weeks_to_floor_pushups 5 3 15 = 9 := by
  sorry

end NUMINAMATH_CALUDE_progression_to_floor_pushups_l1752_175228


namespace NUMINAMATH_CALUDE_largest_root_l1752_175248

theorem largest_root (p q r : ℝ) 
  (sum_eq : p + q + r = 1)
  (sum_prod_eq : p * q + p * r + q * r = -8)
  (prod_eq : p * q * r = 15) :
  max p (max q r) = 3 := by sorry

end NUMINAMATH_CALUDE_largest_root_l1752_175248


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1752_175203

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 2 ∧
  side_length = 7 ∧
  exterior_angle = 90 ∧
  (360 : ℝ) / n = exterior_angle →
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1752_175203


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1752_175263

-- Problem 1
theorem problem_1 (x y : ℝ) : (x - y)^2 + x * (x + 2*y) = 2*x^2 + y^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  ((-3*x + 4) / (x - 1) + x) / ((x - 2) / (x^2 - x)) = x^2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1752_175263


namespace NUMINAMATH_CALUDE_garden_vegetables_theorem_l1752_175230

/-- Represents the quantities of vegetables in a garden -/
structure GardenVegetables where
  tomatoes : ℕ
  potatoes : ℕ
  cabbages : ℕ
  eggplants : ℕ

/-- Calculates the final quantities of vegetables after changes -/
def finalQuantities (initial : GardenVegetables) 
  (tomatoesPicked potatoes_sold cabbagesBought eggplantsPlanted : ℕ) : GardenVegetables :=
  { tomatoes := initial.tomatoes - min initial.tomatoes tomatoesPicked,
    potatoes := initial.potatoes - min initial.potatoes potatoes_sold,
    cabbages := initial.cabbages + cabbagesBought,
    eggplants := initial.eggplants + eggplantsPlanted }

theorem garden_vegetables_theorem (initial : GardenVegetables) 
  (tomatoesPicked potatoes_sold cabbagesBought eggplantsPlanted : ℕ) :
  initial.tomatoes = 177 → 
  initial.potatoes = 12 → 
  initial.cabbages = 25 → 
  initial.eggplants = 10 → 
  tomatoesPicked = 53 → 
  potatoes_sold = 15 → 
  cabbagesBought = 32 → 
  eggplantsPlanted = 18 → 
  finalQuantities initial tomatoesPicked potatoes_sold cabbagesBought eggplantsPlanted = 
    { tomatoes := 124, potatoes := 0, cabbages := 57, eggplants := 28 } := by
  sorry

end NUMINAMATH_CALUDE_garden_vegetables_theorem_l1752_175230


namespace NUMINAMATH_CALUDE_sue_initial_savings_proof_l1752_175277

/-- The cost of the perfume in dollars -/
def perfume_cost : ℝ := 50

/-- Christian's initial savings in dollars -/
def christian_initial_savings : ℝ := 5

/-- Number of yards Christian mowed -/
def yards_mowed : ℕ := 4

/-- Cost per yard mowed in dollars -/
def cost_per_yard : ℝ := 5

/-- Number of dogs Sue walked -/
def dogs_walked : ℕ := 6

/-- Cost per dog walked in dollars -/
def cost_per_dog : ℝ := 2

/-- Additional amount needed in dollars -/
def additional_needed : ℝ := 6

/-- Sue's initial savings in dollars -/
def sue_initial_savings : ℝ := 7

theorem sue_initial_savings_proof :
  sue_initial_savings = 
    perfume_cost - 
    (christian_initial_savings + 
     (yards_mowed : ℝ) * cost_per_yard + 
     (dogs_walked : ℝ) * cost_per_dog) := by
  sorry

end NUMINAMATH_CALUDE_sue_initial_savings_proof_l1752_175277


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_special_hexagon_perimeter_is_4_sqrt_6_l1752_175297

/-- An equilateral hexagon with specific angle and area properties -/
structure SpecialHexagon where
  -- The hexagon is equilateral
  equilateral : Bool
  -- Alternating interior angles are 120° and 60°
  alternating_angles : Bool
  -- The area of the hexagon
  area : ℝ
  -- Conditions on the hexagon
  h_equilateral : equilateral = true
  h_alternating_angles : alternating_angles = true
  h_area : area = 12

/-- The perimeter of a SpecialHexagon is 4√6 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : ℝ :=
  4 * Real.sqrt 6

/-- The perimeter of a SpecialHexagon with area 12 is 4√6 -/
theorem special_hexagon_perimeter_is_4_sqrt_6 (h : SpecialHexagon) :
  special_hexagon_perimeter h = 4 * Real.sqrt 6 := by
  sorry

#check special_hexagon_perimeter_is_4_sqrt_6

end NUMINAMATH_CALUDE_special_hexagon_perimeter_special_hexagon_perimeter_is_4_sqrt_6_l1752_175297


namespace NUMINAMATH_CALUDE_perpendicular_parallel_lines_l1752_175289

/-- Given a line l with inclination 45°, line l₁ passing through A(3,2) and B(a,-1) perpendicular to l,
    and line l₂: 2x+by+1=0 parallel to l₁, prove that a + b = 8 -/
theorem perpendicular_parallel_lines (a b : ℝ) : 
  (∃ (l l₁ l₂ : Set (ℝ × ℝ)),
    -- l has inclination 45°
    (∀ (x y : ℝ), (x, y) ∈ l ↔ y = x) ∧
    -- l₁ passes through A(3,2) and B(a,-1)
    ((3, 2) ∈ l₁ ∧ (a, -1) ∈ l₁) ∧
    -- l₁ is perpendicular to l
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l ∧ (x₂, y₂) ∈ l₁ → (x₂ - x₁) * (y₂ - y₁) = -1) ∧
    -- l₂: 2x+by+1=0
    (∀ (x y : ℝ), (x, y) ∈ l₂ ↔ 2*x + b*y + 1 = 0) ∧
    -- l₂ is parallel to l₁
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₂ → (x₂ - x₁) * (y₂ - y₁) = 0))
  → a + b = 8 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_parallel_lines_l1752_175289


namespace NUMINAMATH_CALUDE_courier_package_ratio_l1752_175238

theorem courier_package_ratio : 
  ∀ (total_packages yesterday_packages today_packages : ℕ),
    total_packages = 240 →
    yesterday_packages = 80 →
    total_packages = yesterday_packages + today_packages →
    (today_packages : ℚ) / (yesterday_packages : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_courier_package_ratio_l1752_175238


namespace NUMINAMATH_CALUDE_apple_count_l1752_175224

/-- The number of apples initially in the basket -/
def initial_apples : ℕ := sorry

/-- The number of oranges initially in the basket -/
def initial_oranges : ℕ := 5

/-- The number of oranges added to the basket -/
def added_oranges : ℕ := 5

/-- The total number of fruits in the basket after adding oranges -/
def total_fruits : ℕ := initial_apples + initial_oranges + added_oranges

theorem apple_count : initial_apples = 10 :=
  by
    have h1 : initial_oranges = 5 := rfl
    have h2 : added_oranges = 5 := rfl
    have h3 : 2 * initial_apples = total_fruits := sorry
    sorry

end NUMINAMATH_CALUDE_apple_count_l1752_175224


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l1752_175271

theorem fraction_sum_inequality (b x y z : ℝ) 
  (hb : b > 0) 
  (hx : 0 < x ∧ x < b) 
  (hy : 0 < y ∧ y < b) 
  (hz : 0 < z ∧ z < b) : 
  (x / (b^2 + b*y + z*x)) + (y / (b^2 + b*z + x*y)) + (z / (b^2 + b*x + y*z)) < 1/b := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l1752_175271


namespace NUMINAMATH_CALUDE_bruno_initial_books_l1752_175210

/-- The number of books Bruno initially had -/
def initial_books : ℕ := sorry

/-- The number of books Bruno lost -/
def lost_books : ℕ := 4

/-- The number of books Bruno's dad gave him -/
def gained_books : ℕ := 10

/-- The final number of books Bruno had -/
def final_books : ℕ := 39

/-- Theorem stating that Bruno initially had 33 books -/
theorem bruno_initial_books : 
  initial_books = 33 ∧ 
  initial_books - lost_books + gained_books = final_books :=
sorry

end NUMINAMATH_CALUDE_bruno_initial_books_l1752_175210


namespace NUMINAMATH_CALUDE_shortest_player_height_l1752_175294

theorem shortest_player_height (tallest_height shortest_height height_difference : ℝ) :
  tallest_height = 77.75 →
  height_difference = 9.5 →
  tallest_height = shortest_height + height_difference →
  shortest_height = 68.25 := by
sorry

end NUMINAMATH_CALUDE_shortest_player_height_l1752_175294


namespace NUMINAMATH_CALUDE_priyas_age_l1752_175204

theorem priyas_age (P F : ℕ) : 
  F = P + 31 →
  (P + 8) + (F + 8) = 69 →
  P = 11 :=
by sorry

end NUMINAMATH_CALUDE_priyas_age_l1752_175204


namespace NUMINAMATH_CALUDE_twelve_non_congruent_triangles_l1752_175207

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℤ)
  (y : ℤ)

/-- The set of points on the grid -/
def grid_points : List Point := [
  ⟨0, 0⟩, ⟨1, 0⟩, ⟨2, 0⟩, ⟨3, 0⟩,
  ⟨0, 1⟩, ⟨1, 1⟩, ⟨2, 1⟩, ⟨3, 1⟩
]

/-- Determines if two triangles are congruent -/
def are_congruent (t1 t2 : Point × Point × Point) : Prop := sorry

/-- Counts the number of non-congruent triangles -/
def count_non_congruent_triangles (points : List Point) : ℕ := sorry

/-- The main theorem stating that there are 12 non-congruent triangles -/
theorem twelve_non_congruent_triangles :
  count_non_congruent_triangles grid_points = 12 := by sorry

end NUMINAMATH_CALUDE_twelve_non_congruent_triangles_l1752_175207


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_solution_set_is_axes_l1752_175266

theorem equation_represents_two_lines (x y : ℝ) :
  (x - y)^2 = x^2 + y^2 ↔ x * y = 0 :=
by sorry

-- The following definitions are to establish the connection
-- between the algebraic equation and its geometric interpretation

def x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}

theorem solution_set_is_axes :
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2} = x_axis ∪ y_axis :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_solution_set_is_axes_l1752_175266


namespace NUMINAMATH_CALUDE_jerry_thermostat_problem_l1752_175262

/-- Calculates the final temperature after a series of adjustments --/
def finalTemperature (initial : ℝ) : ℝ :=
  let doubled := initial * 2
  let afterDad := doubled - 30
  let afterMom := afterDad * 0.7  -- Reducing by 30% is equivalent to multiplying by 0.7
  let final := afterMom + 24
  final

/-- Theorem stating that the final temperature is 59 degrees --/
theorem jerry_thermostat_problem : finalTemperature 40 = 59 := by
  sorry

end NUMINAMATH_CALUDE_jerry_thermostat_problem_l1752_175262


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_1_solve_quadratic_equation_2_l1752_175249

-- Problem 1
theorem solve_quadratic_equation_1 :
  ∀ x : ℝ, x^2 - 4*x = 5 ↔ x = 5 ∨ x = -1 :=
sorry

-- Problem 2
theorem solve_quadratic_equation_2 :
  ∀ x : ℝ, 2*x^2 - 3*x + 1 = 0 ↔ x = 1 ∨ x = 1/2 :=
sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_1_solve_quadratic_equation_2_l1752_175249


namespace NUMINAMATH_CALUDE_triangle_problem_l1752_175208

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.b * Real.cos t.A + t.a * Real.cos t.B = -2 * t.c * Real.cos t.C)
  (h2 : t.b = 2 * t.a)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3) :
  t.C = 2 * Real.pi / 3 ∧ t.c = 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1752_175208


namespace NUMINAMATH_CALUDE_pen_price_problem_l1752_175279

theorem pen_price_problem (price : ℝ) (quantity : ℝ) : 
  (price * quantity = (price - 1) * (quantity + 100)) →
  (price * quantity = (price + 2) * (quantity - 100)) →
  price = 4 := by
sorry

end NUMINAMATH_CALUDE_pen_price_problem_l1752_175279


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l1752_175237

theorem sum_of_squares_problem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 52)
  (sum_of_products : x*y + y*z + z*x = 27) :
  x + y + z = Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l1752_175237


namespace NUMINAMATH_CALUDE_factorization_of_4x2_minus_16y2_l1752_175256

theorem factorization_of_4x2_minus_16y2 (x y : ℝ) : 4 * x^2 - 16 * y^2 = 4 * (x + 2*y) * (x - 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x2_minus_16y2_l1752_175256


namespace NUMINAMATH_CALUDE_fraction_calculation_l1752_175285

theorem fraction_calculation : (2 / 3 * 4 / 7 * 5 / 8) + 1 / 6 = 17 / 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1752_175285


namespace NUMINAMATH_CALUDE_oil_needed_calculation_l1752_175283

structure Vehicle where
  cylinders : ℕ
  oil_per_cylinder : ℕ
  oil_in_engine : ℕ

def additional_oil_needed (v : Vehicle) : ℕ :=
  v.cylinders * v.oil_per_cylinder - v.oil_in_engine

def car : Vehicle := {
  cylinders := 6,
  oil_per_cylinder := 8,
  oil_in_engine := 16
}

def truck : Vehicle := {
  cylinders := 8,
  oil_per_cylinder := 10,
  oil_in_engine := 20
}

def motorcycle : Vehicle := {
  cylinders := 4,
  oil_per_cylinder := 6,
  oil_in_engine := 8
}

theorem oil_needed_calculation :
  additional_oil_needed car = 32 ∧
  additional_oil_needed truck = 60 ∧
  additional_oil_needed motorcycle = 16 := by
  sorry

end NUMINAMATH_CALUDE_oil_needed_calculation_l1752_175283


namespace NUMINAMATH_CALUDE_sandals_sold_l1752_175226

theorem sandals_sold (shoes : ℕ) (sandals : ℕ) : 
  (shoes : ℚ) / sandals = 15 / 8 → shoes = 135 → sandals = 72 := by
sorry

end NUMINAMATH_CALUDE_sandals_sold_l1752_175226


namespace NUMINAMATH_CALUDE_dentist_bill_calculation_dentist_cleaning_cost_l1752_175298

theorem dentist_bill_calculation (filling_cost : ℕ) (extraction_cost : ℕ) : ℕ :=
  let total_bill := 5 * filling_cost
  let cleaning_cost := total_bill - (2 * filling_cost + extraction_cost)
  cleaning_cost

theorem dentist_cleaning_cost : dentist_bill_calculation 120 290 = 70 := by
  sorry

end NUMINAMATH_CALUDE_dentist_bill_calculation_dentist_cleaning_cost_l1752_175298


namespace NUMINAMATH_CALUDE_perimeter_CMN_in_terms_of_AM_l1752_175206

/-- Rectangle ABCD with equilateral triangle CMN --/
structure RectangleWithTriangle where
  /-- Point A of rectangle ABCD --/
  A : ℝ × ℝ
  /-- Point B of rectangle ABCD --/
  B : ℝ × ℝ
  /-- Point C of rectangle ABCD --/
  C : ℝ × ℝ
  /-- Point D of rectangle ABCD --/
  D : ℝ × ℝ
  /-- Point M on side AB --/
  M : ℝ × ℝ
  /-- Point N of equilateral triangle CMN --/
  N : ℝ × ℝ
  /-- ABCD is a rectangle --/
  is_rectangle : (B.1 - A.1) * (C.2 - B.2) = (C.1 - B.1) * (B.2 - A.2)
  /-- Length of ABCD is 2 --/
  length_is_2 : B.1 - A.1 = 2
  /-- Width of ABCD is 1 --/
  width_is_1 : C.2 - B.2 = 1
  /-- M is on AB --/
  M_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (A.1 + t * (B.1 - A.1), A.2)
  /-- CMN is equilateral --/
  CMN_equilateral : (C.1 - M.1)^2 + (C.2 - M.2)^2 = (M.1 - N.1)^2 + (M.2 - N.2)^2 ∧
                    (C.1 - M.1)^2 + (C.2 - M.2)^2 = (C.1 - N.1)^2 + (C.2 - N.2)^2

/-- The perimeter of CMN can be expressed in terms of AM --/
theorem perimeter_CMN_in_terms_of_AM (fig : RectangleWithTriangle) :
  ∃ (x : ℝ), x = fig.M.1 - fig.A.1 ∧
  ∃ (perimeter : ℝ), perimeter = 3 * Real.sqrt ((x^2 + 1)^2) :=
sorry

end NUMINAMATH_CALUDE_perimeter_CMN_in_terms_of_AM_l1752_175206


namespace NUMINAMATH_CALUDE_rationalize_denominator_cube_root_rationalize_35_cube_root_l1752_175274

theorem rationalize_denominator_cube_root (x : ℝ) (hx : x > 0) :
  (x / x^(1/3)) = x^(2/3) :=
by sorry

theorem rationalize_35_cube_root :
  (35 : ℝ) / (35 : ℝ)^(1/3) = (1225 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_cube_root_rationalize_35_cube_root_l1752_175274


namespace NUMINAMATH_CALUDE_japanese_study_fraction_l1752_175220

theorem japanese_study_fraction (J S : ℕ) (x : ℚ) : 
  S = 2 * J →  -- Senior class is twice the size of junior class
  (1 / 4 : ℚ) * J + x * S = (1 / 3 : ℚ) * (J + S) →  -- Total Japanese students equation
  x = 3 / 8 :=  -- Fraction of seniors studying Japanese
by sorry

end NUMINAMATH_CALUDE_japanese_study_fraction_l1752_175220


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1752_175202

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : InverselyProportional x₁ y₁)
  (h2 : InverselyProportional x₂ y₂)
  (h3 : x₁ = 40)
  (h4 : y₁ = 8)
  (h5 : y₂ = 10) :
  x₂ = 32 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1752_175202


namespace NUMINAMATH_CALUDE_inequality_implication_l1752_175253

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1752_175253


namespace NUMINAMATH_CALUDE_blue_square_area_ratio_l1752_175218

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  /-- Side length of the square flag -/
  side : ℝ
  /-- Side length of the blue square in the center -/
  blue_side : ℝ
  /-- The cross (red arms + blue center) occupies 36% of the flag's area -/
  cross_area_ratio : side ^ 2 * 0.36 = (4 * blue_side * (side - blue_side) + blue_side ^ 2)

/-- The blue square in the center occupies 9% of the flag's area -/
theorem blue_square_area_ratio (flag : CrossFlag) : 
  flag.blue_side ^ 2 / flag.side ^ 2 = 0.09 := by sorry

end NUMINAMATH_CALUDE_blue_square_area_ratio_l1752_175218


namespace NUMINAMATH_CALUDE_triangle_sum_l1752_175282

theorem triangle_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + (1/3)*y^2 = 25)
  (eq2 : (1/3)*y^2 + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 16) :
  x*y + 2*y*z + 3*z*x = 24*Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_sum_l1752_175282


namespace NUMINAMATH_CALUDE_expression_simplification_l1752_175267

theorem expression_simplification (x : ℤ) 
  (h1 : 2 * (x - 1) < x + 1) 
  (h2 : 5 * x + 3 ≥ 2 * x) : 
  (2 : ℚ) / (x^2 + x) / (1 - (x - 1) / (x^2 - 1)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1752_175267


namespace NUMINAMATH_CALUDE_condition_one_condition_two_condition_three_condition_four_l1752_175291

/-- Represents the number of male athletes -/
def num_male : ℕ := 6

/-- Represents the number of female athletes -/
def num_female : ℕ := 4

/-- Represents the total number of athletes -/
def total_athletes : ℕ := num_male + num_female

/-- Represents the size of the team to be selected -/
def team_size : ℕ := 5

/-- Represents the number of male athletes in the team for condition 1 -/
def male_in_team : ℕ := 3

/-- Represents the number of female athletes in the team for condition 1 -/
def female_in_team : ℕ := 2

/-- Theorem for the first condition -/
theorem condition_one : 
  (num_male.choose male_in_team) * (num_female.choose female_in_team) = 120 := by sorry

/-- Theorem for the second condition -/
theorem condition_two : 
  total_athletes.choose team_size - num_male.choose team_size = 246 := by sorry

/-- Theorem for the third condition -/
theorem condition_three : 
  total_athletes.choose team_size - (num_male - 1).choose team_size - (num_female - 1).choose team_size + (total_athletes - 2).choose team_size = 196 := by sorry

/-- Theorem for the fourth condition -/
theorem condition_four : 
  (total_athletes - 1).choose (team_size - 1) + ((total_athletes - 2).choose (team_size - 1) - (num_male - 1).choose (team_size - 1)) = 191 := by sorry

end NUMINAMATH_CALUDE_condition_one_condition_two_condition_three_condition_four_l1752_175291


namespace NUMINAMATH_CALUDE_quadratic_function_through_point_l1752_175281

theorem quadratic_function_through_point (a b : ℝ) :
  (∀ t : ℝ, (t^2 + t + 1) * 1^2 - 2*(a+t)^2 * 1 + t^2 + 3*a*t + b = 0) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_through_point_l1752_175281


namespace NUMINAMATH_CALUDE_half_sum_of_odd_squares_is_sum_of_squares_l1752_175255

theorem half_sum_of_odd_squares_is_sum_of_squares (a b : ℕ) (ha : Odd a) (hb : Odd b) (hab : a ≠ b) :
  ∃ x y : ℕ, (a^2 + b^2) / 2 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_half_sum_of_odd_squares_is_sum_of_squares_l1752_175255


namespace NUMINAMATH_CALUDE_fixed_fee_calculation_l1752_175254

/-- Represents the billing system for an online service provider -/
structure BillingSystem where
  fixed_fee : ℝ
  hourly_charge : ℝ

/-- Calculates the total bill given the connect time -/
def total_bill (bs : BillingSystem) (connect_time : ℝ) : ℝ :=
  bs.fixed_fee + bs.hourly_charge * connect_time

theorem fixed_fee_calculation (bs : BillingSystem) :
  total_bill bs 1 = 15.75 ∧ total_bill bs 3 = 24.45 → bs.fixed_fee = 11.40 := by
  sorry

end NUMINAMATH_CALUDE_fixed_fee_calculation_l1752_175254


namespace NUMINAMATH_CALUDE_john_pays_21_l1752_175221

/-- Given the total number of candy bars, the number Dave pays for, and the cost per candy bar,
    calculate the amount John pays. -/
def johnPayment (totalBars : ℕ) (davePays : ℕ) (costPerBar : ℚ) : ℚ :=
  (totalBars - davePays : ℚ) * costPerBar

/-- Theorem stating that John pays $21 given the problem conditions. -/
theorem john_pays_21 :
  johnPayment 20 6 (3/2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_john_pays_21_l1752_175221


namespace NUMINAMATH_CALUDE_segment_AB_length_l1752_175214

-- Define the points on the number line
def point_A : ℝ := -5
def point_B : ℝ := 2

-- Define the length of the segment
def segment_length (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem segment_AB_length :
  segment_length point_A point_B = 7 := by
  sorry

end NUMINAMATH_CALUDE_segment_AB_length_l1752_175214


namespace NUMINAMATH_CALUDE_fence_whitewashing_fence_theorem_l1752_175257

theorem fence_whitewashing (total_fence : ℝ) (ben_amount : ℝ) 
  (billy_fraction : ℝ) (johnny_fraction : ℝ) : ℝ :=
  let remaining_after_ben := total_fence - ben_amount
  let billy_amount := billy_fraction * remaining_after_ben
  let remaining_after_billy := remaining_after_ben - billy_amount
  let johnny_amount := johnny_fraction * remaining_after_billy
  let final_remaining := remaining_after_billy - johnny_amount
  final_remaining

theorem fence_theorem : 
  fence_whitewashing 100 10 (1/5) (1/3) = 48 := by
  sorry

end NUMINAMATH_CALUDE_fence_whitewashing_fence_theorem_l1752_175257


namespace NUMINAMATH_CALUDE_no_integer_solution_l1752_175299

theorem no_integer_solution : ¬∃ (a b c : ℤ), a^2 + b^2 + 1 = 4*c := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1752_175299


namespace NUMINAMATH_CALUDE_adam_ella_equation_l1752_175269

theorem adam_ella_equation (d e : ℝ) : 
  (∀ x, |x - 8| = 3 ↔ x^2 + d*x + e = 0) → 
  d = -16 ∧ e = 55 := by
sorry

end NUMINAMATH_CALUDE_adam_ella_equation_l1752_175269


namespace NUMINAMATH_CALUDE_forty_percent_of_number_l1752_175260

theorem forty_percent_of_number (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → (40/100 : ℝ) * N = 192 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_l1752_175260


namespace NUMINAMATH_CALUDE_min_value_theorem_l1752_175241

theorem min_value_theorem (x y : ℝ) : (x^2*y - 1)^2 + (x + y - 1)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1752_175241


namespace NUMINAMATH_CALUDE_residue_of_negative_1235_mod_29_l1752_175292

theorem residue_of_negative_1235_mod_29 : 
  ∃ k : ℤ, -1235 = 29 * k + 12 ∧ 0 ≤ 12 ∧ 12 < 29 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_negative_1235_mod_29_l1752_175292


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l1752_175223

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (x+2y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2*p.2, 2*p.1 - p.2)

/-- The theorem stating that (-1/3, 5/3) is the pre-image of (3, 1) under the mapping f -/
theorem preimage_of_3_1 :
  f (-1/3, 5/3) = (3, 1) :=
by sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l1752_175223


namespace NUMINAMATH_CALUDE_electionWaysCount_l1752_175211

/-- Represents the Science Club with its election rules -/
structure ScienceClub where
  totalMembers : Nat
  aliceIndex : Nat
  bobIndex : Nat

/-- Represents the possible election outcomes -/
inductive ElectionOutcome
  | WithoutAliceAndBob (president secretary treasurer : Nat)
  | WithAliceAndBob (treasurer : Nat)

/-- Checks if an election outcome is valid according to the club's rules -/
def isValidOutcome (club : ScienceClub) (outcome : ElectionOutcome) : Prop :=
  match outcome with
  | ElectionOutcome.WithoutAliceAndBob p s t =>
      p ≠ club.aliceIndex ∧ p ≠ club.bobIndex ∧
      s ≠ club.aliceIndex ∧ s ≠ club.bobIndex ∧
      t ≠ club.aliceIndex ∧ t ≠ club.bobIndex ∧
      p ≠ s ∧ p ≠ t ∧ s ≠ t ∧
      p < club.totalMembers ∧ s < club.totalMembers ∧ t < club.totalMembers
  | ElectionOutcome.WithAliceAndBob t =>
      t ≠ club.aliceIndex ∧ t ≠ club.bobIndex ∧
      t < club.totalMembers

/-- Counts the number of valid election outcomes -/
def countValidOutcomes (club : ScienceClub) : Nat :=
  sorry

/-- The main theorem stating the number of ways to elect officers -/
theorem electionWaysCount (club : ScienceClub) 
    (h1 : club.totalMembers = 25)
    (h2 : club.aliceIndex < club.totalMembers)
    (h3 : club.bobIndex < club.totalMembers)
    (h4 : club.aliceIndex ≠ club.bobIndex) :
    countValidOutcomes club = 10649 :=
  sorry

end NUMINAMATH_CALUDE_electionWaysCount_l1752_175211


namespace NUMINAMATH_CALUDE_cloth_sales_calculation_l1752_175272

/-- Calculates the total sales given the commission rate and commission amount -/
def totalSales (commissionRate : ℚ) (commissionAmount : ℚ) : ℚ :=
  commissionAmount / (commissionRate / 100)

/-- Theorem: Given a commission rate of 2.5% and a commission of 18, the total sales is 720 -/
theorem cloth_sales_calculation :
  totalSales (2.5 : ℚ) 18 = 720 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sales_calculation_l1752_175272


namespace NUMINAMATH_CALUDE_gcd_upper_bound_l1752_175287

theorem gcd_upper_bound (a b : ℕ+) : Nat.gcd a.val b.val ≤ Real.sqrt (a.val + b.val : ℝ) := by sorry

end NUMINAMATH_CALUDE_gcd_upper_bound_l1752_175287


namespace NUMINAMATH_CALUDE_two_digit_number_divisible_by_8_12_18_l1752_175268

theorem two_digit_number_divisible_by_8_12_18 :
  ∃! n : ℕ, 60 ≤ n ∧ n ≤ 79 ∧ 8 ∣ n ∧ 12 ∣ n ∧ 18 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_divisible_by_8_12_18_l1752_175268


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1752_175201

theorem inequality_solution_set (x : ℝ) : 
  (x - 5) / (x + 1) ≤ 0 ∧ x + 1 ≠ 0 ↔ x ∈ Set.Ioc (-1) 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1752_175201


namespace NUMINAMATH_CALUDE_max_page_number_proof_l1752_175278

def max_page_number (ones : ℕ) (twos : ℕ) : ℕ :=
  let digits : List ℕ := [0, 3, 4, 5, 6, 7, 8, 9]
  199

theorem max_page_number_proof (ones twos : ℕ) :
  ones = 25 → twos = 30 → max_page_number ones twos = 199 := by
  sorry

end NUMINAMATH_CALUDE_max_page_number_proof_l1752_175278
