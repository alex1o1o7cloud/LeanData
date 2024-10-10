import Mathlib

namespace weighted_mean_calculation_l1940_194015

def numbers : List ℝ := [16, 28, 45]
def weights : List ℝ := [2, 3, 5]

theorem weighted_mean_calculation :
  (List.sum (List.zipWith (· * ·) numbers weights)) / (List.sum weights) = 34.1 := by
  sorry

end weighted_mean_calculation_l1940_194015


namespace line_tangent_to_parabola_l1940_194006

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 1 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x ∧ 
    ∀ x' y' : ℝ, y' = 3*x' + c → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  c = 1 := by
sorry

end line_tangent_to_parabola_l1940_194006


namespace special_elements_in_100_l1940_194011

/-- Represents the number of elements in the nth group -/
def group_size (n : ℕ) : ℕ := n + 1

/-- Calculates the total number of elements up to and including the nth group -/
def total_elements (n : ℕ) : ℕ := n * (n + 3) / 2

/-- Represents the number of special elements in the first n groups -/
def special_elements (n : ℕ) : ℕ := n

theorem special_elements_in_100 :
  ∃ n : ℕ, total_elements n ≤ 100 ∧ total_elements (n + 1) > 100 ∧ special_elements n = 12 :=
sorry

end special_elements_in_100_l1940_194011


namespace compatible_pairs_theorem_l1940_194035

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

def is_compatible (a b : ℕ) : Prop :=
  (a = sum_of_digits b ∧ b = product_of_digits a) ∨
  (b = sum_of_digits a ∧ a = product_of_digits b)

def compatible_pairs_within (n : ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 ≤ n ∧ p.2 ≤ n ∧ is_compatible p.1 p.2}

def compatible_pairs_within_one_greater (n m : ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 ≤ n ∧ p.2 ≤ n ∧ (p.1 > m ∨ p.2 > m) ∧ is_compatible p.1 p.2}

theorem compatible_pairs_theorem :
  compatible_pairs_within 100 = {(9, 11), (12, 36)} ∧
  compatible_pairs_within_one_greater 1000 99 = {(135, 19), (144, 19)} := by sorry

end compatible_pairs_theorem_l1940_194035


namespace collinear_points_m_value_l1940_194092

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_m_value :
  ∀ m : ℝ,
  let p1 : Point := ⟨3, -4⟩
  let p2 : Point := ⟨6, 5⟩
  let p3 : Point := ⟨8, m⟩
  collinear p1 p2 p3 → m = 11 := by
  sorry

end collinear_points_m_value_l1940_194092


namespace candy_bar_cost_l1940_194062

theorem candy_bar_cost (initial_amount : ℕ) (change : ℕ) (cost : ℕ) : 
  initial_amount = 50 →
  change = 5 →
  cost = initial_amount - change →
  cost = 45 :=
by sorry

end candy_bar_cost_l1940_194062


namespace quadratic_function_property_l1940_194041

/-- Given a quadratic function y = ax^2 + bx + 3 where a and b are constants and a ≠ 0,
    prove that if (-m,0) and (3m,0) lie on the graph of this function, then b^2 + 4a = 0. -/
theorem quadratic_function_property (a b m : ℝ) (h_a : a ≠ 0) :
  (a * (-m)^2 + b * (-m) + 3 = 0) →
  (a * (3*m)^2 + b * (3*m) + 3 = 0) →
  b^2 + 4*a = 0 := by
  sorry

end quadratic_function_property_l1940_194041


namespace parabola_c_value_l1940_194087

/-- A parabola passing through three given points has a specific c value -/
theorem parabola_c_value (b c : ℝ) :
  (1^2 + b*1 + c = 2) ∧ 
  (4^2 + b*4 + c = 5) ∧ 
  (7^2 + b*7 + c = 2) →
  c = 9 := by
sorry

end parabola_c_value_l1940_194087


namespace rectangle_formation_count_l1940_194064

/-- The number of ways to choose 2 items from a set of 5 items -/
def choose_two_from_five : ℕ := 10

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 5

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 5

/-- The number of ways to choose 4 lines to form a rectangle -/
def ways_to_form_rectangle : ℕ := choose_two_from_five * choose_two_from_five

theorem rectangle_formation_count :
  ways_to_form_rectangle = 100 :=
sorry

end rectangle_formation_count_l1940_194064


namespace min_green_chips_l1940_194058

/-- Given a basket of chips with three colors: green, yellow, and violet.
    This theorem proves that the minimum number of green chips is 120,
    given the conditions stated in the problem. -/
theorem min_green_chips (y v g : ℕ) : 
  v ≥ (2 : ℕ) * y / 3 →  -- violet chips are at least two-thirds of yellow chips
  v ≤ g / 4 →            -- violet chips are at most one-fourth of green chips
  y + v ≥ 75 →           -- sum of yellow and violet chips is at least 75
  g ≥ 120 :=             -- prove that the minimum number of green chips is 120
by sorry

end min_green_chips_l1940_194058


namespace value_of_a_l1940_194061

theorem value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 15 - 6 * a) : a = 3 / 2 := by
  sorry

end value_of_a_l1940_194061


namespace max_value_inequality_l1940_194001

theorem max_value_inequality (x y : ℝ) (hx : x > 1/2) (hy : y > 1) :
  (∃ m : ℝ, ∀ x y : ℝ, x > 1/2 → y > 1 → (4 * x^2) / (y - 1) + y^2 / (2 * x - 1) ≥ m) ∧
  (∀ m : ℝ, (∀ x y : ℝ, x > 1/2 → y > 1 → (4 * x^2) / (y - 1) + y^2 / (2 * x - 1) ≥ m) → m ≤ 8) :=
sorry

end max_value_inequality_l1940_194001


namespace min_value_expression_l1940_194070

theorem min_value_expression (x y : ℝ) : (x * y + 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

end min_value_expression_l1940_194070


namespace ellipse_intersection_range_l1940_194009

-- Define the line equation
def line (k : ℝ) (x y : ℝ) : Prop := y - k * x - 1 = 0

-- Define the ellipse equation
def ellipse (b : ℝ) (x y : ℝ) : Prop := x^2 / 5 + y^2 / b = 1

-- Define the condition that the line always intersects the ellipse
def always_intersects (b : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x y : ℝ, line k x y ∧ ellipse b x y

-- Theorem statement
theorem ellipse_intersection_range :
  ∀ b : ℝ, (always_intersects b) ↔ (b ∈ Set.Icc 1 5 ∪ Set.Ioi 5) :=
sorry

end ellipse_intersection_range_l1940_194009


namespace odd_function_property_l1940_194053

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (hf : IsOdd f) :
  g f 1 = 1 → g f (-1) = 3 := by
  sorry

end odd_function_property_l1940_194053


namespace can_obtain_any_number_l1940_194085

/-- Represents the allowed operations on natural numbers -/
inductive Operation
  | append4 : Operation
  | append0 : Operation
  | divideBy2 : Operation

/-- Applies an operation to a natural number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.append4 => n * 10 + 4
  | Operation.append0 => n * 10
  | Operation.divideBy2 => if n % 2 = 0 then n / 2 else n

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a natural number -/
def applySequence (n : ℕ) (seq : OperationSequence) : ℕ :=
  seq.foldl applyOperation n

/-- Theorem: Any natural number can be obtained from 4 using the allowed operations -/
theorem can_obtain_any_number : ∀ (n : ℕ), ∃ (seq : OperationSequence), applySequence 4 seq = n := by
  sorry

end can_obtain_any_number_l1940_194085


namespace lcm_12_15_18_l1940_194010

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end lcm_12_15_18_l1940_194010


namespace total_placards_taken_l1940_194017

/-- The number of placards taken by people entering a stadium -/
def placards_taken (people : ℕ) (placards_per_person : ℕ) : ℕ :=
  people * placards_per_person

/-- Theorem stating the total number of placards taken by 2841 people -/
theorem total_placards_taken :
  placards_taken 2841 2 = 5682 := by
  sorry

end total_placards_taken_l1940_194017


namespace average_first_17_even_numbers_l1940_194094

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem average_first_17_even_numbers : 
  average (first_n_even_numbers 17) = 20 := by
sorry

end average_first_17_even_numbers_l1940_194094


namespace at_least_one_positive_l1940_194074

theorem at_least_one_positive (x y z : ℝ) : 
  let a := x^2 - 2*x + π/2
  let b := y^2 - 2*y + π/3
  let c := z^2 - 2*z + π/6
  (a > 0) ∨ (b > 0) ∨ (c > 0) := by
sorry

end at_least_one_positive_l1940_194074


namespace stamp_sum_l1940_194090

theorem stamp_sum : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n < 100 ∧ n % 6 = 4 ∧ n % 8 = 2) ∧ 
  (∀ n < 100, n % 6 = 4 ∧ n % 8 = 2 → n ∈ S) ∧
  S.sum id = 68 := by
sorry

end stamp_sum_l1940_194090


namespace min_value_expression_l1940_194054

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≤ b + c) (hbc : b ≤ a + c) (hca : c ≤ a + b) :
  c / (a + b) + b / c ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end min_value_expression_l1940_194054


namespace expression_simplification_l1940_194039

theorem expression_simplification :
  (4^2 * 7) / (8 * 9^2) * (8 * 9 * 11^2) / (4 * 7 * 11) = 44 / 9 := by
  sorry

end expression_simplification_l1940_194039


namespace most_likely_outcome_is_equal_distribution_l1940_194086

def probability_of_outcome (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

theorem most_likely_outcome_is_equal_distribution :
  ∀ k : ℕ, k ≤ 8 →
    probability_of_outcome 8 4 ≥ probability_of_outcome 8 k :=
sorry

end most_likely_outcome_is_equal_distribution_l1940_194086


namespace shot_put_distance_l1940_194038

/-- The horizontal distance at which a shot put hits the ground, given its trajectory. -/
theorem shot_put_distance : ∃ x : ℝ, x > 0 ∧ 
  (-1/12 * x^2 + 2/3 * x + 5/3 = 0) ∧ x = 10 := by sorry

end shot_put_distance_l1940_194038


namespace final_arrangement_decreasing_l1940_194075

/-- Represents a child with a unique height -/
structure Child :=
  (height : ℕ)

/-- Represents a row of children -/
def Row := List Child

/-- The operation of grouping and rearranging children -/
def groupAndRearrange (row : Row) : Row :=
  sorry

/-- Checks if a row is in decreasing order of height -/
def isDecreasingOrder (row : Row) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem final_arrangement_decreasing (n : ℕ) (initial_row : Row) :
  initial_row.length = n →
  (∀ i j, i ≠ j → (initial_row.get i).height ≠ (initial_row.get j).height) →
  isDecreasingOrder ((groupAndRearrange^[n-1]) initial_row) :=
sorry

end final_arrangement_decreasing_l1940_194075


namespace quadratic_factorization_l1940_194031

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end quadratic_factorization_l1940_194031


namespace head_start_problem_l1940_194088

/-- The head start problem -/
theorem head_start_problem (cristina_speed nicky_speed : ℝ) (catch_up_time : ℝ) 
  (h1 : cristina_speed = 5)
  (h2 : nicky_speed = 3)
  (h3 : catch_up_time = 24) :
  cristina_speed * catch_up_time - nicky_speed * catch_up_time = 48 := by
  sorry

#check head_start_problem

end head_start_problem_l1940_194088


namespace right_triangle_area_l1940_194025

/-- A right triangle with one leg of length 15 and an inscribed circle of radius 3 has an area of 60. -/
theorem right_triangle_area (a b c r : ℝ) : 
  a = 15 → -- One leg is 15
  r = 3 → -- Radius of inscribed circle is 3
  a^2 + b^2 = c^2 → -- Right triangle (Pythagorean theorem)
  r * (a + b + c) / 2 = r * b → -- Area formula using semiperimeter and inradius
  a * b / 2 = 60 := by -- Area of the triangle is 60
sorry

end right_triangle_area_l1940_194025


namespace calculate_savings_savings_calculation_l1940_194000

/-- Given a person's income and expenditure ratio, and their income, calculate their savings. -/
theorem calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) : ℕ :=
  let total_ratio := income_ratio + expenditure_ratio
  let expenditure := (expenditure_ratio * income) / income_ratio
  income - expenditure

/-- Prove that given a person's income and expenditure ratio of 10:7 and an income of Rs. 10000, the person's savings are Rs. 3000. -/
theorem savings_calculation : calculate_savings 10 7 10000 = 3000 := by
  sorry

end calculate_savings_savings_calculation_l1940_194000


namespace last_digit_of_one_over_three_to_fifteen_l1940_194057

theorem last_digit_of_one_over_three_to_fifteen (n : ℕ) : 
  n = 15 → (1 : ℚ) / (3 ^ n) * 10^n % 10 = 5 := by
  sorry

end last_digit_of_one_over_three_to_fifteen_l1940_194057


namespace ratio_w_to_y_l1940_194034

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 5 / 4)
  (hy_z : y / z = 3 / 2)
  (hz_x : z / x = 1 / 4)
  (hsum : w + x + y + z = 60) :
  w / y = 10 / 3 := by
sorry

end ratio_w_to_y_l1940_194034


namespace complex_modulus_l1940_194068

theorem complex_modulus (z : ℂ) (h : z * (1 + Complex.I) = Complex.I ^ 2016) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_l1940_194068


namespace paint_remaining_paint_problem_l1940_194052

theorem paint_remaining (initial_paint : ℝ) (first_day_usage : ℝ) (second_day_usage : ℝ) (spill_loss : ℝ) : ℝ :=
  let remaining_after_first_day := initial_paint - first_day_usage
  let remaining_after_second_day := remaining_after_first_day - second_day_usage
  let remaining_after_spill := remaining_after_second_day - spill_loss
  remaining_after_spill

theorem paint_problem : paint_remaining 1 (1/2) ((1/2)/2) ((1/4)/4) = 3/16 := by
  sorry

end paint_remaining_paint_problem_l1940_194052


namespace intersection_point_is_solution_l1940_194013

/-- The intersection point of two lines is the solution to a system of equations -/
theorem intersection_point_is_solution (x y : ℝ) :
  (y = 2*x + 1) ∧ (y = -x + 4) →  -- Given intersection point equations
  (x = 1 ∧ y = 3) →               -- Given intersection point
  (2*x - y = -1) ∧ (x + y = 4)    -- System of equations to prove
  := by sorry

end intersection_point_is_solution_l1940_194013


namespace f_max_value_l1940_194081

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n : ℚ) / ((n + 32 : ℚ) * (S (n + 1) : ℚ))

theorem f_max_value :
  (∀ n : ℕ, f n ≤ 1/50) ∧ (∃ n : ℕ, f n = 1/50) := by sorry

end f_max_value_l1940_194081


namespace condition_A_neither_necessary_nor_sufficient_l1940_194033

/-- Condition A: √(1 + sin θ) = a -/
def condition_A (θ : Real) (a : Real) : Prop :=
  Real.sqrt (1 + Real.sin θ) = a

/-- Condition B: sin(θ/2) + cos(θ/2) = a -/
def condition_B (θ : Real) (a : Real) : Prop :=
  Real.sin (θ / 2) + Real.cos (θ / 2) = a

/-- Theorem stating that condition A is neither necessary nor sufficient for condition B -/
theorem condition_A_neither_necessary_nor_sufficient :
  ¬(∀ θ a, condition_A θ a ↔ condition_B θ a) ∧
  ¬(∀ θ a, condition_A θ a → condition_B θ a) ∧
  ¬(∀ θ a, condition_B θ a → condition_A θ a) := by
  sorry

end condition_A_neither_necessary_nor_sufficient_l1940_194033


namespace max_area_rectangle_max_area_achieved_l1940_194002

/-- The maximum area of a rectangle with integer side lengths and perimeter 160 -/
theorem max_area_rectangle (x y : ℕ) (h : x + y = 80) : x * y ≤ 1600 :=
sorry

/-- The maximum area is achieved when both sides are 40 -/
theorem max_area_achieved (x y : ℕ) (h : x + y = 80) : x * y = 1600 ↔ x = 40 ∧ y = 40 :=
sorry

end max_area_rectangle_max_area_achieved_l1940_194002


namespace ajax_final_weight_l1940_194022

/-- Calculates the final weight in pounds after a weight loss program -/
def final_weight (initial_weight_kg : ℝ) (weight_loss_per_hour : ℝ) (hours_per_day : ℝ) (days : ℝ) : ℝ :=
  let kg_to_pounds : ℝ := 2.2
  let initial_weight_pounds : ℝ := initial_weight_kg * kg_to_pounds
  let total_weight_loss : ℝ := weight_loss_per_hour * hours_per_day * days
  initial_weight_pounds - total_weight_loss

/-- Theorem: Ajax's weight after the exercise program -/
theorem ajax_final_weight :
  final_weight 80 1.5 2 14 = 134 := by
sorry


end ajax_final_weight_l1940_194022


namespace justin_sabrina_pencils_l1940_194050

/-- Given that Justin and Sabrina have 50 pencils combined, Justin has 8 more pencils than m times 
    Sabrina's pencils, and Sabrina has 14 pencils, prove that m = 2. -/
theorem justin_sabrina_pencils (total : ℕ) (justin_extra : ℕ) (sabrina_pencils : ℕ) (m : ℕ) 
  (h1 : total = 50)
  (h2 : justin_extra = 8)
  (h3 : sabrina_pencils = 14)
  (h4 : total = (m * sabrina_pencils + justin_extra) + sabrina_pencils) :
  m = 2 := by sorry

end justin_sabrina_pencils_l1940_194050


namespace triangle_area_comparison_l1940_194046

/-- A triangle with side lengths and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

/-- Predicate to check if a triangle is acute -/
def is_acute (t : Triangle) : Prop := sorry

theorem triangle_area_comparison (t₁ t₂ : Triangle) 
  (h_acute : is_acute t₂)
  (h_a : t₁.a ≤ t₂.a)
  (h_b : t₁.b ≤ t₂.b)
  (h_c : t₁.c ≤ t₂.c) :
  t₁.area ≤ t₂.area :=
sorry

end triangle_area_comparison_l1940_194046


namespace annie_total_blocks_l1940_194014

/-- The total number of blocks Annie traveled -/
def total_blocks : ℕ :=
  let house_to_bus := 5
  let bus_to_train := 7
  let train_to_friend := 10
  let friend_to_coffee := 4
  2 * (house_to_bus + bus_to_train + train_to_friend) + 2 * friend_to_coffee

/-- Theorem stating that Annie traveled 52 blocks in total -/
theorem annie_total_blocks : total_blocks = 52 := by
  sorry

end annie_total_blocks_l1940_194014


namespace circle_distance_inequality_l1940_194069

theorem circle_distance_inequality (x y : ℝ) : 
  x^2 + y^2 + 2*x - 6*y = 6 → (x - 1)^2 + (y - 2)^2 ≠ 2 := by
sorry

end circle_distance_inequality_l1940_194069


namespace consecutive_integer_product_divisibility_l1940_194066

theorem consecutive_integer_product_divisibility (k : ℤ) : 
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 5 * m) →
  (∃ m : ℤ, n = 10 * m) ∧
  (∃ m : ℤ, n = 15 * m) ∧
  (∃ m : ℤ, n = 30 * m) ∧
  (∃ m : ℤ, n = 60 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 20 * m) :=
by sorry

end consecutive_integer_product_divisibility_l1940_194066


namespace ambulance_ride_cost_ambulance_cost_proof_l1940_194026

/-- Calculates the cost of an ambulance ride given a hospital bill breakdown -/
theorem ambulance_ride_cost (total_bill : ℝ) (medication_percentage : ℝ) 
  (overnight_percentage : ℝ) (food_cost : ℝ) : ℝ :=
  let medication_cost := medication_percentage * total_bill
  let remaining_after_medication := total_bill - medication_cost
  let overnight_cost := overnight_percentage * remaining_after_medication
  let ambulance_cost := total_bill - medication_cost - overnight_cost - food_cost
  ambulance_cost

/-- Proves that the ambulance ride cost is $1700 given specific bill details -/
theorem ambulance_cost_proof :
  ambulance_ride_cost 5000 0.5 0.25 175 = 1700 := by
  sorry

end ambulance_ride_cost_ambulance_cost_proof_l1940_194026


namespace lindas_furniture_fraction_l1940_194037

theorem lindas_furniture_fraction (savings : ℚ) (tv_cost : ℚ) 
  (h1 : savings = 920)
  (h2 : tv_cost = 230) :
  (savings - tv_cost) / savings = 3 / 4 := by
  sorry

end lindas_furniture_fraction_l1940_194037


namespace special_function_property_l1940_194084

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x + y)^2) = f x^2 + 2*x*(f y) + y^2

/-- The number of possible values of f(1) -/
def m (f : ℝ → ℝ) : ℕ := sorry

/-- The sum of all possible values of f(1) -/
def t (f : ℝ → ℝ) : ℝ := sorry

/-- The main theorem -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) : 
  (m f : ℝ) * t f = 1 := by sorry

end special_function_property_l1940_194084


namespace eel_count_l1940_194072

theorem eel_count (electric moray freshwater : ℕ) 
  (h1 : moray + freshwater = 12)
  (h2 : electric + freshwater = 14)
  (h3 : electric + moray = 16) :
  electric + moray + freshwater = 21 := by
sorry

end eel_count_l1940_194072


namespace davids_english_marks_l1940_194032

/-- Given David's marks in four subjects and his average across five subjects,
    prove that his marks in English must be 74. -/
theorem davids_english_marks
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (h1 : math_marks = 65)
  (h2 : physics_marks = 82)
  (h3 : chemistry_marks = 67)
  (h4 : biology_marks = 90)
  (h5 : average_marks = 75.6)
  (h6 : (math_marks + physics_marks + chemistry_marks + biology_marks + english_marks : ℚ) / 5 = average_marks)
  : english_marks = 74 := by
  sorry

#check davids_english_marks

end davids_english_marks_l1940_194032


namespace circle_area_difference_l1940_194096

theorem circle_area_difference : 
  let r1 : ℝ := 25
  let d2 : ℝ := 15
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1 ^ 2
  let area2 : ℝ := π * r2 ^ 2
  area1 - area2 = 568.75 * π := by sorry

end circle_area_difference_l1940_194096


namespace functional_equation_solution_l1940_194091

noncomputable def nondecreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem functional_equation_solution
  (f : ℝ → ℝ)
  (h_nondecreasing : nondecreasing_function f)
  (h_f_0 : f 0 = 0)
  (h_f_1 : f 1 = 1)
  (h_equation : ∀ a b, a < 1 ∧ 1 < b → f a + f b = f a * f b + f (a + b - a * b)) :
  ∃ c k, c > 0 ∧ k ≥ 0 ∧
    (∀ x, x > 1 → f x = c * (x - 1) ^ k) ∧
    (∀ x, x < 1 → f x = 1 - (1 - x) ^ k) :=
by sorry

end functional_equation_solution_l1940_194091


namespace cylinder_volume_l1940_194055

/-- Given a cylinder with lateral surface area 100π cm² and an inscribed rectangular solid
    with diagonal length 10√2 cm, prove that the cylinder's volume is 250π cm³. -/
theorem cylinder_volume (r h : ℝ) (lateral_area : 2 * Real.pi * r * h = 100 * Real.pi)
    (diagonal_length : 4 * r^2 + h^2 = 200) : Real.pi * r^2 * h = 250 * Real.pi :=
by sorry

end cylinder_volume_l1940_194055


namespace erased_number_proof_l1940_194060

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  n > 1 →
  (n : ℝ) * ((n : ℝ) + 21) / 2 - x = 23 * ((n : ℝ) - 1) →
  x = 36 := by
sorry

end erased_number_proof_l1940_194060


namespace michael_birdhouse_earnings_l1940_194051

/-- The amount of money Michael made from selling birdhouses -/
def michael_earnings (extra_large_price large_price medium_price small_price extra_small_price : ℕ)
  (extra_large_sold large_sold medium_sold small_sold extra_small_sold : ℕ) : ℕ :=
  extra_large_price * extra_large_sold +
  large_price * large_sold +
  medium_price * medium_sold +
  small_price * small_sold +
  extra_small_price * extra_small_sold

/-- Theorem stating that Michael's earnings from selling birdhouses is $487 -/
theorem michael_birdhouse_earnings :
  michael_earnings 45 22 16 10 5 3 5 7 8 10 = 487 := by sorry

end michael_birdhouse_earnings_l1940_194051


namespace actual_number_is_two_l1940_194018

-- Define the set of people
inductive Person
| Natasha
| Boy1
| Boy2
| Girl1
| Girl2

-- Define a function to represent claims about the number
def claim (p : Person) (n : Nat) : Prop :=
  match p with
  | Person.Natasha => n % 15 = 0
  | _ => true  -- We don't have specific information about other claims

-- Define the conditions of the problem
axiom one_boy_correct : ∃ (b : Person), b = Person.Boy1 ∨ b = Person.Boy2
axiom one_girl_correct : ∃ (g : Person), g = Person.Girl1 ∨ g = Person.Girl2
axiom two_wrong : ∃ (p1 p2 : Person), p1 ≠ p2 ∧ ¬(claim p1 2) ∧ ¬(claim p2 2)

-- The theorem to prove
theorem actual_number_is_two : 
  ∃ (n : Nat), (claim Person.Natasha n = false) ∧ n = 2 :=
sorry

end actual_number_is_two_l1940_194018


namespace matrix_product_equal_l1940_194029

/-- A 3x3 matrix of natural numbers -/
def Matrix3x3 := Fin 3 → Fin 3 → ℕ

/-- Check if all numbers in the matrix are distinct and not exceeding 40 -/
def valid_matrix (m : Matrix3x3) : Prop :=
  (∀ i j, m i j ≤ 40) ∧
  (∀ i j i' j', (i, j) ≠ (i', j') → m i j ≠ m i' j')

/-- Calculate the product of a row -/
def row_product (m : Matrix3x3) (i : Fin 3) : ℕ :=
  (m i 0) * (m i 1) * (m i 2)

/-- Calculate the product of a column -/
def col_product (m : Matrix3x3) (j : Fin 3) : ℕ :=
  (m 0 j) * (m 1 j) * (m 2 j)

/-- Calculate the product of the main diagonal -/
def diag1_product (m : Matrix3x3) : ℕ :=
  (m 0 0) * (m 1 1) * (m 2 2)

/-- Calculate the product of the other diagonal -/
def diag2_product (m : Matrix3x3) : ℕ :=
  (m 0 2) * (m 1 1) * (m 2 0)

/-- The given matrix -/
def given_matrix : Matrix3x3
| 0, 0 => 12
| 0, 1 => 9
| 0, 2 => 2
| 1, 0 => 1
| 1, 1 => 6
| 1, 2 => 36
| 2, 0 => 18
| 2, 1 => 4
| 2, 2 => 3

theorem matrix_product_equal :
  valid_matrix given_matrix ∧
  ∃ p : ℕ, (∀ i : Fin 3, row_product given_matrix i = p) ∧
           (∀ j : Fin 3, col_product given_matrix j = p) ∧
           (diag1_product given_matrix = p) ∧
           (diag2_product given_matrix = p) :=
by sorry

end matrix_product_equal_l1940_194029


namespace room_area_in_sq_meters_l1940_194044

/-- The conversion factor from square feet to square meters -/
def sq_ft_to_sq_m : ℝ := 0.092903

/-- The length of the room in feet -/
def room_length : ℝ := 15

/-- The width of the room in feet -/
def room_width : ℝ := 8

/-- Theorem stating that the area of the room in square meters is approximately 11.14836 -/
theorem room_area_in_sq_meters :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |room_length * room_width * sq_ft_to_sq_m - 11.14836| < ε :=
sorry

end room_area_in_sq_meters_l1940_194044


namespace fraction_to_decimal_l1940_194083

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by sorry

end fraction_to_decimal_l1940_194083


namespace grace_age_calculation_l1940_194059

/-- Grace's age in years -/
def grace_age : ℕ := sorry

/-- Grace's mother's age in years -/
def mother_age : ℕ := 80

/-- Grace's grandmother's age in years -/
def grandmother_age : ℕ := sorry

/-- Theorem stating Grace's age based on the given conditions -/
theorem grace_age_calculation :
  (grace_age = 3 * grandmother_age / 8) ∧
  (grandmother_age = 2 * mother_age) ∧
  (mother_age = 80) →
  grace_age = 60 := by
    sorry

end grace_age_calculation_l1940_194059


namespace iphone_cost_l1940_194049

/-- The cost of the new iPhone given trade-in value, weekly earnings, and work duration -/
theorem iphone_cost (trade_in_value : ℕ) (weekly_earnings : ℕ) (work_weeks : ℕ) : 
  trade_in_value = 240 → weekly_earnings = 80 → work_weeks = 7 →
  trade_in_value + weekly_earnings * work_weeks = 800 := by
  sorry

end iphone_cost_l1940_194049


namespace multiplication_problems_l1940_194089

theorem multiplication_problems :
  (25 * 5 * 2 * 4 = 1000) ∧ (1111 * 9999 = 11108889) := by
  sorry

end multiplication_problems_l1940_194089


namespace tan_fifteen_pi_fourths_l1940_194076

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by
  sorry

end tan_fifteen_pi_fourths_l1940_194076


namespace ratio_comparison_is_three_l1940_194020

/-- Represents the ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
| 0 => 1
| 1 => 12
| 2 => 30

/-- The ratio of flavoring to water in the sport formulation is half that of the standard formulation -/
def sport_water_ratio : ℚ := standard_ratio 2 * 2

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 5

/-- Amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 75

/-- The ratio of flavoring to corn syrup in the sport formulation compared to the standard formulation -/
def ratio_comparison : ℚ :=
  (sport_water / sport_water_ratio) / sport_corn_syrup /
  (standard_ratio 0 / standard_ratio 1)

theorem ratio_comparison_is_three : ratio_comparison = 3 := by
  sorry

end ratio_comparison_is_three_l1940_194020


namespace complement_intersection_theorem_l1940_194036

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by
  sorry

end complement_intersection_theorem_l1940_194036


namespace inscribed_square_area_l1940_194099

theorem inscribed_square_area (XY ZC : ℝ) (h1 : XY = 40) (h2 : ZC = 70) :
  let s := Real.sqrt (XY * ZC)
  s * s = 2800 := by sorry

end inscribed_square_area_l1940_194099


namespace even_n_with_specific_digit_sums_l1940_194071

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem even_n_with_specific_digit_sums 
  (n : ℕ) 
  (n_positive : 0 < n) 
  (sum_n : sum_of_digits n = 2014) 
  (sum_5n : sum_of_digits (5 * n) = 1007) : 
  Even n := by sorry

end even_n_with_specific_digit_sums_l1940_194071


namespace percent_to_decimal_four_percent_to_decimal_l1940_194093

theorem percent_to_decimal (x : ℚ) :
  x / 100 = x * (1 / 100) := by sorry

theorem four_percent_to_decimal :
  (4 : ℚ) / 100 = (4 : ℚ) * (1 / 100) ∧ (4 : ℚ) * (1 / 100) = 0.04 := by sorry

end percent_to_decimal_four_percent_to_decimal_l1940_194093


namespace points_per_enemy_l1940_194012

theorem points_per_enemy (num_enemies : ℕ) (completion_bonus : ℕ) (total_points : ℕ) 
  (h1 : num_enemies = 6)
  (h2 : completion_bonus = 8)
  (h3 : total_points = 62) :
  (total_points - completion_bonus) / num_enemies = 9 := by
  sorry

end points_per_enemy_l1940_194012


namespace min_value_on_interval_l1940_194008

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the theorem
theorem min_value_on_interval (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 6), f x ≥ 9) ∧ (∃ x ∈ Set.Icc a (a + 6), f x = 9) ↔ a = 2 ∨ a = -10 :=
sorry

end min_value_on_interval_l1940_194008


namespace no_real_solutions_l1940_194027

theorem no_real_solutions : ¬∃ (x : ℝ), x + 48 / (x - 3) = -1 := by
  sorry

end no_real_solutions_l1940_194027


namespace uncertain_mushrooms_l1940_194080

/-- Given the total number of mushrooms, the number of safe mushrooms, and the relationship
    between safe and poisonous mushrooms, prove that the number of uncertain mushrooms is 5. -/
theorem uncertain_mushrooms (total : ℕ) (safe : ℕ) (poisonous : ℕ) :
  total = 32 →
  safe = 9 →
  poisonous = 2 * safe →
  total - (safe + poisonous) = 5 := by
  sorry

end uncertain_mushrooms_l1940_194080


namespace cube_edge_length_proof_l1940_194045

/-- The edge length of a cube that, when fully immersed in a rectangular vessel
    with base dimensions 20 cm × 15 cm, causes a water level rise of 5.76 cm. -/
def cube_edge_length : ℝ := 12

/-- The base area of the rectangular vessel in square centimeters. -/
def vessel_base_area : ℝ := 20 * 15

/-- The rise in water level in centimeters when the cube is fully immersed. -/
def water_level_rise : ℝ := 5.76

/-- The volume of water displaced by the cube in cubic centimeters. -/
def displaced_volume : ℝ := vessel_base_area * water_level_rise

theorem cube_edge_length_proof :
  cube_edge_length ^ 3 = displaced_volume :=
by sorry

end cube_edge_length_proof_l1940_194045


namespace hill_climbing_speed_l1940_194042

/-- Proves that given a round trip journey where the total time is 6 hours (4 hours up, 2 hours down)
    and the average speed for the whole journey is 4 km/h, then the average speed while climbing
    to the top is 3 km/h. -/
theorem hill_climbing_speed
  (total_time : ℝ)
  (up_time : ℝ)
  (down_time : ℝ)
  (average_speed : ℝ)
  (h1 : total_time = up_time + down_time)
  (h2 : total_time = 6)
  (h3 : up_time = 4)
  (h4 : down_time = 2)
  (h5 : average_speed = 4) :
  (average_speed * total_time) / (2 * up_time) = 3 := by
  sorry

end hill_climbing_speed_l1940_194042


namespace pages_copied_for_fifteen_dollars_l1940_194021

/-- Given that 5 pages cost 10 cents, prove that $15 can copy 750 pages. -/
theorem pages_copied_for_fifteen_dollars : 
  let cost_per_five_pages : ℚ := 10 / 100  -- 10 cents in dollars
  let total_amount : ℚ := 15  -- $15
  let pages_per_dollar : ℚ := 5 / cost_per_five_pages
  ⌊total_amount * pages_per_dollar⌋ = 750 :=
by sorry

end pages_copied_for_fifteen_dollars_l1940_194021


namespace pizza_area_increase_l1940_194077

/-- The percentage increase in area from a circle with radius 5 to a circle with radius 7 -/
theorem pizza_area_increase : ∀ (π : ℝ), π > 0 →
  (π * 7^2 - π * 5^2) / (π * 5^2) * 100 = 96 := by
  sorry

end pizza_area_increase_l1940_194077


namespace remainder_problem_l1940_194098

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k % 5 = 2) 
  (h3 : k < 41) 
  (h4 : k % 7 = 3) : 
  k % 6 = 5 := by
sorry

end remainder_problem_l1940_194098


namespace gcd_of_multiple_4500_l1940_194056

theorem gcd_of_multiple_4500 (k : ℤ) : 
  let b : ℤ := 4500 * k
  Int.gcd (b^2 + 11*b + 40) (b + 8) = 3 := by
sorry

end gcd_of_multiple_4500_l1940_194056


namespace new_persons_combined_weight_l1940_194097

/-- The combined weight of two new persons in a group, given specific conditions --/
theorem new_persons_combined_weight
  (n : ℕ)
  (avg_increase : ℝ)
  (old_weight1 old_weight2 : ℝ)
  (h1 : n = 15)
  (h2 : avg_increase = 5.2)
  (h3 : old_weight1 = 68)
  (h4 : old_weight2 = 70) :
  ∃ (w1 w2 : ℝ), w1 + w2 = 216 :=
sorry

end new_persons_combined_weight_l1940_194097


namespace difference_of_squares_l1940_194047

theorem difference_of_squares (m n : ℝ) : (m + n) * (-m + n) = -m^2 + n^2 := by
  sorry

end difference_of_squares_l1940_194047


namespace inverse_variation_problem_l1940_194019

theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x^2 * y^5 = k) 
  (h2 : x = 5 ∧ y = 2 → k = 800) :
  y = 4 → x^2 = 25/32 := by
sorry

end inverse_variation_problem_l1940_194019


namespace product_of_sum_and_difference_l1940_194078

theorem product_of_sum_and_difference (a b : ℝ) 
  (sum_eq : a + b = 7)
  (diff_eq : a - b = 2) :
  a * b = 11.25 := by
  sorry

end product_of_sum_and_difference_l1940_194078


namespace arithmetic_expression_equality_l1940_194028

theorem arithmetic_expression_equality : (5 * 4)^2 + (10 * 2) - 36 / 3 = 408 := by
  sorry

end arithmetic_expression_equality_l1940_194028


namespace cubic_root_sum_l1940_194082

theorem cubic_root_sum (p q r : ℂ) : 
  (p^3 - 2*p - 2 = 0) → 
  (q^3 - 2*q - 2 = 0) → 
  (r^3 - 2*r - 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -6 := by
sorry

end cubic_root_sum_l1940_194082


namespace general_equation_l1940_194067

theorem general_equation (n : ℝ) : n ≠ 4 ∧ 8 - n ≠ 4 → 
  (n / (n - 4)) + ((8 - n) / ((8 - n) - 4)) = 2 := by sorry

end general_equation_l1940_194067


namespace figure_area_l1940_194043

/-- The total area of a figure composed of four rectangles with given dimensions --/
def total_area (r1_height r1_width r2_height r2_width r3_height r3_width r4_height r4_width : ℕ) : ℕ :=
  r1_height * r1_width + r2_height * r2_width + r3_height * r3_width + r4_height * r4_width

/-- Theorem stating that the total area of the given figure is 89 square units --/
theorem figure_area : total_area 7 6 2 6 5 4 3 5 = 89 := by
  sorry

end figure_area_l1940_194043


namespace parabola_shift_equation_l1940_194048

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (k : ℝ) : Parabola :=
  { f := λ x => p.f (x + k) }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (m : ℝ) : Parabola :=
  { f := λ x => p.f x - m }

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { f := λ x => x^2 }

/-- The resulting parabola after shifting -/
def shifted_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 3) 4

theorem parabola_shift_equation :
  ∀ x, shifted_parabola.f x = (x + 3)^2 - 4 :=
sorry

end parabola_shift_equation_l1940_194048


namespace polynomial_zero_l1940_194065

-- Define the polynomial
def P (x : ℂ) (p q α β : ℤ) : ℂ := 
  (x - p) * (x - q) * (x^2 + α*x + β)

-- State the theorem
theorem polynomial_zero (p q : ℤ) : 
  ∃ (α β : ℤ), P ((3 + Complex.I * Real.sqrt 15) / 2) p q α β = 0 := by
  sorry

end polynomial_zero_l1940_194065


namespace remainder_problem_l1940_194016

theorem remainder_problem :
  (85^70 + 19^32)^16 ≡ 16 [ZMOD 21] := by
  sorry

end remainder_problem_l1940_194016


namespace arithmetic_sequence_sum_l1940_194024

def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  is_arithmetic_sequence a →
  is_arithmetic_sequence b →
  a 1 = 15 →
  b 1 = 35 →
  a 2 + b 2 = 60 →
  a 36 + b 36 = 400 := by
sorry

end arithmetic_sequence_sum_l1940_194024


namespace stock_percentage_sold_l1940_194063

/-- 
Given:
- cash_realized: The cash realized on selling the stock
- brokerage_rate: The brokerage rate as a percentage
- total_amount: The total amount including brokerage

Prove that the percentage of stock sold is equal to 
(cash_realized / (total_amount - total_amount * brokerage_rate / 100)) * 100
-/
theorem stock_percentage_sold 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (total_amount : ℝ) 
  (h1 : cash_realized = 104.25)
  (h2 : brokerage_rate = 0.25)
  (h3 : total_amount = 104) :
  (cash_realized / (total_amount - total_amount * brokerage_rate / 100)) * 100 = 
    (104.25 / (104 - 104 * 0.25 / 100)) * 100 := by
  sorry

end stock_percentage_sold_l1940_194063


namespace f_decreasing_on_interval_l1940_194003

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 30*x - 33

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 11, f' x < 0 :=
by sorry

end f_decreasing_on_interval_l1940_194003


namespace sin_45_75_plus_sin_45_15_l1940_194005

theorem sin_45_75_plus_sin_45_15 :
  Real.sin (45 * π / 180) * Real.sin (75 * π / 180) +
  Real.sin (45 * π / 180) * Real.sin (15 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end sin_45_75_plus_sin_45_15_l1940_194005


namespace cubic_roots_property_l1940_194040

/-- 
Given a cubic polynomial x^3 + cx^2 + dx + 16c where c and d are nonzero integers,
if two of its roots coincide and all three roots are integers, then |cd| = 2560.
-/
theorem cubic_roots_property (c d : ℤ) (hc : c ≠ 0) (hd : d ≠ 0) : 
  (∃ p q : ℤ, (∀ x : ℝ, x^3 + c*x^2 + d*x + 16*c = (x - p)^2 * (x - q))) →
  |c*d| = 2560 := by
  sorry

end cubic_roots_property_l1940_194040


namespace hyperbola_asymptote_slope_l1940_194023

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 / 4 = 1

/-- The asymptote equation -/
def asymptote (m : ℝ) (x y : ℝ) : Prop := y = m * x ∨ y = -m * x

/-- Theorem: The positive slope of the asymptotes of the given hyperbola is 3/2 -/
theorem hyperbola_asymptote_slope :
  ∃ (m : ℝ), m > 0 ∧ 
  (∀ (x y : ℝ), hyperbola x y → (∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), δ > ε → asymptote m (x + δ) (y + δ))) ∧
  m = 3/2 := by
sorry

end hyperbola_asymptote_slope_l1940_194023


namespace enemy_count_l1940_194079

theorem enemy_count (points_per_enemy : ℕ) (points_earned : ℕ) (enemies_left : ℕ) :
  points_per_enemy = 8 →
  enemies_left = 2 →
  points_earned = 40 →
  ∃ (total_enemies : ℕ), total_enemies = 7 ∧ points_per_enemy * (total_enemies - enemies_left) = points_earned :=
by sorry

end enemy_count_l1940_194079


namespace nonagon_diagonals_l1940_194095

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ :=
  let n : ℕ := 9  -- number of sides in a nonagon
  (n * (n - 3)) / 2

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals :
  num_diagonals_nonagon = 27 := by
  sorry

end nonagon_diagonals_l1940_194095


namespace boys_without_glasses_in_class_l1940_194030

/-- The number of boys who do not wear glasses in Mrs. Lee's class -/
def boys_without_glasses (total_boys : ℕ) (total_with_glasses : ℕ) (girls_with_glasses : ℕ) : ℕ :=
  total_boys - (total_with_glasses - girls_with_glasses)

/-- Theorem stating the number of boys without glasses in Mrs. Lee's class -/
theorem boys_without_glasses_in_class : boys_without_glasses 30 36 21 = 15 := by
  sorry

end boys_without_glasses_in_class_l1940_194030


namespace vertical_line_intercept_difference_l1940_194007

/-- A vertical line passing through two points -/
structure VerticalLine where
  x : ℝ
  y1 : ℝ
  y2 : ℝ

/-- The x-intercept of a vertical line -/
def x_intercept (l : VerticalLine) : ℝ := l.x

/-- Theorem: For a vertical line passing through points C(7, 5) and D(7, -3),
    the difference between the x-intercept of the line and the y-coordinate of point C is 2 -/
theorem vertical_line_intercept_difference (l : VerticalLine) 
    (h1 : l.x = 7) 
    (h2 : l.y1 = 5) 
    (h3 : l.y2 = -3) : 
  x_intercept l - l.y1 = 2 := by
  sorry

end vertical_line_intercept_difference_l1940_194007


namespace new_stationary_points_order_l1940_194073

-- Define the "new stationary point" for each function
def alpha : ℝ := 1

-- β is implicitly defined by the equation ln(β+1) = 1/(β+1)
def beta_equation (x : ℝ) : Prop := Real.log (x + 1) = 1 / (x + 1) ∧ x > 0

-- γ is implicitly defined by the equation γ³ - 1 = 3γ²
def gamma_equation (x : ℝ) : Prop := x^3 - 1 = 3 * x^2 ∧ x > 0

-- State the theorem
theorem new_stationary_points_order 
  (beta : ℝ) (h_beta : beta_equation beta)
  (gamma : ℝ) (h_gamma : gamma_equation gamma) :
  gamma > alpha ∧ alpha > beta := by
  sorry

end new_stationary_points_order_l1940_194073


namespace price_change_difference_l1940_194004

/-- 
Given that a price is increased by x percent and then decreased by y percent, 
resulting in the same price as the initial price, prove that 1/x - 1/y = -1/100.
-/
theorem price_change_difference (x y : ℝ) 
  (h : (1 + x/100) * (1 - y/100) = 1) : 
  1/x - 1/y = -1/100 :=
sorry

end price_change_difference_l1940_194004
