import Mathlib

namespace g_critical_points_l1977_197712

noncomputable def g (x : ℝ) : ℝ :=
  if -3 < x ∧ x ≤ 0 then -x - 3
  else if 0 < x ∧ x ≤ 2 then x - 3
  else if 2 < x ∧ x ≤ 3 then x^2 - 4*x + 6
  else 0  -- Default value for x outside the defined range

def is_critical_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → f y ≤ f x

def is_local_minimum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ δ > 0, ∀ y, |y - x| < δ → f x ≤ f y

theorem g_critical_points :
  is_critical_point g 0 ∧ 
  is_critical_point g 2 ∧
  is_local_minimum g 2 :=
sorry

end g_critical_points_l1977_197712


namespace triangle_area_is_three_l1977_197729

/-- Triangle ABC with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The line equation x - y = 5 -/
def LineEquation (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 = 5

/-- The area of a triangle -/
def TriangleArea (t : Triangle) : ℝ :=
  sorry

/-- The theorem statement -/
theorem triangle_area_is_three :
  ∀ (t : Triangle),
    t.A = (3, 0) →
    t.B = (0, 3) →
    LineEquation t.C →
    TriangleArea t = 3 := by
  sorry

end triangle_area_is_three_l1977_197729


namespace cubic_root_sum_log_l1977_197764

theorem cubic_root_sum_log (a b : ℝ) : 
  (∃ r s t : ℝ, r > 0 ∧ s > 0 ∧ t > 0 ∧ 
   r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
   16 * r^3 + 7 * a * r^2 + 6 * b * r + 2 * a = 0 ∧
   16 * s^3 + 7 * a * s^2 + 6 * b * s + 2 * a = 0 ∧
   16 * t^3 + 7 * a * t^2 + 6 * b * t + 2 * a = 0 ∧
   Real.log r / Real.log 4 + Real.log s / Real.log 4 + Real.log t / Real.log 4 = 3) →
  a = -512 :=
by sorry

end cubic_root_sum_log_l1977_197764


namespace f_of_five_equals_sixtytwo_l1977_197768

/-- Given a function f where f(x) = 2x² + y and f(2) = 20, prove that f(5) = 62 -/
theorem f_of_five_equals_sixtytwo (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 20) : 
  f 5 = 62 := by
  sorry

end f_of_five_equals_sixtytwo_l1977_197768


namespace range_of_increasing_function_l1977_197727

/-- Given an increasing function f: ℝ → ℝ with f(0) = -1 and f(3) = 1,
    the set of x ∈ ℝ such that |f(x+1)| < 1 is equal to [-1, 2] -/
theorem range_of_increasing_function (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_f_0 : f 0 = -1) 
  (h_f_3 : f 3 = 1) : 
  {x : ℝ | |f (x + 1)| < 1} = Set.Icc (-1) 2 := by
  sorry

end range_of_increasing_function_l1977_197727


namespace inverse_equivalent_is_contrapositive_l1977_197736

theorem inverse_equivalent_is_contrapositive (p q : Prop) :
  (q → p) ↔ (¬p → ¬q) :=
sorry

end inverse_equivalent_is_contrapositive_l1977_197736


namespace cos_two_pi_thirds_l1977_197707

theorem cos_two_pi_thirds : Real.cos (2 * Real.pi / 3) = -(1 / 2) := by
  sorry

end cos_two_pi_thirds_l1977_197707


namespace geometric_sum_first_six_terms_l1977_197756

theorem geometric_sum_first_six_terms :
  let a : ℚ := 1/2  -- First term
  let r : ℚ := 1/3  -- Common ratio
  let n : ℕ := 6    -- Number of terms
  let S : ℚ := a * (1 - r^n) / (1 - r)  -- Formula for sum of geometric series
  S = 364/243
  := by sorry

end geometric_sum_first_six_terms_l1977_197756


namespace binomial_10_3_l1977_197740

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l1977_197740


namespace deal_or_no_deal_probability_l1977_197747

theorem deal_or_no_deal_probability (total_boxes : ℕ) (desired_boxes : ℕ) (eliminated_boxes : ℕ) : 
  total_boxes = 30 →
  desired_boxes = 6 →
  eliminated_boxes = 18 →
  (desired_boxes : ℚ) / (total_boxes - eliminated_boxes : ℚ) ≥ 1/2 :=
by sorry

end deal_or_no_deal_probability_l1977_197747


namespace gardening_project_cost_l1977_197700

-- Define constants for the given conditions
def rose_bushes : Nat := 20
def fruit_trees : Nat := 10
def ornamental_shrubs : Nat := 5
def rose_bush_cost : Nat := 150
def fertilizer_cost : Nat := 25
def fruit_tree_cost : Nat := 75
def ornamental_shrub_cost : Nat := 50
def gardener_hourly_rate : Nat := 30
def soil_cost_per_cubic_foot : Nat := 5
def soil_needed : Nat := 100
def tiller_cost_per_day : Nat := 40
def wheelbarrow_cost_per_day : Nat := 10
def rental_days : Nat := 3

def gardener_hours : List Nat := [6, 5, 4, 7]

-- Define functions for calculations
def rose_bush_total_cost : Nat :=
  let base_cost := rose_bushes * rose_bush_cost
  let discount := base_cost * 5 / 100
  base_cost - discount

def fertilizer_total_cost : Nat :=
  let base_cost := rose_bushes * fertilizer_cost
  let discount := base_cost * 10 / 100
  base_cost - discount

def fruit_tree_total_cost : Nat :=
  let free_trees := fruit_trees / 3
  let paid_trees := fruit_trees - free_trees
  paid_trees * fruit_tree_cost

def ornamental_shrub_total_cost : Nat :=
  ornamental_shrubs * ornamental_shrub_cost

def gardener_total_cost : Nat :=
  (gardener_hours.sum) * gardener_hourly_rate

def soil_total_cost : Nat :=
  soil_needed * soil_cost_per_cubic_foot

def tools_rental_total_cost : Nat :=
  (tiller_cost_per_day + wheelbarrow_cost_per_day) * rental_days

-- Define the total cost of the gardening project
def total_gardening_cost : Nat :=
  rose_bush_total_cost +
  fertilizer_total_cost +
  fruit_tree_total_cost +
  ornamental_shrub_total_cost +
  gardener_total_cost +
  soil_total_cost +
  tools_rental_total_cost

-- Theorem statement
theorem gardening_project_cost :
  total_gardening_cost = 6385 := by sorry

end gardening_project_cost_l1977_197700


namespace bathroom_length_proof_l1977_197797

/-- Proves the length of a rectangular bathroom given its width, tile size, and number of tiles needed --/
theorem bathroom_length_proof (width : ℝ) (tile_side : ℝ) (num_tiles : ℕ) (length : ℝ) : 
  width = 6 →
  tile_side = 0.5 →
  num_tiles = 240 →
  width * length = (tile_side * tile_side) * num_tiles →
  length = 10 := by
sorry

end bathroom_length_proof_l1977_197797


namespace bill_total_l1977_197796

/-- Proves that if three people divide a bill evenly and each pays $33, then the total bill is $99. -/
theorem bill_total (people : Fin 3 → ℕ) (h : ∀ i, people i = 33) : 
  (Finset.univ.sum people) = 99 := by
  sorry

end bill_total_l1977_197796


namespace fixed_point_on_line_l1977_197777

theorem fixed_point_on_line (k : ℝ) : k * 1 - k = 0 := by
  sorry

end fixed_point_on_line_l1977_197777


namespace divide_ten_items_between_two_people_l1977_197788

theorem divide_ten_items_between_two_people : 
  Nat.choose 10 5 = 252 := by
  sorry

end divide_ten_items_between_two_people_l1977_197788


namespace pizza_fraction_proof_l1977_197790

theorem pizza_fraction_proof (michael_fraction lamar_fraction treshawn_fraction : ℚ) : 
  michael_fraction = 1/3 →
  lamar_fraction = 1/6 →
  michael_fraction + lamar_fraction + treshawn_fraction = 1 →
  treshawn_fraction = 1/2 := by
sorry

end pizza_fraction_proof_l1977_197790


namespace max_distance_complex_l1977_197765

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (max_dist : ℝ), max_dist = 729 + 162 * Real.sqrt 13 ∧
  ∀ (w : ℂ), Complex.abs w = 3 →
    Complex.abs ((2 + 3*Complex.I)*(w^4) - w^6) ≤ max_dist :=
sorry

end max_distance_complex_l1977_197765


namespace water_consumption_l1977_197751

/-- Proves that given a 1.5-quart bottle of water and a can of water,
    if the total amount of water drunk is 60 ounces,
    and 1 quart is equivalent to 32 ounces,
    then the can of water contains 12 ounces. -/
theorem water_consumption (bottle : ℚ) (can : ℚ) (total : ℚ) (quart_to_ounce : ℚ → ℚ) :
  bottle = 1.5 →
  total = 60 →
  quart_to_ounce 1 = 32 →
  can = total - quart_to_ounce bottle :=
by
  sorry

end water_consumption_l1977_197751


namespace adams_shopping_cost_l1977_197787

/-- Calculates the total cost of Adam's shopping given the specified conditions --/
def calculate_total_cost (sandwich_price : ℚ) (sandwich_count : ℕ) 
                         (chip_price : ℚ) (chip_count : ℕ) 
                         (water_price : ℚ) (water_count : ℕ) : ℚ :=
  let sandwich_cost := (sandwich_count - 1) * sandwich_price
  let chip_cost := chip_count * chip_price * (1 - 0.2)
  let water_cost := water_count * water_price * 1.05
  sandwich_cost + chip_cost + water_cost

/-- Theorem stating that Adam's total shopping cost is $31.75 --/
theorem adams_shopping_cost : 
  calculate_total_cost 4 5 3.5 3 1.75 4 = 31.75 := by
  sorry

end adams_shopping_cost_l1977_197787


namespace quadratic_function_theorem_l1977_197711

/-- A quadratic function f(x) = x^2 + bx + 3 where b is a real number -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- The theorem stating that if the range of f is [0, +∞) and the solution set of f(x) < c
    is an open interval of length 8, then c = 16 -/
theorem quadratic_function_theorem (b : ℝ) (c : ℝ) :
  (∀ x, f b x ≥ 0) →
  (∃ m, ∀ x, f b x < c ↔ m - 8 < x ∧ x < m) →
  c = 16 :=
sorry

end quadratic_function_theorem_l1977_197711


namespace joe_market_spend_l1977_197726

/-- Calculates the total cost of Joe's market purchases -/
def market_total_cost (orange_price : ℚ) (juice_price : ℚ) (honey_price : ℚ) (plant_pair_price : ℚ)
  (orange_count : ℕ) (juice_count : ℕ) (honey_count : ℕ) (plant_count : ℕ) : ℚ :=
  orange_price * orange_count +
  juice_price * juice_count +
  honey_price * honey_count +
  plant_pair_price * (plant_count / 2)

/-- Theorem stating that Joe's total market spend is $68 -/
theorem joe_market_spend :
  market_total_cost 4.5 0.5 5 18 3 7 3 4 = 68 := by
  sorry

end joe_market_spend_l1977_197726


namespace arithmetic_sequence_n_values_l1977_197778

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, a 1 = 1 ∧ ∀ n ≥ 3, a n = 100 ∧ ∀ k : ℕ, a (k + 1) = a k + d

/-- The set of possible n values -/
def PossibleN : Set ℕ := {4, 10, 12, 34, 100}

/-- The main theorem -/
theorem arithmetic_sequence_n_values (a : ℕ → ℕ) :
  ArithmeticSequence a →
  (∀ n : ℕ, n ∈ PossibleN ↔ (n ≥ 3 ∧ a n = 100)) :=
sorry

end arithmetic_sequence_n_values_l1977_197778


namespace inequality_relationship_l1977_197721

theorem inequality_relationship (x : ℝ) : 
  ¬(((x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧ 
    ((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) :=
sorry

end inequality_relationship_l1977_197721


namespace f_range_l1977_197758

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.cos x ^ 2 - 3/4) + Real.sin x

theorem f_range : Set.range f = Set.Icc (-1/2) (Real.sqrt 2 / 2) := by
  sorry

end f_range_l1977_197758


namespace arithmetic_seq_bicolored_l1977_197710

/-- A coloring function for natural numbers -/
def coloring (n : ℕ) : Bool :=
  let segment := (Nat.sqrt (8 * n + 1) - 1) / 2
  segment % 2 = 0

/-- Definition of an arithmetic sequence -/
def isArithmeticSeq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + r

/-- Theorem stating that every infinite arithmetic sequence is bi-colored -/
theorem arithmetic_seq_bicolored :
  ∀ (a : ℕ → ℕ) (r : ℕ), isArithmeticSeq a r →
  (∃ k, coloring (a k) = true) ∧ (∃ m, coloring (a m) = false) :=
sorry

end arithmetic_seq_bicolored_l1977_197710


namespace arithmetic_sequence_common_difference_l1977_197701

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_mean1 : (a 1 + a 2) / 2 = 1)
  (h_mean2 : (a 2 + a 3) / 2 = 2) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 1 :=
sorry

end arithmetic_sequence_common_difference_l1977_197701


namespace circle_M_equation_l1977_197763

/-- A circle M with the following properties:
    1. Tangent to the y-axis
    2. Its center lies on the line y = 1/2x
    3. The chord it cuts on the x-axis is 2√3 long -/
structure CircleM where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_y_axis : abs (center.1) = radius
  center_on_line : center.2 = 1/2 * center.1
  x_axis_chord : 2 * radius = 2 * Real.sqrt 3

/-- The standard equation of circle M is either (x-2)² + (y-1)² = 4 or (x+2)² + (y+1)² = 4 -/
theorem circle_M_equation (M : CircleM) :
  (∀ x y, (x - 2)^2 + (y - 1)^2 = 4) ∨ (∀ x y, (x + 2)^2 + (y + 1)^2 = 4) := by
  sorry

end circle_M_equation_l1977_197763


namespace expression_simplification_l1977_197715

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 1 - (2 * x - 2) / (x + 1)) / ((x^2 - x) / (2 * x + 2)) = 2 - Real.sqrt 2 := by
  sorry

end expression_simplification_l1977_197715


namespace rocket_height_problem_l1977_197779

theorem rocket_height_problem (h : ℝ) : 
  h + 2 * h = 1500 → h = 500 := by
  sorry

end rocket_height_problem_l1977_197779


namespace chocolate_savings_theorem_l1977_197781

/-- Represents the cost and packaging details of a chocolate store -/
structure ChocolateStore where
  cost_per_chocolate : ℚ
  pack_size : ℕ

/-- Calculates the cost for a given number of weeks at a store -/
def calculate_cost (store : ChocolateStore) (weeks : ℕ) : ℚ :=
  let chocolates_needed := 2 * weeks
  let packs_needed := (chocolates_needed + store.pack_size - 1) / store.pack_size
  ↑packs_needed * store.pack_size * store.cost_per_chocolate

/-- The problem statement -/
theorem chocolate_savings_theorem :
  let local_store := ChocolateStore.mk 3 1
  let store_a := ChocolateStore.mk 2 5
  let store_b := ChocolateStore.mk (5/2) 1
  let store_c := ChocolateStore.mk (9/5) 10
  let weeks := 13
  let local_cost := calculate_cost local_store weeks
  let cost_a := calculate_cost store_a weeks
  let cost_b := calculate_cost store_b weeks
  let cost_c := calculate_cost store_c weeks
  let savings_a := local_cost - cost_a
  let savings_b := local_cost - cost_b
  let savings_c := local_cost - cost_c
  let max_savings := max savings_a (max savings_b savings_c)
  max_savings = 28 := by sorry

end chocolate_savings_theorem_l1977_197781


namespace parabolas_intersection_k_l1977_197702

/-- Two different parabolas that intersect on the x-axis -/
def intersecting_parabolas (k : ℝ) : Prop :=
  ∃ x : ℝ, 
    (x^2 + k*x + 1 = 0) ∧ 
    (x^2 - x - k = 0) ∧
    (x^2 + k*x + 1 ≠ x^2 - x - k)

/-- The value of k for which the parabolas intersect on the x-axis -/
theorem parabolas_intersection_k : 
  ∃! k : ℝ, intersecting_parabolas k ∧ k = 2 :=
sorry

end parabolas_intersection_k_l1977_197702


namespace digit_difference_729_l1977_197728

def base_3_digits (n : ℕ) : ℕ := 
  Nat.log 3 n + 1

def base_8_digits (n : ℕ) : ℕ := 
  Nat.log 8 n + 1

theorem digit_difference_729 : 
  base_3_digits 729 - base_8_digits 729 = 4 := by
  sorry

end digit_difference_729_l1977_197728


namespace quadratic_solution_product_l1977_197786

theorem quadratic_solution_product (p q : ℝ) : 
  (3 * p^2 - 9 * p - 15 = 0) → 
  (3 * q^2 - 9 * q - 15 = 0) → 
  (3 * p - 5) * (6 * q - 10) = -130 := by
sorry

end quadratic_solution_product_l1977_197786


namespace seashells_given_to_sam_l1977_197753

theorem seashells_given_to_sam (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 70) 
  (h2 : remaining_seashells = 27) : 
  initial_seashells - remaining_seashells = 43 :=
by
  sorry

end seashells_given_to_sam_l1977_197753


namespace car_pedestrian_speed_ratio_l1977_197744

-- Define the bridge length
variable (L : ℝ)

-- Define the speeds of the pedestrian and the car
variable (v_p v_c : ℝ)

-- Assume positive speeds and bridge length
variable (h_pos_L : L > 0)
variable (h_pos_v_p : v_p > 0)
variable (h_pos_v_c : v_c > 0)

-- Define the theorem
theorem car_pedestrian_speed_ratio
  (h1 : v_c * (L / (5 * v_p)) = L) -- Car covers full bridge in time pedestrian covers 1/5
  (h2 : v_p * (L / (5 * v_p)) = L / 5) -- Pedestrian covers 1/5 bridge in same time
  : v_c / v_p = 5 :=
by sorry

end car_pedestrian_speed_ratio_l1977_197744


namespace ship_passengers_l1977_197785

theorem ship_passengers : ∀ (P : ℕ),
  (P / 12 : ℚ) + (P / 8 : ℚ) + (P / 3 : ℚ) + (P / 6 : ℚ) + 35 = P →
  P = 120 := by
  sorry

end ship_passengers_l1977_197785


namespace sin_squared_sum_less_than_one_l1977_197766

theorem sin_squared_sum_less_than_one (x y z : ℝ) 
  (h1 : Real.tan x + Real.tan y + Real.tan z = 2)
  (h2 : 0 < x ∧ x < Real.pi / 2)
  (h3 : 0 < y ∧ y < Real.pi / 2)
  (h4 : 0 < z ∧ z < Real.pi / 2) :
  Real.sin x ^ 2 + Real.sin y ^ 2 + Real.sin z ^ 2 < 1 := by
  sorry

end sin_squared_sum_less_than_one_l1977_197766


namespace probability_theorem_l1977_197793

def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def green_marbles : ℕ := 5
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles
def marbles_selected : ℕ := 4

def probability_one_red_two_blue_one_green : ℚ :=
  (red_marbles.choose 1 * blue_marbles.choose 2 * green_marbles.choose 1) /
  (total_marbles.choose marbles_selected)

theorem probability_theorem :
  probability_one_red_two_blue_one_green = 411 / 4200 :=
by sorry

end probability_theorem_l1977_197793


namespace common_tangent_line_sum_of_coefficients_l1977_197772

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = x^2 + 121/100

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = y^2 + 49/4

/-- The common tangent line L -/
def L (x y : ℝ) : Prop := x + 25*y = 12

/-- The theorem stating that L is a common tangent to P₁ and P₂, 
    and 1, 25, 12 are the smallest positive integers satisfying the equation -/
theorem common_tangent_line : 
  (∀ x y : ℝ, P₁ x y → L x y → (∃ u : ℝ, ∀ v : ℝ, P₁ u v → L u v → (u, v) = (x, y))) ∧ 
  (∀ x y : ℝ, P₂ x y → L x y → (∃ u : ℝ, ∀ v : ℝ, P₂ u v → L u v → (u, v) = (x, y))) ∧
  (∀ a b c : ℕ+, (∀ x y : ℝ, a*x + b*y = c ↔ L x y) → a ≥ 1 ∧ b ≥ 25 ∧ c ≥ 12) :=
sorry

/-- The sum of the coefficients -/
def coefficient_sum : ℕ := 38

/-- Theorem stating that the sum of coefficients is 38 -/
theorem sum_of_coefficients : 
  ∀ a b c : ℕ+, (∀ x y : ℝ, a*x + b*y = c ↔ L x y) → (a : ℕ) + (b : ℕ) + (c : ℕ) = coefficient_sum :=
sorry

end common_tangent_line_sum_of_coefficients_l1977_197772


namespace joan_total_seashells_l1977_197717

/-- Given that Joan found 79 seashells, received 63 from Mike, and 97 from Alicia,
    prove that the total number of seashells Joan has is 239. -/
theorem joan_total_seashells 
  (joan_found : ℕ) 
  (mike_gave : ℕ) 
  (alicia_gave : ℕ) 
  (h1 : joan_found = 79) 
  (h2 : mike_gave = 63) 
  (h3 : alicia_gave = 97) : 
  joan_found + mike_gave + alicia_gave = 239 := by
  sorry

end joan_total_seashells_l1977_197717


namespace polar_to_cartesian_circle_l1977_197774

/-- The polar equation ρ = 5 sin θ represents a circle in Cartesian coordinates. -/
theorem polar_to_cartesian_circle :
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), ρ = 5 * Real.sin θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end polar_to_cartesian_circle_l1977_197774


namespace total_writing_instruments_l1977_197750

theorem total_writing_instruments (pens pencils markers : ℕ) : 
  (5 * pens = 6 * pencils - 54) →  -- Ratio of pens to pencils is 5:6, and 9 more pencils
  (4 * pencils = 3 * markers) →    -- Ratio of markers to pencils is 4:3
  pens + pencils + markers = 171   -- Total number of writing instruments
  := by sorry

end total_writing_instruments_l1977_197750


namespace boys_share_calculation_l1977_197743

/-- Proves that in a family with a given boy-to-girl ratio and total children, 
    if a certain amount is shared among the boys, each boy receives the calculated amount. -/
theorem boys_share_calculation 
  (total_children : ℕ) 
  (boy_ratio girl_ratio : ℕ) 
  (total_money : ℕ) 
  (h1 : total_children = 180) 
  (h2 : boy_ratio = 5) 
  (h3 : girl_ratio = 7) 
  (h4 : total_money = 3900) :
  total_money / (total_children * boy_ratio / (boy_ratio + girl_ratio)) = 52 := by
sorry


end boys_share_calculation_l1977_197743


namespace problem_solution_l1977_197714

theorem problem_solution : 
  let X := (354 * 28)^2
  let Y := (48 * 14)^2
  (X * 9) / (Y * 2) = 2255688 := by
sorry

end problem_solution_l1977_197714


namespace square_geq_bound_l1977_197789

theorem square_geq_bound (a : ℝ) : (∀ x > 1, x^2 ≥ a) → a ≤ 1 := by
  sorry

end square_geq_bound_l1977_197789


namespace consecutive_numbers_probability_l1977_197742

def choose (n k : ℕ) : ℕ := Nat.choose n k

def p : ℚ :=
  1 - (choose 40 6 + choose 5 1 * choose 39 5 + choose 4 2 * choose 38 4 + choose 37 3) / choose 45 6

theorem consecutive_numbers_probability : 
  ⌊1000 * p⌋ = 56 := by sorry

end consecutive_numbers_probability_l1977_197742


namespace right_triangle_area_l1977_197733

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_angle : a = b) (h_side : a = 5) : (1/2) * a * b = 12.5 := by
  sorry

end right_triangle_area_l1977_197733


namespace rabbit_groupings_count_l1977_197792

/-- The number of ways to divide 12 rabbits into specific groups -/
def rabbit_groupings : ℕ :=
  let total_rabbits : ℕ := 12
  let group1_size : ℕ := 4
  let group2_size : ℕ := 6
  let group3_size : ℕ := 2
  let remaining_rabbits : ℕ := total_rabbits - 2  -- BunBun and Thumper are already placed
  Nat.choose remaining_rabbits (group1_size - 1) * Nat.choose (remaining_rabbits - (group1_size - 1)) (group2_size - 1)

/-- Theorem stating the number of ways to divide the rabbits -/
theorem rabbit_groupings_count : rabbit_groupings = 2520 := by
  sorry

end rabbit_groupings_count_l1977_197792


namespace problem_statement_l1977_197783

-- Define proposition p
def p : Prop := ∀ x : ℝ, (|x| = x ↔ x > 0)

-- Define proposition q
def q : Prop := (¬∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)

-- Theorem to prove
theorem problem_statement : ¬(p ∧ q) := by
  sorry

end problem_statement_l1977_197783


namespace rectangular_solid_surface_area_l1977_197755

-- Define a rectangular solid with prime edge lengths
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ
  length_prime : Nat.Prime length
  width_prime : Nat.Prime width
  height_prime : Nat.Prime height
  different_edges : length ≠ width ∧ width ≠ height ∧ length ≠ height

-- Define the volume of the rectangular solid
def volume (r : RectangularSolid) : ℕ := r.length * r.width * r.height

-- Define the surface area of the rectangular solid
def surfaceArea (r : RectangularSolid) : ℕ :=
  2 * (r.length * r.width + r.width * r.height + r.length * r.height)

-- Theorem statement
theorem rectangular_solid_surface_area :
  ∀ r : RectangularSolid, volume r = 770 → surfaceArea r = 1098 := by
  sorry

end rectangular_solid_surface_area_l1977_197755


namespace intersection_implies_k_value_l1977_197760

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The x-coordinate of the intersection point -/
def x_intersect : ℝ := 2

/-- The y-coordinate of the intersection point -/
def y_intersect : ℝ := 13

/-- Line p with equation y = 5x + 3 -/
def p : Line := { slope := 5, intercept := 3 }

/-- Line q with equation y = kx + 7, where k is to be determined -/
def q (k : ℝ) : Line := { slope := k, intercept := 7 }

/-- Theorem stating that if lines p and q intersect at (2, 13), then k = 3 -/
theorem intersection_implies_k_value :
  y_intersect = p.slope * x_intersect + p.intercept ∧
  y_intersect = (q k).slope * x_intersect + (q k).intercept →
  k = 3 := by sorry

end intersection_implies_k_value_l1977_197760


namespace complex_product_modulus_l1977_197713

theorem complex_product_modulus : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end complex_product_modulus_l1977_197713


namespace unique_H_value_l1977_197784

/-- Represents a digit in the addition problem -/
structure Digit :=
  (value : Nat)
  (is_valid : value < 10)

/-- Represents the addition problem -/
structure AdditionProblem :=
  (T : Digit)
  (H : Digit)
  (R : Digit)
  (E : Digit)
  (F : Digit)
  (I : Digit)
  (V : Digit)
  (S : Digit)
  (all_different : T ≠ H ∧ T ≠ R ∧ T ≠ E ∧ T ≠ F ∧ T ≠ I ∧ T ≠ V ∧ T ≠ S ∧
                   H ≠ R ∧ H ≠ E ∧ H ≠ F ∧ H ≠ I ∧ H ≠ V ∧ H ≠ S ∧
                   R ≠ E ∧ R ≠ F ∧ R ≠ I ∧ R ≠ V ∧ R ≠ S ∧
                   E ≠ F ∧ E ≠ I ∧ E ≠ V ∧ E ≠ S ∧
                   F ≠ I ∧ F ≠ V ∧ F ≠ S ∧
                   I ≠ V ∧ I ≠ S ∧
                   V ≠ S)
  (T_is_eight : T.value = 8)
  (E_is_odd : E.value % 2 = 1)
  (addition_valid : F.value * 10000 + I.value * 1000 + V.value * 100 + E.value * 10 + S.value =
                    (T.value * 1000 + H.value * 100 + R.value * 10 + E.value) * 2)

theorem unique_H_value (p : AdditionProblem) : p.H.value = 7 :=
  sorry

end unique_H_value_l1977_197784


namespace volunteers_arrangement_count_l1977_197782

/-- The number of ways to arrange volunteers for tasks. -/
def arrangeVolunteers (volunteers : ℕ) (tasks : ℕ) : ℕ :=
  (tasks - 1).choose (volunteers - 1) * volunteers.factorial

/-- Theorem stating the number of arrangements for 4 volunteers and 5 tasks. -/
theorem volunteers_arrangement_count :
  arrangeVolunteers 4 5 = 240 := by
  sorry

end volunteers_arrangement_count_l1977_197782


namespace words_with_consonants_l1977_197730

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 6

/-- The number of consonants in the alphabet -/
def consonant_count : ℕ := 4

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 2

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The total number of possible words -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words containing only vowels -/
def vowel_only_words : ℕ := vowel_count ^ word_length

theorem words_with_consonants :
  total_words - vowel_only_words = 7744 :=
sorry

end words_with_consonants_l1977_197730


namespace polynomial_division_result_l1977_197746

theorem polynomial_division_result :
  let f : Polynomial ℝ := 4 * X^4 + 12 * X^3 - 9 * X^2 + X + 3
  let d : Polynomial ℝ := X^2 + 3 * X - 2
  ∀ q r : Polynomial ℝ,
    f = q * d + r →
    (r.degree < d.degree) →
    q.eval 1 + r.eval (-1) = 0 := by
sorry

end polynomial_division_result_l1977_197746


namespace jumping_contest_l1977_197739

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_jump = 58) :
  frog_jump - grasshopper_jump = 39 := by
  sorry

end jumping_contest_l1977_197739


namespace compound_weight_l1977_197735

/-- Given a compound with a molecular weight of 1098 and 9 moles of this compound,
    prove that the total weight is 9882 grams. -/
theorem compound_weight (molecular_weight : ℕ) (moles : ℕ) : 
  molecular_weight = 1098 → moles = 9 → molecular_weight * moles = 9882 := by
  sorry

end compound_weight_l1977_197735


namespace files_deleted_l1977_197708

theorem files_deleted (initial_files remaining_files : ℕ) (h1 : initial_files = 25) (h2 : remaining_files = 2) :
  initial_files - remaining_files = 23 := by
  sorry

end files_deleted_l1977_197708


namespace correct_water_ratio_l1977_197799

/-- Represents the time in minutes to fill the bathtub with hot water -/
def hot_water_fill_time : ℝ := 23

/-- Represents the time in minutes to fill the bathtub with cold water -/
def cold_water_fill_time : ℝ := 17

/-- Represents the ratio of hot water to cold water when the bathtub is full -/
def hot_to_cold_ratio : ℝ := 1.5

/-- Represents the delay in minutes before opening the cold water tap -/
def cold_water_delay : ℝ := 7

/-- Proves that opening the cold water tap after the specified delay results in the correct ratio of hot to cold water -/
theorem correct_water_ratio : 
  let hot_water_volume := (hot_water_fill_time - cold_water_delay) / hot_water_fill_time
  let cold_water_volume := cold_water_delay / cold_water_fill_time
  hot_water_volume = hot_to_cold_ratio * cold_water_volume := by
  sorry

end correct_water_ratio_l1977_197799


namespace turtle_initial_coins_l1977_197798

def bridge_crossing (initial_coins : ℕ) : Prop :=
  let after_first_crossing := 3 * initial_coins - 30
  let after_second_crossing := 3 * after_first_crossing - 30
  after_second_crossing = 0

theorem turtle_initial_coins : 
  ∃ (x : ℕ), bridge_crossing x ∧ x = 15 :=
sorry

end turtle_initial_coins_l1977_197798


namespace correct_calculation_l1977_197761

theorem correct_calculation (x : ℝ) (h : x / 15 = 6) : 15 * x = 1350 := by
  sorry

end correct_calculation_l1977_197761


namespace employed_females_percentage_l1977_197705

theorem employed_females_percentage
  (total_population : ℕ)
  (employed_percentage : ℚ)
  (employed_males_percentage : ℚ)
  (h1 : employed_percentage = 60 / 100)
  (h2 : employed_males_percentage = 48 / 100)
  : (employed_percentage - employed_males_percentage) / employed_percentage = 20 / 100 := by
  sorry

end employed_females_percentage_l1977_197705


namespace comparison_theorem_l1977_197794

theorem comparison_theorem :
  (3 * 10^5 < 2 * 10^6) ∧ (-2 - 1/3 > -3 - 1/2) := by
  sorry

end comparison_theorem_l1977_197794


namespace binomial_15_12_l1977_197757

theorem binomial_15_12 : Nat.choose 15 12 = 455 := by
  sorry

end binomial_15_12_l1977_197757


namespace hyperbola_vertex_distance_l1977_197725

/-- Given a hyperbola with equation x²/48 - y²/16 = 1, 
    the distance between its vertices is 8√3. -/
theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), 
  x^2 / 48 - y^2 / 16 = 1 →
  ∃ (d : ℝ), d = 8 * Real.sqrt 3 ∧ d = 2 * Real.sqrt 48 :=
by sorry

end hyperbola_vertex_distance_l1977_197725


namespace cricket_bat_profit_l1977_197752

/-- Calculates the profit amount for a cricket bat sale -/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 900 →
  profit_percentage = 33.33 →
  ∃ (cost_price : ℝ), 
    cost_price > 0 ∧
    selling_price = cost_price * (1 + profit_percentage / 100) ∧
    selling_price - cost_price = 225 := by
  sorry

end cricket_bat_profit_l1977_197752


namespace triangle_area_from_perimeter_and_inradius_l1977_197759

/-- The area of a triangle given its perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h_perimeter : perimeter = 20) 
  (h_inradius : inradius = 2.5) : 
  area = perimeter / 2 * inradius ∧ area = 25 := by
  sorry

end triangle_area_from_perimeter_and_inradius_l1977_197759


namespace math_class_size_l1977_197762

theorem math_class_size :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 8 = 5 ∧ n % 6 = 1 ∧ n = 13 := by
  sorry

end math_class_size_l1977_197762


namespace two_numbers_sum_difference_product_l1977_197775

theorem two_numbers_sum_difference_product 
  (x y : ℝ) 
  (sum_eq : x + y = 40) 
  (diff_eq : x - y = 16) : 
  x = 28 ∧ y = 12 ∧ x * y = 336 := by
sorry

end two_numbers_sum_difference_product_l1977_197775


namespace cone_prism_volume_ratio_l1977_197771

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism
    to the volume of the prism is π/16. -/
theorem cone_prism_volume_ratio :
  ∀ (cone_volume prism_volume : ℝ) (prism_base_length prism_base_width prism_height : ℝ),
  prism_base_length = 3 →
  prism_base_width = 4 →
  prism_height = 5 →
  prism_volume = prism_base_length * prism_base_width * prism_height →
  cone_volume = (1/3) * π * (prism_base_length/2)^2 * prism_height →
  cone_volume / prism_volume = π/16 := by
sorry

end cone_prism_volume_ratio_l1977_197771


namespace complex_modulus_problem_l1977_197745

theorem complex_modulus_problem (x y : ℝ) (h : (Complex.I : ℂ) / (1 + Complex.I) = x + y * Complex.I) : 
  Complex.abs (x - y * Complex.I) = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_problem_l1977_197745


namespace point_line_plane_relation_l1977_197716

-- Define the types for point, line, and plane
variable (Point Line Plane : Type)

-- Define the relations
variable (lies_on : Point → Line → Prop)
variable (is_in : Line → Plane → Prop)

-- Define the set membership and subset relations
variable (mem : Point → Line → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem point_line_plane_relation 
  (P : Point) (m : Line) (α : Plane) 
  (h1 : lies_on P m) 
  (h2 : is_in m α) : 
  mem P m ∧ subset m α :=
sorry

end point_line_plane_relation_l1977_197716


namespace octagon_area_l1977_197719

/-- Given a square with area 16 and a regular octagon with equal perimeter to the square,
    the area of the octagon is 8(1+√2) -/
theorem octagon_area (s : ℝ) (t : ℝ) : 
  s^2 = 16 →                        -- Square area is 16
  4*s = 8*t →                       -- Equal perimeters
  2*(1+Real.sqrt 2)*t^2 = 8*(1+Real.sqrt 2) := by
sorry

end octagon_area_l1977_197719


namespace group_frequency_l1977_197722

theorem group_frequency (sample_capacity : ℕ) (group_frequency_ratio : ℚ) : 
  sample_capacity = 20 →
  group_frequency_ratio = 1/4 →
  (sample_capacity : ℚ) * group_frequency_ratio = 5 := by
sorry

end group_frequency_l1977_197722


namespace negation_of_universal_proposition_l1977_197741

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end negation_of_universal_proposition_l1977_197741


namespace fruits_given_to_jane_l1977_197703

/- Define the initial number of each type of fruit -/
def plums : ℕ := 25
def guavas : ℕ := 30
def apples : ℕ := 36
def oranges : ℕ := 20
def bananas : ℕ := 15

/- Define the total number of fruits Jacqueline had initially -/
def initial_fruits : ℕ := plums + guavas + apples + oranges + bananas

/- Define the number of fruits Jacqueline had left -/
def fruits_left : ℕ := 38

/- Theorem: The number of fruits Jacqueline gave Jane is equal to 
   the difference between her initial fruits and the fruits left -/
theorem fruits_given_to_jane : 
  initial_fruits - fruits_left = 88 := by sorry

end fruits_given_to_jane_l1977_197703


namespace intersection_of_A_and_B_intersection_of_A_l1977_197769

-- Part I
def A : Set (ℝ × ℝ) := {p | p.2 = p.1^2 + 2}
def B : Set (ℝ × ℝ) := {p | p.2 = 6 - p.1^2}

theorem intersection_of_A_and_B : A ∩ B = {(Real.sqrt 2, 4), (-Real.sqrt 2, 4)} := by sorry

-- Part II
def A' : Set ℝ := {y | ∃ x, y = x^2 + 2}
def B' : Set ℝ := {y | ∃ x, y = 6 - x^2}

theorem intersection_of_A'_and_B' : A' ∩ B' = {y | 2 ≤ y ∧ y ≤ 6} := by sorry

end intersection_of_A_and_B_intersection_of_A_l1977_197769


namespace moving_points_minimum_distance_l1977_197709

/-- Two points moving along perpendicular lines towards their intersection --/
theorem moving_points_minimum_distance 
  (a b v v₁ : ℝ) (ha : a > 0) (hb : b > 0) (hv : v > 0) (hv₁ : v₁ > 0) :
  let min_distance := |b * v - a * v₁| / Real.sqrt (v^2 + v₁^2)
  let vertex_distance_diff := |a * v^2 + a * b * v * v₁| / (v^2 + v₁^2)
  let equal_speed_min_distance := |a - b| / Real.sqrt 2
  let equal_speed_time := (a + b) / (2 * v)
  let equal_speed_distance_a := (a - b) / 2
  let equal_speed_distance_b := (b - a) / 2
  ∃ (t : ℝ), 
    (∀ (s : ℝ), 
      Real.sqrt ((a - v * s)^2 + (b - v₁ * s)^2) ≥ min_distance) ∧
    (Real.sqrt ((a - v * t)^2 + (b - v₁ * t)^2) = min_distance) ∧
    (|(a - v * t) - (b - v₁ * t)| = vertex_distance_diff) ∧
    (v = v₁ → 
      min_distance = equal_speed_min_distance ∧
      t = equal_speed_time ∧
      a - v * t = equal_speed_distance_a ∧
      b - v₁ * t = equal_speed_distance_b) := by sorry

end moving_points_minimum_distance_l1977_197709


namespace jack_christina_lindy_meeting_l1977_197718

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting 
  (initial_distance : ℝ) 
  (christina_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_total_distance : ℝ) 
  (h1 : initial_distance = 240)
  (h2 : christina_speed = 3)
  (h3 : lindy_speed = 9)
  (h4 : lindy_total_distance = 270) :
  ∃ (jack_speed : ℝ), 
    jack_speed = 5 ∧ 
    (lindy_total_distance / lindy_speed) * jack_speed + 
    (lindy_total_distance / lindy_speed) * christina_speed = 
    initial_distance := by
  sorry


end jack_christina_lindy_meeting_l1977_197718


namespace min_positive_period_sin_cos_squared_l1977_197773

theorem min_positive_period_sin_cos_squared (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (Real.sin x + Real.cos x)^2 + 1
  ∃ T : ℝ, T > 0 ∧ (∀ t : ℝ, f (t + T) = f t) ∧
    (∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (t + S) = f t) → T ≤ S) ∧
    T = π :=
by sorry

end min_positive_period_sin_cos_squared_l1977_197773


namespace log_approximation_l1977_197723

-- Define the base of the logarithm
def base : ℝ := 8

-- Define the given logarithmic value
def log_value : ℝ := 2.75

-- Define the approximate result
def approx_result : ℝ := 215

-- Define a tolerance for the approximation
def tolerance : ℝ := 0.1

-- Theorem statement
theorem log_approximation (y : ℝ) (h : Real.log y / Real.log base = log_value) :
  |y - approx_result| < tolerance :=
sorry

end log_approximation_l1977_197723


namespace mountain_paths_theorem_l1977_197706

/-- Number of paths to the mountain top -/
def num_paths : ℕ := 5

/-- Number of people ascending and descending -/
def num_people : ℕ := 2

/-- Calculates the number of ways to ascend and descend the mountain for scenario a -/
def scenario_a (n p : ℕ) : ℕ := 
  Nat.choose n p * Nat.choose (n - p) p

/-- Calculates the number of ways to ascend and descend the mountain for scenario b -/
def scenario_b (n p : ℕ) : ℕ := 
  Nat.choose n p * Nat.choose n p

/-- Calculates the number of ways to ascend and descend the mountain for scenario c -/
def scenario_c (n p : ℕ) : ℕ := 
  (n ^ p) * (n ^ p)

/-- Calculates the number of ways to ascend and descend the mountain for scenario d -/
def scenario_d (n p : ℕ) : ℕ := 
  (Nat.factorial n / Nat.factorial (n - p)) * (Nat.factorial (n - p) / Nat.factorial (n - 2*p))

/-- Calculates the number of ways to ascend and descend the mountain for scenario e -/
def scenario_e (n p : ℕ) : ℕ := 
  (Nat.factorial n / Nat.factorial (n - p)) * (Nat.factorial n / Nat.factorial (n - p))

/-- Calculates the number of ways to ascend and descend the mountain for scenario f -/
def scenario_f (n p : ℕ) : ℕ := 
  (n ^ p) * (n ^ p)

theorem mountain_paths_theorem :
  scenario_a num_paths num_people = 30 ∧
  scenario_b num_paths num_people = 100 ∧
  scenario_c num_paths num_people = 625 ∧
  scenario_d num_paths num_people = 120 ∧
  scenario_e num_paths num_people = 400 ∧
  scenario_f num_paths num_people = 625 :=
by sorry

end mountain_paths_theorem_l1977_197706


namespace average_speed_two_hours_l1977_197724

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 20 → speed2 = 30 → (speed1 + speed2) / 2 = 25 := by
  sorry

end average_speed_two_hours_l1977_197724


namespace complex_sum_argument_l1977_197748

/-- The argument of the sum of five complex exponentials -/
theorem complex_sum_argument :
  let z₁ := Complex.exp (11 * π * Complex.I / 120)
  let z₂ := Complex.exp (31 * π * Complex.I / 120)
  let z₃ := Complex.exp (51 * π * Complex.I / 120)
  let z₄ := Complex.exp (71 * π * Complex.I / 120)
  let z₅ := Complex.exp (91 * π * Complex.I / 120)
  Complex.arg (z₁ + z₂ + z₃ + z₄ + z₅) = 17 * π / 40 :=
by sorry

end complex_sum_argument_l1977_197748


namespace largest_integer_satisfying_inequality_l1977_197737

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (5 * x - 4 < 3 - 2 * x) → x ≤ 0 :=
by
  sorry

end largest_integer_satisfying_inequality_l1977_197737


namespace fruit_arrangement_count_l1977_197732

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem fruit_arrangement_count : 
  let total_fruits : ℕ := 8
  let apples : ℕ := 3
  let oranges : ℕ := 2
  let bananas : ℕ := 3
  factorial total_fruits / (factorial apples * factorial oranges * factorial bananas) = 560 := by
  sorry

end fruit_arrangement_count_l1977_197732


namespace max_pies_without_ingredients_l1977_197731

/-- Represents the number of pies with specific ingredients -/
structure PieCount where
  total : ℕ
  chocolate : ℕ
  marshmallow : ℕ
  cayenne : ℕ
  salted_soy_nut : ℕ

/-- Conditions for the pie problem -/
def pie_conditions (p : PieCount) : Prop :=
  p.total = 48 ∧
  p.chocolate = (5 * p.total) / 8 ∧
  p.marshmallow = (3 * p.total) / 4 ∧
  p.cayenne = (2 * p.total) / 3 ∧
  p.salted_soy_nut = p.total / 4 ∧
  p.salted_soy_nut ≤ p.marshmallow

/-- The theorem stating the maximum number of pies without any of the mentioned ingredients -/
theorem max_pies_without_ingredients (p : PieCount) 
  (h : pie_conditions p) : 
  p.total - max p.chocolate (max p.marshmallow (max p.cayenne p.salted_soy_nut)) ≤ 16 :=
sorry

end max_pies_without_ingredients_l1977_197731


namespace fraction_equality_l1977_197738

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z) / (1/w - 1/z) = 2014) : 
  (w + z) / (w - z) = -2014 := by sorry

end fraction_equality_l1977_197738


namespace sequence_inequality_l1977_197720

theorem sequence_inequality (a : ℕ+ → ℝ) 
  (h : ∀ (k m : ℕ+), |a (k + m) - a k - a m| ≤ 1) :
  ∀ (k m : ℕ+), |a k / k.val - a m / m.val| < 1 / k.val + 1 / m.val := by
  sorry

end sequence_inequality_l1977_197720


namespace polygon_diagonals_l1977_197767

theorem polygon_diagonals (n : ℕ) (h : n ≥ 3) :
  (n * (n - 1)) / 2 - n = 20 → n = 8 := by
sorry

end polygon_diagonals_l1977_197767


namespace complement_of_P_union_Q_is_M_l1977_197780

def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}
def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

theorem complement_of_P_union_Q_is_M : (P ∪ Q)ᶜ = M := by sorry

end complement_of_P_union_Q_is_M_l1977_197780


namespace intercept_plane_equation_point_on_intercept_plane_l1977_197734

/-- A plane in 3D space with intercepts a, b, c on x, y, z axes respectively --/
structure InterceptPlane where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- The equation of a plane given its intercepts --/
def plane_equation (p : InterceptPlane) (x y z : ℝ) : Prop :=
  x / p.a + y / p.b + z / p.c = 1

/-- Theorem stating that the given equation represents the plane with given intercepts --/
theorem intercept_plane_equation (p : InterceptPlane) :
  ∀ x y z : ℝ, (x = p.a ∧ y = 0 ∧ z = 0) ∨ (x = 0 ∧ y = p.b ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = p.c) →
  plane_equation p x y z := by
  sorry

/-- Theorem stating that any point satisfying the equation lies on the plane --/
theorem point_on_intercept_plane (p : InterceptPlane) :
  ∀ x y z : ℝ, plane_equation p x y z →
  ∃ t u v : ℝ, t + u + v = 1 ∧ x = t * p.a ∧ y = u * p.b ∧ z = v * p.c := by
  sorry

end intercept_plane_equation_point_on_intercept_plane_l1977_197734


namespace girls_in_classroom_l1977_197749

theorem girls_in_classroom (total_students : ℕ) (girls_ratio boys_ratio : ℕ) : 
  total_students = 28 → 
  girls_ratio = 3 → 
  boys_ratio = 4 → 
  (girls_ratio + boys_ratio) * (total_students / (girls_ratio + boys_ratio)) = girls_ratio * (total_students / (girls_ratio + boys_ratio)) + boys_ratio * (total_students / (girls_ratio + boys_ratio)) →
  girls_ratio * (total_students / (girls_ratio + boys_ratio)) = 12 := by
sorry

end girls_in_classroom_l1977_197749


namespace remainder_problem_l1977_197791

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 35 = 28 := by
  sorry

end remainder_problem_l1977_197791


namespace compound_weight_is_88_l1977_197704

/-- The molecular weight of the carbon part in the compound C4H8O2 -/
def carbon_weight : ℕ := 48

/-- The molecular weight of the hydrogen part in the compound C4H8O2 -/
def hydrogen_weight : ℕ := 8

/-- The molecular weight of the oxygen part in the compound C4H8O2 -/
def oxygen_weight : ℕ := 32

/-- The total molecular weight of the compound C4H8O2 -/
def total_molecular_weight : ℕ := carbon_weight + hydrogen_weight + oxygen_weight

theorem compound_weight_is_88 : total_molecular_weight = 88 := by
  sorry

end compound_weight_is_88_l1977_197704


namespace largest_number_problem_l1977_197795

theorem largest_number_problem (a b c : ℝ) : 
  a < b ∧ b < c →
  a + b + c = 72 →
  c - b = 5 →
  b - a = 8 →
  c = 30 := by
sorry

end largest_number_problem_l1977_197795


namespace painted_cube_probability_l1977_197770

/-- Represents a 5x5x5 cube with three adjacent faces painted -/
structure PaintedCube :=
  (size : Nat)
  (painted_faces : Nat)

/-- Calculates the number of unit cubes with exactly three painted faces -/
def three_painted_faces (cube : PaintedCube) : Nat :=
  8  -- 8 vertices of the cube

/-- Calculates the number of unit cubes with exactly one painted face -/
def one_painted_face (cube : PaintedCube) : Nat :=
  27  -- 9 cubes per face * 3 painted faces

/-- Calculates the total number of ways to choose two unit cubes -/
def total_choices (cube : PaintedCube) : Nat :=
  (cube.size ^ 3) * (cube.size ^ 3 - 1) / 2

/-- Theorem: The probability of selecting one unit cube with exactly three painted faces
    and another unit cube with exactly one painted face from a 5x5x5 cube with
    three adjacent faces painted is 24/775 -/
theorem painted_cube_probability (cube : PaintedCube)
  (h1 : cube.size = 5)
  (h2 : cube.painted_faces = 3) :
  (three_painted_faces cube * one_painted_face cube : ℚ) / total_choices cube = 24 / 775 := by
  sorry

end painted_cube_probability_l1977_197770


namespace amanda_kitchen_upgrade_cost_l1977_197776

/-- The total cost of Amanda's kitchen upgrade --/
def kitchen_upgrade_cost (cabinet_knobs : ℕ) (knob_price : ℚ) (drawer_pulls : ℕ) (pull_price : ℚ) : ℚ :=
  (cabinet_knobs : ℚ) * knob_price + (drawer_pulls : ℚ) * pull_price

/-- Proof that Amanda's kitchen upgrade costs $77.00 --/
theorem amanda_kitchen_upgrade_cost :
  kitchen_upgrade_cost 18 (5/2) 8 4 = 77 := by
  sorry

end amanda_kitchen_upgrade_cost_l1977_197776


namespace coin_value_equality_l1977_197754

theorem coin_value_equality (n : ℕ) : 
  (20 * 25 + 10 * 10 = 10 * 25 + n * 10) → n = 35 := by
  sorry

end coin_value_equality_l1977_197754
