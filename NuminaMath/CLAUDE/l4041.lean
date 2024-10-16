import Mathlib

namespace NUMINAMATH_CALUDE_slope_determines_m_l4041_404162

theorem slope_determines_m (m : ℝ) : 
  let A : ℝ × ℝ := (-m, 6)
  let B : ℝ × ℝ := (1, 3*m)
  (B.2 - A.2) / (B.1 - A.1) = 12 → m = -2 := by
sorry

end NUMINAMATH_CALUDE_slope_determines_m_l4041_404162


namespace NUMINAMATH_CALUDE_parallel_condition_l4041_404130

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a * y + 3 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y + 6 = 0

-- Define the parallel relation between two lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- State the theorem
theorem parallel_condition (a : ℝ) :
  (parallel (l₁ a) (l₂ a) → (a = 2 ∨ a = -2)) ∧
  ¬(a = 2 ∨ a = -2 → parallel (l₁ a) (l₂ a)) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l4041_404130


namespace NUMINAMATH_CALUDE_tower_arrangements_l4041_404160

def num_red : ℕ := 2
def num_blue : ℕ := 3
def num_green : ℕ := 4
def tower_height : ℕ := 8

theorem tower_arrangements : 
  (Nat.choose (num_red + num_blue + num_green) tower_height) * 
  (Nat.factorial tower_height / (Nat.factorial num_red * Nat.factorial num_blue * Nat.factorial (tower_height - num_red - num_blue))) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_tower_arrangements_l4041_404160


namespace NUMINAMATH_CALUDE_power_of_product_l4041_404142

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l4041_404142


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l4041_404180

/-- Given point A(1, 0) and line l: y = 2x - 4, with point R on line l such that
    vector RA equals vector AP, prove that the trajectory of point P is y = 2x -/
theorem trajectory_of_point_P (R P : ℝ × ℝ) :
  (∃ a : ℝ, R = (a, 2 * a - 4)) →  -- R is on line l: y = 2x - 4
  (R.1 - 1, R.2) = (P.1 - 1, P.2) →  -- vector RA = vector AP
  P.2 = 2 * P.1 :=  -- trajectory of P is y = 2x
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l4041_404180


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_a_binomial_coefficient_identity_b_binomial_coefficient_identity_c_binomial_coefficient_identity_d_binomial_coefficient_identity_e_l4041_404165

-- Part (a)
theorem binomial_coefficient_identity_a (r m k : ℕ) (h1 : k ≤ m) (h2 : m ≤ r) :
  (r.choose m) * (m.choose k) = (r.choose k) * ((r - k).choose (m - k)) := by
  sorry

-- Part (b)
theorem binomial_coefficient_identity_b (n m : ℕ) :
  (n + 1).choose (m + 1) = n.choose m + n.choose (m + 1) := by
  sorry

-- Part (c)
theorem binomial_coefficient_identity_c (n : ℕ) :
  (2 * n).choose n = (Finset.range (n + 1)).sum (λ k => (n.choose k) ^ 2) := by
  sorry

-- Part (d)
theorem binomial_coefficient_identity_d (m n k : ℕ) (h : k ≤ n) :
  (m + n).choose k = (Finset.range (k + 1)).sum (λ p => (n.choose p) * (m.choose (k - p))) := by
  sorry

-- Part (e)
theorem binomial_coefficient_identity_e (n k : ℕ) (h : k ≤ n) :
  n.choose k = (Finset.range (n - k + 1)).sum (λ i => (k + i - 1).choose (k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_a_binomial_coefficient_identity_b_binomial_coefficient_identity_c_binomial_coefficient_identity_d_binomial_coefficient_identity_e_l4041_404165


namespace NUMINAMATH_CALUDE_blue_jellybean_probability_l4041_404102

/-- The probability of drawing 3 blue jellybeans in succession from a bag 
    containing 10 red and 10 blue jellybeans, without replacement. -/
theorem blue_jellybean_probability : 
  let total_jellybeans : ℕ := 20
  let blue_jellybeans : ℕ := 10
  let draws : ℕ := 3
  
  -- The probability is calculated as the product of individual probabilities
  (blue_jellybeans / total_jellybeans) * 
  ((blue_jellybeans - 1) / (total_jellybeans - 1)) * 
  ((blue_jellybeans - 2) / (total_jellybeans - 2)) = 2 / 19 := by
sorry


end NUMINAMATH_CALUDE_blue_jellybean_probability_l4041_404102


namespace NUMINAMATH_CALUDE_ellipse_theorem_l4041_404181

-- Define the ellipse
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition that a > b > 0
def size_condition (a b : ℝ) : Prop := a > b ∧ b > 0

-- Define the angle condition
def angle_condition (PF1F2_angle : ℝ) : Prop := Real.sin PF1F2_angle = 1/3

-- Main theorem
theorem ellipse_theorem (a b : ℝ) (h1 : size_condition a b) 
  (h2 : ∃ P F1 F2 : ℝ × ℝ, 
    ellipse a b (F2.1) (P.2) ∧ 
    angle_condition (Real.arcsin (1/3))) : 
  a = Real.sqrt 2 * b := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l4041_404181


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l4041_404100

/-- Given vectors a and b, if a is perpendicular to (t*a + b), then t = -1 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (t : ℝ) :
  a = (1, -1) →
  b = (6, -4) →
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) →
  t = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l4041_404100


namespace NUMINAMATH_CALUDE_carpet_area_is_27072_l4041_404156

/-- Calculates the area of carpet required for a room with a column -/
def carpet_area (room_length room_width column_side : ℕ) : ℕ :=
  let inches_per_foot := 12
  let room_length_inches := room_length * inches_per_foot
  let room_width_inches := room_width * inches_per_foot
  let column_side_inches := column_side * inches_per_foot
  let total_area := room_length_inches * room_width_inches
  let column_area := column_side_inches * column_side_inches
  total_area - column_area

/-- Theorem: The carpet area for the given room is 27,072 square inches -/
theorem carpet_area_is_27072 :
  carpet_area 16 12 2 = 27072 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_is_27072_l4041_404156


namespace NUMINAMATH_CALUDE_mowgli_nuts_theorem_l4041_404115

/-- The number of monkeys --/
def num_monkeys : ℕ := 5

/-- The number of nuts each monkey gathered initially --/
def nuts_per_monkey : ℕ := 8

/-- The number of nuts thrown by each monkey during the quarrel --/
def nuts_thrown_per_monkey : ℕ := num_monkeys - 1

/-- The total number of nuts thrown during the quarrel --/
def total_nuts_thrown : ℕ := num_monkeys * nuts_thrown_per_monkey

/-- The number of nuts Mowgli received --/
def nuts_received : ℕ := (num_monkeys * nuts_per_monkey) / 2

theorem mowgli_nuts_theorem :
  nuts_received = total_nuts_thrown :=
by sorry

end NUMINAMATH_CALUDE_mowgli_nuts_theorem_l4041_404115


namespace NUMINAMATH_CALUDE_mike_action_figures_l4041_404131

/-- The number of action figures each shelf can hold -/
def figures_per_shelf : ℕ := 8

/-- The number of shelves Mike needs -/
def number_of_shelves : ℕ := 8

/-- The total number of action figures Mike has -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

theorem mike_action_figures :
  total_figures = 64 :=
by sorry

end NUMINAMATH_CALUDE_mike_action_figures_l4041_404131


namespace NUMINAMATH_CALUDE_soda_price_calculation_l4041_404148

def pizza_price : ℚ := 12
def fries_price : ℚ := (3 / 10)
def goal_amount : ℚ := 500
def pizzas_sold : ℕ := 15
def fries_sold : ℕ := 40
def sodas_sold : ℕ := 25
def remaining_amount : ℚ := 258

theorem soda_price_calculation :
  ∃ (soda_price : ℚ),
    soda_price * sodas_sold = 
      goal_amount - remaining_amount - 
      (pizza_price * pizzas_sold + fries_price * fries_sold) ∧
    soda_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_soda_price_calculation_l4041_404148


namespace NUMINAMATH_CALUDE_initial_speed_calculation_l4041_404153

/-- Proves that the initial speed of a person traveling a distance D in time T is 160/3 kmph -/
theorem initial_speed_calculation (D T : ℝ) (h1 : D > 0) (h2 : T > 0) : ∃ S : ℝ,
  (2 / 3 * D) / (1 / 3 * T) = S ∧
  (1 / 3 * D) / 40 = 2 / 3 * T ∧
  S = 160 / 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_calculation_l4041_404153


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l4041_404187

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l4041_404187


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l4041_404183

theorem sum_of_four_numbers : 1.84 + 5.23 + 2.41 + 8.64 = 18.12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l4041_404183


namespace NUMINAMATH_CALUDE_total_routes_is_seven_l4041_404113

/-- The number of routes from A to C -/
def total_routes (highways_AB : ℕ) (paths_BC : ℕ) (direct_waterway : ℕ) : ℕ :=
  highways_AB * paths_BC + direct_waterway

/-- Theorem: Given the specified number of routes, the total number of routes from A to C is 7 -/
theorem total_routes_is_seven :
  total_routes 2 3 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_routes_is_seven_l4041_404113


namespace NUMINAMATH_CALUDE_min_buttons_for_adjacency_l4041_404106

/-- Represents a color of a button -/
inductive Color
| A | B | C | D | E | F

/-- Represents a sequence of buttons -/
def ButtonSequence := List Color

/-- Checks if two colors are adjacent in a button sequence -/
def areColorsAdjacent (seq : ButtonSequence) (c1 c2 : Color) : Prop :=
  ∃ i, (seq.get? i = some c1 ∧ seq.get? (i+1) = some c2) ∨
       (seq.get? i = some c2 ∧ seq.get? (i+1) = some c1)

/-- Checks if a button sequence satisfies the adjacency condition for all color pairs -/
def satisfiesCondition (seq : ButtonSequence) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → areColorsAdjacent seq c1 c2

/-- The main theorem stating the minimum number of buttons required -/
theorem min_buttons_for_adjacency :
  ∃ (seq : ButtonSequence),
    seq.length = 18 ∧
    satisfiesCondition seq ∧
    ∀ (seq' : ButtonSequence), satisfiesCondition seq' → seq'.length ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_min_buttons_for_adjacency_l4041_404106


namespace NUMINAMATH_CALUDE_jasper_kite_raising_time_l4041_404120

/-- Given Omar's kite-raising rate and Jasper's rate being three times Omar's,
    prove that Jasper takes 10 minutes to raise his kite 600 feet. -/
theorem jasper_kite_raising_time 
  (omar_height : ℝ) 
  (omar_time : ℝ) 
  (jasper_height : ℝ) 
  (omar_height_val : omar_height = 240) 
  (omar_time_val : omar_time = 12) 
  (jasper_height_val : jasper_height = 600) 
  (jasper_rate_mul : ℝ) 
  (jasper_rate_rel : jasper_rate_mul = 3) :
  (jasper_height / (jasper_rate_mul * omar_height / omar_time)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_jasper_kite_raising_time_l4041_404120


namespace NUMINAMATH_CALUDE_pats_stickers_l4041_404119

/-- Pat's sticker problem -/
theorem pats_stickers (initial_stickers earned_stickers : ℕ) 
  (h1 : initial_stickers = 39)
  (h2 : earned_stickers = 22) :
  initial_stickers + earned_stickers = 61 :=
by sorry

end NUMINAMATH_CALUDE_pats_stickers_l4041_404119


namespace NUMINAMATH_CALUDE_cost_price_is_seven_l4041_404121

/-- The cost price of an article satisfying the given condition -/
def cost_price : ℕ := sorry

/-- The selling price that results in a profit -/
def profit_price : ℕ := 54

/-- The selling price that results in a loss -/
def loss_price : ℕ := 40

/-- The profit is equal to the loss -/
axiom profit_equals_loss : profit_price - cost_price = cost_price - loss_price

theorem cost_price_is_seven : cost_price = 7 := by sorry

end NUMINAMATH_CALUDE_cost_price_is_seven_l4041_404121


namespace NUMINAMATH_CALUDE_x_minus_y_equals_fourteen_l4041_404193

theorem x_minus_y_equals_fourteen (x y : ℝ) (h : x^2 + y^2 = 16*x - 12*y + 100) : x - y = 14 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_fourteen_l4041_404193


namespace NUMINAMATH_CALUDE_used_car_selection_l4041_404167

theorem used_car_selection (num_cars : ℕ) (num_clients : ℕ) (selections_per_car : ℕ) 
  (h1 : num_cars = 16)
  (h2 : num_clients = 24)
  (h3 : selections_per_car = 3) :
  (num_cars * selections_per_car) / num_clients = 2 := by
  sorry

end NUMINAMATH_CALUDE_used_car_selection_l4041_404167


namespace NUMINAMATH_CALUDE_three_can_volume_l4041_404196

theorem three_can_volume : 
  ∀ (v1 v2 v3 : ℕ),
  v2 = (3 * v1) / 2 →
  v3 = 64 * v1 / 3 →
  v1 + v2 + v3 < 30 →
  v1 + v2 + v3 = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_three_can_volume_l4041_404196


namespace NUMINAMATH_CALUDE_tangent_line_sum_l4041_404138

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

-- State the theorem
theorem tangent_line_sum (h : tangent_line 1 (f 1)) : f 1 + deriv f 1 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l4041_404138


namespace NUMINAMATH_CALUDE_circle_symmetry_line_coefficient_product_l4041_404176

/-- Given a circle and a line, prove that the product of the line's coefficients is non-positive -/
theorem circle_symmetry_line_coefficient_product (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0 →
    ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 1 = 0 ∧
      2*a*x - b*y + 2 = 2*a*x' - b*y' + 2) →
  a * b ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_coefficient_product_l4041_404176


namespace NUMINAMATH_CALUDE_boys_playing_both_sports_l4041_404157

theorem boys_playing_both_sports (total : ℕ) (basketball : ℕ) (football : ℕ) (neither : ℕ) :
  total = 22 →
  basketball = 13 →
  football = 15 →
  neither = 3 →
  ∃ (both : ℕ), both = 9 ∧ total = basketball + football - both + neither :=
by sorry

end NUMINAMATH_CALUDE_boys_playing_both_sports_l4041_404157


namespace NUMINAMATH_CALUDE_cookie_distribution_l4041_404109

/-- The number of cookies Uncle Jude gave to Tim -/
def cookies_to_tim : ℕ := 15

/-- The total number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 256

/-- The number of cookies Uncle Jude gave to Mike -/
def cookies_to_mike : ℕ := 23

/-- The number of cookies Uncle Jude kept in the fridge -/
def cookies_in_fridge : ℕ := 188

theorem cookie_distribution :
  cookies_to_tim + cookies_to_mike + cookies_in_fridge + 2 * cookies_to_tim = total_cookies :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l4041_404109


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l4041_404133

/-- Definition of the diamond operation -/
def diamond (a b : ℝ) : ℝ := 3 * a - b^2

/-- Theorem stating that if a ◇ 6 = 15, then a = 17 -/
theorem diamond_equation_solution (a : ℝ) : diamond a 6 = 15 → a = 17 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l4041_404133


namespace NUMINAMATH_CALUDE_bisected_areas_correct_l4041_404125

/-- A rectangle with sides 2 meters and 4 meters, divided by angle bisectors -/
structure BisectedRectangle where
  /-- The length of the shorter side of the rectangle -/
  short_side : ℝ
  /-- The length of the longer side of the rectangle -/
  long_side : ℝ
  /-- The short side is 2 meters -/
  short_side_eq : short_side = 2
  /-- The long side is 4 meters -/
  long_side_eq : long_side = 4
  /-- The angle bisectors are drawn from angles adjacent to the longer side -/
  bisectors_from_long_side : Bool

/-- The areas into which the rectangle is divided by the angle bisectors -/
def bisected_areas (rect : BisectedRectangle) : List ℝ :=
  [2, 2, 4]

/-- Theorem stating that the bisected areas are correct -/
theorem bisected_areas_correct (rect : BisectedRectangle) :
  bisected_areas rect = [2, 2, 4] := by
  sorry

end NUMINAMATH_CALUDE_bisected_areas_correct_l4041_404125


namespace NUMINAMATH_CALUDE_max_cars_theorem_l4041_404110

/-- Represents the maximum number of cars that can pass a point on a highway in 30 minutes -/
def M : ℕ := 6000

/-- Theorem stating the maximum number of cars and its relation to M/10 -/
theorem max_cars_theorem :
  (∀ (car_length : ℝ) (time : ℝ),
    car_length = 5 ∧ 
    time = 30 ∧ 
    (∀ (speed : ℝ) (distance : ℝ),
      distance = car_length * (speed / 10))) →
  M = 6000 ∧ M / 10 = 600 := by
  sorry

#check max_cars_theorem

end NUMINAMATH_CALUDE_max_cars_theorem_l4041_404110


namespace NUMINAMATH_CALUDE_staircase_steps_l4041_404141

/-- Represents a staircase with a given number of steps. -/
structure Staircase :=
  (steps : ℕ)

/-- Calculates the total number of toothpicks used in a staircase. -/
def toothpicks (s : Staircase) : ℕ :=
  3 * (s.steps * (s.steps + 1)) / 2

/-- Theorem stating that a staircase with 270 toothpicks has 12 steps. -/
theorem staircase_steps : ∃ s : Staircase, toothpicks s = 270 ∧ s.steps = 12 := by
  sorry


end NUMINAMATH_CALUDE_staircase_steps_l4041_404141


namespace NUMINAMATH_CALUDE_profit_percent_for_given_ratio_l4041_404137

/-- If the ratio of cost price to selling price is 4:5, then the profit percent is 25% -/
theorem profit_percent_for_given_ratio : 
  ∀ (cp sp : ℝ), cp > 0 → sp > 0 → cp / sp = 4 / 5 → (sp - cp) / cp * 100 = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_percent_for_given_ratio_l4041_404137


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4041_404104

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) : 
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4041_404104


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l4041_404178

theorem arithmetic_sequence_length :
  ∀ (a₁ aₙ d n : ℕ),
    a₁ = 1 →
    aₙ = 46 →
    d = 3 →
    aₙ = a₁ + (n - 1) * d →
    n = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l4041_404178


namespace NUMINAMATH_CALUDE_james_lollipops_distribution_l4041_404132

/-- The number of lollipops James has left after distributing to his friends -/
def lollipops_left (total_lollipops : ℕ) (num_friends : ℕ) : ℕ :=
  total_lollipops % num_friends

/-- Theorem stating that James has 0 lollipops left after distribution -/
theorem james_lollipops_distribution :
  let total_lollipops : ℕ := 56 + 130 + 10 + 238
  let num_friends : ℕ := 14
  lollipops_left total_lollipops num_friends = 0 := by sorry

end NUMINAMATH_CALUDE_james_lollipops_distribution_l4041_404132


namespace NUMINAMATH_CALUDE_max_identifiable_bulbs_max_identifiable_bulbs_and_switches_l4041_404172

/-- Represents the state of a bulb -/
inductive BulbState
  | On
  | OffWarm
  | OffCold

/-- Represents a trip to the basement -/
def Trip := Nat → BulbState

/-- The number of trips allowed to the basement -/
def numTrips : Nat := 2

/-- The number of possible states for each bulb in a single trip -/
def statesPerTrip : Nat := 3

/-- Theorem: The maximum number of unique bulb configurations identifiable in two trips -/
theorem max_identifiable_bulbs :
  (statesPerTrip ^ numTrips : Nat) = 9 := by
  sorry

/-- Corollary: The maximum number of bulbs and switches that can be identified with each other in two trips -/
theorem max_identifiable_bulbs_and_switches :
  ∃ (n : Nat), n = 9 ∧ n = (statesPerTrip ^ numTrips : Nat) := by
  sorry

end NUMINAMATH_CALUDE_max_identifiable_bulbs_max_identifiable_bulbs_and_switches_l4041_404172


namespace NUMINAMATH_CALUDE_laptop_price_exceeds_savings_l4041_404124

/-- Proves that for any initial laptop price greater than 0, 
    after 2 years of 6% annual price increase, 
    the laptop price will exceed 56358 rubles -/
theorem laptop_price_exceeds_savings (P₀ : ℝ) (h : P₀ > 0) : 
  P₀ * (1 + 0.06)^2 > 56358 := by
  sorry

#check laptop_price_exceeds_savings

end NUMINAMATH_CALUDE_laptop_price_exceeds_savings_l4041_404124


namespace NUMINAMATH_CALUDE_tank_capacity_is_2000_liters_l4041_404136

-- Define the flow rates and time
def inflow_rate : ℚ := 1 / 2 -- kiloliters per minute
def outflow_rate1 : ℚ := 1 / 4 -- kiloliters per minute
def outflow_rate2 : ℚ := 1 / 6 -- kiloliters per minute
def fill_time : ℚ := 12 -- minutes

-- Define the net flow rate
def net_flow_rate : ℚ := inflow_rate - outflow_rate1 - outflow_rate2

-- Define the theorem
theorem tank_capacity_is_2000_liters :
  let volume_added : ℚ := net_flow_rate * fill_time
  let full_capacity_kl : ℚ := 2 * volume_added
  let full_capacity_l : ℚ := 1000 * full_capacity_kl
  full_capacity_l = 2000 := by sorry

end NUMINAMATH_CALUDE_tank_capacity_is_2000_liters_l4041_404136


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l4041_404129

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 + x - 2) / (x - 1) = 0 ∧ x ≠ 1 → x = -2 :=
by
  sorry

#check fraction_zero_solution

end NUMINAMATH_CALUDE_fraction_zero_solution_l4041_404129


namespace NUMINAMATH_CALUDE_parallel_tangents_l4041_404177

/-- A homogeneous differential equation y' = φ(y/x) -/
noncomputable def homogeneous_de (φ : ℝ → ℝ) (x y : ℝ) : ℝ := φ (y / x)

/-- The slope of the tangent line at a point (x, y) -/
noncomputable def tangent_slope (φ : ℝ → ℝ) (x y : ℝ) : ℝ := homogeneous_de φ x y

theorem parallel_tangents (φ : ℝ → ℝ) (x y x₁ y₁ : ℝ) (hx : x ≠ 0) (hx₁ : x₁ ≠ 0) 
  (h_corresp : y / x = y₁ / x₁) :
  tangent_slope φ x y = tangent_slope φ x₁ y₁ := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_l4041_404177


namespace NUMINAMATH_CALUDE_house_position_l4041_404175

theorem house_position (total_houses : Nat) (product_difference : Nat) : 
  total_houses = 11 → product_difference = 5 → 
  ∃ (position : Nat), position = 4 ∧ 
    (position - 1) * (total_houses - position) = 
    (position - 2) * (total_houses - position + 1) + product_difference := by
  sorry

end NUMINAMATH_CALUDE_house_position_l4041_404175


namespace NUMINAMATH_CALUDE_largest_constant_divisor_inequality_l4041_404155

/-- The number of divisors function -/
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The statement of the theorem -/
theorem largest_constant_divisor_inequality :
  (∃ (c : ℝ), c > 0 ∧
    (∀ (n : ℕ), n ≥ 2 →
      (∃ (d : ℕ), d > 0 ∧ d ∣ n ∧ (d : ℝ) ≤ Real.sqrt n ∧
        (tau d : ℝ) ≥ c * Real.sqrt (tau n : ℝ)))) ∧
  (∀ (c : ℝ), c > Real.sqrt (1 / 2) →
    (∃ (n : ℕ), n ≥ 2 ∧
      (∀ (d : ℕ), d > 0 → d ∣ n → (d : ℝ) ≤ Real.sqrt n →
        (tau d : ℝ) < c * Real.sqrt (tau n : ℝ)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_divisor_inequality_l4041_404155


namespace NUMINAMATH_CALUDE_fraction_value_l4041_404144

theorem fraction_value (y : ℝ) (h : 4 - 9/y + 9/(y^2) = 0) : 3/y = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l4041_404144


namespace NUMINAMATH_CALUDE_number_times_99_equals_2376_l4041_404117

theorem number_times_99_equals_2376 : ∃ x : ℕ, x * 99 = 2376 ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_times_99_equals_2376_l4041_404117


namespace NUMINAMATH_CALUDE_circle_equation_l4041_404139

/-- Theorem: Equation of a Circle
    For any point (x, y) on a circle with radius R and center (a, b),
    the equation (x-a)^2 + (y-b)^2 = R^2 holds. -/
theorem circle_equation (R a b x y : ℝ) (h : (x - a)^2 + (y - b)^2 = R^2) :
  (x - a)^2 + (y - b)^2 = R^2 := by
  sorry

#check circle_equation

end NUMINAMATH_CALUDE_circle_equation_l4041_404139


namespace NUMINAMATH_CALUDE_dave_spent_22_tickets_l4041_404194

def tickets_spent_on_beanie (initial_tickets : ℕ) (additional_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  initial_tickets + additional_tickets - remaining_tickets

theorem dave_spent_22_tickets : 
  tickets_spent_on_beanie 25 15 18 = 22 := by
  sorry

end NUMINAMATH_CALUDE_dave_spent_22_tickets_l4041_404194


namespace NUMINAMATH_CALUDE_boat_distance_against_stream_l4041_404169

/-- Calculates the distance a boat travels against the stream in one hour. -/
def distance_against_stream (boat_speed : ℝ) (distance_with_stream : ℝ) : ℝ :=
  boat_speed - (distance_with_stream - boat_speed)

/-- Theorem: Given a boat with speed 10 km/hr in still water that travels 15 km along the stream in one hour,
    the distance it travels against the stream in one hour is 5 km. -/
theorem boat_distance_against_stream :
  distance_against_stream 10 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_against_stream_l4041_404169


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l4041_404134

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l4041_404134


namespace NUMINAMATH_CALUDE_complex_root_equation_l4041_404189

/-- Given a quadratic equation with complex coefficients and a real parameter,
    prove that if it has a real root, then the complex number formed by the
    parameter and the root has a specific value. -/
theorem complex_root_equation (a : ℝ) (b : ℝ) :
  (∃ x : ℝ, x^2 + (4 + Complex.I) * x + (4 : ℂ) + a * Complex.I = 0) →
  (b^2 + (4 + Complex.I) * b + (4 : ℂ) + a * Complex.I = 0) →
  (a + b * Complex.I = 2 - 2 * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_root_equation_l4041_404189


namespace NUMINAMATH_CALUDE_relationship_abc_l4041_404198

theorem relationship_abc : 
  let a := Real.sqrt 2 / 2 * (Real.sin (17 * π / 180) + Real.cos (17 * π / 180))
  let b := 2 * (Real.cos (13 * π / 180))^2 - 1
  let c := Real.sqrt 3 / 2
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l4041_404198


namespace NUMINAMATH_CALUDE_sallys_purchase_l4041_404173

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents50 : ℕ
  dollars5 : ℕ
  dollars10 : ℕ

/-- The problem statement -/
theorem sallys_purchase (counts : ItemCounts) : 
  counts.cents50 + counts.dollars5 + counts.dollars10 = 30 →
  50 * counts.cents50 + 500 * counts.dollars5 + 1000 * counts.dollars10 = 10000 →
  counts.cents50 = 20 := by
  sorry


end NUMINAMATH_CALUDE_sallys_purchase_l4041_404173


namespace NUMINAMATH_CALUDE_f_of_g_10_l4041_404188

def g (x : ℝ) : ℝ := 4 * x + 6

def f (x : ℝ) : ℝ := 6 * x - 10

theorem f_of_g_10 : f (g 10) = 266 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_10_l4041_404188


namespace NUMINAMATH_CALUDE_division_result_l4041_404184

theorem division_result : (3486 : ℚ) / 189 = 18.444444444444443 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l4041_404184


namespace NUMINAMATH_CALUDE_least_number_divisible_by_multiple_l4041_404151

theorem least_number_divisible_by_multiple (n : ℕ) : n = 856 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 24 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 32 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 36 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 54 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 8) = 24 * k₁ ∧ (n + 8) = 32 * k₂ ∧ (n + 8) = 36 * k₃ ∧ (n + 8) = 54 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_multiple_l4041_404151


namespace NUMINAMATH_CALUDE_principal_calculation_l4041_404179

/-- Prove that the principal is 9200 given the specified conditions -/
theorem principal_calculation (r t : ℝ) (h1 : r = 0.12) (h2 : t = 3) : 
  ∃ P : ℝ, P - (P * r * t) = P - 5888 ∧ P = 9200 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l4041_404179


namespace NUMINAMATH_CALUDE_bag_weight_l4041_404101

theorem bag_weight (w : ℝ) (h : w = 16 / (w / 4)) : w = 16 := by
  sorry

end NUMINAMATH_CALUDE_bag_weight_l4041_404101


namespace NUMINAMATH_CALUDE_yogurt_combinations_l4041_404122

theorem yogurt_combinations (flavors : ℕ) (toppings : ℕ) : 
  flavors = 5 → toppings = 7 → flavors * (toppings.choose 3) = 175 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l4041_404122


namespace NUMINAMATH_CALUDE_optimal_transport_l4041_404118

/-- The daily round-trip frequency as a function of the number of carriages -/
def y (x : ℕ) : ℝ := -2 * x + 24

/-- The total number of carriages operated daily -/
def S (x : ℕ) : ℝ := x * y x

/-- The daily number of passengers transported -/
def W (x : ℕ) : ℝ := 110 * S x

/-- The optimal number of carriages -/
def optimal_carriages : ℕ := 6

/-- The maximum number of passengers transported daily -/
def max_passengers : ℕ := 7920

theorem optimal_transport (x : ℕ) (h : x ≤ 12) :
  y 4 = 16 ∧ y 7 = 10 →
  W optimal_carriages ≥ W x ∧
  W optimal_carriages = max_passengers :=
sorry

end NUMINAMATH_CALUDE_optimal_transport_l4041_404118


namespace NUMINAMATH_CALUDE_park_not_crowded_implies_cool_or_rain_l4041_404158

variable (day : Type) -- Type representing days

-- Define predicates for weather conditions and park status
variable (temp_at_least_70 : day → Prop) -- Temperature is at least 70°F
variable (raining : day → Prop) -- It is raining
variable (crowded : day → Prop) -- The park is crowded

-- Given condition: If temp ≥ 70°F and not raining, then the park is crowded
variable (h : ∀ d : day, (temp_at_least_70 d ∧ ¬raining d) → crowded d)

theorem park_not_crowded_implies_cool_or_rain :
  ∀ d : day, ¬crowded d → (¬temp_at_least_70 d ∨ raining d) :=
by
  sorry

#check park_not_crowded_implies_cool_or_rain

end NUMINAMATH_CALUDE_park_not_crowded_implies_cool_or_rain_l4041_404158


namespace NUMINAMATH_CALUDE_g_range_l4041_404174

noncomputable def f (a x : ℝ) : ℝ := a^x / (1 + a^x)

noncomputable def g (a x : ℝ) : ℤ := 
  ⌊f a x - 1/2⌋ + ⌊f a (-x) - 1/2⌋

theorem g_range (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  ∀ x : ℝ, g a x ∈ ({0, -1} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_g_range_l4041_404174


namespace NUMINAMATH_CALUDE_max_value_of_sequence_l4041_404116

def a (n : ℕ) : ℚ := n / (n^2 + 90)

theorem max_value_of_sequence :
  ∃ (M : ℚ), M = 1/19 ∧ ∀ (n : ℕ), a n ≤ M ∧ ∃ (k : ℕ), a k = M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sequence_l4041_404116


namespace NUMINAMATH_CALUDE_player_field_time_l4041_404112

/-- Given a sports tournament with the following conditions:
  * The team has 10 players
  * 8 players are always on the field
  * The match lasts 45 minutes
  * All players must play the same amount of time
This theorem proves that each player will be on the field for 36 minutes. -/
theorem player_field_time 
  (total_players : ℕ) 
  (field_players : ℕ) 
  (match_duration : ℕ) 
  (h1 : total_players = 10)
  (h2 : field_players = 8)
  (h3 : match_duration = 45) :
  (field_players * match_duration) / total_players = 36 := by
  sorry

end NUMINAMATH_CALUDE_player_field_time_l4041_404112


namespace NUMINAMATH_CALUDE_sqrt_sum_rational_iff_equal_and_in_set_l4041_404143

def is_valid_pair (m n : ℤ) : Prop :=
  ∃ (q : ℚ), (Real.sqrt (n + Real.sqrt 2016) + Real.sqrt (m - Real.sqrt 2016) : ℝ) = q

def valid_n_set : Set ℤ := {505, 254, 130, 65, 50, 46, 45}

theorem sqrt_sum_rational_iff_equal_and_in_set (m n : ℤ) :
  is_valid_pair m n ↔ (m = n ∧ n ∈ valid_n_set) :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_rational_iff_equal_and_in_set_l4041_404143


namespace NUMINAMATH_CALUDE_triangle_side_length_l4041_404126

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, angle B is 60°, and a² + c² = 3ac, then the length of side b is 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- Angle B is 60°
  (a^2 + c^2 = 3 * a * c) →  -- Given equation
  (b = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4041_404126


namespace NUMINAMATH_CALUDE_intersection_implies_a_gt_three_l4041_404154

/-- A function f(x) = x³ - ax² + 4 that intersects the positive x-axis at two different points -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- The property that f intersects the positive x-axis at two different points -/
def intersects_positive_x_axis_twice (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0

/-- If f(x) = x³ - ax² + 4 intersects the positive x-axis at two different points, then a > 3 -/
theorem intersection_implies_a_gt_three :
  ∀ a : ℝ, intersects_positive_x_axis_twice a → a > 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_gt_three_l4041_404154


namespace NUMINAMATH_CALUDE_trajectory_of_symmetric_point_l4041_404146

/-- The equation of the trajectory of point N, which is symmetric to a point M on the circle x^2 + y^2 = 4 with respect to the point A(1,1) -/
theorem trajectory_of_symmetric_point (x y : ℝ) :
  (∃ (mx my : ℝ), mx^2 + my^2 = 4 ∧ x = 2 - mx ∧ y = 2 - my) →
  (x - 2)^2 + (y - 2)^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_symmetric_point_l4041_404146


namespace NUMINAMATH_CALUDE_total_golf_balls_l4041_404190

/-- Represents the number of golf balls in one dozen -/
def dozen : ℕ := 12

/-- Represents the number of dozens Dan buys -/
def dan_dozens : ℕ := 5

/-- Represents the number of dozens Gus buys -/
def gus_dozens : ℕ := 3

/-- Represents the number of dozens Chris buys -/
def chris_dozens : ℕ := 4

/-- Represents the additional golf balls Chris buys -/
def chris_extra : ℕ := 6

/-- Represents the number of dozens Emily buys -/
def emily_dozens : ℕ := 2

/-- Represents the number of dozens Fred buys -/
def fred_dozens : ℕ := 1

/-- Theorem stating the total number of golf balls bought by the friends -/
theorem total_golf_balls :
  (dan_dozens + gus_dozens + chris_dozens + emily_dozens + fred_dozens) * dozen + chris_extra = 186 := by
  sorry

end NUMINAMATH_CALUDE_total_golf_balls_l4041_404190


namespace NUMINAMATH_CALUDE_odd_periodic2_sum_zero_l4041_404159

/-- A function that is odd and has a period of 2 -/
def OddPeriodic2 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = f x)

/-- Theorem: For any odd function with period 2, f(1) + f(4) + f(7) = 0 -/
theorem odd_periodic2_sum_zero (f : ℝ → ℝ) (h : OddPeriodic2 f) :
  f 1 + f 4 + f 7 = 0 := by
  sorry


end NUMINAMATH_CALUDE_odd_periodic2_sum_zero_l4041_404159


namespace NUMINAMATH_CALUDE_probability_two_heads_in_three_flips_l4041_404163

/-- The probability of getting exactly k successes in n trials with probability p for each trial -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- A fair coin has probability 0.5 of landing heads -/
def fair_coin_probability : ℝ := 0.5

/-- The number of flips -/
def num_flips : ℕ := 3

/-- The number of heads we want -/
def num_heads : ℕ := 2

theorem probability_two_heads_in_three_flips :
  binomial_probability num_flips num_heads fair_coin_probability = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_in_three_flips_l4041_404163


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l4041_404145

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 - 4*x - 2*k + 8

-- Define the condition for two real roots
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic x₁ k = 0 ∧ quadratic x₂ k = 0

-- Define the additional condition
def roots_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^3 * x₂ + x₁ * x₂^3 = 24

-- Theorem statement
theorem quadratic_roots_theorem :
  ∀ k : ℝ, has_two_real_roots k →
  (∃ x₁ x₂ : ℝ, quadratic x₁ k = 0 ∧ quadratic x₂ k = 0 ∧ roots_condition x₁ x₂) →
  k = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l4041_404145


namespace NUMINAMATH_CALUDE_rebecca_eggs_l4041_404152

/-- The number of marbles Rebecca has -/
def marbles : ℕ := 6

/-- The difference between the number of eggs and marbles -/
def egg_marble_difference : ℕ := 14

/-- The number of eggs Rebecca has -/
def eggs : ℕ := marbles + egg_marble_difference

theorem rebecca_eggs : eggs = 20 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_l4041_404152


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_shifted_l4041_404114

def f (x : ℝ) : ℝ := x^2 + 2*x - 5

theorem decreasing_interval_of_f_shifted :
  let g := fun (x : ℝ) => f (x - 1)
  ∀ x y : ℝ, x < y ∧ y ≤ 0 → g x > g y :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_shifted_l4041_404114


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4041_404105

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) :
  Complex.abs (1 + z) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4041_404105


namespace NUMINAMATH_CALUDE_log_problem_l4041_404128

theorem log_problem (p q r x : ℝ) (d : ℝ) 
  (hp : Real.log x / Real.log p = 2)
  (hq : Real.log x / Real.log q = 3)
  (hr : Real.log x / Real.log r = 6)
  (hd : Real.log x / Real.log (p * q * r) = d)
  (h_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ x > 0) : d = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l4041_404128


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_specific_a_value_l4041_404149

/-- Hyperbola C: x²/a² - y² = 1 (a > 0) -/
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1 ∧ a > 0

/-- Line l: x + y = 1 -/
def line (x y : ℝ) : Prop := x + y = 1

/-- P is the intersection point of l and the y-axis -/
def P : ℝ × ℝ := (0, 1)

/-- A and B are distinct intersection points of C and l -/
def intersection_points (a : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧
    hyperbola a A.1 A.2 ∧ line A.1 A.2 ∧
    hyperbola a B.1 B.2 ∧ line B.1 B.2

/-- PA = (5/12)PB -/
def vector_relation (A B : ℝ × ℝ) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (5/12 * (B.1 - P.1), 5/12 * (B.2 - P.2))

theorem hyperbola_line_intersection (a : ℝ) :
  intersection_points a → (0 < a ∧ a < 1) ∨ (1 < a ∧ a < Real.sqrt 2) :=
sorry

theorem specific_a_value (a : ℝ) (A B : ℝ × ℝ) :
  hyperbola a A.1 A.2 ∧ line A.1 A.2 ∧
  hyperbola a B.1 B.2 ∧ line B.1 B.2 ∧
  vector_relation A B →
  a = 17/13 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_specific_a_value_l4041_404149


namespace NUMINAMATH_CALUDE_alternating_7x7_grid_difference_l4041_404147

/-- Represents a square on the grid -/
inductive Square
| Dark
| Light

/-- Represents a row in the grid -/
def Row := List Square

/-- Represents the entire grid -/
def Grid := List Row

/-- Generates an alternating row starting with the given square type -/
def alternatingRow (start : Square) (length : Nat) : Row :=
  sorry

/-- Counts the number of dark squares in a row -/
def countDarkInRow (row : Row) : Nat :=
  sorry

/-- Counts the number of light squares in a row -/
def countLightInRow (row : Row) : Nat :=
  sorry

/-- Generates a 7x7 grid with alternating squares, starting with a dark square -/
def generateGrid : Grid :=
  sorry

/-- Counts the total number of dark squares in the grid -/
def countTotalDark (grid : Grid) : Nat :=
  sorry

/-- Counts the total number of light squares in the grid -/
def countTotalLight (grid : Grid) : Nat :=
  sorry

theorem alternating_7x7_grid_difference :
  let grid := generateGrid
  countTotalDark grid = countTotalLight grid + 1 := by
  sorry

end NUMINAMATH_CALUDE_alternating_7x7_grid_difference_l4041_404147


namespace NUMINAMATH_CALUDE_initial_oranges_count_l4041_404197

/-- The number of oranges Susan took from the box -/
def oranges_taken : ℕ := 35

/-- The number of oranges left in the box -/
def oranges_left : ℕ := 20

/-- The initial number of oranges in the box -/
def initial_oranges : ℕ := oranges_taken + oranges_left

theorem initial_oranges_count : initial_oranges = 55 := by
  sorry

end NUMINAMATH_CALUDE_initial_oranges_count_l4041_404197


namespace NUMINAMATH_CALUDE_coupon_discount_percentage_l4041_404191

theorem coupon_discount_percentage 
  (total_bill : ℝ) 
  (num_friends : ℕ) 
  (individual_payment : ℝ) 
  (h1 : total_bill = 100) 
  (h2 : num_friends = 5) 
  (h3 : individual_payment = 18.8) : 
  (total_bill - num_friends * individual_payment) / total_bill * 100 = 6 := by
sorry

end NUMINAMATH_CALUDE_coupon_discount_percentage_l4041_404191


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l4041_404103

theorem max_value_sqrt_sum (x : ℝ) (h : x ∈ Set.Icc (-36) 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) ≤ 12 ∧
  ∃ y ∈ Set.Icc (-36) 36, Real.sqrt (36 + y) + Real.sqrt (36 - y) = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l4041_404103


namespace NUMINAMATH_CALUDE_image_equality_under_composition_condition_l4041_404164

universe u

theorem image_equality_under_composition_condition 
  {S : Type u} [Finite S] (f : S → S) :
  (∀ (g : S → S), g ≠ f → (f ∘ g ∘ f) ≠ (g ∘ f ∘ g)) →
  let T := Set.range f
  f '' T = T := by
  sorry

end NUMINAMATH_CALUDE_image_equality_under_composition_condition_l4041_404164


namespace NUMINAMATH_CALUDE_function_always_negative_implies_a_range_l4041_404195

theorem function_always_negative_implies_a_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h : ∀ x ∈ Set.Ioo 0 1, f x < 0) 
  (h_def : ∀ x, f x = x * |x - a| - 2) : 
  -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_function_always_negative_implies_a_range_l4041_404195


namespace NUMINAMATH_CALUDE_existence_of_infinite_set_with_gcd_property_l4041_404107

theorem existence_of_infinite_set_with_gcd_property :
  ∃ (S : Set ℕ), Set.Infinite S ∧
  (∀ (x y z w : ℕ), x ∈ S → y ∈ S → z ∈ S → w ∈ S →
    x < y → z < w → (x, y) ≠ (z, w) →
    Nat.gcd (x * y + 2022) (z * w + 2022) = 1) :=
sorry

end NUMINAMATH_CALUDE_existence_of_infinite_set_with_gcd_property_l4041_404107


namespace NUMINAMATH_CALUDE_principal_is_720_l4041_404140

/-- Calculates the principal amount given simple interest, time, and rate -/
def calculate_principal (simple_interest : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  simple_interest * 100 / (rate * time)

/-- Theorem stating that the principal amount is 720 given the problem conditions -/
theorem principal_is_720 :
  let simple_interest : ℚ := 180
  let time : ℚ := 4
  let rate : ℚ := 6.25
  calculate_principal simple_interest time rate = 720 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_720_l4041_404140


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l4041_404168

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss (weight_lost : ℝ) (current_weight : ℝ) (loss_per_day : ℝ) :
  weight_lost = 126 →
  current_weight = 66 →
  loss_per_day = 0.5 →
  ∃ (initial_weight : ℝ) (days : ℝ),
    initial_weight = current_weight + weight_lost ∧
    initial_weight = 192 ∧
    days * loss_per_day = weight_lost :=
by sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l4041_404168


namespace NUMINAMATH_CALUDE_min_ratio_four_digit_number_l4041_404108

/-- A structure representing a four-digit number with distinct digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  digits_range : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- The sum of digits of a four-digit number -/
def digit_sum (n : FourDigitNumber) : Nat :=
  n.a + n.b + n.c + n.d

/-- The ratio of a four-digit number to the sum of its digits -/
def ratio (n : FourDigitNumber) : Rat :=
  (value n : Rat) / (digit_sum n : Rat)

theorem min_ratio_four_digit_number :
  ∃ (n : FourDigitNumber), 
    (∀ (m : FourDigitNumber), ratio n ≤ ratio m) ∧ 
    (ratio n = 60.5) ∧
    (value n = 1089) := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_four_digit_number_l4041_404108


namespace NUMINAMATH_CALUDE_locus_of_T_is_tangents_to_C_perp_to_L_l4041_404123

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Represents a point in the plane -/
def Point := ℝ × ℝ

/-- The fixed circle C -/
def C : Circle := sorry

/-- The line L passing through the center of C -/
def L : Line := sorry

/-- A variable point P on L -/
def P : Point := sorry

/-- The circle K centered at P and passing through the center of C -/
def K : Circle := sorry

/-- A point T on K where a common tangent to C and K meets K -/
def T : Point := sorry

/-- The locus of point T -/
def locus_of_T : Set Point := sorry

/-- The pair of tangents to C which are perpendicular to L -/
def tangents_to_C_perp_to_L : Set Point := sorry

theorem locus_of_T_is_tangents_to_C_perp_to_L :
  locus_of_T = tangents_to_C_perp_to_L := by sorry

end NUMINAMATH_CALUDE_locus_of_T_is_tangents_to_C_perp_to_L_l4041_404123


namespace NUMINAMATH_CALUDE_solution_set_theorem_l4041_404186

/-- A function f: ℝ → ℝ is increasing -/
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

/-- The set of x where |f(x)| ≥ 2 -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ := {x : ℝ | |f x| ≥ 2}

theorem solution_set_theorem (f : ℝ → ℝ) 
  (h_increasing : IsIncreasing f) 
  (h_f1 : f 1 = -2) 
  (h_f3 : f 3 = 2) : 
  SolutionSet f = Set.Ici 3 ∪ Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l4041_404186


namespace NUMINAMATH_CALUDE_sum_of_inverse_points_l4041_404150

/-- Given an invertible function f, if f(a) = 3 and f(b) = 7, then a + b = 0 -/
theorem sum_of_inverse_points (f : ℝ → ℝ) (a b : ℝ) 
  (h_inv : Function.Injective f) 
  (h_a : f a = 3) 
  (h_b : f b = 7) : 
  a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_inverse_points_l4041_404150


namespace NUMINAMATH_CALUDE_complex_multiplication_l4041_404170

theorem complex_multiplication : (1 - 2*Complex.I) * (3 + 4*Complex.I) * (-1 + Complex.I) = -9 + 13*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l4041_404170


namespace NUMINAMATH_CALUDE_one_third_of_x_l4041_404161

theorem one_third_of_x (x y : ℚ) : 
  x / y = 15 / 3 → y = 24 → x / 3 = 40 := by sorry

end NUMINAMATH_CALUDE_one_third_of_x_l4041_404161


namespace NUMINAMATH_CALUDE_horses_meet_on_day_9_l4041_404127

/-- Represents the day on which the horses meet --/
def meetingDay : ℕ := 9

/-- The distance between Chang'an and Qi in li --/
def totalDistance : ℚ := 1125

/-- The initial distance covered by the good horse on the first day --/
def goodHorseInitial : ℚ := 103

/-- The daily increase in distance for the good horse --/
def goodHorseIncrease : ℚ := 13

/-- The initial distance covered by the mediocre horse on the first day --/
def mediocreHorseInitial : ℚ := 97

/-- The daily decrease in distance for the mediocre horse --/
def mediocreHorseDecrease : ℚ := 1/2

/-- Theorem stating that the horses meet on the 9th day --/
theorem horses_meet_on_day_9 :
  (meetingDay : ℚ) * (goodHorseInitial + mediocreHorseInitial) +
  (meetingDay * (meetingDay - 1) / 2) * (goodHorseIncrease - mediocreHorseDecrease) =
  2 * totalDistance := by
  sorry

#check horses_meet_on_day_9

end NUMINAMATH_CALUDE_horses_meet_on_day_9_l4041_404127


namespace NUMINAMATH_CALUDE_double_root_values_l4041_404199

/-- A polynomial with integer coefficients of the form x^4 + b₃x³ + b₂x² + b₁x + 50 -/
def IntPolynomial (b₃ b₂ b₁ : ℤ) (x : ℝ) : ℝ := x^4 + b₃*x^3 + b₂*x^2 + b₁*x + 50

/-- s is a double root of the polynomial if both the polynomial and its derivative evaluate to 0 at s -/
def IsDoubleRoot (p : ℝ → ℝ) (s : ℝ) : Prop :=
  p s = 0 ∧ (deriv p) s = 0

theorem double_root_values (b₃ b₂ b₁ : ℤ) (s : ℤ) :
  IsDoubleRoot (IntPolynomial b₃ b₂ b₁) s → s = -5 ∨ s = -1 ∨ s = 1 ∨ s = 5 := by
  sorry

end NUMINAMATH_CALUDE_double_root_values_l4041_404199


namespace NUMINAMATH_CALUDE_balls_remaining_l4041_404166

def initial_balls : ℕ := 10
def removed_balls : ℕ := 3

theorem balls_remaining : initial_balls - removed_balls = 7 := by
  sorry

end NUMINAMATH_CALUDE_balls_remaining_l4041_404166


namespace NUMINAMATH_CALUDE_vacation_duration_l4041_404135

/-- The number of emails received on the first day -/
def first_day_emails : ℕ := 16

/-- The ratio of emails received on each subsequent day compared to the previous day -/
def email_ratio : ℚ := 1/2

/-- The total number of emails received during the vacation -/
def total_emails : ℕ := 30

/-- Calculate the sum of a geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The number of days in the vacation -/
def vacation_days : ℕ := 4

theorem vacation_duration :
  geometric_sum first_day_emails email_ratio vacation_days = total_emails := by
  sorry

end NUMINAMATH_CALUDE_vacation_duration_l4041_404135


namespace NUMINAMATH_CALUDE_function_minimum_l4041_404111

def f (x : ℝ) : ℝ := x^2 - 8*x + 5

theorem function_minimum :
  ∃ (x_min : ℝ), 
    (∀ x, f x ≥ f x_min) ∧ 
    x_min = 4 ∧ 
    f x_min = -11 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_l4041_404111


namespace NUMINAMATH_CALUDE_product_sum_8670_l4041_404185

theorem product_sum_8670 : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8670 ∧ 
  a + b = 187 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_8670_l4041_404185


namespace NUMINAMATH_CALUDE_second_row_starts_with_531_l4041_404171

-- Define the grid type
def Grid := Fin 3 → Fin 3 → Nat

-- Define the valid range of numbers
def ValidNumber (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 5

-- No repetition in rows
def NoRowRepetition (grid : Grid) : Prop :=
  ∀ i j k, j ≠ k → grid i j ≠ grid i k

-- No repetition in columns
def NoColumnRepetition (grid : Grid) : Prop :=
  ∀ i j k, i ≠ k → grid i j ≠ grid k j

-- Divisibility condition
def DivisibilityCondition (grid : Grid) : Prop :=
  ∀ i j, i > 0 → grid i j % grid (i-1) j = 0 ∧
  ∀ i j, j > 0 → grid i j % grid i (j-1) = 0

-- All numbers are valid
def AllValidNumbers (grid : Grid) : Prop :=
  ∀ i j, ValidNumber (grid i j)

-- Main theorem
theorem second_row_starts_with_531 (grid : Grid) 
  (h1 : NoRowRepetition grid)
  (h2 : NoColumnRepetition grid)
  (h3 : DivisibilityCondition grid)
  (h4 : AllValidNumbers grid) :
  grid 1 0 = 5 ∧ grid 1 1 = 1 ∧ grid 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_row_starts_with_531_l4041_404171


namespace NUMINAMATH_CALUDE_brad_profit_l4041_404192

/-- Represents the sizes of lemonade glasses -/
inductive Size
| Small
| Medium
| Large

/-- Represents the lemonade stand data -/
structure LemonadeStand where
  yield_per_gallon : Size → ℕ
  cost_per_gallon : Size → ℚ
  price_per_glass : Size → ℚ
  gallons_made : Size → ℕ
  small_drunk : ℕ
  medium_bought : ℕ
  medium_spilled : ℕ
  large_unsold : ℕ

def brad_stand : LemonadeStand :=
  { yield_per_gallon := λ s => match s with
      | Size.Small => 16
      | Size.Medium => 10
      | Size.Large => 6
    cost_per_gallon := λ s => match s with
      | Size.Small => 2
      | Size.Medium => 7/2
      | Size.Large => 5
    price_per_glass := λ s => match s with
      | Size.Small => 1
      | Size.Medium => 7/4
      | Size.Large => 5/2
    gallons_made := λ _ => 2
    small_drunk := 4
    medium_bought := 3
    medium_spilled := 1
    large_unsold := 2 }

def total_cost (stand : LemonadeStand) : ℚ :=
  (stand.cost_per_gallon Size.Small * stand.gallons_made Size.Small) +
  (stand.cost_per_gallon Size.Medium * stand.gallons_made Size.Medium) +
  (stand.cost_per_gallon Size.Large * stand.gallons_made Size.Large)

def total_revenue (stand : LemonadeStand) : ℚ :=
  (stand.price_per_glass Size.Small * (stand.yield_per_gallon Size.Small * stand.gallons_made Size.Small - stand.small_drunk)) +
  (stand.price_per_glass Size.Medium * (stand.yield_per_gallon Size.Medium * stand.gallons_made Size.Medium - stand.medium_bought)) +
  (stand.price_per_glass Size.Large * (stand.yield_per_gallon Size.Large * stand.gallons_made Size.Large - stand.large_unsold))

def net_profit (stand : LemonadeStand) : ℚ :=
  total_revenue stand - total_cost stand

theorem brad_profit :
  net_profit brad_stand = 247/4 := by
  sorry

end NUMINAMATH_CALUDE_brad_profit_l4041_404192


namespace NUMINAMATH_CALUDE_not_square_sum_ceiling_l4041_404182

theorem not_square_sum_ceiling (a b : ℕ+) : ¬∃ (n : ℕ), (n : ℝ)^2 = (a : ℝ)^2 + ⌈(4 * (a : ℝ)^2) / (b : ℝ)⌉ := by
  sorry

end NUMINAMATH_CALUDE_not_square_sum_ceiling_l4041_404182
