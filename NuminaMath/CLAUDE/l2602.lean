import Mathlib

namespace new_total_bill_l2602_260284

def original_order_cost : ℝ := 25
def tomatoes_old_price : ℝ := 0.99
def tomatoes_new_price : ℝ := 2.20
def lettuce_old_price : ℝ := 1.00
def lettuce_new_price : ℝ := 1.75
def celery_old_price : ℝ := 1.96
def celery_new_price : ℝ := 2.00
def delivery_tip_cost : ℝ := 8.00

theorem new_total_bill :
  let price_increase := (tomatoes_new_price - tomatoes_old_price) +
                        (lettuce_new_price - lettuce_old_price) +
                        (celery_new_price - celery_old_price)
  let new_food_cost := original_order_cost + price_increase
  let total_bill := new_food_cost + delivery_tip_cost
  total_bill = 35 := by
  sorry

end new_total_bill_l2602_260284


namespace next_joint_work_day_is_360_l2602_260214

/-- Represents the work schedule of a tutor -/
structure TutorSchedule where
  cycle : ℕ

/-- Represents the lab schedule -/
structure LabSchedule where
  openDays : Fin 7 → Bool

/-- Calculates the next day all tutors work together -/
def nextJointWorkDay (emma noah olivia liam : TutorSchedule) (lab : LabSchedule) : ℕ :=
  sorry

theorem next_joint_work_day_is_360 :
  let emma : TutorSchedule := { cycle := 5 }
  let noah : TutorSchedule := { cycle := 8 }
  let olivia : TutorSchedule := { cycle := 9 }
  let liam : TutorSchedule := { cycle := 10 }
  let lab : LabSchedule := { openDays := fun d => d < 5 }
  nextJointWorkDay emma noah olivia liam lab = 360 := by
  sorry

end next_joint_work_day_is_360_l2602_260214


namespace min_value_expression_l2602_260216

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 2) :
  (a * c / b) + (c / (a * b)) - (c / 2) + (Real.sqrt 5 / (c - 2)) ≥ Real.sqrt 10 + Real.sqrt 5 :=
by sorry

end min_value_expression_l2602_260216


namespace quadratic_monotonicity_l2602_260220

/-- A quadratic function f(x) = 4x^2 - kx - 8 has monotonicity on the interval (∞, 5] if and only if k ≥ 40 -/
theorem quadratic_monotonicity (k : ℝ) :
  (∀ x > 5, Monotone (fun x => 4 * x^2 - k * x - 8)) ↔ k ≥ 40 := by
  sorry

end quadratic_monotonicity_l2602_260220


namespace ratio_AB_BC_l2602_260230

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The diagram configuration -/
structure Diagram where
  rectangles : Fin 5 → Rectangle
  x : ℝ
  h1 : ∀ i, (rectangles i).width = x
  h2 : ∀ i, (rectangles i).length = 3 * x

/-- AB is the sum of two widths and one length -/
def AB (d : Diagram) : ℝ := 2 * d.x + 3 * d.x

/-- BC is the length of one rectangle -/
def BC (d : Diagram) : ℝ := 3 * d.x

theorem ratio_AB_BC (d : Diagram) : AB d / BC d = 5 / 3 := by
  sorry

end ratio_AB_BC_l2602_260230


namespace randolph_is_55_l2602_260231

def sherry_age : ℕ := 25

def sydney_age : ℕ := 2 * sherry_age

def randolph_age : ℕ := sydney_age + 5

theorem randolph_is_55 : randolph_age = 55 := by
  sorry

end randolph_is_55_l2602_260231


namespace complement_M_intersect_N_l2602_260257

/-- Given sets M and N in the real numbers, prove that the intersection of the complement of M and N is the set of all real numbers less than -2. -/
theorem complement_M_intersect_N (M N : Set ℝ) 
  (hM : M = {x : ℝ | -2 ≤ x ∧ x ≤ 2})
  (hN : N = {x : ℝ | x < 1}) :
  (Mᶜ ∩ N) = {x : ℝ | x < -2} := by
  sorry

end complement_M_intersect_N_l2602_260257


namespace sector_central_angle_l2602_260285

/-- Given a sector with perimeter 6 and area 2, its central angle in radians is either 4 or 1 -/
theorem sector_central_angle (r l : ℝ) : 
  2 * r + l = 6 →
  1 / 2 * l * r = 2 →
  l / r = 4 ∨ l / r = 1 :=
by sorry

end sector_central_angle_l2602_260285


namespace profit_percent_when_cost_is_quarter_of_selling_price_l2602_260248

/-- If the cost price is 25% of the selling price, then the profit percent is 300%. -/
theorem profit_percent_when_cost_is_quarter_of_selling_price :
  ∀ (selling_price : ℝ) (cost_price : ℝ),
    selling_price > 0 →
    cost_price = 0.25 * selling_price →
    (selling_price - cost_price) / cost_price * 100 = 300 := by
  sorry

end profit_percent_when_cost_is_quarter_of_selling_price_l2602_260248


namespace race_heartbeats_l2602_260227

/-- Calculates the total number of heartbeats during a race with varying heart rates. -/
def total_heartbeats (base_rate : ℕ) (distance : ℕ) (pace : ℕ) (rate_increase : ℕ) (increase_start : ℕ) : ℕ :=
  let total_time := distance * pace
  let base_beats := base_rate * total_time
  let increased_distance := distance - increase_start
  let increased_beats := increased_distance * (increased_distance + 1) * rate_increase / 2
  base_beats + increased_beats

/-- Theorem stating the total number of heartbeats during a 20-mile race 
    with specific heart rate conditions. -/
theorem race_heartbeats : 
  total_heartbeats 160 20 6 5 10 = 11475 :=
sorry

end race_heartbeats_l2602_260227


namespace disjoint_chords_with_equal_sum_endpoints_l2602_260207

/-- Given 2^500 points numbered 1 to 2^500 arranged on a circle,
    there exist 100 disjoint chords such that the sum of the endpoints
    is the same for each chord. -/
theorem disjoint_chords_with_equal_sum_endpoints :
  ∃ (chords : Finset (Fin (2^500) × Fin (2^500))) (s : ℕ),
    chords.card = 100 ∧
    (∀ (c₁ c₂ : Fin (2^500) × Fin (2^500)), c₁ ∈ chords → c₂ ∈ chords → c₁ ≠ c₂ →
      (c₁.1 ≠ c₂.1 ∧ c₁.1 ≠ c₂.2 ∧ c₁.2 ≠ c₂.1 ∧ c₁.2 ≠ c₂.2)) ∧
    (∀ (c : Fin (2^500) × Fin (2^500)), c ∈ chords →
      c.1.val + c.2.val = s) :=
by sorry

end disjoint_chords_with_equal_sum_endpoints_l2602_260207


namespace extra_people_on_train_l2602_260274

theorem extra_people_on_train (current : ℕ) (initial : ℕ) (got_off : ℕ)
  (h1 : current = 63)
  (h2 : initial = 78)
  (h3 : got_off = 27) :
  current - (initial - got_off) = 12 :=
by sorry

end extra_people_on_train_l2602_260274


namespace log_sum_equals_two_l2602_260239

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_equals_two : 2 * log 5 10 + log 5 0.25 = 2 := by sorry

end log_sum_equals_two_l2602_260239


namespace price_two_bracelets_is_eight_l2602_260277

/-- Represents the bracelet selling scenario -/
structure BraceletSale where
  initialStock : ℕ
  singlePrice : ℕ
  singleRevenue : ℕ
  totalRevenue : ℕ

/-- Calculates the price for two bracelets -/
def priceTwoBracelets (sale : BraceletSale) : ℕ :=
  let singleSold := sale.singleRevenue / sale.singlePrice
  let remainingBracelets := sale.initialStock - singleSold
  let pairRevenue := sale.totalRevenue - sale.singleRevenue
  let pairsSold := remainingBracelets / 2
  pairRevenue / pairsSold

/-- Theorem stating that the price for two bracelets is 8 -/
theorem price_two_bracelets_is_eight (sale : BraceletSale) 
  (h1 : sale.initialStock = 30)
  (h2 : sale.singlePrice = 5)
  (h3 : sale.singleRevenue = 60)
  (h4 : sale.totalRevenue = 132) : 
  priceTwoBracelets sale = 8 := by
  sorry

end price_two_bracelets_is_eight_l2602_260277


namespace optimal_price_increase_maximizes_profit_l2602_260209

/-- Represents the daily profit function for a meal set -/
structure MealSet where
  baseProfit : ℝ
  baseSales : ℝ
  salesDecreaseRate : ℝ

/-- Calculate the daily profit for a meal set given a price increase -/
def dailyProfit (set : MealSet) (priceIncrease : ℝ) : ℝ :=
  (set.baseProfit + priceIncrease) * (set.baseSales - set.salesDecreaseRate * priceIncrease)

/-- The optimal price increase for meal set A maximizes the total profit -/
theorem optimal_price_increase_maximizes_profit 
  (setA setB : MealSet)
  (totalPriceIncrease : ℝ)
  (hA : setA = { baseProfit := 8, baseSales := 90, salesDecreaseRate := 4 })
  (hB : setB = { baseProfit := 10, baseSales := 70, salesDecreaseRate := 2 })
  (hTotal : totalPriceIncrease = 10) :
  ∃ (x : ℝ), x = 4 ∧ 
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ totalPriceIncrease →
      dailyProfit setA x + dailyProfit setB (totalPriceIncrease - x) ≥
      dailyProfit setA y + dailyProfit setB (totalPriceIncrease - y) :=
by sorry


end optimal_price_increase_maximizes_profit_l2602_260209


namespace fraction_equivalence_l2602_260294

theorem fraction_equivalence (x y : ℝ) (h1 : y ≠ 0) (h2 : x + 2*y ≠ 0) :
  (x + y) / (x + 2*y) = y / (2*y) ↔ x = 0 := by
  sorry

end fraction_equivalence_l2602_260294


namespace sequence_sum_l2602_260246

theorem sequence_sum (A B C D E F G H I : ℝ) : 
  D = 8 →
  A + B + C + D = 50 →
  B + C + D + E = 50 →
  C + D + E + F = 50 →
  D + E + F + G = 50 →
  E + F + G + H = 50 →
  F + G + H + I = 50 →
  A + I = 92 := by
sorry

end sequence_sum_l2602_260246


namespace p_sufficient_not_necessary_for_q_l2602_260221

-- Define the propositions
def p : Prop := (m : ℝ) → m = -1

def q (m : ℝ) : Prop := 
  let line1 : ℝ → ℝ → Prop := λ x y => x - y = 0
  let line2 : ℝ → ℝ → Prop := λ x y => x + m^2 * y = 0
  ∀ x1 y1 x2 y2, line1 x1 y1 → line2 x2 y2 → 
    (x2 - x1) * (y2 - y1) + (x2 - x1) * (x2 - x1) = 0

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, p → q m) ∧ (∃ m : ℝ, q m ∧ ¬p) := by sorry

end p_sufficient_not_necessary_for_q_l2602_260221


namespace final_state_is_twelve_and_fourteen_l2602_260211

/-- Represents the numbers on the blackboard -/
inductive Number
  | eleven
  | twelve
  | thirteen
  | fourteen
  | fifteen

/-- The state of the blackboard -/
structure BoardState where
  counts : Number → Nat
  total : Nat

/-- The initial state of the blackboard -/
def initial_state : BoardState := {
  counts := λ n => match n with
    | Number.eleven => 11
    | Number.twelve => 12
    | Number.thirteen => 13
    | Number.fourteen => 14
    | Number.fifteen => 15
  total := 65
}

/-- Represents an operation on the board -/
def operation (s : BoardState) : BoardState :=
  sorry  -- Implementation of the operation

/-- Predicate to check if a state has exactly two numbers remaining -/
def has_two_remaining (s : BoardState) : Prop :=
  (s.total = 2) ∧ (∃ a b : Number, a ≠ b ∧ s.counts a > 0 ∧ s.counts b > 0 ∧ 
    ∀ c : Number, c ≠ a ∧ c ≠ b → s.counts c = 0)

/-- The main theorem -/
theorem final_state_is_twelve_and_fourteen :
  ∃ (n : Nat), 
    let final_state := (operation^[n] initial_state)
    has_two_remaining final_state ∧ 
    final_state.counts Number.twelve > 0 ∧ 
    final_state.counts Number.fourteen > 0 :=
  sorry


end final_state_is_twelve_and_fourteen_l2602_260211


namespace f_zero_points_range_l2602_260261

/-- The function f(x) = ax^2 + x - 1 + 3a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - 1 + 3 * a

/-- The set of a values for which f has zero points in [-1, 1] -/
def A : Set ℝ := {a : ℝ | ∃ x, x ∈ Set.Icc (-1 : ℝ) 1 ∧ f a x = 0}

theorem f_zero_points_range :
  A = Set.Icc (0 : ℝ) (1/2) :=
sorry

end f_zero_points_range_l2602_260261


namespace platform_length_l2602_260262

/-- Given a train of length 200 meters that crosses a platform in 50 seconds
    and a signal pole in 42 seconds, the length of the platform is 38 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) 
  (h1 : train_length = 200)
  (h2 : time_platform = 50)
  (h3 : time_pole = 42) :
  let train_speed := train_length / time_pole
  let platform_length := train_speed * time_platform - train_length
  platform_length = 38 := by
  sorry


end platform_length_l2602_260262


namespace car_wash_earnings_l2602_260244

def weekly_allowance : ℝ := 8
def final_amount : ℝ := 12

theorem car_wash_earnings :
  final_amount - (weekly_allowance / 2) = 8 := by sorry

end car_wash_earnings_l2602_260244


namespace kite_long_diagonal_angle_in_circular_arrangement_l2602_260267

/-- Represents a symmetrical kite in a circular arrangement -/
structure Kite where
  long_diagonal_angle : ℝ
  short_diagonal_angle : ℝ

/-- Represents a circular arrangement of kites -/
structure CircularArrangement where
  num_kites : ℕ
  kites : Fin num_kites → Kite
  covers_circle : Bool
  long_diagonals_meet_center : Bool

/-- The theorem stating the long diagonal angle in the specific arrangement -/
theorem kite_long_diagonal_angle_in_circular_arrangement 
  (arr : CircularArrangement) 
  (h1 : arr.num_kites = 10) 
  (h2 : arr.covers_circle = true) 
  (h3 : arr.long_diagonals_meet_center = true) :
  ∀ i, (arr.kites i).long_diagonal_angle = 162 :=
sorry

end kite_long_diagonal_angle_in_circular_arrangement_l2602_260267


namespace tangent_parallel_points_l2602_260219

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

/-- The slope of the line that the tangent should be parallel to -/
def m : ℝ := 4

theorem tangent_parallel_points :
  {p : ℝ × ℝ | p.1 = -1 ∧ p.2 = -4 ∨ p.1 = 1 ∧ p.2 = 0} =
  {p : ℝ × ℝ | p.2 = f p.1 ∧ f' p.1 = m} :=
sorry

end tangent_parallel_points_l2602_260219


namespace fib_80_mod_7_l2602_260242

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The period of the Fibonacci sequence modulo 7 -/
def fib_mod7_period : ℕ := 16

theorem fib_80_mod_7 :
  fib 80 % 7 = 0 :=
by
  sorry

end fib_80_mod_7_l2602_260242


namespace no_valid_coloring_l2602_260269

/-- Represents a coloring of a rectangular grid --/
def GridColoring (m n : ℕ) := Fin m → Fin n → Bool

/-- Checks if the number of white cells equals the number of black cells --/
def equalColors (m n : ℕ) (coloring : GridColoring m n) : Prop :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => if coloring i j then 1 else 0)) =
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => if coloring i j then 0 else 1))

/-- Checks if more than 3/4 of cells in each row are of the same color --/
def rowColorDominance (m n : ℕ) (coloring : GridColoring m n) : Prop :=
  ∀ i, (4 * (Finset.univ.sum fun j => if coloring i j then 1 else 0) > 3 * n) ∨
       (4 * (Finset.univ.sum fun j => if coloring i j then 0 else 1) > 3 * n)

/-- Checks if more than 3/4 of cells in each column are of the same color --/
def columnColorDominance (m n : ℕ) (coloring : GridColoring m n) : Prop :=
  ∀ j, (4 * (Finset.univ.sum fun i => if coloring i j then 1 else 0) > 3 * m) ∨
       (4 * (Finset.univ.sum fun i => if coloring i j then 0 else 1) > 3 * m)

/-- The main theorem stating that no valid coloring exists --/
theorem no_valid_coloring (m n : ℕ) : ¬∃ (coloring : GridColoring m n),
  equalColors m n coloring ∧ rowColorDominance m n coloring ∧ columnColorDominance m n coloring :=
sorry

end no_valid_coloring_l2602_260269


namespace equilateral_triangle_circle_radius_l2602_260298

theorem equilateral_triangle_circle_radius (r : ℝ) 
  (h : r > 0) : 
  (3 * (r * Real.sqrt 3) = π * r^2) → 
  r = (3 * Real.sqrt 3) / π := by
  sorry

end equilateral_triangle_circle_radius_l2602_260298


namespace min_value_absolute_sum_l2602_260237

theorem min_value_absolute_sum (x y : ℝ) : 
  |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 := by
  sorry

end min_value_absolute_sum_l2602_260237


namespace completing_square_equivalence_l2602_260240

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 - 8*x + 1 = 0) ↔ ((x - 4)^2 = 15) := by
  sorry

end completing_square_equivalence_l2602_260240


namespace clara_stickers_l2602_260283

theorem clara_stickers (initial_stickers : ℕ) (stickers_to_boy : ℕ) (final_stickers : ℕ) : 
  initial_stickers = 100 →
  final_stickers = 45 →
  final_stickers = (initial_stickers - stickers_to_boy) / 2 →
  stickers_to_boy = 10 := by
  sorry

end clara_stickers_l2602_260283


namespace circle_outside_triangle_percentage_l2602_260213

theorem circle_outside_triangle_percentage
  (A : ℝ) -- Total area
  (A_intersection : ℝ) -- Area of intersection
  (A_triangle_outside : ℝ) -- Area of triangle outside circle
  (h1 : A > 0) -- Total area is positive
  (h2 : A_intersection = 0.45 * A) -- Intersection is 45% of total area
  (h3 : A_triangle_outside = 0.4 * A) -- Triangle outside is 40% of total area
  : (A - A_intersection - A_triangle_outside) / (A_intersection + (A - A_intersection - A_triangle_outside)) = 0.25 := by
  sorry

end circle_outside_triangle_percentage_l2602_260213


namespace polar_line_properties_l2602_260256

/-- A line in polar coordinates passing through (2, π/3) and parallel to the polar axis -/
def polar_line (r θ : ℝ) : Prop :=
  r * Real.sin θ = Real.sqrt 3

theorem polar_line_properties :
  ∀ (r θ : ℝ),
    polar_line r θ →
    (r = 2 ∧ θ = π/3 → polar_line 2 (π/3)) ∧
    (∀ (r' : ℝ), polar_line r' θ → r' * Real.sin θ = Real.sqrt 3) :=
by sorry

end polar_line_properties_l2602_260256


namespace race_fourth_part_length_l2602_260255

/-- Given a 4-part race with specified lengths for the first three parts,
    calculate the length of the fourth part. -/
theorem race_fourth_part_length 
  (total_length : ℝ) 
  (first_part : ℝ) 
  (second_part : ℝ) 
  (third_part : ℝ) 
  (h1 : total_length = 74.5)
  (h2 : first_part = 15.5)
  (h3 : second_part = 21.5)
  (h4 : third_part = 21.5) :
  total_length - (first_part + second_part + third_part) = 16 := by
sorry

end race_fourth_part_length_l2602_260255


namespace arc_length_of_sector_l2602_260247

/-- Given a circle with radius 4 cm and a sector with an area of 7 square centimeters,
    the length of the arc forming this sector is 3.5 cm. -/
theorem arc_length_of_sector (r : ℝ) (area : ℝ) (arc_length : ℝ) : 
  r = 4 → area = 7 → arc_length = (area * 2) / r → arc_length = 3.5 := by
  sorry

end arc_length_of_sector_l2602_260247


namespace gcd_lcm_sum_l2602_260282

theorem gcd_lcm_sum : Nat.gcd 54 72 + Nat.lcm 50 15 = 168 := by
  sorry

end gcd_lcm_sum_l2602_260282


namespace dot_path_length_on_rotating_cube_l2602_260204

/-- The path length of a dot on a rotating cube -/
theorem dot_path_length_on_rotating_cube (cube_edge : ℝ) (h_edge : cube_edge = 2) :
  let dot_radius : ℝ := cube_edge / 2
  let path_length : ℝ := 2 * Real.pi * dot_radius
  path_length = 2 * Real.pi := by sorry

end dot_path_length_on_rotating_cube_l2602_260204


namespace cube_root_simplification_l2602_260202

theorem cube_root_simplification :
  (20^3 + 30^3 + 40^3 : ℝ)^(1/3) = 10 * 99^(1/3) :=
by sorry

end cube_root_simplification_l2602_260202


namespace optimal_config_is_minimum_l2602_260280

/-- Represents the types of vans available --/
inductive VanType
  | A
  | B
  | C

/-- Capacity of each van type --/
def vanCapacity : VanType → ℕ
  | VanType.A => 7
  | VanType.B => 9
  | VanType.C => 12

/-- Available number of each van type --/
def availableVans : VanType → ℕ
  | VanType.A => 3
  | VanType.B => 4
  | VanType.C => 2

/-- Total number of people to transport --/
def totalPeople : ℕ := 40 + 14

/-- A configuration of vans --/
structure VanConfiguration where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculate the total capacity of a given van configuration --/
def totalCapacity (config : VanConfiguration) : ℕ :=
  config.typeA * vanCapacity VanType.A +
  config.typeB * vanCapacity VanType.B +
  config.typeC * vanCapacity VanType.C

/-- Check if a van configuration is valid (within available vans) --/
def isValidConfiguration (config : VanConfiguration) : Prop :=
  config.typeA ≤ availableVans VanType.A ∧
  config.typeB ≤ availableVans VanType.B ∧
  config.typeC ≤ availableVans VanType.C

/-- The optimal van configuration --/
def optimalConfig : VanConfiguration :=
  { typeA := 0, typeB := 4, typeC := 2 }

/-- Theorem stating that the optimal configuration is the minimum number of vans needed --/
theorem optimal_config_is_minimum :
  isValidConfiguration optimalConfig ∧
  totalCapacity optimalConfig ≥ totalPeople ∧
  ∀ (config : VanConfiguration),
    isValidConfiguration config →
    totalCapacity config ≥ totalPeople →
    config.typeA + config.typeB + config.typeC ≥
    optimalConfig.typeA + optimalConfig.typeB + optimalConfig.typeC :=
by
  sorry


end optimal_config_is_minimum_l2602_260280


namespace number_of_paths_l2602_260260

def grid_width : ℕ := 6
def grid_height : ℕ := 5
def path_length : ℕ := 8
def steps_right : ℕ := grid_width - 1
def steps_up : ℕ := grid_height - 1

theorem number_of_paths : 
  Nat.choose path_length steps_up = Nat.choose path_length (path_length - steps_right) := by
  sorry

end number_of_paths_l2602_260260


namespace intersection_M_N_l2602_260273

open Set Real

def M : Set ℝ := {x | Real.exp (x - 1) > 1}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_M_N : M ∩ N = Ioo 1 3 := by sorry

end intersection_M_N_l2602_260273


namespace kabulek_numbers_are_correct_l2602_260250

/-- A four-digit number is a Kabulek number if it equals the square of the sum of its first two digits and last two digits. -/
def isKabulek (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ ∃ a b : ℕ, 
    a ≥ 10 ∧ a < 100 ∧ b ≥ 0 ∧ b < 100 ∧
    n = 100 * a + b ∧ n = (a + b)^2

/-- The set of all four-digit Kabulek numbers. -/
def kabulekNumbers : Set ℕ := {2025, 3025, 9801}

/-- Theorem stating that the set of all four-digit Kabulek numbers is exactly {2025, 3025, 9801}. -/
theorem kabulek_numbers_are_correct : 
  ∀ n : ℕ, isKabulek n ↔ n ∈ kabulekNumbers := by sorry

end kabulek_numbers_are_correct_l2602_260250


namespace consecutive_odd_integers_sum_and_product_l2602_260238

theorem consecutive_odd_integers_sum_and_product :
  ∀ x : ℚ,
  (x + 4 = 4 * x) →
  (x + (x + 4) = 20 / 3) ∧
  (x * (x + 4) = 64 / 9) := by
  sorry

end consecutive_odd_integers_sum_and_product_l2602_260238


namespace skittles_distribution_l2602_260249

theorem skittles_distribution (total_skittles : ℕ) (skittles_per_person : ℕ) (people : ℕ) :
  total_skittles = 20 →
  skittles_per_person = 2 →
  people * skittles_per_person = total_skittles →
  people = 10 := by
  sorry

end skittles_distribution_l2602_260249


namespace quadratic_inequality_l2602_260200

theorem quadratic_inequality (x : ℝ) : -3 * x^2 + 9 * x + 6 > 0 ↔ x < -1 ∨ x > 4 := by
  sorry

end quadratic_inequality_l2602_260200


namespace simplify_nested_expression_l2602_260281

theorem simplify_nested_expression (x : ℝ) : 1 - (1 - (1 + (1 - (1 + (1 - x))))) = x := by
  sorry

end simplify_nested_expression_l2602_260281


namespace identify_alkali_metal_l2602_260259

/-- Represents an alkali metal with its atomic mass -/
structure AlkaliMetal where
  atomic_mass : ℝ

/-- Represents a mixture of an alkali metal and its oxide -/
structure Mixture (R : AlkaliMetal) where
  initial_mass : ℝ
  final_mass : ℝ

/-- Theorem: If a mixture of alkali metal R and its oxide R₂O weighs 10.8 grams,
    and after reaction with water and drying, the resulting solid weighs 16 grams,
    then the atomic mass of R is 23. -/
theorem identify_alkali_metal (R : AlkaliMetal) (mix : Mixture R) :
  mix.initial_mass = 10.8 ∧ mix.final_mass = 16 → R.atomic_mass = 23 := by
  sorry

#check identify_alkali_metal

end identify_alkali_metal_l2602_260259


namespace equal_roots_quadratic_l2602_260252

/-- 
For a quadratic equation x^2 - x + n = 0, if it has two equal real roots,
then n = 1/4.
-/
theorem equal_roots_quadratic (n : ℝ) : 
  (∃ x : ℝ, x^2 - x + n = 0 ∧ (∀ y : ℝ, y^2 - y + n = 0 → y = x)) → n = 1/4 := by
  sorry

end equal_roots_quadratic_l2602_260252


namespace banana_orange_equivalence_l2602_260201

theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (3/4 : ℚ) * 16 * banana_value = 12 * orange_value →
  (3/5 : ℚ) * 20 * banana_value = 12 * orange_value := by
sorry

end banana_orange_equivalence_l2602_260201


namespace fraction_simplification_l2602_260293

theorem fraction_simplification (x y : ℚ) (hx : x = 4/3) (hy : y = 8/6) : 
  (6 * x^2 + 4 * y) / (36 * x * y) = 1/4 := by
  sorry

end fraction_simplification_l2602_260293


namespace min_dials_for_same_remainder_l2602_260271

/-- A dial is a regular 12-sided polygon with numbers from 1 to 12 -/
def Dial := Fin 12 → Fin 12

/-- A stack of dials -/
def Stack := ℕ → Dial

/-- The sum of numbers in a column of the stack -/
def columnSum (s : Stack) (col : Fin 12) (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => (s i col).val + 1)

/-- Whether all column sums have the same remainder modulo 12 -/
def allColumnsSameRemainder (s : Stack) (n : ℕ) : Prop :=
  ∀ (c₁ c₂ : Fin 12), columnSum s c₁ n % 12 = columnSum s c₂ n % 12

/-- The theorem stating that 12 is the minimum number of dials required -/
theorem min_dials_for_same_remainder :
  ∀ (s : Stack), (∃ (n : ℕ), allColumnsSameRemainder s n) →
  (∃ (m : ℕ), m = 12 ∧ allColumnsSameRemainder s m ∧
    ∀ (k : ℕ), k < m → ¬allColumnsSameRemainder s k) :=
sorry

end min_dials_for_same_remainder_l2602_260271


namespace gamblers_initial_win_rate_l2602_260233

theorem gamblers_initial_win_rate 
  (initial_games : ℕ) 
  (additional_games : ℕ) 
  (new_win_rate : ℚ) 
  (final_win_rate : ℚ) :
  initial_games = 30 →
  additional_games = 30 →
  new_win_rate = 4/5 →
  final_win_rate = 3/5 →
  ∃ (initial_win_rate : ℚ),
    initial_win_rate = 2/5 ∧
    (initial_win_rate * initial_games + new_win_rate * additional_games) / (initial_games + additional_games) = final_win_rate :=
by sorry

end gamblers_initial_win_rate_l2602_260233


namespace sum_of_squares_l2602_260203

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_cubes_eq_sum_sevenths : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/7 := by
  sorry

end sum_of_squares_l2602_260203


namespace paul_initial_books_l2602_260291

/-- Represents the number of books and pens Paul has -/
structure PaulsItems where
  books : ℕ
  pens : ℕ

/-- Represents the change in Paul's items after the garage sale -/
structure GarageSale where
  booksRemaining : ℕ
  pensRemaining : ℕ
  booksSold : ℕ

def initialItems : PaulsItems where
  books := 0  -- Unknown initial number of books
  pens := 55

def afterSale : GarageSale where
  booksRemaining := 66
  pensRemaining := 59
  booksSold := 42

theorem paul_initial_books :
  initialItems.books = afterSale.booksRemaining + afterSale.booksSold :=
by sorry

end paul_initial_books_l2602_260291


namespace mike_practice_hours_l2602_260212

/-- Calculates the number of hours Mike practices every weekday -/
def weekday_practice_hours (days_in_week : ℕ) (practice_days_per_week : ℕ) 
  (saturday_hours : ℕ) (total_weeks : ℕ) (total_practice_hours : ℕ) : ℕ :=
  let total_practice_days := practice_days_per_week * total_weeks
  let total_saturdays := total_weeks
  let saturday_practice_hours := saturday_hours * total_saturdays
  let weekday_practice_hours := total_practice_hours - saturday_practice_hours
  let total_weekdays := (practice_days_per_week - 1) * total_weeks
  weekday_practice_hours / total_weekdays

/-- Theorem stating that Mike practices 3 hours every weekday -/
theorem mike_practice_hours : 
  weekday_practice_hours 7 6 5 3 60 = 3 := by
  sorry

end mike_practice_hours_l2602_260212


namespace complex_magnitude_one_l2602_260276

theorem complex_magnitude_one (z : ℂ) (p : ℕ) (h : 11 * z^10 + 10 * Complex.I * z^p + 10 * Complex.I * z - 11 = 0) : 
  Complex.abs z = 1 := by
sorry

end complex_magnitude_one_l2602_260276


namespace sum_exterior_angles_dodecagon_l2602_260222

/-- A regular dodecagon is a polygon with 12 sides. -/
def RegularDodecagon : Type := Unit

/-- The sum of exterior angles of a polygon. -/
def SumOfExteriorAngles (p : Type) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of a regular dodecagon is 360°. -/
theorem sum_exterior_angles_dodecagon :
  SumOfExteriorAngles RegularDodecagon = 360 := by sorry

end sum_exterior_angles_dodecagon_l2602_260222


namespace units_digit_of_8429_pow_1246_l2602_260295

theorem units_digit_of_8429_pow_1246 :
  (8429^1246) % 10 = 1 := by
  sorry

end units_digit_of_8429_pow_1246_l2602_260295


namespace ac_plus_bd_equals_negative_ten_l2602_260223

theorem ac_plus_bd_equals_negative_ten
  (a b c d : ℝ)
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 3)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 6) :
  a * c + b * d = -10 := by
  sorry

end ac_plus_bd_equals_negative_ten_l2602_260223


namespace modular_inverse_100_mod_101_l2602_260226

theorem modular_inverse_100_mod_101 : ∃ x : ℕ, x ≤ 100 ∧ (100 * x) % 101 = 1 := by
  sorry

end modular_inverse_100_mod_101_l2602_260226


namespace smallest_product_of_primes_above_50_l2602_260275

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def first_prime_above_50 : ℕ := 53

def second_prime_above_50 : ℕ := 59

theorem smallest_product_of_primes_above_50 :
  (is_prime first_prime_above_50) ∧
  (is_prime second_prime_above_50) ∧
  (first_prime_above_50 > 50) ∧
  (second_prime_above_50 > 50) ∧
  (first_prime_above_50 < second_prime_above_50) ∧
  (∀ p : ℕ, is_prime p ∧ p > 50 ∧ p ≠ first_prime_above_50 → p ≥ second_prime_above_50) →
  first_prime_above_50 * second_prime_above_50 = 3127 :=
by sorry

end smallest_product_of_primes_above_50_l2602_260275


namespace convex_polygon_27_diagonals_has_9_sides_l2602_260217

/-- The number of diagonals in a convex n-gon --/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 27 diagonals has 9 sides --/
theorem convex_polygon_27_diagonals_has_9_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 27 → n = 9 := by
  sorry

end convex_polygon_27_diagonals_has_9_sides_l2602_260217


namespace strings_needed_is_302_l2602_260292

/-- Calculates the total number of strings needed for a set of instruments, including extra strings due to machine malfunction --/
def total_strings_needed (num_basses : ℕ) (strings_per_bass : ℕ) (guitar_multiplier : ℕ) 
  (strings_per_guitar : ℕ) (eight_string_guitar_reduction : ℕ) (strings_per_eight_string_guitar : ℕ)
  (strings_per_twelve_string_guitar : ℕ) (nylon_strings_per_eight_string_guitar : ℕ)
  (nylon_strings_per_twelve_string_guitar : ℕ) (malfunction_rate : ℕ) : ℕ :=
  let num_guitars := num_basses * guitar_multiplier
  let num_eight_string_guitars := num_guitars - eight_string_guitar_reduction
  let num_twelve_string_guitars := num_basses
  let total_strings := 
    num_basses * strings_per_bass +
    num_guitars * strings_per_guitar +
    num_eight_string_guitars * strings_per_eight_string_guitar +
    num_twelve_string_guitars * strings_per_twelve_string_guitar
  let extra_strings := (total_strings + malfunction_rate - 1) / malfunction_rate
  total_strings + extra_strings

/-- Theorem stating that given the specific conditions, the total number of strings needed is 302 --/
theorem strings_needed_is_302 : 
  total_strings_needed 5 4 3 6 2 8 12 2 6 10 = 302 := by
  sorry

end strings_needed_is_302_l2602_260292


namespace horner_rule_v4_horner_rule_correct_l2602_260224

def horner_polynomial (x : ℝ) : ℝ := 3*x^6 + 5*x^5 + 6*x^4 + 20*x^3 - 8*x^2 + 35*x + 12

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 20
  v3 * x - 8

theorem horner_rule_v4 :
  horner_v4 (-2) = -16 :=
by sorry

theorem horner_rule_correct :
  horner_v4 (-2) = horner_polynomial (-2) :=
by sorry

end horner_rule_v4_horner_rule_correct_l2602_260224


namespace intersection_on_line_x_eq_4_l2602_260232

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the intersection points M and N
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line_l m p.1 p.2}

-- Define the line AM
def line_AM (m : ℝ) (x y : ℝ) : Prop :=
  ∃ (M : ℝ × ℝ), M ∈ intersection_points m ∧
  (y - point_A.2) * (M.1 - point_A.1) = (x - point_A.1) * (M.2 - point_A.2)

-- Define the line BN
def line_BN (m : ℝ) (x y : ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), N ∈ intersection_points m ∧
  (y - point_B.2) * (N.1 - point_B.1) = (x - point_B.1) * (N.2 - point_B.2)

-- Theorem statement
theorem intersection_on_line_x_eq_4 (m : ℝ) :
  ∃ (x y : ℝ), line_AM m x y ∧ line_BN m x y → x = 4 := by
  sorry

end intersection_on_line_x_eq_4_l2602_260232


namespace sculpture_cost_in_yuan_l2602_260205

theorem sculpture_cost_in_yuan 
  (usd_to_nam : ℝ) -- Exchange rate from USD to Namibian dollars
  (usd_to_cny : ℝ) -- Exchange rate from USD to Chinese yuan
  (cost_nam : ℝ) -- Cost of the sculpture in Namibian dollars
  (h1 : usd_to_nam = 8) -- 1 USD = 8 Namibian dollars
  (h2 : usd_to_cny = 5) -- 1 USD = 5 Chinese yuan
  (h3 : cost_nam = 160) -- The sculpture costs 160 Namibian dollars
  : cost_nam / usd_to_nam * usd_to_cny = 100 := by
  sorry

end sculpture_cost_in_yuan_l2602_260205


namespace polynomial_difference_divisibility_l2602_260218

theorem polynomial_difference_divisibility 
  (a b c d : ℤ) (x y : ℤ) (h : x ≠ y) :
  ∃ k : ℤ, (x - y) * k = 
    (a * x^3 + b * x^2 + c * x + d) - (a * y^3 + b * y^2 + c * y + d) := by
  sorry

end polynomial_difference_divisibility_l2602_260218


namespace estimate_above_120_l2602_260241

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  total_students : ℕ
  mean : ℝ
  std_dev : ℝ
  prob_100_to_110 : ℝ

/-- Estimates the number of students scoring above a given threshold -/
def estimate_students_above (sd : ScoreDistribution) (threshold : ℝ) : ℕ := sorry

/-- The main theorem to prove -/
theorem estimate_above_120 (sd : ScoreDistribution) 
  (h1 : sd.total_students = 50)
  (h2 : sd.mean = 110)
  (h3 : sd.std_dev = 10)
  (h4 : sd.prob_100_to_110 = 0.36) :
  estimate_students_above sd 120 = 7 := by sorry

end estimate_above_120_l2602_260241


namespace parabola_perpendicular_chords_locus_l2602_260208

/-- Given a parabola y^2 = 4px where p > 0, with two perpendicular chords OA and OB
    drawn from the vertex O(0,0), the locus of the projection of O onto AB
    is a circle with equation (x - 2p)^2 + y^2 = 4p^2 -/
theorem parabola_perpendicular_chords_locus (p : ℝ) (h : p > 0) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*p*x}
  let O := (0 : ℝ × ℝ)
  let perpendicular_chords := {(OA, OB) : (ℝ × ℝ) × (ℝ × ℝ) |
    O.1 = 0 ∧ O.2 = 0 ∧
    OA ∈ parabola ∧ OB ∈ parabola ∧
    (OA.2 - O.2) * (OB.2 - O.2) = -(OA.1 - O.1) * (OB.1 - O.1)}
  let projection := {M : ℝ × ℝ | ∃ (OA OB : ℝ × ℝ), (OA, OB) ∈ perpendicular_chords ∧
    (M.2 - O.2) * (OA.1 - OB.1) = (M.1 - O.1) * (OA.2 - OB.2)}
  projection = {(x, y) : ℝ × ℝ | (x - 2*p)^2 + y^2 = 4*p^2} :=
by sorry


end parabola_perpendicular_chords_locus_l2602_260208


namespace sufficient_not_necessary_l2602_260225

/-- A line in the form y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Checks if a line has equal intercepts on the coordinate axes -/
def hasEqualIntercepts (l : Line) : Prop :=
  ∃ c : ℝ, c ≠ 0 ∧ l.k * c + l.b = -c ∧ l.b = c

/-- The specific line y = kx + 2k - 1 -/
def specificLine (k : ℝ) : Line :=
  { k := k, b := 2 * k - 1 }

/-- The condition k = -1 is sufficient but not necessary for the line to have equal intercepts -/
theorem sufficient_not_necessary :
  (∀ k : ℝ, k = -1 → hasEqualIntercepts (specificLine k)) ∧
  (∃ k : ℝ, k ≠ -1 ∧ hasEqualIntercepts (specificLine k)) :=
sorry

end sufficient_not_necessary_l2602_260225


namespace total_hot_dogs_is_fifteen_l2602_260228

/-- Represents the number of hot dogs served at each meal -/
structure HotDogMeals where
  breakfast : ℕ
  lunch : ℕ
  dinner : ℕ

/-- Proves that the total number of hot dogs served is 15 given the conditions -/
theorem total_hot_dogs_is_fifteen (h : HotDogMeals) :
  (h.breakfast = 2 * h.dinner) →
  (h.lunch = 9) →
  (h.lunch = h.breakfast + h.dinner + 3) →
  (h.breakfast + h.lunch + h.dinner = 15) := by
  sorry

end total_hot_dogs_is_fifteen_l2602_260228


namespace sphere_radius_is_six_l2602_260297

/-- A truncated cone with horizontal bases of radii 12 and 3, and a sphere tangent to its top, bottom, and lateral surface. -/
structure TruncatedConeWithSphere where
  lower_radius : ℝ
  upper_radius : ℝ
  sphere_radius : ℝ
  lower_radius_eq : lower_radius = 12
  upper_radius_eq : upper_radius = 3
  sphere_tangent : True  -- We can't directly express tangency in this simple structure

/-- The radius of the sphere in the TruncatedConeWithSphere is 6. -/
theorem sphere_radius_is_six (cone : TruncatedConeWithSphere) : cone.sphere_radius = 6 := by
  sorry

end sphere_radius_is_six_l2602_260297


namespace disneyland_attractions_permutations_l2602_260254

theorem disneyland_attractions_permutations :
  Nat.factorial 6 = 720 := by
  sorry

end disneyland_attractions_permutations_l2602_260254


namespace book_arrangement_count_l2602_260263

theorem book_arrangement_count : ℕ := by
  -- Define the number of math books and English books
  let math_books : ℕ := 4
  let english_books : ℕ := 4

  -- Define the number of ways to arrange math books
  let math_arrangements : ℕ := Nat.factorial math_books

  -- Define the number of ways to arrange English books
  let english_arrangements : ℕ := Nat.factorial english_books

  -- Define the number of ways to arrange the two blocks (always 1 in this case)
  let block_arrangements : ℕ := 1

  -- Calculate the total number of arrangements
  let total_arrangements : ℕ := block_arrangements * math_arrangements * english_arrangements

  -- Prove that the total number of arrangements is 576
  sorry

-- The final statement to be proven
#check book_arrangement_count

end book_arrangement_count_l2602_260263


namespace jamie_oliver_vacation_cost_l2602_260279

def vacation_cost (num_people : ℕ) (num_days : ℕ) (ticket_cost : ℕ) (hotel_cost_per_day : ℕ) : ℕ :=
  num_people * ticket_cost + num_people * hotel_cost_per_day * num_days

theorem jamie_oliver_vacation_cost :
  vacation_cost 2 3 24 12 = 120 := by
  sorry

end jamie_oliver_vacation_cost_l2602_260279


namespace arrangement_count_is_180_l2602_260234

/-- The number of ways to select 4 students from 5 and assign them to 3 subjects --/
def arrangement_count : ℕ := 180

/-- The total number of students --/
def total_students : ℕ := 5

/-- The number of students to be selected --/
def selected_students : ℕ := 4

/-- The number of subjects --/
def subject_count : ℕ := 3

/-- Theorem stating that the number of arrangements is 180 --/
theorem arrangement_count_is_180 :
  arrangement_count = 
    subject_count * 
    (Nat.choose total_students 2) * 
    (Nat.choose (total_students - 2) 1) * 
    (Nat.choose (total_students - 3) 1) :=
by sorry

end arrangement_count_is_180_l2602_260234


namespace random_events_count_l2602_260288

-- Define the type for events
inductive Event
| DiceRoll : Event
| PearFall : Event
| LotteryWin : Event
| SecondChild : Event
| WaterBoil : Event

-- Define a function to check if an event is random
def isRandom (e : Event) : Bool :=
  match e with
  | Event.DiceRoll => true
  | Event.PearFall => false
  | Event.LotteryWin => true
  | Event.SecondChild => true
  | Event.WaterBoil => false

-- Define the list of events
def eventList : List Event := [
  Event.DiceRoll,
  Event.PearFall,
  Event.LotteryWin,
  Event.SecondChild,
  Event.WaterBoil
]

-- Theorem: The number of random events in the list is 3
theorem random_events_count : 
  (eventList.filter isRandom).length = 3 := by
  sorry

end random_events_count_l2602_260288


namespace difference_of_squares_l2602_260210

theorem difference_of_squares (x : ℝ) : x^2 - 36 = (x + 6) * (x - 6) := by sorry

end difference_of_squares_l2602_260210


namespace debby_water_bottles_l2602_260287

/-- The number of water bottles Debby drank in one day -/
def bottles_drank : ℕ := 144

/-- The number of water bottles Debby has left -/
def bottles_left : ℕ := 157

/-- The initial number of water bottles Debby bought -/
def initial_bottles : ℕ := bottles_drank + bottles_left

theorem debby_water_bottles : initial_bottles = 301 := by
  sorry

end debby_water_bottles_l2602_260287


namespace vegetables_per_week_l2602_260278

theorem vegetables_per_week (total_points : ℕ) (points_per_vegetable : ℕ) 
  (num_students : ℕ) (num_weeks : ℕ) 
  (h1 : total_points = 200)
  (h2 : points_per_vegetable = 2)
  (h3 : num_students = 25)
  (h4 : num_weeks = 2) :
  (total_points / points_per_vegetable / num_students) / num_weeks = 2 :=
by
  sorry

#check vegetables_per_week

end vegetables_per_week_l2602_260278


namespace geometric_series_sum_l2602_260243

/-- The sum of a geometric series with first term 2, common ratio -2, and last term 1024 -/
def geometricSeriesSum : ℤ := -682

/-- The first term of the geometric series -/
def firstTerm : ℤ := 2

/-- The common ratio of the geometric series -/
def commonRatio : ℤ := -2

/-- The last term of the geometric series -/
def lastTerm : ℤ := 1024

theorem geometric_series_sum :
  ∃ (n : ℕ), n > 0 ∧ firstTerm * commonRatio^(n - 1) = lastTerm ∧
  geometricSeriesSum = firstTerm * (commonRatio^n - 1) / (commonRatio - 1) :=
sorry

end geometric_series_sum_l2602_260243


namespace difference_of_numbers_l2602_260265

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 - y^2 = 50) : x - y = 10 := by
  sorry

end difference_of_numbers_l2602_260265


namespace sum_in_base_b_l2602_260286

/-- Given a base b, this function converts a number from base b to base 10 --/
def toBase10 (b : ℕ) (x : ℕ) : ℕ := sorry

/-- Given a base b, this function converts a number from base 10 to base b --/
def fromBase10 (b : ℕ) (x : ℕ) : ℕ := sorry

/-- The product of 12, 15, and 16 in base b --/
def product (b : ℕ) : ℕ := toBase10 b 12 * toBase10 b 15 * toBase10 b 16

/-- The sum of 12, 15, and 16 in base b --/
def sum (b : ℕ) : ℕ := toBase10 b 12 + toBase10 b 15 + toBase10 b 16

theorem sum_in_base_b (b : ℕ) :
  (product b = toBase10 b 3146) → (fromBase10 b (sum b) = 44) := by
  sorry

end sum_in_base_b_l2602_260286


namespace chord_bisected_by_point_l2602_260215

/-- The equation of an ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- The midpoint of two points -/
def is_midpoint (x₁ y₁ x₂ y₂ x_mid y_mid : ℝ) : Prop :=
  x_mid = (x₁ + x₂) / 2 ∧ y_mid = (y₁ + y₂) / 2

/-- A point is on a line -/
def is_on_line (x y : ℝ) : Prop := x + 2*y - 8 = 0

/-- The main theorem -/
theorem chord_bisected_by_point (x₁ y₁ x₂ y₂ : ℝ) :
  is_on_ellipse x₁ y₁ →
  is_on_ellipse x₂ y₂ →
  is_midpoint x₁ y₁ x₂ y₂ 4 2 →
  is_on_line x₁ y₁ ∧ is_on_line x₂ y₂ :=
sorry

end chord_bisected_by_point_l2602_260215


namespace range_of_f_l2602_260270

-- Define the function
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of the function
theorem range_of_f :
  ∀ y : ℝ, (y ≥ -8) ↔ ∃ x : ℝ, f x = y :=
by
  sorry

end range_of_f_l2602_260270


namespace drawing_probability_comparison_l2602_260253

theorem drawing_probability_comparison : 
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let draws : ℕ := 3

  let prob_with_replacement : ℚ := 3 / 8
  let prob_without_replacement : ℚ := 5 / 12

  prob_without_replacement > prob_with_replacement := by
  sorry

end drawing_probability_comparison_l2602_260253


namespace f_minimized_at_x_min_l2602_260251

/-- The quadratic function we're minimizing -/
def f (x : ℝ) := 2 * x^2 - 8 * x + 6

/-- The value of x that minimizes f -/
def x_min : ℝ := 2

theorem f_minimized_at_x_min :
  ∀ x : ℝ, f x_min ≤ f x :=
sorry

end f_minimized_at_x_min_l2602_260251


namespace rent_is_840_l2602_260299

/-- The total rent for a pasture shared by three people --/
def total_rent (a_horses b_horses c_horses : ℕ) (a_months b_months c_months : ℕ) (b_rent : ℕ) : ℕ :=
  let a_horse_months := a_horses * a_months
  let b_horse_months := b_horses * b_months
  let c_horse_months := c_horses * c_months
  let total_horse_months := a_horse_months + b_horse_months + c_horse_months
  (b_rent * total_horse_months) / b_horse_months

/-- Theorem stating that the total rent is 840 given the problem conditions --/
theorem rent_is_840 :
  total_rent 12 16 18 8 9 6 348 = 840 := by
  sorry

end rent_is_840_l2602_260299


namespace symmetric_line_values_l2602_260258

/-- Two lines are symmetric with respect to the origin if for any point (x, y) on one line,
    the point (-x, -y) lies on the other line. -/
def symmetric_lines (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a * x + 3 * y - 9 = 0 ↔ x - 3 * y + b = 0

/-- If the line ax + 3y - 9 = 0 is symmetric to the line x - 3y + b = 0
    with respect to the origin, then a = -1 and b = -9. -/
theorem symmetric_line_values (a b : ℝ) (h : symmetric_lines a b) : a = -1 ∧ b = -9 := by
  sorry

end symmetric_line_values_l2602_260258


namespace inequality_solution_l2602_260296

theorem inequality_solution (x : ℝ) :
  (x + 2) / ((x + 1)^2) < 0 ↔ x < -2 ∧ x ≠ -1 := by
  sorry

end inequality_solution_l2602_260296


namespace imaginary_part_of_complex_fraction_l2602_260245

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i * i = -1 → Complex.im (5 * i / (1 - 2 * i)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l2602_260245


namespace area_between_concentric_circles_l2602_260266

/-- Given two concentric circles where a chord is tangent to the smaller circle,
    this theorem proves the area of the region between the circles. -/
theorem area_between_concentric_circles
  (outer_radius inner_radius chord_length : ℝ)
  (h_outer_positive : 0 < outer_radius)
  (h_inner_positive : 0 < inner_radius)
  (h_outer_greater : inner_radius < outer_radius)
  (h_chord_tangent : chord_length^2 = outer_radius^2 - inner_radius^2)
  (h_chord_length : chord_length = 100) :
  (outer_radius^2 - inner_radius^2) * π = 2000 * π :=
sorry

end area_between_concentric_circles_l2602_260266


namespace cone_max_volume_l2602_260236

/-- A cone with slant height 20 cm has maximum volume when its height is (20√3)/3 cm. -/
theorem cone_max_volume (h : ℝ) (h_pos : 0 < h) (h_bound : h < 20) :
  let r := Real.sqrt (400 - h^2)
  let v := (1/3) * Real.pi * h * r^2
  (∀ h' : ℝ, 0 < h' → h' < 20 → 
    (1/3) * Real.pi * h' * (Real.sqrt (400 - h'^2))^2 ≤ v) →
  h = 20 * Real.sqrt 3 / 3 := by
sorry


end cone_max_volume_l2602_260236


namespace phone_repair_amount_is_10_l2602_260290

/-- The amount earned from repairing a phone -/
def phone_repair_amount : ℝ := sorry

/-- The amount earned from repairing a laptop -/
def laptop_repair_amount : ℝ := 20

/-- The total number of phones repaired -/
def total_phones : ℕ := 3 + 5

/-- The total number of laptops repaired -/
def total_laptops : ℕ := 2 + 4

/-- The total amount earned -/
def total_earned : ℝ := 200

theorem phone_repair_amount_is_10 :
  phone_repair_amount * total_phones + laptop_repair_amount * total_laptops = total_earned ∧
  phone_repair_amount = 10 := by sorry

end phone_repair_amount_is_10_l2602_260290


namespace emma_savings_l2602_260229

theorem emma_savings (initial_savings withdrawal deposit final_savings : ℕ) : 
  initial_savings = 230 →
  final_savings = 290 →
  deposit = 2 * withdrawal →
  final_savings = initial_savings - withdrawal + deposit →
  withdrawal = 60 := by
sorry

end emma_savings_l2602_260229


namespace sum_after_removal_l2602_260235

theorem sum_after_removal (numbers : List ℝ) (avg : ℝ) (removed : ℝ) :
  numbers.length = 8 →
  numbers.sum / numbers.length = avg →
  avg = 5.2 →
  removed = 4.6 →
  removed ∈ numbers →
  (numbers.erase removed).sum = 37 := by
  sorry

end sum_after_removal_l2602_260235


namespace fifteenth_term_ratio_l2602_260206

/-- Represents an arithmetic series -/
structure ArithmeticSeries where
  first_term : ℚ
  common_difference : ℚ

/-- Sum of the first n terms of an arithmetic series -/
def sum_n_terms (series : ArithmeticSeries) (n : ℕ) : ℚ :=
  n * (2 * series.first_term + (n - 1) * series.common_difference) / 2

/-- The nth term of an arithmetic series -/
def nth_term (series : ArithmeticSeries) (n : ℕ) : ℚ :=
  series.first_term + (n - 1) * series.common_difference

theorem fifteenth_term_ratio 
  (series1 series2 : ArithmeticSeries)
  (h : ∀ n : ℕ, sum_n_terms series1 n / sum_n_terms series2 n = (5 * n + 3) / (3 * n + 11)) :
  nth_term series1 15 / nth_term series2 15 = 71 / 52 := by
  sorry

end fifteenth_term_ratio_l2602_260206


namespace largest_centrally_symmetric_polygon_in_triangle_l2602_260268

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A centrally symmetric polygon -/
structure CentrallySymmetricPolygon where
  vertices : List (ℝ × ℝ)
  center : ℝ × ℝ
  isSymmetric : ∀ v ∈ vertices, ∃ v' ∈ vertices, v' = (2 * center.1 - v.1, 2 * center.2 - v.2)

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isPointInTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Check if a polygon is inside a triangle -/
def isPolygonInTriangle (p : CentrallySymmetricPolygon) (t : Triangle) : Prop :=
  ∀ v ∈ p.vertices, isPointInTriangle v t

/-- The area of a centrally symmetric polygon -/
def polygonArea (p : CentrallySymmetricPolygon) : ℝ := sorry

/-- The theorem stating that the largest centrally symmetric polygon 
    inscribed in a triangle has 2/3 the area of the triangle -/
theorem largest_centrally_symmetric_polygon_in_triangle 
  (t : Triangle) : 
  (∃ p : CentrallySymmetricPolygon, 
    isPolygonInTriangle p t ∧ 
    (∀ q : CentrallySymmetricPolygon, 
      isPolygonInTriangle q t → polygonArea q ≤ polygonArea p)) → 
  (∃ p : CentrallySymmetricPolygon, 
    isPolygonInTriangle p t ∧ 
    polygonArea p = (2/3) * triangleArea t) := by
  sorry

end largest_centrally_symmetric_polygon_in_triangle_l2602_260268


namespace complex_number_property_l2602_260272

theorem complex_number_property (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end complex_number_property_l2602_260272


namespace melanie_initial_dimes_l2602_260264

/-- The number of dimes Melanie had initially -/
def initial_dimes : ℕ := sorry

/-- The number of dimes Melanie received from her dad -/
def dimes_from_dad : ℕ := 8

/-- The number of dimes Melanie received from her mother -/
def dimes_from_mom : ℕ := 4

/-- The total number of dimes Melanie has now -/
def total_dimes_now : ℕ := 19

/-- Theorem stating that Melanie initially had 7 dimes -/
theorem melanie_initial_dimes : 
  initial_dimes = 7 :=
by sorry

end melanie_initial_dimes_l2602_260264


namespace largest_divisor_of_even_squares_sum_l2602_260289

theorem largest_divisor_of_even_squares_sum (m n : ℕ) : 
  Even m → Even n → n < m → (∀ k : ℕ, k > 4 → ∃ m' n' : ℕ, 
    Even m' ∧ Even n' ∧ n' < m' ∧ ¬(k ∣ m'^2 + n'^2)) ∧ 
  (∀ m' n' : ℕ, Even m' → Even n' → n' < m' → (4 ∣ m'^2 + n'^2)) := by
  sorry

end largest_divisor_of_even_squares_sum_l2602_260289
