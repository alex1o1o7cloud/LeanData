import Mathlib

namespace george_oranges_l124_12443

def orange_problem (betty sandra emily frank george : ℕ) : Prop :=
  betty = 12 ∧
  sandra = 3 * betty ∧
  emily = 7 * sandra ∧
  frank = 5 * emily ∧
  george = (5/2 : ℚ) * frank

theorem george_oranges :
  ∀ betty sandra emily frank george : ℕ,
  orange_problem betty sandra emily frank george →
  george = 3150 :=
by
  sorry

end george_oranges_l124_12443


namespace quadratic_equation_solution_l124_12429

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = (11 + Real.sqrt 13) / 6 ∧
  x₂ = (11 - Real.sqrt 13) / 6 ∧
  (x₁ - 2) * (3 * x₁ - 5) = 1 ∧
  (x₂ - 2) * (3 * x₂ - 5) = 1 :=
by sorry

end quadratic_equation_solution_l124_12429


namespace expression_evaluation_l124_12447

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := 5
  let z : ℚ := 3
  (3 * x^5 + 4 * y^3 + z^2) / 7 = 605 / 7 := by
  sorry

end expression_evaluation_l124_12447


namespace lights_remaining_on_l124_12410

def total_lights : ℕ := 2007

def lights_on_after_toggle (n : ℕ) : Prop :=
  let multiples_of_2 := (total_lights - 1) / 2
  let multiples_of_3 := total_lights / 3
  let multiples_of_5 := (total_lights - 2) / 5
  let multiples_of_6 := (total_lights - 3) / 6
  let multiples_of_10 := (total_lights - 7) / 10
  let multiples_of_15 := (total_lights - 12) / 15
  let multiples_of_30 := (total_lights - 27) / 30
  let toggled := multiples_of_2 + multiples_of_3 + multiples_of_5 - 
                 multiples_of_6 - multiples_of_10 - multiples_of_15 + 
                 multiples_of_30
  n = total_lights - toggled

theorem lights_remaining_on : lights_on_after_toggle 1004 := by sorry

end lights_remaining_on_l124_12410


namespace equation_solution_l124_12495

theorem equation_solution :
  let f : ℝ → ℝ := fun x => 0.05 * x + 0.09 * (30 + x)
  ∃! x : ℝ, f x = 15.3 - 3.3 ∧ x = 465 / 7 := by
  sorry

end equation_solution_l124_12495


namespace tangent_function_property_l124_12436

theorem tangent_function_property (ω : ℝ) (h1 : ω > 0) : 
  (∀ x : ℝ, ∃ y : ℝ, y > x ∧ Real.tan (ω * y) = Real.tan (ω * x) ∧ y - x = π / 4) → 
  Real.tan (ω * (π / 4)) = 0 := by
sorry

end tangent_function_property_l124_12436


namespace restaurant_hiring_l124_12471

/-- Given a restaurant with cooks and waiters, prove the number of newly hired waiters. -/
theorem restaurant_hiring (initial_cooks initial_waiters new_waiters : ℕ) : 
  initial_cooks * 11 = initial_waiters * 3 →  -- Initial ratio of cooks to waiters is 3:11
  initial_cooks * 5 = (initial_waiters + new_waiters) * 1 →  -- New ratio is 1:5
  initial_cooks = 9 →  -- There are 9 cooks
  new_waiters = 12 :=  -- Prove that 12 waiters were hired
by sorry

end restaurant_hiring_l124_12471


namespace dinner_cost_l124_12468

theorem dinner_cost (total_bill : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (service_rate : ℝ)
  (h_total : total_bill = 34.5)
  (h_tax : tax_rate = 0.095)
  (h_tip : tip_rate = 0.18)
  (h_service : service_rate = 0.05) :
  ∃ (base_cost : ℝ), 
    base_cost * (1 + tax_rate + tip_rate + service_rate) = total_bill ∧ 
    base_cost = 26 := by
  sorry

end dinner_cost_l124_12468


namespace paula_paint_usage_l124_12432

/-- Represents the paint capacity and usage scenario --/
structure PaintScenario where
  initial_capacity : ℕ  -- Initial room painting capacity
  lost_cans : ℕ         -- Number of paint cans lost
  remaining_capacity : ℕ -- Remaining room painting capacity

/-- Calculates the number of cans used given a paint scenario --/
def cans_used (scenario : PaintScenario) : ℕ :=
  scenario.remaining_capacity / ((scenario.initial_capacity - scenario.remaining_capacity) / scenario.lost_cans)

/-- Theorem stating that for the given scenario, 17 cans were used --/
theorem paula_paint_usage : 
  let scenario : PaintScenario := { 
    initial_capacity := 42, 
    lost_cans := 4, 
    remaining_capacity := 34 
  }
  cans_used scenario = 17 := by sorry

end paula_paint_usage_l124_12432


namespace total_vehicles_l124_12401

/-- Proves that the total number of vehicles on a lot is 400, given the specified conditions -/
theorem total_vehicles (total dodge hyundai kia : ℕ) : 
  dodge = total / 2 →
  hyundai = dodge / 2 →
  kia = 100 →
  total = dodge + hyundai + kia →
  total = 400 := by
sorry

end total_vehicles_l124_12401


namespace fruit_ratio_l124_12438

def total_fruit : ℕ := 13
def remaining_fruit : ℕ := 9

def fruit_fell_out : ℕ := total_fruit - remaining_fruit

theorem fruit_ratio : 
  (fruit_fell_out : ℚ) / total_fruit = 4 / 13 := by sorry

end fruit_ratio_l124_12438


namespace ellipse_intersecting_line_fixed_point_l124_12479

/-- An ellipse with center at origin and axes along coordinate axes -/
structure Ellipse where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0
  hne : a ≠ b

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in slope-intercept form -/
structure Line where
  k : ℝ
  t : ℝ

def Ellipse.standardEq (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def Line.eq (l : Line) (p : Point) : Prop :=
  p.y = l.k * p.x + l.t

def tangentAt (e : Ellipse) (l : Line) (p : Point) : Prop :=
  Ellipse.standardEq e p ∧ Line.eq l p

def intersects (e : Ellipse) (l : Line) (a b : Point) : Prop :=
  Ellipse.standardEq e a ∧ Ellipse.standardEq e b ∧ Line.eq l a ∧ Line.eq l b

def circleDiameterPassesThrough (a b c : Point) : Prop :=
  (a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y) = 0

theorem ellipse_intersecting_line_fixed_point 
  (e : Ellipse) (l : Line) (p a b : Point) :
  e.a^2 = 3 →
  e.b^2 = 4 →
  p.x = 3/2 →
  p.y = 1 →
  tangentAt e { k := 2, t := 4 } p →
  (∃ a b, intersects e l a b ∧ 
    a ≠ b ∧ 
    a.x ≠ e.a ∧ a.x ≠ -e.a ∧ 
    b.x ≠ e.a ∧ b.x ≠ -e.a ∧
    circleDiameterPassesThrough a b { x := 0, y := 2 }) →
  l.eq { x := 0, y := 2/7 } :=
sorry

end ellipse_intersecting_line_fixed_point_l124_12479


namespace all_functions_increasing_l124_12494

-- Define the functions
def f₁ (x : ℝ) : ℝ := 2 * x
def f₂ (x : ℝ) : ℝ := x^2 + 2*x - 1
def f₃ (x : ℝ) : ℝ := abs (x + 2)
def f₄ (x : ℝ) : ℝ := abs x + 2

-- Define the interval [0, +∞)
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Theorem statement
theorem all_functions_increasing :
  (∀ x y, nonnegative x → nonnegative y → x < y → f₁ x < f₁ y) ∧
  (∀ x y, nonnegative x → nonnegative y → x < y → f₂ x < f₂ y) ∧
  (∀ x y, nonnegative x → nonnegative y → x < y → f₃ x < f₃ y) ∧
  (∀ x y, nonnegative x → nonnegative y → x < y → f₄ x < f₄ y) :=
by sorry

end all_functions_increasing_l124_12494


namespace at_least_one_negative_l124_12455

theorem at_least_one_negative (a b c d : ℝ) 
  (sum1 : a + b = 1) 
  (sum2 : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  (a < 0) ∨ (b < 0) ∨ (c < 0) ∨ (d < 0) := by
  sorry

end at_least_one_negative_l124_12455


namespace line_intersection_yz_plane_specific_line_intersection_l124_12497

/-- The line passing through two given points intersects the yz-plane at a specific point. -/
theorem line_intersection_yz_plane (p₁ p₂ : ℝ × ℝ × ℝ) (h : p₁ ≠ p₂) :
  let line := λ t : ℝ => p₁ + t • (p₂ - p₁)
  ∃ t, line t = (0, 8, -5/2) :=
by
  sorry

/-- The specific instance of the line intersection problem. -/
theorem specific_line_intersection :
  let p₁ : ℝ × ℝ × ℝ := (3, 5, 1)
  let p₂ : ℝ × ℝ × ℝ := (5, 3, 6)
  let line := λ t : ℝ => p₁ + t • (p₂ - p₁)
  ∃ t, line t = (0, 8, -5/2) :=
by
  sorry

end line_intersection_yz_plane_specific_line_intersection_l124_12497


namespace prob_same_color_is_69_200_l124_12409

def total_balls : ℕ := 8 + 5 + 7

def prob_blue : ℚ := 8 / total_balls
def prob_green : ℚ := 5 / total_balls
def prob_red : ℚ := 7 / total_balls

def prob_same_color : ℚ := prob_blue^2 + prob_green^2 + prob_red^2

theorem prob_same_color_is_69_200 : prob_same_color = 69 / 200 := by
  sorry

end prob_same_color_is_69_200_l124_12409


namespace cone_lateral_surface_area_l124_12407

/-- The lateral surface area of a cone with base radius 1 and height √3 is 2π -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (h^2 + r^2)
  let lateral_area : ℝ := π * r * l
  lateral_area = 2 * π := by sorry

end cone_lateral_surface_area_l124_12407


namespace football_practice_hours_l124_12456

/-- Calculates the daily practice hours for a football team -/
def dailyPracticeHours (totalHours weekDays missedDays : ℕ) : ℚ :=
  totalHours / (weekDays - missedDays)

/-- Proves that the football team practices 5 hours daily -/
theorem football_practice_hours :
  let totalHours : ℕ := 30
  let weekDays : ℕ := 7
  let missedDays : ℕ := 1
  dailyPracticeHours totalHours weekDays missedDays = 5 := by
sorry

end football_practice_hours_l124_12456


namespace chipmunk_acorns_count_l124_12491

/-- The number of acorns hidden by the chipmunk in each hole -/
def chipmunk_acorns_per_hole : ℕ := 3

/-- The number of acorns hidden by the squirrel in each hole -/
def squirrel_acorns_per_hole : ℕ := 4

/-- The number of holes dug by the chipmunk -/
def chipmunk_holes : ℕ := 16

/-- The number of holes dug by the squirrel -/
def squirrel_holes : ℕ := chipmunk_holes - 4

/-- The total number of acorns hidden by the chipmunk -/
def chipmunk_total_acorns : ℕ := chipmunk_acorns_per_hole * chipmunk_holes

/-- The total number of acorns hidden by the squirrel -/
def squirrel_total_acorns : ℕ := squirrel_acorns_per_hole * squirrel_holes

theorem chipmunk_acorns_count : chipmunk_total_acorns = 48 ∧ chipmunk_total_acorns = squirrel_total_acorns :=
by sorry

end chipmunk_acorns_count_l124_12491


namespace periodic_double_period_l124_12446

open Real

/-- A function f is a-periodic if it satisfies the given functional equation. -/
def IsPeriodic (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a > 0 ∧ ∀ x, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)

/-- If a function is a-periodic, then it is also 2a-periodic. -/
theorem periodic_double_period (f : ℝ → ℝ) (a : ℝ) (h : IsPeriodic f a) :
  ∀ x, f (x + 2*a) = f x := by
  sorry

end periodic_double_period_l124_12446


namespace critical_point_and_zeros_l124_12464

noncomputable section

def f (a : ℝ) (x : ℝ) := Real.exp x * Real.sin x - a * Real.log (x + 1)

def f_derivative (a : ℝ) (x : ℝ) := Real.exp x * (Real.sin x + Real.cos x) - a / (x + 1)

theorem critical_point_and_zeros (a : ℝ) :
  (f_derivative a 0 = 0 → a = 1) ∧
  ((∃ x₁ ∈ Set.Ioo (-1 : ℝ) 0, f a x₁ = 0) ∧
   (∃ x₂ ∈ Set.Ioo (Real.pi / 4) Real.pi, f a x₂ = 0) →
   0 < a ∧ a < 1) :=
by sorry

-- Given condition
axiom given_inequality : Real.sqrt 2 / 2 * Real.exp (Real.pi / 4) > 1

end critical_point_and_zeros_l124_12464


namespace max_value_condition_l124_12415

/-- The function f(x) = -x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x

/-- The maximum value of f(x) on the interval [0, 1] is 2 iff a = -2√2 or a = 3 -/
theorem max_value_condition (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧ (∃ x ∈ Set.Icc 0 1, f a x = 2) ↔ 
  (a = -2 * Real.sqrt 2 ∨ a = 3) := by
sorry

end max_value_condition_l124_12415


namespace fifth_day_sale_l124_12405

def average_sale : ℕ := 625
def num_days : ℕ := 5
def day1_sale : ℕ := 435
def day2_sale : ℕ := 927
def day3_sale : ℕ := 855
def day4_sale : ℕ := 230

theorem fifth_day_sale :
  ∃ (day5_sale : ℕ),
    day5_sale = average_sale * num_days - (day1_sale + day2_sale + day3_sale + day4_sale) ∧
    day5_sale = 678 := by
  sorry

end fifth_day_sale_l124_12405


namespace fourth_number_in_first_set_l124_12488

theorem fourth_number_in_first_set (x : ℝ) (y : ℝ) : 
  (28 + x + 70 + y + 104) / 5 = 67 →
  (50 + 62 + 97 + 124 + x) / 5 = 75.6 →
  y = 88 := by
sorry

end fourth_number_in_first_set_l124_12488


namespace solution_set_of_inequality_l124_12433

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set_of_inequality :
  {x : ℝ | f x ≥ x^2 - 8*x + 15} = {x : ℝ | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6} := by
  sorry

end solution_set_of_inequality_l124_12433


namespace factor_implies_sum_l124_12426

theorem factor_implies_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 - 2*X + 5) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) →
  P + Q = 31 :=
by sorry

end factor_implies_sum_l124_12426


namespace application_outcomes_count_l124_12476

/-- The number of colleges available for applications -/
def num_colleges : ℕ := 3

/-- The number of students applying to colleges -/
def num_students : ℕ := 3

/-- The total number of possible application outcomes when three students apply to three colleges,
    with the condition that the first two students must apply to different colleges -/
def total_outcomes : ℕ := num_colleges * (num_colleges - 1) * num_colleges

theorem application_outcomes_count : total_outcomes = 18 := by
  sorry

end application_outcomes_count_l124_12476


namespace increasing_function_inequality_l124_12404

theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum_positive : a + b > 0) : 
  f a + f b > f (-a) + f (-b) := by
sorry

end increasing_function_inequality_l124_12404


namespace carl_index_cards_cost_l124_12481

/-- Calculates the total cost of index cards for Carl's students --/
def total_cost_index_cards (
  cards_6th : ℕ)  -- Number of cards for each 6th grader
  (cards_7th : ℕ) -- Number of cards for each 7th grader
  (cards_8th : ℕ) -- Number of cards for each 8th grader
  (students_6th : ℕ) -- Number of 6th grade students per period
  (students_7th : ℕ) -- Number of 7th grade students per period
  (students_8th : ℕ) -- Number of 8th grade students per period
  (periods : ℕ) -- Number of periods per day
  (pack_size : ℕ) -- Number of cards per pack
  (cost_3x5 : ℕ) -- Cost of a pack of 3x5 cards in dollars
  (cost_4x6 : ℕ) -- Cost of a pack of 4x6 cards in dollars
  : ℕ :=
  let total_cards_6th := cards_6th * students_6th * periods
  let total_cards_7th := cards_7th * students_7th * periods
  let total_cards_8th := cards_8th * students_8th * periods
  let packs_6th := (total_cards_6th + pack_size - 1) / pack_size
  let packs_7th := (total_cards_7th + pack_size - 1) / pack_size
  let packs_8th := (total_cards_8th + pack_size - 1) / pack_size
  packs_6th * cost_3x5 + packs_7th * cost_3x5 + packs_8th * cost_4x6

theorem carl_index_cards_cost : 
  total_cost_index_cards 8 10 12 20 25 30 6 50 3 4 = 326 := by
  sorry

end carl_index_cards_cost_l124_12481


namespace dinner_lunch_cake_difference_l124_12474

theorem dinner_lunch_cake_difference : 
  let lunch_cakes : ℕ := 6
  let dinner_cakes : ℕ := 9
  dinner_cakes - lunch_cakes = 3 := by sorry

end dinner_lunch_cake_difference_l124_12474


namespace product_of_distinct_roots_l124_12418

theorem product_of_distinct_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h1 : x + 3 / x = y + 3 / y) (h2 : x + y = 4) : x * y = 3 := by
  sorry

end product_of_distinct_roots_l124_12418


namespace cubic_sum_minus_product_l124_12419

theorem cubic_sum_minus_product (x y z : ℝ) 
  (sum_eq : x + y + z = 12)
  (sum_product_eq : x * y + x * z + y * z = 30) :
  x^3 + y^3 + z^3 - 3*x*y*z = 648 := by
sorry

end cubic_sum_minus_product_l124_12419


namespace prob_both_odd_bounds_l124_12403

def range_start : ℕ := 1
def range_end : ℕ := 1000

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd : ℕ := (range_end - range_start + 1) / 2

def prob_first_odd : ℚ := count_odd / range_end

def prob_second_odd : ℚ := (count_odd - 1) / (range_end - 1)

def prob_both_odd : ℚ := prob_first_odd * prob_second_odd

theorem prob_both_odd_bounds : 1/6 < prob_both_odd ∧ prob_both_odd < 1/3 := by
  sorry

end prob_both_odd_bounds_l124_12403


namespace base7_divisibility_l124_12469

/-- Converts a base-7 number of the form 3dd6_7 to base 10 -/
def base7ToBase10 (d : ℕ) : ℕ := 3 * 7^3 + d * 7^2 + d * 7 + 6

/-- Checks if a number is a valid base-7 digit -/
def isValidBase7Digit (d : ℕ) : Prop := d ≤ 6

theorem base7_divisibility :
  ∀ d : ℕ, isValidBase7Digit d → (base7ToBase10 d % 13 = 0 ↔ d = 4) :=
by sorry

end base7_divisibility_l124_12469


namespace triangle_angle_bisector_length_l124_12400

noncomputable def angleBisectorLength (PQ PR : ℝ) (cosP : ℝ) : ℝ :=
  let QR := Real.sqrt (PQ^2 + PR^2 - 2 * PQ * PR * cosP)
  let cosHalfP := Real.sqrt ((1 + cosP) / 2)
  let QT := (5 * Real.sqrt 73) / 13
  Real.sqrt (PQ^2 + QT^2 - 2 * PQ * QT * cosHalfP)

theorem triangle_angle_bisector_length :
  ∀ (ε : ℝ), ε > 0 → 
  |angleBisectorLength 5 8 (1/5) - 5.05| < ε :=
sorry

end triangle_angle_bisector_length_l124_12400


namespace johnson_family_seating_l124_12448

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def boys : ℕ := 5
def girls : ℕ := 4
def total_children : ℕ := boys + girls

def block_arrangements : ℕ := 7 * (factorial boys) * (factorial (total_children - 3))

theorem johnson_family_seating :
  factorial total_children - block_arrangements = 60480 := by
  sorry

end johnson_family_seating_l124_12448


namespace opposite_of_negative_one_third_l124_12465

theorem opposite_of_negative_one_third :
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end opposite_of_negative_one_third_l124_12465


namespace sum_div_four_l124_12458

theorem sum_div_four : (4 + 44 + 444) / 4 = 123 := by
  sorry

end sum_div_four_l124_12458


namespace imon_disentanglement_l124_12434

-- Define a graph structure to represent imons and their entanglements
structure ImonGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)

-- Define the two operations
def destroyOddDegreeImon (g : ImonGraph) (v : Nat) : ImonGraph :=
  sorry

def doubleImonFamily (g : ImonGraph) : ImonGraph :=
  sorry

-- Define a predicate to check if a graph has no entanglements
def noEntanglements (g : ImonGraph) : Prop :=
  ∀ v ∈ g.vertices, ∀ w ∈ g.vertices, v ≠ w → (v, w) ∉ g.edges

-- Main theorem
theorem imon_disentanglement (g : ImonGraph) :
  ∃ (ops : List (ImonGraph → ImonGraph)), noEntanglements ((ops.foldl (· ∘ ·) id) g) :=
  sorry

end imon_disentanglement_l124_12434


namespace base_of_first_term_l124_12493

theorem base_of_first_term (x s : ℝ) (h : (x^16) * (25^s) = 5 * (10^16)) : x = 2/5 := by
  sorry

end base_of_first_term_l124_12493


namespace last_two_digits_product_l124_12422

theorem last_two_digits_product (n : ℤ) : 
  (∃ k : ℤ, n = 8 * k) → -- n is divisible by 8
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n % 100 = 10 * a + b ∧ a + b = 15) → -- last two digits sum to 15
  (n % 10) * ((n / 10) % 10) = 54 := by
sorry

end last_two_digits_product_l124_12422


namespace a_8_value_l124_12442

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  ∀ m n : ℕ+, a (m * n) = a m * a n

theorem a_8_value (a : ℕ+ → ℝ) (h_prop : sequence_property a) (h_a2 : a 2 = 3) :
  a 8 = 27 := by
  sorry

end a_8_value_l124_12442


namespace open_box_volume_l124_12482

/-- The volume of an open box formed by cutting squares from a rectangular sheet -/
theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_length : ℝ)
  (h_length : sheet_length = 100)
  (h_width : sheet_width = 50)
  (h_cut : cut_length = 10) :
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 24000 := by
sorry


end open_box_volume_l124_12482


namespace inequality_equivalence_l124_12477

theorem inequality_equivalence (x : ℝ) : 
  (2 * x + 3) / (x^2 - 2 * x + 4) > (4 * x + 5) / (2 * x^2 + 5 * x + 7) ↔ 
  x > (-23 - Real.sqrt 453) / 38 ∧ x < (-23 + Real.sqrt 453) / 38 :=
sorry

end inequality_equivalence_l124_12477


namespace reflection_matrix_iff_l124_12435

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, b],
    ![-3/4, 1/4]]

theorem reflection_matrix_iff (a b : ℚ) :
  (reflection_matrix a b)^2 = 1 ↔ a = -1/4 ∧ b = -5/4 := by
  sorry

end reflection_matrix_iff_l124_12435


namespace unique_a_value_l124_12480

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_value (a : ℝ) (h : 1 ∈ A a) : a = 0 := by
  sorry

end unique_a_value_l124_12480


namespace final_jellybeans_count_l124_12472

-- Define the initial number of jellybeans
def initial_jellybeans : ℕ := 90

-- Define the number of jellybeans Samantha took
def samantha_took : ℕ := 24

-- Define the number of jellybeans Shelby ate
def shelby_ate : ℕ := 12

-- Define the function to calculate the final number of jellybeans
def final_jellybeans : ℕ :=
  initial_jellybeans - (samantha_took + shelby_ate) + (samantha_took + shelby_ate) / 2

-- Theorem statement
theorem final_jellybeans_count : final_jellybeans = 72 := by
  sorry

end final_jellybeans_count_l124_12472


namespace nina_jerome_age_ratio_l124_12411

-- Define the ages as natural numbers
def leonard : ℕ := 6
def nina : ℕ := leonard + 4
def jerome : ℕ := 36 - nina - leonard

-- Theorem statement
theorem nina_jerome_age_ratio :
  nina * 2 = jerome :=
sorry

end nina_jerome_age_ratio_l124_12411


namespace highway_extension_proof_l124_12427

def highway_extension (current_length final_length first_day_miles : ℕ) : Prop :=
  let second_day_miles := 3 * first_day_miles
  let total_built := first_day_miles + second_day_miles
  let total_extension := final_length - current_length
  let remaining_miles := total_extension - total_built
  remaining_miles = 250

theorem highway_extension_proof :
  highway_extension 200 650 50 :=
sorry

end highway_extension_proof_l124_12427


namespace intersection_and_length_l124_12420

-- Define the coordinate system
variable (O : ℝ × ℝ)
variable (A : ℝ × ℝ)
variable (B : ℝ × ℝ)

-- Define lines l₁ and l₂
def l₁ (p : ℝ × ℝ) : Prop := p.1 + p.2 = 4
def l₂ (p : ℝ × ℝ) : Prop := p.2 = 2 * p.1

-- Define the conditions
axiom O_origin : O = (0, 0)
axiom A_on_l₁ : l₁ A
axiom A_on_l₂ : l₂ A
axiom B_on_l₁ : l₁ B
axiom OA_perp_OB : (A.1 * B.1 + A.2 * B.2) = 0

-- State the theorem
theorem intersection_and_length :
  A = (4/3, 8/3) ∧
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 20 * Real.sqrt 2 / 3 :=
sorry

end intersection_and_length_l124_12420


namespace dividend_percentage_l124_12423

theorem dividend_percentage 
  (face_value : ℝ) 
  (purchase_price : ℝ) 
  (return_on_investment : ℝ) 
  (h1 : face_value = 50) 
  (h2 : purchase_price = 31) 
  (h3 : return_on_investment = 0.25) : 
  (return_on_investment * purchase_price) / face_value * 100 = 15.5 := by
sorry

end dividend_percentage_l124_12423


namespace largest_number_l124_12444

theorem largest_number (a b c d e : ℝ) : 
  a = 24680 + 1 / 1357 →
  b = 24680 - 1 / 1357 →
  c = 24680 * (1 / 1357) →
  d = 24680 / (1 / 1357) →
  e = 24680.1357 →
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end largest_number_l124_12444


namespace complex_number_problem_l124_12421

theorem complex_number_problem (a : ℝ) : 
  let z : ℂ := (a / Complex.I) + ((1 - Complex.I) / 2) * Complex.I
  (z.re = 0 ∨ z.im = 0) ∧ (z.re + z.im = 0) → a = 0 := by
  sorry

end complex_number_problem_l124_12421


namespace tom_seashells_count_l124_12462

/-- The number of seashells Tom and Fred found together -/
def total_seashells : ℕ := 58

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := total_seashells - fred_seashells

theorem tom_seashells_count : tom_seashells = 15 := by
  sorry

end tom_seashells_count_l124_12462


namespace second_shift_widget_fraction_l124_12417

/-- The fraction of total widgets produced by the second shift in a factory --/
theorem second_shift_widget_fraction :
  -- Define the relative productivity of second shift compared to first shift
  ∀ (second_shift_productivity : ℚ)
  -- Define the relative number of employees in first shift compared to second shift
  (first_shift_employees : ℚ),
  -- Condition: Second shift productivity is 2/3 of first shift
  second_shift_productivity = 2 / 3 →
  -- Condition: First shift has 3/4 as many employees as second shift
  first_shift_employees = 3 / 4 →
  -- Conclusion: The fraction of total widgets produced by second shift is 8/17
  (second_shift_productivity * (1 / first_shift_employees)) /
  (1 + second_shift_productivity * (1 / first_shift_employees)) = 8 / 17 := by
sorry

end second_shift_widget_fraction_l124_12417


namespace sum_of_repeating_decimals_l124_12461

/-- Represents a repeating decimal with a single-digit repetend -/
def SingleDigitRepeatDecimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with a two-digit repetend -/
def TwoDigitRepeatDecimal (n : ℕ) : ℚ := n / 99

/-- Represents a repeating decimal with a three-digit repetend -/
def ThreeDigitRepeatDecimal (n : ℕ) : ℚ := n / 999

/-- The sum of 0.1̅, 0.02̅, and 0.003̅ is equal to 164/1221 -/
theorem sum_of_repeating_decimals :
  SingleDigitRepeatDecimal 1 + TwoDigitRepeatDecimal 2 + ThreeDigitRepeatDecimal 3 = 164 / 1221 := by
  sorry

end sum_of_repeating_decimals_l124_12461


namespace max_police_officers_l124_12450

theorem max_police_officers (n : ℕ) (h : n = 8) : 
  (n * (n - 1)) / 2 = 28 :=
sorry

end max_police_officers_l124_12450


namespace xyz_value_l124_12473

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 21) :
  x * y * z = 28 / 3 := by
sorry

end xyz_value_l124_12473


namespace decoration_time_is_five_hours_l124_12487

/-- Represents the number of eggs Mia can decorate per hour -/
def mia_eggs_per_hour : ℕ := 24

/-- Represents the number of eggs Billy can decorate per hour -/
def billy_eggs_per_hour : ℕ := 10

/-- Represents the total number of eggs that need to be decorated -/
def total_eggs : ℕ := 170

/-- Calculates the time taken to decorate all eggs when Mia and Billy work together -/
def decoration_time : ℚ :=
  total_eggs / (mia_eggs_per_hour + billy_eggs_per_hour : ℚ)

/-- Theorem stating that the decoration time is 5 hours -/
theorem decoration_time_is_five_hours :
  decoration_time = 5 := by sorry

end decoration_time_is_five_hours_l124_12487


namespace round_robin_tournament_l124_12470

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 15) : n = 6 := by
  sorry

end round_robin_tournament_l124_12470


namespace tan_negative_two_fraction_l124_12441

theorem tan_negative_two_fraction (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := by
  sorry

end tan_negative_two_fraction_l124_12441


namespace log_sum_equality_l124_12459

theorem log_sum_equality (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq1 : q ≠ 1) :
  Real.log p + Real.log q = Real.log (p + q) ↔ p = q / (q - 1) :=
sorry

end log_sum_equality_l124_12459


namespace stratified_sample_size_l124_12454

/-- Represents a company with employees -/
structure Company where
  total_employees : ℕ
  male_employees : ℕ
  female_employees : ℕ

/-- Represents a sample drawn from the company -/
structure Sample where
  female_count : ℕ
  male_count : ℕ

/-- Calculates the sample size given a company and a sample -/
def sample_size (c : Company) (s : Sample) : ℕ :=
  s.female_count + s.male_count

/-- Theorem stating that for a company with 120 employees, of which 90 are male,
    if a stratified sample by gender contains 3 female employees,
    then the total sample size is 12 -/
theorem stratified_sample_size 
  (c : Company) 
  (s : Sample) 
  (h1 : c.total_employees = 120) 
  (h2 : c.male_employees = 90) 
  (h3 : c.female_employees = c.total_employees - c.male_employees) 
  (h4 : s.female_count = 3) :
  sample_size c s = 12 := by
  sorry

#check stratified_sample_size

end stratified_sample_size_l124_12454


namespace harpers_daughter_has_four_teachers_l124_12430

/-- Represents the problem of finding Harper's daughter's teachers -/
def harpers_daughter_teachers (total_spent : ℕ) (gift_cost : ℕ) (sons_teachers : ℕ) : ℕ :=
  total_spent / gift_cost - sons_teachers

/-- Theorem stating that Harper's daughter has 4 teachers -/
theorem harpers_daughter_has_four_teachers :
  harpers_daughter_teachers 70 10 3 = 4 := by
  sorry

end harpers_daughter_has_four_teachers_l124_12430


namespace darry_full_ladder_steps_l124_12486

/-- The number of times Darry climbs his full ladder -/
def full_ladder_climbs : ℕ := 10

/-- The number of steps in Darry's smaller ladder -/
def small_ladder_steps : ℕ := 6

/-- The number of times Darry climbs his smaller ladder -/
def small_ladder_climbs : ℕ := 7

/-- The total number of steps Darry climbed -/
def total_steps : ℕ := 152

/-- The number of steps in Darry's full ladder -/
def full_ladder_steps : ℕ := 11

theorem darry_full_ladder_steps :
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs = total_steps :=
by sorry

end darry_full_ladder_steps_l124_12486


namespace halfway_between_fractions_l124_12406

theorem halfway_between_fractions :
  let a : ℚ := 1/8
  let b : ℚ := 1/3
  (a + b) / 2 = 11/48 := by
sorry

end halfway_between_fractions_l124_12406


namespace algebraic_expression_equality_l124_12499

theorem algebraic_expression_equality (x y : ℝ) (h : 2 * x - 3 * y = 1) :
  6 * y - 4 * x + 8 = 6 := by
  sorry

end algebraic_expression_equality_l124_12499


namespace function_properties_l124_12451

noncomputable def f (x : ℝ) (φ : ℝ) := Real.cos (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2)
  (h3 : ∀ x, f x φ = f (-Real.pi/3 - x) φ) :
  (f (Real.pi/6) φ = -1/2) ∧ 
  (∃! x, x ∈ Set.Ioo (-Real.pi/2) (Real.pi/2) ∧ 
    ∀ y ∈ Set.Ioo (-Real.pi/2) (Real.pi/2), f y φ ≤ f x φ) := by
  sorry

end function_properties_l124_12451


namespace simplify_complex_expression_l124_12445

theorem simplify_complex_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (((a^4 * b^2 * c)^(5/3) * (a^3 * b^2 * c)^2)^3)^(1/11) / a^(5/11) = a^3 * b^2 * c :=
by sorry

end simplify_complex_expression_l124_12445


namespace unique_quadratic_solution_l124_12439

theorem unique_quadratic_solution (a : ℝ) :
  (∃! x, a * x^2 + 2 * x - 1 = 0) → a = 0 ∨ a = -1 := by
  sorry

end unique_quadratic_solution_l124_12439


namespace boys_equation_holds_l124_12466

structure School where
  name : String
  total_students : ℕ

def calculate_boys (s : School) : ℚ :=
  s.total_students / (1 + s.total_students / 100)

theorem boys_equation_holds (s : School) :
  let x := calculate_boys s
  x + (x/100) * s.total_students = s.total_students :=
by sorry

def school_A : School := ⟨"A", 900⟩
def school_B : School := ⟨"B", 1200⟩
def school_C : School := ⟨"C", 1500⟩

#eval calculate_boys school_A
#eval calculate_boys school_B
#eval calculate_boys school_C

end boys_equation_holds_l124_12466


namespace solution_set_equality_l124_12437

theorem solution_set_equality : 
  {x : ℝ | (x + 3) * (1 - x) ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 1} := by sorry

end solution_set_equality_l124_12437


namespace cade_initial_marbles_l124_12416

/-- The number of marbles Cade gave away -/
def marbles_given : ℕ := 8

/-- The number of marbles Cade has left -/
def marbles_left : ℕ := 79

/-- The initial number of marbles Cade had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem cade_initial_marbles : initial_marbles = 87 := by
  sorry

end cade_initial_marbles_l124_12416


namespace final_cucumber_count_l124_12498

theorem final_cucumber_count (initial_total : ℕ) (initial_carrots : ℕ) (added_cucumbers : ℕ)
  (h1 : initial_total = 10)
  (h2 : initial_carrots = 4)
  (h3 : added_cucumbers = 2) :
  initial_total - initial_carrots + added_cucumbers = 8 := by
  sorry

end final_cucumber_count_l124_12498


namespace trajectory_equation_l124_12408

/-- Given two points A and B symmetric about the origin, and a moving point P such that 
    the product of the slopes of AP and BP is -1/3, prove that the trajectory of P 
    is described by the equation x^2 + 3y^2 = 4, where x ≠ ±1 -/
theorem trajectory_equation (A B P : ℝ × ℝ) : 
  A = (-1, 1) →
  B = (1, -1) →
  (∀ x y, P = (x, y) → x ≠ 1 ∧ x ≠ -1 →
    ((y - 1) / (x + 1)) * ((y + 1) / (x - 1)) = -1/3) →
  ∃ x y, P = (x, y) ∧ x^2 + 3*y^2 = 4 ∧ x ≠ 1 ∧ x ≠ -1 :=
by sorry


end trajectory_equation_l124_12408


namespace digit_81_of_325_over_999_l124_12483

theorem digit_81_of_325_over_999 (n : ℕ) (h : n = 81) :
  (325 : ℚ) / 999 * 10^n % 10 = 5 := by
  sorry

end digit_81_of_325_over_999_l124_12483


namespace wages_problem_l124_12453

/-- Represents the wages of a group of people -/
structure Wages where
  men : ℕ
  women : ℕ
  boys : ℕ
  menWage : ℚ
  womenWage : ℚ
  boysWage : ℚ

/-- The problem statement -/
theorem wages_problem (w : Wages) (h1 : w.men = 5) (h2 : w.boys = 8) 
    (h3 : w.men * w.menWage = w.women * w.womenWage) 
    (h4 : w.women * w.womenWage = w.boys * w.boysWage)
    (h5 : w.men * w.menWage + w.women * w.womenWage + w.boys * w.boysWage = 60) :
  w.men * w.menWage = 30 := by
  sorry

end wages_problem_l124_12453


namespace square_carpet_side_length_l124_12457

theorem square_carpet_side_length (area : ℝ) (h : area = 10) :
  ∃ (side : ℝ), side * side = area ∧ 3 < side ∧ side < 4 := by
  sorry

end square_carpet_side_length_l124_12457


namespace triangle_ratio_theorem_l124_12467

/-- Represents a point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a point lies on a line segment between two other points -/
def lies_on (P Q R : Point) : Prop := sorry

/-- Checks if two line segments intersect -/
def intersect (P Q R S : Point) : Prop := sorry

/-- Represents the ratio of distances between points -/
def distance_ratio (P Q R S : Point) : ℚ := sorry

theorem triangle_ratio_theorem (ABC : Triangle) (D E T : Point) :
  lies_on D ABC.B ABC.C →
  lies_on E ABC.A ABC.B →
  intersect ABC.A D ABC.B E →
  (∃ (t : Point), intersect ABC.A D ABC.B E ∧ t = T) →
  distance_ratio ABC.A T T D = 2 →
  distance_ratio ABC.B T T E = 3 →
  distance_ratio ABC.C D D ABC.B = 2/9 := by
  sorry

end triangle_ratio_theorem_l124_12467


namespace function_intersection_condition_l124_12490

/-- The function f(x) = (k+1)x^2 - 2x + 1 has intersections with the x-axis
    if and only if k ≤ 0. -/
theorem function_intersection_condition (k : ℝ) :
  (∃ x, (k + 1) * x^2 - 2 * x + 1 = 0) ↔ k ≤ 0 := by
  sorry

end function_intersection_condition_l124_12490


namespace pat_kate_ratio_l124_12413

/-- Represents the hours charged by each person -/
structure ProjectHours where
  pat : ℝ
  kate : ℝ
  mark : ℝ

/-- Defines the conditions of the problem -/
def satisfiesConditions (h : ProjectHours) : Prop :=
  h.pat + h.kate + h.mark = 135 ∧
  ∃ r : ℝ, h.pat = r * h.kate ∧
  h.pat = (1/3) * h.mark ∧
  h.mark = h.kate + 75

/-- The main theorem to prove -/
theorem pat_kate_ratio (h : ProjectHours) 
  (hcond : satisfiesConditions h) : h.pat / h.kate = 2 := by
  sorry

end pat_kate_ratio_l124_12413


namespace expected_value_of_sum_l124_12412

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def combinations : Finset (Finset ℕ) :=
  marbles.powerset.filter (λ s => s.card = 3)

def sum_of_combination (c : Finset ℕ) : ℕ := c.sum id

def total_sum : ℕ := combinations.sum sum_of_combination

def num_combinations : ℕ := combinations.card

theorem expected_value_of_sum :
  (total_sum : ℚ) / num_combinations = 21/2 := by sorry

end expected_value_of_sum_l124_12412


namespace fraction_equals_sixtythree_l124_12431

theorem fraction_equals_sixtythree : (2200 - 2089)^2 / 196 = 63 := by
  sorry

end fraction_equals_sixtythree_l124_12431


namespace cat_toy_cost_l124_12463

def initial_amount : ℚ := 1173 / 100
def amount_left : ℚ := 151 / 100

theorem cat_toy_cost : initial_amount - amount_left = 1022 / 100 := by
  sorry

end cat_toy_cost_l124_12463


namespace complex_fraction_equals_i_l124_12414

theorem complex_fraction_equals_i : (Complex.I + 1) / (1 - Complex.I) = Complex.I := by
  sorry

end complex_fraction_equals_i_l124_12414


namespace function_equality_implies_m_range_l124_12475

theorem function_equality_implies_m_range :
  ∀ (m : ℝ), m > 0 →
  (∃ (x₁ : ℝ), x₁ ∈ Set.Icc 0 3 ∧
    (∀ (x₂ : ℝ), x₂ ∈ Set.Icc 0 3 →
      x₁^2 - 4*x₁ + 3 = m*(x₂ - 1) + 2)) →
  m ∈ Set.Ioo 0 (1/2) := by
sorry

end function_equality_implies_m_range_l124_12475


namespace negation_of_universal_proposition_l124_12425

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end negation_of_universal_proposition_l124_12425


namespace x_range_given_inequality_l124_12460

theorem x_range_given_inequality (a : ℝ) (h_a : a ∈ Set.Icc (-1) 1) :
  (∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0) →
  {x : ℝ | x < 1 ∨ x > 3}.Nonempty :=
by sorry

end x_range_given_inequality_l124_12460


namespace notebook_spending_l124_12440

/-- Calculates the total amount spent on notebooks --/
def total_spent (total_notebooks : ℕ) (red_notebooks : ℕ) (green_notebooks : ℕ) 
  (red_price : ℕ) (green_price : ℕ) (blue_price : ℕ) : ℕ :=
  let blue_notebooks := total_notebooks - red_notebooks - green_notebooks
  red_notebooks * red_price + green_notebooks * green_price + blue_notebooks * blue_price

/-- Proves that the total amount spent on notebooks is $37 --/
theorem notebook_spending : 
  total_spent 12 3 2 4 2 3 = 37 := by
  sorry

end notebook_spending_l124_12440


namespace max_value_fraction_l124_12485

theorem max_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * y / (4 * x + 9 * y) ≤ a * b / (4 * a + 9 * b)) → 
  a * b / (4 * a + 9 * b) = 1 / 25 :=
by sorry

end max_value_fraction_l124_12485


namespace not_sum_of_three_cubes_l124_12424

theorem not_sum_of_three_cubes : ¬ ∃ (x y z : ℤ), x^3 + y^3 + z^3 = 20042005 := by
  sorry

end not_sum_of_three_cubes_l124_12424


namespace quadratic_factorization_l124_12402

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) 
  (h2 : ∀ x : ℝ, x^2 - 18*x + 77 = (x - a)*(x - b)) : 
  3*b - a = 10 := by
sorry

end quadratic_factorization_l124_12402


namespace max_sum_distances_in_unit_square_l124_12496

theorem max_sum_distances_in_unit_square :
  ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
  let PA := Real.sqrt (x^2 + y^2)
  let PB := Real.sqrt ((1-x)^2 + y^2)
  let PC := Real.sqrt ((1-x)^2 + (1-y)^2)
  let PD := Real.sqrt (x^2 + (1-y)^2)
  PA + PB + PC + PD ≤ 2 + Real.sqrt 2 ∧
  ∃ (x₀ y₀ : ℝ), 0 ≤ x₀ ∧ x₀ ≤ 1 ∧ 0 ≤ y₀ ∧ y₀ ≤ 1 ∧
    let PA₀ := Real.sqrt (x₀^2 + y₀^2)
    let PB₀ := Real.sqrt ((1-x₀)^2 + y₀^2)
    let PC₀ := Real.sqrt ((1-x₀)^2 + (1-y₀)^2)
    let PD₀ := Real.sqrt (x₀^2 + (1-y₀)^2)
    PA₀ + PB₀ + PC₀ + PD₀ = 2 + Real.sqrt 2 :=
by
  sorry

end max_sum_distances_in_unit_square_l124_12496


namespace average_problem_l124_12492

theorem average_problem (a b c X Y Z : ℝ) 
  (h1 : (a + b + c) / 3 = 5)
  (h2 : (X + Y + Z) / 3 = 7) :
  ((2*a + 3*X) + (2*b + 3*Y) + (2*c + 3*Z)) / 3 = 31 := by
  sorry

end average_problem_l124_12492


namespace soccer_league_teams_l124_12452

theorem soccer_league_teams (total_games : ℕ) (h_games : total_games = 45) : 
  ∃ (n : ℕ), n * (n - 1) / 2 = total_games ∧ n = 10 :=
by sorry

end soccer_league_teams_l124_12452


namespace apple_basket_problem_l124_12449

theorem apple_basket_problem (x : ℕ) : 
  (x * 22 = (x + 45) * (22 - 9)) → x * 22 = 1430 := by
sorry

end apple_basket_problem_l124_12449


namespace m_range_l124_12484

def f (x : ℝ) : ℝ := x^3 + x

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m < 1 :=
by sorry

end m_range_l124_12484


namespace intersection_of_A_and_B_l124_12428

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l124_12428


namespace r_geq_one_l124_12489

noncomputable section

variables (m : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := Real.exp x

def g (m : ℝ) (x : ℝ) : ℝ := m * x + 4 * m

def r (m : ℝ) (x : ℝ) : ℝ := 1 / f x + (4 * m * x) / g m x

theorem r_geq_one (h1 : m > 0) (h2 : x ≥ 0) : r m x ≥ 1 := by
  sorry

end

end r_geq_one_l124_12489


namespace triangle_ratio_theorem_l124_12478

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: In a triangle ABC, if 2b * sin(2A) = a * sin(B) and c = 2b, then a/b = 2 -/
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : 2 * t.b * Real.sin (2 * t.A) = t.a * Real.sin t.B)
  (h2 : t.c = 2 * t.b) :
  t.a / t.b = 2 := by
  sorry

end triangle_ratio_theorem_l124_12478
