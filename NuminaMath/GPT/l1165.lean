import Mathlib

namespace NUMINAMATH_GPT_arithmetic_progression_25th_term_l1165_116578

def arithmetic_progression_nth_term (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_progression_25th_term : arithmetic_progression_nth_term 5 7 25 = 173 := by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_25th_term_l1165_116578


namespace NUMINAMATH_GPT_value_of_angle_C_perimeter_range_l1165_116569

-- Part (1): Prove angle C value
theorem value_of_angle_C
  {a b c : ℝ} {A B C : ℝ}
  (acute_ABC : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (m : ℝ × ℝ := (Real.sin C, Real.cos C))
  (n : ℝ × ℝ := (2 * Real.sin A - Real.cos B, -Real.sin B))
  (orthogonal_mn : m.1 * n.1 + m.2 * n.2 = 0) 
  : C = π / 6 := sorry

-- Part (2): Prove perimeter range
theorem perimeter_range
  {a b c : ℝ} {A B C : ℝ}
  (A_range : π / 3 < A ∧ A < π / 2)
  (C_value : C = π / 6)
  (a_value : a = 2)
  (acute_ABC : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  : 3 + 2 * Real.sqrt 3 < a + b + c ∧ a + b + c < 2 + 3 * Real.sqrt 3 := sorry

end NUMINAMATH_GPT_value_of_angle_C_perimeter_range_l1165_116569


namespace NUMINAMATH_GPT_number_of_roses_l1165_116560

def total_flowers : ℕ := 10
def carnations : ℕ := 5
def roses : ℕ := total_flowers - carnations

theorem number_of_roses : roses = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_roses_l1165_116560


namespace NUMINAMATH_GPT_general_term_of_sequence_l1165_116597

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else
  if n = 2 then 2 else
  sorry -- the recurrence relation will go here, but we'll skip its implementation

theorem general_term_of_sequence :
  ∀ n : ℕ, n ≥ 1 → a n = 3 - (2 / n) :=
by sorry

end NUMINAMATH_GPT_general_term_of_sequence_l1165_116597


namespace NUMINAMATH_GPT_mary_needs_more_cups_l1165_116588

theorem mary_needs_more_cups (total_cups required_cups added_cups : ℕ) (h1 : required_cups = 8) (h2 : added_cups = 2) : total_cups = 6 :=
by
  sorry

end NUMINAMATH_GPT_mary_needs_more_cups_l1165_116588


namespace NUMINAMATH_GPT_cocoa_powder_total_l1165_116553

variable (already_has : ℕ) (still_needs : ℕ)

theorem cocoa_powder_total (h₁ : already_has = 259) (h₂ : still_needs = 47) : already_has + still_needs = 306 :=
by
  sorry

end NUMINAMATH_GPT_cocoa_powder_total_l1165_116553


namespace NUMINAMATH_GPT_shirt_to_pants_ratio_l1165_116513

noncomputable def cost_uniforms
  (pants_cost shirt_ratio socks_price total_spending : ℕ) : Prop :=
  ∃ (shirt_cost tie_cost : ℕ),
    shirt_cost = shirt_ratio * pants_cost ∧
    tie_cost = shirt_cost / 5 ∧
    5 * (pants_cost + shirt_cost + tie_cost + socks_price) = total_spending

theorem shirt_to_pants_ratio 
  (pants_cost socks_price total_spending : ℕ)
  (h1 : pants_cost = 20)
  (h2 : socks_price = 3)
  (h3 : total_spending = 355)
  (shirt_ratio : ℕ)
  (h4 : cost_uniforms pants_cost shirt_ratio socks_price total_spending) :
  shirt_ratio = 2 := by
  sorry

end NUMINAMATH_GPT_shirt_to_pants_ratio_l1165_116513


namespace NUMINAMATH_GPT_ages_of_people_l1165_116536

-- Define types
variable (A M B C : ℕ)

-- Define conditions as hypotheses
def conditions : Prop :=
  A = 2 * M ∧
  A = 4 * B ∧
  M = A - 10 ∧
  C = B + 3 ∧
  C = M / 2

-- Define what we want to prove
theorem ages_of_people :
  (conditions A M B C) →
  A = 20 ∧
  M = 10 ∧
  B = 2 ∧
  C = 5 :=
by
  sorry

end NUMINAMATH_GPT_ages_of_people_l1165_116536


namespace NUMINAMATH_GPT_tournament_players_l1165_116564

theorem tournament_players (n : ℕ) :
  (∃ k : ℕ, k = n + 12 ∧
    -- Exactly one-third of the points earned by each player were earned against the twelve players with the least number of points.
    (2 * (1 / 3 * (n * (n - 1) / 2)) + 2 / 3 * 66 + 66 = (k * (k - 1)) / 2) ∧
    --- Solving the quadratic equation derived
    (n = 4)) → 
    k = 16 :=
by
  sorry

end NUMINAMATH_GPT_tournament_players_l1165_116564


namespace NUMINAMATH_GPT_minimum_value_of_fraction_l1165_116524

theorem minimum_value_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (4 / a + 9 / b) ≥ 25 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_fraction_l1165_116524


namespace NUMINAMATH_GPT_pie_chart_degrees_for_cherry_pie_l1165_116517

theorem pie_chart_degrees_for_cherry_pie :
  ∀ (total_students chocolate_pie apple_pie blueberry_pie : ℕ)
    (remaining_students cherry_pie_students lemon_pie_students : ℕ),
    total_students = 40 →
    chocolate_pie = 15 →
    apple_pie = 10 →
    blueberry_pie = 7 →
    remaining_students = total_students - chocolate_pie - apple_pie - blueberry_pie →
    cherry_pie_students = remaining_students / 2 →
    lemon_pie_students = remaining_students / 2 →
    (cherry_pie_students : ℝ) / (total_students : ℝ) * 360 = 36 :=
by
  sorry

end NUMINAMATH_GPT_pie_chart_degrees_for_cherry_pie_l1165_116517


namespace NUMINAMATH_GPT_parallelogram_area_l1165_116557

-- Define a plane rectangular coordinate system
structure PlaneRectangularCoordinateSystem :=
(axis : ℝ)

-- Define the properties of a square
structure Square :=
(side_length : ℝ)

-- Define the properties of a parallelogram in a perspective drawing
structure Parallelogram :=
(side_length: ℝ)

-- Define the conditions of the problem
def problem_conditions (s : Square) (p : Parallelogram) :=
  s.side_length = 4 ∨ s.side_length = 8 ∧ 
  p.side_length = 4

-- Statement of the problem
theorem parallelogram_area (s : Square) (p : Parallelogram)
  (h : problem_conditions s p) :
  p.side_length * p.side_length = 16 ∨ p.side_length * p.side_length = 64 :=
by {
  sorry
}

end NUMINAMATH_GPT_parallelogram_area_l1165_116557


namespace NUMINAMATH_GPT_problem1_problem2_l1165_116502

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

-- Problem (1): Prove the inequality f(x-1) > 0 given b = 1.
theorem problem1 (a x : ℝ) : f (x - 1) a 1 > 0 := sorry

-- Problem (2): Prove the values of a and b such that the range of f(x) for x ∈ [-1, 2] is [5/4, 2].
theorem problem2 (a b : ℝ) (H₁ : f (-1) a b = 5 / 4) (H₂ : f 2 a b = 2) :
    (a = 3 ∧ b = 2) ∨ (a = -4 ∧ b = -3) := sorry

end NUMINAMATH_GPT_problem1_problem2_l1165_116502


namespace NUMINAMATH_GPT_spiderCanEatAllFlies_l1165_116570

-- Define the number of nodes in the grid.
def numNodes := 100

-- Define initial conditions.
def cornerStart := true
def numFlies := 100
def fliesAtNodes (nodes : ℕ) : Prop := nodes = numFlies

-- Define the predicate for whether the spider can eat all flies within a certain number of moves.
def canEatAllFliesWithinMoves (maxMoves : ℕ) : Prop :=
  ∃ (moves : ℕ), moves ≤ maxMoves

-- The theorem we need to prove in Lean 4.
theorem spiderCanEatAllFlies (h1 : cornerStart) (h2 : fliesAtNodes numFlies) : canEatAllFliesWithinMoves 2000 :=
by
  sorry

end NUMINAMATH_GPT_spiderCanEatAllFlies_l1165_116570


namespace NUMINAMATH_GPT_calculate_expression_l1165_116527

theorem calculate_expression :
  15^2 + 2 * 15 * 5 + 5^2 + 5^3 = 525 := 
sorry

end NUMINAMATH_GPT_calculate_expression_l1165_116527


namespace NUMINAMATH_GPT_steven_set_aside_9_grapes_l1165_116508

-- Define the conditions based on the problem statement
def total_seeds_needed : ℕ := 60
def average_seeds_per_apple : ℕ := 6
def average_seeds_per_pear : ℕ := 2
def average_seeds_per_grape : ℕ := 3
def apples_set_aside : ℕ := 4
def pears_set_aside : ℕ := 3
def additional_seeds_needed : ℕ := 3

-- Calculate the number of seeds from apples and pears
def seeds_from_apples : ℕ := apples_set_aside * average_seeds_per_apple
def seeds_from_pears : ℕ := pears_set_aside * average_seeds_per_pear

-- Calculate the number of seeds that Steven already has from apples and pears
def seeds_from_apples_and_pears : ℕ := seeds_from_apples + seeds_from_pears

-- Calculate the remaining seeds needed from grapes
def seeds_needed_from_grapes : ℕ := total_seeds_needed - seeds_from_apples_and_pears - additional_seeds_needed

-- Calculate the number of grapes set aside
def grapes_set_aside : ℕ := seeds_needed_from_grapes / average_seeds_per_grape

theorem steven_set_aside_9_grapes : grapes_set_aside = 9 :=
by 
  sorry

end NUMINAMATH_GPT_steven_set_aside_9_grapes_l1165_116508


namespace NUMINAMATH_GPT_fraction_of_sum_l1165_116587

theorem fraction_of_sum (numbers : List ℝ) (h_len : numbers.length = 21)
  (n : ℝ) (h_n : n ∈ numbers)
  (h_avg : n = 5 * ((numbers.sum - n) / 20)) :
  n / numbers.sum = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_sum_l1165_116587


namespace NUMINAMATH_GPT_train_speed_correct_l1165_116526

-- Definitions for the given conditions
def train_length : ℝ := 320
def time_to_cross : ℝ := 6

-- The speed of the train
def train_speed : ℝ := 53.33

-- The proof statement
theorem train_speed_correct : train_speed = train_length / time_to_cross :=
by
  sorry

end NUMINAMATH_GPT_train_speed_correct_l1165_116526


namespace NUMINAMATH_GPT_other_root_is_minus_5_l1165_116542

-- conditions
def polynomial (x : ℝ) := x^4 - x^3 - 18 * x^2 + 52 * x + (-40 : ℝ)
def r1 := 2
def f_of_r1_eq_zero : polynomial r1 = 0 := by sorry -- given condition

-- the proof problem
theorem other_root_is_minus_5 : ∃ r, polynomial r = 0 ∧ r ≠ r1 ∧ r = -5 :=
by
  sorry

end NUMINAMATH_GPT_other_root_is_minus_5_l1165_116542


namespace NUMINAMATH_GPT_percent_less_than_l1165_116520

-- Definitions based on the given conditions.
variable (y q w z : ℝ)
variable (h1 : w = 0.60 * q)
variable (h2 : q = 0.60 * y)
variable (h3 : z = 1.50 * w)

-- The theorem that the percentage by which z is less than y is 46%.
theorem percent_less_than (y q w z : ℝ) (h1 : w = 0.60 * q) (h2 : q = 0.60 * y) (h3 : z = 1.50 * w) :
  100 - (z / y * 100) = 46 :=
sorry

end NUMINAMATH_GPT_percent_less_than_l1165_116520


namespace NUMINAMATH_GPT_range_of_a_decreasing_function_l1165_116586

theorem range_of_a_decreasing_function (a : ℝ) :
  (∀ x < 1, ∀ y < x, (3 * a - 1) * x + 4 * a ≥ (3 * a - 1) * y + 4 * a) ∧ 
  (∀ x ≥ 1, ∀ y > x, -a * x ≤ -a * y) ∧
  (∀ x < 1, ∀ y ≥ 1, (3 * a - 1) * x + 4 * a ≥ -a * y)  →
  (1 / 8 : ℝ) ≤ a ∧ a < (1 / 3 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_of_a_decreasing_function_l1165_116586


namespace NUMINAMATH_GPT_breadth_of_landscape_l1165_116548

noncomputable def landscape_breadth (L : ℕ) (playground_area : ℕ) (total_area : ℕ) (B : ℕ) : Prop :=
  B = 6 * L ∧ playground_area = 4200 ∧ playground_area = (1 / 7) * total_area ∧ total_area = L * B

theorem breadth_of_landscape : ∃ (B : ℕ), ∀ (L : ℕ), landscape_breadth L 4200 29400 B → B = 420 :=
by
  intros
  sorry

end NUMINAMATH_GPT_breadth_of_landscape_l1165_116548


namespace NUMINAMATH_GPT_range_f_1_range_m_l1165_116530

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 - 2) * (Real.log x / (2 * Real.log 2) - 1/2)

theorem range_f_1 (x : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) : 
  -1/8 ≤ f x ∧ f x ≤ 0 :=
sorry

theorem range_m (m : ℝ) (x : ℝ) (h1 : 4 ≤ x) (h2 : x ≤ 16) (h3 : f x ≥ m * Real.log x / Real.log 2) :
  m ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_f_1_range_m_l1165_116530


namespace NUMINAMATH_GPT_factor_x4_plus_64_monic_real_l1165_116589

theorem factor_x4_plus_64_monic_real :
  ∀ x : ℝ, x^4 + 64 = (x^2 + 4 * x + 8) * (x^2 - 4 * x + 8) := 
by
  intros
  sorry

end NUMINAMATH_GPT_factor_x4_plus_64_monic_real_l1165_116589


namespace NUMINAMATH_GPT_max_crates_first_trip_l1165_116583

theorem max_crates_first_trip (x : ℕ) : (∀ w, w ≥ 120) ∧ (600 ≥ x * 120) → x = 5 := 
by
  -- Condition: The weight of any crate is no less than 120 kg
  intro h
  have h1 : ∀ w, w ≥ 120 := h.left
  
  -- Condition: The maximum weight for the first trip
  have h2 : 600 ≥ x * 120 := h.right 
  
  -- Derivation of maximum crates
  have h3 : x ≤ 600 / 120 := by sorry  -- This inequality follows from h2 by straightforward division
  
  have h4 : x ≤ 5 := by sorry  -- This follows from evaluating 600 / 120 = 5
  
  -- Knowing x is an integer and the maximum possible value is 5
  exact by sorry

end NUMINAMATH_GPT_max_crates_first_trip_l1165_116583


namespace NUMINAMATH_GPT_division_by_fraction_l1165_116528

theorem division_by_fraction :
  (5 / (8 / 15) : ℚ) = 75 / 8 :=
by
  sorry

end NUMINAMATH_GPT_division_by_fraction_l1165_116528


namespace NUMINAMATH_GPT_wall_length_is_7_5_meters_l1165_116507

noncomputable def brick_volume : ℚ := 25 * 11.25 * 6

noncomputable def total_brick_volume : ℚ := 6000 * brick_volume

noncomputable def wall_cross_section : ℚ := 600 * 22.5

noncomputable def wall_length (total_volume : ℚ) (cross_section : ℚ) : ℚ := total_volume / cross_section

theorem wall_length_is_7_5_meters :
  wall_length total_brick_volume wall_cross_section = 7.5 := by
sorry

end NUMINAMATH_GPT_wall_length_is_7_5_meters_l1165_116507


namespace NUMINAMATH_GPT_jog_time_each_morning_is_1_5_hours_l1165_116584

-- Define the total time Mr. John spent jogging
def total_time_spent_jogging : ℝ := 21

-- Define the number of days Mr. John jogged
def number_of_days_jogged : ℕ := 14

-- Define the time Mr. John jogs each morning
noncomputable def time_jogged_each_morning : ℝ := total_time_spent_jogging / number_of_days_jogged

-- State the theorem that the time jogged each morning is 1.5 hours
theorem jog_time_each_morning_is_1_5_hours : time_jogged_each_morning = 1.5 := by
  sorry

end NUMINAMATH_GPT_jog_time_each_morning_is_1_5_hours_l1165_116584


namespace NUMINAMATH_GPT_sufficient_condition_for_parallel_l1165_116551

-- Definitions for lines and planes
variables {Line Plane : Type}

-- Definitions of parallelism and perpendicularity
variable {Parallel Perpendicular : Line → Plane → Prop}
variable {ParallelLines : Line → Line → Prop}

-- Definition of subset relation
variable {Subset : Line → Plane → Prop}

-- Theorems or conditions
variables (a b : Line) (α β : Plane)

-- Assertion of the theorem
theorem sufficient_condition_for_parallel (h1 : ParallelLines a b) (h2 : Parallel b α) (h3 : ¬ Subset a α) : Parallel a α :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_parallel_l1165_116551


namespace NUMINAMATH_GPT_hallie_made_100_per_painting_l1165_116541

-- Define conditions
def num_paintings : ℕ := 3
def total_money_made : ℕ := 300

-- Define the goal
def money_per_painting : ℕ := total_money_made / num_paintings

theorem hallie_made_100_per_painting :
  money_per_painting = 100 :=
sorry

end NUMINAMATH_GPT_hallie_made_100_per_painting_l1165_116541


namespace NUMINAMATH_GPT_stuart_segments_return_l1165_116558

theorem stuart_segments_return (r1 r2 : ℝ) (tangent_chord : ℝ)
  (angle_ABC : ℝ) (h1 : r1 < r2) (h2 : tangent_chord = r1 * 2)
  (h3 : angle_ABC = 75) :
  ∃ (n : ℕ), n = 24 ∧ tangent_chord * n = 360 * (n / 24) :=
by {
  sorry
}

end NUMINAMATH_GPT_stuart_segments_return_l1165_116558


namespace NUMINAMATH_GPT_decimal_equivalent_one_half_pow_five_l1165_116581

theorem decimal_equivalent_one_half_pow_five :
  (1 / 2) ^ 5 = 0.03125 :=
by sorry

end NUMINAMATH_GPT_decimal_equivalent_one_half_pow_five_l1165_116581


namespace NUMINAMATH_GPT_last_digit_of_sum_is_four_l1165_116561

theorem last_digit_of_sum_is_four (x y z : ℕ)
  (hx : 1 ≤ x ∧ x ≤ 9)
  (hy : 0 ≤ y ∧ y ≤ 9)
  (hz : 0 ≤ z ∧ z ≤ 9)
  (h : 1950 ≤ 200 * x + 11 * y + 11 * z ∧ 200 * x + 11 * y + 11 * z < 2000) :
  (200 * x + 11 * y + 11 * z) % 10 = 4 :=
sorry

end NUMINAMATH_GPT_last_digit_of_sum_is_four_l1165_116561


namespace NUMINAMATH_GPT_length_of_GH_l1165_116547

theorem length_of_GH (AB CD GH : ℤ) (h_parallel : AB = 240 ∧ CD = 160 ∧ (AB + CD) = GH*2) : GH = 320 / 3 :=
by sorry

end NUMINAMATH_GPT_length_of_GH_l1165_116547


namespace NUMINAMATH_GPT_intersection_M_N_l1165_116549

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def N : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }

theorem intersection_M_N :
  M ∩ N = { z | 0 ≤ z ∧ z ≤ 1 } := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1165_116549


namespace NUMINAMATH_GPT_value_of_collection_l1165_116506

theorem value_of_collection (n : ℕ) (v : ℕ → ℕ) (h1 : n = 20) 
    (h2 : v 5 = 20) (h3 : ∀ k1 k2, v k1 = v k2) : v n = 80 :=
by
  sorry

end NUMINAMATH_GPT_value_of_collection_l1165_116506


namespace NUMINAMATH_GPT_original_number_of_motorcycles_l1165_116533

theorem original_number_of_motorcycles (x y : ℕ) 
  (h1 : x + 2 * y = 42) 
  (h2 : x > y) 
  (h3 : 2 * (x - 3) + 4 * y = 3 * (x + y - 3)) : x = 16 := 
sorry

end NUMINAMATH_GPT_original_number_of_motorcycles_l1165_116533


namespace NUMINAMATH_GPT_Q_eq_G_l1165_116571

def P := {y | ∃ x, y = x^2 + 1}
def Q := {y : ℝ | ∃ x, y = x^2 + 1}
def E := {x : ℝ | ∃ y, y = x^2 + 1}
def F := {(x, y) | y = x^2 + 1}
def G := {x : ℝ | x ≥ 1}

theorem Q_eq_G : Q = G := by
  sorry

end NUMINAMATH_GPT_Q_eq_G_l1165_116571


namespace NUMINAMATH_GPT_speed_of_mans_train_is_80_kmph_l1165_116505

-- Define the given constants
def length_goods_train : ℤ := 280 -- length in meters
def time_to_pass : ℤ := 9 -- time in seconds
def speed_goods_train : ℤ := 32 -- speed in km/h

-- Define the conversion factor from km/h to m/s
def kmh_to_ms (v : ℤ) : ℤ := v * 1000 / 3600

-- Define the speed of the goods train in m/s
def speed_goods_train_ms := kmh_to_ms speed_goods_train

-- Define the speed of the man's train in km/h
def speed_mans_train : ℤ := 80

-- Prove that the speed of the man's train is 80 km/h given the conditions
theorem speed_of_mans_train_is_80_kmph :
  ∃ V : ℤ,
    (V + speed_goods_train) * 1000 / 3600 = length_goods_train / time_to_pass → 
    V = speed_mans_train :=
by
  sorry

end NUMINAMATH_GPT_speed_of_mans_train_is_80_kmph_l1165_116505


namespace NUMINAMATH_GPT_ellipse_standard_equation_l1165_116591

theorem ellipse_standard_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (-4)^2 / a^2 + 3^2 / b^2 = 1) 
    (h4 : a^2 = b^2 + 5^2) : 
    ∃ (a b : ℝ), a^2 = 40 ∧ b^2 = 15 ∧ 
    (∀ x y : ℝ, x^2 / 40 + y^2 / 15 = 1 → (∃ f1 f2 : ℝ, f1 = 5 ∧ f2 = -5)) :=
by {
    sorry
}

end NUMINAMATH_GPT_ellipse_standard_equation_l1165_116591


namespace NUMINAMATH_GPT_Dana_Colin_relationship_l1165_116580

variable (C : ℝ) -- Let C be the number of cards Colin has.

def Ben_cards (C : ℝ) : ℝ := 1.20 * C -- Ben has 20% more cards than Colin
def Dana_cards (C : ℝ) : ℝ := 1.40 * Ben_cards C + Ben_cards C -- Dana has 40% more cards than Ben

theorem Dana_Colin_relationship : Dana_cards C = 1.68 * C := by
  sorry

end NUMINAMATH_GPT_Dana_Colin_relationship_l1165_116580


namespace NUMINAMATH_GPT_remainder_of_expression_l1165_116514

theorem remainder_of_expression :
  (8 * 7^19 + 1^19) % 9 = 3 :=
  by
    sorry

end NUMINAMATH_GPT_remainder_of_expression_l1165_116514


namespace NUMINAMATH_GPT_flower_counts_l1165_116555

theorem flower_counts (R G Y : ℕ) : (R + G = 62) → (R + Y = 49) → (G + Y = 77) → R = 17 ∧ G = 45 ∧ Y = 32 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_flower_counts_l1165_116555


namespace NUMINAMATH_GPT_xy_condition_l1165_116537

theorem xy_condition : (∀ x y : ℝ, x^2 + y^2 = 0 → xy = 0) ∧ ¬ (∀ x y : ℝ, xy = 0 → x^2 + y^2 = 0) := 
by
  sorry

end NUMINAMATH_GPT_xy_condition_l1165_116537


namespace NUMINAMATH_GPT_smallest_possible_obscured_number_l1165_116552

theorem smallest_possible_obscured_number (a b : ℕ) (cond : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) :
  2 * a = b - 9 →
  42 + 25 + 56 + 10 * a + b = 4 * (4 + 2 + 2 + 5 + 5 + 6 + a + b) →
  10 * a + b = 79 :=
sorry

end NUMINAMATH_GPT_smallest_possible_obscured_number_l1165_116552


namespace NUMINAMATH_GPT_min_xyz_value_l1165_116531

theorem min_xyz_value (x y z : ℝ) (h1 : x + y + z = 1) (h2 : z = 2 * y) (h3 : y ≤ (1 / 3)) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (∀ a b c : ℝ, (a + b + c = 1) → (c = 2 * b) → (b ≤ (1 / 3)) → 0 < a → 0 < b → 0 < c → (a * b * c) ≥ (x * y * z) → (a * b * c) = (8 / 243)) :=
by sorry

end NUMINAMATH_GPT_min_xyz_value_l1165_116531


namespace NUMINAMATH_GPT_plaza_area_increase_l1165_116516

theorem plaza_area_increase (a : ℝ) : 
  ((a + 2)^2 - a^2 = 4 * a + 4) :=
sorry

end NUMINAMATH_GPT_plaza_area_increase_l1165_116516


namespace NUMINAMATH_GPT_simplify_fraction_l1165_116510

theorem simplify_fraction (n : ℕ) (h : 2 ^ n ≠ 0) : 
  (2 ^ (n + 5) - 3 * 2 ^ n) / (3 * 2 ^ (n + 4)) = 29 / 48 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1165_116510


namespace NUMINAMATH_GPT_exists_group_of_four_l1165_116546

-- Define the given conditions
variables (students : Finset ℕ) (h_size : students.card = 21)
variables (done_homework : Finset ℕ → Prop)
variables (hw_unique : ∀ (s : Finset ℕ), s.card = 3 → done_homework s)

-- Define the theorem with the assertion to be proved
theorem exists_group_of_four (students : Finset ℕ) (h_size : students.card = 21)
  (done_homework : Finset ℕ → Prop)
  (hw_unique : ∀ s, s.card = 3 → done_homework s) :
  ∃ (grp : Finset ℕ), grp.card = 4 ∧ 
    (∀ (s : Finset ℕ), s ⊆ grp ∧ s.card = 3 → done_homework s) :=
sorry

end NUMINAMATH_GPT_exists_group_of_four_l1165_116546


namespace NUMINAMATH_GPT_divide_L_shaped_plaque_into_four_equal_parts_l1165_116544

-- Definition of an "L"-shaped plaque and the condition of symmetric cuts
def L_shaped_plaque (a b : ℕ) : Prop := (a > 0) ∧ (b > 0)

-- Statement of the proof problem
theorem divide_L_shaped_plaque_into_four_equal_parts (a b : ℕ) (h : L_shaped_plaque a b) :
  ∃ (p1 p2 : ℕ → ℕ → Prop),
    (∀ x y, p1 x y ↔ (x < a/2 ∧ y < b/2)) ∧
    (∀ x y, p2 x y ↔ (x < a/2 ∧ y >= b/2) ∨ (x >= a/2 ∧ y < b/2) ∨ (x >= a/2 ∧ y >= b/2)) :=
sorry

end NUMINAMATH_GPT_divide_L_shaped_plaque_into_four_equal_parts_l1165_116544


namespace NUMINAMATH_GPT_Olly_needs_24_shoes_l1165_116503

def dogs := 3
def cats := 2
def ferrets := 1
def paws_per_dog := 4
def paws_per_cat := 4
def paws_per_ferret := 4

theorem Olly_needs_24_shoes : (dogs * paws_per_dog) + (cats * paws_per_cat) + (ferrets * paws_per_ferret) = 24 :=
by
  sorry

end NUMINAMATH_GPT_Olly_needs_24_shoes_l1165_116503


namespace NUMINAMATH_GPT_ed_lost_seven_marbles_l1165_116590

theorem ed_lost_seven_marbles (D L : ℕ) (h1 : ∃ (Ed_init Tim_init : ℕ), Ed_init = D + 19 ∧ Tim_init = D - 10)
(h2 : ∃ (Ed_final Tim_final : ℕ), Ed_final = D + 19 - L - 4 ∧ Tim_final = D - 10 + 4 + 3)
(h3 : ∀ (Ed_final : ℕ), Ed_final = D + 8)
(h4 : ∀ (Tim_final : ℕ), Tim_final = D):
  L = 7 :=
by
  sorry

end NUMINAMATH_GPT_ed_lost_seven_marbles_l1165_116590


namespace NUMINAMATH_GPT_helga_shoes_l1165_116519

theorem helga_shoes (x : ℕ) : 
  (x + (x + 2) + 0 + 2 * (x + (x + 2) + 0) = 48) → x = 7 := 
by
  sorry

end NUMINAMATH_GPT_helga_shoes_l1165_116519


namespace NUMINAMATH_GPT_two_baskets_of_peaches_l1165_116568

theorem two_baskets_of_peaches (R G : ℕ) (h1 : G = R + 2) (h2 : 2 * R + 2 * G = 12) : R = 2 :=
by
  sorry

end NUMINAMATH_GPT_two_baskets_of_peaches_l1165_116568


namespace NUMINAMATH_GPT_bus_stop_time_per_hour_l1165_116525

theorem bus_stop_time_per_hour 
  (speed_without_stoppages : ℝ)
  (speed_with_stoppages : ℝ)
  (h1 : speed_without_stoppages = 64)
  (h2 : speed_with_stoppages = 48) : 
  ∃ t : ℝ, t = 15 := 
by
  sorry

end NUMINAMATH_GPT_bus_stop_time_per_hour_l1165_116525


namespace NUMINAMATH_GPT_cos_double_angle_sum_l1165_116501

theorem cos_double_angle_sum
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 := by
  sorry

end NUMINAMATH_GPT_cos_double_angle_sum_l1165_116501


namespace NUMINAMATH_GPT_relationship_among_abc_l1165_116567

noncomputable def a : ℝ := 20.3
noncomputable def b : ℝ := 0.32
noncomputable def c : ℝ := Real.log 25 / Real.log 10

theorem relationship_among_abc : b < a ∧ a < c :=
by
  -- Proof needs to be filled in here
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l1165_116567


namespace NUMINAMATH_GPT_calculate_value_l1165_116500

theorem calculate_value :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end NUMINAMATH_GPT_calculate_value_l1165_116500


namespace NUMINAMATH_GPT_general_term_of_sequence_l1165_116575

theorem general_term_of_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S (n + 1) = 3 * (n + 1) ^ 2 - 2 * (n + 1)) →
  a 1 = 1 →
  (∀ n, a (n + 1) = S (n + 1) - S n) →
  (∀ n, a n = 6 * n - 5) := 
by
  intros hS ha1 ha
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l1165_116575


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1165_116599

/-- Prove the speed of the boat in still water given the conditions -/
theorem boat_speed_in_still_water (V_s : ℝ) (T : ℝ) (D : ℝ) (V_b : ℝ) :
  V_s = 4 ∧ T = 4 ∧ D = 112 ∧ (D / T = V_b + V_s) → V_b = 24 := sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1165_116599


namespace NUMINAMATH_GPT_mark_total_young_fish_l1165_116594

-- Define the conditions
def num_tanks : ℕ := 5
def fish_per_tank : ℕ := 6
def young_per_fish : ℕ := 25

-- Define the total number of young fish
def total_young_fish := num_tanks * fish_per_tank * young_per_fish

-- The theorem statement
theorem mark_total_young_fish : total_young_fish = 750 :=
  by
    sorry

end NUMINAMATH_GPT_mark_total_young_fish_l1165_116594


namespace NUMINAMATH_GPT_percent_decrease_l1165_116577

-- Definitions based on conditions
def originalPrice : ℝ := 100
def salePrice : ℝ := 10

-- The percentage decrease is the main statement to prove
theorem percent_decrease : ((originalPrice - salePrice) / originalPrice) * 100 = 90 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_percent_decrease_l1165_116577


namespace NUMINAMATH_GPT_salary_increase_l1165_116538

theorem salary_increase (x : ℝ) (y : ℝ) :
  (1000 : ℝ) * 80 + 50 = y → y - (50 + 80 * x) = 80 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_salary_increase_l1165_116538


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1165_116576

theorem simplify_and_evaluate_expression (x y : ℝ) (h₁ : x = 2) (h₂ : y = -1) : 
  2 * x * y - (1 / 2) * (4 * x * y - 8 * x^2 * y^2) + 2 * (3 * x * y - 5 * x^2 * y^2) = -36 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1165_116576


namespace NUMINAMATH_GPT_abs_algebraic_expression_l1165_116563

theorem abs_algebraic_expression (x : ℝ) (h : |2 * x - 3| - 3 + 2 * x = 0) : |2 * x - 5| = 5 - 2 * x := 
by sorry

end NUMINAMATH_GPT_abs_algebraic_expression_l1165_116563


namespace NUMINAMATH_GPT_find_integer_for_perfect_square_l1165_116518

theorem find_integer_for_perfect_square :
  ∃ (n : ℤ), ∃ (m : ℤ), n^2 + 20 * n + 11 = m^2 ∧ n = 35 := by
  sorry

end NUMINAMATH_GPT_find_integer_for_perfect_square_l1165_116518


namespace NUMINAMATH_GPT_initial_peanuts_l1165_116565

theorem initial_peanuts (x : ℕ) (h : x + 4 = 8) : x = 4 :=
sorry

end NUMINAMATH_GPT_initial_peanuts_l1165_116565


namespace NUMINAMATH_GPT_max_fraction_l1165_116573

theorem max_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) : 
  ∃ k, k = (x + y) / x ∧ k ≤ -2 := 
sorry

end NUMINAMATH_GPT_max_fraction_l1165_116573


namespace NUMINAMATH_GPT_abs_inequality_range_l1165_116585

theorem abs_inequality_range (x : ℝ) (b : ℝ) (h : 0 < b) : (b > 2) ↔ ∃ x : ℝ, |x - 5| + |x - 7| < b :=
sorry

end NUMINAMATH_GPT_abs_inequality_range_l1165_116585


namespace NUMINAMATH_GPT_smallest_n_interval_l1165_116522

theorem smallest_n_interval :
  ∃ n : ℕ, (∃ x : ℤ, ⌊10 ^ n / x⌋ = 2006) ∧ 7 ≤ n ∧ n ≤ 12 :=
sorry

end NUMINAMATH_GPT_smallest_n_interval_l1165_116522


namespace NUMINAMATH_GPT_hyperbola_circle_intersection_l1165_116566

open Real

theorem hyperbola_circle_intersection (a r : ℝ) (P Q R S : ℝ × ℝ) 
  (hP : P.1^2 - P.2^2 = a^2) (hQ : Q.1^2 - Q.2^2 = a^2) (hR : R.1^2 - R.2^2 = a^2) (hS : S.1^2 - S.2^2 = a^2)
  (hO : r ≥ 0)
  (hPQRS : (P.1 - 0)^2 + (P.2 - 0)^2 = r^2 ∧
            (Q.1 - 0)^2 + (Q.2 - 0)^2 = r^2 ∧
            (R.1 - 0)^2 + (R.2 - 0)^2 = r^2 ∧
            (S.1 - 0)^2 + (S.2 - 0)^2 = r^2) : 
  (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) + (R.1^2 + R.2^2) + (S.1^2 + S.2^2) = 4 * r^2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_circle_intersection_l1165_116566


namespace NUMINAMATH_GPT_sum_of_coefficients_eq_39_l1165_116598

theorem sum_of_coefficients_eq_39 :
  5 * (2 * 1^8 - 3 * 1^3 + 4) - 6 * (1^6 + 4 * 1^3 - 9) = 39 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_eq_39_l1165_116598


namespace NUMINAMATH_GPT_fraction_exponentiation_multiplication_l1165_116572

theorem fraction_exponentiation_multiplication :
  (1 / 3) ^ 4 * (1 / 8) = 1 / 648 :=
by
  sorry

end NUMINAMATH_GPT_fraction_exponentiation_multiplication_l1165_116572


namespace NUMINAMATH_GPT_fred_initial_cards_l1165_116529

variables {n : ℕ}

theorem fred_initial_cards (h : n - 22 = 18) : n = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_fred_initial_cards_l1165_116529


namespace NUMINAMATH_GPT_approx_num_fish_in_pond_l1165_116532

noncomputable def numFishInPond (tagged_in_second: ℕ) (total_second: ℕ) (tagged: ℕ) : ℕ :=
  tagged * total_second / tagged_in_second

theorem approx_num_fish_in_pond :
  numFishInPond 2 50 50 = 1250 := by
  sorry

end NUMINAMATH_GPT_approx_num_fish_in_pond_l1165_116532


namespace NUMINAMATH_GPT_simplify_expression_l1165_116509

theorem simplify_expression :
  let a := 2
  let b := -3
  10 * a^2 * b - (2 * a * b^2 - 2 * (a * b - 5 * a^2 * b)) = -48 := sorry

end NUMINAMATH_GPT_simplify_expression_l1165_116509


namespace NUMINAMATH_GPT_range_of_a_l1165_116539

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, 7 * x1^2 - (a + 13) * x1 + a^2 - a - 2 = 0 ∧
                 7 * x2^2 - (a + 13) * x2 + a^2 - a - 2 = 0 ∧
                 0 < x1 ∧ x1 < 1 ∧ 1 < x2 ∧ x2 < 2) →
  (-2 < a ∧ a < -1) ∨ (3 < a ∧ a < 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1165_116539


namespace NUMINAMATH_GPT_total_balls_l1165_116556

theorem total_balls (r b g : ℕ) (ratio : r = 2 * k ∧ b = 4 * k ∧ g = 6 * k) (green_balls : g = 36) : r + b + g = 72 :=
by
  sorry

end NUMINAMATH_GPT_total_balls_l1165_116556


namespace NUMINAMATH_GPT_rate_of_A_is_8_l1165_116521

noncomputable def rate_of_A (a b : ℕ) : ℕ :=
  if b = a + 4 ∧ 48 * b = 72 * a then a else 0

theorem rate_of_A_is_8 {a b : ℕ} 
  (h1 : b = a + 4)
  (h2 : 48 * b = 72 * a) : 
  rate_of_A a b = 8 :=
by
  -- proof steps can be added here
  sorry

end NUMINAMATH_GPT_rate_of_A_is_8_l1165_116521


namespace NUMINAMATH_GPT_batsman_average_proof_l1165_116579

noncomputable def batsman_average_after_17th_inning (A : ℝ) : ℝ :=
  (A * 16 + 87) / 17

theorem batsman_average_proof (A : ℝ) (h1 : 16 * A + 87 = 17 * (A + 2)) : batsman_average_after_17th_inning 53 = 55 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_proof_l1165_116579


namespace NUMINAMATH_GPT_vector_expression_simplification_l1165_116574

variable (a b : Type)
variable (α : Type) [Field α]
variable [AddCommGroup a] [Module α a]

theorem vector_expression_simplification
  (vector_a vector_b : a) :
  (1/3 : α) • (vector_a - (2 : α) • vector_b) + vector_b = (1/3 : α) • vector_a + (1/3 : α) • vector_b :=
by
  sorry

end NUMINAMATH_GPT_vector_expression_simplification_l1165_116574


namespace NUMINAMATH_GPT_min_value_sum_pos_int_l1165_116593

theorem min_value_sum_pos_int 
  (a b c : ℕ)
  (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots: ∃ (A B : ℝ), A < 0 ∧ A > -1 ∧ B > 0 ∧ B < 1 ∧ (∀ x : ℝ, x^2*x*a + x*b + c = 0 → x = A ∨ x = B))
  : a + b + c = 11 :=
sorry

end NUMINAMATH_GPT_min_value_sum_pos_int_l1165_116593


namespace NUMINAMATH_GPT_remainder_of_max_6_multiple_no_repeated_digits_l1165_116534

theorem remainder_of_max_6_multiple_no_repeated_digits (M : ℕ) 
  (hM : ∃ n, M = 6 * n) 
  (h_unique_digits : ∀ (d : ℕ), d ∈ (M.digits 10) → (M.digits 10).count d = 1) 
  (h_max_M : ∀ (k : ℕ), (∃ n, k = 6 * n) ∧ (∀ (d : ℕ), d ∈ (k.digits 10) → (k.digits 10).count d = 1) → k ≤ M) :
  M % 100 = 78 := 
sorry

end NUMINAMATH_GPT_remainder_of_max_6_multiple_no_repeated_digits_l1165_116534


namespace NUMINAMATH_GPT_car_total_distance_l1165_116515

noncomputable def distance_first_segment (speed1 : ℝ) (time1 : ℝ) : ℝ :=
  speed1 * time1

noncomputable def distance_second_segment (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed2 * time2

noncomputable def distance_final_segment (speed3 : ℝ) (time3 : ℝ) : ℝ :=
  speed3 * time3

noncomputable def total_distance (d1 d2 d3 : ℝ) : ℝ :=
  d1 + d2 + d3

theorem car_total_distance :
  let d1 := distance_first_segment 65 2
  let d2 := distance_second_segment 80 1.5
  let d3 := distance_final_segment 50 2
  total_distance d1 d2 d3 = 350 :=
by
  sorry

end NUMINAMATH_GPT_car_total_distance_l1165_116515


namespace NUMINAMATH_GPT_fraction_identity_l1165_116550

theorem fraction_identity :
  (1721^2 - 1714^2 : ℚ) / (1728^2 - 1707^2) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1165_116550


namespace NUMINAMATH_GPT_power_function_half_l1165_116595

theorem power_function_half (a : ℝ) (ha : (4 : ℝ)^a / (2 : ℝ)^a = 3) : (1 / 2 : ℝ) ^ a = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_power_function_half_l1165_116595


namespace NUMINAMATH_GPT_model1_best_fitting_effect_l1165_116540

-- Definitions for the correlation coefficients of the models
def R1 : ℝ := 0.98
def R2 : ℝ := 0.80
def R3 : ℝ := 0.50
def R4 : ℝ := 0.25

-- Main theorem stating Model 1 has the best fitting effect
theorem model1_best_fitting_effect : |R1| > |R2| ∧ |R1| > |R3| ∧ |R1| > |R4| :=
by sorry

end NUMINAMATH_GPT_model1_best_fitting_effect_l1165_116540


namespace NUMINAMATH_GPT_no_distributive_laws_hold_l1165_116523

def tripledAfterAdding (a b : ℝ) : ℝ := 3 * (a + b)

theorem no_distributive_laws_hold (x y z : ℝ) :
  ¬ (tripledAfterAdding x (y + z) = tripledAfterAdding (tripledAfterAdding x y) (tripledAfterAdding x z)) ∧
  ¬ (x + (tripledAfterAdding y z) = tripledAfterAdding (x + y) (x + z)) ∧
  ¬ (tripledAfterAdding x (tripledAfterAdding y z) = tripledAfterAdding (tripledAfterAdding x y) (tripledAfterAdding x z)) :=
by sorry

end NUMINAMATH_GPT_no_distributive_laws_hold_l1165_116523


namespace NUMINAMATH_GPT_find_a_plus_d_l1165_116512

theorem find_a_plus_d (a b c d : ℝ) (h1 : a + b = 5) (h2 : b + c = 6) (h3 : c + d = 3) : a + d = -1 := 
by 
  -- omit proof
  sorry

end NUMINAMATH_GPT_find_a_plus_d_l1165_116512


namespace NUMINAMATH_GPT_distance_Owlford_Highcastle_l1165_116504

open Complex

theorem distance_Owlford_Highcastle :
  let Highcastle := (0 : ℂ)
  let Owlford := (900 + 1200 * I : ℂ)
  dist Highcastle Owlford = 1500 := by
  sorry

end NUMINAMATH_GPT_distance_Owlford_Highcastle_l1165_116504


namespace NUMINAMATH_GPT_p_q_r_cubic_sum_l1165_116535

theorem p_q_r_cubic_sum (p q r : ℚ) (h1 : p + q + r = 4) (h2 : p * q + p * r + q * r = 6) (h3 : p * q * r = -8) : 
  p^3 + q^3 + r^3 = 8 := by
  sorry

end NUMINAMATH_GPT_p_q_r_cubic_sum_l1165_116535


namespace NUMINAMATH_GPT_triangle_inequality_holds_l1165_116596

theorem triangle_inequality_holds (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^3 + b^3 + c^3 + 4 * a * b * c ≤ (9 / 32) * (a + b + c)^3 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_inequality_holds_l1165_116596


namespace NUMINAMATH_GPT_find_range_of_a_l1165_116545

variable {a : ℝ}
variable {x : ℝ}

theorem find_range_of_a (h₁ : x ∈ Set.Ioo (-2:ℝ) (-1:ℝ)) :
  ∃ a, a ∈ Set.Icc (1:ℝ) (2:ℝ) ∧ (x + 1)^2 < Real.log (|x|) / Real.log a :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_a_l1165_116545


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1165_116543

-- Definitions from conditions
def repeating_decimal_1_3 : ℚ := 1 / 3
def repeating_decimal_2_99 : ℚ := 2 / 99

-- Statement to prove
theorem repeating_decimal_sum : repeating_decimal_1_3 + repeating_decimal_2_99 = 35 / 99 :=
by sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l1165_116543


namespace NUMINAMATH_GPT_population_multiple_of_seven_l1165_116562

theorem population_multiple_of_seven 
  (a b c : ℕ) 
  (h1 : a^2 + 100 = b^2 + 1) 
  (h2 : b^2 + 1 + 100 = c^2) : 
  (∃ k : ℕ, a = 7 * k) :=
sorry

end NUMINAMATH_GPT_population_multiple_of_seven_l1165_116562


namespace NUMINAMATH_GPT_decimal_representation_of_fraction_l1165_116554

theorem decimal_representation_of_fraction :
  (47 : ℝ) / (2^3 * 5^4) = 0.0094 :=
by
  sorry

end NUMINAMATH_GPT_decimal_representation_of_fraction_l1165_116554


namespace NUMINAMATH_GPT_ratio_WX_XY_l1165_116559

theorem ratio_WX_XY (p q : ℝ) (h : 3 * p = 4 * q) : (4 * q) / (3 * p) = 12 / 7 := by
  sorry

end NUMINAMATH_GPT_ratio_WX_XY_l1165_116559


namespace NUMINAMATH_GPT_solve_system_l1165_116582

def system_of_equations (x y : ℝ) : Prop :=
  (4 * (x - y) = 8 - 3 * y) ∧ (x / 2 + y / 3 = 1)

theorem solve_system : ∃ x y : ℝ, system_of_equations x y ∧ x = 2 ∧ y = 0 := 
  by
  sorry

end NUMINAMATH_GPT_solve_system_l1165_116582


namespace NUMINAMATH_GPT_books_withdrawn_is_15_l1165_116511

-- Define the initial condition
def initial_books : ℕ := 250

-- Define the books taken out on Tuesday
def books_taken_out_tuesday : ℕ := 120

-- Define the books returned on Wednesday
def books_returned_wednesday : ℕ := 35

-- Define the books left in library on Thursday
def books_left_thursday : ℕ := 150

-- Define the problem: Determine the number of books withdrawn on Thursday
def books_withdrawn_thursday : ℕ :=
  (initial_books - books_taken_out_tuesday + books_returned_wednesday) - books_left_thursday

-- The statement we want to prove
theorem books_withdrawn_is_15 : books_withdrawn_thursday = 15 := by sorry

end NUMINAMATH_GPT_books_withdrawn_is_15_l1165_116511


namespace NUMINAMATH_GPT_completion_time_B_l1165_116592

-- Definitions based on conditions
def work_rate_A : ℚ := 1 / 10 -- A's rate of completing work per day

def efficiency_B : ℚ := 1.75 -- B is 75% more efficient than A

def work_rate_B : ℚ := efficiency_B * work_rate_A -- B's work rate per day

-- The main theorem that we need to prove
theorem completion_time_B : (1 : ℚ) / work_rate_B = 40 / 7 :=
by 
  sorry

end NUMINAMATH_GPT_completion_time_B_l1165_116592
