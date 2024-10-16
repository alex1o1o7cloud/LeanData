import Mathlib

namespace NUMINAMATH_CALUDE_lacustrine_glacial_monoliths_l2133_213317

-- Define the total number of monoliths
def total_monoliths : ℕ := 98

-- Define the probability of a monolith being sand
def prob_sand : ℚ := 1/7

-- Define the probability of a monolith being marine loam
def prob_marine_loam : ℚ := 9/14

-- Theorem statement
theorem lacustrine_glacial_monoliths :
  let sand_monoliths := (prob_sand * total_monoliths : ℚ).num
  let loam_monoliths := total_monoliths - sand_monoliths
  let marine_loam_monoliths := (prob_marine_loam * loam_monoliths : ℚ).num
  let lacustrine_glacial_loam_monoliths := loam_monoliths - marine_loam_monoliths
  sand_monoliths + lacustrine_glacial_loam_monoliths = 44 := by
  sorry

end NUMINAMATH_CALUDE_lacustrine_glacial_monoliths_l2133_213317


namespace NUMINAMATH_CALUDE_monochromatic_triangle_in_17_vertex_graph_l2133_213399

/-- A coloring of edges in a complete graph -/
def EdgeColoring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A complete graph has a monochromatic triangle if there exist three vertices
    such that all edges between them have the same color -/
def has_monochromatic_triangle (n : ℕ) (coloring : EdgeColoring n) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    coloring i j = coloring j k ∧ coloring j k = coloring i k

/-- In any complete graph with 17 vertices where each edge is colored in one of three colors,
    there exist three vertices such that all edges between them are the same color -/
theorem monochromatic_triangle_in_17_vertex_graph :
  ∀ (coloring : EdgeColoring 17), has_monochromatic_triangle 17 coloring :=
sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_in_17_vertex_graph_l2133_213399


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l2133_213372

/-- Given two people moving in opposite directions, this theorem proves
    the speed of one person given the speed of the other and their final distance. -/
theorem opposite_direction_speed 
  (pooja_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : pooja_speed = 3) 
  (h2 : time = 4) 
  (h3 : final_distance = 32) : 
  ∃ (roja_speed : ℝ), roja_speed = 5 ∧ final_distance = (roja_speed + pooja_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l2133_213372


namespace NUMINAMATH_CALUDE_s_range_l2133_213381

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def divisible_by_11 (n : ℕ) : Prop := ∃ k, n = 11 * k

def s (n : ℕ) : ℕ := sorry

theorem s_range (n : ℕ) (h_composite : is_composite n) (h_div11 : divisible_by_11 n) :
  ∃ (m : ℕ), m ≥ 11 ∧ s n = m ∧ ∀ (k : ℕ), k ≥ 11 → ∃ (p : ℕ), is_composite p ∧ divisible_by_11 p ∧ s p = k :=
sorry

end NUMINAMATH_CALUDE_s_range_l2133_213381


namespace NUMINAMATH_CALUDE_horner_method_correctness_l2133_213357

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

def horner_eval (x : ℝ) : ℝ := 
  let v0 := 1
  let v1 := v0 * x + 0
  let v2 := v1 * x + 2
  let v3 := v2 * x + 3
  let v4 := v3 * x + 1
  v4 * x + 1

theorem horner_method_correctness : f 3 = horner_eval 3 := by sorry

end NUMINAMATH_CALUDE_horner_method_correctness_l2133_213357


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_b_greater_than_nine_l2133_213339

/-- If the solution set of the inequality |x-4|-|x+5| ≥ b about x is empty, then b > 9 -/
theorem empty_solution_set_implies_b_greater_than_nine (b : ℝ) :
  (∀ x : ℝ, |x - 4| - |x + 5| < b) → b > 9 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_b_greater_than_nine_l2133_213339


namespace NUMINAMATH_CALUDE_angle_trisection_l2133_213325

theorem angle_trisection (n : ℕ) (h : ¬ 3 ∣ n) :
  ∃ (a b : ℤ), 3 * a + n * b = 1 :=
by sorry

end NUMINAMATH_CALUDE_angle_trisection_l2133_213325


namespace NUMINAMATH_CALUDE_sum_of_integers_l2133_213336

theorem sum_of_integers (a b c : ℤ) :
  a = (b + c) / 3 →
  b = (a + c) / 5 →
  c = 35 →
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2133_213336


namespace NUMINAMATH_CALUDE_cell_count_after_ten_days_l2133_213342

/-- Represents the cell division process over 10 days -/
def cellDivision (initialCells : ℕ) (firstSplitFactor : ℕ) (laterSplitFactor : ℕ) (totalDays : ℕ) : ℕ :=
  let afterFirstTwoDays := initialCells * firstSplitFactor
  let remainingDivisions := (totalDays - 2) / 2
  afterFirstTwoDays * laterSplitFactor ^ remainingDivisions

/-- Theorem stating the number of cells after 10 days -/
theorem cell_count_after_ten_days :
  cellDivision 5 3 2 10 = 240 := by
  sorry

#eval cellDivision 5 3 2 10

end NUMINAMATH_CALUDE_cell_count_after_ten_days_l2133_213342


namespace NUMINAMATH_CALUDE_divisibility_criterion_l2133_213396

theorem divisibility_criterion (a b c : ℤ) (d : ℤ) (h1 : d = 10*c + 1) (h2 : ∃ k, a - b*c = d*k) : 
  ∃ m, 10*a + b = d*m :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l2133_213396


namespace NUMINAMATH_CALUDE_linear_equation_exponent_sum_l2133_213313

theorem linear_equation_exponent_sum (a b : ℝ) : 
  (∀ x y : ℝ, ∃ k m : ℝ, 4*x^(a+b) - 3*y^(3*a+2*b-4) = k*x + m*y + 2) → 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_sum_l2133_213313


namespace NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l2133_213333

theorem prime_pairs_dividing_sum_of_powers (p q : Nat) : 
  Nat.Prime p ∧ Nat.Prime q → (p * q ∣ (5^p + 5^q)) ↔ 
    ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ 
     (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2) ∨ 
     (p = 5 ∧ q = 5) ∨ (p = 5 ∧ q = 313) ∨ 
     (p = 313 ∧ q = 5)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l2133_213333


namespace NUMINAMATH_CALUDE_farthest_point_is_two_zero_l2133_213334

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0

-- Define the tangency conditions
def externally_tangent (x y : ℝ) : Prop := 
  ∃ (r : ℝ), r > 0 ∧ ∀ (x' y' : ℝ), circle1 x' y' → ((x - x')^2 + (y - y')^2 = (1 + r)^2)

def internally_tangent (x y : ℝ) : Prop := 
  ∃ (r : ℝ), r > 0 ∧ ∀ (x' y' : ℝ), circle2 x' y' → ((x - x')^2 + (y - y')^2 = (3 - r)^2)

-- Define the farthest point condition
def is_farthest_point (x y : ℝ) : Prop :=
  externally_tangent x y ∧ internally_tangent x y ∧
  ∀ (x' y' : ℝ), externally_tangent x' y' → internally_tangent x' y' → 
    (x^2 + y^2 ≥ x'^2 + y'^2)

-- Theorem statement
theorem farthest_point_is_two_zero : is_farthest_point 2 0 := by sorry

end NUMINAMATH_CALUDE_farthest_point_is_two_zero_l2133_213334


namespace NUMINAMATH_CALUDE_connie_markers_count_l2133_213308

/-- The number of red markers Connie has -/
def red_markers : ℕ := 41

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 64

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers

theorem connie_markers_count : total_markers = 105 := by
  sorry

end NUMINAMATH_CALUDE_connie_markers_count_l2133_213308


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2133_213340

theorem parabola_line_intersection (b : ℝ) : 
  (∃! x : ℝ, bx^2 + 5*x + 2 = -2*x - 2) ↔ b = 49/16 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2133_213340


namespace NUMINAMATH_CALUDE_cube_equation_solution_l2133_213380

theorem cube_equation_solution (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 15 * b) : b = 147 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l2133_213380


namespace NUMINAMATH_CALUDE_train_meeting_distance_l2133_213371

theorem train_meeting_distance (route_length : ℝ) (time_x time_y : ℝ) 
  (h1 : route_length = 160)
  (h2 : time_x = 5)
  (h3 : time_y = 3)
  : let speed_x := route_length / time_x
    let speed_y := route_length / time_y
    let meeting_time := route_length / (speed_x + speed_y)
    speed_x * meeting_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_distance_l2133_213371


namespace NUMINAMATH_CALUDE_translation_theorem_l2133_213376

/-- The original function -/
def f (x : ℝ) : ℝ := x^2 + x

/-- The translated function -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The translation amount -/
def a : ℝ := 2

theorem translation_theorem (h : a > 0) : 
  ∀ x, g x = f (x - a) :=
by sorry

end NUMINAMATH_CALUDE_translation_theorem_l2133_213376


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l2133_213359

/-- Represents the neighborhood cable layout problem -/
structure NeighborhoodCable where
  east_west_streets : Nat
  east_west_length : Nat
  north_south_streets : Nat
  north_south_length : Nat
  cable_per_mile : Nat
  cable_cost_per_mile : Nat

/-- Calculates the total cost of cable for the neighborhood -/
def total_cable_cost (n : NeighborhoodCable) : Nat :=
  let total_street_length := n.east_west_streets * n.east_west_length + n.north_south_streets * n.north_south_length
  let total_cable_length := total_street_length * n.cable_per_mile
  total_cable_length * n.cable_cost_per_mile

/-- The theorem stating the total cost of cable for the given neighborhood -/
theorem neighborhood_cable_cost :
  let n : NeighborhoodCable := {
    east_west_streets := 18,
    east_west_length := 2,
    north_south_streets := 10,
    north_south_length := 4,
    cable_per_mile := 5,
    cable_cost_per_mile := 2000
  }
  total_cable_cost n = 760000 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_cable_cost_l2133_213359


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l2133_213397

theorem complex_magnitude_equation (t : ℝ) : 
  (t > 0 ∧ Complex.abs (t + 3 * Complex.I * Real.sqrt 2) * Complex.abs (8 - 3 * Complex.I) = 40) ↔ 
  t = Real.sqrt (286 / 73) := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l2133_213397


namespace NUMINAMATH_CALUDE_circle_center_polar_coordinates_l2133_213365

theorem circle_center_polar_coordinates :
  let ρ : ℝ → ℝ → ℝ := fun θ r ↦ r
  let circle_equation : ℝ → ℝ → Prop := fun θ r ↦ ρ θ r = Real.sqrt 2 * (Real.cos θ + Real.sin θ)
  let is_center : ℝ → ℝ → Prop := fun r θ ↦ ∀ θ' r', circle_equation θ' r' → 
    (r * Real.cos θ - r' * Real.cos θ')^2 + (r * Real.sin θ - r' * Real.sin θ')^2 = r^2
  is_center 1 (Real.pi / 4) := by sorry

end NUMINAMATH_CALUDE_circle_center_polar_coordinates_l2133_213365


namespace NUMINAMATH_CALUDE_multiply_decimal_l2133_213306

theorem multiply_decimal : (3.6 : ℝ) * 0.25 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimal_l2133_213306


namespace NUMINAMATH_CALUDE_increasing_odd_function_bound_l2133_213386

/-- A function f: ℝ → ℝ is a "k-type increasing function" if for all x, f(x + k) > f(x) -/
def is_k_type_increasing (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x, f (x + k) > f x

theorem increasing_odd_function_bound (f : ℝ → ℝ) (a : ℝ) 
    (h_odd : ∀ x, f (-x) = -f x)
    (h_pos : ∀ x > 0, f x = |x - a| - 2*a)
    (h_inc : is_k_type_increasing f 2017) :
    a < 2017/6 := by
  sorry

end NUMINAMATH_CALUDE_increasing_odd_function_bound_l2133_213386


namespace NUMINAMATH_CALUDE_dimitri_burger_calories_l2133_213352

/-- Given that Dimitri eats 3 burgers per day and each burger has 20 calories,
    prove that the total calories consumed after two days is 120 calories. -/
theorem dimitri_burger_calories : 
  let burgers_per_day : ℕ := 3
  let calories_per_burger : ℕ := 20
  let days : ℕ := 2
  burgers_per_day * calories_per_burger * days = 120 := by
  sorry


end NUMINAMATH_CALUDE_dimitri_burger_calories_l2133_213352


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2133_213346

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3}

-- Define set P
def P : Set ℕ := {1, 2}

-- Define set Q
def Q : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_intersection_equals_set : 
  (U \ (P ∩ Q)) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2133_213346


namespace NUMINAMATH_CALUDE_single_elimination_matches_l2133_213301

/-- The number of matches required in a single-elimination tournament -/
def matches_required (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with n participants,
    the number of matches required to determine the winner is n - 1 -/
theorem single_elimination_matches (n : ℕ) (h : n > 0) : 
  matches_required n = n - 1 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_matches_l2133_213301


namespace NUMINAMATH_CALUDE_grill_run_time_theorem_l2133_213307

/-- Represents the time a charcoal grill runs given the rate of coal burning and the amount of coal available -/
def grillRunTime (burnRate : ℕ) (burnTime : ℕ) (bags : ℕ) (coalsPerBag : ℕ) : ℚ :=
  let totalCoals := bags * coalsPerBag
  let minutesPerCycle := burnTime * (totalCoals / burnRate)
  minutesPerCycle / 60

/-- Theorem stating that a grill burning 15 coals every 20 minutes, with 3 bags of 60 coals each, runs for 4 hours -/
theorem grill_run_time_theorem :
  grillRunTime 15 20 3 60 = 4 := by
  sorry

#eval grillRunTime 15 20 3 60

end NUMINAMATH_CALUDE_grill_run_time_theorem_l2133_213307


namespace NUMINAMATH_CALUDE_unique_valid_result_exists_correct_answers_for_71_score_l2133_213378

/-- Represents the score and correct answers for a math competition. -/
structure CompetitionResult where
  groupA_correct : Nat
  groupB_correct : Nat
  groupB_incorrect : Nat
  total_score : Int

/-- Checks if the CompetitionResult is valid according to the competition rules. -/
def is_valid_result (r : CompetitionResult) : Prop :=
  r.groupA_correct ≤ 5 ∧
  r.groupB_correct + r.groupB_incorrect ≤ 12 ∧
  r.total_score = 8 * r.groupA_correct + 5 * r.groupB_correct - 2 * r.groupB_incorrect

/-- Theorem stating that there is a unique valid result with a total score of 71 and 13 correct answers. -/
theorem unique_valid_result_exists :
  ∃! r : CompetitionResult,
    is_valid_result r ∧
    r.total_score = 71 ∧
    r.groupA_correct + r.groupB_correct = 13 :=
  sorry

/-- Theorem stating that any valid result with a total score of 71 must have 13 correct answers. -/
theorem correct_answers_for_71_score :
  ∀ r : CompetitionResult,
    is_valid_result r → r.total_score = 71 →
    r.groupA_correct + r.groupB_correct = 13 :=
  sorry

end NUMINAMATH_CALUDE_unique_valid_result_exists_correct_answers_for_71_score_l2133_213378


namespace NUMINAMATH_CALUDE_perpendicular_lines_triangle_area_l2133_213383

/-- Two perpendicular lines intersecting at (8,6) with y-intercepts differing by 14 form a triangle with area 56 -/
theorem perpendicular_lines_triangle_area :
  ∀ (m₁ m₂ b₁ b₂ : ℝ),
  m₁ * m₂ = -1 →                         -- perpendicular lines
  8 * m₁ + b₁ = 6 →                      -- line 1 passes through (8,6)
  8 * m₂ + b₂ = 6 →                      -- line 2 passes through (8,6)
  b₁ - b₂ = 14 →                         -- difference between y-intercepts
  (1/2) * 8 * |b₁ - b₂| = 56 :=          -- area of triangle
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_triangle_area_l2133_213383


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l2133_213341

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l2133_213341


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2133_213348

theorem complex_equation_solution :
  ∃ (z : ℂ), (3 : ℂ) + 2 * Complex.I * z = (2 : ℂ) - 5 * Complex.I * z ∧ z = (1 / 7 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2133_213348


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2133_213384

theorem polynomial_expansion (x : ℝ) : 
  (2*x^2 + 3*x + 7)*(x - 2) - (x - 2)*(x^2 - 4*x + 9) + (4*x^2 - 3*x + 1)*(x - 2)*(x - 5) = 
  5*x^3 - 26*x^2 + 35*x - 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2133_213384


namespace NUMINAMATH_CALUDE_simplify_expression_combine_like_terms_l2133_213358

-- Define variables
variable (a b : ℝ)

-- Theorem 1
theorem simplify_expression :
  2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b :=
by sorry

-- Theorem 2
theorem combine_like_terms :
  3 * a^2 * b + 2 * a * b^2 - 5 - 3 * a^2 * b - 5 * a * b^2 + 2 = -3 * a * b^2 - 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_combine_like_terms_l2133_213358


namespace NUMINAMATH_CALUDE_sum_of_numbers_greater_than_1_1_l2133_213300

def numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]

theorem sum_of_numbers_greater_than_1_1 : 
  (numbers.filter (λ x => x > 1.1)).sum = 3.9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_greater_than_1_1_l2133_213300


namespace NUMINAMATH_CALUDE_parenthesized_results_l2133_213309

def original_expression : ℚ := 72 / 9 - 3 * 2

def parenthesized_expressions : List ℚ := [
  (72 / 9 - 3) * 2,
  72 / (9 - 3) * 2,
  72 / ((9 - 3) * 2)
]

theorem parenthesized_results :
  original_expression = 2 →
  (parenthesized_expressions.toFinset = {6, 10, 24}) ∧
  (parenthesized_expressions.length = 3) :=
by sorry

end NUMINAMATH_CALUDE_parenthesized_results_l2133_213309


namespace NUMINAMATH_CALUDE_complex_fourth_power_equality_l2133_213312

theorem complex_fourth_power_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4) : b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_equality_l2133_213312


namespace NUMINAMATH_CALUDE_rectangle_area_l2133_213360

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 120) : l * b = 675 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2133_213360


namespace NUMINAMATH_CALUDE_total_weekly_egg_supply_l2133_213314

/-- Represents the daily egg supply to a store -/
structure DailySupply where
  oddDays : ℕ
  evenDays : ℕ

/-- Calculates the total eggs supplied to a store in a week -/
def weeklySupply (supply : DailySupply) : ℕ :=
  4 * supply.oddDays + 3 * supply.evenDays

/-- Converts dozens to individual eggs -/
def dozensToEggs (dozens : ℕ) : ℕ :=
  dozens * 12

theorem total_weekly_egg_supply :
  let store1 := DailySupply.mk (dozensToEggs 5) (dozensToEggs 5)
  let store2 := DailySupply.mk 30 30
  let store3 := DailySupply.mk (dozensToEggs 25) (dozensToEggs 15)
  weeklySupply store1 + weeklySupply store2 + weeklySupply store3 = 2370 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_egg_supply_l2133_213314


namespace NUMINAMATH_CALUDE_ball_hexagons_l2133_213350

/-- A ball made of hexagons and pentagons -/
structure Ball where
  pentagons : ℕ
  hexagons : ℕ
  pentagon_hexagon_edges : ℕ
  hexagon_pentagon_edges : ℕ

/-- Theorem: A ball with 12 pentagons has 20 hexagons -/
theorem ball_hexagons (b : Ball) 
  (h1 : b.pentagons = 12)
  (h2 : b.pentagon_hexagon_edges = 5)
  (h3 : b.hexagon_pentagon_edges = 3) :
  b.hexagons = 20 := by
  sorry

#check ball_hexagons

end NUMINAMATH_CALUDE_ball_hexagons_l2133_213350


namespace NUMINAMATH_CALUDE_bowTie_solution_l2133_213326

noncomputable def bowTie (a b : ℝ) : ℝ := a^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt b^2)))

theorem bowTie_solution (y : ℝ) : bowTie 3 y = 18 → y = 6 * Real.sqrt 2 ∨ y = -6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_bowTie_solution_l2133_213326


namespace NUMINAMATH_CALUDE_count_prime_pairs_sum_80_l2133_213394

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The sum of the pair is 80 -/
def sumIs80 (p q : ℕ) : Prop := p + q = 80

/-- The statement to be proved -/
theorem count_prime_pairs_sum_80 :
  ∃! (pairs : List (ℕ × ℕ)), 
    pairs.length = 4 ∧ 
    (∀ (p q : ℕ), (p, q) ∈ pairs → isPrime p ∧ isPrime q ∧ sumIs80 p q) ∧
    (∀ (p q : ℕ), isPrime p → isPrime q → sumIs80 p q → (p, q) ∈ pairs ∨ (q, p) ∈ pairs) :=
sorry

end NUMINAMATH_CALUDE_count_prime_pairs_sum_80_l2133_213394


namespace NUMINAMATH_CALUDE_race_start_calculation_l2133_213316

/-- Given a kilometer race where runner A can give runner B a 200 meters start,
    and runner B can give runner C a 250 meters start,
    prove that runner A can give runner C a 400 meters start. -/
theorem race_start_calculation (Va Vb Vc : ℝ) 
  (h1 : Va / Vb = 1000 / 800)
  (h2 : Vb / Vc = 1000 / 750) :
  Va / Vc = 1000 / 600 :=
by sorry

end NUMINAMATH_CALUDE_race_start_calculation_l2133_213316


namespace NUMINAMATH_CALUDE_john_unintended_texts_l2133_213324

/-- The number of text messages John receives per week that are not intended for him -/
def unintended_texts_per_week (old_daily_texts old_daily_texts_from_friends new_daily_texts days_per_week : ℕ) : ℕ :=
  (new_daily_texts - old_daily_texts) * days_per_week

/-- Proof that John receives 245 unintended text messages per week -/
theorem john_unintended_texts :
  let old_daily_texts : ℕ := 20
  let new_daily_texts : ℕ := 55
  let days_per_week : ℕ := 7
  unintended_texts_per_week old_daily_texts old_daily_texts new_daily_texts days_per_week = 245 :=
by sorry

end NUMINAMATH_CALUDE_john_unintended_texts_l2133_213324


namespace NUMINAMATH_CALUDE_souvenir_optimal_price_l2133_213387

/-- Represents the optimization problem for a souvenir's selling price --/
def SouvenirOptimization (a : ℝ) : Prop :=
  ∃ (x : ℝ),
    0 < x ∧ x < 1 ∧
    (∀ (z : ℝ), 0 < z ∧ z < 1 →
      5 * a * (1 + 4 * x - x^2 - 4 * x^3) ≥ 5 * a * (1 + 4 * z - z^2 - 4 * z^3)) ∧
    20 * (1 + x) = 30

theorem souvenir_optimal_price (a : ℝ) (h : a > 0) : SouvenirOptimization a := by
  sorry

end NUMINAMATH_CALUDE_souvenir_optimal_price_l2133_213387


namespace NUMINAMATH_CALUDE_prob_different_colors_7_5_l2133_213356

/-- The probability of drawing two chips of different colors from a bag with replacement -/
def prob_different_colors (red_chips green_chips : ℕ) : ℚ :=
  let total_chips := red_chips + green_chips
  let prob_red := red_chips / total_chips
  let prob_green := green_chips / total_chips
  2 * (prob_red * prob_green)

/-- Theorem stating that the probability of drawing two chips of different colors
    from a bag with 7 red chips and 5 green chips, with replacement, is 35/72 -/
theorem prob_different_colors_7_5 :
  prob_different_colors 7 5 = 35 / 72 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_7_5_l2133_213356


namespace NUMINAMATH_CALUDE_least_difference_consecutive_primes_l2133_213373

theorem least_difference_consecutive_primes (x y z p : ℕ) : 
  Prime x ∧ Prime y ∧ Prime z ∧  -- x, y, and z are prime numbers
  x < y ∧ y < z ∧  -- x < y < z
  y - x > 5 ∧  -- y - x > 5
  Even x ∧  -- x is an even integer
  Odd y ∧ Odd z ∧  -- y and z are odd integers
  (∃ k : ℕ, y^2 + x^2 = k * p) ∧  -- (y^2 + x^2) is divisible by a specific prime p
  Prime p →  -- p is prime
  (∃ s : ℕ, s = z - x ∧ ∀ t : ℕ, t = z - x → s ≤ t) ∧ s = 11  -- The least possible value s of z - x is 11
  := by sorry

end NUMINAMATH_CALUDE_least_difference_consecutive_primes_l2133_213373


namespace NUMINAMATH_CALUDE_square_on_circle_radius_l2133_213388

theorem square_on_circle_radius (S : ℝ) (x : ℝ) (R : ℝ) : 
  S = 256 →  -- Area of the square
  x^2 = S →  -- Side length of the square
  (x - R)^2 = R^2 - (x/2)^2 →  -- Pythagorean theorem relation
  R = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_on_circle_radius_l2133_213388


namespace NUMINAMATH_CALUDE_table_people_count_l2133_213345

/-- The number of seeds taken by n people in the first round -/
def first_round_seeds (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of seeds taken by n people in the second round -/
def second_round_seeds (n : ℕ) : ℕ := first_round_seeds n + n^2

/-- The difference in seeds taken between the second and first rounds -/
def seed_difference (n : ℕ) : ℕ := second_round_seeds n - first_round_seeds n

theorem table_people_count : 
  ∃ n : ℕ, n > 0 ∧ seed_difference n = 100 ∧ 
  (∀ m : ℕ, m > 0 → seed_difference m = 100 → m = n) :=
sorry

end NUMINAMATH_CALUDE_table_people_count_l2133_213345


namespace NUMINAMATH_CALUDE_inequality_holds_l2133_213321

theorem inequality_holds (a b c r : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hr1 : r > 0) (hr2 : r < 3) : 
  r * (a * b + b * c + c * a) + (3 - r) * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l2133_213321


namespace NUMINAMATH_CALUDE_sisters_sandcastle_height_l2133_213351

/-- Given the height of Miki's sandcastle and the difference in height between
    the two sandcastles, calculate the height of her sister's sandcastle. -/
theorem sisters_sandcastle_height
  (miki_height : ℝ)
  (height_difference : ℝ)
  (h1 : miki_height = 0.8333333333333334)
  (h2 : height_difference = 0.3333333333333333) :
  miki_height - height_difference = 0.5 := by
sorry

end NUMINAMATH_CALUDE_sisters_sandcastle_height_l2133_213351


namespace NUMINAMATH_CALUDE_green_balloons_count_l2133_213364

theorem green_balloons_count (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 17 → red = 8 → total = red + green → green = 9 := by
  sorry

end NUMINAMATH_CALUDE_green_balloons_count_l2133_213364


namespace NUMINAMATH_CALUDE_number_of_routes_l2133_213323

def grid_size : ℕ := 3

def total_moves : ℕ := 2 * grid_size

def right_moves : ℕ := grid_size

def down_moves : ℕ := grid_size

theorem number_of_routes : Nat.choose total_moves right_moves = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_routes_l2133_213323


namespace NUMINAMATH_CALUDE_mango_rate_is_55_l2133_213377

/-- The rate per kg of mangoes given Bruce's purchase --/
def mango_rate (grape_kg : ℕ) (grape_rate : ℕ) (mango_kg : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - grape_kg * grape_rate) / mango_kg

/-- Theorem stating that the rate per kg of mangoes is 55 --/
theorem mango_rate_is_55 :
  mango_rate 8 70 10 1110 = 55 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_is_55_l2133_213377


namespace NUMINAMATH_CALUDE_target_word_satisfies_conditions_target_word_is_unique_l2133_213353

/-- Represents a word with multiple meanings -/
structure MultiMeaningWord where
  word : String
  soundsLike : String
  usedInSports : Bool
  usedInPensions : Bool

/-- Represents the conditions for the word we're looking for -/
def wordConditions : MultiMeaningWord → Prop := fun w =>
  w.soundsLike = "festive dance event" ∧
  w.usedInSports = true ∧
  w.usedInPensions = true

/-- The word we're looking for -/
def targetWord : MultiMeaningWord := {
  word := "баллы",
  soundsLike := "festive dance event",
  usedInSports := true,
  usedInPensions := true
}

/-- Theorem stating that our target word satisfies all conditions -/
theorem target_word_satisfies_conditions : 
  wordConditions targetWord := by sorry

/-- Theorem stating that our target word is unique -/
theorem target_word_is_unique :
  ∀ w : MultiMeaningWord, wordConditions w → w = targetWord := by sorry

end NUMINAMATH_CALUDE_target_word_satisfies_conditions_target_word_is_unique_l2133_213353


namespace NUMINAMATH_CALUDE_sphere_radius_from_shadow_and_pole_l2133_213318

/-- The radius of a sphere given its shadow and a reference pole -/
theorem sphere_radius_from_shadow_and_pole 
  (sphere_shadow : ℝ) 
  (pole_height : ℝ) 
  (pole_shadow : ℝ) 
  (h_sphere_shadow : sphere_shadow = 15)
  (h_pole_height : pole_height = 1.5)
  (h_pole_shadow : pole_shadow = 3) :
  let tan_theta := pole_height / pole_shadow
  let radius := sphere_shadow * tan_theta
  radius = 7.5 := by sorry

end NUMINAMATH_CALUDE_sphere_radius_from_shadow_and_pole_l2133_213318


namespace NUMINAMATH_CALUDE_max_consecutive_non_palindromic_l2133_213362

/-- A year is palindromic if it reads the same backward and forward -/
def isPalindromic (year : ℕ) : Prop :=
  year ≥ 1000 ∧ year ≤ 9999 ∧ 
  (year / 1000 = year % 10) ∧ ((year / 100) % 10 = (year / 10) % 10)

/-- The maximum number of consecutive non-palindromic years between 1000 and 9999 -/
def maxConsecutiveNonPalindromic : ℕ := 109

theorem max_consecutive_non_palindromic :
  ∀ (start : ℕ) (len : ℕ),
    start ≥ 1000 → start + len ≤ 9999 →
    (∀ y : ℕ, y ≥ start ∧ y < start + len → ¬isPalindromic y) →
    len ≤ maxConsecutiveNonPalindromic :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_non_palindromic_l2133_213362


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2133_213391

/-- The distance from the focus to the directrix of the parabola y = 1/2 * x^2 is 1 -/
theorem parabola_focus_directrix_distance : 
  let p : ℝ → ℝ := fun x ↦ (1/2) * x^2
  ∃ f d : ℝ, 
    (∀ x, p x = (1/4) * (x^2 + 1)) ∧  -- Standard form of parabola
    (f = 1/2) ∧                       -- y-coordinate of focus
    (d = -1/2) ∧                      -- y-coordinate of directrix
    (f - d = 1) :=                    -- Distance between focus and directrix
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2133_213391


namespace NUMINAMATH_CALUDE_even_cube_diff_iff_even_sum_l2133_213374

theorem even_cube_diff_iff_even_sum (p q : ℕ) : 
  Even (p^3 - q^3) ↔ Even (p + q) := by sorry

end NUMINAMATH_CALUDE_even_cube_diff_iff_even_sum_l2133_213374


namespace NUMINAMATH_CALUDE_min_value_of_f_l2133_213382

noncomputable section

def f (x : ℝ) : ℝ := (1/2) * x - Real.sin x

theorem min_value_of_f :
  ∃ (min : ℝ), min = π/6 - Real.sqrt 3/2 ∧
  ∀ x ∈ Set.Ioo 0 π, f x ≥ min :=
sorry

end

end NUMINAMATH_CALUDE_min_value_of_f_l2133_213382


namespace NUMINAMATH_CALUDE_function_symmetry_l2133_213368

theorem function_symmetry (f : ℝ → ℝ) (h : f 0 = 1) : f (4 - 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l2133_213368


namespace NUMINAMATH_CALUDE_factory_working_days_l2133_213354

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 6500

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 1300

/-- The number of working days per week -/
def working_days : ℕ := toys_per_week / toys_per_day

theorem factory_working_days :
  working_days = 5 :=
sorry

end NUMINAMATH_CALUDE_factory_working_days_l2133_213354


namespace NUMINAMATH_CALUDE_particular_solution_correct_l2133_213389

/-- The differential equation xy' = y - 1 -/
def diff_eq (x : ℝ) (y : ℝ → ℝ) : Prop :=
  x * (deriv y x) = y x - 1

/-- The general solution y = Cx + 1 -/
def general_solution (C : ℝ) (x : ℝ) : ℝ :=
  C * x + 1

/-- The particular solution y = 4x + 1 -/
def particular_solution (x : ℝ) : ℝ :=
  4 * x + 1

theorem particular_solution_correct :
  ∀ C : ℝ,
  (∀ x : ℝ, diff_eq x (general_solution C)) →
  general_solution C 1 = 5 →
  ∀ x : ℝ, general_solution C x = particular_solution x :=
by sorry

end NUMINAMATH_CALUDE_particular_solution_correct_l2133_213389


namespace NUMINAMATH_CALUDE_november_december_revenue_ratio_l2133_213367

/-- Proves that the revenue in November is 2/5 of the revenue in December given the conditions --/
theorem november_december_revenue_ratio
  (revenue : Fin 3 → ℝ)  -- revenue function for 3 months (0: November, 1: December, 2: January)
  (h1 : revenue 2 = (1/5) * revenue 0)  -- January revenue is 1/5 of November revenue
  (h2 : revenue 1 = (25/6) * ((revenue 0 + revenue 2) / 2))  -- December revenue condition
  : revenue 0 = (2/5) * revenue 1 := by
  sorry

#check november_december_revenue_ratio

end NUMINAMATH_CALUDE_november_december_revenue_ratio_l2133_213367


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l2133_213363

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 2020 * a + 2024 * b = 2028)
  (eq2 : 2022 * a + 2026 * b = 2030) : 
  a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l2133_213363


namespace NUMINAMATH_CALUDE_solution_count_is_49_l2133_213328

/-- The number of positive integer pairs (x, y) satisfying xy / (x + y) = 1000 -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let (x, y) := p
    x > 0 ∧ y > 0 ∧ x * y / (x + y) = 1000
  ) (Finset.product (Finset.range 2001) (Finset.range 2001))).card

theorem solution_count_is_49 : solution_count = 49 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_is_49_l2133_213328


namespace NUMINAMATH_CALUDE_second_to_third_ratio_l2133_213385

/-- Given three numbers where their sum is 500, the first number is 200, and the third number is 100,
    the ratio of the second number to the third number is 2:1. -/
theorem second_to_third_ratio (a b c : ℚ) : 
  a + b + c = 500 → a = 200 → c = 100 → b / c = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_to_third_ratio_l2133_213385


namespace NUMINAMATH_CALUDE_triangle_pq_distance_l2133_213305

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  AB = 4 ∧ AC = 3 ∧ BC = Real.sqrt 37

-- Define point P as the midpoint of AB
def Midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define point Q on AC at distance 1 from C
def PointOnLine (Q A C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, Q.1 = C.1 + t * (A.1 - C.1) ∧ Q.2 = C.2 + t * (A.2 - C.2)

def DistanceFromC (Q C : ℝ × ℝ) : Prop :=
  Real.sqrt ((Q.1 - C.1)^2 + (Q.2 - C.2)^2) = 1

-- Theorem statement
theorem triangle_pq_distance (A B C P Q : ℝ × ℝ) :
  Triangle A B C → Midpoint P A B → PointOnLine Q A C → DistanceFromC Q C →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_pq_distance_l2133_213305


namespace NUMINAMATH_CALUDE_parallelogram_bisector_theorem_l2133_213390

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Representation of a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := sorry

/-- The length between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is on a line defined by two other points -/
def isOnLine (p1 p2 p : Point) : Prop := sorry

/-- Check if a line is an angle bisector -/
def isAngleBisector (vertex p1 p2 : Point) : Prop := sorry

/-- The main theorem -/
theorem parallelogram_bisector_theorem (ABCD : Parallelogram) (E F : Point) :
  perimeter ABCD = 32 →
  isAngleBisector ABCD.C ABCD.D ABCD.B →
  isOnLine ABCD.A ABCD.D E →
  isOnLine ABCD.A ABCD.B F →
  distance ABCD.A E = 2 →
  (distance ABCD.B F = 7 ∨ distance ABCD.B F = 9) := by sorry

end NUMINAMATH_CALUDE_parallelogram_bisector_theorem_l2133_213390


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2133_213347

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) - 3
  f 2 = -2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2133_213347


namespace NUMINAMATH_CALUDE_tangent_line_slope_l2133_213304

/-- The curve y = x³ + x + 16 -/
def f (x : ℝ) : ℝ := x^3 + x + 16

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

/-- The line ℓ passing through (0,0) and tangent to f -/
structure TangentLine where
  t : ℝ
  slope : ℝ
  tangent_point : (ℝ × ℝ) := (t, f t)
  passes_origin : slope * t = f t
  is_tangent : slope = f' t

theorem tangent_line_slope : 
  ∃ (ℓ : TangentLine), ℓ.slope = 13 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l2133_213304


namespace NUMINAMATH_CALUDE_simplify_expression_l2133_213392

theorem simplify_expression : (6^6 * 12^6 * 6^12 * 12^12 : ℕ) = 72^18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2133_213392


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2133_213361

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m ≥ 24 ∧ (m - 24) % 5 = 0 ∧ (m - 24) % 10 = 0 ∧ (m - 24) % 15 = 0 ∧ (m - 24) % 20 = 0 → m ≥ n) ∧
  n ≥ 24 ∧ (n - 24) % 5 = 0 ∧ (n - 24) % 10 = 0 ∧ (n - 24) % 15 = 0 ∧ (n - 24) % 20 = 0 →
  n = 84 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2133_213361


namespace NUMINAMATH_CALUDE_inequality_solution_empty_l2133_213303

theorem inequality_solution_empty (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_empty_l2133_213303


namespace NUMINAMATH_CALUDE_fraction_transformation_l2133_213329

theorem fraction_transformation (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (a + 2 : ℚ) / (b^3 : ℚ) = a / (3 * b : ℚ) → a = 1 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l2133_213329


namespace NUMINAMATH_CALUDE_extended_triangles_similarity_l2133_213315

/-- Represents a point in the complex plane -/
structure ComplexPoint where
  x : ℝ
  y : ℝ

/-- Represents a triangle in the complex plane -/
structure Triangle where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint

/-- Extends a side of a triangle by a factor k -/
def extendSide (A B : ComplexPoint) (k : ℝ) : ComplexPoint :=
  { x := A.x + k * (B.x - A.x),
    y := A.y + k * (B.y - A.y) }

/-- Extends an altitude of a triangle by a factor k -/
def extendAltitude (A B C : ComplexPoint) (k : ℝ) : ComplexPoint :=
  { x := A.x + k * (C.y - B.y),
    y := A.y - k * (C.x - B.x) }

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
    (T1.B.x - T1.A.x)^2 + (T1.B.y - T1.A.y)^2 = r * ((T2.B.x - T2.A.x)^2 + (T2.B.y - T2.A.y)^2) ∧
    (T1.C.x - T1.B.x)^2 + (T1.C.y - T1.B.y)^2 = r * ((T2.C.x - T2.B.x)^2 + (T2.C.y - T2.B.y)^2) ∧
    (T1.A.x - T1.C.x)^2 + (T1.A.y - T1.C.y)^2 = r * ((T2.A.x - T2.C.x)^2 + (T2.A.y - T2.C.y)^2)

theorem extended_triangles_similarity (ABC : Triangle) :
  ∃ (k : ℝ), k > 1 ∧
    let P := extendSide ABC.A ABC.B k
    let Q := extendSide ABC.B ABC.C k
    let R := extendSide ABC.C ABC.A k
    let A' := extendAltitude ABC.A ABC.B ABC.C k
    let B' := extendAltitude ABC.B ABC.C ABC.A k
    let C' := extendAltitude ABC.C ABC.A ABC.B k
    areSimilar
      { A := P, B := Q, C := R }
      { A := A', B := B', C := C' } :=
by sorry

end NUMINAMATH_CALUDE_extended_triangles_similarity_l2133_213315


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2133_213302

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2133_213302


namespace NUMINAMATH_CALUDE_sin_cos_rational_implies_natural_combination_l2133_213335

theorem sin_cos_rational_implies_natural_combination 
  (x y : ℝ) 
  (h1 : ∃ (a : ℚ), a > 0 ∧ Real.sin x + Real.cos y = a)
  (h2 : ∃ (b : ℚ), b > 0 ∧ Real.sin y + Real.cos x = b) :
  ∃ (m n : ℕ), ∃ (k : ℕ), m * Real.sin x + n * Real.cos x = k := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_rational_implies_natural_combination_l2133_213335


namespace NUMINAMATH_CALUDE_expression_value_l2133_213369

theorem expression_value (x y : ℚ) (h : 12 * x = 4 * y + 2) :
  6 * y - 18 * x + 7 = 4 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2133_213369


namespace NUMINAMATH_CALUDE_certain_number_equation_l2133_213370

theorem certain_number_equation : ∃ x : ℝ, 0.6 * 50 = 0.45 * x + 16.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2133_213370


namespace NUMINAMATH_CALUDE_squad_selection_ways_l2133_213327

/-- The number of ways to choose a squad of 8 players (including one dedicated setter) from a team of 12 members -/
def choose_squad (team_size : ℕ) (squad_size : ℕ) : ℕ :=
  team_size * (Nat.choose (team_size - 1) (squad_size - 1))

/-- Theorem stating that choosing a squad of 8 players (including one dedicated setter) from a team of 12 members can be done in 3960 ways -/
theorem squad_selection_ways :
  choose_squad 12 8 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_squad_selection_ways_l2133_213327


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2133_213393

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 11520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2133_213393


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l2133_213355

/-- Given that 8 oranges weigh the same as 6 apples, 
    prove that 48 oranges weigh the same as 36 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℕ → ℝ),
  (∀ n : ℕ, orange_weight n > 0 ∧ apple_weight n > 0) →
  (orange_weight 8 = apple_weight 6) →
  (orange_weight 48 = apple_weight 36) :=
by sorry

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l2133_213355


namespace NUMINAMATH_CALUDE_jelly_cost_l2133_213398

/-- The cost of jelly for sandwiches --/
theorem jelly_cost (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (3 * B + 7 * J) = 378 → 
  (N * J * 7 : ℚ) / 100 = 294 / 100 := by
  sorry

end NUMINAMATH_CALUDE_jelly_cost_l2133_213398


namespace NUMINAMATH_CALUDE_frenchHorn_trombone_difference_l2133_213379

/-- The number of band members for each instrument and their relationships --/
structure BandComposition where
  flute : ℕ
  trumpet : ℕ
  trombone : ℕ
  drums : ℕ
  clarinet : ℕ
  frenchHorn : ℕ
  fluteCount : flute = 5
  trumpetCount : trumpet = 3 * flute
  tromboneCount : trombone = trumpet - 8
  drumsCount : drums = trombone + 11
  clarinetCount : clarinet = 2 * flute
  frenchHornMoreThanTrombone : frenchHorn > trombone
  totalSeats : flute + trumpet + trombone + drums + clarinet + frenchHorn = 65

/-- The theorem stating the difference between French horn and trombone players --/
theorem frenchHorn_trombone_difference (b : BandComposition) :
  b.frenchHorn - b.trombone = 3 := by
  sorry

end NUMINAMATH_CALUDE_frenchHorn_trombone_difference_l2133_213379


namespace NUMINAMATH_CALUDE_complex_division_result_l2133_213395

theorem complex_division_result : 
  let i := Complex.I
  (3 + i) / (1 + i) = 2 - i := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l2133_213395


namespace NUMINAMATH_CALUDE_upper_limit_n_l2133_213332

def is_integer (x : ℚ) : Prop := ∃ k : ℤ, x = k

def has_exactly_three_prime_factors (n : ℕ) : Prop :=
  ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  n = p * q * r

theorem upper_limit_n :
  ∀ n : ℕ, n > 0 →
  is_integer (14 * n / 60) →
  has_exactly_three_prime_factors n →
  n ≤ 210 :=
sorry

end NUMINAMATH_CALUDE_upper_limit_n_l2133_213332


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l2133_213343

theorem sum_of_x_and_y_equals_two (x y : ℝ) 
  (eq1 : 2 * x + 3 * y = 6)
  (eq2 : 3 * x + 2 * y = 4) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l2133_213343


namespace NUMINAMATH_CALUDE_triangle_problem_l2133_213320

theorem triangle_problem (A B C : Real) (p : Real) :
  0 < A ∧ A < π / 2 ∧
  0 < B ∧ B < π / 2 ∧
  0 < C ∧ C < π / 2 ∧
  A + B + C = π ∧
  (Real.sqrt 3 * Real.sin B - Real.cos B) * (Real.sqrt 3 * Real.sin C - Real.cos C) = 4 * Real.cos B * Real.cos C ∧
  Real.sin B = p * Real.sin C →
  A = π / 3 ∧ 1 / 2 < p ∧ p < 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2133_213320


namespace NUMINAMATH_CALUDE_oranges_per_box_l2133_213311

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (oranges_per_box : ℕ) : 
  total_oranges = 24 → num_boxes = 3 → oranges_per_box * num_boxes = total_oranges → oranges_per_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l2133_213311


namespace NUMINAMATH_CALUDE_triangle_transformation_exists_l2133_213337

/-- Triangle represented by its three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Line represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflect a point over a line -/
def reflect (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

/-- Translate a point by a vector -/
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Apply reflection and translation to a triangle -/
def transformTriangle (t : Triangle) (l : Line) (v : ℝ × ℝ) : Triangle :=
  { A := translate (reflect t.A l) v
  , B := translate (reflect t.B l) v
  , C := translate (reflect t.C l) v }

theorem triangle_transformation_exists :
  ∃ (l : Line) (v : ℝ × ℝ),
    let t1 : Triangle := { A := (0, 0), B := (15, 0), C := (0, 5) }
    let t2 : Triangle := { A := (17.2, 19.6), B := (26.2, 6.6), C := (22, 21) }
    transformTriangle t2 l v = t1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_transformation_exists_l2133_213337


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_24_degrees_l2133_213366

/-- Theorem: For a regular polygon with exterior angles measuring 24 degrees each,
    the number of sides is 15 and the sum of interior angles is 2340 degrees. -/
theorem regular_polygon_exterior_24_degrees :
  ∀ (n : ℕ) (exterior_angle : ℝ),
  exterior_angle = 24 →
  n * exterior_angle = 360 →
  n = 15 ∧ (n - 2) * 180 = 2340 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_24_degrees_l2133_213366


namespace NUMINAMATH_CALUDE_discount_calculation_l2133_213319

theorem discount_calculation (cost : ℝ) (selling_price_discounted selling_price_full : ℝ) :
  selling_price_discounted = cost * 1.2 →
  selling_price_full = cost * 1.3 →
  selling_price_full - selling_price_discounted = cost * 0.1 := by
  sorry

#check discount_calculation

end NUMINAMATH_CALUDE_discount_calculation_l2133_213319


namespace NUMINAMATH_CALUDE_beach_house_pool_problem_l2133_213330

theorem beach_house_pool_problem (total_people : ℕ) (legs_in_pool : ℕ) (legs_per_person : ℕ) :
  total_people = 14 →
  legs_in_pool = 16 →
  legs_per_person = 2 →
  total_people - (legs_in_pool / legs_per_person) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_beach_house_pool_problem_l2133_213330


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l2133_213338

theorem coefficient_x3y5_in_expansion : ∀ (x y : ℝ),
  (Finset.range 9).sum (fun k => (Nat.choose 8 k : ℝ) * x^(8 - k) * y^k) =
  56 * x^3 * y^5 + (Finset.range 9).sum (fun k => if k ≠ 5 then (Nat.choose 8 k : ℝ) * x^(8 - k) * y^k else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l2133_213338


namespace NUMINAMATH_CALUDE_triangle_angles_l2133_213375

/-- Triangle angles theorem -/
theorem triangle_angles (ω φ θ : ℝ) : 
  ω + φ + θ = 180 → 
  2 * ω + θ = 180 → 
  φ = 2 * θ → 
  θ = 36 ∧ φ = 72 ∧ ω = 72 := by
sorry

end NUMINAMATH_CALUDE_triangle_angles_l2133_213375


namespace NUMINAMATH_CALUDE_middle_group_frequency_is_32_l2133_213349

/-- Represents a frequency distribution histogram -/
structure Histogram where
  num_rectangles : ℕ
  sample_size : ℕ
  middle_rectangle_area : ℝ
  other_rectangles_area : ℝ

/-- The frequency of the middle group in the histogram -/
def middle_group_frequency (h : Histogram) : ℕ :=
  h.sample_size / 2

/-- Theorem: The frequency of the middle group is 32 under given conditions -/
theorem middle_group_frequency_is_32 (h : Histogram) 
  (h_num_rectangles : h.num_rectangles = 11)
  (h_sample_size : h.sample_size = 160)
  (h_area_equality : h.middle_rectangle_area = h.other_rectangles_area) :
  middle_group_frequency h = 32 := by
  sorry

end NUMINAMATH_CALUDE_middle_group_frequency_is_32_l2133_213349


namespace NUMINAMATH_CALUDE_line_l1_parallel_line_l1_perpendicular_l2133_213322

-- Define the line l passing through points A(4,0) and B(0,3)
def line_l (x y : ℝ) : Prop := x / 4 + y / 3 = 1

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y = 2

-- Define the intersection point of line1 and line2
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define a parallel line to l
def parallel_line (m : ℝ) (x y : ℝ) : Prop := x / 4 + y / 3 = m

-- Define a perpendicular line to l
def perpendicular_line (n : ℝ) (x y : ℝ) : Prop := x / 3 - y / 4 = n

theorem line_l1_parallel :
  ∃ m : ℝ, (∀ x y : ℝ, intersection_point x y → parallel_line m x y) ∧
           (∀ x y : ℝ, parallel_line m x y ↔ 3 * x + 4 * y - 9 = 0) :=
sorry

theorem line_l1_perpendicular :
  ∃ n1 n2 : ℝ, n1 ≠ n2 ∧
    (∀ x y : ℝ, perpendicular_line n1 x y ↔ 4 * x - 3 * y - 12 = 0) ∧
    (∀ x y : ℝ, perpendicular_line n2 x y ↔ 4 * x - 3 * y + 12 = 0) ∧
    (∀ n : ℝ, n = n1 ∨ n = n2 →
      ∃ x1 y1 x2 y2 : ℝ,
        perpendicular_line n x1 0 ∧ perpendicular_line n 0 y2 ∧
        x1 * y2 / 2 = 6) :=
sorry

end NUMINAMATH_CALUDE_line_l1_parallel_line_l1_perpendicular_l2133_213322


namespace NUMINAMATH_CALUDE_complement_of_P_is_singleton_two_l2133_213310

def U : Set Int := {-1, 0, 1, 2}

def P : Set Int := {x ∈ U | x^2 < 2}

theorem complement_of_P_is_singleton_two :
  (U \ P) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_is_singleton_two_l2133_213310


namespace NUMINAMATH_CALUDE_pens_cost_gained_l2133_213331

/-- Represents the number of pens sold -/
def pens_sold : ℕ := 95

/-- Represents the gain percentage as a fraction -/
def gain_percentage : ℚ := 20 / 100

/-- Calculates the selling price given the cost price and gain percentage -/
def selling_price (cost : ℚ) : ℚ := cost * (1 + gain_percentage)

/-- Theorem stating that the number of pens' cost gained is 19 -/
theorem pens_cost_gained : 
  ∃ (cost : ℚ), cost > 0 ∧ 
  (pens_sold * (selling_price cost - cost) = 19 * cost) := by
  sorry

end NUMINAMATH_CALUDE_pens_cost_gained_l2133_213331


namespace NUMINAMATH_CALUDE_arccos_sqrt2_over_2_l2133_213344

theorem arccos_sqrt2_over_2 : Real.arccos (Real.sqrt 2 / 2) = π / 4 := by sorry

end NUMINAMATH_CALUDE_arccos_sqrt2_over_2_l2133_213344
