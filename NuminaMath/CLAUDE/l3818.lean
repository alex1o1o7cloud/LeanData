import Mathlib

namespace NUMINAMATH_CALUDE_correct_stratified_sample_l3818_381814

/-- Represents the number of students in each group -/
structure StudentCount where
  male : ℕ
  female : ℕ

/-- Represents the number of students to be sampled from each group -/
structure SampleCount where
  male : ℕ
  female : ℕ

/-- Calculates the correct stratified sample given the total student count and sample size -/
def stratifiedSample (students : StudentCount) (sampleSize : ℕ) : SampleCount :=
  { male := (students.male * sampleSize) / (students.male + students.female),
    female := (students.female * sampleSize) / (students.male + students.female) }

theorem correct_stratified_sample :
  let students : StudentCount := { male := 20, female := 30 }
  let sampleSize : ℕ := 10
  let sample := stratifiedSample students sampleSize
  sample.male = 4 ∧ sample.female = 6 := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l3818_381814


namespace NUMINAMATH_CALUDE_conic_eccentricity_l3818_381866

theorem conic_eccentricity (m : ℝ) : 
  (m = Real.sqrt (2 * 8) ∨ m = -Real.sqrt (2 * 8)) →
  let e := if m > 0 
    then Real.sqrt 3 / 2 
    else Real.sqrt 5
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧
    (∀ x y : ℝ, x^2 + y^2/m = 1 ↔ (x/a)^2 + (y/b)^2 = 1) ∧
    e = Real.sqrt (|a^2 - b^2|) / max a b :=
by sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l3818_381866


namespace NUMINAMATH_CALUDE_process_flowchart_is_most_appropriate_l3818_381809

/-- Represents a tool for describing production steps --/
structure ProductionDescriptionTool where
  name : String
  divides_into_processes : Bool
  uses_rectangular_boxes : Bool
  notes_process_info : Bool
  uses_flow_lines : Bool
  can_note_time : Bool

/-- Defines the properties of a Process Flowchart --/
def process_flowchart : ProductionDescriptionTool :=
  { name := "Process Flowchart",
    divides_into_processes := true,
    uses_rectangular_boxes := true,
    notes_process_info := true,
    uses_flow_lines := true,
    can_note_time := true }

/-- Theorem stating that a Process Flowchart is the most appropriate tool for describing production steps --/
theorem process_flowchart_is_most_appropriate :
  ∀ (tool : ProductionDescriptionTool),
    tool.divides_into_processes ∧
    tool.uses_rectangular_boxes ∧
    tool.notes_process_info ∧
    tool.uses_flow_lines ∧
    tool.can_note_time →
    tool = process_flowchart :=
by sorry

end NUMINAMATH_CALUDE_process_flowchart_is_most_appropriate_l3818_381809


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l3818_381854

theorem quadratic_function_minimum (a b c : ℝ) (ha : a > 0) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  let x₀ := -b / (2 * a)
  ¬ (∀ x : ℝ, f x ≤ f x₀) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l3818_381854


namespace NUMINAMATH_CALUDE_yann_and_camille_combinations_l3818_381889

/-- The number of items on the menu -/
def menu_items : ℕ := 12

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- Calculates the number of different meal combinations for two people
    when the first person's choice is not available for the second person -/
def meal_combinations (items : ℕ) : ℕ := items * (items - 1)

/-- Theorem stating that the number of different meal combinations
    for Yann and Camille is 132 -/
theorem yann_and_camille_combinations :
  meal_combinations menu_items = 132 := by sorry

end NUMINAMATH_CALUDE_yann_and_camille_combinations_l3818_381889


namespace NUMINAMATH_CALUDE_intersection_A_B_l3818_381875

-- Define set A
def A : Set ℝ := {x : ℝ | 3 * x + 2 > 0}

-- Define set B
def B : Set ℝ := {x : ℝ | (x + 1) * (x - 3) > 0}

-- Theorem to prove
theorem intersection_A_B : A ∩ B = {x : ℝ | x > 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3818_381875


namespace NUMINAMATH_CALUDE_factorial_simplification_l3818_381802

theorem factorial_simplification : (13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / 
  ((10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) + 3 * (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l3818_381802


namespace NUMINAMATH_CALUDE_jasmine_total_weight_l3818_381807

/-- Calculates the total weight in pounds that Jasmine has to carry given the weight of chips and cookies, and the quantities purchased. -/
theorem jasmine_total_weight (chip_weight : ℕ) (cookie_weight : ℕ) (chip_quantity : ℕ) (cookie_multiplier : ℕ) : 
  chip_weight = 20 →
  cookie_weight = 9 →
  chip_quantity = 6 →
  cookie_multiplier = 4 →
  (chip_weight * chip_quantity + cookie_weight * (cookie_multiplier * chip_quantity)) / 16 = 21 := by
  sorry

#check jasmine_total_weight

end NUMINAMATH_CALUDE_jasmine_total_weight_l3818_381807


namespace NUMINAMATH_CALUDE_spiral_notebook_cost_l3818_381873

theorem spiral_notebook_cost :
  let personal_planner_cost : ℝ := 10
  let discount_rate : ℝ := 0.2
  let total_cost_with_discount : ℝ := 112
  let spiral_notebook_cost : ℝ := 15
  (1 - discount_rate) * (4 * spiral_notebook_cost + 8 * personal_planner_cost) = total_cost_with_discount :=
by sorry

end NUMINAMATH_CALUDE_spiral_notebook_cost_l3818_381873


namespace NUMINAMATH_CALUDE_strawberry_harvest_l3818_381899

/-- Calculates the total number of strawberries harvested from a rectangular garden -/
theorem strawberry_harvest (length width : ℕ) (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) :
  length = 10 →
  width = 7 →
  plants_per_sqft = 3 →
  strawberries_per_plant = 12 →
  length * width * plants_per_sqft * strawberries_per_plant = 2520 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_harvest_l3818_381899


namespace NUMINAMATH_CALUDE_composition_of_odd_is_odd_l3818_381859

-- Define an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem composition_of_odd_is_odd (f : ℝ → ℝ) (h : is_odd_function f) :
  is_odd_function (f ∘ f) :=
by
  sorry


end NUMINAMATH_CALUDE_composition_of_odd_is_odd_l3818_381859


namespace NUMINAMATH_CALUDE_forty_fifth_even_positive_integer_l3818_381834

theorem forty_fifth_even_positive_integer :
  (fun n : ℕ => 2 * n) 45 = 90 := by sorry

end NUMINAMATH_CALUDE_forty_fifth_even_positive_integer_l3818_381834


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3818_381884

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_perimeter (t : Triangle) : 
  (2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c) →
  (t.c = Real.sqrt 7) →
  (1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3 / 2) →
  (t.a + t.b + t.c = 5 + Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3818_381884


namespace NUMINAMATH_CALUDE_total_is_245_l3818_381801

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem setup -/
def problem_setup (d : MoneyDistribution) : Prop :=
  d.y = 0.45 * d.x ∧ d.z = 0.30 * d.x ∧ d.y = 63

/-- The total amount distributed -/
def total_amount (d : MoneyDistribution) : ℝ :=
  d.x + d.y + d.z

/-- The theorem stating the total amount is 245 rupees -/
theorem total_is_245 (d : MoneyDistribution) (h : problem_setup d) : total_amount d = 245 := by
  sorry


end NUMINAMATH_CALUDE_total_is_245_l3818_381801


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l3818_381885

/-- Restaurant bill calculation -/
theorem restaurant_bill_calculation
  (appetizer_cost : ℝ)
  (num_entrees : ℕ)
  (entree_cost : ℝ)
  (tip_percentage : ℝ)
  (h1 : appetizer_cost = 10)
  (h2 : num_entrees = 4)
  (h3 : entree_cost = 20)
  (h4 : tip_percentage = 0.20) :
  appetizer_cost + num_entrees * entree_cost + (appetizer_cost + num_entrees * entree_cost) * tip_percentage = 108 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l3818_381885


namespace NUMINAMATH_CALUDE_fraction_simplification_l3818_381848

theorem fraction_simplification :
  (3 : ℚ) / (2 - 4 / 5) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3818_381848


namespace NUMINAMATH_CALUDE_extremum_point_condition_l3818_381897

open Real

theorem extremum_point_condition (a : ℝ) :
  (∀ b : ℝ, ∃! x : ℝ, x > 0 ∧ 
    (∀ y : ℝ, y > 0 → (exp (a * x) * (log x + b) ≥ exp (a * y) * (log y + b)) ∨
                      (exp (a * x) * (log x + b) ≤ exp (a * y) * (log y + b))))
  → a < 0 := by
sorry

end NUMINAMATH_CALUDE_extremum_point_condition_l3818_381897


namespace NUMINAMATH_CALUDE_trivia_game_win_probability_l3818_381861

def num_questions : ℕ := 4
def num_choices : ℕ := 4
def min_correct : ℕ := 3

def probability_correct_guess : ℚ := 1 / num_choices

def probability_win : ℚ :=
  (probability_correct_guess ^ num_questions) +
  (num_questions * (probability_correct_guess ^ (num_questions - 1)) * (1 - probability_correct_guess))

theorem trivia_game_win_probability :
  probability_win = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_win_probability_l3818_381861


namespace NUMINAMATH_CALUDE_problem_1_l3818_381831

theorem problem_1 : -2 - |(-2)| = -4 := by sorry

end NUMINAMATH_CALUDE_problem_1_l3818_381831


namespace NUMINAMATH_CALUDE_least_bench_sections_l3818_381835

/- A single bench section can hold 5 adults or 13 children -/
def adults_per_bench : ℕ := 5
def children_per_bench : ℕ := 13

/- M bench sections are connected end to end -/
def bench_sections (M : ℕ) : ℕ := M

/- An equal number of adults and children are to occupy all benches completely -/
def equal_occupancy (M : ℕ) : Prop :=
  ∃ x : ℕ, x > 0 ∧ adults_per_bench * bench_sections M = x ∧ children_per_bench * bench_sections M = x

/- The least possible positive integer value of M -/
theorem least_bench_sections : 
  ∃ M : ℕ, M > 0 ∧ equal_occupancy M ∧ ∀ N : ℕ, N > 0 → equal_occupancy N → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_least_bench_sections_l3818_381835


namespace NUMINAMATH_CALUDE_tesseract_simplex_ratio_l3818_381833

-- Define the vertices of the 4-simplex
def v₀ : Fin 4 → ℝ := λ _ => 0
def v₁ : Fin 4 → ℝ := λ i => if i.val < 2 then 1 else 0
def v₂ : Fin 4 → ℝ := λ i => if i.val = 0 ∨ i.val = 2 then 1 else 0
def v₃ : Fin 4 → ℝ := λ i => if i.val = 0 ∨ i.val = 3 then 1 else 0
def v₄ : Fin 4 → ℝ := λ i => if i.val > 0 then 1 else 0

-- Define the 4-simplex
def simplex : Fin 5 → (Fin 4 → ℝ) := λ i =>
  match i with
  | 0 => v₀
  | 1 => v₁
  | 2 => v₂
  | 3 => v₃
  | 4 => v₄

-- Define the hypervolume of a unit tesseract
def tesseract_hypervolume : ℝ := 1

-- Define the function to calculate the hypervolume of the 4-simplex
noncomputable def simplex_hypervolume : ℝ := sorry

-- State the theorem
theorem tesseract_simplex_ratio :
  tesseract_hypervolume / simplex_hypervolume = 24 / Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_tesseract_simplex_ratio_l3818_381833


namespace NUMINAMATH_CALUDE_tangent_length_to_circle_l3818_381881

def origin : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (4, 5)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (10, 25)

theorem tangent_length_to_circle (circle : Set (ℝ × ℝ)) 
  (h1 : A ∈ circle) (h2 : B ∈ circle) (h3 : C ∈ circle) :
  ∃ T ∈ circle, ‖T - origin‖ = Real.sqrt 82 ∧ 
  ∀ P ∈ circle, P ≠ T → ‖P - origin‖ > Real.sqrt 82 := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_to_circle_l3818_381881


namespace NUMINAMATH_CALUDE_birds_in_dozens_l3818_381810

def total_birds : ℕ := 96

theorem birds_in_dozens : (total_birds / 12 : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_dozens_l3818_381810


namespace NUMINAMATH_CALUDE_count_divisors_multiple_of_five_l3818_381826

/-- The number of positive divisors of 7560 that are multiples of 5 -/
def divisors_multiple_of_five : ℕ :=
  (Finset.range 4).card * (Finset.range 4).card * 1 * (Finset.range 2).card

theorem count_divisors_multiple_of_five :
  7560 = 2^3 * 3^3 * 5^1 * 7^1 →
  divisors_multiple_of_five = 32 := by
sorry

end NUMINAMATH_CALUDE_count_divisors_multiple_of_five_l3818_381826


namespace NUMINAMATH_CALUDE_no_additional_coins_needed_l3818_381847

/-- The minimum number of additional coins needed given the number of friends and initial coins. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let required_coins := num_friends * (num_friends + 1) / 2
  if initial_coins ≥ required_coins then 0
  else required_coins - initial_coins

/-- Theorem stating that for 15 friends and 120 initial coins, no additional coins are needed. -/
theorem no_additional_coins_needed :
  min_additional_coins 15 120 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_additional_coins_needed_l3818_381847


namespace NUMINAMATH_CALUDE_problem_statement_l3818_381824

def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^x < a^y

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem problem_statement (a : ℝ) (h1 : a > 0) (h2 : (p a ∨ q a) ∧ ¬(p a ∧ q a)) :
  a ≥ 4 ∨ (0 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3818_381824


namespace NUMINAMATH_CALUDE_negation_of_existence_is_universal_not_negation_of_even_prime_l3818_381828

theorem negation_of_existence_is_universal_not (P : ℕ → Prop) :
  (¬ ∃ n, P n) ↔ (∀ n, ¬ P n) := by sorry

theorem negation_of_even_prime :
  (¬ ∃ n : ℕ, Even n ∧ Prime n) ↔ (∀ n : ℕ, Even n → ¬ Prime n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_universal_not_negation_of_even_prime_l3818_381828


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_unit_cube_l3818_381840

/-- The surface area of a sphere that circumscribes a cube with edge length 1 is 3π. -/
theorem sphere_surface_area_circumscribing_unit_cube : 
  ∃ (S : ℝ), S = 3 * Real.pi ∧ 
  (∃ (r : ℝ), r > 0 ∧ 
    -- The radius is half the length of the cube's space diagonal
    r = (Real.sqrt 3) / 2 ∧ 
    -- The surface area formula
    S = 4 * Real.pi * r^2) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_unit_cube_l3818_381840


namespace NUMINAMATH_CALUDE_always_positive_expression_l3818_381863

theorem always_positive_expression (a : ℝ) : |a| + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_expression_l3818_381863


namespace NUMINAMATH_CALUDE_max_distance_between_cubic_and_quadratic_roots_l3818_381855

open Complex Set

theorem max_distance_between_cubic_and_quadratic_roots : ∃ (max_dist : ℝ),
  max_dist = 3 * Real.sqrt 7 ∧
  ∀ (a b : ℂ),
    (a^3 - 27 = 0) →
    (b^2 - 6*b + 9 = 0) →
    abs (a - b) ≤ max_dist ∧
    ∃ (a₀ b₀ : ℂ),
      (a₀^3 - 27 = 0) ∧
      (b₀^2 - 6*b₀ + 9 = 0) ∧
      abs (a₀ - b₀) = max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_cubic_and_quadratic_roots_l3818_381855


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3818_381877

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3818_381877


namespace NUMINAMATH_CALUDE_min_value_trig_function_min_value_attainable_l3818_381842

theorem min_value_trig_function (θ : Real) (h : 1 - Real.cos θ ≠ 0) :
  (2 - Real.sin θ) / (1 - Real.cos θ) ≥ 3/4 :=
by sorry

theorem min_value_attainable :
  ∃ θ : Real, (1 - Real.cos θ ≠ 0) ∧ (2 - Real.sin θ) / (1 - Real.cos θ) = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_function_min_value_attainable_l3818_381842


namespace NUMINAMATH_CALUDE_total_flowers_l3818_381819

/-- Represents the number of flowers of each color in the garden -/
structure FlowerCounts where
  orange : ℕ
  red : ℕ
  yellow : ℕ
  pink : ℕ
  purple : ℕ

/-- The conditions of the flower garden problem -/
def garden_conditions (f : FlowerCounts) : Prop :=
  f.red = 2 * f.orange ∧
  f.yellow = f.red - 5 ∧
  f.orange = 10 ∧
  f.pink = f.purple ∧
  f.pink + f.purple = 30

/-- The theorem stating the total number of flowers in the garden -/
theorem total_flowers (f : FlowerCounts) 
  (h : garden_conditions f) : 
  f.orange + f.red + f.yellow + f.pink + f.purple = 75 := by
  sorry

#check total_flowers

end NUMINAMATH_CALUDE_total_flowers_l3818_381819


namespace NUMINAMATH_CALUDE_minor_arc_circumference_l3818_381838

theorem minor_arc_circumference (r : ℝ) (angle : ℝ) :
  r = 24 →
  angle = 60 * π / 180 →
  2 * π * r * (angle / (2 * π)) = 8 * π :=
by
  sorry

end NUMINAMATH_CALUDE_minor_arc_circumference_l3818_381838


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3818_381887

theorem chess_tournament_players (total_games : ℕ) (h_total_games : total_games = 42) : 
  ∃ n : ℕ, n > 0 ∧ total_games = n * (n - 1) ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l3818_381887


namespace NUMINAMATH_CALUDE_other_number_proof_l3818_381879

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 5040)
  (h2 : Nat.gcd a b = 24)
  (h3 : a = 240) :
  b = 504 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l3818_381879


namespace NUMINAMATH_CALUDE_det_C_equals_2142_l3818_381836

theorem det_C_equals_2142 (A B C : Matrix (Fin 3) (Fin 3) ℝ) : 
  A = ![![3, 2, 5], ![0, 2, 8], ![4, 1, 7]] →
  B = ![![-2, 3, 4], ![-1, -3, 5], ![0, 4, 3]] →
  C = A * B →
  Matrix.det C = 2142 := by
  sorry

end NUMINAMATH_CALUDE_det_C_equals_2142_l3818_381836


namespace NUMINAMATH_CALUDE_sum_of_possible_n_values_l3818_381829

/-- Given natural numbers 15, 12, and n, where the product of any two is divisible by the third,
    the sum of all possible values of n is 260. -/
theorem sum_of_possible_n_values : ∃ (S : Finset ℕ),
  (∀ n ∈ S, n > 0 ∧ 
    (15 * 12) % n = 0 ∧ 
    (15 * n) % 12 = 0 ∧ 
    (12 * n) % 15 = 0) ∧
  (∀ n > 0, 
    (15 * 12) % n = 0 ∧ 
    (15 * n) % 12 = 0 ∧ 
    (12 * n) % 15 = 0 → n ∈ S) ∧
  S.sum id = 260 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_possible_n_values_l3818_381829


namespace NUMINAMATH_CALUDE_mall_profit_l3818_381804

-- Define the cost prices of type A and B
def cost_A : ℝ := 120
def cost_B : ℝ := 100

-- Define the number of units of each type
def units_A : ℝ := 50
def units_B : ℝ := 30

-- Define the conditions
axiom cost_difference : cost_A = cost_B + 20
axiom cost_equality : 5 * cost_A = 6 * cost_B
axiom total_cost : cost_A * units_A + cost_B * units_B = 9000
axiom total_units : units_A + units_B = 80

-- Define the selling prices
def sell_A : ℝ := cost_A * 1.5 * 0.8
def sell_B : ℝ := cost_B + 30

-- Define the total profit
def total_profit : ℝ := (sell_A - cost_A) * units_A + (sell_B - cost_B) * units_B

-- Theorem to prove
theorem mall_profit : 
  cost_A = 120 ∧ cost_B = 100 ∧ total_profit = 2100 :=
sorry

end NUMINAMATH_CALUDE_mall_profit_l3818_381804


namespace NUMINAMATH_CALUDE_determinant_transformation_l3818_381890

theorem determinant_transformation (a b c d : ℝ) :
  Matrix.det !![a, b; c, d] = 10 →
  Matrix.det !![a + 2*c, b + 3*d; c, d] = 10 - c*d :=
by sorry

end NUMINAMATH_CALUDE_determinant_transformation_l3818_381890


namespace NUMINAMATH_CALUDE_candy_calculation_l3818_381895

/-- 
Given the initial amount of candy, the amount eaten, and the amount received,
prove that the final amount of candy is equal to the initial amount minus
the eaten amount plus the received amount.
-/
theorem candy_calculation (initial eaten received : ℕ) :
  initial - eaten + received = (initial - eaten) + received := by
  sorry

end NUMINAMATH_CALUDE_candy_calculation_l3818_381895


namespace NUMINAMATH_CALUDE_women_decrease_l3818_381868

theorem women_decrease (initial_men : ℕ) (initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  initial_men + 2 = 14 →
  initial_women - 3 = 24 →
  initial_women - 24 = 3 := by
sorry

end NUMINAMATH_CALUDE_women_decrease_l3818_381868


namespace NUMINAMATH_CALUDE_gamma_donuts_l3818_381860

/-- Given a total of 40 donuts, with Delta taking 8 and Beta taking three times as many as Gamma,
    prove that Gamma received 8 donuts. -/
theorem gamma_donuts (total : ℕ) (delta : ℕ) (beta : ℕ) (gamma : ℕ) : 
  total = 40 → 
  delta = 8 → 
  beta = 3 * gamma → 
  total = delta + beta + gamma → 
  gamma = 8 := by
sorry

end NUMINAMATH_CALUDE_gamma_donuts_l3818_381860


namespace NUMINAMATH_CALUDE_min_value_of_f_l3818_381832

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- The theorem stating that the minimum value of f(x) is -1 -/
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3818_381832


namespace NUMINAMATH_CALUDE_max_mondays_in_45_days_l3818_381805

/-- The maximum number of Mondays in 45 consecutive days -/
def max_mondays : ℕ := 7

/-- A function that returns the day number of the nth Monday in a sequence, 
    assuming the first day is a Monday -/
def monday_sequence (n : ℕ) : ℕ := 1 + 7 * n

theorem max_mondays_in_45_days : 
  (∃ (start : ℕ), ∀ (i : ℕ), i < max_mondays → 
    start + monday_sequence i ≤ 45) ∧ 
  (∀ (start : ℕ), ∃ (i : ℕ), i = max_mondays → 
    45 < start + monday_sequence i) :=
sorry

end NUMINAMATH_CALUDE_max_mondays_in_45_days_l3818_381805


namespace NUMINAMATH_CALUDE_square_prism_sum_l3818_381844

theorem square_prism_sum (a b c d e f : ℕ+) (h : a * b * e + a * b * f + a * c * e + a * c * f + 
                                               d * b * e + d * b * f + d * c * e + d * c * f = 1176) : 
  a + b + c + d + e + f = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_prism_sum_l3818_381844


namespace NUMINAMATH_CALUDE_sunday_rounds_count_l3818_381817

/-- Proves the number of rounds completed on Sunday given the conditions of the problem -/
theorem sunday_rounds_count (round_time : ℕ) (saturday_rounds : ℕ) (total_time : ℕ) : 
  round_time = 30 →
  saturday_rounds = 11 →
  total_time = 780 →
  (total_time - saturday_rounds * round_time) / round_time = 15 := by
  sorry


end NUMINAMATH_CALUDE_sunday_rounds_count_l3818_381817


namespace NUMINAMATH_CALUDE_base8_52_equals_base10_42_l3818_381850

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ :=
  let ones := n % 10
  let eights := n / 10
  eights * 8 + ones

/-- The base-8 number 52 is equal to the base-10 number 42 --/
theorem base8_52_equals_base10_42 : base8ToBase10 52 = 42 := by
  sorry

end NUMINAMATH_CALUDE_base8_52_equals_base10_42_l3818_381850


namespace NUMINAMATH_CALUDE_prob_same_student_given_same_look_l3818_381894

/-- Represents a group of identical students -/
structure IdenticalGroup where
  size : Nat
  count : Nat

/-- Represents the Multiples Obfuscation Program -/
def MultiplesObfuscationProgram : List IdenticalGroup :=
  [⟨1, 1⟩, ⟨2, 1⟩, ⟨3, 1⟩, ⟨4, 1⟩, ⟨5, 1⟩, ⟨6, 1⟩, ⟨7, 1⟩, ⟨8, 1⟩]

/-- Total number of students in the program -/
def totalStudents : Nat :=
  MultiplesObfuscationProgram.foldr (fun g acc => g.size * g.count + acc) 0

/-- Number of pairs where students look the same -/
def sameLookPairs : Nat :=
  MultiplesObfuscationProgram.foldr (fun g acc => g.size * g.size * g.count + acc) 0

/-- Probability of encountering the same student twice -/
def probSameStudent : Rat :=
  totalStudents / (totalStudents * totalStudents)

/-- Probability of encountering students that look the same -/
def probSameLook : Rat :=
  sameLookPairs / (totalStudents * totalStudents)

theorem prob_same_student_given_same_look :
  probSameStudent / probSameLook = 3 / 17 := by sorry

end NUMINAMATH_CALUDE_prob_same_student_given_same_look_l3818_381894


namespace NUMINAMATH_CALUDE_same_solution_c_value_l3818_381867

theorem same_solution_c_value (x : ℚ) (c : ℚ) : 
  (3 * x + 5 = 1) ∧ (c * x + 8 = 6) → c = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_c_value_l3818_381867


namespace NUMINAMATH_CALUDE_solve_equation_l3818_381813

theorem solve_equation (k l x : ℚ) 
  (eq1 : 3/4 = k/88)
  (eq2 : 3/4 = (k+l)/120)
  (eq3 : 3/4 = (x-l)/160) : 
  x = 144 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3818_381813


namespace NUMINAMATH_CALUDE_sqrt_10_between_3_and_4_l3818_381822

theorem sqrt_10_between_3_and_4 : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_10_between_3_and_4_l3818_381822


namespace NUMINAMATH_CALUDE_correct_equation_l3818_381815

theorem correct_equation : 500 - 9 * 7 = 437 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l3818_381815


namespace NUMINAMATH_CALUDE_womens_doubles_handshakes_l3818_381892

/-- Calculate the number of handshakes in a tournament -/
def tournament_handshakes (n : ℕ) : ℕ :=
  (n * (n - 2)) / 2

/-- Theorem: In a women's doubles tennis tournament with 4 teams (8 players),
    the total number of handshakes is 24 -/
theorem womens_doubles_handshakes :
  tournament_handshakes 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_womens_doubles_handshakes_l3818_381892


namespace NUMINAMATH_CALUDE_check_error_l3818_381816

theorem check_error (x y : ℤ) 
  (h1 : 10 ≤ x ∧ x ≤ 99) 
  (h2 : 10 ≤ y ∧ y ≤ 99) 
  (h3 : 100 * y + x - (100 * x + y) = 2046) 
  (h4 : x = (3 * y) / 2) : 
  x = 66 := by sorry

end NUMINAMATH_CALUDE_check_error_l3818_381816


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3818_381820

theorem min_value_expression (x : ℝ) (h : x > 0) : x^2 + 8*x + 64/x^3 ≥ 28 := by
  sorry

theorem equality_condition : ∃ x > 0, x^2 + 8*x + 64/x^3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3818_381820


namespace NUMINAMATH_CALUDE_three_consecutive_not_divisible_by_three_l3818_381811

def digit_sum (n : ℕ) : ℕ := sorry

def board_sequence (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => board_sequence initial n + digit_sum (board_sequence initial n)

theorem three_consecutive_not_divisible_by_three (initial : ℕ) :
  ∃ k : ℕ, ¬(board_sequence initial k % 3 = 0) ∧
           ¬(board_sequence initial (k + 1) % 3 = 0) ∧
           ¬(board_sequence initial (k + 2) % 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_three_consecutive_not_divisible_by_three_l3818_381811


namespace NUMINAMATH_CALUDE_smallest_a_value_l3818_381864

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.cos (a * ↑x + b) = Real.sin (36 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, ∃ b' ≥ 0, Real.cos (a' * ↑x + b') = Real.sin (36 * ↑x)) → a' ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3818_381864


namespace NUMINAMATH_CALUDE_starting_lineup_count_l3818_381846

def team_size : ℕ := 12
def lineup_size : ℕ := 5
def non_captain_size : ℕ := lineup_size - 1

theorem starting_lineup_count : 
  team_size * (Nat.choose (team_size - 1) non_captain_size) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l3818_381846


namespace NUMINAMATH_CALUDE_division_problem_l3818_381843

theorem division_problem :
  ∃ (quotient : ℕ),
    15968 = 179 * quotient + 37 ∧
    quotient = 89 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3818_381843


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l3818_381882

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_perimeter_range (t : Triangle) 
  (h1 : Real.sin (3 * t.B / 2 + π / 4) = Real.sqrt 2 / 2)
  (h2 : t.a + t.c = 2) :
  3 ≤ t.a + t.b + t.c ∧ t.a + t.b + t.c < 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l3818_381882


namespace NUMINAMATH_CALUDE_single_working_day_between_holidays_l3818_381876

def is_holiday (n : ℕ) : Prop := n % 6 = 0 ∨ Nat.Prime n

def working_day_between_holidays (n : ℕ) : Prop :=
  n > 1 ∧ n < 40 ∧ is_holiday (n - 1) ∧ ¬is_holiday n ∧ is_holiday (n + 1)

theorem single_working_day_between_holidays :
  ∃! n, working_day_between_holidays n :=
sorry

end NUMINAMATH_CALUDE_single_working_day_between_holidays_l3818_381876


namespace NUMINAMATH_CALUDE_parallelogram_adjacent_side_l3818_381845

/-- The length of the other adjacent side of a parallelogram with perimeter 16 and one side length 5 is 3. -/
theorem parallelogram_adjacent_side (perimeter : ℝ) (side_a : ℝ) (side_b : ℝ) 
  (h1 : perimeter = 16) 
  (h2 : side_a = 5) 
  (h3 : perimeter = 2 * (side_a + side_b)) : 
  side_b = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_adjacent_side_l3818_381845


namespace NUMINAMATH_CALUDE_symmetric_point_example_l3818_381818

/-- Given a point A and a line L, find the point B symmetric to A about L -/
def symmetric_point (A : ℝ × ℝ) (L : ℝ → ℝ → ℝ) : ℝ × ℝ :=
  sorry

/-- The line 2x - 4y + 9 = 0 -/
def line (x y : ℝ) : ℝ := 2 * x - 4 * y + 9

theorem symmetric_point_example :
  symmetric_point (2, 2) line = (1, 4) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_example_l3818_381818


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3818_381825

theorem simplify_trig_expression :
  (Real.sin (15 * π / 180) + Real.sin (45 * π / 180)) /
  (Real.cos (15 * π / 180) + Real.cos (45 * π / 180)) =
  Real.tan (30 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3818_381825


namespace NUMINAMATH_CALUDE_cliffs_rock_collection_l3818_381841

theorem cliffs_rock_collection (igneous_rocks sedimentary_rocks : ℕ) : 
  igneous_rocks = sedimentary_rocks / 2 →
  igneous_rocks / 3 = 30 →
  igneous_rocks + sedimentary_rocks = 270 := by
  sorry

end NUMINAMATH_CALUDE_cliffs_rock_collection_l3818_381841


namespace NUMINAMATH_CALUDE_vending_machine_problem_l3818_381857

/-- The probability of the vending machine failing to drop a snack -/
def fail_prob : ℚ := 1 / 6

/-- The probability of the vending machine dropping two snacks -/
def double_prob : ℚ := 1 / 10

/-- The probability of the vending machine dropping one snack -/
def single_prob : ℚ := 1 - fail_prob - double_prob

/-- The expected number of snacks dropped per person -/
def expected_snacks : ℚ := fail_prob * 0 + double_prob * 2 + single_prob * 1

/-- The total number of snacks dropped -/
def total_snacks : ℕ := 28

/-- The number of people who have used the vending machine -/
def num_people : ℕ := 30

theorem vending_machine_problem :
  (↑num_people : ℚ) * expected_snacks = ↑total_snacks :=
sorry

end NUMINAMATH_CALUDE_vending_machine_problem_l3818_381857


namespace NUMINAMATH_CALUDE_symmetry_x_axis_example_l3818_381853

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis in 3D space -/
def symmetry_x_axis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

/-- The theorem stating that the point symmetric to (-2, 1, 5) with respect to the x-axis
    has coordinates (-2, -1, -5) -/
theorem symmetry_x_axis_example : 
  symmetry_x_axis { x := -2, y := 1, z := 5 } = { x := -2, y := -1, z := -5 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_x_axis_example_l3818_381853


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l3818_381837

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l3818_381837


namespace NUMINAMATH_CALUDE_store_inventory_l3818_381821

theorem store_inventory (ties belts black_shirts : ℕ) 
  (h1 : ties = 34)
  (h2 : belts = 40)
  (h3 : black_shirts = 63)
  (h4 : ∃ white_shirts : ℕ, 
    ∃ jeans : ℕ, 
    ∃ scarves : ℕ,
    jeans = (2 * (black_shirts + white_shirts)) / 3 ∧
    scarves = (ties + belts) / 2 ∧
    jeans = scarves + 33) :
  ∃ white_shirts : ℕ, white_shirts = 42 := by
sorry

end NUMINAMATH_CALUDE_store_inventory_l3818_381821


namespace NUMINAMATH_CALUDE_fraction_equality_implies_x_value_l3818_381874

theorem fraction_equality_implies_x_value :
  ∀ x : ℚ, (x + 6) / (x - 4) = (x - 7) / (x + 2) → x = 16 / 19 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_x_value_l3818_381874


namespace NUMINAMATH_CALUDE_output_is_76_l3818_381856

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 ≤ 40 then step1 + 10 else step1 - 7
  step2 * 2

theorem output_is_76 : function_machine 15 = 76 := by
  sorry

end NUMINAMATH_CALUDE_output_is_76_l3818_381856


namespace NUMINAMATH_CALUDE_g_range_l3818_381898

noncomputable def g (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 11 * Real.sin x ^ 2 + 3 * Real.sin x + 4 * Real.cos x ^ 2 - 10) / (Real.sin x - 2)

theorem g_range : 
  ∀ y ∈ Set.range g, 1 ≤ y ∧ y ≤ 19 ∧ 
  ∃ x : ℝ, g x = 1 ∧ 
  ∃ x : ℝ, g x = 19 :=
sorry

end NUMINAMATH_CALUDE_g_range_l3818_381898


namespace NUMINAMATH_CALUDE_parents_in_auditorium_l3818_381806

/-- Given a school play with girls and boys, and both parents of each kid attending,
    calculate the total number of parents in the auditorium. -/
theorem parents_in_auditorium (girls boys : ℕ) (h1 : girls = 6) (h2 : boys = 8) :
  2 * (girls + boys) = 28 := by
  sorry

end NUMINAMATH_CALUDE_parents_in_auditorium_l3818_381806


namespace NUMINAMATH_CALUDE_one_and_one_third_of_number_is_45_l3818_381891

theorem one_and_one_third_of_number_is_45 :
  ∃ x : ℚ, (4 : ℚ) / 3 * x = 45 ∧ x = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_one_and_one_third_of_number_is_45_l3818_381891


namespace NUMINAMATH_CALUDE_infinite_nested_radical_twenty_l3818_381878

theorem infinite_nested_radical_twenty : ∃! (x : ℝ), x > 0 ∧ x = Real.sqrt (20 + x) ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_infinite_nested_radical_twenty_l3818_381878


namespace NUMINAMATH_CALUDE_total_notes_count_l3818_381888

/-- Given a total amount of 400 rupees in equal numbers of one-rupee, five-rupee, and ten-rupee notes,
    prove that the total number of notes is 75. -/
theorem total_notes_count (total_amount : ℕ) (n : ℕ) 
  (h1 : total_amount = 400)
  (h2 : n * 1 + n * 5 + n * 10 = total_amount) : 
  3 * n = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_notes_count_l3818_381888


namespace NUMINAMATH_CALUDE_beatrice_has_highest_answer_l3818_381852

def albert_calculation (x : ℕ) : ℕ := 2 * ((3 * x + 5) - 3)

def beatrice_calculation (x : ℕ) : ℕ := 2 * ((x * x + 3) - 7)

def carlos_calculation (x : ℕ) : ℚ := ((5 * x - 4 + 6) : ℚ) / 2

theorem beatrice_has_highest_answer :
  let start := 15
  beatrice_calculation start > albert_calculation start ∧
  (beatrice_calculation start : ℚ) > carlos_calculation start := by
sorry

#eval albert_calculation 15
#eval beatrice_calculation 15
#eval carlos_calculation 15

end NUMINAMATH_CALUDE_beatrice_has_highest_answer_l3818_381852


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3818_381883

/-- A line in the form kx - y + 1 = 3k passes through the point (3, 1) for all values of k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), k * 3 - 1 + 1 = 3 * k := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3818_381883


namespace NUMINAMATH_CALUDE_simona_treatment_cost_l3818_381812

/-- Represents the number of complexes after each treatment -/
def complexes_after_treatment (initial : ℕ) : ℕ → ℕ
| 0 => initial
| (n + 1) => (complexes_after_treatment initial n / 2) + ((complexes_after_treatment initial n + 1) / 2)

/-- The cost of treatment for a given number of cured complexes -/
def treatment_cost (cured_complexes : ℕ) : ℕ := 197 * cured_complexes

theorem simona_treatment_cost :
  ∃ (initial : ℕ),
    initial > 0 ∧
    complexes_after_treatment initial 3 = 1 ∧
    treatment_cost (initial - 1) = 1379 :=
by sorry

end NUMINAMATH_CALUDE_simona_treatment_cost_l3818_381812


namespace NUMINAMATH_CALUDE_probability_at_least_four_mismatched_l3818_381893

/-- The number of derangements for n elements -/
def derangement (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | n + 2 => (n + 1) * (derangement (n + 1) + derangement n)

/-- The number of students and subjects -/
def num_students : ℕ := 5

/-- The probability of at least 4 out of 5 students receiving a mismatched test paper -/
def probability_mismatched_tests : ℚ :=
  (num_students * derangement (num_students - 1) + derangement num_students) / (num_students.factorial)

theorem probability_at_least_four_mismatched :
  probability_mismatched_tests = 89 / 120 := by
  sorry


end NUMINAMATH_CALUDE_probability_at_least_four_mismatched_l3818_381893


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt10_l3818_381886

theorem sqrt_sum_equals_2sqrt10 : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt10_l3818_381886


namespace NUMINAMATH_CALUDE_min_value_of_m_plus_2n_l3818_381803

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem min_value_of_m_plus_2n (a : ℝ) (m n : ℝ) :
  (∀ x, f a x ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3) →
  (m > 0) →
  (n > 0) →
  (1 / m + 1 / (2 * n) = a) →
  (∀ p q, p > 0 → q > 0 → 1 / p + 1 / (2 * q) = a → p + 2 * q ≥ m + 2 * n) →
  m + 2 * n = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_m_plus_2n_l3818_381803


namespace NUMINAMATH_CALUDE_chicken_count_l3818_381870

/-- The number of rabbits on the farm -/
def rabbits : ℕ := 49

/-- The number of frogs on the farm -/
def frogs : ℕ := 37

/-- The number of chickens on the farm -/
def chickens : ℕ := 21

/-- The total number of frogs and chickens is 9 more than the number of rabbits -/
axiom farm_equation : frogs + chickens = rabbits + 9

theorem chicken_count : chickens = 21 := by
  sorry

end NUMINAMATH_CALUDE_chicken_count_l3818_381870


namespace NUMINAMATH_CALUDE_floor_sum_abcd_l3818_381827

theorem floor_sum_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + b^2 = 1458) (h2 : c^2 + d^2 = 1458) (h3 : a * c = 1156) (h4 : b * d = 1156) :
  ⌊a + b + c + d⌋ = 77 := by sorry

end NUMINAMATH_CALUDE_floor_sum_abcd_l3818_381827


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3818_381872

theorem inequality_system_solution (x : ℝ) : 
  ((-x + 3) / 2 < x) ∧ (2 * (x + 6) ≥ 5 * x) → 1 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3818_381872


namespace NUMINAMATH_CALUDE_book_pages_count_l3818_381849

/-- The number of pages Frank reads per day -/
def pages_per_day : ℕ := 22

/-- The number of days it took Frank to finish the book -/
def days_to_finish : ℕ := 569

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_day * days_to_finish

theorem book_pages_count : total_pages = 12518 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l3818_381849


namespace NUMINAMATH_CALUDE_f_minus_six_equals_minus_one_l3818_381896

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_minus_six_equals_minus_one 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : is_even f)
  (h2 : has_period f 6)
  (h3 : ∀ x ∈ Set.Icc (-3) 3, f x = (x + 1) * (x - a)) :
  f (-6) = -1 := by
sorry

end NUMINAMATH_CALUDE_f_minus_six_equals_minus_one_l3818_381896


namespace NUMINAMATH_CALUDE_amelia_win_probability_l3818_381858

/-- Probability of Amelia winning the coin toss game -/
theorem amelia_win_probability (amelia_heads_prob : ℚ) (blaine_heads_prob : ℚ)
  (h_amelia : amelia_heads_prob = 1/4)
  (h_blaine : blaine_heads_prob = 3/7) :
  let p := amelia_heads_prob + (1 - amelia_heads_prob) * (1 - blaine_heads_prob) * p
  p = 7/16 := by sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l3818_381858


namespace NUMINAMATH_CALUDE_round_table_knights_and_liars_l3818_381869

theorem round_table_knights_and_liars (n : ℕ) (K : ℕ) : 
  n > 1000 →
  n = K + (n - K) →
  (∀ i : ℕ, i < n → (20 * K) % n = 0) →
  (∀ m : ℕ, m > 1000 → (20 * K) % m = 0 → m ≥ n) →
  n = 1020 :=
sorry

end NUMINAMATH_CALUDE_round_table_knights_and_liars_l3818_381869


namespace NUMINAMATH_CALUDE_sector_central_angle_l3818_381808

theorem sector_central_angle (r : ℝ) (A : ℝ) (θ : ℝ) : 
  r = 2 → A = 4 → A = (1/2) * r^2 * θ → θ = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3818_381808


namespace NUMINAMATH_CALUDE_solve_for_d_l3818_381862

theorem solve_for_d (y : ℝ) (d : ℝ) (h1 : y > 0) (h2 : (6 * y) / 20 + (3 * y) / d = 0.60 * y) : d = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_d_l3818_381862


namespace NUMINAMATH_CALUDE_case_cost_l3818_381871

theorem case_cost (pen ink case : ℝ) 
  (total_cost : pen + ink + case = 2.30)
  (pen_cost : pen = ink + 1.50)
  (case_cost : case = 0.5 * ink) :
  case = 0.1335 := by
sorry

end NUMINAMATH_CALUDE_case_cost_l3818_381871


namespace NUMINAMATH_CALUDE_account_balance_difference_l3818_381830

/-- Computes the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Computes the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- The difference between two account balances after 25 years -/
theorem account_balance_difference : 
  let jessica_balance := compound_interest 12000 0.025 50
  let mark_balance := simple_interest 15000 0.06 25
  ∃ ε > 0, abs (jessica_balance - mark_balance - 3136) < ε :=
sorry

end NUMINAMATH_CALUDE_account_balance_difference_l3818_381830


namespace NUMINAMATH_CALUDE_largest_common_number_l3818_381800

theorem largest_common_number (n m : ℕ) : 
  67 = 1 + 6 * n ∧ 
  67 = 4 + 7 * m ∧ 
  67 ≤ 100 ∧ 
  ∀ k, (∃ p q : ℕ, k = 1 + 6 * p ∧ k = 4 + 7 * q ∧ k ≤ 100) → k ≤ 67 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_number_l3818_381800


namespace NUMINAMATH_CALUDE_combined_population_l3818_381839

/-- The combined population of Port Perry and Lazy Harbor given the specified conditions -/
theorem combined_population (wellington_pop : ℕ) (port_perry_pop : ℕ) (lazy_harbor_pop : ℕ) 
  (h1 : port_perry_pop = 7 * wellington_pop)
  (h2 : port_perry_pop = lazy_harbor_pop + 800)
  (h3 : wellington_pop = 900) : 
  port_perry_pop + lazy_harbor_pop = 11800 := by
  sorry

end NUMINAMATH_CALUDE_combined_population_l3818_381839


namespace NUMINAMATH_CALUDE_married_couples_children_l3818_381865

/-- The fraction of married couples with more than one child -/
def fraction_more_than_one_child : ℚ := 3/5

/-- The fraction of married couples with more than 3 children -/
def fraction_more_than_three_children : ℚ := 2/5

/-- The fraction of married couples with 2 or 3 children -/
def fraction_two_or_three_children : ℚ := 1/5

theorem married_couples_children :
  fraction_more_than_one_child = 
    fraction_more_than_three_children + fraction_two_or_three_children :=
by sorry

end NUMINAMATH_CALUDE_married_couples_children_l3818_381865


namespace NUMINAMATH_CALUDE_system_solution_l3818_381851

theorem system_solution : 
  ∀ x y : ℝ, 
    (x + y = (7 - x) + (7 - y) ∧ 
     x^2 - y = (x - 2) + (y - 2)) ↔ 
    ((x = -5 ∧ y = 12) ∨ (x = 2 ∧ y = 5)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3818_381851


namespace NUMINAMATH_CALUDE_show_length_l3818_381880

/-- Proves that the length of each show is 50 minutes given the conditions -/
theorem show_length (gina_choice_ratio : ℝ) (total_shows : ℕ) (gina_minutes : ℝ) 
  (h1 : gina_choice_ratio = 3)
  (h2 : total_shows = 24)
  (h3 : gina_minutes = 900) : 
  (gina_minutes / (gina_choice_ratio * total_shows / (gina_choice_ratio + 1))) = 50 := by
  sorry


end NUMINAMATH_CALUDE_show_length_l3818_381880


namespace NUMINAMATH_CALUDE_inequality_proof_l3818_381823

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  (a^2 + b^2 + c^2 + d) / (a + b + c)^3 + 
  (b^2 + c^2 + d^2 + a) / (b + c + d)^3 + 
  (c^2 + d^2 + a^2 + b) / (c + d + a)^3 + 
  (d^2 + a^2 + b^2 + c) / (d + a + b)^3 > 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3818_381823
