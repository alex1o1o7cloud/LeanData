import Mathlib

namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l3564_356413

-- Define the quadratic equation
def quadratic_equation (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- Define the roots from the first student's mistake
def root1 : ℝ := 3
def root2 : ℝ := 7

-- Define the roots from the second student's mistake
def root3 : ℝ := 5
def root4 : ℝ := -1

-- Theorem statement
theorem correct_quadratic_equation :
  ∃ (b c : ℝ),
    (root1 + root2 = -b) ∧
    (root3 * root4 = c) ∧
    (∀ x, quadratic_equation b c x = x^2 - 10*x - 5) :=
sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l3564_356413


namespace NUMINAMATH_CALUDE_nest_distance_building_materials_distance_l3564_356404

/-- Given two birds making round trips to collect building materials, 
    calculate the distance from the nest to the materials. -/
theorem nest_distance (num_birds : ℕ) (num_trips : ℕ) (total_distance : ℝ) : ℝ :=
  let distance_per_bird := total_distance / num_birds
  let distance_per_trip := distance_per_bird / num_trips
  distance_per_trip / 4

/-- Prove that for two birds making 10 round trips each, 
    with a total distance of 8000 miles, 
    the building materials are 100 miles from the nest. -/
theorem building_materials_distance : 
  nest_distance 2 10 8000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_nest_distance_building_materials_distance_l3564_356404


namespace NUMINAMATH_CALUDE_power_of_half_l3564_356472

theorem power_of_half (some_power k : ℕ) : 
  (1/2 : ℝ)^some_power * (1/81 : ℝ)^k = 1/(18^16 : ℝ) → 
  k = 8 → 
  some_power = 16 := by
sorry

end NUMINAMATH_CALUDE_power_of_half_l3564_356472


namespace NUMINAMATH_CALUDE_quiz_statistics_l3564_356412

def scores : List ℕ := [7, 5, 6, 8, 7, 9]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem quiz_statistics :
  mean scores = 7 ∧ mode scores = 7 := by
  sorry

end NUMINAMATH_CALUDE_quiz_statistics_l3564_356412


namespace NUMINAMATH_CALUDE_apple_distribution_l3564_356426

theorem apple_distribution (total_apples : ℕ) (new_people : ℕ) (apple_decrease : ℕ) :
  total_apples = 1430 →
  new_people = 45 →
  apple_decrease = 9 →
  ∃ (original_people : ℕ),
    original_people > 0 ∧
    (total_apples / original_people : ℚ) - (total_apples / (original_people + new_people) : ℚ) = apple_decrease ∧
    total_apples / original_people = 22 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l3564_356426


namespace NUMINAMATH_CALUDE_solve_jerichos_money_problem_l3564_356490

def jerichos_money_problem (initial_amount debt_to_annika : ℕ) : Prop :=
  let debt_to_manny := 2 * debt_to_annika
  let total_debt := debt_to_annika + debt_to_manny
  let remaining_amount := initial_amount - total_debt
  (initial_amount = 3 * 90) ∧ 
  (debt_to_annika = 20) ∧
  (remaining_amount = 210)

theorem solve_jerichos_money_problem :
  jerichos_money_problem 270 20 := by sorry

end NUMINAMATH_CALUDE_solve_jerichos_money_problem_l3564_356490


namespace NUMINAMATH_CALUDE_fraction_equality_l3564_356471

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3564_356471


namespace NUMINAMATH_CALUDE_inequality_addition_l3564_356407

theorem inequality_addition (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l3564_356407


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_7n_plus_2_l3564_356446

theorem max_gcd_13n_plus_4_7n_plus_2 :
  (∃ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) = 2) ∧
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_7n_plus_2_l3564_356446


namespace NUMINAMATH_CALUDE_faye_initial_apps_l3564_356465

/-- Represents the number of apps Faye had initially -/
def initial_apps : ℕ := sorry

/-- Represents the number of apps Faye deleted -/
def deleted_apps : ℕ := 8

/-- Represents the number of apps Faye had left after deleting -/
def remaining_apps : ℕ := 4

/-- Theorem stating that the initial number of apps was 12 -/
theorem faye_initial_apps : initial_apps = 12 := by
  sorry

end NUMINAMATH_CALUDE_faye_initial_apps_l3564_356465


namespace NUMINAMATH_CALUDE_wyatt_orange_juice_purchase_l3564_356430

def orange_juice_cartons (initial_money : ℕ) (bread_loaves : ℕ) (bread_cost : ℕ) (juice_cost : ℕ) (remaining_money : ℕ) : ℕ :=
  (initial_money - remaining_money - bread_loaves * bread_cost) / juice_cost

theorem wyatt_orange_juice_purchase :
  orange_juice_cartons 74 5 5 2 41 = 4 := by
  sorry

end NUMINAMATH_CALUDE_wyatt_orange_juice_purchase_l3564_356430


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_for_two_l3564_356464

theorem arithmetic_geometric_mean_inequality_for_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt x + Real.sqrt y) / 2 ≤ Real.sqrt ((x + y) / 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_for_two_l3564_356464


namespace NUMINAMATH_CALUDE_max_log_sum_l3564_356416

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  ∃ (max : ℝ), max = Real.log 4 ∧ ∀ z w : ℝ, z > 0 → w > 0 → z + w = 4 → Real.log z + Real.log w ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_log_sum_l3564_356416


namespace NUMINAMATH_CALUDE_periodic_coloring_divides_l3564_356455

/-- A coloring of the integers -/
def Coloring := ℤ → Bool

/-- A coloring is t-periodic if it repeats every t steps -/
def isPeriodic (c : Coloring) (t : ℕ) : Prop :=
  ∀ x : ℤ, c x = c (x + t)

/-- For a given x, exactly one of x + a₁, ..., x + aₙ is colored -/
def hasUniqueColoredSum (c : Coloring) (a : Fin n → ℕ) : Prop :=
  ∀ x : ℤ, ∃! i : Fin n, c (x + a i)

theorem periodic_coloring_divides (n : ℕ) (t : ℕ) (a : Fin n → ℕ) (h_a : StrictMono a) 
    (c : Coloring) (h_periodic : isPeriodic c t) (h_unique : hasUniqueColoredSum c a) : 
    n ∣ t := by sorry

end NUMINAMATH_CALUDE_periodic_coloring_divides_l3564_356455


namespace NUMINAMATH_CALUDE_ellipse_intersection_ratio_l3564_356434

/-- Given an ellipse mx^2 + ny^2 = 1 intersecting with a line y = -x + 1,
    if a line through the origin and the midpoint of the intersection points
    has slope √2/2, then n/m = √2 -/
theorem ellipse_intersection_ratio (m n : ℝ) (h_pos : m > 0 ∧ n > 0) :
  (∃ A B : ℝ × ℝ,
    (m * A.1^2 + n * A.2^2 = 1) ∧
    (m * B.1^2 + n * B.2^2 = 1) ∧
    (A.2 = -A.1 + 1) ∧
    (B.2 = -B.1 + 1) ∧
    ((A.2 + B.2) / (A.1 + B.1) = Real.sqrt 2 / 2)) →
  n / m = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_ratio_l3564_356434


namespace NUMINAMATH_CALUDE_min_cuts_for_cube_division_l3564_356454

/-- Represents a three-dimensional cube --/
structure Cube where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the process of cutting a cube --/
def cut_cube (initial : Cube) (final_size : ℕ) (allow_rearrange : Bool) : ℕ :=
  sorry

/-- Theorem: The minimum number of cuts to divide a 3x3x3 cube into 27 1x1x1 cubes is 6 --/
theorem min_cuts_for_cube_division :
  let initial_cube : Cube := ⟨3, 3, 3⟩
  let final_size : ℕ := 1
  let num_final_cubes : ℕ := 27
  let allow_rearrange : Bool := true
  (cut_cube initial_cube final_size allow_rearrange = 6) ∧
  (∀ n : ℕ, n < 6 → cut_cube initial_cube final_size allow_rearrange ≠ n) :=
by sorry

end NUMINAMATH_CALUDE_min_cuts_for_cube_division_l3564_356454


namespace NUMINAMATH_CALUDE_negation_equivalence_l3564_356445

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 + Real.sin x < 0)) ↔ (∀ x : ℝ, x > 0 → x^2 + Real.sin x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3564_356445


namespace NUMINAMATH_CALUDE_total_fruits_is_54_l3564_356477

/-- The total number of fruits picked by all people -/
def total_fruits (melanie_plums melanie_apples dan_plums dan_oranges sally_plums sally_cherries thomas_plums thomas_peaches : ℕ) : ℕ :=
  melanie_plums + melanie_apples + dan_plums + dan_oranges + sally_plums + sally_cherries + thomas_plums + thomas_peaches

/-- Theorem stating that the total number of fruits picked is 54 -/
theorem total_fruits_is_54 :
  total_fruits 4 6 9 2 3 10 15 5 = 54 := by
  sorry

#eval total_fruits 4 6 9 2 3 10 15 5

end NUMINAMATH_CALUDE_total_fruits_is_54_l3564_356477


namespace NUMINAMATH_CALUDE_correct_outfit_count_l3564_356482

/-- The number of shirts -/
def num_shirts : ℕ := 5

/-- The number of pants -/
def num_pants : ℕ := 6

/-- The number of formal pants -/
def num_formal_pants : ℕ := 3

/-- The number of casual pants -/
def num_casual_pants : ℕ := num_pants - num_formal_pants

/-- The number of shirts that can be paired with formal pants -/
def num_shirts_for_formal : ℕ := 3

/-- Calculate the number of different outfits -/
def num_outfits : ℕ :=
  (num_casual_pants * num_shirts) + (num_formal_pants * num_shirts_for_formal)

theorem correct_outfit_count : num_outfits = 24 := by
  sorry

end NUMINAMATH_CALUDE_correct_outfit_count_l3564_356482


namespace NUMINAMATH_CALUDE_bridge_length_l3564_356421

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3564_356421


namespace NUMINAMATH_CALUDE_total_time_two_trips_l3564_356489

/-- Represents the time in minutes for a round trip to the beauty parlor -/
structure RoundTrip where
  to_parlor : ℕ
  from_parlor : ℕ
  delay : ℕ
  additional_time : ℕ

/-- Calculates the total time for a round trip -/
def total_time (trip : RoundTrip) : ℕ :=
  trip.to_parlor + trip.from_parlor + trip.delay + trip.additional_time

/-- Represents Naomi's two round trips to the beauty parlor -/
def naomi_trips : (RoundTrip × RoundTrip) :=
  ({ to_parlor := 60
   , from_parlor := 120
   , delay := 15
   , additional_time := 10 }
  ,{ to_parlor := 60
   , from_parlor := 120
   , delay := 20
   , additional_time := 30 })

/-- Theorem stating that the total time for both round trips is 435 minutes -/
theorem total_time_two_trips : 
  total_time naomi_trips.1 + total_time naomi_trips.2 = 435 := by
  sorry

end NUMINAMATH_CALUDE_total_time_two_trips_l3564_356489


namespace NUMINAMATH_CALUDE_max_non_managers_l3564_356469

theorem max_non_managers (managers : ℕ) (ratio_managers : ℕ) (ratio_non_managers : ℕ) :
  managers = 11 →
  ratio_managers = 7 →
  ratio_non_managers = 37 →
  ∀ non_managers : ℕ,
    (managers : ℚ) / (non_managers : ℚ) > (ratio_managers : ℚ) / (ratio_non_managers : ℚ) →
    non_managers ≤ 58 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l3564_356469


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3564_356473

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- Our specific arithmetic sequence satisfying a₃ + a₈ = 6 -/
def our_sequence (a : ℕ → ℝ) : Prop :=
  arithmetic_sequence a ∧ a 3 + a 8 = 6

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : our_sequence a) :
  3 * a 2 + a 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3564_356473


namespace NUMINAMATH_CALUDE_matrix_product_and_curve_transformation_l3564_356484

/-- Given two 2×2 matrices A and B with specific properties, prove statements about their product and a curve transformation. -/
theorem matrix_product_and_curve_transformation :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, a; b, 2]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-1, 2; 1/2, 1]
  ∀ a b c d : ℝ,
  A * B = !![c, 2; 3, d] →
  (a = 0 ∧ b = -2 ∧ c = -1 ∧ d = -2) ∧
  (∀ x y : ℝ, 2 * x^2 - 2 * x * y + 1 = 0 →
              ∃ x' y' : ℝ, x' = x ∧ y' = -2 * x' + 2 * y' ∧ x' * y' = 1) :=
by sorry

end NUMINAMATH_CALUDE_matrix_product_and_curve_transformation_l3564_356484


namespace NUMINAMATH_CALUDE_M_always_positive_l3564_356439

theorem M_always_positive (x y : ℝ) : 3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13 > 0 := by
  sorry

end NUMINAMATH_CALUDE_M_always_positive_l3564_356439


namespace NUMINAMATH_CALUDE_pqr_value_l3564_356428

theorem pqr_value (p q r : ℂ) 
  (eq1 : p * q + 5 * q = -20)
  (eq2 : q * r + 5 * r = -20)
  (eq3 : r * p + 5 * p = -20) :
  p * q * r = 80 := by
sorry

end NUMINAMATH_CALUDE_pqr_value_l3564_356428


namespace NUMINAMATH_CALUDE_min_value_cyclic_fraction_sum_l3564_356453

theorem min_value_cyclic_fraction_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a / b + b / c + c / d + d / a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cyclic_fraction_sum_l3564_356453


namespace NUMINAMATH_CALUDE_f_properties_l3564_356488

noncomputable def f (x : ℝ) := Real.cos (2 * x) + 2 * Real.sin x * Real.sin x

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q) ∧
  (∃ M : ℝ, ∀ x, f x ≤ M ∧ ∃ x, f x = M) ∧
  (∀ k : ℤ, f (k * Real.pi) = 2) ∧
  (∀ A : ℝ, A > 0 ∧ A < Real.pi / 2 →
    f A = 0 →
    ∀ b a : ℝ, b = 5 ∧ a = 7 →
    ∃ c : ℝ, c > 0 ∧
    (1/2) * b * c * Real.sin A = 10) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3564_356488


namespace NUMINAMATH_CALUDE_chemistry_marks_proof_l3564_356433

def english_marks : ℕ := 91
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def biology_marks : ℕ := 85
def average_marks : ℕ := 78
def total_subjects : ℕ := 5

theorem chemistry_marks_proof :
  ∃ (chemistry_marks : ℕ),
    (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks) / total_subjects = average_marks ∧
    chemistry_marks = 67 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_marks_proof_l3564_356433


namespace NUMINAMATH_CALUDE_lemonade_water_calculation_l3564_356499

/-- Represents the ratio of ingredients in the lemonade recipe -/
structure LemonadeRatio where
  water : ℚ
  lemon_juice : ℚ
  sugar : ℚ

/-- Calculates the amount of water needed for a given lemonade recipe and total volume -/
def water_needed (ratio : LemonadeRatio) (total_volume : ℚ) (quarts_per_gallon : ℚ) : ℚ :=
  let total_parts := ratio.water + ratio.lemon_juice + ratio.sugar
  let water_fraction := ratio.water / total_parts
  water_fraction * total_volume * quarts_per_gallon

/-- Proves that the amount of water needed for the given recipe and volume is 4 quarts -/
theorem lemonade_water_calculation (ratio : LemonadeRatio) 
  (h1 : ratio.water = 6)
  (h2 : ratio.lemon_juice = 2)
  (h3 : ratio.sugar = 1)
  (h4 : quarts_per_gallon = 4) :
  water_needed ratio (3/2) quarts_per_gallon = 4 := by
  sorry

#eval water_needed ⟨6, 2, 1⟩ (3/2) 4

end NUMINAMATH_CALUDE_lemonade_water_calculation_l3564_356499


namespace NUMINAMATH_CALUDE_g_domain_is_correct_l3564_356497

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def f_domain : Set ℝ := Set.Icc (-6) 9

-- Define the function g in terms of f
def g (x : ℝ) : ℝ := f (-3 * x)

-- Define the domain of g
def g_domain : Set ℝ := Set.Icc (-3) 2

-- Theorem statement
theorem g_domain_is_correct : 
  {x : ℝ | g x ∈ f_domain} = g_domain := by sorry

end NUMINAMATH_CALUDE_g_domain_is_correct_l3564_356497


namespace NUMINAMATH_CALUDE_fraction_simplification_l3564_356457

theorem fraction_simplification : 
  (1 / 4 - 1 / 5) / (1 / 3 - 1 / 4) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3564_356457


namespace NUMINAMATH_CALUDE_archer_probability_l3564_356431

def prob_not_both_hit (prob_A prob_B : ℚ) : ℚ :=
  1 - (prob_A * prob_B)

theorem archer_probability :
  let prob_A : ℚ := 1/3
  let prob_B : ℚ := 1/2
  prob_not_both_hit prob_A prob_B = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_archer_probability_l3564_356431


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3564_356410

theorem expand_and_simplify (x : ℝ) : 2*(x+3)*(x^2 + 2*x + 7) = 2*x^3 + 10*x^2 + 26*x + 42 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3564_356410


namespace NUMINAMATH_CALUDE_circle_line_distance_l3564_356435

theorem circle_line_distance (a : ℝ) : 
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 = 0}
  let line : Set (ℝ × ℝ) := {p | p.1 - p.2 + a = 0}
  let center : ℝ × ℝ := (1, 2)  -- Derived from completing the square
  let distance := |1 - 2 + a| / Real.sqrt 2
  (∀ p ∈ circle, p.1^2 + p.2^2 - 2*p.1 - 4*p.2 = 0) →
  (∀ p ∈ line, p.1 - p.2 + a = 0) →
  distance = Real.sqrt 2 / 2 →
  a = 2 ∨ a = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_l3564_356435


namespace NUMINAMATH_CALUDE_equation_solution_l3564_356479

theorem equation_solution : ∃ x : ℚ, 3 * x - 6 = |(-21 + 8 - 3)| ∧ x = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3564_356479


namespace NUMINAMATH_CALUDE_triangle_properties_l3564_356440

theorem triangle_properties (a b c A B C S : ℝ) : 
  a = 2 → C = π / 3 → 
  (A = π / 4 → c = Real.sqrt 6) ∧ 
  (S = Real.sqrt 3 → b = 2 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3564_356440


namespace NUMINAMATH_CALUDE_all_nines_multiple_l3564_356400

theorem all_nines_multiple (p : Nat) (hp : Nat.Prime p) (hp2 : p ≠ 2) (hp5 : p ≠ 5) :
  ∃ k : Nat, k > 0 ∧ (((10^k - 1) / 9) % p = 0) := by
  sorry

end NUMINAMATH_CALUDE_all_nines_multiple_l3564_356400


namespace NUMINAMATH_CALUDE_range_of_x_minus_y_l3564_356436

theorem range_of_x_minus_y :
  ∀ x y : ℝ, 2 < x ∧ x < 4 → -1 < y ∧ y < 3 →
  ∃ z : ℝ, -1 < z ∧ z < 5 ∧ z = x - y ∧
  ∀ w : ℝ, w = x - y → -1 < w ∧ w < 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_minus_y_l3564_356436


namespace NUMINAMATH_CALUDE_first_number_proof_l3564_356496

theorem first_number_proof (x y : ℕ) (h1 : x + y = 20) (h2 : y = 15) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l3564_356496


namespace NUMINAMATH_CALUDE_pressure_change_pressure_at_4m3_l3564_356480

/-- Represents the pressure-volume relationship for a gas -/
structure GasPV where
  k : ℝ
  pressure : ℝ → ℝ
  volume : ℝ → ℝ
  inverse_square_relation : ∀ v, pressure v = k / (volume v)^2

/-- The theorem stating the pressure when volume changes -/
theorem pressure_change (gas : GasPV) (v₁ v₂ : ℝ) (p₁ : ℝ) 
    (h₁ : gas.pressure v₁ = p₁)
    (h₂ : v₁ > 0)
    (h₃ : v₂ > 0)
    (h₄ : gas.volume v₁ = v₁)
    (h₅ : gas.volume v₂ = v₂) :
  gas.pressure v₂ = p₁ * (v₁ / v₂)^2 :=
by sorry

/-- The specific problem instance -/
theorem pressure_at_4m3 (gas : GasPV) 
    (h₁ : gas.pressure 2 = 25)
    (h₂ : gas.volume 2 = 2)
    (h₃ : gas.volume 4 = 4) :
  gas.pressure 4 = 6.25 :=
by sorry

end NUMINAMATH_CALUDE_pressure_change_pressure_at_4m3_l3564_356480


namespace NUMINAMATH_CALUDE_f_properties_l3564_356449

/-- The type of pairs of real numbers -/
def RealPair := ℝ × ℝ

/-- The function f mapping from RealPair to RealPair -/
def f (p : RealPair) : RealPair :=
  (p.1 + p.2, p.1 - p.2)

theorem f_properties :
  (∃ (p : RealPair), f p = (4, -2) ∧ p = (1, 3)) ∧
  (∃ (p : RealPair), f p = (1, 3) ∧ p = (2, -1)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3564_356449


namespace NUMINAMATH_CALUDE_cards_in_unfilled_box_l3564_356462

theorem cards_in_unfilled_box (total_cards : ℕ) (cards_per_box : ℕ) 
  (h1 : total_cards = 94) (h2 : cards_per_box = 8) : 
  total_cards % cards_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_cards_in_unfilled_box_l3564_356462


namespace NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l3564_356427

theorem x_less_than_2_necessary_not_sufficient :
  (∃ x : ℝ, x < 2 ∧ x^2 - 2*x ≥ 0) ∧
  (∀ x : ℝ, x^2 - 2*x < 0 → x < 2) :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l3564_356427


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3564_356468

theorem smallest_n_congruence (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < n → ¬(629 * m ≡ 1181 * m [ZMOD 35])) ∧ 
  (629 * n ≡ 1181 * n [ZMOD 35]) → 
  n = 35 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3564_356468


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3564_356424

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (k : ℚ), k * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
    (a * Real.sqrt 6 + b * Real.sqrt 8) / c) →
  (∀ (x y z : ℕ+), 
    (∃ (l : ℚ), l * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
      (x * Real.sqrt 6 + y * Real.sqrt 8) / z) → 
    c ≤ z) →
  a + b + c = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3564_356424


namespace NUMINAMATH_CALUDE_painted_square_ratio_exists_l3564_356405

/-- Represents a square with a painted pattern -/
structure PaintedSquare where
  s : ℝ  -- side length of the square
  w : ℝ  -- width of the brush
  h_positive_s : 0 < s
  h_positive_w : 0 < w
  h_painted_area : w^2 + 2 * Real.sqrt 2 * ((s - w * Real.sqrt 2) / 2)^2 = s^2 / 3

/-- There exists a ratio between the side length and brush width for a painted square -/
theorem painted_square_ratio_exists (ps : PaintedSquare) : 
  ∃ r : ℝ, ps.s = r * ps.w :=
sorry

end NUMINAMATH_CALUDE_painted_square_ratio_exists_l3564_356405


namespace NUMINAMATH_CALUDE_evening_rice_fraction_l3564_356463

/-- 
Given:
- Rose initially has 10 kg of rice
- She cooks 9/10 kg in the morning
- She has 750 g left at the end
Prove that the fraction of remaining rice cooked in the evening is 1/4
-/
theorem evening_rice_fraction (initial_rice : ℝ) (morning_cooked : ℝ) (final_rice : ℝ) :
  initial_rice = 10 →
  morning_cooked = 9/10 →
  final_rice = 750/1000 →
  (initial_rice - morning_cooked - final_rice) / (initial_rice - morning_cooked) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_evening_rice_fraction_l3564_356463


namespace NUMINAMATH_CALUDE_area_is_54_height_is_7_2_l3564_356443

/-- A triangle with side lengths 9, 12, and 15 units -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : a = 9
  h_b : b = 12
  h_c : c = 15
  h_right : a ^ 2 + b ^ 2 = c ^ 2

/-- The area of the triangle is 54 square units -/
theorem area_is_54 (t : RightTriangle) : (1 / 2) * t.a * t.b = 54 := by sorry

/-- The height from the right angle vertex to the hypotenuse is 7.2 units -/
theorem height_is_7_2 (t : RightTriangle) : (t.a * t.b) / t.c = 7.2 := by sorry

end NUMINAMATH_CALUDE_area_is_54_height_is_7_2_l3564_356443


namespace NUMINAMATH_CALUDE_andrew_total_work_hours_l3564_356459

/-- The total hours Andrew worked on his Science report over three days -/
def total_hours (day1 day2 day3 : Real) : Real :=
  day1 + day2 + day3

/-- Theorem stating that Andrew worked 9.25 hours in total -/
theorem andrew_total_work_hours :
  let day1 : Real := 2.5
  let day2 : Real := day1 + 0.5
  let day3 : Real := 3.75
  total_hours day1 day2 day3 = 9.25 := by
  sorry

end NUMINAMATH_CALUDE_andrew_total_work_hours_l3564_356459


namespace NUMINAMATH_CALUDE_calculation_proof_l3564_356494

theorem calculation_proof : (-1) * (-4) + 2^2 / (7 - 5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3564_356494


namespace NUMINAMATH_CALUDE_election_majority_proof_l3564_356401

theorem election_majority_proof (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 440 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 176 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_proof_l3564_356401


namespace NUMINAMATH_CALUDE_product_four_consecutive_odd_integers_is_nine_l3564_356402

theorem product_four_consecutive_odd_integers_is_nine :
  ∃ n : ℤ, (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3) = 9 :=
by sorry

end NUMINAMATH_CALUDE_product_four_consecutive_odd_integers_is_nine_l3564_356402


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3564_356452

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - 2*x - 3 < 0) → (-2 < x ∧ x < 3) ∧
  ∃ y : ℝ, -2 < y ∧ y < 3 ∧ ¬(y^2 - 2*y - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3564_356452


namespace NUMINAMATH_CALUDE_calvin_insect_collection_l3564_356406

/-- Calculates the total number of insects in Calvin's collection. -/
def total_insects (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := scorpions * 2
  roaches + scorpions + crickets + caterpillars

/-- Proves that Calvin has 27 insects in his collection. -/
theorem calvin_insect_collection : total_insects 12 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_calvin_insect_collection_l3564_356406


namespace NUMINAMATH_CALUDE_flower_cost_is_nine_l3564_356460

/-- The cost of planting flowers -/
def flower_planting (flower_cost : ℚ) : Prop :=
  let pot_cost : ℚ := flower_cost + 20
  let soil_cost : ℚ := flower_cost - 2
  flower_cost + pot_cost + soil_cost = 45

/-- Theorem: The cost of the flower is $9 -/
theorem flower_cost_is_nine : ∃ (flower_cost : ℚ), flower_cost = 9 ∧ flower_planting flower_cost := by
  sorry

end NUMINAMATH_CALUDE_flower_cost_is_nine_l3564_356460


namespace NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l3564_356432

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem seventh_term_of_arithmetic_sequence 
  (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : sum_of_arithmetic_sequence a 13 = 39) : 
  a 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l3564_356432


namespace NUMINAMATH_CALUDE_day_of_week_N_minus_1_l3564_356442

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  value : ℕ
  is_leap : Bool

/-- Function to get the day of the week for a given day in a year -/
def day_of_week (year : Year) (day : ℕ) : DayOfWeek := sorry

/-- Function to get the next day of the week -/
def next_day (d : DayOfWeek) : DayOfWeek := sorry

/-- Function to get the previous day of the week -/
def prev_day (d : DayOfWeek) : DayOfWeek := sorry

theorem day_of_week_N_minus_1 
  (N : Year)
  (h1 : N.is_leap = true)
  (h2 : day_of_week N 250 = DayOfWeek.Friday)
  (h3 : (Year.mk (N.value + 1) true).is_leap = true)
  (h4 : day_of_week (Year.mk (N.value + 1) true) 150 = DayOfWeek.Friday) :
  day_of_week (Year.mk (N.value - 1) false) 50 = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_day_of_week_N_minus_1_l3564_356442


namespace NUMINAMATH_CALUDE_square_formation_proof_l3564_356474

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, n = m * m

def piece_sizes : List Nat := [4, 5, 6, 7, 8]

def total_squares : Nat := piece_sizes.sum

theorem square_formation_proof :
  ∃ (removed_piece : Nat),
    removed_piece ∈ piece_sizes ∧
    is_perfect_square (total_squares - removed_piece) ∧
    removed_piece = 5 :=
  sorry

end NUMINAMATH_CALUDE_square_formation_proof_l3564_356474


namespace NUMINAMATH_CALUDE_no_prime_divisible_by_55_l3564_356483

theorem no_prime_divisible_by_55 : ¬ ∃ p : ℕ, Nat.Prime p ∧ 55 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_no_prime_divisible_by_55_l3564_356483


namespace NUMINAMATH_CALUDE_min_area_circle_through_intersections_l3564_356403

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x + y + 4 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the minimum area circle
def min_area_circle (x y : ℝ) : Prop := (x + 13/5)^2 + (y - 6/5)^2 = 4/5

-- Theorem statement
theorem min_area_circle_through_intersections :
  ∀ x y : ℝ, 
  (∃ x1 y1 x2 y2 : ℝ, 
    line_l x1 y1 ∧ circle_C x1 y1 ∧
    line_l x2 y2 ∧ circle_C x2 y2 ∧
    x1 ≠ x2 ∧ y1 ≠ y2 ∧
    min_area_circle x1 y1 ∧
    min_area_circle x2 y2) →
  (∀ r : ℝ, ∀ a b : ℝ,
    ((x - a)^2 + (y - b)^2 = r^2 ∧
     (∃ x1 y1 x2 y2 : ℝ, 
       line_l x1 y1 ∧ circle_C x1 y1 ∧
       line_l x2 y2 ∧ circle_C x2 y2 ∧
       (x1 - a)^2 + (y1 - b)^2 = r^2 ∧
       (x2 - a)^2 + (y2 - b)^2 = r^2)) →
    r^2 ≥ 4/5) :=
sorry

end NUMINAMATH_CALUDE_min_area_circle_through_intersections_l3564_356403


namespace NUMINAMATH_CALUDE_tangency_points_divide_side_l3564_356448

/-- 
Given two positive real numbers a and b representing two sides of a triangle,
this theorem states that if the third side c is divided into three equal segments
by the points of tangency of the inscribed and escribed circles, then c = 3|a - b|,
under the condition that b < a < 2b or a < b < 2a.
-/
theorem tangency_points_divide_side (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b < a ∧ a < 2 * b) ∨ (a < b ∧ b < 2 * a) →
  ∃ c : ℝ, c > 0 ∧ c = 3 * |a - b| ∧
    (∃ m r : ℝ, 0 < m ∧ m < r ∧ r < c ∧
      m = c / 3 ∧ r = 2 * c / 3 ∧
      m = (a + c - b) / 2 ∧ (c - r) = (a + c - b) / 2) :=
by sorry

end NUMINAMATH_CALUDE_tangency_points_divide_side_l3564_356448


namespace NUMINAMATH_CALUDE_ducks_in_lake_l3564_356476

/-- The number of ducks swimming in a lake after multiple groups join -/
def total_ducks (initial : ℕ) (first_group : ℕ) (additional : ℕ) : ℕ :=
  initial + first_group + additional

/-- Theorem stating the total number of ducks in the lake -/
theorem ducks_in_lake : 
  ∀ x : ℕ, total_ducks 13 20 x = 33 + x :=
by
  sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l3564_356476


namespace NUMINAMATH_CALUDE_properties_of_f_l3564_356458

noncomputable def f (x : ℝ) : ℝ := (3/2) ^ x

theorem properties_of_f (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  (f (x₁ + x₂) = f x₁ * f x₂) ∧
  ((f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
  (1 < x₁ → x₁ < x₂ → f x₁ / (x₁ - 1) > f x₂ / (x₂ - 1)) :=
by sorry

end NUMINAMATH_CALUDE_properties_of_f_l3564_356458


namespace NUMINAMATH_CALUDE_sum_of_medians_is_64_l3564_356498

def median (scores : List ℕ) : ℚ :=
  sorry

theorem sum_of_medians_is_64 (scores_A scores_B : List ℕ) : 
  median scores_A + median scores_B = 64 :=
sorry

end NUMINAMATH_CALUDE_sum_of_medians_is_64_l3564_356498


namespace NUMINAMATH_CALUDE_bill_donut_combinations_l3564_356419

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 2 identical objects into 4 distinct boxes --/
def bill_combinations : ℕ := distribute 2 4

/-- Theorem: Bill's donut combinations equal 10 --/
theorem bill_donut_combinations : bill_combinations = 10 := by sorry

end NUMINAMATH_CALUDE_bill_donut_combinations_l3564_356419


namespace NUMINAMATH_CALUDE_simplify_expression_l3564_356481

theorem simplify_expression (x y : ℝ) : -x + y - 2*x - 3*y = -3*x - 2*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3564_356481


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3564_356487

theorem unique_triple_solution :
  ∀ a b c : ℕ+,
    (∃ k₁ : ℕ, a * b + 1 = k₁ * c) ∧
    (∃ k₂ : ℕ, a * c + 1 = k₂ * b) ∧
    (∃ k₃ : ℕ, b * c + 1 = k₃ * a) →
    a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3564_356487


namespace NUMINAMATH_CALUDE_sum_inequality_l3564_356486

theorem sum_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3564_356486


namespace NUMINAMATH_CALUDE_order_of_abc_l3564_356411

noncomputable def a : ℝ := (4 - Real.log 4) / Real.exp 2
noncomputable def b : ℝ := Real.log 2 / 2
noncomputable def c : ℝ := 1 / Real.exp 1

theorem order_of_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l3564_356411


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3564_356470

theorem binomial_coefficient_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₂ + a₄) / (a₁ + a₃) = -3/4 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3564_356470


namespace NUMINAMATH_CALUDE_star_placement_impossible_l3564_356425

/-- Represents a grid of cells that may contain stars. -/
def Grid := Fin 10 → Fin 10 → Bool

/-- Checks if a 2x2 square starting at (i, j) contains exactly two stars. -/
def has_two_stars_2x2 (grid : Grid) (i j : Fin 10) : Prop :=
  (grid i j).toNat + (grid i (j+1)).toNat + (grid (i+1) j).toNat + (grid (i+1) (j+1)).toNat = 2

/-- Checks if a 3x1 rectangle starting at (i, j) contains exactly one star. -/
def has_one_star_3x1 (grid : Grid) (i j : Fin 10) : Prop :=
  (grid i j).toNat + (grid (i+1) j).toNat + (grid (i+2) j).toNat = 1

/-- The main theorem stating the impossibility of the star placement. -/
theorem star_placement_impossible : 
  ¬∃ (grid : Grid), 
    (∀ i j : Fin 9, has_two_stars_2x2 grid i j) ∧ 
    (∀ i : Fin 8, ∀ j : Fin 10, has_one_star_3x1 grid i j) :=
sorry

end NUMINAMATH_CALUDE_star_placement_impossible_l3564_356425


namespace NUMINAMATH_CALUDE_tom_balloons_l3564_356444

theorem tom_balloons (initial given left : ℕ) : 
  given = 16 → left = 14 → initial = given + left :=
by sorry

end NUMINAMATH_CALUDE_tom_balloons_l3564_356444


namespace NUMINAMATH_CALUDE_third_number_proof_l3564_356450

theorem third_number_proof (x : ℝ) : 0.3 * 0.8 + x * 0.5 = 0.29 → x = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l3564_356450


namespace NUMINAMATH_CALUDE_translate_quadratic_function_l3564_356441

/-- Represents a function f(x) = (x-a)^2 + b --/
def quadratic_function (a b : ℝ) : ℝ → ℝ := λ x ↦ (x - a)^2 + b

/-- Translates a function horizontally by h units and vertically by k units --/
def translate (f : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ := λ x ↦ f (x - h) + k

theorem translate_quadratic_function :
  let f := quadratic_function 2 1
  let g := translate f 1 1
  g = quadratic_function 1 2 := by sorry

end NUMINAMATH_CALUDE_translate_quadratic_function_l3564_356441


namespace NUMINAMATH_CALUDE_probability_A_and_B_selected_l3564_356466

/-- The number of students in the group -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B when choosing 3 students from 5 -/
def prob_select_A_and_B : ℚ := 3 / 10

/-- Theorem stating the probability of selecting both A and B -/
theorem probability_A_and_B_selected :
  (Nat.choose (total_students - 2) (selected_students - 2)) / 
  (Nat.choose total_students selected_students) = prob_select_A_and_B := by
  sorry


end NUMINAMATH_CALUDE_probability_A_and_B_selected_l3564_356466


namespace NUMINAMATH_CALUDE_percentage_calculation_l3564_356478

theorem percentage_calculation (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 
  0.5 * (0.4 * 0.3 * x) = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3564_356478


namespace NUMINAMATH_CALUDE_cubic_arithmetic_progression_roots_l3564_356429

/-- A cubic polynomial with coefficients in ℝ -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The roots of a cubic polynomial form an arithmetic progression -/
def roots_in_arithmetic_progression (p : CubicPolynomial) : Prop :=
  ∃ (r d : ℂ), p.a * (r - d)^3 + p.b * (r - d)^2 + p.c * (r - d) + p.d = 0 ∧
                p.a * r^3 + p.b * r^2 + p.c * r + p.d = 0 ∧
                p.a * (r + d)^3 + p.b * (r + d)^2 + p.c * (r + d) + p.d = 0

/-- Two roots of a cubic polynomial are not real -/
def two_roots_not_real (p : CubicPolynomial) : Prop :=
  ∃ (z w : ℂ), p.a * z^3 + p.b * z^2 + p.c * z + p.d = 0 ∧
                p.a * w^3 + p.b * w^2 + p.c * w + p.d = 0 ∧
                z.im ≠ 0 ∧ w.im ≠ 0 ∧ z ≠ w

theorem cubic_arithmetic_progression_roots (a : ℝ) :
  let p := CubicPolynomial.mk 1 (-7) 20 a
  roots_in_arithmetic_progression p ∧ two_roots_not_real p → a = -574/27 := by
  sorry

end NUMINAMATH_CALUDE_cubic_arithmetic_progression_roots_l3564_356429


namespace NUMINAMATH_CALUDE_messages_sent_theorem_l3564_356414

/-- Calculates the total number of messages sent over three days given the conditions -/
def totalMessages (luciaFirstDay : ℕ) (alinaDifference : ℕ) : ℕ :=
  let alinaFirstDay := luciaFirstDay - alinaDifference
  let firstDayTotal := luciaFirstDay + alinaFirstDay
  let luciaSecondDay := luciaFirstDay / 3
  let alinaSecondDay := alinaFirstDay * 2
  let secondDayTotal := luciaSecondDay + alinaSecondDay
  firstDayTotal + secondDayTotal + firstDayTotal

theorem messages_sent_theorem :
  totalMessages 120 20 = 680 := by
  sorry

#eval totalMessages 120 20

end NUMINAMATH_CALUDE_messages_sent_theorem_l3564_356414


namespace NUMINAMATH_CALUDE_competition_results_l3564_356493

def scores_8_1 : List ℕ := [70, 70, 75, 75, 75, 75, 80, 80, 80, 85, 90, 90, 90, 90, 90, 95, 95, 95, 100, 100]
def scores_8_2 : List ℕ := [75, 75, 80, 80, 80, 80, 80, 85, 85, 85, 85, 85, 85, 85, 85, 90, 90, 95, 95, 100]

def median (l : List ℕ) : ℚ := sorry
def mean (l : List ℕ) : ℚ := sorry
def variance (l : List ℕ) : ℚ := sorry

theorem competition_results :
  (median scores_8_1 = 87.5) ∧
  (mean scores_8_2 = 85) ∧
  (variance scores_8_2 < variance scores_8_1) := by
  sorry

end NUMINAMATH_CALUDE_competition_results_l3564_356493


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_iff_a_eq_one_l3564_356475

/-- The quadratic function f(x) = ax^2 - (a+1)x + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1

/-- The solution set of f(x) < 0 is empty --/
def has_empty_solution_set (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≥ 0

theorem quadratic_inequality_empty_iff_a_eq_one :
  ∀ a : ℝ, has_empty_solution_set a ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_iff_a_eq_one_l3564_356475


namespace NUMINAMATH_CALUDE_only_3_4_5_is_right_triangle_l3564_356409

/-- A function that checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def sets : List (ℕ × ℕ × ℕ) :=
  [(1, 2, 2), (3, 4, 5), (3, 4, 9), (4, 5, 7)]

/-- Theorem stating that only (3, 4, 5) forms a right-angled triangle --/
theorem only_3_4_5_is_right_triangle :
  ∃! (a b c : ℕ), (a, b, c) ∈ sets ∧ is_right_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_only_3_4_5_is_right_triangle_l3564_356409


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3564_356420

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, (y : ℚ) / 4 + 3 / 7 > 4 / 7 ↔ y ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3564_356420


namespace NUMINAMATH_CALUDE_triangle_side_count_l3564_356417

theorem triangle_side_count : ∃! n : ℕ, n = (Finset.filter (fun x => x > 3 ∧ x < 11) (Finset.range 11)).card := by sorry

end NUMINAMATH_CALUDE_triangle_side_count_l3564_356417


namespace NUMINAMATH_CALUDE_max_sum_products_l3564_356418

theorem max_sum_products (a b c d : ℕ) : 
  a ∈ ({2, 3, 4, 5} : Set ℕ) → 
  b ∈ ({2, 3, 4, 5} : Set ℕ) → 
  c ∈ ({2, 3, 4, 5} : Set ℕ) → 
  d ∈ ({2, 3, 4, 5} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (∀ x y z w, x ∈ ({2, 3, 4, 5} : Set ℕ) → 
              y ∈ ({2, 3, 4, 5} : Set ℕ) → 
              z ∈ ({2, 3, 4, 5} : Set ℕ) → 
              w ∈ ({2, 3, 4, 5} : Set ℕ) → 
              x ≠ y → x ≠ z → x ≠ w → y ≠ z → y ≠ w → z ≠ w →
              x * y + x * z + x * w + y * z ≤ a * b + a * c + a * d + b * c) →
  a * b + a * c + a * d + b * c = 39 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_products_l3564_356418


namespace NUMINAMATH_CALUDE_constant_expression_value_l3564_356423

-- Define the triangle DEF
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

-- Define the properties of the triangle
def Triangle.sideLengths (t : Triangle) : ℝ × ℝ × ℝ := sorry
def Triangle.circumradius (t : Triangle) : ℝ := sorry
def Triangle.orthocenter (t : Triangle) : ℝ × ℝ := sorry
def Triangle.circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

-- Define the constant expression
def constantExpression (t : Triangle) (Q : ℝ × ℝ) : ℝ :=
  let (d, e, f) := t.sideLengths
  let H := t.orthocenter
  let S := t.circumradius
  (Q.1 - t.D.1)^2 + (Q.2 - t.D.2)^2 +
  (Q.1 - t.E.1)^2 + (Q.2 - t.E.2)^2 +
  (Q.1 - t.F.1)^2 + (Q.2 - t.F.2)^2 -
  ((Q.1 - H.1)^2 + (Q.2 - H.2)^2)

-- State the theorem
theorem constant_expression_value (t : Triangle) :
  ∀ Q ∈ t.circumcircle, constantExpression t Q = 
    let (d, e, f) := t.sideLengths
    let S := t.circumradius
    d^2 + e^2 + f^2 - 4 * S^2 :=
sorry

end NUMINAMATH_CALUDE_constant_expression_value_l3564_356423


namespace NUMINAMATH_CALUDE_jerry_shelves_problem_l3564_356485

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (total_books - books_taken + books_per_shelf - 1) / books_per_shelf

theorem jerry_shelves_problem :
  shelves_needed 34 7 3 = 9 :=
by sorry

end NUMINAMATH_CALUDE_jerry_shelves_problem_l3564_356485


namespace NUMINAMATH_CALUDE_lawn_mowing_difference_l3564_356447

/-- The difference between spring and summer lawn mowing counts -/
theorem lawn_mowing_difference (spring_count summer_count : ℕ) 
  (h1 : spring_count = 8) 
  (h2 : summer_count = 5) : 
  spring_count - summer_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_difference_l3564_356447


namespace NUMINAMATH_CALUDE_identity_is_unique_divisibility_function_l3564_356438

/-- A function f: ℕ → ℕ satisfying the divisibility condition -/
def DivisibilityFunction (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, (m^2 + n)^2 % (f m^2 + f n) = 0

/-- The theorem stating that the identity function is the only function satisfying the divisibility condition -/
theorem identity_is_unique_divisibility_function :
  ∀ f : ℕ → ℕ, DivisibilityFunction f ↔ ∀ n : ℕ, f n = n :=
sorry

end NUMINAMATH_CALUDE_identity_is_unique_divisibility_function_l3564_356438


namespace NUMINAMATH_CALUDE_tom_remaining_pieces_l3564_356415

/-- 
Given the initial number of boxes, the number of boxes given away, 
and the number of pieces per box, calculate the number of pieces Tom still had.
-/
def remaining_pieces (initial_boxes : ℕ) (boxes_given_away : ℕ) (pieces_per_box : ℕ) : ℕ :=
  (initial_boxes - boxes_given_away) * pieces_per_box

theorem tom_remaining_pieces : 
  remaining_pieces 12 7 6 = 30 := by
  sorry

#eval remaining_pieces 12 7 6

end NUMINAMATH_CALUDE_tom_remaining_pieces_l3564_356415


namespace NUMINAMATH_CALUDE_expression_simplification_l3564_356495

theorem expression_simplification (x : ℝ) (h : x^2 - 3*x - 4 = 0) :
  (x / (x + 1) - 2 / (x - 1)) / (1 / (x^2 - 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3564_356495


namespace NUMINAMATH_CALUDE_rachel_furniture_time_l3564_356456

def chairs : ℕ := 7
def tables : ℕ := 3
def time_per_piece : ℕ := 4

theorem rachel_furniture_time :
  chairs * time_per_piece + tables * time_per_piece = 40 := by
  sorry

end NUMINAMATH_CALUDE_rachel_furniture_time_l3564_356456


namespace NUMINAMATH_CALUDE_joan_seashells_l3564_356422

def seashell_problem (initial found : ℕ) (given_away : ℕ) (additional_found : ℕ) (traded : ℕ) (lost : ℕ) : ℕ :=
  initial - given_away + additional_found - traded - lost

theorem joan_seashells :
  seashell_problem 79 63 45 20 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l3564_356422


namespace NUMINAMATH_CALUDE_min_points_all_but_one_hemisphere_l3564_356408

/-- A point on the surface of a sphere -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ
  on_sphere : x^2 + y^2 + z^2 = 1

/-- A hemisphere of a sphere -/
def Hemisphere := Set Point3D

/-- The set of all possible hemispheres of a sphere -/
def AllHemispheres : Set Hemisphere := sorry

/-- A set of points is in all hemispheres except one if it intersects with all but one hemisphere -/
def InAllButOneHemisphere (points : Set Point3D) : Prop :=
  ∃ h : Hemisphere, h ∈ AllHemispheres ∧ 
    ∀ h' : Hemisphere, h' ∈ AllHemispheres → h' ≠ h → (points ∩ h').Nonempty

theorem min_points_all_but_one_hemisphere :
  ∃ (points : Set Point3D), points.ncard = 4 ∧ InAllButOneHemisphere points ∧
    ∀ (points' : Set Point3D), points'.ncard < 4 → ¬InAllButOneHemisphere points' :=
  sorry

end NUMINAMATH_CALUDE_min_points_all_but_one_hemisphere_l3564_356408


namespace NUMINAMATH_CALUDE_cos_585_degrees_l3564_356491

theorem cos_585_degrees :
  Real.cos (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_585_degrees_l3564_356491


namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l3564_356451

/-- The area of a stripe on a cylindrical water tower -/
theorem stripe_area_on_cylinder (diameter : ℝ) (stripe_width : ℝ) (revolutions : ℕ) :
  diameter = 20 ∧ stripe_width = 4 ∧ revolutions = 3 →
  stripe_width * revolutions * π * diameter = 240 * π :=
by sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l3564_356451


namespace NUMINAMATH_CALUDE_max_value_sqrt_function_l3564_356461

theorem max_value_sqrt_function (x : ℝ) (h1 : 2 < x) (h2 : x < 5) :
  ∃ (M : ℝ), M = 4 * Real.sqrt 3 ∧ ∀ y, 2 < y → y < 5 → Real.sqrt (3 * y * (8 - y)) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_function_l3564_356461


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l3564_356467

theorem coconut_grove_problem (x : ℕ) : 
  (∃ (t₄₀ t₁₂₀ t₁₈₀ : ℕ),
    t₄₀ = x + 2 ∧
    t₁₂₀ = x ∧
    t₁₈₀ = x - 2 ∧
    (40 * t₄₀ + 120 * t₁₂₀ + 180 * t₁₈₀) / (t₄₀ + t₁₂₀ + t₁₈₀) = 100) →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l3564_356467


namespace NUMINAMATH_CALUDE_hostel_provisions_l3564_356437

/-- Proves that given the initial conditions of a hostel's food provisions,
    the initial number of days the provisions were planned for is 28. -/
theorem hostel_provisions (initial_men : ℕ) (left_men : ℕ) (days_after_leaving : ℕ) 
  (h1 : initial_men = 250)
  (h2 : left_men = 50)
  (h3 : days_after_leaving = 35) :
  (initial_men * ((initial_men - left_men) * days_after_leaving / initial_men) : ℚ) = 
  (initial_men * 28 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_hostel_provisions_l3564_356437


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3564_356492

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3564_356492
