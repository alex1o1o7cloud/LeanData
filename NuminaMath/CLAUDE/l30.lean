import Mathlib

namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l30_3065

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ (x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l30_3065


namespace NUMINAMATH_CALUDE_bob_has_81_robots_l30_3047

/-- The number of car robots Tom and Michael have together -/
def tom_and_michael_robots : ℕ := 9

/-- The factor by which Bob has more car robots than Tom and Michael -/
def bob_factor : ℕ := 9

/-- The total number of car robots Bob has -/
def bob_robots : ℕ := tom_and_michael_robots * bob_factor

/-- Theorem stating that Bob has 81 car robots -/
theorem bob_has_81_robots : bob_robots = 81 := by
  sorry

end NUMINAMATH_CALUDE_bob_has_81_robots_l30_3047


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l30_3032

theorem triangle_angle_sum (A B C : ℝ) (h : A + B = 80) : C = 100 :=
  by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l30_3032


namespace NUMINAMATH_CALUDE_sum_of_fractions_l30_3063

theorem sum_of_fractions : (48 : ℚ) / 72 + (30 : ℚ) / 45 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l30_3063


namespace NUMINAMATH_CALUDE_tv_series_seasons_l30_3053

theorem tv_series_seasons (episodes_per_season : ℕ) (episodes_per_day : ℕ) (days_to_finish : ℕ) : 
  episodes_per_season = 20 →
  episodes_per_day = 2 →
  days_to_finish = 30 →
  (episodes_per_day * days_to_finish) / episodes_per_season = 3 := by
sorry

end NUMINAMATH_CALUDE_tv_series_seasons_l30_3053


namespace NUMINAMATH_CALUDE_shyne_plants_l30_3068

/-- The number of eggplants that can be grown from one seed packet -/
def eggplants_per_packet : ℕ := 14

/-- The number of sunflowers that can be grown from one seed packet -/
def sunflowers_per_packet : ℕ := 10

/-- The number of eggplant seed packets Shyne bought -/
def eggplant_packets : ℕ := 4

/-- The number of sunflower seed packets Shyne bought -/
def sunflower_packets : ℕ := 6

/-- The total number of plants Shyne can grow -/
def total_plants : ℕ := eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets

theorem shyne_plants : total_plants = 116 := by
  sorry

end NUMINAMATH_CALUDE_shyne_plants_l30_3068


namespace NUMINAMATH_CALUDE_total_situps_is_510_l30_3021

/-- The number of sit-ups Barney can perform in one minute -/
def barney_situps : ℕ := 45

/-- The number of minutes Barney performs sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The number of sit-ups Carrie can perform in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can perform in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ :=
  barney_situps * barney_minutes +
  carrie_situps * carrie_minutes +
  jerrie_situps * jerrie_minutes

/-- Theorem stating that the total number of sit-ups is 510 -/
theorem total_situps_is_510 : total_situps = 510 := by
  sorry

end NUMINAMATH_CALUDE_total_situps_is_510_l30_3021


namespace NUMINAMATH_CALUDE_defective_tubes_count_l30_3060

/-- The probability of selecting two defective tubes without replacement -/
def prob_two_defective : ℝ := 0.05263157894736842

/-- The total number of picture tubes in the consignment -/
def total_tubes : ℕ := 20

/-- The number of defective picture tubes in the consignment -/
def num_defective : ℕ := 5

theorem defective_tubes_count :
  (num_defective : ℝ) / total_tubes * ((num_defective - 1) : ℝ) / (total_tubes - 1) = prob_two_defective := by
  sorry

end NUMINAMATH_CALUDE_defective_tubes_count_l30_3060


namespace NUMINAMATH_CALUDE_square_area_proof_l30_3076

theorem square_area_proof (x : ℚ) :
  (5 * x - 20 : ℚ) = (25 - 2 * x : ℚ) →
  ((5 * x - 20)^2 : ℚ) = 7225 / 49 := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l30_3076


namespace NUMINAMATH_CALUDE_farmer_james_animals_l30_3056

/-- Represents the number of heads for each animal type -/
def heads : Fin 3 → ℕ
  | 0 => 2  -- Hens
  | 1 => 3  -- Peacocks
  | 2 => 6  -- Zombie hens

/-- Represents the number of legs for each animal type -/
def legs : Fin 3 → ℕ
  | 0 => 8  -- Hens
  | 1 => 9  -- Peacocks
  | 2 => 12 -- Zombie hens

/-- The total number of heads on the farm -/
def total_heads : ℕ := 800

/-- The total number of legs on the farm -/
def total_legs : ℕ := 2018

/-- Calculates the total number of animals on the farm -/
def total_animals : ℕ := (total_legs - total_heads) / 6

theorem farmer_james_animals :
  total_animals = 203 ∧
  (∃ (h p z : ℕ),
    h * heads 0 + p * heads 1 + z * heads 2 = total_heads ∧
    h * legs 0 + p * legs 1 + z * legs 2 = total_legs ∧
    h + p + z = total_animals) :=
by sorry

#eval total_animals

end NUMINAMATH_CALUDE_farmer_james_animals_l30_3056


namespace NUMINAMATH_CALUDE_sin_150_degrees_l30_3009

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l30_3009


namespace NUMINAMATH_CALUDE_equation_solutions_l30_3001

theorem equation_solutions :
  (∀ x : ℝ, (x - 2) * (x - 3) = x - 2 ↔ x = 2 ∨ x = 4) ∧
  (∀ x : ℝ, 2 * x^2 - 5 * x + 1 = 0 ↔ x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l30_3001


namespace NUMINAMATH_CALUDE_G_is_odd_and_f_neg_b_value_l30_3094

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x / (Real.exp x + 1)

noncomputable def G (x : ℝ) : ℝ := f x - 1

theorem G_is_odd_and_f_neg_b_value (b : ℝ) (h : f b = 3/2) :
  (∀ x, G (-x) = -G x) ∧ f (-b) = 1/2 := by sorry

end NUMINAMATH_CALUDE_G_is_odd_and_f_neg_b_value_l30_3094


namespace NUMINAMATH_CALUDE_prob_3_heads_12_coins_value_l30_3057

/-- The probability of getting exactly 3 heads when flipping 12 coins -/
def prob_3_heads_12_coins : ℚ :=
  (Nat.choose 12 3 : ℚ) / 2^12

/-- Theorem stating that the probability of getting exactly 3 heads
    when flipping 12 coins is equal to 220/4096 -/
theorem prob_3_heads_12_coins_value :
  prob_3_heads_12_coins = 220 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_3_heads_12_coins_value_l30_3057


namespace NUMINAMATH_CALUDE_batsman_boundaries_l30_3022

theorem batsman_boundaries (total_runs : ℕ) (sixes : ℕ) (run_percentage : ℚ) : 
  total_runs = 120 →
  sixes = 8 →
  run_percentage = 1/2 →
  (∃ (boundaries : ℕ), 
    total_runs = run_percentage * total_runs + sixes * 6 + boundaries * 4 ∧
    boundaries = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_batsman_boundaries_l30_3022


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l30_3023

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : a 3 = 3)
  (h2 : a 11 = 15)
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) :
  a 1 = 0 ∧ a 2 - a 1 = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l30_3023


namespace NUMINAMATH_CALUDE_mollys_age_l30_3012

/-- Given Sandy's age and the ratio of Sandy's age to Molly's age, calculate Molly's age -/
theorem mollys_age (sandy_age : ℕ) (ratio : ℚ) (h1 : sandy_age = 49) (h2 : ratio = 7/9) :
  sandy_age / ratio = 63 :=
sorry

end NUMINAMATH_CALUDE_mollys_age_l30_3012


namespace NUMINAMATH_CALUDE_M_intersect_N_l30_3084

def M : Set ℝ := {x | x^2 - 4 < 0}
def N : Set ℝ := {x | ∃ n : ℤ, x = 2*n + 1}

theorem M_intersect_N : M ∩ N = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l30_3084


namespace NUMINAMATH_CALUDE_polynomial_division_l30_3051

theorem polynomial_division (x : ℤ) : 
  ∃ (p : ℤ → ℤ), x^13 + 2*x + 180 = (x^2 - x + 3) * p x := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_l30_3051


namespace NUMINAMATH_CALUDE_x_minus_p_equals_3_minus_2p_l30_3077

theorem x_minus_p_equals_3_minus_2p (x p : ℝ) (h1 : |x - 3| = p) (h2 : x < 3) :
  x - p = 3 - 2*p := by
  sorry

end NUMINAMATH_CALUDE_x_minus_p_equals_3_minus_2p_l30_3077


namespace NUMINAMATH_CALUDE_boxes_in_case_l30_3016

/-- Proves the number of boxes in a case given the total boxes, eggs per box, and total eggs -/
theorem boxes_in_case 
  (total_boxes : ℕ) 
  (eggs_per_box : ℕ) 
  (total_eggs : ℕ) 
  (h1 : total_boxes = 5)
  (h2 : eggs_per_box = 3)
  (h3 : total_eggs = 15)
  (h4 : total_eggs = total_boxes * eggs_per_box) :
  total_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_in_case_l30_3016


namespace NUMINAMATH_CALUDE_intersection_A_B_l30_3010

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l30_3010


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l30_3087

theorem tan_seventeen_pi_fourths : Real.tan (17 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l30_3087


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l30_3003

theorem sum_of_squares_and_products (x y z : ℝ) 
  (nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (sum_of_squares : x^2 + y^2 + z^2 = 52) 
  (sum_of_products : x*y + y*z + z*x = 24) : 
  x + y + z = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l30_3003


namespace NUMINAMATH_CALUDE_operation_on_number_l30_3017

theorem operation_on_number (x : ℝ) : x^2 = 25 → 2*x = x/5 + 9 := by
  sorry

end NUMINAMATH_CALUDE_operation_on_number_l30_3017


namespace NUMINAMATH_CALUDE_agricultural_equipment_problem_l30_3067

theorem agricultural_equipment_problem 
  (cost_2A_1B : ℝ) 
  (cost_1A_3B : ℝ) 
  (total_budget : ℝ) :
  cost_2A_1B = 4.2 →
  cost_1A_3B = 5.1 →
  total_budget = 10 →
  ∃ (cost_A cost_B : ℝ) (max_units_A : ℕ),
    cost_A = 1.5 ∧
    cost_B = 1.2 ∧
    max_units_A = 3 ∧
    2 * cost_A + cost_B = cost_2A_1B ∧
    cost_A + 3 * cost_B = cost_1A_3B ∧
    (∀ m : ℕ, m * cost_A + (2 * m - 3) * cost_B ≤ total_budget → m ≤ max_units_A) :=
by sorry

end NUMINAMATH_CALUDE_agricultural_equipment_problem_l30_3067


namespace NUMINAMATH_CALUDE_grocery_store_deal_cans_l30_3092

theorem grocery_store_deal_cans (bulk_price : ℝ) (bulk_cans : ℕ) (store_price : ℝ) (price_difference : ℝ) : 
  bulk_price = 12 →
  bulk_cans = 48 →
  store_price = 6 →
  price_difference = 0.25 →
  (store_price / ((bulk_price / bulk_cans) + price_difference)) = 12 := by
sorry

end NUMINAMATH_CALUDE_grocery_store_deal_cans_l30_3092


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l30_3029

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9/14) (h2 : x - y = 3/14) : x^2 - y^2 = 27/196 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l30_3029


namespace NUMINAMATH_CALUDE_largest_fraction_l30_3041

theorem largest_fraction (a b c d e : ℚ) 
  (ha : a = 3/10) (hb : b = 9/20) (hc : c = 12/25) (hd : d = 27/50) (he : e = 49/100) :
  d = max a (max b (max c (max d e))) :=
sorry

end NUMINAMATH_CALUDE_largest_fraction_l30_3041


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l30_3043

theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, |x - 3| - |x - 1| > m) → m < -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l30_3043


namespace NUMINAMATH_CALUDE_thompson_exam_rule_l30_3078

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (answered_all_correctly : Student → Prop)
variable (received_C_or_higher : Student → Prop)

-- State the theorem
theorem thompson_exam_rule 
  (h : ∀ s : Student, ¬(answered_all_correctly s) → ¬(received_C_or_higher s)) :
  ∀ s : Student, received_C_or_higher s → answered_all_correctly s :=
by sorry

end NUMINAMATH_CALUDE_thompson_exam_rule_l30_3078


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l30_3055

-- Define the line and ellipse
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 20 = 1

-- Theorem statement
theorem line_intersects_ellipse (k : ℝ) :
  ∃ x y : ℝ, line k x = y ∧ ellipse x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l30_3055


namespace NUMINAMATH_CALUDE_new_sailor_weight_l30_3075

theorem new_sailor_weight (n : ℕ) (original_weight replaced_weight : ℝ) 
  (h1 : n = 8)
  (h2 : replaced_weight = 56)
  (h3 : ∀ (total_weight new_weight : ℝ), 
    (total_weight + new_weight - replaced_weight) / n = total_weight / n + 1) :
  ∃ (new_weight : ℝ), new_weight = 64 := by
sorry

end NUMINAMATH_CALUDE_new_sailor_weight_l30_3075


namespace NUMINAMATH_CALUDE_division_remainder_problem_l30_3061

theorem division_remainder_problem (L S : ℕ) (h1 : L - S = 1370) (h2 : L = 1626) 
  (h3 : L / S = 6) : L % S = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l30_3061


namespace NUMINAMATH_CALUDE_k_range_l30_3018

def p (k : ℝ) : Prop := ∀ x y : ℝ, x < y → k * x + 1 < k * y + 1

def q (k : ℝ) : Prop := ∃ x : ℝ, x^2 + (2*k - 3)*x + 1 = 0

theorem k_range (k : ℝ) : 
  (¬(p k ∧ q k) ∧ (p k ∨ q k)) → 
  (k ≤ 0 ∨ (1/2 < k ∧ k < 5/2)) :=
sorry

end NUMINAMATH_CALUDE_k_range_l30_3018


namespace NUMINAMATH_CALUDE_solution_sets_equal_implies_alpha_value_l30_3093

/-- The solution set of the inequality |2x-3| < 2 -/
def solution_set_1 : Set ℝ := {x : ℝ | |2*x - 3| < 2}

/-- The solution set of the inequality x^2 + αx + b < 0 -/
def solution_set_2 (α b : ℝ) : Set ℝ := {x : ℝ | x^2 + α*x + b < 0}

/-- Theorem stating that if the solution sets are equal, then α = -3 -/
theorem solution_sets_equal_implies_alpha_value (b : ℝ) :
  (∃ α, solution_set_1 = solution_set_2 α b) → 
  (∃ α, solution_set_1 = solution_set_2 α b ∧ α = -3) :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_equal_implies_alpha_value_l30_3093


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l30_3085

/-- Proves that the repair cost is $300 given the initial purchase price,
    selling price, and gain percentage. -/
theorem repair_cost_calculation (purchase_price selling_price : ℝ) (gain_percentage : ℝ) :
  purchase_price = 900 →
  selling_price = 1500 →
  gain_percentage = 25 →
  (selling_price / (1 + gain_percentage / 100)) - purchase_price = 300 := by
sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l30_3085


namespace NUMINAMATH_CALUDE_square_area_increase_l30_3071

/-- The increase in area of a square when its side length is increased -/
theorem square_area_increase (initial_side : ℝ) (increase : ℝ) : 
  initial_side = 6 → increase = 1 → 
  (initial_side + increase)^2 - initial_side^2 = 13 := by
  sorry

#check square_area_increase

end NUMINAMATH_CALUDE_square_area_increase_l30_3071


namespace NUMINAMATH_CALUDE_remaining_money_l30_3090

def base_8_to_10 (n : Nat) : Nat :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

def savings : Nat := 5377
def airline_ticket : Nat := 1200
def travel_pass : Nat := 600

theorem remaining_money :
  base_8_to_10 savings - airline_ticket - travel_pass = 1015 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l30_3090


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l30_3000

-- Problem 1
theorem problem_1 : Real.sqrt 27 - (1/3) * Real.sqrt 18 - Real.sqrt 12 = Real.sqrt 3 - Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 48 + Real.sqrt 30 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 * Real.sqrt 3 + Real.sqrt 30 + Real.sqrt 6 := by sorry

-- Problem 3
theorem problem_3 : (2 - Real.sqrt 5) * (2 + Real.sqrt 5) - (2 - Real.sqrt 2)^2 = 4 * Real.sqrt 2 - 7 := by sorry

-- Problem 4
theorem problem_4 : (27 : Real)^(1/3) - (Real.sqrt 2 * Real.sqrt 6) / Real.sqrt 3 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l30_3000


namespace NUMINAMATH_CALUDE_z_reciprocal_modulus_l30_3072

theorem z_reciprocal_modulus (i : ℂ) (z : ℂ) : 
  i^2 = -1 → 
  z = i + 2*i^2 + 3*i^3 + 4*i^4 + 5*i^5 + 6*i^6 + 7*i^7 + 8*i^8 → 
  Complex.abs (z⁻¹) = Real.sqrt 2 / 8 := by
  sorry

end NUMINAMATH_CALUDE_z_reciprocal_modulus_l30_3072


namespace NUMINAMATH_CALUDE_solve_equation_l30_3052

theorem solve_equation (x n : ℝ) (h1 : x / 4 - (x - 3) / n = 1) (h2 : x = 6) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l30_3052


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l30_3042

theorem quadratic_equation_root_zero (k : ℝ) : 
  (k + 3 ≠ 0) →
  (∀ x, (k + 3) * x^2 + 5 * x + k^2 + 2 * k - 3 = 0 ↔ x = 0 ∨ x ≠ 0) →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l30_3042


namespace NUMINAMATH_CALUDE_fraction_evaluation_l30_3066

theorem fraction_evaluation : (2 + 3 * 6) / (23 + 6) = 20 / 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l30_3066


namespace NUMINAMATH_CALUDE_every_three_connected_graph_without_K5_K33_is_planar_l30_3074

-- Define a graph type
structure Graph (V : Type) where
  edges : V → V → Prop

-- Define 3-connectivity
def isThreeConnected (G : Graph V) : Prop := sorry

-- Define subgraph relation
def isSubgraph (H G : Graph V) : Prop := sorry

-- Define K^5 graph
def K5 (V : Type) : Graph V := sorry

-- Define K_{3,3} graph
def K33 (V : Type) : Graph V := sorry

-- Define planarity
def isPlanar (G : Graph V) : Prop := sorry

-- The main theorem
theorem every_three_connected_graph_without_K5_K33_is_planar 
  (G : Graph V) 
  (h1 : isThreeConnected G) 
  (h2 : ¬ isSubgraph (K5 V) G) 
  (h3 : ¬ isSubgraph (K33 V) G) : 
  isPlanar G := by sorry

end NUMINAMATH_CALUDE_every_three_connected_graph_without_K5_K33_is_planar_l30_3074


namespace NUMINAMATH_CALUDE_min_value_of_f_l30_3008

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = -Real.exp 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l30_3008


namespace NUMINAMATH_CALUDE_equation_transformation_l30_3048

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 - 2*x^3 - 3*x^2 + 2*x + 1 = 0 ↔ x^2 * (y^2 - y - 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l30_3048


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l30_3034

theorem simplify_and_ratio : 
  (∀ m : ℝ, (6*m + 12) / 3 = 2*m + 4) ∧ (2 / 4 : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l30_3034


namespace NUMINAMATH_CALUDE_metallic_sheet_dimension_l30_3054

/-- Given a rectangular metallic sheet with one dimension of 36 meters,
    where a square of 8 meters is cut from each corner to form an open box,
    if the volume of the resulting box is 5760 cubic meters,
    then the length of the other dimension of the metallic sheet is 52 meters. -/
theorem metallic_sheet_dimension (sheet_width : ℝ) (cut_size : ℝ) (box_volume : ℝ) :
  sheet_width = 36 →
  cut_size = 8 →
  box_volume = 5760 →
  (sheet_width - 2 * cut_size) * (52 - 2 * cut_size) * cut_size = box_volume :=
by sorry

end NUMINAMATH_CALUDE_metallic_sheet_dimension_l30_3054


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_positive_l30_3069

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1}
def B (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 = a}

-- State the theorem
theorem intersection_nonempty_implies_a_positive (a : ℝ) :
  (A ∩ B a).Nonempty → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_positive_l30_3069


namespace NUMINAMATH_CALUDE_gcd_45123_31207_l30_3088

theorem gcd_45123_31207 : Nat.gcd 45123 31207 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45123_31207_l30_3088


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l30_3070

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (h : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l30_3070


namespace NUMINAMATH_CALUDE_rectangle_area_l30_3039

theorem rectangle_area (square_area : Real) (rectangle_width : Real) (rectangle_length : Real) :
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l30_3039


namespace NUMINAMATH_CALUDE_lilies_per_centerpiece_is_six_l30_3007

/-- Calculates the number of lilies per centerpiece given the following conditions:
  * There are 6 centerpieces
  * Each centerpiece uses 8 roses
  * Each centerpiece uses twice as many orchids as roses
  * The total budget is $2700
  * Each flower costs $15
-/
def lilies_per_centerpiece (num_centerpieces : ℕ) (roses_per_centerpiece : ℕ) 
  (orchid_ratio : ℕ) (total_budget : ℕ) (flower_cost : ℕ) : ℕ :=
  let total_roses := num_centerpieces * roses_per_centerpiece
  let total_orchids := num_centerpieces * roses_per_centerpiece * orchid_ratio
  let rose_orchid_cost := (total_roses + total_orchids) * flower_cost
  let remaining_budget := total_budget - rose_orchid_cost
  let total_lilies := remaining_budget / flower_cost
  total_lilies / num_centerpieces

/-- Theorem stating that given the specific conditions, the number of lilies per centerpiece is 6 -/
theorem lilies_per_centerpiece_is_six :
  lilies_per_centerpiece 6 8 2 2700 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_lilies_per_centerpiece_is_six_l30_3007


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l30_3020

theorem arithmetic_calculations : 
  (1 * (-30) - 4 * (-4) = -14) ∧ 
  ((-2)^2 - (1/7) * (-3-4) = 5) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l30_3020


namespace NUMINAMATH_CALUDE_laura_age_l30_3083

def is_divisible_by (a b : ℕ) : Prop := a % b = 0

theorem laura_age :
  ∀ (L A : ℕ),
  is_divisible_by (L - 1) 8 →
  is_divisible_by (A - 1) 8 →
  is_divisible_by (L + 1) 7 →
  is_divisible_by (A + 1) 7 →
  A < 100 →
  L = 41 :=
by sorry

end NUMINAMATH_CALUDE_laura_age_l30_3083


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l30_3095

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point (A : Point) (l1 l2 : Line) :
  A.x = 2 ∧ A.y = -3 ∧
  l1.a = 1 ∧ l1.b = -2 ∧ l1.c = -3 ∧
  l2.a = 2 ∧ l2.b = 1 ∧ l2.c = -1 →
  A.liesOn l2 ∧ l1.perpendicular l2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l30_3095


namespace NUMINAMATH_CALUDE_periodic_function_value_l30_3036

/-- A function satisfying the given conditions -/
def periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 2) * f x = 1) ∧ f 2 = 2

/-- Theorem stating the value of f(2016) given the conditions -/
theorem periodic_function_value (f : ℝ → ℝ) (h : periodic_function f) : f 2016 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l30_3036


namespace NUMINAMATH_CALUDE_product_inequality_l30_3015

theorem product_inequality (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hab : a * b > 1) (hn : n ≥ 2) :
  (a + b)^n > a^n + b^n + 2^n - 2 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l30_3015


namespace NUMINAMATH_CALUDE_hyperbola_chord_midpoint_l30_3027

/-- Given a hyperbola x²/a² - y²/b² = 1 where a, b > 0,
    the midpoint of any chord with slope 1 lies on the line x/a² - y/b² = 0 -/
theorem hyperbola_chord_midpoint (a b x y : ℝ) (ha : a > 0) (hb : b > 0) :
  x^2 / a^2 - y^2 / b^2 = 1 →
  ∃ (m : ℝ), (x + m) / a^2 - (y + m) / b^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_chord_midpoint_l30_3027


namespace NUMINAMATH_CALUDE_division_problem_l30_3082

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 729 →
  divisor = 38 →
  remainder = 7 →
  dividend = divisor * quotient + remainder →
  quotient = 19 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l30_3082


namespace NUMINAMATH_CALUDE_correct_number_of_elements_l30_3073

theorem correct_number_of_elements 
  (n : ℕ) 
  (S : ℝ) 
  (initial_average : ℝ) 
  (correct_average : ℝ) 
  (wrong_number : ℝ) 
  (correct_number : ℝ) 
  (h1 : initial_average = 15) 
  (h2 : correct_average = 16) 
  (h3 : wrong_number = 26) 
  (h4 : correct_number = 36) 
  (h5 : (S + wrong_number) / n = initial_average) 
  (h6 : (S + correct_number) / n = correct_average) : 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_correct_number_of_elements_l30_3073


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l30_3030

theorem triangle_inequality_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) + b / (c + a) + c / (a + b) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l30_3030


namespace NUMINAMATH_CALUDE_sum_consecutive_odd_integers_to_25_l30_3033

/-- Sum of consecutive odd integers from 1 to n -/
def sumConsecutiveOddIntegers (n : ℕ) : ℕ :=
  let k := (n + 1) / 2
  k * k

/-- Theorem: The sum of consecutive odd integers from 1 to 25 is 169 -/
theorem sum_consecutive_odd_integers_to_25 :
  sumConsecutiveOddIntegers 25 = 169 := by
  sorry

#eval sumConsecutiveOddIntegers 25

end NUMINAMATH_CALUDE_sum_consecutive_odd_integers_to_25_l30_3033


namespace NUMINAMATH_CALUDE_ball_distribution_l30_3049

theorem ball_distribution (a b c : ℕ) : 
  a + b + c = 45 →
  a + 2 = b - 1 ∧ a + 2 = c - 1 →
  (a, b, c) = (13, 16, 16) :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_l30_3049


namespace NUMINAMATH_CALUDE_tea_containers_needed_l30_3064

/-- The volume of tea in milliliters that each container can hold -/
def container_volume : ℕ := 500

/-- The minimum volume of tea in liters needed for the event -/
def required_volume : ℕ := 5

/-- Conversion factor from liters to milliliters -/
def liter_to_ml : ℕ := 1000

/-- The minimum number of containers needed to hold at least the required volume of tea -/
def min_containers : ℕ := 10

theorem tea_containers_needed :
  min_containers = 
    (required_volume * liter_to_ml + container_volume - 1) / container_volume :=
by sorry

end NUMINAMATH_CALUDE_tea_containers_needed_l30_3064


namespace NUMINAMATH_CALUDE_gasoline_price_increase_percentage_l30_3050

def highest_price : ℝ := 24
def lowest_price : ℝ := 12

theorem gasoline_price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_percentage_l30_3050


namespace NUMINAMATH_CALUDE_propositions_correctness_l30_3031

theorem propositions_correctness : 
  -- Proposition ②
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  -- Proposition ③
  (∀ a b : ℝ, a > |b| → a > b) ∧
  -- Proposition ① (negation)
  (∃ a b : ℝ, a > b ∧ (1 / a ≥ 1 / b)) ∧
  -- Proposition ④ (negation)
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) :=
by sorry


end NUMINAMATH_CALUDE_propositions_correctness_l30_3031


namespace NUMINAMATH_CALUDE_coffee_mixture_cost_l30_3079

theorem coffee_mixture_cost (cost_A : ℝ) (cost_mixture : ℝ) (total_weight : ℝ) (weight_A : ℝ) (weight_B : ℝ) :
  cost_A = 10 →
  cost_mixture = 11 →
  total_weight = 480 →
  weight_A = 240 →
  weight_B = 240 →
  (total_weight * cost_mixture - weight_A * cost_A) / weight_B = 12 :=
by sorry

end NUMINAMATH_CALUDE_coffee_mixture_cost_l30_3079


namespace NUMINAMATH_CALUDE_incorrect_proposition_l30_3058

theorem incorrect_proposition :
  ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proposition_l30_3058


namespace NUMINAMATH_CALUDE_yellow_balls_count_l30_3019

theorem yellow_balls_count (total : ℕ) (red : ℕ) (yellow : ℕ) (prob : ℚ) : 
  red = 10 →
  yellow + red = total →
  prob = 2 / 5 →
  (red : ℚ) / total = prob →
  yellow = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l30_3019


namespace NUMINAMATH_CALUDE_triangle_circumradius_l30_3038

/-- Given a triangle ABC where:
  * a, b, c are sides opposite to angles A, B, C respectively
  * a = 2
  * b = 3
  * cos C = 1/3
  Then the radius of the circumcircle is 9√2/8 -/
theorem triangle_circumradius (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  b = 3 →
  c = (a^2 + b^2 - 2*a*b*(1/3))^(1/2) →
  let r := c / (2 * (1 - (1/3)^2)^(1/2))
  r = 9 * (2^(1/2)) / 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l30_3038


namespace NUMINAMATH_CALUDE_wang_loss_l30_3081

/-- Represents the financial transaction in Mr. Wang's store --/
structure Transaction where
  gift_cost : ℕ
  gift_price : ℕ
  payment : ℕ
  change_given : ℕ
  returned_to_neighbor : ℕ

/-- Calculates the loss in the transaction --/
def calculate_loss (t : Transaction) : ℕ :=
  t.change_given + t.gift_cost + t.returned_to_neighbor - t.payment

/-- Theorem stating that Mr. Wang's loss in the given transaction is $97 --/
theorem wang_loss (t : Transaction) 
  (h1 : t.gift_cost = 18)
  (h2 : t.gift_price = 21)
  (h3 : t.payment = 100)
  (h4 : t.change_given = 79)
  (h5 : t.returned_to_neighbor = 100) : 
  calculate_loss t = 97 := by
  sorry

#eval calculate_loss { gift_cost := 18, gift_price := 21, payment := 100, change_given := 79, returned_to_neighbor := 100 }

end NUMINAMATH_CALUDE_wang_loss_l30_3081


namespace NUMINAMATH_CALUDE_ounces_per_cup_l30_3080

theorem ounces_per_cup (container_capacity : ℕ) (soap_per_cup : ℕ) (total_soap : ℕ) :
  container_capacity = 40 ∧ soap_per_cup = 3 ∧ total_soap = 15 →
  ∃ (ounces_per_cup : ℕ), ounces_per_cup = 8 ∧ container_capacity = ounces_per_cup * (total_soap / soap_per_cup) :=
by sorry

end NUMINAMATH_CALUDE_ounces_per_cup_l30_3080


namespace NUMINAMATH_CALUDE_circular_garden_radius_l30_3026

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2 → r = 12 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l30_3026


namespace NUMINAMATH_CALUDE_set_inclusion_condition_l30_3098

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 2 * x + 3}
def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = x ^ 2}

-- State the theorem
theorem set_inclusion_condition (a : ℝ) :
  C a ⊆ B a ↔ (1/2 ≤ a ∧ a ≤ 2) ∨ (a ≥ 3) ∨ (a < -2) :=
sorry

end NUMINAMATH_CALUDE_set_inclusion_condition_l30_3098


namespace NUMINAMATH_CALUDE_trapezoid_bc_length_l30_3062

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  ab : ℝ
  cd : ℝ

/-- Theorem stating the length of BC in the trapezoid -/
theorem trapezoid_bc_length (t : Trapezoid) 
  (h_area : t.area = 180)
  (h_altitude : t.altitude = 8)
  (h_ab : t.ab = 14)
  (h_cd : t.cd = 20) :
  ∃ (bc : ℝ), bc = 22.5 - Real.sqrt 33 - 2 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_bc_length_l30_3062


namespace NUMINAMATH_CALUDE_exact_three_primes_l30_3004

/-- The polynomial function f(n) = n^3 - 8n^2 + 20n - 13 -/
def f (n : ℕ) : ℤ := n^3 - 8*n^2 + 20*n - 13

/-- Predicate for primality -/
def isPrime (n : ℤ) : Prop := n > 1 ∧ (∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0))

theorem exact_three_primes : 
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n ∈ s, isPrime (f n) ∧ 
    ∀ n : ℕ, n > 0 → isPrime (f n) → n ∈ s :=
sorry

end NUMINAMATH_CALUDE_exact_three_primes_l30_3004


namespace NUMINAMATH_CALUDE_william_max_riding_time_l30_3086

/-- Represents the maximum number of hours William can ride his horse per day -/
def max_riding_time : ℝ := 6

/-- The total number of days William rode -/
def total_days : ℕ := 6

/-- The number of days William rode for the maximum time -/
def max_time_days : ℕ := 2

/-- The number of days William rode for 1.5 hours -/
def short_ride_days : ℕ := 2

/-- The number of days William rode for half the maximum time -/
def half_time_days : ℕ := 2

/-- The duration of a short ride in hours -/
def short_ride_duration : ℝ := 1.5

/-- The total riding time over all days in hours -/
def total_riding_time : ℝ := 21

theorem william_max_riding_time :
  max_riding_time * max_time_days +
  short_ride_duration * short_ride_days +
  (max_riding_time / 2) * half_time_days = total_riding_time ∧
  max_time_days + short_ride_days + half_time_days = total_days :=
by sorry

end NUMINAMATH_CALUDE_william_max_riding_time_l30_3086


namespace NUMINAMATH_CALUDE_quinns_reading_challenge_l30_3044

/-- Proves the number of weeks Quinn needs to participate in the reading challenge -/
theorem quinns_reading_challenge
  (books_per_donut : ℕ)
  (books_per_week : ℕ)
  (target_donuts : ℕ)
  (h1 : books_per_donut = 5)
  (h2 : books_per_week = 2)
  (h3 : target_donuts = 4) :
  (target_donuts * books_per_donut) / books_per_week = 10 :=
by sorry

end NUMINAMATH_CALUDE_quinns_reading_challenge_l30_3044


namespace NUMINAMATH_CALUDE_work_remaining_fraction_l30_3028

theorem work_remaining_fraction 
  (days_a : ℝ) (days_b : ℝ) (days_c : ℝ) (work_days : ℝ) 
  (h1 : days_a = 10) 
  (h2 : days_b = 20) 
  (h3 : days_c = 30) 
  (h4 : work_days = 5) : 
  1 - work_days * (1 / days_a + 1 / days_b + 1 / days_c) = 5 / 60 := by
  sorry

end NUMINAMATH_CALUDE_work_remaining_fraction_l30_3028


namespace NUMINAMATH_CALUDE_trees_after_typhoon_l30_3097

/-- The number of trees Haley initially grew -/
def initial_trees : ℕ := 17

/-- The number of trees that died after the typhoon -/
def dead_trees : ℕ := 5

/-- Theorem stating that the number of trees left after the typhoon is 12 -/
theorem trees_after_typhoon : initial_trees - dead_trees = 12 := by
  sorry

end NUMINAMATH_CALUDE_trees_after_typhoon_l30_3097


namespace NUMINAMATH_CALUDE_max_sum_on_ellipse_l30_3005

theorem max_sum_on_ellipse :
  ∀ x y : ℝ, (x - 2)^2 / 4 + (y - 1)^2 = 1 →
  ∀ x' y' : ℝ, (x' - 2)^2 / 4 + (y' - 1)^2 = 1 →
  x + y ≤ 3 + Real.sqrt 5 ∧
  ∃ x₀ y₀ : ℝ, (x₀ - 2)^2 / 4 + (y₀ - 1)^2 = 1 ∧ x₀ + y₀ = 3 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_ellipse_l30_3005


namespace NUMINAMATH_CALUDE_m_greater_than_n_l30_3035

/-- Given two quadratic functions M and N, prove that M > N for all real x. -/
theorem m_greater_than_n : ∀ x : ℝ, (x^2 - 3*x + 7) > (-x^2 + x + 1) := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l30_3035


namespace NUMINAMATH_CALUDE_childrens_home_total_l30_3025

theorem childrens_home_total (toddlers teenagers newborns : ℕ) : 
  teenagers = 5 * toddlers →
  toddlers = 6 →
  newborns = 4 →
  toddlers + teenagers + newborns = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_childrens_home_total_l30_3025


namespace NUMINAMATH_CALUDE_stephen_pizza_percentage_l30_3011

theorem stephen_pizza_percentage (total_slices : ℕ) (stephen_percentage : ℚ) (pete_percentage : ℚ) (remaining_slices : ℕ) : 
  total_slices = 24 →
  pete_percentage = 1/2 →
  remaining_slices = 9 →
  (1 - stephen_percentage) * total_slices * (1 - pete_percentage) = remaining_slices →
  stephen_percentage = 1/4 := by
sorry

end NUMINAMATH_CALUDE_stephen_pizza_percentage_l30_3011


namespace NUMINAMATH_CALUDE_area_of_triangle_AGE_l30_3045

/-- Square ABCD with side length 5 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (5, 0) ∧ C = (5, 5) ∧ D = (0, 5))

/-- Point E on side BC such that BE = 2 and EC = 3 -/
def E : ℝ × ℝ := (5, 2)

/-- Point G is the second intersection of circumcircle of ABE with diagonal BD -/
def G : Square → ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem area_of_triangle_AGE (s : Square) :
  triangle_area s.A (G s) E = 44.5 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AGE_l30_3045


namespace NUMINAMATH_CALUDE_y_equals_negative_two_at_x_two_l30_3013

/-- A linear function y = kx - 1 where y decreases as x increases -/
structure DecreasingLinearFunction where
  k : ℝ
  h1 : k < 0

/-- The value of y when x = 2 for a decreasing linear function -/
def y_at_2 (f : DecreasingLinearFunction) : ℝ :=
  f.k * 2 - 1

/-- Theorem stating that y = -2 when x = 2 for a decreasing linear function -/
theorem y_equals_negative_two_at_x_two (f : DecreasingLinearFunction) :
  y_at_2 f = -2 :=
sorry

end NUMINAMATH_CALUDE_y_equals_negative_two_at_x_two_l30_3013


namespace NUMINAMATH_CALUDE_last_digit_of_n_is_five_l30_3059

def sum_powers (n : ℕ) : ℕ := (Finset.range (2*n - 2)).sum (λ i => n^(i + 1))

theorem last_digit_of_n_is_five (n : ℕ) (h1 : n ≥ 3) (h2 : Nat.Prime (sum_powers n - 4)) :
  n % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_n_is_five_l30_3059


namespace NUMINAMATH_CALUDE_xyz_value_l30_3002

theorem xyz_value (a b c x y z : ℂ)
  (eq1 : a = (b + c) / (x - 2))
  (eq2 : b = (c + a) / (y - 2))
  (eq3 : c = (a + b) / (z - 2))
  (sum_prod : x * y + y * z + z * x = 67)
  (sum : x + y + z = 2010) :
  x * y * z = -5892 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l30_3002


namespace NUMINAMATH_CALUDE_chess_tournament_games_l30_3037

theorem chess_tournament_games (n : ℕ) 
  (h1 : n > 0) 
  (h2 : (10 * 9 * n) / 2 = 90) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l30_3037


namespace NUMINAMATH_CALUDE_floor_sum_equals_negative_one_l30_3006

theorem floor_sum_equals_negative_one : ⌊(18.7 : ℝ)⌋ + ⌊(-18.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_equals_negative_one_l30_3006


namespace NUMINAMATH_CALUDE_find_y_value_l30_3024

theorem find_y_value : ∃ y : ℝ, (15^2 * y^3) / 256 = 450 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l30_3024


namespace NUMINAMATH_CALUDE_line_angle_and_triangle_conditions_l30_3040

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def l₁ : Line := { a := 2, b := -1, c := -10 }
def l₂ : Line := { a := 4, b := 3, c := -10 }
def l₃ (a : ℝ) : Line := { a := a, b := 2, c := -8 }

/-- The angle between two lines -/
def angle_between (l1 l2 : Line) : ℝ := sorry

/-- Whether three lines can form a triangle -/
def can_form_triangle (l1 l2 l3 : Line) : Prop := sorry

theorem line_angle_and_triangle_conditions :
  (angle_between l₁ l₂ = Real.arctan 2) ∧
  (∀ a : ℝ, ¬(can_form_triangle l₁ l₂ (l₃ a)) ↔ (a = -4 ∨ a = 8/3 ∨ a = 3)) := by sorry

end NUMINAMATH_CALUDE_line_angle_and_triangle_conditions_l30_3040


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l30_3096

theorem subtraction_multiplication_equality : 10111 - 10 * 2 * 5 = 10011 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l30_3096


namespace NUMINAMATH_CALUDE_odd_increasing_function_inequality_l30_3089

-- Define an odd function that is increasing on [0,+∞)
def OddIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x y, 0 ≤ x → x < y → f x < f y)

-- State the theorem
theorem odd_increasing_function_inequality (f : ℝ → ℝ) (h : OddIncreasingFunction f) :
  ∀ x : ℝ, f (Real.log x) < 0 → 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_increasing_function_inequality_l30_3089


namespace NUMINAMATH_CALUDE_negation_equivalence_l30_3014

/-- A function f: ℝ → ℝ is monotonically increasing on (0, +∞) if for all x₁, x₂ ∈ (0, +∞),
    x₁ < x₂ implies f(x₁) < f(x₂) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The negation of the existence of a real k such that y = k/x is monotonically increasing
    on (0, +∞) is equivalent to the statement that for all real k, y = k/x is not
    monotonically increasing on (0, +∞) -/
theorem negation_equivalence : 
  (¬ ∃ k : ℝ, MonotonicallyIncreasing (fun x ↦ k / x)) ↔ 
  (∀ k : ℝ, ¬ MonotonicallyIncreasing (fun x ↦ k / x)) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l30_3014


namespace NUMINAMATH_CALUDE_factor_implies_s_value_l30_3099

theorem factor_implies_s_value (m s : ℝ) : 
  (m - 8) ∣ (m^2 - s*m - 24) → s = 5 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_s_value_l30_3099


namespace NUMINAMATH_CALUDE_f_properties_l30_3091

open Real

/-- The function f defined by the given conditions -/
noncomputable def f (x : ℝ) : ℝ := x / (1 + 2 * x^2)

/-- The theorem stating the properties of f -/
theorem f_properties :
  ∀ α β x y : ℝ,
  (sin (2 * α + β) = 3 * sin β) →
  (tan α = x) →
  (tan β = y) →
  (y = f x) →
  (0 < α) →
  (α < π / 3) →
  (∀ z : ℝ, 0 < z → z < f x → z < sqrt 2 / 4) ∧
  (f x ≤ sqrt 2 / 4) ∧
  (∃ z : ℝ, 0 < z ∧ z < sqrt 2 / 4 ∧ z = f x) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l30_3091


namespace NUMINAMATH_CALUDE_sum_of_cubes_and_fourth_powers_l30_3046

theorem sum_of_cubes_and_fourth_powers (a b : ℝ) 
  (sum_eq : a + b = 2) 
  (sum_squares_eq : a^2 + b^2 = 2) : 
  a^3 + b^3 = 2 ∧ a^4 + b^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_and_fourth_powers_l30_3046
