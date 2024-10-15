import Mathlib

namespace NUMINAMATH_GPT_simplify_expression_l1057_105718

-- Define the hypotheses and the expression.
variables (x : ℚ)
def expr := (1 + 1 / x) * (1 - 2 / (x + 1)) * (1 + 2 / (x - 1))

-- Define the conditions.
def valid_x : Prop := (x ≠ 0) ∧ (x ≠ -1) ∧ (x ≠ 1)

-- State the main theorem.
theorem simplify_expression (h : valid_x x) : expr x = (x + 1) / x := 
sorry

end NUMINAMATH_GPT_simplify_expression_l1057_105718


namespace NUMINAMATH_GPT_right_triangle_ratio_l1057_105740

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

noncomputable def radius_circumscribed_circle (hypo : ℝ) : ℝ := hypo / 2

noncomputable def radius_inscribed_circle (a b hypo : ℝ) : ℝ := (a + b - hypo) / 2

theorem right_triangle_ratio 
  (a b : ℝ) 
  (ha : a = 6) 
  (hb : b = 8) 
  (right_angle : a^2 + b^2 = hypotenuse a b ^ 2) :
  radius_inscribed_circle a b (hypotenuse a b) / radius_circumscribed_circle (hypotenuse a b) = 2 / 5 :=
by {
  -- The proof to be filled in
  sorry
}

end NUMINAMATH_GPT_right_triangle_ratio_l1057_105740


namespace NUMINAMATH_GPT_triangle_sides_l1057_105784

theorem triangle_sides
  (D E : ℝ) (DE EF : ℝ)
  (h1 : Real.cos (2 * D - E) + Real.sin (D + E) = 2)
  (hDE : DE = 6) :
  EF = 3 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_triangle_sides_l1057_105784


namespace NUMINAMATH_GPT_min_distance_circle_tangent_l1057_105734

theorem min_distance_circle_tangent
  (P : ℝ × ℝ)
  (hP: 3 * P.1 + 4 * P.2 = 11) :
  ∃ d : ℝ, d = 11 / 5 := 
sorry

end NUMINAMATH_GPT_min_distance_circle_tangent_l1057_105734


namespace NUMINAMATH_GPT_intersecting_lines_l1057_105700

-- Definitions based on conditions
def line1 (m : ℝ) (x : ℝ) : ℝ := m * x + 4
def line2 (b : ℝ) (x : ℝ) : ℝ := 3 * x + b

-- Lean 4 Statement of the problem
theorem intersecting_lines (m b : ℝ) (h1 : line1 m 6 = 10) (h2 : line2 b 6 = 10) : b + m = -7 :=
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_l1057_105700


namespace NUMINAMATH_GPT_dans_car_mpg_l1057_105787

noncomputable def milesPerGallon (distance money gas_price : ℝ) : ℝ :=
  distance / (money / gas_price)

theorem dans_car_mpg :
  let gas_price := 4
  let distance := 432
  let money := 54
  milesPerGallon distance money gas_price = 32 :=
by
  simp [milesPerGallon]
  sorry

end NUMINAMATH_GPT_dans_car_mpg_l1057_105787


namespace NUMINAMATH_GPT_median_of_first_fifteen_positive_integers_l1057_105768

theorem median_of_first_fifteen_positive_integers : ∃ m : ℝ, m = 7.5 :=
by
  let seq := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median := (seq[6] + seq[7]) / 2
  use median
  sorry

end NUMINAMATH_GPT_median_of_first_fifteen_positive_integers_l1057_105768


namespace NUMINAMATH_GPT_perpendicular_planes_parallel_l1057_105796

-- Define the lines m and n, and planes alpha and beta
def Line := Unit
def Plane := Unit

-- Define perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p₁ p₂ : Plane) : Prop := sorry

-- The main theorem statement: If m ⊥ α and m ⊥ β, then α ∥ β
theorem perpendicular_planes_parallel (m : Line) (α β : Plane)
  (h₁ : perpendicular m α) (h₂ : perpendicular m β) : parallel α β :=
sorry

end NUMINAMATH_GPT_perpendicular_planes_parallel_l1057_105796


namespace NUMINAMATH_GPT_max_balls_of_clay_l1057_105770

theorem max_balls_of_clay (radius cube_side_length : ℝ) (V_cube : ℝ) (V_ball : ℝ) (num_balls : ℕ) :
  radius = 3 ->
  cube_side_length = 10 ->
  V_cube = cube_side_length ^ 3 ->
  V_ball = (4 / 3) * π * radius ^ 3 ->
  num_balls = ⌊ V_cube / V_ball ⌋ ->
  num_balls = 8 :=
by
  sorry

end NUMINAMATH_GPT_max_balls_of_clay_l1057_105770


namespace NUMINAMATH_GPT_leftover_space_desks_bookcases_l1057_105756

theorem leftover_space_desks_bookcases 
  (number_of_desks : ℕ) (number_of_bookcases : ℕ)
  (wall_length : ℝ) (desk_length : ℝ) (bookcase_length : ℝ) (space_between : ℝ)
  (equal_number : number_of_desks = number_of_bookcases)
  (wall_length_eq : wall_length = 15)
  (desk_length_eq : desk_length = 2)
  (bookcase_length_eq : bookcase_length = 1.5)
  (space_between_eq : space_between = 0.5) :
  ∃ k : ℝ, k = 3 := 
by
  sorry

end NUMINAMATH_GPT_leftover_space_desks_bookcases_l1057_105756


namespace NUMINAMATH_GPT_elvis_squares_count_l1057_105731

theorem elvis_squares_count :
  ∀ (total : ℕ) (Elvis_squares Ralph_squares squares_used_by_Ralph matchsticks_left : ℕ)
  (uses_by_Elvis_per_square uses_by_Ralph_per_square : ℕ),
  total = 50 →
  uses_by_Elvis_per_square = 4 →
  uses_by_Ralph_per_square = 8 →
  Ralph_squares = 3 →
  matchsticks_left = 6 →
  squares_used_by_Ralph = Ralph_squares * uses_by_Ralph_per_square →
  total = (Elvis_squares * uses_by_Elvis_per_square) + squares_used_by_Ralph + matchsticks_left →
  Elvis_squares = 5 :=
by
  sorry

end NUMINAMATH_GPT_elvis_squares_count_l1057_105731


namespace NUMINAMATH_GPT_percentage_decrease_l1057_105725

theorem percentage_decrease (x : ℝ) 
  (h1 : 400 * (1 - x / 100) * 1.40 = 476) : 
  x = 15 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_decrease_l1057_105725


namespace NUMINAMATH_GPT_households_with_two_types_of_vehicles_households_with_exactly_one_type_of_vehicle_households_with_at_least_one_type_of_vehicle_l1057_105720

namespace VehicleHouseholds

-- Definitions for the conditions
def totalHouseholds : ℕ := 250
def householdsNoVehicles : ℕ := 25
def householdsAllVehicles : ℕ := 36
def householdsCarOnly : ℕ := 62
def householdsBikeOnly : ℕ := 45
def householdsScooterOnly : ℕ := 30

-- Proof Statements
theorem households_with_two_types_of_vehicles :
  (totalHouseholds - householdsNoVehicles - householdsAllVehicles - 
  (householdsCarOnly + householdsBikeOnly + householdsScooterOnly)) = 52 := by
  sorry

theorem households_with_exactly_one_type_of_vehicle :
  (householdsCarOnly + householdsBikeOnly + householdsScooterOnly) = 137 := by
  sorry

theorem households_with_at_least_one_type_of_vehicle :
  (totalHouseholds - householdsNoVehicles) = 225 := by
  sorry

end VehicleHouseholds

end NUMINAMATH_GPT_households_with_two_types_of_vehicles_households_with_exactly_one_type_of_vehicle_households_with_at_least_one_type_of_vehicle_l1057_105720


namespace NUMINAMATH_GPT_simple_interest_calculation_l1057_105778

variable (P : ℝ) (R : ℝ) (T : ℝ)

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem simple_interest_calculation (hP : P = 10000) (hR : R = 0.09) (hT : T = 1) :
    simple_interest P R T = 900 := by
  rw [hP, hR, hT]
  sorry

end NUMINAMATH_GPT_simple_interest_calculation_l1057_105778


namespace NUMINAMATH_GPT_remaining_customers_after_some_left_l1057_105739

-- Define the initial conditions and question (before proving it)
def initial_customers := 8
def new_customers := 99
def total_customers_after_new := 104

-- Define the hypothesis based on the total customers after new customers added
theorem remaining_customers_after_some_left (x : ℕ) (h : x + new_customers = total_customers_after_new) : x = 5 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_remaining_customers_after_some_left_l1057_105739


namespace NUMINAMATH_GPT_simplify_expression_l1057_105733

variable {R : Type*} [CommRing R]

theorem simplify_expression (x y : R) : (x + 2 * y) * (x - 2 * y) - y * (3 - 4 * y) = x^2 - 3 * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1057_105733


namespace NUMINAMATH_GPT_calories_per_serving_is_120_l1057_105792

-- Define the conditions
def servings : ℕ := 3
def halfCalories : ℕ := 180
def totalCalories : ℕ := 2 * halfCalories

-- Define the target value
def caloriesPerServing : ℕ := totalCalories / servings

-- The proof goal
theorem calories_per_serving_is_120 : caloriesPerServing = 120 :=
by 
  sorry

end NUMINAMATH_GPT_calories_per_serving_is_120_l1057_105792


namespace NUMINAMATH_GPT_fourth_grade_planted_89_l1057_105794

-- Define the number of trees planted by the fifth grade
def fifth_grade_trees : Nat := 114

-- Define the condition that the fifth grade planted twice as many trees as the third grade
def third_grade_trees : Nat := fifth_grade_trees / 2

-- Define the condition that the fourth grade planted 32 more trees than the third grade
def fourth_grade_trees : Nat := third_grade_trees + 32

-- Theorem to prove the number of trees planted by the fourth grade is 89
theorem fourth_grade_planted_89 : fourth_grade_trees = 89 := by
  sorry

end NUMINAMATH_GPT_fourth_grade_planted_89_l1057_105794


namespace NUMINAMATH_GPT_average_age_after_leaves_is_27_l1057_105747

def average_age_of_remaining_people (initial_avg_age : ℕ) (initial_people_count : ℕ) 
    (age_leave1 : ℕ) (age_leave2 : ℕ) (remaining_people_count : ℕ) : ℕ :=
  let initial_total_age := initial_avg_age * initial_people_count
  let new_total_age := initial_total_age - (age_leave1 + age_leave2)
  new_total_age / remaining_people_count

theorem average_age_after_leaves_is_27 :
  average_age_of_remaining_people 25 6 20 22 4 = 27 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_average_age_after_leaves_is_27_l1057_105747


namespace NUMINAMATH_GPT_rectangular_solid_volume_l1057_105716

theorem rectangular_solid_volume (a b c : ℝ) 
  (h1 : a * b = 15) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : a * b * c = 60 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_solid_volume_l1057_105716


namespace NUMINAMATH_GPT_divisor_of_p_l1057_105724

-- Define the necessary variables and assumptions
variables (p q r s : ℕ)

-- State the conditions
def conditions := gcd p q = 28 ∧ gcd q r = 45 ∧ gcd r s = 63 ∧ 80 < gcd s p ∧ gcd s p < 120 

-- State the proposition to prove: 11 divides p
theorem divisor_of_p (h : conditions p q r s) : 11 ∣ p := 
sorry

end NUMINAMATH_GPT_divisor_of_p_l1057_105724


namespace NUMINAMATH_GPT_students_in_class_l1057_105777

theorem students_in_class (S : ℕ) 
  (h1 : (1 / 4) * (9 / 10 : ℚ) * S = 9) : S = 40 :=
sorry

end NUMINAMATH_GPT_students_in_class_l1057_105777


namespace NUMINAMATH_GPT_total_profit_l1057_105776

-- Define the relevant variables and conditions
variables (x y : ℝ) -- Cost prices of the two music players

-- Given conditions
axiom cost_price_first : x * 1.2 = 132
axiom cost_price_second : y * 1.1 = 132

theorem total_profit : 132 + 132 - y - x = 34 :=
by
  -- The proof body is not required
  sorry

end NUMINAMATH_GPT_total_profit_l1057_105776


namespace NUMINAMATH_GPT_solve_equation_l1057_105783

theorem solve_equation :
  ∀ x : ℝ, 81 * (1 - x) ^ 2 = 64 ↔ x = 1 / 9 ∨ x = 17 / 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1057_105783


namespace NUMINAMATH_GPT_total_fiscal_revenue_scientific_notation_l1057_105797

theorem total_fiscal_revenue_scientific_notation : 
  ∃ a n, (1073 * 10^8 : ℝ) = a * 10^n ∧ (1 ≤ |a| ∧ |a| < 10) ∧ a = 1.07 ∧ n = 11 :=
by
  use 1.07, 11
  simp
  sorry

end NUMINAMATH_GPT_total_fiscal_revenue_scientific_notation_l1057_105797


namespace NUMINAMATH_GPT_parabola_point_distance_to_focus_l1057_105763

theorem parabola_point_distance_to_focus :
  ∀ (x y : ℝ), (y^2 = 12 * x) → (∃ (xf : ℝ), xf = 3 ∧ 0 ≤ y) → (∃ (d : ℝ), d = 7) → x = 4 :=
by
  intros x y parabola_focus distance_to_focus distance
  sorry

end NUMINAMATH_GPT_parabola_point_distance_to_focus_l1057_105763


namespace NUMINAMATH_GPT_qiuqiu_servings_l1057_105781

-- Define the volume metrics
def bottles : ℕ := 1
def cups_per_bottle_kangkang : ℕ := 4
def foam_expansion : ℕ := 3
def foam_fraction : ℚ := 1 / 2

-- Calculate the effective cup volume under Qiuqiu's serving method
def beer_fraction_per_cup_qiuqiu : ℚ := 1 / 2 + (1 / foam_expansion) * foam_fraction

-- Calculate the number of cups Qiuqiu can serve from one bottle
def qiuqiu_cups_from_bottle : ℚ := cups_per_bottle_kangkang / beer_fraction_per_cup_qiuqiu

-- The theorem statement
theorem qiuqiu_servings :
  qiuqiu_cups_from_bottle = 6 := by
  sorry

end NUMINAMATH_GPT_qiuqiu_servings_l1057_105781


namespace NUMINAMATH_GPT_factor_difference_of_cubes_l1057_105760

theorem factor_difference_of_cubes (t : ℝ) : 
  t^3 - 125 = (t - 5) * (t^2 + 5 * t + 25) :=
sorry

end NUMINAMATH_GPT_factor_difference_of_cubes_l1057_105760


namespace NUMINAMATH_GPT_contrapositive_of_a_gt_1_then_a_sq_gt_1_l1057_105773

theorem contrapositive_of_a_gt_1_then_a_sq_gt_1 : 
  (∀ a : ℝ, a > 1 → a^2 > 1) → (∀ a : ℝ, a^2 ≤ 1 → a ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_contrapositive_of_a_gt_1_then_a_sq_gt_1_l1057_105773


namespace NUMINAMATH_GPT_diagonal_BD_eq_diagonal_AD_eq_l1057_105774

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-1, 2⟩
def C : Point := ⟨5, 4⟩
def line_AB (p : Point) : Prop := p.x - p.y + 3 = 0

theorem diagonal_BD_eq :
  (∃ M : Point, M = ⟨2, 3⟩) →
  (∃ BD : Point → Prop, (BD = fun p => 3*p.x + p.y - 9 = 0)) :=
by
  sorry

theorem diagonal_AD_eq :
  (∃ M : Point, M = ⟨2, 3⟩) →
  (∃ AD : Point → Prop, (AD = fun p => p.x + 7*p.y - 13 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_diagonal_BD_eq_diagonal_AD_eq_l1057_105774


namespace NUMINAMATH_GPT_index_card_area_l1057_105759

theorem index_card_area 
  (a b : ℕ)
  (ha : a = 5)
  (hb : b = 7)
  (harea : (a - 2) * b = 21) :
  (a * (b - 2) = 25) :=
by
  sorry

end NUMINAMATH_GPT_index_card_area_l1057_105759


namespace NUMINAMATH_GPT_matthew_hotdogs_needed_l1057_105799

def total_hotdogs (ella_hotdogs emma_hotdogs : ℕ) (luke_multiplier hunter_multiplier : ℕ) : ℕ :=
  let total_sisters := ella_hotdogs + emma_hotdogs
  let luke := luke_multiplier * total_sisters
  let hunter := hunter_multiplier * total_sisters / 2  -- because 1.5 = 3/2 and hunter_multiplier = 3
  total_sisters + luke + hunter

theorem matthew_hotdogs_needed :
  total_hotdogs 2 2 2 3 = 18 := 
by
  -- This proof is correct given the calculations above
  sorry

end NUMINAMATH_GPT_matthew_hotdogs_needed_l1057_105799


namespace NUMINAMATH_GPT_mean_absolute_temperature_correct_l1057_105779

noncomputable def mean_absolute_temperature (temps : List ℝ) : ℝ :=
  (temps.map (λ x => |x|)).sum / temps.length

theorem mean_absolute_temperature_correct :
  mean_absolute_temperature [-6, -3, -3, -6, 0, 4, 3] = 25 / 7 :=
by
  sorry

end NUMINAMATH_GPT_mean_absolute_temperature_correct_l1057_105779


namespace NUMINAMATH_GPT_alice_sales_goal_l1057_105702

def price_adidas := 45
def price_nike := 60
def price_reeboks := 35
def price_puma := 50
def price_converse := 40

def num_adidas := 10
def num_nike := 12
def num_reeboks := 15
def num_puma := 8
def num_converse := 14

def quota := 2000

def total_sales :=
  (num_adidas * price_adidas) +
  (num_nike * price_nike) +
  (num_reeboks * price_reeboks) +
  (num_puma * price_puma) +
  (num_converse * price_converse)

def exceed_amount := total_sales - quota

theorem alice_sales_goal : exceed_amount = 655 := by
  -- calculation steps would go here
  sorry

end NUMINAMATH_GPT_alice_sales_goal_l1057_105702


namespace NUMINAMATH_GPT_polynomial_bound_l1057_105745

noncomputable def P (x : ℝ) : ℝ := sorry  -- Placeholder for the polynomial P(x)

theorem polynomial_bound (n : ℕ) (hP : ∀ y : ℝ, 0 ≤ y ∧ y ≤ 1 → |P y| ≤ 1) :
  P (-1 / n) ≤ 2^(n + 1) - 1 :=
sorry

end NUMINAMATH_GPT_polynomial_bound_l1057_105745


namespace NUMINAMATH_GPT_inequality_proof_l1057_105755

variables {a b c : ℝ}

theorem inequality_proof (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * c < b * c :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1057_105755


namespace NUMINAMATH_GPT_intersection_y_axis_parabola_l1057_105732

theorem intersection_y_axis_parabola : (0, -4) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, x^2 - 4) ∧ x = 0 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_y_axis_parabola_l1057_105732


namespace NUMINAMATH_GPT_andrew_friends_brought_food_l1057_105746

theorem andrew_friends_brought_food (slices_per_friend total_slices : ℕ) (h1 : slices_per_friend = 4) (h2 : total_slices = 16) :
  total_slices / slices_per_friend = 4 :=
by
  sorry

end NUMINAMATH_GPT_andrew_friends_brought_food_l1057_105746


namespace NUMINAMATH_GPT_abc_cube_geq_abc_sum_l1057_105749

theorem abc_cube_geq_abc_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a ^ a * b ^ b * c ^ c) ^ 3 ≥ (a * b * c) ^ (a + b + c) :=
by
  sorry

end NUMINAMATH_GPT_abc_cube_geq_abc_sum_l1057_105749


namespace NUMINAMATH_GPT_spheres_max_min_dist_l1057_105780

variable {R_1 R_2 d : ℝ}

noncomputable def max_min_dist (R_1 R_2 d : ℝ) (sep : d > R_1 + R_2) :
  ℝ × ℝ :=
(d + R_1 + R_2, d - R_1 - R_2)

theorem spheres_max_min_dist {R_1 R_2 d : ℝ} (sep : d > R_1 + R_2) :
  max_min_dist R_1 R_2 d sep = (d + R_1 + R_2, d - R_1 - R_2) := by
sorry

end NUMINAMATH_GPT_spheres_max_min_dist_l1057_105780


namespace NUMINAMATH_GPT_quad_form_unique_solution_l1057_105701

theorem quad_form_unique_solution (d e f : ℤ) (h1 : d * d = 16) (h2 : 2 * d * e = -40) (h3 : e * e + f = -56) : d * e = -20 :=
by sorry

end NUMINAMATH_GPT_quad_form_unique_solution_l1057_105701


namespace NUMINAMATH_GPT_largest_integral_solution_l1057_105765

theorem largest_integral_solution : ∃ x : ℤ, (1 / 4 < x / 7 ∧ x / 7 < 3 / 5) ∧ ∀ y : ℤ, (1 / 4 < y / 7 ∧ y / 7 < 3 / 5) → y ≤ x := sorry

end NUMINAMATH_GPT_largest_integral_solution_l1057_105765


namespace NUMINAMATH_GPT_calculate_expression_l1057_105707

theorem calculate_expression :
  (2^3 * 3 * 5) + (18 / 2) = 129 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_calculate_expression_l1057_105707


namespace NUMINAMATH_GPT_fractional_part_frustum_l1057_105772

noncomputable def base_edge : ℝ := 24
noncomputable def original_altitude : ℝ := 18
noncomputable def smaller_altitude : ℝ := original_altitude / 3

noncomputable def volume_pyramid (base_edge : ℝ) (altitude : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * altitude

noncomputable def volume_original : ℝ := volume_pyramid base_edge original_altitude
noncomputable def similarity_ratio : ℝ := (smaller_altitude / original_altitude) ^ 3
noncomputable def volume_smaller : ℝ := similarity_ratio * volume_original
noncomputable def volume_frustum : ℝ := volume_original - volume_smaller

noncomputable def fractional_volume_frustum : ℝ := volume_frustum / volume_original

theorem fractional_part_frustum : fractional_volume_frustum = 26 / 27 := by
  sorry

end NUMINAMATH_GPT_fractional_part_frustum_l1057_105772


namespace NUMINAMATH_GPT_arc_length_150_deg_max_area_sector_l1057_105793

noncomputable def alpha := 150 * (Real.pi / 180)
noncomputable def r := 6
noncomputable def perimeter := 24

-- 1. Proving the arc length when α = 150° and r = 6
theorem arc_length_150_deg : alpha * r = 5 * Real.pi := by
  sorry

-- 2. Proving the maximum area and corresponding alpha given the perimeter of 24
theorem max_area_sector : ∃ (α : ℝ), α = 2 ∧ (1 / 2) * ((perimeter - 2 * r) * r) = 36 := by
  sorry

end NUMINAMATH_GPT_arc_length_150_deg_max_area_sector_l1057_105793


namespace NUMINAMATH_GPT_find_number_l1057_105743

theorem find_number (x : ℚ) (h : 0.5 * x = (3/5) * x - 10) : x = 100 := 
sorry

end NUMINAMATH_GPT_find_number_l1057_105743


namespace NUMINAMATH_GPT_geometric_sequence_a5_eq_8_l1057_105744

variable (a : ℕ → ℝ)
variable (n : ℕ)

-- Conditions
axiom pos (n : ℕ) : a n > 0
axiom prod_eq (a3 a7 : ℝ) : a 3 * a 7 = 64

-- Statement to prove
theorem geometric_sequence_a5_eq_8
  (pos : ∀ n, a n > 0)
  (prod_eq : a 3 * a 7 = 64) :
  a 5 = 8 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_eq_8_l1057_105744


namespace NUMINAMATH_GPT_tangent_line_to_ellipse_l1057_105786

variable (a b x y x₀ y₀ : ℝ)

-- Definitions
def is_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def point_on_ellipse (x₀ y₀ a b : ℝ) : Prop :=
  x₀^2 / a^2 + y₀^2 / b^2 = 1

-- Theorem
theorem tangent_line_to_ellipse
  (h₁ : point_on_ellipse x₀ y₀ a b) :
    (x₀ * x) / (a^2) + (y₀ * y) / (b^2) = 1 :=
sorry

end NUMINAMATH_GPT_tangent_line_to_ellipse_l1057_105786


namespace NUMINAMATH_GPT_remainder_of_7_9_power_2008_mod_64_l1057_105706

theorem remainder_of_7_9_power_2008_mod_64 :
  (7^2008 + 9^2008) % 64 = 2 := 
sorry

end NUMINAMATH_GPT_remainder_of_7_9_power_2008_mod_64_l1057_105706


namespace NUMINAMATH_GPT_least_positive_integer_to_multiple_of_5_l1057_105791

theorem least_positive_integer_to_multiple_of_5 : ∃ (n : ℕ), n > 0 ∧ (725 + n) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_to_multiple_of_5_l1057_105791


namespace NUMINAMATH_GPT_smallest_of_six_consecutive_even_numbers_l1057_105789

theorem smallest_of_six_consecutive_even_numbers (h : ∃ n : ℤ, (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 390) : ∃ m : ℤ, m = 60 :=
by
  have ex : ∃ n : ℤ, 6 * n + 6 = 390 := by sorry
  obtain ⟨n, hn⟩ := ex
  use (n - 4)
  sorry

end NUMINAMATH_GPT_smallest_of_six_consecutive_even_numbers_l1057_105789


namespace NUMINAMATH_GPT_denise_removed_bananas_l1057_105753

theorem denise_removed_bananas (initial_bananas remaining_bananas : ℕ) 
  (h_initial : initial_bananas = 46) (h_remaining : remaining_bananas = 41) : 
  initial_bananas - remaining_bananas = 5 :=
by
  sorry

end NUMINAMATH_GPT_denise_removed_bananas_l1057_105753


namespace NUMINAMATH_GPT_circle_circumference_l1057_105705

theorem circle_circumference (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi :=
by
  -- Use the given dimensions to find the diagonal (which is the diameter).
  -- Calculate the circumference using the calculated diameter.
  sorry

end NUMINAMATH_GPT_circle_circumference_l1057_105705


namespace NUMINAMATH_GPT_find_positive_difference_l1057_105767

theorem find_positive_difference 
  (p1 p2 : ℝ × ℝ) (q1 q2 : ℝ × ℝ) 
  (h_p1 : p1 = (0, 8)) (h_p2 : p2 = (4, 0))
  (h_q1 : q1 = (0, 5)) (h_q2 : q2 = (10, 0))
  (y : ℝ) (hy : y = 20) :
  let m_p := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b_p := p1.2 - m_p * p1.1
  let x_p := (y - b_p) / m_p
  let m_q := (q2.2 - q1.2) / (q2.1 - q1.1)
  let b_q := q1.2 - m_q * q1.1
  let x_q := (y - b_q) / m_q
  abs (x_p - x_q) = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_difference_l1057_105767


namespace NUMINAMATH_GPT_lcm_inequality_l1057_105766

theorem lcm_inequality (m n : ℕ) (h : n > m) :
  Nat.lcm m n + Nat.lcm (m + 1) (n + 1) ≥ 2 * m * n / Real.sqrt (n - m) := 
sorry

end NUMINAMATH_GPT_lcm_inequality_l1057_105766


namespace NUMINAMATH_GPT_tank_capacity_l1057_105790

theorem tank_capacity (w c: ℚ) (h1: w / c = 1 / 3) (h2: (w + 5) / c = 2 / 5) :
  c = 37.5 := 
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l1057_105790


namespace NUMINAMATH_GPT_lowest_temperature_l1057_105750

-- Define the temperatures in the four cities.
def temp_Harbin := -20
def temp_Beijing := -10
def temp_Hangzhou := 0
def temp_Jinhua := 2

-- The proof statement asserting the lowest temperature.
theorem lowest_temperature :
  min temp_Harbin (min temp_Beijing (min temp_Hangzhou temp_Jinhua)) = -20 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_lowest_temperature_l1057_105750


namespace NUMINAMATH_GPT_find_crew_members_l1057_105758

noncomputable def passengers_initial := 124
noncomputable def passengers_texas := passengers_initial - 58 + 24
noncomputable def passengers_nc := passengers_texas - 47 + 14
noncomputable def total_people_virginia := 67

theorem find_crew_members (passengers_initial passengers_texas passengers_nc total_people_virginia : ℕ) :
  passengers_initial = 124 →
  passengers_texas = passengers_initial - 58 + 24 →
  passengers_nc = passengers_texas - 47 + 14 →
  total_people_virginia = 67 →
  ∃ crew_members : ℕ, total_people_virginia = passengers_nc + crew_members ∧ crew_members = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_crew_members_l1057_105758


namespace NUMINAMATH_GPT_radius_of_inner_circle_l1057_105727

def right_triangle_legs (AC BC : ℝ) : Prop :=
  AC = 3 ∧ BC = 4

theorem radius_of_inner_circle (AC BC : ℝ) (h : right_triangle_legs AC BC) :
  ∃ r : ℝ, r = 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_inner_circle_l1057_105727


namespace NUMINAMATH_GPT_reciprocal_of_subtraction_l1057_105757

-- Defining the conditions
def x : ℚ := 1 / 9
def y : ℚ := 2 / 3

-- Defining the main theorem statement
theorem reciprocal_of_subtraction : (1 / (y - x)) = 9 / 5 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_subtraction_l1057_105757


namespace NUMINAMATH_GPT_product_not_perfect_power_l1057_105703

theorem product_not_perfect_power (n : ℕ) : ¬∃ (k : ℕ) (a : ℤ), k > 1 ∧ n * (n + 1) = a^k := by
  sorry

end NUMINAMATH_GPT_product_not_perfect_power_l1057_105703


namespace NUMINAMATH_GPT_max_rectangle_area_max_rectangle_area_exists_l1057_105711

theorem max_rectangle_area (l w : ℕ) (h : l + w = 20) : l * w ≤ 100 :=
by sorry

-- Alternatively, to also show the existence of the maximum value.
theorem max_rectangle_area_exists : ∃ l w : ℕ, l + w = 20 ∧ l * w = 100 :=
by sorry

end NUMINAMATH_GPT_max_rectangle_area_max_rectangle_area_exists_l1057_105711


namespace NUMINAMATH_GPT_algebraic_expression_equivalence_l1057_105722

theorem algebraic_expression_equivalence (x : ℝ) : 
  x^2 - 6*x + 10 = (x - 3)^2 + 1 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_equivalence_l1057_105722


namespace NUMINAMATH_GPT_proof_smallest_lcm_1_to_12_l1057_105751

noncomputable def smallest_lcm_1_to_12 : ℕ := 27720

theorem proof_smallest_lcm_1_to_12 :
  ∀ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ↔ n = 27720 :=
by
  sorry

end NUMINAMATH_GPT_proof_smallest_lcm_1_to_12_l1057_105751


namespace NUMINAMATH_GPT_add_fraction_l1057_105728

theorem add_fraction (x : ℚ) (h : x - 7/3 = 3/2) : x + 7/3 = 37/6 :=
by
  sorry

end NUMINAMATH_GPT_add_fraction_l1057_105728


namespace NUMINAMATH_GPT_missing_number_l1057_105710

theorem missing_number (x : ℝ) : (306 / x) * 15 + 270 = 405 ↔ x = 34 := 
by
  sorry

end NUMINAMATH_GPT_missing_number_l1057_105710


namespace NUMINAMATH_GPT_distinct_ordered_pairs_count_l1057_105712

theorem distinct_ordered_pairs_count :
  ∃ (n : ℕ), n = 29 ∧ (∀ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b → a + b = 30 → ∃! p : ℕ × ℕ, p = (a, b)) :=
sorry

end NUMINAMATH_GPT_distinct_ordered_pairs_count_l1057_105712


namespace NUMINAMATH_GPT_existence_of_epsilon_and_u_l1057_105738

theorem existence_of_epsilon_and_u (n : ℕ) (h : 0 < n) :
  ∀ k ≥ 1, ∃ ε : ℝ, (0 < ε ∧ ε < 1 / k) ∧
  (∀ (a : Fin n → ℝ), (∀ i, 0 < a i) → ∃ u > 0, ∀ i, ε < (u * a i - ⌊u * a i⌋) ∧ (u * a i - ⌊u * a i⌋) < 1 / k) :=
by {
  sorry
}

end NUMINAMATH_GPT_existence_of_epsilon_and_u_l1057_105738


namespace NUMINAMATH_GPT_range_of_c_value_of_c_given_perimeter_l1057_105785

variables (a b c : ℝ)

-- Question 1: Proving the range of values for c
theorem range_of_c (h1 : a + b = 3 * c - 2) (h2 : a - b = 2 * c - 6) :
  1 < c ∧ c < 6 :=
sorry

-- Question 2: Finding the value of c for a given perimeter
theorem value_of_c_given_perimeter (h1 : a + b = 3 * c - 2) (h2 : a - b = 2 * c - 6) (h3 : a + b + c = 18) :
  c = 5 :=
sorry

end NUMINAMATH_GPT_range_of_c_value_of_c_given_perimeter_l1057_105785


namespace NUMINAMATH_GPT_value_range_of_a_l1057_105752

variable (a : ℝ)
variable (suff_not_necess : ∀ x, x ∈ ({3, a} : Set ℝ) → 2 * x^2 - 5 * x - 3 ≥ 0)

theorem value_range_of_a :
  (a ≤ -1/2 ∨ a > 3) :=
sorry

end NUMINAMATH_GPT_value_range_of_a_l1057_105752


namespace NUMINAMATH_GPT_triangle_rectangle_ratio_l1057_105723

theorem triangle_rectangle_ratio (s b w l : ℕ) 
(h1 : 2 * s + b = 60) 
(h2 : 2 * (w + l) = 60) 
(h3 : 2 * w = l) 
(h4 : b = w) 
: s / w = 5 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_rectangle_ratio_l1057_105723


namespace NUMINAMATH_GPT_g_ln_1_over_2017_l1057_105730

theorem g_ln_1_over_2017 (a : ℝ) (h_a_pos : 0 < a) (h_a_neq_1 : a ≠ 1) (f g : ℝ → ℝ)
  (h_f_add : ∀ m n : ℝ, f (m + n) = f m + f n - 1)
  (h_g : ∀ x : ℝ, g x = f x + a^x / (a^x + 1))
  (h_g_ln_2017 : g (Real.log 2017) = 2018) :
  g (Real.log (1 / 2017)) = -2015 :=
sorry

end NUMINAMATH_GPT_g_ln_1_over_2017_l1057_105730


namespace NUMINAMATH_GPT_correct_product_of_a_and_b_l1057_105736

theorem correct_product_of_a_and_b
    (a b : ℕ)
    (h_a_pos : 0 < a)
    (h_b_pos : 0 < b)
    (h_a_two_digits : 10 ≤ a ∧ a < 100)
    (a' : ℕ)
    (h_a' : a' = (a % 10) * 10 + (a / 10))
    (h_product_erroneous : a' * b = 198) :
  a * b = 198 :=
sorry

end NUMINAMATH_GPT_correct_product_of_a_and_b_l1057_105736


namespace NUMINAMATH_GPT_probability_dice_sum_perfect_square_l1057_105741

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end NUMINAMATH_GPT_probability_dice_sum_perfect_square_l1057_105741


namespace NUMINAMATH_GPT_election_winner_percentage_l1057_105775

theorem election_winner_percentage :
    let votes_candidate1 := 2500
    let votes_candidate2 := 5000
    let votes_candidate3 := 15000
    let total_votes := votes_candidate1 + votes_candidate2 + votes_candidate3
    let winning_votes := votes_candidate3
    (winning_votes / total_votes) * 100 = 75 := 
by 
    sorry

end NUMINAMATH_GPT_election_winner_percentage_l1057_105775


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1057_105719

theorem necessary_and_sufficient_condition 
  (a b c : ℝ) :
  (a^2 = b^2 + c^2) ↔
  (∃ x : ℝ, x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0) := 
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1057_105719


namespace NUMINAMATH_GPT_sum_in_range_l1057_105704

theorem sum_in_range :
  let a := (27 : ℚ) / 8
  let b := (22 : ℚ) / 5
  let c := (67 : ℚ) / 11
  13 < a + b + c ∧ a + b + c < 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_in_range_l1057_105704


namespace NUMINAMATH_GPT_remainder_sum_div_40_l1057_105709

variable (k m n : ℤ)
variables (a b c : ℤ)
variable (h1 : a % 80 = 75)
variable (h2 : b % 120 = 115)
variable (h3 : c % 160 = 155)

theorem remainder_sum_div_40 : (a + b + c) % 40 = 25 :=
by
  -- Use sorry as we are not required to fill in the proof
  sorry

end NUMINAMATH_GPT_remainder_sum_div_40_l1057_105709


namespace NUMINAMATH_GPT_problem1_problem2_l1057_105762

def f (x b : ℝ) : ℝ := |x - b| + |x + b|

theorem problem1 (x : ℝ) : (∀ y, y = 1 → f x y ≤ x + 2) ↔ (0 ≤ x ∧ x ≤ 2) :=
sorry

theorem problem2 (a b : ℝ) (h : a ≠ 0) : (∀ y, y = 1 → f y b ≥ (|a + 1| - |2 * a - 1|) / |a|) ↔ (b ≤ -3 / 2 ∨ b ≥ 3 / 2) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1057_105762


namespace NUMINAMATH_GPT_complex_proof_problem_l1057_105798

theorem complex_proof_problem (i : ℂ) (h1 : i^2 = -1) :
  (i^2 + i^3 + i^4) / (1 - i) = (1 / 2) - (1 / 2) * i :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_complex_proof_problem_l1057_105798


namespace NUMINAMATH_GPT_product_of_consecutive_integers_l1057_105748

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_l1057_105748


namespace NUMINAMATH_GPT_hyperbola_asymptote_l1057_105771

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) (hyp_eq : ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / 81 = 1 → y = 3 * x) : a = 3 := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l1057_105771


namespace NUMINAMATH_GPT_work_completion_time_l1057_105721

theorem work_completion_time (P W : ℕ) (h : P * 8 = W) : 2 * P * 2 = W / 2 := by
  sorry

end NUMINAMATH_GPT_work_completion_time_l1057_105721


namespace NUMINAMATH_GPT_loraine_total_wax_l1057_105761

-- Conditions
def large_animal_wax := 4
def small_animal_wax := 2
def small_animal_count := 12 / small_animal_wax
def large_animal_count := small_animal_count / 3
def total_wax := 12 + (large_animal_count * large_animal_wax)

-- The proof problem
theorem loraine_total_wax : total_wax = 20 := by
  sorry

end NUMINAMATH_GPT_loraine_total_wax_l1057_105761


namespace NUMINAMATH_GPT_original_proposition_false_implies_negation_true_l1057_105769

-- Define the original proposition and its negation
def original_proposition (x y : ℝ) : Prop := (x + y > 0) → (x > 0 ∧ y > 0)
def negation (x y : ℝ) : Prop := ¬ original_proposition x y

-- Theorem statement
theorem original_proposition_false_implies_negation_true (x y : ℝ) : ¬ original_proposition x y → negation x y :=
by
  -- Since ¬ original_proposition x y implies the negation is true
  intro h
  exact h

end NUMINAMATH_GPT_original_proposition_false_implies_negation_true_l1057_105769


namespace NUMINAMATH_GPT_smallest_angle_measure_l1057_105715

-- Define the conditions
def is_spherical_triangle (a b c : ℝ) : Prop :=
  a + b + c > 180 ∧ a + b + c < 540

def angles (k : ℝ) : Prop :=
  let a := 3 * k
  let b := 4 * k
  let c := 5 * k
  is_spherical_triangle a b c ∧ a + b + c = 270

-- Statement of the theorem
theorem smallest_angle_measure (k : ℝ) (h : angles k) : 3 * k = 67.5 :=
sorry

end NUMINAMATH_GPT_smallest_angle_measure_l1057_105715


namespace NUMINAMATH_GPT_absolute_sum_l1057_105726

def S (n : ℕ) : ℤ := n^2 - 4 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem absolute_sum : 
    (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|) = 68 :=
by
  sorry

end NUMINAMATH_GPT_absolute_sum_l1057_105726


namespace NUMINAMATH_GPT_total_chocolate_bar_count_l1057_105717

def large_box_count : ℕ := 150
def small_box_count_per_large_box : ℕ := 45
def chocolate_bar_count_per_small_box : ℕ := 35

theorem total_chocolate_bar_count :
  large_box_count * small_box_count_per_large_box * chocolate_bar_count_per_small_box = 236250 :=
by
  sorry

end NUMINAMATH_GPT_total_chocolate_bar_count_l1057_105717


namespace NUMINAMATH_GPT_problem1_solutionset_problem2_minvalue_l1057_105735

noncomputable def f (x : ℝ) : ℝ := 45 * abs (2 * x - 1)
noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem problem1_solutionset :
  {x : ℝ | 0 < x ∧ x < 2 / 3} = {x : ℝ | f x + abs (x + 1) < 2} :=
by
  sorry

theorem problem2_minvalue (a : ℝ) (m n : ℝ) (h : m + n = a ∧ m > 0 ∧ n > 0) :
  a = 2 → (4 / m + 1 / n) ≥ 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_solutionset_problem2_minvalue_l1057_105735


namespace NUMINAMATH_GPT_repaved_before_today_l1057_105788

variable (total_repaved today_repaved : ℕ)

theorem repaved_before_today (h1 : total_repaved = 4938) (h2 : today_repaved = 805) :
  total_repaved - today_repaved = 4133 :=
by 
  -- variables are integers and we are performing a subtraction
  sorry

end NUMINAMATH_GPT_repaved_before_today_l1057_105788


namespace NUMINAMATH_GPT_frosting_cupcakes_l1057_105729

theorem frosting_cupcakes :
  let r1 := 1 / 15
  let r2 := 1 / 25
  let r3 := 1 / 40
  let t := 600
  t * (r1 + r2 + r3) = 79 :=
by
  sorry

end NUMINAMATH_GPT_frosting_cupcakes_l1057_105729


namespace NUMINAMATH_GPT_number_of_men_in_first_group_l1057_105714

/-
Given the initial conditions:
1. Some men can color a 48 m long cloth in 2 days.
2. 6 men can color a 36 m long cloth in 1 day.

We need to prove that the number of men in the first group is equal to 9.
-/

theorem number_of_men_in_first_group (M : ℕ)
    (h1 : ∃ (x : ℕ), x * 48 = M * 2)
    (h2 : 6 * 36 = 36 * 1) :
    M = 9 :=
by
sorry

end NUMINAMATH_GPT_number_of_men_in_first_group_l1057_105714


namespace NUMINAMATH_GPT_find_function_l1057_105764

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 3 * x) : 
  ∀ x : ℝ, x ≠ 0 → f x = -x + 2 / x := 
by
  sorry

end NUMINAMATH_GPT_find_function_l1057_105764


namespace NUMINAMATH_GPT_smallest_n_l1057_105708

-- Define the costs of candies
def cost_purple := 24
def cost_yellow := 30

-- Define the number of candies Lara can buy
def pieces_red := 10
def pieces_green := 16
def pieces_blue := 18
def pieces_yellow := 22

-- Define the total money Lara has equivalently expressed by buying candies
def lara_total_money (n : ℕ) := n * cost_purple

-- Prove the smallest value of n that satisfies the conditions stated
theorem smallest_n : ∀ n : ℕ, 
  (lara_total_money n = 10 * pieces_red * cost_purple) ∧
  (lara_total_money n = 16 * pieces_green * cost_purple) ∧
  (lara_total_money n = 18 * pieces_blue * cost_purple) ∧
  (lara_total_money n = pieces_yellow * cost_yellow) → 
  n = 30 :=
by
  intro
  sorry

end NUMINAMATH_GPT_smallest_n_l1057_105708


namespace NUMINAMATH_GPT_tile_ratio_l1057_105782

/-- Given the initial configuration and extension method, the ratio of black tiles to white tiles in the new design is 22/27. -/
theorem tile_ratio (initial_black : ℕ) (initial_white : ℕ) (border_black : ℕ) (border_white : ℕ) (total_tiles : ℕ)
  (h1 : initial_black = 10)
  (h2 : initial_white = 15)
  (h3 : border_black = 12)
  (h4 : border_white = 12)
  (h5 : total_tiles = 49) :
  (initial_black + border_black) / (initial_white + border_white) = 22 / 27 := 
by {
  /- 
     Here we would provide the proof steps if needed.
     This is a theorem stating that the ratio of black to white tiles 
     in the new design is 22 / 27 given the initial conditions.
  -/
  sorry 
}

end NUMINAMATH_GPT_tile_ratio_l1057_105782


namespace NUMINAMATH_GPT_probability_at_least_75_cents_l1057_105795

theorem probability_at_least_75_cents (p n d q c50 : Prop) 
  (Hp : p = tt ∨ p = ff)
  (Hn : n = tt ∨ n = ff)
  (Hd : d = tt ∨ d = ff)
  (Hq : q = tt ∨ q = ff)
  (Hc50 : c50 = tt ∨ c50 = ff) :
  (1 / 2 : ℝ) = 
  ((if c50 = tt then (if q = tt then 1 else 0) else 0) + 
  (if c50 = tt then 2^3 else 0)) / 2^5 :=
by sorry

end NUMINAMATH_GPT_probability_at_least_75_cents_l1057_105795


namespace NUMINAMATH_GPT_absolute_difference_probability_l1057_105737

-- Define the conditions
def num_red_marbles : ℕ := 1500
def num_black_marbles : ℕ := 2000
def total_marbles : ℕ := num_red_marbles + num_black_marbles

def P_s : ℚ :=
  let ways_to_choose_2_red := (num_red_marbles * (num_red_marbles - 1)) / 2
  let ways_to_choose_2_black := (num_black_marbles * (num_black_marbles - 1)) / 2
  let total_favorable_outcomes := ways_to_choose_2_red + ways_to_choose_2_black
  total_favorable_outcomes / (total_marbles * (total_marbles - 1) / 2)

def P_d : ℚ :=
  (num_red_marbles * num_black_marbles) / (total_marbles * (total_marbles - 1) / 2)

-- Prove the statement
theorem absolute_difference_probability : |P_s - P_d| = 1 / 50 := by
  sorry

end NUMINAMATH_GPT_absolute_difference_probability_l1057_105737


namespace NUMINAMATH_GPT_find_other_number_l1057_105754

-- Define the conditions
variable (B : ℕ)
variable (HCF : ℕ → ℕ → ℕ)
variable (LCM : ℕ → ℕ → ℕ)

axiom hcf_cond : HCF 24 B = 15
axiom lcm_cond : LCM 24 B = 312

-- The theorem statement
theorem find_other_number (B : ℕ) (HCF : ℕ → ℕ → ℕ) (LCM : ℕ → ℕ → ℕ) 
  (hcf_cond : HCF 24 B = 15) (lcm_cond : LCM 24 B = 312) : 
  B = 195 :=
sorry

end NUMINAMATH_GPT_find_other_number_l1057_105754


namespace NUMINAMATH_GPT_soda_ratio_l1057_105742

theorem soda_ratio (total_sodas diet_sodas regular_sodas : ℕ) (h1 : total_sodas = 64) (h2 : diet_sodas = 28) (h3 : regular_sodas = total_sodas - diet_sodas) : regular_sodas / Nat.gcd regular_sodas diet_sodas = 9 ∧ diet_sodas / Nat.gcd regular_sodas diet_sodas = 7 :=
by
  sorry

end NUMINAMATH_GPT_soda_ratio_l1057_105742


namespace NUMINAMATH_GPT_horner_method_V3_correct_when_x_equals_2_l1057_105713

-- Polynomial f(x)
noncomputable def f (x : ℝ) : ℝ :=
  2 * x^5 - 3 * x^3 + 2 * x^2 + x - 3

-- Horner's method for evaluating f(x)
noncomputable def V3 (x : ℝ) : ℝ :=
  (((((2 * x + 0) * x - 3) * x + 2) * x + 1) * x - 3)

-- Proof that V3 = 12 when x = 2
theorem horner_method_V3_correct_when_x_equals_2 : V3 2 = 12 := by
  sorry

end NUMINAMATH_GPT_horner_method_V3_correct_when_x_equals_2_l1057_105713
