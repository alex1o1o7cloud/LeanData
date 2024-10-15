import Mathlib

namespace NUMINAMATH_GPT_range_m_l1604_160418

theorem range_m (m : ℝ) : 
  (∀ x : ℝ, ((m * x - 1) * (x - 2) > 0) ↔ (1/m < x ∧ x < 2)) → m < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_m_l1604_160418


namespace NUMINAMATH_GPT_probability_select_cooking_l1604_160435

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_select_cooking_l1604_160435


namespace NUMINAMATH_GPT_tables_capacity_l1604_160430

theorem tables_capacity (invited attended : ℕ) (didn't_show_up : ℕ) (tables : ℕ) (capacity : ℕ) 
    (h1 : invited = 24) (h2 : didn't_show_up = 10) (h3 : attended = invited - didn't_show_up) 
    (h4 : attended = 14) (h5 : tables = 2) : capacity = attended / tables :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_tables_capacity_l1604_160430


namespace NUMINAMATH_GPT_triangle_sin_double_angle_l1604_160425

open Real

theorem triangle_sin_double_angle (A : ℝ) (h : cos (π / 4 + A) = 5 / 13) : sin (2 * A) = 119 / 169 :=
by
  sorry

end NUMINAMATH_GPT_triangle_sin_double_angle_l1604_160425


namespace NUMINAMATH_GPT_max_total_length_of_cuts_l1604_160477

theorem max_total_length_of_cuts (A : ℕ) (n : ℕ) (m : ℕ) (P : ℕ) (Q : ℕ)
  (h1 : A = 30 * 30)
  (h2 : n = 225)
  (h3 : m = A / n)
  (h4 : m = 4)
  (h5 : Q = 4 * 30)
  (h6 : P = 225 * 10 - Q)
  (h7 : P / 2 = 1065) :
  P / 2 = 1065 :=
by 
  exact h7

end NUMINAMATH_GPT_max_total_length_of_cuts_l1604_160477


namespace NUMINAMATH_GPT_min_value_function_l1604_160483

open Real

theorem min_value_function (x y : ℝ) 
  (hx : x > -2 ∧ x < 2) 
  (hy : y > -2 ∧ y < 2) 
  (hxy : x * y = -1) : 
  (∃ u : ℝ, u = (4 / (4 - x^2) + 9 / (9 - y^2)) ∧ u = 12 / 5) :=
sorry

end NUMINAMATH_GPT_min_value_function_l1604_160483


namespace NUMINAMATH_GPT_range_of_a_l1604_160467

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ x^2 + 2 * a * x + 2 * a + 3 < 0) ↔ a < -1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1604_160467


namespace NUMINAMATH_GPT_tan_sum_identity_l1604_160459

open Real

theorem tan_sum_identity : 
  tan (80 * π / 180) + tan (40 * π / 180) - sqrt 3 * tan (80 * π / 180) * tan (40 * π / 180) = -sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_identity_l1604_160459


namespace NUMINAMATH_GPT_find_dividend_l1604_160493

-- Define the given constants
def quotient : ℕ := 909899
def divisor : ℕ := 12

-- Define the dividend as the product of divisor and quotient
def dividend : ℕ := divisor * quotient

-- The theorem stating the equality we need to prove
theorem find_dividend : dividend = 10918788 := by
  sorry

end NUMINAMATH_GPT_find_dividend_l1604_160493


namespace NUMINAMATH_GPT_circumcircle_radius_proof_l1604_160492

noncomputable def circumcircle_radius (AB A S : ℝ) : ℝ :=
  if AB = 3 ∧ A = 120 ∧ S = 9 * Real.sqrt 3 / 4 then 3 else 0

theorem circumcircle_radius_proof :
  circumcircle_radius 3 120 (9 * Real.sqrt 3 / 4) = 3 := by
  sorry

end NUMINAMATH_GPT_circumcircle_radius_proof_l1604_160492


namespace NUMINAMATH_GPT_cakes_left_correct_l1604_160428

def number_of_cakes_left (total_cakes sold_cakes : ℕ) : ℕ :=
  total_cakes - sold_cakes

theorem cakes_left_correct :
  number_of_cakes_left 54 41 = 13 :=
by
  sorry

end NUMINAMATH_GPT_cakes_left_correct_l1604_160428


namespace NUMINAMATH_GPT_rectangle_diagonal_length_l1604_160438

theorem rectangle_diagonal_length (P L W k d : ℝ) 
  (h1 : P = 72) 
  (h2 : L / W = 3 / 2) 
  (h3 : L = 3 * k) 
  (h4 : W = 2 * k) 
  (h5 : P = 2 * (L + W))
  (h6 : d = Real.sqrt ((L^2) + (W^2))) :
  d = 25.96 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_length_l1604_160438


namespace NUMINAMATH_GPT_simplify_polynomial_l1604_160464

theorem simplify_polynomial :
  (3 * x^3 + 4 * x^2 + 8 * x - 5) - (2 * x^3 + x^2 + 6 * x - 7) = x^3 + 3 * x^2 + 2 * x + 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1604_160464


namespace NUMINAMATH_GPT_player1_coins_l1604_160496

theorem player1_coins (coin_distribution : Fin 9 → ℕ) :
  let rotations := 11
  let player_4_coins := 90
  let player_8_coins := 35
  ∀ player : Fin 9, player = 0 → 
    let player_1_coins := coin_distribution player
    (coin_distribution 3 = player_4_coins) →
    (coin_distribution 7 = player_8_coins) →
    player_1_coins = 57 := 
sorry

end NUMINAMATH_GPT_player1_coins_l1604_160496


namespace NUMINAMATH_GPT_euclidean_division_l1604_160437

theorem euclidean_division (a b : ℕ) (hb : b ≠ 0) : ∃ q r : ℤ, 0 ≤ r ∧ r < b ∧ a = b * q + r :=
by sorry

end NUMINAMATH_GPT_euclidean_division_l1604_160437


namespace NUMINAMATH_GPT_student_game_incorrect_statement_l1604_160414

theorem student_game_incorrect_statement (a : ℚ) : ¬ (∀ a : ℚ, -a - 2 < 0) :=
by
  -- skip the proof for now
  sorry

end NUMINAMATH_GPT_student_game_incorrect_statement_l1604_160414


namespace NUMINAMATH_GPT_arg_cubed_eq_pi_l1604_160498

open Complex

theorem arg_cubed_eq_pi (z1 z2 : ℂ) (h1 : abs z1 = 3) (h2 : abs z2 = 5) (h3 : abs (z1 + z2) = 7) : 
  arg (z2 / z1) ^ 3 = π :=
by
  sorry

end NUMINAMATH_GPT_arg_cubed_eq_pi_l1604_160498


namespace NUMINAMATH_GPT_suki_bag_weight_is_22_l1604_160489

noncomputable def weight_of_suki_bag : ℝ :=
  let bags_suki := 6.5
  let bags_jimmy := 4.5
  let weight_jimmy_per_bag := 18.0
  let total_containers := 28
  let weight_per_container := 8.0
  let total_weight_jimmy := bags_jimmy * weight_jimmy_per_bag
  let total_weight_combined := total_containers * weight_per_container
  let total_weight_suki := total_weight_combined - total_weight_jimmy
  total_weight_suki / bags_suki

theorem suki_bag_weight_is_22 : weight_of_suki_bag = 22 :=
by
  sorry

end NUMINAMATH_GPT_suki_bag_weight_is_22_l1604_160489


namespace NUMINAMATH_GPT_parallel_lines_slope_l1604_160478

theorem parallel_lines_slope (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = (3 * k) * x + 7) → k = 5 / 3 := 
sorry

end NUMINAMATH_GPT_parallel_lines_slope_l1604_160478


namespace NUMINAMATH_GPT_bread_per_day_baguettes_per_day_croissants_per_day_l1604_160461

-- Define the conditions
def loaves_per_hour : ℕ := 10
def hours_per_day : ℕ := 6
def baguettes_per_2hours : ℕ := 30
def croissants_per_75minutes : ℕ := 20

-- Conversion factors
def minutes_per_hour : ℕ := 60
def minutes_per_block : ℕ := 75
def blocks_per_75minutes : ℕ := 360 / 75

-- Proof statements
theorem bread_per_day :
  loaves_per_hour * hours_per_day = 60 := by sorry

theorem baguettes_per_day :
  (hours_per_day / 2) * baguettes_per_2hours = 90 := by sorry

theorem croissants_per_day :
  (blocks_per_75minutes * croissants_per_75minutes) = 80 := by sorry

end NUMINAMATH_GPT_bread_per_day_baguettes_per_day_croissants_per_day_l1604_160461


namespace NUMINAMATH_GPT_distance_to_post_office_l1604_160472

theorem distance_to_post_office
  (D : ℝ)
  (travel_rate : ℝ) (walk_rate : ℝ)
  (total_time_hours : ℝ)
  (h1 : travel_rate = 25)
  (h2 : walk_rate = 4)
  (h3 : total_time_hours = 5 + 48 / 60) :
  D = 20 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_post_office_l1604_160472


namespace NUMINAMATH_GPT_trigonometric_identity_l1604_160411

theorem trigonometric_identity (φ : ℝ) 
  (h : Real.cos (π / 2 + φ) = (Real.sqrt 3) / 2) : 
  Real.cos (3 * π / 2 - φ) + Real.sin (φ - π) = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1604_160411


namespace NUMINAMATH_GPT_simplify_expression_l1604_160452

theorem simplify_expression (x : ℝ) (hx2 : x ≠ 2) (hx_2 : x ≠ -2) (hx1 : x ≠ 1) : 
  (1 + 1 / (x - 2)) / ((x^2 - 2 * x + 1) / (x^2 - 4)) = (x + 2) / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1604_160452


namespace NUMINAMATH_GPT_minimum_zeros_l1604_160479

theorem minimum_zeros (n : ℕ) (a : Fin n → ℤ) (h : n = 2011)
  (H : ∀ i j k : Fin n, a i + a j + a k ∈ Set.range a) : 
  ∃ (num_zeros : ℕ), num_zeros ≥ 2009 ∧ (∃ f : Fin (num_zeros) → Fin n, ∀ i : Fin (num_zeros), a (f i) = 0) :=
sorry

end NUMINAMATH_GPT_minimum_zeros_l1604_160479


namespace NUMINAMATH_GPT_definitely_incorrect_conclusions_l1604_160405

theorem definitely_incorrect_conclusions (a b c : ℝ) (x1 x2 : ℝ) 
  (h1 : a * x1^2 + b * x1 + c = 0) 
  (h2 : a * x2^2 + b * x2 + c = 0)
  (h3 : x1 > 0) 
  (h4 : x2 > 0) 
  (h5 : x1 + x2 = -b / a) 
  (h6 : x1 * x2 = c / a) : 
  (a > 0 ∧ b > 0 ∧ c > 0) = false ∧ 
  (a < 0 ∧ b < 0 ∧ c < 0) = false ∧ 
  (a > 0 ∧ b < 0 ∧ c < 0) = true ∧ 
  (a < 0 ∧ b > 0 ∧ c > 0) = true :=
sorry

end NUMINAMATH_GPT_definitely_incorrect_conclusions_l1604_160405


namespace NUMINAMATH_GPT_polygon_E_has_largest_area_l1604_160484

-- Define the areas of square and right triangle
def area_square (side : ℕ): ℕ := side * side
def area_right_triangle (leg : ℕ): ℕ := (leg * leg) / 2

-- Define the areas of each polygon
def area_polygon_A : ℕ := 2 * (area_square 2) + (area_right_triangle 2)
def area_polygon_B : ℕ := 3 * (area_square 2)
def area_polygon_C : ℕ := (area_square 2) + 4 * (area_right_triangle 2)
def area_polygon_D : ℕ := 3 * (area_right_triangle 2)
def area_polygon_E : ℕ := 4 * (area_square 2)

-- The theorem assertion
theorem polygon_E_has_largest_area : 
  area_polygon_E = 16 ∧ 
  16 > area_polygon_A ∧
  16 > area_polygon_B ∧
  16 > area_polygon_C ∧
  16 > area_polygon_D := 
sorry

end NUMINAMATH_GPT_polygon_E_has_largest_area_l1604_160484


namespace NUMINAMATH_GPT_perpendicular_lines_values_of_a_l1604_160482

theorem perpendicular_lines_values_of_a (a : ℝ) :
  (∃ (a : ℝ), (∀ x y : ℝ, a * x - y + 2 * a = 0 ∧ (2 * a - 1) * x + a * y = 0) 
    ↔ (a = 0 ∨ a = 1))
  := sorry

end NUMINAMATH_GPT_perpendicular_lines_values_of_a_l1604_160482


namespace NUMINAMATH_GPT_sqrt5_lt_sqrt2_plus_1_l1604_160465

theorem sqrt5_lt_sqrt2_plus_1 : Real.sqrt 5 < Real.sqrt 2 + 1 :=
sorry

end NUMINAMATH_GPT_sqrt5_lt_sqrt2_plus_1_l1604_160465


namespace NUMINAMATH_GPT_polygon_area_l1604_160441

theorem polygon_area (sides : ℕ) (perpendicular_adjacent : Bool) (congruent_sides : Bool) (perimeter : ℝ) (area : ℝ) :
  sides = 32 → 
  perpendicular_adjacent = true → 
  congruent_sides = true →
  perimeter = 64 →
  area = 64 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_polygon_area_l1604_160441


namespace NUMINAMATH_GPT_oil_leakage_problem_l1604_160408

theorem oil_leakage_problem :
    let l_A := 25  -- Leakage rate of Pipe A (gallons/hour)
    let l_B := 37  -- Leakage rate of Pipe B (gallons/hour)
    let l_C := 55  -- Leakage rate of Pipe C (gallons/hour)
    let l_D := 41  -- Leakage rate of Pipe D (gallons/hour)
    let l_E := 30  -- Leakage rate of Pipe E (gallons/hour)

    let t_A := 10  -- Time taken to fix Pipe A (hours)
    let t_B := 7   -- Time taken to fix Pipe B (hours)
    let t_C := 12  -- Time taken to fix Pipe C (hours)
    let t_D := 9   -- Time taken to fix Pipe D (hours)
    let t_E := 14  -- Time taken to fix Pipe E (hours)

    let leak_A := l_A * t_A  -- Total leaked from Pipe A (gallons)
    let leak_B := l_B * t_B  -- Total leaked from Pipe B (gallons)
    let leak_C := l_C * t_C  -- Total leaked from Pipe C (gallons)
    let leak_D := l_D * t_D  -- Total leaked from Pipe D (gallons)
    let leak_E := l_E * t_E  -- Total leaked from Pipe E (gallons)
  
    let overall_total := leak_A + leak_B + leak_C + leak_D + leak_E
  
    leak_A = 250 ∧
    leak_B = 259 ∧
    leak_C = 660 ∧
    leak_D = 369 ∧
    leak_E = 420 ∧
    overall_total = 1958 :=
by
    sorry

end NUMINAMATH_GPT_oil_leakage_problem_l1604_160408


namespace NUMINAMATH_GPT_unique_line_equal_intercepts_l1604_160474

-- Definitions of the point and line
structure Point where
  x : ℝ
  y : ℝ

def passesThrough (L : ℝ → ℝ) (P : Point) : Prop :=
  L P.x = P.y

noncomputable def hasEqualIntercepts (L : ℝ → ℝ) : Prop :=
  ∃ a, L 0 = a ∧ L a = 0

-- The main theorem statement
theorem unique_line_equal_intercepts (L : ℝ → ℝ) (P : Point) (hP : P.x = 2 ∧ P.y = 1) (h_equal_intercepts : hasEqualIntercepts L) :
  ∃! (L : ℝ → ℝ), passesThrough L P ∧ hasEqualIntercepts L :=
sorry

end NUMINAMATH_GPT_unique_line_equal_intercepts_l1604_160474


namespace NUMINAMATH_GPT_solve_system_of_equations_l1604_160434

theorem solve_system_of_equations (x y : ℝ) :
  (x^4 + (7/2) * x^2 * y + 2 * y^3 = 0) ∧
  (4 * x^2 + 7 * x * y + 2 * y^3 = 0) →
  (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -1) ∨ (x = -11 / 2 ∧ y = -11 / 2) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1604_160434


namespace NUMINAMATH_GPT_height_of_triangle_l1604_160487

theorem height_of_triangle
    (A : ℝ) (b : ℝ) (h : ℝ)
    (h1 : A = 30)
    (h2 : b = 12)
    (h3 : A = (b * h) / 2) :
    h = 5 :=
by
  sorry

end NUMINAMATH_GPT_height_of_triangle_l1604_160487


namespace NUMINAMATH_GPT_change_sum_equals_108_l1604_160431

theorem change_sum_equals_108 :
  ∃ (amounts : List ℕ), (∀ a ∈ amounts, a < 100 ∧ ((a % 25 = 4) ∨ (a % 5 = 4))) ∧
    amounts.sum = 108 := 
by
  sorry

end NUMINAMATH_GPT_change_sum_equals_108_l1604_160431


namespace NUMINAMATH_GPT_sculpture_and_base_height_l1604_160460

def height_sculpture_ft : ℕ := 2
def height_sculpture_in : ℕ := 10
def height_base_in : ℕ := 2

def total_height_in (ft : ℕ) (inch1 inch2 : ℕ) : ℕ :=
  (ft * 12) + inch1 + inch2

def total_height_ft (total_in : ℕ) : ℕ :=
  total_in / 12

theorem sculpture_and_base_height :
  total_height_ft (total_height_in height_sculpture_ft height_sculpture_in height_base_in) = 3 :=
by
  sorry

end NUMINAMATH_GPT_sculpture_and_base_height_l1604_160460


namespace NUMINAMATH_GPT_number_of_cars_l1604_160423

theorem number_of_cars (b c : ℕ) (h1 : b = c / 10) (h2 : c - b = 90) : c = 100 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cars_l1604_160423


namespace NUMINAMATH_GPT_polynomial_roots_arithmetic_progression_complex_root_l1604_160486

theorem polynomial_roots_arithmetic_progression_complex_root :
  ∃ a : ℝ, (∀ (r d : ℂ), (r - d) + r + (r + d) = 9 → (r - d) * r + (r - d) * (r + d) + r * (r + d) = 30 → d^2 = -3 → 
  (r - d) * r * (r + d) = -a) → a = -12 :=
by sorry

end NUMINAMATH_GPT_polynomial_roots_arithmetic_progression_complex_root_l1604_160486


namespace NUMINAMATH_GPT_inverse_proportion_expression_and_calculation_l1604_160442

theorem inverse_proportion_expression_and_calculation :
  (∃ k : ℝ, (∀ (x y : ℝ), y = k / x) ∧
   (∀ x y : ℝ, y = 400 ∧ x = 0.25 → k = 100) ∧
   (∀ x : ℝ, 200 = 100 / x → x = 0.5)) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_expression_and_calculation_l1604_160442


namespace NUMINAMATH_GPT_intercepts_of_line_l1604_160402

theorem intercepts_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) :
  (∃ x_intercept : ℝ, x_intercept = 7 ∧ (4 * x_intercept + 7 * 0 = 28)) ∧
  (∃ y_intercept : ℝ, y_intercept = 4 ∧ (4 * 0 + 7 * y_intercept = 28)) :=
by
  sorry

end NUMINAMATH_GPT_intercepts_of_line_l1604_160402


namespace NUMINAMATH_GPT_hyperbola_correct_l1604_160443

noncomputable def hyperbola_properties : Prop :=
  let h := 2
  let k := 0
  let a := 4
  let c := 8
  let b := Real.sqrt ((c^2) - (a^2))
  (h + k + a + b = 4 * Real.sqrt 3 + 6)

theorem hyperbola_correct : hyperbola_properties :=
by
  unfold hyperbola_properties
  let h := 2
  let k := 0
  let a := 4
  let c := 8
  have b : ℝ := Real.sqrt ((c^2) - (a^2))
  sorry

end NUMINAMATH_GPT_hyperbola_correct_l1604_160443


namespace NUMINAMATH_GPT_larger_number_of_two_l1604_160440

theorem larger_number_of_two (x y : ℝ) (h1 : x - y = 3) (h2 : x + y = 29) (h3 : x * y > 200) : x = 16 :=
by sorry

end NUMINAMATH_GPT_larger_number_of_two_l1604_160440


namespace NUMINAMATH_GPT_janet_counts_total_birds_l1604_160404

theorem janet_counts_total_birds :
  let crows := 30
  let hawks := crows + (60 / 100) * crows
  hawks + crows = 78 :=
by
  sorry

end NUMINAMATH_GPT_janet_counts_total_birds_l1604_160404


namespace NUMINAMATH_GPT_emmanuel_jelly_beans_l1604_160417

theorem emmanuel_jelly_beans (total_jelly_beans : ℕ)
      (thomas_percentage : ℕ)
      (barry_ratio : ℕ)
      (emmanuel_ratio : ℕ)
      (h1 : total_jelly_beans = 200)
      (h2 : thomas_percentage = 10)
      (h3 : barry_ratio = 4)
      (h4 : emmanuel_ratio = 5) :
  let thomas_jelly_beans := (thomas_percentage * total_jelly_beans) / 100
  let remaining_jelly_beans := total_jelly_beans - thomas_jelly_beans
  let total_ratio := barry_ratio + emmanuel_ratio
  let per_part_jelly_beans := remaining_jelly_beans / total_ratio
  let emmanuel_jelly_beans := emmanuel_ratio * per_part_jelly_beans
  emmanuel_jelly_beans = 100 :=
by
  sorry

end NUMINAMATH_GPT_emmanuel_jelly_beans_l1604_160417


namespace NUMINAMATH_GPT_find_y_l1604_160455

theorem find_y (y : ℕ) : (1 / 8) * 2^36 = 8^y → y = 11 := by
  sorry

end NUMINAMATH_GPT_find_y_l1604_160455


namespace NUMINAMATH_GPT_symmetric_curve_eq_l1604_160463

theorem symmetric_curve_eq : 
  (∃ x' y', (x' - 3)^2 + 4*(y' - 5)^2 = 4 ∧ (x' - 6 = x' + x) ∧ (y' - 10 = y' + y)) ->
  (∃ x y, (x - 6) ^ 2 + 4 * (y - 10) ^ 2 = 4) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_curve_eq_l1604_160463


namespace NUMINAMATH_GPT_renovation_project_cement_loads_l1604_160422

theorem renovation_project_cement_loads
  (s : ℚ) (d : ℚ) (t : ℚ)
  (hs : s = 0.16666666666666666) 
  (hd : d = 0.3333333333333333)
  (ht : t = 0.6666666666666666) :
  t - (s + d) = 0.1666666666666666 := by
  sorry

end NUMINAMATH_GPT_renovation_project_cement_loads_l1604_160422


namespace NUMINAMATH_GPT_wholesale_price_is_90_l1604_160436

theorem wholesale_price_is_90 
  (R S W: ℝ)
  (h1 : R = 120)
  (h2 : S = R - 0.1 * R)
  (h3 : S = W + 0.2 * W)
  : W = 90 := 
by
  sorry

end NUMINAMATH_GPT_wholesale_price_is_90_l1604_160436


namespace NUMINAMATH_GPT_min_next_score_to_increase_avg_l1604_160439

def Liam_initial_scores : List ℕ := [72, 85, 78, 66, 90, 82]

def current_average (scores: List ℕ) : ℚ :=
  (scores.sum / scores.length : ℚ)

def next_score_requirement (initial_scores: List ℕ) (desired_increase: ℚ) : ℚ :=
  let current_avg := current_average initial_scores
  let desired_avg := current_avg + desired_increase
  let total_tests := initial_scores.length + 1
  let total_required := desired_avg * total_tests
  total_required - initial_scores.sum

theorem min_next_score_to_increase_avg :
  next_score_requirement Liam_initial_scores 5 = 115 := by
  sorry

end NUMINAMATH_GPT_min_next_score_to_increase_avg_l1604_160439


namespace NUMINAMATH_GPT_inequality_holds_l1604_160499

theorem inequality_holds (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  (a^3 / (a^3 + 15 * b * c * d))^(1/2) ≥ a^(15/8) / (a^(15/8) + b^(15/8) + c^(15/8) + d^(15/8)) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l1604_160499


namespace NUMINAMATH_GPT_velocity_of_current_l1604_160497

theorem velocity_of_current (v : ℝ) 
  (row_speed : ℝ) (distance : ℝ) (total_time : ℝ) 
  (h_row_speed : row_speed = 5)
  (h_distance : distance = 2.4)
  (h_total_time : total_time = 1)
  (h_equation : distance / (row_speed + v) + distance / (row_speed - v) = total_time) :
  v = 1 :=
sorry

end NUMINAMATH_GPT_velocity_of_current_l1604_160497


namespace NUMINAMATH_GPT_polynomial_perfect_square_trinomial_l1604_160490

theorem polynomial_perfect_square_trinomial (k : ℝ) :
  (∀ x : ℝ, 4 * x^2 + 2 * k * x + 25 = (2 * x + 5) * (2 * x + 5)) → (k = 10 ∨ k = -10) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_perfect_square_trinomial_l1604_160490


namespace NUMINAMATH_GPT_decimal_equivalent_of_quarter_cubed_l1604_160473

theorem decimal_equivalent_of_quarter_cubed :
    (1 / 4 : ℝ) ^ 3 = 0.015625 := 
by
    sorry

end NUMINAMATH_GPT_decimal_equivalent_of_quarter_cubed_l1604_160473


namespace NUMINAMATH_GPT_rhombus_area_l1604_160453

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 150 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_rhombus_area_l1604_160453


namespace NUMINAMATH_GPT_union_of_A_and_B_is_R_l1604_160458

open Set Real

def A := {x : ℝ | log x > 0}
def B := {x : ℝ | x ≤ 1}

theorem union_of_A_and_B_is_R : A ∪ B = univ := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_is_R_l1604_160458


namespace NUMINAMATH_GPT_profit_achieved_at_50_yuan_l1604_160466

theorem profit_achieved_at_50_yuan :
  ∀ (x : ℝ), (30 ≤ x ∧ x ≤ 54) → 
  ((x - 30) * (80 - 2 * (x - 40)) = 1200) →
  x = 50 :=
by
  intros x h_range h_profit
  sorry

end NUMINAMATH_GPT_profit_achieved_at_50_yuan_l1604_160466


namespace NUMINAMATH_GPT_digit_to_make_52B6_divisible_by_3_l1604_160451

theorem digit_to_make_52B6_divisible_by_3 (B : ℕ) (hB : 0 ≤ B ∧ B ≤ 9) : 
  (5 + 2 + B + 6) % 3 = 0 ↔ (B = 2 ∨ B = 5 ∨ B = 8) := 
by
  sorry

end NUMINAMATH_GPT_digit_to_make_52B6_divisible_by_3_l1604_160451


namespace NUMINAMATH_GPT_order_of_x_y_z_l1604_160456

variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- Conditions
axiom h1 : 0.9 < x
axiom h2 : x < 1.0
axiom h3 : y = x^x
axiom h4 : z = x^(x^x)

-- Theorem to be proved
theorem order_of_x_y_z (h1 : 0.9 < x) (h2 : x < 1.0) (h3 : y = x^x) (h4 : z = x^(x^x)) : x < z ∧ z < y :=
by
  sorry

end NUMINAMATH_GPT_order_of_x_y_z_l1604_160456


namespace NUMINAMATH_GPT_rachel_pool_fill_time_l1604_160421

theorem rachel_pool_fill_time :
  ∀ (pool_volume : ℕ) (num_hoses : ℕ) (hose_rate : ℕ),
  pool_volume = 30000 →
  num_hoses = 5 →
  hose_rate = 3 →
  (pool_volume / (num_hoses * hose_rate * 60) : ℤ) = 33 :=
by
  intros pool_volume num_hoses hose_rate h1 h2 h3
  sorry

end NUMINAMATH_GPT_rachel_pool_fill_time_l1604_160421


namespace NUMINAMATH_GPT_proof1_proof2_l1604_160485

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  |a * x - 2| - |x + 2|

-- Statement for proof 1
theorem proof1 (x : ℝ)
  (a : ℝ) (h : a = 2) (hx : f 2 x ≤ 1) : -1/3 ≤ x ∧ x ≤ 5 :=
sorry

-- Statement for proof 2
theorem proof2 (a : ℝ)
  (h : ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4) : a = 1 ∨ a = -1 :=
sorry

end NUMINAMATH_GPT_proof1_proof2_l1604_160485


namespace NUMINAMATH_GPT_simplify_expression_l1604_160481

variable {a : ℝ}

theorem simplify_expression (h1 : a ≠ 2) (h2 : a ≠ -2) :
  ((a^2 + 4*a + 4) / (a^2 - 4) - (a + 3) / (a - 2)) / ((a + 2) / (a - 2)) = -1 / (a + 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1604_160481


namespace NUMINAMATH_GPT_number_of_integers_between_sqrt10_and_sqrt100_l1604_160449

theorem number_of_integers_between_sqrt10_and_sqrt100 :
  (∃ n : ℕ, n = 7) :=
sorry

end NUMINAMATH_GPT_number_of_integers_between_sqrt10_and_sqrt100_l1604_160449


namespace NUMINAMATH_GPT_find_D_E_l1604_160495

/--
Consider the circle given by \( x^2 + y^2 + D \cdot x + E \cdot y + F = 0 \) that is symmetrical with
respect to the line \( l_1: x - y + 4 = 0 \) and the line \( l_2: x + 3y = 0 \). Prove that the values 
of \( D \) and \( E \) are \( 12 \) and \( -4 \), respectively.
-/
theorem find_D_E (D E F : ℝ) (h1 : -D/2 + E/2 + 4 = 0) (h2 : -D/2 - 3*E/2 = 0) : D = 12 ∧ E = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_D_E_l1604_160495


namespace NUMINAMATH_GPT_complete_collection_probability_l1604_160401

namespace Stickers

open scoped Classical

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.Ico 1 (n + 1))

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_complete_collection :
  ℚ := (combinations 6 6 * combinations 12 4) / combinations 18 10

theorem complete_collection_probability :
  probability_complete_collection = (5 / 442 : ℚ) := by
  sorry

end Stickers

end NUMINAMATH_GPT_complete_collection_probability_l1604_160401


namespace NUMINAMATH_GPT_distinct_value_expression_l1604_160447

def tri (a b : ℕ) : ℕ := min a b
def nabla (a b : ℕ) : ℕ := max a b

theorem distinct_value_expression (x : ℕ) : (nabla 5 (nabla 4 (tri x 4))) = 5 := 
by
  sorry

end NUMINAMATH_GPT_distinct_value_expression_l1604_160447


namespace NUMINAMATH_GPT_yonderland_license_plates_l1604_160403

/-!
# Valid License Plates in Yonderland

A valid license plate in Yonderland consists of three letters followed by four digits. 

We are tasked with determining the number of valid license plates possible under this format.
-/

def num_letters : ℕ := 26
def num_digits : ℕ := 10
def letter_combinations : ℕ := num_letters ^ 3
def digit_combinations : ℕ := num_digits ^ 4
def total_combinations : ℕ := letter_combinations * digit_combinations

theorem yonderland_license_plates : total_combinations = 175760000 := by
  sorry

end NUMINAMATH_GPT_yonderland_license_plates_l1604_160403


namespace NUMINAMATH_GPT_total_amount_shared_l1604_160427

-- Define the amounts for Ken and Tony based on the conditions
def ken_amt : ℤ := 1750
def tony_amt : ℤ := 2 * ken_amt

-- The proof statement that the total amount shared is $5250
theorem total_amount_shared : ken_amt + tony_amt = 5250 :=
by 
  sorry

end NUMINAMATH_GPT_total_amount_shared_l1604_160427


namespace NUMINAMATH_GPT_total_assembly_time_l1604_160406

-- Define the conditions
def chairs : ℕ := 2
def tables : ℕ := 2
def time_per_piece : ℕ := 8
def total_pieces : ℕ := chairs + tables

-- State the theorem
theorem total_assembly_time :
  total_pieces * time_per_piece = 32 :=
sorry

end NUMINAMATH_GPT_total_assembly_time_l1604_160406


namespace NUMINAMATH_GPT_combined_marble_remainder_l1604_160470

theorem combined_marble_remainder (l j : ℕ) (h_l : l % 8 = 5) (h_j : j % 8 = 6) : (l + j) % 8 = 3 := by
  sorry

end NUMINAMATH_GPT_combined_marble_remainder_l1604_160470


namespace NUMINAMATH_GPT_annual_income_is_correct_l1604_160424

noncomputable def total_investment : ℝ := 4455
noncomputable def price_per_share : ℝ := 8.25
noncomputable def dividend_rate : ℝ := 12 / 100
noncomputable def face_value : ℝ := 10

noncomputable def number_of_shares : ℝ := total_investment / price_per_share
noncomputable def dividend_per_share : ℝ := dividend_rate * face_value
noncomputable def annual_income : ℝ := dividend_per_share * number_of_shares

theorem annual_income_is_correct : annual_income = 648 := by
  sorry

end NUMINAMATH_GPT_annual_income_is_correct_l1604_160424


namespace NUMINAMATH_GPT_maximum_partial_sum_l1604_160426

theorem maximum_partial_sum (a : ℕ → ℕ) (d : ℕ) (S : ℕ → ℕ)
    (h_arith_seq : ∀ n, a n = a 0 + n * d)
    (h8_13 : 3 * a 8 = 5 * a 13)
    (h_pos : a 0 > 0)
    (h_sn_def : ∀ n, S n = n * (2 * a 0 + (n - 1) * d) / 2) :
  S 20 = max (max (S 10) (S 11)) (max (S 20) (S 21)) := 
sorry

end NUMINAMATH_GPT_maximum_partial_sum_l1604_160426


namespace NUMINAMATH_GPT_sanAntonioToAustin_passes_austinToSanAntonio_l1604_160462

noncomputable def buses_passed : ℕ :=
  let austinToSanAntonio (n : ℕ) : ℕ := n * 2
  let sanAntonioToAustin (n : ℕ) : ℕ := n * 2 + 1
  let tripDuration : ℕ := 3
  if (austinToSanAntonio 3 - 0) <= tripDuration then 2 else 0

-- Proof statement
theorem sanAntonioToAustin_passes_austinToSanAntonio :
  buses_passed = 2 :=
  sorry

end NUMINAMATH_GPT_sanAntonioToAustin_passes_austinToSanAntonio_l1604_160462


namespace NUMINAMATH_GPT_rope_length_comparison_l1604_160444

theorem rope_length_comparison
  (L : ℝ)
  (hL1 : L > 0) 
  (cut1 cut2 : ℝ)
  (hcut1 : cut1 = 0.3)
  (hcut2 : cut2 = 3) :
  L - cut1 > L - cut2 :=
by
  sorry

end NUMINAMATH_GPT_rope_length_comparison_l1604_160444


namespace NUMINAMATH_GPT_inequality_holds_l1604_160475

theorem inequality_holds (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 12 → x^2 + 25 + |x^3 - 5 * x^2| ≥ a * x) ↔ a ≤ 2.5 := 
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1604_160475


namespace NUMINAMATH_GPT_total_distance_walked_l1604_160446

-- Define the given conditions
def walks_to_work_days := 5
def walks_dog_days := 7
def walks_to_friend_days := 1
def walks_to_store_days := 2

def distance_to_work := 6
def distance_dog_walk := 2
def distance_to_friend := 1
def distance_to_store := 3

-- The proof statement
theorem total_distance_walked :
  (walks_to_work_days * (distance_to_work * 2)) +
  (walks_dog_days * (distance_dog_walk * 2)) +
  (walks_to_friend_days * distance_to_friend) +
  (walks_to_store_days * distance_to_store) = 95 := 
sorry

end NUMINAMATH_GPT_total_distance_walked_l1604_160446


namespace NUMINAMATH_GPT_total_population_l1604_160409

variable (b g t s : ℕ)

theorem total_population (hb : b = 4 * g) (hg : g = 8 * t) (ht : t = 2 * s) :
  b + g + t + s = (83 * g) / 16 :=
by sorry

end NUMINAMATH_GPT_total_population_l1604_160409


namespace NUMINAMATH_GPT_rectangle_perimeter_l1604_160433

-- Define the conditions
variables (z w : ℕ)
-- Define the side lengths of the rectangles
def rectangle_long_side := z - w
def rectangle_short_side := w

-- Theorem: The perimeter of one of the four rectangles
theorem rectangle_perimeter : 2 * (rectangle_long_side z w) + 2 * (rectangle_short_side w) = 2 * z :=
by sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1604_160433


namespace NUMINAMATH_GPT_smallest_n_sum_gt_10_pow_5_l1604_160457

theorem smallest_n_sum_gt_10_pow_5 :
  ∃ (n : ℕ), (n ≥ 142) ∧ (5 * n^2 + 4 * n ≥ 100000) :=
by
  use 142
  sorry

end NUMINAMATH_GPT_smallest_n_sum_gt_10_pow_5_l1604_160457


namespace NUMINAMATH_GPT_cindy_correct_answer_l1604_160494

theorem cindy_correct_answer (x : ℤ) (h : (x - 7) / 5 = 37) : (x - 5) / 7 = 26 :=
sorry

end NUMINAMATH_GPT_cindy_correct_answer_l1604_160494


namespace NUMINAMATH_GPT_lucas_siblings_product_is_35_l1604_160400

-- Definitions based on the given conditions
def total_girls (lauren_sisters : ℕ) : ℕ := lauren_sisters + 1
def total_boys (lauren_brothers : ℕ) : ℕ := lauren_brothers + 1

-- Given conditions
def lauren_sisters : ℕ := 4
def lauren_brothers : ℕ := 7

-- Compute number of sisters (S) and brothers (B) Lucas has
def lucas_sisters : ℕ := total_girls lauren_sisters
def lucas_brothers : ℕ := lauren_brothers

theorem lucas_siblings_product_is_35 : 
  (lucas_sisters * lucas_brothers = 35) := by
  -- Asserting the correctness based on given family structure conditions
  sorry

end NUMINAMATH_GPT_lucas_siblings_product_is_35_l1604_160400


namespace NUMINAMATH_GPT_repeating_decimal_fraction_eq_l1604_160454

-- Define repeating decimal and its equivalent fraction
def repeating_decimal_value : ℚ := 7 + 123 / 999

theorem repeating_decimal_fraction_eq :
  repeating_decimal_value = 2372 / 333 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_eq_l1604_160454


namespace NUMINAMATH_GPT_complex_multiplication_l1604_160450

-- Definition of the imaginary unit i
def i : ℂ := Complex.I

-- The theorem stating the equality
theorem complex_multiplication : (2 + i) * (3 + i) = 5 + 5 * i := 
sorry

end NUMINAMATH_GPT_complex_multiplication_l1604_160450


namespace NUMINAMATH_GPT_David_min_max_rides_l1604_160419

-- Definitions based on the conditions
variable (Alena_rides : ℕ := 11)
variable (Bara_rides : ℕ := 20)
variable (Cenek_rides : ℕ := 4)
variable (every_pair_rides_at_least_once : Prop := true)

-- Hypotheses for the problem
axiom Alena_has_ridden : Alena_rides = 11
axiom Bara_has_ridden : Bara_rides = 20
axiom Cenek_has_ridden : Cenek_rides = 4
axiom Pairs_have_ridden : every_pair_rides_at_least_once

-- Statement for the minimum and maximum rides of David
theorem David_min_max_rides (David_rides : ℕ) :
  (David_rides = 11) ∨ (David_rides = 29) :=
sorry

end NUMINAMATH_GPT_David_min_max_rides_l1604_160419


namespace NUMINAMATH_GPT_farmer_purchase_l1604_160407

theorem farmer_purchase : ∃ r c : ℕ, 30 * r + 45 * c = 1125 ∧ r > 0 ∧ c > 0 ∧ r = 3 ∧ c = 23 := 
by 
  sorry

end NUMINAMATH_GPT_farmer_purchase_l1604_160407


namespace NUMINAMATH_GPT_real_imag_equal_complex_l1604_160415

/-- Given i is the imaginary unit, and a is a real number,
if the real part and the imaginary part of the complex number -3i(a+i) are equal,
then a = -1. -/
theorem real_imag_equal_complex (a : ℝ) (i : ℂ) (h_i : i * i = -1) 
    (h_eq : (3 : ℂ) = -(3 : ℂ) * a * i) : a = -1 :=
sorry

end NUMINAMATH_GPT_real_imag_equal_complex_l1604_160415


namespace NUMINAMATH_GPT_max_viewers_per_week_l1604_160488

theorem max_viewers_per_week :
  ∃ (x y : ℕ), 80 * x + 40 * y ≤ 320 ∧ x + y ≥ 6 ∧ 600000 * x + 200000 * y = 2000000 :=
by
  sorry

end NUMINAMATH_GPT_max_viewers_per_week_l1604_160488


namespace NUMINAMATH_GPT_tv_show_years_l1604_160448

theorem tv_show_years (s1 s2 s3 : ℕ) (e1 e2 e3 : ℕ) (avg : ℕ) :
  s1 = 8 → e1 = 15 →
  s2 = 4 → e2 = 20 →
  s3 = 2 → e3 = 12 →
  avg = 16 →
  (s1 * e1 + s2 * e2 + s3 * e3) / avg = 14 := by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end NUMINAMATH_GPT_tv_show_years_l1604_160448


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1604_160420

def speed_of_stream : ℝ := 8
def downstream_distance : ℝ := 64
def upstream_distance : ℝ := 32

theorem boat_speed_in_still_water (x : ℝ) (t : ℝ) 
  (HS_downstream : t = downstream_distance / (x + speed_of_stream)) 
  (HS_upstream : t = upstream_distance / (x - speed_of_stream)) :
  x = 24 := by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1604_160420


namespace NUMINAMATH_GPT_negation_of_proposition_l1604_160416

-- Definitions using the conditions stated
def p (x : ℝ) : Prop := x^2 - x + 1/4 ≥ 0

-- The statement to prove
theorem negation_of_proposition :
  (¬ (∀ x : ℝ, p x)) = (∃ x : ℝ, ¬ p x) :=
by
  -- Proof will go here; replaced by sorry as per instruction
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1604_160416


namespace NUMINAMATH_GPT_cost_of_replaced_tomatoes_l1604_160410

def original_order : ℝ := 25
def delivery_tip : ℝ := 8
def new_total : ℝ := 35
def original_tomatoes : ℝ := 0.99
def original_lettuce : ℝ := 1.00
def new_lettuce : ℝ := 1.75
def original_celery : ℝ := 1.96
def new_celery : ℝ := 2.00

def increase_in_lettuce := new_lettuce - original_lettuce
def increase_in_celery := new_celery - original_celery
def total_increase_except_tomatoes := increase_in_lettuce + increase_in_celery
def original_total_with_delivery := original_order + delivery_tip
def total_increase := new_total - original_total_with_delivery
def increase_due_to_tomatoes := total_increase - total_increase_except_tomatoes
def replaced_tomatoes := original_tomatoes + increase_due_to_tomatoes

theorem cost_of_replaced_tomatoes : replaced_tomatoes = 2.20 := by
  sorry

end NUMINAMATH_GPT_cost_of_replaced_tomatoes_l1604_160410


namespace NUMINAMATH_GPT_petya_result_less_than_one_tenth_l1604_160413

theorem petya_result_less_than_one_tenth 
  (a b c d e f : ℕ) 
  (ha: a.gcd b = 1) (hb: c.gcd d = 1)
  (hc: e.gcd f = 1) 
  (vasya_correct: (a / b) + (c / d) + (e / f) = 1) :
  (a + c + e) / (b + d + f) < 1 / 10 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_petya_result_less_than_one_tenth_l1604_160413


namespace NUMINAMATH_GPT_cars_on_happy_street_l1604_160480

theorem cars_on_happy_street :
  let cars_tuesday := 25
  let cars_monday := cars_tuesday - cars_tuesday * 20 / 100
  let cars_wednesday := cars_monday + 2
  let cars_thursday : ℕ := 10
  let cars_friday : ℕ := 10
  let cars_saturday : ℕ := 5
  let cars_sunday : ℕ := 5
  let total_cars := cars_monday + cars_tuesday + cars_wednesday + cars_thursday + cars_friday + cars_saturday + cars_sunday
  total_cars = 97 :=
by
  sorry

end NUMINAMATH_GPT_cars_on_happy_street_l1604_160480


namespace NUMINAMATH_GPT_percent_increase_from_first_to_second_quarter_l1604_160432

theorem percent_increase_from_first_to_second_quarter 
  (P : ℝ) :
  ((1.60 * P - 1.20 * P) / (1.20 * P)) * 100 = 33.33 := by
  sorry

end NUMINAMATH_GPT_percent_increase_from_first_to_second_quarter_l1604_160432


namespace NUMINAMATH_GPT_jenny_chocolate_squares_l1604_160429

theorem jenny_chocolate_squares (mike_chocolates : ℕ) (jenny_chocolates : ℕ) 
  (h_mike : mike_chocolates = 20) 
  (h_jenny : jenny_chocolates = 3 * mike_chocolates + 5) :
  jenny_chocolates = 65 :=
by
  sorry

end NUMINAMATH_GPT_jenny_chocolate_squares_l1604_160429


namespace NUMINAMATH_GPT_intersection_P_Q_eq_Q_l1604_160476

-- Definitions of P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Statement to prove P ∩ Q = Q
theorem intersection_P_Q_eq_Q : P ∩ Q = Q := 
by 
  sorry

end NUMINAMATH_GPT_intersection_P_Q_eq_Q_l1604_160476


namespace NUMINAMATH_GPT_negation_of_prop_p_l1604_160491

theorem negation_of_prop_p (p : Prop) (h : ∀ x: ℝ, 0 < x → x > Real.log x) :
  (¬ (∀ x: ℝ, 0 < x → x > Real.log x)) ↔ (∃ x_0: ℝ, 0 < x_0 ∧ x_0 ≤ Real.log x_0) :=
by sorry

end NUMINAMATH_GPT_negation_of_prop_p_l1604_160491


namespace NUMINAMATH_GPT_mcq_options_l1604_160469

theorem mcq_options :
  ∃ n : ℕ, (1/n : ℝ) * (1/2) * (1/2) = (1/12) ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_mcq_options_l1604_160469


namespace NUMINAMATH_GPT_john_has_18_blue_pens_l1604_160445

variables (R B Bl : ℕ)

-- Conditions from the problem
def john_has_31_pens : Prop := R + B + Bl = 31
def black_pens_5_more_than_red : Prop := B = R + 5
def blue_pens_twice_black : Prop := Bl = 2 * B

theorem john_has_18_blue_pens :
  john_has_31_pens R B Bl ∧ black_pens_5_more_than_red R B ∧ blue_pens_twice_black B Bl →
  Bl = 18 :=
by
  sorry

end NUMINAMATH_GPT_john_has_18_blue_pens_l1604_160445


namespace NUMINAMATH_GPT_tan_subtraction_modified_l1604_160412

theorem tan_subtraction_modified (α β : ℝ) (h1 : Real.tan α = 9) (h2 : Real.tan β = 6) :
  Real.tan (α - β) = (3 : ℝ) / (157465 : ℝ) := by
  have h3 : Real.tan (α - β) = (Real.tan α - Real.tan β) / (1 + (Real.tan α * Real.tan β)^3) :=
    sorry -- this is assumed as given in the conditions
  sorry -- rest of the proof

end NUMINAMATH_GPT_tan_subtraction_modified_l1604_160412


namespace NUMINAMATH_GPT_B_squared_ge_AC_l1604_160471

variable {a b c A B C : ℝ}

theorem B_squared_ge_AC
  (h1 : b^2 < a * c)
  (h2 : a * C - 2 * b * B + c * A = 0) :
  B^2 ≥ A * C := 
sorry

end NUMINAMATH_GPT_B_squared_ge_AC_l1604_160471


namespace NUMINAMATH_GPT_revenue_growth_20_percent_l1604_160468

noncomputable def revenue_increase (R2000 R2003 R2005 : ℝ) : ℝ :=
  ((R2005 - R2003) / R2003) * 100

theorem revenue_growth_20_percent (R2000 : ℝ) (h1 : R2003 = 1.5 * R2000) (h2 : R2005 = 1.8 * R2000) :
  revenue_increase R2000 R2003 R2005 = 20 :=
by
  sorry

end NUMINAMATH_GPT_revenue_growth_20_percent_l1604_160468
