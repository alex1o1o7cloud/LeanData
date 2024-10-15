import Mathlib

namespace NUMINAMATH_GPT_inequality_solution_l839_83969

theorem inequality_solution (x : ℝ) : 
  (x < 2 ∨ x = 3) ↔ (x - 3) / ((x - 2) * (x - 3)) ≤ 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_solution_l839_83969


namespace NUMINAMATH_GPT_cube_difference_l839_83946

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) :
  a^3 - b^3 = 108 :=
sorry

end NUMINAMATH_GPT_cube_difference_l839_83946


namespace NUMINAMATH_GPT_matrix_det_is_neg16_l839_83932

def matrix := Matrix (Fin 2) (Fin 2) ℤ
def given_matrix : matrix := ![![ -7, 5], ![6, -2]]

theorem matrix_det_is_neg16 : Matrix.det given_matrix = -16 := 
by
  sorry

end NUMINAMATH_GPT_matrix_det_is_neg16_l839_83932


namespace NUMINAMATH_GPT_previous_salary_is_40_l839_83959

-- Define the conditions
def new_salary : ℕ := 80
def percentage_increase : ℕ := 100

-- Proven goal: John's previous salary before the raise
def previous_salary : ℕ := new_salary / 2

theorem previous_salary_is_40 : previous_salary = 40 := 
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_previous_salary_is_40_l839_83959


namespace NUMINAMATH_GPT_sqrt_expression_non_negative_l839_83914

theorem sqrt_expression_non_negative (x : ℝ) : 4 + 2 * x ≥ 0 ↔ x ≥ -2 :=
by sorry

end NUMINAMATH_GPT_sqrt_expression_non_negative_l839_83914


namespace NUMINAMATH_GPT_union_complement_equals_set_l839_83943

universe u

variable {I A B : Set ℕ}

def universal_set : Set ℕ := {0, 1, 2, 3, 4}
def set_A : Set ℕ := {1, 2}
def set_B : Set ℕ := {2, 3, 4}
def complement_B : Set ℕ := { x ∈ universal_set | x ∉ set_B }

theorem union_complement_equals_set :
  set_A ∪ complement_B = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_union_complement_equals_set_l839_83943


namespace NUMINAMATH_GPT_inequality_division_l839_83909

variable (m n : ℝ)

theorem inequality_division (h : m > n) : (m / 4) > (n / 4) :=
sorry

end NUMINAMATH_GPT_inequality_division_l839_83909


namespace NUMINAMATH_GPT_mother_to_father_age_ratio_l839_83913

def DarcieAge : ℕ := 4
def FatherAge : ℕ := 30
def MotherAge : ℕ := DarcieAge * 6

theorem mother_to_father_age_ratio :
  (MotherAge : ℚ) / (FatherAge : ℚ) = (4 / 5) := by
  sorry

end NUMINAMATH_GPT_mother_to_father_age_ratio_l839_83913


namespace NUMINAMATH_GPT_derivative_at_one_max_value_l839_83920

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Prove that f'(1) = 0
theorem derivative_at_one : deriv f 1 = 0 :=
by sorry

-- Prove that the maximum value of f(x) is 2
theorem max_value : ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = 2 :=
by sorry

end NUMINAMATH_GPT_derivative_at_one_max_value_l839_83920


namespace NUMINAMATH_GPT_sum_of_valid_b_values_l839_83996

/-- Given a quadratic equation 3x² + 7x + b = 0, where b is a positive integer,
and the requirement that the equation must have rational roots, the sum of all
possible positive integer values of b is 6. -/
theorem sum_of_valid_b_values : 
  ∃ (b_values : List ℕ), 
    (∀ b ∈ b_values, 0 < b ∧ ∃ n : ℤ, 49 - 12 * b = n^2) ∧ b_values.sum = 6 :=
by sorry

end NUMINAMATH_GPT_sum_of_valid_b_values_l839_83996


namespace NUMINAMATH_GPT_Ron_needs_to_drink_80_percent_l839_83927

theorem Ron_needs_to_drink_80_percent 
  (volume_each : ℕ)
  (volume_intelligence : ℕ)
  (volume_beauty : ℕ)
  (volume_strength : ℕ)
  (volume_second_pitcher : ℕ)
  (effective_volume : ℕ)
  (volume_intelligence_left : ℕ)
  (volume_beauty_left : ℕ)
  (volume_strength_left : ℕ)
  (total_volume : ℕ)
  (Ron_needs : ℕ)
  (intelligence_condition : effective_volume = 30)
  (initial_volumes : volume_each = 300)
  (first_drink : volume_intelligence = volume_each / 2)
  (mix_before_second_drink : volume_second_pitcher = volume_intelligence + volume_beauty)
  (Hermione_drink : volume_second_pitcher / 2 = volume_intelligence_left + volume_beauty_left)
  (Harry_drink : volume_strength_left = volume_each / 2)
  (second_mix : volume_second_pitcher = volume_intelligence_left + volume_beauty_left + volume_strength_left)
  (final_mix : volume_second_pitcher / 2 = volume_intelligence_left + volume_beauty_left + volume_strength_left)
  (Ron_needs_condition : Ron_needs = effective_volume / volume_intelligence_left * 100)
  : Ron_needs = 80 := sorry

end NUMINAMATH_GPT_Ron_needs_to_drink_80_percent_l839_83927


namespace NUMINAMATH_GPT_lucky_larry_l839_83926

theorem lucky_larry (a b c d e k : ℤ) 
    (h1 : a = 2) 
    (h2 : b = 3) 
    (h3 : c = 4) 
    (h4 : d = 5)
    (h5 : a - b - c - d + e = 2 - (b - (c - (d + e)))) 
    (h6 : k * 2 = e) : 
    k = 2 := by
  sorry

end NUMINAMATH_GPT_lucky_larry_l839_83926


namespace NUMINAMATH_GPT_find_A_l839_83981

theorem find_A (A B C : ℕ) (h1 : A = B * C + 8) (h2 : A + B + C = 2994) : A = 8 ∨ A = 2864 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l839_83981


namespace NUMINAMATH_GPT_total_number_of_books_ways_to_select_books_l839_83918

def first_layer_books : ℕ := 6
def second_layer_books : ℕ := 5
def third_layer_books : ℕ := 4

theorem total_number_of_books : first_layer_books + second_layer_books + third_layer_books = 15 := by
  sorry

theorem ways_to_select_books : first_layer_books * second_layer_books * third_layer_books = 120 := by
  sorry

end NUMINAMATH_GPT_total_number_of_books_ways_to_select_books_l839_83918


namespace NUMINAMATH_GPT_amount_in_paise_l839_83980

theorem amount_in_paise (a : ℝ) (h_a : a = 170) (percentage_value : ℝ) (h_percentage : percentage_value = 0.5 / 100) : 
  (percentage_value * a * 100) = 85 := 
by
  sorry

end NUMINAMATH_GPT_amount_in_paise_l839_83980


namespace NUMINAMATH_GPT_gcd_459_357_l839_83967

theorem gcd_459_357 : Nat.gcd 459 357 = 51 :=
by
  sorry

end NUMINAMATH_GPT_gcd_459_357_l839_83967


namespace NUMINAMATH_GPT_domain_f_domain_g_intersection_M_N_l839_83994

namespace MathProof

open Set

def M : Set ℝ := { x | -2 < x ∧ x < 4 }
def N : Set ℝ := { x | x < 1 ∨ x ≥ 3 }

theorem domain_f :
  (M = { x : ℝ | -2 < x ∧ x < 4 }) := by
  sorry

theorem domain_g :
  (N = { x : ℝ | x < 1 ∨ x ≥ 3 }) := by
  sorry

theorem intersection_M_N : 
  (M ∩ N = { x : ℝ | (-2 < x ∧ x < 1) ∨ (3 ≤ x ∧ x < 4) }) := by
  sorry

end MathProof

end NUMINAMATH_GPT_domain_f_domain_g_intersection_M_N_l839_83994


namespace NUMINAMATH_GPT_find_x_l839_83992

-- Declaration for the custom operation
def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

-- Theorem statement
theorem find_x (x : ℝ) (h : star 3 x = 23) : x = 29 / 6 :=
by {
    sorry -- The proof steps are to be filled here.
}

end NUMINAMATH_GPT_find_x_l839_83992


namespace NUMINAMATH_GPT_total_amount_earned_is_90_l839_83993

variable (W : ℕ)

-- Define conditions
def work_capacity_condition : Prop :=
  5 = W ∧ W = 8

-- Define wage per man in Rs.
def wage_per_man : ℕ := 6

-- Define total amount earned by 5 men
def total_earned_by_5_men : ℕ := 5 * wage_per_man

-- Define total amount for the problem
def total_earned (W : ℕ) : ℕ :=
  3 * total_earned_by_5_men

-- The final proof statement
theorem total_amount_earned_is_90 (W : ℕ) (h : work_capacity_condition W) : total_earned W = 90 := by
  sorry

end NUMINAMATH_GPT_total_amount_earned_is_90_l839_83993


namespace NUMINAMATH_GPT_function_evaluation_l839_83968

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = 2 * x ^ 2 + 1) : 
  ∀ x : ℝ, f x = 2 * x ^ 2 - 4 * x + 3 :=
sorry

end NUMINAMATH_GPT_function_evaluation_l839_83968


namespace NUMINAMATH_GPT_possible_values_of_sum_l839_83937

theorem possible_values_of_sum
  (p q r : ℝ)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_system : q = p * (4 - p) ∧ r = q * (4 - q) ∧ p = r * (4 - r)) :
  p + q + r = 6 ∨ p + q + r = 7 := by
  sorry

end NUMINAMATH_GPT_possible_values_of_sum_l839_83937


namespace NUMINAMATH_GPT_negation_equivalent_statement_l839_83928

theorem negation_equivalent_statement (x y : ℝ) :
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0)) :=
sorry

end NUMINAMATH_GPT_negation_equivalent_statement_l839_83928


namespace NUMINAMATH_GPT_area_of_square_land_l839_83947

-- Define the problem conditions
variable (A P : ℕ)

-- Define the main theorem statement: proving area A given the conditions
theorem area_of_square_land (h₁ : 5 * A = 10 * P + 45) (h₂ : P = 36) : A = 81 := by
  sorry

end NUMINAMATH_GPT_area_of_square_land_l839_83947


namespace NUMINAMATH_GPT_prove_ineq_l839_83923

-- Define the quadratic equation
def quadratic_eqn (a b x : ℝ) : Prop :=
  3 * x^2 + 3 * (a + b) * x + 4 * a * b = 0

-- Define the root relation
def root_relation (x1 x2 : ℝ) : Prop :=
  x1 * (x1 + 1) + x2 * (x2 + 1) = (x1 + 1) * (x2 + 1)

-- State the theorem
theorem prove_ineq (a b : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_eqn a b x1 ∧ quadratic_eqn a b x2 ∧ root_relation x1 x2) →
  (a + b)^2 ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_prove_ineq_l839_83923


namespace NUMINAMATH_GPT_largest_angle_smallest_angle_middle_angle_l839_83931

-- Definitions for angles of a triangle in degrees
variable (α β γ : ℝ)
variable (h_sum : α + β + γ = 180)

-- Largest angle condition
theorem largest_angle (h1 : α ≥ β) (h2 : α ≥ γ) : (60 ≤ α ∧ α < 180) :=
  sorry

-- Smallest angle condition
theorem smallest_angle (h1 : α ≤ β) (h2 : α ≤ γ) : (0 < α ∧ α ≤ 60) :=
  sorry

-- Middle angle condition
theorem middle_angle (h1 : α > β ∧ α < γ ∨ α < β ∧ α > γ) : (0 < α ∧ α < 90) :=
  sorry

end NUMINAMATH_GPT_largest_angle_smallest_angle_middle_angle_l839_83931


namespace NUMINAMATH_GPT_regular_polygon_sides_l839_83997

-- Define the main theorem statement
theorem regular_polygon_sides (n : ℕ) : 
  (n > 2) ∧ 
  ((n - 2) * 180 / n - 360 / n = 90) → 
  n = 8 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l839_83997


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l839_83990

/-- 
  Given an isosceles triangle with two sides of length 6 and the third side of length 2,
  prove that the perimeter of the triangle is 14.
-/
theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 6) (h2 : b = 6) (h3 : c = 2) 
  (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  a + b + c = 14 :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l839_83990


namespace NUMINAMATH_GPT_range_of_k_l839_83904

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x = 4 ∧ x < 2) → (k < 1 ∨ k > 3) := 
by 
  sorry

end NUMINAMATH_GPT_range_of_k_l839_83904


namespace NUMINAMATH_GPT_equal_expense_sharing_l839_83906

variables (O L B : ℝ)

theorem equal_expense_sharing (h1 : O < L) (h2 : O < B) : 
    (L + B - 2 * O) / 6 = (O + L + B) / 3 - O :=
by
    sorry

end NUMINAMATH_GPT_equal_expense_sharing_l839_83906


namespace NUMINAMATH_GPT_movie_marathon_first_movie_length_l839_83989

theorem movie_marathon_first_movie_length 
  (x : ℝ)
  (h2 : 1.5 * x = second_movie)
  (h3 : second_movie + x - 1 = last_movie)
  (h4 : (x + second_movie + last_movie) = 9)
  (h5 : last_movie = 2.5 * x - 1) :
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_movie_marathon_first_movie_length_l839_83989


namespace NUMINAMATH_GPT_company_ordered_weight_of_stone_l839_83910

theorem company_ordered_weight_of_stone :
  let weight_concrete := 0.16666666666666666
  let weight_bricks := 0.16666666666666666
  let total_material := 0.8333333333333334
  let weight_stone := total_material - (weight_concrete + weight_bricks)
  weight_stone = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_company_ordered_weight_of_stone_l839_83910


namespace NUMINAMATH_GPT_intersection_A_B_l839_83917

def A := { x : ℝ | -1 < x ∧ x ≤ 3 }
def B := { x : ℝ | 0 < x ∧ x < 10 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 0 < x ∧ x ≤ 3 } :=
  by sorry

end NUMINAMATH_GPT_intersection_A_B_l839_83917


namespace NUMINAMATH_GPT_mean_temperature_l839_83938

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem mean_temperature (temps : List ℝ) (length_temps_10 : temps.length = 10)
    (temps_vals : temps = [78, 80, 82, 85, 88, 90, 92, 95, 97, 95]) : 
    mean temps = 88.2 := by
  sorry

end NUMINAMATH_GPT_mean_temperature_l839_83938


namespace NUMINAMATH_GPT_abs_neg_six_l839_83972

theorem abs_neg_six : abs (-6) = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_abs_neg_six_l839_83972


namespace NUMINAMATH_GPT_tv_horizontal_length_l839_83942

-- Conditions
def is_rectangular_tv (width height : ℝ) : Prop :=
width / height = 9 / 12

def diagonal_is (d : ℝ) : Prop :=
d = 32

-- Theorem to prove
theorem tv_horizontal_length (width height diagonal : ℝ) 
(h1 : is_rectangular_tv width height) 
(h2 : diagonal_is diagonal) : 
width = 25.6 := by 
sorry

end NUMINAMATH_GPT_tv_horizontal_length_l839_83942


namespace NUMINAMATH_GPT_gravel_cost_l839_83902

-- Definitions of conditions
def lawn_length : ℝ := 70
def lawn_breadth : ℝ := 30
def road_width : ℝ := 5
def gravel_cost_per_sqm : ℝ := 4

-- Theorem statement
theorem gravel_cost : (lawn_length * road_width + lawn_breadth * road_width - road_width * road_width) * gravel_cost_per_sqm = 1900 :=
by
  -- Definitions used in the problem
  let area_first_road := lawn_length * road_width
  let area_second_road := lawn_breadth * road_width
  let area_intersection := road_width * road_width

  -- Total area to be graveled
  let total_area_to_be_graveled := area_first_road + area_second_road - area_intersection

  -- Calculate the cost
  let cost := total_area_to_be_graveled * gravel_cost_per_sqm

  show cost = 1900
  sorry

end NUMINAMATH_GPT_gravel_cost_l839_83902


namespace NUMINAMATH_GPT_no_real_solution_implies_a_range_l839_83901

noncomputable def quadratic (a x : ℝ) : ℝ := x^2 - 4 * x + a^2

theorem no_real_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, quadratic a x ≤ 0 → false) ↔ a < -2 ∨ a > 2 := 
sorry

end NUMINAMATH_GPT_no_real_solution_implies_a_range_l839_83901


namespace NUMINAMATH_GPT_initial_logs_l839_83900

theorem initial_logs (x : ℕ) (h1 : x - 3 - 3 - 3 + 2 + 2 + 2 = 3) : x = 6 := by
  sorry

end NUMINAMATH_GPT_initial_logs_l839_83900


namespace NUMINAMATH_GPT_division_problem_solution_l839_83963

theorem division_problem_solution (x : ℝ) (h : (2.25 / x) * 12 = 9) : x = 3 :=
sorry

end NUMINAMATH_GPT_division_problem_solution_l839_83963


namespace NUMINAMATH_GPT_parts_processed_per_hour_before_innovation_l839_83941

variable (x : ℝ) (h : 1500 / x - 1500 / (2.5 * x) = 18)

theorem parts_processed_per_hour_before_innovation : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_parts_processed_per_hour_before_innovation_l839_83941


namespace NUMINAMATH_GPT_unique_intersection_point_l839_83905

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 9 * x + 15

theorem unique_intersection_point : ∃ a : ℝ, f a = a ∧ f a = -1 ∧ f a = f⁻¹ a :=
by 
  sorry

end NUMINAMATH_GPT_unique_intersection_point_l839_83905


namespace NUMINAMATH_GPT_superhero_speed_l839_83944

def convert_speed (speed_mph : ℕ) (mile_to_km : ℚ) : ℚ :=
  let speed_kmh := (speed_mph : ℚ) * (1 / mile_to_km)
  speed_kmh / 60

theorem superhero_speed :
  convert_speed 36000 (6 / 10) = 1000 :=
by sorry

end NUMINAMATH_GPT_superhero_speed_l839_83944


namespace NUMINAMATH_GPT_factor_expression_l839_83930

theorem factor_expression (x : ℝ) : 
  (21 * x ^ 4 + 90 * x ^ 3 + 40 * x - 10) - (7 * x ^ 4 + 6 * x ^ 3 + 8 * x - 6) = 
  2 * x * (7 * x ^ 3 + 42 * x ^ 2 + 16) - 4 :=
by sorry

end NUMINAMATH_GPT_factor_expression_l839_83930


namespace NUMINAMATH_GPT_additional_pots_last_hour_l839_83929

theorem additional_pots_last_hour (h1 : 60 / 6 = 10) (h2 : 60 / 5 = 12) : 12 - 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_additional_pots_last_hour_l839_83929


namespace NUMINAMATH_GPT_find_difference_l839_83916

theorem find_difference (x y : ℚ) (h₁ : x + y = 520) (h₂ : x / y = 3 / 4) : y - x = 520 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_l839_83916


namespace NUMINAMATH_GPT_simplify_evaluate_expr_l839_83965

theorem simplify_evaluate_expr (x y : ℚ) (h₁ : x = -1) (h₂ : y = -1 / 2) :
  (4 * x * y + (2 * x^2 + 5 * x * y - y^2) - 2 * (x^2 + 3 * x * y)) = 5 / 4 :=
by
  rw [h₁, h₂]
  -- Here we would include the specific algebra steps to convert the LHS to 5/4.
  sorry

end NUMINAMATH_GPT_simplify_evaluate_expr_l839_83965


namespace NUMINAMATH_GPT_common_factor_l839_83949

theorem common_factor (x y a b : ℤ) : 
  3 * x * (a - b) - 9 * y * (b - a) = 3 * (a - b) * (x + 3 * y) :=
by {
  sorry
}

end NUMINAMATH_GPT_common_factor_l839_83949


namespace NUMINAMATH_GPT_find_triples_l839_83960

theorem find_triples (a b c : ℝ) : 
  a + b + c = 14 ∧ a^2 + b^2 + c^2 = 84 ∧ a^3 + b^3 + c^3 = 584 ↔ (a = 4 ∧ b = 2 ∧ c = 8) ∨ (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 8 ∧ b = 2 ∧ c = 4) :=
by
  sorry

end NUMINAMATH_GPT_find_triples_l839_83960


namespace NUMINAMATH_GPT_expression_never_prime_l839_83986

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem expression_never_prime (p : ℕ) (hp : is_prime p) : ¬ is_prime (p^2 + 20) := sorry

end NUMINAMATH_GPT_expression_never_prime_l839_83986


namespace NUMINAMATH_GPT_rational_solutions_exist_l839_83957

theorem rational_solutions_exist (x p q : ℚ) (h : p^2 - x * q^2 = 1) :
  ∃ (a b : ℤ), p = (a^2 + x * b^2) / (a^2 - x * b^2) ∧ q = (2 * a * b) / (a^2 - x * b^2) :=
by
  sorry

end NUMINAMATH_GPT_rational_solutions_exist_l839_83957


namespace NUMINAMATH_GPT_x_squared_minus_y_squared_l839_83908

theorem x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 9 / 16)
  (h2 : x - y = 5 / 16) :
  x^2 - y^2 = 45 / 256 :=
by
  sorry

end NUMINAMATH_GPT_x_squared_minus_y_squared_l839_83908


namespace NUMINAMATH_GPT_ratio_of_areas_l839_83984

-- Definitions of perimeter in Lean terms
def P_A : ℕ := 16
def P_B : ℕ := 32

-- Ratio of the area of region A to region C
theorem ratio_of_areas (s_A s_C : ℕ) (h₀ : 4 * s_A = P_A)
  (h₁ : 4 * s_C = 12) : s_A^2 / s_C^2 = 1 / 9 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l839_83984


namespace NUMINAMATH_GPT_fraction_of_garden_occupied_by_triangle_beds_l839_83974

theorem fraction_of_garden_occupied_by_triangle_beds :
  ∀ (rect_height rect_width trapezoid_short_base trapezoid_long_base : ℝ) 
    (num_triangles : ℕ) 
    (triangle_leg_length : ℝ)
    (total_area_triangles : ℝ)
    (total_garden_area : ℝ)
    (fraction : ℝ),
  rect_height = 10 → rect_width = 30 →
  trapezoid_short_base = 20 → trapezoid_long_base = 30 → num_triangles = 3 →
  triangle_leg_length = 10 / 3 →
  total_area_triangles = 3 * (1 / 2 * (triangle_leg_length ^ 2)) →
  total_garden_area = rect_height * rect_width →
  fraction = total_area_triangles / total_garden_area →
  fraction = 1 / 18 := by
  intros rect_height rect_width trapezoid_short_base trapezoid_long_base
         num_triangles triangle_leg_length total_area_triangles
         total_garden_area fraction
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_fraction_of_garden_occupied_by_triangle_beds_l839_83974


namespace NUMINAMATH_GPT_volume_and_surface_area_of_inscribed_sphere_l839_83971

theorem volume_and_surface_area_of_inscribed_sphere (edge_length : ℝ) (h_edge : edge_length = 10) :
    let r := edge_length / 2
    let V := (4 / 3) * π * r^3
    let A := 4 * π * r^2
    V = (500 / 3) * π ∧ A = 100 * π := 
by
  sorry

end NUMINAMATH_GPT_volume_and_surface_area_of_inscribed_sphere_l839_83971


namespace NUMINAMATH_GPT_categorize_numbers_l839_83912

def numbers : List ℚ := [-16/10, -5/6, 89/10, -7, 1/12, 0, 25]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x.den ≠ 1
def is_negative_integer (x : ℚ) : Prop := x < 0 ∧ x.den = 1

theorem categorize_numbers :
  { x | x ∈ numbers ∧ is_positive x } = { 89 / 10, 1 / 12, 25 } ∧
  { x | x ∈ numbers ∧ is_negative_fraction x } = { -5 / 6 } ∧
  { x | x ∈ numbers ∧ is_negative_integer x } = { -7 } := by
  sorry

end NUMINAMATH_GPT_categorize_numbers_l839_83912


namespace NUMINAMATH_GPT_greatest_number_of_cool_cells_l839_83975

noncomputable def greatest_cool_cells (n : ℕ) (grid : Matrix (Fin n) (Fin n) ℝ) : ℕ :=
n^2 - 2 * n + 1

theorem greatest_number_of_cool_cells (n : ℕ) (grid : Matrix (Fin n) (Fin n) ℝ) (h : 0 < n) :
  ∃ m, m = (n - 1)^2 ∧ m = greatest_cool_cells n grid :=
sorry

end NUMINAMATH_GPT_greatest_number_of_cool_cells_l839_83975


namespace NUMINAMATH_GPT_cylinder_radius_and_volume_l839_83924

theorem cylinder_radius_and_volume
  (h : ℝ) (surface_area : ℝ) :
  h = 8 ∧ surface_area = 130 * Real.pi →
  ∃ (r : ℝ) (V : ℝ), r = 5 ∧ V = 200 * Real.pi := by
  sorry

end NUMINAMATH_GPT_cylinder_radius_and_volume_l839_83924


namespace NUMINAMATH_GPT_measure_of_angle_C_maximum_area_of_triangle_l839_83939

noncomputable def triangle (A B C a b c : ℝ) : Prop :=
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C ∧
  2 * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 2 * a - b) * Real.sin B

theorem measure_of_angle_C :
  ∀ (A B C a b c : ℝ),
  triangle A B C a b c →
  C = π / 4 :=
by
  intros A B C a b c h
  sorry

theorem maximum_area_of_triangle :
  ∀ (A B C a b c : ℝ),
  triangle A B C a b c →
  C = π / 4 →
  1 / 2 * a * b * Real.sin C = (Real.sqrt 2 / 2 + 1 / 2) :=
by
  intros A B C a b c h hC
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_maximum_area_of_triangle_l839_83939


namespace NUMINAMATH_GPT_team_leads_per_supervisor_l839_83985

def num_workers : ℕ := 390
def num_supervisors : ℕ := 13
def leads_per_worker_ratio : ℕ := 10

theorem team_leads_per_supervisor : (num_workers / leads_per_worker_ratio) / num_supervisors = 3 :=
by
  sorry

end NUMINAMATH_GPT_team_leads_per_supervisor_l839_83985


namespace NUMINAMATH_GPT_smallest_root_of_g_l839_83966

noncomputable def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem smallest_root_of_g : ∀ x : ℝ, g x = 0 → x = - Real.sqrt (3 / 7) :=
by
  sorry

end NUMINAMATH_GPT_smallest_root_of_g_l839_83966


namespace NUMINAMATH_GPT_maximum_consecutive_positive_integers_sum_500_l839_83955

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end NUMINAMATH_GPT_maximum_consecutive_positive_integers_sum_500_l839_83955


namespace NUMINAMATH_GPT_probability_of_three_primes_out_of_six_l839_83952

-- Define the conditions
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

-- Given six 12-sided fair dice
def total_dice : ℕ := 6
def sides : ℕ := 12

-- Probability of rolling a prime number on one die
def prime_probability : ℚ := 5 / 12

-- Probability of rolling a non-prime number on one die
def non_prime_probability : ℚ := 7 / 12

-- Number of ways to choose 3 dice from 6 to show a prime number
def combination (n k : ℕ) : ℕ := n.choose k
def choose_3_out_of_6 : ℕ := combination total_dice 3

-- Combined probability for exactly 3 primes and 3 non-primes
def combined_probability : ℚ :=
  (prime_probability ^ 3) * (non_prime_probability ^ 3)

-- Total probability
def total_probability : ℚ :=
  choose_3_out_of_6 * combined_probability

-- Main theorem statement
theorem probability_of_three_primes_out_of_six :
  total_probability = 857500 / 5177712 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_three_primes_out_of_six_l839_83952


namespace NUMINAMATH_GPT_find_k_l839_83936

theorem find_k (k : ℝ) (h_line : ∀ x y : ℝ, 3 * x + 5 * y + k = 0)
    (h_sum_intercepts : - (k / 3) - (k / 5) = 16) : k = -30 := by
  sorry

end NUMINAMATH_GPT_find_k_l839_83936


namespace NUMINAMATH_GPT_remainder_of_n_mod_1000_l839_83978

-- Definition of the set S
def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 15 }

-- Define the number of sets of three non-empty disjoint subsets of S
def num_sets_of_three_non_empty_disjoint_subsets (S : Set ℕ) : ℕ :=
  let total_partitions := 4^15
  let single_empty_partition := 3 * 3^15
  let double_empty_partition := 3 * 2^15
  let all_empty_partition := 1
  total_partitions - single_empty_partition + double_empty_partition - all_empty_partition

-- Compute the result of the number modulo 1000
def result := (num_sets_of_three_non_empty_disjoint_subsets S) % 1000

-- Theorem that states the remainder when n is divided by 1000
theorem remainder_of_n_mod_1000 : result = 406 := by
  sorry

end NUMINAMATH_GPT_remainder_of_n_mod_1000_l839_83978


namespace NUMINAMATH_GPT_delivery_driver_stops_l839_83973

theorem delivery_driver_stops (total_boxes : ℕ) (boxes_per_stop : ℕ) (stops : ℕ) :
  total_boxes = 27 → boxes_per_stop = 9 → stops = total_boxes / boxes_per_stop → stops = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_delivery_driver_stops_l839_83973


namespace NUMINAMATH_GPT_fraction_sum_l839_83907

theorem fraction_sum :
  (1 / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 5 / 6 : ℚ) = -5 / 6 :=
by sorry

end NUMINAMATH_GPT_fraction_sum_l839_83907


namespace NUMINAMATH_GPT_combinations_sum_l839_83933
open Nat

theorem combinations_sum : 
  let d := [1, 2, 3, 4]
  let count_combinations (n : Nat) := factorial n
  count_combinations 1 + count_combinations 2 + count_combinations 3 + count_combinations 4 = 64 :=
  by
    sorry

end NUMINAMATH_GPT_combinations_sum_l839_83933


namespace NUMINAMATH_GPT_days_in_month_l839_83935

theorem days_in_month 
  (S : ℕ) (D : ℕ) (h1 : 150 * S + 120 * D = (S + D) * 125) (h2 : S = 5) :
  S + D = 30 :=
by
  sorry

end NUMINAMATH_GPT_days_in_month_l839_83935


namespace NUMINAMATH_GPT_find_ax5_by5_l839_83919

theorem find_ax5_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 21)
  (h4 : a * x^4 + b * y^4 = 55) :
  a * x^5 + b * y^5 = -131 :=
sorry

end NUMINAMATH_GPT_find_ax5_by5_l839_83919


namespace NUMINAMATH_GPT_revenue_fraction_l839_83958

variable (N D J : ℝ)
variable (h1 : J = 1 / 5 * N)
variable (h2 : D = 4.166666666666666 * (N + J) / 2)

theorem revenue_fraction (h1 : J = 1 / 5 * N) (h2 : D = 4.166666666666666 * (N + J) / 2) : N / D = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_revenue_fraction_l839_83958


namespace NUMINAMATH_GPT_intersection_of_circle_and_line_l839_83948

theorem intersection_of_circle_and_line 
  (α : ℝ) 
  (x y : ℝ)
  (h1 : x = Real.cos α) 
  (h2 : y = 1 + Real.sin α) 
  (h3 : y = 1) :
  (x, y) = (1, 1) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_circle_and_line_l839_83948


namespace NUMINAMATH_GPT_sum_possible_x_coordinates_l839_83998

-- Define the vertices of the parallelogram
def A := (1, 2)
def B := (3, 8)
def C := (4, 1)

-- Definition of what it means to be a fourth vertex that forms a parallelogram
def is_fourth_vertex (D : ℤ × ℤ) : Prop :=
  (D = (6, 7)) ∨ (D = (2, -5)) ∨ (D = (0, 9))

-- The sum of possible x-coordinates for the fourth vertex
def sum_x_coordinates : ℤ :=
  6 + 2 + 0

theorem sum_possible_x_coordinates :
  (∃ D, is_fourth_vertex D) → sum_x_coordinates = 8 :=
by
  -- Sorry is used to skip the detailed proof steps
  sorry

end NUMINAMATH_GPT_sum_possible_x_coordinates_l839_83998


namespace NUMINAMATH_GPT_range_of_a_l839_83964

def P (x : ℝ) : Prop := x^2 ≤ 1

def M (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) (h : ∀ x, (P x ∨ x = a) ↔ P x) : P a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l839_83964


namespace NUMINAMATH_GPT_hemisphere_containers_needed_l839_83921

theorem hemisphere_containers_needed 
  (total_volume : ℕ) (volume_per_hemisphere : ℕ) 
  (h₁ : total_volume = 11780) 
  (h₂ : volume_per_hemisphere = 4) : 
  total_volume / volume_per_hemisphere = 2945 := 
by
  sorry

end NUMINAMATH_GPT_hemisphere_containers_needed_l839_83921


namespace NUMINAMATH_GPT_resulting_curve_eq_l839_83922

def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

def transformed_curve (x y: ℝ) : Prop := 
  ∃ (x0 y0 : ℝ), 
    is_on_circle x0 y0 ∧ 
    x = x0 ∧ 
    y = 4 * y0

theorem resulting_curve_eq : ∀ (x y : ℝ), transformed_curve x y → (x^2 / 9 + y^2 / 144 = 1) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_resulting_curve_eq_l839_83922


namespace NUMINAMATH_GPT_problem_condition_relationship_l839_83999

theorem problem_condition_relationship (x : ℝ) :
  (x^2 - x - 2 > 0) → (|x - 1| > 1) := 
sorry

end NUMINAMATH_GPT_problem_condition_relationship_l839_83999


namespace NUMINAMATH_GPT_ratio_blue_gill_to_bass_l839_83956

theorem ratio_blue_gill_to_bass (bass trout blue_gill : ℕ) 
  (h1 : bass = 32)
  (h2 : trout = bass / 4)
  (h3 : bass + trout + blue_gill = 104) 
: blue_gill / bass = 2 := 
sorry

end NUMINAMATH_GPT_ratio_blue_gill_to_bass_l839_83956


namespace NUMINAMATH_GPT_range_of_f_l839_83970

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2

theorem range_of_f : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f x ∈ Set.Icc (-18 : ℝ) (2 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_of_f_l839_83970


namespace NUMINAMATH_GPT_number_division_l839_83995

theorem number_division (x : ℚ) (h : x / 6 = 1 / 10) : (x / (3 / 25)) = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_division_l839_83995


namespace NUMINAMATH_GPT_sum_of_fractions_eq_five_fourteen_l839_83951

theorem sum_of_fractions_eq_five_fourteen :
  (1 : ℚ) / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) = 5 / 14 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_eq_five_fourteen_l839_83951


namespace NUMINAMATH_GPT_count_right_triangles_with_given_conditions_l839_83979

-- Define the type of our points as a pair of integers
def Point := (ℤ × ℤ)

-- Define the orthocenter being a specific point
def isOrthocenter (P : Point) := P = (-1, 7)

-- Define that a given triangle has a right angle at the origin
def rightAngledAtOrigin (O A B : Point) :=
  O = (0, 0) ∧
  (A.fst = 0 ∨ A.snd = 0) ∧
  (B.fst = 0 ∨ B.snd = 0) ∧
  (A.fst ≠ 0 ∨ A.snd ≠ 0) ∧
  (B.fst ≠ 0 ∨ B.snd ≠ 0)

-- Define that the points are lattice points
def areLatticePoints (O A B : Point) :=
  ∃ t k : ℤ, (A = (3 * t, 4 * t) ∧ B = (-4 * k, 3 * k)) ∨
            (B = (3 * t, 4 * t) ∧ A = (-4 * k, 3 * k))

-- Define the number of right triangles given the constraints
def numberOfRightTriangles : ℕ := 2

-- Statement of the problem
theorem count_right_triangles_with_given_conditions :
  ∃ (O A B : Point),
    rightAngledAtOrigin O A B ∧
    isOrthocenter (-1, 7) ∧
    areLatticePoints O A B ∧
    numberOfRightTriangles = 2 :=
  sorry

end NUMINAMATH_GPT_count_right_triangles_with_given_conditions_l839_83979


namespace NUMINAMATH_GPT_positive_integer_with_four_smallest_divisors_is_130_l839_83954

theorem positive_integer_with_four_smallest_divisors_is_130:
  ∃ n : ℕ, ∀ p1 p2 p3 p4 : ℕ, 
    n = p1^2 + p2^2 + p3^2 + p4^2 ∧
    p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧
    ∀ p : ℕ, p ∣ n → (p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4) → 
    n = 130 :=
  by
  sorry

end NUMINAMATH_GPT_positive_integer_with_four_smallest_divisors_is_130_l839_83954


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l839_83940

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_sum
  (h1 : a 1 = 2)
  (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l839_83940


namespace NUMINAMATH_GPT_find_S25_l839_83961

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Definitions based on conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) - a n = a 1 - a 0
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

-- Condition that given S_{15} - S_{10} = 1
axiom sum_difference : S 15 - S 10 = 1

-- Theorem we need to prove
theorem find_S25 (h_arith : is_arithmetic_sequence a) (h_sum : sum_of_first_n_terms a S) : S 25 = 5 :=
by
-- Placeholder for the actual proof
sorry

end NUMINAMATH_GPT_find_S25_l839_83961


namespace NUMINAMATH_GPT_player_holds_seven_black_cards_l839_83976

theorem player_holds_seven_black_cards
    (total_cards : ℕ := 13)
    (num_red_cards : ℕ := 6)
    (S D H C : ℕ)
    (h1 : D = 2 * S)
    (h2 : H = 2 * D)
    (h3 : C = 6)
    (h4 : S + D + H + C = total_cards) :
    S + C = 7 := 
by
  sorry

end NUMINAMATH_GPT_player_holds_seven_black_cards_l839_83976


namespace NUMINAMATH_GPT_winner_percentage_l839_83962

theorem winner_percentage (votes_winner : ℕ) (votes_difference : ℕ) (total_votes : ℕ) 
  (h1 : votes_winner = 1044) 
  (h2 : votes_difference = 288) 
  (h3 : total_votes = votes_winner + (votes_winner - votes_difference)) :
  (votes_winner * 100) / total_votes = 58 :=
by
  sorry

end NUMINAMATH_GPT_winner_percentage_l839_83962


namespace NUMINAMATH_GPT_jason_attended_games_l839_83925

-- Define the conditions as given in the problem
def games_planned_this_month : ℕ := 11
def games_planned_last_month : ℕ := 17
def games_missed : ℕ := 16

-- Define the total number of games planned
def games_planned_total : ℕ := games_planned_this_month + games_planned_last_month

-- Define the number of games attended
def games_attended : ℕ := games_planned_total - games_missed

-- Prove that Jason attended 12 games
theorem jason_attended_games : games_attended = 12 := by
  -- The proof is omitted, but the theorem statement is required
  sorry

end NUMINAMATH_GPT_jason_attended_games_l839_83925


namespace NUMINAMATH_GPT_general_term_sequence_sum_of_cn_l839_83977

theorem general_term_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : S 2 = 3)
  (hS_eq : ∀ n, 2 * S n = n + n * a n) :
  ∀ n, a n = n :=
by
  sorry

theorem sum_of_cn (S : ℕ → ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℕ)
  (hS : S 2 = 3)
  (hS_eq : ∀ n, 2 * S n = n + n * a n)
  (ha : ∀ n, a n = n)
  (hc_odd : ∀ n, c (2 * n - 1) = a (2 * n))
  (hc_even : ∀ n, c (2 * n) = 3 * 2^(a (2 * n - 1)) + 1) :
  ∀ n, T (2 * n) = 2^(2 * n + 1) + n^2 + 2 * n - 2 :=
by
  sorry

end NUMINAMATH_GPT_general_term_sequence_sum_of_cn_l839_83977


namespace NUMINAMATH_GPT_min_value_of_expression_l839_83987

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) : x^2 + (1 / 4) * y^2 ≥ 1 / 8 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l839_83987


namespace NUMINAMATH_GPT_describe_T_correctly_l839_83911

def T (x y : ℝ) : Prop :=
(x = 2 ∧ y < 7) ∨ (y = 7 ∧ x < 2) ∨ (y = x + 5 ∧ x > 2)

theorem describe_T_correctly :
  (∀ x y : ℝ, T x y ↔
    ((x = 2 ∧ y < 7) ∨ (y = 7 ∧ x < 2) ∨ (y = x + 5 ∧ x > 2))) :=
by
  sorry

end NUMINAMATH_GPT_describe_T_correctly_l839_83911


namespace NUMINAMATH_GPT_cylinder_volume_l839_83983

theorem cylinder_volume (short_side long_side : ℝ) (h_short_side : short_side = 12) (h_long_side : long_side = 18) : 
  ∀ (r h : ℝ) (h_radius : r = short_side / 2) (h_height : h = long_side), 
    volume = π * r^2 * h := 
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_l839_83983


namespace NUMINAMATH_GPT_find_xyz_l839_83945

variables (A B C B₁ A₁ C₁ : Type)
variables [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B] [AddCommGroup C] [Module ℝ C]

def AC1 (AB BC CC₁ : A) (x y z : ℝ) : A :=
  x • AB + 2 • y • BC + 3 • z • CC₁

theorem find_xyz (AB BC CC₁ AC1 : A)
  (h1 : AC1 = AB + BC + CC₁)
  (h2 : AC1 = x • AB + 2 • y • BC + 3 • z • CC₁) :
  x + y + z = 11 / 6 :=
sorry

end NUMINAMATH_GPT_find_xyz_l839_83945


namespace NUMINAMATH_GPT_find_a_and_tangent_point_l839_83950

noncomputable def tangent_line_and_curve (a : ℚ) (P : ℚ × ℚ) : Prop :=
  ∃ (x₀ : ℚ), (P = (x₀, x₀ + a)) ∧ (P = (x₀, x₀^3 - x₀^2 + 1)) ∧ (3*x₀^2 - 2*x₀ = 1)

theorem find_a_and_tangent_point :
  ∃ (a : ℚ) (P : ℚ × ℚ), tangent_line_and_curve a P ∧ a = 32/27 ∧ P = (-1/3, 23/27) :=
sorry

end NUMINAMATH_GPT_find_a_and_tangent_point_l839_83950


namespace NUMINAMATH_GPT_acute_angle_inequality_l839_83915

theorem acute_angle_inequality (α : ℝ) (h₀ : 0 < α) (h₁ : α < π / 2) :
  α < (Real.sin α + Real.tan α) / 2 := 
sorry

end NUMINAMATH_GPT_acute_angle_inequality_l839_83915


namespace NUMINAMATH_GPT_find_p_q_sum_l839_83934

theorem find_p_q_sum (p q : ℝ) 
  (sum_condition : p / 3 = 8) 
  (product_condition : q / 3 = 12) : 
  p + q = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_sum_l839_83934


namespace NUMINAMATH_GPT_probability_ace_king_queen_same_suit_l839_83903

theorem probability_ace_king_queen_same_suit :
  let total_probability := (1 : ℝ) / 52 * (1 : ℝ) / 51 * (1 : ℝ) / 50
  total_probability = (1 : ℝ) / 132600 :=
by
  sorry

end NUMINAMATH_GPT_probability_ace_king_queen_same_suit_l839_83903


namespace NUMINAMATH_GPT_meghan_total_money_l839_83988

theorem meghan_total_money :
  let num_100_bills := 2
  let num_50_bills := 5
  let num_10_bills := 10
  let value_100_bills := num_100_bills * 100
  let value_50_bills := num_50_bills * 50
  let value_10_bills := num_10_bills * 10
  let total_money := value_100_bills + value_50_bills + value_10_bills
  total_money = 550 := by sorry

end NUMINAMATH_GPT_meghan_total_money_l839_83988


namespace NUMINAMATH_GPT_greatest_power_of_two_factor_l839_83953

theorem greatest_power_of_two_factor (n : ℕ) (h : n = 1000) :
  ∃ k, 2^k ∣ 10^n + 4^(n/2) ∧ k = 1003 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_power_of_two_factor_l839_83953


namespace NUMINAMATH_GPT_daily_wage_c_l839_83991

theorem daily_wage_c (a_days b_days c_days total_earnings : ℕ)
  (ratio_a_b ratio_b_c : ℚ)
  (a_wage b_wage c_wage : ℚ) :
  a_days = 6 →
  b_days = 9 →
  c_days = 4 →
  total_earnings = 1480 →
  ratio_a_b = 3 / 4 →
  ratio_b_c = 4 / 5 →
  b_wage = ratio_a_b * a_wage → 
  c_wage = ratio_b_c * b_wage → 
  a_days * a_wage + b_days * b_wage + c_days * c_wage = total_earnings →
  c_wage = 100 / 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_daily_wage_c_l839_83991


namespace NUMINAMATH_GPT_equal_costs_l839_83982

noncomputable def cost_scheme_1 (x : ℕ) : ℝ := 350 + 5 * x

noncomputable def cost_scheme_2 (x : ℕ) : ℝ := 360 + 4.5 * x

theorem equal_costs (x : ℕ) : cost_scheme_1 x = cost_scheme_2 x ↔ x = 20 := by
  sorry

end NUMINAMATH_GPT_equal_costs_l839_83982
