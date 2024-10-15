import Mathlib

namespace NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l422_42228

theorem sum_of_other_endpoint_coordinates (x y : ℤ)
  (h1 : (6 + x) / 2 = 3)
  (h2 : (-1 + y) / 2 = 6) :
  x + y = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l422_42228


namespace NUMINAMATH_GPT_length_of_second_train_l422_42249

theorem length_of_second_train 
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (time_to_cross : ℝ)
  (relative_speed : ℝ)
  (total_distance_covered : ℝ)
  (L : ℝ)
  (h1 : length_first_train = 210)
  (h2 : speed_first_train = 120 * 1000 / 3600)
  (h3 : speed_second_train = 80 * 1000 / 3600)
  (h4 : time_to_cross = 9)
  (h5 : relative_speed = (120 * 1000 / 3600) + (80 * 1000 / 3600))
  (h6 : total_distance_covered = relative_speed * time_to_cross)
  (h7 : total_distance_covered = length_first_train + L) : 
  L = 289.95 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_of_second_train_l422_42249


namespace NUMINAMATH_GPT_most_likely_outcome_is_draw_l422_42237

variable (P_A_wins : ℝ) (P_A_not_loses : ℝ)

def P_draw (P_A_wins P_A_not_loses : ℝ) : ℝ := 
  P_A_not_loses - P_A_wins

def P_B_wins (P_A_not_loses P_A_wins : ℝ) : ℝ :=
  1 - P_A_not_loses

theorem most_likely_outcome_is_draw 
  (h₁: P_A_wins = 0.3) 
  (h₂: P_A_not_loses = 0.7)
  (h₃: 0 ≤ P_A_wins) 
  (h₄: P_A_wins ≤ 1) 
  (h₅: 0 ≤ P_A_not_loses) 
  (h₆: P_A_not_loses ≤ 1) : 
  max (P_A_wins) (max (P_B_wins P_A_not_loses P_A_wins) (P_draw P_A_wins P_A_not_loses)) = P_draw P_A_wins P_A_not_loses :=
by
  sorry

end NUMINAMATH_GPT_most_likely_outcome_is_draw_l422_42237


namespace NUMINAMATH_GPT_part1_l422_42200

variable (A B C : ℝ)
variable (a b c S : ℝ)
variable (h1 : a * (1 + Real.cos C) + c * (1 + Real.cos A) = (5 / 2) * b)
variable (h2 : a * Real.cos C + c * Real.cos A = b)

theorem part1 : 2 * (a + c) = 3 * b := 
sorry

end NUMINAMATH_GPT_part1_l422_42200


namespace NUMINAMATH_GPT_weekly_allowance_l422_42224

theorem weekly_allowance
  (video_game_cost : ℝ)
  (sales_tax_percentage : ℝ)
  (weeks_to_save : ℕ)
  (total_with_tax : ℝ)
  (total_savings : ℝ) :
  video_game_cost = 50 →
  sales_tax_percentage = 0.10 →
  weeks_to_save = 11 →
  total_with_tax = video_game_cost * (1 + sales_tax_percentage) →
  total_savings = weeks_to_save * (0.5 * total_savings) →
  total_savings = total_with_tax →
  total_savings = 55 :=
by
  intros
  sorry

end NUMINAMATH_GPT_weekly_allowance_l422_42224


namespace NUMINAMATH_GPT_fraction_product_l422_42278

theorem fraction_product : (1/2) * (3/5) * (7/11) * (4/13) = 84/1430 := by
  sorry

end NUMINAMATH_GPT_fraction_product_l422_42278


namespace NUMINAMATH_GPT_animal_group_divisor_l422_42272

theorem animal_group_divisor (cows sheep goats total groups : ℕ)
    (hc : cows = 24) 
    (hs : sheep = 7) 
    (hg : goats = 113) 
    (ht : total = cows + sheep + goats) 
    (htotal : total = 144) 
    (hdiv : groups ∣ total) 
    (hexclude1 : groups ≠ 1) 
    (hexclude144 : groups ≠ 144) : 
    ∃ g, g = groups ∧ g ∈ [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72] :=
  by 
  sorry

end NUMINAMATH_GPT_animal_group_divisor_l422_42272


namespace NUMINAMATH_GPT_claudia_total_earnings_l422_42215

def cost_per_beginner_class : Int := 15
def cost_per_advanced_class : Int := 20
def num_beginner_kids_saturday : Int := 20
def num_advanced_kids_saturday : Int := 10
def num_sibling_pairs : Int := 5
def sibling_discount : Int := 3

theorem claudia_total_earnings : 
  let beginner_earnings_saturday := num_beginner_kids_saturday * cost_per_beginner_class
  let advanced_earnings_saturday := num_advanced_kids_saturday * cost_per_advanced_class
  let total_earnings_saturday := beginner_earnings_saturday + advanced_earnings_saturday
  
  let num_beginner_kids_sunday := num_beginner_kids_saturday / 2
  let num_advanced_kids_sunday := num_advanced_kids_saturday / 2
  let beginner_earnings_sunday := num_beginner_kids_sunday * cost_per_beginner_class
  let advanced_earnings_sunday := num_advanced_kids_sunday * cost_per_advanced_class
  let total_earnings_sunday := beginner_earnings_sunday + advanced_earnings_sunday

  let total_earnings_no_discount := total_earnings_saturday + total_earnings_sunday

  let total_sibling_discount := num_sibling_pairs * 2 * sibling_discount
  
  let total_earnings := total_earnings_no_discount - total_sibling_discount
  total_earnings = 720 := 
by
  sorry

end NUMINAMATH_GPT_claudia_total_earnings_l422_42215


namespace NUMINAMATH_GPT_sets_of_headphones_l422_42240

-- Definitions of the conditions
variable (M H : ℕ)

-- Theorem statement for proving the question given the conditions
theorem sets_of_headphones (h1 : 5 * M + 30 * H = 840) (h2 : 3 * M + 120 = 480) : H = 8 := by
  sorry

end NUMINAMATH_GPT_sets_of_headphones_l422_42240


namespace NUMINAMATH_GPT_sector_area_max_radius_l422_42227

noncomputable def arc_length (R : ℝ) : ℝ := 20 - 2 * R

noncomputable def sector_area (R : ℝ) : ℝ :=
  let l := arc_length R
  0.5 * l * R

theorem sector_area_max_radius :
  ∃ (R : ℝ), sector_area R = -R^2 + 10 * R ∧
             R = 5 :=
sorry

end NUMINAMATH_GPT_sector_area_max_radius_l422_42227


namespace NUMINAMATH_GPT_figure_count_mistake_l422_42220

theorem figure_count_mistake
    (b g : ℕ)
    (total_figures : ℕ)
    (boy_circles boy_squares girl_circles girl_squares : ℕ)
    (total_figures_counted : ℕ) :
  boy_circles = 3 → boy_squares = 8 → girl_circles = 9 → girl_squares = 2 →
  total_figures_counted = 4046 →
  (∃ (b g : ℕ), 11 * b + 11 * g ≠ 4046) :=
by
  intros
  sorry

end NUMINAMATH_GPT_figure_count_mistake_l422_42220


namespace NUMINAMATH_GPT_probability_prime_and_cube_is_correct_l422_42281

-- Conditions based on the problem
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_cube (n : ℕ) : Prop :=
  n = 1 ∨ n = 8

def possible_outcomes := 8 * 8
def successful_outcomes := 4 * 2

noncomputable def probability_of_prime_and_cube :=
  (successful_outcomes : ℝ) / (possible_outcomes : ℝ)

theorem probability_prime_and_cube_is_correct :
  probability_of_prime_and_cube = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_prime_and_cube_is_correct_l422_42281


namespace NUMINAMATH_GPT_condition_needs_l422_42267

theorem condition_needs (a b c d : ℝ) :
  a + c > b + d → (¬ (a > b ∧ c > d) ∧ (a > b ∧ c > d)) :=
by
  sorry

end NUMINAMATH_GPT_condition_needs_l422_42267


namespace NUMINAMATH_GPT_M_eq_N_l422_42275

def M : Set ℤ := { u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l }
def N : Set ℤ := { u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r }

theorem M_eq_N : M = N := by
  sorry

end NUMINAMATH_GPT_M_eq_N_l422_42275


namespace NUMINAMATH_GPT_tower_height_l422_42241

theorem tower_height (h d : ℝ) 
  (tan_30_eq : Real.tan (Real.pi / 6) = h / d)
  (tan_45_eq : Real.tan (Real.pi / 4) = h / (d - 20)) :
  h = 20 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tower_height_l422_42241


namespace NUMINAMATH_GPT_f_inequality_l422_42269

def f (x : ℝ) : ℝ := sorry

axiom f_defined : ∀ x : ℝ, 0 < x → ∃ y : ℝ, f x = y

axiom f_additive : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y

axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

axiom f_two : f 2 = 1

theorem f_inequality (x : ℝ) : 3 < x → x ≤ 4 → f x + f (x - 3) ≤ 2 :=
sorry

end NUMINAMATH_GPT_f_inequality_l422_42269


namespace NUMINAMATH_GPT_solve_conjugate_l422_42293
open Complex

-- Problem definition:
def Z (a : ℝ) : ℂ := ⟨a, 1⟩  -- Z = a + i

def conj_Z (a : ℝ) : ℂ := ⟨a, -1⟩  -- conjugate of Z

theorem solve_conjugate (a : ℝ) (h : Z a + conj_Z a = 4) : conj_Z 2 = 2 - I := by
  sorry

end NUMINAMATH_GPT_solve_conjugate_l422_42293


namespace NUMINAMATH_GPT_jimmy_sells_less_l422_42258

-- Definitions based on conditions
def num_figures : ℕ := 5
def value_figure_1_to_4 : ℕ := 15
def value_figure_5 : ℕ := 20
def total_earned : ℕ := 55

-- Formulation of the problem statement in Lean
theorem jimmy_sells_less (total_value : ℕ := (4 * value_figure_1_to_4) + value_figure_5) (difference : ℕ := total_value - total_earned) (amount_less_per_figure : ℕ := difference / num_figures) : amount_less_per_figure = 5 := by
  sorry

end NUMINAMATH_GPT_jimmy_sells_less_l422_42258


namespace NUMINAMATH_GPT_value_fraction_l422_42288

variables {x y : ℝ}
variables (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + 2 * y) / (x - 4 * y) = 3)

theorem value_fraction : (x + 4 * y) / (4 * x - y) = 10 / 57 :=
by { sorry }

end NUMINAMATH_GPT_value_fraction_l422_42288


namespace NUMINAMATH_GPT_domain_of_fx_l422_42236

theorem domain_of_fx {x : ℝ} : (2 * x) / (x - 1) = (2 * x) / (x - 1) ↔ x ∈ {y : ℝ | y ≠ 1} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_fx_l422_42236


namespace NUMINAMATH_GPT_square_area_of_triangle_on_hyperbola_l422_42274

noncomputable def centroid_is_vertex (triangle : Set (ℝ × ℝ)) : Prop :=
  ∃ v : ℝ × ℝ, v ∈ triangle ∧ v.1 * v.2 = 4

noncomputable def triangle_properties (triangle : Set (ℝ × ℝ)) : Prop :=
  centroid_is_vertex triangle ∧
  (∃ centroid : ℝ × ℝ, 
    centroid_is_vertex triangle ∧ 
    (∀ p ∈ triangle, centroid ∈ triangle))

theorem square_area_of_triangle_on_hyperbola :
  ∃ triangle : Set (ℝ × ℝ), triangle_properties triangle ∧ (∃ area_sq : ℝ, area_sq = 1728) :=
by
  sorry

end NUMINAMATH_GPT_square_area_of_triangle_on_hyperbola_l422_42274


namespace NUMINAMATH_GPT_trajectory_of_P_l422_42282

theorem trajectory_of_P (M P : ℝ × ℝ) (OM OP : ℝ) (x y : ℝ) :
  (M = (4, y)) →
  (P = (x, y)) →
  (OM = Real.sqrt (4^2 + y^2)) →
  (OP = Real.sqrt ((x - 4)^2 + y^2)) →
  (OM * OP = 16) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_GPT_trajectory_of_P_l422_42282


namespace NUMINAMATH_GPT_parallel_lines_condition_l422_42291

theorem parallel_lines_condition (a : ℝ) :
  ( ∀ x y : ℝ, (a * x + 2 * y + 2 = 0 → ∃ C₁ : ℝ, x - 2 * y = C₁) 
  ∧ (x + (a - 1) * y + 1 = 0 → ∃ C₂ : ℝ, x - 2 * y = C₂) )
  ↔ a = -1 :=
sorry

end NUMINAMATH_GPT_parallel_lines_condition_l422_42291


namespace NUMINAMATH_GPT_max_of_a_l422_42283

theorem max_of_a (a b c d : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0)
  (h5 : a + b + c + d = 4) (h6 : a^2 + b^2 + c^2 + d^2 = 8) : a ≤ 1 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_of_a_l422_42283


namespace NUMINAMATH_GPT_CatCafePawRatio_l422_42247

-- Define the context
def CatCafeMeow (P : ℕ) := 3 * P
def CatCafePaw (P : ℕ) := P
def CatCafeCool := 5
def TotalCats (P : ℕ) := CatCafeMeow P + CatCafePaw P

-- State the theorem
theorem CatCafePawRatio (P : ℕ) (n : ℕ) : 
  CatCafeCool = 5 →
  CatCafeMeow P = 3 * CatCafePaw P →
  TotalCats P = 40 →
  P = 10 →
  n * CatCafeCool = P →
  n = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_CatCafePawRatio_l422_42247


namespace NUMINAMATH_GPT_smallest_perfect_square_divisible_by_2_3_5_l422_42257

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end NUMINAMATH_GPT_smallest_perfect_square_divisible_by_2_3_5_l422_42257


namespace NUMINAMATH_GPT_sum_M_N_K_l422_42252

theorem sum_M_N_K (d K M N : ℤ) 
(h : ∀ x : ℤ, (x^2 + 3*x + 1) ∣ (x^4 - d*x^3 + M*x^2 + N*x + K)) :
  M + N + K = 5*K - 4*d - 11 := 
sorry

end NUMINAMATH_GPT_sum_M_N_K_l422_42252


namespace NUMINAMATH_GPT_percentage_passed_l422_42239

def swim_club_members := 100
def not_passed_course_taken := 40
def not_passed_course_not_taken := 30
def not_passed := not_passed_course_taken + not_passed_course_not_taken

theorem percentage_passed :
  ((swim_club_members - not_passed).toFloat / swim_club_members.toFloat * 100) = 30 := by
  sorry

end NUMINAMATH_GPT_percentage_passed_l422_42239


namespace NUMINAMATH_GPT_smallest_x_for_cubic_l422_42212

theorem smallest_x_for_cubic (x N : ℕ) (h1 : 1260 * x = N^3) : x = 7350 :=
sorry

end NUMINAMATH_GPT_smallest_x_for_cubic_l422_42212


namespace NUMINAMATH_GPT_prime_solution_unique_l422_42261

theorem prime_solution_unique {x y : ℕ} 
  (hx : Nat.Prime x)
  (hy : Nat.Prime y)
  (h : x ^ y - y ^ x = x * y ^ 2 - 19) :
  (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) :=
sorry

end NUMINAMATH_GPT_prime_solution_unique_l422_42261


namespace NUMINAMATH_GPT_value_of_fraction_l422_42226

theorem value_of_fraction (x y : ℤ) (h : x / y = 7 / 2) : (x - 2 * y) / y = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l422_42226


namespace NUMINAMATH_GPT_ratio_problem_l422_42296

theorem ratio_problem 
  (a b c d : ℚ)
  (h1 : a / b = 3)
  (h2 : b / c = 2 / 5)
  (h3 : c / d = 9) : 
  d / a = 5 / 54 :=
by
  sorry

end NUMINAMATH_GPT_ratio_problem_l422_42296


namespace NUMINAMATH_GPT_negation_of_squared_inequality_l422_42268

theorem negation_of_squared_inequality (p : ∀ n : ℕ, n^2 ≤ 2*n + 5) : 
  ∃ n : ℕ, n^2 > 2*n + 5 :=
sorry

end NUMINAMATH_GPT_negation_of_squared_inequality_l422_42268


namespace NUMINAMATH_GPT_independent_variable_range_l422_42209

/-- In the function y = 1 / (x - 2), the range of the independent variable x is all real numbers except 2. -/
theorem independent_variable_range (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_independent_variable_range_l422_42209


namespace NUMINAMATH_GPT_pizzas_returned_l422_42251

theorem pizzas_returned (total_pizzas served_pizzas : ℕ) (h_total : total_pizzas = 9) (h_served : served_pizzas = 3) : (total_pizzas - served_pizzas) = 6 :=
by
  sorry

end NUMINAMATH_GPT_pizzas_returned_l422_42251


namespace NUMINAMATH_GPT_find_b_l422_42235

-- Definitions based on the conditions in the problem
def eq1 (a : ℝ) := 3 * a + 3 = 0
def eq2 (a b : ℝ) := 2 * b - a = 4

-- Statement of the proof problem
theorem find_b (a b : ℝ) (h1 : eq1 a) (h2 : eq2 a b) : b = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l422_42235


namespace NUMINAMATH_GPT_total_time_taken_l422_42256

theorem total_time_taken
  (speed_boat : ℝ)
  (speed_stream : ℝ)
  (distance : ℝ)
  (h_boat : speed_boat = 12)
  (h_stream : speed_stream = 5)
  (h_distance : distance = 325) :
  (distance / (speed_boat - speed_stream) + distance / (speed_boat + speed_stream)) = 65.55 :=
by
  sorry

end NUMINAMATH_GPT_total_time_taken_l422_42256


namespace NUMINAMATH_GPT_multiply_and_divide_equiv_l422_42270

/-- Defines the operation of first multiplying by 4/5 and then dividing by 4/7 -/
def multiply_and_divide (x : ℚ) : ℚ :=
  (x * (4 / 5)) / (4 / 7)

/-- Statement to prove the operation is equivalent to multiplying by 7/5 -/
theorem multiply_and_divide_equiv (x : ℚ) : 
  multiply_and_divide x = x * (7 / 5) :=
by 
  -- This requires a proof, which we can assume here
  sorry

end NUMINAMATH_GPT_multiply_and_divide_equiv_l422_42270


namespace NUMINAMATH_GPT_original_pumpkins_count_l422_42280

def pumpkins_eaten_by_rabbits : ℕ := 23
def pumpkins_left : ℕ := 20
def original_pumpkins : ℕ := pumpkins_left + pumpkins_eaten_by_rabbits

theorem original_pumpkins_count :
  original_pumpkins = 43 :=
sorry

end NUMINAMATH_GPT_original_pumpkins_count_l422_42280


namespace NUMINAMATH_GPT_operation_result_l422_42232

-- Define the new operation x # y
def op (x y : ℕ) : ℤ := 2 * x * y - 3 * x + y

-- Prove that (6 # 4) - (4 # 6) = -8
theorem operation_result : op 6 4 - op 4 6 = -8 :=
by
  sorry

end NUMINAMATH_GPT_operation_result_l422_42232


namespace NUMINAMATH_GPT_initial_oranges_in_bowl_l422_42222

theorem initial_oranges_in_bowl (A O : ℕ) (R : ℚ) (h1 : A = 14) (h2 : R = 0.7) 
    (h3 : R * (A + O - 15) = A) : O = 21 := 
by 
  sorry

end NUMINAMATH_GPT_initial_oranges_in_bowl_l422_42222


namespace NUMINAMATH_GPT_remainder_3_pow_19_mod_10_l422_42204

theorem remainder_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_3_pow_19_mod_10_l422_42204


namespace NUMINAMATH_GPT_southton_capsule_depth_l422_42207

theorem southton_capsule_depth :
  ∃ S : ℕ, 4 * S + 12 = 48 ∧ S = 9 :=
by
  sorry

end NUMINAMATH_GPT_southton_capsule_depth_l422_42207


namespace NUMINAMATH_GPT_chord_length_perpendicular_l422_42246

theorem chord_length_perpendicular 
  (R a b : ℝ)  
  (h1 : a + b = R)
  (h2 : (1 / 2) * Real.pi * R^2 - (1 / 2) * Real.pi * (a^2 + b^2) = 10 * Real.pi) :
  2 * Real.sqrt 10 = 6.32 :=
by 
  sorry

end NUMINAMATH_GPT_chord_length_perpendicular_l422_42246


namespace NUMINAMATH_GPT_g_difference_l422_42297

def g (n : ℕ) : ℚ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

theorem g_difference (s : ℕ) : g s - g (s - 1) = s * (s + 1) * (s + 2) := 
by sorry

end NUMINAMATH_GPT_g_difference_l422_42297


namespace NUMINAMATH_GPT_tiles_on_square_area_l422_42260

theorem tiles_on_square_area (n : ℕ) (h1 : 2 * n - 1 = 25) : n ^ 2 = 169 :=
by
  sorry

end NUMINAMATH_GPT_tiles_on_square_area_l422_42260


namespace NUMINAMATH_GPT_non_zero_real_solution_l422_42238

theorem non_zero_real_solution (x : ℝ) (hx : x ≠ 0) (h : (3 * x)^5 = (9 * x)^4) : x = 27 :=
sorry

end NUMINAMATH_GPT_non_zero_real_solution_l422_42238


namespace NUMINAMATH_GPT_work_completion_time_for_A_l422_42201

-- Define the conditions
def B_completion_time : ℕ := 30
def joint_work_days : ℕ := 4
def work_left_fraction : ℚ := 2 / 3

-- Define the required proof statement
theorem work_completion_time_for_A (x : ℚ) : 
  (4 * (1 / x + 1 / B_completion_time) = 1 / 3) → x = 20 := 
by
  sorry

end NUMINAMATH_GPT_work_completion_time_for_A_l422_42201


namespace NUMINAMATH_GPT_total_meters_examined_l422_42255

-- Define the conditions
def proportion_defective : ℝ := 0.1
def defective_meters : ℕ := 10

-- The statement to prove
theorem total_meters_examined (T : ℝ) (h : proportion_defective * T = defective_meters) : T = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_meters_examined_l422_42255


namespace NUMINAMATH_GPT_jerry_total_miles_l422_42299

def monday : ℕ := 15
def tuesday : ℕ := 18
def wednesday : ℕ := 25
def thursday : ℕ := 12
def friday : ℕ := 10

def total : ℕ := monday + tuesday + wednesday + thursday + friday

theorem jerry_total_miles : total = 80 := by
  sorry

end NUMINAMATH_GPT_jerry_total_miles_l422_42299


namespace NUMINAMATH_GPT_gcd_459_357_eq_51_l422_42243

theorem gcd_459_357_eq_51 :
  gcd 459 357 = 51 := 
by
  sorry

end NUMINAMATH_GPT_gcd_459_357_eq_51_l422_42243


namespace NUMINAMATH_GPT_initial_eggs_proof_l422_42203

noncomputable def initial_eggs (total_cost : ℝ) (price_per_egg : ℝ) (leftover_eggs : ℝ) : ℝ :=
  let eggs_sold := total_cost / price_per_egg
  eggs_sold + leftover_eggs

theorem initial_eggs_proof : initial_eggs 5 0.20 5 = 30 := by
  sorry

end NUMINAMATH_GPT_initial_eggs_proof_l422_42203


namespace NUMINAMATH_GPT_keaton_annual_profit_l422_42218

theorem keaton_annual_profit :
  let orange_harvests_per_year := 12 / 2
  let apple_harvests_per_year := 12 / 3
  let peach_harvests_per_year := 12 / 4
  let blackberry_harvests_per_year := 12 / 6

  let orange_profit_per_harvest := 50 - 20
  let apple_profit_per_harvest := 30 - 15
  let peach_profit_per_harvest := 45 - 25
  let blackberry_profit_per_harvest := 70 - 30

  let total_orange_profit := orange_harvests_per_year * orange_profit_per_harvest
  let total_apple_profit := apple_harvests_per_year * apple_profit_per_harvest
  let total_peach_profit := peach_harvests_per_year * peach_profit_per_harvest
  let total_blackberry_profit := blackberry_harvests_per_year * blackberry_profit_per_harvest

  let total_annual_profit := total_orange_profit + total_apple_profit + total_peach_profit + total_blackberry_profit

  total_annual_profit = 380
:= by
  sorry

end NUMINAMATH_GPT_keaton_annual_profit_l422_42218


namespace NUMINAMATH_GPT_total_prizes_l422_42262

-- Definitions of the conditions
def stuffedAnimals : ℕ := 14
def frisbees : ℕ := 18
def yoYos : ℕ := 18

-- The statement to be proved
theorem total_prizes : stuffedAnimals + frisbees + yoYos = 50 := by
  sorry

end NUMINAMATH_GPT_total_prizes_l422_42262


namespace NUMINAMATH_GPT_neznaika_mistake_correct_numbers_l422_42277

theorem neznaika_mistake_correct_numbers (N : ℕ) :
  (10 ≤ N) ∧ (N ≤ 99) ∧
  ¬ (N % 30 = 0) ∧
  (N % 3 = 0) ∧
  ¬ (N % 10 = 0) ∧
  ((N % 9 = 0) ∨ (N % 15 = 0) ∨ (N % 18 = 0)) ∧
  (N % 5 ≠ 0) ∧
  (N % 4 ≠ 0) → N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end NUMINAMATH_GPT_neznaika_mistake_correct_numbers_l422_42277


namespace NUMINAMATH_GPT_square_garden_dimensions_and_area_increase_l422_42233

def original_length : ℝ := 60
def original_width : ℝ := 20

def original_area : ℝ := original_length * original_width
def original_perimeter : ℝ := 2 * (original_length + original_width)

theorem square_garden_dimensions_and_area_increase
    (L : ℝ := 60) (W : ℝ := 20)
    (orig_area : ℝ := L * W)
    (orig_perimeter : ℝ := 2 * (L + W))
    (square_side_length : ℝ := orig_perimeter / 4)
    (new_area : ℝ := square_side_length * square_side_length)
    (area_increase : ℝ := new_area - orig_area) :
    square_side_length = 40 ∧ area_increase = 400 :=
by {sorry}

end NUMINAMATH_GPT_square_garden_dimensions_and_area_increase_l422_42233


namespace NUMINAMATH_GPT_fabric_needed_for_coats_l422_42271

variable (m d : ℝ)

def condition1 := 4 * m + 2 * d = 16
def condition2 := 2 * m + 6 * d = 18

theorem fabric_needed_for_coats (h1 : condition1 m d) (h2 : condition2 m d) :
  m = 3 ∧ d = 2 :=
by
  sorry

end NUMINAMATH_GPT_fabric_needed_for_coats_l422_42271


namespace NUMINAMATH_GPT_distinct_sums_count_l422_42230

theorem distinct_sums_count (n : ℕ) (a : Fin n.succ → ℕ) (h_distinct : Function.Injective a) :
  ∃ (S : Finset ℕ), S.card ≥ n * (n + 1) / 2 := sorry

end NUMINAMATH_GPT_distinct_sums_count_l422_42230


namespace NUMINAMATH_GPT_total_distance_traveled_l422_42225

theorem total_distance_traveled (d d1 d2 d3 d4 d5 : ℕ) 
  (h1 : d1 = d)
  (h2 : d2 = 2 * d)
  (h3 : d3 = 40)
  (h4 : d = 2 * d3)
  (h5 : d4 = 2 * (d1 + d2 + d3))
  (h6 : d5 = 3 * d4 / 2) 
  : d1 + d2 + d3 + d4 + d5 = 1680 :=
by
  have hd : d = 80 := sorry
  have hd1 : d1 = 80 := sorry
  have hd2 : d2 = 160 := sorry
  have hd4 : d4 = 560 := sorry
  have hd5 : d5 = 840 := sorry
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l422_42225


namespace NUMINAMATH_GPT_polygon_sides_l422_42286

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 1080) : n = 8 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l422_42286


namespace NUMINAMATH_GPT_parking_methods_count_l422_42210

theorem parking_methods_count : 
  ∃ (n : ℕ), n = 72 ∧ (∃ (spaces cars slots remainingSlots : ℕ), 
  spaces = 7 ∧ cars = 3 ∧ slots = 1 ∧ remainingSlots = 4 ∧
  ∃ (perm_ways slot_ways : ℕ), perm_ways = 6 ∧ slot_ways = 12 ∧ n = perm_ways * slot_ways) :=
  by
    sorry

end NUMINAMATH_GPT_parking_methods_count_l422_42210


namespace NUMINAMATH_GPT_floor_sqrt_23_squared_l422_42294

theorem floor_sqrt_23_squared : (Int.floor (Real.sqrt 23))^2 = 16 := 
by
  -- conditions
  have h1 : 4^2 = 16 := by norm_num
  have h2 : 5^2 = 25 := by norm_num
  have h3 : 16 < 23 := by norm_num
  have h4 : 23 < 25 := by norm_num
  -- statement (goal)
  sorry

end NUMINAMATH_GPT_floor_sqrt_23_squared_l422_42294


namespace NUMINAMATH_GPT_coords_of_a_in_m_n_l422_42208

variable {R : Type} [Field R]

def coords_in_basis (a : R × R) (p q : R × R) (c1 c2 : R) : Prop :=
  a = c1 • p + c2 • q

theorem coords_of_a_in_m_n
  (a p q m n : R × R)
  (hp : p = (1, -1)) (hq : q = (2, 1)) (hm : m = (-1, 1)) (hn : n = (1, 2))
  (coords_pq : coords_in_basis a p q (-2) 2) :
  coords_in_basis a m n 0 2 :=
by
  sorry

end NUMINAMATH_GPT_coords_of_a_in_m_n_l422_42208


namespace NUMINAMATH_GPT_apple_counting_l422_42254

theorem apple_counting
  (n m : ℕ)
  (vasya_trees_a_b petya_trees_a_b vasya_trees_b_c petya_trees_b_c vasya_trees_c_d petya_trees_c_d vasya_apples_a_b petya_apples_a_b vasya_apples_c_d petya_apples_c_d : ℕ)
  (h1 : petya_trees_a_b = 2 * vasya_trees_a_b)
  (h2 : petya_apples_a_b = 7 * vasya_apples_a_b)
  (h3 : petya_trees_b_c = 2 * vasya_trees_b_c)
  (h4 : petya_trees_c_d = 2 * vasya_trees_c_d)
  (h5 : n = vasya_trees_a_b + petya_trees_a_b)
  (h6 : m = vasya_apples_a_b + petya_apples_a_b)
  (h7 : vasya_trees_c_d = n / 3)
  (h8 : petya_trees_c_d = 2 * (n / 3))
  (h9 : vasya_apples_c_d = 3 * petya_apples_c_d)
  : vasya_apples_c_d = 3 * petya_apples_c_d :=
by 
  sorry

end NUMINAMATH_GPT_apple_counting_l422_42254


namespace NUMINAMATH_GPT_no_angle_sat_sin_cos_eq_sin_40_l422_42287

open Real

theorem no_angle_sat_sin_cos_eq_sin_40 :
  ¬∃ α : ℝ, sin α * cos α = sin (40 * π / 180) := 
by 
  sorry

end NUMINAMATH_GPT_no_angle_sat_sin_cos_eq_sin_40_l422_42287


namespace NUMINAMATH_GPT_bob_wins_l422_42265

-- Define the notion of nim-sum used in nim-games
def nim_sum (a b : ℕ) : ℕ := Nat.xor a b

-- Define nim-values for given walls based on size
def nim_value : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| 3 => 3
| 4 => 1
| 5 => 4
| 6 => 3
| 7 => 2
| _ => 0

-- Calculate the nim-value of a given configuration
def nim_config (c : List ℕ) : ℕ :=
c.foldl (λ acc n => nim_sum acc (nim_value n)) 0

-- Prove that the configuration (7, 3, 1) gives a nim-value of 0
theorem bob_wins : nim_config [7, 3, 1] = 0 := by
  sorry

end NUMINAMATH_GPT_bob_wins_l422_42265


namespace NUMINAMATH_GPT_strawberry_cost_l422_42289

theorem strawberry_cost (price_per_basket : ℝ) (num_baskets : ℕ) (total_cost : ℝ)
  (h1 : price_per_basket = 16.50) (h2 : num_baskets = 4) : total_cost = 66.00 :=
by
  sorry

end NUMINAMATH_GPT_strawberry_cost_l422_42289


namespace NUMINAMATH_GPT_initial_friends_l422_42273

theorem initial_friends (n : ℕ) (h1 : 120 / (n - 4) = 120 / n + 8) : n = 10 := 
by
  sorry

end NUMINAMATH_GPT_initial_friends_l422_42273


namespace NUMINAMATH_GPT_number_of_pecan_pies_is_4_l422_42229

theorem number_of_pecan_pies_is_4 (apple_pies pumpkin_pies total_pies pecan_pies : ℕ) 
  (h1 : apple_pies = 2) 
  (h2 : pumpkin_pies = 7) 
  (h3 : total_pies = 13) 
  (h4 : pecan_pies = total_pies - (apple_pies + pumpkin_pies)) 
  : pecan_pies = 4 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_pecan_pies_is_4_l422_42229


namespace NUMINAMATH_GPT_base7_to_base5_l422_42248

theorem base7_to_base5 (n : ℕ) (h : n = 305) : 
    3 * 7 ^ 2 + 0 * 7 ^ 1 + 5 = 152 → 152 = 1 * 5 ^ 3 + 1 * 5 ^ 2 + 0 * 5 ^ 1 + 2 * 5 ^ 0 → 305 = 1102 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_base7_to_base5_l422_42248


namespace NUMINAMATH_GPT_average_speed_second_half_l422_42266

theorem average_speed_second_half
  (d : ℕ) (s1 : ℕ) (t : ℕ)
  (h1 : d = 3600)
  (h2 : s1 = 90)
  (h3 : t = 30) :
  (d / 2) / (t - (d / 2 / s1)) = 180 := by
  sorry

end NUMINAMATH_GPT_average_speed_second_half_l422_42266


namespace NUMINAMATH_GPT_trig_identity_l422_42295

theorem trig_identity :
  (Real.cos (105 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) * Real.sin (105 * Real.pi / 180)) = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_trig_identity_l422_42295


namespace NUMINAMATH_GPT_parabola_distance_l422_42276

theorem parabola_distance (p : ℝ) (hp : 0 < p) (hf : ∀ P : ℝ × ℝ, P ∈ {Q : ℝ × ℝ | Q.1^2 = 2 * p * Q.2} →
  dist P (0, p / 2) = 16) (hx : ∀ P : ℝ × ℝ, P ∈ {Q : ℝ × ℝ | Q.1^2 = 2 * p * Q.2} →
  P.2 = 10) : p = 12 :=
sorry

end NUMINAMATH_GPT_parabola_distance_l422_42276


namespace NUMINAMATH_GPT_Matt_overall_profit_l422_42290

def initialValue : ℕ := 8 * 6

def valueGivenAwayTrade1 : ℕ := 2 * 6
def valueReceivedTrade1 : ℕ := 3 * 2 + 9

def valueGivenAwayTrade2 : ℕ := 2 + 6
def valueReceivedTrade2 : ℕ := 2 * 5 + 8

def valueGivenAwayTrade3 : ℕ := 5 + 9
def valueReceivedTrade3 : ℕ := 3 * 3 + 10 + 1

def valueGivenAwayTrade4 : ℕ := 2 * 3 + 8
def valueReceivedTrade4 : ℕ := 2 * 7 + 4

def overallProfit : ℕ :=
  (valueReceivedTrade1 - valueGivenAwayTrade1) +
  (valueReceivedTrade2 - valueGivenAwayTrade2) +
  (valueReceivedTrade3 - valueGivenAwayTrade3) +
  (valueReceivedTrade4 - valueGivenAwayTrade4)

theorem Matt_overall_profit : overallProfit = 23 :=
by
  unfold overallProfit valueReceivedTrade1 valueGivenAwayTrade1 valueReceivedTrade2 valueGivenAwayTrade2 valueReceivedTrade3 valueGivenAwayTrade3 valueReceivedTrade4 valueGivenAwayTrade4
  linarith

end NUMINAMATH_GPT_Matt_overall_profit_l422_42290


namespace NUMINAMATH_GPT_compare_abc_l422_42245

noncomputable def a : ℝ := 1 / Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.exp 0.5
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > c ∧ c > a := by
  sorry

end NUMINAMATH_GPT_compare_abc_l422_42245


namespace NUMINAMATH_GPT_johns_age_l422_42244

theorem johns_age (J : ℕ) (h : J + 9 = 3 * (J - 11)) : J = 21 :=
sorry

end NUMINAMATH_GPT_johns_age_l422_42244


namespace NUMINAMATH_GPT_rate_of_current_is_5_l422_42285

theorem rate_of_current_is_5 
  (speed_still_water : ℕ)
  (distance_travelled : ℕ)
  (time_travelled : ℚ) 
  (effective_speed_with_current : ℚ) : 
  speed_still_water = 20 ∧ distance_travelled = 5 ∧ time_travelled = 1/5 ∧ 
  effective_speed_with_current = (speed_still_water + 5) →
  effective_speed_with_current * time_travelled = distance_travelled :=
by
  sorry

end NUMINAMATH_GPT_rate_of_current_is_5_l422_42285


namespace NUMINAMATH_GPT_mike_total_spending_l422_42253

def mike_spent_on_speakers : ℝ := 235.87
def mike_spent_on_tires : ℝ := 281.45
def mike_spent_on_steering_wheel_cover : ℝ := 179.99
def mike_spent_on_seat_covers : ℝ := 122.31
def mike_spent_on_headlights : ℝ := 98.63

theorem mike_total_spending :
  mike_spent_on_speakers + mike_spent_on_tires + mike_spent_on_steering_wheel_cover + mike_spent_on_seat_covers + mike_spent_on_headlights = 918.25 :=
  sorry

end NUMINAMATH_GPT_mike_total_spending_l422_42253


namespace NUMINAMATH_GPT_trig_order_l422_42205

theorem trig_order (θ : ℝ) (h1 : -Real.pi / 8 < θ) (h2 : θ < 0) : Real.tan θ < Real.sin θ ∧ Real.sin θ < Real.cos θ := 
sorry

end NUMINAMATH_GPT_trig_order_l422_42205


namespace NUMINAMATH_GPT_number_of_lines_l422_42292

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero: a ≠ 0 ∨ b ≠ 0

-- Definition of a line passing through a point P
def passes_through (l : Line) (P : Point) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Definition of a line having equal intercepts on x-axis and y-axis
def equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.a = l.b

-- Definition of a specific point P
def P : Point := { x := 1, y := 2 }

-- The theorem statement
theorem number_of_lines : ∃ (lines : Finset Line), (∀ l ∈ lines, passes_through l P ∧ equal_intercepts l) ∧ lines.card = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_lines_l422_42292


namespace NUMINAMATH_GPT_sum_of_7_more_likely_than_sum_of_8_l422_42250

noncomputable def probability_sum_equals_seven : ℚ := 6 / 36
noncomputable def probability_sum_equals_eight : ℚ := 5 / 36

theorem sum_of_7_more_likely_than_sum_of_8 :
  probability_sum_equals_seven > probability_sum_equals_eight :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_7_more_likely_than_sum_of_8_l422_42250


namespace NUMINAMATH_GPT_garden_perimeter_is_64_l422_42223

theorem garden_perimeter_is_64 :
    ∀ (width_garden length_garden width_playground length_playground : ℕ),
    width_garden = 24 →
    width_playground = 12 →
    length_playground = 16 →
    width_playground * length_playground = width_garden * length_garden →
    2 * length_garden + 2 * width_garden = 64 :=
by
  intros width_garden length_garden width_playground length_playground
  intro h1
  intro h2
  intro h3
  intro h4
  sorry

end NUMINAMATH_GPT_garden_perimeter_is_64_l422_42223


namespace NUMINAMATH_GPT_smallest_b_l422_42259

theorem smallest_b {a b c d : ℕ} (r : ℕ) 
  (h1 : a = b - r) (h2 : c = b + r) (h3 : d = b + 2 * r) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h5 : a * b * c * d = 256) : b = 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_l422_42259


namespace NUMINAMATH_GPT_fraction_meaningful_l422_42217

theorem fraction_meaningful (a : ℝ) : (a + 3 ≠ 0) ↔ (a ≠ -3) :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l422_42217


namespace NUMINAMATH_GPT_interest_rate_second_part_l422_42242

theorem interest_rate_second_part 
    (total_investment : ℝ) 
    (annual_interest : ℝ) 
    (P1 : ℝ) 
    (rate1 : ℝ) 
    (P2 : ℝ)
    (rate2 : ℝ) : 
    total_investment = 3600 → 
    annual_interest = 144 → 
    P1 = 1800 → 
    rate1 = 3 → 
    P2 = total_investment - P1 → 
    (annual_interest - (P1 * rate1 / 100)) = (P2 * rate2 / 100) →
    rate2 = 5 :=
by 
  intros total_investment_eq annual_interest_eq P1_eq rate1_eq P2_eq interest_eq
  sorry

end NUMINAMATH_GPT_interest_rate_second_part_l422_42242


namespace NUMINAMATH_GPT_identity_function_l422_42279

theorem identity_function (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) :
  ∀ y : ℝ, f y = y :=
by 
  sorry

end NUMINAMATH_GPT_identity_function_l422_42279


namespace NUMINAMATH_GPT_complement_of_P_l422_42214

def U : Set ℤ := {-1, 0, 1, 2}
def P : Set ℤ := {x | x^2 < 2}

theorem complement_of_P :
  (U \ P) = {2} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_P_l422_42214


namespace NUMINAMATH_GPT_evaluate_expression_l422_42206

theorem evaluate_expression : 6 / (-1 / 2 + 1 / 3) = -36 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l422_42206


namespace NUMINAMATH_GPT_least_faces_triangular_pyramid_l422_42202

def triangular_prism_faces : ℕ := 5
def quadrangular_prism_faces : ℕ := 6
def triangular_pyramid_faces : ℕ := 4
def quadrangular_pyramid_faces : ℕ := 5
def truncated_quadrangular_pyramid_faces : ℕ := 5 -- assuming the minimum possible value

theorem least_faces_triangular_pyramid :
  triangular_pyramid_faces < triangular_prism_faces ∧
  triangular_pyramid_faces < quadrangular_prism_faces ∧
  triangular_pyramid_faces < quadrangular_pyramid_faces ∧
  triangular_pyramid_faces ≤ truncated_quadrangular_pyramid_faces :=
by
  sorry

end NUMINAMATH_GPT_least_faces_triangular_pyramid_l422_42202


namespace NUMINAMATH_GPT_range_of_a_for_monotonicity_l422_42284

noncomputable def f (x : ℝ) (a : ℝ) := (Real.sqrt (x^2 + 1)) - a * x

theorem range_of_a_for_monotonicity (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x a < f y a) ↔ a ≥ 1 := sorry

end NUMINAMATH_GPT_range_of_a_for_monotonicity_l422_42284


namespace NUMINAMATH_GPT_king_plan_feasibility_l422_42298

-- Create a predicate for the feasibility of the king's plan
def feasible (n : ℕ) : Prop :=
  (n = 6 ∧ true) ∨ (n = 2004 ∧ false)

theorem king_plan_feasibility :
  ∀ n : ℕ, feasible n :=
by
  intro n
  sorry

end NUMINAMATH_GPT_king_plan_feasibility_l422_42298


namespace NUMINAMATH_GPT_a_is_zero_l422_42231

theorem a_is_zero (a b : ℤ)
  (h : ∀ n : ℕ, ∃ x : ℤ, a * 2013^n + b = x^2) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_a_is_zero_l422_42231


namespace NUMINAMATH_GPT_clea_total_time_l422_42264

-- Definitions based on conditions given
def walking_time_on_stationary (x y : ℝ) (h1 : 80 * x = y) : ℝ :=
  80

def walking_time_on_moving (x y : ℝ) (k : ℝ) (h2 : 32 * (x + k) = y) : ℝ :=
  32

def escalator_speed (x k : ℝ) (h3 : k = 1.5 * x) : ℝ :=
  1.5 * x

-- The actual theorem based on the question
theorem clea_total_time 
  (x y k : ℝ)
  (h1 : 80 * x = y)
  (h2 : 32 * (x + k) = y)
  (h3 : k = 1.5 * x) :
  let t1 := y / (2 * x)
  let t2 := y / (3 * x)
  t1 + t2 = 200 / 3 :=
by
  sorry

end NUMINAMATH_GPT_clea_total_time_l422_42264


namespace NUMINAMATH_GPT_positive_integer_solution_l422_42219

theorem positive_integer_solution (x : Int) (h_pos : x > 0) (h_cond : x + 1000 > 1000 * x) : x = 2 :=
sorry

end NUMINAMATH_GPT_positive_integer_solution_l422_42219


namespace NUMINAMATH_GPT_lunch_cost_before_tip_l422_42211

theorem lunch_cost_before_tip (C : ℝ) (h : C + 0.20 * C = 60.24) : C = 50.20 :=
sorry

end NUMINAMATH_GPT_lunch_cost_before_tip_l422_42211


namespace NUMINAMATH_GPT_power_difference_l422_42221

theorem power_difference (x : ℝ) (hx : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end NUMINAMATH_GPT_power_difference_l422_42221


namespace NUMINAMATH_GPT_total_payment_correct_l422_42216

theorem total_payment_correct 
  (bob_bill : ℝ) 
  (kate_bill : ℝ) 
  (bob_discount_rate : ℝ) 
  (kate_discount_rate : ℝ) 
  (bob_discount : ℝ := bob_bill * bob_discount_rate / 100) 
  (kate_discount : ℝ := kate_bill * kate_discount_rate / 100) 
  (bob_final_payment : ℝ := bob_bill - bob_discount) 
  (kate_final_payment : ℝ := kate_bill - kate_discount) : 
  (bob_bill = 30) → 
  (kate_bill = 25) → 
  (bob_discount_rate = 5) → 
  (kate_discount_rate = 2) → 
  (bob_final_payment + kate_final_payment = 53) :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_payment_correct_l422_42216


namespace NUMINAMATH_GPT_kids_played_on_tuesday_l422_42213

-- Define the total number of kids Julia played with
def total_kids : ℕ := 18

-- Define the number of kids Julia played with on Monday
def monday_kids : ℕ := 4

-- Define the number of kids Julia played with on Tuesday
def tuesday_kids : ℕ := total_kids - monday_kids

-- The proof goal:
theorem kids_played_on_tuesday : tuesday_kids = 14 :=
by sorry

end NUMINAMATH_GPT_kids_played_on_tuesday_l422_42213


namespace NUMINAMATH_GPT_alex_score_l422_42234

theorem alex_score (initial_students : ℕ) (initial_average : ℕ) (total_students : ℕ) (new_average : ℕ) (initial_total : ℕ) (new_total : ℕ) :
  initial_students = 19 →
  initial_average = 76 →
  total_students = 20 →
  new_average = 78 →
  initial_total = initial_students * initial_average →
  new_total = total_students * new_average →
  new_total - initial_total = 116 :=
by
  sorry

end NUMINAMATH_GPT_alex_score_l422_42234


namespace NUMINAMATH_GPT_simplify_expression_l422_42263

theorem simplify_expression (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : a + c > b) :
  |a + b - c| - |b - a - c| = 2 * b - 2 * c :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l422_42263
