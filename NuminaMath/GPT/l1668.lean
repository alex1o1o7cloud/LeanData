import Mathlib

namespace NUMINAMATH_GPT_value_of_f_at_3_l1668_166875

def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem value_of_f_at_3 : f 3 = 15 :=
by
  -- This proof needs to be filled in
  sorry

end NUMINAMATH_GPT_value_of_f_at_3_l1668_166875


namespace NUMINAMATH_GPT_minimum_flour_cost_l1668_166828

-- Definitions based on conditions provided
def loaves : ℕ := 12
def flour_per_loaf : ℕ := 4
def flour_needed : ℕ := loaves * flour_per_loaf

def ten_pound_bag_weight : ℕ := 10
def ten_pound_bag_cost : ℕ := 10

def twelve_pound_bag_weight : ℕ := 12
def twelve_pound_bag_cost : ℕ := 13

def cost_10_pound_bags : ℕ := (flour_needed + ten_pound_bag_weight - 1) / ten_pound_bag_weight * ten_pound_bag_cost
def cost_12_pound_bags : ℕ := (flour_needed + twelve_pound_bag_weight - 1) / twelve_pound_bag_weight * twelve_pound_bag_cost

theorem minimum_flour_cost : min cost_10_pound_bags cost_12_pound_bags = 50 := by
  sorry

end NUMINAMATH_GPT_minimum_flour_cost_l1668_166828


namespace NUMINAMATH_GPT_c_work_rate_l1668_166827

theorem c_work_rate (x : ℝ) : 
  (1 / 7 + 1 / 14 + 1 / x = 1 / 4) → x = 28 :=
by
  sorry

end NUMINAMATH_GPT_c_work_rate_l1668_166827


namespace NUMINAMATH_GPT_quadratic_ineq_solutions_l1668_166863

theorem quadratic_ineq_solutions (c : ℝ) (h : c > 0) : c < 16 ↔ ∀ x : ℝ, x^2 - 8 * x + c < 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x = x1 ∨ x = x2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_ineq_solutions_l1668_166863


namespace NUMINAMATH_GPT_candy_bar_cost_correct_l1668_166822

def initial_amount : ℕ := 4
def remaining_amount : ℕ := 3
def candy_bar_cost : ℕ := initial_amount - remaining_amount

theorem candy_bar_cost_correct : candy_bar_cost = 1 := by
  unfold candy_bar_cost
  sorry

end NUMINAMATH_GPT_candy_bar_cost_correct_l1668_166822


namespace NUMINAMATH_GPT_factory_profit_l1668_166823

def cost_per_unit : ℝ := 2.00
def fixed_cost : ℝ := 500.00
def selling_price_per_unit : ℝ := 2.50

theorem factory_profit (x : ℕ) (hx : x > 1000) :
  selling_price_per_unit * x > fixed_cost + cost_per_unit * x :=
by
  sorry

end NUMINAMATH_GPT_factory_profit_l1668_166823


namespace NUMINAMATH_GPT_brian_total_video_length_l1668_166891

theorem brian_total_video_length :
  let cat_length := 4
  let dog_length := 2 * cat_length
  let gorilla_length := cat_length ^ 2
  let elephant_length := cat_length + dog_length + gorilla_length
  let cat_dog_gorilla_elephant_sum := cat_length + dog_length + gorilla_length + elephant_length
  let penguin_length := cat_dog_gorilla_elephant_sum ^ 3
  let dolphin_length := cat_length + dog_length + gorilla_length + elephant_length + penguin_length
  let total_length := cat_length + dog_length + gorilla_length + elephant_length + penguin_length + dolphin_length
  total_length = 351344 := by
    sorry

end NUMINAMATH_GPT_brian_total_video_length_l1668_166891


namespace NUMINAMATH_GPT_determine_k_l1668_166856

noncomputable def f (x k : ℝ) : ℝ := -4 * x^3 + k * x

theorem determine_k : ∀ k : ℝ, (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x k ≤ 1) → k = 3 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_l1668_166856


namespace NUMINAMATH_GPT_min_troublemakers_l1668_166866

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end NUMINAMATH_GPT_min_troublemakers_l1668_166866


namespace NUMINAMATH_GPT_sqrt_expression_l1668_166806

theorem sqrt_expression :
  (Real.sqrt (2 ^ 4 * 3 ^ 6 * 5 ^ 2)) = 540 := sorry

end NUMINAMATH_GPT_sqrt_expression_l1668_166806


namespace NUMINAMATH_GPT_Abby_sits_in_seat_3_l1668_166890

theorem Abby_sits_in_seat_3:
  ∃ (positions : Fin 5 → String),
  (positions 3 = "Abby") ∧
  (positions 4 = "Bret") ∧
  ¬ ((positions 3 = "Dana") ∨ (positions 5 = "Dana")) ∧
  ¬ ((positions 2 = "Erin") ∧ (positions 3 = "Carl") ∨
    (positions 3 = "Erin") ∧ (positions 5 = "Carl")) :=
  sorry

end NUMINAMATH_GPT_Abby_sits_in_seat_3_l1668_166890


namespace NUMINAMATH_GPT_remainder_of_2n_l1668_166816

theorem remainder_of_2n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := 
sorry

end NUMINAMATH_GPT_remainder_of_2n_l1668_166816


namespace NUMINAMATH_GPT_real_solution_l1668_166800

noncomputable def condition_1 (x : ℝ) : Prop := 
  4 ≤ x / (2 * x - 7)

noncomputable def condition_2 (x : ℝ) : Prop := 
  x / (2 * x - 7) < 10

noncomputable def solution_set : Set ℝ :=
  { x | (70 / 19 : ℝ) < x ∧ x ≤ 4 }

theorem real_solution (x : ℝ) : 
  (condition_1 x ∧ condition_2 x) ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_GPT_real_solution_l1668_166800


namespace NUMINAMATH_GPT_bus_stoppage_time_l1668_166859

theorem bus_stoppage_time
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (reduction_in_speed : speed_excluding_stoppages - speed_including_stoppages = 8) :
  ∃ t : ℝ, t = 9.6 := 
sorry

end NUMINAMATH_GPT_bus_stoppage_time_l1668_166859


namespace NUMINAMATH_GPT_remainder_of_sum_l1668_166842

theorem remainder_of_sum (n1 n2 n3 n4 n5 : ℕ) (h1 : n1 = 7145) (h2 : n2 = 7146)
  (h3 : n3 = 7147) (h4 : n4 = 7148) (h5 : n5 = 7149) :
  ((n1 + n2 + n3 + n4 + n5) % 8) = 7 :=
by sorry

end NUMINAMATH_GPT_remainder_of_sum_l1668_166842


namespace NUMINAMATH_GPT_total_coins_l1668_166886

theorem total_coins (total_value : ℕ) (value_2_coins : ℕ) (num_2_coins : ℕ) (num_1_coins : ℕ) : 
  total_value = 402 ∧ value_2_coins = 2 * num_2_coins ∧ num_2_coins = 148 ∧ total_value = value_2_coins + num_1_coins →
  num_1_coins + num_2_coins = 254 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_total_coins_l1668_166886


namespace NUMINAMATH_GPT_part1_part2_l1668_166803

noncomputable def f (x a : ℝ) : ℝ := abs (x + 2 * a) + abs (x - 1)

noncomputable def g (a : ℝ) : ℝ := abs ((1 : ℝ) / a + 2 * a) + abs ((1 : ℝ) / a - 1)

theorem part1 (x : ℝ) : f x 1 ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := by
  sorry

theorem part2 (a : ℝ) (h : a ≠ 0) : g a ≤ 4 ↔ (1 / 2) ≤ a ∧ a ≤ (3 / 2) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1668_166803


namespace NUMINAMATH_GPT_sin_alpha_value_l1668_166811

open Real


theorem sin_alpha_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : π / 2 < β ∧ β < π)
  (h_sin_alpha_beta : sin (α + β) = 3 / 5) (h_cos_beta : cos β = -5 / 13) :
  sin α = 33 / 65 := 
by
  sorry

end NUMINAMATH_GPT_sin_alpha_value_l1668_166811


namespace NUMINAMATH_GPT_races_to_champion_l1668_166839

theorem races_to_champion (num_sprinters : ℕ) (sprinters_per_race : ℕ) (advancing_per_race : ℕ)
  (eliminated_per_race : ℕ) (initial_races : ℕ) (total_races : ℕ):
  num_sprinters = 360 ∧ sprinters_per_race = 8 ∧ advancing_per_race = 2 ∧ 
  eliminated_per_race = 6 ∧ initial_races = 45 ∧ total_races = 62 →
  initial_races + (initial_races / sprinters_per_race +
  ((initial_races / sprinters_per_race) / sprinters_per_race +
  (((initial_races / sprinters_per_race) / sprinters_per_race) / sprinters_per_race + 1))) = total_races :=
sorry

end NUMINAMATH_GPT_races_to_champion_l1668_166839


namespace NUMINAMATH_GPT_georgia_black_buttons_l1668_166896

theorem georgia_black_buttons : 
  ∀ (B : ℕ), 
  (4 + B + 3 = 9) → 
  B = 2 :=
by
  introv h
  linarith

end NUMINAMATH_GPT_georgia_black_buttons_l1668_166896


namespace NUMINAMATH_GPT_sum_of_exterior_angles_of_convex_quadrilateral_l1668_166880

theorem sum_of_exterior_angles_of_convex_quadrilateral:
  ∀ (α β γ δ : ℝ),
  (α + β + γ + δ = 360) → 
  (∀ (θ₁ θ₂ θ₃ θ₄ : ℝ),
    (θ₁ = 180 - α ∧ θ₂ = 180 - β ∧ θ₃ = 180 - γ ∧ θ₄ = 180 - δ) → 
    θ₁ + θ₂ + θ₃ + θ₄ = 360) := 
by 
  intros α β γ δ h1 θ₁ θ₂ θ₃ θ₄ h2
  rcases h2 with ⟨hα, hβ, hγ, hδ⟩
  rw [hα, hβ, hγ, hδ]
  linarith

end NUMINAMATH_GPT_sum_of_exterior_angles_of_convex_quadrilateral_l1668_166880


namespace NUMINAMATH_GPT_rectangle_length_l1668_166851

theorem rectangle_length (P L B : ℕ) (hP : P = 500) (hB : B = 100) (hP_eq : P = 2 * (L + B)) : L = 150 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l1668_166851


namespace NUMINAMATH_GPT_gcd_of_repeated_three_digit_integers_is_1001001_l1668_166894

theorem gcd_of_repeated_three_digit_integers_is_1001001 :
  ∀ (n : ℕ), (100 ≤ n ∧ n <= 999) →
  ∃ d : ℕ, d = 1001001 ∧
    (∀ m : ℕ, m = n * 1001001 →
      ∃ k : ℕ, m = k * d) :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_repeated_three_digit_integers_is_1001001_l1668_166894


namespace NUMINAMATH_GPT_simplify_expression_l1668_166843

theorem simplify_expression (x : ℝ) : 5 * x + 6 - x + 12 = 4 * x + 18 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1668_166843


namespace NUMINAMATH_GPT_no_solution_x_to_2n_plus_y_to_2n_eq_z_sq_l1668_166837

theorem no_solution_x_to_2n_plus_y_to_2n_eq_z_sq (n : ℕ) (h : ∀ (x y z : ℕ), x^n + y^n ≠ z^n) : ∀ (x y z : ℕ), x^(2*n) + y^(2*n) ≠ z^2 :=
by 
  intro x y z
  sorry

end NUMINAMATH_GPT_no_solution_x_to_2n_plus_y_to_2n_eq_z_sq_l1668_166837


namespace NUMINAMATH_GPT_fraction_of_second_eq_fifth_of_first_l1668_166835

theorem fraction_of_second_eq_fifth_of_first 
  (a b x y : ℕ)
  (h1 : y = 40)
  (h2 : x + 35 = 4 * y)
  (h3 : (1 / 5) * x = (a / b) * y) 
  (hb : b ≠ 0):
  a / b = 5 / 8 := by
  sorry

end NUMINAMATH_GPT_fraction_of_second_eq_fifth_of_first_l1668_166835


namespace NUMINAMATH_GPT_Joann_lollipop_theorem_l1668_166870

noncomputable def Joann_lollipops (a : ℝ) : ℝ := a + 9

theorem Joann_lollipop_theorem (a : ℝ) (total_lollipops : ℝ) 
  (h1 : a + (a + 3) + (a + 6) + (a + 9) + (a + 12) + (a + 15) = 150) 
  (h2 : total_lollipops = 150) : 
  Joann_lollipops a = 26.5 :=
by
  sorry

end NUMINAMATH_GPT_Joann_lollipop_theorem_l1668_166870


namespace NUMINAMATH_GPT_expansion_coefficient_l1668_166845

theorem expansion_coefficient (x : ℝ) (h : x ≠ 0): 
  (∃ r : ℕ, (7 - (3 / 2 : ℝ) * r = 1) ∧ Nat.choose 7 r = 35) := 
  sorry

end NUMINAMATH_GPT_expansion_coefficient_l1668_166845


namespace NUMINAMATH_GPT_negation_example_l1668_166805

theorem negation_example :
  ¬ (∀ n : ℕ, (n^2 + n) % 2 = 0) ↔ ∃ n : ℕ, (n^2 + n) % 2 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l1668_166805


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1668_166814

theorem simplify_and_evaluate_expression (m n : ℤ) (h_m : m = -1) (h_n : n = 2) :
  3 * m^2 * n - 2 * m * n^2 - 4 * m^2 * n + m * n^2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1668_166814


namespace NUMINAMATH_GPT_shortest_side_length_l1668_166869

theorem shortest_side_length (perimeter : ℝ) (shortest : ℝ) (side1 side2 side3 : ℝ) 
  (h1 : side1 + side2 + side3 = perimeter)
  (h2 : side1 = 2 * shortest)
  (h3 : side2 = 2 * shortest) :
  shortest = 3 := by
  sorry

end NUMINAMATH_GPT_shortest_side_length_l1668_166869


namespace NUMINAMATH_GPT_find_m_l1668_166882

theorem find_m (m : ℝ) (h : (4 * (-1)^3 + 3 * m * (-1)^2 + 6 * (-1) = 2)) :
  m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1668_166882


namespace NUMINAMATH_GPT_domain_all_real_iff_l1668_166893

theorem domain_all_real_iff (k : ℝ) :
  (∀ x : ℝ, -3 * x ^ 2 - x + k ≠ 0 ) ↔ k < -1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_domain_all_real_iff_l1668_166893


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1668_166850

theorem sum_of_two_numbers (a b S : ℤ) (h : a + b = S) : 
  3 * (a + 5) + 3 * (b + 7) = 3 * S + 36 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1668_166850


namespace NUMINAMATH_GPT_min_value_x_4_over_x_min_value_x_4_over_x_eq_l1668_166820

theorem min_value_x_4_over_x (x : ℝ) (h : x > 0) : x + 4 / x ≥ 4 :=
sorry

theorem min_value_x_4_over_x_eq (x : ℝ) (h : x > 0) : (x + 4 / x = 4) ↔ (x = 2) :=
sorry

end NUMINAMATH_GPT_min_value_x_4_over_x_min_value_x_4_over_x_eq_l1668_166820


namespace NUMINAMATH_GPT_max_S_possible_l1668_166871

theorem max_S_possible (nums : List ℝ) (h_nums_in_bound : ∀ n ∈ nums, 0 ≤ n ∧ n ≤ 1) (h_sum_leq_253_div_12 : nums.sum ≤ 253 / 12) :
  ∃ (A B : List ℝ), (∀ x ∈ A, x ∈ nums) ∧ (∀ y ∈ B, y ∈ nums) ∧ A.union B = nums ∧ A.sum ≤ 11 ∧ B.sum ≤ 11 :=
sorry

end NUMINAMATH_GPT_max_S_possible_l1668_166871


namespace NUMINAMATH_GPT_max_angle_in_hexagon_l1668_166812

-- Definition of the problem
theorem max_angle_in_hexagon :
  ∃ (a d : ℕ), a + a + d + a + 2*d + a + 3*d + a + 4*d + a + 5*d = 720 ∧ 
               a + 5 * d < 180 ∧ 
               (∀ a d : ℕ, a + a + d + a + 2*d + a + 3*d + a + 4*d + a + 5*d = 720 → 
               a + 5*d < 180 → m <= 175) :=
sorry

end NUMINAMATH_GPT_max_angle_in_hexagon_l1668_166812


namespace NUMINAMATH_GPT_least_integer_value_l1668_166867

theorem least_integer_value 
  (x : ℤ) (h : |3 * x - 5| ≤ 22) : x = -5 ↔ ∃ (k : ℤ), k = -5 ∧ |3 * k - 5| ≤ 22 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_value_l1668_166867


namespace NUMINAMATH_GPT_joe_speed_l1668_166878

theorem joe_speed (P : ℝ) (J : ℝ) (h1 : J = 2 * P) (h2 : 2 * P * (2 / 3) + P * (2 / 3) = 16) : J = 16 := 
by
  sorry

end NUMINAMATH_GPT_joe_speed_l1668_166878


namespace NUMINAMATH_GPT_intersection_A_B_l1668_166831

def A : Set ℝ := {x | x * (x - 4) < 0}
def B : Set ℝ := {0, 1, 5}

theorem intersection_A_B : (A ∩ B) = {1} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1668_166831


namespace NUMINAMATH_GPT_bacon_calories_percentage_l1668_166847

-- Mathematical statement based on the problem
theorem bacon_calories_percentage :
  ∀ (total_sandwich_calories : ℕ) (number_of_bacon_strips : ℕ) (calories_per_strip : ℕ),
    total_sandwich_calories = 1250 →
    number_of_bacon_strips = 2 →
    calories_per_strip = 125 →
    (number_of_bacon_strips * calories_per_strip) * 100 / total_sandwich_calories = 20 :=
by
  intros total_sandwich_calories number_of_bacon_strips calories_per_strip h1 h2 h3 
  sorry

end NUMINAMATH_GPT_bacon_calories_percentage_l1668_166847


namespace NUMINAMATH_GPT_longer_segment_of_triangle_l1668_166815

theorem longer_segment_of_triangle {a b c : ℝ} (h_triangle : a = 40 ∧ b = 90 ∧ c = 100) (h_altitude : ∃ h, h > 0) : 
  ∃ (longer_segment : ℝ), longer_segment = 82.5 :=
by 
  sorry

end NUMINAMATH_GPT_longer_segment_of_triangle_l1668_166815


namespace NUMINAMATH_GPT_exists_quadratic_function_l1668_166889

theorem exists_quadratic_function :
  (∃ (a b c : ℝ), ∀ (k : ℕ), k > 0 → (a * (5 / 9 * (10^k - 1))^2 + b * (5 / 9 * (10^k - 1)) + c = 5/9 * (10^(2*k) - 1))) :=
by
  have a := 9 / 5
  have b := 2
  have c := 0
  use a, b, c
  intros k hk
  sorry

end NUMINAMATH_GPT_exists_quadratic_function_l1668_166889


namespace NUMINAMATH_GPT_area_increase_l1668_166873

theorem area_increase (original_length original_width new_length : ℝ)
  (h1 : original_length = 20)
  (h2 : original_width = 5)
  (h3 : new_length = original_length + 10) :
  (new_length * original_width - original_length * original_width) = 50 := by
  sorry

end NUMINAMATH_GPT_area_increase_l1668_166873


namespace NUMINAMATH_GPT_find_f_find_g_l1668_166801

-- Problem 1: Finding f(x) given f(x+1) = x^2 - 2x
theorem find_f (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 2 * x) :
  ∀ x, f x = x^2 - 4 * x + 3 :=
sorry

-- Problem 2: Finding g(x) given roots and a point
theorem find_g (g : ℝ → ℝ) (h1 : g (-2) = 0) (h2 : g 3 = 0) (h3 : g 0 = -3) :
  ∀ x, g x = (1 / 2) * x^2 - (1 / 2) * x - 3 :=
sorry

end NUMINAMATH_GPT_find_f_find_g_l1668_166801


namespace NUMINAMATH_GPT_quadratic_roots_vieta_l1668_166830

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end NUMINAMATH_GPT_quadratic_roots_vieta_l1668_166830


namespace NUMINAMATH_GPT_arc_length_correct_l1668_166887

noncomputable def radius : ℝ :=
  5

noncomputable def area_of_sector : ℝ :=
  8.75

noncomputable def arc_length (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * 2 * Real.pi * r

theorem arc_length_correct :
  ∃ θ, arc_length θ radius = 3.5 ∧ (θ / 360) * Real.pi * radius^2 = area_of_sector :=
by
  sorry

end NUMINAMATH_GPT_arc_length_correct_l1668_166887


namespace NUMINAMATH_GPT_time_boarding_in_London_l1668_166846

open Nat

def time_in_ET_to_London_time (time_et: ℕ) : ℕ :=
  (time_et + 5) % 24

def subtract_hours (time: ℕ) (hours: ℕ) : ℕ :=
  (time + 24 * (hours / 24) - (hours % 24)) % 24

theorem time_boarding_in_London :
  let cape_town_arrival_time_et := 10
  let flight_duration_ny_to_cape := 10
  let ny_departure_time := subtract_hours cape_town_arrival_time_et flight_duration_ny_to_cape
  let flight_duration_london_to_ny := 18
  let ny_arrival_time := subtract_hours ny_departure_time flight_duration_london_to_ny
  let london_time := time_in_ET_to_London_time ny_arrival_time
  let london_departure_time := subtract_hours london_time flight_duration_london_to_ny
  london_departure_time = 17 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_time_boarding_in_London_l1668_166846


namespace NUMINAMATH_GPT_bumper_car_rides_l1668_166832

-- Define the conditions
def rides_on_ferris_wheel : ℕ := 7
def cost_per_ride : ℕ := 5
def total_tickets : ℕ := 50

-- Formulate the statement to be proved
theorem bumper_car_rides : ∃ n : ℕ, 
  total_tickets = (rides_on_ferris_wheel * cost_per_ride) + (n * cost_per_ride) ∧ n = 3 :=
sorry

end NUMINAMATH_GPT_bumper_car_rides_l1668_166832


namespace NUMINAMATH_GPT_delta_five_three_l1668_166897

def Δ (a b : ℕ) : ℕ := 4 * a - 6 * b

theorem delta_five_three :
  Δ 5 3 = 2 := by
  sorry

end NUMINAMATH_GPT_delta_five_three_l1668_166897


namespace NUMINAMATH_GPT_shaded_area_of_octagon_l1668_166876

noncomputable def areaOfShadedRegion (s : ℝ) (r : ℝ) (theta : ℝ) : ℝ :=
  let n := 8
  let octagonArea := n * 0.5 * s^2 * (Real.sin (Real.pi/n) / Real.sin (Real.pi/(2 * n)))
  let sectorArea := n * 0.5 * r^2 * (theta / (2 * Real.pi))
  octagonArea - sectorArea

theorem shaded_area_of_octagon (h_s : 5 = 5) (h_r : 3 = 3) (h_theta : 45 = 45) :
  areaOfShadedRegion 5 3 (45 * (Real.pi / 180)) = 100 - 9 * Real.pi := by
  sorry

end NUMINAMATH_GPT_shaded_area_of_octagon_l1668_166876


namespace NUMINAMATH_GPT_no_linear_factor_with_integer_coefficients_l1668_166864

def expression (x y z : ℤ) : ℤ :=
  x^2 - y^2 - z^2 + 3 * y * z + x + 2 * y - z

theorem no_linear_factor_with_integer_coefficients:
  ¬ ∃ (a b c d : ℤ), a ≠ 0 ∧ 
                      ∀ (x y z : ℤ), 
                        expression x y z = a * x + b * y + c * z + d := by
  sorry

end NUMINAMATH_GPT_no_linear_factor_with_integer_coefficients_l1668_166864


namespace NUMINAMATH_GPT_inequality_holds_iff_x_in_interval_l1668_166807

theorem inequality_holds_iff_x_in_interval (x : ℝ) :
  (∀ n : ℕ, 0 < n → (1 + x)^n ≤ 1 + (2^n - 1) * x) ↔ (0 ≤ x ∧ x ≤ 1) :=
sorry

end NUMINAMATH_GPT_inequality_holds_iff_x_in_interval_l1668_166807


namespace NUMINAMATH_GPT_find_b_l1668_166810

theorem find_b (a b : ℤ) (h1 : 3 * a + 1 = 1) (h2 : b - a = 2) : b = 2 := by
  sorry

end NUMINAMATH_GPT_find_b_l1668_166810


namespace NUMINAMATH_GPT_part1_part2_l1668_166860

open Real

-- Condition: tan(alpha) = 3
variable {α : ℝ} (h : tan α = 3)

-- Proof of first part
theorem part1 : (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 :=
by
  sorry

-- Proof of second part
theorem part2 : 1 - 4 * sin α * cos α + 2 * cos α ^ 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1668_166860


namespace NUMINAMATH_GPT_find_x_l1668_166852

theorem find_x (x : ℚ) : x * 9999 = 724827405 → x = 72492.75 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1668_166852


namespace NUMINAMATH_GPT_remainder_of_difference_divided_by_prime_l1668_166895

def largest_three_digit_number : ℕ := 999
def smallest_five_digit_number : ℕ := 10000
def smallest_prime_greater_than_1000 : ℕ := 1009

theorem remainder_of_difference_divided_by_prime :
  (smallest_five_digit_number - largest_three_digit_number) % smallest_prime_greater_than_1000 = 945 :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_remainder_of_difference_divided_by_prime_l1668_166895


namespace NUMINAMATH_GPT_cos_square_theta_plus_pi_over_4_eq_one_fourth_l1668_166848

variable (θ : ℝ)

theorem cos_square_theta_plus_pi_over_4_eq_one_fourth
  (h : Real.tan θ + 1 / Real.tan θ = 4) :
  Real.cos (θ + Real.pi / 4) ^ 2 = 1 / 4 :=
sorry

end NUMINAMATH_GPT_cos_square_theta_plus_pi_over_4_eq_one_fourth_l1668_166848


namespace NUMINAMATH_GPT_cost_price_article_l1668_166883

variable (SP : ℝ := 21000)
variable (d : ℝ := 0.10)
variable (p : ℝ := 0.08)

theorem cost_price_article : (SP * (1 - d)) / (1 + p) = 17500 := by
  sorry

end NUMINAMATH_GPT_cost_price_article_l1668_166883


namespace NUMINAMATH_GPT_expected_heads_value_in_cents_l1668_166854

open ProbabilityTheory

-- Define the coins and their respective values
def penny_value := 1
def nickel_value := 5
def half_dollar_value := 50
def dollar_value := 100

-- Define the probability of landing heads for each coin
def heads_prob := 1 / 2

-- Define the expected value function
noncomputable def expected_value_of_heads : ℝ :=
  heads_prob * (penny_value + nickel_value + half_dollar_value + dollar_value)

theorem expected_heads_value_in_cents : expected_value_of_heads = 78 := by
  sorry

end NUMINAMATH_GPT_expected_heads_value_in_cents_l1668_166854


namespace NUMINAMATH_GPT_prism_volume_l1668_166855

noncomputable def volume_of_prism (l w h : ℝ) : ℝ :=
l * w * h

theorem prism_volume (l w h : ℝ) (h1 : l = 2 * w) (h2 : l * w = 10) (h3 : w * h = 18) (h4 : l * h = 36) :
  volume_of_prism l w h = 36 * Real.sqrt 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_prism_volume_l1668_166855


namespace NUMINAMATH_GPT_largest_difference_l1668_166872

noncomputable def A := 3 * (2010: ℕ) ^ 2011
noncomputable def B := (2010: ℕ) ^ 2011
noncomputable def C := 2009 * (2010: ℕ) ^ 2010
noncomputable def D := 3 * (2010: ℕ) ^ 2010
noncomputable def E := (2010: ℕ) ^ 2010
noncomputable def F := (2010: ℕ) ^ 2009

theorem largest_difference :
  (A - B) > (B - C) ∧ (A - B) > (C - D) ∧ (A - B) > (D - E) ∧ (A - B) > (E - F) :=
by
  sorry

end NUMINAMATH_GPT_largest_difference_l1668_166872


namespace NUMINAMATH_GPT_mice_path_count_l1668_166824

theorem mice_path_count
  (x y : ℕ)
  (left_house_yesterday top_house_yesterday right_house_yesterday : ℕ)
  (left_house_today top_house_today right_house_today : ℕ)
  (h_left_yesterday : left_house_yesterday = 8)
  (h_top_yesterday : top_house_yesterday = 4)
  (h_right_yesterday : right_house_yesterday = 7)
  (h_left_today : left_house_today = 4)
  (h_top_today : top_house_today = 4)
  (h_right_today : right_house_today = 7)
  (h_eq : (left_house_yesterday - left_house_today) + 
          (right_house_yesterday - right_house_today) = 
          top_house_today - top_house_yesterday) :
  x + y = 11 :=
by
  sorry

end NUMINAMATH_GPT_mice_path_count_l1668_166824


namespace NUMINAMATH_GPT_periodic_odd_function_value_at_7_l1668_166892

noncomputable def f : ℝ → ℝ := sorry -- Need to define f appropriately, skipped for brevity

theorem periodic_odd_function_value_at_7
    (f_odd : ∀ x : ℝ, f (-x) = -f x)
    (f_periodic : ∀ x : ℝ, f (x + 4) = f x)
    (f_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x) :
    f 7 = -1 := sorry

end NUMINAMATH_GPT_periodic_odd_function_value_at_7_l1668_166892


namespace NUMINAMATH_GPT_least_number_to_add_l1668_166885

theorem least_number_to_add {n : ℕ} (h : n = 1202) : (∃ k : ℕ, (n + k) % 4 = 0 ∧ ∀ m : ℕ, (m < k → (n + m) % 4 ≠ 0)) ∧ k = 2 := by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l1668_166885


namespace NUMINAMATH_GPT_exists_zero_point_in_interval_l1668_166898

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x - 2 * x

theorem exists_zero_point_in_interval :
  ∃ c ∈ Set.Ioo 1 (Real.pi / 2), f c = 0 := 
sorry

end NUMINAMATH_GPT_exists_zero_point_in_interval_l1668_166898


namespace NUMINAMATH_GPT_hot_peppers_percentage_correct_l1668_166819

def sunday_peppers : ℕ := 7
def monday_peppers : ℕ := 12
def tuesday_peppers : ℕ := 14
def wednesday_peppers : ℕ := 12
def thursday_peppers : ℕ := 5
def friday_peppers : ℕ := 18
def saturday_peppers : ℕ := 12
def non_hot_peppers : ℕ := 64

def total_peppers : ℕ := sunday_peppers + monday_peppers + tuesday_peppers + wednesday_peppers + thursday_peppers + friday_peppers + saturday_peppers
def hot_peppers : ℕ := total_peppers - non_hot_peppers
def hot_peppers_percentage : ℕ := (hot_peppers * 100) / total_peppers

theorem hot_peppers_percentage_correct : hot_peppers_percentage = 20 := 
by 
  sorry

end NUMINAMATH_GPT_hot_peppers_percentage_correct_l1668_166819


namespace NUMINAMATH_GPT_people_eat_only_vegetarian_l1668_166861

def number_of_people_eat_only_veg (total_veg : ℕ) (both_veg_nonveg : ℕ) : ℕ :=
  total_veg - both_veg_nonveg

theorem people_eat_only_vegetarian
  (total_veg : ℕ) (both_veg_nonveg : ℕ)
  (h1 : total_veg = 28)
  (h2 : both_veg_nonveg = 12)
  : number_of_people_eat_only_veg total_veg both_veg_nonveg = 16 := by
  sorry

end NUMINAMATH_GPT_people_eat_only_vegetarian_l1668_166861


namespace NUMINAMATH_GPT_N_eq_M_union_P_l1668_166874

open Set

def M : Set ℝ := { x | ∃ n : ℤ, x = n }
def N : Set ℝ := { x | ∃ n : ℤ, x = n / 2 }
def P : Set ℝ := { x | ∃ n : ℤ, x = n + 1/2 }

theorem N_eq_M_union_P : N = M ∪ P := 
sorry

end NUMINAMATH_GPT_N_eq_M_union_P_l1668_166874


namespace NUMINAMATH_GPT_min_value_range_of_x_l1668_166817

variables (a b x : ℝ)

-- Problem 1: Prove the minimum value of 1/a + 4/b given a + b = 1, a > 0, b > 0
theorem min_value (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : 
  ∃ c, c = 9 ∧ ∀ y, ∃ (a b : ℝ), a + b = 1 ∧ a > 0 ∧ b > 0 → (1/a + 4/b) ≥ y :=
sorry

-- Problem 2: Prove the range of x for which 1/a + 4/b ≥ |2x - 1| - |x + 1|
theorem range_of_x (h : ∀ (a b : ℝ), a + b = 1 → a > 0 → b > 0 → (1/a + 4/b) ≥ (|2*x - 1| - |x + 1|)) :
  -7 ≤ x ∧ x ≤ 11 :=
sorry

end NUMINAMATH_GPT_min_value_range_of_x_l1668_166817


namespace NUMINAMATH_GPT_factorize_ax2_minus_a_l1668_166808

theorem factorize_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_ax2_minus_a_l1668_166808


namespace NUMINAMATH_GPT_isosceles_with_60_eq_angle_is_equilateral_l1668_166844

open Real

noncomputable def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) :=
  A = 60 ∧ B = 60 ∧ C = 60

noncomputable def is_isosceles_triangle (a b c : ℝ) (A B C : ℝ) :=
  (a = b ∨ b = c ∨ c = a) ∧ (A + B + C = 180)

theorem isosceles_with_60_eq_angle_is_equilateral
  (a b c A B C : ℝ)
  (h_iso : is_isosceles_triangle a b c A B C)
  (h_angle : A = 60 ∨ B = 60 ∨ C = 60) :
  is_equilateral_triangle a b c A B C :=
sorry

end NUMINAMATH_GPT_isosceles_with_60_eq_angle_is_equilateral_l1668_166844


namespace NUMINAMATH_GPT_hyperbola_midpoint_exists_l1668_166884

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end NUMINAMATH_GPT_hyperbola_midpoint_exists_l1668_166884


namespace NUMINAMATH_GPT_product_of_distinct_numbers_l1668_166849

theorem product_of_distinct_numbers (x y : ℝ) (h1 : x ≠ y)
  (h2 : 1 / (1 + x^2) + 1 / (1 + y^2) = 2 / (1 + x * y)) :
  x * y = 1 := 
sorry

end NUMINAMATH_GPT_product_of_distinct_numbers_l1668_166849


namespace NUMINAMATH_GPT_find_k_l1668_166877

theorem find_k (a b c k : ℝ) 
  (h : ∀ x : ℝ, 
    (a * x^2 + b * x + c + b * x^2 + a * x - 7 + k * x^2 + c * x + 3) / (x^2 - 2 * x - 5) = (x^2 - 2*x - 5)) :
  k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1668_166877


namespace NUMINAMATH_GPT_exists_arithmetic_progression_2004_perfect_powers_perfect_powers_not_infinite_arithmetic_progression_l1668_166838

-- Definition: A positive integer n is a perfect power if n = a ^ b for some integers a, b with b > 1.
def isPerfectPower (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ n = a^b

-- Part (a): Prove the existence of an arithmetic progression of 2004 perfect powers.
theorem exists_arithmetic_progression_2004_perfect_powers :
  ∃ (x r : ℕ), (∀ n : ℕ, n < 2004 → ∃ a b : ℕ, b > 1 ∧ (x + n * r) = a^b) :=
sorry

-- Part (b): Prove that perfect powers cannot form an infinite arithmetic progression.
theorem perfect_powers_not_infinite_arithmetic_progression :
  ¬ ∃ (x r : ℕ), (∀ n : ℕ, ∃ a b : ℕ, b > 1 ∧ (x + n * r) = a^b) :=
sorry

end NUMINAMATH_GPT_exists_arithmetic_progression_2004_perfect_powers_perfect_powers_not_infinite_arithmetic_progression_l1668_166838


namespace NUMINAMATH_GPT_slope_equal_angles_l1668_166829

-- Define the problem
theorem slope_equal_angles (k : ℝ) :
  (∀ (l1 l2 : ℝ), l1 = 1 ∧ l2 = 2 → (abs ((k - l1) / (1 + k * l1)) = abs ((l2 - k) / (1 + l2 * k)))) →
  (k = (1 + Real.sqrt 10) / 3 ∨ k = (1 - Real.sqrt 10) / 3) :=
by
  intros
  sorry

end NUMINAMATH_GPT_slope_equal_angles_l1668_166829


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1668_166804

theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  (1 - (1 / (x - 1))) / ((x ^ 2 - 4 * x + 4) / (x ^ 2 - 1)) = (2 / 5) :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1668_166804


namespace NUMINAMATH_GPT_composite_infinitely_many_l1668_166862

theorem composite_infinitely_many (t : ℕ) (ht : t ≥ 2) :
  ∃ n : ℕ, n = 3 ^ (2 ^ t) - 2 ^ (2 ^ t) ∧ (3 ^ (n - 1) - 2 ^ (n - 1)) % n = 0 :=
by
  use 3 ^ (2 ^ t) - 2 ^ (2 ^ t)
  sorry 

end NUMINAMATH_GPT_composite_infinitely_many_l1668_166862


namespace NUMINAMATH_GPT_billy_tickets_used_l1668_166879

-- Definitions for the number of rides and cost per ride
def ferris_wheel_rides : Nat := 7
def bumper_car_rides : Nat := 3
def ticket_per_ride : Nat := 5

-- Total number of rides
def total_rides : Nat := ferris_wheel_rides + bumper_car_rides

-- Total tickets used
def total_tickets : Nat := total_rides * ticket_per_ride

-- Theorem stating the number of tickets Billy used in total
theorem billy_tickets_used : total_tickets = 50 := by
  sorry

end NUMINAMATH_GPT_billy_tickets_used_l1668_166879


namespace NUMINAMATH_GPT_tax_percentage_l1668_166825

theorem tax_percentage (car_price tax_paid first_tier_price : ℝ) (first_tier_tax_rate : ℝ) (tax_second_tier : ℝ) :
  car_price = 30000 ∧
  tax_paid = 5500 ∧
  first_tier_price = 10000 ∧
  first_tier_tax_rate = 0.25 ∧
  tax_second_tier = 0.15
  → (tax_second_tier) = 0.15 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4, h5⟩
  sorry

end NUMINAMATH_GPT_tax_percentage_l1668_166825


namespace NUMINAMATH_GPT_matrix_subtraction_l1668_166809

-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 4, -3 ],
  ![ 2,  8 ]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 1,  5 ],
  ![ -3,  6 ]
]

-- Define the result matrix as given in the problem
def result : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 3, -8 ],
  ![ 5,  2 ]
]

-- The theorem to prove
theorem matrix_subtraction : A - B = result := 
by 
  sorry

end NUMINAMATH_GPT_matrix_subtraction_l1668_166809


namespace NUMINAMATH_GPT_find_arithmetic_progression_terms_l1668_166881

noncomputable def arithmetic_progression_terms (a1 a2 a3 : ℕ) (d : ℕ) 
  (condition1 : a1 + (a1 + d) = 3 * 2^2) 
  (condition2 : a1 + (a1 + d) + (a1 + 2 * d) = 3 * 3^2) : Prop := 
  a1 = 3 ∧ a2 = 9 ∧ a3 = 15

theorem find_arithmetic_progression_terms
  (a1 a2 a3 : ℕ) (d : ℕ)
  (cond1 : a1 + (a1 + d) = 3 * 2^2)
  (cond2 : a1 + (a1 + d) + (a1 + 2 * d) = 3 * 3^2) :
  arithmetic_progression_terms a1 a2 a3 d cond1 cond2 :=
sorry

end NUMINAMATH_GPT_find_arithmetic_progression_terms_l1668_166881


namespace NUMINAMATH_GPT_james_total_riding_time_including_rest_stop_l1668_166840

theorem james_total_riding_time_including_rest_stop :
  let distance1 := 40 -- miles
  let speed1 := 16 -- miles per hour
  let distance2 := 40 -- miles
  let speed2 := 20 -- miles per hour
  let rest_stop := 20 -- minutes
  let rest_stop_in_hours := rest_stop / 60 -- convert to hours
  let time1 := distance1 / speed1 -- time for the first part
  let time2 := distance2 / speed2 -- time for the second part
  let total_time := time1 + rest_stop_in_hours + time2 -- total time including rest
  total_time = 4.83 :=
by
  sorry

end NUMINAMATH_GPT_james_total_riding_time_including_rest_stop_l1668_166840


namespace NUMINAMATH_GPT_algorithm_output_l1668_166899

noncomputable def algorithm (x : ℝ) : ℝ :=
if x < 0 then x + 1 else -x^2

theorem algorithm_output :
  algorithm (-2) = -1 ∧ algorithm 3 = -9 :=
by
  -- proof omitted using sorry
  sorry

end NUMINAMATH_GPT_algorithm_output_l1668_166899


namespace NUMINAMATH_GPT_bob_got_15_candies_l1668_166836

-- Define the problem conditions
def bob_neighbor_sam : Prop := true -- Bob is Sam's next door neighbor
def bob_accompany_sam_home : Prop := true -- Bob decided to accompany Sam home

def bob_share_chewing_gums : ℕ := 15 -- Bob's share of chewing gums
def bob_share_chocolate_bars : ℕ := 20 -- Bob's share of chocolate bars
def bob_share_candies : ℕ := 15 -- Bob's share of assorted candies

-- Define the main assertion
theorem bob_got_15_candies : bob_share_candies = 15 := 
by sorry

end NUMINAMATH_GPT_bob_got_15_candies_l1668_166836


namespace NUMINAMATH_GPT_negation_of_square_positive_l1668_166818

open Real

-- Define the original proposition
def prop_square_positive : Prop :=
  ∀ x : ℝ, x^2 > 0

-- Define the negation of the original proposition
def prop_square_not_positive : Prop :=
  ∃ x : ℝ, ¬ (x^2 > 0)

-- The theorem that asserts the logical equivalence for the negation
theorem negation_of_square_positive :
  ¬ prop_square_positive ↔ prop_square_not_positive :=
by sorry

end NUMINAMATH_GPT_negation_of_square_positive_l1668_166818


namespace NUMINAMATH_GPT_property_holds_for_1_and_4_l1668_166826

theorem property_holds_for_1_and_4 (n : ℕ) : 
  (∀ q : ℕ, n % q^2 < q^(q^2) / 2) ↔ (n = 1 ∨ n = 4) :=
by sorry

end NUMINAMATH_GPT_property_holds_for_1_and_4_l1668_166826


namespace NUMINAMATH_GPT_remainder_div_x_plus_2_l1668_166858

def f (x : ℤ) : ℤ := x^15 + 3

theorem remainder_div_x_plus_2 : f (-2) = -32765 := by
  sorry

end NUMINAMATH_GPT_remainder_div_x_plus_2_l1668_166858


namespace NUMINAMATH_GPT_fraction_is_meaningful_l1668_166834

theorem fraction_is_meaningful (x : ℝ) : x ≠ 1 ↔ ∃ y : ℝ, y = 8 / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_meaningful_l1668_166834


namespace NUMINAMATH_GPT_exists_powers_of_7_difference_div_by_2021_l1668_166865

theorem exists_powers_of_7_difference_div_by_2021 :
  ∃ n m : ℕ, n > m ∧ 2021 ∣ (7^n - 7^m) := 
by
  sorry

end NUMINAMATH_GPT_exists_powers_of_7_difference_div_by_2021_l1668_166865


namespace NUMINAMATH_GPT_seq_integer_l1668_166857

theorem seq_integer (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 1) (h3 : a 3 = 249)
(h_rec : ∀ n, a (n + 3) = (1991 + a (n + 2) * a (n + 1)) / a n) :
∀ n, ∃ b : ℤ, a n = b :=
by
  sorry

end NUMINAMATH_GPT_seq_integer_l1668_166857


namespace NUMINAMATH_GPT_find_xy_l1668_166833

theorem find_xy (p x y : ℕ) (hp : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  p * (x - y) = x * y ↔ (x, y) = (p^2 - p, p + 1) := by
  sorry

end NUMINAMATH_GPT_find_xy_l1668_166833


namespace NUMINAMATH_GPT_reflection_across_x_axis_l1668_166802

theorem reflection_across_x_axis :
  let initial_point := (-3, 5)
  let reflected_point := (-3, -5)
  reflected_point = (initial_point.1, -initial_point.2) :=
by
  sorry

end NUMINAMATH_GPT_reflection_across_x_axis_l1668_166802


namespace NUMINAMATH_GPT_part_one_part_two_l1668_166813

-- Given that tan α = 2, prove that the following expressions are correct:

theorem part_one (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (Real.pi - α) + Real.cos (α - Real.pi / 2) - Real.cos (3 * Real.pi + α)) / 
  (Real.cos (Real.pi / 2 + α) - Real.sin (2 * Real.pi + α) + 2 * Real.sin (α - Real.pi / 2)) = 
  -5 / 6 := 
by
  -- Proof skipped
  sorry

theorem part_two (α : ℝ) (h : Real.tan α = 2) :
  Real.cos (2 * α) + Real.sin α * Real.cos α = -1 / 5 := 
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_part_one_part_two_l1668_166813


namespace NUMINAMATH_GPT_max_value_of_ab_expression_l1668_166821

noncomputable def max_ab_expression : ℝ :=
  let a := 4
  let b := 20 / 3
  a * b * (60 - 5 * a - 3 * b)

theorem max_value_of_ab_expression :
  ∀ (a b : ℝ), 0 < a → 0 < b → 5 * a + 3 * b < 60 →
  ab * (60 - 5 * a - 3 * b) ≤ max_ab_expression :=
sorry

end NUMINAMATH_GPT_max_value_of_ab_expression_l1668_166821


namespace NUMINAMATH_GPT_determine_digit_square_l1668_166853

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_palindrome (n : ℕ) : Prop :=
  let d1 := (n / 100000) % 10
  let d2 := (n / 10000) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 100) % 10
  let d5 := (n / 10) % 10
  let d6 := n % 10
  d1 = d6 ∧ d2 = d5 ∧ d3 = d4

def is_multiple_of_6 (n : ℕ) : Prop := is_even (n % 10) ∧ is_divisible_by_3 (List.sum (Nat.digits 10 n))

theorem determine_digit_square :
  ∃ (square : ℕ),
  (is_palindrome (53700000 + square * 10 + 735) ∧ is_multiple_of_6 (53700000 + square * 10 + 735)) ∧ square = 6 := by
  sorry

end NUMINAMATH_GPT_determine_digit_square_l1668_166853


namespace NUMINAMATH_GPT_triangle_area_eq_l1668_166888

noncomputable def areaOfTriangle (a b c A B C: ℝ): ℝ :=
1 / 2 * a * c * (Real.sin A)

theorem triangle_area_eq
  (a b c A B C : ℝ)
  (h1 : a = 2)
  (h2 : A = Real.pi / 3)
  (h3 : Real.sqrt 3 / 2 - Real.sin (B - C) = Real.sin (2 * B)) :
  areaOfTriangle a b c A B C = Real.sqrt 3 ∨ areaOfTriangle a b c A B C = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_eq_l1668_166888


namespace NUMINAMATH_GPT_intersection_M_N_l1668_166868

noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def N : Set ℝ := {x | abs x < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1668_166868


namespace NUMINAMATH_GPT_max_yellow_apples_can_take_max_total_apples_can_take_l1668_166841

structure Basket :=
  (total_apples : ℕ)
  (green_apples : ℕ)
  (yellow_apples : ℕ)
  (red_apples : ℕ)
  (green_lt_yellow : green_apples < yellow_apples)
  (yellow_lt_red : yellow_apples < red_apples)

def basket_conditions : Basket :=
  { total_apples := 44,
    green_apples := 11,
    yellow_apples := 14,
    red_apples := 19,
    green_lt_yellow := sorry,  -- 11 < 14
    yellow_lt_red := sorry }   -- 14 < 19

theorem max_yellow_apples_can_take : basket_conditions.yellow_apples = 14 := sorry

theorem max_total_apples_can_take : basket_conditions.green_apples 
                                     + basket_conditions.yellow_apples 
                                     + (basket_conditions.red_apples - 2) = 42 := sorry

end NUMINAMATH_GPT_max_yellow_apples_can_take_max_total_apples_can_take_l1668_166841
