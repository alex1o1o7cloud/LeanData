import Mathlib

namespace NUMINAMATH_GPT_sum_of_roots_eq_k_div_4_l2293_229356

variables {k d y_1 y_2 : ℝ}

theorem sum_of_roots_eq_k_div_4 (h1 : y_1 ≠ y_2)
                                  (h2 : 4 * y_1^2 - k * y_1 = d)
                                  (h3 : 4 * y_2^2 - k * y_2 = d) :
  y_1 + y_2 = k / 4 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_eq_k_div_4_l2293_229356


namespace NUMINAMATH_GPT_dartboard_area_ratio_l2293_229395

theorem dartboard_area_ratio
    (larger_square_side_length : ℝ)
    (inner_square_side_length : ℝ)
    (angle_division : ℝ)
    (s : ℝ)
    (p : ℝ)
    (h1 : larger_square_side_length = 4)
    (h2 : inner_square_side_length = 2)
    (h3 : angle_division = 45)
    (h4 : s = 1/4)
    (h5 : p = 3) :
    p / s = 12 :=
by
    sorry

end NUMINAMATH_GPT_dartboard_area_ratio_l2293_229395


namespace NUMINAMATH_GPT_christine_min_bottles_l2293_229377

theorem christine_min_bottles
  (fluid_ounces_needed : ℕ)
  (bottle_volume_ml : ℕ)
  (fluid_ounces_per_liter : ℝ)
  (liters_in_milliliter : ℕ)
  (required_bottles : ℕ)
  (h1 : fluid_ounces_needed = 45)
  (h2 : bottle_volume_ml = 200)
  (h3 : fluid_ounces_per_liter = 33.8)
  (h4 : liters_in_milliliter = 1000)
  (h5 : required_bottles = 7) :
  required_bottles = ⌈(fluid_ounces_needed * liters_in_milliliter) / (bottle_volume_ml * fluid_ounces_per_liter)⌉ := by
  sorry

end NUMINAMATH_GPT_christine_min_bottles_l2293_229377


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l2293_229398

-- Proof for the first equation
theorem solve_eq1 (y : ℝ) : 8 * y - 4 * (3 * y + 2) = 6 ↔ y = -7 / 2 := 
by 
  sorry

-- Proof for the second equation
theorem solve_eq2 (x : ℝ) : 2 - (x + 2) / 3 = x - (x - 1) / 6 ↔ x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l2293_229398


namespace NUMINAMATH_GPT_f_prime_at_pi_over_six_l2293_229305

noncomputable def f (f'_0 : ℝ) (x : ℝ) : ℝ := (1/2)*x^2 + 2*f'_0*(Real.cos x) + x

theorem f_prime_at_pi_over_six (f'_0 : ℝ) (h : f'_0 = 1) :
  (deriv (f f'_0)) (Real.pi / 6) = Real.pi / 6 := by
  sorry

end NUMINAMATH_GPT_f_prime_at_pi_over_six_l2293_229305


namespace NUMINAMATH_GPT_zhijie_suanjing_l2293_229340

theorem zhijie_suanjing :
  ∃ (x y: ℕ), x + y = 100 ∧ 3 * x + y / 3 = 100 :=
by
  sorry

end NUMINAMATH_GPT_zhijie_suanjing_l2293_229340


namespace NUMINAMATH_GPT_incorrect_transformation_l2293_229335

theorem incorrect_transformation (a b c : ℝ) (h1 : a = b) (h2 : c = 0) : ¬(a / c = b / c) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_transformation_l2293_229335


namespace NUMINAMATH_GPT_community_theater_ticket_sales_l2293_229364

theorem community_theater_ticket_sales (A C : ℕ) 
  (h1 : 12 * A + 4 * C = 840) 
  (h2 : A + C = 130) :
  A = 40 :=
sorry

end NUMINAMATH_GPT_community_theater_ticket_sales_l2293_229364


namespace NUMINAMATH_GPT_cuboid_diagonal_cubes_l2293_229311

def num_cubes_intersecting_diagonal (a b c : ℕ) : ℕ :=
  a + b + c - 2

theorem cuboid_diagonal_cubes :
  num_cubes_intersecting_diagonal 77 81 100 = 256 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_diagonal_cubes_l2293_229311


namespace NUMINAMATH_GPT_SoccerBallPrices_SoccerBallPurchasingPlans_l2293_229360

theorem SoccerBallPrices :
  ∃ (priceA priceB : ℕ), priceA = 100 ∧ priceB = 80 ∧ (900 / priceA) = (720 / (priceB - 20)) :=
sorry

theorem SoccerBallPurchasingPlans :
  ∃ (m n : ℕ), (m + n = 90) ∧ (m ≥ 2 * n) ∧ (100 * m + 80 * n ≤ 8500) ∧
  (m ∈ Finset.range 66 \ Finset.range 60) ∧ 
  (∀ k ∈ Finset.range 66 \ Finset.range 60, 100 * k + 80 * (90 - k) ≥ 8400) :=
sorry

end NUMINAMATH_GPT_SoccerBallPrices_SoccerBallPurchasingPlans_l2293_229360


namespace NUMINAMATH_GPT_geom_seq_inequality_l2293_229320

theorem geom_seq_inequality 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_pos : ∀ n : ℕ, a n > 0) 
  (h_q : q ≠ 1) : 
  a 1 + a 4 > a 2 + a 3 := 
sorry

end NUMINAMATH_GPT_geom_seq_inequality_l2293_229320


namespace NUMINAMATH_GPT_palindrome_count_l2293_229317

theorem palindrome_count :
  let A_choices := 9
  let B_choices := 10
  let C_choices := 10
  (A_choices * B_choices * C_choices) = 900 :=
by
  let A_choices := 9
  let B_choices := 10
  let C_choices := 10
  show (A_choices * B_choices * C_choices) = 900
  sorry

end NUMINAMATH_GPT_palindrome_count_l2293_229317


namespace NUMINAMATH_GPT_total_raisins_l2293_229355

theorem total_raisins (yellow raisins black raisins : ℝ) (h_yellow : yellow = 0.3) (h_black : black = 0.4) : yellow + black = 0.7 := 
by
  sorry

end NUMINAMATH_GPT_total_raisins_l2293_229355


namespace NUMINAMATH_GPT_icosahedron_inscribed_in_cube_l2293_229372

theorem icosahedron_inscribed_in_cube (a m : ℝ) (points_on_faces : Fin 6 → Fin 2 → ℝ × ℝ × ℝ) :
  (∃ points : Fin 12 → ℝ × ℝ × ℝ, 
   (∀ i : Fin 12, ∃ j : Fin 6, (points i).fst = (points_on_faces j 0).fst ∨ (points i).fst = (points_on_faces j 1).fst) ∧
   ∃ segments : Fin 12 → Fin 12 → ℝ, 
   (∀ i j : Fin 12, (segments i j) = m ∨ (segments i j) = a)) →
  a^2 - a*m - m^2 = 0 := sorry

end NUMINAMATH_GPT_icosahedron_inscribed_in_cube_l2293_229372


namespace NUMINAMATH_GPT_translate_point_A_l2293_229324

theorem translate_point_A :
  let A : ℝ × ℝ := (-1, 2)
  let x_translation : ℝ := 4
  let y_translation : ℝ := -2
  let A1 : ℝ × ℝ := (A.1 + x_translation, A.2 + y_translation)
  A1 = (3, 0) :=
by
  let A : ℝ × ℝ := (-1, 2)
  let x_translation : ℝ := 4
  let y_translation : ℝ := -2
  let A1 : ℝ × ℝ := (A.1 + x_translation, A.2 + y_translation)
  show A1 = (3, 0)
  sorry

end NUMINAMATH_GPT_translate_point_A_l2293_229324


namespace NUMINAMATH_GPT_packs_of_gum_bought_l2293_229309

noncomputable def initial_amount : ℝ := 10.00
noncomputable def gum_cost : ℝ := 1.00
noncomputable def choc_bars : ℝ := 5.00
noncomputable def choc_bar_cost : ℝ := 1.00
noncomputable def candy_canes : ℝ := 2.00
noncomputable def candy_cane_cost : ℝ := 0.50
noncomputable def leftover_amount : ℝ := 1.00

theorem packs_of_gum_bought : (initial_amount - leftover_amount - (choc_bars * choc_bar_cost + candy_canes * candy_cane_cost)) / gum_cost = 3 :=
by
  sorry

end NUMINAMATH_GPT_packs_of_gum_bought_l2293_229309


namespace NUMINAMATH_GPT_no_common_root_l2293_229300

theorem no_common_root (a b c d : ℝ) (ha : 0 < a) (hb : a < b) (hc : b < c) (hd : c < d) :
  ¬ ∃ x : ℝ, (x^2 + b * x + c = 0) ∧ (x^2 + a * x + d = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_common_root_l2293_229300


namespace NUMINAMATH_GPT_max_value_frac_x1_x2_et_l2293_229393

theorem max_value_frac_x1_x2_et (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x * Real.exp x)
  (hg : ∀ x, g x = - (Real.log x) / x)
  (x1 x2 t : ℝ)
  (hx1 : f x1 = t)
  (hx2 : g x2 = t)
  (ht_pos : t > 0) :
  ∃ x1 x2, (f x1 = t ∧ g x2 = t) ∧ (∀ u v, (f u = t ∧ g v = t → u / (v * Real.exp t) ≤ 1 / Real.exp 1)) :=
by
  sorry

end NUMINAMATH_GPT_max_value_frac_x1_x2_et_l2293_229393


namespace NUMINAMATH_GPT_relationship_inequality_l2293_229338

variable {a b c d : ℝ}

-- Define the conditions
def is_largest (a b c : ℝ) : Prop := a > b ∧ a > c
def positive_numbers (a b c d : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
def ratio_condition (a b c d : ℝ) : Prop := a / b = c / d

-- The theorem statement
theorem relationship_inequality 
  (h_largest : is_largest a b c)
  (h_positive : positive_numbers a b c d)
  (h_ratio : ratio_condition a b c d) :
  a + d > b + c :=
sorry

end NUMINAMATH_GPT_relationship_inequality_l2293_229338


namespace NUMINAMATH_GPT_line_segment_intersection_range_l2293_229346

theorem line_segment_intersection_range (P Q : ℝ × ℝ) (m : ℝ)
  (hP : P = (-1, 1)) (hQ : Q = (2, 2)) :
  ∃ m : ℝ, (x + m * y + m = 0) ∧ (-3 < m ∧ m < -2/3) := 
sorry

end NUMINAMATH_GPT_line_segment_intersection_range_l2293_229346


namespace NUMINAMATH_GPT_ashok_total_subjects_l2293_229327

variable (n : ℕ) (T : ℕ)

theorem ashok_total_subjects (h_ave_all : 75 * n = T + 80)
                       (h_ave_first : T = 74 * (n - 1)) :
  n = 6 := sorry

end NUMINAMATH_GPT_ashok_total_subjects_l2293_229327


namespace NUMINAMATH_GPT_length_real_axis_hyperbola_l2293_229349

theorem length_real_axis_hyperbola :
  (∃ (C : ℝ → ℝ → Prop) (a b : ℝ), (a > 0) ∧ (b > 0) ∧ 
    (∀ x y : ℝ, C x y = ((x ^ 2) / a ^ 2 - (y ^ 2) / b ^ 2 = 1)) ∧
      (∀ x y : ℝ, ((x ^ 2) / 9 - (y ^ 2) / 16 = 1) → ((x ^ 2) / a ^ 2 - (y ^ 2) / b ^ 2 = 1)) ∧
      C (-3) (2 * Real.sqrt 3)) →
  2 * (3 / 2) = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_real_axis_hyperbola_l2293_229349


namespace NUMINAMATH_GPT_num_integers_achievable_le_2014_l2293_229389

def floor_div (x : ℤ) : ℤ := x / 2

def button1 (x : ℤ) : ℤ := floor_div x

def button2 (x : ℤ) : ℤ := 4 * x + 1

def num_valid_sequences (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 2
  else num_valid_sequences (n - 1) + num_valid_sequences (n - 2)

theorem num_integers_achievable_le_2014 :
  num_valid_sequences 11 = 233 :=
  by
    -- Proof starts here
    sorry

end NUMINAMATH_GPT_num_integers_achievable_le_2014_l2293_229389


namespace NUMINAMATH_GPT_f_transform_l2293_229388

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + 4 * x - 5

theorem f_transform (x h : ℝ) : 
  f (x + h) - f x = 6 * x ^ 2 - 6 * x + 6 * x * h + 2 * h ^ 2 - 3 * h + 4 := 
by
  sorry

end NUMINAMATH_GPT_f_transform_l2293_229388


namespace NUMINAMATH_GPT_minimum_value_is_4_l2293_229354

noncomputable def minimum_value (m n : ℝ) : ℝ :=
  if h : m > 0 ∧ n > 0 ∧ m + n = 1 then (1 / m) + (1 / n) else 0

theorem minimum_value_is_4 :
  (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m + n = 1) →
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m + n = 1 ∧ minimum_value m n = 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_is_4_l2293_229354


namespace NUMINAMATH_GPT_book_difference_l2293_229318

def initial_books : ℕ := 75
def borrowed_books : ℕ := 18
def difference : ℕ := initial_books - borrowed_books

theorem book_difference : difference = 57 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_book_difference_l2293_229318


namespace NUMINAMATH_GPT_nicole_initial_candies_l2293_229319

theorem nicole_initial_candies (x : ℕ) (h1 : x / 3 + 5 + 10 = x) : x = 23 := by
  sorry

end NUMINAMATH_GPT_nicole_initial_candies_l2293_229319


namespace NUMINAMATH_GPT_sets_are_equal_l2293_229366

-- Define sets according to the given options
def option_a_M : Set (ℕ × ℕ) := {(3, 2)}
def option_a_N : Set (ℕ × ℕ) := {(2, 3)}

def option_b_M : Set ℕ := {3, 2}
def option_b_N : Set (ℕ × ℕ) := {(3, 2)}

def option_c_M : Set (ℕ × ℕ) := {(x, y) | x + y = 1}
def option_c_N : Set ℕ := { y | ∃ x, x + y = 1 }

def option_d_M : Set ℕ := {3, 2}
def option_d_N : Set ℕ := {2, 3}

-- Proof goal
theorem sets_are_equal : option_d_M = option_d_N :=
sorry

end NUMINAMATH_GPT_sets_are_equal_l2293_229366


namespace NUMINAMATH_GPT_least_possible_average_of_integers_l2293_229306

theorem least_possible_average_of_integers :
  ∃ (a b c d : ℤ), a < b ∧ b < c ∧ c < d ∧ d = 90 ∧ a ≥ 21 ∧ (a + b + c + d) / 4 = 39 := by
sorry

end NUMINAMATH_GPT_least_possible_average_of_integers_l2293_229306


namespace NUMINAMATH_GPT_inequality_proof_l2293_229352

variable (k : ℕ) (a b c : ℝ)
variables (hk : 0 < k) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem inequality_proof (hk : k > 0) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * (1 - a^k) + b * (1 - (a + b)^k) + c * (1 - (a + b + c)^k) < k / (k + 1) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2293_229352


namespace NUMINAMATH_GPT_toads_l2293_229374

theorem toads (Tim Jim Sarah : ℕ) 
  (h1 : Jim = Tim + 20) 
  (h2 : Sarah = 2 * Jim) 
  (h3 : Sarah = 100) : Tim = 30 := 
by 
  -- Proof will be provided later
  sorry

end NUMINAMATH_GPT_toads_l2293_229374


namespace NUMINAMATH_GPT_find_y_satisfies_equation_l2293_229304

theorem find_y_satisfies_equation :
  ∃ y : ℝ, 3 * y + 6 = |(-20 + 2)| :=
by
  sorry

end NUMINAMATH_GPT_find_y_satisfies_equation_l2293_229304


namespace NUMINAMATH_GPT_travel_time_difference_l2293_229376

variable (x : ℝ)

theorem travel_time_difference 
  (distance : ℝ) 
  (speed_diff : ℝ)
  (time_diff_minutes : ℝ)
  (personB_speed : ℝ) 
  (personA_speed := personB_speed - speed_diff) 
  (time_diff_hours := time_diff_minutes / 60) :
  distance = 30 ∧ speed_diff = 3 ∧ time_diff_minutes = 40 ∧ personB_speed = x → 
    (30 / (x - 3)) - (30 / x) = 40 / 60 := 
by 
  sorry

end NUMINAMATH_GPT_travel_time_difference_l2293_229376


namespace NUMINAMATH_GPT_gcd_2_l2293_229301

-- Define the two numbers obtained from the conditions.
def n : ℕ := 3589 - 23
def m : ℕ := 5273 - 41

-- State that the GCD of n and m is 2.
theorem gcd_2 : Nat.gcd n m = 2 := by
  sorry

end NUMINAMATH_GPT_gcd_2_l2293_229301


namespace NUMINAMATH_GPT_hyperbola_eccentricity_range_l2293_229341

-- Definitions of hyperbola and distance condition
def hyperbola (x y a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def distance_condition (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), hyperbola x y a b → (b * x + a * y - 2 * a * b) > a

-- The range of the eccentricity
theorem hyperbola_eccentricity_range (a b : ℝ) (h : hyperbola 0 1 a b) 
  (dist_cond : distance_condition a b) : 
  ∃ e : ℝ, e ≥ (2 * Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_range_l2293_229341


namespace NUMINAMATH_GPT_weekly_rental_cost_l2293_229325

theorem weekly_rental_cost (W : ℝ) 
  (monthly_cost : ℝ := 40)
  (months_in_year : ℝ := 12)
  (weeks_in_year : ℝ := 52)
  (savings : ℝ := 40)
  (total_year_cost_month : ℝ := months_in_year * monthly_cost)
  (total_year_cost_week : ℝ := total_year_cost_month + savings) :
  (total_year_cost_week / weeks_in_year) = 10 :=
by 
  sorry

end NUMINAMATH_GPT_weekly_rental_cost_l2293_229325


namespace NUMINAMATH_GPT_quadratic_roots_p_l2293_229369

noncomputable def equation : Type* := sorry

theorem quadratic_roots_p
  (α β : ℝ)
  (K : ℝ)
  (h1 : 3 * α ^ 2 + 7 * α + K = 0)
  (h2 : 3 * β ^ 2 + 7 * β + K = 0)
  (sum_roots : α + β = -7 / 3)
  (prod_roots : α * β = K / 3)
  : ∃ p : ℝ, p = -70 / 9 + 2 * K / 3 := 
sorry

end NUMINAMATH_GPT_quadratic_roots_p_l2293_229369


namespace NUMINAMATH_GPT_exists_seq_two_reals_l2293_229361

theorem exists_seq_two_reals (x y : ℝ) (a : ℕ → ℝ) (h_recur : ∀ n, a (n + 2) = x * a (n + 1) + y * a n) :
  (∀ r > 0, ∃ i j : ℕ, 0 < |a i| ∧ |a i| < r ∧ r < |a j|) → ∃ x y : ℝ, ∃ a : ℕ → ℝ, (∀ n, a (n + 2) = x * a (n + 1) + y * a n) :=
by
  sorry

end NUMINAMATH_GPT_exists_seq_two_reals_l2293_229361


namespace NUMINAMATH_GPT_find_packs_size_l2293_229348

theorem find_packs_size (y : ℕ) :
  (24 - 2 * y) * (36 + 4 * y) = 864 → y = 3 :=
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_packs_size_l2293_229348


namespace NUMINAMATH_GPT_total_shaded_area_l2293_229332

theorem total_shaded_area
  (carpet_side : ℝ)
  (large_square_side : ℝ)
  (small_square_side : ℝ)
  (ratio_large : carpet_side / large_square_side = 4)
  (ratio_small : large_square_side / small_square_side = 2) : 
  (1 * large_square_side^2 + 12 * small_square_side^2 = 64) := 
by 
  sorry

end NUMINAMATH_GPT_total_shaded_area_l2293_229332


namespace NUMINAMATH_GPT_base_height_is_two_inches_l2293_229302

noncomputable def height_sculpture_feet : ℝ := 2 + (10 / 12)
noncomputable def combined_height_feet : ℝ := 3
noncomputable def base_height_feet : ℝ := combined_height_feet - height_sculpture_feet
noncomputable def base_height_inches : ℝ := base_height_feet * 12

theorem base_height_is_two_inches :
  base_height_inches = 2 := by
  sorry

end NUMINAMATH_GPT_base_height_is_two_inches_l2293_229302


namespace NUMINAMATH_GPT_product_arithmetic_sequence_mod_100_l2293_229370

def is_arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ → Prop) : Prop :=
  ∀ k, n k → k = a + d * (k / d)

theorem product_arithmetic_sequence_mod_100 :
  ∀ P : ℕ,
    (∀ k, 7 ≤ k ∧ k ≤ 1999 ∧ ((k - 7) % 12 = 0) → P = k) →
    (P % 100 = 75) :=
by {
  sorry
}

end NUMINAMATH_GPT_product_arithmetic_sequence_mod_100_l2293_229370


namespace NUMINAMATH_GPT_negation_of_universal_quantification_l2293_229358

theorem negation_of_universal_quantification (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x| > 1) ↔ ∃ x ∈ S, |x| ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_quantification_l2293_229358


namespace NUMINAMATH_GPT_kira_breakfast_time_l2293_229333

theorem kira_breakfast_time (n_sausages : ℕ) (n_eggs : ℕ) (t_fry_per_sausage : ℕ) (t_scramble_per_egg : ℕ) (total_time : ℕ) :
  n_sausages = 3 → n_eggs = 6 → t_fry_per_sausage = 5 → t_scramble_per_egg = 4 → total_time = (n_sausages * t_fry_per_sausage + n_eggs * t_scramble_per_egg) →
  total_time = 39 :=
by
  intros h_sausages h_eggs h_fry h_scramble h_total
  rw [h_sausages, h_eggs, h_fry, h_scramble] at h_total
  exact h_total

end NUMINAMATH_GPT_kira_breakfast_time_l2293_229333


namespace NUMINAMATH_GPT_range_of_x_f_greater_than_4_l2293_229330

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else x^2

theorem range_of_x_f_greater_than_4 :
  { x : ℝ | f x > 4 } = { x : ℝ | x < -2 ∨ x > 2 } :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_f_greater_than_4_l2293_229330


namespace NUMINAMATH_GPT_inequality_solution_l2293_229331

theorem inequality_solution (a : ℝ) (h : a > 0) :
  (if a = 2 then {x : ℝ | false}
   else if 0 < a ∧ a < 2 then {x : ℝ | 1 < x ∧ x ≤ 2 / a}
   else if a > 2 then {x : ℝ | 2 / a ≤ x ∧ x < 1}
   else ∅) =
    {x : ℝ | (a + 2) * x - 4 ≤ 2 * (x - 1)} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2293_229331


namespace NUMINAMATH_GPT_constant_temperature_l2293_229378

def stable_system (T : ℤ × ℤ × ℤ → ℝ) : Prop :=
  ∀ (a b c : ℤ), T (a, b, c) = (1 / 6) * (T (a + 1, b, c) + T (a - 1, b, c) + T (a, b + 1, c) + T (a, b - 1, c) + T (a, b, c + 1) + T (a, b, c - 1))

theorem constant_temperature (T : ℤ × ℤ × ℤ → ℝ) 
    (h1 : ∀ (x : ℤ × ℤ × ℤ), 0 ≤ T x ∧ T x ≤ 1)
    (h2 : stable_system T) : 
  ∃ c : ℝ, ∀ x : ℤ × ℤ × ℤ, T x = c := 
sorry

end NUMINAMATH_GPT_constant_temperature_l2293_229378


namespace NUMINAMATH_GPT_odd_primes_mod_32_l2293_229313

-- Define the set of odd primes less than 2^5
def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

-- Define the product of all elements in the list
def N : ℕ := odd_primes_less_than_32.foldl (·*·) 1

-- State the theorem
theorem odd_primes_mod_32 :
  N % 32 = 9 :=
sorry

end NUMINAMATH_GPT_odd_primes_mod_32_l2293_229313


namespace NUMINAMATH_GPT_valuable_files_count_l2293_229379

theorem valuable_files_count 
    (initial_files : ℕ) 
    (deleted_fraction_initial : ℚ) 
    (additional_files : ℕ) 
    (irrelevant_fraction_additional : ℚ) 
    (h1 : initial_files = 800) 
    (h2 : deleted_fraction_initial = (70:ℚ) / 100)
    (h3 : additional_files = 400)
    (h4 : irrelevant_fraction_additional = (3:ℚ) / 5) : 
    (initial_files - ⌊deleted_fraction_initial * initial_files⌋ + additional_files - ⌊irrelevant_fraction_additional * additional_files⌋) = 400 :=
by sorry

end NUMINAMATH_GPT_valuable_files_count_l2293_229379


namespace NUMINAMATH_GPT_ab_divisible_by_six_l2293_229384

def last_digit (n : ℕ) : ℕ :=
  (2 ^ n) % 10

def b_value (n : ℕ) (a : ℕ) : ℕ :=
  2 ^ n - a

theorem ab_divisible_by_six (n : ℕ) (h : n > 3) :
  let a := last_digit n
  let b := b_value n a
  ∃ k : ℕ, ab = 6 * k :=
by
  sorry

end NUMINAMATH_GPT_ab_divisible_by_six_l2293_229384


namespace NUMINAMATH_GPT_midpoint_of_segment_l2293_229312

def A : ℝ × ℝ × ℝ := (10, -3, 5)
def B : ℝ × ℝ × ℝ := (-2, 7, -4)

theorem midpoint_of_segment :
  let M_x := (10 + -2 : ℝ) / 2
  let M_y := (-3 + 7 : ℝ) / 2
  let M_z := (5 + -4 : ℝ) / 2
  (M_x, M_y, M_z) = (4, 2, 0.5) :=
by
  let M_x : ℝ := (10 + -2) / 2
  let M_y : ℝ := (-3 + 7) / 2
  let M_z : ℝ := (5 + -4) / 2
  show (M_x, M_y, M_z) = (4, 2, 0.5)
  repeat { sorry }

end NUMINAMATH_GPT_midpoint_of_segment_l2293_229312


namespace NUMINAMATH_GPT_largest_x_value_l2293_229344

theorem largest_x_value
  (x : ℝ)
  (h : (17 * x^2 - 46 * x + 21) / (5 * x - 3) + 7 * x = 8 * x - 2)
  : x = 5 / 3 :=
sorry

end NUMINAMATH_GPT_largest_x_value_l2293_229344


namespace NUMINAMATH_GPT_converse_statement_l2293_229326

theorem converse_statement (x : ℝ) :
  x^2 + 3 * x - 2 < 0 → x < 1 :=
sorry

end NUMINAMATH_GPT_converse_statement_l2293_229326


namespace NUMINAMATH_GPT_calculateL_l2293_229307

-- Defining the constants T, H, and C
def T : ℕ := 5
def H : ℕ := 10
def C : ℕ := 3

-- Definition of the formula for L
def crushingLoad (T H C : ℕ) : ℚ := (15 * T^3 : ℚ) / (H^2 + C)

-- The theorem to prove
theorem calculateL : crushingLoad T H C = 1875 / 103 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_calculateL_l2293_229307


namespace NUMINAMATH_GPT_find_cost_l2293_229397

def cost_of_article (C : ℝ) (G : ℝ) : Prop :=
  (580 = C + G) ∧ (600 = C + G + 0.05 * G)

theorem find_cost (C : ℝ) (G : ℝ) (h : cost_of_article C G) : C = 180 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_l2293_229397


namespace NUMINAMATH_GPT_cos_double_alpha_two_alpha_minus_beta_l2293_229315

variable (α β : ℝ)
variable (α_pos : 0 < α)
variable (α_lt_pi : α < π)
variable (tan_α : Real.tan α = 2)

variable (β_pos : 0 < β)
variable (β_lt_pi : β < π)
variable (cos_β : Real.cos β = -((7 * Real.sqrt 2) / 10))

theorem cos_double_alpha (hα : 0 < α ∧ α < π) (htan : Real.tan α = 2) : 
  Real.cos (2 * α) = -3 / 5 := by
  sorry

theorem two_alpha_minus_beta (hα : 0 < α ∧ α < π) (htan : Real.tan α = 2)
  (hβ : 0 < β ∧ β < π) (hcosβ : Real.cos β = -((7 * Real.sqrt 2) / 10)) : 
  2 * α - β = -π / 4 := by
  sorry

end NUMINAMATH_GPT_cos_double_alpha_two_alpha_minus_beta_l2293_229315


namespace NUMINAMATH_GPT_mary_remaining_money_l2293_229363

variable (p : ℝ) -- p is the price per drink in dollars

def drinks_cost : ℝ := 3 * p
def medium_pizzas_cost : ℝ := 2 * (2 * p)
def large_pizza_cost : ℝ := 3 * p

def total_cost : ℝ := drinks_cost p + medium_pizzas_cost p + large_pizza_cost p

theorem mary_remaining_money : 
  30 - total_cost p = 30 - 10 * p := 
by
  sorry

end NUMINAMATH_GPT_mary_remaining_money_l2293_229363


namespace NUMINAMATH_GPT_product_of_three_consecutive_not_div_by_5_adjacency_l2293_229375

theorem product_of_three_consecutive_not_div_by_5_adjacency (a b c : ℕ) (h₁ : a + 1 = b) (h₂ : b + 1 = c) (h₃ : a % 5 ≠ 0) (h₄ : b % 5 ≠ 0) (h₅ : c % 5 ≠ 0) :
  ((a * b * c) % 5 = 1) ∨ ((a * b * c) % 5 = 4) := 
sorry

end NUMINAMATH_GPT_product_of_three_consecutive_not_div_by_5_adjacency_l2293_229375


namespace NUMINAMATH_GPT_donation_amount_is_correct_l2293_229314

def stuffed_animals_barbara : ℕ := 9
def stuffed_animals_trish : ℕ := 2 * stuffed_animals_barbara
def stuffed_animals_sam : ℕ := stuffed_animals_barbara + 5
def stuffed_animals_linda : ℕ := stuffed_animals_sam - 7

def price_per_barbara : ℝ := 2
def price_per_trish : ℝ := 1.5
def price_per_sam : ℝ := 2.5
def price_per_linda : ℝ := 3

def total_amount_collected : ℝ := 
  stuffed_animals_barbara * price_per_barbara +
  stuffed_animals_trish * price_per_trish +
  stuffed_animals_sam * price_per_sam +
  stuffed_animals_linda * price_per_linda

def discount : ℝ := 0.10

def final_amount : ℝ := total_amount_collected * (1 - discount)

theorem donation_amount_is_correct : final_amount = 90.90 := sorry

end NUMINAMATH_GPT_donation_amount_is_correct_l2293_229314


namespace NUMINAMATH_GPT_contrapositive_proposition_contrapositive_version_l2293_229380

variable {a b : ℝ}

theorem contrapositive_proposition (h : a + b = 1) : a^2 + b^2 ≥ 1/2 :=
sorry

theorem contrapositive_version : a^2 + b^2 < 1/2 → a + b ≠ 1 :=
by
  intros h
  intro hab
  apply not_le.mpr h
  exact contrapositive_proposition hab

end NUMINAMATH_GPT_contrapositive_proposition_contrapositive_version_l2293_229380


namespace NUMINAMATH_GPT_find_additional_discount_l2293_229385

noncomputable def calculate_additional_discount (msrp : ℝ) (regular_discount_percent : ℝ) (final_price : ℝ) : ℝ :=
  let regular_discounted_price := msrp * (1 - regular_discount_percent / 100)
  let additional_discount_percent := ((regular_discounted_price - final_price) / regular_discounted_price) * 100
  additional_discount_percent

theorem find_additional_discount :
  calculate_additional_discount 35 30 19.6 = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_additional_discount_l2293_229385


namespace NUMINAMATH_GPT_find_pairs_l2293_229342

noncomputable def x (a b : ℝ) : ℝ := b^2 - (a - 1)/2
noncomputable def y (a b : ℝ) : ℝ := a^2 + (b + 1)/2
def valid_pair (a b : ℝ) : Prop := max (x a b) (y a b) ≤ 7 / 16

theorem find_pairs : valid_pair (1/4) (-1/4) :=
  sorry

end NUMINAMATH_GPT_find_pairs_l2293_229342


namespace NUMINAMATH_GPT_francie_remaining_money_l2293_229350

-- Define the initial weekly allowance for the first period
def initial_weekly_allowance : ℕ := 5
-- Length of the first period in weeks
def first_period_weeks : ℕ := 8
-- Define the raised weekly allowance for the second period
def raised_weekly_allowance : ℕ := 6
-- Length of the second period in weeks
def second_period_weeks : ℕ := 6
-- Cost of the video game
def video_game_cost : ℕ := 35

-- Define the total savings before buying new clothes
def total_savings :=
  first_period_weeks * initial_weekly_allowance + second_period_weeks * raised_weekly_allowance

-- Define the money spent on new clothes
def money_spent_on_clothes :=
  total_savings / 2

-- Define the remaining money after buying the video game
def remaining_money :=
  money_spent_on_clothes - video_game_cost

-- Prove that Francie has $3 remaining after buying the video game
theorem francie_remaining_money : remaining_money = 3 := by
  sorry

end NUMINAMATH_GPT_francie_remaining_money_l2293_229350


namespace NUMINAMATH_GPT_cos_225_degrees_l2293_229382

theorem cos_225_degrees : Real.cos (225 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_cos_225_degrees_l2293_229382


namespace NUMINAMATH_GPT_train_speed_l2293_229373

theorem train_speed (length : ℕ) (time : ℕ) (v : ℕ)
  (h1 : length = 750)
  (h2 : time = 1)
  (h3 : v = (length + length) / time)
  (h4 : v = 1500) :
  (v * 60 / 1000 = 90) :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l2293_229373


namespace NUMINAMATH_GPT_problem_proof_l2293_229322

theorem problem_proof (a b c : ℝ) (h1 : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h2 : a / (b - c) + b / (c - a) + c / (a - b) = 0) : 
  a / (b - c) ^ 2 + b / (c - a) ^ 2 + c / (a - b) ^ 2 = 0 :=
sorry

end NUMINAMATH_GPT_problem_proof_l2293_229322


namespace NUMINAMATH_GPT_gain_percent_l2293_229328

variable (C S : ℝ)

theorem gain_percent (h : 50 * C = 28 * S) : ((S - C) / C) * 100 = 78.57 := by
  sorry

end NUMINAMATH_GPT_gain_percent_l2293_229328


namespace NUMINAMATH_GPT_max_sn_at_16_l2293_229383

variable {a : ℕ → ℝ} -- the sequence a_n is represented by a

-- Conditions given in the problem
def isArithmetic (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def bn (a : ℕ → ℝ) (n : ℕ) : ℝ := a n * a (n + 1) * a (n + 2)

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := (Finset.range n).sum (bn a)

-- Condition: a_{12} = 3/8 * a_5 and a_12 > 0
def specificCondition (a : ℕ → ℝ) : Prop := a 12 = (3 / 8) * a 5 ∧ a 12 > 0

-- The theorem to prove that for S n, the maximum value is reached at n = 16
theorem max_sn_at_16 (a : ℕ → ℝ) (h_arithmetic : isArithmetic a) (h_condition : specificCondition a) :
  ∀ n : ℕ, Sn a n ≤ Sn a 16 := sorry

end NUMINAMATH_GPT_max_sn_at_16_l2293_229383


namespace NUMINAMATH_GPT_gateway_academy_problem_l2293_229394

theorem gateway_academy_problem :
  let total_students := 100
  let students_like_skating := 0.4 * total_students
  let students_dislike_skating := total_students - students_like_skating
  let like_and_say_like := 0.7 * students_like_skating
  let like_and_say_dislike := students_like_skating - like_and_say_like
  let dislike_and_say_dislike := 0.8 * students_dislike_skating
  let dislike_and_say_like := students_dislike_skating - dislike_and_say_dislike
  let says_dislike := like_and_say_dislike + dislike_and_say_dislike
  (like_and_say_dislike / says_dislike) = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_gateway_academy_problem_l2293_229394


namespace NUMINAMATH_GPT_diophantine_solution_exists_l2293_229362

theorem diophantine_solution_exists (D : ℤ) : 
  ∃ (x y z : ℕ), x^2 - D * y^2 = z^2 ∧ ∃ m n : ℕ, m^2 > D * n^2 :=
sorry

end NUMINAMATH_GPT_diophantine_solution_exists_l2293_229362


namespace NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l2293_229392

def p (x : ℝ) : Prop := x = 1
def q (x : ℝ) : Prop := x = 1 ∨ x = -2

theorem p_sufficient_but_not_necessary_for_q (x : ℝ) : (p x → q x) ∧ ¬(q x → p x) := 
by {
  sorry
}

end NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l2293_229392


namespace NUMINAMATH_GPT_three_is_square_root_of_nine_l2293_229357

theorem three_is_square_root_of_nine :
  ∃ x : ℝ, x * x = 9 ∧ x = 3 :=
sorry

end NUMINAMATH_GPT_three_is_square_root_of_nine_l2293_229357


namespace NUMINAMATH_GPT_distance_to_place_equals_2_point_25_l2293_229359

-- Definitions based on conditions
def rowing_speed : ℝ := 4
def river_speed : ℝ := 2
def total_time_hours : ℝ := 1.5

-- Downstream speed = rowing_speed + river_speed
def downstream_speed : ℝ := rowing_speed + river_speed
-- Upstream speed = rowing_speed - river_speed
def upstream_speed : ℝ := rowing_speed - river_speed

-- Define the distance d
def distance (d : ℝ) : Prop :=
  (d / downstream_speed + d / upstream_speed = total_time_hours)

-- The theorem statement
theorem distance_to_place_equals_2_point_25 :
  ∃ d : ℝ, distance d ∧ d = 2.25 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_place_equals_2_point_25_l2293_229359


namespace NUMINAMATH_GPT_common_difference_range_l2293_229339

variable (d : ℝ)

def a (n : ℕ) : ℝ := -5 + (n - 1) * d

theorem common_difference_range (H1 : a 10 > 0) (H2 : a 9 ≤ 0) :
  (5 / 9 < d) ∧ (d ≤ 5 / 8) :=
by
  sorry

end NUMINAMATH_GPT_common_difference_range_l2293_229339


namespace NUMINAMATH_GPT_positive_integer_solutions_eq_17_l2293_229337

theorem positive_integer_solutions_eq_17 :
  {x : ℕ // x > 0} × {y : ℕ // y > 0} → 5 * x + 10 * y = 100 ->
  ∃ (n : ℕ), n = 17 := sorry

end NUMINAMATH_GPT_positive_integer_solutions_eq_17_l2293_229337


namespace NUMINAMATH_GPT_cost_price_l2293_229336

/-- A person buys an article at some price. 
They sell the article to make a profit of 24%. 
The selling price of the article is Rs. 595.2. 
Prove that the cost price (CP) is Rs. 480. -/
theorem cost_price (SP CP : ℝ) (h1 : SP = 595.2) (h2 : SP = CP * (1 + 0.24)) : CP = 480 := 
by sorry 

end NUMINAMATH_GPT_cost_price_l2293_229336


namespace NUMINAMATH_GPT_area_of_rectangular_plot_l2293_229323

theorem area_of_rectangular_plot (breadth : ℝ) (length : ℝ) 
    (h1 : breadth = 17) 
    (h2 : length = 3 * breadth) : 
    length * breadth = 867 := 
by
  sorry

end NUMINAMATH_GPT_area_of_rectangular_plot_l2293_229323


namespace NUMINAMATH_GPT_harriet_current_age_l2293_229343

theorem harriet_current_age (peter_age harriet_age : ℕ) (mother_age : ℕ := 60) (h₁ : peter_age = mother_age / 2) 
  (h₂ : peter_age + 4 = 2 * (harriet_age + 4)) : harriet_age = 13 :=
by
  sorry

end NUMINAMATH_GPT_harriet_current_age_l2293_229343


namespace NUMINAMATH_GPT_parallelepiped_diagonal_l2293_229368

theorem parallelepiped_diagonal 
  (x y z m n p d : ℝ)
  (h1 : x^2 + y^2 = m^2)
  (h2 : x^2 + z^2 = n^2)
  (h3 : y^2 + z^2 = p^2)
  : d = Real.sqrt ((m^2 + n^2 + p^2) / 2) := 
sorry

end NUMINAMATH_GPT_parallelepiped_diagonal_l2293_229368


namespace NUMINAMATH_GPT_square_area_eq_l2293_229316

-- Define the side length of the square and the diagonal relationship
variables (s : ℝ) (h : s * Real.sqrt 2 = s + 1)

-- State the theorem to solve
theorem square_area_eq :
  s * Real.sqrt 2 = s + 1 → (s ^ 2 = 3 + 2 * Real.sqrt 2) :=
by
  -- Assume the given condition
  intro h
  -- Insert proof steps here, analysis follows the provided solution steps.
  sorry

end NUMINAMATH_GPT_square_area_eq_l2293_229316


namespace NUMINAMATH_GPT_fraction_of_suitable_dishes_l2293_229399

theorem fraction_of_suitable_dishes {T : Type} (total_menu: ℕ) (vegan_dishes: ℕ) (vegan_fraction: ℚ) (gluten_inclusive_vegan_dishes: ℕ) (low_sugar_gluten_free_vegan_dishes: ℕ) 
(h1: vegan_dishes = 6)
(h2: vegan_fraction = 1/4)
(h3: gluten_inclusive_vegan_dishes = 4)
(h4: low_sugar_gluten_free_vegan_dishes = 1)
(h5: total_menu = vegan_dishes / vegan_fraction) :
(1 : ℚ) / (total_menu : ℚ) = (1 : ℚ) / 24 := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_suitable_dishes_l2293_229399


namespace NUMINAMATH_GPT_c_sub_a_equals_90_l2293_229321

variables (a b c : ℝ)

theorem c_sub_a_equals_90 (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 90) : c - a = 90 :=
by
  sorry

end NUMINAMATH_GPT_c_sub_a_equals_90_l2293_229321


namespace NUMINAMATH_GPT_remainder_71_3_73_5_mod_8_l2293_229391

theorem remainder_71_3_73_5_mod_8 :
  (71^3) * (73^5) % 8 = 7 :=
by {
  -- hint, use the conditions given: 71 ≡ -1 (mod 8) and 73 ≡ 1 (mod 8)
  sorry
}

end NUMINAMATH_GPT_remainder_71_3_73_5_mod_8_l2293_229391


namespace NUMINAMATH_GPT_segment_length_greater_than_inradius_sqrt_two_l2293_229386

variables {a b c : ℝ} -- sides of the triangle
variables {P Q : ℝ} -- points on sides of the triangle
variables {S_ABC S_PCQ : ℝ} -- areas of the triangles
variables {s : ℝ} -- semi-perimeter of the triangle
variables {r : ℝ} -- radius of the inscribed circle
variables {ℓ : ℝ} -- length of segment dividing the triangle's area

-- Given conditions in the form of assumptions
variables (h1 : S_PCQ = S_ABC / 2)
variables (h2 : PQ = ℓ)
variables (h3 : r = S_ABC / s)

-- The statement of the theorem
theorem segment_length_greater_than_inradius_sqrt_two
  (h1 : S_PCQ = S_ABC / 2) 
  (h2 : PQ = ℓ) 
  (h3 : r = S_ABC / s)
  (h4 : s = (a + b + c) / 2) 
  (h5 : S_ABC = Real.sqrt (s * (s - a) * (s - b) * (s - c))) 
  (h6 : ℓ^2 = a^2 + b^2 - (a^2 + b^2 - c^2) / 2) :
  ℓ > r * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_segment_length_greater_than_inradius_sqrt_two_l2293_229386


namespace NUMINAMATH_GPT_min_value_inequality_l2293_229329

theorem min_value_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l2293_229329


namespace NUMINAMATH_GPT_find_d_vector_l2293_229347

theorem find_d_vector (x y t : ℝ) (v d : ℝ × ℝ)
  (hline : y = (5 * x - 7) / 2)
  (hparam : ∃ t : ℝ, (x, y) = (4, 2) + t • d)
  (hdist : ∀ {x : ℝ}, x ≥ 4 → dist (x, (5 * x - 7) / 2) (4, 2) = t) :
  d = (2 / Real.sqrt 29, 5 / Real.sqrt 29) := 
sorry

end NUMINAMATH_GPT_find_d_vector_l2293_229347


namespace NUMINAMATH_GPT_binary_division_remainder_l2293_229367

theorem binary_division_remainder : 
  let b := 0b101101011010
  let n := 8
  b % n = 2 
:= by 
  sorry

end NUMINAMATH_GPT_binary_division_remainder_l2293_229367


namespace NUMINAMATH_GPT_cookies_ratio_l2293_229381

theorem cookies_ratio (T : ℝ) (h1 : 0 ≤ T) (h_total : 5 + T + 1.4 * T = 29) : T / 5 = 2 :=
by sorry

end NUMINAMATH_GPT_cookies_ratio_l2293_229381


namespace NUMINAMATH_GPT_cube_side_length_and_combined_volume_l2293_229351

theorem cube_side_length_and_combined_volume
  (surface_area_large_cube : ℕ)
  (h_surface_area : surface_area_large_cube = 864)
  (side_length_large_cube : ℕ)
  (combined_volume : ℕ) :
  side_length_large_cube = 12 ∧ combined_volume = 1728 :=
by
  -- Since we only need the statement, the proof steps are not included.
  sorry

end NUMINAMATH_GPT_cube_side_length_and_combined_volume_l2293_229351


namespace NUMINAMATH_GPT_total_people_in_bus_l2293_229365

-- Definitions based on the conditions
def left_seats : Nat := 15
def right_seats := left_seats - 3
def people_per_seat := 3
def back_seat_people := 9

-- Theorem statement
theorem total_people_in_bus : 
  (left_seats * people_per_seat) +
  (right_seats * people_per_seat) + 
  back_seat_people = 90 := 
by sorry

end NUMINAMATH_GPT_total_people_in_bus_l2293_229365


namespace NUMINAMATH_GPT_same_face_probability_l2293_229353

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end NUMINAMATH_GPT_same_face_probability_l2293_229353


namespace NUMINAMATH_GPT_set_equality_l2293_229387

def M : Set ℝ := {x | x^2 - x > 0}

def N : Set ℝ := {x | 1 / x < 1}

theorem set_equality : M = N := 
by
  sorry

end NUMINAMATH_GPT_set_equality_l2293_229387


namespace NUMINAMATH_GPT_negation_of_exists_cond_l2293_229334

theorem negation_of_exists_cond (x : ℝ) (h : x > 0) : ¬ (∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_exists_cond_l2293_229334


namespace NUMINAMATH_GPT_hexagon_circle_radius_l2293_229396

theorem hexagon_circle_radius (r : ℝ) :
  let side_length := 3
  let probability := (1 : ℝ) / 3
  (probability = 1 / 3) →
  r = 12 * Real.sqrt 3 / (Real.sqrt 6 - Real.sqrt 2) :=
by
  -- Begin proof here
  sorry

end NUMINAMATH_GPT_hexagon_circle_radius_l2293_229396


namespace NUMINAMATH_GPT_last_year_sales_l2293_229308

-- Define the conditions as constants
def sales_this_year : ℝ := 480
def percent_increase : ℝ := 0.50

-- The main theorem statement
theorem last_year_sales : 
  ∃ sales_last_year : ℝ, sales_this_year = sales_last_year * (1 + percent_increase) ∧ sales_last_year = 320 := 
by 
  sorry

end NUMINAMATH_GPT_last_year_sales_l2293_229308


namespace NUMINAMATH_GPT_union_complement_l2293_229345

open Set

def U : Set ℤ := {x | -3 < x ∧ x < 3}

def A : Set ℤ := {1, 2}

def B : Set ℤ := {-2, -1, 2}

theorem union_complement :
  A ∪ (U \ B) = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_union_complement_l2293_229345


namespace NUMINAMATH_GPT_area_of_wall_photo_l2293_229310

theorem area_of_wall_photo (width_frame : ℕ) (width_paper : ℕ) (length_paper : ℕ) 
  (h_width_frame : width_frame = 2) (h_width_paper : width_paper = 8) (h_length_paper : length_paper = 12) :
  (width_paper + 2 * width_frame) * (length_paper + 2 * width_frame) = 192 :=
by
  sorry

end NUMINAMATH_GPT_area_of_wall_photo_l2293_229310


namespace NUMINAMATH_GPT_stadium_height_l2293_229390

theorem stadium_height
  (l w d : ℕ) (h : ℕ) 
  (hl : l = 24) 
  (hw : w = 18) 
  (hd : d = 34) 
  (h_eq : d^2 = l^2 + w^2 + h^2) : 
  h = 16 := by 
  sorry

end NUMINAMATH_GPT_stadium_height_l2293_229390


namespace NUMINAMATH_GPT_iso_triangle_perimeter_l2293_229371

theorem iso_triangle_perimeter :
  ∃ p : ℕ, (p = 11 ∨ p = 13) ∧ ∃ a b : ℕ, a ≠ b ∧ a^2 - 8 * a + 15 = 0 ∧ b^2 - 8 * b + 15 = 0 :=
by
  sorry

end NUMINAMATH_GPT_iso_triangle_perimeter_l2293_229371


namespace NUMINAMATH_GPT_shirts_not_washed_l2293_229303

def total_shortsleeve_shirts : Nat := 40
def total_longsleeve_shirts : Nat := 23
def washed_shirts : Nat := 29

theorem shirts_not_washed :
  (total_shortsleeve_shirts + total_longsleeve_shirts) - washed_shirts = 34 :=
by
  sorry

end NUMINAMATH_GPT_shirts_not_washed_l2293_229303
