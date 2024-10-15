import Mathlib

namespace NUMINAMATH_GPT_smallest_positive_integer_expr_2010m_44000n_l1368_136812

theorem smallest_positive_integer_expr_2010m_44000n :
  ∃ (m n : ℤ), 10 = gcd 2010 44000 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_expr_2010m_44000n_l1368_136812


namespace NUMINAMATH_GPT_average_weight_of_Arun_l1368_136800

def Arun_weight_opinion (w : ℝ) : Prop :=
  (66 < w) ∧ (w < 72)

def Brother_weight_opinion (w : ℝ) : Prop :=
  (60 < w) ∧ (w < 70)

def Mother_weight_opinion (w : ℝ) : Prop :=
  w ≤ 69

def Father_weight_opinion (w : ℝ) : Prop :=
  (65 ≤ w) ∧ (w ≤ 71)

def Sister_weight_opinion (w : ℝ) : Prop :=
  (62 < w) ∧ (w ≤ 68)

def All_opinions (w : ℝ) : Prop :=
  Arun_weight_opinion w ∧
  Brother_weight_opinion w ∧
  Mother_weight_opinion w ∧
  Father_weight_opinion w ∧
  Sister_weight_opinion w

theorem average_weight_of_Arun : ∃ avg : ℝ, avg = 67.5 ∧ (∀ w, All_opinions w → (w = 67 ∨ w = 68)) :=
by
  sorry

end NUMINAMATH_GPT_average_weight_of_Arun_l1368_136800


namespace NUMINAMATH_GPT_find_b_l1368_136889

theorem find_b (b : ℤ) (h₁ : b < 0) : (∃ n : ℤ, (x : ℤ) * x + b * x - 36 = (x + n) * (x + n) - 20) → b = -8 :=
by
  intro hX
  sorry

end NUMINAMATH_GPT_find_b_l1368_136889


namespace NUMINAMATH_GPT_probability_same_color_socks_l1368_136870

-- Define the total number of socks and the groups
def total_socks : ℕ := 30
def blue_socks : ℕ := 16
def green_socks : ℕ := 10
def red_socks : ℕ := 4

-- Define combinatorial functions to calculate combinations
def comb (n m : ℕ) : ℕ := n.choose m

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  comb blue_socks 2 +
  comb green_socks 2 +
  comb red_socks 2

-- Calculate the total number of possible outcomes
def total_outcomes : ℕ := comb total_socks 2

-- Calculate the probability as a ratio of favorable outcomes to total outcomes
def probability := favorable_outcomes / total_outcomes

-- Prove the probability is 19/45
theorem probability_same_color_socks : probability = 19 / 45 := by
  sorry

end NUMINAMATH_GPT_probability_same_color_socks_l1368_136870


namespace NUMINAMATH_GPT_increasing_inverse_relation_l1368_136836

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry -- This is the inverse function f^-1

theorem increasing_inverse_relation {a b c : ℝ} 
  (h_inc_f : ∀ x y, x < y → f x < f y)
  (h_inc_f_inv : ∀ x y, x < y → f_inv x < f_inv y)
  (h_f3 : f 3 = 0)
  (h_f2 : f 2 = a)
  (h_f_inv2 : f_inv 2 = b)
  (h_f_inv0 : f_inv 0 = c) :
  b > c ∧ c > a := sorry

end NUMINAMATH_GPT_increasing_inverse_relation_l1368_136836


namespace NUMINAMATH_GPT_S15_constant_l1368_136851

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Given condition: a_5 + a_8 + a_11 is constant
axiom const_sum : ∀ (a1 d : ℤ), a 5 a1 d + a 8 a1 d + a 11 a1 d = 3 * a1 + 21 * d

-- The equivalent proof problem
theorem S15_constant (a1 d : ℤ) : S 15 a1 d = 5 * (3 * a1 + 21 * d) :=
by
  sorry

end NUMINAMATH_GPT_S15_constant_l1368_136851


namespace NUMINAMATH_GPT_candle_height_relation_l1368_136884

theorem candle_height_relation : 
  ∀ (h : ℝ) (t : ℝ), h = 1 → (∀ (h1_burn_rate : ℝ), h1_burn_rate = 1 / 5) → (∀ (h2_burn_rate : ℝ), h2_burn_rate = 1 / 6) →
  (1 - t * 1 / 5 = 3 * (1 - t * 1 / 6)) → t = 20 / 3 :=
by
  intros h t h_init h1_burn_rate h2_burn_rate height_eq
  sorry

end NUMINAMATH_GPT_candle_height_relation_l1368_136884


namespace NUMINAMATH_GPT_justin_home_time_l1368_136869

noncomputable def dinner_duration : ℕ := 45
noncomputable def homework_duration : ℕ := 30
noncomputable def cleaning_room_duration : ℕ := 30
noncomputable def taking_out_trash_duration : ℕ := 5
noncomputable def emptying_dishwasher_duration : ℕ := 10

noncomputable def total_time_required : ℕ :=
  dinner_duration + homework_duration + cleaning_room_duration + taking_out_trash_duration + emptying_dishwasher_duration

noncomputable def latest_start_time_hour : ℕ := 18 -- 6 pm in 24-hour format
noncomputable def total_time_required_hours : ℕ := 2
noncomputable def movie_time_hour : ℕ := 20 -- 8 pm in 24-hour format

theorem justin_home_time : latest_start_time_hour - total_time_required_hours = 16 := -- 4 pm in 24-hour format
by
  sorry

end NUMINAMATH_GPT_justin_home_time_l1368_136869


namespace NUMINAMATH_GPT_min_m_value_inequality_x2y2z_l1368_136856

theorem min_m_value (a b : ℝ) (h1 : a * b > 0) (h2 : a^2 * b = 2) : 
  ∃ (m : ℝ), m = a * b + a^2 ∧ m = 3 :=
sorry

theorem inequality_x2y2z 
  (t : ℝ) (ht : t = 3) (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x^2 + y^2 + z^2 = t / 3) : 
  |x + 2 * y + 2 * z| ≤ 3 :=
sorry

end NUMINAMATH_GPT_min_m_value_inequality_x2y2z_l1368_136856


namespace NUMINAMATH_GPT_distance_from_apex_to_larger_cross_section_l1368_136881

noncomputable def area1 : ℝ := 324 * Real.sqrt 2
noncomputable def area2 : ℝ := 648 * Real.sqrt 2
def distance_between_planes : ℝ := 12

theorem distance_from_apex_to_larger_cross_section
  (area1 area2 : ℝ)
  (distance_between_planes : ℝ)
  (h_area1 : area1 = 324 * Real.sqrt 2)
  (h_area2 : area2 = 648 * Real.sqrt 2)
  (h_distance : distance_between_planes = 12) :
  ∃ (H : ℝ), H = 24 + 12 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_distance_from_apex_to_larger_cross_section_l1368_136881


namespace NUMINAMATH_GPT_functions_equal_l1368_136832

noncomputable def f (x : ℝ) : ℝ := x^0
noncomputable def g (x : ℝ) : ℝ := x / x

theorem functions_equal (x : ℝ) (hx : x ≠ 0) : f x = g x :=
by
  unfold f g
  sorry

end NUMINAMATH_GPT_functions_equal_l1368_136832


namespace NUMINAMATH_GPT_log_one_plus_two_x_lt_two_x_l1368_136873
open Real

theorem log_one_plus_two_x_lt_two_x {x : ℝ} (hx : x > 0) : log (1 + 2 * x) < 2 * x :=
sorry

end NUMINAMATH_GPT_log_one_plus_two_x_lt_two_x_l1368_136873


namespace NUMINAMATH_GPT_max_area_triangle_max_area_quadrilateral_l1368_136858

-- Define the terms and conditions

variables {A O : Point}
variables {r d : ℝ}
variables {C D B : Point}

-- Problem (a)
theorem max_area_triangle (A O : Point) (d : ℝ) :
  (∃ x : ℝ, x = (3 / 4) * d) :=
sorry

-- Problem (b)
theorem max_area_quadrilateral (A O : Point) (d : ℝ) :
  (∃ x : ℝ, x = (1 / 2) * d) :=
sorry

end NUMINAMATH_GPT_max_area_triangle_max_area_quadrilateral_l1368_136858


namespace NUMINAMATH_GPT_find_value_of_g1_l1368_136877

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = g x

theorem find_value_of_g1 (h1 : is_odd f)
  (h2 : is_even g)
  (h3 : f (-1) + g 1 = 2)
  (h4 : f 1 + g (-1) = 4) : 
  g 1 = 3 :=
sorry

end NUMINAMATH_GPT_find_value_of_g1_l1368_136877


namespace NUMINAMATH_GPT_final_price_including_tax_l1368_136892

noncomputable def increasedPrice (originalPrice : ℝ) (increasePercentage : ℝ) : ℝ :=
  originalPrice + originalPrice * increasePercentage

noncomputable def discountedPrice (increasedPrice : ℝ) (discountPercentage : ℝ) : ℝ :=
  increasedPrice - increasedPrice * discountPercentage

noncomputable def finalPrice (discountedPrice : ℝ) (salesTax : ℝ) : ℝ :=
  discountedPrice + discountedPrice * salesTax

theorem final_price_including_tax :
  let originalPrice := 200
  let increasePercentage := 0.30
  let discountPercentage := 0.30
  let salesTax := 0.07
  let incPrice := increasedPrice originalPrice increasePercentage
  let disPrice := discountedPrice incPrice discountPercentage
  finalPrice disPrice salesTax = 194.74 :=
by
  simp [increasedPrice, discountedPrice, finalPrice]
  sorry

end NUMINAMATH_GPT_final_price_including_tax_l1368_136892


namespace NUMINAMATH_GPT_volume_of_sphere_inscribed_in_cube_of_edge_8_l1368_136861

noncomputable def volume_of_inscribed_sphere (edge_length : ℝ) : ℝ := 
  (4 / 3) * Real.pi * (edge_length / 2) ^ 3

theorem volume_of_sphere_inscribed_in_cube_of_edge_8 :
  volume_of_inscribed_sphere 8 = (256 / 3) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_volume_of_sphere_inscribed_in_cube_of_edge_8_l1368_136861


namespace NUMINAMATH_GPT_converse_even_sum_l1368_136871

variable (a b : ℤ)

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem converse_even_sum (h : is_even (a + b)) : is_even a ∧ is_even b :=
sorry

end NUMINAMATH_GPT_converse_even_sum_l1368_136871


namespace NUMINAMATH_GPT_frustum_midsection_area_l1368_136857

theorem frustum_midsection_area (r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 3) :
  let r_mid := (r1 + r2) / 2
  let area_mid := Real.pi * r_mid^2
  area_mid = 25 * Real.pi / 4 := by
  sorry

end NUMINAMATH_GPT_frustum_midsection_area_l1368_136857


namespace NUMINAMATH_GPT_simplify_exponents_l1368_136879

theorem simplify_exponents (x : ℝ) : x^5 * x^3 = x^8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_exponents_l1368_136879


namespace NUMINAMATH_GPT_expected_winnings_l1368_136868

theorem expected_winnings :
  let p_heads : ℚ := 1 / 4
  let p_tails : ℚ := 1 / 2
  let p_edge : ℚ := 1 / 4
  let win_heads : ℚ := 1
  let win_tails : ℚ := 3
  let loss_edge : ℚ := -8
  (p_heads * win_heads + p_tails * win_tails + p_edge * loss_edge) = -0.25 := 
by sorry

end NUMINAMATH_GPT_expected_winnings_l1368_136868


namespace NUMINAMATH_GPT_circle_tangent_to_x_axis_at_origin_l1368_136831

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h1 : ∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 → x = 0 ∧ y = 0 ∨ y = -D/E ∧ x = 0 ∧ F = 0):
  D = 0 ∧ E ≠ 0 ∧ F = 0 :=
sorry

end NUMINAMATH_GPT_circle_tangent_to_x_axis_at_origin_l1368_136831


namespace NUMINAMATH_GPT_archery_competition_l1368_136808

theorem archery_competition (points : Finset ℕ) (product : ℕ) : 
  points = {11, 7, 5, 2} ∧ product = 38500 → 
  ∃ n : ℕ, n = 7 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_archery_competition_l1368_136808


namespace NUMINAMATH_GPT_JulioHasMoreSoda_l1368_136834

-- Define the number of bottles each person has
def JulioOrangeBottles : ℕ := 4
def JulioGrapeBottles : ℕ := 7
def MateoOrangeBottles : ℕ := 1
def MateoGrapeBottles : ℕ := 3

-- Define the volume of each bottle in liters
def BottleVolume : ℕ := 2

-- Define the total liters of soda each person has
def JulioTotalLiters : ℕ := JulioOrangeBottles * BottleVolume + JulioGrapeBottles * BottleVolume
def MateoTotalLiters : ℕ := MateoOrangeBottles * BottleVolume + MateoGrapeBottles * BottleVolume

-- Prove the difference in total liters of soda between Julio and Mateo
theorem JulioHasMoreSoda : JulioTotalLiters - MateoTotalLiters = 14 := by
  sorry

end NUMINAMATH_GPT_JulioHasMoreSoda_l1368_136834


namespace NUMINAMATH_GPT_starting_number_unique_l1368_136837

-- Definitions based on conditions
def has_two_threes (n : ℕ) : Prop :=
  (n / 10 = 3 ∧ n % 10 = 3)

def is_starting_number (n m : ℕ) : Prop :=
  ∃ k, n + k = m ∧ k < (m - n) ∧ has_two_threes m

-- Theorem stating the proof problem
theorem starting_number_unique : ∃ n, is_starting_number n 30 ∧ n = 32 := 
sorry

end NUMINAMATH_GPT_starting_number_unique_l1368_136837


namespace NUMINAMATH_GPT_add_to_divisible_l1368_136883

theorem add_to_divisible (n d x : ℕ) (h : n = 987654) (h1 : d = 456) (h2 : x = 222) : 
  (n + x) % d = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_add_to_divisible_l1368_136883


namespace NUMINAMATH_GPT_proportion_margin_l1368_136863

theorem proportion_margin (S M C : ℝ) (n : ℝ) (hM : M = S / n) (hC : C = (1 - 1 / n) * S) :
  M / C = 1 / (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_proportion_margin_l1368_136863


namespace NUMINAMATH_GPT_math_problem_l1368_136872

theorem math_problem : 
  ∀ n : ℕ, 
  n = 5 * 96 → 
  ((n + 17) * 69) = 34293 := 
by
  intros n h
  sorry

end NUMINAMATH_GPT_math_problem_l1368_136872


namespace NUMINAMATH_GPT_no_solution_exists_l1368_136844

theorem no_solution_exists : ¬ ∃ n : ℕ, (n^2 ≡ 1 [MOD 5]) ∧ (n^3 ≡ 3 [MOD 5]) := 
sorry

end NUMINAMATH_GPT_no_solution_exists_l1368_136844


namespace NUMINAMATH_GPT_approx_change_in_y_l1368_136895

-- Definition of the function
def y (x : ℝ) : ℝ := x^3 - 7 * x^2 + 80

-- Derivative of the function, calculated manually
def y_prime (x : ℝ) : ℝ := 3 * x^2 - 14 * x

-- The change in x
def delta_x : ℝ := 0.01

-- The given value of x
def x_initial : ℝ := 5

-- To be proved: the approximate change in y
theorem approx_change_in_y : (y_prime x_initial) * delta_x = 0.05 :=
by
  -- Imported and recognized theorem verifications skipped
  sorry

end NUMINAMATH_GPT_approx_change_in_y_l1368_136895


namespace NUMINAMATH_GPT_maximize_tetrahedron_volume_l1368_136802

noncomputable def volume_of_tetrahedron (a : ℝ) : ℝ :=
  a / 6

theorem maximize_tetrahedron_volume (a : ℝ) (h_a : 0 < a) 
  (P Q X Y : ℝ × ℝ × ℝ) (h_PQ : dist P Q = 1) (h_XY : dist X Y = 1) :
  volume_of_tetrahedron a = a / 6 :=
by
  sorry

end NUMINAMATH_GPT_maximize_tetrahedron_volume_l1368_136802


namespace NUMINAMATH_GPT_probability_of_selection_of_Ram_l1368_136813

noncomputable def P_Ravi : ℚ := 1 / 5
noncomputable def P_Ram_and_Ravi : ℚ := 57 / 1000  -- This is the exact form of 0.05714285714285714

axiom independent_selection : ∀ (P_Ram P_Ravi : ℚ), P_Ram_and_Ravi = P_Ram * P_Ravi

theorem probability_of_selection_of_Ram (P_Ram : ℚ) : P_Ram_and_Ravi = P_Ram * P_Ravi → P_Ram = 2 / 7 := by
  intro h
  have h1 : P_Ram = P_Ram_and_Ravi / P_Ravi := sorry
  rw [h1, P_Ram_and_Ravi, P_Ravi]
  norm_num
  exact sorry

end NUMINAMATH_GPT_probability_of_selection_of_Ram_l1368_136813


namespace NUMINAMATH_GPT_spatial_quadrilateral_angle_sum_l1368_136875

theorem spatial_quadrilateral_angle_sum (A B C D : ℝ) (ABD DBC ADB BDC : ℝ) :
  (A <= ABD + DBC) → (C <= ADB + BDC) → 
  (A + C + B + D <= 360) := 
by
  intros
  sorry

end NUMINAMATH_GPT_spatial_quadrilateral_angle_sum_l1368_136875


namespace NUMINAMATH_GPT_simplify_expression_l1368_136805

theorem simplify_expression : -Real.sqrt 4 + abs (Real.sqrt 2 - 2) - 2023^0 = -2 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1368_136805


namespace NUMINAMATH_GPT_sum_of_integers_l1368_136847

theorem sum_of_integers : (∀ (x y : ℤ), x = -4 ∧ y = -5 ∧ x - y = 1 → x + y = -9) := 
by 
  intros x y
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1368_136847


namespace NUMINAMATH_GPT_mary_fruits_left_l1368_136897

-- Conditions as definitions:
def mary_bought_apples : ℕ := 14
def mary_bought_oranges : ℕ := 9
def mary_bought_blueberries : ℕ := 6

def mary_ate_apples : ℕ := 1
def mary_ate_oranges : ℕ := 1
def mary_ate_blueberries : ℕ := 1

-- The problem statement:
theorem mary_fruits_left : 
  (mary_bought_apples - mary_ate_apples) + 
  (mary_bought_oranges - mary_ate_oranges) + 
  (mary_bought_blueberries - mary_ate_blueberries) = 26 := by
  sorry

end NUMINAMATH_GPT_mary_fruits_left_l1368_136897


namespace NUMINAMATH_GPT_find_original_price_l1368_136821

theorem find_original_price (reduced_price : ℝ) (percent : ℝ) (original_price : ℝ) 
  (h1 : reduced_price = 6) (h2 : percent = 0.25) (h3 : reduced_price = percent * original_price) : 
  original_price = 24 :=
sorry

end NUMINAMATH_GPT_find_original_price_l1368_136821


namespace NUMINAMATH_GPT_solve_for_b_l1368_136824

theorem solve_for_b (b : ℚ) (h : b + b / 4 - 1 = 3 / 2) : b = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_b_l1368_136824


namespace NUMINAMATH_GPT_gcd_multiple_less_than_120_l1368_136848

theorem gcd_multiple_less_than_120 (n : ℕ) (h1 : n < 120) (h2 : n % 10 = 0) (h3 : n % 15 = 0) : n ≤ 90 :=
by {
  sorry
}

end NUMINAMATH_GPT_gcd_multiple_less_than_120_l1368_136848


namespace NUMINAMATH_GPT_flag_movement_distance_l1368_136819

theorem flag_movement_distance 
  (flagpole_length : ℝ)
  (half_mast : ℝ)
  (top_to_halfmast : ℝ)
  (halfmast_to_top : ℝ)
  (top_to_bottom : ℝ)
  (H1 : flagpole_length = 60)
  (H2 : half_mast = flagpole_length / 2)
  (H3 : top_to_halfmast = half_mast)
  (H4 : halfmast_to_top = half_mast)
  (H5 : top_to_bottom = flagpole_length) :
  top_to_halfmast + halfmast_to_top + top_to_halfmast + top_to_bottom = 180 := 
sorry

end NUMINAMATH_GPT_flag_movement_distance_l1368_136819


namespace NUMINAMATH_GPT_negation_proof_l1368_136882

theorem negation_proof :
  ¬ (∀ x : ℝ, 0 < x ∧ x < (π / 2) → x > Real.sin x) ↔ 
  ∃ x : ℝ, 0 < x ∧ x < (π / 2) ∧ x ≤ Real.sin x := 
sorry

end NUMINAMATH_GPT_negation_proof_l1368_136882


namespace NUMINAMATH_GPT_trigonometric_identity_l1368_136853

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α - Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 1 / 13 := 
by
-- The proof goes here
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1368_136853


namespace NUMINAMATH_GPT_lefty_jazz_non_basketball_l1368_136899

-- Definitions
def total_members : ℕ := 30
def left_handed_members : ℕ := 12
def jazz_loving_members : ℕ := 20
def right_handed_non_jazz_non_basketball : ℕ := 5
def basketball_players : ℕ := 10
def left_handed_jazz_loving_basketball_players : ℕ := 3

-- Problem Statement: Prove the number of lefty jazz lovers who do not play basketball.
theorem lefty_jazz_non_basketball (x : ℕ) :
  (x + left_handed_jazz_loving_basketball_players) + (left_handed_members - x - left_handed_jazz_loving_basketball_players) + 
  (jazz_loving_members - x - left_handed_jazz_loving_basketball_players) + 
  right_handed_non_jazz_non_basketball + left_handed_jazz_loving_basketball_players = 
  total_members → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_lefty_jazz_non_basketball_l1368_136899


namespace NUMINAMATH_GPT_a4_is_5_l1368_136811

-- Definitions based on the given conditions in the problem
def sum_arith_seq (n a1 d : ℤ) : ℤ := n * a1 + (n * (n-1)) / 2 * d

def S6 : ℤ := 24
def S9 : ℤ := 63

-- The proof problem: we need to prove that a4 = 5 given the conditions
theorem a4_is_5 (a1 d : ℤ) (h_S6 : sum_arith_seq 6 a1 d = S6) (h_S9 : sum_arith_seq 9 a1 d = S9) : 
  a1 + 3 * d = 5 :=
sorry

end NUMINAMATH_GPT_a4_is_5_l1368_136811


namespace NUMINAMATH_GPT_cricket_initial_overs_l1368_136820

theorem cricket_initial_overs
  (x : ℕ)
  (hx1 : ∃ x : ℕ, 0 ≤ x)
  (initial_run_rate : ℝ)
  (remaining_run_rate : ℝ)
  (remaining_overs : ℕ)
  (target_runs : ℕ)
  (H1 : initial_run_rate = 3.2)
  (H2 : remaining_run_rate = 6.25)
  (H3 : remaining_overs = 40)
  (H4 : target_runs = 282) :
  3.2 * (x : ℝ) + 6.25 * 40 = 282 → x = 10 := 
by 
  simp only [H1, H2, H3, H4]
  sorry

end NUMINAMATH_GPT_cricket_initial_overs_l1368_136820


namespace NUMINAMATH_GPT_find_value_of_A_l1368_136833

theorem find_value_of_A (A ω φ c : ℝ)
  (a : ℕ+ → ℝ)
  (h_seq : ∀ n : ℕ+, a n * a (n + 1) * a (n + 2) = a n + a (n + 1) + a (n + 2))
  (h_neq : ∀ n : ℕ+, a n * a (n + 1) ≠ 1)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 2)
  (h_form : ∀ n : ℕ+, a n = A * Real.sin (ω * n + φ) + c)
  (h_ω_gt_0 : ω > 0)
  (h_phi_lt_pi_div_2 : |φ| < Real.pi / 2) :
  A = -2 * Real.sqrt 3 / 3 := 
sorry

end NUMINAMATH_GPT_find_value_of_A_l1368_136833


namespace NUMINAMATH_GPT_find_five_digit_number_l1368_136842

theorem find_five_digit_number (a b c d e : ℕ) 
  (h : [ (10 * a + a), (10 * a + b), (10 * a + b), (10 * a + b), (10 * a + c), 
         (10 * b + c), (10 * b + b), (10 * b + c), (10 * c + b), (10 * c + b)] = 
         [33, 37, 37, 37, 38, 73, 77, 78, 83, 87]) :
  10000 * a + 1000 * b + 100 * c + 10 * d + e = 37837 :=
sorry

end NUMINAMATH_GPT_find_five_digit_number_l1368_136842


namespace NUMINAMATH_GPT_variance_of_data_l1368_136860

theorem variance_of_data :
  let data := [3, 1, 0, -1, -3]
  let mean := (3 + 1 + 0 - 1 - 3) / (5:ℝ)
  let variance := (1 / 5:ℝ) * (3^2 + 1^2 + (-1)^2 + (-3)^2)
  variance = 4 := sorry

end NUMINAMATH_GPT_variance_of_data_l1368_136860


namespace NUMINAMATH_GPT_probability_obtuse_triangle_is_one_fourth_l1368_136843

-- Define the set of possible integers
def S : Set ℤ := {1, 2, 3, 4, 5, 6}

-- Condition for forming an obtuse triangle
def is_obtuse_triangle (a b c : ℤ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b ∧ 
  (a^2 + b^2 < c^2 ∨ a^2 + c^2 < b^2 ∨ b^2 + c^2 < a^2)

-- List of valid triples that can form an obtuse triangle
def valid_obtuse_triples : List (ℤ × ℤ × ℤ) :=
  [(2, 3, 4), (2, 4, 5), (2, 5, 6), (3, 4, 6), (3, 5, 6)]

-- Total number of combinations
def total_combinations : Nat := 20

-- Number of valid combinations for obtuse triangles
def valid_combinations : Nat := 5

-- Calculate the probability
def probability_obtuse_triangle : ℚ := valid_combinations / total_combinations

theorem probability_obtuse_triangle_is_one_fourth :
  probability_obtuse_triangle = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_obtuse_triangle_is_one_fourth_l1368_136843


namespace NUMINAMATH_GPT_marbles_per_friend_l1368_136840

theorem marbles_per_friend (total_marbles friends : ℕ) (h1 : total_marbles = 5504) (h2 : friends = 64) :
  total_marbles / friends = 86 :=
by {
  -- Proof will be added here
  sorry
}

end NUMINAMATH_GPT_marbles_per_friend_l1368_136840


namespace NUMINAMATH_GPT_equalities_imply_forth_l1368_136838

variables {a b c d e f g h S1 S2 S3 O2 O3 : ℕ}

def S1_def := S1 = a + b + c
def S2_def := S2 = d + e + f
def S3_def := S3 = b + c + g + h - d
def O2_def := O2 = b + e + g
def O3_def := O3 = c + f + h

theorem equalities_imply_forth (h1 : S1 = S2) (h2 : S1 = S3) (h3 : S1 = O2) : S1 = O3 :=
  by sorry

end NUMINAMATH_GPT_equalities_imply_forth_l1368_136838


namespace NUMINAMATH_GPT_geom_sequence_sum_l1368_136829

theorem geom_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n : ℕ, a n > 0)
  (h_geom : ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q)
  (h_eq : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) :
  a 5 + a 7 = 6 :=
sorry

end NUMINAMATH_GPT_geom_sequence_sum_l1368_136829


namespace NUMINAMATH_GPT_min_balls_to_draw_l1368_136896

theorem min_balls_to_draw (red blue green yellow white black : ℕ) (h_red : red = 35) (h_blue : blue = 25) (h_green : green = 22) (h_yellow : yellow = 18) (h_white : white = 14) (h_black : black = 12) : 
  ∃ n, n = 95 ∧ ∀ (r b g y w bl : ℕ), r ≤ red ∧ b ≤ blue ∧ g ≤ green ∧ y ≤ yellow ∧ w ≤ white ∧ bl ≤ black → (r + b + g + y + w + bl = 95 → r ≥ 18 ∨ b ≥ 18 ∨ g ≥ 18 ∨ y ≥ 18 ∨ w ≥ 18 ∨ bl ≥ 18) :=
by sorry

end NUMINAMATH_GPT_min_balls_to_draw_l1368_136896


namespace NUMINAMATH_GPT_min_value_a_b_c_l1368_136894

theorem min_value_a_b_c (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : 9 * a + 4 * b = a * b * c) :
  a + b + c = 10 := sorry

end NUMINAMATH_GPT_min_value_a_b_c_l1368_136894


namespace NUMINAMATH_GPT_candy_store_sampling_l1368_136846

theorem candy_store_sampling (total_customers sampling_customers caught_customers not_caught_customers : ℝ)
    (h1 : caught_customers = 0.22 * total_customers)
    (h2 : not_caught_customers = 0.15 * sampling_customers)
    (h3 : sampling_customers = caught_customers + not_caught_customers):
    sampling_customers = 0.2588 * total_customers := by
  sorry

end NUMINAMATH_GPT_candy_store_sampling_l1368_136846


namespace NUMINAMATH_GPT_solve_equation_l1368_136849

theorem solve_equation (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^(2 * y - 1) + (x + 1)^(2 * y - 1) = (x + 2)^(2 * y - 1) ↔ (x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1368_136849


namespace NUMINAMATH_GPT_angle_c_in_triangle_l1368_136891

theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A/B = 1/3) (h3 : A/C = 1/5) : C = 100 :=
by
  sorry

end NUMINAMATH_GPT_angle_c_in_triangle_l1368_136891


namespace NUMINAMATH_GPT_Don_poured_milk_correct_amount_l1368_136814

theorem Don_poured_milk_correct_amount :
  let original_milk : ℚ := 3 / 8
  let portion_poured : ℚ := 5 / 6
  let poured_milk : ℚ := portion_poured * original_milk
  poured_milk = 5 / 16 :=
by
  let original_milk : ℚ := 3 / 8
  let portion_poured : ℚ := 5 / 6
  let poured_milk : ℚ := portion_poured * original_milk
  show poured_milk = 5 / 16
  sorry

end NUMINAMATH_GPT_Don_poured_milk_correct_amount_l1368_136814


namespace NUMINAMATH_GPT_time_jran_l1368_136859

variable (D : ℕ) (S : ℕ)

theorem time_jran (hD: D = 80) (hS : S = 10) : D / S = 8 := 
  sorry

end NUMINAMATH_GPT_time_jran_l1368_136859


namespace NUMINAMATH_GPT_sqrt_five_minus_one_range_l1368_136876

theorem sqrt_five_minus_one_range (h : 2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3) : 
  1 < Real.sqrt 5 - 1 ∧ Real.sqrt 5 - 1 < 2 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_five_minus_one_range_l1368_136876


namespace NUMINAMATH_GPT_calc_ratio_of_d_to_s_l1368_136862

theorem calc_ratio_of_d_to_s {n s d : ℝ} (h_n_eq_24 : n = 24)
    (h_tiles_area_64_pct : (576 * s^2) = 0.64 * (n * s + d)^2) : 
    d / s = 6 / 25 :=
by
  sorry

end NUMINAMATH_GPT_calc_ratio_of_d_to_s_l1368_136862


namespace NUMINAMATH_GPT_correct_total_annual_cost_l1368_136867

def cost_after_coverage (cost: ℕ) (coverage: ℕ) : ℕ :=
  cost - (cost * coverage / 100)

def epiPen_costs : ℕ :=
  (cost_after_coverage 500 75) +
  (cost_after_coverage 550 60) +
  (cost_after_coverage 480 70) +
  (cost_after_coverage 520 65)

def monthly_medical_expenses : ℕ :=
  (cost_after_coverage 250 80) +
  (cost_after_coverage 180 70) +
  (cost_after_coverage 300 75) +
  (cost_after_coverage 350 60) +
  (cost_after_coverage 200 70) +
  (cost_after_coverage 400 80) +
  (cost_after_coverage 150 90) +
  (cost_after_coverage 100 100) +
  (cost_after_coverage 300 60) +
  (cost_after_coverage 350 90) +
  (cost_after_coverage 450 85) +
  (cost_after_coverage 500 65)

def total_annual_cost : ℕ :=
  epiPen_costs + monthly_medical_expenses

theorem correct_total_annual_cost :
  total_annual_cost = 1542 :=
  by sorry

end NUMINAMATH_GPT_correct_total_annual_cost_l1368_136867


namespace NUMINAMATH_GPT_edward_remaining_money_l1368_136816

def initial_amount : ℕ := 19
def spent_amount : ℕ := 13
def remaining_amount : ℕ := initial_amount - spent_amount

theorem edward_remaining_money : remaining_amount = 6 := by
  sorry

end NUMINAMATH_GPT_edward_remaining_money_l1368_136816


namespace NUMINAMATH_GPT_largest_n_arithmetic_sequences_l1368_136855

theorem largest_n_arithmetic_sequences
  (a : ℕ → ℤ) (b : ℕ → ℤ) (x y : ℤ)
  (a_1 : a 1 = 2) (b_1 : b 1 = 3)
  (a_formula : ∀ n : ℕ, a n = 2 + (n - 1) * x)
  (b_formula : ∀ n : ℕ, b n = 3 + (n - 1) * y)
  (x_lt_y : x < y)
  (product_condition : ∃ n : ℕ, a n * b n = 1638) :
  ∃ n : ℕ, a n * b n = 1638 ∧ n = 35 := 
sorry

end NUMINAMATH_GPT_largest_n_arithmetic_sequences_l1368_136855


namespace NUMINAMATH_GPT_trigonometric_expression_proof_l1368_136841

theorem trigonometric_expression_proof :
  (Real.cos (76 * Real.pi / 180) * Real.cos (16 * Real.pi / 180) +
   Real.cos (14 * Real.pi / 180) * Real.cos (74 * Real.pi / 180) -
   2 * Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_proof_l1368_136841


namespace NUMINAMATH_GPT_radius_large_circle_l1368_136825

/-- Definitions for the problem context -/
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

noncomputable def tangent_circles (c1 c2 : Circle) : Prop :=
dist c1.center c2.center = c1.radius + c2.radius

/-- Theorem to prove the radius of the large circle -/
theorem radius_large_circle 
  (small_circle : Circle)
  (h_radius : small_circle.radius = 2)
  (large_circle : Circle)
  (h_tangency1 : tangent_circles small_circle large_circle)
  (small_circle2 : Circle)
  (small_circle3 : Circle)
  (h_tangency2 : tangent_circles small_circle small_circle2)
  (h_tangency3 : tangent_circles small_circle small_circle3)
  (h_tangency4 : tangent_circles small_circle2 large_circle)
  (h_tangency5 : tangent_circles small_circle3 large_circle)
  (h_tangency6 : tangent_circles small_circle2 small_circle3)
  : large_circle.radius = 2 * (Real.sqrt 3 + 1) :=
sorry

end NUMINAMATH_GPT_radius_large_circle_l1368_136825


namespace NUMINAMATH_GPT_find_M_base7_l1368_136835

theorem find_M_base7 :
  ∃ M : ℕ, M = 48 ∧ (M^2).digits 7 = [6, 6] ∧ (∃ (m : ℕ), 49 ≤ m^2 ∧ m^2 < 343 ∧ M = m - 1) :=
sorry

end NUMINAMATH_GPT_find_M_base7_l1368_136835


namespace NUMINAMATH_GPT_sequence_diff_l1368_136839

theorem sequence_diff (a : ℕ → ℕ) (S : ℕ → ℕ)
  (hSn : ∀ n, S n = n^2)
  (hS1 : a 1 = S 1)
  (ha_n : ∀ n, a (n + 1) = S (n + 1) - S n) :
  a 3 - a 2 = 2 := sorry

end NUMINAMATH_GPT_sequence_diff_l1368_136839


namespace NUMINAMATH_GPT_find_k_l1368_136807

theorem find_k (k : ℕ) (h : 2 * 3 - k + 1 = 0) : k = 7 :=
sorry

end NUMINAMATH_GPT_find_k_l1368_136807


namespace NUMINAMATH_GPT_number_of_children_l1368_136887

theorem number_of_children (n m : ℕ) (h1 : 11 * (m + 6) + n * m = n^2 + 3 * n - 2) : n = 9 :=
sorry

end NUMINAMATH_GPT_number_of_children_l1368_136887


namespace NUMINAMATH_GPT_vertex_of_parabola_l1368_136852

theorem vertex_of_parabola (a b : ℝ) (roots_condition : ∀ x, -x^2 + a * x + b ≤ 0 ↔ (x ≤ -3 ∨ x ≥ 5)) :
  ∃ v : ℝ × ℝ, v = (1, 16) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1368_136852


namespace NUMINAMATH_GPT_largest_3_digit_sum_l1368_136822

-- Defining the condition that ensures X, Y, Z are different digits ranging from 0 to 9
def valid_digits (X Y Z : ℕ) : Prop :=
  X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X < 10 ∧ Y < 10 ∧ Z < 10

-- Problem statement: Proving the largest possible 3-digit sum is 994
theorem largest_3_digit_sum : ∃ (X Y Z : ℕ), valid_digits X Y Z ∧ 111 * X + 11 * Y + Z = 994 :=
by
  sorry

end NUMINAMATH_GPT_largest_3_digit_sum_l1368_136822


namespace NUMINAMATH_GPT_max_acute_triangles_l1368_136810

theorem max_acute_triangles (n : ℕ) (hn : n ≥ 3) :
  (∃ k, k = if n % 2 = 0 then (n * (n-2) * (n+2)) / 24 else (n * (n-1) * (n+1)) / 24) :=
by 
  sorry

end NUMINAMATH_GPT_max_acute_triangles_l1368_136810


namespace NUMINAMATH_GPT_valid_odd_and_increasing_functions_l1368_136826

   def is_odd_function (f : ℝ → ℝ) : Prop :=
     ∀ x, f (-x) = -f (x)

   def is_increasing_function (f : ℝ → ℝ) : Prop :=
     ∀ x y, x < y → f (x) < f (y)

   noncomputable def f1 (x : ℝ) : ℝ := 3 * x^2
   noncomputable def f2 (x : ℝ) : ℝ := 6 * x
   noncomputable def f3 (x : ℝ) : ℝ := x * abs x
   noncomputable def f4 (x : ℝ) : ℝ := x + 1 / x

   theorem valid_odd_and_increasing_functions :
     (is_odd_function f2 ∧ is_increasing_function f2) ∧
     (is_odd_function f3 ∧ is_increasing_function f3) :=
   by
     sorry -- Proof goes here
   
end NUMINAMATH_GPT_valid_odd_and_increasing_functions_l1368_136826


namespace NUMINAMATH_GPT_marie_socks_problem_l1368_136850

theorem marie_socks_problem (x y z : ℕ) : 
  x + y + z = 15 → 
  2 * x + 3 * y + 5 * z = 36 → 
  1 ≤ x → 
  1 ≤ y → 
  1 ≤ z → 
  x = 11 :=
by
  sorry

end NUMINAMATH_GPT_marie_socks_problem_l1368_136850


namespace NUMINAMATH_GPT_convince_jury_l1368_136854

def not_guilty : Prop := sorry  -- definition indicating the defendant is not guilty
def not_liar : Prop := sorry    -- definition indicating the defendant is not a liar
def innocent_knight_statement : Prop := sorry  -- statement "I am an innocent knight"

theorem convince_jury (not_guilty : not_guilty) (not_liar : not_liar) : innocent_knight_statement :=
sorry

end NUMINAMATH_GPT_convince_jury_l1368_136854


namespace NUMINAMATH_GPT_cannot_be_value_of_A_plus_P_l1368_136866

theorem cannot_be_value_of_A_plus_P (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (a_neq_b: a ≠ b) :
  let A : ℕ := a * b
  let P : ℕ := 2 * a + 2 * b
  A + P ≠ 102 :=
by
  sorry

end NUMINAMATH_GPT_cannot_be_value_of_A_plus_P_l1368_136866


namespace NUMINAMATH_GPT_train_speed_problem_l1368_136878

open Real

/-- Given specific conditions about the speeds and lengths of trains, prove the speed of the third train is 99 kmph. -/
theorem train_speed_problem
  (man_train_speed_kmph : ℝ)
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (goods_train_time : ℝ)
  (third_train_length : ℝ)
  (third_train_time : ℝ) :
  man_train_speed_kmph = 45 →
  man_train_speed = 45 * 1000 / 3600 →
  goods_train_length = 340 →
  goods_train_time = 8 →
  third_train_length = 480 →
  third_train_time = 12 →
  (third_train_length / third_train_time - man_train_speed) * 3600 / 1000 = 99 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_train_speed_problem_l1368_136878


namespace NUMINAMATH_GPT_probability_no_intersecting_chords_l1368_136817

open Nat

def double_factorial (n : Nat) : Nat :=
  if n = 0 ∨ n = 1 then 1 else n * double_factorial (n - 2)

def catalan_number (n : Nat) : Nat :=
  (factorial (2 * n)) / (factorial n * factorial (n + 1))

theorem probability_no_intersecting_chords (n : Nat) (h : n > 0) :
  (catalan_number n) / (double_factorial (2 * n - 1)) = 2^n / (factorial (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_probability_no_intersecting_chords_l1368_136817


namespace NUMINAMATH_GPT_ratio_to_percent_l1368_136830

theorem ratio_to_percent (a b : ℕ) (h : a = 6) (h2 : b = 3) :
  ((a / b : ℚ) * 100 = 200) :=
by
  have h3 : a = 6 := h
  have h4 : b = 3 := h2
  sorry

end NUMINAMATH_GPT_ratio_to_percent_l1368_136830


namespace NUMINAMATH_GPT_area_N1N2N3_relative_l1368_136827

-- Definitions
variable (A B C D E F N1 N2 N3 : Type)
-- Assuming D, E, F are points on sides BC, CA, AB respectively such that CD, AE, BF are one-fourth of their respective sides.
variable (area_ABC : ℝ)  -- Total area of triangle ABC
variable (area_N1N2N3 : ℝ)  -- Area of triangle N1N2N3

-- Given conditions
variable (H1 : CD = 1 / 4 * BC)
variable (H2 : AE = 1 / 4 * CA)
variable (H3 : BF = 1 / 4 * AB)

-- The expected result
theorem area_N1N2N3_relative :
  area_N1N2N3 = 7 / 15 * area_ABC :=
sorry

end NUMINAMATH_GPT_area_N1N2N3_relative_l1368_136827


namespace NUMINAMATH_GPT_length_of_FD_l1368_136864

/-- In a square of side length 8 cm, point E is located on side AD,
2 cm from A and 6 cm from D. Point F lies on side CD such that folding
the square so that C coincides with E creates a crease along GF. 
Prove that the length of segment FD is 7/4 cm. -/
theorem length_of_FD (x : ℝ) (h_square : ∀ (A B C D : ℝ), A = 8 ∧ B = 8 ∧ C = 8 ∧ D = 8)
    (h_AE : ∀ (A E : ℝ), A - E = 2) (h_ED : ∀ (E D : ℝ), E - D = 6)
    (h_pythagorean : ∀ (x : ℝ), (8 - x)^2 = x^2 + 6^2) : x = 7/4 :=
by
  sorry

end NUMINAMATH_GPT_length_of_FD_l1368_136864


namespace NUMINAMATH_GPT_sin_squared_plus_one_l1368_136823

theorem sin_squared_plus_one (x : ℝ) (hx : Real.tan x = 2) : Real.sin x ^ 2 + 1 = 9 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_sin_squared_plus_one_l1368_136823


namespace NUMINAMATH_GPT_no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l1368_136845

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end NUMINAMATH_GPT_no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l1368_136845


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1368_136898

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def f (x : ℝ) : ℝ :=
  dot_product (Real.cos x, Real.cos x) (Real.sqrt 3 * Real.cos x, Real.sin x)

theorem problem_part1 :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, (x ∈ Set.Icc (k * π + π / 12) (k * π + 7 * π / 12)) → MonotoneOn f (Set.Icc (k * π + π / 12) (k * π + 7 * π / 12))) :=
sorry

theorem problem_part2 (A : ℝ) (a b c : ℝ) (area : ℝ) :
  f (A / 2 - π / 6) = Real.sqrt 3 ∧ 
  c = 2 ∧ 
  area = 2 * Real.sqrt 3 →
  a = 2 * Real.sqrt 3 ∨ a = 2 * Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1368_136898


namespace NUMINAMATH_GPT_initial_goldfish_correct_l1368_136893

-- Define the constants related to the conditions
def weekly_die := 5
def weekly_purchase := 3
def final_goldfish := 4
def weeks := 7

-- Define the initial number of goldfish that we need to prove
def initial_goldfish := 18

-- The proof statement: initial_goldfish - weekly_change * weeks = final_goldfish
theorem initial_goldfish_correct (G : ℕ)
  (h : G - weeks * (weekly_purchase - weekly_die) = final_goldfish) :
  G = initial_goldfish := by
  sorry

end NUMINAMATH_GPT_initial_goldfish_correct_l1368_136893


namespace NUMINAMATH_GPT_river_bank_depth_l1368_136865

-- Definitions related to the problem
def is_trapezium (top_width bottom_width height area : ℝ) :=
  area = 1 / 2 * (top_width + bottom_width) * height

-- The theorem we want to prove
theorem river_bank_depth :
  ∀ (top_width bottom_width area : ℝ), 
    top_width = 12 → 
    bottom_width = 8 → 
    area = 500 → 
    ∃ h : ℝ, is_trapezium top_width bottom_width h area ∧ h = 50 :=
by
  intros top_width bottom_width area ht hb ha
  sorry

end NUMINAMATH_GPT_river_bank_depth_l1368_136865


namespace NUMINAMATH_GPT_find_d_in_triangle_ABC_l1368_136806

theorem find_d_in_triangle_ABC (AB BC AC : ℝ) (P : Type) (d : ℝ) 
  (h_AB : AB = 480) (h_BC : BC = 500) (h_AC : AC = 550)
  (h_segments_equal : ∀ (D D' E E' F F' : Type), true) : 
  d = 132000 / 654 :=
sorry

end NUMINAMATH_GPT_find_d_in_triangle_ABC_l1368_136806


namespace NUMINAMATH_GPT_farmer_initial_tomatoes_l1368_136828

theorem farmer_initial_tomatoes 
  (T : ℕ) -- The initial number of tomatoes
  (picked : ℕ)   -- The number of tomatoes picked
  (diff : ℕ) -- The difference between initial number of tomatoes and picked
  (h1 : picked = 9) -- The farmer picked 9 tomatoes
  (h2 : diff = 8) -- The difference is 8
  (h3 : T - picked = diff) -- T - 9 = 8
  :
  T = 17 := sorry

end NUMINAMATH_GPT_farmer_initial_tomatoes_l1368_136828


namespace NUMINAMATH_GPT_find_b_l1368_136885

theorem find_b (x : ℝ) (b : ℝ) :
  (∃ t u : ℝ, (bx^2 + 18 * x + 9) = (t * x + u)^2 ∧ u^2 = 9 ∧ 2 * t * u = 18 ∧ t^2 = b) →
  b = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1368_136885


namespace NUMINAMATH_GPT_general_equation_of_line_cartesian_equation_of_curve_minimum_AB_l1368_136874

open Real

-- Definitions for the problem
variable (t : ℝ) (φ θ : ℝ) (x y P : ℝ)

-- Conditions
def line_parametric := x = t * sin φ ∧ y = 1 + t * cos φ
def curve_polar := P * (cos θ)^2 = 4 * sin θ
def curve_cartesian := x^2 = 4 * y
def line_general := x * cos φ - y * sin φ + sin φ = 0

-- Proof problem statements

-- 1. Prove the general equation of line l
theorem general_equation_of_line (h : line_parametric t φ x y) : line_general φ x y :=
sorry

-- 2. Prove the cartesian coordinate equation of curve C
theorem cartesian_equation_of_curve (h : curve_polar P θ) : curve_cartesian x y :=
sorry

-- 3. Prove the minimum |AB| where line l intersects curve C
theorem minimum_AB (h_line : line_parametric t φ x y) (h_curve : curve_cartesian x y) : ∃ (min_ab : ℝ), min_ab = 4 :=
sorry

end NUMINAMATH_GPT_general_equation_of_line_cartesian_equation_of_curve_minimum_AB_l1368_136874


namespace NUMINAMATH_GPT_rectangle_width_length_ratio_l1368_136818

theorem rectangle_width_length_ratio (w l P : ℕ) (h_l : l = 10) (h_P : P = 30) (h_perimeter : 2*w + 2*l = P) :
  w / l = 1 / 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_width_length_ratio_l1368_136818


namespace NUMINAMATH_GPT_goldfish_count_15_weeks_l1368_136886

def goldfish_count_after_weeks (initial : ℕ) (weeks : ℕ) : ℕ :=
  let deaths := λ n => 10 + 2 * (n - 1)
  let purchases := λ n => 5 + 2 * (n - 1)
  let rec update_goldfish (current : ℕ) (week : ℕ) :=
    if week = 0 then current
    else 
      let new_count := current - deaths week + purchases week
      update_goldfish new_count (week - 1)
  update_goldfish initial weeks

theorem goldfish_count_15_weeks : goldfish_count_after_weeks 35 15 = 15 :=
  by
  sorry

end NUMINAMATH_GPT_goldfish_count_15_weeks_l1368_136886


namespace NUMINAMATH_GPT_reflection_matrix_condition_l1368_136803

noncomputable def reflection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![-(3/4 : ℝ), 1/4]]

noncomputable def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![0, 1]]

theorem reflection_matrix_condition (a b : ℝ) :
  (reflection_matrix a b)^2 = identity_matrix ↔ a = -(1/4) ∧ b = -(3/4) :=
  by
  sorry

end NUMINAMATH_GPT_reflection_matrix_condition_l1368_136803


namespace NUMINAMATH_GPT_managers_participation_l1368_136815

theorem managers_participation (teams : ℕ) (people_per_team : ℕ) (employees : ℕ) (total_people : teams * people_per_team = 6) (num_employees : employees = 3) :
  teams * people_per_team - employees = 3 :=
by
  sorry

end NUMINAMATH_GPT_managers_participation_l1368_136815


namespace NUMINAMATH_GPT_part_I_part_II_l1368_136801

def setA (x : ℝ) : Prop := 0 ≤ x - 1 ∧ x - 1 ≤ 2

def setB (x : ℝ) (a : ℝ) : Prop := 1 < x - a ∧ x - a < 2 * a + 3

def complement_R (x : ℝ) (a : ℝ) : Prop := x ≤ 2 ∨ x ≥ 6

theorem part_I (a : ℝ) (x : ℝ) (ha : a = 1) : 
  setA x ∨ setB x a ↔ (1 ≤ x ∧ x < 6) ∧ 
  (setA x ∧ complement_R x a ↔ 1 ≤ x ∧ x ≤ 2) := 
by
  sorry

theorem part_II (a : ℝ) : 
  (∃ x, setA x ∧ setB x a) ↔ -2/3 < a ∧ a < 2 := 
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1368_136801


namespace NUMINAMATH_GPT_problem_1_problem_2_l1368_136888

-- Problem (1)
theorem problem_1 (a c : ℝ) (h1 : ∀ x, -1/3 < x ∧ x < 1/2 → ax^2 + 2*x + c > 0) :
  ∃ s, s = { x | -2 < x ∧ x < 3 } ∧ (∀ x, x ∈ s → cx^2 - 2*x + a < 0) := 
sorry

-- Problem (2)
theorem problem_2 (m : ℝ) (h : ∀ x : ℝ, x > 0 → x^2 - m*x + 4 > 0) :
  m < 4 := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1368_136888


namespace NUMINAMATH_GPT_part1_part2_l1368_136809

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 1

theorem part1 :
  ∀ x : ℝ, f (x + π) = f x :=
by sorry

theorem part2 :
  ∃ (max_x min_x : ℝ), max_x ∈ Set.Icc (π/12) (π/4) ∧ min_x ∈ Set.Icc (π/12) (π/4) ∧
    f max_x = 7 / 4 ∧ f min_x = (5 + Real.sqrt 3) / 4 ∧
    (max_x = π / 6) ∧ (min_x = π / 12 ∨ min_x = π / 4) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1368_136809


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1368_136880

variables (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
variables (c e : ℝ)

-- Define the eccentricy condition for hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_eccentricity :
  -- Conditions regarding the hyperbola and the distances
  (∀ x y : ℝ, hyperbola a b x y → 
    (∃ x y : ℝ, y = (2 / 3) * c ∧ x = 2 * a + (2 / 3) * c ∧
    ((2 / 3) * c)^2 + (2 * a + (2 / 3) * c)^2 = 4 * c^2 ∧
    (7 * e^2 - 6 * e - 9 = 0))) →
  -- Proving that the eccentricity e is as given
  e = (3 + Real.sqrt 6) / 7 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1368_136880


namespace NUMINAMATH_GPT_tan_add_pi_over_4_sin_over_expression_l1368_136890

variable (α : ℝ)

theorem tan_add_pi_over_4 (h : Real.tan α = 2) : 
  Real.tan (α + π / 4) = -3 := 
  sorry

theorem sin_over_expression (h : Real.tan α = 2) : 
  (Real.sin (2 * α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1) = 1 := 
  sorry

end NUMINAMATH_GPT_tan_add_pi_over_4_sin_over_expression_l1368_136890


namespace NUMINAMATH_GPT_parallel_lines_k_l1368_136804

theorem parallel_lines_k (k : ℝ) 
  (h₁ : k ≠ 0)
  (h₂ : ∀ x y : ℝ, (x - k * y - k = 0) = (y = (1 / k) * x - 1))
  (h₃ : ∀ x : ℝ, (y = k * (x - 1))) :
  k = -1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_k_l1368_136804
