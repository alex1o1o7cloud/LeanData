import Mathlib

namespace NUMINAMATH_GPT_multiply_polynomials_l2011_201170

theorem multiply_polynomials (x : ℝ) : 
  (x^6 + 64 * x^3 + 4096) * (x^3 - 64) = x^9 - 262144 :=
by
  sorry

end NUMINAMATH_GPT_multiply_polynomials_l2011_201170


namespace NUMINAMATH_GPT_part_a_part_b_l2011_201193

namespace TrihedralAngle

-- Part (a)
theorem part_a (α β γ : ℝ) (h1 : β = 70) (h2 : γ = 100) (h3 : α < β + γ) (h4 : β < α + γ) (h5 : γ < α + β) (h6 : α + β + γ < 360) : 
    30 < α ∧ α < 170 := 
sorry

-- Part (b)
theorem part_b (α β γ : ℝ) (h1 : β = 130) (h2 : γ = 150) (h3 : α < β + γ) (h4 : β < α + γ) (h5 : γ < α + β) (h6 : α + β + γ < 360) : 
    20 < α ∧ α < 80 := 
sorry

end TrihedralAngle

end NUMINAMATH_GPT_part_a_part_b_l2011_201193


namespace NUMINAMATH_GPT_largest_fraction_of_three_l2011_201174

theorem largest_fraction_of_three (a b c : Nat) (h1 : Nat.gcd a 6 = 1)
  (h2 : Nat.gcd b 15 = 1) (h3 : Nat.gcd c 20 = 1)
  (h4 : (a * b * c) = 60) :
  max (a / 6) (max (b / 15) (c / 20)) = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_largest_fraction_of_three_l2011_201174


namespace NUMINAMATH_GPT_packages_per_box_l2011_201133

theorem packages_per_box (P : ℕ) (h1 : 192 > 0) (h2 : 2 > 0) (total_soaps : 2304 > 0) (h : 2 * P * 192 = 2304) : P = 6 :=
by
  sorry

end NUMINAMATH_GPT_packages_per_box_l2011_201133


namespace NUMINAMATH_GPT_find_f_19_l2011_201172

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the given function

-- Define the conditions
axiom even_function : ∀ x : ℝ, f x = f (-x) 
axiom periodicity : ∀ x : ℝ, f (x + 2) = -f x

-- The statement we need to prove
theorem find_f_19 : f 19 = 0 := 
by
  sorry -- placeholder for the proof

end NUMINAMATH_GPT_find_f_19_l2011_201172


namespace NUMINAMATH_GPT_correct_factorization_l2011_201109

theorem correct_factorization (a : ℝ) : 
  (a ^ 2 + 4 * a ≠ a ^ 2 * (a + 4)) ∧ 
  (a ^ 2 - 9 ≠ (a + 9) * (a - 9)) ∧ 
  (a ^ 2 + 4 * a + 2 ≠ (a + 2) ^ 2) → 
  (a ^ 2 - 2 * a + 1 = (a - 1) ^ 2) :=
by sorry

end NUMINAMATH_GPT_correct_factorization_l2011_201109


namespace NUMINAMATH_GPT_combined_stickers_count_l2011_201108

theorem combined_stickers_count :
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given
  june_total + bonnie_total = 189 :=
by
  -- Definitions
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  
  -- Calculations
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given

  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_combined_stickers_count_l2011_201108


namespace NUMINAMATH_GPT_price_of_tray_l2011_201175

noncomputable def price_per_egg : ℕ := 50
noncomputable def tray_eggs : ℕ := 30
noncomputable def discount_per_egg : ℕ := 10

theorem price_of_tray : (price_per_egg - discount_per_egg) * tray_eggs / 100 = 12 :=
by
  sorry

end NUMINAMATH_GPT_price_of_tray_l2011_201175


namespace NUMINAMATH_GPT_sheep_per_herd_l2011_201128

theorem sheep_per_herd (herds : ℕ) (total_sheep : ℕ) (h_herds : herds = 3) (h_total_sheep : total_sheep = 60) : 
  (total_sheep / herds) = 20 :=
by
  sorry

end NUMINAMATH_GPT_sheep_per_herd_l2011_201128


namespace NUMINAMATH_GPT_necessarily_positive_y_plus_z_l2011_201120

theorem necessarily_positive_y_plus_z
  (x y z : ℝ)
  (hx : 0 < x ∧ x < 0.5) 
  (hy : -0.5 < y ∧ y < 0) 
  (hz : 0.5 < z ∧ z < 1) :
  y + z > 0 := 
by
  sorry

end NUMINAMATH_GPT_necessarily_positive_y_plus_z_l2011_201120


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2011_201149

theorem solution_set_of_inequality (x : ℝ) : x * (x + 2) ≥ 0 ↔ x ≤ -2 ∨ x ≥ 0 := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2011_201149


namespace NUMINAMATH_GPT_investment_simple_compound_l2011_201155

theorem investment_simple_compound (P y : ℝ) 
    (h1 : 600 = P * y * 2 / 100)
    (h2 : 615 = P * (1 + y/100)^2 - P) : 
    P = 285.71 :=
by
    sorry

end NUMINAMATH_GPT_investment_simple_compound_l2011_201155


namespace NUMINAMATH_GPT_problem_l2011_201196

variable (x : ℝ) (Q : ℝ)

theorem problem (h : 2 * (5 * x + 3 * Real.pi) = Q) : 4 * (10 * x + 6 * Real.pi + 2) = 4 * Q + 8 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2011_201196


namespace NUMINAMATH_GPT_interior_angle_regular_octagon_exterior_angle_regular_octagon_l2011_201197

-- Definitions
def sumInteriorAngles (n : ℕ) : ℕ := 180 * (n - 2)
def oneInteriorAngle (n : ℕ) (sumInterior : ℕ) : ℕ := sumInterior / n
def sumExteriorAngles : ℕ := 360
def oneExteriorAngle (n : ℕ) (sumExterior : ℕ) : ℕ := sumExterior / n

-- Theorem statements
theorem interior_angle_regular_octagon : oneInteriorAngle 8 (sumInteriorAngles 8) = 135 := by sorry

theorem exterior_angle_regular_octagon : oneExteriorAngle 8 sumExteriorAngles = 45 := by sorry

end NUMINAMATH_GPT_interior_angle_regular_octagon_exterior_angle_regular_octagon_l2011_201197


namespace NUMINAMATH_GPT_new_person_weight_l2011_201131

theorem new_person_weight
    (avg_weight_20 : ℕ → ℕ)
    (total_weight_20 : ℕ)
    (avg_weight_21 : ℕ)
    (count_20 : ℕ)
    (count_21 : ℕ) :
    avg_weight_20 count_20 = 58 →
    total_weight_20 = count_20 * avg_weight_20 count_20 →
    avg_weight_21 = 53 →
    count_21 = count_20 + 1 →
    ∃ (W : ℕ), total_weight_20 + W = count_21 * avg_weight_21 ∧ W = 47 := 
by 
  sorry

end NUMINAMATH_GPT_new_person_weight_l2011_201131


namespace NUMINAMATH_GPT_mike_games_l2011_201144

theorem mike_games (initial_money spent_money game_cost remaining_games : ℕ)
  (h1 : initial_money = 101)
  (h2 : spent_money = 47)
  (h3 : game_cost = 6)
  (h4 : remaining_games = (initial_money - spent_money) / game_cost) :
  remaining_games = 9 := by
  sorry

end NUMINAMATH_GPT_mike_games_l2011_201144


namespace NUMINAMATH_GPT_find_base_l2011_201192

theorem find_base (b : ℕ) : (b^3 ≤ 64 ∧ 64 < b^4) ↔ b = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_base_l2011_201192


namespace NUMINAMATH_GPT_find_x_perpendicular_l2011_201195

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 3)
def b (x : ℝ) : ℝ × ℝ := (-3, x)

-- Define the condition that the dot product of vectors a and b is zero
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Statement we need to prove
theorem find_x_perpendicular (x : ℝ) (h : perpendicular a (b x)) : x = -1 :=
by sorry

end NUMINAMATH_GPT_find_x_perpendicular_l2011_201195


namespace NUMINAMATH_GPT_multiple_of_kids_finishing_early_l2011_201182

-- Definitions based on conditions
def num_10_percent_kids (total_kids : ℕ) : ℕ := (total_kids * 10) / 100

def num_remaining_kids (total_kids kids_less_6 kids_more_14 : ℕ) : ℕ := total_kids - kids_less_6 - kids_more_14

def num_multiple_finishing_less_8 (total_kids : ℕ) (multiple : ℕ) : ℕ := multiple * num_10_percent_kids total_kids

-- Main theorem statement
theorem multiple_of_kids_finishing_early 
  (total_kids : ℕ)
  (h_total_kids : total_kids = 40)
  (kids_more_14 : ℕ)
  (h_kids_more_14 : kids_more_14 = 4)
  (h_1_6_remaining : kids_more_14 = num_remaining_kids total_kids (num_10_percent_kids total_kids) kids_more_14 / 6)
  : (num_multiple_finishing_less_8 total_kids 3) = (total_kids - num_10_percent_kids total_kids - kids_more_14) := 
by 
  sorry

end NUMINAMATH_GPT_multiple_of_kids_finishing_early_l2011_201182


namespace NUMINAMATH_GPT_area_of_triangle_l2011_201177

open Real

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : sin A = sqrt 3 * sin C)
                        (h2 : B = π / 6) (h3 : b = 2) :
    1 / 2 * a * c * sin B = sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l2011_201177


namespace NUMINAMATH_GPT_find_x_l2011_201150

theorem find_x (x : ℝ) (h : 49 / x = 700) : x = 0.07 :=
sorry

end NUMINAMATH_GPT_find_x_l2011_201150


namespace NUMINAMATH_GPT_mean_of_three_is_90_l2011_201153

-- Given conditions as Lean definitions
def mean_twelve (s : ℕ) : Prop := s = 12 * 40
def added_sum (x y z : ℕ) (s : ℕ) : Prop := s + x + y + z = 15 * 50
def z_value (x z : ℕ) : Prop := z = x + 10

-- Theorem statement to prove the mean of x, y, and z is 90
theorem mean_of_three_is_90 (x y z s : ℕ) : 
  (mean_twelve s) → (z_value x z) → (added_sum x y z s) → 
  (x + y + z) / 3 = 90 := 
by 
  intros h1 h2 h3 
  sorry

end NUMINAMATH_GPT_mean_of_three_is_90_l2011_201153


namespace NUMINAMATH_GPT_radius_of_spherical_circle_correct_l2011_201168

noncomputable def radius_of_spherical_circle (rho theta phi : ℝ) : ℝ :=
  if rho = 1 ∧ phi = Real.pi / 4 then Real.sqrt 2 / 2 else 0

theorem radius_of_spherical_circle_correct :
  ∀ (theta : ℝ), radius_of_spherical_circle 1 theta (Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_spherical_circle_correct_l2011_201168


namespace NUMINAMATH_GPT_cylinder_volume_l2011_201115

-- Definitions based on conditions
def lateral_surface_to_rectangle (generatrix_a generatrix_b : ℝ) (volume : ℝ) :=
  -- Condition: Rectangle with sides 8π and 4π
  (generatrix_a = 8 * Real.pi ∧ volume = 32 * Real.pi^2) ∨
  (generatrix_a = 4 * Real.pi ∧ volume = 64 * Real.pi^2)

-- Statement
theorem cylinder_volume (generatrix_a generatrix_b : ℝ)
  (h : (generatrix_a = 8 * Real.pi ∨ generatrix_b = 4 * Real.pi) ∧ (generatrix_b = 4 * Real.pi ∨ generatrix_b = 8 * Real.pi)) :
  ∃ (volume : ℝ), lateral_surface_to_rectangle generatrix_a generatrix_b volume :=
sorry

end NUMINAMATH_GPT_cylinder_volume_l2011_201115


namespace NUMINAMATH_GPT_original_price_of_article_l2011_201184

theorem original_price_of_article (P : ℝ) : 
  (P - 0.30 * P) * (1 - 0.20) = 1120 → P = 2000 :=
by
  intro h
  -- h represents the given condition for the problem
  sorry  -- proof will go here

end NUMINAMATH_GPT_original_price_of_article_l2011_201184


namespace NUMINAMATH_GPT_domain_of_composite_l2011_201126

theorem domain_of_composite (f : ℝ → ℝ) (x : ℝ) (hf : ∀ y, (0 ≤ y ∧ y ≤ 1) → f y = f y) :
  (0 ≤ x ∧ x ≤ 1) → (0 ≤ x ∧ x ≤ 1) → (0 ≤ x ∧ x ≤ 1) →
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ 2*x ∧ 2*x ≤ 1 ∧ 0 ≤ x + 1/3 ∧ x + 1/3 ≤ 1 →
  0 ≤ x ∧ x ≤ 1/2 :=
by
  intro h1 h2 h3 h4
  have h5: 0 ≤ 2*x ∧ 2*x ≤ 1 := sorry
  have h6: 0 ≤ x + 1/3 ∧ x + 1/3 ≤ 1 := sorry
  sorry

end NUMINAMATH_GPT_domain_of_composite_l2011_201126


namespace NUMINAMATH_GPT_selection_methods_correct_l2011_201161

-- Define the number of students in each year
def first_year_students : ℕ := 3
def second_year_students : ℕ := 5
def third_year_students : ℕ := 4

-- Define the total number of different selection methods
def total_selection_methods : ℕ := first_year_students + second_year_students + third_year_students

-- Lean statement to prove the question is equivalent to the answer
theorem selection_methods_correct :
  total_selection_methods = 12 := by
  sorry

end NUMINAMATH_GPT_selection_methods_correct_l2011_201161


namespace NUMINAMATH_GPT_train_cross_time_platform_l2011_201100

def speed := 36 -- in kmph
def time_for_pole := 12 -- in seconds
def time_for_platform := 44.99736021118311 -- in seconds

theorem train_cross_time_platform :
  time_for_platform = 44.99736021118311 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_time_platform_l2011_201100


namespace NUMINAMATH_GPT_tetrahedron_equal_reciprocal_squares_l2011_201111

noncomputable def tet_condition_heights (h_1 h_2 h_3 h_4 : ℝ) : Prop :=
True

noncomputable def tet_condition_distances (d_1 d_2 d_3 : ℝ) : Prop :=
True

theorem tetrahedron_equal_reciprocal_squares
  (h_1 h_2 h_3 h_4 d_1 d_2 d_3 : ℝ)
  (hc_hts : tet_condition_heights h_1 h_2 h_3 h_4)
  (hc_dsts : tet_condition_distances d_1 d_2 d_3) :
  1 / (h_1 ^ 2) + 1 / (h_2 ^ 2) + 1 / (h_3 ^ 2) + 1 / (h_4 ^ 2) =
  1 / (d_1 ^ 2) + 1 / (d_2 ^ 2) + 1 / (d_3 ^ 2) :=
sorry

end NUMINAMATH_GPT_tetrahedron_equal_reciprocal_squares_l2011_201111


namespace NUMINAMATH_GPT_value_of_C_l2011_201187

theorem value_of_C (k : ℝ) (C : ℝ) (h : k = 0.4444444444444444) :
  (2 * k * 0 ^ 2 + 6 * k * 0 + C = 0) ↔ C = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_C_l2011_201187


namespace NUMINAMATH_GPT_tan_2016_l2011_201119

-- Define the given condition
def sin_36 (a : ℝ) : Prop := Real.sin (36 * Real.pi / 180) = a

-- Prove the required statement given the condition
theorem tan_2016 (a : ℝ) (h : sin_36 a) : Real.tan (2016 * Real.pi / 180) = a / Real.sqrt (1 - a^2) :=
sorry

end NUMINAMATH_GPT_tan_2016_l2011_201119


namespace NUMINAMATH_GPT_problem_l2011_201138

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition

theorem problem :
  (∀ x, f (-x) = -f x) → -- f is odd
  (∀ x, f (x + 2) = -1 / f x) → -- Functional equation
  (∀ x, 0 < x ∧ x < 1 → f x = 3 ^ x) → -- Definition on interval (0,1)
  f (Real.log (54) / Real.log 3) = -3 / 2 := sorry

end NUMINAMATH_GPT_problem_l2011_201138


namespace NUMINAMATH_GPT_sqrt_product_eq_sixty_sqrt_two_l2011_201141

theorem sqrt_product_eq_sixty_sqrt_two : (Real.sqrt 50) * (Real.sqrt 18) * (Real.sqrt 8) = 60 * (Real.sqrt 2) := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_product_eq_sixty_sqrt_two_l2011_201141


namespace NUMINAMATH_GPT_x_eq_y_sufficient_not_necessary_abs_l2011_201101

theorem x_eq_y_sufficient_not_necessary_abs (x y : ℝ) : (x = y → |x| = |y|) ∧ (|x| = |y| → x = y ∨ x = -y) :=
by {
  sorry
}

end NUMINAMATH_GPT_x_eq_y_sufficient_not_necessary_abs_l2011_201101


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2011_201157

namespace IntersectionProblem

def setA : Set ℝ := {0, 1, 2}
def setB : Set ℝ := {x | x^2 - x ≤ 0}
def intersection : Set ℝ := {0, 1}

theorem intersection_of_A_and_B : A ∩ B = intersection := sorry

end IntersectionProblem

end NUMINAMATH_GPT_intersection_of_A_and_B_l2011_201157


namespace NUMINAMATH_GPT_henry_needs_30_dollars_l2011_201162

def henry_action_figures_completion (current_figures total_figures cost_per_figure : ℕ) : ℕ :=
  (total_figures - current_figures) * cost_per_figure

theorem henry_needs_30_dollars : henry_action_figures_completion 3 8 6 = 30 := by
  sorry

end NUMINAMATH_GPT_henry_needs_30_dollars_l2011_201162


namespace NUMINAMATH_GPT_range_of_x_inequality_l2011_201194

theorem range_of_x_inequality (x : ℝ) (h : |2 * x - 1| + x + 3 ≤ 5) : -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_inequality_l2011_201194


namespace NUMINAMATH_GPT_find_number_l2011_201199

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def XiaoQian_statements (n : ℕ) : Prop :=
  is_perfect_square n ∧ n < 5

def XiaoLu_statements (n : ℕ) : Prop :=
  n < 7 ∧ 10 ≤ n ∧ n < 100

def XiaoDai_statements (n : ℕ) : Prop :=
  is_perfect_square n ∧ ¬ (n < 5)

theorem find_number :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 99 ∧ 
    ( (XiaoQian_statements n ∧ ¬XiaoLu_statements n ∧ ¬XiaoDai_statements n) ∨
      (¬XiaoQian_statements n ∧ XiaoLu_statements n ∧ ¬XiaoDai_statements n) ∨
      (¬XiaoQian_statements n ∧ ¬XiaoLu_statements n ∧ XiaoDai_statements n) ) ∧
    n = 9 :=
sorry

end NUMINAMATH_GPT_find_number_l2011_201199


namespace NUMINAMATH_GPT_Mike_given_total_cookies_l2011_201147

-- All given conditions
variables (total Tim fridge Mike Anna : Nat)
axiom h1 : total = 256
axiom h2 : Tim = 15
axiom h3 : fridge = 188
axiom h4 : Anna = 2 * Tim
axiom h5 : total = Tim + Anna + fridge + Mike

-- The goal of the proof
theorem Mike_given_total_cookies : Mike = 23 :=
by
  sorry

end NUMINAMATH_GPT_Mike_given_total_cookies_l2011_201147


namespace NUMINAMATH_GPT_wendy_pictures_l2011_201167

theorem wendy_pictures (album1_pics rest_albums albums each_album_pics : ℕ)
    (h1 : album1_pics = 44)
    (h2 : rest_albums = 5)
    (h3 : each_album_pics = 7)
    (h4 : albums = rest_albums * each_album_pics)
    (h5 : albums = 5 * 7):
  album1_pics + albums = 79 :=
by
  -- We leave the proof as an exercise
  sorry

end NUMINAMATH_GPT_wendy_pictures_l2011_201167


namespace NUMINAMATH_GPT_min_omega_value_l2011_201113

theorem min_omega_value (ω : ℝ) (hω : ω > 0)
  (f : ℝ → ℝ)
  (hf_def : ∀ x, f x = Real.cos (ω * x - (Real.pi / 6))) :
  (∀ x, f x ≤ f (Real.pi / 4)) → ω = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_omega_value_l2011_201113


namespace NUMINAMATH_GPT_num_valid_configurations_l2011_201124

-- Definitions used in the problem
def grid := (Fin 8) × (Fin 8)
def knights_tell_truth := true
def knaves_lie := true
def statement (i j : Fin 8) (r c : grid → ℕ) := (c ⟨0,j⟩ > r ⟨i,0⟩)

-- The theorem statement to prove
theorem num_valid_configurations : ∃ n : ℕ, n = 255 :=
sorry

end NUMINAMATH_GPT_num_valid_configurations_l2011_201124


namespace NUMINAMATH_GPT_tan_half_angle_l2011_201166

theorem tan_half_angle {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) :
  Real.tan ((α + β) / 2) = 1 + Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_tan_half_angle_l2011_201166


namespace NUMINAMATH_GPT_number_of_pink_cookies_l2011_201132

def total_cookies : ℕ := 86
def red_cookies : ℕ := 36

def pink_cookies (total red : ℕ) : ℕ := total - red

theorem number_of_pink_cookies : pink_cookies total_cookies red_cookies = 50 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pink_cookies_l2011_201132


namespace NUMINAMATH_GPT_gcd_18_30_is_6_gcd_18_30_is_even_l2011_201134

def gcd_18_30 : ℕ := Nat.gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 := by
  sorry

theorem gcd_18_30_is_even : Even gcd_18_30 := by
  sorry

end NUMINAMATH_GPT_gcd_18_30_is_6_gcd_18_30_is_even_l2011_201134


namespace NUMINAMATH_GPT_pages_share_units_digit_l2011_201121

def units_digit (n : Nat) : Nat :=
  n % 10

theorem pages_share_units_digit :
  (∃ (x_set : Finset ℕ), (∀ (x : ℕ), x ∈ x_set ↔ (1 ≤ x ∧ x ≤ 63 ∧ units_digit x = units_digit (64 - x))) ∧ x_set.card = 13) :=
by
  sorry

end NUMINAMATH_GPT_pages_share_units_digit_l2011_201121


namespace NUMINAMATH_GPT_solveForX_l2011_201112

theorem solveForX : ∃ (x : ℚ), x + 5/8 = 7/24 + 1/4 ∧ x = -1/12 := by
  sorry

end NUMINAMATH_GPT_solveForX_l2011_201112


namespace NUMINAMATH_GPT_manufacturing_cost_eq_210_l2011_201171

theorem manufacturing_cost_eq_210 (transport_cost : ℝ) (shoecount : ℕ) (selling_price : ℝ) (gain : ℝ) (M : ℝ) :
  transport_cost = 500 / 100 →
  shoecount = 100 →
  selling_price = 258 →
  gain = 0.20 →
  M = (selling_price / (1 + gain)) - (transport_cost) :=
by
  intros
  sorry

end NUMINAMATH_GPT_manufacturing_cost_eq_210_l2011_201171


namespace NUMINAMATH_GPT_first_term_geometric_sequence_l2011_201145

variable {a : ℕ → ℝ} -- Define the geometric sequence a_n
variable (q : ℝ) -- Define the common ratio q which is a real number

-- Conditions given in the problem
def geom_seq_first_term (a : ℕ → ℝ) (q : ℝ) :=
  a 3 = 2 ∧ a 4 = 4 ∧ (∀ n : ℕ, a (n+1) = a n * q)

-- Assert that if these conditions hold, then the first term is 1/2
theorem first_term_geometric_sequence (hq : geom_seq_first_term a q) : a 1 = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_first_term_geometric_sequence_l2011_201145


namespace NUMINAMATH_GPT_bus_driver_total_hours_l2011_201152

variables (R OT : ℕ)

-- Conditions
def regular_rate := 16
def overtime_rate := 28
def max_regular_hours := 40
def total_compensation := 864

-- Proof goal: total hours worked is 48
theorem bus_driver_total_hours :
  (regular_rate * R + overtime_rate * OT = total_compensation) →
  (R ≤ max_regular_hours) →
  (R + OT = 48) :=
by
  sorry

end NUMINAMATH_GPT_bus_driver_total_hours_l2011_201152


namespace NUMINAMATH_GPT_sequence_formula_l2011_201158

theorem sequence_formula (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 6)
  (h4 : a 4 = 10)
  (h5 : ∀ n > 0, a (n + 1) - a n = n + 1) :
  ∀ n, a n = n * (n + 1) / 2 :=
by 
  sorry

end NUMINAMATH_GPT_sequence_formula_l2011_201158


namespace NUMINAMATH_GPT_at_least_one_ge_two_l2011_201191

theorem at_least_one_ge_two (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 3) :
  (a + 1/b >= 2) ∨ (b + 1/c >= 2) ∨ (c + 1/a >= 2) :=
sorry

end NUMINAMATH_GPT_at_least_one_ge_two_l2011_201191


namespace NUMINAMATH_GPT_aqua_park_earnings_l2011_201164

def admission_cost : ℕ := 12
def tour_cost : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

theorem aqua_park_earnings :
  (group1_size * admission_cost + group1_size * tour_cost) + (group2_size * admission_cost) = 240 :=
by
  sorry

end NUMINAMATH_GPT_aqua_park_earnings_l2011_201164


namespace NUMINAMATH_GPT_football_team_practice_hours_l2011_201190

-- Definitions for each day's practice adjusted for weather events
def monday_hours : ℕ := 4
def tuesday_hours : ℕ := 5 - 1
def wednesday_hours : ℕ := 0
def thursday_hours : ℕ := 5
def friday_hours : ℕ := 3 + 2
def saturday_hours : ℕ := 4
def sunday_hours : ℕ := 0

-- Total practice hours calculation
def total_practice_hours : ℕ := 
  monday_hours + tuesday_hours + wednesday_hours + 
  thursday_hours + friday_hours + saturday_hours + 
  sunday_hours

-- Statement to prove
theorem football_team_practice_hours : total_practice_hours = 22 := by
  sorry

end NUMINAMATH_GPT_football_team_practice_hours_l2011_201190


namespace NUMINAMATH_GPT_round_robin_pairing_possible_l2011_201114

def players : Set String := {"A", "B", "C", "D", "E", "F"}

def is_pairing (pairs : List (String × String)) : Prop :=
  ∀ (p : String × String), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ players ∧ p.2 ∈ players

def unique_pairs (rounds : List (List (String × String))) : Prop :=
  ∀ r, r ∈ rounds → is_pairing r ∧ (∀ p1 p2, p1 ∈ r → p2 ∈ r → p1 ≠ p2 → p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)

def all_players_paired (rounds : List (List (String × String))) : Prop :=
  ∀ p, p ∈ players →
  (∀ q, q ∈ players → p ≠ q → 
    (∃ r, r ∈ rounds ∧ (p,q) ∈ r ∨ (q,p) ∈ r))

theorem round_robin_pairing_possible : 
  ∃ rounds, List.length rounds = 5 ∧ unique_pairs rounds ∧ all_players_paired rounds :=
  sorry

end NUMINAMATH_GPT_round_robin_pairing_possible_l2011_201114


namespace NUMINAMATH_GPT_box_volume_correct_l2011_201185

def volume_of_box (x : ℝ) : ℝ := (16 - 2 * x) * (12 - 2 * x) * x

theorem box_volume_correct {x : ℝ} (h1 : 1 ≤ x) (h2 : x ≤ 3) : 
  volume_of_box x = 4 * x^3 - 56 * x^2 + 192 * x := 
by 
  unfold volume_of_box 
  sorry

end NUMINAMATH_GPT_box_volume_correct_l2011_201185


namespace NUMINAMATH_GPT_upstream_travel_time_l2011_201137

-- Define the given conditions
def downstream_time := 1 -- 1 hour
def stream_speed := 3 -- 3 kmph
def boat_speed_still_water := 15 -- 15 kmph

-- Compute the downstream speed
def downstream_speed : Nat := boat_speed_still_water + stream_speed

-- Compute the distance covered downstream
def distance_downstream : Nat := downstream_speed * downstream_time

-- Compute the upstream speed
def upstream_speed : Nat := boat_speed_still_water - stream_speed

-- The goal is to prove the time it takes to cover the distance upstream is 1.5 hours
theorem upstream_travel_time : (distance_downstream : Real) / upstream_speed = 1.5 := by
  sorry

end NUMINAMATH_GPT_upstream_travel_time_l2011_201137


namespace NUMINAMATH_GPT_first_valve_time_l2011_201183

noncomputable def first_valve_filling_time (V1 V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ) : ℝ :=
  pool_capacity / V1

theorem first_valve_time :
  ∀ (V1 : ℝ) (V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ),
    V2 = V1 + 50 →
    V1 + V2 = pool_capacity / combined_time →
    combined_time = 48 →
    pool_capacity = 12000 →
    first_valve_filling_time V1 V2 pool_capacity combined_time / 60 = 2 :=
  
by
  intros V1 V2 pool_capacity combined_time h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_first_valve_time_l2011_201183


namespace NUMINAMATH_GPT_door_height_eight_l2011_201160

theorem door_height_eight (x : ℝ) (h w : ℝ) (H1 : w = x - 4) (H2 : h = x - 2) (H3 : x^2 = (x - 4)^2 + (x - 2)^2) : h = 8 :=
by
  sorry

end NUMINAMATH_GPT_door_height_eight_l2011_201160


namespace NUMINAMATH_GPT_francie_remaining_money_l2011_201122

noncomputable def total_savings_before_investment : ℝ :=
  (5 * 8) + (6 * 6) + 20

noncomputable def investment_return : ℝ :=
  0.05 * 10

noncomputable def total_savings_after_investment : ℝ :=
  total_savings_before_investment + investment_return

noncomputable def spent_on_clothes : ℝ :=
  total_savings_after_investment / 2

noncomputable def remaining_after_clothes : ℝ :=
  total_savings_after_investment - spent_on_clothes

noncomputable def amount_remaining : ℝ :=
  remaining_after_clothes - 35

theorem francie_remaining_money : amount_remaining = 13.25 := 
  sorry

end NUMINAMATH_GPT_francie_remaining_money_l2011_201122


namespace NUMINAMATH_GPT_ellipse_eccentricity_l2011_201107

theorem ellipse_eccentricity (m : ℝ) (e : ℝ) : 
  (∀ x y : ℝ, (x^2 / m) + (y^2 / 4) = 1) ∧ foci_y_axis ∧ e = 1 / 2 → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l2011_201107


namespace NUMINAMATH_GPT_repeated_digit_squares_l2011_201110

theorem repeated_digit_squares :
  {n : ℕ | ∃ d : Fin 10, n = d ^ 2 ∧ (∀ m < n, m % 10 = d % 10)} ⊆ {0, 1, 4, 9} := by
  sorry

end NUMINAMATH_GPT_repeated_digit_squares_l2011_201110


namespace NUMINAMATH_GPT_kayla_scores_on_sixth_level_l2011_201163

-- Define the sequence of points scored in each level
def points (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 3
  | 2 => 5
  | 3 => 8
  | 4 => 12
  | n + 5 => points (n + 4) + (n + 1) + 1

-- Statement to prove that Kayla scores 17 points on the sixth level
theorem kayla_scores_on_sixth_level : points 5 = 17 :=
by
  sorry

end NUMINAMATH_GPT_kayla_scores_on_sixth_level_l2011_201163


namespace NUMINAMATH_GPT_james_total_matches_l2011_201102

-- Define the conditions
def dozen : Nat := 12
def boxes_per_dozen : Nat := 5
def matches_per_box : Nat := 20

-- Calculate the expected number of matches
def expected_matches : Nat := boxes_per_dozen * dozen * matches_per_box

-- State the theorem to be proved
theorem james_total_matches : expected_matches = 1200 := by
  sorry

end NUMINAMATH_GPT_james_total_matches_l2011_201102


namespace NUMINAMATH_GPT_final_song_count_l2011_201159

theorem final_song_count {init_songs added_songs removed_songs doubled_songs final_songs : ℕ} 
    (h1 : init_songs = 500)
    (h2 : added_songs = 500)
    (h3 : doubled_songs = (init_songs + added_songs) * 2)
    (h4 : removed_songs = 50)
    (h_final : final_songs = doubled_songs - removed_songs) : 
    final_songs = 2950 :=
by
  sorry

end NUMINAMATH_GPT_final_song_count_l2011_201159


namespace NUMINAMATH_GPT_inequality_holds_l2011_201129

theorem inequality_holds (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l2011_201129


namespace NUMINAMATH_GPT_find_min_y_l2011_201123

theorem find_min_y (x y : ℕ) (hx : x = y + 8) 
    (h : Nat.gcd ((x^3 + y^3) / (x + y)) (x * y) = 16) : 
    y = 4 :=
sorry

end NUMINAMATH_GPT_find_min_y_l2011_201123


namespace NUMINAMATH_GPT_cuboid_surface_area_l2011_201165

-- Definitions
def Length := 12  -- meters
def Breadth := 14  -- meters
def Height := 7  -- meters

-- Surface area of a cuboid formula
def surfaceAreaOfCuboid (l b h : Nat) : Nat :=
  2 * (l * b + l * h + b * h)

-- Proof statement
theorem cuboid_surface_area : surfaceAreaOfCuboid Length Breadth Height = 700 := by
  sorry

end NUMINAMATH_GPT_cuboid_surface_area_l2011_201165


namespace NUMINAMATH_GPT_half_angle_in_first_or_third_quadrant_l2011_201198

noncomputable 
def angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < (2 * k + 1) * Real.pi / 2

noncomputable 
def angle_in_first_or_third_quadrant (β : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi < β ∧ β < (k + 1/4) * Real.pi ∨
  ∃ i : ℤ, (2 * i + 1) * Real.pi < β ∧ β < (2 * i + 5/4) * Real.pi 

theorem half_angle_in_first_or_third_quadrant (α : ℝ) (h : angle_in_first_quadrant α) :
  angle_in_first_or_third_quadrant (α / 2) :=
  sorry

end NUMINAMATH_GPT_half_angle_in_first_or_third_quadrant_l2011_201198


namespace NUMINAMATH_GPT_factorization_correct_l2011_201130

theorem factorization_correct {c d : ℤ} (h1 : c + 4 * d = 4) (h2 : c * d = -32) :
  c - d = 12 :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l2011_201130


namespace NUMINAMATH_GPT_passengers_taken_at_second_station_l2011_201148

noncomputable def initial_passengers : ℕ := 270
noncomputable def passengers_dropped_first_station := initial_passengers / 3
noncomputable def passengers_after_first_station := initial_passengers - passengers_dropped_first_station + 280
noncomputable def passengers_dropped_second_station := passengers_after_first_station / 2
noncomputable def passengers_after_second_station (x : ℕ) := passengers_after_first_station - passengers_dropped_second_station + x
noncomputable def passengers_at_third_station := 242

theorem passengers_taken_at_second_station : ∃ x : ℕ,
  passengers_after_second_station x = passengers_at_third_station ∧ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_passengers_taken_at_second_station_l2011_201148


namespace NUMINAMATH_GPT_product_of_intersection_coordinates_l2011_201186

noncomputable def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 5)^2 = 1
noncomputable def circle2 (x y : ℝ) : Prop := (x - 5)^2 + (y - 5)^2 = 4

theorem product_of_intersection_coordinates :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧ x * y = 15 :=
by
  sorry

end NUMINAMATH_GPT_product_of_intersection_coordinates_l2011_201186


namespace NUMINAMATH_GPT_correct_subtraction_result_l2011_201103

-- Definition of numbers:
def tens_digit := 2
def ones_digit := 4
def correct_number := 10 * tens_digit + ones_digit
def incorrect_number := 59
def incorrect_result := 14
def Z := incorrect_result + incorrect_number

-- Statement of the theorem
theorem correct_subtraction_result : Z - correct_number = 49 :=
by
  sorry

end NUMINAMATH_GPT_correct_subtraction_result_l2011_201103


namespace NUMINAMATH_GPT_otimes_h_h_h_eq_h_l2011_201117

variable (h : ℝ)

def otimes (x y : ℝ) : ℝ := x^3 - y

theorem otimes_h_h_h_eq_h : otimes h (otimes h h) = h := by
  -- Proof goes here, but is omitted
  sorry

end NUMINAMATH_GPT_otimes_h_h_h_eq_h_l2011_201117


namespace NUMINAMATH_GPT_inequality_my_problem_l2011_201106

theorem inequality_my_problem (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a * b + b * c + c * a = 1) :
  (Real.sqrt ((1 / a) + 6 * b)) + (Real.sqrt ((1 / b) + 6 * c)) + (Real.sqrt ((1 / c) + 6 * a)) ≤ (1 / (a * b * c)) :=
  sorry

end NUMINAMATH_GPT_inequality_my_problem_l2011_201106


namespace NUMINAMATH_GPT_sequence_increasing_and_bounded_sequence_decreasing_and_bounded_l2011_201176

def recurrence_relation (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x (n + 1) = (2 * x n ^ 2 - x n) / (3 * (x n - 2))

-- For the first problem
theorem sequence_increasing_and_bounded (x : ℕ → ℝ) (h_rec : ∀ n, recurrence_relation x n)
  (h_initial : 4 < x 0 ∧ x 0 < 5) : ∀ n, x n < x (n + 1) ∧ x (n + 1) < 5 :=
by
  sorry

-- For the second problem
theorem sequence_decreasing_and_bounded (x : ℕ → ℝ) (h_rec : ∀ n, recurrence_relation x n)
  (h_initial : x 0 > 5) : ∀ n, 5 < x (n + 1) ∧ x (n + 1) < x n :=
by
  sorry

end NUMINAMATH_GPT_sequence_increasing_and_bounded_sequence_decreasing_and_bounded_l2011_201176


namespace NUMINAMATH_GPT_length_first_train_correct_l2011_201118

noncomputable def length_first_train 
    (speed_train1_kmph : ℕ := 120)
    (speed_train2_kmph : ℕ := 80)
    (length_train2_m : ℝ := 290.04)
    (time_sec : ℕ := 9) 
    (conversion_factor : ℝ := (5 / 18)) : ℝ :=
  let relative_speed_kmph := speed_train1_kmph + speed_train2_kmph
  let relative_speed_mps := relative_speed_kmph * conversion_factor
  let total_distance_m := relative_speed_mps * time_sec
  let length_train1_m := total_distance_m - length_train2_m
  length_train1_m

theorem length_first_train_correct 
    (L1_approx : ℝ := 210) :
    length_first_train = L1_approx :=
  by
  sorry

end NUMINAMATH_GPT_length_first_train_correct_l2011_201118


namespace NUMINAMATH_GPT_incorrect_conclusion_l2011_201178

noncomputable def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

theorem incorrect_conclusion (m : ℝ) (hx : m - 2 = 0) :
  ¬(∀ x : ℝ, quadratic m x = 2 ↔ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_conclusion_l2011_201178


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2011_201169

noncomputable 
def expr (a b : ℚ) := 2*(a^2*b - 2*a*b) - 3*(a^2*b - 3*a*b) + a^2*b

theorem simplify_and_evaluate :
  let a := (-2 : ℚ) 
  let b := (1/3 : ℚ)
  expr a b = -10/3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2011_201169


namespace NUMINAMATH_GPT_inequality_solution_set_l2011_201139

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + (a-1)*x^2

theorem inequality_solution_set (a : ℝ) (ha : ∀ x : ℝ, f x a = -f (-x) a) :
  {x : ℝ | f (a*x) a > f (a-x) a} = {x : ℝ | x > 1/2} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l2011_201139


namespace NUMINAMATH_GPT_cabin_charges_per_night_l2011_201179

theorem cabin_charges_per_night 
  (total_lodging_cost : ℕ)
  (hostel_cost_per_night : ℕ)
  (hostel_days : ℕ)
  (total_cabin_days : ℕ)
  (friends_sharing_expenses : ℕ)
  (jimmy_lodging_expense : ℕ) 
  (total_cost_paid_by_jimmy : ℕ) :
  total_lodging_cost = total_cost_paid_by_jimmy →
  hostel_cost_per_night = 15 →
  hostel_days = 3 →
  total_cabin_days = 2 →
  friends_sharing_expenses = 3 →
  jimmy_lodging_expense = 75 →
  ∃ cabin_cost_per_night, cabin_cost_per_night = 45 :=
by
  sorry

end NUMINAMATH_GPT_cabin_charges_per_night_l2011_201179


namespace NUMINAMATH_GPT_find_larger_number_l2011_201188

theorem find_larger_number (L S : ℕ)
  (h1 : L - S = 1370)
  (h2 : L = 6 * S + 15) :
  L = 1641 := sorry

end NUMINAMATH_GPT_find_larger_number_l2011_201188


namespace NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l2011_201140

-- Definitions based on conditions
def is_additive_inverse (x y : Int) : Prop := x + y = 0

-- The proof statement
theorem opposite_of_2023_is_neg_2023 : is_additive_inverse 2023 (-2023) :=
by
  -- This is where the proof would go, but it is marked as sorry for now
  sorry

end NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l2011_201140


namespace NUMINAMATH_GPT_find_grape_juice_l2011_201105

variables (milk water: ℝ) (limit total_before_test grapejuice: ℝ)

-- Conditions
def milk_amt: ℝ := 8
def water_amt: ℝ := 8
def limit_amt: ℝ := 32

-- The total liquid consumed before the test can be computed
def total_before_test_amt (milk water: ℝ) : ℝ := limit_amt - water_amt

-- The given total liquid consumed must be (milk + grape juice)
def total_consumed (milk grapejuice: ℝ) : ℝ := milk + grapejuice

theorem find_grape_juice :
    total_before_test_amt milk_amt water_amt = total_consumed milk_amt grapejuice →
    grapejuice = 16 :=
by
    unfold total_before_test_amt total_consumed
    sorry

end NUMINAMATH_GPT_find_grape_juice_l2011_201105


namespace NUMINAMATH_GPT_smallest_nat_divisible_by_48_squared_l2011_201189

theorem smallest_nat_divisible_by_48_squared :
  ∃ n : ℕ, (n % (48^2) = 0) ∧ 
           (∀ (d : ℕ), d ∈ (Nat.digits n 10) → d = 0 ∨ d = 1) ∧ 
           (n = 11111111100000000) := sorry

end NUMINAMATH_GPT_smallest_nat_divisible_by_48_squared_l2011_201189


namespace NUMINAMATH_GPT_ferris_wheel_capacity_l2011_201154

theorem ferris_wheel_capacity :
  let num_seats := 4
  let people_per_seat := 4
  num_seats * people_per_seat = 16 := 
by
  let num_seats := 4
  let people_per_seat := 4
  sorry

end NUMINAMATH_GPT_ferris_wheel_capacity_l2011_201154


namespace NUMINAMATH_GPT_negation_of_P_l2011_201146

variable (x : ℝ)

def P : Prop := ∀ x : ℝ, x^2 + 2*x + 3 ≥ 0

theorem negation_of_P : ¬P ↔ ∃ x : ℝ, x^2 + 2*x + 3 < 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_P_l2011_201146


namespace NUMINAMATH_GPT_kids_left_playing_l2011_201180

-- Define the conditions
def initial_kids : ℝ := 22.0
def kids_went_home : ℝ := 14.0

-- Theorem statement: Prove that the number of kids left playing is 8.0
theorem kids_left_playing : initial_kids - kids_went_home = 8.0 :=
by
  sorry -- Proof is left as an exercise

end NUMINAMATH_GPT_kids_left_playing_l2011_201180


namespace NUMINAMATH_GPT_expected_value_is_90_l2011_201143

noncomputable def expected_value_coins_heads : ℕ :=
  let nickel := 5
  let quarter := 25
  let half_dollar := 50
  let dollar := 100
  1/2 * (nickel + quarter + half_dollar + dollar)

theorem expected_value_is_90 : expected_value_coins_heads = 90 := by
  sorry

end NUMINAMATH_GPT_expected_value_is_90_l2011_201143


namespace NUMINAMATH_GPT_max_value_of_expression_l2011_201136

theorem max_value_of_expression :
  ∀ r : ℝ, -3 * r^2 + 30 * r + 8 ≤ 83 :=
by
  -- Proof needed
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l2011_201136


namespace NUMINAMATH_GPT_system1_solution_l2011_201151

theorem system1_solution (x y : ℝ) (h1 : 4 * x - 3 * y = 1) (h2 : 3 * x - 2 * y = -1) : x = -5 ∧ y = 7 :=
sorry

end NUMINAMATH_GPT_system1_solution_l2011_201151


namespace NUMINAMATH_GPT_counting_five_digit_numbers_l2011_201156

theorem counting_five_digit_numbers :
  ∃ (M : ℕ), 
    (∃ (b : ℕ), (∃ (y : ℕ), 10000 * b + y = 8 * y ∧ 10000 * b = 7 * y ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1429 ≤ y ∧ y ≤ 9996)) ∧ 
    (M = 1224) := 
by
  sorry

end NUMINAMATH_GPT_counting_five_digit_numbers_l2011_201156


namespace NUMINAMATH_GPT_path_length_cube_dot_l2011_201125

-- Define the edge length of the cube
def edge_length : ℝ := 2

-- Define the distance of the dot from the center of the top face
def dot_distance_from_center : ℝ := 0.5

-- Define the number of complete rolls
def complete_rolls : ℕ := 2

-- Calculate the constant c such that the path length of the dot is c * π
theorem path_length_cube_dot : ∃ c : ℝ, dot_distance_from_center = 2.236 :=
by
  sorry

end NUMINAMATH_GPT_path_length_cube_dot_l2011_201125


namespace NUMINAMATH_GPT_total_cost_l2011_201181

noncomputable def cost_sandwich : ℝ := 2.44
noncomputable def quantity_sandwich : ℕ := 2
noncomputable def cost_soda : ℝ := 0.87
noncomputable def quantity_soda : ℕ := 4

noncomputable def total_cost_sandwiches : ℝ := cost_sandwich * quantity_sandwich
noncomputable def total_cost_sodas : ℝ := cost_soda * quantity_soda

theorem total_cost (total_cost_sandwiches total_cost_sodas : ℝ) : (total_cost_sandwiches + total_cost_sodas = 8.36) :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l2011_201181


namespace NUMINAMATH_GPT_balls_in_boxes_l2011_201116

/-- Prove that the number of ways to put 6 distinguishable balls in 3 distinguishable boxes is 729 (which is 3^6). -/
theorem balls_in_boxes : (3 ^ 6) = 729 :=
by 
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l2011_201116


namespace NUMINAMATH_GPT_value_of_7th_term_l2011_201173

noncomputable def arithmetic_sequence_a1_d_n (a1 d n a7 : ℝ) : Prop := 
  ((5 * a1 + 10 * d = 68) ∧ 
   (5 * (a1 + (n - 1) * d) - 10 * d = 292) ∧
   (n / 2 * (2 * a1 + (n - 1) * d) = 234) ∧ 
   (a1 + 6 * d = a7))

theorem value_of_7th_term (a1 d n a7 : ℝ) : 
  arithmetic_sequence_a1_d_n a1 d n 18 := 
by
  simp [arithmetic_sequence_a1_d_n]
  sorry

end NUMINAMATH_GPT_value_of_7th_term_l2011_201173


namespace NUMINAMATH_GPT_pills_in_a_week_l2011_201142

def insulin_pills_per_day : Nat := 2
def blood_pressure_pills_per_day : Nat := 3
def anticonvulsant_pills_per_day : Nat := 2 * blood_pressure_pills_per_day

def total_pills_per_day : Nat := insulin_pills_per_day + blood_pressure_pills_per_day + anticonvulsant_pills_per_day

theorem pills_in_a_week : total_pills_per_day * 7 = 77 := by
  sorry

end NUMINAMATH_GPT_pills_in_a_week_l2011_201142


namespace NUMINAMATH_GPT_complete_the_square_d_l2011_201127

theorem complete_the_square_d (x : ℝ) : (∃ c d : ℝ, x^2 + 6 * x - 4 = 0 → (x + c)^2 = d) ∧ d = 13 :=
by
  sorry

end NUMINAMATH_GPT_complete_the_square_d_l2011_201127


namespace NUMINAMATH_GPT_average_speed_correct_l2011_201135

-- Definitions for the conditions
def distance1 : ℚ := 40
def speed1 : ℚ := 8
def time1 : ℚ := distance1 / speed1

def distance2 : ℚ := 20
def speed2 : ℚ := 40
def time2 : ℚ := distance2 / speed2

def total_distance : ℚ := distance1 + distance2
def total_time : ℚ := time1 + time2

-- Definition of average speed
def average_speed : ℚ := total_distance / total_time

-- Proof statement that needs to be proven
theorem average_speed_correct : average_speed = 120 / 11 :=
by 
  -- The details for the proof will be filled here
  sorry

end NUMINAMATH_GPT_average_speed_correct_l2011_201135


namespace NUMINAMATH_GPT_solution_set_f_inequality_l2011_201104

variable (f : ℝ → ℝ)

axiom domain_of_f : ∀ x : ℝ, true
axiom avg_rate_of_f : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 3
axiom f_at_5 : f 5 = 18

theorem solution_set_f_inequality : {x : ℝ | f (3 * x - 1) > 9 * x} = {x : ℝ | x > 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_f_inequality_l2011_201104
