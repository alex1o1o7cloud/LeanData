import Mathlib

namespace goose_eggs_count_l332_33286

theorem goose_eggs_count (E : ℕ) 
  (h1 : (1/2 : ℝ) * E = E/2)
  (h2 : (3/4 : ℝ) * (E/2) = (3 * E) / 8)
  (h3 : (2/5 : ℝ) * ((3 * E) / 8) = (3 * E) / 20)
  (h4 : (3 * E) / 20 = 120) :
  E = 400 :=
sorry

end goose_eggs_count_l332_33286


namespace max_true_statements_l332_33228

theorem max_true_statements (a b : ℝ) :
  ((a < b) → (b < 0) → (a < 0) → ¬(1 / a < 1 / b)) ∧
  ((a < b) → (b < 0) → (a < 0) → ¬(a^2 < b^2)) →
  3 = 3
:=
by
  intros
  sorry

end max_true_statements_l332_33228


namespace jack_travel_total_hours_l332_33256

theorem jack_travel_total_hours :
  (20 + 14 * 24) + (15 + 10 * 24) + (10 + 7 * 24) = 789 := by
  sorry

end jack_travel_total_hours_l332_33256


namespace simplify_and_evaluate_expression_l332_33270

theorem simplify_and_evaluate_expression (a : ℝ) (h : a^2 + 2 * a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2)) / (a^2 + 2 * a) / (a - 2) = 1 / 4 := 
by 
  sorry

end simplify_and_evaluate_expression_l332_33270


namespace sum_not_prime_if_product_equality_l332_33212

theorem sum_not_prime_if_product_equality 
  (a b c d : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a * b = c * d) : ¬Nat.Prime (a + b + c + d) := 
by
  sorry

end sum_not_prime_if_product_equality_l332_33212


namespace minimum_value_l332_33255

variable {x : ℝ}

theorem minimum_value (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
by
  sorry

end minimum_value_l332_33255


namespace price_of_food_before_tax_and_tip_l332_33209

noncomputable def actual_price_of_food (total_paid : ℝ) (tip_rate tax_rate : ℝ) : ℝ :=
  total_paid / (1 + tip_rate) / (1 + tax_rate)

theorem price_of_food_before_tax_and_tip :
  actual_price_of_food 211.20 0.20 0.10 = 160 :=
by
  sorry

end price_of_food_before_tax_and_tip_l332_33209


namespace area_of_PDCE_l332_33229

/-- A theorem to prove the area of quadrilateral PDCE given conditions in triangle ABC. -/
theorem area_of_PDCE
  (ABC_area : ℝ)
  (BD_to_CD_ratio : ℝ)
  (E_is_midpoint : Prop)
  (AD_intersects_BE : Prop)
  (P : Prop)
  (area_PDCE : ℝ) :
  (ABC_area = 1) →
  (BD_to_CD_ratio = 2 / 1) →
  E_is_midpoint →
  AD_intersects_BE →
  ∃ P, P →
    area_PDCE = 7 / 30 :=
by sorry

end area_of_PDCE_l332_33229


namespace basin_more_than_tank2_l332_33272

/-- Define the water volumes in milliliters -/
def volume_bottle1 : ℕ := 1000 -- 1 liter = 1000 milliliters
def volume_bottle2 : ℕ := 400  -- 400 milliliters
def volume_tank : ℕ := 2800    -- 2800 milliliters
def volume_basin : ℕ := volume_bottle1 + volume_bottle2 + volume_tank -- total volume in basin
def volume_tank2 : ℕ := 4000 + 100 -- 4 liters 100 milliliters tank

/-- Theorem: The basin can hold 100 ml more water than the 4-liter 100-milliliter tank -/
theorem basin_more_than_tank2 : volume_basin = volume_tank2 + 100 :=
by
  -- This is where the proof would go, but it is not required for this exercise
  sorry

end basin_more_than_tank2_l332_33272


namespace cube_faces_sum_eq_neg_3_l332_33299

theorem cube_faces_sum_eq_neg_3 
    (a b c d e f : ℤ)
    (h1 : a = -3)
    (h2 : b = a + 1)
    (h3 : c = b + 1)
    (h4 : d = c + 1)
    (h5 : e = d + 1)
    (h6 : f = e + 1)
    (h7 : a + f = b + e)
    (h8 : b + e = c + d) :
  a + b + c + d + e + f = -3 := sorry

end cube_faces_sum_eq_neg_3_l332_33299


namespace root_in_interval_l332_33241

theorem root_in_interval (a b c : ℝ) (h_a : a ≠ 0)
    (h_table : ∀ x y, (x = 1.2 ∧ y = -1.16) ∨ (x = 1.3 ∧ y = -0.71) ∨ (x = 1.4 ∧ y = -0.24) ∨ (x = 1.5 ∧ y = 0.25) ∨ (x = 1.6 ∧ y = 0.76) → y = a * x^2 + b * x + c ) :
  ∃ x₁, 1.4 < x₁ ∧ x₁ < 1.5 ∧ a * x₁^2 + b * x₁ + c = 0 :=
by sorry

end root_in_interval_l332_33241


namespace minimize_folded_area_l332_33290

-- defining the problem as statements in Lean
variables (a M N : ℝ) (M_on_AB : M > 0 ∧ M < a) (N_on_CD : N > 0 ∧ N < a)

-- main theorem statement
theorem minimize_folded_area :
  BM = 5 * a / 8 →
  CN = a / 8 →
  S = 3 * a ^ 2 / 8 := sorry

end minimize_folded_area_l332_33290


namespace Lenora_scored_30_points_l332_33218

variable (x y : ℕ)
variable (hx : x + y = 40)
variable (three_point_success_rate : ℚ := 25 / 100)
variable (free_throw_success_rate : ℚ := 50 / 100)
variable (points_three_point : ℚ := 3)
variable (points_free_throw : ℚ := 1)
variable (three_point_contribution : ℚ := three_point_success_rate * points_three_point * x)
variable (free_throw_contribution : ℚ := free_throw_success_rate * points_free_throw * y)
variable (total_points : ℚ := three_point_contribution + free_throw_contribution)

theorem Lenora_scored_30_points : total_points = 30 :=
by
  sorry

end Lenora_scored_30_points_l332_33218


namespace solve_for_x_l332_33267

theorem solve_for_x (x : ℝ) (h : (3 * x + 15)^2 = 3 * (4 * x + 40)) :
  x = -5 / 3 ∨ x = -7 :=
sorry

end solve_for_x_l332_33267


namespace janet_daily_search_time_l332_33230

-- Define the conditions
def minutes_looking_for_keys_per_day (x : ℕ) := 
  let total_time_per_day := x + 3
  let total_time_per_week := 7 * total_time_per_day
  total_time_per_week = 77

-- State the theorem
theorem janet_daily_search_time : 
  ∃ x : ℕ, minutes_looking_for_keys_per_day x ∧ x = 8 := by
  sorry

end janet_daily_search_time_l332_33230


namespace sam_seashell_count_l332_33210

/-!
# Problem statement:
-/
def initialSeashells := 35
def seashellsGivenToJoan := 18
def seashellsFoundToday := 20
def seashellsGivenToTom := 5

/-!
# Proof goal: Prove that the current number of seashells Sam has is 32.
-/
theorem sam_seashell_count :
  initialSeashells - seashellsGivenToJoan + seashellsFoundToday - seashellsGivenToTom = 32 :=
  sorry

end sam_seashell_count_l332_33210


namespace marching_band_formations_l332_33247

open Nat

theorem marching_band_formations :
  ∃ g, (g = 9) ∧ ∀ s t : ℕ, (s * t = 480 ∧ 15 ≤ t ∧ t ≤ 60) ↔ 
    (t = 15 ∨ t = 16 ∨ t = 20 ∨ t = 24 ∨ t = 30 ∨ t = 32 ∨ t = 40 ∨ t = 48 ∨ t = 60) :=
by
  -- Skipped proof.
  sorry

end marching_band_formations_l332_33247


namespace trip_time_difference_l332_33202

-- Define the speed of the motorcycle
def speed : ℤ := 60

-- Define the distances for the two trips
def distance1 : ℤ := 360
def distance2 : ℤ := 420

-- Define the time calculation function
def time (distance speed : ℤ) : ℤ := distance / speed

-- Prove the problem statement
theorem trip_time_difference : (time distance2 speed - time distance1 speed) * 60 = 60 := by
  -- Provide the proof here
  sorry

end trip_time_difference_l332_33202


namespace wheels_motion_is_rotation_l332_33222

def motion_wheel_car := "rotation"
def question_wheels_motion := "What is the type of motion exhibited by the wheels of a moving car?"

theorem wheels_motion_is_rotation :
  (question_wheels_motion = "What is the type of motion exhibited by the wheels of a moving car?" ∧ 
   motion_wheel_car = "rotation") → motion_wheel_car = "rotation" :=
by
  sorry

end wheels_motion_is_rotation_l332_33222


namespace probability_log2_x_between_1_and_2_l332_33258

noncomputable def probability_log_between : ℝ :=
  let favorable_range := (4:ℝ) - (2:ℝ)
  let total_range := (6:ℝ) - (0:ℝ)
  favorable_range / total_range

theorem probability_log2_x_between_1_and_2 :
  probability_log_between = 1 / 3 :=
sorry

end probability_log2_x_between_1_and_2_l332_33258


namespace problem_1_problem_2_l332_33213

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * a * x^2 + 2 * x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x - g x a
noncomputable def h' (x : ℝ) (a : ℝ) : ℝ := (1 / x) - a * x - 2
noncomputable def G (x : ℝ) : ℝ := ((1 / x) - 1) ^ 2 - 1

theorem problem_1 (a : ℝ): 
  (∃ x : ℝ, 0 < x ∧ h' x a < 0) ↔ a > -1 :=
by sorry

theorem problem_2 (a : ℝ):
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → h' x a ≤ 0) ↔ a ≥ -(7 / 16) :=
by sorry

end problem_1_problem_2_l332_33213


namespace area_of_rectangular_park_l332_33249

theorem area_of_rectangular_park
  (l w : ℕ) 
  (h_perimeter : 2 * l + 2 * w = 80)
  (h_length : l = 3 * w) :
  l * w = 300 :=
sorry

end area_of_rectangular_park_l332_33249


namespace cats_remaining_proof_l332_33227

def initial_siamese : ℕ := 38
def initial_house : ℕ := 25
def sold_cats : ℕ := 45

def total_cats (s : ℕ) (h : ℕ) : ℕ := s + h
def remaining_cats (total : ℕ) (sold : ℕ) : ℕ := total - sold

theorem cats_remaining_proof : remaining_cats (total_cats initial_siamese initial_house) sold_cats = 18 :=
by
  sorry

end cats_remaining_proof_l332_33227


namespace total_cookies_baked_l332_33264

def cookies_baked_yesterday : ℕ := 435
def cookies_baked_today : ℕ := 139

theorem total_cookies_baked : cookies_baked_yesterday + cookies_baked_today = 574 := by
  sorry

end total_cookies_baked_l332_33264


namespace three_g_two_plus_two_g_neg_four_l332_33295

def g (x : ℝ) : ℝ := 2 * x ^ 2 - 2 * x + 11

theorem three_g_two_plus_two_g_neg_four : 3 * g 2 + 2 * g (-4) = 147 := by
  sorry

end three_g_two_plus_two_g_neg_four_l332_33295


namespace area_triangle_FQH_l332_33244

open Set

structure Point where
  x : ℝ
  y : ℝ

def Rectangle (A B C D : Point) : Prop :=
  A.x = B.x ∧ C.x = D.x ∧ A.y = D.y ∧ B.y = C.y

def IsMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def AreaTrapezoid (A B C D : Point) : ℝ :=
  0.5 * (B.x - A.x + D.x - C.x) * (A.y - C.y)

def AreaTriangle (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

variables (E P R H F Q G : Point)

-- Conditions
axiom h1 : Rectangle E F G H
axiom h2 : E.y - P.y = 8
axiom h3 : R.y - H.y = 8
axiom h4 : F.x - E.x = 16
axiom h5 : AreaTrapezoid P R H G = 160

-- Target to prove
theorem area_triangle_FQH : AreaTriangle F Q H = 80 :=
sorry

end area_triangle_FQH_l332_33244


namespace smallest_n_mod5_l332_33279

theorem smallest_n_mod5 :
  ∃ n : ℕ, n > 0 ∧ 6^n % 5 = n^6 % 5 ∧ ∀ m : ℕ, m > 0 ∧ 6^m % 5 = m^6 % 5 → n ≤ m :=
by
  sorry

end smallest_n_mod5_l332_33279


namespace contrapositive_l332_33268

theorem contrapositive (a b : ℕ) : (a = 0 → ab = 0) → (ab ≠ 0 → a ≠ 0) :=
by
  sorry

end contrapositive_l332_33268


namespace evaluate_g_neg5_l332_33232

def g (x : ℝ) : ℝ := 4 * x - 2

theorem evaluate_g_neg5 : g (-5) = -22 := 
  by sorry

end evaluate_g_neg5_l332_33232


namespace equal_roots_implies_c_value_l332_33292

theorem equal_roots_implies_c_value (c : ℝ) 
  (h : ∃ x : ℝ, (x^2 + 6 * x - c = 0) ∧ (2 * x + 6 = 0)) :
  c = -9 :=
sorry

end equal_roots_implies_c_value_l332_33292


namespace tree_height_is_12_l332_33284

-- Let h be the height of the tree in meters.
def height_of_tree (h : ℝ) : Prop :=
  ∃ h, (h / 8 = 150 / 100) → h = 12

theorem tree_height_is_12 : ∃ h : ℝ, height_of_tree h :=
by {
  sorry
}

end tree_height_is_12_l332_33284


namespace part1_part2_part3_l332_33219

-- Defining the quadratic function
def quadratic (t : ℝ) (x : ℝ) : ℝ := x^2 - 2 * t * x + 3

-- Part (1)
theorem part1 (t : ℝ) (h : quadratic t 2 = 1) : t = 3 / 2 :=
by sorry

-- Part (2)
theorem part2 (t : ℝ) (h : ∀x, 0 ≤ x → x ≤ 3 → (quadratic t x) ≥ -2) : t = Real.sqrt 5 :=
by sorry

-- Part (3)
theorem part3 (m a b : ℝ) (hA : quadratic t (m - 2) = a) (hB : quadratic t 4 = b) 
              (hC : quadratic t m = a) (ha : a < b) (hb : b < 3) (ht : t > 0) : 
              (3 < m ∧ m < 4) ∨ (m > 6) :=
by sorry

end part1_part2_part3_l332_33219


namespace waiter_customer_count_l332_33288

def initial_customers := 33
def customers_left := 31
def new_customers := 26

theorem waiter_customer_count :
  (initial_customers - customers_left) + new_customers = 28 :=
by
  -- This is a placeholder for the proof that can be filled later.
  sorry

end waiter_customer_count_l332_33288


namespace johns_weekly_allowance_l332_33260

theorem johns_weekly_allowance
    (A : ℝ)
    (h1 : ∃ A, (4/15) * A = 0.64) :
    A = 2.40 :=
by
  sorry

end johns_weekly_allowance_l332_33260


namespace Shekar_biology_marks_l332_33271

theorem Shekar_biology_marks 
  (math_marks : ℕ := 76) 
  (science_marks : ℕ := 65) 
  (social_studies_marks : ℕ := 82) 
  (english_marks : ℕ := 47) 
  (average_marks : ℕ := 71) 
  (num_subjects : ℕ := 5) 
  (biology_marks : ℕ) :
  (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / num_subjects = average_marks → biology_marks = 85 := 
by 
  sorry

end Shekar_biology_marks_l332_33271


namespace range_of_a_l332_33266

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ -x^2 + 4*x + a = 0) ↔ (-3 ≤ a ∧ a ≤ 21) :=
by
  sorry

end range_of_a_l332_33266


namespace age_difference_l332_33224

theorem age_difference (x : ℕ) (older_age younger_age : ℕ) 
  (h1 : 3 * x = older_age)
  (h2 : 2 * x = younger_age)
  (h3 : older_age + younger_age = 60) : 
  older_age - younger_age = 12 := 
by
  sorry

end age_difference_l332_33224


namespace sum_abcd_eq_neg_46_div_3_l332_33257

theorem sum_abcd_eq_neg_46_div_3
  (a b c d : ℝ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 15) :
  a + b + c + d = -46 / 3 := 
by sorry

end sum_abcd_eq_neg_46_div_3_l332_33257


namespace num_integers_in_set_x_l332_33204

-- Definition and conditions
variable (x y : Finset ℤ)
variable (h1 : y.card = 10)
variable (h2 : (x ∩ y).card = 6)
variable (h3 : (x.symmDiff y).card = 6)

-- Proof statement
theorem num_integers_in_set_x : x.card = 8 := by
  sorry

end num_integers_in_set_x_l332_33204


namespace randy_wipes_days_l332_33289

theorem randy_wipes_days (wipes_per_pack : ℕ) (packs_needed : ℕ) (wipes_per_walk : ℕ) (walks_per_day : ℕ) (total_wipes : ℕ) (wipes_per_day : ℕ) (days_needed : ℕ) 
(h1 : wipes_per_pack = 120)
(h2 : packs_needed = 6)
(h3 : wipes_per_walk = 4)
(h4 : walks_per_day = 2)
(h5 : total_wipes = packs_needed * wipes_per_pack)
(h6 : wipes_per_day = wipes_per_walk * walks_per_day)
(h7 : days_needed = total_wipes / wipes_per_day) : 
days_needed = 90 :=
by sorry

end randy_wipes_days_l332_33289


namespace max_sum_of_factors_l332_33221

theorem max_sum_of_factors (a b : ℕ) (h : a * b = 42) : a + b ≤ 43 :=
by
  -- sorry to skip the proof
  sorry

end max_sum_of_factors_l332_33221


namespace problem_l332_33235

def m (x : ℝ) : ℝ := (x + 2) * (x + 3)
def n (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 9

theorem problem (x : ℝ) : m x < n x :=
by sorry

end problem_l332_33235


namespace probability_log_interval_l332_33211

open Set Real

noncomputable def probability_in_interval (a b c d : ℝ) (I J : Set ℝ) := 
  (b - a) / (d - c)

theorem probability_log_interval : 
  probability_in_interval 2 4 0 6 (Icc 0 6) (Ioo 2 4) = 1 / 3 := 
sorry

end probability_log_interval_l332_33211


namespace john_ultramarathon_distance_l332_33275

theorem john_ultramarathon_distance :
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  initial_time * (1 + time_increase_percentage) * (initial_speed + speed_increase) = 168 :=
by
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  sorry

end john_ultramarathon_distance_l332_33275


namespace cryptarithm_solution_l332_33239

theorem cryptarithm_solution (A B C D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_adjacent : A = C + 1 ∨ A = C - 1)
  (h_diff : B = D + 2 ∨ B = D - 2) :
  1000 * A + 100 * B + 10 * C + D = 5240 :=
sorry

end cryptarithm_solution_l332_33239


namespace village_Y_initial_population_l332_33263

def population_X := 76000
def decrease_rate_X := 1200
def increase_rate_Y := 800
def years := 17

def population_X_after_17_years := population_X - decrease_rate_X * years
def population_Y_after_17_years (P : Nat) := P + increase_rate_Y * years

theorem village_Y_initial_population (P : Nat) (h : population_Y_after_17_years P = population_X_after_17_years) : P = 42000 :=
by
  sorry

end village_Y_initial_population_l332_33263


namespace car_travel_distance_l332_33277

theorem car_travel_distance 
  (v_train : ℝ) (h_train_speed : v_train = 90) 
  (v_car : ℝ) (h_car_speed : v_car = (2 / 3) * v_train) 
  (t : ℝ) (h_time : t = 0.5) :
  ∃ d : ℝ, d = v_car * t ∧ d = 30 := 
sorry

end car_travel_distance_l332_33277


namespace peaches_eaten_l332_33238

theorem peaches_eaten (P B Baskets P_each R Boxes P_box : ℕ) 
  (h1 : B = 5) 
  (h2 : P_each = 25)
  (h3 : Baskets = B * P_each)
  (h4 : R = 8) 
  (h5 : P_box = 15)
  (h6 : Boxes = R * P_box)
  (h7 : P = Baskets - Boxes) : P = 5 :=
by sorry

end peaches_eaten_l332_33238


namespace probability_of_B_l332_33233

-- Define the events and their probabilities according to the problem description
def A₁ := "Event where a red ball is taken from bag A"
def A₂ := "Event where a white ball is taken from bag A"
def A₃ := "Event where a black ball is taken from bag A"
def B := "Event where a red ball is taken from bag B"

-- Types of bags A and B containing balls
structure Bag where
  red : Nat
  white : Nat
  black : Nat

-- Initial bags
def bagA : Bag := ⟨ 3, 2, 5 ⟩
def bagB : Bag := ⟨ 3, 3, 4 ⟩

-- Probabilities of each event in bagA
def P_A₁ : ℚ := 3 / 10
def P_A₂ : ℚ := 2 / 10
def P_A₃ : ℚ := 5 / 10

-- Probability of event B under conditions A₁, A₂, A₃
def P_B_given_A₁ : ℚ := 4 / 11
def P_B_given_A₂ : ℚ := 3 / 11
def P_B_given_A₃ : ℚ := 3 / 11

-- Goal: Prove that the probability of drawing a red ball from bag B (P(B)) is 3/10
theorem probability_of_B : 
  (P_A₁ * P_B_given_A₁ + P_A₂ * P_B_given_A₂ + P_A₃ * P_B_given_A₃) = (3 / 10) :=
by
  -- Placeholder for the proof
  sorry

end probability_of_B_l332_33233


namespace min_value_of_squares_l332_33231

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) : a^2 + b^2 + c^2 ≥ t^2 / 3 :=
sorry

end min_value_of_squares_l332_33231


namespace georgia_vs_texas_license_plates_l332_33283

theorem georgia_vs_texas_license_plates :
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  georgia_plates - texas_plates = 731161600 :=
by
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  show georgia_plates - texas_plates = 731161600
  sorry

end georgia_vs_texas_license_plates_l332_33283


namespace find_abc_l332_33280

theorem find_abc
  (a b c : ℝ)
  (h1 : a^2 * (b + c) = 2011)
  (h2 : b^2 * (a + c) = 2011)
  (h3 : a ≠ b) : 
  a * b * c = -2011 := 
by 
sorry

end find_abc_l332_33280


namespace calculate_p_op_l332_33240

def op (x y : ℝ) := x * y^2 - x

theorem calculate_p_op (p : ℝ) : op p (op p p) = p^7 - 2*p^5 + p^3 - p :=
by
  sorry

end calculate_p_op_l332_33240


namespace only_nonneg_solution_l332_33261

theorem only_nonneg_solution :
  ∀ (x y : ℕ), 2^x = y^2 + y + 1 → (x, y) = (0, 0) := by
  intros x y h
  sorry

end only_nonneg_solution_l332_33261


namespace sum_of_cubes_divisible_by_9_l332_33217

theorem sum_of_cubes_divisible_by_9 (n : ℕ) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := 
  sorry

end sum_of_cubes_divisible_by_9_l332_33217


namespace difference_max_min_eq_2log2_minus_1_l332_33273

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem difference_max_min_eq_2log2_minus_1 :
  let M := max (f (1 / 2)) (max (f 1) (f 2))
  let N := min (f (1 / 2)) (min (f 1) (f 2))
  M - N = 2 * Real.log 2 - 1 :=
by
  let M := max (f (1 / 2)) (max (f 1) (f 2))
  let N := min (f (1 / 2)) (min (f 1) (f 2))
  sorry

end difference_max_min_eq_2log2_minus_1_l332_33273


namespace sammy_remaining_problems_l332_33262

variable (total_problems : Nat)
variable (fraction_problems : Nat) (decimal_problems : Nat) (multiplication_problems : Nat) (division_problems : Nat)
variable (completed_fraction_problems : Nat) (completed_decimal_problems : Nat)
variable (completed_multiplication_problems : Nat) (completed_division_problems : Nat)
variable (remaining_problems : Nat)

theorem sammy_remaining_problems
  (h₁ : total_problems = 115)
  (h₂ : fraction_problems = 35)
  (h₃ : decimal_problems = 40)
  (h₄ : multiplication_problems = 20)
  (h₅ : division_problems = 20)
  (h₆ : completed_fraction_problems = 11)
  (h₇ : completed_decimal_problems = 17)
  (h₈ : completed_multiplication_problems = 9)
  (h₉ : completed_division_problems = 5)
  (h₁₀ : remaining_problems =
    fraction_problems - completed_fraction_problems +
    decimal_problems - completed_decimal_problems +
    multiplication_problems - completed_multiplication_problems +
    division_problems - completed_division_problems) :
  remaining_problems = 73 :=
  by
    -- proof to be written
    sorry

end sammy_remaining_problems_l332_33262


namespace twentieth_term_is_78_l332_33254

-- Define the arithmetic sequence parameters
def first_term : ℤ := 2
def common_difference : ℤ := 4

-- Define the function to compute the n-th term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ := first_term + (n - 1) * common_difference

-- Formulate the theorem to prove
theorem twentieth_term_is_78 : nth_term 20 = 78 :=
by
  sorry

end twentieth_term_is_78_l332_33254


namespace brick_width_l332_33234

theorem brick_width (l_brick : ℕ) (w_courtyard l_courtyard : ℕ) (num_bricks : ℕ) (w_brick : ℕ)
  (H1 : l_courtyard = 24) 
  (H2 : w_courtyard = 14) 
  (H3 : num_bricks = 8960) 
  (H4 : l_brick = 25) 
  (H5 : (w_courtyard * 100 * l_courtyard * 100 = (num_bricks * (l_brick * w_brick)))) :
  w_brick = 15 :=
by
  sorry

end brick_width_l332_33234


namespace find_rate_of_interest_l332_33298

noncomputable def rate_of_interest (P : ℝ) (r : ℝ) : Prop :=
  let CI2 := P * (1 + r)^2 - P
  let CI3 := P * (1 + r)^3 - P
  CI2 = 1200 ∧ CI3 = 1272 → r = 0.06

theorem find_rate_of_interest (P : ℝ) (r : ℝ) : rate_of_interest P r :=
by sorry

end find_rate_of_interest_l332_33298


namespace total_plants_in_garden_l332_33214

-- Definitions based on conditions
def basil_plants : ℕ := 5
def oregano_plants : ℕ := 2 + 2 * basil_plants

-- Theorem statement
theorem total_plants_in_garden : basil_plants + oregano_plants = 17 := by
  -- Skipping the proof with sorry
  sorry

end total_plants_in_garden_l332_33214


namespace gcd_779_209_589_eq_19_l332_33203

theorem gcd_779_209_589_eq_19 : Nat.gcd 779 (Nat.gcd 209 589) = 19 := by
  sorry

end gcd_779_209_589_eq_19_l332_33203


namespace number_of_pencils_is_11_l332_33276

noncomputable def numberOfPencils (A B : ℕ) :  ℕ :=
  2 * A + 1 * B

theorem number_of_pencils_is_11 (A B : ℕ) (h1 : A + 2 * B = 16) (h2 : A + B = 9) : numberOfPencils A B = 11 :=
  sorry

end number_of_pencils_is_11_l332_33276


namespace percentage_increase_numerator_l332_33223

variable (N D : ℝ) (P : ℝ)
variable (h1 : N / D = 0.75)
variable (h2 : (N * (1 + P / 100)) / (D * 0.92) = 15 / 16)

theorem percentage_increase_numerator :
  P = 15 :=
by
  sorry

end percentage_increase_numerator_l332_33223


namespace find_number_l332_33245

theorem find_number (x : ℝ) (h : 0.8 * x = (4/5 : ℝ) * 25 + 16) : x = 45 :=
by
  sorry

end find_number_l332_33245


namespace find_pairs_l332_33282

theorem find_pairs (n p : ℕ) (hp : Prime p) (hnp : n ≤ 2 * p) (hdiv : (p - 1) * n + 1 % n^(p-1) = 0) :
  (n = 1 ∧ Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
sorry

end find_pairs_l332_33282


namespace train_passing_platform_time_l332_33251

-- Conditions
variable (l t : ℝ) -- Length of the train and time to pass the pole
variable (v : ℝ) -- Velocity of the train
variable (n : ℝ) -- Multiple of t seconds to pass the platform
variable (d_platform : ℝ) -- Length of the platform

-- Theorem statement
theorem train_passing_platform_time (h1 : d_platform = 3 * l) (h2 : v = l / t) (h3 : n = (l + d_platform) / l) :
  n = 4 := by
  sorry

end train_passing_platform_time_l332_33251


namespace find_a_if_f_is_odd_l332_33287

noncomputable def f (a x : ℝ) : ℝ := (Real.logb 2 ((a - x) / (1 + x))) 

theorem find_a_if_f_is_odd (a : ℝ) (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 1 := sorry

end find_a_if_f_is_odd_l332_33287


namespace Mary_and_Sandra_solution_l332_33259

theorem Mary_and_Sandra_solution (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) :
  (2 * 40 + 3 * 60) * n / (5 * n) = (4 * 30 * n + 80 * m) / (4 * n + m) →
  m + n = 29 :=
by
  intro h
  sorry

end Mary_and_Sandra_solution_l332_33259


namespace children_ticket_price_l332_33252

theorem children_ticket_price
  (C : ℝ)
  (adult_ticket_price : ℝ)
  (total_payment : ℝ)
  (total_tickets : ℕ)
  (children_tickets : ℕ)
  (H1 : adult_ticket_price = 8)
  (H2 : total_payment = 201)
  (H3 : total_tickets = 33)
  (H4 : children_tickets = 21)
  : C = 5 :=
by
  sorry

end children_ticket_price_l332_33252


namespace no_solution_condition_l332_33216

theorem no_solution_condition (m : ℝ) : (∀ x : ℝ, (3 * x - m) / (x - 2) ≠ 1) → m = 6 :=
by
  sorry

end no_solution_condition_l332_33216


namespace unique_solution_a_eq_sqrt_three_l332_33205

theorem unique_solution_a_eq_sqrt_three {a : ℝ} (h1 : ∀ x y : ℝ, x^2 + a * abs x + a^2 - 3 = 0 ∧ y^2 + a * abs y + a^2 - 3 = 0 → x = y)
  (h2 : a > 0) : a = Real.sqrt 3 := by
  sorry

end unique_solution_a_eq_sqrt_three_l332_33205


namespace road_network_possible_l332_33253

theorem road_network_possible (n : ℕ) :
  (n = 6 → true) ∧ (n = 1986 → false) :=
by {
  -- Proof of the statement goes here.
  sorry
}

end road_network_possible_l332_33253


namespace percentage_difference_y_less_than_z_l332_33278

-- Define the variables and the conditions
variables (x y z : ℝ)
variables (h₁ : x = 12 * y)
variables (h₂ : z = 1.2 * x)

-- Define the theorem statement
theorem percentage_difference_y_less_than_z (h₁ : x = 12 * y) (h₂ : z = 1.2 * x) :
  ((z - y) / z) * 100 = 93.06 := by
  sorry

end percentage_difference_y_less_than_z_l332_33278


namespace solution_set_of_inequality_l332_33225

theorem solution_set_of_inequality : 
  { x : ℝ | (1 : ℝ) * (2 * x + 1) < (x + 1) } = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end solution_set_of_inequality_l332_33225


namespace length_of_each_train_l332_33226

theorem length_of_each_train (L : ℝ) 
  (speed_faster : ℝ := 45 * 5 / 18) -- converting 45 km/hr to m/s
  (speed_slower : ℝ := 36 * 5 / 18) -- converting 36 km/hr to m/s
  (time : ℝ := 36) 
  (relative_speed : ℝ := speed_faster - speed_slower) 
  (total_distance : ℝ := relative_speed * time) 
  (length_each_train : ℝ := total_distance / 2) 
  : length_each_train = 45 := 
by 
  sorry

end length_of_each_train_l332_33226


namespace remainder_x_plus_3uy_plus_u_div_y_l332_33250

theorem remainder_x_plus_3uy_plus_u_div_y (x y u v : ℕ) (hx : x = u * y + v) (hu : 0 ≤ v) (hv : v < y) (huv : u + v < y) : 
  (x + 3 * u * y + u) % y = u + v :=
by
  sorry

end remainder_x_plus_3uy_plus_u_div_y_l332_33250


namespace min_right_triangles_cover_equilateral_triangle_l332_33294

theorem min_right_triangles_cover_equilateral_triangle :
  let side_length_equilateral := 12
  let legs_right_triangle := 1
  let area_equilateral := (Real.sqrt 3 / 4) * side_length_equilateral ^ 2
  let area_right_triangle := (1 / 2) * legs_right_triangle * legs_right_triangle
  let triangles_needed := area_equilateral / area_right_triangle
  triangles_needed = 72 * Real.sqrt 3 := 
by 
  sorry

end min_right_triangles_cover_equilateral_triangle_l332_33294


namespace twenty_percent_l332_33242

-- Given condition
def condition (X : ℝ) : Prop := 0.4 * X = 160

-- Theorem to show that 20% of X equals 80 given the condition
theorem twenty_percent (X : ℝ) (h : condition X) : 0.2 * X = 80 :=
by sorry

end twenty_percent_l332_33242


namespace total_clouds_count_l332_33265

def carson_clouds := 12
def little_brother_clouds := 5 * carson_clouds
def older_sister_clouds := carson_clouds / 2
def cousin_clouds := 2 * (older_sister_clouds + carson_clouds)

theorem total_clouds_count : carson_clouds + little_brother_clouds + older_sister_clouds + cousin_clouds = 114 := by
  sorry

end total_clouds_count_l332_33265


namespace fewest_four_dollar_frisbees_l332_33246

theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : x + y = 64) (h2 : 3 * x + 4 * y = 196) : y = 4 :=
by
  sorry

end fewest_four_dollar_frisbees_l332_33246


namespace rental_cost_equation_l332_33297

theorem rental_cost_equation (x : ℕ) (h : x > 0) :
  180 / x - 180 / (x + 2) = 3 :=
sorry

end rental_cost_equation_l332_33297


namespace sum_of_triangles_l332_33281

def triangle (a b c : ℕ) : ℕ :=
  (a * b) + c

theorem sum_of_triangles : 
  triangle 4 2 3 + triangle 5 3 2 = 28 :=
by
  sorry

end sum_of_triangles_l332_33281


namespace banana_permutations_l332_33208

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l332_33208


namespace algebraic_expression_value_l332_33236

theorem algebraic_expression_value : 
  ∀ (a b : ℝ), (∃ x, x = -2 ∧ a * x - b = 1) → 4 * a + 2 * b + 7 = 5 :=
by
  intros a b h
  cases' h with x hx
  cases' hx with hx1 hx2
  rw [hx1] at hx2
  sorry

end algebraic_expression_value_l332_33236


namespace calculate_b_l332_33293

open Real

theorem calculate_b (b : ℝ) (h : ∫ x in e..b, 2 / x = 6) : b = exp 4 := 
sorry

end calculate_b_l332_33293


namespace uphill_flat_road_system_l332_33274

variables {x y : ℝ}

theorem uphill_flat_road_system :
  (3 : ℝ)⁻¹ * x + (4 : ℝ)⁻¹ * y = 70 / 60 ∧
  (4 : ℝ)⁻¹ * y + (5 : ℝ)⁻¹ * x = 54 / 60 :=
sorry

end uphill_flat_road_system_l332_33274


namespace no_constant_term_l332_33237

theorem no_constant_term (n : ℕ) (hn : ∀ r : ℕ, ¬(n = (4 * r) / 3)) : n ≠ 8 :=
by 
  intro h
  sorry

end no_constant_term_l332_33237


namespace min_mn_sum_l332_33201

theorem min_mn_sum :
  ∃ (m n : ℕ), n > m ∧ m ≥ 1 ∧ 
  (1978^n % 1000 = 1978^m % 1000) ∧ (m + n = 106) :=
sorry

end min_mn_sum_l332_33201


namespace total_earnings_l332_33296

def num_members : ℕ := 20
def candy_bars_per_member : ℕ := 8
def cost_per_candy_bar : ℝ := 0.5

theorem total_earnings :
  (num_members * candy_bars_per_member * cost_per_candy_bar) = 80 :=
by
  sorry

end total_earnings_l332_33296


namespace number_of_solutions_l332_33243

noncomputable def g (x : ℝ) : ℝ := -3 * Real.sin (2 * Real.pi * x)

theorem number_of_solutions (h : -1 ≤ x ∧ x ≤ 1) : 
  (∃ s : ℕ, s = 21 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g (g (g x)) = g x) :=
sorry

end number_of_solutions_l332_33243


namespace B_fraction_l332_33200

theorem B_fraction (A_s B_s C_s : ℕ) (h1 : A_s = 600) (h2 : A_s = (2 / 5) * (B_s + C_s))
  (h3 : A_s + B_s + C_s = 1800) :
  B_s / (A_s + C_s) = 1 / 6 :=
by
  sorry

end B_fraction_l332_33200


namespace hangar_length_l332_33207

-- Define the conditions
def num_planes := 7
def length_per_plane := 40 -- in feet

-- Define the main theorem to be proven
theorem hangar_length : num_planes * length_per_plane = 280 := by
  -- Proof omitted with sorry
  sorry

end hangar_length_l332_33207


namespace shortest_side_of_similar_triangle_l332_33248

theorem shortest_side_of_similar_triangle (a1 a2 h1 h2 : ℝ)
  (h1_eq : a1 = 24)
  (h2_eq : h1 = 37)
  (h2_eq' : h2 = 74)
  (h_similar : h2 / h1 = 2)
  (h_a2_eq : a2 = 2 * Real.sqrt 793):
  a2 = 2 * Real.sqrt 793 := by
  sorry

end shortest_side_of_similar_triangle_l332_33248


namespace edward_candy_purchase_l332_33269

theorem edward_candy_purchase (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) 
  (h1 : whack_a_mole_tickets = 3) (h2 : skee_ball_tickets = 5) (h3 : candy_cost = 4) :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 := 
by 
  sorry

end edward_candy_purchase_l332_33269


namespace div_condition_l332_33220

theorem div_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  4 * (m * n + 1) % (m + n)^2 = 0 ↔ m = n := 
sorry

end div_condition_l332_33220


namespace weight_of_five_single_beds_l332_33291

-- Define the problem conditions and the goal
theorem weight_of_five_single_beds :
  ∃ S D : ℝ, (2 * S + 4 * D = 100) ∧ (D = S + 10) → (5 * S = 50) :=
by
  sorry

end weight_of_five_single_beds_l332_33291


namespace sequence_sum_l332_33206

theorem sequence_sum (S : ℕ → ℕ) (a : ℕ → ℕ) : 
  (∀ n, S n = 2^n) →
  (a 1 = S 1) ∧ (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  a 3 + a 4 = 12 :=
by
  sorry

end sequence_sum_l332_33206


namespace Alyssa_has_37_balloons_l332_33285

variable (Sandy_balloons : ℕ) (Sally_balloons : ℕ) (Total_balloons : ℕ)

-- Conditions
axiom Sandy_Condition : Sandy_balloons = 28
axiom Sally_Condition : Sally_balloons = 39
axiom Total_Condition : Total_balloons = 104

-- Definition of Alyssa's balloons
def Alyssa_balloons : ℕ := Total_balloons - (Sandy_balloons + Sally_balloons)

-- The proof statement 
theorem Alyssa_has_37_balloons 
: Alyssa_balloons Sandy_balloons Sally_balloons Total_balloons = 37 :=
by
  -- The proof body will be placed here, but we will leave it as a placeholder for now
  sorry

end Alyssa_has_37_balloons_l332_33285


namespace max_k_inequality_l332_33215

open Real

theorem max_k_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b) * (a * b + 1) * (b + 1) ≥ (27 / 4) * a * b^2 :=
by
  sorry

end max_k_inequality_l332_33215
