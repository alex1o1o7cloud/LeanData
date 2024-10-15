import Mathlib

namespace NUMINAMATH_GPT_nurses_count_l1680_168053

theorem nurses_count (total_medical_staff : ℕ) (ratio_doctors : ℕ) (ratio_nurses : ℕ) 
  (total_ratio_parts : ℕ) (h1 : total_medical_staff = 200) 
  (h2 : ratio_doctors = 4) (h3 : ratio_nurses = 6) (h4 : total_ratio_parts = ratio_doctors + ratio_nurses) :
  (ratio_nurses * total_medical_staff) / total_ratio_parts = 120 :=
by
  sorry

end NUMINAMATH_GPT_nurses_count_l1680_168053


namespace NUMINAMATH_GPT_orange_cost_l1680_168011

-- Definitions based on the conditions
def dollar_per_pound := 5 / 6
def pounds : ℕ := 18
def total_cost := pounds * dollar_per_pound

-- The statement to be proven
theorem orange_cost : total_cost = 15 :=
by
  sorry

end NUMINAMATH_GPT_orange_cost_l1680_168011


namespace NUMINAMATH_GPT_gasoline_price_increase_l1680_168038

theorem gasoline_price_increase 
  (P Q : ℝ)
  (h_intends_to_spend : ∃ M, M = P * Q * 1.15)
  (h_reduction : ∃ N, N = Q * (1 - 0.08))
  (h_equation : P * Q * 1.15 = P * (1 + x) * Q * (1 - 0.08)) :
  x = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_gasoline_price_increase_l1680_168038


namespace NUMINAMATH_GPT_circle_equation_l1680_168083

theorem circle_equation (x y : ℝ) (h : ∀ x y : ℝ, x^2 + y^2 ≥ 64) :
  x^2 + y^2 - 64 = 0 ↔ x = 0 ∧ y = 0 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l1680_168083


namespace NUMINAMATH_GPT_find_m_l1680_168010

theorem find_m (x y m : ℝ)
  (h1 : 2 * x + y = 6 * m)
  (h2 : 3 * x - 2 * y = 2 * m)
  (h3 : x / 3 - y / 5 = 4) :
  m = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1680_168010


namespace NUMINAMATH_GPT_tabletop_qualification_l1680_168045

theorem tabletop_qualification (length width diagonal : ℕ) :
  length = 60 → width = 32 → diagonal = 68 → (diagonal * diagonal = length * length + width * width) :=
by
  intros
  sorry

end NUMINAMATH_GPT_tabletop_qualification_l1680_168045


namespace NUMINAMATH_GPT_five_letter_words_with_one_consonant_l1680_168086

theorem five_letter_words_with_one_consonant :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F']
  let vowels := ['A', 'E']
  let consonants := ['B', 'C', 'D', 'F']
  let total_words := (letters.length : ℕ)^5
  let vowel_only_words := (vowels.length : ℕ)^5
  total_words - vowel_only_words = 7744 :=
by
  sorry

end NUMINAMATH_GPT_five_letter_words_with_one_consonant_l1680_168086


namespace NUMINAMATH_GPT_no_values_of_b_l1680_168067

def f (b x : ℝ) := x^2 + b * x - 1

theorem no_values_of_b : ∀ b : ℝ, ∃ x : ℝ, f b x = 3 :=
by
  intro b
  use 0  -- example, needs actual computation
  sorry

end NUMINAMATH_GPT_no_values_of_b_l1680_168067


namespace NUMINAMATH_GPT_central_angle_remains_unchanged_l1680_168059

theorem central_angle_remains_unchanged
  (r l : ℝ)
  (h_r : r > 0)
  (h_l : l > 0) :
  (l / r) = (2 * l) / (2 * r) :=
by
  sorry

end NUMINAMATH_GPT_central_angle_remains_unchanged_l1680_168059


namespace NUMINAMATH_GPT_height_of_tower_l1680_168029

-- Definitions for points and distances
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := { x := 0, y := 0, z := 0 }
def C : Point := { x := 0, y := 0, z := 129 }
def D : Point := { x := 0, y := 0, z := 258 }
def B : Point  := { x := 0, y := 305, z := 305 }

-- Given conditions
def angle_elevation_A_to_B : ℝ := 45 -- degrees
def angle_elevation_D_to_B : ℝ := 60 -- degrees
def distance_A_to_D : ℝ := 258 -- meters

-- The problem is to prove the height of the tower is 305 meters given the conditions
theorem height_of_tower : B.y = 305 :=
by
  -- This spot would contain the actual proof
  sorry

end NUMINAMATH_GPT_height_of_tower_l1680_168029


namespace NUMINAMATH_GPT_find_m_value_l1680_168092

theorem find_m_value (f g : ℝ → ℝ) (m : ℝ)
  (h1 : ∀ (x : ℝ), f x = 4 * x^2 - 3 * x + 5)
  (h2 : ∀ (x : ℝ), g x = 2 * x^2 - m * x + 8)
  (h3 : f 5 - g 5 = 15) :
  m = -17 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l1680_168092


namespace NUMINAMATH_GPT_counterexample_statement_l1680_168002

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n

theorem counterexample_statement (n : ℕ) : is_composite n ∧ (is_prime (n - 3) ∨ is_prime (n - 2)) ↔ n = 22 :=
by
  sorry

end NUMINAMATH_GPT_counterexample_statement_l1680_168002


namespace NUMINAMATH_GPT_range_of_m_l1680_168012

noncomputable def f (x m : ℝ) := Real.exp x + x^2 / m^2 - x

theorem range_of_m (m : ℝ) (hm : m ≠ 0) :
  (∀ a b : ℝ, a ∈ Set.Icc (-1) 1 -> b ∈ Set.Icc (-1) 1 -> |f a m - f b m| ≤ Real.exp 1) ↔
  (m ∈ Set.Iic (-Real.sqrt 2 / 2) ∪ Set.Ici (Real.sqrt 2 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1680_168012


namespace NUMINAMATH_GPT_solid_is_frustum_l1680_168026

-- Definitions for views
def front_view_is_isosceles_trapezoid (S : Type) : Prop := sorry
def side_view_is_isosceles_trapezoid (S : Type) : Prop := sorry
def top_view_is_concentric_circles (S : Type) : Prop := sorry

-- Define the target solid as a frustum
def is_frustum (S : Type) : Prop := sorry

-- The theorem statement
theorem solid_is_frustum
  (S : Type) 
  (h1 : front_view_is_isosceles_trapezoid S)
  (h2 : side_view_is_isosceles_trapezoid S)
  (h3 : top_view_is_concentric_circles S) :
  is_frustum S :=
sorry

end NUMINAMATH_GPT_solid_is_frustum_l1680_168026


namespace NUMINAMATH_GPT_usual_eggs_accepted_l1680_168015

theorem usual_eggs_accepted (A R : ℝ) (h1 : A / R = 1 / 4) (h2 : (A + 12) / (R - 4) = 99 / 1) (h3 : A + R = 400) :
  A = 392 :=
by
  sorry

end NUMINAMATH_GPT_usual_eggs_accepted_l1680_168015


namespace NUMINAMATH_GPT_function_identity_l1680_168062

variables {R : Type*} [LinearOrderedField R]

-- Define real-valued functions f, g, h
variables (f g h : R → R)

-- Define function composition and multiplication
def comp (f g : R → R) (x : R) := f (g x)
def mul (f g : R → R) (x : R) := f x * g x

-- The statement to prove
theorem function_identity (x : R) : 
  comp (mul f g) h x = mul (comp f h) (comp g h) x :=
sorry

end NUMINAMATH_GPT_function_identity_l1680_168062


namespace NUMINAMATH_GPT_johns_profit_l1680_168048

def profit (n : ℕ) (p c : ℕ) : ℕ :=
  n * p - c

theorem johns_profit :
  profit 20 15 100 = 200 :=
by
  sorry

end NUMINAMATH_GPT_johns_profit_l1680_168048


namespace NUMINAMATH_GPT_distance_planes_A_B_l1680_168008

noncomputable def distance_between_planes : ℝ :=
  let d1 := 1
  let d2 := 2
  let a := 1
  let b := 1
  let c := 1
  (|d2 - d1|) / (Real.sqrt (a^2 + b^2 + c^2))

theorem distance_planes_A_B :
  let A := fun (x y z : ℝ) => x + y + z = 1
  let B := fun (x y z : ℝ) => x + y + z = 2
  distance_between_planes = 1 / Real.sqrt 3 :=
  by
    -- Proof steps will be here
    sorry

end NUMINAMATH_GPT_distance_planes_A_B_l1680_168008


namespace NUMINAMATH_GPT_nest_building_twig_count_l1680_168007

theorem nest_building_twig_count
    (total_twigs_to_weave : ℕ)
    (found_twigs : ℕ)
    (remaining_twigs : ℕ)
    (n : ℕ)
    (x : ℕ)
    (h1 : total_twigs_to_weave = 12 * x)
    (h2 : found_twigs = (total_twigs_to_weave) / 3)
    (h3 : remaining_twigs = 48)
    (h4 : found_twigs + remaining_twigs = total_twigs_to_weave) :
    x = 18 := 
by
  sorry

end NUMINAMATH_GPT_nest_building_twig_count_l1680_168007


namespace NUMINAMATH_GPT_x_gt_1_sufficient_not_necessary_x_squared_gt_1_l1680_168016

variable {x : ℝ}

-- Condition: $x > 1$
def condition_x_gt_1 (x : ℝ) : Prop := x > 1

-- Condition: $x^2 > 1$
def condition_x_squared_gt_1 (x : ℝ) : Prop := x^2 > 1

-- Theorem: Prove that $x > 1$ is a sufficient but not necessary condition for $x^2 > 1$
theorem x_gt_1_sufficient_not_necessary_x_squared_gt_1 :
  (condition_x_gt_1 x → condition_x_squared_gt_1 x) ∧ (¬ ∀ x, condition_x_squared_gt_1 x → condition_x_gt_1 x) :=
sorry

end NUMINAMATH_GPT_x_gt_1_sufficient_not_necessary_x_squared_gt_1_l1680_168016


namespace NUMINAMATH_GPT_train_speed_l1680_168022

theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) (h_train_length : train_length = 100) (h_bridge_length : bridge_length = 300) (h_crossing_time : crossing_time = 12) : 
  (train_length + bridge_length) / crossing_time = 33.33 := 
by 
  -- sorry allows us to skip the proof
  sorry

end NUMINAMATH_GPT_train_speed_l1680_168022


namespace NUMINAMATH_GPT_tigers_wins_l1680_168049

def totalGames : ℕ := 56
def losses : ℕ := 12
def ties : ℕ := losses / 2

theorem tigers_wins : totalGames - losses - ties = 38 := by
  sorry

end NUMINAMATH_GPT_tigers_wins_l1680_168049


namespace NUMINAMATH_GPT_smallest_k_DIVISIBLE_by_3_67_l1680_168050

theorem smallest_k_DIVISIBLE_by_3_67 :
  ∃ k : ℕ, (∀ n : ℕ, (2016^k % 3^67 = 0 ∧ (2016^n % 3^67 = 0 → k ≤ n)) ∧ k = 34) := by
  sorry

end NUMINAMATH_GPT_smallest_k_DIVISIBLE_by_3_67_l1680_168050


namespace NUMINAMATH_GPT_final_milk_concentration_l1680_168080

theorem final_milk_concentration
  (initial_mixture_volume : ℝ)
  (initial_milk_volume : ℝ)
  (replacement_volume : ℝ)
  (replacements_count : ℕ)
  (final_milk_volume : ℝ) :
  initial_mixture_volume = 100 → 
  initial_milk_volume = 36 → 
  replacement_volume = 50 →
  replacements_count = 2 →
  final_milk_volume = 9 →
  (final_milk_volume / initial_mixture_volume * 100) = 9 :=
by
  sorry

end NUMINAMATH_GPT_final_milk_concentration_l1680_168080


namespace NUMINAMATH_GPT_product_probability_probability_one_l1680_168072

def S : Set Int := {13, 57}

theorem product_probability (a b : Int) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : a ≠ b) : 
  (a * b > 15) := 
by 
  sorry

theorem probability_one : 
  (∃ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a * b > 15) ∧ 
  (∀ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b → a * b > 15) :=
by 
  sorry

end NUMINAMATH_GPT_product_probability_probability_one_l1680_168072


namespace NUMINAMATH_GPT_fraction_multiplication_l1680_168034

theorem fraction_multiplication (x : ℚ) (h : x = 236 / 100) : x * 3 = 177 / 25 :=
by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_l1680_168034


namespace NUMINAMATH_GPT_min_value_xyz_l1680_168079

theorem min_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 8) : 
  x + 2 * y + 4 * z ≥ 12 := sorry

end NUMINAMATH_GPT_min_value_xyz_l1680_168079


namespace NUMINAMATH_GPT_distance_travelled_downstream_l1680_168036

theorem distance_travelled_downstream :
  let speed_boat_still_water := 42 -- km/hr
  let rate_current := 7 -- km/hr
  let time_travelled_min := 44 -- minutes
  let time_travelled_hrs := time_travelled_min / 60.0 -- converting minutes to hours
  let effective_speed_downstream := speed_boat_still_water + rate_current -- km/hr
  let distance_downstream := effective_speed_downstream * time_travelled_hrs
  distance_downstream = 35.93 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_distance_travelled_downstream_l1680_168036


namespace NUMINAMATH_GPT_number_of_buses_required_l1680_168068

def total_seats : ℕ := 28
def students_per_bus : ℝ := 14.0

theorem number_of_buses_required :
  (total_seats / students_per_bus) = 2 := 
by
  -- The actual proof is intentionally left out.
  sorry

end NUMINAMATH_GPT_number_of_buses_required_l1680_168068


namespace NUMINAMATH_GPT_andrew_made_35_sandwiches_l1680_168058

-- Define the number of friends and sandwiches per friend
def num_friends : ℕ := 7
def sandwiches_per_friend : ℕ := 5

-- Define the total number of sandwiches and prove it equals 35
def total_sandwiches : ℕ := num_friends * sandwiches_per_friend

theorem andrew_made_35_sandwiches : total_sandwiches = 35 := by
  sorry

end NUMINAMATH_GPT_andrew_made_35_sandwiches_l1680_168058


namespace NUMINAMATH_GPT_greatest_divisor_of_remainders_l1680_168025

theorem greatest_divisor_of_remainders (x : ℕ) :
  (1442 % x = 12) ∧ (1816 % x = 6) ↔ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_remainders_l1680_168025


namespace NUMINAMATH_GPT_probability_green_dinosaur_or_blue_robot_l1680_168061

theorem probability_green_dinosaur_or_blue_robot (t: ℕ) (blue_dinosaurs green_robots blue_robots: ℕ) 
(h1: blue_dinosaurs = 16) (h2: green_robots = 14) (h3: blue_robots = 36) (h4: t = 93):
  t = 93 → (blue_dinosaurs = 16) → (green_robots = 14) → (blue_robots = 36) → 
  (∃ green_dinosaurs: ℕ, t = blue_dinosaurs + green_robots + blue_robots + green_dinosaurs ∧ 
    (∃ k: ℕ, k = (green_dinosaurs + blue_robots) / (t / 31) ∧ k = 21 / 31)) := sorry

end NUMINAMATH_GPT_probability_green_dinosaur_or_blue_robot_l1680_168061


namespace NUMINAMATH_GPT_original_number_l1680_168004

theorem original_number (x : ℝ) (h : 1.4 * x = 700) : x = 500 :=
sorry

end NUMINAMATH_GPT_original_number_l1680_168004


namespace NUMINAMATH_GPT_terminating_decimal_expansion_of_17_div_625_l1680_168065

theorem terminating_decimal_expansion_of_17_div_625 : 
  ∃ d : ℚ, d = 17 / 625 ∧ d = 0.0272 :=
by
  sorry

end NUMINAMATH_GPT_terminating_decimal_expansion_of_17_div_625_l1680_168065


namespace NUMINAMATH_GPT_x_axis_line_l1680_168098

variable (A B C : ℝ)

theorem x_axis_line (h : ∀ x : ℝ, A * x + B * 0 + C = 0) : B ≠ 0 ∧ A = 0 ∧ C = 0 := by
  sorry

end NUMINAMATH_GPT_x_axis_line_l1680_168098


namespace NUMINAMATH_GPT_proj_eq_line_eqn_l1680_168003

theorem proj_eq_line_eqn (x y : ℝ)
  (h : (6 * x + 3 * y) * 6 / 45 = -3 ∧ (6 * x + 3 * y) * 3 / 45 = -3 / 2) :
  y = -2 * x - 15 / 2 :=
by
  sorry

end NUMINAMATH_GPT_proj_eq_line_eqn_l1680_168003


namespace NUMINAMATH_GPT_range_of_a_l1680_168039

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x - 1/2

noncomputable def g (x a : ℝ) : ℝ := x^2 + Real.log (x + a)

theorem range_of_a : 
  (∀ x ∈ Set.Iio 0, ∃ y, f x = g y a ∧ y = -x) →
  a < Real.sqrt (Real.exp 1) :=
  sorry

end NUMINAMATH_GPT_range_of_a_l1680_168039


namespace NUMINAMATH_GPT_max_robot_weight_l1680_168082

-- Definitions of the given conditions
def standard_robot_weight : ℕ := 100
def battery_weight : ℕ := 20
def min_payload : ℕ := 10
def max_payload : ℕ := 25
def min_robot_weight_extra : ℕ := 5
def min_robot_weight : ℕ := standard_robot_weight + min_robot_weight_extra

-- Definition for total minimum weight of the robot
def min_total_weight : ℕ := min_robot_weight + battery_weight + min_payload

-- Proposition for the maximum weight condition
theorem max_robot_weight :
  2 * min_total_weight = 270 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_max_robot_weight_l1680_168082


namespace NUMINAMATH_GPT_fraction_halfway_between_l1680_168085

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end NUMINAMATH_GPT_fraction_halfway_between_l1680_168085


namespace NUMINAMATH_GPT_serenity_total_shoes_l1680_168075

def pairs_of_shoes : ℕ := 3
def shoes_per_pair : ℕ := 2

theorem serenity_total_shoes : pairs_of_shoes * shoes_per_pair = 6 := by
  sorry

end NUMINAMATH_GPT_serenity_total_shoes_l1680_168075


namespace NUMINAMATH_GPT_buses_needed_l1680_168033

def total_students : ℕ := 111
def seats_per_bus : ℕ := 3

theorem buses_needed : total_students / seats_per_bus = 37 :=
by
  sorry

end NUMINAMATH_GPT_buses_needed_l1680_168033


namespace NUMINAMATH_GPT_evaluate_expression_l1680_168019

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 5 * x + 7

theorem evaluate_expression : 3 * f 2 - 2 * f (-2) = -31 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1680_168019


namespace NUMINAMATH_GPT_factory_workers_l1680_168081

-- Define parameters based on given conditions
def sewing_factory_x : ℤ := 1995
def shoe_factory_y : ℤ := 1575

-- Conditions based on the problem setup
def shoe_factory_of_sewing_factory := (15 * sewing_factory_x) / 19 = shoe_factory_y
def shoe_factory_plan_exceed := (3 * shoe_factory_y) / 7 < 1000
def sewing_factory_plan_exceed := (3 * sewing_factory_x) / 5 > 1000

-- Theorem stating the problem's assertion
theorem factory_workers (x y : ℤ) 
  (h1 : (15 * x) / 19 = y)
  (h2 : (4 * y) / 7 < 1000)
  (h3 : (3 * x) / 5 > 1000) : 
  x = 1995 ∧ y = 1575 :=
sorry

end NUMINAMATH_GPT_factory_workers_l1680_168081


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l1680_168066

theorem asymptotes_of_hyperbola (a b : ℝ) (h_cond1 : a > b) (h_cond2 : b > 0) 
  (h_eq_ell : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h_eq_hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (h_product : ∀ e1 e2 : ℝ, (e1 = Real.sqrt (1 - (b^2 / a^2))) → 
                (e2 = Real.sqrt (1 + (b^2 / a^2))) → 
                (e1 * e2 = Real.sqrt 3 / 2)) :
  ∀ x y : ℝ, x + Real.sqrt 2 * y = 0 ∨ x - Real.sqrt 2 * y = 0 :=
sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l1680_168066


namespace NUMINAMATH_GPT_decimal_to_base13_185_l1680_168094

theorem decimal_to_base13_185 : 
  ∀ n : ℕ, n = 185 → 
      ∃ a b c : ℕ, a * 13^2 + b * 13 + c = n ∧ 0 ≤ a ∧ a < 13 ∧ 0 ≤ b ∧ b < 13 ∧ 0 ≤ c ∧ c < 13 ∧ (a, b, c) = (1, 1, 3) := 
by
  intros n hn
  use 1, 1, 3
  sorry

end NUMINAMATH_GPT_decimal_to_base13_185_l1680_168094


namespace NUMINAMATH_GPT_range_of_m_l1680_168055

noncomputable def quadratic_function (m x : ℝ) : ℝ := (m-2) * x^2 + 2 * m * x - (3 - m)

theorem range_of_m (m : ℝ) (h_vertex_third_quadrant : (-(m) / (m-2) < 0) ∧ ((-5)*m + 6) / (m-2) < 0)
                   (h_parabola_opens_upwards : m - 2 > 0)
                   (h_intersects_negative_y_axis : m < 3) : 2 < m ∧ m < 3 :=
by {
    sorry
}

end NUMINAMATH_GPT_range_of_m_l1680_168055


namespace NUMINAMATH_GPT_total_area_of_figure_l1680_168044

theorem total_area_of_figure :
  let rect1_height := 7
  let rect1_width := 6
  let rect2_height := 2
  let rect2_width := 6
  let rect3_height := 5
  let rect3_width := 4
  let rect4_height := 3
  let rect4_width := 5
  let area1 := rect1_height * rect1_width
  let area2 := rect2_height * rect2_width
  let area3 := rect3_height * rect3_width
  let area4 := rect4_height * rect4_width
  let total_area := area1 + area2 + area3 + area4
  total_area = 89 := by
  -- Definitions
  let rect1_height := 7
  let rect1_width := 6
  let rect2_height := 2
  let rect2_width := 6
  let rect3_height := 5
  let rect3_width := 4
  let rect4_height := 3
  let rect4_width := 5
  let area1 := rect1_height * rect1_width
  let area2 := rect2_height * rect2_width
  let area3 := rect3_height * rect3_width
  let area4 := rect4_height * rect4_width
  let total_area := area1 + area2 + area3 + area4
  -- Proof
  sorry

end NUMINAMATH_GPT_total_area_of_figure_l1680_168044


namespace NUMINAMATH_GPT_not_square_of_expression_l1680_168088

theorem not_square_of_expression (n : ℕ) (h : n > 0) : ¬ ∃ m : ℤ, m * m = 2 * n * n + 2 - n :=
by
  sorry

end NUMINAMATH_GPT_not_square_of_expression_l1680_168088


namespace NUMINAMATH_GPT_volume_remaining_proof_l1680_168073

noncomputable def volume_remaining_part (v_original v_total_small : ℕ) : ℕ := v_original - v_total_small

def original_edge_length := 9
def small_edge_length := 3
def num_edges := 12

def volume_original := original_edge_length ^ 3
def volume_small := small_edge_length ^ 3
def volume_total_small := num_edges * volume_small

theorem volume_remaining_proof : volume_remaining_part volume_original volume_total_small = 405 := by
  sorry

end NUMINAMATH_GPT_volume_remaining_proof_l1680_168073


namespace NUMINAMATH_GPT_quadratic_function_symmetry_l1680_168095

-- Define the quadratic function
def f (x : ℝ) (b c : ℝ) : ℝ := -x^2 + b * x + c

-- State the problem as a theorem
theorem quadratic_function_symmetry (b c : ℝ) (h_symm : ∀ x, f x b c = f (4 - x) b c) :
  f 2 b c > f 1 b c ∧ f 1 b c > f 4 b c :=
by
  -- Include a placeholder for the proof
  sorry

end NUMINAMATH_GPT_quadratic_function_symmetry_l1680_168095


namespace NUMINAMATH_GPT_find_algebraic_expression_value_l1680_168014

theorem find_algebraic_expression_value (x : ℝ) (h : 3 * x^2 + 5 * x + 1 = 0) : 
  (x + 2) ^ 2 + x * (2 * x + 1) = 3 := 
by 
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_find_algebraic_expression_value_l1680_168014


namespace NUMINAMATH_GPT_problem_statement_l1680_168006

variable {a b c x y z : ℝ}
variable (h1 : 17 * x + b * y + c * z = 0)
variable (h2 : a * x + 29 * y + c * z = 0)
variable (h3 : a * x + b * y + 53 * z = 0)
variable (ha : a ≠ 17)
variable (hx : x ≠ 0)

theorem problem_statement : 
  (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1680_168006


namespace NUMINAMATH_GPT_smallest_n_divides_999_l1680_168078

/-- 
Given \( 1 \leq n < 1000 \), \( n \) divides 999, and \( n+6 \) divides 99,
prove that the smallest possible value of \( n \) is 27.
 -/
theorem smallest_n_divides_999 (n : ℕ) 
  (h1 : 1 ≤ n) 
  (h2 : n < 1000) 
  (h3 : n ∣ 999) 
  (h4 : n + 6 ∣ 99) : 
  n = 27 :=
  sorry

end NUMINAMATH_GPT_smallest_n_divides_999_l1680_168078


namespace NUMINAMATH_GPT_domain_and_range_of_f_l1680_168051

noncomputable def f (a x : ℝ) : ℝ := Real.log (a - a * x) / Real.log a

theorem domain_and_range_of_f (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, a - a * x > 0 → x < 1) ∧ 
  (∀ t : ℝ, 0 < t ∧ t < a → ∃ x : ℝ, t = a - a * x) :=
by
  sorry

end NUMINAMATH_GPT_domain_and_range_of_f_l1680_168051


namespace NUMINAMATH_GPT_chloe_paid_per_dozen_l1680_168091

-- Definitions based on conditions
def half_dozen_sale_price : ℕ := 30
def profit : ℕ := 500
def dozens_sold : ℕ := 50
def full_dozen_sale_price := 2 * half_dozen_sale_price
def total_revenue := dozens_sold * full_dozen_sale_price
def total_cost := total_revenue - profit

-- Proof problem
theorem chloe_paid_per_dozen : (total_cost / dozens_sold) = 50 :=
by
  sorry

end NUMINAMATH_GPT_chloe_paid_per_dozen_l1680_168091


namespace NUMINAMATH_GPT_value_of_expression_l1680_168063

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) : 18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1680_168063


namespace NUMINAMATH_GPT_square_area_l1680_168018

theorem square_area (x : ℚ) (side_length : ℚ) 
  (h1 : side_length = 3 * x - 12) 
  (h2 : side_length = 24 - 2 * x) : 
  side_length ^ 2 = 92.16 := 
by 
  sorry

end NUMINAMATH_GPT_square_area_l1680_168018


namespace NUMINAMATH_GPT_find_expression_for_f_l1680_168040

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem find_expression_for_f (x : ℝ) (h : x ≠ -1) 
    (hf : f ((1 - x) / (1 + x)) = x) : 
    f x = (1 - x) / (1 + x) :=
sorry

end NUMINAMATH_GPT_find_expression_for_f_l1680_168040


namespace NUMINAMATH_GPT_length_of_segment_AB_l1680_168023

-- Define the parabola and its properties
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus of the parabola y^2 = 4x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define the midpoint condition
def midpoint_condition (A B : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ C.1 = 3

-- Main statement of the problem
theorem length_of_segment_AB
  (A B : ℝ × ℝ)
  (hA : parabola_equation A.1 A.2)
  (hB : parabola_equation B.1 B.2)
  (C : ℝ × ℝ)
  (hfoc : focus (1, 0))
  (hm : midpoint_condition A B C) :
  dist A B = 8 :=
by sorry

end NUMINAMATH_GPT_length_of_segment_AB_l1680_168023


namespace NUMINAMATH_GPT_total_chairs_l1680_168041

/-- Susan loves chairs. In her house, there are red chairs, yellow chairs, blue chairs, and green chairs.
    There are 5 red chairs. There are 4 times as many yellow chairs as red chairs.
    There are 2 fewer blue chairs than yellow chairs. The number of green chairs is half the sum of the number of red chairs and blue chairs (rounded down).
    We want to determine the total number of chairs in Susan's house. -/
theorem total_chairs (r y b g : ℕ) 
  (hr : r = 5)
  (hy : y = 4 * r) 
  (hb : b = y - 2) 
  (hg : g = (r + b) / 2) :
  r + y + b + g = 54 := 
sorry

end NUMINAMATH_GPT_total_chairs_l1680_168041


namespace NUMINAMATH_GPT_smallest_prime_x_l1680_168027

-- Define prime number checker
def is_prime (n : ℕ) : Prop := n ≠ 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem conditions and proof goal
theorem smallest_prime_x 
  (x y z : ℕ) 
  (hx : is_prime x)
  (hy : is_prime y)
  (hz : is_prime z)
  (hxy : x ≠ y)
  (hxz : x ≠ z)
  (hyz : y ≠ z)
  (hd : ∀ d : ℕ, d ∣ (x * x * y * z) ↔ (d = 1 ∨ d = x ∨ d = x * x ∨ d = y ∨ d = x * y ∨ d = x * x * y ∨ d = z ∨ d = x * z ∨ d = x * x * z ∨ d = y * z ∨ d = x * y * z ∨ d = x * x * y * z)) 
  : x = 2 := 
sorry

end NUMINAMATH_GPT_smallest_prime_x_l1680_168027


namespace NUMINAMATH_GPT_area_of_yard_l1680_168028

theorem area_of_yard (L W : ℕ) (h1 : L = 40) (h2 : L + 2 * W = 64) : L * W = 480 := by
  sorry

end NUMINAMATH_GPT_area_of_yard_l1680_168028


namespace NUMINAMATH_GPT_color_cube_color_octahedron_l1680_168047

theorem color_cube (colors : Fin 6) : ∃ (ways : Nat), ways = 30 :=
  sorry

theorem color_octahedron (colors : Fin 8) : ∃ (ways : Nat), ways = 1680 :=
  sorry

end NUMINAMATH_GPT_color_cube_color_octahedron_l1680_168047


namespace NUMINAMATH_GPT_proportion_solution_l1680_168060

theorem proportion_solution (x : ℝ) (h : 0.6 / x = 5 / 8) : x = 0.96 :=
by 
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_proportion_solution_l1680_168060


namespace NUMINAMATH_GPT_principal_amount_borrowed_l1680_168032

theorem principal_amount_borrowed (P R T SI : ℕ) (h₀ : SI = (P * R * T) / 100) (h₁ : SI = 5400) (h₂ : R = 12) (h₃ : T = 3) : P = 15000 :=
by
  sorry

end NUMINAMATH_GPT_principal_amount_borrowed_l1680_168032


namespace NUMINAMATH_GPT_distance_of_route_l1680_168056

theorem distance_of_route (Vq : ℝ) (Vy : ℝ) (D : ℝ) (h1 : Vy = 1.5 * Vq) (h2 : D = Vq * 2) (h3 : D = Vy * 1.3333333333333333) : D = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_distance_of_route_l1680_168056


namespace NUMINAMATH_GPT_find_investment_period_l1680_168071

variable (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ)

theorem find_investment_period (hP : P = 12000)
                               (hr : r = 0.10)
                               (hn : n = 2)
                               (hA : A = 13230) :
                               ∃ t : ℝ, A = P * (1 + r / n)^(n * t) ∧ t = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_investment_period_l1680_168071


namespace NUMINAMATH_GPT_work_days_for_A_and_B_l1680_168021

theorem work_days_for_A_and_B (W_A W_B : ℝ) (h1 : W_A = (1/2) * W_B) (h2 : W_B = 1/21) : 
  1 / (W_A + W_B) = 14 := by 
  sorry

end NUMINAMATH_GPT_work_days_for_A_and_B_l1680_168021


namespace NUMINAMATH_GPT_alpha_half_quadrant_l1680_168090

theorem alpha_half_quadrant (k : ℤ) (α : ℝ)
  (h : 2 * k * Real.pi - Real.pi / 2 < α ∧ α < 2 * k * Real.pi) :
  (∃ n : ℤ, 2 * n * Real.pi - Real.pi / 4 < α / 2 ∧ α / 2 < 2 * n * Real.pi) ∨
  (∃ n : ℤ, (2 * n + 1) * Real.pi - Real.pi / 4 < α / 2 ∧ α / 2 < (2 * n + 1) * Real.pi) :=
sorry

end NUMINAMATH_GPT_alpha_half_quadrant_l1680_168090


namespace NUMINAMATH_GPT_solve_for_x_l1680_168024

theorem solve_for_x (x : ℝ) (h : (x - 5)^4 = (1 / 16)⁻¹) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1680_168024


namespace NUMINAMATH_GPT_total_legs_in_christophers_room_l1680_168001

def total_legs (num_spiders num_legs_per_spider num_ants num_butterflies num_beetles num_legs_per_insect : ℕ) : ℕ :=
  let spider_legs := num_spiders * num_legs_per_spider
  let ant_legs := num_ants * num_legs_per_insect
  let butterfly_legs := num_butterflies * num_legs_per_insect
  let beetle_legs := num_beetles * num_legs_per_insect
  spider_legs + ant_legs + butterfly_legs + beetle_legs

theorem total_legs_in_christophers_room : total_legs 12 8 10 5 5 6 = 216 := by
  -- Calculation and reasoning omitted
  sorry

end NUMINAMATH_GPT_total_legs_in_christophers_room_l1680_168001


namespace NUMINAMATH_GPT_monotone_f_solve_inequality_range_of_a_l1680_168030

noncomputable def e := Real.exp 1
noncomputable def f (x : ℝ) : ℝ := e^x + 1/(e^x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.log ((3 - a) * (f x - 1/e^x) + 1) - Real.log (3 * a) - 2 * x

-- Part 1: Monotonicity of f(x)
theorem monotone_f : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 :=
by sorry

-- Part 2: Solving the inequality f(2x) ≥ f(x + 1)
theorem solve_inequality : ∀ x : ℝ, f (2 * x) ≥ f (x + 1) ↔ x ≥ 1 ∨ x ≤ -1 / 3 :=
by sorry

-- Part 3: Finding the range of a
theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x → g x a ≤ 0) ↔ 1 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_GPT_monotone_f_solve_inequality_range_of_a_l1680_168030


namespace NUMINAMATH_GPT_area_of_quadrilateral_PQRS_l1680_168076

noncomputable def calculate_area_of_quadrilateral_PQRS (PQ PR : ℝ) (PS_corrected : ℝ) : ℝ :=
  let area_ΔPQR := (1/2) * PQ * PR
  let RS := Real.sqrt (PR^2 - PQ^2)
  let area_ΔPRS := (1/2) * PR * RS
  area_ΔPQR + area_ΔPRS

theorem area_of_quadrilateral_PQRS :
  let PQ := 8
  let PR := 10
  let PS_corrected := Real.sqrt (PQ^2 + PR^2)
  calculate_area_of_quadrilateral_PQRS PQ PR PS_corrected = 70 := 
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_PQRS_l1680_168076


namespace NUMINAMATH_GPT_new_prism_volume_l1680_168013

theorem new_prism_volume (L W H : ℝ) 
  (h_volume : L * W * H = 54)
  (L_new : ℝ := 2 * L)
  (W_new : ℝ := 3 * W)
  (H_new : ℝ := 1.5 * H) :
  L_new * W_new * H_new = 486 := 
by
  sorry

end NUMINAMATH_GPT_new_prism_volume_l1680_168013


namespace NUMINAMATH_GPT_cube_side_length_l1680_168037

theorem cube_side_length (n : ℕ) (h : n^3 - (n-2)^3 = 98) : n = 5 :=
by sorry

end NUMINAMATH_GPT_cube_side_length_l1680_168037


namespace NUMINAMATH_GPT_cos_pi_plus_2alpha_l1680_168017

theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin ((Real.pi / 2) + α) = 1 / 3) : Real.cos (Real.pi + 2 * α) = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_plus_2alpha_l1680_168017


namespace NUMINAMATH_GPT_parallel_lines_condition_l1680_168057

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y: ℝ, (x + a * y + 6 = 0) ↔ ((a - 2) * x + 3 * y + 2 * a = 0)) ↔ a = -1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_condition_l1680_168057


namespace NUMINAMATH_GPT_confidence_of_independence_test_l1680_168009

-- Define the observed value of K^2
def K2_obs : ℝ := 5

-- Define the critical value(s) of K^2 for different confidence levels
def K2_critical_0_05 : ℝ := 3.841
def K2_critical_0_01 : ℝ := 6.635

-- Define the confidence levels corresponding to the critical values
def P_K2_ge_3_841 : ℝ := 0.05
def P_K2_ge_6_635 : ℝ := 0.01

-- Define the statement to be proved: there is 95% confidence that "X and Y are related".
theorem confidence_of_independence_test
  (K2_obs K2_critical_0_05 P_K2_ge_3_841 : ℝ)
  (hK2_obs_gt_critical : K2_obs > K2_critical_0_05)
  (hP : P_K2_ge_3_841 = 0.05) :
  1 - P_K2_ge_3_841 = 0.95 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_confidence_of_independence_test_l1680_168009


namespace NUMINAMATH_GPT_smallest_k_674_l1680_168093

theorem smallest_k_674 :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 2017) → (S.card = 674) → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (672 < a - b) ∧ (a - b < 1344) ∨ (672 < b - a) ∧ (b - a < 1344) :=
by sorry

end NUMINAMATH_GPT_smallest_k_674_l1680_168093


namespace NUMINAMATH_GPT_total_monkeys_is_correct_l1680_168070

-- Define the parameters
variables (m n : ℕ)

-- Define the conditions as separate definitions
def monkeys_on_n_bicycles : ℕ := 3 * n
def monkeys_on_remaining_bicycles : ℕ := 5 * (m - n)

-- Define the total number of monkeys
def total_monkeys : ℕ := monkeys_on_n_bicycles n + monkeys_on_remaining_bicycles m n

-- State the theorem
theorem total_monkeys_is_correct : total_monkeys m n = 5 * m - 2 * n :=
by
  sorry

end NUMINAMATH_GPT_total_monkeys_is_correct_l1680_168070


namespace NUMINAMATH_GPT_company_pays_per_box_per_month_l1680_168046

/-
  Given:
  - The dimensions of each box are 15 inches by 12 inches by 10 inches
  - The total volume occupied by all boxes is 1,080,000 cubic inches
  - The total cost for record storage per month is $480

  Prove:
  - The company pays $0.80 per box per month for record storage
-/

theorem company_pays_per_box_per_month :
  let length := 15
  let width := 12
  let height := 10
  let box_volume := length * width * height
  let total_volume := 1080000
  let total_cost := 480
  let num_boxes := total_volume / box_volume
  let cost_per_box_per_month := total_cost / num_boxes
  cost_per_box_per_month = 0.80 :=
by
  let length := 15
  let width := 12
  let height := 10
  let box_volume := length * width * height
  let total_volume := 1080000
  let total_cost := 480
  let num_boxes := total_volume / box_volume
  let cost_per_box_per_month := total_cost / num_boxes
  sorry

end NUMINAMATH_GPT_company_pays_per_box_per_month_l1680_168046


namespace NUMINAMATH_GPT_george_hours_tuesday_l1680_168054

def wage_per_hour := 5
def hours_monday := 7
def total_earnings := 45

theorem george_hours_tuesday : ∃ (hours_tuesday : ℕ), 
  hours_tuesday = (total_earnings - (hours_monday * wage_per_hour)) / wage_per_hour := 
by
  sorry

end NUMINAMATH_GPT_george_hours_tuesday_l1680_168054


namespace NUMINAMATH_GPT_minimal_colors_l1680_168074

def complete_graph (n : ℕ) := Type

noncomputable def color_edges (G : complete_graph 2015) := ℕ → ℕ → ℕ

theorem minimal_colors (G : complete_graph 2015) (color : color_edges G) :
  (∀ {u v w : ℕ} (h1 : u ≠ v) (h2 : v ≠ w) (h3 : w ≠ u), color u v ≠ color v w ∧ color u v ≠ color u w ∧ color u w ≠ color v w) →
  ∃ C: ℕ, C = 2015 := 
sorry

end NUMINAMATH_GPT_minimal_colors_l1680_168074


namespace NUMINAMATH_GPT_GCF_LCM_18_30_10_45_eq_90_l1680_168000

-- Define LCM and GCF functions
def LCM (a b : ℕ) := a / Nat.gcd a b * b
def GCF (a b : ℕ) := Nat.gcd a b

-- Define the problem
theorem GCF_LCM_18_30_10_45_eq_90 : 
  GCF (LCM 18 30) (LCM 10 45) = 90 := by
sorry

end NUMINAMATH_GPT_GCF_LCM_18_30_10_45_eq_90_l1680_168000


namespace NUMINAMATH_GPT_prove_remainder_l1680_168042

def problem_statement : Prop := (33333332 % 8 = 4)

theorem prove_remainder : problem_statement := 
by
  sorry

end NUMINAMATH_GPT_prove_remainder_l1680_168042


namespace NUMINAMATH_GPT_two_digit_number_condition_l1680_168031

theorem two_digit_number_condition :
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 2 = 0 ∧ (n + 1) % 3 = 0 ∧ (n + 2) % 4 = 0 ∧ (n + 3) % 5 = 0 ∧ n = 62 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_number_condition_l1680_168031


namespace NUMINAMATH_GPT_book_total_pages_l1680_168096

theorem book_total_pages (x : ℕ) (h1 : x * (3 / 5) * (3 / 8) = 36) : x = 120 := 
by
  -- Proof should be supplied here, but we only need the statement
  sorry

end NUMINAMATH_GPT_book_total_pages_l1680_168096


namespace NUMINAMATH_GPT_large_number_divisible_by_12_l1680_168043

theorem large_number_divisible_by_12 : (Nat.digits 10 ((71 * 10^168) + (72 * 10^166) + (73 * 10^164) + (74 * 10^162) + (75 * 10^160) + (76 * 10^158) + (77 * 10^156) + (78 * 10^154) + (79 * 10^152) + (80 * 10^150) + (81 * 10^148) + (82 * 10^146) + (83 * 10^144) + 84)).foldl (λ x y => x * 10 + y) 0 % 12 = 0 := 
sorry

end NUMINAMATH_GPT_large_number_divisible_by_12_l1680_168043


namespace NUMINAMATH_GPT_john_weekly_earnings_before_raise_l1680_168099

theorem john_weekly_earnings_before_raise :
  ∀(x : ℝ), (70 = 1.0769 * x) → x = 64.99 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_john_weekly_earnings_before_raise_l1680_168099


namespace NUMINAMATH_GPT_min_value_arith_geo_seq_l1680_168077

theorem min_value_arith_geo_seq (A B C D : ℕ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h4 : 0 < D)
  (h_arith : C - B = B - A) (h_geo : C * C = B * D) (h_frac : 4 * C = 7 * B) :
  A + B + C + D = 97 :=
sorry

end NUMINAMATH_GPT_min_value_arith_geo_seq_l1680_168077


namespace NUMINAMATH_GPT_tank_full_capacity_l1680_168064

variable (T : ℝ) -- Define T as a real number representing the total capacity of the tank.

-- The main condition: (3 / 4) * T + 5 = (7 / 8) * T
axiom condition : (3 / 4) * T + 5 = (7 / 8) * T

-- Proof statement: Prove that T = 40
theorem tank_full_capacity : T = 40 :=
by
  -- Using the given condition to derive that T = 40.
  sorry

end NUMINAMATH_GPT_tank_full_capacity_l1680_168064


namespace NUMINAMATH_GPT_value_of_b_l1680_168005

-- Defining the number sum in circles and overlap
def circle_sum := 21
def num_circles := 5
def total_sum := 69

-- Overlapping numbers
def overlap_1 := 2
def overlap_2 := 8
def overlap_3 := 9
variable (b d : ℕ)

-- Circle equation containing d
def circle_with_d := d + 5 + 9

-- Prove b = 10 given the conditions
theorem value_of_b (h₁ : num_circles * circle_sum = 105)
    (h₂ : 105 - (overlap_1 + overlap_2 + overlap_3 + b + d) = total_sum)
    (h₃ : circle_with_d d = 21) : b = 10 :=
by sorry

end NUMINAMATH_GPT_value_of_b_l1680_168005


namespace NUMINAMATH_GPT_simplify_expression_l1680_168052

variable {x y : ℝ}

theorem simplify_expression : (x^5 * x^3 * y^2 * y^4) = (x^8 * y^6) := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1680_168052


namespace NUMINAMATH_GPT_molly_bike_miles_l1680_168087

def total_miles_ridden (daily_miles years_riding days_per_year : ℕ) : ℕ :=
  daily_miles * years_riding * days_per_year

theorem molly_bike_miles :
  total_miles_ridden 3 3 365 = 3285 :=
by
  -- The definition and theorem are provided; the implementation will be done by the prover.
  sorry

end NUMINAMATH_GPT_molly_bike_miles_l1680_168087


namespace NUMINAMATH_GPT_difference_in_soda_bottles_l1680_168097

def diet_soda_bottles : ℕ := 4
def regular_soda_bottles : ℕ := 83

theorem difference_in_soda_bottles :
  regular_soda_bottles - diet_soda_bottles = 79 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_soda_bottles_l1680_168097


namespace NUMINAMATH_GPT_length_of_bridge_l1680_168089

noncomputable def speed_kmhr_to_ms (v : ℕ) : ℝ := (v : ℝ) * (1000 / 3600)

noncomputable def distance_traveled (v : ℝ) (t : ℕ) : ℝ := v * (t : ℝ)

theorem length_of_bridge 
  (length_train : ℕ) -- 90 meters
  (speed_train_kmhr : ℕ) -- 45 km/hr
  (time_cross_bridge : ℕ) -- 30 seconds
  (conversion_factor : ℝ := 1000 / 3600) 
  : ℝ := 
  let speed_train_ms := speed_kmhr_to_ms speed_train_kmhr
  let total_distance := distance_traveled speed_train_ms time_cross_bridge
  total_distance - (length_train : ℝ)

example : length_of_bridge 90 45 30 = 285 := by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1680_168089


namespace NUMINAMATH_GPT_geometric_sequence_proof_l1680_168035

theorem geometric_sequence_proof (a : ℕ → ℝ) (q : ℝ) (h1 : q > 1) (h2 : a 1 > 0)
    (h3 : a 2 * a 4 + a 4 * a 10 - a 4 * a 6 - (a 5)^2 = 9) :
  a 3 - a 7 = -3 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_proof_l1680_168035


namespace NUMINAMATH_GPT_eval_expression_l1680_168020

theorem eval_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ x) (hz' : z ≠ -x) :
  ((x / (x + z) + z / (x - z)) / (z / (x + z) - x / (x - z)) = -1) :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1680_168020


namespace NUMINAMATH_GPT_find_n_l1680_168069

theorem find_n (n : ℕ) (h : n * n.factorial + n.factorial = 720) : n = 5 :=
sorry

end NUMINAMATH_GPT_find_n_l1680_168069


namespace NUMINAMATH_GPT_range_of_a_l1680_168084

open Set Real

def set_M (a : ℝ) : Set ℝ := { x | x * (x - a - 1) < 0 }
def set_N : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }

theorem range_of_a (a : ℝ) : set_M a ⊆ set_N ↔ -2 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1680_168084
