import Mathlib

namespace NUMINAMATH_GPT_ring_toss_total_l1186_118614

theorem ring_toss_total (money_per_day : ℕ) (days : ℕ) (total_money : ℕ) 
(h1 : money_per_day = 140) (h2 : days = 3) : total_money = 420 :=
by
  sorry

end NUMINAMATH_GPT_ring_toss_total_l1186_118614


namespace NUMINAMATH_GPT_total_handshakes_l1186_118628

def people := 40
def groupA := 25
def groupB := 15
def knownByGroupB (x : ℕ) : ℕ := 5
def interactionsWithinGroupB : ℕ := 105
def interactionsBetweenGroups : ℕ := 75

theorem total_handshakes : (groupB * knownByGroupB 0) + interactionsWithinGroupB = 180 :=
by
  sorry

end NUMINAMATH_GPT_total_handshakes_l1186_118628


namespace NUMINAMATH_GPT_is_incorrect_B_l1186_118694

variable {a b c : ℝ}

theorem is_incorrect_B :
  ¬ ((a > b ∧ b > c) → (1 / (b - c)) < (1 / (a - c))) :=
sorry

end NUMINAMATH_GPT_is_incorrect_B_l1186_118694


namespace NUMINAMATH_GPT_evaluate_expression_l1186_118678

theorem evaluate_expression : 1234562 - (12 * 3 * (2 + 7)) = 1234238 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1186_118678


namespace NUMINAMATH_GPT_sqrt_sum_simplify_l1186_118642

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_sqrt_sum_simplify_l1186_118642


namespace NUMINAMATH_GPT_inequality_am_gm_l1186_118633

theorem inequality_am_gm (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (y * z) + y / (z * x) + z / (x * y)) ≥ (1 / x + 1 / y + 1 / z) := 
by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l1186_118633


namespace NUMINAMATH_GPT_quadratic_inequality_range_l1186_118682

variable (x : ℝ)

-- Statement of the mathematical problem
theorem quadratic_inequality_range (h : ¬ (x^2 - 5 * x + 4 > 0)) : 1 ≤ x ∧ x ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_range_l1186_118682


namespace NUMINAMATH_GPT_ellipse_foci_distance_sum_l1186_118636

theorem ellipse_foci_distance_sum
    (x y : ℝ)
    (PF1 PF2 : ℝ)
    (a : ℝ)
    (h_ellipse : (x^2 / 36) + (y^2 / 16) = 1)
    (h_foci : ∀F1 F2, ∃e > 0, F1 = (e, 0) ∧ F2 = (-e, 0))
    (h_point_on_ellipse : ∀x y, (x^2 / 36) + (y^2 / 16) = 1 → (x, y) = (PF1, PF2))
    (h_semi_major_axis : a = 6):
    |PF1| + |PF2| = 12 := 
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_sum_l1186_118636


namespace NUMINAMATH_GPT_product_of_consecutive_even_numbers_divisible_by_8_l1186_118615

theorem product_of_consecutive_even_numbers_divisible_by_8 (n : ℤ) : 8 ∣ (2 * n * (2 * n + 2)) :=
by sorry

end NUMINAMATH_GPT_product_of_consecutive_even_numbers_divisible_by_8_l1186_118615


namespace NUMINAMATH_GPT_PQR_positive_iff_P_Q_R_positive_l1186_118613

noncomputable def P (a b c : ℝ) : ℝ := a + b - c
noncomputable def Q (a b c : ℝ) : ℝ := b + c - a
noncomputable def R (a b c : ℝ) : ℝ := c + a - b

theorem PQR_positive_iff_P_Q_R_positive (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (P a b c * Q a b c * R a b c > 0) ↔ (P a b c > 0 ∧ Q a b c > 0 ∧ R a b c > 0) :=
sorry

end NUMINAMATH_GPT_PQR_positive_iff_P_Q_R_positive_l1186_118613


namespace NUMINAMATH_GPT_determine_A_l1186_118649

theorem determine_A (A M C : ℕ) (h1 : A < 10) (h2 : M < 10) (h3 : C < 10) 
(h4 : (100 * A + 10 * M + C) * (A + M + C) = 2244) : A = 3 :=
sorry

end NUMINAMATH_GPT_determine_A_l1186_118649


namespace NUMINAMATH_GPT_sector_triangle_radii_l1186_118684

theorem sector_triangle_radii 
  (r : ℝ) (theta : ℝ) (radius : ℝ) 
  (h_theta_eq: theta = 60)
  (h_radius_eq: radius = 10) :
  let R := (radius * Real.sqrt 3) / 3
  let r_in := (radius * Real.sqrt 3) / 6
  R = 10 * (Real.sqrt 3) / 3 ∧ r_in = 10 * (Real.sqrt 3) / 6 := 
by
  sorry

end NUMINAMATH_GPT_sector_triangle_radii_l1186_118684


namespace NUMINAMATH_GPT_algebraic_expression_correct_l1186_118632

theorem algebraic_expression_correct (x y : ℝ) :
  (x - y)^2 - (x^2 - y^2) = (x - y)^2 - (x^2 - y^2) :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_correct_l1186_118632


namespace NUMINAMATH_GPT_intersection_of_sets_l1186_118693

open Set

theorem intersection_of_sets : 
  let M : Set ℕ := {0, 2, 4, 8}
  let N : Set ℕ := { x | ∃ a, a ∈ M ∧ x = 2 * a }
  M ∩ N = {0, 4, 8} := 
by
  let M : Set ℕ := {0, 2, 4, 8}
  let N : Set ℕ := { x | ∃ a, a ∈ M ∧ x = 2 * a }
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1186_118693


namespace NUMINAMATH_GPT_find_number_l1186_118644

theorem find_number 
  (x : ℝ) 
  (h1 : 3 * (2 * x + 9) = 69) : x = 7 := by
  sorry

end NUMINAMATH_GPT_find_number_l1186_118644


namespace NUMINAMATH_GPT_kenneth_distance_past_finish_l1186_118608

noncomputable def distance_past_finish_line (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) : ℕ :=
  let biff_time := race_distance / biff_speed
  let kenneth_distance := kenneth_speed * biff_time
  kenneth_distance - race_distance

theorem kenneth_distance_past_finish (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) (finish_line_distance : ℕ) : 
  race_distance = 500 ->
  biff_speed = 50 -> 
  kenneth_speed = 51 ->
  finish_line_distance = 10 ->
  distance_past_finish_line race_distance biff_speed kenneth_speed = finish_line_distance := by
  sorry

end NUMINAMATH_GPT_kenneth_distance_past_finish_l1186_118608


namespace NUMINAMATH_GPT_decimal_111_to_base_5_l1186_118680

def decimal_to_base_5 (n : ℕ) : ℕ :=
  let rec loop (n : ℕ) (acc : ℕ) (place : ℕ) :=
    if n = 0 then acc
    else 
      let rem := n % 5
      let q := n / 5
      loop q (acc + rem * place) (place * 10)
  loop n 0 1

theorem decimal_111_to_base_5 : decimal_to_base_5 111 = 421 :=
  sorry

end NUMINAMATH_GPT_decimal_111_to_base_5_l1186_118680


namespace NUMINAMATH_GPT_five_digit_numbers_greater_21035_and_even_correct_five_digit_numbers_even_with_odd_positions_correct_l1186_118698

noncomputable def count_five_digit_numbers_greater_21035_and_even : Nat :=
  sorry -- insert combinatorial logic to count the numbers

theorem five_digit_numbers_greater_21035_and_even_correct :
  count_five_digit_numbers_greater_21035_and_even = 39 :=
  sorry

noncomputable def count_five_digit_numbers_even_with_odd_positions : Nat :=
  sorry -- insert combinatorial logic to count the numbers

theorem five_digit_numbers_even_with_odd_positions_correct :
  count_five_digit_numbers_even_with_odd_positions = 8 :=
  sorry

end NUMINAMATH_GPT_five_digit_numbers_greater_21035_and_even_correct_five_digit_numbers_even_with_odd_positions_correct_l1186_118698


namespace NUMINAMATH_GPT_katie_earnings_l1186_118671

theorem katie_earnings :
  4 * 3 + 3 * 7 + 2 * 5 + 5 * 2 = 53 := 
by 
  sorry

end NUMINAMATH_GPT_katie_earnings_l1186_118671


namespace NUMINAMATH_GPT_find_q_l1186_118689

variable {m n q : ℝ}

theorem find_q (h1 : m = 3 * n + 5) (h2 : m + 2 = 3 * (n + q) + 5) : q = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_find_q_l1186_118689


namespace NUMINAMATH_GPT_probability_different_colors_l1186_118640

-- Define the total number of blue and yellow chips
def blue_chips : ℕ := 5
def yellow_chips : ℕ := 7
def total_chips : ℕ := blue_chips + yellow_chips

-- Define the probability of drawing a blue chip and a yellow chip
def prob_blue : ℚ := blue_chips / total_chips
def prob_yellow : ℚ := yellow_chips / total_chips

-- Define the probability of drawing two chips of different colors
def prob_different_colors := 2 * (prob_blue * prob_yellow)

theorem probability_different_colors :
  prob_different_colors = (35 / 72) := by
  sorry

end NUMINAMATH_GPT_probability_different_colors_l1186_118640


namespace NUMINAMATH_GPT_sum_of_variables_l1186_118664

theorem sum_of_variables (a b c d : ℕ) (h1 : ac + bd + ad + bc = 1997) : a + b + c + d = 1998 :=
sorry

end NUMINAMATH_GPT_sum_of_variables_l1186_118664


namespace NUMINAMATH_GPT_min_lamps_l1186_118631

theorem min_lamps (n p : ℕ) (h1: p > 0) (h_total_profit : 3 * (3 * p / 4 / n) + (n - 3) * (p / n + 10) - p = 100) : n = 13 :=
by
  sorry

end NUMINAMATH_GPT_min_lamps_l1186_118631


namespace NUMINAMATH_GPT_percentage_salt_solution_l1186_118622

-- Definitions
def P : ℝ := 60
def ounces_added := 40
def initial_solution_ounces := 40
def initial_solution_percentage := 0.20
def final_solution_percentage := 0.40
def final_solution_ounces := 80

-- Lean Statement
theorem percentage_salt_solution (P : ℝ) :
  (8 + 0.01 * P * ounces_added) = 0.40 * final_solution_ounces → P = 60 := 
by
  sorry

end NUMINAMATH_GPT_percentage_salt_solution_l1186_118622


namespace NUMINAMATH_GPT_ratio_of_age_difference_l1186_118663

-- Define the ages of the scrolls and the ratio R
variables (S1 S2 S3 S4 S5 : ℕ)
variables (R : ℚ)

-- Conditions
axiom h1 : S1 = 4080
axiom h5 : S5 = 20655
axiom h2 : S2 - S1 = R * S5
axiom h3 : S3 - S2 = R * S5
axiom h4 : S4 - S3 = R * S5
axiom h6 : S5 - S4 = R * S5

-- The theorem to prove
theorem ratio_of_age_difference : R = 16575 / 82620 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_age_difference_l1186_118663


namespace NUMINAMATH_GPT_closest_integer_to_cube_root_of_150_l1186_118610

theorem closest_integer_to_cube_root_of_150 : 
  ∃ (n : ℤ), ∀ m : ℤ, abs (150 - 5 ^ 3) < abs (150 - m ^ 3) → n = 5 :=
by
  sorry

end NUMINAMATH_GPT_closest_integer_to_cube_root_of_150_l1186_118610


namespace NUMINAMATH_GPT_cube_removal_minimum_l1186_118653

theorem cube_removal_minimum (l w h : ℕ) (hu : l = 4) (hv : w = 5) (hw : h = 6) :
  ∃ num_cubes_removed : ℕ, 
    (l * w * h - num_cubes_removed = 4 * 4 * 4) ∧ 
    num_cubes_removed = 56 := 
by
  sorry

end NUMINAMATH_GPT_cube_removal_minimum_l1186_118653


namespace NUMINAMATH_GPT_restaurant_total_earnings_l1186_118603

noncomputable def restaurant_earnings (weekdays weekends : ℕ) (weekday_earnings : ℝ) 
    (weekend_min_earnings weekend_max_earnings discount special_event_earnings : ℝ) : ℝ :=
  let num_mondays := weekdays / 5 
  let weekday_earnings_with_discount := weekday_earnings - (weekday_earnings * discount)
  let earnings_mondays := num_mondays * weekday_earnings_with_discount
  let earnings_other_weekdays := (weekdays - num_mondays) * weekday_earnings
  let average_weekend_earnings := (weekend_min_earnings + weekend_max_earnings) / 2
  let total_weekday_earnings := earnings_mondays + earnings_other_weekdays
  let total_weekend_earnings := 2 * weekends * average_weekend_earnings
  total_weekday_earnings + total_weekend_earnings + special_event_earnings

theorem restaurant_total_earnings 
  (weekdays weekends : ℕ)
  (weekday_earnings weekend_min_earnings weekend_max_earnings discount special_event_earnings total_earnings : ℝ)
  (h_weekdays : weekdays = 22)
  (h_weekends : weekends = 8)
  (h_weekday_earnings : weekday_earnings = 600)
  (h_weekend_min_earnings : weekend_min_earnings = 1000)
  (h_weekend_max_earnings : weekend_max_earnings = 1500)
  (h_discount : discount = 0.1)
  (h_special_event_earnings : special_event_earnings = 500)
  (h_total_earnings : total_earnings = 33460) :
  restaurant_earnings weekdays weekends weekday_earnings weekend_min_earnings weekend_max_earnings discount special_event_earnings = total_earnings := 
by
  sorry

end NUMINAMATH_GPT_restaurant_total_earnings_l1186_118603


namespace NUMINAMATH_GPT_opposite_of_neg_five_l1186_118635

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_five_l1186_118635


namespace NUMINAMATH_GPT_f_even_l1186_118617

-- Define E_x^n as specified
def E_x (n : ℕ) (x : ℝ) : ℝ := List.prod (List.map (λ i => x + i) (List.range n))

-- Define the function f(x)
def f (x : ℝ) : ℝ := x * E_x 5 (x - 2)

-- Define the statement to prove f(x) is even
theorem f_even (x : ℝ) : f x = f (-x) := by
  sorry

end NUMINAMATH_GPT_f_even_l1186_118617


namespace NUMINAMATH_GPT_mixture_percentage_l1186_118654

variable (P : ℝ)
variable (x_ryegrass_percent : ℝ := 0.40)
variable (y_ryegrass_percent : ℝ := 0.25)
variable (final_mixture_ryegrass_percent : ℝ := 0.32)

theorem mixture_percentage (h : 0.40 * P + 0.25 * (1 - P) = 0.32) : P = 0.07 / 0.15 := by
  sorry

end NUMINAMATH_GPT_mixture_percentage_l1186_118654


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1186_118690

def isHyperbolaWithEccentricity (e : ℝ) : Prop :=
  ∃ (a b : ℝ), a = 4 * b ∧ e = (Real.sqrt (a^2 + b^2)) / a

theorem hyperbola_eccentricity : isHyperbolaWithEccentricity (Real.sqrt 17 / 4) :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1186_118690


namespace NUMINAMATH_GPT_total_amount_paid_l1186_118606

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_tshirts : ℕ := 6

theorem total_amount_paid : 
  (number_of_tshirts : ℝ) * (original_price * discount_rate) = 60 := by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1186_118606


namespace NUMINAMATH_GPT_michael_truck_meet_once_l1186_118688

/-- Michael walks at 6 feet per second -/
def michael_speed := 6

/-- Trash pails are located every 300 feet along the path -/
def pail_distance := 300

/-- A garbage truck travels at 15 feet per second -/
def truck_speed := 15

/-- The garbage truck stops for 45 seconds at each pail -/
def truck_stop_time := 45

/-- Michael passes a pail just as the truck leaves the next pail -/
def initial_distance := 300

/-- Prove that Michael and the truck meet exactly 1 time -/
theorem michael_truck_meet_once :
  ∀ (meeting_times : ℕ), meeting_times = 1 := by
  sorry

end NUMINAMATH_GPT_michael_truck_meet_once_l1186_118688


namespace NUMINAMATH_GPT_cone_fits_in_cube_l1186_118619

noncomputable def height_cone : ℝ := 15
noncomputable def diameter_cone_base : ℝ := 8
noncomputable def side_length_cube : ℝ := 15
noncomputable def volume_cube : ℝ := side_length_cube ^ 3

theorem cone_fits_in_cube :
  (height_cone = 15) →
  (diameter_cone_base = 8) →
  (height_cone ≤ side_length_cube ∧ diameter_cone_base ≤ side_length_cube) →
  volume_cube = 3375 := by
  intros h_cone d_base fits
  sorry

end NUMINAMATH_GPT_cone_fits_in_cube_l1186_118619


namespace NUMINAMATH_GPT_incorrect_statement_c_l1186_118672

-- Definitions based on conditions
variable (p q : Prop)

-- Lean 4 statement to check the logical proposition
theorem incorrect_statement_c (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_c_l1186_118672


namespace NUMINAMATH_GPT_meaningful_expression_range_l1186_118683

theorem meaningful_expression_range (x : ℝ) : 
  (x - 1 ≥ 0) ∧ (x ≠ 3) ↔ (x ≥ 1 ∧ x ≠ 3) := 
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_range_l1186_118683


namespace NUMINAMATH_GPT_intersection_complement_correct_l1186_118637

def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 5}
def B : Set ℕ := {1, 4}
def C_I (s : Set ℕ) := I \ s  -- set complement

theorem intersection_complement_correct: A ∩ C_I B = {3, 5} := by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_intersection_complement_correct_l1186_118637


namespace NUMINAMATH_GPT_find_x_y_z_sum_l1186_118646

theorem find_x_y_z_sum :
  ∃ (x y z : ℝ), 
    x^2 + 27 = -8 * y + 10 * z ∧
    y^2 + 196 = 18 * z + 13 * x ∧
    z^2 + 119 = -3 * x + 30 * y ∧
    x + 3 * y + 5 * z = 127.5 :=
sorry

end NUMINAMATH_GPT_find_x_y_z_sum_l1186_118646


namespace NUMINAMATH_GPT_trigonometric_identity_l1186_118695

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1186_118695


namespace NUMINAMATH_GPT_discount_percentage_is_25_l1186_118645

-- Define the conditions
def cost_of_coffee : ℕ := 6
def cost_of_cheesecake : ℕ := 10
def final_price_with_discount : ℕ := 12

-- Define the total cost without discount
def total_cost_without_discount : ℕ := cost_of_coffee + cost_of_cheesecake

-- Define the discount amount
def discount_amount : ℕ := total_cost_without_discount - final_price_with_discount

-- Define the percentage discount
def percentage_discount : ℕ := (discount_amount * 100) / total_cost_without_discount

-- Proof Statement
theorem discount_percentage_is_25 : percentage_discount = 25 := by
  sorry

end NUMINAMATH_GPT_discount_percentage_is_25_l1186_118645


namespace NUMINAMATH_GPT_sin_double_angle_identity_l1186_118630

theorem sin_double_angle_identity 
  (α : ℝ) 
  (h₁ : α ∈ Set.Ioo (Real.pi / 2) Real.pi) 
  (h₂ : Real.sin α = 1 / 5) : 
  Real.sin (2 * α) = - (4 * Real.sqrt 6) / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_identity_l1186_118630


namespace NUMINAMATH_GPT_line_canonical_form_l1186_118691

theorem line_canonical_form :
  ∃ (x y z : ℝ),
  x + y + z - 2 = 0 ∧
  x - y - 2 * z + 2 = 0 →
  ∃ (k : ℝ),
  x / k = -1 ∧
  (y - 2) / (3 * k) = 1 ∧
  z / (-2 * k) = 1 :=
sorry

end NUMINAMATH_GPT_line_canonical_form_l1186_118691


namespace NUMINAMATH_GPT_find_D_l1186_118602

theorem find_D (A D : ℝ) (h1 : D + A = 5) (h2 : D - A = -3) : D = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_D_l1186_118602


namespace NUMINAMATH_GPT_t_range_l1186_118685

noncomputable def exists_nonneg_real_numbers_satisfying_conditions (t : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
  (3 * x^2 + 3 * z * x + z^2 = 1) ∧ 
  (3 * y^2 + 3 * y * z + z^2 = 4) ∧ 
  (x^2 - x * y + y^2 = t)

theorem t_range : ∀ t : ℝ, exists_nonneg_real_numbers_satisfying_conditions t → 
  (t ≥ (3 - Real.sqrt 5) / 2 ∧ t ≤ 1) :=
sorry

end NUMINAMATH_GPT_t_range_l1186_118685


namespace NUMINAMATH_GPT_gift_contributors_l1186_118652

theorem gift_contributors :
  (∃ (n : ℕ), n ≥ 1 ∧ n ≤ 20 ∧ ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → (9 : ℕ) ≤ 20) →
  (∃ (n : ℕ), n = 12) :=
by
  sorry

end NUMINAMATH_GPT_gift_contributors_l1186_118652


namespace NUMINAMATH_GPT_part1_part2_l1186_118634

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1186_118634


namespace NUMINAMATH_GPT_delivery_truck_speed_l1186_118666

theorem delivery_truck_speed :
  ∀ d t₁ t₂: ℝ,
    (t₁ = 15 / 60) ∧ (t₂ = -15 / 60) ∧ 
    (t₁ = d / 20 - 1 / 4) ∧ (t₂ = d / 60 + 1 / 4) →
    (d = 15) →
    (t = 1 / 2) →
    ( ∃ v: ℝ, t = d / v ∧ v = 30 ) :=
by sorry

end NUMINAMATH_GPT_delivery_truck_speed_l1186_118666


namespace NUMINAMATH_GPT_solve_equation_l1186_118667

theorem solve_equation (x : ℝ) :
  (2 * x - 1)^2 - 25 = 0 ↔ (x = 3 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1186_118667


namespace NUMINAMATH_GPT_polynomial_divisibility_l1186_118687

theorem polynomial_divisibility (n : ℕ) (h : n > 2) : 
    (∀ k : ℕ, n = 3 * k + 1) ↔ ∃ (k : ℕ), n = 3 * k + 1 := 
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1186_118687


namespace NUMINAMATH_GPT_merchant_marked_price_l1186_118651

-- Given conditions: 30% discount on list price, 10% discount on marked price, 25% profit on selling price
variable (L : ℝ) -- List price
variable (C : ℝ) -- Cost price after discount: C = 0.7 * L
variable (M : ℝ) -- Marked price
variable (S : ℝ) -- Selling price after discount on marked price: S = 0.9 * M

noncomputable def proof_problem : Prop :=
  C = 0.7 * L ∧
  C = 0.75 * S ∧
  S = 0.9 * M ∧
  M = 103.7 / 100 * L

theorem merchant_marked_price (L : ℝ) (C : ℝ) (S : ℝ) (M : ℝ) :
  (C = 0.7 * L) → 
  (C = 0.75 * S) → 
  (S = 0.9 * M) → 
  M = 103.7 / 100 * L :=
by
  sorry

end NUMINAMATH_GPT_merchant_marked_price_l1186_118651


namespace NUMINAMATH_GPT_ratio_of_ages_l1186_118655

variables (R J K : ℕ)

axiom h1 : R = J + 8
axiom h2 : R + 4 = 2 * (J + 4)
axiom h3 : (R + 4) * (K + 4) = 192

theorem ratio_of_ages : (R - J) / (R - K) = 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_ages_l1186_118655


namespace NUMINAMATH_GPT_share_of_A_correct_l1186_118660

theorem share_of_A_correct :
  let investment_A1 := 20000
  let investment_A2 := 15000
  let investment_B1 := 20000
  let investment_B2 := 16000
  let investment_C1 := 20000
  let investment_C2 := 26000
  let total_months1 := 5
  let total_months2 := 7
  let total_profit := 69900

  let total_investment_A := (investment_A1 * total_months1) + (investment_A2 * total_months2)
  let total_investment_B := (investment_B1 * total_months1) + (investment_B2 * total_months2)
  let total_investment_C := (investment_C1 * total_months1) + (investment_C2 * total_months2)
  let total_investment := total_investment_A + total_investment_B + total_investment_C

  let share_A := (total_investment_A : ℝ) / (total_investment : ℝ)
  let profit_A := share_A * (total_profit : ℝ)

  profit_A = 20500.99 :=
by
  sorry

end NUMINAMATH_GPT_share_of_A_correct_l1186_118660


namespace NUMINAMATH_GPT_damian_serena_passing_times_l1186_118686

/-- 
  Damian and Serena are running on a circular track for 40 minutes.
  Damian runs clockwise at 220 m/min on the inner lane with a radius of 45 meters.
  Serena runs counterclockwise at 260 m/min on the outer lane with a radius of 55 meters.
  They start on the same radial line.
  Prove that they pass each other exactly 184 times in 40 minutes. 
-/
theorem damian_serena_passing_times
  (time_run : ℕ)
  (damian_speed : ℕ)
  (serena_speed : ℕ)
  (damian_radius : ℝ)
  (serena_radius : ℝ)
  (start_same_line : Prop) :
  time_run = 40 →
  damian_speed = 220 →
  serena_speed = 260 →
  damian_radius = 45 →
  serena_radius = 55 →
  start_same_line →
  ∃ n : ℕ, n = 184 :=
by
  sorry

end NUMINAMATH_GPT_damian_serena_passing_times_l1186_118686


namespace NUMINAMATH_GPT_ratio_of_x_and_y_l1186_118696

theorem ratio_of_x_and_y {x y a b : ℝ} (h1 : (2 * a - x) / (3 * b - y) = 3) (h2 : a / b = 4.5) : x / y = 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_x_and_y_l1186_118696


namespace NUMINAMATH_GPT_alex_blueberry_pies_l1186_118668

-- Definitions based on given conditions:
def total_pies : ℕ := 30
def ratio (a b c : ℕ) : Prop := (a : ℚ) / b = 2 / 3 ∧ (b : ℚ) / c = 3 / 5

-- Statement to prove the number of blueberry pies
theorem alex_blueberry_pies :
  ∃ (a b c : ℕ), ratio a b c ∧ a + b + c = total_pies ∧ b = 9 :=
by
  sorry

end NUMINAMATH_GPT_alex_blueberry_pies_l1186_118668


namespace NUMINAMATH_GPT_knocks_to_knicks_l1186_118605

variable (knicks knacks knocks : ℝ)

def knicks_eq_knacks : Prop := 
  8 * knicks = 3 * knacks

def knacks_eq_knocks : Prop := 
  4 * knacks = 5 * knocks

theorem knocks_to_knicks
  (h1 : knicks_eq_knacks knicks knacks)
  (h2 : knacks_eq_knocks knacks knocks) :
  20 * knocks = 320 / 15 * knicks :=
  sorry

end NUMINAMATH_GPT_knocks_to_knicks_l1186_118605


namespace NUMINAMATH_GPT_inverse_function_l1186_118662

theorem inverse_function (x : ℝ) (hx : x > 1) : ∃ y : ℝ, x = 2^y + 1 ∧ y = Real.logb 2 (x - 1) :=
sorry

end NUMINAMATH_GPT_inverse_function_l1186_118662


namespace NUMINAMATH_GPT_find_x_y_sum_l1186_118604

variable {x y : ℝ}

theorem find_x_y_sum (h₁ : (x-1)^3 + 1997 * (x-1) = -1) (h₂ : (y-1)^3 + 1997 * (y-1) = 1) : 
  x + y = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_x_y_sum_l1186_118604


namespace NUMINAMATH_GPT_sophie_saves_money_l1186_118679

variable (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ)
variable (given_on_birthday : Bool)

noncomputable def money_saved_per_year (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ) : ℝ :=
  (loads_per_week * dryer_sheets_per_load * weeks_per_year / sheets_per_box) * cost_per_box

theorem sophie_saves_money (h_loads_per_week : loads_per_week = 4) (h_dryer_sheets_per_load : dryer_sheets_per_load = 1)
                           (h_weeks_per_year : weeks_per_year = 52) (h_cost_per_box : cost_per_box = 5.50)
                           (h_sheets_per_box : sheets_per_box = 104) (h_given_on_birthday : given_on_birthday = true) :
  money_saved_per_year 4 1 52 5.50 104 = 11 :=
by
  have h1 : loads_per_week = 4 := h_loads_per_week
  have h2 : dryer_sheets_per_load = 1 := h_dryer_sheets_per_load
  have h3 : weeks_per_year = 52 := h_weeks_per_year
  have h4 : cost_per_box = 5.50 := h_cost_per_box
  have h5 : sheets_per_box = 104 := h_sheets_per_box
  have h6 : given_on_birthday = true := h_given_on_birthday
  sorry

end NUMINAMATH_GPT_sophie_saves_money_l1186_118679


namespace NUMINAMATH_GPT_autumn_pencils_l1186_118616

-- Define the conditions of the problem.
def initial_pencils := 20
def misplaced_pencils := 7
def broken_pencils := 3
def found_pencils := 4
def bought_pencils := 2

-- Define the number of pencils lost and gained.
def pencils_lost := misplaced_pencils + broken_pencils
def pencils_gained := found_pencils + bought_pencils

-- Define the final number of pencils.
def final_pencils := initial_pencils - pencils_lost + pencils_gained

-- The theorem we want to prove.
theorem autumn_pencils : final_pencils = 16 := by
  sorry

end NUMINAMATH_GPT_autumn_pencils_l1186_118616


namespace NUMINAMATH_GPT_condition_neither_sufficient_nor_necessary_l1186_118600

theorem condition_neither_sufficient_nor_necessary (p q : Prop) :
  (¬ (p ∧ q)) → (p ∨ q) → False :=
by sorry

end NUMINAMATH_GPT_condition_neither_sufficient_nor_necessary_l1186_118600


namespace NUMINAMATH_GPT_find_n_l1186_118620

theorem find_n (n : ℕ) (h : (n + 1) * n.factorial = 5040) : n = 6 := 
by sorry

end NUMINAMATH_GPT_find_n_l1186_118620


namespace NUMINAMATH_GPT_product_pqr_l1186_118607

/-- Mathematical problem statement -/
theorem product_pqr (p q r : ℤ) (hp: p ≠ 0) (hq: q ≠ 0) (hr: r ≠ 0)
  (h1 : p + q + r = 36)
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 540 / (p * q * r) = 1) :
  p * q * r = 864 :=
sorry

end NUMINAMATH_GPT_product_pqr_l1186_118607


namespace NUMINAMATH_GPT_cost_price_of_article_l1186_118639

-- Definitions based on the conditions
def sellingPrice : ℝ := 800
def profitPercentage : ℝ := 25

-- Statement to prove the cost price
theorem cost_price_of_article :
  ∃ cp : ℝ, profitPercentage = ((sellingPrice - cp) / cp) * 100 ∧ cp = 640 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l1186_118639


namespace NUMINAMATH_GPT_intersection_sum_l1186_118621

noncomputable def f (x : ℝ) : ℝ := 5 - (x - 1) ^ 2 / 3

theorem intersection_sum :
  ∃ a b : ℝ, f a = f (a - 4) ∧ b = f a ∧ a + b = 16 / 3 :=
sorry

end NUMINAMATH_GPT_intersection_sum_l1186_118621


namespace NUMINAMATH_GPT_evelyn_lost_bottle_caps_l1186_118674

-- Definitions from the conditions
def initial_amount : ℝ := 63.0
def final_amount : ℝ := 45.0
def lost_amount : ℝ := 18.0

-- Statement to be proved
theorem evelyn_lost_bottle_caps : initial_amount - final_amount = lost_amount := 
by 
  sorry

end NUMINAMATH_GPT_evelyn_lost_bottle_caps_l1186_118674


namespace NUMINAMATH_GPT_first_mission_days_l1186_118624

-- Definitions
variable (x : ℝ) (extended_first_mission : ℝ) (second_mission : ℝ) (total_mission_time : ℝ)

axiom h1 : extended_first_mission = 1.60 * x
axiom h2 : second_mission = 3
axiom h3 : total_mission_time = 11
axiom h4 : extended_first_mission + second_mission = total_mission_time

-- Theorem to prove
theorem first_mission_days : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_first_mission_days_l1186_118624


namespace NUMINAMATH_GPT_sin_sum_arcsin_arctan_l1186_118676

theorem sin_sum_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (1 / 2)
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_sum_arcsin_arctan_l1186_118676


namespace NUMINAMATH_GPT_cos_3theta_value_l1186_118626

open Real

noncomputable def cos_3theta (theta : ℝ) : ℝ := 4 * (cos theta)^3 - 3 * (cos theta)

theorem cos_3theta_value (theta : ℝ) (h : cos theta = 1 / 3) : cos_3theta theta = - 23 / 27 :=
by
  sorry

end NUMINAMATH_GPT_cos_3theta_value_l1186_118626


namespace NUMINAMATH_GPT_sum_cubes_coeffs_l1186_118612

theorem sum_cubes_coeffs :
  ∃ a b c d e : ℤ, 
  (1000 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
  (a + b + c + d + e = 92) :=
sorry

end NUMINAMATH_GPT_sum_cubes_coeffs_l1186_118612


namespace NUMINAMATH_GPT_system_solution_ratio_l1186_118657

theorem system_solution_ratio (x y z : ℝ) (h_xyz_nonzero: x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h1 : x + (95/9)*y + 4*z = 0) (h2 : 4*x + (95/9)*y - 3*z = 0) (h3 : 3*x + 5*y - 4*z = 0) :
  (x * z) / (y ^ 2) = 175 / 81 := 
by sorry

end NUMINAMATH_GPT_system_solution_ratio_l1186_118657


namespace NUMINAMATH_GPT_probability_odd_sum_probability_even_product_l1186_118669
open Classical

noncomputable def number_of_possible_outcomes : ℕ := 36
noncomputable def number_of_odd_sum_outcomes : ℕ := 18
noncomputable def number_of_even_product_outcomes : ℕ := 27

theorem probability_odd_sum (n : ℕ) (m_1 : ℕ) (h1 : n = number_of_possible_outcomes)
  (h2 : m_1 = number_of_odd_sum_outcomes) : (m_1 : ℝ) / n = 1 / 2 :=
by
  sorry

theorem probability_even_product (n : ℕ) (m_2 : ℕ) (h1 : n = number_of_possible_outcomes)
  (h2 : m_2 = number_of_even_product_outcomes) : (m_2 : ℝ) / n = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_odd_sum_probability_even_product_l1186_118669


namespace NUMINAMATH_GPT_find_fourth_term_geometric_progression_l1186_118658

theorem find_fourth_term_geometric_progression (x : ℝ) (a1 a2 a3 : ℝ) (r : ℝ)
  (h1 : a1 = x)
  (h2 : a2 = 3 * x + 6)
  (h3 : a3 = 7 * x + 21)
  (h4 : ∃ r, a2 / a1 = r ∧ a3 / a2 = r)
  (hx : x = 3 / 2) :
  7 * (7 * x + 21) = 220.5 :=
by
  sorry

end NUMINAMATH_GPT_find_fourth_term_geometric_progression_l1186_118658


namespace NUMINAMATH_GPT_number_put_in_machine_l1186_118670

theorem number_put_in_machine (x : ℕ) (y : ℕ) (h1 : y = x + 15 - 6) (h2 : y = 77) : x = 68 :=
by
  sorry

end NUMINAMATH_GPT_number_put_in_machine_l1186_118670


namespace NUMINAMATH_GPT_place_value_ratio_l1186_118618

def number : ℝ := 90347.6208
def place_value_0 : ℝ := 10000 -- tens of thousands
def place_value_6 : ℝ := 0.1 -- tenths

theorem place_value_ratio : 
  place_value_0 / place_value_6 = 100000 := by 
    sorry

end NUMINAMATH_GPT_place_value_ratio_l1186_118618


namespace NUMINAMATH_GPT_mutually_exclusive_necessary_not_sufficient_complementary_l1186_118638

variables {Ω : Type} {A1 A2 : Set Ω}

/-- Definition of mutually exclusive events -/
def mutually_exclusive (A1 A2 : Set Ω) : Prop :=
  A1 ∩ A2 = ∅

/-- Definition of complementary events -/
def complementary (A1 A2 : Set Ω) : Prop :=
  A1 ∪ A2 = Set.univ ∧ mutually_exclusive A1 A2

/-- The proposition that mutually exclusive events are necessary but not sufficient for being complementary -/
theorem mutually_exclusive_necessary_not_sufficient_complementary :
  (mutually_exclusive A1 A2 → complementary A1 A2) = false 
  ∧ (complementary A1 A2 → mutually_exclusive A1 A2) = true :=
sorry

end NUMINAMATH_GPT_mutually_exclusive_necessary_not_sufficient_complementary_l1186_118638


namespace NUMINAMATH_GPT_water_breaks_frequency_l1186_118661

theorem water_breaks_frequency :
  ∃ W : ℕ, (240 / 120 + 10) = 240 / W :=
by
  existsi (20 : ℕ)
  sorry

end NUMINAMATH_GPT_water_breaks_frequency_l1186_118661


namespace NUMINAMATH_GPT_option_d_may_not_hold_l1186_118673

theorem option_d_may_not_hold (a b : ℝ) (m : ℝ) (h : a < b) : ¬ (m^2 * a > m^2 * b) :=
sorry

end NUMINAMATH_GPT_option_d_may_not_hold_l1186_118673


namespace NUMINAMATH_GPT_round_table_vip_arrangements_l1186_118629

-- Define the conditions
def number_of_people : ℕ := 10
def vip_seats : ℕ := 2

noncomputable def number_of_arrangements : ℕ :=
  let total_arrangements := Nat.factorial number_of_people
  let vip_choices := Nat.choose number_of_people vip_seats
  let remaining_arrangements := Nat.factorial (number_of_people - vip_seats)
  vip_choices * remaining_arrangements

-- Theorem stating the result
theorem round_table_vip_arrangements : number_of_arrangements = 1814400 := by
  sorry

end NUMINAMATH_GPT_round_table_vip_arrangements_l1186_118629


namespace NUMINAMATH_GPT_min_value_expression_l1186_118627

theorem min_value_expression :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ (∀ θ' : ℝ, 0 < θ' ∧ θ' < π / 2 → 
    (3 * Real.sin θ' + 4 / Real.cos θ' + 2 * Real.sqrt 3 * Real.tan θ') ≥ 9 * Real.sqrt 3) ∧ 
    (3 * Real.sin θ + 4 / Real.cos θ + 2 * Real.sqrt 3 * Real.tan θ = 9 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1186_118627


namespace NUMINAMATH_GPT_tan_sub_eq_minus_2sqrt3_l1186_118611

theorem tan_sub_eq_minus_2sqrt3 
  (h1 : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3)
  (h2 : Real.tan (5 * Real.pi / 12) = 2 + Real.sqrt 3) : 
  Real.tan (Real.pi / 12) - Real.tan (5 * Real.pi / 12) = -2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_sub_eq_minus_2sqrt3_l1186_118611


namespace NUMINAMATH_GPT_incorrect_statement_about_GIS_l1186_118601

def statement_A := "GIS can provide information for geographic decision-making"
def statement_B := "GIS are computer systems specifically designed to process geographic spatial data"
def statement_C := "Urban management is one of the earliest and most effective fields of GIS application"
def statement_D := "GIS's main functions include data collection, data analysis, decision-making applications, etc."

def correct_answer := statement_B

theorem incorrect_statement_about_GIS:
  correct_answer = statement_B := 
sorry

end NUMINAMATH_GPT_incorrect_statement_about_GIS_l1186_118601


namespace NUMINAMATH_GPT_factory_employees_l1186_118647

def num_employees (n12 n14 n17 : ℕ) : ℕ := n12 + n14 + n17

def total_cost (n12 n14 n17 : ℕ) : ℕ := 
    (200 * 12 * 8) + (40 * 14 * 8) + (n17 * 17 * 8)

theorem factory_employees (n17 : ℕ) 
    (h_cost : total_cost 200 40 n17 = 31840) : 
    num_employees 200 40 n17 = 300 := 
by 
    sorry

end NUMINAMATH_GPT_factory_employees_l1186_118647


namespace NUMINAMATH_GPT_unique_ab_not_determined_l1186_118648

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - Real.sqrt 2

theorem unique_ab_not_determined :
  ∀ (a b : ℝ), a > 0 → b > 0 → 
  f a b (f a b (Real.sqrt 2)) = 1 → False := 
by
  sorry

end NUMINAMATH_GPT_unique_ab_not_determined_l1186_118648


namespace NUMINAMATH_GPT_car_cost_difference_l1186_118681

-- Definitions based on the problem's conditions
def car_cost_ratio (C A : ℝ) := C / A = 3 / 2
def ac_cost := 1500

-- Theorem statement that needs proving
theorem car_cost_difference (C A : ℝ) (h1 : car_cost_ratio C A) (h2 : A = ac_cost) : C - A = 750 := 
by sorry

end NUMINAMATH_GPT_car_cost_difference_l1186_118681


namespace NUMINAMATH_GPT_min_value_sin_cos_l1186_118692

open Real

theorem min_value_sin_cos : ∀ x : ℝ, 
  ∃ (y : ℝ), (∀ x, y ≤ sin x ^ 6 + (5 / 3) * cos x ^ 6) ∧ y = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_sin_cos_l1186_118692


namespace NUMINAMATH_GPT_expression_not_equal_one_l1186_118699

-- Definitions of the variables and the conditions
def a : ℝ := sorry  -- Non-zero real number a
def y : ℝ := sorry  -- Real number y

axiom h1 : a ≠ 0
axiom h2 : y ≠ a
axiom h3 : y ≠ -a

-- The main theorem statement
theorem expression_not_equal_one (h1 : a ≠ 0) (h2 : y ≠ a) (h3 : y ≠ -a) : 
  ( (a / (a - y) + y / (a + y)) / (y / (a - y) - a / (a + y)) ) ≠ 1 :=
sorry

end NUMINAMATH_GPT_expression_not_equal_one_l1186_118699


namespace NUMINAMATH_GPT_sprinter_speed_l1186_118625

theorem sprinter_speed
  (distance : ℝ)
  (time : ℝ)
  (H1 : distance = 100)
  (H2 : time = 10) :
    (distance / time = 10) ∧
    ((distance / time) * 60 = 600) ∧
    (((distance / time) * 60 * 60) / 1000 = 36) :=
by
  sorry

end NUMINAMATH_GPT_sprinter_speed_l1186_118625


namespace NUMINAMATH_GPT_price_reduction_l1186_118665

theorem price_reduction (x : ℝ) (h : 560 * (1 - x) * (1 - x) = 315) : 
  560 * (1 - x)^2 = 315 := 
by
  sorry

end NUMINAMATH_GPT_price_reduction_l1186_118665


namespace NUMINAMATH_GPT_postman_pete_mileage_l1186_118643

theorem postman_pete_mileage :
  let initial_steps := 30000
  let resets := 72
  let final_steps := 45000
  let steps_per_mile := 1500
  let steps_per_full_cycle := 99999 + 1
  let total_steps := initial_steps + resets * steps_per_full_cycle + final_steps
  total_steps / steps_per_mile = 4850 := 
by 
  sorry

end NUMINAMATH_GPT_postman_pete_mileage_l1186_118643


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1186_118656

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1186_118656


namespace NUMINAMATH_GPT_intersection_is_isosceles_right_angled_l1186_118677

def is_isosceles_triangle (x : Type) : Prop := sorry -- Definition of isosceles triangle
def is_right_angled_triangle (x : Type) : Prop := sorry -- Definition of right-angled triangle

def M : Set Type := {x | is_isosceles_triangle x}
def N : Set Type := {x | is_right_angled_triangle x}

theorem intersection_is_isosceles_right_angled :
  (M ∩ N) = {x | is_isosceles_triangle x ∧ is_right_angled_triangle x} := by
  sorry

end NUMINAMATH_GPT_intersection_is_isosceles_right_angled_l1186_118677


namespace NUMINAMATH_GPT_min_value_l1186_118659

theorem min_value (m n : ℝ) (h1 : 2 * m + n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, (2 * m + n = 1 → m > 0 → n > 0 → y = (1 / m) + (1 / n) → y ≥ x)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_l1186_118659


namespace NUMINAMATH_GPT_evaluate_ratio_l1186_118609

theorem evaluate_ratio : (2^2003 * 3^2002) / (6^2002) = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_evaluate_ratio_l1186_118609


namespace NUMINAMATH_GPT_find_first_number_l1186_118697

theorem find_first_number (HCF LCM num2 num1 : ℕ) (hcf_cond : HCF = 20) (lcm_cond : LCM = 396) (num2_cond : num2 = 220) 
    (relation_cond : HCF * LCM = num1 * num2) : num1 = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l1186_118697


namespace NUMINAMATH_GPT_number_of_ways_to_sum_to_4_l1186_118641

-- Definitions deriving from conditions
def cards : List ℕ := [0, 1, 2, 3, 4]

-- Goal to prove
theorem number_of_ways_to_sum_to_4 : 
  let pairs := List.product cards cards
  let valid_pairs := pairs.filter (λ (x, y) => x + y = 4)
  List.length valid_pairs = 5 := 
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_sum_to_4_l1186_118641


namespace NUMINAMATH_GPT_negation_example_l1186_118650

theorem negation_example : (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l1186_118650


namespace NUMINAMATH_GPT_tetrahedron_volume_le_one_eight_l1186_118675

theorem tetrahedron_volume_le_one_eight {A B C D : Type} 
  (e₁_AB e₂_AC e₃_AD e₄_BC e₅_BD : ℝ) (h₁ : e₁_AB ≤ 1) (h₂ : e₂_AC ≤ 1) (h₃ : e₃_AD ≤ 1)
  (h₄ : e₄_BC ≤ 1) (h₅ : e₅_BD ≤ 1) : 
  ∃ (vol : ℝ), vol ≤ 1 / 8 :=
sorry

end NUMINAMATH_GPT_tetrahedron_volume_le_one_eight_l1186_118675


namespace NUMINAMATH_GPT_rectangles_in_5x5_grid_l1186_118623

theorem rectangles_in_5x5_grid : 
  ∃ n : ℕ, n = 100 ∧ (∀ (grid : Fin 6 → Fin 6 → Prop), 
  (∃ (vlines hlines : Finset (Fin 6)),
   (vlines.card = 2 ∧ hlines.card = 2) ∧
   n = (vlines.card.choose 2) * (hlines.card.choose 2))) :=
by
  sorry

end NUMINAMATH_GPT_rectangles_in_5x5_grid_l1186_118623
