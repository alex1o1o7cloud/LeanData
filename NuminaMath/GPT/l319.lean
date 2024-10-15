import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_line_l319_31949

theorem arithmetic_sequence_line (A B C x y : ℝ) :
  (2 * B = A + C) → (A * 1 + B * -2 + C = 0) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_line_l319_31949


namespace NUMINAMATH_GPT_area_of_shaded_region_l319_31913

theorem area_of_shaded_region (r R : ℝ) (π : ℝ) (h1 : R = 3 * r) (h2 : 2 * r = 6) : 
  (π * R^2) - (π * r^2) = 72 * π :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l319_31913


namespace NUMINAMATH_GPT_horizon_distance_ratio_l319_31907

def R : ℝ := 6000000
def h1 : ℝ := 1
def h2 : ℝ := 2

noncomputable def distance_to_horizon (R h : ℝ) : ℝ :=
  Real.sqrt (2 * R * h)

noncomputable def d1 : ℝ := distance_to_horizon R h1
noncomputable def d2 : ℝ := distance_to_horizon R h2

theorem horizon_distance_ratio : d2 / d1 = Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_horizon_distance_ratio_l319_31907


namespace NUMINAMATH_GPT_ab_value_l319_31959

theorem ab_value 
  (a b c : ℝ)
  (h1 : a - b = 5)
  (h2 : a^2 + b^2 = 34)
  (h3 : a^3 - b^3 = 30)
  (h4 : a^2 + b^2 - c^2 = 50)
  (h5 : c = 2 * a - b) : 
  a * b = 17 := 
by 
  sorry

end NUMINAMATH_GPT_ab_value_l319_31959


namespace NUMINAMATH_GPT_cos_150_eq_neg_half_l319_31915

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_cos_150_eq_neg_half_l319_31915


namespace NUMINAMATH_GPT_worker_bees_hive_empty_l319_31978

theorem worker_bees_hive_empty:
  ∀ (initial_worker: ℕ) (leave_nectar: ℕ) (reassign_guard: ℕ) (return_trip: ℕ) (multiplier: ℕ),
  initial_worker = 400 →
  leave_nectar = 28 →
  reassign_guard = 30 →
  return_trip = 15 →
  multiplier = 5 →
  ((initial_worker - leave_nectar - reassign_guard + return_trip) * (1 - multiplier)) = 0 :=
by
  intros initial_worker leave_nectar reassign_guard return_trip multiplier
  sorry

end NUMINAMATH_GPT_worker_bees_hive_empty_l319_31978


namespace NUMINAMATH_GPT_find_x_l319_31931

theorem find_x (x : ℝ) : |2 * x - 6| = 3 * x + 1 ↔ x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l319_31931


namespace NUMINAMATH_GPT_race_cars_count_l319_31985

theorem race_cars_count:
  (1 / 7 + 1 / 3 + 1 / 5 = 0.6761904761904762) -> 
  (∀ N : ℕ, (1 / N = 1 / 7 ∨ 1 / N = 1 / 3 ∨ 1 / N = 1 / 5)) -> 
  (1 / 105 = 0.6761904761904762) :=
by
  intro h_sum_probs h_indiv_probs
  sorry

end NUMINAMATH_GPT_race_cars_count_l319_31985


namespace NUMINAMATH_GPT_minimize_quadratic_l319_31974

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_minimize_quadratic_l319_31974


namespace NUMINAMATH_GPT_negation_of_p_l319_31932

variable {x : ℝ}

def p := ∀ x : ℝ, x^3 - x^2 + 1 < 0

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_p_l319_31932


namespace NUMINAMATH_GPT_last_two_digits_of_9_pow_2008_l319_31918

theorem last_two_digits_of_9_pow_2008 : (9 ^ 2008) % 100 = 21 := 
by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_9_pow_2008_l319_31918


namespace NUMINAMATH_GPT_no_matching_formula_l319_31958

def formula_A (x : ℕ) : ℕ := 4 * x - 2
def formula_B (x : ℕ) : ℕ := x^3 - x^2 + 2 * x
def formula_C (x : ℕ) : ℕ := 2 * x^2
def formula_D (x : ℕ) : ℕ := x^2 + 2 * x + 1

theorem no_matching_formula :
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_A x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_B x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_C x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_D x)
  :=
by
  sorry

end NUMINAMATH_GPT_no_matching_formula_l319_31958


namespace NUMINAMATH_GPT_equivalent_operation_l319_31945

theorem equivalent_operation (x : ℚ) : 
  (x * (5 / 6) / (2 / 7)) = x * (35 / 12) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_operation_l319_31945


namespace NUMINAMATH_GPT_fa_plus_fb_gt_zero_l319_31914

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- Define the conditions for a and b
variables (a b : ℝ)
axiom ab_pos : a + b > 0

-- State the theorem
theorem fa_plus_fb_gt_zero : f a + f b > 0 :=
sorry

end NUMINAMATH_GPT_fa_plus_fb_gt_zero_l319_31914


namespace NUMINAMATH_GPT_range_of_a_for_no_extreme_points_l319_31928

theorem range_of_a_for_no_extreme_points :
  ∀ (a : ℝ), (∀ x : ℝ, x * (x - 2 * a) * x + 1 ≠ 0) ↔ -1 ≤ a ∧ a ≤ 1 := sorry

end NUMINAMATH_GPT_range_of_a_for_no_extreme_points_l319_31928


namespace NUMINAMATH_GPT_geometric_sequence_problem_l319_31923

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_problem (a : ℕ → ℝ) (ha : geometric_sequence a) (h : a 4 + a 8 = 1 / 2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 1 / 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l319_31923


namespace NUMINAMATH_GPT_min_value_expression_l319_31906

theorem min_value_expression (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : x * y * z = 2 / 3) :
  x^2 + 6 * x * y + 18 * y^2 + 12 * y * z + 4 * z^2 ≥ 18 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l319_31906


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l319_31997

variable (a : ℝ) (D : Set ℝ)

def p : Prop := a ∈ D
def q : Prop := ∃ x₀ : ℝ, x₀^2 - a * x₀ - a ≤ -3

theorem necessary_not_sufficient_condition (h : p a D → q a) : D = {x : ℝ | x < -4 ∨ x > 0} :=
sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l319_31997


namespace NUMINAMATH_GPT_ratio_proof_l319_31934

-- Definitions and conditions
variables {A B C : ℕ}

-- Given condition: A : B : C = 3 : 2 : 5
def ratio_cond (A B C : ℕ) := 3 * B = 2 * A ∧ 5 * B = 2 * C

-- Theorem statement
theorem ratio_proof (h : ratio_cond A B C) : (2 * A + 3 * B) / (A + 5 * C) = 3 / 7 :=
by sorry

end NUMINAMATH_GPT_ratio_proof_l319_31934


namespace NUMINAMATH_GPT_intersection_M_N_l319_31991

def M : Set ℝ := { y | ∃ x : ℝ, y = x^2 }
def N : Set ℝ := { y | ∃ x : ℝ, y = x }

theorem intersection_M_N : (M ∩ N) = { y : ℝ | 0 ≤ y } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l319_31991


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l319_31910

-- Conditions for Problem 1
def problem1_condition (x : ℝ) : Prop := 
  5 * (x - 20) + 2 * x = 600

-- Proof for Problem 1 Goal
theorem problem1_solution (x : ℝ) (h : problem1_condition x) : x = 100 := 
by sorry

-- Conditions for Problem 2
def problem2_condition (m : ℝ) : Prop :=
  (360 / m) + (540 / (1.2 * m)) = (900 / 100)

-- Proof for Problem 2 Goal
theorem problem2_solution (m : ℝ) (h : problem2_condition m) : m = 90 := 
by sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l319_31910


namespace NUMINAMATH_GPT_alternating_draws_probability_l319_31940

noncomputable def probability_alternating_draws : ℚ :=
  let total_draws := 11
  let white_balls := 5
  let black_balls := 6
  let successful_sequences := 1
  let total_sequences := @Nat.choose total_draws black_balls
  successful_sequences / total_sequences

theorem alternating_draws_probability :
  probability_alternating_draws = 1 / 462 := by
  sorry

end NUMINAMATH_GPT_alternating_draws_probability_l319_31940


namespace NUMINAMATH_GPT_equal_tuesdays_and_fridays_l319_31962

theorem equal_tuesdays_and_fridays (days_in_month : ℕ) (days_of_week : ℕ) (extra_days : ℕ) (starting_days : Finset ℕ) :
  days_in_month = 30 → days_of_week = 7 → extra_days = 2 →
  starting_days = {0, 3, 6} →
  ∃ n : ℕ, n = 3 :=
by
  sorry

end NUMINAMATH_GPT_equal_tuesdays_and_fridays_l319_31962


namespace NUMINAMATH_GPT_initial_bird_families_l319_31979

/- Definitions: -/
def birds_away_africa : ℕ := 23
def birds_away_asia : ℕ := 37
def birds_left_mountain : ℕ := 25

/- Theorem (Question and Correct Answer): -/
theorem initial_bird_families : birds_away_africa + birds_away_asia + birds_left_mountain = 85 := by
  sorry

end NUMINAMATH_GPT_initial_bird_families_l319_31979


namespace NUMINAMATH_GPT_train_speed_proof_l319_31984

theorem train_speed_proof
  (length_of_train : ℕ)
  (length_of_bridge : ℕ)
  (time_to_cross_bridge : ℕ)
  (h_train_length : length_of_train = 145)
  (h_bridge_length : length_of_bridge = 230)
  (h_time : time_to_cross_bridge = 30) :
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 18 / 5 = 45 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_proof_l319_31984


namespace NUMINAMATH_GPT_sheets_in_stack_l319_31992

theorem sheets_in_stack (h : 200 * t = 2.5) (h_pos : t > 0) : (5 / t) = 400 :=
by
  sorry

end NUMINAMATH_GPT_sheets_in_stack_l319_31992


namespace NUMINAMATH_GPT_mixed_doubles_pairing_l319_31988

theorem mixed_doubles_pairing: 
  let males := 5
  let females := 4
  let choose_males := Nat.choose males 2
  let choose_females := Nat.choose females 2
  let arrangements := Nat.factorial 2
  choose_males * choose_females * arrangements = 120 := by
  sorry

end NUMINAMATH_GPT_mixed_doubles_pairing_l319_31988


namespace NUMINAMATH_GPT_jackson_paintable_area_l319_31946

namespace PaintWallCalculation

def length := 14
def width := 11
def height := 9
def windowArea := 70
def bedrooms := 4

def area_one_bedroom : ℕ :=
  2 * (length * height) + 2 * (width * height)

def paintable_area_one_bedroom : ℕ :=
  area_one_bedroom - windowArea

def total_paintable_area : ℕ :=
  bedrooms * paintable_area_one_bedroom

theorem jackson_paintable_area :
  total_paintable_area = 1520 :=
sorry

end PaintWallCalculation

end NUMINAMATH_GPT_jackson_paintable_area_l319_31946


namespace NUMINAMATH_GPT_kombucha_bottles_after_refund_l319_31903

noncomputable def bottles_per_month : ℕ := 15
noncomputable def cost_per_bottle : ℝ := 3.0
noncomputable def refund_per_bottle : ℝ := 0.10
noncomputable def months_in_year : ℕ := 12

theorem kombucha_bottles_after_refund :
  let bottles_per_year := bottles_per_month * months_in_year
  let total_refund := bottles_per_year * refund_per_bottle
  let bottles_bought_with_refund := total_refund / cost_per_bottle
  bottles_bought_with_refund = 6 := sorry

end NUMINAMATH_GPT_kombucha_bottles_after_refund_l319_31903


namespace NUMINAMATH_GPT_alicia_tax_deduction_l319_31967

theorem alicia_tax_deduction (earnings_per_hour_in_cents : ℕ) (tax_rate : ℚ) 
  (h1 : earnings_per_hour_in_cents = 2500) (h2 : tax_rate = 0.02) : 
  earnings_per_hour_in_cents * tax_rate = 50 := 
  sorry

end NUMINAMATH_GPT_alicia_tax_deduction_l319_31967


namespace NUMINAMATH_GPT_max_students_distributing_items_l319_31977

-- Define the given conditions
def pens : Nat := 1001
def pencils : Nat := 910

-- Define the statement
theorem max_students_distributing_items :
  Nat.gcd pens pencils = 91 :=
by
  sorry

end NUMINAMATH_GPT_max_students_distributing_items_l319_31977


namespace NUMINAMATH_GPT_unique_corresponding_point_l319_31938

-- Define the points for the squares
structure Point := (x : ℝ) (y : ℝ)

structure Square :=
  (a b c d : Point)

def contains (sq1 sq2: Square) : Prop :=
  sq2.a.x >= sq1.a.x ∧ sq2.a.y >= sq1.a.y ∧
  sq2.b.x <= sq1.b.x ∧ sq2.b.y >= sq1.b.y ∧
  sq2.c.x <= sq1.c.x ∧ sq2.c.y <= sq1.c.y ∧
  sq2.d.x >= sq1.d.x ∧ sq2.d.y <= sq1.d.y

theorem unique_corresponding_point
  (sq1 sq2 : Square)
  (h1 : contains sq1 sq2)
  (h2 : sq1.a.x - sq1.c.x = sq2.a.x - sq2.c.x ∧ sq1.a.y - sq1.c.y = sq2.a.y - sq2.c.y):
  ∃! (O : Point), ∃ O' : Point, contains sq1 sq2 ∧ 
  (O.x - sq1.a.x) / (sq1.b.x - sq1.a.x) = (O'.x - sq2.a.x) / (sq2.b.x - sq2.a.x) ∧ 
  (O.y - sq1.a.y) / (sq1.d.y - sq1.a.y) = (O'.y - sq2.a.y) / (sq2.d.y - sq2.a.y) := 
sorry

end NUMINAMATH_GPT_unique_corresponding_point_l319_31938


namespace NUMINAMATH_GPT_potential_values_of_k_l319_31917

theorem potential_values_of_k :
  ∃ k : ℚ, ∀ (a b : ℕ), 
  (10 * a + b = k * (a + b)) ∧ (10 * b + a = (13 - k) * (a + b)) → k = 11/2 :=
by
  sorry

end NUMINAMATH_GPT_potential_values_of_k_l319_31917


namespace NUMINAMATH_GPT_winning_percentage_is_62_l319_31960

-- Definitions based on given conditions
def candidate_winner_votes : ℕ := 992
def candidate_win_margin : ℕ := 384
def total_votes : ℕ := candidate_winner_votes + (candidate_winner_votes - candidate_win_margin)

-- The key proof statement
theorem winning_percentage_is_62 :
  ((candidate_winner_votes : ℚ) / total_votes) * 100 = 62 := 
sorry

end NUMINAMATH_GPT_winning_percentage_is_62_l319_31960


namespace NUMINAMATH_GPT_restaurant_june_production_l319_31990

-- Define the given conditions
def daily_hot_dogs := 60
def daily_pizzas := daily_hot_dogs + 40
def june_days := 30
def daily_total := daily_hot_dogs + daily_pizzas
def june_total := daily_total * june_days

-- The goal is to prove that the total number of pizzas and hot dogs made in June is 4800
theorem restaurant_june_production : june_total = 4800 := by
  -- Sorry to skip proof
  sorry

end NUMINAMATH_GPT_restaurant_june_production_l319_31990


namespace NUMINAMATH_GPT_max_regions_11_l319_31900

noncomputable def max_regions (n : ℕ) : ℕ :=
  1 + n * (n + 1) / 2

theorem max_regions_11 : max_regions 11 = 67 := by
  unfold max_regions
  norm_num

end NUMINAMATH_GPT_max_regions_11_l319_31900


namespace NUMINAMATH_GPT_sum_symmetric_prob_43_l319_31925

def prob_symmetric_sum_43_with_20 : Prop :=
  let n_dice := 9
  let min_sum := n_dice * 1
  let max_sum := n_dice * 6
  let midpoint := (min_sum + max_sum) / 2
  let symmetric_sum := 2 * midpoint - 20
  symmetric_sum = 43

theorem sum_symmetric_prob_43 (n_dice : ℕ) (h₁ : n_dice = 9) (h₂ : ∀ i : ℕ, i ≥ 1 ∧ i ≤ 6) :
  prob_symmetric_sum_43_with_20 :=
by
  sorry

end NUMINAMATH_GPT_sum_symmetric_prob_43_l319_31925


namespace NUMINAMATH_GPT_price_reduction_eq_l319_31930

theorem price_reduction_eq (x : ℝ) (price_original price_final : ℝ) 
    (h1 : price_original = 400) 
    (h2 : price_final = 200) 
    (h3 : price_final = price_original * (1 - x) * (1 - x)) :
  400 * (1 - x)^2 = 200 :=
by
  sorry

end NUMINAMATH_GPT_price_reduction_eq_l319_31930


namespace NUMINAMATH_GPT_factorization_correct_l319_31972

theorem factorization_correct : 
  ∀ x : ℝ, (x^2 + 1) * (x^3 - x^2 + x - 1) = (x^2 + 1)^2 * (x - 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_factorization_correct_l319_31972


namespace NUMINAMATH_GPT_two_person_subcommittees_from_six_l319_31924

theorem two_person_subcommittees_from_six :
  (Nat.choose 6 2) = 15 := by
  sorry

end NUMINAMATH_GPT_two_person_subcommittees_from_six_l319_31924


namespace NUMINAMATH_GPT_quadratic_always_real_roots_rhombus_area_when_m_minus_7_l319_31982

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : ℝ := 2 * x^2 + (m - 2) * x - m

-- Statement 1: For any real number m, the quadratic equation always has real roots.
theorem quadratic_always_real_roots (m : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 :=
by {
  -- Proof omitted
  sorry
}

-- Statement 2: When m = -7, the area of the rhombus whose diagonals are the roots of the quadratic equation is 7/4.
theorem rhombus_area_when_m_minus_7 : (∃ x1 x2 : ℝ, quadratic_eq (-7) x1 = 0 ∧ quadratic_eq (-7) x2 = 0 ∧ (1 / 2) * x1 * x2 = 7 / 4) :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_quadratic_always_real_roots_rhombus_area_when_m_minus_7_l319_31982


namespace NUMINAMATH_GPT_point_on_inverse_proportion_l319_31916

theorem point_on_inverse_proportion :
  ∀ (k x y : ℝ), 
    (∀ (x y: ℝ), (x = -2 ∧ y = 6) → y = k / x) →
    k = -12 →
    y = k / x →
    (x = 1 ∧ y = -12) :=
by
  sorry

end NUMINAMATH_GPT_point_on_inverse_proportion_l319_31916


namespace NUMINAMATH_GPT_truck_travel_distance_l319_31957

theorem truck_travel_distance (b t : ℝ) (h1 : t > 0) :
  (300 * (b / 4) / t) / 3 = (25 * b) / t :=
by
  sorry

end NUMINAMATH_GPT_truck_travel_distance_l319_31957


namespace NUMINAMATH_GPT_length_of_crease_l319_31986

/-- 
  Given a rectangular piece of paper 8 inches wide that is folded such that one corner 
  touches the opposite side at an angle θ from the horizontal, and one edge of the paper 
  remains aligned with the base, 
  prove that the length of the crease L is given by L = 8 * tan θ / (1 + tan θ). 
--/
theorem length_of_crease (theta : ℝ) (h : 0 < theta ∧ theta < Real.pi / 2): 
  ∃ L : ℝ, L = 8 * Real.tan theta / (1 + Real.tan theta) :=
sorry

end NUMINAMATH_GPT_length_of_crease_l319_31986


namespace NUMINAMATH_GPT_inequality_transformation_l319_31902

theorem inequality_transformation (x : ℝ) (n : ℕ) (hn : 0 < n) (hx : 0 < x) : 
  x + (n^n) / (x^n) ≥ n + 1 := 
sorry

end NUMINAMATH_GPT_inequality_transformation_l319_31902


namespace NUMINAMATH_GPT_find_K_l319_31943

theorem find_K (Z K : ℕ) (hZ1 : 1000 < Z) (hZ2 : Z < 8000) (hK : Z = K^3) : 11 ≤ K ∧ K ≤ 19 :=
sorry

end NUMINAMATH_GPT_find_K_l319_31943


namespace NUMINAMATH_GPT_find_k_l319_31926

theorem find_k (k : ℝ) (x₁ x₂ : ℝ) (h_distinct_roots : (2*k + 3)^2 - 4*k^2 > 0)
  (h_roots : ∀ (x : ℝ), x^2 + (2*k + 3)*x + k^2 = 0 ↔ x = x₁ ∨ x = x₂)
  (h_reciprocal_sum : 1/x₁ + 1/x₂ = -1) : k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l319_31926


namespace NUMINAMATH_GPT_caps_eaten_correct_l319_31998

def initial_bottle_caps : ℕ := 34
def remaining_bottle_caps : ℕ := 26
def eaten_bottle_caps (k_i k_r : ℕ) : ℕ := k_i - k_r

theorem caps_eaten_correct :
  eaten_bottle_caps initial_bottle_caps remaining_bottle_caps = 8 :=
by
  sorry

end NUMINAMATH_GPT_caps_eaten_correct_l319_31998


namespace NUMINAMATH_GPT_problem_l319_31921

noncomputable def a : Real := 9^(1/3)
noncomputable def b : Real := 3^(2/5)
noncomputable def c : Real := 4^(1/5)

theorem problem (a := 9^(1/3)) (b := 3^(2/5)) (c := 4^(1/5)) : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_problem_l319_31921


namespace NUMINAMATH_GPT_binomial_coefficient_example_l319_31961

theorem binomial_coefficient_example :
  2 * (Nat.choose 7 4) = 70 := 
sorry

end NUMINAMATH_GPT_binomial_coefficient_example_l319_31961


namespace NUMINAMATH_GPT_option_c_correct_l319_31948

theorem option_c_correct (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x < 2 → x^2 - a ≤ 0) : 4 < a :=
by
  sorry

end NUMINAMATH_GPT_option_c_correct_l319_31948


namespace NUMINAMATH_GPT_sequence_is_geometric_l319_31980

def is_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → S n = 3 * a n - 3

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) (a₁ : ℝ) : Prop :=
∀ n : ℕ, a n = a₁ * r ^ n

theorem sequence_is_geometric (S : ℕ → ℝ) (a : ℕ → ℝ) :
  is_sequence_sum S a →
  (∃ a₁ : ℝ, ∃ r : ℝ, geometric_sequence a r a₁ ∧ a₁ = 3 / 2 ∧ r = 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sequence_is_geometric_l319_31980


namespace NUMINAMATH_GPT_original_price_eq_600_l319_31927

theorem original_price_eq_600 (P : ℝ) (h1 : 300 = P * 0.5) : 
  P = 600 :=
sorry

end NUMINAMATH_GPT_original_price_eq_600_l319_31927


namespace NUMINAMATH_GPT_students_in_classroom_l319_31970

/-- There are some students in a classroom. Half of them have 5 notebooks each and the other half have 3 notebooks each. There are 112 notebooks in total in the classroom. Prove the number of students is 28. -/
theorem students_in_classroom (S : ℕ) (h1 : (S / 2) * 5 + (S / 2) * 3 = 112) : S = 28 := 
sorry

end NUMINAMATH_GPT_students_in_classroom_l319_31970


namespace NUMINAMATH_GPT_moles_of_KI_formed_l319_31994

-- Define the given conditions
def moles_KOH : ℕ := 1
def moles_NH4I : ℕ := 1
def balanced_equation (KOH NH4I KI NH3 H2O : ℕ) : Prop :=
  (KOH = 1) ∧ (NH4I = 1) ∧ (KI = 1) ∧ (NH3 = 1) ∧ (H2O = 1)

-- The proof problem statement
theorem moles_of_KI_formed (h : balanced_equation moles_KOH moles_NH4I 1 1 1) : 
  1 = 1 :=
by sorry

end NUMINAMATH_GPT_moles_of_KI_formed_l319_31994


namespace NUMINAMATH_GPT_middle_term_in_expansion_sum_of_odd_coefficients_weighted_sum_of_coefficients_l319_31976

noncomputable def term_in_expansion (n k : ℕ) : ℚ :=
  (Nat.choose n k) * ((-1/2) ^ k)

theorem middle_term_in_expansion :
  term_in_expansion 8 4 = 35 / 8 := by
  sorry

theorem sum_of_odd_coefficients :
  (term_in_expansion 8 1 + term_in_expansion 8 3 + term_in_expansion 8 5 + term_in_expansion 8 7) = -(205 / 16) := by
  sorry

theorem weighted_sum_of_coefficients :
  ((1 * term_in_expansion 8 1) + (2 * term_in_expansion 8 2) + (3 * term_in_expansion 8 3) + (4 * term_in_expansion 8 4) +
  (5 * term_in_expansion 8 5) + (6 * term_in_expansion 8 6) + (7 * term_in_expansion 8 7) + (8 * term_in_expansion 8 8)) =
  -(1 / 32) := by
  sorry

end NUMINAMATH_GPT_middle_term_in_expansion_sum_of_odd_coefficients_weighted_sum_of_coefficients_l319_31976


namespace NUMINAMATH_GPT_find_x_l319_31936

theorem find_x (x : ℕ) (h : (85 + 32 / x : ℝ) * x = 9637) : x = 113 :=
sorry

end NUMINAMATH_GPT_find_x_l319_31936


namespace NUMINAMATH_GPT_square_side_length_l319_31953

theorem square_side_length (length width : ℕ) (h1 : length = 10) (h2 : width = 5) (cut_across_length : length % 2 = 0) :
  ∃ square_side : ℕ, square_side = 5 := by
  sorry

end NUMINAMATH_GPT_square_side_length_l319_31953


namespace NUMINAMATH_GPT_total_eggs_l319_31947

theorem total_eggs (eggs_today eggs_yesterday : ℕ) (h_today : eggs_today = 30) (h_yesterday : eggs_yesterday = 19) : eggs_today + eggs_yesterday = 49 :=
by
  sorry

end NUMINAMATH_GPT_total_eggs_l319_31947


namespace NUMINAMATH_GPT_number_of_ways_to_choose_l319_31995

-- Define the teachers and classes
def teachers : ℕ := 5
def classes : ℕ := 4
def choices (t : ℕ) : ℕ := classes

-- Formalize the problem statement
theorem number_of_ways_to_choose : (choices teachers) ^ teachers = 1024 :=
by
  -- We denote the computation of (4^5)
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_l319_31995


namespace NUMINAMATH_GPT_greatest_n_divides_l319_31987

theorem greatest_n_divides (m : ℕ) (hm : 0 < m) : 
  ∃ n : ℕ, (n = m^4 - m^2 + m) ∧ (m^2 + n) ∣ (n^2 + m) := 
by {
  sorry
}

end NUMINAMATH_GPT_greatest_n_divides_l319_31987


namespace NUMINAMATH_GPT_triangle_obtuse_l319_31955

variable {a b c : ℝ}

theorem triangle_obtuse (h : 2 * c^2 = 2 * a^2 + 2 * b^2 + a * b) :
  ∃ C : ℝ, 0 ≤ C ∧ C ≤ π ∧ Real.cos C = -1/4 ∧ C > Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_obtuse_l319_31955


namespace NUMINAMATH_GPT_solve_for_q_l319_31911

theorem solve_for_q (q : ℝ) (p : ℝ) (h : p = 15 * q^2 - 5) : p = 40 → q = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_q_l319_31911


namespace NUMINAMATH_GPT_seat_arrangement_l319_31939

theorem seat_arrangement (seats : ℕ) (people : ℕ) (min_empty_between : ℕ) : 
  seats = 9 ∧ people = 3 ∧ min_empty_between = 2 → 
  ∃ ways : ℕ, ways = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_seat_arrangement_l319_31939


namespace NUMINAMATH_GPT_tins_per_case_is_24_l319_31944

def total_cases : ℕ := 15
def damaged_percentage : ℝ := 0.05
def remaining_tins : ℕ := 342

theorem tins_per_case_is_24 (x : ℕ) (h : (1 - damaged_percentage) * (total_cases * x) = remaining_tins) : x = 24 :=
  sorry

end NUMINAMATH_GPT_tins_per_case_is_24_l319_31944


namespace NUMINAMATH_GPT_solve_equation_l319_31935

theorem solve_equation (x : ℝ) (h : (3*x^2 + 2*x + 1) / (x - 1) = 3*x + 1) : x = -1/2 :=
sorry

end NUMINAMATH_GPT_solve_equation_l319_31935


namespace NUMINAMATH_GPT_shopkeeper_profit_percentage_l319_31956

theorem shopkeeper_profit_percentage
  (cost_price : ℝ)
  (goods_lost_pct : ℝ)
  (loss_pct : ℝ)
  (remaining_goods : ℝ)
  (selling_price : ℝ)
  (profit_pct : ℝ)
  (h1 : cost_price = 100)
  (h2 : goods_lost_pct = 0.20)
  (h3 : loss_pct = 0.12)
  (h4 : remaining_goods = cost_price * (1 - goods_lost_pct))
  (h5 : selling_price = cost_price * (1 - loss_pct))
  (h6 : profit_pct = ((selling_price - remaining_goods) / remaining_goods) * 100) : 
  profit_pct = 10 := 
sorry

end NUMINAMATH_GPT_shopkeeper_profit_percentage_l319_31956


namespace NUMINAMATH_GPT_f_1001_value_l319_31929

noncomputable def f : ℕ → ℝ := sorry

theorem f_1001_value :
  (∀ a b n : ℕ, a + b = 2^n → f a + f b = n^2) →
  f 1 = 1 →
  f 1001 = 83 :=
by
  intro h₁ h₂
  sorry

end NUMINAMATH_GPT_f_1001_value_l319_31929


namespace NUMINAMATH_GPT_exam_paper_max_marks_l319_31993

/-- A candidate appearing for an examination has to secure 40% marks to pass paper i.
    The candidate secured 40 marks and failed by 20 marks.
    Prove that the maximum mark for paper i is 150. -/
theorem exam_paper_max_marks (p : ℝ) (s f : ℝ) (M : ℝ) (h1 : p = 0.40) (h2 : s = 40) (h3 : f = 20) (h4 : p * M = s + f) :
  M = 150 :=
sorry

end NUMINAMATH_GPT_exam_paper_max_marks_l319_31993


namespace NUMINAMATH_GPT_miles_driven_l319_31908

theorem miles_driven (rental_fee charge_per_mile total_amount_paid : ℝ) (h₁ : rental_fee = 20.99) (h₂ : charge_per_mile = 0.25) (h₃ : total_amount_paid = 95.74) :
  (total_amount_paid - rental_fee) / charge_per_mile = 299 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_miles_driven_l319_31908


namespace NUMINAMATH_GPT_paint_cans_used_l319_31904

theorem paint_cans_used (initial_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) 
    (h1 : initial_rooms = 50) (h2 : lost_cans = 5) (h3 : remaining_rooms = 40) : 
    (remaining_rooms / (initial_rooms - remaining_rooms) / lost_cans) = 20 :=
by
  sorry

end NUMINAMATH_GPT_paint_cans_used_l319_31904


namespace NUMINAMATH_GPT_find_misread_solution_l319_31952

theorem find_misread_solution:
  ∃ a b : ℝ, 
  a = 5 ∧ b = 2 ∧ 
    (a^2 - 2 * a * b + b^2 = 9) ∧ 
    (∀ x y : ℝ, (5 * x + 4 * y = 23) ∧ (3 * x - 2 * y = 5) → (x = 3) ∧ (y = 2)) := by
    sorry

end NUMINAMATH_GPT_find_misread_solution_l319_31952


namespace NUMINAMATH_GPT_eval_expression_l319_31954

theorem eval_expression (x y z : ℝ) 
  (h1 : z = y - 11) 
  (h2 : y = x + 3) 
  (h3 : x = 5)
  (h4 : x + 2 ≠ 0) 
  (h5 : y - 3 ≠ 0) 
  (h6 : z + 7 ≠ 0) : 
  ( (x + 3) / (x + 2) * (y - 1) / (y - 3) * (z + 9) / (z + 7) ) = 2.4 := 
by
  sorry

end NUMINAMATH_GPT_eval_expression_l319_31954


namespace NUMINAMATH_GPT_base3_addition_l319_31975

theorem base3_addition :
  (2 + 1 * 3 + 2 * 9 + 1 * 27 + 2 * 81) + (1 + 1 * 3 + 2 * 9 + 2 * 27) + (2 * 9 + 1 * 27 + 0 * 81 + 2 * 243) + (1 + 1 * 3 + 1 * 9 + 2 * 27 + 2 * 81) = 
  2 + 1 * 3 + 1 * 9 + 2 * 27 + 2 * 81 + 1 * 243 + 1 * 729 := sorry

end NUMINAMATH_GPT_base3_addition_l319_31975


namespace NUMINAMATH_GPT_remaining_blocks_correct_l319_31912

-- Define the initial number of blocks
def initial_blocks : ℕ := 59

-- Define the number of blocks used
def used_blocks : ℕ := 36

-- Define the remaining blocks equation
def remaining_blocks : ℕ := initial_blocks - used_blocks

-- Prove that the number of remaining blocks is 23
theorem remaining_blocks_correct : remaining_blocks = 23 := by
  sorry

end NUMINAMATH_GPT_remaining_blocks_correct_l319_31912


namespace NUMINAMATH_GPT_sum_of_roots_l319_31905

theorem sum_of_roots: (∃ a b : ℝ, (a - 3)^2 = 16 ∧ (b - 3)^2 = 16 ∧ a ≠ b ∧ a + b = 6) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l319_31905


namespace NUMINAMATH_GPT_expected_yield_correct_l319_31969

-- Conditions
def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def step_length_ft : ℝ := 2.5
def yield_per_sqft_pounds : ℝ := 0.75

-- Related quantities
def garden_length_ft : ℝ := garden_length_steps * step_length_ft
def garden_width_ft : ℝ := garden_width_steps * step_length_ft
def garden_area_sqft : ℝ := garden_length_ft * garden_width_ft
def expected_yield_pounds : ℝ := garden_area_sqft * yield_per_sqft_pounds

-- Statement to prove
theorem expected_yield_correct : expected_yield_pounds = 2109.375 := by
  sorry

end NUMINAMATH_GPT_expected_yield_correct_l319_31969


namespace NUMINAMATH_GPT_Albert_more_rocks_than_Joshua_l319_31966

-- Definitions based on the conditions
def Joshua_rocks : ℕ := 80
def Jose_rocks : ℕ := Joshua_rocks - 14
def Albert_rocks : ℕ := Jose_rocks + 20

-- Statement to prove
theorem Albert_more_rocks_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end NUMINAMATH_GPT_Albert_more_rocks_than_Joshua_l319_31966


namespace NUMINAMATH_GPT_parabola_position_l319_31973

-- Define the two parabolas as functions
def parabola1 (x : ℝ) : ℝ := x^2 - 2 * x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + 2 * x + 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (1, parabola1 1) -- (1, 2)
def vertex2 : ℝ × ℝ := (-1, parabola2 (-1)) -- (-1, 0)

-- Define the proof problem where we show relative positions
theorem parabola_position :
  (vertex1.1 > vertex2.1) ∧ (vertex1.2 > vertex2.2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_position_l319_31973


namespace NUMINAMATH_GPT_correct_operation_l319_31996

theorem correct_operation (a : ℝ) : a^5 / a^2 = a^3 := by
  -- Proof steps will be supplied here
  sorry

end NUMINAMATH_GPT_correct_operation_l319_31996


namespace NUMINAMATH_GPT_find_a_range_l319_31937

variable (a k : ℝ)
variable (x : ℝ) (hx : x > 0)

def p := ∀ x > 0, x + a / x ≥ 2
def q := ∀ k : ℝ, ∃ x y : ℝ, k * x - y + 2 = 0 ∧ x^2 + y^2 / a^2 = 1

theorem find_a_range :
  (a > 0) →
  ((p a) ∨ (q a)) ∧ ¬ ((p a) ∧ (q a)) ↔ 1 ≤ a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_find_a_range_l319_31937


namespace NUMINAMATH_GPT_final_hair_length_is_14_l319_31933

def initial_hair_length : ℕ := 24

def half_hair_cut (l : ℕ) : ℕ := l / 2

def hair_growth (l : ℕ) : ℕ := l + 4

def final_hair_cut (l : ℕ) : ℕ := l - 2

theorem final_hair_length_is_14 :
  final_hair_cut (hair_growth (half_hair_cut initial_hair_length)) = 14 := by
  sorry

end NUMINAMATH_GPT_final_hair_length_is_14_l319_31933


namespace NUMINAMATH_GPT_number_of_terms_in_sequence_l319_31901

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem number_of_terms_in_sequence : 
  ∃ n : ℕ, arithmetic_sequence (-3) 4 n = 53 ∧ n = 15 :=
by
  use 15
  constructor
  · unfold arithmetic_sequence
    norm_num
  · norm_num

end NUMINAMATH_GPT_number_of_terms_in_sequence_l319_31901


namespace NUMINAMATH_GPT_seq_v13_eq_b_l319_31999

noncomputable def seq (v : ℕ → ℝ) (b : ℝ) : Prop :=
v 1 = b ∧ ∀ n ≥ 1, v (n + 1) = -1 / (v n + 2)

theorem seq_v13_eq_b (b : ℝ) (hb : 0 < b) (v : ℕ → ℝ) (hs : seq v b) : v 13 = b := by
  sorry

end NUMINAMATH_GPT_seq_v13_eq_b_l319_31999


namespace NUMINAMATH_GPT_intersection_point_l319_31968

variable (x y z t : ℝ)

-- Conditions
def line_parametric : Prop := 
  (x = 1 + 2 * t) ∧ 
  (y = 2) ∧ 
  (z = 4 + t)

def plane_equation : Prop :=
  x - 2 * y + 4 * z - 19 = 0

-- Problem statement
theorem intersection_point (h_line: line_parametric x y z t) (h_plane: plane_equation x y z):
  x = 3 ∧ y = 2 ∧ z = 5 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_l319_31968


namespace NUMINAMATH_GPT_curry_draymond_ratio_l319_31963

theorem curry_draymond_ratio :
  ∃ (curry draymond kelly durant klay : ℕ),
    draymond = 12 ∧
    kelly = 9 ∧
    durant = 2 * kelly ∧
    klay = draymond / 2 ∧
    curry + draymond + kelly + durant + klay = 69 ∧
    curry = 24 ∧ -- Curry's points calculated in the solution
    draymond = 12 → -- Draymond's points reaffirmed
    curry / draymond = 2 :=
by
  sorry

end NUMINAMATH_GPT_curry_draymond_ratio_l319_31963


namespace NUMINAMATH_GPT_num_values_sum_l319_31950

noncomputable def g : ℝ → ℝ :=
sorry

theorem num_values_sum (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - 2 * x * y - 2 * x + 2) :
  ∃ n s : ℕ, (n = 1 ∧ s = 3 ∧ n * s = 3) :=
sorry

end NUMINAMATH_GPT_num_values_sum_l319_31950


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l319_31941

noncomputable def f (x a : ℝ) : ℝ := abs (x - a)

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≤ -2) ↔ (∀ x y : ℝ, (-1 ≤ x) → (x ≤ y) → (f x a ≤ f y a)) ∧ ¬ (∀ x y : ℝ, (-1 ≤ x) → (x ≤ y) → (f x a ≤ f y a) → (a ≤ -2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l319_31941


namespace NUMINAMATH_GPT_melanie_missed_games_l319_31920

-- Define the total number of soccer games played and the number attended by Melanie
def total_games : ℕ := 64
def attended_games : ℕ := 32

-- Statement to be proven
theorem melanie_missed_games : total_games - attended_games = 32 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_melanie_missed_games_l319_31920


namespace NUMINAMATH_GPT_sequence_general_formula_l319_31971

/--
A sequence a_n is defined such that the first term a_1 = 3 and the recursive formula 
a_{n+1} = (3 * a_n - 4) / (a_n - 2).

We aim to prove that the general term of the sequence is given by:
a_n = ( (-2)^(n+2) - 1 ) / ( (-2)^n - 1 )
-/
theorem sequence_general_formula (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 3)
  (hr : ∀ n, a (n + 1) = (3 * a n - 4) / (a n - 2)) :
  a n = ( (-2:ℝ)^(n+2) - 1 ) / ( (-2:ℝ)^n - 1) :=
sorry

end NUMINAMATH_GPT_sequence_general_formula_l319_31971


namespace NUMINAMATH_GPT_pyramid_volume_l319_31922

noncomputable def volume_of_pyramid (l : ℝ) : ℝ :=
  (l^3 / 24) * (Real.sqrt (Real.sqrt 2 + 1))

theorem pyramid_volume (l : ℝ) (α β : ℝ)
  (hα : α = π / 8)
  (hβ : β = π / 4)
  (hl : l = 6) :
  volume_of_pyramid l = 9 * Real.sqrt (Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_GPT_pyramid_volume_l319_31922


namespace NUMINAMATH_GPT_cds_unique_to_either_l319_31919

-- Declare the variables for the given problem
variables (total_alice_shared : ℕ) (total_alice : ℕ) (unique_bob : ℕ)

-- The given conditions in the problem
def condition_alice : Prop := total_alice_shared + unique_bob + (total_alice - total_alice_shared) = total_alice

-- The theorem to prove: number of CDs in either Alice's or Bob's collection but not both is 19
theorem cds_unique_to_either (h1 : total_alice = 23) 
                             (h2 : total_alice_shared = 12) 
                             (h3 : unique_bob = 8) : 
                             (total_alice - total_alice_shared) + unique_bob = 19 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end NUMINAMATH_GPT_cds_unique_to_either_l319_31919


namespace NUMINAMATH_GPT_basketball_games_won_difference_l319_31989

theorem basketball_games_won_difference :
  ∀ (total_games games_won games_lost difference_won_lost : ℕ),
  total_games = 62 →
  games_won = 45 →
  games_lost = 17 →
  difference_won_lost = games_won - games_lost →
  difference_won_lost = 28 :=
by
  intros total_games games_won games_lost difference_won_lost
  intros h_total h_won h_lost h_diff
  rw [h_won, h_lost] at h_diff
  exact h_diff

end NUMINAMATH_GPT_basketball_games_won_difference_l319_31989


namespace NUMINAMATH_GPT_ben_paints_150_square_feet_l319_31964

-- Define the given conditions
def ratio_allen_ben : ℕ := 3
def ratio_ben_allen : ℕ := 5
def total_work : ℕ := 240

-- Define the total amount of parts
def total_parts : ℕ := ratio_allen_ben + ratio_ben_allen

-- Define the work per part
def work_per_part : ℕ := total_work / total_parts

-- Define the work done by Ben
def ben_parts : ℕ := ratio_ben_allen
def ben_work : ℕ := work_per_part * ben_parts

-- The statement to be proved
theorem ben_paints_150_square_feet : ben_work = 150 :=
by
  sorry

end NUMINAMATH_GPT_ben_paints_150_square_feet_l319_31964


namespace NUMINAMATH_GPT_range_of_x_l319_31981

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (x / (1 + 2 * x))

theorem range_of_x (x : ℝ) :
  f (x * (3 * x - 2)) < -1 / 3 ↔ (-(1 / 3) < x ∧ x < 0) ∨ ((2 / 3) < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l319_31981


namespace NUMINAMATH_GPT_smallest_number_is_correct_largest_number_is_correct_l319_31942

def initial_sequence := "123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960"

def remove_digits (n : ℕ) (s : String) : String := sorry  -- Placeholder function for removing n digits

noncomputable def smallest_number_after_removal (s : String) : String :=
  -- Function to find the smallest number possible after removing digits
  remove_digits 100 s

noncomputable def largest_number_after_removal (s : String) : String :=
  -- Function to find the largest number possible after removing digits
  remove_digits 100 s

theorem smallest_number_is_correct : smallest_number_after_removal initial_sequence = "123450" :=
  sorry

theorem largest_number_is_correct : largest_number_after_removal initial_sequence = "56758596049" :=
  sorry

end NUMINAMATH_GPT_smallest_number_is_correct_largest_number_is_correct_l319_31942


namespace NUMINAMATH_GPT_combined_salaries_BCDE_l319_31909

-- Define the given conditions
def salary_A : ℕ := 10000
def average_salary : ℕ := 8400
def num_individuals : ℕ := 5

-- Define the total salary of all individuals
def total_salary_all : ℕ := average_salary * num_individuals

-- Define the proof problem
theorem combined_salaries_BCDE : (total_salary_all - salary_A) = 32000 := by
  sorry

end NUMINAMATH_GPT_combined_salaries_BCDE_l319_31909


namespace NUMINAMATH_GPT_Angie_age_ratio_l319_31965

-- Define Angie's age as a variable
variables (A : ℕ)

-- Give the condition
def Angie_age_condition := A + 4 = 20

-- State the theorem to be proved
theorem Angie_age_ratio (h : Angie_age_condition A) : (A : ℚ) / (A + 4) = 4 / 5 := 
sorry

end NUMINAMATH_GPT_Angie_age_ratio_l319_31965


namespace NUMINAMATH_GPT_min_value_of_f_l319_31951

noncomputable def f (x y z : ℝ) : ℝ :=
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)

theorem min_value_of_f (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : c * y + b * z = a) (h2 : a * z + c * x = b) (h3 : b * x + a * y = c) :
  ∃ x y z : ℝ, f x y z = 1 / 2 := sorry

end NUMINAMATH_GPT_min_value_of_f_l319_31951


namespace NUMINAMATH_GPT_solution_set_quadratic_inequality_l319_31983

def quadratic_inequality_solution (x : ℝ) : Prop := x^2 + x - 2 > 0

theorem solution_set_quadratic_inequality :
  {x : ℝ | quadratic_inequality_solution x} = {x : ℝ | x < -2 ∨ x > 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_quadratic_inequality_l319_31983
