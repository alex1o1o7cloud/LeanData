import Mathlib

namespace NUMINAMATH_GPT_space_is_volume_stuff_is_capacity_film_is_surface_area_l2108_210827

-- Let's define the properties based on the conditions
def size_of_space (box : Type) : Type := 
  sorry -- This will be volume later

def stuff_can_hold (box : Type) : Type :=
  sorry -- This will be capacity later

def film_needed_to_cover (box : Type) : Type :=
  sorry -- This will be surface area later

-- Now prove the correspondences
theorem space_is_volume (box : Type) :
  size_of_space box = volume := 
by 
  sorry

theorem stuff_is_capacity (box : Type) :
  stuff_can_hold box = capacity := 
by 
  sorry

theorem film_is_surface_area (box : Type) :
  film_needed_to_cover box = surface_area := 
by 
  sorry

end NUMINAMATH_GPT_space_is_volume_stuff_is_capacity_film_is_surface_area_l2108_210827


namespace NUMINAMATH_GPT_ratio_sum_l2108_210808

variable (x y z : ℝ)

-- Conditions
axiom geometric_sequence : 16 * y^2 = 15 * x * z
axiom arithmetic_sequence : 2 / y = 1 / x + 1 / z

-- Theorem to prove
theorem ratio_sum : x ≠ 0 → y ≠ 0 → z ≠ 0 → 
  (16 * y^2 = 15 * x * z) → (2 / y = 1 / x + 1 / z) → (x / z + z / x = 34 / 15) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_ratio_sum_l2108_210808


namespace NUMINAMATH_GPT_exist_interval_l2108_210803

noncomputable def f (x : ℝ) := Real.log x + x - 4

theorem exist_interval (x₀ : ℝ) (h₀ : f x₀ = 0) : 2 < x₀ ∧ x₀ < 3 :=
by
  sorry

end NUMINAMATH_GPT_exist_interval_l2108_210803


namespace NUMINAMATH_GPT_evaluate_expression_l2108_210825

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2108_210825


namespace NUMINAMATH_GPT_consecutive_integers_exist_l2108_210802

def good (n : ℕ) : Prop :=
∃ (k : ℕ) (a : ℕ → ℕ), 
  (∀ i j, 1 ≤ i → i < j → j ≤ k → a i < a j) ∧ 
  (∀ i j i' j', 1 ≤ i → i < j → j ≤ k → 1 ≤ i' → i' < j' → j' ≤ k → a i + a j = a i' + a j' → i = i' ∧ j = j') ∧ 
  (∃ (t : ℕ), ∀ m, 0 ≤ m → m < n → ∃ i j, 1 ≤ i → i < j → j ≤ k → a i + a j = t + m)

theorem consecutive_integers_exist (n : ℕ) (h : n = 1000) : good n :=
sorry

end NUMINAMATH_GPT_consecutive_integers_exist_l2108_210802


namespace NUMINAMATH_GPT_connie_total_markers_l2108_210816

theorem connie_total_markers (red_markers : ℕ) (blue_markers : ℕ) 
                              (h1 : red_markers = 41)
                              (h2 : blue_markers = 64) : 
                              red_markers + blue_markers = 105 := by
  sorry

end NUMINAMATH_GPT_connie_total_markers_l2108_210816


namespace NUMINAMATH_GPT_find_kg_of_mangoes_l2108_210884

-- Define the conditions
def cost_of_grapes : ℕ := 8 * 70
def total_amount_paid : ℕ := 965
def cost_of_mangoes (m : ℕ) : ℕ := 45 * m

-- Formalize the proof problem
theorem find_kg_of_mangoes (m : ℕ) :
  cost_of_grapes + cost_of_mangoes m = total_amount_paid → m = 9 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_kg_of_mangoes_l2108_210884


namespace NUMINAMATH_GPT_part_I_part_II_l2108_210807

noncomputable def general_term (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (a 2 = 1 ∧ ∀ n, a (n + 1) - a n = d) ∧
  (d ≠ 0 ∧ (a 3)^2 = (a 2) * (a 6))

theorem part_I (a : ℕ → ℤ) (d : ℤ) : general_term a d → 
  ∀ n, a n = 2 * n - 3 := 
sorry

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) : Prop :=
  (∀ n, S n = n * (a 1 + a n) / 2) ∧ 
  (general_term a d)

theorem part_II (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) : sum_of_first_n_terms a d S → 
  ∃ n, n > 7 ∧ S n > 35 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l2108_210807


namespace NUMINAMATH_GPT_sum_of_digits_inequality_l2108_210824

-- Assume that S(x) represents the sum of the digits of x in its decimal representation.
axiom sum_of_digits (x : ℕ) : ℕ

-- Given condition: for any natural numbers a and b, the sum of digits function satisfies the inequality
axiom sum_of_digits_add (a b : ℕ) : sum_of_digits (a + b) ≤ sum_of_digits a + sum_of_digits b

-- Theorem statement we want to prove
theorem sum_of_digits_inequality (k : ℕ) : sum_of_digits k ≤ 8 * sum_of_digits (8 * k) := 
  sorry

end NUMINAMATH_GPT_sum_of_digits_inequality_l2108_210824


namespace NUMINAMATH_GPT_breadth_increase_25_percent_l2108_210857

variable (L B : ℝ) 

-- Conditions
def original_area := L * B
def increased_length := 1.10 * L
def increased_area := 1.375 * (original_area L B)

-- The breadth increase percentage (to be proven as 25)
def percentage_increase_breadth (p : ℝ) := 
  increased_area L B = increased_length L * (B * (1 + p/100))

-- The statement to be proven
theorem breadth_increase_25_percent : 
  percentage_increase_breadth L B 25 := 
sorry

end NUMINAMATH_GPT_breadth_increase_25_percent_l2108_210857


namespace NUMINAMATH_GPT_problem_solution_l2108_210805

theorem problem_solution (x1 x2 x3 : ℝ) (h1: x1 < x2) (h2: x2 < x3)
(h3 : 10 * x1^3 - 201 * x1^2 + 3 = 0)
(h4 : 10 * x2^3 - 201 * x2^2 + 3 = 0)
(h5 : 10 * x3^3 - 201 * x3^2 + 3 = 0) :
x2 * (x1 + x3) = 398 :=
sorry

end NUMINAMATH_GPT_problem_solution_l2108_210805


namespace NUMINAMATH_GPT_binom_n_n_sub_2_l2108_210869

theorem binom_n_n_sub_2 (n : ℕ) (h : n > 0) : (Nat.choose n (n - 2)) = (n * (n - 1)) / 2 := by
  sorry

end NUMINAMATH_GPT_binom_n_n_sub_2_l2108_210869


namespace NUMINAMATH_GPT_jack_walked_time_l2108_210830

def jack_distance : ℝ := 9
def jack_rate : ℝ := 7.2
def jack_time : ℝ := 1.25

theorem jack_walked_time : jack_time = jack_distance / jack_rate := by
  sorry

end NUMINAMATH_GPT_jack_walked_time_l2108_210830


namespace NUMINAMATH_GPT_bob_initial_cats_l2108_210837

theorem bob_initial_cats (B : ℕ) (h : 21 - 4 = B + 14) : B = 3 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_bob_initial_cats_l2108_210837


namespace NUMINAMATH_GPT_average_stoppage_time_per_hour_l2108_210841

theorem average_stoppage_time_per_hour :
    ∀ (v1_excl v1_incl v2_excl v2_incl v3_excl v3_incl : ℝ),
    v1_excl = 54 → v1_incl = 36 →
    v2_excl = 72 → v2_incl = 48 →
    v3_excl = 90 → v3_incl = 60 →
    ( ((54 / v1_excl - 54 / v1_incl) + (72 / v2_excl - 72 / v2_incl) + (90 / v3_excl - 90 / v3_incl)) / 3 = 0.5 ) := 
by
    intros v1_excl v1_incl v2_excl v2_incl v3_excl v3_incl
    sorry

end NUMINAMATH_GPT_average_stoppage_time_per_hour_l2108_210841


namespace NUMINAMATH_GPT_minimize_fees_at_5_l2108_210879

noncomputable def minimize_costs (x : ℝ) (y1 y2 : ℝ) : Prop :=
  let k1 := 40
  let k2 := 8 / 5
  y1 = k1 / x ∧ y2 = k2 * x ∧ (∀ x, y1 + y2 ≥ 16 ∧ (y1 + y2 = 16 ↔ x = 5))

theorem minimize_fees_at_5 :
  minimize_costs 5 4 16 :=
sorry

end NUMINAMATH_GPT_minimize_fees_at_5_l2108_210879


namespace NUMINAMATH_GPT_Mickey_horses_per_week_l2108_210847

-- Definitions based on the conditions
def days_in_week : Nat := 7
def Minnie_mounts_per_day : Nat := days_in_week + 3 
def Mickey_mounts_per_day : Nat := 2 * Minnie_mounts_per_day - 6
def Mickey_mounts_per_week : Nat := Mickey_mounts_per_day * days_in_week

-- Theorem statement
theorem Mickey_horses_per_week : Mickey_mounts_per_week = 98 :=
by
  sorry

end NUMINAMATH_GPT_Mickey_horses_per_week_l2108_210847


namespace NUMINAMATH_GPT_continuity_necessity_not_sufficiency_l2108_210833

theorem continuity_necessity_not_sufficiency (f : ℝ → ℝ) (x₀ : ℝ) :
  ((∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε) → f x₀ = f x₀) ∧ ¬ ((f x₀ = f x₀) → (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε)) := 
sorry

end NUMINAMATH_GPT_continuity_necessity_not_sufficiency_l2108_210833


namespace NUMINAMATH_GPT_raft_travel_time_l2108_210881

-- Define the problem conditions:
def steamboat_time (distance : ℕ) := 1 -- in hours
def motorboat_time (distance : ℕ) : ℚ := 3 / 4 -- in hours
def speed_ratio := 2 -- motorboat speed is twice the speed of steamboat

-- Define the time for the raft to travel the distance:
def raft_time (distance : ℕ) (current_speed : ℚ) := distance / current_speed

-- Given the conditions, prove that the raft time equals to 90 minutes
theorem raft_travel_time (distance : ℕ) (rafter_speed : ℚ) (current_speed : ℚ) :
  steamboat_time distance = 1 ∧ motorboat_time distance = 3 / 4 ∧ rafter_speed = current_speed →
  rafter_speed = current_speed ∧ raft_time distance current_speed = 3 / 2 → -- hours
  raft_time distance current_speed * 60 = 90 := -- convert hours to minutes
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_raft_travel_time_l2108_210881


namespace NUMINAMATH_GPT_area_of_cross_l2108_210845

-- Definitions based on the conditions
def congruent_squares (n : ℕ) := n = 5
def perimeter_of_cross (p : ℕ) := p = 72

-- Targeting the proof that the area of the cross formed by the squares is 180 square units
theorem area_of_cross (n p : ℕ) (h1 : congruent_squares n) (h2 : perimeter_of_cross p) : 
  5 * (p / 12) ^ 2 = 180 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_cross_l2108_210845


namespace NUMINAMATH_GPT_smallest_solution_eq_sqrt_104_l2108_210810

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end NUMINAMATH_GPT_smallest_solution_eq_sqrt_104_l2108_210810


namespace NUMINAMATH_GPT_max_crosses_in_grid_l2108_210856

theorem max_crosses_in_grid : ∀ (n : ℕ), n = 16 → (∃ X : ℕ, X = 30 ∧
  ∀ (i j : ℕ), i < n → j < n → 
    (∀ k, k < n → (i ≠ k → X ≠ k)) ∧ 
    (∀ l, l < n → (j ≠ l → X ≠ l))) :=
by
  sorry

end NUMINAMATH_GPT_max_crosses_in_grid_l2108_210856


namespace NUMINAMATH_GPT_series_largest_prime_factor_of_111_l2108_210862

def series := [368, 689, 836]  -- given sequence series

def div_condition (n : Nat) := 
  ∃ k : Nat, n = 111 * k

def largest_prime_factor (n : Nat) (p : Nat) := 
  Prime p ∧ ∀ q : Nat, Prime q → q ∣ n → q ≤ p

theorem series_largest_prime_factor_of_111 :
  largest_prime_factor 111 37 := 
by
  sorry

end NUMINAMATH_GPT_series_largest_prime_factor_of_111_l2108_210862


namespace NUMINAMATH_GPT_shortest_distance_to_circle_l2108_210852

variable (A O T : Type)
variable (r d : ℝ)
variable [MetricSpace A]
variable [MetricSpace O]
variable [MetricSpace T]

open Real

theorem shortest_distance_to_circle (h : d = (4 / 3) * r) : 
  OA = (5 / 3) * r → shortest_dist = (2 / 3) * r :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_to_circle_l2108_210852


namespace NUMINAMATH_GPT_compare_negative_sqrt_values_l2108_210872

theorem compare_negative_sqrt_values : -3 * Real.sqrt 3 > -2 * Real.sqrt 7 := 
sorry

end NUMINAMATH_GPT_compare_negative_sqrt_values_l2108_210872


namespace NUMINAMATH_GPT_remainder_101_pow_47_mod_100_l2108_210870

theorem remainder_101_pow_47_mod_100 : (101 ^ 47) % 100 = 1 := by 
  sorry

end NUMINAMATH_GPT_remainder_101_pow_47_mod_100_l2108_210870


namespace NUMINAMATH_GPT_set_representation_l2108_210804

theorem set_representation : 
  { x : ℕ | x < 5 } = {0, 1, 2, 3, 4} :=
sorry

end NUMINAMATH_GPT_set_representation_l2108_210804


namespace NUMINAMATH_GPT_cost_of_each_pair_of_shorts_l2108_210840

variable (C : ℝ)
variable (h_discount : 3 * C - 2.7 * C = 3)

theorem cost_of_each_pair_of_shorts : C = 10 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_each_pair_of_shorts_l2108_210840


namespace NUMINAMATH_GPT_magazine_ad_extra_cost_l2108_210835

/--
The cost of purchasing a laptop through a magazine advertisement includes four monthly 
payments of $60.99 each and a one-time shipping and handling fee of $19.99. The in-store 
price of the laptop is $259.99. Prove that purchasing the laptop through the magazine 
advertisement results in an extra cost of 396 cents.
-/
theorem magazine_ad_extra_cost : 
  let in_store_price := 259.99
  let monthly_payment := 60.99
  let num_payments := 4
  let shipping_handling := 19.99
  let total_magazine_cost := (num_payments * monthly_payment) + shipping_handling
  (total_magazine_cost - in_store_price) * 100 = 396 := 
by
  sorry

end NUMINAMATH_GPT_magazine_ad_extra_cost_l2108_210835


namespace NUMINAMATH_GPT_second_divisor_is_24_l2108_210878

theorem second_divisor_is_24 (m n k l : ℤ) (hm : m = 288 * k + 47) (hn : m = n * l + 23) : n = 24 :=
by
  sorry

end NUMINAMATH_GPT_second_divisor_is_24_l2108_210878


namespace NUMINAMATH_GPT_total_length_correct_l2108_210892

def segment_lengths_Figure1 : List ℕ := [10, 3, 1, 1, 5, 7]

def removed_segments : List ℕ := [3, 1, 1, 5]

def remaining_segments_Figure2 : List ℕ := [10, (3 + 1 + 1), 7, 1]

def total_length_Figure2 : ℕ := remaining_segments_Figure2.sum

theorem total_length_correct :
  total_length_Figure2 = 23 :=
by
  sorry

end NUMINAMATH_GPT_total_length_correct_l2108_210892


namespace NUMINAMATH_GPT_part1_A_intersect_B_l2108_210864

def setA : Set ℝ := { x | x ^ 2 - 2 * x - 3 ≤ 0 }
def setB (m : ℝ) : Set ℝ := { x | (x - (m - 1)) * (x - (m + 1)) > 0 }

theorem part1_A_intersect_B (m : ℝ) (h : m = 0) : 
  setA ∩ setB m = { x | 1 < x ∧ x ≤ 3 } :=
sorry

end NUMINAMATH_GPT_part1_A_intersect_B_l2108_210864


namespace NUMINAMATH_GPT_fractional_eq_no_real_roots_l2108_210882

theorem fractional_eq_no_real_roots (k : ℝ) :
  (∀ x : ℝ, (x - 1) ≠ 0 → (k / (x - 1) + 3 ≠ x / (1 - x))) → k = -1 :=
by
  sorry

end NUMINAMATH_GPT_fractional_eq_no_real_roots_l2108_210882


namespace NUMINAMATH_GPT_largest_fraction_is_D_l2108_210806

-- Define the fractions as Lean variables
def A : ℚ := 2 / 6
def B : ℚ := 3 / 8
def C : ℚ := 4 / 12
def D : ℚ := 7 / 16
def E : ℚ := 9 / 24

-- Define a theorem to prove the largest fraction is D
theorem largest_fraction_is_D : max (max (max A B) (max C D)) E = D :=
by
  sorry

end NUMINAMATH_GPT_largest_fraction_is_D_l2108_210806


namespace NUMINAMATH_GPT_scientific_notation_of_1650000_l2108_210818

theorem scientific_notation_of_1650000 : (1650000 : ℝ) = 1.65 * 10^6 := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_scientific_notation_of_1650000_l2108_210818


namespace NUMINAMATH_GPT_length_of_hallway_is_six_l2108_210887

noncomputable def length_of_hallway (total_area_square_feet : ℝ) (central_area_side_length : ℝ) (hallway_width : ℝ) : ℝ :=
  (total_area_square_feet - (central_area_side_length * central_area_side_length)) / hallway_width

theorem length_of_hallway_is_six 
  (total_area_square_feet : ℝ)
  (central_area_side_length : ℝ)
  (hallway_width : ℝ)
  (h1 : total_area_square_feet = 124)
  (h2 : central_area_side_length = 10)
  (h3 : hallway_width = 4) :
  length_of_hallway total_area_square_feet central_area_side_length hallway_width = 6 := by
  sorry

end NUMINAMATH_GPT_length_of_hallway_is_six_l2108_210887


namespace NUMINAMATH_GPT_abc_unique_l2108_210809

theorem abc_unique (n : ℕ) (hn : 0 < n) (p : ℕ) (hp : Nat.Prime p) 
                   (a b c : ℤ) 
                   (h : a^n + p * b = b^n + p * c ∧ b^n + p * c = c^n + p * a) 
                   : a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_abc_unique_l2108_210809


namespace NUMINAMATH_GPT_integer_pairs_satisfy_equation_l2108_210891

theorem integer_pairs_satisfy_equation :
  ∀ (x y : ℤ), (x^2 * y + y^2 = x^3) → (x = 0 ∧ y = 0) ∨ (x = -4 ∧ y = -8) :=
by
  sorry

end NUMINAMATH_GPT_integer_pairs_satisfy_equation_l2108_210891


namespace NUMINAMATH_GPT_cities_real_distance_l2108_210815

def map_scale := 7 -- number of centimeters representing 35 kilometers
def real_distance_equiv := 35 -- number of kilometers that corresponds to map_scale

def centimeters_per_kilometer := real_distance_equiv / map_scale -- kilometers per centimeter

def distance_on_map := 49 -- number of centimeters cities are separated by on the map

theorem cities_real_distance : distance_on_map * centimeters_per_kilometer = 245 :=
by
  sorry

end NUMINAMATH_GPT_cities_real_distance_l2108_210815


namespace NUMINAMATH_GPT_calculate_expression_l2108_210859

theorem calculate_expression (x : ℝ) (h : x + 1/x = 3) : x^12 - 7 * x^6 + x^2 = 45363 * x - 17327 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2108_210859


namespace NUMINAMATH_GPT_am_gm_inequality_l2108_210855

open Real

theorem am_gm_inequality (
    a b c d e f : ℝ
) (h_nonneg_a : 0 ≤ a)
  (h_nonneg_b : 0 ≤ b)
  (h_nonneg_c : 0 ≤ c)
  (h_nonneg_d : 0 ≤ d)
  (h_nonneg_e : 0 ≤ e)
  (h_nonneg_f : 0 ≤ f)
  (h_cond_ab : a + b ≤ e)
  (h_cond_cd : c + d ≤ f) :
  sqrt (a * c) + sqrt (b * d) ≤ sqrt (e * f) := 
  by sorry

end NUMINAMATH_GPT_am_gm_inequality_l2108_210855


namespace NUMINAMATH_GPT_min_function_value_in_domain_l2108_210886

theorem min_function_value_in_domain :
  ∃ (x y : ℝ), (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) ∧ (∀ (x y : ℝ), (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) → (xy / (x^2 + y^2)) ≥ (60 / 169)) :=
sorry

end NUMINAMATH_GPT_min_function_value_in_domain_l2108_210886


namespace NUMINAMATH_GPT_digging_depth_l2108_210843

theorem digging_depth :
  (∃ (D : ℝ), 750 * D = 75000) → D = 100 :=
by
  sorry

end NUMINAMATH_GPT_digging_depth_l2108_210843


namespace NUMINAMATH_GPT_total_distance_covered_l2108_210896

variable (h : ℝ) (initial_height : ℝ := h) (bounce_ratio : ℝ := 0.8)

theorem total_distance_covered :
  initial_height + 2 * initial_height * bounce_ratio / (1 - bounce_ratio) = 13 * h :=
by 
  -- Proof omitted for now
  sorry

end NUMINAMATH_GPT_total_distance_covered_l2108_210896


namespace NUMINAMATH_GPT_leah_ride_time_l2108_210885

theorem leah_ride_time (x y : ℝ) (h1 : 90 * x = y) (h2 : 30 * (x + 2 * x) = y)
: ∃ t : ℝ, t = 67.5 :=
by
  -- Define 50% increase in length
  let y' := 1.5 * y
  -- Define escalator speed without Leah walking
  let k := 2 * x
  -- Calculate the time taken
  let t := y' / k
  -- Prove that this time is 67.5 seconds
  have ht : t = 67.5 := sorry
  exact ⟨t, ht⟩

end NUMINAMATH_GPT_leah_ride_time_l2108_210885


namespace NUMINAMATH_GPT_miki_pear_juice_l2108_210865

def total_pears : ℕ := 18
def total_oranges : ℕ := 10
def pear_juice_per_pear : ℚ := 10 / 2
def orange_juice_per_orange : ℚ := 12 / 3
def max_blend_volume : ℚ := 44

theorem miki_pear_juice : (total_oranges * orange_juice_per_orange = 40) ∧ (max_blend_volume - 40 = 4) → 
  ∃ p : ℚ, p * pear_juice_per_pear = 4 ∧ p = 0 :=
by
  sorry

end NUMINAMATH_GPT_miki_pear_juice_l2108_210865


namespace NUMINAMATH_GPT_find_g_3_l2108_210861

def g (x : ℝ) : ℝ := sorry

theorem find_g_3 (h : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = 3) : g 3 = 0 := 
by
  sorry

end NUMINAMATH_GPT_find_g_3_l2108_210861


namespace NUMINAMATH_GPT_cos_C_in_triangle_l2108_210836

theorem cos_C_in_triangle (A B C : ℝ)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_cos_A : Real.cos A = 3/5)
  (h_sin_B : Real.sin B = 12/13) :
  Real.cos C = 63/65 ∨ Real.cos C = 33/65 :=
sorry

end NUMINAMATH_GPT_cos_C_in_triangle_l2108_210836


namespace NUMINAMATH_GPT_difference_of_numbers_l2108_210871

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : |x - y| = 7 := 
by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l2108_210871


namespace NUMINAMATH_GPT_total_amount_l2108_210894

theorem total_amount (W X Y Z : ℝ) (h1 : X = 0.8 * W) (h2 : Y = 0.65 * W) (h3 : Z = 0.45 * W) (h4 : Y = 78) : 
  W + X + Y + Z = 348 := by
  sorry

end NUMINAMATH_GPT_total_amount_l2108_210894


namespace NUMINAMATH_GPT_students_taking_neither_l2108_210801

theorem students_taking_neither (total_students music_students art_students dance_students music_art music_dance art_dance music_art_dance : ℕ) :
  total_students = 2500 →
  music_students = 200 →
  art_students = 150 →
  dance_students = 100 →
  music_art = 75 →
  art_dance = 50 →
  music_dance = 40 →
  music_art_dance = 25 →
  total_students - ((music_students + art_students + dance_students) - (music_art + art_dance + music_dance) + music_art_dance) = 2190 :=
by
  intros
  sorry

end NUMINAMATH_GPT_students_taking_neither_l2108_210801


namespace NUMINAMATH_GPT_total_cows_l2108_210875

def number_of_cows_in_herd : ℕ := 40
def number_of_herds : ℕ := 8
def total_number_of_cows (cows_per_herd herds : ℕ) : ℕ := cows_per_herd * herds

theorem total_cows : total_number_of_cows number_of_cows_in_herd number_of_herds = 320 := by
  sorry

end NUMINAMATH_GPT_total_cows_l2108_210875


namespace NUMINAMATH_GPT_inclination_angle_of_line_m_l2108_210851

theorem inclination_angle_of_line_m
  (m : ℝ → ℝ → Prop)
  (l₁ l₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ x - y + 1 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ x - y - 1 = 0)
  (intersect_segment_length : ℝ)
  (h₃ : intersect_segment_length = 2 * Real.sqrt 2) :
  (∃ α : ℝ, (α = 15 ∨ α = 75) ∧ (∃ k : ℝ, ∀ x y, m x y ↔ y = k * x)) :=
by
  sorry

end NUMINAMATH_GPT_inclination_angle_of_line_m_l2108_210851


namespace NUMINAMATH_GPT_sum_of_first_five_terms_sequence_l2108_210866

-- Definitions derived from conditions
def seventh_term : ℤ := 4
def eighth_term : ℤ := 10
def ninth_term : ℤ := 16

-- The main theorem statement
theorem sum_of_first_five_terms_sequence : 
  ∃ (a d : ℤ), 
    a + 6 * d = seventh_term ∧
    a + 7 * d = eighth_term ∧
    a + 8 * d = ninth_term ∧
    (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = -100) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_five_terms_sequence_l2108_210866


namespace NUMINAMATH_GPT_find_original_number_l2108_210848

def digitsGPA (A B C : ℕ) : Prop := B^2 = A * C
def digitsAPA (X Y Z : ℕ) : Prop := 2 * Y = X + Z

theorem find_original_number (A B C X Y Z : ℕ) :
  100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C ≤ 999 ∧
  digitsGPA A B C ∧
  100 * X + 10 * Y + Z = (100 * A + 10 * B + C) - 200 ∧
  digitsAPA X Y Z →
  (100 * A + 10 * B + C) = 842 :=
sorry

end NUMINAMATH_GPT_find_original_number_l2108_210848


namespace NUMINAMATH_GPT_function_is_odd_and_increasing_l2108_210883

theorem function_is_odd_and_increasing :
  (∀ x : ℝ, (x^(1/3) : ℝ) = -( (-x)^(1/3) : ℝ)) ∧ (∀ x y : ℝ, x < y → (x^(1/3) : ℝ) < (y^(1/3) : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_function_is_odd_and_increasing_l2108_210883


namespace NUMINAMATH_GPT_courier_speeds_correctness_l2108_210800

noncomputable def courier_speeds : Prop :=
  ∃ (s1 s2 : ℕ), (s1 * 8 + s2 * 8 = 176) ∧ (s1 = 60 / 5) ∧ (s2 = 60 / 6)

theorem courier_speeds_correctness : courier_speeds :=
by
  sorry

end NUMINAMATH_GPT_courier_speeds_correctness_l2108_210800


namespace NUMINAMATH_GPT_fraction_exponent_multiplication_l2108_210890

theorem fraction_exponent_multiplication :
  ( (8/9 : ℚ)^2 * (1/3 : ℚ)^2 = (64/729 : ℚ) ) :=
by
  -- here we would write out the detailed proof
  sorry

end NUMINAMATH_GPT_fraction_exponent_multiplication_l2108_210890


namespace NUMINAMATH_GPT_polynomial_roots_arithmetic_progression_l2108_210844

theorem polynomial_roots_arithmetic_progression (m n : ℝ)
  (h : ∃ a : ℝ, ∃ d : ℝ, ∃ b : ℝ,
   (a = b ∧ (b + d) + (b + 2*d) + (b + 3*d) + b = 0) ∧
   (b * (b + d) * (b + 2*d) * (b + 3*d) = 144) ∧
   b ≠ (b + d) ∧ (b + d) ≠ (b + 2*d) ∧ (b + 2*d) ≠ (b + 3*d)) :
  m = -40 := sorry

end NUMINAMATH_GPT_polynomial_roots_arithmetic_progression_l2108_210844


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2108_210889

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_perimeter 
  (a b c : ℝ) 
  (h_iso : is_isosceles a b c) 
  (h1 : a = 2 ∨ a = 4) 
  (h2 : b = 2 ∨ b = 4) 
  (h3 : c = 2 ∨ c = 4) :
  a + b + c = 10 :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2108_210889


namespace NUMINAMATH_GPT_radius_scientific_notation_l2108_210814

theorem radius_scientific_notation :
  696000 = 6.96 * 10^5 :=
sorry

end NUMINAMATH_GPT_radius_scientific_notation_l2108_210814


namespace NUMINAMATH_GPT_gross_revenue_is_47_l2108_210854

def total_net_profit : ℤ := 44
def babysitting_profit : ℤ := 31
def lemonade_stand_expense : ℤ := 34

def gross_revenue_from_lemonade_stand (P_t P_b E : ℤ) : ℤ :=
  P_t - P_b + E

theorem gross_revenue_is_47 :
  gross_revenue_from_lemonade_stand total_net_profit babysitting_profit lemonade_stand_expense = 47 :=
by
  sorry

end NUMINAMATH_GPT_gross_revenue_is_47_l2108_210854


namespace NUMINAMATH_GPT_greatest_distance_centers_of_circles_in_rectangle_l2108_210819

/--
Two circles are drawn in a 20-inch by 16-inch rectangle,
each circle with a diameter of 8 inches.
Prove that the greatest possible distance between 
the centers of the two circles without extending beyond the 
rectangular region is 4 * sqrt 13 inches.
-/
theorem greatest_distance_centers_of_circles_in_rectangle :
  let diameter := 8
  let width := 20
  let height := 16
  let radius := diameter / 2
  let reduced_width := width - 2 * radius
  let reduced_height := height - 2 * radius
  let distance := Real.sqrt ((reduced_width ^ 2) + (reduced_height ^ 2))
  distance = 4 * Real.sqrt 13 := by
    sorry

end NUMINAMATH_GPT_greatest_distance_centers_of_circles_in_rectangle_l2108_210819


namespace NUMINAMATH_GPT_find_k_l2108_210877

-- Definitions of the conditions as given in the problem
def total_amount (A B C : ℕ) : Prop := A + B + C = 585
def c_share (C : ℕ) : Prop := C = 260
def equal_shares (A B C k : ℕ) : Prop := 4 * A = k * C ∧ 6 * B = k * C

-- The theorem we need to prove
theorem find_k (A B C k : ℕ) (h_tot: total_amount A B C)
  (h_c: c_share C) (h_eq: equal_shares A B C k) : k = 3 := by 
  sorry

end NUMINAMATH_GPT_find_k_l2108_210877


namespace NUMINAMATH_GPT_sara_initial_peaches_l2108_210832

variable (p : ℕ)

def initial_peaches (picked_peaches total_peaches : ℕ) :=
  total_peaches - picked_peaches

theorem sara_initial_peaches :
  initial_peaches 37 61 = 24 :=
by
  -- This follows directly from the definition of initial_peaches
  sorry

end NUMINAMATH_GPT_sara_initial_peaches_l2108_210832


namespace NUMINAMATH_GPT_find_rs_l2108_210899

theorem find_rs :
  ∃ r s : ℝ, ∀ x : ℝ, 8 * x^4 - 4 * x^3 - 42 * x^2 + 45 * x - 10 = 8 * (x - r) ^ 2 * (x - s) * (x - 1) :=
sorry

end NUMINAMATH_GPT_find_rs_l2108_210899


namespace NUMINAMATH_GPT_find_halls_per_floor_l2108_210820

theorem find_halls_per_floor
  (H : ℤ)
  (floors_first_wing : ℤ := 9)
  (rooms_per_hall_first_wing : ℤ := 32)
  (floors_second_wing : ℤ := 7)
  (halls_per_floor_second_wing : ℤ := 9)
  (rooms_per_hall_second_wing : ℤ := 40)
  (total_rooms : ℤ := 4248) :
  9 * H * 32 + 7 * 9 * 40 = 4248 → H = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_halls_per_floor_l2108_210820


namespace NUMINAMATH_GPT_xy_zero_l2108_210822

theorem xy_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 0 :=
by
  sorry

end NUMINAMATH_GPT_xy_zero_l2108_210822


namespace NUMINAMATH_GPT_omega_range_l2108_210817

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem omega_range (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 4), f ω x ≥ -2) :
  0 < ω ∧ ω ≤ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_omega_range_l2108_210817


namespace NUMINAMATH_GPT_number_of_girls_in_class_l2108_210898

theorem number_of_girls_in_class (B G : ℕ) (h1 : G = 4 * B / 10) (h2 : B + G = 35) : G = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_in_class_l2108_210898


namespace NUMINAMATH_GPT_max_cookies_andy_could_have_eaten_l2108_210874

theorem max_cookies_andy_could_have_eaten (x k : ℕ) (hk : k > 0) 
  (h_total : x + k * x + 2 * x = 36) : x ≤ 9 :=
by
  -- Using the conditions to construct the proof (which is not required based on the instructions)
  sorry

end NUMINAMATH_GPT_max_cookies_andy_could_have_eaten_l2108_210874


namespace NUMINAMATH_GPT_quadratic_form_m_neg3_l2108_210895

theorem quadratic_form_m_neg3
  (m : ℝ)
  (h_exp : m^2 - 7 = 2)
  (h_coef : m ≠ 3) :
  m = -3 := by
  sorry

end NUMINAMATH_GPT_quadratic_form_m_neg3_l2108_210895


namespace NUMINAMATH_GPT_log_expression_evaluation_l2108_210829

noncomputable def log2 : ℝ := Real.log 2
noncomputable def log5 : ℝ := Real.log 5

theorem log_expression_evaluation (condition : log2 + log5 = 1) :
  log2^2 + log2 * log5 + log5 - (Real.sqrt 2 - 1)^0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_log_expression_evaluation_l2108_210829


namespace NUMINAMATH_GPT_geometric_seq_tenth_term_l2108_210826

theorem geometric_seq_tenth_term :
  let a := 12
  let r := (1 / 2 : ℝ)
  (a * r^9) = (3 / 128 : ℝ) :=
by
  let a := 12
  let r := (1 / 2 : ℝ)
  show a * r^9 = 3 / 128
  sorry

end NUMINAMATH_GPT_geometric_seq_tenth_term_l2108_210826


namespace NUMINAMATH_GPT_petya_vasya_meet_at_lamp_64_l2108_210839

-- Definitions of positions of Petya and Vasya
def Petya_position (x : ℕ) : ℕ := x - 21 -- Petya starts from the 1st lamp and is at the 22nd lamp
def Vasya_position (x : ℕ) : ℕ := 88 - x -- Vasya starts from the 100th lamp and is at the 88th lamp

-- Condition that both lanes add up to 64
theorem petya_vasya_meet_at_lamp_64 : ∀ x y : ℕ, 
    Petya_position x = Vasya_position y -> x = 64 :=
by
  intro x y
  rw [Petya_position, Vasya_position]
  sorry

end NUMINAMATH_GPT_petya_vasya_meet_at_lamp_64_l2108_210839


namespace NUMINAMATH_GPT_percent_gold_coins_l2108_210867

variables (total_objects : ℝ) (coins_beads_percent beads_percent gold_coins_percent : ℝ)
           (h1 : coins_beads_percent = 0.75)
           (h2 : beads_percent = 0.15)
           (h3 : gold_coins_percent = 0.60)

theorem percent_gold_coins : (gold_coins_percent * (coins_beads_percent - beads_percent)) = 0.36 :=
by
  have coins_percent := coins_beads_percent - beads_percent
  have gold_coins_total_percent := gold_coins_percent * coins_percent
  exact sorry

end NUMINAMATH_GPT_percent_gold_coins_l2108_210867


namespace NUMINAMATH_GPT_simplify_exponents_product_l2108_210846

theorem simplify_exponents_product :
  (10^0.5) * (10^0.25) * (10^0.15) * (10^0.05) * (10^1.05) = 100 := by
sorry

end NUMINAMATH_GPT_simplify_exponents_product_l2108_210846


namespace NUMINAMATH_GPT_initial_birds_in_cage_l2108_210876

-- Define a theorem to prove the initial number of birds in the cage
theorem initial_birds_in_cage (B : ℕ) 
  (H1 : 2 / 15 * B = 8) : B = 60 := 
by sorry

end NUMINAMATH_GPT_initial_birds_in_cage_l2108_210876


namespace NUMINAMATH_GPT_triangle_area_50_l2108_210893

theorem triangle_area_50 :
  let A := (0, 0)
  let B := (0, 10)
  let C := (-10, 0)
  let base := 10
  let height := 10
  0 + base * height / 2 = 50 := by
sorry

end NUMINAMATH_GPT_triangle_area_50_l2108_210893


namespace NUMINAMATH_GPT_abs_diff_roots_eq_sqrt_13_l2108_210888

theorem abs_diff_roots_eq_sqrt_13 {x₁ x₂ : ℝ} (h : x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0) :
  |x₁ - x₂| = Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_abs_diff_roots_eq_sqrt_13_l2108_210888


namespace NUMINAMATH_GPT_trapezium_area_l2108_210842

theorem trapezium_area (a b d : ℕ) (h₁ : a = 28) (h₂ : b = 18) (h₃ : d = 15) :
  (a + b) * d / 2 = 345 := by
{
  sorry
}

end NUMINAMATH_GPT_trapezium_area_l2108_210842


namespace NUMINAMATH_GPT_mr_zander_total_payment_l2108_210813

noncomputable def total_cost (cement_bags : ℕ) (price_per_bag : ℝ) (sand_lorries : ℕ) 
(tons_per_lorry : ℝ) (price_per_ton : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let cement_cost_before_discount := cement_bags * price_per_bag
  let discount := cement_cost_before_discount * discount_rate
  let cement_cost_after_discount := cement_cost_before_discount - discount
  let sand_cost_before_tax := sand_lorries * tons_per_lorry * price_per_ton
  let tax := sand_cost_before_tax * tax_rate
  let sand_cost_after_tax := sand_cost_before_tax + tax
  cement_cost_after_discount + sand_cost_after_tax

theorem mr_zander_total_payment :
  total_cost 500 10 20 10 40 0.05 0.07 = 13310 := 
sorry

end NUMINAMATH_GPT_mr_zander_total_payment_l2108_210813


namespace NUMINAMATH_GPT_sum_of_possible_values_of_N_l2108_210823

theorem sum_of_possible_values_of_N :
  (∃ N : ℝ, N * (N - 7) = 12) → (∃ N₁ N₂ : ℝ, (N₁ * (N₁ - 7) = 12 ∧ N₂ * (N₂ - 7) = 12) ∧ N₁ + N₂ = 7) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_N_l2108_210823


namespace NUMINAMATH_GPT_bullet_train_speed_is_70kmph_l2108_210863

noncomputable def bullet_train_speed (train_length time_man  : ℚ) (man_speed_kmph : ℕ) : ℚ :=
  let man_speed_ms : ℚ := man_speed_kmph * 1000 / 3600
  let relative_speed : ℚ := train_length / time_man
  let train_speed_ms : ℚ := relative_speed - man_speed_ms
  train_speed_ms * 3600 / 1000

theorem bullet_train_speed_is_70kmph :
  bullet_train_speed 160 7.384615384615384 8 = 70 :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_bullet_train_speed_is_70kmph_l2108_210863


namespace NUMINAMATH_GPT_determine_b_l2108_210868

theorem determine_b (A B C : ℝ) (a b c : ℝ)
  (angle_C_eq_4A : C = 4 * A)
  (a_eq_30 : a = 30)
  (c_eq_48 : c = 48)
  (law_of_sines : ∀ x y, x / Real.sin A = y / Real.sin (4 * A))
  (cos_eq_solution : 4 * Real.cos A ^ 3 - 4 * Real.cos A = 8 / 5) :
  ∃ b : ℝ, b = 30 * (5 - 20 * (1 - Real.cos A ^ 2) + 16 * (1 - Real.cos A ^ 2) ^ 2) :=
by 
  sorry

end NUMINAMATH_GPT_determine_b_l2108_210868


namespace NUMINAMATH_GPT_triangle_inequality_right_triangle_l2108_210897

theorem triangle_inequality_right_triangle
  (a b c : ℝ) (h : c^2 = a^2 + b^2) : (a + b) / Real.sqrt 2 ≤ c :=
by sorry

end NUMINAMATH_GPT_triangle_inequality_right_triangle_l2108_210897


namespace NUMINAMATH_GPT_total_students_l2108_210834

variables (B G : ℕ)
variables (two_thirds_boys : 2 * B = 3 * 400)
variables (three_fourths_girls : 3 * G = 4 * 150)
variables (total_participants : B + G = 800)

theorem total_students (B G : ℕ)
  (two_thirds_boys : 2 * B = 3 * 400)
  (three_fourths_girls : 3 * G = 4 * 150)
  (total_participants : B + G = 800) :
  B + G = 800 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l2108_210834


namespace NUMINAMATH_GPT_total_number_of_bottles_l2108_210858

def water_bottles := 2 * 12
def orange_juice_bottles := (7 / 4) * 12
def apple_juice_bottles := water_bottles + 6
def total_bottles := water_bottles + orange_juice_bottles + apple_juice_bottles

theorem total_number_of_bottles :
  total_bottles = 75 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_bottles_l2108_210858


namespace NUMINAMATH_GPT_division_remainder_example_l2108_210838

theorem division_remainder_example :
  ∃ n, n = 20 * 10 + 10 ∧ n = 210 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_example_l2108_210838


namespace NUMINAMATH_GPT_ratio_of_lengths_l2108_210828

noncomputable def total_fence_length : ℝ := 640
noncomputable def short_side_length : ℝ := 80

theorem ratio_of_lengths (L S : ℝ) (h1 : 2 * L + 2 * S = total_fence_length) (h2 : S = short_side_length) :
  L / S = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_lengths_l2108_210828


namespace NUMINAMATH_GPT_quadratic_complete_square_l2108_210811

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 - 7 * x + 6) = (x - 7 / 2) ^ 2 - 25 / 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l2108_210811


namespace NUMINAMATH_GPT_sum_of_digits_next_exact_multiple_l2108_210831

noncomputable def Michael_next_age_sum_of_digits (L M T n : ℕ) : ℕ :=
  let next_age := M + n
  ((next_age / 10) % 10) + (next_age % 10)

theorem sum_of_digits_next_exact_multiple :
  ∀ (L M T n : ℕ),
    T = 2 →
    M = L + 4 →
    (∀ k : ℕ, k < 8 → ∃ m : ℕ, L = m * T + k * T) →
    (∃ n, (M + n) % (T + n) = 0) →
    Michael_next_age_sum_of_digits L M T n = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_digits_next_exact_multiple_l2108_210831


namespace NUMINAMATH_GPT_sum_of_squares_of_non_zero_digits_from_10_to_99_l2108_210812

-- Definition of the sum of squares of digits from 1 to 9
def P : ℕ := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2)

-- Definition of the sum of squares of the non-zero digits of the integers from 10 to 99
def T : ℕ := 20 * P

-- Theorem stating that T equals 5700
theorem sum_of_squares_of_non_zero_digits_from_10_to_99 : T = 5700 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_non_zero_digits_from_10_to_99_l2108_210812


namespace NUMINAMATH_GPT_customer_payment_strawberries_watermelons_max_discount_value_l2108_210880

-- Definitions for prices
def price_strawberries : ℕ := 60
def price_jingbai_pears : ℕ := 65
def price_watermelons : ℕ := 80
def price_peaches : ℕ := 90

-- Definition for condition on minimum purchase for promotion
def min_purchase_for_promotion : ℕ := 120

-- Definition for percentage Li Ming receives
def li_ming_percentage : ℕ := 80
def customer_percentage : ℕ := 100

-- Proof problem for part 1
theorem customer_payment_strawberries_watermelons (x : ℕ) (total_price : ℕ) :
  x = 10 →
  total_price = price_strawberries + price_watermelons →
  total_price >= min_purchase_for_promotion →
  total_price - x = 130 :=
  by sorry

-- Proof problem for part 2
theorem max_discount_value (m x : ℕ) :
  m >= min_purchase_for_promotion →
  (m - x) * li_ming_percentage / customer_percentage ≥ m * 7 / 10 →
  x ≤ m / 8 :=
  by sorry

end NUMINAMATH_GPT_customer_payment_strawberries_watermelons_max_discount_value_l2108_210880


namespace NUMINAMATH_GPT_avg_transformation_l2108_210860

theorem avg_transformation
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 2) :
  ((3 * x₁ + 1) + (3 * x₂ + 1) + (3 * x₃ + 1) + (3 * x₄ + 1) + (3 * x₅ + 1)) / 5 = 7 :=
by
  sorry

end NUMINAMATH_GPT_avg_transformation_l2108_210860


namespace NUMINAMATH_GPT_at_least_one_l2108_210849

axiom P : Prop  -- person A is an outstanding student
axiom Q : Prop  -- person B is an outstanding student

theorem at_least_one (H : ¬(¬P ∧ ¬Q)) : P ∨ Q :=
sorry

end NUMINAMATH_GPT_at_least_one_l2108_210849


namespace NUMINAMATH_GPT_find_x_l2108_210873

theorem find_x (x y z : ℤ) (h1 : 4 * x + y + z = 80) (h2 : 2 * x - y - z = 40) (h3 : 3 * x + y - z = 20) : x = 20 := by
  sorry

end NUMINAMATH_GPT_find_x_l2108_210873


namespace NUMINAMATH_GPT_basketball_club_members_l2108_210821

theorem basketball_club_members :
  let sock_cost := 6
  let tshirt_additional_cost := 8
  let total_cost := 4440
  let cost_per_member := sock_cost + 2 * (sock_cost + tshirt_additional_cost)
  total_cost / cost_per_member = 130 :=
by
  sorry

end NUMINAMATH_GPT_basketball_club_members_l2108_210821


namespace NUMINAMATH_GPT_jill_sales_goal_l2108_210850

def first_customer : ℕ := 5
def second_customer : ℕ := 4 * first_customer
def third_customer : ℕ := second_customer / 2
def fourth_customer : ℕ := 3 * third_customer
def fifth_customer : ℕ := 10
def boxes_sold : ℕ := first_customer + second_customer + third_customer + fourth_customer + fifth_customer
def boxes_left : ℕ := 75
def sales_goal : ℕ := boxes_sold + boxes_left

theorem jill_sales_goal : sales_goal = 150 := by
  sorry

end NUMINAMATH_GPT_jill_sales_goal_l2108_210850


namespace NUMINAMATH_GPT_tangent_lines_count_l2108_210853

noncomputable def number_of_tangent_lines (r1 r2 : ℝ) (k : ℕ) : ℕ :=
if r1 = 2 ∧ r2 = 3 then 5 else 0

theorem tangent_lines_count: 
∃ k : ℕ, number_of_tangent_lines 2 3 k = 5 :=
by sorry

end NUMINAMATH_GPT_tangent_lines_count_l2108_210853
