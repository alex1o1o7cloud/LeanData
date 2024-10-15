import Mathlib

namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l115_11557

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = -1 ∧ b = 11 ∧ ∀ x, x > a ∧ x < b → (deriv f x) < 0 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l115_11557


namespace NUMINAMATH_GPT_line_intersects_circle_l115_11532

/-- The positional relationship between the line y = ax + 1 and the circle x^2 + y^2 - 2x - 3 = 0
    is always intersecting for any real number a. -/
theorem line_intersects_circle (a : ℝ) : 
    ∀ a : ℝ, ∃ x y : ℝ, y = a * x + 1 ∧ x^2 + y^2 - 2 * x - 3 = 0 :=
by
    sorry

end NUMINAMATH_GPT_line_intersects_circle_l115_11532


namespace NUMINAMATH_GPT_rainfall_ratio_l115_11522

theorem rainfall_ratio (R1 R2 : ℕ) (H1 : R2 = 18) (H2 : R1 + R2 = 30) : R2 / R1 = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_rainfall_ratio_l115_11522


namespace NUMINAMATH_GPT_probability_not_buy_l115_11571

-- Define the given probability of Sam buying a new book
def P_buy : ℚ := 5 / 8

-- Theorem statement: The probability that Sam will not buy a new book is 3 / 8
theorem probability_not_buy : 1 - P_buy = 3 / 8 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_probability_not_buy_l115_11571


namespace NUMINAMATH_GPT_runs_in_last_match_l115_11526

-- Definitions based on the conditions
def initial_bowling_average : ℝ := 12.4
def wickets_last_match : ℕ := 7
def decrease_average : ℝ := 0.4
def new_average : ℝ := initial_bowling_average - decrease_average
def approximate_wickets_before : ℕ := 145

-- The Lean statement of the problem
theorem runs_in_last_match (R : ℝ) :
  ((initial_bowling_average * approximate_wickets_before + R) / 
   (approximate_wickets_before + wickets_last_match) = new_average) →
   R = 28 :=
by
  sorry

end NUMINAMATH_GPT_runs_in_last_match_l115_11526


namespace NUMINAMATH_GPT_additional_hours_needed_l115_11503

-- Define the conditions
def speed : ℕ := 5  -- kilometers per hour
def total_distance : ℕ := 30 -- kilometers
def hours_walked : ℕ := 3 -- hours

-- Define the statement to prove
theorem additional_hours_needed : total_distance / speed - hours_walked = 3 := 
by
  sorry

end NUMINAMATH_GPT_additional_hours_needed_l115_11503


namespace NUMINAMATH_GPT_purely_imaginary_a_eq_1_fourth_quadrant_a_range_l115_11567

-- Definitions based on given conditions
def z (a : ℝ) := (a^2 - 7 * a + 6) + (a^2 - 5 * a - 6) * Complex.I

-- Purely imaginary proof statement
theorem purely_imaginary_a_eq_1 (a : ℝ) 
  (hz : (a^2 - 7 * a + 6) + (a^2 - 5 * a - 6) * Complex.I = (0 : ℂ) + (a^2 - 5 * a - 6) * Complex.I) :
  a = 1 := by 
  sorry

-- Fourth quadrant proof statement
theorem fourth_quadrant_a_range (a : ℝ) 
  (hz1 : a^2 - 7 * a + 6 > 0) 
  (hz2 : a^2 - 5 * a - 6 < 0) : 
  -1 < a ∧ a < 1 := by 
  sorry

end NUMINAMATH_GPT_purely_imaginary_a_eq_1_fourth_quadrant_a_range_l115_11567


namespace NUMINAMATH_GPT_greatest_divisor_of_sum_of_arith_seq_l115_11518

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end NUMINAMATH_GPT_greatest_divisor_of_sum_of_arith_seq_l115_11518


namespace NUMINAMATH_GPT_intersection_A_B_l115_11525

-- Definition of sets A and B
def A : Set ℤ := {0, 1, 2, 3}
def B : Set ℤ := { x | -1 ≤ x ∧ x < 3 }

-- Statement to prove
theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := 
sorry

end NUMINAMATH_GPT_intersection_A_B_l115_11525


namespace NUMINAMATH_GPT_new_radius_of_circle_l115_11555

theorem new_radius_of_circle
  (r_1 : ℝ)
  (A_1 : ℝ := π * r_1^2)
  (r_2 : ℝ)
  (A_2 : ℝ := 0.64 * A_1) 
  (h1 : r_1 = 5) 
  (h2 : A_2 = π * r_2^2) : 
  r_2 = 4 :=
by 
  sorry

end NUMINAMATH_GPT_new_radius_of_circle_l115_11555


namespace NUMINAMATH_GPT_find_x_l115_11594

/-- 
Prove that the value of x is 25 degrees, given the following conditions:
1. The sum of the angles in triangle BAC: angle_BAC + 50° + 55° = 180°
2. The angles forming a straight line DAE: 80° + angle_BAC + x = 180°
-/
theorem find_x (angle_BAC : ℝ) (x : ℝ)
  (h1 : angle_BAC + 50 + 55 = 180)
  (h2 : 80 + angle_BAC + x = 180) :
  x = 25 :=
  sorry

end NUMINAMATH_GPT_find_x_l115_11594


namespace NUMINAMATH_GPT_max_probability_pc_l115_11529

variables (p1 p2 p3 : ℝ)
variable (h : p3 > p2 ∧ p2 > p1 ∧ p1 > 0)

def PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem max_probability_pc : PC > PA ∧ PC > PB := 
by 
  sorry

end NUMINAMATH_GPT_max_probability_pc_l115_11529


namespace NUMINAMATH_GPT_length_of_train_is_correct_l115_11549

-- Definitions based on conditions
def speed_kmh := 90
def time_sec := 10

-- Convert speed from km/hr to m/s
def speed_ms := speed_kmh * (1000 / 3600)

-- Calculate the length of the train
def length_of_train := speed_ms * time_sec

-- Theorem to prove the length of the train
theorem length_of_train_is_correct : length_of_train = 250 := by
  sorry

end NUMINAMATH_GPT_length_of_train_is_correct_l115_11549


namespace NUMINAMATH_GPT_intersection_A_B_intersection_A_complementB_l115_11569

-- Definitions of the sets A and B
def setA : Set ℝ := { x | -5 ≤ x ∧ x ≤ 3 }
def setB : Set ℝ := { x | x < -2 ∨ x > 4 }

-- Proof problem 1: A ∩ B = { x | -5 ≤ x < -2 }
theorem intersection_A_B:
  setA ∩ setB = { x : ℝ | -5 ≤ x ∧ x < -2 } :=
sorry

-- Definition of the complement of B
def complB : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

-- Proof problem 2: A ∩ (complB) = { x | -2 ≤ x ≤ 3 }
theorem intersection_A_complementB:
  setA ∩ complB = { x : ℝ | -2 ≤ x ∧ x ≤ 3 } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_intersection_A_complementB_l115_11569


namespace NUMINAMATH_GPT_running_speed_l115_11575

theorem running_speed (side : ℕ) (time_seconds : ℕ) (speed_result : ℕ) 
  (h1 : side = 50) (h2 : time_seconds = 60) (h3 : speed_result = 12) : 
  (4 * side * 3600) / (time_seconds * 1000) = speed_result :=
by
  sorry

end NUMINAMATH_GPT_running_speed_l115_11575


namespace NUMINAMATH_GPT_cats_to_dogs_ratio_l115_11533

theorem cats_to_dogs_ratio
    (cats dogs : ℕ)
    (ratio : cats / dogs = 3 / 4)
    (num_cats : cats = 18) :
    dogs = 24 :=
by
    sorry

end NUMINAMATH_GPT_cats_to_dogs_ratio_l115_11533


namespace NUMINAMATH_GPT_growth_rate_double_l115_11595

noncomputable def lake_coverage (days : ℕ) : ℝ := if days = 39 then 1 else if days = 38 then 0.5 else 0  -- Simplified condition statement

theorem growth_rate_double (days : ℕ) : 
  (lake_coverage 39 = 1) → (lake_coverage 38 = 0.5) → (∀ n, lake_coverage (n + 1) = 2 * lake_coverage n) := 
  by 
  intros h39 h38 
  apply sorry  -- Proof not required

end NUMINAMATH_GPT_growth_rate_double_l115_11595


namespace NUMINAMATH_GPT_total_cost_of_tshirts_l115_11561

theorem total_cost_of_tshirts
  (White_packs : ℕ := 3) (Blue_packs : ℕ := 2) (Red_packs : ℕ := 4) (Green_packs : ℕ := 1) 
  (White_price_per_pack : ℝ := 12) (Blue_price_per_pack : ℝ := 8) (Red_price_per_pack : ℝ := 10) (Green_price_per_pack : ℝ := 6) 
  (White_discount : ℝ := 0.10) (Blue_discount : ℝ := 0.05) (Red_discount : ℝ := 0.15) (Green_discount : ℝ := 0.00) :
  White_packs * White_price_per_pack * (1 - White_discount) +
  Blue_packs * Blue_price_per_pack * (1 - Blue_discount) +
  Red_packs * Red_price_per_pack * (1 - Red_discount) +
  Green_packs * Green_price_per_pack * (1 - Green_discount) = 87.60 := by
    sorry

end NUMINAMATH_GPT_total_cost_of_tshirts_l115_11561


namespace NUMINAMATH_GPT_not_integer_fraction_l115_11593

theorem not_integer_fraction (a b : ℕ) (h1 : a > b) (h2 : b > 2) : ¬ (∃ k : ℤ, (2^a + 1) = k * (2^b - 1)) :=
sorry

end NUMINAMATH_GPT_not_integer_fraction_l115_11593


namespace NUMINAMATH_GPT_decreasing_power_function_l115_11528

theorem decreasing_power_function (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(m^2 + m - 1) < (m^2 - m - 1) * (x + 1) ^ (m^2 + m - 1)) →
  m = -1 :=
sorry

end NUMINAMATH_GPT_decreasing_power_function_l115_11528


namespace NUMINAMATH_GPT_length_of_train_l115_11597

-- Definitions for the given conditions:
def speed : ℝ := 60   -- in kmph
def time : ℝ := 20    -- in seconds
def platform_length : ℝ := 213.36  -- in meters

-- Conversion factor from km/h to m/s
noncomputable def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Total distance covered by train while crossing the platform
noncomputable def total_distance (speed_in_kmph : ℝ) (time_in_seconds : ℝ) : ℝ := 
  (kmph_to_mps speed_in_kmph) * time_in_seconds

-- Length of the train
noncomputable def train_length (total_distance_covered : ℝ) (platform_len : ℝ) : ℝ :=
  total_distance_covered - platform_len

-- Expected length of the train
def expected_train_length : ℝ := 120.04

-- Theorem to prove the length of the train given the conditions
theorem length_of_train : 
  train_length (total_distance speed time) platform_length = expected_train_length :=
by 
  sorry

end NUMINAMATH_GPT_length_of_train_l115_11597


namespace NUMINAMATH_GPT_range_of_m_l115_11564

theorem range_of_m (m : ℝ) :
  (∃ (x1 x2 : ℝ), (2*x1^2 - 2*x1 + 3*m - 1 = 0 ∧ 2*x2^2 - 2*x2 + 3*m - 1 = 0) ∧ (x1 * x2 > x1 + x2 - 4)) →
  -5/3 < m ∧ m ≤ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l115_11564


namespace NUMINAMATH_GPT_worms_stolen_correct_l115_11521

-- Given conditions translated into Lean statements
def num_babies : ℕ := 6
def worms_per_baby_per_day : ℕ := 3
def papa_bird_worms : ℕ := 9
def mama_bird_initial_worms : ℕ := 13
def additional_worms_needed : ℕ := 34

-- From the conditions, determine the total number of worms needed for 3 days
def total_worms_needed : ℕ := worms_per_baby_per_day * num_babies * 3

-- Calculate how many worms they will have after catching additional worms
def total_worms_after_catching_more : ℕ := papa_bird_worms + mama_bird_initial_worms + additional_worms_needed

-- Amount suspected to be stolen
def worms_stolen : ℕ := total_worms_after_catching_more - total_worms_needed

theorem worms_stolen_correct : worms_stolen = 2 :=
by sorry

end NUMINAMATH_GPT_worms_stolen_correct_l115_11521


namespace NUMINAMATH_GPT_meaningful_expression_range_l115_11544

theorem meaningful_expression_range (x : ℝ) : (3 * x + 9 ≥ 0) ∧ (x ≠ 2) ↔ (x ≥ -3 ∧ x ≠ 2) := by
  sorry

end NUMINAMATH_GPT_meaningful_expression_range_l115_11544


namespace NUMINAMATH_GPT_calculator_display_after_50_presses_l115_11540

theorem calculator_display_after_50_presses :
  let initial_display := 3
  let operation (x : ℚ) := 1 / (1 - x)
  (Nat.iterate operation 50 initial_display) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_calculator_display_after_50_presses_l115_11540


namespace NUMINAMATH_GPT_max_last_digit_of_sequence_l115_11543

theorem max_last_digit_of_sequence :
  ∀ (s : Fin 1001 → ℕ), 
  (s 0 = 2) →
  (∀ (i : Fin 1000), (s i) * 10 + (s i.succ) ∈ {n | n % 17 = 0 ∨ n % 23 = 0}) →
  ∃ (d : ℕ), (d = s ⟨1000, sorry⟩) ∧ (∀ (d' : ℕ), d' = s ⟨1000, sorry⟩ → d' ≤ d) ∧ (d = 2) :=
by
  intros s h1 h2
  use 2
  sorry

end NUMINAMATH_GPT_max_last_digit_of_sequence_l115_11543


namespace NUMINAMATH_GPT_cricket_player_average_increase_l115_11599

theorem cricket_player_average_increase (total_innings initial_innings next_run : ℕ) (initial_average desired_increase : ℕ) 
(h1 : initial_innings = 10) (h2 : initial_average = 32) (h3 : next_run = 76) : desired_increase = 4 :=
by
  sorry

end NUMINAMATH_GPT_cricket_player_average_increase_l115_11599


namespace NUMINAMATH_GPT_intersect_range_k_l115_11535

theorem intersect_range_k : 
  ∀ k : ℝ, (∃ x y : ℝ, x^2 - (kx + 2)^2 = 6) ↔ 
  -Real.sqrt (5 / 3) < k ∧ k < Real.sqrt (5 / 3) := 
by sorry

end NUMINAMATH_GPT_intersect_range_k_l115_11535


namespace NUMINAMATH_GPT_find_quadratic_eq_l115_11596

theorem find_quadratic_eq (x y : ℝ) (hx : x + y = 10) (hy : |x - y| = 12) :
    ∃ a b c : ℝ, a = 1 ∧ b = -10 ∧ c = -11 ∧ (x^2 + b * x + c = 0) ∧ (y^2 + b * y + c = 0) := by
  sorry

end NUMINAMATH_GPT_find_quadratic_eq_l115_11596


namespace NUMINAMATH_GPT_text_message_cost_eq_l115_11586

theorem text_message_cost_eq (x : ℝ) (CA CB : ℝ) : 
  (CA = 0.25 * x + 9) → (CB = 0.40 * x) → CA = CB → x = 60 :=
by
  intros hCA hCB heq
  sorry

end NUMINAMATH_GPT_text_message_cost_eq_l115_11586


namespace NUMINAMATH_GPT_Carla_is_2_years_older_than_Karen_l115_11560

-- Define the current age of Karen.
def Karen_age : ℕ := 2

-- Define the current age of Frank given that in 5 years he will be 36 years old.
def Frank_age : ℕ := 36 - 5

-- Define the current age of Ty given that Frank will be 3 times his age in 5 years.
def Ty_age : ℕ := 36 / 3

-- Define Carla's current age given that Ty is currently 4 years more than two times Carla's age.
def Carla_age : ℕ := (Ty_age - 4) / 2

-- Define the difference in age between Carla and Karen.
def Carla_Karen_age_diff : ℕ := Carla_age - Karen_age

-- The statement to be proven.
theorem Carla_is_2_years_older_than_Karen : Carla_Karen_age_diff = 2 := by
  -- The proof is not required, so we use sorry.
  sorry

end NUMINAMATH_GPT_Carla_is_2_years_older_than_Karen_l115_11560


namespace NUMINAMATH_GPT_max_sum_nonneg_l115_11553

theorem max_sum_nonneg (a b c d : ℝ) (h : a + b + c + d = 0) : 
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0 := 
sorry

end NUMINAMATH_GPT_max_sum_nonneg_l115_11553


namespace NUMINAMATH_GPT_find_root_product_l115_11547

theorem find_root_product :
  (∃ r s t : ℝ, (∀ x : ℝ, (x - r) * (x - s) * (x - t) = x^3 - 15 * x^2 + 26 * x - 8) ∧
  (1 + r) * (1 + s) * (1 + t) = 50) :=
sorry

end NUMINAMATH_GPT_find_root_product_l115_11547


namespace NUMINAMATH_GPT_original_price_l115_11500

theorem original_price (sale_price : ℝ) (discount : ℝ) : 
  sale_price = 55 → discount = 0.45 → 
  ∃ (P : ℝ), 0.55 * P = sale_price ∧ P = 100 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l115_11500


namespace NUMINAMATH_GPT_functional_equation_solution_l115_11508

noncomputable def f (x : ℚ) : ℚ := sorry

theorem functional_equation_solution (f : ℚ → ℚ) (f_pos_rat : ∀ x : ℚ, 0 < x → 0 < f x) :
  (∀ x y : ℚ, 0 < x → 0 < y → f x + f y + 2 * x * y * f (x * y) = f (x * y) / f (x + y)) →
  (∀ x : ℚ, 0 < x → f x = 1 / x ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l115_11508


namespace NUMINAMATH_GPT_A_and_B_finish_together_in_11_25_days_l115_11592

theorem A_and_B_finish_together_in_11_25_days (A_rate B_rate : ℝ)
    (hA : A_rate = 1/18) (hB : B_rate = 1/30) :
    1 / (A_rate + B_rate) = 11.25 := by
  sorry

end NUMINAMATH_GPT_A_and_B_finish_together_in_11_25_days_l115_11592


namespace NUMINAMATH_GPT_discounted_price_l115_11566

theorem discounted_price (P : ℝ) (original_price : ℝ) (discount_rate : ℝ)
  (h1 : original_price = 975)
  (h2 : discount_rate = 0.20)
  (h3 : P = original_price - discount_rate * original_price) : 
  P = 780 := 
by
  sorry

end NUMINAMATH_GPT_discounted_price_l115_11566


namespace NUMINAMATH_GPT_factorize_expression_l115_11510

theorem factorize_expression (x a : ℝ) : 4 * x - x * a^2 = x * (2 - a) * (2 + a) :=
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l115_11510


namespace NUMINAMATH_GPT_find_denominator_x_l115_11598

noncomputable def sum_fractions : ℝ := 
    3.0035428163476343

noncomputable def fraction1 (x : ℝ) : ℝ :=
    2007 / x

noncomputable def fraction2 : ℝ :=
    8001 / 5998

noncomputable def fraction3 : ℝ :=
    2001 / 3999

-- Problem statement in Lean
theorem find_denominator_x (x : ℝ) :
  sum_fractions = fraction1 x + fraction2 + fraction3 ↔ x = 1717 :=
by sorry

end NUMINAMATH_GPT_find_denominator_x_l115_11598


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l115_11548

def A : Set ℤ := { -3, -1, 0, 1 }
def B : Set ℤ := { x | (-2 < x) ∧ (x < 1) }

theorem intersection_of_A_and_B : A ∩ B = { -1, 0 } := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l115_11548


namespace NUMINAMATH_GPT_union_complement_l115_11591

universe u

def U : Set ℕ := {0, 2, 4, 6, 8, 10}
def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1}

theorem union_complement (U A B : Set ℕ) (hU : U = {0, 2, 4, 6, 8, 10}) (hA : A = {2, 4, 6}) (hB : B = {1}) :
  (U \ A) ∪ B = {0, 1, 8, 10} :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_union_complement_l115_11591


namespace NUMINAMATH_GPT_Jake_weight_is_118_l115_11588

-- Define the current weights of Jake, his sister, and Mark
variable (J S M : ℕ)

-- Define the given conditions
axiom h1 : J - 12 = 2 * (S + 4)
axiom h2 : M = J + S + 50
axiom h3 : J + S + M = 385

theorem Jake_weight_is_118 : J = 118 :=
by
  sorry

end NUMINAMATH_GPT_Jake_weight_is_118_l115_11588


namespace NUMINAMATH_GPT_my_op_five_four_l115_11542

-- Define the operation a * b
def my_op (a b : ℤ) := a^2 + a * b - b^2

-- Define the theorem to prove 5 * 4 = 29 given the defined operation my_op
theorem my_op_five_four : my_op 5 4 = 29 := 
by 
sorry

end NUMINAMATH_GPT_my_op_five_four_l115_11542


namespace NUMINAMATH_GPT_work_days_B_works_l115_11501

theorem work_days_B_works (x : ℕ) (A_work_rate B_work_rate : ℚ) (A_remaining_days : ℕ) (total_work : ℚ) :
  A_work_rate = (1 / 12) ∧
  B_work_rate = (1 / 15) ∧
  A_remaining_days = 4 ∧
  total_work = 1 →
  x * B_work_rate + A_remaining_days * A_work_rate = total_work →
  x = 10 :=
sorry

end NUMINAMATH_GPT_work_days_B_works_l115_11501


namespace NUMINAMATH_GPT_teacher_work_months_l115_11559

variable (periods_per_day : ℕ) (pay_per_period : ℕ) (days_per_month : ℕ) (total_earnings : ℕ)

def monthly_earnings (periods_per_day : ℕ) (pay_per_period : ℕ) (days_per_month : ℕ) : ℕ :=
  periods_per_day * pay_per_period * days_per_month

def number_of_months_worked (total_earnings : ℕ) (monthly_earnings : ℕ) : ℕ :=
  total_earnings / monthly_earnings

theorem teacher_work_months :
  let periods_per_day := 5
  let pay_per_period := 5
  let days_per_month := 24
  let total_earnings := 3600
  number_of_months_worked total_earnings (monthly_earnings periods_per_day pay_per_period days_per_month) = 6 :=
by
  sorry

end NUMINAMATH_GPT_teacher_work_months_l115_11559


namespace NUMINAMATH_GPT_solve_for_x_l115_11585

variable (x : ℝ)

-- Define the condition: 20% of x = 300
def twenty_percent_eq_300 := (0.20 * x = 300)

-- Define the goal: 120% of x = 1800
def one_twenty_percent_eq_1800 := (1.20 * x = 1800)

theorem solve_for_x (h : twenty_percent_eq_300 x) : one_twenty_percent_eq_1800 x :=
sorry

end NUMINAMATH_GPT_solve_for_x_l115_11585


namespace NUMINAMATH_GPT_sum_of_roots_l115_11584

theorem sum_of_roots (a b c d : ℝ) (h : ∀ x : ℝ, 
  a * (x ^ 3 - x) ^ 3 + b * (x ^ 3 - x) ^ 2 + c * (x ^ 3 - x) + d 
  ≥ a * (x ^ 2 + x + 1) ^ 3 + b * (x ^ 2 + x + 1) ^ 2 + c * (x ^ 2 + x + 1) + d) :
  b / a = -6 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l115_11584


namespace NUMINAMATH_GPT_system_of_equations_correct_l115_11556

theorem system_of_equations_correct (x y : ℝ) (h1 : x + y = 2000) (h2 : y = x * 0.30) :
  x + y = 2000 ∧ y = x * 0.30 :=
by 
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_system_of_equations_correct_l115_11556


namespace NUMINAMATH_GPT_dot_product_EC_ED_l115_11546

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end NUMINAMATH_GPT_dot_product_EC_ED_l115_11546


namespace NUMINAMATH_GPT_find_k_l115_11504

theorem find_k (k : ℝ) (h : (3 : ℝ)^2 - k * (3 : ℝ) - 6 = 0) : k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l115_11504


namespace NUMINAMATH_GPT_max_cube_side_length_max_parallelepiped_dimensions_l115_11512

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end NUMINAMATH_GPT_max_cube_side_length_max_parallelepiped_dimensions_l115_11512


namespace NUMINAMATH_GPT_original_number_is_0_2_l115_11530

theorem original_number_is_0_2 :
  ∃ x : ℝ, (1 / (1 / x - 1) - 1 = -0.75) ∧ x = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_0_2_l115_11530


namespace NUMINAMATH_GPT_sugar_for_third_layer_l115_11514

theorem sugar_for_third_layer (s1 : ℕ) (s2 : ℕ) (s3 : ℕ) 
  (h1 : s1 = 2) 
  (h2 : s2 = 2 * s1) 
  (h3 : s3 = 3 * s2) : 
  s3 = 12 := 
sorry

end NUMINAMATH_GPT_sugar_for_third_layer_l115_11514


namespace NUMINAMATH_GPT_least_positive_integer_l115_11541

noncomputable def hasProperty (x n d p : ℕ) : Prop :=
  x = 10^p * d + n ∧ n = x / 19

theorem least_positive_integer : 
  ∃ (x n d p : ℕ), hasProperty x n d p ∧ x = 950 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_l115_11541


namespace NUMINAMATH_GPT_maynard_dog_holes_l115_11527

open Real

theorem maynard_dog_holes (h_filled : ℝ) (h_unfilled : ℝ) (percent_filled : ℝ) 
  (percent_unfilled : ℝ) (total_holes : ℝ) :
  percent_filled = 0.75 →
  percent_unfilled = 0.25 →
  h_unfilled = 2 →
  h_filled = total_holes * percent_filled →
  total_holes = 8 :=
by
  intros hf pu hu hf_total
  sorry

end NUMINAMATH_GPT_maynard_dog_holes_l115_11527


namespace NUMINAMATH_GPT_base7_digit_divisibility_l115_11581

-- Define base-7 digit integers
notation "digit" => Fin 7

-- Define conversion from base-7 to base-10 for the form 3dd6_7
def base7_to_base10 (d : digit) : ℤ := 3 * (7^3) + (d:ℤ) * (7^2) + (d:ℤ) * 7 + 6

-- Define the property of being divisible by 13
def is_divisible_by_13 (n : ℤ) : Prop := ∃ k : ℤ, n = 13 * k

-- Formalize the theorem
theorem base7_digit_divisibility (d : digit) :
  is_divisible_by_13 (base7_to_base10 d) ↔ d = 4 :=
sorry

end NUMINAMATH_GPT_base7_digit_divisibility_l115_11581


namespace NUMINAMATH_GPT_first_term_of_geometric_series_l115_11562

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_series_l115_11562


namespace NUMINAMATH_GPT_rhombus_area_correct_l115_11536

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_correct (x : ℝ) (h1 : rhombus_area 7 (abs (8 - x)) = 56) 
    (h2 : x ≠ 8) : x = -8 ∨ x = 24 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_correct_l115_11536


namespace NUMINAMATH_GPT_jovana_shells_l115_11583

def initial_weight : ℕ := 5
def added_weight : ℕ := 23
def total_weight : ℕ := 28

theorem jovana_shells :
  initial_weight + added_weight = total_weight :=
by
  sorry

end NUMINAMATH_GPT_jovana_shells_l115_11583


namespace NUMINAMATH_GPT_inequality_proof_l115_11563

theorem inequality_proof
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : c^2 + a * b = a^2 + b^2) :
  c^2 + a * b ≤ a * c + b * c := sorry

end NUMINAMATH_GPT_inequality_proof_l115_11563


namespace NUMINAMATH_GPT_axis_of_symmetry_and_vertex_l115_11570

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

theorem axis_of_symmetry_and_vertex :
  (∃ (a : ℝ), (f a = -2 * (a - 1)^2 + 3) ∧ a = 1) ∧ ∃ v, (v = (1, 3) ∧ ∀ x, f x = -2 * (x - 1)^2 + 3) :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_and_vertex_l115_11570


namespace NUMINAMATH_GPT_fewest_presses_to_original_l115_11545

theorem fewest_presses_to_original (x : ℝ) (hx : x = 16) (f : ℝ → ℝ)
    (hf : ∀ y : ℝ, f y = 1 / y) : (f (f x)) = x :=
by
  sorry

end NUMINAMATH_GPT_fewest_presses_to_original_l115_11545


namespace NUMINAMATH_GPT_solve_for_x_l115_11507

theorem solve_for_x : ∃ x : ℝ, 5 * x + 9 * x = 570 - 12 * (x - 5) ∧ x = 315 / 13 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l115_11507


namespace NUMINAMATH_GPT_meal_cost_is_25_l115_11577

def total_cost_samosas : ℕ := 3 * 2
def total_cost_pakoras : ℕ := 4 * 3
def cost_mango_lassi : ℕ := 2
def tip_percentage : ℝ := 0.25

def total_food_cost : ℕ := total_cost_samosas + total_cost_pakoras + cost_mango_lassi
def tip_amount : ℝ := total_food_cost * tip_percentage
def total_meal_cost : ℝ := total_food_cost + tip_amount

theorem meal_cost_is_25 : total_meal_cost = 25 := by
    sorry

end NUMINAMATH_GPT_meal_cost_is_25_l115_11577


namespace NUMINAMATH_GPT_number_of_space_diagonals_l115_11589

theorem number_of_space_diagonals (V E F tF qF : ℕ)
    (hV : V = 30) (hE : E = 72) (hF : F = 44) (htF : tF = 34) (hqF : qF = 10) : 
    V * (V - 1) / 2 - E - qF * 2 = 343 :=
by
  sorry

end NUMINAMATH_GPT_number_of_space_diagonals_l115_11589


namespace NUMINAMATH_GPT_percentage_of_life_in_accounting_jobs_l115_11578

-- Define the conditions
def years_as_accountant : ℕ := 25
def years_as_manager : ℕ := 15
def lifespan : ℕ := 80

-- Define the proof problem statement
theorem percentage_of_life_in_accounting_jobs :
  (years_as_accountant + years_as_manager) / lifespan * 100 = 50 := 
by sorry

end NUMINAMATH_GPT_percentage_of_life_in_accounting_jobs_l115_11578


namespace NUMINAMATH_GPT_ratio_of_customers_third_week_l115_11576

def ratio_of_customers (c1 c3 : ℕ) (s k t : ℕ) : Prop := s = 500 ∧ k = 50 ∧ t = 760 ∧ c1 = 35 ∧ c3 = 105 ∧ (t - s - k) - (35 + 70) = c1 ∧ c3 = 105 ∧ (c3 / c1 = 3)

theorem ratio_of_customers_third_week (c1 c3 : ℕ) (s k t : ℕ)
  (h1 : s = 500)
  (h2 : k = 50)
  (h3 : t = 760)
  (h4 : c1 = 35)
  (h5 : c3 = 105)
  (h6 : (t - s - k) - (35 + 70) = c1)
  (h7 : c3 = 105) :
  (c3 / c1) = 3 :=
  sorry

end NUMINAMATH_GPT_ratio_of_customers_third_week_l115_11576


namespace NUMINAMATH_GPT_how_much_milk_did_joey_drink_l115_11565

theorem how_much_milk_did_joey_drink (initial_milk : ℚ) (rachel_fraction : ℚ) (joey_fraction : ℚ) :
  initial_milk = 3/4 →
  rachel_fraction = 1/3 →
  joey_fraction = 1/2 →
  joey_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/4 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_how_much_milk_did_joey_drink_l115_11565


namespace NUMINAMATH_GPT_probability_of_same_color_is_correct_l115_11582

def probability_same_color (blue_balls yellow_balls : ℕ) : ℚ :=
  let total_balls := blue_balls + yellow_balls
  let prob_blue := (blue_balls / total_balls : ℚ)
  let prob_yellow := (yellow_balls / total_balls : ℚ)
  (prob_blue ^ 2) + (prob_yellow ^ 2)

theorem probability_of_same_color_is_correct :
  probability_same_color 8 5 = 89 / 169 :=
by 
  sorry

end NUMINAMATH_GPT_probability_of_same_color_is_correct_l115_11582


namespace NUMINAMATH_GPT_range_of_a_l115_11580

noncomputable def f (a x : ℝ) := a * x - x^2 - Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 2*x₁*x₁ - a*x₁ + 1 = 0 ∧ 
  2*x₂*x₂ - a*x₂ + 1 = 0 ∧ f a x₁ + f a x₂ ≥ 4 + Real.log 2) ↔ 
  a ∈ Set.Ici (2 * Real.sqrt 3) := 
sorry

end NUMINAMATH_GPT_range_of_a_l115_11580


namespace NUMINAMATH_GPT_transform_negation_l115_11552

-- Define the terms a, b, and c as real numbers
variables (a b c : ℝ)

-- State the theorem we want to prove
theorem transform_negation (a b c : ℝ) : 
  - (a - b + c) = -a + b - c :=
sorry

end NUMINAMATH_GPT_transform_negation_l115_11552


namespace NUMINAMATH_GPT_beatrice_tv_ratio_l115_11574

theorem beatrice_tv_ratio (T1 T2 T Ttotal : ℕ)
  (h1 : T1 = 8)
  (h2 : T2 = 10)
  (h_total : Ttotal = 42)
  (h_T : T = Ttotal - T1 - T2) :
  (T / gcd T T1, T1 / gcd T T1) = (3, 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_beatrice_tv_ratio_l115_11574


namespace NUMINAMATH_GPT_polynomial_divisible_l115_11523

theorem polynomial_divisible (a b c : ℕ) :
  (X^(3 * a) + X^(3 * b + 1) + X^(3 * c + 2)) % (X^2 + X + 1) = 0 :=
by sorry

end NUMINAMATH_GPT_polynomial_divisible_l115_11523


namespace NUMINAMATH_GPT_sum_of_perimeters_l115_11506

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) : 
  4 * (Real.sqrt 63 + Real.sqrt 22) = 4 * x + 4 * y := by
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_l115_11506


namespace NUMINAMATH_GPT_average_after_15th_inning_l115_11531

theorem average_after_15th_inning (A : ℝ) 
    (h_avg_increase : (14 * A + 75) = 15 * (A + 3)) : 
    A + 3 = 33 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_after_15th_inning_l115_11531


namespace NUMINAMATH_GPT_find_m_plus_n_l115_11509

theorem find_m_plus_n (m n : ℤ) 
  (H1 : (x^3 + m*x + n) * (x^2 - 3*x + 1) ≠ 1 * x^2 + 1 * x^3) 
  (H2 : (x^3 + m*x + n) * (x^2 - 3*x + 1) ≠ 1 * x^2 + 1 * x^3) : 
  m + n = -4 := 
by
  sorry

end NUMINAMATH_GPT_find_m_plus_n_l115_11509


namespace NUMINAMATH_GPT_arman_hourly_rate_increase_l115_11516

theorem arman_hourly_rate_increase :
  let last_week_hours := 35
  let last_week_rate := 10
  let this_week_hours := 40
  let total_payment := 770
  let last_week_earnings := last_week_hours * last_week_rate
  let this_week_earnings := total_payment - last_week_earnings
  let this_week_rate := this_week_earnings / this_week_hours
  let rate_increase := this_week_rate - last_week_rate
  rate_increase = 0.50 :=
by {
  sorry
}

end NUMINAMATH_GPT_arman_hourly_rate_increase_l115_11516


namespace NUMINAMATH_GPT_expression_equals_100_l115_11537

-- Define the terms in the numerator and their squares
def num1 := 0.02
def num2 := 0.52
def num3 := 0.035

def num1_sq := num1^2
def num2_sq := num2^2
def num3_sq := num3^2

-- Define the terms in the denominator and their squares
def denom1 := 0.002
def denom2 := 0.052
def denom3 := 0.0035

def denom1_sq := denom1^2
def denom2_sq := denom2^2
def denom3_sq := denom3^2

-- Define the sums of the squares
def sum_numerator := num1_sq + num2_sq + num3_sq
def sum_denominator := denom1_sq + denom2_sq + denom3_sq

-- Define the final expression
def expression := sum_numerator / sum_denominator

-- Prove the expression equals the correct answer
theorem expression_equals_100 : expression = 100 := by sorry

end NUMINAMATH_GPT_expression_equals_100_l115_11537


namespace NUMINAMATH_GPT_find_sides_from_diagonals_l115_11550

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end NUMINAMATH_GPT_find_sides_from_diagonals_l115_11550


namespace NUMINAMATH_GPT_length_of_tangent_l115_11539

/-- 
Let O and O1 be the centers of the larger and smaller circles respectively with radii 8 and 3. 
The circles touch each other internally. Let A be the point of tangency and OM be the tangent from center O to the smaller circle. 
Prove that the length of this tangent is 4.
--/
theorem length_of_tangent {O O1 : Type} (radius_large : ℝ) (radius_small : ℝ) (OO1 : ℝ) 
  (OM O1M : ℝ) (h : 8 - 3 = 5) (h1 : OO1 = 5) (h2 : O1M = 3): OM = 4 :=
by
  sorry

end NUMINAMATH_GPT_length_of_tangent_l115_11539


namespace NUMINAMATH_GPT_fraction_saved_l115_11502

variable {P : ℝ} (hP : P > 0)

theorem fraction_saved (f : ℝ) (hf0 : 0 ≤ f) (hf1 : f ≤ 1) (condition : 12 * f * P = 4 * (1 - f) * P) : f = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_saved_l115_11502


namespace NUMINAMATH_GPT_prove_x_ge_neg_one_sixth_l115_11551

variable (x y : ℝ)

theorem prove_x_ge_neg_one_sixth (h : x^4 * y^2 + y^4 + 2 * x^3 * y + 6 * x^2 * y + x^2 + 8 ≤ 0) :
  x ≥ -1 / 6 :=
sorry

end NUMINAMATH_GPT_prove_x_ge_neg_one_sixth_l115_11551


namespace NUMINAMATH_GPT_same_terminal_side_l115_11568

theorem same_terminal_side (k : ℤ) : 
  ((2 * k + 1) * 180) % 360 = ((4 * k + 1) * 180) % 360 ∨ ((2 * k + 1) * 180) % 360 = ((4 * k - 1) * 180) % 360 := 
sorry

end NUMINAMATH_GPT_same_terminal_side_l115_11568


namespace NUMINAMATH_GPT_jan_total_skips_l115_11538

def jan_initial_speed : ℕ := 70
def jan_training_factor : ℕ := 2
def jan_skipping_time : ℕ := 5

theorem jan_total_skips :
  (jan_initial_speed * jan_training_factor) * jan_skipping_time = 700 := by
  sorry

end NUMINAMATH_GPT_jan_total_skips_l115_11538


namespace NUMINAMATH_GPT_horner_eval_at_minus_point_two_l115_11590

def f (x : ℝ) : ℝ :=
  1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem horner_eval_at_minus_point_two :
  f (-0.2) = 0.81873 :=
by 
  sorry

end NUMINAMATH_GPT_horner_eval_at_minus_point_two_l115_11590


namespace NUMINAMATH_GPT_population_multiple_of_18_l115_11515

theorem population_multiple_of_18
  (a b c P : ℕ)
  (ha : P = a^2)
  (hb : P + 200 = b^2 + 1)
  (hc : b^2 + 301 = c^2) :
  ∃ k, P = 18 * k := 
sorry

end NUMINAMATH_GPT_population_multiple_of_18_l115_11515


namespace NUMINAMATH_GPT_find_angle_l115_11534

theorem find_angle (r1 r2 : ℝ) (h_r1 : r1 = 1) (h_r2 : r2 = 2) 
(h_shaded : ∀ α : ℝ, 0 < α ∧ α < 2 * π → 
  (360 / 360 * pi * r1^2 + (α / (2 * π)) * pi * r2^2 - (α / (2 * π)) * pi * r1^2 = (1/3) * (pi * r2^2))) : 
  (∀ α : ℝ, 0 < α ∧ α < 2 * π ↔ 
  α = π / 3 ) :=
by
  sorry

end NUMINAMATH_GPT_find_angle_l115_11534


namespace NUMINAMATH_GPT_equation_of_hyperbola_l115_11587

variable (a b c : ℝ)
variable (x y : ℝ)

theorem equation_of_hyperbola :
  (0 < a) ∧ (0 < b) ∧ (c / a = Real.sqrt 3) ∧ (a^2 / c = 1) ∧ (c = 3) ∧ (b = Real.sqrt 6)
  → (x^2 / 3 - y^2 / 6 = 1) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_hyperbola_l115_11587


namespace NUMINAMATH_GPT_all_sets_B_l115_11572

open Set

theorem all_sets_B (B : Set ℕ) :
  { B | {1, 2} ∪ B = {1, 2, 3} } =
  ({ {3}, {1, 3}, {2, 3}, {1, 2, 3} } : Set (Set ℕ)) :=
sorry

end NUMINAMATH_GPT_all_sets_B_l115_11572


namespace NUMINAMATH_GPT_product_gcd_lcm_15_9_l115_11505

theorem product_gcd_lcm_15_9 : Nat.gcd 15 9 * Nat.lcm 15 9 = 135 := 
by
  -- skipping proof as instructed
  sorry

end NUMINAMATH_GPT_product_gcd_lcm_15_9_l115_11505


namespace NUMINAMATH_GPT_rose_days_to_complete_work_l115_11573

theorem rose_days_to_complete_work (R : ℝ) (h1 : 1 / 10 + 1 / R = 1 / 8) : R = 40 := 
sorry

end NUMINAMATH_GPT_rose_days_to_complete_work_l115_11573


namespace NUMINAMATH_GPT_probability_at_least_one_red_l115_11524

def total_balls : ℕ := 6
def red_balls : ℕ := 4
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_at_least_one_red :
  (choose_two red_balls + red_balls * (total_balls - red_balls - 1) / 2) / choose_two total_balls = 14 / 15 :=
sorry

end NUMINAMATH_GPT_probability_at_least_one_red_l115_11524


namespace NUMINAMATH_GPT_people_dislike_both_radio_and_music_l115_11579

theorem people_dislike_both_radio_and_music (N : ℕ) (p_r p_rm : ℝ) (hN : N = 2000) (hp_r : p_r = 0.25) (hp_rm : p_rm = 0.15) : 
  N * p_r * p_rm = 75 :=
by {
  sorry
}

end NUMINAMATH_GPT_people_dislike_both_radio_and_music_l115_11579


namespace NUMINAMATH_GPT_hajar_score_l115_11519

variables (F H : ℕ)

theorem hajar_score 
  (h1 : F - H = 21)
  (h2 : F + H = 69)
  (h3 : F > H) :
  H = 24 :=
sorry

end NUMINAMATH_GPT_hajar_score_l115_11519


namespace NUMINAMATH_GPT_number_of_intersections_is_four_l115_11554

def LineA (x y : ℝ) : Prop := 3 * x - 2 * y + 4 = 0
def LineB (x y : ℝ) : Prop := 6 * x + 4 * y - 12 = 0
def LineC (x y : ℝ) : Prop := x - y + 1 = 0
def LineD (x y : ℝ) : Prop := y - 2 = 0

def is_intersection (L1 L2 : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := L1 p.1 p.2 ∧ L2 p.1 p.2

theorem number_of_intersections_is_four :
  (∃ p1 : ℝ × ℝ, is_intersection LineA LineB p1) ∧
  (∃ p2 : ℝ × ℝ, is_intersection LineC LineD p2) ∧
  (∃ p3 : ℝ × ℝ, is_intersection LineA LineD p3) ∧
  (∃ p4 : ℝ × ℝ, is_intersection LineB LineD p4) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :=
by
  sorry

end NUMINAMATH_GPT_number_of_intersections_is_four_l115_11554


namespace NUMINAMATH_GPT_range_of_m_l115_11558

noncomputable def problem (x m : ℝ) (p q : Prop) : Prop :=
  (¬ p → ¬ q) ∧ (¬ q → ¬ p → False) ∧ (p ↔ |1 - (x - 1) / 3| ≤ 2) ∧ 
  (q ↔ x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0)

theorem range_of_m (m : ℝ) (x : ℝ) (p q : Prop) 
  (h : problem x m p q) : m ≥ 9 :=
sorry

end NUMINAMATH_GPT_range_of_m_l115_11558


namespace NUMINAMATH_GPT_measure_angle_BRC_l115_11513

inductive Point : Type
| A 
| B 
| C 
| P 
| Q 
| R 

open Point

def is_inside_triangle (P : Point) (A B C : Point) : Prop := sorry

def intersection (a b c : Point) : Point := sorry

def length (a b : Point) : ℝ := sorry

def angle (a b c : Point) : ℝ := sorry

theorem measure_angle_BRC 
  (P : Point) (A B C : Point)
  (h_inside : is_inside_triangle P A B C)
  (hQ : Q = intersection A C P)
  (hR : R = intersection A B P)
  (h_lengths_equal : length A R = length R B ∧ length R B = length C P)
  (h_CQ_PQ : length C Q = length P Q) :
  angle B R C = 120 := 
sorry

end NUMINAMATH_GPT_measure_angle_BRC_l115_11513


namespace NUMINAMATH_GPT_five_year_salary_increase_l115_11511

noncomputable def salary_growth (S : ℝ) := S * (1.08)^5

theorem five_year_salary_increase (S : ℝ) : 
  salary_growth S = S * 1.4693 := 
sorry

end NUMINAMATH_GPT_five_year_salary_increase_l115_11511


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l115_11517

def A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def CR_A : Set ℝ := { x | x < 3 ∨ x > 7 }

theorem problem_part1 : A ∪ B = { x | 3 ≤ x ∧ x ≤ 7 } := by
  sorry

theorem problem_part2 : (CR_A ∩ B) = { x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10) } := by
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l115_11517


namespace NUMINAMATH_GPT_construct_triangle_l115_11520

variables (a : ℝ) (α : ℝ) (d : ℝ)

-- Helper definitions
def is_triangle_valid (a α d : ℝ) : Prop := sorry

-- The theorem to be proven
theorem construct_triangle (a α d : ℝ) : is_triangle_valid a α d :=
sorry

end NUMINAMATH_GPT_construct_triangle_l115_11520
