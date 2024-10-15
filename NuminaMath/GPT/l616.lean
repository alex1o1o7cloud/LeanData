import Mathlib

namespace NUMINAMATH_GPT_sin_add_arcsin_arctan_l616_61654

theorem sin_add_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (Real.sqrt 3)
  Real.sin (a + b) = (2 + 3 * Real.sqrt 3) / 10 :=
by
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (Real.sqrt 3)
  sorry

end NUMINAMATH_GPT_sin_add_arcsin_arctan_l616_61654


namespace NUMINAMATH_GPT_eval_expr_l616_61657

theorem eval_expr : 4 * (-3) + 60 / (-15) = -16 := by
  sorry

end NUMINAMATH_GPT_eval_expr_l616_61657


namespace NUMINAMATH_GPT_polygon_interior_exterior_equal_l616_61672

theorem polygon_interior_exterior_equal (n : ℕ) :
  (n - 2) * 180 = 360 → n = 4 :=
by
  sorry

end NUMINAMATH_GPT_polygon_interior_exterior_equal_l616_61672


namespace NUMINAMATH_GPT_Warriors_won_25_games_l616_61626

def CricketResults (Sharks Falcons Warriors Foxes Knights : ℕ) :=
  Sharks > Falcons ∧
  (Warriors > Foxes ∧ Warriors < Knights) ∧
  Foxes > 15 ∧
  (Foxes = 20 ∨ Foxes = 25 ∨ Foxes = 30) ∧
  (Warriors = 20 ∨ Warriors = 25 ∨ Warriors = 30) ∧
  (Knights = 20 ∨ Knights = 25 ∨ Knights = 30)

theorem Warriors_won_25_games (Sharks Falcons Warriors Foxes Knights : ℕ) 
  (h : CricketResults Sharks Falcons Warriors Foxes Knights) :
  Warriors = 25 :=
by
  sorry

end NUMINAMATH_GPT_Warriors_won_25_games_l616_61626


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l616_61615

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 3 ≥ 0} = {x : ℝ | x ≤ -1 ∨ x ≥ 3 / 2} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l616_61615


namespace NUMINAMATH_GPT_balloon_descent_rate_l616_61699

theorem balloon_descent_rate (D : ℕ) 
    (rate_of_ascent : ℕ := 50) 
    (time_chain_pulled_1 : ℕ := 15) 
    (time_chain_pulled_2 : ℕ := 15) 
    (time_chain_released_1 : ℕ := 10) 
    (highest_elevation : ℕ := 1400) :
    (time_chain_pulled_1 + time_chain_pulled_2) * rate_of_ascent - time_chain_released_1 * D = highest_elevation 
    → D = 10 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_balloon_descent_rate_l616_61699


namespace NUMINAMATH_GPT_exists_integers_m_n_l616_61632

theorem exists_integers_m_n (x y : ℝ) (hxy : x ≠ y) : 
  ∃ (m n : ℤ), (m * x + n * y > 0) ∧ (n * x + m * y < 0) :=
sorry

end NUMINAMATH_GPT_exists_integers_m_n_l616_61632


namespace NUMINAMATH_GPT_upper_limit_for_y_l616_61683

theorem upper_limit_for_y (x y : ℝ) (hx : 5 < x) (hx' : x < 8) (hy : 8 < y) (h_diff : y - x = 7) : y ≤ 14 :=
by
  sorry

end NUMINAMATH_GPT_upper_limit_for_y_l616_61683


namespace NUMINAMATH_GPT_determine_l_l616_61630

theorem determine_l :
  ∃ l : ℤ, (2^2000 - 2^1999 - 3 * 2^1998 + 2^1997 = l * 2^1997) ∧ l = -1 :=
by
  sorry

end NUMINAMATH_GPT_determine_l_l616_61630


namespace NUMINAMATH_GPT_common_rational_root_is_neg_one_third_l616_61648

theorem common_rational_root_is_neg_one_third (a b c d e f g : ℚ) :
  ∃ k : ℚ, (75 * k^4 + a * k^3 + b * k^2 + c * k + 12 = 0) ∧
           (12 * k^5 + d * k^4 + e * k^3 + f * k^2 + g * k + 75 = 0) ∧
           (¬ k.isInt) ∧ (k < 0) ∧ (k = -1/3) :=
sorry

end NUMINAMATH_GPT_common_rational_root_is_neg_one_third_l616_61648


namespace NUMINAMATH_GPT_percent_defective_units_shipped_l616_61625

theorem percent_defective_units_shipped :
  let total_units_defective := 6 / 100
  let defective_units_shipped := 4 / 100
  let percent_defective_units_shipped := (total_units_defective * defective_units_shipped) * 100
  percent_defective_units_shipped = 0.24 := by
  sorry

end NUMINAMATH_GPT_percent_defective_units_shipped_l616_61625


namespace NUMINAMATH_GPT_smallest_mu_real_number_l616_61693

theorem smallest_mu_real_number (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) :
  a^2 + b^2 + c^2 + d^2 ≤ ab + (3/2) * bc + cd :=
sorry

end NUMINAMATH_GPT_smallest_mu_real_number_l616_61693


namespace NUMINAMATH_GPT_curve_is_circle_l616_61628

-- Definition of the curve in polar coordinates
def curve (r θ : ℝ) : Prop :=
  r = 3 * Real.sin θ

-- The theorem to prove
theorem curve_is_circle : ∀ θ : ℝ, ∃ r : ℝ, curve r θ → (∃ c : ℝ × ℝ, ∃ R : ℝ, ∀ p : ℝ × ℝ, (Real.sqrt ((p.1 - c.1) ^ 2 + (p.2 - c.2) ^ 2) = R)) :=
by
  sorry

end NUMINAMATH_GPT_curve_is_circle_l616_61628


namespace NUMINAMATH_GPT_cost_of_45_lilies_l616_61612

-- Definitions of the given conditions
def cost_per_lily := 30 / 18
def lilies_18_bouquet_cost := 30
def number_of_lilies_in_bouquet := 45

-- Theorem stating the mathematical proof problem
theorem cost_of_45_lilies : cost_per_lily * number_of_lilies_in_bouquet = 75 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_cost_of_45_lilies_l616_61612


namespace NUMINAMATH_GPT_quadratic_nonneg_for_all_t_l616_61688

theorem quadratic_nonneg_for_all_t (x y : ℝ) : 
  (y ≤ x + 1) → (y ≥ -x - 1) → (x ≥ y^2 / 4) → (∀ (t : ℝ), (|t| ≤ 1) → t^2 + y * t + x ≥ 0) :=
by
  intro h1 h2 h3 t ht
  sorry

end NUMINAMATH_GPT_quadratic_nonneg_for_all_t_l616_61688


namespace NUMINAMATH_GPT_factorize_expression_l616_61675

theorem factorize_expression (a : ℝ) : 3 * a^2 + 6 * a + 3 = 3 * (a + 1)^2 := 
by sorry

end NUMINAMATH_GPT_factorize_expression_l616_61675


namespace NUMINAMATH_GPT_find_difference_l616_61605

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_difference_l616_61605


namespace NUMINAMATH_GPT_minimum_value_l616_61697

open Real

theorem minimum_value (x : ℝ) (h : 0 < x) : 
  ∃ y, (∀ z > 0, 3 * sqrt z + 2 / z ≥ y) ∧ y = 5 := by
  sorry

end NUMINAMATH_GPT_minimum_value_l616_61697


namespace NUMINAMATH_GPT_part_a_l616_61650

theorem part_a (n : ℕ) : ((x^2 + x + 1) ∣ (x^(2 * n) + x^n + 1)) ↔ (n % 3 = 0) := sorry

end NUMINAMATH_GPT_part_a_l616_61650


namespace NUMINAMATH_GPT_typing_time_l616_61692

def original_speed : ℕ := 212
def reduction : ℕ := 40
def new_speed : ℕ := original_speed - reduction
def document_length : ℕ := 3440
def required_time : ℕ := 20

theorem typing_time :
  document_length / new_speed = required_time :=
by
  sorry

end NUMINAMATH_GPT_typing_time_l616_61692


namespace NUMINAMATH_GPT_apples_difference_l616_61652

-- Definitions based on conditions
def JackiesApples : Nat := 10
def AdamsApples : Nat := 8

-- Statement
theorem apples_difference : JackiesApples - AdamsApples = 2 := by
  sorry

end NUMINAMATH_GPT_apples_difference_l616_61652


namespace NUMINAMATH_GPT_bottles_needed_to_fill_large_bottle_l616_61629

def medium_bottle_ml : ℕ := 150
def large_bottle_ml : ℕ := 1200

theorem bottles_needed_to_fill_large_bottle : large_bottle_ml / medium_bottle_ml = 8 :=
by
  sorry

end NUMINAMATH_GPT_bottles_needed_to_fill_large_bottle_l616_61629


namespace NUMINAMATH_GPT_tan_of_alpha_l616_61671

theorem tan_of_alpha 
  (α : ℝ)
  (h1 : Real.sin α = (3 / 5))
  (h2 : α ∈ Set.Ioo (π / 2) π) : Real.tan α = -3 / 4 :=
sorry

end NUMINAMATH_GPT_tan_of_alpha_l616_61671


namespace NUMINAMATH_GPT_unique_function_satisfying_condition_l616_61621

theorem unique_function_satisfying_condition :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) ↔ f = id :=
sorry

end NUMINAMATH_GPT_unique_function_satisfying_condition_l616_61621


namespace NUMINAMATH_GPT_min_value_l616_61602

-- Definition of the conditions
def positive (a : ℝ) : Prop := a > 0

theorem min_value (a : ℝ) (h : positive a) : 
  ∃ m : ℝ, (m = 2 * Real.sqrt 6) ∧ (∀ x : ℝ, positive x → (3 / (2 * x) + 4 * x) ≥ m) :=
sorry

end NUMINAMATH_GPT_min_value_l616_61602


namespace NUMINAMATH_GPT_find_a_l616_61600

theorem find_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x^2 - 2*a*x - 8*(a^2) < 0) (h3 : x2 - x1 = 15) : a = 5 / 2 :=
by
  -- Sorry is used to skip the actual proof.
  sorry

end NUMINAMATH_GPT_find_a_l616_61600


namespace NUMINAMATH_GPT_minute_hand_gains_per_hour_l616_61642

theorem minute_hand_gains_per_hour (total_gain : ℕ) (total_hours : ℕ) (gain_by_6pm : total_gain = 63) (hours_from_9_to_6 : total_hours = 9) : (total_gain / total_hours) = 7 :=
by
  -- The proof is not required as per instruction.
  sorry

end NUMINAMATH_GPT_minute_hand_gains_per_hour_l616_61642


namespace NUMINAMATH_GPT_radiator_water_fraction_l616_61685

noncomputable def fraction_of_water_after_replacements (initial_water : ℚ) (initial_antifreeze : ℚ) (removal_fraction : ℚ)
  (num_replacements : ℕ) : ℚ :=
  initial_water * (removal_fraction ^ num_replacements)

theorem radiator_water_fraction :
  let initial_water := 10
  let initial_antifreeze := 10
  let total_volume := 20
  let removal_volume := 5
  let removal_fraction := 3 / 4
  let num_replacements := 4
  fraction_of_water_after_replacements initial_water initial_antifreeze removal_fraction num_replacements / total_volume = 0.158 := 
sorry

end NUMINAMATH_GPT_radiator_water_fraction_l616_61685


namespace NUMINAMATH_GPT_initial_investors_and_contribution_l616_61638

theorem initial_investors_and_contribution :
  ∃ (x y : ℕ), 
    (x - 10) * (y + 1) = x * y ∧
    (x - 25) * (y + 3) = x * y ∧
    x = 100 ∧ 
    y = 9 :=
by
  sorry

end NUMINAMATH_GPT_initial_investors_and_contribution_l616_61638


namespace NUMINAMATH_GPT_amusement_park_admission_l616_61663

def number_of_children (children_fee : ℤ) (adults_fee : ℤ) (total_people : ℤ) (total_fees : ℤ) : ℤ :=
  let y := (total_fees - total_people * children_fee) / (adults_fee - children_fee)
  total_people - y

theorem amusement_park_admission :
  number_of_children 15 40 315 8100 = 180 :=
by
  -- Fees in cents to avoid decimals
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_amusement_park_admission_l616_61663


namespace NUMINAMATH_GPT_candy_store_price_per_pound_fudge_l616_61676

theorem candy_store_price_per_pound_fudge 
  (fudge_pounds : ℕ)
  (truffles_dozen : ℕ)
  (truffles_price_each : ℝ)
  (pretzels_dozen : ℕ)
  (pretzels_price_each : ℝ)
  (total_revenue : ℝ) 
  (truffles_total : ℕ := truffles_dozen * 12)
  (pretzels_total : ℕ := pretzels_dozen * 12)
  (truffles_revenue : ℝ := truffles_total * truffles_price_each)
  (pretzels_revenue : ℝ := pretzels_total * pretzels_price_each)
  (fudge_revenue : ℝ := total_revenue - (truffles_revenue + pretzels_revenue))
  (fudge_price_per_pound : ℝ := fudge_revenue / fudge_pounds) :
  fudge_pounds = 20 →
  truffles_dozen = 5 →
  truffles_price_each = 1.50 →
  pretzels_dozen = 3 →
  pretzels_price_each = 2.00 →
  total_revenue = 212 →
  fudge_price_per_pound = 2.5 :=
by 
  sorry

end NUMINAMATH_GPT_candy_store_price_per_pound_fudge_l616_61676


namespace NUMINAMATH_GPT_abigail_time_to_finish_l616_61684

noncomputable def words_total : ℕ := 1000
noncomputable def words_per_30_min : ℕ := 300
noncomputable def words_already_written : ℕ := 200
noncomputable def time_per_word : ℝ := 30 / words_per_30_min

theorem abigail_time_to_finish :
  (words_total - words_already_written) * time_per_word = 80 :=
by
  sorry

end NUMINAMATH_GPT_abigail_time_to_finish_l616_61684


namespace NUMINAMATH_GPT_average_age_when_youngest_born_l616_61681

theorem average_age_when_youngest_born (n : ℕ) (avg_age current_y : ℕ) (total_yr : ℕ) (reduction_yr yr_older : ℕ) (avg_age_older : ℕ) 
  (h1 : n = 7)
  (h2 : avg_age = 30)
  (h3 : current_y = 7)
  (h4 : total_yr = n * avg_age)
  (h5 : reduction_yr = (n - 1) * current_y)
  (h6 : yr_older = total_yr - reduction_yr)
  (h7 : avg_age_older = yr_older / (n - 1)) :
  avg_age_older = 28 :=
by 
  sorry

end NUMINAMATH_GPT_average_age_when_youngest_born_l616_61681


namespace NUMINAMATH_GPT_smallest_positive_expr_l616_61643

theorem smallest_positive_expr (m n : ℤ) : ∃ (m n : ℤ), 216 * m + 493 * n = 1 := 
sorry

end NUMINAMATH_GPT_smallest_positive_expr_l616_61643


namespace NUMINAMATH_GPT_parabolas_intersect_at_point_l616_61674

theorem parabolas_intersect_at_point :
  ∀ (p q : ℝ), p + q = 2019 → (1 : ℝ)^2 + (p : ℝ) * 1 + q = 2020 :=
by
  intros p q h
  sorry

end NUMINAMATH_GPT_parabolas_intersect_at_point_l616_61674


namespace NUMINAMATH_GPT_polygon_sidedness_l616_61653

-- Define the condition: the sum of the interior angles of the polygon
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Given condition
def given_condition : ℝ := 1260

-- Target proposition to prove
theorem polygon_sidedness (n : ℕ) (h : sum_of_interior_angles n = given_condition) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sidedness_l616_61653


namespace NUMINAMATH_GPT_number_of_paths_l616_61618

/-
We need to define the conditions and the main theorem
-/

def grid_width : ℕ := 5
def grid_height : ℕ := 4
def total_steps : ℕ := 8
def steps_right : ℕ := 5
def steps_up : ℕ := 3

theorem number_of_paths : (Nat.choose total_steps steps_up) = 56 := by
  sorry

end NUMINAMATH_GPT_number_of_paths_l616_61618


namespace NUMINAMATH_GPT_sequence_formula_l616_61631

theorem sequence_formula (u : ℕ → ℤ) (h0 : u 0 = 1) (h1 : u 1 = 4)
  (h_rec : ∀ n : ℕ, u (n + 2) = 5 * u (n + 1) - 6 * u n) :
  ∀ n : ℕ, u n = 2 * 3^n - 2^n :=
by 
  sorry

end NUMINAMATH_GPT_sequence_formula_l616_61631


namespace NUMINAMATH_GPT_find_parallelogram_height_l616_61622

def parallelogram_height (base area : ℕ) : ℕ := area / base

theorem find_parallelogram_height :
  parallelogram_height 32 448 = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_parallelogram_height_l616_61622


namespace NUMINAMATH_GPT_inverse_proportion_order_l616_61690

theorem inverse_proportion_order (k : ℝ) (y1 y2 y3 : ℝ) 
  (h1 : k > 0) 
  (ha : y1 = k / (-3)) 
  (hb : y2 = k / (-2)) 
  (hc : y3 = k / 2) : 
  y2 < y1 ∧ y1 < y3 := 
sorry

end NUMINAMATH_GPT_inverse_proportion_order_l616_61690


namespace NUMINAMATH_GPT_tan_sum_property_l616_61678

theorem tan_sum_property (t23 t37 : ℝ) (h1 : 23 + 37 = 60) (h2 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3) :
  Real.tan (23 * Real.pi / 180) + Real.tan (37 * Real.pi / 180) + Real.sqrt 3 * Real.tan (23 * Real.pi / 180) * Real.tan (37 * Real.pi / 180) = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_sum_property_l616_61678


namespace NUMINAMATH_GPT_sin_cos_identity_tan_identity_l616_61644

open Real

namespace Trigonometry

variable (α : ℝ)

-- Given conditions
def given_conditions := (sin α + cos α = (1/5)) ∧ (0 < α) ∧ (α < π)

-- Prove that sin(α) * cos(α) = -12/25
theorem sin_cos_identity (h : given_conditions α) : sin α * cos α = -12/25 := 
sorry

-- Prove that tan(α) = -4/3
theorem tan_identity (h : given_conditions α) : tan α = -4/3 :=
sorry

end Trigonometry

end NUMINAMATH_GPT_sin_cos_identity_tan_identity_l616_61644


namespace NUMINAMATH_GPT_value_of_six_prime_prime_l616_61655

-- Define the function q' 
def prime (q : ℝ) : ℝ := 3 * q - 3

-- Stating the main theorem we want to prove
theorem value_of_six_prime_prime : prime (prime 6) = 42 :=
by
  sorry

end NUMINAMATH_GPT_value_of_six_prime_prime_l616_61655


namespace NUMINAMATH_GPT_sandy_age_l616_61607

variable (S M N : ℕ)

theorem sandy_age (h1 : M = S + 20)
                  (h2 : (S : ℚ) / M = 7 / 9)
                  (h3 : S + M + N = 120)
                  (h4 : N - M = (S - M) / 2) :
                  S = 70 := 
sorry

end NUMINAMATH_GPT_sandy_age_l616_61607


namespace NUMINAMATH_GPT_simplify_expression_l616_61603

variable (x : ℝ)

theorem simplify_expression :
  (2 * x * (4 * x^2 - 3) - 4 * (x^2 - 3 * x + 6)) = (8 * x^3 - 4 * x^2 + 6 * x - 24) := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l616_61603


namespace NUMINAMATH_GPT_fixed_point_exists_l616_61662

theorem fixed_point_exists : ∀ (m : ℝ), (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
by
  intro m
  have h : (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
    sorry
  exact h

end NUMINAMATH_GPT_fixed_point_exists_l616_61662


namespace NUMINAMATH_GPT_women_ratio_l616_61633

theorem women_ratio (pop : ℕ) (w_retail : ℕ) (w_fraction : ℚ) (h_pop : pop = 6000000) (h_w_retail : w_retail = 1000000) (h_w_fraction : w_fraction = 1 / 3) : 
  (3000000 : ℚ) / (6000000 : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_women_ratio_l616_61633


namespace NUMINAMATH_GPT_exponent_arithmetic_proof_l616_61639

theorem exponent_arithmetic_proof :
  ( (6 ^ 6 / 6 ^ 5) ^ 3 * 8 ^ 3 / 4 ^ 3) = 1728 := by
  sorry

end NUMINAMATH_GPT_exponent_arithmetic_proof_l616_61639


namespace NUMINAMATH_GPT_solve_for_Theta_l616_61668

-- Define the two-digit number representation condition
def fourTheta (Θ : ℕ) : ℕ := 40 + Θ

-- Main theorem statement
theorem solve_for_Theta (Θ : ℕ) (h1 : 198 / Θ = fourTheta Θ + Θ) (h2 : 0 < Θ ∧ Θ < 10) : Θ = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_Theta_l616_61668


namespace NUMINAMATH_GPT_range_of_a_l616_61691

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 - x + (a - 4) = 0 ∧ y^2 - y + (a - 4) = 0 ∧ x > 0 ∧ y < 0) → a < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l616_61691


namespace NUMINAMATH_GPT_area_of_quadrilateral_ABCD_l616_61634

theorem area_of_quadrilateral_ABCD
  (BD : ℝ) (hA : ℝ) (hC : ℝ) (angle_ABD : ℝ) :
  BD = 28 ∧ hA = 8 ∧ hC = 2 ∧ angle_ABD = 60 →
  ∃ (area_ABCD : ℝ), area_ABCD = 140 :=
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_ABCD_l616_61634


namespace NUMINAMATH_GPT_bus_stops_for_45_minutes_per_hour_l616_61694

-- Define the conditions
def speed_excluding_stoppages : ℝ := 48 -- in km/hr
def speed_including_stoppages : ℝ := 12 -- in km/hr

-- Define the statement to be proven
theorem bus_stops_for_45_minutes_per_hour :
  let speed_reduction := speed_excluding_stoppages - speed_including_stoppages
  let time_stopped : ℝ := (speed_reduction / speed_excluding_stoppages) * 60
  time_stopped = 45 :=
by
  sorry

end NUMINAMATH_GPT_bus_stops_for_45_minutes_per_hour_l616_61694


namespace NUMINAMATH_GPT_pete_and_ray_spent_200_cents_l616_61698

-- Define the basic units
def cents_in_a_dollar := 100
def value_of_a_nickel := 5
def value_of_a_dime := 10

-- Define the initial amounts and spending
def pete_initial_amount := 250  -- cents
def ray_initial_amount := 250  -- cents
def pete_spent_nickels := 4 * value_of_a_nickel
def ray_remaining_dimes := 7 * value_of_a_dime

-- Calculate total amounts spent
def total_spent_pete := pete_spent_nickels
def total_spent_ray := ray_initial_amount - ray_remaining_dimes
def total_spent := total_spent_pete + total_spent_ray

-- The proof problem statement
theorem pete_and_ray_spent_200_cents : total_spent = 200 := by {
 sorry
}

end NUMINAMATH_GPT_pete_and_ray_spent_200_cents_l616_61698


namespace NUMINAMATH_GPT_y_intercept_is_2_l616_61695

def equation_of_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

def point_P : ℝ × ℝ := (-1, 1)

def y_intercept_of_tangent_line (m c x y : ℝ) : Prop :=
  equation_of_circle x y ∧
  ((y = m * x + c) ∧ (point_P.1, point_P.2) ∈ {(x, y) | y = m * x + c})

theorem y_intercept_is_2 :
  ∃ m c : ℝ, y_intercept_of_tangent_line m c 0 2 :=
sorry

end NUMINAMATH_GPT_y_intercept_is_2_l616_61695


namespace NUMINAMATH_GPT_smallest_integer_l616_61667

/-- The smallest integer m such that m > 1 and m has a remainder of 1 when divided by any of 5, 7, and 3 is 106. -/
theorem smallest_integer (m : ℕ) : m > 1 ∧ m % 5 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 ↔ m = 106 :=
by
    sorry

end NUMINAMATH_GPT_smallest_integer_l616_61667


namespace NUMINAMATH_GPT_minimum_value_of_a_l616_61601

theorem minimum_value_of_a (a b : ℕ) (h₁ : b - a = 2013) 
(h₂ : ∃ x : ℕ, x^2 - a * x + b = 0) : a = 93 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_a_l616_61601


namespace NUMINAMATH_GPT_root_sum_value_l616_61645

theorem root_sum_value (r s t : ℝ) (h1: r + s + t = 24) (h2: r * s + s * t + t * r = 50) (h3: r * s * t = 24) :
  r / (1/r + s * t) + s / (1/s + t * r) + t / (1/t + r * s) = 19.04 :=
sorry

end NUMINAMATH_GPT_root_sum_value_l616_61645


namespace NUMINAMATH_GPT_equation_of_perpendicular_line_l616_61656

theorem equation_of_perpendicular_line (x y c : ℝ) (h₁ : x = -1) (h₂ : y = 2)
  (h₃ : 2 * x - 3 * y = -c) (h₄ : 3 * x + 2 * y - 7 = 0) :
  2 * x - 3 * y + 8 = 0 :=
sorry

end NUMINAMATH_GPT_equation_of_perpendicular_line_l616_61656


namespace NUMINAMATH_GPT_required_brick_volume_l616_61658

theorem required_brick_volume :
  let height := 4 / 12 -- in feet
  let length := 6 -- in feet
  let thickness := 4 / 12 -- in feet
  let volume := height * length * thickness
  let rounded_volume := Nat.ceil volume
  rounded_volume = 1 := 
by
  let height := 1 / 3
  let length := 6
  let thickness := 1 / 3
  let volume := height * length * thickness
  let rounded_volume := Nat.ceil volume
  show rounded_volume = 1
  sorry

end NUMINAMATH_GPT_required_brick_volume_l616_61658


namespace NUMINAMATH_GPT_compute_value_of_fractions_l616_61647

theorem compute_value_of_fractions (a b c : ℝ) 
  (h1 : (ac / (a + b)) + (ba / (b + c)) + (cb / (c + a)) = 0)
  (h2 : (bc / (a + b)) + (ca / (b + c)) + (ab / (c + a)) = 1) :
  (b / (a + b)) + (c / (b + c)) + (a / (c + a)) = 5 / 2 :=
sorry

end NUMINAMATH_GPT_compute_value_of_fractions_l616_61647


namespace NUMINAMATH_GPT_calculate_expression_l616_61664

theorem calculate_expression (m n : ℝ) : 9 * m^2 - (m - 2 * n)^2 = 4 * (2 * m - n) * (m + n) :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l616_61664


namespace NUMINAMATH_GPT_train_time_to_pass_bridge_l616_61660

theorem train_time_to_pass_bridge
  (length_train : ℝ) (length_bridge : ℝ) (speed_kmph : ℝ)
  (h1 : length_train = 500) (h2 : length_bridge = 200) (h3 : speed_kmph = 72) :
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_mps
  time = 35 :=
by
  sorry

end NUMINAMATH_GPT_train_time_to_pass_bridge_l616_61660


namespace NUMINAMATH_GPT_lemonade_water_requirement_l616_61635

variables (W S L H : ℕ)

-- Definitions based on the conditions
def water_equation (W S : ℕ) := W = 5 * S
def sugar_equation (S L : ℕ) := S = 3 * L
def honey_equation (H L : ℕ) := H = L
def lemon_juice_amount (L : ℕ) := L = 2

-- Theorem statement for the proof problem
theorem lemonade_water_requirement :
  ∀ (W S L H : ℕ), 
  (water_equation W S) →
  (sugar_equation S L) →
  (honey_equation H L) →
  (lemon_juice_amount L) →
  W = 30 :=
by
  intros W S L H hW hS hH hL
  sorry

end NUMINAMATH_GPT_lemonade_water_requirement_l616_61635


namespace NUMINAMATH_GPT_goods_train_speed_l616_61649

theorem goods_train_speed :
  ∀ (length_train length_platform time : ℝ),
    length_train = 250.0416 →
    length_platform = 270 →
    time = 26 →
    (length_train + length_platform) / time = 20 :=
by
  intros length_train length_platform time H_train H_platform H_time
  rw [H_train, H_platform, H_time]
  norm_num
  sorry

end NUMINAMATH_GPT_goods_train_speed_l616_61649


namespace NUMINAMATH_GPT_diamond_example_l616_61624

def diamond (a b : ℕ) : ℤ := 4 * a + 5 * b - a^2 * b

theorem diamond_example : diamond 3 4 = -4 :=
by
  rw [diamond]
  calc
    4 * 3 + 5 * 4 - 3^2 * 4 = 12 + 20 - 36 := by norm_num
                           _              = -4 := by norm_num

end NUMINAMATH_GPT_diamond_example_l616_61624


namespace NUMINAMATH_GPT_total_profit_l616_61617

-- Definitions based on the conditions
variables (A B C : ℝ) (P : ℝ)
variables (hA : A = 3 * B) (hB : B = (2 / 3) * C) (hB_share : ((2 / 11) * P) = 1400)

-- The theorem we are going to prove
theorem total_profit (A B C P : ℝ) (hA : A = 3 * B) (hB : B = (2 / 3) * C) (hB_share : ((2 / 11) * P) = 1400) : 
  P = 7700 :=
by
  sorry

end NUMINAMATH_GPT_total_profit_l616_61617


namespace NUMINAMATH_GPT_limit_hours_overtime_l616_61613

theorem limit_hours_overtime (R O : ℝ) (earnings total_hours : ℕ) (L : ℕ) 
    (hR : R = 16)
    (hO : O = R + 0.75 * R)
    (h_earnings : earnings = 864)
    (h_total_hours : total_hours = 48)
    (calc_earnings : earnings = L * R + (total_hours - L) * O) :
    L = 40 := by
  sorry

end NUMINAMATH_GPT_limit_hours_overtime_l616_61613


namespace NUMINAMATH_GPT_square_sum_zero_real_variables_l616_61686

theorem square_sum_zero_real_variables (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end NUMINAMATH_GPT_square_sum_zero_real_variables_l616_61686


namespace NUMINAMATH_GPT_remainder_y150_div_yminus2_4_l616_61661

theorem remainder_y150_div_yminus2_4 (y : ℝ) :
  (y ^ 150) % ((y - 2) ^ 4) = 554350 * (y - 2) ^ 3 + 22350 * (y - 2) ^ 2 + 600 * (y - 2) + 8 * 2 ^ 147 :=
by
  sorry

end NUMINAMATH_GPT_remainder_y150_div_yminus2_4_l616_61661


namespace NUMINAMATH_GPT_task1_task2_task3_task4_l616_61620

-- Definitions of the given conditions
def cost_price : ℝ := 16
def selling_price_range (x : ℝ) : Prop := 16 ≤ x ∧ x ≤ 48
def init_selling_price : ℝ := 20
def init_sales_volume : ℝ := 360
def decreasing_sales_rate : ℝ := 10
def daily_sales_vol (x : ℝ) : ℝ := 360 - 10 * (x - 20)
def daily_total_profit (x : ℝ) (y : ℝ) : ℝ := y * (x - cost_price)

-- Proof task (1)
theorem task1 : daily_sales_vol 25 = 310 ∧ daily_total_profit 25 (daily_sales_vol 25) = 2790 := 
by 
    -- Your code here
    sorry

-- Proof task (2)
theorem task2 : ∀ x, daily_sales_vol x = -10 * x + 560 := 
by 
    -- Your code here
    sorry

-- Proof task (3)
theorem task3 : ∀ x, 
    W = (x - 16) * (daily_sales_vol x) 
    ∧ W = -10 * x ^ 2 + 720 * x - 8960 
    ∧ (∃ x, -10 * x ^ 2 + 720 * x - 8960 = 4000 ∧ selling_price_range x) := 
by 
    -- Your code here 
    sorry

-- Proof task (4)
theorem task4 : ∃ x, 
    -10 * (x - 36) ^ 2 + 4000 = 3000 
    ∧ selling_price_range x 
    ∧ (x = 26 ∨ x = 46) := 
by 
    -- Your code here 
    sorry

end NUMINAMATH_GPT_task1_task2_task3_task4_l616_61620


namespace NUMINAMATH_GPT_vanessa_video_files_initial_l616_61609

theorem vanessa_video_files_initial (m v r d t : ℕ) (h1 : m = 13) (h2 : r = 33) (h3 : d = 10) (h4 : t = r + d) (h5 : t = m + v) : v = 30 :=
by
  sorry

end NUMINAMATH_GPT_vanessa_video_files_initial_l616_61609


namespace NUMINAMATH_GPT_simplify_expression_l616_61646

theorem simplify_expression (y : ℝ) :
  4 * y - 3 * y^3 + 6 - (1 - 4 * y + 3 * y^3) = -6 * y^3 + 8 * y + 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l616_61646


namespace NUMINAMATH_GPT_evaluate_expression_l616_61610

theorem evaluate_expression : - (16 / 4 * 7 + 25 - 2 * 7) = -39 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l616_61610


namespace NUMINAMATH_GPT_simplify_neg_expression_l616_61608

variable (a b c : ℝ)

theorem simplify_neg_expression : 
  - (a - (b - c)) = -a + b - c :=
sorry

end NUMINAMATH_GPT_simplify_neg_expression_l616_61608


namespace NUMINAMATH_GPT_evaluate_expression_l616_61604

variable (a : ℤ) (x : ℤ)

theorem evaluate_expression (h : x = a + 9) : x - a + 5 = 14 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l616_61604


namespace NUMINAMATH_GPT_caleb_ice_cream_vs_frozen_yoghurt_l616_61669

theorem caleb_ice_cream_vs_frozen_yoghurt :
  let cost_chocolate_ice_cream := 6 * 5
  let discount_chocolate := 0.10 * cost_chocolate_ice_cream
  let total_chocolate_ice_cream := cost_chocolate_ice_cream - discount_chocolate

  let cost_vanilla_ice_cream := 4 * 4
  let discount_vanilla := 0.07 * cost_vanilla_ice_cream
  let total_vanilla_ice_cream := cost_vanilla_ice_cream - discount_vanilla

  let total_ice_cream := total_chocolate_ice_cream + total_vanilla_ice_cream

  let cost_strawberry_yoghurt := 3 * 3
  let tax_strawberry := 0.05 * cost_strawberry_yoghurt
  let total_strawberry_yoghurt := cost_strawberry_yoghurt + tax_strawberry

  let cost_mango_yoghurt := 2 * 2
  let tax_mango := 0.03 * cost_mango_yoghurt
  let total_mango_yoghurt := cost_mango_yoghurt + tax_mango

  let total_yoghurt := total_strawberry_yoghurt + total_mango_yoghurt

  (total_ice_cream - total_yoghurt = 28.31) := by
  sorry

end NUMINAMATH_GPT_caleb_ice_cream_vs_frozen_yoghurt_l616_61669


namespace NUMINAMATH_GPT_common_remainder_is_zero_l616_61627

noncomputable def least_number := 100040

theorem common_remainder_is_zero 
  (n : ℕ) 
  (h1 : n = least_number) 
  (condition1 : 4 ∣ n)
  (condition2 : 610 ∣ n)
  (condition3 : 15 ∣ n)
  (h2 : (n.digits 10).sum = 5)
  : ∃ r : ℕ, ∀ (a : ℕ), (a ∈ [4, 610, 15] → n % a = r) ∧ r = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_common_remainder_is_zero_l616_61627


namespace NUMINAMATH_GPT_average_age_of_5_students_l616_61614

theorem average_age_of_5_students
  (avg_age_20_students : ℕ → ℕ → ℕ → ℕ)
  (total_age_20 : avg_age_20_students 20 20 0 = 400)
  (total_age_9 : 9 * 16 = 144)
  (age_20th_student : ℕ := 186) :
  avg_age_20_students 5 ((400 - 144 - 186) / 5) 5 = 14 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_5_students_l616_61614


namespace NUMINAMATH_GPT_cookie_ratio_l616_61651

theorem cookie_ratio (f : ℚ) (h_monday : 32 = 32) (h_tuesday : (f : ℚ) * 32 = 32 * (f : ℚ)) 
    (h_wednesday : 3 * (f : ℚ) * 32 - 4 + 32 + (f : ℚ) * 32 = 92) :
    f = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_cookie_ratio_l616_61651


namespace NUMINAMATH_GPT_probability_ace_then_king_l616_61623

-- Definitions of the conditions
def custom_deck := 65
def extra_spades := 14
def total_aces := 4
def total_kings := 4

-- Probability calculations
noncomputable def P_ace_first : ℚ := total_aces / custom_deck
noncomputable def P_king_second : ℚ := total_kings / (custom_deck - 1)

theorem probability_ace_then_king :
  (P_ace_first * P_king_second) = 1 / 260 :=
by
  sorry

end NUMINAMATH_GPT_probability_ace_then_king_l616_61623


namespace NUMINAMATH_GPT_largest_digit_divisible_by_6_l616_61640

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ (56780 + N) % 6 = 0 ∧ (∀ M : ℕ, M ≤ 9 → (M % 2 = 0 ∧ (56780 + M) % 3 = 0) → M ≤ N) :=
by
  sorry

end NUMINAMATH_GPT_largest_digit_divisible_by_6_l616_61640


namespace NUMINAMATH_GPT_history_book_cost_is_correct_l616_61616

-- Define the conditions
def total_books : ℕ := 80
def math_book_cost : ℕ := 4
def total_price : ℕ := 390
def math_books_purchased : ℕ := 10

-- The number of history books
def history_books_purchased : ℕ := total_books - math_books_purchased

-- The total cost of math books
def total_cost_math_books : ℕ := math_books_purchased * math_book_cost

-- The total cost of history books
def total_cost_history_books : ℕ := total_price - total_cost_math_books

-- Define the cost of each history book
def history_book_cost : ℕ := total_cost_history_books / history_books_purchased

-- The theorem to be proven
theorem history_book_cost_is_correct : history_book_cost = 5 := 
by
  sorry

end NUMINAMATH_GPT_history_book_cost_is_correct_l616_61616


namespace NUMINAMATH_GPT_line_slope_angle_y_intercept_l616_61687

theorem line_slope_angle_y_intercept :
  ∀ (x y : ℝ), x - y - 1 = 0 → 
    (∃ k b : ℝ, y = x - 1 ∧ k = 1 ∧ b = -1 ∧ θ = 45 ∧ θ = Real.arctan k) := 
    by
      sorry

end NUMINAMATH_GPT_line_slope_angle_y_intercept_l616_61687


namespace NUMINAMATH_GPT_kite_area_l616_61677

theorem kite_area {length height : ℕ} (h_length : length = 8) (h_height : height = 10): 
  2 * (1/2 * (length * 2) * (height * 2 / 2)) = 160 :=
by
  rw [h_length, h_height]
  norm_num
  sorry

end NUMINAMATH_GPT_kite_area_l616_61677


namespace NUMINAMATH_GPT_time_to_cover_escalator_l616_61619

-- Definitions of the rates and length
def escalator_speed : ℝ := 12
def person_speed : ℝ := 2
def escalator_length : ℝ := 210

-- Theorem statement that we need to prove
theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed) = 15) :=
by
  sorry

end NUMINAMATH_GPT_time_to_cover_escalator_l616_61619


namespace NUMINAMATH_GPT_noelle_speed_l616_61680

theorem noelle_speed (v d : ℝ) (h1 : d > 0) (h2 : v > 0) 
  (h3 : (2 * d) / ((d / v) + (d / 15)) = 5) : v = 3 := 
sorry

end NUMINAMATH_GPT_noelle_speed_l616_61680


namespace NUMINAMATH_GPT_auntie_em_parking_probability_l616_61611

theorem auntie_em_parking_probability :
  let total_spaces := 20
  let cars := 15
  let empty_spaces := total_spaces - cars
  let possible_configurations := Nat.choose total_spaces cars
  let unfavourable_configurations := Nat.choose (empty_spaces - 8 + 5) (empty_spaces - 8)
  let favourable_probability := 1 - ((unfavourable_configurations : ℚ) / (possible_configurations : ℚ))
  (favourable_probability = 1839 / 1938) :=
by
  -- sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_auntie_em_parking_probability_l616_61611


namespace NUMINAMATH_GPT_number_decomposition_l616_61637

theorem number_decomposition : 10101 = 10000 + 100 + 1 :=
by
  sorry

end NUMINAMATH_GPT_number_decomposition_l616_61637


namespace NUMINAMATH_GPT_evaluate_fraction_l616_61636

theorem evaluate_fraction:
  (125 : ℝ)^(1/3) / (64 : ℝ)^(1/2) * (81 : ℝ)^(1/4) = 15 / 8 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l616_61636


namespace NUMINAMATH_GPT_storks_more_than_birds_l616_61670

-- Definitions based on given conditions
def initial_birds : ℕ := 3
def added_birds : ℕ := 2
def total_birds : ℕ := initial_birds + added_birds
def storks : ℕ := 6

-- Statement to prove the correct answer
theorem storks_more_than_birds : (storks - total_birds = 1) :=
by
  sorry

end NUMINAMATH_GPT_storks_more_than_birds_l616_61670


namespace NUMINAMATH_GPT_function_range_l616_61606

theorem function_range (f : ℝ → ℝ) (s : Set ℝ) (h : s = Set.Ico (-5 : ℝ) 2) (h_f : ∀ x ∈ s, f x = 3 * x - 1) :
  Set.image f s = Set.Ico (-16 : ℝ) 5 :=
sorry

end NUMINAMATH_GPT_function_range_l616_61606


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l616_61641

-- Define set M
def M : Set ℝ := {x | Real.log x > 0}

-- Define set N
def N : Set ℝ := {x | x^2 ≤ 4}

-- Define the target set
def target : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem intersection_of_M_and_N :
  M ∩ N = target :=
sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l616_61641


namespace NUMINAMATH_GPT_rhombus_angles_l616_61666

-- Define the conditions for the proof
variables (a e f : ℝ) (α β : ℝ)

-- Using the geometric mean condition
def geometric_mean_condition := a^2 = e * f

-- Using the condition that diagonals of a rhombus intersect at right angles and bisect each other
def diagonals_intersect_perpendicularly := α + β = 180 ∧ α = 30 ∧ β = 150

-- Prove the question assuming the given conditions
theorem rhombus_angles (h1 : geometric_mean_condition a e f) (h2 : diagonals_intersect_perpendicularly α β) : 
  (α = 30) ∧ (β = 150) :=
sorry

end NUMINAMATH_GPT_rhombus_angles_l616_61666


namespace NUMINAMATH_GPT_inequality_holds_infinitely_many_times_l616_61673

variable {a : ℕ → ℝ}

theorem inequality_holds_infinitely_many_times
    (h_pos : ∀ n, 0 < a n) :
    ∃ᶠ n in at_top, 1 + a n > a (n - 1) * 2^(1 / n) :=
sorry

end NUMINAMATH_GPT_inequality_holds_infinitely_many_times_l616_61673


namespace NUMINAMATH_GPT_preimage_of_4_neg_2_eq_1_3_l616_61696

def mapping (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

theorem preimage_of_4_neg_2_eq_1_3 : ∃ x y : ℝ, mapping x y = (4, -2) ∧ (x = 1) ∧ (y = 3) :=
by 
  sorry

end NUMINAMATH_GPT_preimage_of_4_neg_2_eq_1_3_l616_61696


namespace NUMINAMATH_GPT_slope_range_l616_61665

theorem slope_range (a b : ℝ) (h₁ : a ≠ -2) (h₂ : a ≠ 2) 
  (h₃ : a^2 / 4 + b^2 / 3 = 1) (h₄ : -2 ≤ b / (a - 2) ∧ b / (a - 2) ≤ -1) :
  (3 / 8 ≤ b / (a + 2) ∧ b / (a + 2) ≤ 3 / 4) :=
sorry

end NUMINAMATH_GPT_slope_range_l616_61665


namespace NUMINAMATH_GPT_jackson_maximum_usd_l616_61682

-- Define the rates for chores in various currencies
def usd_per_hour : ℝ := 5
def gbp_per_hour : ℝ := 3
def jpy_per_hour : ℝ := 400
def eur_per_hour : ℝ := 4

-- Define the hours Jackson worked for each task
def usd_hours_vacuuming : ℝ := 2 * 2
def gbp_hours_washing_dishes : ℝ := 0.5
def jpy_hours_cleaning_bathroom : ℝ := 1.5
def eur_hours_sweeping_yard : ℝ := 1

-- Define the exchange rates over three days
def exchange_rates_day1 := (1.35, 0.009, 1.18)  -- (GBP to USD, JPY to USD, EUR to USD)
def exchange_rates_day2 := (1.38, 0.0085, 1.20)
def exchange_rates_day3 := (1.33, 0.0095, 1.21)

-- Define a function to convert currency to USD based on best exchange rates
noncomputable def max_usd (gbp_to_usd jpy_to_usd eur_to_usd : ℝ) : ℝ :=
  (usd_hours_vacuuming * usd_per_hour) +
  (gbp_hours_washing_dishes * gbp_per_hour * gbp_to_usd) +
  (jpy_hours_cleaning_bathroom * jpy_per_hour * jpy_to_usd) +
  (eur_hours_sweeping_yard * eur_per_hour * eur_to_usd)

-- Prove the maximum USD Jackson can have by choosing optimal rates is $32.61
theorem jackson_maximum_usd : max_usd 1.38 0.0095 1.21 = 32.61 :=
by
  sorry

end NUMINAMATH_GPT_jackson_maximum_usd_l616_61682


namespace NUMINAMATH_GPT_region_area_correct_l616_61659

noncomputable def region_area : ℝ :=
  let region := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 6}
  let area := (3 - -3) * (3 - -3)
  area

theorem region_area_correct : region_area = 36 :=
by sorry

end NUMINAMATH_GPT_region_area_correct_l616_61659


namespace NUMINAMATH_GPT_quadratic_passing_origin_l616_61689

theorem quadratic_passing_origin (a b c : ℝ) (h : a ≠ 0) :
  ((∀ x y : ℝ, x = 0 → y = 0 → y = a * x^2 + b * x + c) ↔ c = 0) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_passing_origin_l616_61689


namespace NUMINAMATH_GPT_sqrt_floor_eq_sqrt_sqrt_floor_l616_61679

theorem sqrt_floor_eq_sqrt_sqrt_floor (a : ℝ) (h : a > 1) :
  Int.floor (Real.sqrt (Int.floor (Real.sqrt a))) = Int.floor (Real.sqrt (Real.sqrt a)) :=
sorry

end NUMINAMATH_GPT_sqrt_floor_eq_sqrt_sqrt_floor_l616_61679
