import Mathlib

namespace NUMINAMATH_GPT_chairs_to_remove_is_33_l855_85519

-- Definitions for the conditions
def chairs_per_row : ℕ := 11
def total_chairs : ℕ := 110
def students : ℕ := 70

-- Required statement
theorem chairs_to_remove_is_33 
  (h_divisible_by_chairs_per_row : ∀ n, n = total_chairs - students → ∃ k, n = chairs_per_row * k) :
  ∃ rem_chairs : ℕ, rem_chairs = total_chairs - 77 ∧ rem_chairs = 33 := sorry

end NUMINAMATH_GPT_chairs_to_remove_is_33_l855_85519


namespace NUMINAMATH_GPT_max_sum_of_squares_eq_l855_85543

theorem max_sum_of_squares_eq (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := 
by
  sorry

end NUMINAMATH_GPT_max_sum_of_squares_eq_l855_85543


namespace NUMINAMATH_GPT_number_of_young_fish_l855_85581

-- Define the conditions
def tanks : ℕ := 3
def pregnantFishPerTank : ℕ := 4
def youngPerFish : ℕ := 20

-- Define the proof problem
theorem number_of_young_fish : (tanks * pregnantFishPerTank * youngPerFish) = 240 := by
  sorry

end NUMINAMATH_GPT_number_of_young_fish_l855_85581


namespace NUMINAMATH_GPT_canal_cross_section_area_l855_85520

/-- Definitions of the conditions -/
def top_width : Real := 6
def bottom_width : Real := 4
def depth : Real := 257.25

/-- Proof statement -/
theorem canal_cross_section_area : 
  (1 / 2) * (top_width + bottom_width) * depth = 1286.25 :=
by
  sorry

end NUMINAMATH_GPT_canal_cross_section_area_l855_85520


namespace NUMINAMATH_GPT_episodes_count_l855_85534

variable (minutes_per_episode : ℕ) (total_watching_time_minutes : ℕ)
variable (episodes_watched : ℕ)

theorem episodes_count 
  (h1 : minutes_per_episode = 50) 
  (h2 : total_watching_time_minutes = 300) 
  (h3 : total_watching_time_minutes / minutes_per_episode = episodes_watched) :
  episodes_watched = 6 := sorry

end NUMINAMATH_GPT_episodes_count_l855_85534


namespace NUMINAMATH_GPT_find_x_l855_85554

noncomputable def area_of_figure (x : ℝ) : ℝ :=
  let A_rectangle := 3 * x * 2 * x
  let A_square1 := x ^ 2
  let A_square2 := (4 * x) ^ 2
  let A_triangle := (3 * x * 2 * x) / 2
  A_rectangle + A_square1 + A_square2 + A_triangle

theorem find_x (x : ℝ) : area_of_figure x = 1250 → x = 6.93 :=
  sorry

end NUMINAMATH_GPT_find_x_l855_85554


namespace NUMINAMATH_GPT_prob_score_3_points_l855_85580

-- Definitions for the probabilities
def probability_hit_A := 3/4
def score_hit_A := 1
def score_miss_A := -1

def probability_hit_B := 2/3
def score_hit_B := 2
def score_miss_B := 0

-- Conditional probabilities and their calculations
noncomputable def prob_scenario_1 : ℚ := 
  probability_hit_A * 2 * probability_hit_B * (1 - probability_hit_B)

noncomputable def prob_scenario_2 : ℚ := 
  (1 - probability_hit_A) * probability_hit_B^2

noncomputable def total_prob : ℚ := 
  prob_scenario_1 + prob_scenario_2

-- The final proof statement
theorem prob_score_3_points : total_prob = 4/9 := sorry

end NUMINAMATH_GPT_prob_score_3_points_l855_85580


namespace NUMINAMATH_GPT_sin_780_eq_sqrt3_div_2_l855_85506

theorem sin_780_eq_sqrt3_div_2 :
  Real.sin (780 * Real.pi / 180) = (Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_780_eq_sqrt3_div_2_l855_85506


namespace NUMINAMATH_GPT_proof_problem_l855_85510

variables {x1 y1 x2 y2 : ℝ}

-- Definitions
def unit_vector (x y : ℝ) : Prop := x^2 + y^2 = 1
def angle_with_p (x y : ℝ) : Prop := (x + y) / Real.sqrt 2 = Real.sqrt 3 / 2
def m := (x1, y1)
def n := (x2, y2)
def p := (1, 1)

-- Conditions
lemma unit_m : unit_vector x1 y1 := sorry
lemma unit_n : unit_vector x2 y2 := sorry
lemma angle_m_p : angle_with_p x1 y1 := sorry
lemma angle_n_p : angle_with_p x2 y2 := sorry

-- Theorem to prove
theorem proof_problem (h1 : unit_vector x1 y1)
                      (h2 : unit_vector x2 y2)
                      (h3 : angle_with_p x1 y1)
                      (h4 : angle_with_p x2 y2) :
                      (x1 * x2 + y1 * y2 = 1/2) ∧ (y1 * y2 / (x1 * x2) = 1) :=
sorry

end NUMINAMATH_GPT_proof_problem_l855_85510


namespace NUMINAMATH_GPT_part_a_solution_part_b_solution_l855_85596

-- Part (a) Statement in Lean 4
theorem part_a_solution (N : ℕ) (a b : ℕ) (h : N = a * 10^n + b * 10^(n-1)) :
  ∃ (m : ℕ), (N / 10 = m) -> m * 10 = N := sorry

-- Part (b) Statement in Lean 4
theorem part_b_solution (N : ℕ) (a b c : ℕ) (h : N = a * 10^n + b * 10^(n-1) + c * 10^(n-2)) :
  ∃ (m : ℕ), (N / 10^(n-1) = m) -> m * 10^(n-1) = N := sorry

end NUMINAMATH_GPT_part_a_solution_part_b_solution_l855_85596


namespace NUMINAMATH_GPT_first_year_exceeds_two_million_l855_85558

-- Definition of the initial R&D investment in 2015
def initial_investment : ℝ := 1.3

-- Definition of the annual growth rate
def growth_rate : ℝ := 1.12

-- Definition of the investment function for year n
def investment (n : ℕ) : ℝ := initial_investment * growth_rate ^ (n - 2015)

-- The problem statement to be proven
theorem first_year_exceeds_two_million : ∃ n : ℕ, n > 2015 ∧ investment n > 2 ∧ ∀ m : ℕ, (m < n ∧ m > 2015) → investment m ≤ 2 := by
  sorry

end NUMINAMATH_GPT_first_year_exceeds_two_million_l855_85558


namespace NUMINAMATH_GPT_age_difference_l855_85515

theorem age_difference (A B : ℕ) (h1 : B = 37) (h2 : A + 10 = 2 * (B - 10)) : A - B = 7 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l855_85515


namespace NUMINAMATH_GPT_problem_statement_l855_85528

theorem problem_statement (r p q : ℝ) (hr : r < 0) (hpq_ne_zero : p * q ≠ 0) (hp2r_gt_q2r : p^2 * r > q^2 * r) :
  ¬ (-p > -q) ∧ ¬ (-p < q) ∧ ¬ (1 < -q / p) ∧ ¬ (1 > q / p) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l855_85528


namespace NUMINAMATH_GPT_max_value_of_x_l855_85578

theorem max_value_of_x (x y : ℝ) (h : x^2 + y^2 = 18 * x + 20 * y) : x ≤ 9 + Real.sqrt 181 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_x_l855_85578


namespace NUMINAMATH_GPT_find_shirt_numbers_calculate_profit_l855_85541

def total_shirts_condition (x y : ℕ) : Prop := x + y = 200
def total_cost_condition (x y : ℕ) : Prop := 25 * x + 15 * y = 3500
def profit_calculation (x y : ℕ) : ℕ := (50 - 25) * x + (35 - 15) * y

theorem find_shirt_numbers (x y : ℕ) (h1 : total_shirts_condition x y) (h2 : total_cost_condition x y) :
  x = 50 ∧ y = 150 :=
sorry

theorem calculate_profit (x y : ℕ) (h1 : total_shirts_condition x y) (h2 : total_cost_condition x y) :
  profit_calculation x y = 4250 :=
sorry

end NUMINAMATH_GPT_find_shirt_numbers_calculate_profit_l855_85541


namespace NUMINAMATH_GPT_min_sum_four_consecutive_nat_nums_l855_85576

theorem min_sum_four_consecutive_nat_nums (a : ℕ) (h1 : a % 11 = 0) (h2 : (a + 1) % 7 = 0)
    (h3 : (a + 2) % 5 = 0) (h4 : (a + 3) % 3 = 0) : a + (a + 1) + (a + 2) + (a + 3) = 1458 :=
  sorry

end NUMINAMATH_GPT_min_sum_four_consecutive_nat_nums_l855_85576


namespace NUMINAMATH_GPT_alpine_school_math_students_l855_85598

theorem alpine_school_math_students (total_players : ℕ) (physics_players : ℕ) (both_players : ℕ) :
  total_players = 15 → physics_players = 9 → both_players = 4 → 
  ∃ math_players : ℕ, math_players = total_players - (physics_players - both_players) + both_players := by
  sorry

end NUMINAMATH_GPT_alpine_school_math_students_l855_85598


namespace NUMINAMATH_GPT_bus_ride_cost_l855_85549

noncomputable def bus_cost : ℝ := 1.75

theorem bus_ride_cost (B T : ℝ) (h1 : T = B + 6.35) (h2 : T + B = 9.85) : B = bus_cost :=
by
  sorry

end NUMINAMATH_GPT_bus_ride_cost_l855_85549


namespace NUMINAMATH_GPT_lcm_two_numbers_l855_85561

theorem lcm_two_numbers
  (a b : ℕ)
  (hcf_ab : Nat.gcd a b = 20)
  (product_ab : a * b = 2560) :
  Nat.lcm a b = 128 :=
by
  sorry

end NUMINAMATH_GPT_lcm_two_numbers_l855_85561


namespace NUMINAMATH_GPT_value_of_f_f_3_l855_85592

def f (x : ℝ) := 3 * x^2 + 3 * x - 2

theorem value_of_f_f_3 : f (f 3) = 3568 :=
by {
  -- Definition of f is already given in the conditions
  sorry
}

end NUMINAMATH_GPT_value_of_f_f_3_l855_85592


namespace NUMINAMATH_GPT_Connie_total_markers_l855_85564

/--
Connie has 41 red markers and 64 blue markers. 
We want to prove that the total number of markers Connie has is 105.
-/
theorem Connie_total_markers : 
  let red_markers := 41
  let blue_markers := 64
  let total_markers := red_markers + blue_markers
  total_markers = 105 :=
by
  sorry

end NUMINAMATH_GPT_Connie_total_markers_l855_85564


namespace NUMINAMATH_GPT_product_of_distinct_nonzero_real_numbers_l855_85573

variable {x y : ℝ}

theorem product_of_distinct_nonzero_real_numbers (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 4 / x = y + 4 / y) : x * y = 4 := 
sorry

end NUMINAMATH_GPT_product_of_distinct_nonzero_real_numbers_l855_85573


namespace NUMINAMATH_GPT_total_apples_l855_85546

theorem total_apples (apples_per_person : ℕ) (num_people : ℕ) (h1 : apples_per_person = 25) (h2 : num_people = 6) : apples_per_person * num_people = 150 := by
  sorry

end NUMINAMATH_GPT_total_apples_l855_85546


namespace NUMINAMATH_GPT_product_of_b_product_of_values_l855_85563

/-- 
If the distance between the points (3b, b+2) and (6, 3) is 3√5 units,
then the product of all possible values of b is -0.8.
-/
theorem product_of_b (b : ℝ)
  (h : (6 - 3 * b)^2 + (3 - (b + 2))^2 = (3 * Real.sqrt 5)^2) :
  b = 4 ∨ b = -0.2 := sorry

/--
The product of the values satisfying the theorem product_of_b is -0.8.
-/
theorem product_of_values : (4 : ℝ) * (-0.2) = -0.8 := 
by norm_num -- using built-in arithmetic simplification

end NUMINAMATH_GPT_product_of_b_product_of_values_l855_85563


namespace NUMINAMATH_GPT_white_paint_amount_is_correct_l855_85503

noncomputable def totalAmountOfPaint (bluePaint: ℝ) (bluePercentage: ℝ): ℝ :=
  bluePaint / bluePercentage

noncomputable def whitePaintAmount (totalPaint: ℝ) (whitePercentage: ℝ): ℝ :=
  totalPaint * whitePercentage

theorem white_paint_amount_is_correct (bluePaint: ℝ) (bluePercentage: ℝ) (whitePercentage: ℝ) (totalPaint: ℝ) :
  bluePaint = 140 → bluePercentage = 0.7 → whitePercentage = 0.1 → totalPaint = totalAmountOfPaint 140 0.7 →
  whitePaintAmount totalPaint 0.1 = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_white_paint_amount_is_correct_l855_85503


namespace NUMINAMATH_GPT_air_conditioner_sale_price_l855_85570

theorem air_conditioner_sale_price (P : ℝ) (d1 d2 : ℝ) (hP : P = 500) (hd1 : d1 = 0.10) (hd2 : d2 = 0.20) :
  ((P * (1 - d1)) * (1 - d2)) / P * 100 = 72 :=
by
  sorry

end NUMINAMATH_GPT_air_conditioner_sale_price_l855_85570


namespace NUMINAMATH_GPT_parabola_midpoint_locus_minimum_slope_difference_exists_l855_85585

open Real

def parabola_locus (x y : ℝ) : Prop :=
  x^2 = 4 * y

def slope_difference_condition (x1 x2 k1 k2 : ℝ) : Prop :=
  |k1 - k2| = 1

theorem parabola_midpoint_locus :
  ∀ (x y : ℝ), parabola_locus x y :=
by
  intros x y
  apply sorry

theorem minimum_slope_difference_exists :
  ∀ {x1 y1 x2 y2 k1 k2 : ℝ},
  slope_difference_condition x1 x2 k1 k2 :=
by
  intros x1 y1 x2 y2 k1 k2
  apply sorry

end NUMINAMATH_GPT_parabola_midpoint_locus_minimum_slope_difference_exists_l855_85585


namespace NUMINAMATH_GPT_min_value_of_expression_l855_85568

noncomputable def expression (x : ℝ) : ℝ := (15 - x) * (12 - x) * (15 + x) * (12 + x)

theorem min_value_of_expression :
  ∃ x : ℝ, (expression x) = -1640.25 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l855_85568


namespace NUMINAMATH_GPT_lesser_fraction_l855_85575

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3/4) (h₂ : x * y = 1/8) : min x y = 1/4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_lesser_fraction_l855_85575


namespace NUMINAMATH_GPT_slope_of_tangent_at_A_l855_85540

def f (x : ℝ) : ℝ := x^2 + 3 * x

def f' (x : ℝ) : ℝ := 2 * x + 3

theorem slope_of_tangent_at_A : f' 2 = 7 := by
  sorry

end NUMINAMATH_GPT_slope_of_tangent_at_A_l855_85540


namespace NUMINAMATH_GPT_find_different_weighted_coins_l855_85550

-- Define the conditions and the theorem
def num_coins : Nat := 128
def weight_types : Nat := 2
def coins_of_each_weight : Nat := 64

theorem find_different_weighted_coins (weighings_at_most : Nat := 7) :
  ∃ (w1 w2 : Nat) (coins : Fin num_coins → Nat), w1 ≠ w2 ∧ 
  (∃ (pair : Fin num_coins × Fin num_coins), pair.fst ≠ pair.snd ∧ coins pair.fst ≠ coins pair.snd) :=
sorry

end NUMINAMATH_GPT_find_different_weighted_coins_l855_85550


namespace NUMINAMATH_GPT_log_xy_l855_85571

-- Definitions from conditions
def log (z : ℝ) : ℝ := sorry -- Assume a definition of log function
variables (x y : ℝ)
axiom h1 : log (x^2 * y^2) = 1
axiom h2 : log (x^3 * y) = 2

-- The proof goal
theorem log_xy (x y : ℝ) (h1 : log (x^2 * y^2) = 1) (h2 : log (x^3 * y) = 2) : log (x * y) = 1/2 :=
sorry

end NUMINAMATH_GPT_log_xy_l855_85571


namespace NUMINAMATH_GPT_remainder_of_product_l855_85513

theorem remainder_of_product (a b c : ℕ) (hc : c ≥ 3) (h1 : a % c = 1) (h2 : b % c = 2) : (a * b) % c = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_product_l855_85513


namespace NUMINAMATH_GPT_estimate_2_sqrt_5_l855_85559

theorem estimate_2_sqrt_5: 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 :=
by
  sorry

end NUMINAMATH_GPT_estimate_2_sqrt_5_l855_85559


namespace NUMINAMATH_GPT_domain_of_function_l855_85590

theorem domain_of_function :
  ∀ x : ℝ, (2 - x > 0) ∧ (2 * x + 1 > 0) ↔ (-1 / 2 < x) ∧ (x < 2) :=
sorry

end NUMINAMATH_GPT_domain_of_function_l855_85590


namespace NUMINAMATH_GPT_contrapositive_of_sum_of_squares_l855_85544

theorem contrapositive_of_sum_of_squares
  (a b : ℝ)
  (h : a ≠ 0 ∨ b ≠ 0) :
  a^2 + b^2 ≠ 0 := 
sorry

end NUMINAMATH_GPT_contrapositive_of_sum_of_squares_l855_85544


namespace NUMINAMATH_GPT_quadratic_solution_exists_l855_85502

-- Define the conditions
variables (a b : ℝ) (h₀ : a ≠ 0)
-- The condition that the first quadratic equation has at most one solution
def has_at_most_one_solution (a b : ℝ) : Prop :=
  b^2 + 4*a*(a - 3) <= 0

-- The second quadratic equation
def second_equation (a b x : ℝ) : ℝ :=
  (b - 3) * x^2 + (a - 2 * b) * x + 3 * a + 3
  
-- The proof problem invariant in Lean 4
theorem quadratic_solution_exists (h₁ : has_at_most_one_solution a b) :
  ∃ x : ℝ, second_equation a b x = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_exists_l855_85502


namespace NUMINAMATH_GPT_work_together_l855_85572

theorem work_together (A B : ℝ) (hA : A = 1/3) (hB : B = 1/6) : (1 / (A + B)) = 2 := by
  sorry

end NUMINAMATH_GPT_work_together_l855_85572


namespace NUMINAMATH_GPT_scooter_cost_l855_85547

variable (saved needed total_cost : ℕ)

-- The conditions given in the problem
def greg_saved_57 : saved = 57 := sorry
def greg_needs_33_more : needed = 33 := sorry

-- The proof goal
theorem scooter_cost (h1 : saved = 57) (h2 : needed = 33) :
  total_cost = saved + needed → total_cost = 90 := by
  sorry

end NUMINAMATH_GPT_scooter_cost_l855_85547


namespace NUMINAMATH_GPT_Jeanine_more_pencils_than_Clare_l855_85517

variables (Jeanine_pencils : ℕ) (Clare_pencils : ℕ)

def Jeanine_initial_pencils := 18
def Clare_initial_pencils := Jeanine_initial_pencils / 2
def Jeanine_pencils_given_to_Abby := Jeanine_initial_pencils / 3
def Jeanine_remaining_pencils := Jeanine_initial_pencils - Jeanine_pencils_given_to_Abby

theorem Jeanine_more_pencils_than_Clare :
  Jeanine_remaining_pencils - Clare_initial_pencils = 3 :=
by
  -- This is just the statement, the proof is not provided as instructed.
  sorry

end NUMINAMATH_GPT_Jeanine_more_pencils_than_Clare_l855_85517


namespace NUMINAMATH_GPT_probability_three_digit_divisible_by_5_with_ones_digit_9_l855_85521

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ones_digit (n : ℕ) : ℕ := n % 10

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem probability_three_digit_divisible_by_5_with_ones_digit_9 : 
  ∀ (M : ℕ), is_three_digit M → ones_digit M = 9 → ¬ is_divisible_by_5 M := by
  intros M h1 h2
  sorry

end NUMINAMATH_GPT_probability_three_digit_divisible_by_5_with_ones_digit_9_l855_85521


namespace NUMINAMATH_GPT_total_cakes_served_l855_85532

-- Conditions
def cakes_lunch : Nat := 6
def cakes_dinner : Nat := 9

-- Statement of the problem
theorem total_cakes_served : cakes_lunch + cakes_dinner = 15 := 
by
  sorry

end NUMINAMATH_GPT_total_cakes_served_l855_85532


namespace NUMINAMATH_GPT_nonnegative_difference_roots_eq_12_l855_85584

theorem nonnegative_difference_roots_eq_12 :
  ∀ (x : ℝ), (x^2 + 40 * x + 300 = -64) →
  ∃ (r₁ r₂ : ℝ), (x^2 + 40 * x + 364 = 0) ∧ 
  (r₁ = -26 ∧ r₂ = -14)
  ∧ (|r₁ - r₂| = 12) :=
by
  sorry

end NUMINAMATH_GPT_nonnegative_difference_roots_eq_12_l855_85584


namespace NUMINAMATH_GPT_ratio_of_inscribed_squares_l855_85501

-- Definitions of the conditions
def right_triangle_sides (a b c : ℕ) : Prop := a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2

def inscribed_square_1 (x : ℚ) : Prop := x = 18 / 7

def inscribed_square_2 (y : ℚ) : Prop := y = 32 / 7

-- Statement of the problem
theorem ratio_of_inscribed_squares (x y : ℚ) : right_triangle_sides 6 8 10 ∧ inscribed_square_1 x ∧ inscribed_square_2 y → (x / y) = 9 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_inscribed_squares_l855_85501


namespace NUMINAMATH_GPT_youngest_child_age_l855_85577

theorem youngest_child_age (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 55) : x = 7 := 
by
  sorry

end NUMINAMATH_GPT_youngest_child_age_l855_85577


namespace NUMINAMATH_GPT_secondTrain_speed_l855_85589

/-
Conditions:
1. Two trains start from A and B and travel towards each other.
2. The distance between them is 1100 km.
3. At the time of their meeting, one train has traveled 100 km more than the other.
4. The first train's speed is 50 kmph.
-/

-- Let v be the speed of the second train
def secondTrainSpeed (v : ℝ) : Prop :=
  ∃ d : ℝ, 
    d > 0 ∧
    v > 0 ∧
    (d + (d - 100) = 1100) ∧
    ((d / 50) = ((d - 100) / v))

-- Here is the main theorem translating the problem statement:
theorem secondTrain_speed :
  secondTrainSpeed (250 / 6) :=
by
  sorry

end NUMINAMATH_GPT_secondTrain_speed_l855_85589


namespace NUMINAMATH_GPT_line_perpendicular_through_P_l855_85538

/-
  Given:
  1. The point P(-2, 2).
  2. The line 2x - y + 1 = 0.
  Prove:
  The equation of the line that passes through P and is perpendicular to the given line is x + 2y - 2 = 0.
-/

def P : ℝ × ℝ := (-2, 2)
def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0

theorem line_perpendicular_through_P :
  ∃ (x y : ℝ) (m : ℝ), (x = -2) ∧ (y = 2) ∧ (m = -1/2) ∧ 
  (∀ (x₁ y₁ : ℝ), (y₁ - y) = m * (x₁ - x)) ∧ 
  (∀ (lx ly : ℝ), line1 lx ly → x + 2 * y - 2 = 0) := sorry

end NUMINAMATH_GPT_line_perpendicular_through_P_l855_85538


namespace NUMINAMATH_GPT_area_of_EFCD_l855_85524

theorem area_of_EFCD (AB CD h : ℝ) (H_AB : AB = 10) (H_CD : CD = 30) (H_h : h = 15) :
  let EF := (AB + CD) / 2
  let h_EFCD := h / 2
  let area_EFCD := (1 / 2) * (CD + EF) * h_EFCD
  area_EFCD = 187.5 :=
by
  intros EF h_EFCD area_EFCD
  sorry

end NUMINAMATH_GPT_area_of_EFCD_l855_85524


namespace NUMINAMATH_GPT_factorization_of_a_cubed_minus_a_l855_85504

variable (a : ℝ)

theorem factorization_of_a_cubed_minus_a : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_GPT_factorization_of_a_cubed_minus_a_l855_85504


namespace NUMINAMATH_GPT_truth_of_compound_proposition_l855_85597

def p := ∃ x : ℝ, x - 2 > Real.log x
def q := ∀ x : ℝ, x^2 > 0

theorem truth_of_compound_proposition : p ∧ ¬ q :=
by
  sorry

end NUMINAMATH_GPT_truth_of_compound_proposition_l855_85597


namespace NUMINAMATH_GPT_hypotenuse_of_right_triangle_l855_85535

theorem hypotenuse_of_right_triangle (h : height_dropped_to_hypotenuse = 1) (a : acute_angle = 15) :
∃ (hypotenuse : ℝ), hypotenuse = 4 :=
sorry

end NUMINAMATH_GPT_hypotenuse_of_right_triangle_l855_85535


namespace NUMINAMATH_GPT_length_of_one_side_of_regular_octagon_l855_85551

theorem length_of_one_side_of_regular_octagon
  (a b : ℕ)
  (h_pentagon : a = 16)   -- Side length of regular pentagon
  (h_total_yarn_pentagon : b = 80)  -- Total yarn for pentagon
  (hpentagon_yarn_length : 5 * a = b)  -- Total yarn condition
  (hoctagon_total_sides : 8 = 8)   -- Number of sides of octagon
  (hoctagon_side_length : 10 = b / 8)  -- Side length condition for octagon
  : 10 = 10 :=
by
  sorry

end NUMINAMATH_GPT_length_of_one_side_of_regular_octagon_l855_85551


namespace NUMINAMATH_GPT_find_max_n_l855_85574

variables {α : Type*} [LinearOrderedField α]

-- Define the sum S_n of the first n terms of an arithmetic sequence
noncomputable def S_n (a d : α) (n : ℕ) : α := 
  (n : α) / 2 * (2 * a + (n - 1) * d)

-- Given conditions
variable {a d : α}
axiom S11_pos : S_n a d 11 > 0
axiom S12_neg : S_n a d 12 < 0

theorem find_max_n : ∃ (n : ℕ), ∀ k < n, S_n a d k ≤ S_n a d n ∧ (k ≠ n → S_n a d k < S_n a d n) :=
sorry

end NUMINAMATH_GPT_find_max_n_l855_85574


namespace NUMINAMATH_GPT_Joey_weekend_study_hours_l855_85536

noncomputable def hours_weekday_per_week := 2 * 5 -- 2 hours/night * 5 nights/week
noncomputable def total_hours_weekdays := hours_weekday_per_week * 6 -- Multiply by 6 weeks
noncomputable def remaining_hours_weekends := 96 - total_hours_weekdays -- 96 total hours - weekday hours
noncomputable def total_weekend_days := 6 * 2 -- 6 weekends * 2 days/weekend
noncomputable def hours_per_day_weekend := remaining_hours_weekends / total_weekend_days

theorem Joey_weekend_study_hours : hours_per_day_weekend = 3 :=
by
  sorry

end NUMINAMATH_GPT_Joey_weekend_study_hours_l855_85536


namespace NUMINAMATH_GPT_bigger_part_of_sum_and_linear_combination_l855_85591

theorem bigger_part_of_sum_and_linear_combination (x y : ℕ) 
  (h1 : x + y = 24) 
  (h2 : 7 * x + 5 * y = 146) : x = 13 :=
by 
  sorry

end NUMINAMATH_GPT_bigger_part_of_sum_and_linear_combination_l855_85591


namespace NUMINAMATH_GPT_solve_for_y_l855_85537

theorem solve_for_y : ∀ y : ℚ, (8 * y^2 + 78 * y + 5) / (2 * y + 19) = 4 * y + 2 → y = -16.5 :=
by
  intro y
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_y_l855_85537


namespace NUMINAMATH_GPT_intersection_correct_l855_85530

def M : Set Int := {-1, 1, 3, 5}
def N : Set Int := {-3, 1, 5}

theorem intersection_correct : M ∩ N = {1, 5} := 
by 
    sorry

end NUMINAMATH_GPT_intersection_correct_l855_85530


namespace NUMINAMATH_GPT_math_problem_l855_85548

theorem math_problem (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 3) : a^(2008 : ℕ) + b^(2008 : ℕ) + c^(2008 : ℕ) = 3 :=
by 
  let h1' : a + b + c = 3 := h1
  let h2' : a^2 + b^2 + c^2 = 3 := h2
  sorry

end NUMINAMATH_GPT_math_problem_l855_85548


namespace NUMINAMATH_GPT_initial_lives_l855_85555

theorem initial_lives (x : ℕ) (h1 : x - 23 + 46 = 70) : x = 47 := 
by 
  sorry

end NUMINAMATH_GPT_initial_lives_l855_85555


namespace NUMINAMATH_GPT_Samantha_purse_value_l855_85533

def cents_per_penny := 1
def cents_per_nickel := 5
def cents_per_dime := 10
def cents_per_quarter := 25

def number_of_pennies := 2
def number_of_nickels := 1
def number_of_dimes := 3
def number_of_quarters := 2

def total_cents := 
  number_of_pennies * cents_per_penny + 
  number_of_nickels * cents_per_nickel + 
  number_of_dimes * cents_per_dime + 
  number_of_quarters * cents_per_quarter

def percent_of_dollar := (total_cents * 100) / 100

theorem Samantha_purse_value : percent_of_dollar = 87 := by
  sorry

end NUMINAMATH_GPT_Samantha_purse_value_l855_85533


namespace NUMINAMATH_GPT_largest_bucket_capacity_l855_85525

-- Let us define the initial conditions
def capacity_5_liter_bucket : ℕ := 5
def capacity_3_liter_bucket : ℕ := 3
def remaining_after_pour := capacity_5_liter_bucket - capacity_3_liter_bucket
def additional_capacity_without_overflow : ℕ := 4

-- Problem statement: Prove that the capacity of the largest bucket is 6 liters
theorem largest_bucket_capacity : ∀ (c : ℕ), remaining_after_pour + additional_capacity_without_overflow = c → c = 6 := 
by
  sorry

end NUMINAMATH_GPT_largest_bucket_capacity_l855_85525


namespace NUMINAMATH_GPT_susie_investment_l855_85516

theorem susie_investment :
  ∃ x : ℝ, x * (1 + 0.04)^3 + (2000 - x) * (1 + 0.06)^3 = 2436.29 → x = 820 :=
by
  sorry

end NUMINAMATH_GPT_susie_investment_l855_85516


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l855_85526

theorem roots_of_quadratic_eq (x : ℝ) : (x + 1) ^ 2 = 0 → x = -1 := by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l855_85526


namespace NUMINAMATH_GPT_person_B_age_l855_85500

variables (a b c d e f g : ℕ)

-- Conditions
axiom cond1 : a = b + 2
axiom cond2 : b = 2 * c
axiom cond3 : c = d / 2
axiom cond4 : d = e - 3
axiom cond5 : f = a * d
axiom cond6 : g = b + e
axiom cond7 : a + b + c + d + e + f + g = 292

-- Theorem statement
theorem person_B_age : b = 14 :=
sorry

end NUMINAMATH_GPT_person_B_age_l855_85500


namespace NUMINAMATH_GPT_annual_interest_rate_l855_85507

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  ((A / P) ^ (1 / t)) - 1

-- Define the given parameters
def P : ℝ := 1200
def A : ℝ := 2488.32
def n : ℕ := 1
def t : ℕ := 4

theorem annual_interest_rate : compound_interest_rate P A n t = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_annual_interest_rate_l855_85507


namespace NUMINAMATH_GPT_bells_toll_together_l855_85505

theorem bells_toll_together (a b c d : ℕ) (h1 : a = 5) (h2 : b = 8) (h3 : c = 11) (h4 : d = 15) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 1320 :=
by
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_bells_toll_together_l855_85505


namespace NUMINAMATH_GPT_ad_equals_two_l855_85593

noncomputable def geometric_sequence (a b c d : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c)

theorem ad_equals_two (a b c d : ℝ) 
  (h1 : geometric_sequence a b c d) 
  (h2 : ∃ (b c : ℝ), (1, 2) = (b, c) ∧ b = 1 ∧ c = 2) :
  a * d = 2 :=
by
  sorry

end NUMINAMATH_GPT_ad_equals_two_l855_85593


namespace NUMINAMATH_GPT_digit_B_condition_l855_85545

theorem digit_B_condition {B : ℕ} (h10 : ∃ d : ℕ, 58709310 = 10 * d)
  (h5 : ∃ e : ℕ, 58709310 = 5 * e)
  (h6 : ∃ f : ℕ, 58709310 = 6 * f)
  (h4 : ∃ g : ℕ, 58709310 = 4 * g)
  (h3 : ∃ h : ℕ, 58709310 = 3 * h)
  (h2 : ∃ i : ℕ, 58709310 = 2 * i) :
  B = 0 := by
  sorry

end NUMINAMATH_GPT_digit_B_condition_l855_85545


namespace NUMINAMATH_GPT_composite_has_at_least_three_factors_l855_85509

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem composite_has_at_least_three_factors (n : ℕ) (h : is_composite n) : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ∣ n ∧ b ∣ n ∧ c ∣ n :=
sorry

end NUMINAMATH_GPT_composite_has_at_least_three_factors_l855_85509


namespace NUMINAMATH_GPT_train_speed_correct_l855_85522

noncomputable def train_speed_kmh (length : ℝ) (time : ℝ) (conversion_factor : ℝ) : ℝ :=
  (length / time) * conversion_factor

theorem train_speed_correct 
  (length : ℝ := 350) 
  (time : ℝ := 8.7493) 
  (conversion_factor : ℝ := 3.6) : 
  train_speed_kmh length time conversion_factor = 144.02 := 
sorry

end NUMINAMATH_GPT_train_speed_correct_l855_85522


namespace NUMINAMATH_GPT_find_a_l855_85562

variable (f g : ℝ → ℝ) (a : ℝ)

-- Conditions
axiom h1 : ∀ x, f x = a^x * g x
axiom h2 : ∀ x, g x ≠ 0
axiom h3 : ∀ x, f x * (deriv g x) > (deriv f x) * g x

-- Question and target proof
theorem find_a (h4 : (f 1) / (g 1) + (f (-1)) / (g (-1)) = 5 / 2) : a = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_find_a_l855_85562


namespace NUMINAMATH_GPT_ramesh_paid_price_l855_85512

variable (P : ℝ) (P_paid : ℝ)

-- conditions
def discount_price (P : ℝ) : ℝ := 0.80 * P
def additional_cost : ℝ := 125 + 250
def total_cost_with_discount (P : ℝ) : ℝ := discount_price P + additional_cost
def selling_price_without_discount (P : ℝ) : ℝ := 1.10 * P
def given_selling_price : ℝ := 18975

-- the theorem to prove
theorem ramesh_paid_price :
  (∃ P : ℝ, selling_price_without_discount P = given_selling_price ∧ total_cost_with_discount P = 14175) :=
by
  sorry

end NUMINAMATH_GPT_ramesh_paid_price_l855_85512


namespace NUMINAMATH_GPT_min_omega_sin_two_max_l855_85557

theorem min_omega_sin_two_max (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → ∃ k : ℤ, (ω * x = (2 + 2 * k) * π)) →
  ∃ ω_min : ℝ, ω_min = 4 * π :=
by
  sorry

end NUMINAMATH_GPT_min_omega_sin_two_max_l855_85557


namespace NUMINAMATH_GPT_quadratic_binomial_form_l855_85529

theorem quadratic_binomial_form (y : ℝ) : ∃ (k : ℝ), y^2 + 14 * y + 40 = (y + 7)^2 + k :=
by
  use -9
  sorry

end NUMINAMATH_GPT_quadratic_binomial_form_l855_85529


namespace NUMINAMATH_GPT_wooden_parallelepiped_length_l855_85553

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end NUMINAMATH_GPT_wooden_parallelepiped_length_l855_85553


namespace NUMINAMATH_GPT_susan_ate_6_candies_l855_85552

-- Definitions based on the problem conditions
def candies_tuesday := 3
def candies_thursday := 5
def candies_friday := 2
def candies_left := 4

-- The total number of candies bought
def total_candies_bought := candies_tuesday + candies_thursday + candies_friday

-- The number of candies eaten
def candies_eaten := total_candies_bought - candies_left

-- Theorem statement to prove that Susan ate 6 candies
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_susan_ate_6_candies_l855_85552


namespace NUMINAMATH_GPT_negation_of_exists_l855_85595

theorem negation_of_exists (x : ℝ) (h : ∃ x : ℝ, x^2 - x + 1 ≤ 0) : 
  (∀ x : ℝ, x^2 - x + 1 > 0) :=
sorry

end NUMINAMATH_GPT_negation_of_exists_l855_85595


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l855_85587

theorem isosceles_triangle_perimeter : 
  ∀ a b c : ℝ, a^2 - 6 * a + 5 = 0 → b^2 - 6 * b + 5 = 0 → 
    (a = b ∨ b = c ∨ a = c) →
    (a + b > c ∧ b + c > a ∧ a + c > b) →
    a + b + c = 11 := 
by
  intros a b c ha hb hiso htri
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l855_85587


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l855_85579

theorem arithmetic_sequence_sum (b : ℕ → ℝ) (h_arith : ∀ n, b (n+1) - b n = b 2 - b 1) (h_b5 : b 5 = 2) :
  b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 = 18 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l855_85579


namespace NUMINAMATH_GPT_total_profit_calculation_l855_85531

variable (investment_Tom : ℝ) (investment_Jose : ℝ) (time_Jose : ℝ) (share_Jose : ℝ) (total_time : ℝ) 
variable (total_profit : ℝ)

theorem total_profit_calculation 
  (h1 : investment_Tom = 30000) 
  (h2 : investment_Jose = 45000) 
  (h3 : time_Jose = 10) -- Jose joined 2 months later, so he invested for 10 months out of 12
  (h4 : share_Jose = 30000) 
  (h5 : total_time = 12) 
  : total_profit = 54000 :=
sorry

end NUMINAMATH_GPT_total_profit_calculation_l855_85531


namespace NUMINAMATH_GPT_twice_product_of_numbers_l855_85518

theorem twice_product_of_numbers (x y : ℝ) (h1 : x + y = 80) (h2 : x - y = 10) : 2 * (x * y) = 3150 := by
  sorry

end NUMINAMATH_GPT_twice_product_of_numbers_l855_85518


namespace NUMINAMATH_GPT_fraction_comparison_l855_85508

theorem fraction_comparison :
  (1998:ℝ) ^ 2000 / (2000:ℝ) ^ 1998 > (1997:ℝ) ^ 1999 / (1999:ℝ) ^ 1997 :=
by sorry

end NUMINAMATH_GPT_fraction_comparison_l855_85508


namespace NUMINAMATH_GPT_right_angled_triangle_sets_l855_85514

theorem right_angled_triangle_sets :
  (¬ (1 ^ 2 + 2 ^ 2 = 3 ^ 2)) ∧
  (¬ (2 ^ 2 + 3 ^ 2 = 4 ^ 2)) ∧
  (3 ^ 2 + 4 ^ 2 = 5 ^ 2) ∧
  (¬ (4 ^ 2 + 5 ^ 2 = 6 ^ 2)) :=
by
  sorry

end NUMINAMATH_GPT_right_angled_triangle_sets_l855_85514


namespace NUMINAMATH_GPT_find_interval_for_inequality_l855_85560

open Set

theorem find_interval_for_inequality :
  {x : ℝ | (1 / (x^2 + 2) > 4 / x + 21 / 10)} = Ioo (-2 : ℝ) (0 : ℝ) := 
sorry

end NUMINAMATH_GPT_find_interval_for_inequality_l855_85560


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l855_85594

variable (n : ℕ) (a S : ℕ → ℕ)

theorem arithmetic_sequence_problem
  (h1 : a 2 + a 8 = 82)
  (h2 : S 41 = S 9)
  (hSn : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n, a n = 51 - 2 * n) ∧ (∀ n, S n ≤ 625) := sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l855_85594


namespace NUMINAMATH_GPT_shortest_altitude_triangle_l855_85523

/-- Given a triangle with sides 18, 24, and 30, prove that its shortest altitude is 18. -/
theorem shortest_altitude_triangle (a b c : ℝ) (h1 : a = 18) (h2 : b = 24) (h3 : c = 30) 
  (h_right : a ^ 2 + b ^ 2 = c ^ 2) : 
  exists h : ℝ, h = 18 :=
by
  sorry

end NUMINAMATH_GPT_shortest_altitude_triangle_l855_85523


namespace NUMINAMATH_GPT_watermelon_slices_l855_85582

theorem watermelon_slices (total_seeds slices_black seeds_white seeds_per_slice num_slices : ℕ)
  (h1 : seeds_black = 20)
  (h2 : seeds_white = 20)
  (h3 : seeds_per_slice = seeds_black + seeds_white)
  (h4 : total_seeds = 1600)
  (h5 : num_slices = total_seeds / seeds_per_slice) :
  num_slices = 40 :=
by
  sorry

end NUMINAMATH_GPT_watermelon_slices_l855_85582


namespace NUMINAMATH_GPT_trig_identity_l855_85539

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_trig_identity_l855_85539


namespace NUMINAMATH_GPT_remainder_division_of_product_l855_85583

theorem remainder_division_of_product
  (h1 : 1225 % 12 = 1)
  (h2 : 1227 % 12 = 3) :
  ((1225 * 1227 * 1) % 12) = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_division_of_product_l855_85583


namespace NUMINAMATH_GPT_solution_l855_85599

theorem solution (a b : ℝ) (h1 : a^2 + 2 * a - 2016 = 0) (h2 : b^2 + 2 * b - 2016 = 0) :
  a^2 + 3 * a + b = 2014 := 
sorry

end NUMINAMATH_GPT_solution_l855_85599


namespace NUMINAMATH_GPT_find_b_if_even_function_l855_85586

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b * x + c

theorem find_b_if_even_function (h : ∀ x : ℝ, f (-x) = f (x)) : b = 0 := by
  sorry

end NUMINAMATH_GPT_find_b_if_even_function_l855_85586


namespace NUMINAMATH_GPT_probability_not_black_l855_85542

theorem probability_not_black (white_balls black_balls red_balls : ℕ) (total_balls : ℕ) (non_black_balls : ℕ) :
  white_balls = 7 → black_balls = 6 → red_balls = 4 →
  total_balls = white_balls + black_balls + red_balls →
  non_black_balls = white_balls + red_balls →
  (non_black_balls / total_balls : ℚ) = 11 / 17 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_black_l855_85542


namespace NUMINAMATH_GPT_total_fencing_cost_is_5300_l855_85588

-- Define the conditions
def length_more_than_breadth_condition (l b : ℕ) := l = b + 40
def fencing_cost_per_meter : ℝ := 26.50
def given_length : ℕ := 70

-- Define the perimeter calculation
def perimeter (l b : ℕ) := 2 * l + 2 * b

-- Define the total cost calculation
def total_cost (P : ℕ) (cost_per_meter : ℝ) := P * cost_per_meter

-- State the theorem to be proven
theorem total_fencing_cost_is_5300 (b : ℕ) (l := given_length) :
  length_more_than_breadth_condition l b →
  total_cost (perimeter l b) fencing_cost_per_meter = 5300 :=
by
  sorry

end NUMINAMATH_GPT_total_fencing_cost_is_5300_l855_85588


namespace NUMINAMATH_GPT_f_cos_x_l855_85567

theorem f_cos_x (f : ℝ → ℝ) (x : ℝ) (h : f (Real.sin x) = 2 - Real.cos x ^ 2) : f (Real.cos x) = 2 + Real.sin x ^ 2 := by
  sorry

end NUMINAMATH_GPT_f_cos_x_l855_85567


namespace NUMINAMATH_GPT_ads_ratio_l855_85511

theorem ads_ratio 
  (first_ads : ℕ := 12)
  (second_ads : ℕ)
  (third_ads := second_ads + 24)
  (fourth_ads := (3 / 4) * second_ads)
  (clicked_ads := 68)
  (total_ads := (3 / 2) * clicked_ads == 102)
  (ads_eq : first_ads + second_ads + third_ads + fourth_ads = total_ads) :
  second_ads / first_ads = 2 :=
by sorry

end NUMINAMATH_GPT_ads_ratio_l855_85511


namespace NUMINAMATH_GPT_fraction_equality_l855_85566

theorem fraction_equality (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 := 
sorry

end NUMINAMATH_GPT_fraction_equality_l855_85566


namespace NUMINAMATH_GPT_function_positive_for_x_gt_neg1_l855_85556

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (3*x^2 + 6*x + 9)

theorem function_positive_for_x_gt_neg1 : ∀ (x : ℝ), x > -1 → f x > 0.5 :=
by
  sorry

end NUMINAMATH_GPT_function_positive_for_x_gt_neg1_l855_85556


namespace NUMINAMATH_GPT_algae_difference_l855_85527

theorem algae_difference :
  let original_algae := 809
  let current_algae := 3263
  current_algae - original_algae = 2454 :=
by
  sorry

end NUMINAMATH_GPT_algae_difference_l855_85527


namespace NUMINAMATH_GPT_toys_calculation_l855_85565

-- Define the number of toys each person has as variables
variables (Jason John Rachel : ℕ)

-- State the conditions
variables (h1 : Jason = 3 * John)
variables (h2 : John = Rachel + 6)
variables (h3 : Jason = 21)

-- Define the theorem to prove the number of toys Rachel has
theorem toys_calculation : Rachel = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_toys_calculation_l855_85565


namespace NUMINAMATH_GPT_correct_calculation_l855_85569

variable (a : ℝ) -- assuming a ∈ ℝ

theorem correct_calculation : (a ^ 3) ^ 2 = a ^ 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_calculation_l855_85569
