import Mathlib

namespace NUMINAMATH_GPT_circumscribed_sphere_radius_l906_90643

noncomputable def radius_of_circumscribed_sphere (a : ℝ) (α : ℝ) : ℝ :=
  a / (3 * Real.sin α)

theorem circumscribed_sphere_radius (a α : ℝ) :
  radius_of_circumscribed_sphere a α = a / (3 * Real.sin α) :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_sphere_radius_l906_90643


namespace NUMINAMATH_GPT_boys_from_school_A_study_science_l906_90651

theorem boys_from_school_A_study_science (total_boys school_A_percent non_science_boys school_A_boys study_science_boys: ℕ) 
(h1 : total_boys = 300)
(h2 : school_A_percent = 20)
(h3 : non_science_boys = 42)
(h4 : school_A_boys = (school_A_percent * total_boys) / 100)
(h5 : study_science_boys = school_A_boys - non_science_boys) :
(study_science_boys * 100 / school_A_boys) = 30 :=
by
  sorry

end NUMINAMATH_GPT_boys_from_school_A_study_science_l906_90651


namespace NUMINAMATH_GPT_part1_equation_part2_equation_l906_90631

-- Part (Ⅰ)
theorem part1_equation :
  (- ((-1) ^ 1000) - 2.45 * 8 + 2.55 * (-8) = -41) :=
by
  sorry

-- Part (Ⅱ)
theorem part2_equation :
  ((1 / 6 - 1 / 3 + 0.25) / (- (1 / 12)) = -1) :=
by
  sorry

end NUMINAMATH_GPT_part1_equation_part2_equation_l906_90631


namespace NUMINAMATH_GPT_regular_polygon_sides_l906_90674

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 135) : n = 8 := 
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l906_90674


namespace NUMINAMATH_GPT_eating_contest_l906_90697

variables (hotdog_weight burger_weight pie_weight : ℕ)
variable (noah_burgers jacob_pies mason_hotdogs : ℕ)
variable (total_weight_mason_hotdogs : ℕ)

theorem eating_contest :
  hotdog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  noah_burgers = 8 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  total_weight_mason_hotdogs = mason_hotdogs * hotdog_weight →
  total_weight_mason_hotdogs = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_eating_contest_l906_90697


namespace NUMINAMATH_GPT_george_purchased_two_large_pizzas_l906_90611

noncomputable def small_slices := 4
noncomputable def large_slices := 8
noncomputable def small_pizzas_purchased := 3
noncomputable def george_slices := 3
noncomputable def bob_slices := george_slices + 1
noncomputable def susie_slices := bob_slices / 2
noncomputable def bill_slices := 3
noncomputable def fred_slices := 3
noncomputable def mark_slices := 3
noncomputable def leftover_slices := 10

noncomputable def total_slices_consumed := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

noncomputable def total_slices_before_eating := total_slices_consumed + leftover_slices

noncomputable def small_pizza_total_slices := small_pizzas_purchased * small_slices

noncomputable def large_pizza_total_slices := total_slices_before_eating - small_pizza_total_slices

noncomputable def large_pizzas_purchased := large_pizza_total_slices / large_slices

theorem george_purchased_two_large_pizzas : large_pizzas_purchased = 2 :=
sorry

end NUMINAMATH_GPT_george_purchased_two_large_pizzas_l906_90611


namespace NUMINAMATH_GPT_max_number_of_eligible_ages_l906_90658

-- Definitions based on the problem conditions
def average_age : ℝ := 31
def std_dev : ℝ := 5
def acceptable_age_range (a : ℝ) : Prop := 26 ≤ a ∧ a ≤ 36
def has_masters_degree : Prop := 24 ≤ 26  -- simplified for context indicated in problem
def has_work_experience : Prop := 26 ≥ 26

-- Define the maximum number of different ages of the eligible applicants
noncomputable def max_diff_ages : ℕ := 36 - 26 + 1  -- This matches the solution step directly

-- The theorem stating the result
theorem max_number_of_eligible_ages :
  max_diff_ages = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_number_of_eligible_ages_l906_90658


namespace NUMINAMATH_GPT_sum_of_common_ratios_is_five_l906_90630

theorem sum_of_common_ratios_is_five {k p r : ℝ} 
  (h1 : p ≠ r)                       -- different common ratios
  (h2 : k ≠ 0)                       -- non-zero k
  (a2 : ℝ := k * p)                  -- term a2
  (a3 : ℝ := k * p^2)                -- term a3
  (b2 : ℝ := k * r)                  -- term b2
  (b3 : ℝ := k * r^2)                -- term b3
  (h3 : a3 - b3 = 5 * (a2 - b2))     -- given condition
  : p + r = 5 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_common_ratios_is_five_l906_90630


namespace NUMINAMATH_GPT_recycling_target_l906_90638

/-- Six Grade 4 sections launched a recycling drive where they collect old newspapers to recycle.
Each section collected 280 kilos in two weeks. After the third week, they found that they need 320 kilos more to reach their target.
  How many kilos of the newspaper is their target? -/
theorem recycling_target (sections : ℕ) (kilos_collected_2_weeks : ℕ) (additional_kilos : ℕ) : 
  sections = 6 ∧ kilos_collected_2_weeks = 280 ∧ additional_kilos = 320 → 
  (sections * (kilos_collected_2_weeks / 2) * 3 + additional_kilos) = 2840 :=
by
  sorry

end NUMINAMATH_GPT_recycling_target_l906_90638


namespace NUMINAMATH_GPT_sqrt_164_between_12_and_13_l906_90669

theorem sqrt_164_between_12_and_13 : 12 < Real.sqrt 164 ∧ Real.sqrt 164 < 13 :=
sorry

end NUMINAMATH_GPT_sqrt_164_between_12_and_13_l906_90669


namespace NUMINAMATH_GPT_num_positive_four_digit_integers_of_form_xx75_l906_90646

theorem num_positive_four_digit_integers_of_form_xx75 : 
  ∃ n : ℕ, n = 90 ∧ ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → (∃ x: ℕ, x = 1000 * a + 100 * b + 75 ∧ 1000 ≤ x ∧ x < 10000) → n = 90 :=
sorry

end NUMINAMATH_GPT_num_positive_four_digit_integers_of_form_xx75_l906_90646


namespace NUMINAMATH_GPT_gcd_of_324_and_135_l906_90679

theorem gcd_of_324_and_135 : Nat.gcd 324 135 = 27 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_324_and_135_l906_90679


namespace NUMINAMATH_GPT_largest_square_plots_l906_90657

theorem largest_square_plots (width length pathway_material : Nat) (width_eq : width = 30) (length_eq : length = 60) (pathway_material_eq : pathway_material = 2010) : ∃ (n : Nat), n * (2 * n) = 578 := 
by
  sorry

end NUMINAMATH_GPT_largest_square_plots_l906_90657


namespace NUMINAMATH_GPT_coefficient_x_neg_4_expansion_l906_90615

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the function to calculate the coefficient of the term containing x^(-4)
def coeff_term_x_neg_4 : ℕ :=
  let k := 10
  binom 12 k

theorem coefficient_x_neg_4_expansion :
  coeff_term_x_neg_4 = 66 := by
  -- Calculation here would show that binom 12 10 is indeed 66
  sorry

end NUMINAMATH_GPT_coefficient_x_neg_4_expansion_l906_90615


namespace NUMINAMATH_GPT_ratio_equivalence_l906_90655

theorem ratio_equivalence (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : x ≠ z)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h : y / (x - z) = (x + 2 * y) / z ∧ (x + 2 * y) / z = x / (y + z)) :
  x / (y + z) = (2 * y - z) / (y + z) :=
by
  sorry

end NUMINAMATH_GPT_ratio_equivalence_l906_90655


namespace NUMINAMATH_GPT_union_of_A_and_B_l906_90641

-- Condition definitions
def A : Set ℝ := {x : ℝ | abs (x - 3) < 2}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 0}

-- The theorem we need to prove
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 5} :=
by
  -- This is where the proof would go if it were required
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l906_90641


namespace NUMINAMATH_GPT_average_of_multiples_of_9_l906_90661

-- Define the problem in Lean
theorem average_of_multiples_of_9 :
  let pos_multiples := [9, 18, 27, 36, 45]
  let neg_multiples := [-9, -18, -27, -36, -45]
  (pos_multiples.sum + neg_multiples.sum) / 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_average_of_multiples_of_9_l906_90661


namespace NUMINAMATH_GPT_divide_and_add_l906_90684

variable (number : ℝ)

theorem divide_and_add (h : 4 * number = 166.08) : number / 4 + 0.48 = 10.86 := by
  -- assume the proof follows accurately
  sorry

end NUMINAMATH_GPT_divide_and_add_l906_90684


namespace NUMINAMATH_GPT_hyperbola_trajectory_center_l906_90600

theorem hyperbola_trajectory_center :
  ∀ m : ℝ, ∃ (x y : ℝ), x^2 - y^2 - 6 * m * x - 4 * m * y + 5 * m^2 - 1 = 0 ∧ 2 * x + 3 * y = 0 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_trajectory_center_l906_90600


namespace NUMINAMATH_GPT_jinsu_work_per_hour_l906_90692

theorem jinsu_work_per_hour (t : ℝ) (h : t = 4) : (1 / t = 1 / 4) :=
by {
    sorry
}

end NUMINAMATH_GPT_jinsu_work_per_hour_l906_90692


namespace NUMINAMATH_GPT_joan_gave_28_seashells_to_sam_l906_90650

/-- 
Given:
- Joan found 70 seashells on the beach.
- After giving away some seashells, she has 27 left.
- She gave twice as many seashells to Sam as she gave to her friend Lily.

Show that:
- Joan gave 28 seashells to Sam.
-/
theorem joan_gave_28_seashells_to_sam (L S : ℕ) 
  (h1 : S = 2 * L) 
  (h2 : 70 - 27 = 43) 
  (h3 : L + S = 43) :
  S = 28 :=
by
  sorry

end NUMINAMATH_GPT_joan_gave_28_seashells_to_sam_l906_90650


namespace NUMINAMATH_GPT_parabola_intersection_difference_l906_90653

noncomputable def parabola1 (x : ℝ) := 3 * x^2 - 6 * x + 6
noncomputable def parabola2 (x : ℝ) := -2 * x^2 + 2 * x + 6

theorem parabola_intersection_difference :
  let a := 0
  let c := 8 / 5
  c - a = 8 / 5 := by
  sorry

end NUMINAMATH_GPT_parabola_intersection_difference_l906_90653


namespace NUMINAMATH_GPT_last_three_digits_of_7_to_50_l906_90642

theorem last_three_digits_of_7_to_50 : (7^50) % 1000 = 991 := 
by 
  sorry

end NUMINAMATH_GPT_last_three_digits_of_7_to_50_l906_90642


namespace NUMINAMATH_GPT_problem_statement_l906_90622

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : y - x > 1) :
  (1 - y) / x < 1 ∨ (1 + 3 * x) / y < 1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l906_90622


namespace NUMINAMATH_GPT_total_runs_of_a_b_c_l906_90681

/-- Suppose a, b, and c are the runs scored by three players in a cricket match. The ratios of the runs are given as a : b = 1 : 3 and b : c = 1 : 5. Additionally, c scored 75 runs. Prove that the total runs scored by all of them is 95. -/
theorem total_runs_of_a_b_c (a b c : ℕ) (h1 : a * 3 = b) (h2 : b * 5 = c) (h3 : c = 75) : a + b + c = 95 := 
by sorry

end NUMINAMATH_GPT_total_runs_of_a_b_c_l906_90681


namespace NUMINAMATH_GPT_equation_of_tangent_line_l906_90670

-- Definitions for the given conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x
def P : ℝ × ℝ := (-1, 4)
def slope_of_tangent (a : ℝ) (x : ℝ) : ℝ := -6 * x^2 - 2

-- The main theorem to prove the equation of the tangent line
theorem equation_of_tangent_line (a : ℝ) (ha : f a (-1) = 4) :
  8 * x + y + 4 = 0 := by
  sorry

end NUMINAMATH_GPT_equation_of_tangent_line_l906_90670


namespace NUMINAMATH_GPT_center_of_circle_l906_90673

theorem center_of_circle (x y : ℝ) : 
    (∃ x y : ℝ, x^2 + y^2 = 4*x - 6*y + 9) → (x, y) = (2, -3) := 
by sorry

end NUMINAMATH_GPT_center_of_circle_l906_90673


namespace NUMINAMATH_GPT_dihedral_angle_proof_l906_90606

noncomputable def angle_between_planes 
  (α β : Real) : Real :=
  Real.arcsin (Real.sin α * Real.sin β)

theorem dihedral_angle_proof 
  (α β : Real) 
  (α_non_neg : 0 ≤ α) 
  (α_non_gtr : α ≤ Real.pi / 2) 
  (β_non_neg : 0 ≤ β) 
  (β_non_gtr : β ≤ Real.pi / 2) :
  angle_between_planes α β = Real.arcsin (Real.sin α * Real.sin β) :=
by
  sorry

end NUMINAMATH_GPT_dihedral_angle_proof_l906_90606


namespace NUMINAMATH_GPT_max_value_2x_plus_y_l906_90628

theorem max_value_2x_plus_y (x y : ℝ) (h : y^2 / 4 + x^2 / 3 = 1) : 2 * x + y ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_max_value_2x_plus_y_l906_90628


namespace NUMINAMATH_GPT_distribution_count_l906_90695

def num_distributions (novels poetry students : ℕ) : ℕ :=
  -- This is where the formula for counting would go, but we'll just define it as sorry for now
  sorry

theorem distribution_count : num_distributions 3 2 4 = 28 :=
by
  sorry

end NUMINAMATH_GPT_distribution_count_l906_90695


namespace NUMINAMATH_GPT_original_amount_of_money_l906_90648

-- Define the conditions
variables (x : ℕ) -- daily allowance

-- Spending details
def spend_10_days := 6 * 10 - 6 * x
def spend_15_days := 15 * 3 - 3 * x

-- Lean proof statement
theorem original_amount_of_money (h : spend_10_days = spend_15_days) : (6 * 10 - 6 * x) = 30 :=
by
  sorry

end NUMINAMATH_GPT_original_amount_of_money_l906_90648


namespace NUMINAMATH_GPT_ratio_of_squares_l906_90672

theorem ratio_of_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a / b = 1 / 3) :
  (4 * a / (4 * b) = 1 / 3) ∧ (a * a / (b * b) = 1 / 9) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_squares_l906_90672


namespace NUMINAMATH_GPT_toys_produced_each_day_l906_90675

-- Define the conditions
def total_weekly_production : ℕ := 8000
def days_worked_per_week : ℕ := 4
def daily_production : ℕ := total_weekly_production / days_worked_per_week

-- The statement to be proved
theorem toys_produced_each_day : daily_production = 2000 := sorry

end NUMINAMATH_GPT_toys_produced_each_day_l906_90675


namespace NUMINAMATH_GPT_largest_prime_factor_of_891_l906_90639

theorem largest_prime_factor_of_891 : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ 891 ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ 891 → q ≤ p :=
by
  sorry

end NUMINAMATH_GPT_largest_prime_factor_of_891_l906_90639


namespace NUMINAMATH_GPT_distinct_numbers_div_sum_diff_l906_90649

theorem distinct_numbers_div_sum_diff (n : ℕ) : 
  ∃ (numbers : Fin n → ℕ), 
    ∀ i j, i ≠ j → (numbers i + numbers j) % (numbers i - numbers j) = 0 := 
by
  sorry

end NUMINAMATH_GPT_distinct_numbers_div_sum_diff_l906_90649


namespace NUMINAMATH_GPT_y_value_l906_90671

def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

theorem y_value (x y : ℤ) (h1 : star 5 0 2 (-2) = (3, -2)) (h2 : star x y 0 3 = (3, -2)) :
  y = -5 :=
sorry

end NUMINAMATH_GPT_y_value_l906_90671


namespace NUMINAMATH_GPT_watch_loss_percentage_l906_90685

noncomputable def initial_loss_percentage : ℝ :=
  let CP := 350
  let SP_new := 364
  let delta_SP := 140
  show ℝ from 
  sorry

theorem watch_loss_percentage (CP SP_new delta_SP : ℝ) (h₁ : CP = 350)
  (h₂ : SP_new = 364) (h₃ : delta_SP = 140) : 
  initial_loss_percentage = 36 :=
by
  -- Use the hypothesis and solve the corresponding problem
  sorry

end NUMINAMATH_GPT_watch_loss_percentage_l906_90685


namespace NUMINAMATH_GPT_calculate_expression_l906_90691

theorem calculate_expression (p q : ℝ) (hp : p + q = 7) (hq : p * q = 12) :
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 3691 := 
by sorry

end NUMINAMATH_GPT_calculate_expression_l906_90691


namespace NUMINAMATH_GPT_advance_tickets_sold_20_l906_90612

theorem advance_tickets_sold_20 :
  ∃ (A S : ℕ), 20 * A + 30 * S = 1600 ∧ A + S = 60 ∧ A = 20 :=
by
  sorry

end NUMINAMATH_GPT_advance_tickets_sold_20_l906_90612


namespace NUMINAMATH_GPT_correct_difference_is_nine_l906_90699

-- Define the conditions
def misunderstood_number : ℕ := 35
def actual_number : ℕ := 53
def incorrect_difference : ℕ := 27

-- Define the two-digit number based on Yoongi's incorrect calculation
def original_number : ℕ := misunderstood_number + incorrect_difference

-- State the theorem
theorem correct_difference_is_nine : (original_number - actual_number) = 9 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_correct_difference_is_nine_l906_90699


namespace NUMINAMATH_GPT_simplify_and_substitute_l906_90654

theorem simplify_and_substitute (x : ℝ) (h1 : x ≠ 1) (h3 : x ≠ 3) : 
  ((1 - (2 / (x - 1))) * ((x^2 - x) / (x^2 - 6*x + 9))) = (x / (x - 3)) ∧ 
  (2 / (2 - 3)) = -2 := by
  sorry

end NUMINAMATH_GPT_simplify_and_substitute_l906_90654


namespace NUMINAMATH_GPT_divides_y_l906_90644

theorem divides_y
  (x y : ℤ)
  (h1 : 2 * x + 1 ∣ 8 * y) : 
  2 * x + 1 ∣ y :=
sorry

end NUMINAMATH_GPT_divides_y_l906_90644


namespace NUMINAMATH_GPT_convert_to_rectangular_form_l906_90688

noncomputable def θ : ℝ := 15 * Real.pi / 2

noncomputable def EulerFormula (θ : ℝ) : ℂ := Complex.exp (Complex.I * θ)

theorem convert_to_rectangular_form : EulerFormula θ = Complex.I := by
  sorry

end NUMINAMATH_GPT_convert_to_rectangular_form_l906_90688


namespace NUMINAMATH_GPT_average_expenditure_week_l906_90640

theorem average_expenditure_week (avg_3_days: ℝ) (avg_4_days: ℝ) (total_days: ℝ):
  avg_3_days = 350 → avg_4_days = 420 → total_days = 7 → 
  ((3 * avg_3_days + 4 * avg_4_days) / total_days = 390) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_average_expenditure_week_l906_90640


namespace NUMINAMATH_GPT_expand_expression_l906_90626

theorem expand_expression (x : ℝ) : (x + 3) * (4 * x - 8) - 2 * x = 4 * x^2 + 2 * x - 24 := by
  sorry

end NUMINAMATH_GPT_expand_expression_l906_90626


namespace NUMINAMATH_GPT_sum_of_digits_of_A15B94_multiple_of_99_l906_90632

theorem sum_of_digits_of_A15B94_multiple_of_99 (A B : ℕ) 
  (hA : A < 10) (hB : B < 10)
  (h_mult_99 : ∃ n : ℕ, (100000 * A + 10000 + 5000 + 100 * B + 90 + 4) = 99 * n) :
  A + B = 8 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_A15B94_multiple_of_99_l906_90632


namespace NUMINAMATH_GPT_four_digit_square_l906_90677

/-- A four-digit square number that satisfies the given conditions -/
theorem four_digit_square (a b c d : ℕ) (h₁ : b + c = a) (h₂ : a + c = 10 * d) :
  1000 * a + 100 * b + 10 * c + d = 6241 :=
sorry

end NUMINAMATH_GPT_four_digit_square_l906_90677


namespace NUMINAMATH_GPT_probability_of_log2N_is_integer_and_N_is_even_l906_90667

-- Defining the range of N as a four-digit number in base four
def is_base4_four_digit (N : ℕ) : Prop := 64 ≤ N ∧ N ≤ 255

-- Defining the condition that log_2 N is an integer
def is_power_of_two (N : ℕ) : Prop := ∃ k : ℕ, N = 2^k

-- Defining the condition that N is even
def is_even (N : ℕ) : Prop := N % 2 = 0

-- Combining all conditions
def meets_conditions (N : ℕ) : Prop := is_base4_four_digit N ∧ is_power_of_two N ∧ is_even N

-- Total number of four-digit numbers in base four
def total_base4_four_digits : ℕ := 192

-- Set of N values that meet the conditions
def valid_N_values : Finset ℕ := {64, 128}

-- The probability calculation
def calculated_probability : ℚ := valid_N_values.card / total_base4_four_digits

-- The final proof statement
theorem probability_of_log2N_is_integer_and_N_is_even : calculated_probability = 1 / 96 :=
by
  -- Prove the equality here (matching the solution given)
  sorry

end NUMINAMATH_GPT_probability_of_log2N_is_integer_and_N_is_even_l906_90667


namespace NUMINAMATH_GPT_product_of_consecutive_sums_not_eq_111111111_l906_90613

theorem product_of_consecutive_sums_not_eq_111111111 :
  ∀ (a : ℤ), (3 * a + 3) * (3 * a + 12) ≠ 111111111 := 
by
  intros a
  sorry

end NUMINAMATH_GPT_product_of_consecutive_sums_not_eq_111111111_l906_90613


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_l906_90660

theorem arithmetic_sequence_a6 (a : ℕ → ℝ)
  (h4_8 : ∃ a4 a8, (a 4 = a4) ∧ (a 8 = a8) ∧ a4^2 - 6*a4 + 5 = 0 ∧ a8^2 - 6*a8 + 5 = 0) :
  a 6 = 3 := by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_l906_90660


namespace NUMINAMATH_GPT_wool_usage_l906_90652

def total_balls_of_wool_used (scarves_aaron sweaters_aaron sweaters_enid : ℕ) (wool_per_scarf wool_per_sweater : ℕ) : ℕ :=
  (scarves_aaron * wool_per_scarf) + (sweaters_aaron * wool_per_sweater) + (sweaters_enid * wool_per_sweater)

theorem wool_usage :
  total_balls_of_wool_used 10 5 8 3 4 = 82 :=
by
  -- calculations done in solution steps
  -- total_balls_of_wool_used (10 scarves * 3 balls/scarf) + (5 sweaters * 4 balls/sweater) + (8 sweaters * 4 balls/sweater)
  -- total_balls_of_wool_used (30) + (20) + (32)
  -- total_balls_of_wool_used = 30 + 20 + 32 = 82
  sorry

end NUMINAMATH_GPT_wool_usage_l906_90652


namespace NUMINAMATH_GPT_fraction_eq_l906_90618

theorem fraction_eq {x : ℝ} (h : 1 - 6 / x + 9 / x ^ 2 - 2 / x ^ 3 = 0) :
  3 / x = 3 / 2 ∨ 3 / x = 3 / (2 + Real.sqrt 3) ∨ 3 / x = 3 / (2 - Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_fraction_eq_l906_90618


namespace NUMINAMATH_GPT_quadratic_has_solution_zero_l906_90637

theorem quadratic_has_solution_zero (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 + 3 * x + k^2 - 4 = 0) →
  ((k - 2) ≠ 0) → k = -2 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_has_solution_zero_l906_90637


namespace NUMINAMATH_GPT_calculation_l906_90620

theorem calculation :
  7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end NUMINAMATH_GPT_calculation_l906_90620


namespace NUMINAMATH_GPT_ice_cream_melt_time_l906_90645

theorem ice_cream_melt_time :
  let blocks := 16
  let block_length := 1.0/8.0 -- miles per block
  let distance := blocks * block_length -- in miles
  let speed := 12.0 -- miles per hour
  let time := distance / speed -- in hours
  let time_in_minutes := time * 60 -- converted to minutes
  time_in_minutes = 10 := by sorry

end NUMINAMATH_GPT_ice_cream_melt_time_l906_90645


namespace NUMINAMATH_GPT_regression_line_l906_90614

theorem regression_line (m x1 y1 : ℝ) (h_slope : m = 1.23) (h_center : (x1, y1) = (4, 5)) :
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ y = 1.23 * x + 0.08) :=
by
  use 0.08
  sorry

end NUMINAMATH_GPT_regression_line_l906_90614


namespace NUMINAMATH_GPT_base_b_square_of_15_l906_90656

theorem base_b_square_of_15 (b : ℕ) (h : (b + 5) * (b + 5) = 4 * b^2 + 3 * b + 6) : b = 8 :=
sorry

end NUMINAMATH_GPT_base_b_square_of_15_l906_90656


namespace NUMINAMATH_GPT_union_of_sets_l906_90629

theorem union_of_sets (x y : ℕ) (A B : Set ℕ) (hA : A = {x, y}) (hB : B = {x + 1, 5}) (h_inter : A ∩ B = {2}) :
  A ∪ B = {1, 2, 5} :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l906_90629


namespace NUMINAMATH_GPT_floor_diff_bounds_l906_90605

theorem floor_diff_bounds (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  0 ≤ Int.floor (a + b) - (Int.floor a + Int.floor b) ∧ 
  Int.floor (a + b) - (Int.floor a + Int.floor b) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_floor_diff_bounds_l906_90605


namespace NUMINAMATH_GPT_find_volume_of_12_percent_solution_l906_90665

variable (x y : ℝ)

theorem find_volume_of_12_percent_solution
  (h1 : x + y = 60)
  (h2 : 0.02 * x + 0.12 * y = 3) :
  y = 18 := 
sorry

end NUMINAMATH_GPT_find_volume_of_12_percent_solution_l906_90665


namespace NUMINAMATH_GPT_roots_operation_zero_l906_90604

def operation (a b : ℝ) : ℝ := a * b - a - b

theorem roots_operation_zero {x1 x2 : ℝ}
  (h1 : x1 + x2 = -1)
  (h2 : x1 * x2 = -1) :
  operation x1 x2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_roots_operation_zero_l906_90604


namespace NUMINAMATH_GPT_lizard_ratio_l906_90608

def lizard_problem (W S : ℕ) : Prop :=
  (S = 7 * W) ∧ (3 = S + W - 69) ∧ (W / 3 = 3)

theorem lizard_ratio (W S : ℕ) (h : lizard_problem W S) : W / 3 = 3 :=
  by
    rcases h with ⟨h1, h2, h3⟩
    exact h3

end NUMINAMATH_GPT_lizard_ratio_l906_90608


namespace NUMINAMATH_GPT_ninth_term_of_sequence_is_4_l906_90623

-- Definition of the first term and common ratio
def a1 : ℚ := 4
def r : ℚ := 1

-- Definition of the nth term of a geometric sequence
def a (n : ℕ) : ℚ := a1 * r^(n-1)

-- Proof that the ninth term of the sequence is 4
theorem ninth_term_of_sequence_is_4 : a 9 = 4 := by
  sorry

end NUMINAMATH_GPT_ninth_term_of_sequence_is_4_l906_90623


namespace NUMINAMATH_GPT_evaluate_fraction_l906_90621

theorem evaluate_fraction : (5 / 6 : ℚ) / (9 / 10) - 1 = -2 / 27 := by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l906_90621


namespace NUMINAMATH_GPT_freq_count_of_third_group_l906_90609

theorem freq_count_of_third_group
  (sample_size : ℕ) 
  (freq_third_group : ℝ) 
  (h1 : sample_size = 100) 
  (h2 : freq_third_group = 0.2) : 
  (sample_size * freq_third_group) = 20 :=
by 
  sorry

end NUMINAMATH_GPT_freq_count_of_third_group_l906_90609


namespace NUMINAMATH_GPT_range_of_m_l906_90693

variable (m : ℝ)

def prop_p : Prop := ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + m*x1 + 1 = 0) ∧ (x2^2 + m*x2 + 1 = 0)

def prop_q : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem range_of_m (h₁ : prop_p m) (h₂ : ¬prop_q m) : m < -2 ∨ m ≥ 3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l906_90693


namespace NUMINAMATH_GPT_sum_of_three_digit_numbers_l906_90625

theorem sum_of_three_digit_numbers (a b c : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
  222 * (a + b + c) ≠ 2021 := 
sorry

end NUMINAMATH_GPT_sum_of_three_digit_numbers_l906_90625


namespace NUMINAMATH_GPT_new_concentration_l906_90602

def vessel1 := (3 : ℝ)  -- 3 litres
def conc1 := (0.25 : ℝ) -- 25% alcohol

def vessel2 := (5 : ℝ)  -- 5 litres
def conc2 := (0.40 : ℝ) -- 40% alcohol

def vessel3 := (7 : ℝ)  -- 7 litres
def conc3 := (0.60 : ℝ) -- 60% alcohol

def vessel4 := (4 : ℝ)  -- 4 litres
def conc4 := (0.15 : ℝ) -- 15% alcohol

def total_volume := (25 : ℝ) -- Total vessel capacity

noncomputable def alcohol_total : ℝ :=
  (vessel1 * conc1) + (vessel2 * conc2) + (vessel3 * conc3) + (vessel4 * conc4)

theorem new_concentration : (alcohol_total / total_volume = 0.302) :=
  sorry

end NUMINAMATH_GPT_new_concentration_l906_90602


namespace NUMINAMATH_GPT_square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive_l906_90610

theorem square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive :
  (∀ n : ℝ, 0 ≤ n → n^2 ≤ 0 → False) ↔ (∀ m : ℝ, m < 0 → m^2 ≤ 0) := 
sorry

end NUMINAMATH_GPT_square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive_l906_90610


namespace NUMINAMATH_GPT_cos_double_angle_at_origin_l906_90678

noncomputable def vertex : ℝ × ℝ := (0, 0)
noncomputable def initial_side : ℝ × ℝ := (1, 0)
noncomputable def terminal_side : ℝ × ℝ := (-1, 3)
noncomputable def cos2alpha (v i t : ℝ × ℝ) : ℝ :=
  2 * ((t.1) / (Real.sqrt (t.1 ^ 2 + t.2 ^ 2))) ^ 2 - 1

theorem cos_double_angle_at_origin :
  cos2alpha vertex initial_side terminal_side = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_at_origin_l906_90678


namespace NUMINAMATH_GPT_min_pairs_with_same_sum_l906_90682

theorem min_pairs_with_same_sum (n : ℕ) (h1 : n > 0) :
  (∀ weights : Fin n → ℕ, (∀ i, weights i ≤ 21) → (∃ i j k l : Fin n,
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    weights i + weights j = weights k + weights l)) ↔ n ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_min_pairs_with_same_sum_l906_90682


namespace NUMINAMATH_GPT_anglet_angle_measurement_l906_90663

-- Definitions based on conditions
def anglet_measurement := 1
def sixth_circle_degrees := 360 / 6
def anglets_in_sixth_circle := 6000

-- Lean theorem statement proving the implied angle measurement
theorem anglet_angle_measurement (one_percent : Real := 0.01) :
  (anglets_in_sixth_circle * one_percent * sixth_circle_degrees) = anglet_measurement * 60 := 
  sorry

end NUMINAMATH_GPT_anglet_angle_measurement_l906_90663


namespace NUMINAMATH_GPT_correct_sunset_time_l906_90698

-- Definitions corresponding to the conditions
def length_of_daylight : ℕ × ℕ := (10, 30) -- (hours, minutes)
def sunrise_time : ℕ × ℕ := (6, 50) -- (hours, minutes)

-- The reaching goal is to prove the sunset time
def sunset_time (sunrise : ℕ × ℕ) (daylight : ℕ × ℕ) : ℕ × ℕ :=
  let (sh, sm) := sunrise
  let (dh, dm) := daylight
  let total_minutes := sm + dm
  let extra_hour := total_minutes / 60
  let final_minutes := total_minutes % 60
  (sh + dh + extra_hour, final_minutes)

-- The theorem to prove
theorem correct_sunset_time :
  sunset_time sunrise_time length_of_daylight = (17, 20) := sorry

end NUMINAMATH_GPT_correct_sunset_time_l906_90698


namespace NUMINAMATH_GPT_dozen_chocolate_bars_cost_l906_90686

theorem dozen_chocolate_bars_cost
  (cost_mag : ℕ → ℝ) (cost_choco_bar : ℕ → ℝ)
  (H1 : cost_mag 1 = 1)
  (H2 : 4 * (cost_choco_bar 1) = 8 * (cost_mag 1)) :
  12 * (cost_choco_bar 1) = 24 := 
sorry

end NUMINAMATH_GPT_dozen_chocolate_bars_cost_l906_90686


namespace NUMINAMATH_GPT_consultation_session_probability_l906_90689

noncomputable def consultation_probability : ℝ :=
  let volume_cube := 3 * 3 * 3
  let volume_valid := 9 - 2 * (1/3 * 2.25 * 1.5)
  volume_valid / volume_cube

theorem consultation_session_probability : consultation_probability = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_consultation_session_probability_l906_90689


namespace NUMINAMATH_GPT_Hay_s_Linens_sales_l906_90634

theorem Hay_s_Linens_sales :
  ∃ (n : ℕ), 500 ≤ 52 * n ∧ 52 * n ≤ 700 ∧
             ∀ m, (500 ≤ 52 * m ∧ 52 * m ≤ 700) → n ≤ m :=
sorry

end NUMINAMATH_GPT_Hay_s_Linens_sales_l906_90634


namespace NUMINAMATH_GPT_sum_of_cubes_of_consecutive_integers_l906_90616

-- Define the given condition
def sum_of_squares_of_consecutive_integers (n : ℕ) : Prop :=
  (n - 1)^2 + n^2 + (n + 1)^2 = 7805

-- Define the statement we want to prove
theorem sum_of_cubes_of_consecutive_integers (n : ℕ) (h : sum_of_squares_of_consecutive_integers n) : 
  (n - 1)^3 + n^3 + (n + 1)^3 = 398259 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_consecutive_integers_l906_90616


namespace NUMINAMATH_GPT_original_cost_of_dress_l906_90607

theorem original_cost_of_dress (x: ℝ) 
  (h1: x / 2 - 10 < x) 
  (h2: x - (x / 2 - 10) = 80) : 
  x = 140 :=
sorry

end NUMINAMATH_GPT_original_cost_of_dress_l906_90607


namespace NUMINAMATH_GPT_marble_189_is_gray_l906_90627

def marble_color (n : ℕ) : String :=
  let cycle_length := 14
  let gray_thres := 5
  let white_thres := 9
  let black_thres := 12
  let position := (n - 1) % cycle_length + 1
  if position ≤ gray_thres then "gray"
  else if position ≤ white_thres then "white"
  else if position ≤ black_thres then "black"
  else "blue"

theorem marble_189_is_gray : marble_color 189 = "gray" :=
by {
  -- We assume the necessary definitions and steps discussed above.
  sorry
}

end NUMINAMATH_GPT_marble_189_is_gray_l906_90627


namespace NUMINAMATH_GPT_trigonometric_identity_l906_90696

theorem trigonometric_identity 
  (x : ℝ)
  (h : Real.sin (2 * x + Real.pi / 5) = Real.sqrt 3 / 3) : 
  Real.sin (4 * Real.pi / 5 - 2 * x) + Real.sin (3 * Real.pi / 10 - 2 * x)^2 = (2 + Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l906_90696


namespace NUMINAMATH_GPT_smallest_integer_n_condition_l906_90680

theorem smallest_integer_n_condition :
  (∃ n : ℕ, n > 0 ∧ (∀ (m : ℤ), (1 ≤ m ∧ m ≤ 1992) → (∃ (k : ℤ), (m : ℚ) / 1993 < k / n ∧ k / n < (m + 1 : ℚ) / 1994))) ↔ n = 3987 :=
sorry

end NUMINAMATH_GPT_smallest_integer_n_condition_l906_90680


namespace NUMINAMATH_GPT_john_collects_crabs_l906_90636

-- Definitions for the conditions
def baskets_per_week : ℕ := 3
def crabs_per_basket : ℕ := 4
def price_per_crab : ℕ := 3
def total_income : ℕ := 72

-- Definition for the question
def times_per_week_to_collect (baskets_per_week crabs_per_basket price_per_crab total_income : ℕ) : ℕ :=
  (total_income / price_per_crab) / (baskets_per_week * crabs_per_basket)

-- The theorem statement
theorem john_collects_crabs (h1 : baskets_per_week = 3) (h2 : crabs_per_basket = 4) (h3 : price_per_crab = 3) (h4 : total_income = 72) :
  times_per_week_to_collect baskets_per_week crabs_per_basket price_per_crab total_income = 2 :=
by
  sorry

end NUMINAMATH_GPT_john_collects_crabs_l906_90636


namespace NUMINAMATH_GPT_marly_100_bills_l906_90624

-- Define the number of each type of bill Marly has
def num_20_bills := 10
def num_10_bills := 8
def num_5_bills := 4

-- Define the values of the bills
def value_20_bill := 20
def value_10_bill := 10
def value_5_bill := 5

-- Define the total amount of money Marly has
def total_amount := num_20_bills * value_20_bill + num_10_bills * value_10_bill + num_5_bills * value_5_bill

-- Define the value of a $100 bill
def value_100_bill := 100

-- Now state the main theorem
theorem marly_100_bills : total_amount / value_100_bill = 3 := by
  sorry

end NUMINAMATH_GPT_marly_100_bills_l906_90624


namespace NUMINAMATH_GPT_marbles_given_to_juan_l906_90635

def initial : ℕ := 776
def left : ℕ := 593

theorem marbles_given_to_juan : initial - left = 183 :=
by sorry

end NUMINAMATH_GPT_marbles_given_to_juan_l906_90635


namespace NUMINAMATH_GPT_solution_y_amount_l906_90690

-- Definitions based on the conditions
def alcohol_content_x : ℝ := 0.10
def alcohol_content_y : ℝ := 0.30
def initial_volume_x : ℝ := 50
def final_alcohol_percent : ℝ := 0.25

-- Function to calculate the amount of solution y needed
def required_solution_y (y : ℝ) : Prop :=
  (alcohol_content_x * initial_volume_x + alcohol_content_y * y) / (initial_volume_x + y) = final_alcohol_percent

theorem solution_y_amount : ∃ y : ℝ, required_solution_y y ∧ y = 150 := by
  sorry

end NUMINAMATH_GPT_solution_y_amount_l906_90690


namespace NUMINAMATH_GPT_degree_sequence_a_invalid_degree_sequence_b_invalid_degree_sequence_c_invalid_all_sequences_invalid_l906_90659

-- Definition of the "isValidGraph" function based on degree sequences
-- Placeholder for the actual definition
def isValidGraph (degrees : List ℕ) : Prop :=
  sorry

-- Degree sequences given in the problem
def d_a := [8, 6, 5, 4, 4, 3, 2, 2]
def d_b := [7, 7, 6, 5, 4, 2, 2, 1]
def d_c := [6, 6, 6, 5, 5, 3, 2, 2]

-- Statement that proves none of these sequences can form a valid graph
theorem degree_sequence_a_invalid : ¬ isValidGraph d_a :=
  sorry

theorem degree_sequence_b_invalid : ¬ isValidGraph d_b :=
  sorry

theorem degree_sequence_c_invalid : ¬ isValidGraph d_c :=
  sorry

-- Final statement combining all individual proofs
theorem all_sequences_invalid :
  ¬ isValidGraph d_a ∧ ¬ isValidGraph d_b ∧ ¬ isValidGraph d_c :=
  ⟨degree_sequence_a_invalid, degree_sequence_b_invalid, degree_sequence_c_invalid⟩

end NUMINAMATH_GPT_degree_sequence_a_invalid_degree_sequence_b_invalid_degree_sequence_c_invalid_all_sequences_invalid_l906_90659


namespace NUMINAMATH_GPT_total_votes_l906_90666

theorem total_votes (emma_votes : ℕ) (vote_fraction : ℚ) (h_emma : emma_votes = 45) (h_fraction : vote_fraction = 3/7) :
  emma_votes = vote_fraction * 105 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_votes_l906_90666


namespace NUMINAMATH_GPT_algebraic_expression_value_l906_90619

variables (a b c d m : ℝ)

theorem algebraic_expression_value :
  a = -b → cd = 1 → m^2 = 1 →
  -(a + b) - cd / 2022 + m^2 / 2022 = 0 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l906_90619


namespace NUMINAMATH_GPT_range_of_a_range_of_f_diff_l906_90694

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + x + 1
noncomputable def f' (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem range_of_a (a : ℝ) : (∃ x1 x2 : ℝ, f' a x1 = 0 ∧ f' a x2 = 0 ∧ x1 ≠ x2) ↔ (a < -Real.sqrt 3 ∨ a > Real.sqrt 3) :=
by
  sorry

theorem range_of_f_diff (a x1 x2 : ℝ) (h1 : f' a x1 = 0) (h2 : f' a x2 = 0) (h12 : x1 ≠ x2) : 
  0 < f a x1 - f a x2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_range_of_f_diff_l906_90694


namespace NUMINAMATH_GPT_find_points_l906_90601

def acute_triangle (A B C : ℝ × ℝ × ℝ) : Prop :=
  -- Definition to ensure that the triangle formed by A, B, and C is an acute-angled triangle.
  sorry -- This would be formalized ensuring all angles are less than 90 degrees.

def no_three_collinear (A B C D E : ℝ × ℝ × ℝ) : Prop :=
  -- Definition that ensures no three points among A, B, C, D, and E are collinear.
  sorry

def line_normal_to_plane (P Q R S : ℝ × ℝ × ℝ) : Prop :=
  -- Definition to ensure that the line through any two points P, Q is normal to the plane containing R, S, and the other point.
  sorry

theorem find_points (A B C : ℝ × ℝ × ℝ) (h_acute : acute_triangle A B C) :
  ∃ (D E : ℝ × ℝ × ℝ), no_three_collinear A B C D E ∧
    (∀ (P Q R R' : ℝ × ℝ × ℝ), 
      (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E) →
      (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E) →
      (R = A ∨ R = B ∨ R = C ∨ R = D ∨ R = E) →
      (R' = A ∨ R' = B ∨ R' = C ∨ R' = D ∨ R' = E) →
      P ≠ Q → Q ≠ R → R ≠ R' →
      line_normal_to_plane P Q R R') :=
sorry

end NUMINAMATH_GPT_find_points_l906_90601


namespace NUMINAMATH_GPT_exist_unique_xy_solution_l906_90676

theorem exist_unique_xy_solution :
  ∃! (x y : ℝ), x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3 ∧ x = 1 / 3 ∧ y = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_exist_unique_xy_solution_l906_90676


namespace NUMINAMATH_GPT_proof_of_problem_l906_90664

variable (f : ℝ → ℝ)
variable (h_nonzero : ∀ x, f x ≠ 0)
variable (h_equation : ∀ x y, f (x * y) = y * f x + x * f y)

theorem proof_of_problem :
  f 1 = 0 ∧ f (-1) = 0 ∧ (∀ x, f (-x) = -f x) :=
by
  sorry

end NUMINAMATH_GPT_proof_of_problem_l906_90664


namespace NUMINAMATH_GPT_initial_winnings_l906_90647

theorem initial_winnings (X : ℝ) 
  (h1 : X - 0.25 * X = 0.75 * X)
  (h2 : 0.75 * X - 0.10 * (0.75 * X) = 0.675 * X)
  (h3 : 0.675 * X - 0.15 * (0.675 * X) = 0.57375 * X)
  (h4 : 0.57375 * X = 240) :
  X = 418 := by
  sorry

end NUMINAMATH_GPT_initial_winnings_l906_90647


namespace NUMINAMATH_GPT_degree_of_minus_5x4y_l906_90662

def degree_of_monomial (coeff : Int) (x_exp y_exp : Nat) : Nat :=
  x_exp + y_exp

theorem degree_of_minus_5x4y : degree_of_monomial (-5) 4 1 = 5 :=
by
  sorry

end NUMINAMATH_GPT_degree_of_minus_5x4y_l906_90662


namespace NUMINAMATH_GPT_ninth_term_of_geometric_sequence_l906_90633

theorem ninth_term_of_geometric_sequence :
  let a1 := (5 : ℚ)
  let r := (3 / 4 : ℚ)
  (a1 * r^8) = (32805 / 65536 : ℚ) :=
by {
  sorry
}

end NUMINAMATH_GPT_ninth_term_of_geometric_sequence_l906_90633


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l906_90617

-- Definitions based on problem conditions
variable (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ) -- terms of the sequence
variable (S_3 S_6 S_9 : ℤ)

-- Given conditions
variable (h1 : S_3 = 3 * a_1 + 3 * (a_2 - a_1))
variable (h2 : S_6 = 6 * a_1 + 15 * (a_2 - a_1))
variable (h3 : S_3 = 9)
variable (h4 : S_6 = 36)

-- Theorem to prove
theorem arithmetic_sequence_sum : S_9 = 81 :=
by
  -- We just state the theorem here and will provide a proof later
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l906_90617


namespace NUMINAMATH_GPT_download_time_l906_90668

def file_size : ℕ := 90
def rate_first_part : ℕ := 5
def rate_second_part : ℕ := 10
def size_first_part : ℕ := 60

def time_first_part : ℕ := size_first_part / rate_first_part
def size_second_part : ℕ := file_size - size_first_part
def time_second_part : ℕ := size_second_part / rate_second_part
def total_time : ℕ := time_first_part + time_second_part

theorem download_time :
  total_time = 15 := by
  -- sorry can be replaced with the actual proof if needed
  sorry

end NUMINAMATH_GPT_download_time_l906_90668


namespace NUMINAMATH_GPT_range_of_a_l906_90603

noncomputable def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }
noncomputable def B (a : ℝ) : Set ℝ := { x | x ^ 2 + 2 * x + a ≥ 0 }

theorem range_of_a (a : ℝ) : (a > -8) → (∃ x, x ∈ A ∧ x ∈ B a) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l906_90603


namespace NUMINAMATH_GPT_infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017_l906_90687

theorem infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017 :
  ∀ n : ℕ, ∃ m : ℕ, (m ∈ {x | ∀ d ∈ Nat.digits 10 x, d = 0 ∨ d = 1}) ∧ 2017 ∣ m :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017_l906_90687


namespace NUMINAMATH_GPT_correct_sequence_is_A_l906_90683

def Step := String
def Sequence := List Step

def correct_sequence : Sequence :=
  ["Buy a ticket", "Wait for the train", "Check the ticket", "Board the train"]

def option_A : Sequence :=
  ["Buy a ticket", "Wait for the train", "Check the ticket", "Board the train"]
def option_B : Sequence :=
  ["Wait for the train", "Buy a ticket", "Board the train", "Check the ticket"]
def option_C : Sequence :=
  ["Buy a ticket", "Wait for the train", "Board the train", "Check the ticket"]
def option_D : Sequence :=
  ["Repair the train", "Buy a ticket", "Check the ticket", "Board the train"]

theorem correct_sequence_is_A :
  correct_sequence = option_A :=
sorry

end NUMINAMATH_GPT_correct_sequence_is_A_l906_90683
