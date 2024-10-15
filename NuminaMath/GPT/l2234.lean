import Mathlib

namespace NUMINAMATH_GPT_average_rainfall_correct_l2234_223456

/-- In July 1861, 366 inches of rain fell in Cherrapunji, India. -/
def total_rainfall : ℤ := 366

/-- July has 31 days. -/
def days_in_july : ℤ := 31

/-- Each day has 24 hours. -/
def hours_per_day : ℤ := 24

/-- The total number of hours in July -/
def total_hours_in_july : ℤ := days_in_july * hours_per_day

/-- The average rainfall in inches per hour during July 1861 in Cherrapunji, India -/
def average_rainfall_per_hour : ℤ := total_rainfall / total_hours_in_july

/-- Proof that the average rainfall in inches per hour is 366 / (31 * 24) -/
theorem average_rainfall_correct : average_rainfall_per_hour = 366 / (31 * 24) :=
by
  /- We skip the proof as it is not required. -/
  sorry

end NUMINAMATH_GPT_average_rainfall_correct_l2234_223456


namespace NUMINAMATH_GPT_sally_picked_peaches_l2234_223424

-- Definitions from the conditions
def originalPeaches : ℕ := 13
def totalPeaches : ℕ := 55

-- The proof statement
theorem sally_picked_peaches : totalPeaches - originalPeaches = 42 := by
  sorry

end NUMINAMATH_GPT_sally_picked_peaches_l2234_223424


namespace NUMINAMATH_GPT_bus_stop_time_l2234_223401

theorem bus_stop_time 
  (bus_speed_without_stoppages : ℤ)
  (bus_speed_with_stoppages : ℤ)
  (h1 : bus_speed_without_stoppages = 54)
  (h2 : bus_speed_with_stoppages = 36) :
  ∃ t : ℕ, t = 20 :=
by
  sorry

end NUMINAMATH_GPT_bus_stop_time_l2234_223401


namespace NUMINAMATH_GPT_proof_problem_l2234_223472

noncomputable def expr (a b : ℚ) : ℚ :=
  ((a / b + b / a + 2) * ((a + b) / (2 * a) - (b / (a + b)))) /
  ((a + 2 * b + b^2 / a) * (a / (a + b) + b / (a - b)))

theorem proof_problem : expr (3/4 : ℚ) (4/3 : ℚ) = -7/24 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2234_223472


namespace NUMINAMATH_GPT_probability_15th_roll_last_is_approximately_l2234_223469

noncomputable def probability_15th_roll_last : ℝ :=
  (7 / 8) ^ 13 * (1 / 8)

theorem probability_15th_roll_last_is_approximately :
  abs (probability_15th_roll_last - 0.022) < 0.001 :=
by sorry

end NUMINAMATH_GPT_probability_15th_roll_last_is_approximately_l2234_223469


namespace NUMINAMATH_GPT_triangle_height_in_terms_of_s_l2234_223433

theorem triangle_height_in_terms_of_s (s h : ℝ)
  (rectangle_area : 2 * s * s = 2 * s^2)
  (base_of_triangle : base = s)
  (areas_equal : (1 / 2) * s * h = 2 * s^2) :
  h = 4 * s :=
by
  sorry

end NUMINAMATH_GPT_triangle_height_in_terms_of_s_l2234_223433


namespace NUMINAMATH_GPT_largest_value_of_b_l2234_223493

theorem largest_value_of_b (b : ℚ) (h : (2 * b + 5) * (b - 1) = 6 * b) : b = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_largest_value_of_b_l2234_223493


namespace NUMINAMATH_GPT_max_trains_ratio_l2234_223444

theorem max_trains_ratio (years : ℕ) 
    (birthday_trains : ℕ) 
    (christmas_trains : ℕ) 
    (total_trains : ℕ)
    (parents_multiple : ℕ) 
    (h_years : years = 5)
    (h_birthday_trains : birthday_trains = 1)
    (h_christmas_trains : christmas_trains = 2)
    (h_total_trains : total_trains = 45)
    (h_parents_multiple : parents_multiple = 2) :
  let trains_received_in_years := years * (birthday_trains + 2 * christmas_trains)
  let trains_given_by_parents := total_trains - trains_received_in_years
  let trains_before_gift := total_trains - trains_given_by_parents
  trains_given_by_parents / trains_before_gift = parents_multiple := by
  sorry

end NUMINAMATH_GPT_max_trains_ratio_l2234_223444


namespace NUMINAMATH_GPT_wrongly_noted_mark_l2234_223421

theorem wrongly_noted_mark (x : ℕ) (h_wrong_avg : (30 : ℕ) * 100 = 3000)
    (h_correct_avg : (30 : ℕ) * 98 = 2940) (h_correct_sum : 3000 - x + 10 = 2940) : 
    x = 70 := by
  sorry

end NUMINAMATH_GPT_wrongly_noted_mark_l2234_223421


namespace NUMINAMATH_GPT_c_value_l2234_223414

theorem c_value (c : ℝ) : (∃ a : ℝ, (x : ℝ) → x^2 + 200 * x + c = (x + a)^2) → c = 10000 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_c_value_l2234_223414


namespace NUMINAMATH_GPT_DF_length_l2234_223434

-- Definitions for the given problem.
variable (AB DC EB DE : ℝ)
variable (parallelogram_ABCD : Prop)
variable (DE_altitude_AB : Prop)
variable (DF_altitude_BC : Prop)

-- Conditions
axiom AB_eq_DC : AB = DC
axiom EB_eq_5 : EB = 5
axiom DE_eq_8 : DE = 8

-- The main theorem to prove
theorem DF_length (hAB : AB = 15) (hDC : DC = 15) (hEB : EB = 5) (hDE : DE = 8)
  (hPar : parallelogram_ABCD)
  (hAltAB : DE_altitude_AB)
  (hAltBC : DF_altitude_BC) :
  ∃ DF : ℝ, DF = 8 := 
sorry

end NUMINAMATH_GPT_DF_length_l2234_223434


namespace NUMINAMATH_GPT_green_notebook_cost_l2234_223467

def total_cost : ℕ := 45
def black_cost : ℕ := 15
def pink_cost : ℕ := 10
def num_green_notebooks : ℕ := 2

theorem green_notebook_cost :
  (total_cost - (black_cost + pink_cost)) / num_green_notebooks = 10 :=
by
  sorry

end NUMINAMATH_GPT_green_notebook_cost_l2234_223467


namespace NUMINAMATH_GPT_Eva_needs_weeks_l2234_223482

theorem Eva_needs_weeks (apples : ℕ) (days_in_week : ℕ) (weeks : ℕ) 
  (h1 : apples = 14)
  (h2 : days_in_week = 7) 
  (h3 : apples = weeks * days_in_week) : 
  weeks = 2 := 
by 
  sorry

end NUMINAMATH_GPT_Eva_needs_weeks_l2234_223482


namespace NUMINAMATH_GPT_math_problem_l2234_223481

variable {R : Type} [LinearOrderedField R]

theorem math_problem
  (a b : R) (ha : 0 < a) (hb : 0 < b)
  (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2234_223481


namespace NUMINAMATH_GPT_area_percentage_increase_l2234_223487

theorem area_percentage_increase (r₁ r₂ : ℝ) (π : ℝ) :
  r₁ = 6 ∧ r₂ = 4 ∧ π > 0 →
  (π * r₁^2 - π * r₂^2) / (π * r₂^2) * 100 = 125 := 
by {
  sorry
}

end NUMINAMATH_GPT_area_percentage_increase_l2234_223487


namespace NUMINAMATH_GPT_point_B_coordinates_l2234_223407

theorem point_B_coordinates (A B : ℝ) (hA : A = -2) (hDist : |A - B| = 3) : B = -5 ∨ B = 1 :=
by
  sorry

end NUMINAMATH_GPT_point_B_coordinates_l2234_223407


namespace NUMINAMATH_GPT_simplify_expression_l2234_223411

theorem simplify_expression (p q r : ℝ) (hp : p ≠ 7) (hq : q ≠ 8) (hr : r ≠ 9) :
  ( ( (p - 7) / (9 - r) ) * ( (q - 8) / (7 - p) ) * ( (r - 9) / (8 - q) ) ) = -1 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2234_223411


namespace NUMINAMATH_GPT_cost_of_items_l2234_223470

theorem cost_of_items {x y z : ℕ} (h1 : x + 3 * y + 2 * z = 98)
                      (h2 : 3 * x + y = 5 * z - 36)
                      (even_x : x % 2 = 0) :
  x = 4 ∧ y = 22 ∧ z = 14 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_items_l2234_223470


namespace NUMINAMATH_GPT_minimum_rectangle_area_l2234_223415

theorem minimum_rectangle_area (l w : ℕ) (h : 2 * (l + w) = 84) : 
  (l * w) = 41 :=
by sorry

end NUMINAMATH_GPT_minimum_rectangle_area_l2234_223415


namespace NUMINAMATH_GPT_cos_squared_identity_l2234_223484

variable (θ : ℝ)

-- Given condition
def tan_theta : Prop := Real.tan θ = 2

-- Question: Find the value of cos²(θ + π/4)
theorem cos_squared_identity (h : tan_theta θ) : Real.cos (θ + Real.pi / 4) ^ 2 = 1 / 10 := 
  sorry

end NUMINAMATH_GPT_cos_squared_identity_l2234_223484


namespace NUMINAMATH_GPT_b_over_a_squared_eq_seven_l2234_223446

theorem b_over_a_squared_eq_seven (a b k : ℕ) (ha : a > 1) (hb : b = a * (10^k + 1)) (hdiv : a^2 ∣ b) :
  b / a^2 = 7 :=
sorry

end NUMINAMATH_GPT_b_over_a_squared_eq_seven_l2234_223446


namespace NUMINAMATH_GPT_water_volume_correct_l2234_223455

def total_initial_solution : ℚ := 0.08 + 0.04 + 0.02
def fraction_water_in_initial : ℚ := 0.04 / total_initial_solution
def desired_total_volume : ℚ := 0.84
def required_water_volume : ℚ := desired_total_volume * fraction_water_in_initial

theorem water_volume_correct : 
  required_water_volume = 0.24 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_water_volume_correct_l2234_223455


namespace NUMINAMATH_GPT_correct_regression_eq_l2234_223471

-- Definitions related to the conditions
def negative_correlation (y x : ℝ) : Prop :=
  -- y is negatively correlated with x implies a negative slope in regression
  ∃ a b : ℝ, a < 0 ∧ ∀ x, y = a * x + b

-- The potential regression equations
def regression_eq1 (x : ℝ) : ℝ := -10 * x + 200
def regression_eq2 (x : ℝ) : ℝ := 10 * x + 200
def regression_eq3 (x : ℝ) : ℝ := -10 * x - 200
def regression_eq4 (x : ℝ) : ℝ := 10 * x - 200

-- Prove that the correct regression equation is selected given the conditions
theorem correct_regression_eq (y x : ℝ) (h : negative_correlation y x) : 
  (∀ x : ℝ, y = regression_eq1 x) ∨ (∀ x : ℝ, y = regression_eq2 x) ∨ 
  (∀ x : ℝ, y = regression_eq3 x) ∨ (∀ x : ℝ, y = regression_eq4 x) →
  ∀ x : ℝ, y = regression_eq1 x := by
  -- This theorem states that given negative correlation and the possible options, 
  -- the correct regression equation consistent with all conditions must be regression_eq1.
  sorry

end NUMINAMATH_GPT_correct_regression_eq_l2234_223471


namespace NUMINAMATH_GPT_total_homework_problems_l2234_223403

-- Define the conditions as Lean facts
def finished_problems : ℕ := 45
def ratio_finished_to_left := (9, 4)
def problems_left (L : ℕ) := finished_problems * ratio_finished_to_left.2 = L * ratio_finished_to_left.1 

-- State the theorem
theorem total_homework_problems (L : ℕ) (h : problems_left L) : finished_problems + L = 65 :=
sorry

end NUMINAMATH_GPT_total_homework_problems_l2234_223403


namespace NUMINAMATH_GPT_inscribed_squares_ratio_l2234_223489

theorem inscribed_squares_ratio (x y : ℝ) 
  (h₁ : 5^2 + 12^2 = 13^2)
  (h₂ : x = 144 / 17)
  (h₃ : y = 5) :
  x / y = 144 / 85 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_squares_ratio_l2234_223489


namespace NUMINAMATH_GPT_four_angles_for_shapes_l2234_223462

-- Definitions for the shapes
def is_rectangle (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

def is_square (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

def is_parallelogram (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

-- Main proposition
theorem four_angles_for_shapes {fig : Type} :
  (is_rectangle fig) ∧ (is_square fig) ∧ (is_parallelogram fig) →
  ∀ shape : fig, ∃ angles : ℕ, angles = 4 := by
  sorry

end NUMINAMATH_GPT_four_angles_for_shapes_l2234_223462


namespace NUMINAMATH_GPT_polynomial_divisible_l2234_223496

theorem polynomial_divisible (A B : ℝ) (h : ∀ x : ℂ, x^2 - x + 1 = 0 → x^103 + A * x + B = 0) : A + B = -1 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisible_l2234_223496


namespace NUMINAMATH_GPT_initial_amount_invested_l2234_223437

-- Definition of the conditions as Lean definitions
def initial_amount_interest_condition (A r : ℝ) : Prop := 25000 = A * r
def interest_rate_condition (r : ℝ) : Prop := r = 5

-- The main theorem we want to prove
theorem initial_amount_invested (A r : ℝ) (h1 : initial_amount_interest_condition A r) (h2 : interest_rate_condition r) : A = 5000 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_amount_invested_l2234_223437


namespace NUMINAMATH_GPT_min_M_value_l2234_223420

variable {a b c t : ℝ}

theorem min_M_value (h1 : a < b)
                    (h2 : a > 0)
                    (h3 : b^2 - 4 * a * c ≤ 0)
                    (h4 : b = t + a)
                    (h5 : t > 0)
                    (h6 : c ≥ (t + a)^2 / (4 * a)) :
    ∃ M : ℝ, (∀ x : ℝ, (a * x^2 + b * x + c) ≥ 0) → M = 3 := 
  sorry

end NUMINAMATH_GPT_min_M_value_l2234_223420


namespace NUMINAMATH_GPT_correct_product_of_a_and_b_l2234_223447

-- Define reversal function for two-digit numbers
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

-- State the main problem
theorem correct_product_of_a_and_b (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 0 < b) 
  (h : (reverse_digits a) * b = 284) : a * b = 68 :=
sorry

end NUMINAMATH_GPT_correct_product_of_a_and_b_l2234_223447


namespace NUMINAMATH_GPT_six_digit_number_theorem_l2234_223465

-- Define the problem conditions
def six_digit_number_condition (N : ℕ) (x : ℕ) : Prop :=
  N = 200000 + x ∧ N < 1000000 ∧ (10 * x + 2 = 3 * N)

-- Define the value of x
def value_of_x : ℕ := 85714

-- Main theorem to prove
theorem six_digit_number_theorem (N : ℕ) (x : ℕ) (h1 : x = value_of_x) :
  six_digit_number_condition N x → N = 285714 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_six_digit_number_theorem_l2234_223465


namespace NUMINAMATH_GPT_value_of_expression_l2234_223406

theorem value_of_expression (m : ℝ) (h : m^2 - m - 1 = 0) : m^2 - m + 5 = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2234_223406


namespace NUMINAMATH_GPT_not_all_perfect_squares_l2234_223468

theorem not_all_perfect_squares (d : ℕ) (hd : 0 < d) :
  ¬ (∃ (x y z : ℕ), 2 * d - 1 = x^2 ∧ 5 * d - 1 = y^2 ∧ 13 * d - 1 = z^2) :=
by
  sorry

end NUMINAMATH_GPT_not_all_perfect_squares_l2234_223468


namespace NUMINAMATH_GPT_final_coordinates_of_A_l2234_223495

-- Define the initial points
def A : ℝ × ℝ := (3, -2)
def B : ℝ × ℝ := (5, -5)
def C : ℝ × ℝ := (2, -4)

-- Define the translation operation
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

-- Define the rotation operation (180 degrees around a point (h, k))
def rotate180 (p : ℝ × ℝ) (h k : ℝ) : ℝ × ℝ :=
  (2 * h - p.1, 2 * k - p.2)

-- Translate point A
def A' := translate A 4 3

-- Rotate the translated point A' 180 degrees around the point (4, 0)
def A'' := rotate180 A' 4 0

-- The final coordinates of point A after transformations should be (1, -1)
theorem final_coordinates_of_A : A'' = (1, -1) :=
  sorry

end NUMINAMATH_GPT_final_coordinates_of_A_l2234_223495


namespace NUMINAMATH_GPT_proposition_false_n5_l2234_223452

variable (P : ℕ → Prop)

-- Declaring the conditions as definitions:
def condition1 (k : ℕ) (hk : k > 0) : Prop := P k → P (k + 1)
def condition2 : Prop := ¬ P 6

-- Theorem statement which leverages the conditions to prove the desired result.
theorem proposition_false_n5 (h1: ∀ k (hk : k > 0), condition1 P k hk) (h2: condition2 P) : ¬ P 5 :=
sorry

end NUMINAMATH_GPT_proposition_false_n5_l2234_223452


namespace NUMINAMATH_GPT_smallest_AAAB_value_l2234_223460

theorem smallest_AAAB_value : ∃ (A B : ℕ), A ≠ B ∧ A < 10 ∧ B < 10 ∧ 111 * A + B = 7 * (10 * A + B) ∧ 111 * A + B = 667 :=
by sorry

end NUMINAMATH_GPT_smallest_AAAB_value_l2234_223460


namespace NUMINAMATH_GPT_greatest_m_value_l2234_223457

noncomputable def find_greatest_m : ℝ := sorry

theorem greatest_m_value :
  ∃ m : ℝ, 
    (∀ x, x^2 - m * x + 8 = 0 → x ∈ {x | ∃ y, y^2 = 116}) ∧ 
    m = 2 * Real.sqrt 29 :=
sorry

end NUMINAMATH_GPT_greatest_m_value_l2234_223457


namespace NUMINAMATH_GPT_sequence_general_term_l2234_223458

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (n / (n + 1 : ℝ)) * a n) : 
  ∀ n, a n = 1 / n := by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l2234_223458


namespace NUMINAMATH_GPT_Woojin_harvested_weight_l2234_223441

-- Definitions based on conditions
def younger_brother_harvest : Float := 3.8
def older_sister_harvest : Float := younger_brother_harvest + 8.4
def one_tenth_older_sister : Float := older_sister_harvest / 10
def woojin_extra_g : Float := 3720

-- Convert grams to kilograms
def grams_to_kg (g : Float) : Float := g / 1000

-- Theorem to be proven
theorem Woojin_harvested_weight :
  grams_to_kg (one_tenth_older_sister * 1000 + woojin_extra_g) = 4.94 :=
by
  sorry

end NUMINAMATH_GPT_Woojin_harvested_weight_l2234_223441


namespace NUMINAMATH_GPT_find_math_marks_l2234_223492

theorem find_math_marks :
  ∀ (english marks physics chemistry biology : ℕ) (average : ℕ),
  average = 78 →
  english = 91 →
  physics = 82 →
  chemistry = 67 →
  biology = 85 →
  (english + marks + physics + chemistry + biology) / 5 = average →
  marks = 65 :=
by
  intros english marks physics chemistry biology average h_average h_english h_physics h_chemistry h_biology h_avg_eq
  sorry

end NUMINAMATH_GPT_find_math_marks_l2234_223492


namespace NUMINAMATH_GPT_remainder_7n_mod_4_l2234_223431

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_7n_mod_4_l2234_223431


namespace NUMINAMATH_GPT_largest_consecutive_integer_product_2520_l2234_223486

theorem largest_consecutive_integer_product_2520 :
  ∃ (n : ℕ), n * (n + 1) * (n + 2) * (n + 3) = 2520 ∧ (n + 3) = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_consecutive_integer_product_2520_l2234_223486


namespace NUMINAMATH_GPT_quadratic_real_roots_range_l2234_223442

theorem quadratic_real_roots_range (m : ℝ) :
  ∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0 ↔ m ≤ 4 ∧ m ≠ 3 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_l2234_223442


namespace NUMINAMATH_GPT_fraction_doubled_l2234_223425

variable (x y : ℝ)

theorem fraction_doubled (x y : ℝ) : 
  (x + y) ≠ 0 → (2 * x * 2 * y) / (2 * x + 2 * y) = 2 * (x * y / (x + y)) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_fraction_doubled_l2234_223425


namespace NUMINAMATH_GPT_disjoint_subsets_same_sum_l2234_223499

/-- 
Given a set of 10 distinct integers between 1 and 100, 
there exist two disjoint subsets of this set that have the same sum.
-/
theorem disjoint_subsets_same_sum : ∃ (x : Finset ℤ), (x.card = 10) ∧ (∀ i ∈ x, 1 ≤ i ∧ i ≤ 100) → 
  ∃ (A B : Finset ℤ), (A ⊆ x) ∧ (B ⊆ x) ∧ (A ∩ B = ∅) ∧ (A.sum id = B.sum id) :=
by
  sorry

end NUMINAMATH_GPT_disjoint_subsets_same_sum_l2234_223499


namespace NUMINAMATH_GPT_min_value_y_l2234_223443

noncomputable def y (x : ℝ) := (2 - Real.cos x) / Real.sin x

theorem min_value_y (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) : 
  ∃ c ≥ 0, ∀ x, 0 < x ∧ x < Real.pi → y x ≥ c ∧ c = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_min_value_y_l2234_223443


namespace NUMINAMATH_GPT_cos_double_angle_l2234_223409

theorem cos_double_angle (α β : Real) 
    (h1 : Real.sin α = Real.cos β) 
    (h2 : Real.sin α * Real.cos β - 2 * Real.cos α * Real.sin β = 1 / 2) :
    Real.cos (2 * β) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l2234_223409


namespace NUMINAMATH_GPT_boxes_in_case_correct_l2234_223449

-- Given conditions
def total_boxes : Nat := 2
def blocks_per_box : Nat := 6
def total_blocks : Nat := 12

-- Define the number of boxes in a case as a result of total_blocks divided by blocks_per_box
def boxes_in_case : Nat := total_blocks / blocks_per_box

-- Prove the number of boxes in a case is 2
theorem boxes_in_case_correct : boxes_in_case = 2 := by
  -- Place the actual proof here
  sorry

end NUMINAMATH_GPT_boxes_in_case_correct_l2234_223449


namespace NUMINAMATH_GPT_paul_score_higher_by_26_l2234_223429

variable {R : Type} [LinearOrderedField R]

variables (A1 A2 A3 P1 P2 P3 : R)

-- hypotheses
variable (h1 : A1 = P1 + 10)
variable (h2 : A2 = P2 + 4)
variable (h3 : (P1 + P2 + P3) / 3 = (A1 + A2 + A3) / 3 + 4)

-- goal
theorem paul_score_higher_by_26 : P3 - A3 = 26 := by
  sorry

end NUMINAMATH_GPT_paul_score_higher_by_26_l2234_223429


namespace NUMINAMATH_GPT_min_sum_intercepts_l2234_223423

theorem min_sum_intercepts (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (1 : ℝ) * a + (1 : ℝ) * b = a * b) : a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_intercepts_l2234_223423


namespace NUMINAMATH_GPT_intersection_with_y_axis_l2234_223451

theorem intersection_with_y_axis :
  ∀ (y : ℝ), (∃ x : ℝ, y = 2 * x + 2 ∧ x = 0) → y = 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_with_y_axis_l2234_223451


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2234_223497

theorem quadratic_has_two_distinct_real_roots (k : ℝ) : 
  ((k - 1) * x^2 + 2 * x - 2 = 0) → (1 / 2 < k ∧ k ≠ 1) :=
sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l2234_223497


namespace NUMINAMATH_GPT_emily_weight_l2234_223438

theorem emily_weight (h_weight : 87 = 78 + e_weight) : e_weight = 9 := by
  sorry

end NUMINAMATH_GPT_emily_weight_l2234_223438


namespace NUMINAMATH_GPT_boys_number_is_60_l2234_223478

-- Definitions based on the conditions
variables (x y : ℕ)

def sum_boys_girls (x y : ℕ) : Prop := 
  x + y = 150

def girls_percentage (x y : ℕ) : Prop := 
  y = (x * 150) / 100

-- Prove that the number of boys equals 60
theorem boys_number_is_60 (x y : ℕ) 
  (h1 : sum_boys_girls x y) 
  (h2 : girls_percentage x y) : 
  x = 60 := by
  sorry

end NUMINAMATH_GPT_boys_number_is_60_l2234_223478


namespace NUMINAMATH_GPT_smallest_integer_y_l2234_223400

theorem smallest_integer_y : ∃ (y : ℤ), (7 + 3 * y < 25) ∧ (∀ z : ℤ, (7 + 3 * z < 25) → y ≤ z) ∧ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_y_l2234_223400


namespace NUMINAMATH_GPT_max_profit_at_grade_5_l2234_223432

-- Defining the conditions
def profit_per_item (x : ℕ) : ℕ :=
  4 * (x - 1) + 8

def production_count (x : ℕ) : ℕ := 
  60 - 6 * (x - 1)

def daily_profit (x : ℕ) : ℕ :=
  profit_per_item x * production_count x

-- The grade range
def grade_range (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 10

-- Prove that the grade that maximizes the profit is 5
theorem max_profit_at_grade_5 : (1 ≤ x ∧ x ≤ 10) → daily_profit x ≤ daily_profit 5 :=
sorry

end NUMINAMATH_GPT_max_profit_at_grade_5_l2234_223432


namespace NUMINAMATH_GPT_option_d_correct_l2234_223494

variable (a b m n : ℝ)

theorem option_d_correct :
  6 * a + a ≠ 6 * a ^ 2 ∧
  -2 * a + 5 * b ≠ 3 * a * b ∧
  4 * m ^ 2 * n - 2 * m * n ^ 2 ≠ 2 * m * n ∧
  3 * a * b ^ 2 - 5 * b ^ 2 * a = -2 * a * b ^ 2 := by
  sorry

end NUMINAMATH_GPT_option_d_correct_l2234_223494


namespace NUMINAMATH_GPT_fixed_point_of_line_l2234_223422

theorem fixed_point_of_line (a : ℝ) : 
  (a + 3) * (-2) + (2 * a - 1) * 1 + 7 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_fixed_point_of_line_l2234_223422


namespace NUMINAMATH_GPT_mixed_number_solution_l2234_223440

noncomputable def mixed_number_problem : Prop :=
  let a := 4 + 2 / 7
  let b := 5 + 1 / 2
  let c := 3 + 1 / 3
  let d := 2 + 1 / 6
  (a * b) - (c + d) = 18 + 1 / 14

theorem mixed_number_solution : mixed_number_problem := by 
  sorry

end NUMINAMATH_GPT_mixed_number_solution_l2234_223440


namespace NUMINAMATH_GPT_sequence_property_l2234_223445

variable (a : ℕ → ℝ)

theorem sequence_property (h : ∀ n : ℕ, 0 < a n) 
  (h_property : ∀ n : ℕ, (a n)^2 ≤ a n - a (n + 1)) :
  ∀ n : ℕ, a n < 1 / n :=
by
  sorry

end NUMINAMATH_GPT_sequence_property_l2234_223445


namespace NUMINAMATH_GPT_solution_exists_l2234_223479

theorem solution_exists (a b c : ℝ) : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 := 
sorry

end NUMINAMATH_GPT_solution_exists_l2234_223479


namespace NUMINAMATH_GPT_number_of_fiction_books_l2234_223412

theorem number_of_fiction_books (F NF : ℕ) (h1 : F + NF = 52) (h2 : NF = 7 * F / 6) : F = 24 := 
by
  sorry

end NUMINAMATH_GPT_number_of_fiction_books_l2234_223412


namespace NUMINAMATH_GPT_cannot_factorize_using_difference_of_squares_l2234_223480

theorem cannot_factorize_using_difference_of_squares (x y : ℝ) :
  ¬ ∃ a b : ℝ, -x^2 - y^2 = a^2 - b^2 :=
sorry

end NUMINAMATH_GPT_cannot_factorize_using_difference_of_squares_l2234_223480


namespace NUMINAMATH_GPT_rectangular_prism_volume_l2234_223476

theorem rectangular_prism_volume (w : ℝ) (w_pos : 0 < w) 
    (h_edges_sum : 4 * w + 8 * (2 * w) + 4 * (w / 2) = 88) :
    (2 * w) * w * (w / 2) = 85184 / 343 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_volume_l2234_223476


namespace NUMINAMATH_GPT_quotient_remainder_threefold_l2234_223485

theorem quotient_remainder_threefold (a b c d : ℤ)
  (h : a = b * c + d) :
  3 * a = 3 * b * c + 3 * d :=
by sorry

end NUMINAMATH_GPT_quotient_remainder_threefold_l2234_223485


namespace NUMINAMATH_GPT_square_roots_N_l2234_223488

theorem square_roots_N (m N : ℤ) (h1 : (3 * m - 4) ^ 2 = N) (h2 : (7 - 4 * m) ^ 2 = N) : N = 25 := 
by
  sorry

end NUMINAMATH_GPT_square_roots_N_l2234_223488


namespace NUMINAMATH_GPT_sum_of_possible_amounts_l2234_223410

-- Definitions based on conditions:
def possible_quarters_amounts : Finset ℕ := {5, 30, 55, 80}
def possible_dimes_amounts : Finset ℕ := {15, 20, 30, 35, 40, 50, 60, 70, 80, 90}
def both_possible_amounts : Finset ℕ := possible_quarters_amounts ∩ possible_dimes_amounts

-- Statement of the problem:
theorem sum_of_possible_amounts : (both_possible_amounts.sum id) = 110 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_amounts_l2234_223410


namespace NUMINAMATH_GPT_fourth_root_sq_eq_sixteen_l2234_223450

theorem fourth_root_sq_eq_sixteen (x : ℝ) (h : (x^(1/4))^2 = 16) : x = 256 :=
sorry

end NUMINAMATH_GPT_fourth_root_sq_eq_sixteen_l2234_223450


namespace NUMINAMATH_GPT_shoe_price_monday_final_price_l2234_223427

theorem shoe_price_monday_final_price : 
  let thursday_price := 50
  let friday_markup_rate := 0.15
  let monday_discount_rate := 0.12
  let friday_price := thursday_price * (1 + friday_markup_rate)
  let monday_price := friday_price * (1 - monday_discount_rate)
  monday_price = 50.6 := by
  sorry

end NUMINAMATH_GPT_shoe_price_monday_final_price_l2234_223427


namespace NUMINAMATH_GPT_range_of_a_l2234_223448

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - (1 / 2) * a * x^2 - 2 * x

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x - a * x - 1

theorem range_of_a
  (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f_prime x1 a = 0 ∧ f_prime x2 a = 0) ↔
  0 < a ∧ a < Real.exp (-2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2234_223448


namespace NUMINAMATH_GPT_solve_for_x_l2234_223474

theorem solve_for_x (x : ℝ) (h : 2 / 3 + 1 / x = 7 / 9) : x = 9 := 
  sorry

end NUMINAMATH_GPT_solve_for_x_l2234_223474


namespace NUMINAMATH_GPT_average_weight_of_all_boys_l2234_223490

theorem average_weight_of_all_boys 
  (n₁ n₂ : ℕ) (w₁ w₂ : ℝ) 
  (h₁ : n₁ = 20) (h₂ : w₁ = 50.25) 
  (h₃ : n₂ = 8) (h₄ : w₂ = 45.15) :
  (n₁ * w₁ + n₂ * w₂) / (n₁ + n₂) = 48.79 := 
by
  sorry

end NUMINAMATH_GPT_average_weight_of_all_boys_l2234_223490


namespace NUMINAMATH_GPT_sum_of_n_and_k_l2234_223436

open Nat

theorem sum_of_n_and_k (n k : ℕ)
  (h1 : 2 = n - 3 * k)
  (h2 : 8 = 2 * n - 5 * k) :
  n + k = 18 :=
sorry

end NUMINAMATH_GPT_sum_of_n_and_k_l2234_223436


namespace NUMINAMATH_GPT_min_value_of_f_l2234_223435

-- Define the problem domain: positive real numbers
variables (a b c x y z : ℝ)
variables (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0)
variables (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0)

-- Define the given equations
variables (h1 : c * y + b * z = a)
variables (h2 : a * z + c * x = b)
variables (h3 : b * x + a * y = c)

-- Define the function f(x, y, z)
noncomputable def f (x y z : ℝ) : ℝ :=
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)

-- The theorem statement: under the given conditions the minimum value of f(x, y, z) is 1/2
theorem min_value_of_f :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    c * y + b * z = a →
    a * z + c * x = b →
    b * x + a * y = c →
    f x y z = 1 / 2) :=
sorry

end NUMINAMATH_GPT_min_value_of_f_l2234_223435


namespace NUMINAMATH_GPT_jane_evening_pages_l2234_223498

theorem jane_evening_pages :
  ∀ (P : ℕ), (7 * (5 + P) = 105) → P = 10 :=
by
  intros P h
  sorry

end NUMINAMATH_GPT_jane_evening_pages_l2234_223498


namespace NUMINAMATH_GPT_div_relation_l2234_223408

theorem div_relation (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 3) : c / a = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_div_relation_l2234_223408


namespace NUMINAMATH_GPT_positive_b_3b_sq_l2234_223466

variable (a b c : ℝ)

theorem positive_b_3b_sq (h1 : 0 < a ∧ a < 0.5) (h2 : -0.5 < b ∧ b < 0) (h3 : 1 < c ∧ c < 3) : b + 3 * b^2 > 0 :=
sorry

end NUMINAMATH_GPT_positive_b_3b_sq_l2234_223466


namespace NUMINAMATH_GPT_travel_distance_l2234_223477

variables (speed time : ℕ) (distance : ℕ)

theorem travel_distance (hspeed : speed = 75) (htime : time = 4) : distance = speed * time → distance = 300 :=
by
  intros hdist
  rw [hspeed, htime] at hdist
  simp at hdist
  assumption

end NUMINAMATH_GPT_travel_distance_l2234_223477


namespace NUMINAMATH_GPT_range_of_k_l2234_223404

noncomputable def f (x : ℝ) : ℝ := Real.log x + x

def is_ktimes_value_function (f : ℝ → ℝ) (k : ℝ) (a b : ℝ) : Prop :=
  0 < k ∧ a < b ∧ f a = k * a ∧ f b = k * b

theorem range_of_k (k : ℝ) : (∃ a b : ℝ, is_ktimes_value_function f k a b) ↔ 1 < k ∧ k < 1 + 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_range_of_k_l2234_223404


namespace NUMINAMATH_GPT_winning_candidate_votes_l2234_223405

theorem winning_candidate_votes  (V W : ℝ) (hW : W = 0.5666666666666664 * V) (hV : V = W + 7636 + 11628) : 
  W = 25216 := 
by 
  sorry

end NUMINAMATH_GPT_winning_candidate_votes_l2234_223405


namespace NUMINAMATH_GPT_arithmetic_sequence_S9_l2234_223418

noncomputable def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_S9 (a : ℕ → ℕ)
    (h1 : 2 * a 6 = 6 + a 7) :
    Sn a 9 = 54 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_S9_l2234_223418


namespace NUMINAMATH_GPT_smallest_n_for_constant_term_l2234_223459

theorem smallest_n_for_constant_term :
  ∃ (n : ℕ), (n > 0) ∧ ((∃ (r : ℕ), 2 * n = 5 * r) ∧ (∀ (m : ℕ), m > 0 → (∃ (r' : ℕ), 2 * m = 5 * r') → n ≤ m)) ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_constant_term_l2234_223459


namespace NUMINAMATH_GPT_remainder_add_mod_l2234_223402

theorem remainder_add_mod (n : ℕ) (h : n % 7 = 2) : (n + 1470) % 7 = 2 := 
by sorry

end NUMINAMATH_GPT_remainder_add_mod_l2234_223402


namespace NUMINAMATH_GPT_fraction_of_defective_engines_l2234_223463

theorem fraction_of_defective_engines
  (total_batches : ℕ)
  (engines_per_batch : ℕ)
  (non_defective_engines : ℕ)
  (H1 : total_batches = 5)
  (H2 : engines_per_batch = 80)
  (H3 : non_defective_engines = 300)
  : (total_batches * engines_per_batch - non_defective_engines) / (total_batches * engines_per_batch) = 1 / 4 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_fraction_of_defective_engines_l2234_223463


namespace NUMINAMATH_GPT_dropped_student_score_l2234_223413

theorem dropped_student_score (total_students : ℕ) (remaining_students : ℕ) (initial_average : ℝ) (new_average : ℝ) (x : ℝ) 
  (h1 : total_students = 16) 
  (h2 : remaining_students = 15) 
  (h3 : initial_average = 62.5) 
  (h4 : new_average = 63.0) 
  (h5 : total_students * initial_average - remaining_students * new_average = x) : 
  x = 55 := 
sorry

end NUMINAMATH_GPT_dropped_student_score_l2234_223413


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2234_223473

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -9 * x^2 + 6 * x - 8 < 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_inequality_solution_l2234_223473


namespace NUMINAMATH_GPT_sin_10pi_over_3_l2234_223426

theorem sin_10pi_over_3 : Real.sin (10 * Real.pi / 3) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_10pi_over_3_l2234_223426


namespace NUMINAMATH_GPT_sequence_term_formula_l2234_223428

theorem sequence_term_formula 
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (h : ∀ n, S n = n^2 + 3 * n)
  (h₁ : a 1 = 4)
  (h₂ : ∀ n, 1 < n → a n = S n - S (n - 1)) :
  ∀ n, a n = 2 * n + 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_formula_l2234_223428


namespace NUMINAMATH_GPT_simplify_sqrt_expression_eq_l2234_223461

noncomputable def simplify_sqrt_expression (x : ℝ) : ℝ :=
  let sqrt_45x := Real.sqrt (45 * x)
  let sqrt_20x := Real.sqrt (20 * x)
  let sqrt_30x := Real.sqrt (30 * x)
  sqrt_45x * sqrt_20x * sqrt_30x

theorem simplify_sqrt_expression_eq (x : ℝ) :
  simplify_sqrt_expression x = 30 * x * Real.sqrt 30 := by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_eq_l2234_223461


namespace NUMINAMATH_GPT_total_number_of_trees_l2234_223475

theorem total_number_of_trees (D P : ℕ) (cost_D cost_P total_cost : ℕ)
  (hD : D = 350)
  (h_cost_D : cost_D = 300)
  (h_cost_P : cost_P = 225)
  (h_total_cost : total_cost = 217500)
  (h_cost_equation : cost_D * D + cost_P * P = total_cost) :
  D + P = 850 :=
by
  rw [hD, h_cost_D, h_cost_P, h_total_cost] at h_cost_equation
  sorry

end NUMINAMATH_GPT_total_number_of_trees_l2234_223475


namespace NUMINAMATH_GPT_sum_greater_than_four_l2234_223491

theorem sum_greater_than_four (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hprod : x * y > x + y) : x + y > 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_greater_than_four_l2234_223491


namespace NUMINAMATH_GPT_estimate_total_height_l2234_223483

theorem estimate_total_height :
  let middle_height := 100
  let left_height := 0.80 * middle_height
  let right_height := (left_height + middle_height) - 20
  left_height + middle_height + right_height = 340 := 
by
  sorry

end NUMINAMATH_GPT_estimate_total_height_l2234_223483


namespace NUMINAMATH_GPT_functional_inequality_solution_l2234_223419

theorem functional_inequality_solution (f : ℝ → ℝ) (h : ∀ a b : ℝ, f (a^2) - f (b^2) ≤ (f (a) + b) * (a - f (b))) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := 
sorry

end NUMINAMATH_GPT_functional_inequality_solution_l2234_223419


namespace NUMINAMATH_GPT_candy_bars_per_bag_l2234_223439

/-
Define the total number of candy bars and the number of bags
-/
def totalCandyBars : ℕ := 75
def numberOfBags : ℚ := 15.0

/-
Prove that the number of candy bars per bag is 5
-/
theorem candy_bars_per_bag : totalCandyBars / numberOfBags = 5 := by
  sorry

end NUMINAMATH_GPT_candy_bars_per_bag_l2234_223439


namespace NUMINAMATH_GPT_min_reciprocal_sum_l2234_223416

theorem min_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  (1 / a) + (1 / b) ≥ 2 := by
  sorry

end NUMINAMATH_GPT_min_reciprocal_sum_l2234_223416


namespace NUMINAMATH_GPT_order_DABC_l2234_223464

-- Definitions of the variables given in the problem
def A : ℕ := 77^7
def B : ℕ := 7^77
def C : ℕ := 7^7^7
def D : ℕ := Nat.factorial 7

-- The theorem stating the required ascending order
theorem order_DABC : D < A ∧ A < B ∧ B < C :=
by sorry

end NUMINAMATH_GPT_order_DABC_l2234_223464


namespace NUMINAMATH_GPT_shaded_area_eq_l2234_223454

noncomputable def diameter_AB : ℝ := 6
noncomputable def diameter_BC : ℝ := 6
noncomputable def diameter_CD : ℝ := 6
noncomputable def diameter_DE : ℝ := 6
noncomputable def diameter_EF : ℝ := 6
noncomputable def diameter_FG : ℝ := 6
noncomputable def diameter_AG : ℝ := 6 * 6 -- 36

noncomputable def area_small_semicircle (d : ℝ) : ℝ :=
  (1/8) * Real.pi * d^2

noncomputable def area_large_semicircle (d : ℝ) : ℝ :=
  (1/8) * Real.pi * d^2

theorem shaded_area_eq :
  area_large_semicircle diameter_AG + area_small_semicircle diameter_AB = 166.5 * Real.pi :=
  sorry

end NUMINAMATH_GPT_shaded_area_eq_l2234_223454


namespace NUMINAMATH_GPT_inhabitants_number_even_l2234_223430

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem inhabitants_number_even
  (K L : ℕ)
  (hK : is_even K)
  (hL : is_even L) :
  ¬ is_even (K + L + 1) :=
by
  sorry

end NUMINAMATH_GPT_inhabitants_number_even_l2234_223430


namespace NUMINAMATH_GPT_find_r_over_s_at_0_l2234_223453

noncomputable def r (x : ℝ) : ℝ := -3 * (x + 1) * (x - 2)
noncomputable def s (x : ℝ) : ℝ := (x + 1) * (x - 3)

theorem find_r_over_s_at_0 : (r 0) / (s 0) = 2 := by
  sorry

end NUMINAMATH_GPT_find_r_over_s_at_0_l2234_223453


namespace NUMINAMATH_GPT_log_ab_is_pi_l2234_223417

open Real

noncomputable def log_ab (a b : ℝ) : ℝ :=
(log b) / (log a)

theorem log_ab_is_pi (a b : ℝ)  (ha_pos: 0 < a) (ha_ne_one: a ≠ 1) (hb_pos: 0 < b) 
  (cond1 : log (a ^ 3) = log (b ^ 6)) (cond2 : cos (π * log a) = 1) : log_ab a b = π :=
by
  sorry

end NUMINAMATH_GPT_log_ab_is_pi_l2234_223417
