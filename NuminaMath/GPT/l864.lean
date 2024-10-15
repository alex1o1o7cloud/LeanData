import Mathlib

namespace NUMINAMATH_GPT_math_vs_english_time_difference_l864_86400

-- Definitions based on the conditions
def english_total_questions : ℕ := 30
def math_total_questions : ℕ := 15
def english_total_time_minutes : ℕ := 60 -- 1 hour = 60 minutes
def math_total_time_minutes : ℕ := 90 -- 1.5 hours = 90 minutes

noncomputable def time_per_english_question : ℕ :=
  english_total_time_minutes / english_total_questions

noncomputable def time_per_math_question : ℕ :=
  math_total_time_minutes / math_total_questions

-- Theorem based on the question and correct answer
theorem math_vs_english_time_difference :
  (time_per_math_question - time_per_english_question) = 4 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_math_vs_english_time_difference_l864_86400


namespace NUMINAMATH_GPT_final_value_after_three_years_l864_86490

theorem final_value_after_three_years (X : ℝ) :
  (X - 0.40 * X) * (1 - 0.10) * (1 - 0.20) = 0.432 * X := by
  sorry

end NUMINAMATH_GPT_final_value_after_three_years_l864_86490


namespace NUMINAMATH_GPT_sum_smallest_numbers_eq_six_l864_86431

theorem sum_smallest_numbers_eq_six :
  let smallest_natural := 0
  let smallest_prime := 2
  let smallest_composite := 4
  smallest_natural + smallest_prime + smallest_composite = 6 := by
  sorry

end NUMINAMATH_GPT_sum_smallest_numbers_eq_six_l864_86431


namespace NUMINAMATH_GPT_rainbow_nerds_total_l864_86458

theorem rainbow_nerds_total
  (purple yellow green red blue : ℕ)
  (h1 : purple = 10)
  (h2 : yellow = purple + 4)
  (h3 : green = yellow - 2)
  (h4 : red = 3 * green)
  (h5 : blue = red / 2) :
  (purple + yellow + green + red + blue = 90) :=
by
  sorry

end NUMINAMATH_GPT_rainbow_nerds_total_l864_86458


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l864_86435

-- Define the conditions
variables {S_3 a_1 a_3 : ℕ}
variables (d : ℕ)
axiom h1 : S_3 = 6
axiom h2 : a_3 = 4
axiom h3 : S_3 = 3 * (a_1 + a_3) / 2

-- Prove that the common difference d is 2
theorem arithmetic_sequence_common_difference :
  d = (a_3 - a_1) / 2 → d = 2 :=
by
  sorry -- Proof to be completed

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l864_86435


namespace NUMINAMATH_GPT_determine_a_from_equation_l864_86449

theorem determine_a_from_equation (a : ℝ) (x : ℝ) (h1 : x = 1) (h2 : a * x + 3 * x = 2) : a = -1 := by
  sorry

end NUMINAMATH_GPT_determine_a_from_equation_l864_86449


namespace NUMINAMATH_GPT_correct_M_l864_86473

-- Definition of the function M for calculating the position number
def M (k : ℕ) : ℕ :=
  if k % 2 = 1 then
    4 * k^2 - 4 * k + 2
  else
    4 * k^2 - 2 * k + 2

-- Theorem stating the correctness of the function M
theorem correct_M (k : ℕ) : M k = if k % 2 = 1 then 4 * k^2 - 4 * k + 2 else 4 * k^2 - 2 * k + 2 := 
by
  -- The proof is to be done later.
  -- sorry is used to indicate a placeholder.
  sorry

end NUMINAMATH_GPT_correct_M_l864_86473


namespace NUMINAMATH_GPT_cone_heights_l864_86419

theorem cone_heights (H x r1 r2 : ℝ) (H_frustum : H - x = 18)
  (A_lower : 400 * Real.pi = Real.pi * r1^2)
  (A_upper : 100 * Real.pi = Real.pi * r2^2)
  (ratio_radii : r2 / r1 = 1 / 2)
  (ratio_heights : x / H = 1 / 2) :
  x = 18 ∧ H = 36 :=
by
  sorry

end NUMINAMATH_GPT_cone_heights_l864_86419


namespace NUMINAMATH_GPT_austin_pairs_of_shoes_l864_86444

theorem austin_pairs_of_shoes (S : ℕ) :
  0.45 * (S : ℝ) + 11 = S → S / 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_austin_pairs_of_shoes_l864_86444


namespace NUMINAMATH_GPT_seq_value_at_2018_l864_86429

noncomputable def f (x : ℝ) : ℝ := sorry
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = f 0 ∧ ∀ (n : ℕ), n > 0 → f (a (n + 1)) = 1 / f (-2 - a n)

theorem seq_value_at_2018 (a : ℕ → ℝ) (h_seq : seq a) : a 2018 = 4035 := 
by sorry

end NUMINAMATH_GPT_seq_value_at_2018_l864_86429


namespace NUMINAMATH_GPT_customers_tipped_count_l864_86476

variable (initial_customers : ℕ)
variable (added_customers : ℕ)
variable (customers_no_tip : ℕ)

def total_customers (initial_customers added_customers : ℕ) : ℕ :=
  initial_customers + added_customers

theorem customers_tipped_count 
  (h_init : initial_customers = 29)
  (h_added : added_customers = 20)
  (h_no_tip : customers_no_tip = 34) :
  (total_customers initial_customers added_customers - customers_no_tip) = 15 :=
by
  sorry

end NUMINAMATH_GPT_customers_tipped_count_l864_86476


namespace NUMINAMATH_GPT_correct_exp_operation_l864_86411

theorem correct_exp_operation (a b : ℝ) : (-a^3 * b) ^ 2 = a^6 * b^2 :=
  sorry

end NUMINAMATH_GPT_correct_exp_operation_l864_86411


namespace NUMINAMATH_GPT_fraction_broke_off_l864_86424

variable (p p_1 p_2 : ℝ)
variable (k : ℝ)

-- Conditions
def initial_mass : Prop := p_1 + p_2 = p
def value_relation : Prop := p_1^2 + p_2^2 = 0.68 * p^2

-- Goal
theorem fraction_broke_off (h1 : initial_mass p p_1 p_2)
                           (h2 : value_relation p p_1 p_2) :
  (p_2 / p) = 1 / 5 :=
sorry

end NUMINAMATH_GPT_fraction_broke_off_l864_86424


namespace NUMINAMATH_GPT_volume_ratio_of_cubes_l864_86499

theorem volume_ratio_of_cubes 
  (P_A P_B : ℕ) 
  (h_A : P_A = 40) 
  (h_B : P_B = 64) : 
  (∃ s_A s_B V_A V_B, 
    s_A = P_A / 4 ∧ 
    s_B = P_B / 4 ∧ 
    V_A = s_A^3 ∧ 
    V_B = s_B^3 ∧ 
    (V_A : ℚ) / V_B = 125 / 512) := 
by
  sorry

end NUMINAMATH_GPT_volume_ratio_of_cubes_l864_86499


namespace NUMINAMATH_GPT_identify_mathematicians_l864_86471

def famous_people := List (Nat × String)

def is_mathematician : Nat → Bool
| 1 => false  -- Bill Gates
| 2 => true   -- Gauss
| 3 => false  -- Yuan Longping
| 4 => false  -- Nobel
| 5 => true   -- Chen Jingrun
| 6 => true   -- Hua Luogeng
| 7 => false  -- Gorky
| 8 => false  -- Einstein
| _ => false  -- default case

theorem identify_mathematicians (people : famous_people) : 
  (people.filter (fun (n, _) => is_mathematician n)) = [(2, "Gauss"), (5, "Chen Jingrun"), (6, "Hua Luogeng")] :=
by sorry

end NUMINAMATH_GPT_identify_mathematicians_l864_86471


namespace NUMINAMATH_GPT_exists_c_with_same_nonzero_decimal_digits_l864_86479

theorem exists_c_with_same_nonzero_decimal_digits (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  ∃ (c : ℕ), 0 < c ∧ (∃ (k : ℕ), (c * m) % 10^k = (c * n) % 10^k) := 
sorry

end NUMINAMATH_GPT_exists_c_with_same_nonzero_decimal_digits_l864_86479


namespace NUMINAMATH_GPT_B_share_is_2400_l864_86422

noncomputable def calculate_B_share (total_profit : ℝ) (x : ℝ) : ℝ :=
  let A_investment_months := 3 * x * 12
  let B_investment_months := x * 6
  let C_investment_months := (3/2) * x * 9
  let D_investment_months := (3/2) * x * 8
  let total_investment_months := A_investment_months + B_investment_months + C_investment_months + D_investment_months
  (B_investment_months / total_investment_months) * total_profit

theorem B_share_is_2400 :
  calculate_B_share 27000 1 = 2400 :=
sorry

end NUMINAMATH_GPT_B_share_is_2400_l864_86422


namespace NUMINAMATH_GPT_value_of_x_l864_86447

noncomputable def sum_integers_30_to_50 : ℕ :=
  (50 - 30 + 1) * (30 + 50) / 2

def even_count_30_to_50 : ℕ :=
  11

theorem value_of_x 
  (x := sum_integers_30_to_50)
  (y := even_count_30_to_50)
  (h : x + y = 851) : x = 840 :=
sorry

end NUMINAMATH_GPT_value_of_x_l864_86447


namespace NUMINAMATH_GPT_system_of_equations_has_solution_l864_86440

theorem system_of_equations_has_solution (m : ℝ) : 
  ∃ (x y : ℝ), y = 3 * m * x + 4 ∧ y = (3 * m - 1) * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_has_solution_l864_86440


namespace NUMINAMATH_GPT_simplify_expression_l864_86445

variable (x y : ℕ)

theorem simplify_expression :
  7 * x + 9 * y + 3 - x + 12 * y + 15 = 6 * x + 21 * y + 18 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l864_86445


namespace NUMINAMATH_GPT_joan_trip_time_l864_86426

-- Definitions of given conditions as parameters
def distance : ℕ := 480
def speed : ℕ := 60
def lunch_break_minutes : ℕ := 30
def bathroom_break_minutes : ℕ := 15
def number_of_bathroom_breaks : ℕ := 2

-- Conversion factors
def minutes_to_hours (m : ℕ) : ℚ := m / 60

-- Calculation of total time taken
def total_time : ℚ := 
  (distance / speed) + 
  (minutes_to_hours lunch_break_minutes) + 
  (number_of_bathroom_breaks * minutes_to_hours bathroom_break_minutes)

-- Statement of the problem
theorem joan_trip_time : total_time = 9 := 
  by 
    sorry

end NUMINAMATH_GPT_joan_trip_time_l864_86426


namespace NUMINAMATH_GPT_circle_inscribed_radius_l864_86407

theorem circle_inscribed_radius (R α : ℝ) (hα : α < Real.pi) : 
  ∃ x : ℝ, x = R * (Real.sin (α / 4))^2 :=
sorry

end NUMINAMATH_GPT_circle_inscribed_radius_l864_86407


namespace NUMINAMATH_GPT_perimeter_of_wheel_K_l864_86481

theorem perimeter_of_wheel_K
  (L_turns_K : 4 / 5 = 1 / (length_of_K / length_of_L))
  (L_turns_M : 6 / 7 = 1 / (length_of_L / length_of_M))
  (M_perimeter : length_of_M = 30) :
  length_of_K = 28 := 
sorry

end NUMINAMATH_GPT_perimeter_of_wheel_K_l864_86481


namespace NUMINAMATH_GPT_tan_A_in_triangle_ABC_l864_86417

theorem tan_A_in_triangle_ABC (a b c : ℝ) (A B C : ℝ) (ha : 0 < A) (ha_90 : A < π / 2) 
(hb : b = 3 * a * Real.sin B) : Real.tan A = Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_GPT_tan_A_in_triangle_ABC_l864_86417


namespace NUMINAMATH_GPT_max_value_a_plus_b_plus_c_l864_86484

-- Definitions used in the problem
def A_n (a n : ℕ) : ℕ := a * (10^n - 1) / 9
def B_n (b n : ℕ) : ℕ := b * (10^n - 1) / 9
def C_n (c n : ℕ) : ℕ := c * (10^(2 * n) - 1) / 9

-- Main statement of the problem
theorem max_value_a_plus_b_plus_c (n : ℕ) (a b c : ℕ) (h : n > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_eq : ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ C_n c n1 - B_n b n1 = 2 * (A_n a n1)^2 ∧ C_n c n2 - B_n b n2 = 2 * (A_n a n2)^2) :
  a + b + c ≤ 18 :=
sorry

end NUMINAMATH_GPT_max_value_a_plus_b_plus_c_l864_86484


namespace NUMINAMATH_GPT_rectangle_perimeter_eq_l864_86488

noncomputable def rectangle_perimeter (x y : ℝ) := 2 * (x + y)

theorem rectangle_perimeter_eq (x y a b : ℝ)
  (h_area_rect : x * y = 2450)
  (h_area_ellipse : a * b = 2450)
  (h_foci_distance : x + y = 2 * a)
  (h_diag : x^2 + y^2 = 4 * (a^2 - b^2))
  (h_b : b = Real.sqrt (a^2 - 1225))
  : rectangle_perimeter x y = 120 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_eq_l864_86488


namespace NUMINAMATH_GPT_alex_score_l864_86477

theorem alex_score (n : ℕ) (avg19 avg20 alex : ℚ)
  (h1 : n = 20)
  (h2 : avg19 = 72)
  (h3 : avg20 = 74)
  (h_totalscore19 : 19 * avg19 = 1368)
  (h_totalscore20 : 20 * avg20 = 1480)
  (h_alexscore : alex = 112) :
  alex = (1480 - 1368 : ℚ) := 
sorry

end NUMINAMATH_GPT_alex_score_l864_86477


namespace NUMINAMATH_GPT_irrational_b_eq_neg_one_l864_86401

theorem irrational_b_eq_neg_one
  (a : ℝ) (b : ℝ)
  (h_irrational : ¬ ∃ q : ℚ, a = (q : ℝ))
  (h_eq : ab + a - b = 1) :
  b = -1 :=
sorry

end NUMINAMATH_GPT_irrational_b_eq_neg_one_l864_86401


namespace NUMINAMATH_GPT_total_apples_l864_86461

theorem total_apples (baskets apples_per_basket : ℕ) (h1 : baskets = 37) (h2 : apples_per_basket = 17) : baskets * apples_per_basket = 629 := by
  sorry

end NUMINAMATH_GPT_total_apples_l864_86461


namespace NUMINAMATH_GPT_verify_tin_amount_l864_86450

def ratio_to_fraction (part1 part2 : ℕ) : ℚ :=
  part2 / (part1 + part2 : ℕ)

def tin_amount_in_alloy (total_weight : ℚ) (ratio : ℚ) : ℚ :=
  total_weight * ratio

def alloy_mixture_tin_weight_is_correct
    (weight_A weight_B : ℚ)
    (ratio_A_lead ratio_A_tin : ℕ)
    (ratio_B_tin ratio_B_copper : ℕ) : Prop :=
  let tin_ratio_A := ratio_to_fraction ratio_A_lead ratio_A_tin
  let tin_ratio_B := ratio_to_fraction ratio_B_tin ratio_B_copper
  let tin_weight_A := tin_amount_in_alloy weight_A tin_ratio_A
  let tin_weight_B := tin_amount_in_alloy weight_B tin_ratio_B
  tin_weight_A + tin_weight_B = 146.57

theorem verify_tin_amount :
    alloy_mixture_tin_weight_is_correct 130 160 2 3 3 4 :=
by
  sorry

end NUMINAMATH_GPT_verify_tin_amount_l864_86450


namespace NUMINAMATH_GPT_range_of_a_l864_86451

theorem range_of_a (a : ℝ) : 
  ( ∃ x y : ℝ, (x^2 + 4 * (y - a)^2 = 4) ∧ (x^2 = 4 * y)) ↔ a ∈ Set.Ico (-1 : ℝ) (5 / 4 : ℝ) := 
sorry

end NUMINAMATH_GPT_range_of_a_l864_86451


namespace NUMINAMATH_GPT_middle_tree_less_half_tallest_tree_l864_86443

theorem middle_tree_less_half_tallest_tree (T M S : ℝ)
  (hT : T = 108)
  (hS : S = 1/4 * M)
  (hS_12 : S = 12) :
  (1/2 * T) - M = 6 := 
sorry

end NUMINAMATH_GPT_middle_tree_less_half_tallest_tree_l864_86443


namespace NUMINAMATH_GPT_find_a_l864_86404

noncomputable def f (a x : ℝ) := 3*x^3 - 9*x + a
noncomputable def f' (x : ℝ) : ℝ := 9*x^2 - 9

theorem find_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :
  a = 6 ∨ a = -6 :=
by sorry

end NUMINAMATH_GPT_find_a_l864_86404


namespace NUMINAMATH_GPT_smallest_natural_number_exists_l864_86416

theorem smallest_natural_number_exists (n : ℕ) : (∃ n, ∃ a b c : ℕ, n = 15 ∧ 1998 = a * (5 ^ 4) + b * (3 ^ 4) + c * (1 ^ 4) ∧ a + b + c = 15) :=
sorry

end NUMINAMATH_GPT_smallest_natural_number_exists_l864_86416


namespace NUMINAMATH_GPT_line_tangent_to_circle_perpendicular_l864_86469

theorem line_tangent_to_circle_perpendicular 
  (l₁ l₂ : String)
  (C : String)
  (h1 : l₂ = "4 * x - 3 * y + 1 = 0")
  (h2 : C = "x^2 + y^2 + 2 * y - 3 = 0") :
  (l₁ = "3 * x + 4 * y + 14 = 0" ∨ l₁ = "3 * x + 4 * y - 6 = 0") :=
by
  sorry

end NUMINAMATH_GPT_line_tangent_to_circle_perpendicular_l864_86469


namespace NUMINAMATH_GPT_closest_perfect_square_to_350_l864_86497

theorem closest_perfect_square_to_350 : 
  ∃ n : ℤ, n^2 < 350 ∧ 350 < (n + 1)^2 ∧ (350 - n^2 ≤ (n + 1)^2 - 350) ∨ (350 - n^2 ≥ (n + 1)^2 - 350) ∧ 
  (if (350 - n^2 < (n + 1)^2 - 350) then n+1 else n) = 19 := 
by
  sorry

end NUMINAMATH_GPT_closest_perfect_square_to_350_l864_86497


namespace NUMINAMATH_GPT_marys_mother_paid_correct_total_l864_86434

def mary_and_friends_payment_per_person : ℕ := 1 -- $1 each
def number_of_people : ℕ := 3 -- Mary and two friends

def total_chicken_cost : ℕ := mary_and_friends_payment_per_person * number_of_people -- Total cost of the chicken

def beef_cost_per_pound : ℕ := 4 -- $4 per pound
def total_beef_pounds : ℕ := 3 -- 3 pounds of beef
def total_beef_cost : ℕ := beef_cost_per_pound * total_beef_pounds -- Total cost of the beef

def oil_cost : ℕ := 1 -- $1 for 1 liter of oil

def total_grocery_cost : ℕ := total_chicken_cost + total_beef_cost + oil_cost -- Total grocery cost

theorem marys_mother_paid_correct_total : total_grocery_cost = 16 := by
  -- Here you would normally provide the proof steps which we're skipping per instructions.
  sorry

end NUMINAMATH_GPT_marys_mother_paid_correct_total_l864_86434


namespace NUMINAMATH_GPT_unique_exponential_solution_l864_86457

theorem unique_exponential_solution (a x : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hx_pos : 0 < x) :
  ∃! y : ℝ, a^y = x :=
by
  sorry

end NUMINAMATH_GPT_unique_exponential_solution_l864_86457


namespace NUMINAMATH_GPT_ellipse_equation_is_correct_line_equation_is_correct_l864_86498

-- Given conditions
variable (a b e x y : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (ab_order : b < a)
variable (minor_axis_half_major_axis : 2 * a * (1 / 2) = 2 * b)
variable (right_focus_shortest_distance : a - e = 2 - Real.sqrt 3)
variable (ellipse_equation : a^2 = b^2 + e^2)
variable (m : ℝ)
variable (area_triangle_AOB_is_1 : 1 = 1)

-- Part (I) Prove the equation of ellipse C
theorem ellipse_equation_is_correct :
  (∀ x y : ℝ, (x^2 / 4 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
sorry

-- Part (II) Prove the equation of line l
theorem line_equation_is_correct :
  (∀ x y : ℝ, (y = x + m) ↔ ((y = x + (Real.sqrt 10 / 2)) ∨ (y = x - (Real.sqrt 10 / 2)))) :=
sorry

end NUMINAMATH_GPT_ellipse_equation_is_correct_line_equation_is_correct_l864_86498


namespace NUMINAMATH_GPT_incorrect_statements_l864_86430

-- Define basic properties for lines and their equations.
def point_slope_form (y y1 x x1 k : ℝ) : Prop := (y - y1) = k * (x - x1)
def intercept_form (x y a b : ℝ) : Prop := x / a + y / b = 1
def distance_to_origin_on_y_axis (k b : ℝ) : ℝ := abs b
def slope_intercept_form (y m x c : ℝ) : Prop := y = m * x + c

-- The conditions specified in the problem.
variables (A B C D : Prop)
  (hA : A ↔ ∀ (y y1 x x1 k : ℝ), ¬point_slope_form y y1 x x1 k)
  (hB : B ↔ ∀ (x y a b : ℝ), intercept_form x y a b)
  (hC : C ↔ ∀ (k b : ℝ), distance_to_origin_on_y_axis k b = abs b)
  (hD : D ↔ ∀ (y m x c : ℝ), slope_intercept_form y m x c)

theorem incorrect_statements : ¬ B ∧ ¬ C ∧ ¬ D :=
by
  -- Intermediate steps would be to show each statement B, C, and D are false.
  sorry

end NUMINAMATH_GPT_incorrect_statements_l864_86430


namespace NUMINAMATH_GPT_crossing_time_proof_l864_86441

/-
  Problem:
  Given:
  1. length_train: 600 (length of the train in meters)
  2. time_signal_post: 40 (time taken to cross the signal post in seconds)
  3. time_bridge_minutes: 20 (time taken to cross the bridge in minutes)

  Prove:
  t_cross_bridge: the time it takes to cross the bridge and the full length of the train is 1240 seconds
-/

def length_train : ℕ := 600
def time_signal_post : ℕ := 40
def time_bridge_minutes : ℕ := 20

-- Converting time to cross the bridge from minutes to seconds
def time_bridge_seconds : ℕ := time_bridge_minutes * 60

-- Finding the speed
def speed_train : ℕ := length_train / time_signal_post

-- Finding the length of the bridge
def length_bridge : ℕ := speed_train * time_bridge_seconds

-- Finding the total distance covered
def total_distance : ℕ := length_train + length_bridge

-- Given distance and speed, find the time to cross
def time_to_cross : ℕ := total_distance / speed_train

theorem crossing_time_proof : time_to_cross = 1240 := by
  sorry

end NUMINAMATH_GPT_crossing_time_proof_l864_86441


namespace NUMINAMATH_GPT_solve_sum_of_coefficients_l864_86492

theorem solve_sum_of_coefficients (a b : ℝ) 
  (h1 : ∀ x, ax^2 - bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) : a + b = -10 :=
  sorry

end NUMINAMATH_GPT_solve_sum_of_coefficients_l864_86492


namespace NUMINAMATH_GPT_q_is_necessary_but_not_sufficient_for_p_l864_86472

theorem q_is_necessary_but_not_sufficient_for_p (a : ℝ) : 
  (∀ x : ℝ, (x^2 + 2*x + 1 - a^2 < 0) → (-1 + a < x ∧ x < -1 - a)) → (a < 1) ∧ (¬ (a < 1 → (∀ x : ℝ, (x^2 + 2*x + 1 - a^2 < 0) → (-1 + a < x ∧ x < -1 - a)))) :=
by
  sorry

end NUMINAMATH_GPT_q_is_necessary_but_not_sufficient_for_p_l864_86472


namespace NUMINAMATH_GPT_hiker_miles_l864_86487

-- Defining the conditions as a def
def total_steps (flips : ℕ) (additional_steps : ℕ) : ℕ := flips * 100000 + additional_steps

def steps_per_mile : ℕ := 1500

-- The target theorem to prove the number of miles walked
theorem hiker_miles (flips : ℕ) (additional_steps : ℕ) (s_per_mile : ℕ) 
  (h_flips : flips = 72) (h_additional_steps : additional_steps = 25370) 
  (h_s_per_mile : s_per_mile = 1500) : 
  (total_steps flips additional_steps) / s_per_mile = 4817 :=
by
  -- sorry is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_hiker_miles_l864_86487


namespace NUMINAMATH_GPT_solve_quadratic_and_compute_l864_86483

theorem solve_quadratic_and_compute (y : ℝ) (h : 4 * y^2 + 7 = 6 * y + 12) : (8 * y - 2)^2 = 248 := 
sorry

end NUMINAMATH_GPT_solve_quadratic_and_compute_l864_86483


namespace NUMINAMATH_GPT_deposit_amount_correct_l864_86494

noncomputable def deposit_amount (initial_amount : ℝ) : ℝ :=
  let first_step := 0.30 * initial_amount
  let second_step := 0.25 * first_step
  0.20 * second_step

theorem deposit_amount_correct :
  deposit_amount 50000 = 750 :=
by
  sorry

end NUMINAMATH_GPT_deposit_amount_correct_l864_86494


namespace NUMINAMATH_GPT_opposite_of_neg_quarter_l864_86475

theorem opposite_of_neg_quarter : -(- (1/4 : ℝ)) = (1/4 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_quarter_l864_86475


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l864_86446

def sum_of_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a₁ d : ℕ)
  (h₁ : a₁ + (a₁ + 6 * d) + (a₁ + 13 * d) + (a₁ + 17 * d) = 120) :
  sum_of_arithmetic_sequence a₁ d 19 = 570 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l864_86446


namespace NUMINAMATH_GPT_jose_profit_share_correct_l864_86489

-- Definitions for the conditions
def tom_investment : ℕ := 30000
def tom_months : ℕ := 12
def jose_investment : ℕ := 45000
def jose_months : ℕ := 10
def total_profit : ℕ := 36000

-- Capital months calculations
def tom_capital_months : ℕ := tom_investment * tom_months
def jose_capital_months : ℕ := jose_investment * jose_months
def total_capital_months : ℕ := tom_capital_months + jose_capital_months

-- Jose's share of the profit calculation
def jose_share_of_profit : ℕ := (jose_capital_months * total_profit) / total_capital_months

-- The theorem to prove
theorem jose_profit_share_correct : jose_share_of_profit = 20000 := by
  -- This is where the proof steps would go
  sorry

end NUMINAMATH_GPT_jose_profit_share_correct_l864_86489


namespace NUMINAMATH_GPT_gcd_lcm_240_l864_86421

theorem gcd_lcm_240 (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 240) : 
  ∃ n, ∃ gcds : Finset ℕ, (gcds.card = n) ∧ (Nat.gcd a b ∈ gcds) :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_240_l864_86421


namespace NUMINAMATH_GPT_initial_volume_of_solution_l864_86470

theorem initial_volume_of_solution (V : ℝ) :
  (∀ (init_vol : ℝ), 0.84 * init_vol / (init_vol + 26.9) = 0.58) →
  V = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_volume_of_solution_l864_86470


namespace NUMINAMATH_GPT_complement_A_in_U_l864_86495

def U : Set ℕ := {x | x ≥ 2}
def A : Set ℕ := {x | x^2 ≥ 5}

theorem complement_A_in_U : (U \ A) = {2} := by
  sorry

end NUMINAMATH_GPT_complement_A_in_U_l864_86495


namespace NUMINAMATH_GPT_find_c_solution_l864_86491

theorem find_c_solution {c : ℚ} 
  (h₁ : ∃ x : ℤ, 2 * (x : ℚ)^2 + 17 * x - 55 = 0 ∧ x = ⌊c⌋)
  (h₂ : ∃ x : ℚ, 6 * x^2 - 23 * x + 7 = 0 ∧ 0 ≤ x ∧ x < 1 ∧ x = c - ⌊c⌋) :
  c = -32 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_c_solution_l864_86491


namespace NUMINAMATH_GPT_cars_meet_and_crush_fly_l864_86402

noncomputable def time_to_meet (L v_A v_B : ℝ) : ℝ := L / (v_A + v_B)

theorem cars_meet_and_crush_fly :
  ∀ (L v_A v_B v_fly : ℝ), L = 300 → v_A = 50 → v_B = 100 → v_fly = 150 → time_to_meet L v_A v_B = 2 :=
by
  intros L v_A v_B v_fly L_eq v_A_eq v_B_eq v_fly_eq
  rw [L_eq, v_A_eq, v_B_eq]
  simp [time_to_meet]
  norm_num

end NUMINAMATH_GPT_cars_meet_and_crush_fly_l864_86402


namespace NUMINAMATH_GPT_emily_coloring_books_l864_86439

variable (initial_books : ℕ) (given_away : ℕ) (total_books : ℕ) (bought_books : ℕ)

theorem emily_coloring_books :
  initial_books = 7 →
  given_away = 2 →
  total_books = 19 →
  initial_books - given_away + bought_books = total_books →
  bought_books = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_emily_coloring_books_l864_86439


namespace NUMINAMATH_GPT_find_g_neg1_l864_86433

-- Define the function f and its property of being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

-- Define the function g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- Given conditions
variables {f : ℝ → ℝ}
variable  (h_odd : odd_function f)
variable  (h_g1 : g f 1 = 1)

-- The statement we want to prove
theorem find_g_neg1 : g f (-1) = 3 :=
sorry

end NUMINAMATH_GPT_find_g_neg1_l864_86433


namespace NUMINAMATH_GPT_final_ranking_l864_86456

-- Define data types for participants and their initial positions
inductive Participant
| X
| Y
| Z

open Participant

-- Define the initial conditions and number of position changes
def initial_positions : List Participant := [X, Y, Z]

def position_changes : Participant → Nat
| X => 5
| Y => 0  -- Not given explicitly but derived from the conditions.
| Z => 6

-- Final condition stating Y finishes before X
def Y_before_X : Prop := True

-- The theorem stating the final ranking
theorem final_ranking :
  Y_before_X →
  (initial_positions = [X, Y, Z]) →
  (position_changes X = 5) →
  (position_changes Z = 6) →
  (position_changes Y = 0) →
  [Y, X, Z] = [Y, X, Z] :=
by
  intros
  exact rfl

end NUMINAMATH_GPT_final_ranking_l864_86456


namespace NUMINAMATH_GPT_sum_of_ages_l864_86437

-- Definitions based on given conditions
def J : ℕ := 19
def age_difference (B J : ℕ) : Prop := B - J = 32

-- Theorem stating the problem
theorem sum_of_ages (B : ℕ) (H : age_difference B J) : B + J = 70 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l864_86437


namespace NUMINAMATH_GPT_expression_in_terms_of_x_difference_between_x_l864_86485

variable (E x : ℝ)

theorem expression_in_terms_of_x (h1 : E / (2 * x + 15) = 3) : E = 6 * x + 45 :=
by 
  sorry

variable (x1 x2 : ℝ)

theorem difference_between_x (h1 : E / (2 * x1 + 15) = 3) (h2: E / (2 * x2 + 15) = 3) (h3 : x2 - x1 = 12) : True :=
by 
  sorry

end NUMINAMATH_GPT_expression_in_terms_of_x_difference_between_x_l864_86485


namespace NUMINAMATH_GPT_numbers_difference_l864_86405

theorem numbers_difference (A B C : ℝ) (h1 : B = 10) (h2 : B - A = C - B) (h3 : A * B = 85) (h4 : B * C = 115) : 
  B - A = 1.5 ∧ C - B = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_numbers_difference_l864_86405


namespace NUMINAMATH_GPT_find_x_when_y_is_10_l864_86460

-- Definitions of inverse proportionality and initial conditions
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k

-- Given constants
def k : ℝ := 160
def x_initial : ℝ := 40
def y_initial : ℝ := 4

-- Theorem statement to prove the value of x when y = 10
theorem find_x_when_y_is_10 (h : inversely_proportional x_initial y_initial k) : 
  ∃ (x : ℝ), inversely_proportional x 10 k :=
sorry

end NUMINAMATH_GPT_find_x_when_y_is_10_l864_86460


namespace NUMINAMATH_GPT_game_winning_strategy_l864_86423

theorem game_winning_strategy (n : ℕ) (h : n ≥ 3) :
  (∃ k : ℕ, n = 3 * k + 2) → (∃ k : ℕ, n = 3 * k + 2 ∨ ∀ k : ℕ, n ≠ 3 * k + 2) :=
by
  sorry

end NUMINAMATH_GPT_game_winning_strategy_l864_86423


namespace NUMINAMATH_GPT_cost_price_of_cricket_bat_l864_86420

variable (CP_A CP_B SP_C : ℝ)

-- Conditions
def condition1 : CP_B = 1.20 * CP_A := sorry
def condition2 : SP_C = 1.25 * CP_B := sorry
def condition3 : SP_C = 234 := sorry

-- The statement to prove
theorem cost_price_of_cricket_bat : CP_A = 156 := sorry

end NUMINAMATH_GPT_cost_price_of_cricket_bat_l864_86420


namespace NUMINAMATH_GPT_sum_of_squares_expressible_l864_86408

theorem sum_of_squares_expressible (a b c : ℕ) (h1 : c^2 = a^2 + b^2) : 
  ∃ x y : ℕ, x^2 + y^2 = c^2 + a*b ∧ ∃ u v : ℕ, u^2 + v^2 = c^2 - a*b :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_expressible_l864_86408


namespace NUMINAMATH_GPT_complementary_angles_positive_difference_l864_86409

theorem complementary_angles_positive_difference
  (x : ℝ)
  (h1 : 3 * x + x = 90): 
  |(3 * x) - x| = 45 := 
by
  -- Proof would go here (details skipped)
  sorry

end NUMINAMATH_GPT_complementary_angles_positive_difference_l864_86409


namespace NUMINAMATH_GPT_proof_problem_l864_86448

-- Define the proportional relationship
def proportional_relationship (y x : ℝ) (k : ℝ) : Prop :=
  y - 1 = k * (x + 2)

-- Define the function y = 2x + 5
def function_y_x (y x : ℝ) : Prop :=
  y = 2 * x + 5

-- The theorem for part (1) and (2)
theorem proof_problem (x y a : ℝ) (h1 : proportional_relationship 7 1 2) (h2 : proportional_relationship y x 2) :
  function_y_x y x ∧ function_y_x (-2) a → a = -7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l864_86448


namespace NUMINAMATH_GPT_perpendicular_lines_solve_for_a_l864_86478

theorem perpendicular_lines_solve_for_a :
  ∀ (a : ℝ), 
  ((3 * a + 2) * (5 * a - 2) + (1 - 4 * a) * (a + 4) = 0) → 
  (a = 0 ∨ a = 1) :=
by
  intro a h
  sorry

end NUMINAMATH_GPT_perpendicular_lines_solve_for_a_l864_86478


namespace NUMINAMATH_GPT_solve_for_x_l864_86474

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : x^3 - 2 * x^2 = 0 ↔ x = 2 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l864_86474


namespace NUMINAMATH_GPT_lateral_surface_area_of_square_pyramid_l864_86412

-- Definitions based on the conditions in a)
def baseEdgeLength : ℝ := 4
def slantHeight : ℝ := 3

-- Lean 4 statement for the proof problem
theorem lateral_surface_area_of_square_pyramid :
  let height := Real.sqrt (slantHeight^2 - (baseEdgeLength / 2)^2)
  let lateralArea := (1 / 2) * 4 * (baseEdgeLength * height)
  lateralArea = 8 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_square_pyramid_l864_86412


namespace NUMINAMATH_GPT_find_common_difference_l864_86468

theorem find_common_difference (a a_n S_n : ℝ) (h1 : a = 3) (h2 : a_n = 50) (h3 : S_n = 318) : 
  ∃ d n, (a + (n - 1) * d = a_n) ∧ (n / 2 * (a + a_n) = S_n) ∧ (d = 47 / 11) := 
by
  sorry

end NUMINAMATH_GPT_find_common_difference_l864_86468


namespace NUMINAMATH_GPT_count_even_digits_in_base_5_of_567_l864_86415

def is_even (n : ℕ) : Bool := n % 2 = 0

def base_5_representation (n : ℕ) : List ℕ :=
  if h : n > 0 then
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc else loop (n / 5) ((n % 5) :: acc)
    loop n []
  else [0]

def count_even_digits_in_base_5 (n : ℕ) : ℕ :=
  (base_5_representation n).filter is_even |>.length

theorem count_even_digits_in_base_5_of_567 :
  count_even_digits_in_base_5 567 = 2 := by
  sorry

end NUMINAMATH_GPT_count_even_digits_in_base_5_of_567_l864_86415


namespace NUMINAMATH_GPT_average_score_l864_86406

theorem average_score (avg1 avg2 : ℕ) (n1 n2 total_matches : ℕ) (total_avg : ℕ) 
  (h1 : avg1 = 60) 
  (h2 : avg2 = 70) 
  (h3 : n1 = 10) 
  (h4 : n2 = 15) 
  (h5 : total_matches = 25) 
  (h6 : total_avg = 66) :
  (( (avg1 * n1) + (avg2 * n2) ) / total_matches = total_avg) :=
by
  sorry

end NUMINAMATH_GPT_average_score_l864_86406


namespace NUMINAMATH_GPT_average_distance_to_sides_l864_86463

open Real

noncomputable def side_length : ℝ := 15
noncomputable def diagonal_distance : ℝ := 9.3
noncomputable def right_turn_distance : ℝ := 3

theorem average_distance_to_sides :
  let d1 := 9.58
  let d2 := 6.58
  let d3 := 5.42
  let d4 := 8.42
  (d1 + d2 + d3 + d4) / 4 = 7.5 :=
by
  let d1 := 9.58
  let d2 := 6.58
  let d3 := 5.42
  let d4 := 8.42
  have h : (d1 + d2 + d3 + d4) / 4 = 7.5
  { sorry }
  exact h

end NUMINAMATH_GPT_average_distance_to_sides_l864_86463


namespace NUMINAMATH_GPT_isosceles_triangle_third_vertex_y_coord_l864_86403

theorem isosceles_triangle_third_vertex_y_coord :
  ∀ (A B : ℝ × ℝ) (θ : ℝ), 
  A = (0, 5) → B = (8, 5) → θ = 60 → 
  ∃ (C : ℝ × ℝ), C.fst > 0 ∧ C.snd > 5 ∧ C.snd = 5 + 4 * Real.sqrt 3 :=
by
  intros A B θ hA hB hθ
  use (4, 5 + 4 * Real.sqrt 3)
  sorry

end NUMINAMATH_GPT_isosceles_triangle_third_vertex_y_coord_l864_86403


namespace NUMINAMATH_GPT_phase_shift_right_by_pi_div_3_l864_86496

noncomputable def graph_shift_right_by_pi_div_3 
  (A : ℝ := 1) 
  (ω : ℝ := 1) 
  (φ : ℝ := - (Real.pi / 3)) 
  (y : ℝ → ℝ := fun x => Real.sin (x - Real.pi / 3)) : 
  Prop :=
  y = fun x => Real.sin (x - (Real.pi / 3))

theorem phase_shift_right_by_pi_div_3 (A : ℝ := 1) (ω : ℝ := 1) (φ : ℝ := - (Real.pi / 3)) :
  graph_shift_right_by_pi_div_3 A ω φ (fun x => Real.sin (x - Real.pi / 3)) :=
sorry

end NUMINAMATH_GPT_phase_shift_right_by_pi_div_3_l864_86496


namespace NUMINAMATH_GPT_find_MN_l864_86410

theorem find_MN (d D : ℝ) (h_d_lt_D : d < D) :
  ∃ MN : ℝ, MN = (d * D) / (D - d) :=
by
  sorry

end NUMINAMATH_GPT_find_MN_l864_86410


namespace NUMINAMATH_GPT_square_number_increased_decreased_by_five_remains_square_l864_86493

theorem square_number_increased_decreased_by_five_remains_square :
  ∃ x : ℤ, ∃ u v : ℤ, x^2 + 5 = u^2 ∧ x^2 - 5 = v^2 := by
  sorry

end NUMINAMATH_GPT_square_number_increased_decreased_by_five_remains_square_l864_86493


namespace NUMINAMATH_GPT_cube_probability_l864_86438

theorem cube_probability :
  let m := 1
  let n := 504
  ∀ (faces : Finset (Fin 6)) (nums : Finset (Fin 9)), 
    faces.card = 6 → nums.card = 9 →
    (∀ f ∈ faces, ∃ n ∈ nums, true) →
    m + n = 505 :=
by
  sorry

end NUMINAMATH_GPT_cube_probability_l864_86438


namespace NUMINAMATH_GPT_find_angle_l864_86425

theorem find_angle (θ : Real) (h1 : 0 ≤ θ ∧ θ ≤ π) (h2 : Real.sin θ = (Real.sqrt 2) / 2) :
  θ = Real.pi / 4 ∨ θ = 3 * Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_l864_86425


namespace NUMINAMATH_GPT_back_seat_capacity_l864_86467

def left_seats : Nat := 15
def right_seats : Nat := left_seats - 3
def seats_per_person : Nat := 3
def total_capacity : Nat := 92
def regular_seats_people : Nat := (left_seats + right_seats) * seats_per_person

theorem back_seat_capacity :
  total_capacity - regular_seats_people = 11 :=
by
  sorry

end NUMINAMATH_GPT_back_seat_capacity_l864_86467


namespace NUMINAMATH_GPT_second_competitor_distance_difference_l864_86413

theorem second_competitor_distance_difference (jump1 jump2 jump3 jump4 : ℕ) : 
  jump1 = 22 → 
  jump4 = 24 → 
  jump3 = jump2 - 2 → 
  jump4 = jump3 + 3 → 
  jump2 - jump1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_second_competitor_distance_difference_l864_86413


namespace NUMINAMATH_GPT_euler_no_k_divisible_l864_86486

theorem euler_no_k_divisible (n : ℕ) (k : ℕ) (h : k < 5^n - 5^(n-1)) : ¬ (5^n ∣ 2^k - 1) := 
sorry

end NUMINAMATH_GPT_euler_no_k_divisible_l864_86486


namespace NUMINAMATH_GPT_income_calculation_l864_86427

theorem income_calculation (x : ℕ) (h1 : ∃ x : ℕ, income = 8*x ∧ expenditure = 7*x)
  (h2 : savings = 5000)
  (h3 : income = expenditure + savings) : income = 40000 :=
by {
  sorry
}

end NUMINAMATH_GPT_income_calculation_l864_86427


namespace NUMINAMATH_GPT_sum_of_ages_l864_86432

theorem sum_of_ages (rachel_age leah_age : ℕ) 
  (h1 : rachel_age = leah_age + 4) 
  (h2 : rachel_age = 19) : rachel_age + leah_age = 34 :=
by
  -- Proof steps are omitted since we only need the statement
  sorry

end NUMINAMATH_GPT_sum_of_ages_l864_86432


namespace NUMINAMATH_GPT_x_is_perfect_square_l864_86466

theorem x_is_perfect_square (x y : ℕ) (hxy : x > y) (hdiv : xy ∣ x ^ 2022 + x + y ^ 2) : ∃ n : ℕ, x = n^2 := 
sorry

end NUMINAMATH_GPT_x_is_perfect_square_l864_86466


namespace NUMINAMATH_GPT_recreation_percentage_l864_86454

variable (W : ℝ) 

def recreation_last_week (W : ℝ) : ℝ := 0.10 * W
def wages_this_week (W : ℝ) : ℝ := 0.90 * W
def recreation_this_week (W : ℝ) : ℝ := 0.40 * (wages_this_week W)

theorem recreation_percentage : 
  (recreation_this_week W) / (recreation_last_week W) * 100 = 360 :=
by sorry

end NUMINAMATH_GPT_recreation_percentage_l864_86454


namespace NUMINAMATH_GPT_georgie_enter_and_exit_ways_l864_86459

-- Define the number of windows
def num_windows := 8

-- Define the magical barrier window
def barrier_window := 8

-- Define a function to count the number of ways Georgie can enter and exit the house
def count_ways_to_enter_and_exit : Nat :=
  let entry_choices := num_windows
  let exit_choices_from_normal := 6
  let exit_choices_from_barrier := 7
  let ways_from_normal := (entry_choices - 1) * exit_choices_from_normal  -- entering through windows 1 to 7
  let ways_from_barrier := 1 * exit_choices_from_barrier  -- entering through window 8
  ways_from_normal + ways_from_barrier

-- Prove the correct number of ways is 49
theorem georgie_enter_and_exit_ways : count_ways_to_enter_and_exit = 49 :=
by
  -- The calculation details are skipped with 'sorry'
  sorry

end NUMINAMATH_GPT_georgie_enter_and_exit_ways_l864_86459


namespace NUMINAMATH_GPT_min_ab_12_min_rec_expression_2_l864_86414

noncomputable def condition1 (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 / a + 3 / b = 1)

theorem min_ab_12 {a b : ℝ} (h : condition1 a b) : 
  a * b = 12 :=
sorry

theorem min_rec_expression_2 {a b : ℝ} (h : condition1 a b) :
  (1 / (a - 1)) + (3 / (b - 3)) = 2 :=
sorry

end NUMINAMATH_GPT_min_ab_12_min_rec_expression_2_l864_86414


namespace NUMINAMATH_GPT_students_in_school_l864_86452

variable (S : ℝ)
variable (W : ℝ)
variable (L : ℝ)

theorem students_in_school {S W L : ℝ} 
  (h1 : W = 0.55 * 0.25 * S)
  (h2 : L = 0.45 * 0.25 * S)
  (h3 : W = L + 50) : 
  S = 2000 := 
sorry

end NUMINAMATH_GPT_students_in_school_l864_86452


namespace NUMINAMATH_GPT_angle_expr_correct_l864_86482

noncomputable def angle_expr : Real :=
  Real.cos (40 * Real.pi / 180) * Real.cos (160 * Real.pi / 180) +
  Real.sin (40 * Real.pi / 180) * Real.sin (20 * Real.pi / 180)

theorem angle_expr_correct : angle_expr = -1 / 2 := 
by 
   sorry

end NUMINAMATH_GPT_angle_expr_correct_l864_86482


namespace NUMINAMATH_GPT_maximum_value_of_f_l864_86453

noncomputable def f (x : ℝ) : ℝ := ((x - 3) * (12 - x)) / x

theorem maximum_value_of_f :
  ∀ x : ℝ, 3 < x ∧ x < 12 → f x ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_l864_86453


namespace NUMINAMATH_GPT_difference_in_nickels_is_correct_l864_86436

variable (q : ℤ)

def charles_quarters : ℤ := 7 * q + 2
def richard_quarters : ℤ := 3 * q + 8

theorem difference_in_nickels_is_correct :
  5 * (charles_quarters - richard_quarters) = 20 * q - 30 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_nickels_is_correct_l864_86436


namespace NUMINAMATH_GPT_smallest_class_number_l864_86428

theorem smallest_class_number (sum_classes : ℕ) (n_classes interval number_of_classes : ℕ) 
                              (h_sum : sum_classes = 87) (h_n_classes : n_classes = 30) 
                              (h_interval : interval = 5) (h_number_of_classes : number_of_classes = 6) : 
                              ∃ x, x + (interval + x) + (2 * interval + x) + (3 * interval + x) 
                              + (4 * interval + x) + (5 * interval + x) = sum_classes ∧ x = 2 :=
by {
  use 2,
  sorry
}

end NUMINAMATH_GPT_smallest_class_number_l864_86428


namespace NUMINAMATH_GPT_smallest_positive_value_l864_86480

theorem smallest_positive_value (x : ℝ) (hx : x > 0) (h : x / 7 + 2 / (7 * x) = 1) : 
  x = (7 - Real.sqrt 41) / 2 :=
sorry

end NUMINAMATH_GPT_smallest_positive_value_l864_86480


namespace NUMINAMATH_GPT_determine_m_l864_86455

noncomputable def has_equal_real_roots (m : ℝ) : Prop :=
  m ≠ 0 ∧ (m^2 - 8 * m = 0)

theorem determine_m (m : ℝ) (h : has_equal_real_roots m) : m = 8 :=
  sorry

end NUMINAMATH_GPT_determine_m_l864_86455


namespace NUMINAMATH_GPT_polynomial_divisible_l864_86442

theorem polynomial_divisible (p q : ℤ) (h_p : p = -26) (h_q : q = 25) :
  ∀ x : ℤ, (x^4 + p*x^2 + q) % (x^2 - 6*x + 5) = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisible_l864_86442


namespace NUMINAMATH_GPT_alice_favorite_number_l864_86418

-- Define the conditions for Alice's favorite number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

-- Define the problem statement
theorem alice_favorite_number :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 200 ∧
           n % 13 = 0 ∧
           n % 3 ≠ 0 ∧
           sum_of_digits n % 4 = 0 ∧
           n = 130 :=
by
  sorry

end NUMINAMATH_GPT_alice_favorite_number_l864_86418


namespace NUMINAMATH_GPT_triangle_area_proof_l864_86465

noncomputable def cos_fun1 (x : ℝ) : ℝ := 2 * Real.cos (3 * x) + 1
noncomputable def cos_fun2 (x : ℝ) : ℝ := - Real.cos (2 * x)

theorem triangle_area_proof :
  let P := (5 * Real.pi, cos_fun1 (5 * Real.pi))
  let Q := (9 * Real.pi / 2, cos_fun2 (9 * Real.pi / 2))
  let m := (Q.snd - P.snd) / (Q.fst - P.fst)
  let y_intercept := P.snd - m * P.fst
  let y_intercept_point := (0, y_intercept)
  let x_intercept := -y_intercept / m
  let x_intercept_point := (x_intercept, 0)
  let base := x_intercept
  let height := y_intercept
  17 * Real.pi / 4 ≤ P.fst ∧ P.fst ≤ 21 * Real.pi / 4 ∧
  17 * Real.pi / 4 ≤ Q.fst ∧ Q.fst ≤ 21 * Real.pi / 4 ∧
  (P.fst = 5 * Real.pi ∧ Q.fst = 9 * Real.pi / 2) →
  1/2 * base * height = 361 * Real.pi / 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_proof_l864_86465


namespace NUMINAMATH_GPT_incorrect_observation_value_l864_86462

-- Definitions stemming from the given conditions
def initial_mean : ℝ := 100
def corrected_mean : ℝ := 99.075
def number_of_observations : ℕ := 40
def correct_observation_value : ℝ := 50

-- Lean theorem statement to prove the incorrect observation value
theorem incorrect_observation_value (initial_mean corrected_mean correct_observation_value : ℝ) (number_of_observations : ℕ) :
  (initial_mean * number_of_observations - corrected_mean * number_of_observations + correct_observation_value) = 87 := 
sorry

end NUMINAMATH_GPT_incorrect_observation_value_l864_86462


namespace NUMINAMATH_GPT_bread_cost_equality_l864_86464

variable (B : ℝ)
variable (C1 : B + 3 + 2 * B = 9)  -- $3 for butter, 2B for juice, total spent is 9 dollars

theorem bread_cost_equality : B = 2 :=
by
  sorry

end NUMINAMATH_GPT_bread_cost_equality_l864_86464
