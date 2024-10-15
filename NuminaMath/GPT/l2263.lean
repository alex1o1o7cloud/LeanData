import Mathlib

namespace NUMINAMATH_GPT_choose_copresidents_l2263_226357

theorem choose_copresidents (total_members : ℕ) (departments : ℕ) (members_per_department : ℕ) 
    (h1 : total_members = 24) (h2 : departments = 4) (h3 : members_per_department = 6) :
    ∃ ways : ℕ, ways = 54 :=
by
  sorry

end NUMINAMATH_GPT_choose_copresidents_l2263_226357


namespace NUMINAMATH_GPT_sufficient_condition_for_inequality_l2263_226305

theorem sufficient_condition_for_inequality (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) → a ≥ 5 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_inequality_l2263_226305


namespace NUMINAMATH_GPT_solve_inequality_min_value_F_l2263_226371

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 1)
def m := 3    -- Arbitrary constant, m + n = 7 implies n = 4
def n := 4

-- First statement: Solve the inequality f(x) ≥ (m + n)x
theorem solve_inequality (x : ℝ) : f x ≥ (m + n) * x ↔ x ≤ 0 := by
  sorry

noncomputable def F (x y : ℝ) : ℝ := max (abs (x^2 - 4 * y + m)) (abs (y^2 - 2 * x + n))

-- Second statement: Find the minimum value of F
theorem min_value_F (x y : ℝ) : (F x y) ≥ 1 ∧ (∃ x y, (F x y) = 1) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_min_value_F_l2263_226371


namespace NUMINAMATH_GPT_circle_radius_of_diameter_l2263_226347

theorem circle_radius_of_diameter (d : ℝ) (h : d = 22) : d / 2 = 11 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_of_diameter_l2263_226347


namespace NUMINAMATH_GPT_age_problem_l2263_226325

-- Define the ages of a, b, and c
variables (a b c : ℕ)

-- State the conditions
theorem age_problem (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 22) : b = 8 :=
by
  sorry

end NUMINAMATH_GPT_age_problem_l2263_226325


namespace NUMINAMATH_GPT_combined_seq_20th_term_l2263_226342

def arithmetic_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d
def geometric_seq (g : ℕ) (r : ℕ) (n : ℕ) : ℕ := g * r^(n - 1)

theorem combined_seq_20th_term :
  let a := 3
  let d := 4
  let g := 2
  let r := 2
  let n := 20
  arithmetic_seq a d n + geometric_seq g r n = 1048655 :=
by 
  sorry

end NUMINAMATH_GPT_combined_seq_20th_term_l2263_226342


namespace NUMINAMATH_GPT_find_n_l2263_226386

theorem find_n (n : ℕ) 
    (h : 6 * 4 * 3 * n = Nat.factorial 8) : n = 560 := 
sorry

end NUMINAMATH_GPT_find_n_l2263_226386


namespace NUMINAMATH_GPT_exists_number_between_70_and_80_with_gcd_10_l2263_226368

theorem exists_number_between_70_and_80_with_gcd_10 :
  ∃ n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 30 n = 10 :=
sorry

end NUMINAMATH_GPT_exists_number_between_70_and_80_with_gcd_10_l2263_226368


namespace NUMINAMATH_GPT_gain_percent_is_80_l2263_226308

theorem gain_percent_is_80 (C S : ℝ) (h : 81 * C = 45 * S) : ((S - C) / C) * 100 = 80 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_is_80_l2263_226308


namespace NUMINAMATH_GPT_largest_m_for_negative_integral_solutions_l2263_226362

theorem largest_m_for_negative_integral_solutions :
  ∃ m : ℕ, (∀ p q : ℤ, 10 * p * p + (-m) * p + 560 = 0 ∧ p < 0 ∧ q < 0 ∧ p * q = 56 → m ≤ 570) ∧ m = 570 :=
sorry

end NUMINAMATH_GPT_largest_m_for_negative_integral_solutions_l2263_226362


namespace NUMINAMATH_GPT_intersection_A_B_intersection_CR_A_B_l2263_226316

noncomputable def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
noncomputable def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
noncomputable def CR_A : Set ℝ := {x : ℝ | x < 3} ∪ {x : ℝ | 7 ≤ x}

theorem intersection_A_B :
  A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 7} :=
by
  sorry

theorem intersection_CR_A_B :
  CR_A ∩ B = ({x : ℝ | 2 < x ∧ x < 3} ∪ {x : ℝ | 7 ≤ x ∧ x < 10}) :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_intersection_CR_A_B_l2263_226316


namespace NUMINAMATH_GPT_domain_of_f_l2263_226376

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x)

theorem domain_of_f :
  {x : ℝ | f x = Real.log (x^2 - x)} = {x : ℝ | x < 0 ∨ x > 1} :=
sorry

end NUMINAMATH_GPT_domain_of_f_l2263_226376


namespace NUMINAMATH_GPT_car_speed_first_hour_l2263_226382

theorem car_speed_first_hour
  (x : ℕ)
  (speed_second_hour : ℕ := 80)
  (average_speed : ℕ := 90)
  (total_time : ℕ := 2)
  (h : average_speed = (x + speed_second_hour) / total_time) :
  x = 100 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_first_hour_l2263_226382


namespace NUMINAMATH_GPT_sqrt_200_eq_10_sqrt_2_l2263_226393

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_sqrt_200_eq_10_sqrt_2_l2263_226393


namespace NUMINAMATH_GPT_extraMaterialNeeded_l2263_226314

-- Box dimensions
def smallBoxLength (a : ℝ) : ℝ := a
def smallBoxWidth (b : ℝ) : ℝ := 1.5 * b
def smallBoxHeight (c : ℝ) : ℝ := c

def largeBoxLength (a : ℝ) : ℝ := 1.5 * a
def largeBoxWidth (b : ℝ) : ℝ := 2 * b
def largeBoxHeight (c : ℝ) : ℝ := 2 * c

-- Volume calculations
def volumeSmallBox (a b c : ℝ) : ℝ := a * (1.5 * b) * c
def volumeLargeBox (a b c : ℝ) : ℝ := (1.5 * a) * (2 * b) * (2 * c)

-- Surface area calculations
def surfaceAreaSmallBox (a b c : ℝ) : ℝ := 2 * (a * (1.5 * b)) + 2 * (a * c) + 2 * ((1.5 * b) * c)
def surfaceAreaLargeBox (a b c : ℝ) : ℝ := 2 * ((1.5 * a) * (2 * b)) + 2 * ((1.5 * a) * (2 * c)) + 2 * ((2 * b) * (2 * c))

-- Proof statement
theorem extraMaterialNeeded (a b c : ℝ) :
  (volumeSmallBox a b c = 1.5 * a * b * c) ∧ (volumeLargeBox a b c = 6 * a * b * c) ∧ 
  (surfaceAreaLargeBox a b c - surfaceAreaSmallBox a b c = 3 * a * b + 4 * a * c + 5 * b * c) :=
by
  sorry

end NUMINAMATH_GPT_extraMaterialNeeded_l2263_226314


namespace NUMINAMATH_GPT_ava_planted_9_trees_l2263_226340

theorem ava_planted_9_trees
  (L : ℕ)
  (hAva : ∀ L, Ava = L + 3)
  (hTotal : L + (L + 3) = 15) : 
  Ava = 9 :=
by
  sorry

end NUMINAMATH_GPT_ava_planted_9_trees_l2263_226340


namespace NUMINAMATH_GPT_find_angle_A_find_area_l2263_226303

noncomputable def triangle_area (a b c : ℝ) (A : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

theorem find_angle_A (a b c A : ℝ)
  (h1: ∀ x, 4 * Real.cos x * Real.sin (x - π/6) ≤ 4 * Real.cos A * Real.sin (A - π/6))
  (h2: a = b^2 + c^2 - 2 * b * c * Real.cos A) : 
  A = π / 3 := by
  sorry

theorem find_area (a b c : ℝ)
  (A : ℝ) (hA : A = π / 3)
  (ha : a = Real.sqrt 7) (hb : b = 2) 
  : triangle_area a b c A = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_GPT_find_angle_A_find_area_l2263_226303


namespace NUMINAMATH_GPT_determine_p_l2263_226313

-- Define the quadratic equation
def quadratic_eq (p x : ℝ) : ℝ := 3 * x^2 - 5 * (p - 1) * x + (p^2 + 2)

-- Define the conditions for the roots x1 and x2
def conditions (p x1 x2 : ℝ) : Prop :=
  quadratic_eq p x1 = 0 ∧
  quadratic_eq p x2 = 0 ∧
  x1 + 4 * x2 = 14

-- Define the theorem to prove the correct values of p
theorem determine_p (p : ℝ) (x1 x2 : ℝ) :
  conditions p x1 x2 → p = 742 / 127 ∨ p = 4 :=
by
  sorry

end NUMINAMATH_GPT_determine_p_l2263_226313


namespace NUMINAMATH_GPT_simplify_3_375_to_fraction_l2263_226301

def simplified_fraction_of_3_375 : ℚ := 3.375

theorem simplify_3_375_to_fraction : simplified_fraction_of_3_375 = 27 / 8 := 
by
  sorry

end NUMINAMATH_GPT_simplify_3_375_to_fraction_l2263_226301


namespace NUMINAMATH_GPT_geom_seq_sum_eqn_l2263_226375

theorem geom_seq_sum_eqn (n : ℕ) (a : ℚ) (r : ℚ) (S_n : ℚ) : 
  a = 1/3 → r = 1/3 → S_n = 80/243 → S_n = a * (1 - r^n) / (1 - r) → n = 5 :=
by
  intros ha hr hSn hSum
  sorry

end NUMINAMATH_GPT_geom_seq_sum_eqn_l2263_226375


namespace NUMINAMATH_GPT_max_area_rectangle_l2263_226373

theorem max_area_rectangle (P : ℝ) (x : ℝ) (h1 : P = 40) (h2 : 6 * x = P) : 
  2 * (x ^ 2) = 800 / 9 :=
by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_l2263_226373


namespace NUMINAMATH_GPT_vacation_trip_l2263_226380

theorem vacation_trip (airbnb_cost : ℕ) (car_rental_cost : ℕ) (share_per_person : ℕ) (total_people : ℕ) :
  airbnb_cost = 3200 → car_rental_cost = 800 → share_per_person = 500 → airbnb_cost + car_rental_cost / share_per_person = 8 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_vacation_trip_l2263_226380


namespace NUMINAMATH_GPT_sunzi_oranges_l2263_226318

theorem sunzi_oranges :
  ∃ (a : ℕ), ( 5 * a + 10 * 3 = 60 ) ∧ ( ∀ n, n = 0 → a = 6 ) :=
by
  sorry

end NUMINAMATH_GPT_sunzi_oranges_l2263_226318


namespace NUMINAMATH_GPT_f_is_periodic_with_period_4a_l2263_226388

variable (f : ℝ → ℝ) (a : ℝ)

theorem f_is_periodic_with_period_4a (h : ∀ x : ℝ, f (x + a) = (1 + f x) / (1 - f x)) : ∀ x : ℝ, f (x + 4 * a) = f x :=
by
  sorry

end NUMINAMATH_GPT_f_is_periodic_with_period_4a_l2263_226388


namespace NUMINAMATH_GPT_expression_value_l2263_226387

theorem expression_value : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l2263_226387


namespace NUMINAMATH_GPT_total_space_after_compaction_correct_l2263_226330

noncomputable def problem : Prop :=
  let num_small_cans := 50
  let num_large_cans := 50
  let small_can_size := 20
  let large_can_size := 40
  let small_can_compaction := 0.30
  let large_can_compaction := 0.40
  let small_cans_compacted := num_small_cans * small_can_size * small_can_compaction
  let large_cans_compacted := num_large_cans * large_can_size * large_can_compaction
  let total_space_after_compaction := small_cans_compacted + large_cans_compacted
  total_space_after_compaction = 1100

theorem total_space_after_compaction_correct :
  problem :=
  by
    unfold problem
    sorry

end NUMINAMATH_GPT_total_space_after_compaction_correct_l2263_226330


namespace NUMINAMATH_GPT_x_one_minus_f_eq_one_l2263_226326

noncomputable def x : ℝ := (1 + Real.sqrt 2) ^ 500
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem x_one_minus_f_eq_one : x * (1 - f) = 1 :=
by
  sorry

end NUMINAMATH_GPT_x_one_minus_f_eq_one_l2263_226326


namespace NUMINAMATH_GPT_simplify_expression_l2263_226374

theorem simplify_expression (z : ℝ) : (3 - 5 * z^2) - (4 * z^2 + 2 * z - 5) = 8 - 9 * z^2 - 2 * z :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2263_226374


namespace NUMINAMATH_GPT_gianna_saved_for_365_days_l2263_226315

-- Define the total amount saved and the amount saved each day
def total_amount_saved : ℕ := 14235
def amount_saved_each_day : ℕ := 39

-- Define the problem statement to prove the number of days saved
theorem gianna_saved_for_365_days :
  (total_amount_saved / amount_saved_each_day) = 365 :=
sorry

end NUMINAMATH_GPT_gianna_saved_for_365_days_l2263_226315


namespace NUMINAMATH_GPT_pos_diff_is_multiple_of_9_l2263_226304

theorem pos_diff_is_multiple_of_9 
  (q r : ℕ) 
  (h_qr : 10 ≤ q ∧ q < 100 ∧ 10 ≤ r ∧ r < 100 ∧ (q % 10) * 10 + (q / 10) = r)
  (h_max_diff : q - r = 63) : 
  ∃ k : ℕ, q - r = 9 * k :=
by
  sorry

end NUMINAMATH_GPT_pos_diff_is_multiple_of_9_l2263_226304


namespace NUMINAMATH_GPT_solve_equation_l2263_226331

theorem solve_equation (x a b : ℝ) (h : x^2 - 6*x + 11 = 27) (sol_a : a = 8) (sol_b : b = -2) :
  3 * a - 2 * b = 28 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2263_226331


namespace NUMINAMATH_GPT_vec_same_direction_l2263_226349

theorem vec_same_direction (k : ℝ) : (k = 2) ↔ ∃ m : ℝ, m > 0 ∧ (k, 2) = (m * 1, m * 1) :=
by
  sorry

end NUMINAMATH_GPT_vec_same_direction_l2263_226349


namespace NUMINAMATH_GPT_round_trip_completion_percentage_l2263_226385

-- Define the distances for each section
def sectionA_distance : Float := 10
def sectionB_distance : Float := 20
def sectionC_distance : Float := 15

-- Define the speeds for each section
def sectionA_speed : Float := 50
def sectionB_speed : Float := 40
def sectionC_speed : Float := 60

-- Define the delays for each section
def sectionA_delay : Float := 1.15
def sectionB_delay : Float := 1.10

-- Calculate the time for each section without delays
def sectionA_time : Float := sectionA_distance / sectionA_speed
def sectionB_time : Float := sectionB_distance / sectionB_speed
def sectionC_time : Float := sectionC_distance / sectionC_speed

-- Calculate the time with delays for the trip to the center
def sectionA_time_with_delay : Float := sectionA_time * sectionA_delay
def sectionB_time_with_delay : Float := sectionB_time * sectionB_delay
def sectionC_time_with_delay : Float := sectionC_time

-- Total time with delays to the center
def total_time_to_center : Float := sectionA_time_with_delay + sectionB_time_with_delay + sectionC_time_with_delay

-- Total distance to the center
def total_distance_to_center : Float := sectionA_distance + sectionB_distance + sectionC_distance

-- Total round trip distance
def total_round_trip_distance : Float := total_distance_to_center * 2

-- Distance covered on the way back
def distance_back : Float := total_distance_to_center * 0.2

-- Total distance covered considering the delays and the return trip
def total_distance_covered : Float := total_distance_to_center + distance_back

-- Effective completion percentage of the round trip
def completion_percentage : Float := (total_distance_covered / total_round_trip_distance) * 100

-- The main theorem statement
theorem round_trip_completion_percentage :
  completion_percentage = 60 := by
  sorry

end NUMINAMATH_GPT_round_trip_completion_percentage_l2263_226385


namespace NUMINAMATH_GPT_nickys_running_pace_l2263_226379

theorem nickys_running_pace (head_start : ℕ) (pace_cristina : ℕ) (time_nicky : ℕ) (distance_meet : ℕ) :
  head_start = 12 →
  pace_cristina = 5 →
  time_nicky = 30 →
  distance_meet = (pace_cristina * (time_nicky - head_start)) →
  (distance_meet / time_nicky = 3) :=
by
  intros h_start h_pace_c h_time_n d_meet
  sorry

end NUMINAMATH_GPT_nickys_running_pace_l2263_226379


namespace NUMINAMATH_GPT_tree_height_increase_fraction_l2263_226341

theorem tree_height_increase_fraction :
  ∀ (initial_height annual_increase : ℝ) (additional_years₄ additional_years₆ : ℕ),
    initial_height = 4 →
    annual_increase = 0.4 →
    additional_years₄ = 4 →
    additional_years₆ = 6 →
    ((initial_height + annual_increase * additional_years₆) - (initial_height + annual_increase * additional_years₄)) / (initial_height + annual_increase * additional_years₄) = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_tree_height_increase_fraction_l2263_226341


namespace NUMINAMATH_GPT_percentage_error_is_94_l2263_226332

theorem percentage_error_is_94 (x : ℝ) (hx : 0 < x) :
  let correct_result := 4 * x
  let error_result := x / 4
  let error := |correct_result - error_result|
  let percentage_error := (error / correct_result) * 100
  percentage_error = 93.75 := by
    sorry

end NUMINAMATH_GPT_percentage_error_is_94_l2263_226332


namespace NUMINAMATH_GPT_problem1_problem2_l2263_226337

def M (x : ℝ) : Prop := (x + 5) / (x - 8) ≥ 0

def N (x : ℝ) (a : ℝ) : Prop := a - 1 ≤ x ∧ x ≤ a + 1

theorem problem1 : ∀ (x : ℝ), (M x ∨ (N x 9)) ↔ (x ≤ -5 ∨ x ≥ 8) :=
by
  sorry

theorem problem2 : ∀ (a : ℝ), (∀ (x : ℝ), N x a → M x) ↔ (a ≤ -6 ∨ 9 < a) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2263_226337


namespace NUMINAMATH_GPT_binom_two_formula_l2263_226355

def binom (n k : ℕ) : ℕ :=
  n.choose k

-- Formalizing the conditions
variable (n : ℕ)
variable (h : n ≥ 2)

-- Stating the problem mathematically in Lean
theorem binom_two_formula :
  binom n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_GPT_binom_two_formula_l2263_226355


namespace NUMINAMATH_GPT_cost_of_flowers_cost_function_minimum_cost_l2263_226356

-- Define the costs in terms of yuan
variables (n m : ℕ) -- n is the cost of one lily, m is the cost of one carnation.

-- Define the conditions
axiom cost_condition1 : 2 * n + m = 14
axiom cost_condition2 : 3 * m = 2 * n + 2

-- Prove the cost of one carnation and one lily
theorem cost_of_flowers : n = 5 ∧ m = 4 :=
by {
  sorry
}

-- Variables for the second part
variables (w x : ℕ) -- w is the total cost, x is the number of carnations.

-- Define the conditions
axiom total_condition : 11 = 2 + x + (11 - x)
axiom min_lilies_condition : 11 - x ≥ 2

-- State the relationship between w and x
theorem cost_function : w = 55 - x :=
by {
  sorry
}

-- Prove the minimum cost
theorem minimum_cost : ∃ x, (x ≤ 9 ∧  w = 46) :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_flowers_cost_function_minimum_cost_l2263_226356


namespace NUMINAMATH_GPT_max_profit_l2263_226348

-- Definitions based on conditions from the problem
def L1 (x : ℕ) : ℤ := -5 * (x : ℤ)^2 + 900 * (x : ℤ) - 16000
def L2 (x : ℕ) : ℤ := 300 * (x : ℤ) - 2000
def total_vehicles := 110
def total_profit (x : ℕ) : ℤ := L1 x + L2 (total_vehicles - x)

-- Statement of the problem
theorem max_profit :
  ∃ x y : ℕ, x + y = 110 ∧ x ≥ 0 ∧ y ≥ 0 ∧
  (L1 x + L2 y = 33000 ∧
   (∀ z w : ℕ, z + w = 110 ∧ z ≥ 0 ∧ w ≥ 0 → L1 z + L2 w ≤ 33000)) :=
sorry

end NUMINAMATH_GPT_max_profit_l2263_226348


namespace NUMINAMATH_GPT_emily_irises_after_addition_l2263_226353

theorem emily_irises_after_addition
  (initial_roses : ℕ)
  (added_roses : ℕ)
  (ratio_irises_roses : ℕ)
  (ratio_roses_irises : ℕ)
  (h_ratio : ratio_irises_roses = 3 ∧ ratio_roses_irises = 7)
  (h_initial_roses : initial_roses = 35)
  (h_added_roses : added_roses = 30) :
  ∃ irises_after_addition : ℕ, irises_after_addition = 27 :=
  by
    sorry

end NUMINAMATH_GPT_emily_irises_after_addition_l2263_226353


namespace NUMINAMATH_GPT_problem_trapezoid_l2263_226344

noncomputable def ratio_of_areas (AB CD : ℝ) (h : ℝ) (ratio : ℝ) :=
  let area_trapezoid := (AB + CD) * h / 2
  let area_triangle_AZW := (4 * h) / 15
  ratio = area_triangle_AZW / area_trapezoid

theorem problem_trapezoid :
  ratio_of_areas 2 5 h (8 / 105) :=
by
  sorry

end NUMINAMATH_GPT_problem_trapezoid_l2263_226344


namespace NUMINAMATH_GPT_range_of_m_l2263_226319

variable (m : ℝ)

def p : Prop := m + 1 ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h : ¬ (p m ∧ q m)) : m ≤ -2 ∨ m > -1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2263_226319


namespace NUMINAMATH_GPT_rectangle_ratio_l2263_226372

theorem rectangle_ratio (s y x : ℝ) 
  (inner_square_area outer_square_area : ℝ) 
  (h1 : inner_square_area = s^2)
  (h2 : outer_square_area = 9 * inner_square_area)
  (h3 : outer_square_area = (3 * s)^2)
  (h4 : s + 2 * y = 3 * s)
  (h5 : x + y = 3 * s)
  : x / y = 2 := 
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l2263_226372


namespace NUMINAMATH_GPT_inverse_contrapositive_l2263_226324

theorem inverse_contrapositive (a b c : ℝ) (h : a > b → a + c > b + c) :
  a + c ≤ b + c → a ≤ b :=
sorry

end NUMINAMATH_GPT_inverse_contrapositive_l2263_226324


namespace NUMINAMATH_GPT_gaeun_taller_than_nana_l2263_226358

def nana_height_m : ℝ := 1.618
def gaeun_height_cm : ℝ := 162.3
def nana_height_cm : ℝ := nana_height_m * 100

theorem gaeun_taller_than_nana : gaeun_height_cm - nana_height_cm = 0.5 := by
  sorry

end NUMINAMATH_GPT_gaeun_taller_than_nana_l2263_226358


namespace NUMINAMATH_GPT_find_x_y_l2263_226370

theorem find_x_y (x y : ℝ)
  (h1 : (x - 1) ^ 2003 + 2002 * (x - 1) = -1)
  (h2 : (y - 2) ^ 2003 + 2002 * (y - 2) = 1) :
  x + y = 3 :=
sorry

end NUMINAMATH_GPT_find_x_y_l2263_226370


namespace NUMINAMATH_GPT_range_of_m_l2263_226381

theorem range_of_m (m x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (h3 : (1 - 2 * m) / x1 < (1 - 2 * m) / x2) : m < 1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2263_226381


namespace NUMINAMATH_GPT_intersection_N_complement_M_l2263_226364

def U : Set ℝ := Set.univ
def M : Set ℝ := {x : ℝ | x < -2 ∨ x > 2}
def CU_M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | (1 - x) / (x - 3) > 0}

theorem intersection_N_complement_M :
  N ∩ CU_M = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_intersection_N_complement_M_l2263_226364


namespace NUMINAMATH_GPT_value_of_4k_minus_1_l2263_226327

theorem value_of_4k_minus_1 (k x y : ℝ)
  (h1 : x + y - 5 * k = 0)
  (h2 : x - y - 9 * k = 0)
  (h3 : 2 * x + 3 * y = 6) :
  4 * k - 1 = 2 :=
  sorry

end NUMINAMATH_GPT_value_of_4k_minus_1_l2263_226327


namespace NUMINAMATH_GPT_textbook_weight_ratio_l2263_226394

def jon_textbooks_weights : List ℕ := [2, 8, 5, 9]
def brandon_textbooks_weight : ℕ := 8

theorem textbook_weight_ratio : 
  (jon_textbooks_weights.sum : ℚ) / (brandon_textbooks_weight : ℚ) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_textbook_weight_ratio_l2263_226394


namespace NUMINAMATH_GPT_marked_price_l2263_226329

theorem marked_price (initial_price : ℝ) (discount_percent : ℝ) (profit_margin_percent : ℝ) (final_discount_percent : ℝ) (marked_price : ℝ) :
  initial_price = 40 → 
  discount_percent = 0.25 → 
  profit_margin_percent = 0.50 → 
  final_discount_percent = 0.10 → 
  marked_price = 50 := by
  sorry

end NUMINAMATH_GPT_marked_price_l2263_226329


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_relation_l2263_226384

theorem arithmetic_geometric_sequence_relation 
  (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (hA : ∀ n: ℕ, a (n + 1) - a n = a 1) 
  (hG : ∀ n: ℕ, b (n + 1) / b n = b 1) 
  (h1 : a 1 = b 1) 
  (h11 : a 11 = b 11) 
  (h_pos : 0 < a 1 ∧ 0 < a 11 ∧ 0 < b 11 ∧ 0 < b 1) :
  a 6 ≥ b 6 := sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_relation_l2263_226384


namespace NUMINAMATH_GPT_Sandra_brought_20_pairs_l2263_226363

-- Definitions for given conditions
variable (S : ℕ) -- S for Sandra's pairs of socks
variable (C : ℕ) -- C for Lisa's cousin's pairs of socks

-- Conditions translated into Lean definitions
def initial_pairs : ℕ := 12
def mom_pairs : ℕ := 3 * initial_pairs + 8 -- Lisa's mom brought 8 more than three times the number of pairs Lisa started with
def cousin_pairs (S : ℕ) : ℕ := S / 5       -- Lisa's cousin brought one-fifth the number of pairs that Sandra bought
def total_pairs (S : ℕ) : ℕ := initial_pairs + S + cousin_pairs S + mom_pairs -- Total pairs of socks Lisa ended up with

-- The theorem to prove
theorem Sandra_brought_20_pairs (h : total_pairs S = 80) : S = 20 :=
by
  sorry

end NUMINAMATH_GPT_Sandra_brought_20_pairs_l2263_226363


namespace NUMINAMATH_GPT_average_speed_round_trip_l2263_226398

theorem average_speed_round_trip (v1 v2 : ℝ) (h1 : v1 = 60) (h2 : v2 = 100) :
  (2 * v1 * v2) / (v1 + v2) = 75 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_round_trip_l2263_226398


namespace NUMINAMATH_GPT_annika_total_distance_l2263_226354

/--
Annika hikes at a constant rate of 12 minutes per kilometer. She has hiked 2.75 kilometers
east from the start of a hiking trail when she realizes that she has to be back at the start
of the trail in 51 minutes. Prove that the total distance Annika hiked east is 3.5 kilometers.
-/
theorem annika_total_distance :
  (hike_rate : ℝ) = 12 → 
  (initial_distance_east : ℝ) = 2.75 → 
  (total_time : ℝ) = 51 → 
  (total_distance_east : ℝ) = 3.5 :=
by 
  intro hike_rate initial_distance_east total_time 
  sorry

end NUMINAMATH_GPT_annika_total_distance_l2263_226354


namespace NUMINAMATH_GPT_likes_spinach_not_music_lover_l2263_226309

universe u

variable (Person : Type u)
variable (likes_spinach is_pearl_diver is_music_lover : Person → Prop)

theorem likes_spinach_not_music_lover :
  (∃ x, likes_spinach x ∧ ¬ is_pearl_diver x) →
  (∀ x, is_music_lover x → (is_pearl_diver x ∨ ¬ likes_spinach x)) →
  (∀ x, (¬ is_pearl_diver x → is_music_lover x) ∨ (is_pearl_diver x → ¬ is_music_lover x)) →
  (∀ x, likes_spinach x → ¬ is_music_lover x) :=
by
  sorry

end NUMINAMATH_GPT_likes_spinach_not_music_lover_l2263_226309


namespace NUMINAMATH_GPT_lanies_salary_l2263_226322

variables (hours_worked_per_week : ℚ) (hourly_rate : ℚ)

namespace Lanie
def salary (fraction_of_weekly_hours : ℚ) : ℚ :=
  (fraction_of_weekly_hours * hours_worked_per_week) * hourly_rate

theorem lanies_salary : 
  hours_worked_per_week = 40 ∧
  hourly_rate = 15 ∧
  fraction_of_weekly_hours = 4 / 5 →
  salary fraction_of_weekly_hours = 480 :=
by
  -- Proof steps go here
  sorry
end Lanie

end NUMINAMATH_GPT_lanies_salary_l2263_226322


namespace NUMINAMATH_GPT_eventually_one_student_answers_yes_l2263_226383

-- Conditions and Definitions
variable (a b r₁ r₂ : ℕ)
variable (h₁ : r₁ ≠ r₂)   -- r₁ and r₂ are distinct
variable (h₂ : r₁ = a + b ∨ r₂ = a + b) -- One of r₁ or r₂ is the sum a + b
variable (h₃ : a > 0) -- a is a positive integer
variable (h₄ : b > 0) -- b is a positive integer

theorem eventually_one_student_answers_yes (a b r₁ r₂ : ℕ) (h₁ : r₁ ≠ r₂) (h₂ : r₁ = a + b ∨ r₂ = a + b) (h₃ : a > 0) (h₄ : b > 0) :
  ∃ n : ℕ, (∃ c : ℕ, (r₁ = c + b ∨ r₂ = c + b) ∧ (c = a ∨ c ≤ r₁ ∨ c ≤ r₂)) ∨ 
  (∃ c : ℕ, (r₁ = a + c ∨ r₂ = a + c) ∧ (c = b ∨ c ≤ r₁ ∨ c ≤ r₂)) :=
sorry

end NUMINAMATH_GPT_eventually_one_student_answers_yes_l2263_226383


namespace NUMINAMATH_GPT_prime_numbers_r_s_sum_l2263_226339

theorem prime_numbers_r_s_sum (p q r s : ℕ) (hp : Fact (Nat.Prime p)) (hq : Fact (Nat.Prime q)) 
  (hr : Fact (Nat.Prime r)) (hs : Fact (Nat.Prime s)) (h1 : p < q) (h2 : q < r) (h3 : r < s) 
  (eqn : p * q * r * s + 1 = 4^(p + q)) : r + s = 274 :=
by
  sorry

end NUMINAMATH_GPT_prime_numbers_r_s_sum_l2263_226339


namespace NUMINAMATH_GPT_prob_neither_defective_l2263_226366

-- Definitions for the conditions
def totalPens : ℕ := 8
def defectivePens : ℕ := 2
def nonDefectivePens : ℕ := totalPens - defectivePens
def selectedPens : ℕ := 2

-- Theorem statement for the probability that neither of the two selected pens is defective
theorem prob_neither_defective : 
  (nonDefectivePens / totalPens) * ((nonDefectivePens - 1) / (totalPens - 1)) = 15 / 28 := 
  sorry

end NUMINAMATH_GPT_prob_neither_defective_l2263_226366


namespace NUMINAMATH_GPT_analytical_expression_of_f_range_of_f_on_interval_l2263_226317

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := a * x^3 + c * x

theorem analytical_expression_of_f
  (a c : ℝ)
  (h1 : a > 0)
  (h2 : ∀ x, f x a c = a * x^3 + c * x) 
  (h3 : 3 * a + c = -6)
  (h4 : ∀ x, (3 * a * x ^ 2 + c) ≥ -12) :
    a = 2 ∧ c = -12 :=
by
  sorry

theorem range_of_f_on_interval
  (h1 : ∃ a c, a = 2 ∧ c = -12)
  (h2 : ∀ x, f x 2 (-12) = 2 * x^3 - 12 * x)
  :
    Set.range (fun x => f x 2 (-12)) = Set.Icc (-8 * Real.sqrt 2) (8 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_analytical_expression_of_f_range_of_f_on_interval_l2263_226317


namespace NUMINAMATH_GPT_symmetrical_ring_of_polygons_l2263_226369

theorem symmetrical_ring_of_polygons (m n : ℕ) (hn : n ≥ 7) (hm : m ≥ 3) 
  (condition1 : ∀ p1 p2 : ℕ, p1 ≠ p2 → n = 1) 
  (condition2 : ∀ p : ℕ, p * (n - 2) = 4) 
  (condition3 : ∀ p : ℕ, 2 * m - (n - 2) = 4) :
  ∃ k, (k = 6) :=
by
  -- This block is only a placeholder. The actual proof would go here.
  sorry

end NUMINAMATH_GPT_symmetrical_ring_of_polygons_l2263_226369


namespace NUMINAMATH_GPT_alan_tickets_l2263_226310

variables (A M : ℕ)

def condition1 := A + M = 150
def condition2 := M = 5 * A - 6

theorem alan_tickets : A = 26 :=
by
  have h1 : condition1 A M := sorry
  have h2 : condition2 A M := sorry
  sorry

end NUMINAMATH_GPT_alan_tickets_l2263_226310


namespace NUMINAMATH_GPT_school_accomodation_proof_l2263_226345

theorem school_accomodation_proof
  (total_classrooms : ℕ) 
  (fraction_classrooms_45 : ℕ) 
  (fraction_classrooms_38 : ℕ)
  (fraction_classrooms_32 : ℕ)
  (fraction_classrooms_25 : ℕ)
  (desks_45 : ℕ)
  (desks_38 : ℕ)
  (desks_32 : ℕ)
  (desks_25 : ℕ)
  (student_capacity_limit : ℕ) :
  total_classrooms = 50 ->
  fraction_classrooms_45 = (3 / 10) * total_classrooms -> 
  fraction_classrooms_38 = (1 / 4) * total_classrooms -> 
  fraction_classrooms_32 = (1 / 5) * total_classrooms -> 
  fraction_classrooms_25 = (total_classrooms - fraction_classrooms_45 - fraction_classrooms_38 - fraction_classrooms_32) ->
  desks_45 = 15 * 45 -> 
  desks_38 = 12 * 38 -> 
  desks_32 = 10 * 32 -> 
  desks_25 = fraction_classrooms_25 * 25 -> 
  student_capacity_limit = 1800 -> 
  fraction_classrooms_45 * 45 +
  fraction_classrooms_38 * 38 +
  fraction_classrooms_32 * 32 + 
  fraction_classrooms_25 * 25 = 1776 + sorry
  :=
sorry

end NUMINAMATH_GPT_school_accomodation_proof_l2263_226345


namespace NUMINAMATH_GPT_nebraska_license_plate_increase_l2263_226360

open Nat

theorem nebraska_license_plate_increase :
  let old_plates : ℕ := 26 * 10^3
  let new_plates : ℕ := 26^2 * 10^4
  new_plates / old_plates = 260 :=
by
  -- Definitions based on conditions
  let old_plates : ℕ := 26 * 10^3
  let new_plates : ℕ := 26^2 * 10^4
  -- Assertion to prove
  show new_plates / old_plates = 260
  sorry

end NUMINAMATH_GPT_nebraska_license_plate_increase_l2263_226360


namespace NUMINAMATH_GPT_polar_to_cartesian_2_pi_over_6_l2263_226321

theorem polar_to_cartesian_2_pi_over_6 :
  let r : ℝ := 2
  let θ : ℝ := (Real.pi / 6)
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (Real.sqrt 3, 1) := by
    -- Initialize the constants and their values
    let r := 2
    let θ := Real.pi / 6
    let x := r * Real.cos θ
    let y := r * Real.sin θ
    -- Placeholder for the actual proof
    sorry

end NUMINAMATH_GPT_polar_to_cartesian_2_pi_over_6_l2263_226321


namespace NUMINAMATH_GPT_angle_B_is_pi_over_3_l2263_226333

theorem angle_B_is_pi_over_3
  (A B C : ℝ) (a b c : ℝ)
  (h_triangle : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)
  (h_sin_ratios : ∃ k > 0, a = 5*k ∧ b = 7*k ∧ c = 8*k) :
  B = π / 3 := 
by
  sorry

end NUMINAMATH_GPT_angle_B_is_pi_over_3_l2263_226333


namespace NUMINAMATH_GPT_count_multiples_of_70_in_range_200_to_500_l2263_226338

theorem count_multiples_of_70_in_range_200_to_500 : 
  ∃! count, count = 5 ∧ (∀ n, 200 ≤ n ∧ n ≤ 500 ∧ (n % 70 = 0) ↔ n = 210 ∨ n = 280 ∨ n = 350 ∨ n = 420 ∨ n = 490) :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_of_70_in_range_200_to_500_l2263_226338


namespace NUMINAMATH_GPT_printer_task_total_pages_l2263_226346

theorem printer_task_total_pages
  (A B : ℕ)
  (h1 : 1 / A + 1 / B = 1 / 24)
  (h2 : 1 / A = 1 / 60)
  (h3 : B = A + 6) :
  60 * A = 720 := by
  sorry

end NUMINAMATH_GPT_printer_task_total_pages_l2263_226346


namespace NUMINAMATH_GPT_expressions_positive_l2263_226351

-- Definitions based on given conditions
def A := 2.5
def B := -0.8
def C := -2.2
def D := 1.1
def E := -3.1

-- The Lean statement to prove the necessary expressions are positive numbers.

theorem expressions_positive :
  (B + C) / E = 0.97 ∧
  B * D - A * C = 4.62 ∧
  C / (A * B) = 1.1 :=
by
  -- Assuming given conditions and steps to prove the theorem.
  sorry

end NUMINAMATH_GPT_expressions_positive_l2263_226351


namespace NUMINAMATH_GPT_B_is_empty_l2263_226312

def A : Set ℤ := {0}
def B : Set ℤ := {x | x > 8 ∧ x < 5}
def C : Set ℕ := {x | x - 1 = 0}
def D : Set ℤ := {x | x > 4}

theorem B_is_empty : B = ∅ := by
  sorry

end NUMINAMATH_GPT_B_is_empty_l2263_226312


namespace NUMINAMATH_GPT_compare_diff_functions_l2263_226302

variable {R : Type*} [LinearOrderedField R]
variable {f g : R → R}
variable (h_fg : ∀ x, f' x > g' x)
variable {x1 x2 : R}

theorem compare_diff_functions (h : x1 < x2) : f x1 - f x2 < g x1 - g x2 :=
  sorry

end NUMINAMATH_GPT_compare_diff_functions_l2263_226302


namespace NUMINAMATH_GPT_max_marks_test_l2263_226367

theorem max_marks_test (M : ℝ) : 
  (0.30 * M = 80 + 100) -> 
  M = 600 :=
by 
  sorry

end NUMINAMATH_GPT_max_marks_test_l2263_226367


namespace NUMINAMATH_GPT_correct_fraction_l2263_226320

theorem correct_fraction (x y : ℕ) (h1 : 480 * 5 / 6 = 480 * x / y + 250) : x / y = 5 / 16 :=
by
  sorry

end NUMINAMATH_GPT_correct_fraction_l2263_226320


namespace NUMINAMATH_GPT_mean_value_z_l2263_226365

theorem mean_value_z (z : ℚ) (h : (7 + 10 + 23) / 3 = (18 + z) / 2) : z = 26 / 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_value_z_l2263_226365


namespace NUMINAMATH_GPT_sum_of_squares_of_ages_eq_35_l2263_226399

theorem sum_of_squares_of_ages_eq_35
  (d t h : ℕ)
  (h1 : 3 * d + 4 * t = 2 * h + 2)
  (h2 : 2 * d^2 + t^2 = 6 * h)
  (relatively_prime : Nat.gcd (Nat.gcd d t) h = 1) :
  d^2 + t^2 + h^2 = 35 := 
sorry

end NUMINAMATH_GPT_sum_of_squares_of_ages_eq_35_l2263_226399


namespace NUMINAMATH_GPT_number_of_children_l2263_226395

-- Definitions based on the conditions provided in the problem
def total_population : ℕ := 8243
def grown_ups : ℕ := 5256

-- Statement of the proof problem
theorem number_of_children : total_population - grown_ups = 2987 := by
-- This placeholder 'sorry' indicates that the proof is omitted.
sorry

end NUMINAMATH_GPT_number_of_children_l2263_226395


namespace NUMINAMATH_GPT_find_f_two_l2263_226397

-- The function f is defined on (0, +∞) and takes positive values
noncomputable def f : ℝ → ℝ := sorry

-- The given condition that areas of triangle AOB and trapezoid ABH_BH_A are equal
axiom equalAreas (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) : 
  (1 / 2) * |x1 * f x2 - x2 * f x1| = (1 / 2) * (x2 - x1) * (f x1 + f x2)

-- The specific given value
axiom f_one : f 1 = 4

-- The theorem we need to prove
theorem find_f_two : f 2 = 2 :=
sorry

end NUMINAMATH_GPT_find_f_two_l2263_226397


namespace NUMINAMATH_GPT_interest_calculation_years_l2263_226392

noncomputable def principal : ℝ := 625
noncomputable def rate : ℝ := 0.04
noncomputable def difference : ℝ := 1

theorem interest_calculation_years (n : ℕ) : 
    (principal * (1 + rate)^n - principal - (principal * rate * n) = difference) → 
    n = 2 :=
by sorry

end NUMINAMATH_GPT_interest_calculation_years_l2263_226392


namespace NUMINAMATH_GPT_find_whole_number_N_l2263_226334

theorem find_whole_number_N (N : ℕ) (h1 : 6.75 < (N / 4 : ℝ)) (h2 : (N / 4 : ℝ) < 7.25) : N = 28 := 
by 
  sorry

end NUMINAMATH_GPT_find_whole_number_N_l2263_226334


namespace NUMINAMATH_GPT_geom_seq_seventh_term_l2263_226378

theorem geom_seq_seventh_term (a r : ℝ) (n : ℕ) (h1 : a = 2) (h2 : r^8 * a = 32) :
  a * r^6 = 128 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_seventh_term_l2263_226378


namespace NUMINAMATH_GPT_maximize_magnitude_l2263_226390

theorem maximize_magnitude (a x y : ℝ) 
(h1 : 4 * x^2 + 4 * y^2 = -a^2 + 16 * a - 32)
(h2 : 2 * x * y = a) : a = 8 := 
sorry

end NUMINAMATH_GPT_maximize_magnitude_l2263_226390


namespace NUMINAMATH_GPT_square_area_l2263_226377

theorem square_area (s : ℝ) (h : s = 12) : s * s = 144 :=
by
  rw [h]
  norm_num

end NUMINAMATH_GPT_square_area_l2263_226377


namespace NUMINAMATH_GPT_james_bought_dirt_bikes_l2263_226300

variable (D : ℕ)

-- Definitions derived from conditions
def cost_dirt_bike := 150
def cost_off_road_vehicle := 300
def registration_fee := 25
def num_off_road_vehicles := 4
def total_paid := 1825

-- Auxiliary definitions
def total_cost_dirt_bike := cost_dirt_bike + registration_fee
def total_cost_off_road_vehicle := cost_off_road_vehicle + registration_fee
def total_cost_off_road_vehicles := num_off_road_vehicles * total_cost_off_road_vehicle
def total_cost_dirt_bikes := total_paid - total_cost_off_road_vehicles

-- The final statement we need to prove
theorem james_bought_dirt_bikes : D = total_cost_dirt_bikes / total_cost_dirt_bike ↔ D = 3 := by
  sorry

end NUMINAMATH_GPT_james_bought_dirt_bikes_l2263_226300


namespace NUMINAMATH_GPT_alex_original_seat_l2263_226335

-- We define a type for seats
inductive Seat where
  | s1 | s2 | s3 | s4 | s5 | s6
  deriving DecidableEq, Inhabited

open Seat

-- Define the initial conditions and movements
def initial_seats : (Fin 6 → Seat) := ![s1, s2, s3, s4, s5, s6]

def move_bella (s : Seat) : Seat :=
  match s with
  | s1 => s2
  | s2 => s3
  | s3 => s4
  | s4 => s5
  | s5 => s6
  | s6 => s1  -- invalid movement for the problem context, can handle separately

def move_coral (s : Seat) : Seat :=
  match s with
  | s1 => s6  -- two seats left from s1 wraps around to s6
  | s2 => s1
  | s3 => s2
  | s4 => s3
  | s5 => s4
  | s6 => s5

-- Dan and Eve switch seats among themselves
def switch_dan_eve (s : Seat) : Seat :=
  match s with
  | s3 => s4
  | s4 => s3
  | _ => s  -- all other positions remain the same

def move_finn (s : Seat) : Seat :=
  match s with
  | s1 => s2
  | s2 => s3
  | s3 => s4
  | s4 => s5
  | s5 => s6
  | s6 => s1  -- invalid movement for the problem context, can handle separately

-- Define the final seat for Alex
def alex_final_seat : Seat := s6  -- Alex returns to one end seat

-- Define a theorem for the proof of Alex's original seat being Seat.s1
theorem alex_original_seat :
  ∃ (original_seat : Seat), original_seat = s1 :=
  sorry

end NUMINAMATH_GPT_alex_original_seat_l2263_226335


namespace NUMINAMATH_GPT_probability_without_order_knowledge_correct_probability_with_order_knowledge_correct_l2263_226336

def TianJi_top {α : Type} [LinearOrder α] (a1 a2 : α) (b1 : α) : Prop :=
  a2 < b1 ∧ b1 < a1

def TianJi_middle {α : Type} [LinearOrder α] (a3 a2 : α) (b2 : α) : Prop :=
  a3 < b2 ∧ b2 < a2

def TianJi_bottom {α : Type} [LinearOrder α] (a3 : α) (b3 : α) : Prop :=
  b3 < a3

def without_order_knowledge_probability (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : ℚ :=
  -- Formula for the probability of Tian Ji winning without knowing the order
  1 / 6

theorem probability_without_order_knowledge_correct (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : 
  without_order_knowledge_probability a1 a2 a3 b1 b2 b3 h_top h_middle h_bottom = 1 / 6 :=
sorry

def with_order_knowledge_probability (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : ℚ :=
  -- Formula for the probability of Tian Ji winning with specific group knowledge
  1 / 2

theorem probability_with_order_knowledge_correct (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : 
  with_order_knowledge_probability a1 a2 a3 b1 b2 b3 h_top h_middle h_bottom = 1 / 2 :=
sorry

end NUMINAMATH_GPT_probability_without_order_knowledge_correct_probability_with_order_knowledge_correct_l2263_226336


namespace NUMINAMATH_GPT_solve_for_x_l2263_226311

noncomputable def x : ℚ := 45^2 / (7 - (3 / 4))

theorem solve_for_x : x = 324 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2263_226311


namespace NUMINAMATH_GPT_total_fish_weight_is_25_l2263_226361

-- Define the conditions and the problem
def num_trout : ℕ := 4
def weight_trout : ℝ := 2
def num_catfish : ℕ := 3
def weight_catfish : ℝ := 1.5
def num_bluegills : ℕ := 5
def weight_bluegill : ℝ := 2.5

-- Calculate the total weight of each type of fish
def total_weight_trout : ℝ := num_trout * weight_trout
def total_weight_catfish : ℝ := num_catfish * weight_catfish
def total_weight_bluegills : ℝ := num_bluegills * weight_bluegill

-- Calculate the total weight of all fish
def total_weight_fish : ℝ := total_weight_trout + total_weight_catfish + total_weight_bluegills

-- Statement to be proved
theorem total_fish_weight_is_25 : total_weight_fish = 25 := by
  sorry

end NUMINAMATH_GPT_total_fish_weight_is_25_l2263_226361


namespace NUMINAMATH_GPT_transform_eq_l2263_226328

theorem transform_eq (x y : ℝ) (h : 5 * x - 6 * y = 4) : 
  y = (5 / 6) * x - (2 / 3) :=
  sorry

end NUMINAMATH_GPT_transform_eq_l2263_226328


namespace NUMINAMATH_GPT_cubic_sum_identity_l2263_226391

   theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 3) (h3 : abc = -1) :
     a^3 + b^3 + c^3 = 12 :=
   by
     sorry
   
end NUMINAMATH_GPT_cubic_sum_identity_l2263_226391


namespace NUMINAMATH_GPT_number_of_ways_to_divide_friends_l2263_226343

theorem number_of_ways_to_divide_friends :
  (4 ^ 8 = 65536) := by
  -- result obtained through the multiplication principle
  sorry

end NUMINAMATH_GPT_number_of_ways_to_divide_friends_l2263_226343


namespace NUMINAMATH_GPT_comparison_of_a_and_c_l2263_226350

variable {α : Type _} [LinearOrderedField α]

theorem comparison_of_a_and_c (a b c : α) (h1 : a > b) (h2 : (a - b) * (b - c) * (c - a) > 0) : a > c :=
sorry

end NUMINAMATH_GPT_comparison_of_a_and_c_l2263_226350


namespace NUMINAMATH_GPT_number_of_students_l2263_226396

-- Definitions based on problem conditions
def age_condition (a n : ℕ) : Prop :=
  7 * (a - 1) + 2 * (a + 2) + (n - 9) * a = 330

-- Main theorem to prove the correct number of students
theorem number_of_students (a n : ℕ) (h : age_condition a n) : n = 37 :=
  sorry

end NUMINAMATH_GPT_number_of_students_l2263_226396


namespace NUMINAMATH_GPT_greatest_possible_integer_l2263_226389

theorem greatest_possible_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℕ, n = 9 * k - 1) (h3 : ∃ l : ℕ, n = 10 * l - 4) : n = 86 := 
sorry

end NUMINAMATH_GPT_greatest_possible_integer_l2263_226389


namespace NUMINAMATH_GPT_time_for_5x5_grid_l2263_226306

-- Definitions based on the conditions
def total_length_3x7 : ℕ := 4 * 7 + 8 * 3
def time_for_3x7 : ℕ := 26
def time_per_unit_length : ℚ := time_for_3x7 / total_length_3x7
def total_length_5x5 : ℕ := 6 * 5 + 6 * 5
def expected_time_for_5x5 : ℚ := total_length_5x5 * time_per_unit_length

-- Theorem statement to prove the total time for 5x5 grid
theorem time_for_5x5_grid : expected_time_for_5x5 = 30 := by
  sorry

end NUMINAMATH_GPT_time_for_5x5_grid_l2263_226306


namespace NUMINAMATH_GPT_range_of_abs_2z_minus_1_l2263_226352

open Complex

theorem range_of_abs_2z_minus_1
  (z : ℂ)
  (h : abs (z + 2 - I) = 1) :
  abs (2 * z - 1) ∈ Set.Icc (Real.sqrt 29 - 2) (Real.sqrt 29 + 2) :=
sorry

end NUMINAMATH_GPT_range_of_abs_2z_minus_1_l2263_226352


namespace NUMINAMATH_GPT_simplify_999_times_neg13_simplify_complex_expr_correct_division_calculation_l2263_226307

-- Part 1: Proving the simplified form of arithmetic operations
theorem simplify_999_times_neg13 : 999 * (-13) = -12987 := by
  sorry

theorem simplify_complex_expr :
  999 * (118 + 4 / 5) + 333 * (-3 / 5) - 999 * (18 + 3 / 5) = 99900 := by
  sorry

-- Part 2: Proving the correct calculation of division
theorem correct_division_calculation : 6 / (-1 / 2 + 1 / 3) = -36 := by
  sorry

end NUMINAMATH_GPT_simplify_999_times_neg13_simplify_complex_expr_correct_division_calculation_l2263_226307


namespace NUMINAMATH_GPT_intersection_points_of_circle_and_line_l2263_226323

theorem intersection_points_of_circle_and_line :
  (∃ y, (4, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 25}) → 
  ∃ s : Finset (ℝ × ℝ), s.card = 2 ∧ ∀ p ∈ s, (p.1 = 4 ∧ (p.1 ^ 2 + p.2 ^ 2 = 25)) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_of_circle_and_line_l2263_226323


namespace NUMINAMATH_GPT_total_distance_journey_l2263_226359

def miles_driven : ℕ := 384
def miles_remaining : ℕ := 816

theorem total_distance_journey :
  miles_driven + miles_remaining = 1200 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_journey_l2263_226359
