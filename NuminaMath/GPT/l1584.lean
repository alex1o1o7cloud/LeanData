import Mathlib

namespace NUMINAMATH_GPT_vertical_asymptotes_polynomial_l1584_158469

theorem vertical_asymptotes_polynomial (a b : ℝ) (h₁ : -3 * 2 = b) (h₂ : -3 + 2 = a) : a + b = -5 := by
  sorry

end NUMINAMATH_GPT_vertical_asymptotes_polynomial_l1584_158469


namespace NUMINAMATH_GPT_fifth_largest_divisor_of_1209600000_is_75600000_l1584_158442

theorem fifth_largest_divisor_of_1209600000_is_75600000 :
  let n : ℤ := 1209600000
  let fifth_largest_divisor : ℤ := 75600000
  n = 2^10 * 5^5 * 3 * 503 →
  fifth_largest_divisor = n / 2^5 :=
by
  sorry

end NUMINAMATH_GPT_fifth_largest_divisor_of_1209600000_is_75600000_l1584_158442


namespace NUMINAMATH_GPT_range_f_l1584_158454

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x + 1)

theorem range_f : (Set.range f) = Set.univ := by
  sorry

end NUMINAMATH_GPT_range_f_l1584_158454


namespace NUMINAMATH_GPT_shopkeeper_net_loss_percent_l1584_158499

theorem shopkeeper_net_loss_percent (cp : ℝ)
  (sp1 sp2 sp3 sp4 : ℝ)
  (h_cp : cp = 1000)
  (h_sp1 : sp1 = cp * 1.1)
  (h_sp2 : sp2 = cp * 0.9)
  (h_sp3 : sp3 = cp * 1.2)
  (h_sp4 : sp4 = cp * 0.75) :
  ((cp + cp + cp + cp) - (sp1 + sp2 + sp3 + sp4)) / (cp + cp + cp + cp) * 100 = 1.25 :=
by sorry

end NUMINAMATH_GPT_shopkeeper_net_loss_percent_l1584_158499


namespace NUMINAMATH_GPT_line_tangent_to_ellipse_l1584_158450

theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! x : ℝ, x^2 + 4 * (m * x + 1)^2 = 1) → m^2 = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_line_tangent_to_ellipse_l1584_158450


namespace NUMINAMATH_GPT_solve_absolute_value_equation_l1584_158463

theorem solve_absolute_value_equation (x : ℝ) :
  |2 * x - 3| = x + 1 → (x = 4 ∨ x = 2 / 3) := by
  sorry

end NUMINAMATH_GPT_solve_absolute_value_equation_l1584_158463


namespace NUMINAMATH_GPT_minimum_value_f_range_of_m_l1584_158484

noncomputable def f (x m : ℝ) := x - m * Real.log x - (m - 1) / x
noncomputable def g (x : ℝ) := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

theorem minimum_value_f (m : ℝ) (x : ℝ) (hx : 1 ≤ x ∧ x ≤ Real.exp 1) : 
  if m ≤ 2 then f x m = 2 - m 
  else if m ≥ Real.exp 1 + 1 then f x m = Real.exp 1 - m - (m - 1) / Real.exp 1 
  else f x m = m - 2 - m * Real.log (m - 1) :=
sorry

theorem range_of_m (m : ℝ) :
  (m ≤ 2 ∧ ∀ x2 ∈ [-2, 0], ∃ x1 ∈ [Real.exp 1, Real.exp 2], f x1 m ≤ g x2) ↔
  (m ∈ [ (Real.exp 2 - Real.exp 1 + 1) / (Real.exp 1 + 1), 2 ]) :=
sorry

end NUMINAMATH_GPT_minimum_value_f_range_of_m_l1584_158484


namespace NUMINAMATH_GPT_first_pump_half_time_l1584_158485

theorem first_pump_half_time (t : ℝ) : 
  (∃ (t : ℝ), (1/(2*t) + 1/1.1111111111111112) * (1/2) = 1/2) -> 
  t = 5 :=
by
  sorry

end NUMINAMATH_GPT_first_pump_half_time_l1584_158485


namespace NUMINAMATH_GPT_red_toys_removed_l1584_158433

theorem red_toys_removed (R W : ℕ) (h1 : R + W = 134) (h2 : 2 * W = 88) (h3 : R - 2 * W / 2 = 88) : R - 88 = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_red_toys_removed_l1584_158433


namespace NUMINAMATH_GPT_find_g_six_l1584_158471

noncomputable def g : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : g (x + y) = g x + g y
axiom g_five : g 5 = 6

theorem find_g_six : g 6 = 36/5 := 
by 
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_find_g_six_l1584_158471


namespace NUMINAMATH_GPT_disproving_iff_l1584_158477

theorem disproving_iff (a b : ℤ) (h1 : a = -3) (h2 : b = 2) : (a^2 > b^2) ∧ ¬(a > b) :=
by
  sorry

end NUMINAMATH_GPT_disproving_iff_l1584_158477


namespace NUMINAMATH_GPT_no_green_ball_in_bag_l1584_158467

theorem no_green_ball_in_bag (bag : Set String) (h : bag = {"red", "yellow", "white"}): ¬ ("green" ∈ bag) :=
by
  sorry

end NUMINAMATH_GPT_no_green_ball_in_bag_l1584_158467


namespace NUMINAMATH_GPT_bags_of_hammers_to_load_l1584_158446

noncomputable def total_crate_capacity := 15 * 20
noncomputable def weight_of_nails := 4 * 5
noncomputable def weight_of_planks := 10 * 30
noncomputable def weight_to_be_left_out := 80
noncomputable def effective_capacity := total_crate_capacity - weight_to_be_left_out
noncomputable def weight_of_loaded_planks := 220

theorem bags_of_hammers_to_load : (effective_capacity - weight_of_nails - weight_of_loaded_planks = 0) :=
by
  sorry

end NUMINAMATH_GPT_bags_of_hammers_to_load_l1584_158446


namespace NUMINAMATH_GPT_maximum_n_for_sequence_l1584_158423

theorem maximum_n_for_sequence :
  ∃ (n : ℕ), 
  (∀ a S : ℕ → ℝ, 
    a 1 = 1 → 
    (∀ n : ℕ, n > 0 → 2 * a (n + 1) + S n = 2) → 
    (1001 / 1000 < S (2 * n) / S n ∧ S (2 * n) / S n < 11 / 10)) →
  n = 9 :=
sorry

end NUMINAMATH_GPT_maximum_n_for_sequence_l1584_158423


namespace NUMINAMATH_GPT_find_slope_of_line_l1584_158425

-- Define the parabola, point M, and the conditions leading to the slope k.
theorem find_slope_of_line (k : ℝ) :
  let C := {p : ℝ × ℝ | p.2^2 = 4 * p.1}
  let focus : ℝ × ℝ := (1, 0)
  let M : ℝ × ℝ := (-1, 1)
  let line (k : ℝ) (x : ℝ) := k * (x - 1)
  ∃ A B : (ℝ × ℝ), 
    A ∈ C ∧ B ∈ C ∧
    A ≠ B ∧
    A.1 + 1 = B.1 + 1 ∧ 
    A.2 - 1 = B.2 - 1 ∧
    ((A.1 + 1) * (B.1 + 1) + (A.2 - 1) * (B.2 - 1) = 0) -> k = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_slope_of_line_l1584_158425


namespace NUMINAMATH_GPT_susie_remaining_money_l1584_158457

noncomputable def calculate_remaining_money : Float :=
  let weekday_hours := 4.0
  let weekday_rate := 12.0
  let weekdays := 5.0
  let weekend_hours := 2.5
  let weekend_rate := 15.0
  let weekends := 2.0
  let total_weekday_earnings := weekday_hours * weekday_rate * weekdays
  let total_weekend_earnings := weekend_hours * weekend_rate * weekends
  let total_earnings := total_weekday_earnings + total_weekend_earnings
  let spent_makeup := 3 / 8 * total_earnings
  let remaining_after_makeup := total_earnings - spent_makeup
  let spent_skincare := 2 / 5 * remaining_after_makeup
  let remaining_after_skincare := remaining_after_makeup - spent_skincare
  let spent_cellphone := 1 / 6 * remaining_after_skincare
  let final_remaining := remaining_after_skincare - spent_cellphone
  final_remaining

theorem susie_remaining_money : calculate_remaining_money = 98.4375 := by
  sorry

end NUMINAMATH_GPT_susie_remaining_money_l1584_158457


namespace NUMINAMATH_GPT_gray_region_area_l1584_158419

-- Definitions based on given conditions
def radius_inner (r : ℝ) := r
def radius_outer (r : ℝ) := r + 3

-- Statement to prove: the area of the gray region
theorem gray_region_area (r : ℝ) : 
  (π * (radius_outer r)^2 - π * (radius_inner r)^2) = 6 * π * r + 9 * π := by
  sorry

end NUMINAMATH_GPT_gray_region_area_l1584_158419


namespace NUMINAMATH_GPT_min_y_ellipse_l1584_158468

-- Defining the ellipse equation
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 49) + ((y - 3)^2 / 25) = 1

-- Problem statement: Prove that the smallest y-coordinate is -2
theorem min_y_ellipse : 
  ∀ x y, ellipse x y → y ≥ -2 :=
sorry

end NUMINAMATH_GPT_min_y_ellipse_l1584_158468


namespace NUMINAMATH_GPT_kelly_initial_apples_l1584_158452

theorem kelly_initial_apples : ∀ (T P I : ℕ), T = 105 → P = 49 → I + P = T → I = 56 :=
by
  intros T P I ht hp h
  rw [ht, hp] at h
  linarith

end NUMINAMATH_GPT_kelly_initial_apples_l1584_158452


namespace NUMINAMATH_GPT_total_feet_l1584_158403

theorem total_feet (H C : ℕ) (h1 : H + C = 48) (h2 : H = 28) : 2 * H + 4 * C = 136 := 
by
  sorry

end NUMINAMATH_GPT_total_feet_l1584_158403


namespace NUMINAMATH_GPT_girls_together_count_l1584_158459

-- Define the problem conditions
def boys : ℕ := 4
def girls : ℕ := 2
def total_entities : ℕ := boys + (girls - 1) -- One entity for the two girls together

-- Calculate the factorial
noncomputable def factorial (n: ℕ) : ℕ :=
  if n = 0 then 1 else (List.range (n+1)).foldl (λx y => x * y) 1

-- Define the total number of ways girls can be together
noncomputable def ways_girls_together : ℕ :=
  factorial total_entities * factorial girls

-- State the theorem that needs to be proved
theorem girls_together_count : ways_girls_together = 240 := by
  sorry

end NUMINAMATH_GPT_girls_together_count_l1584_158459


namespace NUMINAMATH_GPT_count_positive_numbers_is_three_l1584_158418

def negative_three := -3
def zero := 0
def negative_three_squared := (-3) ^ 2
def absolute_negative_nine := |(-9)|
def negative_one_raised_to_four := -1 ^ 4

def number_list : List Int := [ -negative_three, zero, negative_three_squared, absolute_negative_nine, negative_one_raised_to_four ]

def count_positive_numbers (lst: List Int) : Nat :=
  lst.foldl (λ acc x => if x > 0 then acc + 1 else acc) 0

theorem count_positive_numbers_is_three : count_positive_numbers number_list = 3 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_count_positive_numbers_is_three_l1584_158418


namespace NUMINAMATH_GPT_prime_divisors_of_1320_l1584_158497

theorem prime_divisors_of_1320 : 
  ∃ (S : Finset ℕ), (S = {2, 3, 5, 11}) ∧ S.card = 4 := 
by
  sorry

end NUMINAMATH_GPT_prime_divisors_of_1320_l1584_158497


namespace NUMINAMATH_GPT_buns_cost_eq_1_50_l1584_158447

noncomputable def meat_cost : ℝ := 2 * 3.50
noncomputable def tomato_cost : ℝ := 1.5 * 2.00
noncomputable def pickles_cost : ℝ := 2.50 - 1.00
noncomputable def lettuce_cost : ℝ := 1.00
noncomputable def total_other_items_cost : ℝ := meat_cost + tomato_cost + pickles_cost + lettuce_cost
noncomputable def total_amount_spent : ℝ := 20.00 - 6.00
noncomputable def buns_cost : ℝ := total_amount_spent - total_other_items_cost

theorem buns_cost_eq_1_50 : buns_cost = 1.50 := by
  sorry

end NUMINAMATH_GPT_buns_cost_eq_1_50_l1584_158447


namespace NUMINAMATH_GPT_find_a_with_constraints_l1584_158461

theorem find_a_with_constraints (x y a : ℝ) 
  (h1 : 2 * x - y + 2 ≥ 0) 
  (h2 : x - 3 * y + 1 ≤ 0)
  (h3 : x + y - 2 ≤ 0)
  (h4 : a > 0)
  (h5 : ∃ (x1 x2 x3 y1 y2 y3 : ℝ), 
    ((x1, y1) = (1, 1) ∨ (x1, y1) = (5 / 3, 1 / 3) ∨ (x1, y1) = (2, 0)) ∧ 
    ((x2, y2) = (1, 1) ∨ (x2, y2) = (5 / 3, 1 / 3) ∨ (x2, y2) = (2, 0)) ∧ 
    ((x3, y3) = (1, 1) ∨ (x3, y3) = (5 / 3, 1 / 3) ∨ (x3, y3) = (2, 0)) ∧ 
    (ax1 - y1 = ax2 - y2) ∧ (ax2 - y2 = ax3 - y3)) :
  a = 1 / 3 :=
sorry

end NUMINAMATH_GPT_find_a_with_constraints_l1584_158461


namespace NUMINAMATH_GPT_derivative_at_one_l1584_158458

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_one : deriv f 1 = 2 * Real.exp 1 := by
sorry

end NUMINAMATH_GPT_derivative_at_one_l1584_158458


namespace NUMINAMATH_GPT_ratio_of_rises_l1584_158420

noncomputable def radius_narrower_cone : ℝ := 4
noncomputable def radius_wider_cone : ℝ := 8
noncomputable def sphere_radius : ℝ := 2

noncomputable def height_ratio (h1 h2 : ℝ) : Prop := h1 = 4 * h2

noncomputable def volume_displacement := (4 / 3) * Real.pi * (sphere_radius^3)

noncomputable def new_height_narrower (h1 : ℝ) : ℝ := h1 + (volume_displacement / ((Real.pi * (radius_narrower_cone^2))))

noncomputable def new_height_wider (h2 : ℝ) : ℝ := h2 + (volume_displacement / ((Real.pi * (radius_wider_cone^2))))

theorem ratio_of_rises (h1 h2 : ℝ) (hr : height_ratio h1 h2) :
  (new_height_narrower h1 - h1) / (new_height_wider h2 - h2) = 4 :=
sorry

end NUMINAMATH_GPT_ratio_of_rises_l1584_158420


namespace NUMINAMATH_GPT_point_B_possible_values_l1584_158402

-- Define point A
def A : ℝ := 1

-- Define the condition that B is 3 units away from A
def units_away (a b : ℝ) : ℝ := abs (b - a)

theorem point_B_possible_values :
  ∃ B : ℝ, units_away A B = 3 ∧ (B = 4 ∨ B = -2) := by
  sorry

end NUMINAMATH_GPT_point_B_possible_values_l1584_158402


namespace NUMINAMATH_GPT_total_dolphins_l1584_158415

theorem total_dolphins (initial_dolphins : ℕ) (triple_of_initial : ℕ) (final_dolphins : ℕ) 
    (h1 : initial_dolphins = 65) (h2 : triple_of_initial = 3 * initial_dolphins) (h3 : final_dolphins = initial_dolphins + triple_of_initial) : 
    final_dolphins = 260 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_dolphins_l1584_158415


namespace NUMINAMATH_GPT_g_five_l1584_158435

def g (x : ℝ) : ℝ := 4 * x + 2

theorem g_five : g 5 = 22 := by
  sorry

end NUMINAMATH_GPT_g_five_l1584_158435


namespace NUMINAMATH_GPT_part1_monotonicity_part2_inequality_l1584_158430

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem part1_monotonicity (a : ℝ) :
  (∀ x : ℝ, a ≤ 0 → f a x < f a (x + 1)) ∧
  (a > 0 → ∀ x : ℝ, (x < Real.log (1 / a) → f a x > f a (x + 1)) ∧
  (x > Real.log (1 / a) → f a x < f a (x + 1))) := sorry

theorem part2_inequality (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, f a x > 2 * Real.log a + (3 / 2) := sorry

end NUMINAMATH_GPT_part1_monotonicity_part2_inequality_l1584_158430


namespace NUMINAMATH_GPT_find_solutions_l1584_158428

theorem find_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + 4^y = 5^z ↔ (x = 3 ∧ y = 2 ∧ z = 2) ∨ (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 11 ∧ y = 1 ∧ z = 3) :=
by sorry

end NUMINAMATH_GPT_find_solutions_l1584_158428


namespace NUMINAMATH_GPT_logistics_company_freight_l1584_158448

theorem logistics_company_freight :
  ∃ (x y : ℕ), 
    50 * x + 30 * y = 9500 ∧
    70 * x + 40 * y = 13000 ∧
    x = 100 ∧
    y = 140 :=
by
  -- The proof is skipped here
  sorry

end NUMINAMATH_GPT_logistics_company_freight_l1584_158448


namespace NUMINAMATH_GPT_min_value_of_a_l1584_158411

/-- Given the inequality |x - 1| + |x + a| ≤ 8, prove that the minimum value of a is -9 -/

theorem min_value_of_a (a : ℝ) (h : ∀ x : ℝ, |x - 1| + |x + a| ≤ 8) : a = -9 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l1584_158411


namespace NUMINAMATH_GPT_B_and_C_mutually_exclusive_l1584_158439

-- Defining events in terms of products being defective or not
def all_not_defective (products : List Bool) : Prop := 
  ∀ x ∈ products, ¬x

def all_defective (products : List Bool) : Prop := 
  ∀ x ∈ products, x

def not_all_defective (products : List Bool) : Prop := 
  ∃ x ∈ products, ¬x

-- Given a batch of three products, define events A, B, and C
def A (products : List Bool) : Prop := all_not_defective products
def B (products : List Bool) : Prop := all_defective products
def C (products : List Bool) : Prop := not_all_defective products

-- The theorem to prove that B and C are mutually exclusive
theorem B_and_C_mutually_exclusive (products : List Bool) (h : products.length = 3) : 
  ¬ (B products ∧ C products) :=
by
  sorry

end NUMINAMATH_GPT_B_and_C_mutually_exclusive_l1584_158439


namespace NUMINAMATH_GPT_num_mappings_from_A_to_A_is_4_l1584_158478

-- Define the number of elements in set A
def set_A_card := 2

-- Define the proof problem
theorem num_mappings_from_A_to_A_is_4 (h : set_A_card = 2) : (set_A_card ^ set_A_card) = 4 :=
by
  sorry

end NUMINAMATH_GPT_num_mappings_from_A_to_A_is_4_l1584_158478


namespace NUMINAMATH_GPT_find_ff_of_five_half_l1584_158422

noncomputable def f (x : ℝ) : ℝ :=
if x <= 1 then 2^x - 2 else Real.log x / Real.log 2

theorem find_ff_of_five_half : f (f (5/2)) = -1/2 := by
  sorry

end NUMINAMATH_GPT_find_ff_of_five_half_l1584_158422


namespace NUMINAMATH_GPT_angela_january_additional_sleep_l1584_158432

-- Definitions corresponding to conditions in part a)
def december_sleep_hours : ℝ := 6.5
def january_sleep_hours : ℝ := 8.5
def days_in_january : ℕ := 31

-- The proof statement, proving the January's additional sleep hours
theorem angela_january_additional_sleep :
  (january_sleep_hours - december_sleep_hours) * days_in_january = 62 :=
by
  -- Since the focus is only on the statement, we skip the actual proof.
  sorry

end NUMINAMATH_GPT_angela_january_additional_sleep_l1584_158432


namespace NUMINAMATH_GPT_largest_non_sum_of_multiple_of_30_and_composite_l1584_158438

theorem largest_non_sum_of_multiple_of_30_and_composite :
  ∃ (n : ℕ), n = 211 ∧ ∀ a b : ℕ, (a > 0) → (b > 0) → (b < 30) → 
  n ≠ 30 * a + b ∧ ¬ ∃ k : ℕ, k > 1 ∧ k < b ∧ b % k = 0 :=
sorry

end NUMINAMATH_GPT_largest_non_sum_of_multiple_of_30_and_composite_l1584_158438


namespace NUMINAMATH_GPT_fraction_to_decimal_l1584_158479

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1584_158479


namespace NUMINAMATH_GPT_distinct_exponentiation_values_l1584_158401

theorem distinct_exponentiation_values : 
  let a := 3^(3^(3^3))
  let b := 3^((3^3)^3)
  let c := ((3^3)^3)^3
  let d := 3^((3^3)^(3^2))
  (a ≠ b) → (a ≠ c) → (a ≠ d) → (b ≠ c) → (b ≠ d) → (c ≠ d) → 
  ∃ n, n = 3 := 
sorry

end NUMINAMATH_GPT_distinct_exponentiation_values_l1584_158401


namespace NUMINAMATH_GPT_seven_n_form_l1584_158436

theorem seven_n_form (n : ℤ) (a b : ℤ) (h : 7 * n = a^2 + 3 * b^2) : 
  ∃ c d : ℤ, n = c^2 + 3 * d^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_seven_n_form_l1584_158436


namespace NUMINAMATH_GPT_Dan_running_speed_is_10_l1584_158444

noncomputable def running_speed
  (d : ℕ)
  (S : ℕ)
  (avg : ℚ) : ℚ :=
  let total_distance := 2 * d
  let total_time := d / (avg * 60) 
  let swim_time := d / S
  let run_time := total_time - swim_time
  total_distance / run_time

theorem Dan_running_speed_is_10
  (d S : ℕ)
  (avg : ℚ)
  (h1 : d = 4)
  (h2 : S = 6)
  (h3 : avg = 0.125) :
  running_speed d S (avg * 60) = 10 := by 
  sorry

end NUMINAMATH_GPT_Dan_running_speed_is_10_l1584_158444


namespace NUMINAMATH_GPT_running_speed_l1584_158487

theorem running_speed
  (walking_speed : Float)
  (walking_time : Float)
  (running_time : Float)
  (distance : Float) :
  walking_speed = 8 → walking_time = 3 → running_time = 1.5 → distance = walking_speed * walking_time → 
  (distance / running_time) = 16 :=
by
  intros h_walking_speed h_walking_time h_running_time h_distance
  sorry

end NUMINAMATH_GPT_running_speed_l1584_158487


namespace NUMINAMATH_GPT_students_in_class_l1584_158431

theorem students_in_class (n S : ℕ) 
    (h1 : S = 15 * n)
    (h2 : (S + 56) / (n + 1) = 16) : n = 40 :=
by
  sorry

end NUMINAMATH_GPT_students_in_class_l1584_158431


namespace NUMINAMATH_GPT_hypotenuse_is_correct_l1584_158426

noncomputable def hypotenuse_of_right_triangle (a b : ℕ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem hypotenuse_is_correct :
  hypotenuse_of_right_triangle 140 210 = 70 * Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_is_correct_l1584_158426


namespace NUMINAMATH_GPT_max_value_9_l1584_158406

noncomputable def max_ab_ac_bc (a b c : ℝ) : ℝ :=
  max (a * b) (max (a * c) (b * c))

theorem max_value_9 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 12) (h_prod : a * b + b * c + c * a = 27) :
  max_ab_ac_bc a b c = 9 :=
sorry

end NUMINAMATH_GPT_max_value_9_l1584_158406


namespace NUMINAMATH_GPT_radar_arrangements_l1584_158460

-- Define the number of letters in the word RADAR
def total_letters : Nat := 5

-- Define the number of times each letter is repeated
def repetition_R : Nat := 2
def repetition_A : Nat := 2

-- The expected number of unique arrangements
def expected_unique_arrangements : Nat := 30

theorem radar_arrangements :
  (Nat.factorial total_letters) / (Nat.factorial repetition_R * Nat.factorial repetition_A) = expected_unique_arrangements := by
  sorry

end NUMINAMATH_GPT_radar_arrangements_l1584_158460


namespace NUMINAMATH_GPT_number_of_small_cubes_l1584_158440

-- Definition of the conditions from the problem
def painted_cube (n : ℕ) :=
  6 * (n - 2) * (n - 2) = 54

-- The theorem we need to prove
theorem number_of_small_cubes (n : ℕ) (h : painted_cube n) : n^3 = 125 :=
by
  have h1 : 6 * (n - 2) * (n - 2) = 54 := h
  sorry

end NUMINAMATH_GPT_number_of_small_cubes_l1584_158440


namespace NUMINAMATH_GPT_ten_digit_number_l1584_158492

open Nat

theorem ten_digit_number (a : Fin 10 → ℕ) (h1 : a 4 = 2)
  (h2 : a 8 = 3)
  (h3 : ∀ i, i < 8 → a i * a (i + 1) * a (i + 2) = 24) :
  a = ![4, 2, 3, 4, 2, 3, 4, 2, 3, 4] :=
sorry

end NUMINAMATH_GPT_ten_digit_number_l1584_158492


namespace NUMINAMATH_GPT_weight_of_each_package_l1584_158416

theorem weight_of_each_package (W : ℝ) 
  (h1: 10 * W + 7 * W + 8 * W = 100) : W = 4 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_each_package_l1584_158416


namespace NUMINAMATH_GPT_points_on_line_sufficient_but_not_necessary_l1584_158413

open Nat

-- Define the sequence a_n
def sequence_a (n : ℕ) : ℕ := n + 1

-- Define a general arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) := ∀ n m : ℕ, n < m → a (m) - a (n) = (m - n) * (a 1 - a 0)

-- Define the condition that points (n, a_n), where n is a natural number, lie on the line y = x + 1
def points_on_line (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a (n) = n + 1

-- Prove that points_on_line is sufficient but not necessary for is_arithmetic_sequence
theorem points_on_line_sufficient_but_not_necessary :
  (∀ a : ℕ → ℕ, points_on_line a → is_arithmetic_sequence a)
  ∧ ∃ a : ℕ → ℕ, is_arithmetic_sequence a ∧ ¬ points_on_line a := 
by 
  sorry

end NUMINAMATH_GPT_points_on_line_sufficient_but_not_necessary_l1584_158413


namespace NUMINAMATH_GPT_grade_assignments_count_l1584_158427

theorem grade_assignments_count (n : ℕ) (g : ℕ) (h : n = 15) (k : g = 4) : g^n = 1073741824 :=
by
  sorry

end NUMINAMATH_GPT_grade_assignments_count_l1584_158427


namespace NUMINAMATH_GPT_congruent_semicircles_span_diameter_l1584_158475

theorem congruent_semicircles_span_diameter (N : ℕ) (r : ℝ) 
  (h1 : 2 * N * r = 2 * (N * r)) 
  (h2 : (N * (π * r^2 / 2)) / ((N^2 * (π * r^2 / 2)) - (N * (π * r^2 / 2))) = 1/4) 
  : N = 5 :=
by
  sorry

end NUMINAMATH_GPT_congruent_semicircles_span_diameter_l1584_158475


namespace NUMINAMATH_GPT_tonya_large_lemonade_sales_l1584_158408

theorem tonya_large_lemonade_sales 
  (price_small : ℝ)
  (price_medium : ℝ)
  (price_large : ℝ)
  (total_revenue : ℝ)
  (revenue_small : ℝ)
  (revenue_medium : ℝ)
  (n : ℝ)
  (h_price_small : price_small = 1)
  (h_price_medium : price_medium = 2)
  (h_price_large : price_large = 3)
  (h_total_revenue : total_revenue = 50)
  (h_revenue_small : revenue_small = 11)
  (h_revenue_medium : revenue_medium = 24)
  (h_revenue_large : n = (total_revenue - revenue_small - revenue_medium) / price_large) :
  n = 5 :=
sorry

end NUMINAMATH_GPT_tonya_large_lemonade_sales_l1584_158408


namespace NUMINAMATH_GPT_initial_balls_in_bag_l1584_158445

theorem initial_balls_in_bag (n : ℕ) 
  (h_add_white : ∀ x : ℕ, x = n + 1)
  (h_probability : (5 / 8) = 0.625):
  n = 7 :=
sorry

end NUMINAMATH_GPT_initial_balls_in_bag_l1584_158445


namespace NUMINAMATH_GPT_probability_A_B_l1584_158490

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_probability_A_B_l1584_158490


namespace NUMINAMATH_GPT_fraction_operation_correct_l1584_158443

theorem fraction_operation_correct 
  (a b : ℝ) : 
  (0.2 * (3 * a + 10 * b) = 6 * a + 20 * b) → 
  (0.1 * (2 * a + 5 * b) = 2 * a + 5 * b) →
  (∀ c : ℝ, c ≠ 0 → (a / b = (a * c) / (b * c))) ∨
  (∀ x y : ℝ, ((x - y) / (x + y) ≠ (y - x) / (x - y))) ∨
  (∀ x : ℝ, (x + x * x * x + x * y ≠ 1 / x * x)) →
  ((0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b)) :=
sorry

end NUMINAMATH_GPT_fraction_operation_correct_l1584_158443


namespace NUMINAMATH_GPT_tickets_sold_in_total_l1584_158470

def total_tickets
    (adult_price student_price : ℕ)
    (total_revenue adult_tickets student_tickets : ℕ) : ℕ :=
  adult_tickets + student_tickets

theorem tickets_sold_in_total 
    (adult_price student_price : ℕ)
    (total_revenue adult_tickets student_tickets : ℕ)
    (h1 : adult_price = 6)
    (h2 : student_price = 3)
    (h3 : total_revenue = 3846)
    (h4 : adult_tickets = 410)
    (h5 : student_tickets = 436) :
  total_tickets adult_price student_price total_revenue adult_tickets student_tickets = 846 :=
by
  sorry

end NUMINAMATH_GPT_tickets_sold_in_total_l1584_158470


namespace NUMINAMATH_GPT_problem1_sol_l1584_158434

noncomputable def problem1 :=
  let total_people := 200
  let avg_feelings_total := 70
  let female_total := 100
  let a := 30 -- derived from 2a + (70 - a) = 100
  let chi_square := 200 * (70 * 40 - 30 * 60) ^ 2 / (130 * 70 * 100 * 100)
  let k_95 := 3.841 -- critical value for 95% confidence
  let p_xi_2 := (1 / 3)
  let p_xi_3 := (1 / 2)
  let p_xi_4 := (1 / 6)
  let exi := (2 * (1 / 3)) + (3 * (1 / 2)) + (4 * (1 / 6))
  chi_square < k_95 ∧ exi = 17 / 6

theorem problem1_sol : problem1 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_problem1_sol_l1584_158434


namespace NUMINAMATH_GPT_total_age_proof_l1584_158464

variable (K : ℕ) -- Kaydence's age
variable (T : ℕ) -- Total age of people in the gathering

def Kaydence_father_age : ℕ := 60
def Kaydence_mother_age : ℕ := Kaydence_father_age - 2
def Kaydence_brother_age : ℕ := Kaydence_father_age / 2
def Kaydence_sister_age : ℕ := 40
def elder_cousin_age : ℕ := Kaydence_brother_age + 2 * Kaydence_sister_age
def younger_cousin_age : ℕ := elder_cousin_age / 2 + 3
def grandmother_age : ℕ := 3 * Kaydence_mother_age - 5

theorem total_age_proof (K : ℕ) : T = 525 + K :=
by 
  sorry

end NUMINAMATH_GPT_total_age_proof_l1584_158464


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1584_158409

noncomputable def a : ℕ → ℝ := sorry

theorem geometric_sequence_problem :
  a 4 = 4 →
  a 8 = 8 →
  a 12 = 16 :=
by
  intros h4 h8
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1584_158409


namespace NUMINAMATH_GPT_sin_thirty_degree_l1584_158424

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_thirty_degree_l1584_158424


namespace NUMINAMATH_GPT_Binkie_gemstones_l1584_158465

-- Define the number of gemstones each cat has
variables (F S B : ℕ)

-- Conditions based on the problem statement
axiom Spaatz_has_one : S = 1
axiom Spaatz_equation : S = F / 2 - 2
axiom Binkie_equation : B = 4 * F

-- Theorem statement
theorem Binkie_gemstones : B = 24 :=
by
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_Binkie_gemstones_l1584_158465


namespace NUMINAMATH_GPT_no_three_real_numbers_satisfy_inequalities_l1584_158486

theorem no_three_real_numbers_satisfy_inequalities (a b c : ℝ) :
  ¬ (|a| < |b - c| ∧ |b| < |c - a| ∧ |c| < |a - b| ) :=
by
  sorry

end NUMINAMATH_GPT_no_three_real_numbers_satisfy_inequalities_l1584_158486


namespace NUMINAMATH_GPT_leftover_coverage_l1584_158449

variable (bagCoverage lawnLength lawnWidth bagsPurchased : ℕ)

def area_of_lawn (length width : ℕ) : ℕ :=
  length * width

def total_coverage (bagCoverage bags : ℕ) : ℕ :=
  bags * bagCoverage

theorem leftover_coverage :
  let lawnLength := 22
  let lawnWidth := 36
  let bagCoverage := 250
  let bagsPurchased := 4
  let lawnArea := area_of_lawn lawnLength lawnWidth
  let totalSeedCoverage := total_coverage bagCoverage bagsPurchased
  totalSeedCoverage - lawnArea = 208 := by
  sorry

end NUMINAMATH_GPT_leftover_coverage_l1584_158449


namespace NUMINAMATH_GPT_area_of_square_l1584_158453

-- Definitions
def radius_ratio (r R : ℝ) : Prop := R = 7 / 3 * r
def small_circle_circumference (r : ℝ) : Prop := 2 * Real.pi * r = 8
def square_side_length (R side : ℝ) : Prop := side = 2 * R
def square_area (side area : ℝ) : Prop := area = side * side

-- Problem statement
theorem area_of_square (r R side area : ℝ) 
    (h1 : radius_ratio r R)
    (h2 : small_circle_circumference r)
    (h3 : square_side_length R side)
    (h4 : square_area side area) :
    area = 3136 / (9 * Real.pi^2) := 
  by sorry

end NUMINAMATH_GPT_area_of_square_l1584_158453


namespace NUMINAMATH_GPT_base9_addition_l1584_158410

-- Define the numbers in base 9
def num1 : ℕ := 1 * 9^2 + 7 * 9^1 + 5 * 9^0
def num2 : ℕ := 7 * 9^2 + 1 * 9^1 + 4 * 9^0
def num3 : ℕ := 6 * 9^1 + 1 * 9^0
def result : ℕ := 1 * 9^3 + 0 * 9^2 + 6 * 9^1 + 1 * 9^0

-- State the theorem
theorem base9_addition : num1 + num2 + num3 = result := by
  sorry

end NUMINAMATH_GPT_base9_addition_l1584_158410


namespace NUMINAMATH_GPT_chess_tournament_game_count_l1584_158429

theorem chess_tournament_game_count (n : ℕ) (h1 : ∃ n, ∀ i j, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ i ≠ j → ∃ games_between, games_between = n ∧ games_between * (Nat.choose 6 2) = 30) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_game_count_l1584_158429


namespace NUMINAMATH_GPT_maximize_root_product_l1584_158441

theorem maximize_root_product :
  (∃ k : ℝ, ∀ x : ℝ, 6 * x^2 - 5 * x + k = 0 ∧ (25 - 24 * k ≥ 0)) →
  ∃ k : ℝ, k = 25 / 24 :=
by
  sorry

end NUMINAMATH_GPT_maximize_root_product_l1584_158441


namespace NUMINAMATH_GPT_sum_of_possible_values_of_x_l1584_158414

namespace ProofProblem

-- Assume we are working in degrees for angles
def is_scalene_triangle (A B C : ℝ) (a b c : ℝ) :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

def triangle_angle_sum (A B C : ℝ) : Prop :=
  A + B + C = 180

noncomputable def problem_statement (x : ℝ) (A B C : ℝ) (a b c : ℝ) : Prop :=
  is_scalene_triangle A B C a b c ∧
  B = 45 ∧
  (A = x ∨ C = x) ∧
  (a = b ∨ b = c ∨ c = a) ∧
  triangle_angle_sum A B C

theorem sum_of_possible_values_of_x (x : ℝ) (A B C : ℝ) (a b c : ℝ) :
  problem_statement x A B C a b c →
  x = 45 :=
sorry

end ProofProblem

end NUMINAMATH_GPT_sum_of_possible_values_of_x_l1584_158414


namespace NUMINAMATH_GPT_solution_set_inequalities_l1584_158462

theorem solution_set_inequalities (x : ℝ) :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 2 * x) / 3 > x - 1) → (x ≤ 1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solution_set_inequalities_l1584_158462


namespace NUMINAMATH_GPT_total_dollars_l1584_158489

def mark_dollars : ℚ := 4 / 5
def carolyn_dollars : ℚ := 2 / 5
def jack_dollars : ℚ := 1 / 2

theorem total_dollars :
  mark_dollars + carolyn_dollars + jack_dollars = 1.7 := 
sorry

end NUMINAMATH_GPT_total_dollars_l1584_158489


namespace NUMINAMATH_GPT_population_decrease_rate_l1584_158493

theorem population_decrease_rate (r : ℕ) (h₀ : 6000 > 0) (h₁ : 4860 = 6000 * (1 - r / 100)^2) : r = 10 :=
by sorry

end NUMINAMATH_GPT_population_decrease_rate_l1584_158493


namespace NUMINAMATH_GPT_lark_lock_combination_count_l1584_158498

-- Definitions for the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def lark_lock_combination (a b c : ℕ) : Prop := 
  is_odd a ∧ is_even b ∧ is_multiple_of_5 c ∧ 1 ≤ a ∧ a ≤ 30 ∧ 1 ≤ b ∧ b ≤ 30 ∧ 1 ≤ c ∧ c ≤ 30

-- The core theorem
theorem lark_lock_combination_count : 
  (∃ a b c : ℕ, lark_lock_combination a b c) ↔ (15 * 15 * 6 = 1350) :=
by
  sorry

end NUMINAMATH_GPT_lark_lock_combination_count_l1584_158498


namespace NUMINAMATH_GPT_function_value_at_6000_l1584_158412

theorem function_value_at_6000
  (f : ℝ → ℝ)
  (h0 : f 0 = 1)
  (h1 : ∀ x : ℝ, f (x + 3) = f x + 2 * x + 3) :
  f 6000 = 12000001 :=
by
  sorry

end NUMINAMATH_GPT_function_value_at_6000_l1584_158412


namespace NUMINAMATH_GPT_fixed_point_exists_l1584_158473

noncomputable def fixed_point : Prop := ∀ d : ℝ, ∃ (p q : ℝ), (p = -3) ∧ (q = 45) ∧ (q = 5 * p^2 + d * p + 3 * d)

theorem fixed_point_exists : fixed_point :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_exists_l1584_158473


namespace NUMINAMATH_GPT_impossible_distinct_values_l1584_158483

theorem impossible_distinct_values :
  ∀ a b c : ℝ, 
  (a * (a - 4) = 12) → 
  (b * (b - 4) = 12) → 
  (c * (c - 4) = 12) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) → 
  false := 
sorry

end NUMINAMATH_GPT_impossible_distinct_values_l1584_158483


namespace NUMINAMATH_GPT_horner_rule_example_l1584_158451

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_rule_example : f 2 = 62 := by
  sorry

end NUMINAMATH_GPT_horner_rule_example_l1584_158451


namespace NUMINAMATH_GPT_cost_price_of_article_l1584_158495

theorem cost_price_of_article (C MP : ℝ) (h1 : 0.90 * MP = 1.25 * C) (h2 : 1.25 * C = 65.97) : C = 52.776 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l1584_158495


namespace NUMINAMATH_GPT_min_sum_intercepts_of_line_l1584_158480

theorem min_sum_intercepts_of_line (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 2 / b = 1) : a + b = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_intercepts_of_line_l1584_158480


namespace NUMINAMATH_GPT_tiffany_found_bags_l1584_158491

theorem tiffany_found_bags (initial_bags : ℕ) (total_bags : ℕ) (found_bags : ℕ) :
  initial_bags = 4 ∧ total_bags = 12 ∧ total_bags = initial_bags + found_bags → found_bags = 8 :=
by
  sorry

end NUMINAMATH_GPT_tiffany_found_bags_l1584_158491


namespace NUMINAMATH_GPT_train_length_is_300_l1584_158437

theorem train_length_is_300 (L V : ℝ)
    (h1 : L = V * 20)
    (h2 : L + 285 = V * 39) :
    L = 300 := by
  sorry

end NUMINAMATH_GPT_train_length_is_300_l1584_158437


namespace NUMINAMATH_GPT_total_black_dots_l1584_158494

def num_butterflies : ℕ := 397
def black_dots_per_butterfly : ℕ := 12

theorem total_black_dots : num_butterflies * black_dots_per_butterfly = 4764 := by
  sorry

end NUMINAMATH_GPT_total_black_dots_l1584_158494


namespace NUMINAMATH_GPT_Tom_total_yearly_intake_l1584_158466

def soda_weekday := 5 * 12
def water_weekday := 64
def juice_weekday := 3 * 8
def sports_drink_weekday := 2 * 16

def total_weekday_intake := soda_weekday + water_weekday + juice_weekday + sports_drink_weekday

def soda_weekend_holiday := 5 * 12
def water_weekend_holiday := 64
def juice_weekend_holiday := 3 * 8
def sports_drink_weekend_holiday := 1 * 16
def fruit_smoothie_weekend_holiday := 32

def total_weekend_holiday_intake := soda_weekend_holiday + water_weekend_holiday + juice_weekend_holiday + sports_drink_weekend_holiday + fruit_smoothie_weekend_holiday

def weekdays := 260
def weekend_days := 104
def holidays := 1

def total_yearly_intake := (weekdays * total_weekday_intake) + (weekend_days * total_weekend_holiday_intake) + (holidays * total_weekend_holiday_intake)

theorem Tom_total_yearly_intake :
  total_yearly_intake = 67380 := by
  sorry

end NUMINAMATH_GPT_Tom_total_yearly_intake_l1584_158466


namespace NUMINAMATH_GPT_transform_quadratic_to_linear_l1584_158405

theorem transform_quadratic_to_linear (x y : ℝ) : 
  x^2 - 4 * x * y + 4 * y^2 = 4 ↔ (x - 2 * y + 2 = 0 ∨ x - 2 * y - 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_transform_quadratic_to_linear_l1584_158405


namespace NUMINAMATH_GPT_handshakes_total_count_l1584_158496

/-
Statement:
There are 30 gremlins and 20 imps at a Regional Mischief Meet. Only half of the imps are willing to shake hands with each other.
All cooperative imps shake hands with each other. All imps shake hands with each gremlin. Gremlins shake hands with every
other gremlin as well as all the imps. Each pair of creatures shakes hands at most once. Prove that the total number of handshakes is 1080.
-/

theorem handshakes_total_count (gremlins imps cooperative_imps : ℕ)
  (H1 : gremlins = 30)
  (H2 : imps = 20)
  (H3 : cooperative_imps = imps / 2) :
  let handshakes_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_cooperative_imps := cooperative_imps * (cooperative_imps - 1) / 2
  let handshakes_imps_gremlins := imps * gremlins
  handshakes_gremlins + handshakes_cooperative_imps + handshakes_imps_gremlins = 1080 := 
by {
  sorry
}

end NUMINAMATH_GPT_handshakes_total_count_l1584_158496


namespace NUMINAMATH_GPT_least_wins_to_40_points_l1584_158476

theorem least_wins_to_40_points 
  (points_per_victory : ℕ)
  (points_per_draw : ℕ)
  (points_per_defeat : ℕ)
  (total_matches : ℕ)
  (initial_points : ℕ)
  (matches_played : ℕ)
  (target_points : ℕ) :
  points_per_victory = 3 →
  points_per_draw = 1 →
  points_per_defeat = 0 →
  total_matches = 20 →
  initial_points = 12 →
  matches_played = 5 →
  target_points = 40 →
  ∃ wins_needed : ℕ, wins_needed = 10 :=
by
  sorry

end NUMINAMATH_GPT_least_wins_to_40_points_l1584_158476


namespace NUMINAMATH_GPT_usual_time_to_bus_stop_l1584_158417

theorem usual_time_to_bus_stop
  (T : ℕ) (S : ℕ)
  (h : S * T = (4/5 * S) * (T + 9)) :
  T = 36 :=
by
  sorry

end NUMINAMATH_GPT_usual_time_to_bus_stop_l1584_158417


namespace NUMINAMATH_GPT_parallel_condition_l1584_158474

theorem parallel_condition (a : ℝ) : (a = -1) ↔ (¬ (a = -1 ∧ a ≠ 1)) ∧ (¬ (a ≠ -1 ∧ a = 1)) :=
by
  sorry

end NUMINAMATH_GPT_parallel_condition_l1584_158474


namespace NUMINAMATH_GPT_original_price_l1584_158404

variables (q r : ℝ) (h1 : 0 ≤ q) (h2 : 0 ≤ r)

theorem original_price (h : (2 : ℝ) = (1 + q / 100) * (1 - r / 100) * x) :
  x = 200 / (100 + q - r - (q * r) / 100) :=
by
  sorry

end NUMINAMATH_GPT_original_price_l1584_158404


namespace NUMINAMATH_GPT_floor_inequality_sqrt_l1584_158407

theorem floor_inequality_sqrt (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (⌊ m * Real.sqrt 2 ⌋) * (⌊ n * Real.sqrt 7 ⌋) < (⌊ m * n * Real.sqrt 14 ⌋) := 
by
  sorry

end NUMINAMATH_GPT_floor_inequality_sqrt_l1584_158407


namespace NUMINAMATH_GPT_new_total_weight_correct_l1584_158472

-- Definitions based on the problem statement
variables (R S k : ℝ)
def ram_original_weight : ℝ := 2 * k
def shyam_original_weight : ℝ := 5 * k
def ram_new_weight : ℝ := 1.10 * (ram_original_weight k)
def shyam_new_weight : ℝ := 1.17 * (shyam_original_weight k)

-- Definition for total original weight and increased weight
def total_original_weight : ℝ := ram_original_weight k + shyam_original_weight k
def total_weight_increased : ℝ := 1.15 * total_original_weight k
def new_total_weight : ℝ := ram_new_weight k + shyam_new_weight k

-- The proof statement
theorem new_total_weight_correct :
  new_total_weight k = total_weight_increased k :=
by
  sorry

end NUMINAMATH_GPT_new_total_weight_correct_l1584_158472


namespace NUMINAMATH_GPT_trapezoid_perimeter_l1584_158400

theorem trapezoid_perimeter (x y : ℝ) (h1 : x ≠ 0)
  (h2 : ∀ (AB CD AD BC : ℝ), AB = 2 * x ∧ CD = 4 * x ∧ AD = 2 * y ∧ BC = y) :
  (∀ (P : ℝ), P = AB + BC + CD + AD → P = 6 * x + 3 * y) :=
by sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l1584_158400


namespace NUMINAMATH_GPT_overall_winning_percentage_is_fifty_l1584_158488

def winning_percentage_of_first_games := (40 / 100) * 30
def total_games_played := 40
def remaining_games := total_games_played - 30
def winning_percentage_of_remaining_games := (80 / 100) * remaining_games
def total_games_won := winning_percentage_of_first_games + winning_percentage_of_remaining_games

theorem overall_winning_percentage_is_fifty : 
  (total_games_won / total_games_played) * 100 = 50 := 
by
  sorry

end NUMINAMATH_GPT_overall_winning_percentage_is_fifty_l1584_158488


namespace NUMINAMATH_GPT_remaining_coins_denomination_l1584_158481

def denomination_of_remaining_coins (total_coins : ℕ) (total_value : ℕ) (paise_20_count : ℕ) (paise_20_value : ℕ) : ℕ :=
  let remaining_coins := total_coins - paise_20_count
  let remaining_value := total_value - paise_20_count * paise_20_value
  remaining_value / remaining_coins

theorem remaining_coins_denomination :
  denomination_of_remaining_coins 334 7100 250 20 = 25 :=
by
  sorry

end NUMINAMATH_GPT_remaining_coins_denomination_l1584_158481


namespace NUMINAMATH_GPT_given_problem_l1584_158421

theorem given_problem (x y : ℝ) (hx : x ≠ 0) (hx4 : x ≠ 4) (hy : y ≠ 0) (hy6 : y ≠ 6) :
  (2 / x + 3 / y = 1 / 2) ↔ (4 * y / (y - 6) = x) :=
sorry

end NUMINAMATH_GPT_given_problem_l1584_158421


namespace NUMINAMATH_GPT_angle_bisectors_triangle_l1584_158456

theorem angle_bisectors_triangle
  (A B C I D K E : Type)
  (triangle : ∀ (A B C : Type), Prop)
  (is_incenter : ∀ (I A B C : Type), Prop)
  (is_on_arc_centered_at : ∀ (X Y : Type), Prop)
  (is_altitude_intersection : ∀ (X Y : Type), Prop)
  (angle_BIC : ∀ (B C : Type), ℝ)
  (angle_DKE : ∀ (D K E : Type), ℝ)
  (α β γ : ℝ)
  (h_sum_ang : α + β + γ = 180) :
  is_incenter I A B C →
  is_on_arc_centered_at D A → is_on_arc_centered_at K A → is_on_arc_centered_at E A →
  is_altitude_intersection E A →
  angle_BIC B C = 180 - (β + γ) / 2 →
  angle_DKE D K E = (360 - α) / 2 →
  angle_BIC B C + angle_DKE D K E = 270 :=
by sorry

end NUMINAMATH_GPT_angle_bisectors_triangle_l1584_158456


namespace NUMINAMATH_GPT_remainder_when_abc_divided_by_7_l1584_158455

theorem remainder_when_abc_divided_by_7 (a b c : ℕ) (h0 : a < 7) (h1 : b < 7) (h2 : c < 7)
  (h3 : (a + 2 * b + 3 * c) % 7 = 0)
  (h4 : (2 * a + 3 * b + c) % 7 = 4)
  (h5 : (3 * a + b + 2 * c) % 7 = 4) :
  (a * b * c) % 7 = 6 := 
sorry

end NUMINAMATH_GPT_remainder_when_abc_divided_by_7_l1584_158455


namespace NUMINAMATH_GPT_expansion_correct_l1584_158482

variable (x y : ℝ)

theorem expansion_correct : 
  (3 * x - 15) * (4 * y + 20) = 12 * x * y + 60 * x - 60 * y - 300 :=
by
  sorry

end NUMINAMATH_GPT_expansion_correct_l1584_158482
