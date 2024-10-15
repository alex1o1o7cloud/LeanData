import Mathlib

namespace NUMINAMATH_GPT_original_number_l309_30947

theorem original_number (x : ℕ) : x * 16 = 3408 → x = 213 := by
  intro h
  sorry

end NUMINAMATH_GPT_original_number_l309_30947


namespace NUMINAMATH_GPT_approximation_example1_approximation_example2_approximation_example3_l309_30912

theorem approximation_example1 (α β : ℝ) (hα : α = 0.0023) (hβ : β = 0.0057) :
  (1 + α) * (1 + β) = 1.008 := sorry

theorem approximation_example2 (α β : ℝ) (hα : α = 0.05) (hβ : β = -0.03) :
  (1 + α) * (10 + β) = 10.02 := sorry

theorem approximation_example3 (α β γ : ℝ) (hα : α = 0.03) (hβ : β = -0.01) (hγ : γ = -0.02) :
  (1 + α) * (1 + β) * (1 + γ) = 1 := sorry

end NUMINAMATH_GPT_approximation_example1_approximation_example2_approximation_example3_l309_30912


namespace NUMINAMATH_GPT_AgathaAdditionalAccessories_l309_30973

def AgathaBudget : ℕ := 250
def Frame : ℕ := 85
def FrontWheel : ℕ := 35
def RearWheel : ℕ := 40
def Seat : ℕ := 25
def HandlebarTape : ℕ := 15
def WaterBottleCage : ℕ := 10
def BikeLock : ℕ := 20
def FutureExpenses : ℕ := 10

theorem AgathaAdditionalAccessories :
  AgathaBudget - (Frame + FrontWheel + RearWheel + Seat + HandlebarTape + WaterBottleCage + BikeLock + FutureExpenses) = 10 := by
  sorry

end NUMINAMATH_GPT_AgathaAdditionalAccessories_l309_30973


namespace NUMINAMATH_GPT_transformed_ellipse_l309_30933

-- Define the original equation and the transformation
def orig_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def trans_x (x' : ℝ) : ℝ := x' / 5
noncomputable def trans_y (y' : ℝ) : ℝ := y' / 4

-- Prove that the transformed equation is an ellipse with specified properties
theorem transformed_ellipse :
  (∃ x' y' : ℝ, (trans_x x')^2 + (trans_y y')^2 = 1) →
  ∃ a b : ℝ, (a = 10) ∧ (b = 8) ∧ (∀ x' y' : ℝ, x'^2 / (a/2)^2 + y'^2 / (b/2)^2 = 1) :=
sorry

end NUMINAMATH_GPT_transformed_ellipse_l309_30933


namespace NUMINAMATH_GPT_cost_per_order_of_pakoras_l309_30935

noncomputable def samosa_cost : ℕ := 2
noncomputable def samosa_count : ℕ := 3
noncomputable def mango_lassi_cost : ℕ := 2
noncomputable def pakora_count : ℕ := 4
noncomputable def tip_percentage : ℚ := 0.25
noncomputable def total_cost_with_tax : ℚ := 25

theorem cost_per_order_of_pakoras (P : ℚ)
  (h1 : samosa_cost * samosa_count = 6)
  (h2 : mango_lassi_cost = 2)
  (h3 : 1.25 * (samosa_cost * samosa_count + mango_lassi_cost + pakora_count * P) = total_cost_with_tax) :
  P = 3 :=
by
  -- sorry ⟹ sorry
  sorry

end NUMINAMATH_GPT_cost_per_order_of_pakoras_l309_30935


namespace NUMINAMATH_GPT_sequence_a4_value_l309_30923

theorem sequence_a4_value :
  ∃ (a : ℕ → ℕ), (a 1 = 1) ∧ (∀ n, a (n+1) = 2 * a n + 1) ∧ (a 4 = 15) :=
by
  sorry

end NUMINAMATH_GPT_sequence_a4_value_l309_30923


namespace NUMINAMATH_GPT_prod_of_extrema_l309_30964

noncomputable def f (x k : ℝ) : ℝ := (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

theorem prod_of_extrema (k : ℝ) (h : ∀ x : ℝ, f x k ≥ 0 ∧ f x k ≤ 1 + (k - 1) / 3) :
  (∀ x : ℝ, f x k ≤ (k + 2) / 3) ∧ (∀ x : ℝ, f x k ≥ 1) → 
  (∃ φ ψ : ℝ, φ = 1 ∧ ψ = (k + 2) / 3 ∧ ∀ x y : ℝ, f x k = φ → f y k = ψ) → 
  (∃ φ ψ : ℝ, φ * ψ = (k + 2) / 3) :=
sorry

end NUMINAMATH_GPT_prod_of_extrema_l309_30964


namespace NUMINAMATH_GPT_four_students_same_acquaintances_l309_30965

theorem four_students_same_acquaintances
  (students : Finset ℕ)
  (acquainted : ∀ s ∈ students, (students \ {s}).card ≥ 68)
  (count : students.card = 102) :
  ∃ n, ∃ cnt, cnt ≥ 4 ∧ (∃ S, S ⊆ students ∧ S.card = cnt ∧ ∀ x ∈ S, (students \ {x}).card = n) :=
sorry

end NUMINAMATH_GPT_four_students_same_acquaintances_l309_30965


namespace NUMINAMATH_GPT_min_value_of_expr_l309_30908

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  ((x^2 + 1 / y^2 + 1) * (x^2 + 1 / y^2 - 1000)) +
  ((y^2 + 1 / x^2 + 1) * (y^2 + 1 / x^2 - 1000))

theorem min_value_of_expr :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ min_value_expr x y = -498998 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expr_l309_30908


namespace NUMINAMATH_GPT_division_result_l309_30941

theorem division_result : 180 / 6 / 3 / 2 = 5 := by
  sorry

end NUMINAMATH_GPT_division_result_l309_30941


namespace NUMINAMATH_GPT_paint_room_alone_l309_30926

theorem paint_room_alone (x : ℝ) (hx : (1 / x) + (1 / 4) = 1 / 1.714) : x = 3 :=
by sorry

end NUMINAMATH_GPT_paint_room_alone_l309_30926


namespace NUMINAMATH_GPT_paula_candies_distribution_l309_30952

-- Defining the given conditions and the question in Lean
theorem paula_candies_distribution :
  ∀ (initial_candies additional_candies friends : ℕ),
  initial_candies = 20 →
  additional_candies = 4 →
  friends = 6 →
  (initial_candies + additional_candies) / friends = 4 :=
by
  -- We skip the actual proof here
  intros initial_candies additional_candies friends h1 h2 h3
  sorry

end NUMINAMATH_GPT_paula_candies_distribution_l309_30952


namespace NUMINAMATH_GPT_find_remainder_l309_30988

noncomputable def remainder_expr_division (β : ℂ) (hβ : β^4 + β^3 + β^2 + β + 1 = 0) : ℂ :=
  1 - β

theorem find_remainder (β : ℂ) (hβ : β^4 + β^3 + β^2 + β + 1 = 0) :
  ∃ r, (x^45 + x^34 + x^23 + x^12 + 1) % (x^4 + x^3 + x^2 + x + 1) = r ∧ r = remainder_expr_division β hβ :=
sorry

end NUMINAMATH_GPT_find_remainder_l309_30988


namespace NUMINAMATH_GPT_inequality_condition_l309_30996

theorem inequality_condition {a b x y : ℝ} (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) : 
  (a^2 / x) + (b^2 / y) ≥ ((a + b)^2 / (x + y)) ∧ (a^2 / x) + (b^2 / y) = ((a + b)^2 / (x + y)) ↔ (x / y) = (a / b) :=
sorry

end NUMINAMATH_GPT_inequality_condition_l309_30996


namespace NUMINAMATH_GPT_zoey_preparation_months_l309_30951
open Nat

-- Define months as integers assuming 1 = January, 5 = May, 9 = September, etc.
def month_start : ℕ := 5 -- May
def month_exam : ℕ := 9 -- September

-- The function to calculate the number of months of preparation excluding the exam month.
def months_of_preparation (start : ℕ) (exam : ℕ) : ℕ := (exam - start)

theorem zoey_preparation_months :
  months_of_preparation month_start month_exam = 4 := by
  sorry

end NUMINAMATH_GPT_zoey_preparation_months_l309_30951


namespace NUMINAMATH_GPT_remaining_lawn_mowing_l309_30903

-- Definitions based on the conditions in the problem.
def Mary_mowing_time : ℝ := 3  -- Mary can mow the lawn in 3 hours
def John_mowing_time : ℝ := 6  -- John can mow the lawn in 6 hours
def John_work_time : ℝ := 3    -- John works for 3 hours

-- Question: How much of the lawn remains to be mowed?
theorem remaining_lawn_mowing : (Mary_mowing_time = 3) ∧ (John_mowing_time = 6) ∧ (John_work_time = 3) →
  (1 - (John_work_time / John_mowing_time) = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_remaining_lawn_mowing_l309_30903


namespace NUMINAMATH_GPT_donuts_eaten_on_monday_l309_30953

theorem donuts_eaten_on_monday (D : ℕ) (h1 : D + D / 2 + 4 * D = 49) : 
  D = 9 :=
sorry

end NUMINAMATH_GPT_donuts_eaten_on_monday_l309_30953


namespace NUMINAMATH_GPT_relationship_y1_y2_l309_30956

theorem relationship_y1_y2 (y1 y2 : ℝ) (m : ℝ) (h_m : m ≠ 0) 
  (hA : y1 = m * (-2) + 4) (hB : 3 = m * 1 + 4) (hC : y2 = m * 3 + 4) : y1 > y2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_l309_30956


namespace NUMINAMATH_GPT_max_ab_l309_30997

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 8) : 
  ab ≤ 8 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2 * b = 8 ∧ ab = 8 :=
by
  sorry

end NUMINAMATH_GPT_max_ab_l309_30997


namespace NUMINAMATH_GPT_paper_clips_distribution_l309_30931

theorem paper_clips_distribution (total_clips : ℕ) (num_boxes : ℕ) (clip_per_box : ℕ) 
  (h1 : total_clips = 81) (h2 : num_boxes = 9) : clip_per_box = 9 :=
by sorry

end NUMINAMATH_GPT_paper_clips_distribution_l309_30931


namespace NUMINAMATH_GPT_wire_attachment_distance_l309_30928

theorem wire_attachment_distance :
  ∃ x : ℝ, 
    (∀ z y : ℝ, z = Real.sqrt (x ^ 2 + 3.6 ^ 2) ∧ y = Real.sqrt ((x + 5) ^ 2 + 3.6 ^ 2) →
      z + y = 13) ∧
    abs ((x : ℝ) - 2.7) < 0.01 := -- Assuming numerical closeness within a small epsilon for practical solutions.
sorry -- Proof not provided.

end NUMINAMATH_GPT_wire_attachment_distance_l309_30928


namespace NUMINAMATH_GPT_jill_first_show_length_l309_30984

theorem jill_first_show_length : 
  ∃ (x : ℕ), (x + 4 * x = 150) ∧ (x = 30) :=
sorry

end NUMINAMATH_GPT_jill_first_show_length_l309_30984


namespace NUMINAMATH_GPT_greatest_integer_2e_minus_5_l309_30980

noncomputable def e : ℝ := 2.718

theorem greatest_integer_2e_minus_5 : ⌊2 * e - 5⌋ = 0 :=
by
  -- This is a placeholder for the actual proof. 
  sorry

end NUMINAMATH_GPT_greatest_integer_2e_minus_5_l309_30980


namespace NUMINAMATH_GPT_buttons_on_first_type_of_shirt_l309_30910

/--
The GooGoo brand of clothing manufactures two types of shirts.
- The first type of shirt has \( x \) buttons.
- The second type of shirt has 5 buttons.
- The department store ordered 200 shirts of each type.
- A total of 1600 buttons are used for the entire order.

Prove that the first type of shirt has exactly 3 buttons.
-/
theorem buttons_on_first_type_of_shirt (x : ℕ) 
  (h1 : 200 * x + 200 * 5 = 1600) : 
  x = 3 :=
  sorry

end NUMINAMATH_GPT_buttons_on_first_type_of_shirt_l309_30910


namespace NUMINAMATH_GPT_compute_modulo_l309_30963

theorem compute_modulo :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_GPT_compute_modulo_l309_30963


namespace NUMINAMATH_GPT_students_taking_history_but_not_statistics_l309_30959

-- Definitions based on conditions
def T : Nat := 150
def H : Nat := 58
def S : Nat := 42
def H_union_S : Nat := 95

-- Statement to prove
theorem students_taking_history_but_not_statistics : H - (H + S - H_union_S) = 53 :=
by
  sorry

end NUMINAMATH_GPT_students_taking_history_but_not_statistics_l309_30959


namespace NUMINAMATH_GPT_tom_should_pay_times_original_price_l309_30948

-- Definitions of the given conditions
def original_price : ℕ := 3
def amount_paid : ℕ := 9

-- The theorem to prove
theorem tom_should_pay_times_original_price : ∃ k : ℕ, amount_paid = k * original_price ∧ k = 3 :=
by 
  -- Using sorry to skip the proof for now
  sorry

end NUMINAMATH_GPT_tom_should_pay_times_original_price_l309_30948


namespace NUMINAMATH_GPT_largest_negative_integer_l309_30981

theorem largest_negative_integer :
  ∃ (n : ℤ), (∀ m : ℤ, m < 0 → m ≤ n) ∧ n = -1 := by
  sorry

end NUMINAMATH_GPT_largest_negative_integer_l309_30981


namespace NUMINAMATH_GPT_solution_1_solution_2_l309_30938

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (2 * x + 3)

lemma f_piecewise (x : ℝ) : 
  f x = if x ≤ -3 / 2 then -4 * x - 2
        else if -3 / 2 < x ∧ x < 1 / 2 then 4
        else 4 * x + 2 := 
by
-- This lemma represents the piecewise definition of f(x)
sorry

theorem solution_1 : 
  (∀ x : ℝ, f x < 5 ↔ (-7 / 4 < x ∧ x < 3 / 4)) := 
by 
-- Proof of the inequality solution
sorry

theorem solution_2 : 
  (∀ t : ℝ, (∀ x : ℝ, f x - t ≥ 0) → t ≤ 4) :=
by
-- Proof that the maximum value of t is 4
sorry

end NUMINAMATH_GPT_solution_1_solution_2_l309_30938


namespace NUMINAMATH_GPT_baseball_batter_at_bats_left_l309_30936

theorem baseball_batter_at_bats_left (L R H_L H_R : ℕ) (h1 : L + R = 600)
    (h2 : H_L + H_R = 192) (h3 : H_L = 25 / 100 * L) (h4 : H_R = 35 / 100 * R) : 
    L = 180 :=
by
  sorry

end NUMINAMATH_GPT_baseball_batter_at_bats_left_l309_30936


namespace NUMINAMATH_GPT_initial_bacteria_count_l309_30924

theorem initial_bacteria_count (doubling_time : ℕ) (initial_time : ℕ) (initial_bacteria : ℕ) 
(final_bacteria : ℕ) (doubling_rate : initial_time / doubling_time = 8 ∧ final_bacteria = 524288) : 
  initial_bacteria = 2048 :=
by
  sorry

end NUMINAMATH_GPT_initial_bacteria_count_l309_30924


namespace NUMINAMATH_GPT_find_smallest_even_number_l309_30993

theorem find_smallest_even_number (x : ℕ) (h1 : 
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14)) = 424) : 
  x = 46 := 
by
  sorry

end NUMINAMATH_GPT_find_smallest_even_number_l309_30993


namespace NUMINAMATH_GPT_floating_time_l309_30999

theorem floating_time (boat_with_current: ℝ) (boat_against_current: ℝ) (distance: ℝ) (time: ℝ) : 
boat_with_current = 28 ∧ boat_against_current = 24 ∧ distance = 20 ∧ 
time = distance / ((boat_with_current - boat_against_current) / 2) → 
time = 10 := by
  sorry

end NUMINAMATH_GPT_floating_time_l309_30999


namespace NUMINAMATH_GPT_knights_and_liars_solution_l309_30998

-- Definitions of each person's statement as predicates
def person1_statement (liar : ℕ → Prop) : Prop := liar 2 ∧ liar 3 ∧ liar 4 ∧ liar 5 ∧ liar 6
def person2_statement (liar : ℕ → Prop) : Prop := liar 1 ∧ ∀ i, i ≠ 1 → ¬ liar i
def person3_statement (liar : ℕ → Prop) : Prop := liar 4 ∧ liar 5 ∧ liar 6 ∧ ¬ liar 3 ∧ ¬ liar 2 ∧ ¬ liar 1
def person4_statement (liar : ℕ → Prop) : Prop := liar 1 ∧ liar 2 ∧ liar 3 ∧ ∀ i, i > 3 → ¬ liar i
def person5_statement (liar : ℕ → Prop) : Prop := liar 6 ∧ ∀ i, i ≠ 6 → ¬ liar i
def person6_statement (liar : ℕ → Prop) : Prop := liar 5 ∧ ∀ i, i ≠ 5 → ¬ liar i

-- Definition of a knight and a liar
def is_knight (statement : Prop) : Prop := statement
def is_liar (statement : Prop) : Prop := ¬ statement

-- Defining the theorem
theorem knights_and_liars_solution (knight liar : ℕ → Prop) : 
  is_liar (person1_statement liar) ∧ 
  is_knight (person2_statement liar) ∧ 
  is_liar (person3_statement liar) ∧ 
  is_liar (person4_statement liar) ∧ 
  is_knight (person5_statement liar) ∧ 
  is_liar (person6_statement liar) :=
by
  sorry

end NUMINAMATH_GPT_knights_and_liars_solution_l309_30998


namespace NUMINAMATH_GPT_geometric_sequence_a5_l309_30932

theorem geometric_sequence_a5 {a : ℕ → ℝ} 
  (h_geom : ∃ r, ∀ n, a (n + 1) = r * a n) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) : 
  a 5 = -8 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l309_30932


namespace NUMINAMATH_GPT_area_of_circle_l309_30983

def circle_area (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y = 1

theorem area_of_circle : ∃ (area : ℝ), area = 6 * Real.pi :=
by sorry

end NUMINAMATH_GPT_area_of_circle_l309_30983


namespace NUMINAMATH_GPT_factorial_of_6_is_720_l309_30934

theorem factorial_of_6_is_720 : (Nat.factorial 6) = 720 := by
  sorry

end NUMINAMATH_GPT_factorial_of_6_is_720_l309_30934


namespace NUMINAMATH_GPT_mark_total_cents_l309_30942

theorem mark_total_cents (dimes nickels : ℕ) (h1 : nickels = dimes + 3) (h2 : dimes = 5) : 
  dimes * 10 + nickels * 5 = 90 := by
  sorry

end NUMINAMATH_GPT_mark_total_cents_l309_30942


namespace NUMINAMATH_GPT_functional_equation_solution_l309_30944

open Function

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x ^ 2 + f y) = y + f x ^ 2) → (∀ x : ℝ, f x = x) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l309_30944


namespace NUMINAMATH_GPT_line_parallel_unique_a_l309_30957

theorem line_parallel_unique_a (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y + a + 3 = 0 → x + (a + 1)*y + 4 = 0) → a = -2 :=
  by
  sorry

end NUMINAMATH_GPT_line_parallel_unique_a_l309_30957


namespace NUMINAMATH_GPT_geometric_sequence_sum_l309_30940

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l309_30940


namespace NUMINAMATH_GPT_flight_height_l309_30904

theorem flight_height (flights : ℕ) (step_height_in_inches : ℕ) (total_steps : ℕ) 
    (H1 : flights = 9) (H2 : step_height_in_inches = 18) (H3 : total_steps = 60) : 
    (total_steps * step_height_in_inches) / 12 / flights = 10 :=
by
  sorry

end NUMINAMATH_GPT_flight_height_l309_30904


namespace NUMINAMATH_GPT_arithmetic_sequence_and_sum_properties_l309_30901

noncomputable def a_n (n : ℕ) : ℤ := 30 - 2 * n
noncomputable def S_n (n : ℕ) : ℤ := -n^2 + 29 * n

theorem arithmetic_sequence_and_sum_properties :
  (a_n 3 = 24 ∧ a_n 6 = 18) ∧
  (∀ n : ℕ, (S_n n = (n * (a_n 1 + a_n n)) / 2) ∧ ((a_n 3 = 24 ∧ a_n 6 = 18) → ∀ n : ℕ, a_n n = 30 - 2 * n)) ∧
  (S_n 14 = 210) :=
by 
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_and_sum_properties_l309_30901


namespace NUMINAMATH_GPT_students_in_diligence_before_transfer_l309_30978

theorem students_in_diligence_before_transfer (D I P : ℕ)
  (h_total : D + I + P = 75)
  (h_equal : D + 2 = I - 2 + 3 ∧ D + 2 = P - 3) :
  D = 23 :=
by
  sorry

end NUMINAMATH_GPT_students_in_diligence_before_transfer_l309_30978


namespace NUMINAMATH_GPT_junior_girls_count_l309_30930

theorem junior_girls_count 
  (total_players : ℕ) 
  (boys_percentage : ℝ) 
  (junior_girls : ℕ)
  (h_team : total_players = 50)
  (h_boys_pct : boys_percentage = 0.6)
  (h_junior_girls : junior_girls = ((total_players : ℝ) * (1 - boys_percentage) * 0.5)) : 
  junior_girls = 10 := 
by 
  sorry

end NUMINAMATH_GPT_junior_girls_count_l309_30930


namespace NUMINAMATH_GPT_true_proposition_B_l309_30909

theorem true_proposition_B : (3 > 4) ∨ (3 < 4) :=
sorry

end NUMINAMATH_GPT_true_proposition_B_l309_30909


namespace NUMINAMATH_GPT_fraction_equality_implies_equality_l309_30990

theorem fraction_equality_implies_equality (a b c : ℝ) (hc : c ≠ 0) :
  (a / c = b / c) → (a = b) :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_equality_implies_equality_l309_30990


namespace NUMINAMATH_GPT_hyperbola_equation_l309_30946

theorem hyperbola_equation
  (a b : ℝ) 
  (a_pos : a > 0) 
  (b_pos : b > 0) 
  (focus_at_five : a^2 + b^2 = 25) 
  (asymptote_ratio : b / a = 3 / 4) :
  (a = 4 ∧ b = 3 ∧ ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1) ↔ ( ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1 ):=
sorry 

end NUMINAMATH_GPT_hyperbola_equation_l309_30946


namespace NUMINAMATH_GPT_solve_for_M_l309_30920

def M : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ 2 * x + y = 2 ∧ x - y = 1 }

theorem solve_for_M : M = { (1, 0) } := by
  sorry

end NUMINAMATH_GPT_solve_for_M_l309_30920


namespace NUMINAMATH_GPT_at_least_one_non_negative_l309_30949

theorem at_least_one_non_negative 
  (a b c d e f g h : ℝ) : 
  ac + bd ≥ 0 ∨ ae + bf ≥ 0 ∨ ag + bh ≥ 0 ∨ ce + df ≥ 0 ∨ cg + dh ≥ 0 ∨ eg + fh ≥ 0 := 
sorry

end NUMINAMATH_GPT_at_least_one_non_negative_l309_30949


namespace NUMINAMATH_GPT_polynomial_divisible_x_minus_2_l309_30921

theorem polynomial_divisible_x_minus_2 (m : ℝ) : 
  (3 * 2^2 - 9 * 2 + m = 0) → m = 6 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisible_x_minus_2_l309_30921


namespace NUMINAMATH_GPT_point_in_second_quadrant_l309_30911

theorem point_in_second_quadrant (m : ℝ) (h : 2 > 0 ∧ m < 0) : m < 0 :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l309_30911


namespace NUMINAMATH_GPT_mean_of_combined_sets_l309_30977

theorem mean_of_combined_sets (mean_set1 : ℝ) (mean_set2 : ℝ) (n1 : ℕ) (n2 : ℕ)
  (h1 : mean_set1 = 15) (h2 : mean_set2 = 27) (h3 : n1 = 7) (h4 : n2 = 8) :
  (mean_set1 * n1 + mean_set2 * n2) / (n1 + n2) = 21.4 := 
sorry

end NUMINAMATH_GPT_mean_of_combined_sets_l309_30977


namespace NUMINAMATH_GPT_problem_statement_l309_30925

variables {x y x1 y1 a b c d : ℝ}

-- The main theorem statement
theorem problem_statement (h0 : ∀ (x y : ℝ), 6 * y ^ 2 = 2 * x ^ 3 + 3 * x ^ 2 + x) 
                           (h1 : x1 = a * x + b) 
                           (h2 : y1 = c * y + d) 
                           (h3 : y1 ^ 2 = x1 ^ 3 - 36 * x1) : 
                           a + b + c + d = 90 := sorry

end NUMINAMATH_GPT_problem_statement_l309_30925


namespace NUMINAMATH_GPT_problem_statement_l309_30968

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - 2 / x^2 + a / x

theorem problem_statement (a : ℝ) (k : ℝ) : 
  0 < a ∧ a ≤ 4 →
  (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 →
  |f x1 a - f x2 a| > k * |x1 - x2|) ↔
  k ≤ 2 - a^3 / 108 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l309_30968


namespace NUMINAMATH_GPT_candidate_knows_Excel_and_willing_nights_l309_30970

variable (PExcel PXNight : ℝ)
variable (H1 : PExcel = 0.20) (H2 : PXNight = 0.30)

theorem candidate_knows_Excel_and_willing_nights : (PExcel * PXNight) = 0.06 :=
by
  rw [H1, H2]
  norm_num

end NUMINAMATH_GPT_candidate_knows_Excel_and_willing_nights_l309_30970


namespace NUMINAMATH_GPT_price_per_butterfly_l309_30974

theorem price_per_butterfly (jars : ℕ) (caterpillars_per_jar : ℕ) (fail_percentage : ℝ) (total_money : ℝ) (price : ℝ) :
  jars = 4 →
  caterpillars_per_jar = 10 →
  fail_percentage = 0.40 →
  total_money = 72 →
  price = 3 :=
by
  intros h_jars h_caterpillars h_fail_percentage h_total_money
  -- Full proof here
  sorry

end NUMINAMATH_GPT_price_per_butterfly_l309_30974


namespace NUMINAMATH_GPT_unique_solution_inequality_l309_30992

theorem unique_solution_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a * x + a ∧ x^2 - a * x + a ≤ 1) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_inequality_l309_30992


namespace NUMINAMATH_GPT_contrapositive_proposition_l309_30979

theorem contrapositive_proposition :
  (∀ x : ℝ, (x^2 < 4 → -2 < x ∧ x < 2)) ↔ (∀ x : ℝ, (x ≤ -2 ∨ x ≥ 2 → x^2 ≥ 4)) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_proposition_l309_30979


namespace NUMINAMATH_GPT_vehicles_traveled_l309_30982

theorem vehicles_traveled (V : ℕ)
  (h1 : 40 * V = 800 * 100000000) : 
  V = 2000000000 := 
sorry

end NUMINAMATH_GPT_vehicles_traveled_l309_30982


namespace NUMINAMATH_GPT_same_terminal_side_angles_l309_30915

theorem same_terminal_side_angles (α : ℝ) : 
  (∃ k : ℤ, α = -457 + k * 360) ↔ (∃ k : ℤ, α = 263 + k * 360) :=
sorry

end NUMINAMATH_GPT_same_terminal_side_angles_l309_30915


namespace NUMINAMATH_GPT_total_amount_invested_l309_30906

variable (T : ℝ)

def income_first (T : ℝ) : ℝ :=
  0.10 * (T - 700)

def income_second : ℝ :=
  0.08 * 700

theorem total_amount_invested :
  income_first T - income_second = 74 → T = 2000 :=
by
  intros h
  sorry 

end NUMINAMATH_GPT_total_amount_invested_l309_30906


namespace NUMINAMATH_GPT_line_quadrants_l309_30939

theorem line_quadrants (k b : ℝ) (h : ∃ x y : ℝ, y = k * x + b ∧ 
                                          ((x > 0 ∧ y > 0) ∧   -- First quadrant
                                           (x < 0 ∧ y < 0) ∧   -- Third quadrant
                                           (x > 0 ∧ y < 0))) : -- Fourth quadrant
  k > 0 :=
sorry

end NUMINAMATH_GPT_line_quadrants_l309_30939


namespace NUMINAMATH_GPT_shortest_path_l309_30919

noncomputable def diameter : ℝ := 18
noncomputable def radius : ℝ := diameter / 2
noncomputable def AC : ℝ := 7
noncomputable def BD : ℝ := 7
noncomputable def CD : ℝ := diameter - AC - BD
noncomputable def CP : ℝ := Real.sqrt (radius ^ 2 - (CD / 2) ^ 2)
noncomputable def DP : ℝ := CP

theorem shortest_path (C P D : ℝ) :
  (C - 7) ^ 2 + (D - 7) ^ 2 = CD ^ 2 →
  (C = AC) ∧ (D = BD) →
  2 * CP = 2 * Real.sqrt 77 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_shortest_path_l309_30919


namespace NUMINAMATH_GPT_dividend_rate_correct_l309_30907

-- Define the stock's yield and market value
def stock_yield : ℝ := 0.08
def market_value : ℝ := 175

-- Dividend rate definition based on given yield and market value
def dividend_rate (yield market_value : ℝ) : ℝ :=
  (yield * market_value)

-- The problem statement to be proven in Lean
theorem dividend_rate_correct :
  dividend_rate stock_yield market_value = 14 := by
  sorry

end NUMINAMATH_GPT_dividend_rate_correct_l309_30907


namespace NUMINAMATH_GPT_tan_150_eq_neg_inv_sqrt3_l309_30905

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end NUMINAMATH_GPT_tan_150_eq_neg_inv_sqrt3_l309_30905


namespace NUMINAMATH_GPT_consumption_increased_by_27_91_percent_l309_30989
noncomputable def percentage_increase_in_consumption (T C : ℝ) : ℝ :=
  let new_tax_rate := 0.86 * T
  let new_revenue_effect := 1.1000000000000085
  let cons_percentage_increase (P : ℝ) := (new_tax_rate * (C * (1 + P))) = new_revenue_effect * (T * C)
  let P_solution := 0.2790697674418605
  if cons_percentage_increase P_solution then P_solution * 100 else 0

-- The statement we are proving
theorem consumption_increased_by_27_91_percent (T C : ℝ) (hT : 0 < T) (hC : 0 < C) :
  percentage_increase_in_consumption T C = 27.91 :=
by
  sorry

end NUMINAMATH_GPT_consumption_increased_by_27_91_percent_l309_30989


namespace NUMINAMATH_GPT_fraction_of_students_paired_l309_30969

theorem fraction_of_students_paired {t s : ℕ} 
  (h1 : t / 4 = s / 3) : 
  (t / 4 + s / 3) / (t + s) = 2 / 7 := by sorry

end NUMINAMATH_GPT_fraction_of_students_paired_l309_30969


namespace NUMINAMATH_GPT_range_of_m_cond_l309_30913

noncomputable def quadratic_inequality (x m : ℝ) : Prop :=
  x^2 + m * x + 2 * m - 3 ≥ 0

theorem range_of_m_cond (m : ℝ) (h1 : 2 ≤ m) (h2 : m ≤ 6) (x : ℝ) :
  quadratic_inequality x m :=
sorry

end NUMINAMATH_GPT_range_of_m_cond_l309_30913


namespace NUMINAMATH_GPT_sum_first_n_odd_eq_n_squared_l309_30917

theorem sum_first_n_odd_eq_n_squared (n : ℕ) : (Finset.sum (Finset.range n) (fun k => (2 * k + 1)) = n^2) := sorry

end NUMINAMATH_GPT_sum_first_n_odd_eq_n_squared_l309_30917


namespace NUMINAMATH_GPT_problem_divisibility_l309_30950

theorem problem_divisibility (n : ℕ) : ∃ k : ℕ, 2 ^ (3 ^ n) + 1 = 3 ^ (n + 1) * k :=
sorry

end NUMINAMATH_GPT_problem_divisibility_l309_30950


namespace NUMINAMATH_GPT_pairs_of_old_roller_skates_l309_30937

def cars := 2
def bikes := 2
def trash_can := 1
def tricycle := 1
def car_wheels := 4
def bike_wheels := 2
def trash_can_wheels := 2
def tricycle_wheels := 3
def total_wheels := 25

def roller_skates_wheels := 2
def skates_per_pair := 2

theorem pairs_of_old_roller_skates : (total_wheels - (cars * car_wheels + bikes * bike_wheels + trash_can * trash_can_wheels + tricycle * tricycle_wheels)) / roller_skates_wheels / skates_per_pair = 2 := by
  sorry

end NUMINAMATH_GPT_pairs_of_old_roller_skates_l309_30937


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l309_30972

theorem right_triangle_hypotenuse
  (a b c : ℝ)
  (h₀ : a = 24)
  (h₁ : a^2 + b^2 + c^2 = 2500)
  (h₂ : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l309_30972


namespace NUMINAMATH_GPT_wire_leftover_length_l309_30922

-- Define given conditions as variables/constants
def initial_wire_length : ℝ := 60
def side_length : ℝ := 9
def sides_in_square : ℕ := 4

-- Define the theorem: prove leftover wire length is 24 after creating the square
theorem wire_leftover_length :
  initial_wire_length - sides_in_square * side_length = 24 :=
by
  -- proof steps are not required, so we use sorry to indicate where the proof should be
  sorry

end NUMINAMATH_GPT_wire_leftover_length_l309_30922


namespace NUMINAMATH_GPT_part1_part2_l309_30966

theorem part1 (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a * b > 0) : a + b = 8 ∨ a + b = -8 :=
sorry

theorem part2 (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h4 : |a + b| = a + b) : a - b = 4 ∨ a - b = 8 :=
sorry

end NUMINAMATH_GPT_part1_part2_l309_30966


namespace NUMINAMATH_GPT_equal_mass_piles_l309_30955

theorem equal_mass_piles (n : ℕ) (hn : n > 3) (hn_mod : n % 3 = 0 ∨ n % 3 = 2) : 
  ∃ A B C : Finset ℕ, A ∪ B ∪ C = {i | i ∈ Finset.range (n + 1)} ∧
  Disjoint A B ∧ Disjoint A C ∧ Disjoint B C ∧
  A.sum id = B.sum id ∧ B.sum id = C.sum id :=
sorry

end NUMINAMATH_GPT_equal_mass_piles_l309_30955


namespace NUMINAMATH_GPT_initial_deposit_l309_30961

variable (P R : ℝ)

theorem initial_deposit (h1 : P + (P * R * 3) / 100 = 11200)
                       (h2 : P + (P * (R + 2) * 3) / 100 = 11680) :
  P = 8000 :=
by
  sorry

end NUMINAMATH_GPT_initial_deposit_l309_30961


namespace NUMINAMATH_GPT_tiger_initial_leaps_behind_l309_30994

theorem tiger_initial_leaps_behind (tiger_leap_distance deer_leap_distance tiger_leaps_per_minute deer_leaps_per_minute total_distance_to_catch initial_leaps_behind : ℕ) 
  (h1 : tiger_leap_distance = 8) 
  (h2 : deer_leap_distance = 5) 
  (h3 : tiger_leaps_per_minute = 5) 
  (h4 : deer_leaps_per_minute = 4) 
  (h5 : total_distance_to_catch = 800) :
  initial_leaps_behind = 40 := 
by
  -- Leaving proof body incomplete as it is not required
  sorry

end NUMINAMATH_GPT_tiger_initial_leaps_behind_l309_30994


namespace NUMINAMATH_GPT_incorrect_option_B_l309_30975

-- Definitions of the given conditions
def optionA (a : ℝ) : Prop := (8 * a = 8 * a)
def optionB (a : ℝ) : Prop := (a - (0.08 * a) = 8 * a)
def optionC (a : ℝ) : Prop := (8 * a = 8 * a)
def optionD (a : ℝ) : Prop := (a * 8 = 8 * a)

-- The statement to be proved
theorem incorrect_option_B (a : ℝ) : 
  optionA a ∧ ¬optionB a ∧ optionC a ∧ optionD a := 
by
  sorry

end NUMINAMATH_GPT_incorrect_option_B_l309_30975


namespace NUMINAMATH_GPT_tan_alpha_minus_beta_l309_30929

theorem tan_alpha_minus_beta
  (α β : ℝ)
  (tan_alpha : Real.tan α = 2)
  (tan_beta : Real.tan β = -7) :
  Real.tan (α - β) = -9 / 13 :=
by sorry

end NUMINAMATH_GPT_tan_alpha_minus_beta_l309_30929


namespace NUMINAMATH_GPT_find_larger_number_l309_30987

theorem find_larger_number 
  (x y : ℤ)
  (h1 : x + y = 37)
  (h2 : x - y = 5) : max x y = 21 := 
sorry

end NUMINAMATH_GPT_find_larger_number_l309_30987


namespace NUMINAMATH_GPT_range_of_independent_variable_l309_30902

theorem range_of_independent_variable (x : ℝ) : 
  (∃ y, y = x / (Real.sqrt (x + 4)) + 1 / (x - 1)) ↔ x > -4 ∧ x ≠ 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_independent_variable_l309_30902


namespace NUMINAMATH_GPT_contestant_score_l309_30991

theorem contestant_score (highest_score lowest_score : ℕ) (average_score : ℕ)
  (h_hs : highest_score = 86)
  (h_ls : lowest_score = 45)
  (h_avg : average_score = 76) :
  (76 * 9 - 86 - 45) / 7 = 79 := 
by 
  sorry

end NUMINAMATH_GPT_contestant_score_l309_30991


namespace NUMINAMATH_GPT_rowing_time_l309_30967

def man_speed_still := 10.0
def river_speed := 1.2
def total_distance := 9.856

def upstream_speed := man_speed_still - river_speed
def downstream_speed := man_speed_still + river_speed

def one_way_distance := total_distance / 2
def time_upstream := one_way_distance / upstream_speed
def time_downstream := one_way_distance / downstream_speed

theorem rowing_time :
  time_upstream + time_downstream = 1 :=
by
  sorry

end NUMINAMATH_GPT_rowing_time_l309_30967


namespace NUMINAMATH_GPT_probability_X_eq_Y_l309_30914

theorem probability_X_eq_Y
  (x y : ℝ)
  (h1 : -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi)
  (h2 : -5 * Real.pi ≤ y ∧ y ≤ 5 * Real.pi)
  (h3 : Real.cos (Real.cos x) = Real.cos (Real.cos y)) :
  (∃ N : ℕ, N = 100 ∧ ∃ M : ℕ, M = 11 ∧ M / N = (11 : ℝ) / 100) :=
by sorry

end NUMINAMATH_GPT_probability_X_eq_Y_l309_30914


namespace NUMINAMATH_GPT_sequence_sum_l309_30943

def alternating_sum : List ℤ := [2, -7, 10, -15, 18, -23, 26, -31, 34, -39, 40, -45, 48]

theorem sequence_sum : alternating_sum.sum = 13 := by
  sorry

end NUMINAMATH_GPT_sequence_sum_l309_30943


namespace NUMINAMATH_GPT_music_player_winner_l309_30971

theorem music_player_winner (n : ℕ) (h1 : ∀ k, k % n = 0 → k = 35) (h2 : 35 % 7 = 0) (h3 : 35 % n = 0) (h4 : n ≠ 1) (h5 : n ≠ 7) (h6 : n ≠ 35) : n = 5 := 
sorry

end NUMINAMATH_GPT_music_player_winner_l309_30971


namespace NUMINAMATH_GPT_problem_solution_l309_30954

theorem problem_solution :
  ∀ (a b c d : ℝ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
    (a^2 = 7 ∨ a^2 = 8) →
    (b^2 = 7 ∨ b^2 = 8) →
    (c^2 = 7 ∨ c^2 = 8) →
    (d^2 = 7 ∨ d^2 = 8) →
    a^2 + b^2 + c^2 + d^2 = 30 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l309_30954


namespace NUMINAMATH_GPT_part1_part2_part3_l309_30958

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  abs (x^2 - 1) + x^2 + k * x

theorem part1 (h : 2 = 2) :
  (f (- (1 + Real.sqrt 3) /2) 2 = 0) ∧ (f (-1/2) 2 = 0) := by
  sorry

theorem part2 (h_alpha : 0 < α) (h_beta : α < β) (h_beta2 : β < 2) (h_f_alpha : f α k = 0) (h_f_beta : f β k = 0) :
  -7/2 < k ∧ k < -1 := by
  sorry

theorem part3 (h_alpha : 0 < α) (h_alpha1 : α ≤ 1) (h_beta1 : 1 < β) (h_beta2 : β < 2) (h1 : k = - 1 / α) (h2 : 2 * β^2 + k * β - 1 = 0) :
  1/α + 1/β < 4 := by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l309_30958


namespace NUMINAMATH_GPT_solution_set_inequality_l309_30976

noncomputable def solution_set (x : ℝ) : Prop :=
  (2 * x - 1) / (x + 2) > 1

theorem solution_set_inequality :
  { x : ℝ | solution_set x } = { x : ℝ | x < -2 ∨ x > 3 } := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l309_30976


namespace NUMINAMATH_GPT_minimum_value_expr_l309_30985

theorem minimum_value_expr (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  (1 + (1 / m)) * (1 + (1 / n)) = 9 :=
sorry

end NUMINAMATH_GPT_minimum_value_expr_l309_30985


namespace NUMINAMATH_GPT_part1_part2_part3_l309_30962

-- Part 1: Proving a₁ for given a₃, p, and q
theorem part1 (a : ℕ → ℝ) (p q : ℝ) (h1 : p = (1/2)) (h2 : q = 2) 
  (h3 : a 3 = 41 / 20) (h4 : ∀ n, a (n + 1) = p * a n + q / a n) :
  a 1 = 1 ∨ a 1 = 4 := 
sorry

-- Part 2: Finding the sum Sₙ of the first n terms given a₁ and p * q = 0
theorem part2 (a : ℕ → ℝ) (p q : ℝ) (h1 : a 1 = 5) (h2 : p * q = 0) 
  (h3 : ∀ n, a (n + 1) = p * a n + q / a n) 
  (S : ℕ → ℝ) (n : ℕ) :
    S n = (25 * n + q * n + q - 25) / 10 ∨ 
    S n = (25 * n + q * n) / 10 ∨ 
    S n = (5 * (p^n - 1)) / (p - 1) ∨ 
    S n = 5 * n :=
sorry

-- Part 3: Proving the range of p given a₁, q and that the sequence is monotonically decreasing
theorem part3 (a : ℕ → ℝ) (p q : ℝ) (h1 : a 1 = 2) (h2 : q = 1) 
  (h3 : ∀ n, a (n + 1) = p * a n + q / a n) 
  (h4 : ∀ n, a (n + 1) < a n) :
  1/2 < p ∧ p < 3/4 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l309_30962


namespace NUMINAMATH_GPT_alpha_beta_sum_pi_over_2_l309_30927

theorem alpha_beta_sum_pi_over_2 (α β : ℝ) (hα : 0 < α) (hα_lt : α < π / 2) (hβ : 0 < β) (hβ_lt : β < π / 2) (h : Real.sin (α + β) = Real.sin α ^ 2 + Real.sin β ^ 2) : α + β = π / 2 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_alpha_beta_sum_pi_over_2_l309_30927


namespace NUMINAMATH_GPT_max_distance_with_optimal_tire_swapping_l309_30916

theorem max_distance_with_optimal_tire_swapping
  (front_tires_last : ℕ)
  (rear_tires_last : ℕ)
  (front_tires_last_eq : front_tires_last = 20000)
  (rear_tires_last_eq : rear_tires_last = 30000) :
  ∃ D : ℕ, D = 30000 :=
by
  sorry

end NUMINAMATH_GPT_max_distance_with_optimal_tire_swapping_l309_30916


namespace NUMINAMATH_GPT_find_triangle_sides_l309_30945

noncomputable def triangle_sides (x : ℝ) : Prop :=
  let a := x - 2
  let b := x
  let c := x + 2
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a + 2 = b ∧ b + 2 = c ∧ area = 6 ∧
  a = 2 * Real.sqrt 6 - 2 ∧
  b = 2 * Real.sqrt 6 ∧
  c = 2 * Real.sqrt 6 + 2

theorem find_triangle_sides :
  ∃ x : ℝ, triangle_sides x := by
  sorry

end NUMINAMATH_GPT_find_triangle_sides_l309_30945


namespace NUMINAMATH_GPT_behemoth_and_rita_finish_ice_cream_l309_30960

theorem behemoth_and_rita_finish_ice_cream (x y : ℝ) (h : 3 * x + 2 * y = 1) : 3 * (x + y) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_behemoth_and_rita_finish_ice_cream_l309_30960


namespace NUMINAMATH_GPT_m_eq_half_l309_30986

theorem m_eq_half (m : ℝ) (h1 : m > 0) (h2 : ∀ x, (0 < x ∧ x < m) → (x * (x - 1) < 0))
  (h3 : ∃ x, (0 < x ∧ x < 1) ∧ ¬(0 < x ∧ x < m)) : m = 1 / 2 :=
sorry

end NUMINAMATH_GPT_m_eq_half_l309_30986


namespace NUMINAMATH_GPT_solve_for_y_l309_30995

-- Define the variables and conditions
variable (y : ℝ)
variable (h_pos : y > 0)
variable (h_seq : (4 + y^2 = 2 * y^2 ∧ y^2 + 25 = 2 * y^2))

-- State the theorem
theorem solve_for_y : y = Real.sqrt 14.5 :=
by sorry

end NUMINAMATH_GPT_solve_for_y_l309_30995


namespace NUMINAMATH_GPT_complex_value_of_product_l309_30918

theorem complex_value_of_product (r : ℂ) (hr : r^7 = 1) (hr1 : r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := 
by sorry

end NUMINAMATH_GPT_complex_value_of_product_l309_30918


namespace NUMINAMATH_GPT_Jill_talking_time_total_l309_30900

-- Definition of the sequence of talking times
def talking_time : ℕ → ℕ 
| 0 => 5
| (n+1) => 2 * talking_time n

-- The statement we need to prove
theorem Jill_talking_time_total :
  (talking_time 0) + (talking_time 1) + (talking_time 2) + (talking_time 3) + (talking_time 4) = 155 :=
by
  sorry

end NUMINAMATH_GPT_Jill_talking_time_total_l309_30900
