import Mathlib

namespace keiko_speed_l582_582551

theorem keiko_speed (a b : ℝ) (s : ℝ) (h1 : 0 < b) (h2 : 0 < s) (h3 : 2 * a + 2 * π * (b + 8) / s = (2 * a + 2 * π * b) / s + 48) : s = π / 3 :=
by
  have h_lengths : 2 * a + 2 * π * (b + 8) = 2 * a + 2 * π * b + 16 * π := by
    calc
      2 * a + 2 * π * (b + 8) = 2 * a + 2 * (π * b + 8 * π) : by rw [mul_add]
      ... = 2 * a + (2 * π * b + 16 * π) : by ring
      ... = 2 * a + 2 * π * b + 16 * π : by ring
  have h_diff : 2 * a + 2 * π * (b + 8) / s = (2 * a + 2 * π * b) / s + 48 :=
    by sorry
  calc
    s = 16 * π / 48 : by sorry
    ... = π / 3 : by norm_num

end keiko_speed_l582_582551


namespace largest_convex_ngon_with_integer_tangents_l582_582791

-- Definitions of conditions and the statement
def isConvex (n : ℕ) : Prop := n ≥ 3 -- Condition 1: n is at least 3
def isConvexPolygon (n : ℕ) : Prop := isConvex n -- Condition 2: the polygon is convex
def tanInteriorAnglesAreIntegers (n : ℕ) : Prop := true -- Placeholder for Condition 3

-- Statement to prove
theorem largest_convex_ngon_with_integer_tangents : 
  ∀ n : ℕ, isConvexPolygon n → tanInteriorAnglesAreIntegers n → n ≤ 8 :=
by
  intros n h_convex h_tangents
  sorry

end largest_convex_ngon_with_integer_tangents_l582_582791


namespace minimum_rounds_for_early_winner_l582_582520

theorem minimum_rounds_for_early_winner :
  ∀ (n : ℕ) (points_per_round : ℕ) (draw_points: ℝ),
  n = 10 ∧ points_per_round = 5 ∧ draw_points = 0.5 →
  exists (r : ℕ), r = 7 :=
by
  assume n points_per_round draw_points h,
  sorry

end minimum_rounds_for_early_winner_l582_582520


namespace solve_distance_equation_l582_582610

theorem solve_distance_equation :
  (∀ (x : ℝ), |sqrt (x^2 + 8 * x + 20) - sqrt (x^2 - 2 * x + 2)| = sqrt 26) ↔ x = 6 := by
  sorry

end solve_distance_equation_l582_582610


namespace polynomial_square_l582_582098

theorem polynomial_square (x : ℝ) : x^4 + 2*x^3 - 2*x^2 - 4*x - 5 = y^2 → x = 3 ∨ x = -3 := by
  sorry

end polynomial_square_l582_582098


namespace option_a_identical_option_b_not_identical_option_c_not_identical_option_d_not_identical_identify_identical_function_l582_582666

-- Definitions of the function pairs
def f_a_x (x : ℝ) : ℝ := x^2 - 2 * x
def f_a_t (t : ℝ) : ℝ := t^2 - 2 * t 

def f_b_x (x : ℝ) : ℝ := x^0
def f_b : ℝ → ℝ := λ x, 1

def f_c_x (x : ℝ) : ℝ := Real.sqrt (x + 1)^2
def f_c : ℝ → ℝ := λ x, x + 1

def f_d_x (x : ℝ) : ℝ := Real.log x^2
def f_d : ℝ → ℝ := λ x, 2 * Real.log x

-- Theorem for proving the identity of the functions in Option A
theorem option_a_identical :
  ∀ x t, f_a_x x = f_a_t t ↔ x = t := sorry

-- Theorem to show that functions in Option B are not identical
theorem option_b_not_identical :
  ∃ x, f_b_x x ≠ f_b x := sorry

-- Theorem to show that functions in Option C are not identical
theorem option_c_not_identical :
  ∃ x, f_c_x x ≠ f_c x := sorry

-- Theorem to show that functions in Option D are not identical
theorem option_d_not_identical :
  ∃ x, f_d_x x ≠ f_d x := sorry

-- Main theorem to assert Option A is the correct choice
theorem identify_identical_function :
  (∀ x t, f_a_x x = f_a_t t ↔ x = t) ∧
  ¬ (∃ x, f_b_x x = f_b x) ∧
  ¬ (∃ x, f_c_x x = f_c x) ∧
  ¬ (∃ x, f_d_x x = f_d x) := 
begin
  split,
  { exact option_a_identical },
  split,
  { exact option_b_not_identical },
  split,
  { exact option_c_not_identical },
  { exact option_d_not_identical }
end

end option_a_identical_option_b_not_identical_option_c_not_identical_option_d_not_identical_identify_identical_function_l582_582666


namespace max_value_t_l582_582214

noncomputable theory

open Real

theorem max_value_t : ∀ (x y : ℝ), (x > 0) → (y > 0) → (let t := min (2 * x + y) (2 * y / (x^2 + 2 * y^2)) in t ≤ sqrt 2) :=
sorry

end max_value_t_l582_582214


namespace dice_probability_l582_582342

theorem dice_probability :
  let p_one_digit := 1 / 2 in
  let p_two_digit := 1 / 2 in
  let num_dice := 6 in
  let num_ways := Nat.choose 6 3 in
  (num_ways * (p_one_digit ^ 3 * p_two_digit ^ 3)) = 5 / 16 :=
by
  sorry

end dice_probability_l582_582342


namespace determine_b_l582_582630

-- Definitions and known conditions
variables (x y z b : ℝ)
variables (k : ℝ)
hypothesis h_ratio : x = 4 * k ∧ y = 5 * k ∧ z = 6 * k
hypothesis h_sum : x + y + z = 90
hypothesis h_y : y = 15 * b - 5

-- The proof statement
theorem determine_b : b = 7 / 3 :=
by
  -- Sorry for skipping the proof
  sorry

end determine_b_l582_582630


namespace measure_of_angle_B_l582_582191

theorem measure_of_angle_B (A B C : Type) [triangle : Triangle A B C] (exterior_angle_A : ∠ (exterior A) = 110) :
  ∠B = 55 ∨ ∠B = 70 ∨ ∠B = 40 :=
by
  sorry

end measure_of_angle_B_l582_582191


namespace sum_of_arithmetic_series_l582_582263

theorem sum_of_arithmetic_series (k : ℕ) : 
  let a : ℕ := k^2 - k + 1,
      d : ℕ := 1,
      n : ℕ := 2 * k in
  (n * (a + (a + (n - 1) * d)) / 2) = 2 * k^3 + k :=
by
  sorry

end sum_of_arithmetic_series_l582_582263


namespace discount_rate_is_50_l582_582681

theorem discount_rate_is_50
  (marked_price : ℝ) (selling_price : ℝ)
  (hmp : marked_price = 240) (hsp : selling_price = 120) :
  let discount := marked_price - selling_price in
  let rate_of_discount := (discount / marked_price) * 100 in
  rate_of_discount = 50 := 
by {
  sorry
}

end discount_rate_is_50_l582_582681


namespace distribute_doctors_l582_582414

theorem distribute_doctors :
  let doctors := {A, B, C, D, E, F, G}
  -- Condition: Doctors A and B are not in the same team
  let not_same_team : ∀ t1 t2 t3 : finset α, (A ∈ t1 ∧ B ∈ t1) ∨ (A ∈ t2 ∧ B ∈ t2) ∨ (A ∈ t3 ∧ B ∈ t3) → false
  -- Condition: Teams definition
  let team_structure : ∀ t1 t2 t3 : finset α, (t1 ∪ t2 ∪ t3 = doctors) ∧ (t1.card = 3 ∧ t2.card = 2 ∧ t3.card = 2)
  -- Correct Answer: 80 ways to distribute the teams
  ∃ t1 t2 t3 : finset α, (t1 ∪ t2 ∪ t3 = doctors) ∧ (t1.card = 3 ∧ t2.card = 2 ∧ t3.card = 2) ∧
  ((A ∈ t1 ∧ B ∈ t2 ∧ A ∉ t3 ∧ B ∉ t3) ∨ (A ∈ t1 ∧ B ∉ t2 ∧ B ∉ t3 ∧ A ∉ t2 ∧ A ∉ t3)) →
  nat.choose 5 2 * 3 + nat.choose 5 3 * 2 = 80 :=
by
  sorry

end distribute_doctors_l582_582414


namespace no_fractional_solutions_l582_582756

theorem no_fractional_solutions (x y : ℚ) (hx : x.denom ≠ 1) (hy : y.denom ≠ 1) :
  ¬ (∃ m n : ℤ, 13 * x + 4 * y = m ∧ 10 * x + 3 * y = n) :=
sorry

end no_fractional_solutions_l582_582756


namespace loss_per_meter_of_cloth_l582_582705

theorem loss_per_meter_of_cloth :
  ∀ (total_meters : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ),
  total_meters = 450 →
  total_selling_price = 18000 →
  cost_price_per_meter = 45 →
  let total_cost_price := total_meters * cost_price_per_meter in
  let total_loss := total_cost_price - total_selling_price in
  let loss_per_meter := total_loss / total_meters in
  loss_per_meter = 5 := by
  intros total_meters total_selling_price cost_price_per_meter
  intros htotal_meters htotal_selling_price hcost_price_per_meter
  let total_cost_price := total_meters * cost_price_per_meter
  let total_loss := total_cost_price - total_selling_price
  let loss_per_meter := total_loss / total_meters
  sorry

end loss_per_meter_of_cloth_l582_582705


namespace acute_triangle_perimeter_inequality_l582_582180

variables {A B C D E F P Q R : Type}
variables [Field A] [Field B] [Field C] [Field D] [Field E] [Field F] [Field P] [Field Q] [Field R]

open Real

-- Define a function to compute the perimeter of a triangle given the length of its sides.
noncomputable def perimeter (a b c : ℝ) : ℝ :=
a + b + c

-- Define the problem statement in Lean
theorem acute_triangle_perimeter_inequality
  (hABC : ∀ A B C : ℝ, A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (hDEF : ∀ A B C : ℝ, D = orthocenter A B C ∧ E = orthocenter B C A ∧ F = orthocenter C A B)
  (hPQR : ∀ A B C D E F : ℝ, P = foot_of_perpendicular A E F ∧ Q = foot_of_perpendicular B F D ∧ R = foot_of_perpendicular C D E) :
  perimeter A B C * perimeter P Q R ≥ (perimeter D E F) ^2 := 
sorry

end acute_triangle_perimeter_inequality_l582_582180


namespace equal_real_roots_eq_one_l582_582262

theorem equal_real_roots_eq_one (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y * y = x) ∧ (∀ x y : ℝ, x^2 - 2 * x + m = 0 ↔ (x = y) → b^2 - 4 * a * c = 0) → m = 1 := 
sorry

end equal_real_roots_eq_one_l582_582262


namespace initial_flowers_per_bunch_l582_582030

theorem initial_flowers_per_bunch (x : ℕ) (h₁: 8 * x = 72) : x = 9 :=
  by
  sorry

end initial_flowers_per_bunch_l582_582030


namespace find_equation_of_m_l582_582059

theorem find_equation_of_m :
  ∃ m : ℝ → ℝ → Prop,
    (∀ x y : ℝ, m x y ↔ 5 * x + y = 0) ∧
    ((∀ x y : ℝ, 4 * x + y = 0) → (P'' = (-2, 3)) ∧ (∃ P : ℝ × ℝ, P = (3, -2)) ∧ (∀ x y m, (P' = reflect_about (4 * x + y = 0) P) ∧ (P'' = reflect_about m P')) ∧ (∃ℓ m, lines_intersect_at_origin ℓ m ∧ perpenducular ℓ m)) :=
by sorry

end find_equation_of_m_l582_582059


namespace base_for_ACAC_form_l582_582318

theorem base_for_ACAC_form (b : ℕ) :
  (b^3 ≤ 777 ∧ 777 < b^4) ∧ 
  (let digits := [((777 % b^4) / b^3), ((777 % b^3) / b^2), ((777 % b^2) / b), (777 % b)] 
   in digits.length = 4 ∧ digits.nth 0 = digits.nth 2 ∧ digits.nth 1 = digits.nth 3 ∧ digits.nth 0 ≠ digits.nth 1) 
  → b = 9 :=
sorry

end base_for_ACAC_form_l582_582318


namespace OP_solution_l582_582794

noncomputable def find_OP (a b c d : ℝ) : ℝ :=
  (2 * a - c + 2 * b - 2 * d + sqrt ((2 * a - c + 2 * b - 2 * d) ^ 2 - 8 * (a * c - b * d)) ) / 2

theorem OP_solution (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d)
  (h₅ : b ≤ find_OP a b c d) (h₆ : find_OP a b c d ≤ c)
  (h₇ : ∣2 * a - find_OP a b c d∣ / ∣find_OP a b c d - 2 * d∣ = 2 * ∣b - find_OP a b c d∣ / ∣find_OP a b c d - c∣) :
  find_OP a b c d = (2 * a - c + 2 * b - 2 * d + sqrt ((2 * a - c + 2 * b - 2 * d) ^ 2 - 8 * (a * c - b * d)) ) / 2 :=
sorry

end OP_solution_l582_582794


namespace min_value_of_f_range_of_m_l582_582140

noncomputable def f (x a : ℝ) := |x - a| + |x - 3 * a|

theorem min_value_of_f (a : ℝ) : (∀ x, f x a ≥ 2) → (∀ x, f x a = 2 → a = 1 ∨ a = -1) :=
begin
  intros h h₂,
  sorry
end

theorem range_of_m (m : ℝ) : (∀ x, ∃ a ∈ Icc (-1:ℝ) 1, m^2 - |m| < f x a) → -2 < m ∧ m < 2 :=
begin
  intros h,
  sorry
end

end min_value_of_f_range_of_m_l582_582140


namespace area_of_triangle_ADC_l582_582899

theorem area_of_triangle_ADC (BD DC : ℝ) (A : ℝ) (h : ℝ)
  (h_ratio : BD / DC = 4 / 3)
  (area_ABD : 1 / 2 * BD * h = 24) :
  1 / 2 * DC * h = 18 :=
by
  have h1 : 1 / 2 * BD * h / (1 / 2 * DC * h) = BD / DC := by
    sorry
  rw [h1, h_ratio, area_ABD] at h1
  sorry

end area_of_triangle_ADC_l582_582899


namespace purple_valley_skirts_l582_582243

theorem purple_valley_skirts (azure_valley_skirts : ℕ) (h1 : azure_valley_skirts = 60) :
    let seafoam_valley_skirts := (2 / 3 : ℚ) * azure_valley_skirts in
    let purple_valley_skirts := (1 / 4 : ℚ) * seafoam_valley_skirts in
    purple_valley_skirts = 10 :=
by
  let seafoam_valley_skirts := (2 / 3 : ℚ) * azure_valley_skirts
  let purple_valley_skirts := (1 / 4 : ℚ) * seafoam_valley_skirts
  have h2 : seafoam_valley_skirts = (2 / 3 : ℚ) * 60 := by
    rw [h1]
  have h3 : purple_valley_skirts = (1 / 4 : ℚ) * ((2 / 3 : ℚ) * 60) := by
    rw [h2]
  have h4 : purple_valley_skirts = (1 / 4 : ℚ) * 40 := by
    norm_num [h3]
  have h5 : purple_valley_skirts = 10 := by
    norm_num [h4]
  exact h5

end purple_valley_skirts_l582_582243


namespace term_2023_of_sequence_is_370_l582_582604

def sum_of_cubes_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.map (λ c => (c.toNat - '0'.toNat)^3)).sum

def sequence_term (n : ℕ) : ℕ :=
  Nat.iterate (λ x => sum_of_cubes_of_digits x) n

theorem term_2023_of_sequence_is_370 :
  sequence_term 2023^2023 = 370 :=
sorry

end term_2023_of_sequence_is_370_l582_582604


namespace no_fractional_xy_l582_582764

theorem no_fractional_xy (x y : ℚ) (m n : ℤ) (h1 : 13 * x + 4 * y = m) (h2 : 10 * x + 3 * y = n) : ¬ (¬(x ∈ ℤ) ∨ ¬(y ∈ ℤ)) :=
sorry

end no_fractional_xy_l582_582764


namespace loisa_saves_70_l582_582228

def tablet_cash_price : ℕ := 450
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 40
def next_4_months_payment : ℕ := 35
def last_4_months_payment : ℕ := 30
def total_installment_payment : ℕ := down_payment + (4 * first_4_months_payment) + (4 * next_4_months_payment) + (4 * last_4_months_payment)
def savings : ℕ := total_installment_payment - tablet_cash_price

theorem loisa_saves_70 : savings = 70 := by
  sorry

end loisa_saves_70_l582_582228


namespace g_neither_even_nor_odd_l582_582537

def g (x : ℝ) : ℝ := ⌊x^3⌋ + 1/3

theorem g_neither_even_nor_odd : ¬even_function g ∧ ¬odd_function g := by
  sorry

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - (f x)

end g_neither_even_nor_odd_l582_582537


namespace tan_alpha_plus_pi_div4_sin2alpha_over_expr_l582_582800

variables (α : ℝ) (h : Real.tan α = 3)

-- Problem 1
theorem tan_alpha_plus_pi_div4 : Real.tan (α + π / 4) = -2 :=
by
  sorry

-- Problem 2
theorem sin2alpha_over_expr : (Real.sin (2 * α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1) = 3 / 5 :=
by
  sorry

end tan_alpha_plus_pi_div4_sin2alpha_over_expr_l582_582800


namespace f_2009_eq_11_l582_582160

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

noncomputable def f (n : ℕ) : ℕ := 
  sum_of_digits (n^2 + 1)

noncomputable def f_iterate (k : ℕ) (n : ℕ) : ℕ :=
  nat.rec_on k n (λ k' fk', f fk')

theorem f_2009_eq_11 : f_iterate 2009 9 = 11 :=
  sorry

end f_2009_eq_11_l582_582160


namespace females_count_l582_582368

-- Defining variables and constants
variables (P M F : ℕ)
-- The condition given the total population
def town_population := P = 600
-- The condition given the proportion of males
def proportion_of_males := M = P / 3
-- The condition determining the number of females
def number_of_females := F = P - M

-- The theorem stating the number of females is 400
theorem females_count (P M F : ℕ) (h1 : town_population P)
  (h2 : proportion_of_males P M) 
  (h3 : number_of_females P M F) : 
  F = 400 := 
sorry

end females_count_l582_582368


namespace gh_of_2_l582_582505

def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem gh_of_2 :
  g (h 2) = 3269 :=
by
  sorry

end gh_of_2_l582_582505


namespace calculate_total_money_l582_582950

theorem calculate_total_money (n100 n50 n10 : ℕ) 
  (h1 : n100 = 2) (h2 : n50 = 5) (h3 : n10 = 10) : 
  (n100 * 100 + n50 * 50 + n10 * 10 = 550) :=
by
  sorry

end calculate_total_money_l582_582950


namespace triangle_CM_eq_l582_582111

open Classical

noncomputable def triangle_side_CM (α β a : Real) : Real :=
  a * (Real.sin α) / (Real.sin β)

theorem triangle_CM_eq (α β a : Real) (ABC : Triangle) (D : Point) (M : Point) :
  (ABC.angles.A = α) → 
  (ABC.angles.B = β) → 
  (D ∈ ABC.side.AB) → 
  (M ∈ ABC.side.AC) → 
  (AngleBisector ABC.angle.C D → 
  (Parallel (Line DM) (Line BC)) → 
  (M.distance A = a) → 
  (C.distance M = triangle_side_CM α β a) :=
sorry

end triangle_CM_eq_l582_582111


namespace purple_valley_skirts_l582_582247

def AzureValley : ℕ := 60

def SeafoamValley (A : ℕ) : ℕ := (2 * A) / 3

def PurpleValley (S : ℕ) : ℕ := S / 4

theorem purple_valley_skirts :
  PurpleValley (SeafoamValley AzureValley) = 10 :=
by
  sorry

end purple_valley_skirts_l582_582247


namespace commodity_price_difference_l582_582181

theorem commodity_price_difference (r : ℝ) (t : ℕ) :
  let P_X (t : ℕ) := 4.20 * (1 + (2*r + 10)/100)^(t - 2001)
  let P_Y (t : ℕ) := 4.40 * (1 + (r + 15)/100)^(t - 2001)
  P_X t = P_Y t + 0.90  ->
  ∃ t : ℕ, true :=
by
  sorry

end commodity_price_difference_l582_582181


namespace sum_of_digits_9ab_l582_582534

theorem sum_of_digits_9ab : 
  let a := 4 * (10^1995 - 1) / 9
  let b := 7 * (10^1995 - 1) / 9
  (sum_of_digits (9 * a * b)) = 17965 :=
sorry

end sum_of_digits_9ab_l582_582534


namespace sum_first_twelve_multiples_of_17_l582_582311

theorem sum_first_twelve_multiples_of_17 : ∑ k in finset.range (12 + 1), 17 * k = 1326 :=
by
  -- proof steps would go here, but for the task, we add 'sorry' to skip the proof
  sorry

end sum_first_twelve_multiples_of_17_l582_582311


namespace PU_is_one_l582_582297

noncomputable def bisector_theorem_proof : Prop :=
  let P := (0, 0)
  let Q := (13, 0)
  let R := (x:R, y:R) -- Additional conditions to define R would be needed
  let PQ := dist P Q
  let QR := dist Q R
  let PR := dist P R
  let S := (a, b) -- Coordinates for S derived from bisector properties
  let T := (c, d) -- Coordinates for T from circumcircle conditions
  let U := (e, f) -- Coordinates for U from perpendicular bisector

  in PQ = 13 ∧ QR = 26 ∧ PR = 24 ∧ PU = 1

theorem PU_is_one : bisector_theorem_proof := sorry

end PU_is_one_l582_582297


namespace cut_alloy_weight_l582_582043

noncomputable def cut_weight (x y : ℝ) (h_diff : x ≠ y) : ℝ :=
  let z := (21 : ℝ) / 10 in z

theorem cut_alloy_weight {x y : ℝ} (h_diff : x ≠ y) :
  cut_weight x y h_diff = (21 : ℝ) / 10 :=
by
  unfold cut_weight
  sorry

end cut_alloy_weight_l582_582043


namespace combined_instruments_correct_l582_582728

-- Definitions of initial conditions
def Charlie_flutes : Nat := 1
def Charlie_horns : Nat := 2
def Charlie_harps : Nat := 1
def Carli_flutes : Nat := 2 * Charlie_flutes
def Carli_horns : Nat := Charlie_horns / 2
def Carli_harps : Nat := 0

-- Calculation of total instruments
def Charlie_total_instruments : Nat := Charlie_flutes + Charlie_horns + Charlie_harps
def Carli_total_instruments : Nat := Carli_flutes + Carli_horns + Carli_harps
def combined_total_instruments : Nat := Charlie_total_instruments + Carli_total_instruments

-- Theorem statement
theorem combined_instruments_correct : combined_total_instruments = 7 := 
by
  sorry

end combined_instruments_correct_l582_582728


namespace range_of_alpha_minus_beta_l582_582129

theorem range_of_alpha_minus_beta (α β : ℝ) (h1 : -180 < α) (h2 : α < β) (h3 : β < 180) :
  -360 < α - β ∧ α - β < 0 :=
by
  sorry

end range_of_alpha_minus_beta_l582_582129


namespace maximize_rectangle_area_l582_582028

-- Define the given right-angled triangle
def right_triangle_valid (A B C : Point) : Prop :=
  B.x = 1 ∧ B.y = 0 ∧ C.x = 1 ∧ C.y = Real.sqrt 3 ∧ dist A B = 1 ∧ dist B C = 2

-- Define when a rectangle is inscribed in the right-angled triangle
def rectangle_inscribed (A F D E : Point) : Prop :=
  F.x = 0 ∧ F.y < 1 ∧ D.x < 1 ∧ D.y = 0 ∧ E.x < 1 ∧ E.y < Real.sqrt 3 

-- Define the hypothesis and the proof goal
theorem maximize_rectangle_area (A B C F D E : Point) :
  right_triangle_valid A B C →
  rectangle_inscribed A F D E →
  F.y = 1 - F.y :=
sorry

end maximize_rectangle_area_l582_582028


namespace missing_number_unique_l582_582420

theorem missing_number_unique (x : ℤ) 
  (h : |9 - x * (3 - 12)| - |5 - 11| = 75) : 
  x = 8 :=
sorry

end missing_number_unique_l582_582420


namespace sum_of_coefficients_eq_neg_six_l582_582256

theorem sum_of_coefficients_eq_neg_six
  (a b c : ℤ)
  (h_gcd : Polynomial.gcd (Polynomial.C a * Polynomial.X + Polynomial.C b) (Polynomial.C b * Polynomial.X + Polynomial.C c) = Polynomial.X + 1)
  (h_lcm : Polynomial.lcm (Polynomial.C a * Polynomial.X + Polynomial.C b) (Polynomial.C b * Polynomial.X + Polynomial.C c) = Polynomial.X^3 - 4 * Polynomial.X^2 + Polynomial.X + 6) :
  a + b + c = -6 :=
sorry

end sum_of_coefficients_eq_neg_six_l582_582256


namespace arithmetic_sequence_solution_l582_582813

variable {a : ℕ → ℤ}  -- assuming our sequence is integer-valued for simplicity

-- a is an arithmetic sequence if there exists a common difference d such that 
-- ∀ n, a_{n+1} = a_n + d
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- sum of the terms from a₁ to a₁₀₁₇ is equal to zero
def sum_condition (a : ℕ → ℤ) : Prop :=
  (Finset.range 2017).sum a = 0

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (h_arith : is_arithmetic_sequence a) (h_sum : sum_condition a) :
  a 3 + a 2013 = 0 :=
sorry

end arithmetic_sequence_solution_l582_582813


namespace exists_convex_quadrilateral_with_prime_side_lengths_l582_582943

structure Point :=
  (x : ℤ)
  (y : ℤ)

def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

lemma primes_are_distinct (a b c d : ℕ) (ha : nat.prime a) (hb : nat.prime b) (hc : nat.prime c) (hd : nat.prime d) (h : list.nodup [a, b, c, d]) : true := true.intro

theorem exists_convex_quadrilateral_with_prime_side_lengths :
  ∃ (A B C D : Point),
  let AB := dist A B in
  let BC := dist B C in
  let CD := dist C D in
  let DA := dist D A in
  (nat.prime (AB.to_nat) ∧ nat.prime (BC.to_nat) ∧ nat.prime (CD.to_nat) ∧ nat.prime (DA.to_nat)) ∧
  ([AB.to_nat, BC.to_nat, CD.to_nat, DA.to_nat].nodup) ∧
  ∃ θ : ℝ, θ < 180 := 
by {
  let A := Point.mk 0 0,
  let B := Point.mk 5 0,
  let C := Point.mk 5 17,
  let D := Point.mk 0 29,
  use [A, B, C, D],
  split,
  { repeat {split};
    norm_num;
    {apply nat.prime_of_nat 5, exact dec_trivial}
    {apply nat.prime_of_fermat 13, exact dec_trivial}
    {apply nat.prime_of_fermat 17, exact dec_trivial}
    {apply nat.prime_of_fermat 29, exact dec_trivial} },
  { norm_num,
    split,
    { apply list.nodup_cons,
      exact dec_trivial },
    { intro θ,
      by_cases hθ: θ < 180, 
      { right, exact hθ },
      { left, have := dec_trivial, contradiction }
    } 
  }
}

end exists_convex_quadrilateral_with_prime_side_lengths_l582_582943


namespace num_distinct_triangles_l582_582862

theorem num_distinct_triangles : 
  let points := [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)] in
  let is_collinear (p1 p2 p3 : (ℕ × ℕ)) := 
    (p2.1 - p1.1) * (p3.2 - p1.2) = (p3.1 - p1.1) * (p2.2 - p1.2) in
  (count (λ s : finset (ℕ × ℕ), s.card = 3 ∧ ¬is_collinear s) (finset.powerset points)) = 18 :=
by sorry

end num_distinct_triangles_l582_582862


namespace list_satisfaction_l582_582075

theorem list_satisfaction (x : ℕ → ℝ) (x_condition : ∀ i j, 1 ≤ i → i ≤ j → j ≤ 2020 → x i ≤ x j)
  (x_last_condition : x 2020 ≤ x 1 + 1)
  (perm : ∃ y : ℕ → ℝ, (∀ i, 1 ≤ i → i ≤ 2020 → ∃ j, 1 ≤ j → j ≤ 2020 → y j = x i) ∧ 
  (∑ i in (finset.range 2020), ((x i + 1) * (y i + 1))^2) = 8 * (∑ i in (finset.range 2020), (x i)^3)) :
  ∀ k, 1 ≤ k → k ≤ 2020 → x k = if k ≤ 1010 then 0 else if k = 1011 then 1 else x 1 + 1 :=
by
  sorry

end list_satisfaction_l582_582075


namespace find_m_l582_582480

noncomputable def f (x : ℝ) (m : ℝ) := x^3 - (3 / 2) * x^2 + m

theorem find_m (m : ℝ) :
  (∃ x ∈ Ioo 0 2, (deriv (f x m) x = 0) ∧ f x m = 3 / 2) → m = 2 :=
by
  sorry

end find_m_l582_582480


namespace alpha_in_first_quadrant_l582_582109

-- Definition of the point P
def P := (real.sqrt 3 / 2, 1 / 2)

-- Definition of quadrant check
def inFirstQuadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

-- Theorem statement to prove the quadrant of angle α given the point P
theorem alpha_in_first_quadrant (x y : ℝ) (h : (x, y) = P) : inFirstQuadrant x y :=
by
  -- We provide a 'sorry' to skip the proof as requested
  sorry

end alpha_in_first_quadrant_l582_582109


namespace unique_differences_count_l582_582100

def positive_nat_range := {n // 1 ≤ n ∧ n ≤ 4}
def complex_set : Set ℂ := {z : ℂ | ∃ x y : positive_nat_range, x < y ∧ z = (x.val : ℂ) + (y.val : ℂ) * complex.i}

theorem unique_differences_count :
  (∃ S : Set ℂ, S = complex_set ∧ (S.pairs (≠)).map (λ ⟨a, b⟩, a - b)).toFinset.card = 9 :=
sorry

end unique_differences_count_l582_582100


namespace find_angle_C_find_area_of_triangle_l582_582281

-- Define the given conditions
variables {A B C : ℝ} {a b c : ℝ}
-- | Conditions for part I
axiom cos_angle_condition : cos (A - C) + cos B = 1
axiom side_ratio_condition : a = 2 * c

-- Definition for angle C
noncomputable def angle_C : ℝ := C = 30 * π / 180

-- Statement for part I
theorem find_angle_C (h1 : cos_angle_condition)
                     (h2 : side_ratio_condition) : angle_C :=
sorry

-- Conditions for part II
axiom side_a_condition : a = sqrt 2

-- Definition for area of the triangle
noncomputable def area_of_triangle : ℝ :=
  1/2 * b * c * sin (A)

-- Statement for part II
theorem find_area_of_triangle (h1 : cos_angle_condition)
                              (h2 : side_ratio_condition)
                              (h3 : side_a_condition) : 
  area_of_triangle = sqrt 3 / 4 :=
sorry

end find_angle_C_find_area_of_triangle_l582_582281


namespace number_of_possible_values_for_n_l582_582616

/-- 
The set {1, 4, n} has the property that when any two distinct elements are chosen 
and 2112 is added to their product, the result is a perfect square. 
Prove that the number of possible positive integer values for n is 7.
-/

theorem number_of_possible_values_for_n : 
  ∃ (N : ℕ), 
  (∀ n : ℕ, (∃ x : ℕ, n + 2112 = x^2) ∧ (∃ y : ℕ, 4 * n + 2112 = y^2) → n > 0 →  n ∈ {1, 4, N}) →  N = 7 :=
sorry

end number_of_possible_values_for_n_l582_582616


namespace find_other_cosine_roots_l582_582538

theorem find_other_cosine_roots (t : ℝ) :
  t = cos (6 * (π / 180)) →
  (∃ x, x = cos (66 * (π / 180))) ∧
  (∃ x, x = cos (78 * (π / 180))) ∧
  (∃ x, x = cos (138 * (π / 180))) ∧
  (∃ x, x = cos (150 * (π / 180))) :=
begin
  sorry
end

end find_other_cosine_roots_l582_582538


namespace flagpole_arrangements_remainder_l582_582292

theorem flagpole_arrangements_remainder (
  n_blue : ℕ,
  n_green : ℕ,
  total_flags : ℕ,
  modulus : ℕ
) (h1 : n_blue = 12)
  (h2 : n_green = 9)
  (h3 : total_flags = 21)
  (h4 : modulus = 1000)
  (h5 : n_blue + n_green = total_flags)
  (h6 : 21 ≤ total_flags)
  (h7 : ∀ (blue_flags green_flags : ℕ), (blue_flags ≥ 1 ∧ green_flags ≥ 1) ∧ (∀ (i j : ℕ), (i ≤ n_blue ∧ j ≤ n_green) → ¬(i = j + 1)))
  : ((M n_blue n_green total_flags) % modulus) = 596 := sorry

end flagpole_arrangements_remainder_l582_582292


namespace combined_score_is_210_l582_582171

theorem combined_score_is_210 :
  ∀ (total_questions : ℕ) (marks_per_question : ℕ) (jose_wrong : ℕ) 
    (meghan_less_than_jose : ℕ) (jose_more_than_alisson : ℕ) (jose_total : ℕ),
  total_questions = 50 →
  marks_per_question = 2 →
  jose_wrong = 5 →
  meghan_less_than_jose = 20 →
  jose_more_than_alisson = 40 →
  jose_total = total_questions * marks_per_question - (jose_wrong * marks_per_question) →
  (jose_total - meghan_less_than_jose) + jose_total + (jose_total - jose_more_than_alisson) = 210 :=
by
  intros total_questions marks_per_question jose_wrong meghan_less_than_jose jose_more_than_alisson jose_total
  intros h1 h2 h3 h4 h5 h6
  sorry

end combined_score_is_210_l582_582171


namespace find_x_l582_582498

theorem find_x :
  (10 ^ (Real.logb 10 (9 - 3)) = 7 * x + 4) → x = 2 / 7 :=
by
  sorry

end find_x_l582_582498


namespace surface_area_of_sphere_l582_582821

open Real

noncomputable def O_radius (S A B C : Point3) : ℝ :=
  let r := 1
  r

theorem surface_area_of_sphere
  (S A B C : Point3) -- Points on the surface of the sphere O
  (h1 : dist S A = 1)
  (h2 : S.x = A.x ∧ S.z = A.z) -- SA ⊥ plane ABC
  (h3 : dist A B = 1)
  (h4 : dist B C = sqrt 2)
  (h5 : dist OA OB = 1) -- OA is radius 
  (h6 : dist OA OC = 1)
  (h7 : dist OS OB = 1) :
  4 * Real.pi * (O_radius S A B C)^2 = 4 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l582_582821


namespace number_of_seats_in_classroom_l582_582389

theorem number_of_seats_in_classroom 
    (seats_per_row_condition : 7 + 13 = 19) 
    (rows_condition : 8 + 14 = 21) : 
    19 * 21 = 399 := 
by 
    sorry

end number_of_seats_in_classroom_l582_582389


namespace arrangement_methods_count_l582_582026

open Nat

/-- Theorem stating the number of distinct ways to arrange 8 identical square pieces of 
    paper into a multi-layer shape with at least two layers, such that each square 
    on the upper layer has two vertices at the midpoints of one side of a square 
    on the layer below. -/
theorem arrangement_methods_count : 
  ∃ (count : ℕ), count = 17 ∧
  ∀ n, n = 8 →
    (∃ (arrangement : list (list ℕ)), 
      valid_arrangement n arrangement → 
      length (distinct_placements arrangement) = count) :=
begin
  sorry
end

/-- Predicate that checks if the given arrangement is valid for n squares. -/
def valid_arrangement (n : ℕ) (arrangement : list (list ℕ)) : Prop :=
  -- Placeholder for the detailed definition according to conditions
  sorry

/-- Function to count distinct placements of a given valid arrangement. -/
def distinct_placements (arrangement : list (list ℕ)) : list (list ℕ) :=
  -- Placeholder for actual distinct placement counting algorithm
  sorry

end arrangement_methods_count_l582_582026


namespace eval_expression_l582_582067

theorem eval_expression : 
  (⌈(17 / 7) - ⌈27 / 17⌉⌉ / ⌈(27 / 7) + ⌈(7 * 17) / 27⌉⌉) = 1 / 9 :=
by
  have h1 : ⌈27 / 17⌉ = 2 := by sorry
  have h2 : ⌈(7 * 17) / 27⌉ = 5 := by sorry
  have h3 : ⌈3 / 7⌉ = 1 := by sorry
  have h4 : ⌈62 / 7⌉ = 9 := by sorry
  sorry

end eval_expression_l582_582067


namespace polycarp_error_l582_582574

def three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem polycarp_error (a b n : ℕ) (ha : three_digit a) (hb : three_digit b)
  (h : 10000 * a + b = n * a * b) : n = 73 :=
by
  sorry

end polycarp_error_l582_582574


namespace side_length_square_l582_582624

-- Define the length and width of the rectangle
def length_rect := 10 -- cm
def width_rect := 8 -- cm

-- Define the perimeter of the rectangle
def perimeter_rect := 2 * (length_rect + width_rect)

-- Define the perimeter of the square
def perimeter_square (s : ℕ) := 4 * s

-- The theorem to prove
theorem side_length_square : ∃ s : ℕ, perimeter_rect = perimeter_square s ∧ s = 9 :=
by
  sorry

end side_length_square_l582_582624


namespace only_n_equal_five_satisfies_condition_l582_582777

def digit_sum (n : ℕ) : ℕ :=
  n.digits.foldl (λ sum d, sum + d) 0

def satisfies_condition (n : ℕ) : Prop :=
  digit_sum (2^n) = 5

theorem only_n_equal_five_satisfies_condition :
  ∀ n : ℕ, satisfies_condition n ↔ n = 5 :=
by
  sorry

end only_n_equal_five_satisfies_condition_l582_582777


namespace intersection_product_l582_582887

def point (α : Type) := (α × α) -- Defined a point in a Cartesian plane
def inclination_angle := Real.pi / 6
def P : point ℝ := (-1, 2)
def polar_curve_radius := 3

def line_parametric_eq (t : ℝ) : point ℝ :=
  let slope_x := Real.cos inclination_angle
  let slope_y := Real.sin inclination_angle
  (-1 + slope_x * t, 2 + slope_y * t)

def cartesian_curve_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

lemma product_of_distances (t₁ t₂ : ℝ) (h : t₁ * t₂ = -4 ) : |t₁ * t₂| = 4 :=
by { rw abs_eq, exact abs_of_nonneg h, linarith }

theorem intersection_product (M N : point ℝ) (l_eq : ∃ t : ℝ, M = line_parametric_eq t)
  (n_eq : ∃ t : ℝ, N = line_parametric_eq t) : |P.1 * P.2| = 4 :=
begin
  sorry -- proof
end

end intersection_product_l582_582887


namespace largest_possible_n_l582_582737

def triangle_property (s : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), {a, b, c} ⊆ s → c ≤ a + b

def largest_n_with_triangle_property (m : ℕ) : Prop :=
  (∀ s : Finset ℕ, s.card = 10 → triangle_property s) →
  m = 363

theorem largest_possible_n :
  ∃ n, largest_n_with_triangle_property n :=
begin
  use 363,
  intros s h,
  -- The proof would go here
  sorry
end

end largest_possible_n_l582_582737


namespace num_arrangements_no_consecutive_ABC_l582_582894

-- Define the conditions and correct answer.
def numArrangements : ℕ :=
  10! - (7! * 4!)

theorem num_arrangements_no_consecutive_ABC(D : ℕ) :
  (10! - (7! * 4!)) = 3507840 := 
sorry

end num_arrangements_no_consecutive_ABC_l582_582894


namespace problem_part1_problem_part2_l582_582921

variable {S a : ℕ → ℝ}

-- Conditions
def condition_1 (n : ℕ) : Prop := S n = ∑ i in Finset.range n, a i
def condition_2 : Prop := a 1 = -1
def condition_3 (n : ℕ) : Prop := a (n+1) = S n * S (n+1)

-- Proof statements
theorem problem_part1 :
  (∀ n : ℕ, condition_1 n) →
  condition_2 →
  (∀ n : ℕ, n > 0 → condition_3 n) →
  (∀ n : ℕ, 1 / S (n+1) - 1 / S n = -1) ∧ (S n = - (1 / n)) :=
begin
  intros h1 h2 h3,
  sorry
end

theorem problem_part2 :
  (∀ n : ℕ, condition_1 n) →
  condition_2 →
  (∀ n : ℕ, n > 0 → condition_3 n) →
  (∑ i in Finset.range n, 1 / (a (i+2)) = n / (n+1)) :=
begin
  intros h1 h2 h3,
  sorry
end

end problem_part1_problem_part2_l582_582921


namespace number_of_subsets_correct_l582_582487

-- Definitions based on the given conditions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 2, 4}

-- Complement of B in U
def complement_U_B : Set ℕ := U \ B

-- Union of complement_U_B and A
def union_complement_U_B_A : Set ℕ := complement_U_B ∪ A

theorem number_of_subsets_correct :
  (union_complement_U_B_A = {1, 3, 5}) → (union_complement_U_B_A.size = 3) → (2^union_complement_U_B_A.size = 8) :=
by
  intros h_union h_size
  rw h_union at h_size
  rw set.size_union_complement_U_B_A
  sorry

end number_of_subsets_correct_l582_582487


namespace angle_quadrant_half_l582_582499

theorem angle_quadrant_half (k : ℤ) (α : ℝ) (h : α ∈ Ioo (2*k*Real.pi + 3*Real.pi/2) (2*k*Real.pi + 2*Real.pi)) : 
  (∃ n : ℤ, (α / 2) ∈ Ioo (n*Real.pi + 3*Real.pi/4) (n*Real.pi + Real.pi)) :=
sorry

end angle_quadrant_half_l582_582499


namespace mode_is_14_l582_582984

def ages : List ℕ := [12, 13, 13, 13, 14, 14, 14, 14, 15, 15, 16, 16]

theorem mode_is_14 : Multiset.mode (Multiset.ofList ages) = 14 := by 
  sorry

end mode_is_14_l582_582984


namespace mean_difference_l582_582119

variable (a1 a2 a3 a4 a5 a6 A : ℝ)

-- Arithmetic mean of six numbers is A
axiom mean_six_numbers : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

-- Arithmetic mean of the first four numbers is A + 10
axiom mean_first_four : (a1 + a2 + a3 + a4) / 4 = A + 10

-- Arithmetic mean of the last four numbers is A - 7
axiom mean_last_four : (a3 + a4 + a5 + a6) / 4 = A - 7

-- Prove the arithmetic mean of the first, second, fifth, and sixth numbers differs from A by 3
theorem mean_difference :
  (a1 + a2 + a5 + a6) / 4 = A - 3 := 
sorry

end mean_difference_l582_582119


namespace max_value_of_f_l582_582078

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x)

theorem max_value_of_f :
  ∃ x : ℝ, ∃ k : ℤ, f x = 3 ∧ x = k * Real.pi :=
by
  -- The proof is omitted
  sorry

end max_value_of_f_l582_582078


namespace total_cost_is_96_l582_582912

noncomputable def hair_updo_cost : ℕ := 50
noncomputable def manicure_cost : ℕ := 30
noncomputable def tip_rate : ℚ := 0.20

def total_cost_with_tip (hair_cost manicure_cost : ℕ) (tip_rate : ℚ) : ℚ :=
  let hair_tip := hair_cost * tip_rate
  let manicure_tip := manicure_cost * tip_rate
  let total_tips := hair_tip + manicure_tip
  let total_before_tips := (hair_cost : ℚ) + (manicure_cost : ℚ)
  total_before_tips + total_tips

theorem total_cost_is_96 :
  total_cost_with_tip hair_updo_cost manicure_cost tip_rate = 96 := by
  sorry

end total_cost_is_96_l582_582912


namespace rhombus_area_l582_582327

structure Point :=
(x : ℝ)
(y : ℝ)

def vertex1 : Point := {x := 0, y := 3.5}
def vertex2 : Point := {x := 6, y := 0}
def vertex3 : Point := {x := 0, y := -3.5}
def vertex4 : Point := {x := -6, y := 0}

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem rhombus_area : 
  let d1 := distance vertex1 vertex3 in
  let d2 := distance vertex2 vertex4 in
  (d1 * d2) / 2 = 42 :=
by
  sorry

end rhombus_area_l582_582327


namespace floor_value_correct_l582_582418

def calc_floor_value : ℤ :=
  let a := (15 : ℚ) / 8
  let b := a^2
  let c := (225 : ℚ) / 64
  let d := 4
  let e := (19 : ℚ) / 5
  let f := d + e
  ⌊f⌋

theorem floor_value_correct : calc_floor_value = 7 := by
  sorry

end floor_value_correct_l582_582418


namespace minimum_students_excellent_on_three_exams_l582_582965

variable (students : ℕ) (M1 M2 M3 M12 M13 M23 M123 : ℕ)

-- Given conditions
def condition1 : Prop :=
  M1 + M12 + M13 + M123 = 231

def condition2 : Prop :=
  M2 + M12 + M23 + M123 = 213

def condition3 : Prop :=
  M3 + M23 + M13 + M123 = 183

-- Problem statement
theorem minimum_students_excellent_on_three_exams (students : ℕ) (M1 M2 M3 M12 M13 M23 M123 : ℕ) :
  condition1 students M1 M2 M3 M12 M13 M23 M123 →
  condition2 students M1 M2 M3 M12 M13 M23 M123 →
  condition3 students M1 M2 M3 M12 M13 M23 M123 →
  students = 300 →
  ∃ M123, M123 ≥ 27 :=
by
  intros h1 h2 h3 h_total
  sorry

end minimum_students_excellent_on_three_exams_l582_582965


namespace john_tips_problem_l582_582910

theorem john_tips_problem
  (A M : ℝ)
  (H1 : ∀ (A : ℝ), M * A = 0.5 * (6 * A + M * A)) :
  M = 6 := 
by
  sorry

end john_tips_problem_l582_582910


namespace range_of_g_l582_582027

def g (x : ℝ) : ℝ := (Real.sin x) ^ 4 + (Real.cos x) ^ 4

theorem range_of_g : set.range g = set.Icc (1/2 : ℝ) 1 :=
by
  sorry

end range_of_g_l582_582027


namespace symmetric_point_l582_582988

theorem symmetric_point (P Q : ℝ × ℝ)
  (l : ℝ → ℝ)
  (P_coords : P = (-1, 2))
  (l_eq : ∀ x, l x = x - 1) :
  Q = (3, -2) :=
by
  sorry

end symmetric_point_l582_582988


namespace int_div_condition_l582_582166

theorem int_div_condition (n : ℕ) (hn₁ : ∃ m : ℤ, 2^n - 2 = m * n) :
  ∃ k : ℤ, 2^(2^n - 1) - 2 = k * (2^n - 1) :=
by sorry

end int_div_condition_l582_582166


namespace smallest_percentage_owns_90_percent_l582_582907

-- Given data
variable {P₁ M₁ P₂ M₂ : ℝ}
axiom H1 : P₁ = 0.20
axiom H2 : M₁ ≥ 0.80

-- To prove
theorem smallest_percentage_owns_90_percent :
  ∃ P, P = 0.60 ∧ (P₁ + P₂ = P) ∧ (M₁ + M₂ ≥ 0.90) :=
by
  -- Definitions from conditions
  let P := 0.60
  have H3 : P₂ = P - P₁ := by sorry
  have H4 : M₂ = 1 - M₁ := by sorry
  have H5 : M₂ = 0.20 := by linarith [H2]
  have H6: P₂ * 0.25 = M₂ := by sorry
  have H7: P₂ = 0.80 := by sorry
  have H8: P = P₁ + P₂ := by linarith [H1, H7]
  exact ⟨P, by norm_num, H8, by linarith [H2, H5]⟩

end smallest_percentage_owns_90_percent_l582_582907


namespace free_time_left_after_cleaning_l582_582153

-- Define the time it takes for each task
def vacuuming_time : ℤ := 45
def dusting_time : ℤ := 60
def mopping_time : ℤ := 30
def brushing_time_per_cat : ℤ := 5
def number_of_cats : ℤ := 3
def total_free_time_in_minutes : ℤ := 3 * 60 -- 3 hours converted to minutes

-- Define the total cleaning time
def total_cleaning_time : ℤ := vacuuming_time + dusting_time + mopping_time + (brushing_time_per_cat * number_of_cats)

-- Prove that the free time left after cleaning is 30 minutes
theorem free_time_left_after_cleaning : (total_free_time_in_minutes - total_cleaning_time) = 30 :=
by
  sorry

end free_time_left_after_cleaning_l582_582153


namespace polynomial_bound_l582_582810

variable {R : Type*} [CommRing R] [IsDomain R]

def polynomial_bounded_numbers (f : Polynomial R) (c : R) (p : Polynomial R) (n : ℕ) :=
  ∀ x : ℤ, Polynomial.eval x p ∟ Polynomial.eval (Polynomial.eval x p) f ≤ c

theorem polynomial_bound
  (f : Polynomial ℝ) (deg_f_ge_1 : 1 ≤ f.natDegree)
  (c : ℝ) (c_gt_0 : 0 < c) :
  ∃ n_0 : ℕ, ∀ (p : Polynomial ℝ), p.LC = 1 → n_0 ≤ p.natDegree →
  ∀ (x : ℤ), polynomial_bounded_numbers f c p x ≤ p.natDegree := sorry

end polynomial_bound_l582_582810


namespace shortest_path_length_l582_582189

/-- The distance between two points in the plane. -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

/-- The length of the shortest path from (0, 0) to (20, 25) 
    avoiding the interior of the circle centered at (10, 12.5) with radius 8 is 
    27.732 + 8π / 3. -/
theorem shortest_path_length 
  (A D O : ℝ × ℝ)
  (r : ℝ)
  (circle_eq : ∀ (P : ℝ × ℝ), (P.1 - O.1) ^ 2 + (P.2 - O.2) ^ 2 = r ^ 2 ↔ P = A ∨ P = D) :
  A = (0, 0) → D = (20, 25) → O = (10, 12.5) → r = 8 →
  distance A O = real.sqrt (10^2 + 12.5^2) → 
  ∃ (B C : ℝ × ℝ), 
  distance A B + distance B C + distance C D = 27.732 + 8 * real.pi / 3 := 
by 
  rintros rfl rfl rfl rfl rfl,
  sorry

end shortest_path_length_l582_582189


namespace annual_interest_rate_correct_l582_582956

noncomputable def annual_interest_rate (A P t : ℝ) (n : ℕ) : ℝ :=
  let x := (A / P) ^ (1 / (n * t))
  in 2 * (x - 1)

theorem annual_interest_rate_correct :
  annual_interest_rate 1.3382255776 1 2.5 2 = 0.1214 :=
by
  -- The translation of the provided steps directly into Lean
  sorry

end annual_interest_rate_correct_l582_582956


namespace modulus_of_z_l582_582165

variables (a b : ℝ)

def z : ℂ := a + b * complex.I
def z_conj : ℂ := a - b * complex.I

theorem modulus_of_z :
  (z + 2 * z_conj = 3 + 2 * complex.I) →
  complex.abs z = real.sqrt 5 :=
by sorry

end modulus_of_z_l582_582165


namespace seashells_total_after_giving_l582_582859

/-- Prove that the total number of seashells among Henry, Paul, and Leo is 53 after Leo gives away a quarter of his collection. -/
theorem seashells_total_after_giving :
  ∀ (henry_seashells paul_seashells total_initial_seashells leo_given_fraction : ℕ),
    henry_seashells = 11 →
    paul_seashells = 24 →
    total_initial_seashells = 59 →
    leo_given_fraction = 1 / 4 →
    let leo_seashells := total_initial_seashells - henry_seashells - paul_seashells in
    let leo_seashells_after := leo_seashells - (leo_seashells * leo_given_fraction) in
    henry_seashells + paul_seashells + leo_seashells_after = 53 :=
by
  intros
  sorry

end seashells_total_after_giving_l582_582859


namespace probability_first_card_heart_second_king_l582_582645

theorem probability_first_card_heart_second_king :
  ∀ (deck : Finset ℕ) (is_heart : ℕ → Prop) (is_king : ℕ → Prop),
  deck.card = 52 →
  (∀ card ∈ deck, is_heart card ∨ ¬ is_heart card) →
  (∀ card ∈ deck, is_king card ∨ ¬ is_king card) →
  (∃ p : ℚ, p = 1/52) :=
by
  intros deck is_heart is_king h_card h_heart h_king,
  sorry

end probability_first_card_heart_second_king_l582_582645


namespace trig_problem_l582_582133

/-- Given that the terminal side of angle φ passes through point P(3, -4),
implies cos(φ) = 3/5 and sin(φ) = -4/5, and that the distance between two 
adjacent symmetry axes of the function f(x) = sin(ωx + φ) (ω > 0) is π/2,
which implies ω = 2, prove that f(π/4) = 3/5 for f(x) = sin(2x + φ). -/
theorem trig_problem (φ : ℝ) (h_cos : Real.cos φ = 3 / 5) (h_sin : Real.sin φ = - 4 / 5) :
  let ω := 2
  let f : ℝ → ℝ := λ x, Real.sin (ω * x + φ)
  f (Real.pi / 4) = 3 / 5 :=
by {
   sorry
}

end trig_problem_l582_582133


namespace problem_statement_l582_582851

universe u

variables {U : Type u} 

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {1, 3, 6}

theorem problem_statement :
  let complement_I (s : Set ℕ) : Set ℕ := I \ s
  in {2, 7, 8} = (complement_I A) ∩ (complement_I B) :=
by
  sorry

end problem_statement_l582_582851


namespace find_b_for_continuity_l582_582844

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x >= 5 then x + 4 else 3*x + b

theorem find_b_for_continuity (b : ℝ) : (∀ x : ℝ, (f x b) tends to (if x >= 5 then x + 4 else 3*x + b) as x tends to 5) ↔ b = -6 := sorry

end find_b_for_continuity_l582_582844


namespace find_good_functions_l582_582223

-- Definitions of the conditions
def good_function (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1) ∧
  (∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → (abs (x - y))^2 ≤ abs (f x - f y) ∧ abs (f x - f y) ≤ abs (x - y))

-- The main theorem
theorem find_good_functions (f : ℝ → ℝ) (hgood : good_function f) :
  ∃ c : ℝ, (f = λ x, x + c) ∨ (f = λ x, -x + c) :=
sorry

end find_good_functions_l582_582223


namespace find_a_and_monotonicity_l582_582937

noncomputable def f (x a : ℝ) := x^3 + a * x^2 - 9 * x - 1

theorem find_a_and_monotonicity (a : ℝ) (h : a < 0) :
  (∃ x : ℝ, deriv (f x) x = -12) ∧
  (intervals_of_monotonicity : (f (-3) = ∅ ∨ increasing_on (-∞, -1)) ∧ 
                               (f (-1) = ∅ ∨ decreasing_on (-1, 3)) ∧ 
                               (f (3) = ∅ ∨ increasing_on (3, ∞))) := sorry

end find_a_and_monotonicity_l582_582937


namespace find_other_leg_length_l582_582964

theorem find_other_leg_length (a b c : ℝ) (h1 : a = 15) (h2 : b = 5 * Real.sqrt 3) (h3 : c = 2 * (5 * Real.sqrt 3)) (h4 : a^2 + b^2 = c^2)
  (angle_A : ℝ) (h5 : angle_A = Real.pi / 3) (h6 : angle_A ≠ Real.pi / 2) :
  b = 5 * Real.sqrt 3 :=
by
  sorry

end find_other_leg_length_l582_582964


namespace range_of_a_l582_582509

open Real

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → -1 ≤ a ∧ a ≤ 3 :=
by
  intro h
  -- insert the actual proof here
  sorry

end range_of_a_l582_582509


namespace can_construct_triangle_l582_582900

noncomputable
def constructible_triangle {A B C D : Type} [EuclideanGeometry A B C D] : Prop :=
  ∃ (AC BC AB : ℝ) (angle_BAC angle_BDC : ℝ),
    (AC = BC) ∧
    (AC > 0) ∧
    (AD = AB) ∧
    (BCD_is_isosceles : is_isosceles ∠BDC ∠BCD) ∧
    (constructible : (angle_BAC = 36 ∨ angle_BAC = 72))

theorem can_construct_triangle 
  {A B C D : Type} [EuclideanGeometry A B C D] 
  (AC BC AB : ℝ) 
  (h1 : AC = BC) 
  (h2 : AD = AB) 
  (h3 : is_isosceles ∠BDC ∠BCD) 
  (h4 : AB > 0) :
  constructible_triangle AC BC AB :=
begin
  sorry,
end

end can_construct_triangle_l582_582900


namespace sum_of_y_coordinate_digits_of_C_l582_582555

-- Definitions of points on the parabola and conditions.
structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_on_parabola (p : Point) : Prop := p.y = p.x^2

def is_horizontal (p1 p2 : Point) : Prop := p1.y = p2.y

def is_isosceles_right_triangle (p1 p2 p3 : Point) : Prop :=
  (p1.x = p2.x ∧ p1.y = p2.y ∧ (p3.x = p1.x + p1.y - p1.y ∧ p3.y = p1.y + 10 ∨ p3.x = p1.x ∧ p3.y = p1.y + 10))

def triangle_area_is (p1 p2 p3 : Point) (area : ℝ) : Prop :=
  (1 / 2) * abs ((p1.x - p3.x) * (p2.y - p3.y) - (p1.y - p3.y) * (p2.x - p3.x)) = area

-- The final theorem statement
theorem sum_of_y_coordinate_digits_of_C :
  ∃ (A B C : Point), 
  is_on_parabola A ∧ is_on_parabola B ∧ is_on_parabola C ∧
  is_horizontal A B ∧ 
  A.y = 25 ∧
  B.y = 25 ∧
  is_isosceles_right_triangle A B C ∧ 
  triangle_area_is A B C 50 ∧ 
  (let yC := C.y.to_digits.sum in yC = 8) :=
by
  sorry

end sum_of_y_coordinate_digits_of_C_l582_582555


namespace probability_of_differ_by_three_is_one_eighth_l582_582711

-- Define that a standard 8-sided die has outcomes 1 to 8
def die := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define a function to check if two rolls differ by exactly 3
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

-- Define what it means to roll the die twice
def roll_die_twice := (die × die)

-- Condition: Count the number of successful outcomes
def successful_outcomes := { (x, y) ∈ roll_die_twice | differ_by_three x y }

-- Condition: Total number of possible outcomes
def total_outcomes := die.card * die.card

-- Define the probability as the ratio of successful outcomes to total outcomes
def probability := successful_outcomes.card.toNat / total_outcomes.toNat

-- Theorem to prove
theorem probability_of_differ_by_three_is_one_eighth : probability = 1 / 8 := by
  sorry

end probability_of_differ_by_three_is_one_eighth_l582_582711


namespace sum_of_extreme_a_l582_582795

theorem sum_of_extreme_a (a : ℝ) (h : ∀ x, x^2 - a*x - 20*a^2 < 0) (h_diff : |5*a - (-4*a)| ≤ 9) : 
  -1 ≤ a ∧ a ≤ 1 ∧ a ≠ 0 → a_min + a_max = 0 :=
by 
  sorry

end sum_of_extreme_a_l582_582795


namespace part_I_part_II_l582_582484

noncomputable def f (x a : ℝ) := 2 * |x - 1| - a
noncomputable def g (x m : ℝ) := - |x + m|

theorem part_I (a : ℝ) : 
  (∃! x : ℤ, x = -3 ∧ g x 3 > -1) → m = 3 := 
sorry

theorem part_II (m : ℝ) : 
  (∀ x : ℝ, f x a > g x m) → a < 4 := 
sorry

end part_I_part_II_l582_582484


namespace sin_x_sin_y_eq_sin_beta_sin_gamma_l582_582533

theorem sin_x_sin_y_eq_sin_beta_sin_gamma
  (A B C M : Type)
  (AM BM CM : ℝ)
  (alpha beta gamma x y : ℝ)
  (h1 : AM * AM = BM * CM)
  (h2 : BM ≠ 0)
  (h3 : CM ≠ 0)
  (hx : AM / BM = Real.sin beta / Real.sin x)
  (hy : AM / CM = Real.sin gamma / Real.sin y) :
  Real.sin x * Real.sin y = Real.sin beta * Real.sin gamma := 
sorry

end sin_x_sin_y_eq_sin_beta_sin_gamma_l582_582533


namespace meghan_total_money_l582_582948

theorem meghan_total_money :
  let num_100_bills := 2
  let num_50_bills := 5
  let num_10_bills := 10
  let value_100_bills := num_100_bills * 100
  let value_50_bills := num_50_bills * 50
  let value_10_bills := num_10_bills * 10
  let total_money := value_100_bills + value_50_bills + value_10_bills
  total_money = 550 := by sorry

end meghan_total_money_l582_582948


namespace log_subtraction_l582_582068

theorem log_subtraction : (log 5 625) - (log 5 25) = 2 :=
by
  have log_625 := log_eq_of_pow (5 : ℝ) 625 4 
  have log_25 := log_eq_of_pow (5 : ℝ) 25 2
  rw [log_625, log_25]
  norm_num

end log_subtraction_l582_582068


namespace relationship_between_volume_and_time_time_for_specified_volume_l582_582364

-- Definitions for the problem conditions
def initial_volume : ℝ := 300
def drainage_rate : ℝ := 25
def remaining_volume (t : ℝ) : ℝ := initial_volume - drainage_rate * t

-- The first part of the proof: the function relationship
theorem relationship_between_volume_and_time :
  ∀ t : ℝ, remaining_volume t = 300 - 25 * t :=
by
  intro t
  refl

-- The second part of the proof: finding the time when the remaining volume is 150 m³
theorem time_for_specified_volume :
  ∃ t : ℝ, remaining_volume t = 150 :=
by
  use 6
  sorry

end relationship_between_volume_and_time_time_for_specified_volume_l582_582364


namespace find_c_plus_d_l582_582873

variables {a b c d : ℝ}

theorem find_c_plus_d (h1 : a + b = 16) (h2 : b + c = 9) (h3 : a + d = 10) : c + d = 3 :=
by
  sorry

end find_c_plus_d_l582_582873


namespace janet_total_owed_l582_582544

def warehouseHourlyWage : ℝ := 15
def managerHourlyWage : ℝ := 20
def numWarehouseWorkers : ℕ := 4
def numManagers : ℕ := 2
def workDaysPerMonth : ℕ := 25
def workHoursPerDay : ℕ := 8
def ficaTaxRate : ℝ := 0.10

theorem janet_total_owed : 
  let warehouseWorkerMonthlyWage := warehouseHourlyWage * workDaysPerMonth * workHoursPerDay
  let managerMonthlyWage := managerHourlyWage * workDaysPerMonth * workHoursPerDay
  let totalMonthlyWages := (warehouseWorkerMonthlyWage * numWarehouseWorkers) + (managerMonthlyWage * numManagers)
  let ficaTaxes := totalMonthlyWages * ficaTaxRate
  let totalAmountOwed := totalMonthlyWages + ficaTaxes
  totalAmountOwed = 22000 := by
  sorry

end janet_total_owed_l582_582544


namespace marble_game_solution_l582_582552

theorem marble_game_solution (B R : ℕ) (h1 : B + R = 21) (h2 : (B * (B - 1)) / (21 * 20) = 1 / 2) : B^2 + R^2 = 261 :=
by
  sorry

end marble_game_solution_l582_582552


namespace find_parabola_equation_l582_582469

noncomputable def given_conditions (A B : Point) (O : Point) (line_eq : Line) (p : ℝ) (b : ℝ) :=
  let yx := λ x => x + b in
  let y2 := λ y => y^2 - 2*p*x in
  let OA := vector O A in
  let OB := vector O B in
  (p > 0) ∧  -- condition for p
  (line_eq = yx) ∧  -- line equation
  (y2 = 2 * p) ∧  -- parabola equation
  (OA ⟂ OB) ∧  -- perpendicular condition
  (triangle_area O A B = 2 * sqrt 5) -- area condition

theorem find_parabola_equation (A B : Point) (O : Point) (line_eq : Line) (p : ℝ) (b : ℝ) :
  given_conditions A B O line_eq p b -> (p = 1) :=
by
sorry

end find_parabola_equation_l582_582469


namespace sally_initial_cards_l582_582973

theorem sally_initial_cards
  (dan_cards : ℕ)
  (sally_extra : ℕ)
  (sally_more_than_dan : ℕ)
  (h1 : dan_cards = 41)
  (h2 : sally_extra = 20)
  (h3 : sally_more_than_dan = 6) :
  ∃ S : ℕ, S + sally_extra = dan_cards + sally_more_than_dan ∧ S = 27 :=
by {
  use 27,
  split,
  { calc
      27 + sally_extra = 27 + 20     : by rw h2
      ...             = 47            : by norm_num
      ...             = 41 + 6        : by rw [h1, h3]
      ...             = dan_cards + sally_more_than_dan : by rw [h1, h3] },
  { refl }
}

end sally_initial_cards_l582_582973


namespace greatest_gcd_of_6T_n_and_n_plus_1_l582_582094

theorem greatest_gcd_of_6T_n_and_n_plus_1 (n : ℕ) (h_pos : 0 < n) :
  let T_n := n * (n + 1) / 2 in
  gcd (6 * T_n) (n + 1) = 3 ↔ (n + 1) % 3 = 0 :=
by
  sorry

end greatest_gcd_of_6T_n_and_n_plus_1_l582_582094


namespace berries_count_l582_582340

theorem berries_count (total_berries : ℕ)
  (h1 : total_berries = 42)
  (h2 : total_berries / 2 = 21)
  (h3 : total_berries / 3 = 14) :
  total_berries - (total_berries / 2 + total_berries / 3) = 7 :=
by
  rw [h1, h2, h3]
  norm_num
  exact rfl

end berries_count_l582_582340


namespace probability_of_first_heart_second_king_l582_582643

noncomputable def probability_first_heart_second_king : ℚ :=
  1 / 52 * 3 / 51 + 12 / 52 * 4 / 51

theorem probability_of_first_heart_second_king :
  probability_first_heart_second_king = 1 / 52 :=
by
  sorry

end probability_of_first_heart_second_king_l582_582643


namespace find_triple_l582_582427
-- Import necessary libraries

-- Define the required predicates and conditions
def satisfies_conditions (x y z : ℕ) : Prop :=
  x ≤ y ∧ y ≤ z ∧ x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2)

-- The main theorem statement
theorem find_triple : 
  ∀ (x y z : ℕ), satisfies_conditions x y z → (x, y, z) = (2, 251, 252) :=
by
  sorry

end find_triple_l582_582427


namespace find_second_annual_rate_l582_582358

noncomputable def calc_second_interest_rate
    (initial_investment : ℕ)
    (first_period_months : ℕ)
    (first_annual_rate_percent : ℕ)
    (second_period_months : ℕ)
    (final_investment_value : ℚ)
    (total_value_after_first_period : ℚ) : ℚ :=
  let first_interest_rate := first_annual_rate_percent * first_period_months / 12
  let value_after_first := initial_investment * (1 + first_interest_rate / 100)
  have value_correct : value_after_first = total_value_after_first_period := by
    simp [value_after_first, total_value_after_first_period]
  let second_interest_rate := (final_investment_value / total_value_after_first_period - 1) * 300 / 2
  second_interest_rate

theorem find_second_annual_rate :
  calc_second_interest_rate 12000 8 8 8 13662.40 12640 = 12.15 := by
  sorry

end find_second_annual_rate_l582_582358


namespace no_passing_quadrant_III_l582_582452

def y (k x : ℝ) : ℝ := k * x - k

theorem no_passing_quadrant_III (k : ℝ) (h : k < 0) :
  ¬(∃ x y : ℝ, x < 0 ∧ y < 0 ∧ y = k * x - k) :=
sorry

end no_passing_quadrant_III_l582_582452


namespace inf_many_solutions_to_ineq_l582_582863

theorem inf_many_solutions_to_ineq (x : ℕ) : (15 < 2 * x + 20) ↔ x ≥ 1 :=
by
  sorry

end inf_many_solutions_to_ineq_l582_582863


namespace balance_pots_l582_582699

theorem balance_pots 
  (w1 : ℕ) (w2 : ℕ) (m : ℕ)
  (h_w1 : w1 = 645)
  (h_w2 : w2 = 237)
  (h_m : m = 1000) :
  ∃ (m1 m2 : ℕ), 
  (w1 + m1 = w2 + m2) ∧ 
  (m1 + m2 = m) ∧ 
  (m1 = 296) ∧ 
  (m2 = 704) := by
  sorry

end balance_pots_l582_582699


namespace rooks_in_every_square_l582_582569

/-- A 300x300 board setup where rooks are placed to beat the entire board.
  Each rook beats no more than one other rook.
  Show that there is at least one rook in every 201x201 square. -/
theorem rooks_in_every_square (board : ℕ) (k : ℕ) (h_board : board = 300)
  (h1 : ∀ i j, i ≤ board → j ≤ board → beats board i j)
  (h2 : ∀ i j, ∃! r, rook r ∧ beats r (i,j)) :
  ∃ k, k = 201 :=
by 
  sorry

/-- Definition of a rook on a board position -/
def rook (r : ℕ) : Prop := sorry

/-- Definition of a board position beating another -/
def beats (board : ℕ) (i j : ℕ) : Prop := sorry

end rooks_in_every_square_l582_582569


namespace percentage_increase_l582_582888

theorem percentage_increase :
  let original_employees := 852
  let new_employees := 1065
  let increase := new_employees - original_employees
  let percentage := (increase.toFloat / original_employees.toFloat) * 100
  percentage = 25 := 
by 
  sorry

end percentage_increase_l582_582888


namespace sphere_volume_given_surface_area_l582_582510

theorem sphere_volume_given_surface_area (r : ℝ) (V : ℝ) (S : ℝ)
  (hS : S = 36 * Real.pi)
  (h_surface_area : 4 * Real.pi * r^2 = S)
  (h_volume : V = (4/3) * Real.pi * r^3) : V = 36 * Real.pi := by
  sorry

end sphere_volume_given_surface_area_l582_582510


namespace sum_of_arithmetic_subsequence_l582_582816

noncomputable def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a (n) + d

theorem sum_of_arithmetic_subsequence (a : ℕ → ℤ) (d : ℤ) (h1 : d = -2)
  (h2 : arithmetic_sequence a)
  (h3 : ∑ i in (finset.range 33).map (function.embedding.subtype (λ n, true)), a (3 * i + 1) = 50) :
  ∑ i in (finset.range 33).map (function.embedding.subtype (λ n, true)), a (3 * i + 3) = -66 :=
sorry

end sum_of_arithmetic_subsequence_l582_582816


namespace units_digit_of_n_l582_582128

theorem units_digit_of_n
  (m n : ℕ)
  (h1 : m * n = 23^7)
  (h2 : m % 10 = 9) : n % 10 = 3 :=
sorry

end units_digit_of_n_l582_582128


namespace evaluate_expression_l582_582415

theorem evaluate_expression : 8^3 + 3 * 8^2 + 3 * 8 + 1 = 729 := by
  sorry

end evaluate_expression_l582_582415


namespace no_fractional_solution_l582_582768

theorem no_fractional_solution (x y : ℚ)
  (h₁ : ∃ m : ℤ, 13 * x + 4 * y = m)
  (h₂ : ∃ n : ℤ, 10 * x + 3 * y = n) :
  (∃ a b : ℤ, x ≠ a ∧ y ≠ b) → false :=
by {
  sorry
}

end no_fractional_solution_l582_582768


namespace sin_alpha_value_l582_582446

theorem sin_alpha_value (α : Real) (h1 : tan (α + π / 4) = 1 / 2) (h2 : -π / 2 < α ∧ α < 0) :
  sin α = - sqrt 10 / 10 := by
  sorry

end sin_alpha_value_l582_582446


namespace triangle_area_is_div_result_l582_582772

noncomputable def area_of_triangle (Ω1 Ω2 Ω3 : Circle) (P1 P2 P3 : Point) 
(radius : ℝ) (equilateral: EquilateralTriangle P1 P2 P3)
(tangent: ∀ i, TangentToCircle (P_iP_(i+1)) (ω_i))
: ℝ :=
(sqrt (13 * sqrt 3 / 4 + 3 * sqrt 10))

theorem triangle_area_is_div_result
  {Ω1 Ω2 Ω3 : Circle}
  (P1 P2 P3 : Point)
  (radius := 3)
  (equilateral: EquilateralTriangle P1 P2 P3)
  (tangent: ∀ i, TangentToCircle (P_iP_(i+1)) (ω_i))
  : let a := 507
    let b := 90 in a + b = 597 := 
by
  sorry

end triangle_area_is_div_result_l582_582772


namespace max_value_abs_diff_l582_582853

def a : ℝ × ℝ := (Real.sqrt 2, -Real.sqrt 2)
def b (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)

theorem max_value_abs_diff (α : ℝ) : ∃ α, |(a.1 - b(α).1, a.2 - b(α).2)| = 3 :=
  sorry

end max_value_abs_diff_l582_582853


namespace solve_equation_l582_582065

theorem solve_equation : ∃ x : ℝ, 2 * x + 3 = 7 ∧ x = 2 := by
  exists 2
  split
  · ring
  · rfl

end solve_equation_l582_582065


namespace ratio_constant_l582_582383

variables {O1 O2 A P C : Point} (R1 R2 : ℝ)
variables (circle1 : Circle O1 R1) (circle2 : Circle O2 R2)

-- Hypotheses
hypothesis h1 : tangent circle1 circle2 A
hypothesis h2 : OnCircle P circle1
hypothesis h3 : tangent P C circle2

theorem ratio_constant : ∀ (O1 O2 A P C : Point) (R1 R2 : ℝ),
  tangent circle1 circle2 A →
  OnCircle P circle1 →
  tangent P C circle2 →
  ∃ k : ℝ, k = sqrt ((R1 - R2) / R1) ∧ (PC / PA) = k := 
sorry

end ratio_constant_l582_582383


namespace additional_cost_is_three_dollars_l582_582958

-- Definition of the cost of peanut butter per jar
def peanut_butter_cost : ℝ := 3

-- Given that the almond butter costs three times as much as the peanut butter
def almond_butter_multiplier : ℝ := 3

-- Definition of the cost of almond butter per jar
def almond_butter_cost : ℝ := almond_butter_multiplier * peanut_butter_cost

-- Given that it takes half a jar to make a batch of cookies
def jar_fraction_per_batch : ℝ := 1 / 2

-- Calculate the cost per batch for peanut butter and almond butter
def peanut_butter_cost_per_batch : ℝ := jar_fraction_per_batch * peanut_butter_cost
def almond_butter_cost_per_batch : ℝ := jar_fraction_per_batch * almond_butter_cost

-- The additional cost per batch of almond butter cookies compared to peanut butter cookies
def additional_cost_per_batch : ℝ := almond_butter_cost_per_batch - peanut_butter_cost_per_batch

-- The theorem stating the additional cost per batch is 3 dollars
theorem additional_cost_is_three_dollars : additional_cost_per_batch = 3 := by
  sorry

end additional_cost_is_three_dollars_l582_582958


namespace line_equations_passing_through_point_with_angle_l582_582143

theorem line_equations_passing_through_point_with_angle
  (P : ℝ × ℝ)
  (l_slope : ℝ)
  (l : ℝ × ℝ → ℝ)
  (θ : ℝ) 
  (l_through_point : ∀ x y, y - 1 = l_slope * (x - 2)) :
  P = (2, 1) →
  l_slope = sqrt 3 →
  θ = π / 6 →
  (∀ x y, y - 1 = (sqrt 3 / 3) * (x - 2) → x - sqrt 3 * y - 2 + sqrt 3 = 0) ∨
  (∀ x, x = 2) :=
by
  intros hP hslope ht
  dsimp at *
  sorry

end line_equations_passing_through_point_with_angle_l582_582143


namespace factor_expression_l582_582396

def expression := (12 * x^6 + 30 * x^4 - 6) - (2 * x^6 - 4 * x^4 - 6)

theorem factor_expression (x : ℝ) : expression = 2 * x^4 * (5 * x^2 + 17) :=
by
  sorry

end factor_expression_l582_582396


namespace cube_root_sum_is_integer_iff_l582_582977

theorem cube_root_sum_is_integer_iff (n m : ℤ) (hn : n = m * (m^2 + 3) / 2) :
  ∃ (k : ℤ), (n + Real.sqrt (n^2 + 1))^(1/3) + (n - Real.sqrt (n^2 + 1))^(1/3) = k :=
by
  sorry

end cube_root_sum_is_integer_iff_l582_582977


namespace river_flow_rate_l582_582011

theorem river_flow_rate
  (depth width volume_per_minute : ℝ)
  (h1 : depth = 2)
  (h2 : width = 45)
  (h3 : volume_per_minute = 6000) :
  (volume_per_minute / (depth * width)) * (1 / 1000) * 60 = 4.0002 :=
by
  -- Sorry is used to skip the proof.
  sorry

end river_flow_rate_l582_582011


namespace no_fractional_solutions_l582_582757

theorem no_fractional_solutions (x y : ℚ) (hx : x.denom ≠ 1) (hy : y.denom ≠ 1) :
  ¬ (∃ m n : ℤ, 13 * x + 4 * y = m ∧ 10 * x + 3 * y = n) :=
sorry

end no_fractional_solutions_l582_582757


namespace find_r_for_f_of_3_eq_0_l582_582212

noncomputable def f (x r : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + x^2 - 4 * x + r

theorem find_r_for_f_of_3_eq_0 : ∃ r : ℝ, f 3 r = 0 ∧ r = -186 := by
  sorry

end find_r_for_f_of_3_eq_0_l582_582212


namespace value_of_a_l582_582881

theorem value_of_a (x y a : ℝ) (h1 : x - 2 * y = a - 6) (h2 : 2 * x + 5 * y = 2 * a) (h3 : x + y = 9) : a = 11 := 
by
  sorry

end value_of_a_l582_582881


namespace coefficient_x4_in_PQ_product_l582_582305

def P (x : ℝ) : ℝ := x^5 - 4*x^4 + 3*x^3 - 5*x^2 + x - 2
def Q (x : ℝ) : ℝ := 3*x^2 - 2*x + 5

theorem coefficient_x4_in_PQ_product : 
  (∀ x : ℝ, polynomial.coeff (polynomial.mul (polynomial.of_fn P) (polynomial.of_fn Q)) 4) = -17 :=
by sorry

end coefficient_x4_in_PQ_product_l582_582305


namespace find_M_l582_582974

noncomputable def side_length : ℝ := 3
noncomputable def cube_surface_area (s: ℝ) : ℝ := 6 * s^2
noncomputable def sphere_surface_area (A: ℝ) (r: ℝ) : Prop := 4 * Real.pi * r^2 = A
noncomputable def sphere_volume (r: ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def M_from_volume (v: ℝ) : ℝ := (v * Real.sqrt Real.pi) / Real.sqrt 3

theorem find_M : 
  let surface_area := cube_surface_area side_length in
  ∃ r : ℝ, sphere_surface_area surface_area r ∧ M_from_volume (sphere_volume r) = 36 :=
by 
  sorry

end find_M_l582_582974


namespace simplified_expression_value_l582_582725

theorem simplified_expression_value :
  (2 + 7/9 : ℚ)^0.5 + (0.1 : ℚ)^(-2) + (2 + 10/27 : ℚ)^(-2/3) - (Real.pi)^0 + (37/48 : ℚ) = 807/8 := 
by
  sorry

end simplified_expression_value_l582_582725


namespace circle_radius_l582_582084

theorem circle_radius :
  ∃ r : ℝ, (∃ x y : ℝ, (16 * x^2 + 32 * x + 16 * y^2 - 48 * y + 76 = 0) ∧ r = sqrt (3 / 2)) :=
sorry

end circle_radius_l582_582084


namespace largest_possible_n_l582_582738

def triangle_property (s : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), {a, b, c} ⊆ s → c ≤ a + b

def largest_n_with_triangle_property (m : ℕ) : Prop :=
  (∀ s : Finset ℕ, s.card = 10 → triangle_property s) →
  m = 363

theorem largest_possible_n :
  ∃ n, largest_n_with_triangle_property n :=
begin
  use 363,
  intros s h,
  -- The proof would go here
  sorry
end

end largest_possible_n_l582_582738


namespace ball_returns_to_Bella_after_thirteen_throws_l582_582625

/-- There are thirteen girls standing in a circle. The ball is passed clockwise.
    The first girl, Bella, starts with the ball, skips the next four girls, and throws to the sixth girl.
    The girl receiving the ball then skips the next four girls and continues the pattern.
    How many throws are needed for the ball to come back to Bella? -/
theorem ball_returns_to_Bella_after_thirteen_throws :
  (∃ k, k = 13 ∧
    (∀ n, ∃ m, m = (n + 6) % 13) ∧
    (let Bella := 1, next_pos := ((Bella + 6) % 13) in 
      next_pos = Bella) 
  ) :=
sorry

end ball_returns_to_Bella_after_thirteen_throws_l582_582625


namespace scientific_notation_of_number_l582_582194

def number := 460000000
def scientific_notation (n : ℕ) (s : ℝ) := s * 10 ^ n

theorem scientific_notation_of_number :
  scientific_notation 8 4.6 = number :=
sorry

end scientific_notation_of_number_l582_582194


namespace jill_savings_percentage_l582_582771

-- Definitions of all the conditions stated in the problem.
def net_monthly_salary : ℝ := 3500
def discretionary_income : ℝ := net_monthly_salary / 5
def vacation_fund : ℝ := 0.3 * discretionary_income
def eating_out_socializing : ℝ := 0.35 * discretionary_income
def gifts_and_charity : ℝ := 105

-- Statement of the proof problem showing that the percentage for savings is 20%.
theorem jill_savings_percentage :
  ∃ (savings_percentage : ℝ), savings_percentage = 20 ∧
  discretionary_income = vacation_fund + eating_out_socializing + gifts_and_charity + (savings_percentage / 100 * discretionary_income) :=
sorry

end jill_savings_percentage_l582_582771


namespace concert_attendance_l582_582715

open Real

theorem concert_attendance :
  (∀ P : ℝ, (2 / 3 * P) * (1 / 2) * (2 / 5) = 20 → P = 150) :=
by
  intro P
  intro h
  have h1 : (2 / 3) * (1 / 2) = 1 / 3 := by rw [mul_div₀, one_div, mul_one]
  have h2 : (1 / 3) * (2 / 5) = 2 / 15 := by rw [mul_comm_div·trans]
  rw h1 at h
  rw h2 at h
  sorry

end concert_attendance_l582_582715


namespace usha_drank_fraction_of_bottle_l582_582942

theorem usha_drank_fraction_of_bottle (total_bottle : ℝ) (t : ℝ) (mala_speed : ℝ) (usha_speed : ℝ) 
(h_total : total_bottle = 1) 
(h_times_equal : t > 0) 
(h_speed_ratio : mala_speed = 4 * usha_speed) :
  usha_speed * t / total_bottle = 1 / 5 :=
by
  have h : 5 * usha_speed * t = 1,
  { rw [←h_speed_ratio],
    ring_nf,
    rw [mul_assoc, mul_comm 4 usha_speed, mul_assoc, ←add_mul, ←add_assoc],
    norm_num,
    simp only [one_mul],
    apply h_times_equal,
    apply h_total },
  simp [h],
  field_simp,
  ring
  sorry

end usha_drank_fraction_of_bottle_l582_582942


namespace max_product_geom_sequence_l582_582806

theorem max_product_geom_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h_decreasing : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a2 : a 2 = 2)
  (h_sum : a 1 + a 2 + a 3 = 7) :
  ∃ n, n = 2 ∨ n = 3 ∧ (∀ k, k ≤ n → a k ≥ 1) → 
  (∀ m, ∏ i in finset.range m, a i ≤ ∏ j in finset.range n, a j) :=
by sorry

end max_product_geom_sequence_l582_582806


namespace calculation_factorial_difference_l582_582663

theorem calculation_factorial_difference :
  (13! - 12!) / 10! = 1584 := by
  sorry

end calculation_factorial_difference_l582_582663


namespace Loisa_saves_70_l582_582226

-- Define the conditions
def tablet_cost_cash := 450
def down_payment := 100
def payment_first_4_months := 40 * 4
def payment_next_4_months := 35 * 4
def payment_last_4_months := 30 * 4

-- Define the total installment payment
def total_installment_payment := down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

-- Define the amount saved by paying cash instead of on installment
def amount_saved := total_installment_payment - tablet_cost_cash

-- The theorem to prove the savings amount
theorem Loisa_saves_70 : amount_saved = 70 := by
  -- Direct calculation or further proof steps here
  sorry

end Loisa_saves_70_l582_582226


namespace excircle_incircle_tangents_equal_l582_582536

theorem excircle_incircle_tangents_equal
  (ABC : Triangle)
  (B1 : Point)
  (B2 : Point)
  (B3 : Point)
  (C1 : Point)
  (C2 : Point)
  (C3 : Point)
  (excircle_opposite_A_touches_AC_at_B1 : ABC.excircle(opposite=ABC.A).touches(ABC.AC, at=B1))
  (BB1_intersects_excircle_at_B2 : segment(B=ABC.B, B1).intersects(ABC.excircle(opposite=ABC.A), again_at=B2))
  (tangent_from_B2_to_BC_intersects_at_B3 : tangent(ABC.excircle(opposite=ABC.A), from=B2).intersects(ABC.BC, at=B3))
  (incircle_touches_AB_at_C1 : incircle(ABC).touches(ABC.AB, at=C1))
  (CC1_intersects_incircle_at_C2 : segment(C=ABC.C, C1).intersects(incircle(ABC), again_at=C2))
  (tangent_from_C2_to_BC_intersects_at_C3 : tangent(incircle(ABC), from=C2).intersects(ABC.BC, at=C3)) :
  segment(B=B2, B3.length) = segment(C=C2, C3.length) := 
sorry

end excircle_incircle_tangents_equal_l582_582536


namespace math_problem_l582_582038

variable (x b : ℝ)
variable (h1 : x < b)
variable (h2 : b < 0)
variable (h3 : b = -2)

theorem math_problem : x^2 > b * x ∧ b * x > b^2 :=
by
  sorry

end math_problem_l582_582038


namespace max_power_of_29_dividing_factorial_2003_l582_582036

open Nat

theorem max_power_of_29_dividing_factorial_2003 :
  (∑ k in range (log 2003 29 + 1), 2003 / 29 ^ k) = 71 := by
  sorry

end max_power_of_29_dividing_factorial_2003_l582_582036


namespace card_selection_proof_l582_582622

def cardSelectionMethods (n : ℕ) : ℕ :=
  if n = 0 then 0 else 5 * n^2 + 6 * n + 1

theorem card_selection_proof :
  ∀ n : ℕ, cardSelectionMethods n = 5 * n^2 + 6 * n + 1 :=
by
  intro n
  cases n
  · simp [cardSelectionMethods]
  · unfold cardSelectionMethods
    induction n with k ih
    · rw [Nat.add_one]
      simp
    · rw [Nat.add_one, ih]
      sorry

end card_selection_proof_l582_582622


namespace find_a2_l582_582842

-- Define the geometric sequence and its properties
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions 
variables (a : ℕ → ℝ) (h_geom : is_geometric a)
variables (h_a1 : a 1 = 1/4)
variables (h_condition : a 3 * a 5 = 4 * (a 4 - 1))

-- The goal is to prove a 2 = 1/2
theorem find_a2 : a 2 = 1/2 :=
by
  sorry

end find_a2_l582_582842


namespace calculate_total_money_l582_582949

theorem calculate_total_money (n100 n50 n10 : ℕ) 
  (h1 : n100 = 2) (h2 : n50 = 5) (h3 : n10 = 10) : 
  (n100 * 100 + n50 * 50 + n10 * 10 = 550) :=
by
  sorry

end calculate_total_money_l582_582949


namespace trig_identity_problem_l582_582080

theorem trig_identity_problem :
  (sin (Real.pi / 12) * cos (Real.pi / 12) + cos (11 * Real.pi / 12) * cos (7 * Real.pi / 12)) /
  (sin (19 * Real.pi / 180) * cos (11 * Real.pi / 180) + cos (161 * Real.pi / 180) * cos (101 * Real.pi / 180)) = 1 := sorry

end trig_identity_problem_l582_582080


namespace sunflower_count_l582_582252

/--
Rose bought 24 flowers in total.
Out of these, 3 are daisies.
The remaining flowers are divided among tulips, sunflowers, and carnations.
The ratio of tulips to sunflowers is 2:3.
40% of the remaining flowers are carnations.
Prove that the number of sunflowers is 6.
-/
theorem sunflower_count :
  ∃ (n : ℕ), n = 6 :=
by
  let total := 24
  let daisies := 3
  let remaining := total - daisies
  let carnations := 0.4 * remaining
  let carnations_int := floor carnations
  let tulips_sunflowers_total := remaining - carnations_int
  let (t : ℕ) := 2
  let (s : ℕ) := 3
  let ratio_total := t + s
  let sunflower_count := (s * tulips_sunflowers_total) / ratio_total
  exact ⟨sunflower_count, sorry⟩

end sunflower_count_l582_582252


namespace least_positive_multiple_of_17_sum_of_digits_510_l582_582307

theorem least_positive_multiple_of_17 (n : ℕ) (h1 : n % 17 = 0) (h2 : n > 500) : n = 510 :=
by sorry

theorem sum_of_digits_510 : (sum_of_digits 510) = 6 :=
by sorry

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.to_string.to_list.map (fun c => c.to_nat - '0'.to_nat)
  digits.sum

end least_positive_multiple_of_17_sum_of_digits_510_l582_582307


namespace star_example_l582_582045

def star (a b : ℤ) : ℤ := a * b^3 - 2 * b + 2

theorem star_example : star 2 3 = 50 := by
  sorry

end star_example_l582_582045


namespace KP_through_X_locus_P_circle_l582_582104

open real -- assuming we're working in the plane with real coordinates

-- Define the prerequisites
variables (c : ℝ → ℝ → Prop) (A B : ℝ × ℝ)
variable (M : ℝ × ℝ)
variable {K : ℝ × ℝ}
variable {P : ℝ × ℝ}

-- Circle condition
axiom circle_eq : c M.1 M.2 ↔ (∃ (O : ℝ × ℝ) (r : ℝ), r > 0 ∧ (O, r) = ((0, 0), 1) ∧ real.sqrt(((M.1 - O.1)^2 + (M.2 - O.2)^2)) = r)

-- Midpoint condition
axiom midpoint_def : K = (0.5 * (B.1 + M.1), 0.5 * (B.2 + M.2))

-- Perpendicular foot condition
axiom foot_def : P = let (a1, a2, m1, m2) := (A.1, A.2, M.1, M.2) in
  if a1 ≠ m1
  then ((a1 * m1 + a2 * m2 - (a1 * m2 + a2 * m1) * (a1 - m1) / (a2 - m2)) / (m1 - (a1 + m1 * (a1 - m1) / (a2 - m2))), 
       (a2 + m2 - (a1 * m1 + a2 * m2 - (a1 * m2 + a2 * m1) * (a1 - m1) / (a2 - m2)) / (m1 - (a1 + m1 * (a1 - m1) / (a2 - m2))))
  else A

-- The fixed point X
constants (X : ℝ × ℝ) 

-- Proof statements for part (a)
theorem KP_through_X : ∀ M on c, KP ↔ ℝ × ℝ := sorry

-- Proof statements for part (b)
theorem locus_P_circle : ∃ (O : ℝ × ℝ) (r : ℝ), r > 0 ∧ ∀ P, P.1^2 + P.2^2 = r^2 := sorry

end KP_through_X_locus_P_circle_l582_582104


namespace four_b_is_222_22_percent_of_a_l582_582255

-- noncomputable is necessary because Lean does not handle decimal numbers directly
noncomputable def a (b : ℝ) : ℝ := 1.8 * b
noncomputable def four_b (b : ℝ) : ℝ := 4 * b

theorem four_b_is_222_22_percent_of_a (b : ℝ) : four_b b = 2.2222 * a b := 
by
  sorry

end four_b_is_222_22_percent_of_a_l582_582255


namespace quadrilateral_symmetry_l582_582700

variables {α : Type*} [metric_space α] {A B C D E P P' t: α}

/-- Let quadrilateral ABCD be inscribed in a circle O.
    Let E be the intersection of opposite sides AB and CD.
    Line t passes through E and is perpendicular to OE.
    Diagonals AC and BD intersect line t at points P and P', respectively.
    We prove that the points P and P' are symmetrical with respect to E. -/
theorem quadrilateral_symmetry
  (h_incircle : ∃ O : α, circle O A = circle O B ∧ circle O C = circle O D)
  (h_intersect : ∃ E : α, line A B ∩ line C D = E)
  (h_perp : ∃ t : α, line t = line_perp E (line O E))
  (h_intersect_diag1 : ∃ P : α, line A C ∩ t = P)
  (h_intersect_diag2 : ∃ P' : α, line B D ∩ t = P') :
  symmetric_to E P P' := sorry

end quadrilateral_symmetry_l582_582700


namespace total_students_high_school_l582_582609

theorem total_students_high_school 
  (students_grade3 : ℕ)
  (total_sample : ℕ)
  (sample_grade1 : ℕ)
  (sample_grade2 : ℕ)
  (students_grade3 = 1000)
  (total_sample = 180)
  (sample_grade1 = 70)
  (sample_grade2 = 60) : 
  let sample_grade3 := total_sample - sample_grade1 - sample_grade2,
      prob_grade3 := sample_grade3 / students_grade3 in
  total_sample / prob_grade3 = 3600 := 
by {
  sorry
}

end total_students_high_school_l582_582609


namespace problem1_problem2_l582_582136

noncomputable def f (x : ℝ) : ℝ := log (2^(x) + 1) / log 2
noncomputable def f_inv (x : ℝ) : ℝ := log (2^(x) - 1) / log 2

theorem problem1 (x : ℝ) (h : x > 1) : f_inv (f x) = x :=
by sorry

theorem problem2 (x : ℝ) (h : 2 * f x - f_inv x = 3) : x = log 3 / log 2 :=
by sorry

end problem1_problem2_l582_582136


namespace largest_consecutive_set_having_triangle_property_l582_582743

noncomputable def max_n_with_triangle_property : ℕ :=
  let S := {6, 7, 13, 20, 33, 53, 86, 139, 225, 364}
  363

theorem largest_consecutive_set_having_triangle_property :
  ∀ S : set ℕ, {6, 7, 8, ..., max_n_with_triangle_property} ⊆ S →
  (∀ s ⊆ S, s.card = 10 →  (∀ a b c ∈ s, a + b > c ∧ b + c > a ∧ a + c > b)) :=
sorry

end largest_consecutive_set_having_triangle_property_l582_582743


namespace find_S2_side_length_l582_582250

theorem find_S2_side_length 
    (x r : ℝ)
    (h1 : 2 * r + x = 2100)
    (h2 : 3 * x + 300 = 3500)
    : x = 1066.67 := 
sorry

end find_S2_side_length_l582_582250


namespace shifted_graphs_equivalent_l582_582633

/-- Define the original function f(x) = 3^x -/
def f (x : ℝ) : ℝ := 3^x

/-- Define the transformed function g(x) = 9 * 3^x + 5 -/
def g (x : ℝ) : ℝ := 9 * 3^x + 5

/-- Prove that to obtain the graph of g from f, the graph of f should be shifted 2 units to the left and 5 units up. -/
theorem shifted_graphs_equivalent :
  ∀ x : ℝ, g(x) = f(x + 2) + 5 :=
by
  -- Proof omitted
  sorry

end shifted_graphs_equivalent_l582_582633


namespace sum_cubes_even_numbers_l582_582397

theorem sum_cubes_even_numbers :
  let n := 50
  let S_pos := ∑ k in finset.range(n).map (λ i, 2 * (i + 1)), (2 * (k + 1))^3 
  let S_neg := ∑ k in finset.range(n).map (λ i, 2 * (i + 1)), (-(2 * (k + 1)))^3 
  let S_add_4 := ∑ k in finset.range(n).map (λ i, 2 * (i + 1)), ((2 * (k + 1))^3 + 4)
  S_pos + S_neg + S_add_4 = 200 := 
by {
  sorry
}

end sum_cubes_even_numbers_l582_582397


namespace Jamie_remaining_ounces_l582_582540

-- Setting up the definitions and the proof statement
namespace JamieProblem

def milliliters_of_milk : ℕ := 250
def milliliters_of_grape_juice : ℕ := 500
def total_limit_milliliters : ℕ := 1000
def milliliters_per_ounce : ℝ := 29.57

-- Sum of milk and grape juice consumed
def total_consumed : ℕ := milliliters_of_milk + milliliters_of_grape_juice

-- Remaining milliliters she can drink before reaching the limit
def remaining_milliliters : ℕ := total_limit_milliliters - total_consumed

-- Conversion of remaining milliliters to ounces
def remaining_ounces : ℝ := remaining_milliliters / milliliters_per_ounce

-- The problem statement to be proved
theorem Jamie_remaining_ounces {ε : ℝ} (hε : 0 < ε) : 
  abs (remaining_ounces - 8.45) < ε :=
by
  -- The proof will be filled in here
  sorry

end JamieProblem

end Jamie_remaining_ounces_l582_582540


namespace possible_values_of_a_l582_582849

theorem possible_values_of_a (a : ℝ) :
  (∃ x, ∀ y, (y = x) ↔ (a * y^2 + 2 * y + a = 0))
  → (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  sorry

end possible_values_of_a_l582_582849


namespace num_diagonals_in_decagon_l582_582861

def num_vertices : ℕ := 10

def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def num_connections (n : ℕ) : ℕ :=
  binom n 2

def num_sides (n : ℕ) : ℕ := n

def num_diagonals (n : ℕ) : ℕ :=
  num_connections n - num_sides n

theorem num_diagonals_in_decagon : num_diagonals num_vertices = 35 := by
  have h := calc
    num_connections num_vertices = binom 10 2 := rfl
    ...                         = 45        := by norm_num
  have hs := calc
    num_sides num_vertices = 10 := rfl
  show num_diagonals num_vertices = 45 - 10, from
    calc num_diagonals 10 = 45 - 10 := by rw [h, hs]
  norm_num
  sorry

end num_diagonals_in_decagon_l582_582861


namespace unique_abm_fibonacci_l582_582589

def fibonacci : ℕ → ℕ
| 1 := 1
| 2 := 1
| (n+3) := fibonacci (n+2) + fibonacci (n+1)

theorem unique_abm_fibonacci :
  ∃! (a b m : ℤ), 0 < a ∧ a < m ∧ 0 < b ∧ b < m ∧ (∀ n : ℕ, 0 < n → (fibonacci n - a * n * b^n) % m = 0) :=
begin
  use [2, 3, 5],
  split,
  { split,
    split,
    norm_num, norm_num, norm_num,
    split,
    norm_num,
    intros n hn,
    sorry,
  },
  rintro ⟨a, b, m⟩ ⟨ha0, ham, hb0, hbm, h⟩,
  -- Prove uniqueness of a, b, m as 2, 3, 5.
  sorry
end

end unique_abm_fibonacci_l582_582589


namespace mod_add_5000_l582_582320

theorem mod_add_5000 (n : ℤ) (h : n % 6 = 4) : (n + 5000) % 6 = 0 :=
sorry

end mod_add_5000_l582_582320


namespace min_distance_on_curve_l582_582453

theorem min_distance_on_curve :
  let curve (x y : ℝ) := x * y - (5 / 2) * x - 2 * y + 3 = 0
  ∃ (x y : ℝ), curve x y ∧ ∀ (x' y' : ℝ), curve x' y' → sqrt (x'^2 + y'^2) ≥ sqrt (5) / 2 :=
by sorry

end min_distance_on_curve_l582_582453


namespace purple_valley_skirts_l582_582246

def AzureValley : ℕ := 60

def SeafoamValley (A : ℕ) : ℕ := (2 * A) / 3

def PurpleValley (S : ℕ) : ℕ := S / 4

theorem purple_valley_skirts :
  PurpleValley (SeafoamValley AzureValley) = 10 :=
by
  sorry

end purple_valley_skirts_l582_582246


namespace find_sum_l582_582219

variables {ℝ : Type*} [inner_product_space ℝ (euclidean_space ℝ (fin 3))]
open euclidean_space
open_locale big_operators

-- Given vectors a, b, c in 3D space that are mutually orthogonal unit vectors
variables (a b c : euclidean_space ℝ (fin 3))
variables (s t u : ℝ)

-- Conditions from the problem
axiom h1 : ⟪a, b⟫ = 0 ∧ ⟪b, c⟫ = 0 ∧ ⟪c, a⟫ = 0  -- Mutual orthogonality
axiom ha : ∥a∥ = 1 -- a is a unit vector
axiom hb : ∥b∥ = 1 -- b is a unit vector
axiom hc : ∥c∥ = 1 -- c is a unit vector
axiom h2 : b = s • (a ⨯ b) + t • (b ⨯ c) + u • (c ⨯ a)  -- Given equation
axiom h3 : ⟪b, c ⨯ a⟫ = 1  -- Given dot product condition

theorem find_sum : s + t + u = 1 :=
sorry

end find_sum_l582_582219


namespace coefficient_of_x4_in_expansion_l582_582939

def f (x : ℝ) (n : ℝ) : ℝ := (x^2 + 1/x)^n

noncomputable def n : ℝ := 5 * (∫ x in 0..π/2, Real.cos x)

theorem coefficient_of_x4_in_expansion :
  n = 5 * (∫ x in 0..π/2, Real.cos x) →
  ∃ a : ℝ, a = 10 ∧ -- the coefficient of x^4 in the expansion of f(x, n)
  true :=  -- To simply hold the place as true.
by
  sorry  -- proof placeholder

end coefficient_of_x4_in_expansion_l582_582939


namespace no_fractional_solutions_l582_582754

theorem no_fractional_solutions (x y : ℚ) (hx : x.denom ≠ 1) (hy : y.denom ≠ 1) :
  ¬ (∃ m n : ℤ, 13 * x + 4 * y = m ∧ 10 * x + 3 * y = n) :=
sorry

end no_fractional_solutions_l582_582754


namespace unique_surjective_f_l582_582782

-- Define the problem conditions
variable (f : ℕ → ℕ)

-- Define that f is surjective
axiom surjective_f : Function.Surjective f

-- Define condition that for every m, n and prime p
axiom condition_f : ∀ m n : ℕ, ∀ p : ℕ, Nat.Prime p → (p ∣ f (m + n) ↔ p ∣ f m + f n)

-- The theorem we need to prove: the only surjective function f satisfying the condition is the identity function
theorem unique_surjective_f : ∀ x : ℕ, f x = x :=
by
  sorry

end unique_surjective_f_l582_582782


namespace find_p_from_quadratic_l582_582145

variable (p q : ℝ)
variable (p_pos q_pos : p > 0 ∧ q > 0)
variable (h : (p^2 - 4*q) = 4)

theorem find_p_from_quadratic (p q : ℝ)
  (p_pos : p > 0)
  (q_pos : q > 0)
  (h : (p^2 - 4*q) = 4) :
  p = 2 * sqrt (q + 1) :=
sorry

end find_p_from_quadratic_l582_582145


namespace lambda_value_l582_582151

-- Three non-collinear points
variables {A B C O P : Type}
-- OA, OB, and OC are vectors in some vector space
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup P]
variables [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ P]

-- Given conditions
variables (OA OC : P)
variables (PO_OP : P)
variables (O_outside_plane_ABC : ¬affine_span ℝ ({A, B, C} : Set A) O)

def OP_eq_combination : P := (1/5 : ℝ) • OA + (2/3 : ℝ) • (6/5 : ℝ) • OC

-- The point P lies on the plane ABC
axiom P_on_plane_ABC : P ∈ affine_span ℝ ({A, B, C} : Set A)

-- The goal is to prove that the above conditions implies λ = 6/5
theorem lambda_value : ∃ (λ : ℝ), OP_eq_combination = (1/5) • OA + (2/3 * λ) • OC ∧ λ = 6/5 :=
sorry

end lambda_value_l582_582151


namespace square_difference_l582_582150

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x - 2) * (x + 2) = 9797 :=
by 
  have diff_squares : (x - 2) * (x + 2) = x^2 - 4 := by ring
  rw [diff_squares, h]
  norm_num

end square_difference_l582_582150


namespace f_2012_l582_582406

noncomputable def f : ℝ → ℝ := sorry -- provided as a 'sorry' to be determined

axiom odd_function (hf : ℝ → ℝ) : ∀ x : ℝ, hf (-x) = -hf (x)

axiom f_shift : ∀ x : ℝ, f (x + 3) = -f (x)
axiom f_one : f 1 = 2

theorem f_2012 : f 2012 = 2 :=
by
  -- proofs would go here, but 'sorry' is enough to define the theorem statement
  sorry

end f_2012_l582_582406


namespace wage_recovery_raise_l582_582373

theorem wage_recovery_raise (W : ℝ) :
  let new_wage_after_cut := 0.75 * W in
  let new_wage_after_increase := 1.10 * new_wage_after_cut in
  let required_raise := (W / new_wage_after_increase - 1) in
  required_raise = 0.2121 :=
by
  sorry

end wage_recovery_raise_l582_582373


namespace initial_bacteria_count_l582_582258

theorem initial_bacteria_count :
  ∃ n : ℕ, (4 ^ 10) * n = 1_048_576 ∧ n = 1 :=
by
  use 1
  split
  · exact pow_succ' _ _ -- 4 ^ 10 = 1_048_576
  · rfl

end initial_bacteria_count_l582_582258


namespace nishita_common_shares_l582_582567

def annual_dividend_preferred_shares (num_preferred_shares : ℕ) (par_value : ℕ) (dividend_rate_preferred : ℕ) : ℕ :=
  (dividend_rate_preferred * par_value * num_preferred_shares) / 100

def annual_dividend_common_shares (total_dividend : ℕ) (dividend_preferred : ℕ) : ℕ :=
  total_dividend - dividend_preferred

def number_of_common_shares (annual_dividend_common : ℕ) (par_value : ℕ) (annual_rate_common : ℕ) : ℕ :=
  annual_dividend_common / ((annual_rate_common * par_value) / 100)

theorem nishita_common_shares (total_annual_dividend : ℕ) (num_preferred_shares : ℕ)
                             (par_value : ℕ) (dividend_rate_preferred : ℕ)
                             (semi_annual_rate_common : ℕ) : 
                             (number_of_common_shares (annual_dividend_common_shares total_annual_dividend 
                             (annual_dividend_preferred_shares num_preferred_shares par_value dividend_rate_preferred)) 
                             par_value (semi_annual_rate_common * 2)) = 3000 :=
by
  -- Provide values specific to the problem
  let total_annual_dividend := 16500
  let num_preferred_shares := 1200
  let par_value := 50
  let dividend_rate_preferred := 10
  let semi_annual_rate_common := 3.5
  sorry

end nishita_common_shares_l582_582567


namespace remainder_2023rd_term_div5_l582_582746

-- Conditions: Seq represents the modified sequence pattern
def Seq : ℕ → ℕ
| 0 := 1
| (n+1) := if h : n < ∑ k in finset.range(n + 2), k + 1 then n + 1 else (Seq n)

-- Define the problem as the remainder when the 2023rd term is divided by 5
theorem remainder_2023rd_term_div5 : (Seq 2023) % 5 = 3 := sorry

end remainder_2023rd_term_div5_l582_582746


namespace simple_interest_difference_l582_582282

-- Definitions from the problem conditions
def principal : ℝ := 2500
def rate : ℝ := 4 / 100
def time : ℝ := 5

-- Definition of simple interest calculation
def simple_interest (P R T : ℝ) : ℝ :=
  P * R * T

-- The target proof problem in Lean 4
theorem simple_interest_difference :
  principal - simple_interest principal rate time = 2000 := by
  sorry

end simple_interest_difference_l582_582282


namespace inequality_abcd_l582_582449

theorem inequality_abcd (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) :
    (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) >= 2 / 3) :=
by
  sorry

end inequality_abcd_l582_582449


namespace largest_n_for_triangle_property_l582_582741

def has_triangle_property (S : Set ℕ) : Prop :=
  ∀ {a b c : ℕ}, a ∈ S → b ∈ S → c ∈ S → a + b > c ∧ b + c > a ∧ c + a > b

theorem largest_n_for_triangle_property : 
  ∀ (n : ℕ), n < 364 ↔ ∀ (S : Set ℕ), (∀ (x : ℕ), 6 ≤ x → x ≤ n → x ∈ S) → (∀ T : Set ℕ, T ⊆ S → T.card = 10 → has_triangle_property T) :=
begin
  sorry
end

end largest_n_for_triangle_property_l582_582741


namespace CattleMarketAnimals_l582_582293

noncomputable def numAnimalsCattleMarket : ℤ :=
  let ⟨J, H, D⟩ := (7, 11, 21)
  in J + H + D

theorem CattleMarketAnimals:
  ∃ J H D : ℤ, J = 7 ∧ H = 11 ∧ D = 21 ∧ 
               (J + 5 = 2 * (H - 5)) ∧ (H + 13 = 3 * (D - 13)) ∧ (D + 3 = 6 * (J - 3)) :=
by {
  use [7, 11, 21],
  split,
  { exact rfl },
  split,
  { exact rfl },
  split,
  { exact rfl },
  repeat {sorry},
}

end CattleMarketAnimals_l582_582293


namespace maria_net_spent_l582_582321

def initial_amount := 87
def spent_on_rides := 25
def won_at_booth := 10
def spent_on_food := 12
def found_while_walking := 5
def final_amount := 16

theorem maria_net_spent :
  initial_amount - final_amount = spent_on_rides + spent_on_food - (won_at_booth + found_while_walking) :=
by
  simp
  sorry

end maria_net_spent_l582_582321


namespace kite_area_l582_582090

theorem kite_area :
  let grid_spacing := 2
  let A := (0, 6)
  let B := (4, 9)
  let C := (8, 6)
  let D := (4, 0)
  let base := 8 * grid_spacing
  let height_top := 3 * grid_spacing
  let height_bottom := 6 * grid_spacing
  let area := (1/2) * base * height_top + (1/2) * base * height_bottom
  in area = 96 := 
by
  -- Definitions
  let grid_spacing := 2
  let A := (0, 6)
  let B := (4, 9)
  let C := (8, 6)
  let D := (4, 0)
  let base := 8 * grid_spacing
  let height_top := 3 * grid_spacing
  let height_bottom := 6 * grid_spacing

  -- Area Calculation
  let area_top := (1/2 : ℝ) * base * height_top
  let area_bottom := (1/2 : ℝ) * base * height_bottom
  let area := area_top + area_bottom
  -- Proof
  have h_area_top : area_top = 48 := sorry
  have h_area_bottom : area_bottom = 48 := sorry
  calc 
    area = area_top + area_bottom : by rfl
      ... = 48 + 48 : by rw [h_area_top, h_area_bottom]
      ... = 96 : by norm_num

end kite_area_l582_582090


namespace steve_commute_l582_582260

theorem steve_commute :
  ∃ (D : ℝ), 
    (∃ (V : ℝ), 2 * V = 5 ∧ (D / V + D / (2 * V) = 6)) ∧ D = 10 :=
by
  sorry

end steve_commute_l582_582260


namespace probability_even_sum_l582_582299

theorem probability_even_sum :
  let total_outcomes := 12 * 11,
      favorable_even_even := 6 * 5,
      favorable_odd_odd := 6 * 5,
      favorable_outcomes := favorable_even_even + favorable_odd_odd,
      probability := favorable_outcomes / total_outcomes in
  probability = (5 : ℚ) / 11 :=
by
  sorry

end probability_even_sum_l582_582299


namespace belongs_to_one_progression_l582_582488

-- Define the arithmetic progression and membership property
def is_arith_prog (P : ℕ → Prop) : Prop :=
  ∃ a d, ∀ n, P (a + n * d)

-- Define the given conditions
def condition (P1 P2 P3 : ℕ → Prop) : Prop :=
  is_arith_prog P1 ∧ is_arith_prog P2 ∧ is_arith_prog P3 ∧
  (P1 1 ∨ P2 1 ∨ P3 1) ∧
  (P1 2 ∨ P2 2 ∨ P3 2) ∧
  (P1 3 ∨ P2 3 ∨ P3 3) ∧
  (P1 4 ∨ P2 4 ∨ P3 4) ∧
  (P1 5 ∨ P2 5 ∨ P3 5) ∧
  (P1 6 ∨ P2 6 ∨ P3 6) ∧
  (P1 7 ∨ P2 7 ∨ P3 7) ∧
  (P1 8 ∨ P2 8 ∨ P3 8)

-- Statement to prove
theorem belongs_to_one_progression (P1 P2 P3 : ℕ → Prop) (h : condition P1 P2 P3) : 
  P1 1980 ∨ P2 1980 ∨ P3 1980 := 
by
sorry

end belongs_to_one_progression_l582_582488


namespace problem1_tangent_line_problem2_inequality_l582_582482

-- Condition: g is derived from f and a/x - 1
def f (x : ℝ) : ℝ := Real.log x
def g (x a : ℝ) : ℝ := f x + a / x - 1

-- Problem (1) Proving a == 4 such that the tangent line criterion holds
theorem problem1_tangent_line (a : ℝ) (ha : a = 4) :
  let g' := λ x : ℝ, 1 / x - a / (x * x)
  g' 2 = -1 / 2 :=
sorry

-- Problem (2) Proving the inequality with m > n > 0
theorem problem2_inequality (m n : ℝ) (hmn : m > n ∧ n > 0) :
  (m - n) / (m + n) < (Real.log m - Real.log n) / 2 :=
sorry

end problem1_tangent_line_problem2_inequality_l582_582482


namespace factorize_polynomial_find_value_l582_582249

-- Problem 1: Factorize a^3 - 3a^2 - 4a + 12
theorem factorize_polynomial (a : ℝ) :
  a^3 - 3 * a^2 - 4 * a + 12 = (a - 3) * (a - 2) * (a + 2) :=
sorry

-- Problem 2: Given m + n = 5 and m - n = 1, prove m^2 - n^2 + 2m - 2n = 7
theorem find_value (m n : ℝ) (h1 : m + n = 5) (h2 : m - n = 1) :
  m^2 - n^2 + 2 * m - 2 * n = 7 :=
sorry

end factorize_polynomial_find_value_l582_582249


namespace number_of_people_l582_582723

-- Define the given constants
def total_cookies := 35
def cookies_per_person := 7

-- Goal: Prove that the number of people equal to 5
theorem number_of_people : total_cookies / cookies_per_person = 5 :=
by
  sorry

end number_of_people_l582_582723


namespace lily_overall_percentage_correct_l582_582198

variables (t : ℝ) -- total number of problems in the homework assignment

-- Conditions
def james_correct_alone := 0.70 * (1 / 2) * t
def james_overall_correct := 0.82 * t
def problems_solved_together := james_overall_correct - james_correct_alone

def lily_correct_alone := 0.85 * (1 / 2) * t
def lily_total_correct := lily_correct_alone + problems_solved_together

-- Question
def lily_overall_percentage : ℝ := (lily_total_correct / t) * 100

-- Correct answer
theorem lily_overall_percentage_correct : lily_overall_percentage t = 90 :=
by
  sorry

end lily_overall_percentage_correct_l582_582198


namespace log_expression_value_l582_582317

theorem log_expression_value :
  (log 3 243 / log 27 3) - (log 3 729 / log 81 3) = -9 := sorry

end log_expression_value_l582_582317


namespace trajectory_midpoint_eq_line_intersects_circle_eq_l582_582843

section GeometryProblem

variable {A B M : Type} [InnerProductSpace ℝ M]

-- Definitions of points and circles
def B : M := ⟨0, 3⟩
def circleC : Set M := {P | let ⟨x, y⟩ := P in (x + 1) ^ 2 + y ^ 2 = 4}

-- Theorem 1: Trajectory of the midpoint M
theorem trajectory_midpoint_eq :
  ∀ A M : M, 
    (A ∈ circleC) →
    ((M = (A + B) / 2) → 
    ∃ x y : ℝ, M = (⟨x, y⟩ : M) ∧ (x^2 + (y - 1.5)^2 = 1)) :=
by 
  intros A M hA hM
  sorry

-- Theorem 2: Equation of the line passing through B intersecting C
theorem line_intersects_circle_eq : 
  ∀ l k : ℝ,
  (let line_l := {P | let ⟨x, y⟩ := P in y - 3 = k * (x - 0)};
   let chord_length := dist A B = (2 * sqrt(19)) / 5;
   l ∈ line_l ∧ A ∈ circleC ∧ B ∈ circleC ∧ dist A B = chord_length →
   k = 3 + sqrt(22) / 2 ∨ k = 3 - sqrt(22) / 2) :=
by 
  intros A B hA hB
  sorry

end GeometryProblem

end trajectory_midpoint_eq_line_intersects_circle_eq_l582_582843


namespace linear_function_common_quadrants_l582_582107

theorem linear_function_common_quadrants {k b : ℝ} (h : k * b < 0) :
  (exists (q1 q2 : ℕ), q1 = 1 ∧ q2 = 4) := 
sorry

end linear_function_common_quadrants_l582_582107


namespace minimum_value_inequality_equality_condition_exists_l582_582216

theorem minimum_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  6 * c / (3 * a + b) + 6 * a / (b + 3 * c) + 2 * b / (a + c) ≥ 12 := by
  sorry

theorem equality_condition_exists : 
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (6 * c / (3 * a + b) + 6 * a / (b + 3 * c) + 2 * b / (a + c) = 12) := by
  sorry

end minimum_value_inequality_equality_condition_exists_l582_582216


namespace ants_on_cube_distance_l582_582063

-- There are 8 ants placed on the edges of a cube.
-- Ants are placed on the edges of a cube of edge length 1.

theorem ants_on_cube_distance :
  ∀ (ants : Fin 8 → (ℝ × ℝ × ℝ)),
  (∀ i, ∃ x y z, ants i = (x, y, z) ∧ (x = 0 ∨ x = 1) ∧ (y = 0 ∨ y = 1) ∧ (z = 0 ∨ z = 1)) →
  ∃ i j, i ≠ j ∧ (∃ path, 0 ≤ path ∧ path ≤ 1 ∧ path = distance ants i ants j) :=
by
  sorry

end ants_on_cube_distance_l582_582063


namespace gnoll_valid_sentences_count_l582_582590

def gnoll_words : List String := ["splargh", "glumph", "amr", "drung"]

def isValidSentence (words : List String) : Bool :=
  ∀ i, (i < words.length - 1) → (words.nthLe i sorry ≠ "splargh" ∨ words.nthLe (i + 1) sorry ≠ "glumph") ∧
       (words.nthLe i sorry ≠ "drung" ∨ words.nthLe (i + 1) sorry ≠ "amr")

def countValidSentences : Nat :=
  let allSentences := List.replicateM 3 gnoll_words
  allSentences.count isValidSentence

theorem gnoll_valid_sentences_count : countValidSentences = 48 :=
by sorry

end gnoll_valid_sentences_count_l582_582590


namespace sum_first_n_terms_l582_582400

-- Definition of the sequence terms based on given conditions
def a (n : ℕ) : ℕ := 
  if n = 1 then 2 else
  if n = 2 then 8 else
  if n = 3 then 26 else 
  -- Placeholder for general term as formula needs coefficients solved
  sorry

-- Definition of the sum of the first n terms of the sequence
def S (n : ℕ) :=
  ∑ i in Finset.range (n + 1).succ, a i

-- Statement of the mathematic problem
theorem sum_first_n_terms (n : ℕ) :
  S n = (n * n * (n + 1) * (n + 1)) / 4 :=
sorry

end sum_first_n_terms_l582_582400


namespace find_k_values_l582_582441

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem find_k_values :
  let a := 3
  let b := -(10 + k)
  let c := 4
  discriminant a b c = 0 → 
  (k = 4 * real.sqrt 3 - 10 ∨ k = -4 * real.sqrt 3 - 10) :=
begin
  let a := 3,
  let b := -(10 + k),
  let c := 4,
  assume h : discriminant a b c = 0,
  -- Use the discriminant calculation
  sorry
end

end find_k_values_l582_582441


namespace laser_path_ratio_l582_582967

def length_AB : ℝ := 13
def length_BC : ℝ := 14
def length_CA : ℝ := 15

noncomputable def perimeter_of_triangle_ABC : ℝ :=
  length_AB + length_BC + length_CA

noncomputable def ratio_of_perimeters : ℝ := 168 / 295

theorem laser_path_ratio (T ∞ : ℝ) :
  (T ∞ / perimeter_of_triangle_ABC) = ratio_of_perimeters :=
sorry

end laser_path_ratio_l582_582967


namespace ellipse_standard_eq_area_of_triangle_l582_582677

noncomputable def ellipse_eq : Prop :=
  ∃ a b c : ℝ, 
    2 * a = 10 ∧ c = 4 ∧ a = 5 ∧ b^2 = 25 - c^2 
    ∧ (a^2 = 25) ∧ (b^2 = 9) ∧ 
    ∀ x y, x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_standard_eq (a b c : ℝ) (ha : 2 * a = 10) (hc : c = 4) (h_eq : c^2 = a^2 - b^2) :
  ellipse_eq :=
sorry

noncomputable def triangle_area : ℝ :=
  let a := 5
  let c := 4 in
  ∀ P : (ℝ × ℝ), (P ∈ (C)) ∧ ( (PF₁ ⟂ PF₂)) → 
    ∃ (PF₁ PF₂ : ℝ), (PF₁ + PF₂ = 2 * a) ∧ 
    (PF₁^2 + PF₂^2 = 4 * c^2) ∧
    (PF₁ * PF₂ = 18) ∧ 
    (1 / 2 * PF₁ * PF₂ = 9)

theorem area_of_triangle (P : (ℝ × ℝ))
  (hP : ∀ P : (ℝ × ℝ), P ∈ (∑ x y, x^2 / 25 + y^2 / 9 = 1) ∧ (PF₁ ⟂ PF₂)) :
  triangle_area :=
sorry

end ellipse_standard_eq_area_of_triangle_l582_582677


namespace total_cost_l582_582366

def c_teacher : ℕ := 60
def c_student : ℕ := 40

theorem total_cost (x : ℕ) : ∃ y : ℕ, y = c_student * x + c_teacher := by
  sorry

end total_cost_l582_582366


namespace log_three_twenty_seven_sqrt_three_l582_582775

noncomputable def twenty_seven : ℝ := 27
noncomputable def sqrt_three : ℝ := Real.sqrt 3

theorem log_three_twenty_seven_sqrt_three :
  Real.logb 3 (twenty_seven * sqrt_three) = 7 / 2 :=
by
  sorry -- Proof omitted

end log_three_twenty_seven_sqrt_three_l582_582775


namespace original_price_eq_36_l582_582349

-- Definitions for the conditions
def first_cup_price (x : ℕ) : ℕ := x
def second_cup_price (x : ℕ) : ℕ := x / 2
def third_cup_price : ℕ := 3
def total_cost (x : ℕ) : ℕ := x + (x / 2) + third_cup_price
def average_price (total : ℕ) : ℕ := total / 3

-- The proof statement
theorem original_price_eq_36 (x : ℕ) (h : total_cost x = 57) : x = 36 :=
  sorry

end original_price_eq_36_l582_582349


namespace find_a_l582_582840

namespace MathProblem

noncomputable def f (x a : ℝ) := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

theorem find_a :
  (∀ x ∈ (set.Icc (0:ℝ) 2), f x a ≤ 3) → (a = 5 - Real.sqrt 10 ∨ a = 1 + Real.sqrt 2) :=
by
  sorry

end MathProblem

end find_a_l582_582840


namespace find_f_neg5_l582_582826

-- Constants and definitions
constant f : ℝ → ℝ
constant h_even : ∀ x : ℝ, f (-x) = f x
constant h_shift : ∀ x : ℝ, f (2 + x) = f (2 - x)
noncomputable def f_restricted (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : ℝ := x^2 - 2 * x

-- Proof problem statement
theorem find_f_neg5 :
  f(-5) = -1 :=
sorry

end find_f_neg5_l582_582826


namespace triangle_equilateral_l582_582811

theorem triangle_equilateral
  (a b c : ℝ)
  (h : a^4 + b^4 + c^4 - a^2 * b^2 - b^2 * c^2 - a^2 * c^2 = 0) :
  a = b ∧ b = c ∧ a = c := 
by
  sorry

end triangle_equilateral_l582_582811


namespace proof_minimum_initial_good_dwarves_l582_582288

/-- There are 2012 dwarves, each either good or bad. Every day they attend a meeting in groups
  of 3 or 5. During each meeting, if the majority are good, all attendees of the meeting become good;
  if the majority are bad, all attendees become bad. If after the third day's meetings all 2012 dwarves 
  have become good, we show the minimum number of good dwarves before the first day's meetings is 435. -/
def minimum_initial_good_dwarves : ℕ :=
  let initial_count := 435 in
  initial_count

theorem proof_minimum_initial_good_dwarves :
  let good_dwarves := (3 : ℕ) 
  ∀ dwarves_total : ℕ, 
    dwarves_total = 2012 → 
    (∀ meeting_size : ℕ, meeting_size ∈ {3, 5} → 
      (∀ majority : ℕ, majority = (meeting_size / 2).ceil → 
        (∀ day := 3, 
          if ∀ all_good_dwarves : ℕ, all_good_dwarves = dwarves_total → 
              initial_count ≥ good_dwarves 
            then true 
            else false))) :=
sorry

end proof_minimum_initial_good_dwarves_l582_582288


namespace set_union_identity_l582_582850

theorem set_union_identity (a b : ℤ) (h : A = \{-1, a\}) (h1 : B = \{2^a, b\}) (h2 : A ∩ B = \{1\}) :
  A ∪ B = \{-1, 1, 2\} :=
begin
  sorry
end

end set_union_identity_l582_582850


namespace find_x_log_eq_l582_582871

theorem find_x_log_eq (b x : ℝ) (h₀ : b > 0) (h₁ : b ≠ 1) (h₂ : x ≠ 1) :
  (log (x) / log (b^3) + log (b) / log (x^3) = 1) :=
by {
  sorry
}

end find_x_log_eq_l582_582871


namespace truck_total_distance_l582_582020

noncomputable def truck_distance (b t : ℝ) : ℝ :=
  let acceleration := b / 3
  let time_seconds := 300 + t
  let distance_feet := (1 / 2) * (acceleration / t) * time_seconds^2
  distance_feet / 5280

theorem truck_total_distance (b t : ℝ) : 
  truck_distance b t = b * (90000 + 600 * t + t ^ 2) / (31680 * t) :=
by
  sorry

end truck_total_distance_l582_582020


namespace sunglasses_and_cap_probability_l582_582961

/-
On a beach:
  - 50 people are wearing sunglasses.
  - 35 people are wearing caps.
  - The probability that randomly selected person wearing a cap is also wearing sunglasses is 2/5.
  
Prove that the probability that a randomly selected person wearing sunglasses is also wearing a cap is 7/25.
-/

theorem sunglasses_and_cap_probability :
  let total_sunglasses := 50
  let total_caps := 35
  let cap_with_sunglasses_probability := (2 : ℚ) / 5
  let both := cap_with_sunglasses_probability * total_caps
  (both / total_sunglasses) = (7 : ℚ) / 25 :=
by
  -- definitions
  let total_sunglasses := 50
  let total_caps := 35
  let cap_with_sunglasses_probability := (2 : ℚ) / 5
  let both := cap_with_sunglasses_probability * (total_caps : ℚ)
  have prob : (both / (total_sunglasses : ℚ)) = (7 : ℚ) / 25 := sorry
  exact prob

end sunglasses_and_cap_probability_l582_582961


namespace find_k_values_l582_582208

noncomputable def possible_values_of_k (a b c : ℝ → ℝ) : set ℝ :=
  {k | k = (2 * real.sqrt 3) / 3 ∨ k = -(2 * real.sqrt 3) / 3}

theorem find_k_values (a b c : vector ℝ ℝ) (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) 
 (h3 : ∥c∥ = 1) (h4 : a ⬝ b = 0) (h5 : a ⬝ c = 0)
 (h6 : real.angle b c = real.pi / 3) : 
 ∃ k, a = k • (b × c) ∧ k ∈ possible_values_of_k a b c :=
 sorry

end find_k_values_l582_582208


namespace experiments_needed_proof_l582_582654

noncomputable def experiments_needed (factor : ℝ) (target_ratio : ℝ) : ℕ :=
  let n := (Real.log target_ratio) / (Real.log factor) in
  Int.toNat (Int.floor n)

theorem experiments_needed_proof :
  experiments_needed 0.618 0.618^4 = 4 :=
by
  rw [experiments_needed, Real.log_pow, ←Real.log_pow 0.618 4]
  sorry

end experiments_needed_proof_l582_582654


namespace quadratic_root_b_l582_582809

theorem quadratic_root_b (b c : ℝ) (i : ℂ) (h_eq : 1 + 0 * i = (2 + i).re) (h_eq1 : 0 = (2 + i).im) :
  (x^2 + b * x + c = 0) ∧ (2 + i) ∈ set_of (λ z, ∃ a b c, a * z^2 + b * z + c = 0) →
  b = -4 := sorry

end quadratic_root_b_l582_582809


namespace find_a_b_l582_582852

variables {α : Type*}

noncomputable def A (a : ℝ) : set (ℝ × ℝ) := {p | p.2 = a * p.1 + 1}
noncomputable def B (b : ℝ) : set (ℝ × ℝ) := {p | p.2 = p.1 + b}

theorem find_a_b (a b : ℝ)
  (hA : ∀ x y, (x, y) ∈ A a ↔ y = a * x + 1)
  (hB : ∀ x y, (x, y) ∈ B b ↔ y = x + b)
  (h_intersection : (2, 5) ∈ A a ∩ B b) :
  a + b = 5 :=
by
  sorry

end find_a_b_l582_582852


namespace gloria_payment_l582_582384

theorem gloria_payment
  (P N E : ℕ)
  (h1 : P + N = 80)
  (h2 : P + E = 45)
  (h3 : 3 * P + 3 * N + 3 * E = 315) :
  N + E = 85 := 
begin
  -- proof part here
  sorry
end

end gloria_payment_l582_582384


namespace six_digit_permutations_divisible_l582_582013

theorem six_digit_permutations_divisible (N : ℕ) :
  (N ≥ 10^5 ∧ N < 10^6) ∧ (∀ i j : ℕ, i ≠ j → i ≠ 0 ∧ j ≠ 0 ∧ (N.digit i ≠ N.digit j)) ∧ (37 ∣ N) →
  ∃ (S : finset ℕ), S.card ≥ 24 ∧ (∀ M ∈ S, 37 ∣ M) ∧ (∀ M ∈ S, ∀ i : ℕ, i < 10^6 → N.digit i = M.digit i) :=
by
  sorry

end six_digit_permutations_divisible_l582_582013


namespace quilt_percentage_shaded_l582_582989

theorem quilt_percentage_shaded :
  ∀ (total_squares full_shaded half_shaded quarter_shaded : ℕ),
    total_squares = 25 →
    full_shaded = 4 →
    half_shaded = 8 →
    quarter_shaded = 4 →
    ((full_shaded + half_shaded * 1 / 2 + quarter_shaded * 1 / 2) / total_squares * 100 = 40) :=
by
  intros
  sorry

end quilt_percentage_shaded_l582_582989


namespace n_must_be_multiple_of_3_l582_582779

theorem n_must_be_multiple_of_3
  (n : ℕ) (hn : n ≥ 3) 
  (a : Fin n → ℝ) 
  (h1 : a ⟨n + 1, hn.trans (zero_add 1).le⟩ = a 0) 
  (h2 : a ⟨n + 2, hn.trans (zero_add 2).le⟩ = a 1) 
  (h3 : ∀ i : Fin n, a i * (a ⟨i + 1 % n, sorry⟩) + 1 = a ⟨i + 2 % n, sorry⟩):
  ∃ k : ℕ, n = 3 * k := 
sorry

end n_must_be_multiple_of_3_l582_582779


namespace color_x_green_l582_582230

def color := {red, green, blue}

structure Triangle :=
  (side1 : color)
  (side2 : color)
  (side3 : color)

def sides_colored_correctly (t : Triangle) : Prop :=
  -- Each triangle side must have one red, one green, and one blue side
  t.side1 ≠ t.side2 ∧ t.side2 ≠ t.side3 ∧ t.side1 ≠ t.side3

variables (a e b d c x : color)

-- Given assumptions
axiom h1 : a = color.red
axiom h2 : e = color.red
axiom h3 : b = color.blue
axiom h4 : d = color.green
axiom h5 : c = color.red

-- The statement to prove
theorem color_x_green : x = color.green :=
by
  -- proof goes here
  sorry

end color_x_green_l582_582230


namespace sum_of_possible_values_of_a_l582_582752

theorem sum_of_possible_values_of_a :
  (∀ r s : ℤ, r + s = a ∧ r * s = 3 * a) → ∃ a : ℤ, (a = 12) :=
by
  sorry

end sum_of_possible_values_of_a_l582_582752


namespace parabola_eq_and_product_slopes_ratio_l582_582471

open Real

noncomputable def parabola_equation : Prop :=
  ∀ (P Q : ℝ×ℝ) (x1 y1 x2 y2 : ℝ), 
    -- Conditions:
    (P = (x1, y1)) → 
    (Q = (x2, y2)) → 
    -- The vertex is at the origin and the directrix is x = 1
    ((y1^2 = -4 * x1) ∧ (y2^2 = -4 * x2)) ∧ 
    -- Equation of line through A(-2, 0) intersecting parabola at P, Q 
    ((x1 = y1^2 / 4) ∧ (x2 = y2^2 / 4)) →
    -- Statement to prove:
    (y1 * y2 = -8 : Prop)

noncomputable def slopes_ratio_constant: Prop :=
  ∀ (P Q M N : ℝ×ℝ) (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) (k1 k2 : ℝ),
    -- Conditions:
    (P = (x1, y1)) → 
    (Q = (x2, y2)) → 
    (M = (x3, y3)) → 
    (N = (x4, y4)) → 
    ((x1 = y1^2 / 4) ∧ ((x2 = y2^2 / 4) ∧ (x3 = y3^2 / 4) ∧ (x4 = y4^2 / 4))) →
    ((k1 = (y1 - y2) / (x1 - x2)) ∧ (k2 = (y3 - y4) / (x3 - x4))) →
    -- Statement to prove:
    (k1 / k2 = 1 / 2 : Prop)

theorem parabola_eq_and_product (P Q : ℝ×ℝ) (x1 y1 x2 y2 : ℝ) :
  parabola_equation P Q x1 y1 x2 y2 :=
sorry

theorem slopes_ratio (P Q M N : ℝ×ℝ) (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) (k1 k2 : ℝ) :
  slopes_ratio_constant P Q M N x1 y1 x2 y2 x3 y3 x4 y4 k1 k2 :=
sorry

end parabola_eq_and_product_slopes_ratio_l582_582471


namespace coefficient_of_x2_in_expansion_l582_582529

noncomputable def binom_expansion_coefficient_x2 : ℤ :=
  let coeff := -20 in
  coeff

theorem coefficient_of_x2_in_expansion :
  let x := ℚ
  let y := ℚ
  (∀ (x y : ℚ), coeff_of_term (x - 2 * (y ^ 3)) ((x + 1 / y) ^ 5) x^2 = -20) := sorry

end coefficient_of_x2_in_expansion_l582_582529


namespace sum_of_quadrilateral_angles_l582_582076

-- Define the conditions as given in part a
def side_lengths : List ℕ := [15, 20, 25, 33]
def angles : List ℕ := [100, 130, 105, 125]

-- The main theorem statement
theorem sum_of_quadrilateral_angles (a b c d : ℕ) :
  a = 100 → b = 130 → c = 105 → d = 125 →
  a + b + c + d = 360 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  exact Nat.add_assoc 100 130 (105 + 125) ▸ 
        Nat.add_assoc 100 (130 + 105) 125 ▸ 
        Nat.add_comm 130 105 ▸ 
        Nat.add_assoc (105 + 130) 100 125 ▸
        rfl

end sum_of_quadrilateral_angles_l582_582076


namespace pi_times_difference_of_volumes_correct_l582_582379

structure Cylinder where
  circumference : ℝ
  height : ℝ

def volume (C : Cylinder) : ℝ :=
  let r := C.circumference / (2 * Real.pi)
  Real.pi * r^2 * C.height

def Amy_sheet : Cylinder := 
  ⟨10, 7⟩

def Belinda_sheet : Cylinder := 
  ⟨12, 9⟩

theorem pi_times_difference_of_volumes_correct :
  Real.pi * (volume Belinda_sheet - volume Amy_sheet) = 149 := by
  sorry

end pi_times_difference_of_volumes_correct_l582_582379


namespace point_p_trajectory_is_right_branch_of_hyperbola_l582_582460

noncomputable theory

-- Define the points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the condition on the distances
def distance_condition (P : ℝ × ℝ) : Prop :=
  (real.sqrt ((P.1 - M.1) ^ 2 + (P.2 - M.2) ^ 2) - 
   real.sqrt ((P.1 - N.1) ^ 2 + (P.2 - N.2) ^ 2)) = 3

-- Define the trajectory of point P
def trajectory (P : ℝ × ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * P.1 ^ 2 - b * P.2 ^ 2 = 1 ∧ P.1 > 0

-- The problem statement to prove in Lean 4
theorem point_p_trajectory_is_right_branch_of_hyperbola (P : ℝ × ℝ) :
  distance_condition P → trajectory P :=
begin
  sorry
end

end point_p_trajectory_is_right_branch_of_hyperbola_l582_582460


namespace problem_1_problem_2_l582_582561

noncomputable def f (x a : ℝ) := log x - a * x
noncomputable def g (x a : ℝ) := exp x - a * x

theorem problem_1 (a : ℝ) : 
    (∀ x > 1, (1/x - a) < 0) ∧ 
    (∃ c ∈ Ioi (1 : ℝ), (g c a) = 0 ∧ (∀ x > 1, (1/x - a) < 0)) → 
    (a ∈ Ioi real.exp) := 
sorry

theorem problem_2 (a : ℝ) :
    (∀ x > -1, exp x - a > 0) → 
    (if a ≤ 0 ∨ a = real.exp⁻¹ then
        (∃! x, x > 0 ∧ f x a = 0)
    else if 0 < a ∧ a < real.exp⁻¹ then 
        (∃ x₁ x₂, x₁ < x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f x₁ a = 0 ∧ f x₂ a = 0)) := 
sorry

end problem_1_problem_2_l582_582561


namespace delaney_travel_time_l582_582047

def bus_leaves_at := 8 * 60
def delaney_left_at := 7 * 60 + 50
def missed_by := 20

theorem delaney_travel_time
  (bus_leaves_at : ℕ) (delaney_left_at : ℕ) (missed_by : ℕ) :
  delaney_left_at + (bus_leaves_at + missed_by - bus_leaves_at) - delaney_left_at = 30 :=
by
  exact sorry

end delaney_travel_time_l582_582047


namespace right_triangle_side_lengths_l582_582280

theorem right_triangle_side_lengths (a b c : ℝ) (varrho r : ℝ) (h_varrho : varrho = 8) (h_r : r = 41) : 
  (a = 80 ∧ b = 18 ∧ c = 82) ∨ (a = 18 ∧ b = 80 ∧ c = 82) :=
by
  sorry

end right_triangle_side_lengths_l582_582280


namespace sum_of_coefficients_l582_582457

noncomputable def u : ℕ → ℕ
| 0       => 5
| (n + 1) => u n + (3 + 4 * (n - 1))

theorem sum_of_coefficients :
  (2 + -3 + 6) = 5 :=
by {
  sorry
}

end sum_of_coefficients_l582_582457


namespace probability_of_satisfying_condition_l582_582866

-- Let p be an integer between 1 and 15 inclusive
def is_valid_p (p : ℤ) : Prop := 1 ≤ p ∧ p ≤ 15

-- Define the equation condition
def satisfies_equation (p q : ℤ) : Prop := p * q - 5 * p - 3 * q = 3

-- Define the probability question: probability that there exists q such that p and q satisfy the equation
theorem probability_of_satisfying_condition : 
  (∃ p ∈ { p : ℤ | is_valid_p p }, ∃ q : ℤ, satisfies_equation p q) →
  (finset.filter (λ p : ℤ, ∃ q : ℤ, satisfies_equation p q) (finset.Icc 1 15)).card / (finset.Icc 1 15).card = 1 / 3 :=
sorry

end probability_of_satisfying_condition_l582_582866


namespace median_of_siblings_list_l582_582169

def siblings_list : List ℕ := [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6]

theorem median_of_siblings_list : List.median siblings_list = 3 := sorry

end median_of_siblings_list_l582_582169


namespace find_coefficients_l582_582211

-- Define the polynomial
def polynomial (a b : ℝ) (x : ℂ) : ℂ := x^3 + (a : ℂ) * x^2 - 2 * x + (b : ℂ)

-- Given conditions
variables (a b : ℝ)
variable (h_root : polynomial a b (2 - 3 * complex.I) = 0)

-- Statement of the theorem
theorem find_coefficients :
  (a, b) = (-1 / 4, 195 / 4) :=
sorry

end find_coefficients_l582_582211


namespace product_Q_roots_l582_582845

-- Definitions of the polynomials P(x) and Q(x)
def P (x : ℂ) : ℂ := x^5 - x^2 + 1
def Q (x : ℂ) : ℂ := x^2 + 1

-- Given that r_1, r_2, ..., r_5 are the roots of P(x)
axiom roots_r : ∃ (r : Fin 5 → ℂ), ∀ j, P (r j) = 0

-- The statement to be proved
theorem product_Q_roots :
  let r : Fin 5 → ℂ := Classical.choose roots_r in
  ∏ j, Q (r j) = 5 :=
by
  let r : Fin 5 → ℂ := Classical.choose roots_r
  sorry

end product_Q_roots_l582_582845


namespace cory_fruits_l582_582748

theorem cory_fruits (apples oranges bananas grapes days : ℕ)
  (h_apples : apples = 4)
  (h_oranges : oranges = 3)
  (h_bananas : bananas = 2)
  (h_grapes : grapes = 1)
  (h_days : days = 10)
  : ∃ ways : ℕ, ways = Nat.factorial days / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas * Nat.factorial grapes) ∧ ways = 12600 :=
by
  sorry

end cory_fruits_l582_582748


namespace exist_unique_xy_solution_l582_582099

theorem exist_unique_xy_solution :
  ∃! (x y : ℝ), x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3 ∧ x = 1 / 3 ∧ y = 2 / 3 :=
by
  sorry

end exist_unique_xy_solution_l582_582099


namespace roots_of_equation_l582_582412

theorem roots_of_equation :
  ∀ x : ℝ, (x^4 + x^2 - 20 = 0) ↔ (x = 2 ∨ x = -2) :=
by
  -- This will be the proof.
  -- We are claiming that x is a root of the polynomial if and only if x = 2 or x = -2.
  sorry

end roots_of_equation_l582_582412


namespace comprehensive_equation_l582_582035

def combination_of_equations : Prop :=
  (15 ÷ 5 = 3) ∧ (24 - 3 = 21) → (24 - (15 ÷ 5) = 24 - 3)

theorem comprehensive_equation (h : combination_of_equations) :
  24 - (15 ÷ 5) = 24 - 3 :=
by {
  sorry, 
}

end comprehensive_equation_l582_582035


namespace initial_yellow_hard_hats_count_l582_582523

noncomputable def initial_yellow_hard_hats := 24

theorem initial_yellow_hard_hats_count
  (initial_pink: ℕ)
  (initial_green: ℕ)
  (carl_pink: ℕ)
  (john_pink: ℕ)
  (john_green: ℕ)
  (total_remaining: ℕ)
  (remaining_pink: ℕ)
  (remaining_green: ℕ)
  (initial_yellow: ℕ) :
  initial_pink = 26 →
  initial_green = 15 →
  carl_pink = 4 →
  john_pink = 6 →
  john_green = 2 * john_pink →
  total_remaining = 43 →
  remaining_pink = initial_pink - carl_pink - john_pink →
  remaining_green = initial_green - john_green →
  initial_yellow = total_remaining - remaining_pink - remaining_green →
  initial_yellow = initial_yellow_hard_hats :=
by
  intros
  sorry

end initial_yellow_hard_hats_count_l582_582523


namespace probability_a_plus_ab_plus_abc_divisible_by_3_l582_582575

theorem probability_a_plus_ab_plus_abc_divisible_by_3 :
  let S := finset.range (2013 + 1)  -- the set {1, 2, ..., 2013}
  let count_multiples_of_3 (n : ℕ) : ℕ := finset.card (finset.filter (λ x, x % 3 = 0) (finset.range (n + 1)))
  let probability_div_by_3 (n : ℕ) : ℚ := (count_multiples_of_3 n).to_rat / n.to_rat
  ∃ (P : ℚ), P = (probability_div_by_3 2013) + (2/3 * (2/9)) :=
  P = (13 / 27) :=
by
  -- Proof Steps and Calculations would go here
  sorry

end probability_a_plus_ab_plus_abc_divisible_by_3_l582_582575


namespace janet_total_gas_l582_582542

-- Definitions / Conditions
def distance_dermatologist := 60
def distance_gynecologist := 80
def distance_cardiologist := 100
def mpg := 15
def extra_gas_dermatologist := 0.5
def extra_gas_gynecologist := 0.7
def extra_gas_cardiologist := 1.0

-- Total distance calculation
def total_distance : ℝ := 
  2 * distance_dermatologist + 2 * distance_gynecologist + 2 * distance_cardiologist

-- Normal gas usage calculation
def normal_gas_usage : ℝ := total_distance / mpg

-- Total extra gas consumption calculation
def total_extra_gas : ℝ := 
  extra_gas_dermatologist + extra_gas_gynecologist + extra_gas_cardiologist

-- Total gas calculation
def total_gas_usage : ℝ := normal_gas_usage + total_extra_gas

-- Lean statement proving the total gas used
theorem janet_total_gas : total_gas_usage = 34.2 :=
by
  -- exact calculation is omitted here for the sake of structure
  sorry

end janet_total_gas_l582_582542


namespace empty_can_weight_l582_582201

theorem empty_can_weight :
  (let num_soda_cans := 6
       soda_weight := 12
       total_weight := 88
       additional_empty_cans := 2 in
   let total_cans := num_soda_cans + additional_empty_cans
       soda_total_weight := num_soda_cans * soda_weight
       empty_cans_weight := total_weight - soda_total_weight in
   empty_cans_weight / total_cans = 2) :=
by sorry

end empty_can_weight_l582_582201


namespace small_cube_edge_length_l582_582257

theorem small_cube_edge_length
  (large_cube_volume : ℝ)
  (num_small_cubes : ℝ)
  (h1 : large_cube_volume = 1000)
  (h2 : num_small_cubes = 8) :
  ∃ (edge_length : ℝ), edge_length = 5 :=
by
  -- Define the volume of a small cube based on the given conditions
  let small_cube_volume := large_cube_volume / num_small_cubes
  -- Assert the cube root of the small cube's volume is the edge length
  let edge_length := Real.cbrt small_cube_volume
  use edge_length
  sorry

end small_cube_edge_length_l582_582257


namespace sheelas_income_l582_582253

variable {Rs: ℝ}

variables (Income Savings Investment LivingExpenses Tax HealthInsurance: ℝ)

def monthly_income (Income Savings: ℝ): Prop :=
  Savings = 0.32 * Income ∧ Income = 3800 / 0.32

def total_net_saving (Income Savings Investment: ℝ): Prop :=
  Investment = 0.15 * Income ∧ Savings = 3800 ∧
  0.32 * Income = 3800 → TotalSavings = Savings + Investment ∧ TotalSavings = 5581.25

theorem sheelas_income :
  monthly_income 11875 3800 →
  total_net_saving 11875 3800 1781.25 := 
  by
  sorry

end sheelas_income_l582_582253


namespace unique_integer_n_l582_582500

theorem unique_integer_n (n : ℤ) (h : ⌊(n^2 : ℚ) / 5⌋ - ⌊(n / 2 : ℚ)⌋^2 = 3) : n = 5 :=
  sorry

end unique_integer_n_l582_582500


namespace Jane_shopping_oranges_l582_582541

theorem Jane_shopping_oranges 
  (o a : ℕ)
  (h1 : a + o = 5)
  (h2 : 30 * a + 45 * o + 20 = n)
  (h3 : ∃ k : ℕ, n = 100 * k) : 
  o = 2 :=
by
  sorry

end Jane_shopping_oranges_l582_582541


namespace f_neg_one_f_four_l582_582220

-- Define the function f
def f (x : ℝ) : ℝ :=
if x < 0 then 4 * x - 2 else 12 - 3 * x

-- State the proof problems
theorem f_neg_one : f (-1) = -6 := by
  sorry

theorem f_four : f (4) = 0 := by
  sorry

end f_neg_one_f_four_l582_582220


namespace ryan_recruit_people_l582_582971

noncomputable def total_amount_needed : ℕ := 1000
noncomputable def amount_already_have : ℕ := 200
noncomputable def average_funding_per_person : ℕ := 10
noncomputable def additional_funding_needed : ℕ := total_amount_needed - amount_already_have
noncomputable def number_of_people_recruit : ℕ := additional_funding_needed / average_funding_per_person

theorem ryan_recruit_people : number_of_people_recruit = 80 := by
  sorry

end ryan_recruit_people_l582_582971


namespace intersection_x_coordinate_l582_582265

theorem intersection_x_coordinate (k b : ℝ) (h : k ≠ b) :
  (∃ x y : ℝ, y = k * x + b ∧ y = b * x + k) → (∃ x : ℝ, x = 1) :=
by
  intro h_intersect
  cases h_intersect with x h_xy
  cases h_xy with y h_1
  use 1
  sorry

end intersection_x_coordinate_l582_582265


namespace primes_with_difference_and_concatenation_l582_582599

open Nat

/-- Define the two primes and the concatenated number relationship. -/
def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ (∀ d, d ∣ n → d = 1 ∨ d = n)

noncomputable def concatenate (a b : ℕ) : ℕ := a * 10 ^ (Nat.log10 b + 1) + b

theorem primes_with_difference_and_concatenation :
  ∃ (p q : ℕ), p < q ∧ is_prime p ∧ is_prime q ∧ q - p = 100 ∧ is_prime (concatenate p q) := 
begin
  existsi 3,
  existsi 103,
  split,
  { exact nat.lt_succ_self 102 }, -- proving p < q
  split,
  { split, -- proving p = 3 is prime
    norm_num,
    { intros d hd, fin_cases hd; norm_num } },
  split,
  { split, -- proving q = 103 is prime
    norm_num,
    { intros d hd, fin_cases hd; norm_num }},
  split,
  { norm_num }, -- proving q - p = 100
  { split, -- proving concatenated number is prime
    norm_num,
    { intros d hd, fin_cases hd,
      { norm_num },
      { norm_num },
      { exact nat.prime_def_lt'.mpr ⟨dec_trivial, λ d hd, by norm_num⟩ } } }
end

end primes_with_difference_and_concatenation_l582_582599


namespace borrowed_amount_l582_582697

theorem borrowed_amount (P : ℝ) 
    (borrow_rate : ℝ := 4) 
    (lend_rate : ℝ := 6) 
    (borrow_time : ℝ := 2) 
    (lend_time : ℝ := 2) 
    (gain_per_year : ℝ := 140) 
    (h₁ : ∀ (P : ℝ), P / 8.333 - P / 12.5 = 280) 
    : P = 7000 := 
sorry

end borrowed_amount_l582_582697


namespace largest_consecutive_set_having_triangle_property_l582_582744

noncomputable def max_n_with_triangle_property : ℕ :=
  let S := {6, 7, 13, 20, 33, 53, 86, 139, 225, 364}
  363

theorem largest_consecutive_set_having_triangle_property :
  ∀ S : set ℕ, {6, 7, 8, ..., max_n_with_triangle_property} ⊆ S →
  (∀ s ⊆ S, s.card = 10 →  (∀ a b c ∈ s, a + b > c ∧ b + c > a ∧ a + c > b)) :=
sorry

end largest_consecutive_set_having_triangle_property_l582_582744


namespace framed_painting_ratio_l582_582343

-- Definitions and conditions
def painting_width : ℕ := 20
def painting_height : ℕ := 30
def frame_side_width (x : ℕ) : ℕ := x
def frame_top_bottom_width (x : ℕ) : ℕ := 3 * x

-- Overall dimensions of the framed painting
def framed_painting_width (x : ℕ) : ℕ := painting_width + 2 * frame_side_width x
def framed_painting_height (x : ℕ) : ℕ := painting_height + 2 * frame_top_bottom_width x

-- Area of the painting
def painting_area : ℕ := painting_width * painting_height

-- Area of the frame
def frame_area (x : ℕ) : ℕ := framed_painting_width x * framed_painting_height x - painting_area

-- Condition that frame area equals painting area
def frame_area_condition (x : ℕ) : Prop := frame_area x = painting_area

-- Theoretical ratio of smaller to larger dimension of the framed painting
def dimension_ratio (x : ℕ) : ℚ := (framed_painting_width x : ℚ) / (framed_painting_height x)

-- The mathematical problem to prove
theorem framed_painting_ratio : ∃ x : ℕ, frame_area_condition x ∧ dimension_ratio x = (4 : ℚ) / 7 :=
by
  sorry

end framed_painting_ratio_l582_582343


namespace angle_C_max_l582_582879

theorem angle_C_max (A B C : ℝ) (h_triangle : A + B + C = Real.pi)
  (h_cond : Real.sin B / Real.sin A = 2 * Real.cos (A + B))
  (h_max_B : B = Real.pi / 3) :
  C = 2 * Real.pi / 3 :=
by
  sorry

end angle_C_max_l582_582879


namespace find_multiple_l582_582251

variable R Remy_g : ℕ
variable (M : ℕ)

theorem find_multiple (h1 : Remy_g = M * R + 1) (h2 : R + Remy_g = 33) (h3 : Remy_g = 25) :
  M = 3 := by
  sorry

end find_multiple_l582_582251


namespace f_has_one_zero_max_integer_value_of_a_for_g_increasing_l582_582838

noncomputable def f (x : ℝ) := (x - 2) * Real.log x + 2 * x - 3

theorem f_has_one_zero (h : ∀ x : ℝ, x ≥ 1) : ∃! x ≥ 1, f x = 0 :=
  sorry

noncomputable def g (x a : ℝ) := (x - a) * Real.log x + a * (x - 1) / x

theorem max_integer_value_of_a_for_g_increasing :
  ∀ (x : ℝ), (x ≥ 1) → ∀ a : ℝ, (∀ x : ℝ, (x ≥ 1) → (Real.log x + 1 - a / x + a / x ^ 2) ≥ 0)
  → (a : ℤ) = 6 :=
  sorry

end f_has_one_zero_max_integer_value_of_a_for_g_increasing_l582_582838


namespace p_or_q_not_necessarily_true_l582_582167

theorem p_or_q_not_necessarily_true (p q : Prop) (hnp : ¬p) (hpq : ¬(p ∧ q)) : ¬(p ∨ q) ∨ (p ∨ q) :=
by
  sorry

end p_or_q_not_necessarily_true_l582_582167


namespace max_strings_cut_volleyball_net_l582_582402

-- Define the structure of a volleyball net with 10x20 cells where each cell is divided into 4 triangles.
structure VolleyballNet : Type where
  -- The dimensions of the volleyball net
  rows : ℕ
  cols : ℕ
  -- Number of nodes (vertices + centers)
  nodes : ℕ
  -- Maximum number of strings (edges) connecting neighboring nodes that can be cut without disconnecting the net
  max_cut_without_disconnection : ℕ

-- Define the specific volleyball net in question
def volleyball_net : VolleyballNet := 
  { rows := 10, 
    cols := 20, 
    nodes := (11 * 21) + (10 * 20), -- vertices + center nodes
    max_cut_without_disconnection := 800 
  }

-- The main theorem stating that we can cut these strings without the net falling apart
theorem max_strings_cut_volleyball_net (net : VolleyballNet) 
    (h_dim : net.rows = 10) 
    (h_dim2 : net.cols = 20) :
  net.max_cut_without_disconnection = 800 :=
sorry -- The proof is omitted

end max_strings_cut_volleyball_net_l582_582402


namespace john_total_marks_l582_582202

noncomputable def total_marks : ℕ :=
  let q1 := 50
  let r1 := 0.85
  let q2 := 60
  let r2 := 0.70
  let q3 := 40
  let r3 := 0.95
  let d := 0.25
  let correct1 := real.to_nat (r1 * q1)
  let correct2 := real.to_nat (r2 * q2)
  let correct3 := real.to_nat (r3 * q3)
  let incorrect1 := q1 - correct1
  let incorrect2 := q2 - correct2
  let incorrect3 := q3 - correct3
  let total_correct := correct1 + correct2 + correct3
  let total_incorrect := incorrect1 + incorrect2 + incorrect3
  let marks_deducted := d * total_incorrect
  let total_marks := total_correct - marks_deducted
  total_marks

theorem john_total_marks : total_marks = 115 :=
by sorry

end john_total_marks_l582_582202


namespace coefficient_of_x2_y4_l582_582898

-- Define the given functions
def f (x y : ℂ) : ℂ := (1/x - 2*y) * (2*x + y)^5

-- State the theorem
theorem coefficient_of_x2_y4 (x y : ℂ) : (coeff (x^2 * y^4) (f x y)) = -80 := 
by sorry

end coefficient_of_x2_y4_l582_582898


namespace check_right_angled_triangle_l582_582322

def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem check_right_angled_triangle :
  ¬is_right_angled_triangle 1.1 1.5 1.9 ∧
  ¬is_right_angled_triangle 5 11 12 ∧
  is_right_angled_triangle 1.2 1.6 2.0 ∧
  ¬is_right_angled_triangle 3 4 8 :=
by
  split
  sorry -- Proof for ¬is_right_angled_triangle 1.1 1.5 1.9
  split
  sorry -- Proof for ¬is_right_angled_triangle 5 11 12
  split
  sorry -- Proof for is_right_angled_triangle 1.2 1.6 2.0
  sorry -- Proof for ¬is_right_angled_triangle 3 4 8

end check_right_angled_triangle_l582_582322


namespace eval_cbrt_8_p6_l582_582066

theorem eval_cbrt_8_p6 : (real.cbrt 8) ^ 6 = 64 := 
begin
    sorry,
end

end eval_cbrt_8_p6_l582_582066


namespace min_x_plus_y_l582_582131

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y ≥ 9 :=
sorry

end min_x_plus_y_l582_582131


namespace slope_range_l582_582354

theorem slope_range {A : ℝ × ℝ} (k : ℝ) : 
  A = (1, 1) → (0 < 1 - k ∧ 1 - k < 2) → -1 < k ∧ k < 1 :=
by
  sorry

end slope_range_l582_582354


namespace factorization_l582_582071

theorem factorization (a : ℝ) : 2 * a ^ 2 - 8 = 2 * (a + 2) * (a - 2) := 
by
  sorry

end factorization_l582_582071


namespace heather_walking_distance_l582_582490

theorem heather_walking_distance :
  let dist1 := 0.33
  let dist2 := 0.33
  let dist3 := 0.08
  dist1 + dist2 + dist3 = 0.74 :=
by 
  let dist1 := 0.33
  let dist2 := 0.33
  let dist3 := 0.08
  calc
    dist1 + dist2 + dist3 = 0.33 + 0.33 + 0.08 : by rfl
    ... = 0.66 + 0.08 : by norm_num
    ... = 0.74 : by norm_num

end heather_walking_distance_l582_582490


namespace lowest_two_digit_number_whose_digits_product_is_12_l582_582659

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 <= n ∧ n < 100 ∧ ∃ d1 d2 : ℕ, 1 ≤ d1 ∧ d1 < 10 ∧ 1 ≤ d2 ∧ d2 < 10 ∧ n = 10 * d1 + d2 ∧ d1 * d2 = 12

theorem lowest_two_digit_number_whose_digits_product_is_12 :
  ∃ n : ℕ, is_valid_two_digit_number n ∧ ∀ m : ℕ, is_valid_two_digit_number m → n ≤ m ∧ n = 26 :=
sorry

end lowest_two_digit_number_whose_digits_product_is_12_l582_582659


namespace total_amount_l582_582248

theorem total_amount (rahul_days rajesh_days : ℕ) (rahul_share : ℕ) (h_rahul_days : rahul_days = 3) (h_rajesh_days : rajesh_days = 2) (h_rahul_share : rahul_share = 100) :
  let rahul_work := 1 / (rahul_days : ℝ),
      rajesh_work := 1 / (rajesh_days : ℝ),
      total_work := rahul_work + rajesh_work,
      ratio_rahul_rajesh := rahul_work / rajesh_work,
      total_parts := 2 + 3,
      part_value := rahul_share / 2
  in total_work = 5 / 6 ∧ ratio_rahul_rajesh = 2 / 3 ∧ total_parts * part_value = 250 :=
by
  sorry

end total_amount_l582_582248


namespace plane_intersects_unit_cubes_l582_582353

-- Definitions:
def isLargeCube (cube : ℕ × ℕ × ℕ) : Prop := cube = (4, 4, 4)
def isUnitCube (size : ℕ) : Prop := size = 1

-- The main theorem we want to prove:
theorem plane_intersects_unit_cubes :
  ∀ (cube : ℕ × ℕ × ℕ) (plane : (ℝ × ℝ × ℝ) → ℝ),
  isLargeCube cube →
  (∀ point : ℝ × ℝ × ℝ, plane point = 0 → 
       ∃ (x y z : ℕ), x < 4 ∧ y < 4 ∧ z < 4 ∧ 
                     (x, y, z) ∈ { coords : ℕ × ℕ × ℕ | true }) →
  (∃ intersects : ℕ, intersects = 16) :=
by
  intros cube plane Hcube Hplane
  sorry

end plane_intersects_unit_cubes_l582_582353


namespace feathers_per_flamingo_l582_582564

theorem feathers_per_flamingo (num_boa : ℕ) (feathers_per_boa : ℕ) (num_flamingoes : ℕ) (pluck_rate : ℚ)
  (total_feathers : ℕ) (feathers_per_flamingo : ℕ) :
  num_boa = 12 →
  feathers_per_boa = 200 →
  num_flamingoes = 480 →
  pluck_rate = 0.25 →
  total_feathers = num_boa * feathers_per_boa →
  total_feathers = num_flamingoes * feathers_per_flamingo * pluck_rate →
  feathers_per_flamingo = 20 :=
by
  intros h_num_boa h_feathers_per_boa h_num_flamingoes h_pluck_rate h_total_feathers h_feathers_eq
  sorry

end feathers_per_flamingo_l582_582564


namespace Meghan_total_money_l582_582952

theorem Meghan_total_money (h100 : ℕ) (h50 : ℕ) (h10 : ℕ) : 
  h100 = 2 → h50 = 5 → h10 = 10 → 100 * h100 + 50 * h50 + 10 * h10 = 550 :=
by
  sorry

end Meghan_total_money_l582_582952


namespace tan_C_l582_582535

variable (A B C : Type) [metric_space A]

noncomputable def is_right_triangle (ABC : A × B × C) : Prop :=
  ∃ (A B C : A), ∠A B C = π / 2

variable (ABC : is_right_triangle (A, B, C))
variable (AB : ℝ) (AC : ℝ)

variable [hAB : AB = 5] [hAC : AC = real.sqrt 34]

theorem tan_C : tan (angle B C) = 3 / 5 :=
by
  sorry

end tan_C_l582_582535


namespace weight_in_pounds_approx_l582_582621

-- Define the conversion factor
def kilogram_to_pound : ℝ := 1 / 0.9072

-- Define the initial weight in kilograms
def weight_kg : ℝ := 350

-- State the theorem for the final weight in pounds to the nearest whole number
theorem weight_in_pounds_approx : 
  (weight_kg * kilogram_to_pound).round = 386 :=
sorry

end weight_in_pounds_approx_l582_582621


namespace B_wins_coloring_strategy_l582_582301

theorem B_wins_coloring_strategy :
  ∀ (A B : Type) (turn : ℕ → A ⊕ B) (color : ℕ → option bool),
    (∀ n, 1 ≤ n ∧ n ≤ 2019) →
    (∀ n, (color n ≠ none → ∀ m, abs (m - n) = 1 → color m = color n)) →
    (∀ k, (∃ n, color n = none) → turn k = sum.inl A → (∀ n, color n = none → color (2020 - n) ≠ color n)) →
    (∃ n, color n = none) → sum.inr B wins :=
begin
  sorry
end

end B_wins_coloring_strategy_l582_582301


namespace verify_statements_l582_582096

noncomputable def f (x : ℝ) : ℝ := 10 ^ x

theorem verify_statements (x1 x2 : ℝ) (h : x1 ≠ x2) :
  (f (x1 + x2) = f x1 * f x2) ∧
  (f x1 - f x2) / (x1 - x2) > 0 :=
by
  sorry

end verify_statements_l582_582096


namespace tan_eq_2sqrt3_over_3_l582_582869

theorem tan_eq_2sqrt3_over_3 (θ : ℝ) (h : 2 * Real.cos (θ - Real.pi / 3) = 3 * Real.cos θ) : 
  Real.tan θ = 2 * Real.sqrt 3 / 3 :=
by 
  sorry -- Proof is omitted as per the instructions

end tan_eq_2sqrt3_over_3_l582_582869


namespace optimal_prob_win_at_least_4_of_8_l582_582995

theorem optimal_prob_win_at_least_4_of_8 :
  let n := 8
  let p := 1 / 2
  (∑ k in Finset.range (n + 1), if 4 ≤ k then (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) else 0) = 163 / 256 := by
  sorry

end optimal_prob_win_at_least_4_of_8_l582_582995


namespace ratio_a_to_d_l582_582506

theorem ratio_a_to_d (a b c d : ℕ) 
  (h1 : a * 4 = b * 3) 
  (h2 : b * 9 = c * 7) 
  (h3 : c * 7 = d * 5) : 
  a * 3 = d := 
sorry

end ratio_a_to_d_l582_582506


namespace smallest_area_of_2020th_square_l582_582015

theorem smallest_area_of_2020th_square :
  ∃ (S : ℤ) (A : ℕ), 
    (S * S - 2019 = A) ∧ 
    (∃ k : ℕ, k * k = A) ∧ 
    (∀ (T : ℤ) (B : ℕ), ((T * T - 2019 = B) ∧ (∃ l : ℕ, l * l = B)) → (A ≤ B)) :=
sorry

end smallest_area_of_2020th_square_l582_582015


namespace lineEF_not_tangent_to_A_excircle_l582_582931

-- Define the triangle with given properties
variables (α β γ : Type) [MetricSpace α] [MetricSpace β] [MetricSpace γ]
variable (A B C : α)
variable (E F : β)
variable [Field γ] (angle_B_obtuse: ∃ θ : γ, θ = Angle A B C ∧ θ > 90)

-- Assume the feet of altitudes
variable (altitude_B: foot B A E)
variable (altitude_C: foot C A F)

-- Prove that line EF is not tangent to the A-excircle
theorem lineEF_not_tangent_to_A_excircle (h : ¬ tangent (line E F) (excircle A B C)) : 
  ¬ isTangent (line E F) (A_excircle A B C) :=
begin
  sorry
end

end lineEF_not_tangent_to_A_excircle_l582_582931


namespace bonifac_distance_l582_582294

/-- Given the conditions provided regarding the paths of Pankrác, Servác, and Bonifác,
prove that the total distance Bonifác walked is 625 meters. -/
theorem bonifac_distance
  (path_Pankrac : ℕ)  -- distance of Pankráč's path in segments
  (meters_Pankrac : ℕ)  -- distance Pankráč walked in meters
  (path_Bonifac : ℕ)  -- distance of Bonifác's path in segments
  (meters_per_segment : ℚ)  -- meters per segment walked
  (Hp : path_Pankrac = 40)  -- Pankráč's path in segments
  (Hm : meters_Pankrac = 500)  -- Pankráč walked 500 meters
  (Hms : meters_per_segment = 500 / 40)  -- meters per segment
  (Hb : path_Bonifac = 50)  -- Bonifác's path in segments
  : path_Bonifac * meters_per_segment = 625 := sorry

end bonifac_distance_l582_582294


namespace geom_seq_sum_seven_terms_l582_582433

-- Defining the conditions
def a0 : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 7

-- Definition for the sum of the first n terms in a geometric series
def geom_series_sum (a r : ℚ) (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

-- Statement to prove the sum of the first seven terms equals 1093/2187
theorem geom_seq_sum_seven_terms : geom_series_sum a0 r n = 1093 / 2187 := 
by 
  sorry

end geom_seq_sum_seven_terms_l582_582433


namespace pizza_fractions_l582_582336

/-- Given conditions for the pizza and pepperoni circles. -/
variables (r circle_area pizza_area : ℝ)
variables (n m : ℕ)

/-- Nine pepperoni circles fit exactly across the diameter of an 18-inch pizza -/
def diameter := 18
def radius := diameter / (2 * n)

/-- Calculation to determine the radius of pepperoni circles -/
def pepperoni_circle_radius (diam : ℝ) (n : ℕ) : ℝ := diam / (2 * n)

/-- Calculation for area of one pepperoni circle given radius -/
def pepperoni_circle_area (r : ℝ) : ℝ := Real.pi * r ^ 2

/-- Calculation for total area of 36 pepperoni circles -/
def total_pepperoni_area (area : ℝ) (m : ℕ) : ℝ := m * area

/-- Calculation for area of the pizza given pizza radius -/
def pizza_area (r : ℝ) : ℝ := Real.pi * r ^ 2

/-- Fraction of the pizza covered by pepperoni -/
def fraction_covered (pepperoni_area pizza_area : ℝ) : ℝ := pepperoni_area / pizza_area

theorem pizza_fractions (h₁ : r = pepperoni_circle_radius diameter 9) 
                        (h₂ : circle_area = pepperoni_circle_area r) 
                        (h₃ : pizza_area = pizza_area (diameter / 2))
                        (h₄ : m = 36)
                        (pepperoni_area := total_pepperoni_area circle_area m)
                        (fraction := fraction_covered pepperoni_area pizza_area):
  fraction = 4 / 9 := by
  sorry

end pizza_fractions_l582_582336


namespace largest_expression_value_l582_582670

theorem largest_expression_value :
  let A := 2 + 0 + 1 + 7,
      B := 2 * 0 + 1 + 7,
      C := 2 + 0 * 1 + 7,
      D := 2 + 0 + 1 * 7,
      E := 2 * 0 * 1 * 7
  in max A (max B (max C (max D E))) = 10 := 
by
  let A := 2 + 0 + 1 + 7
  let B := 2 * 0 + 1 + 7
  let C := 2 + 0 * 1 + 7
  let D := 2 + 0 + 1 * 7
  let E := 2 * 0 * 1 * 7
  sorry

end largest_expression_value_l582_582670


namespace arithmetic_sequence_fraction_zero_l582_582818

noncomputable def arithmetic_sequence_term (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_fraction_zero (a1 d : ℝ) 
    (h1 : a1 ≠ 0) (h9 : arithmetic_sequence_term a1 d 9 = 0) :
  (arithmetic_sequence_term a1 d 1 + 
   arithmetic_sequence_term a1 d 8 + 
   arithmetic_sequence_term a1 d 11 + 
   arithmetic_sequence_term a1 d 16) / 
  (arithmetic_sequence_term a1 d 7 + 
   arithmetic_sequence_term a1 d 8 + 
   arithmetic_sequence_term a1 d 14) = 0 :=
by
  sorry

end arithmetic_sequence_fraction_zero_l582_582818


namespace pens_left_is_25_l582_582001

def total_pens_left (initial_blue initial_black initial_red removed_blue removed_black : Nat) : Nat :=
  let blue_left := initial_blue - removed_blue
  let black_left := initial_black - removed_black
  let red_left := initial_red
  blue_left + black_left + red_left

theorem pens_left_is_25 :
  total_pens_left 9 21 6 4 7 = 25 :=
by 
  rw [total_pens_left, show 9 - 4 = 5 from Nat.sub_eq_of_eq_add (rfl), show 21 - 7 = 14 from Nat.sub_eq_of_eq_add (rfl)]
  rfl

end pens_left_is_25_l582_582001


namespace probability_heart_king_l582_582649

theorem probability_heart_king :
  let total_cards := 52
  let total_kings := 4
  let hearts_count := 13
  let king_of_hearts := 1 in
  let prob_king_of_hearts_first := (1 : ℚ) / total_cards
  let prob_other_heart_first := (hearts_count - king_of_hearts : ℚ) / total_cards
  let prob_king_second_if_king_heart_first := (total_kings - king_of_hearts : ℚ) / (total_cards - 1)
  let prob_king_second_if_other_heart_first := (total_kings : ℚ) / (total_cards - 1) in
  prob_king_of_hearts_first * prob_king_second_if_king_heart_first +
  prob_other_heart_first * prob_king_second_if_other_heart_first = (1 : ℚ) / total_cards :=
by sorry

end probability_heart_king_l582_582649


namespace amount_paid_to_z_l582_582329

-- Definitions based on conditions
def work_per_day (d : ℕ) : ℚ := 1 / d

def x_work_rate : ℚ := work_per_day 15
def y_work_rate : ℚ := work_per_day 10

-- Total payment and collective work rate definitions
def total_payment : ℚ := 720
def combined_work_rate : ℚ := work_per_day 5

-- Needed as part of the definition that will be used in the proof
def z_work_rate : ℚ := combined_work_rate - x_work_rate - y_work_rate

-- Verification statement to show the amount paid to z is Rs. 120
theorem amount_paid_to_z :
  z_work_rate * total_payment / (x_work_rate + y_work_rate + z_work_rate) = 120 := 
by
  sorry

end amount_paid_to_z_l582_582329


namespace covered_squares_count_l582_582684

theorem covered_squares_count :
  let diameter := 4
  let radius := diameter / 2
  let board_size := 10
  let square_size := 1
  ∀ (disc_center checkerboard_center : ℝ × ℝ) (center_check: disc_center = checkerboard_center)
  (disc_center = (board_size/2, board_size/2))
  (checkerboard_center = (board_size/2, board_size/2)),
  let covered_squares := 4 * 12 in
  covered_squares = 48 :=
by
  let diameter := 4
  let radius := diameter / 2
  let board_size := 10
  let square_size := 1
  assume disc_center checkerboard_center : ℝ × ℝ
  assume center_check: disc_center = checkerboard_center
  have hc: disc_center = (board_size/2, board_size/2) := sorry
  have hc2: checkerboard_center = (board_size/2, board_size/2) := sorry
  let covered_squares := 4 * 12
  show covered_squares = 48, from sorry

end covered_squares_count_l582_582684


namespace my_cousin_reading_time_l582_582236

-- Define the conditions
def reading_time_me_hours : ℕ := 3
def reading_speed_ratio : ℕ := 5
def reading_time_me_min : ℕ := reading_time_me_hours * 60

-- Define the statement to be proved
theorem my_cousin_reading_time : (reading_time_me_min / reading_speed_ratio) = 36 := by
  sorry

end my_cousin_reading_time_l582_582236


namespace no_distinct_pairs_l582_582053

theorem no_distinct_pairs : 
  ∀ (x y : ℤ), 0 < x ∧ x < y ∧ real.sqrt 2500 = real.sqrt x + 2 * real.sqrt y → False :=
by
  assume x y,
  intro h,
  sorry

end no_distinct_pairs_l582_582053


namespace squares_area_ratios_l582_582403

noncomputable def squareC_area (x : ℝ) : ℝ := x ^ 2
noncomputable def squareD_area (x : ℝ) : ℝ := 3 * x ^ 2
noncomputable def squareE_area (x : ℝ) : ℝ := 6 * x ^ 2

theorem squares_area_ratios (x : ℝ) (h : x ≠ 0) :
  (squareC_area x / squareE_area x = 1 / 36) ∧ (squareD_area x / squareE_area x = 1 / 4) := by
  sorry

end squares_area_ratios_l582_582403


namespace arithmetic_sequence_a5_value_l582_582528

variable {a_n : ℕ → ℝ}

theorem arithmetic_sequence_a5_value
  (h : a_n 2 + a_n 8 = 15 - a_n 5) :
  a_n 5 = 5 :=
sorry

end arithmetic_sequence_a5_value_l582_582528


namespace problem1_problem2_l582_582123

-- Define the function f(x)
def f (x : Real) (a : Real) : Real := (sin x - a) * (a - cos x) + Real.sqrt 2 * a

-- Problem 1: Prove the range of f(x) when a = 1 is [-3/2, sqrt(2)]
theorem problem1 :
  (∀ x : Real, 0 ≤ x ∧ x ≤ Real.pi → -3 / 2 ≤ f x 1 ∧ f x 1 ≤ Real.sqrt 2) ∧
  (∃ x1 x2 : Real, 0 ≤ x1 ∧ x1 ≤ Real.pi ∧ f x1 1 = -3 / 2 ∧ 0 ≤ x2 ∧ x2 ≤ Real.pi ∧ f x2 1 = Real.sqrt 2) :=
sorry

-- Problem 2: Prove the range of a such that f(x) has exactly one zero point in [0, π] is either 1 ≤ a < sqrt(2) + 1 or a = sqrt(2) + sqrt(6)/2

theorem problem2 :
  (∀ a : Real, 1 ≤ a → (∃! x : Real, 0 ≤ x ∧ x ≤ Real.pi ∧ f x a = 0) ↔ (1 ≤ a ∧ a < Real.sqrt 2 + 1 ∨ a = Real.sqrt 2 + Real.sqrt 6 / 2)) :=
sorry

end problem1_problem2_l582_582123


namespace bicycle_saves_time_l582_582539

-- Define the conditions
def time_to_walk : ℕ := 98
def time_saved_by_bicycle : ℕ := 34

-- Prove the question equals the answer
theorem bicycle_saves_time :
  time_saved_by_bicycle = 34 := 
by
  sorry

end bicycle_saves_time_l582_582539


namespace sum_of_differences_l582_582006

-- State the given quantities
def dog_food := 600
def cat_food := 327
def bird_food := 415
def fish_food := 248

-- Define the differences
def diff_dog_cat := dog_food - cat_food
def diff_cat_bird := bird_food - cat_food
def diff_bird_fish := bird_food - fish_food

-- Prove the sum of the differences
theorem sum_of_differences : 
  diff_dog_cat + diff_cat_bird + diff_bird_fish = 528 := 
by
  have h1 : diff_dog_cat = dog_food - cat_food := rfl
  have h2 : diff_cat_bird = bird_food - cat_food := rfl
  have h3 : diff_bird_fish = bird_food - fish_food := rfl
  calc 
    diff_dog_cat + diff_cat_bird + diff_bird_fish 
      = (dog_food - cat_food) + (bird_food - cat_food) + (bird_food - fish_food) : by rw [h1, h2, h3]
    ... = 273 + 88 + 167 : by simp
    ... = 528 : by norm_num

end sum_of_differences_l582_582006


namespace types_of_tessellating_polygons_l582_582496

theorem types_of_tessellating_polygons : 
  ∃ (n : ℕ), n = 3 ∧ (∀ (p : ℕ), p ∈ {3, 4, 6} ↔ (360 % (180 - (360 / p)) = 0)) :=
begin
  -- Proof goes here
  sorry
end

end types_of_tessellating_polygons_l582_582496


namespace question1_question2_question3_l582_582524

-- Define the conditions
def condition1 (a b c : ℝ) : Prop := (a + c + b) * (a + c - b) = (2 + Real.sqrt 3) * a * c
def condition2 (A C : ℝ) : Prop := Real.cos A + Real.sin C = Real.sqrt 6 / 2
def condition3 (b : ℝ) : Prop := b = Real.sqrt 6 - Real.sqrt 2

-- Question 1: Prove that B = π / 6 given condition1
theorem question1 (a b c : ℝ) (h1 : condition1 a b c) : B = π / 6 :=
sorry

-- Question 2: Prove the area of triangle ABC is 1 given B = π / 6, condition2, and condition3
theorem question2 (a b c A C : ℝ) (hB : B = π / 6) (h2 : condition2 A C) (h3 : condition3 b) : area a b c = 1 :=
sorry

-- Question 3: Prove the range of cos A + sin C given conditions
theorem question3 (A C : ℝ) (h2 : condition2 A C) : Real.sqrt 3 / 2 < Real.cos A + Real.sin C ∧ Real.cos A + Real.sin C < 3 / 2 :=
sorry

end question1_question2_question3_l582_582524


namespace temp_diff_not_exactly_3_l582_582370

variable {V : ℝ} -- volume of the vessel
variable {T_c T_h : ℝ} -- initial temperature of cold water (T_c) and temperature of hot water (T_h)

-- Definition: Temperature after n procedures
def temperature_after_n_procedures (n : ℕ) : ℝ :=
  T_h - (1 / 4) ^ n * (T_h - T_c)

-- Given: The increase in temperature after the first procedure is 16 degrees
axiom initial_temp_inc (ΔT : ℝ) : ΔT = (1 / 4) * (T_h - T_c)
axiom temp_diff : T_h - T_c = 64 / 3

-- Question A: Find n such that the temperature difference is 0.5 degrees
def exists_procedure_count_for_temp_diff : ∃ (n : ℕ), ∥temperature_after_n_procedures n - T_h∥ = 0.5 := sorry

-- Question B: Prove it's impossible for the temperature difference to be exactly 3 degrees
theorem temp_diff_not_exactly_3 : ¬ ∃ (n : ℕ), ∥temperature_after_n_procedures n - T_h∥ = 3 := sorry

end temp_diff_not_exactly_3_l582_582370


namespace shortest_tangent_segment_length_l582_582927

noncomputable def C1 := {p : ℝ × ℝ | (p.1 - 12)^2 + p.2^2 = 49}
noncomputable def C2 := {p : ℝ × ℝ | (p.1 + 18)^2 + p.2^2 = 64}

theorem shortest_tangent_segment_length :
  ∃ P Q : ℝ × ℝ, P ∈ C1 ∧ Q ∈ C2 ∧ ∀ R S : ℝ × ℝ, R ∈ C1 ∧ S ∈ C2 → ℝ.dist P Q ≤ ℝ.dist R S →
    ℝ.dist P Q = 30 :=
by sorry

end shortest_tangent_segment_length_l582_582927


namespace radius_of_perpendicular_intersection_l582_582999

theorem radius_of_perpendicular_intersection 
  (r : ℝ) (hr : r > 0)
  (d : ℝ) (hd : d = (√2 / 2) * r) :
  (3 * 0 - 4 * 0 - 1 = d) →
  r = √2 / 5 :=
by 
  intro h
  have h_dist := calc 
    d = (√2 / 2) * r : hd
  sorry

end radius_of_perpendicular_intersection_l582_582999


namespace largest_n_unique_k_l582_582655

theorem largest_n_unique_k :
  ∃ (n : ℕ), (∀ (k1 k2 : ℕ), 
    (9 / 17 < n / (n + k1) → n / (n + k1) < 8 / 15 → 9 / 17 < n / (n + k2) → n / (n + k2) < 8 / 15 → k1 = k2) ∧ 
    n = 72) :=
sorry

end largest_n_unique_k_l582_582655


namespace minimum_blocks_needed_l582_582344

-- Definitions of the conditions
def wall_length : ℝ := 120
def wall_height : ℝ := 8
def block_height : ℝ := 1
def block_length1 : ℝ := 3
def block_length2 : ℝ := 1

-- The statement to prove
theorem minimum_blocks_needed :
  ∀ (wall_length wall_height block_height block_length1 block_length2 : ℝ)
  (h1 : wall_length = 120)
  (h2 : wall_height = 8)
  (h3 : block_height = 1)
  (h4 : block_length1 = 3)
  (h5 : block_length2 = 1),
  (minimum_blocks wall_length wall_height block_height block_length1 block_length2) = 324 :=
by 
  sorry

end minimum_blocks_needed_l582_582344


namespace remainder_of_product_mod_12_l582_582310

-- Define the given constants
def a := 1125
def b := 1127
def c := 1129
def d := 12

-- State the conditions as Lean hypotheses
lemma mod_eq_1125 : a % d = 9 := by sorry
lemma mod_eq_1127 : b % d = 11 := by sorry
lemma mod_eq_1129 : c % d = 1 := by sorry

-- Define the theorem to prove
theorem remainder_of_product_mod_12 : (a * b * c) % d = 3 := by
  -- Use the conditions stated above to prove the theorem
  sorry

end remainder_of_product_mod_12_l582_582310


namespace color_triangle_vertices_no_same_color_l582_582864

-- Define the colors and the vertices
inductive Color | red | green | blue | yellow
inductive Vertex | A | B | C 

-- Define a function that counts ways to color the triangle given constraints
def count_valid_colorings (colors : List Color) (vertices : List Vertex) : Nat := 
  -- There are 4 choices for the first vertex, 3 for the second, 2 for the third
  4 * 3 * 2

-- The theorem we want to prove
theorem color_triangle_vertices_no_same_color : count_valid_colorings [Color.red, Color.green, Color.blue, Color.yellow] [Vertex.A, Vertex.B, Vertex.C] = 24 := by
  sorry

end color_triangle_vertices_no_same_color_l582_582864


namespace recursive_relation_l582_582503

noncomputable def f (n : ℕ) : ℕ := ∑ i in finRange(2 * n + 1), i ^ 2

theorem recursive_relation (k : ℕ) : 
    f (k + 1) = f k + (2 * k + 1) ^ 2 + (2 * k + 2) ^ 2 := 
by
  sorry

end recursive_relation_l582_582503


namespace poly_mul_expansion_l582_582776

noncomputable def poly1 := (λ z : ℚ, 3 * z^2 - 4 * z + 1)
noncomputable def poly2 := (λ z : ℚ, 4 * z^3 + z^2 - 5 * z + 3)
noncomputable def result := (λ z : ℚ, 12 * z^5 + 3 * z^4 + 32 * z^3 + z^2 - 7 * z + 3)

theorem poly_mul_expansion (z : ℚ) :
  (poly1 z) * (poly2 z) = result z :=
by sorry

end poly_mul_expansion_l582_582776


namespace pow_1999_mod_26_l582_582309

theorem pow_1999_mod_26 (n : ℕ) (h1 : 17^1 % 26 = 17)
  (h2 : 17^2 % 26 = 17) (h3 : 17^3 % 26 = 17) : 17^1999 % 26 = 17 := by
  sorry

end pow_1999_mod_26_l582_582309


namespace no_non_periodic_function_exists_l582_582061

noncomputable theory

def non_periodic (f : ℝ → ℝ) : Prop :=
  ∀ T > 0, ∃ x, f (x + T) ≠ f x

def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f x * (f x + 1)

theorem no_non_periodic_function_exists : 
  ¬ ∃ f : ℝ → ℝ, non_periodic f ∧ functional_eq f :=
by sorry

end no_non_periodic_function_exists_l582_582061


namespace max_size_T_l582_582553

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def T (S : Set (Set ℕ)) : Prop :=
  ∀ {a b x y : ℕ} (h1 : {a, b} ⊆ M) (h2 : {x, y} ⊆ M) (h3 : {a, b} ≠ {x, y})
  ({a, b}, {x, y} ∈ S),
  (a * x + b * y) * (a * y + b * x) % 11 ≠ 0

theorem max_size_T (S : Set (Set ℕ)) (hT : T S) : ∃ n, n = 25 :=
sorry

end max_size_T_l582_582553


namespace Diego_more_than_half_Martha_l582_582944

theorem Diego_more_than_half_Martha (M D : ℕ) (H1 : M = 90)
  (H2 : D > M / 2)
  (H3 : M + D = 145):
  D - M / 2 = 10 :=
by
  sorry

end Diego_more_than_half_Martha_l582_582944


namespace smallest_integer_n_l582_582877

theorem smallest_integer_n (n : ℕ) (h : ∃ k : ℕ, 432 * n = k ^ 2) : n = 3 := 
sorry

end smallest_integer_n_l582_582877


namespace function_shape_is_graph_l582_582966

/-- Define the Cartesian coordinate system and the concept of graphing a function -/
def is_graph_of_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, (x, f x) ∈ set_of (λ p : ℝ × ℝ, p.2 = f p.1)

/-- Prove that the shape formed by points representing function f in Cartesian coordinate system is the graph of f -/
theorem function_shape_is_graph (f : ℝ → ℝ) :
  ∀ x y : ℝ, (x, y) ∈ set_of (λ p : ℝ × ℝ, p.2 = f p.1) ↔ y = f x :=
by
  sorry

end function_shape_is_graph_l582_582966


namespace consecutive_good_air_quality_l582_582990

def air_quality_index : ℕ → ℕ
| 1 := 90
| 2 := 95
| 3 := 110
| 4 := 130
| 5 := 80
| 6 := 70
| 7 := 120
| 8 := 140
| 9 := 85
| 10 := 105

def good_air_quality (index : ℕ) : Prop :=
  index < 100

theorem consecutive_good_air_quality:
  (good_air_quality (air_quality_index 1) ∧ good_air_quality (air_quality_index 2)) ∧
  (good_air_quality (air_quality_index 5) ∧ good_air_quality (air_quality_index 6)) :=
by
  sorry

end consecutive_good_air_quality_l582_582990


namespace tan_5105_eq_tan_85_l582_582732

noncomputable def tan_deg (d : ℝ) := Real.tan (d * Real.pi / 180)

theorem tan_5105_eq_tan_85 :
  tan_deg 5105 = tan_deg 85 := by
  have eq_265 : tan_deg 5105 = tan_deg 265 := by sorry
  have eq_neg : tan_deg 265 = tan_deg 85 := by sorry
  exact Eq.trans eq_265 eq_neg

end tan_5105_eq_tan_85_l582_582732


namespace common_terms_count_l582_582491

/-- The sequences and their parameters. --/
def seq1 (n : ℕ) : ℕ := 2 + (n - 1) * 3
def seq2 (m : ℕ) : ℕ := 4 + (m - 1) * 5

/-- Find the number of common terms between the two sequences up to their boundaries. --/
theorem common_terms_count :
  let count := ∃ t : ℕ, 1 ≤ t ∧ t ≤ 134 ∧ (∃ n m : ℕ, seq1 (5 * t) = seq2 (3 * t)) in
  finset.card (finset.filter (λ t, 1 ≤ t ∧ t ≤ 134 ∧ (∃ n m : ℕ, seq1 (5 * t) = seq2 (3 * t))) (finset.range 135)) = 134 :=
by
  sorry

end common_terms_count_l582_582491


namespace log_cos_eq_half_m_minus_n_l582_582127

def A := sorry
def sin_A := Real.sin A
def cos_A := Real.cos A
def lg := Real.log10

variable {A : Real}
variable (is_acute : 0 < A ∧ A < π / 2)
variable (m n : Real)
variable (h1 : lg (1 + sin_A) = m)
variable (h2 : lg (1 / (1 - sin_A)) = n)

theorem log_cos_eq_half_m_minus_n 
  (is_acute : 0 < A ∧ A < π / 2) 
  (h1 : lg (1 + sin_A) = m) 
  (h2 : lg (1 / (1 - sin_A)) = n) : 
  lg (cos_A) = (1 / 2) * (m - n) := 
by
  sorry

end log_cos_eq_half_m_minus_n_l582_582127


namespace g_of_73_l582_582044

def g : ℤ → ℤ
| n => if n ≥ 500 then n - 3 else g (g (n + 7))

theorem g_of_73 : g 73 = 497 := by sorry

end g_of_73_l582_582044


namespace number_of_bracelets_l582_582566

-- Define the conditions as constants
def metal_beads_nancy := 40
def pearl_beads_nancy := 60
def crystal_beads_rose := 20
def stone_beads_rose := 40
def beads_per_bracelet := 2

-- Define the number of sets each person can make
def sets_of_metal_beads := metal_beads_nancy / beads_per_bracelet
def sets_of_pearl_beads := pearl_beads_nancy / beads_per_bracelet
def sets_of_crystal_beads := crystal_beads_rose / beads_per_bracelet
def sets_of_stone_beads := stone_beads_rose / beads_per_bracelet

-- Define the theorem to prove
theorem number_of_bracelets : min sets_of_metal_beads (min sets_of_pearl_beads (min sets_of_crystal_beads sets_of_stone_beads)) = 10 := by
  -- Placeholder for the proof
  sorry

end number_of_bracelets_l582_582566


namespace squared_sum_of_a_b_l582_582854

theorem squared_sum_of_a_b (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 3) : (a + b) ^ 2 = 16 :=
by
  sorry

end squared_sum_of_a_b_l582_582854


namespace dry_grapes_weight_l582_582442

theorem dry_grapes_weight 
  (fresh_weight : ℝ)
  (fresh_water_content : ℝ)
  (dried_water_content : ℝ)
  (fresh_weight_condition : fresh_weight = 5)
  (fresh_water_content_condition : fresh_water_content = 0.90)
  (dried_water_content_condition : dried_water_content = 0.20) :
  let non_water_weight := fresh_weight * (1 - fresh_water_content) in
  let dried_grapes_weight := non_water_weight / (1 - dried_water_content) in
  dried_grapes_weight = 0.625 :=
by
  sorry

end dry_grapes_weight_l582_582442


namespace S_2014_l582_582532

noncomputable def a : ℕ → ℝ
| 1     := 1
| (n+1) := a n + Real.sin ((n+1) * Real.pi / 2)

noncomputable def S : ℕ → ℝ
| 0     := 0
| (n+1) := S n + a (n+1)

theorem S_2014 : S 2014 = 1008 := 
by
  sorry

end S_2014_l582_582532


namespace part_a_part_b_l582_582254

open Real

-- Definitions for the geometric setup
structure Square (A B C D : Point) extends ConvexQuadrilateral A B C D := 
  (ABCD_is_square : ∀ (a b c d : ℝ), Distance a b = Distance b c ∧ 
                   Distance b c = Distance c d ∧ 
                   Distance c d = Distance d a ∧ 
                   ∠ A B C = 90 ∧ 
                   ∠ B C D = 90 ∧
                   ∠ C D A = 90 ∧
                   ∠ D A B = 90)

structure Circle (Omega : Set Point) :=
  (center : Point)
  (radius : ℝ)
  (circle_eq : ∀ x, x ∈ Omega ↔ Distance center x = radius)

-- Given Square and Circle
variables (A B C D : Point) (Omega : Set Point)
variables [Square A B C D] [Circle Omega] 

-- Intersection points
variables (E F G H I J K L : Point)
variable arcs_intersect : ∀ u v w x y z t s ∈ Omega, 
  intersects A B C D Omega [u v, w x, y z, t s] 

-- Define the curvilinear triangles
structure CurvilinearTriangle (P Q R : Set Point) :=
  (arc : Set Point)
  (arc_in_circle : arc ⊆ Omega)

noncomputable def Triangle_AEF : CurvilinearTriangle A E F := sorry
noncomputable def Triangle_BGH : CurvilinearTriangle B G H := sorry
noncomputable def Triangle_CIJ : CurvilinearTriangle C I J := sorry
noncomputable def Triangle_DKL : CurvilinearTriangle D K L := sorry

-- Define lengths and perimeters
def length_arc {P Q : Point} (arc : Set Point) := sorry

def Perimeter {P Q R : Point} (T : CurvilinearTriangle P Q R) :=
  length_arc T.arc + Distance P Q + Distance Q R + Distance R P

-- The two main theorems to be proved
theorem part_a : 
  length_arc Triangle_AEF.arc + length_arc Triangle_CIJ.arc = 
  length_arc Triangle_BGH.arc + length_arc Triangle_DKL.arc := 
  sorry 

theorem part_b : 
  Perimeter Triangle_AEF + Perimeter Triangle_CIJ = 
  Perimeter Triangle_BGH + Perimeter Triangle_DKL := 
  sorry

end part_a_part_b_l582_582254


namespace no_fractional_x_y_l582_582758

theorem no_fractional_x_y (x y : ℚ) (H1 : ¬ (x.denom = 1 ∧ y.denom = 1)) (H2 : ∃ m : ℤ, 13 * x + 4 * y = m) (H3 : ∃ n : ℤ, 10 * x + 3 * y = n) : false :=
sorry

end no_fractional_x_y_l582_582758


namespace circumcircles_cover_quadrilateral_l582_582963

-- Define the points and properties described in the conditions
variables {A B C D E F G H P : Type} [MetricSpace P]
variables (A B C D E F G H : P)

-- Define the convex quadrilateral
def is_convex (ABCD : Quadrilateral P) : Prop :=
  convex_hull (insert A (insert B (insert C (insert D ∅)))) = set_of {x : P | x is_in_convex_hull_of [A, B, C, D]}

-- Define the points E, F, G and H on the respective sides
def points_on_sides : Prop :=
  E ∈ line_segment A B ∧
  F ∈ line_segment B C ∧
  G ∈ line_segment C D ∧
  H ∈ line_segment D A

-- Definition of the circumcircle covering point
def circumcircle_cover (O : P) : Prop :=
  point_inside_circumcircle H A E O ∨
  point_inside_circumcircle E B F O ∨
  point_inside_circumcircle F C G O ∨
  point_inside_circumcircle G D H O

-- The main theorem statement
theorem circumcircles_cover_quadrilateral (O : P) :
  is_convex ⟨A, B, C, D⟩ →
  points_on_sides A B C D E F G H →
  O ∈ convex_hull (insert A (insert B (insert C (insert D ∅)))) →
  circumcircle_cover A B C D E F G H O :=
begin
  sorry
end

end circumcircles_cover_quadrilateral_l582_582963


namespace comparison_l582_582158

noncomputable def a := Real.log 3000 / Real.log 9
noncomputable def b := Real.log 2023 / Real.log 4
noncomputable def c := (11 * Real.exp (0.01 * Real.log 1.001)) / 2

theorem comparison : a < b ∧ b < c :=
by
  sorry

end comparison_l582_582158


namespace problem_solution_l582_582221

def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then |x - 1| - 1
  else 1 / (1 + x^2)

theorem problem_solution : f (f (1/2)) = 1/2 :=
  by
    sorry

end problem_solution_l582_582221


namespace axis_of_symmetry_of_f_l582_582595

noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x + 1)

theorem axis_of_symmetry_of_f : (axis_of_symmetry : ℝ) = -1 :=
by
  sorry

end axis_of_symmetry_of_f_l582_582595


namespace volume_of_pyramid_is_3380_l582_582401

-- Define the vertices of the original triangle
def A := (0, 0 : ℝ × ℝ)
def B := (30, 0 : ℝ × ℝ)
def C := (18, 26 : ℝ × ℝ)

-- Define the midpoints of the sides of the triangle
def M1 := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2 : ℝ × ℝ)
def M2 := ((B.fst + C.fst) / 2, (B.snd + C.snd) / 2 : ℝ × ℝ)
def M3 := ((C.fst + A.fst) / 2, (C.snd + A.snd) / 2 : ℝ × ℝ)

-- Define the area of the triangle ABC
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs (A.fst * (B.snd - C.snd) + B.fst * (C.snd - A.snd) + C.fst * (A.snd - B.snd)) / 2

-- Define the centroid of the triangle ABC
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.fst + B.fst + C.fst) / 3, (A.snd + B.snd + C.snd) / 3)

-- Define the volume of a triangular pyramid
def volume_pyramid (base_area height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- Prove the volume of the pyramid is 3380
theorem volume_of_pyramid_is_3380 : 
  volume_pyramid (area_triangle A B C) (centroid A B C).snd = 3380 := by
  -- Using placeholder for proof
  sorry

end volume_of_pyramid_is_3380_l582_582401


namespace find_m_l582_582695

section
variable (L : List ℤ)
variable (m n : ℤ)

-- Conditions
def list_has_mode_40 := L.mode = 40
def list_mean_is_35 := L.mean = 35
def smallest_num_is_20 := L.minimum = 20
def m_is_in_list := m ∈ L
def replace_m_with_m_plus_15 := L.replace_nth (L.index_of m) (m + 15) = L.mean = 40 ∧ L.median = m + 15
def replace_m_with_m_minus_10 := L.replace_nth (L.index_of m) (m - 10) = L.median = m - 5

-- The proof goal
theorem find_m (L : List ℤ) (m : ℤ)
  (H1 : list_has_mode_40 L)
  (H2 : list_mean_is_35 L)
  (H3 : smallest_num_is_20 L)
  (H4 : m_is_in_list L m)
  (H5 : replace_m_with_m_plus_15 L m)
  (H6 : replace_m_with_m_minus_10 L m) : 
  m = 40 :=
sorry
end

end find_m_l582_582695


namespace circle_color_areas_possible_l582_582713

def equilateral_triangle (a : ℝ) (v1 v2 v3 : EuclideanSpace ℝ (Fin 2)) :=
  dist v1 v2 = a ∧ dist v2 v3 = a ∧ dist v3 v1 = a

def circle_area (r : ℝ) : ℝ :=
  π * r ^ 2

def color_areas (a : ℝ) (r1 r2 r3 : ℝ) (v1 v2 v3 : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (yellow_area green_area blue_area : ℝ),
    yellow_area = 100 ∧ green_area = 10 ∧ blue_area = 1 ∧
    circle_area r1 > a ∧ circle_area r2 > a ∧ circle_area r3 > a

theorem circle_color_areas_possible :
  ∃ (a r1 r2 r3 : ℝ) (v1 v2 v3 : EuclideanSpace ℝ (Fin 2)),
    0 < a ∧ 0 < r1 ∧ 0 < r2 ∧ 0 < r3 ∧
    equilateral_triangle a v1 v2 v3 →
    color_areas a r1 r2 r3 v1 v2 v3 :=
sorry

end circle_color_areas_possible_l582_582713


namespace triangle_inequality_l582_582512

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) (S : ℝ) (hA : A + B + C = π) 
  (ha : a = 2 * S / (b * c * sin A))
  (hb : b = 2 * S / (a * c * sin B))
  (hc : c = 2 * S / (a * b * sin C)) :
  a^2 * tan (A/2) + b^2 * tan (B/2) + c^2 * tan (C/2) ≥ 4 * S := 
sorry

end triangle_inequality_l582_582512


namespace min_value_of_expr_l582_582924

noncomputable def min_expr (a b c : ℝ) := (2 * a / b) + (3 * b / c) + (4 * c / a)

theorem min_value_of_expr (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) 
    (habc : a * b * c = 1) : 
  min_expr a b c ≥ 9 := 
sorry

end min_value_of_expr_l582_582924


namespace tangent_identity_l582_582577

theorem tangent_identity :
  Real.tan (55 * Real.pi / 180) * 
  Real.tan (65 * Real.pi / 180) * 
  Real.tan (75 * Real.pi / 180) = 
  Real.tan (85 * Real.pi / 180) :=
sorry

end tangent_identity_l582_582577


namespace find_f3_l582_582560

def f : ℤ → ℤ
| x => if x >= 6 then 3 * x - 5 else f (x + 2)

theorem find_f3 : f 3 = 16 :=
by
  sorry

end find_f3_l582_582560


namespace infinite_triples_exist_l582_582916

theorem infinite_triples_exist (n : ℕ) (hn : 0 < n) :
  ∃^∞ (x y z : ℤ), (n * x^2 + y^3 = z^4) ∧ (Nat.gcd x.natAbs y.natAbs = 1) ∧
    (Nat.gcd y.natAbs z.natAbs = 1) ∧ (Nat.gcd z.natAbs x.natAbs = 1) :=
sorry

end infinite_triples_exist_l582_582916


namespace f_finite_f_equality_l582_582334

variables (a b c : ℤ) (P n : ℕ) (k : ℕ)

noncomputable def f (n : ℕ) : ℕ :=
  if H : (a > 0) ∧ (b + c + n ≥ 0) then
    -- count number of pairs (d, e) such that a d^2 + 2 b d e + c e^2 = n
    ⟨d, ⟨e, b, c, a⟩ | a * d^2 + 2 * b * d * e + c * e^2 = n⟩ 
  else 0

theorem f_finite (h_pos : a > 0) (hc_pos : ac - b^2 = P ∧ squarefree P) (hP_pos : 0 < P) :
  ∃ M, ∀ n, f n < M :=
sorry

theorem f_equality (k_pos : k > 0) (h_pos : a > 0) (hc_pos : ac - b^2 = P ∧ squarefree P) (hP_pos : 0 < P) :
  ∀ n, f n = f (P ^ k * n) :=
sorry

end f_finite_f_equality_l582_582334


namespace factorization_and_evaluation_l582_582914

noncomputable def polynomial_q1 (x : ℝ) : ℝ := x
noncomputable def polynomial_q2 (x : ℝ) : ℝ := x^2 - 2
noncomputable def polynomial_q3 (x : ℝ) : ℝ := x^2 + x + 1
noncomputable def polynomial_q4 (x : ℝ) : ℝ := x^2 + 1

theorem factorization_and_evaluation :
  polynomial_q1 3 + polynomial_q2 3 + polynomial_q3 3 + polynomial_q4 3 = 33 := by
  sorry

end factorization_and_evaluation_l582_582914


namespace gcd_6Tn_nplus1_l582_582093

theorem gcd_6Tn_nplus1 (n : ℕ) (h : 0 < n) : gcd (3 * n * n + 3 * n) (n + 1) = 1 := by
  sorry

end gcd_6Tn_nplus1_l582_582093


namespace tetrahedron_edge_pairs_l582_582178

-- Problem statement
theorem tetrahedron_edge_pairs (tetrahedron_edges : ℕ) (tetrahedron_edges_eq_six : tetrahedron_edges = 6) :
  ∃ n, n = (Nat.choose tetrahedron_edges 2) ∧ n = 15 :=
by
  use Nat.choose 6 2
  constructor
  . rfl
  . exact Nat.choose_succ_self_right 5

end tetrahedron_edge_pairs_l582_582178


namespace tile_B_is_II_l582_582631

structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

def tile_I : Tile := { top := 2, right := 6, bottom := 5, left := 3 }
def tile_II : Tile := { top := 6, right := 2, bottom := 3, left := 5 }
def tile_III : Tile := { top := 5, right := 7, bottom := 1, left := 2 }
def tile_IV : Tile := { top := 3, right := 5, bottom := 6, left := 7 }

def is_adjacent (t1 t2 : Tile) : Prop := 
  t1.right = t2.left ∨ t1.left = t2.right ∨ t1.top = t2.bottom ∨ t1.bottom = t2.top

def RectangleB_has_tileII : Prop := 
  ∃ tile_map : Rect → Tile, 
    tile_map B = tile_II ∧ 
    is_adjacent tile_map B tile_map D ∧ 
    tile_map D = tile_III

theorem tile_B_is_II : RectangleB_has_tileII := sorry

end tile_B_is_II_l582_582631


namespace kates_discount_is_8_percent_l582_582387

-- Definitions based on the problem's conditions
def bobs_bill : ℤ := 30
def kates_bill : ℤ := 25
def total_paid : ℤ := 53
def total_without_discount : ℤ := bobs_bill + kates_bill
def discount_received : ℤ := total_without_discount - total_paid
def kates_discount_percentage : ℚ := (discount_received : ℚ) / kates_bill * 100

-- The theorem to prove
theorem kates_discount_is_8_percent : kates_discount_percentage = 8 :=
by
  sorry

end kates_discount_is_8_percent_l582_582387


namespace log_subtraction_proof_l582_582726

theorem log_subtraction_proof : log 2 3 - log 2 6 = -1 := by
  sorry

end log_subtraction_proof_l582_582726


namespace no_fractional_solution_l582_582769

theorem no_fractional_solution (x y : ℚ)
  (h₁ : ∃ m : ℤ, 13 * x + 4 * y = m)
  (h₂ : ∃ n : ℤ, 10 * x + 3 * y = n) :
  (∃ a b : ℤ, x ≠ a ∧ y ≠ b) → false :=
by {
  sorry
}

end no_fractional_solution_l582_582769


namespace triangle_perpendicular_bisectors_eq_l582_582903

open EuclideanGeometry

theorem triangle_perpendicular_bisectors_eq
  {A B C N M : Point}
  {α : ℝ} (hα : α = 60) 
  (h1 : B ≠ C) 
  (h2 : is_perpendicular_bisector (segment A B) (line_through A N)) 
  (h3 : is_perpendicular_bisector (segment A C) (line_through A M)) :
  dist B C = dist M N :=
sorry

end triangle_perpendicular_bisectors_eq_l582_582903


namespace chord_length_AB_l582_582269

noncomputable def length_of_chord (x y : ℝ) : ℝ :=
  let d := 2 / real.sqrt (1 + 3) in
  let radius := 2 in
  let L := 2 * real.sqrt (radius^2 - d^2) in
  L

theorem chord_length_AB : length_of_chord 2 (real.sqrt 3) = 2 * real.sqrt 3 := by
  sorry

end chord_length_AB_l582_582269


namespace probability_first_spade_second_king_l582_582637

/--
In a standard deck of 52 cards, the probability of drawing the first card as a ♠ and the second card as a king is 1/52.
-/
theorem probability_first_spade_second_king : 
  let deck_size := 52 in
  let hearts_count := 13 in
  let kings_count := 4 in
  let prob := (1 / deck_size : ℚ) * (kings_count / (deck_size - 1)) + ((hearts_count - 1) / deck_size) * (kings_count / (deck_size - 1)) 
  in 
  prob = 1 / deck_size :=
by
  sorry

end probability_first_spade_second_king_l582_582637


namespace equation_of_tangent_circle_l582_582347

-- Definitions for the problem conditions
def tangent_line : (ℝ × ℝ) → Prop 
| (x, y) => 4 * x - 3 * y + 6 = 0

def point_A := (3 : ℝ, 6 : ℝ)
def point_B := (5 : ℝ, 2 : ℝ)

-- Statement of the problem
theorem equation_of_tangent_circle :
  (∃ (a b r : ℝ), (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2 →
    (tangent_line (a, b) ∧ 
    (a, b) = point_A ∧ 
    (x - 5)^2 + (y - 2)^2 = r^2) →
    x^2 + y^2 - 10*x - 9*y + 39 = 0)) :=
sorry

end equation_of_tangent_circle_l582_582347


namespace simplify_and_evaluate_expression_l582_582978

theorem simplify_and_evaluate_expression (x : ℤ) (h : x = -2) : 
  2 * x * (x - 3) - (x - 2) * (x + 1) = 16 :=
by
  sorry

end simplify_and_evaluate_expression_l582_582978


namespace trig_identity_simplify_l582_582583

theorem trig_identity_simplify (α : ℝ) :
  sin^2 (α - π / 6) + sin^2 (α + π / 6) - sin^2 α = 1 / 2 :=
by
  sorry

end trig_identity_simplify_l582_582583


namespace integral_is_e_l582_582421

noncomputable def integral_problem : ℝ :=
  ∫ x in 0..1, (Real.exp x + 2 * x)

theorem integral_is_e : integral_problem = Real.exp 1 :=
by
  sorry

end integral_is_e_l582_582421


namespace C_plus_D_58_l582_582413

theorem C_plus_D_58 (D C : ℚ) 
  (h : ∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 → (Dx - 17)/(x^2 - 8x + 15) = C/(x - 3) + 5/(x - 5)) :
  C + D = 5.8 := 
sorry

end C_plus_D_58_l582_582413


namespace line_passes_through_first_and_fourth_quadrants_l582_582502

theorem line_passes_through_first_and_fourth_quadrants (b k : ℝ) (H : b * k < 0) :
  (∃x₁, k * x₁ + b > 0) ∧ (∃x₂, k * x₂ + b < 0) :=
by
  sorry

end line_passes_through_first_and_fourth_quadrants_l582_582502


namespace island_connected_to_all_l582_582289

theorem island_connected_to_all (n : ℕ) (h_n : n ≥ 3) 
  (initial_routes : ∀ {A B : Fin n}, A ≠ B → ∃ C : Fin n, C ≠ A ∧ C ≠ B ∧ connected_to_initial_routes (A, B, C))
  (close_route_add_new_routes : ∀ X Y : Fin n, X ≠ Y → ∀ {A B : Fin n}, 
    connected_to (X, A) ∧ ¬ connected_to (Y, A) ∧ 
    ¬ connected_to (X, B) ∧ connected_to (Y, B) → 
    add_new_routes (A, X) (B, Y)) : 
  ∃ Z : Fin n, ∀ W : Fin n, W ≠ Z → connected_to (Z, W) :=
sorry

end island_connected_to_all_l582_582289


namespace no_fractional_solution_l582_582766

theorem no_fractional_solution (x y : ℚ)
  (h₁ : ∃ m : ℤ, 13 * x + 4 * y = m)
  (h₂ : ∃ n : ℤ, 10 * x + 3 * y = n) :
  (∃ a b : ℤ, x ≠ a ∧ y ≠ b) → false :=
by {
  sorry
}

end no_fractional_solution_l582_582766


namespace rectangle_area_from_perimeter_l582_582993

theorem rectangle_area_from_perimeter
  (a : ℝ)
  (shorter_side := 12 * a)
  (longer_side := 22 * a)
  (P := 2 * (shorter_side + longer_side))
  (hP : P = 102) :
  (shorter_side * longer_side = 594) := by
  sorry

end rectangle_area_from_perimeter_l582_582993


namespace distance_from_P_to_plane_ABC_is_3_l582_582103

def point (x y z : ℝ) := (x, y, z)

def distance_to_plane_of_points (P A B C : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) = A
  let (x2, y2, z2) = B
  let (x3, y3, z3) = C
  let (px, py, pz) = P
  let n_x = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
  let n_y = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
  let n_z = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
  let d = -(n_x * x1 + n_y * y1 + n_z * z1)
  let numerator = abs (n_x * px + n_y * py + n_z * pz + d)
  let denominator = real.sqrt (n_x ^ 2 + n_y ^ 2 + n_z ^ 2)
  numerator / denominator

theorem distance_from_P_to_plane_ABC_is_3
  (x y : ℝ) :
  let A := point 1 0 0
  let B := point 0 1 0
  let C := point 1 1 0
  let P := point x y 3
  distance_to_plane_of_points P A B C = 3 :=
sorry

end distance_from_P_to_plane_ABC_is_3_l582_582103


namespace probability_raining_both_cities_l582_582730

theorem probability_raining_both_cities {A B : Type} [ProbabilitySpace A] [ProbabilitySpace B] (PA : Event A) (PB : Event B) 
  (hPA : Pr PA = 0.2) (hPB : Pr PB = 0.18) (indep : IndepEvents PA PB) : 
  Pr (PA ∩ PB) = 0.036 :=
by
  sorry

end probability_raining_both_cities_l582_582730


namespace modular_inverse_15_mod_16_l582_582079

theorem modular_inverse_15_mod_16 :
  ∃ b : ℤ, (15 * b) % 16 = 1 :=
begin
  use 15,
  sorry
end

end modular_inverse_15_mod_16_l582_582079


namespace largest_n_for_triangle_property_l582_582742

def has_triangle_property (S : Set ℕ) : Prop :=
  ∀ {a b c : ℕ}, a ∈ S → b ∈ S → c ∈ S → a + b > c ∧ b + c > a ∧ c + a > b

theorem largest_n_for_triangle_property : 
  ∀ (n : ℕ), n < 364 ↔ ∀ (S : Set ℕ), (∀ (x : ℕ), 6 ≤ x → x ≤ n → x ∈ S) → (∀ T : Set ℕ, T ⊆ S → T.card = 10 → has_triangle_property T) :=
begin
  sorry
end

end largest_n_for_triangle_property_l582_582742


namespace incorrect_statement_B_l582_582669

-- Definitions corresponding to each statement
def statementA : Prop := 
  ∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6} → (n % 2 = 0 ∨ n % 2 = 1)

def statementB : Prop := 
  ∃ deck : Set ℕ, 53 ∈ deck ∧ ∀ c ∈ deck, c = 53

def statementC : Prop := 
  ∃ B : Set ℤ, ∃ S : Set ℤ, ∃ n : ℤ, S ⊆ B ∧ 0 < n ∧ B.card = n

def statementD : Prop := 
  ∀ p : ℚ, p = 95/100 → (¬p = 1)

-- Main Theorem
theorem incorrect_statement_B : 
  statementB = false := 
sorry

end incorrect_statement_B_l582_582669


namespace range_of_a_l582_582511

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ exp x * (x - a) < 1) → a > -1 :=
by
  sorry

end range_of_a_l582_582511


namespace num_exhibits_is_four_l582_582712

theorem num_exhibits_is_four (h₁ : ∃ E : ℕ, 15 % E ≠ 0) (h₂ : ∃ E : ℕ, 16 % E = 0) (h₃ : ∃ E : ℕ, E ≠ 1 ∧ E ≠ 16) (h₄ : ∃ E : ℕ, E ∉ {1, 3, 5, 15}) : ∃ E : ℕ, E = 4 :=
by {
  sorry
}

end num_exhibits_is_four_l582_582712


namespace domain_of_function_l582_582601

noncomputable def domain (x : ℝ) : Prop := 
  log 0.5 (4 * x - 3) ≥ 0

theorem domain_of_function :
  {x : ℝ | domain x} = {x : ℝ | 3/4 < x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l582_582601


namespace triangle_ratios_l582_582904

-- Define the problem context
variables {A B C M K P : Type} 
variables [Triangle A B C]
variables [Bisector A M]
variables [Median B K A C]
variables [Perpendicular A M B K]
variables [Intersection P A M B K]

-- Define the theorem to prove ratios
theorem triangle_ratios (h1 : bisector A M) 
                        (h2 : median B K A C)
                        (h3 : perpendicular A M B K)
                        (h4 : intersection P A M B K) :
  (BP = PK ∧ AP = 3 * PM) :=
begin
  -- Proof goes here
  sorry
end

end triangle_ratios_l582_582904


namespace infinite_integer_solutions_l582_582578

theorem infinite_integer_solutions :
  ∃ (f : ℤ → ℤ × ℤ × ℤ), (∀ a : ℤ, (let ⟨x, y, z⟩ := f a in x^2 + y^2 - z^2 - x - 3*y - z - 4 = 0)) ∧
                         (∀ m n : ℤ, f m ≠ f n ↔ m ≠ n) :=
sorry

end infinite_integer_solutions_l582_582578


namespace Meghan_total_money_l582_582953

theorem Meghan_total_money (h100 : ℕ) (h50 : ℕ) (h10 : ℕ) : 
  h100 = 2 → h50 = 5 → h10 = 10 → 100 * h100 + 50 * h50 + 10 * h10 = 550 :=
by
  sorry

end Meghan_total_money_l582_582953


namespace memory_efficiency_problem_l582_582892

theorem memory_efficiency_problem (x : ℝ) (hx : x ≠ 0) :
  (100 / x - 100 / (1.2 * x) = 5 / 12) ↔ (100 / x - 100 / ((1 + 0.20) * x) = 5 / 12) :=
by sorry

end memory_efficiency_problem_l582_582892


namespace hexagon_coloring_l582_582186

theorem hexagon_coloring (n : ℕ) (h1 : n = 7) :
  (∑ k in finset.range(2 * n + 1), if k % 2 = 1 then 6 * ⌈k / 2⌉ else 0) + 1 = 96 ∧ 
  (∑ k in finset.range(2 * n + 1), if k % 2 = 0 then 6 * k / 2 else 0) = 73 := 
by
  sorry

end hexagon_coloring_l582_582186


namespace sum_of_ages_in_5_years_l582_582058

noncomputable def age_will_three_years_ago := 4
noncomputable def years_elapsed := 3
noncomputable def age_will_now := age_will_three_years_ago + years_elapsed
noncomputable def age_diane_now := 2 * age_will_now
noncomputable def years_into_future := 5
noncomputable def age_will_in_future := age_will_now + years_into_future
noncomputable def age_diane_in_future := age_diane_now + years_into_future

theorem sum_of_ages_in_5_years :
  age_will_in_future + age_diane_in_future = 31 := by
  sorry

end sum_of_ages_in_5_years_l582_582058


namespace physics_marks_l582_582016

variables (P C M : ℕ)

theorem physics_marks (h1 : P + C + M = 195)
                      (h2 : P + M = 180)
                      (h3 : P + C = 140) : P = 125 :=
by
  sorry

end physics_marks_l582_582016


namespace tan_sum_identity_l582_582799

theorem tan_sum_identity (α : ℝ) (h : sin α + cos α = - real.sqrt 2) :
  tan α + 1 / tan α = 2 := 
sorry

end tan_sum_identity_l582_582799


namespace women_per_table_l582_582371

theorem women_per_table 
  (total_tables : ℕ)
  (men_per_table : ℕ)
  (total_customers : ℕ) 
  (h_total_tables : total_tables = 6)
  (h_men_per_table : men_per_table = 5)
  (h_total_customers : total_customers = 48) :
  (total_customers - (men_per_table * total_tables)) / total_tables = 3 :=
by
  subst h_total_tables
  subst h_men_per_table
  subst h_total_customers
  sorry

end women_per_table_l582_582371


namespace total_clients_correct_l582_582361

-- Define the number of each type of cars and total cars
def num_cars : ℕ := 12
def num_sedans : ℕ := 4
def num_coupes : ℕ := 4
def num_suvs : ℕ := 4

-- Define the number of selections per car and total selections required
def selections_per_car : ℕ := 3

-- Define the number of clients per type of car
def num_clients_who_like_sedans : ℕ := (num_sedans * selections_per_car) / 2
def num_clients_who_like_coupes : ℕ := (num_coupes * selections_per_car) / 2
def num_clients_who_like_suvs : ℕ := (num_suvs * selections_per_car) / 2

-- Compute total number of clients
def total_clients : ℕ := num_clients_who_like_sedans + num_clients_who_like_coupes + num_clients_who_like_suvs

-- Prove that the total number of clients is 18
theorem total_clients_correct : total_clients = 18 := by
  sorry

end total_clients_correct_l582_582361


namespace baron_munchausen_correct_l582_582031

structure Polygon where
  vertices : Finset (ℝ × ℝ)

def isInside (p : ℝ × ℝ) (poly : Polygon) : Prop :=
  sorry -- Assume we have a predicate to determine if a point is inside a given polygon.

def dividesIntoThreePolygons (p : ℝ × ℝ) (poly : Polygon) : Prop :=
  ∀ l : (ℝ × ℝ) × (ℝ × ℝ), 
    line_through_point l p → 
    (split_polygon_by_line poly l).length = 3 

theorem baron_munchausen_correct :
  ∃ (poly : Polygon) (p : ℝ × ℝ),
    isInside p poly ∧ dividesIntoThreePolygons p poly :=
sorry

end baron_munchausen_correct_l582_582031


namespace max_value_of_x_l582_582874

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem max_value_of_x 
  (x : ℤ) 
  (h : log_base (1 / 4 : ℝ) (2 * x + 1) < log_base (1 / 2 : ℝ) (x - 1)) : x ≤ 3 :=
sorry

end max_value_of_x_l582_582874


namespace smallest_prime_p_l582_582086

theorem smallest_prime_p (p : ℕ) (h_prime : Prime p) (h_divisors : (factors_count (p^3 + 2 * p^2 + p) = 42)) : p = 23 := by
  sorry

end smallest_prime_p_l582_582086


namespace exists_two_subsets_with_one_common_element_l582_582522

open Set

theorem exists_two_subsets_with_one_common_element (S : Finset α) (n : ℕ) (T : Finset (Finset α))
  (hS_card : S.card = n) (hT_card : T.card = n + 1) (hS_geq_5 : 5 ≤ n) :
  ∃ A B ∈ T, A ≠ B ∧ (A ∩ B).card = 1 :=
by
  sorry

end exists_two_subsets_with_one_common_element_l582_582522


namespace cones_sold_on_seventh_day_l582_582385

theorem cones_sold_on_seventh_day :
  let cones : List ℕ := [100, 92, 109, 96, 103, 96]
  let mean : ℚ := 100.1
  let number_of_days := 7
  let total_cones_sold := mean * number_of_days
  let sum_first_six_days := list.sum cones
  let seventh_day_cones := total_cones_sold - sum_first_six_days
  seventh_day_cones.round = 105 :=
by
  sorry

end cones_sold_on_seventh_day_l582_582385


namespace cos_alpha_second_quadrant_l582_582126

variable (α : Real)
variable (h₁ : α ∈ Set.Ioo (π / 2) π)
variable (h₂ : Real.sin α = 5 / 13)

theorem cos_alpha_second_quadrant : Real.cos α = -12 / 13 := by
  sorry

end cos_alpha_second_quadrant_l582_582126


namespace train_length_l582_582018

theorem train_length (speed_kmph : ℤ) (time_sec : ℤ) (expected_length_m : ℤ) 
    (speed_kmph_eq : speed_kmph = 72)
    (time_sec_eq : time_sec = 7)
    (expected_length_eq : expected_length_m = 140) :
    expected_length_m = (speed_kmph * 1000 / 3600) * time_sec :=
by 
    sorry

end train_length_l582_582018


namespace binary_quadratic_lines_value_m_l582_582878

theorem binary_quadratic_lines_value_m (m : ℝ) :
  (∀ x y : ℝ, x^2 + 2 * x * y + 8 * y^2 + 14 * y + m = 0) →
  m = 7 :=
sorry

end binary_quadratic_lines_value_m_l582_582878


namespace circle_center_radius_sum_l582_582919

theorem circle_center_radius_sum :
  let C := { p : ℝ × ℝ | (p.1^2 - 8*p.1) - (p.2^2 + 6*p.2) = 2 }
  ∃ (a b r : ℝ), (∀ p ∈ C, (p.1 - a)^2 + (p.2 - b)^2 = r^2) ∧
                  a = 4 ∧ b = -3 ∧ r = 3 * Real.sqrt 3 ∧
                  a + b + r = 1 + 3 * Real.sqrt 3 :=
by
  let C := { p : ℝ × ℝ | (p.1^2 - 8*p.1) - (p.2^2 + 6*p.2) = 2 }
  use 4, -3, 3 * Real.sqrt(3)
  have eq_center : ∀ p ∈ C, (p.1 - 4)^2 + (p.2 + 3)^2 = 27 := sorry
  have eq_sum := by norm_num
  exact ⟨eq_center, rfl, rfl, rfl, eq_sum⟩

end circle_center_radius_sum_l582_582919


namespace no_a_where_A_eq_B_singleton_l582_582940

def f (a x : ℝ) := x^2 + 4 * x - 2 * a
def g (a x : ℝ) := x^2 - a * x + a + 3

theorem no_a_where_A_eq_B_singleton :
  ∀ a : ℝ,
    (∃ x₁ : ℝ, (f a x₁ ≤ 0 ∧ ∀ x₂, f a x₂ ≤ 0 → x₂ = x₁)) ∧
    (∃ y₁ : ℝ, (g a y₁ ≤ 0 ∧ ∀ y₂, g a y₂ ≤ 0 → y₂ = y₁)) →
    (¬ ∃ z : ℝ, (f a z ≤ 0) ∧ (g a z ≤ 0)) := 
by
  sorry

end no_a_where_A_eq_B_singleton_l582_582940


namespace triangle_ratios_l582_582905

-- Define the problem context
variables {A B C M K P : Type} 
variables [Triangle A B C]
variables [Bisector A M]
variables [Median B K A C]
variables [Perpendicular A M B K]
variables [Intersection P A M B K]

-- Define the theorem to prove ratios
theorem triangle_ratios (h1 : bisector A M) 
                        (h2 : median B K A C)
                        (h3 : perpendicular A M B K)
                        (h4 : intersection P A M B K) :
  (BP = PK ∧ AP = 3 * PM) :=
begin
  -- Proof goes here
  sorry
end

end triangle_ratios_l582_582905


namespace xy_product_of_sample_l582_582146

/-- Given a sample {9, 10, 11, x, y} such that the average is 10 and the standard deviation is sqrt(2), 
    prove that the product of x and y is 96. -/
theorem xy_product_of_sample (x y : ℝ) 
  (h_avg : (9 + 10 + 11 + x + y) / 5 = 10)
  (h_stddev : ( (9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2 ) / 5 = 2) :
  x * y = 96 :=
by
  -- Proof goes here
  sorry

end xy_product_of_sample_l582_582146


namespace log_geometric_sequence_l582_582902

def positive_geometric_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + m) = (a n) * (a m)

theorem log_geometric_sequence (a : ℕ → ℝ) (h_seq : positive_geometric_sequence a) 
  (h_a3_a11 : a 3 * a 11 = 16) : 
  log 2 (a 2) + log 2 (a 12) = 4 := by
  -- proof outline:
  -- 1. express log_2 (a_2) + log_2 (a_12) using logarithm rules
  -- 2. use properties of the geometric sequence
  -- 3. use the given condition a_3 * a_11 = 16
  -- 4. apply logarithm properties to reach the final equality
  sorry

end log_geometric_sequence_l582_582902


namespace find_b_l582_582696

theorem find_b (a b : ℝ) (h1 : (-6) * a^2 = 3 * (4 * a + b))
  (h2 : a = 1) : b = -6 :=
by 
  sorry

end find_b_l582_582696


namespace trig_identity_l582_582082

theorem trig_identity :
  (sin (15 * Real.pi / 180) * cos (15 * Real.pi / 180) + cos (165 * Real.pi / 180) * cos (105 * Real.pi / 180)) /
  (sin (19 * Real.pi / 180) * cos (11 * Real.pi / 180) + cos (161 * Real.pi / 180) * cos (101 * Real.pi / 180)) = 1 :=
by
  sorry

end trig_identity_l582_582082


namespace evaluate_expression_l582_582870

-- Given condition
def M_gt_1 (M : ℝ) : Prop := M > 1

theorem evaluate_expression (M : ℝ) (h : M_gt_1 M) : sqrt(M * (∛(M * sqrt M))) = M^(3/4) :=
by
  sorry

end evaluate_expression_l582_582870


namespace clothes_and_transport_expense_l582_582565

variables 
  (S : ℝ) -- Mr. Yadav's monthly salary
  (annual_savings : ℝ := 24624) -- Mr. Yadav's annual savings

-- Conditions
def consumable_items_expense := 0.60 * S
def rent_expense := 0.20 * S
def utilities_expense := 0.10 * S
def entertainment_expense := 0.05 * S
def remaining_salary := S - (consumable_items_expense + rent_expense + utilities_expense + entertainment_expense)
def monthly_savings := annual_savings / 12
def spend_on_clothes_and_transport := 0.50 * remaining_salary

theorem clothes_and_transport_expense :
  spend_on_clothes_and_transport = 2052 :=
by
  sorry

end clothes_and_transport_expense_l582_582565


namespace range_of_a_for_inequality_l582_582144

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) ↔ a ≥ 2 :=
by {
  sorry
}

end range_of_a_for_inequality_l582_582144


namespace determine_k_l582_582445

-- noncomputable to allow for decimal representation of k like -1/4
noncomputable def collinear_points_are_distinct (k : ℝ) : Prop :=
  let OA := (k, 2)
  let OB := (1, 2 * k)
  let OC := (1 - k, -1)
  let AB := (fst OB - fst OA, snd OB - snd OA)
  let BC := (fst OC - fst OB, snd OC - snd OB)
  AB.1 * BC.2 - AB.2 * BC.1 = 0

theorem determine_k (k : ℝ) (h : collinear_points_are_distinct k) : k = -1 / 4 :=
sorry

end determine_k_l582_582445


namespace matrix_transforms_correctly_l582_582430

def matrix_transformation (M A : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ := 
  M.mul A

theorem matrix_transforms_correctly (a b c d : ℝ) :
  let A := (λ i j, if (i = 0 ∧ j = 0) then a else if (i = 0 ∧ j = 1) then b else if (i = 1 ∧ j = 0) then c else d) : Matrix (Fin 2) (Fin 2) ℝ in
  let M := (λ i j, if (i = 0 ∧ j = 0) then 2 else if (i = 0 ∧ j = 1) then 1 else if (i = 1 ∧ j = 0) then 2 else 1) : Matrix (Fin 2) (Fin 2) ℝ in
  matrix_transformation M A = λ i j, if (i = 0 ∧ j = 0) then (2 * a + b) else if (i = 0 ∧ j = 1) then (2 * b + a) else if (i = 1 ∧ j = 0) then (2 * c + d) else (2 * d + c) :=
sorry

end matrix_transforms_correctly_l582_582430


namespace range_of_a_l582_582473

noncomputable def f (x a : ℝ) := (2 - x) * Real.exp x - a * x - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a > 0 → x ∈ {1, 2}) →
  a ∈ Set.Ico (-1 / 4 * Real.exp 3) 0 :=
by
  sorry

end range_of_a_l582_582473


namespace average_postcards_per_day_l582_582203

noncomputable section
open_locale big_operators

def sum_arithmetic_sequence (a d n : ℕ) : ℕ :=
  n * a + d * (n * (n - 1)) / 2

def avg_arithmetic_sequence (a d n : ℕ) : ℕ :=
  (sum_arithmetic_sequence a d n) / n

theorem average_postcards_per_day :
  avg_arithmetic_sequence 12 10 7 = 42 :=
by
  sorry

end average_postcards_per_day_l582_582203


namespace two_pow_neg_f_decreasing_l582_582586

open Real

-- Define an increasing function on ℝ
variable (f : ℝ → ℝ)
hypothesis (h_inc : ∀ x y : ℝ, x < y → f x < f y)

-- Prove that if f is an increasing function, then 2^(-f(x)) is decreasing
theorem two_pow_neg_f_decreasing : ∀ x y : ℝ, x < y → 2^(-f y) < 2^(-f x) :=
by
  sorry

end two_pow_neg_f_decreasing_l582_582586


namespace inscribed_sphere_radius_l582_582458

theorem inscribed_sphere_radius {V S1 S2 S3 S4 R : ℝ} :
  (1/3) * R * (S1 + S2 + S3 + S4) = V → 
  R = 3 * V / (S1 + S2 + S3 + S4) :=
by
  intro h
  sorry

end inscribed_sphere_radius_l582_582458


namespace perimeter_inequality_l582_582812

-- Definitions for points and triangles
variables {A B C K L M : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space K] [metric_space L] [metric_space M]

-- Definitions for congruence of triangles
def congruent (T1 T2 : triangle) : Prop :=
  T1.side_lengths = T2.side_lengths

-- Perimeter of a triangle
def perimeter (T : triangle) : ℝ :=
  T.side_lengths.1 + T.side_lengths.2 + T.side_lengths.3

-- Semiperimeter of a triangle
def semiperimeter (T : triangle) : ℝ :=
  (perimeter T) / 2

-- Assumptions and proof goal
theorem perimeter_inequality 
  (h₁ : congruent (triangle.mk A B K) (triangle.mk A B C))
  (h₂ : congruent (triangle.mk L B C) (triangle.mk A B C))
  (h₃ : congruent (triangle.mk A M C) (triangle.mk A B C)) :
  perimeter (triangle.mk A B C) ≥ semiperimeter (triangle.mk K L M) :=
sorry

end perimeter_inequality_l582_582812


namespace rate_of_mangoes_per_kg_l582_582855

variable (grapes_qty : ℕ := 8)
variable (grapes_rate_per_kg : ℕ := 70)
variable (mangoes_qty : ℕ := 9)
variable (total_amount_paid : ℕ := 1055)

theorem rate_of_mangoes_per_kg :
  (total_amount_paid - grapes_qty * grapes_rate_per_kg) / mangoes_qty = 55 :=
by
  sorry

end rate_of_mangoes_per_kg_l582_582855


namespace all_zeros_after_10th_row_max_non_zero_rows_l582_582674

def row_transform (row : List ℕ) : List ℕ :=
  List.ofFn (λ i, row.drop (i + 1) |> List.filter (λ x, x > row.get! i) |> List.length)

def get_nth_row (n : ℕ) (first_row : List ℕ) : List ℕ :=
  (List.iterate row_transform n).get! first_row

-- Part (a) Proof that from the 11th row onward, all elements will be zero
theorem all_zeros_after_10th_row (first_row : List ℕ) (h_first_row : first_row.length = 10) :
  ∀ k ≥ 11, get_nth_row k first_row = List.repeat 0 10 := sorry

-- Part (b) Proof that the maximum number of non-zero rows is 10
theorem max_non_zero_rows (first_row : List ℕ) (h_first_row : first_row.length = 10) :
  ∃ k, k ≤ 10 ∧ get_nth_row (k + 1) first_row = List.repeat 0 10 := 
begin
  use 10,
  split,
  { exact le_rfl, },  -- k ≤ 10
  { sorry }   -- get_nth_row 11 = zero
end

end all_zeros_after_10th_row_max_non_zero_rows_l582_582674


namespace coefficient_of_term_without_x_l582_582050

theorem coefficient_of_term_without_x 
  (xy : ℚ) (x : ℚ) : 
  (∑ k in Finset.range 7, (Nat.choose 6 k) * (xy)^k * ((-1/x)^(6-k))
    ) 
    = -20 :=
by
  sorry

end coefficient_of_term_without_x_l582_582050


namespace circular_garden_area_l582_582634

theorem circular_garden_area (A B D C : Point)
  (hAB : dist A B = 20)
  (hDC : dist D C = 12)
  (hD_mid : midpoint D A B)
  (hDC_perp : ∠( D, C, A ) = 90°) :
  area_circle (dist C A) = 244 * pi :=
by
  sorry

end circular_garden_area_l582_582634


namespace no_fractional_xy_l582_582762

theorem no_fractional_xy (x y : ℚ) (m n : ℤ) (h1 : 13 * x + 4 * y = m) (h2 : 10 * x + 3 * y = n) : ¬ (¬(x ∈ ℤ) ∨ ¬(y ∈ ℤ)) :=
sorry

end no_fractional_xy_l582_582762


namespace hyperbola_eccentricity_is_sqrt5_l582_582438

-- Given conditions:
variables {a b m : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : m ≠ 0) (h4 : b = 2 * a)

-- Hyperbola definition: x²/a² - y²/b² = 1
def hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Line definition: y = 2x + m
def line (x y m : ℝ) : Prop := y = 2 * x + m

-- Eccentricity calculation
def eccentricity (a b : ℝ) : ℝ := (Real.sqrt (a^2 + b^2)) / a

-- Proof statement: The eccentricity of the hyperbola given the conditions
theorem hyperbola_eccentricity_is_sqrt5 : 
  ∀ (x y a b m : ℝ), a > 0 → b > 0 → m ≠ 0 → b = 2 * a → line x y m → hyperbola x y a b → 
  eccentricity a b = Real.sqrt 5 := 
by
  intros x y a b m ha hb hm hb2 line_eq hyp_eq
  sorry

end hyperbola_eccentricity_is_sqrt5_l582_582438


namespace percentage_boys_from_school_A_is_20_l582_582515

noncomputable def boysFromSchoolA (totalBoys : ℕ) (boysNotStudyingScience : ℕ) (percentageStudyingScience : ℝ) : ℝ :=
  let percentageNotStudyingScience := 1 - percentageStudyingScience
  let boysFromSchoolA := boysNotStudyingScience / percentageNotStudyingScience
  (boysFromSchoolA / totalBoys) * 100

theorem percentage_boys_from_school_A_is_20 :
  ∀ (totalBoys boysNotStudyingScience : ℕ) (percentageStudyingScience : ℝ),
  boysNotStudyingScience = 49 →
  totalBoys ≈ 350 →
  percentageStudyingScience = 0.30 →
  boysFromSchoolA totalBoys boysNotStudyingScience percentageStudyingScience = 20 :=
by
  intros totalBoys boysNotStudyingScience percentageStudyingScience h₁ h₂ h₃
  -- Proof skipped
  sorry

end percentage_boys_from_school_A_is_20_l582_582515


namespace maximal_area_quadrilateral_l582_582287

variables {α β : ℝ} {O A M N : EuclideanGeometry.Point}

-- Definition of angle vertex at O
-- Fixed point A inside the angle
-- Points B and C on the sides of the angle such that angle BAC = β 

def conditions (α β : ℝ) (A O M N : EuclideanGeometry.Point) : Prop :=
  -- Ensure constraints on α and β
  α + β < 180 ∧
  -- Ensure equal lengths of AM and AN
  (EuclideanGeometry.dist A M = EuclideanGeometry.dist A N) ∧
  -- Ensure ∠MAN = β
  (EuclideanGeometry.angle A M N = β)

theorem maximal_area_quadrilateral
  (h : conditions α β A O M N) :
  ∀ (M' N' : EuclideanGeometry.Point), conditions α β A O M' N' → 
  EuclideanGeometry.area (EuclideanGeometry.quadrilateral O M A N) ≥ 
  EuclideanGeometry.area (EuclideanGeometry.quadrilateral O M' A N') :=
sorry

end maximal_area_quadrilateral_l582_582287


namespace t_shirts_to_buy_l582_582613

variable (P T : ℕ)

def condition1 : Prop := 3 * P + 6 * T = 750
def condition2 : Prop := P + 12 * T = 750

theorem t_shirts_to_buy (h1 : condition1 P T) (h2 : condition2 P T) :
  400 / T = 8 :=
by
  sorry

end t_shirts_to_buy_l582_582613


namespace three_digit_numbers_count_l582_582495

theorem three_digit_numbers_count : 
  let numbers := [(h, t, o) | h <- [1, 2, 3, 4, 5, 6, 7, 8, 9],
                             t <- [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                             o <- [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                             h > t ∧ h > o ∧ t ≠ o] in
  numbers.length = 120 := 
by
  sorry

end three_digit_numbers_count_l582_582495


namespace odd_three_digit_integers_divisible_by_five_no_digit_five_l582_582492

theorem odd_three_digit_integers_divisible_by_five_no_digit_five : 
  let A := {d | d ≠ 5 ∧ d ≠ 0 ∧ d ∈ {1, 3, 7, 9}},
      B := {d | d ≠ 5 ∧ d ∈ {0, 1, 2, 3, 4, 6, 7, 8, 9}},
      C := {d | d ≠ 5 ∧ d ∈ {1, 3, 7, 9}} in
  ∑ (a : ℕ) in A, ∑ (b : ℕ) in B, ∑ (c : ℕ) in C, 1 = 144 := 
by {
  let numA := 4, -- {1, 3, 7, 9}
  let numB := 9, -- {0, 1, 2, 3, 4, 6, 7, 8, 9}
  let numC := 4, -- {1, 3, 7, 9}
  have : numA * numB * numC = 144 := by norm_num,
  exact this,
  }

end odd_three_digit_integers_divisible_by_five_no_digit_five_l582_582492


namespace solution_to_problem_l582_582069

def number_exists (n : ℝ) : Prop :=
  n / 0.25 = 400

theorem solution_to_problem : ∃ n : ℝ, number_exists n ∧ n = 100 := by
  sorry

end solution_to_problem_l582_582069


namespace largest_n_unique_k_l582_582656

theorem largest_n_unique_k :
  ∃ (n : ℕ), (∀ (k1 k2 : ℕ), 
    (9 / 17 < n / (n + k1) → n / (n + k1) < 8 / 15 → 9 / 17 < n / (n + k2) → n / (n + k2) < 8 / 15 → k1 = k2) ∧ 
    n = 72) :=
sorry

end largest_n_unique_k_l582_582656


namespace tangent_to_circumscribed_circle_l582_582969

theorem tangent_to_circumscribed_circle 
  (ABC : Triangle) 
  (isosceles : ABC.is_isosceles AB BC)
  (O1 : Point) (O2 : Point)
  (H1 : O1 = ABC.circumcenter)
  (H2 : O2 = ABC.incenter)
  (circumABC : Circumcircle ABC O1)
  (circumO1A2A : Circumcircle (triangle.mk O1 O2 A) O1):
  let A, B, C := ABC.vertices in
  let D := (Circumcircle.intersect circumABC circumO1A2A).2 in
  ∃Tangent : Line, Tangent.tangent_to_circumscribed_circle_of_triangle (triangle.mk O1 O2 A) BD ⟨D, _⟩ :=
sorry

end tangent_to_circumscribed_circle_l582_582969


namespace no_such_n_exists_l582_582060

-- Definition of concatenated numbers
def concat_digits (a n b : ℕ) : ℕ := 
  a * 10^(nat.log10 n + 2) + n * 10 + b

-- Main theorem statement
theorem no_such_n_exists :
  ¬ ∃ (n : ℕ), ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 → (10 * a + b) ∣ concat_digits a n b :=
by 
  sorry

end no_such_n_exists_l582_582060


namespace sum_powers_of_i_l582_582979

noncomputable def complex_i : ℂ := complex.I

theorem sum_powers_of_i :
  ∑ k in finset.range 2012, complex_i ^ (k + 1) = complex_i := by
  sorry

end sum_powers_of_i_l582_582979


namespace sum_of_fractions_not_integer_l582_582970

theorem sum_of_fractions_not_integer :
  let S := { (m, n) | m ∈ ℕ ∧ n ∈ ℕ ∧ 1 ≤ m ∧ m < n ∧ n ≤ 1986 } in
  let sum := ∑ s in S, (1:ℚ) / ((s.1:ℚ)^(s.2:ℕ)) in
  ¬ (∃ (k:ℤ), sum = (k:ℚ)) := 
sorry

end sum_of_fractions_not_integer_l582_582970


namespace Mina_additional_miles_l582_582233

theorem Mina_additional_miles:
  let distance1 := 20 -- distance in miles for the first part of the trip
  let speed1 := 40 -- speed in mph for the first part of the trip
  let speed2 := 60 -- speed in mph for the second part of the trip
  let avg_speed := 55 -- average speed needed for the entire trip in mph
  let distance2 := (distance1 / speed1 + (avg_speed * (distance1 / speed1)) / (speed1 - avg_speed * speed1 / speed2)) * speed2 -- formula to find the additional distance
  distance2 = 90 :=
by {
  sorry
}

end Mina_additional_miles_l582_582233


namespace sum_of_integers_from_100_to_1999_l582_582393

theorem sum_of_integers_from_100_to_1999 : 
  let a := 100 in
  let l := 1999 in
  let n := l - a + 1 in
  let sum := (n * (a + l)) / 2 in
  sum = 1994050 := 
by
  let a := 100
  let l := 1999
  let n := l - a + 1
  let sum := (n * (a + l)) / 2
  show sum = 1994050
  sorry

end sum_of_integers_from_100_to_1999_l582_582393


namespace value_of_v_l582_582437

-- Define the operation on real numbers
def star (v : ℝ) : ℝ := v - v / 3

-- Define the main theorem to be proven
theorem value_of_v : ∃ v : ℝ, star (star v) = 24 ∧ v = 54 := by
  sorry

end value_of_v_l582_582437


namespace number_of_factors_of_N_l582_582154

def N : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_factors_of_N : 
  ∃ (factors : ℕ), factors = 5 * 4 * 3 * 2 ∧ factors = 120 :=
by
  use 5 * 4 * 3 * 2
  split
  · refl
  · refl
  sorry

end number_of_factors_of_N_l582_582154


namespace right_triangle_hypotenuse_segment_ratio_l582_582010

theorem right_triangle_hypotenuse_segment_ratio
  (x : ℝ)
  (h₀ : 0 < x)
  (AB BC : ℝ)
  (h₁ : AB = 3 * x)
  (h₂ : BC = 4 * x) :
  ∃ AD DC : ℝ, AD / DC = 3 := 
by
  sorry

end right_triangle_hypotenuse_segment_ratio_l582_582010


namespace sin_phi_value_l582_582608

theorem sin_phi_value 
  (φ α : ℝ)
  (hφ : φ = 2 * α)
  (hα1 : Real.sin α = (Real.sqrt 5) / 5)
  (hα2 : Real.cos α = 2 * (Real.sqrt 5) / 5) 
  : Real.sin φ = 4 / 5 := 
by 
  sorry

end sin_phi_value_l582_582608


namespace how_many_leaves_l582_582290

def ladybugs_per_leaf : ℕ := 139
def total_ladybugs : ℕ := 11676

theorem how_many_leaves : total_ladybugs / ladybugs_per_leaf = 84 :=
by
  sorry

end how_many_leaves_l582_582290


namespace sequence_satisfy_conditions_l582_582409

theorem sequence_satisfy_conditions (n : ℕ) (a : ℕ → ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ n → (1 ≤ a i ∧ a i ≤ n)) ∧
  (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → |a i - a j| = |i - j|) →
  (∀ i, 1 ≤ i ∧ i ≤ n → (a i = i) ∨ (a i = n + 1 - i)) :=
by sorry

end sequence_satisfy_conditions_l582_582409


namespace binom_28_7_l582_582120

theorem binom_28_7 (h1 : Nat.choose 26 3 = 2600) (h2 : Nat.choose 26 4 = 14950) (h3 : Nat.choose 26 5 = 65780) : 
  Nat.choose 28 7 = 197340 :=
by
  sorry

end binom_28_7_l582_582120


namespace min_a_plus_b_l582_582822

open Real

theorem min_a_plus_b (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : log 4 (3 * a + 4 * b) = log 2 (sqrt (2 * a * b))) :
  a + b = (7 + 4 * sqrt 3) / 2 :=
by
  -- sorry should only be used to skip the proof, not part of the conditions.
  sorry

end min_a_plus_b_l582_582822


namespace binomial_product_l582_582411

theorem binomial_product (x : ℝ) : (4 * x + 3) * (x - 6) = 4 * x ^ 2 - 21 * x - 18 := 
sorry

end binomial_product_l582_582411


namespace rotate_circle_sectors_l582_582516

theorem rotate_circle_sectors (n : ℕ) (h : n > 0) :
  (∀ i, i < n → ∃ θ : ℝ, θ < (π / (n^2 - n + 1))) →
  ∃ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧
  (∀ i : ℕ, i < n → (θ * i) % (2 * π) > (π / (n^2 - n + 1))) :=
sorry

end rotate_circle_sectors_l582_582516


namespace min_value_reciprocal_sum_l582_582872

theorem min_value_reciprocal_sum (m n : ℝ) (hmn : m + n = 1) (hm_pos : m > 0) (hn_pos : n > 0) :
  1 / m + 1 / n ≥ 4 :=
sorry

end min_value_reciprocal_sum_l582_582872


namespace six_medial_triangles_form_triangle_l582_582628

-- Definitions and Conditions
def triangle := {A B C : Type} (noncomputable : Type)
def median (T : triangle) := {M1 M2 M3 : triangle} -- representing medians as triangles
def medial_triangles (T : triangle) : list triangle := sorry -- define a list of triangles resulting from medians

-- Theorem statement
theorem six_medial_triangles_form_triangle (T : triangle) :
  (exists (S : triangle), S ∈ medial_triangles T) := sorry

end six_medial_triangles_form_triangle_l582_582628


namespace range_of_a_in_acute_triangle_l582_582184

theorem range_of_a_in_acute_triangle (a b c : ℝ) (h : 0 < a) (hb : b = 1) (hc : c = 2) (h_acute : ∀ A B C : ℝ, A < 90 ∧ B < 90 ∧ C < 90) :
  sqrt 3 < a ∧ a < sqrt 5 :=
sorry

end range_of_a_in_acute_triangle_l582_582184


namespace complex_number_modulus_l582_582801

theorem complex_number_modulus (z : ℂ) (hz : (1 - complex.I) * z = 1 + complex.I) : 
  complex.abs z = 1 := 
sorry

end complex_number_modulus_l582_582801


namespace probability_first_spade_second_king_l582_582639

/--
In a standard deck of 52 cards, the probability of drawing the first card as a ♠ and the second card as a king is 1/52.
-/
theorem probability_first_spade_second_king : 
  let deck_size := 52 in
  let hearts_count := 13 in
  let kings_count := 4 in
  let prob := (1 / deck_size : ℚ) * (kings_count / (deck_size - 1)) + ((hearts_count - 1) / deck_size) * (kings_count / (deck_size - 1)) 
  in 
  prob = 1 / deck_size :=
by
  sorry

end probability_first_spade_second_king_l582_582639


namespace polar_coordinates_of_P_l582_582164

noncomputable def P_cartesian : ℝ × ℝ := (-real.sqrt 3, 1)

def rho (P : ℝ × ℝ) : ℝ := real.sqrt (P.1^2 + P.2^2)

def theta (P : ℝ × ℝ) : ℝ := real.arctan (P.2 / P.1) + if P.1 < 0 then real.pi else 0

theorem polar_coordinates_of_P :
  let P_polar := (rho P_cartesian, theta P_cartesian) in
  P_polar = (2, 5 * real.pi / 6) :=
by
  let P_polar := (rho P_cartesian, theta P_cartesian)
  have h1 : rho P_cartesian = 2, by sorry
  have h2 : theta P_cartesian = 5 * real.pi / 6, by sorry
  exact ⟨h1, h2⟩

end polar_coordinates_of_P_l582_582164


namespace bisection_next_step_l582_582302

variable {α : Type*} [partial_order α] [has_lt α] [add_group α] [has_div α]
variable {f : α → α}
variable (a b c : α)

def mid (x y : α) : α := (x + y) / 2

theorem bisection_next_step (h₀ : f a < 0) (h₁ : f b < 0) (h₂ : f c > 0) (h_intv : a < b ∧ b < c) : mid b c = 0.75 := 
sorry

end bisection_next_step_l582_582302


namespace part_a_part_b_part_c_l582_582930

-- Define the set S
def S : Set ℕ := { n | ∃ m k : ℕ, m ≥ 2 ∧ k ≥ 2 ∧ n = m^k }

-- Define f(n)
def f (n : ℕ) : ℕ := 
  ( {s : Finset ℕ | s ⊆ S ∧ s.sum = n} ).to_finset.card

-- Part (a): Prove that f(30) = 0
theorem part_a : f 30 = 0 :=
  sorry

-- Part (b): Show that f(n) ≥ 1 for n ≥ 31
theorem part_b (n : ℕ) (h : n ≥ 31) : f n ≥ 1 :=
  sorry

-- Define T as the set of integers for which f(n) = 3
def T : Set ℕ := { n | f n = 3 }

-- Part (c): Prove that T is finite and non-empty, and find the largest element of T
theorem part_c : T.Nonempty ∧ T.Finite ∧ (∃ N, T = {n ∈ T | n ≤ N} ∧ N = 111) :=
  sorry

end part_a_part_b_part_c_l582_582930


namespace pencils_per_row_l582_582074

-- Define the conditions as parameters
variables (total_pencils : Int) (rows : Int) 

-- State the proof problem using the conditions and the correct answer
theorem pencils_per_row (h₁ : total_pencils = 12) (h₂ : rows = 3) : total_pencils / rows = 4 := 
by 
  sorry

end pencils_per_row_l582_582074


namespace r_limit_l582_582206

def L (m : ℝ) : ℝ := -real.sqrt (6 + m)

def r (m : ℝ) : ℝ := (L (-m) - L m) / m

noncomputable def limit_r : ℝ := real.sqrt 6⁻¹

theorem r_limit (h : ∀ (m : ℝ), -6 < m ∧ m < 6) : 
  tendsto (λ m, r m) (nhds 0) (nhds limit_r) :=
sorry

end r_limit_l582_582206


namespace probability_remainder_one_l582_582380

theorem probability_remainder_one (N : ℕ) (h1 : 1 ≤ N) (h2 : N ≤ 1000) : 
  (let favorable_cases := (N % 4 = 1) ∨ (N % 4 = 3) in
   (if favorable_cases then 1 else 0) / (4) = 1 / 2) :=
by 
  sorry

end probability_remainder_one_l582_582380


namespace polynomial_sum_pqrs_l582_582277

theorem polynomial_sum_pqrs (p q r s : ℝ) :
  (∃ g : ℂ → ℂ, 
    (g = (λ x, x^4 + p*x^3 + q*x^2 + r*x + s)) ∧ 
    (∀ z : ℂ, g (3 * Complex.I) = 0) ∧ 
    (∀ z : ℂ, g (1 + 2*Complex.I) = 0) ∧ 
    (∀ x : ℝ, g x = g (Complex.conj x)))
  → p + q + r + s = -41 :=
by
  sorry

end polynomial_sum_pqrs_l582_582277


namespace actual_length_of_tunnel_in_km_l582_582962

-- Define the conditions
def scale_factor : ℝ := 30000
def length_on_map_cm : ℝ := 7

-- Using the conditions, we need to prove the actual length is 2.1 km
theorem actual_length_of_tunnel_in_km :
  (length_on_map_cm * scale_factor / 100000) = 2.1 :=
by sorry

end actual_length_of_tunnel_in_km_l582_582962


namespace sum_integers_neg50_to_60_l582_582315

theorem sum_integers_neg50_to_60 : 
  (Finset.sum (Finset.Icc (-50 : ℤ) 60) id) = 555 := 
by
  -- Placeholder for the actual proof
  sorry

end sum_integers_neg50_to_60_l582_582315


namespace track_circumference_is_720_l582_582675

-- Define the conditions of the problem
variables {A B : Type} [UniformSpace A] [UniformSpace B]
variables (circumference : ℝ) (x distanceA distanceB : ℝ)

-- Define the initial conditions
def initial_conditions (circumference : ℝ) :=
  ∃ (x : ℝ), 
  2 * x = circumference ∧                 -- The full circumference is twice x
  distanceB = 150 ∧                      -- B has traveled 150 yards at the first meeting
  (x - 150) = distanceA ∧                -- A has traveled x - 150 yards at the first meeting 
  (2 * x - 90) = distanceA ∧             -- A has traveled 2x - 90 yards at the second meeting
  (x + 90) = distanceB                    -- B has traveled x + 90 yards at the second meeting

-- The theorem to be proved
theorem track_circumference_is_720 (circumference distanceA distanceB : ℝ) :
  initial_conditions circumference →
  circumference = 720 :=
begin
  sorry
end

end track_circumference_is_720_l582_582675


namespace extra_sweets_per_child_l582_582570

theorem extra_sweets_per_child (n a s : ℕ) (h_n : n = 112) (h_a : a = 32) (h_s : s = 15) :
  let total_sweets := n * s in
  let present_children := n - a in
  let new_sweets_per_child := total_sweets / present_children in
  new_sweets_per_child - s = 6 :=
by
  sorry

end extra_sweets_per_child_l582_582570


namespace major_axis_of_ellipse_l582_582025

structure Ellipse :=
(center : ℝ × ℝ)
(tangent_y_axis : Bool)
(tangent_y_eq_3 : Bool)
(focus_1 : ℝ × ℝ)
(focus_2 : ℝ × ℝ)

noncomputable def major_axis_length (e : Ellipse) : ℝ :=
  2 * (e.focus_1.2 - e.center.2)

theorem major_axis_of_ellipse : 
  ∀ (e : Ellipse), 
    e.center = (3, 0) ∧
    e.tangent_y_axis = true ∧
    e.tangent_y_eq_3 = true ∧
    e.focus_1 = (3, 2 + Real.sqrt 2) ∧
    e.focus_2 = (3, -2 - Real.sqrt 2) →
      major_axis_length e = 4 + 2 * Real.sqrt 2 :=
by
  intro e
  intro h
  sorry

end major_axis_of_ellipse_l582_582025


namespace arithmetic_sequence_a5_l582_582185

-- Assume the sequence and the given conditions
variables (a : ℕ → ℝ)
hypothesis (arith_seq : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)
hypothesis (h : a 3 + a 7 = 16)

-- Define what we need to prove
theorem arithmetic_sequence_a5 : a 5 = 8 :=
by
  sorry

end arithmetic_sequence_a5_l582_582185


namespace expected_left_handed_l582_582573

theorem expected_left_handed (p : ℚ) (n : ℕ) (h : p = 1/6) (hs : n = 300) : n * p = 50 :=
by 
  -- Proof goes here
  sorry

end expected_left_handed_l582_582573


namespace polygon_sides_l582_582007

theorem polygon_sides (n : ℕ) (h_sum : 180 * (n - 2) = 1980) : n = 13 :=
by {
  sorry
}

end polygon_sides_l582_582007


namespace trigonometric_identity_l582_582434

theorem trigonometric_identity : (1 / 4) * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 16 := by
  sorry

end trigonometric_identity_l582_582434


namespace vertices_1010th_square_l582_582266

theorem vertices_1010th_square : 
  let initial_position := "ABCD" 
  let rotate_90_ccw (pos : String) := 
    match pos with 
    | "ABCD" => "BCDA" 
    | "BCDA" => "CDAB" 
    | "CDAB" => "DABC" 
    | "DABC" => "ABCD" 
    | _ => pos 
  let reflect_horizontal (pos : String) := 
    match pos with 
    | "ABCD" => "DCBA" 
    | "BCDA" => "ADCB" 
    | "CDAB" => "BADC" 
    | "DABC" => "CBAD" 
    | _ => pos 
  let transform (pos : String) (n : Nat) := 
    if n % 2 == 0 then rotate_90_ccw (rotate_90_ccw pos) 
    else rotate_90_ccw (reflect_horizontal (rotate_90_ccw pos)) 
  in transform initial_position 1010 = "BCDA" := 
by 
  sorry

end vertices_1010th_square_l582_582266


namespace polynomial_square_b_value_l582_582272

theorem polynomial_square_b_value
  (a b : ℚ)
  (h : ∃ (p q r : ℚ), (x^4 - x^3 + x^2 + a * x + b) = (p * x^2 + q * x + r)^2) :
  b = 9 / 64 :=
sorry

end polynomial_square_b_value_l582_582272


namespace a_100_is_one_div_fifty_l582_582168

noncomputable def a_n : ℕ → ℚ
| 1     := 2
| 2     := 1
| (n+1) := a_n n * (a_n n - a_n (n-1)) / (a_n (n-1) - a_n n)

theorem a_100_is_one_div_fifty (n : ℕ) (h0 : a_n 1 = 2) (h1 : a_n 2 = 1)
  (h : ∀ n, n ≥ 2 → a_n n * a_n (n - 1) / (a_n (n - 1) - a_n n) = a_n n * a_n (n + 1) / (a_n n - a_n (n + 1))) : 
  a_n 100 = 1 / 50 := 
sorry

end a_100_is_one_div_fifty_l582_582168


namespace two_digit_number_divisible_8_12_18_in_range_l582_582708

def is_divisible_by (n d : ℕ) : Prop := d ∣ n

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem two_digit_number_divisible_8_12_18_in_range :
  ∃ (n : ℕ), is_two_digit n ∧ is_divisible_by n 8 ∧ is_divisible_by n 12 ∧ is_divisible_by n 18 ∧ (n ≥ 60 ∧ n < 80) :=
begin
  sorry
end

end two_digit_number_divisible_8_12_18_in_range_l582_582708


namespace no_such_function_exists_l582_582579

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ n : ℕ, f (f n) = n + 1987 := 
sorry

end no_such_function_exists_l582_582579


namespace apprentice_daily_output_l582_582355

namespace Production

variables (x y : ℝ)

theorem apprentice_daily_output
  (h1 : 4 * x + 7 * y = 765)
  (h2 : 6 * x + 2 * y = 765) :
  y = 45 :=
sorry

end Production

end apprentice_daily_output_l582_582355


namespace problem_l582_582162

theorem problem (a b : ℝ) (h1 : |a - 2| + (b + 1)^2 = 0) : a - b = 3 := by
  sorry

end problem_l582_582162


namespace AF_passes_through_incenter_l582_582719

-- Define the geometry setup in Lean
variables {A B C D E F P Q L M N : Type*}
  [ordered_field F]
  {V : Type*} [add_comm_group V] [module F V] [affine_space V P] 

variables (a b c : F) (ABC : triangle A B C)
  (L : midpoint A B C B)
  (M : midpoint C B A A)
  (N : midpoint B A A C)

-- Points D and E bisect perimeter
variables (D : point_on A B C B)
  (E : point_on B A C A)

-- Points P and Q are reflections of D and E across L and N
variables (P : reflection D L)
  (Q : reflection E N)

-- Point F is intersection of PQ and LM
variables (PQ : line P Q)
variables (LM : line L M)
variables (F : intersection PQ LM)

-- Hypothesis
variables (h1 : bisects_perimeter A D B C B A)
variables (h2 : bisects_perimeter C E A B A C)
variables (h3 : B > C)

theorem AF_passes_through_incenter :
  passes_through_incenter α F :=
sorry

end AF_passes_through_incenter_l582_582719


namespace angle_AGH_90_l582_582891

variables {A B C H G : Type} (S : Triangle → ℝ)
variables {s₁ s₂ s₃ : ℝ} (h : 
  acute_triangle A B C ∧
  B ≠ C ∧
  orthocenter A B C H ∧
  centroid A B C G ∧
  (1 / S ⟨H, A, B⟩ + 1 / S ⟨H, A, C⟩ = 2 / S ⟨H, B, C⟩))

theorem angle_AGH_90 (hAssumptions : h): angle A G H = 90 := by
  sorry

end angle_AGH_90_l582_582891


namespace min_value_of_quadratic_l582_582308

theorem min_value_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, 3 * x^2 - 18 * x + 2000 ≤ 3 * y^2 - 18 * y + 2000) ∧ (3 * x^2 - 18 * x + 2000 = 1973) :=
by
  sorry

end min_value_of_quadratic_l582_582308


namespace largest_area_is_circle_l582_582024

noncomputable def triangle_area : ℝ := (3 + real.sqrt 3) / 4
noncomputable def trapezoid_area : ℝ := (3 + real.sqrt 3) / 4
noncomputable def circle_area : ℝ := real.pi
noncomputable def square_area : ℝ := 25 / 8

theorem largest_area_is_circle : 
  circle_area > trapezoid_area ∧ circle_area > square_area ∧ circle_area > triangle_area :=
begin
  -- Prove that the circle has the largest area among all given shapes
  sorry
end

end largest_area_is_circle_l582_582024


namespace compare_powers_l582_582827

theorem compare_powers (m n p : ℝ) 
  (h1 : 0 < m ∧ m < 1) 
  (h2 : 0 < n ∧ n < 1)
  (h3 : 0 < p ∧ p < 1)
  (h4 : Real.logBase 3 m = Real.logBase 5 n)
  (h5 : Real.logBase 5 n = Real.logBase 10 p) :
  m^(1/3) < n^(1/5) ∧ n^(1/5) < p^(1/10) :=
sorry

end compare_powers_l582_582827


namespace floor_ceil_expression_l582_582416

theorem floor_ceil_expression :
  (Int.floor ∘ (λ x => x + ↑(19/5)) ∘ Int.ceil ∘ λ x => x^2) (15/8) = 7 := 
by 
  sorry

end floor_ceil_expression_l582_582416


namespace who_is_knight_and_liar_l582_582572

-- Define the types for islanders A and B
inductive Islander
| Knight
| Liar

-- Define the axioms for the behaviors of knights and liars. Since this has to be kept theoretical, we use axiom statements to replace actual logical behavior.
axiom A_response : Islander -> Prop -- A's response to the question "Is either of you a knight?"
axiom definitive_truth : ∀ (A B : Islander), (A_response A = true) -> (A = Islander.Liar ∧ B = Islander.Knight)

-- The theorem to be proven
theorem who_is_knight_and_liar (A B : Islander) (h : definitive_truth A B (A_response A)) : A = Islander.Liar ∧ B = Islander.Knight :=
    sorry

end who_is_knight_and_liar_l582_582572


namespace burger_cost_l582_582204

theorem burger_cost {B : ℝ} (sandwich_cost : ℝ) (smoothies_cost : ℝ) (total_cost : ℝ)
  (H1 : sandwich_cost = 4)
  (H2 : smoothies_cost = 8)
  (H3 : total_cost = 17)
  (H4 : B + sandwich_cost + smoothies_cost = total_cost) :
  B = 5 :=
sorry

end burger_cost_l582_582204


namespace should_increase_speed_by_30_l582_582196

variable (T : ℝ) -- Usual travel time in minutes.
variable (v : ℝ) -- Usual speed.
variable (D : ℝ) -- Distance to work.
variable (increased_v : ℝ) := 1.6 * v -- Speed increased by 60%.

-- Given usual travel speed equation
variable (h_v : v = D / T)

-- Given increased speed equation and time saved
variable (delayed_time : ℝ) := T + 40  -- 40 minutes late
variable (arrival_time_saved : ℝ) := 65 -- Total of 65 minutes saved because of increased speed and late start

-- Setting up the new equations based on the conditions
variable (new_travel_time : ℝ) := (T - 65)
variable (new_speed_calculation : ℝ) := v * (T / (T - 40))

-- Prove that increasing the usual speed by 30% results in arriving at 9:00 AM if departed 40 minutes late.
theorem should_increase_speed_by_30 :
  (new_speed_calculation = 1.3 * v) :=
sorry

end should_increase_speed_by_30_l582_582196


namespace contrapositive_proposition_l582_582598

theorem contrapositive_proposition (a b : ℝ) :
  (¬ ((a - b) * (a + b) = 0) → ¬ (a - b = 0)) :=
sorry

end contrapositive_proposition_l582_582598


namespace sum_of_m_n_p_chord_length_l582_582729
noncomputable def circle_radius_3 := 3
noncomputable def circle_radius_9 := 9

theorem sum_of_m_n_p_chord_length :
  ∃ m n p : ℕ, gcd m p = 1 ∧ (∀ k : ℕ, k^2 ∣ n → k = 1) ∧
  ∃ (C3 : circle) (C1 C2 : circle) (O1 O2 O3 : point) (r1 r2 r3 : ℝ),
  circle_radius_3 = 3 →
  circle_radius_9 = 9 →
  C1.radius = r1 → C2.radius = r2 → C3.radius = r3 →
  C1.radius = circle_radius_3 → C2.radius = circle_radius_9 →
  ∃ chord_length : ℝ, chord_length = m * (real.sqrt n) / p ∧
  m + n + p = 413 :=
begin
  sorry
end

end sum_of_m_n_p_chord_length_l582_582729


namespace range_of_a_l582_582839

theorem range_of_a :
  ∀ (a : ℝ), (∀ (x : ℝ), (2 ≤ x ∧ x ≤ 3) → 0 < log a (1 / 2 * a * x^2 - x + 1 / 2)) ↔ 
             (3 / 4 < a ∧ a < 7 / 9) ∨ (5 / 4 < a) := by
sorry

end range_of_a_l582_582839


namespace constants_exist_l582_582785

theorem constants_exist :
  ∃ (P Q R : ℚ),
  (P = -8 / 15 ∧ Q = -7 / 6 ∧ R = 27 / 10) ∧
  (∀ x, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
  (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) =
  P / (x - 1) + Q / (x - 4) + R / (x - 6)) :=
by {
  use [-8/15, -7/6, 27/10],
  split,
  { split; refl },
  intro x,
  intro h,
  sorry
}

end constants_exist_l582_582785


namespace percent_fewer_than_50000_is_75_l582_582991

-- Define the given conditions as hypotheses
variables {P_1 P_2 P_3 P_4 : ℝ}
variable (h1 : P_1 = 0.35)
variable (h2 : P_2 = 0.40)
variable (h3 : P_3 = 0.15)
variable (h4 : P_4 = 0.10)

-- Define the percentage of counties with fewer than 50,000 residents
def percent_fewer_than_50000 (P_1 P_2 : ℝ) : ℝ :=
  P_1 + P_2

-- The theorem statement we need to prove
theorem percent_fewer_than_50000_is_75 (h1 : P_1 = 0.35) (h2 : P_2 = 0.40) :
  percent_fewer_than_50000 P_1 P_2 = 0.75 :=
by
  sorry

end percent_fewer_than_50000_is_75_l582_582991


namespace factorization_l582_582073

theorem factorization (a : ℝ) : 2 * a ^ 2 - 8 = 2 * (a + 2) * (a - 2) := 
by
  sorry

end factorization_l582_582073


namespace second_derivative_y_wrt_x_l582_582432

-- Define the parametric equations and the required conditions
noncomputable def x (t : ℝ) : ℝ := Real.cos t
noncomputable def y (t : ℝ) : ℝ := (Real.sin (t / 2)) ^ 4

-- Proof statement for the second derivative with respect to x
theorem second_derivative_y_wrt_x :
  ∀ t : ℝ, deriv (deriv (λ t, y t) / deriv (λ t, x t)) t = 
           (1 + Real.cos(t / 2) ^ 2) / (4 * Real.cos(t / 2) ^ 3) :=
by
  sorry

end second_derivative_y_wrt_x_l582_582432


namespace constants_exist_l582_582784

theorem constants_exist :
  ∃ (P Q R : ℚ),
  (P = -8 / 15 ∧ Q = -7 / 6 ∧ R = 27 / 10) ∧
  (∀ x, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
  (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) =
  P / (x - 1) + Q / (x - 4) + R / (x - 6)) :=
by {
  use [-8/15, -7/6, 27/10],
  split,
  { split; refl },
  intro x,
  intro h,
  sorry
}

end constants_exist_l582_582784


namespace joshua_orange_profit_l582_582550

theorem joshua_orange_profit:
  let cost_per_orange := (1250 / 25) / 100 in
  let sell_price_per_orange := 60 / 100 in
  let profit_per_orange := (sell_price_per_orange - cost_per_orange) * 100 in
  profit_per_orange = 10 :=
by
  sorry

end joshua_orange_profit_l582_582550


namespace minimum_value_of_y_l582_582793

noncomputable def y (x : ℝ) : ℝ :=
  x^2 + 12 * x + 108 / x^4

theorem minimum_value_of_y : ∃ x > 0, y x = 49 :=
by
  sorry

end minimum_value_of_y_l582_582793


namespace find_m_l582_582618

theorem find_m 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a (n - 1) + a (n + 1) = 2 * a n)
  (h_cond1 : a (m - 1) + a (m + 1) - a m ^ 2 = 0)
  (h_cond2 : S (2 * m - 1) = 38) 
  : m = 10 :=
sorry

end find_m_l582_582618


namespace brown_house_number_l582_582388

-- Defining the problem conditions
def sum_arithmetic_series (k : ℕ) := k * (k + 1) / 2

theorem brown_house_number (t n : ℕ) (h1 : 20 < t) (h2 : t < 500)
    (h3 : sum_arithmetic_series n = sum_arithmetic_series t / 2) : n = 84 := by
  sorry

end brown_house_number_l582_582388


namespace property_P_X1_property_P_X2_property_P_X_l582_582440

theorem property_P_X1 :
  let X1 := {-1, 1, 2}
  let Y1 := {(s, t) | s ∈ X1 ∧ t ∈ X1 }
  (∀ a1 ∈ Y1, ∃ a2 ∈ Y1, a1.1 * a2.1 + a1.2 * a2.2 = 0) → (True) := sorry

theorem property_P_X2 :
  let x := 4
  let X2 := {-1, 1, 2, x}
  let Y2 := {(s, t) | s ∈ X2 ∧ t ∈ X2}
  (∀ a1 ∈ Y2, ∃ a2 ∈ Y2, a1.1 * a2.1 + a1.2 * a2.2 = 0) → (x = 4) := sorry

theorem property_P_X :
  ∀ (X : Set ℝ) (n : ℕ),
  (0 < n) →
  (-1 ∈ X) →
  (∀ i < n, 0 < (X.to_list.nth i).get_or_else (-2)) →
  (∀ i < n - 1, (X.to_list.nth i).get_or_else (-2) < (X.to_list.nth (i + 1)).get_or_else (-2)) →
  let Y := {(s, t) | s ∈ X ∧ t ∈ X}
  (∀ a1 ∈ Y, ∃ a2 ∈ Y, a1.1 * a2.1 + a1.2 * a2.2 = 0) →
  (1 ∈ X ∧ (∃ xn, xn ∈ X ∧ xn > 1 → (∃ x1, x1 ∈ X ∧ x1 = 1))) := sorry

end property_P_X1_property_P_X2_property_P_X_l582_582440


namespace factor_of_expression_l582_582751

-- Definition of the given expression.
def f (x y z : ℝ) : ℝ := x^2 - y^2 - z^2 + 2 * y * z + x - y - z + 2

-- Definition of what it means to be a factor.
def is_factor (g h : ℝ → ℝ → ℝ → ℝ) : Prop := 
  ∃ k : ℝ → ℝ → ℝ → ℝ, ∀ x y z, g x y z = k x y z * h x y z

-- Proof problem: Prove that (x - y + z + 1) is a factor of the given expression f.
theorem factor_of_expression : 
  is_factor f (λ x y z, x - y + z + 1) := 
sorry

end factor_of_expression_l582_582751


namespace first_group_persons_l582_582980

-- Define the conditions as formal variables
variables (P : ℕ) (hours_per_day_1 days_1 hours_per_day_2 days_2 num_persons_2 : ℕ)

-- Define the conditions from the problem
def first_group_work := P * days_1 * hours_per_day_1
def second_group_work := num_persons_2 * days_2 * hours_per_day_2

-- Set the conditions based on the problem statement
axiom conditions : 
  hours_per_day_1 = 5 ∧ 
  days_1 = 12 ∧ 
  hours_per_day_2 = 6 ∧
  days_2 = 26 ∧
  num_persons_2 = 30 ∧
  first_group_work = second_group_work

-- Statement to prove
theorem first_group_persons : P = 78 :=
by
  -- The proof goes here
  sorry

end first_group_persons_l582_582980


namespace fraction_equality_l582_582394

theorem fraction_equality (x y : ℚ) (hx : x = 4 / 7) (hy : y = 5 / 11) : 
  (7 * x + 11 * y) / (77 * x * y) = 9 / 20 :=
by
  -- proof can be provided here.
  sorry

end fraction_equality_l582_582394


namespace part_a_part_b_part_c_part_d_l582_582691

def hexagon := Fin 6

def neigh (v : hexagon) : List hexagon :=
  match v with
  | 0 => [1, 5]
  | 1 => [0, 2]
  | 2 => [1, 3]
  | 3 => [2, 4]
  | 4 => [3, 5]
  | 5 => [0, 4]
  | _ => []

def steps (n : ℕ) (start finish : hexagon) : ℕ := sorry

theorem part_a (n : ℕ) : n % 2 = 0 → steps n 0 2 = 1/3 * (4 ^ (n / 2) - 1) := sorry

theorem part_b (n : ℕ) : n % 2 = 0 → steps n 0 2 / 3 ^ (n / 2 - 1) = 1 := sorry

def alive_prob (n : ℕ) : ℚ := sorry

theorem part_c (n : ℕ) : n % 2 = 0 → alive_prob n = (3/4) ^ (n / 2 - 1) := sorry

theorem part_d : avg_life_expectancy = 9 := sorry

end part_a_part_b_part_c_part_d_l582_582691


namespace separation_of_homologous_chromosomes_only_in_meiosis_l582_582279

-- We start by defining the conditions extracted from the problem.
def chromosome_replication (phase: String) : Prop :=  
  phase = "S phase"

def separation_of_homologous_chromosomes (process: String) : Prop := 
  process = "meiosis I"

def separation_of_chromatids (process: String) : Prop := 
  process = "mitosis anaphase" ∨ process = "meiosis II anaphase II"

def cytokinesis (end_phase: String) : Prop := 
  end_phase = "end mitosis" ∨ end_phase = "end meiosis"

-- Now, we state that the separation of homologous chromosomes does not occur during mitosis.
theorem separation_of_homologous_chromosomes_only_in_meiosis :
  ∀ (process: String), ¬ separation_of_homologous_chromosomes "mitosis" := 
sorry

end separation_of_homologous_chromosomes_only_in_meiosis_l582_582279


namespace problem_correctness_p_plus_q_result_l582_582727

noncomputable def probability_fourth_roll_six
  (p_fair : ℚ := 1/6)
  (p_biased_1 : ℚ := 1/3)
  (p_biased_2 : ℚ := 1/3)
  (p_biased_rest : ℚ := 1/15) 
  (prior_fair : ℚ := 1/2)
  (prior_biased : ℚ := 1/2)
  (likelihood_fair : ℚ := (1/6)^3)
  (likelihood_biased : ℚ := (1/3)^3) :
  ℚ :=
let posterior_fair := prior_fair * likelihood_fair / (prior_fair * likelihood_fair + prior_biased * likelihood_biased) in
let posterior_biased := prior_biased * likelihood_biased / (prior_fair * likelihood_fair + prior_biased * likelihood_biased) in
posterior_fair * p_fair + posterior_biased * p_biased_2

theorem problem_correctness : probability_fourth_roll_six = 17/54 :=
sorry

theorem p_plus_q_result : 
  let (p, q) := (17, 54) in
  p + q = 71 :=
by {
  simp,
}

end problem_correctness_p_plus_q_result_l582_582727


namespace degree_of_polynomial_l582_582306

def polynomial : polynomial ℚ :=
  3 * polynomial.X ^ 5 + 7 * polynomial.Y ^ 3 + 4 * polynomial.X ^ 2 * polynomial.Y ^ 2
  + 9 * polynomial.X + 50 * polynomial.Y + 15

theorem degree_of_polynomial : polynomial.degree polynomial = 5 :=
by sorry

end degree_of_polynomial_l582_582306


namespace problem1_problem2_problem3_l582_582033

-- Proof Problem 1
theorem problem1 : -12 - (-18) + (-7) = -1 := 
by {
  sorry
}

-- Proof Problem 2
theorem problem2 : ((4 / 7) - (1 / 9) + (2 / 21)) * (-63) = -35 := 
by {
  sorry
}

-- Proof Problem 3
theorem problem3 : ((-4) ^ 2) / 2 + 9 * (-1 / 3) - abs (3 - 4) = 4 := 
by {
  sorry
}

end problem1_problem2_problem3_l582_582033


namespace find_constants_PQR_l582_582786

theorem find_constants_PQR :
  ∃ P Q R : ℚ, 
    (P = (-8 / 15)) ∧ 
    (Q = (-7 / 6)) ∧ 
    (R = (27 / 10)) ∧
    (∀ x : ℚ, 
      (x - 1) ≠ 0 ∧ (x - 4) ≠ 0 ∧ (x - 6) ≠ 0 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6)) :=
by
  sorry

end find_constants_PQR_l582_582786


namespace find_prime_p_l582_582554

def f (x : ℕ) : ℕ :=
  (x^4 + 2 * x^3 + 4 * x^2 + 2 * x + 1)^5

theorem find_prime_p : ∃! p, Nat.Prime p ∧ f p = 418195493 := by
  sorry

end find_prime_p_l582_582554


namespace find_k_l582_582209

variables (a b c : ℝ^3) (k : ℝ)
variable [inner_product_space ℝ ℝ^3]

-- Conditions
hypothesis (h1 : ∥a∥ = 1)
hypothesis (h2 : ∥b∥ = 1)
hypothesis (h3 : ∥c∥ = 1)
hypothesis (h4 : ⟪a, b⟫ = 0)
hypothesis (h5 : ⟪a, c⟫ = 0)
hypothesis (h6 : real.angle b c = real.pi / 3)

-- Equivalent proof problem
theorem find_k : a = k • (b ⊗ c) ↔ k = 2*sqrt 3/3 ∨ k = -2*sqrt 3/3 :=
sorry

end find_k_l582_582209


namespace chip_credit_card_balance_l582_582034

theorem chip_credit_card_balance : 
  let C_i := 50.0 in
  let Rate := 0.20 in
  let Extra := 20.0 in
  let first_month_interest := C_i * Rate in
  let balance_after_first_month := C_i + first_month_interest in
  let balance_after_adding_20 := balance_after_first_month - Extra in
  let second_month_interest := balance_after_adding_20 * Rate in
  let final_balance := balance_after_adding_20 + second_month_interest in
  final_balance = 48.0 :=
by
  sorry

end chip_credit_card_balance_l582_582034


namespace probability_first_card_heart_second_king_l582_582646

theorem probability_first_card_heart_second_king :
  ∀ (deck : Finset ℕ) (is_heart : ℕ → Prop) (is_king : ℕ → Prop),
  deck.card = 52 →
  (∀ card ∈ deck, is_heart card ∨ ¬ is_heart card) →
  (∀ card ∈ deck, is_king card ∨ ¬ is_king card) →
  (∃ p : ℚ, p = 1/52) :=
by
  intros deck is_heart is_king h_card h_heart h_king,
  sorry

end probability_first_card_heart_second_king_l582_582646


namespace least_pos_int_solution_l582_582429

theorem least_pos_int_solution (x : ℤ) : x + 4609 ≡ 2104 [ZMOD 12] → x = 3 := by
  sorry

end least_pos_int_solution_l582_582429


namespace monotonicity_a_eq_1_range_of_a_inequality_proof_l582_582474

-- Part 1: Monotonicity when a = 1
def f (x : ℝ) : ℝ := x * Real.exp(x) - Real.exp(x)

theorem monotonicity_a_eq_1 :
  (∀ x : ℝ, 0 < x → (0 < deriv f x)) ∧ (∀ x : ℝ, x < 0 → (deriv f x < 0)) :=
by
  sorry

-- Part 2: Range of values for 'a' when f(x) < -1 and x > 0
def g (a x : ℝ) : ℝ := x * Real.exp(a * x) - Real.exp(x)

theorem range_of_a (a : ℝ) (x : ℝ) (h : 0 < x) (hf : g a x < -1) : a ≤ 1 / 2 :=
by
  sorry

-- Part 3: Inequality proof for natural numbers
def sum_ineq (n : ℕ) : ℝ := ∑ i in Finset.range n, 1 / Real.sqrt (↑i ^ 2 + ↑i)

theorem inequality_proof (n : ℕ) (h : 0 < n) : sum_ineq n > Real.log (n+1) :=
by
  sorry

end monotonicity_a_eq_1_range_of_a_inequality_proof_l582_582474


namespace duration_of_spliced_video_l582_582724

/-- Vasya takes 8 minutes to walk from home to school -/
def vasya_time : ℕ := 8

/-- Petya takes 5 minutes to walk from home to school -/
def petya_time : ℕ := 5

/-- The duration of the resulting spliced video is the sum of Vasya's journey time
    to the midpoint and Petya's journey from this midpoint to home in reverse -/
theorem duration_of_spliced_video (vasya_time petya_time : ℕ) (h₁ : vasya_time = 8) (h₂ : petya_time = 5) :
  let merge_time := (vasya_time / 2) + petya_time in
  merge_time = 13 :=
sorry

end duration_of_spliced_video_l582_582724


namespace principal_is_400_l582_582660

-- Define the conditions
def rate_of_interest : ℚ := 12.5
def simple_interest : ℚ := 100
def time_in_years : ℚ := 2

-- Define the formula for principal amount based on the given conditions
def principal_amount (SI R T : ℚ) : ℚ := SI * 100 / (R * T)

-- Prove that the principal amount is 400
theorem principal_is_400 :
  principal_amount simple_interest rate_of_interest time_in_years = 400 := 
by
  simp [principal_amount, simple_interest, rate_of_interest, time_in_years]
  sorry

end principal_is_400_l582_582660


namespace problem_1_problem_2_l582_582481

def f1 (x : ℝ) := |x + 1| + |x - 3|

def f2 (a x : ℝ) := |x + a^2| + |x + 2a - 5|

theorem problem_1 (x : ℝ) : f1 x < 5 ↔ (-3 / 2 : ℝ) < x ∧ x < 7 / 2 :=
sorry

theorem problem_2 (a : ℝ) : (∃ x, f2 a x < 5) ↔ 0 < a ∧ a < 2 :=
sorry

end problem_1_problem_2_l582_582481


namespace trig_identity_l582_582672

-- Define the statement with the trigonometric functions and the known identities
theorem trig_identity (α : ℝ) : 4.44 * tan (2 * α) + cot (2 * α) + tan (6 * α) + cot (6 * α) = (8 * (cos (4 * α))^2) / (sin (12 * α)) := 
sorry

end trig_identity_l582_582672


namespace two_digit_number_multiple_l582_582709

noncomputable def is_divisible_by (n : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, n = k * d

theorem two_digit_number_multiple :
  ∃ n, (10 ≤ n ∧ n < 100) ∧ is_divisible_by n 8 ∧ is_divisible_by n 12 ∧ is_divisible_by n 18 ∧ (60 ≤ n ∧ n ≤ 79) :=
begin
  sorry
end

end two_digit_number_multiple_l582_582709


namespace gravitational_force_geostationary_l582_582996

variable (d f : ℝ) (k : ℝ := 600 * 4000^2) 

theorem gravitational_force_geostationary : 
  (d = 22300) → f = k / d^2 → f = 96163 / 5000 := 
by
  assume h1 : d = 22300
  assume h2 : f = k / d^2
  rw [h1] at h2
  sorry

end gravitational_force_geostationary_l582_582996


namespace johns_age_is_25_l582_582548

variable (JohnAge DadAge SisterAge : ℕ)

theorem johns_age_is_25
    (h1 : JohnAge = DadAge - 30)
    (h2 : JohnAge + DadAge = 80)
    (h3 : SisterAge = JohnAge - 5) :
    JohnAge = 25 := 
sorry

end johns_age_is_25_l582_582548


namespace arithmetic_seq_sum_equality_l582_582814

theorem arithmetic_seq_sum_equality (a : ℕ → ℝ) (d : ℝ) (h_d : d = -2) 
(h_sum : (finset.range 34).sum (λ k, a (1 + 3 * k)) = 50) :
(finset.range 34).sum (λ k, a (3 + 3 * k)) = -82 := 
sorry

end arithmetic_seq_sum_equality_l582_582814


namespace part1_part2_l582_582346

noncomputable def calculate_prob_A_lunch_given_A_dinner
(P_Al : ℚ) (P_Bl : ℚ) (P_A_d_given_Al : ℚ) (P_B_d_given_Al : ℚ) (P_A_d_given_Bl : ℚ) (P_B_d_given_Bl : ℚ) : ℚ :=
  let P_A_and_A_d := P_Al * P_A_d_given_Al
  let P_B_and_A_d := P_Bl * P_A_d_given_Bl
  let P_A_d := P_A_and_A_d + P_B_and_A_d
  P_A_and_A_d / P_A_d

theorem part1
  (P_Al : ℚ) (P_Bl : ℚ) (P_A_d_given_Al : ℚ) (P_B_d_given_Al : ℚ) (P_A_d_given_Bl : ℚ) (P_B_d_given_Bl : ℚ) :
  P_Al = 2/3 ∧ P_Bl = 1/3 ∧ P_A_d_given_Al = 1/4 ∧ P_B_d_given_Al = 3/4 ∧ P_A_d_given_Bl = 1/2 ∧ P_B_d_given_Bl = 1/2 →
  calculate_prob_A_lunch_given_A_dinner P_Al P_Bl P_A_d_given_Al P_B_d_given_Al P_A_d_given_Bl P_B_d_given_Bl = 1/2 :=
sorry

-- For part 2, we need to define the binomial distribution and its expectations
noncomputable def X_distribution 
(n : ℕ) (p : ℚ) : Fin n → ℚ 
| 0 => (1 - p) ^ n
| k + 1 => ↑(Nat.choose n (k + 1)) * p^(k + 1) * (1 - p)^(n - (k + 1))

noncomputable def expected_X (n : ℕ) (p : ℚ) : ℚ :=
  ∑ i in Finset.range (n + 1), (↑i) * (X_distribution n p i)

theorem part2
  (P_Al : ℚ) (P_Bl : ℚ) (P_A_d_given_Al : ℚ) (P_B_d_given_Al : ℚ) (P_A_d_given_Bl : ℚ) (P_B_d_given_Bl : ℚ) :
  P_Al = 2/3 ∧ P_Bl = 1/3 ∧ P_A_d_given_Al = 1/4 ∧ P_B_d_given_Al = 3/4 ∧ P_A_d_given_Bl = 1/2 ∧ P_B_d_given_Bl = 1/2 →
  let P_B_d := (2/3) * (3/4) + (1/3) * (1/2) in
  X_distribution 4 P_B_d = ![(1/81), (8/81), (8/27), (32/81), (16/81)] ∧ 
  expected_X 4 P_B_d = 8/3 :=
sorry


end part1_part2_l582_582346


namespace α_is_contrapositive_of_β_l582_582463

-- Define the propositions α and β
def α (x : ℝ) : Prop := x < 3 → x < 5
def β (x : ℝ) : Prop := x ≥ 5 → x ≥ 3

-- State the main theorem
theorem α_is_contrapositive_of_β (x : ℝ) : (α x ↔ ∀ (x : ℝ), β x) :=
begin
  sorry
end

end α_is_contrapositive_of_β_l582_582463


namespace annual_interest_rate_is_10_percent_l582_582005

-- Define the principal amount borrowed
def principal : ℤ := 150

-- Define the total amount repaid after one year
def total_repaid : ℤ := 165

-- The annual interest rate calculated using given conditions
def annual_interest_rate (P R : ℕ) : ℚ :=
  ((R - P : ℕ) / P : ℚ) * 100

-- The target interest rate to prove
def target_rate : ℚ := 10

-- The theorem to prove that the calculated interest rate is 10%
theorem annual_interest_rate_is_10_percent :
  annual_interest_rate principal total_repaid = target_rate :=
by sorry

end annual_interest_rate_is_10_percent_l582_582005


namespace population_after_three_years_l582_582278

theorem population_after_three_years 
  (initial_population : ℕ)
  (decrease_rate1 decrease_rate2 decrease_rate3 : ℝ)
  (h_initial : initial_population = 15000)
  (h_decrease1 : decrease_rate1 = 0.20)
  (h_decrease2 : decrease_rate2 = 0.15)
  (h_decrease3 : decrease_rate3 = 0.25) :
  let first_year_population := initial_population - (initial_population * decrease_rate1).to_nat in
  let second_year_population := first_year_population - (first_year_population * decrease_rate2).to_nat in
  let third_year_population := second_year_population - (second_year_population * decrease_rate3).to_nat in
  third_year_population = 7650 :=
by {
  rw [h_initial, h_decrease1, h_decrease2, h_decrease3],
  let first_year_population := 15000 - (15000 * 0.20).to_nat,
  let second_year_population := first_year_population - (first_year_population * 0.15).to_nat,
  let third_year_population := second_year_population - (second_year_population * 0.25).to_nat,
  have : first_year_population = 12000 := by norm_num,
  have : second_year_population = 10200 := by norm_num,
  have : third_year_population = 7650 := by norm_num,
  exact this,
}

end population_after_three_years_l582_582278


namespace no_fractional_xy_l582_582765

theorem no_fractional_xy (x y : ℚ) (m n : ℤ) (h1 : 13 * x + 4 * y = m) (h2 : 10 * x + 3 * y = n) : ¬ (¬(x ∈ ℤ) ∨ ¬(y ∈ ℤ)) :=
sorry

end no_fractional_xy_l582_582765


namespace crystal_meal_combinations_l582_582720

-- Definitions for conditions:
def entrees := 4
def drinks := 4
def desserts := 3 -- includes two desserts and the option of no dessert

-- Statement of the problem as a theorem:
theorem crystal_meal_combinations : entrees * drinks * desserts = 48 := by
  sorry

end crystal_meal_combinations_l582_582720


namespace circle_value_in_grid_l582_582525

theorem circle_value_in_grid :
  ∃ (min_circle_val : ℕ), min_circle_val = 21 ∧ (∀ (max_circle_val : ℕ), ∃ (L : ℕ), L > max_circle_val) :=
by
  sorry

end circle_value_in_grid_l582_582525


namespace number_of_factors_of_N_l582_582155

def N : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_factors_of_N : 
  ∃ (factors : ℕ), factors = 5 * 4 * 3 * 2 ∧ factors = 120 :=
by
  use 5 * 4 * 3 * 2
  split
  · refl
  · refl
  sorry

end number_of_factors_of_N_l582_582155


namespace dice_probability_four_less_than_five_l582_582773

noncomputable def probability_exactly_four_less_than_five (n : ℕ) : ℚ :=
  if n = 8 then (Nat.choose 8 4) * (1 / 2)^8 else 0

theorem dice_probability_four_less_than_five : probability_exactly_four_less_than_five 8 = 35 / 128 :=
by
  -- statement is correct, proof to be provided
  sorry

end dice_probability_four_less_than_five_l582_582773


namespace divide_cards_into_sums_l582_582620

theorem divide_cards_into_sums (n : ℕ) (x : Fin (2 * n) → ℝ) 
  (h : ∀ i, 1 ≤ x i ∧ x i ≤ 2) 
  (h_sorted : ∀ i j, i < j → x i ≤ x j) :
  ∃ (s1 s2 : ℝ), 
    s1 = ∑ i in (Finset.range n).filter (λ k, k % 2 = 0), x (Fin.mk k sorry) ∧
    s2 = ∑ i in (Finset.range n).filter (λ k, k % 2 = 1), x (Fin.mk k sorry) ∧
    (n : ℝ) / (n + 1) ≤ s1 / s2 ∧ s1 / s2 ≤ 1 := 
sorry

end divide_cards_into_sums_l582_582620


namespace int_pairs_satisfy_eq_l582_582778

theorem int_pairs_satisfy_eq (x y : ℤ) : (x^2 = y^2 + 2 * y + 13) ↔ ((x = 4 ∧ y = 1) ∨ (x = -4 ∧ y = -5)) :=
by 
  sorry

end int_pairs_satisfy_eq_l582_582778


namespace valid_ways_to_assign_volunteers_l582_582679

noncomputable def validAssignments : ℕ := 
  (Nat.choose 5 2) * (Nat.choose 3 2) + (Nat.choose 5 1) * (Nat.choose 4 2)

theorem valid_ways_to_assign_volunteers : validAssignments = 60 := 
  by
    simp [validAssignments]
    sorry

end valid_ways_to_assign_volunteers_l582_582679


namespace real_number_condition_pure_imaginary_condition_on_line_condition_l582_582987
variable (m : ℝ)
-- Define the complex number z
noncomputable def z := (1 + complex.i) * m^2 + (5 - 2 * complex.i) * m + (6 - 15 * complex.i)
-- Real part of z
noncomputable def real_part := (m^2 + 5 * m + 6 : ℝ)
-- Imaginary part of z
noncomputable def imag_part := (m^2 - 2 * m - 15 : ℝ)

-- Proving the conditions
theorem real_number_condition : imag_part m = 0 ↔ (m = 5 ∨ m = -3) := sorry
theorem pure_imaginary_condition : real_part m = 0 ∧ imag_part m ≠ 0 ↔ (m = -2) := sorry
theorem on_line_condition : real_part m + imag_part m + 7 = 0 ↔ (m = 1/2 ∨ m = -2) := sorry

end real_number_condition_pure_imaginary_condition_on_line_condition_l582_582987


namespace probability_of_satisfying_condition_l582_582867

-- Let p be an integer between 1 and 15 inclusive
def is_valid_p (p : ℤ) : Prop := 1 ≤ p ∧ p ≤ 15

-- Define the equation condition
def satisfies_equation (p q : ℤ) : Prop := p * q - 5 * p - 3 * q = 3

-- Define the probability question: probability that there exists q such that p and q satisfy the equation
theorem probability_of_satisfying_condition : 
  (∃ p ∈ { p : ℤ | is_valid_p p }, ∃ q : ℤ, satisfies_equation p q) →
  (finset.filter (λ p : ℤ, ∃ q : ℤ, satisfies_equation p q) (finset.Icc 1 15)).card / (finset.Icc 1 15).card = 1 / 3 :=
sorry

end probability_of_satisfying_condition_l582_582867


namespace sum_first_six_terms_minimum_l582_582820

noncomputable def arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ q : ℝ, a (n+1) = q * a n

theorem sum_first_six_terms_minimum (a : ℕ → ℝ) (q : ℝ)
    (h1 : arithmetic_geometric_sequence a)
    (h2 : 2 * (a 3 + a 4) = 2 - a 1 - a 2) :
    (∃ q : ℝ, q = (sqrt 3 - 1) / 2 ∧ 
     ∀ S6 : ℝ, S6 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 → 
     S6 = Real.sqrt 3) :=
sorry

end sum_first_six_terms_minimum_l582_582820


namespace minute_hand_turns_6_degrees_per_minute_hour_hand_turns_0_5_degrees_per_minute_l582_582390

theorem minute_hand_turns_6_degrees_per_minute (full_revolution_minutes : ℕ) (minutes_per_hour : ℕ) (degrees_per_revolution : ℕ) :
  full_revolution_minutes = 60 →
  minutes_per_hour = 60 →
  degrees_per_revolution = 360 →
  (degrees_per_revolution / full_revolution_minutes) = 6 :=
by
  intros
  sorry

theorem hour_hand_turns_0_5_degrees_per_minute (full_revolution_hours : ℕ) (hours_per_revolution : ℕ) (minutes_per_hour : ℕ) (degrees_per_revolution : ℕ) :
  full_revolution_hours = 12 →
  hours_per_revolution = 12 →
  minutes_per_hour = 60 →
  degrees_per_revolution = 360 →
  (degrees_per_revolution / full_revolution_hours / minutes_per_hour) = 0.5 :=
by
  intros
  sorry

end minute_hand_turns_6_degrees_per_minute_hour_hand_turns_0_5_degrees_per_minute_l582_582390


namespace unique_arrangements_boo_boo_l582_582054

theorem unique_arrangements_boo_boo : 
  let word := "BOOBOO" 
  let nB := 2 
  let nO := 4 
  let totalLetters := 6 in 
  ∑ (⅓) / (⅓.factorial * (⅗).factorial) = 15 := 
sorry

end unique_arrangements_boo_boo_l582_582054


namespace ratio_x_y_l582_582908

-- Definitions for conditions
variable (d : ℝ) -- Total distance
variable (x : ℝ) -- Total time taken by Jill
variable (y : ℝ) -- Total time taken by Jack

-- Conditions derived from the problem
def jill_time (d : ℝ) : ℝ := d / 12 + d / 24
def jack_time (d : ℝ) : ℝ := d / 15 + 2 * d / 45

-- Proof statement
theorem ratio_x_y (d : ℝ) : jill_time d = x → jack_time d = y → x / y = 9 / 8 :=
by
  intros hjill hjack
  sorry

end ratio_x_y_l582_582908


namespace probability_first_spade_second_king_l582_582636

/--
In a standard deck of 52 cards, the probability of drawing the first card as a ♠ and the second card as a king is 1/52.
-/
theorem probability_first_spade_second_king : 
  let deck_size := 52 in
  let hearts_count := 13 in
  let kings_count := 4 in
  let prob := (1 / deck_size : ℚ) * (kings_count / (deck_size - 1)) + ((hearts_count - 1) / deck_size) * (kings_count / (deck_size - 1)) 
  in 
  prob = 1 / deck_size :=
by
  sorry

end probability_first_spade_second_king_l582_582636


namespace parabola_equation_l582_582837

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the parabola C2 with parameter p
def C2 (p x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

-- Define the distance between two points A and B
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The main theorem to prove:
theorem parabola_equation (p : ℝ) : 
  (∃ A B : ℝ × ℝ, C1 A.1 A.2 ∧ C1 B.1 B.2 ∧ C2 p A.1 A.2 ∧ C2 p B.1 B.2 ∧ distance A B = 8 * real.sqrt 5 / 5)
    → y^2 = 32 / 5 * x :=
sorry

end parabola_equation_l582_582837


namespace find_k_eq_l582_582319

theorem find_k_eq (n : ℝ) (k m : ℤ) (h : ∀ n : ℝ, n * (n + 1) * (n + 2) * (n + 3) + m = (n^2 + k * n + 1)^2) : k = 3 := 
sorry

end find_k_eq_l582_582319


namespace theater_group_min_students_exists_l582_582367

theorem theater_group_min_students_exists : 
  ∃ n : ℕ, (n % 9 = 0) ∧ (n % 10 = 0) ∧ (n % 11 = 0) ∧ (n % 12 = 1) ∧ (∀ m : ℕ, (m % 9 = 0) ∧ (m % 10 = 0) ∧ (m % 11 = 0) ∧ (m % 12 = 1) → n ≤ m) :=
begin
  -- Proof goes here
  sorry
end

end theater_group_min_students_exists_l582_582367


namespace probability_units_digit_odd_l582_582690

theorem probability_units_digit_odd :
  (1 / 2 : ℚ) = 5 / 10 :=
by {
  -- This is the equivalent mathematically correct theorem statement
  -- The proof is omitted as per instructions
  sorry
}

end probability_units_digit_odd_l582_582690


namespace complex_sum_evaluation_l582_582936

noncomputable def sum_evaluation (ω : ℂ) (b : ℕ → ℝ) (n : ℕ) : ℂ :=
  ∑ k in Finset.range n, (3 * b k + 1) / ((b k)^2 - b k + 1)

theorem complex_sum_evaluation
  (ω : ℂ)
  (hω1 : ω^3 = 1)
  (hω2 : ω ≠ 1)
  (hω3 : ω^2 + ω + 1 = 0)
  (b : ℕ → ℝ)
  (n : ℕ)
  (h1 : ∑ k in Finset.range n, 1 / (b k + ω) = 3 - 4 * complex.I) :
  sum_evaluation ω b n = 12 :=
sorry

end complex_sum_evaluation_l582_582936


namespace partition_lords_l582_582591

theorem partition_lords (G : SimpleGraph V) (h_deg : ∀ v : V, G.degree v ≤ 3) : 
  ∃ (A B : Set V), A ∩ B = ∅ ∧ A ∪ B = Set.univ ∧ ∀ v ∈ A, ∃! u ∈ A, G.Adj v u :=
by
  sorry

end partition_lords_l582_582591


namespace sin_cos_diff_l582_582828

noncomputable def θ : Real := sorry
noncomputable def π : Real := Real.pi

def angle_in_second_quadrant (θ : Real) : Prop := 
  θ > π / 2 ∧ θ < π

def tan_add (θ : Real) : Prop := 
  Real.tan(θ + π / 4) = 1 / 2

theorem sin_cos_diff (θ : Real) (h1 : angle_in_second_quadrant θ) (h2 : tan_add θ) :
  Real.sin θ - Real.cos θ = 2 * Real.sqrt 10 / 5 := 
sorry

end sin_cos_diff_l582_582828


namespace probability_one_instrument_one_sport_l582_582889

theorem probability_one_instrument_one_sport
  (n : ℕ) (n = 1500)
  (inst : ℕ := (3 / 7 : ℚ) * n)
  (sport : ℕ := (5 / 14 : ℚ) * n)
  (both : ℕ := (1 / 6 : ℚ) * n)
  (two_or_more_inst : ℤ := 0.095 * inst)
  (one_inst := inst - two_or_more_inst)
  (one_sport := sport - both)
  (probability := both / n) :
  probability = 1 / 6 :=
by sorry

end probability_one_instrument_one_sport_l582_582889


namespace expected_rolls_in_leap_year_l582_582375

theorem expected_rolls_in_leap_year :
  let E := (3/4) * 1 + (1/4) * (1 + E) in  -- Expected value equation
  E = 4/3 →
  let E_total := E * 366 in
  E_total = 488 :=
by
  sorry

end expected_rolls_in_leap_year_l582_582375


namespace liam_arrival_time_l582_582224

-- Definitions
def commute_distance : ℝ := 40
def actual_speed : ℝ := 60
def reduced_speed : ℝ := 55

-- Statements
theorem liam_arrival_time : 
  let time_actual := commute_distance / actual_speed in
  let time_reduced := commute_distance / reduced_speed in
  (time_reduced - time_actual) * 60 = 3.64 :=
by
  sorry

end liam_arrival_time_l582_582224


namespace kevin_bought_3_muffins_l582_582205

theorem kevin_bought_3_muffins (m : ℕ) (h1 : 0.75 * m + 1.45 = 3.70) : m = 3 :=
by
  sorry

end kevin_bought_3_muffins_l582_582205


namespace organization_population_after_five_years_l582_582362

noncomputable def b : ℕ → ℝ
| 0     := 25
| (n+1) := 2.7 * (b n - 5) + 5

theorem organization_population_after_five_years :
  b 5 ≈ 2875 :=
sorry

end organization_population_after_five_years_l582_582362


namespace seashells_total_after_giving_l582_582860

/-- Prove that the total number of seashells among Henry, Paul, and Leo is 53 after Leo gives away a quarter of his collection. -/
theorem seashells_total_after_giving :
  ∀ (henry_seashells paul_seashells total_initial_seashells leo_given_fraction : ℕ),
    henry_seashells = 11 →
    paul_seashells = 24 →
    total_initial_seashells = 59 →
    leo_given_fraction = 1 / 4 →
    let leo_seashells := total_initial_seashells - henry_seashells - paul_seashells in
    let leo_seashells_after := leo_seashells - (leo_seashells * leo_given_fraction) in
    henry_seashells + paul_seashells + leo_seashells_after = 53 :=
by
  intros
  sorry

end seashells_total_after_giving_l582_582860


namespace apples_per_pie_l582_582259

-- Definitions of given conditions
def total_apples : ℕ := 75
def handed_out_apples : ℕ := 19
def remaining_apples : ℕ := total_apples - handed_out_apples
def pies_made : ℕ := 7

-- Statement of the problem to be proved
theorem apples_per_pie : remaining_apples / pies_made = 8 := by
  sorry

end apples_per_pie_l582_582259


namespace proof_inequality_l582_582808

noncomputable def problem_statement (n : ℕ) (x : Fin n → ℝ) : Prop :=
  (n > 3) → 
  (∀ i : Fin n, x i > 0) → 
  (∏ i, x i = 1) → 
  (∑ i : Fin n, 1 / (1 + x i + x i * x ((i + 1) % n).val)) > 1

theorem proof_inequality {n : ℕ} (x : Fin n → ℝ) : 
  problem_statement n x :=
begin
  intros,
  sorry
end

end proof_inequality_l582_582808


namespace rational_expression_value_l582_582125

noncomputable def rational_abs_div (x y z : ℚ) (h1 : x + y + z = 0) (h2 : x * y * z ≠ 0) : ℝ :=
  abs (x : ℝ) / (y + z : ℝ) + abs (y : ℝ) / (x + z : ℝ) + abs (z : ℝ) / (x + y : ℝ)

theorem rational_expression_value {x y z : ℚ} (h1 : x + y + z = 0) (h2 : x * y * z ≠ 0) :
  rational_abs_div x y z h1 h2 = 1 ∨ rational_abs_div x y z h1 h2 = -1 :=
begin
  sorry,
end

end rational_expression_value_l582_582125


namespace line_intersects_ellipse_max_chord_length_l582_582472

theorem line_intersects_ellipse (m : ℝ) : 
  (∃ x y : ℝ, (y = (3/2 : ℝ) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1)) ↔ 
  (-3 * Real.sqrt 2 ≤ m ∧ m ≤ 3 * Real.sqrt 2) := 
by sorry

theorem max_chord_length : 
  (∃ m : ℝ, (m = 0) ∧ 
    (∀ x y x1 y1 : ℝ, (y = (3/2 : ℝ) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1) ∧ 
     (y1 = (3/2 : ℝ) * x1 + m) ∧ (x1^2 / 4 + y1^2 / 9 = 1) ∧ 
     (x ≠ x1 ∨ y ≠ y1) → 
     (Real.sqrt (13 / 9) * Real.sqrt (18 - m^2) = Real.sqrt 26))) := 
by sorry

end line_intersects_ellipse_max_chord_length_l582_582472


namespace number_of_blueberries_l582_582337

def total_berries : ℕ := 42
def raspberries : ℕ := total_berries / 2
def blackberries : ℕ := total_berries / 3
def blueberries : ℕ := total_berries - (raspberries + blackberries)

theorem number_of_blueberries :
  blueberries = 7 :=
by
  sorry

end number_of_blueberries_l582_582337


namespace find_angle_A_find_area_of_triangle_l582_582885

-- Part 1: Prove that ∠A = π/3
theorem find_angle_A
(a b : ℝ)
(h_parallel : (a, (Real.sqrt 3) * b) = λ k, (k * Real.cos (Real.pi / 3), k * Real.sin (B))) :
A = Real.pi / 3 :=
sorry

-- Part 2: Prove the area of ΔABC is 3√3/2 given a = √7, b = 2, and ∠A = π/3
theorem find_area_of_triangle
(a b c : ℝ)
(h_a : a = Real.sqrt 7)
(h_b : b = 2)
(h_A : A = Real.pi / 3)
(h_c : c = 3) :
(1 / 2) * b * c * Real.sin (Real.pi / 3) = (3 * Real.sqrt 3) / 2 :=
sorry

end find_angle_A_find_area_of_triangle_l582_582885


namespace smallest_positive_value_of_e_l582_582040

noncomputable def polynomial : ℕ → ℕ → ℕ → ℕ → ℕ → (ℚ → ℚ) := sorry

theorem smallest_positive_value_of_e :
  ∃ a b c d e : ℤ, 
    (polynomial a b c d e = 0) ∧ 
    (roots a b c d e = [-4, 3, 7, 1 / 2]) ∧ 
    (e > 0) ∧ 
    (∀ (e' : ℤ), (e' = ((-4) * 3 * 7 * (1 / 2) * (-1)^4) * (-1)) → (e' > 0) → (e ≤ e')) :=
sorry

end smallest_positive_value_of_e_l582_582040


namespace choose_three_positive_or_two_negative_l582_582671

theorem choose_three_positive_or_two_negative (n : ℕ) (hn : n ≥ 3) (a : Fin n → ℝ) :
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (0 < a i + a j + a k) ∨ ∃ (i j : Fin n), i ≠ j ∧ (a i + a j < 0) := sorry

end choose_three_positive_or_two_negative_l582_582671


namespace min_inequality_l582_582934

theorem min_inequality (r s u v : ℝ) : 
  min (min (r - s^2) (min (s - u^2) (min (u - v^2) (v - r^2)))) ≤ 1 / 4 :=
by sorry

end min_inequality_l582_582934


namespace no_fractional_x_y_l582_582759

theorem no_fractional_x_y (x y : ℚ) (H1 : ¬ (x.denom = 1 ∧ y.denom = 1)) (H2 : ∃ m : ℤ, 13 * x + 4 * y = m) (H3 : ∃ n : ℤ, 10 * x + 3 * y = n) : false :=
sorry

end no_fractional_x_y_l582_582759


namespace octal_addition_l582_582088

theorem octal_addition (x y : ℕ) (h1 : x = 1 * 8^3 + 4 * 8^2 + 6 * 8^1 + 3 * 8^0)
                     (h2 : y = 2 * 8^2 + 7 * 8^1 + 5 * 8^0) :
  x + y = 1 * 8^3 + 7 * 8^2 + 5 * 8^1 + 0 * 8^0 := sorry

end octal_addition_l582_582088


namespace max_score_proof_l582_582172

-- Define the conditions
def conditions (scores: list ℕ) : Prop :=
  scores.length = 12 ∧ -- team of 12 players 
  (∀ s ∈ scores, 7 ≤ s ∧ s ≤ 20) ∧  -- Each player scores between 7 and 20 points
  (∀ s ∈ scores, s ≤ 10 + 10 + 10) ∧ -- No player takes more than 10 shots (3-pointers capped implicitly by score limit)
  list.sum scores = 100 -- The team scores a total of 100 points

-- Define the maximum score to be proven from the conditions
def max_individual_score (scores: list ℕ) : ℕ :=
  list.maximum scores

theorem max_score_proof (scores: list ℕ) (h : conditions scores) : max_individual_score scores = 20 :=
sorry -- proof

end max_score_proof_l582_582172


namespace projection_onto_plane_l582_582929

-- Define the plane Q passing through the origin
def PlaneQ (n : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) := n.1 * p.1 + n.2 * p.2 + n.3 * p.3 = 0

-- Given the normal vector n to the plane Q
def n : ℝ × ℝ × ℝ := (3, 4, 2)

-- The projection of vector v₁ onto plane Q is defined
def v₁ : ℝ × ℝ × ℝ := (4, 7, 3)
def p₁ : ℝ × ℝ × ℝ := (1, 3, 1)

-- Given vector v to be projected onto the plane Q
def v : ℝ × ℝ × ℝ := (3, 1, 4)

-- The resulting projection vector p
def p : ℝ × ℝ × ℝ := (47/29, 9/29, 38/29)

-- Prove that the projection of vector v onto the plane Q results in the vector p
theorem projection_onto_plane : 
  let 
    nlen_sq := n.1 * n.1 + n.2 * n.2 + n.3 * n.3,
    projection_scalar := (v.1 * n.1 + v.2 * n.2 + v.3 * n.3) / nlen_sq,
    v_parallel := (projection_scalar * n.1, projection_scalar * n.2, projection_scalar * n.3),
    v_perpendicular := (v.1 - v_parallel.1, v.2 - v_parallel.2, v.3 - v_parallel.3)
  in 
    v_perpendicular = p :=
  sorry

end projection_onto_plane_l582_582929


namespace seashells_total_now_l582_582857

def henry_collected : ℕ := 11
def paul_collected : ℕ := 24
def total_initial : ℕ := 59
def leo_initial (henry_collected paul_collected total_initial : ℕ) : ℕ := total_initial - (henry_collected + paul_collected)
def leo_gave (leo_initial : ℕ) : ℕ := leo_initial / 4
def total_now (total_initial leo_gave : ℕ) : ℕ := total_initial - leo_gave

theorem seashells_total_now :
  total_now total_initial (leo_gave (leo_initial henry_collected paul_collected total_initial)) = 53 :=
sorry

end seashells_total_now_l582_582857


namespace thomas_saves_40_per_month_l582_582626

variables (T J : ℝ) (months : ℝ := 72) 

theorem thomas_saves_40_per_month 
  (h1 : J = (3/5) * T)
  (h2 : 72 * T + 72 * J = 4608) : 
  T = 40 :=
by sorry

end thomas_saves_40_per_month_l582_582626


namespace max_negatives_l582_582876

theorem max_negatives (a b c d e f : ℤ) (h : ab + cdef < 0) : ∃ w : ℤ, w = 4 := 
sorry

end max_negatives_l582_582876


namespace find_b_vector_l582_582922

noncomputable def b : ℝ × ℝ × ℝ := (129 / 13, -35 / 13, -150 / 13)

def a : ℝ × ℝ × ℝ := (5, -3, -6)

def c : ℝ × ℝ × ℝ := (-3, -2, 3)

def isCollinear (u v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k₁ k₂ : ℝ), u.1 = k₁ * v.1 ∧ u.2 = k₁ * v.2 ∧ u.3 = k₁ * v.3 ∧ w.1 = k₂ * v.1 ∧ w.2 = k₂ * v.2 ∧ w.3 = k₂ * v.3

def bisectsAngle (u v w : ℝ × ℝ × ℝ) : Prop :=
  (u.1 * v.1 + u.2 * v.2 + u.3 * v.3) / (real.sqrt (u.1^2 + u.2^2 + u.3^2) * real.sqrt (v.1^2 + v.2^2 + v.3^2))
  =
  (v.1 * w.1 + v.2 * w.2 + v.3 * w.3) / (real.sqrt (w.1^2 + w.2^2 + w.3^2) * real.sqrt (v.1^2 + v.2^2 + v.3^2))

theorem find_b_vector :
  isCollinear a b c ∧ bisectsAngle a b c :=
by
  sorry

end find_b_vector_l582_582922


namespace tan_pi_over_4_minus_alpha_l582_582803

-- Define the problem
theorem tan_pi_over_4_minus_alpha (α: ℝ) 
  (hα₁: 0 < α) 
  (hα₂: α < real.pi) 
  (hsin: real.sin α = 3/5) 
  : (α < real.pi / 2) → real.tan (real.pi / 4 - α) = 1/7 
    ∧ (α > real.pi / 2) → real.tan (real.pi / 4 - α) = 7 := 
sorry

end tan_pi_over_4_minus_alpha_l582_582803


namespace a_n3_l582_582718

def right_angled_triangle_array (a : ℕ → ℕ → ℚ) : Prop :=
  ∀ i j, 1 ≤ j ∧ j ≤ i →
    (j = 1 → a i j = 1 / 4 + (i - 1) / 4) ∧
    (i ≥ 3 → (1 < j → a i j = a i 1 * (1 / 2)^(j - 1)))

theorem a_n3 (a : ℕ → ℕ → ℚ) (n : ℕ) (h : right_angled_triangle_array a) : a n 3 = n / 16 :=
sorry

end a_n3_l582_582718


namespace convex_37_gon_three_identical_angles_l582_582351

theorem convex_37_gon_three_identical_angles 
  (angles : Fin 37 → ℕ)
  (sum_angles : ∑ i, angles i = 6300) 
  (whole_number_degrees : ∀ i, 1 ≤ angles i ∧ angles i ≤ 179) 
  : ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ angles i = angles j ∧ angles j = angles k :=
  sorry

end convex_37_gon_three_identical_angles_l582_582351


namespace goods_train_speed_correct_l582_582693

def length_train : ℝ := 350.048
def length_platform : ℝ := 250
def time_to_cross : ℝ := 30

noncomputable def speed_of_goods_train_kmph : ℝ :=
  (length_train + length_platform) / time_to_cross * 3.6

theorem goods_train_speed_correct :
  Real.floor (speed_of_goods_train_kmph * 1000) / 1000 = 72.006 := by
  sorry

end goods_train_speed_correct_l582_582693


namespace possible_values_of_a_l582_582846

def has_two_subsets (A : Set ℝ) : Prop :=
  ∃ (x : ℝ), A = {x}

theorem possible_values_of_a (a : ℝ) (A : Set ℝ) :
  (A = {x | a * x^2 + 2 * x + a = 0}) →
  (has_two_subsets A) ↔ (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  intros hA
  sorry

end possible_values_of_a_l582_582846


namespace number_of_natural_factors_of_N_l582_582156

def N : ℕ := 2^4 * 3^3 * 5^2 * 7

theorem number_of_natural_factors_of_N : nat.divisors N = 120 := by
  sorry

end number_of_natural_factors_of_N_l582_582156


namespace initial_number_of_men_l582_582985

theorem initial_number_of_men (M A : ℕ) : 
  (∀ (M A : ℕ), ((M * A) - 40 + 61) / M = (A + 3)) ∧ (30.5 = 30.5) → 
  M = 7 :=
by
  sorry

end initial_number_of_men_l582_582985


namespace min_circles_infinite_intersecting_lines_l582_582906

theorem min_circles (s : ℝ) (k : ℕ) (r : ℕ → ℝ) (h1 : ∑ i in range k, 2 * Real.pi * (r i) = 8 * s) : k ≥ 3 :=
sorry

theorem infinite_intersecting_lines (s : ℝ) (k : ℕ) (r : ℕ → ℝ) (h1 : ∑ i in range k, 2 * Real.pi * (r i) = 8 * s)
(h2 : k ≥ 3) : ∃ f : ℝ → set (Euclidean_space ℝ 2), (∀ x, ∃ l, ∈ infiniteIntersec¡t f x) ∧ (∀ x, l having_property x intersectAtLeastThree r) :=
sorry

end min_circles_infinite_intersecting_lines_l582_582906


namespace largest_n_unique_k_l582_582657

-- Defining the main theorem statement
theorem largest_n_unique_k :
  ∃ (n : ℕ), (n = 63) ∧ (∃! (k : ℤ), (9 / 17 : ℚ) < (n : ℚ) / ((n + k) : ℚ) ∧ (n : ℚ) / ((n + k) : ℚ) < (8 / 15 : ℚ)) :=
sorry

end largest_n_unique_k_l582_582657


namespace arithmetic_sequence_term_count_l582_582089

theorem arithmetic_sequence_term_count (a d l n : ℕ) (h_a : a = 165) (h_d : d = -5) (h_l : l = 30) (h_formula : l = a + (n - 1) * d) : n = 28 := by
  sorry

end arithmetic_sequence_term_count_l582_582089


namespace total_water_carried_l582_582688

noncomputable theory

def num_trucks : ℕ := 3
def tanks_per_truck : ℕ := 3
def liters_per_tank : ℕ := 150

theorem total_water_carried : num_trucks * (tanks_per_truck * liters_per_ttank) = 1350 := 
   sorry

end total_water_carried_l582_582688


namespace possible_values_of_a_l582_582847

def has_two_subsets (A : Set ℝ) : Prop :=
  ∃ (x : ℝ), A = {x}

theorem possible_values_of_a (a : ℝ) (A : Set ℝ) :
  (A = {x | a * x^2 + 2 * x + a = 0}) →
  (has_two_subsets A) ↔ (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  intros hA
  sorry

end possible_values_of_a_l582_582847


namespace angle_AED_obtuse_l582_582968

-- Define the geometric points and distances
variables (A B C D E : Point)

-- Define the conditions of the problem
axiom collinear : Collinear A B C D
axiom plane_containing_line : InPlaneContainingLine E A B C D
axiom AB_eq_BE : dist A B = dist B E
axiom EC_eq_CD : dist E C = dist C D

-- Define relevant angles
def angle_BAE := angle B A E
def angle_AEB := angle A E B
def angle_CDE := angle C D E
def angle_CED := angle C E D
def angle_AED := angle A E D

-- State the problem to be proved
theorem angle_AED_obtuse : obtuse angle_AED := sorry

end angle_AED_obtuse_l582_582968


namespace yanna_gave_9_apples_to_zenny_l582_582323

variable (Z : ℕ)
variable (A : ℕ := 60)
variable (K : ℕ := 36)
variable (A_z : ℕ := Z + 6)

theorem yanna_gave_9_apples_to_zenny (h : Z + A_z + K = A) : Z = 9 :=
by
  have h1 : 2 * Z + 42 = 60 := by
    calc
      Z + A_z + K = A           : h
      Z + (Z + 6) + 36 = 60     : rfl
      2 * Z + 6 + 36 = 60       : by ring
      2 * Z + 42 = 60           : by ring
  have h2 : 2 * Z = 18 := by
    linarith
  have h3 : Z = 18 / 2 := by
    exact Eq.symm (Nat.div_eq_of_eq_mul_left zero_lt_two h2)
  exact Eq.symm (Nat.div_eq_of_eq_mul_left zero_lt_two h2).symm

end yanna_gave_9_apples_to_zenny_l582_582323


namespace relationship_abc_l582_582451

variable (f : ℝ → ℝ)

axiom condition1 : ∀ x1 x2 : ℝ, 4 ≤ x1 ∧ x1 ≤ 8 ∧ 4 ≤ x2 ∧ x2 ≤ 8 ∧ x1 < x2 → (f(x1) - f(x2)) / (x1 - x2) > 0
axiom condition2 : ∀ x : ℝ, f(x + 4) = -f(x)
axiom condition3 : ∀ x : ℝ, f(x + 4) = f(-x + 4)

noncomputable def a : ℝ := f 6
noncomputable def b : ℝ := f 11
noncomputable def c : ℝ := f 2017

theorem relationship_abc : b < a ∧ a < c := sorry

end relationship_abc_l582_582451


namespace polynomial_square_solution_l582_582275

variable (a b : ℝ)

theorem polynomial_square_solution (h : 
  ∃ g : Polynomial ℝ, g^2 = Polynomial.C (1 : ℝ) * Polynomial.X^4 -
  Polynomial.C (1 : ℝ) * Polynomial.X^3 +
  Polynomial.C (1 : ℝ) * Polynomial.X^2 +
  Polynomial.C a * Polynomial.X +
  Polynomial.C b) : b = 9 / 64 :=
by sorry

end polynomial_square_solution_l582_582275


namespace school_dance_l582_582770

theorem school_dance (x : ℕ) (pairs : ℕ) (dancing_pattern : Π i : ℕ, i < x → ℕ)
  (h_sum : (∑ i in Finset.range x, dancing_pattern i (Finset.mem_range.2 i.lt_of_le_pred)) = pairs)
  (h_pattern : ∀ i, i < x → dancing_pattern i (Finset.mem_range.2 i.lt_of_le_pred) = i + 11)
  (h_pairs : pairs = 430) :
  x = 20 ∧ (x + 11) = 31 :=
by
  sorry

end school_dance_l582_582770


namespace meghan_total_money_l582_582946

theorem meghan_total_money :
  let num_100_bills := 2
  let num_50_bills := 5
  let num_10_bills := 10
  let value_100_bills := num_100_bills * 100
  let value_50_bills := num_50_bills * 50
  let value_10_bills := num_10_bills * 10
  let total_money := value_100_bills + value_50_bills + value_10_bills
  total_money = 550 := by sorry

end meghan_total_money_l582_582946


namespace license_plate_increase_l582_582617

def old_license_plates : ℕ := 26 * (10^5)

def new_license_plates : ℕ := 26^2 * (10^4)

theorem license_plate_increase :
  (new_license_plates / old_license_plates : ℝ) = 2.6 := by
  sorry

end license_plate_increase_l582_582617


namespace prime_gt_p_l582_582918

theorem prime_gt_p (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hgt : q > 5) (hdiv : q ∣ 2^p + 3^p) : q > p := 
sorry

end prime_gt_p_l582_582918


namespace exists_x_l582_582783

noncomputable def g (x : ℝ) : ℝ := (2 / 7) ^ x + (3 / 7) ^ x + (6 / 7) ^ x

theorem exists_x (x : ℝ) : ∃ c : ℝ, g c = 1 :=
sorry

end exists_x_l582_582783


namespace fill_box_with_L_blocks_l582_582303

theorem fill_box_with_L_blocks (m n k : ℕ) 
  (hm : m > 1) (hn : n > 1) (hk : k > 1) (hk_div3 : k % 3 = 0) : 
  ∃ (fill : ℕ → ℕ → ℕ → Prop), fill m n k → True := 
by
  sorry

end fill_box_with_L_blocks_l582_582303


namespace evaluate_g_at_i_l582_582557

def g (x : ℂ) : ℂ := (x^6 + x^3) / (x + 2)

theorem evaluate_g_at_i :
  let i := complex.I in g i = -3/5 - (1/5 : ℂ) * complex.I :=
by
  let i := complex.I
  have h : g i = (-3 - complex.I) / 5 := sorry
  rw [h]
  norm_num

end evaluate_g_at_i_l582_582557


namespace find_k_collinear_l582_582121

variable (k : ℝ)

def vector_PB : ℝ × ℝ := (4, 5)
def vector_PA : ℝ × ℝ := (k, 12)
def vector_PC : ℝ × ℝ := (10, k)

def vector_AB : ℝ × ℝ := (fst vector_PB - fst vector_PA, snd vector_PB - snd vector_PA)
def vector_BC : ℝ × ℝ := (fst vector_PC - fst vector_PB, snd vector_PC - snd vector_PB)

def collinear (u v : ℝ × ℝ) : Prop :=
  fst u * snd v = snd u * fst v

theorem find_k_collinear :
  collinear (vector_AB k) (vector_BC k) ↔ (k = -2 ∨ k = 11) :=
by
  sorry

end find_k_collinear_l582_582121


namespace circle_tangent_problem_l582_582489

open Real

def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 3)^2 = 1
def circle2 (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1
def line (x y : ℝ) : Prop := x - y - 1 = 0
def is_tangent (P M : ℝ × ℝ) (r : ℝ) (C : ℝ × ℝ → ℝ) : Prop := 
  (dist P M = r) ∧ (C M = r^2)

theorem circle_tangent_problem (a b : ℝ)
  (h1 : ∀ x y, circle1 x y)
  (h2 : ∀ x y, circle2 x y a b)
  (h3 : ∀ P, line P.1 P.2)
  (h4 : ∃ P, line P.1 P.2 ∧ ∀ (M N : ℝ × ℝ), is_tangent P M 1 circle1 ∧ 
          is_tangent P N 1 (λ Q, circle2 Q.1 Q.2 a b) ∧ 
          dist P M = dist P N) :
  a + b = -2 :=
sorry

end circle_tangent_problem_l582_582489


namespace pens_left_is_25_l582_582002

def total_pens_left (initial_blue initial_black initial_red removed_blue removed_black : Nat) : Nat :=
  let blue_left := initial_blue - removed_blue
  let black_left := initial_black - removed_black
  let red_left := initial_red
  blue_left + black_left + red_left

theorem pens_left_is_25 :
  total_pens_left 9 21 6 4 7 = 25 :=
by 
  rw [total_pens_left, show 9 - 4 = 5 from Nat.sub_eq_of_eq_add (rfl), show 21 - 7 = 14 from Nat.sub_eq_of_eq_add (rfl)]
  rfl

end pens_left_is_25_l582_582002


namespace first_terrific_tuesday_l582_582182

-- Definitions for days of the week
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq

-- Definition to calculate the day of the week for a given date
def day_of_week (start_day : Day) (start_date : Nat) (target_date : Nat) : Day :=
  Day.recOn start_day
    (λ n, Day) -- function to add the offset correctly (not detailed here)

-- Definitions for months and their lengths
def days_in_month : Nat → Nat
| 10 => 31  -- October
| 11 => 30  -- November
| 12 => 31  -- December
| _ => 0    -- Other months, not needed here

-- Definitions for school schedule and terrific Tuesday rule
def start_date : Nat := 3
def start_day : Day := Day.Tuesday
def is_terrific_tuesday (month : Nat) (date : Nat) : Bool :=
  let days := days_in_month month
  let tuesdays := List.filter (λ d, day_of_week start_day start_date d = Day.Tuesday) (List.range days)
  tuesdays.length = 5 && tuesdays.getLast? = some date

-- Formal statement of the problem
theorem first_terrific_tuesday :
  ∃ date, date = 31 ∧ is_terrific_tuesday 12 date :=
by
  sorry

end first_terrific_tuesday_l582_582182


namespace seq_a_expression_sum_a_products_l582_582455

noncomputable def seq_a (n : ℕ) : ℝ :=
if h : n = 0 then 1/3 else sorry -- this is a placeholder

theorem seq_a_expression (n : ℕ) (h : n > 0) :
  ∃ (a : ℕ → ℝ), 
    (∀ (n : ℕ), a n ≠ 0) ∧ 
    (a 1 = 1/3) ∧ 
    (∀ (n : ℕ), 2 ≤ n → a (n - 1) - a n = 2 * a (n - 1) * a n) → 
    (a n = 1/(2*n+1)) :=
begin
  sorry
end

theorem sum_a_products (n : ℕ) (h : n > 0) :
  ∃ (a : ℕ → ℝ), 
    (∀ (n : ℕ), a n ≠ 0) ∧ 
    (a 1 = 1/3) ∧ 
    (∀ (n : ℕ), 2 ≤ n → a (n - 1) - a n = 2 * a (n - 1) * a n) → 
    (finset.sum (finset.range n) (λ k, a (k+1) * a (k+2)) = n/(6*n+9)) :=
begin
  sorry
end

end seq_a_expression_sum_a_products_l582_582455


namespace sum_of_first_twelve_multiples_of_17_l582_582313

theorem sum_of_first_twelve_multiples_of_17 : 
  (∑ i in Finset.range 12, 17 * (i + 1)) = 1326 := 
by
  sorry

end sum_of_first_twelve_multiples_of_17_l582_582313


namespace two_digit_number_multiple_l582_582710

noncomputable def is_divisible_by (n : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, n = k * d

theorem two_digit_number_multiple :
  ∃ n, (10 ≤ n ∧ n < 100) ∧ is_divisible_by n 8 ∧ is_divisible_by n 12 ∧ is_divisible_by n 18 ∧ (60 ≤ n ∧ n ≤ 79) :=
begin
  sorry
end

end two_digit_number_multiple_l582_582710


namespace hexagon_area_ratio_l582_582009

theorem hexagon_area_ratio (r : ℝ) (A_small A_large : ℝ) 
  (h1 : A_small = (3 * (r^2) * real.sqrt 3) / 2)
  (h2 : A_large = 2 * (3 * (2 * r / real.sqrt 3)^2 * real.sqrt 3) / 2) :
  A_large / A_small = 4 / 3 := 
sorry

end hexagon_area_ratio_l582_582009


namespace symmetric_difference_card_l582_582884

open Finset

variable (x y : Finset ℤ)
variable (hx : x.card = 8)
variable (hy : y.card = 18)
variable (hxy : (x ∩ y).card = 6)

theorem symmetric_difference_card : (x \ y ∪ y \ x).card = 14 := by
  sorry

end symmetric_difference_card_l582_582884


namespace inverse_function_ratio_l582_582607

def g (x : ℝ) : ℝ := (3 * x - 2) / (x - 4)
noncomputable def g_inv (x : ℝ) : ℝ := (4 * x - 2) / (3 - x)
def a : ℝ := 4
def b : ℝ := -2
def c : ℝ := -1
def d : ℝ := 3

theorem inverse_function_ratio :
  ∀ x, g (g_inv x) = x ∧ (g_inv (g x) = x) → (a / c = -4) :=
by
  sorry

end inverse_function_ratio_l582_582607


namespace triangle_Y_distance_sum_l582_582190

noncomputable def distance_sum_Y (PQ QR PR : ℝ) (A B C Y : ℝ×ℝ) : ℝ :=
  YA + YB + YC -- Distance formula should be defined accordingly

theorem triangle_Y_distance_sum :
  ∀ (P Q R A B C Y : ℝ×ℝ),
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 10^2 ∧
    (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = 24^2 ∧
    (P.1 - R.1)^2 + (P.2 - R.2)^2 = 26^2 ∧
    A = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) ∧
    B = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) ∧
    C = ((P.1 + R.1) / 2, (P.2 + R.2) / 2) ∧
    -- Condition for point Y being intersection of the circumcircles of ∆QAB and ∆RBC
    -- Should be defined based on the specific geometric constraints.
    Y ≠ B ∧
    intersect_circumcircles (Q, A, B) (R, B, C) Y → -- Helper theorem for circumcircle intersection
    distance_sum_Y 10 24 26 A B C Y = 2160 / real.sqrt 2975 :=
sorry -- Proof omitted

end triangle_Y_distance_sum_l582_582190


namespace boys_and_girls_at_bus_stop_l582_582345

theorem boys_and_girls_at_bus_stop (H M : ℕ) 
  (h1 : H = 2 * (M - 15)) 
  (h2 : M - 15 = 5 * (H - 45)) : 
  H = 50 ∧ M = 40 := 
by 
  sorry

end boys_and_girls_at_bus_stop_l582_582345


namespace decreasing_linear_function_l582_582796

theorem decreasing_linear_function (k : ℝ) : 
  (∀ x1 x2 : ℝ, x1 < x2 → (k - 3) * x1 + 2 > (k - 3) * x2 + 2) → k < 3 := 
by 
  sorry

end decreasing_linear_function_l582_582796


namespace decagon_diagonals_from_vertex_l582_582431

/-- 
  A decagon has 10 vertices. From any given vertex, you can draw a line to 9 other vertices,
  but the lines connecting to the two adjacent vertices are sides, not diagonals.
  Therefore, the number of diagonals that can be drawn from one vertex is 7.
-/
theorem decagon_diagonals_from_vertex (n : ℕ) (h1 : n = 10) :
  let connections := n - 1 in
  let adjacents := 2 in
  connections - adjacents = 7 :=
by
  -- Using the given conditions and proper mathematical reasoning
  sorry

end decagon_diagonals_from_vertex_l582_582431


namespace distance_between_symmetry_axes_l582_582600

theorem distance_between_symmetry_axes :
  let f := λ x, sin ((2/3) * x) + cos ((2/3) * x) in
  ∃ T : ℝ, (∀ x, f (x + T) = f x) ∧ (∃ d : ℝ, d = T / 2 ∧ d = 3 * π / 2) :=
by
  sorry

end distance_between_symmetry_axes_l582_582600


namespace inscribed_circle_radius_l582_582976

-- Define the conditions of the problem
def sector_radius : ℝ := 5
def sector_angle : ℝ := π / 3 -- Since it's a third of a full circle, π/3 radians

-- Theorem statement: Prove the radius of the inscribed circle is as given
theorem inscribed_circle_radius :
  ∃ r : ℝ, r = (5 * real.sqrt 3 - 5) / 2 :=
begin
  use (5 * real.sqrt 3 - 5) / 2,
  sorry, -- Proof to be completed
end

end inscribed_circle_radius_l582_582976


namespace general_term_formula_initial_condition_sum_of_first_n_terms_l582_582819

-- Define the sequence {a_n}
def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2^(n-1)

-- Prove the general term formula
theorem general_term_formula :
  ∀ n : ℕ, n > 0 → sequence_a (n+1) = 1 + (∑ i in Finset.range (n+1), sequence_a i) :=
by sorry

theorem initial_condition :
  sequence_a 2 = 2 * sequence_a 1 :=
by sorry

-- Define the sequence {b_n}
def sequence_b (n : ℕ) : ℤ :=
  let a := sequence_a n in
  a * Int.log2 a + (-1)^(n : ℤ) * n

-- Define the sum of the first n terms for the sequence {b_n}
def sum_b (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    (n - 2) * 2^n + (n + 4) / 2
  else
    (n - 2) * 2^n - (n - 3) / 2

-- Prove the sum of the first n terms
theorem sum_of_first_n_terms :
  ∀ n : ℕ, (∑ i in Finset.range (n+1), sequence_b i) = sum_b n :=
by sorry

end general_term_formula_initial_condition_sum_of_first_n_terms_l582_582819


namespace no_fractional_xy_l582_582763

theorem no_fractional_xy (x y : ℚ) (m n : ℤ) (h1 : 13 * x + 4 * y = m) (h2 : 10 * x + 3 * y = n) : ¬ (¬(x ∈ ℤ) ∨ ¬(y ∈ ℤ)) :=
sorry

end no_fractional_xy_l582_582763


namespace Hulk_jump_more_than_500_l582_582983

theorem Hulk_jump_more_than_500 :
  ∀ n : ℕ, 2 * 3^(n - 1) > 500 → n = 7 :=
by
  sorry

end Hulk_jump_more_than_500_l582_582983


namespace sophia_fraction_of_pie_l582_582585

theorem sophia_fraction_of_pie
  (weight_fridge : ℕ) (weight_eaten : ℕ)
  (h1 : weight_fridge = 1200)
  (h2 : weight_eaten = 240) :
  (weight_eaten : ℚ) / ((weight_fridge + weight_eaten : ℚ)) = (1 / 6) :=
by
  sorry

end sophia_fraction_of_pie_l582_582585


namespace vasya_wins_l582_582291

noncomputable def combine_and_divide (piles : List ℕ) : List ℕ :=
  sorry -- Placeholder for game mechanics implementation

def initial_piles := [40, 40, 40]

-- max_moves calculates the total number of moves required to change from 3 to 119 piles
def max_moves := (119 - 3) / 2

theorem vasya_wins (initial_piles : List ℕ) (max_moves : ℕ) : ∃ player, player = "Vasya" := by
  have h_initial := initial_piles.length = 3
  have h_moves := max_moves = 58
  have h_odd_piles : ∀ k, k.even → initial_piles.length + 2 * k ≡ 1 [MOD 2] := sorry
  have h_even_moves: max_moves % 2 = 0 := by norm_num
  have h_final_move : "Vasya" := by sorry
  show ∃ player, player = "Vasya" from
    exists.intro "Vasya" h_final_move

end vasya_wins_l582_582291


namespace chord_length_of_intersecting_line_and_circle_l582_582896

theorem chord_length_of_intersecting_line_and_circle :
  ∀ (x y : ℝ), (3 * x + 4 * y - 5 = 0) ∧ (x^2 + y^2 = 4) →
  ∃ (AB : ℝ), AB = 2 * Real.sqrt 3 := 
sorry

end chord_length_of_intersecting_line_and_circle_l582_582896


namespace janet_owes_wages_and_taxes_l582_582546

theorem janet_owes_wages_and_taxes :
  (∀ (workdays : ℕ) (hours : ℕ) (warehouse_workers : ℕ) (manager_workers : ℕ) (warehouse_wage : ℕ) (manager_wage : ℕ) (tax_rate : ℚ),
    workdays = 25 →
    hours = 8 →
    warehouse_workers = 4 →
    manager_workers = 2 →
    warehouse_wage = 15 →
    manager_wage = 20 →
    tax_rate = 0.1 →
    let total_hours := workdays * hours
        warehouse_monthly := total_hours * warehouse_wage
        manager_monthly := total_hours * manager_wage
        total_wage := warehouse_monthly * warehouse_workers + manager_monthly * manager_workers
        total_taxes := total_wage * tax_rate in
    total_wage + total_taxes = 22000) :=
begin
  intros,
  rw [← mul_assoc, mul_comm 25 8, mul_assoc],
  have h1 : 25 * 8 = 200, {norm_num},
  rw h1,
  have h2 : 200 * 15 * 4 = 12000, {norm_num},
  have h3 : 200 * 20 * 2 = 8000, {norm_num},
  rw [h2, h3],
  have h4 : 12000 + 8000 = 20000, {norm_num},
  have h5 : 20000 * 0.1 = 2000, {norm_num},
  rw [h4, h5],
  norm_num,
end

end janet_owes_wages_and_taxes_l582_582546


namespace meghan_total_money_l582_582947

theorem meghan_total_money :
  let num_100_bills := 2
  let num_50_bills := 5
  let num_10_bills := 10
  let value_100_bills := num_100_bills * 100
  let value_50_bills := num_50_bills * 50
  let value_10_bills := num_10_bills * 10
  let total_money := value_100_bills + value_50_bills + value_10_bills
  total_money = 550 := by sorry

end meghan_total_money_l582_582947


namespace polynomial_square_b_value_l582_582273

theorem polynomial_square_b_value
  (a b : ℚ)
  (h : ∃ (p q r : ℚ), (x^4 - x^3 + x^2 + a * x + b) = (p * x^2 + q * x + r)^2) :
  b = 9 / 64 :=
sorry

end polynomial_square_b_value_l582_582273


namespace num_sets_C_l582_582114

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}

def B : Set ℕ := {x | 0 < x ∧ x < 5}

theorem num_sets_C (A_subset_C_subset_B : ∀ C : Set ℕ, A ⊆ C → C ⊆ B):
  {C : Set ℕ | A ⊆ C ∧ C ⊆ B}.finite.card = 4 :=
sorry

end num_sets_C_l582_582114


namespace yeast_cells_at_10_18_l582_582955

noncomputable def yeastPopulation (initial_population : ℕ) (a : ℕ) (b : ℕ) : ℕ :=
  initial_population * a^b

theorem yeast_cells_at_10_18 (initial_population : ℕ) (triple_interval : ℕ) (time_intervals : ℕ) :
  initial_population = 50 → triple_interval = 3 → time_intervals = 3 → 
  yeastPopulation initial_population triple_interval time_intervals = 1350 :=
begin
  intros h_initial h_triple h_intervals,
  rw [h_initial, h_triple, h_intervals],
  exact eq.refl 1350,
end

end yeast_cells_at_10_18_l582_582955


namespace fib_math_competition_l582_582422

theorem fib_math_competition :
  ∃ (n9 n8 n7 : ℕ), 
    n9 * 4 = n8 * 7 ∧ 
    n9 * 3 = n7 * 10 ∧ 
    n9 + n8 + n7 = 131 :=
sorry

end fib_math_competition_l582_582422


namespace correct_option_l582_582667

theorem correct_option : ∀ (x y : ℝ), 10 * x * y - 10 * y * x = 0 :=
by 
  intros x y
  sorry

end correct_option_l582_582667


namespace largest_number_with_cube_product_l582_582790

theorem largest_number_with_cube_product : ∃ (n : ℕ), 
  (∃ digits : List ℕ, 
    (∀ d ∈ digits, d < 10) ∧ -- Digits are between 0 and 9
    digits.nodup ∧          -- Digits are all different
    digits.product = (some m, m^3)) ∧ -- Product of digits is a cube
  n = 984321 := 
sorry

end largest_number_with_cube_product_l582_582790


namespace janet_speed_l582_582200

def janet_sister_speed : ℝ := 12
def lake_width : ℝ := 60
def wait_time : ℝ := 3

theorem janet_speed :
  (lake_width / (lake_width / janet_sister_speed - wait_time)) = 30 := 
sorry

end janet_speed_l582_582200


namespace distance_sum_interval_l582_582518

-- Define the points as pairs of integers
def A : ℝ × ℝ := (15, 3)
def B : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (6, 8)

-- Calculate the Euclidean distance function
def dist (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the distances AD and BD
def AD := dist A D
def BD := dist B D

-- Prove that the sum of the distances lies between 20 and 21
theorem distance_sum_interval : 20 < AD + BD ∧ AD + BD < 21 := by
  sorry

end distance_sum_interval_l582_582518


namespace high_school_total_students_l582_582352

theorem high_school_total_students (N_seniors N_sample N_freshmen_sample N_sophomores_sample N_total : ℕ)
  (h_seniors : N_seniors = 1000)
  (h_sample : N_sample = 185)
  (h_freshmen_sample : N_freshmen_sample = 75)
  (h_sophomores_sample : N_sophomores_sample = 60)
  (h_proportion : N_seniors * (N_sample - (N_freshmen_sample + N_sophomores_sample)) = N_total * (N_sample - N_freshmen_sample - N_sophomores_sample)) :
  N_total = 3700 :=
by
  sorry

end high_school_total_students_l582_582352


namespace opposite_sides_parallel_l582_582285

-- Definition of the vertices of the hexagon and angles summing conditions
def hexagon (A B C D E F : Type) : Prop :=
  (convex_hexagon A B C D E F) ∧
  (equal_sides A B C D E F) ∧
  (angle_sum A C E = 360) ∧
  (angle_sum B D F = 360)

-- The theorem that needs to be proved
theorem opposite_sides_parallel (A B C D E F : Type) (h : hexagon A B C D E F) : 
  parallel_sides A B C D E F :=
  sorry

end opposite_sides_parallel_l582_582285


namespace range_of_b_fixed_points_of_C_l582_582901

-- Condition on the quadratic function and the requirement for three intersections
def quadratic_function_condition (f : ℝ → ℝ) (b : ℝ) : Prop :=
  f = λ x, x^2 + 2*x + b ∧ (f 0 = b ∧ discriminant (λ x, x^2 + 2*x + b) > 0)

-- Defining the discriminant for a quadratic function of the form ax^2 + bx + c
def discriminant (f : ℝ → ℝ) : ℝ :=
  match coeffs_of_quadratic f with
  | ⟨a, b, c⟩ => b^2 - 4 * a * c
  | _          => 0
  end

-- Theorem 1: The range of b such that the quadratic function intersects the x-axis and y-axis at distinct points
theorem range_of_b (b : ℝ) : quadratic_function_condition (λ x, x^2 + 2*x + b) b → b < 1 ∧ b ≠ 0 :=
sorry

-- Theorem 2: The circle passes through fixed points (0, 1) and (-2, 1) for the given b range
def circle_passes_fixed_points (b : ℝ) : Prop :=
  ∀ (C : ℝ → ℝ → Prop), (C 0 1) ∧ (C (-2) 1)

theorem fixed_points_of_C (b : ℝ) : quadratic_function_condition (λ x, x^2 + 2*x + b) b → circle_passes_fixed_points b :=
sorry

end range_of_b_fixed_points_of_C_l582_582901


namespace right_triangle_num_array_l582_582721

theorem right_triangle_num_array (n : ℕ) (hn : 0 < n) 
    (a : ℕ → ℕ → ℝ) 
    (h1 : a 1 1 = 1/4)
    (hd : ∀ i j, 0 < j → j <= i → a (i+1) 1 = a i 1 + 1/4)
    (hq : ∀ i j, 2 < i → 0 < j → j ≤ i → a i (j+1) = a i j * (1/2)) :
  a n 3 = n / 16 := 
by 
  sorry

end right_triangle_num_array_l582_582721


namespace concert_total_audience_l582_582716

-- Definitions based on conditions
def audience_for_second_band (total_audience : ℕ) : ℕ := (2 * total_audience) / 3
def audience_for_first_band (total_audience : ℕ) : ℕ := total_audience / 3

def under_30_audience_for_second_band (total_audience : ℕ) : ℕ := audience_for_second_band(total_audience) / 2

def men_under_30_for_second_band : ℕ := 20
def percentage_men_under_30_for_second_band : ℕ := 40 -- 40%

def total_under_30_for_second_band : ℕ := men_under_30_for_second_band * 100 / percentage_men_under_30_for_second_band

-- Main theorem statement
theorem concert_total_audience (total_audience : ℕ) :
  audience_for_second_band(total_audience) = 100 →
  (audience_for_second_band(total_audience) * 50) / 100 = total_under_30_for_second_band →
  total_audience = 150 :=
by
  intros h1 h2
  sorry

end concert_total_audience_l582_582716


namespace number_to_replace_x_l582_582611

theorem number_to_replace_x (x : ℕ) :
  (1 + 2 + 3 + 4 + 5 + 6 = 21)
  ∧ (∀ n ∈ {1, 2, 3, 4, 5, 6}, n presents exactly twice in the sum of circles)
  ∧ (sum of each circle = 14)
  ∧ (6 is placed already) 
  → x = 1 := by
  sorry

end number_to_replace_x_l582_582611


namespace final_position_correct_l582_582267

-- Define the initial position of the letter G
def initial_position : (string × (bool × bool)) := ("G", (true, true))

-- Define transformation operations
def rotate180 (pos : (string × (bool × bool))) : (string × (bool × bool)) :=
  ("G", (not pos.2.1, not pos.2.2))

def reflect_x (pos : (string × (bool × bool))) : (string × (bool × bool)) :=
  ("G", (pos.2.1, not pos.2.2))

def rotate270_counterclockwise (pos : (string × (bool × bool))) : (string × (bool × bool)) :=
  ("G", (pos.2.2, not pos.2.1))

-- Apply transformations sequentially
def final_position : (string × (bool × bool)) :=
  let step1 := rotate180 initial_position
  let step2 := reflect_x step1
  let step3 := rotate270_counterclockwise step2
  step3

-- Statement to prove
theorem final_position_correct : final_position = ("G", (true, false)) :=
  by
  sorry

end final_position_correct_l582_582267


namespace F_property_l582_582507

noncomputable def F (x : ℝ) : ℝ := sorry -- Volume function F(x)

theorem F_property :
  (∃ x_max : ℝ, ∀ x : ℝ, F(x) ≤ F(x_max)) ∧ ¬(∀ x1 x2 : ℝ, x1 < x2 → F(x1) < F(x2)) :=
by
  -- Definitions based on given conditions
  -- Prove the main theorem
  sorry

end F_property_l582_582507


namespace sum_first_twelve_multiples_of_17_l582_582312

theorem sum_first_twelve_multiples_of_17 : ∑ k in finset.range (12 + 1), 17 * k = 1326 :=
by
  -- proof steps would go here, but for the task, we add 'sorry' to skip the proof
  sorry

end sum_first_twelve_multiples_of_17_l582_582312


namespace problem_part1_problem_part2_l582_582483

def f (x a : ℝ) := x^4 + x^2 + (a - 1) * x + 1

theorem problem_part1 (a : ℝ) (h1 : a = 1) : 
  (∀ x : ℝ, x > 0 → deriv (λ x, f x a) x > 0) ∧
  (∀ x : ℝ, x < 0 → deriv (λ x, f x a) x < 0) :=
sorry

theorem problem_part2 (a : ℝ) (h : ∀ x : ℝ, x > 0 → f x a ≤ x^4 + exp x) : 
  a ≤ real.exp 1 - 1 :=
sorry

end problem_part1_problem_part2_l582_582483


namespace probability_first_spade_second_king_l582_582638

/--
In a standard deck of 52 cards, the probability of drawing the first card as a ♠ and the second card as a king is 1/52.
-/
theorem probability_first_spade_second_king : 
  let deck_size := 52 in
  let hearts_count := 13 in
  let kings_count := 4 in
  let prob := (1 / deck_size : ℚ) * (kings_count / (deck_size - 1)) + ((hearts_count - 1) / deck_size) * (kings_count / (deck_size - 1)) 
  in 
  prob = 1 / deck_size :=
by
  sorry

end probability_first_spade_second_king_l582_582638


namespace semi_ellipse_perimeter_approx_9_49_l582_582834

noncomputable def semiEllipsePerimeterApprox (a b : ℝ) : ℝ :=
  let semi_major_axis := a / 2
  let semi_minor_axis := b / 2
  let C := Real.pi * (3 * (semi_major_axis + semi_minor_axis) - Real.sqrt ((3 * semi_major_axis + semi_minor_axis) * (semi_major_axis + 3 * semi_minor_axis)))
  C / 2

theorem semi_ellipse_perimeter_approx_9_49 :
  semiEllipsePerimeterApprox 7 5 ≈ 9.49 := 
by
  -- This is to indicate that π ≈ 3.14159 when necessary.
  norm_num [Real.pi]
  sorry

end semi_ellipse_perimeter_approx_9_49_l582_582834


namespace value_of_M_l582_582735

-- Define M as given in the conditions
def M : ℤ :=
  (150^2 + 2) + (149^2 - 2) - (148^2 + 2) - (147^2 - 2) + (146^2 + 2) +
  (145^2 - 2) - (144^2 + 2) - (143^2 - 2) + (142^2 + 2) + (141^2 - 2) -
  (140^2 + 2) - (139^2 - 2) + (138^2 + 2) + (137^2 - 2) - (136^2 + 2) -
  (135^2 - 2) + (134^2 + 2) + (133^2 - 2) - (132^2 + 2) - (131^2 - 2) +
  (130^2 + 2) + (129^2 - 2) - (128^2 + 2) - (127^2 - 2) + (126^2 + 2) +
  (125^2 - 2) - (124^2 + 2) - (123^2 - 2) + (122^2 + 2) + (121^2 - 2) -
  (120^2 + 2) - (119^2 - 2) + (118^2 + 2) + (117^2 - 2) - (116^2 + 2) -
  (115^2 - 2) + (114^2 + 2) + (113^2 - 2) - (112^2 + 2) - (111^2 - 2) +
  (110^2 + 2) + (109^2 - 2) - (108^2 + 2) - (107^2 - 2) + (106^2 + 2) +
  (105^2 - 2) - (104^2 + 2) - (103^2 - 2) + (102^2 + 2) + (101^2 - 2) -
  (100^2 + 2) - (99^2 - 2) + (98^2 + 2) + (97^2 - 2) - (96^2 + 2) -
  (95^2 - 2) + (94^2 + 2) + (93^2 - 2) - (92^2 + 2) - (91^2 - 2) +
  (90^2 + 2) + (89^2 - 2) - (88^2 + 2) - (87^2 - 2) + (86^2 + 2) +
  (85^2 - 2) - (84^2 + 2) - (83^2 - 2) + (82^2 + 2) + (81^2 - 2) -
  (80^2 + 2) - (79^2 - 2) + (78^2 + 2) + (77^2 - 2) - (76^2 + 2) -
  (75^2 - 2) + (74^2 + 2) + (73^2 - 2) - (72^2 + 2) - (71^2 - 2) +
  (70^2 + 2) + (69^2 - 2) - (68^2 + 2) - (67^2 - 2) + (66^2 + 2) +
  (65^2 - 2) - (64^2 + 2) - (63^2 - 2) + (62^2 + 2) + (61^2 - 2) -
  (60^2 + 2) - (59^2 - 2) + (58^2 + 2) + (57^2 - 2) - (56^2 + 2) -
  (55^2 - 2) + (54^2 + 2) + (53^2 - 2) - (52^2 + 2) - (51^2 - 2) +
  (50^2 + 2) + (49^2 - 2) - (48^2 + 2) - (47^2 - 2) + (46^2 + 2) +
  (45^2 - 2) - (44^2 + 2) - (43^2 - 2) + (42^2 + 2) + (41^2 - 2) -
  (40^2 + 2) - (39^2 - 2) + (38^2 + 2) + (37^2 - 2) - (36^2 + 2) -
  (35^2 - 2) + (34^2 + 2) + (33^2 - 2) - (32^2 + 2) - (31^2 - 2) +
  (30^2 + 2) + (29^2 - 2) - (28^2 + 2) - (27^2 - 2) + (26^2 + 2) +
  (25^2 - 2) - (24^2 + 2) - (23^2 - 2) + (22^2 + 2) + (21^2 - 2) -
  (20^2 + 2) - (19^2 - 2) + (18^2 + 2) + (17^2 - 2) - (16^2 + 2) -
  (15^2 - 2) + (14^2 + 2) + (13^2 - 2) - (12^2 + 2) - (11^2 - 2) +
  (10^2 + 2) + (9^2 - 2) - (8^2 + 2) - (7^2 - 2) + (6^2 + 2) +
  (5^2 - 2) - (4^2 + 2) - (3^2 - 2) + (2^2 + 2) + (1^2 - 2)

-- Statement to prove that the value of M is 22700
theorem value_of_M : M = 22700 :=
  by sorry

end value_of_M_l582_582735


namespace proof_complex_ratio_l582_582925

noncomputable def condition1 (x y : ℂ) (k : ℝ) : Prop :=
  (x + k * y) / (x - k * y) + (x - k * y) / (x + k * y) = 1

theorem proof_complex_ratio (x y : ℂ) (k : ℝ) (h : condition1 x y k) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = (41 / 20 : ℂ) :=
by 
  sorry

end proof_complex_ratio_l582_582925


namespace find_angle_B_l582_582124

theorem find_angle_B (A B C a b c : ℝ)
  (h_sides : ∀ (triangle : Triangle), triangle.oppositeSidesEqual (a, b, c) = (A, B, C))
  (h_eq : sqrt 3 * b * sin A - a * cos B - 2 * a = 0) :
  B = 2 * π / 3 :=
sorry

end find_angle_B_l582_582124


namespace speeds_are_correct_l582_582998

variables (b c : ℝ)
hypothesis hb : b > 0
hypothesis hc : c > 0

noncomputable def speeds_of_riders : ℝ × ℝ :=
  let Δ := c ^ 2 + 120 * b * c in
  let x := (c + real.sqrt Δ) / 2 in
  let y := (-c + real.sqrt Δ) / 2 in
  (x, y)

theorem speeds_are_correct :
  let (x, y) := speeds_of_riders b c in
  (∃ x y : ℝ, 
    (b / y - b / x = 1 / 30) ∧ 
    (b / (x - c) - b / (y + c) = 1 / 30) ∧ 
    x = (c + real.sqrt (c^2 + 120 * b * c)) / 2 ∧
    y = (-c + real.sqrt (c^2 + 120 * b * c)) / 2
  ) :=
by
  sorry

end speeds_are_correct_l582_582998


namespace arithmetic_sequence_cans_l582_582706

noncomputable def total_cans : ℕ := by
  let a := 35   -- First term
  let d := -4    -- Common difference
  let l := 3    -- Last term
  -- Calculating the number of terms
  let n := (l - a) / d + 1
  -- Sum of the sequence
  let S_n := n * (a + l) / 2
  exact S_n

theorem arithmetic_sequence_cans (a d l n : ℤ) (h1 : a = 35) (h2 : d = -4) (h3 : l = 3) (h4 : n = (l - a) / d + 1) :
  let S_n := n * (a + l) / 2 in S_n = 171 := by
  rw [h1, h2, h3, h4]
  exact sorry

end arithmetic_sequence_cans_l582_582706


namespace max_lines_dividing_n_gon_l582_582105

theorem max_lines_dividing_n_gon (n : ℕ) (P : convex_polygon ℝ) (H_non_parallel : ∀ (i j: fin n), i ≠ j → ¬parallel (side P i) (side P j)) (H_convex : convex_polygon.is_convex P) (O : point ℝ) (H_inside : O ∈ P) : 
  ∀ (k : ℕ), (∀ (i : fin k), divides_polygon_in_half_through (line_through_point P O i) P) → k ≤ n :=
sorry

end max_lines_dividing_n_gon_l582_582105


namespace calculate_total_money_l582_582951

theorem calculate_total_money (n100 n50 n10 : ℕ) 
  (h1 : n100 = 2) (h2 : n50 = 5) (h3 : n10 = 10) : 
  (n100 * 100 + n50 * 50 + n10 * 10 = 550) :=
by
  sorry

end calculate_total_money_l582_582951


namespace intersection_complement_l582_582149

open Set

def P := {x : ℝ | 0 ≤ x ∧ x ≤ 3}
def Q := {x : ℝ | x ≤ -2 ∨ x ≥ 2}
def negQ := {x : ℝ | -2 < x ∧ x < 2}

theorem intersection_complement :
  P ∩ negQ = {x : ℝ | 0 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_complement_l582_582149


namespace additional_cost_is_three_dollars_l582_582959

-- Definition of the cost of peanut butter per jar
def peanut_butter_cost : ℝ := 3

-- Given that the almond butter costs three times as much as the peanut butter
def almond_butter_multiplier : ℝ := 3

-- Definition of the cost of almond butter per jar
def almond_butter_cost : ℝ := almond_butter_multiplier * peanut_butter_cost

-- Given that it takes half a jar to make a batch of cookies
def jar_fraction_per_batch : ℝ := 1 / 2

-- Calculate the cost per batch for peanut butter and almond butter
def peanut_butter_cost_per_batch : ℝ := jar_fraction_per_batch * peanut_butter_cost
def almond_butter_cost_per_batch : ℝ := jar_fraction_per_batch * almond_butter_cost

-- The additional cost per batch of almond butter cookies compared to peanut butter cookies
def additional_cost_per_batch : ℝ := almond_butter_cost_per_batch - peanut_butter_cost_per_batch

-- The theorem stating the additional cost per batch is 3 dollars
theorem additional_cost_is_three_dollars : additional_cost_per_batch = 3 := by
  sorry

end additional_cost_is_three_dollars_l582_582959


namespace sparkling_water_annual_cost_l582_582945

theorem sparkling_water_annual_cost:
  (sparkling fraction_per_night: ℝ) (cost_per_bottle: ℝ) (days_per_year: ℕ):
  fraction_per_night = 1/5 →
  cost_per_bottle = 2 →
  days_per_year = 365 →
  cost_per_bottle * (days_per_year / (1 / fraction_per_night)) = 146 :=
by 
  intros fraction_per_night cost_per_bottle days_per_year h_fraction hn_cost hb_cost
  sorry

end sparkling_water_annual_cost_l582_582945


namespace find_a_l582_582824

variables (a b c : ℝ) (A B C : ℝ) (sin : ℝ → ℝ)
variables (sqrt_three_two sqrt_two_two : ℝ)

-- Assume that A = 60 degrees, B = 45 degrees, and b = sqrt(6)
def angle_A : A = π / 3 := by
  sorry

def angle_B : B = π / 4 := by
  sorry

def side_b : b = Real.sqrt 6 := by
  sorry

def sin_60 : sin (π / 3) = sqrt_three_two := by
  sorry

def sin_45 : sin (π / 4) = sqrt_two_two := by
  sorry

-- Prove that a = 3 based on the given conditions
theorem find_a (sin_rule : a / sin A = b / sin B)
  (sin_60_def : sqrt_three_two = Real.sqrt 3 / 2)
  (sin_45_def : sqrt_two_two = Real.sqrt 2 / 2) : a = 3 := by
  sorry

end find_a_l582_582824


namespace max_angle_C_l582_582513

-- Define the necessary context and conditions
variable {a b c : ℝ}

-- Condition that a^2 + b^2 = 2c^2 in a triangle
axiom triangle_condition : a^2 + b^2 = 2 * c^2

-- Theorem statement
theorem max_angle_C (h : a^2 + b^2 = 2 * c^2) : ∃ C : ℝ, C = Real.pi / 3 := sorry

end max_angle_C_l582_582513


namespace _l582_582580

open Nat

def number_of_ways_to_distribute_balls (balls boxes : Nat) (min_balls_per_box : Fin boxes → Nat) : Nat :=
  let remaining_balls := balls - (∑ j in Finset.range boxes, min_balls_per_box ⟨j, by linarith⟩)
  -- Using the stars and bars theorem to distribute remaining balls
  Nat.choose (remaining_balls + boxes - 1) (boxes - 1)

def min_balls : Fin 3 → Nat
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 2
| ⟨2, _⟩ => 3

example : number_of_ways_to_distribute_balls 10 3 min_balls = 15 :=
by
  simp [number_of_ways_to_distribute_balls, min_balls]
  norm_num
  exact Nat.choose_succ_succ 5 1

end _l582_582580


namespace tan_a6_of_arithmetic_sequence_l582_582501

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem tan_a6_of_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (H1 : arithmetic_sequence a)
  (H2 : sum_of_first_n_terms a S)
  (H3 : S 11 = 22 * Real.pi / 3) : 
  Real.tan (a 6) = -Real.sqrt 3 :=
sorry

end tan_a6_of_arithmetic_sequence_l582_582501


namespace find_k_values_l582_582207

noncomputable def possible_values_of_k (a b c : ℝ → ℝ) : set ℝ :=
  {k | k = (2 * real.sqrt 3) / 3 ∨ k = -(2 * real.sqrt 3) / 3}

theorem find_k_values (a b c : vector ℝ ℝ) (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) 
 (h3 : ∥c∥ = 1) (h4 : a ⬝ b = 0) (h5 : a ⬝ c = 0)
 (h6 : real.angle b c = real.pi / 3) : 
 ∃ k, a = k • (b × c) ∧ k ∈ possible_values_of_k a b c :=
 sorry

end find_k_values_l582_582207


namespace base_conversion_b_l582_582596

-- Define the problem in Lean
theorem base_conversion_b (b : ℕ) : 
  (b^2 + 2 * b - 16 = 0) → b = 4 := 
by
  intro h
  sorry

end base_conversion_b_l582_582596


namespace find_constants_PQR_l582_582787

theorem find_constants_PQR :
  ∃ P Q R : ℚ, 
    (P = (-8 / 15)) ∧ 
    (Q = (-7 / 6)) ∧ 
    (R = (27 / 10)) ∧
    (∀ x : ℚ, 
      (x - 1) ≠ 0 ∧ (x - 4) ≠ 0 ∧ (x - 6) ≠ 0 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6)) :=
by
  sorry

end find_constants_PQR_l582_582787


namespace tan_double_angle_l582_582101

theorem tan_double_angle (x : ℝ) (hx1 : cos x = 3/5) (hx2 : -π/2 < x ∧ x < 0) :
  tan (2 * x) = 24 / 7 := 
by
  sorry

end tan_double_angle_l582_582101


namespace max_radius_of_six_circles_on_unit_sphere_l582_582304

theorem max_radius_of_six_circles_on_unit_sphere (r : ℝ) :
  (∀ (C : fin 6 → ℝ × ℝ × ℝ), 
    (∀ i j, i ≠ j → dist (C i) (C j) > 2 * r) ∧
    (∀ i, dist (C i) (0, 0, 0) = 1)) → 
  r ≤ (Real.sqrt 2 / 2) :=
by
  sorry

end max_radius_of_six_circles_on_unit_sphere_l582_582304


namespace two_digit_number_divisible_8_12_18_in_range_l582_582707

def is_divisible_by (n d : ℕ) : Prop := d ∣ n

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem two_digit_number_divisible_8_12_18_in_range :
  ∃ (n : ℕ), is_two_digit n ∧ is_divisible_by n 8 ∧ is_divisible_by n 12 ∧ is_divisible_by n 18 ∧ (n ≥ 60 ∧ n < 80) :=
begin
  sorry
end

end two_digit_number_divisible_8_12_18_in_range_l582_582707


namespace infinite_primes_primitive_root_gt_l582_582917

theorem infinite_primes_primitive_root_gt (n : ℕ) (hn : n > 0) :
  ∃ᶠ p in (nat.filter prime), ∀ g, is_primitive_root g p → g > n :=
by
  sorry -- Proof omitted, as per instruction.

end infinite_primes_primitive_root_gt_l582_582917


namespace ratio_mark_to_rob_sam_l582_582175

noncomputable def sam_hunts (s: ℕ) := s = 6
noncomputable def rob_hunts (r: ℕ) := r = s / 2
noncomputable def rob_sam_total (rs: ℕ) := rs = s + r
noncomputable def mark_hunts (m: ℕ) (x: ℝ) := m = x * rs
noncomputable def peter_hunts (p: ℕ) := p = 3 * m
noncomputable def total_hunts (t: ℕ) := t = s + r + m + p
theorem ratio_mark_to_rob_sam (s r rs m p t: ℕ) (x : ℝ) 
  (h_sam_hunts: sam_hunts s) (h_rob_hunts: rob_hunts r)
  (h_rob_sam_total: rob_sam_total rs) (h_mark_hunts: mark_hunts m x)
  (h_peter_hunts: peter_hunts p) (h_total_hunts: total_hunts t)
  (h_total_eq_21: t = 21) (h_x_val: x = 1/3) :
  m / rs = 1 / 3 := by
  sorry

end ratio_mark_to_rob_sam_l582_582175


namespace math_problem_l582_582831

theorem math_problem 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f(x) = f(6 - x) + f(3))
  (h2 : ∀ x, f(x + π) = -f(-x - π))  -- implies symmetry about (-π,0)
  (h3 : f(1) = 2022) :
  (∀ x, f(x + 12) = f(x)) ∧    -- The period of f(x) is 12
  (f(2023) = -2022) ∧          -- f(2023) = -2022
  (∀ x, f(0.5 * x - 1) + π = f(0.5 * (2 - x) + 1) + π) := -- Symmetry about (2, π)
sorry

end math_problem_l582_582831


namespace infinite_sum_equals_l582_582926

noncomputable def infinite_sum (x : ℝ) (h : 1 < x) : ℝ :=
  ∑' n, 1 / (x^(3^n) - x^(-3^n))

theorem infinite_sum_equals (x : ℝ) (h : 1 < x) :
  infinite_sum x h = 1 / (x - 1) := by
  sorry

end infinite_sum_equals_l582_582926


namespace son_age_l582_582865

theorem son_age (A S : ℕ) (h1 : A = 45) (h2 : A = 3 * S) (h3 : A + 5 = 2.5 * (S + 5)) : S = 15 :=
by
  sorry

end son_age_l582_582865


namespace constructible_triangle_condition_l582_582805

theorem constructible_triangle_condition
  (A : Point)
  (e f : Ray)
  (P : Point)
  (s : ℝ)
  (convex_angle : ConvexAngle A e f)
  (outside_angle : ¬(is_within_or_on_boundary P convex_angle))
  : exists (B C : Point), 
    intersect_ray (line_through P) e = some B ∧
    intersect_ray (line_through P) f = some C ∧
    triangle_perimeter A B C = 2 * s :=
sorry

end constructible_triangle_condition_l582_582805


namespace average_weight_increase_l582_582986

theorem average_weight_increase :
  (∀ (A : ℝ) (number_of_persons : ℝ),
      number_of_persons = 8 →
      (old_person_weight new_person_weight : ℝ),
      old_person_weight = 65 →
      new_person_weight = 98.6 →
      ((new_person_weight - old_person_weight) / number_of_persons) = 4.2) :=
begin
  -- Proof omitted
  sorry
end

end average_weight_increase_l582_582986


namespace rubles_distribution_l582_582330

theorem rubles_distribution : 
  ∃ (wallet : ℕ → ℕ), 
  (∀ i, i < 7 → wallet i = 2^i) ∧ 
  (∑ i in finset.range 7, wallet i = 127) ∧ 
  (∀ (x : ℕ), 1 ≤ x ∧ x ≤ 127 → ∃ (s : finset ℕ), s ⊆ finset.range 7 ∧ x = (∑ i in s, wallet i)) := 
begin
  sorry
end

end rubles_distribution_l582_582330


namespace water_needed_in_pints_l582_582868

-- Define the input data
def parts_water : ℕ := 5
def parts_lemon : ℕ := 2
def pints_per_gallon : ℕ := 8
def total_gallons : ℕ := 3

-- Define the total parts of the mixture
def total_parts : ℕ := parts_water + parts_lemon

-- Define the total pints of lemonade
def total_pints : ℕ := total_gallons * pints_per_gallon

-- Define the pints per part of the mixture
def pints_per_part : ℚ := total_pints / total_parts

-- Define the total pints of water needed
def pints_water : ℚ := parts_water * pints_per_part

-- The theorem stating what we need to prove
theorem water_needed_in_pints : pints_water = 17 + 1 / 7 := by
  sorry

end water_needed_in_pints_l582_582868


namespace number_of_red_integers_le_phi_l582_582485

theorem number_of_red_integers_le_phi (n : ℕ) (X : finset ℕ)
  (hX : ∀ a b c ∈ X, (a * (b - c)) % n = 0 ↔ b = c) :
  X.card ≤ nat.totient n :=
sorry

end number_of_red_integers_le_phi_l582_582485


namespace sum_of_imaginary_parts_l582_582041

theorem sum_of_imaginary_parts (p r t : ℂ) (q s u : ℂ) (h1 : q = 4) (h2 : t = -p - r)
  (h3 : (p + q * complex.I) + (r + s * complex.I) + (t + u * complex.I) = 3 * complex.I) : s + u = -1 := by
  -- The proof is omitted
  sorry

end sum_of_imaginary_parts_l582_582041


namespace purple_valley_skirts_l582_582242

theorem purple_valley_skirts (azure_valley_skirts : ℕ) (h1 : azure_valley_skirts = 60) :
    let seafoam_valley_skirts := (2 / 3 : ℚ) * azure_valley_skirts in
    let purple_valley_skirts := (1 / 4 : ℚ) * seafoam_valley_skirts in
    purple_valley_skirts = 10 :=
by
  let seafoam_valley_skirts := (2 / 3 : ℚ) * azure_valley_skirts
  let purple_valley_skirts := (1 / 4 : ℚ) * seafoam_valley_skirts
  have h2 : seafoam_valley_skirts = (2 / 3 : ℚ) * 60 := by
    rw [h1]
  have h3 : purple_valley_skirts = (1 / 4 : ℚ) * ((2 / 3 : ℚ) * 60) := by
    rw [h2]
  have h4 : purple_valley_skirts = (1 / 4 : ℚ) * 40 := by
    norm_num [h3]
  have h5 : purple_valley_skirts = 10 := by
    norm_num [h4]
  exact h5

end purple_valley_skirts_l582_582242


namespace identify_organizers_l582_582568

section assassination_attempts

-- Assassination attempts dates
inductive AttemptDate
| Christmas
| NewYear
| July14

-- Terrorist organizations
inductive Organization
| Corsican
| Breton
| Basque

-- Statements provided
def statement1 (orgAt : AttemptDate → Organization) : Prop :=
  orgAt AttemptDate.Christmas = Organization.Basque ∧ orgAt AttemptDate.NewYear ≠ Organization.Basque

def statement2 (orgAt : AttemptDate → Organization) : Prop :=
  orgAt AttemptDate.July14 ≠ Organization.Breton

-- Assumption: Only one statement is true
axiom exactly_one_true (orgAt : AttemptDate → Organization) :
  (statement1 orgAt ∧ ¬statement2 orgAt) ∨
  (¬statement1 orgAt ∧ statement2 orgAt) :=
  sorry  -- This is an axiom necessary for the proof

-- Theorem to be proved: the assignments are as follows
theorem identify_organizers (orgAt : AttemptDate → Organization) (h : exactly_one_true orgAt) :
  orgAt AttemptDate.Christmas = Organization.Basque ∧
  orgAt AttemptDate.NewYear = Organization.Corsican ∧
  orgAt AttemptDate.July14 = Organization.Breton :=
begin
  sorry  -- Proof to be completed
end

end assassination_attempts

end identify_organizers_l582_582568


namespace complement_U_A_union_B_is_1_and_9_l582_582147

-- Define the universe set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define set A according to the given condition
def is_elem_of_A (x : ℕ) : Prop := 2 < x ∧ x ≤ 6
def A : Set ℕ := {x | is_elem_of_A x}

-- Define set B explicitly
def B : Set ℕ := {0, 2, 4, 5, 7, 8}

-- Define the union A ∪ B
def A_union_B : Set ℕ := A ∪ B

-- Define the complement of A ∪ B in U
def complement_U_A_union_B : Set ℕ := {x ∈ U | x ∉ A_union_B}

-- State the theorem
theorem complement_U_A_union_B_is_1_and_9 :
  complement_U_A_union_B = {1, 9} :=
by
  sorry

end complement_U_A_union_B_is_1_and_9_l582_582147


namespace point_X_divides_hypotenuse_BC_in_ratio_l582_582008

theorem point_X_divides_hypotenuse_BC_in_ratio
  (A B C E F X : Type)
  [decidable_eq X]
  {A B C E F: X}
  {length_AB length_AC length_AF length_BC : ℝ}
  (h1 : right_triangle ABC)
  (h2 : rectangle ABEF)
  (h3 : length_AC = 3 * length_AB)
  (h4 : length_AF = 2 * length_AB)
  (h5 : X = intersection_of_diagonal_AE_and_hypotenuse_BC ABEF ABC)
  (h6 : divides_hypotenuse_in_ratio X BC (2/3)) :
  | X := sorry

end point_X_divides_hypotenuse_BC_in_ratio_l582_582008


namespace eight_row_triangle_pieces_l582_582021

def unit_rods (n : ℕ) : ℕ := 3 * (n * (n + 1)) / 2

def connectors (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem eight_row_triangle_pieces : unit_rods 8 + connectors 9 = 153 :=
by
  sorry

end eight_row_triangle_pieces_l582_582021


namespace pure_imaginary_condition_l582_582676

def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = complex.I * b

theorem pure_imaginary_condition (a : ℝ) (b : ℝ) (h : a = 0) :
  (∃ b, b ≠ 0 ∧ (a + b * complex.I) = complex.I * b) ↔ (a = 0) :=
by
  sorry

end pure_imaginary_condition_l582_582676


namespace no_fractional_x_y_l582_582761

theorem no_fractional_x_y (x y : ℚ) (H1 : ¬ (x.denom = 1 ∧ y.denom = 1)) (H2 : ∃ m : ℤ, 13 * x + 4 * y = m) (H3 : ∃ n : ℤ, 10 * x + 3 * y = n) : false :=
sorry

end no_fractional_x_y_l582_582761


namespace sum_equality_proof_l582_582802

theorem sum_equality_proof (n : ℕ) (h : n > 0) : 
    (∑ i in finset.range n, (2 * i - 1) * (2 * i)^2 - 2 * i * (2 * i + 1)^2) = 
    -n * (n + 1) * (4 * n + 3) :=
sorry

end sum_equality_proof_l582_582802


namespace max_f_value_cos_x1_x2_value_l582_582139

noncomputable def f (x : ℝ) : ℝ := (Real.sin (π / 2 - x)) * (Real.sin x) - (Real.sqrt 3) * (Real.cos x) ^ 2 + (Real.sqrt 3) / 2

theorem max_f_value : 
  ∃ x : ℝ, (f x = 1) ∧ (∀ y : ℝ, f y ≤ 1) := 
begin
  sorry
end

theorem cos_x1_x2_value (x1 x2 : ℝ) (h1 : 0 < x1 ∧ x1 < π) (h2 : 0 < x2 ∧ x2 < π) (h3 : f x1 = 2 / 3) (h4 : f x2 = 2 / 3) : 
  Real.cos (x1 - x2) = 2 / 3 := 
begin
  sorry
end

end max_f_value_cos_x1_x2_value_l582_582139


namespace units_digit_of_sum_of_squares_of_first_2025_odd_integers_l582_582662

theorem units_digit_of_sum_of_squares_of_first_2025_odd_integers:
  let odd_squares : List ℕ := (List.range 2025).map (λ n, (2 * n + 1) ^ 2)
  let sum_of_odd_squares : ℕ := (odd_squares.sum)
  (sum_of_odd_squares % 10) = 5 :=
by
  sorry

end units_digit_of_sum_of_squares_of_first_2025_odd_integers_l582_582662


namespace find_number_l582_582341

theorem find_number (x : ℝ) : 
  0.05 * x = 0.20 * 650 + 190 → x = 6400 :=
by
  intro h
  sorry

end find_number_l582_582341


namespace integer_solutions_sum_eq_six_l582_582478

def f (x : ℝ) : ℝ :=
  (∑ k in finset.range 2012, |x + k|) +
  (∑ k in finset.range 1 2012, |x - k|)

theorem integer_solutions_sum_eq_six : 
  ∃ a : set ℤ, (∀ a ∈ a, f ((a : ℝ)^2 - 3 * (a : ℝ) + 2) = f ((a : ℝ) - 1)) ∧ (∑ a in a, a) = 6 :=
by
  sorry

end integer_solutions_sum_eq_six_l582_582478


namespace no_sum_of_consecutive_integers_to_420_l582_582183

noncomputable def perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

def sum_sequence (n a : ℕ) : ℕ :=
n * a + n * (n - 1) / 2

theorem no_sum_of_consecutive_integers_to_420 
  (h1 : 420 > 0)
  (h2 : ∀ (n a : ℕ), n ≥ 2 → sum_sequence n a = 420 → perfect_square a)
  (h3 : ∃ n a, n ≥ 2 ∧ sum_sequence n a = 420 ∧ perfect_square a) :
  false :=
by
  sorry

end no_sum_of_consecutive_integers_to_420_l582_582183


namespace value_range_of_f_l582_582286

noncomputable def f (x : ℝ) : ℝ := 2 + Real.logb 5 (x + 3)

theorem value_range_of_f :
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ∈ Set.Icc (2 : ℝ) 3 := 
by
  sorry

end value_range_of_f_l582_582286


namespace digits_packages_needed_l582_582997

-- We define the problem statement
theorem digits_packages_needed :
  (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 150 ∨ 200 ≤ n ∧ n ≤ 250 ∨ 300 ≤ n ∧ n ≤ 350 → 
  (∃ p : ℕ, ∀ d : fin 10, count_digit d.to_nat ((fin n).val : ℕ) ≤ count_digit d.to_nat p)) := by
  sorry

-- Helper function to count the occurrences of a digit in a number
noncomputable def count_digit (d n : ℕ) : ℕ :=
nat.digits 10 n |>.filter (λ x, x = d) |>.length

-- The proof is omitted and marked as sorry

end digits_packages_needed_l582_582997


namespace no_fractional_solutions_l582_582755

theorem no_fractional_solutions (x y : ℚ) (hx : x.denom ≠ 1) (hy : y.denom ≠ 1) :
  ¬ (∃ m n : ℤ, 13 * x + 4 * y = m ∧ 10 * x + 3 * y = n) :=
sorry

end no_fractional_solutions_l582_582755


namespace t_shirt_cost_calculation_l582_582234

variables (initial_amount ticket_cost food_cost money_left t_shirt_cost : ℕ)

axiom h1 : initial_amount = 75
axiom h2 : ticket_cost = 30
axiom h3 : food_cost = 13
axiom h4 : money_left = 9

theorem t_shirt_cost_calculation : 
  t_shirt_cost = initial_amount - (ticket_cost + food_cost) - money_left :=
sorry

end t_shirt_cost_calculation_l582_582234


namespace probability_same_carriage_l582_582627

theorem probability_same_carriage (num_carriages num_people : ℕ) (h1 : num_carriages = 10) (h2 : num_people = 3) : 
  ∃ p : ℚ, p = 7/25 ∧ p = 1 - (10 * 9 * 8) / (10^3) :=
by
  sorry

end probability_same_carriage_l582_582627


namespace num_ways_for_grade_in_A_l582_582064

-- Define the grades and the car seating arrangement
inductive Grade : Type
| first
| second
| third
| fourth

structure Student :=
  (name : String)
  (grade : Grade)

structure Car :=
  (students : List Student)

-- Define the condition of no splitting of twins
def twin_condition (students : List Student) : Prop :=
  ∀ s1 s2 : Student, (s1.grade = Grade.first ∧ s2.grade = Grade.first) → 
                      (s1.name ≠ s2.name → ∀ car1 car2 : Car, (car1.students.contains s1 ∧ car2.students.contains s2) → car1 = car2)

-- Define the problem conditions
def eight_students : List Student := [
  {name := "A1", grade := Grade.first}, {name := "A2", grade := Grade.first},
  {name := "B1", grade := Grade.second}, {name := "B2", grade := Grade.second},
  {name := "C1", grade := Grade.third}, {name := "C2", grade := Grade.third},
  {name := "D1", grade := Grade.fourth}, {name := "D2", grade := Grade.fourth}
]

-- Define function to count possible arrangements meeting the conditions
noncomputable def count_arrangements : Nat :=
  -- Logic to count the number of valid arrangements (abstracted for brevity)
  sorry

-- The theorem statement we aim to prove
theorem num_ways_for_grade_in_A : count_arrangements = 24 :=
  sorry

end num_ways_for_grade_in_A_l582_582064


namespace visits_per_hour_l582_582911

open Real

theorem visits_per_hour (price_per_visit : ℝ) (hours_per_day : ℕ) (days_per_month : ℕ) (total_earnings : ℝ) 
  (h_price : price_per_visit = 0.10)
  (h_hours : hours_per_day = 24)
  (h_days : days_per_month = 30)
  (h_earnings : total_earnings = 3600) :
  (total_earnings / (price_per_visit * hours_per_day * days_per_month) : ℝ) = 50 :=
by
  sorry

end visits_per_hour_l582_582911


namespace manuscript_pages_l582_582395

theorem manuscript_pages (P : ℝ)
  (h1 : 10 * (0.05 * P) + 10 * 5 = 250) : P = 400 :=
sorry

end manuscript_pages_l582_582395


namespace hitting_target_at_least_once_is_exclusive_of_both_shots_miss_l582_582890

-- Definitions extracted from the conditions:
def Event (Ω : Type) := Ω → Prop
variable {Ω : Type}
variable {shoots_twice : Event Ω}
variable hitting_both_shots : Event Ω
variable hitting_one_shot : Event Ω

-- The hypothesis
def hitting_target_at_least_once (e : Ω) : Prop :=
  hitting_both_shots e ∨ hitting_one_shot e

-- The mutually exclusive event to "hitting the target at least once" 
def both_shots_miss (e : Ω) : Prop :=
  ¬ hitting_both_shots e ∧ ¬ hitting_one_shot e

-- The statement we need to prove
theorem hitting_target_at_least_once_is_exclusive_of_both_shots_miss :
  ∀ e, hitting_target_at_least_once e ↔ ¬ both_shots_miss e :=
by
  sorry

end hitting_target_at_least_once_is_exclusive_of_both_shots_miss_l582_582890


namespace part_I_purely_imaginary_part_II_positive_l582_582459

def z1 (m : ℝ) : ℂ := (2 * m^2 : ℂ) / (1 - (1 : ℂ) * (0 + 1 * I))
def z2 (m : ℝ) : ℂ := (2 + I) * (m : ℂ) - 3 * ((1 : ℂ) + 2 * I)

-- Part (I)
theorem part_I_purely_imaginary (m : ℝ) :
  (z1 m + z2 m).re = 0 → m = 1 := 
sorry

-- Part (II)
theorem part_II_positive (m : ℝ) :
  (z1 m + z2 m).re > 0 → (z1 m * z2 m = 20 - 12 * I) := 
sorry

end part_I_purely_imaginary_part_II_positive_l582_582459


namespace absolute_value_fraction_l582_582935

variable (α β : ℂ) (k : ℝ)
variable (h₁ : β = 1)
variable (h₂ : α = k)
variable (h₃ : k > 1)

theorem absolute_value_fraction (h₁ : β = 1) (h₂ : α = k) (h₃ : k > 1) :
  abs ((β - α) / (1 - (conj α * β))) = 1 := by
  sorry

end absolute_value_fraction_l582_582935


namespace range_a_l582_582135

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x / x) - log a

noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 + 1) / (Real.exp 1 * x)

def intersect_points (a : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), x1 ∈ Set.Ioo (-1) 0 ∪ Set.Ioc 0 1 ∧ x2 ∈ Set.Ioo (-1) 0 ∪ Set.Ioc 0 1 ∧ f a x1 = g x1 ∧ f a x2 = g x2

theorem range_a (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  (intersect_points a ↔ (a ∈ (Set.Ioo 0 (1/Real.exp 1)) ∪ Set.Ici (Real.exp 1))) :=
sorry

end range_a_l582_582135


namespace seashells_total_now_l582_582858

def henry_collected : ℕ := 11
def paul_collected : ℕ := 24
def total_initial : ℕ := 59
def leo_initial (henry_collected paul_collected total_initial : ℕ) : ℕ := total_initial - (henry_collected + paul_collected)
def leo_gave (leo_initial : ℕ) : ℕ := leo_initial / 4
def total_now (total_initial leo_gave : ℕ) : ℕ := total_initial - leo_gave

theorem seashells_total_now :
  total_now total_initial (leo_gave (leo_initial henry_collected paul_collected total_initial)) = 53 :=
sorry

end seashells_total_now_l582_582858


namespace number_of_complex_numbers_l582_582410

theorem number_of_complex_numbers (z : ℂ) : |z| = 1 ∧ |((z^2) / (conj(z)^2)) + ((conj(z)^2) / (z^2))| = 1 → 
  ∃ n, n = 12 := 
sorry

end number_of_complex_numbers_l582_582410


namespace M_function_inequality_l582_582807

-- Define what it means to be an M-function
def is_M_function (f : ℝ → ℝ) : Prop := 
  ∀ x, 0 < x → (x * (f'' x)) > f x

-- State the main theorem
theorem M_function_inequality {f : ℝ → ℝ} (h : is_M_function f) 
  (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) : 
  f(x1) + f(x2) < f(x1 + x2) := 
sorry

end M_function_inequality_l582_582807


namespace gcd_gx_x_l582_582466

noncomputable def g (x : ℕ) := (5 * x + 3) * (11 * x + 2) * (6 * x + 7) * (3 * x + 8)

theorem gcd_gx_x {x : ℕ} (hx : 36000 ∣ x) : Nat.gcd (g x) x = 144 := by
  sorry

end gcd_gx_x_l582_582466


namespace intersection_complement_P_CUQ_l582_582562

universe U

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}
def CUQ : Set ℕ := U \ Q

theorem intersection_complement_P_CUQ : 
  (P ∩ CUQ) = {1, 2} :=
by 
  sorry

end intersection_complement_P_CUQ_l582_582562


namespace compound_propositions_l582_582112

def p := ∀ x y : ℝ, x > y → -x < -y
def q := ∀ x y : ℝ, x < y → x^2 > y^2

theorem compound_propositions :
  (p ∨ q) ∧ (p ∧ ¬ q) :=
by {
  have h1 : p := λ x y h, by linarith,
  have h2 : ¬ q := λ x y, by {
    intros h1, 
    use [1, -1],
    linarith,
  },
  exact ⟨or.inl h1, ⟨h1, h2⟩⟩,
}

end compound_propositions_l582_582112


namespace trig_identity_problem_l582_582081

theorem trig_identity_problem :
  (sin (Real.pi / 12) * cos (Real.pi / 12) + cos (11 * Real.pi / 12) * cos (7 * Real.pi / 12)) /
  (sin (19 * Real.pi / 180) * cos (11 * Real.pi / 180) + cos (161 * Real.pi / 180) * cos (101 * Real.pi / 180)) = 1 := sorry

end trig_identity_problem_l582_582081


namespace bacterium_length_in_scientific_notation_l582_582960

theorem bacterium_length_in_scientific_notation :
  (1 : ℝ) * 10^-9 * (50 : ℝ) = 5 * 10^-8 :=
by
  sorry

end bacterium_length_in_scientific_notation_l582_582960


namespace susie_vacuums_each_room_in_20_minutes_l582_582588

theorem susie_vacuums_each_room_in_20_minutes
  (total_time_hours : ℕ)
  (number_of_rooms : ℕ)
  (total_time_minutes : ℕ)
  (time_per_room : ℕ)
  (h1 : total_time_hours = 2)
  (h2 : number_of_rooms = 6)
  (h3 : total_time_minutes = total_time_hours * 60)
  (h4 : time_per_room = total_time_minutes / number_of_rooms) :
  time_per_room = 20 :=
by
  sorry

end susie_vacuums_each_room_in_20_minutes_l582_582588


namespace part1_part2_l582_582479

def f (x : ℝ) : ℝ :=
  if x > 1 then 1 + 1/x
  else if x >= -1 then x^2 + 1
  else 2*x + 3

theorem part1 : f (f (f (-2))) = 3 / 2 := 
by sorry 

theorem part2 (a: ℝ) (h : f a = 3 / 2) : a = 2 ∨ a = sqrt 2 / 2 ∨ a = - sqrt 2 / 2 :=
by sorry

end part1_part2_l582_582479


namespace compound_h_atoms_l582_582686

theorem compound_h_atoms 
  (weight_H : ℝ) (weight_C : ℝ) (weight_O : ℝ)
  (num_C : ℕ) (num_O : ℕ)
  (total_molecular_weight : ℝ)
  (atomic_weight_H : ℝ) (atomic_weight_C : ℝ) (atomic_weight_O : ℝ)
  (H_w_is_1 : atomic_weight_H = 1)
  (C_w_is_12 : atomic_weight_C = 12)
  (O_w_is_16 : atomic_weight_O = 16)
  (C_atoms_is_1 : num_C = 1)
  (O_atoms_is_3 : num_O = 3)
  (total_mw_is_62 : total_molecular_weight = 62)
  (mw_C : weight_C = num_C * atomic_weight_C)
  (mw_O : weight_O = num_O * atomic_weight_O)
  (mw_CO : weight_C + weight_O = 60)
  (H_weight_contrib : total_molecular_weight - (weight_C + weight_O) = weight_H)
  (H_atoms_calc : weight_H = 2 * atomic_weight_H) :
  2 = 2 :=
by 
  sorry

end compound_h_atoms_l582_582686


namespace equal_expected_displeasure_l582_582238

-- Definitions
def mistake_probability := 1 / 3
def displeasure_no_pie := 2 * α
def displeasure_two_pies := α
def madame_displeasure_self := 1 / 2 * α
def son_displeasure := 1 / 3 * 2 * α + 1 / 3 * α

-- Proof statement
theorem equal_expected_displeasure (α : ℝ):
  madame_displeasure_self = son_displeasure :=
by
  sorry

end equal_expected_displeasure_l582_582238


namespace fertilizer_percentage_l582_582235

theorem fertilizer_percentage (total_volume : ℝ) (vol_74 : ℝ) (vol_53 : ℝ) (perc_74 : ℝ) (perc_53 : ℝ) (final_perc : ℝ) :
  total_volume = 42 ∧ vol_74 = 20 ∧ vol_53 = total_volume - vol_74 ∧ perc_74 = 0.74 ∧ perc_53 = 0.53 
  → final_perc = ((vol_74 * perc_74 + vol_53 * perc_53) / total_volume) * 100
  → final_perc = 63.0 :=
by
  intros
  sorry

end fertilizer_percentage_l582_582235


namespace unique_tuple_of_sums_l582_582915

-- Define the problem in Lean, stating that b is a positive integer and all terms meet the condition
theorem unique_tuple_of_sums (b : ℕ) (hb : 0 < b) (a : Fin 2002 → ℕ) :
  (∑ i, (a i) ^ (a i) = 2002 * b ^ b) → (∀ i, a i = b) := 
sorry

end unique_tuple_of_sums_l582_582915


namespace largest_possible_n_l582_582739

def triangle_property (s : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), {a, b, c} ⊆ s → c ≤ a + b

def largest_n_with_triangle_property (m : ℕ) : Prop :=
  (∀ s : Finset ℕ, s.card = 10 → triangle_property s) →
  m = 363

theorem largest_possible_n :
  ∃ n, largest_n_with_triangle_property n :=
begin
  use 363,
  intros s h,
  -- The proof would go here
  sorry
end

end largest_possible_n_l582_582739


namespace circles_intersect_l582_582612

def circle (c : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) := 
  { p | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 }

def C1 := circle (4, 0) 2 
def C2 := circle (0, 3) 4

theorem circles_intersect : 
  ∃ p : ℝ × ℝ, p ∈ C1 ∧ p ∈ C2 := 
sorry

end circles_intersect_l582_582612


namespace should_increase_speed_by_30_l582_582195

variable (T : ℝ) -- Usual travel time in minutes.
variable (v : ℝ) -- Usual speed.
variable (D : ℝ) -- Distance to work.
variable (increased_v : ℝ) := 1.6 * v -- Speed increased by 60%.

-- Given usual travel speed equation
variable (h_v : v = D / T)

-- Given increased speed equation and time saved
variable (delayed_time : ℝ) := T + 40  -- 40 minutes late
variable (arrival_time_saved : ℝ) := 65 -- Total of 65 minutes saved because of increased speed and late start

-- Setting up the new equations based on the conditions
variable (new_travel_time : ℝ) := (T - 65)
variable (new_speed_calculation : ℝ) := v * (T / (T - 40))

-- Prove that increasing the usual speed by 30% results in arriving at 9:00 AM if departed 40 minutes late.
theorem should_increase_speed_by_30 :
  (new_speed_calculation = 1.3 * v) :=
sorry

end should_increase_speed_by_30_l582_582195


namespace compound_interest_second_year_l582_582597

theorem compound_interest_second_year
  (P : ℝ) (r : ℝ) (CI_3 : ℝ) (CI_2 : ℝ) 
  (h1 : r = 0.08) 
  (h2 : CI_3 = 1512)
  (h3 : CI_3 = CI_2 * (1 + r)) :
  CI_2 = 1400 :=
by
  rw [h1, h2] at h3
  sorry

end compound_interest_second_year_l582_582597


namespace factorable_polynomial_with_integer_coeffs_l582_582056

theorem factorable_polynomial_with_integer_coeffs (m : ℤ) : 
  ∃ A B C D E F : ℤ, 
  (A * D = 1) ∧ (B * E = 0) ∧ (A * E + B * D = 5) ∧ 
  (A * F + C * D = 1) ∧ (B * F + C * E = 2 * m) ∧ (C * F = -10) ↔ m = 5 := sorry

end factorable_polynomial_with_integer_coeffs_l582_582056


namespace smallest_positive_integers_satisfying_cos_eq_1_l582_582057

theorem smallest_positive_integers_satisfying_cos_eq_1 :
  ∃ (k₁ k₂ : ℕ), (cos ((k₁^2 + 64 : ℕ) * π / 180) = 1) ∧ (cos ((k₂^2 + 64 : ℕ) * π / 180) = 1) ∧ 
    k₁ < k₂ ∧ (∀ k : ℕ, (cos ((k^2 + 64 : ℕ) * π / 180) = 1) → k = 24 ∨ k = 40) :=
by
  -- Statement indicates k₁ and k₂ are the two smallest positive integers satisfying the condition
  sorry

end smallest_positive_integers_satisfying_cos_eq_1_l582_582057


namespace money_combination_l582_582295

variable (Raquel Tom Nataly Sam : ℝ)

-- Given Conditions 
def condition1 : Prop := Tom = (1 / 4) * Nataly
def condition2 : Prop := Nataly = 3 * Raquel
def condition3 : Prop := Sam = 2 * Nataly
def condition4 : Prop := Nataly = (5 / 3) * Sam
def condition5 : Prop := Raquel = 40

-- Proving this combined total
def combined_total : Prop := Tom + Raquel + Nataly + Sam = 262

theorem money_combination (h1: condition1 Tom Nataly) 
                          (h2: condition2 Nataly Raquel) 
                          (h3: condition3 Sam Nataly) 
                          (h4: condition4 Nataly Sam) 
                          (h5: condition5 Raquel) 
                          : combined_total Tom Raquel Nataly Sam :=
sorry

end money_combination_l582_582295


namespace correct_answer_is_D_l582_582213

variables {Line Plane : Type}
variable (m n : Line)
variable (alpha beta : Plane)

-- Definitions for lines being contained in a plane, perpendicular, and parallel to planes
def is_contained_in (l : Line) (p : Plane) : Prop := sorry
def is_perpendicular_to (l : Line) (p : Plane) : Prop := sorry
def is_parallel_to (l : Line) (p : Plane) : Prop := sorry

-- Proposition 1
def prop1 : Prop := 
  is_contained_in m beta ∧ is_perpendicular_to alpha beta → is_perpendicular_to m alpha

-- Proposition 2
def prop2 : Prop := 
  is_contained_in m beta ∧ is_parallel_to alpha beta → is_parallel_to m alpha

-- Proposition 3
def prop3 : Prop := 
  is_perpendicular_to m alpha ∧ is_perpendicular_to m beta ∧ is_perpendicular_to n alpha → 
  is_perpendicular_to n beta

-- Proposition 4
def prop4 : Prop := 
  is_parallel_to m alpha ∧ is_parallel_to m beta ∧ is_parallel_to n alpha → 
  is_parallel_to n beta

-- Final statement to prove correct answer
theorem correct_answer_is_D : (prop2 ∧ prop3) ∧ ¬ prop1 ∧ ¬ prop4 :=
sorry

end correct_answer_is_D_l582_582213


namespace binomial_expansion_limit_l582_582134

open Real BigOperators

theorem binomial_expansion_limit (a : ℕ → ℕ → ℤ) (T R : ℕ → ℤ) :
  (∀ n, (3 * (x : ℝ) - 1)^(2*n) = (∑ i in range (2*n + 1), a n i * x^i)) →
  (∀ n, T n = ∑ i in range (n + 1), a n (2 * i)) →
  (∀ n, R n = ∑ i in range (n + 1), a n (2 * i + 1)) →
  (∀ n, ∀ x, (3 * x - 1)^(2 * n) = 2^(2 * n)) →
  (∀ n, ∀ x, (3 * (-x) - 1)^(2 * n) = 4^(2 * n)) →
  tendsto (λ n, (T n)/(R n) : ℕ → ℝ) at_top (𝓝 (-1)) :=
  by
  sorry

end binomial_expansion_limit_l582_582134


namespace program_output_l582_582665

def program (a b n : ℤ) : ℤ :=
  let rec loop (a b c i : ℤ) : ℤ :=
    if i > n - 2 then
      c
    else
      loop b (a + b) (a + b) (i + 1)
  loop a b 0 1

theorem program_output (a b n : ℤ) (hₐ : a = 3) (h_b : b = -1) (h_n : n = 5) : 
  program a b n = 3 := 
by
  sorry

end program_output_l582_582665


namespace solve_3x_plus_7y_eq_23_l582_582797

theorem solve_3x_plus_7y_eq_23 :
  ∃ (x y : ℕ), 3 * x + 7 * y = 23 ∧ x = 3 ∧ y = 2 := by
sorry

end solve_3x_plus_7y_eq_23_l582_582797


namespace factorization_l582_582070

theorem factorization (a : ℝ) : 2 * a ^ 2 - 8 = 2 * (a + 2) * (a - 2) := 
by
  sorry

end factorization_l582_582070


namespace probability_of_first_heart_second_king_l582_582640

noncomputable def probability_first_heart_second_king : ℚ :=
  1 / 52 * 3 / 51 + 12 / 52 * 4 / 51

theorem probability_of_first_heart_second_king :
  probability_first_heart_second_king = 1 / 52 :=
by
  sorry

end probability_of_first_heart_second_king_l582_582640


namespace concert_total_audience_l582_582717

-- Definitions based on conditions
def audience_for_second_band (total_audience : ℕ) : ℕ := (2 * total_audience) / 3
def audience_for_first_band (total_audience : ℕ) : ℕ := total_audience / 3

def under_30_audience_for_second_band (total_audience : ℕ) : ℕ := audience_for_second_band(total_audience) / 2

def men_under_30_for_second_band : ℕ := 20
def percentage_men_under_30_for_second_band : ℕ := 40 -- 40%

def total_under_30_for_second_band : ℕ := men_under_30_for_second_band * 100 / percentage_men_under_30_for_second_band

-- Main theorem statement
theorem concert_total_audience (total_audience : ℕ) :
  audience_for_second_band(total_audience) = 100 →
  (audience_for_second_band(total_audience) * 50) / 100 = total_under_30_for_second_band →
  total_audience = 150 :=
by
  intros h1 h2
  sorry

end concert_total_audience_l582_582717


namespace probability_of_first_heart_second_king_l582_582642

noncomputable def probability_first_heart_second_king : ℚ :=
  1 / 52 * 3 / 51 + 12 / 52 * 4 / 51

theorem probability_of_first_heart_second_king :
  probability_first_heart_second_king = 1 / 52 :=
by
  sorry

end probability_of_first_heart_second_king_l582_582642


namespace smallest_integer_correct_l582_582085

def odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

def smallest_positive_integer_with_conditions : ℕ :=
  11341

theorem smallest_integer_correct :
  let n := smallest_positive_integer_with_conditions in
  n > 10000 ∧
  odd_digits n ∧
  n % 11 = 0 :=
by
  let n := smallest_positive_integer_with_conditions
  have h₁ : n > 10000 := by sorry
  have h₂ : odd_digits n := by sorry
  have h₃ : n % 11 = 0 := by sorry
  exact ⟨h₁, ⟨h₂, h₃⟩⟩

end smallest_integer_correct_l582_582085


namespace num_arrangements_l582_582685

-- Definitions for the conditions in the problem.
def is_adjacent (x y : ℕ) (l : list ℕ) : Prop :=
  ∃ (i : ℕ), l.nth i = some x ∧ l.nth (i + 1) = some y ∨ l.nth (i - 1) = some y

def not_adjacent (x y : ℕ) (l : list ℕ) : Prop :=
  ∀ (i : ℕ), l.nth i = some x → l.nth (i + 1) ≠ some y ∧ l.nth (i - 1) ≠ some y

variable programs : list ℕ := [1, 2, 3, 4, 5, 6]  -- Assume each number represents a program where 1 = A, 2 = B, 3 = C, 4 = D, 5 = E, 6 = F

-- Main statement to prove the number of arrangements
theorem num_arrangements : 
  let l := programs in 
  let arrangements := l.permutations.filter (λ p, is_adjacent 1 2 p ∧ not_adjacent 3 4 p) in
  arrangements.length = 144 :=
by sorry

end num_arrangements_l582_582685


namespace constant_term_in_binomial_expansion_l582_582443

noncomputable def integral_value : ℝ := (1 / Real.pi) * (Real.intervalIntegral.integral (λ x, Real.sqrt (1 - x^2) + Real.sin x) (-1) 1)

theorem constant_term_in_binomial_expansion :
  integral_value = 1 / 2 →
  let a := integral_value in
  (let expr := (2 * x - a / x^2)^9 in
   expr.expand // some method to compute the expansion and extract a specific term
   ) 
  ∃ constant_term, constant_term = -672 :=
sorry

end constant_term_in_binomial_expansion_l582_582443


namespace term_2023_of_sequence_is_370_l582_582605

def sum_of_cubes_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.map (λ c => (c.toNat - '0'.toNat)^3)).sum

def sequence_term (n : ℕ) : ℕ :=
  Nat.iterate (λ x => sum_of_cubes_of_digits x) n

theorem term_2023_of_sequence_is_370 :
  sequence_term 2023^2023 = 370 :=
sorry

end term_2023_of_sequence_is_370_l582_582605


namespace linear_systems_l582_582039

def is_linear_system (S : list (ℚ × ℚ × ℚ)) : Prop :=
  ∀ (eq : ℚ × ℚ × ℚ), eq ∈ S → True

def set_1 : list (ℚ × ℚ × ℚ) := [(1, 0, 2), (0, 1, 3)]
def set_2 : list (ℚ × ℚ × ℚ) := [(4, 0, 15), (3, -4, -3)]
def set_3 : list (ℚ × ℚ × ℚ) := [(1, 1, 16), (1, 0, 4)] -- Note: Non-linear, but keeping for structure
def set_4 : list (ℚ × ℚ × ℚ) := [(1, 1, 35), (2, 4, 94)]

theorem linear_systems (S₁ S₂ S₃ S₄ : list (ℚ × ℚ × ℚ)) 
  (h₁ : S₁ = set_1) (h₂ : S₂ = set_2) (h₃ : S₃ = set_3) (h₄ : S₄ = set_4) :
  is_linear_system S₁ ∧ is_linear_system S₂ ∧ is_linear_system S₄ :=
by {
  subst h₁, subst h₂, subst h₃, subst h₄,
  split,
  { intros eq heq, trivial },
  split,
  { intros eq heq, trivial },
  { intros eq heq, trivial },
}

end linear_systems_l582_582039


namespace percent_problem_l582_582504

variable (x : ℝ)

theorem percent_problem (h : 0.30 * 0.15 * x = 27) : 0.15 * 0.30 * x = 27 :=
by sorry

end percent_problem_l582_582504


namespace country_x_income_l582_582404

theorem country_x_income (I : ℝ) (h1 : I > 40000) (_ : 0.15 * 40000 + 0.20 * (I - 40000) = 8000) : I = 50000 :=
sorry

end country_x_income_l582_582404


namespace Loisa_saves_70_l582_582227

-- Define the conditions
def tablet_cost_cash := 450
def down_payment := 100
def payment_first_4_months := 40 * 4
def payment_next_4_months := 35 * 4
def payment_last_4_months := 30 * 4

-- Define the total installment payment
def total_installment_payment := down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

-- Define the amount saved by paying cash instead of on installment
def amount_saved := total_installment_payment - tablet_cost_cash

-- The theorem to prove the savings amount
theorem Loisa_saves_70 : amount_saved = 70 := by
  -- Direct calculation or further proof steps here
  sorry

end Loisa_saves_70_l582_582227


namespace quadratic_symmetric_l582_582592

theorem quadratic_symmetric {a b c : ℝ} :
  ∃ a b c, 
  (∀ x, x^2 - 3 * x - 4 = a * x^2 + b * x + c) →
  (y=x^2 - 3 * x - 4 → symmetric about O(0,0)) →
  (a = -1) ∧ (b = -3) ∧ (c = 4) :=
by
  sorry

end quadratic_symmetric_l582_582592


namespace molecular_weight_constant_l582_582049

-- Definitions of atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

-- Definition of molecular weight calculation for Aluminum carbonate
def molecular_weight_Al2CO3 : ℝ :=
  2 * atomic_weight_Al + 3 * atomic_weight_C + 9 * atomic_weight_O

-- Theorem stating the molecular weight remains constant regardless of given conditions
theorem molecular_weight_constant : molecular_weight_Al2CO3 = 233.99 :=
by
  have h : molecular_weight_Al2CO3 = 2 * 26.98 + 3 * 12.01 + 9 * 16.00
  calc
    molecular_weight_Al2CO3 = 2 * atomic_weight_Al + 3 * atomic_weight_C + 9 * atomic_weight_O : by rfl
    ... = 2 * 26.98 + 3 * 12.01 + 9 * 16.00                             : by sorry
    ... = 53.96 + 36.03 + 144.00                                        : by sorry
    ... = 233.99                                                        : by sorry

end molecular_weight_constant_l582_582049


namespace matrix_problem_l582_582382

noncomputable def matrix_5x5 := Matrix (Fin 5) (Fin 5) ℝ

variables {a : matrix_5x5}
variables {r1 : ∀ j : Fin 5, a 0 j = a 0 0 + j * d}
variables {d : ℝ}
variables {q : ℝ}

-- Condition: The first row forms an arithmetic sequence.
-- Each column forms a geometric sequence with the same common ratio
variables (h_geom_seq : ∀ i j : Fin 5, i ≠ 0 → a i j = a 0 j * q ^ i)

-- Specific values
variables (h24 : a (2 : Fin 5) (3 : Fin 5) = 4) 
variables (h41 : a (3 : Fin 5) (0 : Fin 5) = -2) 
variables (h43 : a (3 : Fin 5) (2 : Fin 5) = 10)

theorem matrix_problem : a (0 : Fin 5) (0 : Fin 5) * a (4 : Fin 5) (4 : Fin 5) = -11 := 
  sorry

end matrix_problem_l582_582382


namespace polynomial_square_solution_l582_582274

variable (a b : ℝ)

theorem polynomial_square_solution (h : 
  ∃ g : Polynomial ℝ, g^2 = Polynomial.C (1 : ℝ) * Polynomial.X^4 -
  Polynomial.C (1 : ℝ) * Polynomial.X^3 +
  Polynomial.C (1 : ℝ) * Polynomial.X^2 +
  Polynomial.C a * Polynomial.X +
  Polynomial.C b) : b = 9 / 64 :=
by sorry

end polynomial_square_solution_l582_582274


namespace sequence_2023rd_term_is_153_l582_582603

def sum_cubes_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d, d ^ 3).sum

def sequence_term (n : ℕ) : ℕ :=
  (nat.iterate sum_cubes_of_digits n) n

theorem sequence_2023rd_term_is_153 : sequence_term 2023 = 153 := 
sorry

end sequence_2023rd_term_is_153_l582_582603


namespace sin_beta_value_l582_582163

theorem sin_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
  (h1 : Real.cos α = 4 / 5) (h2 : Real.cos (α + β) = 5 / 13) :
  Real.sin β = 33 / 65 :=
sorry

end sin_beta_value_l582_582163


namespace circle_area_l582_582883

theorem circle_area (r : ℝ) (h : 3 * (1 / (2 * π * r)) = r) : π * r^2 = 3 / 2 :=
by
  -- We leave this place for computations and derivations.
  sorry

end circle_area_l582_582883


namespace simplify_and_evaluate_expression_l582_582584

theorem simplify_and_evaluate_expression (x : ℤ) (h1 : -2 < x) (h2 : x < 3) :
    (x ≠ 1) → (x ≠ -1) → (x ≠ 0) → 
    ((x / (x + 1) - (3 * x) / (x - 1)) / (x / (x^2 - 1))) = -8 :=
by 
  intro h3 h4 h5
  sorry

end simplify_and_evaluate_expression_l582_582584


namespace area_R3_l582_582360

-- Define the initial dimensions of rectangle R1
def length_R1 := 8
def width_R1 := 4

-- Define the dimensions of rectangle R2 after bisecting R1
def length_R2 := length_R1 / 2
def width_R2 := width_R1

-- Define the dimensions of rectangle R3 after bisecting R2
def length_R3 := length_R2 / 2
def width_R3 := width_R2

-- Prove that the area of R3 is 8
theorem area_R3 : (length_R3 * width_R3) = 8 := by
  -- Calculation for the theorem
  sorry

end area_R3_l582_582360


namespace total_routes_from_A_to_C_l582_582835

theorem total_routes_from_A_to_C (highways_from_A_to_B paths_from_B_to_C waterway_from_A_to_C : ℕ)
    (h1 : highways_from_A_to_B = 2)
    (h2 : paths_from_B_to_C = 3)
    (h3 : waterway_from_A_to_C = 1) :
    highways_from_A_to_B * paths_from_B_to_C + waterway_from_A_to_C = 7 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end total_routes_from_A_to_C_l582_582835


namespace non_working_games_count_l582_582957

def total_games : ℕ := 16
def price_each : ℕ := 7
def total_earnings : ℕ := 56

def working_games : ℕ := total_earnings / price_each
def non_working_games : ℕ := total_games - working_games

theorem non_working_games_count : non_working_games = 8 := by
  sorry

end non_working_games_count_l582_582957


namespace locus_of_X_is_line_segments_to_midpoints_l582_582399

-- Define the problem statement
noncomputable def regular_ngon (n : ℕ) (h : n ≥ 5) : Type := sorry

-- Definition of center point O of the regular n-gon
def center (n : ℕ) (h : n ≥ 5) : regular_ngon n h → pt := sorry

-- Definitions of points A and B as vertices of the polygon
def vertex_A (n : ℕ) (h : n ≥ 5) : regular_ngon n h → pt := sorry
def vertex_B (n : ℕ) (h : n ≥ 5) : regular_ngon n h → pt := sorry

-- Definition of triangle XYZ congruent to triangle OAB
structure triangle (A B C : pt) : Type := mk :: (a : pt) (b : pt) (c : pt)

def congruent_triangles (ABC XYZ : triangle) : Prop := sorry

-- Initial position of triangle XYZ overlapping triangle OAB
def initial_position (n : ℕ) (h : n ≥ 5) : Prop :=
  ∃ (X O : pt) (Y A Z B : pt), 
    triangle.mk O A B = triangle.mk X Y Z ∧ congruent_triangles (triangle.mk O A B) (triangle.mk X Y Z)

-- Movement of Y and Z on the perimeter and X within the polygon
def movement (n : ℕ) (h : n ≥ 5) : Prop := sorry

-- The final proof statement
theorem locus_of_X_is_line_segments_to_midpoints (n : ℕ) (h : n ≥ 5) : 
  ∀ (polygon : regular_ngon n h) (O A B : pt), 
    initial_position n h → 
    (X : pt), movement n h → 
    (X.locus_inside_polygon n h) :=
sorry

end locus_of_X_is_line_segments_to_midpoints_l582_582399


namespace sum_of_repeating_decimals_l582_582734

-- Definitions of repeating decimals x and y
def x : ℚ := 25 / 99
def y : ℚ := 87 / 99

-- The assertion that the sum of these repeating decimals is equal to 112/99 as a fraction
theorem sum_of_repeating_decimals: x + y = 112 / 99 := by
  sorry

end sum_of_repeating_decimals_l582_582734


namespace more_girls_than_boys_l582_582517

variables (boys girls : ℕ)

def ratio_condition : Prop := (3 * girls = 4 * boys)
def total_students_condition : Prop := (boys + girls = 42)

theorem more_girls_than_boys (h1 : ratio_condition boys girls) (h2 : total_students_condition boys girls) :
  (girls - boys = 6) :=
sorry

end more_girls_than_boys_l582_582517


namespace part_I_part_II_part_III_l582_582841

noncomputable def f_I (x : ℝ) : ℝ := x * sin x - cos x + x
def tangent_line_form (y : ℝ) (m : ℝ) (x: ℝ) : ℝ := m * x + y

theorem part_I (x : ℝ) : 
  tangent_line_form (f_I 0) (deriv f_I 0) x = x - 1 :=
sorry

noncomputable def f_II (x : ℝ) : ℝ := x * sin x + 2 * cos x + x

theorem part_II : 
  (∀ x ∈ set.Icc 0 (π / 2), f_II x ≤ f_II (π / 2)) ∧
  (∀ x ∈ set.Icc 0 (π / 2), f_II x ≥ f_II 0) :=
sorry

noncomputable def f_III (a : ℝ) (x : ℝ) : ℝ := x * sin x + a * cos x + x

theorem part_III (a : ℝ) : 
  (2 < a ∧ a ≤ 3) → 
  (∃! x ∈ set.Icc 0 (π / 2), f_III a x = 3) :=
sorry

end part_I_part_II_part_III_l582_582841


namespace integral_solution_l582_582077

noncomputable def integral_identity (x : ℝ) : ℝ :=
  ∫ (dx : ℝ) in (1 / (Real.sin x ^ 2 * Real.cos x ^ 2))

theorem integral_solution :
  ∃ C : ℝ, integral_identity = -Real.cot x + Real.tan x + C :=
by
  have h1 : ∀ x, Real.sin x ^ 2 + Real.cos x ^ 2 = 1 := by sorry
  have h2 : ∀ x, ∫ (dx : ℝ) in (1 / (Real.sin x ^ 2)) = -Real.cot x := by sorry
  have h3 : ∀ x, ∫ (dx : ℝ) in (1 / (Real.cos x ^ 2)) = Real.tan x := by sorry
  sorry

end integral_solution_l582_582077


namespace paula_walked_approximately_3200_km_l582_582576

theorem paula_walked_approximately_3200_km
  (resets_at : ℕ)
  (initial_reading : ℕ)
  (flip_times : ℕ)
  (final_reading : ℕ)
  (steps_per_km : ℕ)
  (h1 : resets_at = 49999)
  (h2 : initial_reading = 0)
  (h3 : flip_times = 76)
  (h4 : final_reading = 25000)
  (h5 : steps_per_km = 1200) :
  (flip_times * (resets_at + 1) + final_reading) / steps_per_km ≈ 3200 := by
sorry

end paula_walked_approximately_3200_km_l582_582576


namespace find_z_coordinate_l582_582694

-- Definition of the points
def point1 := (3, 3, 2)
def point2 := (6, 2, 0)

-- Definition of the direction vector
def direction_vector := (3, -1, -2)

-- Definition of the parameterized line
def param_line (t : ℝ) := (3 + 3 * t, 3 - t, 2 - 2 * t)

-- Main theorem stating the desired z-coordinate when x-coordinate is 7
theorem find_z_coordinate : 
  ∃ t : ℝ, (param_line t).1 = 7 → (param_line t).3 = -2 / 3 :=
by
  existsi 4 / 3
  intros h
  sorry

end find_z_coordinate_l582_582694


namespace find_n_l582_582692

open Real -- or open Complex if you prefer complex numbers

noncomputable theory

variable (x : ℝ)
variable (b : ℕ → ℝ)

-- Conditions
def geometric_sequence : Prop := ∀ n, b (n + 1) = x * b n
def initial_terms : Prop := b 1 = exp x ∧ b 2 = x * exp x

-- Statement to be proved
theorem find_n (h1 : geometric_sequence b x) (h2 : initial_terms b x) : b 3 = x^2 * exp x → 3 = 3 := sorry

end find_n_l582_582692


namespace centers_of_inscribed_circles_form_rectangle_l582_582359

/-!
Given a cyclic quadrilateral ABCD, prove that the centers of the inscribed circles
in the triangles formed by the diagonals are the vertices of a rectangle.
-/

theorem centers_of_inscribed_circles_form_rectangle
  (A B C D : Point) (O₁ O₂ O₃ O₄ : Point)
  (h1 : CyclicQuadrilateral A B C D)
  (h2 : CenterOfInscribedCircle (Triangle A B E) O₁)
  (h3 : CenterOfInscribedCircle (Triangle B C F) O₂)
  (h4 : CenterOfInscribedCircle (Triangle C D G) O₃)
  (h5 : CenterOfInscribedCircle (Triangle D A H) O₄) :
  IsRectangle O₁ O₂ O₃ O₄ :=
sorry

end centers_of_inscribed_circles_form_rectangle_l582_582359


namespace probability_of_first_heart_second_king_l582_582641

noncomputable def probability_first_heart_second_king : ℚ :=
  1 / 52 * 3 / 51 + 12 / 52 * 4 / 51

theorem probability_of_first_heart_second_king :
  probability_first_heart_second_king = 1 / 52 :=
by
  sorry

end probability_of_first_heart_second_king_l582_582641


namespace complex_power_sum_l582_582218

theorem complex_power_sum (z : ℂ) (h₁ : z = (1 + Complex.i) / Real.sqrt 2) (h₂ : z = Complex.exp (Real.pi * Complex.i / 4)) :
  (∑ k in Finset.range 12, z^(k^2)) * (∑ k in Finset.range 12, (1 / z)^(k^2)) = 36 :=
by 
  have : Complex.abs (z * Complex.conj z) = 1 := sorry  -- This uses the property |z|^2 = 1
  sorry

end complex_power_sum_l582_582218


namespace sum_of_common_ratios_l582_582941

-- Definitions for the geometric sequence conditions
def geom_seq_a (m : ℝ) (s : ℝ) (n : ℕ) : ℝ := m * s^n
def geom_seq_b (m : ℝ) (t : ℝ) (n : ℕ) : ℝ := m * t^n

-- Theorem statement
theorem sum_of_common_ratios (m s t : ℝ) (h₀ : m ≠ 0) (h₁ : s ≠ t) 
    (h₂ : geom_seq_a m s 2 - geom_seq_b m t 2 = 3 * (geom_seq_a m s 1 - geom_seq_b m t 1)) :
    s + t = 3 :=
by
  sorry

end sum_of_common_ratios_l582_582941


namespace f_x_plus_3_eq_f_x_l582_582938

def f (x : ℕ) : ℕ := x % 3

theorem f_x_plus_3_eq_f_x (x : ℕ) : f (x + 3) = f x :=
by
  unfold f
  have : (x + 3) % 3 = x % 3; sorry

end f_x_plus_3_eq_f_x_l582_582938


namespace janet_total_owed_l582_582543

def warehouseHourlyWage : ℝ := 15
def managerHourlyWage : ℝ := 20
def numWarehouseWorkers : ℕ := 4
def numManagers : ℕ := 2
def workDaysPerMonth : ℕ := 25
def workHoursPerDay : ℕ := 8
def ficaTaxRate : ℝ := 0.10

theorem janet_total_owed : 
  let warehouseWorkerMonthlyWage := warehouseHourlyWage * workDaysPerMonth * workHoursPerDay
  let managerMonthlyWage := managerHourlyWage * workDaysPerMonth * workHoursPerDay
  let totalMonthlyWages := (warehouseWorkerMonthlyWage * numWarehouseWorkers) + (managerMonthlyWage * numManagers)
  let ficaTaxes := totalMonthlyWages * ficaTaxRate
  let totalAmountOwed := totalMonthlyWages + ficaTaxes
  totalAmountOwed = 22000 := by
  sorry

end janet_total_owed_l582_582543


namespace zeoland_speeding_fine_l582_582886

-- Define the conditions
def fine_per_mph (total_fine : ℕ) (actual_speed : ℕ) (speed_limit : ℕ) : ℕ :=
  total_fine / (actual_speed - speed_limit)

-- Variables for the given problem
variables (total_fine : ℕ) (actual_speed : ℕ) (speed_limit : ℕ)
variable (fine_per_mph_over_limit : ℕ)

-- Theorem statement
theorem zeoland_speeding_fine :
  total_fine = 256 ∧ speed_limit = 50 ∧ actual_speed = 66 →
  fine_per_mph total_fine actual_speed speed_limit = 16 :=
by
  sorry

end zeoland_speeding_fine_l582_582886


namespace proposition_contradiction_l582_582436

-- Define the proposition P for natural numbers.
def P (n : ℕ+) : Prop := sorry

theorem proposition_contradiction (h1 : ∀ k : ℕ+, P k → P (k + 1)) (h2 : ¬ P 5) : ¬ P 4 :=
by
  sorry

end proposition_contradiction_l582_582436


namespace equal_sums_products_l582_582798

noncomputable def distinct_ints (a b c d : ℤ) : Prop :=
  (abs a > 10^6) ∧ (abs b > 10^6) ∧ (abs c > 10^6) ∧ (abs d > 10^6) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧
  (∀ n > 1, ¬ (n ∣ a ∧ n ∣ b ∧ n ∣ c ∧ n ∣ d))

theorem equal_sums_products (a b c d : ℤ) (h : distinct_ints a b c d) :
  ∃ (p : (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ)),
  let (sums1, sums2, sums3) :=
    ((a + b) * (a + c), (a + d) * (b + c), (b + d) * (c + d)) in
  sums1 = sums2 ∧ sums2 = sums3 := sorry

end equal_sums_products_l582_582798


namespace trig_identity_l582_582083

theorem trig_identity :
  (sin (15 * Real.pi / 180) * cos (15 * Real.pi / 180) + cos (165 * Real.pi / 180) * cos (105 * Real.pi / 180)) /
  (sin (19 * Real.pi / 180) * cos (11 * Real.pi / 180) + cos (161 * Real.pi / 180) * cos (101 * Real.pi / 180)) = 1 :=
by
  sorry

end trig_identity_l582_582083


namespace sum_of_gp_l582_582012

theorem sum_of_gp (d : ℝ) 
  (h1 : (8 + d)^2 = 5 * (35 + 2d))
  (h2 : ∃ d : ℝ, d = -3 + 20 * real.sqrt 6 ∨ d = -3 - 20 * real.sqrt 6) :
  5 + (5 + d + 3) + (35 + 2d) = 39 + 30 * real.sqrt 6 :=
by
  sorry

end sum_of_gp_l582_582012


namespace smallest_positive_period_of_f_find_a_and_b_l582_582138

-- Define the function f
def f (x : ℝ) : ℝ :=
  (√2) * sin (x / 2) * cos (x / 2) - (√2) * (sin (x / 2))^2

-- Statement for Problem 1
theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * π :=
sorry

-- Statement for Problem 2
theorem find_a_and_b
  (A B C : ℝ)
  (a b c : ℝ)
  (c_value : c = √3)
  (f_C_zero : f C = 0)
  (collinear_vectors : let m := (1, sin A); let n := (√2, sin B); ∃ k, m = k • n) :
  a = √3 * sin A ∧ b = √6 * sin A :=
sorry

end smallest_positive_period_of_f_find_a_and_b_l582_582138


namespace cost_saving_solution_l582_582880

theorem cost_saving_solution 
  (x m : ℕ)
  (h1 : m < x)
  (cost_function : ℕ → ℕ := λ (n : ℕ), 22 * n + 800) :
  (x = 11) → (m = 9) → cost_function x = 1042 :=
by
  intros
  sorry

end cost_saving_solution_l582_582880


namespace sum_of_squares_of_roots_cubic_eq_l582_582037

theorem sum_of_squares_of_roots_cubic_eq :
  let r := [r, s, t] -- roots of 2y^3 - 7y^2 + 8y + 5 = 0
  in (r^2 + s^2 + t^2 = 17 / 4) :=
by
  let q := Polynomial.cubic 2 (-7) 8 5 in  
  let roots := Polynomial.roots q in
  let sum_of_squares := roots.map (λ x, x^2).sum in
  have h: roots.card = 3 := by sorry
  have hs: (roots.map id).sum = 7 / 2 := by sorry
  have hp: (roots.map (λ R, R * roots)).sum = 4 := by sorry
  have heq: sum_of_squares = 17 / 4 := by
    calc (∑ x in roots, x^2)
      = (∑ x in roots, x)^2 - 2 * ∑ x in roots.product₂ 𝕜 roots x.1 x.2
      = by sorry
    sorry
  exact heq

end sum_of_squares_of_roots_cubic_eq_l582_582037


namespace probability_heart_king_l582_582648

theorem probability_heart_king :
  let total_cards := 52
  let total_kings := 4
  let hearts_count := 13
  let king_of_hearts := 1 in
  let prob_king_of_hearts_first := (1 : ℚ) / total_cards
  let prob_other_heart_first := (hearts_count - king_of_hearts : ℚ) / total_cards
  let prob_king_second_if_king_heart_first := (total_kings - king_of_hearts : ℚ) / (total_cards - 1)
  let prob_king_second_if_other_heart_first := (total_kings : ℚ) / (total_cards - 1) in
  prob_king_of_hearts_first * prob_king_second_if_king_heart_first +
  prob_other_heart_first * prob_king_second_if_other_heart_first = (1 : ℚ) / total_cards :=
by sorry

end probability_heart_king_l582_582648


namespace alok_mixed_vegetable_plates_l582_582023

theorem alok_mixed_vegetable_plates 
  (chapati_count : ℕ) (rice_count : ℕ) (mixed_vegetable_price : ℝ)
  (chapati_price : ℝ) (rice_price : ℝ) (total_paid : ℝ) (final_cost : ℝ)
  (total: ℝ): chapati_count = 16 ∧ rice_count = 5 ∧ chapati_price = 6 ∧ rice_price = 45 ∧ mixed_vegetable_price = 70 ∧ total_paid = 985 
  ∧ final_cost = total_paid - (chapati_count * chapati_price + rice_count * rice_price) 
  ∧ total = final_cost / mixed_vegetable_price → total.floor = 9 := 
by 
  intros h
  obtain ⟨hc, hr, hcp, hrp, hmvp, htp, hfc, ht⟩ := h
  sorry 

end alok_mixed_vegetable_plates_l582_582023


namespace monotonicity_a_eq_1_range_of_a_inequality_proof_l582_582475

-- Part 1: Monotonicity when a = 1
def f (x : ℝ) : ℝ := x * Real.exp(x) - Real.exp(x)

theorem monotonicity_a_eq_1 :
  (∀ x : ℝ, 0 < x → (0 < deriv f x)) ∧ (∀ x : ℝ, x < 0 → (deriv f x < 0)) :=
by
  sorry

-- Part 2: Range of values for 'a' when f(x) < -1 and x > 0
def g (a x : ℝ) : ℝ := x * Real.exp(a * x) - Real.exp(x)

theorem range_of_a (a : ℝ) (x : ℝ) (h : 0 < x) (hf : g a x < -1) : a ≤ 1 / 2 :=
by
  sorry

-- Part 3: Inequality proof for natural numbers
def sum_ineq (n : ℕ) : ℝ := ∑ i in Finset.range n, 1 / Real.sqrt (↑i ^ 2 + ↑i)

theorem inequality_proof (n : ℕ) (h : 0 < n) : sum_ineq n > Real.log (n+1) :=
by
  sorry

end monotonicity_a_eq_1_range_of_a_inequality_proof_l582_582475


namespace pencil_cost_l582_582683

/-- Define the unit price based on the cost of a box of pencils -/
def unit_price (box_cost : ℕ) (box_quantity : ℕ) : ℚ :=
box_cost / box_quantity

/-- Define the cost of purchasing a number of pencils without discount -/
def bulk_order_cost (unit_price : ℚ) (quantity : ℕ) : ℚ :=
unit_price * quantity

/-- Define the discount amount if applicable -/
def discount (cost : ℚ) (quantity : ℕ) (threshold : ℕ) (discount_rate : ℚ) : ℚ :=
if quantity > threshold then cost * discount_rate else 0

/-- Define the final cost after applying discount -/
def final_cost (cost : ℚ) (discount : ℚ) : ℚ :=
cost - discount

/-- Given the conditions, prove the total cost to purchase 3000 pencils is $675 -/
theorem pencil_cost
  (box_cost : ℕ := 50)
  (box_quantity : ℕ := 200)
  (quantity : ℕ := 3000)
  (threshold : ℕ := 1000)
  (discount_rate : ℚ := 0.10) :
  let unit_price := unit_price box_cost box_quantity in
  let cost_before_discount := bulk_order_cost unit_price quantity in
  let applied_discount := discount cost_before_discount quantity threshold discount_rate in
  final_cost cost_before_discount applied_discount = 675 :=
by
  let unit_price := unit_price box_cost box_quantity
  let cost_before_discount := bulk_order_cost unit_price quantity
  let applied_discount := discount cost_before_discount quantity threshold discount_rate
  exact sorry

end pencil_cost_l582_582683


namespace floor_ceil_expression_l582_582417

theorem floor_ceil_expression :
  (Int.floor ∘ (λ x => x + ↑(19/5)) ∘ Int.ceil ∘ λ x => x^2) (15/8) = 7 := 
by 
  sorry

end floor_ceil_expression_l582_582417


namespace cyclic_quadrilateral_l582_582687

theorem cyclic_quadrilateral (
  (conv_quad_intersect_circle : set ℝ) 
  [is_convex_quad : convex conv_quad_intersect_circle]
  (A1 A2 B1 B2 C1 C2 D1 D2 : ℝ)
  (h1 : A1 ∈ conv_quad_intersect_circle)
  (h2 : A2 ∈ conv_quad_intersect_circle)
  (h3 : B1 ∈ conv_quad_intersect_circle)
  (h4 : B2 ∈ conv_quad_intersect_circle)
  (h5 : C1 ∈ conv_quad_intersect_circle)
  (h6 : C2 ∈ conv_quad_intersect_circle)
  (h7 : D1 ∈ conv_quad_intersect_circle)
  (h8 : D2 ∈ conv_quad_intersect_circle)
  (h_order : [A1, B2, B1, C2, C1, D2, D1, A2] in (cyclic_perm (conv_quad_intersect_circle.elements)))
  (h_length_eq : |A1 - B2| = |B1 - C2| ∧ |B1 - C2| = |C1 - D2| ∧ |C1 - D2| = |D1 - A2|)
  ) : 
  cyclic [A1, B2, B1, C2, C1, D2, D1, A2] :=
begin
  sorry,
end

end cyclic_quadrilateral_l582_582687


namespace problem_statement_l582_582833

def g (x : ℝ) : ℝ := 2 ^ x

def f (x : ℝ) : ℝ := (1 - g x) / (1 + g x)

theorem problem_statement :
  (∀ x y : ℝ, x < y → f x > f y) ∧
  (∃ m ∈ (0, 1/3], ∃ x ∈ [-1, 0), f x = m → f (1 / m) ≤ -7 / 9) :=
sorry

end problem_statement_l582_582833


namespace determine_radius_l582_582348

variable (R r : ℝ)

theorem determine_radius (h1 : R = 10) (h2 : π * R^2 = 2 * (π * R^2 - π * r^2)) : r = 5 * Real.sqrt 2 :=
  sorry

end determine_radius_l582_582348


namespace exists_isosceles_right_triangle_l582_582335

-- Definitions for the given conditions
def is_isosceles_right_triangle (A B C : Point) : Prop :=
  let ab : ℚ := (B - A).norm in
  let ac : ℚ := (C - A).norm in
  let bc : ℚ := (C - B).norm in
  ab = ac ∧ ab^2 + ac^2 = bc^2

variable {A B C D E : Point}
variable {EC : LineSegment E C}

-- Theorem statement for the problem
theorem exists_isosceles_right_triangle (hABC : is_isosceles_right_triangle A B C)
    (hADE : is_isosceles_right_triangle A D E)
    (h_noncongruence : ¬ congruence A B C A D E) :
  ∃ M ∈ EC, is_isosceles_right_triangle B M D :=
by
  sorry

end exists_isosceles_right_triangle_l582_582335


namespace submersion_depth_l582_582680

-- Definitions based on the given conditions
def volume_cone : ℝ := 350 -- cm³
def specific_gravity_iron : ℝ := 7.2 -- g/cm³
def specific_gravity_mercury : ℝ := 13.6 -- g/cm³
def pi_approx : ℝ := 3.14259

-- The target theorem
theorem submersion_depth (h : ℝ) (r : ℝ) (h_eq_r : h = r) (x : ℝ) :
  (1/3) * pi_approx * h^3 = volume_cone →
  (specific_gravity_iron * volume_cone = specific_gravity_mercury * ((1/3) * pi_approx * x^3)) →
  x ≈ 5.6141 :=
begin
  sorry
end

end submersion_depth_l582_582680


namespace log_equation_l582_582102

theorem log_equation (b : ℝ) (a c d : ℝ) (hb : 0 < b) (h1 : log 3 b = a) (h2 : log 6 b = c) (h3 : 3 ^ d = 6) : a = c * d := 
sorry

end log_equation_l582_582102


namespace find_c_eight_unit_squares_l582_582774

theorem find_c_eight_unit_squares :
  let line_eq (c : ℝ) := λ x : ℝ, (4 / (4 - c)) * (x - c)
  in ∃ c : ℝ, (1 / 2) * (4 - c) * 4 = (8 / 3) ∧ c = 8 / 9 := sorry

end find_c_eight_unit_squares_l582_582774


namespace surface_area_solid_prism_l582_582014

noncomputable def surface_area_CXYZ' (height : ℝ) (a b c : ℝ) : ℝ :=
  let half_a := a / 2
  let half_height := height / 2
  let half_base := c / 2
  let area_CZ'X' := (1 / 2) * half_a * half_height
  let area_CZ'Y' := (1 / 2) * half_a * half_height
  let CM := real.sqrt (half_a ^ 2 - (half_base / 2) ^ 2)
  let area_CX'Y' := (1 / 2) * half_base * CM
  let Z'M := real.sqrt ((half_height ^ 2 + half_a ^ 2) - (half_base / 2) ^ 2)
  let area_X'Y'Z' := (1 / 2) * half_base * Z'M
  area_CZ'X' + area_CZ'Y' + area_CX'Y' + area_X'Y'Z'

theorem surface_area_solid_prism :
  surface_area_CXYZ' 20 12 18 = 126.285 :=
by
  sorry

end surface_area_solid_prism_l582_582014


namespace tan_arccot_eq_l582_582733

theorem tan_arccot_eq : Real.tan (Real.arccot (4 / 7)) = 7 / 4 := by
  sorry

end tan_arccot_eq_l582_582733


namespace Joey_weekend_study_hours_l582_582909

noncomputable def hours_weekday_per_week := 2 * 5 -- 2 hours/night * 5 nights/week
noncomputable def total_hours_weekdays := hours_weekday_per_week * 6 -- Multiply by 6 weeks
noncomputable def remaining_hours_weekends := 96 - total_hours_weekdays -- 96 total hours - weekday hours
noncomputable def total_weekend_days := 6 * 2 -- 6 weekends * 2 days/weekend
noncomputable def hours_per_day_weekend := remaining_hours_weekends / total_weekend_days

theorem Joey_weekend_study_hours : hours_per_day_weekend = 3 :=
by
  sorry

end Joey_weekend_study_hours_l582_582909


namespace factorization_l582_582072

theorem factorization (a : ℝ) : 2 * a ^ 2 - 8 = 2 * (a + 2) * (a - 2) := 
by
  sorry

end factorization_l582_582072


namespace derivative_chain_rule_l582_582788

noncomputable def dx_dt (t : ℝ) : ℝ := 
  (Real.ln (Real.cot t))' t

noncomputable def dy_dt (t : ℝ) : ℝ := 
  (1 / (Real.cos t) ^ 2)' t

theorem derivative_chain_rule (t : ℝ) :
  let dx_dt := (Real.ln (Real.cot t))' t,
      dy_dt := (1 / (Real.cos t) ^ 2)' t in
  (dy_dt / dx_dt) = -2 * (Real.tan t) ^ 2 :=
by
  sorry

end derivative_chain_rule_l582_582788


namespace floor_sum_eq_126_l582_582923

-- Define the problem conditions
variable (a b c d : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
variable (h5 : a^2 + b^2 = 2008) (h6 : c^2 + d^2 = 2008)
variable (h7 : a * c = 1000) (h8 : b * d = 1000)

-- Prove the solution
theorem floor_sum_eq_126 : ⌊a + b + c + d⌋ = 126 :=
by
  sorry

end floor_sum_eq_126_l582_582923


namespace pieces_of_trash_outside_classrooms_l582_582635

theorem pieces_of_trash_outside_classrooms (total_trash : ℕ) (classroom_trash : ℕ) (h_total : total_trash = 1576) (h_classroom : classroom_trash = 344) :
  (total_trash - classroom_trash) = 1232 :=
by
  rw [h_total, h_classroom]
  exact rfl

end pieces_of_trash_outside_classrooms_l582_582635


namespace no_fractional_solution_l582_582767

theorem no_fractional_solution (x y : ℚ)
  (h₁ : ∃ m : ℤ, 13 * x + 4 * y = m)
  (h₂ : ∃ n : ℤ, 10 * x + 3 * y = n) :
  (∃ a b : ℤ, x ≠ a ∧ y ≠ b) → false :=
by {
  sorry
}

end no_fractional_solution_l582_582767


namespace arithmetic_seq_sum_equality_l582_582815

theorem arithmetic_seq_sum_equality (a : ℕ → ℝ) (d : ℝ) (h_d : d = -2) 
(h_sum : (finset.range 34).sum (λ k, a (1 + 3 * k)) = 50) :
(finset.range 34).sum (λ k, a (3 + 3 * k)) = -82 := 
sorry

end arithmetic_seq_sum_equality_l582_582815


namespace sum_of_powers_i_l582_582558

-- Conditions
def i : ℂ := Complex.I

-- The sum we need to prove
theorem sum_of_powers_i : (∑ n in Finset.range 2015, i ^ n) = i := by
  sorry

end sum_of_powers_i_l582_582558


namespace arc_length_polar_curve_l582_582032

/-- The length of the arc of the curve given by ρ = 2φ for 0 ≤ φ ≤ 4/3 equals 20/9 + ln(3). -/
theorem arc_length_polar_curve : 
  let ρ (φ : ℝ) := 2 * φ in
  ∫ φ in 0..(4 / 3), sqrt ((ρ φ) ^ 2 + (deriv ρ φ) ^ 2) = 20 / 9 + Real.log 3 := by
  sorry

end arc_length_polar_curve_l582_582032


namespace circular_sequence_exists_l582_582448

/-- Given 100 non-negative real numbers whose sum is 1, 
prove that these numbers can be arranged in a circular sequence 
such that the sum of the products of every two adjacent numbers 
does not exceed 0.01. --/
theorem circular_sequence_exists (x : Fin 100 → ℝ)
  (h_nonneg : ∀ i, 0 ≤ x i)
  (h_sum : ∑ i, x i = 1 ) :
  ∃ σ : Fin 100 → Fin 100, 
  (∑ i : Fin 100, x (σ i) * x (σ ((i + 1) % 100))) ≤ 0.01 := sorry

end circular_sequence_exists_l582_582448


namespace problem_l582_582992

noncomputable def f : ℝ → ℝ := sorry

theorem problem 
  (domain : ℝ → ℝ)
  (symm_y : ∀ x : ℝ, f(x) = f(-x))
  (symm_point : ∀ x : ℝ, f(x) = -f(2 - x))
  : ∀ x : ℝ, f(x + 4) = f(x) :=
sorry

end problem_l582_582992


namespace find_g_neg2_l582_582465

-- Definitions of the conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x 

variables (f : ℝ → ℝ) (g : ℝ → ℝ)
variables (h_even_f : even_function f)
variables (h_g_def : ∀ x, g x = f x + x^3)
variables (h_g_2 : g 2 = 10)

-- Statement to prove
theorem find_g_neg2 : g (-2) = -6 :=
sorry

end find_g_neg2_l582_582465


namespace three_digit_numbers_divisible_by_17_l582_582494

theorem three_digit_numbers_divisible_by_17 :
  ∃ n : ℕ, n = (999 \div 17) - (100 \div 17) + 1 ∧ n = 53 := by
  sorry

end three_digit_numbers_divisible_by_17_l582_582494


namespace sum_place_values_of_specified_digits_l582_582316

def numeral := 95378637153370261

def place_values_of_3s := [3 * 100000000000, 3 * 10]
def place_values_of_7s := [7 * 10000000000, 7 * 1000000, 7 * 100]
def place_values_of_5s := [5 * 10000000000000, 5 * 1000, 5 * 10000, 5 * 1]

def sum_place_values (lst : List ℕ) : ℕ :=
  lst.foldl (· + ·) 0

def sum_of_place_values := 
  sum_place_values place_values_of_3s + 
  sum_place_values place_values_of_7s + 
  sum_place_values place_values_of_5s

theorem sum_place_values_of_specified_digits :
  sum_of_place_values = 350077055735 :=
by
  sorry

end sum_place_values_of_specified_digits_l582_582316


namespace solution_valid_1380456_l582_582426

theorem solution_valid_1380456 :
  (∃ x y z : ℕ, x = 8 ∧ y = 0 ∧ z = 6 ∧
    (450 + z) % 8 = 0 ∧
    (19 + x + y) % 9 = 0 ∧
    (x - y + 3) % 11 = 0
    ∧ x < 10 ∧ y < 10 ∧ z < 10) :=
by
  use 8
  use 0
  use 6
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end solution_valid_1380456_l582_582426


namespace tips_fraction_l582_582372

-- Define the conditions
variables (S T : ℝ) (h : T = (2 / 4) * S)

-- The statement to be proved
theorem tips_fraction : (T / (S + T)) = 1 / 3 :=
by
  sorry

end tips_fraction_l582_582372


namespace original_perimeter_of_rectangle_l582_582022

theorem original_perimeter_of_rectangle
  (a b : ℝ)
  (h : (a + 3) * (b + 3) - a * b = 90) :
  2 * (a + b) = 54 :=
sorry

end original_perimeter_of_rectangle_l582_582022


namespace conference_min_duration_l582_582350

theorem conference_min_duration : Nat.gcd 9 11 = 1 ∧ Nat.gcd 9 12 = 3 ∧ Nat.gcd 11 12 = 1 ∧ Nat.lcm 9 (Nat.lcm 11 12) = 396 := by
  sorry

end conference_min_duration_l582_582350


namespace intersection_complement_eq_l582_582148

open Set

variable (U A B : Set ℕ)

def U := {1, 2, 3, 4, 5, 6, 7}
def A := {2, 3, 4, 5}
def B := {2, 3, 6, 7}

theorem intersection_complement_eq : B ∩ (U \ A) = {6, 7} := by
  sorry

end intersection_complement_eq_l582_582148


namespace planes_divide_space_l582_582895

theorem planes_divide_space (n : ℕ) (h1 : ∀ (p1 p2 : set ℝ^3), p1 ≠ p2 → ∃ l : set ℝ^3, l ∈ p1 ∩ p2) 
(h2 : ∀ (p1 p2 p3 : set ℝ^3), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ∃ pt : ℝ^3, pt ∈ p1 ∩ p2 ∩ p3) 
(h3 : ∀ (p1 p2 p3 p4 : set ℝ^3), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p4 → 
¬ ( (∃ pt : ℝ^3, pt ∈ p1 ∩ p2 ∩ p3 ∩ p4) ) ) :
  (∑ k in (finset.range n).succ, 1 + k + (k * (k - 1)) / 2 + (k * (k - 1) * (k - 2)) / 6) = (n ^ 3 + 5 * n + 6) / 6 := 
by
  sorry

end planes_divide_space_l582_582895


namespace value_of_b2023_l582_582704

-- Define the sequence recursively
def seq (n : ℕ) : ℚ
| 1     := 2
| 2     := 6 / 13
| (n+1) := (seq (n-1) * seq n) / (3 * seq (n-1) - seq n) when n ≥ 2

-- Prove that the value of b_2023 is 6/12137
theorem value_of_b2023 : ∃ p q : ℕ, Nat.coprime p q ∧ seq 2023 = p / q ∧ p + q = 12143 := by
  sorry

end value_of_b2023_l582_582704


namespace find_t_l582_582530

open BigOperators

-- Definition of a geometric sequence
def is_geometric_sequence {R : Type*} [Field R] (a : ℕ → R) (q : R) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Sum of the first n terms of a sequence
def sum_first_n {R : Type*} [CommRing R] (a : ℕ → R) (n : ℕ) : R :=
  ∑ k in Finset.range n, a k

-- Given conditions:
variable {R : Type*} [CommRing R] [Field R]
variables (a : ℕ → R) (n : ℕ) (q : R)
variable (t : R)

axiom h_a1 : a 1 = 1
axiom h_a4 : a 4 = 8
axiom h_geometric : is_geometric_sequence a q

-- S3n and Tn
def S3n := sum_first_n a (3 * n)
def Tn := sum_first_n (λ k, (a k)^3) n

-- Problem statement: Prove that t = 7 given the conditions
theorem find_t : S3n = t * Tn → t = 7 := by
  sorry

end find_t_l582_582530


namespace urn_contains_seven_red_five_blue_l582_582381

noncomputable def urn_probability : ℚ := sorry

theorem urn_contains_seven_red_five_blue
  (initial_red : ℕ) (initial_blue : ℕ)
  (operations : ℕ)
  (final_red : ℕ) (final_blue : ℕ)
  (draw_add_red_prob : list ℚ) (draw_add_blue_prob : list ℚ) :
  initial_red = 2 →
  initial_blue = 1 →
  operations = 5 →
  final_red = 7 →
  final_blue = 5 →
  urn_probability = 25 / 224 := 
sorry

end urn_contains_seven_red_five_blue_l582_582381


namespace price_of_mixture_l582_582193

theorem price_of_mixture :
  (1 * 64 + 1 * 74) / (1 + 1) = 69 :=
by
  sorry

end price_of_mixture_l582_582193


namespace minimum_value_l582_582113

variable (m n x y : ℝ)

theorem minimum_value (h1 : m^2 + n^2 = 1) (h2 : x^2 + y^2 = 4) : 
  ∃ (min_val : ℝ), min_val = -2 ∧ ∀ (my_nx : ℝ), my_nx = my + nx → my_nx ≥ min_val :=
by
  sorry

end minimum_value_l582_582113


namespace min_k_l582_582920

def a_n (n : ℕ) : ℕ :=
  n

def b_n (n : ℕ) : ℚ :=
  a_n n / 3^n

def T_n (n : ℕ) : ℚ :=
  (List.range n).foldl (λ acc i => acc + b_n (i + 1)) 0

theorem min_k (k : ℕ) (h : ∀ n : ℕ, n ≥ k → |T_n n - 3/4| < 1/(4*n)) : k = 4 :=
  sorry

end min_k_l582_582920


namespace line_intersects_circle_shortest_chord_length_l582_582836

noncomputable def circle : set (ℝ × ℝ) :=
  { p | (p.1 - 3)^2 + (p.2 - 4)^2 = 4 }

noncomputable def line (k : ℝ) : set (ℝ × ℝ) :=
  { p | k * p.1 - p.2 - 4 * k + 3 = 0 }

/- Prove that no matter what value k takes, the line l always intersects the circle at two distinct points. -/
theorem line_intersects_circle (k : ℝ) : ∃! (p1 p2 : (ℝ × ℝ)), p1 ≠ p2 ∧ p1 ∈ circle ∧ p2 ∈ circle ∧ p1 ∈ line k ∧ p2 ∈ line k :=
by 
  sorry

/- Find the value of k that makes the chord intercepted by the line l on the circle the shortest, and calculate the length of this shortest chord. -/
theorem shortest_chord_length : ∃ k : ℝ, (∀ (k' : ℝ), k' ≠ k → chord_length (line k') > chord_length (line k)) ∧ k = 1 ∧ chord_length (line k) = 2 * sqrt 2 :=
by
  sorry

noncomputable def chord_length (l : set (ℝ × ℝ)) : ℝ :=
  let points := { p | p ∈ l ∧ p ∈ circle} in
  let ps := finset.filter (λ (p : ℝ × ℝ), true) points.to_finset in
  match finset.image (λ (p : ℝ × ℝ), p.1) ps with
  | ⟨[p1, p2], _⟩ => dist p1 p2 
  | _ => 0 -- fallback case; shouldn't happen under correct conditions

end line_intersects_circle_shortest_chord_length_l582_582836


namespace qt_q_t_not_prime_l582_582326

theorem qt_q_t_not_prime (q t : ℕ) (hq: q > 1) (ht: t > 1): ¬ nat.prime (q * t + q + t) :=
by 
  sorry

end qt_q_t_not_prime_l582_582326


namespace seasons_before_announcement_l582_582549

theorem seasons_before_announcement (S : ℕ) (episodes_per_season : ℕ) (extra_episodes_last_season : ℕ) 
  (episode_length : ℕ) (total_watch_time_hours : ℕ) (total_episodes : ℕ) (episodes_last_season: ℕ) (S_eq: 22S = 198):
  episodes_per_season = 22 ∧ extra_episodes_last_season = 4 ∧ episode_length = 1 ∧ total_watch_time_hours = 112 ∧
  total_episodes = 224 → S = 9 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  cases h6 with h7 h8
  cases h8 with h9 h10
  rw h10 at *
  rw h7 at *
  rw h9 at *
  have z := congr_arg (λ x, x) S_eq
  assumption
  sorry

end seasons_before_announcement_l582_582549


namespace largest_n_for_triangle_property_l582_582740

def has_triangle_property (S : Set ℕ) : Prop :=
  ∀ {a b c : ℕ}, a ∈ S → b ∈ S → c ∈ S → a + b > c ∧ b + c > a ∧ c + a > b

theorem largest_n_for_triangle_property : 
  ∀ (n : ℕ), n < 364 ↔ ∀ (S : Set ℕ), (∀ (x : ℕ), 6 ≤ x → x ≤ n → x ∈ S) → (∀ T : Set ℕ, T ⊆ S → T.card = 10 → has_triangle_property T) :=
begin
  sorry
end

end largest_n_for_triangle_property_l582_582740


namespace find_x2_times_x1_plus_x3_l582_582215

noncomputable def a : ℝ := Real.sqrt 2023

def poly (x : ℝ) : ℝ := a * x^3 - 4047 * x^2 + 4046 * x - 1

def is_root (x : ℝ) : Prop := poly x = 0

variable (x1 x2 x3 : ℝ)
variable h_order : x1 < x2 ∧ x2 < x3
variable h_roots : is_root x1 ∧ is_root x2 ∧ is_root x3

theorem find_x2_times_x1_plus_x3 : x2 * (x1 + x3) = 2 + 1 / 2023 :=
by
  have h_x2_eq_inv_a : x2 = 1 / a := sorry
  have h_sum_x1_x3 : x1 + x3 = 2 * a + 1 / a := sorry
  rw [h_x2_eq_inv_a, h_sum_x1_x3]
  calc
    (1 / a) * (2 * a + 1 / a)
      = (1 / a) * (2 * a) + (1 / a) * (1 / a) : by rw [mul_add]
  ... = 2 + 1 / (a * a) : by { field_simp [a_ne_zero], ring }
  ... = 2 + 1 / 2023 : by rw [←Real.mul_self_sqrt zero_le_of_real_pos]

end find_x2_times_x1_plus_x3_l582_582215


namespace find_b_l582_582284

noncomputable def f (x : ℝ) : ℝ := (x+1)^3 + (x / (x + 1))

theorem find_b (b : ℝ) (h_sum : ∃ x1 x2 : ℝ, f x1 = -x1 + b ∧ f x2 = -x2 + b ∧ x1 + x2 = -2) : b = 0 :=
by
  sorry

end find_b_l582_582284


namespace probability_of_B_win_best_of_three_l582_582514

noncomputable def probA : ℝ := 0.6
noncomputable def probB : ℝ := 0.4

/-- The probability of player B winning a best-of-three billiards match -/
theorem probability_of_B_win_best_of_three :
  let P_BB := probB * probB in
  let P_BLB := probB * probA * probB in
  let P_LBB := probA * probB * probB in
  P_BB + P_BLB + P_LBB = 0.352 :=
by
  let P_BB := probB * probB
  let P_BLB := probB * probA * probB
  let P_LBB := probA * probB * probB
  have h : P_BB + P_BLB + P_LBB = 0.352, from sorry
  exact h

end probability_of_B_win_best_of_three_l582_582514


namespace find_f_log2_1_div_24_l582_582932

-- Define the conditions under which we will prove the statement
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f (x)

def f_defined_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x < 1 → f(x) = 2^x - 1

-- The main statement to prove
theorem find_f_log2_1_div_24 (f : ℝ → ℝ) 
  (h_odd: is_odd_function f) 
  (h_periodic: is_periodic f 2) 
  (h_def: f_defined_on_interval f) :
  f (Real.log2 (1 / 24)) = -1/2 :=
by
  sorry

end find_f_log2_1_div_24_l582_582932


namespace sequence_problem_l582_582456

noncomputable def a (n : ℕ) : ℕ

axiom seq_recurrence (n : ℕ) : a (n + 1) = 2 * a n
axiom initial_condition : a 1 + a 4 = 2

theorem sequence_problem : a 5 + a 8 = 32 := 
by
  sorry

end sequence_problem_l582_582456


namespace triangle_third_side_l582_582019

theorem triangle_third_side (x : ℕ) : 
  (3 < x) ∧ (x < 17) → 
  (x = 11) :=
by
  sorry

end triangle_third_side_l582_582019


namespace final_number_independent_of_operations_l582_582239

theorem final_number_independent_of_operations (p q r : ℕ) : 
  ∃ final_num : ℕ, ∀ seq : list (ℕ × ℕ × ℕ), 
  (final_num = 0 ∧ (p % 2 = 1) ∨
   final_num = 1 ∧ (q % 2 = 1) ∨
   final_num = 2 ∧ (r % 2 = 1)) :=
sorry

end final_number_independent_of_operations_l582_582239


namespace isosceles_triangle_base_length_l582_582276

theorem isosceles_triangle_base_length (a b : ℝ) (h1 : a = 3 ∨ b = 3) (h2 : a + a + b = 15 ∨ a + b + b = 15) :
  b = 3 := 
sorry

end isosceles_triangle_base_length_l582_582276


namespace largest_shaded_area_figure_C_l582_582994

noncomputable def area_of_square (s : ℝ) : ℝ := s^2
noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def shaded_area_of_figure_A : ℝ := 4 - Real.pi
noncomputable def shaded_area_of_figure_B : ℝ := 4 - Real.pi
noncomputable def shaded_area_of_figure_C : ℝ := Real.pi - 2

theorem largest_shaded_area_figure_C : shaded_area_of_figure_C > shaded_area_of_figure_A ∧ shaded_area_of_figure_C > shaded_area_of_figure_B := by
  sorry

end largest_shaded_area_figure_C_l582_582994


namespace probability_first_card_heart_second_king_l582_582647

theorem probability_first_card_heart_second_king :
  ∀ (deck : Finset ℕ) (is_heart : ℕ → Prop) (is_king : ℕ → Prop),
  deck.card = 52 →
  (∀ card ∈ deck, is_heart card ∨ ¬ is_heart card) →
  (∀ card ∈ deck, is_king card ∨ ¬ is_king card) →
  (∃ p : ℚ, p = 1/52) :=
by
  intros deck is_heart is_king h_card h_heart h_king,
  sorry

end probability_first_card_heart_second_king_l582_582647


namespace prove_t_eq_9_l582_582062

variable {a p v m t r : ℕ}

-- Conditions as definitions
def cond1 : Prop := a + p = v
def cond2 : Prop := v + m = t
def cond3 : Prop := t + a = r
def cond4 : Prop := p + m + r = 18

-- The goal is to prove t = 9 given the conditions
theorem prove_t_eq_9 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : t = 9 :=
by
  sorry

end prove_t_eq_9_l582_582062


namespace sin_sum_eq_l582_582428

theorem sin_sum_eq :
  (∑ k in (finset.Ico 45 134), 1 / (Real.sin (k : ℝ) * Real.sin (k + 1))) = 1 / (Real.sin 1) := 
  sorry

end sin_sum_eq_l582_582428


namespace taxi_fare_proportional_l582_582374

theorem taxi_fare_proportional (cost_per_km : ℕ → ℕ) (H : ∀ d₁ d₂ : ℕ, cost_per_km (d₁ + d₂) = cost_per_km d₁ + cost_per_km d₂) :
  cost_per_km 80 = 160 → cost_per_km 100 = 200 :=
begin
  sorry
end

end taxi_fare_proportional_l582_582374


namespace find_f1_plus_g1_l582_582825

-- Definition of f being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

-- Definition of g being an odd function
def is_odd_function (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = -g x

-- Statement of the proof problem
theorem find_f1_plus_g1 
  (f g : ℝ → ℝ) 
  (hf : is_even_function f) 
  (hg : is_odd_function g)
  (hfg : ∀ x : ℝ, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 2 :=
sorry

end find_f1_plus_g1_l582_582825


namespace sum_of_squares_lt_l582_582237

theorem sum_of_squares_lt {n : ℕ} (h : 2 ≤ n) :
  (1 + ∑ i in Finset.range n, (1 / (i + 2)^2 : ℝ)) < (2 * n - 1) / n := 
sorry

end sum_of_squares_lt_l582_582237


namespace division_by_fraction_l582_582392

theorem division_by_fraction : 5 / (1 / 5) = 25 := by
  sorry

end division_by_fraction_l582_582392


namespace intersection_x_coordinate_l582_582264

theorem intersection_x_coordinate (k b : ℝ) (h : k ≠ b) :
  (∃ x y : ℝ, y = k * x + b ∧ y = b * x + k) → (∃ x : ℝ, x = 1) :=
by
  intro h_intersect
  cases h_intersect with x h_xy
  cases h_xy with y h_1
  use 1
  sorry

end intersection_x_coordinate_l582_582264


namespace sheep_per_herd_l582_582749

theorem sheep_per_herd (herds : ℕ) (total_sheep : ℕ) (h_herds : herds = 3) (h_total_sheep : total_sheep = 60) : 
  (total_sheep / herds) = 20 :=
by
  sorry

end sheep_per_herd_l582_582749


namespace max_S_value_l582_582435

def f (t : ℕ) : ℕ :=
  -t + 30

def g (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 10 then 2 * t + 40
  else if 11 ≤ t ∧ t ≤ 20 then 15
  else 0 -- Assuming the function is zero outside the defined intervals

def S (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 10 then (-t + 30) * (2 * t + 40)
  else if 11 ≤ t ∧ t ≤ 20 then 15 * (-t + 30)
  else 0 -- Assuming the function is zero outside the defined intervals

theorem max_S_value : 
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 20 ∧ S t = 1250) :=
begin
  use 5,
  split,
  { exact nat.le_of_lt 5,
    exact nat.succ_le_of_lt 5 },
  { split, exact S(5) = 1250 as required,
    sorry, -- Proof steps for the calculation and comparison of max value
  }
end

end max_S_value_l582_582435


namespace locus_of_point_P_l582_582108

noncomputable def point_P_locus : Set (ℂ) :=
  { zP : ℂ | ∃ (zD : ℂ), abs(zD) = 3 ∧ zP = (4 - 3 * complex.I) / 3 + (2 / 3) * zD }

theorem locus_of_point_P : 
  ∃ (center : ℂ) (radius : ℝ), 
    center = (8 / 3) - 2 * complex.I ∧ 
    radius = 2 ∧ 
    point_P_locus = { zP : ℂ | abs(zP - center) = radius } :=
by
  sorry

end locus_of_point_P_l582_582108


namespace total_water_carried_l582_582689

noncomputable theory

def num_trucks : ℕ := 3
def tanks_per_truck : ℕ := 3
def liters_per_tank : ℕ := 150

theorem total_water_carried : num_trucks * (tanks_per_truck * liters_per_ttank) = 1350 := 
   sorry

end total_water_carried_l582_582689


namespace loisa_saves_70_l582_582229

def tablet_cash_price : ℕ := 450
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 40
def next_4_months_payment : ℕ := 35
def last_4_months_payment : ℕ := 30
def total_installment_payment : ℕ := down_payment + (4 * first_4_months_payment) + (4 * next_4_months_payment) + (4 * last_4_months_payment)
def savings : ℕ := total_installment_payment - tablet_cash_price

theorem loisa_saves_70 : savings = 70 := by
  sorry

end loisa_saves_70_l582_582229


namespace find_k_l582_582210

variables (a b c : ℝ^3) (k : ℝ)
variable [inner_product_space ℝ ℝ^3]

-- Conditions
hypothesis (h1 : ∥a∥ = 1)
hypothesis (h2 : ∥b∥ = 1)
hypothesis (h3 : ∥c∥ = 1)
hypothesis (h4 : ⟪a, b⟫ = 0)
hypothesis (h5 : ⟪a, c⟫ = 0)
hypothesis (h6 : real.angle b c = real.pi / 3)

-- Equivalent proof problem
theorem find_k : a = k • (b ⊗ c) ↔ k = 2*sqrt 3/3 ∨ k = -2*sqrt 3/3 :=
sorry

end find_k_l582_582210


namespace last_digit_of_a2009_div_a2006_is_6_l582_582750
open Nat

def ratio_difference_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 2) * a n = (a (n + 1)) ^ 2 + d * a (n + 1)

theorem last_digit_of_a2009_div_a2006_is_6
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = 2)
  (d : ℕ)
  (h4 : ratio_difference_sequence a d) :
  (a 2009 / a 2006) % 10 = 6 :=
by
  sorry

end last_digit_of_a2009_div_a2006_is_6_l582_582750


namespace michael_twenty_dollar_bills_l582_582232

/--
Michael has $280 dollars and each bill is $20 dollars.
We need to prove that the number of $20 dollar bills Michael has is 14.
-/
theorem michael_twenty_dollar_bills (total_money : ℕ) (bill_denomination : ℕ) (number_of_bills : ℕ) :
  total_money = 280 →
  bill_denomination = 20 →
  number_of_bills = total_money / bill_denomination →
  number_of_bills = 14 :=
by
  intros h1 h2 h3
  sorry

end michael_twenty_dollar_bills_l582_582232


namespace range_of_a_l582_582130

theorem range_of_a (a : ℝ) (h : a > 0) :
  let circle_eq : (ℝ × ℝ) → Prop := λ p, (p.1^2 + p.2^2 - 2*a*p.1 - 2*a*p.2 = 0)
  let A : ℝ × ℝ := (0, 2)
  (∀ (T : ℝ × ℝ), circle_eq T → ∃ (M : ℝ × ℝ), ∃ θ : ℝ, θ = 45 ∧ M = (a, a) ∧ (angle M A T) = θ) →
  a ∈ set.Ico (Real.sqrt 3 - 1) 1 := by
  sorry -- Proof goes here

end range_of_a_l582_582130


namespace sequence_all_two_l582_582781

theorem sequence_all_two {x : ℕ → ℕ} (h0 : x 0 = x 20) (h1 : x 21 = x 1) (h2 : x 22 = x 2)
  (h : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 20 → (x (i+2))^2 = Nat.lcm (x (i+1)) (x i) + Nat.lcm (x i) (x (i-1))) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 20 → x i = 2 := 
sorry

end sequence_all_two_l582_582781


namespace domain_expression_l582_582789

theorem domain_expression (x : ℝ) :
  (2 * x - 6 ≥ 0) → (10 - x ≥ 0) →  (sqrt (10 - x) - 1 ≠ 0) → 
  3 ≤ x ∧ x < 9 :=
by
  intros h1 h2 h3
  have h4 : 3 ≤ x, exact (by linarith : 3 ≤ x)
  have h5 : x < 9, exact (by linarith : x < 9)
  exact ⟨h4, h5⟩

end domain_expression_l582_582789


namespace probability_abs_greater_than_two_l582_582454

/-- Let ξ be a random variable following the normal distribution N(0, σ^2).
    Given that P(-2 ≤ ξ ≤ 0) = 0.4, prove that P(|ξ| > 2) = 0.2. -/
theorem probability_abs_greater_than_two (σ : ℝ) [nonneg_real : σ ≥ 0] :
  ∀ (ξ : ℝ), ¬(∃ r, ξ = r - 0) → P(-2 ≤ ξ ∧ ξ ≤ 0) = 0.4 → P(|ξ| > 2) = 0.2 :=
by
  sorry

end probability_abs_greater_than_two_l582_582454


namespace each_friend_paid_five_l582_582629

-- Define the quantities and their relationships
def bag_cost : ℕ := 3
def number_of_bags : ℕ := 5
def number_of_friends : ℕ := 3

-- Calculate the total cost
def total_cost : ℕ := number_of_bags * bag_cost

-- Define the expected individual share
def individual_payment : ℕ := total_cost / number_of_friends

-- Prove that each friend paid $5
theorem each_friend_paid_five : individual_payment = 5 := by
  -- We define the values explicitly for clarity
  have h1 : total_cost = 15 := by
    unfold total_cost
    unfold bag_cost
    unfold number_of_bags
    norm_num
  have h2 : individual_payment = 15 / number_of_friends := by
    unfold individual_payment
    congr
    exact h1
  have h3 : 15 / number_of_friends = 5 := by
    norm_num
  rw [h2, h3]
  norm_num

end each_friend_paid_five_l582_582629


namespace discriminant_eq_complete_square_form_l582_582161

theorem discriminant_eq_complete_square_form (a b c t : ℝ) (h : a ≠ 0) (ht : a * t^2 + b * t + c = 0) :
  (b^2 - 4 * a * c) = (2 * a * t + b)^2 := 
sorry

end discriminant_eq_complete_square_form_l582_582161


namespace max_diff_arithmetic_progression_l582_582271

-- Define the conditions for a decreasing arithmetic progression
def is_decreasing_arithmetic_progression (a b c : ℤ) : Prop :=
  ∃ (d : ℤ), d < 0 ∧ b = a + d ∧ c = a + 2 * d

-- Define the conditions for six quadratic equations to have two distinct real roots
def has_two_distinct_real_roots (a b c : ℤ) : Prop :=
  let ψ := [
    (a, 2 * b, 4 * c), (a, 4 * c, 2 * b),
    (2 * b, a, 4 * c), (2 * b, 4 * c, a),
    (4 * c, a, 2 * b), (4 * c, 2 * b, a)
  ] in
  ∀ (coef : ℤ × ℤ × ℤ), coef ∈ ψ ->
    (coef.2.1) ^ 2 - 4 * coef.1 * coef.2.2 > 0

-- Define the task to prove the maximum d and the corresponding values of a, b, c
theorem max_diff_arithmetic_progression :
  ∃ (a b c : ℤ), is_decreasing_arithmetic_progression a b c ∧ has_two_distinct_real_roots a b c ∧ 
  (-3 = b - a ∧ a = 4 ∧ b = 1 ∧ c = -2) := 
by
  sorry

end max_diff_arithmetic_progression_l582_582271


namespace functional_equation_solution_l582_582424

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) * (f x - f y) = (x - y) * f x * f y) →
  ∃ C : ℝ, (f = λ x, C * x) ∨ (f = λ x, 0) :=
by {
  sorry
}

end functional_equation_solution_l582_582424


namespace prob_three_red_cards_l582_582179

noncomputable def probability_of_three_red_cards : ℚ :=
  let total_ways := 52 * 51 * 50
  let ways_to_choose_red_cards := 26 * 25 * 24
  ways_to_choose_red_cards / total_ways

theorem prob_three_red_cards : probability_of_three_red_cards = 4 / 17 := sorry

end prob_three_red_cards_l582_582179


namespace residual_at_sample_point_l582_582486

theorem residual_at_sample_point :
  ∀ (x y : ℝ), (8 * x - 70 = 10) → (x = 10) → (y = 13) → (13 - (8 * x - 70) = 3) :=
by
  intros x y h1 h2 h3
  sorry

end residual_at_sample_point_l582_582486


namespace janet_dresses_pockets_l582_582199

theorem janet_dresses_pockets :
  ∀ (x : ℕ), (∀ (dresses_with_pockets remaining_dresses total_pockets : ℕ),
  dresses_with_pockets = 24 / 2 →
  total_pockets = 32 →
  remaining_dresses = dresses_with_pockets - dresses_with_pockets / 3 →
  (dresses_with_pockets / 3) * x + remaining_dresses * 3 = total_pockets →
  x = 2) :=
by
  intros x dresses_with_pockets remaining_dresses total_pockets h1 h2 h3 h4
  sorry

end janet_dresses_pockets_l582_582199


namespace body_diagonal_length_l582_582142

-- Defining the side lengths of the rectangular prism
def length : ℝ := 3
def width : ℝ := 4
def height : ℝ := 5

-- Defining the body diagonal length calculation
def body_diagonal (l w h : ℝ) : ℝ := Real.sqrt (l^2 + w^2 + h^2)

-- Statement of the theorem
theorem body_diagonal_length : body_diagonal length width height = 5 * Real.sqrt 2 :=
by
  sorry

end body_diagonal_length_l582_582142


namespace purple_valley_skirts_l582_582245

def AzureValley : ℕ := 60

def SeafoamValley (A : ℕ) : ℕ := (2 * A) / 3

def PurpleValley (S : ℕ) : ℕ := S / 4

theorem purple_valley_skirts :
  PurpleValley (SeafoamValley AzureValley) = 10 :=
by
  sorry

end purple_valley_skirts_l582_582245


namespace range_of_x_l582_582832

theorem range_of_x (f : ℝ → ℝ)
  (h_even : ∀ x, f(x) = f(-x))
  (h_def : ∀ x, 0 ≤ x → f(x) = x - 1) :
  { x : ℝ | f(x - 1) < 0 } = { x | 0 < x ∧ x < 2 } :=
by
  sorry

end range_of_x_l582_582832


namespace power_function_properties_l582_582508

-- Define the initial condition
def passes_through (α : ℝ) : Prop :=
  (2 : ℝ)^α = (√2 : ℝ)

-- Main theorem without the proof
theorem power_function_properties :
  ∀ α : ℝ, passes_through α → (α = 1/2) ∧ (¬ (∀ x : ℝ, x ≥ 0 → x^α = (-x)^α)) ∧ (∀ x y : ℝ, 0 < x → x < y → x^α < y^α) :=
sorry

end power_function_properties_l582_582508


namespace probability_heart_king_l582_582650

theorem probability_heart_king :
  let total_cards := 52
  let total_kings := 4
  let hearts_count := 13
  let king_of_hearts := 1 in
  let prob_king_of_hearts_first := (1 : ℚ) / total_cards
  let prob_other_heart_first := (hearts_count - king_of_hearts : ℚ) / total_cards
  let prob_king_second_if_king_heart_first := (total_kings - king_of_hearts : ℚ) / (total_cards - 1)
  let prob_king_second_if_other_heart_first := (total_kings : ℚ) / (total_cards - 1) in
  prob_king_of_hearts_first * prob_king_second_if_king_heart_first +
  prob_other_heart_first * prob_king_second_if_other_heart_first = (1 : ℚ) / total_cards :=
by sorry

end probability_heart_king_l582_582650


namespace beth_gave_away_54_crayons_l582_582722

-- Define the initial number of crayons
def initialCrayons : ℕ := 106

-- Define the number of crayons left
def remainingCrayons : ℕ := 52

-- Define the number of crayons given away
def crayonsGiven (initial remaining: ℕ) : ℕ := initial - remaining

-- The goal is to prove that Beth gave away 54 crayons
theorem beth_gave_away_54_crayons : crayonsGiven initialCrayons remainingCrayons = 54 :=
by
  sorry

end beth_gave_away_54_crayons_l582_582722


namespace total_cost_is_96_l582_582913

noncomputable def hair_updo_cost : ℕ := 50
noncomputable def manicure_cost : ℕ := 30
noncomputable def tip_rate : ℚ := 0.20

def total_cost_with_tip (hair_cost manicure_cost : ℕ) (tip_rate : ℚ) : ℚ :=
  let hair_tip := hair_cost * tip_rate
  let manicure_tip := manicure_cost * tip_rate
  let total_tips := hair_tip + manicure_tip
  let total_before_tips := (hair_cost : ℚ) + (manicure_cost : ℚ)
  total_before_tips + total_tips

theorem total_cost_is_96 :
  total_cost_with_tip hair_updo_cost manicure_cost tip_rate = 96 := by
  sorry

end total_cost_is_96_l582_582913


namespace cube_root_problem_l582_582052

noncomputable def cube_root_rounded (x : ℝ) : ℝ :=
  Real.cbrt x

theorem cube_root_problem :
  let a := 2.5 * 4325
  let b := a / 7.5
  let c := b * (2^5)
  let d := 7 - Real.sqrt 36
  let e := (3^3) * d
  let f := c - e
  Real.floor (cube_root_rounded f * 1000) / 1000 = 35.877 :=
by
  sorry

end cube_root_problem_l582_582052


namespace supremum_integral_ratio_l582_582928

theorem supremum_integral_ratio (P : convex_polygon) 
  (h₁ : ∀ side ∈ P.sides, touches_circle side (circle 1)) :
  ∀ A : set (point ℝ), (h₂ : ∀ (x y : ℝ), point_in_A x y ↔ distance (x, y) P ≤ 1) →
  let f : point ℝ → ℝ := λ (p : point ℝ), num_intersections_unit_circle p P in
  sup (λ P : convex_polygon,
    (1 / measure_of A) * ∫ (p : point ℝ) in A, f p) = 8 / 3 :=
sorry

end supremum_integral_ratio_l582_582928


namespace theorem_most_suitable_for_factorization_is_B_l582_582378

def equationA : Prop := (x + 1) * (x - 3) = 2
def equationB : Prop := 2 * (x - 2)^2 = x^2 - 4
def equationC : Prop := x^2 + 3 * x - 1 = 0
def equationD : Prop := 5 * (2 - x)^2 = 3

theorem theorem_most_suitable_for_factorization_is_B :
  (equationA ∨ equationC ∨ equationD → False) ∧ (equationB → True) := 
by 
  sorry

end theorem_most_suitable_for_factorization_is_B_l582_582378


namespace range_sin_add_arcsin_l582_582615

open Real

theorem range_sin_add_arcsin :
  let f := λ x, sin x + arcsin x in
  -1 ≤ 1 ∧ (∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y) →
  set.range f = set.Icc (-sin 1 - π / 2) (sin 1 + π / 2) :=
by
  intros f h
  sorry

end range_sin_add_arcsin_l582_582615


namespace f_gt_2_l582_582606

open Real

noncomputable def f (x : ℝ) := exp x - log x

lemma monotonic_interval_f' : ∀ x > 0, monotone_on (fun t => exp t - 1 / t) (Ioi 0) := by
  sorry

theorem f_gt_2 : ∀ x > 0, f x > 2 := by
  sorry

end f_gt_2_l582_582606


namespace angle_D_in_convex_quadrilateral_l582_582450

theorem angle_D_in_convex_quadrilateral (A B C D : ℝ) (α β γ δ : ℝ) 
  (h_quad: α + β + γ + δ = 360)
  (h_γ: γ = 57) 
  (h_sin_sum: sin α + sin β = sqrt 2)
  (h_cos_sum: cos α + cos β = 2 - sqrt 2) :
  δ = 168 :=
by 
  sorry

end angle_D_in_convex_quadrilateral_l582_582450


namespace max_lateral_surface_area_of_prism_l582_582110

theorem max_lateral_surface_area_of_prism (a h r: ℝ) 
  (prism_condition : ∀ (x: ℝ), x = sqrt(3)) 
  (vertex_on_sphere : (sqrt (a^2 + a^2 + h^2)) = 2 * r) 
  (sphere_surface_area : 4 * π * r^2 = 12 * π) : 
  4 * a * h ≤ 12 * sqrt(2) :=
begin
  sorry
end

end max_lateral_surface_area_of_prism_l582_582110


namespace handshaking_remainder_l582_582177

theorem handshaking_remainder :
  let N := (stopheric12 (6! / 2) (choose 6 12) * (6! / 2)) + ((choose 4 12) * (3! / 2) * (7! / 2)) + ((11! / 2)) in
  N % 1000 = 600 := by
  sorry

end handshaking_remainder_l582_582177


namespace smallest_w_l582_582673

theorem smallest_w (w : ℕ) (h1 : 2^4 ∣ 1452 * w) (h2 : 3^3 ∣ 1452 * w) (h3 : 13^3 ∣ 1452 * w) : w = 79132 :=
by
  sorry

end smallest_w_l582_582673


namespace monotonicity_fx_range_of_a_sum_sqrt_ineq_l582_582476

-- Problem 1: Monotonicity of f(x)
theorem monotonicity_fx (x : ℝ) : 
  let f (x : ℝ) := x * Real.exp x - Real.exp x in
  if x > 0 then (∀ x > 0, Real.deriv f x > 0) 
  else if x < 0 then (∀ x < 0, Real.deriv f x < 0) :=
by skip_proof

-- Problem 2: Range of values for a
theorem range_of_a {x a : ℝ} (hx : x > 0) (hf : x * Real.exp (a * x) - Real.exp x < -1) : 
  a ≤ 0.5 :=
by skip_proof

-- Problem 3: Proving the inequality
open Real 

theorem sum_sqrt_ineq (n : ℕ) (h : n > 0) :
  (finset.range n).sum (λ i, 1 / sqrt ((i + 1)^2 + (i + 1))) > log (n + 1) :=
by skip_proof

end monotonicity_fx_range_of_a_sum_sqrt_ineq_l582_582476


namespace greatest_lower_bound_of_sum_of_squares_l582_582571

theorem greatest_lower_bound_of_sum_of_squares (n : ℕ) (a_{n-1} : ℝ) (roots : Fin n → ℝ)
  (h_poly : ∃ p : Polynomial ℝ, p.nat_degree = n ∧ p.leadingCoeff = 1 ∧ ∀ i : Fin n, p.eval (roots i) = 0 ∧ p.coeff (n - 1) = a_{n-1} ∧ p.coeff (n - 2) = a_{n-1}) :
  let sum_of_squares := ∑ i in Finset.range n, (roots ⟨i, (Finset.mem_range.2 (Nat.lt_of_add_one_lt (Nat.succ_pos i)))⟩)^2 in
  (∀ r : ℝ, r * r - 2 * r ≥ sum_of_squares) ↔ sum_of_squares = 1 :=
sorry

end greatest_lower_bound_of_sum_of_squares_l582_582571


namespace henry_stickers_distribution_l582_582856

theorem henry_stickers_distribution :
  ∃ (ways : ℕ), ways = 126 ∧
    (let stickers := 10
     let sheets := 5
     (number_of_ways stickers sheets)) 
  := sorry

noncomputable def number_of_ways (stickers sheets : ℕ) : ℕ :=
  if h : sheets > 0 then 
    let remaining_stickers := stickers - sheets in 
    number_of_partitions remaining_stickers sheets
  else 0

noncomputable def number_of_partitions (remaining_stickers sheets : ℕ) : ℕ := sorry

end henry_stickers_distribution_l582_582856


namespace geometric_sequence_a7_l582_582176

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) := a_1 * q^(n - 1)

theorem geometric_sequence_a7 
  (a1 q : ℝ)
  (a1_neq_zero : a1 ≠ 0)
  (a9_eq_256 : a_n a1 q 9 = 256)
  (a1_a3_eq_4 : a_n a1 q 1 * a_n a1 q 3 = 4) :
  a_n a1 q 7 = 64 := 
sorry

end geometric_sequence_a7_l582_582176


namespace locus_centers_of_rectangles_l582_582792

-- Definitions of relevant geometric concepts
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A B C : Point

structure Rectangle where
  K L M N : Point

def described (T : Triangle) (R : Rectangle) : Prop :=
  R.K = T.A ∧ 
  ∃ P1 P2, P1 ≠ T.A ∧ P2 ≠ T.A ∧ 
  SegmentContains P1 T.B R.L ∧ 
  SegmentContains P2 T.C R.M

def isAcute (T : Triangle) : Prop :=
  ∠ A < 90 ∧ ∠ B < 90 ∧ ∠ C < 90 
    where 
      A := calcAngle T.A T.B T.C
      B := calcAngle T.B T.A T.C
      C := calcAngle T.C T.A T.B

def calcCenter (R : Rectangle) : Point := 
  midpoint (midpoint R.K R.L) (midpoint R.M R.N)

def locusCentersRectangles (T : Triangle) : Set Point :=
  { O : Point | ∃ R : Rectangle, described T R ∧ O = calcCenter R }

def midpoint (A B : Point) : Point := 
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

-- Proof problem statement
theorem locus_centers_of_rectangles {T : Triangle} :
  isAcute T →
  (locusCentersRectangles T = curvilinearTriangle T.midlines) ∧ 
  (¬isAcute T → locusCentersRectangles T = twoArcsSemicircles T.midlines) := sorry

end locus_centers_of_rectangles_l582_582792


namespace sequence_a_n_T_2012_l582_582470

def S (n : ℕ) (a : ℕ → ℝ) := (1/2) - (1/2) * (a n)
def f (x : ℝ) := Real.log x / Real.log 3
def b (n : ℕ) (a : ℕ → ℝ) := ∑ i in Finset.range n, f (a (i + 1))
def T (n : ℕ) (a : ℕ → ℝ) := ∑ i in Finset.range n, 1 / b (i + 1) a

theorem sequence_a_n (n : ℕ) (a : ℕ → ℝ)
  (hS : ∀ n, S n a = (1/2) - (1/2) * a n) :
  a 1 = (1 / 3) ∧ (n > 0 → a (n + 1) = (1 / 3^(n + 1))) := sorry

theorem T_2012 (a : ℕ → ℝ)
  (hb : ∀ n, b n a = - (n * (n + 1)) / 2)
  (hT : ∀ n, T n a = -2 * (1 - 1 / (n + 1))) :
  T 2012 a = -4024 / 2013 := sorry

end sequence_a_n_T_2012_l582_582470


namespace distance_between_opposite_faces_of_regular_octahedron_l582_582702

theorem distance_between_opposite_faces_of_regular_octahedron (a : ℝ) (h : a = 1) : ∃ d : ℝ, d = (Real.sqrt 6) / 3 :=
by
  use (Real.sqrt 6) / 3
  sorry

end distance_between_opposite_faces_of_regular_octahedron_l582_582702


namespace smallest_three_digit_times_largest_single_digit_l582_582283

theorem smallest_three_digit_times_largest_single_digit :
  let x := 100
  let y := 9
  ∃ z : ℕ, z = x * y ∧ 100 ≤ z ∧ z < 1000 :=
by
  let x := 100
  let y := 9
  use x * y
  sorry

end smallest_three_digit_times_largest_single_digit_l582_582283


namespace exists_circle_in_convex_polygon_l582_582241

open Set

-- Define a convex polygon as a set of points in ℝ²
variable {P : Type*} [MetricSpace P] [ConvexSpace P]

-- Define the area of the polygon
variable (A : ℝ)

-- Define the perimeter of the polygon
variable (P_perimeter : ℝ)

-- Define convex polygon
variable (polygon : Set P)
variable [h1 : Convex polygon]

-- Define the existence of a point within the polygon such that the distance is met
theorem exists_circle_in_convex_polygon (A P_perimeter : ℝ) (polygon : Set P) [h1 : Convex polygon polygon] :
  ∃ (O : P), ∀ (side : Set P), side ∈ (boundary polygon) →
  (euclideanDist O side ≥ A / P_perimeter) := sorry

end exists_circle_in_convex_polygon_l582_582241


namespace find_speed_of_second_train_l582_582369

-- Definitions of the given conditions
def length_first_train : ℕ := 150
def speed_first_train_kmph : ℕ := 60
def speed_first_train_mps : ℝ := 60 * (1000 / 3600)  -- Converting km/hr to m/s
def length_second_train : ℕ := 180
def time_to_cross : ℝ := 4
def total_distance : ℕ := length_first_train + length_second_train

-- The definition of the theorem to find the speed of the second train in km/hr
theorem find_speed_of_second_train : 
  let relative_speed := total_distance / time_to_cross in
  let speed_second_train_mps := relative_speed - speed_first_train_mps in
  let speed_second_train_kmph := speed_second_train_mps * (3600 / 1000) in
  abs (speed_second_train_kmph - 237) < 1 :=
by
  sorry

end find_speed_of_second_train_l582_582369


namespace sally_jolly_money_sum_l582_582581

/-- Prove the combined amount of money of Sally and Jolly is $150 given the conditions. -/
theorem sally_jolly_money_sum (S J x : ℝ) (h1 : S - x = 80) (h2 : J + 20 = 70) (h3 : S + J = 150) : S + J = 150 :=
by
  sorry

end sally_jolly_money_sum_l582_582581


namespace lines_parallel_or_skew_l582_582527

variables {Point Line Plane : Type}
variable  (a b : Line)
variable  (α β : Plane)

-- Conditions
axiom line_in_plane_a  : ∀ {p : Point}, p ∈ a → p ∈ α
axiom line_in_plane_b  : ∀ {p : Point}, p ∈ b → p ∈ β
axiom planes_parallel  : ∀ {p : Point}, p ∈ α → p ∈ β → false

-- Theorem: Positional relationship between lines a and b
theorem lines_parallel_or_skew : (∀ {p q : Point}, p ∈ a → q ∈ b → p = q → false) → (a = b ∨ a.skew_with b) :=
by 
  -- sorry is used to skip the proof
  sorry

end lines_parallel_or_skew_l582_582527


namespace intersection_point_has_radial_distance_2_l582_582519

noncomputable def polar_intersection_radial_distance : ℝ :=
  let l := λ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) - 2
  let x_squared_plus_y_squared := λ (x y : ℝ), x^2 + y^2 - 4
  let points := [(0, 2), (2, 0)]
  let ρ := λ x y, Real.sqrt (x^2 + y^2)
  if h : ∀ (x y : ℝ), (x, y) ∈ points ∧ x_squared_plus_y_squared x y = 0 then 2 else 0

theorem intersection_point_has_radial_distance_2
  (ρ θ : ℝ)
  (hline : l ρ θ = 0)
  (hcirc : x_squared_plus_y_squared ρ 0 = 0) :
  polar_intersection_radial_distance = 2 :=
by
  sorry

end intersection_point_has_radial_distance_2_l582_582519


namespace fraction_of_300_greater_than_3_fifths_of_125_l582_582391

theorem fraction_of_300_greater_than_3_fifths_of_125 (f : ℚ)
    (h : f * 300 = 3 / 5 * 125 + 45) : 
    f = 2 / 5 :=
sorry

end fraction_of_300_greater_than_3_fifths_of_125_l582_582391


namespace functional_relationship_value_of_a_l582_582467

-- Define the direct proportionality condition
def proportional (x y : ℝ) (k : ℝ) := y = k * (2 * x + 3)

-- Given conditions
variable (k : ℝ)
variable (y : ℝ)
variable (x : ℝ)

-- First part: Prove the functional relationship
theorem functional_relationship : proportional x y k ∧ y = -5 ∧ x = 1 → y = -2 * x - 3 :=
by
  intros h,
  sorry

-- Second part: Given the point (a, 2) prove the value of a
variable (a : ℝ)

theorem value_of_a : y = -2 * x - 3 ∧ y = 2 → x = a :=
by
  intros h,
  sorry

end functional_relationship_value_of_a_l582_582467


namespace positive_number_property_l582_582698

theorem positive_number_property (y : ℝ) (hy : 0 < y) : 
  (y^2 / 100) + 6 = 10 → y = 20 := by
  sorry

end positive_number_property_l582_582698


namespace inequality_solution_l582_582780

theorem inequality_solution (x : ℝ) :
  (frac (1) (x^2 + 4) > frac (5) (x) + frac (21) (10)) ↔ (x > -2 ∧ x < 0) :=
sorry

end inequality_solution_l582_582780


namespace smallest_possible_n_l582_582559

def arithmeticSeq (x : ℕ → ℝ) : Prop :=
  ∃ a d, ∀ i, x i = a + i * d

def allAbsLessThanOne (x : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i < n → |x i| < 1

noncomputable def absSum (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, |x i|

noncomputable def sum (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, x i

theorem smallest_possible_n :
  ∃ n (x : ℕ → ℝ), arithmeticSeq x ∧ allAbsLessThanOne x n ∧ 
                    absSum x n = 25 + |sum x n| ∧ 
                    ∀ m, (∀ y, arithmeticSeq y ∧ allAbsLessThanOne y m ∧ 
                               absSum y m = 25 + |sum y m| → m ≥ n) ∧ 
                    n = 26 :=
by
  sorry

end smallest_possible_n_l582_582559


namespace math_problem_l582_582497

theorem math_problem (a b : ℝ) : (60^a = 3) → (60^b = 5) → 12^((1 - a - b) / (2 * (1 - b))) = 2 :=
by
  sorry

end math_problem_l582_582497


namespace gcd_6Tn_nplus1_l582_582092

theorem gcd_6Tn_nplus1 (n : ℕ) (h : 0 < n) : gcd (3 * n * n + 3 * n) (n + 1) = 1 := by
  sorry

end gcd_6Tn_nplus1_l582_582092


namespace beta_bound_l582_582461

noncomputable def omega (M : ℕ) : ℕ :=
  Multiset.card (Nat.factors M).toFinset

noncomputable def h (M n : ℕ) : ℕ :=
  (Finset.range n).filter (λ x, Nat.coprime (x + 1) M).card

noncomputable def beta (M : ℕ) : ℚ :=
  h M M / M

theorem beta_bound (M : ℕ) (h : ℕ → ℕ → ℕ) (beta : ℚ) :
    0 < M →
    (∀ n, h M n = (Finset.range n).filter (λ x, Nat.coprime (x + 1) M).card) →
    beta = h M M / M →
    ∃ (S : Finset ℕ), S.card ≥ M/3 ∧ ∀ n ∈ S, 
      abs (h M n - beta * n) ≤ sqrt (beta * 2 ^ (omega M - 3)) + 1 :=
sorry

end beta_bound_l582_582461


namespace min_disks_needed_l582_582197

-- Define the file sizes and counts
def num_files_total : ℕ := 40
def file_size_1 : ℝ := 1.2
def file_size_2 : ℝ := 0.9
def file_size_3 : ℝ := 0.5
def num_files_1 : ℕ := 5
def num_files_2 : ℕ := 15
def num_files_3 : ℕ := num_files_total - num_files_1 - num_files_2

-- Define the disk capacity
def disk_capacity : ℝ := 2.0

-- Define the problem statement to prove
theorem min_disks_needed : ∃ disks_needed : ℕ, disks_needed = 16 ∧
  (∀ (files_packed : ℕ → ℝ), (∀ i, i < num_files_total → 
  ((∃ f1 f2, f1 + f2 ≤ disk_capacity ∧ f1 ∈ {1.2, 0.9, 0.5} ∧ f2 ∈ {1.2, 0.9, 0.5} ∧ files_packed i = (f1 + f2)) ∨
  (files_packed i ≤ disk_capacity ∧ files_packed i ∈ {1.2, 0.9, 0.5})))
  → (∃ d, d ≤ 16 ∧ ∀ i, i < d → files_packed i ≤ disk_capacity))
  sorry

end min_disks_needed_l582_582197


namespace mean_first_second_fifth_sixth_diff_l582_582117

def six_numbers_arithmetic_mean_condition (a1 a2 a3 a4 a5 a6 A : ℝ) :=
  (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

def mean_first_four_numbers (a1 a2 a3 a4 A : ℝ) :=
  (a1 + a2 + a3 + a4) / 4 = A + 10

def mean_last_four_numbers (a3 a4 a5 a6 A : ℝ) :=
  (a3 + a4 + a5 + a6) / 4 = A - 7

theorem mean_first_second_fifth_sixth_diff (a1 a2 a3 a4 a5 a6 A : ℝ) :
  six_numbers_arithmetic_mean_condition a1 a2 a3 a4 a5 a6 A →
  mean_first_four_numbers a1 a2 a3 a4 A →
  mean_last_four_numbers a3 a4 a5 a6 A →
  ((a1 + a2 + a5 + a6) / 4) = A - 3 :=
by
  intros h1 h2 h3
  sorry

end mean_first_second_fifth_sixth_diff_l582_582117


namespace unit_digit_of_7_pow_6_l582_582661

theorem unit_digit_of_7_pow_6 : Nat.digit (7^6) 0 = 9 := by
  sorry

end unit_digit_of_7_pow_6_l582_582661


namespace greatest_gcd_of_6T_n_and_n_plus_1_l582_582095

theorem greatest_gcd_of_6T_n_and_n_plus_1 (n : ℕ) (h_pos : 0 < n) :
  let T_n := n * (n + 1) / 2 in
  gcd (6 * T_n) (n + 1) = 3 ↔ (n + 1) % 3 = 0 :=
by
  sorry

end greatest_gcd_of_6T_n_and_n_plus_1_l582_582095


namespace phase_shift_of_function_l582_582055

theorem phase_shift_of_function :
  ∀ (A B C : ℝ), A = 3 ∧ B = 4 ∧ C = -π/4 → (C/B) = -π/16 :=
by
  intros A B C h
  cases h with ha hb
  cases hb with hb hc
  simp [ha, hb, hc]
  sorry

end phase_shift_of_function_l582_582055


namespace lambda_range_l582_582106

variables {x y λ1 λ2 : ℝ}

def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2) = 1
def line_passing_through (M : ℝ × ℝ) (x y : ℝ) (k : ℝ) : Prop := y = k * (x - M.1) + M.2
def intersects_x_axis (line : ℝ -> ℝ) (N : ℝ × ℝ) : Prop := line N.1 = 0
def vector_relationship (M N A B : ℝ × ℝ) (λ1 λ2 : ℝ) : Prop :=
  let vMA := (A.1 - M.1, A.2 - M.2)
  let vAN := (N.1 - A.1, N.2 - A.2)
  let vMB := (B.1 - M.1, B.2 - M.2)
  let vBN := (N.1 - B.1, N.2 - B.2)
  (vMA = (λ1) • vAN) ∧ (vMB = (λ2) • vBN)

noncomputable def range_of_values (λ1 λ2 : ℝ) : set ℝ :=
  {r | r ∈ (-∞, -74/35) ∪ {x | x ≥ 2}}

theorem lambda_range (M A B N : ℝ × ℝ) (k : ℝ) :
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  line_passing_through M A.1 A.2 k →
  line_passing_through M B.1 B.2 k →
  intersects_x_axis (λ x, k * (x - M.1) + M.2) N →
  vector_relationship M N A B λ1 λ2 →
  ∃ (r : ℝ), r = λ1 / λ2 + λ2 / λ1 ∧ r ∈ range_of_values λ1 λ2 :=
by sorry

end lambda_range_l582_582106


namespace part1_part2_l582_582531

/-- Consider the sequence {a_n} defined as follows -/
def a : ℕ → ℝ
| 0     := 0     -- This is just a placeholder since the sequence actually starts from n = 1.
| 1     := 1 / 2
| (n + 1) := (n + 1) / (2 * n) * a n

/-- Define the sequence {a_n / n} -/
def b (n : ℕ) : ℝ := a n / n

/-- Define the sum of the first n terms of the sequence {a_n} -/
def S_n (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

/-- Prove that the sequence {a_n / n} is a geometric sequence -/
theorem part1 : ∃ r : ℝ, ∃ b₁ : ℝ, ∀ n : ℕ, b (n + 1) = b₁ * r^n :=
sorry

/-- Prove that the sum of the first n terms of the sequence {a_n} is S_n = 2 - (n + 2) / 2^n -/
theorem part2 (n : ℕ) : S_n n = 2 - (n + 2) / 2^n :=
sorry

end part1_part2_l582_582531


namespace monotonicity_fx_range_of_a_sum_sqrt_ineq_l582_582477

-- Problem 1: Monotonicity of f(x)
theorem monotonicity_fx (x : ℝ) : 
  let f (x : ℝ) := x * Real.exp x - Real.exp x in
  if x > 0 then (∀ x > 0, Real.deriv f x > 0) 
  else if x < 0 then (∀ x < 0, Real.deriv f x < 0) :=
by skip_proof

-- Problem 2: Range of values for a
theorem range_of_a {x a : ℝ} (hx : x > 0) (hf : x * Real.exp (a * x) - Real.exp x < -1) : 
  a ≤ 0.5 :=
by skip_proof

-- Problem 3: Proving the inequality
open Real 

theorem sum_sqrt_ineq (n : ℕ) (h : n > 0) :
  (finset.range n).sum (λ i, 1 / sqrt ((i + 1)^2 + (i + 1))) > log (n + 1) :=
by skip_proof

end monotonicity_fx_range_of_a_sum_sqrt_ineq_l582_582477


namespace periodic_F_F_periodic_l582_582261

variable (f : ℝ → ℝ)
variable (h : ∀ x : ℝ, f(x + 1) - f(x) = 1)

theorem periodic_F : ∀ x : ℝ, F(x + 1) = F(x) :=
  by
  sorry

def F (x : ℝ) : ℝ := f(x) - x

-- Conclude that F is periodic with period 1
theorem F_periodic : ∀ x : ℝ, F(x + 1) = F(x) :=
  by
  intro x
  -- all additional steps are omitted
  sorry

end periodic_F_F_periodic_l582_582261


namespace more_philosophers_than_mathematicians_l582_582376

theorem more_philosophers_than_mathematicians (x : ℕ) (h1 : ∃ y, 7 * y = x) (h2 : ∃ z, 9 * z = x) : x ≠ 0 → 9 * x > 7 * x :=
by
  intros hx
  have h : x > 0 := by
    contrapose! hx
    exact eq_zero_of_mul_eq_zero_left hx
  exact mul_lt_mul_of_pos_right (by norm_num) h

end more_philosophers_than_mathematicians_l582_582376


namespace years_to_earn_house_l582_582003

-- Defining the variables
variables (E S H : ℝ)

-- Defining the assumptions
def annual_expenses_savings_relation (E S : ℝ) : Prop :=
  8 * E = 12 * S

def annual_income_relation (H E S : ℝ) : Prop :=
  H / 24 = E + S

-- Theorem stating that it takes 60 years to earn the amount needed to buy the house
theorem years_to_earn_house (E S H : ℝ) 
  (h1 : annual_expenses_savings_relation E S) 
  (h2 : annual_income_relation H E S) : 
  H / S = 60 :=
by
  sorry

end years_to_earn_house_l582_582003


namespace berries_count_l582_582339

theorem berries_count (total_berries : ℕ)
  (h1 : total_berries = 42)
  (h2 : total_berries / 2 = 21)
  (h3 : total_berries / 3 = 14) :
  total_berries - (total_berries / 2 + total_berries / 3) = 7 :=
by
  rw [h1, h2, h3]
  norm_num
  exact rfl

end berries_count_l582_582339


namespace rectangle_area_l582_582623

theorem rectangle_area (a b k : ℕ)
  (h1 : k = 6 * (a + b) + 36)
  (h2 : k = 114)
  (h3 : a / b = 8 / 5) :
  a * b = 40 :=
by {
  sorry
}

end rectangle_area_l582_582623


namespace number_of_pairs_l582_582462

theorem number_of_pairs : 
  ∃ n : ℕ, 
  (∀ (a b : ℕ), (a > 0) → (b > 0) → (1 / (a : ℚ) - 1 / (b : ℚ) = 1 / 2018) ↔ ∃ (a b : ℕ), (a, b) ∈ {(2017, 2018 * 2018 - 2018), (2016, 2 * 1009 * 1009 - 2018), (2014, 1009 * 1009 - 2018), (1009, 4 * 1009 - 2018)}) ∧ n = 4 :=
sorry

end number_of_pairs_l582_582462


namespace floor_value_correct_l582_582419

def calc_floor_value : ℤ :=
  let a := (15 : ℚ) / 8
  let b := a^2
  let c := (225 : ℚ) / 64
  let d := 4
  let e := (19 : ℚ) / 5
  let f := d + e
  ⌊f⌋

theorem floor_value_correct : calc_floor_value = 7 := by
  sorry

end floor_value_correct_l582_582419


namespace series_sum_correct_l582_582753

def serieSum (n : ℕ) (changeSigns: List ℕ) : ℤ :=
  let rec helper (k : ℕ) (currentSign : Int) (changes : List ℕ): ℤ :=
    if k > n then 0
    else if changes = [] then currentSign * k + helper (k + 1) currentSign changes
    else if k = List.head changes then currentSign * k + helper (k + 1) (-currentSign) (List.tail changes)
    else currentSign * k + helper (k + 1) currentSign changes
  helper 1 1 changeSigns

theorem series_sum_correct :
  serieSum 100 [4, 9, 25, 49] = 3116 :=
  by
  sorry

end series_sum_correct_l582_582753


namespace probability_first_card_heart_second_king_l582_582644

theorem probability_first_card_heart_second_king :
  ∀ (deck : Finset ℕ) (is_heart : ℕ → Prop) (is_king : ℕ → Prop),
  deck.card = 52 →
  (∀ card ∈ deck, is_heart card ∨ ¬ is_heart card) →
  (∀ card ∈ deck, is_king card ∨ ¬ is_king card) →
  (∃ p : ℚ, p = 1/52) :=
by
  intros deck is_heart is_king h_card h_heart h_king,
  sorry

end probability_first_card_heart_second_king_l582_582644


namespace largest_consecutive_set_having_triangle_property_l582_582745

noncomputable def max_n_with_triangle_property : ℕ :=
  let S := {6, 7, 13, 20, 33, 53, 86, 139, 225, 364}
  363

theorem largest_consecutive_set_having_triangle_property :
  ∀ S : set ℕ, {6, 7, 8, ..., max_n_with_triangle_property} ⊆ S →
  (∀ s ⊆ S, s.card = 10 →  (∀ a b c ∈ s, a + b > c ∧ b + c > a ∧ a + c > b)) :=
sorry

end largest_consecutive_set_having_triangle_property_l582_582745


namespace lilith_regular_price_l582_582225

theorem lilith_regular_price (n : ℕ) (r_intended : ℝ) (r_actual : ℝ) (P : ℝ) 
  (h1 : n = 60)
  (h2 : r_intended = 120)
  (h3 : r_actual = 111)
  (h4 : P = r_actual / n) : 
  P = 1.85 := 
by
  rw [h4, h3, h1]
  norm_num
  sorry

end lilith_regular_price_l582_582225


namespace purple_valley_skirts_l582_582244

theorem purple_valley_skirts (azure_valley_skirts : ℕ) (h1 : azure_valley_skirts = 60) :
    let seafoam_valley_skirts := (2 / 3 : ℚ) * azure_valley_skirts in
    let purple_valley_skirts := (1 / 4 : ℚ) * seafoam_valley_skirts in
    purple_valley_skirts = 10 :=
by
  let seafoam_valley_skirts := (2 / 3 : ℚ) * azure_valley_skirts
  let purple_valley_skirts := (1 / 4 : ℚ) * seafoam_valley_skirts
  have h2 : seafoam_valley_skirts = (2 / 3 : ℚ) * 60 := by
    rw [h1]
  have h3 : purple_valley_skirts = (1 / 4 : ℚ) * ((2 / 3 : ℚ) * 60) := by
    rw [h2]
  have h4 : purple_valley_skirts = (1 / 4 : ℚ) * 40 := by
    norm_num [h3]
  have h5 : purple_valley_skirts = 10 := by
    norm_num [h4]
  exact h5

end purple_valley_skirts_l582_582244


namespace select_four_distinct_ns_l582_582439

theorem select_four_distinct_ns (p : ℝ) (hp : p ≥ 1) :
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  p < ↑a ∧ ↑a < (2 + sqrt (p + 1 / 4))^2 ∧
  p < ↑b ∧ ↑b < (2 + sqrt (p + 1 / 4))^2 ∧
  p < ↑c ∧ ↑c < (2 + sqrt (p + 1 / 4))^2 ∧
  p < ↑d ∧ ↑d < (2 + sqrt (p + 1 / 4))^2 ∧
  a * b = c * d :=
sorry

end select_four_distinct_ns_l582_582439


namespace textbook_distribution_ways_l582_582703

theorem textbook_distribution_ways : 
  let total_books := 8
  let valid_distributions := {n // 1 ≤ n ∧ n ≤ total_books - 1}
  ∃n, ∑ n in valid_distributions, 1 = 7 :=
by sorry

end textbook_distribution_ways_l582_582703


namespace sqrt4_add_sqrt5_ne_sqrt9_l582_582668

theorem sqrt4_add_sqrt5_ne_sqrt9 : ¬ (sqrt 4 + sqrt 5 = sqrt 9) :=
sorry

end sqrt4_add_sqrt5_ne_sqrt9_l582_582668


namespace constant_term_in_expansion_l582_582051

theorem constant_term_in_expansion : 
  let a := (x : ℝ)
  let b := - (2 / Real.sqrt x)
  let n := 6
  let general_term (r : Nat) : ℝ := Nat.choose n r * a * (b ^ (n - r))
  (∀ x : ℝ, ∃ (r : Nat), r = 4 ∧ (1 - (n - r) / 2 = 0) →
  general_term 4 = 60) :=
by
  sorry

end constant_term_in_expansion_l582_582051


namespace mean_first_second_fifth_sixth_diff_l582_582116

def six_numbers_arithmetic_mean_condition (a1 a2 a3 a4 a5 a6 A : ℝ) :=
  (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

def mean_first_four_numbers (a1 a2 a3 a4 A : ℝ) :=
  (a1 + a2 + a3 + a4) / 4 = A + 10

def mean_last_four_numbers (a3 a4 a5 a6 A : ℝ) :=
  (a3 + a4 + a5 + a6) / 4 = A - 7

theorem mean_first_second_fifth_sixth_diff (a1 a2 a3 a4 a5 a6 A : ℝ) :
  six_numbers_arithmetic_mean_condition a1 a2 a3 a4 a5 a6 A →
  mean_first_four_numbers a1 a2 a3 a4 A →
  mean_last_four_numbers a3 a4 a5 a6 A →
  ((a1 + a2 + a5 + a6) / 4) = A - 3 :=
by
  intros h1 h2 h3
  sorry

end mean_first_second_fifth_sixth_diff_l582_582116


namespace minimize_degree_2_l582_582174

theorem minimize_degree_2 (G : SimpleGraph V)
  (h_connected : G.Connected)
  (h_complete_subgraph : ∀ (A B : V), G.Adj A B → ∃ (C D : V), G.Adj C D ∧ G.Adj A C ∧ G.Adj A D ∧ G.Adj B C ∧ G.Adj B D ∧ G.Adj C D) :
  ∃ (T : SimpleGraph V), T.SpanningTree G ∧
  (T.degree_two_vertices.count ≤ 99) ∧ 
  (∀ (u v : V), u ≠ v → ∃! (p : T.Path u v), p.Simple) :=
sorry

end minimize_degree_2_l582_582174


namespace simplify_fraction_l582_582731

theorem simplify_fraction : 
  (1722^2 - 1715^2) / (1731^2 - 1708^2) = (7 * 3437) / (23 * 3439) :=
by
  -- Proof will go here
  sorry

end simplify_fraction_l582_582731


namespace biker_bob_east_distance_l582_582386

-- Here are the conditions
def biker_bob_rides_west := 20
def biker_bob_rides_north_1 := 6
def biker_bob_rides_north_2 := 18
def distance_between_towns := 26

-- Proving Biker Bob rode approximately 18.76 miles east
theorem biker_bob_east_distance :
  let x := real.sqrt (distance_between_towns^2 - (biker_bob_rides_north_1 + biker_bob_rides_north_2)^2)
  x = 18.76 :=
by
  sorry

end biker_bob_east_distance_l582_582386


namespace squares_in_rectangle_l582_582170

theorem squares_in_rectangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a ≤ 1) (h5 : b ≤ 1) (h6 : c ≤ 1) (h7 : a + b + c = 2)  : 
  a + b + c ≤ 2 := sorry

end squares_in_rectangle_l582_582170


namespace marie_survey_total_l582_582521

theorem marie_survey_total (y : ℕ) : (0.923 * y = 48) ∧ (0.375 * x = 18 → x = 48) → y = 52 :=
by 
  sorry

end marie_survey_total_l582_582521


namespace smallest_n_inequality_l582_582933

theorem smallest_n_inequality :
  ∀ (x y z : ℝ), (x^2 + y^2 + z^2) ≤ 3 * (x^4 + y^4 + z^4) :=
sorry

example : ∀ (n : ℕ), (∀ (x y z : ℝ), (x^2 + y^2 + z^2) ≤ n * (x^4 + y^4 + z^4)) → n ≥ 3 :=
by {
  intros n h,
  -- We need to show n ≥ 3
  have key : (n = 0 ∨ n = 1 ∨ n = 2 ∨ n ≥ 3), by omega,
  cases key,
  { -- n = 0
    exfalso,
    specialize h 1 1 1,
    linarith,
  },
  cases key,
  { -- n = 1
    exfalso,
    specialize h 1 1 1,
    linarith,
  },
  cases key,
  { -- n = 2
    exfalso,
    specialize h 1 1 1,
    linarith,
  },
  -- n ≥ 3
  exact key,
}

end smallest_n_inequality_l582_582933


namespace weight_of_each_bag_of_food_l582_582547

theorem weight_of_each_bag_of_food
  (horses : ℕ)
  (feedings_per_day : ℕ)
  (pounds_per_feeding : ℕ)
  (days : ℕ)
  (bags : ℕ)
  (total_food_in_pounds : ℕ)
  (h1 : horses = 25)
  (h2 : feedings_per_day = 2)
  (h3 : pounds_per_feeding = 20)
  (h4 : days = 60)
  (h5 : bags = 60)
  (h6 : total_food_in_pounds = horses * (feedings_per_day * pounds_per_feeding) * days) :
  total_food_in_pounds / bags = 1000 :=
by
  sorry

end weight_of_each_bag_of_food_l582_582547


namespace no_fractional_x_y_l582_582760

theorem no_fractional_x_y (x y : ℚ) (H1 : ¬ (x.denom = 1 ∧ y.denom = 1)) (H2 : ∃ m : ℤ, 13 * x + 4 * y = m) (H3 : ∃ n : ℤ, 10 * x + 3 * y = n) : false :=
sorry

end no_fractional_x_y_l582_582760


namespace number_of_natural_factors_of_N_l582_582157

def N : ℕ := 2^4 * 3^3 * 5^2 * 7

theorem number_of_natural_factors_of_N : nat.divisors N = 120 := by
  sorry

end number_of_natural_factors_of_N_l582_582157


namespace maximum_cos_value_l582_582192

theorem maximum_cos_value (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A = 60) :
  (cos A + 2 * cos ((B + C) / 2) = 3 / 2) :=
by
  -- Given: A + B + C = 180°
  -- Given: A = 60°
  -- Show: cos A + 2 * cos ((B + C) / 2) = 3 / 2
  sorry

end maximum_cos_value_l582_582192


namespace find_a_l582_582447

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + 1) / (x + 1)

theorem find_a (a : ℝ) (h1 : ∃ t, t = (f a 1 - 1) / (1 - 0) ∧ t = ((3 * a - 1) / 4)) : a = -1 :=
by
  -- Auxiliary steps to frame the Lean theorem precisely
  let f1 := f a 1
  have h2 : f1 = (a + 1) / 2 := sorry
  have slope_tangent : ∀ t : ℝ, t = (3 * a - 1) / 4 := sorry
  have tangent_eq : (∀ (x y : ℝ), y - f1 = ((3 * a - 1) / 4) * (x - 1)) := sorry
  have pass_point : ∀ (x y : ℝ), (x, y) = (0, 1) -> (1 : ℝ) - ((a + 1) / 2) = ((1 - 3 * a) / 4) := sorry
  exact sorry

end find_a_l582_582447


namespace euler_totient_bound_l582_582240

theorem euler_totient_bound (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : (Nat.totient^[k]) n = 1) :
  n ≤ 3^k :=
sorry

end euler_totient_bound_l582_582240


namespace probability_sum_leq_10_l582_582653

def eight_sided_die : Finset ℕ := Finset.range 9 \ {0}
def six_sided_die : Finset ℕ := Finset.range 7 \ {0}

theorem probability_sum_leq_10 :
  let total_outcomes := 48
      favorable_outcomes := 39
      probability := favorable_outcomes / total_outcomes.to_rat
  in probability = (13 / 16 : ℚ) :=
by 
  sorry

end probability_sum_leq_10_l582_582653


namespace inscribed_square_area_l582_582526

theorem inscribed_square_area (s : ℝ) (h : s = 28) : s^2 = 784 :=
by {
  rw h,
  norm_num
}

end inscribed_square_area_l582_582526


namespace speed_conversion_l582_582042

theorem speed_conversion (speed_m_s : ℚ) (conversion_factor : ℚ) : 
  let speed_km_h := speed_m_s * conversion_factor in
  speed_m_s = 13/54 → conversion_factor = 3.6 → 
  (Float.ofRat speed_km_h).round (some 2) = 0.87 := by
    intros h_speed h_conversion
    -- Convert the speed from m/s to km/h
    let speed_km_h := speed_m_s * conversion_factor
    have h_correct : (Float.ofRat speed_km_h).round (some 2) = 0.87 := 
      sorry
    exact h_correct

end speed_conversion_l582_582042


namespace log_difference_example_l582_582398

theorem log_difference_example :
  ∀ (log : ℕ → ℝ),
    log 3 * 24 - log 3 * 8 = 1 := 
by
sorry

end log_difference_example_l582_582398


namespace possible_values_of_a_l582_582848

theorem possible_values_of_a (a : ℝ) :
  (∃ x, ∀ y, (y = x) ↔ (a * y^2 + 2 * y + a = 0))
  → (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  sorry

end possible_values_of_a_l582_582848


namespace probability_vector_length_leq_6_leq_3_sqrt_5_div_10_l582_582091

noncomputable def vector_length (alpha n : ℝ) : ℝ :=
  (2 * n + 3 * real.cos alpha)^2 + (n - 3 * real.sin alpha)^2

theorem probability_vector_length_leq_6_leq_3_sqrt_5_div_10 (alpha : ℝ) :
  ∃ p, p = 3 * real.sqrt 5 / 10 ∧ ∀ n, 0 ≤ n ∧ n ≤ 2 → vector_length alpha n ≤ 6 → 
  (p = (3 * real.sqrt 5) / 10) :=
begin
  sorry
end

end probability_vector_length_leq_6_leq_3_sqrt_5_div_10_l582_582091


namespace problem_statement_l582_582423

def exists_set_10_distinct_pos_ints_no_6_sum_div_6 : Prop :=
  ∃ S : Finset ℕ, S.card = 10 ∧ (∀ T ⊆ S, T.card = 6 → (T.sum % 6 ≠ 0))

def not_exists_set_11_distinct_pos_ints_no_6_sum_div_6 : Prop :=
  ¬ ∃ S : Finset ℕ, S.card = 11 ∧ (∀ T ⊆ S, T.card = 6 → (T.sum % 6 ≠ 0))

theorem problem_statement :
  exists_set_10_distinct_pos_ints_no_6_sum_div_6 ∧ not_exists_set_11_distinct_pos_ints_no_6_sum_div_6 :=
by {
  sorry -- proof to be added
}

end problem_statement_l582_582423


namespace Tim_movie_marathon_length_l582_582632

def movie_marathon (length1 length2 length3 total_time : ℕ) : Prop :=
  length1 = 2 ∧
  length2 = length1 + length1 / 2 ∧
  length3 = (length1 + length2) - 1 ∧
  total_time = length1 + length2 + length3

theorem Tim_movie_marathon_length : 
  ∃ length1 length2 length3 total_time, movie_marathon length1 length2 length3 total_time ∧ 
  total_time = 9 :=
by
  use 2, 3, 4, 9
  dsimp [movie_marathon]
  repeat { constructor }
  · rfl
  · norm_num
  · norm_num
  · norm_num

end Tim_movie_marathon_length_l582_582632


namespace tetrahedrons_from_triangular_prism_l582_582493

theorem tetrahedrons_from_triangular_prism : 
  let n := 6
  let choose4 := Nat.choose n 4
  let coplanar_cases := 3
  choose4 - coplanar_cases = 12 := by
  sorry

end tetrahedrons_from_triangular_prism_l582_582493


namespace parallel_planes_of_perpendicular_line_l582_582444

-- Definitions of planes and their properties

variable {α β : Plane}
variable {l : Line}

-- Conditions
axiom line_perpendicular_to_planes (l_perp_α : l ⊥ α) (l_perp_β : l ⊥ β) : Prop

-- Proof statement
theorem parallel_planes_of_perpendicular_line (l_perp_α : l ⊥ α) (l_perp_β : l ⊥ β) : α ∥ β :=
sorry

end parallel_planes_of_perpendicular_line_l582_582444


namespace sum_of_first_twelve_multiples_of_17_l582_582314

theorem sum_of_first_twelve_multiples_of_17 : 
  (∑ i in Finset.range 12, 17 * (i + 1)) = 1326 := 
by
  sorry

end sum_of_first_twelve_multiples_of_17_l582_582314


namespace largest_n_unique_k_l582_582658

-- Defining the main theorem statement
theorem largest_n_unique_k :
  ∃ (n : ℕ), (n = 63) ∧ (∃! (k : ℤ), (9 / 17 : ℚ) < (n : ℚ) / ((n + k) : ℚ) ∧ (n : ℚ) / ((n + k) : ℚ) < (8 / 15 : ℚ)) :=
sorry

end largest_n_unique_k_l582_582658


namespace segments_equal_l582_582152

-- Definitions
variables {O A B C D : Point}
variables {r₁ r₂ : ℝ} -- Radii of the smaller and larger circles
variable (Line : Points → Line)
variable (dist : (Point × Point) → ℝ)

-- Conditions for concentric circles and intersecting line
def concentric_circle1 (O : Point) (r₁ : ℝ) : Circle := Circle.mk O r₁
def concentric_circle2 (O : Point) (r₂ : ℝ) : Circle := Circle.mk O r₂

def line_intersects_circles (Line : Points → Line) (circle1 circle2 : Circle)
  := intersects Line circle1 ∧ intersects Line circle2

-- Problem to prove
theorem segments_equal
  (O A B C D : Point) (r₁ r₂ : ℝ) (Line : Points → Line)
  (dist : (Point × Point) → ℝ)
  (h1 : concentric_circle1 O r₁)
  (h2 : concentric_circle2 O r₂)
  (h3 : line_intersects_circles Line h1 h2) :
  dist (A, B) = dist (C,D) := 
sorry

end segments_equal_l582_582152


namespace positivity_of_xyz_l582_582333

variable {x y z : ℝ}

theorem positivity_of_xyz
  (h1 : x + y + z > 0)
  (h2 : xy + yz + zx > 0)
  (h3 : xyz > 0) :
  x > 0 ∧ y > 0 ∧ z > 0 := 
sorry

end positivity_of_xyz_l582_582333


namespace interval_length_bounds_l582_582407

theorem interval_length_bounds (a b : ℝ) (h1 : ∀ x, a ≤ x ∧ x ≤ b → 3^|x| ∈ Set.Icc 1 9) :
  (b - a = 4 ∨ b - a = 2) :=
sorry

end interval_length_bounds_l582_582407


namespace logging_problem_l582_582619

theorem logging_problem :
  ∃ (D P : ℝ), D + P = 850 ∧ 300 * D + 225 * P = 217500 ∧ D = 350 :=
by
  let D : ℝ := 350
  let P : ℝ := 500
  use [D, P]
  simp [D, P]
  sorry

end logging_problem_l582_582619


namespace ellipse_standard_equation_rm_dot_rn_constant_l582_582468

section EllipseProof

variable (C : Type*) [Ellipse C]
variable [HasFoci C Point]
variable [HasStandardEquation C Type]
variable [MovingPointOnEllipse C Point]
variable [HasVertices C Point]
variable [LineThroughPointWithSlope C Line Point]
variable (F1 F2 : Point) (P : Point) (R : Point) (M N : Point) (l2 : Line) (k : Real) (x y a b c : Real)

-- Given conditions
def condition1 : F1 = Point.mk (-2 * Real.sqrt 2) 0 := sorry
def condition2 : F2 = Point.mk (2 * Real.sqrt 2) 0 := sorry
def condition3 : |P - F1| + |P - F2| = 4 * Real.sqrt 3 := sorry
def condition4 : O = Point.mk 0 0 := sorry
def condition5 : R = Point.mk 0 (-2) := sorry
def condition6 : LineThroughPointWithSlope.mk (Point.mk 0 1) k = l2 := sorry

-- Proof problem for the standard equation of the ellipse
theorem ellipse_standard_equation : IsStandardEquation C (x / 12) (y / 4) := sorry

-- Proof problem for the constancy of \(\overrightarrow{RM} \cdot \overrightarrow{RN}\)
theorem rm_dot_rn_constant : ∀ k : Real, DotProduct (Vector.mk R M) (Vector.mk R N) = 0 := sorry

end EllipseProof

end ellipse_standard_equation_rm_dot_rn_constant_l582_582468


namespace solve_for_square_solve_for_cube_l582_582875

variable (x : ℂ)

-- Given condition
def condition := x + 1/x = 8

-- Prove that x^2 + 1/x^2 = 62 given the condition
theorem solve_for_square (h : condition x) : x^2 + 1/x^2 = 62 := 
  sorry

-- Prove that x^3 + 1/x^3 = 488 given the condition
theorem solve_for_cube (h : condition x) : x^3 + 1/x^3 = 488 :=
  sorry

end solve_for_square_solve_for_cube_l582_582875


namespace volume_of_inscribed_cube_l582_582363

theorem volume_of_inscribed_cube
    (a : ℝ) (r : ℝ) (s : ℝ)
    (h₁ : a = 12)
    (h₂ : r = a / 2)
    (h₃ : s * Real.sqrt 3 = 2 * r) :
  s ^ 3 = 192 * Real.sqrt 3 :=
by
  -- Assuming conditions
  have h₄ : r = 6 := by rw [h₁, div_eq_mul_inv, mul_inv_cancel_right₀]; exact is_unit.mul_left_injective r₀.of_ne_zero
  have h₅ : 2 * r = 12 := by rw [h₄, mul_comm 2 6]; exact 12
  have h₆ : s * Real.sqrt 3 = 12 := by rw [h₅, mul_comm]
  have h₇ : s = 4 * Real.sqrt 3 := by rw [h₆, mul_div_right_comm 4 3 ℝ]; exact is_unit.mul_right_injective r₀.of_ne_zero
  rw [← h₇, mul_pow 4 3]; exact 192 * Real.sqrt 3


end volume_of_inscribed_cube_l582_582363


namespace rice_grains_difference_l582_582736

theorem rice_grains_difference : 
  3^15 - (3^1 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9 + 3^10) = 14260335 := 
by
  sorry

end rice_grains_difference_l582_582736


namespace accurate_to_hundreds_place_l582_582097

def rounded_number : ℝ := 8.80 * 10^4

theorem accurate_to_hundreds_place
  (n : ℝ) (h : n = rounded_number) : 
  exists (d : ℤ), n = d * 100 ∧ |round n - n| < 50 :=
sorry

end accurate_to_hundreds_place_l582_582097


namespace effective_area_percentage_difference_is_1566_67_l582_582652

-- Definitions for the problems conditions
def ratio_radii := 4 / 10
def soil_quality_index_first := 0.8
def soil_quality_index_second := 1.2
def water_allocation_first := 15000
def water_allocation_second := 30000
def crop_yield_factor_first := 1.5
def crop_yield_factor_second := 2

-- Calculation of areas based on given ratios and simplifying to a relationship
def radius_first (x : ℝ) := 4 * x
def radius_second (x : ℝ) := 10 * x

def area_first (x : ℝ) := Mathlib.pi * (radius_first x)^2
def area_second (x : ℝ) := Mathlib.pi * (radius_second x)^2

-- Effective area calculations
def effective_area_first (x : ℝ) := area_first x * soil_quality_index_first * water_allocation_first * crop_yield_factor_first
def effective_area_second (x : ℝ) := area_second x * soil_quality_index_second * water_allocation_second * crop_yield_factor_second

-- Calculate effective area percentage difference
def percentage_difference (x : ℝ) := ((effective_area_second x - effective_area_first x) / effective_area_first x) * 100

theorem effective_area_percentage_difference_is_1566_67 (x : ℝ) : percentage_difference x = 1566.67 :=
by
   sorry

end effective_area_percentage_difference_is_1566_67_l582_582652


namespace total_distance_race_l582_582972

theorem total_distance_race
  (t_Sadie : ℝ) (s_Sadie : ℝ) (t_Ariana : ℝ) (s_Ariana : ℝ) 
  (s_Sarah : ℝ) (tt : ℝ)
  (h_Sadie : t_Sadie = 2) (hs_Sadie : s_Sadie = 3) 
  (h_Ariana : t_Ariana = 0.5) (hs_Ariana : s_Ariana = 6) 
  (hs_Sarah : s_Sarah = 4)
  (h_tt : tt = 4.5) : 
  (s_Sadie * t_Sadie + s_Ariana * t_Ariana + s_Sarah * (tt - (t_Sadie + t_Ariana))) = 17 := 
  by {
    sorry -- proof goes here
  }

end total_distance_race_l582_582972


namespace total_ages_l582_582405

variable (Craig_age Mother_age : ℕ)

theorem total_ages (h1 : Craig_age = 16) (h2 : Mother_age = Craig_age + 24) : Craig_age + Mother_age = 56 := by
  sorry

end total_ages_l582_582405


namespace cosine_arithmetic_sequence_l582_582587

theorem cosine_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_sum : (∑ i in Finset.range 15, a (i + 1)) = 5 * Real.pi) :
  Real.cos (a 4 + a 12) = -1 / 2 := 
sorry

end cosine_arithmetic_sequence_l582_582587


namespace hyperbola_eccentricity_proof_l582_582464

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_eq : c^2 / a^2 - 4 = 1) : ℝ :=
  real.sqrt 5

theorem hyperbola_eccentricity_proof (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
  (h_eq : c^2 / a^2 - 4 = 1) : hyperbola_eccentricity a b c h_a_pos h_b_pos h_eq = real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_proof_l582_582464


namespace number_of_blueberries_l582_582338

def total_berries : ℕ := 42
def raspberries : ℕ := total_berries / 2
def blackberries : ℕ := total_berries / 3
def blueberries : ℕ := total_berries - (raspberries + blackberries)

theorem number_of_blueberries :
  blueberries = 7 :=
by
  sorry

end number_of_blueberries_l582_582338


namespace pens_left_is_25_l582_582000

def total_pens_left (initial_blue initial_black initial_red removed_blue removed_black : Nat) : Nat :=
  let blue_left := initial_blue - removed_blue
  let black_left := initial_black - removed_black
  let red_left := initial_red
  blue_left + black_left + red_left

theorem pens_left_is_25 :
  total_pens_left 9 21 6 4 7 = 25 :=
by 
  rw [total_pens_left, show 9 - 4 = 5 from Nat.sub_eq_of_eq_add (rfl), show 21 - 7 = 14 from Nat.sub_eq_of_eq_add (rfl)]
  rfl

end pens_left_is_25_l582_582000


namespace household_count_correct_l582_582678

def num_buildings : ℕ := 4
def floors_per_building : ℕ := 6
def households_first_floor : ℕ := 2
def households_other_floors : ℕ := 3
def total_households : ℕ := 68

theorem household_count_correct :
  num_buildings * (households_first_floor + (floors_per_building - 1) * households_other_floors) = total_households :=
by
  sorry

end household_count_correct_l582_582678


namespace bathing_suits_total_l582_582682

def men_bathing_suits : ℕ := 14797
def women_bathing_suits : ℕ := 4969
def total_bathing_suits : ℕ := 19766

theorem bathing_suits_total :
  men_bathing_suits + women_bathing_suits = total_bathing_suits := by
  sorry

end bathing_suits_total_l582_582682


namespace sarah_problem_sum_l582_582975

theorem sarah_problem_sum (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 100 ≤ y ∧ y < 1000) (h : 1000 * x + y = 9 * x * y) :
  x + y = 126 :=
sorry

end sarah_problem_sum_l582_582975


namespace solve_inequality_l582_582132

def is_odd (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f (x)

def given_function (x : ℝ) : ℝ :=
  if x > 0 then 1 - Real.logb 2 x else if x = 0 then 0 else Real.logb 2 (-x) - 1

theorem solve_inequality :
  is_odd given_function →
  (∀ x > 0, given_function x = 1 - Real.logb 2 x) →
  {x : ℝ | given_function x ≤ 0} = set.Icc (-2 : ℝ) 0 ∪ set.Ici 2 :=
by
  sorry

end solve_inequality_l582_582132


namespace f_of_x_l582_582556

theorem f_of_x (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x-1) = 3*x - 1) : ∀ x : ℤ, f x = 3*x + 2 :=
by
  sorry

end f_of_x_l582_582556


namespace length_of_arcs_l582_582173

variables (O : Type) [MetricSpace O] (circle : O) (S P X : O)
variables (OS SIP SXP : ℝ) (h_OS : OS = 12) (h_SIP : SIP = 48) (h_SXP : SXP = 24)

theorem length_of_arcs 
  (h_circle_center : ∃ O : O, True) -- Assuming presence of the center of the circle
  (h_SIP : ∠ SIP = 48) 
  (h_OS : OS = 12) 
  (h_SXP : ∠ SXP = 24) : 
  (arc_length SP = 6.4 * π) ∧ (arc_length SXP = 3.2 * π) := 
by sorry

end length_of_arcs_l582_582173


namespace find_a_for_odd_function_l582_582222

noncomputable def f (a x : ℝ) : ℝ := ((x + 1) * (x + a)) / x

theorem find_a_for_odd_function (a : ℝ) :
  (∀ x : ℝ, f a x + f a (-x) = 0) ↔ a = -1 := sorry

end find_a_for_odd_function_l582_582222


namespace sum_of_altitudes_of_triangle_l582_582268

theorem sum_of_altitudes_of_triangle (x y : ℝ) (hline : 10 * x + 8 * y = 80) :
  let x_intercept := 80 / 10
      y_intercept := 80 / 8
      area := (1 / 2) * x_intercept * y_intercept
      altitude1 := x_intercept
      altitude2 := y_intercept
      altitude3 := 40 / Real.sqrt 41 in
  altitude1 + altitude2 + altitude3 = (18 * Real.sqrt 41 + 40) / Real.sqrt 41 :=
by
  let x_intercept := 80 / 10
  let y_intercept := 80 / 8
  let area := (1 / 2) * x_intercept * y_intercept
  let altitude1 := x_intercept
  let altitude2 := y_intercept
  let altitude3 := 40 / Real.sqrt 41
  sorry

end sum_of_altitudes_of_triangle_l582_582268


namespace probability_prime_sums_is_one_third_l582_582300

/- This problem deals with probabilities on the sums of two spinners having prime outcomes. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def outcomes_spinner1 := {1, 4, 6}
def outcomes_spinner2 := {1, 3, 5}

def possible_sums := {x + y | x in outcomes_spinner1, y in outcomes_spinner2}

def prime_sums := {n | n ∈ possible_sums ∧ is_prime n}

def probability_of_prime_sums :=
  prime_sums.card.to_rat / possible_sums.card.to_rat

theorem probability_prime_sums_is_one_third :
  probability_of_prime_sums = 1 / 3 := 
sorry

end probability_prime_sums_is_one_third_l582_582300


namespace no_such_i_exists_l582_582048

def is_perfect_cube (i : ℕ) : Prop :=
  ∃ m : ℕ, i = m^3

def f (i : ℕ) : ℕ := 1 + i + (i^(1/3))

theorem no_such_i_exists : ∀ (i : ℕ), (1 ≤ i ∧ i ≤ 3000) → is_perfect_cube i → f i ≠ 1 + i + ⌊(i : ℝ)^(1/3)⌋ := 
by {
  intros i h_range h_cube,
  sorry
}

end no_such_i_exists_l582_582048


namespace find_reflected_ray_eq_l582_582701

noncomputable def reflected_line_eq (A B A' : ℝ × ℝ) (hA : A = (-1/2, 0)) (hB : B = (0, 1)) (hA' : A' = (1/2, 0)) : Prop :=
  ∃ (a b c : ℝ), a * (A'.fst) + b * (A'.snd) + c = 0 ∧ a * (B.fst) + b * (B.snd) + c = 0 ∧
  ∀ (x y : ℝ), a * x + b * y + c = 0 → 2 * x + y - 1 = 0

theorem find_reflected_ray_eq :
  reflected_line_eq (-1/2, 0) (0, 1) (1/2, 0) (-1/2, 0) (0, 1) (1/2, 0) :=
begin
  sorry
end

end find_reflected_ray_eq_l582_582701


namespace num_ways_to_select_three_sum_divisible_by_three_l582_582582

def A : Finset ℕ := {1, 2, 3, ..., 20}

def B : Finset ℕ := {3, 6, 9, 12, 15, 18}
def C : Finset ℕ := {1, 4, 7, 10, 13, 16, 19}
def D : Finset ℕ := {2, 5, 8, 11, 14, 17, 20}

theorem num_ways_to_select_three_sum_divisible_by_three :
  (B.card.choose 3 + C.card.choose 3 + D.card.choose 3 + B.card * C.card * D.card = 384) :=
  sorry

end num_ways_to_select_three_sum_divisible_by_three_l582_582582


namespace f_2010_of_8_l582_582159

/-- sum_digits is a helper function to calculate the sum of the digits of a natural number -/
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- f is defined as the sum of the digits of (n^2 + 1) -/
def f (n : ℕ) : ℕ :=
  sum_digits (n^2 + 1)

def f_iter (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, n := n
| (k+1), n := f (f_iter f k n)

/-- We are asked to prove that f_{2010}(8) = 8 -/
theorem f_2010_of_8 : f_iter f 2010 8 = 8 :=
by sorry

end f_2010_of_8_l582_582159


namespace area_of_enclosed_shape_l582_582593

noncomputable def areaEnclosedByCurves : ℝ :=
  ∫ x in (-2:ℝ)..(1:ℝ), (2 - x^2 - x)

theorem area_of_enclosed_shape :
  areaEnclosedByCurves = 9 / 2 :=
by
  sorry

end area_of_enclosed_shape_l582_582593


namespace sin_double_angle_l582_582823

theorem sin_double_angle (α : ℝ) (h1 : sin (π / 4 - α) = 3 / 5) (h2 : 0 < α ∧ α < π / 4) : 
  sin (2 * α) = 7 / 25 :=
by
  sorry

end sin_double_angle_l582_582823


namespace find_third_number_l582_582357

theorem find_third_number (x : ℕ) (h : (6 + 16 + x) / 3 = 13) : x = 17 :=
by
  sorry

end find_third_number_l582_582357


namespace janet_owes_wages_and_taxes_l582_582545

theorem janet_owes_wages_and_taxes :
  (∀ (workdays : ℕ) (hours : ℕ) (warehouse_workers : ℕ) (manager_workers : ℕ) (warehouse_wage : ℕ) (manager_wage : ℕ) (tax_rate : ℚ),
    workdays = 25 →
    hours = 8 →
    warehouse_workers = 4 →
    manager_workers = 2 →
    warehouse_wage = 15 →
    manager_wage = 20 →
    tax_rate = 0.1 →
    let total_hours := workdays * hours
        warehouse_monthly := total_hours * warehouse_wage
        manager_monthly := total_hours * manager_wage
        total_wage := warehouse_monthly * warehouse_workers + manager_monthly * manager_workers
        total_taxes := total_wage * tax_rate in
    total_wage + total_taxes = 22000) :=
begin
  intros,
  rw [← mul_assoc, mul_comm 25 8, mul_assoc],
  have h1 : 25 * 8 = 200, {norm_num},
  rw h1,
  have h2 : 200 * 15 * 4 = 12000, {norm_num},
  have h3 : 200 * 20 * 2 = 8000, {norm_num},
  rw [h2, h3],
  have h4 : 12000 + 8000 = 20000, {norm_num},
  have h5 : 20000 * 0.1 = 2000, {norm_num},
  rw [h4, h5],
  norm_num,
end

end janet_owes_wages_and_taxes_l582_582545


namespace segment_movement_sweep_reduction_l582_582296

noncomputable def segment_move_area_proof (AB d : ℝ) : Prop :=
  ∃ α : ℝ, (α * AB * AB < (AB * d / 10000))

theorem segment_movement_sweep_reduction (AB CD d : ℝ) (e f : set ℝ)
  (h_parallel : ∀ x ∈ e, ∀ y ∈ f, x - y = d) :
  segment_move_area_proof AB d :=
sorry

end segment_movement_sweep_reduction_l582_582296


namespace total_amount_l582_582017

theorem total_amount (W X Y Z : ℝ) (h1 : X = 0.8 * W) (h2 : Y = 0.65 * W) (h3 : Z = 0.45 * W) (h4 : Y = 78) : 
  W + X + Y + Z = 348 := by
  sorry

end total_amount_l582_582017


namespace Euler_polyhedron_no_convex_polyhedron_all_faces_gt_5_sides_no_convex_polyhedron_all_vertices_gt_5_faces_l582_582325

-- Definition of Euler's formula for a polyhedron.
theorem Euler_polyhedron (f p a : ℕ) (h : f + p - a = 2) : True :=
by trivial

-- Part (a): Prove there does not exist a convex polyhedron in which all faces have more than 5 sides.
theorem no_convex_polyhedron_all_faces_gt_5_sides :
  ¬ ∃ (f p a : ℕ), (∀ i, i ≤ 5 → faces_with_i_sides f 0) ∧ Euler_polyhedron f p a := 
sorry

-- Part (b): Prove there does not exist a convex polyhedron in which all polyhedral angles have more than 5 faces.
theorem no_convex_polyhedron_all_vertices_gt_5_faces :
  ¬ ∃ (f p a : ℕ), (∀ i, i ≤ 5 → vertices_with_i_faces p 0) ∧ Euler_polyhedron f p a :=
sorry


end Euler_polyhedron_no_convex_polyhedron_all_faces_gt_5_sides_no_convex_polyhedron_all_vertices_gt_5_faces_l582_582325


namespace concert_attendance_l582_582714

open Real

theorem concert_attendance :
  (∀ P : ℝ, (2 / 3 * P) * (1 / 2) * (2 / 5) = 20 → P = 150) :=
by
  intro P
  intro h
  have h1 : (2 / 3) * (1 / 2) = 1 / 3 := by rw [mul_div₀, one_div, mul_one]
  have h2 : (1 / 3) * (2 / 5) = 2 / 15 := by rw [mul_comm_div·trans]
  rw h1 at h
  rw h2 at h
  sorry

end concert_attendance_l582_582714


namespace union_sets_l582_582115

def M (a : ℕ) : Set ℕ := {a, 0}
def N : Set ℕ := {1, 2}

theorem union_sets (a : ℕ) (h_inter : M a ∩ N = {2}) : M a ∪ N = {0, 1, 2} :=
by
  sorry

end union_sets_l582_582115


namespace find_circle_l582_582614

def circle_equation := ∃ (a : ℝ), 
  ((x - (2 + (2 * Real.sqrt 10)))^2 + (y - 4)^2 = 16) ∨
  ((x - (2 - (2 * Real.sqrt 10)))^2 + (y - 4)^2 = 16) ∨
  ((x - (2 + (2 * Real.sqrt 6)))^2 + (y + 4)^2 = 16) ∨
  ((x - (2 - (2 * Real.sqrt 6)))^2 + (y + 4)^2 = 16)

theorem find_circle :
  ∃ (x y : ℝ), 
  (radius = 4) ∧ 
  (tangent_line y 0) ∧ 
  (tangent_circle (x-2)^2 + (y-1)^2 - 9)) → circle_equation
:= by
  sorry

end find_circle_l582_582614


namespace segment_ratios_l582_582893

variables {A B C D A1 B1 C1 D1 M N : ℝ}
variables (AC BD1 : ℝ)
variables (angle_NMC angle_MNB : ℝ)

noncomputable def cube_edge_length := 1
noncomputable def point_on_AC (M : ℝ) : Prop := M ∈ Icc 0 AC
noncomputable def point_on_BD1 (N : ℝ) : Prop := N ∈ Icc 0 BD1
def angle_60_degrees : ℝ := 60
def angle_45_degrees : ℝ := 45

theorem segment_ratios
  (cube : ℝ = cube_edge_length)
  (M_on_AC : point_on_AC M)
  (N_on_BD1 : point_on_BD1 N)
  (angle_NMC_eq : angle_NMC = angle_60_degrees)
  (angle_MNB_eq : angle_MNB = angle_45_degrees) :
  let x := 1 / Real.sqrt 6,
  let y := 2 / Real.sqrt 3 in
  (| AM : AC = 2 - Real.sqrt 3) ∧ (| BN : BD1 = 2) := sorry

end segment_ratios_l582_582893


namespace range_of_x_l582_582830

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(-x)

def monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ a b, 0 ≤ a → 0 ≤ b → a < b → f(a) < f(b)

theorem range_of_x (h_even : even_function f) (h_monotone : monotone_increasing_on_nonneg f) :
  ∀ x, f(2 * x - 1) < f(1) → (0 < x ∧ x < 1) :=
sorry

end range_of_x_l582_582830


namespace probability_heart_king_l582_582651

theorem probability_heart_king :
  let total_cards := 52
  let total_kings := 4
  let hearts_count := 13
  let king_of_hearts := 1 in
  let prob_king_of_hearts_first := (1 : ℚ) / total_cards
  let prob_other_heart_first := (hearts_count - king_of_hearts : ℚ) / total_cards
  let prob_king_second_if_king_heart_first := (total_kings - king_of_hearts : ℚ) / (total_cards - 1)
  let prob_king_second_if_other_heart_first := (total_kings : ℚ) / (total_cards - 1) in
  prob_king_of_hearts_first * prob_king_second_if_king_heart_first +
  prob_other_heart_first * prob_king_second_if_other_heart_first = (1 : ℚ) / total_cards :=
by sorry

end probability_heart_king_l582_582651


namespace determine_g_l582_582270

theorem determine_g (t : ℝ) : ∃ (g : ℝ → ℝ), (∀ x y, y = 2 * x - 40 ∧ y = 20 * t - 14 → g t = 10 * t + 13) :=
by
  sorry

end determine_g_l582_582270


namespace problem_statement_l582_582982

noncomputable def prove_parallel (A B C P₁ P₂ P₃ P₄ P₅ P₆ : Point) (h₁ : P₁ ∈ line_segment B C)
  (h₂ : parallel (line_segment P₁ P₂) (line_segment A C)) (h₃ : P₂ ∈ line_segment A B)
  (h₄ : parallel (line_segment P₂ P₃) (line_segment B C)) (h₅ : P₃ ∈ line_segment C A)
  (h₆ : parallel (line_segment P₃ P₄) (line_segment A B)) (h₇ : P₄ ∈ line_segment B C)
  (h₈ : parallel (line_segment P₄ P₅) (line_segment C A)) (h₉ : P₅ ∈ line_segment A B)
  (h₁₀ : parallel (line_segment P₅ P₆) (line_segment B C)) (h₁₁ : P₆ ∈ line_segment C A) : Prop :=
  parallel (line_segment P₆ P₁) (line_segment A B)

theorem problem_statement (A B C P₁ P₂ P₃ P₄ P₅ P₆ : Point)
  (h₁ : P₁ ∈ line_segment B C)
  (h₂ : parallel (line_segment P₁ P₂) (line_segment A C)) (h₃ : P₂ ∈ line_segment A B)
  (h₄ : parallel (line_segment P₂ P₃) (line_segment B C)) (h₅ : P₃ ∈ line_segment C A)
  (h₆ : parallel (line_segment P₃ P₄) (line_segment A B)) (h₇ : P₄ ∈ line_segment B C)
  (h₈ : parallel (line_segment P₄ P₅) (line_segment C A)) (h₉ : P₅ ∈ line_segment A B)
  (h₁₀ : parallel (line_segment P₅ P₆) (line_segment B C)) (h₁₁ : P₆ ∈ line_segment C A) :
  prove_parallel A B C P₁ P₂ P₃ P₄ P₅ P₆ h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈ h₉ h₁₀ h₁₁ :=
sorry

end problem_statement_l582_582982


namespace Meghan_total_money_l582_582954

theorem Meghan_total_money (h100 : ℕ) (h50 : ℕ) (h10 : ℕ) : 
  h100 = 2 → h50 = 5 → h10 = 10 → 100 * h100 + 50 * h50 + 10 * h10 = 550 :=
by
  sorry

end Meghan_total_money_l582_582954


namespace sum_converges_to_1_l582_582087

noncomputable def sum_of_series : ℕ → ℝ 
| n := ∑ k in finset.range n, (1 : ℝ) / (↑k * (↑k + 1))

theorem sum_converges_to_1 : 
  ∃ l : ℝ, tendsto sum_of_series at_top (𝓝 l) ∧ l = 1 := 
by
  sorry

end sum_converges_to_1_l582_582087


namespace mean_proportional_between_234_and_104_l582_582328

def mean_proportional (a b : ℕ) : ℝ :=
  Real.sqrt (a * b)

theorem mean_proportional_between_234_and_104 :
  mean_proportional 234 104 = 156 :=
by
  sorry

end mean_proportional_between_234_and_104_l582_582328


namespace thomas_lost_pieces_l582_582029

theorem thomas_lost_pieces (audrey_lost : ℕ) (total_pieces_left : ℕ) (initial_pieces_each : ℕ) (total_pieces_initial : ℕ) (audrey_remaining_pieces : ℕ) (thomas_remaining_pieces : ℕ) : 
  audrey_lost = 6 → total_pieces_left = 21 → initial_pieces_each = 16 → total_pieces_initial = 32 → 
  audrey_remaining_pieces = initial_pieces_each - audrey_lost → 
  thomas_remaining_pieces = total_pieces_left - audrey_remaining_pieces → 
  initial_pieces_each - thomas_remaining_pieces = 5 :=
by
  sorry

end thomas_lost_pieces_l582_582029


namespace sum_of_arithmetic_subsequence_l582_582817

noncomputable def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a (n) + d

theorem sum_of_arithmetic_subsequence (a : ℕ → ℤ) (d : ℤ) (h1 : d = -2)
  (h2 : arithmetic_sequence a)
  (h3 : ∑ i in (finset.range 33).map (function.embedding.subtype (λ n, true)), a (3 * i + 1) = 50) :
  ∑ i in (finset.range 33).map (function.embedding.subtype (λ n, true)), a (3 * i + 3) = -66 :=
sorry

end sum_of_arithmetic_subsequence_l582_582817


namespace lost_words_due_to_forbidden_seventh_letter_l582_582187

def number_of_letters : ℕ := 65
def is_seventh_letter_forbidden : Prop := true

theorem lost_words_due_to_forbidden_seventh_letter
  (h1 : number_of_letters = 65)
  (h2 : is_seventh_letter_forbidden) :
  (1 + 2 * (number_of_letters - 1) ) * (number_of_letters - 1) - 8278 + 131 :=           -- Used brute force calculation to provide correct proof -
by sorry

end lost_words_due_to_forbidden_seventh_letter_l582_582187


namespace number_of_sides_of_polygon_l582_582882

theorem number_of_sides_of_polygon (n : ℕ) (h : (n - 2) * 180 = 540) : n = 5 :=
by
  sorry

end number_of_sides_of_polygon_l582_582882


namespace water_evaporation_l582_582365

def initial_volume : ℝ := 10000
def initial_percentage : ℝ := 0.05
def final_percentage : ℝ := 1 / 9

theorem water_evaporation (x : ℝ) : 500 / (initial_volume - x) = final_percentage → x = 5500 :=
by
  sorry

end water_evaporation_l582_582365


namespace range_f_range_m_l582_582137

noncomputable def f (x: ℝ) : ℝ := |(x - 2)| - |(x - 5)|

theorem range_f :
  (∀ x: ℝ, f(x) ∈ set.Icc (-3) 3) :=
begin
  sorry
end

theorem range_m (m: ℝ) :
  (∀ x: ℝ, f(x) + 2 * m - 1 ≥ 0) ↔ (2 ≤ m) :=
begin
  sorry
end

end range_f_range_m_l582_582137


namespace symmetry_axis_find_b_l582_582141

noncomputable def f : ℝ → ℝ := λ x, sqrt 3 * sin x * cos x - cos x ^ 2 - 1 / 2

-- 1. Proving the equation of the symmetry axis
theorem symmetry_axis : ∃ k : ℤ, ∀ x, f x = f (x + k * (π/2) + (π/3)) := 
sorry

noncomputable def g : ℝ → ℝ := λ x, sin (x + π / 6) - 1

-- 2. Given ΔABC with a = 2, c = 4 and g(B) = 0, proving that b = 2√3
theorem find_b (a c : ℝ) (A B C : ℝ) : 
  a = 2 → c = 4 → g B = 0 → B = π / 3 → (a^2 + c^2 - 2 * a * c * cos B) = (2 * sqrt 3)^2 :=
sorry

end symmetry_axis_find_b_l582_582141


namespace geometric_sequence_increasing_condition_l582_582122

theorem geometric_sequence_increasing_condition (a₁ a₂ a₄ : ℝ) (q : ℝ) (n : ℕ) (a : ℕ → ℝ):
  (∀ n, a n = a₁ * q^n) →
  (a₁ < a₂ ∧ a₂ < a₄) → 
  ¬ (∀ n, a n < a (n + 1)) → 
  (a₁ < a₂ ∧ a₂ < a₄) ∧ ¬ (∀ n, a n < a (n + 1)) :=
sorry

end geometric_sequence_increasing_condition_l582_582122


namespace proposition_C_true_single_correct_proposition_l582_582377

variables {m n : Type} -- Define m and n as types representing lines
variables {α : Type} -- Define α as a type representing a plane

-- Conditions for Proposition C
variables (m_proj : m → α → Option (m × α)) -- Projection of m onto α
variables (n_proj : n → α → Option (n × α)) -- Projection of n onto α
variables (perp_m_n : m → n → Prop) -- m is perpendicular to n
variables (perp_p : n → α → Prop) -- n is perpendicular to plane α
variables (inc_p : n → α → Prop) -- n is contained in plane α
variables (par_p : n → α → Prop) -- n is parallel to plane α
variables (point_on_alpha : m → α → Prop) -- Projection of m is a point on α

-- Proposition C statement
theorem proposition_C_true :
  (∀ m n α, -- for any lines m, n and plane α
    (point_on_alpha m_proj) ∧ (par_p n_proj) ∧
    (perp_m_n m n) → (inc_p n α ∨ par_p n α)) :=
begin
  sorry,
end

-- Add the variables' projections
axiom m_projection : ∀ m α, ∃ (p : α), point_on_alpha m α
axiom n_projection : ∀ n α, ∃ (p : α), par_p n α
axiom perpendicular_lines : ∃ m n, perp_m_n m n

-- Add the theorem stating that Proposition C is the only correct one
theorem single_correct_proposition :
  (prop_C_true ∧ ¬ prop_A ∧ ¬ prop_B ∧ ¬ prop_D) :=
begin
  sorry,
end

end proposition_C_true_single_correct_proposition_l582_582377


namespace average_salary_8800_l582_582594

theorem average_salary_8800 
  (average_salary_start : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_salary : ℝ)
  (avg_specific_months : ℝ)
  (jan_salary_rate : average_salary_start * 4 = total_salary)
  (may_salary_rate : total_salary - salary_jan = total_salary - 3300)
  (final_salary_rate : total_salary - salary_jan + salary_may = 35200)
  (specific_avg_calculation : 35200 / 4 = avg_specific_months)
  : avg_specific_months = 8800 :=
sorry -- Proof steps will be filled in later

end average_salary_8800_l582_582594


namespace number_of_sequences_18_l582_582563

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def T : Triangle := {
  p1 := ⟨0, 0⟩,
  p2 := ⟨5, 0⟩,
  p3 := ⟨0, 4⟩
}

def rotate_90 (p : Point) : Point := ⟨-p.y, p.x⟩
def rotate_270 (p : Point) : Point := ⟨p.y, -p.x⟩
def reflect_x (p : Point) : Point := ⟨p.x, -p.y⟩
def reflect_y (p : Point) : Point := ⟨-p.x, p.y⟩
def translate_54 (p : Point) : Point := ⟨p.x + 5, p.y + 4⟩

def transformation_sequence_returns_to_original (seq : List (Point → Point)) (T : Triangle) : Bool :=
  let T' := { T with 
    p1 := seq.foldr (· ∘ ·) id T.p1, 
    p2 := seq.foldr (· ∘ ·) id T.p2, 
    p3 := seq.foldr (· ∘ ·) id T.p3 
  }
  T' = T

theorem number_of_sequences_18 : 
  (List.permutations_of (List.replicate 243 ↑[rotate_90, rotate_270, reflect_x, reflect_y, translate_54]).map (List.take 3)).count (transformation_sequence_returns_to_original T) = 18 := sorry

end number_of_sequences_18_l582_582563


namespace find_ordered_pairs_l582_582425

theorem find_ordered_pairs (x y : ℤ) : 
  (3^x * 4^y = 2^(x+y) + 2^(2*(x+y)-1)) ↔ ((x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 1)) := by
  sorry

end find_ordered_pairs_l582_582425


namespace lemniscate_properties_l582_582981

noncomputable def lemniscate (a : ℝ) (x y : ℝ) : Prop :=
  ((x + a) ^ 2 + y ^ 2) * ((x - a) ^ 2 + y ^ 2) = a ^ 4

theorem lemniscate_properties :
  let a := 1 in
  lemma_eq (x y : ℝ) : (x^2 + y^2)^2 = 2 * (x^2 - y^2) ↔ lemniscate a x y
  ∧ (∀ x y : ℝ, lemniscate a x y → lemniscate a (-x) y ∧ lemniscate a x (-y))
  ∧ (∃ P : ℝ × ℝ, lemniscate a P.1 P.2 ∧ dist P (-(1:ℝ), 0) = dist P (1, 0)) := 
sorry

end lemniscate_properties_l582_582981


namespace f_increasing_f_inequality_interval_m_range_l582_582829

variable {f : ℝ → ℝ}

-- Condition: f is odd, f(1) = 1, and for x,y in [-1,1] with x + y ≠ 0, (x + y) * (f(x) + f(y)) > 0
axiom f_odd : ∀ x, f(-x) = -f(x)
axiom f_at_1 : f(1) = 1
axiom f_condition : ∀ x y, x ∈ Icc (-1 : ℝ) 1 → y ∈ Icc (-1 : ℝ) 1 → x + y ≠ 0 → (x + y) * (f(x) + f(y)) > 0

-- Problem 1: Prove f(x) is increasing on [-1, 1]
theorem f_increasing (x1 x2 : ℝ) (hx1 : x1 ∈ Icc (-1 : ℝ) 1) (hx2 : x2 ∈ Icc (-1 : ℝ) 1) (h : x1 < x2) : f(x1) < f(x2) :=
sorry

-- Problem 2: Find the interval where f(x + 1/2) < f(1 - 2x)
theorem f_inequality_interval : Icc (0 : ℝ) (1/6) = {x | f(x + 1/2) < f(1 - 2x)} :=
sorry

-- Problem 3: Find the range of m where f(x) ≤ m^2 - 2am + 1 for all x, a ∈ [-1, 1]
axiom f_bound : ∀ x (hx : x ∈ Icc (-1 : ℝ) 1), ∀ a, a ∈ Icc (-1 : ℝ) 1 → f(x) ≤ m^2 - 2 * a * m + 1
theorem m_range (m : ℝ) : m = 0 ∨ m ≤ -2 ∨ m ≥ 2 :=
sorry

end f_increasing_f_inequality_interval_m_range_l582_582829


namespace sequence_2023rd_term_is_153_l582_582602

def sum_cubes_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d, d ^ 3).sum

def sequence_term (n : ℕ) : ℕ :=
  (nat.iterate sum_cubes_of_digits n) n

theorem sequence_2023rd_term_is_153 : sequence_term 2023 = 153 := 
sorry

end sequence_2023rd_term_is_153_l582_582602


namespace maximum_sum_l582_582332

theorem maximum_sum (a b c d : ℕ) (h₀ : a < b ∧ b < c ∧ c < d)
  (h₁ : (c + d) + (a + b + c) = 2017) : a + b + c + d ≤ 806 :=
sorry

end maximum_sum_l582_582332


namespace probability_top_card_is_star_l582_582356

theorem probability_top_card_is_star :
  let total_cards := 65
  let suits := 5
  let ranks_per_suit := 13
  let star_cards := 13
  (star_cards / total_cards) = 1 / 5 :=
by
  sorry

end probability_top_card_is_star_l582_582356


namespace mean_difference_l582_582118

variable (a1 a2 a3 a4 a5 a6 A : ℝ)

-- Arithmetic mean of six numbers is A
axiom mean_six_numbers : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

-- Arithmetic mean of the first four numbers is A + 10
axiom mean_first_four : (a1 + a2 + a3 + a4) / 4 = A + 10

-- Arithmetic mean of the last four numbers is A - 7
axiom mean_last_four : (a3 + a4 + a5 + a6) / 4 = A - 7

-- Prove the arithmetic mean of the first, second, fifth, and sixth numbers differs from A by 3
theorem mean_difference :
  (a1 + a2 + a5 + a6) / 4 = A - 3 := 
sorry

end mean_difference_l582_582118


namespace no_b_satisfies_condition_l582_582408

noncomputable def f (b x : ℝ) : ℝ :=
  x^2 + 3 * b * x + 5 * b

theorem no_b_satisfies_condition :
  ∀ b : ℝ, ¬ (∃ x : ℝ, ∀ y : ℝ, |f b y| ≤ 5 → y = x) :=
by
  sorry

end no_b_satisfies_condition_l582_582408


namespace find_some_number_l582_582324

theorem find_some_number (some_number : ℝ) (h : (3.242 * some_number) / 100 = 0.045388) : some_number = 1.400 := 
sorry

end find_some_number_l582_582324


namespace seq_inequality_l582_582188

def sequence (a : ℕ → ℝ) (n : ℕ) :=
a 0 = 1 / 2 ∧ ∀ k < n, a (k + 1) = a k + (1 / n) * a k ^ 2

theorem seq_inequality (a : ℕ → ℝ) (n : ℕ) (h : sequence a n) :
1 - (1 / n) < a n ∧ a n < 1 :=
sorry

end seq_inequality_l582_582188


namespace sum_digits_9A_l582_582897

theorem sum_digits_9A (A : Nat) (digits : List Nat) (h_digits : ∀ i j, i < j → digits.nth i < digits.nth j) :
  (sum_digits (9 * A) = 9) := 
sorry

end sum_digits_9A_l582_582897


namespace number_of_subsets_of_starOperation_l582_582046

open Set

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 5}

def starOperation (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem number_of_subsets_of_starOperation :
  starOperation A B = {1, 7} →
  (λ S, S ⊆ starOperation A B) '' (univ : Set (Set ℕ)).toFinset.card = 4 := by
  intro h
  rw h
  exact sorry

end number_of_subsets_of_starOperation_l582_582046


namespace least_possible_integral_QR_l582_582298

theorem least_possible_integral_QR (PQ PR SR SQ QR : ℝ) (hPQ : PQ = 7) (hPR : PR = 10) (hSR : SR = 15) (hSQ : SQ = 24) :
  9 ≤ QR ∧ QR < 17 :=
by
  sorry

end least_possible_integral_QR_l582_582298


namespace tangent_line_ellipse_l582_582804

theorem tangent_line_ellipse (a b x y x₀ y₀ : ℝ) (h : a > 0) (hb : b > 0) (ha_gt_hb : a > b) 
(h_on_ellipse : (x₀^2 / a^2) + (y₀^2 / b^2) = 1) :
    (x₀ * x / a^2) + (y₀ * y / b^2) = 1 := 
sorry

end tangent_line_ellipse_l582_582804


namespace circle_area_greater_than_triangle_fraction_l582_582331

variable {R : ℝ}

/-- In triangle ABC, three circles with equal radii R touch two sides of the 
  triangle. One of the circles with center O₁ touches the other two circles 
  with centers O₂ and O₃ respectively and ∠ O₂ O₁ O₃ = 90°. Prove that the area
  of the circle with center O₁ is greater than one-fifth the area of the triangle ABC.-/
theorem circle_area_greater_than_triangle_fraction (hR : 0 < R)
  (h_angle : ∠ (O₂: ℝ × ℝ) (O₁: ℝ × ℝ) (O₃: ℝ × ℝ) = real.pi / 2)
  (h_tangent : ∀ P: ℝ × ℝ , (∃ Q: ℝ × ℝ, dist P Q = R ∧ P = (A + Q) / 2)) : 
  π * R^2 > (1/5) * (1/2 * (AB: ℝ) * (BC: ℝ)) :=
    by 
    sorry

end circle_area_greater_than_triangle_fraction_l582_582331


namespace counterexample_l582_582747

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem counterexample (n1 n2 : ℕ)
  (h1 : n1 = 14) (h2 : n2 = 20)
  (hn1 : ¬ is_perfect_square n1) (hn2 : ¬ is_perfect_square n2)
  (hp1 : ¬ is_prime (n1 + 4)) (hp2 : ¬ is_prime (n2 + 4)) : 
  ∃ (n : ℕ), ¬ is_perfect_square n ∧ ¬ is_prime (n + 4) :=
by
  existsi n1
  exact ⟨hn1, hp1⟩
  -- or
  existsi n2
  exact ⟨hn2, hp2⟩

end counterexample_l582_582747


namespace age_multiplier_l582_582004

theorem age_multiplier (S F M X : ℕ) (h1 : S = 27) (h2 : F = 48) (h3 : S + F = 75)
  (h4 : 27 - X = F - S) (h5 : F = M * X) : M = 8 :=
by
  -- Proof will be filled in here
  sorry

end age_multiplier_l582_582004


namespace min_value_pt_qu_rv_sw_l582_582217

theorem min_value_pt_qu_rv_sw (p q r s t u v w : ℝ) (h1 : p * q * r * s = 8) (h2 : t * u * v * w = 27) :
  (p * t) ^ 2 + (q * u) ^ 2 + (r * v) ^ 2 + (s * w) ^ 2 ≥ 96 :=
by
  sorry

end min_value_pt_qu_rv_sw_l582_582217


namespace rehabilitation_centers_l582_582664

def Lisa : ℕ := 6 
def Jude : ℕ := Lisa / 2
def Han : ℕ := 2 * Jude - 2
def Jane : ℕ := 27 - Lisa - Jude - Han
def x : ℕ := 2

theorem rehabilitation_centers:
  Jane = x * Han + 6 := 
by
  -- Proof goes here (not required)
  sorry

end rehabilitation_centers_l582_582664


namespace mabel_tomatoes_l582_582231

theorem mabel_tomatoes :
  ∃ (x : ℕ), (8 + (8 + x) + 2 * (48 + 3 * x) = 140) ∧ x = 4 :=
by
  use 4
  split
  · ring_nf
  · rfl
  sorry

end mabel_tomatoes_l582_582231
