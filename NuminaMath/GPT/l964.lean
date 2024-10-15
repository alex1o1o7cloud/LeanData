import Mathlib

namespace NUMINAMATH_GPT_simplify_and_evaluate_l964_96457

theorem simplify_and_evaluate (x : ℝ) (h : x^2 + 4 * x - 4 = 0) :
  3 * (x - 2) ^ 2 - 6 * (x + 1) * (x - 1) = 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l964_96457


namespace NUMINAMATH_GPT_pow_five_2010_mod_seven_l964_96436

theorem pow_five_2010_mod_seven :
  (5 ^ 2010) % 7 = 1 :=
by
  have h : (5 ^ 6) % 7 = 1 := sorry
  sorry

end NUMINAMATH_GPT_pow_five_2010_mod_seven_l964_96436


namespace NUMINAMATH_GPT_bananas_to_oranges_l964_96483

theorem bananas_to_oranges :
  (3 / 4) * 16 * (1 / 1 : ℝ) = 10 * (1 / 1 : ℝ) → 
  (3 / 5) * 15 * (1 / 1 : ℝ) = 7.5 * (1 / 1 : ℝ) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_bananas_to_oranges_l964_96483


namespace NUMINAMATH_GPT_buy_beams_l964_96442

theorem buy_beams (C T x : ℕ) (hC : C = 6210) (hT : T = 3) (hx: x > 0):
  T * (x - 1) = C / x :=
by
  rw [hC, hT]
  sorry

end NUMINAMATH_GPT_buy_beams_l964_96442


namespace NUMINAMATH_GPT_range_of_m_l964_96458

-- Define the constants used in the problem
def a : ℝ := 0.8
def b : ℝ := 1.2

-- Define the logarithmic inequality problem
theorem range_of_m (m : ℝ) : (a^(b^m) < b^(a^m)) → m < 0 := sorry

end NUMINAMATH_GPT_range_of_m_l964_96458


namespace NUMINAMATH_GPT_smallest_integer_of_inequality_l964_96473

theorem smallest_integer_of_inequality :
  ∃ x : ℤ, (8 - 7 * x ≥ 4 * x - 3) ∧ (∀ y : ℤ, (8 - 7 * y ≥ 4 * y - 3) → y ≥ x) ∧ x = 1 :=
sorry

end NUMINAMATH_GPT_smallest_integer_of_inequality_l964_96473


namespace NUMINAMATH_GPT_number_of_people_l964_96478

open Nat

theorem number_of_people (n : ℕ) (h : n^2 = 100) : n = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_people_l964_96478


namespace NUMINAMATH_GPT_find_a_range_of_a_l964_96466

noncomputable def f (x a : ℝ) := x + a * Real.log x

-- Proof problem 1: Prove that a = 2 given f' (1) = 3 for f (x) = x + a log x
theorem find_a (a : ℝ) : 
  (1 + a = 3) → (a = 2) := sorry

-- Proof problem 2: Prove that the range of a such that f(x) ≥ a always holds is [-e^2, 0]
theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≥ a) → (-Real.exp 2 ≤ a ∧ a ≤ 0) := sorry

end NUMINAMATH_GPT_find_a_range_of_a_l964_96466


namespace NUMINAMATH_GPT_count_integers_satisfy_inequality_l964_96402

theorem count_integers_satisfy_inequality : 
  ∃ l : List Int, (∀ n ∈ l, (n - 3) * (n + 5) < 0) ∧ l.length = 7 :=
by
  sorry

end NUMINAMATH_GPT_count_integers_satisfy_inequality_l964_96402


namespace NUMINAMATH_GPT_units_digit_34_pow_30_l964_96495

theorem units_digit_34_pow_30 :
  (34 ^ 30) % 10 = 6 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_34_pow_30_l964_96495


namespace NUMINAMATH_GPT_simplify_trig_l964_96499

theorem simplify_trig (x : ℝ) :
  (1 + Real.sin x + Real.cos x + Real.sqrt 2 * Real.sin x * Real.cos x) / 
  (1 - Real.sin x + Real.cos x - Real.sqrt 2 * Real.sin x * Real.cos x) = 
  1 + (Real.sqrt 2 - 1) * Real.tan (x / 2) :=
by 
  sorry

end NUMINAMATH_GPT_simplify_trig_l964_96499


namespace NUMINAMATH_GPT_volume_increase_factor_l964_96414

   variable (π : ℝ) (r h : ℝ)

   def original_volume : ℝ := π * r^2 * h

   def new_height : ℝ := 3 * h

   def new_radius : ℝ := 2.5 * r

   def new_volume : ℝ := π * (new_radius r)^2 * (new_height h)

   theorem volume_increase_factor :
     new_volume π r h = 18.75 * original_volume π r h := 
   by
     sorry
   
end NUMINAMATH_GPT_volume_increase_factor_l964_96414


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_range_of_a_l964_96404

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) * ((a / x) + a + 1)

theorem monotonic_decreasing_interval (a : ℝ) (h : a ≥ -1) :
  (a = -1 → ∀ x, x < -1 → f a x < f a (x + 1)) ∧
  (a ≠ -1 → (∀ x, -1 < a ∧ x < -1 ∨ x > 1 / (a + 1) → f a x < f a (x + 1)) ∧
                (∀ x, -1 < a ∧ -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 / (a + 1) → f a x < f a (x + 1)))
:= sorry

theorem range_of_a (a : ℝ) (h : a ≥ -1) :
  (∃ x1 x2, x1 > 0 ∧ x2 < 0 ∧ f a x1 < f a x2 → -1 ≤ a ∧ a < 0)
:= sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_range_of_a_l964_96404


namespace NUMINAMATH_GPT_Fabian_total_cost_correct_l964_96497

noncomputable def total_spent_by_Fabian (mouse_cost : ℝ) : ℝ :=
  let keyboard_cost := 2 * mouse_cost
  let headphones_cost := mouse_cost + 15
  let usb_hub_cost := 36 - mouse_cost
  let webcam_cost := keyboard_cost / 2
  let total_cost := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost + webcam_cost
  let discounted_total := total_cost * 0.90
  let final_total := discounted_total * 1.05
  final_total

theorem Fabian_total_cost_correct :
  total_spent_by_Fabian 20 = 123.80 :=
by
  sorry

end NUMINAMATH_GPT_Fabian_total_cost_correct_l964_96497


namespace NUMINAMATH_GPT_pencil_cost_is_4_l964_96437

variables (pencils pens : ℕ) (pen_cost total_cost : ℕ)

def total_pencils := 15 * 80
def total_pens := (2 * total_pencils) + 300
def total_pen_cost := total_pens * pen_cost
def total_pencil_cost := total_cost - total_pen_cost
def pencil_cost := total_pencil_cost / total_pencils

theorem pencil_cost_is_4
  (pen_cost_eq_5 : pen_cost = 5)
  (total_cost_eq_18300 : total_cost = 18300)
  : pencil_cost = 4 :=
by
  sorry

end NUMINAMATH_GPT_pencil_cost_is_4_l964_96437


namespace NUMINAMATH_GPT_compute_radii_sum_l964_96475

def points_on_circle (A B C D : ℝ × ℝ) (r : ℝ) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist A B) * (dist C D) = (dist A C) * (dist B D)

theorem compute_radii_sum :
  ∃ (r1 r2 : ℝ), points_on_circle (0,0) (-1,-1) (5,2) (6,2) r1
               ∧ points_on_circle (0,0) (-1,-1) (34,14) (35,14) r2
               ∧ r1 > 0
               ∧ r2 > 0
               ∧ r1 < r2
               ∧ r1^2 + r2^2 = 1381 :=
by {
  sorry -- proof not required
}

end NUMINAMATH_GPT_compute_radii_sum_l964_96475


namespace NUMINAMATH_GPT_find_k_l964_96407

theorem find_k (k : ℝ) (h : ∀ x : ℝ, x^2 + 10 * x + k = 0 → (∃ a : ℝ, a > 0 ∧ (x = -3 * a ∨ x = -a))) :
  k = 18.75 :=
sorry

end NUMINAMATH_GPT_find_k_l964_96407


namespace NUMINAMATH_GPT_group_B_population_calculation_l964_96470

variable {total_population : ℕ}
variable {sample_size : ℕ}
variable {sample_A : ℕ}
variable {total_B : ℕ}

theorem group_B_population_calculation 
  (h_total : total_population = 200)
  (h_sample_size : sample_size = 40)
  (h_sample_A : sample_A = 16)
  (h_sample_B : sample_size - sample_A = 24) :
  total_B = 120 :=
sorry

end NUMINAMATH_GPT_group_B_population_calculation_l964_96470


namespace NUMINAMATH_GPT_correlation_statements_l964_96447

variables {x y : ℝ}
variables (r : ℝ) (h1 : r > 0) (h2 : r = 1) (h3 : r = -1)

theorem correlation_statements :
  (r > 0 → (∀ x y, x > 0 → y > 0)) ∧
  (r = 1 ∨ r = -1 → (∀ x y, ∃ m b : ℝ, y = m * x + b)) :=
sorry

end NUMINAMATH_GPT_correlation_statements_l964_96447


namespace NUMINAMATH_GPT_Joyce_final_apples_l964_96456

def initial_apples : ℝ := 350.5
def apples_given_to_larry : ℝ := 218.7
def percentage_given_to_neighbors : ℝ := 0.375
def final_apples : ℝ := 82.375

theorem Joyce_final_apples :
  (initial_apples - apples_given_to_larry - percentage_given_to_neighbors * (initial_apples - apples_given_to_larry)) = final_apples :=
by
  sorry

end NUMINAMATH_GPT_Joyce_final_apples_l964_96456


namespace NUMINAMATH_GPT_solve_equation_l964_96409

theorem solve_equation : ∀ (x : ℝ), (2 * x + 5 = 3 * x - 2) → (x = 7) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l964_96409


namespace NUMINAMATH_GPT_quadratic_root_relationship_l964_96463

theorem quadratic_root_relationship (a b c : ℂ) (alpha beta : ℂ) (h1 : a ≠ 0) (h2 : alpha + beta = -b / a) (h3 : alpha * beta = c / a) (h4 : beta = 3 * alpha) : 3 * b ^ 2 = 16 * a * c := by
  sorry

end NUMINAMATH_GPT_quadratic_root_relationship_l964_96463


namespace NUMINAMATH_GPT_Newville_Academy_fraction_l964_96486

theorem Newville_Academy_fraction :
  let total_students := 100
  let enjoy_sports := 0.7 * total_students
  let not_enjoy_sports := 0.3 * total_students
  let say_enjoy_right := 0.75 * enjoy_sports
  let say_not_enjoy_wrong := 0.25 * enjoy_sports
  let say_not_enjoy_right := 0.85 * not_enjoy_sports
  let say_enjoy_wrong := 0.15 * not_enjoy_sports
  let say_not_enjoy_total := say_not_enjoy_wrong + say_not_enjoy_right
  let say_not_enjoy_but_enjoy := say_not_enjoy_wrong
  (say_not_enjoy_but_enjoy / say_not_enjoy_total) = (7 / 17) := by
  sorry

end NUMINAMATH_GPT_Newville_Academy_fraction_l964_96486


namespace NUMINAMATH_GPT_ratio_of_dad_to_jayson_l964_96413

-- Define the conditions
def JaysonAge : ℕ := 10
def MomAgeWhenBorn : ℕ := 28
def MomCurrentAge (JaysonAge : ℕ) (MomAgeWhenBorn : ℕ) : ℕ := MomAgeWhenBorn + JaysonAge
def DadCurrentAge (MomCurrentAge : ℕ) : ℕ := MomCurrentAge + 2

-- Define the proof problem
theorem ratio_of_dad_to_jayson (JaysonAge : ℕ) (MomAgeWhenBorn : ℕ)
  (h1 : JaysonAge = 10) (h2 : MomAgeWhenBorn = 28) :
  DadCurrentAge (MomCurrentAge JaysonAge MomAgeWhenBorn) / JaysonAge = 4 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_dad_to_jayson_l964_96413


namespace NUMINAMATH_GPT_find_a2016_l964_96453

-- Define the sequence according to the conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - a n

-- State the main theorem we want to prove
theorem find_a2016 :
  ∃ a : ℕ → ℤ, seq a ∧ a 2016 = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_a2016_l964_96453


namespace NUMINAMATH_GPT_eval_expression_l964_96493

theorem eval_expression : (5 + 2 + 6) * 2 / 3 - 4 / 3 = 22 / 3 := sorry

end NUMINAMATH_GPT_eval_expression_l964_96493


namespace NUMINAMATH_GPT_chef_bought_almonds_l964_96476

theorem chef_bought_almonds (total_nuts pecans : ℝ)
  (h1 : total_nuts = 0.52) (h2 : pecans = 0.38) :
  total_nuts - pecans = 0.14 :=
by
  sorry

end NUMINAMATH_GPT_chef_bought_almonds_l964_96476


namespace NUMINAMATH_GPT_arithmetic_geometric_l964_96440

theorem arithmetic_geometric (a : ℕ → ℤ) (d : ℤ) (h1 : d = 2)
  (h2 : ∀ n, a (n + 1) - a n = d)
  (h3 : ∃ r, a 1 * r = a 3 ∧ a 3 * r = a 4) :
  a 2 = -6 :=
by sorry

end NUMINAMATH_GPT_arithmetic_geometric_l964_96440


namespace NUMINAMATH_GPT_tourist_tax_l964_96492

theorem tourist_tax (total_value : ℝ) (non_taxable_amount : ℝ) (tax_rate : ℝ) 
  (h1 : total_value = 1720) (h2 : non_taxable_amount = 600) (h3 : tax_rate = 0.08) : 
  ((total_value - non_taxable_amount) * tax_rate = 89.60) :=
by 
  sorry

end NUMINAMATH_GPT_tourist_tax_l964_96492


namespace NUMINAMATH_GPT_right_triangle_altitude_l964_96471

theorem right_triangle_altitude {DE DF EF altitude : ℝ} (h_right_triangle : DE^2 = DF^2 + EF^2)
  (h_DE : DE = 15) (h_DF : DF = 9) (h_EF : EF = 12) (h_area : (DF * EF) / 2 = 54) :
  altitude = 7.2 := 
  sorry

end NUMINAMATH_GPT_right_triangle_altitude_l964_96471


namespace NUMINAMATH_GPT_range_of_a_l964_96433

noncomputable def f (a x : ℝ) : ℝ :=
  if x > 1 then x + a / x + 1 else -x^2 + 2 * x

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, x ≤ y → f a x ≤ f a y) : -1 ≤ a ∧ a ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l964_96433


namespace NUMINAMATH_GPT_middle_elementary_students_l964_96435

theorem middle_elementary_students (S S_PS S_MS S_MR : ℕ) 
  (h1 : S = 12000)
  (h2 : S_PS = (15 * S) / 16)
  (h3 : S_MS = S - S_PS)
  (h4 : S_MR + S_MS = (S_PS) / 2) : 
  S_MR = 4875 :=
by
  sorry

end NUMINAMATH_GPT_middle_elementary_students_l964_96435


namespace NUMINAMATH_GPT_negation_of_universal_is_existential_l964_96403

theorem negation_of_universal_is_existential :
  ¬ (∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2 * x + 4 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_is_existential_l964_96403


namespace NUMINAMATH_GPT_fractional_inequality_solution_l964_96446

theorem fractional_inequality_solution :
  {x : ℝ | (2 * x - 1) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 1 / 2} := 
by
  sorry

end NUMINAMATH_GPT_fractional_inequality_solution_l964_96446


namespace NUMINAMATH_GPT_figure_area_l964_96450

-- Given conditions
def right_angles (α β γ δ: ℕ): Prop :=
  α = 90 ∧ β = 90 ∧ γ = 90 ∧ δ = 90

def segment_lengths (a b c d e f g: ℕ): Prop :=
  a = 15 ∧ b = 8 ∧ c = 7 ∧ d = 3 ∧ e = 4 ∧ f = 2 ∧ g = 5

-- Define the problem
theorem figure_area :
  ∀ (α β γ δ a b c d e f g: ℕ),
    right_angles α β γ δ →
    segment_lengths a b c d e f g →
    a * b - (g * 1 + (d * f)) = 109 :=
by
  sorry

end NUMINAMATH_GPT_figure_area_l964_96450


namespace NUMINAMATH_GPT_solve_for_k_l964_96485

theorem solve_for_k : 
  ∃ k : ℤ, (k + 2) / 4 - (2 * k - 1) / 6 = 1 ∧ k = -4 := 
by
  use -4
  sorry

end NUMINAMATH_GPT_solve_for_k_l964_96485


namespace NUMINAMATH_GPT_problem1_l964_96464

theorem problem1 (α : Real) (h : Real.tan (Real.pi / 4 + α) = 2) : 
  (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 := 
sorry

end NUMINAMATH_GPT_problem1_l964_96464


namespace NUMINAMATH_GPT_total_ticket_sales_is_48_l964_96498

noncomputable def ticket_sales (total_revenue : ℕ) (price_per_ticket : ℕ) (discount_1 : ℕ) (discount_2 : ℕ) : ℕ :=
  let number_first_batch := 10
  let number_second_batch := 20
  let revenue_first_batch := number_first_batch * (price_per_ticket - (price_per_ticket * discount_1 / 100))
  let revenue_second_batch := number_second_batch * (price_per_ticket - (price_per_ticket * discount_2 / 100))
  let revenue_full_price := total_revenue - (revenue_first_batch + revenue_second_batch)
  let number_full_price_tickets := revenue_full_price / price_per_ticket
  number_first_batch + number_second_batch + number_full_price_tickets

theorem total_ticket_sales_is_48 : ticket_sales 820 20 40 15 = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_ticket_sales_is_48_l964_96498


namespace NUMINAMATH_GPT_find_a4_l964_96400

noncomputable def geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

theorem find_a4 (a_n : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a_n q →
  a_n 1 + a_n 2 = -1 →
  a_n 1 - a_n 3 = -3 →
  a_n 4 = -8 :=
by 
  sorry

end NUMINAMATH_GPT_find_a4_l964_96400


namespace NUMINAMATH_GPT_find_f_log_log_3_value_l964_96432

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x - b * Real.logb 3 (Real.sqrt (x*x + 1) - x) + 1

theorem find_f_log_log_3_value
  (a b : ℝ)
  (h1 : f a b (Real.log 10 / Real.log 3) = 5) :
  f a b (-Real.log 10 / Real.log 3) = -3 :=
  sorry

end NUMINAMATH_GPT_find_f_log_log_3_value_l964_96432


namespace NUMINAMATH_GPT_parabola_standard_equation_l964_96424

theorem parabola_standard_equation (directrix : ℝ) (h_directrix : directrix = 1) : 
  ∃ (a : ℝ), y^2 = a * x ∧ a = -4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_standard_equation_l964_96424


namespace NUMINAMATH_GPT_derivative_of_f_l964_96454

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 4 * x - 2 :=
by
  intro x
  -- proof skipped
  sorry

end NUMINAMATH_GPT_derivative_of_f_l964_96454


namespace NUMINAMATH_GPT_tens_digit_of_seven_times_cubed_is_one_l964_96477

-- Variables and definitions
variables (p : ℕ) (h1 : p < 10)

-- Main theorem statement
theorem tens_digit_of_seven_times_cubed_is_one (hp : p < 10) :
  let N := 11 * p
  let m := 7
  let result := m * N^3
  (result / 10) % 10 = 1 := 
sorry

end NUMINAMATH_GPT_tens_digit_of_seven_times_cubed_is_one_l964_96477


namespace NUMINAMATH_GPT_find_value_of_expression_l964_96422

theorem find_value_of_expression (a b c d : ℤ) (h₁ : a = -1) (h₂ : b + c = 0) (h₃ : abs d = 2) :
  4 * a + (b + c) - abs (3 * d) = -10 := by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l964_96422


namespace NUMINAMATH_GPT_certain_number_l964_96472

theorem certain_number (n w : ℕ) (h1 : w = 132)
  (h2 : ∃ m1 m2 m3, 32 = 2^5 * 3^3 * 11^2 * m1 * m2 * m3)
  (h3 : n * w = 132 * 2^3 * 3^2 * 11)
  (h4 : m1 = 1) (h5 : m2 = 1) (h6 : m3 = 1): 
  n = 792 :=
by sorry

end NUMINAMATH_GPT_certain_number_l964_96472


namespace NUMINAMATH_GPT_sum_of_digits_square_1111111_l964_96425

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_square_1111111 :
  sum_of_digits (1111111 * 1111111) = 49 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_square_1111111_l964_96425


namespace NUMINAMATH_GPT_retirement_hire_year_l964_96419

theorem retirement_hire_year (A : ℕ) (R : ℕ) (Y : ℕ) (W : ℕ) 
  (h1 : A + W = 70) 
  (h2 : A = 32) 
  (h3 : R = 2008) 
  (h4 : W = R - Y) : Y = 1970 :=
by
  sorry

end NUMINAMATH_GPT_retirement_hire_year_l964_96419


namespace NUMINAMATH_GPT_proof_problem_l964_96449

-- Given conditions for propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

-- Combined proposition p and q
def p_and_q (a : ℝ) := p a ∧ q a

-- Statement of the proof problem: Prove that p_and_q a → a ≤ -1
theorem proof_problem (a : ℝ) : p_and_q a → (a ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l964_96449


namespace NUMINAMATH_GPT_new_determinant_l964_96445

-- Given the condition that the determinant of the original matrix is 12
def original_determinant (x y z w : ℝ) : Prop :=
  x * w - y * z = 12

-- Proof that the determinant of the new matrix equals the expected result
theorem new_determinant (x y z w : ℝ) (h : original_determinant x y z w) :
  (2 * x + z) * w - (2 * y - w) * z = 24 + z * w + w * z := by
  sorry

end NUMINAMATH_GPT_new_determinant_l964_96445


namespace NUMINAMATH_GPT_vote_percentage_for_candidate_A_l964_96461

noncomputable def percent_democrats : ℝ := 0.60
noncomputable def percent_republicans : ℝ := 0.40
noncomputable def percent_voting_a_democrats : ℝ := 0.70
noncomputable def percent_voting_a_republicans : ℝ := 0.20

theorem vote_percentage_for_candidate_A :
    (percent_democrats * percent_voting_a_democrats + percent_republicans * percent_voting_a_republicans) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_vote_percentage_for_candidate_A_l964_96461


namespace NUMINAMATH_GPT_functional_eq_one_l964_96469

theorem functional_eq_one (f : ℝ → ℝ) (h1 : ∀ x, 0 < x → 0 < f x) 
    (h2 : ∀ x > 0, ∀ y > 0, f x * f (y * f x) = f (x + y)) :
    ∀ x, 0 < x → f x = 1 := 
by
  sorry

end NUMINAMATH_GPT_functional_eq_one_l964_96469


namespace NUMINAMATH_GPT_find_other_number_l964_96465

theorem find_other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 2310) (h_hcf : Nat.gcd A B = 61) (h_a : A = 210) : B = 671 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l964_96465


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l964_96430

theorem quadratic_inequality_solution (x m : ℝ) :
  (x^2 + (2*m + 1)*x + m^2 + m > 0) ↔ (x > -m ∨ x < -m - 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l964_96430


namespace NUMINAMATH_GPT_determine_k_linear_l964_96405

theorem determine_k_linear (k : ℝ) : |k| = 1 ∧ k + 1 ≠ 0 ↔ k = 1 := by
  sorry

end NUMINAMATH_GPT_determine_k_linear_l964_96405


namespace NUMINAMATH_GPT_second_number_is_34_l964_96426

theorem second_number_is_34 (x y z : ℝ) (h1 : x + y + z = 120) 
  (h2 : x / y = 3 / 4) (h3 : y / z = 4 / 7) : y = 34 :=
by 
  sorry

end NUMINAMATH_GPT_second_number_is_34_l964_96426


namespace NUMINAMATH_GPT_xyz_sum_eq_48_l964_96421

theorem xyz_sum_eq_48 (x y z : ℕ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : x * z + y = 47) : 
  x + y + z = 48 := by
  sorry

end NUMINAMATH_GPT_xyz_sum_eq_48_l964_96421


namespace NUMINAMATH_GPT_library_book_configurations_l964_96482

def number_of_valid_configurations (total_books : ℕ) (min_in_library : ℕ) (min_checked_out : ℕ) : ℕ :=
  (total_books - (min_in_library + min_checked_out + 1)) + 1

theorem library_book_configurations : number_of_valid_configurations 8 2 2 = 5 :=
by
  -- Here we would write the Lean proof, but since we are only interested in the statement:
  sorry

end NUMINAMATH_GPT_library_book_configurations_l964_96482


namespace NUMINAMATH_GPT_roots_difference_squared_l964_96416

-- Defining the solutions to the quadratic equation
def quadratic_equation_roots (a b : ℚ) : Prop :=
  (2 * a^2 - 7 * a + 6 = 0) ∧ (2 * b^2 - 7 * b + 6 = 0)

-- The main theorem we aim to prove
theorem roots_difference_squared (a b : ℚ) (h : quadratic_equation_roots a b) :
    (a - b)^2 = 1 / 4 := 
  sorry

end NUMINAMATH_GPT_roots_difference_squared_l964_96416


namespace NUMINAMATH_GPT_find_ABC_sum_l964_96427

-- Conditions
def poly (A B C : ℤ) (x : ℤ) := x^3 + A * x^2 + B * x + C
def roots_condition (A B C : ℤ) := poly A B C (-1) = 0 ∧ poly A B C 3 = 0 ∧ poly A B C 4 = 0

-- Proof goal
theorem find_ABC_sum (A B C : ℤ) (h : roots_condition A B C) : A + B + C = 11 :=
sorry

end NUMINAMATH_GPT_find_ABC_sum_l964_96427


namespace NUMINAMATH_GPT_smallest_positive_integer_solution_l964_96488

theorem smallest_positive_integer_solution (x : ℤ) 
  (hx : |5 * x - 8| = 47) : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_solution_l964_96488


namespace NUMINAMATH_GPT_find_prime_and_int_solutions_l964_96452

-- Define the conditions
def is_solution (p x : ℕ) : Prop :=
  x^(p-1) ∣ (p-1)^x + 1

-- Define the statement to be proven
theorem find_prime_and_int_solutions :
  ∀ p x : ℕ, Prime p → (1 ≤ x ∧ x ≤ 2 * p) →
  (is_solution p x ↔ 
    (p = 2 ∧ (x = 1 ∨ x = 2)) ∨ 
    (p = 3 ∧ (x = 1 ∨ x = 3)) ∨
    (x = 1))
:=
by sorry

end NUMINAMATH_GPT_find_prime_and_int_solutions_l964_96452


namespace NUMINAMATH_GPT_keanu_total_spending_l964_96406

-- Definitions based on conditions
def dog_fish : Nat := 40
def cat_fish : Nat := dog_fish / 2
def total_fish : Nat := dog_fish + cat_fish
def cost_per_fish : Nat := 4
def total_cost : Nat := total_fish * cost_per_fish

-- Theorem statement
theorem keanu_total_spending : total_cost = 240 :=
by 
    sorry

end NUMINAMATH_GPT_keanu_total_spending_l964_96406


namespace NUMINAMATH_GPT_percent_decaffeinated_second_batch_l964_96479

theorem percent_decaffeinated_second_batch :
  ∀ (initial_stock : ℝ) (initial_percent : ℝ) (additional_stock : ℝ) (total_percent : ℝ) (second_batch_percent : ℝ),
  initial_stock = 400 →
  initial_percent = 0.20 →
  additional_stock = 100 →
  total_percent = 0.26 →
  (initial_percent * initial_stock + second_batch_percent * additional_stock = total_percent * (initial_stock + additional_stock)) →
  second_batch_percent = 0.50 :=
by
  intros initial_stock initial_percent additional_stock total_percent second_batch_percent
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_percent_decaffeinated_second_batch_l964_96479


namespace NUMINAMATH_GPT_andrew_vacation_days_l964_96459

-- Andrew's working days and vacation accrual rate
def days_worked : ℕ := 300
def vacation_rate : Nat := 10
def vacation_days_earned : ℕ := days_worked / vacation_rate

-- Days off in March and September
def days_off_march : ℕ := 5
def days_off_september : ℕ := 2 * days_off_march
def total_days_off : ℕ := days_off_march + days_off_september

-- Remaining vacation days calculation
def remaining_vacation_days : ℕ := vacation_days_earned - total_days_off

-- Problem statement to prove
theorem andrew_vacation_days : remaining_vacation_days = 15 :=
by
  -- Substitute the known values and perform the calculation
  unfold remaining_vacation_days vacation_days_earned total_days_off vacation_rate days_off_march days_off_september days_worked
  norm_num
  sorry

end NUMINAMATH_GPT_andrew_vacation_days_l964_96459


namespace NUMINAMATH_GPT_find_a_l964_96431

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

theorem find_a (a : ℝ) (h : {x | x^2 - 3 * x + 2 = 0} ∩ {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0} = {2}) :
  a = -3 ∨ a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l964_96431


namespace NUMINAMATH_GPT_expected_value_of_die_l964_96420

noncomputable def expected_value : ℚ :=
  (1/14) * 1 + (1/14) * 2 + (1/14) * 3 + (1/14) * 4 + (1/14) * 5 + (1/14) * 6 + (1/14) * 7 + (3/8) * 8

theorem expected_value_of_die : expected_value = 5 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_die_l964_96420


namespace NUMINAMATH_GPT_identical_digit_square_l964_96428

theorem identical_digit_square {b x y : ℕ} (hb : b ≥ 2) (hx : x < b) (hy : y < b) (hx_pos : x ≠ 0) (hy_pos : y ≠ 0) :
  (x * b + x)^2 = y * b^3 + y * b^2 + y * b + y ↔ b = 7 :=
by
  sorry

end NUMINAMATH_GPT_identical_digit_square_l964_96428


namespace NUMINAMATH_GPT_green_papayas_left_l964_96418

/-- Define the initial number of green papayas on the tree -/
def initial_green_papayas : ℕ := 14

/-- Define the number of papayas that turned yellow on Friday -/
def friday_yellow_papayas : ℕ := 2

/-- Define the number of papayas that turned yellow on Sunday -/
def sunday_yellow_papayas : ℕ := 2 * friday_yellow_papayas

/-- The remaining number of green papayas after Friday and Sunday -/
def remaining_green_papayas : ℕ := initial_green_papayas - friday_yellow_papayas - sunday_yellow_papayas

theorem green_papayas_left : remaining_green_papayas = 8 := by
  sorry

end NUMINAMATH_GPT_green_papayas_left_l964_96418


namespace NUMINAMATH_GPT_min_value_of_S_l964_96455

variable (x : ℝ)
def S (x : ℝ) : ℝ := (x - 10)^2 + (x + 5)^2

theorem min_value_of_S : ∀ x : ℝ, S x ≥ 112.5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_S_l964_96455


namespace NUMINAMATH_GPT_price_per_litre_of_second_oil_l964_96481

-- Define the conditions given in the problem
def oil1_volume : ℝ := 10 -- 10 litres of first oil
def oil1_rate : ℝ := 50 -- Rs. 50 per litre

def oil2_volume : ℝ := 5 -- 5 litres of the second oil
def total_mixed_volume : ℝ := oil1_volume + oil2_volume -- Total volume of mixed oil

def mixed_rate : ℝ := 55.33 -- Rs. 55.33 per litre for the mixed oil

-- Define the target value to prove: price per litre of the second oil
def price_of_second_oil : ℝ := 65.99

-- Prove the statement
theorem price_per_litre_of_second_oil : 
  (oil1_volume * oil1_rate + oil2_volume * price_of_second_oil) = total_mixed_volume * mixed_rate :=
by 
  sorry -- actual proof to be provided

end NUMINAMATH_GPT_price_per_litre_of_second_oil_l964_96481


namespace NUMINAMATH_GPT_father_ate_oranges_l964_96401

theorem father_ate_oranges (initial_oranges : ℝ) (remaining_oranges : ℝ) (eaten_oranges : ℝ) : 
  initial_oranges = 77.0 → remaining_oranges = 75 → eaten_oranges = initial_oranges - remaining_oranges → eaten_oranges = 2.0 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_father_ate_oranges_l964_96401


namespace NUMINAMATH_GPT_highest_possible_average_l964_96441

theorem highest_possible_average (average_score : ℕ) (total_tests : ℕ) (lowest_score : ℕ) 
  (total_marks : ℕ := total_tests * average_score)
  (new_total_tests : ℕ := total_tests - 1)
  (resulting_average : ℚ := (total_marks - lowest_score) / new_total_tests) :
  average_score = 68 ∧ total_tests = 9 ∧ lowest_score = 0 → resulting_average = 76.5 := sorry

end NUMINAMATH_GPT_highest_possible_average_l964_96441


namespace NUMINAMATH_GPT_initial_men_garrison_l964_96460

-- Conditions:
-- A garrison has provisions for 31 days.
-- At the end of 16 days, a reinforcement of 300 men arrives.
-- The provisions last only for 5 days more after the reinforcement arrives.

theorem initial_men_garrison (M : ℕ) (P : ℕ) (d1 d2 : ℕ) (r : ℕ) (remaining1 remaining2 : ℕ) :
  P = M * d1 →
  remaining1 = P - M * d2 →
  remaining2 = r * (d1 - d2) →
  remaining1 = remaining2 →
  r = M + 300 →
  d1 = 31 →
  d2 = 16 →
  M = 150 :=
by 
  sorry

end NUMINAMATH_GPT_initial_men_garrison_l964_96460


namespace NUMINAMATH_GPT_problem_l964_96467

theorem problem (x : ℝ) (h : 3 * x^2 - 2 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2 / 3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_l964_96467


namespace NUMINAMATH_GPT_harry_worked_34_hours_l964_96496

noncomputable def Harry_hours_worked (x : ℝ) : ℝ := 34

theorem harry_worked_34_hours (x : ℝ)
  (H : ℝ) (James_hours : ℝ) (Harry_pay James_pay: ℝ) 
  (h1 : Harry_pay = 18 * x + 1.5 * x * (H - 18)) 
  (h2 : James_pay = 40 * x + 2 * x * (James_hours - 40)) 
  (h3 : James_hours = 41) 
  (h4 : Harry_pay = James_pay) : 
  H = Harry_hours_worked x :=
by
  sorry

end NUMINAMATH_GPT_harry_worked_34_hours_l964_96496


namespace NUMINAMATH_GPT_amount_of_flour_per_large_tart_l964_96434

-- Statement without proof
theorem amount_of_flour_per_large_tart 
  (num_small_tarts : ℕ) (flour_per_small_tart : ℚ) 
  (num_large_tarts : ℕ) (total_flour : ℚ) 
  (h1 : num_small_tarts = 50) 
  (h2 : flour_per_small_tart = 1/8) 
  (h3 : num_large_tarts = 25) 
  (h4 : total_flour = num_small_tarts * flour_per_small_tart) : 
  total_flour = num_large_tarts * (1/4) := 
sorry

end NUMINAMATH_GPT_amount_of_flour_per_large_tart_l964_96434


namespace NUMINAMATH_GPT_sum_of_variables_is_38_l964_96448

theorem sum_of_variables_is_38
  (x y z w : ℤ)
  (h₁ : x - y + z = 10)
  (h₂ : y - z + w = 15)
  (h₃ : z - w + x = 9)
  (h₄ : w - x + y = 4) :
  x + y + z + w = 38 := by
  sorry

end NUMINAMATH_GPT_sum_of_variables_is_38_l964_96448


namespace NUMINAMATH_GPT_min_workers_to_profit_l964_96462

/-- Definitions of constants used in the problem. --/
def daily_maintenance_cost : ℕ := 500
def wage_per_hour : ℕ := 20
def widgets_per_hour_per_worker : ℕ := 5
def sell_price_per_widget : ℕ := 350 / 100 -- since the input is 3.50
def workday_hours : ℕ := 8

/-- Profit condition: the revenue should be greater than the cost. 
    The problem specifies that the number of workers must be at least 26 to make a profit. --/

theorem min_workers_to_profit (n : ℕ) :
  (widgets_per_hour_per_worker * workday_hours * sell_price_per_widget * n > daily_maintenance_cost + (workday_hours * wage_per_hour * n)) → n ≥ 26 :=
sorry


end NUMINAMATH_GPT_min_workers_to_profit_l964_96462


namespace NUMINAMATH_GPT_can_cut_rectangle_l964_96490

def original_rectangle_width := 100
def original_rectangle_height := 70
def total_area := original_rectangle_width * original_rectangle_height

def area1 := 1000
def area2 := 2000
def area3 := 4000

theorem can_cut_rectangle : 
  (area1 + area2 + area3 = total_area) ∧ 
  (area1 * 2 = area2) ∧ 
  (area1 * 4 = area3) ∧ 
  (area1 > 0) ∧ (area2 > 0) ∧ (area3 > 0) ∧
  (∃ (w1 h1 w2 h2 w3 h3 : ℕ), 
    w1 * h1 = area1 ∧ w2 * h2 = area2 ∧ w3 * h3 = area3 ∧
    ((w1 + w2 ≤ original_rectangle_width ∧ max h1 h2 + h3 ≤ original_rectangle_height) ∨
     (h1 + h2 ≤ original_rectangle_height ∧ max w1 w2 + w3 ≤ original_rectangle_width)))
:=
  sorry

end NUMINAMATH_GPT_can_cut_rectangle_l964_96490


namespace NUMINAMATH_GPT_student_allowance_l964_96439

theorem student_allowance (A : ℝ) (h1 : A * (2/5) = A - (A * (3/5)))
  (h2 : (A - (A * (2/5))) * (1/3) = ((A - (A * (2/5))) - ((A - (A * (2/5))) * (1/3))) * (1/3))
  (h3 : ((A - (A * (2/5))) - ((A - (A * (2/5))) * (1/3))) = 1.20) :
  A = 3.00 :=
by
  sorry

end NUMINAMATH_GPT_student_allowance_l964_96439


namespace NUMINAMATH_GPT_triangular_weight_is_60_l964_96429

/-- Suppose there are weights: 5 identical round, 2 identical triangular, and 1 rectangular weight of 90 grams.
    The conditions are: 
    1. One round weight and one triangular weight balance three round weights.
    2. Four round weights and one triangular weight balance one triangular weight, one round weight, and one rectangular weight.
    Prove that the weight of the triangular weight is 60 grams. -/
theorem triangular_weight_is_60 
  (R T : ℕ)  -- We declare weights of round and triangular weights as natural numbers
  (h1 : R + T = 3 * R)  -- The first balance condition
  (h2 : 4 * R + T = T + R + 90)  -- The second balance condition
  : T = 60 := 
by
  sorry  -- Proof omitted

end NUMINAMATH_GPT_triangular_weight_is_60_l964_96429


namespace NUMINAMATH_GPT_standard_heat_of_formation_Fe2O3_l964_96491

def Q_form_Al2O3 := 1675.5 -- kJ/mol

def Q1 := 854.2 -- kJ

-- Definition of the standard heat of formation of Fe2O3
def Q_form_Fe2O3 := Q_form_Al2O3 - Q1

-- The proof goal
theorem standard_heat_of_formation_Fe2O3 : Q_form_Fe2O3 = 821.3 := by
  sorry

end NUMINAMATH_GPT_standard_heat_of_formation_Fe2O3_l964_96491


namespace NUMINAMATH_GPT_zoe_has_47_nickels_l964_96415

theorem zoe_has_47_nickels (x : ℕ) 
  (h1 : 5 * x + 10 * x + 50 * x = 3050) : 
  x = 47 := 
sorry

end NUMINAMATH_GPT_zoe_has_47_nickels_l964_96415


namespace NUMINAMATH_GPT_imaginary_part_of_complex_l964_96438

open Complex

theorem imaginary_part_of_complex (i : ℂ) (z : ℂ) (h1 : i^2 = -1) (h2 : z = (3 - 2 * i^3) / (1 + i)) : z.im = -1 / 2 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_imaginary_part_of_complex_l964_96438


namespace NUMINAMATH_GPT_choose_with_at_least_one_girl_l964_96417

theorem choose_with_at_least_one_girl :
  let boys := 4
  let girls := 2
  let total_students := boys + girls
  let ways_choose_4 := Nat.choose total_students 4
  let ways_all_boys := Nat.choose boys 4
  ways_choose_4 - ways_all_boys = 14 := by
  sorry

end NUMINAMATH_GPT_choose_with_at_least_one_girl_l964_96417


namespace NUMINAMATH_GPT_group_selection_l964_96480

theorem group_selection (m f : ℕ) (h1 : m + f = 8) (h2 : (m * (m - 1) / 2) * f = 30) : f = 3 :=
sorry

end NUMINAMATH_GPT_group_selection_l964_96480


namespace NUMINAMATH_GPT_expression_1_expression_2_expression_3_expression_4_l964_96410

section problem1

variable {x : ℝ}

theorem expression_1:
  (x^2 - 1 + x)*(x^2 - 1 + 3*x) + x^2  = x^4 + 4*x^3 + 4*x^2 - 4*x - 1 :=
sorry

end problem1

section problem2

variable {x a : ℝ}

theorem expression_2:
  (x - a)^4 + 4*a^4 = (x^2 + a^2)*(x^2 - 4*a*x + 5*a^2) :=
sorry

end problem2

section problem3

variable {a : ℝ}

theorem expression_3:
  (a + 1)^4 + 2*(a + 1)^3 + a*(a + 2) = (a + 1)^4 + 2*(a + 1)^3 + 1 :=
sorry

end problem3

section problem4

variable {p : ℝ}

theorem expression_4:
  (p + 2)^4 + 2*(p^2 - 4)^2 + (p - 2)^4 = 4*p^4 :=
sorry

end problem4

end NUMINAMATH_GPT_expression_1_expression_2_expression_3_expression_4_l964_96410


namespace NUMINAMATH_GPT_cube_dimension_ratio_l964_96411

theorem cube_dimension_ratio (V1 V2 : ℕ) (h1 : V1 = 27) (h2 : V2 = 216) :
  ∃ r : ℕ, r = 2 ∧ (∃ l1 l2 : ℕ, l1 * l1 * l1 = V1 ∧ l2 * l2 * l2 = V2 ∧ l2 = r * l1) :=
by
  sorry

end NUMINAMATH_GPT_cube_dimension_ratio_l964_96411


namespace NUMINAMATH_GPT_find_m_l964_96444

theorem find_m (m : ℕ) : 5 ^ m = 5 * 25 ^ 2 * 125 ^ 3 ↔ m = 14 := by
  sorry

end NUMINAMATH_GPT_find_m_l964_96444


namespace NUMINAMATH_GPT_area_ratio_of_triangles_l964_96489

theorem area_ratio_of_triangles (AC AD : ℝ) (h : ℝ) (hAC : AC = 1) (hAD : AD = 4) :
  (AC * h / 2) / ((AD - AC) * h / 2) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_of_triangles_l964_96489


namespace NUMINAMATH_GPT_find_a1_l964_96443

variable {a_n : ℕ → ℤ}
variable (common_difference : ℤ) (a1 : ℤ)

-- Define that a_n is an arithmetic sequence with common difference of 2
def is_arithmetic_seq (a_n : ℕ → ℤ) (common_difference : ℤ) : Prop :=
  ∀ n, a_n (n + 1) - a_n n = common_difference

-- State the condition that a1, a2, a4 form a geometric sequence
def forms_geometric_seq (a_n : ℕ → ℤ) : Prop :=
  ∃ a1 a2 a4, a2 * a2 = a1 * a4 ∧ a_n 1 = a1 ∧ a_n 2 = a2 ∧ a_n 4 = a4

-- Define the problem statement
theorem find_a1 (h_arith : is_arithmetic_seq a_n 2) (h_geom : forms_geometric_seq a_n) :
  a_n 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_l964_96443


namespace NUMINAMATH_GPT_rectangle_area_y_value_l964_96487

theorem rectangle_area_y_value :
  ∀ (y : ℝ), 
  let P := (1, 1)
  let Q := (1, 4)
  let R := (y, 4)
  let S := (y, 1)
  (y > 1) → 
  (abs (R.1 - P.1) * abs (Q.2 - P.2) = 36) → 
  y = 13 :=
by
  intros y P Q R S hy harea
  let P := (1, 1)
  let Q := (1, 4)
  let R := (y, 4)
  let S := (y, 1)
  sorry

end NUMINAMATH_GPT_rectangle_area_y_value_l964_96487


namespace NUMINAMATH_GPT_two_pow_n_minus_one_divisible_by_seven_l964_96412

theorem two_pow_n_minus_one_divisible_by_seven (n : ℕ) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, n = 3 * k := 
sorry

end NUMINAMATH_GPT_two_pow_n_minus_one_divisible_by_seven_l964_96412


namespace NUMINAMATH_GPT_sampling_method_D_is_the_correct_answer_l964_96484

def sampling_method_A_is_simple_random_sampling : Prop :=
  false

def sampling_method_B_is_simple_random_sampling : Prop :=
  false

def sampling_method_C_is_simple_random_sampling : Prop :=
  false

def sampling_method_D_is_simple_random_sampling : Prop :=
  true

theorem sampling_method_D_is_the_correct_answer :
  sampling_method_A_is_simple_random_sampling = false ∧
  sampling_method_B_is_simple_random_sampling = false ∧
  sampling_method_C_is_simple_random_sampling = false ∧
  sampling_method_D_is_simple_random_sampling = true :=
by
  sorry

end NUMINAMATH_GPT_sampling_method_D_is_the_correct_answer_l964_96484


namespace NUMINAMATH_GPT_train_length_l964_96494

-- Define the conditions
def equal_length_trains (L : ℝ) : Prop :=
  ∃ (length : ℝ), length = L

def train_speeds : Prop :=
  ∃ v_fast v_slow : ℝ, v_fast = 46 ∧ v_slow = 36

def pass_time (t : ℝ) : Prop :=
  t = 36

-- The proof problem
theorem train_length (L : ℝ) 
  (h_equal_length : equal_length_trains L) 
  (h_speeds : train_speeds)
  (h_time : pass_time 36) : 
  L = 50 :=
sorry

end NUMINAMATH_GPT_train_length_l964_96494


namespace NUMINAMATH_GPT_third_consecutive_even_l964_96474

theorem third_consecutive_even {a b c d : ℕ} (h1 : b = a + 2) (h2 : c = a + 4) (h3 : d = a + 6) (h_sum : a + b + c + d = 52) : c = 14 :=
by
  sorry

end NUMINAMATH_GPT_third_consecutive_even_l964_96474


namespace NUMINAMATH_GPT_weight_of_sugar_is_16_l964_96468

def weight_of_sugar_bag (weight_of_sugar weight_of_salt remaining_weight weight_removed : ℕ) : Prop :=
  weight_of_sugar + weight_of_salt - weight_removed = remaining_weight

theorem weight_of_sugar_is_16 :
  ∃ (S : ℕ), weight_of_sugar_bag S 30 42 4 ∧ S = 16 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_sugar_is_16_l964_96468


namespace NUMINAMATH_GPT_maximize_profit_l964_96451

noncomputable def annual_profit : ℝ → ℝ
| x => if x < 80 then - (1/3) * x^2 + 40 * x - 250 
       else 1200 - (x + 10000 / x)

theorem maximize_profit : ∃ x : ℝ, x = 100 ∧ annual_profit x = 1000 :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l964_96451


namespace NUMINAMATH_GPT_ratio_josh_to_selena_l964_96423

def total_distance : ℕ := 36
def selena_distance : ℕ := 24

def josh_distance (td sd : ℕ) : ℕ := td - sd

theorem ratio_josh_to_selena : (josh_distance total_distance selena_distance) / selena_distance = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_josh_to_selena_l964_96423


namespace NUMINAMATH_GPT_polynomial_coeff_divisible_by_5_l964_96408

theorem polynomial_coeff_divisible_by_5 (a b c d : ℤ) 
  (h : ∀ (x : ℤ), (a * x^3 + b * x^2 + c * x + d) % 5 = 0) : 
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_coeff_divisible_by_5_l964_96408
