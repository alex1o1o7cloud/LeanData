import Mathlib

namespace NUMINAMATH_GPT_repeating_decimal_ratio_eq_4_l1311_131194

-- Definitions for repeating decimals
def rep_dec_36 := 0.36 -- 0.\overline{36}
def rep_dec_09 := 0.09 -- 0.\overline{09}

-- Lean 4 statement of proof problem
theorem repeating_decimal_ratio_eq_4 :
  (rep_dec_36 / rep_dec_09) = 4 :=
sorry

end NUMINAMATH_GPT_repeating_decimal_ratio_eq_4_l1311_131194


namespace NUMINAMATH_GPT_baron_munchausen_correct_l1311_131158

noncomputable def P (x : ℕ) : ℕ := sorry -- Assume non-constant polynomial with non-negative integer coefficients
noncomputable def Q (x : ℕ) : ℕ := sorry -- Assume non-constant polynomial with non-negative integer coefficients

theorem baron_munchausen_correct (b p0 : ℕ) 
  (hP2 : P 2 = b) 
  (hPp2 : P b = p0) 
  (hQ2 : Q 2 = b) 
  (hQp2 : Q b = p0) : 
  P = Q := sorry

end NUMINAMATH_GPT_baron_munchausen_correct_l1311_131158


namespace NUMINAMATH_GPT_students_exceed_rabbits_l1311_131170

theorem students_exceed_rabbits (students_per_classroom rabbits_per_classroom number_of_classrooms : ℕ) 
  (h_students : students_per_classroom = 18)
  (h_rabbits : rabbits_per_classroom = 2)
  (h_classrooms : number_of_classrooms = 4) : 
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 64 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_exceed_rabbits_l1311_131170


namespace NUMINAMATH_GPT_diameter_of_circular_field_l1311_131190

noncomputable def diameter (C : ℝ) : ℝ := C / Real.pi

theorem diameter_of_circular_field :
  let cost_per_meter := 3
  let total_cost := 376.99
  let circumference := total_cost / cost_per_meter
  diameter circumference = 40 :=
by
  let cost_per_meter : ℝ := 3
  let total_cost : ℝ := 376.99
  let circumference : ℝ := total_cost / cost_per_meter
  have : circumference = 125.66333333333334 := by sorry
  have : diameter circumference = 40 := by sorry
  sorry

end NUMINAMATH_GPT_diameter_of_circular_field_l1311_131190


namespace NUMINAMATH_GPT_rotations_needed_to_reach_goal_l1311_131187

-- Define the given conditions
def rotations_per_block : ℕ := 200
def blocks_goal : ℕ := 8
def current_rotations : ℕ := 600

-- Define total_rotations_needed and more_rotations_needed
def total_rotations_needed : ℕ := blocks_goal * rotations_per_block
def more_rotations_needed : ℕ := total_rotations_needed - current_rotations

-- Theorem stating the solution
theorem rotations_needed_to_reach_goal : more_rotations_needed = 1000 := by
  -- proof steps are omitted
  sorry

end NUMINAMATH_GPT_rotations_needed_to_reach_goal_l1311_131187


namespace NUMINAMATH_GPT_roots_of_equation_l1311_131193

theorem roots_of_equation:
  ∀ x : ℝ, (x - 2) * (x - 3) = x - 2 → x = 2 ∨ x = 4 := by
  sorry

end NUMINAMATH_GPT_roots_of_equation_l1311_131193


namespace NUMINAMATH_GPT_initial_fee_correct_l1311_131186

-- Define the relevant values
def initialFee := 2.25
def chargePerSegment := 0.4
def totalDistance := 3.6
def totalCharge := 5.85
noncomputable def segments := (totalDistance * (5 / 2))
noncomputable def costForDistance := segments * chargePerSegment

-- Define the theorem
theorem initial_fee_correct :
  totalCharge = initialFee + costForDistance :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_initial_fee_correct_l1311_131186


namespace NUMINAMATH_GPT_two_b_is_16667_percent_of_a_l1311_131136

theorem two_b_is_16667_percent_of_a {a b : ℝ} (h : a = 1.2 * b) : (2 * b / a) = 5 / 3 := by
  sorry

end NUMINAMATH_GPT_two_b_is_16667_percent_of_a_l1311_131136


namespace NUMINAMATH_GPT_solve_for_x_l1311_131128

theorem solve_for_x (x : ℝ) 
  (h : (2 / (x + 3)) + (3 * x / (x + 3)) - (5 / (x + 3)) = 2) : 
  x = 9 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1311_131128


namespace NUMINAMATH_GPT_volume_at_target_temperature_l1311_131196

-- Volume expansion relationship
def volume_change_per_degree_rise (ΔT V_real : ℝ) : Prop :=
  ΔT = 2 ∧ V_real = 3

-- Initial conditions
def initial_conditions (V_initial T_initial : ℝ) : Prop :=
  V_initial = 36 ∧ T_initial = 30

-- Target temperature
def target_temperature (T_target : ℝ) : Prop :=
  T_target = 20

-- Theorem stating the volume at the target temperature
theorem volume_at_target_temperature (ΔT V_real T_initial V_initial T_target V_target : ℝ) 
  (h_rel : volume_change_per_degree_rise ΔT V_real)
  (h_init : initial_conditions V_initial T_initial)
  (h_target : target_temperature T_target) :
  V_target = V_initial + V_real * ((T_target - T_initial) / ΔT) :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_volume_at_target_temperature_l1311_131196


namespace NUMINAMATH_GPT_overall_percentage_decrease_l1311_131140

theorem overall_percentage_decrease (P x y : ℝ) (hP : P = 100) 
  (h : (P - (x / 100) * P) - (y / 100) * (P - (x / 100) * P) = 55) : 
  ((P - 55) / P) * 100 = 45 := 
by 
  sorry

end NUMINAMATH_GPT_overall_percentage_decrease_l1311_131140


namespace NUMINAMATH_GPT_exists_two_natural_pairs_satisfying_equation_l1311_131173

theorem exists_two_natural_pairs_satisfying_equation :
  ∃ (x1 y1 x2 y2 : ℕ), (2 * x1^3 = y1^4) ∧ (2 * x2^3 = y2^4) ∧ ¬(x1 = x2 ∧ y1 = y2) :=
sorry

end NUMINAMATH_GPT_exists_two_natural_pairs_satisfying_equation_l1311_131173


namespace NUMINAMATH_GPT_smallest_x_l1311_131171

-- Define 450 and provide its factorization.
def n1 := 450
def n1_factors := 2^1 * 3^2 * 5^2

-- Define 675 and provide its factorization.
def n2 := 675
def n2_factors := 3^3 * 5^2

-- State the theorem that proves the smallest x for the condition
theorem smallest_x (x : ℕ) (hx : 450 * x % 675 = 0) : x = 3 := sorry

end NUMINAMATH_GPT_smallest_x_l1311_131171


namespace NUMINAMATH_GPT_initial_apples_l1311_131126

theorem initial_apples (C : ℝ) (h : C + 7.0 = 27) : C = 20.0 := by
  sorry

end NUMINAMATH_GPT_initial_apples_l1311_131126


namespace NUMINAMATH_GPT_Timmy_needs_to_go_faster_l1311_131144

-- Define the trial speeds and the required speed
def s1 : ℕ := 36
def s2 : ℕ := 34
def s3 : ℕ := 38
def s_req : ℕ := 40

-- Statement of the theorem
theorem Timmy_needs_to_go_faster :
  s_req - (s1 + s2 + s3) / 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_Timmy_needs_to_go_faster_l1311_131144


namespace NUMINAMATH_GPT_drops_of_glue_needed_l1311_131183

def number_of_clippings (friend : ℕ) : ℕ :=
  match friend with
  | 1 => 4
  | 2 => 7
  | 3 => 5
  | 4 => 3
  | 5 => 5
  | 6 => 8
  | 7 => 2
  | 8 => 6
  | _ => 0

def total_drops_of_glue : ℕ :=
  (number_of_clippings 1 +
   number_of_clippings 2 +
   number_of_clippings 3 +
   number_of_clippings 4 +
   number_of_clippings 5 +
   number_of_clippings 6 +
   number_of_clippings 7 +
   number_of_clippings 8) * 6

theorem drops_of_glue_needed : total_drops_of_glue = 240 :=
by
  sorry

end NUMINAMATH_GPT_drops_of_glue_needed_l1311_131183


namespace NUMINAMATH_GPT_find_S11_l1311_131188

variable {a : ℕ → ℚ} -- Define the arithmetic sequence as a function

-- Define conditions
def arithmetic_sequence (a : ℕ → ℚ) :=
∀ n m, a (n + m) = a n + a m

def S (n : ℕ) (a : ℕ → ℚ) : ℚ := (n / 2 : ℚ) * (a 1 + a n)

-- Define the problem statement to be proved
theorem find_S11 (h_arith : arithmetic_sequence a) (h_eq : a 3 + a 6 + a 9 = 54) : 
  S 11 a = 198 :=
sorry

end NUMINAMATH_GPT_find_S11_l1311_131188


namespace NUMINAMATH_GPT_product_of_g_of_roots_l1311_131130

noncomputable def f (x : ℝ) : ℝ := x^5 - 2*x^3 + x + 1
noncomputable def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem product_of_g_of_roots (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h₁ : f x₁ = 0) (h₂ : f x₂ = 0) (h₃ : f x₃ = 0)
  (h₄ : f x₄ = 0) (h₅ : f x₅ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = f (-1 + Real.sqrt 2) * f (-1 - Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_product_of_g_of_roots_l1311_131130


namespace NUMINAMATH_GPT_Clara_sells_third_type_boxes_l1311_131159

variable (total_cookies boxes_first boxes_second boxes_third : ℕ)
variable (cookies_per_first cookies_per_second cookies_per_third : ℕ)

theorem Clara_sells_third_type_boxes (h1 : cookies_per_first = 12)
                                    (h2 : boxes_first = 50)
                                    (h3 : cookies_per_second = 20)
                                    (h4 : boxes_second = 80)
                                    (h5 : cookies_per_third = 16)
                                    (h6 : total_cookies = 3320) :
                                    boxes_third = 70 :=
by
  sorry

end NUMINAMATH_GPT_Clara_sells_third_type_boxes_l1311_131159


namespace NUMINAMATH_GPT_max_complexity_51_l1311_131134

-- Define the complexity of a number 
def complexity (x : ℚ) : ℕ := sorry -- Placeholder for the actual complexity function definition

-- Define the sequence for m values
def m_sequence (k : ℕ) : List ℕ :=
  List.range' 1 (2^(k-1)) |>.filter (λ n => n % 2 = 1)

-- Define the candidate number
def candidate_number (k : ℕ) : ℚ :=
  (2^(k + 1) + (-1)^k) / (3 * 2^k)

theorem max_complexity_51 : 
  ∃ m, m ∈ m_sequence 50 ∧ 
  (∀ n, n ∈ m_sequence 50 → complexity (n / 2^50) ≤ complexity (candidate_number 50 / 2^50)) :=
sorry

end NUMINAMATH_GPT_max_complexity_51_l1311_131134


namespace NUMINAMATH_GPT_math_problem_l1311_131129

theorem math_problem
  (x y z : ℕ)
  (h1 : z = 4)
  (h2 : x + y = 7)
  (h3 : x + z = 8) :
  x + y + z = 11 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l1311_131129


namespace NUMINAMATH_GPT_anie_days_to_finish_task_l1311_131133

def extra_hours : ℕ := 5
def normal_work_hours : ℕ := 10
def total_project_hours : ℕ := 1500

theorem anie_days_to_finish_task : (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end NUMINAMATH_GPT_anie_days_to_finish_task_l1311_131133


namespace NUMINAMATH_GPT_text_messages_in_march_l1311_131123

theorem text_messages_in_march
  (nov_texts : ℕ)
  (dec_texts : ℕ)
  (jan_texts : ℕ)
  (feb_texts : ℕ)
  (double_pattern : ∀ n m : ℕ, m = 2 * n)
  (h_nov : nov_texts = 1)
  (h_dec : dec_texts = 2 * nov_texts)
  (h_jan : jan_texts = 2 * dec_texts)
  (h_feb : feb_texts = 2 * jan_texts) : 
  ∃ mar_texts : ℕ, mar_texts = 2 * feb_texts ∧ mar_texts = 16 := 
by
  sorry

end NUMINAMATH_GPT_text_messages_in_march_l1311_131123


namespace NUMINAMATH_GPT_wrapping_paper_solution_l1311_131160

variable (P1 P2 P3 : ℝ)

def wrapping_paper_problem : Prop :=
  P1 = 2 ∧
  P3 = P1 + P2 ∧
  P1 + P2 + P3 = 7 →
  (P2 / P1) = 3 / 4

theorem wrapping_paper_solution : wrapping_paper_problem P1 P2 P3 :=
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_solution_l1311_131160


namespace NUMINAMATH_GPT_john_duck_price_l1311_131179

theorem john_duck_price
  (n_ducks : ℕ)
  (cost_per_duck : ℕ)
  (weight_per_duck : ℕ)
  (total_profit : ℕ)
  (total_cost : ℕ)
  (total_weight : ℕ)
  (total_revenue : ℕ)
  (price_per_pound : ℕ)
  (h1 : n_ducks = 30)
  (h2 : cost_per_duck = 10)
  (h3 : weight_per_duck = 4)
  (h4 : total_profit = 300)
  (h5 : total_cost = n_ducks * cost_per_duck)
  (h6 : total_weight = n_ducks * weight_per_duck)
  (h7 : total_revenue = total_cost + total_profit)
  (h8 : price_per_pound = total_revenue / total_weight) :
  price_per_pound = 5 := 
sorry

end NUMINAMATH_GPT_john_duck_price_l1311_131179


namespace NUMINAMATH_GPT_liza_final_balance_l1311_131157

def initial_balance : ℕ := 800
def rent : ℕ := 450
def paycheck : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

theorem liza_final_balance :
  initial_balance - rent + paycheck - (electricity_bill + internet_bill) - phone_bill = 1563 := by
  sorry

end NUMINAMATH_GPT_liza_final_balance_l1311_131157


namespace NUMINAMATH_GPT_cos_B_equals_3_over_4_l1311_131145

variables {A B C : ℝ} {a b c R : ℝ} (h₁ : b * Real.sin B - a * Real.sin A = (1/2) * a * Real.sin C)
  (h₂ :  2 * R ^ 2 * Real.sin B * (1 - Real.cos (2 * A)) = (1 / 2) * a * b * Real.sin C)

theorem cos_B_equals_3_over_4 : Real.cos B = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_cos_B_equals_3_over_4_l1311_131145


namespace NUMINAMATH_GPT_find_a_l1311_131184

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1311_131184


namespace NUMINAMATH_GPT_inverse_proportionality_example_l1311_131121

theorem inverse_proportionality_example (k : ℝ) (x : ℝ) (y : ℝ) (h1 : 5 * 10 = k) (h2 : x * 40 = k) : x = 5 / 4 :=
by
  -- sorry is used to skip the proof.
  sorry

end NUMINAMATH_GPT_inverse_proportionality_example_l1311_131121


namespace NUMINAMATH_GPT_volume_ratio_l1311_131107

theorem volume_ratio (x : ℝ) (h : x > 0) : 
  let V_Q := x^3
  let V_P := (3 * x)^3
  (V_Q / V_P) = (1 / 27) :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_l1311_131107


namespace NUMINAMATH_GPT_relative_prime_in_consecutive_integers_l1311_131105

theorem relative_prime_in_consecutive_integers (n : ℤ) : 
  ∃ k, n ≤ k ∧ k ≤ n + 5 ∧ ∀ m, n ≤ m ∧ m ≤ n + 5 ∧ m ≠ k → Int.gcd k m = 1 :=
sorry

end NUMINAMATH_GPT_relative_prime_in_consecutive_integers_l1311_131105


namespace NUMINAMATH_GPT_evaluate_expr_l1311_131137

def x := 2
def y := -1
def z := 3
def expr := 2 * x^2 + y^2 - z^2 + 3 * x * y

theorem evaluate_expr : expr = -6 :=
by sorry

end NUMINAMATH_GPT_evaluate_expr_l1311_131137


namespace NUMINAMATH_GPT_greatest_int_less_than_neg_17_div_3_l1311_131192

theorem greatest_int_less_than_neg_17_div_3 : 
  ∀ (x : ℚ), x = -17/3 → ⌊x⌋ = -6 :=
by
  sorry

end NUMINAMATH_GPT_greatest_int_less_than_neg_17_div_3_l1311_131192


namespace NUMINAMATH_GPT_prove_a_range_l1311_131172

-- Defining the propositions p and q
def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1 : ℝ) 1, a^2 * x^2 + a * x - 2 = 0
def q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- The proposition to prove
theorem prove_a_range (a : ℝ) (hpq : ¬(p a ∨ q a)) : a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 :=
by
  sorry

end NUMINAMATH_GPT_prove_a_range_l1311_131172


namespace NUMINAMATH_GPT_percentage_of_alcohol_in_vessel_Q_l1311_131189

theorem percentage_of_alcohol_in_vessel_Q
  (x : ℝ)
  (h_mix : 2.5 + 0.04 * x = 6) :
  x = 87.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_alcohol_in_vessel_Q_l1311_131189


namespace NUMINAMATH_GPT_truncated_cone_contact_radius_l1311_131112

theorem truncated_cone_contact_radius (R r r' ζ : ℝ)
  (h volume_condition : ℝ)
  (R_pos : 0 < R)
  (r_pos : 0 < r)
  (r'_pos : 0 < r')
  (ζ_pos : 0 < ζ)
  (h_eq : h = 2 * R)
  (volume_condition_eq :
    (2 : ℝ) * ((4 / 3) * Real.pi * R^3) = 
    (2 / 3) * Real.pi * h * (r^2 + r * r' + r'^2)) :
  ζ = (2 * R * Real.sqrt 5) / 5 :=
by
  sorry

end NUMINAMATH_GPT_truncated_cone_contact_radius_l1311_131112


namespace NUMINAMATH_GPT_find_quadratic_polynomial_l1311_131153

def quadratic_polynomial (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem find_quadratic_polynomial : 
  ∃ a b c: ℝ, (∀ x : ℂ, quadratic_polynomial a b c x.re = 0 → (x = 3 + 4*I) ∨ (x = 3 - 4*I)) 
  ∧ (b = 8) 
  ∧ (a = -4/3) 
  ∧ (c = -50/3) :=
by
  sorry

end NUMINAMATH_GPT_find_quadratic_polynomial_l1311_131153


namespace NUMINAMATH_GPT_trigonometric_identity_l1311_131152

open Real

theorem trigonometric_identity
  (α β γ φ : ℝ)
  (h1 : sin α + 7 * sin β = 4 * (sin γ + 2 * sin φ))
  (h2 : cos α + 7 * cos β = 4 * (cos γ + 2 * cos φ)) :
  2 * cos (α - φ) = 7 * cos (β - γ) :=
by sorry

end NUMINAMATH_GPT_trigonometric_identity_l1311_131152


namespace NUMINAMATH_GPT_initial_investment_B_l1311_131176

theorem initial_investment_B (A_initial : ℝ) (B : ℝ) (total_profit : ℝ) (A_profit : ℝ) 
(A_withdraw : ℝ) (B_advance : ℝ) : 
  A_initial = 3000 → B_advance = 1000 → A_withdraw = 1000 → total_profit = 756 → A_profit = 288 → 
  (8 * A_initial + 4 * (A_initial - A_withdraw)) / (8 * B + 4 * (B + B_advance)) = A_profit / (total_profit - A_profit) → 
  B = 4000 := 
by 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end NUMINAMATH_GPT_initial_investment_B_l1311_131176


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1311_131122

variable (a b c d1 d2 : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
variable (h4 : 2 * c = (d1 + d2) / 2)
variable (h5 : d1 + d2 = 2 * a)

theorem eccentricity_of_ellipse : (c / a) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1311_131122


namespace NUMINAMATH_GPT_inequality_problem_l1311_131181

theorem inequality_problem (a b c : ℝ) (h : a < b ∧ b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  -- The proof is supposed to be here
  sorry

end NUMINAMATH_GPT_inequality_problem_l1311_131181


namespace NUMINAMATH_GPT_mass_percentage_I_in_CaI2_l1311_131180

theorem mass_percentage_I_in_CaI2 :
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_I : ℝ := 126.90
  let molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I
  let mass_percentage_I : ℝ := (2 * molar_mass_I / molar_mass_CaI2) * 100
  mass_percentage_I = 86.36 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_I_in_CaI2_l1311_131180


namespace NUMINAMATH_GPT_day_of_50th_day_l1311_131149

theorem day_of_50th_day (days_250_N days_150_N1 : ℕ) 
  (h₁ : days_250_N % 7 = 5) (h₂ : days_150_N1 % 7 = 5) : 
  ((50 + 315 - 150 + 365 * 2) % 7) = 4 := 
  sorry

end NUMINAMATH_GPT_day_of_50th_day_l1311_131149


namespace NUMINAMATH_GPT_range_of_slopes_of_line_AB_l1311_131198

variables {x y : ℝ}

/-- (O is the coordinate origin),
    (the parabola y² = 4x),
    (points A and B in the first quadrant),
    (the product of the slopes of lines OA and OB being 1) -/
theorem range_of_slopes_of_line_AB
  (O : ℝ) 
  (A B : ℝ × ℝ)
  (hxA : 0 < A.fst)
  (hyA : 0 < A.snd)
  (hxB : 0 < B.fst)
  (hyB : 0 < B.snd)
  (hA_on_parabola : A.snd^2 = 4 * A.fst)
  (hB_on_parabola : B.snd^2 = 4 * B.fst)
  (h_product_slopes : (A.snd / A.fst) * (B.snd / B.fst) = 1) :
  (0 < (B.snd - A.snd) / (B.fst - A.fst) ∧ (B.snd - A.snd) / (B.fst - A.fst) < 1/2) := 
by
  sorry

end NUMINAMATH_GPT_range_of_slopes_of_line_AB_l1311_131198


namespace NUMINAMATH_GPT_expenditure_ratio_l1311_131125

theorem expenditure_ratio 
  (I1 : ℝ) (I2 : ℝ) (E1 : ℝ) (E2 : ℝ) (S1 : ℝ) (S2 : ℝ)
  (h1 : I1 = 3500)
  (h2 : I2 = (4 / 5) * I1)
  (h3 : S1 = I1 - E1)
  (h4 : S2 = I2 - E2)
  (h5 : S1 = 1400)
  (h6 : S2 = 1400) : 
  E1 / E2 = 3 / 2 :=
by
  -- Steps of the proof will go here
  sorry

end NUMINAMATH_GPT_expenditure_ratio_l1311_131125


namespace NUMINAMATH_GPT_sum_of_fraction_numerator_and_denominator_l1311_131161

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fraction_numerator_and_denominator_l1311_131161


namespace NUMINAMATH_GPT_basketball_scores_l1311_131166

theorem basketball_scores :
  ∃ P: Finset ℕ, (∀ x y: ℕ, (x + y = 7 → P = {p | ∃ x y: ℕ, p = 3 * x + 2 * y})) ∧ (P.card = 8) :=
sorry

end NUMINAMATH_GPT_basketball_scores_l1311_131166


namespace NUMINAMATH_GPT_num_monic_quadratic_trinomials_l1311_131141

noncomputable def count_monic_quadratic_trinomials : ℕ :=
  4489

theorem num_monic_quadratic_trinomials :
  count_monic_quadratic_trinomials = 4489 :=
by
  sorry

end NUMINAMATH_GPT_num_monic_quadratic_trinomials_l1311_131141


namespace NUMINAMATH_GPT_problem_diamond_value_l1311_131155

def diamond (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem problem_diamond_value :
  diamond 3 4 = 36 := 
by
  sorry

end NUMINAMATH_GPT_problem_diamond_value_l1311_131155


namespace NUMINAMATH_GPT_values_of_2n_plus_m_l1311_131195

theorem values_of_2n_plus_m (n m : ℤ) (h1 : 3 * n - m ≤ 4) (h2 : n + m ≥ 27) (h3 : 3 * m - 2 * n ≤ 45) 
  (h4 : n = 8) (h5 : m = 20) : 2 * n + m = 36 := by
  sorry

end NUMINAMATH_GPT_values_of_2n_plus_m_l1311_131195


namespace NUMINAMATH_GPT_zero_in_P_two_not_in_P_l1311_131119

variables (P : Set Int)

-- Conditions
def condition_1 := ∃ x ∈ P, x > 0 ∧ ∃ y ∈ P, y < 0
def condition_2 := ∃ x ∈ P, x % 2 = 0 ∧ ∃ y ∈ P, y % 2 ≠ 0 
def condition_3 := 1 ∉ P
def condition_4 := ∀ x y, x ∈ P → y ∈ P → x + y ∈ P

-- Proving 0 ∈ P
theorem zero_in_P (h1 : condition_1 P) (h2 : condition_2 P) (h3 : condition_3 P) (h4 : condition_4 P) : 0 ∈ P := 
sorry

-- Proving 2 ∉ P
theorem two_not_in_P (h1 : condition_1 P) (h2 : condition_2 P) (h3 : condition_3 P) (h4 : condition_4 P) : 2 ∉ P := 
sorry

end NUMINAMATH_GPT_zero_in_P_two_not_in_P_l1311_131119


namespace NUMINAMATH_GPT_initial_amount_spent_l1311_131108

theorem initial_amount_spent (X : ℝ) 
    (h_bread : X - 3 ≥ 0) 
    (h_candy : X - 3 - 2 ≥ 0) 
    (h_turkey : X - 3 - 2 - (1/3) * (X - 3 - 2) ≥ 0) 
    (h_remaining : X - 3 - 2 - (1/3) * (X - 3 - 2) = 18) : X = 32 := 
sorry

end NUMINAMATH_GPT_initial_amount_spent_l1311_131108


namespace NUMINAMATH_GPT_function_above_x_axis_l1311_131118

theorem function_above_x_axis (m : ℝ) : 
  (∀ x : ℝ, x > 0 → 9^x - m * 3^x + m + 1 > 0) ↔ m < 2 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_function_above_x_axis_l1311_131118


namespace NUMINAMATH_GPT_painting_combinations_l1311_131178

-- Define the conditions and the problem statement
def top_row_paint_count := 2
def total_lockers_per_row := 4
def valid_paintings := Nat.choose total_lockers_per_row top_row_paint_count

theorem painting_combinations : valid_paintings = 6 := by
  -- Use the derived conditions to provide the proof
  sorry

end NUMINAMATH_GPT_painting_combinations_l1311_131178


namespace NUMINAMATH_GPT_periodic_even_function_l1311_131147

open Real

noncomputable def f : ℝ → ℝ := sorry

theorem periodic_even_function (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ x, f (-x) = f x)
  (h3 : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x = x) :
  ∀ x, -2 ≤ x ∧ x ≤ 0 → f x = 3 - abs (x + 1) :=
sorry

end NUMINAMATH_GPT_periodic_even_function_l1311_131147


namespace NUMINAMATH_GPT_a_and_b_are_kth_powers_l1311_131146

theorem a_and_b_are_kth_powers (k : ℕ) (h_k : 1 < k) (a b : ℤ) (h_rel_prime : Int.gcd a b = 1)
  (c : ℤ) (h_ab_power : a * b = c^k) : ∃ (m n : ℤ), a = m^k ∧ b = n^k :=
by
  sorry

end NUMINAMATH_GPT_a_and_b_are_kth_powers_l1311_131146


namespace NUMINAMATH_GPT_graphs_intersection_points_l1311_131109

theorem graphs_intersection_points {g : ℝ → ℝ} (h_injective : Function.Injective g) :
  ∃ (x1 x2 x3 : ℝ), (g (x1^3) = g (x1^5)) ∧ (g (x2^3) = g (x2^5)) ∧ (g (x3^3) = g (x3^5)) ∧ 
  x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ ∀ (x : ℝ), (g (x^3) = g (x^5)) → (x = x1 ∨ x = x2 ∨ x = x3) := 
by
  sorry

end NUMINAMATH_GPT_graphs_intersection_points_l1311_131109


namespace NUMINAMATH_GPT_avery_donation_l1311_131132

theorem avery_donation (shirts pants shorts : ℕ)
  (h_shirts : shirts = 4)
  (h_pants : pants = 2 * shirts)
  (h_shorts : shorts = pants / 2) :
  shirts + pants + shorts = 16 := by
  sorry

end NUMINAMATH_GPT_avery_donation_l1311_131132


namespace NUMINAMATH_GPT_line_equation_l1311_131106

theorem line_equation (θ : Real) (b : Real) (h1 : θ = 45) (h2 : b = 2) : (y = x + b) :=
by
  -- Assume θ = 45°. The corresponding slope is k = tan(θ) = 1.
  -- Since the y-intercept b = 2, the equation of the line y = mx + b = x + 2.
  sorry

end NUMINAMATH_GPT_line_equation_l1311_131106


namespace NUMINAMATH_GPT_quadratic_expression_neg_for_all_x_l1311_131197

theorem quadratic_expression_neg_for_all_x (m : ℝ) :
  (∀ x : ℝ, m*x^2 + (m-1)*x + (m-1) < 0) ↔ m < -1/3 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_neg_for_all_x_l1311_131197


namespace NUMINAMATH_GPT_problem_a_plus_b_equals_10_l1311_131191

theorem problem_a_plus_b_equals_10 (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) 
  (h_equation : 3 * a + 4 * b = 10 * a + b) : a + b = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_a_plus_b_equals_10_l1311_131191


namespace NUMINAMATH_GPT_monthly_sales_fraction_l1311_131162

theorem monthly_sales_fraction (V S_D T : ℝ) 
  (h1 : S_D = 6 * V) 
  (h2 : S_D = 0.35294117647058826 * T) 
  : V = (1 / 17) * T :=
sorry

end NUMINAMATH_GPT_monthly_sales_fraction_l1311_131162


namespace NUMINAMATH_GPT_cost_price_computer_table_l1311_131138

theorem cost_price_computer_table 
  (CP SP : ℝ)
  (h1 : SP = CP * 1.20)
  (h2 : SP = 8400) :
  CP = 7000 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_computer_table_l1311_131138


namespace NUMINAMATH_GPT_increase_in_expenses_is_20_percent_l1311_131167

noncomputable def man's_salary : ℝ := 6500
noncomputable def initial_savings : ℝ := 0.20 * man's_salary
noncomputable def new_savings : ℝ := 260
noncomputable def reduction_in_savings : ℝ := initial_savings - new_savings
noncomputable def initial_expenses : ℝ := 0.80 * man's_salary
noncomputable def increase_in_expenses_percentage : ℝ := (reduction_in_savings / initial_expenses) * 100

theorem increase_in_expenses_is_20_percent :
  increase_in_expenses_percentage = 20 := by
  sorry

end NUMINAMATH_GPT_increase_in_expenses_is_20_percent_l1311_131167


namespace NUMINAMATH_GPT_eggs_in_box_l1311_131182

theorem eggs_in_box (initial_count : ℝ) (added_count : ℝ) (total_count : ℝ) 
  (h_initial : initial_count = 47.0) 
  (h_added : added_count = 5.0) : total_count = 52.0 :=
by 
  sorry

end NUMINAMATH_GPT_eggs_in_box_l1311_131182


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1311_131174

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x - 1) / (2 - x) ≥ 0} = {x : ℝ | 1/3 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1311_131174


namespace NUMINAMATH_GPT_a_7_value_l1311_131101

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

-- Given conditions
def geometric_sequence_positive_terms (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

def geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (a 0 * (1 - ((a (1 + n)) / a 0))) / (1 - (a 1 / a 0))

def S_4_eq_3S_2 (S : ℕ → ℝ) : Prop :=
S 4 = 3 * S 2

def a_3_eq_2 (a : ℕ → ℝ) : Prop :=
a 3 = 2

-- The statement to prove
theorem a_7_value (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  geometric_sequence_positive_terms a →
  geometric_sequence_sum a S →
  S_4_eq_3S_2 S →
  a_3_eq_2 a →
  a 7 = 8 :=
by
  sorry

end NUMINAMATH_GPT_a_7_value_l1311_131101


namespace NUMINAMATH_GPT_no_real_m_perpendicular_l1311_131142

theorem no_real_m_perpendicular (m : ℝ) : 
  ¬ ∃ m, ((m - 2) * m = -3) := 
sorry

end NUMINAMATH_GPT_no_real_m_perpendicular_l1311_131142


namespace NUMINAMATH_GPT_set_C_is_correct_l1311_131103

open Set

noncomputable def set_A : Set ℝ := {x | x ^ 2 - x - 12 ≤ 0}
noncomputable def set_B : Set ℝ := {x | (x + 1) / (x - 1) < 0}
noncomputable def set_C : Set ℝ := {x | x ∈ set_A ∧ x ∉ set_B}

theorem set_C_is_correct : set_C = {x | -3 ≤ x ∧ x ≤ -1} ∪ {x | 1 ≤ x ∧ x ≤ 4} :=
by
  sorry

end NUMINAMATH_GPT_set_C_is_correct_l1311_131103


namespace NUMINAMATH_GPT_exists_four_integers_multiple_1984_l1311_131169

theorem exists_four_integers_multiple_1984 (a : Fin 97 → ℕ) (h_distinct : Function.Injective a) :
  ∃ i j k l : Fin 97, i ≠ j ∧ k ≠ l ∧ 1984 ∣ (a i - a j) * (a k - a l) :=
sorry

end NUMINAMATH_GPT_exists_four_integers_multiple_1984_l1311_131169


namespace NUMINAMATH_GPT_Faye_created_rows_l1311_131156

theorem Faye_created_rows (total_crayons : ℕ) (crayons_per_row : ℕ) (rows : ℕ) 
  (h1 : total_crayons = 210) (h2 : crayons_per_row = 30) : rows = 7 :=
by
  sorry

end NUMINAMATH_GPT_Faye_created_rows_l1311_131156


namespace NUMINAMATH_GPT_painter_rooms_painted_l1311_131110

theorem painter_rooms_painted (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ) 
    (h1 : total_rooms = 9) (h2 : hours_per_room = 8) (h3 : remaining_hours = 32) : 
    total_rooms - (remaining_hours / hours_per_room) = 5 :=
by
  sorry

end NUMINAMATH_GPT_painter_rooms_painted_l1311_131110


namespace NUMINAMATH_GPT_modulus_of_z_l1311_131114

open Complex

theorem modulus_of_z (z : ℂ) (h : z^2 = (3/4 : ℝ) - I) : abs z = Real.sqrt 5 / 2 := 
  sorry

end NUMINAMATH_GPT_modulus_of_z_l1311_131114


namespace NUMINAMATH_GPT_jonah_total_lemonade_l1311_131124

theorem jonah_total_lemonade : 
  0.25 + 0.4166666666666667 + 0.25 + 0.5833333333333334 = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_jonah_total_lemonade_l1311_131124


namespace NUMINAMATH_GPT_range_of_a_l1311_131177

open Real

theorem range_of_a (a : ℝ) 
  (h : ¬ ∃ x₀ : ℝ, 2 ^ x₀ - 2 ≤ a ^ 2 - 3 * a) : 1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1311_131177


namespace NUMINAMATH_GPT_length_of_train_l1311_131148

noncomputable def train_length : ℕ := 1200

theorem length_of_train 
  (L : ℝ) 
  (speed_km_per_hr : ℝ) 
  (time_min : ℕ) 
  (speed_m_per_s : ℝ) 
  (time_sec : ℕ) 
  (distance : ℝ) 
  (cond1 : L = L)
  (cond2 : speed_km_per_hr = 144) 
  (cond3 : time_min = 1)
  (cond4 : speed_m_per_s = speed_km_per_hr * 1000 / 3600)
  (cond5 : time_sec = time_min * 60)
  (cond6 : distance = speed_m_per_s * time_sec)
  (cond7 : 2 * L = distance)
  : L = train_length := 
sorry

end NUMINAMATH_GPT_length_of_train_l1311_131148


namespace NUMINAMATH_GPT_comparison_l1311_131163

noncomputable def a := Real.log 3000 / Real.log 9
noncomputable def b := Real.log 2023 / Real.log 4
noncomputable def c := (11 * Real.exp (0.01 * Real.log 1.001)) / 2

theorem comparison : a < b ∧ b < c :=
by
  sorry

end NUMINAMATH_GPT_comparison_l1311_131163


namespace NUMINAMATH_GPT_average_height_l1311_131139

theorem average_height (avg1 avg2 : ℕ) (n1 n2 : ℕ) (total_students : ℕ)
  (h1 : avg1 = 20) (h2 : avg2 = 20) (h3 : n1 = 20) (h4 : n2 = 11) (h5 : total_students = 31) :
  (n1 * avg1 + n2 * avg2) / total_students = 20 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_average_height_l1311_131139


namespace NUMINAMATH_GPT_compare_logs_l1311_131113

noncomputable def a := Real.log 2 / Real.log 3
noncomputable def b := Real.log 3 / Real.log 5
noncomputable def c := Real.log 5 / Real.log 8

theorem compare_logs : a < b ∧ b < c := by
  sorry

end NUMINAMATH_GPT_compare_logs_l1311_131113


namespace NUMINAMATH_GPT_ratio_of_capitals_l1311_131131

noncomputable def Ashok_loss (total_loss : ℝ) (Pyarelal_loss : ℝ) : ℝ := total_loss - Pyarelal_loss

theorem ratio_of_capitals (total_loss : ℝ) (Pyarelal_loss : ℝ) (Ashok_capital Pyarelal_capital : ℝ) 
    (h_total_loss : total_loss = 1200)
    (h_Pyarelal_loss : Pyarelal_loss = 1080)
    (h_Ashok_capital : Ashok_capital = 120)
    (h_Pyarelal_capital : Pyarelal_capital = 1080) :
    Ashok_capital / Pyarelal_capital = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_capitals_l1311_131131


namespace NUMINAMATH_GPT_find_x_set_l1311_131102

theorem find_x_set (x : ℝ) : ((x - 2) ^ 2 < 3 * x + 4) ↔ (0 ≤ x ∧ x < 7) := 
sorry

end NUMINAMATH_GPT_find_x_set_l1311_131102


namespace NUMINAMATH_GPT_total_pages_is_905_l1311_131150

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def math_pages : ℕ := (history_pages + geography_pages) / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_is_905 : total_pages = 905 := by
  sorry

end NUMINAMATH_GPT_total_pages_is_905_l1311_131150


namespace NUMINAMATH_GPT_expression_evaluation_l1311_131199

theorem expression_evaluation :
  10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1311_131199


namespace NUMINAMATH_GPT_last_integer_in_sequence_is_one_l1311_131100

theorem last_integer_in_sequence_is_one :
  ∀ seq : ℕ → ℕ, (seq 0 = 37) ∧ (∀ n, seq (n + 1) = seq n / 2) → (∃ n, seq (n + 1) = 0 ∧ seq n = 1) :=
by
  sorry

end NUMINAMATH_GPT_last_integer_in_sequence_is_one_l1311_131100


namespace NUMINAMATH_GPT_bake_sale_comparison_l1311_131117

theorem bake_sale_comparison :
  let tamara_small_brownies := 4 * 2
  let tamara_large_brownies := 12 * 3
  let tamara_cookies := 36 * 1.5
  let tamara_total := tamara_small_brownies + tamara_large_brownies + tamara_cookies

  let sarah_muffins := 24 * 1.75
  let sarah_choco_cupcakes := 7 * 2.5
  let sarah_vanilla_cupcakes := 8 * 2
  let sarah_strawberry_cupcakes := 15 * 2.75
  let sarah_total := sarah_muffins + sarah_choco_cupcakes + sarah_vanilla_cupcakes + sarah_strawberry_cupcakes

  sarah_total - tamara_total = 18.75 := by
  sorry

end NUMINAMATH_GPT_bake_sale_comparison_l1311_131117


namespace NUMINAMATH_GPT_exists_n_geq_k_l1311_131165

theorem exists_n_geq_k (a : ℕ → ℕ) (h_distinct : ∀ i j : ℕ, i ≠ j → a i ≠ a j) 
    (h_positive : ∀ i : ℕ, a i > 0) :
    ∀ k : ℕ, ∃ n : ℕ, n > k ∧ a n ≥ n :=
by
  intros k
  sorry

end NUMINAMATH_GPT_exists_n_geq_k_l1311_131165


namespace NUMINAMATH_GPT_equal_acutes_l1311_131127

open Real

theorem equal_acutes (a b c : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2) (hc : 0 < c ∧ c < π / 2)
  (h1 : sin b = (sin a + sin c) / 2) (h2 : cos b ^ 2 = cos a * cos c) : a = b ∧ b = c := 
by
  -- We have to fill the proof steps here.
  sorry

end NUMINAMATH_GPT_equal_acutes_l1311_131127


namespace NUMINAMATH_GPT_part1_part2_l1311_131104

-- Define set A
def set_A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }

-- Define set B depending on m
def set_B (m : ℝ) : Set ℝ := { x | 2 * m - 1 ≤ x ∧ x ≤ m + 1 }

-- Part 1: When m = -3, find A ∩ B
theorem part1 : set_B (-3) ∩ set_A = { x | -3 ≤ x ∧ x ≤ -2 } := 
sorry

-- Part 2: Find the range of m such that B ⊆ A
theorem part2 (m : ℝ) : set_B m ⊆ set_A ↔ m ≥ -1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1311_131104


namespace NUMINAMATH_GPT_faucet_leakage_volume_l1311_131115

def leakage_rate : ℝ := 0.1
def time_seconds : ℝ := 14400
def expected_volume : ℝ := 1.4 * 10^3

theorem faucet_leakage_volume : 
  leakage_rate * time_seconds = expected_volume := 
by
  -- proof
  sorry

end NUMINAMATH_GPT_faucet_leakage_volume_l1311_131115


namespace NUMINAMATH_GPT_smallest_n_inequality_l1311_131135

variable {x y z : ℝ}

theorem smallest_n_inequality :
  ∃ (n : ℕ), (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
    (∀ m : ℕ, (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ m * (x^4 + y^4 + z^4)) → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_n_inequality_l1311_131135


namespace NUMINAMATH_GPT_new_savings_after_expense_increase_l1311_131175

theorem new_savings_after_expense_increase
    (monthly_salary : ℝ)
    (initial_saving_percent : ℝ)
    (expense_increase_percent : ℝ)
    (initial_salary : monthly_salary = 20000)
    (saving_rate : initial_saving_percent = 0.1)
    (increase_rate : expense_increase_percent = 0.1) :
    monthly_salary - (monthly_salary * (1 - initial_saving_percent + (1 - initial_saving_percent) * expense_increase_percent)) = 200 :=
by
  sorry

end NUMINAMATH_GPT_new_savings_after_expense_increase_l1311_131175


namespace NUMINAMATH_GPT_import_tax_applied_amount_l1311_131151

theorem import_tax_applied_amount 
    (total_value : ℝ) 
    (import_tax_paid : ℝ)
    (tax_rate : ℝ) 
    (excess_amount : ℝ) 
    (condition1 : total_value = 2580) 
    (condition2 : import_tax_paid = 110.60) 
    (condition3 : tax_rate = 0.07) 
    (condition4 : import_tax_paid = tax_rate * (total_value - excess_amount)) : 
    excess_amount = 1000 :=
by
  sorry

end NUMINAMATH_GPT_import_tax_applied_amount_l1311_131151


namespace NUMINAMATH_GPT_winning_candidate_percentage_l1311_131120

theorem winning_candidate_percentage
  (total_votes : ℕ)
  (vote_majority : ℕ)
  (winning_candidate_votes : ℕ)
  (losing_candidate_votes : ℕ) :
  total_votes = 400 →
  vote_majority = 160 →
  winning_candidate_votes = total_votes * 70 / 100 →
  losing_candidate_votes = total_votes - winning_candidate_votes →
  winning_candidate_votes - losing_candidate_votes = vote_majority →
  winning_candidate_votes = 280 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l1311_131120


namespace NUMINAMATH_GPT_average_number_of_visitors_is_25_l1311_131164

-- Define the sequence parameters
def a : ℕ := 10  -- First term
def d : ℕ := 5   -- Common difference
def n : ℕ := 7   -- Number of days

-- Define the sequence for the number of visitors on each day
def visitors (i : ℕ) : ℕ := a + (i - 1) * d

-- Define the average number of visitors
def avg_visitors : ℕ := (List.sum (List.map visitors [1, 2, 3, 4, 5, 6, 7])) / n

-- Prove the average
theorem average_number_of_visitors_is_25 : avg_visitors = 25 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_average_number_of_visitors_is_25_l1311_131164


namespace NUMINAMATH_GPT_func_inequality_l1311_131143

noncomputable def f (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Given function properties
variables {a b c : ℝ} (h_a : a > 0) (symmetry : ∀ x : ℝ, f a b c (2 + x) = f a b c (2 - x))

theorem func_inequality : f a b c 2 < f a b c 1 ∧ f a b c 1 < f a b c 4 :=
by
  sorry

end NUMINAMATH_GPT_func_inequality_l1311_131143


namespace NUMINAMATH_GPT_product_base9_l1311_131168

open Nat

noncomputable def base9_product (a b : ℕ) : ℕ := 
  let a_base10 := 3*9^2 + 6*9^1 + 2*9^0
  let b_base10 := 7
  let product_base10 := a_base10 * b_base10
  -- converting product_base10 from base 10 to base 9
  2 * 9^3 + 8 * 9^2 + 7 * 9^1 + 5 * 9^0 -- which simplifies to 2875 in base 9

theorem product_base9: base9_product 362 7 = 2875 :=
by
  -- Here should be the proof or a computational check
  sorry

end NUMINAMATH_GPT_product_base9_l1311_131168


namespace NUMINAMATH_GPT_sum_areas_of_square_and_rectangle_l1311_131111

theorem sum_areas_of_square_and_rectangle (s w l : ℝ) 
  (h1 : s^2 + w * l = 130)
  (h2 : 4 * s - 2 * (w + l) = 20)
  (h3 : l = 2 * w) : 
  s^2 + 2 * w^2 = 118 :=
by
  -- Provide space for proof
  sorry

end NUMINAMATH_GPT_sum_areas_of_square_and_rectangle_l1311_131111


namespace NUMINAMATH_GPT_integer_sided_triangle_with_60_degree_angle_exists_l1311_131185

theorem integer_sided_triangle_with_60_degree_angle_exists 
  (m n t : ℤ) : 
  ∃ (x y z : ℤ), (x = (m^2 - n^2) * t) ∧ 
                  (y = m * (m - 2 * n) * t) ∧ 
                  (z = (m^2 - m * n + n^2) * t) := by
  sorry

end NUMINAMATH_GPT_integer_sided_triangle_with_60_degree_angle_exists_l1311_131185


namespace NUMINAMATH_GPT_find_k_for_circle_radius_5_l1311_131116

theorem find_k_for_circle_radius_5 (k : ℝ) :
  (∃ x y : ℝ, (x^2 + 12 * x + y^2 + 8 * y - k = 0)) → k = -27 :=
by
  sorry

end NUMINAMATH_GPT_find_k_for_circle_radius_5_l1311_131116


namespace NUMINAMATH_GPT_fill_time_correct_l1311_131154

-- Define the conditions
def rightEyeTime := 2 * 24 -- hours
def leftEyeTime := 3 * 24 -- hours
def rightFootTime := 4 * 24 -- hours
def throatTime := 6       -- hours

def rightEyeRate := 1 / rightEyeTime
def leftEyeRate := 1 / leftEyeTime
def rightFootRate := 1 / rightFootTime
def throatRate := 1 / throatTime

-- Combined rate calculation
def combinedRate := rightEyeRate + leftEyeRate + rightFootRate + throatRate

-- Goal definition
def fillTime := 288 / 61 -- hours

-- Prove that the calculated time to fill the pool matches the given answer
theorem fill_time_correct : (1 / combinedRate) = fillTime :=
by {
  sorry
}

end NUMINAMATH_GPT_fill_time_correct_l1311_131154
