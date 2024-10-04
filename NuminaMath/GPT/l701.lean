import Mathlib
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Fin
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Setoid.Basic
import Mathlib.Data.Sort
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic

namespace second_derivative_eq_function_l701_701049

def f : ℝ → ℝ := fun x => x

def g : ℝ → ℝ := sin

def h : ℝ → ℝ := exp

def k : ℝ → ℝ := log

theorem second_derivative_eq_function :
  (∀ x, f'' x ≠ f x) ∧
  (∀ x, (derivative^[2] g) x ≠ g x) ∧
  (∀ x, (derivative^[2] h) x = h x) ∧
  (∀ x, (derivative^[2] k) x ≠ k x) :=
by
  -- Here proofs would follow
  simp only [f, g, h, k, (@derivative^[2])]; sorry

end second_derivative_eq_function_l701_701049


namespace largest_n_unique_k_l701_701918

theorem largest_n_unique_k :
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, (3 : ℚ) / 7 < (n : ℚ) / ((n + k : ℕ) : ℚ) ∧ 
  (n : ℚ) / ((n + k : ℕ) : ℚ) < (8 : ℚ) / 19 → k = 1 := by
sorry

end largest_n_unique_k_l701_701918


namespace projection_theorem_l701_701914

noncomputable def projection_property (A B C T P : Point) (h : is_right_triangle A B C) 
(ha : ∠CAB = 90) (hb : angle_bisector A C B T) 
(hc : perpendicular A AT P) : Prop :=
  lies_on_midline P A B C

theorem projection_theorem (A B C T P : Point) (h₁ : is_right_triangle A B C) 
(h₂ : ∠CAB = 90) (h₃ : angle_bisector A C B T) 
(h₄ : perpendicular A AT P) : lies_on_midline P A B C :=
sorry

end projection_theorem_l701_701914


namespace domain_of_f_l701_701840

noncomputable def f (x : ℝ) := 1 / Real.log (x + 1) + Real.sqrt (9 - x^2)

theorem domain_of_f :
  {x : ℝ | 0 < x + 1} ∩ {x : ℝ | x ≠ 0} ∩ {x : ℝ | 9 - x^2 ≥ 0} = (Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioc 0 (3 : ℝ)) :=
by
  sorry

end domain_of_f_l701_701840


namespace rationalize_denominator_of_seven_over_sqrt_343_l701_701455

noncomputable def rationalize_denominator (x : ℝ) : ℝ := x * sqrt 7 / 7

theorem rationalize_denominator_of_seven_over_sqrt_343 :
  (343 = 7^3) → (sqrt 343 = 7 * sqrt 7) →
  (7 / sqrt 343 = sqrt 7 / 7) :=
by
  intros h1 h2
  sorry

end rationalize_denominator_of_seven_over_sqrt_343_l701_701455


namespace prime_p_sum_of_squares_l701_701292

theorem prime_p_sum_of_squares (p : ℕ) (hp : p.Prime) 
  (h : ∃ (a : ℕ), 2 * p = a^2 + (a + 1)^2 + (a + 2)^2 + (a + 3)^2) : 
  36 ∣ (p - 7) :=
by 
  sorry

end prime_p_sum_of_squares_l701_701292


namespace bianca_total_cupcakes_l701_701626

theorem bianca_total_cupcakes : 
  ∃ (initial_cupcakes remaining_cupcakes additional_cupcakes : ℝ),
  initial_cupcakes = 14 ∧ 
  remaining_cupcakes = initial_cupcakes - 10 ∧
  additional_cupcakes = 4 ∧
  remaining_cupcakes + additional_cupcakes = 8 :=
begin
  use 14,
  use 4,
  use 4,
  split,
  { refl, },
  split,
  { 
    calc 14 - 10 = 4 : by norm_num,
  },
  { 
    calc 4 + 4 = 8 : by norm_num,
  },
end

end bianca_total_cupcakes_l701_701626


namespace michael_eggs_count_l701_701442

-- Define the conditions: crates bought on Tuesday, crates given to Susan, crates bought on Thursday, and eggs per crate.
def crates_bought_on_tuesday : ℕ := 6
def crates_given_to_susan : ℕ := 2
def crates_bought_on_thursday : ℕ := 5
def eggs_per_crate : ℕ := 30

-- State the theorem to prove.
theorem michael_eggs_count :
  let crates_left = crates_bought_on_tuesday - crates_given_to_susan
  let total_crates = crates_left + crates_bought_on_thursday
  total_crates * eggs_per_crate = 270 :=
by
  -- Proof goes here
  sorry

end michael_eggs_count_l701_701442


namespace find_m_l701_701334

theorem find_m {m : ℝ} (a b : ℝ × ℝ) (H : a = (3, m) ∧ b = (2, -1)) (H_dot : a.1 * b.1 + a.2 * b.2 = 0) : m = 6 := 
by
  sorry

end find_m_l701_701334


namespace sum_a_100_l701_701686

-- Define the sequence according to the given conditions
def a : ℕ → ℕ
| 1     := 1
| (2 * n) := n - a n
| (2 * n + 1) := a n + 1

-- Prove that the sum of the first 100 terms of the sequence is 1306
theorem sum_a_100 : (∑ i in Finset.range 100, a (i + 1)) = 1306 :=
sorry

end sum_a_100_l701_701686


namespace gcd_1987_2025_l701_701915

theorem gcd_1987_2025 : Nat.gcd 1987 2025 = 1 := by
  sorry

end gcd_1987_2025_l701_701915


namespace calculate_expression_l701_701232

noncomputable theory

def expr1 : ℝ := real.sqrt 16
def expr2 : ℝ := abs (real.sqrt 2 - 2)
def expr3 : ℝ := real.cbrt (-64)
def expr4 : ℝ := 2 * (1 + real.sqrt 2)

theorem calculate_expression : expr1 + expr2 + expr3 - expr4 = -3 * real.sqrt 2 :=
by 
  sorry

end calculate_expression_l701_701232


namespace smallest_x_l701_701107

theorem smallest_x (M x : ℕ) (h : 720 * x = M^3) : x = 300 :=
by
  sorry

end smallest_x_l701_701107


namespace building_shadow_length_l701_701590

theorem building_shadow_length
  (flagpole_height : ℝ) (flagpole_shadow : ℝ) (building_height : ℝ)
  (h_flagpole : flagpole_height = 18) (s_flagpole : flagpole_shadow = 45) 
  (h_building : building_height = 26) :
  ∃ (building_shadow : ℝ), (building_height / building_shadow = flagpole_height / flagpole_shadow) ∧ building_shadow = 65 :=
by
  use 65
  sorry

end building_shadow_length_l701_701590


namespace largest_three_digit_solution_l701_701127

theorem largest_three_digit_solution :
  ∃ (n : ℤ), 100 ≤ n ∧ n < 1000 ∧ 70 * n ≡ 210 [MOD 350] ∧
             ∀ m, 100 ≤ m ∧ m < 1000 ∧ 70 * m ≡ 210 [MOD 350] → m ≤ n :=
begin
  sorry
end

end largest_three_digit_solution_l701_701127


namespace abs_sum_of_roots_l701_701276

theorem abs_sum_of_roots (m p q r : ℤ) (h1 : PolynomialRootsInt (Polynomial.C m + Polynomial.X^3 - 707 * Polynomial.X))
  (h2 : Polynomial.C (m, p, q, r : ℤ)
  (h3 : ∃ r, ∃ m, (m: ℤ) = -1*(p+q)

:= abs p + abs q + abs r = 122 :=
begin
  sorry,
end

end abs_sum_of_roots_l701_701276


namespace two_lines_perpendicular_to_same_plane_are_parallel_l701_701283

variables {Plane Line : Type} 
variables (perp : Line → Plane → Prop) (parallel : Line → Line → Prop)

theorem two_lines_perpendicular_to_same_plane_are_parallel
  (a b : Line) (α : Plane) (ha : perp a α) (hb : perp b α) : parallel a b :=
sorry

end two_lines_perpendicular_to_same_plane_are_parallel_l701_701283


namespace rancher_unique_solution_l701_701563

-- Defining the main problem statement
theorem rancher_unique_solution : ∃! (b h : ℕ), 30 * b + 32 * h = 1200 ∧ b > h := by
  sorry

end rancher_unique_solution_l701_701563


namespace raking_leaves_l701_701445

theorem raking_leaves (T : ℕ) : (∃ (k : ℕ), k = 3 ∧ T = 15 * (8 / k)) → T = 40 :=
by
  intro h
  obtain ⟨k, hk, hT⟩ := h
  rw [hk, mul_div_cancel' _ (nat.cast_pos.mpr (nat.succ_pos _))] at hT
  rw [mul_comm, nat.cast_mul, nat.cast_bit0, nat.cast_one] at hT
  norm_cast at hT
  exact hT

end raking_leaves_l701_701445


namespace complex_equality_solution_l701_701672

noncomputable def z1 (x y : ℝ) : ℂ := 4 * complex.I - 2 * x * y - x * y * complex.I
noncomputable def z2 (x y : ℝ) : ℂ := y^2 * complex.I - x^2 + 3

theorem complex_equality_solution (x y : ℝ) :
  z1 x y = z2 x y ↔ 
  (x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = -1) ∨ 
  (x = sqrt 3 / 3 ∧ y = -4 * sqrt 3 / 3) ∨ 
  (x = -sqrt 3 / 3 ∧ y = 4 * sqrt 3 / 3) :=
by sorry

end complex_equality_solution_l701_701672


namespace optimal_tax_choice_l701_701138

def TotalIncome (revenue advances : ℕ) : ℕ :=
  revenue + advances

def MonthlyExpenses (rent oils salaries insurance accounting advertising retraining misc : ℕ) : ℕ :=
  rent + oils + salaries + insurance + accounting + advertising + retraining + misc

def AnnualExpenses (monthlyExpenses : ℕ) : ℕ :=
  monthlyExpenses * 12

def IncomeTax (totalIncome : ℕ) (rate : ℕ) : ℕ :=
  totalIncome * rate / 100

def ExpenditureTax (totalIncome annualExpenses : ℕ) (rate : ℕ) : ℕ :=
  (totalIncome - annualExpenses) * rate / 100

def MinimumTax (totalIncome : ℕ) (rate : ℕ) : ℕ :=
  totalIncome * rate / 100

def FinalTaxPayable (incomeTax deductionLimit : ℕ) : ℕ :=
  incomeTax - deductionLimit

theorem optimal_tax_choice 
  (revenue advances rent oils salaries insurance accounting advertising retraining misc : ℕ)
  (insuranceLimitRate incomeTaxRate expenditureTaxRate minTaxRate : ℕ) :
  let totalIncome := TotalIncome revenue advances,
      monthlyExpenses := MonthlyExpenses rent oils salaries insurance accounting advertising retraining misc,
      annualExpenses := AnnualExpenses monthlyExpenses,
      incomeTaxBase := totalIncome,
      expenditureTaxBase := totalIncome - annualExpenses,
      incomeTax := IncomeTax incomeTaxBase incomeTaxRate,
      deductionLimit := incomeTax * insuranceLimitRate / 100,
      incomeTaxPayable := FinalTaxPayable incomeTax deductionLimit,
      expenditureTax := ExpenditureTax totalIncome annualExpenses expenditureTaxRate,
      minTax := MinimumTax totalIncome minTaxRate in
  (totalIncome = revenue + advances ∧ 
  monthlyExpenses = rent + oils + salaries + insurance + accounting + advertising + retraining + misc ∧
  annualExpenses = monthlyExpenses * 12 ∧
  incomeTax = incomeTaxBase * incomeTaxRate / 100 ∧
  deductionLimit = incomeTax * insuranceLimitRate / 100 ∧
  incomeTaxPayable = incomeTax - deductionLimit ∧
  expenditureTax = (totalIncome - annualExpenses) * expenditureTaxRate / 100 ∧
  minTax = totalIncome * minTaxRate / 100 ∧
  incomeTaxPayable > minTax) → 
  (IncomeTaxPayable > MinTax) 
  :=
begin
  sorry
end

end optimal_tax_choice_l701_701138


namespace num_squares_below_line_l701_701867

theorem num_squares_below_line : 
  let f (x : ℕ) := -(12 / 180) * x + 12 in
  ∃ n : ℕ, n = 984 ∧
    ∀ (x y : ℕ), x ≥ 0 ∧ y ≥ 0 ∧ y < f x → x * y < 12 * 180 := 
sorry

end num_squares_below_line_l701_701867


namespace solve_binomial_equation_l701_701108

theorem solve_binomial_equation (x : ℕ) :
  binomial 11 x = binomial 11 (2 * x - 4) ↔ (x = 4 ∨ x = 5) :=
by
  sorry

end solve_binomial_equation_l701_701108


namespace range_of_z_l701_701281

theorem range_of_z (a b : ℝ) (h1 : 2 < a) (h2 : a < 3) (h3 : -2 < b) (h4 : b < -1) :
  5 < 2 * a - b ∧ 2 * a - b < 8 :=
by
  sorry

end range_of_z_l701_701281


namespace slope_of_parallel_line_l701_701966

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l701_701966


namespace smallest_positive_integer_multiple_of_6_and_15_is_30_l701_701263

theorem smallest_positive_integer_multiple_of_6_and_15_is_30 :
  ∃ b : ℕ, b > 0 ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ ∀ n, (n > 0 ∧ (6 ∣ n) ∧ (15 ∣ n)) → b ≤ n :=
  let b := 30 in
  ⟨b, by simp [b, dvd_refl, nat.succ_pos'], sorry⟩

end smallest_positive_integer_multiple_of_6_and_15_is_30_l701_701263


namespace product_of_18396_and_9999_l701_701666

theorem product_of_18396_and_9999 : 18396 * 9999 = 183962604 :=
by
  sorry

end product_of_18396_and_9999_l701_701666


namespace max_value_of_squared_distances_l701_701040

theorem max_value_of_squared_distances (z : ℂ) (h : complex.abs (z - (3 - 3 * complex.I)) = 4) : 
  ∃ x : ℝ, x = 26.088 ∧ (complex.abs (z - (2 + complex.I))^2 + complex.abs (z - (6 - 2 * complex.I))^2 ≤ x) := 
sorry

end max_value_of_squared_distances_l701_701040


namespace volume_of_prism_l701_701078

theorem volume_of_prism (x y z : ℝ) (hx : x * y = 28) (hy : x * z = 45) (hz : y * z = 63) : x * y * z = 282 := by
  sorry

end volume_of_prism_l701_701078


namespace scientific_notation_representation_l701_701390

-- Define the concept of nanoseconds in seconds
def nanosecond_in_seconds : ℝ := 10 ^ (-9)

-- Define the commitment time in nanoseconds
def commitment_time_ns : ℝ := 20

-- The goal is to prove that the commitment time in scientific notation is 2 × 10^(-8) seconds
theorem scientific_notation_representation : (commitment_time_ns * nanosecond_in_seconds) = 2 * 10^(-8) :=
by
  sorry

end scientific_notation_representation_l701_701390


namespace equivalent_interest_rate_l701_701061

noncomputable def quarterly_interest_rate (annual_rate : ℝ) : ℝ :=
  annual_rate / 4

noncomputable def effective_annual_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate)^4

noncomputable def equivalent_annual_rate (effective_rate : ℝ) : ℝ :=
  (effective_rate - 1) * 100

theorem equivalent_interest_rate (annual_rate : ℝ) (h : annual_rate = 10) 
  : equivalent_annual_rate (effective_annual_rate (quarterly_interest_rate annual_rate)) ≈ 10.17 :=
by
  sorry

end equivalent_interest_rate_l701_701061


namespace slope_of_parallel_line_l701_701989

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l701_701989


namespace circle_equation_exists_l701_701857

theorem circle_equation_exists :
  ∃ D E F : ℝ, (∀ (x y : ℝ), (x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
                      (x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
                      (x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
                      (D = -4) ∧ (E = -6) ∧ (F = 0) :=
by
  use [-4, -6, 0]
  intro x y
  split
  { intros hx hy, simp [hx, hy] }
  split
  { intros hx hy, simp [hx, hy], linarith }
  { intros hx hy, simp [hx, hy], linarith }
  sorry

end circle_equation_exists_l701_701857


namespace determine_radius_l701_701242

noncomputable def radius_of_circle (x y : ℝ) (h : x^2 + y^2 - 8 = 2x + 4y) : ℝ :=
√13

theorem determine_radius (x y : ℝ) (h : x^2 + y^2 - 8 = 2x + 4y) : radius_of_circle x y h = √13 :=
by
  sorry

end determine_radius_l701_701242


namespace smallest_number_is_minus_three_l701_701222

noncomputable def smallest_number : ℚ :=
  let numbers := {-1, 0, 1, -3}
  in (if -3 ∈ numbers then -3 else 0) -- This is a placeholder to define the set

theorem smallest_number_is_minus_three :
  ∀ x ∈ {-1, 0, 1, -3}, x ≥ -3 :=
by
  intro x hx
  fin_cases hx <;> simp <;> linarith

end smallest_number_is_minus_three_l701_701222


namespace plane_angle_sine_relation_l701_701876

-- Define the plane angles α, β, γ and edge angles a, b, c
variables (α β γ a b c : Real)

-- State the theorem to prove the required relationship
theorem plane_angle_sine_relation
  (h : sin γ ≠ 0 ∧ sin b ≠ 0 ∧ sin c ≠ 0 ∧ sin a ≠ 0 ∧ sin β ≠ 0 ∧ sin α ≠ 0) :
  sin α * sin a = sin β * sin b ∧ sin β * sin b = sin γ * sin c := sorry

end plane_angle_sine_relation_l701_701876


namespace slope_of_parallel_line_l701_701955

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l701_701955


namespace eunji_received_900_won_l701_701565

-- Define the conditions
def eunji_pocket_money (X : ℝ) : Prop :=
  (X / 2 + 550 = 1000)

-- Define the theorem to prove the question equals the correct answer
theorem eunji_received_900_won {X : ℝ} (h : eunji_pocket_money X) : X = 900 :=
  by
    sorry

end eunji_received_900_won_l701_701565


namespace eleventh_number_in_ordered_list_of_digit_sum_12_l701_701220

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def ordered_list_of_digit_sum (s : ℕ) : List ℕ :=
  List.filter (λ n => digit_sum n = s) ((List.range 1000).map (λ x => x + 1)).sort

theorem eleventh_number_in_ordered_list_of_digit_sum_12 : 
  ordered_list_of_digit_sum 12 !! 10 = 156 := 
by 
  sorry

end eleventh_number_in_ordered_list_of_digit_sum_12_l701_701220


namespace calculate_total_cost_l701_701438

def num_chicken_nuggets := 100
def num_per_box := 20
def cost_per_box := 4

theorem calculate_total_cost :
  (num_chicken_nuggets / num_per_box) * cost_per_box = 20 := by
  sorry

end calculate_total_cost_l701_701438


namespace exceeds_500_bacteria_l701_701366

noncomputable def bacteria_count (n : Nat) : Nat :=
  4 * 3^n

theorem exceeds_500_bacteria (n : Nat) (h : 4 * 3^n > 500) : n ≥ 6 :=
by
  sorry

end exceeds_500_bacteria_l701_701366


namespace dot_product_parallel_vectors_is_minus_ten_l701_701333

-- Definitions from the conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -4)
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2) ∨ v = (k * u.1, k * u.2)

theorem dot_product_parallel_vectors_is_minus_ten (x : ℝ) (h : are_parallel vector_a (vector_b x)) : (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2) = -10 :=
by
  sorry

end dot_product_parallel_vectors_is_minus_ten_l701_701333


namespace parallel_line_slope_l701_701993

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l701_701993


namespace min_time_to_cross_river_l701_701613

-- Definitions for the time it takes each horse to cross the river
def timeA : ℕ := 2
def timeB : ℕ := 3
def timeC : ℕ := 7
def timeD : ℕ := 6

-- Definition for the minimum time required for all horses to cross the river
def min_crossing_time : ℕ := 18

-- Theorem stating the problem: 
theorem min_time_to_cross_river :
  ∀ (timeA timeB timeC timeD : ℕ), timeA = 2 → timeB = 3 → timeC = 7 → timeD = 6 →
  min_crossing_time = 18 :=
sorry

end min_time_to_cross_river_l701_701613


namespace min_distance_from_curve_to_line_l701_701354

def curve (x : ℝ) : ℝ := (3 / 2) * x^2 - 2 * Real.log x

def line (x : ℝ) : ℝ := x - (5 / 2)

theorem min_distance_from_curve_to_line :
  let P := (1 : ℝ, (3 / 2) : ℝ)
  let distance (p1 p2 : ℝ × ℝ) := abs (p1.1 - p2.1) / Real.sqrt 2
  distance P ((1, (3 / 2)) : ℝ × ℝ) = (3 * Real.sqrt 2) / 2 :=
sorry

end min_distance_from_curve_to_line_l701_701354


namespace determine_winning_strategy_l701_701790

noncomputable def winning_strategy (n : ℕ) (h : n ≥ 2) : Prop :=
  if n = 2 ∨ n = 4 ∨ n = 8 then
    "Ariane has a winning strategy"
  else
    "Bérénice has a winning strategy"

theorem determine_winning_strategy (n : ℕ) (h : n ≥ 2) :
  winning_strategy n h :=
sorry

end determine_winning_strategy_l701_701790


namespace part1_part2_l701_701760

theorem part1 (m : ℝ) (P : ℝ × ℝ) : (P = (3*m - 6, m + 1)) → (P.1 = 0) → (P = (0, 3)) :=
by
  sorry

theorem part2 (m : ℝ) (A P : ℝ × ℝ) : A = (1, -2) → (P = (3*m - 6, m + 1)) → (P.2 = A.2) → (P = (-15, -2)) :=
by
  sorry

end part1_part2_l701_701760


namespace min_h_for_circle_l701_701288

theorem min_h_for_circle (h : ℝ) :
  (∀ x y : ℝ, (x - h)^2 + (y - 1)^2 = 1 → x + y + 1 ≥ 0) →
  h = Real.sqrt 2 - 2 :=
sorry

end min_h_for_circle_l701_701288


namespace sum_of_three_integers_l701_701099

theorem sum_of_three_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) (h7 : a * b * c = 5^3) : a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_l701_701099


namespace Maddie_bought_palettes_l701_701436

-- Defining constants and conditions as per the problem statement.
def cost_per_palette : ℝ := 15
def number_of_lipsticks : ℝ := 4
def cost_per_lipstick : ℝ := 2.50
def number_of_hair_boxes : ℝ := 3
def cost_per_hair_box : ℝ := 4
def total_paid : ℝ := 67

-- Defining the condition which we need to prove for number of makeup palettes bought.
theorem Maddie_bought_palettes (P : ℝ) :
  (number_of_lipsticks * cost_per_lipstick) +
  (number_of_hair_boxes * cost_per_hair_box) +
  (cost_per_palette * P) = total_paid →
  P = 3 :=
sorry

end Maddie_bought_palettes_l701_701436


namespace area_EBF_l701_701400

def triangle_area (A B C : Point) : ℝ := sorry -- Definition of triangle area

-- Definition of points and line segments
variables (A B C D E F : Point)

-- Conditions
def AD_DC_ratio := (AD_length D C) = (1 : ℝ) / (2 : ℝ)
def E_midpoint_BD := midpoint E B D
def AE_intersects_BC_at_F := intersects A E B C F
def ABC_area := (triangle_area A B C) = 360

-- Proof statement
theorem area_EBF :
  AD_DC_ratio →
  E_midpoint_BD →
  AE_intersects_BC_at_F →
  ABC_area →
  (triangle_area E B F) = 30 :=
by
  intros
  sorry

end area_EBF_l701_701400


namespace truck_gas_consumption_l701_701228

theorem truck_gas_consumption :
  ∀ (initial_gasoline total_distance remaining_gasoline : ℝ),
    initial_gasoline = 12 →
    total_distance = (2 * 5 + 2 + 2 * 2 + 6) →
    remaining_gasoline = 2 →
    (initial_gasoline - remaining_gasoline) ≠ 0 →
    (total_distance / (initial_gasoline - remaining_gasoline)) = 2.2 :=
by
  intros initial_gasoline total_distance remaining_gasoline
  intro h_initial_gas h_total_distance h_remaining_gas h_non_zero
  sorry

end truck_gas_consumption_l701_701228


namespace length_of_7kg_rod_l701_701746

-- Define the given conditions
def weight_per_meter (total_weight : ℝ) (total_length : ℝ) : ℝ :=
  total_weight / total_length

def proportional_length (weight : ℝ) (weight_per_meter : ℝ) : ℝ :=
  weight / weight_per_meter

-- Assume the given conditions
variables (weight12kg : ℝ) (length12m : ℝ) (weight7kg : ℝ)
hypothesis (h1 : weight12kg = 14) (h2 : length12m = 12) (h3 : weight7kg = 7)

-- The final theorem to be proven
theorem length_of_7kg_rod :
  let w_m := weight_per_meter weight12kg length12m in
  proportional_length weight7kg w_m = 6 := sorry

end length_of_7kg_rod_l701_701746


namespace surface_area_ratio_proof_volume_ratio_proof_l701_701067

noncomputable def surface_area_ratio (a b c : ℝ) (ha : a = b) (hc : c = a * Real.sqrt 2) : ℝ × ℝ × ℝ :=
let F_a := a^2 * Real.pi * (3 + Real.sqrt 2),
    F_b := a^2 * Real.pi * (3 + Real.sqrt 2),
    F_c := a^2 * Real.pi * (2 + Real.sqrt 2) in
let ratio := (1, 1, (F_c / F_a)) in
ratio

noncomputable def volume_ratio (a b c : ℝ) (ha : a = b) (hc : c = a * Real.sqrt 2) : ℝ × ℝ × ℝ :=
let K_a := (2/3) * Real.pi * a^3,
    K_b := (2/3) * Real.pi * a^3,
    K_c := (Real.sqrt 2 / 3) * Real.pi * a^3 in
let ratio := (1, 1, K_c / K_a) in
ratio

theorem surface_area_ratio_proof (a b c : ℝ) (ha : a = b) (hc : c = a * Real.sqrt 2) :
    surface_area_ratio a b c ha hc = (1, 1, (4 - Real.sqrt 2) / 7) :=
by
    sorry

theorem volume_ratio_proof (a b c : ℝ) (ha : a = b) (hc : c = a * Real.sqrt 2) :
    volume_ratio a b c ha hc = (1, 1, Real.sqrt 2 / 3) :=
by
    sorry

end surface_area_ratio_proof_volume_ratio_proof_l701_701067


namespace coefficient_x4y3_in_expansion_l701_701657

/-- Using the binomial theorem properties to prove the coefficient of x^4 y^3
    in the expansion of (1+x)^6 * (2+y)^4 is 120. -/
theorem coefficient_x4y3_in_expansion :
  let binom := λ n k, nat.choose n k in
  binom 6 4 * binom 4 3 * 2 = 120 :=
by {
  let binom := λ n k, nat.choose n k,
  sorry
}

end coefficient_x4y3_in_expansion_l701_701657


namespace tv_weekly_cost_l701_701019

theorem tv_weekly_cost(
  (tv_power : ℕ) (tv_hours_per_day : ℕ) (cost_per_kWh : ℕ)
  (kilowatt_in_watts : ℕ) (dollar_in_cents : ℕ)
  (h1 : tv_power = 125) (h2 : tv_hours_per_day = 4) 
  (h3 : cost_per_kWh = 14) (h4 : kilowatt_in_watts = 1000) 
  (h5 : dollar_in_cents = 100) :
  (tv_power * tv_hours_per_day * 7 / kilowatt_in_watts * cost_per_kWh = 49) :=
sorry

end tv_weekly_cost_l701_701019


namespace local_tax_deduction_per_hour_l701_701219

def wage_per_hour (w : ℕ) (h : ℕ) : ℕ := w * h

def tax_deduction (percent : ℤ) (wage : ℤ) : ℤ := (percent * wage) / 1000

theorem local_tax_deduction_per_hour :
  let w := 25 * 100 in          -- Alicia earns $25 per hour in cents
  let percent := 22 in          -- The local tax rate is 2.2%, represented as 22 (0.022 * 1000)
  tax_deduction percent (w : ℕ × ℕ) = 55 := by
  sorry

end local_tax_deduction_per_hour_l701_701219


namespace middle_bead_price_l701_701607

theorem middle_bead_price (a : ℕ) 
  (h1 : ∀ (n : ℕ), (0 ≤ n ∧ n < 15) → (a - 3 * 15)) -- Price difference sequence on one side.
  (h2 : ∀ (n : ℕ), (0 ≤ n ∧ n < 15) → (a - 4 * 15)) -- Price difference sequence on the other side.
  (total_value : 15 * (a - 3 * 8) + 15 * (a - 4 * 8) + a = 2012) : 
  a = 92 :=
by
  sorry

end middle_bead_price_l701_701607


namespace monthly_income_of_P_l701_701079

-- Define variables and assumptions
variables (P Q R : ℝ)
axiom avg_P_Q : (P + Q) / 2 = 5050
axiom avg_Q_R : (Q + R) / 2 = 6250
axiom avg_P_R : (P + R) / 2 = 5200

-- Prove that the monthly income of P is 4000
theorem monthly_income_of_P : P = 4000 :=
by
  sorry

end monthly_income_of_P_l701_701079


namespace minimum_value_x_plus_2y_l701_701355

theorem minimum_value_x_plus_2y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 + 6 * x * y - 1 = 0) :
  x + 2 * y = 2 * real.sqrt 2 / 3 :=
by
  sorry

end minimum_value_x_plus_2y_l701_701355


namespace range_of_f_l701_701131

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 5*x + 2) / (x + 1)

def in_range (y : ℝ) : Prop := ∃ x : ℝ, x ≠ -1 ∧ f x = y

theorem range_of_f : set.range f = {y : ℝ | y > -1/4} :=
by sorry

end range_of_f_l701_701131


namespace no_such_integers_exist_l701_701253

theorem no_such_integers_exist :
  ¬ ∃ (a b : ℕ), a ≥ 1 ∧ b ≥ 1 ∧ ∃ k₁ k₂ : ℕ, (a^5 * b + 3 = k₁^3) ∧ (a * b^5 + 3 = k₂^3) :=
by
  sorry

end no_such_integers_exist_l701_701253


namespace intersection_A_B_l701_701716

noncomputable def A : set ℤ := {-2, -1, 0, 1, 2}
noncomputable def B : set ℤ := {x : ℤ | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end intersection_A_B_l701_701716


namespace karl_savings_l701_701016

def original_folder_cost : ℝ := 2.50
def original_pen_cost : ℝ := 1.00
def num_folders : ℕ := 7
def num_pens : ℕ := 10
def folder_discount : ℝ := 0.30
def pen_discount : ℝ := 0.15

theorem karl_savings :
  let total_original_cost := num_folders * original_folder_cost + num_pens * original_pen_cost,
      total_discounted_cost := num_folders * (original_folder_cost * (1 - folder_discount)) + num_pens * (original_pen_cost * (1 - pen_discount)),
      total_savings := total_original_cost - total_discounted_cost
  in total_savings = 6.75 :=
by
  sorry

end karl_savings_l701_701016


namespace product_of_slopes_l701_701904

theorem product_of_slopes (m n p : ℝ) (θ1 θ2 θ3 : ℝ) 
    (h1 : θ1 = 3 * θ2) 
    (h2 : θ3 = θ1 / 2) 
    (h3 : m = 3 * n) 
    (h4 : m = 5 * p) 
    (h5 : m ≠ 0) 
    (h6 : n ≠ 0) 
    (h7 : p ≠ 0) :
    m * n * p = sqrt 3 / 15 :=
begin
  sorry
end

end product_of_slopes_l701_701904


namespace mother_daughter_age_equality_l701_701801

theorem mother_daughter_age_equality :
  ∀ (x : ℕ), (24 * 12 + 3) + x = 12 * ((-5 : ℤ) + x) → x = 32 := 
by
  intros x h
  sorry

end mother_daughter_age_equality_l701_701801


namespace count_three_digit_numbers_l701_701341

theorem count_three_digit_numbers : 
  let a_even := {a : Nat | a = 2 ∨ a = 4 ∨ a = 6 ∨ a = 8},
      b := {b : Nat | ∃ a c, a ∈ a_even ∧ c ∈ {c | c > a ∧ c % 2 = 1} ∧ a < b ∧ b < c}
  in (∃ (nums : Finset (Fin 1000)), 
      ∀ x ∈ nums, let d := digits x; 100 ≤ x ∧ x < 1000 
                    ∧ d.get 0 ∈ a_even ∧ d.get 2 ∈ {c : Nat | c % 2 = 1}
                    ∧ d.get 0 < d.get 1 ∧ d.get 1 < d.get 2 
                    ∧ nums.card = 20) :=
  sorry

end count_three_digit_numbers_l701_701341


namespace impossible_triangle_l701_701812

noncomputable def triangle_inequality_violation : Prop :=
  let AB := 1
  let CD := 2
  let EF := 4
  let BD := Real.sqrt 2
  let DF := Real.sqrt 5
  let FB := Real.sqrt 13

  let AC := Real.sqrt (BD^2 - (CD - AB)^2)
  let CE := Real.sqrt (DF^2 - (EF - CD)^2)
  let EA := Real.sqrt (FB^2 - (EF - AB)^2)

  EA ≠ AC + CE

theorem impossible_triangle : ¬ exists A B C D E F : Point,
  A.distance C = AC ∧ C.distance E = CE ∧ E.distance A = EA ∧
  B.distance D = BD ∧ D.distance F = DF ∧ F.distance B = FB :=
by {
  intro h,
  obtain ⟨A, B, C, D, E, F, hAC, hCE, hEA, hBD, hDF, hFB⟩ := h,
  have h1 := hEA,
  rw [Real.sqrt_eq_rfl, Real.sqrt_eq_rfl, Real.sqrt_eq_rfl] at *,
  sorry
}

end impossible_triangle_l701_701812


namespace order_of_abc_l701_701286

noncomputable def a : ℝ := (1 / 3) * Real.logb 2 (1 / 4)
noncomputable def b : ℝ := 1 - Real.logb 2 3
noncomputable def c : ℝ := Real.cos (5 * Real.pi / 6)

theorem order_of_abc : c < a ∧ a < b := by
  sorry

end order_of_abc_l701_701286


namespace arthur_speed_l701_701620

/-- Suppose Arthur drives to David's house and aims to arrive exactly on time. 
If he drives at 60 km/h, he arrives 5 minutes late. 
If he drives at 90 km/h, he arrives 5 minutes early. 
We want to find the speed n in km/h at which he arrives exactly on time. -/
theorem arthur_speed (n : ℕ) :
  (∀ t, 1 * (t + 5) = (3 / 2) * (t - 5)) → 
  (60 : ℝ) = 1 →
  (90 : ℝ) = (3 / 2) → 
  n = 72 := by
sorry

end arthur_speed_l701_701620


namespace total_snow_volume_l701_701017

-- Define the conditions
def length := 30 -- in feet
def width := 3 -- in feet
def depth1 := 0.6 -- in feet
def depth2 := 0.4 -- in feet

-- Define the volumes of each layer
def volume1 := length * width * depth1
def volume2 := length * width * depth2

-- Prove that the total volume is 90 cubic feet
theorem total_snow_volume : (volume1 + volume2) = 90 := by
  sorry

end total_snow_volume_l701_701017


namespace slope_of_parallel_line_l701_701984

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l701_701984


namespace range_of_k_l701_701358

theorem range_of_k {x y k : ℝ} :
  (∀ x y, 2 * x - y ≤ 1 ∧ x + y ≥ 2 ∧ y - x ≤ 2) →
  (z = k * x + 2 * y) →
  (∀ (x y : ℝ), z = k * x + 2 * y → (x = 1) ∧ (y = 1)) →
  -4 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l701_701358


namespace systematic_sampling_IDs_l701_701751

theorem systematic_sampling_IDs (class_size sample_size : ℕ)  
  (H_class_size : class_size = 56) 
  (H_sample_size : sample_size = 4) 
  (IDs_in_sample : (4 ∈ {4, 32, 46}) ∧ (32 ∈ {4, 32, 46}) ∧ (46 ∈ {4, 32, 46})) : 
  ∃ x, x = 18 ∧ x ∉ {4, 32, 46} ∧ (x ∈ {4, 32, 46, 18}) := 
sorry

end systematic_sampling_IDs_l701_701751


namespace slope_of_parallel_line_l701_701927

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l701_701927


namespace max_periodic_sum_19_l701_701827

theorem max_periodic_sum_19 (p : ℕ) (hp : Nat.Prime p) (d k : ℕ) (a : ℕ → ℕ) :
  (d = 3 * p) →
  let N_p := 10^(2*p) + 10^p + 1 in
  (N_p % 9 = 3) →
  ∃ q : ℕ, Nat.Prime q ∧ q ∣ 10^(3*p) - 1 ∧ k = p ∧
  (∀ i : ℕ, a_i + a_{i+k} + a_{i+2k} ≤ 19) :=
sorry

end max_periodic_sum_19_l701_701827


namespace polygon_side_not_intersected_l701_701636

-- Definitions used in the conditions
variables {n : ℕ} (P : ℝ × ℝ) (polygon : list (ℝ × ℝ))

-- Conditions: P is inside a polygon with 2n vertices
def point_inside_polygon (P : ℝ × ℝ) (polygon : list (ℝ × ℝ)) : Prop :=
  sorry -- A placeholder to specify the geometric condition

def even_sided_polygon (polygon : list (ℝ × ℝ)) : Prop :=
  polygon.length = 2 * n

-- The theorem statement
theorem polygon_side_not_intersected {n : ℕ} (P : ℝ × ℝ) (polygon : list (ℝ × ℝ))
    (P_in_poly : point_inside_polygon P polygon) (event_polygon : even_sided_polygon polygon) :
    ∃ (side : (ℝ × ℝ) × (ℝ × ℝ)), ¬ ∃ (vertex : ℝ × ℝ), vertex ∈ polygon ∧ _root_.line_through P vertex ∩ _root_.segment side ≠ ∅ :=
  sorry

end polygon_side_not_intersected_l701_701636


namespace work_days_together_l701_701192

theorem work_days_together (A B : Type) (R_A R_B : ℝ) 
  (h1 : R_A = 1/2 * R_B) (h2 : R_B = 1 / 27) : 
  (1 / (R_A + R_B)) = 18 :=
by
  sorry

end work_days_together_l701_701192


namespace plane_equation_l701_701258

theorem plane_equation (p q r : ℝ × ℝ × ℝ)
  (h₁ : p = (2, -1, 3))
  (h₂ : q = (0, -1, 5))
  (h₃ : r = (-1, -3, 4)) :
  ∃ A B C D : ℤ, A = 1 ∧ B = 2 ∧ C = -1 ∧ D = 3 ∧
               A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧
               ∀ x y z : ℝ, A * x + B * y + C * z + D = 0 ↔
                             (x, y, z) = p ∨ (x, y, z) = q ∨ (x, y, z) = r :=
by
  sorry

end plane_equation_l701_701258


namespace right_heavier_by_250_l701_701068

variable square_weight : ℕ := 300
variable triangular_weight : ℕ := 150
variable round_weight : ℕ := 200

variable left_num_square : ℕ := 1
variable left_num_triangular : ℕ := 2
variable left_num_round : ℕ := 3

variable right_num_square : ℕ := 3
variable right_num_triangular : ℕ := 1
variable right_num_round : ℕ := 2

def left_pan_weight : ℕ := 
  left_num_square * square_weight + 
  left_num_triangular * triangular_weight + 
  left_num_round * round_weight

def right_pan_weight : ℕ := 
  right_num_square * square_weight + 
  right_num_triangular * triangular_weight + 
  right_num_round * round_weight

theorem right_heavier_by_250 : right_pan_weight = left_pan_weight + 250 := 
  by
    sorry

end right_heavier_by_250_l701_701068


namespace min_side_length_square_in_rectangle_l701_701303

theorem min_side_length_square_in_rectangle (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) : 
  let side := if a < (Real.sqrt 2 + 1) * b then a else (Real.sqrt 2 / 2) * (a + b) in
  (∃ S : ℝ, S = side) :=
by
-- Equivalent mathematical problem statement translated into Lean.
sorry

end min_side_length_square_in_rectangle_l701_701303


namespace find_two_digit_number_l701_701251

theorem find_two_digit_number : ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 10 * x + y = x^3 + y^2 ∧ 10 * x + y = 24 := by
  sorry

end find_two_digit_number_l701_701251


namespace defective_units_shipped_percent_l701_701573

def percent_defective_units_shipped (p_defective : ℝ) (p_shipped_defective : ℝ) : ℝ :=
  (p_defective / 100) * (p_shipped_defective / 100) * 100

theorem defective_units_shipped_percent (h1 : 4 = (p_defective : ℝ)) (h2 : 4 = (p_shipped_defective : ℝ)) :
  percent_defective_units_shipped 4 4 = 0.16 :=
by
  -- Admitted proof
  sorry

end defective_units_shipped_percent_l701_701573


namespace beatty_one_exact_cover_l701_701064

theorem beatty_one_exact_cover (α β : ℝ) (h_irrational_α : irrational α) (h_irrational_β : irrational β)
  (h_eq : 1 / α + 1 / β = 1) :
  ∀ k : ℕ, (∃ n : ℕ, k = ⌊n * α⌋) ∨ (∃ m : ℕ, k = ⌊m * β⌋) ∧ ¬ (∃ n m, k = ⌊n * α⌋ ∧ k = ⌊m * β⌋) :=
sorry

end beatty_one_exact_cover_l701_701064


namespace smallest_n_for_inequality_l701_701313

theorem smallest_n_for_inequality :
  ∃ (n : ℕ), n = 4003 ∧ (∀ m : ℤ, (0 < m ∧ m < 2001) →
  ∃ k : ℤ, (m / 2001 : ℚ) < (k / n : ℚ) ∧ (k / n : ℚ) < ((m + 1) / 2002 : ℚ)) :=
sorry

end smallest_n_for_inequality_l701_701313


namespace two_rides_combinations_l701_701409

-- Define the number of friends
def num_friends : ℕ := 7

-- Define the size of the group for one ride
def ride_group_size : ℕ := 4

-- Define the number of combinations of choosing 'ride_group_size' out of 'num_friends'
def combinations_first_ride : ℕ := Nat.choose num_friends ride_group_size

-- Define the number of friends left for the second ride
def remaining_friends : ℕ := num_friends - ride_group_size

-- Define the number of combinations of choosing 'ride_group_size' out of 'remaining_friends' friends
def combinations_second_ride : ℕ := Nat.choose remaining_friends ride_group_size

-- Define the total number of possible combinations for two rides
def total_combinations : ℕ := combinations_first_ride * combinations_second_ride

-- The final theorem stating the total number of combinations is equal to 525
theorem two_rides_combinations : total_combinations = 525 := by
  -- Placeholder for proof
  sorry

end two_rides_combinations_l701_701409


namespace distance_to_y_axis_l701_701741

theorem distance_to_y_axis 
  (M : ℝ × ℝ) 
  (h₁ : M.1 = (M.2 ^ 2) / 4) 
  (focus_dist : real.sqrt ((M.1 - 1)^2 + M.2^2) = 10) : 
  abs M.1 = 9 :=
by {
  sorry
}

end distance_to_y_axis_l701_701741


namespace number_one_appears_infinitely_many_times_every_natural_number_appears_infinitely_many_times_l701_701384

-- Define the sequence according to the problem statement
def seq : ℕ → ℕ
| 1     := 1
| (n+1) := if (n % 4 == 1)
           then seq n + 1
           else if (n % 4 == 3)
                then seq n - 1
                else seq n

theorem number_one_appears_infinitely_many_times :
  ∀ (n : ℕ), ∃ m ≥ n, seq m = 1 :=
sorry

theorem every_natural_number_appears_infinitely_many_times :
  ∀ (k : ℕ), ∀ (n : ℕ), ∃ m ≥ n, seq m = k :=
sorry

end number_one_appears_infinitely_many_times_every_natural_number_appears_infinitely_many_times_l701_701384


namespace orthocenter_of_triangle_l701_701165

variables {A B C D E F H : Point}
variables (triangle : Triangle A B C)
variables (H_AB_AC : A ≠ B ∧ A ≠ C)
variables (incircle_tangency_D : incircle(triangle).ContactPoint B C = D)
variables (incircle_tangency_E : incircle(triangle).ContactPoint C A = E)
variables (incircle_tangency_F : incircle(triangle).ContactPoint A B = F)
variables (H_on_EF : H ∈ Seg E F)
variables (H_perp_EF : Perpendicular (Line D H) (Line E F))
variables (AH_perp_BC : Perpendicular (Line A H) (Line B C))

theorem orthocenter_of_triangle : is_orthocenter H triangle :=
sorry

end orthocenter_of_triangle_l701_701165


namespace min_PD_plus_PE_l701_701622

theorem min_PD_plus_PE (A B C D E P : Point)
(angle_ABC : ∠ABC = 60)
(AE_len : AE = 8) 
(BE_len : BE = 4)
(is_altitude_BD : BD⊥AC) 
(is_altitude_CE : CE⊥AB)
(is_on_side_PBC : P ∈ lineBC) :
PD + PE = (20 * Real.sqrt 7) / 7 := 
sorry

end min_PD_plus_PE_l701_701622


namespace integer_part_of_product_l701_701094

theorem integer_part_of_product :
  (⌊1.1 * 1.2 * 1.3 * 1.4 * 1.5 * 1.6⌋ : ℤ) = 1 := sorry

end integer_part_of_product_l701_701094


namespace max_distinct_triangles_l701_701604

def is_valid_triangle (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ 6 ∧ b ≤ 6 ∧ c ≤ 6 ∧ a ≥ b ∧ b ≥ c ∧ b + c > a

def are_similar (a1 b1 c1 a2 b2 c2 : ℕ) : Prop :=
  a1 * b2 = a2 * b1 ∧ a1 * c2 = a2 * c1 ∧ b1 * c2 = b2 * c1

def are_congruent (a1 b1 c1 a2 b2 c2 : ℕ) : Prop :=
  (a1 = a2 ∧ b1 = b2 ∧ c1 = c2) ∨ (a1 = a2 ∧ b1 = c2 ∧ c1 = b2) ∨ (a1 = b2 ∧ b1 = a2 ∧ c1 = c2) ∨ (a1 = b2 ∧ b1 = c2 ∧ c1 = a2) ∨ (a1 = c2 ∧ b1 = a2 ∧ c1 = b2) ∨ (a1 = c2 ∧ b1 = b2 ∧ c1 = a2)

def are_similar_or_congruent (t1 t2 : ℕ × ℕ × ℕ) : Prop :=
  let (a1, b1, c1) := t1
  let (a2, b2, c2) := t2
  are_similar a1 b1 c1 a2 b2 c2 ∨ are_congruent a1 b1 c1 a2 b2 c2

def S : set (ℕ × ℕ × ℕ) :=
  {t | let (a, b, c) := t in is_valid_triangle a b c ∧ ∀ t' ∈ S, ∀ t' ≠ t, ¬ are_similar_or_congruent t t'}

theorem max_distinct_triangles : ∃ (n : ℕ), S.to_finset.card = 17 := sorry

end max_distinct_triangles_l701_701604


namespace length_of_KL_l701_701483

theorem length_of_KL (A B C D E P Q R S K L : Type) 
  (hA: A = (1 : ℝ)) 
  (hB: B = (1 : ℝ)) 
  (hC: C = (1 : ℝ)) 
  (hD: D = (1 : ℝ)) 
  (hE: E = (1 : ℝ)) 
  (hP: P = (1 / 2 : ℝ)) 
  (hQ: Q = (1 / 2 : ℝ)) 
  (hR: R = (1 / 2 : ℝ)) 
  (hS: S = (1 / 2 : ℝ)) 
  (hK: K = (1 / 2 : ℝ)) 
  (hL: L = (1 / 2 : ℝ)) : 
  KL = (1 / 4 : ℝ) := 
sorry

end length_of_KL_l701_701483


namespace segment_KL_length_l701_701501

noncomputable def pentagon_side_length : ℝ := 1

variable {A B C D E P Q R S K L : Type}
          [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
          [LinearOrderedField D] [LinearOrderedField E]
          [LinearOrderedField P] [LinearOrderedField Q]
          [LinearOrderedField R] [LinearOrderedField S]
          [LinearOrderedField K] [LinearOrderedField L]

def mid_point (X Y : ℝ) : ℝ := (X + Y) / 2

axiom AB_eq_1 : dist A B = pentagon_side_length
axiom BC_eq_1 : dist B C = pentagon_side_length
axiom CD_eq_1 : dist C D = pentagon_side_length
axiom DE_eq_1 : dist D E = pentagon_side_length
axiom EA_eq_1 : dist E A = pentagon_side_length

axiom P_mid_AB : P = mid_point A B
axiom Q_mid_BC : Q = mid_point B C
axiom R_mid_CD : R = mid_point C D
axiom S_mid_DE : S = mid_point D E

axiom K_mid_PR : K = mid_point P R
axiom L_mid_QS : L = mid_point Q S 

theorem segment_KL_length : dist K L = 1 / 4 := sorry

end segment_KL_length_l701_701501


namespace jack_received_more_emails_l701_701402

-- Definitions representing the conditions
def morning_emails : ℕ := 6
def afternoon_emails : ℕ := 8

-- The theorem statement
theorem jack_received_more_emails : afternoon_emails - morning_emails = 2 := 
by 
  sorry

end jack_received_more_emails_l701_701402


namespace radius_of_tangent_circle_l701_701261

theorem radius_of_tangent_circle (R1 R2 : ℝ) (h1 : 0 < R1) (h2 : 0 < R2) : 
  ∃ x : ℝ, x = R1 * R2 / (ℝ.sqrt R1 + ℝ.sqrt R2) ^ 2 ∨ x = R1 * R2 / (ℝ.sqrt R1 - ℝ.sqrt R2) ^ 2 :=
by
  sorry

end radius_of_tangent_circle_l701_701261


namespace curve_transformation_l701_701004

def matrix_transform (a : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (0 * x + 1 * y, a * x + 0 * y)

def curve_eq (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 = 1

def transformed_curve_eq (x y : ℝ) : Prop :=
  x ^ 2 + (y ^ 2) / 4 = 1

theorem curve_transformation (a : ℝ) 
  (h₁ : matrix_transform a 2 (-2) = (-2, 4))
  (h₂ : ∀ x y, curve_eq x y → transformed_curve_eq (matrix_transform a x y).fst (matrix_transform a x y).snd) :
  a = 2 ∧ ∀ x y, curve_eq x y → transformed_curve_eq (0 * x + 1 * y) (2 * x + 0 * y) :=
by
  sorry

end curve_transformation_l701_701004


namespace length_of_KL_l701_701493

open Finset
open Classical

-- Define the pentagon and relevant points
def Pentagon :=
  {a b c d e : Point}

-- Conditions: all sides of the pentagon are equal to 1
axiom side_lengths (p : Pentagon) : length (p.a, p.b) = 1 ∧ length (p.b, p.c) = 1 ∧ length (p.c, p.d) = 1 ∧ length (p.d, p.e) = 1 ∧ length (p.e, p.a) = 1

-- Define midpoints
def midpoint (p1 p2 : Point) : Point
axiom midpoint_is_between (p1 p2 : Point) : midpoint p1 p2 = (p1 + p2) / 2

-- Define points P, Q, R, S, K, L based on the problem statement
def P (p : Pentagon) := midpoint p.a p.b
def Q (p : Pentagon) := midpoint p.b p.c
def R (p : Pentagon) := midpoint p.c p.d
def S (p : Pentagon) := midpoint p.d p.e
def K (p : Pentagon) := midpoint (P p) (R p)
def L (p : Pentagon) := midpoint (Q p) (S p)

-- The theorem stating the length of segment KL
theorem length_of_KL (p : Pentagon) :
  length (K p) (L p) = 1 / 4 := sorry

end length_of_KL_l701_701493


namespace fold_creates_crease_l701_701555

theorem fold_creates_crease (P : Type) [plane_space P] : 
  ∀ (p1 p2 : P), folded p1 p2 → ∃ l : Line, intersection p1 p2 = l :=
by
  sorry

end fold_creates_crease_l701_701555


namespace factors_multiple_of_10_120_l701_701340

theorem factors_multiple_of_10_120 (p : ℕ) (h : p = 120) :
  (∃ k, k = 10 ∨ k = 20 ∨ k = 30 ∨ k = 40 ∨ k = 60 ∨ k = 120) ↔
  (k ∈ (range (p + 1)).filter (λ i, p % i = 0 ∧ i % 10 = 0)) ∉ {10, 20, 30, 40, 60, 120}.card = 6 :=
by
  sorry

end factors_multiple_of_10_120_l701_701340


namespace midpoint_MN_is_incenter_l701_701586

-- Given conditions
variables {A B C M N : Point}

-- Assume triangle ABC with tangent circle touching at points M and N
variables 
  (h1 : TangentCircle ABC M N)
  (h2 : InternallyTouchingCircumcircle ABC M N)

-- Prove that the midpoint of segment MN is the center of the incircle of triangle ABC.
theorem midpoint_MN_is_incenter (O : Point) :
  (Midpoint M N = Incenter ABC) := 
sorry

end midpoint_MN_is_incenter_l701_701586


namespace sum_of_first_11_terms_l701_701883

theorem sum_of_first_11_terms (a1 d : ℝ) (h : 2 * a1 + 10 * d = 8) : 
  (11 / 2) * (2 * a1 + 10 * d) = 44 := 
by sorry

end sum_of_first_11_terms_l701_701883


namespace current_time_is_10_15_l701_701740

theorem current_time_is_10_15 : 
  ∃ t : ℚ, (60 * 10 ≤ t) ∧ (t < 60 * 11) ∧ 
  let x := (t / 60 * 10 - 10) in 
  (360 - |6 * (x + 6) - (30 + 0.5 * (x - 3))|) = 180 
  ∧ (t = 10 * 60 + 15) := 
by
  existsi 10 * 60 + 15
  split
  { linarith }
  split
  { linarith }
  split
  { simp only [mul_add, mul_sub, div_eq_mul_one_div, rational.affine_map]
    field_simp
    ring }
  sorry

end current_time_is_10_15_l701_701740


namespace find_triples_l701_701875

noncomputable def root_triples (a b c : ℂ) : Prop :=
  ∃ (d : ℂ), 
    d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
    (∀ z, (z = a ∨ z = b ∨ z = c ∨ z = d) → 
        z^4 - (a^3 : ℂ) - (b : ℕ) * z + c = 0)

theorem find_triples (a b c : ℂ) :
  (root_triples a 0 0) ∨ 
  (root_triples (-(1:ℂ) + Complex.I * Real.sqrt 3 / 2) 1 
   (-(1:ℂ) + Complex.I * Real.sqrt 3 / 2)) ∨
  (root_triples (-(1:ℂ) - Complex.I * Real.sqrt 3 / 2) 1
   (-(1:ℂ) - Complex.I * Real.sqrt 3 / 2)) ∨
  (root_triples (-(1:ℂ) - Complex.I * Real.sqrt 3 / 2) -1
   (1 - Complex.I * Real.sqrt 3 / 2)) ∨
  (root_triples (-(1:ℂ) + Complex.I * Real.sqrt 3 / 2) -1
   (1 + Complex.I * Real.sqrt 3 / 2)) :=
sorry

end find_triples_l701_701875


namespace smallest_b_multiple_of_6_and_15_is_30_l701_701268

theorem smallest_b_multiple_of_6_and_15_is_30 : ∃ b : ℕ, b > 0 ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ b = 30 :=
by
  use 30
  split
  . trivial
  . split
    . sorry
    . sorry

end smallest_b_multiple_of_6_and_15_is_30_l701_701268


namespace sum_of_coordinates_A_l701_701029

-- Define the points A, B, and C and the given conditions
variables (A B C : ℝ × ℝ)
variables (h_ratio1 : dist A C / dist A B = 1 / 3)
variables (h_ratio2 : dist B C / dist A B = 1 / 3)
variables (h_B : B = (2, 8))
variables (h_C : C = (0, 2))

-- Lean 4 statement to prove the sum of the coordinates of A is -14
theorem sum_of_coordinates_A : (A.1 + A.2) = -14 :=
sorry

end sum_of_coordinates_A_l701_701029


namespace f_at_2_is_4_l701_701038

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then log2 ((2 - x) ^ 2)
  else 0 -- unspecified for x > 0, can be extended

theorem f_at_2_is_4 (f_even : is_even_function f) (f_def : ∀ x ≤ 0, f x = log2 ((2 - x) ^ 2)) : f 2 = 4 :=
by
  sorry

end f_at_2_is_4_l701_701038


namespace circle_passing_through_points_l701_701849

theorem circle_passing_through_points :
  ∃ D E F : ℝ, ∀ (x y : ℝ),
    ((x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) →
    (x^2 + y^2 + D*x + E*y + F = 0) ↔
    (x^2 + y^2 - 4*x - 6*y = 0) :=
begin
  sorry,
end

end circle_passing_through_points_l701_701849


namespace dot_product_of_vectors_l701_701332

variables {V : Type*} [InnerProductSpace ℝ V]

noncomputable def given_vectors (a b : V) : Prop :=
  norm a = 5 ∧ norm b = 3 ∧ norm (a - b) = 7

theorem dot_product_of_vectors {a b : V} (h : given_vectors a b) :
  inner a b = -15 / 2 :=
sorry

end dot_product_of_vectors_l701_701332


namespace segment_KL_length_l701_701499

noncomputable def pentagon_side_length : ℝ := 1

variable {A B C D E P Q R S K L : Type}
          [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
          [LinearOrderedField D] [LinearOrderedField E]
          [LinearOrderedField P] [LinearOrderedField Q]
          [LinearOrderedField R] [LinearOrderedField S]
          [LinearOrderedField K] [LinearOrderedField L]

def mid_point (X Y : ℝ) : ℝ := (X + Y) / 2

axiom AB_eq_1 : dist A B = pentagon_side_length
axiom BC_eq_1 : dist B C = pentagon_side_length
axiom CD_eq_1 : dist C D = pentagon_side_length
axiom DE_eq_1 : dist D E = pentagon_side_length
axiom EA_eq_1 : dist E A = pentagon_side_length

axiom P_mid_AB : P = mid_point A B
axiom Q_mid_BC : Q = mid_point B C
axiom R_mid_CD : R = mid_point C D
axiom S_mid_DE : S = mid_point D E

axiom K_mid_PR : K = mid_point P R
axiom L_mid_QS : L = mid_point Q S 

theorem segment_KL_length : dist K L = 1 / 4 := sorry

end segment_KL_length_l701_701499


namespace slope_of_parallel_line_l701_701926

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l701_701926


namespace part_I_part_II_l701_701007

-- Define the polar curve C in Cartesian coordinates
def curveC (x y : ℝ) : Prop :=
  (x - 1 / 2)^2 + (y - (Real.sqrt 3 / 2))^2 = 1

-- Define line l in parametric form
def lineL (t : ℝ) (x y : ℝ) : Prop :=
  x = 1 / 2 - (Real.sqrt 3 / 2) * t ∧ y = Real.sqrt 3 / 2 + (1 / 2) * t

-- Points A and B in Cartesian coordinates
def pointA := (3 / 2, Real.sqrt 3 / 2)
def pointB := (0, Real.sqrt 3)

-- Square of Euclidean distance between two points
def distanceSquare (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 - x2)^2 + (y1 - y2)^2

-- Prove Part I: Cartesian equation of curve C and parametric equations of line l
theorem part_I : 
  ∀ x y t : ℝ, 
  curveC x y ↔ ((curveC x y) ∧ (lineL t x y)) := 
sorry

-- Prove Part II: Range of |MA|^2 + |MB|^2
theorem part_II : 
  ∀ θ : ℝ, 
  2 ≤ (distanceSquare (1 / 2 + Real.cos θ) (Real.sqrt 3 / 2 + Real.sin θ) (3 / 2) (Real.sqrt 3 / 2) 
       + distanceSquare (1 / 2 + Real.cos θ) (Real.sqrt 3 / 2 + Real.sin θ) 0 (Real.sqrt 3)) 
  ∧ (distanceSquare (1 / 2 + Real.cos θ) (Real.sqrt 3 / 2 + Real.sin θ) (3 / 2) (Real.sqrt 3 / 2) 
  + distanceSquare (1 / 2 + Real.cos θ) (Real.sqrt 3 / 2 + Real.sin θ) 0 (Real.sqrt 3)) ≤ 6 :=
sorry

end part_I_part_II_l701_701007


namespace problem_statement_l701_701421

noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 30
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by 
  sorry

end problem_statement_l701_701421


namespace four_digit_numbers_divisible_by_4_and_5_l701_701339

open Finset

theorem four_digit_numbers_divisible_by_4_and_5 (digits : Finset ℕ) :
  digits = {0, 2, 4, 6, 8, 1, 3, 5} → 
  card {n : ℕ | (1000 ≤ n ∧ n ≤ 9999) ∧ (∀ d ∈ (n.digits 10).to_finset, d ∈ digits) ∧ (n % 4 = 0) ∧ (n % 5 = 0)} = 96 := 
by
  sorry

end four_digit_numbers_divisible_by_4_and_5_l701_701339


namespace triangle_shaded_to_non_shaded_ratio_l701_701532

noncomputable def equilateral_triangle_side_length := 6

def midpoint (A B : ℝ×ℝ) : ℝ×ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def point_D (A B : ℝ×ℝ) : ℝ×ℝ := midpoint A B
def point_E (B C : ℝ×ℝ) : ℝ×ℝ := midpoint B C
def point_F (C A : ℝ×ℝ) : ℝ×ℝ := midpoint C A
def point_G (D F : ℝ×ℝ) : ℝ×ℝ := midpoint D F
def point_H (F E : ℝ×ℝ) : ℝ×ℝ := midpoint F E

noncomputable def shaded_area_ratio (ABC DEF : set (ℝ × ℝ)) (shaded nonshaded : ℝ) : Prop :=
shaded / nonshaded = 5 / 11

theorem triangle_shaded_to_non_shaded_ratio :
  ∀ (A B C : ℝ × ℝ),
    dist A B = equilateral_triangle_side_length →
    dist B C = equilateral_triangle_side_length →
    dist C A = equilateral_triangle_side_length →
    let D := point_D A B,
    let E := point_E B C,
    let F := point_F C A,
    let G := point_G D F,
    let H := point_H F E,
    let total_area := (equilateral_triangle_side_length ^ 2) * (cmath.sqrt 3 / 4),
    let def_area := (equilateral_triangle_side_length / 2) ^ 2 * (cmath.sqrt 3 / 4),
    let smaller_triangle_area := (equilateral_triangle_side_length / 4) ^ 2 * (cmath.sqrt 3 / 4),
    let shaded_area := def_area + 3 * smaller_triangle_area,
    let non_shaded_area := total_area - shaded_area in
    shaded_area_ratio {A, B, C} {D, E, F} shaded_area non_shaded_area :=
sorry

end triangle_shaded_to_non_shaded_ratio_l701_701532


namespace num_integers_between_100_and_200_with_decreasing_digits_l701_701723

theorem num_integers_between_100_and_200_with_decreasing_digits: 
  let count_integers := (9 + 8 + 7 + 6 + 5 + 4 + 3 + 2)
  in count_integers = 44 :=
by
  let count_integers := (9 + 8 + 7 + 6 + 5 + 4 + 3 + 2)
  exact rfl

end num_integers_between_100_and_200_with_decreasing_digits_l701_701723


namespace parallel_line_slope_l701_701996

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l701_701996


namespace distance_between_points_l701_701240

theorem distance_between_points (a b c d m k : ℝ) 
  (h1 : b = 2 * m * a + k) (h2 : d = -m * c + k) : 
  (Real.sqrt ((c - a)^2 + (d - b)^2)) = Real.sqrt ((1 + m^2) * (c - a)^2) := 
by {
  sorry
}

end distance_between_points_l701_701240


namespace slope_of_parallel_line_l701_701924

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l701_701924


namespace positive_abc_l701_701285

theorem positive_abc (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : a > 0 ∧ b > 0 ∧ c > 0 := 
by
  sorry

end positive_abc_l701_701285


namespace f_three_l701_701428

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_additive : ∀ (x y : ℝ), f(x + y) = f(x) + f(y)
axiom f_one : f(1) = -2

theorem f_three : f(3) = -6 := by
  sorry

end f_three_l701_701428


namespace number_of_5_dollar_coins_l701_701525

-- Define the context and the proof problem
theorem number_of_5_dollar_coins (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 5 * y = 125) : y = 15 :=
by sorry

end number_of_5_dollar_coins_l701_701525


namespace inequality_x_n_l701_701669

theorem inequality_x_n (x : ℝ) (n : ℕ) (hx : |x| < 1) (hn : n ≥ 2) : (1 - x)^n + (1 + x)^n < 2^n := 
sorry

end inequality_x_n_l701_701669


namespace slope_of_parallel_line_l701_701970

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l701_701970


namespace f_even_analytic_expression_of_f_find_a_l701_701698

-- Define the function f
def f (x : ℝ) : ℝ :=
  if 0 ≤ x then 2 * x + 3 else -2 * x + 3

-- f is an even function
theorem f_even (x : ℝ) : f (-x) = f x := by
  unfold f
  split_ifs
  case _ h₀ =>
    rw [neg_nonpos] at h₀
    simp [h₀]
  case _ h₁ =>
    have : (0:ℝ) ≤ -x := by
      exact le_of_not_le h₁
    simp [this]

-- Prove the analytic expression of f(x)
theorem analytic_expression_of_f :
  ∀ x : ℝ, f x = (if 0 ≤ x then 2 * x + 3 else -2 * x + 3) :=
by intro x; unfold f; split_ifs; refl

-- Given f(a) = 7, find the possible values of a
theorem find_a (a : ℝ) (h : f a = 7) : a = 2 ∨ a = -2 := by
  unfold f at h
  split_ifs at h with h₀ h₁
  case _ =>
    -- f(a) = 2a + 3 = 7
    have : 2 * a + 3 = 7 := h
    linarith
  case _ =>
    -- f(a) = -2a + 3 = 7
    have : -2 * a + 3 = 7 := h
    linarith

end f_even_analytic_expression_of_f_find_a_l701_701698


namespace scientific_notation_of_20_nanoseconds_l701_701393

def nanosecond_to_seconds (nanoseconds : ℕ) : ℝ :=
  nanoseconds * (1 * 10 ^ (-9))

theorem scientific_notation_of_20_nanoseconds :
  nanosecond_to_seconds 20 = 2 * 10 ^ (-8) := by
  sorry

end scientific_notation_of_20_nanoseconds_l701_701393


namespace prove_ratio_l701_701417

noncomputable def box_dimensions : ℝ × ℝ × ℝ := (2, 3, 5)
noncomputable def d := (2 * 3 * 5 : ℝ)
noncomputable def a := ((4 * Real.pi) / 3 : ℝ)
noncomputable def b := (10 * Real.pi : ℝ)
noncomputable def c := (62 : ℝ)

theorem prove_ratio :
  (b * c) / (a * d) = (15.5 : ℝ) :=
by
  unfold a b c d
  sorry

end prove_ratio_l701_701417


namespace min_teacher_time_l701_701530

-- Define the students and their respective times
def student_A_total_time := 16
def student_B_total_time := 13
def student_C_total_time := 19
def teacher_explanation_time := 3
def student_A_correction_time := 10
def student_B_correction_time := 10
def student_C_correction_time := 13

-- Define the proof statement
theorem min_teacher_time :
  3 * student_B_total_time + 2 * student_A_total_time + student_C_total_time = 90 :=
by
  unfold student_A_total_time student_B_total_time student_C_total_time
  rw [mul_comm 3 13, mul_comm 2 16, mul_comm 1 19]
  norm_num
  sorry

end min_teacher_time_l701_701530


namespace length_KL_l701_701506

variables {Point : Type} [metric_space Point]
variables {A B C D E P Q R S K L M : Point}
variables (AB BC CD DE AE PR QS : ℝ)
variables (midpoint : Point → Point → Point)
variables (length : Point → Point → ℝ)

-- Conditions given
axiom pentagon_eq_sides (AB BC CD DE AE : ℝ) : AB = 1 ∧ BC = 1 ∧ CD = 1 ∧ DE = 1
axiom is_midpoint (X Y Z : Point) :
  midpoint X Y = Z ↔ length X Z = length Y Z / 2
axiom midpoints 
  (P Q R S K L : Point)
  (AB BC CD DE PR QS : ℝ) 
  (MP AB BC CD DE PR QS length midpoint) :
  midpoint A B = P ∧ midpoint B C = Q ∧ midpoint C D = R ∧ midpoint D E = S ∧ 
  midpoint P R = K ∧ midpoint Q S = L

-- Geometry proof problem
theorem length_KL (A B C D E P Q R S K L : Point) :
  pentagon_eq_sides AB BC CD DE AE →
  midpoints P Q R S K L AB BC CD DE PR QS length midpoint →
  length K L = 1 / 4 * length A E := 
sorry

end length_KL_l701_701506


namespace minimum_positive_period_of_f_l701_701872

noncomputable def f (x : ℝ) : ℝ := abs (sin x + cos (x - π / 4))

theorem minimum_positive_period_of_f : ∃ p > 0, (∀ x : ℝ, f (x + p) = f x) ∧ p = π :=
sorry

end minimum_positive_period_of_f_l701_701872


namespace parallel_line_slope_l701_701998

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l701_701998


namespace part_a_part_b_part_c_part_d_l701_701779

-- Part (a)
theorem part_a (glu: ℕ) (blu: ℕ) (zips: ℕ)  
(h_cond: glu = 4 ∧ blu = 3 ∧ zips = 60)
(h_glu: glu = 28) :
  blu = 21 :=
sorry

-- Part (b)
theorem part_b (glu: ℕ) (blu: ℕ) (zips: ℕ) (N: ℕ)
(h_cond: glu = 4 ∧ blu = 3 ∧ zips = 60)
(h_glu_blu: glu = 48 ∨ blu = 48):
  N = 36 ∨ N = 64 :=
sorry

-- Part (c)
theorem part_c (glu: ℕ) (blu: ℕ) (zips: ℕ)
(h_cond: glu = 4 ∧ blu = 3 ∧ zips = 60)
(h_avail: glu = 64 ∧ blu = 42) :
  zips = 840 :=
sorry

-- Part (d)
theorem part_d (zips: ℕ) (price_per_zip: ℕ) (profit_per_zip: ℕ) (cost_gluze: ℕ) (cost_blurpos: ℕ)
(h_cond: zips = 60 ∧ price_per_zip = 0.5 ∧ profit_per_zip = 0.3 ∧ cost_gluze = 1.8)
(h_g: cost_blurpos = 1.6) :
  cost_blurpos = 1.6 :=
sorry

end part_a_part_b_part_c_part_d_l701_701779


namespace domain_h_l701_701257

noncomputable def h (x : ℝ) : ℝ := (3 * x - 1) / Real.sqrt (x - 5)

theorem domain_h (x : ℝ) : h x = (3 * x - 1) / Real.sqrt (x - 5) → (x > 5) :=
by
  intro hx
  have hx_nonneg : x - 5 >= 0 := sorry
  have sqrt_nonzero : Real.sqrt (x - 5) ≠ 0 := sorry
  sorry

end domain_h_l701_701257


namespace length_of_KL_l701_701496

open Finset
open Classical

-- Define the pentagon and relevant points
def Pentagon :=
  {a b c d e : Point}

-- Conditions: all sides of the pentagon are equal to 1
axiom side_lengths (p : Pentagon) : length (p.a, p.b) = 1 ∧ length (p.b, p.c) = 1 ∧ length (p.c, p.d) = 1 ∧ length (p.d, p.e) = 1 ∧ length (p.e, p.a) = 1

-- Define midpoints
def midpoint (p1 p2 : Point) : Point
axiom midpoint_is_between (p1 p2 : Point) : midpoint p1 p2 = (p1 + p2) / 2

-- Define points P, Q, R, S, K, L based on the problem statement
def P (p : Pentagon) := midpoint p.a p.b
def Q (p : Pentagon) := midpoint p.b p.c
def R (p : Pentagon) := midpoint p.c p.d
def S (p : Pentagon) := midpoint p.d p.e
def K (p : Pentagon) := midpoint (P p) (R p)
def L (p : Pentagon) := midpoint (Q p) (S p)

-- The theorem stating the length of segment KL
theorem length_of_KL (p : Pentagon) :
  length (K p) (L p) = 1 / 4 := sorry

end length_of_KL_l701_701496


namespace total_selling_price_l701_701210

theorem total_selling_price (cost_price_per_metre profit_per_metre : ℝ)
  (total_metres_sold : ℕ) :
  cost_price_per_metre = 58.02564102564102 → 
  profit_per_metre = 29 → 
  total_metres_sold = 78 →
  (cost_price_per_metre + profit_per_metre) * total_metres_sold = 6788 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  -- backend calculation, checking computation level;
  sorry

end total_selling_price_l701_701210


namespace length_of_KL_l701_701487

theorem length_of_KL (A B C D E P Q R S K L : Type) 
  (hA: A = (1 : ℝ)) 
  (hB: B = (1 : ℝ)) 
  (hC: C = (1 : ℝ)) 
  (hD: D = (1 : ℝ)) 
  (hE: E = (1 : ℝ)) 
  (hP: P = (1 / 2 : ℝ)) 
  (hQ: Q = (1 / 2 : ℝ)) 
  (hR: R = (1 / 2 : ℝ)) 
  (hS: S = (1 / 2 : ℝ)) 
  (hK: K = (1 / 2 : ℝ)) 
  (hL: L = (1 / 2 : ℝ)) : 
  KL = (1 / 4 : ℝ) := 
sorry

end length_of_KL_l701_701487


namespace ryan_sandwiches_l701_701463

theorem ryan_sandwiches (sandwich_slices : ℕ) (total_slices : ℕ) (h1 : sandwich_slices = 3) (h2 : total_slices = 15) :
  total_slices / sandwich_slices = 5 :=
by
  sorry

end ryan_sandwiches_l701_701463


namespace projection_matrix_transformation_l701_701637

theorem projection_matrix_transformation (v0 : Vector ℝ 2) :
  let p1 := (Matrix.vecCons (Matrix.vecCons (4/5) (2/5) Matrix.vecEmpty) (Matrix.vecCons (2/5) (1/5) Matrix.vecEmpty) : Matrix (Fin 2) (Fin 2) ℝ)
      p2 := (Matrix.vecCons (Matrix.vecCons (4/5) (-2/5) Matrix.vecEmpty) (Matrix.vecCons (-2/5) (1/5) Matrix.vecEmpty) : Matrix (Fin 2) (Fin 2) ℝ)
      result := (Matrix.vecCons (Matrix.vecCons (12/25) 0 Matrix.vecEmpty) (Matrix.vecCons 0 (3/25) Matrix.vecEmpty) : Matrix (Fin 2) (Fin 2) ℝ)
  (p2 * p1) = result :=
by
  sorry

end projection_matrix_transformation_l701_701637


namespace geometric_sequence_first_term_l701_701105

theorem geometric_sequence_first_term (a r : ℝ) (ar_eq_4 : a * r = 4) (ar^3_eq_16 : a * r^3 = 16) :
  a = 2 :=
begin
  sorry
end

end geometric_sequence_first_term_l701_701105


namespace length_KL_eq_one_fourth_l701_701489

section Pentagon

variable {A B C D E P Q R S K L : Type}
variable [AddCommGroup P] [Module ℕ P]

-- Assuming all sides of the pentagon are of length 1
variable (length_AB length_BC length_CD length_DE length_AE : ℕ := 1)

-- Define midpoints assumptions
variable (is_midpoint_AB : P = midpoint A B)
variable (is_midpoint_BC : Q = midpoint B C)
variable (is_midpoint_CD : R = midpoint C D)
variable (is_midpoint_DE : S = midpoint D E)

-- Define KL midpoints
variable (is_midpoint_PR : K = midpoint P R)
variable (is_midpoint_QS : L = midpoint Q S)

-- Prove the length of segment KL
theorem length_KL_eq_one_fourth :
  dist K L = 1 / 4 :=
sorry

end Pentagon

end length_KL_eq_one_fourth_l701_701489


namespace calibration_measurements_l701_701577

theorem calibration_measurements (holes : Fin 15 → ℝ) (diameter : ℝ)
  (h1 : ∀ i : Fin 15, holes i = 10 + i.val * 0.04)
  (h2 : 10 ≤ diameter ∧ diameter ≤ 10 + 14 * 0.04) :
  ∃ tries : ℕ, (tries ≤ 4) ∧ (∀ (i : Fin 15), if diameter ≤ holes i then True else False) :=
sorry

end calibration_measurements_l701_701577


namespace Tyler_CDs_count_l701_701540

-- Definitions using the conditions
def initial_CDs := 21
def fraction_given_away := 1 / 3
def CDs_bought := 8

-- The problem is to prove the final number of CDs Tyler has
theorem Tyler_CDs_count (initial_CDs : ℕ) 
  (fraction_given_away : ℚ) -- using ℚ for fractions
  (CDs_bought : ℕ) : 
  let CDs_given_away := initial_CDs * fraction_given_away in
  let remaining_CDs := initial_CDs - CDs_given_away in
  let final_CDs := remaining_CDs + CDs_bought in
  final_CDs = 22 := 
by 
  sorry

end Tyler_CDs_count_l701_701540


namespace flags_on_same_circle_l701_701517

theorem flags_on_same_circle (a : ℝ) (h_angle : a < 180) (d : ℝ) :
  ∀ (n : ℕ), ∃ (C : ℝ × ℝ) (r : ℝ), 
  ∀ (k : ℕ), k ≤ n → ∃ (x y : ℝ), (x, y) = nth_point (k, d, a) ∧ dist_sq (x, y) C = r^2 :=
by
  sorry

def nth_point (k : ℕ, d : ℝ, a : ℝ) : ℝ × ℝ :=
  -- This function would compute the position (x, y) for the k-th flag.
  sorry

def dist_sq (x y : ℝ) (C : ℝ × ℝ) : ℝ :=
  -- This function calculates the squared distance from (x, y) to center C.
  sorry

end flags_on_same_circle_l701_701517


namespace find_a_b_f_max_value_and_set_f_increasing_intervals_l701_701720

noncomputable def m := (a : ℝ, b : ℝ)
noncomputable def n (x : ℝ) := (Real.sin (2 * x), 2 * Real.cos x ^ 2)
noncomputable def f (a b x : ℝ) := (a * Real.sin (2 * x) + 2 * b * Real.cos x ^ 2)

theorem find_a_b (f0 := f a b 0 = 8) (fpi6 := f a b (Real.pi / 6) = 12) :
  a = 4 * Real.sqrt 3 ∧ b = 4 := sorry

theorem f_max_value_and_set (a_value : a = 4 * Real.sqrt 3) (b_value : b = 4) :
  ∃ x_max : ℝ, f a b x_max = 12 ∧ ∀ k : ℤ, x_max = k * Real.pi + Real.pi / 6 := sorry

theorem f_increasing_intervals (a_value : a = 4 * Real.sqrt 3) (b_value : b = 4) :
  ∀ k : ℤ, (k * Real.pi - Real.pi / 3, k * Real.pi + Real.pi / 6) = {x : ℝ |
    m ≤ x ∧ x ≤ n} := sorry

end find_a_b_f_max_value_and_set_f_increasing_intervals_l701_701720


namespace sum_of_other_endpoint_coordinates_l701_701516

theorem sum_of_other_endpoint_coordinates :
  -- The point with coordinates (3, -7) is the midpoint
  -- of a segment with one endpoint at (5, -3).
  ∃ (x y : ℤ), 
  (3 = (x + 5) / 2) ∧ (-7 = (y - 3) / 2) ∧ 
  x + y = -10 :=
by 
  -- Definitions based on the conditions
  use 1, -11
  -- The midpoint conditions
  split
  -- First coordinate midpoint condition
  { norm_num }
  split
  -- Second coordinate midpoint condition
  { norm_num }
  -- Sum of coordinates
  { norm_num }

end sum_of_other_endpoint_coordinates_l701_701516


namespace distance_stockholm_lund_malmo_l701_701082

def distance_map (start: String) (end: String) (distance_cm : ℕ) : ℕ := distance_cm

def map_scale : ℕ := 20

def convert_distance (distance_cm : ℕ) (scale : ℕ) : ℕ := distance_cm * scale

theorem distance_stockholm_lund_malmo :
  let stockholm_malmo_map := distance_map "Stockholm" "Malmo" 120 in
  let lund_malmo_map := distance_map "Lund" "Malmo" 30 in
  let stockholm_lund_map := stockholm_malmo_map - lund_malmo_map in
  let stockholm_lund_real := convert_distance stockholm_lund_map map_scale in
  let lund_malmo_real := convert_distance lund_malmo_map map_scale in
  let total_distance := stockholm_lund_real + lund_malmo_real in
  total_distance = 2400 := by sorry

end distance_stockholm_lund_malmo_l701_701082


namespace team_seating_arrangements_l701_701755

theorem team_seating_arrangements :
  let cubs := 4
  let red_sox := 3
  let yankees := 3
  let teams := 3
  (nat.factorial teams) * (nat.factorial cubs) * (nat.factorial red_sox) * (nat.factorial yankees) = 5184 := 
by
  let cubs := 4
  let red_sox := 3
  let yankees := 3
  let teams := 3
  show (nat.factorial teams) * (nat.factorial cubs) * (nat.factorial red_sox) * (nat.factorial yankees) = 5184 from sorry

end team_seating_arrangements_l701_701755


namespace MN_perpendicular_BC_l701_701002

-- Definitions:
variables {A B C K N M : Type} 
variables [RightAngleTriangle A B C] [AngleAeq90 A] [AngleCeq30 C] 

-- Circle Γ passes through point A and is tangent to side BC at midpoint K
variable (Γ : Circle) 
variable [CircPassesThroughA Γ A] [TangentAtMidpointBC Γ (Midpoint B C K)]

-- Circle Γ intersects side AC and the circumcircle of ABC at points N and M
variable [IntersectAC Γ N AC] [IntersectCircumcircleABC Γ M (Circumcircle A B C)]

-- Goal:
theorem MN_perpendicular_BC : Perpendicular M N B C :=
  sorry

end MN_perpendicular_BC_l701_701002


namespace circle_passing_through_points_l701_701850

theorem circle_passing_through_points :
  ∃ D E F : ℝ, ∀ (x y : ℝ),
    ((x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) →
    (x^2 + y^2 + D*x + E*y + F = 0) ↔
    (x^2 + y^2 - 4*x - 6*y = 0) :=
begin
  sorry,
end

end circle_passing_through_points_l701_701850


namespace area_of_figured_outside_three_tangent_circles_l701_701527

theorem area_of_figured_outside_three_tangent_circles (r : ℝ) (h : r > 0) :
  let area_triangle := (sqrt 3) * r^2
  let area_sectors := (3 * (pi / 6) * r^2)
  let area_figure := area_triangle - area_sectors
  area_figure = r^2 * (sqrt 3 - (pi / 2)) :=
by
  sorry

end area_of_figured_outside_three_tangent_circles_l701_701527


namespace slope_of_parallel_line_l701_701925

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l701_701925


namespace three_digit_integers_with_3_without_5_l701_701726

theorem three_digit_integers_with_3_without_5 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let contains_3 := ∃ d ∈ [n.digits].reverse, d = 3
  let without_5 := ∀ d ∈ [n.digits].reverse, d ≠ 5
in 
  (finset.filter (λ n, contains_3 n ∧ without_5 n) three_digit_numbers).card = 200 :=
by
  sorry

end three_digit_integers_with_3_without_5_l701_701726


namespace total_cost_is_346_l701_701012

-- Definitions of the given conditions
def total_people : ℕ := 35 + 5 + 1
def total_lunches : ℕ := total_people + 3
def vegetarian_lunches : ℕ := 10
def gluten_free_lunches : ℕ := 5
def nut_free_lunches : ℕ := 3
def halal_lunches : ℕ := 4
def veg_and_gluten_free_lunches : ℕ := 2
def regular_cost : ℕ := 7
def special_cost : ℕ := 8
def veg_and_gluten_free_cost : ℕ := 9

-- Calculate regular lunches considering dietary overlaps
def regular_lunches : ℕ := 
  total_lunches - vegetarian_lunches - gluten_free_lunches - nut_free_lunches - halal_lunches + veg_and_gluten_free_lunches

-- Calculate costs per category of lunches
def total_regular_cost : ℕ := regular_lunches * regular_cost
def total_vegetarian_cost : ℕ := (vegetarian_lunches - veg_and_gluten_free_lunches) * special_cost
def total_gluten_free_cost : ℕ := gluten_free_lunches * special_cost
def total_nut_free_cost : ℕ := nut_free_lunches * special_cost
def total_halal_cost : ℕ := halal_lunches * special_cost
def total_veg_and_gluten_free_cost : ℕ := veg_and_gluten_free_lunches * veg_and_gluten_free_cost

-- Calculate total cost
def total_cost : ℕ :=
  total_regular_cost + total_vegetarian_cost + total_gluten_free_cost + total_nut_free_cost + total_halal_cost + total_veg_and_gluten_free_cost

-- Theorem stating the main question
theorem total_cost_is_346 : total_cost = 346 :=
  by
    -- This is where the proof would go
    sorry

end total_cost_is_346_l701_701012


namespace number_of_correct_propositions_l701_701618

-- Define the propositions for the polyhedra
def proposition1 : Prop := ∀ (T : Tetrahedron), ∃ face1 face2 face3 face4 : Triangle,
  obtuse face1 ∧ obtuse face2 ∧ obtuse face3 ∧ obtuse face4

def proposition2 : Prop := ∀ (P : Polyhedron), ∀ (F : Polygon), (∀ t : Triangle, isFace t P → hasCommonVertex t F) → isPyramid P

def proposition3 : Prop := ∀ (P : Polyhedron), ∃ parallelFace1 parallelFace2 : Face,
  parallel parallelFace1 parallelFace2 ∧ (∀ t : Face, t ≠ parallelFace1 ∧ t ≠ parallelFace2 → isTrapezoid t) → isFrustum P

-- Main theorem to prove the number of correct propositions is zero
theorem number_of_correct_propositions : (¬ proposition1) ∧ (¬ proposition2) ∧ (¬ proposition3) → (∃ n : ℕ, n = 0) :=
by
  -- We would provide the proof here
  sorry

end number_of_correct_propositions_l701_701618


namespace find_sinA_and_AC_l701_701361

theorem find_sinA_and_AC (A B C : Type) [triangle A B C]
  (AB BC AC : ℝ) (hAB : AB = sqrt 2) (hBC : BC = 1) (h_cosC : cos C = 3 / 4) :
  (sin A = sqrt 14 / 8) ∧ (AC = 2) := by
  sorry

end find_sinA_and_AC_l701_701361


namespace max_x_satisfying_ineq_l701_701126

theorem max_x_satisfying_ineq : ∃ (x : ℤ), (x ≤ 1 ∧ ∀ (y : ℤ), (y > x → y > 1) ∧ (y ≤ 1 → (y : ℚ) / 3 + 7 / 4 < 9 / 4)) := 
by
  sorry

end max_x_satisfying_ineq_l701_701126


namespace solve_domain_set_solve_log_inequality_set_union_sets_intersection_sets_complement_l701_701432

noncomputable def sqrt_domain_set := {x : ℝ | x >= -1}
noncomputable def log_inequality_set := {x : ℝ | 1 < x ∧ x <= 3}

theorem solve_domain_set (x : ℝ) :
  (sqrt x + 1 ∈ set.range sqrt_domain_set ↔ x >= -1) :=
by sorry

theorem solve_log_inequality_set (x : ℝ) :
  (log 2 (x - 1) <= 1 ↔ 1 < x ∧ x <= 3) :=
by sorry

theorem union_sets (x : ℝ) :
  (sqrt_domain_set x ∨ log_inequality_set x ↔ x >= -1) :=
by sorry

theorem intersection_sets_complement (x : ℝ) :
  (sqrt_domain_set x ∧ (x <= 1 ∨ x > 3) ↔ (-1 <= x ∧ x <= 1) ∨ x > 3) :=
by sorry

end solve_domain_set_solve_log_inequality_set_union_sets_intersection_sets_complement_l701_701432


namespace sufficient_condition_of_implications_l701_701088

variables (P1 P2 θ : Prop)

theorem sufficient_condition_of_implications
  (h1 : P1 → θ)
  (h2 : P2 → P1) :
  P2 → θ :=
by sorry

end sufficient_condition_of_implications_l701_701088


namespace ferry_time_difference_l701_701279

theorem ferry_time_difference :
  ∃ (t : ℕ), (∀ (dP : ℕ) (sP : ℕ) (sQ : ℕ), dP = sP * 3 →
   dP = 24 →
   sP = 8 →
   sQ = sP + 1 →
   t = (dP * 3) / sQ - 3) ∧ t = 5 := 
  sorry

end ferry_time_difference_l701_701279


namespace graph_passes_through_point_l701_701092

theorem graph_passes_through_point (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) :
  (0, 2) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^x + 1)} :=
by
  simp
  use 0
  simp [pow_zero, add_comm]
  split
  { exact h_pos }
  { exact h_neq_one }
  sorry

end graph_passes_through_point_l701_701092


namespace positive_number_and_cube_l701_701880

theorem positive_number_and_cube (n : ℕ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 ∧ n^3 = 2744 :=
by sorry

end positive_number_and_cube_l701_701880


namespace circle_passing_points_l701_701855

theorem circle_passing_points (x y : ℝ) :
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 - 4*x - 6*y = 0 :=
by
  intros h
  cases h
  case inl h₁ => 
    rw [h₁.1, h₁.2]
    ring
  case inr h₁ =>
    cases h₁
    case inl h₂ => 
      rw [h₂.1, h₂.2]
      ring
    case inr h₂ =>
      rw [h₂.1, h₂.2]
      ring

end circle_passing_points_l701_701855


namespace students_in_johnsons_class_l701_701802

-- Define the conditions as constants/variables
def studentsInFinleysClass : ℕ := 24
def studentsAdditionalInJohnsonsClass : ℕ := 10

-- State the problem as a theorem
theorem students_in_johnsons_class : 
  let halfFinleysClass := studentsInFinleysClass / 2
  let johnsonsClass := halfFinleysClass + studentsAdditionalInJohnsonsClass
  johnsonsClass = 22 :=
by
  sorry

end students_in_johnsons_class_l701_701802


namespace marie_finishes_fourth_task_l701_701437

theorem marie_finishes_fourth_task :
  (∀ (task : ℕ), task ∈ {1, 2, 3, 4} → 
   ∃ (t_start t_end : ℕ), task ≠ 0 ∧ t_start + t_end = 210 ∧ 
   (t_start, t_end) = if task = 4 then (11 * 60 + 30, 70) else (8 * 60, _)) →
  11 * 60 + 30 + 70 = 12 * 60 + 40 :=
by
  sorry

end marie_finishes_fourth_task_l701_701437


namespace expression_varies_l701_701236

variables {x : ℝ}

noncomputable def expression (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x - 5) / ((x + 1) * (x - 3)) - (8 + x) / ((x + 1) * (x - 3))

theorem expression_varies (h1 : x ≠ -1) (h2 : x ≠ 3) : 
  ∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧ 
  expression x₀ ≠ expression x₁ :=
by
  sorry

end expression_varies_l701_701236


namespace f_28_l701_701328

def f1 (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

def f (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then x else f1 (f (n - 1) x)

theorem f_28 (x : ℝ) (h : f 35 x = f 5 x) : f 28 x = -(1 / (x - 1)) :=
by
  sorry

end f_28_l701_701328


namespace condition_1_condition_2_l701_701142

theorem condition_1 (m : ℝ) : (m^2 - 2*m - 15 > 0) ↔ (m < -3 ∨ m > 5) :=
sorry

theorem condition_2 (m : ℝ) : (2*m^2 + 3*m - 9 = 0) ∧ (7*m + 21 ≠ 0) ↔ (m = 3/2) :=
sorry

end condition_1_condition_2_l701_701142


namespace closest_d_l701_701771

noncomputable def isosceles_triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem closest_d
  (c d : ℝ)
  (h1 : c > 0)
  (h2 : d > 0)
  (h3 : 2 * c + d = 22)
  (h4 : isosceles_triangle_area 6 6 10 = isosceles_triangle_area c c d) :
  abs (d - 7) < 0.1 :=
by
  sorry

end closest_d_l701_701771


namespace factor_54x5_135x9_l701_701249

theorem factor_54x5_135x9 (x : ℝ) :
  54 * x ^ 5 - 135 * x ^ 9 = -27 * x ^ 5 * (5 * x ^ 4 - 2) :=
by 
  sorry

end factor_54x5_135x9_l701_701249


namespace prove_pqrstu_eq_416_l701_701733

-- Define the condition 1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v
def condition (p q r s t u v : ℤ) (x : ℤ) : Prop :=
  1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v

-- State the theorem to prove p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416
theorem prove_pqrstu_eq_416 (p q r s t u v : ℤ) (h : ∀ x, condition p q r s t u v x) : 
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416 :=
sorry

end prove_pqrstu_eq_416_l701_701733


namespace slope_of_parallel_line_l701_701961

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l701_701961


namespace transformer_coils_flawless_l701_701907

theorem transformer_coils_flawless (x y : ℕ) (hx : x + y = 8200)
  (hdef : (2 * x / 100) + (3 * y / 100) = 216) :
  ((x = 3000 ∧ y = 5200) ∧ ((x * 98 / 100) = 2940) ∧ ((y * 97 / 100) = 5044)) :=
by
  sorry

end transformer_coils_flawless_l701_701907


namespace problem_equivalent_l701_701111

theorem problem_equivalent (a c : ℕ) (h : (3 * 100 + a * 10 + 7) + 214 = 5 * 100 + c * 10 + 1) (h5c1_div3 : (5 + c + 1) % 3 = 0) : a + c = 4 :=
sorry

end problem_equivalent_l701_701111


namespace harry_spends_to_feed_each_snake_l701_701337

theorem harry_spends_to_feed_each_snake
  (num_geckos : ℕ := 3)
  (num_iguanas : ℕ := 2)
  (num_snakes : ℕ := 4)
  (cost_per_iguana_per_month : ℕ := 5)
  (cost_per_gecko_per_month : ℕ := 15)
  (total_annual_cost : ℕ := 1140) :
  let total_monthly_cost := total_annual_cost / 12 in
  let cost_per_snake_per_month := (total_monthly_cost - (num_geckos * cost_per_gecko_per_month + num_iguanas * cost_per_iguana_per_month)) / num_snakes in
  cost_per_snake_per_month = 10 :=
by
  sorry

end harry_spends_to_feed_each_snake_l701_701337


namespace olivia_bags_count_l701_701808

def cans_per_bag : ℕ := 5
def total_cans : ℕ := 20

theorem olivia_bags_count : total_cans / cans_per_bag = 4 := by
  sorry

end olivia_bags_count_l701_701808


namespace number_of_non_congruent_squares_on_6_by_6_grid_l701_701724

-- Define the 6x6 grid
def six_by_six_grid := {p : ℤ × ℤ // 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

-- Define the concept of a lattice point
def is_lattice_point (p : ℤ × ℤ) : Prop :=
  ∃ i j, p = (i, j) ∧ 0 ≤ i ∧ i ≤ 6 ∧ 0 ≤ j ∧ j ≤ 6

-- Define the concept of a square with vertices at lattice points
def is_square (s : set (ℤ × ℤ)) : Prop :=
  ∃ a b c d, s = {a, b, c, d} ∧
    (is_lattice_point a ∧ is_lattice_point b ∧ is_lattice_point c ∧ is_lattice_point d) ∧
    -- Additional conditions to ensure they form a square can be added

-- The theorem we want to prove
theorem number_of_non_congruent_squares_on_6_by_6_grid : 
  (∃ s : set (set (ℤ × ℤ)), 
    (∀ t ∈ s, is_square t) ∧ 
    s.card = 75) :=
sorry

end number_of_non_congruent_squares_on_6_by_6_grid_l701_701724


namespace number_of_ways_is_120_l701_701675

noncomputable def number_of_ways : ℕ :=
  let pairs := 5 in
  let choose_one_pair := pairs in
  let choose_remaining_pairs := Nat.choose 4 2 in
  let boots_per_pair := 2 in
  choose_one_pair * choose_remaining_pairs * boots_per_pair * boots_per_pair

theorem number_of_ways_is_120 (pairs : ℕ) (boots_per_pair : ℕ) (h_pairs : pairs = 5) (h_boots : boots_per_pair = 2) :
  number_of_ways = 120 :=
by
  rw [number_of_ways]
  rw [h_pairs, h_boots]
  -- More steps can go here
  sorry

end number_of_ways_is_120_l701_701675


namespace exists_constant_linear_combination_l701_701789

noncomputable def is_monotone (f : ℝ → ℝ) : Prop := sorry

theorem exists_constant_linear_combination
  (f1 f2 : ℝ → ℝ)
  (hf : ∀ a1 a2 : ℝ, is_monotone (λ x, a1 * f1 x + a2 * f2 x))
  (h_f2_not_constant : f2 0 ≠ f2 1) :
  ∃ a : ℝ, ∀ x : [0,1], f1 x - a * f2 x = f1 0 - a * f2 0 :=
sorry

end exists_constant_linear_combination_l701_701789


namespace smallest_positive_integer_multiple_of_6_and_15_is_30_l701_701264

theorem smallest_positive_integer_multiple_of_6_and_15_is_30 :
  ∃ b : ℕ, b > 0 ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ ∀ n, (n > 0 ∧ (6 ∣ n) ∧ (15 ∣ n)) → b ≤ n :=
  let b := 30 in
  ⟨b, by simp [b, dvd_refl, nat.succ_pos'], sorry⟩

end smallest_positive_integer_multiple_of_6_and_15_is_30_l701_701264


namespace parallel_slope_l701_701930

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l701_701930


namespace circle_passing_through_points_l701_701846

noncomputable def circle_equation (D E F : ℝ) : ℝ × ℝ → ℝ :=
λ p, p.1^2 + p.2^2 + D * p.1 + E * p.2 + F

theorem circle_passing_through_points : ∃ D E F : ℝ, 
  circle_equation D E F (0, 0) = 0 ∧
  circle_equation D E F (4, 0) = 0 ∧
  circle_equation D E F (-1, 1) = 0 ∧
  D = -4 ∧ 
  E = -6 ∧ 
  F = 0 :=
begin
  sorry
end

end circle_passing_through_points_l701_701846


namespace parallel_line_slope_l701_701975

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l701_701975


namespace harry_total_cost_l701_701190

-- Define the price of each type of seed packet
def pumpkin_price : ℝ := 2.50
def tomato_price : ℝ := 1.50
def chili_pepper_price : ℝ := 0.90
def zucchini_price : ℝ := 1.20
def eggplant_price : ℝ := 1.80

-- Define the quantities Harry wants to buy
def pumpkin_qty : ℕ := 4
def tomato_qty : ℕ := 6
def chili_pepper_qty : ℕ := 7
def zucchini_qty : ℕ := 3
def eggplant_qty : ℕ := 5

-- Calculate the total cost
def total_cost : ℝ :=
  pumpkin_qty * pumpkin_price +
  tomato_qty * tomato_price +
  chili_pepper_qty * chili_pepper_price +
  zucchini_qty * zucchini_price +
  eggplant_qty * eggplant_price

-- The proof problem
theorem harry_total_cost : total_cost = 38.90 := by
  sorry

end harry_total_cost_l701_701190


namespace advise_friends_choice_l701_701141

noncomputable def plannedRevenue : ℕ := 120000000
noncomputable def advancesReceived : ℕ := 30000000
noncomputable def totalIncome : ℕ := plannedRevenue + advancesReceived

noncomputable def monthlyRent : ℕ := 770000
noncomputable def monthlyVariousOils : ℕ := 1450000
noncomputable def monthlySalaries : ℕ := 4600000
noncomputable def monthlyInsuranceContributions : ℕ := 1380000
noncomputable def monthlyAccountingServices : ℕ := 340000
noncomputable def monthlyAdvertising : ℕ := 1800000
noncomputable def monthlyReTraining : ℕ := 800000
noncomputable def monthlyMiscellaneous : ℕ := 650000
noncomputable def monthlyExpenses : ℕ := monthlyRent + monthlyVariousOils + monthlySalaries + monthlyInsuranceContributions + monthlyAccountingServices + monthlyAdvertising + monthlyReTraining + monthlyMiscellaneous

noncomputable def annualExpenses : ℕ := monthlyExpenses * 12

noncomputable def incomeTaxRate : ℚ := 0.06
noncomputable def incomeTaxBase : ℕ := totalIncome

noncomputable def incomeMinusExpensesTaxRate : ℚ := 0.15
noncomputable def expenditureTaxBase : ℕ := totalIncome - annualExpenses

noncomputable def insuranceContributionsPerMonth : ℕ := monthlyInsuranceContributions
noncomputable def annualInsuranceContributions : ℕ := insuranceContributionsPerMonth * 12
noncomputable def minTaxRate : ℚ := 0.01
noncomputable def deductionLimit : ℕ := (totalIncome * incomeTaxRate).toNat / 2

noncomputable def incomeTax : ℕ := (totalIncome * incomeTaxRate).toNat - [deductionLimit, annualInsuranceContributions].min
noncomputable def expenditureTax : ℕ := ((totalIncome - annualExpenses) * incomeMinusExpensesTaxRate).toNat
noncomputable def minimumTax : ℕ := (totalIncome * minTaxRate).toNat

noncomputable def taxPayableOptionA : ℕ := (totalIncome * incomeTaxRate).toNat - [deductionLimit, annualInsuranceContributions].min
noncomputable def taxPayableOptionB : ℕ := max expenditureTax minimumTax

theorem advise_friends_choice : taxPayableOptionB < taxPayableOptionA → true := sorry

end advise_friends_choice_l701_701141


namespace gcd_exp_sub_eq_l701_701784

theorem gcd_exp_sub_eq {a b m n : ℤ} (h : gcd a b = 1) : 
  gcd (a^m - b^m) (a^n - b^n) = a^(gcd m n) - b^(gcd m n) :=
sorry

end gcd_exp_sub_eq_l701_701784


namespace slope_of_parallel_line_l701_701957

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l701_701957


namespace parallel_planes_l701_701035

variables (a b : Line) (α β : Plane)

-- Conditions
axiom line_parallel : parallel a b
axiom line_perpendicular_to_plane_a : perpendicular a α
axiom line_perpendicular_to_plane_b : perpendicular b β

-- Goal
theorem parallel_planes (a b : Line) (α β : Plane) 
  (H1 : parallel a b) 
  (H2 : perpendicular a α) 
  (H3 : perpendicular b β) 
  : parallel α β :=
sorry

end parallel_planes_l701_701035


namespace A_profit_share_l701_701215

theorem A_profit_share (A_shares : ℚ) (B_shares : ℚ) (C_shares : ℚ) (D_shares : ℚ) (total_profit : ℚ) (A_profit : ℚ) :
  A_shares = 1/3 → B_shares = 1/4 → C_shares = 1/5 → 
  D_shares = 1 - (A_shares + B_shares + C_shares) → total_profit = 2445 → A_profit = 815 →
  A_shares * total_profit = A_profit :=
by sorry

end A_profit_share_l701_701215


namespace number_of_ordered_triples_l701_701833

theorem number_of_ordered_triples (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
    (h_eq : a * b * c - b * c - a * c - a * b + a + b + c = 2013) :
    ∃ n, n = 39 :=
by
  sorry

end number_of_ordered_triples_l701_701833


namespace slope_of_parallel_line_l701_701953

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l701_701953


namespace smallest_multiple_of_6_and_15_l701_701266

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ b' : ℕ, b' > 0 ∧ b' % 6 = 0 ∧ b' % 15 = 0 → b ≤ b' :=
sorry

end smallest_multiple_of_6_and_15_l701_701266


namespace smallest_n_l701_701310

theorem smallest_n (n : ℕ) :
  (∀ m : ℤ, 0 < m ∧ m < 2001 →
    ∃ k : ℤ, (m : ℚ) / 2001 < (k : ℚ) / n ∧ (k : ℚ) / n < (m + 1 : ℚ) / 2002) ↔ n = 4003 :=
by
  sorry

end smallest_n_l701_701310


namespace abs_inequality_l701_701737

def abs (x : ℝ) : ℝ := if x ≥ 0 then x else -x

theorem abs_inequality (a : ℝ) :
  (∀ x : ℝ, abs (x - 3) + abs (x + 2) > a) → a < 5 :=
sorry

end abs_inequality_l701_701737


namespace ratio_greatest_to_smallest_distance_l701_701816

-- Definitions and conditions
def ConvexQuadrilateral (A B C D : Type) := 
-- Some definition to represent a quadrilateral being convex (not fully detailed here)
sorry

def greatest_distance_between_vertices (A B C D : Type) := 
-- Definition to compute the greatest distance between vertices (not fully detailed here)
sorry

def smallest_distance_between_vertices (A B C D : Type) :=
-- Definition to compute the smallest distance between vertices (not fully detailed here)
sorry

theorem ratio_greatest_to_smallest_distance (A B C D : Type) (h : ConvexQuadrilateral A B C D) :
  let M := greatest_distance_between_vertices A B C D in
  let m := smallest_distance_between_vertices A B C D in
  m > 0 → 
  M / m ≥ Real.sqrt 2 :=
by
  sorry

end ratio_greatest_to_smallest_distance_l701_701816


namespace min_u_condition_l701_701694

-- Define the function u and the condition
def u (x y : ℝ) : ℝ := x^2 + 4 * x + y^2 - 2 * y

def condition (x y : ℝ) : Prop := 2 * x + y ≥ 1

-- The statement we want to prove
theorem min_u_condition : ∃ (x y : ℝ), condition x y ∧ u x y = -9/5 := 
by
  sorry

end min_u_condition_l701_701694


namespace total_digits_first_1500_even_integers_l701_701136

/-- Prove that the total number of digits used to write the first 1500 positive even integers is 5448. -/
theorem total_digits_first_1500_even_integers : 
  let num_one_digit_even := 4 in
  let num_two_digit_even := 45 in
  let num_three_digit_even := 450 in
  let num_four_digit_even := 1001 in
  let digits_one_digit := num_one_digit_even * 1 in
  let digits_two_digit := num_two_digit_even * 2 in
  let digits_three_digit := num_three_digit_even * 3 in
  let digits_four_digit := num_four_digit_even * 4 in
  digits_one_digit + digits_two_digit + digits_three_digit + digits_four_digit = 5448 := by
  sorry

end total_digits_first_1500_even_integers_l701_701136


namespace rationalize_denominator_div_l701_701452

theorem rationalize_denominator_div (h : 343 = 7 ^ 3) : 7 / Real.sqrt 343 = Real.sqrt 7 / 7 := 
by 
  sorry

end rationalize_denominator_div_l701_701452


namespace cost_of_running_tv_l701_701021

-- Definitions based on the given conditions
def daily_usage_watt_hours := 125 * 4
def weekly_usage_watt_hours := daily_usage_watt_hours * 7
def weekly_usage_kw_hours := weekly_usage_watt_hours / 1000
def cost_per_week_cents := weekly_usage_kw_hours * 14

-- The theorem stating the desired conclusion
theorem cost_of_running_tv (h1: daily_usage_watt_hours = 500) 
                           (h2: weekly_usage_watt_hours = 3500) 
                           (h3: weekly_usage_kw_hours = 3.5) 
                           (h4: cost_per_week_cents = 49) : 
  cost_per_week_cents = 49 := 
by { sorry }

end cost_of_running_tv_l701_701021


namespace main_theorem_l701_701062

noncomputable def is_obtuse_angle (a b : ℝ) : Prop :=
  let dot_product := a * b
  ∃ (angle : ℝ), dot_product < 0 ∧ angle > π / 2

noncomputable def p (a b : ℝ) : Prop :=
  dot_product < 0 → is_obtuse_angle a b

noncomputable def q (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  deriv f x₀ = 0 → ∃ (x : ℝ), x = x₀ ∧ is_extremum_point f x₀

theorem main_theorem (a b : ℝ) (f : ℝ → ℝ) (x₀ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬p a b ∧ ¬q f x₀ := by
  sorry

end main_theorem_l701_701062


namespace right_triangle_inequality_l701_701782

theorem right_triangle_inequality {a b c : ℝ} (h : c^2 = a^2 + b^2) : 
  a + b ≤ c * Real.sqrt 2 :=
sorry

end right_triangle_inequality_l701_701782


namespace only_valid_product_is_24_l701_701646

noncomputable def exists_factorial_sum_eq_24 : Prop :=
  ∃ x a b c d : ℕ, (factorial x = factorial a + factorial b + factorial c + factorial d ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d) → 
  factorial x = 24

theorem only_valid_product_is_24 : exists_factorial_sum_eq_24 :=
  sorry

end only_valid_product_is_24_l701_701646


namespace fuel_cost_equation_l701_701278

theorem fuel_cost_equation (x : ℝ) (h : (x / 4) - (x / 6) = 8) : x = 96 :=
sorry

end fuel_cost_equation_l701_701278


namespace alternating_series_sum_eq_neg_1012_l701_701633

-- Definition of the alternating series from -1 to -2023
def alternating_series_sum : ℤ :=
  List.sum [ -i * (-1) ^ i | i in List.range (2024)]

-- The statement we want to prove
theorem alternating_series_sum_eq_neg_1012 :
  alternating_series_sum = -1012 :=
sorry -- proof is yet to be written

end alternating_series_sum_eq_neg_1012_l701_701633


namespace arithmetic_sequence_ninth_term_l701_701889

theorem arithmetic_sequence_ninth_term
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29) :
  a + 8 * d = 35 :=
sorry

end arithmetic_sequence_ninth_term_l701_701889


namespace solve_arcsin_equation_l701_701071

   theorem solve_arcsin_equation (x : ℝ) (h₁ : -1 ≤ x ∧ x ≤ 1) (h₂ : -1 ≤ 3 * x ∧ 3 * x ≤ 1) :
     arcsin x + arcsin (3 * x) = π / 4 ↔ x = sqrt (1 / (9 + 4 * sqrt 2)) ∨ x = -sqrt (1 / (9 + 4 * sqrt 2)) :=
   by
     sorry
   
end solve_arcsin_equation_l701_701071


namespace arithmetic_sequence_a_n_formula_sum_first_n_terms_l701_701518

open_locale big_operators

noncomputable def a_seq (n : ℕ) : ℕ := n + 6

theorem arithmetic_sequence_a_n_formula (a : ℕ → ℕ) (h_arith : ∀ m n, a (m + 1) - a m = a (n + 1) - a n)
  (h_a4 : a 4 = 10) (h_geom : a 6 * a 6 = a 3 * a 10) :
  ∀ n, a n = n + 6 :=
sorry

noncomputable def b_seq (n : ℕ) : ℕ := 2^n * a_seq n

noncomputable def sum_b_seq (n : ℕ) : ℕ :=
∑ i in finset.range n, b_seq i

theorem sum_first_n_terms (S : ℕ → ℕ) (h_S_def : ∀ n, S n = ∑ i in finset.range n, b_seq i) :
  ∀ n, S n = (n + 5) * 2^(n + 1) - 10 :=
sorry

end arithmetic_sequence_a_n_formula_sum_first_n_terms_l701_701518


namespace sequence_adjacent_diff_l701_701224

theorem sequence_adjacent_diff (f : ℕ → ℕ) (h_range : ∀ n, 1 ≤ f n ∧ f n ≤ 100) 
  (h_diff : ∀ n, n < 99 → abs (f (n + 1) - f n) ≥ 50) : 
  ( ∃ seq : ℕ → ℕ, ∀ i, seq i = if i % 2 = 0 then 51 + i / 2 else i / 2 + 1 ∧ 
                      h_range (seq i) ∧ 
                      h_diff (seq i) ) :=
by
  sorry

end sequence_adjacent_diff_l701_701224


namespace slope_of_parallel_line_l701_701965

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l701_701965


namespace vegetarians_primes_cannibals_lie_l701_701810

def is_vegetarian (n : Nat) : Prop := sorry
def is_prime (n : Nat) : Prop := sorry

theorem vegetarians_primes_cannibals_lie (N : Nat) (inhabitants : Fin N → Bool): 
  (∀ i, inhabitants i = true → (∀ j, is_vegetarian j → is_prime (abs (i - j))))
  ∧ (∀ i, inhabitants i = false → ¬ (∀ j, is_vegetarian j → is_prime (abs (i - j)))) :=
sorry

end vegetarians_primes_cannibals_lie_l701_701810


namespace percentage_40_number_l701_701356

theorem percentage_40_number (x y z P : ℝ) (hx : x = 93.75) (hy : y = 0.40 * x) (hz : z = 6) (heq : (P / 100) * y = z) :
  P = 16 :=
sorry

end percentage_40_number_l701_701356


namespace exists_rank_with_profit_2016_l701_701427

theorem exists_rank_with_profit_2016 : ∃ n : ℕ, n * (n + 1) / 2 = 2016 :=
by 
  sorry

end exists_rank_with_profit_2016_l701_701427


namespace function_range_l701_701521

noncomputable def y (x : ℝ) := sqrt (1 + 2 * x) + sqrt (1 - 2 * x)

theorem function_range : (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → sqrt(2) ≤ y x ∧ y x ≤ 2) :=
by
  sorry

end function_range_l701_701521


namespace shaded_square_area_is_two_l701_701214

noncomputable theory

-- Define the unit square and larger square
def unit_square : ℝ := 1
def larger_square : ℝ := 2 * unit_square

-- State the problem conditions
def equilateral_triangle_side : ℝ := unit_square
def shaded_square_diagonal_length : ℝ := larger_square

-- Theorem statement for the area of the shaded square
theorem shaded_square_area_is_two (side_length shaded_square : ℝ) 
  (h1 : equilateral_triangle_side = unit_square)
  (h2 : 4 * unit_square = larger_square)
  (h3 : shaded_square_diagonal_length = larger_square)
  (hc : shaded_square_diagonal_length = side_length * real.sqrt 2) :
  side_length ^ 2 = 2 :=
begin
  sorry
end

end shaded_square_area_is_two_l701_701214


namespace percentage_of_marks_l701_701173
theorem percentage_of_marks (T P : ℕ) (first_candidate_marks : ℕ) (second_candidate_ratio : ℕ → ℤ)
    (passing_marks_value : P = 160)
    (second_candidate_equation : second_candidate_ratio T = P + 20)
    (first_candidate_equation : first_candidate_marks + 40 = P)
    (second_candidate_ratio_value : second_candidate_ratio = λ t, (60 * t) / 100) 
    (total_marks_value : T = 300) :
  (first_candidate_marks * 100 / T) = 40 :=
by {
  -- skipped the proof steps, just constructing the statement
  sorry
}

end percentage_of_marks_l701_701173


namespace profit_relationship_profit_range_max_profit_l701_701475

noncomputable def profit (x : ℝ) : ℝ :=
  -20 * x ^ 2 + 100 * x + 6000

theorem profit_relationship (x : ℝ) :
  profit (x) = (60 - x) * (300 + 20 * x) - 40 * (300 + 20 * x) :=
by
  sorry
  
theorem profit_range (x : ℝ) (h : 0 ≤ x ∧ x < 20) : 
  0 ≤ profit (x) :=
by
  sorry

theorem max_profit (x : ℝ) :
  (2.5 ≤ x ∧ x < 2.6) → profit (x) ≤ 6125 := 
by
  sorry  

end profit_relationship_profit_range_max_profit_l701_701475


namespace problem_solution_l701_701685

noncomputable def sequence_a (n : ℕ) : ℚ := 
  match n with
  | 0     => 0
  | 1     => 1 / 2
  | n + 1 => (n + sequence_a n) / 2

def sequence_b (n : ℕ) : ℚ := 
  sequence_a (n + 1) - sequence_a n - 1

def sum_a (n : ℕ) : ℚ := 
  ∑ i in range n, sequence_a (i + 1)

def sum_b (n : ℕ) : ℚ := 
  ∑ i in range n, sequence_b (i + 1)

theorem problem_solution :
  (sequence_a 2 = 3 / 4) ∧
  (sequence_a 3 = 11 / 8) ∧
  (sequence_a 4 = 35 / 16) ∧
  (∀ n : ℕ, sequence_b (n + 1) = (n + 1) / 2 * sequence_b n ∧ sequence_b 1 = -3 / 4 ∧ sequence_b (n + 1) / sequence_b n = 1 / 2) ∧
  (sum_a n = (n * (n - 3)) / 2 + 3 - (3 / 2^n)) ∧
  (sum_b n = (3 / 2^(n + 1)) - 3 / 2) :=
by sorry

end problem_solution_l701_701685


namespace min_value_of_F_l701_701813

variable (x1 x2 : ℝ)

def constraints :=
  2 - 2 * x1 - x2 ≥ 0 ∧
  2 - x1 + x2 ≥ 0 ∧
  5 - x1 - x2 ≥ 0 ∧
  0 ≤ x1 ∧
  0 ≤ x2

noncomputable def F := x2 - x1

theorem min_value_of_F : constraints x1 x2 → ∃ (minF : ℝ), minF = -2 :=
by
  sorry

end min_value_of_F_l701_701813


namespace batsman_average_after_12th_innings_l701_701566

theorem batsman_average_after_12th_innings (A : ℤ) :
  (∀ A : ℤ, (11 * A + 60 = 12 * (A + 2))) → (A = 36) → (A + 2 = 38) := 
by
  intro h_avg_increase h_init_avg
  sorry

end batsman_average_after_12th_innings_l701_701566


namespace analytical_expression_l701_701471

noncomputable def amplitude : ℝ := 2
noncomputable def min_period : ℝ := π / 2
noncomputable def initial_phase : ℝ := -3

theorem analytical_expression (A ω : ℝ) (hA : A > 0) (hω : ω > 0)
  (amp : A = 2) (per : (2 * π / ω) = (π / 2)) (phi : initial_phase = -3) :
  ∃ (x : ℝ) (y : ℝ), y = 2 * sin (4 * x - 3) :=
by
  sorry

end analytical_expression_l701_701471


namespace range_of_m_l701_701713

def f (a x : ℝ) : ℝ := a * (x^2 + 1)

theorem range_of_m (a x m : ℝ) (h1 : a ∈ Ioo (-4) (-2)) (h2 : x ∈ Icc (1) (3)) 
  (h3 : ∀ (a ∈ Ioo (-4) (-2)) (x ∈ Icc (1) (3)), m * a - f a x > a^2 + Real.log x) : 
  m ≤ -2 :=
sorry

end range_of_m_l701_701713


namespace simplify_polynomial_l701_701465

variable (p : ℕ)

theorem simplify_polynomial :
  (2 * p^4 + 5 * p^3 - 3 * p + 4) + (-p^4 + 2 * p^3 - 7 * p^2 + 4 * p - 2) = 
  p^4 + 7 * p^3 - 7 * p^2 + p + 2 :=
by
  sorry

end simplify_polynomial_l701_701465


namespace limit_expression_l701_701314

variable (f : ℝ → ℝ)

theorem limit_expression (h : deriv f 1 = 1) :
  tendsto (λ x : ℝ, (f (1 - x) - f (1 + x)) / (5 * x)) (𝓝 0) (𝓝 (-2 / 5)) := by
  sorry

end limit_expression_l701_701314


namespace sum_of_solutions_l701_701133

theorem sum_of_solutions : 
  (∑ x in (finset.range 31).filter (λ x, 0 < x ∧ (17 * (4 * x - 3) % 10 = 34 % 10)), x) = 51 := 
by sorry

end sum_of_solutions_l701_701133


namespace max_edges_for_winning_strategy_l701_701217

theorem max_edges_for_winning_strategy (G : Graph) (n : ℕ) (m : ℕ) (A B C : G.Vertex)
  (h_n_gt_2 : n > 2) (h_connected : G.Connected)
  (h_AB_not_adj : ¬ G.Adjacent A B)
  (h_A_wins_strategy : ∃ (winning_strategy : ∀ {A B : G.Vertex}, G.winning_strategy A B), true) :
  m = nat.choose (n - 1) 2 + 1 :=
sorry

end max_edges_for_winning_strategy_l701_701217


namespace length_KL_eq_one_fourth_l701_701488

section Pentagon

variable {A B C D E P Q R S K L : Type}
variable [AddCommGroup P] [Module ℕ P]

-- Assuming all sides of the pentagon are of length 1
variable (length_AB length_BC length_CD length_DE length_AE : ℕ := 1)

-- Define midpoints assumptions
variable (is_midpoint_AB : P = midpoint A B)
variable (is_midpoint_BC : Q = midpoint B C)
variable (is_midpoint_CD : R = midpoint C D)
variable (is_midpoint_DE : S = midpoint D E)

-- Define KL midpoints
variable (is_midpoint_PR : K = midpoint P R)
variable (is_midpoint_QS : L = midpoint Q S)

-- Prove the length of segment KL
theorem length_KL_eq_one_fourth :
  dist K L = 1 / 4 :=
sorry

end Pentagon

end length_KL_eq_one_fourth_l701_701488


namespace robie_chocolate_purchase_l701_701823

theorem robie_chocolate_purchase
  (initial_bags : ℕ := 3)
  (pieces_per_bag : ℕ := 30)
  (cost_per_bag : ℕ := 12)
  (additional_bags : ℕ := 3)
  (discount_rate : ℝ := 0.50)
  (sibling_dist : ℝ × ℝ × ℝ := (0.40, 0.30, 0.30)) :
  let initial_total_chocolates := initial_bags * pieces_per_bag,
      distributed_chocolates := 2 * pieces_per_bag,
      oldest_chocolates := sibling_dist.1 * distributed_chocolates,
      second_chocolates := sibling_dist.2 * distributed_chocolates,
      last_two_shared := (sibling_dist.3 * distributed_chocolates) / 2,
      discounted_third_bag := cost_per_bag * discount_rate,
      total_additional_cost := (additional_bags - 1) * cost_per_bag + discounted_third_bag,
      initial_cost := initial_bags * cost_per_bag,
      total_cost := initial_cost + total_additional_cost,
      remaining_chocolates := 3 * pieces_per_bag in
  total_cost = 66 ∧ 
  remaining_chocolates = 90 ∧
  (oldest_chocolates = 24 ∧ 
  second_chocolates = 18 ∧ 
  last_two_shared = 9 ∧ 
  remaining_chocolates = 90) := 
by 
  intros; 
  sorry

end robie_chocolate_purchase_l701_701823


namespace intersection_sets_l701_701717

theorem intersection_sets (M N : Set ℝ) :
  (M = {x | x * (x - 3) < 0}) → (N = {x | |x| < 2}) → (M ∩ N = {x | 0 < x ∧ x < 2}) :=
by
  intro hM hN
  rw [hM, hN]
  sorry

end intersection_sets_l701_701717


namespace smallest_multiple_of_6_and_15_l701_701267

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ b' : ℕ, b' > 0 ∧ b' % 6 = 0 ∧ b' % 15 = 0 → b ≤ b' :=
sorry

end smallest_multiple_of_6_and_15_l701_701267


namespace find_angle_A_l701_701748

noncomputable def angle_A_measure : ℝ :=
  let m := (Real.cos (3*angle_A/2), Real.sin (3*angle_A/2))
  let n := (Real.cos (angle_A/2), Real.sin (angle_A/2))
  if (Real.sqrt ((m.1 + n.1)^2 + (m.2 + n.2)^2) = Real.sqrt 3) && (0 < angle_A) && (angle_A < Real.pi) then
    angle_A
  else 
    0

theorem find_angle_A (A : ℝ) : 
  let m := (Real.cos (3 * A / 2), Real.sin (3 * A / 2))
  let n := (Real.cos (A / 2), Real.sin (A / 2))
  sqrt ((m.1 + n.1)^2 + (m.2 + n.2)^2) = sqrt 3 → 
  0 < A → 
  A < pi → 
  A = pi / 3 :=
by
  intros m n cond1 cond2 cond3
  sorry

end find_angle_A_l701_701748


namespace mod_calculation_l701_701132

theorem mod_calculation : (9^7 + 8^8 + 7^9) % 5 = 2 := by
  sorry

end mod_calculation_l701_701132


namespace jane_spent_more_on_ice_cream_l701_701773

-- Definitions based on the conditions
def ice_cream_cone_cost : ℕ := 5
def pudding_cup_cost : ℕ := 2
def ice_cream_cones_bought : ℕ := 15
def pudding_cups_bought : ℕ := 5

-- The mathematically equivalent proof statement
theorem jane_spent_more_on_ice_cream : 
  (ice_cream_cones_bought * ice_cream_cone_cost - pudding_cups_bought * pudding_cup_cost) = 65 := 
by
  sorry

end jane_spent_more_on_ice_cream_l701_701773


namespace advise_friends_choice_l701_701140

noncomputable def plannedRevenue : ℕ := 120000000
noncomputable def advancesReceived : ℕ := 30000000
noncomputable def totalIncome : ℕ := plannedRevenue + advancesReceived

noncomputable def monthlyRent : ℕ := 770000
noncomputable def monthlyVariousOils : ℕ := 1450000
noncomputable def monthlySalaries : ℕ := 4600000
noncomputable def monthlyInsuranceContributions : ℕ := 1380000
noncomputable def monthlyAccountingServices : ℕ := 340000
noncomputable def monthlyAdvertising : ℕ := 1800000
noncomputable def monthlyReTraining : ℕ := 800000
noncomputable def monthlyMiscellaneous : ℕ := 650000
noncomputable def monthlyExpenses : ℕ := monthlyRent + monthlyVariousOils + monthlySalaries + monthlyInsuranceContributions + monthlyAccountingServices + monthlyAdvertising + monthlyReTraining + monthlyMiscellaneous

noncomputable def annualExpenses : ℕ := monthlyExpenses * 12

noncomputable def incomeTaxRate : ℚ := 0.06
noncomputable def incomeTaxBase : ℕ := totalIncome

noncomputable def incomeMinusExpensesTaxRate : ℚ := 0.15
noncomputable def expenditureTaxBase : ℕ := totalIncome - annualExpenses

noncomputable def insuranceContributionsPerMonth : ℕ := monthlyInsuranceContributions
noncomputable def annualInsuranceContributions : ℕ := insuranceContributionsPerMonth * 12
noncomputable def minTaxRate : ℚ := 0.01
noncomputable def deductionLimit : ℕ := (totalIncome * incomeTaxRate).toNat / 2

noncomputable def incomeTax : ℕ := (totalIncome * incomeTaxRate).toNat - [deductionLimit, annualInsuranceContributions].min
noncomputable def expenditureTax : ℕ := ((totalIncome - annualExpenses) * incomeMinusExpensesTaxRate).toNat
noncomputable def minimumTax : ℕ := (totalIncome * minTaxRate).toNat

noncomputable def taxPayableOptionA : ℕ := (totalIncome * incomeTaxRate).toNat - [deductionLimit, annualInsuranceContributions].min
noncomputable def taxPayableOptionB : ℕ := max expenditureTax minimumTax

theorem advise_friends_choice : taxPayableOptionB < taxPayableOptionA → true := sorry

end advise_friends_choice_l701_701140


namespace min_distance_from_P_to_circle_l701_701290

-- Define the circle and point P
def circle (x y : ℝ) := (x - 1)^2 + y^2 = 4
def P : ℝ × ℝ := (-2, -3)

-- Define the distance between two points
def dist (A B : ℝ × ℝ) := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Point Q is any point on the circle
def Q (x y : ℝ) := circle x y

-- Minimum distance from point P to any point Q on the circle
theorem min_distance_from_P_to_circle :
  ∃ x y, circle x y ∧ dist P (x, y) = 3 * real.sqrt 2 - 2 := sorry

end min_distance_from_P_to_circle_l701_701290


namespace probability_meeting_l701_701123

-- Definitions based on the conditions
def meets_within_10_days_and_3_wait (x y : ℝ) : Prop :=
  x ≥ 0 ∧ x ≤ 10 ∧ y ≥ 0 ∧ y ≤ 10 ∧ abs (x - y) ≤ 3

-- Statement to prove the probability
theorem probability_meeting (A B : Type) [measure_theory.measure_space A] [measure_theory.measure_space B]
  (uniform_A : measure_theory.measure A) (uniform_B : measure_theory.measure B)
  (hA : uniform_A.set_of (λ x, x ∈ set.Icc (0:ℝ) 10) = 1)
  (hB : uniform_B.set_of (λ y, y ∈ set.Icc (0:ℝ) 10) = 1) :
  measure_theory.measure ((uniform_A : measure_theory.measure (A × B)).prod uniform_B)
    {p | meets_within_10_days_and_3_wait p.1 p.2} = 0.7 :=
sorry

end probability_meeting_l701_701123


namespace even_digits_count_529_base7_l701_701641

-- Lean 4 statement: Prove that the number of even digits in the base-7 representation of 529 (base 10) is 1.

theorem even_digits_count_529_base7 : 
  let n := 529
  let base := 7
  let digits := [1, 3, 5, 4] -- Conversion of 529_10 to base-7 equivalent which is 1354_7
  even_digits := digits.filter (λ d => d % 2 = 0)
in
  even_digits.length = 1
:= by sorry

end even_digits_count_529_base7_l701_701641


namespace factor_expression_l701_701250

theorem factor_expression (y : ℝ) : 49 - 16*y^2 + 8*y = (7 - 4*y)*(7 + 4*y) := 
sorry

end factor_expression_l701_701250


namespace purely_imaginary_fourth_quadrant_l701_701430

noncomputable section

open Complex

-- Problem 1: Prove m = 2/3 given the conditions for z to be purely imaginary
theorem purely_imaginary {m : ℝ} :
  let z := (3 * m - 2) + (m - 1) * complex.i in
  (3 * m - 2 = 0) ∧ (m - 1 ≠ 0) → m = 2 / 3 := 
sorry

-- Problem 2: Prove 2/3 < m < 1 given the conditions for z to lie in the fourth quadrant
theorem fourth_quadrant {m : ℝ} :
  let z := (3 * m - 2) + (m - 1) * complex.i in
  (3 * m - 2 > 0) ∧ (m - 1 < 0) → (2 / 3 < m) ∧ (m < 1) := 
sorry

end purely_imaginary_fourth_quadrant_l701_701430


namespace value_of_V2_at_x_equals_2_l701_701125

def f (x : ℝ) : ℝ := 2 * x^5 - 3 * x^3 + 2 * x^2 - x + 5

theorem value_of_V2_at_x_equals_2 : 
  let V0 := 2 in
  let V1 := V0 * 2 in
  let V2 := V1 * 2 - 3 in
  V2 = 5 := by
  let V0 := 2
  let V1 := V0 * 2
  let V2 := V1 * 2 - 3
  exact eq.refl V2

end value_of_V2_at_x_equals_2_l701_701125


namespace find_divisor_l701_701662

noncomputable def divisor_of_nearest_divisible (a b : ℕ) (d : ℕ) : ℕ :=
  if h : b % d = 0 ∧ (b - a < d) then d else 0

theorem find_divisor (a b : ℕ) (d : ℕ) (h1 : b = 462) (h2 : a = 457)
  (h3 : b % d = 0) (h4 : b - a < d) :
  d = 5 :=
sorry

end find_divisor_l701_701662


namespace planes_perpendicular_l701_701420

theorem planes_perpendicular (l : line) (α β : plane) 
  (h1 : l ∥ α) (h2 : l ⊥ β) : α ⊥ β :=
by
  sorry

end planes_perpendicular_l701_701420


namespace four_ways_to_cut_square_l701_701619

def cutPosition : Type := ℕ × ℕ
def isCongruent (part1 part2 : set cutPosition) : Prop := sorry

structure Square :=
  (size : ℕ)
  (holes : set cutPosition)
  (cuts : set (set cutPosition))

noncomputable def waysToCutSquare : Square := 
{ size := 4,
  holes := {(0, 0), (3, 3)},   -- Assume these are the hole positions as per the example
  cuts := { 
    { set_of (λ pos : cutPosition, pos.1 < 2), set_of (λ pos : cutPosition, pos.1 ≥ 2) },  -- Vertical cut
    { set_of (λ pos : cutPosition, pos.2 < 2), set_of (λ pos : cutPosition, pos.2 ≥ 2) },  -- Horizontal cut
    { set_of (λ pos : cutPosition, pos.1 - pos.2 < 0), set_of (λ pos : cutPosition, pos.1 - pos.2 ≥ 0) },  -- Diagonal \( \)
    { set_of (λ pos : cutPosition, pos.1 + pos.2 < 4), set_of (λ pos : cutPosition, pos.1 + pos.2 ≥ 4) }  -- Diagonal / 
  } }

theorem four_ways_to_cut_square : 
  ∃ (cut_patterns : set (set cutPosition)), 
    (∀ cut_pattern ∈ cut_patterns, 
      ∃ (part1 part2 : set cutPosition), 
        isCongruent part1 part2 ∧ part1 ∪ part2 = set.univ \ waysToCutSquare.holes) 
    ∧ card cut_patterns ≥ 4 :=
sorry

end four_ways_to_cut_square_l701_701619


namespace parallel_slope_l701_701933

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l701_701933


namespace cost_of_running_tv_l701_701020

-- Definitions based on the given conditions
def daily_usage_watt_hours := 125 * 4
def weekly_usage_watt_hours := daily_usage_watt_hours * 7
def weekly_usage_kw_hours := weekly_usage_watt_hours / 1000
def cost_per_week_cents := weekly_usage_kw_hours * 14

-- The theorem stating the desired conclusion
theorem cost_of_running_tv (h1: daily_usage_watt_hours = 500) 
                           (h2: weekly_usage_watt_hours = 3500) 
                           (h3: weekly_usage_kw_hours = 3.5) 
                           (h4: cost_per_week_cents = 49) : 
  cost_per_week_cents = 49 := 
by { sorry }

end cost_of_running_tv_l701_701020


namespace gcd_98_63_l701_701545

theorem gcd_98_63 : Nat.gcd 98 63 = 7 :=
by
  sorry

end gcd_98_63_l701_701545


namespace find_side_a_of_triangle_l701_701363

theorem find_side_a_of_triangle
  (b c : ℝ)
  (h₀ : b + c = 7)
  (h₁ : b * c = 11)
  (hA : ∠ABC = 60) :
  ∃ a : ℝ, a = 4 :=
by {
  have h2 : b^2 + c^2 = 27 := by {
    calc
      b^2 + c^2 = (b + c)^2 - 2 * b * c : by ring
            ... = 7^2 - 2 * 11 : by rw [h₀, h₁]
            ... = 27 : by norm_num,
  },
  use 4,
  have h3 : cos 60 = 1/2 := by norm_num,
  have h4 : cos 60 = (27 - 4^2) / (2 * 11) := by {
    calc
      cos 60 = (b^2 + c^2 - 4^2) / (2 * b * c) : by { rw hA, ring, }
            ... = (27 - 16) / 22 : by { rw [h2], ring, }
            ... = 11 / 22 : by ring
            ... = 1/2 : by norm_num,
  },
  rw ← h4 at h3, exact eq.symm h3,
  sorry
}

end find_side_a_of_triangle_l701_701363


namespace at_least_half_sectors_contain_frogs_l701_701176

variables {n : ℕ} (frogs : ℕ → (fin n → ℤ))

theorem at_least_half_sectors_contain_frogs
  (h1 : n > 0)
  (h2 : ∃ k > 0, frogs k = n + 1)
  (h3 : ∀ k, frogs (k + 1) = frogs k - 2 ∨ frogs (k + 1) = frogs k + 2) :
  ∃ t, ∃ s ⊆ fin n, s.card ≥ n / 2 ∧ ∀ i ∈ s, frogs t i > 0 := sorry

end at_least_half_sectors_contain_frogs_l701_701176


namespace erick_total_money_collected_l701_701144

def initial_fruit_quantities : ℕ × ℕ × ℕ × ℕ := (80, 140, 60, 100)
def original_prices : ℕ × ℕ × ℕ × ℕ := (8, 7, 5, 4)
def price_increases : ℕ × ℕ × ℕ × ℕ := (50, 25, 10, 20)

theorem erick_total_money_collected :
  let (lemons_q, grapes_q, oranges_q, apples_q) := initial_fruit_quantities
  let (lemons_p, grapes_p, oranges_p, apples_p) := original_prices
  let (lemons_inc, grapes_inc, oranges_inc, apples_inc) := price_increases
  let new_lemons_p := lemons_p + lemons_p * lemons_inc / 100
  let new_grapes_p := grapes_p + grapes_p * grapes_inc / 100
  let new_oranges_p := oranges_p + oranges_p * oranges_inc / 100
  let new_apples_p := apples_p + apples_p * apples_inc / 100
  lemons_q * new_lemons_p 
  + grapes_q * new_grapes_p 
  + oranges_q * new_oranges_p 
  + apples_q * new_apples_p = 2995 :=
begin
  sorry
end

end erick_total_money_collected_l701_701144


namespace slope_of_parallel_line_l701_701987

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l701_701987


namespace length_of_KL_l701_701485

theorem length_of_KL (A B C D E P Q R S K L : Type) 
  (hA: A = (1 : ℝ)) 
  (hB: B = (1 : ℝ)) 
  (hC: C = (1 : ℝ)) 
  (hD: D = (1 : ℝ)) 
  (hE: E = (1 : ℝ)) 
  (hP: P = (1 / 2 : ℝ)) 
  (hQ: Q = (1 / 2 : ℝ)) 
  (hR: R = (1 / 2 : ℝ)) 
  (hS: S = (1 / 2 : ℝ)) 
  (hK: K = (1 / 2 : ℝ)) 
  (hL: L = (1 / 2 : ℝ)) : 
  KL = (1 / 4 : ℝ) := 
sorry

end length_of_KL_l701_701485


namespace find_angle_C_l701_701362

noncomputable def triangle_side_relation (a b c : ℝ) : Prop := a + b = sqrt 2 * c
noncomputable def triangle_area_relation (a b c : ℝ) (S : ℝ) : Prop := S = (1/2) * a * b * sin c
noncomputable def sin_sum_relation (A B C : ℝ) : Prop := sin A + sin B = sqrt 2 * sin C

theorem find_angle_C (a b c A B C S : ℝ)
  (h1 : a + b = sqrt 2)
  (h2 : S = (1/6) * sin C)
  (h3 : sin A + sin B = sqrt 2 * sin C) :
  C = π / 3 :=
sorry

end find_angle_C_l701_701362


namespace planting_scheme_correct_l701_701756

-- Setting up the problem as the conditions given
def types_of_seeds := ["peanuts", "Chinese cabbage", "potatoes", "corn", "wheat", "apples"]

def first_plot_seeds := ["corn", "apples"]

def planting_schemes_count : ℕ :=
  let choose_first_plot := 2  -- C(2, 1), choosing either "corn" or "apples" for the first plot
  let remaining_seeds := 5  -- 6 - 1 = 5 remaining seeds after choosing for the first plot
  let arrangements_remaining := 5 * 4 * 3  -- A(5, 3), arrangements of 3 plots from 5 remaining seeds
  choose_first_plot * arrangements_remaining

theorem planting_scheme_correct : planting_schemes_count = 120 := by
  sorry

end planting_scheme_correct_l701_701756


namespace volume_parallelepiped_l701_701347

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (unit_a : ‖a‖ = 1) (unit_b : ‖b‖ = 1)
variable (angle_ab : real.angle (a ⬝ b) = π / 3)

theorem volume_parallelepiped :
  let v1 := a
  let v2 := b + (b ×ₐ a)
  let v3 := b
  ‖ v1 ⬝ (v2 ×ₙ v3) ‖ = 3 / 4 :=
by
  sorry

end volume_parallelepiped_l701_701347


namespace main_expression_l701_701667

/-- Define the logarithm with base 3 and square root of 27 calculation -/
def log3_sqrt27 : ℝ := real.log 27 / real.log 3 / 2

/-- Define the logarithm base 10 of 25 and 4 and their product -/
def lg_25 : ℝ := real.log10 25
def lg_4 : ℝ := real.log10 4

/-- Calculate 7 raised to the power of log base 7 of 2 -/
def exp_base7_log2 : ℝ := 2

/-- Calculate (-9.8) raised to the power of 0 -/
def neg_9_8_pow0 : ℝ := 1

/-- The main expression to be proven -/
theorem main_expression : log3_sqrt27 + lg_25 + lg_4 - exp_base7_log2 - neg_9_8_pow0 = 1 / 2 :=
by
  sorry

end main_expression_l701_701667


namespace parallel_slope_l701_701935

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l701_701935


namespace valid_arrangements_count_is_20_l701_701070

noncomputable def count_valid_arrangements : ℕ :=
  sorry

theorem valid_arrangements_count_is_20 :
  count_valid_arrangements = 20 :=
  by
    sorry

end valid_arrangements_count_is_20_l701_701070


namespace michael_eggs_count_l701_701443

-- Define the conditions: crates bought on Tuesday, crates given to Susan, crates bought on Thursday, and eggs per crate.
def crates_bought_on_tuesday : ℕ := 6
def crates_given_to_susan : ℕ := 2
def crates_bought_on_thursday : ℕ := 5
def eggs_per_crate : ℕ := 30

-- State the theorem to prove.
theorem michael_eggs_count :
  let crates_left = crates_bought_on_tuesday - crates_given_to_susan
  let total_crates = crates_left + crates_bought_on_thursday
  total_crates * eggs_per_crate = 270 :=
by
  -- Proof goes here
  sorry

end michael_eggs_count_l701_701443


namespace cubes_side_length_l701_701909

theorem cubes_side_length (s : ℝ) (h : 2 * (s * s + s * 2 * s + s * 2 * s) = 10) : s = 1 :=
by
  sorry

end cubes_side_length_l701_701909


namespace part1_part2_part3_l701_701785

-- Definitions for the given conditions
def z (n : ℕ) : ℂ := 
  if h : n = 0 then 3 + 4 * complex.I else sorry -- Initial value and recurrence relation can be inserted later

-- Part (1): Proving the values of z_2, z_3, and z_4
theorem part1 (n : ℕ) (h : n = 2 ∨ n = 3 ∨ n = 4) : 
  (z 2 = -1 + 7 * complex.I) ∧ 
  (z 3 = -8 + 6 * complex.I) ∧ 
  (z 4 = -14 - 2 * complex.I) := 
by 
  sorry

-- Part (2): Proving the existence of positive integer n
theorem part2 (n : ℕ) : 
  (∃ k : ℕ, n = 4 * k + 1) ↔ (z n = (1 + complex.I)^(n - 1) * (3 + 4 * complex.I)) := 
by
  sorry

-- Part (3): Proving the sum of first 102 terms of x_n * y_n
noncomputable def x_n (n : ℕ) : ℝ := sorry
noncomputable def y_n (n : ℕ) : ℝ := sorry
def x_y (n : ℕ) : ℝ := x_n n * y_n n

theorem part3 :
  (Σ n in range 102, x_y n) = 2^26 - 1 := 
by
  sorry

end part1_part2_part3_l701_701785


namespace frequency_group_5_l701_701179

theorem frequency_group_5 (total_students : ℕ) (freq1 freq2 freq3 freq4 : ℕ)
  (h1 : total_students = 45)
  (h2 : freq1 = 12)
  (h3 : freq2 = 11)
  (h4 : freq3 = 9)
  (h5 : freq4 = 4) :
  ((total_students - (freq1 + freq2 + freq3 + freq4)) / total_students : ℚ) = 0.2 := 
sorry

end frequency_group_5_l701_701179


namespace constructible_triangle_l701_701639

theorem constructible_triangle (AB m_C : ℝ) (hAB_pos : AB > 0) :
  (∃ C : ℝ × ℝ, -- C is some (x, y) coordinates
    let x := C.1,
    let y := C.2,
    y = m_C ∧
    -- Thales' circle constraint
    x^2 + y^2 = (AB / 2)^2 ∧ 
    -- Validity constraint based on altitude
    m_C ≤ AB / 2 ) ↔ m_C ≤ AB / 2 :=
by
  sorry

end constructible_triangle_l701_701639


namespace find_k_l701_701557

theorem find_k (x y k : ℝ) 
  (line1 : y = 3 * x + 2) 
  (line2 : y = -4 * x - 14) 
  (line3 : y = 2 * x + k) :
  k = -2 / 7 := 
by {
  sorry
}

end find_k_l701_701557


namespace geometric_probability_l701_701820

-- Definition of intervals
def total_interval := [-1, 2]
def desired_interval := [0, 1]

-- Definition of lengths of intervals
def length (a b : Real) : Real := b - a

-- The main theorem stating the probability
theorem geometric_probability :
  (length 0 1) / (length (-1) 2) = 1 / 3 :=
by
  -- Length of the desired interval [0, 1]
  have h1: length 0 1 = 1 := by norm_num

  -- Length of the total interval [-1, 2]
  have h2: length (-1) 2 = 3 := by norm_num

  -- Compute the probability
  rw [h1, h2]
  norm_num
  sorry

end geometric_probability_l701_701820


namespace line_equation_l701_701086

-- Define the conditions: point (2,1) on the line and slope is 2
def point_on_line (x y : ℝ) (m b : ℝ) : Prop := y = m * x + b

def slope_of_line (m : ℝ) : Prop := m = 2

-- Prove the equation of the line is 2x - y - 3 = 0
theorem line_equation (b : ℝ) (h1 : point_on_line 2 1 2 b) : 2 * 2 - 1 - 3 = 0 := by
  sorry

end line_equation_l701_701086


namespace area_bounded_by_graphs_l701_701655

theorem area_bounded_by_graphs :
  let square := set.Icc (0 : ℝ) 2 ×ˢ set.Icc (0 : ℝ) 2 in
  set.volume (square) = 4 := by
  sorry

end area_bounded_by_graphs_l701_701655


namespace penalty_kicks_total_l701_701836

theorem penalty_kicks_total (total_players goalies : ℕ) (h_total : total_players = 20) (h_goalies : goalies = 3) :
  let field_players := total_players - goalies in
  let shooters := field_players + goalies in
  let kicks_per_goalie := shooters - 1 in
  let total_kicks := goalies * kicks_per_goalie in
  total_kicks = 57 :=
by
  -- bringing the conditions into scope
  rw [h_total, h_goalies]
  -- unfold definitions
  let field_players := 20 - 3
  let shooters := field_players + 3
  let kicks_per_goalie := shooters - 1
  let total_kicks := 3 * kicks_per_goalie
  -- show the total kicks is 57
  calc
    total_kicks = 3 * (20 - 3) := by rw [h_total, h_goalies]
    ...         = 3 * 19 := by norm_num
    ...         = 57 := by norm_num
  sorry

end penalty_kicks_total_l701_701836


namespace fly_distance_covered_l701_701535

theorem fly_distance_covered (cyclist_speed : ℝ) (initial_distance : ℝ) (fly_speed : ℝ)
    (cyclist_speed_pos : cyclist_speed > 0) (initial_distance_pos : initial_distance > 0) (fly_speed_pos : fly_speed > 0) : 
    initial_distance / (2 * cyclist_speed) * fly_speed = 37.5 := 
by 
  -- Let's assert the calculations for clarification
  let relative_speed := 2 * cyclist_speed
  have calc_relative_speed : relative_speed = 2 * cyclist_speed := rfl
  let time_to_meet := initial_distance / relative_speed
  have calc_time_to_meet : time_to_meet = initial_distance / relative_speed := rfl
  let distance_covered := time_to_meet * fly_speed
  have calc_distance_covered : distance_covered = time_to_meet * fly_speed := rfl
  -- Given specific values
  have specific_relative_speed : relative_speed = 20 := by rw [calc_relative_speed]; norm_num
  have specific_time_to_meet : time_to_meet = 2.5 := by rw [← calc_time_to_meet, specific_relative_speed]; norm_num
  have specific_distance_covered : distance_covered = 37.5 := by rw [← calc_distance_covered, specific_time_to_meet]; norm_num
  exact specific_distance_covered


end fly_distance_covered_l701_701535


namespace total_bees_in_hive_l701_701895

theorem total_bees_in_hive (initial_bees : ℕ) (incoming_bees : ℕ) (total_bees : ℕ) :
  initial_bees = 16 ∧ incoming_bees = 8 → total_bees = initial_bees + incoming_bees → total_bees = 24 :=
by
  intros h₁ h₂
  cases h₁ with h₁₁ h₁₂
  rw [h₁₁, h₁₂] at h₂
  simp at h₂
  exact h₂

end total_bees_in_hive_l701_701895


namespace nancy_savings_l701_701444

theorem nancy_savings :
  let dozen := 12 in
  let quarters := 1 * dozen in
  let quarter_value := 25 in
  let total_cents := quarters * quarter_value in
  let dollars := total_cents / 100 in
  dollars = 3 :=
sorry

end nancy_savings_l701_701444


namespace subtraction_addition_example_l701_701674

theorem subtraction_addition_example :
  1500000000000 - 877888888888 + 123456789012 = 745567900124 :=
by
  sorry

end subtraction_addition_example_l701_701674


namespace smallest_positive_integer_multiple_of_6_and_15_is_30_l701_701262

theorem smallest_positive_integer_multiple_of_6_and_15_is_30 :
  ∃ b : ℕ, b > 0 ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ ∀ n, (n > 0 ∧ (6 ∣ n) ∧ (15 ∣ n)) → b ≤ n :=
  let b := 30 in
  ⟨b, by simp [b, dvd_refl, nat.succ_pos'], sorry⟩

end smallest_positive_integer_multiple_of_6_and_15_is_30_l701_701262


namespace probability_three_specific_cards_l701_701735

noncomputable def deck_size : ℕ := 52
noncomputable def num_suits : ℕ := 4
noncomputable def cards_per_suit : ℕ := 13
noncomputable def p_king_spades : ℚ := 1 / deck_size
noncomputable def p_10_hearts : ℚ := 1 / (deck_size - 1)
noncomputable def p_queen : ℚ := 4 / (deck_size - 2)

theorem probability_three_specific_cards :
  (p_king_spades * p_10_hearts * p_queen) = 1 / 33150 := 
sorry

end probability_three_specific_cards_l701_701735


namespace analytical_expression_of_C3_l701_701097

def C1 (x : ℝ) : ℝ := x^2 - 2*x + 3
def C2 (x : ℝ) : ℝ := C1 (x + 1)
def C3 (x : ℝ) : ℝ := C2 (-x)

theorem analytical_expression_of_C3 :
  ∀ x, C3 x = x^2 + 2 := by
  sorry

end analytical_expression_of_C3_l701_701097


namespace equation_of_line_through_midpoint_and_intercepting_ellipse_l701_701695

noncomputable def midpoint := (4, 2 : ℝ)

noncomputable def ellipse (x y : ℝ) := (x^2 / 36) + (y^2 / 9) = 1

theorem equation_of_line_through_midpoint_and_intercepting_ellipse :
  ∀ l : ℝ × ℝ → Prop,
  (∀ P1 P2 : ℝ × ℝ, l P1 ∧ l P2 → ellipse P1.fst P1.snd ∧ ellipse P2.fst P2.snd → 
    (P1.fst + P2.fst) / 2 = midpoint.fst ∧ (P1.snd + P2.snd) / 2 = midpoint.snd) →
  (∃ m b : ℝ, ∀ x y : ℝ, l (x, y) ↔ y = m * x + b) → 
  ∃ m b : ℝ, m = -1/2 ∧ 8 = b ∧ ∀ x y : ℝ, l (x, y) ↔ x + 2 * y - 8 = 0 :=
by
  sorry

end equation_of_line_through_midpoint_and_intercepting_ellipse_l701_701695


namespace probability_area_less_than_perimeter_l701_701206

def valid_sum (s : ℕ) : Prop := 2 ≤ s ∧ s ≤ 3

def dice_sum (d1 d2 : ℕ) : ℕ := d1 + d2

theorem probability_area_less_than_perimeter :
  (finset.filter (λ s, valid_sum s) (finset.Icc 2 14)).card / 48 = 1 / 16 :=
by
  sorry

end probability_area_less_than_perimeter_l701_701206


namespace segments_form_triangle_l701_701225

/-- Point D on diameter AB of a unit circle such that segments AD, BD, and CD form a triangle. -/
theorem segments_form_triangle (x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 1) :
  (2 - Real.sqrt 5 < x) ∧ (x < Real.sqrt 5 - 2) ↔
  let AD := 1 + x; let BD := 1 - x; let CD := Real.sqrt (1 - x^2)
  in BD^2 + CD^2 > AD^2 :=
sorry

end segments_form_triangle_l701_701225


namespace number_of_terms_added_l701_701544

open Nat

theorem number_of_terms_added (k : ℕ) (h : k > 1) 
    (H : ∑ i in range (2^k - 1), (1 / (i + 1)) < k) : 
    (2^k = (2^(k + 1) - 1) - (2^k - 1) + 1) :=
sorry

end number_of_terms_added_l701_701544


namespace prove_f_odd_range_not_mentioned_prove_inequality_x1_x2_exists_x_gt_0_l701_701324

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x + x / (x^2 + 1)

theorem prove_f_odd :
  ∀ x : ℝ, f (-x) = -f (x) :=
sorry

theorem range_not_mentioned : 
  ¬ (∀ y : ℝ, y ∈ Set.range f → y ∈ (-⨆ -5/2 Set.le' ⨆ 5 / 2)) :=
sorry

theorem prove_inequality_x1_x2 :
  ∀ (x1 x2 : ℝ), (x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ f x1 = f x2) → x1 + x2 > 2 :=
sorry

theorem exists_x_gt_0 :
  ∃ x : ℝ, (x > 0) ∧  ¬ (f x ≥ 5 / 2 * x) :=
sorry

end prove_f_odd_range_not_mentioned_prove_inequality_x1_x2_exists_x_gt_0_l701_701324


namespace weight_loss_comparison_l701_701226

-- Define the conditions
def weight_loss_Barbi : ℝ := 1.5 * 24
def weight_loss_Luca : ℝ := 9 * 15
def weight_loss_Kim : ℝ := (2 * 12) + (3 * 60)

-- Define the combined weight loss of Luca and Kim
def combined_weight_loss_Luca_Kim : ℝ := weight_loss_Luca + weight_loss_Kim

-- Define the difference in weight loss between Luca and Kim combined and Barbi
def weight_loss_difference : ℝ := combined_weight_loss_Luca_Kim - weight_loss_Barbi

-- State the theorem to be proved
theorem weight_loss_comparison : weight_loss_difference = 303 := by
  sorry

end weight_loss_comparison_l701_701226


namespace apex_angle_of_fourth_cone_l701_701119

theorem apex_angle_of_fourth_cone (
  (vertex_A : Point)
  (apex_angle : ℝ) (h_apex : apex_angle = π / 3)
  (touching_cones : ∀ (cone : Cone), cone.vertex = vertex_A ∧ cone.apex_angle = apex_angle)
) : 
  let fourth_cone := Cone vertex_A (π / 3 + 2 * arcsin (1 / sqrt 3)) in
  fourth_cone.apex_angle = π / 3 + 2 * arcsin (1 / sqrt 3) := by
sorry

end apex_angle_of_fourth_cone_l701_701119


namespace four_color_theorem_l701_701191

theorem four_color_theorem : ∀ (G : Type _) [graph G], planar_graph G → (chromatic_number G ≤ 4) :=
sorry

end four_color_theorem_l701_701191


namespace uniqueness_solution_l701_701638

variable (A_t B_t : ℝ → ℝ) -- Functions A_t and B_t
variable (locally_bounded_variation : ∀ t ≥ 0, bounded_var (A_t t) ∧ bounded_var (B_t t)) -- Local bounded variation
variable (right_continuous : ∀ t ≥ 0, right_cont (A_t t) ∧ right_cont (B_t t)) -- Right-continuous functions

def ΔA (t : ℝ) : ℝ := A_t t - A_t t⁻ -- Definition of ∆A

noncomputable def ℰ_A (t : ℝ) : ℝ :=
  exp (A_t t - A_t 0) * ∏ (s : ℝ) in {s | 0 < s ≤ t}, (1 + ΔA s) * exp(-ΔA s)

noncomputable def ℰ_AB (t : ℝ) : ℝ :=
  ℰ_A t * ∫ s in 0..t, dB_t s / ℰ_A s⁻

theorem uniqueness_solution
  (locally_bounded : ∀ t ≥ 0, bounded_var (A_t t) ∧ bounded_var (B_t t))
  (right_cont_lim : ∀ t ≥ 0, rctoc (A_t t) ∧ rctoc (B_t t)) :
  ∃! Z_t : ℝ → ℝ, Z_t = B_t + ∫ s in 0..t, Z_s⁻ * dA_s,
  Z_t = ℰ_AB t + B_0 * ℰ_A t :=
sorry -- Proof not required

end uniqueness_solution_l701_701638


namespace parallel_line_slope_l701_701997

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l701_701997


namespace polar_to_cartesian_l701_701702

theorem polar_to_cartesian (ρ θ : ℝ) (h : ρ = 4 * Real.cos θ) :
  ∃ x y : ℝ, (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧
  (x - 2)^2 + y^2 = 4) :=
sorry

end polar_to_cartesian_l701_701702


namespace Eve_total_running_distance_l701_701651

def Eve_walked_distance := 0.6

def Eve_ran_distance := Eve_walked_distance + 0.1

theorem Eve_total_running_distance : Eve_ran_distance = 0.7 := 
by sorry

end Eve_total_running_distance_l701_701651


namespace example_problem_l701_701701

theorem example_problem (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f(x) + f(x + 2) = 0)
  (h2: ∀ x : ℝ, f(-x) = -f(x))
  (h3 : f(1) = -2) :
  f(5) + f(2) = -2 :=
sorry

end example_problem_l701_701701


namespace speed_of_water_is_10_l701_701597

/-- Define the conditions -/
def swimming_speed_in_still_water : ℝ := 12 -- km/h
def time_to_swim_against_current : ℝ := 4 -- hours
def distance_against_current : ℝ := 8 -- km

/-- Define the effective speed against the current and the proof goal -/
def speed_of_water (v : ℝ) : Prop :=
  (swimming_speed_in_still_water - v) = distance_against_current / time_to_swim_against_current

theorem speed_of_water_is_10 : speed_of_water 10 :=
by
  unfold speed_of_water
  sorry

end speed_of_water_is_10_l701_701597


namespace right_triangle_third_side_product_l701_701539

theorem right_triangle_third_side_product (a b : ℝ) (h₁ : a = 8) (h₂ : b = 15) :
  let hypotenuse := real.sqrt (a^2 + b^2)
  let other_leg := real.sqrt (b^2 - a^2)
  (a < b) →
  (round (hypotenuse * other_leg * 10) / 10) = 215.9 :=
by
  intros
  rw [h₁, h₂]
  have hypo_side : hypotenuse = real.sqrt (8^2 + 15^2), by rw [← h₁, ← h₂]
  have other_leg_side : other_leg = real.sqrt (15^2 - 8^2), by rw [← h₂, ← h₁]
  sorry

end right_triangle_third_side_product_l701_701539


namespace frood_game_least_n_l701_701397

theorem frood_game_least_n (n : ℕ) (h : n > 0) (drop_score : ℕ := n * (n + 1) / 2) (eat_score : ℕ := 15 * n) 
  : drop_score > eat_score ↔ n ≥ 30 :=
by
  sorry

end frood_game_least_n_l701_701397


namespace parallel_line_slope_l701_701999

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l701_701999


namespace proof_problem_l701_701736

variables (x y b z a : ℝ)

def condition1 : Prop := x * y + x^2 = b
def condition2 : Prop := (1 / x^2) - (1 / y^2) = a
def z_def : Prop := z = x + y

theorem proof_problem (x y b z a : ℝ) (h1 : condition1 x y b) (h2 : condition2 x y a) (hz : z_def x y z) : (x + y) ^ 2 = z ^ 2 :=
by {
  sorry
}

end proof_problem_l701_701736


namespace polygon_sides_l701_701703

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 360 + 720)
  (h2 : ∀ n, (n - 2) * 180 = (sum_interior_angles n))
  (h3 : (sum_exterior_angles = 360)) : 
  n = 8 := 
  sorry

end polygon_sides_l701_701703


namespace general_formula_a_n_sum_S_n_l701_701294

def a (n : ℕ) : ℕ :=
  if n = 1 then 4 else n^2

def sqrt_a_sum (n : ℕ) : ℕ :=
  (List.range n).map (λ i, (a (i+1)).sqrt).sum

theorem general_formula_a_n (n : ℕ) :
  sqrt_a_sum n = (n^2 + n + 2) / 2 :=
sorry

def b (n : ℕ) : ℚ :=
  1 / (a n + (a n).sqrt)

def S (n : ℕ) : ℚ :=
  (List.range n).map (λ i, b (i+1)).sum

theorem sum_S_n (n : ℕ) :
  S n = (2 * n - 1) / (3 * (n + 1)) :=
sorry

end general_formula_a_n_sum_S_n_l701_701294


namespace real_solutions_of_equation_l701_701254

theorem real_solutions_of_equation (x : ℝ) :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 12) ↔ (x = 13 ∨ x = -5) :=
by
  sorry

end real_solutions_of_equation_l701_701254


namespace triangle_perpendiculars_are_concurrent_l701_701579

noncomputable def triangle_perpendicular_concurrent
  (A B C A1 B1 C1 : Type*)
  [InnerProductSpace ℝ A]
  [InnerProductSpace ℝ B]
  [InnerProductSpace ℝ C]
  [InnerProductSpace ℝ A1]
  [InnerProductSpace ℝ B1]
  [InnerProductSpace ℝ C1]
  (excircle_touch_A1 : excircle A1 (triangle.point_excircle A B C))
  (excircle_touch_B1 : excircle B1 (triangle.point_excircle B C A))
  (excircle_touch_C1 : excircle C1 (triangle.point_excircle C A B)) :
  Prop := 
  ∃ (P : Type*), 
  perpendicular (line_through A1 B C) P ∧ 
  perpendicular (line_through B1 A C) P ∧ 
  perpendicular (line_through C1 A B) P

-- Skips proof of the concurrent point
theorem triangle_perpendiculars_are_concurrent
  (A B C A1 B1 C1 : Type*)
  [InnerProductSpace ℝ A]
  [InnerProductSpace ℝ B]
  [InnerProductSpace ℝ C]
  [InnerProductSpace ℝ A1]
  [InnerProductSpace ℝ B1]
  [InnerProductSpace ℝ C1]
  (excircle_touch_A1 : excircle A1 (triangle.point_excircle A B C))
  (excircle_touch_B1 : excircle B1 (triangle.point_excircle B C A))
  (excircle_touch_C1 : excircle C1 (triangle.point_excircle C A B)) :
  triangle_perpendicular_concurrent A B C A1 B1 C1 excircle_touch_A1 excircle_touch_B1 excircle_touch_C1 := 
  sorry

end triangle_perpendiculars_are_concurrent_l701_701579


namespace farmer_children_l701_701186

noncomputable def numberOfChildren 
  (totalLeft : ℕ)
  (eachChildCollected : ℕ)
  (eatenApples : ℕ)
  (soldApples : ℕ) : ℕ :=
  let totalApplesEaten := eatenApples * 2
  let initialCollection := eachChildCollected * (totalLeft + totalApplesEaten + soldApples) / eachChildCollected
  initialCollection / eachChildCollected

theorem farmer_children (totalLeft : ℕ) (eachChildCollected : ℕ) (eatenApples : ℕ) (soldApples : ℕ) : 
  totalLeft = 60 → eachChildCollected = 15 → eatenApples = 4 → soldApples = 7 → 
  numberOfChildren totalLeft eachChildCollected eatenApples soldApples = 5 := 
by
  intro h_totalLeft h_eachChildCollected h_eatenApples h_soldApples
  unfold numberOfChildren
  simp
  sorry

end farmer_children_l701_701186


namespace solve_x_from_operation_l701_701642

def operation (a b c d : ℝ) : ℝ := a * c + b * d

theorem solve_x_from_operation :
  ∀ x : ℝ, operation (2 * x) 3 3 (-1) = 3 → x = 1 :=
by
  intros x h
  sorry

end solve_x_from_operation_l701_701642


namespace candidate_percentage_l701_701172

theorem candidate_percentage (P : ℝ) (l : ℝ) (V : ℝ) : 
  l = 5000.000000000007 ∧ 
  V = 25000.000000000007 ∧ 
  V - 2 * (P / 100) * V = l →
  P = 40 :=
by
  sorry

end candidate_percentage_l701_701172


namespace ratio_AM_MC_l701_701425

theorem ratio_AM_MC :
  ∀ (C1 C2 : Circle) (A B C D E F M : Point),
  Concentric C1 C2 ∧
  In_circle A C1 ∧
  Tangent_line (A, B) C2 ∧
  In_circle B C2 ∧
  Ray_intersection A B C C1 ∧
  Midpoint D A B ∧
  Line_through A E F ∧
  In_circle E C2 ∧
  In_circle F C2 ∧
  Intersecting_perpendicular_bisectors D E C F M ∧
  On_line M A B → 
  AM/MC = 5/3 :=
sorry

end ratio_AM_MC_l701_701425


namespace cultivated_land_percentage_l701_701884

theorem cultivated_land_percentage : 
  let water : ℚ := 7 / 10
  let land : ℚ := 3 / 10
  let deserts_or_ice : ℚ := 2 / 5
  let pastures_forests_mountains : ℚ := 1 / 3
  ∃ cultivated_land_fraction : ℚ, 
    cultivated_land_fraction = land * (1 - deserts_or_ice - pastures_forests_mountains) ∧ 
    (cultivated_land_fraction * 100) = 8 := 
by
  sorry

end cultivated_land_percentage_l701_701884


namespace farmer_children_l701_701184

noncomputable def numberOfChildren 
  (totalLeft : ℕ)
  (eachChildCollected : ℕ)
  (eatenApples : ℕ)
  (soldApples : ℕ) : ℕ :=
  let totalApplesEaten := eatenApples * 2
  let initialCollection := eachChildCollected * (totalLeft + totalApplesEaten + soldApples) / eachChildCollected
  initialCollection / eachChildCollected

theorem farmer_children (totalLeft : ℕ) (eachChildCollected : ℕ) (eatenApples : ℕ) (soldApples : ℕ) : 
  totalLeft = 60 → eachChildCollected = 15 → eatenApples = 4 → soldApples = 7 → 
  numberOfChildren totalLeft eachChildCollected eatenApples soldApples = 5 := 
by
  intro h_totalLeft h_eachChildCollected h_eatenApples h_soldApples
  unfold numberOfChildren
  simp
  sorry

end farmer_children_l701_701184


namespace physics_kit_prices_l701_701057

theorem physics_kit_prices :
  ∃ (price_A price_B : ℝ), price_A = 180 ∧ price_B = 150 ∧
    price_A = 1.2 * price_B ∧
    9900 / price_A = 7500 / price_B + 5 :=
by
  use 180, 150
  sorry

end physics_kit_prices_l701_701057


namespace matrix_sum_of_rank_one_matrices_identity_matrix_not_sum_of_less_than_four_rank_one_matrices_l701_701153

-- Part (a)
theorem matrix_sum_of_rank_one_matrices (A : Matrix (Fin 4) (Fin 4) ℂ) : 
  ∃ B1 B2 B3 B4 : Matrix (Fin 4) (Fin 4) ℂ, (∀ i, Matrix.rank (nth i [B1, B2, B3, B4]) = 1) ∧ A = B1 + B2 + B3 + B4 := 
sorry

-- Part (b)
theorem identity_matrix_not_sum_of_less_than_four_rank_one_matrices : 
  ¬ ∃ (s : ℕ) (B : Fin s → Matrix (Fin 4) (Fin 4) ℂ), s < 4 ∧ (∀ i, Matrix.rank (B i) = 1) ∧ Matrix.identity (Fin 4) = (Finset.univ.sum fun i : Fin s => B i) := 
sorry

end matrix_sum_of_rank_one_matrices_identity_matrix_not_sum_of_less_than_four_rank_one_matrices_l701_701153


namespace smallest_positive_period_of_sin_2x_l701_701519

def period_of_sinusoidal_function (ω : ℝ) : ℝ :=
  2 * Real.pi / ω

theorem smallest_positive_period_of_sin_2x : period_of_sinusoidal_function 2 = Real.pi :=
by
  sorry

end smallest_positive_period_of_sin_2x_l701_701519


namespace points_satisfying_clubsuit_l701_701238

def clubsuit (a b : ℝ) : ℝ := a^2 * b + a * b^2

theorem points_satisfying_clubsuit (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = 0 ∨ y = 0 ∨ x + y = 0) :=
by
  sorry

end points_satisfying_clubsuit_l701_701238


namespace magnitude_of_vector_sum_l701_701284

-- Definition of the vectors a and b
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (2, 1)

-- Definition of the given vector addition and scalar multiplication
def vector_sum : ℝ × ℝ := (a.1 + 3 * b.1, a.2 + 3 * b.2)

-- Prove that the magnitude of the vector sum is √58
theorem magnitude_of_vector_sum : 
  real.sqrt (vector_sum.1^2 + vector_sum.2^2) = real.sqrt 58 := sorry

end magnitude_of_vector_sum_l701_701284


namespace javiers_household_legs_l701_701013

-- Definitions given the problem conditions
def humans : ℕ := 6
def human_legs : ℕ := 2

def dogs : ℕ := 2
def dog_legs : ℕ := 4

def cats : ℕ := 1
def cat_legs : ℕ := 4

def parrots : ℕ := 1
def parrot_legs : ℕ := 2

def lizards : ℕ := 1
def lizard_legs : ℕ := 4

def stool_legs : ℕ := 3
def table_legs : ℕ := 4
def cabinet_legs : ℕ := 6

-- Problem statement
theorem javiers_household_legs :
  (humans * human_legs) + (dogs * dog_legs) + (cats * cat_legs) + (parrots * parrot_legs) +
  (lizards * lizard_legs) + stool_legs + table_legs + cabinet_legs = 43 := by
  -- We leave the proof as an exercise for the reader
  sorry

end javiers_household_legs_l701_701013


namespace quadratic_no_real_solution_l701_701282

theorem quadratic_no_real_solution (a : ℝ) : (∀ x : ℝ, x^2 - x + a ≠ 0) → a > 1 / 4 :=
by
  intro h
  sorry

end quadratic_no_real_solution_l701_701282


namespace arithmetic_sequence_ninth_term_l701_701887

theorem arithmetic_sequence_ninth_term :
  ∃ a d : ℤ, (a + 2 * d = 23) ∧ (a + 5 * d = 29) ∧ (a + 8 * d = 35) :=
by
  sorry

end arithmetic_sequence_ninth_term_l701_701887


namespace barneys_grocery_store_items_left_l701_701227

theorem barneys_grocery_store_items_left 
    (ordered_items : ℕ) 
    (sold_items : ℕ) 
    (storeroom_items : ℕ) 
    (damaged_percentage : ℝ)
    (h1 : ordered_items = 4458) 
    (h2 : sold_items = 1561) 
    (h3 : storeroom_items = 575) 
    (h4 : damaged_percentage = 5/100) : 
    ordered_items - (sold_items + ⌊damaged_percentage * ordered_items⌋) + storeroom_items = 3250 :=
by
    sorry

end barneys_grocery_store_items_left_l701_701227


namespace determine_angle_YOZ_l701_701398

-- Define triangle and angles
variables (X Y Z O : Type) [inhabited X] [inhabited Y] [inhabited Z] [inhabited O]
variables (triangle_XYZ: X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z)
variables (incircle_O: ∀ {P : Type}, P = X ∨ P = Y ∨ P = Z → P ≠ O)

-- Define given angles
def angle_XYZ := 75
def angle_YXZ := 45

-- Define angles relation for sum of triangle angles
def sum_of_triangle_angles := angle_XYZ + angle_YXZ + angle_ZYX = 180

-- Define angle bisector
def angle_bisector := angle_YOZ = angle_ZYX / 2

-- Prove the required angle
theorem determine_angle_YOZ :
  sum_of_triangle_angles →
  ∃ (angle_ZYX : ℕ), angle_ZYX = 180 - angle_XYZ - angle_YXZ →
  ∃ (angle_YOZ : ℕ), angle_YOZ = angle_ZYX / 2 →
  angle_YOZ = 30 :=
by
  sorry

end determine_angle_YOZ_l701_701398


namespace slope_of_parallel_line_l701_701974

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l701_701974


namespace circumradius_of_hyperbola_triangle_l701_701422

theorem circumradius_of_hyperbola_triangle (x₀ y₀ : ℝ) (h : (x₀ / 2) ^ 2 - (y₀ / 6) ^ 2 = 1) 
  (r₁ r₂ : ℝ) (h₁ : r₁ - r₂ = 4) (h₂ : 2 * (sqrt (4 + 12) = 8)) : 
  (let R := (sqrt (4 * 2 ^ 2 - 2 * (4 ^ 2)) / (x₀ + y₀)) / 2 in R = 5) :=
sorry

end circumradius_of_hyperbola_triangle_l701_701422


namespace calculate_brick_height_cm_l701_701174

noncomputable def wall_length_cm : ℕ := 1000  -- 10 m converted to cm
noncomputable def wall_width_cm : ℕ := 800   -- 8 m converted to cm
noncomputable def wall_height_cm : ℕ := 2450 -- 24.5 m converted to cm

noncomputable def wall_volume_cm3 : ℕ := wall_length_cm * wall_width_cm * wall_height_cm

noncomputable def brick_length_cm : ℕ := 20
noncomputable def brick_width_cm : ℕ := 10
noncomputable def number_of_bricks : ℕ := 12250

noncomputable def brick_area_cm2 : ℕ := brick_length_cm * brick_width_cm

theorem calculate_brick_height_cm (h : ℕ) : brick_area_cm2 * h * number_of_bricks = wall_volume_cm3 → 
  h = wall_volume_cm3 / (brick_area_cm2 * number_of_bricks) := by
  sorry

end calculate_brick_height_cm_l701_701174


namespace geometric_sequence_product_l701_701345

variable {a b c : ℝ}

theorem geometric_sequence_product (h : ∃ r : ℝ, r ≠ 0 ∧ -4 = c * r ∧ c = b * r ∧ b = a * r ∧ a = -1 * r) (hb : b < 0) : a * b * c = -8 :=
by
  sorry

end geometric_sequence_product_l701_701345


namespace scientific_notation_of_20_ns_l701_701385

def nanosecond_to_scientific_notation (n: ℕ) : ℝ := n * 10⁻⁹

theorem scientific_notation_of_20_ns :
  nanosecond_to_scientific_notation 20 = 2 * 10⁻⁸ := 
by {
  sorry
}

end scientific_notation_of_20_ns_l701_701385


namespace min_value_is_neg_500000_l701_701034

noncomputable def min_expression_value (a b : ℝ) : ℝ :=
  let term1 := a + 1/b
  let term2 := b + 1/a
  (term1 * (term1 - 1000) + term2 * (term2 - 1000))

theorem min_value_is_neg_500000 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_expression_value a b ≥ -500000 :=
sorry

end min_value_is_neg_500000_l701_701034


namespace smallest_non_zero_natural_sum_l701_701554

theorem smallest_non_zero_natural_sum :
  ∃ n : ℕ, n ≠ 0 ∧ (∃ x1 : ℕ, n = 2010 * x1) ∧ (∃ x2 : ℕ, n = 2012 * x2) ∧ (∃ x3 : ℕ, n = 2013 * x3) ∧ n = 6036 :=
begin
  sorry
end

end smallest_non_zero_natural_sum_l701_701554


namespace problem_b_lt_a_lt_c_l701_701793

theorem problem_b_lt_a_lt_c (a b c : ℝ)
  (h1 : 1.001 * Real.exp a = Real.exp 1.001)
  (h2 : b - Real.sqrt (1000 / 1001) = 1.001 - Real.sqrt 1.001)
  (h3 : c = 1.001) : b < a ∧ a < c := by
  sorry

end problem_b_lt_a_lt_c_l701_701793


namespace symmetric_point_x_axis_l701_701299

theorem symmetric_point_x_axis (P Q : ℝ × ℝ)
  (hP : P = (-2, 1))
  (hQ : Q = (-2, -1))
  (h_symmetry : Q = (P.1, -P.2)) :
  Q = (-2, -1) :=
by
  rw [←hP, h_symmetry]
  exact hQ

end symmetric_point_x_axis_l701_701299


namespace calculate_product_l701_701230

theorem calculate_product : (Real.cbrt 64) * (Real.cbrt 27) * (Real.sqrt 16) = 48 := 
  sorry

end calculate_product_l701_701230


namespace part1_solution_part2_solution_part3_solution_l701_701580

-- Part (1): Prove the solution of the system of equations 
theorem part1_solution (x y : ℝ) (h1 : x - y - 1 = 0) (h2 : 4 * (x - y) - y = 5) : 
  x = 0 ∧ y = -1 := 
sorry

-- Part (2): Prove the solution of the system of equations 
theorem part2_solution (x y : ℝ) (h1 : 2 * x - 3 * y - 2 = 0) 
  (h2 : (2 * x - 3 * y + 5) / 7 + 2 * y = 9) : 
  x = 7 ∧ y = 4 := 
sorry

-- Part (3): Prove the range of the parameter m
theorem part3_solution (m : ℕ) (h1 : 2 * (2 : ℝ) * x + y = (-3 : ℝ) * ↑m + 2) 
  (h2 : x + 2 * y = 7) (h3 : x + y > -5 / 6) : 
  m = 1 ∨ m = 2 ∨ m = 3 :=
sorry

end part1_solution_part2_solution_part3_solution_l701_701580


namespace total_area_of_removed_triangles_l701_701606

theorem total_area_of_removed_triangles : 
  ∀ (side_length_of_square : ℝ) (hypotenuse_length_of_triangle : ℝ),
  side_length_of_square = 20 →
  hypotenuse_length_of_triangle = 10 →
  4 * (1/2 * (hypotenuse_length_of_triangle^2 / 2)) = 100 :=
by
  intros side_length_of_square hypotenuse_length_of_triangle h_side_length h_hypotenuse_length
  -- Proof would go here, but we add "sorry" to complete the statement
  sorry

end total_area_of_removed_triangles_l701_701606


namespace intersection_S_T_l701_701689

open Finset

-- Define the sets S and T
def S := {1, 4, 5} : Finset ℕ
def T := {2, 3, 4} : Finset ℕ

-- Prove that the intersection of S and T is {4}
theorem intersection_S_T : S ∩ T = {4} :=
by sorry

end intersection_S_T_l701_701689


namespace right_triangle_third_side_length_l701_701369

theorem right_triangle_third_side_length (a b : ℝ) (c : ℝ) (h_right_triangle: a = 3 ∧ b = 4 ∧ c^2 = 3^2 + 4^2 ∨
 b = 3 ∧ c = 4 ∧ a^2 + 3^2 = 4^2) : 
 a = 5 ∨ a = real.sqrt 7 :=
sorry

end right_triangle_third_side_length_l701_701369


namespace remainder_of_prime_powers_l701_701308

theorem remainder_of_prime_powers (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  (p^(q-1) + q^(p-1)) % (p * q) = 1 := 
sorry

end remainder_of_prime_powers_l701_701308


namespace exists_equilateral_DEF_l701_701614

-- Definitions of the concepts.
variable (ABC : Triangle)
variable [AcuteAngledTriangle ABC]
variable (P : Point)
variable [InsideCircumcircle P ABC]

theorem exists_equilateral_DEF :
  ∃ P : Point, (InsideCircumcircle P ABC) ∧ (EquilateralTriangle (DEF P ABC)) := 
sorry

end exists_equilateral_DEF_l701_701614


namespace slope_of_parallel_line_l701_701963

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l701_701963


namespace TriangleInequality_of_tangent_l701_701280

-- Definition of the triangle with vertices A, B, C
variables (A B C : Point)

-- Centroid G of triangle ABC
def G := centroid A B C

-- Line AC is tangent to the circle passing through points A, B, G
def tangent_to_circle (AC circle_ABG : Line) : Prop :=
  is_tangent AC circle_ABG

-- Given constraints
axiom tangent_AC_circle_ABG : tangent_to_circle (line_through A C) (circle_through B G)

-- Proof requirement
theorem TriangleInequality_of_tangent :
  AB + BC \leq 2 * AC :=
sorry

end TriangleInequality_of_tangent_l701_701280


namespace ellipse_and_isosceles_triangle_l701_701317

-- Define the problem statement
theorem ellipse_and_isosceles_triangle (a b : ℝ) (h1 : a > b) (h2 : b > 0) (e : ℝ) 
    (h3 : e = sqrt 3 / 2) (P A B Q C D E : ℝ × ℝ) 
    (hP : P = (-2, 1)) (hA : A = (-2, -1)) (hB : B = (2, 1)) (hQ : Q = (2, -1)) 
    (hE : E = -C) (hEllipse : ∀ (x y : ℝ), 
                x^2 / a^2 + y^2 / b^2 = 1 <-> (P = (-2, 1) → (a = 2 * sqrt 2 ∧ b = sqrt 2)))
    (hParallelAB : ∀ (l : ℝ), 
                (l = λ x : ℝ, 1/2 * x + t) → 
                (C ∈ set_of (λ p : ℝ × ℝ, x^2 / 8 + y^2 / 2 = 1)) ∧ 
                (D ≠ P ∧ D ≠ Q) ∧ 
                (C ≠ E))
-- Prove the final result
:
  (C_1_eq: a = 2 * sqrt 2 ∧ b = sqrt 2 → ∃ (C D : ℝ × ℝ) (t : ℝ), (t ≠ 0 ∧ -2 < t < 2) ∧ 
            (x^2 / 8 + y^2 / 2 = 1) ∧ 
            (C y = 1/2 * x + t) ∧ 
            (D y = 1/2 * x + t) ∧ 
            PD_parallel_to_PE ∧ 
            ((C_1_eq_eq: x1 + x2 = -2 * t) ∧ 
            C_1_mult_eq: x1 * x2 = 2 * t^2 - 4) ∧ 
            (y1 = 1/2 * x1 + t) ∧ 
            (y2 = 1/2 * x2 + t) ∧ 
                (iso_triangle PD PE yaxis ∧
                (let k1 := slope (PD) in let k2 := slope (PE)
                     (k1_eq_k2_inv : k1 + k2 = 0))
                ))) sorry

end ellipse_and_isosceles_triangle_l701_701317


namespace probability_in_triangle_OBC_l701_701429

open_locale classical
noncomputable theory

variables {V : Type*} [inner_product_space ℝ V] 

def in_triangle (A B C O : V) : Prop :=
  ∃ (u v : ℝ), 0 ≤ u ∧ 0 ≤ v ∧ u + v ≤ 1 ∧ O = u • A + v • B + (1 - u - v) • C

def satisfies_condition (A B C O : V) : Prop :=
  4 • O - A + (O - B) + (O - C) = (0 : V)

theorem probability_in_triangle_OBC (A B C O : V)
  (hA : in_triangle A B C O)
  (hC : satisfies_condition A B C O) :
  ∃ (P : ℝ), P = 2/3 := sorry

end probability_in_triangle_OBC_l701_701429


namespace min_triangles_cover_issue_l701_701551

noncomputable def min_equilateral_triangles_needed : Real :=
  let side_length_large_triangle : Real := 12
  let side_length_square : Real := 4
  let area_large : Real := (Real.sqrt 3 * side_length_large_triangle^2) / 4
  let area_square : Real := side_length_square^2
  let total_area : Real := area_large + area_square
  let area_small_triangle : Real := (Real.sqrt 3) / 4
  let num_triangles := total_area / area_small_triangle
  Real.ceil(num_triangles)

theorem min_triangles_cover_issue :
  min_equilateral_triangles_needed = 145 * Real.sqrt 3 + 64 := sorry

end min_triangles_cover_issue_l701_701551


namespace molecular_weight_of_7_moles_AlPO4_is_correct_l701_701130

def atomic_weight_Al : Float := 26.98
def atomic_weight_P : Float := 30.97
def atomic_weight_O : Float := 16.00

def molecular_weight_AlPO4 : Float :=
  (1 * atomic_weight_Al) + (1 * atomic_weight_P) + (4 * atomic_weight_O)

noncomputable def weight_of_7_moles_AlPO4 : Float :=
  7 * molecular_weight_AlPO4

theorem molecular_weight_of_7_moles_AlPO4_is_correct :
  weight_of_7_moles_AlPO4 = 853.65 := by
  -- computation goes here
  sorry

end molecular_weight_of_7_moles_AlPO4_is_correct_l701_701130


namespace parabola_directrix_eq_l701_701478

theorem parabola_directrix_eq (x y : ℝ) : x^2 + 12 * y = 0 → y = 3 := 
by sorry

end parabola_directrix_eq_l701_701478


namespace necessary_sufficient_condition_for_shaded_areas_l701_701761

def equality_of_shaded_areas (r : ℝ) (φ : ℝ) : Prop :=
  ∃ (O : Point) (B C E D : Point), 
    φ > 0 ∧ φ < π / 2 ∧
    is_center O ∧
    on_line_segment O C D OAE ∧
    tangent OB Circle_at B ∧
    shaded_area_equality O B C E D r

axiom tangent (ob : Line) (circle_at : Point) : Prop
axiom shaded_area_equality (O B C E D: Point) (r : ℝ) : Prop

theorem necessary_sufficient_condition_for_shaded_areas (r : ℝ) (φ : ℝ) :
  φ > 0 ∧ φ < π / 2 → equality_of_shaded_areas r φ ↔ tan φ = 2 * φ :=
by 
  sorry

end necessary_sufficient_condition_for_shaded_areas_l701_701761


namespace forces_magnitude_l701_701537

-- Definitions and assumptions
def x : ℝ := 2  -- Magnitude of the smaller force
def theta : ℝ := 30 * (π / 180)  -- Angle between forces in radians (30 degrees)
def F1 : ℝ := x  -- The smaller force
def F2 : ℝ := 7 * sqrt 3 * x  -- The larger force
def R : ℝ := 24 + x  -- The resultant force

-- Statement to prove
theorem forces_magnitude (h1 : x = 2) (h2 : theta = 30 * (π / 180)) (h3 : F1 = x) (h4 : F2 = 7 * sqrt 3 * x) (h5 : R = 24 + x) :
  F1 = 2 ∧ R = 26 :=
by
  sorry

end forces_magnitude_l701_701537


namespace hyperbola_eccentricity_l701_701834

variable {F₁ F₂ P : Type}
variable {C : Type}
variable (symmetric_point : P → F₁ → Prop)
variable (eccentricity : C → ℝ)
variable [Hyperbola C F₁ F₂ P]

theorem hyperbola_eccentricity (C : Hyperbola) (F₁ F₂ : C → AffinePlane.Point) (P : AffinePlane.Point) : 
  symmetric_point P F₁ → eccentricity C = Real.sqrt 5 := 
by
  sorry

end hyperbola_eccentricity_l701_701834


namespace symmetric_point_x_axis_l701_701298

theorem symmetric_point_x_axis (P Q : ℝ × ℝ)
  (hP : P = (-2, 1))
  (hQ : Q = (-2, -1))
  (h_symmetry : Q = (P.1, -P.2)) :
  Q = (-2, -1) :=
by
  rw [←hP, h_symmetry]
  exact hQ

end symmetric_point_x_axis_l701_701298


namespace slope_of_parallel_line_l701_701923

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l701_701923


namespace roller_coaster_seating_l701_701116

theorem roller_coaster_seating :
  ∀ (total_people runs cars : ℕ),
  total_people = 84 →
  runs = 6 →
  cars = 7 →
  (total_people / runs) / cars = 2 :=
by
  intros total_people runs cars h1 h2 h3
  rw [h1, h2, h3]
  exact Nat.div_eq_of_eq_mul (by norm_num) (by norm_num) (by norm_num)

end roller_coaster_seating_l701_701116


namespace max_value_piecewise_function_l701_701096

def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + 3
  else if x ≤ 1 then x + 3
  else -x + 5

theorem max_value_piecewise_function : ∃ y, y = 4 ∧ ∀ x, piecewise_function x ≤ y := 
by
  sorry

end max_value_piecewise_function_l701_701096


namespace altitude_equals_median_implies_angle_MBC_30_l701_701399

/-- 
  In triangle ABC, if the altitude AH equals the median BM, 
  then the angle MBC is 30 degrees.
-/
theorem altitude_equals_median_implies_angle_MBC_30 
  (A B C M H : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space M] [metric_space H]
  (triangle_ABC : triangle A B C)
  (altitude_AH : altitude A H C) (median_BM : median B M A C)
  (h_eq : distance_altitude_AH = distance_median_BM) : angle M B C = 30 :=
sorry

end altitude_equals_median_implies_angle_MBC_30_l701_701399


namespace circle_passing_points_l701_701852

theorem circle_passing_points (x y : ℝ) :
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 - 4*x - 6*y = 0 :=
by
  intros h
  cases h
  case inl h₁ => 
    rw [h₁.1, h₁.2]
    ring
  case inr h₁ =>
    cases h₁
    case inl h₂ => 
      rw [h₂.1, h₂.2]
      ring
    case inr h₂ =>
      rw [h₂.1, h₂.2]
      ring

end circle_passing_points_l701_701852


namespace prove_inequality_l701_701583

variable (f : ℝ → ℝ)

def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def isMonotonicOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

theorem prove_inequality
  (h1 : isEvenFunction f)
  (h2 : isMonotonicOnInterval f 0 5)
  (h3 : f (-3) < f 1) :
  f 0 > f 1 :=
sorry

end prove_inequality_l701_701583


namespace area_of_BMND_l701_701381

-- Definitions for points and square side length
def A := (0, 0)
def B := (2, 0)
def C := (2, 2)
def D := (0, 2)
def M := (2, 1)  -- Midpoint of BC
def N := (1, 2)  -- Midpoint of CD

-- The area of the shaded region BMND
theorem area_of_BMND :
  let side := 2
  let pointA := A
  let pointB := B
  let pointC := C
  let pointD := D
  let pointM := M
  let pointN := N
  area_of_quadrilateral pointB pointM pointN pointD = 3 / 2 :=
by
  sorry

end area_of_BMND_l701_701381


namespace money_on_tuesday_l701_701809

-- Defining the problem conditions
variables (T : ℤ)
-- Condition for Wednesday
def money_wednesday := 5 * T
-- Condition for Thursday
def money_thursday := money_wednesday T + 9
-- Given equation
axiom condition : money_thursday T = T + 41

-- Prove the amount of money given on Tuesday
theorem money_on_tuesday : T = 8 := 
by {
  -- substituting money_wednesday and money_thursday
  calc
    money_thursday T
       = (5 * T) + 9      : rfl
   ... = T + 41           : condition
   ... ⟹ 5 * T + 9 = T + 41 
   ... ⟹ 4 * T = 32
   ... ⟹ T = 8
}

end money_on_tuesday_l701_701809


namespace circle_passing_points_l701_701854

theorem circle_passing_points (x y : ℝ) :
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 - 4*x - 6*y = 0 :=
by
  intros h
  cases h
  case inl h₁ => 
    rw [h₁.1, h₁.2]
    ring
  case inr h₁ =>
    cases h₁
    case inl h₂ => 
      rw [h₂.1, h₂.2]
      ring
    case inr h₂ =>
      rw [h₂.1, h₂.2]
      ring

end circle_passing_points_l701_701854


namespace average_speed_correct_l701_701612

def swim_distance : ℝ := 1.5
def bike_distance : ℝ := 3
def run_distance : ℝ := 2

def swim_speed : ℝ := 2
def bike_speed : ℝ := 25
def run_speed : ℝ := 8

def total_distance : ℝ := swim_distance + bike_distance + run_distance

def swim_time : ℝ := swim_distance / swim_speed
def bike_time : ℝ := bike_distance / bike_speed
def run_time : ℝ := run_distance / run_speed

def total_time : ℝ := swim_time + bike_time + run_time

def average_speed : ℝ := total_distance / total_time

theorem average_speed_correct : abs (average_speed - 5.80) < 0.01 → ∃ n : ℕ, n = 6 :=
by
  sorry

end average_speed_correct_l701_701612


namespace candidate_function_is_odd_and_increasing_l701_701221

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_function (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

def candidate_function (x : ℝ) : ℝ := x * |x|

theorem candidate_function_is_odd_and_increasing :
  is_odd_function candidate_function ∧ is_increasing_function candidate_function :=
by
  sorry

end candidate_function_is_odd_and_increasing_l701_701221


namespace cos_alpha_plus_two_pi_over_three_l701_701731

theorem cos_alpha_plus_two_pi_over_three (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.cos (α + 2 * π / 3) = -1 / 3 :=
by
  sorry

end cos_alpha_plus_two_pi_over_three_l701_701731


namespace parallel_line_slope_l701_701977

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l701_701977


namespace range_of_a_l701_701047

noncomputable def P (X : ℕ → ℝ) (i : ℕ) : ℝ := i / 10

theorem range_of_a (a : ℝ) :
  (P (λ i, P i) 1 + P (λ i, P i) 2 + P (λ i, P i) 3 = 3/5) →
  3 < a ∧ a ≤ 4 :=
by
  assume h : P (λ i, P i) 1 + P (λ i, P i) 2 + P (λ i, P i) 3 = 3/5
  sorry

end range_of_a_l701_701047


namespace strength_coefficients_not_all_positive_strength_coefficients_not_all_negative_l701_701059

section ChessTournament
variables {T : Type} [Fintype T] -- T is the type of participants and there are finitely many participants
variables (P : T → ℝ) -- P represents the points of each participant
variables (defeated_by : T → set T) -- represents the set of participants defeated by a participant
variables (defeater_of : T → set T) -- represents the set of participants who defeated a participant

-- Definition of the strength coefficient
def strength_coefficient (A : T) : ℝ :=
  (∑ (x in defeated_by A), P x) - (∑ (x in defeater_of A), P x)

-- Part (a): Prove that it is impossible for all participants to have positive strength coefficients.
theorem strength_coefficients_not_all_positive :
  ¬(∀ A : T, strength_coefficient P defeated_by defeater_of A > 0) :=
sorry

-- Part (b): Prove that it is impossible for all participants to have negative strength coefficients.
theorem strength_coefficients_not_all_negative :
  ¬(∀ A : T, strength_coefficient P defeated_by defeater_of A < 0) :=
sorry

end ChessTournament

end strength_coefficients_not_all_positive_strength_coefficients_not_all_negative_l701_701059


namespace x_solves_quadratic_and_sum_is_75_l701_701084

theorem x_solves_quadratic_and_sum_is_75
  (x a b : ℕ) (h : x^2 + 10 * x = 45) (hx_pos : 0 < x) (hx_form : x = Nat.sqrt a - b) 
  (ha_pos : 0 < a) (hb_pos : 0 < b)
  : a + b = 75 := 
sorry

end x_solves_quadratic_and_sum_is_75_l701_701084


namespace vidya_mother_age_multiple_l701_701546

variable (Vidya_age : ℕ) (Mother_age : ℕ) (additional_years : ℕ)

def multiple_of_age (vidya_age mother_age additional_years : ℕ) : ℕ :=
  (mother_age - additional_years) / vidya_age

theorem vidya_mother_age_multiple (h : Vidya_age = 13) (h1 : Mother_age = 44) (h2 : additional_years = 5) :
  multiple_of_age Vidya_age Mother_age additional_years = 3 :=
  by
    subst h
    subst h1
    subst h2
    unfold multiple_of_age
    have : (44 - 5) / 13 = 3 := by norm_num
    rw this
    rfl

end vidya_mother_age_multiple_l701_701546


namespace ratio_x_y_l701_701743

theorem ratio_x_y (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : x / y = 11 / 6 :=
by
  sorry

end ratio_x_y_l701_701743


namespace avg_english_score_of_class_l701_701114

-- Definitions
def total_students : ℕ := 30
def avg_score_26_students : ℕ := 82
def score_student1 : ℕ := 90
def score_student2 : ℕ := 85
def score_student3 : ℕ := 88
def score_student4 : ℕ := 80

-- Proof statement
theorem avg_english_score_of_class :
  let total_score_26_students := 26 * avg_score_26_students in
  let total_score_4_students := score_student1 + score_student2 + score_student3 + score_student4 in
  let total_score_30_students := total_score_26_students + total_score_4_students in
  (total_score_30_students : ℚ) / total_students = 82.5 :=
by
  let total_score_26_students := 26 * avg_score_26_students
  let total_score_4_students := score_student1 + score_student2 + score_student3 + score_student4
  let total_score_30_students := total_score_26_students + total_score_4_students
  norm_num
  -- sorry to skip the proof
  sorry

end avg_english_score_of_class_l701_701114


namespace probability_no_defective_pencils_l701_701155

theorem probability_no_defective_pencils :
  let total_pencils := 6
  let defective_pencils := 2
  let pencils_chosen := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_ways := Nat.choose total_pencils pencils_chosen
  let non_defective_ways := Nat.choose non_defective_pencils pencils_chosen
  (non_defective_ways / total_ways : ℚ) = 1 / 5 :=
by
  sorry

end probability_no_defective_pencils_l701_701155


namespace complex_numbers_are_real_l701_701826

theorem complex_numbers_are_real
  (a b c : ℂ)
  (h1 : (a + b) * (a + c) = b)
  (h2 : (b + c) * (b + a) = c)
  (h3 : (c + a) * (c + b) = a) : 
  a.im = 0 ∧ b.im = 0 ∧ c.im = 0 :=
sorry

end complex_numbers_are_real_l701_701826


namespace safe_occupancy_after_ventilation_l701_701800

noncomputable def formaldehyde_concentration (x : ℕ) (a k : ℝ) : ℝ :=
  0.48 - 0.1 * (Real.log (k * (x^2 + 2 * x + 1)) / Real.log a)

theorem safe_occupancy_after_ventilation (a k : ℝ) (h_k_pos : k > 0) :
  (Real.log (k * 9) / Real.log a = 2) →
  (Real.log (k * 81) / Real.log a = 3) →
  ∀ x:ℕ, 1 ≤ x ∧ x ≤ 50 →
  (formal_dehyde_concentration x a k ≤ 0.08) →
  x ≥ 26 :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end safe_occupancy_after_ventilation_l701_701800


namespace Tyler_CDs_after_giveaway_and_purchase_l701_701543

theorem Tyler_CDs_after_giveaway_and_purchase :
  (∃ cds_initial cds_giveaway_fraction cds_bought cds_final, 
     cds_initial = 21 ∧ 
     cds_giveaway_fraction = 1 / 3 ∧ 
     cds_bought = 8 ∧ 
     cds_final = cds_initial - (cds_initial * cds_giveaway_fraction) + cds_bought ∧
     cds_final = 22) := 
sorry

end Tyler_CDs_after_giveaway_and_purchase_l701_701543


namespace total_chairs_agreed_proof_l701_701635

/-
Conditions:
- Carey moved 28 chairs
- Pat moved 29 chairs
- They have 17 chairs left to move
Question:
- How many chairs did they agree to move in total?
Proof Problem:
- Prove that the total number of chairs they agreed to move is equal to 74.
-/

def carey_chairs : ℕ := 28
def pat_chairs : ℕ := 29
def chairs_left : ℕ := 17
def total_chairs_agreed : ℕ := carey_chairs + pat_chairs + chairs_left

theorem total_chairs_agreed_proof : total_chairs_agreed = 74 := 
by
  sorry

end total_chairs_agreed_proof_l701_701635


namespace club_committee_probability_l701_701180

noncomputable def probability_at_least_two_boys_and_two_girls (total_members boys girls committee_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_members committee_size
  let ways_fewer_than_two_boys := (Nat.choose girls committee_size) + (boys * Nat.choose girls (committee_size - 1))
  let ways_fewer_than_two_girls := (Nat.choose boys committee_size) + (girls * Nat.choose boys (committee_size - 1))
  let ways_invalid := ways_fewer_than_two_boys + ways_fewer_than_two_girls
  (total_ways - ways_invalid) / total_ways

theorem club_committee_probability :
  probability_at_least_two_boys_and_two_girls 30 12 18 6 = 457215 / 593775 :=
by
  sorry

end club_committee_probability_l701_701180


namespace part_i_part_ii_l701_701287

-- Define f(x)
noncomputable def f (x : ℝ) : ℝ := sin x * cos x - cos^2 (x + (π / 4))

-- The first goal of checking the interval of monotonically increasing:
theorem part_i (k : ℤ) :
  ∀ x, (- π / 4 + k * π) ≤ x ∧ x ≤ (π / 4 + k * π) → (∀ y: ℝ, (-π / 4 + k*π) ≤ y ∧ y ≤ x → f(y) ≤ f(x)) :=
sorry

-- Define the area function and conditions for triangle
noncomputable def area (b c : ℝ) (A : ℝ) : ℝ := (1 / 2) * b * c * sin A

-- The second goal of checking the maximum area:
theorem part_ii (a b c : ℝ) (A : ℝ) (hA : f(A / 2) = 0) (ha : a = 1) : area b c A ≤ (2 + sqrt 3) / 4 :=
sorry

end part_i_part_ii_l701_701287


namespace circle_equation_exists_l701_701859

theorem circle_equation_exists :
  ∃ D E F : ℝ, (∀ (x y : ℝ), (x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
                      (x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
                      (x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
                      (D = -4) ∧ (E = -6) ∧ (F = 0) :=
by
  use [-4, -6, 0]
  intro x y
  split
  { intros hx hy, simp [hx, hy] }
  split
  { intros hx hy, simp [hx, hy], linarith }
  { intros hx hy, simp [hx, hy], linarith }
  sorry

end circle_equation_exists_l701_701859


namespace no_valid_3_digit_number_from_2_3_5_6_9_l701_701338

theorem no_valid_3_digit_number_from_2_3_5_6_9 :
  ¬ ∃ n, 
  (∃ (d1 d2 d3 : ℕ), 
    n = d1 * 100 + d2 * 10 + d3 ∧ 
    d1 ∈ {2, 3, 5, 6, 9} ∧ 
    d2 ∈ {2, 3, 5, 6, 9} ∧ 
    d3 ∈ {2, 3, 5, 6, 9} ∧ 
    d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
    n % 5 = 0 ∧ 
    (d1 + d2 + d3) % 3 = 0) := 
begin
  sorry
end

end no_valid_3_digit_number_from_2_3_5_6_9_l701_701338


namespace central_angle_of_sector_l701_701203

-- Given conditions as hypotheses
variable (r θ : ℝ)
variable (h₁ : (1/2) * θ * r^2 = 1)
variable (h₂ : 2 * r + θ * r = 4)

-- The goal statement to be proved
theorem central_angle_of_sector :
  θ = 2 :=
by sorry

end central_angle_of_sector_l701_701203


namespace parallel_line_slope_l701_701995

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l701_701995


namespace largest_n_unique_k_l701_701919

theorem largest_n_unique_k :
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, (3 : ℚ) / 7 < (n : ℚ) / ((n + k : ℕ) : ℚ) ∧ 
  (n : ℚ) / ((n + k : ℕ) : ℚ) < (8 : ℚ) / 19 → k = 1 := by
sorry

end largest_n_unique_k_l701_701919


namespace slope_of_parallel_line_l701_701962

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l701_701962


namespace parallel_slope_l701_701938

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l701_701938


namespace rectangle_pairs_l701_701828

theorem rectangle_pairs :
  {p : ℕ × ℕ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 * p.2 = 18} = {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} :=
by { sorry }

end rectangle_pairs_l701_701828


namespace only_one_tuple_exists_l701_701259

theorem only_one_tuple_exists :
  ∃! (x : Fin 15 → ℝ),
    (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2
    + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + (x 7 - x 8)^2 + (x 8 - x 9)^2
    + (x 9 - x 10)^2 + (x 10 - x 11)^2 + (x 11 - x 12)^2 + (x 12 - x 13)^2
    + (x 13 - x 14)^2 + (x 14)^2 = 1 / 16 := by
  sorry

end only_one_tuple_exists_l701_701259


namespace slope_of_parallel_line_l701_701954

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l701_701954


namespace sum_abs_roots_of_polynomial_l701_701665

noncomputable def polynomial_sum_abs_roots : ℂ :=
  abs ((3 + real.sqrt 13) / 2) + abs ((3 - real.sqrt 13) / 2) + abs ((3 + real.sqrt 19 * complex.I) / 2) + abs ((3 - real.sqrt 19 * complex.I) / 2)

theorem sum_abs_roots_of_polynomial :
  polynomial_sum_abs_roots = real.sqrt 13 + 2 * real.sqrt 7 := by
sorry

end sum_abs_roots_of_polynomial_l701_701665


namespace incircle_excircle_tangency_l701_701869

theorem incircle_excircle_tangency
  (a b c : ℝ)
  (K L : ℝ)
  (h1 : ∃ A B C : ℝ, A + B = C + K)
  (h2 : ∃ D E F : ℝ, D + E = F + L)
  (h3 : K = a + b - c / 2)
  (h4 : L = a + b - c / 2)
  : K = L :=
begin
  sorry
end

end incircle_excircle_tangency_l701_701869


namespace number_of_pairs_l701_701538

theorem number_of_pairs :
  ∃ n, (∀ a b : ℕ, a + b = 915 ∧ Nat.gcd a b = 61 → n = 8) :=
begin
  -- We know that if a and b are such that a + b = 915 and Nat.gcd a b = 61,
  -- then we can write a = 61 * x and b = 61 * y for some positive integers x and y,
  -- such that x + y = 15 and Nat.gcd x y = 1.
  --
  -- We need to show that the number of such pairs (x, y) is 8.
  sorry
end

end number_of_pairs_l701_701538


namespace q_is_composite_and_divides_power_l701_701415

variable (p : ℕ)

def is_prime (n : ℕ) : Prop := sorry -- Placeholder for prime checker definition

def q (p : ℕ) := (4^p - 1) / 3

theorem q_is_composite_and_divides_power (hp1 : p > 3) (hp2 : is_prime p) : 
  Nat.is_composite (q p) ∧ q p ∣ (2^(q p - 1) - 1) := 
sorry

end q_is_composite_and_divides_power_l701_701415


namespace slope_of_parallel_line_l701_701952

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l701_701952


namespace length_KL_l701_701504

variables {Point : Type} [metric_space Point]
variables {A B C D E P Q R S K L M : Point}
variables (AB BC CD DE AE PR QS : ℝ)
variables (midpoint : Point → Point → Point)
variables (length : Point → Point → ℝ)

-- Conditions given
axiom pentagon_eq_sides (AB BC CD DE AE : ℝ) : AB = 1 ∧ BC = 1 ∧ CD = 1 ∧ DE = 1
axiom is_midpoint (X Y Z : Point) :
  midpoint X Y = Z ↔ length X Z = length Y Z / 2
axiom midpoints 
  (P Q R S K L : Point)
  (AB BC CD DE PR QS : ℝ) 
  (MP AB BC CD DE PR QS length midpoint) :
  midpoint A B = P ∧ midpoint B C = Q ∧ midpoint C D = R ∧ midpoint D E = S ∧ 
  midpoint P R = K ∧ midpoint Q S = L

-- Geometry proof problem
theorem length_KL (A B C D E P Q R S K L : Point) :
  pentagon_eq_sides AB BC CD DE AE →
  midpoints P Q R S K L AB BC CD DE PR QS length midpoint →
  length K L = 1 / 4 * length A E := 
sorry

end length_KL_l701_701504


namespace area_of_paper_is_500_l701_701897

-- Define the width and length of the rectangular drawing paper
def width := 25
def length := 20

-- Define the formula for the area of a rectangle
def area (w : Nat) (l : Nat) : Nat := w * l

-- Prove that the area of the paper is 500 square centimeters
theorem area_of_paper_is_500 : area width length = 500 := by
  -- placeholder for the proof
  sorry

end area_of_paper_is_500_l701_701897


namespace correct_option_l701_701558

-- Declare the problem stating conditions and what we need to prove
theorem correct_option : 
  (2 * (Real.sqrt 2) - (Real.sqrt 2) ≠ 2) ∧ 
  (Real.sqrt (-5)^2 ≠ -5) ∧ 
  ((- Real.sqrt 2) ^ 2 ≠ - Real.sqrt 2) → 
  Real.cbrt -8 = -2 :=
by
  intros h
  sorry

end correct_option_l701_701558


namespace gain_percentage_44_l701_701596

theorem gain_percentage_44
  (selling_price_36 : ℚ := 1 / 36)
  (selling_price_24 : ℚ := 1 / 24)
  (loss_percentage : ℚ := 4 / 100)
  (cost_price : ℚ := selling_price_36 / (1 - loss_percentage))
  (gain_percentage : ℚ := ((selling_price_24 - cost_price) / cost_price) * 100) :
  gain_percentage ≈ 44 := 
sorry

end gain_percentage_44_l701_701596


namespace min_time_to_cook_noodles_l701_701052

/-- 
Li Ming needs to cook noodles, following these steps: 
① Boil the noodles for 4 minutes; 
② Wash vegetables for 5 minutes; 
③ Prepare the noodles and condiments for 2 minutes; 
④ Boil the water in the pot for 10 minutes; 
⑤ Wash the pot and add water for 2 minutes. 
Apart from step ④, only one step can be performed at a time. 
Prove that the minimum number of minutes needed to complete these tasks is 16.
-/
def total_time : Nat :=
  let t5 := 2 -- Wash the pot and add water
  let t4 := 10 -- Boil the water in the pot
  let t2 := 5 -- Wash vegetables
  let t3 := 2 -- Prepare the noodles and condiments
  let t1 := 4 -- Boil the noodles
  t5 + t4.max (t2 + t3) + t1

theorem min_time_to_cook_noodles : total_time = 16 :=
by
  sorry

end min_time_to_cook_noodles_l701_701052


namespace find_BE_l701_701815

namespace geometry

variables {A B C D F E G : Point}
variables {AD BC : Line}
variables {ABCD : Parallelogram A B C D}
variables (EF GF BE : ℝ)
variables (h1 : F ∈ Line.extension AD)
variables (h2 : E ∈ Line.inter BF AC)
variables (h3 : G ∈ Line.inter BF DC)
variables (h4 : EF = 32)
variables (h5 : GF = 24)

theorem find_BE : BE = 16 :=
sorry

end geometry

end find_BE_l701_701815


namespace scientific_notation_of_20_ns_l701_701387

def nanosecond_to_scientific_notation (n: ℕ) : ℝ := n * 10⁻⁹

theorem scientific_notation_of_20_ns :
  nanosecond_to_scientific_notation 20 = 2 * 10⁻⁸ := 
by {
  sorry
}

end scientific_notation_of_20_ns_l701_701387


namespace trigonometric_values_and_identity_l701_701704

noncomputable def point_to_trig (x y : ℝ) : ℝ × ℝ × ℝ :=
  let r := real.sqrt(x^2 + y^2)
  (y / r, x / r, y / x)

def trig_identity (sin_alpha cos_alpha : ℝ) : ℝ :=
  ((-sin_alpha) + 2 * cos_alpha) / (-2 * cos_alpha)

theorem trigonometric_values_and_identity : 
  let P : ℝ × ℝ := (4, -3)
  let (sin_alpha, cos_alpha, tan_alpha) := point_to_trig P.fst P.snd in 
  sin_alpha = -3/5 ∧ cos_alpha = 4/5 ∧ tan_alpha = -3/4 ∧
  trig_identity sin_alpha cos_alpha = -11/8 := 
by
  sorry

end trigonometric_values_and_identity_l701_701704


namespace total_eggs_michael_has_l701_701440

-- Define the initial number of crates
def initial_crates : ℕ := 6

-- Define the number of crates given to Susan
def crates_given_to_susan : ℕ := 2

-- Define the number of crates bought on Thursday
def crates_bought_thursday : ℕ := 5

-- Define the number of eggs per crate
def eggs_per_crate : ℕ := 30

-- Theorem stating the total number of eggs Michael has now
theorem total_eggs_michael_has :
  (initial_crates - crates_given_to_susan + crates_bought_thursday) * eggs_per_crate = 270 :=
sorry

end total_eggs_michael_has_l701_701440


namespace tv_weekly_cost_l701_701018

theorem tv_weekly_cost(
  (tv_power : ℕ) (tv_hours_per_day : ℕ) (cost_per_kWh : ℕ)
  (kilowatt_in_watts : ℕ) (dollar_in_cents : ℕ)
  (h1 : tv_power = 125) (h2 : tv_hours_per_day = 4) 
  (h3 : cost_per_kWh = 14) (h4 : kilowatt_in_watts = 1000) 
  (h5 : dollar_in_cents = 100) :
  (tv_power * tv_hours_per_day * 7 / kilowatt_in_watts * cost_per_kWh = 49) :=
sorry

end tv_weekly_cost_l701_701018


namespace range_of_a_l701_701243

theorem range_of_a (a : ℝ) : (∀ x : ℝ, sin x ^ 6 + cos x ^ 6 + 2 * a * sin x * cos x ≥ 0) → |a| ≤ 1 / 4 :=
by
  sorry

end range_of_a_l701_701243


namespace largest_n_unique_k_l701_701916

theorem largest_n_unique_k : ∃ n : ℕ, n = 24 ∧ (∃! k : ℕ, 
  3 / 7 < n / (n + k: ℤ) ∧ n / (n + k: ℤ) < 8 / 19) :=
by
  sorry

end largest_n_unique_k_l701_701916


namespace fraction_eggs_used_for_cupcakes_l701_701906

theorem fraction_eggs_used_for_cupcakes:
  ∀ (total_eggs crepes_fraction remaining_eggs after_cupcakes_eggs used_for_cupcakes_fraction: ℚ),
  total_eggs = 36 →
  crepes_fraction = 1 / 4 →
  after_cupcakes_eggs = 9 →
  used_for_cupcakes_fraction = 2 / 3 →
  (total_eggs * (1 - crepes_fraction) - after_cupcakes_eggs) / (total_eggs * (1 - crepes_fraction)) = used_for_cupcakes_fraction :=
by
  intros
  sorry

end fraction_eggs_used_for_cupcakes_l701_701906


namespace centroid_median_ratio_l701_701479

noncomputable def centroid {α : Type*} [metric_space α] [normed_group α] (A B C : α) : α := 
  1 / 3 • (A + B + C)

def median_intersection_circumcircle {α : Type*} [metric_space α] [normed_group α]
  (A B C G : α) : (α × α × α) :=
  sorry -- This would be a detailed geometric construction dependent on Lean's geometry library.

theorem centroid_median_ratio (A B C G A1 B1 C1 : ℝ) :
  let A1 := median_intersection_circumcircle A B C G in
  let B1 := median_intersection_circumcircle A B C G in
  let C1 := median_intersection_circumcircle A B C G in
  G = centroid A B C →
  (∃ A1 B1 C1, 
  ∃ (G = centroid A B C)
  ∧ 
  (median_intersection_circumcircle A B C G = (A1, B1, C1))) →
  (1 / (1 / 3 • (A + B + C)) = ((A + B + C) / 3) 
  ∧ (A + B + C) / 3 = 1 / (A + B + C)) ↔
  (let AG := dist A G, 
   let BG := dist B G, 
   let CG := dist C G, 
   let GA1 := dist G A1, 
   let GB1 := dist G B1, 
   let GC1 := dist G C1 in
   (AG / GA1) + (BG / GB1) + (CG / GC1) = 3) :=
sorry

end centroid_median_ratio_l701_701479


namespace linear_function_product_neg_l701_701306

theorem linear_function_product_neg (a1 b1 a2 b2 : ℝ) (hP : b1 = -3 * a1 + 4) (hQ : b2 = -3 * a2 + 4) :
  (a1 - a2) * (b1 - b2) < 0 :=
by
  sorry

end linear_function_product_neg_l701_701306


namespace min_rectangles_needed_l701_701912

theorem min_rectangles_needed 
  (type1_corners type2_corners : ℕ)
  (rectangles_cover : ℕ → ℕ)
  (h1 : type1_corners = 12)
  (h2 : type2_corners = 12)
  (h3 : ∀ n, rectangles_cover (3 * n) = n) : 
  (rectangles_cover type2_corners) + (rectangles_cover type1_corners) = 12 := 
sorry

end min_rectangles_needed_l701_701912


namespace scientific_notation_of_20_nanoseconds_l701_701392

def nanosecond_to_seconds (nanoseconds : ℕ) : ℝ :=
  nanoseconds * (1 * 10 ^ (-9))

theorem scientific_notation_of_20_nanoseconds :
  nanosecond_to_seconds 20 = 2 * 10 ^ (-8) := by
  sorry

end scientific_notation_of_20_nanoseconds_l701_701392


namespace smallest_int_k_for_64_pow_k_l701_701553

theorem smallest_int_k_for_64_pow_k (k : ℕ) (base : ℕ) (h₁ : k = 7) : 
  64^k > base^20 → base = 4 := by
  sorry

end smallest_int_k_for_64_pow_k_l701_701553


namespace inequality_holds_l701_701273

theorem inequality_holds (θ : ℝ) (a : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π / 2) :
  (∃ a, a > 3) → 
  sqrt 2 * (2 * a + 3) * cos (θ - π / 4) + 6 / (sin θ + cos θ) - 2 * sin (2 * θ) < 3 * a + 6 :=
by
  intro ha
  sorry

end inequality_holds_l701_701273


namespace length_KL_eq_one_fourth_l701_701492

section Pentagon

variable {A B C D E P Q R S K L : Type}
variable [AddCommGroup P] [Module ℕ P]

-- Assuming all sides of the pentagon are of length 1
variable (length_AB length_BC length_CD length_DE length_AE : ℕ := 1)

-- Define midpoints assumptions
variable (is_midpoint_AB : P = midpoint A B)
variable (is_midpoint_BC : Q = midpoint B C)
variable (is_midpoint_CD : R = midpoint C D)
variable (is_midpoint_DE : S = midpoint D E)

-- Define KL midpoints
variable (is_midpoint_PR : K = midpoint P R)
variable (is_midpoint_QS : L = midpoint Q S)

-- Prove the length of segment KL
theorem length_KL_eq_one_fourth :
  dist K L = 1 / 4 :=
sorry

end Pentagon

end length_KL_eq_one_fourth_l701_701492


namespace exists_arithmetic_progression_with_11_numbers_exists_arithmetic_progression_with_10000_numbers_not_exists_infinite_arithmetic_progression_l701_701163

open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_arithmetic_progression_with_11_numbers :
  ∃ a d : ℕ, ∀ i j : ℕ, i < 11 → j < 11 → i < j → a + i * d < a + j * d ∧ 
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d) := by
  sorry

theorem exists_arithmetic_progression_with_10000_numbers :
  ∃ a d : ℕ, ∀ i j : ℕ, i < 10000 → j < 10000 → i < j → a + i * d < a + j * d ∧
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d) := by
  sorry

theorem not_exists_infinite_arithmetic_progression :
  ¬ (∃ a d : ℕ, ∀ i j : ℕ, i < j → a + i * d < a + j * d ∧
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d)) := by
  sorry

end exists_arithmetic_progression_with_11_numbers_exists_arithmetic_progression_with_10000_numbers_not_exists_infinite_arithmetic_progression_l701_701163


namespace tilly_counts_513_stars_l701_701531

noncomputable def tillyStarsTotal : ℕ := 
  let east := 120
  let west := 2 * east
  let north := east + Nat.floor (0.15 * east)
  let south := Nat.floor (Real.sqrt west)
  east + west + north + south

theorem tilly_counts_513_stars :
  tillyStarsTotal = 513 :=
  sorry

end tilly_counts_513_stars_l701_701531


namespace S_n_formula_l701_701882

-- Define sequence terms and conditions
def a : ℕ → ℝ
def S : ℕ → ℝ

axiom a_1 : a 1 = 1
axiom S_def : ∀ n : ℕ, n > 0 → S n = 3 * a (n + 1)

-- The goal is to prove the following statement
theorem S_n_formula : ∀ n : ℕ, n > 0 → S n = (4 / 3)^(n - 1) :=
by 
  -- Proof omitted
  sorry

end S_n_formula_l701_701882


namespace slope_of_parallel_line_l701_701948

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l701_701948


namespace decrypt_last_block_l701_701913

/-- Define the permutation P. -/
def P : List Nat := [8, 9, 7, 2, 4, 1, 10, 6, 5, 3]

/-- Apply a given permutation to a list. -/
def apply_permutation (perm : List Nat) (lst : List Char) : List Char :=
  List.map (fun i => lst.get! (i - 1)) perm

noncomputable def P_times_k (k : Nat) (lst : List Char) : List Char :=
  (Finset.range k).foldl (fun acc _ => apply_permutation P acc) lst

/-- Given the final encrypted message, prove that the last word of the original text is "зоопарк". -/
theorem decrypt_last_block :
  let encrypted_message := "авзодптмееарпазркоов"
  let decrypted_message := P_times_k 334 encrypted_message.toList
  decrypted_message.drop(20) = "зоопарк".toList :=
sorry

end decrypt_last_block_l701_701913


namespace algebraic_identity_simplification_l701_701245

theorem algebraic_identity_simplification :
  (5 + 1) * (5^2 + 1^2) * (5^4 + 1^4) * (5^8 + 1^8) * (5^{16} + 1^{16}) * (5^{32} + 1^{32}) * (5^{64} + 1^{64}) = 5^{128} - 1^{128} :=
by
  sorry

end algebraic_identity_simplification_l701_701245


namespace circle_passing_through_points_l701_701844

noncomputable def circle_equation (D E F : ℝ) : ℝ × ℝ → ℝ :=
λ p, p.1^2 + p.2^2 + D * p.1 + E * p.2 + F

theorem circle_passing_through_points : ∃ D E F : ℝ, 
  circle_equation D E F (0, 0) = 0 ∧
  circle_equation D E F (4, 0) = 0 ∧
  circle_equation D E F (-1, 1) = 0 ∧
  D = -4 ∧ 
  E = -6 ∧ 
  F = 0 :=
begin
  sorry
end

end circle_passing_through_points_l701_701844


namespace max_product_f_h_l701_701075

-- Define the function ranges and the relationship between h(x) and g(x)
variable {f : ℝ → ℝ} (hf : ∀ x, f(x) ∈ Set.Icc (-3 : ℝ) 4)
variable {h : ℝ → ℝ} (hh : ∀ x, h(x) ∈ Set.Icc (-1 : ℝ) 2)
variable {g : ℝ → ℝ} (hg : ∀ x, g(x) ∈ Set.Icc (-2 : ℝ) 1)
variable (h_eq_g_plus_one : ∀ x, h(x) = g(x) + 1)

-- Prove the largest possible value of f(x) * h(x)
theorem max_product_f_h : ∃ x, f(x) * h(x) = 12 :=
sorry

end max_product_f_h_l701_701075


namespace max_possible_value_of_expression_l701_701424

noncomputable def calc_expression (x : ℝ) : ℝ :=
(x^4 + 6 - (x^8 + 8).sqrt) / x^2

theorem max_possible_value_of_expression:
  ∀ (x : ℝ), 0 < x →
  calc_expression x ≤ 4 * (14 * Real.sqrt 7 - 35) :=
by {
  intro x hx,
  sorry
}

end max_possible_value_of_expression_l701_701424


namespace duration_of_call_l701_701024

def initial_credit : ℝ := 30
def remaining_credit : ℝ := 26.48
def cost_per_minute : ℝ := 0.16
def credit_used : ℝ := initial_credit - remaining_credit
def minutes_of_call : ℕ := (credit_used / cost_per_minute).toInt

theorem duration_of_call :
  minutes_of_call = 22 :=
by
  -- Placeholder proof statement
  sorry

end duration_of_call_l701_701024


namespace triangle_inradius_condition_l701_701902

variables {A B C O : Type} [metric_space O]
variable {r : ℝ}

-- Assuming A, B, C are points on a circle centered at O with radius 1
def on_circle (A B C O : O) (radius : ℝ) : Prop :=
  dist A O = radius ∧ dist B O = radius ∧ dist C O = radius ∧ radius = 1

-- Given A, B, C form a triangle and we are given r which is the inradius of the triangle
def inradius_le_half (A B C O : O) (r : ℝ) : Prop :=
  on_circle A B C O 1 → r ≤ 1 / 2

theorem triangle_inradius_condition {A B C O : O} {r : ℝ} :
  inradius_le_half A B C O r :=
begin
  sorry
end

end triangle_inradius_condition_l701_701902


namespace count_multiples_7_not_14_l701_701725

theorem count_multiples_7_not_14 (n : ℕ) (h1 : n < 300) (h2 : n % 7 = 0) (h3 : n % 14 ≠ 0) :
  (finset.filter (λ x, (x < 300) ∧ (x % 7 = 0) ∧ (x % 14 ≠ 0))
    (finset.range 300)).card = 21 := by
  sorry

end count_multiples_7_not_14_l701_701725


namespace circle_passing_through_points_l701_701847

noncomputable def circle_equation (D E F : ℝ) : ℝ × ℝ → ℝ :=
λ p, p.1^2 + p.2^2 + D * p.1 + E * p.2 + F

theorem circle_passing_through_points : ∃ D E F : ℝ, 
  circle_equation D E F (0, 0) = 0 ∧
  circle_equation D E F (4, 0) = 0 ∧
  circle_equation D E F (-1, 1) = 0 ∧
  D = -4 ∧ 
  E = -6 ∧ 
  F = 0 :=
begin
  sorry
end

end circle_passing_through_points_l701_701847


namespace car_division_ways_l701_701584

/-- 
Prove that the number of ways to divide 6 people 
into two different cars, with each car holding 
a maximum of 4 people, is equal to 50. 
-/
theorem car_division_ways : 
  (∃ s1 s2 : Finset ℕ, s1.card = 2 ∧ s2.card = 4) ∨ 
  (∃ s1 s2 : Finset ℕ, s1.card = 3 ∧ s2.card = 3) ∨ 
  (∃ s1 s2 : Finset ℕ, s1.card = 4 ∧ s2.card = 2) →
  (15 + 20 + 15 = 50) := 
by 
  sorry

end car_division_ways_l701_701584


namespace goods_train_speed_is_52_l701_701196

def man_train_speed : ℕ := 60 -- speed of the man's train in km/h
def goods_train_length : ℕ := 280 -- length of the goods train in meters
def time_to_pass : ℕ := 9 -- time for the goods train to pass the man in seconds
def relative_speed_kmph : ℕ := (goods_train_length * 3600) / (time_to_pass * 1000) -- relative speed in km/h, calculated as (0.28 km / (9/3600) h)
def goods_train_speed : ℕ := relative_speed_kmph - man_train_speed -- speed of the goods train in km/h

theorem goods_train_speed_is_52 : goods_train_speed = 52 := by
  sorry

end goods_train_speed_is_52_l701_701196


namespace total_weight_of_soil_extracted_l701_701001

noncomputable def pond_soil_weight
  (l_min l_max w_min w_max d_min d_max : ℝ) (ρ1 ρ2 : ℝ) : ℝ :=
  let L := (l_min + l_max) / 2
  let W := (w_min + w_max) / 2
  let D := (d_min + d_max) / 2
  let V1 := L * W * 2.5
  let V2 := L * W * 2.5
  let weight1 := V1 * ρ1
  let weight2 := V2 * ρ2
  weight1 + weight2

theorem total_weight_of_soil_extracted :
  pond_soil_weight 18 22 14 16 4.5 5.5 1800 2200 = 3_000_000 := by
  sorry

end total_weight_of_soil_extracted_l701_701001


namespace slope_of_parallel_line_l701_701959

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l701_701959


namespace log_sin_cos_increasing_interval_l701_701510

theorem log_sin_cos_increasing_interval (a : ℝ) (k : ℤ) : 
  0 < a ∧ a < 1 → 
  ∀ x, 2 * k * Real.pi + Real.pi / 4 ≤ x ∧ x < 2 * k * Real.pi + 3 * Real.pi / 4 → 
  StrictMonoOn (fun x => Real.log a (Real.sin x + Real.cos x))
    (Icc (2 * k * Real.pi + Real.pi / 4) (2 * k * Real.pi + 3 * Real.pi / 4)) :=
sorry

end log_sin_cos_increasing_interval_l701_701510


namespace intersection_complement_M_N_eq_456_l701_701329

def UniversalSet := { n : ℕ | 1 ≤ n ∧ n < 9 }
def M : Set ℕ := { 1, 2, 3 }
def N : Set ℕ := { 3, 4, 5, 6 }

theorem intersection_complement_M_N_eq_456 : 
  (UniversalSet \ M) ∩ N = { 4, 5, 6 } :=
by
  sorry

end intersection_complement_M_N_eq_456_l701_701329


namespace no_fermat_in_sequence_l701_701395

def sequence_term (n k : ℕ) : ℕ :=
  (k - 2) * n * (n - 1) / 2 + n

def is_fermat_number (a : ℕ) : Prop :=
  ∃ m : ℕ, a = 2^(2^m) + 1

theorem no_fermat_in_sequence (k n : ℕ) (hk : k > 2) (hn : n > 2) :
  ¬ is_fermat_number (sequence_term n k) :=
sorry

end no_fermat_in_sequence_l701_701395


namespace lexi_laps_l701_701795

theorem lexi_laps (total_distance lap_distance : ℝ) (h1 : total_distance = 3.25) (h2 : lap_distance = 0.25) :
  total_distance / lap_distance = 13 :=
by
  sorry

end lexi_laps_l701_701795


namespace total_crew_members_l701_701588

def num_islands : ℕ := 3
def ships_per_island : ℕ := 12
def crew_per_ship : ℕ := 24

theorem total_crew_members : num_islands * ships_per_island * crew_per_ship = 864 := by
  sorry

end total_crew_members_l701_701588


namespace circle_passing_through_points_l701_701848

theorem circle_passing_through_points :
  ∃ D E F : ℝ, ∀ (x y : ℝ),
    ((x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) →
    (x^2 + y^2 + D*x + E*y + F = 0) ↔
    (x^2 + y^2 - 4*x - 6*y = 0) :=
begin
  sorry,
end

end circle_passing_through_points_l701_701848


namespace number_of_elements_less_than_2004_l701_701042

theorem number_of_elements_less_than_2004 (f : ℕ → ℕ) 
    (h0 : f 0 = 0) 
    (h1 : ∀ n : ℕ, (f (2 * n + 1)) ^ 2 - (f (2 * n)) ^ 2 = 6 * f n + 1) 
    (h2 : ∀ n : ℕ, f (2 * n) > f n) 
  : ∃ m : ℕ,  m = 128 ∧ ∀ x : ℕ, f x < 2004 → x < m := sorry

end number_of_elements_less_than_2004_l701_701042


namespace kaleb_candy_problem_l701_701015

-- Define the initial problem with given conditions

theorem kaleb_candy_problem :
  ∀ (total_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ),
    total_boxes = 14 →
    given_away_boxes = 5 →
    pieces_per_box = 6 →
    (total_boxes - given_away_boxes) * pieces_per_box = 54 :=
by
  intros total_boxes given_away_boxes pieces_per_box
  intros h1 h2 h3
  -- Use assumptions
  sorry

end kaleb_candy_problem_l701_701015


namespace total_fruit_weight_l701_701464

def melon_weight : Real := 0.35
def berries_weight : Real := 0.48
def grapes_weight : Real := 0.29
def pineapple_weight : Real := 0.56
def oranges_weight : Real := 0.17

theorem total_fruit_weight : melon_weight + berries_weight + grapes_weight + pineapple_weight + oranges_weight = 1.85 :=
by
  unfold melon_weight berries_weight grapes_weight pineapple_weight oranges_weight
  sorry

end total_fruit_weight_l701_701464


namespace circle_passing_through_points_l701_701851

theorem circle_passing_through_points :
  ∃ D E F : ℝ, ∀ (x y : ℝ),
    ((x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) →
    (x^2 + y^2 + D*x + E*y + F = 0) ↔
    (x^2 + y^2 - 4*x - 6*y = 0) :=
begin
  sorry,
end

end circle_passing_through_points_l701_701851


namespace length_KL_l701_701503

variables {Point : Type} [metric_space Point]
variables {A B C D E P Q R S K L M : Point}
variables (AB BC CD DE AE PR QS : ℝ)
variables (midpoint : Point → Point → Point)
variables (length : Point → Point → ℝ)

-- Conditions given
axiom pentagon_eq_sides (AB BC CD DE AE : ℝ) : AB = 1 ∧ BC = 1 ∧ CD = 1 ∧ DE = 1
axiom is_midpoint (X Y Z : Point) :
  midpoint X Y = Z ↔ length X Z = length Y Z / 2
axiom midpoints 
  (P Q R S K L : Point)
  (AB BC CD DE PR QS : ℝ) 
  (MP AB BC CD DE PR QS length midpoint) :
  midpoint A B = P ∧ midpoint B C = Q ∧ midpoint C D = R ∧ midpoint D E = S ∧ 
  midpoint P R = K ∧ midpoint Q S = L

-- Geometry proof problem
theorem length_KL (A B C D E P Q R S K L : Point) :
  pentagon_eq_sides AB BC CD DE AE →
  midpoints P Q R S K L AB BC CD DE PR QS length midpoint →
  length K L = 1 / 4 * length A E := 
sorry

end length_KL_l701_701503


namespace optimal_tax_choice_l701_701139

def TotalIncome (revenue advances : ℕ) : ℕ :=
  revenue + advances

def MonthlyExpenses (rent oils salaries insurance accounting advertising retraining misc : ℕ) : ℕ :=
  rent + oils + salaries + insurance + accounting + advertising + retraining + misc

def AnnualExpenses (monthlyExpenses : ℕ) : ℕ :=
  monthlyExpenses * 12

def IncomeTax (totalIncome : ℕ) (rate : ℕ) : ℕ :=
  totalIncome * rate / 100

def ExpenditureTax (totalIncome annualExpenses : ℕ) (rate : ℕ) : ℕ :=
  (totalIncome - annualExpenses) * rate / 100

def MinimumTax (totalIncome : ℕ) (rate : ℕ) : ℕ :=
  totalIncome * rate / 100

def FinalTaxPayable (incomeTax deductionLimit : ℕ) : ℕ :=
  incomeTax - deductionLimit

theorem optimal_tax_choice 
  (revenue advances rent oils salaries insurance accounting advertising retraining misc : ℕ)
  (insuranceLimitRate incomeTaxRate expenditureTaxRate minTaxRate : ℕ) :
  let totalIncome := TotalIncome revenue advances,
      monthlyExpenses := MonthlyExpenses rent oils salaries insurance accounting advertising retraining misc,
      annualExpenses := AnnualExpenses monthlyExpenses,
      incomeTaxBase := totalIncome,
      expenditureTaxBase := totalIncome - annualExpenses,
      incomeTax := IncomeTax incomeTaxBase incomeTaxRate,
      deductionLimit := incomeTax * insuranceLimitRate / 100,
      incomeTaxPayable := FinalTaxPayable incomeTax deductionLimit,
      expenditureTax := ExpenditureTax totalIncome annualExpenses expenditureTaxRate,
      minTax := MinimumTax totalIncome minTaxRate in
  (totalIncome = revenue + advances ∧ 
  monthlyExpenses = rent + oils + salaries + insurance + accounting + advertising + retraining + misc ∧
  annualExpenses = monthlyExpenses * 12 ∧
  incomeTax = incomeTaxBase * incomeTaxRate / 100 ∧
  deductionLimit = incomeTax * insuranceLimitRate / 100 ∧
  incomeTaxPayable = incomeTax - deductionLimit ∧
  expenditureTax = (totalIncome - annualExpenses) * expenditureTaxRate / 100 ∧
  minTax = totalIncome * minTaxRate / 100 ∧
  incomeTaxPayable > minTax) → 
  (IncomeTaxPayable > MinTax) 
  :=
begin
  sorry
end

end optimal_tax_choice_l701_701139


namespace three_digit_numbers_proof_l701_701610

-- Definitions and conditions
def are_digits_distinct (A B C : ℕ) := (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C)

def is_arithmetic_mean (A B C : ℕ) := 2 * B = A + C

def geometric_mean_property (A B C : ℕ) := 
  (100 * A + 10 * B + C) * (100 * C + 10 * A + B) = (100 * B + 10 * C + A)^2

-- statement of the proof problem
theorem three_digit_numbers_proof :
  ∃ A B C : ℕ, (10 ≤ A) ∧ (A ≤ 99) ∧ (10 ≤ B) ∧ (B ≤ 99) ∧ (10 ≤ C) ∧ (C ≤ 99) ∧
  (A * 100 + B * 10 + C = 432 ∨ A * 100 + B * 10 + C = 864) ∧
  are_digits_distinct A B C ∧
  is_arithmetic_mean A B C ∧
  geometric_mean_property A B C :=
by {
  -- The Lean proof goes here
  sorry
}

end three_digit_numbers_proof_l701_701610


namespace parallel_lines_l701_701320

theorem parallel_lines :
  (∃ m: ℚ, (∀ x y: ℚ, (4 * y - 3 * x = 16 → y = m * x + (16 / 4)) ∧
                      (-3 * x - 4 * y = 15 → y = -m * x - (15 / 4)) ∧
                      (4 * y + 3 * x = 16 → y = -m * x + (16 / 4)) ∧
                      (3 * y + 4 * x = 15) → False)) :=
sorry

end parallel_lines_l701_701320


namespace solve_for_a_l701_701003

noncomputable def focuses_of_C2 (a : ℝ) (ha : 1 < a) : Set (ℝ × ℝ) :=
{ ( √(a^2 - 1), 0 ), (-√(a^2 - 1), 0 )}

-- Define curve C1 parametrically
def curveC1 (t : ℝ) : ℝ × ℝ := (t + 1/t, t - 1/t)

-- Define the condition that curve C1 passes through the foci of C2
def C1_passes_through_foci (a : ℝ) (ha : 1 < a) : Prop :=
  ∀ focus ∈ focuses_of_C2 a ha, ∃ t, curveC1 t = focus

-- Theorem that states that a must be sqrt(5) under given conditions
theorem solve_for_a (a : ℝ) (ha : 1 < a) (h : C1_passes_through_foci a ha) : a = √5 :=
sorry

end solve_for_a_l701_701003


namespace slope_parallel_to_original_line_l701_701946

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l701_701946


namespace xiao_hua_commemorative_stamp_ways_l701_701564

theorem xiao_hua_commemorative_stamp_ways (has_yuan : ℕ := 7) (face_values : set ℕ := {2, 3}) :
    ∃ ways : ℕ, ways = 7 ∧ 
               (∀ n : ℕ, (n > 0 ∧ ∃ a b : ℕ, a * 2 + b * 3 = n) → 
                          (∃ counts : ℕ, counts = if n = 2 then 1 else 
                                                  if n = 3 then 1 else 
                                                  if n = 4 then 1 else 
                                                  if n = 5 then 1 else 
                                                  if n = 6 then 1 else 0)) :=
by
  sorry

end xiao_hua_commemorative_stamp_ways_l701_701564


namespace students_in_Johnsons_class_l701_701805

theorem students_in_Johnsons_class :
  ∀ (students_Finley : ℕ) (students_Johnson : ℕ),
    students_Finley = 24 →
    students_Johnson = (students_Finley / 2) + 10 →
    students_Johnson = 22 :=
by
  intros students_Finley students_Johnson hFinley hJohnson
  rw [hFinley, Nat.div_add_self 12 0],
  exact hJohnson

sorry

end students_in_Johnsons_class_l701_701805


namespace spring_spending_l701_701480

theorem spring_spending (end_of_feb : ℝ) (end_of_may : ℝ) (h_end_of_feb : end_of_feb = 0.8) (h_end_of_may : end_of_may = 2.5)
  : (end_of_may - end_of_feb) = 1.7 :=
by
  have spending_end_of_feb : end_of_feb = 0.8 := h_end_of_feb
  have spending_end_of_may : end_of_may = 2.5 := h_end_of_may
  sorry

end spring_spending_l701_701480


namespace min_value_f_l701_701681

noncomputable def f : ℝ → ℝ := λ x, Real.cos x + x * Real.sin x

theorem min_value_f : ∃ x ∈ set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 ∧ ∀ y ∈ set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ f y := 
sorry

end min_value_f_l701_701681


namespace slope_of_parallel_line_l701_701988

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l701_701988


namespace teacher_drank_milk_false_l701_701110

-- Define the condition that the volume of milk a teacher can reasonably drink in a day is more appropriately measured in milliliters rather than liters.
def reasonable_volume_units := "milliliters"

-- Define the statement to be judged
def teacher_milk_intake := 250

-- Define the unit of the statement
def unit_of_statement := "liters"

-- The proof goal is to conclude that the statement "The teacher drank 250 liters of milk today" is false, given the condition on volume units.
theorem teacher_drank_milk_false (vol : ℕ) (unit : String) (reasonable_units : String) :
  vol = 250 ∧ unit = "liters" ∧ reasonable_units = "milliliters" → false :=
by
  sorry

end teacher_drank_milk_false_l701_701110


namespace triangle_side_relationship_l701_701747

theorem triangle_side_relationship 
  (a b c : ℝ) 
  (h1 : a^2 = b * (b + c)) 
  (h2 : ∠C > 90) 
  : a < 2 * b ∧ 2 * b < c :=
sorry

end triangle_side_relationship_l701_701747


namespace original_population_l701_701208

theorem original_population (p : ℝ) (h : 0.85 * p + 1500 = p - 50) : p = 10333.3 :=
by skip -- sorry to skip the proof

end original_population_l701_701208


namespace length_of_KL_l701_701484

theorem length_of_KL (A B C D E P Q R S K L : Type) 
  (hA: A = (1 : ℝ)) 
  (hB: B = (1 : ℝ)) 
  (hC: C = (1 : ℝ)) 
  (hD: D = (1 : ℝ)) 
  (hE: E = (1 : ℝ)) 
  (hP: P = (1 / 2 : ℝ)) 
  (hQ: Q = (1 / 2 : ℝ)) 
  (hR: R = (1 / 2 : ℝ)) 
  (hS: S = (1 / 2 : ℝ)) 
  (hK: K = (1 / 2 : ℝ)) 
  (hL: L = (1 / 2 : ℝ)) : 
  KL = (1 / 4 : ℝ) := 
sorry

end length_of_KL_l701_701484


namespace coefficient_of_x6_in_sum_of_powers_l701_701656

theorem coefficient_of_x6_in_sum_of_powers :
  let f := (1 : ℚ) + x
  ∑ i in finset.range 3, (f ^ (5 + i)).coeff 6 = 8 :=
by sorry

end coefficient_of_x6_in_sum_of_powers_l701_701656


namespace minimize_area_OMAN_maximize_on_parallel_equal_length_l701_701330

open Real EuclideanGeometry

-- Given a triangle OBC with angle BOC = alpha
variables {O B C A M N : Point}
variable  (α β : ℝ)
variable [Triangle O B C]
variable (h_angle_boc : ∡ O B C = α)

-- Points A on BC, M on OB, and N on OC such that angle MAN = beta
variables (h_A_on_BC : A ∈ LineSegment B C)
variables (h_M_on_OB : M ∈ LineSegment O B)
variables (h_N_on_OC : N ∈ LineSegment O C)
variables (h_angle_MAN : ∡ M A N = β)

theorem minimize_area_OMAN_maximize_on_parallel_equal_length :
  (| M A | = | A N | ∧ parallel (Line M N) (Line B C)) → 
  minimal_area (Quadrilateral O M A N) :=
sorry

end minimize_area_OMAN_maximize_on_parallel_equal_length_l701_701330


namespace hyperbola_b_value_equivalent_l701_701327

theorem hyperbola_b_value_equivalent
  (b : ℝ)
  (x_1 y_1 x_2 y_2 : ℝ)
  -- Conditions including the equation of the hyperbola at points A and B
  (h_A : x_1^2 / 4 - y_1^2 / b^2 = 1)
  (h_B : x_2^2 / 4 - y_2^2 / b^2 = 1)
  -- Condition of midpoint P
  (h_midpoint_x : 2 * (x_1 + x_2) = x_1 + x_2)
  (h_midpoint_y : 2 * (y_1 + y_2) = y_1 + y_2)
  -- Given slope of the line OP
  (h_slope_OP : (y_1 + y_2) / (x_1 + x_2) = 1 / 4) 
  : b = sqrt 2 :=
by 
  sorry

end hyperbola_b_value_equivalent_l701_701327


namespace circle_passing_points_l701_701853

theorem circle_passing_points (x y : ℝ) :
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 - 4*x - 6*y = 0 :=
by
  intros h
  cases h
  case inl h₁ => 
    rw [h₁.1, h₁.2]
    ring
  case inr h₁ =>
    cases h₁
    case inl h₂ => 
      rw [h₂.1, h₂.2]
      ring
    case inr h₂ =>
      rw [h₂.1, h₂.2]
      ring

end circle_passing_points_l701_701853


namespace distance_Q_to_BD_l701_701200

-- Define a structure for a point in 2D
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Define midpoint function
def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

-- Define the line function and the projection formula
def line (p1 p2 : Point) : ℝ := (p2.y - p1.y) / (p2.x - p1.x)

-- Define the point projections onto lines function
-- This function is used to find the perpendicular projection of a point onto a line
noncomputable def proj (p1 p2 p : Point) : Point :=
  let m := line p1 p2
  let b := p1.y - m * p1.x
  let x := (m * p.y + p.x - m * b) / (m^2 + 1)
  let y := (m^2 * p.y + m * p.x + b) / (m^2 + 1)
  { x := x, y := y }

-- Define distance from a point to a line
noncomputable def distance_to_line (p1 p2 p : Point) : ℝ :=
  let p_proj := proj p1 p2 p
  distance p p_proj

-- Given conditions for point placements and lengths
constant A : Point
constant B : Point
constant C : Point
constant D : Point
constant P : Point
constant Q : Point

axiom h_AB_len : distance A B = 14
axiom h_BC_len : distance B C = 8
axiom h_AC_len : distance A C = 18
axiom h_P_div : distance A P = (2 / 5) * distance A B
axiom h_midpoint_Q : Q = midpoint A C

-- The goal is to prove that the distance from Q to the line BD is 24√5 / 11
theorem distance_Q_to_BD : distance_to_line B D Q = 24 * Real.sqrt 5 / 11 := by 
  sorry

end distance_Q_to_BD_l701_701200


namespace pin_pierces_all_sheets_l701_701825

theorem pin_pierces_all_sheets (sheets : ℕ) (top_sheet : ℝ → ℝ → Prop) (other_sheets : ℕ → ℝ → ℝ → Prop) :
  (∀ i, top_sheet = other_sheets i) → 
  (∀ i, ∃ A B,
   A < B ∧
   (area (slice_of (other_sheets i)) < (1/2) * area (slice_of top_sheet))) →
  ∃ O, 
    (∀ i, O ∈ slice_of top_sheet ∧ O ∈ slice_of (other_sheets i)) :=
sorry

end pin_pierces_all_sheets_l701_701825


namespace min_value_expression_l701_701450
-- We begin by importing the necessary mathematical libraries

-- Definitions based on the conditions
def X_distribution : ℝ → ℝ := sorry -- we assume a normal distribution N(10, σ^2)

-- Define the probabilities given in the conditions
def P_X_gt_12 : ℝ := sorry -- P(X > 12) = m
def P_8_le_X_le_10 : ℝ := sorry -- P(8 ≤ X ≤ 10) = n

-- Define the expressions we are interested in
def m : ℝ := P_X_gt_12
def n : ℝ := P_8_le_X_le_10
def expression := (2 / m) + (1 / n)

-- State the theorem to be proven
theorem min_value_expression : expression = 6 + 4*real.sqrt 2 := sorry

end min_value_expression_l701_701450


namespace increasing_interval_f_max_min_g_in_interval_num_ortho_vectors_l701_701707

noncomputable def f (x : Real) := 4 * Real.sin (2 * x + Real.pi / 3)
def a (x : Real) := (-1, f x)
def b (x : Real) := (f (-x), 1)
def g (x : Real) := a x.1 * b x.1 + a x.2 * b x.2

theorem increasing_interval_f (k : ℤ) :
  ∀ x : Real, x ∈ [k * Real.pi - 5 * Real.pi / 12, k * Real.pi + Real.pi / 12] → 0 < 2 :=
sorry

theorem max_min_g_in_interval :
  ∃ x : Real, x ∈ [Real.pi / 8, Real.pi / 3] ∧ (g x = 4 ∨ g x = 2 * Real.sqrt 2) :=
sorry

theorem num_ortho_vectors :
  ∃ n : ℕ, x ∈ [0, 2015 * Real.pi] ∧ g x = 0 → n = 4031 :=
sorry

end increasing_interval_f_max_min_g_in_interval_num_ortho_vectors_l701_701707


namespace spending_difference_is_65_l701_701775

-- Definitions based on conditions
def ice_cream_cones : ℕ := 15
def pudding_cups : ℕ := 5
def ice_cream_cost_per_unit : ℝ := 5
def pudding_cost_per_unit : ℝ := 2

-- The solution requires the calculation of the total cost and the difference
def total_ice_cream_cost : ℝ := ice_cream_cones * ice_cream_cost_per_unit
def total_pudding_cost : ℝ := pudding_cups * pudding_cost_per_unit
def spending_difference : ℝ := total_ice_cream_cost - total_pudding_cost

-- Theorem statement proving the difference is 65
theorem spending_difference_is_65 : spending_difference = 65 := by
  -- The proof is omitted with sorry
  sorry

end spending_difference_is_65_l701_701775


namespace data_groups_l701_701204

theorem data_groups (max_val min_val interval : ℕ) (h1 : max_val = 90) (h2 : min_val = 39) (h3 : interval = 10) : 
  ceil ((max_val - min_val) / interval) = 6 :=
by
  rw [h1, h2, h3]
  sorry

end data_groups_l701_701204


namespace c_120_eq_0_l701_701237

def c : ℕ → ℚ
| 1     := 2
| 2     := 1
| (n+3) := (1 - c (n+2)) / (c (n+1)^2 + 1)

theorem c_120_eq_0 : c 120 = 0 :=
sorry

end c_120_eq_0_l701_701237


namespace arithmetic_sequence_ninth_term_l701_701885

theorem arithmetic_sequence_ninth_term :
  ∃ a d : ℤ, (a + 2 * d = 23) ∧ (a + 5 * d = 29) ∧ (a + 8 * d = 35) :=
by
  sorry

end arithmetic_sequence_ninth_term_l701_701885


namespace geometric_sequence_a11_l701_701762

theorem geometric_sequence_a11
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h3 : a 3 = 4)
  (h7 : a 7 = 12) : 
  a 11 = 36 :=
by
  sorry

end geometric_sequence_a11_l701_701762


namespace number_of_swaps_independent_l701_701118

theorem number_of_swaps_independent (n : ℕ) (hn : n = 20) (p : Fin n → Fin n) :
    (∀ i, p i ≠ i → ∃ j, p j ≠ j ∧ p (p j) = j) →
    ∃ s : List (Fin n × Fin n), List.length s ≤ n ∧
    (∀ σ : List (Fin n × Fin n), (∀ (i j : Fin n), (i, j) ∈ σ → p i ≠ i → ∃ p', σ = (i, p') :: (p', j) :: σ) →
     List.length σ = List.length s) :=
  sorry

end number_of_swaps_independent_l701_701118


namespace number_of_female_officers_l701_701157

theorem number_of_female_officers (h1 : 0.19 * T = 76) (h2 : T = 152 / 2) : T = 400 :=
by
  sorry

end number_of_female_officers_l701_701157


namespace max_gcd_consecutive_terms_l701_701650

noncomputable def a (n : Nat) : Nat := Nat.factorial n + 3 * n

theorem max_gcd_consecutive_terms : 
  ∃ n, ∀ m, m ≥ 0 → gcd (a m) (a (m + 1)) = 3 :=
sorry

end max_gcd_consecutive_terms_l701_701650


namespace total_complaints_l701_701838

-- Conditions as Lean definitions
def normal_complaints : ℕ := 120
def short_staffed_20 (c : ℕ) := c + c / 3
def short_staffed_40 (c : ℕ) := c + 2 * c / 3
def self_checkout_partial (c : ℕ) := c + c / 10
def self_checkout_complete (c : ℕ) := c + c / 5
def day1_complaints : ℕ := normal_complaints + normal_complaints / 3 + normal_complaints / 5
def day2_complaints : ℕ := normal_complaints + 2 * normal_complaints / 3 + normal_complaints / 10
def day3_complaints : ℕ := normal_complaints + 2 * normal_complaints / 3 + normal_complaints / 5

-- Prove the total complaints
theorem total_complaints : day1_complaints + day2_complaints + day3_complaints = 620 :=
by
  sorry

end total_complaints_l701_701838


namespace fraction_of_white_surface_area_l701_701194

-- Define the conditions
def cube_edge_length : ℕ := 4
def smaller_cubes : ℕ := 64
def white_cubes : ℕ := 48
def black_cubes : ℕ := 16
def black_cubes_placement : Prop := 
  black_cubes = 8 ∧ ∀ cube, cube ∈ {corners, top_edge}

-- Define the theorem to be proven
theorem fraction_of_white_surface_area :
  let total_surface_area := 6 * cube_edge_length ^ 2
  let black_faces := 24 + 4  -- 8 corner cubes with 3 faces each + 4 top edge cubes with 1 face each
  let white_faces := total_surface_area - black_faces
  (white_faces : ℤ) / total_surface_area = 17 / 24 :=
by sorry

end fraction_of_white_surface_area_l701_701194


namespace last_four_digits_5_pow_2017_l701_701056

theorem last_four_digits_5_pow_2017 : (5 ^ 2017) % 10000 = 3125 :=
by sorry

end last_four_digits_5_pow_2017_l701_701056


namespace cos_neg_x_increasing_l701_701864

/- Define the function f: ℝ → ℝ where f(x) = cos(-x) -/
def f (x : ℝ) : ℝ := cos (-x)

/- Define the condition for the function to be increasing on (-∞, ∞) -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

/- State the theorem to prove that the function is increasing on (-∞, ∞) -/
theorem cos_neg_x_increasing : is_increasing f :=
by { sorry }

end cos_neg_x_increasing_l701_701864


namespace inverse_modulo_1000000_l701_701781

def A : ℕ := 123456
def B : ℕ := 769230
def N : ℕ := 1053

theorem inverse_modulo_1000000 : (A * B * N) % 1000000 = 1 := 
  by 
  sorry

end inverse_modulo_1000000_l701_701781


namespace parallel_line_slope_l701_701976

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l701_701976


namespace card_number_is_8_l701_701523

theorem card_number_is_8 (f : ℕ → ℕ) (h_periodic : ∀ i, f (i + 15) = f i) 
(h_consecutive_sum : ∀ i, f i + f (i + 1) + f (i + 2) + f (i + 3) + f (i + 4) + f (i + 5) = 50)
(a_left : ℕ) (a_right : ℕ) :
a_left = 7 → a_right = 10 → ∃ (x : ℕ), x = 8 :=
by {
  intros ha_left ha_right,
  -- Rest of the proof will be here
  sorry
}

end card_number_is_8_l701_701523


namespace sum_of_squares_of_zeros_l701_701090

def f : ℝ → ℝ :=
  λ x, if x ≠ 1 then 1 / |x - 1| else 1

def h (x : ℝ) := f x * f x + (-3 / 2) * f x + 1 / 2

noncomputable def zeros : fin 5 → ℝ :=
  ![1, 0, 2, -1, 3] -- Assume these are the zeros based on solution given

theorem sum_of_squares_of_zeros :
  ∑ i in finset.univ, (zeros i) ^ 2 = 15 :=
by
  simp [zeros]
  norm_num

end sum_of_squares_of_zeros_l701_701090


namespace largest_integer_in_list_l701_701195

noncomputable def max_possible_integer (l : List ℕ) : ℕ :=
  if h : l.length = 7 ∧ (∃ k, l.count 10 = k ∧ k > 1) ∧ (MedianList l = 12) ∧ (AverageList l = 15) then
    max 0 (l.last sorry)
  else
    0

theorem largest_integer_in_list (l : List ℕ) :
  l.length = 7 →
  (∃ k, l.count 10 = k ∧ k > 1) →
  MedianList l = 12 →
  AverageList l = 15 →
  max_possible_integer l = 31 := 
  by
    sorry

end largest_integer_in_list_l701_701195


namespace fraction_zero_l701_701143

theorem fraction_zero (x : ℝ) (h : x ≠ 3) : (2 * x^2 - 6 * x) / (x - 3) = 0 ↔ x = 0 := 
by
  sorry

end fraction_zero_l701_701143


namespace smallest_b_multiple_of_6_and_15_is_30_l701_701270

theorem smallest_b_multiple_of_6_and_15_is_30 : ∃ b : ℕ, b > 0 ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ b = 30 :=
by
  use 30
  split
  . trivial
  . split
    . sorry
    . sorry

end smallest_b_multiple_of_6_and_15_is_30_l701_701270


namespace zero_exists_in_interval_l701_701095

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem zero_exists_in_interval :
  ∃ (x : ℝ), 2 < x ∧ x < 3 ∧ f x = 0 :=
begin
  sorry
end

end zero_exists_in_interval_l701_701095


namespace incenter_distance_l701_701112

-- Definitions and conditions
def isosceles_right_triangle
  (D E F : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (DE DF : ℝ) (EF : ℝ) : Prop :=
∃ (D E F : Point), dist D E = DE ∧ dist D F = DF ∧ dist E F = EF ∧ 
∠DEF = 90

noncomputable def incenter_distance_EJ 
  (D E F : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (DE : ℝ) (angle_E : ℝ) (isosceles_right_triangle : isosceles_right_triangle D E F DE DE (6*sqrt 2))
  : ℝ :=
if isosceles_right_triangle ∧ angle_E = 90 then 6 - 3*sqrt 2 else 0

theorem incenter_distance 
  (D E F : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (DE : ℝ) (angle_E : ℝ) 
  (isosceles_right_triangle : isosceles_right_triangle D E F DE DE (6*sqrt 2)) :
  incenter_distance_EJ D E F DE angle_E isosceles_right_triangle = 6 - 3*sqrt 2 :=
by sorry

end incenter_distance_l701_701112


namespace curved_surface_area_cone_l701_701289

theorem curved_surface_area_cone :
  let r := 8  -- base radius in cm
  let l := 19  -- lateral edge length in cm
  let π := Real.pi
  let CSA := π * r * l
  477.5 < CSA ∧ CSA < 478 := by
  sorry

end curved_surface_area_cone_l701_701289


namespace sin_2alpha_eq_neg_24_over_25_l701_701678

theorem sin_2alpha_eq_neg_24_over_25 (α : ℝ) (h : sin α + cos α = 1 / 5) : sin (2 * α) = -24 / 25 := 
  sorry

end sin_2alpha_eq_neg_24_over_25_l701_701678


namespace smallest_multiple_of_6_and_15_l701_701265

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ b' : ℕ, b' > 0 ∧ b' % 6 = 0 ∧ b' % 15 = 0 → b ≤ b' :=
sorry

end smallest_multiple_of_6_and_15_l701_701265


namespace investment_difference_l701_701905

theorem investment_difference (x y z : ℕ) 
  (h1 : x + (x + y) + (x + 2 * y) = 9000)
  (h2 : (z / 9000) = (800 / 1800)) 
  (h3 : z = x + 2 * y) :
  y = 1000 := 
by
  -- omitted proof steps
  sorry

end investment_difference_l701_701905


namespace solution_set_of_inequality_l701_701699

def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 4*x else x^2 + 4*x

theorem solution_set_of_inequality :
  {x : ℝ | f (2*x + 3) ≤ 5} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end solution_set_of_inequality_l701_701699


namespace point_symmetric_property_l701_701301

def point := (ℝ × ℝ)

def symmetric_with_respect_to_x_axis (p : point) : point :=
  (p.1, -p.2)

theorem point_symmetric_property :
  let P : point := (-2, 1)
  let Q : point := symmetric_with_respect_to_x_axis P
  Q = (-2, -1) :=
by
  intros
  simp [symmetric_with_respect_to_x_axis]
  sorry

end point_symmetric_property_l701_701301


namespace log3_intersects_x_axis_l701_701481

theorem log3_intersects_x_axis : ∃ x : ℝ, x > 0 ∧ log 3 x = 0 ∧ (x = 1) := 
by {
  -- Given y = log_3 x, we want to prove that it intersects the x-axis at (1, 0)
  sorry
}

end log3_intersects_x_axis_l701_701481


namespace find_polyhedron_l701_701587

-- Define the required variables and conditions:
variables (c l e : ℕ) -- c: vertices, l: faces, e: edges
def is_geometric_progression (c l e : ℕ) := (l : ℝ) / (c : ℝ) = 3 / 2 ∧ (e : ℝ) / (l : ℝ) = 3 / 2

-- Translate Euler's formula into Lean.
def euler_formula (c l e : ℕ) := c + l = e + 2

-- Define a Lean instance for isosceles triangle condition given in problem.
def is_congruent_isosceles_triangles (k sides base : ℕ) := 
  ∀ f : ℕ, f < k → f + 2 = sides + base -- arbitrary condition simplification for isosceles triangles

-- Problem statement according to conditions and question:
theorem find_polyhedron :
  is_congruent_isosceles_triangles l 2 1 ∧
  is_geometric_progression c l e ∧
  euler_formula c l e → 
  c = 8 ∧ l = 12 ∧ e = 18 :=
begin
  sorry
end

end find_polyhedron_l701_701587


namespace dice_probability_sum_leq_4_l701_701556

theorem dice_probability_sum_leq_4 :
  let total_outcomes := 36 in
  let favorable_outcomes := 6 in
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 6 :=
by
  sorry

end dice_probability_sum_leq_4_l701_701556


namespace polynomial_division_l701_701920

noncomputable def p : Polynomial ℚ :=
  10 * X ^ 4 + 5 * X ^ 3 - 8 * X ^ 2 + 7 * X - 3

noncomputable def d : Polynomial ℚ :=
  3 * X + 2

noncomputable def q : Polynomial ℚ :=
  (10 / 3) * X ^ 3 - (5 / 9) * X ^ 2 - (31 / 27) * X + (143 / 81)

theorem polynomial_division :
  Polynomial.quotient p d = q :=
sorry

end polynomial_division_l701_701920


namespace maclaurin_inequality_l701_701792

variables {x : ℕ → ℝ} {n : ℕ}
noncomputable def sigma (k : ℕ) : ℝ := (Finset.range n).sum (λ s, (x s))

theorem maclaurin_inequality
    (h_pos : ∀ (i : ℕ), i < n → 0 < x i)
    (h_n : 2 ≤ n) :
    sigma 1 ≥ real.sqrt (sigma 2) ∧ real.sqrt (sigma 2) ≥ real.cbrt (sigma 3) ∧ ... ∧ real.root n (sigma n) :=
sorry

end maclaurin_inequality_l701_701792


namespace existence_of_similar_triangles_l701_701769

noncomputable def similarTrianglesExist : Prop :=
  ∃ (X₁ X₂ Y₁ Y₂ Z₁ Z₂ : ℝ × ℝ),
    (∀ i j k : Fin 2, sim (triangle (X₁, Y₁, Z₁)) (triangle ((X₁, Y₁, Z₁)[i], (X₁, Y₁, Z₁)[j], (X₁, Y₁, Z₁)[k])))

theorem existence_of_similar_triangles : similarTrianglesExist := sorry

end existence_of_similar_triangles_l701_701769


namespace number_of_arithmetic_sequences_l701_701643

theorem number_of_arithmetic_sequences : 
  ∃ n : ℕ, n = 16 ∧
  ∀ (a b c : ℕ), (a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ c ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) →
  a < b → b < c →
  (b - a = c - b) →
  {a, b, c} ∈ arithmetic_sequences_from_digits 1 9
  :=
by sorry

def arithmetic_sequences_from_digits (start end_ : ℕ) : finset (finset ℕ) :=
  { S ∈ powerset (finset.range' start (end_ - start + 1)) | S.card = 3 ∧ 
  ∃ (a b c : ℕ), 
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
  b - a = c - b }

end number_of_arithmetic_sequences_l701_701643


namespace circle_equation_exists_l701_701858

theorem circle_equation_exists :
  ∃ D E F : ℝ, (∀ (x y : ℝ), (x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
                      (x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
                      (x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
                      (D = -4) ∧ (E = -6) ∧ (F = 0) :=
by
  use [-4, -6, 0]
  intro x y
  split
  { intros hx hy, simp [hx, hy] }
  split
  { intros hx hy, simp [hx, hy], linarith }
  { intros hx hy, simp [hx, hy], linarith }
  sorry

end circle_equation_exists_l701_701858


namespace percentage_calculation_l701_701351

theorem percentage_calculation 
  (number : ℝ)
  (h1 : 0.035 * number = 700) :
  0.024 * (1.5 * number) = 720 := 
by
  sorry

end percentage_calculation_l701_701351


namespace sum_of_sequences_contains_repetition_l701_701051

open Finset

/-- Define the alphabet positions from 1 to 26 -/
def alphabet_positions : Finset ℕ := range 27

noncomputable def sum_sequences (seq1 seq2 : Fin 27 → ℕ) : Fin 27 → ℕ :=
  λ i => ((seq1 i) + (seq2 i)) % 26+1

theorem sum_of_sequences_contains_repetition (seq1 seq2 : Fin 27 → ℕ)
  (h_seq1 : ∀ i, seq1 i ∈ alphabet_positions) 
  (h_seq2 : ∀ i, seq2 i ∈ alphabet_positions) 
  (h_distinct : injective seq1) :
  ¬ injective (sum_sequences seq1 seq2) :=
sorry

end sum_of_sequences_contains_repetition_l701_701051


namespace sum_of_fractions_l701_701629

theorem sum_of_fractions (a b c d : ℚ) (ha : a = 2 / 5) (hb : b = 3 / 8) :
  (a + b = 31 / 40) :=
by
  sorry

end sum_of_fractions_l701_701629


namespace yarn_total_length_l701_701868

variable (green_length : ℕ) (red_length : ℕ) (blue_length : ℕ) (total_length : ℕ)

def green_length_def : green_length = 156 := sorry
def red_length_def : red_length = 3 * green_length + 8 := sorry
def blue_length_def : blue_length = (green_length + red_length) / 2 := sorry
def total_length_def : total_length = green_length + red_length + blue_length := sorry

theorem yarn_total_length : total_length = 948 := 
by 
  rw [total_length_def, blue_length_def, red_length_def, green_length_def]
  simp
  sorry

end yarn_total_length_l701_701868


namespace page_mistakenly_added_twice_l701_701513

theorem page_mistakenly_added_twice (n k: ℕ) (h₁: n = 77) (h₂: (n * (n + 1)) / 2 + k = 3050) : k = 47 :=
by
  -- sorry here to indicate the proof is not needed
  sorry

end page_mistakenly_added_twice_l701_701513


namespace students_taking_test_paper_C_l701_701202

theorem students_taking_test_paper_C (h1 : ∀ n, 1 ≤ n → n ≤ 40 → ∃ s, s = 30 * n - 2 ∧ 1 ≤ s ∧ s ≤ 1200) :
  (∃ k, k = ∑ i in finset.Icc 26 40, 1) ∧ 
  k = 15 :=
by {
  have group_bounds : 26 ≤ 40 := by norm_num,
  let num_students := 40,
  sorry
}

end students_taking_test_paper_C_l701_701202


namespace jane_spent_more_on_ice_cream_l701_701774

-- Definitions based on the conditions
def ice_cream_cone_cost : ℕ := 5
def pudding_cup_cost : ℕ := 2
def ice_cream_cones_bought : ℕ := 15
def pudding_cups_bought : ℕ := 5

-- The mathematically equivalent proof statement
theorem jane_spent_more_on_ice_cream : 
  (ice_cream_cones_bought * ice_cream_cone_cost - pudding_cups_bought * pudding_cup_cost) = 65 := 
by
  sorry

end jane_spent_more_on_ice_cream_l701_701774


namespace binomial_variance_example_l701_701359

noncomputable def variance {n : ℕ} {p : ℝ} (X : ℕ → ℝ) [Binomial X n p] : ℝ :=
  n * p * (1 - p)

theorem binomial_variance_example :
  let X : ℕ → ℝ := sorry in
  ∀ (X : ℕ → ℝ), Binomial X 10 (2 / 3) → variance X = 20 / 9 :=
by
  intros X hXh
  rw [variance, hXh]
  sorry

end binomial_variance_example_l701_701359


namespace arithmetic_sequence_ninth_term_l701_701890

theorem arithmetic_sequence_ninth_term
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29) :
  a + 8 * d = 35 :=
sorry

end arithmetic_sequence_ninth_term_l701_701890


namespace Alice_winning_strategy_l701_701218

-- Define the conditions of the game
variable (m n : ℕ)

-- Define what it means for Alice to have a winning strategy
-- using the condition derived from the problem
theorem Alice_winning_strategy :
  ¬((n + 1) ∣ m) → ∃ k (1 ≤ k ≤ m), (m - k) % (n + 1) ≠ 0 :=
begin
  sorry
end

end Alice_winning_strategy_l701_701218


namespace part1_part2_l701_701325

-- Part (1)
def f (x : ℝ) (a : ℝ) : ℝ := real.exp x + 2 * real.exp (-x) + a
def g (x : ℝ) : ℝ := x^2 - x + 1
def tangent_line (m x : ℝ) : ℝ := m * x + 1

theorem part1 (a : ℝ) :
  (∃ x₀ m, (f x₀ a = tangent_line m x₀) ∧ (f'.eval x₀ = m) ∧ (g x₀ = tangent_line m x₀) ∧ (g'.eval x₀ = m)) ↔ a = -2 :=
sorry

-- Part (2)
theorem part2 (x₁ x₂ k : ℝ) 
  (h₁ : real.exp x₁ + real.exp x₂ = 3)
  (h₂ : f x₁ -2 * f x₂ -2 ≥ 3 * (x₁ + x₂ + k)) :
  k ≤ (25/108 - 2 * real.log (3 / 2)) :=
sorry

end part1_part2_l701_701325


namespace no_shaded_area_l701_701605

structure Point :=
(x : ℝ)
(y : ℝ)

structure Square :=
(lower_right : Point)
(side_length : ℝ)

structure RightTriangle :=
(lower_left : Point)
(leg_length : ℝ)

def is_shaded_region_zero (square : Square) 
                          (triangle : RightTriangle) 
                          (line_from_square : Point) 
                          (line_to_triangle : Point) : Prop :=
  let A := Point.mk (square.lower_right.x - square.side_length) (square.lower_right.y + square.side_length) in
  let E := Point.mk (triangle.lower_left.x + triangle.leg_length) (triangle.lower_left.y + triangle.leg_length) in
  (line_from_square = A) → (line_to_triangle = E) → 0 = 0

theorem no_shaded_area : 
  ∀ (square : Square) (triangle : RightTriangle) 
  (line_from_square : Point) (line_to_triangle : Point),
  square.lower_right = Point.mk 20 0 → 
  square.side_length = 10 →
  triangle.lower_left = Point.mk 20 0 →
  triangle.leg_length = 10 →
  is_shaded_region_zero square triangle line_from_square line_to_triangle := by
  intros
  apply fun A E => rfl
  sorry

end no_shaded_area_l701_701605


namespace number_of_moles_NaOH_l701_701260

/-- Given a balanced chemical equation NaH + H2O → NaOH + H2,
    where 3 moles of NaH react with 3 moles of H2O,
    prove that the number of moles of NaOH formed is 3. -/
theorem number_of_moles_NaOH (NaH H2O NaOH H2 : Type) 
  (r1 : NaH → H2O → NaOH → H2 → Prop)
  (three_moles_NaH : ∀ (a : NaH), a = 3) 
  (three_moles_H2O : ∀ (b : H2O), b = 3)
  (one_to_one_to_one_ratio : ∀ (a : NaH) (b : H2O) (c : NaOH) (d : H2), 
      r1 a b c d → a = b ∧ b = c): 
  ∃ (c : NaOH), c = 3 :=
by
  sorry

end number_of_moles_NaOH_l701_701260


namespace find_S9_l701_701235

-- Setting up basic definitions for arithmetic sequence and the sum of its terms
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d
def sum_arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := n * (a + arithmetic_seq a d n) / 2

-- Given conditions
variables (a d : ℤ)
axiom h : 2 * arithmetic_seq a d 3 = 3 + a

-- Theorem to prove
theorem find_S9 : sum_arithmetic_seq a d 9 = 27 :=
by {
  sorry
}

end find_S9_l701_701235


namespace range_a_l701_701322

def f(x : ℝ) : ℝ := 
  if x ≤ 0 then 4 + x else x^2  

theorem range_a (a : ℝ) (h : f[f a] > f[f a + 1]) : a ∈ Ioc (-5) (-4) :=
sorry

end range_a_l701_701322


namespace max_value_y_eq_x_mul_2_minus_x_min_value_y_eq_x_plus_4_div_x_minus_3_l701_701582

theorem max_value_y_eq_x_mul_2_minus_x (x : ℝ) (h : 0 < x ∧ x < 3 / 2) : ∃ y : ℝ, y = x * (2 - x) ∧ y ≤ 1 :=
sorry

theorem min_value_y_eq_x_plus_4_div_x_minus_3 (x : ℝ) (h : x > 3) : ∃ y : ℝ, y = x + 4 / (x - 3) ∧ y ≥ 7 :=
sorry

end max_value_y_eq_x_mul_2_minus_x_min_value_y_eq_x_plus_4_div_x_minus_3_l701_701582


namespace calc_f_r_sub_f_r_minus_1_l701_701350

-- Define the function f
def f (n : ℕ) : ℚ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

-- State the theorem to prove
theorem calc_f_r_sub_f_r_minus_1 (r : ℕ) :
  f(r) - f(r-1) = r * (r+1) * (r+2) :=
by
  -- Proof goes here
  sorry

end calc_f_r_sub_f_r_minus_1_l701_701350


namespace complete_laps_l701_701798

-- Definitions based on conditions
def total_distance := 3.25  -- total distance Lexi wants to run
def lap_distance := 0.25    -- distance of one lap

-- Proof statement: Total number of complete laps to cover the given distance
theorem complete_laps (h1 : total_distance = 3 + 1/4) (h2 : lap_distance = 1/4) :
  (total_distance / lap_distance) = 13 :=
by 
  sorry

end complete_laps_l701_701798


namespace similar_rectangles_area_l701_701601

noncomputable def area_of_similar_rectangle (a₁ a₂ d₂ : ℝ) : ℝ :=
  let ratio := a₂ / a₁ in
  let a₂ := Math.sqrt (d₂ ^ 2 / (1 + ratio ^ 2)) in
  let b₂ := ratio * a₂ in
  a₂ * b₂

theorem similar_rectangles_area
  (a₁ a₂ d₂ : ℝ)
  (h₁ : a₁ = 3)
  (h₂ : a₁ * a₂ = 21)
  (h₃ : d₂ = 20) :
  area_of_similar_rectangle a₁ a₂ d₂ = 144.6 :=
by
  unfold area_of_similar_rectangle
  rw [h₁, h₂, h₃]
  sorry

end similar_rectangles_area_l701_701601


namespace slope_parallel_to_original_line_l701_701940

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l701_701940


namespace Portia_students_l701_701060

-- Define the problem conditions as Lean definitions
def Lara_high_school_students : ℕ := L
def Portia_high_school_students : ℕ := P

-- Assume the given conditions
axiom condition1 : P = 4 * L
axiom condition2 : P + L = 3000

-- State the problem: Prove that P = 2400 given the conditions
theorem Portia_students : Portia_high_school_students = 2400 :=
by
  sorry

end Portia_students_l701_701060


namespace decreasing_function_inequality_l701_701692

variable {R : Type*} [OrderedAddCommGroup R]

-- Assuming f is a decreasing function
variable {f : R → R} (h_decreasing : ∀ x y : R, x ≤ y → f y ≤ f x)

-- Given condition a + b ≤ 0
variable (a b : R) (h_le : a + b ≤ 0)

theorem decreasing_function_inequality : f(a) + f(b) ≥ f(-a) + f(-b) :=
sorry

end decreasing_function_inequality_l701_701692


namespace Jacob_eats_more_calories_than_planned_l701_701405

theorem Jacob_eats_more_calories_than_planned 
  (planned_calories : ℕ) (actual_calories : ℕ)
  (h1 : planned_calories < 1800) 
  (h2 : actual_calories = 400 + 900 + 1100)
  : actual_calories - planned_calories = 600 := by
  sorry

end Jacob_eats_more_calories_than_planned_l701_701405


namespace pos_int_divides_l701_701256

theorem pos_int_divides (n : ℕ) (h₀ : 0 < n) (h₁ : (n - 1) ∣ (n^3 + 4)) : n = 2 ∨ n = 6 :=
by sorry

end pos_int_divides_l701_701256


namespace min_abs_diff_l701_701835

theorem min_abs_diff
(a b : ℕ) -- positive integers
(h : a * b - 4 * a + 3 * b = 475):
∃ (a b : ℕ), (a * b - 4 * a + 3 * b = 475) ∧ |a - b| = 455 :=
begin
  sorry
end

end min_abs_diff_l701_701835


namespace lengths_of_diagonals_and_t_value_l701_701378

-- Define the coordinates of points A, B, and C
def A : ℝ × ℝ := ⟨-1, -2⟩
def B : ℝ × ℝ := ⟨2, 3⟩
def C : ℝ × ℝ := ⟨-2, -1⟩

-- Define vectors AB and AC
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def OC : ℝ × ℝ := C

-- Define dot product for pairs of real numbers
def dot (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define magnitudes of vectors AB + AC and AB - AC
def len_diag1 : ℝ := Real.sqrt ((AB.1 + AC.1)^2 + (AB.2 + AC.2)^2)
def len_diag2 : ℝ := Real.sqrt ((AB.1 - AC.1)^2 + (AB.2 - AC.2)^2)

-- Define expression for t
def t_eqn (t : ℝ) : ℝ × ℝ := (AB.1 + t * OC.1, AB.2 + t * OC.2)

-- Lean statement for the proof problem
theorem lengths_of_diagonals_and_t_value :
  len_diag1 = 2 * Real.sqrt(10) ∧ len_diag2 = 4 * Real.sqrt(2) ∧
  (∃ t : ℝ, dot (AB.1 - t * OC.1, AB.2 - t * OC.2) OC = 0 ∧ t = -11 / 5) :=
by
  sorry

end lengths_of_diagonals_and_t_value_l701_701378


namespace width_of_grass_field_l701_701603

noncomputable def grassFieldWidth (path_width field_length total_path_cost cost_per_sq_m total_length total_width : ℝ): ℝ :=
let area_of_path := total_path_cost / cost_per_sq_m in
let eq := 100 * (total_width + path_width) - field_length * total_width = area_of_path in
if h : eq then
    total_width
else
    0 -- placeholder for actual solution (should not occur if all values are correct)

theorem width_of_grass_field :
  ∀ (path_width field_length total_path_cost cost_per_sq_m total_length total_width width_of_grass_field : ℝ),
    path_width = 2.5 →
    field_length = 95 →
    total_path_cost = 1550 →
    cost_per_sq_m = 2 →
    total_length = field_length + 2 * path_width →
    total_width + 2 * path_width = 55 →
    grassFieldWidth path_width field_length total_path_cost cost_per_sq_m total_length total_width = 55 :=
begin
  intros,
  sorry
end

end width_of_grass_field_l701_701603


namespace collinear_X_Y_Z_l701_701368

variables {A B C D E F X Y Z : Type}

-- Assuming essential definitions and properties for triangle and midpoints
variables [is_triangle A B C] [midpoint D B C] [midpoint E C A] [midpoint F A B]

theorem collinear_X_Y_Z (h₁ : non_isosceles_triangle A B C) (h₂ : tangent_to_incircle_at D X EF) 
  (h₃ : tangent_to_incircle_at E Y DF) (h₄ : tangent_to_incircle_at F Z DE) :
  collinear X Y Z := 
sorry

end collinear_X_Y_Z_l701_701368


namespace find_point_B_l701_701758

-- Define the points as given in the problem.
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := -2 }
def C : Point := { x := 4, y := 1 }

-- Define the properties of the square.
def isSquare (A B C D : Point) : Prop :=
  (A.x - B.x) ^ 2 + (A.y - B.y) ^ 2 = (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 ∧
  (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 = (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 ∧
  (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 = (D.x - A.x) ^ 2 + (D.y - A.y) ^ 2 ∧
  (A.x - B.x) * (B.x - C.x) + (A.y - B.y) * (B.y - C.y) = 0

-- Given that AB is parallel to x-axis and AD is parallel to y-axis.
def AB_parallel_x (A B : Point) : Prop := A.y = B.y

def BC_parallel_y (B C : Point) : Prop := B.x = C.x

-- The coordinates of point B we need to prove.
def B : Point := { x := 4, y := -2 }

-- Prove that point B satisfies the conditions of the problem.
theorem find_point_B
  (A C B : Point)
  (h1 : A = { x := 1, y := -2 })
  (h2 : C = { x := 4, y := 1 })
  (hsquare : isSquare A B C B)
  (hab : AB_parallel_x A B)
  (hbc : BC_parallel_y B C) :
  B = { x := 4, y := -2 } :=
begin
  sorry
end

end find_point_B_l701_701758


namespace prob_A_second_day_is_correct_l701_701625

-- Definitions for the problem conditions
def prob_first_day_A : ℝ := 0.5
def prob_A_given_A : ℝ := 0.6
def prob_first_day_B : ℝ := 0.5
def prob_A_given_B : ℝ := 0.5

-- Calculate the probability of going to A on the second day
def prob_A_second_day : ℝ :=
  prob_first_day_A * prob_A_given_A + prob_first_day_B * prob_A_given_B

-- The theorem statement
theorem prob_A_second_day_is_correct : 
  prob_A_second_day = 0.55 :=
by
  unfold prob_A_second_day prob_first_day_A prob_A_given_A prob_first_day_B prob_A_given_B
  sorry

end prob_A_second_day_is_correct_l701_701625


namespace sum_of_numbers_l701_701160

theorem sum_of_numbers (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : ab + bc + ca = 100) :
  a + b + c = 21 :=
sorry

end sum_of_numbers_l701_701160


namespace point_lies_on_hyperbola_l701_701275

noncomputable def point_on_curve (s : ℝ) (hs : s ≠ 0) : ℝ × ℝ :=
  ( (s + 2) / (s - 1), (s - 2) / (s + 1) )

theorem point_lies_on_hyperbola {s : ℝ} (hs : s ≠ 0) :
  let (x, y) := point_on_curve s hs
  in ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * (x * x) - b * (y * y) = 1 :=
sorry

end point_lies_on_hyperbola_l701_701275


namespace extra_large_yellow_curlers_l701_701647

def total_curlers : ℕ := 120
def small_pink_curlers : ℕ := total_curlers / 5
def medium_blue_curlers : ℕ := 2 * small_pink_curlers
def large_green_curlers : ℕ := total_curlers / 4

theorem extra_large_yellow_curlers : 
  total_curlers - small_pink_curlers - medium_blue_curlers - large_green_curlers = 18 :=
by
  sorry

end extra_large_yellow_curlers_l701_701647


namespace segments_form_right_triangle_l701_701005

theorem segments_form_right_triangle
  {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (BC AD AC : ℝ) 
  (angle_CBD_eq_angle_CAB : ∀ {A B C D: Type}, metric.angle (metric.sphere_angle A B C) D = metric.angle (metric.sphere_angle A B D) C)
  (angle_ACD_eq_angle_ADB : ∀ {A B C D: Type}, metric.angle (metric.sphere_angle A C D) B = metric.angle (metric.sphere_angle A D B) C) :
  BC * BC + AD * AD = AC * AC := 
  sorry

end segments_form_right_triangle_l701_701005


namespace surjective_and_condition_l701_701255

noncomputable def f : matrix (fin n) (fin n) ℝ → fin (n + 1) := λ X, fin.of_nat' (rank X)

theorem surjective_and_condition (f : matrix (fin n) (fin n) ℝ → fin (n + 1)) :
  surjective f ∧ (∀ (X Y : matrix (fin n) (fin n) ℝ), f (X * Y) ≤ min (f X) (f Y)) ↔ 
  (∀ (X : matrix (fin n) (fin n) ℝ), f X = fin.of_nat' (rank X)) :=
sorry

end surjective_and_condition_l701_701255


namespace non_empty_solution_set_l701_701349

theorem non_empty_solution_set (a : ℝ) (h : a > 0) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 :=
by
  sorry

end non_empty_solution_set_l701_701349


namespace four_digit_numbers_count_four_digit_numbers_adjacent_evens_four_digit_numbers_non_adjacent_evens_l701_701676

theorem four_digit_numbers_count :
  let evens := {4, 6, 8}
  let odds := {3, 5, 7}
  let total_count := (evens.card.choose 2) * (odds.card.choose 2) * (4.factorial)
  in total_count = 216 :=
by
  sorry

theorem four_digit_numbers_adjacent_evens :
  let evens := {4, 6, 8}
  let odds := {3, 5, 7}
  let count_adjacent := (evens.card.choose 2) * (odds.card.choose 2) * (2.factorial * 3.permute 3)
  in count_adjacent = 108 :=
by 
  sorry

theorem four_digit_numbers_non_adjacent_evens :
  let evens := {4, 6, 8}
  let odds := {3, 5, 7}
  let count_non_adjacent := (evens.card.choose 2) * (odds.card.choose 2) * (3.permute 3) * (2.factorial)
  in count_non_adjacent = 54 :=
by 
  sorry

end four_digit_numbers_count_four_digit_numbers_adjacent_evens_four_digit_numbers_non_adjacent_evens_l701_701676


namespace parallel_line_slope_l701_701983

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l701_701983


namespace pool_capacity_total_is_correct_l701_701599

noncomputable def section_capacity_A_fill {
  initial_percent_A : ℕ := 60
  final_percent_A : ℕ := 70
  water_added_A : ℕ := 300
  percent_increase_A : ℕ := final_percent_A - initial_percent_A
  capacity_A := water_added_A * 100 / percent_increase_A
}

noncomputable def section_capacity_B_fill {
  initial_percent_B : ℕ := 50
  final_percent_B : ℕ := 65
  water_added_B : ℕ := 350
  percent_increase_B : ℕ := final_percent_B - initial_percent_B
  capacity_B := water_added_B * 100 / percent_increase_B
}

noncomputable def section_capacity_C_fill {
  initial_percent_C : ℕ := 40
  final_percent_C : ℕ := 60
  water_added_C : ℕ := 400
  percent_increase_C : ℕ := final_percent_C - initial_percent_C
  capacity_C := water_added_C * 100 / percent_increase_C
}

noncomputable def total_pool_capacity {
  capacity_A := section_capacity_A_fill.capacity_A
  capacity_B := section_capacity_B_fill.capacity_B
  capacity_C := section_capacity_C_fill.capacity_C
  total_capacity := capacity_A + capacity_B + capacity_C
}

theorem pool_capacity_total_is_correct {
  total_capacity := total_pool_capacity
  total_capacity = 7333 := sorry
}

end pool_capacity_total_is_correct_l701_701599


namespace largest_average_speed_second_hour_l701_701093

-- Define the problem conditions
def distance (t : ℕ) : ℝ := sorry -- The distance function based on time t, to be provided

-- Define the time intervals we're considering
def time_intervals := { 0, 1, 2, 3, 8, 11 }

-- Calculate the average speed (slope) for a given time interval
def average_speed (d : ℕ → ℝ) (t1 t2 : ℕ) : ℝ :=
  (d t2 - d t1) / (t2 - t1)

-- Prove that the second hour has the largest average speed
theorem largest_average_speed_second_hour :
  ∀ d : ℕ → ℝ, ∀ t1 t2 ∈ time_intervals, t1 < t2 →
  average_speed d 1 2 > average_speed d t1 t2 :=
by
  intros d t1 t2 itv1 itv2 h
  sorry

end largest_average_speed_second_hour_l701_701093


namespace students_in_Johnsons_class_l701_701804

theorem students_in_Johnsons_class :
  ∀ (students_Finley : ℕ) (students_Johnson : ℕ),
    students_Finley = 24 →
    students_Johnson = (students_Finley / 2) + 10 →
    students_Johnson = 22 :=
by
  intros students_Finley students_Johnson hFinley hJohnson
  rw [hFinley, Nat.div_add_self 12 0],
  exact hJohnson

sorry

end students_in_Johnsons_class_l701_701804


namespace original_cost_price_l701_701054

-- Define the conditions
def selling_price : ℝ := 24000
def discount_rate : ℝ := 0.15
def tax_rate : ℝ := 0.02
def profit_rate : ℝ := 0.12

-- Define the necessary calculations
def discounted_price (sp : ℝ) (dr : ℝ) : ℝ := sp * (1 - dr)
def total_tax (sp : ℝ) (tr : ℝ) : ℝ := sp * tr
def profit (c : ℝ) (pr : ℝ) : ℝ := c * (1 + pr)

-- The problem is to prove that the original cost price is $17,785.71
theorem original_cost_price : 
  ∃ (C : ℝ), C = 17785.71 ∧ 
  selling_price * (1 - discount_rate - tax_rate) = (1 + profit_rate) * C :=
sorry

end original_cost_price_l701_701054


namespace part1_part2_l701_701302

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

def p (m : ℝ) : Prop :=
  let Δ := discriminant 1 m 1
  Δ > 0 ∧ -m / 2 < 0

def q (m : ℝ) : Prop :=
  let Δ := discriminant 4 (4 * (m - 2)) 1
  Δ < 0

theorem part1 (m : ℝ) (hp : p m) : m > 2 := 
sorry

theorem part2 (m : ℝ) (h : ¬(p m ∧ q m) ∧ (p m ∨ q m)) : (m ≥ 3) ∨ (1 < m ∧ m ≤ 2) := 
sorry

end part1_part2_l701_701302


namespace domain_of_log_func_l701_701841

theorem domain_of_log_func : {x : ℝ | 0 < x} = {x : ℝ | ∃ y, y = log 2 (2^x - 1)} :=
by sorry

end domain_of_log_func_l701_701841


namespace alternating_series_value_l701_701631

open Nat

theorem alternating_series_value : 
  let s := ∑ k in range 2023, ((-1) ^ k) * (k + 1)
  s = -1012 :=
by
  let s := ∑ k in range 2023, ((-1) ^ k) * (k + 1)
  show s = -1012
  sorry

end alternating_series_value_l701_701631


namespace Jacob_eats_more_calories_than_planned_l701_701406

theorem Jacob_eats_more_calories_than_planned 
  (planned_calories : ℕ) (actual_calories : ℕ)
  (h1 : planned_calories < 1800) 
  (h2 : actual_calories = 400 + 900 + 1100)
  : actual_calories - planned_calories = 600 := by
  sorry

end Jacob_eats_more_calories_than_planned_l701_701406


namespace angle_four_correct_l701_701621

theorem angle_four_correct : 
  ∀ (angle1 angle2 angle3 angle4 : ℝ),
  angle1 = 100 ∧ angle2 = 60 ∧ angle3 = 90 ∧ 
  (angle1 + angle2 + angle3 + angle4 = 360) → 
  angle4 = 110 :=
by
  intros angle1 angle2 angle3 angle4 h
  cases h with h1 h2
  cases h1 with h1_100 h2_60
  cases h2 with h2_90 h_angle_sum
  have : angle1 + angle2 + angle3 + angle4 = 360 := h_angle_sum
  have : angle1 = 100 := h1_100
  have : angle2 = 60 := h2_60
  have : angle3 = 90 := h2_90
  sorry

end angle_four_correct_l701_701621


namespace minimum_value_of_f_l701_701509

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 2 → f x ≥ 4) ∧ (∃ x : ℝ, x > 2 ∧ f x = 4) :=
by {
  sorry
}

end minimum_value_of_f_l701_701509


namespace find_x_plus_y_l701_701104

theorem find_x_plus_y
  (x y : ℝ)
  (hx : x^3 - 3 * x^2 + 5 * x - 17 = 0)
  (hy : y^3 - 3 * y^2 + 5 * y + 11 = 0) :
  x + y = 2 := 
sorry

end find_x_plus_y_l701_701104


namespace length_KL_eq_one_fourth_l701_701490

section Pentagon

variable {A B C D E P Q R S K L : Type}
variable [AddCommGroup P] [Module ℕ P]

-- Assuming all sides of the pentagon are of length 1
variable (length_AB length_BC length_CD length_DE length_AE : ℕ := 1)

-- Define midpoints assumptions
variable (is_midpoint_AB : P = midpoint A B)
variable (is_midpoint_BC : Q = midpoint B C)
variable (is_midpoint_CD : R = midpoint C D)
variable (is_midpoint_DE : S = midpoint D E)

-- Define KL midpoints
variable (is_midpoint_PR : K = midpoint P R)
variable (is_midpoint_QS : L = midpoint Q S)

-- Prove the length of segment KL
theorem length_KL_eq_one_fourth :
  dist K L = 1 / 4 :=
sorry

end Pentagon

end length_KL_eq_one_fourth_l701_701490


namespace simplify_and_evaluate_expression_l701_701467

theorem simplify_and_evaluate_expression : 
  let a := Real.tan (Real.pi / 3)
  let b := Real.sin (Real.pi / 3)
  (b^2 + a^2) / a - 2 * b) / (1 - b / a) = (Real.sqrt 3) / 2 := 
by
  sorry

end simplify_and_evaluate_expression_l701_701467


namespace parallel_slope_l701_701931

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l701_701931


namespace integer_part_sum_l701_701106

noncomputable def x_seq (n : ℕ) : ℚ :=
  Nat.recOn n (1 / 2) (fun k x_k => x_k^2 + x_k)

theorem integer_part_sum :
  let S := ∑ k in Finset.range 2009, 1 / (x_seq k + 1)
  ⌊S⌋ = 1 :=
by
  let S := ∑ k in Finset.range 2009, 1 / (x_seq k + 1)
  sorry

end integer_part_sum_l701_701106


namespace area_of_triangle_ABC_l701_701087

theorem area_of_triangle_ABC :
  ∀ (ABC : Type) (a b c : ABC),
  ∀ (area_shaded : ℝ),
  (equilateral_triangle ABC a b c) →
  (congruent_triangles_copies ABC a b c) →
  (sliding_distance ABC a b c = (1 / 8) * distance a b) →
  (area_shaded = 300) →
  area ABC a b c = 768 := by
  sorry

-- Definitions to support the theorem statement.
def equilateral_triangle (ABC : Type) (a b c : ABC) : Prop := sorry
def congruent_triangles_copies (ABC : Type) (a b c : ABC) : Prop := sorry
def sliding_distance (ABC : Type) (a b c : ABC) : ℝ := sorry
def distance (a b : ABC) : ℝ := sorry
def area (ABC : Type) (a b c : ABC) : ℝ := sorry

end area_of_triangle_ABC_l701_701087


namespace dot_product_of_AC_BD_l701_701473

variables (AB CD : ℝ) (perpendicular : Prop)
variables (a b : EuclideanSpace ℝ (Fin 2))
variables (AC BD : EuclideanSpace ℝ (Fin 2))

-- Defining the given conditions
def conditions (AB CD : ℝ) (perpendicular : a ⬝ b = 0) : Prop :=
  AB = 65 ∧ CD = 31 ∧ perpendicular

theorem dot_product_of_AC_BD
  (h : conditions AB CD (a ⬝ b = 0)):
  (a - (31 / 65) • b) ⬝ (b - (31 / 65) • a) = -2015 :=
sorry

end dot_product_of_AC_BD_l701_701473


namespace slope_parallel_to_original_line_l701_701945

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l701_701945


namespace turquoise_more_green_l701_701168

-- Definitions based on given conditions
def total_people : ℕ := 150
def more_blue : ℕ := 90
def both_blue_green : ℕ := 40
def neither_blue_green : ℕ := 20

-- Theorem statement to prove the number of people who believe turquoise is more green
theorem turquoise_more_green : (total_people - neither_blue_green - (more_blue - both_blue_green) - both_blue_green) + both_blue_green = 80 := by
  sorry

end turquoise_more_green_l701_701168


namespace moles_of_ammonium_hydroxide_l701_701661

-- Define the chemical reaction with the given conditions
def ammoniumChloride := 3 -- moles of NH₄Cl
def water := 3 -- moles of H₂O
def molar_ratio (a b c d : ℕ) : Prop := a = b ∧ b = c ∧ c = d

-- The balanced equation in terms of molar ratios
axiom balanced_equation : molar_ratio 1 1 1 1

theorem moles_of_ammonium_hydroxide :
  ammoniumChloride = water →
  (∃ nh4oh : ℕ, balanced_equation ∧ nh4oh = ammoniumChloride) :=
by
  intros h1
  use (ammoniumChloride)
  split
  . exact balanced_equation
  . rw h1
    rfl

end moles_of_ammonium_hydroxide_l701_701661


namespace hyperbola_eccentricity_l701_701670

variables (a b : ℝ) (ha : a > 0) (hb : b > 0)

def hyperbola (P : ℝ × ℝ) : Prop :=
  ∃ (m n : ℝ), P = (m, n) ∧ (n^2 / a^2) - (m^2 / b^2) = 1

def asymptote_point (n : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), 
    ((n = (a / b) * x₁ ∨ n = -(a / b) * x₁) ∧ (x₂ = - (bn / a))) ∧ 
    ((n = (a / b) * x₂ ∨ n = -(a / b) * x₂) ∧ (x₁ = (bn / a)))

def dot_product_condition (m n : ℝ) : Prop :=
  let PA := ((bn / a) - m, 0)
  let PB := (-(bn / a) - m, 0) in
  (PA.1 * PB.1 + PA.2 * PB.2) = - (a^2 / 4)

theorem hyperbola_eccentricity : 
  ∀ (P : ℝ × ℝ), hyperbola a b P → 
  ∀ (m n : ℝ), P = (m, n) → dot_product_condition a b m n → 
  (sqrt (1 + (b^2 / a^2)) = sqrt (1 + 1 / 4) := sqrt (5) / 2) := 
sorry

end hyperbola_eccentricity_l701_701670


namespace slope_of_parallel_line_l701_701986

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l701_701986


namespace total_amount_paid_l701_701336

-- Define the quantities and rates as constants
def quantity_grapes : ℕ := 8
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost functions
def cost_grapes (q : ℕ) (r : ℕ) : ℕ := q * r
def cost_mangoes (q : ℕ) (r : ℕ) : ℕ := q * r

-- Define the total cost function
def total_cost (c1 : ℕ) (c2 : ℕ) : ℕ := c1 + c2

-- State the proof problem
theorem total_amount_paid :
  total_cost (cost_grapes quantity_grapes rate_grapes) (cost_mangoes quantity_mangoes rate_mangoes) = 1055 :=
by
  sorry

end total_amount_paid_l701_701336


namespace total_items_children_carry_l701_701645

theorem total_items_children_carry 
  (pieces_per_pizza : ℕ) (number_of_fourthgraders : ℕ) (pizza_per_fourthgrader : ℕ) 
  (pepperoni_per_pizza : ℕ) (mushrooms_per_pizza : ℕ) (olives_per_pizza : ℕ) 
  (total_pizzas : ℕ) (total_pieces_of_pizza : ℕ) (total_pepperoni : ℕ) (total_mushrooms : ℕ) 
  (total_olives : ℕ) (total_toppings : ℕ) (total_items : ℕ) : 
  pieces_per_pizza = 6 →
  number_of_fourthgraders = 10 →
  pizza_per_fourthgrader = 20 →
  pepperoni_per_pizza = 5 →
  mushrooms_per_pizza = 3 →
  olives_per_pizza = 8 →
  total_pizzas = number_of_fourthgraders * pizza_per_fourthgrader →
  total_pieces_of_pizza = total_pizzas * pieces_per_pizza →
  total_pepperoni = total_pizzas * pepperoni_per_pizza →
  total_mushrooms = total_pizzas * mushrooms_per_pizza →
  total_olives = total_pizzas * olives_per_pizza →
  total_toppings = total_pepperoni + total_mushrooms + total_olives →
  total_items = total_pieces_of_pizza + total_toppings →
  total_items = 4400 :=
by
  sorry

end total_items_children_carry_l701_701645


namespace count_integers_satisfying_inequality_l701_701434

theorem count_integers_satisfying_inequality :
  {n : Int | -12 ≤ n ∧ n ≤ 12 ∧ (n - 3) * (n + 3) * (n + 7) < 0}.card = 8 :=
by
  sorry

end count_integers_satisfying_inequality_l701_701434


namespace total_households_surveyed_l701_701198

theorem total_households_surveyed :
  ∀ (H_neither H_onlyA H_both : ℕ)
  (H_onlyB : ℕ := 3 * H_both),
  H_neither = 80 →
  H_onlyA = 60 →
  H_both = 30 →
  H_neither + H_onlyA + H_onlyB + H_both = 260 :=
by
  intros H_neither H_onlyA H_both H_onlyB h_neither h_onlyA h_both
  have h_onlyB : H_onlyB = 3 * H_both := rfl
  rw [h_neither, h_onlyA, h_both, h_onlyB]
  exact rfl

end total_households_surveyed_l701_701198


namespace students_in_classes_saved_money_strategy_class7_1_l701_701894

-- Part (1): Prove the number of students in each class
theorem students_in_classes (x : ℕ) (h1 : 40 < x) (h2 : x < 50) 
  (h3 : 105 - x > 50) (h4 : 15 * x + 12 * (105 - x) = 1401) : x = 47 ∧ (105 - x) = 58 := by
  sorry

-- Part (2): Prove the amount saved by purchasing tickets together
theorem saved_money(amt_per_ticket : ℕ → ℕ) 
  (h1 : amt_per_ticket 105 = 1401) 
  (h2 : ∀n, n > 100 → amt_per_ticket n = 1050) : amt_per_ticket 105 - 1050 = 351 := by
  sorry

-- Part (3): Strategy to save money for class 7 (1)
theorem strategy_class7_1 (students_1 : ℕ) (h1 : students_1 = 47) 
  (cost_15 : students_1 * 15 = 705) 
  (cost_51 : 51 * 12 = 612) : 705 - 612 = 93 := by
  sorry

end students_in_classes_saved_money_strategy_class7_1_l701_701894


namespace ellipse_chord_through_focus_length_l701_701843

theorem ellipse_chord_through_focus_length :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | (p.1 ^ 2 / 36) + (p.2 ^ 2 / 16) = 1} →
  ∃ F : ℝ × ℝ, let fx := 2 * Real.sqrt 5 in
  F = (fx, 0) → 
  Real.dist (x, y) F = 2 →
  Real.dist (another_point, another_point_y) F = 4 * Real.sqrt 5 / 3 :=
sorry

end ellipse_chord_through_focus_length_l701_701843


namespace value_of_f2019_l701_701592

noncomputable def f : ℤ → ℤ := sorry

theorem value_of_f2019 (f : ℤ → ℤ) (h : ∀ n : ℤ, (n - 2019) * f(n) - f(2019 - n) = 2019) : f(2019) = 2019 * 2018 :=
sorry

end value_of_f2019_l701_701592


namespace Robert_GRE_exam_l701_701066

/-- Robert started preparation for GRE entrance examination in the month of January and prepared for 5 months. Prove that he could write the examination any date after the end of May.-/
theorem Robert_GRE_exam (start_month : ℕ) (prep_duration : ℕ) : 
  start_month = 1 → prep_duration = 5 → ∃ exam_date, exam_date > 5 :=
by
  sorry

end Robert_GRE_exam_l701_701066


namespace largest_class_and_girls_l701_701372

theorem largest_class_and_girls (numbers : List ℕ)
  (condition1 : numbers.length = 8)
  (condition2 : 70 < numbers.sum)
  (condition3 : numbers = [7, 9, 10, 12, 15, 16, 19, 21]) :
  (21 ∈ numbers) ∧ (let total_students := numbers.sum in total_students - 38 = 33) :=
by
  sorry

end largest_class_and_girls_l701_701372


namespace difference_in_cans_l701_701335

-- Definitions of the conditions
def total_cans_collected : ℕ := 9
def cans_in_bag : ℕ := 7

-- Statement of the proof problem
theorem difference_in_cans :
  total_cans_collected - cans_in_bag = 2 := by
  sorry

end difference_in_cans_l701_701335


namespace sufficient_condition_for_ellipse_l701_701644

theorem sufficient_condition_for_ellipse (m : ℝ) (h : m^2 > 5) : m^2 > 4 := by
  sorry

end sufficient_condition_for_ellipse_l701_701644


namespace no_two_items_share_color_l701_701342

theorem no_two_items_share_color (shirts pants hats : Fin 5) :
  ∃ num_outfits : ℕ, num_outfits = 60 :=
by
  sorry

end no_two_items_share_color_l701_701342


namespace min_n_consecutive_integers_sum_of_digits_is_multiple_of_8_l701_701353

theorem min_n_consecutive_integers_sum_of_digits_is_multiple_of_8 
: ∃ n : ℕ, (∀ (nums : Fin n.succ → ℕ), 
              (∀ i j, i < j → nums i < nums j → nums j = nums i + 1) →
              ∃ i, (nums i) % 8 = 0) ∧ n = 15 := 
sorry

end min_n_consecutive_integers_sum_of_digits_is_multiple_of_8_l701_701353


namespace parallel_slope_l701_701937

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l701_701937


namespace smallest_n_l701_701783

-- Define the conditions
def V : Type := fin 2019
def coplanar_condition (P : fin 2019 → fin 2019 → fin 2019 → fin 2019 → Prop) : Prop :=
  ∀ (a b c d : fin 2019), ¬P a b c d

def E (V : Type) : Type := { e : V × V // e.1 ≠ e.2 }

-- Define the problem statement
theorem smallest_n 
    (V : Type)
    [fintype V]
    (E : set (E V))
    (H: coplanar_condition (λ a b c d, true)) -- no four points are coplanar
    (n : ℕ)
    (H1 : E.card ≥ n) :
    ∃ (n : ℕ), n = 1018 ∧ (∃ (pairs : finset (E V × E V)), pairs.card = 908 ∧ 
                           (∀ p1 p2 ∈ pairs, p1 ≠ p2 ∧ p1.1 ≠ p1.2 ∧ p2.1 ≠ p2.2)) :=
sorry

end smallest_n_l701_701783


namespace speed_in_terms_of_time_l701_701426

variable (a b x : ℝ)

-- Conditions
def condition1 : Prop := 1000 = a * x
def condition2 : Prop := 833 = b * x

-- The theorem to prove
theorem speed_in_terms_of_time (h1 : condition1 a x) (h2 : condition2 b x) :
  a = 1000 / x ∧ b = 833 / x :=
by
  sorry

end speed_in_terms_of_time_l701_701426


namespace waterAmount_initialA_20kg_l701_701117

noncomputable def waterInBucketA (initialWaterA : ℝ) (bucketB : ℝ) : ℝ := 
  initialWaterA

theorem waterAmount_initialA_20kg :
  ∃ initialWaterA : ℝ, 
  ∀ bucketB : ℝ,
  (0.2 * initialWaterA = 2/5 * bucketB) ∧ (6 + 2/5 * bucketB = bucketB) → 
  initialWaterA = 20 :=
begin
  sorry
end

end waterAmount_initialA_20kg_l701_701117


namespace train_crossing_time_l701_701611

def length_of_train : ℝ := 145
def length_of_bridge : ℝ := 230
def speed_km_per_hr : ℝ := 45
def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)
def total_distance : ℝ := length_of_train + length_of_bridge
def expected_time : ℝ := 30

theorem train_crossing_time :
  (total_distance / speed_m_per_s) = expected_time := by
  sorry

end train_crossing_time_l701_701611


namespace range_of_m_l701_701680

theorem range_of_m (m : ℝ) (p q : Prop) (p_def: p ↔ (m > 2 ∨ m < -2)) (q_def: q ↔ (m > 1 ∨ m < 0)) :
  (p ∨ q) ∧ ¬p → (−2 ≤ m ∧ m < 0) ∨ (1 < m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l701_701680


namespace segment_KL_length_l701_701502

noncomputable def pentagon_side_length : ℝ := 1

variable {A B C D E P Q R S K L : Type}
          [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
          [LinearOrderedField D] [LinearOrderedField E]
          [LinearOrderedField P] [LinearOrderedField Q]
          [LinearOrderedField R] [LinearOrderedField S]
          [LinearOrderedField K] [LinearOrderedField L]

def mid_point (X Y : ℝ) : ℝ := (X + Y) / 2

axiom AB_eq_1 : dist A B = pentagon_side_length
axiom BC_eq_1 : dist B C = pentagon_side_length
axiom CD_eq_1 : dist C D = pentagon_side_length
axiom DE_eq_1 : dist D E = pentagon_side_length
axiom EA_eq_1 : dist E A = pentagon_side_length

axiom P_mid_AB : P = mid_point A B
axiom Q_mid_BC : Q = mid_point B C
axiom R_mid_CD : R = mid_point C D
axiom S_mid_DE : S = mid_point D E

axiom K_mid_PR : K = mid_point P R
axiom L_mid_QS : L = mid_point Q S 

theorem segment_KL_length : dist K L = 1 / 4 := sorry

end segment_KL_length_l701_701502


namespace relation_among_a_b_c_l701_701036

noncomputable def a := Real.pi ^ 0.3
noncomputable def b := Real.log 3 / Real.log Real.pi -- This is equivalent to log_base(pi)(3)
def c := 1

theorem relation_among_a_b_c : a > c ∧ c > b := by
  sorry

end relation_among_a_b_c_l701_701036


namespace number_of_zeros_pattern_l701_701806

theorem number_of_zeros_pattern (n : ℕ) (hn : n > 0) : 
  let m := 10^n - 1 in 
  let num_zeros := (n - 1) in 
  ∃ k : ℕ, m^2 = k * 10^(n - 1) + r ∧ r = 0 ∧ ((k ≠ 0) → num_zeros = n - 1) :=
begin
  -- Proof would go here
  sorry
end

end number_of_zeros_pattern_l701_701806


namespace range_of_a_l701_701691

theorem range_of_a (e : ℝ) (ln : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Icc (1/e) 1, ∃! y ∈ Icc (-1 : ℝ) 1, (ln x - x + 1 + a = y^2 * (Real.exp y))) ->
  (2 / e < a ∧ a ≤ Real.exp 1) := 
by
  sorry

end range_of_a_l701_701691


namespace principal_made_mistake_l701_701908

-- Definitions based on given conditions
def students_per_class (x : ℤ) : Prop := x > 0
def total_students (x : ℤ) : ℤ := 2 * x
def non_failing_grades (y : ℤ) : ℤ := y
def failing_grades (y : ℤ) : ℤ := y + 11
def total_grades (x y : ℤ) : Prop := total_students x = non_failing_grades y + failing_grades y

-- Proposition stating the principal made a mistake
theorem principal_made_mistake (x y : ℤ) (hx : students_per_class x) : ¬ total_grades x y :=
by
  -- Assume the proof for the hypothesis is required here
  sorry

end principal_made_mistake_l701_701908


namespace Oates_reunion_attendance_l701_701536

theorem Oates_reunion_attendance :
  ∀ (total Oates Hall Both : ℕ),
    total = 100 →
    Hall = 62 →
    Both = 12 →
    total = Oates + Hall - Both →
    Oates = 50 :=
by
  intros total Oates Hall Both ht hH hB h_eq
  rw [ht, hH, hB] at h_eq
  linarith


end Oates_reunion_attendance_l701_701536


namespace angle_bisector_problem_l701_701166

/-- In triangle PQR with a right angle at R and angle P = 30 degrees, if QS is the angle bisector
of ∠QPR and S is on line PR, then ∠QSR is 60 degrees. -/
theorem angle_bisector_problem (P Q R S : Type) [linear_ordered_ring P Q R S] 
  (R_right : ∠PQR = 90)
  (angle_P : ∠P = 30)
  (QS_bisector : angle_bisector ∠QPR QS)
  (S_on_PR : S ∈ line PR) :
  ∠QSR = 60 :=
sorry

end angle_bisector_problem_l701_701166


namespace inscribed_circle_radius_l701_701103

-- Defining the variables and the equation
def eq : Prop :=
  ∃ (r : ℚ),
  let a := 3,
      b := 6,
      c := 18 in
  (1 : ℚ) / r = (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 2 * real.sqrt ((1 : ℚ) / (a*b) + (1 : ℚ) / (a*c) + (1 : ℚ) / (b*c))
  ∧ r = 9 / 8

-- The theorem we need to prove
theorem inscribed_circle_radius : eq :=
begin
  sorry
end

end inscribed_circle_radius_l701_701103


namespace length_KL_l701_701507

variables {Point : Type} [metric_space Point]
variables {A B C D E P Q R S K L M : Point}
variables (AB BC CD DE AE PR QS : ℝ)
variables (midpoint : Point → Point → Point)
variables (length : Point → Point → ℝ)

-- Conditions given
axiom pentagon_eq_sides (AB BC CD DE AE : ℝ) : AB = 1 ∧ BC = 1 ∧ CD = 1 ∧ DE = 1
axiom is_midpoint (X Y Z : Point) :
  midpoint X Y = Z ↔ length X Z = length Y Z / 2
axiom midpoints 
  (P Q R S K L : Point)
  (AB BC CD DE PR QS : ℝ) 
  (MP AB BC CD DE PR QS length midpoint) :
  midpoint A B = P ∧ midpoint B C = Q ∧ midpoint C D = R ∧ midpoint D E = S ∧ 
  midpoint P R = K ∧ midpoint Q S = L

-- Geometry proof problem
theorem length_KL (A B C D E P Q R S K L : Point) :
  pentagon_eq_sides AB BC CD DE AE →
  midpoints P Q R S K L AB BC CD DE PR QS length midpoint →
  length K L = 1 / 4 * length A E := 
sorry

end length_KL_l701_701507


namespace greatest_b_value_l701_701660

theorem greatest_b_value : ∃ (b : ℝ), (-b^2 + 9 * b - 18 ≥ 0) ∧ ∀ (b' : ℝ), (-b'^2 + 9 * b' - 18 ≥ 0) → b' ≤ b :=
by
  let b := 6
  existsi b
  split
  { -- Prove (-b^2 + 9 * b - 18 ≥ 0) for b = 6
    sorry },
  { -- Prove ∀ (b' : ℝ), (-b'^2 + 9 * b' - 18 ≥ 0) → b' ≤ b
    sorry }

end greatest_b_value_l701_701660


namespace slope_of_parallel_line_l701_701922

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l701_701922


namespace solve_equation_l701_701073

-- Definitions for the variables and the main equation
def equation (x y z : ℤ) : Prop :=
  5 * x^2 + y^2 + 3 * z^2 - 2 * y * z = 30

-- The statement that needs to be proved
theorem solve_equation (x y z : ℤ) :
  equation x y z ↔ (x, y, z) = (1, 5, 0) ∨ (x, y, z) = (1, -5, 0) ∨ (x, y, z) = (-1, 5, 0) ∨ (x, y, z) = (-1, -5, 0) :=
by
  sorry

end solve_equation_l701_701073


namespace parallel_lines_solution_l701_701331

theorem parallel_lines_solution (a : ℝ) :
  (∃ (k1 k2 : ℝ), k1 ≠ 0 ∧ k2 ≠ 0 ∧ 
  ∀ x y : ℝ, x + a^2 * y + 6 = 0 → k1*y = x ∧ 
             (a-2) * x + 3 * a * y + 2 * a = 0 → k2*y = x) 
  → (a = -1 ∨ a = 0) :=
by
  sorry

end parallel_lines_solution_l701_701331


namespace geometric_sequence_problem_l701_701683

theorem geometric_sequence_problem
  (a : ℕ → ℝ) (r : ℝ)
  (h₀ : ∀ n, a n > 0)
  (h₁ : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)
  (h₂ : ∀ n, a (n + 1) = a n * r) :
  a 3 + a 5 = 5 :=
sorry

end geometric_sequence_problem_l701_701683


namespace total_profit_correct_l701_701569

noncomputable def total_profit (a b c : ℕ) (c_share : ℕ) : ℕ :=
  let ratio := a + b + c
  let part_value := c_share / c
  ratio * part_value

theorem total_profit_correct (h_a : ℕ := 5000) (h_b : ℕ := 8000) (h_c : ℕ := 9000) (h_c_share : ℕ := 36000) :
  total_profit h_a h_b h_c h_c_share = 88000 :=
by
  sorry

end total_profit_correct_l701_701569


namespace find_mini_cupcakes_l701_701520

-- Definitions of the conditions
def number_of_donut_holes := 12
def number_of_students := 13
def desserts_per_student := 2

-- Statement of the theorem to prove the number of mini-cupcakes is 14
theorem find_mini_cupcakes :
  let D := number_of_donut_holes
  let N := number_of_students
  let total_desserts := N * desserts_per_student
  let C := total_desserts - D
  C = 14 :=
by
  sorry

end find_mini_cupcakes_l701_701520


namespace determine_a_l701_701738

theorem determine_a (a : ℝ) : (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) := 
sorry

end determine_a_l701_701738


namespace parallel_line_slope_l701_701980

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l701_701980


namespace circle_equation_exists_l701_701856

theorem circle_equation_exists :
  ∃ D E F : ℝ, (∀ (x y : ℝ), (x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
                      (x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
                      (x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
                      (D = -4) ∧ (E = -6) ∧ (F = 0) :=
by
  use [-4, -6, 0]
  intro x y
  split
  { intros hx hy, simp [hx, hy] }
  split
  { intros hx hy, simp [hx, hy], linarith }
  { intros hx hy, simp [hx, hy], linarith }
  sorry

end circle_equation_exists_l701_701856


namespace rationalize_denominator_of_seven_over_sqrt_343_l701_701454

noncomputable def rationalize_denominator (x : ℝ) : ℝ := x * sqrt 7 / 7

theorem rationalize_denominator_of_seven_over_sqrt_343 :
  (343 = 7^3) → (sqrt 343 = 7 * sqrt 7) →
  (7 / sqrt 343 = sqrt 7 / 7) :=
by
  intros h1 h2
  sorry

end rationalize_denominator_of_seven_over_sqrt_343_l701_701454


namespace sum_of_three_distinct_integers_l701_701101

theorem sum_of_three_distinct_integers (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c) 
  (h₄ : a * b * c = 5^3) (h₅ : a > 0) (h₆ : b > 0) (h₇ : c > 0) : a + b + c = 31 :=
by
  sorry

end sum_of_three_distinct_integers_l701_701101


namespace prove_inequality_for_k_eq_2_l701_701321

noncomputable def f (a : ℝ) (x : ℝ) (k : ℕ) : ℝ := 2 ^ ((a - x) ^ k)

theorem prove_inequality_for_k_eq_2 (a : ℝ) :
  (f a 1 2 > f a 3 2) ∧ (f a 2 2 > f a 3 2) → |a - 1| > |a - 2| :=
by
  intros h,
  sorry

end prove_inequality_for_k_eq_2_l701_701321


namespace find_a_l701_701763

-- Define the parametric equations of the line l
def line_l (t a : ℝ) :=
  (t, t - a)

-- Define the parametric equations of the ellipse C
def ellipse_C (φ : ℝ) :=
  (3 * Real.cos φ, 2 * Real.sin φ)

-- Define the right vertex of the ellipse C
def right_vertex_ellipse_C :=
  (3, 0)

-- State the proof problem
theorem find_a (a : ℝ) (h : (∃ t : ℝ, line_l t a = right_vertex_ellipse_C)) :
  a = 3 :=
sorry

end find_a_l701_701763


namespace quadratic_inequality_contains_conditional_branch_l701_701617

/-- Among the algorithmic options presented, prove that solving a quadratic inequality 
    requires a conditional branch structure. -/
theorem quadratic_inequality_contains_conditional_branch :
  ∀ (alg : Type), 
    (alg = "Calculating the product of two numbers" →
     ¬ contains_conditional_branch alg) →
    (alg = "Calculating the distance from a point to a line" →
     ¬ contains_conditional_branch alg) →
    (alg = "Solving a quadratic inequality" →
     contains_conditional_branch alg) →
    (alg = "Calculating the area of a trapezoid given the lengths of its bases and height" →
     ¬ contains_conditional_branch alg) →
    contains_conditional_branch "Solving a quadratic inequality" :=
by
  intros alg h_prod h_distance h_quad h_area
  apply h_quad
  rfl

-- Helper predicate to describe if an algorithm contains a conditional branch structure
def contains_conditional_branch (alg : Type) : Prop := sorry

end quadratic_inequality_contains_conditional_branch_l701_701617


namespace billed_minutes_l701_701671

noncomputable def John_bill (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) : ℝ :=
  (total_bill - monthly_fee) / cost_per_minute

theorem billed_minutes : ∀ (monthly_fee cost_per_minute total_bill : ℝ), 
  monthly_fee = 5 → 
  cost_per_minute = 0.25 → 
  total_bill = 12.02 → 
  John_bill monthly_fee cost_per_minute total_bill = 28 :=
by
  intros monthly_fee cost_per_minute total_bill hf hm hb
  rw [hf, hm, hb, John_bill]
  norm_num
  sorry

end billed_minutes_l701_701671


namespace circle_contains_13_points_l701_701811

theorem circle_contains_13_points (points : Fin 25 → ℝ × ℝ) 
  (h : ∀ (i j k : Fin 25), i ≠ j → j ≠ k → i ≠ k →
        (dist (points i) (points j) < 1 ∨ 
         dist (points j) (points k) < 1 ∨ 
         dist (points i) (points k) < 1)) :
  ∃ (center : ℝ × ℝ), (∃ (radius : ℝ), radius = 1) ∧ (13 ≤ (Finset.univ.filter (λ i, dist (points i) center ≤ 1)).card) :=
sorry

end circle_contains_13_points_l701_701811


namespace x21_eq_zero_l701_701791

noncomputable def x_sequence (x1 x2 : ℕ) : ℕ → ℕ
| 1 := x1
| 2 := x2
| 3 := abs (x1 - x2)
| k + 1 := if k = 3 then min (abs (x1 - x2)) (min (abs (x1 - abs (x1 - x2))) (abs (x2 - abs (x1 - x2))))
            else min (abs (x_sequence x1 x2 (k - 1) - x_sequence x1 x2 (k - 2))) 0  -- condition for sequence when k >= 5

theorem x21_eq_zero (x1 x2 : ℕ) (h1 : x1 < 1000) (h2 : x2 < 1000) : x_sequence x1 x2 21 = 0 :=
sorry

end x21_eq_zero_l701_701791


namespace line_parabola_intersection_l701_701271

theorem line_parabola_intersection (k : ℝ) :
  (∀ x y, y = k * x + 1 ∧ y^2 = 4 * x → y = 1 ∧ x = 1 / 4) ∨
  (∀ x y, y = k * x + 1 ∧ y^2 = 4 * x → (k^2 * x^2 + (2 * k - 4) * x + 1 = 0) ∧ (4 * k * k - 16 * k + 16 - 4 * k * k = 0) → k = 1) :=
sorry

end line_parabola_intersection_l701_701271


namespace river_depth_mid_may_l701_701377

variable (DepthMidMay DepthMidJune DepthMidJuly : ℕ)

theorem river_depth_mid_may :
  (DepthMidJune = DepthMidMay + 10) →
  (DepthMidJuly = 3 * DepthMidJune) →
  (DepthMidJuly = 45) →
  DepthMidMay = 5 :=
by
  intros h1 h2 h3
  sorry

end river_depth_mid_may_l701_701377


namespace rectangle_length_l701_701154

theorem rectangle_length
  (side_length_square : ℝ)
  (width_rectangle : ℝ)
  (area_equiv : side_length_square ^ 2 = width_rectangle * l)
  : l = 24 := by
  sorry

end rectangle_length_l701_701154


namespace smallest_positive_period_maximum_value_of_f_range_of_f_on_interval_l701_701709

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin x - Real.sqrt 3 * (Real.cos x)^2

theorem smallest_positive_period : (∃ p > 0, ∀ x : ℝ, f(x + p) = f(x)) ∧ ∀ p' > 0, (∃ x : ℝ, f(x + p') ≠ f(x)) → p = π :=
by
  sorry

theorem maximum_value_of_f : ∀ x : ℝ, (max (f x) = 1 - Real.sqrt 3 / 2) :=
by
  sorry

theorem range_of_f_on_interval : ∀ x : ℝ, (x ≥ π/6 ∧ x ≤ 2*π/3) → (f x ≥ -Real.sqrt 3 / 2 ∧ f x ≤ 1 - Real.sqrt 3 / 2) :=
by
  sorry

end smallest_positive_period_maximum_value_of_f_range_of_f_on_interval_l701_701709


namespace slope_of_parallel_line_l701_701958

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l701_701958


namespace part_a_part_b_part_c_l701_701274

noncomputable def S (n : ℕ) : ℕ :=
  Nat.findGreatest (λ k, k ≤ n^2 ∧ ∃ (l : ℕ), l ≤ k ∧ (n^2 = k + l)) (n^2)

theorem part_a (n : ℕ) (h : n ≥ 4) : S n ≤ n^2 - 14 := sorry

theorem part_b : S 13 = 169 - 14 := sorry

theorem part_c : ∃ infinitely_many_n, ∀ n, S n = n^2 - 14 := sorry

end part_a_part_b_part_c_l701_701274


namespace determine_dresses_and_shoes_colors_l701_701529

variables (dress_color shoe_color : String → String)
variables (Tamara Valya Lida : String)

-- Conditions
def condition_1 : Prop := ∀ x : String, x ≠ Tamara → dress_color x ≠ shoe_color x
def condition_2 : Prop := shoe_color Valya = "white"
def condition_3 : Prop := dress_color Lida ≠ "red" ∧ shoe_color Lida ≠ "red"
def condition_4 : Prop := ∀ x : String, dress_color x ∈ ["white", "red", "blue"] ∧ shoe_color x ∈ ["white", "red", "blue"]

-- Desired conclusion
def conclusion : Prop :=
  dress_color Valya = "blue" ∧ shoe_color Valya = "white" ∧
  dress_color Lida = "white" ∧ shoe_color Lida = "blue" ∧
  dress_color Tamara = "red" ∧ shoe_color Tamara = "red"

theorem determine_dresses_and_shoes_colors
  (Tamara Valya Lida : String)
  (h1 : condition_1 dress_color shoe_color Tamara)
  (h2 : condition_2 shoe_color Valya)
  (h3 : condition_3 dress_color shoe_color Lida)
  (h4 : condition_4 dress_color shoe_color) :
  conclusion dress_color shoe_color Valya Lida Tamara :=
sorry

end determine_dresses_and_shoes_colors_l701_701529


namespace cookie_jar_initial_amount_l701_701900

variable (initial_amount : ℕ)
variable (doris_spent : ℕ := 6)
variable (martha_spent : ℕ := doris_spent / 2)
variable (remaining : ℕ := 15)

theorem cookie_jar_initial_amount :
  initial_amount = doris_spent + martha_spent + remaining :=
begin
  sorry
end

end cookie_jar_initial_amount_l701_701900


namespace slope_of_parallel_line_l701_701973

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l701_701973


namespace mary_flour_total_l701_701439

-- Definitions for conditions
def initial_flour : ℝ := 7.0
def extra_flour : ℝ := 2.0
def total_flour (x y : ℝ) : ℝ := x + y

-- The statement we want to prove
theorem mary_flour_total : total_flour initial_flour extra_flour = 9.0 := 
by sorry

end mary_flour_total_l701_701439


namespace meat_division_l701_701600

theorem meat_division (w1 w2 meat : ℕ) (h1 : w1 = 645) (h2 : w2 = 237) (h3 : meat = 1000) :
  ∃ (m1 m2 : ℕ), m1 = 296 ∧ m2 = 704 ∧ w1 + m1 = w2 + m2 := by
  sorry

end meat_division_l701_701600


namespace farmer_children_count_l701_701189

def apples_each_bag := 15
def eaten_each := 4
def sold_apples := 7
def apples_left := 60

theorem farmer_children_count : 
  ∃ (n : ℕ), 15 * n - (2 * 4 + 7) = 60 ∧ n = 5 :=
by
  use 5
  sorry

end farmer_children_count_l701_701189


namespace valid_permutations_count_l701_701757

-- Define the conditions for the sequence
def starts_with_one (s : List ℕ) : Prop :=
  s.head = 1

def is_permutation_of_seq (s : List ℕ) : Prop :=
  s ~ [1, 2, 3, 4, 5, 6]

def no_three_consecutive_increasing (s : List ℕ) : Prop :=
  ∀ (i : ℕ), i + 2 < s.length → ¬(s.nth i < s.nth (i + 1) ∧ s.nth (i + 1) < s.nth (i + 2))

def no_three_consecutive_decreasing (s : List ℕ) : Prop :=
  ∀ (i : ℕ), i + 2 < s.length → ¬(s.nth i > s.nth (i + 1) ∧ s.nth (i + 1) > s.nth (i + 2))

theorem valid_permutations_count :
  (∃ s : List ℕ, starts_with_one s ∧ is_permutation_of_seq s ∧ no_three_consecutive_increasing s ∧ no_three_consecutive_decreasing s) → 
  s.card = 24 :=
sorry

end valid_permutations_count_l701_701757


namespace chef_initial_potatoes_l701_701175

theorem chef_initial_potatoes (fries_per_potato : ℕ) (total_fries_needed : ℕ) (leftover_potatoes : ℕ) 
  (H1 : fries_per_potato = 25) 
  (H2 : total_fries_needed = 200) 
  (H3 : leftover_potatoes = 7) : 
  (total_fries_needed / fries_per_potato + leftover_potatoes = 15) :=
by
  sorry

end chef_initial_potatoes_l701_701175


namespace find_f240_l701_701879

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

lemma f_increasing (n : ℕ) : f(n) < f(n+1) := sorry
lemma g_increasing (n : ℕ) : g(n) < g(n+1) := sorry
lemma g_definition (n : ℕ) (hn : n ≥ 1) : g(n) = f(f(n)) + 1 := sorry

theorem find_f240 : f(240) = 388 := 
by
  -- Using previously established conditions and properties to show f(240) = 388
  sorry

end find_f240_l701_701879


namespace find_angles_of_triangle_ABC_l701_701370

theorem find_angles_of_triangle_ABC
  (ABC : Triangle)
  (BH : is_height ABC B H)
  (AP : is_angle_bisector ABC A P)
  (cyclic_ABPH : is_cyclic_quadrilateral A B P H)
  (angle_CPH : angle C P H = 20) :
  angle ABC.bac = 20 ∧ angle ABC.abc = 80 ∧ angle ABC.acb = 80 :=
by sorry

end find_angles_of_triangle_ABC_l701_701370


namespace complex_point_in_second_quadrant_l701_701098

def complex_point (z : ℂ) : ℝ × ℝ :=
  (z.re, z.im)

theorem complex_point_in_second_quadrant :
  let z1 : ℂ := -1 + 2 * Complex.i
  let z2 : ℂ := 3 - 1 * Complex.i
  let z3 := z1 * z2
  complex_point z3 ∈ {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0} :=
by
  sorry

end complex_point_in_second_quadrant_l701_701098


namespace jen_age_proof_l701_701777

variable (JenAge : ℕ) (SonAge : ℕ)

theorem jen_age_proof (h1 : SonAge = 16) (h2 : JenAge = 3 * SonAge - 7) : JenAge = 41 :=
by
  -- conditions
  rw [h1] at h2
  -- substitution and simplification
  have h3 : JenAge = 3 * 16 - 7 := h2
  norm_num at h3
  exact h3

end jen_age_proof_l701_701777


namespace product_of_a_and_b_l701_701739

variable (a b : ℕ)

-- Conditions
def LCM(a b : ℕ) : ℕ := Nat.lcm a b
def HCF(a b : ℕ) : ℕ := Nat.gcd a b

-- Assertion: product of a and b
theorem product_of_a_and_b (h_lcm: LCM a b = 72) (h_hcf: HCF a b = 6) : a * b = 432 := by
  sorry

end product_of_a_and_b_l701_701739


namespace correct_loop_condition_l701_701863

theorem correct_loop_condition (S : ℕ) (i : ℕ) (x : ℕ → ℕ) : 
  i = 1 → 
  (∀ j, 1 ≤ j ∧ j ≤ 20 → x j ∈ ℕ) → 
  ∃ k, k = 20 ∧ S = ∑ i in finset.range 21, x i ∧ a = S / 20 :=
by
  sorry

end correct_loop_condition_l701_701863


namespace total_surfers_l701_701526

/-- There are three beaches: Malibu beach, Santa Monica beach, and Venice beach.
    The ratio of the number of surfers on these beaches is 5:3:2, respectively.
    There are 30 surfers on Santa Monica beach. -/
def surfers_ratio (r1 r2 r3 : ℕ) := (r1, r2, r3) = (5, 3, 2)

def surfers_on_santa_monica := 30

theorem total_surfers (r1 r2 r3 : ℕ) (s : ℕ) (h_ratio: surfers_ratio r1 r2 r3) (h_santa_mon : s = surfers_on_santa_monica) :
  let u := s / r2 in
  let surfersMalibu := r1 * u in
  let surfersVenice := r3 * u in
  surfersMalibu + s + surfersVenice = 100 :=
by
  sorry

end total_surfers_l701_701526


namespace prove_concurrent_l701_701318

open Real
open Topology

noncomputable def ellipse (P : ℝ × ℝ) : Prop := 
  (P.1^2 / 4 + P.2^2 = 1)

def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, -1)

def line_l1 (P : ℝ × ℝ) : Prop := 
  (P.1 = -2)

def line_l2 (P : ℝ × ℝ) : Prop := 
  (P.2 = -1)

variables {P : ℝ × ℝ} (hP : ellipse P) (hP_pos : P.1 > 0 ∧ P.2 > 0)

def line_l3 : ℝ × ℝ → ℝ × ℝ → Prop 
| Q R := (P.1 * Q.1 / 4 + P.2 * Q.2 = 1 
           ∧ Q = R)

def point_C : Prop := line_l1 point_A ∧ line_l2 point_A
def point_D : ℝ × ℝ → Prop := line_l2 P ∧ line_l3 point_A P
def point_E : ℝ × ℝ → Prop := line_l1 P ∧ line_l3 point_A P

def concurrent_lines : Prop := 
  ∃ Q, (∃ R, line_l3 P R ∧
    ((P.1 * (Q.1 + 2) / 4 + P.2 * (Q.2 + 2) = 1 
     ∧ line_l1 P) ∧
    ((Q.1 + 2) ^ 2 / 4 + (Q.2 + 2) ^ 2 = 1 
     ∨ (2 * P.1 + P.1^2) / (8 * P.2 + 8 * P.2^2) = 1 / 2)))

theorem prove_concurrent : concurrent_lines := 
sorry

end prove_concurrent_l701_701318


namespace sum_of_three_distinct_integers_l701_701102

theorem sum_of_three_distinct_integers (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c) 
  (h₄ : a * b * c = 5^3) (h₅ : a > 0) (h₆ : b > 0) (h₇ : c > 0) : a + b + c = 31 :=
by
  sorry

end sum_of_three_distinct_integers_l701_701102


namespace coeff_x3_expansion_l701_701474

theorem coeff_x3_expansion : 
  ∑ i in Finset.range 4, 
    (∑ j in Finset.range (i + 1), (-1 : ℤ) ^ (j + i) * Nat.choose 5 j * Nat.choose 3 (i - j)) = -14 :=
by
  sorry

end coeff_x3_expansion_l701_701474


namespace slope_of_parallel_line_l701_701964

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l701_701964


namespace average_speed_calculation_l701_701571

-- Define the conditions
def altitude_km := 0.23 -- altitude in kilometers
def speed_up := 32 -- speed while traveling up in km/hr
def speed_down := 48 -- speed while traveling down in km/hr

-- Define the times for traveling up and down
def time_up := altitude_km / speed_up
def time_down := altitude_km / speed_down

-- Define the total time and total distance
def total_time := time_up + time_down
def total_distance := 2 * altitude_km

-- Define the average speed
def avg_speed := total_distance / total_time

-- Theorem statement
theorem average_speed_calculation :
  avg_speed = 38.39 := by
  sorry

end average_speed_calculation_l701_701571


namespace Tyler_CDs_count_l701_701541

-- Definitions using the conditions
def initial_CDs := 21
def fraction_given_away := 1 / 3
def CDs_bought := 8

-- The problem is to prove the final number of CDs Tyler has
theorem Tyler_CDs_count (initial_CDs : ℕ) 
  (fraction_given_away : ℚ) -- using ℚ for fractions
  (CDs_bought : ℕ) : 
  let CDs_given_away := initial_CDs * fraction_given_away in
  let remaining_CDs := initial_CDs - CDs_given_away in
  let final_CDs := remaining_CDs + CDs_bought in
  final_CDs = 22 := 
by 
  sorry

end Tyler_CDs_count_l701_701541


namespace complete_laps_l701_701797

-- Definitions based on conditions
def total_distance := 3.25  -- total distance Lexi wants to run
def lap_distance := 0.25    -- distance of one lap

-- Proof statement: Total number of complete laps to cover the given distance
theorem complete_laps (h1 : total_distance = 3 + 1/4) (h2 : lap_distance = 1/4) :
  (total_distance / lap_distance) = 13 :=
by 
  sorry

end complete_laps_l701_701797


namespace geom_seq_sixth_term_l701_701593

def geom_seq (a r : ℕ) : ℕ → ℕ
| 0 => a
| (n+1) => r * geom_seq n

theorem geom_seq_sixth_term {r : ℕ} (h : geom_seq 3 r 4 = 375) : geom_seq 3 r 5 = 9375 := 
by
  sorry

end geom_seq_sixth_term_l701_701593


namespace slope_parallel_to_original_line_l701_701939

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l701_701939


namespace eval_expr_l701_701648

-- Definitions
def i := Complex.I
def i_squared := i ^ 2 = -1
def i_to_four := i ^ 4 = 1
def i_inverse := i ^ (-1) = -i

-- Theorem statement
theorem eval_expr : (i^8 + i^20 + i^(-14))^2 = 1 :=
by
  have h1: i^2 = -1 := i_squared
  have h2: i^4 = 1 := i_to_four
  have h3: i^(-1) = -i := i_inverse
  sorry

end eval_expr_l701_701648


namespace segment_KL_length_l701_701500

noncomputable def pentagon_side_length : ℝ := 1

variable {A B C D E P Q R S K L : Type}
          [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
          [LinearOrderedField D] [LinearOrderedField E]
          [LinearOrderedField P] [LinearOrderedField Q]
          [LinearOrderedField R] [LinearOrderedField S]
          [LinearOrderedField K] [LinearOrderedField L]

def mid_point (X Y : ℝ) : ℝ := (X + Y) / 2

axiom AB_eq_1 : dist A B = pentagon_side_length
axiom BC_eq_1 : dist B C = pentagon_side_length
axiom CD_eq_1 : dist C D = pentagon_side_length
axiom DE_eq_1 : dist D E = pentagon_side_length
axiom EA_eq_1 : dist E A = pentagon_side_length

axiom P_mid_AB : P = mid_point A B
axiom Q_mid_BC : Q = mid_point B C
axiom R_mid_CD : R = mid_point C D
axiom S_mid_DE : S = mid_point D E

axiom K_mid_PR : K = mid_point P R
axiom L_mid_QS : L = mid_point Q S 

theorem segment_KL_length : dist K L = 1 / 4 := sorry

end segment_KL_length_l701_701500


namespace Jack_typing_time_l701_701014

theorem Jack_typing_time (P : ℝ) (hP : P > 0) :
  let John's_rate := P / 5 in
  let pages_typed_by_John := 3 * (John's_rate) in
  let remaining_pages := P - pages_typed_by_John in
  let Jack's_rate := (2 / 5) * (John's_rate) in
  let Jack_typing_time := remaining_pages / Jack's_rate in
  Jack_typing_time = 5 :=
by 
  -- Here would go the proof, which is not necessary for this task
  sorry

end Jack_typing_time_l701_701014


namespace minimize_average_cost_maximize_profit_l701_701182

-- Condition definitions:
def fixed_cost : ℝ := 10000000
def direct_total_cost (x : ℝ) : ℝ := 50 * x + (1 / 100) * x^2
def selling_price (a b x : ℝ) : ℝ := a + x / b

-- Calculation for average cost and quantity to minimize average cost
theorem minimize_average_cost :
  ∃ x : ℝ, x = 10000 ∧
  ( let avg_cost : ℝ := (fixed_cost + direct_total_cost x) / x
    in avg_cost = 250 ) :=
sorry

-- Calculation for maximizing profit and solving for a and b
theorem maximize_profit (a b x : ℝ) :
  ( ∀ x : ℝ, profit = x * selling_price a b x - (fixed_cost + direct_total_cost x) → ∀ y : ℝ, profit (y := 15000) && (selling_price a b 15000 = 300) → 
  a = 250 ∧ b = 300 ) :=
sorry

end minimize_average_cost_maximize_profit_l701_701182


namespace parallel_line_slope_l701_701979

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l701_701979


namespace river_depth_mid_may_l701_701375

-- Definitions corresponding to the conditions
def depth_mid_june (D : ℕ) : ℕ := D + 10
def depth_mid_july (D : ℕ) : ℕ := 3 * (depth_mid_june D)

-- The theorem statement
theorem river_depth_mid_may (D : ℕ) (h : depth_mid_july D = 45) : D = 5 :=
by
  sorry

end river_depth_mid_may_l701_701375


namespace magnitude_of_conjugate_complex_l701_701043

theorem magnitude_of_conjugate_complex (α β : ℂ) (h_conj : β = conj α) (h_diff : complex.abs (α - β) = 2 * real.sqrt 3) (h_real : is_real (α / β^2)) : complex.abs α = 2 :=
sorry

end magnitude_of_conjugate_complex_l701_701043


namespace parallel_line_slope_l701_701978

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l701_701978


namespace function_properties_l701_701323

noncomputable def f (x : ℝ) := cos(x)^4 + 2 * sin(x) * cos(x) - sin(x)^4

theorem function_properties :
  (¬(∀ x, f(-x) = f(x)) ∧ ¬(∀ x, f(-x) = -f(x))) ∧
  (∃ T > 0, ∀ x, f(x + T) = f(x)) ∧
  (T = π) ∧
  (∀ k ∈ ℤ, ∀ x, - (3 * π / 8) + k * π ≤ x ∧ x ≤ k * π + (π / 8) → (deriv f x ≥ 0)) ∧
  (∀ x ∈ Icc (0 : ℝ) (π / 2), f(x) ≤ sqrt(2) ∧ f(x) ≥ -1) :=
by
  sorry

end function_properties_l701_701323


namespace positive_conditions_when_x_is_negative_l701_701734

theorem positive_conditions_when_x_is_negative (x : ℝ) (hx : x < 0) :
  (- x / |x| > 0) ∧ (2^(-x) > 0) :=
by
  sorry

end positive_conditions_when_x_is_negative_l701_701734


namespace not_round_to_6514_l701_701560

def round_nearest_hundredth (x : ℝ) : ℝ := 
  (Real.floor (x * 100 + 0.5)) / 100

theorem not_round_to_6514 : 
  round_nearest_hundredth 65.1339999 ≠ 65.14 :=
by
  sorry

end not_round_to_6514_l701_701560


namespace arithmetic_sequence_ninth_term_l701_701888

theorem arithmetic_sequence_ninth_term
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29) :
  a + 8 * d = 35 :=
sorry

end arithmetic_sequence_ninth_term_l701_701888


namespace no_six_in_c_l701_701673

noncomputable def has_median (l : List ℤ) (m : ℤ) : Prop :=
  ∃ (sorted_l : List ℤ), l = sorted_l.sorted ∧ List.nth sorted_l (sorted_l.length / 2) = m

noncomputable def has_mode (l : List ℤ) (m : ℤ) : Prop :=
  ∃ (n : ℕ), n = l.count m ∧ ∀ x, l.count x ≤ n

noncomputable def has_mean (l : List ℤ) (μ : ℤ) : Prop :=
  l.sum / l.length = μ

noncomputable def has_range (l : List ℤ) (r : ℤ) : Prop :=
  l.maximum - l.minimum = r

noncomputable def has_standard_deviation (l : List ℤ) (σ : ℤ) : Prop :=
  let μ := l.sum / l.length in
  (l.map (λ x, (x - μ) ^ 2)).sum / l.length = σ ^ 2

theorem no_six_in_c :
  ∃ l : List ℤ, has_median l 2 ∧ has_range l 2 ∧ ¬ 6 ∈ l :=
begin
  sorry
end

end no_six_in_c_l701_701673


namespace cubic_equation_real_root_l701_701072

theorem cubic_equation_real_root :
  ∃ (a b c : ℕ), ∀ x : ℝ,
  (27 * x^3 - 24 * x^2 - 6 * x - 2 = 0) →
  (x = (real.cbrt a + real.cbrt b + 1) / c) ∧
  (a + b + c = 3) :=
begin
  sorry
end

end cubic_equation_real_root_l701_701072


namespace simplify_and_evaluate_l701_701830
  
theorem simplify_and_evaluate (a : ℤ) 
    (h1 : -real.sqrt 2 < a) 
    (h2 : a < real.sqrt 5) : 
    (a ≠ -1 ∧ a ≠ 2) → 
    (a = 0 → (a - 1 - 3 / (a + 1)) / ((a ^ 2 - 4 * a + 4) / (a + 1)) = -1) ∧ 
    (a = 1 → (a - 1 - 3 / (a + 1)) / ((a ^ 2 - 4 * a + 4) / (a + 1)) = -3) :=
by
  sorry

end simplify_and_evaluate_l701_701830


namespace similitude_circle_is_diameter_KO_l701_701818

variables {ABC : Type} [triangle ABC] (K : point) (O : point)
variables [isLemoinePoint K ABC] [isCircumcenter O ABC]

theorem similitude_circle_is_diameter_KO :
  ∃ (S : circle), S.diameter = segment (K, O) := 
sorry

end similitude_circle_is_diameter_KO_l701_701818


namespace part1_quantity_of_vegetables_part2_functional_relationship_part3_min_vegetable_a_l701_701911

/-- Part 1: Quantities of vegetables A and B wholesaled. -/
theorem part1_quantity_of_vegetables (x y : ℝ) 
  (h1 : x + y = 40) 
  (h2 : 4.8 * x + 4 * y = 180) : 
  x = 25 ∧ y = 15 :=
sorry

/-- Part 2: Functional relationship between m and n. -/
theorem part2_functional_relationship (n m : ℝ) 
  (h : n ≤ 80) 
  (h2 : m = 4.8 * n + 4 * (80 - n)) : 
  m = 0.8 * n + 320 :=
sorry

/-- Part 3: Minimum amount of vegetable A to ensure profit of at least 176 yuan -/
theorem part3_min_vegetable_a (n : ℝ) 
  (h : 0.8 * n + 128 ≥ 176) : 
  n ≥ 60 :=
sorry

end part1_quantity_of_vegetables_part2_functional_relationship_part3_min_vegetable_a_l701_701911


namespace length_KL_l701_701505

variables {Point : Type} [metric_space Point]
variables {A B C D E P Q R S K L M : Point}
variables (AB BC CD DE AE PR QS : ℝ)
variables (midpoint : Point → Point → Point)
variables (length : Point → Point → ℝ)

-- Conditions given
axiom pentagon_eq_sides (AB BC CD DE AE : ℝ) : AB = 1 ∧ BC = 1 ∧ CD = 1 ∧ DE = 1
axiom is_midpoint (X Y Z : Point) :
  midpoint X Y = Z ↔ length X Z = length Y Z / 2
axiom midpoints 
  (P Q R S K L : Point)
  (AB BC CD DE PR QS : ℝ) 
  (MP AB BC CD DE PR QS length midpoint) :
  midpoint A B = P ∧ midpoint B C = Q ∧ midpoint C D = R ∧ midpoint D E = S ∧ 
  midpoint P R = K ∧ midpoint Q S = L

-- Geometry proof problem
theorem length_KL (A B C D E P Q R S K L : Point) :
  pentagon_eq_sides AB BC CD DE AE →
  midpoints P Q R S K L AB BC CD DE PR QS length midpoint →
  length K L = 1 / 4 * length A E := 
sorry

end length_KL_l701_701505


namespace cosine_of_angle_between_vectors_l701_701719

def vec (x y : ℝ) : ℝ × ℝ := (x, y)

def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b = (k * a.1, k * a.2)

theorem cosine_of_angle_between_vectors (k : ℝ) (hk : k ∉ {0})
  (h : collinear (vec 1 k + vec 2 2) (vec 1 k)) :
  real.cos (0 : ℝ) = 1 :=
by
  -- Proof is omitted
  sorry

end cosine_of_angle_between_vectors_l701_701719


namespace smallest_n_l701_701311

theorem smallest_n (n : ℕ) :
  (∀ m : ℤ, 0 < m ∧ m < 2001 →
    ∃ k : ℤ, (m : ℚ) / 2001 < (k : ℚ) / n ∧ (k : ℚ) / n < (m + 1 : ℚ) / 2002) ↔ n = 4003 :=
by
  sorry

end smallest_n_l701_701311


namespace number_of_students_in_class_l701_701608

theorem number_of_students_in_class
  (x : ℕ)
  (S : ℝ)
  (incorrect_score correct_score : ℝ)
  (incorrect_score_mistake : incorrect_score = 85)
  (correct_score_corrected : correct_score = 78)
  (average_difference : ℝ)
  (average_difference_value : average_difference = 0.75)
  (test_attendance : ℕ)
  (test_attendance_value : test_attendance = x - 3)
  (average_difference_condition : (S + incorrect_score) / test_attendance - (S + correct_score) / test_attendance = average_difference) :
  x = 13 :=
by
  sorry

end number_of_students_in_class_l701_701608


namespace books_inequality_system_l701_701246

theorem books_inequality_system (x : ℕ) (n : ℕ) (h1 : x = 5 * n + 6) (h2 : (1 ≤ x - 7 * (x - 6) / 5 + 7)) :
  1 ≤ x - 7 * (x - 6) / 5 + 7 ∧ x - 7 * (x - 6) / 5 + 7 < 7 := 
by
  sorry

end books_inequality_system_l701_701246


namespace what_to_do_first_l701_701433

-- Definition of the conditions
def eat_or_sleep_to_survive (days_without_eat : ℕ) (days_without_sleep : ℕ) : Prop :=
  (days_without_eat = 7 → days_without_sleep ≠ 7) ∨ (days_without_sleep = 7 → days_without_eat ≠ 7)

-- Theorem statement based on the problem and its conditions
theorem what_to_do_first (days_without_eat days_without_sleep : ℕ) :
  days_without_eat = 7 ∨ days_without_sleep = 7 →
  eat_or_sleep_to_survive days_without_eat days_without_sleep :=
by sorry

end what_to_do_first_l701_701433


namespace reflection_matrix_correct_l701_701031

noncomputable def reflect_matrix_through_plane (n : ℝ × ℝ × ℝ) : matrix (fin 3) (fin 3) ℝ :=
  let ⟨a, b, c⟩ := n;
  let d := a^2 + b^2 + c^2;
  let s := 2/d;
  ![![1 - s*a*a, - s*a*b, - s*a*c],
    ![- s*b*a, 1 - s*b*b, - s*b*c],
    ![- s*c*a, - s*c*b, 1 - s*c*c]]

theorem reflection_matrix_correct :
  reflect_matrix_through_plane (2, -1, 2) =
    (matrix.of ![![1/9, 4/9, -8/9], ![4/9, 7/9, 4/9], ![-8/9, 4/9, 10/9]]) :=
 by sorry

end reflection_matrix_correct_l701_701031


namespace probability_bob_closer_to_abe_l701_701528

theorem probability_bob_closer_to_abe :
  let C := (0, 0) in
  let A := (10, 0) in
  ∀ (θ : ℝ), θ ∈ set.Icc 0 (2 * Real.pi) →
  (∃ B : (ℝ × ℝ),
      B = (10 * Real.cos θ, 10 * Real.sin θ) ∧
      (dist B A < dist A C) →
    ∫ θ in 0..(2 * Real.pi), if (θ < (Real.pi / 3)) then 1 else 0) / (2 * Real.pi) = 1 / 3 :=
begin
  sorry,
end

end probability_bob_closer_to_abe_l701_701528


namespace chord_bisected_vertically_by_line_l701_701714

theorem chord_bisected_vertically_by_line (p : ℝ) (h : p > 0) (l : ℝ → ℝ) (focus : ℝ × ℝ) 
  (h_focus: focus = (p / 2, 0)) (h_line: ∀ x, l x ≠ 0) :
  ¬ ∃ (A B : ℝ × ℝ), 
     A.1 ≠ B.1 ∧
     A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1 ∧ 
     (A.1 + B.1) / 2 = focus.1 ∧ 
     l ((A.1 + B.1) / 2) = focus.2 :=
sorry

end chord_bisected_vertically_by_line_l701_701714


namespace slope_of_parallel_line_l701_701991

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l701_701991


namespace fliers_left_for_next_day_l701_701576

def initial_fliers : ℕ := 2500
def morning_fraction : ℚ := 1 / 5
def afternoon_fraction : ℚ := 1 / 4

theorem fliers_left_for_next_day : 
  let fliers_after_morning := initial_fliers - initial_fliers * morning_fraction,
      fliers_after_afternoon := fliers_after_morning - fliers_after_morning * afternoon_fraction
  in fliers_after_afternoon = 1500 := 
by 
  sorry

end fliers_left_for_next_day_l701_701576


namespace quadratic_intersection_l701_701799

theorem quadratic_intersection (a : ℝ) : 
  (∀ x : ℝ, y = (x-a)^2 + (x-2a)^2 + (x-3a)^2 - 2a^2 + a ∧ y = 4 → -12a + 48 > 0) ∧
  (2a > 0) →
  (0 < a ∧ a < 4) :=
sorry

end quadratic_intersection_l701_701799


namespace triangle_perimeter_l701_701305

-- Define the given quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ :=
  x^2 - (5 + m) * x + 5 * m

-- Define the isosceles triangle with sides given by the roots of the equation
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

-- Defining the fact that 2 is a root of the given quadratic equation with an unknown m
lemma two_is_root (m : ℝ) : quadratic_equation m 2 = 0 := sorry

-- Prove that the perimeter of triangle ABC is 12 given the conditions
theorem triangle_perimeter (α β γ : ℝ) (m : ℝ) (h1 : quadratic_equation m α = 0) 
  (h2 : quadratic_equation m β = 0) 
  (h3 : is_isosceles_triangle α β γ) : α + β + γ = 12 := sorry

end triangle_perimeter_l701_701305


namespace rhombus_perimeter_l701_701477

noncomputable def perimeter_of_rhombus_with_diagonals (d1 d2 : ℕ) : ℕ :=
  4 * (Int.sqrt (d1 / 2 * d1 / 2 + d2 / 2 * d2 / 2))

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 14) (h2 : d2 = 48) :
  perimeter_of_rhombus_with_diagonals d1 d2 = 100 := by
  sorry

end rhombus_perimeter_l701_701477


namespace max_teams_tied_most_wins_l701_701754

-- Definitions based on conditions in part a)
def round_robin_tournament (teams : Type) := (team_plays : Π (t u : teams), Prop)

axiom eight_teams (T : Type) (ht : round_robin_tournament T) :
  ∃ (n : ℕ) (teams : Fin n → T), n = 8

axiom each_team_plays_against_every_other_team (T : Type) (ht : round_robin_tournament T) (a b : T) :
  a ≠ b → ht.1 a b

axiom game_result (T : Type) (ht : round_robin_tournament T) 
  (result : ∀ (a b : T), ht.1 a b → (wins : T × T → Prop) 
  (losses : T × T → Prop) (draws : T × T → Prop) :
    ∀ (a b : T), (result a b).1 → ¬(result a b).2 ∧ ¬(result a b).3 ∧
                 (result b a).2 ∧ ¬(result b a).1 ∧ ¬(result b a).3 ∨ 
                 (result a b).3 ∧ (result b a).3 ∧ ¬(result a b).1 ∧ ¬(result b a).1 ∧ 
                 ¬(result a b).2 ∧ ¬(result b a).2

axiom teams_ranked_by_wins (T : Type) (ht : round_robin_tournament T) 
  (rank : (teams : Π (a : T), ℕ) → T → T → Prop) :
  ∀ a b : T, (∑ i in teams, rank i a) > (∑ i in teams, rank i b) ∨
               (∑ i in teams, rank i a) < (∑ i in teams, rank i b) ∨
               (∑ i in teams, rank i a) = (∑ i in teams, rank i b)

-- The main theorem to be proven
theorem max_teams_tied_most_wins (T : Type) (ht : round_robin_tournament T) : 
  ∀ (a b : T), 
  ∑ i in T, wins i a = ∑ i in T, wins i b → -- tie condition
  card { a : T | ∑ i in T, wins i a = k } ≤ 7 :=
sorry

end max_teams_tied_most_wins_l701_701754


namespace tan_20_plus_tan_40_sin_50_times_1_plus_sqrt3_tan_10_l701_701466

noncomputable def proof1 : Prop :=
  tan (20 * (π / 180)) + tan (40 * (π / 180)) + real.sqrt 3 * tan (20 * (π / 180)) * tan (40 * (π / 180)) = real.sqrt 3

noncomputable def proof2 : Prop :=
  sin (50 * (π / 180)) * (1 + real.sqrt 3 * tan (10 * (π / 180))) = 1

theorem tan_20_plus_tan_40 (h1 : proof1) : true :=
by sorry

theorem sin_50_times_1_plus_sqrt3_tan_10 (h2 : proof2) : true :=
by sorry

end tan_20_plus_tan_40_sin_50_times_1_plus_sqrt3_tan_10_l701_701466


namespace field_elements_representation_l701_701788

-- Let k be a field and f(x) be an irreducible polynomial over k of degree n
variables (k : Type*) [field k] {n : ℕ} (f : polynomial k)
-- α is a root of the polynomial f(x)
variable (α : k)

-- Stating that f is irreducible
variable [irreducible f]

-- Defining k(α) as the field extension of k by α
def field_extension := adjoin_root f

-- The main goal: proving that every element of the field k(α) can be expressed as a 
-- polynomial in α of degree less than n with coefficients in k
theorem field_elements_representation : 
  ∀ (β : field_extension f), ∃ (c : fin n → k), β = ∑ i : fin n, c i • (adjoin_root.mk f α ^ (i : ℕ)) := 
sorry

end field_elements_representation_l701_701788


namespace exists_close_sqrt_pair_l701_701026

theorem exists_close_sqrt_pair {n : ℕ} (a : Fin n → ℝ) 
 (h_nonneg : ∀ i, 0 ≤ a i) (h_sum : ∑ i, a i ≤ 25) : 
  ∃ i j : Fin n, i ≠ j ∧ |Real.sqrt (a i) - Real.sqrt (a j)| ≤ 5 / 2003 := 
by
  sorry

end exists_close_sqrt_pair_l701_701026


namespace largest_real_part_z_plus_w_l701_701470

open Complex

theorem largest_real_part_z_plus_w (z w : ℂ) (hz : abs z = 1) (hw : abs w = 1) (h : z * conj w + conj z * w = 1) :
  ∃ (a : ℝ), a = Real.sqrt 3 ∧ (∀ x y : ℝ, x = z.re + w.re → y = z.im + w.im → x ≤ a) :=
sorry

end largest_real_part_z_plus_w_l701_701470


namespace slope_of_parallel_line_l701_701968

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l701_701968


namespace sufficient_condition_l701_701297

variables {V : Type*} [inner_product_space ℝ V]

theorem sufficient_condition (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a + 2 • b = 0) → (∥a - b∥ = ∥a∥ + ∥b∥) :=
by
  sorry

end sufficient_condition_l701_701297


namespace no_integer_factorization_l701_701050

theorem no_integer_factorization (k : ℤ) (h : ¬ 5 ∣ k) : ¬ ∃ (p q : polynomial ℤ), 
  p.degree + q.degree = 5 ∧ p * q = polynomial.C k + (polynomial.X^5 - polynomial.X) := sorry

end no_integer_factorization_l701_701050


namespace xy_plus_y_eq_square_l701_701277

theorem xy_plus_y_eq_square
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z)
  (h : 1 / (x:ℚ) + 1 / (y:ℚ) = 1 / (z:ℚ)) 
  (h_no_common_divisor: Nat.gcd x (Nat.gcd y z) = 1) 
  : ∃ m n : ℕ, x + y = (m + n) * (m + n) :=
begin
  sorry
end

end xy_plus_y_eq_square_l701_701277


namespace arithmetic_sequence_ninth_term_l701_701892

theorem arithmetic_sequence_ninth_term 
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29)
  : a + 8 * d = 35 :=
by
  sorry

end arithmetic_sequence_ninth_term_l701_701892


namespace solve_equation_l701_701150

theorem solve_equation (m x : ℝ) (hm_pos : m > 0) (hm_ne_one : m ≠ 1) :
  7.320 * m^(1 + Real.log x / Real.log 3) + m^(1 - Real.log x / Real.log 3) = m^2 + 1 ↔ x = 3 ∨ x = 1/3 :=
by
  sorry

end solve_equation_l701_701150


namespace ratio_length_to_width_l701_701201

def garden_length := 80
def garden_perimeter := 240

theorem ratio_length_to_width : ∃ W, 2 * garden_length + 2 * W = garden_perimeter ∧ garden_length / W = 2 := by
  sorry

end ratio_length_to_width_l701_701201


namespace plant_arrangement_l701_701780

theorem plant_arrangement (rose_plants daisy_plants : Finset ℕ) :
    rose_plants.card = 5 →
    daisy_plants.card = 5 →
    ∃ (total_arrangements : ℕ), total_arrangements = 86400 :=
by
  intros h_rose h_daisy
  use (6! * 5!)
  sorry

end plant_arrangement_l701_701780


namespace plane_tiled_squares_triangles_percentage_l701_701515

theorem plane_tiled_squares_triangles_percentage :
    (percent_triangle_area : ℚ) = 625 / 10000 := sorry

end plane_tiled_squares_triangles_percentage_l701_701515


namespace train_speed_correct_l701_701193

noncomputable def jogger_speed_km_per_hr := 9
noncomputable def jogger_speed_m_per_s := 9 * 1000 / 3600
noncomputable def train_speed_km_per_hr := 45
noncomputable def distance_ahead_m := 270
noncomputable def train_length_m := 120
noncomputable def total_distance_m := distance_ahead_m + train_length_m
noncomputable def time_seconds := 39

theorem train_speed_correct :
  let relative_speed_m_per_s := total_distance_m / time_seconds
  let train_speed_m_per_s := relative_speed_m_per_s + jogger_speed_m_per_s
  let train_speed_km_per_hr_calculated := train_speed_m_per_s * 3600 / 1000
  train_speed_km_per_hr_calculated = train_speed_km_per_hr :=
by
  sorry

end train_speed_correct_l701_701193


namespace range_of_a_l701_701319

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, 0 < x ∧ x ≤ 3 ∧ a * x^2 + x + 3 * a + 1 = 0) : a ∈ Icc (-1/2) (-1/3) :=
sorry

end range_of_a_l701_701319


namespace length_KL_eq_one_fourth_l701_701491

section Pentagon

variable {A B C D E P Q R S K L : Type}
variable [AddCommGroup P] [Module ℕ P]

-- Assuming all sides of the pentagon are of length 1
variable (length_AB length_BC length_CD length_DE length_AE : ℕ := 1)

-- Define midpoints assumptions
variable (is_midpoint_AB : P = midpoint A B)
variable (is_midpoint_BC : Q = midpoint B C)
variable (is_midpoint_CD : R = midpoint C D)
variable (is_midpoint_DE : S = midpoint D E)

-- Define KL midpoints
variable (is_midpoint_PR : K = midpoint P R)
variable (is_midpoint_QS : L = midpoint Q S)

-- Prove the length of segment KL
theorem length_KL_eq_one_fourth :
  dist K L = 1 / 4 :=
sorry

end Pentagon

end length_KL_eq_one_fourth_l701_701491


namespace omega_range_l701_701712

theorem omega_range (ω : ℝ) : 
  (∀ x ∈ Ioo (0 : ℝ) (π / 6), ∀ y ∈ Ioo x (π / 6), 
  (sin (ω * x + π / 6) < sin (ω * y + π / 6))) ∧ 
  (∀ x ∈ Ioo (π / 6) (π / 3), (sin (ω * x + π / 6) ≤ sin (ω * (π / 3) + π / 6))) → 
  (1 < ω ∧ ω < 2) := 
sorry

end omega_range_l701_701712


namespace find_a10_l701_701419

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ m n : ℕ, m > 0 → n > 0 → a (m + n) = a m + a n + m * n

theorem find_a10 (a : ℕ → ℕ) (h : sequence a) : a 10 = 65 :=
sorry

end find_a10_l701_701419


namespace smallest_b_multiple_of_6_and_15_is_30_l701_701269

theorem smallest_b_multiple_of_6_and_15_is_30 : ∃ b : ℕ, b > 0 ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ b = 30 :=
by
  use 30
  split
  . trivial
  . split
    . sorry
    . sorry

end smallest_b_multiple_of_6_and_15_is_30_l701_701269


namespace spending_difference_is_65_l701_701776

-- Definitions based on conditions
def ice_cream_cones : ℕ := 15
def pudding_cups : ℕ := 5
def ice_cream_cost_per_unit : ℝ := 5
def pudding_cost_per_unit : ℝ := 2

-- The solution requires the calculation of the total cost and the difference
def total_ice_cream_cost : ℝ := ice_cream_cones * ice_cream_cost_per_unit
def total_pudding_cost : ℝ := pudding_cups * pudding_cost_per_unit
def spending_difference : ℝ := total_ice_cream_cost - total_pudding_cost

-- Theorem statement proving the difference is 65
theorem spending_difference_is_65 : spending_difference = 65 := by
  -- The proof is omitted with sorry
  sorry

end spending_difference_is_65_l701_701776


namespace batsman_average_after_11th_inning_l701_701151

theorem batsman_average_after_11th_inning
  (x : ℝ)  -- the average score of the batsman before the 11th inning
  (h1 : 10 * x + 85 = 11 * (x + 5))  -- given condition from the problem
  : x + 5 = 35 :=   -- goal statement proving the new average
by
  -- We need to prove that new average after the 11th inning is 35
  sorry

end batsman_average_after_11th_inning_l701_701151


namespace cost_of_pen_l701_701199

-- define the conditions
def notebook_cost (pen_cost : ℝ) : ℝ := 3 * pen_cost
def total_cost (notebook_cost : ℝ) : ℝ := 4 * notebook_cost

-- theorem stating the problem we need to prove
theorem cost_of_pen (pen_cost : ℝ) (h1 : total_cost (notebook_cost pen_cost) = 18) : pen_cost = 1.5 :=
by
  -- proof to be constructed
  sorry

end cost_of_pen_l701_701199


namespace circle_area_difference_l701_701574

noncomputable def difference_of_circle_areas (C1 C2 : ℝ) : ℝ :=
  let π := Real.pi
  let r1 := C1 / (2 * π)
  let r2 := C2 / (2 * π)
  let A1 := π * r1 ^ 2
  let A2 := π * r2 ^ 2
  A2 - A1

theorem circle_area_difference :
  difference_of_circle_areas 396 704 = 26948.4 :=
by
  sorry

end circle_area_difference_l701_701574


namespace vladimir_digits_l701_701547

theorem vladimir_digits (a b c : ℕ) (abc cba cab : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (h4 : abc = 100 * a + 10 * b + c) (h5 : cba = 100 * c + 10 * b + a)
  (h6 : cab = 100 * c + 10 * a + b) (h7 : abc = cba + cab) : 
  (a = 9 ∧ b = 5 ∧ c = 4) :=
begin
  sorry
end

end vladimir_digits_l701_701547


namespace point_symmetric_property_l701_701300

def point := (ℝ × ℝ)

def symmetric_with_respect_to_x_axis (p : point) : point :=
  (p.1, -p.2)

theorem point_symmetric_property :
  let P : point := (-2, 1)
  let Q : point := symmetric_with_respect_to_x_axis P
  Q = (-2, -1) :=
by
  intros
  simp [symmetric_with_respect_to_x_axis]
  sorry

end point_symmetric_property_l701_701300


namespace parallel_slope_l701_701932

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l701_701932


namespace Greg_PPO_reward_is_108_l701_701721

variable (max_ProcGen_reward : ℝ := 240)
variable (PPO_percentage : ℝ := 0.9)

def max_CoinRun_reward : ℝ := max_ProcGen_reward / 2

def Greg_PPO_reward : ℝ := PPO_percentage * max_CoinRun_reward

theorem Greg_PPO_reward_is_108
  (H1 : max_ProcGen_reward = 240)
  (H2 : PPO_percentage = 0.9)
  : Greg_PPO_reward = 108 := by {
    -- Proof omitted
    sorry
}

end Greg_PPO_reward_is_108_l701_701721


namespace calculate_expression_l701_701231

theorem calculate_expression : 
  (sqrt 48 - 3 * sqrt (1 / 3)) / sqrt 3 = 3 := 
by 
  sorry

end calculate_expression_l701_701231


namespace hanks_pancakes_needed_l701_701624

/-- Hank's pancake calculation problem -/
theorem hanks_pancakes_needed 
    (pancakes_per_big_stack : ℕ := 5)
    (pancakes_per_short_stack : ℕ := 3)
    (big_stack_orders : ℕ := 6)
    (short_stack_orders : ℕ := 9) :
    (pancakes_per_short_stack * short_stack_orders) + (pancakes_per_big_stack * big_stack_orders) = 57 := by {
  sorry
}

end hanks_pancakes_needed_l701_701624


namespace exp_to_rectangular_l701_701640

noncomputable def euler_formula (x : ℝ) : ℂ := complex.exp (x * complex.I) = complex.cos x + complex.sin x * complex.I

theorem exp_to_rectangular (x : ℝ) :
  euler_formula (7 * π / 3) = (1 / 2 : ℝ) + (complex.sqrt 3 / 2) * complex.I := by
  unfold euler_formula
  apply complex.exp_eq_cos_add_sin_mul_I
  -- rest of proof steps
  sorry

end exp_to_rectangular_l701_701640


namespace slope_of_parallel_line_l701_701969

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l701_701969


namespace increase_in_circumference_l701_701786

theorem increase_in_circumference (d e : ℝ) : (fun d e => let C := π * d; let C_new := π * (d + e); C_new - C) d e = π * e :=
by sorry

end increase_in_circumference_l701_701786


namespace count_digits_first_1500_even_l701_701135

theorem count_digits_first_1500_even :
  (∑ i in (finset.range 1500).map (λ n, 2*(n+1)), nat.digits 10 i).sum = 5448 :=
sorry

end count_digits_first_1500_even_l701_701135


namespace quadrilateral_area_l701_701394

-- Define the angles in the quadrilateral ABCD
def ABD : ℝ := 20
def DBC : ℝ := 60
def ADB : ℝ := 30
def BDC : ℝ := 70

-- Define the side lengths
variables (AB CD AD BC AC BD : ℝ)

-- Prove that the area of the quadrilateral ABCD is half the product of its sides
theorem quadrilateral_area (h1 : ABD = 20) (h2 : DBC = 60) (h3 : ADB = 30) (h4 : BDC = 70)
  : (1 / 2) * (AB * CD + AD * BC) = (1 / 2) * (AB * CD + AD * BC) :=
by
  sorry

end quadrilateral_area_l701_701394


namespace circle_radius_through_incenter_l701_701162

noncomputable def radius_of_circle_passing_through_incenter
  (b : ℝ) (α : ℝ) (h_nonzero_cos: cos (α / 2) ≠ 0) : ℝ :=
  b / (2 * cos (α / 2))

theorem circle_radius_through_incenter (b α : ℝ) (h_nonzero_cos: cos (α / 2) ≠ 0):
  radius_of_circle_passing_through_incenter b α h_nonzero_cos = b / (2 * cos (α / 2)) := 
by 
  simp [radius_of_circle_passing_through_incenter]
  sorry

end circle_radius_through_incenter_l701_701162


namespace farmer_children_count_l701_701187

def apples_each_bag := 15
def eaten_each := 4
def sold_apples := 7
def apples_left := 60

theorem farmer_children_count : 
  ∃ (n : ℕ), 15 * n - (2 * 4 + 7) = 60 ∧ n = 5 :=
by
  use 5
  sorry

end farmer_children_count_l701_701187


namespace problem_arithmetic_sequence_l701_701687

-- Define the arithmetic sequence and its properties
def a (n : ℕ) : ℤ := 2 - n

-- Define the sum of the first n terms of the sequence {a_n * 3^(n-1)}
def s (n : ℕ) : ℤ := ((5 / 4 : ℚ) - (n / 2 : ℚ)) * (3^n : ℚ) - (5 / 4 : ℚ)

theorem problem_arithmetic_sequence :
  ∀ n : ℕ, 
    (a 2 = 0) ∧ (a 6 + a 8 = -10) →
    (a n = 2 - n) ∧ (s n = ((5 / 4 : ℚ) - (n / 2 : ℚ)) * (3^n : ℚ) - (5 / 4 : ℚ)) :=
by
  sorry

end problem_arithmetic_sequence_l701_701687


namespace latoya_call_duration_l701_701022

theorem latoya_call_duration
  (initial_credit remaining_credit : ℝ) (cost_per_minute : ℝ) (t : ℝ)
  (h1 : initial_credit = 30)
  (h2 : remaining_credit = 26.48)
  (h3 : cost_per_minute = 0.16)
  (h4 : initial_credit - remaining_credit = t * cost_per_minute) :
  t = 22 := 
sorry

end latoya_call_duration_l701_701022


namespace taxi_driver_analysis_l701_701076

theorem taxi_driver_analysis 
  (trips : List ℤ)
  (start_price : ℕ)
  (extra_fare_per_km : ℕ)
  (fare_threshold : ℕ)
  (fuel_consumption_per_100_km : ℝ)
  (fuel_price_per_liter : ℝ) :
  trips = [-2, -3, -6, 8, -9, -7, -5, 13] →
  start_price = 10 →
  extra_fare_per_km = 2 →
  fare_threshold = 3 →
  fuel_consumption_per_100_km = 8 →
  fuel_price_per_liter = 8 →
  let
    total_distance := trips.sum_nat_abs,
    final_position := trips.sum,
    total_fare := trips.map (fun trip => 
      start_price + if abs trip ≤ fare_threshold then 0 
      else extra_fare_per_km * (abs trip - fare_threshold))
      .sum,
    fuel_cost := total_distance * fuel_consumption_per_100_km / 100 * fuel_price_per_liter,
    profit := total_fare - fuel_cost
  in
  final_position = -11 ∧
  total_fare = 140 ∧
  profit = 106.08 :=
begin
  sorry
end

end taxi_driver_analysis_l701_701076


namespace tim_movie_marathon_l701_701120

variables (first_movie second_movie third_movie fourth_movie fifth_movie sixth_movie seventh_movie : ℝ)

/-- Tim's movie marathon --/
theorem tim_movie_marathon
  (first_movie_duration : first_movie = 2)
  (second_movie_duration : second_movie = 1.5 * first_movie)
  (third_movie_duration : third_movie = 0.8 * (first_movie + second_movie))
  (fourth_movie_duration : fourth_movie = 2 * second_movie)
  (fifth_movie_duration : fifth_movie = third_movie - 0.5)
  (sixth_movie_duration : sixth_movie = (second_movie + fourth_movie) / 2)
  (seventh_movie_duration : seventh_movie = 45 / fifth_movie) :
  first_movie + second_movie + third_movie + fourth_movie + fifth_movie + sixth_movie + seventh_movie = 35.8571 :=
sorry

end tim_movie_marathon_l701_701120


namespace slope_parallel_to_original_line_l701_701943

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l701_701943


namespace maximum_length_OP_l701_701742

open Real

def line1 (k x y : ℝ) := k * x - y - k + 2 = 0
def line2 (k x y : ℝ) := x + k * y - 2 * k - 3 = 0

theorem maximum_length_OP (k x y : ℝ) :
  (line1 k x y) ∧ (line2 k x y) →
  let Px := (k^2 + 3) / (k^2 + 1);
      Py := (2 * k) / (k^2 + 1) + 2 in
  ∀ x y, max_dist : ℝ,
  max_dist = sqrt((Px - 0)^2 + (Py - 0)^2) →
  max_dist = 2 * sqrt 2 + 1 :=
sorry

end maximum_length_OP_l701_701742


namespace number_of_ordered_pairs_l701_701787

noncomputable def max (x y : ℕ) : ℕ := if x > y then x else y

def valid_pair_count (k : ℕ) : ℕ := 2 * k + 1

def pairs_count (a b : ℕ) : ℕ := 
  valid_pair_count 5 * valid_pair_count 3 * valid_pair_count 2 * valid_pair_count 1

theorem number_of_ordered_pairs : pairs_count 2 3 = 1155 := 
  sorry

end number_of_ordered_pairs_l701_701787


namespace slope_of_parallel_line_l701_701928

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l701_701928


namespace find_f_l701_701316

def f (x : ℝ) : ℝ := 2 * x * f' 1 + Real.log x -- Define the function f(x)
def f' := λ x, 2 * f' 1 + 1 / x -- Define the derivative f'(x)

theorem find_f'_at_1 : f' 1 = -1 :=
by sorry

end find_f_l701_701316


namespace height_of_water_l701_701053

theorem height_of_water :
  ∃ h : ℝ, 
    (2 * 4 * h + π * 1^2 * h = 80) ∧ 
    (h ≈ 80 / (8 + π)) :=
by {
  use 80 / (8 + π),
  split,
  {
    ring_nf,
    norm_num,
    sorry, -- Volume equation check
  },
  {
    field_simp [add_comm _ π],
    norm_num,
    linarith,
    sorry, -- Approximation check
  }
}

end height_of_water_l701_701053


namespace sequence_a_n_l701_701765

theorem sequence_a_n (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n > 0 → (n^2 + n) * (a (n + 1) - a n) = 2) :
  a 20 = 29 / 10 :=
by
  sorry

end sequence_a_n_l701_701765


namespace unit_cost_decreases_l701_701878

def regression_equation (x : ℝ) : ℝ := 356 - 1.5 * x

theorem unit_cost_decreases (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -1.5 := 
by sorry


end unit_cost_decreases_l701_701878


namespace smallest_number_of_blocks_l701_701171

theorem smallest_number_of_blocks (wall_length : ℝ) (wall_height : ℝ)
    (block_lengths : Set ℝ) (block_height : ℝ)
    (h_wall_length : wall_length = 120)
    (h_wall_height : wall_height = 8)
    (h_block_lengths : block_lengths = {1, 2, 3})
    (h_block_height : block_height = 1)
    (h_staggered: ∀ (i: ℕ), staggered (blocks_in_row i))
    (h_even_ends: ∀ (i: ℕ), even_ends (blocks_in_row i))
    (h_no_adj_same_start: ∀ (i : ℕ), start_diff (blocks_in_row i) (blocks_in_row (i + 1))) :
    num_blocks_needed wall_length wall_height block_lengths block_height = 480 := 
sorry

end smallest_number_of_blocks_l701_701171


namespace PS_div_QR_eq_sqrt3_l701_701382

open Real

theorem PS_div_QR_eq_sqrt3 {P Q R S : Type} [normed_field P] [normed_field Q] [normed_field R] [normed_field S] :
  (QR : ℝ) = t → is_equilateral (triangle P Q R) → is_equilateral (triangle Q R S) → PS / QR = sqrt 3 :=
by
  intros hQR hPQR hQRS
  sorry

end PS_div_QR_eq_sqrt3_l701_701382


namespace sum_of_products_l701_701634

theorem sum_of_products : 1 * 15 + 2 * 14 + 3 * 13 + 4 * 12 + 5 * 11 + 6 * 10 + 7 * 9 + 8 * 8 = 372 := by
  sorry

end sum_of_products_l701_701634


namespace probability_boy_girl_selection_l701_701594

/-- 
Given a group with 8 boys and 3 girls, and randomly selecting 1 boy and 2 girls,
we need to prove that the probability of selecting both boy A and girl B equals 1/12.
-/
theorem probability_boy_girl_selection 
  (group : Finset (String × Bool)) -- Assume a set of tuples where (name, is_boy)
  (boyA : String) (girlB : String) 
  (hA : (boyA, true) ∈ group) 
  (hB : (girlB, false) ∈ group)
  (boys : Finset (String × Bool)) 
  (girls : Finset (String × Bool)) 
  (hboys : boys = group.filter (fun x => x.2)) 
  (hgirls : girls = group.filter (fun x => ¬x.2)) 
  (hsize_boys : boys.card = 8)
  (hsize_girls : girls.card = 3) :
  (1 : ℚ) / (boys.card.choose 1 * girls.card.choose 2) = 1 / 12 :=
begin
  sorry,
end

end probability_boy_girl_selection_l701_701594


namespace farmer_children_count_l701_701188

def apples_each_bag := 15
def eaten_each := 4
def sold_apples := 7
def apples_left := 60

theorem farmer_children_count : 
  ∃ (n : ℕ), 15 * n - (2 * 4 + 7) = 60 ∧ n = 5 :=
by
  use 5
  sorry

end farmer_children_count_l701_701188


namespace power_mod_remainder_l701_701552

theorem power_mod_remainder (a : ℕ) (n : ℕ) (h1 : 3^5 % 11 = 1) (h2 : 221 % 5 = 1) : 3^221 % 11 = 3 :=
by
  sorry

end power_mod_remainder_l701_701552


namespace triangle_area_correct_l701_701910

-- Define the points of intersection based on the given conditions
def A : ℝ × ℝ := (3, 3)
def B : ℝ × ℝ := (4.5, 7.5)
def C : ℝ × ℝ := (7.5, 4.5)

-- Function to calculate the area of a triangle given three points using the determinant formula
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))).abs

-- The theorem to be proven
theorem triangle_area_correct : triangle_area A B C = 8.625 :=
by 
  -- Placeholder for the actual proof
  sorry

end triangle_area_correct_l701_701910


namespace sin_cos_sequences_equality_no_geometric_sequence_l701_701304

open Real

-- Part 1: Proving the equality involving sin and cos of sequences
theorem sin_cos_sequences_equality (a b : ℕ → ℝ)
  (h1 : ∀ n, sin (a n.succ) = sin (a n) + cos (b n))
  (h2 : ∀ n, cos (b n.succ) = cos (b n) - sin (a n))
  : ∀ n, sin (a n.succ)^2 + cos (b n.succ)^2 = 2 * (sin (a n)^2 + cos (b n)^2) :=
sorry

-- Part 2: Proving the existence or non-existence of initial conditions for a geometric sequence
theorem no_geometric_sequence (a b : ℕ → ℝ)
  (h1 : ∀ n, sin (a n.succ) = sin (a n) + cos (b n))
  (h2 : ∀ n, cos (b n.succ) = cos (b n) - sin (a n))
  : ∀ a1 b1, ¬ (∃ r, ∀ n, sin (a (n+1))^2 + cos (b (n+1))^2 = r * (sin (a n)^2 + cos (b n)^2)) :=
sorry

end sin_cos_sequences_equality_no_geometric_sequence_l701_701304


namespace alcohol_water_ratio_l701_701575

theorem alcohol_water_ratio 
  (P_alcohol_pct : ℝ) (Q_alcohol_pct : ℝ) 
  (P_volume : ℝ) (Q_volume : ℝ) 
  (mixture_alcohol : ℝ) (mixture_water : ℝ)
  (h1 : P_alcohol_pct = 62.5)
  (h2 : Q_alcohol_pct = 87.5)
  (h3 : P_volume = 4)
  (h4 : Q_volume = 4)
  (ha : mixture_alcohol = (P_volume * (P_alcohol_pct / 100)) + (Q_volume * (Q_alcohol_pct / 100)))
  (hm : mixture_water = (P_volume + Q_volume) - mixture_alcohol) :
  mixture_alcohol / mixture_water = 3 :=
by
  sorry

end alcohol_water_ratio_l701_701575


namespace parallel_line_slope_l701_701982

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l701_701982


namespace slope_of_parallel_line_l701_701972

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l701_701972


namespace person_a_age_l701_701109

theorem person_a_age (A B : ℕ) (h1 : A + B = 43) (h2 : A + 4 = B + 7) : A = 23 :=
by sorry

end person_a_age_l701_701109


namespace maximum_xy_l701_701693

theorem maximum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_parallel : 2 * x + y = 2) : 
  xy ≤ 1/2 := 
  sorry

end maximum_xy_l701_701693


namespace find_c_l701_701006

theorem find_c (c : ℝ) (h : (-c / 4) + (-c / 7) = 22) : c = -56 :=
by
  sorry

end find_c_l701_701006


namespace scientific_notation_of_20_ns_l701_701386

def nanosecond_to_scientific_notation (n: ℕ) : ℝ := n * 10⁻⁹

theorem scientific_notation_of_20_ns :
  nanosecond_to_scientific_notation 20 = 2 * 10⁻⁸ := 
by {
  sorry
}

end scientific_notation_of_20_ns_l701_701386


namespace man_speed_in_still_water_l701_701152

theorem man_speed_in_still_water :
  ∃ (V_m V_s : ℝ), 
  V_m + V_s = 14 ∧ 
  V_m - V_s = 6 ∧ 
  V_m = 10 :=
by
  sorry

end man_speed_in_still_water_l701_701152


namespace P_on_line_l_min_dist_correct_l701_701379

-- Define the line equation
def line_l (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the parametric equation of curve C
def curve_C (α : ℝ) : ℝ × ℝ := (sqrt 3 * cos α, sin α)

-- Define the point P in Cartesian coordinates, given polar coordinates
def point_P : ℝ × ℝ := (0, 4)

-- Prove that point P lies on line l
theorem P_on_line_l : line_l (point_P.fst) (point_P.snd) :=
by
  -- Proof placeholder
  sorry

-- Define the distance from point Q on curve C to line l
def distance (α : ℝ) : ℝ :=
  abs (sqrt 3 * cos α - sin α + 4) / sqrt 2

-- Find the minimum distance
def min_distance : ℝ := sqrt 2

-- Prove that the minimum distance from Q to line l is sqrt 2
theorem min_dist_correct : ∃ α : ℝ, 0 ≤ α ∧ α < 2 * π ∧ distance α = min_distance :=
by
  -- Proof placeholder
  sorry

end P_on_line_l_min_dist_correct_l701_701379


namespace number_of_polynomials_deg_le_3_abs_sum_coefs_eq_5_l701_701241

theorem number_of_polynomials_deg_le_3_abs_sum_coefs_eq_5 :
  ∃ (n : ℕ) (polys : list (fin n → ℤ)), n ≤ 4 ∧
  (∀ p ∈ polys, (∑ i, |p i|) = 5) ∧ polys.length = 32 :=
sorry

end number_of_polynomials_deg_le_3_abs_sum_coefs_eq_5_l701_701241


namespace geometric_properties_l701_701581

-- Definitions of geometric entities and properties.

variable {α : Type*} [MetricSpace α] (O A B D S T E F H G : α)
variable (circle_O : α → ℝ) -- Definition of the circle
variable (tangent_DS tangent_DT : α → Prop) -- Tangents definitions
variable (secant_DEF : α → Prop) -- Secant definition
variable (harmonic_division : α → α → α → α → Prop)
variable (PowerOfCircle : α → α → ℝ)

-- Conditions
def point_on_extension_of_diameter : Prop :=
  (collinear O A B) ∧ (collinear A B D)

def tangents : Prop :=
  tangent_DS S ∧ tangent_DT T

def secant_intersects_circle : Prop :=
  secant_DEF E ∧ secant_DEF F

def intersection_points : Prop :=
  S ≠ T ∧ E ≠ F ∧ intersection ST EF = H

-- Define the harmonic division property
def harmonic : Prop :=
  harmonic_division D H E F

-- Define power of a point property
def power_of_point_theorem : Prop :=
  let GF := dist G F in
  let GH := dist G H in
  let GD := dist G D in
  GF^2 = GH * GD

-- Main theorem statement
theorem geometric_properties :
  point_on_extension_of_diameter O A B D ∧
  tangents D S T O ∧ 
  secant_intersects_circle D E F O ∧
  intersection_points S T E F H →
  harmonic D H E F ∧ 
  power_of_point_theorem G F H D O :=
begin
  intros,
  split,
  {
    -- Proof for harmonic division (DH, EF) = -1
    sorry
  },
  {
    -- Proof for GF^2 = GH * GD
    sorry
  }
end

end geometric_properties_l701_701581


namespace new_students_admitted_l701_701159

theorem new_students_admitted (S : ℕ) (h1 : 24 * S = 24 * 13) (h2 : S + 3 = 16) : 
  21 * 16 - 24 * 13 = 24 :=
by
  have hS : S = 13 := by linarith
  rw [hS]
  have h_total_students_before : 24 * 13 = 312 := by norm_num
  have h_total_students_after : 21 * 16 = 336 := by norm_num
  have h_diff : 336 - 312 = 24 := by norm_num
  exact h_diff

end new_students_admitted_l701_701159


namespace parabola_directrix_l701_701085

theorem parabola_directrix (x y : ℝ) (h : y = - (1/8) * x^2) : y = 2 :=
sorry

end parabola_directrix_l701_701085


namespace length_of_KL_l701_701495

open Finset
open Classical

-- Define the pentagon and relevant points
def Pentagon :=
  {a b c d e : Point}

-- Conditions: all sides of the pentagon are equal to 1
axiom side_lengths (p : Pentagon) : length (p.a, p.b) = 1 ∧ length (p.b, p.c) = 1 ∧ length (p.c, p.d) = 1 ∧ length (p.d, p.e) = 1 ∧ length (p.e, p.a) = 1

-- Define midpoints
def midpoint (p1 p2 : Point) : Point
axiom midpoint_is_between (p1 p2 : Point) : midpoint p1 p2 = (p1 + p2) / 2

-- Define points P, Q, R, S, K, L based on the problem statement
def P (p : Pentagon) := midpoint p.a p.b
def Q (p : Pentagon) := midpoint p.b p.c
def R (p : Pentagon) := midpoint p.c p.d
def S (p : Pentagon) := midpoint p.d p.e
def K (p : Pentagon) := midpoint (P p) (R p)
def L (p : Pentagon) := midpoint (Q p) (S p)

-- The theorem stating the length of segment KL
theorem length_of_KL (p : Pentagon) :
  length (K p) (L p) = 1 / 4 := sorry

end length_of_KL_l701_701495


namespace divide_weights_into_three_equal_piles_l701_701115

theorem divide_weights_into_three_equal_piles : 
  (∃ (pile1 pile2 pile3 : finset ℕ), 
     pile1 ∪ pile2 ∪ pile3 = finset.range 556 ∧ 
     pile1 ∩ pile2 = ∅ ∧ 
     pile2 ∩ pile3 = ∅ ∧ 
     pile3 ∩ pile1 = ∅ ∧ 
     pile1.sum (λ x, x) = pile2.sum (λ x, x) ∧ 
     pile2.sum (λ x, x) = pile3.sum (λ x, x) ∧ 
     pile1.sum (λ x, x) = 51430) := sorry

end divide_weights_into_three_equal_piles_l701_701115


namespace Tyler_CDs_after_giveaway_and_purchase_l701_701542

theorem Tyler_CDs_after_giveaway_and_purchase :
  (∃ cds_initial cds_giveaway_fraction cds_bought cds_final, 
     cds_initial = 21 ∧ 
     cds_giveaway_fraction = 1 / 3 ∧ 
     cds_bought = 8 ∧ 
     cds_final = cds_initial - (cds_initial * cds_giveaway_fraction) + cds_bought ∧
     cds_final = 22) := 
sorry

end Tyler_CDs_after_giveaway_and_purchase_l701_701542


namespace sum_of_three_integers_l701_701100

theorem sum_of_three_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) (h7 : a * b * c = 5^3) : a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_l701_701100


namespace D_is_quadratic_l701_701147

-- Define the equations
def eq_A (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def eq_B (x : ℝ) : Prop := 2 * x^2 - 3 * x = 2 * (x^2 - 2)
def eq_C (x : ℝ) : Prop := x^3 - 2 * x + 7 = 0
def eq_D (x : ℝ) : Prop := (x - 2)^2 - 4 = 0

-- Define what it means to be a quadratic equation
def is_quadratic (f : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x ↔ a * x^2 + b * x + c = 0)

theorem D_is_quadratic : is_quadratic eq_D :=
sorry

end D_is_quadratic_l701_701147


namespace river_depth_mid_may_l701_701376

variable (DepthMidMay DepthMidJune DepthMidJuly : ℕ)

theorem river_depth_mid_may :
  (DepthMidJune = DepthMidMay + 10) →
  (DepthMidJuly = 3 * DepthMidJune) →
  (DepthMidJuly = 45) →
  DepthMidMay = 5 :=
by
  intros h1 h2 h3
  sorry

end river_depth_mid_may_l701_701376


namespace slope_parallel_to_original_line_l701_701944

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l701_701944


namespace original_amount_in_cookie_jar_l701_701898

theorem original_amount_in_cookie_jar (doris_spent martha_spent money_left_in_jar original_amount : ℕ)
  (h1 : doris_spent = 6)
  (h2 : martha_spent = doris_spent / 2)
  (h3 : money_left_in_jar = 15)
  (h4 : original_amount = money_left_in_jar + doris_spent + martha_spent) :
  original_amount = 24 := 
sorry

end original_amount_in_cookie_jar_l701_701898


namespace slope_of_parallel_line_l701_701956

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l701_701956


namespace total_frogs_l701_701113

theorem total_frogs (frogs_inside frogs_outside : Nat) (h1 : frogs_inside = 12) (h2 : frogs_outside = 6) :
    frogs_inside + frogs_outside = 18 :=
by
  rw [h1, h2]
  sorry

end total_frogs_l701_701113


namespace locus_of_point_M_l701_701697

open Real

def distance (x y: ℝ × ℝ): ℝ :=
  ((x.1 - y.1)^2 + (x.2 - y.2)^2)^(1/2)

theorem locus_of_point_M :
  (∀ (M : ℝ × ℝ), 
     distance M (2, 0) + 1 = abs (M.1 + 3)) 
  → ∀ (M : ℝ × ℝ), M.2^2 = 8 * M.1 :=
sorry

end locus_of_point_M_l701_701697


namespace triangle_side_comparison_l701_701080

-- Let triangle ABC be a triangle where D is the foot of the perpendicular from A to BC.
variables (A B C D : Point)
variable [Nonempty (Triangle A B C)] -- We assume ABC forms a triangle

-- Define point D as the foot of the perpendicular from A to BC
def is_foot_of_perpendicular (A B C D : Point) : Prop :=
  ∃ (h : Height A BC), D = h.foot

-- Assume angle BAD > angle CAD
variable (h1 : ∠BAD > ∠CAD)

-- State the theorem
theorem triangle_side_comparison : AB > AC :=
sorry

end triangle_side_comparison_l701_701080


namespace trajectory_intersection_l701_701365

-- Define the Cartesian coordinates and necessary conditions
variables {a b c x y : ℝ} (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)

-- Assume the conditions outlined in the problem
def conditions : Prop :=
  A = (a, 0) ∧ 
  B = (b, 0) ∧ 
  C = (0, c) ∧ 
  a ≠ 0 ∧ 
  b ≠ 0 ∧ 
  a ≠ b ∧ 
  c ≠ 0 

-- Define the proof problem statement
theorem trajectory_intersection 
    (h : conditions A B C) :
  ∃ x y, (x - b / 2)^2 / (b^2 / 4) + y^2 / (ab / 4) = 1 := 
sorry

end trajectory_intersection_l701_701365


namespace lexi_laps_l701_701796

theorem lexi_laps (total_distance lap_distance : ℝ) (h1 : total_distance = 3.25) (h2 : lap_distance = 0.25) :
  total_distance / lap_distance = 13 :=
by
  sorry

end lexi_laps_l701_701796


namespace hyperbola_eccentricity_l701_701326

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hyp : ∀ (x y : ℝ), 3 * x + 2 * y = 0 → x / a = - y / b) :
  let e := (λ (a b : ℝ), Real.sqrt (a ^ 2 + b ^ 2) / a) in
  e a b = Real.sqrt 13 / 2 :=
by 
  sorry

end hyperbola_eccentricity_l701_701326


namespace gcd_459_357_polynomial_at_neg4_l701_701167

-- Statement for the GCD problem
theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

-- Definition of the polynomial
def f (x : Int) : Int :=
  12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

-- Statement for the polynomial evaluation problem
theorem polynomial_at_neg4 : f (-4) = 3392 := by
  sorry

end gcd_459_357_polynomial_at_neg4_l701_701167


namespace vector_subtraction_equal_l701_701348

variables (x : ℝ)

def a := (x, 2 : ℝ × ℝ)
def b := (1/2, 1 : ℝ × ℝ)
def c := (a.1 + 2 * b.1, a.2 + 2 * b.2)
def d := (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem vector_subtraction_equal :
  (c x).1 - 2 * (d x).1 = -1 ∧ (c x).2 - 2 * (d x).2 = -2 := sorry

end vector_subtraction_equal_l701_701348


namespace cards_not_power_of_two_l701_701063

theorem cards_not_power_of_two :
  ∀ (numbers : Fin 88889 → Fin 88889 → ℕ),
  (∀ i j, 11111 ≤ numbers i j ∧ numbers i j ≤ 99999) →
  (¬ ∃ (S : ℕ), (∑ i in Finset.range 88889, numbers ⟨i, by norm_num⟩ ⟨i, by norm_num⟩ * 10^(5 * i)) = 2 ^ S) := by
  sorry

end cards_not_power_of_two_l701_701063


namespace slope_parallel_to_original_line_l701_701941

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l701_701941


namespace win_sector_area_l701_701178

theorem win_sector_area (r : ℝ) (p_win : ℝ) (area_total : ℝ) 
  (h1 : r = 8)
  (h2 : p_win = 3 / 8)
  (h3 : area_total = π * r^2) :
  ∃ area_win, area_win = 24 * π ∧ area_win = p_win * area_total :=
by
  sorry

end win_sector_area_l701_701178


namespace Bill_composes_20_problems_l701_701627

theorem Bill_composes_20_problems :
  ∀ (B : ℕ), (∀ R : ℕ, R = 2 * B) →
    (∀ F : ℕ, F = 3 * R) →
    (∀ T : ℕ, T = 4) →
    (∀ P : ℕ, P = 30) →
    (∀ F : ℕ, F = T * P) →
    (∃ B : ℕ, B = 20) :=
by sorry

end Bill_composes_20_problems_l701_701627


namespace radius_of_circle_with_equal_area_l701_701128

-- Define the conditions
def radius_sphere := 2 -- Radius of the sphere
def surface_area_sphere (R : ℝ) := 4 * Math.pi * R^2 -- Surface area of the sphere
def area_circle (r : ℝ) := Math.pi * r^2 -- Area of the circle

-- The theorem to prove
theorem radius_of_circle_with_equal_area (R r : ℝ) (h1 : R = radius_sphere) (h2 : surface_area_sphere R = area_circle r) : r = 4 :=
by
  -- Given conditions
  sorry -- Placeholder for the actual proof

end radius_of_circle_with_equal_area_l701_701128


namespace find_line_equation_l701_701658

open Real

section 

variable {x y : ℝ}

def condition_1 (l : ℝ → ℝ → Prop) : Prop :=
  l (-4) 0 ∧ sin (real.atan2 3) = (sqrt 10) / 10

def condition_2 (l : ℝ → ℝ → Prop) : Prop :=
  l (-1) (-3) ∧ real.atan2 (-4) 3 = real.atan2 3 - π / 2

def condition_3 (l : ℝ → ℝ → Prop) : Prop :=
  l (-3) 4 ∧ (∃ a b : ℝ, l = λ x y, x / a + y / b = 1 ∧ a + b = 12)

def equation_1 (x y : ℝ) : Prop := x - 3 * y + 4 = 0 ∨ x + 3 * y + 4 = 0
def equation_2 (x y : ℝ) : Prop := 3 * x + 4 * y + 15 = 0
def equation_3 (x y : ℝ) : Prop := x + 3 * y - 9 = 0 ∨ 4 * x - y + 16 = 0

theorem find_line_equation :
  ∃ l : ℝ → ℝ → Prop, 
    condition_1 l ∧ equation_1 l ∧ 
    condition_2 l ∧ equation_2 l ∧ 
    condition_3 l ∧ equation_3 l :=
by
  sorry

end

end find_line_equation_l701_701658


namespace total_eggs_michael_has_l701_701441

-- Define the initial number of crates
def initial_crates : ℕ := 6

-- Define the number of crates given to Susan
def crates_given_to_susan : ℕ := 2

-- Define the number of crates bought on Thursday
def crates_bought_thursday : ℕ := 5

-- Define the number of eggs per crate
def eggs_per_crate : ℕ := 30

-- Theorem stating the total number of eggs Michael has now
theorem total_eggs_michael_has :
  (initial_crates - crates_given_to_susan + crates_bought_thursday) * eggs_per_crate = 270 :=
sorry

end total_eggs_michael_has_l701_701441


namespace total_shaded_area_l701_701602

def rectangle_area (R : ℝ) : ℝ := R * R
def square_area (S : ℝ) : ℝ := S * S

theorem total_shaded_area 
  (R S : ℝ)
  (h1 : 18 = 2 * R)
  (h2 : R = 4 * S) :
  rectangle_area R + 12 * square_area S = 141.75 := 
  by 
    sorry

end total_shaded_area_l701_701602


namespace rationalize_denominator_of_seven_over_sqrt_343_l701_701456

noncomputable def rationalize_denominator (x : ℝ) : ℝ := x * sqrt 7 / 7

theorem rationalize_denominator_of_seven_over_sqrt_343 :
  (343 = 7^3) → (sqrt 343 = 7 * sqrt 7) →
  (7 / sqrt 343 = sqrt 7 / 7) :=
by
  intros h1 h2
  sorry

end rationalize_denominator_of_seven_over_sqrt_343_l701_701456


namespace distance_between_foci_l701_701091

theorem distance_between_foci
  (x y : ℝ)
  (eq_ellipse : sqrt((x - 3) ^ 2 + (y + 4) ^ 2) + sqrt((x + 5) ^ 2 + (y - 8) ^ 2) = 20) :
  dist (3, -4) (-5, 8) = 4 * sqrt 13 :=
by
  sorry

end distance_between_foci_l701_701091


namespace smallest_period_f_symmetry_f_l701_701711

noncomputable def f (x : ℝ) : ℝ :=
  sin x * cos x - sqrt 3 * cos x ^ 2 + sqrt 3 / 2

theorem smallest_period_f :
  ∀ x : ℝ, f (x + π) = f x :=
sorry

theorem symmetry_f :
  ∃ k : ℤ, ∀ x : ℝ,
  (f (x) = f (π / 6 + k * (π / 2)) ∧ f (x) = 0 →
  ((∃ k : ℤ, x = 5 * π / 12 + k * (π / 2)) ∧ (x, 0) = (π / 6 + k * (π / 2), 0))) :=
sorry

end smallest_period_f_symmetry_f_l701_701711


namespace unique_positive_integer_solution_l701_701239

-- Definitions of the given points
def P1 : ℚ × ℚ := (4, 11)
def P2 : ℚ × ℚ := (16, 1)

-- Definition for the line equation in standard form
def line_equation (x y : ℤ) : Prop := 5 * x + 6 * y = 43

-- Proof for the existence of only one solution with positive integer coordinates
theorem unique_positive_integer_solution :
  ∃ P : ℤ × ℤ, P.1 > 0 ∧ P.2 > 0 ∧ line_equation P.1 P.2 ∧ (∀ Q : ℤ × ℤ, line_equation Q.1 Q.2 → Q.1 > 0 ∧ Q.2 > 0 → Q = (5, 3)) :=
by 
  sorry

end unique_positive_integer_solution_l701_701239


namespace roy_age_product_l701_701824

theorem roy_age_product (R J K : ℕ) 
  (h1 : R = J + 8)
  (h2 : R = K + (R - J) / 2)
  (h3 : R + 2 = 3 * (J + 2)) :
  (R + 2) * (K + 2) = 96 :=
by
  sorry

end roy_age_product_l701_701824


namespace count_odd_tens_95_l701_701009

def has_odd_tens_digit (n : ℕ) : Prop :=
  let d := (n / 10) % 10
  d % 2 = 1

noncomputable def count_odd_tens (upper_limit : ℕ) : ℕ :=
  ((list.range (upper_limit + 1)).filter (λ n, has_odd_tens_digit (n * n))).length

theorem count_odd_tens_95 : count_odd_tens 95 = 19 :=
  sorry

end count_odd_tens_95_l701_701009


namespace reflection_matrix_correct_l701_701032

noncomputable def reflect_matrix_through_plane (n : ℝ × ℝ × ℝ) : matrix (fin 3) (fin 3) ℝ :=
  let ⟨a, b, c⟩ := n;
  let d := a^2 + b^2 + c^2;
  let s := 2/d;
  ![![1 - s*a*a, - s*a*b, - s*a*c],
    ![- s*b*a, 1 - s*b*b, - s*b*c],
    ![- s*c*a, - s*c*b, 1 - s*c*c]]

theorem reflection_matrix_correct :
  reflect_matrix_through_plane (2, -1, 2) =
    (matrix.of ![![1/9, 4/9, -8/9], ![4/9, 7/9, 4/9], ![-8/9, 4/9, 10/9]]) :=
 by sorry

end reflection_matrix_correct_l701_701032


namespace range_of_a_l701_701046

variable {Ω : Type*} [MeasurableSpace Ω]

noncomputable def X : Ω → ℕ := sorry -- Define X as a random variable
axiom X_distrib : ∀ i ∈ {1, 2, 3, 4}, ℙ(X = i) = i / 10

theorem range_of_a (a : ℝ) : 
  (ℙ(1 ≤ X ∧ X < a) = 3 / 5) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l701_701046


namespace slope_of_parallel_line_l701_701990

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l701_701990


namespace find_y_l701_701730

def G (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y (y : ℕ) (h : G 3 y 5 18 = 500) : y = 6 :=
sorry

end find_y_l701_701730


namespace bowling_prize_orders_l701_701371

theorem bowling_prize_orders :
  (let 
    games : list (nat × nat) := [ (6, 5), (4, 5), (3, 4), (2, 3), (1, 2)], 
    possible_outcomes := list.prod (repeat [true, false] 5)
   in 
    possible_outcomes.length = 32) :=
by 
  sorry

end bowling_prize_orders_l701_701371


namespace probability_within_two_units_of_origin_correct_l701_701598

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let square_area := 36
  let circle_area := 4 * Real.pi
  circle_area / square_area

theorem probability_within_two_units_of_origin_correct :
  probability_within_two_units_of_origin = Real.pi / 9 := by
  sorry

end probability_within_two_units_of_origin_correct_l701_701598


namespace slope_of_parallel_line_l701_701949

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l701_701949


namespace exists_bijection_iff_n_plus_one_prime_l701_701578

-- Define the theorem in Lean 4
theorem exists_bijection_iff_n_plus_one_prime (m : ℕ) (hm : Even m ∧ m > 0) :
  ∃ (n : ℕ), (∃ (f : Fin n → Fin n) (hf : Function.Bijective f), 
  ∀ (x y : Fin n), (m * x.val - y.val) % n.val = 0 → (n + 1) ∣ (f x).val ^ m - (f y).val) ↔ Prime (n + 1) :=
begin
  sorry
end

end exists_bijection_iff_n_plus_one_prime_l701_701578


namespace original_population_l701_701209

variable (n : ℝ)

theorem original_population
  (h1 : n + 1500 - 0.15 * (n + 1500) = n - 45) :
  n = 8800 :=
sorry

end original_population_l701_701209


namespace all_three_toppings_zero_l701_701169

namespace PizzaProblem

variables (slices : Finset ℕ)
variables (pepperoni mushrooms olives : set ℕ)
variables [decidable_pred (λ x, x ∈ slices)]
variables {p_m slices_m m_o : set ℕ}

def p_m (s : set ℕ) := {n | n ∈ s ∧ (n ∈ pepperoni ∧ n ∈ mushrooms)}
def p_o (s : set ℕ) := {n | n ∈ s ∧ (n ∈ pepperoni ∧ n ∈ olives)}
def m_o (s : set ℕ) := {n | n ∈ s ∧ (n ∈ mushrooms ∧ n ∈ olives)}
def all_3 (s : set ℕ) := {n | n ∈ s ∧ (n ∈ pepperoni ∧ n ∈ mushrooms ∧ n ∈ olives)}

def PizzaConditions (s : set ℕ) : Prop :=
  | ∀ n, n ∈ s 
  | finset.card slices = 24
  | finset.card pepperoni = 15
  | finset.card mushrooms = 14
  | finset.card olives = 12
  | finset.card (p_m slices) = 6
  | finset.card (m_o slices) = 5
  | finset.card (p_o slices) = 4

theorem all_three_toppings_zero (slices : Finset ℕ) (pepperoni mushrooms olives : set ℕ) :
  PizzaConditions slices pepperoni mushrooms olives → finset.card (all_3 slices) = 0 :=
by
  sorry

end PizzaProblem

end all_three_toppings_zero_l701_701169


namespace jacob_calories_l701_701403

theorem jacob_calories (goal : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) 
  (h_goal : goal = 1800) 
  (h_breakfast : breakfast = 400) 
  (h_lunch : lunch = 900) 
  (h_dinner : dinner = 1100) : 
  (breakfast + lunch + dinner) - goal = 600 :=
by 
  sorry

end jacob_calories_l701_701403


namespace largest_n_unique_k_l701_701917

theorem largest_n_unique_k : ∃ n : ℕ, n = 24 ∧ (∃! k : ℕ, 
  3 / 7 < n / (n + k: ℤ) ∧ n / (n + k: ℤ) < 8 / 19) :=
by
  sorry

end largest_n_unique_k_l701_701917


namespace sum_of_series_l701_701037

noncomputable def series_given_cond (c d : ℝ) :=
  (c / d + c / (d ^ 2) + c / (d ^ 3) + ...) + 
  (1 / d + 1 / (d ^ 2) + 1 / (d ^ 3) + ...) = 6

noncomputable def series_to_find (c d : ℝ) :=
  (c / (c + 2 * d) + c / (c + 2 * d) ^ 2 + c / (c + 2 * d) ^ 3 + ...)

theorem sum_of_series (c d : ℝ) (h : series_given_cond c d) :
  series_to_find c d = (6 * d - 7) / (8 * (d - 1)) :=
sorry

end sum_of_series_l701_701037


namespace sum_of_cubes_equals_square_l701_701807

theorem sum_of_cubes_equals_square :
  1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 := 
by 
  sorry

end sum_of_cubes_equals_square_l701_701807


namespace correct_calculation_only_option_B_l701_701559

theorem correct_calculation_only_option_B :
  (∀ x : ℝ, (sqrt ((-2:ℝ)^2) ≠ -2) ∧ 
            ((-sqrt 2)^2 = 2) ∧ 
            (sqrt (9/4) ≠ ±(3/2)) ∧ 
            (∀ y z : ℝ, sqrt 6 ≠ sqrt (-2) * sqrt (-3))) :=
by sorry

end correct_calculation_only_option_B_l701_701559


namespace incorrect_statement_of_parallelogram_l701_701148

theorem incorrect_statement_of_parallelogram (hA : 27 % 3 = 0 ∨ 27 % 9 = 0)
    (hB : ∀ p q : Type, p ≠ q → (p ∧ q) ↔ False)
    (hC : ∀ p q : Type, p ∨ q)
    (hD1 : is_root (λ x : ℝ, x - 1) 1)
    (hD2 : is_root (λ x : ℝ, x^2 - 5 * x + 4) 1) : 
    (The_diagonals_of_a_parallelogram_perpendicular_bisect_each_other ↔ False) := 
by
  sorry

end incorrect_statement_of_parallelogram_l701_701148


namespace max_projection_area_l701_701534

-- Define the dihedral angle and side length condition
def dihedral_angle := 30 * Real.pi / 180
def side_length := 3

-- Calculate the area of an equilateral triangle with side length 3
def triangle_area (a : ℝ) : ℝ := (Real.sqrt 3 / 4) * a^2

-- Given the specific side length, calculate the area S
def S : ℝ := triangle_area side_length

-- Problem statement: Prove that the maximum projection area is
theorem max_projection_area : ∃ (Pi : ℝ), Pi = S ∧ Pi = 9 * Real.sqrt 3 / 4 :=
by
  use S
  split
  . refl
  . unfold S triangle_area side_length
    norm_num
    rw ← Real.sqrt_div 27 3
    norm_num
    sorry

end max_projection_area_l701_701534


namespace range_of_a_l701_701706

-- Define the condition where z is a complex number with a real and imaginary part.
def z (a : ℝ) : ℂ := 3 + complex.I * a

-- Statement of the problem: Given the conditions, find the range of 'a'.
theorem range_of_a (a : ℝ) (hz : complex.abs (z a) < 4) : a ∈ set.Ioo (-real.sqrt 7) (real.sqrt 7) := 
sorry

end range_of_a_l701_701706


namespace even_sum_sufficient_not_necessary_l701_701679

theorem even_sum_sufficient_not_necessary (m n : ℤ) : 
  (∀ m n : ℤ, (Even m ∧ Even n) → Even (m + n)) 
  ∧ (∀ a b : ℤ, Even (a + b) → ¬ (Odd a ∧ Odd b)) :=
by
  sorry

end even_sum_sufficient_not_necessary_l701_701679


namespace number_of_ways_to_sum_100_l701_701373

noncomputable def pos_perfect_squares_sum (n : ℕ) : Finset (ℕ × ℕ × ℕ) :=
  (Finset.range n).filter (λ x, ∃ a b c, a^2 + b^2 + c^2 = n ∧ 
                                         0 < a ∧ 0 < b ∧ 0 < c ∧ 
                                         (a = 5 ∨ b = 5 ∨ c = 5 ∨ 
                                          a = 6 ∨ b = 6 ∨ c = 6 ∨ 
                                          a = 7 ∨ b = 7 ∨ c = 7 ∨ 
                                          a = 8 ∨ b = 8 ∨ c = 8 ))

theorem number_of_ways_to_sum_100 : 
  (pos_perfect_squares_sum 100).card = 2 := 
sorry

end number_of_ways_to_sum_100_l701_701373


namespace scientific_notation_representation_l701_701389

-- Define the concept of nanoseconds in seconds
def nanosecond_in_seconds : ℝ := 10 ^ (-9)

-- Define the commitment time in nanoseconds
def commitment_time_ns : ℝ := 20

-- The goal is to prove that the commitment time in scientific notation is 2 × 10^(-8) seconds
theorem scientific_notation_representation : (commitment_time_ns * nanosecond_in_seconds) = 2 * 10^(-8) :=
by
  sorry

end scientific_notation_representation_l701_701389


namespace kernels_in_second_bag_l701_701058

theorem kernels_in_second_bag (x : ℕ) (h1 : 60 / 75 * 100 = 80)
                              (h2 : 82 /100 * 100 = 82)
                              (h3 : 82 = (80 + 4200 / x + 82) / 3) : x = 50 := 
by
  -- Definition of percentages for the first and third bags
  have h4 : 80 = 60 / 75 * 100 := by { exact h1, }
  have h5 : 82 = 82 / 100 * 100 := by { exact h2, }
  
  -- Simplifying the given average percentage condition
  have h_avg : 82 = (80 + 4200 / x + 82) / 3 := by { exact h3, }
  
  -- Continue proof (not included here)
  sorry

end kernels_in_second_bag_l701_701058


namespace count_zeros_cos_x2_l701_701727

theorem count_zeros_cos_x2 :
  let f : ℝ → ℝ := λ x, Real.cos (x^2)
  in Finset.card (Finset.filter (λ x, x > -2 * Real.pi ∧ x < 2 * Real.pi) (Finset.image (λ k, Real.sqrt ((Real.pi / 2) + k * Real.pi)) (Finset.range 25))) = 25 :=
by
  sorry

end count_zeros_cos_x2_l701_701727


namespace parallelogram_area_l701_701753

theorem parallelogram_area (ABCD : Type*) [parallelogram ABCD]
  (Ac Bd c : ℝ) (AC_is_c : AC = c) (BD_is_Bd : BD = (√3 / √2) * c) (CAB_is_60 : ∠CAB = 60) :
  area ABCD = (c^2 / 8 * (3 + √3)) := by sorry

end parallelogram_area_l701_701753


namespace isosceles_right_triangle_area_l701_701295

/--
Given an isosceles right triangle with a hypotenuse of 6√2 units, prove that the area
of this triangle is 18 square units.
-/
theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (hyp : h = 6 * Real.sqrt 2) 
  (isosceles : h = l * Real.sqrt 2) : 
  (1/2) * l^2 = 18 :=
by
  sorry

end isosceles_right_triangle_area_l701_701295


namespace johnny_laps_per_minute_l701_701410

theorem johnny_laps_per_minute :
  let total_laps := 10
  let total_minutes := 3.33
  total_laps / total_minutes ≈ 3 :=
sorry

end johnny_laps_per_minute_l701_701410


namespace area_inside_circle_but_outside_square_l701_701585

noncomputable def area_difference : Real :=
  let r := 1
  let s := Real.sqrt 2
  let area_circle := π * (r ^ 2)
  let area_square := s ^ 2
  area_circle - area_square

theorem area_inside_circle_but_outside_square :
  area_difference = π - 2 := by
  sorry

end area_inside_circle_but_outside_square_l701_701585


namespace project_completion_time_l701_701460

theorem project_completion_time (Renu_time Suma_time Arun_time : ℚ)
  (hR: Renu_time = 5)
  (hS: Suma_time = 8)
  (hA: Arun_time = 10) :
  (1 / Renu_time + 1 / Suma_time + 1 / Arun_time)⁻¹ = 40 / 17 :=
by
  rw [hR, hS, hA]
  -- Here we calculate the combined rate and then find its reciprocal
  calc
    (1 / 5 + 1 / 8 + 1 / 10)⁻¹
        = (1 / 5 + 1 / 8 + 1 / 10)⁻¹ : rfl
    ... = (8 / 40 + 5 / 40 + 4 / 40)⁻¹ : by norm_num; rfl
    ... = (17 / 40)⁻¹ : by norm_num
    ... = 40 / 17 : by norm_num

end project_completion_time_l701_701460


namespace arithmetic_sequence_ninth_term_l701_701886

theorem arithmetic_sequence_ninth_term :
  ∃ a d : ℤ, (a + 2 * d = 23) ∧ (a + 5 * d = 29) ∧ (a + 8 * d = 35) :=
by
  sorry

end arithmetic_sequence_ninth_term_l701_701886


namespace percentage_of_40_l701_701663

theorem percentage_of_40 (P : ℝ) (h1 : 8/100 * 24 = 1.92) (h2 : P/100 * 40 + 1.92 = 5.92) : P = 10 :=
sorry

end percentage_of_40_l701_701663


namespace find_c_l701_701860

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 8 = 6) : c = 3 / 2 := 
sorry

end find_c_l701_701860


namespace sum_bn_l701_701309

-- Definitions given in the conditions
def a : ℕ → ℕ := λ n, (if n = 1 then 4 - 2 else (if n = 2 then 4 else 2 * 4))

def S (n : ℕ) : ℕ := 2^(n + 1) - 2

def b (n : ℕ) : ℕ := (2 * n - 1) * S n

-- Now we provide the Lean statement
theorem sum_bn (n : ℕ) :
  (finset.range n).sum b = (2*n - 3) * 2^(n+2) + 12 - 2*n^2 :=
sorry

end sum_bn_l701_701309


namespace remainder_is_602_l701_701041

-- Definition of the problem's conditions
def conditions (a b c : ℕ) : Prop := 
  (a ≤ b ∧ b ≤ c) ∧ (Nat.gcd (Nat.gcd a b) c = 1) ∧ (a * b * c = 6^2020)

-- The main theorem to be proven
theorem remainder_is_602 (N : ℕ) (h : ∃ (a b c : ℕ), conditions a b c ∧ N = ((count_triples a b c) % 1000)) : N = 602 := sorry

end remainder_is_602_l701_701041


namespace sandwiches_count_l701_701839

theorem sandwiches_count :
  let breads := 5 in
  let meats := 7 in
  let cheeses := 6 in
  let total_combinations := breads * meats * cheeses in
  let forbidden_turkey_swiss := breads in
  let forbidden_rye_roastbeef := cheeses in
  total_combinations - forbidden_turkey_swiss - forbidden_rye_roastbeef = 199 :=
by
  sorry

end sandwiches_count_l701_701839


namespace find_fourth_student_number_l701_701364

theorem find_fourth_student_number 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (student1_num : ℕ) 
  (student2_num : ℕ) 
  (student3_num : ℕ) 
  (student4_num : ℕ)
  ( H1 : total_students = 52 )
  ( H2 : sample_size = 4 )
  ( H3 : student1_num = 6 )
  ( H4 : student2_num = 32 )
  ( H5 : student3_num = 45 ) :
  student4_num = 19 :=
sorry

end find_fourth_student_number_l701_701364


namespace arithmetic_sequence_sum_S7_l701_701682

/-- Let {a_n} be a decreasing arithmetic sequence with common difference d and
    sum of first n terms S_n. Given:
    - a_3 = -1
    - a_4 is the geometric mean between a_1 and -a_6

    Prove that S_7 = -14 --/
theorem arithmetic_sequence_sum_S7 :
  ∃ (a₁ d : ℝ),  
    (a₁ + 2 * d = -1) ∧ 
    ((a₁ + 3 * d)^2 = -a₁ * (a₁ + 5 * d)) →  
    7 * a₁ + 21 * d = -14 := 
by
  intros a₁ d h
  have : 7 * a₁ + 21 * d = -14
  sorry

end arithmetic_sequence_sum_S7_l701_701682


namespace calculated_area_of_coverage_valid_n_values_l701_701164

noncomputable def satellite_coverage_area : ℝ :=
let R : ℝ := 6400
let H : ℝ := 500
let cos_alpha := R / (R + H)
let h := R * (1 - cos_alpha)
let S := 2 * Real.pi * R * h
S

theorem calculated_area_of_coverage :
  satellite_coverage_area ≈ 17809000 := by
sorry

noncomputable def feasible_n_values : Set ℕ :=
{n : ℕ | n > 1 ∧ 3 ≤ n ∧ n ≤ 5}

theorem valid_n_values :
  feasible_n_values = {3, 4, 5} := by
sorry

end calculated_area_of_coverage_valid_n_values_l701_701164


namespace rectangle_area_proof_l701_701357

def rectangle_area (L W : ℝ) : ℝ := L * W

theorem rectangle_area_proof (L W : ℝ) (h1 : L + W = 23) (h2 : L^2 + W^2 = 289) : rectangle_area L W = 120 := by
  sorry

end rectangle_area_proof_l701_701357


namespace slope_of_parallel_line_l701_701950

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l701_701950


namespace hotel_total_towels_l701_701595

theorem hotel_total_towels :
  let rooms_A := 25
  let rooms_B := 30
  let rooms_C := 15
  let members_per_room_A := 5
  let members_per_room_B := 6
  let members_per_room_C := 4
  let towels_per_member_A := 3
  let towels_per_member_B := 2
  let towels_per_member_C := 4
  (rooms_A * members_per_room_A * towels_per_member_A) +
  (rooms_B * members_per_room_B * towels_per_member_B) +
  (rooms_C * members_per_room_C * towels_per_member_C) = 975
:= by
  sorry

end hotel_total_towels_l701_701595


namespace water_level_in_cubic_tank_is_one_l701_701589

def cubic_tank : Type := {s : ℝ // s > 0}

def water_volume (s : cubic_tank) : ℝ := 
  let ⟨side, _⟩ := s 
  side^3

def water_level (s : cubic_tank) (volume : ℝ) (fill_ratio : ℝ) : ℝ := 
  let ⟨side, _⟩ := s 
  fill_ratio * side

theorem water_level_in_cubic_tank_is_one
  (s : cubic_tank)
  (h1 : water_volume s = 64)
  (h2 : water_volume s / 4 = 16)
  (h3 : 0 < 0.25 ∧ 0.25 ≤ 1) :
  water_level s 16 0.25 = 1 :=
by 
  sorry

end water_level_in_cubic_tank_is_one_l701_701589


namespace inequality_solution_l701_701296

theorem inequality_solution 
  (a b c d e f : ℕ) 
  (h1 : a * d * f > b * c * f)
  (h2 : c * f * b > d * e * b) 
  (h3 : a * f - b * e = 1) 
  : d ≥ b + f := by
  -- Proof goes here
  sorry

end inequality_solution_l701_701296


namespace insects_per_group_correct_l701_701616

-- Define the numbers of insects collected by boys and girls
def boys_insects : ℕ := 200
def girls_insects : ℕ := 300
def total_insects : ℕ := boys_insects + girls_insects

-- Define the number of groups
def groups : ℕ := 4

-- Define the expected number of insects per group using total insects and groups
def insects_per_group : ℕ := total_insects / groups

-- Prove that each group gets 125 insects
theorem insects_per_group_correct : insects_per_group = 125 :=
by
  -- The proof is omitted (just setting up the theorem statement)
  sorry

end insects_per_group_correct_l701_701616


namespace law_of_sines_l701_701819

theorem law_of_sines (a b c : ℝ) (A B C : ℝ) (R : ℝ) 
  (hA : a = 2 * R * Real.sin A)
  (hEquilateral1 : b = 2 * R * Real.sin B)
  (hEquilateral2 : c = 2 * R * Real.sin C):
  (a / Real.sin A) = (b / Real.sin B) ∧ 
  (b / Real.sin B) = (c / Real.sin C) ∧ 
  (c / Real.sin C) = 2 * R :=
by
  sorry

end law_of_sines_l701_701819


namespace find_x_l701_701744

noncomputable def x : ℝ := 28571.42

theorem find_x (h : (0.0077 * 4.5) / (x * 0.1 * 0.007) ≈ 990) : x ≈ 28571.42 := 
by
  sorry

end find_x_l701_701744


namespace slope_of_parallel_line_l701_701967

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l701_701967


namespace isosceles_trapezoid_rotation_produces_frustum_l701_701461

-- Definitions based purely on conditions
structure IsoscelesTrapezoid :=
(a b c d : ℝ) -- sides
(ha : a = c) -- isosceles property
(hb : b ≠ d) -- non-parallel sides

def rotateAroundSymmetryAxis (shape : IsoscelesTrapezoid) : Type :=
-- We need to define what the rotation of the trapezoid produces
sorry

theorem isosceles_trapezoid_rotation_produces_frustum (shape : IsoscelesTrapezoid) :
  rotateAroundSymmetryAxis shape = Frustum :=
sorry

end isosceles_trapezoid_rotation_produces_frustum_l701_701461


namespace latoya_call_duration_l701_701023

theorem latoya_call_duration
  (initial_credit remaining_credit : ℝ) (cost_per_minute : ℝ) (t : ℝ)
  (h1 : initial_credit = 30)
  (h2 : remaining_credit = 26.48)
  (h3 : cost_per_minute = 0.16)
  (h4 : initial_credit - remaining_credit = t * cost_per_minute) :
  t = 22 := 
sorry

end latoya_call_duration_l701_701023


namespace arithmetic_sequence_ninth_term_l701_701893

theorem arithmetic_sequence_ninth_term 
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29)
  : a + 8 * d = 35 :=
by
  sorry

end arithmetic_sequence_ninth_term_l701_701893


namespace Q1_NE_R1_l701_701291
noncomputable def P : Polynomial ℝ := 3 * X^3 - 5 * X + 4

def product_of_nonzero_coeffs (p : Polynomial ℝ) : ℝ :=
  p.coeff 3 * p.coeff 1 * p.coeff 0

def sum_of_abs_vals_of_coeffs (p : Polynomial ℝ) : ℝ :=
  |p.coeff 3| + |p.coeff 1| + |p.coeff 0|

noncomputable def Q : Polynomial ℝ :=
  (product_of_nonzero_coeffs P) * X^3 + (product_of_nonzero_coeffs P) * X + (product_of_nonzero_coeffs P)

noncomputable def R : Polynomial ℝ :=
  (sum_of_abs_vals_of_coeffs P) * X^3 - (sum_of_abs_vals_of_coeffs P) * X + (sum_of_abs_vals_of_coeffs P)

#eval Q.eval 1 -- Expected: -180
#eval R.eval 1 -- Expected: 12

theorem Q1_NE_R1 : Q.eval 1 ≠ R.eval 1 :=
by
  have hQ : Q.eval 1 = (-60) * 1^3 + (-60) * 1 + (-60) := sorry -- skip the proof
  have hR : R.eval 1 = 12 * 1^3 - 12 * 1 + 12 := sorry -- skip the proof
  rw [hQ, hR]
  simp
  sorry

end Q1_NE_R1_l701_701291


namespace expression_bounds_l701_701416

theorem expression_bounds (p q r s : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) (hr : 0 ≤ r ∧ r ≤ 1) (hs : 0 ≤ s ∧ s ≤ 1) :
  2 * Real.sqrt 2 ≤ (Real.sqrt (p^2 + (1 - q)^2) + Real.sqrt (q^2 + (1 - r)^2) + Real.sqrt (r^2 + (1 - s)^2) + Real.sqrt (s^2 + (1 - p)^2)) ∧
  (Real.sqrt (p^2 + (1 - q)^2) + Real.sqrt (q^2 + (1 - r)^2) + Real.sqrt (r^2 + (1 - s)^2) + Real.sqrt (s^2 + (1 - p)^2)) ≤ 4 :=
by
  sorry

end expression_bounds_l701_701416


namespace inequality_solution_l701_701831

theorem inequality_solution :
  { x : ℝ // -3 * x^2 + 6 * x + 9 > 0 } = Subtype.mk (-1, 3) sorry

end inequality_solution_l701_701831


namespace euler_line_l701_701448

open Classical

noncomputable theory

variables {α : Type*} [EuclideanSpace α] 

-- Definitions for triangle vertices
variables (A B C : α)

-- Definition of midpoint between points
def midpoint (x y : α) : α := (1 / 2) • (x + y)

-- Definitions for the midpoints of the sides of the triangle
def A1 : α := midpoint B C
def B1 : α := midpoint A C
def C1 : α := midpoint A B

-- Definition for the centroid of a triangle
def centroid (A B C : α) : α := 
  (1 / 3) • (A + B + C)

-- The centroid G of ΔABC
def G : α := centroid A B C

-- Definition for the orthocenter
def orthocenter (A B C : α) : α := sorry

-- The orthocenter H of ΔABC
def H : α := orthocenter A B C

-- Definition for the circumcenter
def circumcenter (A B C : α) : α := sorry

-- The circumcenter O of ΔABC
def O : α := circumcenter A B C

-- The Euler line theorem
theorem euler_line (A B C : α) : 
  collinear {G, H, O} :=
sorry

end euler_line_l701_701448


namespace inclination_angle_l701_701482

theorem inclination_angle (a : ℝ) : 
  let m := - (Real.sqrt 3) / 3 in 
  let α := Real.arctan m in
  α * 180 / Real.pi = 150 :=
by 
  sorry

end inclination_angle_l701_701482


namespace smallest_angle_beta_l701_701033

-- Define unit vectors and angles between them
variables {a b c : EuclideanSpace ℝ (Fin 3)}

-- Conditions
axiom unit_a : ∥a∥ = 1
axiom unit_b : ∥b∥ = 1
axiom unit_c : ∥c∥ = 1
axiom angle_ab : ∃ β : ℝ, inner a b = real.cos β
axiom angle_cb_abm : ∃ β : ℝ, inner c (a × b) = real.cos (β / 2)
axiom scalar_triple_prod : inner b (c × a) = 1 / 8

-- Theorem statement
theorem smallest_angle_beta : ∃ β : ℝ, β = real.to_Radians 30 :=
sorry

end smallest_angle_beta_l701_701033


namespace sequence_2005th_term_1730_l701_701862

-- Define the function to calculate the next term in the sequence
def next_term (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d, d * d).sum

-- Define the sequence function
def sequence (n : ℕ) : ℕ → ℕ
| 0     := n
| (k+1) := next_term (sequence k)

-- Define the specific sequence with the first term being 1730
def seq_1730 := sequence 1730

theorem sequence_2005th_term_1730 : seq_1730 2004 = 145 := 
by sorry

end sequence_2005th_term_1730_l701_701862


namespace euler_totient_problem_l701_701027

open Nat

def is_odd (n : ℕ) := n % 2 = 1

def is_power_of_2 (m : ℕ) := ∃ k : ℕ, m = 2^k

theorem euler_totient_problem (n : ℕ) (h1 : n > 0) (h2 : is_odd n) (h3 : is_power_of_2 (φ n)) (h4 : is_power_of_2 (φ (n + 1))) :
  is_power_of_2 (n + 1) ∨ n = 5 := 
sorry

end euler_totient_problem_l701_701027


namespace badgers_win_at_least_three_l701_701077

noncomputable def probability_badgers_win_at_least_three : ℝ :=
  ∑ k in Finset.range (5 + 1), if k ≥ 3 then Nat.choose 5 k * (0.5)^k * (0.5)^(5 - k) else 0

theorem badgers_win_at_least_three :
  probability_badgers_win_at_least_three = 1 / 2 :=
by sorry

end badgers_win_at_least_three_l701_701077


namespace probability_sum_five_l701_701524

def card_numbers : List ℕ := [1, 2, 3, 4]

def all_possible_pairs : List (ℕ × ℕ) :=
  List.bind card_numbers (λ x, List.bind card_numbers (λ y, [(x, y)]))

def favorable_pairs : List (ℕ × ℕ) :=
  List.filter (λ pair, pair.fst + pair.snd = 5) all_possible_pairs

theorem probability_sum_five :
  (favorable_pairs.length : ℚ) / (all_possible_pairs.length : ℚ) = 1 / 4 :=
by
  sorry

end probability_sum_five_l701_701524


namespace parallel_line_slope_l701_701994

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l701_701994


namespace farmer_children_l701_701185

noncomputable def numberOfChildren 
  (totalLeft : ℕ)
  (eachChildCollected : ℕ)
  (eatenApples : ℕ)
  (soldApples : ℕ) : ℕ :=
  let totalApplesEaten := eatenApples * 2
  let initialCollection := eachChildCollected * (totalLeft + totalApplesEaten + soldApples) / eachChildCollected
  initialCollection / eachChildCollected

theorem farmer_children (totalLeft : ℕ) (eachChildCollected : ℕ) (eatenApples : ℕ) (soldApples : ℕ) : 
  totalLeft = 60 → eachChildCollected = 15 → eatenApples = 4 → soldApples = 7 → 
  numberOfChildren totalLeft eachChildCollected eatenApples soldApples = 5 := 
by
  intro h_totalLeft h_eachChildCollected h_eatenApples h_soldApples
  unfold numberOfChildren
  simp
  sorry

end farmer_children_l701_701185


namespace broccoli_price_l701_701408

noncomputable def price_per_pound_broccoli (price_broccoli_per_pound: ℝ) (price_oranges_each: ℝ) (price_cabbage: ℝ) (price_bacon: ℝ) (price_chicken_per_pound: ℝ) (budget_percentage_on_meat: ℝ) : Prop :=
  let oranges_count := 3 in
  let broccoli_pounds := 3 in
  let meat_pounds_chicken := 2 in
  let meat_total := price_bacon + meat_pounds_chicken * price_chicken_per_pound in
  let total_excluding_broccoli := oranges_count * price_oranges_each + price_cabbage + price_bacon + meat_pounds_chicken * price_chicken_per_pound in
  let total_budget := meat_total / budget_percentage_on_meat in
  let cost_broccoli := total_budget - total_excluding_broccoli in
  price_broccoli_per_pound = (cost_broccoli / broccoli_pounds)

theorem broccoli_price :
  price_per_pound_broccoli 4.01 0.75 3.75 3 3 0.33 :=
  sorry

end broccoli_price_l701_701408


namespace area_of_ABC_l701_701413

theorem area_of_ABC (ABC G: Triangle) 
  (h_G : is_centroid G ABC)
  (h_BC : side_length ABC B C = 3)
  (h_sim : similar_tri ABC GAB) :
  area ABC = 3 * sqrt 2 / 2 :=
by
  sorry

end area_of_ABC_l701_701413


namespace slope_of_parallel_line_l701_701951

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l701_701951


namespace order_scores_l701_701772

theorem order_scores
  (J K M Q S : ℕ)
  (h1 : J ≥ Q) (h2 : J ≥ M) (h3 : J ≥ S) (h4 : J ≥ K)
  (h5 : M > Q ∨ M > S ∨ M > K)
  (h6 : K < S) (h7 : S < J) :
  K < S ∧ S < M ∧ M < Q :=
by
  sorry

end order_scores_l701_701772


namespace fourth_polygon_is_square_l701_701170

theorem fourth_polygon_is_square
  (angle_triangle angle_square angle_hexagon : ℕ)
  (h_triangle : angle_triangle = 60)
  (h_square : angle_square = 90)
  (h_hexagon : angle_hexagon = 120)
  (h_total : angle_triangle + angle_square + angle_hexagon = 270) :
  ∃ angle_fourth : ℕ, angle_fourth = 90 ∧ (angle_fourth + angle_triangle + angle_square + angle_hexagon = 360) :=
sorry

end fourth_polygon_is_square_l701_701170


namespace cricket_initial_average_l701_701472

theorem cricket_initial_average (A : ℕ) (h1 : ∀ A, A * 20 + 137 = 21 * (A + 5)) : A = 32 := by
  -- assumption and proof placeholder
  sorry

end cricket_initial_average_l701_701472


namespace simplify_fraction_l701_701069

variable {a : ℝ}
variable (h1 : a ≠ 1) (h2 : a ≠ -1) (h3 : a ≠ 2) (h4 : a ≠ -2)

theorem simplify_fraction :
  ( (a + 1) / (a^2 - 1) / ((a^2 - 4) / (a^2 + a - 2)) - (1 - a) / (a - 2) ) = a / (a - 2) :=
by
  have h5 : a^2 - 1 = (a + 1) * (a - 1) := sorry -- Factorization of a^2 - 1
  have h6 : a^2 - 4 = (a + 2) * (a - 2) := sorry -- Factorization of a^2 - 4
  have h7 : a^2 + a - 2 = (a + 2) * (a - 1) := sorry -- Factorization of a^2 + a - 2
  calc
    (a+1) / (a^2-1) / ((a^2-4) / (a^2+a-2)) - (1-a) / (a-2)
        = (a+1) / ((a+1)*(a-1)) * ((a^2+a-2) / (a^2-4)) - (1-a) / (a-2) : by sorry -- Rewrite division as multiplication and factorize
    ... = (a+1) / ((a+1)*(a-1)) * ((a+2)*(a-1) / (a+2)*(a-2)) - (1-a) / (a-2) : by sorry -- Substitute factorizations
    ... = (1) / (a-2) + (a-1) / (a-2) : by sorry -- Simplify fractions 
    ... = (a / (a - 2)) : by sorry -- Combine fractions and simplify

end simplify_fraction_l701_701069


namespace find_alpha_l701_701759

variable {α : ℝ}

def parametric_eq_line (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, 1 + t * Real.sin α)

def cartesian_eq_curve_C : ℝ × ℝ → Prop :=
  λ p, p.1 ^ 2 = 4 * p.2

def line_intersects_C (t : ℝ) (α : ℝ) : Prop :=
  let pt := parametric_eq_line t α in
  cartesian_eq_curve_C pt

theorem find_alpha (h_intersecs : ∀ t1 t2, t1 ≠ t2 → line_intersects_C t1 α ∧ line_intersects_C t2 α)
  (h_AB : ∃ t1 t2, |t1 - t2| = 8 ∧ t1 ≠ t2) :
  α = (π / 4) ∨ α = (3 * π / 4) :=
sorry

end find_alpha_l701_701759


namespace pipe_a_fills_pool_in_12_hours_l701_701814

-- Definitions based on the conditions
def rate_pipe_a (t : ℝ) : ℝ := 1 / t
def rate_pipe_b (t : ℝ) : ℝ := 1 / (3 * t)
def filled_by_a (t : ℝ) : ℝ := 8 * rate_pipe_a t
def remaining_to_fill (t : ℝ) : ℝ := 1 - filled_by_a t
def time_to_fill_by_b (t : ℝ) : ℝ := 12 * rate_pipe_b t

-- The statement to prove
theorem pipe_a_fills_pool_in_12_hours : ∃ t : ℝ, 
  filled_by_a t + time_to_fill_by_b t = 1 ∧ 1 / t = rate_pipe_a t :=
sorry

end pipe_a_fills_pool_in_12_hours_l701_701814


namespace length_of_KL_l701_701486

theorem length_of_KL (A B C D E P Q R S K L : Type) 
  (hA: A = (1 : ℝ)) 
  (hB: B = (1 : ℝ)) 
  (hC: C = (1 : ℝ)) 
  (hD: D = (1 : ℝ)) 
  (hE: E = (1 : ℝ)) 
  (hP: P = (1 / 2 : ℝ)) 
  (hQ: Q = (1 / 2 : ℝ)) 
  (hR: R = (1 / 2 : ℝ)) 
  (hS: S = (1 / 2 : ℝ)) 
  (hK: K = (1 / 2 : ℝ)) 
  (hL: L = (1 / 2 : ℝ)) : 
  KL = (1 / 4 : ℝ) := 
sorry

end length_of_KL_l701_701486


namespace power_addition_proof_l701_701628

theorem power_addition_proof :
  (-2) ^ 48 + 3 ^ (4 ^ 3 + 5 ^ 2 - 7 ^ 2) = 2 ^ 48 + 3 ^ 40 := 
by
  sorry

end power_addition_proof_l701_701628


namespace evaluate_expr_l701_701649

theorem evaluate_expr : ⌊3.998⌋ + ⌈7.002⌉ = 11 := 
by
  have h1 : ⌊3.998⌋ = 3 := sorry -- Floor function calculation
  have h2 : ⌈7.002⌉ = 8 := sorry -- Ceiling function calculation
  show 3 + 8 = 11 from sorry -- Summation

end evaluate_expr_l701_701649


namespace part1_part2_l701_701821

-- Part (1)
theorem part1 (x y : ℚ) 
  (h1 : 2022 * x + 2020 * y = 2021)
  (h2 : 2023 * x + 2021 * y = 2022) :
  x = 1/2 ∧ y = 1/2 :=
by
  -- Placeholder for the proof
  sorry

-- Part (2)
theorem part2 (x y a b : ℚ)
  (ha : a ≠ b) 
  (h1 : (a + 1) * x + (a - 1) * y = a)
  (h2 : (b + 1) * x + (b - 1) * y = b) :
  x = 1/2 ∧ y = 1/2 :=
by
  -- Placeholder for the proof
  sorry

end part1_part2_l701_701821


namespace cone_lateral_surface_area_l701_701181

theorem cone_lateral_surface_area (r : ℝ) (theta : ℝ) (h_r : r = 3) (h_theta : theta = 90) : 
  let base_circumference := 2 * Real.pi * r
  let R := 12
  let lateral_surface_area := (1 / 2) * base_circumference * R 
  lateral_surface_area = 36 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l701_701181


namespace L_shaped_figure_perimeter_is_14_l701_701074

-- Define the side length of each square as a constant
def side_length : ℕ := 2

-- Define the horizontal base length
def base_length : ℕ := 3 * side_length

-- Define the height of the vertical stack
def vertical_stack_height : ℕ := 2 * side_length

-- Define the total perimeter of the "L" shaped figure
def L_shaped_figure_perimeter : ℕ :=
  base_length + side_length + vertical_stack_height + side_length + side_length + vertical_stack_height

-- The theorem that states the perimeter of the L-shaped figure is 14 units
theorem L_shaped_figure_perimeter_is_14 : L_shaped_figure_perimeter = 14 := sorry

end L_shaped_figure_perimeter_is_14_l701_701074


namespace pentagon_to_triangle_l701_701623

noncomputable def area_of_triangle_ACE : Prop :=
  let area_of_pentagon := 2014
  let ratio_CD_ED := ⅗
  let area_of_ACE := 1325
  ∃ (a: ℝ) (s: ℝ), 
  area_of_pentagon = 13 * a^2 + 6 * a^2 ∧
  s = sqrt(13) * a ∧ 
  area_of_ACE = ½ * 5 * a * 5 * a

theorem pentagon_to_triangle (area_of_pentagon: ℝ) (ratio_CD_ED: ℝ)
  (area_of_ACE: ℝ) : 
  area_of_pentagon = 2014 → 
  ratio_CD_ED = ⅗ → 
  area_of_ACE = 1325 → 
  area_of_triangle_ACE :=
by
  intros
  simp [area_of_pentagon, ratio_CD_ED, area_of_ACE]
  sorry

#eval pentagon_to_triangle 2014 ⅗ 1325 -- Expected output: true

end pentagon_to_triangle_l701_701623


namespace find_x_l701_701522

theorem find_x (α : Real) 
  (h1 : sin α = 1 / 3) 
  (M : ℝ → ℝ → Prop)
  (h2 : M (x : ℝ) (1 : ℝ)) : 
  x = 2 * real.sqrt 2 ∨ x = -2 * real.sqrt 2 := 
sorry

end find_x_l701_701522


namespace average_speed_return_trip_l701_701212

/--
A train travels from Albany to Syracuse, a distance of 120 miles,
at an average rate of 50 miles per hour. The train then continues
to Rochester, which is 90 miles from Syracuse, before returning
to Albany. On its way to Rochester, the train's average speed is
60 miles per hour. Finally, the train travels back to Albany from
Rochester, with the total travel time of the train, including all
three legs of the journey, being 9 hours and 15 minutes. What was
the average rate of speed of the train on the return trip from
Rochester to Albany?
-/
theorem average_speed_return_trip :
  let dist_Albany_Syracuse := 120 -- miles
  let speed_Albany_Syracuse := 50 -- miles per hour
  let dist_Syracuse_Rochester := 90 -- miles
  let speed_Syracuse_Rochester := 60 -- miles per hour
  let total_travel_time := 9.25 -- hours (9 hours 15 minutes)
  let time_Albany_Syracuse := dist_Albany_Syracuse / speed_Albany_Syracuse
  let time_Syracuse_Rochester := dist_Syracuse_Rochester / speed_Syracuse_Rochester
  let total_time_so_far := time_Albany_Syracuse + time_Syracuse_Rochester
  let time_return_trip := total_travel_time - total_time_so_far
  let dist_return_trip := dist_Albany_Syracuse + dist_Syracuse_Rochester
  let average_speed_return := dist_return_trip / time_return_trip
  average_speed_return = 39.25 :=
by
  -- sorry placeholder for the actual proof
  sorry

end average_speed_return_trip_l701_701212


namespace triangle_area_l701_701000

noncomputable theory

variables {A B C : Type*} [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C]

-- Given the two altitudes and the ratio in the third altitude in an acute-angled triangle.
def altitude1 : A := 3
def altitude2 : B := 2 * real.sqrt 2
def ratio : C := 5 / 1

-- Prove that the area of the acute-angled triangle is 6.
theorem triangle_area (h_alt1 : A = 3) (h_alt2 : B = 2 * real.sqrt 2) (h_ratio : C = 5 / 1) : real :=
  let area : real := 6
  area

end triangle_area_l701_701000


namespace slope_of_parallel_line_l701_701985

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l701_701985


namespace donation_addition_median_mode_l701_701272

def initial_donations : list ℕ := [5, 3, 6, 5, 10]

def sorted_initial_donations : list ℕ := initial_donations.qsort (≤)

-- Proof that a = 1 or a = 2 maintains the median and mode.
def median_donation (l : list ℕ) : ℕ :=
l[(l.length / 2 : ℕ)] -- Assuming list is sorted

def mode_donation (l : list ℕ) : list ℕ :=
l.foldl (λ acc x, if list.count l x > list.count l (list.head! acc) then [x] else if list.count l x = list.count l (list.head! acc) && x ≠ list.head! acc then x :: acc else acc) [list.head! l]

theorem donation_addition_median_mode (a : ℕ) :
  (a = 1 ∨ a = 2) ↔ 
  median_donation ([4, 5, 5, 6, 10].qsort (≤)) = median_donation sorted_initial_donations ∧ 
  mode_donation ([4, 5, 5, 6, 10].qsort (≤)) = mode_donation sorted_initial_donations ∨ 
  median_donation ([5, 5, 5, 6, 10].qsort (≤)) = median_donation sorted_initial_donations ∧ 
  mode_donation ([5, 5, 5, 6, 10].qsort (≤)) = mode_donation sorted_initial_donations ∧ 
  median_donation updated_donations = 5 ∧ mode_donation updated_donations = [5] :=
  sorry

end donation_addition_median_mode_l701_701272


namespace centers_form_rectangle_l701_701229

open Real

noncomputable def C1 := circle (origin) 1
noncomputable def C2 := circle (origin) 1
noncomputable def C := circle (origin) 2

constant center_O : pt
constant center_A : pt
constant center_X : pt
constant center_Y : pt

-- Define distances based on geometric properties
axiom dist_O_to_A : dist center_O center_A = 1
axiom dist_O_to_X : dist center_O center_X = 4 / 3
axiom dist_A_to_X : dist center_A center_X = 1 + 2 / 3

axiom dist_O_to_Y : dist center_O center_Y = 5 / 3
axiom dist_X_to_Y : dist center_X center_Y = 1 + 1 / 3

-- Theorem to prove centers form a rectangle
theorem centers_form_rectangle :
  ¬ collinear center_O center_A center_X center_Y ∧
  dist center_O center_X == dist center_A center_Y ∧ 
  dist center_O center_Y == dist center_A center_X :=
by sorry

end centers_form_rectangle_l701_701229


namespace integral_solution_infinite_l701_701817

/-- Prove that the equation x^3 + y^4 = z^31 has infinitely many integral solutions. -/
theorem integral_solution_infinite : ∃ f : ℤ → ℤ × ℤ × ℤ, (∀ t : ℤ, let (x, y, z) := f t in x^3 + y^4 = z^31) ∧ (function.injective f) :=
sorry

end integral_solution_infinite_l701_701817


namespace analysis_error_l701_701307

theorem analysis_error (x : ℝ) (h1 : x + 1 / x ≥ 2) : 
  x + 1 / x ≥ 2 :=
by {
  sorry
}

end analysis_error_l701_701307


namespace count_digits_first_1500_even_l701_701134

theorem count_digits_first_1500_even :
  (∑ i in (finset.range 1500).map (λ n, 2*(n+1)), nat.digits 10 i).sum = 5448 :=
sorry

end count_digits_first_1500_even_l701_701134


namespace total_cost_of_dishes_l701_701233

theorem total_cost_of_dishes
  (e t b : ℝ)
  (h1 : 4 * e + 5 * t + 2 * b = 8.20)
  (h2 : 6 * e + 3 * t + 4 * b = 9.40) :
  5 * e + 6 * t + 3 * b = 12.20 := 
sorry

end total_cost_of_dishes_l701_701233


namespace jon_monthly_earnings_l701_701778

def earnings_per_person : ℝ := 0.10
def visits_per_hour : ℕ := 50
def hours_per_day : ℕ := 24
def days_per_month : ℕ := 30

theorem jon_monthly_earnings : 
  (earnings_per_person * visits_per_hour * hours_per_day * days_per_month) = 3600 :=
by
  sorry

end jon_monthly_earnings_l701_701778


namespace proof_f_five_l701_701708

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 4 * x + 3 else 3 - x

theorem proof_f_five : f (f 5) = -1 :=
by sorry

end proof_f_five_l701_701708


namespace students_in_johnsons_class_l701_701803

-- Define the conditions as constants/variables
def studentsInFinleysClass : ℕ := 24
def studentsAdditionalInJohnsonsClass : ℕ := 10

-- State the problem as a theorem
theorem students_in_johnsons_class : 
  let halfFinleysClass := studentsInFinleysClass / 2
  let johnsonsClass := halfFinleysClass + studentsAdditionalInJohnsonsClass
  johnsonsClass = 22 :=
by
  sorry

end students_in_johnsons_class_l701_701803


namespace large_jars_count_l701_701896

theorem large_jars_count (S L : ℕ) (h1 : S + L = 100) (h2 : S = 62) (h3 : 3 * S + 5 * L = 376) : L = 38 :=
by
  sorry

end large_jars_count_l701_701896


namespace num_4_digit_numbers_l701_701722

-- Definitions based on conditions:
def digits := [1, 0, 3, 3]

def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def has_exact_digits (n : ℕ) : Prop :=
  let digits_of_n := List.map (fun c => c.to_nat! - '0'.to_nat!) (n.repr.to_list)
  List.perm digits digits_of_n

-- The main statement to prove:
theorem num_4_digit_numbers : 
  ∃! n, n = 9 ∧ (∃ (k : ℕ), has_exact_digits k ∧ is_4_digit k) :=
begin
  sorry
end

end num_4_digit_numbers_l701_701722


namespace initial_bookmarks_before_march_l701_701247

/-- 
Siena bookmarks 30 pages every day in March. She will have 1330 pages 
in her bookmarks library at the end of March. Prove that Siena had 
400 pages in her bookmarks library before she started bookmarking 
pages in March.
-/
theorem initial_bookmarks_before_march (d : ℕ) (p : ℕ) (t : ℕ) (a : ℕ) :
  d = 31 →
  p = 30 →
  t = 1330 →
  a = p * d →
  t - a = 400 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end initial_bookmarks_before_march_l701_701247


namespace total_digits_first_1500_even_integers_l701_701137

/-- Prove that the total number of digits used to write the first 1500 positive even integers is 5448. -/
theorem total_digits_first_1500_even_integers : 
  let num_one_digit_even := 4 in
  let num_two_digit_even := 45 in
  let num_three_digit_even := 450 in
  let num_four_digit_even := 1001 in
  let digits_one_digit := num_one_digit_even * 1 in
  let digits_two_digit := num_two_digit_even * 2 in
  let digits_three_digit := num_three_digit_even * 3 in
  let digits_four_digit := num_four_digit_even * 4 in
  digits_one_digit + digits_two_digit + digits_three_digit + digits_four_digit = 5448 := by
  sorry

end total_digits_first_1500_even_integers_l701_701137


namespace WX_eq_YZ_l701_701615

variables {Point : Type} [euclidean_geometry Point]

-- Definitions of the parallelogram and points
variables {A B C D E F X Y W Z : Point}
variables (parallelogram_ABCD : parallelogram A B C D)
variables (excircle_center_E : excircle_center E A B C)
variables (excircle_center_F : excircle_center F A D C)
variables (X_touch_point : touches_line E A B X)
variables (Y_touch_point : touches_line F A D Y)
variables (W_intersection : intersects_line F C A B W)
variables (Z_intersection : intersects_line E C A D Z)

-- Problem statement
theorem WX_eq_YZ :
  WX = YZ := 
sorry

end WX_eq_YZ_l701_701615


namespace y_exceeds_x_by_100_percent_l701_701572

theorem y_exceeds_x_by_100_percent (x y : ℝ) (h : x = 0.5 * y) : (y - x) / x = 1 := by
sorry

end y_exceeds_x_by_100_percent_l701_701572


namespace range_of_a_l701_701045

variable {Ω : Type*} [MeasurableSpace Ω]

noncomputable def X : Ω → ℕ := sorry -- Define X as a random variable
axiom X_distrib : ∀ i ∈ {1, 2, 3, 4}, ℙ(X = i) = i / 10

theorem range_of_a (a : ℝ) : 
  (ℙ(1 ≤ X ∧ X < a) = 3 / 5) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l701_701045


namespace find_f_sum_l701_701865

def f (x : ℝ) : ℝ := sorry

axiom symmetry : ∀ x : ℝ, f(x) = -f(-x - 3/2)
axiom functional_eq : ∀ x : ℝ, f(x) = -f(x + 3/2)
axiom f_neg1 : f(-1) = 1
axiom f_0 : f(0) = -2

theorem find_f_sum : (finset.range 2008).sum (λ n, f (1 + n : ℕ)) = 1 :=
sorry

end find_f_sum_l701_701865


namespace complex_quadrant_l701_701729

theorem complex_quadrant (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : 1 + a * i = (b + i) * (1 + i)) : 
  (a - b * i).re > 0 ∧ (a - b * i).im < 0 :=
by
  have h1 : 1 + a * i = (b - 1) + (b + 1) * i := by sorry
  have h2 : a = b + 1 := by sorry
  have h3 : b - 1 = 1 := by sorry
  have h4 : b = 2 := by sorry
  have h5 : a = 3 := by sorry
  have h6 : (a - b * i).re = 3 := by sorry
  have h7 : (a - b * i).im = -2 := by sorry
  exact ⟨by linarith, by linarith⟩

end complex_quadrant_l701_701729


namespace granola_bars_relation_l701_701055

theorem granola_bars_relation (x y z : ℕ) (h1 : z = x / (3 * y)) : z = x / (3 * y) :=
by {
    sorry
}

end granola_bars_relation_l701_701055


namespace sector_area_l701_701696

theorem sector_area (theta : ℝ) (r : ℝ) (h_theta : theta = 2 * π / 3) (h_r : r = 3) : 
    (theta / (2 * π) * π * r^2) = 3 * π :=
by 
  -- Placeholder for the actual proof
  sorry

end sector_area_l701_701696


namespace find_m_b_l701_701145

-- Definitions of the given polynomials and conditions
def f (x : ℝ) : ℝ := x^4 - 4 * x^3 + 13 * x^2 - 14 * x + 4
def g (x m : ℝ) : ℝ := x^2 - 3 * x + m
def remainder (m b x : ℝ) := (3 * m - 11) * x + (m^2 - 13 * m + 4)

-- Problem statement as a Lean theorem
theorem find_m_b : ∃ m b : ℝ, 3 * m - 11 = 2 ∧ b = m^2 - 13 * m + 4 ∧ (f x) % (g x m) = (2 * x + b) :=
begin
  sorry -- Proof steps go here
end

end find_m_b_l701_701145


namespace lucy_flour_purchase_l701_701435

theorem lucy_flour_purchase (initial : ℕ) (used : ℕ) (half_spilled_ratio : ℕ) (final : ℕ) : 
  initial = 500 → used = 240 → half_spilled_ratio = 2 → final = 500 → 
  let remaining := initial - used in 
  let spilled := remaining / half_spilled_ratio in 
  final - spilled = 370 := 
begin
  intros h_initial h_used h_half_spilled_ratio h_final,
  rw h_initial,
  rw h_used,
  rw h_half_spilled_ratio,
  rw h_final,
  simp,
  -- Proof should go here, but it is skipped as per instruction
  sorry,
end

end lucy_flour_purchase_l701_701435


namespace smallest_n_for_inequality_l701_701312

theorem smallest_n_for_inequality :
  ∃ (n : ℕ), n = 4003 ∧ (∀ m : ℤ, (0 < m ∧ m < 2001) →
  ∃ k : ℤ, (m / 2001 : ℚ) < (k / n : ℚ) ∧ (k / n : ℚ) < ((m + 1) / 2002 : ℚ)) :=
sorry

end smallest_n_for_inequality_l701_701312


namespace simplify_complex_fraction_l701_701829

-- Define the numerator as a complex number
def numerator : ℂ := 3 - 5 * complex.I

-- Define the denominator as a complex number
def denominator : ℂ := 2 - 7 * complex.I

-- Define the expected simplified result
def expected_result : ℂ := - (41 : ℝ) / 45 - (11 : ℝ) / 45 * complex.I

-- State the theorem that represents the given mathematical problem
theorem simplify_complex_fraction : (numerator / denominator) = expected_result :=
by 
  sorry

end simplify_complex_fraction_l701_701829


namespace line_never_passes_through_fixed_points_l701_701508

-- Definitions for the fixed points
def point1 := (-2, 3)
def point2 := (2, 3)

-- Condition: the line equation in terms of variable a
def line_eq (a : ℝ) (x : ℝ) (y : ℝ) : Prop := (a - 1) * x - y + 2 * a + 1 = 0

-- Statement we need to prove
theorem line_never_passes_through_fixed_points : 
  ∀ a : ℝ, ¬ (∀ (x y : ℝ), (x, y) = point1 → line_eq a x y)
    ∧ ¬ (∀ (x y : ℝ), (x, y) = point2 → line_eq a x y) := 
by 
  sorry

end line_never_passes_through_fixed_points_l701_701508


namespace relationship_between_a_b_c_l701_701700

noncomputable def f : ℝ → ℝ := sorry

variables (x : ℝ) (x₁ x₂ : ℝ)
variable (h_even : ∀ x, f(x + 1) = f(1 - x))
variable (h_increasing : ∀ x₁ x₂, 1 ≤ x₁ ∧ x₁ < x₂ → (f(x₂) - f(x₁)) / (x₂ - x₁) > 0)

def a : ℝ := f(-1/2)
def b : ℝ := f(1)
def c : ℝ := f(2)

theorem relationship_between_a_b_c (h : True) : b < c ∧ c < a :=
by
  sorry

end relationship_between_a_b_c_l701_701700


namespace delivery_driver_net_rate_of_pay_l701_701183

-- Conditions
def travel_time : ℕ := 3 -- hours
def speed : ℕ := 50 -- miles per hour
def fuel_efficiency : ℕ := 25 -- miles per gallon
def earnings_per_mile : ℕ := 0.60 -- dollars per mile (assuming Lean recognizes dollars as real numbers)
def gas_price_per_gallon : ℕ := 2.50 -- dollars per gallon

-- Proof Statement
theorem delivery_driver_net_rate_of_pay :
  let distance := speed * travel_time in
  let gasoline_used := distance / fuel_efficiency in
  let total_earnings := earnings_per_mile * distance in
  let gas_cost := gas_price_per_gallon * gasoline_used in
  let net_earnings := total_earnings - gas_cost in
  let net_rate_of_pay := net_earnings / travel_time in
  net_rate_of_pay = 25 := 
by 
  sorry

end delivery_driver_net_rate_of_pay_l701_701183


namespace arithmetic_sequence_max_min_b_l701_701234

-- Define the sequence a_n
def S (n : ℕ) : ℚ := (1/2) * n^2 - 2 * n
def a (n : ℕ) : ℚ := S n - S (n - 1)

-- Question 1: Prove that {a_n} is an arithmetic sequence with a common difference of 1
theorem arithmetic_sequence (n : ℕ) (hn : n ≥ 2) : 
  a n - a (n - 1) = 1 :=
sorry

-- Define the sequence b_n
def b (n : ℕ) : ℚ := (a n + 1) / a n

-- Question 2: Prove that b_3 is the maximum value and b_2 is the minimum value in {b_n}
theorem max_min_b (hn2 : 2 ≥ 1) (hn3 : 3 ≥ 1) : 
  b 3 = 3 ∧ b 2 = -1 :=
sorry

end arithmetic_sequence_max_min_b_l701_701234


namespace simplest_quadratic_radicals_l701_701745

theorem simplest_quadratic_radicals (a : ℝ) :
  (3 * a - 8 ≥ 0) ∧ (17 - 2 * a ≥ 0) → a = 5 :=
by
  intro h
  sorry

end simplest_quadratic_radicals_l701_701745


namespace dyslexian_alphabet_size_l701_701837

theorem dyslexian_alphabet_size (c v : ℕ) (h1 : (c * v * c * v * c + v * c * v * c * v) = 4800) : c + v = 12 :=
by
  sorry

end dyslexian_alphabet_size_l701_701837


namespace scientific_notation_of_20_nanoseconds_l701_701391

def nanosecond_to_seconds (nanoseconds : ℕ) : ℝ :=
  nanoseconds * (1 * 10 ^ (-9))

theorem scientific_notation_of_20_nanoseconds :
  nanosecond_to_seconds 20 = 2 * 10 ^ (-8) := by
  sorry

end scientific_notation_of_20_nanoseconds_l701_701391


namespace pentagon_area_l701_701030

theorem pentagon_area :
  ∀ (P : ℕ → ℝ × ℝ)
    (Q : ℕ → ℝ × ℝ)
    (apothem : ℝ),
    (∀ i : ℕ, i < 5 → dist P i P ((i+1) % 5) = side_length) →
    apothem = 3 →
    (∀ i : ℕ, i < 5 → Q i = (1/3) • P i + (2/3) • P ((i+1) % 5)) →
    area_of_pentagon Q = 14.5 :=
begin
  intros P Q apothem hPentagon side_cond apothem_cond Q_cond,
  sorry
end

end pentagon_area_l701_701030


namespace circle_passing_through_points_l701_701845

noncomputable def circle_equation (D E F : ℝ) : ℝ × ℝ → ℝ :=
λ p, p.1^2 + p.2^2 + D * p.1 + E * p.2 + F

theorem circle_passing_through_points : ∃ D E F : ℝ, 
  circle_equation D E F (0, 0) = 0 ∧
  circle_equation D E F (4, 0) = 0 ∧
  circle_equation D E F (-1, 1) = 0 ∧
  D = -4 ∧ 
  E = -6 ∧ 
  F = 0 :=
begin
  sorry
end

end circle_passing_through_points_l701_701845


namespace mass_in_scientific_notation_l701_701871

def mass_oxygen := 0.00000000000000000000000002657
def significant_figures := 2.657
def exponent := -26

theorem mass_in_scientific_notation : mass_oxygen = significant_figures * 10^exponent := 
by 
  -- Place the proof here
  sorry

end mass_in_scientific_notation_l701_701871


namespace rationalize_denominator_div_l701_701453

theorem rationalize_denominator_div (h : 343 = 7 ^ 3) : 7 / Real.sqrt 343 = Real.sqrt 7 / 7 := 
by 
  sorry

end rationalize_denominator_div_l701_701453


namespace monotonic_intervals_extreme_point_max_g_x_l701_701431

noncomputable def f (x a b : ℝ) : ℝ := x^3 - a * x - b

theorem monotonic_intervals (a b : ℝ) :
  if a > 0 then
    ∃ c : ℝ, c = sqrt(3 * a) / 3 ∧
    (∀ x : ℝ, x < -c → deriv (λ x, f x a b) x > 0) ∧
    (∀ x : ℝ, -c < x ∧ x < c → deriv (λ x, f x a b) x < 0) ∧
    (∀ x : ℝ, x > c → deriv (λ x, f x a b) x > 0)
  else
    ∀ x : ℝ, deriv (λ x, f x a b) x ≥ 0 :=
sorry

theorem extreme_point (a b x₀ x₁ : ℝ) (h₀ : deriv (λ x, f x a b) x₀ = 0) (h₁ : f x₁ a b = f x₀ a b) (h₂ : x₁ ≠ x₀) :
  x₁ + 2 * x₀ = 0 :=
sorry

theorem max_g_x (a b : ℝ) (h : a > 0) :
  ∃ M : ℝ, M = max (|f (-1) a b|) (|f 1 a b|) ∧ M ≥ 1 / 4 :=
sorry

end monotonic_intervals_extreme_point_max_g_x_l701_701431


namespace isosceles_triangles_in_polygon_l701_701414

theorem isosceles_triangles_in_polygon (n : ℕ) (h : n ≥ 2) (P : Polygon) (hreg : P.regular (2 * n + 1)) (hdiv : divides_triangles P (2 * n - 1) ) : ∃ (T1 T2 T3 : Triangle), T1.isosceles ∧ T2.isosceles ∧ T3.isosceles :=
sorry

end isosceles_triangles_in_polygon_l701_701414


namespace rationalize_denominator_l701_701459

theorem rationalize_denominator (a b : ℝ) (h : b = 343) (h_nonzero : b ≠ 0) : (a = 7) → (\sqrt b = 7 * \sqrt 7) → \frac{a}{\sqrt b} = \frac{\sqrt 7}{7} :=
by
  sorry

end rationalize_denominator_l701_701459


namespace rationalize_denominator_div_l701_701451

theorem rationalize_denominator_div (h : 343 = 7 ^ 3) : 7 / Real.sqrt 343 = Real.sqrt 7 / 7 := 
by 
  sorry

end rationalize_denominator_div_l701_701451


namespace sum_first_nine_terms_of_seq_l701_701293

noncomputable def seq (a : ℕ → ℕ) := ∀ n : ℕ, n > 0 → a (n + 1)^2 / a n = 4 * (a (n + 1) - a n)

noncomputable def a_1 : ℕ := 2

noncomputable def S_9 := ∑ i in range 9, seq a (i + 1)

theorem sum_first_nine_terms_of_seq : 
  (a : ℕ → ℕ) : (∀ n : ℕ, n > 0 → a (n + 1)^2 / a n = 4 * (a (n + 1) - a n)) → 
  a 1 = 2 → S_9 a = 1022 := 
by sorry

end sum_first_nine_terms_of_seq_l701_701293


namespace area_of_ABCD_l701_701383

theorem area_of_ABCD (area_AMOP area_CNOQ : ℝ) 
  (h1: area_AMOP = 8) (h2: area_CNOQ = 24.5) : 
  ∃ (area_ABCD : ℝ), area_ABCD = 60.5 :=
by
  sorry

end area_of_ABCD_l701_701383


namespace jeans_original_price_l701_701207

theorem jeans_original_price 
  (discount : ℝ -> ℝ)
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (final_price : ℝ) 
  (customer_payment : ℝ) : 
  discount_percentage = 0.10 -> 
  discount x = x * (1 - discount_percentage) -> 
  final_price = discount (2 * original_price) + original_price -> 
  customer_payment = 112 -> 
  final_price = 112 -> 
  original_price = 40 := 
by
  intros
  sorry

end jeans_original_price_l701_701207


namespace largest_A_telephone_number_l701_701609

variable {A B C D E F G H I J : ℕ}

theorem largest_A_telephone_number :
  -- Conditions:
  (A > B) ∧ (B > C) ∧ (D > E) ∧ (E > F) ∧ (G > H) ∧ (H > I) ∧ (I > J) ∧
  (∀ x, x ∈ {D, E, F} → x ∈ {0, 1, 4, 9}) ∧
  (∀ x, x ∈ {G, H, I, J} → x ∈ {0, 1, 3, 6}) ∧
  (A + B + C = 10) →
  -- Proof:
  (A = 5) :=
begin
  sorry
end

end largest_A_telephone_number_l701_701609


namespace slope_parallel_to_original_line_l701_701947

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l701_701947


namespace minimum_value_of_f_l701_701873

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 ^ (Real.sqrt (x^2 - 4*x + 4)) + Real.sqrt (x^2 - 2*x)

-- Define the domain conditions
def domain_condition_1 (x : ℝ) : Prop := x^2 - 4*x + 4 ≥ 0
def domain_condition_2 (x : ℝ) : Prop := x^2 - 2*x ≥ 0
def domain (x : ℝ) : Prop := (x ≤ 0 ∨ x ≥ 2)

-- State the theorem
theorem minimum_value_of_f : ∃ (x : ℝ), domain x ∧ domain_condition_1 x ∧ domain_condition_2 x ∧ f(x) = 1 :=
sorry

end minimum_value_of_f_l701_701873


namespace center_of_gravity_shift_center_of_gravity_shift_result_l701_701213

variable (l s : ℝ) (s_val : s = 60)
#check (s_val : s = 60)

theorem center_of_gravity_shift : abs ((l / 2) - ((l - s) / 2)) = s / 2 := 
by sorry

theorem center_of_gravity_shift_result : (s / 2 = 30) :=
by sorry

end center_of_gravity_shift_center_of_gravity_shift_result_l701_701213


namespace function_decreasing_l701_701705

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if 0 < x ∧ x ≤ 1 then -x^2 + 2*(a+1)*x + 4
     else if x > 1 then x^a
     else 0

theorem function_decreasing (a : ℝ) :
  (∀ x y : ℝ, 0 < x ∧ y > x → f a y ≤ f a x) ↔ -2 ≤ a ∧ a ≤ -1 := 
by {
  sorry
}

end function_decreasing_l701_701705


namespace max_points_l701_701010

variable (M : ℕ)
variable (D : ℕ) (V : ℕ)
variable (opponents_points : ℕ) (team_points_diff : ℕ)

theorem max_points :
  D = 3 →
  V = 2 * (M + D) →
  opponents_points = 40 →
  team_points_diff = 16 →
  M + D + V = opponents_points - team_points_diff →
  M = 5 :=
by
  intros,
  sorry

end max_points_l701_701010


namespace sum_of_ages_today_l701_701822

variable (RizaWas25WhenSonBorn : ℕ) (SonCurrentAge : ℕ) (SumOfAgesToday : ℕ)

theorem sum_of_ages_today (h1 : RizaWas25WhenSonBorn = 25) (h2 : SonCurrentAge = 40) : SumOfAgesToday = 105 :=
by
  sorry

end sum_of_ages_today_l701_701822


namespace average_yield_per_tree_l701_701752

theorem average_yield_per_tree :
  let t1 := 3
  let t2 := 2
  let t3 := 1
  let nuts1 := 60
  let nuts2 := 120
  let nuts3 := 180
  let total_nuts := t1 * nuts1 + t2 * nuts2 + t3 * nuts3
  let total_trees := t1 + t2 + t3
  let average_yield := total_nuts / total_trees
  average_yield = 100 := 
by
  sorry

end average_yield_per_tree_l701_701752


namespace part1_part2_l701_701653

-- Definition of completeness in residues
def complete_residue_system (a : ℕ → ℕ) (n : ℕ) :=
  ∀ (r : ℕ), r < n → ∃ (i : ℕ), i < n ∧ a i ≡ r [MOD n]

-- Part 1 Condition and Conclusion
theorem part1 (n : ℕ) (h1 : complete_residue_system id n) (h2 : complete_residue_system (λ i, i + i) n) : n % 2 = 1 :=
sorry

-- Part 2 Condition and Conclusion
theorem part2 (n : ℕ) (h1 : (Nat.gcd n 6) = 1) (h2 : complete_residue_system id n) (h3 : complete_residue_system (λ i, i + i) n) (h4 : complete_residue_system (λ i, i - i) n) : (Nat.gcd n 6) = 1 :=
sorry

end part1_part2_l701_701653


namespace symmetric_matrix_diagonal_l701_701161

theorem symmetric_matrix_diagonal (n : ℕ) (A : Matrix (Fin n) (Fin n) ℕ)
  (hn : Odd n)
  (h1 : ∀ i j, 1 ≤ A i j ∧ A i j ≤ n)
  (h2 : ∀ i, (Finset.univ.image (λ j, A i j)).card = n)
  (h3 : ∀ j, (Finset.univ.image (λ i, A i j)).card = n)
  (hsymm : ∀ i j, A i j = A j i) :
  ∀ k, 1 ≤ k ∧ k ≤ n → ∃ i, A i i = k := sorry

end symmetric_matrix_diagonal_l701_701161


namespace find_a_l701_701654

def polynomial (a : ℝ) := λ x : ℝ, x^3 - 6 * x^2 + a * x + a

def root_condition (x1 x2 x3 : ℝ) := (x1 - 3)^2 + (x2 - 3)^3 + (x3 - 3)^3 = 0

theorem find_a (a : ℝ) (x1 x2 x3 : ℝ) :
  (polynomial a).is_root x1 → (polynomial a).is_root x2 → (polynomial a).is_root x3 →
  root_condition x1 x2 x3 → a = -9 :=
sorry

end find_a_l701_701654


namespace no_rational_roots_l701_701028

theorem no_rational_roots (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = p) (h2 : Prime p) (h3: Nat.digits 10 p = [a, b, c, d]) : 
  ¬ ∃ x : ℚ, a * x^3 + b * x^2 + c * x + d = 0 :=
by
  sorry

end no_rational_roots_l701_701028


namespace different_answers_due_to_different_cuts_l701_701248

noncomputable def problem_89914 (bub : Type) (cut : bub → (bub × bub)) (is_log_cut : bub → Prop) (is_halved_log : bub × bub → Prop) : Prop :=
  ∀ b : bub, (is_log_cut b) → is_halved_log (cut b)

noncomputable def problem_89915 (bub : Type) (cut : bub → (bub × bub)) (is_sector_cut : bub → Prop) (is_sectors : bub × bub → Prop) : Prop :=
  ∀ b : bub, (is_sector_cut b) → is_sectors (cut b)

theorem different_answers_due_to_different_cuts
  (bub : Type)
  (cut : bub → (bub × bub))
  (is_log_cut : bub → Prop)
  (is_halved_log : bub × bub → Prop)
  (is_sector_cut : bub → Prop)
  (is_sectors : bub × bub → Prop) :
  problem_89914 bub cut is_log_cut is_halved_log ∧ problem_89915 bub cut is_sector_cut is_sectors →
  ∃ b : bub, (is_log_cut b ∧ ¬ is_sector_cut b) ∨ (¬ is_log_cut b ∧ is_sector_cut b) := sorry

end different_answers_due_to_different_cuts_l701_701248


namespace simplify_and_evaluate_expression_l701_701468

variables (m n : ℚ)

theorem simplify_and_evaluate_expression (h1 : m = -1) (h2 : n = 1 / 2) :
  ( (2 / (m - n) - 1 / (m + n)) / ((m * n + 3 * n ^ 2) / (m ^ 3 - m * n ^ 2)) ) = -2 :=
by
  sorry

end simplify_and_evaluate_expression_l701_701468


namespace problem_l701_701344

theorem problem (x a : ℝ) (h : x^5 - x^3 + x = a) : x^6 ≥ 2 * a - 1 := 
by 
  sorry

end problem_l701_701344


namespace ratio_of_additional_class_l701_701462

-- Definitions based on provided conditions
def cost_per_pack : ℝ := 75
def classes_per_pack : ℕ := 10
def total_classes : ℕ := 13
def total_cost : ℝ := 105

-- Declaration of the ratio proof
theorem ratio_of_additional_class :
  let avg_cost_per_class := cost_per_pack / classes_per_pack
  let additional_classes := total_classes - classes_per_pack
  let cost_of_additional_classes := total_cost - cost_per_pack
  let cost_per_additional_class := cost_of_additional_classes / additional_classes
  (cost_per_additional_class / avg_cost_per_class).to_rat = (4 / 3).to_rat :=
by
  sorry

end ratio_of_additional_class_l701_701462


namespace percent_of_grade_C_l701_701866

def scores_in_mr_freemans_class : List ℕ := [89, 72, 54, 97, 77, 92, 85, 74, 75, 63, 84, 78, 71, 80, 90]

def is_grade_C (score : ℕ) : Prop :=
  score >= 75 ∧ score <= 84

theorem percent_of_grade_C :
  (↑(scores_in_mr_freemans_class.filter is_grade_C).length / scores_in_mr_freemans_class.length) * 100 = 33 + 1/3 :=
by
  sorry

end percent_of_grade_C_l701_701866


namespace rectangle_perimeter_l701_701533

-- Define the sides of the triangle
def a : ℕ := 5
def b : ℕ := 12
def c : ℕ := 13

-- Define the width of the rectangle
def width : ℕ := 5

-- Define the area of the triangle
def triangle_area : ℕ := (1 / 2 : ℝ) * a * b

-- Define the length of the rectangle based on the area
def length : ℝ := (triangle_area / width : ℝ)

-- Prove that the perimeter of the rectangle is 22 units
theorem rectangle_perimeter : 2 * (length + width) = 22 := by
  sorry

end rectangle_perimeter_l701_701533


namespace total_infections_second_wave_l701_701874

theorem total_infections_second_wave (cases_per_day_first_wave : ℕ)
                                     (factor_increase : ℕ)
                                     (duration_weeks : ℕ)
                                     (days_per_week : ℕ) :
                                     cases_per_day_first_wave = 300 →
                                     factor_increase = 4 →
                                     duration_weeks = 2 →
                                     days_per_week = 7 →
                                     (duration_weeks * days_per_week) * (cases_per_day_first_wave + factor_increase * cases_per_day_first_wave) = 21000 :=
by 
  intros h1 h2 h3 h4
  sorry

end total_infections_second_wave_l701_701874


namespace abby_bridget_adjacent_probability_l701_701216

noncomputable theory

-- Define the seating grid
def grid := matrix (fin 3) (fin 3) (option string)

-- Define the students present
def students : list string := ["Abby", "Bridget", "Student1", "Student2", "Student3", "Student4", "Student5", "Student6"]

-- Calculate the total number of distinct seating arrangements
def total_arrangements : ℕ := nat.factorial 9 / nat.factorial 1

-- Calculate the probability of Abby and Bridget being adjacent
def probability_adjacent : ℚ := (9 + 6) * 2 * nat.factorial 7 / nat.factorial 9

-- Prove that the probability of Abby and Bridget being adjacent is 5/12
theorem abby_bridget_adjacent_probability :
  probability_adjacent = 5 / 12 :=
sorry

end abby_bridget_adjacent_probability_l701_701216


namespace Jan_additional_distance_proof_l701_701149

-- Define variables and conditions
variables (d t s : ℕ) -- Ian's distance, time, and speed
variables (d_Han m_Jan : ℕ) -- Han and Jan's distances respectively

-- Han's and Jan's conditions
def Han_condition := d_Han = (s + 5) * (t + 2)
def Jan_condition := m_Jan = (s + 15) * (t + 3)

-- Additional information
axiom Han_distance : d_Han = d + 110
axiom distance_equation1 : d = s * t

-- Goal
theorem Jan_additional_distance_proof (h_Han : Han_condition) (h_Jan : Jan_condition) (h_Han_miles : Han_distance) (eq1: distance_equation1) : 
  ∃ n : ℕ, n = m_Jan - d ∧ n = 195 :=
by
  sorry

end Jan_additional_distance_proof_l701_701149


namespace choose_five_pairwise_coprime_from_six_l701_701447

open Nat

theorem choose_five_pairwise_coprime_from_six (a1 a2 a3 a4 a5 a6 : ℕ) 
  (h1 : a1 >= 1000) (h2 : a1 <= 9999)
  (h3 : a2 >= 1000) (h4 : a2 <= 9999)
  (h5 : a3 >= 1000) (h6 : a3 <= 9999)
  (h7 : a4 >= 1000) (h8 : a4 <= 9999)
  (h9 : a5 >= 1000) (h10 : a5 <= 9999)
  (h11 : a6 >= 1000) (h12 : a6 <= 9999)
  (coprime1 : gcd a1 a2 = 1) (coprime2 : gcd a1 a3 = 1) (coprime3 : gcd a1 a4 = 1)
  (coprime4 : gcd a1 a5 = 1) (coprime5 : gcd a1 a6 = 1) (coprime6 : gcd a2 a3 = 1)
  (coprime7 : gcd a2 a4 = 1) (coprime8 : gcd a2 a5 = 1) (coprime9 : gcd a2 a6 = 1)
  (coprime10 : gcd a3 a4 = 1) (coprime11 : gcd a3 a5 = 1) (coprime12 : gcd a3 a6 = 1)
  (coprime13 : gcd a4 a5 = 1) (coprime14 : gcd a4 a6 = 1) (coprime15 : gcd a5 a6 = 1)
  : ∃ b1 b2 b3 b4 b5, 
    (b1 >= 1000) ∧ (b1 <= 9999) ∧
    (b2 >= 1000) ∧ (b2 <= 9999) ∧
    (b3 >= 1000) ∧ (b3 <= 9999) ∧
    (b4 >= 1000) ∧ (b4 <= 9999) ∧
    (b5 >= 1000) ∧ (b5 <= 9999) ∧ 
    ({b1, b2, b3, b4, b5} ⊆ {a1, a2, a3, a4, a5, a6}) ∧ 
    (gcd b1 b2 = 1) ∧ (gcd b1 b3 = 1) ∧ (gcd b1 b4 = 1) ∧ (gcd b1 b5 = 1) ∧
    (gcd b2 b3 = 1) ∧ (gcd b2 b4 = 1) ∧ (gcd b2 b5 = 1) ∧
    (gcd b3 b4 = 1) ∧ (gcd b3 b5 = 1) ∧
    (gcd b4 b5 = 1) := sorry

end choose_five_pairwise_coprime_from_six_l701_701447


namespace inequality_proof_equality_condition_l701_701446

theorem inequality_proof (a : ℝ) : (a^2 + 5)^2 + 4 * a * (10 - a) ≥ 8 * a^3  :=
by sorry

theorem equality_condition (a : ℝ) : ((a^2 + 5)^2 + 4 * a * (10 - a) = 8 * a^3) ↔ (a = 5 ∨ a = -1) :=
by sorry

end inequality_proof_equality_condition_l701_701446


namespace stone_counted_as_68_is_10_l701_701652

-- Definitions based on the conditions
def stones : List ℕ := List.range 1 (15 + 1)

def count_with_cycles (n : ℕ) : ℕ :=
  let cycle_length := 29
  let effective_pos := n % cycle_length
  if effective_pos = 0 then
    15
  else if effective_pos ≤ 15 then
    effective_pos
  else
    15 - (effective_pos - 15)

-- Lean statement to prove
theorem stone_counted_as_68_is_10 :
  count_with_cycles 68 = 10 := by
  sorry

end stone_counted_as_68_is_10_l701_701652


namespace find_side_a_l701_701767

noncomputable def side_a (b : ℝ) (A : ℝ) (S : ℝ) : ℝ :=
  2 * S / (b * Real.sin A)

theorem find_side_a :
  let b := 2
  let A := Real.pi * 2 / 3 -- 120 degrees in radians
  let S := 2 * Real.sqrt 3
  side_a b A S = 4 :=
by
  let b := 2
  let A := Real.pi * 2 / 3
  let S := 2 * Real.sqrt 3
  show side_a b A S = 4
  sorry

end find_side_a_l701_701767


namespace probability_sum_le_4_of_two_dice_tossed_l701_701122

open ProbabilityTheory

def diceSumLE4 : MeasureTheory.ProbabilityMeasure (Set (Fin 6 × Fin 6)) :=
  sorry

theorem probability_sum_le_4_of_two_dice_tossed :
  diceSumLE4 { p : Fin 6 × Fin 6 | p.1 + p.2 + 2 ≤ 4 } = 1 / 6 :=
sorry

end probability_sum_le_4_of_two_dice_tossed_l701_701122


namespace total_shipping_cost_l701_701205

-- Define the total cost function
def totalCost (W : ℝ) : ℝ :=
  2 + 5 * ⌈W⌉

-- State the theorem
theorem total_shipping_cost (W : ℝ) : totalCost W = 2 + 5 * ⌈W⌉ :=
  sorry

end total_shipping_cost_l701_701205


namespace ellipse_eccentricity_l701_701315

theorem ellipse_eccentricity (a b : ℝ) (ha : a > b) (hb : b > 0)
  (M : ℝ × ℝ == (2, 1)) (slope_angle : real.angle == real.angle.two_div_pi (3 / 4)) 
  (ellipse_eq : ∀ (x y : ℝ), (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1^2 / a^2 + p.2^2 / b^2 = 1))) :
  let c := real.sqrt (a^2 - b^2) in
  let e := c / a in
  e = real.sqrt 2 / 2 :=
sorry

end ellipse_eccentricity_l701_701315


namespace error_percent_in_area_l701_701156

theorem error_percent_in_area 
    (L W : ℝ) 
    (measured_length : ℝ := 1.09 * L) 
    (measured_width : ℝ := 0.92 * W) 
    (correct_area : ℝ := L * W) 
    (incorrect_area : ℝ := measured_length * measured_width) :
    100 * (incorrect_area - correct_area) / correct_area = 0.28 :=
by
  sorry

end error_percent_in_area_l701_701156


namespace sequence_sum_100_l701_701668

noncomputable def sequence_sum : ℕ → ℤ
| 0 => 1
| 1 => -2
| n + 2 => sequence_sum (n + 1) - sequence_sum n

theorem sequence_sum_100 :
  (∑ i in Finset.range 100, sequence_sum i) = -60 :=
by
  sorry

end sequence_sum_100_l701_701668


namespace divisor_of_1076_plus_least_addend_l701_701549

theorem divisor_of_1076_plus_least_addend (a d : ℕ) (h1 : 1076 + a = 1081) (h2 : d ∣ 1081) (ha : a = 5) : d = 13 := 
sorry

end divisor_of_1076_plus_least_addend_l701_701549


namespace max_value_of_n_l701_701688

-- Define the functions f and g
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := x^2 - x + 2

-- Maximum value of n problem statement
theorem max_value_of_n :
  ∃ (n : ℕ), (∀ (x : ℝ), x ∈ set.Icc 0 (9/2)) →
  (∑ i in (finset.range (n - 1)).map nat.cast, f x + g (∑ i in (finset.range (n - 1)).map nat.cast f x)) =
  (∑ i in (finset.range (n - 1)).map nat.cast, g x + f (∑ i in (finset.range (n - 1)).map nat.cast g x)) →
  n = 14 :=
begin
  sorry
end

end max_value_of_n_l701_701688


namespace Q_is_constant_l701_701423

-- Define the problem context and assumptions
variables (P Q : ℕ → ℕ)
variables [∀ n : ℕ, 0 < n → 0 < P n]  -- P(n) is strictly positive
variables [∀ n : ℕ, 0 < n → 0 < Q n]  -- Q(n) is strictly positive
variable  (h_divisibility : ∀ n : ℕ, 0 < n → (2 ^ Q n - 1) ∣ (3 ^ P n - 1))  -- Divisibility condition

-- Define the gcd condition: any polynomial with rational coefficients 
-- that divides both P and Q is constant
variable  (h_gcd : ∀ R : ℕ → ℕ, (∀ n : ℕ, 0 < R n → R n ∣ P n ∧ R n ∣ Q n) → is_constant R)

-- Prove that Q is constant under the given conditions
theorem Q_is_constant : (∀ n m : ℕ, Q n = Q m) :=
sorry

end Q_is_constant_l701_701423


namespace slope_of_parallel_line_l701_701992

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l701_701992


namespace slope_of_parallel_line_l701_701971

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l701_701971


namespace time_to_empty_tank_l701_701903

-- Define the rates for each pump in terms of the fraction of the tank they fill per hour
def ratePumpA := 1/6 : ℚ
def ratePumpB := 1/8 : ℚ
def ratePumpC := 1/12 : ℚ

-- Combined rate of the pumps
def combinedRatePumps := ratePumpA + ratePumpB + ratePumpC

-- Effective rate due to filling a tank in 15 hours
def effectiveRate := 1/15 : ℚ

-- Definition of the combined leak rate L
def leakRate := combinedRatePumps - effectiveRate

-- Time required to empty the full tank
theorem time_to_empty_tank : 1 / leakRate = 120/37 := by
  sorry

end time_to_empty_tank_l701_701903


namespace largest_k_divides_P_l701_701570

theorem largest_k_divides_P :
  ∃ k : ℕ, k = 1009 ∧
  ∃ P : ℤ, 
    (∀ grid : ℕ × ℕ, ((grid = (2, 2018)) → 
      (∀ block : ℕ × ℕ, (block = (2, 2)) → 
        ∃ white_block : ℕ , (1 ≤ white_block ∧ white_block ≤ 4))) → 
      3^k ∣ P) :=
begin
  sorry
end

end largest_k_divides_P_l701_701570


namespace sum_of_coefficients_l701_701244

theorem sum_of_coefficients (x : ℝ) (h : x ≠ 0) : 
  let expression := (2 - 1/x)^7 in
  (is_expanded_sum_of_coeffs expression = 1) :=
by
  have expression := (2 - 1/x)^7
  sorry

end sum_of_coefficients_l701_701244


namespace plane_coloring_l701_701039

-- Statement of the problem in the Lean 4 theorem
theorem plane_coloring (n : ℕ) (h : n ≥ 1) :
  ∃ (col : ℝ × ℝ → bool), 
  ∀ (p q : ℝ × ℝ), 
  (∃ (circle_eq : ℝ × ℝ → Prop), 
    circle_eq p ↔ circle_eq q ∧ dist p q = 0) → col p ≠ col q :=
by
  sorry

end plane_coloring_l701_701039


namespace correct_calculation_D_l701_701146

theorem correct_calculation_D (m : ℕ) : 
  (2 * m ^ 3) * (3 * m ^ 2) = 6 * m ^ 5 :=
by
  sorry

end correct_calculation_D_l701_701146


namespace axis_of_symmetry_translated_graph_l701_701476

/--
The determinant operation is defined as follows:

\[
  \begin{vmatrix}
    a_1 & a_2 \\
    a_3 & a_4
  \end{vmatrix}
  = a_1 a_4 - a_2 a_3.
\]

The given function is

\[
  f(x) = \begin{vmatrix}
    \sin 2x & \sqrt{3} \\
    \cos 2x & 1
  \end{vmatrix}
\]

The function \( f(x) \) is translated to the right by \( \frac{\pi}{6} \) units.

Prove that \( x = \frac{7\pi}{12} \) is one of the axes of symmetry of the resulting graph.
-/
theorem axis_of_symmetry_translated_graph :
    (∃ f : ℝ → ℝ, f = λ x, 2 * (Real.sin (2 * x - Real.pi / 3)) →
     ∃ g : ℝ → ℝ, g = λ x, 2 * (Real.sin (2 * (x - Real.pi / 6) - Real.pi / 3)) /\
     (λ x, g (x + Real.pi / 6)) ∈ λ x, 2 * (Real.sin (2 * x - 2 * Real.pi / 3)) →
     ∃ k : ℤ, x = (k * Real.pi) / 2 + 7 * Real.pi / 12) :=
begin
    sorry
end

end axis_of_symmetry_translated_graph_l701_701476


namespace probability_even_sum_l701_701360

namespace ProbabilityPrimeSum

def first_eight_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

def is_even (n : ℕ) : Prop := n % 2 = 0

def count_pairs {α : Type*} (s : Finset α) (p : α → α → Prop) : ℕ :=
  (s.product s).filter (λ t => ∧ t.1 ≠ t.2 ∧ p t.1 t.2).card

theorem probability_even_sum : 
  (count_pairs first_eight_primes (λ x y => is_even (x + y))).val / (first_eight_primes.card.choose 2) = 3 / 4 := by 
  sorry

end ProbabilityPrimeSum

end probability_even_sum_l701_701360


namespace real_solutions_eq_l701_701089

/-- Define the function f with the given condition on non-zero real numbers -/
def f : ℝ → ℝ := sorry

/-- Given condition for non-zero real numbers x -/
axiom f_func (x : ℝ) (h : x ≠ 0) : f(x) + 2 * f(1 / x) = 5 * x

/-- Function that returns the same value for f(x) and f(-x) -/
theorem real_solutions_eq (x : ℝ) (h : f(x) = f(-x)) : x = real.sqrt 2 ∨ x = - real.sqrt 2 :=
sorry

end real_solutions_eq_l701_701089


namespace polar_equations_and_area_of_triangle_l701_701008

theorem polar_equations_and_area_of_triangle :
  let C1 := { (ρ θ : ℝ) | ρ = 2 * Real.sin θ }
  let C2 := { (ρ θ : ℝ) | ρ = 2 * Real.cos θ }
  (A B M : ℝ × ℝ) 
  (θ1 θ2 θ3 : ℝ)
  (hC1 : A = (Real.sqrt 3, θ1) ∧ θ1 = π / 3) 
  (hC2 : B = (1, θ2) ∧ θ2 = π / 3)
  (hM : M = (4, 0))
  (hθ : θ3 = π / 3)
  (A_in_C1 : C1 (Real.sqrt 3) θ1)
  (B_in_C2 : C2 1 θ2)
  (area_triangle : ℝ) :=
  area_triangle = 3 - Real.sqrt 3 := by
  sorry

end polar_equations_and_area_of_triangle_l701_701008


namespace circle_touches_inscribed_iff_radius_eq_excircle_l701_701065

theorem circle_touches_inscribed_iff_radius_eq_excircle (ABC : Triangle) 
  (I I_c: Point) (r r_c : ℝ) (C1 C2 C0 : Point) (a b c : ℝ) (p : ℝ)
  (h1 : I = incenter ABC) 
  (h2 : I_c = excenter_C ABC) 
  (h3 : r = radius_incircle ABC) 
  (h4 : r_c = radius_excircle_C ABC) 
  (h5 : C1 = incircle_tangent_point_AB ABC) 
  (h6 : C2 = excircle_C_tangent_point_AB ABC) 
  (h7 : C0 = midpoint (vertex_A ABC) (vertex_B ABC))
  (h8 : a = side_length_BC ABC) 
  (h9 : b = side_length_CA ABC)
  (h10 : c = side_length_AB ABC)
  (h11 : p = semiperimeter ABC) :
  (construct_circle_on_AB_diameter_touches_incircle ABC) ↔ (side_length_AB ABC = radius_excircle_C ABC) := sorry

end circle_touches_inscribed_iff_radius_eq_excircle_l701_701065


namespace semicircle_radius_right_triangle_l701_701764

-- Define the points and characteristics of the triangle
variables {P Q R : Type} [OrderedField P] [OrderedField Q] [OrderedField R]

-- Define the properties of the right triangle
def right_triangle (PQ QR PR : P) : Prop :=
PQ * PQ + QR * QR = PR * PR ∧ PQ = 15 ∧ QR = 8

-- Define the function to calculate the radius of the inscribed semicircle
noncomputable def inscribed_semicircle_radius {PQ QR PR : P}
  (h : right_triangle PQ QR PR) : P :=
15 * QR / (PQ + PQ + QR) -- This corresponds to the radius of the semicircle

-- The theorem stating the radius calculation result
theorem semicircle_radius_right_triangle :
  ∀ {PQ QR PR : P}, right_triangle PQ QR PR → inscribed_semicircle_radius = 24 / 5 :=
by
  intros PQ QR PR h,
  sorry

end semicircle_radius_right_triangle_l701_701764


namespace concyclic_D_H_E_F_l701_701380

open EuclideanGeometry

variable {A B C D E F H O P : Point}

-- Given: In an acute triangle ABC, O is the circumcenter, AH is the altitude.
axiom acute_triangle (h₁ : acute_triangle ABC) : True
axiom circumcenter_O (h₂ : circumcenter O A B C) : True
axiom altitude_AH (h₃ : altitude AH A B C) : True

-- Given: P is a point on AO.
axiom point_on_AO (h₄ : point_on_line P A O) : True

-- Given: PD, PE, PF are the angle bisectors of ∠BPC, ∠CPA, ∠APB, respectively.
axiom angle_bisectors (h₅ : angle_bisector P D B C) (h₆ : angle_bisector P E C A) (h₇ : angle_bisector P F A B) : True

-- Prove: D, H, E, and F are concyclic.
theorem concyclic_D_H_E_F : concyclic D H E F :=
sorry

end concyclic_D_H_E_F_l701_701380


namespace find_a_plus_c_l701_701794

open Function

noncomputable def parabolas_intersect (a b c d : ℝ) : Prop :=
  (-(3 - a)^2 + b = 6) ∧
  ((3 - c)^2 + d = 6) ∧
  (-(9 - a)^2 + b = 0) ∧
  ((9 - c)^2 + d = 0)

theorem find_a_plus_c (a b c d : ℝ) (h : parabolas_intersect a b c d) : a + c = 12 :=
by {
  sorry
}

end find_a_plus_c_l701_701794


namespace hyperbola_eccentricity_l701_701177

theorem hyperbola_eccentricity (a b : ℝ) (h0 : a > 0) (h1 : b > 0) :
  (∃ c : ℝ, (∀ M : ℝ × ℝ, (M.1 = c ∧ M.2 = (b^2 / a) ∧ (M.1^2 / a^2 - M.2^2 / b^2 = 1 ∧ c = sqrt(3) * b)) ∧
              ∀ P Q : ℝ × ℝ, (P.2 = -Q.2 ∧ P.1 = Q.1 = 0 ∧ (2 * sqrt(b^4 / a^2 - c^2) = sqrt(3) * (b^4 / a^2 - c^2))) ∧
              (△ (M, P, Q) is equilateral) ∧ (∃ e : ℝ, e = c / a ∧ e = sqrt(3)))) :=
sorry

end hyperbola_eccentricity_l701_701177


namespace evaluate_f_of_sin_pi_over_12_l701_701732

theorem evaluate_f_of_sin_pi_over_12 (f : ℝ → ℝ) (h : ∀ x, f (cos x) = cos (2 * x)) : 
  f (sin (Real.pi / 12)) = - (Real.sqrt 3) / 2 :=
by
  sorry

end evaluate_f_of_sin_pi_over_12_l701_701732


namespace slope_of_parallel_line_l701_701929

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l701_701929


namespace yellow_block_correct_weight_l701_701411

-- Conditions
variables (y : ℝ) (green_block_weight : ℝ) (diff : ℝ)
-- Definitions used in the conditions
def green_block_weight_value : green_block_weight = 0.4 := by sorry
def weight_difference : diff = 0.2 := by sorry
def yellow_block_weight : y = green_block_weight + diff := by sorry

-- Theorem to prove the weight of the yellow block
theorem yellow_block_correct_weight : y = 0.6 :=
begin
  sorry
end

end yellow_block_correct_weight_l701_701411


namespace midpoint_distance_l701_701749

variables (a b c d : ℝ)

def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem midpoint_distance :
  let M  := midpoint a b c d,
      A' := (a + 3, b + 5),
      B' := (c - 6, d - 3),
      M' := midpoint (A'.1) (A'.2) (B'.1) (B'.2)
  in distance M M' = Real.sqrt 13 / 2 :=
sorry

end midpoint_distance_l701_701749


namespace length_of_KL_l701_701497

open Finset
open Classical

-- Define the pentagon and relevant points
def Pentagon :=
  {a b c d e : Point}

-- Conditions: all sides of the pentagon are equal to 1
axiom side_lengths (p : Pentagon) : length (p.a, p.b) = 1 ∧ length (p.b, p.c) = 1 ∧ length (p.c, p.d) = 1 ∧ length (p.d, p.e) = 1 ∧ length (p.e, p.a) = 1

-- Define midpoints
def midpoint (p1 p2 : Point) : Point
axiom midpoint_is_between (p1 p2 : Point) : midpoint p1 p2 = (p1 + p2) / 2

-- Define points P, Q, R, S, K, L based on the problem statement
def P (p : Pentagon) := midpoint p.a p.b
def Q (p : Pentagon) := midpoint p.b p.c
def R (p : Pentagon) := midpoint p.c p.d
def S (p : Pentagon) := midpoint p.d p.e
def K (p : Pentagon) := midpoint (P p) (R p)
def L (p : Pentagon) := midpoint (Q p) (S p)

-- The theorem stating the length of segment KL
theorem length_of_KL (p : Pentagon) :
  length (K p) (L p) = 1 / 4 := sorry

end length_of_KL_l701_701497


namespace division_remainder_l701_701664

noncomputable def remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  p % q

theorem division_remainder :
  remainder (Polynomial.X ^ 3) (Polynomial.X ^ 2 + 7 * Polynomial.X + 2) = 47 * Polynomial.X + 14 :=
by
  sorry

end division_remainder_l701_701664


namespace cubic_polynomial_roots_l701_701044

noncomputable def polynomial := fun x : ℝ => x^3 - 2*x - 2

theorem cubic_polynomial_roots
  (x y z : ℝ) 
  (h1: polynomial x = 0)
  (h2: polynomial y = 0)
  (h3: polynomial z = 0):
  x * (y - z)^2 + y * (z - x)^2 + z * (x - y)^2 = 0 :=
by
  -- Solution steps will be filled here to prove the theorem
  sorry

end cubic_polynomial_roots_l701_701044


namespace alternating_series_sum_eq_neg_1012_l701_701632

-- Definition of the alternating series from -1 to -2023
def alternating_series_sum : ℤ :=
  List.sum [ -i * (-1) ^ i | i in List.range (2024)]

-- The statement we want to prove
theorem alternating_series_sum_eq_neg_1012 :
  alternating_series_sum = -1012 :=
sorry -- proof is yet to be written

end alternating_series_sum_eq_neg_1012_l701_701632


namespace good_implies_good_squared_l701_701728

def avg_seq (a : ℕ → ℝ) (k : ℕ) : ℝ := (a k + a (k + 1)) / 2

def is_good (a : ℕ → ℝ) : Prop :=
  ∀ k, int (a k) ∧ int (avg_seq a k) ∧ int (avg_seq avg_seq a k)

theorem good_implies_good_squared (x : ℕ → ℝ) (h : is_good x) : 
  is_good (λ k, (x k) ^ 2) := 
sorry

end good_implies_good_squared_l701_701728


namespace problem_l701_701881

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def sum_arithmetic_sequence (a d n : ℕ) : ℕ := n * a + (n * (n - 1) * d) / 2

theorem problem (a1 S3 : ℕ) (a1_eq : a1 = 2) (S3_eq : S3 = 12) : 
  ∃ a6 : ℕ, a6 = 12 := by
  let a2 := (S3 - a1) / 2
  let d := a2 - a1
  let a6 := a1 + 5 * d
  use a6
  sorry

end problem_l701_701881


namespace f_increasing_range_of_a_l701_701401

noncomputable def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom f_domain : ∀ x, x > 0 → (0 < f(x) ∨ f(x) = 0)
axiom f_multiplicative : ∀ x y, x > 0 → y > 0 → f(x * y) = f(x) + f(y)
axiom f_positive : ∀ x, x > 1 → f(x) > 0

-- Question 1: Prove that f(x) is an increasing function on (0, +∞)

theorem f_increasing (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x < y) : f(x) < f(y) := sorry

-- Question 2: Suppose f(3) = 1 and define sets A and B
def f3 := (f 3 = 1)

def A := {x : ℝ | f(x) > f(x - 1) + 2}

def B (a : ℝ) := {x : ℝ | ∀ a : ℝ, f((a + 1) * x - 1) / (x + 1) > 0}

theorem range_of_a (a : ℝ) : (A ∩ (B a) = ∅) → a ≤ 16 / 9 := sorry

end f_increasing_range_of_a_l701_701401


namespace slope_of_parallel_line_l701_701960

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l701_701960


namespace train_speed_proof_l701_701211

noncomputable def train_speed_km_per_hr (train_length bridge_length time_seconds : ℕ) : ℕ :=
  let total_distance := train_length + bridge_length
  let speed_m_per_s := total_distance / time_seconds.toRat
  let speed_km_per_hr := speed_m_per_s * 3.6
  speed_km_per_hr.toNat

theorem train_speed_proof 
  (train_length : ℕ) 
  (bridge_length : ℕ) 
  (time_seconds : ℕ) 
  (h_train_length : train_length = 140) 
  (h_bridge_length : bridge_length = 235) 
  (h_time_seconds : time_seconds = 30) :
  train_speed_km_per_hr train_length bridge_length time_seconds = 45 := 
by
  rw [h_train_length, h_bridge_length, h_time_seconds]
  unfold train_speed_km_per_hr
  norm_num
  sorry

end train_speed_proof_l701_701211


namespace segment_KL_length_l701_701498

noncomputable def pentagon_side_length : ℝ := 1

variable {A B C D E P Q R S K L : Type}
          [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
          [LinearOrderedField D] [LinearOrderedField E]
          [LinearOrderedField P] [LinearOrderedField Q]
          [LinearOrderedField R] [LinearOrderedField S]
          [LinearOrderedField K] [LinearOrderedField L]

def mid_point (X Y : ℝ) : ℝ := (X + Y) / 2

axiom AB_eq_1 : dist A B = pentagon_side_length
axiom BC_eq_1 : dist B C = pentagon_side_length
axiom CD_eq_1 : dist C D = pentagon_side_length
axiom DE_eq_1 : dist D E = pentagon_side_length
axiom EA_eq_1 : dist E A = pentagon_side_length

axiom P_mid_AB : P = mid_point A B
axiom Q_mid_BC : Q = mid_point B C
axiom R_mid_CD : R = mid_point C D
axiom S_mid_DE : S = mid_point D E

axiom K_mid_PR : K = mid_point P R
axiom L_mid_QS : L = mid_point Q S 

theorem segment_KL_length : dist K L = 1 / 4 := sorry

end segment_KL_length_l701_701498


namespace alternating_series_value_l701_701630

open Nat

theorem alternating_series_value : 
  let s := ∑ k in range 2023, ((-1) ^ k) * (k + 1)
  s = -1012 :=
by
  let s := ∑ k in range 2023, ((-1) ^ k) * (k + 1)
  show s = -1012
  sorry

end alternating_series_value_l701_701630


namespace logarithmic_inequality_range_l701_701346

theorem logarithmic_inequality_range (a : ℝ) (h₁ : log a (a^2 + 1) < log a (2 * a)) (h₂ : log a (2 * a) < 0) : 0 < a ∧ a < 1 := 
sorry

end logarithmic_inequality_range_l701_701346


namespace TB2AC2_is_rectangle_l701_701396

-- Define the triangle, medians, and centroid
variables {A B C A1 B1 C1 T C2 B2 : Point}
variable {BA1_eq_TA1 : BA1 = TA1}
def median (X Y Z : Point) : Line := Line.mk X (midpoint Y Z)
def centroid (X Y Z : Point) : Point := Point.mk ((X.coords + Y.coords + Z.coords) / 3)

-- Define the conditions
def isCentroid : Prop :=
  T = centroid A B C ∧
  BA1_eq_TA1 ∧
  Line.concur (median A B C) (median B C A) (median C A B)

def onExtensions (X1 X2 X : Point) (frac : ℝ) : Prop :=
  dist X1 X2 = frac * dist X1 X

def geomExtensions : Prop :=
  onExtensions C1 C2 C (1/3) ∧
  onExtensions B1 B2 B (1/3)

-- Rectangle condition from parallelogram
def rectangleTB2AC2 (T B2 A C2 : Point) : Prop :=
  isParallelogram T B2 A C2 ∧
  dist T B2 = dist A C2 ∧
  T B2 ∥ A C2 ∧
  B2 C2 ∥ T A ∧
  dist T A = dist B2 C2

-- The goal
theorem TB2AC2_is_rectangle (h1 : isCentroid) (h2 : geomExtensions) : rectangle TB2AC2 :=
sorry

end TB2AC2_is_rectangle_l701_701396


namespace gcd_65_130_l701_701659

theorem gcd_65_130 : Int.gcd 65 130 = 65 := by
  sorry

end gcd_65_130_l701_701659


namespace angle_MBC_is_30_degrees_l701_701011

-- Definition of the problem conditions and the required proof
theorem angle_MBC_is_30_degrees (ABCD : square) (M : point)
  (h1 : ∠MAB = 60) (h2 : ∠MCD = 15) : ∠MBC = 30 :=
sorry

end angle_MBC_is_30_degrees_l701_701011


namespace composite_divisors_property_l701_701252

theorem composite_divisors_property : ∀ n : ℕ, (∃ (k : ℕ) (divisors : Fin k → ℕ), 
  (1 = divisors 0) ∧ (n = divisors (k - 1)) ∧ 
  (∀ i : Fin (k-1), divisors (i+1) - divisors i = (i.val + 1) * (divisors 1 - 1)) ∧ 
  (∀ i : Fin (k-2), ∃ p : ℕ, p.prime ∧ divisors 1 = p)) → n = 4 :=
by
  sorry

end composite_divisors_property_l701_701252


namespace length_of_KL_l701_701494

open Finset
open Classical

-- Define the pentagon and relevant points
def Pentagon :=
  {a b c d e : Point}

-- Conditions: all sides of the pentagon are equal to 1
axiom side_lengths (p : Pentagon) : length (p.a, p.b) = 1 ∧ length (p.b, p.c) = 1 ∧ length (p.c, p.d) = 1 ∧ length (p.d, p.e) = 1 ∧ length (p.e, p.a) = 1

-- Define midpoints
def midpoint (p1 p2 : Point) : Point
axiom midpoint_is_between (p1 p2 : Point) : midpoint p1 p2 = (p1 + p2) / 2

-- Define points P, Q, R, S, K, L based on the problem statement
def P (p : Pentagon) := midpoint p.a p.b
def Q (p : Pentagon) := midpoint p.b p.c
def R (p : Pentagon) := midpoint p.c p.d
def S (p : Pentagon) := midpoint p.d p.e
def K (p : Pentagon) := midpoint (P p) (R p)
def L (p : Pentagon) := midpoint (Q p) (S p)

-- The theorem stating the length of segment KL
theorem length_of_KL (p : Pentagon) :
  length (K p) (L p) = 1 / 4 := sorry

end length_of_KL_l701_701494


namespace duration_of_call_l701_701025

def initial_credit : ℝ := 30
def remaining_credit : ℝ := 26.48
def cost_per_minute : ℝ := 0.16
def credit_used : ℝ := initial_credit - remaining_credit
def minutes_of_call : ℕ := (credit_used / cost_per_minute).toInt

theorem duration_of_call :
  minutes_of_call = 22 :=
by
  -- Placeholder proof statement
  sorry

end duration_of_call_l701_701025


namespace scientific_notation_representation_l701_701388

-- Define the concept of nanoseconds in seconds
def nanosecond_in_seconds : ℝ := 10 ^ (-9)

-- Define the commitment time in nanoseconds
def commitment_time_ns : ℝ := 20

-- The goal is to prove that the commitment time in scientific notation is 2 × 10^(-8) seconds
theorem scientific_notation_representation : (commitment_time_ns * nanosecond_in_seconds) = 2 * 10^(-8) :=
by
  sorry

end scientific_notation_representation_l701_701388


namespace problem_l701_701710

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem problem (a : ℝ) :
  (∀ x : ℝ, 1 < x → f a x + a < 0) ↔ a ≥ 1 :=
begin
  sorry
end

end problem_l701_701710


namespace original_amount_in_cookie_jar_l701_701899

theorem original_amount_in_cookie_jar (doris_spent martha_spent money_left_in_jar original_amount : ℕ)
  (h1 : doris_spent = 6)
  (h2 : martha_spent = doris_spent / 2)
  (h3 : money_left_in_jar = 15)
  (h4 : original_amount = money_left_in_jar + doris_spent + martha_spent) :
  original_amount = 24 := 
sorry

end original_amount_in_cookie_jar_l701_701899


namespace Laura_runs_8_24_mph_approx_l701_701412

noncomputable def LauraWorkout : ℝ :=
  let total_minutes := 110
  let transition_minutes := 5
  let total_hours := (total_minutes - transition_minutes) / 60
  let distance_bike := 20
  let distance_run := 5
  λ x: ℝ,
    let time_bike := distance_bike / (2 * x + 1)
    let time_run := distance_run / x
    time_bike + time_run = total_hours ∧ x ≈ 8.24


theorem Laura_runs_8_24_mph_approx (x : ℝ) : 
  (LauraWorkout x) → x ≈ 8.24 :=
by 
  sorry

end Laura_runs_8_24_mph_approx_l701_701412


namespace product_of_digits_l701_701352

theorem product_of_digits (n A B : ℕ) (h1 : n % 6 = 0) (h2 : A + B = 12) (h3 : n = 10 * A + B) : 
  (A * B = 32 ∨ A * B = 36) :=
by 
  sorry

end product_of_digits_l701_701352


namespace regular_decagon_interior_angle_l701_701548

theorem regular_decagon_interior_angle :
  ∀ (n : ℕ), n = 10 → (n - 2) * 180 / n = 144 := by
  intro n hn
  rw [hn]
  norm_num
  sorry

end regular_decagon_interior_angle_l701_701548


namespace distance_swum_downstream_l701_701197

-- Definitions of the given conditions
def V_m := 6.5 -- Speed of man in still water (km/h)
def D_u := 10 -- Distance swum upstream (km)
def T_u := 2 -- Time to swim upstream (hours)
def T_d := 2 -- Time to swim downstream (hours)

-- Statement to prove: the distance swum downstream
theorem distance_swum_downstream : 
  ∃ D_d V_c : ℝ, 
    V_u = V_m - V_c ∧ 
    D_u / V_u = T_u ∧ 
    D_d / (V_m + V_c) = T_d ∧ 
    D_d = 16 :=
by etc.

end distance_swum_downstream_l701_701197


namespace parallel_slope_l701_701934

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l701_701934


namespace average_visitors_per_day_correct_l701_701567

-- Define the average number of visitors on Sundays
def avg_visitors_sunday : ℕ := 660

-- Define the average number of visitors on other days
def avg_visitors_other : ℕ := 240

-- Define the number of Sundays in a 30-day month starting with a Sunday
def num_sundays_in_month : ℕ := 5

-- Define the number of other days in a 30-day month starting with a Sunday
def num_other_days_in_month : ℕ := 25

-- Calculate the total number of visitors in the month
def total_visitors_in_month : ℕ :=
  (num_sundays_in_month * avg_visitors_sunday) + (num_other_days_in_month * avg_visitors_other)

-- Define the number of days in the month
def days_in_month : ℕ := 30

-- Define the average number of visitors per day
def avg_visitors_per_day := total_visitors_in_month / days_in_month

-- State the theorem to be proved
theorem average_visitors_per_day_correct :
  avg_visitors_per_day = 310 :=
by
  sorry

end average_visitors_per_day_correct_l701_701567


namespace length_of_other_train_l701_701124

def speed_train1 := 60 -- km/hr
def speed_train2 := 40 -- km/hr
def length_train1 := 140 -- meters
def crossing_time := 11.519078473722104 -- seconds
def relative_speed : ℝ := (100 * 1000) / 3600 -- km/hr converted to m/s

theorem length_of_other_train :
  length_train1 + (relative_speed * crossing_time - length_train1) = 180 := by
  sorry

end length_of_other_train_l701_701124


namespace system_equation_max_y_l701_701832

theorem system_equation_max_y :
  ∃ y x : ℝ, (3 * x^2 - x * y = 1 ∧ 9 * x * y + y^2 = 22) ∧ y = 5 :=
begin
  sorry
end

end system_equation_max_y_l701_701832


namespace AX_is_symmedian_l701_701861

theorem AX_is_symmedian 
  (A B C D E X : Point) 
  (h1 : ∃ (α β γ : Point) (H : Triangle α β γ), A = α ∧ B = β ∧ C = γ) 
  (h2 : InternalAngleBisector A B C D) 
  (h3 : ExternalAngleBisector A B C E) 
  (h4 : LineContainsDNE D E) 
  (h5 : CircleIntersectsCircumcircle A B C D E X) : 
  IsSymmedian A X B C := 
sorry

end AX_is_symmedian_l701_701861


namespace slope_parallel_to_original_line_l701_701942

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l701_701942


namespace jacob_calories_l701_701404

theorem jacob_calories (goal : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) 
  (h_goal : goal = 1800) 
  (h_breakfast : breakfast = 400) 
  (h_lunch : lunch = 900) 
  (h_dinner : dinner = 1100) : 
  (breakfast + lunch + dinner) - goal = 600 :=
by 
  sorry

end jacob_calories_l701_701404


namespace find_b_l701_701512

def nabla (a b : ℤ) (h : a ≠ b) : ℤ := (a + b) / (a - b)

theorem find_b (b : ℤ) (h : 3 ≠ b) (h_eq : nabla 3 b h = -4) : b = 5 :=
sorry

end find_b_l701_701512


namespace cookie_jar_initial_amount_l701_701901

variable (initial_amount : ℕ)
variable (doris_spent : ℕ := 6)
variable (martha_spent : ℕ := doris_spent / 2)
variable (remaining : ℕ := 15)

theorem cookie_jar_initial_amount :
  initial_amount = doris_spent + martha_spent + remaining :=
begin
  sorry
end

end cookie_jar_initial_amount_l701_701901


namespace greatest_value_q_minus_r_l701_701158

def max_difference_q_r (q r : ℕ) : ℕ :=
  if q > r then q - r else r - q

theorem greatest_value_q_minus_r :
  ∀ (q r : ℕ), 
    (9 < q) ∧ (q < 100) ∧ (9 < r) ∧ (r < 100) ∧ 
    (∃ x y : ℕ, (x > 0) ∧ (x < 10) ∧ (y > 0) ∧ (y < 10) ∧ (q = 10 * x + y) ∧ (r = 10 * y + x)) ∧ 
    max_difference_q_r q r < 30 →
    ∃ (x y : ℕ), q − r = 27 :=
by
  sorry

end greatest_value_q_minus_r_l701_701158


namespace modulo_17_residue_l701_701129

theorem modulo_17_residue : (3^4 + 6 * 49 + 8 * 137 + 7 * 34) % 17 = 5 := 
by
  sorry

end modulo_17_residue_l701_701129


namespace max_value_quadratic_l701_701550

theorem max_value_quadratic : ∀ s : ℝ, ∃ M : ℝ, (∀ s : ℝ, -3 * s^2 + 54 * s - 27 ≤ M) ∧ M = 216 :=
by
  sorry

end max_value_quadratic_l701_701550


namespace arithmetic_sequence_ninth_term_l701_701891

theorem arithmetic_sequence_ninth_term 
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29)
  : a + 8 * d = 35 :=
by
  sorry

end arithmetic_sequence_ninth_term_l701_701891


namespace willie_initial_bananas_l701_701562

/-- Given that Willie will have 13 bananas, we need to prove that the initial number of bananas Willie had was some specific number X. --/
theorem willie_initial_bananas (initial_bananas : ℕ) (final_bananas : ℕ) 
    (h : final_bananas = 13) : initial_bananas = initial_bananas :=
by
  sorry

end willie_initial_bananas_l701_701562


namespace parallel_line_slope_l701_701981

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l701_701981


namespace tan_angle_identity_l701_701690

open Real

theorem tan_angle_identity (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : sin β / cos β = (1 + cos (2 * α)) / (2 * cos α + sin (2 * α))) :
  tan (α + 2 * β + π / 4) = -1 := 
sorry

end tan_angle_identity_l701_701690


namespace PQ_parallel_AD_BC_l701_701766

-- Define the points and lines in the conditions
variables {Point Line : Type} [IncidenceGeometry Point Line]

variables (A B C D P Q : Point) -- Points in the problem
variables (AD BC CD AB BP CQ : Line) -- Lines in the problem

-- Assume the configuration of trapezoid and parallel lines
axiom Trapezoid_ABCD : Trapezoid A B C D AD BC
axiom Line_BP_parallel_CD : Parallel BP CD
axiom Line_CQ_parallel_AB : Parallel CQ AB
axiom B_on_BP : Incident B BP
axiom C_on_CQ : Incident C CQ
axiom P_on_BP : Incident P BP
axiom Q_on_CQ : Incident Q CQ
axiom P_on_AC : Incident P (LineJoin A C)
axiom Q_on_BD : Incident Q (LineJoin B D)

-- Define the goal: proving PQ is parallel to AD and BC
theorem PQ_parallel_AD_BC :
  Parallel (LineJoin P Q) AD ∧ Parallel (LineJoin P Q) BC :=
sorry

end PQ_parallel_AD_BC_l701_701766


namespace slope_of_parallel_line_l701_701921

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l701_701921


namespace valid_song_distribution_l701_701223

open Finset

noncomputable def count_ways : ℕ :=
  let AB := {1}  -- At least one song liked by Amy and Beth but not Jo
  let BC := {2}  -- At least one song liked by Beth and Jo but not Amy
  let CA := {3}  -- At least one song liked by Jo and Amy but not Beth
  let remaining_songs := {4, 5}
  let choices := {AB, BC, CA, ∅, {4}, {5}}
  (4^2 + 4 * 3) -- Case 1: Remaining two in {N, A, B, C}; Case 2: One in {N, A, B, C} and one more in {AB, BC, CA}

theorem valid_song_distribution :
  count_ways = 28 :=
by {
  sorry
}

end valid_song_distribution_l701_701223


namespace reduced_price_is_55_l701_701568

variables (P R : ℝ) (X : ℕ)

-- Conditions
def condition1 : R = 0.75 * P := sorry
def condition2 : P * X = 1100 := sorry
def condition3 : 0.75 * P * (X + 5) = 1100 := sorry

-- Theorem
theorem reduced_price_is_55 (P R : ℝ) (X : ℕ) (h1 : R = 0.75 * P) (h2 : P * X = 1100) (h3 : 0.75 * P * (X + 5) = 1100) :
  R = 55 :=
sorry

end reduced_price_is_55_l701_701568


namespace max_value_equal_l701_701449

-- Define the general setup
def log_base (b a : ℝ) := Real.log a / Real.log b

noncomputable def a := log_base 5 6
noncomputable def b := log_base 6 5

-- Define the trigonometric identity for the condition
axiom trig_identity (x : ℝ) : (Real.sin x)^2 + (Real.cos x)^2 = 1

-- Define the maximum value proof
theorem max_value_equal :
  ∃ x, Real.sin x = 1 ∨ Real.cos x = 1 ∧ 
  a = log_base 5 6 ∧
  b = log_base 6 5 ∧ 
  b = 1 / a ∧
  (a ^ (Real.sin x) = 1 ∧ (b ^ (Real.cos x) = 1)) :=
sorry

end max_value_equal_l701_701449


namespace rationalize_denominator_l701_701458

theorem rationalize_denominator (a b : ℝ) (h : b = 343) (h_nonzero : b ≠ 0) : (a = 7) → (\sqrt b = 7 * \sqrt 7) → \frac{a}{\sqrt b} = \frac{\sqrt 7}{7} :=
by
  sorry

end rationalize_denominator_l701_701458


namespace number_of_possible_r_l701_701511

def is_four_place_decimal (r : ℚ) : Prop :=
  ∃ (a b c d : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
  r = (10 ^ (-4 : ℤ)) * (a + 10 * b + 10^2 * c + 10^3 * d)

def in_range (r : ℚ) : Prop := 
  (2429 : ℚ) / 10000 ≤ r ∧ r ≤ (2857 : ℚ) / 10000

theorem number_of_possible_r : ∃ (n : ℕ), 
  (∀ (r : ℚ), is_four_place_decimal r → in_range r → r ∈ {(2429/10000), ..., (2857/10000)}) ∧ 
  n = 428 := 
by 
sory -- As requested, the proof is omitted.

end number_of_possible_r_l701_701511


namespace tingting_solution_correct_l701_701121

noncomputable def product_of_square_roots : ℝ :=
  (Real.sqrt 8) * (Real.sqrt 18)

theorem tingting_solution_correct : product_of_square_roots = 12 := by
  sorry

end tingting_solution_correct_l701_701121


namespace arithmetic_sequence_geometric_sum_l701_701418

theorem arithmetic_sequence_geometric_sum (a₁ a₂ d : ℕ) (h₁ : d ≠ 0) 
    (h₂ : (2 * a₁ + d)^2 = a₁ * (4 * a₁ + 6 * d)) :
    a₂ = 3 * a₁ :=
by
  sorry

end arithmetic_sequence_geometric_sum_l701_701418


namespace river_depth_mid_may_l701_701374

-- Definitions corresponding to the conditions
def depth_mid_june (D : ℕ) : ℕ := D + 10
def depth_mid_july (D : ℕ) : ℕ := 3 * (depth_mid_june D)

-- The theorem statement
theorem river_depth_mid_may (D : ℕ) (h : depth_mid_july D = 45) : D = 5 :=
by
  sorry

end river_depth_mid_may_l701_701374


namespace point_not_on_graph_l701_701561

-- Define the function y = (x-1)/(x+2)
def f (x : ℝ) : ℝ := (x - 1) / (x + 2)

-- Define the proposed points
def p₁ : ℝ × ℝ := (-2, -3)
def p₂ : ℝ × ℝ := (0, -1 / 2)
def p₃ : ℝ × ℝ := (1, 0)
def p₄ : ℝ × ℝ := (-1, 2 / 3)
def p₅ : ℝ × ℝ := (-3, -1)

-- State the theorem to prove
theorem point_not_on_graph : ¬ (p₁.2 = f p₁.1) :=
by sorry

end point_not_on_graph_l701_701561


namespace intersect_complement_eq_l701_701718

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {1, 3, 6}
def comp_B : Set ℕ := U \ B

theorem intersect_complement_eq :
  A ∩ comp_B = {4, 5} := by
  sorry

end intersect_complement_eq_l701_701718


namespace eccentricity_of_ellipse_l701_701842

-- Definitions from the problem conditions
def ellipse := { x : ℝ | ∃ y, (x^2 / m^2 + y^2 / 9 = 1) ∧ m > 0 ∧ (4, 0) is_a_focus }

-- Lean statement to prove the eccentricity
theorem eccentricity_of_ellipse (m : ℝ) (h : m > 0) : 
  (one_focus (4, 0) → eccentricity (ellipse) = 4 / 5) :=
sorry

end eccentricity_of_ellipse_l701_701842


namespace discount_correct_l701_701081

variable {a : ℝ} (discount_percent : ℝ) (profit_percent : ℝ → ℝ)

noncomputable def calc_discount : ℝ :=
  discount_percent

theorem discount_correct :
  (discount_percent / 100) = (33 + 1 / 3) / 100 →
  profit_percent (discount_percent / 100) = (3 / 2) * (discount_percent / 100) →
  a * (1 - discount_percent / 100) * (1 + profit_percent (discount_percent / 100)) = a →
  discount_percent = 33 + 1 / 3 :=
by sorry

end discount_correct_l701_701081


namespace measure_of_B_area_of_triangle_l701_701768

variables (A B C a b c R : ℝ)

-- Conditions
axiom triangle_sides_opposite (A B C a b c : ℝ) (hA : a = 2 * R * Real.sin A)
  (hB : b = 2 * R * Real.sin B) (hC : c = 2 * R * Real.sin C)
  (h_cos_ratio : Real.cos B / Real.cos C = -b / (2 * a + c)) : Prop

-- Problem 1
theorem measure_of_B (hA : a = 2 * R * Real.sin A) (hB : b = 2 * R * Real.sin B)
  (hC : c = 2 * R * Real.sin C) (h_cos_ratio : Real.cos B / Real.cos C = -b / (2 * a + c))
  (hB_inside : A + B + C = Real.pi) (h_sin_A : Real.sin A ≠ 0) :
  B = 2 * Real.pi / 3 :=
by
  apply triangle_sides_opposite A B C a b c hA hB hC h_cos_ratio
  sorry

-- Problem 2
theorem area_of_triangle (hA : a = 2 * R * Real.sin A) (hB : b = 2 * R * Real.sin B)
  (hC : c = 2 * R * Real.sin C) (h_cos_ratio : Real.cos B / Real.cos C = -b / (2 * a + c))
  (hB_inside : A + B + C = Real.pi) (h_sin_A : Real.sin A ≠ 0)
  (hb_val : b = Real.sqrt 13) (ha_c_val : a + c = 4) (h_cos_B : Real.cos B = -1/2):
  let ac := sqrt (a * c) in -- placeholder for calculating ac = 3
  (1/2) * a * c * Real.sin B = 3 * Real.sqrt 3 / 4 :=
by
  apply triangle_sides_opposite A B C a b c hA hB hC h_cos_ratio
  sorry

end measure_of_B_area_of_triangle_l701_701768


namespace rationalize_denominator_l701_701457

theorem rationalize_denominator (a b : ℝ) (h : b = 343) (h_nonzero : b ≠ 0) : (a = 7) → (\sqrt b = 7 * \sqrt 7) → \frac{a}{\sqrt b} = \frac{\sqrt 7}{7} :=
by
  sorry

end rationalize_denominator_l701_701457


namespace negation_of_proposition_range_of_m_l701_701715

noncomputable def proposition (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * x - m - 1 < 0

theorem negation_of_proposition (m : ℝ) : ¬ proposition m ↔ ∀ x : ℝ, x^2 + 2 * x - m - 1 ≥ 0 :=
sorry

theorem range_of_m (m : ℝ) : proposition m → m > -2 :=
sorry

end negation_of_proposition_range_of_m_l701_701715


namespace range_of_a_l701_701048

noncomputable def P (X : ℕ → ℝ) (i : ℕ) : ℝ := i / 10

theorem range_of_a (a : ℝ) :
  (P (λ i, P i) 1 + P (λ i, P i) 2 + P (λ i, P i) 3 = 3/5) →
  3 < a ∧ a ≤ 4 :=
by
  assume h : P (λ i, P i) 1 + P (λ i, P i) 2 + P (λ i, P i) 3 = 3/5
  sorry

end range_of_a_l701_701048


namespace remainder_of_digit_sum_divisible_l701_701469

def digit_sum_contribution (n : ℕ) : ℕ := sorry -- Helper function to calculate the digit sum contribution, not required for problem statement

theorem remainder_of_digit_sum_divisible :
  let seq_digits := 198 in
  let sum_of_digits := digit_sum_contribution seq_digits in
  sum_of_digits % 9 = 6 := 
by 
  sorry

end remainder_of_digit_sum_divisible_l701_701469


namespace Jamir_swims_more_l701_701407

def Julien_distance_per_day : ℕ := 50
def Sarah_distance_per_day (J : ℕ) : ℕ := 2 * J
def combined_distance_per_week (J S M : ℕ) : ℕ := 7 * (J + S + M)

theorem Jamir_swims_more :
  let J := Julien_distance_per_day
  let S := Sarah_distance_per_day J
  ∃ M, combined_distance_per_week J S M = 1890 ∧ (M - S = 20) := by
    let J := Julien_distance_per_day
    let S := Sarah_distance_per_day J
    use 120
    sorry

end Jamir_swims_more_l701_701407


namespace Isabela_spent_l701_701770

theorem Isabela_spent (num_pencils : ℕ) (cost_per_item : ℕ) (num_cucumbers : ℕ)
  (h1 : cost_per_item = 20)
  (h2 : num_cucumbers = 100)
  (h3 : num_cucumbers = 2 * num_pencils)
  (discount : ℚ := 0.20) :
  let pencil_cost := num_pencils * cost_per_item
  let cucumber_cost := num_cucumbers * cost_per_item
  let discounted_pencil_cost := pencil_cost * (1 - discount)
  let total_cost := cucumber_cost + discounted_pencil_cost
  total_cost = 2800 := by
  -- Begin proof. We will add actual proof here later.
  sorry

end Isabela_spent_l701_701770


namespace high_low_card_game_l701_701750

theorem high_low_card_game (h l : ℕ) 
    (total_cards: h + l = 52) 
    (equal_cards: h = l)
    (value_high: ∀ h, 2 * h)
    (value_low: ∀ l, l)
    (points: 2 * h + l = 5) :
    l = 1 ∨ l = 3 ∨ l = 5 :=
by sorry

end high_low_card_game_l701_701750


namespace parallel_slope_l701_701936

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l701_701936


namespace parabola_trajectory_l701_701083

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def distance_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / real.sqrt (a^2 + b^2)

theorem parabola_trajectory :
  ∀ (M : ℝ × ℝ),
  distance M (0, 4) = distance_to_line M 0 1 4 →
  M.1^2 = 16 * M.2 :=
by
  sorry

end parabola_trajectory_l701_701083


namespace probability_xi_lt_70_l701_701367

noncomputable def normalDistr (μ σ : ℝ) : ProbabilityTheory.Measure ℝ :=
ProbabilityTheory.Measure.gaussian μ σ

theorem probability_xi_lt_70 {σ : ℝ} (hσ : σ > 0)  
  (h1 : ∀ ξ, ProbabilityTheory.Measure.prob (normalDistr 90 σ) (set.Icc 70 110) = 0.6) :
  ProbabilityTheory.Measure.prob (normalDistr 90 σ) (set.Iic 70) = 0.2 :=
sorry

end probability_xi_lt_70_l701_701367


namespace min_value_f_l701_701343

noncomputable def f (x : ℝ): ℝ := 
  if 0 < x ∧ x < 1 then x * Real.log x / Real.log 2 + (1 - x) * Real.log (1 - x) / Real.log 2 
  else 0

theorem min_value_f : ∃ (y : ℝ), (∀ x, 0 < x ∧ x < 1 → f(x) ≥ y) ∧ y = -1 :=
sorry

end min_value_f_l701_701343


namespace line_ellipse_intersect_l701_701870

theorem line_ellipse_intersect (m k : ℝ) (h₀ : ∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1) : m ≥ 1 ∧ m ≠ 5 :=
sorry

end line_ellipse_intersect_l701_701870


namespace vegetables_sold_mass_l701_701591

/-- Define the masses of the vegetables --/
def mass_carrots : ℕ := 15
def mass_zucchini : ℕ := 13
def mass_broccoli : ℕ := 8

/-- Define the total mass of installed vegetables --/
def total_mass : ℕ := mass_carrots + mass_zucchini + mass_broccoli

/-- Define the mass of vegetables sold (half of the total mass) --/
def mass_sold : ℕ := total_mass / 2

/-- Prove that the mass of vegetables sold is 18 kg --/
theorem vegetables_sold_mass : mass_sold = 18 := by
  sorry

end vegetables_sold_mass_l701_701591


namespace negation_of_proposition_p_l701_701877

theorem negation_of_proposition_p :
  (¬(∃ x : ℝ, 0 < x ∧ Real.log x > x - 1)) ↔ (∀ x : ℝ, 0 < x → Real.log x ≤ x - 1) :=
by
  sorry

end negation_of_proposition_p_l701_701877


namespace intersection_sums_l701_701514

def parabola1 (x : ℝ) : ℝ := (x - 2)^2
def parabola2 (y : ℝ) : ℝ := (y - 2)^2 - 6

theorem intersection_sums (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) 
  (h1 : y1 = parabola1 x1) (h2 : y2 = parabola1 x2)
  (h3 : y3 = parabola1 x3) (h4 : y4 = parabola1 x4)
  (k1 : x1 + 6 = y1^2 - 4*y1 + 4) (k2 : x2 + 6 = y2^2 - 4*y2 + 4)
  (k3 : x3 + 6 = y3^2 - 4*y3 + 4) (k4 : x4 + 6 = y4^2 - 4*y4 + 4) :
  x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 16 := 
sorry

end intersection_sums_l701_701514


namespace compare_P_Q_l701_701677

variable (a : ℝ)

def P (a : ℝ) := sqrt a + sqrt (a + 7)
def Q (a : ℝ) := sqrt (a + 3) + sqrt (a + 4)

theorem compare_P_Q (h : 0 ≤ a) : P a < Q a :=
by
  sorry

end compare_P_Q_l701_701677


namespace cannot_divide_good_triangle_l701_701684

def is_right_triangle (a b c : ℕ) :=
  a^2 + b^2 = c^2

def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

def area (a b : ℕ) : ℕ :=
  (a * b) / 2

def is_good_polygon (a b c : ℕ) : Prop :=
  perimeter a b c = area a b

theorem cannot_divide_good_triangle (a b c : ℕ) (h1 : is_right_triangle a b c) (h2 : is_good_polygon a b c) :
  ¬ (∃ (n : ℕ) (a1 a2 b1 b2 c1 c2 : vector ℕ n), 
    ∀ i, is_good_polygon (a1.get i) (b1.get i) (c1.get i) ∧ is_good_polygon (a2.get i) (b2.get i) (c2.get i) ∧
    a = a1.to_list.sum + a2.to_list.sum ∧
    b = b1.to_list.sum + b2.to_list.sum ∧
    c = c1.to_list.sum + c2.to_list.sum) :=
begin
  sorry
end

end cannot_divide_good_triangle_l701_701684
