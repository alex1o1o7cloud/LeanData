import Mathlib

namespace cannot_obtain_100_pieces_l485_485952

theorem cannot_obtain_100_pieces : ¬ ∃ n : ℕ, 1 + 2 * n = 100 := by
  sorry

end cannot_obtain_100_pieces_l485_485952


namespace square_simplification_eq_l485_485032

theorem square_simplification_eq (y : ℝ) : (5 - real.sqrt (y^2 - 25)) ^ 2 = y^2 - 10 * real.sqrt (y^2 - 25) :=
by 
  sorry

end square_simplification_eq_l485_485032


namespace arithmetic_sequence_problem_l485_485349

theorem arithmetic_sequence_problem (a1 d : ℝ) (S : ℕ → ℝ) 
  (h1 : S 10 = 100) 
  (h2 : S 100 = 10) 
  (h_seq : ∀ (n : ℕ), 
    S n = n * a1 + (n * (n - 1)) / 2 * d) : 
  S 110 = -110 :=
begin
  sorry
end

end arithmetic_sequence_problem_l485_485349


namespace rationalize_sqrt_l485_485429

theorem rationalize_sqrt (a : ℝ) (h : a = 2/(3 - real.sqrt 7)) : 
  3 * a^2 - 6 * a - 1 = 7 := 
by
  sorry

end rationalize_sqrt_l485_485429


namespace mrs_thomson_saved_l485_485815

noncomputable def initial_incentive : ℤ := 240
noncomputable def spent_on_food : ℤ := (1/3 : ℚ) * initial_incentive
noncomputable def spent_on_clothes : ℤ := (1/5 : ℚ) * initial_incentive
noncomputable def total_spent : ℤ := spent_on_food + spent_on_clothes
noncomputable def remaining_money : ℤ := initial_incentive - total_spent
noncomputable def saved_money : ℤ := (3/4 : ℚ) * remaining_money

theorem mrs_thomson_saved : saved_money = 84 := by sorry

end mrs_thomson_saved_l485_485815


namespace balance_scale_equation_l485_485817

theorem balance_scale_equation 
  (G Y B W : ℝ)
  (h1 : 4 * G = 8 * B)
  (h2 : 3 * Y = 6 * B)
  (h3 : 2 * B = 3 * W) : 
  3 * G + 4 * Y + 3 * W = 16 * B :=
by
  sorry

end balance_scale_equation_l485_485817


namespace smallest_value_of_n_l485_485635

/-- Given that Casper has exactly enough money to buy either 
  18 pieces of red candy, 20 pieces of green candy, 
  25 pieces of blue candy, or n pieces of purple candy where 
  each purple candy costs 30 cents, prove that the smallest 
  possible value of n is 30.
-/
theorem smallest_value_of_n
  (r g b n : ℕ)
  (h : 18 * r = 20 * g ∧ 20 * g = 25 * b ∧ 25 * b = 30 * n) : 
  n = 30 :=
sorry

end smallest_value_of_n_l485_485635


namespace no_three_distinct_rational_solutions_l485_485629

theorem no_three_distinct_rational_solutions : 
  ¬ ∃ (r : ℝ), ∃ (a b c k : ℤ), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (a / k, b / k, c / k).Prod.mem (roots (X^3 - 2023 * X^2 - 2023 * X + polynomial.C r)) ∧ 
  (Int.gcd (Int.gcd a b) (Int.gcd c k)) = 1 := 
sorry

end no_three_distinct_rational_solutions_l485_485629


namespace coefficient_x2_binomial_l485_485320

def integral_condition (f : ℝ → ℝ) (a b c : ℝ) := ∫ x in a..b, f x = c

-- Given conditions
constant n : ℝ
constant integral_eq : integral_condition (λ x, |x - 5|) 0 n 25

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Statement to prove
theorem coefficient_x2_binomial (n : ℕ) (h1 : ∫ x in 0..n, |x - 5| = 25) : 
  binomial n 2 * 2^2 * (-1)^8 = 180 :=
sorry

end coefficient_x2_binomial_l485_485320


namespace value_of_y_at_64_l485_485724

theorem value_of_y_at_64 (x y k : ℝ) (h1 : y = k * x^(1/3)) (h2 : 8^(1/3) = 2) (h3 : y = 4 ∧ x = 8):
  y = 8 :=
by {
  sorry
}

end value_of_y_at_64_l485_485724


namespace A_finishes_work_in_10_days_l485_485147

-- Define the conditions and what needs to be proven as a Lean theorem
theorem A_finishes_work_in_10_days 
    (A B : ℕ)
    (h1 : B = 15)
    (h2 : (1920 : ℚ) / 3200 = 3 / 5)
    (h3 : B ∈ ℕ) :
    A = 10 :=
by
  -- Given conditions about B and wage distribution.
  have B_rate := (1 : ℚ) / B -- B's rate of work per day
  have total_wages := 3200
  have A_wage := 1920
  have wage_ratio := A_wage / total_wages
  
  -- Confirm the wage ratio
  rw [h2] at wage_ratio
  have total_wage_ratio := (3 : ℚ) / 5
  
  -- Convert the conditions to numeric form
  have A_work_ratio := total_wage_ratio * (1 / B_rate)
  
  -- Working together means combined rates needed
  have A_B_work_ratio := A_work_ratio * (B_rate / (B_rate + (1 : ℚ) / A))
  
  -- Proving that A's work rate is proportional to B's work rate and the given combined work ratio
  rw [h1] at B_rate
  have combined_rate := A_work_ratio / (A_work_ratio + B_rate)
  have final_rate := (3 / 5) * combined_rate
  
  -- Finally deducing A's total days given the rates
  assumption sorry

end A_finishes_work_in_10_days_l485_485147


namespace number_of_tessellating_polygons_l485_485258

-- Define the types of regular polygons we are considering
inductive RegularPolygon
| triangle
| square
| pentagon
| hexagon
| octagon
| decagon
| dodecagon

-- Define a predicate that returns true if a regular polygon can tessellate a plane
def can_tessellate : RegularPolygon → Prop
| RegularPolygon.triangle := true
| RegularPolygon.square := true
| RegularPolygon.hexagon := true
| _ := false

-- Define the target statement
theorem number_of_tessellating_polygons :
  (Finset.filter can_tessellate
    (Finset.of_list [RegularPolygon.triangle, RegularPolygon.square, RegularPolygon.pentagon,
                      RegularPolygon.hexagon, RegularPolygon.octagon, RegularPolygon.decagon,
                      RegularPolygon.dodecagon])).card = 3 :=
by sorry

end number_of_tessellating_polygons_l485_485258


namespace expected_defective_products_l485_485334

theorem expected_defective_products (genuine defective : ℕ) (h_g : genuine = 9) (h_d : defective = 3) : 
  (9 : ℚ) / (5 : ℚ) = ( Σ ( ξ in finset.range 4 ), 
    if ξ = 0 
    then (9 / 12 : ℚ) 
    else if ξ = 1 
    then (3 / 12 * 9 / 11 : ℚ) 
    else if ξ = 2 
    then (3 / 12 * 2 / 11 * 9 / 10 * ξ : ℚ) 
    else (3 / 12 * 2 / 11 * 1 / 10 * ξ : ℚ) ) := 
by 
  sorry

end expected_defective_products_l485_485334


namespace parabola_symmetry_l485_485267

theorem parabola_symmetry (a b x1 x2 m : ℝ) (hx1x2 : x1 * x2 = -1/2)
  (ha : a = 2) (habs : b > 0) 
  (hy : ∀ x, y = a * x^2) (hsym : ∀ A B m, symmetric A B (y = x + m)) :
  m = 3/2 := by
  sorry

end parabola_symmetry_l485_485267


namespace box_width_l485_485912

theorem box_width (W : ℕ) (h₁ : 15 * W * 13 = 3120) : W = 16 := by
  sorry

end box_width_l485_485912


namespace inequality_holds_for_any_x_l485_485374

theorem inequality_holds_for_any_x (n : ℕ) (a : Fin n → ℝ) (h₀ : n > 0) 
  (h₁ : ∀ i j, i ≤ j → a i ≤ a j) (h₂ : (Finset.sum (Finset.range n) (λ i, (i + 1) * (a ⟨i, Nat.lt_of_lt_of_le (Nat.lt_succ_self i) (Nat.lt_of_succ_lt_succ h₀).le⟩))) = 0) :
  ∀ x : ℝ, 0 ≤ (Finset.sum (Finset.range n) (λ i, (a ⟨i, Nat.lt_of_lt_of_le (Nat.lt_succ_self i) (Nat.lt_of_succ_lt_succ h₀).le⟩) * ⌊(i + 1) * x⌋)) :=
begin
  sorry
end

end inequality_holds_for_any_x_l485_485374


namespace gcd_45_75_l485_485091

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485091


namespace binomial_prime_divisibility_l485_485425

theorem binomial_prime_divisibility (p : ℕ) (a b : ℤ) (hp : Nat.Prime p) : p ∣ ((a + b)^p - a^p - b^p) := 
sorry

end binomial_prime_divisibility_l485_485425


namespace triangle_rearrangement_count_l485_485630

-- Define the conditions as part of a structured theorem
theorem triangle_rearrangement_count :
  ∀ (ABC : Triangle),
  -- Dividing side BC into 4 equal parts
  ∃ (D E F : Point), 
  divides_equally BC 4 D E F ∧
  -- Connecting points D, E, F to A and drawing parallel lines
  ∃ (lines_drawn : Construction),
  lines_drawn_correct ABC D E F lines_drawn →
  -- The total number of ways to rearrange the parts
  num_ways_to_rearrange ABC lines_drawn 4782969 :=
sorry

end triangle_rearrangement_count_l485_485630


namespace peter_needs_5000_for_vacation_l485_485421

variable (currentSavings : ℕ) (monthlySaving : ℕ) (months : ℕ)

-- Conditions
def peterSavings := currentSavings
def monthlySavings := monthlySaving
def savingDuration := months

-- Goal
def vacationFundsRequired (currentSavings monthlySaving months : ℕ) : ℕ :=
  currentSavings + (monthlySaving * months)

theorem peter_needs_5000_for_vacation
  (h1 : currentSavings = 2900)
  (h2 : monthlySaving = 700)
  (h3 : months = 3) :
  vacationFundsRequired currentSavings monthlySaving months = 5000 := by
  sorry

end peter_needs_5000_for_vacation_l485_485421


namespace pascals_triangle_even_count_15_l485_485614

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

def is_even (x : ℕ) : Prop := x % 2 = 0

def count_even_in_row (n : ℕ) : ℕ :=
  (finset.range (n + 1)).count (λ k, is_even (pascal n k))

def count_even_in_first_15_rows : ℕ :=
  (finset.range 15).sum count_even_in_row

theorem pascals_triangle_even_count_15 :
  count_even_in_first_15_rows = 64 :=
by sorry

end pascals_triangle_even_count_15_l485_485614


namespace gcd_45_75_l485_485115

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485115


namespace even_count_in_pascal_triangle_l485_485609

-- Define the binomial coefficient as a function
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define a predicate to check if a number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Count the number of even integers in the top 15 rows of Pascal's Triangle
def count_even_pascal (rows : ℕ) : ℕ :=
  (Finset.range rows).sum (λ n => (Finset.range (n + 1)).count (λ k => is_even (binom n k)))

-- Statement of the problem
theorem even_count_in_pascal_triangle : count_even_pascal 15 = 84 :=
  by
    sorry

end even_count_in_pascal_triangle_l485_485609


namespace complex_modulus_product_l485_485588

noncomputable def z1 : ℂ := 4 - 3 * Complex.I
noncomputable def z2 : ℂ := 4 + 3 * Complex.I

theorem complex_modulus_product : Complex.abs z1 * Complex.abs z2 = 25 := by 
  sorry

end complex_modulus_product_l485_485588


namespace interval_satisfies_ineq_l485_485019

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end interval_satisfies_ineq_l485_485019


namespace angles_congruence_mod_360_l485_485574

theorem angles_congruence_mod_360 (a b c d : ℤ) : 
  (a = 30) → (b = -30) → (c = 630) → (d = -630) →
  (b % 360 = 330 % 360) ∧ 
  (a % 360 ≠ 330 % 360) ∧ (c % 360 ≠ 330 % 360) ∧ (d % 360 ≠ 330 % 360) :=
by
  intros
  sorry

end angles_congruence_mod_360_l485_485574


namespace firefighter_target_heart_rate_l485_485553

theorem firefighter_target_heart_rate 
  (age : ℕ) -- Defining age as a natural number
  (adjusted_max_hr : ℕ := 225 - age) -- Definition of adjusted maximum heart rate
  (target_hr : ℚ := 0.88 * adjusted_max_hr) -- Definition of target heart rate as a rational number
  (rounded_target_hr : ℤ := Int.ofNat (target_hr.toNat)) -- Rounding to the nearest whole number
  : age = 35 → rounded_target_hr = 167 := 
by
  intros h_age_eq_35
  sorry -- Proof omitted.

end firefighter_target_heart_rate_l485_485553


namespace problem_prime_square_plus_two_l485_485318

theorem problem_prime_square_plus_two (P : ℕ) (hP_prime : Prime P) (hP2_plus_2_prime : Prime (P^2 + 2)) : P^4 + 1921 = 2002 :=
by
  sorry

end problem_prime_square_plus_two_l485_485318


namespace raisins_in_other_boxes_l485_485207

theorem raisins_in_other_boxes (total_raisins : ℕ) (raisins_box1 : ℕ) (raisins_box2 : ℕ) (other_boxes : ℕ) (num_other_boxes : ℕ) :
  total_raisins = 437 →
  raisins_box1 = 72 →
  raisins_box2 = 74 →
  num_other_boxes = 3 →
  other_boxes = (total_raisins - raisins_box1 - raisins_box2) / num_other_boxes →
  other_boxes = 97 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end raisins_in_other_boxes_l485_485207


namespace max_value_g_l485_485463

def g : ℕ → ℕ
| n => if n < 7 then n + 8 else g (n - 3)

theorem max_value_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 14 := by
  sorry

end max_value_g_l485_485463


namespace vector_perpendicular_l485_485259

theorem vector_perpendicular (k : ℝ) (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (2, k)) (h3 : (2 • a + b) ∙ a = 0) : k = -6 :=
begin
  sorry
end

end vector_perpendicular_l485_485259


namespace equilibrium_constant_comparison_l485_485206

theorem equilibrium_constant_comparison
  (K1 : ℝ) (K2 : ℝ)
  (H1 : K1 = 3.84 * 10^(-31))
  (H2 : K2 = 3.10 * 10^(25)) :
  K1 < K2 :=
by {
  rw [H1, H2],
  sorry
}

end equilibrium_constant_comparison_l485_485206


namespace no_power_of_two_l485_485818

theorem no_power_of_two (N : ℕ) (nums : list ℕ)
  (h_nums : nums = list.range' 11111 88889) (h_len : N = (nums.map (λ x, x * 10^(5 * x))).sum) :
  ¬ is_power_of_two N :=
by sorry

end no_power_of_two_l485_485818


namespace DVDs_on_fifth_rack_l485_485132

theorem DVDs_on_fifth_rack :
  ∀ (n : ℕ), (n = 1 → DVDs n = 2) ∧ (n = 2 → DVDs n = 4) ∧ (n = 3 → DVDs n = 8) ∧ (n = 4 → DVDs n = 16) ∧ (n = 6 → DVDs n = 64) ∧ (∀ n ≥ 1, DVDs (n + 1) = 2 * DVDs n) → DVDs 5 = 32 :=
by sorry

end DVDs_on_fifth_rack_l485_485132


namespace question_1_question_2_l485_485289

-- Define the function f(x) given k
def f (x k : ℝ) : ℝ := Real.log (4 ^ x + 1) / Real.log 4 + k * x

-- Statement for Question 1
theorem question_1 (x : ℝ) (h : f x 0 > 1 / 2) : x ∈ Set.Ioi 0 := by
  sorry

-- Statement for Question 2
theorem question_2 (hx : ∀ x : ℝ, f (-x) k = f x k) : k = -1 / 2 := by
  sorry

end question_1_question_2_l485_485289


namespace arrangements_with_AB_together_l485_485243

theorem arrangements_with_AB_together (students : Finset α) (A B : α) (hA : A ∈ students) (hB : B ∈ students) (h_students : students.card = 5) : 
  ∃ n, n = 48 :=
sorry

end arrangements_with_AB_together_l485_485243


namespace problem_1_problem_2_l485_485674

variable {a : ℕ → ℝ}
variable (n : ℕ)

-- Conditions of the problem
def seq_positive : ∀ (k : ℕ), a k > 0 := sorry
def a1 : a 1 = 1 := sorry
def recurrence (n : ℕ) : a (n + 1) = (a n + 1) / (12 * a n) := sorry

-- Proofs to be provided
theorem problem_1 : ∀ n : ℕ, a (2 * n + 1) < a (2 * n - 1) := 
by 
  apply sorry 

theorem problem_2 : ∀ n : ℕ, 1 / 6 ≤ a n ∧ a n ≤ 1 := 
by 
  apply sorry 

end problem_1_problem_2_l485_485674


namespace train_crosses_bridge_in_specific_time_l485_485524

noncomputable def train_crossing_time (train_length bridge_length : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := train_length + bridge_length
  total_distance / speed_mps

theorem train_crosses_bridge_in_specific_time :
  train_crossing_time 100 150 42 ≈ 21.42 :=
by
  sorry

end train_crosses_bridge_in_specific_time_l485_485524


namespace average_calculation_l485_485837

def average_two (a b : ℚ) : ℚ := (a + b) / 2
def average_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem average_calculation :
  average_three (average_three 2 2 0) (average_two 1 2) 1 = 23 / 18 :=
by sorry

end average_calculation_l485_485837


namespace circle_tangent_to_line_l485_485155

theorem circle_tangent_to_line :
  ∃ r : ℝ, ∃ (h k : ℝ), (h = 2) ∧ (k = -1) ∧
  (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2) ∧
  (∃ a b c : ℝ, a = 3 ∧ b = -4 ∧ c = 5 ∧
      abs (a * h + b * k + c) = r * sqrt (a^2 + b^2)) ∧
      ((x - 2)^2 + (y + 1)^2 = 9) :=
begin
  sorry
end

end circle_tangent_to_line_l485_485155


namespace angle_ABC_bisector_l485_485000

theorem angle_ABC_bisector (θ : ℝ) (h : θ / 2 = (1 / 3) * (180 - θ)) : θ = 72 :=
by
  sorry

end angle_ABC_bisector_l485_485000


namespace even_count_in_top_15_rows_l485_485627

-- We want to count the number of even binomial coefficients in the first 15 rows of Pascal's Triangle.
-- Specifically, we are considering rows 0 to 14.

def is_even (n : ℕ) := n % 2 = 0

def count_even_binomials_in_pascals_triangle_up_to_row (max_row : ℕ) : ℕ := 
  ∑ n in Finset.range (max_row + 1), 
    ∑ k in Finset.range (n + 1), 
      if is_even (Nat.choose n k) then 1 else 0

theorem even_count_in_top_15_rows : count_even_binomials_in_pascals_triangle_up_to_row 14 = 49 := 
by 
  sorry

end even_count_in_top_15_rows_l485_485627


namespace recommendation_plan_count_l485_485228

-- Definitions for the given conditions
def num_males := 3
def num_females := 2
def spots_russian := 2
def spots_japanese := 2
def spots_spanish := 1

-- Theorem statement proving the number of different recommendation plans
theorem recommendation_plan_count :
  let total_recommendation_plans :=
    (num_males * num_females * (num_males - 1) * (num_females - 1) * 1 +
    ((num_males * (num_males - 1)) / 2 * 1 * num_females * 1)) * 2 in
  total_recommendation_plans = 36 :=
by
  sorry

end recommendation_plan_count_l485_485228


namespace green_pill_cost_l485_485186

theorem green_pill_cost : 
  ∃ (g p : ℝ), (14 * (g + p) = 546) ∧ (g = p + 1) ∧ (g = 20) :=
by
  let g := 20
  let p := g - 1
  have h1 : 14 * (g + p) = 546 := sorry -- 14 days total cost
  have h2 : g = p + 1 := by simp [p] -- cost difference
  exact ⟨g, p, h1, h2, rfl⟩

end green_pill_cost_l485_485186


namespace reduce_gas_consumption_to_maintain_expenditure_l485_485528

theorem reduce_gas_consumption_to_maintain_expenditure 
  (P C : ℝ) 
  (hP : 0 < P) (hC : 0 < C) :
  let P1 := 1.30 * P,
      P2 := 1.56 * P,
      C_new := C / 1.56,
      percentage_reduction := (1 - (1 / 1.56)) * 100 in
  percentage_reduction = 35.897 := 
sorry

end reduce_gas_consumption_to_maintain_expenditure_l485_485528


namespace maximum_distinct_numbers_l485_485710

theorem maximum_distinct_numbers (n : ℕ) (hsum : n = 250) : 
  ∃ k ≤ 21, k = 21 :=
by
  sorry

end maximum_distinct_numbers_l485_485710


namespace smallest_x_value_l485_485492

theorem smallest_x_value (x : ℤ) (h : 3 * x^2 - 4 < 20) : x = -2 :=
sorry

end smallest_x_value_l485_485492


namespace arithmetic_sequence_sum_l485_485751

noncomputable theory
open_locale big_operators

variables {a : ℕ → ℝ}

theorem arithmetic_sequence_sum :
  (∃ a : ℕ → ℝ, ∃ d : ℝ, a 1 + a 2011 = 10 ∧ (∀ n : ℕ, a (n + 1) = a n + d)) →
  a 2 + a 1006 + a 2010 = 15 :=
begin
  sorry
end

end arithmetic_sequence_sum_l485_485751


namespace infinite_representations_of_one_l485_485831

def has_infinite_representations : Prop :=
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, 5 < a n) ∧ (StrictMono a) ∧ (∀ n : ℕ, ∃ k, ∑ i in Finset.range k, 1 / (5 : ℝ) + 1 / (a i : ℝ) = 1)

theorem infinite_representations_of_one : has_infinite_representations :=
by
  sorry

end infinite_representations_of_one_l485_485831


namespace annalise_spending_l485_485581

theorem annalise_spending
  (n_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (cost_per_tissue : ℝ)
  (h1 : n_boxes = 10)
  (h2 : packs_per_box = 20)
  (h3 : tissues_per_pack = 100)
  (h4 : cost_per_tissue = 0.05) :
  n_boxes * packs_per_box * tissues_per_pack * cost_per_tissue = 1000 := 
  by
  sorry

end annalise_spending_l485_485581


namespace zero_point_in_interval_l485_485466

noncomputable def f (x : ℝ) : ℝ := x + Real.exp x

theorem zero_point_in_interval : ∃ x ∈ set.Icc (-1 : ℝ) (-1/2), f x = 0 :=
by {
  -- proof to be provided
  sorry
}

end zero_point_in_interval_l485_485466


namespace cone_cylinder_ratio_l485_485598

noncomputable def ratio_of_heights (h H: ℝ) (r: ℝ) : Prop :=
(π * r^2 * h / 2 / 4 / 3) = (1/3 * π * r^2 * H / 2)

theorem cone_cylinder_ratio (h H: ℝ) (r: ℝ) :
    (π * r^2 * h / 2 / 4 / 3) = (1/3 * π * r^2 * H / 2) → 
    H / h = real.cbrt 2 :=
begin
    sorry
end

end cone_cylinder_ratio_l485_485598


namespace football_team_total_players_l485_485888

/-- Let's denote the total number of players on the football team as P.
    We know that there are 31 throwers, and all of them are right-handed.
    The rest of the team is divided so one third are left-handed and the rest are right-handed.
    There are a total of 57 right-handed players on the team.
    Prove that the total number of players on the football team is 70. -/
theorem football_team_total_players 
  (P : ℕ) -- total number of players
  (T : ℕ := 31) -- number of throwers
  (L : ℕ) -- number of left-handed players
  (R : ℕ := 57) -- total number of right-handed players
  (H_all_throwers_rhs: ∀ x : ℕ, (x < P) → (x < T) → (x = T → x < R)) -- all throwers are right-handed
  (H_rest_division: ∀ x : ℕ, (x < P - T) → (x = L) → (x = 2 * L))
  : P = 70 :=
  sorry

end football_team_total_players_l485_485888


namespace negation_exists_ln_le_x_add_one_l485_485467

theorem negation_exists_ln_le_x_add_one :
  (¬ ∃ x : ℝ, x > 0 ∧ ln x ≤ x + 1) ↔ (∀ x : ℝ, x > 0 → ln x > x + 1) :=
by
  sorry

end negation_exists_ln_le_x_add_one_l485_485467


namespace gcd_of_45_and_75_l485_485085

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l485_485085


namespace laptop_total_reduction_l485_485159

theorem laptop_total_reduction (P : ℝ) (hP : P > 0) : 
  let final_price := 0.5 * (0.7 * P) in
  (P - final_price) / P * 100 = 65 := 
by 
  let initial_reduction := 0.7 * P;
  let second_reduction := 0.5 * initial_reduction;
  let reduction := P - second_reduction; 
  calc
    (P - final_price) / P * 100
      = (P - (0.5 * (0.7 * P))) / P * 100 : by rfl
  ... = 0.65 * 100 : by ring
  ... = 65 : by norm_num

end laptop_total_reduction_l485_485159


namespace pure_gala_trees_l485_485520

theorem pure_gala_trees (T F G : ℝ) (h1 : F + 0.10 * T = 221)
  (h2 : F = 0.75 * T) : G = T - F - 0.10 * T := 
by 
  -- We define G and show it equals 39
  have eq : T = F / 0.75 := by sorry
  have G_eq : G = T - F - 0.10 * T := by sorry 
  exact G_eq

end pure_gala_trees_l485_485520


namespace g_22_minus_g_4_l485_485801

open Function

variable {R : Type} [LinearOrderedField R]

-- Given conditions
def g (x : R) : R := sorry  -- g is a linear function
axiom h₁ : g(10) - g(4) = 18

-- Proof statement
theorem g_22_minus_g_4 : g(22) - g(4) = 54 :=
sorry

end g_22_minus_g_4_l485_485801


namespace derivative_of_y_l485_485645

-- Define the function y
def y (x : ℝ) : ℝ := (1 + cos (2 * x))^2

-- State the theorem to prove
theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = -4 * sin (2 * x) - 2 * sin (4 * x) :=
by
  sorry 

end derivative_of_y_l485_485645


namespace solve_z_solve_m_l485_485201

noncomputable def complex_number (a b: ℝ) (z: ℂ) := a + b * complex.I = z

theorem solve_z (a b: ℝ) (z: ℂ) (h1: |z| = sqrt 10) (h2: (1 + 2 * complex.I) * z = (a - 2 * b) + (2 * a + b) * complex.I)
  (h3: a > 0 ∧ complex_number a b z) : z = 3 - complex.I :=
by {
  sorry
}

theorem solve_m (a: ℝ) (b: ℝ) (z: ℂ) (h1: |z| = sqrt 10) (h2: (1 + 2 * complex.I) * z = (a - 2 * b) + (2 * a + b) * complex.I)
  (h3: a > 0 ∧ complex_number a b z) (h4: z = 3 - complex.I) (m: ℝ) (h5: (conjugate z) + ((m - complex.I) / (1 + complex.I)) = 0 + 0 * complex.I)  
  : m = -5 :=
by {
  sorry
}

end solve_z_solve_m_l485_485201


namespace increase_by_percentage_l485_485499

-- Define the initial number.
def initial_number : ℝ := 75

-- Define the percentage increase as a decimal.
def percentage_increase : ℝ := 1.5

-- Define the expected final result after applying the increase.
def expected_result : ℝ := 187.5

-- The proof statement.
theorem increase_by_percentage : initial_number * (1 + percentage_increase) = expected_result :=
by
  sorry

end increase_by_percentage_l485_485499


namespace floor_1000_cos_B_eq_neg950_l485_485345

theorem floor_1000_cos_B_eq_neg950
  (A B C D : Type)
  (AB BC AD CD : ℝ)
  (∠_B ∠_D : ℝ)
  (cong_angles : ∠_B = ∠_D)
  (AB_eq_BC : AB = 200)
  (AD_ne_CD : AD ≠ CD)
  (perimeter_eq_780 : AB + BC + AD + CD = 780):
    floor (1000 * Real.cos ∠_B) = -950 := sorry

end floor_1000_cos_B_eq_neg950_l485_485345


namespace gcd_45_75_l485_485110

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485110


namespace incorrect_propositions_are_prop1_and_prop3_l485_485889

-- Define the propositions as booleans.
def prop1 : Prop := ∃ (a b c : Point), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ Plane.contains a ∧ Plane.contains b ∧ Plane.contains c
def prop2 : Prop := Rectangle.is_plane_figure
def prop3 : Prop := ∃ (l1 l2 l3 : Line), l1 ≠ l2 ∧ l2 ≠ l3 ∧ l3 ≠ l1 ∧ Plane.contains l1 ∧ Plane.contains l2 ∧ Plane.contains l3
def prop4 : Prop := ∃ (p1 p2 : Plane), p1 ≠ p2 ∧ Space.region_count p1 p2 = 4

-- Theorem stating that the incorrect propositions are exactly prop1 and prop3.
theorem incorrect_propositions_are_prop1_and_prop3 :
  ¬(prop1) ∧ ¬(prop3) ∧ prop2 ∧ prop4 :=
by sorry

end incorrect_propositions_are_prop1_and_prop3_l485_485889


namespace arrangements_and_combinations_l485_485532

theorem arrangements_and_combinations : 
  (∑ perm 2 of 7) + ((∑ comb 3 of 7) + (∑ comb 4 of 7)) = 112 := sorry

end arrangements_and_combinations_l485_485532


namespace find_coefficients_l485_485873

noncomputable def polynomial_h (x : ℚ) : ℚ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def polynomial_j (b c d x : ℚ) : ℚ := x^3 + b * x^2 + c * x + d

theorem find_coefficients :
  (∃ b c d : ℚ,
     (∀ s : ℚ, polynomial_h s = 0 → polynomial_j b c d (s^3) = 0) ∧
     (b, c, d) = (6, 12, 8)) :=
sorry

end find_coefficients_l485_485873


namespace gcd_45_75_l485_485072

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485072


namespace exist_valid_board_l485_485967

def board : Type := array (2 × 25) (char) -- Each cell can be either 'X' or 'O'

def valid_X_neighbor (b : board) (i j : nat) (h : b[i][j] = 'X') : bool := sorry
-- Function to check if an 'X' has exactly one neighboring 'X'

def valid_O_neighbor (b : board) (i j : nat) (h : b[i][j] = 'O') : bool := sorry
-- Function to check if an 'O' has exactly two neighboring 'O's

def valid_board (b : board) : bool :=
  (∀ i < 2, ∀ j < 25, if b[i][j] = 'X' then valid_X_neighbor b i j else valid_O_neighbor b i j)

theorem exist_valid_board : ∃ b : board, valid_board b :=
begin
  sorry
end


end exist_valid_board_l485_485967


namespace intersection_of_P_and_Q_l485_485305

def P : Set ℝ := { x | 1 ≤ 2^x ∧ 2^x < 4 }
def Q : Set ℕ := {1, 2, 3}

theorem intersection_of_P_and_Q : P ∩ Q = {1} :=
by
  sorry

end intersection_of_P_and_Q_l485_485305


namespace tangent_lines_through_P_line_l_through_P_with_length_AB_trajectory_of_Q_l485_485288

-- Definitions and conditions
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4
def point_P (x y : ℝ) : Prop := x = 1 ∧ y = 2
def segment_length (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statements
theorem tangent_lines_through_P (x y : ℝ) (P : point_P x y) :
  (∃ k : ℝ, y = k * (x - 1) + 2 ∧ circle_C x y ∧ ∃ k, k = 0 ∨ k = -4/3) ∨ y = 2 ∨ (4*x + 3*y - 10 = 0) :=
sorry

theorem line_l_through_P_with_length_AB (x y : ℝ) (P : point_P x y) (A B : ℝ × ℝ) : 
  segment_length A B = 2 * real.sqrt 3 → (3*x - 4*y + 5 = 0) ∨ (x = 1) :=
sorry

theorem trajectory_of_Q (x y x0 y0 : ℝ) : 
  circle_C x0 y0 → (∃ M N Q: ℝ × ℝ, M = (x0, y0) ∧ N = (0, y0) ∧ Q = ((x0 + 0) / 2, (y0 + y0) / 2)) → 
  (x^2 + (y/2)^2 = 1) :=
sorry

end tangent_lines_through_P_line_l_through_P_with_length_AB_trajectory_of_Q_l485_485288


namespace piecewise_function_range_l485_485857

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else x^(1/2)

theorem piecewise_function_range :
  {x : ℝ | f x > 1} = {x : ℝ | x < -1 ∨ x > 1} :=
by
  sorry

end piecewise_function_range_l485_485857


namespace first_expression_result_eq_2_second_expression_result_eq_1_l485_485591

-- Define constants for the first problem
def pi := Real.pi
def e := Real.exp 1
def sqrt := Real.sqrt
def lg := λ x : ℝ, Real.log10 x

-- Proof problems
theorem first_expression_result_eq_2 : 
  sqrt (25 / 9) - (8 / 27)^(1 / 3) - (pi + e)^0 + (1 / 4)^(-1 / 2) = 2 := 
by 
  sorry

theorem second_expression_result_eq_1 : 
  (lg 2)^2 + lg 2 * lg 5 + sqrt ((lg 2)^2 - lg 4 + 1) = 1 := 
by 
  sorry

end first_expression_result_eq_2_second_expression_result_eq_1_l485_485591


namespace complex_expression_power_48_l485_485979

open Complex

noncomputable def complex_expression := (1 + I) / Real.sqrt 2

theorem complex_expression_power_48 : complex_expression ^ 48 = 1 := by
  sorry

end complex_expression_power_48_l485_485979


namespace solution_set_eq_l485_485461

noncomputable def f : ℝ → ℝ := sorry
def f_prime : ℝ → ℝ := sorry

theorem solution_set_eq : 
  (∀ x : ℝ, f x ∈ ℝ) ∧ f (-2) = 2 ∧ (∀ x : ℝ, f_prime x > 2) → 
  {x : ℝ | f x > 2 * x + 6} = Ioo (-2) +∞ :=
begin
  sorry
end

end solution_set_eq_l485_485461


namespace gcd_45_75_l485_485087

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485087


namespace hexagon_equilateral_triangle_l485_485758

theorem hexagon_equilateral_triangle
  (angleA : ℝ)
  (sideAB : ℝ) (sideCD : ℝ) (sideEF : ℝ)
  (sideBC : ℝ) (sideDE : ℝ) (sideFA : ℝ)
  (D H E G : ℝ × ℝ)
  (DH_length : ℝ)
  (AD_x : ℝ) (AD_y : ℝ)
  (CF_x : ℝ) (CF_y : ℝ) :
  -- Condition: All angles in the hexagon are 120 degrees
  angleA = 120 ∧
  -- Condition: Side lengths AB = CD = EF = 3 and BC = DE = FA = 2
  sideAB = 3 ∧ sideCD = 3 ∧ sideEF = 3 ∧ 
  sideBC = 2 ∧ sideDE = 2 ∧ sideFA = 2 ∧
  -- Coordinates of point E (origin)
  E = (0, 0) ∧
  -- Coordinates of point D
  D = (2, 0) ∧
  -- Coordinates of point C based on given calculations
  (let C := (7 / 2, 3 * Real.sqrt 3 / 2) in 
  -- Coordinates of point H given DH = 1
  H = (5 / 2, Real.sqrt 3 / 2) ∧
  -- Coordinates of intersection point G (found from intersection of AD and CF)
  G = (1 / 2, 3 * Real.sqrt 3 / 2) ∧
  -- Length from D to H
  DH_length = 1) →
  -- Conclusion: Triangle EGH is equilateral
  (Real.dist E G = Real.dist G H ∧ Real.dist G H = Real.dist H E := by sorry)

end hexagon_equilateral_triangle_l485_485758


namespace rons_chocolate_cost_l485_485538

theorem rons_chocolate_cost :
  let cost_per_bar := 1.5
  let sections_per_bar := 3
  let scouts := 15
  let smores_per_scout := 2
  let total_smores := scouts * smores_per_scout
  let bars_needed := total_smores / sections_per_bar in
  bars_needed * cost_per_bar = 15.0 := by
  sorry

end rons_chocolate_cost_l485_485538


namespace system_of_inequalities_l485_485011

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end system_of_inequalities_l485_485011


namespace circles_common_tangents_l485_485894

noncomputable def circle (r: ℝ) := {x : ℝ × ℝ // x.1^2 + x.2^2 = r^2}

def common_tangents (c1 c2 : circle ℝ) : ℕ :=
  -- definition to determine the number of common tangents
  sorry

theorem circles_common_tangents :
  ∀ (c1 c2 : circle ℝ),
    (c1 = circle 3) → (c2 = circle 5) →
    (common_tangents c1 c2 ≠ 3) :=
by {
  intros c1 c2 hc1 hc2,
  sorry
}

end circles_common_tangents_l485_485894


namespace math_proof_problem1_math_proof_problem2_l485_485810

noncomputable def problem1 : Prop :=
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 3
  let C : ℝ := 5 * Real.pi / 6
  let area : ℝ := Real.sqrt 3 / 2
  let c : ℝ := Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)
  c = Real.sqrt 13

noncomputable def problem2 : Prop :=
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 3
  let c : ℝ := 2 * Real.sin (60 * Real.pi / 180)
  let B : ℝ := Real.pi / 3
  let f : ℝ → ℝ := λ C : ℝ, 2 * c - a
  ∀ (C : ℝ), C ∈ Ioo 0 (2 * Real.pi / 3) → f C ∈ Ioo (-Real.sqrt 3) (2 * Real.sqrt 3)

theorem math_proof_problem1 : problem1 :=
by sorry

theorem math_proof_problem2 : problem2 :=
by sorry

end math_proof_problem1_math_proof_problem2_l485_485810


namespace interval_satisfies_ineq_l485_485018

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end interval_satisfies_ineq_l485_485018


namespace largest_prime_divisor_of_Q_l485_485874

open Nat

theorem largest_prime_divisor_of_Q :
  let primes := {p | p ∈ {11, 13, 17, 19}}
  let Q := primes.sum 
  let prime_factors := {p | p ∈ ({2, 3, 5} : Set ℕ)}
  Q = 60 ∧ ∀ p ∈ prime_factors, is_prime p ∧ p ∣ Q → 5 ≤ p :=
by
  let primes := {11, 13, 17, 19}
  let Q := 11 + 13 + 17 + 19
  let prime_factors := {2, 3, 5}
  have hQ : Q = 60 := by 
    norm_num [Q]
    sorry
  have h : ∀ p ∈ prime_factors, is_prime p ∧ p ∣ Q → 5 ≤ p := by
    sorry
  exact ⟨hQ, h⟩

end largest_prime_divisor_of_Q_l485_485874


namespace negative_value_among_options_l485_485226

theorem negative_value_among_options :
  (sin 1100 * π / 180 > 0) ∧
  (cos (-2200) * π / 180 > 0) ∧
  (tan (-10) < 0) ∧
  (sin (7 * π / 10) * cos π * tan (17 * π / 9) > 0) :=
by
  sorry

end negative_value_among_options_l485_485226


namespace store_loss_l485_485568

theorem store_loss (x y : ℝ) (hx : x + 0.25 * x = 135) (hy : y - 0.25 * y = 135) : 
  (135 * 2) - (x + y) = -18 := 
by
  sorry

end store_loss_l485_485568


namespace count_valid_permutations_between_100_and_999_l485_485315

def is_multiple_of_9 (n : ℕ) : Prop :=
  9 ∣ n

def is_between_100_and_999 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def valid_permutations (n : ℕ) (m : ℕ) : Prop :=
  let digits := n.digits 10 in
  let perm := m.digits 10 in
  perm.all (λ d, d ≠ 0) ∧ List.perm digits perm ∧ is_between_100_and_999 m

theorem count_valid_permutations_between_100_and_999 :
  ∑ k in (Finset.filter (λ n, is_multiple_of_9 n ∧ is_between_100_and_999 n) (Finset.range 1000)), 
    ∑ m in (Finset.filter (λ m, valid_permutations k m) (Finset.range 1000)), 1 = 390 :=
by
  sorry

end count_valid_permutations_between_100_and_999_l485_485315


namespace rope_length_l485_485482

theorem rope_length (d : ℕ) (L : ℕ) : d = 30 → L = 360 :=
by
  assume h : d = 30
  have hL : L = 4 * d := by linarith
  rw h at hL
  rw mul_comm at hL
  exact hL
  sorry

end rope_length_l485_485482


namespace triangle_area_l485_485584

theorem triangle_area (points : Finset Point) (H : points.card = 21)
    (area_eq_1 : ∀ (A B C : Point), A ∈ points → B ∈ points → C ∈ points → equilateral_triangle A B C → triangle_area A B C = 1) :
    ∃ A B C : Point, A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ triangle_area A B C = 13 :=
by
  sorry

end triangle_area_l485_485584


namespace area_of_triangle_DEF_is_30_l485_485559

noncomputable def point_in_triangle (P A B C : Type) [P A B C] : Prop := sorry -- Placeholder for a point in triangle definition
noncomputable def lines_parallel (P Q A B C : Type) [P Q A B C] : Prop := sorry -- Placeholder for lines parallel definition
noncomputable def area_t (u : Type) : ℝ := sorry -- Placeholder for area function definition

variables (DEF Q : Type) [Point DEF] [Point Q]
variables (u1 u2 u3 : Type) [Triangle u1] [Triangle u2] [Triangle u3]
variables (A Q_in_DEF : point_in_triangle Q DEF DEF DEF)
variables (P1 P2 P3 : point_in_triangle P1 DEF DEF DEF)
variables (u1_area : area_t u1 = 3)
variables (u2_area : area_t u2 = 12)
variables (u3_area : area_t u3 = 15)

theorem area_of_triangle_DEF_is_30 : area_t DEF = 30 := 
sorry

end area_of_triangle_DEF_is_30_l485_485559


namespace remainder_when_xy_div_by_22_l485_485811

theorem remainder_when_xy_div_by_22
  (x y : ℤ)
  (h1 : x % 126 = 37)
  (h2 : y % 176 = 46) : 
  (x + y) % 22 = 21 := by
  sorry

end remainder_when_xy_div_by_22_l485_485811


namespace gcd_45_75_l485_485057

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485057


namespace sin_order_equilateral_triangle_l485_485330

variable {A B C : ℝ}
variable {a b c : ℝ}

theorem sin_order (h1 : A > B > C) : sin A > sin B > sin C := sorry

theorem equilateral_triangle 
  (h2 : a / cos (A / 2) = b / cos (B / 2) = c / cos (C / 2)) 
  : A = B ∧ B = C := sorry

end sin_order_equilateral_triangle_l485_485330


namespace min_value_a_l485_485735

theorem min_value_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_value_a_l485_485735


namespace area_of_figured_formed_by_func_l485_485840

-- Definitions of the conditions as hypotheses.
def func (x : ℝ) := 3 * x^2
def x1 : ℝ := 1
def x2 : ℝ := 2

-- The statement to prove: the area under the curve formed by the given conditions is 7.
theorem area_of_figured_formed_by_func : (∫ x in x1..x2, func x) = 7 :=
by
sorend

end area_of_figured_formed_by_func_l485_485840


namespace total_cups_l485_485480

theorem total_cups (cups_of_juice cups_of_milk : ℕ) (h1 : cups_of_juice = 3) (h2 : cups_of_milk = 4) : cups_of_juice + cups_of_milk = 7 :=
by 
  rw [h1, h2]
  rfl

end total_cups_l485_485480


namespace prob_no_adjacent_same_roll_l485_485244

-- Definition of the problem conditions
def num_people : ℕ := 5
def num_sides_die : ℕ := 6

-- Probability that no two adjacent people roll the same number
def prob_no_adj_same : ℚ :=
  1 * (5/6) * (5/6) * (5/6) * (5/6)

-- Proof statement
theorem prob_no_adjacent_same_roll : prob_no_adj_same = 625 / 1296 := by
  sorry

end prob_no_adjacent_same_roll_l485_485244


namespace find_n_with_divisors_conditions_l485_485866

theorem find_n_with_divisors_conditions :
  ∃ n : ℕ, 
    (∀ d : ℕ, d ∣ n → d ∈ [1, n] ∧ 
    (∃ a b c : ℕ, a = 1 ∧ b = d / a ∧ c = d / b ∧ b = 7 * a ∧ d = 10 + b)) →
    n = 2891 :=
by
  sorry

end find_n_with_divisors_conditions_l485_485866


namespace transform_correct_l485_485042

theorem transform_correct :
  ∀ (x : ℝ),
    (∃ f : ℝ → ℝ, (f x) = (sqrt 2) * cos (x)) →
    ((∃ g : ℝ → ℝ, (g x) = (sqrt 2) * sin (2 * x + (π/4))) →
      ∀ x, (sqrt 2) * cos (x) = (sqrt 2) * cos ((2 * x) - (π / 4))) :=
begin
  intros x f hf g hg,
  sorry
end

end transform_correct_l485_485042


namespace fare_expression_fare_for_19_4_km_distance_for_fare_19_8_l485_485028

variable (x : ℝ)

-- Condition for the mileage fare relationship
def taxi_fare (x : ℝ) : ℝ := 
  if x <= 3 then 9 
  else 1.2 * x + 5.4

-- Proof problem 1: Algebraic expression for the taxi fare when x > 3
theorem fare_expression (x : ℝ) (h : x > 3) : taxi_fare x = 1.2 * x + 5.4 := by
  sorry

-- Proof problem 2: Fare for 19.4 kilometers
theorem fare_for_19_4_km : taxi_fare 19.4 = 29.4 := by
  sorry

-- Proof problem 3: Distance traveled for a fare of 19.8 yuan
theorem distance_for_fare_19_8 (d : ℝ) (h : taxi_fare d = 19.8) : d = 12 := by
  sorry

end fare_expression_fare_for_19_4_km_distance_for_fare_19_8_l485_485028


namespace match_schemes_count_l485_485556

theorem match_schemes_count : 
  ∀ (teachers students : ℕ), teachers = 2 → students = 4 →
  (∃ count : ℕ, count = 12) :=
by
  intros teachers students h_teachers h_students
  use 12
  sorry

end match_schemes_count_l485_485556


namespace Ron_spends_15_dollars_l485_485542

theorem Ron_spends_15_dollars (cost_per_bar : ℝ) (sections_per_bar : ℕ) (num_scouts : ℕ) (s'mores_per_scout : ℕ) :
  cost_per_bar = 1.50 ∧ sections_per_bar = 3 ∧ num_scouts = 15 ∧ s'mores_per_scout = 2 →
  cost_per_bar * (num_scouts * s'mores_per_scout / sections_per_bar) = 15 :=
by
  sorry

end Ron_spends_15_dollars_l485_485542


namespace f_continuous_example_continuous_f_b_discontinuous_l485_485916

section part_a

variables {R : Type*} [TopologicalSpace R] [AddGroup R]

def f (x : R) : R := sorry -- f is our function to prove continuity.

def g (x : R) : R := f(x) + f(2 * x)

def h (x : R) : R := f(x) + f(4 * x)

-- Assume g and h are continuous
axiom g_cont : Continuous g
axiom h_cont : Continuous h

-- We need to prove f is continuous.
theorem f_continuous : Continuous f := sorry

end part_a

section part_b

variables {R : Type*} [TopologicalSpace R] [AddGroup R]

-- Define the discontinuous function example
def f_b (x : R) : R := if x = 0 then 0 else x / |x|

-- Define the interval I
def I : Set R := { x | x < 0 }

-- Define g_a given a in I
def g_a (a : R) (x : R) : R := f_b(x) + f_b(a * x)

-- For all a in I, g_a is continuous
theorem example_continuous (a : R) (ha : a ∈ I) : Continuous (g_a a) := 
sorry

-- Prove that f_b is discontinuous, by definition.
theorem f_b_discontinuous : ¬Continuous f_b := sorry

end part_b

end f_continuous_example_continuous_f_b_discontinuous_l485_485916


namespace scientific_notation_2150_l485_485514

theorem scientific_notation_2150 : ∃ (a : ℝ) (n : ℤ), (1 ≤ a ∧ a < 10) ∧ 2150 = a * 10^n ∧ a = 2.15 ∧ n = 3 :=
by
  use 2.15
  use 3
  split
  · split
    · norm_num
  · split
    · norm_num
  · norm_num
  sorry

end scientific_notation_2150_l485_485514


namespace find_y_in_rectangle_l485_485945

theorem find_y_in_rectangle : 
  ∃ y : ℝ, y > 0 ∧ (let length := 8 in 
                    let height := y - 2 in 
                    length * height = 64) ∧ y = 10 :=
sorry

end find_y_in_rectangle_l485_485945


namespace suraj_average_after_10_innings_l485_485921

theorem suraj_average_after_10_innings (A : ℝ) (h : A = 120)
  (total_runs_first_9_innings : 9 * A)
  (score_in_10th_innings : 200) 
  (new_average_increase : (10 * (A + 8) = (9 * A + 200))) :
  (A + 8 = 128) :=
by
  sorry

end suraj_average_after_10_innings_l485_485921


namespace find_ten_x_l485_485788

theorem find_ten_x (x : ℝ) 
  (h : 4^(2*x) + 2^(-x) + 1 = (129 + 8 * Real.sqrt 2) * (4^x + 2^(- x) - 2^x)) : 
  10 * x = 35 := 
sorry

end find_ten_x_l485_485788


namespace kataleya_paid_correct_amount_l485_485957

-- Definitions based on the conditions
def cost_per_peach : ℝ := 0.40 -- dollars
def number_of_peaches : ℕ := 400
def discount_per_10_dollars : ℝ := 2 -- dollars
def threshold_purchase_amount : ℝ := 10 -- dollars

-- Calculation based on the problem statement
def total_cost : ℝ := number_of_peaches * cost_per_peach
def total_10_dollar_purchases : ℕ := total_cost / threshold_purchase_amount
def total_discount : ℝ := (total_10_dollar_purchases : ℝ) * discount_per_10_dollars
def final_amount_paid : ℝ := total_cost - total_discount

-- Statement to prove
theorem kataleya_paid_correct_amount : final_amount_paid = 128 := 
by
  sorry

end kataleya_paid_correct_amount_l485_485957


namespace find_m_l485_485285

theorem find_m (S : ℕ → ℝ) (a : ℕ → ℝ) (m : ℝ) (hS : ∀ n, S n = m * 2^(n-1) - 3) 
               (ha1 : a 1 = S 1) (han : ∀ n > 1, a n = S n - S (n - 1)) 
               (ratio : ∀ n > 1, a (n+1) / a n = 1/2): 
  m = 6 := 
sorry

end find_m_l485_485285


namespace green_pill_cost_l485_485187

theorem green_pill_cost : 
  ∃ (g p : ℝ), (14 * (g + p) = 546) ∧ (g = p + 1) ∧ (g = 20) :=
by
  let g := 20
  let p := g - 1
  have h1 : 14 * (g + p) = 546 := sorry -- 14 days total cost
  have h2 : g = p + 1 := by simp [p] -- cost difference
  exact ⟨g, p, h1, h2, rfl⟩

end green_pill_cost_l485_485187


namespace constant_term_is_minus_160_l485_485261

noncomputable def integral_a : ℝ :=
  ∫ x in -1..1, real.sqrt (1 - x^2)

theorem constant_term_is_minus_160 :
  (∫ x in -1..1, real.sqrt (1 - x^2) + 2 - (real.pi / 2)) = real.pi / 2 →
  let expr := λ x : ℝ, (integral_a + 2 - real.pi / 2) * x - (1 / x) in
  (expr x)^6 = (2 * x - 1 / x)^6 →
  ∃ c : ℝ, (6 : ℕ) = 2 * (3 : ℕ) →
  (integral_a + 2 - real.pi / 2) = real.pi / 2 → c = -160 :=
begin
  sorry
end

end constant_term_is_minus_160_l485_485261


namespace min_distance_midpoint_y_axis_l485_485003

noncomputable def parabola_eq := λ (y : ℝ), y^2

def length_AB {A B : ℝ × ℝ} : Prop := (A.1 - B.1)^2 + (A.2 - B.2)^2 = 9

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem min_distance_midpoint_y_axis {A B : ℝ × ℝ}
  (hA1 : parabola_eq A.2 = A.1)
  (hB1 : parabola_eq B.2 = B.1)
  (hAB : length_AB) :
  let M := midpoint A B in
  abs(M.1) = 5 / 4 :=
sorry

end min_distance_midpoint_y_axis_l485_485003


namespace ratio_of_segments_l485_485804

noncomputable def points_on_sides_of_triangle (A B C P Q R : Point) : Prop :=
  (P ∈ Line B C) ∧ (Q ∈ Line C A) ∧ (R ∈ Line A B)

noncomputable def circumcircle_of_triangle (A B C : Point) : Circle :=
  -- Assume a function that constructs a circumcircle of triangle ABC
  circumcircle_of_triangle_ABC A B C

noncomputable def line_intersect_circle (AP : Line) (circle : Circle) : Point :=
  -- Assume intersection point, it should be further specified
  point_of_intersection AP circle

theorem ratio_of_segments (A B C P Q R X Y Z : Point)
  (h1 : points_on_sides_of_triangle A B C P Q R)
  (h2 : let Gamma_A := circumcircle_of_triangle A Q R
           Gamma_B := circumcircle_of_triangle B R P
           Gamma_C := circumcircle_of_triangle C P Q in
         X = line_intersect_circle (Line.mk A P) Gamma_A ∧
         Y = line_intersect_circle (Line.mk A P) Gamma_B ∧
         Z = line_intersect_circle (Line.mk A P) Gamma_C)
  : (dist Y X / dist X Z = dist B P / dist P C) :=
sorry

end ratio_of_segments_l485_485804


namespace max_value_x_plus_2y_l485_485311

theorem max_value_x_plus_2y (x y : ℝ) (h : |x| + |y| ≤ 1) : x + 2 * y ≤ 2 :=
sorry

end max_value_x_plus_2y_l485_485311


namespace crossing_time_l485_485544

-- Declarations and conditions
def length_first_train := 290.0 -- in meters
def speed_first_train := 120.0 * (1000 / 3600) -- converted to m/s
def length_second_train := 210.04 -- in meters
def speed_second_train := 80.0 * (1000 / 3600) -- converted to m/s

-- Relative speed when trains run in opposite directions
def relative_speed := speed_first_train + speed_second_train

-- Total length covered when trains cross each other
def total_length := length_first_train + length_second_train

-- Time taken for the trains to cross each other
def time_to_cross := total_length / relative_speed

-- Statement to prove
theorem crossing_time : time_to_cross ≈ 9 :=
  by
    sorry

end crossing_time_l485_485544


namespace increase_75_by_150_percent_l485_485502

noncomputable def original_number : Real := 75
noncomputable def percentage_increase : Real := 1.5
noncomputable def increase_amount : Real := original_number * percentage_increase
noncomputable def result : Real := original_number + increase_amount

theorem increase_75_by_150_percent : result = 187.5 := by
  sorry

end increase_75_by_150_percent_l485_485502


namespace gcd_of_45_and_75_l485_485077

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l485_485077


namespace periodic_odd_function_l485_485278

noncomputable def f (x : ℝ) : ℝ := 
if 2 ≤ x ∧ x ≤ 3 then log x / log 2 - log 1 / log 2 else sorry

theorem periodic_odd_function : 
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (2-x) = f x) ∧ (∀ x, 2 ≤ x ∧ x ≤ 3 → f x = log x / log 2 - log 1 / log 2) →
  f (1/3) = log 3 / log 2 - 2 :=
by
  sorry

end periodic_odd_function_l485_485278


namespace ellipse_sum_a_k_l485_485575

theorem ellipse_sum_a_k {a b h k : ℝ}
  (foci1 foci2 : ℝ × ℝ)
  (point_on_ellipse : ℝ × ℝ)
  (h_center : h = (foci1.1 + foci2.1) / 2)
  (k_center : k = (foci1.2 + foci2.2) / 2)
  (distance1 : ℝ := Real.sqrt ((point_on_ellipse.1 - foci1.1)^2 + (point_on_ellipse.2 - foci1.2)^2))
  (distance2 : ℝ := Real.sqrt ((point_on_ellipse.1 - foci2.1)^2 + (point_on_ellipse.2 - foci2.2)^2))
  (major_axis_length : ℝ := distance1 + distance2)
  (h_a : a = major_axis_length / 2)
  (c := Real.sqrt ((foci2.1 - foci1.1)^2 + (foci2.2 - foci1.2)^2) / 2)
  (h_b : b^2 = a^2 - c^2) :
  a + k = (7 + Real.sqrt 13) / 2 := 
by
  sorry

end ellipse_sum_a_k_l485_485575


namespace unique_fraction_representation_l485_485200

theorem unique_fraction_representation (p : ℕ) (h_prime : Nat.Prime p) (h_gt_2 : p > 2) :
  ∃! (x y : ℕ), (x ≠ y) ∧ (2 * x * y = p * (x + y)) :=
by
  sorry

end unique_fraction_representation_l485_485200


namespace triangle_bc_length_l485_485764

theorem triangle_bc_length (A B C : Type) [triangle A B C] (angle_A angle_B : ℝ) (a b c : ℝ)
  (h_A : angle_A = 45) 
  (h_B : angle_B = 60) 
  (h_AC : AC = 6) 
  (law_of_sines : ∀ A B C a b c, b = a * sin A / sin B) :
  BC = 2 * sqrt 6 := 
by
  sorry

end triangle_bc_length_l485_485764


namespace probability_of_vowel_initials_l485_485333

theorem probability_of_vowel_initials (students : Fin 24 → Char) :
  (∀ i j, i ≠ j → students i ≠ students j) →
  (∀ i, students i ∉ {'N', 'O'}) →
  (∀ i, students i ∈ Set.univ) →
  (5/24 : ℚ) = (students.filter (λ student, student ∈ {'A', 'E', 'I', 'U', 'Y'})).size / (students.val.size) := 
by
  sorry

end probability_of_vowel_initials_l485_485333


namespace largest_20_supporting_number_l485_485404

theorem largest_20_supporting_number :
  ∃ X : ℝ, X = 0.025 ∧ (∀ (a : fin 20 → ℝ), (∑ i, a i).isInt → (∃ i, |a i - 0.5| ≥ X)) :=
sorry

end largest_20_supporting_number_l485_485404


namespace MrBensonPaidCorrectAmount_l485_485156

-- Definitions based on the conditions
def generalAdmissionTicketPrice : ℤ := 40
def VIPTicketPrice : ℤ := 60
def premiumTicketPrice : ℤ := 80

def generalAdmissionTicketsBought : ℤ := 10
def VIPTicketsBought : ℤ := 3
def premiumTicketsBought : ℤ := 2

def generalAdmissionExcessThreshold : ℤ := 8
def VIPExcessThreshold : ℤ := 2
def premiumExcessThreshold : ℤ := 1

def generalAdmissionDiscountPercentage : ℤ := 3
def VIPDiscountPercentage : ℤ := 7
def premiumDiscountPercentage : ℤ := 10

-- Function to calculate the cost without discounts
def costWithoutDiscount : ℤ :=
  (generalAdmissionTicketsBought * generalAdmissionTicketPrice) +
  (VIPTicketsBought * VIPTicketPrice) +
  (premiumTicketsBought * premiumTicketPrice)

-- Function to calculate the total discount
def totalDiscount : ℤ :=
  let generalAdmissionDiscount := if generalAdmissionTicketsBought > generalAdmissionExcessThreshold then 
    (generalAdmissionTicketsBought - generalAdmissionExcessThreshold) * generalAdmissionTicketPrice * generalAdmissionDiscountPercentage / 100 else 0
  let VIPDiscount := if VIPTicketsBought > VIPExcessThreshold then 
    (VIPTicketsBought - VIPExcessThreshold) * VIPTicketPrice * VIPDiscountPercentage / 100 else 0
  let premiumDiscount := if premiumTicketsBought > premiumExcessThreshold then 
    (premiumTicketsBought - premiumExcessThreshold) * premiumTicketPrice * premiumDiscountPercentage / 100 else 0
  generalAdmissionDiscount + VIPDiscount + premiumDiscount

-- Function to calculate the total cost after discounts
def totalCostAfterDiscount : ℤ := costWithoutDiscount - totalDiscount

-- Proof statement
theorem MrBensonPaidCorrectAmount :
  totalCostAfterDiscount = 723 :=
by
  sorry

end MrBensonPaidCorrectAmount_l485_485156


namespace math_problem_l485_485795

def Q (x : ℝ) : ℝ := x^2 - 5 * x - 7

theorem math_problem (a b c d e : ℕ) (h1 : 3 ≤ x ∧ x ≤ 10) 
(h2 : (⌊ sqrt (Q x) ⌋ = sqrt (Q (⌊ x ⌋))) →
  (sqrt a + sqrt b + sqrt c - d) / e = probability_of_event) :
  a + b + c + d + e = 75 := sorry

end math_problem_l485_485795


namespace B_work_rate_l485_485134

variables (W : ℝ)
noncomputable def A_rate : ℝ := W / 4
noncomputable def B_C_rate : ℝ := W / 3
noncomputable def A_C_rate : ℝ := W / 2

theorem B_work_rate : ∀ (W : ℝ), (∃ B_rate : ℝ, B_rate = W / 12) :=
by
  -- Define the rates
  let A_rate := W / 4
  let B_C_rate := W / 3
  let A_C_rate := W / 2
    
  -- Assume A_rate, B_C_rate and A_C_rate
  assume (W : ℝ),
  
  -- Using solution steps, the proof will follow here.
  sorry

end B_work_rate_l485_485134


namespace part_I_part_II_l485_485678

variables (A B : Matrix (Fin 2) (Fin 2) ℝ)
variable (A_inv : Matrix (Fin 2) (Fin 2) ℝ)

-- Definitions
def matrix_A : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 1], ![2, 3]]
def matrix_B : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![2, 3]]
def matrix_A_inv : Matrix (Fin 2) (Fin 2) ℝ := ![![3, -1], ![-2, 1]]

-- Theorem statements
theorem part_I : A = matrix_A → A⁻¹ = matrix_A_inv := by 
  assume hA : A = matrix_A
  unfold matrix_A_inv
  sorry

theorem part_II : (A = matrix_A) ∧ (B = matrix_B) ∧ (A⁻¹ = matrix_A_inv) →
  let T := matrix_A_inv * matrix_B in
  ∀ (x y : ℝ), (x + y - 1 = 0) →
  (let x' := T 0 0 * x + T 0 1 * y in
   let y' := T 1 0 * x + T 1 1 * y in
   x' - 2 * y' - 1 = 0) := by
  assume h : (A = matrix_A) ∧ (B = matrix_B) ∧ (A⁻¹ = matrix_A_inv)
  sorry

end part_I_part_II_l485_485678


namespace count_true_propositions_l485_485423

theorem count_true_propositions :
  let propA := ¬∃ (x : ℝ), x^2 + 2 * x + 3 = 0,
      propB := ¬∃ (line : ℝ × ℝ), ∀ (plane1 plane2 : (ℝ × ℝ × ℝ) → ℝ),
        (∀ (p : ℝ × ℝ × ℝ), plane1 p = 0 ∧ plane2 p = 0 → ∃ (p' : ℝ × ℝ × ℝ), plane1 p' = 0 ∧ p' ≠ p) ∧
        (∀ (d : (ℝ × ℝ)- ℕ), (∀ p, line = p × (ℝ × ℝ) → d)),
      propC := ∃ (n : ℕ), (∀ m, m > 0 → m ≤ n → m = 1 ∨ m = n) in
  (if propA then 1 else 0) + (if propB then 1 else 0) + (if propC then 1 else 0) = 1 :=
by
  sorry

end count_true_propositions_l485_485423


namespace garden_bed_length_l485_485969

theorem garden_bed_length (total_area : ℕ) (garden_area : ℕ) (width : ℕ) (n : ℕ)
  (total_area_eq : total_area = 42)
  (garden_area_eq : garden_area = 9)
  (num_gardens_eq : n = 2)
  (width_eq : width = 3)
  (lhs_eq : lhs = total_area - n * garden_area)
  (area_to_length_eq : length = lhs / width) :
  length = 8 := by
  sorry

end garden_bed_length_l485_485969


namespace original_planned_production_l485_485152

theorem original_planned_production (x : ℝ) (hx1 : x ≠ 0) (hx2 : 210 / x - 210 / (1.5 * x) = 5) : x = 14 :=
by sorry

end original_planned_production_l485_485152


namespace rons_chocolate_cost_l485_485539

theorem rons_chocolate_cost :
  let cost_per_bar := 1.5
  let sections_per_bar := 3
  let scouts := 15
  let smores_per_scout := 2
  let total_smores := scouts * smores_per_scout
  let bars_needed := total_smores / sections_per_bar in
  bars_needed * cost_per_bar = 15.0 := by
  sorry

end rons_chocolate_cost_l485_485539


namespace gcd_45_75_l485_485073

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485073


namespace total_logs_in_stack_l485_485174

/-- The total number of logs in a stack where the top row has 5 logs,
each succeeding row has one more log than the one above,
and the bottom row has 15 logs. -/
theorem total_logs_in_stack :
  let a := 5               -- first term (logs in the top row)
  let l := 15              -- last term (logs in the bottom row)
  let n := l - a + 1       -- number of terms (rows)
  let S := n / 2 * (a + l) -- sum of the arithmetic series
  S = 110 := sorry

end total_logs_in_stack_l485_485174


namespace rectangle_area_probability_l485_485266

def probability_rect_area_geq_9 {α : Type} [linear_ordered_field α] : Prop :=
  let length_AB := 10
  let area_threshold := 9
  let prob :=
    let x := @set.Icc α _ 1 9
    let total := @set.Icc α _ 0 10
    (@set.Icc_cardinal α _ 1 9) / (@set.Icc_cardinal α _ 0 10)
  in prob = 4 / 5

-- The main theorem statement
theorem rectangle_area_probability : probability_rect_area_geq_9 :=
sorry

end rectangle_area_probability_l485_485266


namespace math_proof_problem_l485_485324

theorem math_proof_problem
  (a b c d : ℤ)
  (m : ℤ := -1)
  (h1 : |2 + a| + (b - 3)^2 = 0)
  (h2 : c + d = 0) :
  |- (a:ℤ)^b + c - m + d| = 9 :=
by
  sorry

end math_proof_problem_l485_485324


namespace algae_coverage_l485_485153

theorem algae_coverage (quadruple_growth : ∀ (t : ℕ), algae_coverage t = algae_coverage (t+2) * 4)
  (full_coverage : algae_coverage 24 = 1) :
  algae_coverage 20 = 0.0625 :=
by sorry

end algae_coverage_l485_485153


namespace find_third_side_of_triangle_l485_485487

noncomputable def area_triangle_given_sides_angle {a b c : ℝ} (A : ℝ) : Prop :=
  A = 1/2 * a * b * Real.sin c

noncomputable def cosine_law_third_side {a b c : ℝ} (cosα : ℝ) : Prop :=
  c^2 = a^2 + b^2 - 2 * a * b * cosα

theorem find_third_side_of_triangle (a b : ℝ) (Area : ℝ) (h_a : a = 2 * Real.sqrt 2) (h_b : b = 3) (h_Area : Area = 3) :
  ∃ c : ℝ, (c = Real.sqrt 5 ∨ c = Real.sqrt 29) :=
by
  sorry

end find_third_side_of_triangle_l485_485487


namespace find_d_l485_485280

-- Define the proportional condition
def in_proportion (a b c d : ℕ) : Prop := a * d = b * c

-- Given values as parameters
variables {a b c d : ℕ}

-- Theorem to be proven
theorem find_d (h : in_proportion a b c d) (ha : a = 1) (hb : b = 2) (hc : c = 3) : d = 6 :=
sorry

end find_d_l485_485280


namespace max_page_number_with_given_fives_l485_485823

theorem max_page_number_with_given_fives (plenty_digit_except_five : ℕ → ℕ) 
  (H0 : ∀ d ≠ 5, ∀ n, plenty_digit_except_five d = n)
  (H5 : plenty_digit_except_five 5 = 30) : ∃ (n : ℕ), n = 154 :=
by {
  sorry
}

end max_page_number_with_given_fives_l485_485823


namespace geometric_sequence_sum_l485_485688

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_a1 : a 1 = 3)
  (h_sum_first_three : a 1 + a 2 + a 3 = 21) :
  a 4 + a 5 + a 6 = 168 := 
sorry

end geometric_sequence_sum_l485_485688


namespace perimeter_of_triangle_ABC_l485_485339

-- Define the focal points and their radius
def radius : ℝ := 2

-- Define the distances between centers of the tangent circles
def center_distance : ℝ := 2 * radius

-- Define the lengths of the sides of the triangle ABC based on the problem constraints
def AB : ℝ := 2 * radius + 2 * center_distance
def BC : ℝ := 2 * radius + center_distance
def CA : ℝ := 2 * radius + center_distance

-- Define the perimeter calculation
def perimeter : ℝ := AB + BC + CA

-- Theorem stating the actual perimeter of the triangle ABC
theorem perimeter_of_triangle_ABC : perimeter = 28 := by
  sorry

end perimeter_of_triangle_ABC_l485_485339


namespace even_polynomial_coefficient_property_l485_485786

theorem even_polynomial_coefficient_property 
    (p q : ℤ[X])
    (h_pq_even : ∀ i : ℕ, (p * q).coeff i % 2 = 0)
    (h_not_all_div_by_4 : ∃ i : ℕ, (p * q).coeff i % 4 ≠ 0) :
    (∃ (i : ℕ), p.coeff i % 2 = 1 ∧ ∀ j : ℕ, q.coeff j % 2 = 0) ∨ 
    (∃ (i : ℕ), q.coeff i % 2 = 1 ∧ ∀ j : ℕ, p.coeff j % 2 = 0) :=
sorry

end even_polynomial_coefficient_property_l485_485786


namespace count_perfect_cubes_l485_485711

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1600) :
  ∃ (n : ℕ), n = 6 :=
by
  sorry

end count_perfect_cubes_l485_485711


namespace system_of_inequalities_l485_485014

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end system_of_inequalities_l485_485014


namespace profit_difference_eq_1200_l485_485964

noncomputable def A_investment : ℕ := 8000
noncomputable def B_investment : ℕ := 10000
noncomputable def C_investment : ℕ := 12000
noncomputable def B_profit_share : ℕ := 3000

theorem profit_difference_eq_1200 :
  let total_investment := A_investment + B_investment + C_investment
      parts_A := A_investment / 2000
      parts_B := B_investment / 2000
      parts_C := C_investment / 2000
      total_parts := parts_A + parts_B + parts_C
      part_value := B_profit_share / parts_B
      profit_A := parts_A * part_value
      profit_C := parts_C * part_value
      D := profit_C - profit_A
  in D = 1200 := by
  sorry

end profit_difference_eq_1200_l485_485964


namespace green_pill_cost_l485_485184

theorem green_pill_cost (p g : ℕ) (h1 : g = p + 1) (h2 : 14 * (p + g) = 546) : g = 20 :=
by
  sorry

end green_pill_cost_l485_485184


namespace polynomial_bound_proof_l485_485393

open Real

noncomputable theory

def polynomial_bound (n : ℕ) (a : Fin n → ℝ) : 
  ∀ (b : Fin (n + 1) → ℤ),
  ∃ j : Fin (n + 1), 
    (| (∑ i, a i * (b j)^i: ℝ) | ≥ (nat.factorial n / 2^n: ℝ)) :=
by 
  intro n a b
  use sorry

theorem polynomial_bound_proof (n : ℕ) (a : Fin n → ℝ) (b : Fin (n + 1) → ℤ) :
  ∃ j : Fin (n + 1), 
    (| (∑ i, a i * (b j)^i: ℝ) | ≥ (nat.factorial n / 2^n: ℝ)) :=
sorry

end polynomial_bound_proof_l485_485393


namespace prime_square_remainders_l485_485976

theorem prime_square_remainders (p : ℕ) (h1 : Nat.Prime p) (h2 : p > 3) :
  ∃ r ∈ {1, 49}, p^2 % 60 = r :=
by
  sorry

end prime_square_remainders_l485_485976


namespace adult_ticket_cost_l485_485488

variables (x : ℝ)

-- Conditions
def total_tickets := 510
def senior_tickets := 327
def senior_ticket_cost := 15
def total_receipts := 8748

-- Calculation based on the conditions
def adult_tickets := total_tickets - senior_tickets
def senior_receipts := senior_tickets * senior_ticket_cost
def adult_receipts := total_receipts - senior_receipts

-- Define the problem as an assertion to prove
theorem adult_ticket_cost :
  adult_receipts / adult_tickets = 21 := by
  -- Proof steps will go here, but for now, we'll use sorry.
  sorry

end adult_ticket_cost_l485_485488


namespace fraction_representation_correct_l485_485698

theorem fraction_representation_correct (h : ∀ (x y z w: ℕ), 9*x = y ∧ 47*z = w ∧ 2*47*5 = 235):
  (18: ℚ) / (9 * 47 * 5) = (2: ℚ) / 235 :=
by
  sorry

end fraction_representation_correct_l485_485698


namespace max_intersection_points_l485_485154

theorem max_intersection_points (circle : Set Point) (lines : Finset (Set Point)) 
  (h_circle : IsCircle circle) (h_lines : lines.card = 3)
  (h_distinct : ∀ l1 l2 ∈ lines, l1 ≠ l2 → l1 ∩ l2 = ∅ ∨ l1 ∩ l2.card = 1) :
  ∃ n <= 9, count_intersection_points(circle, lines) = n := 
sorry

end max_intersection_points_l485_485154


namespace count_perfect_cubes_l485_485712

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1600) :
  ∃ (n : ℕ), n = 6 :=
by
  sorry

end count_perfect_cubes_l485_485712


namespace payment_for_C_l485_485545

theorem payment_for_C (A_rate B_rate : ℚ) (total_payment : ℚ) (days_worked : ℕ) :
  A_rate = 1/6 → B_rate = 1/8 → total_payment = 3840 → days_worked = 3 →
  let total_work := 1 in
  let work_done_by_AB := (A_rate + B_rate) * days_worked in
  let work_done_by_C := total_work - work_done_by_AB in
  let payment_for_C := work_done_by_C * total_payment in
  payment_for_C = 480 :=
sorry

end payment_for_C_l485_485545


namespace sum_abc_l485_485719

theorem sum_abc (a b c : ℕ) (h1 : ab + 2c + 3 = 47) (h2 : bc + 2a + 3 = 47) (h3 : ac + 2b + 3 = 47) : a + b + c = 16 := by
  sorry

end sum_abc_l485_485719


namespace calculate_expression_l485_485422

theorem calculate_expression : 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) = 3^16 - 1 :=
by 
  sorry

end calculate_expression_l485_485422


namespace determine_a_l485_485835

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem determine_a (a : ℝ) (h : f' a (-1) = 4) : a = 10 / 3 :=
by {
  sorry
}

end determine_a_l485_485835


namespace triangulate_color_eq_area_l485_485789

theorem triangulate_color_eq_area (m n : ℕ) (h_n : 3 ≤ n) (h_pos : 0 < m) :
  (∃ (triangulation : list (triangle ℂ)), ∀ color : ℕ, 
     color < m → 
     ∑ triangle in triangulation, 
     if triangle.color = color then triangle.area else 0 
     = (∑ triangle in triangulation, triangle.area) / m) ↔ m ∣ n :=
sorry

end triangulate_color_eq_area_l485_485789


namespace merchant_profit_l485_485525

theorem merchant_profit (C S : ℝ) (h: 20 * C = 15 * S) : 
  (S - C) / C * 100 = 33.33 := by
sorry

end merchant_profit_l485_485525


namespace distance_and_velocity_proof_l485_485662

-- Definitions based on the conditions
def radius := 1.2 * 10^5 -- km
def initial_velocity_moon := 3   -- km/s (rounded up from 3.27 km/s)
def time_intersection := 2 * 10^4 -- seconds
def sub_probe_velocity := 6   -- km/s
def approx_pi := 3.14

-- Correct answer constants
def distance_calculated := 1.2 * 10^5 * (Real.sqrt 3 - 1)  -- km
def velocity_on_CD := 3   -- km/s

-- Proof statement
theorem distance_and_velocity_proof :
  let R := radius,
      V_T := initial_velocity_moon,
      t := time_intersection,
      V_1 := sub_probe_velocity,
      π := approx_pi,
      distance := distance_calculated,
      V_2 := velocity_on_CD
  in
    ((R * (Real.sqrt 3 - 1) = distance) 
     ∧ (V_2 = velocity_on_CD)) :=
by 
  let R := radius,
      V_T := initial_velocity_moon,
      t := time_intersection,
      V_1 := sub_probe_velocity,
      π := approx_pi,
      distance := distance_calculated,
      V_2 := velocity_on_CD
  have dist_eq : R * (Real.sqrt 3 - 1) = distance := sorry
  have vel_eq : V_2 = velocity_on_CD := sorry
  exact ⟨dist_eq, vel_eq⟩

end distance_and_velocity_proof_l485_485662


namespace complex_modulus_calc_l485_485637

theorem complex_modulus_calc :
  let z := complex.mk (3 / 4) (- (5 / 6))
  abs z = real.sqrt (181) / 12 := by
  sorry

end complex_modulus_calc_l485_485637


namespace calculate_total_cost_l485_485580

def total_cost (num_boxes : ℕ) (packs_per_box : ℕ) (tissues_per_pack : ℕ) (cost_per_tissue : ℝ) : ℝ :=
  num_boxes * packs_per_box * tissues_per_pack * cost_per_tissue

theorem calculate_total_cost :
  total_cost 10 20 100 0.05 = 1000 := 
by
  sorry

end calculate_total_cost_l485_485580


namespace smallest_n_l485_485836

noncomputable def smallest_positive_integer (x y : ℤ) (h1 : (x + 1) % 7 = 0) (h2 : (y - 5) % 7 = 0) : ℕ :=
  if 3 % 7 = 0 then 7 else 7

theorem smallest_n (x y : ℤ) (h1 : (x + 1) % 7 = 0) (h2 : (y - 5) % 7 = 0) : smallest_positive_integer x y h1 h2 = 7 := 
  by
  admit

end smallest_n_l485_485836


namespace inequality_solution_l485_485008

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end inequality_solution_l485_485008


namespace range_of_values_for_a_l485_485702

theorem range_of_values_for_a :
  (∀ b : ℝ, ∃ x : ℝ, f x = b) →
  ∃ a : ℝ, a ∈ [-11, 5] :=
begin
  sorry
end

def f (x a : ℝ) : ℝ :=
  if x < a then x + 10 else x^2 - 2 * x

def is_surjective (f : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, f x = y

end range_of_values_for_a_l485_485702


namespace solve_for_x_l485_485242

theorem solve_for_x (x : ℝ) : sqrt (5 * x + 13) = 15 → x = 212 / 5 :=
by
  intro h
  sorry

end solve_for_x_l485_485242


namespace part_I_part_II_l485_485670

-- Part I: Inequality solution
theorem part_I (x : ℝ) : 
  (abs (x - 1) ≥ 4 - abs (x - 3)) ↔ (x ≤ 0 ∨ x ≥ 4) := 
sorry

-- Part II: Minimum value of mn
theorem part_II (m n : ℕ) (h1 : (1:ℝ)/m + (1:ℝ)/(2*n) = 1) (hm : 0 < m) (hn : 0 < n) :
  (mn : ℕ) = 2 :=
sorry

end part_I_part_II_l485_485670


namespace frank_initial_money_l485_485255

theorem frank_initial_money (cost_cheapest : ℕ) (most_expensive_ratio : ℕ) (remaining_money : ℕ) :
  cost_cheapest = 20 → most_expensive_ratio = 3 → remaining_money = 30 → 
  let cost_most_expensive := most_expensive_ratio * cost_cheapest in
  let initial_money := cost_most_expensive + remaining_money in
  initial_money = 90 :=
by
  intros h1 h2 h3
  unfold cost_most_expensive initial_money
  rw [h1, h2, h3]
  simp
  sorry

end frank_initial_money_l485_485255


namespace cyclist_average_speed_l485_485519

def total_distance : ℝ := 9 + 11
def total_time : ℝ := (9 / 11) + (11 / 9)
def average_speed : ℝ := total_distance / total_time

theorem cyclist_average_speed : average_speed ≈ 9.8 := by
  have h1 : total_distance = 20 := by norm_num
  have h2 : total_time ≈ 2.04 := by norm_num
  have h3 : average_speed = total_distance / total_time := rfl
  have h4 : 20 / 2.04 ≈ 9.8 := by norm_num
  exact h4

end cyclist_average_speed_l485_485519


namespace distinct_flags_count_l485_485935

open Finset

def colors : Finset ℕ := {0, 1, 2, 3, 4} -- Representing the colors red, white, blue, green, yellow as 0, 1, 2, 3, 4

theorem distinct_flags_count :
  let choices := colors.card,
      adj_distinct (x y : ℕ) := x ≠ y in
  ∑ middle in colors, ∑ top in colors.filter (adj_distinct middle), ∑ bottom in colors.filter (adj_distinct middle) = 80 :=
by sorry

end distinct_flags_count_l485_485935


namespace number_of_ways_to_arrange_cups_l485_485934

theorem number_of_ways_to_arrange_cups :
  let total_cups := 9
  let yellow_cups := 4
  let blue_cups := 3
  let red_cups := 2
  ∃ ways : Nat, 
    (ways = fact total_cups / ((fact yellow_cups) * (fact blue_cups) * (fact red_cups)) / total_cups -
            fact (total_cups - 1) / ((fact (yellow_cups - 1)) * (fact blue_cups) * (fact 1)) / (total_cups - 1)) →
    ways = 105 := sorry

end number_of_ways_to_arrange_cups_l485_485934


namespace total_students_in_school_l485_485746

-- Define the conditions
def num_girls : ℕ := 739
def num_more_girls_than_boys : ℕ := 402

-- Question to prove
theorem total_students_in_school : (num_students : ℕ) :=
  let num_boys := num_girls - num_more_girls_than_boys
  let total_students := num_girls + num_boys
  total_students = 1076 :=
by
  sorry

end total_students_in_school_l485_485746


namespace find_A_y_coordinate_l485_485138

/-- Definition of distances between points in 3D space -/
def dist (p q : ℝ^3) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

/-- Given points A, B, and C, prove that if A is equidistant from B and C,
    then the y-coordinate of A is 1/2 -/
theorem find_A_y_coordinate :
  ∀ (y : ℝ),
    let A := (0, y, 0) in
    let B := (3, 0, 3) in
    let C := (0, 2, 4) in
    dist A B = dist A C → y = 1 / 2 :=
by
  intros y A B C h
  have h1 : dist A B = dist (0, y, 0) (3, 0, 3) := rfl
  have h2 : dist A C = dist (0, y, 0) (0, 2, 4) := rfl
  rw h1 at h
  rw h2 at h
  -- Simplified distances using given distances in the problem
  sorry

end find_A_y_coordinate_l485_485138


namespace equal_sharing_l485_485915

theorem equal_sharing (total_cards friends : ℕ) (h1 : total_cards = 455) (h2 : friends = 5) : total_cards / friends = 91 := by
  sorry

end equal_sharing_l485_485915


namespace smallest_period_f_range_f_on_interval_l485_485291

def f (x : ℝ) : ℝ := (sqrt 3) * sin x * cos x - (cos x)^2 + 1/2

theorem smallest_period_f :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T')
  → T = π := 
sorry

theorem range_f_on_interval :
  ∀ x ∈ Icc 0 (π / 4), f x ∈ Icc (-1 / 2) (sqrt 3 / 2) :=
sorry

end smallest_period_f_range_f_on_interval_l485_485291


namespace unique_value_expression_l485_485826

theorem unique_value_expression (m n : ℤ) : 
  (mn + 13 * m + 13 * n - m^2 - n^2 = 169) → 
  ∃! (m n : ℤ), mn + 13 * m + 13 * n - m^2 - n^2 = 169 := 
by
  sorry

end unique_value_expression_l485_485826


namespace shift_quadratic_function_left_l485_485734

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2

-- Define the shifted quadratic function
def shifted_function (x : ℝ) : ℝ := (x + 1)^2

-- Theorem statement
theorem shift_quadratic_function_left :
  ∀ x : ℝ, shifted_function x = original_function (x + 1) := by
  sorry

end shift_quadratic_function_left_l485_485734


namespace cube_paint_problem_l485_485549

theorem cube_paint_problem : 
  ∀ (n : ℕ),
  n = 6 →
  (∃ k : ℕ, 216 = k^3 ∧ k = n) →
  ∀ (faces inner_faces total_cubelets : ℕ),
  faces = 6 →
  inner_faces = 4 →
  total_cubelets = faces * (inner_faces * inner_faces) →
  total_cubelets = 96 :=
by 
  intros n hn hc faces hfaces inner_faces hinner_faces total_cubelets htotal_cubelets
  sorry

end cube_paint_problem_l485_485549


namespace arithmetic_sequence_k_value_l485_485380

theorem arithmetic_sequence_k_value (a_1 d : ℕ) (h1 : a_1 = 1) (h2 : d = 2) (k : ℕ) (S : ℕ → ℕ) (h_sum : ∀ n, S n = n * (2 * a_1 + (n - 1) * d) / 2) (h_condition : S (k + 2) - S k = 24) : k = 5 :=
by {
  sorry
}

end arithmetic_sequence_k_value_l485_485380


namespace total_pens_l485_485444

theorem total_pens (black_pens blue_pens : ℕ) (h1 : black_pens = 4) (h2 : blue_pens = 4) : black_pens + blue_pens = 8 :=
by
  sorry

end total_pens_l485_485444


namespace kenya_peanuts_eq_133_l485_485780

def num_peanuts_jose : Nat := 85
def additional_peanuts_kenya : Nat := 48

def peanuts_kenya (jose_peanuts : Nat) (additional_peanuts : Nat) : Nat :=
  jose_peanuts + additional_peanuts

theorem kenya_peanuts_eq_133 : peanuts_kenya num_peanuts_jose additional_peanuts_kenya = 133 := by
  sorry

end kenya_peanuts_eq_133_l485_485780


namespace flowers_around_fish_pond_l485_485158

theorem flowers_around_fish_pond :
  ∀ (perimeter interval flowers_per_interval : ℕ),
  perimeter = 52 →
  interval = 4 →
  flowers_per_interval = 3 →
  (perimeter / interval) * flowers_per_interval = 39 :=
by
  intros perimeter interval flowers_per_interval h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end flowers_around_fish_pond_l485_485158


namespace scientific_notation_456000_l485_485634

-- Define the number of tourists
def tourists := 456000

-- Scientific notation definition
def scientific_notation (n : ℕ) : Prop :=
  n = 456000 → n = 4.56 * 10^5

-- Lean statement to prove
theorem scientific_notation_456000 : scientific_notation tourists :=
by
  sorry

end scientific_notation_456000_l485_485634


namespace chess_bishop_game_winner_l485_485416

-- Define the board side as a natural number
def N : ℕ := sorry

-- Define a predicate to check if N is even or odd
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- Main theorem determining the winner -/
theorem chess_bishop_game_winner (N : ℕ)
  (hN : N > 0) :
  (is_odd N → "Danil wins") ∧ (is_even N → "Alexei wins") := 
sorry

end chess_bishop_game_winner_l485_485416


namespace range_of_a_l485_485273

theorem range_of_a (a x : ℝ) (p : -4 < x - a ∧ x - a < 4)
    (q : 1 < x ∧ x < 3)
    (h : (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x)) :
    -1 ≤ a ∧ a ≤ 5 :=
by
  sorry

end range_of_a_l485_485273


namespace similar_triangles_l485_485383

theorem similar_triangles {Γ1 Γ2 : Type} [circle Γ1] [circle Γ2]
  (O1 O2 X Y A B : Type)
  [H1: center Γ1 = O1]
  [H2: center Γ2 = O2]
  [H3: X ∈ Γ1 ∧ X ∈ Γ2]
  [H4: Y ∈ Γ1 ∧ Y ∈ Γ2]
  [H5: A ∈ Γ1 ∧ A ≠ X ∧ A ≠ Y]
  [H6: B ∈ Γ2 ∧ lies_on_line B (line_through A Y)] :
  similar (triangle X O1 O2) (triangle X A B) :=
sorry

end similar_triangles_l485_485383


namespace distance_from_center_to_midpoint_apothem_l485_485459

variables (α S : ℝ)

open_locale real

noncomputable def distance_to_midpoint_apothem (α S : ℝ) : ℝ :=
  (sqrt (S * sqrt 3 * cos α)) / (6 * cos α)

theorem distance_from_center_to_midpoint_apothem :
  ∀ (α S : ℝ), 
  distance_to_midpoint_apothem α S = (sqrt (S * sqrt 3 * cos α)) / (6 * cos α) :=
by
  intros
  sorry

end distance_from_center_to_midpoint_apothem_l485_485459


namespace ron_chocolate_bar_cost_l485_485535

-- Definitions of the conditions given in the problem
def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def s'mores_needed_per_scout : ℕ := 2
def total_s'mores_needed : ℕ := scouts * s'mores_needed_per_scout
def chocolate_bars_needed : ℕ := total_s'mores_needed / sections_per_chocolate_bar
def total_cost_of_chocolate_bars : ℝ := chocolate_bars_needed * cost_per_chocolate_bar

-- Proving the question equals the answer given conditions
theorem ron_chocolate_bar_cost : total_cost_of_chocolate_bars = 15.00 := by
  sorry

end ron_chocolate_bar_cost_l485_485535


namespace unique_n_for_prime_p_l485_485829

theorem unique_n_for_prime_p (p : ℕ) (hp1 : p > 2) (hp2 : Nat.Prime p) :
  ∃! (n : ℕ), (∃ (k : ℕ), n^2 + n * p = k^2) ∧ n = (p - 1) / 2 ^ 2 :=
sorry

end unique_n_for_prime_p_l485_485829


namespace eval_expression_l485_485671

-- Define the new operation \(\otimes\)
def op (a b : ℝ) : ℝ := (a^2 + b^2) / (a - b)

-- State the theorem to prove the given expression
theorem eval_expression : op (op 7 5) 4 = 42 + 1 / 33 :=
by
  -- Provide proof steps here 
  sorry

end eval_expression_l485_485671


namespace find_complex_solution_l485_485238

def complex_modulus (z : ℂ) : ℝ := complex.abs z

noncomputable def z : ℂ := -1 - complex.I

theorem find_complex_solution :
  complex_modulus (z - 2) = complex_modulus (z + 4) ∧
  complex_modulus (z + 4) = complex_modulus (z - 2*complex.I) :=
by
  sorry

end find_complex_solution_l485_485238


namespace gcd_45_75_l485_485054

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485054


namespace problem_conditions_l485_485721

theorem problem_conditions (x y : ℝ) (h : x^2 + y^2 - x * y = 1) :
  ¬ (x + y ≤ 1) ∧ (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) ∧ ¬ (x^2 + y^2 ≥ 1) :=
by
  sorry

end problem_conditions_l485_485721


namespace kenya_peanuts_l485_485783

def jose_peanuts : ℕ := 85
def difference : ℕ := 48

theorem kenya_peanuts : jose_peanuts + difference = 133 := by
  sorry

end kenya_peanuts_l485_485783


namespace find_matrix_and_inverse_l485_485302

open Matrix
open LinearMap

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 2, 1]
noncomputable def matrix_A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-1/4, 3/4; 1/2, -1/2]
noncomputable def evec1 : Vector (Fin 2) ℝ := !![1, -1]
noncomputable def evec2 : Vector (Fin 2) ℝ := !![3, 2]
noncomputable def eval1 : ℝ := -1
noncomputable def eval2 : ℝ := 4

theorem find_matrix_and_inverse :
  let A := !![2, 3; 2, 1] in
  let A_inv := !![-1/4, 3/4; 1/2, -1/2] in
  A = matrix_A ∧ A_inv = matrix_A_inv :=
by
  sorry

end find_matrix_and_inverse_l485_485302


namespace even_count_in_pascal_triangle_l485_485611

-- Define the binomial coefficient as a function
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define a predicate to check if a number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Count the number of even integers in the top 15 rows of Pascal's Triangle
def count_even_pascal (rows : ℕ) : ℕ :=
  (Finset.range rows).sum (λ n => (Finset.range (n + 1)).count (λ k => is_even (binom n k)))

-- Statement of the problem
theorem even_count_in_pascal_triangle : count_even_pascal 15 = 84 :=
  by
    sorry

end even_count_in_pascal_triangle_l485_485611


namespace pascals_triangle_even_count_15_l485_485615

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

def is_even (x : ℕ) : Prop := x % 2 = 0

def count_even_in_row (n : ℕ) : ℕ :=
  (finset.range (n + 1)).count (λ k, is_even (pascal n k))

def count_even_in_first_15_rows : ℕ :=
  (finset.range 15).sum count_even_in_row

theorem pascals_triangle_even_count_15 :
  count_even_in_first_15_rows = 64 :=
by sorry

end pascals_triangle_even_count_15_l485_485615


namespace problem_1_expr_problem_2_simplify_l485_485144

-- Problem 1
theorem problem_1_expr : 
  27^(2/3:ℝ) + 16^(-1/2:ℝ) - (1/2:ℝ)^(-2) - (8/27:ℝ)^(-2/3) = 3 := 
  by
  sorry

-- Problem 2
theorem problem_2_simplify (a : ℝ) (h : 1 ≤ a) : 
  (sqrt (a - 1))^2 + sqrt ((1 - a)^2) + cbrt ((1 - a)^3) = a - 1 := 
  by
  sorry

end problem_1_expr_problem_2_simplify_l485_485144


namespace count_valid_sequences_l485_485458

theorem count_valid_sequences : 
  (∃ (A B C : ℕ), 
    A < B ∧ B < C ∧ 
    B = (A + C) / 2 ∧ 
    1 ≤ A ∧ A ≤ 9 ∧ 
    1 ≤ B ∧ B ≤ 9 ∧ 
    1 ≤ C ∧ C ≤ 9) = 14 := 
sorry

end count_valid_sequences_l485_485458


namespace interval_of_systematic_sampling_l485_485043

theorem interval_of_systematic_sampling (total_students sample_size : ℕ) (h1 : total_students = 72) (h2 : sample_size = 8) : total_students / sample_size = 9 :=
by
  rw [h1, h2]
  simp
  sorry

end interval_of_systematic_sampling_l485_485043


namespace line_properties_l485_485030

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + y + 1 = 0

-- Define what it means to be the slope of a line
def slope (m : ℝ) (x y : ℝ) : Prop := line_eq x y → m = -1

-- Define what it means to be the y-intercept of a line
def y_intercept (b : ℝ) (x y : ℝ) : Prop :=
    line_eq x y → x = 0 → y = b

-- Define what it means to be the slope angle
def slope_angle (θ : ℝ) : Prop := θ = 135 -- The slope angle corresponding to a slope of -1 is 135 degrees

-- Formalized theorem
theorem line_properties :
    slope_angle 135 ∧ ∃ x y, y_intercept (-1) x y :=
by
  -- Add sorry to skip the proof
  sorry

end line_properties_l485_485030


namespace diagonal_path_exists_l485_485346

theorem diagonal_path_exists {m n : ℕ} (d : Fin m → Fin n → Bool) : 
  (∃ path : List (Fin m × Fin n), 
      path.head.1 = 0 ∧ path.last.1 = m - 1) ∨
  (∃ path : List (Fin m × Fin n), 
      path.head.2 = 0 ∧ path.last.2 = n - 1) :=
sorry

end diagonal_path_exists_l485_485346


namespace simplify_expression_1_simplify_expression_2_l485_485592

theorem simplify_expression_1 (x y : ℝ) :
  x^2 + 5*y - 4*x^2 - 3*y = -3*x^2 + 2*y :=
sorry

theorem simplify_expression_2 (a b : ℝ) :
  7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b :=
sorry

end simplify_expression_1_simplify_expression_2_l485_485592


namespace man_profit_and_percent_l485_485163

noncomputable def total_cost_price (n : ℕ) : ℝ :=
  let marked_price := 2.40
  let discount_5 := 0.05
  let discount_10 := 0.10
  let discount_15 := 0.15
  let cost_first_50 := 50 * marked_price * (1 - discount_5)
  let cost_next_50 := 50 * marked_price * (1 - discount_10)
  let cost_remaining := (n - 100) * marked_price * (1 - discount_15)
  in cost_first_50 + cost_next_50 + cost_remaining

noncomputable def total_selling_price (n : ℕ) : ℝ :=
  let marked_price := 2.40
  let discount_4 := 0.04
  let discount_2 := 0.02
  let sell_first_75 := 75 * marked_price * (1 - discount_4)
  let sell_remaining_75 := (n - 75) * marked_price * (1 - discount_2)
  in sell_first_75 + sell_remaining_75

noncomputable def profit (n : ℕ) : ℝ :=
  total_selling_price n - total_cost_price n

noncomputable def profit_percent (n : ℕ) : ℝ :=
  (profit n / total_cost_price n) * 100

theorem man_profit_and_percent :
  profit 150 = 25.20 ∧ profit_percent 150 ≈ 7.78 := by
  sorry

end man_profit_and_percent_l485_485163


namespace rhombus_area_l485_485237

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 20) : 
  (d1 * d2) / 2 = 160 := by
sorry

end rhombus_area_l485_485237


namespace cricketer_percentage_runs_by_running_l485_485518

noncomputable def percent_runs_by_running (total_runs boundaries sixes : ℕ) :=
  let runs_from_boundaries := boundaries * 4
  let runs_from_sixes := sixes * 6
  let total_runs_from_boundaries_and_sixes := runs_from_boundaries + runs_from_sixes
  let runs_by_running := total_runs - total_runs_from_boundaries_and_sixes
  (runs_by_running / total_runs.toFloat) * 100

theorem cricketer_percentage_runs_by_running :
  percent_runs_by_running 134 12 2 ≈ 55.22 :=
by
  sorry

end cricketer_percentage_runs_by_running_l485_485518


namespace sum_of_valid_a_l485_485328

theorem sum_of_valid_a :
  (∀ (x a : ℝ), x + 1 > (x - 1) / 3 ∧ x + a < 3 → 3 - a > -2 ∧ a < 5) →
  (∀ (y a : ℝ), a < 5 ∧ (y - a) / (y - 2) + 1 = 1 / (y - 2) → 
     (∃ y : ℕ, y > 0 ∧ y = (a + 3) / 2 ∧ y ≠ 2)) →
  ∑ a in [3, -1], a = 2 :=
by sorry

end sum_of_valid_a_l485_485328


namespace probability_even_distinct_nonzero_digits_l485_485214

open Nat

-- Definitions for conditions
def isEven (n : ℕ) : Prop := n % 2 = 0

def allDistinctDigits (n : ℕ) : Prop :=
  let digits := (toDigits 10 n).erase 0
  digits.nodup

def inRange (n : ℕ) : Prop := 2000 ≤ n ∧ n ≤ 9999

-- Statement of the problem to be proven
theorem probability_even_distinct_nonzero_digits : 
  let favorableCount := 
    (filter (λ n, isEven n ∧ allDistinctDigits n ∧ inRange n) 
            (list.range' 2000 (9999 - 2000 + 1))).length
  let totalCount := 8000
  favorableCount / totalCount = 21 / 125 := 
by
  sorry

end probability_even_distinct_nonzero_digits_l485_485214


namespace inequality_proof_l485_485807

variable {n : ℕ} (hn : n > 1)
variable {x : Fin n → ℝ} (hx : ∀ i, 0 < x i) (hsum : ∑ i, x i = 1)

theorem inequality_proof : 
  ∑ i : Fin n, x i / (x ((i + 1) % n) - (x ((i + 1) % n)) ^ 3) ≥ n ^ 3 / (n ^ 2 - 1) :=
sorry

end inequality_proof_l485_485807


namespace gcd_of_45_and_75_l485_485081

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l485_485081


namespace incorrect_inequality_l485_485321

variable (a b : ℝ)

theorem incorrect_inequality (h : a > b) : ¬ (-2 * a > -2 * b) :=
by sorry

end incorrect_inequality_l485_485321


namespace amare_needs_four_dresses_l485_485417

noncomputable def num_dresses (fabric_per_dress : ℝ) (fabric_amare_has_ft : ℝ) (fabric_amare_needs_ft : ℝ) : ℝ :=
  let fabric_per_yard := 3.0
  let fabric_amare_has_yd := fabric_amare_has_ft / fabric_per_yard
  let fabric_amare_needs_yd := fabric_amare_needs_ft / fabric_per_yard
  let total_fabric_needed := fabric_amare_has_yd + fabric_amare_needs_yd
  total_fabric_needed / fabric_per_dress

theorem amare_needs_four_dresses :
  num_dresses 5.5 7 59 = 4 :=
by {
  -- Definition and transformation using conditions
  let fabric_per_yard := 3
  let fabric_amare_has_yd := 7 / fabric_per_yard
  let fabric_amare_needs_yd := 59 / fabric_per_yard
  have fabric_required := fabric_amare_has_yd + fabric_amare_needs_yd
  have number_of_dresses := fabric_required / 5.5
  -- Conclude the theorem
  sorry
}

end amare_needs_four_dresses_l485_485417


namespace even_integers_in_pascals_triangle_top_15_rows_l485_485621

/-- Prove that the total number of even integers in the top 15 rows of Pascal's Triangle is exactly 90.
  Pascal's Triangle's elements, binomial(n,k), are even unless every binary digit of k 
  is present in n when both are expressed in binary. -/
theorem even_integers_in_pascals_triangle_top_15_rows : 
  ∑ n in Finset.range 15, ∑ k in Finset.range (n + 1), if (∀ i, (n.bits.get i) = 1 → (k.bits.get i) = 1) then 0 else 1 = 90 :=
sorry

end even_integers_in_pascals_triangle_top_15_rows_l485_485621


namespace solve_quadratic_inequality_l485_485440

theorem solve_quadratic_inequality (x : ℝ) : 3 * x^2 - 5 * x - 2 < 0 → (-1 / 3 < x ∧ x < 2) :=
by
  intro h
  sorry

end solve_quadratic_inequality_l485_485440


namespace largest_20_supporting_X_l485_485407

-- Define the predicate for a number X being 20-supporting
def is_20_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 20 → ℝ), (∑ i, a i).toInt = ∑ i, a i →
  ∃ i, |a i - 0.5| ≥ X

-- Statement to prove the largest 20-supporting X, which is 1/40
theorem largest_20_supporting_X : ∃ X : ℝ, X = 1 / 40 ∧ is_20_supporting X := by
  sorry

end largest_20_supporting_X_l485_485407


namespace part1_expression_evaluation_l485_485533

theorem part1_expression_evaluation :
    3 * (-4)^3 - (1/2)^0 + (0.25)^(1/2) * ((-1)/(Real.sqrt 2))^(-4) = -3 := 
by
  sorry

end part1_expression_evaluation_l485_485533


namespace age_of_15th_person_l485_485446

variable (avg_age_20 : ℕ) (avg_age_5 : ℕ) (avg_age_9 : ℕ) (A : ℕ)
variable (num_20 : ℕ) (num_5 : ℕ) (num_9 : ℕ)

theorem age_of_15th_person (h1 : avg_age_20 = 15) (h2 : avg_age_5 = 14) (h3 : avg_age_9 = 16)
  (h4 : num_20 = 20) (h5 : num_5 = 5) (h6 : num_9 = 9) :
  (num_20 * avg_age_20) = (num_5 * avg_age_5) + (num_9 * avg_age_9) + A → A = 86 :=
by
  sorry

end age_of_15th_person_l485_485446


namespace even_count_in_top_15_rows_l485_485605

def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

def count_even_in_row (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ k, is_even (binom n k)).card

def count_even_in_top_15_rows : ℕ :=
  (Finset.range 15).sum count_even_in_row

theorem even_count_in_top_15_rows :
  count_even_in_top_15_rows = "Sum of all identified evens" := by
sorry

end even_count_in_top_15_rows_l485_485605


namespace sum_induction_step_l485_485424

theorem sum_induction_step (k : ℕ) (h : 1 + 2 + 3 + ... + (2 * k + 1) = (k + 1) * (2 * k + 1)) :
  1 + 2 + 3 + ... + (2 * k + 1) + (2 * k + 2) + (2 * k + 3) = (k + 2) * (2 * k + 3) := by
  sorry

end sum_induction_step_l485_485424


namespace routes_from_P_to_Q_l485_485898

-- Definitions for the conditions
def connects_to (a b : String) : Prop := 
  (a, b) ∈ [("P", "X"), ("X", "Y"), ("X", "Z"), ("X", "Q"), ("Y", "Q"), ("Y", "R"), ("Z", "Q"), ("R", "Q")]

-- Define the number of routes
def routes (start end : String) : Nat :=
  if (start, end) ∈ [("P", "Q")] then 4
  else if (start, end) ∈ [("X", "Q")] then 4
  else if (start, end) ∈ [("Y", "Q")] then 2
  else if (start, end) ∈ [("Z", "Q"), ("R", "Q")] then 1
  else if (start, end) ∈ [("P", "X")] then 1
  else 0

-- Prove that the number of different routes from P to Q is 4
theorem routes_from_P_to_Q : routes "P" "Q" = 4 := 
  by 
    -- skip proof
    sorry

end routes_from_P_to_Q_l485_485898


namespace alpha_plus_beta_l485_485313

theorem alpha_plus_beta (α β : ℝ) 
  (h₁ : 0 < α ∧ α < (π / 2)) 
  (h₂ : 0 < β ∧ β < (π / 2)) 
  (h₃ : Real.cot α = 4) 
  (h₄ : Real.cot β = (5 / 3)) : 
  α + β = π / 4 :=
sorry

end alpha_plus_beta_l485_485313


namespace integer_for_finitely_many_n_l485_485437

theorem integer_for_finitely_many_n (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ N : ℕ, ∀ n : ℕ, N < n → ¬ ∃ k : ℤ, (a + 1 / 2) ^ n + (b + 1 / 2) ^ n = k := 
sorry

end integer_for_finitely_many_n_l485_485437


namespace point_B_l485_485772

-- Define constants for perimeter and speed factor
def perimeter : ℕ := 24
def speed_factor : ℕ := 2

-- Define the speeds of Jane and Hector
def hector_speed (s : ℕ) : ℕ := s
def jane_speed (s : ℕ) : ℕ := speed_factor * s

-- Define the times until they meet
def time_until_meeting (s : ℕ) : ℚ := perimeter / (hector_speed s + jane_speed s)

-- Distances walked by Hector and Jane upon meeting
noncomputable def hector_distance (s : ℕ) : ℚ := hector_speed s * time_until_meeting s
noncomputable def jane_distance (s : ℕ) : ℚ := jane_speed s * time_until_meeting s

-- Map the perimeter position to a point
def position_on_track (d : ℚ) : ℚ := d % perimeter

-- When they meet
theorem point_B (s : ℕ) (h₀ : 0 < s) : position_on_track (hector_distance s) = position_on_track (jane_distance s) → 
                          position_on_track (hector_distance s) = 8 := 
by 
  sorry

end point_B_l485_485772


namespace complex_multiplication_conjugate_l485_485287

-- Given condition
def z : ℂ := 2 - complex.i

-- Proof statement
theorem complex_multiplication_conjugate : z * (conj z) = 5 :=
by
  sorry

end complex_multiplication_conjugate_l485_485287


namespace savanna_total_animals_l485_485432

def num_lions_safari := 100
def num_snakes_safari := num_lions_safari / 2
def num_giraffes_safari := num_snakes_safari - 10
def num_elephants_safari := num_lions_safari / 4

def num_lions_savanna := num_lions_safari * 2
def num_snakes_savanna := num_snakes_safari * 3
def num_giraffes_savanna := num_giraffes_safari + 20
def num_elephants_savanna := num_elephants_safari * 5
def num_zebras_savanna := (num_lions_savanna + num_snakes_savanna) / 2

def total_animals_savanna := 
  num_lions_savanna 
  + num_snakes_savanna 
  + num_giraffes_savanna 
  + num_elephants_savanna 
  + num_zebras_savanna

open Nat
theorem savanna_total_animals : total_animals_savanna = 710 := by
  sorry

end savanna_total_animals_l485_485432


namespace ab_bc_ca_lt_quarter_l485_485680

theorem ab_bc_ca_lt_quarter (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 1) :
  (a * b)^(5/4) + (b * c)^(5/4) + (c * a)^(5/4) < 1/4 :=
sorry

end ab_bc_ca_lt_quarter_l485_485680


namespace number_of_arrangements_with_one_person_between_A_and_B_is_36_l485_485481

theorem number_of_arrangements_with_one_person_between_A_and_B_is_36 :
  ∃ (A B P1 P2 P3 : Type), P1 ≠ A ∧ P1 ≠ B ∧ 
  P2 ≠ A ∧ P2 ≠ B ∧ P2 ≠ P1 ∧ 
  P3 ≠ A ∧ P3 ≠ B ∧ P3 ≠ P1 ∧ P3 ≠ P2 ∧
  (P1 ≠ P2 ∧ P1 ≠ P3 ∧ P2 ≠ P3) ∧ 
  ( ∃ (arr : list Type), arr.perm [A, P1, B, P2, P3] ∧ length arr = 5 ∧ 
  (arr.index_of A + 2 = arr.index_of B ∨ arr.index_of B + 2 = arr.index_of A)) 
   → 36 := sorry

end number_of_arrangements_with_one_person_between_A_and_B_is_36_l485_485481


namespace fifteenth_prime_is_47_l485_485435

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_prime (n : ℕ) : ℕ :=
  if h : n > 0 then (Nat.filter is_prime (List.range (n + 100))).get ⟨n - 1, sorry⟩ else 0

theorem fifteenth_prime_is_47 :
  nth_prime 15 = 47 :=
by
  sorry

end fifteenth_prime_is_47_l485_485435


namespace right_triangles_not_1000_l485_485886

-- Definitions based on the conditions
def numPoints := 100
def numDiametricallyOppositePairs := numPoints / 2
def rightTrianglesPerPair := numPoints - 2
def totalRightTriangles := numDiametricallyOppositePairs * rightTrianglesPerPair

-- Theorem stating the final evaluation of the problem
theorem right_triangles_not_1000 :
  totalRightTriangles ≠ 1000 :=
by
  -- calculation shows it's impossible
  sorry

end right_triangles_not_1000_l485_485886


namespace min_area_sum_of_k_l485_485870

-- Definition of the points and condition
def pointA := (2, 9)
def pointB := (14, 18)
def pointC (k : ℤ) := (6, k)

-- Question: Prove that the minimum area sum of k values is 24
theorem min_area_sum_of_k : 
  let k_values := {k : ℤ | k = 11 ∨ k = 13} in
  ∑ k in k_values, k = 24 := by
sorry

end min_area_sum_of_k_l485_485870


namespace gcd_45_75_l485_485119

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485119


namespace minimum_value_a_plus_2b_l485_485260

theorem minimum_value_a_plus_2b {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : 2 * a + b - a * b = 0) : a + 2 * b = 9 :=
by sorry

end minimum_value_a_plus_2b_l485_485260


namespace jerry_age_l485_485813

theorem jerry_age (M J : ℤ) (h1 : M = 16) (h2 : M = 2 * J - 8) : J = 12 :=
by
  sorry

end jerry_age_l485_485813


namespace weight_of_A_l485_485447

theorem weight_of_A (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : (B + C + D + E) / 4 = 79) 
  (h4 : E = D + 7): 
  A = 79 := by
  have h5 : A + B + C = 252 := by
    linarith [h1]
  have h6 : A + B + C + D = 320 := by
    linarith [h2]
  have h7 : B + C + D + E = 316 := by
    linarith [h3]
  have hD : D = 68 := by
    linarith [h5, h6]
  have hE : E = 75 := by
    linarith [hD, h4]
  have hBC : B + C = 252 - A := by
    linarith [h5]
  have : 252 - A + 68 + 75 = 316 := by
    linarith [h7, hBC, hD, hE]
  linarith

end weight_of_A_l485_485447


namespace katherine_savings_multiple_l485_485583

variable (A K : ℕ)

theorem katherine_savings_multiple
  (h1 : A + K = 750)
  (h2 : A - 150 = 1 / 3 * K) :
  2 * K / A = 3 :=
sorry

end katherine_savings_multiple_l485_485583


namespace cube_root_of_product_equals_integer_l485_485431

theorem cube_root_of_product_equals_integer :
    (∛(2^6 * 3^3 * 11^3) = 132) :=
  sorry

end cube_root_of_product_equals_integer_l485_485431


namespace problem_one_problem_two_l485_485297

-- Define the function f(x) = ax + ln(x) for x in [1, e]
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
-- Define the range of x being [1, e]
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ Real.exp 1

-- Proof problem statements
theorem problem_one (a : ℝ) : 
  (a = 1) → (∀ x, domain x → f a x ≤ f a (Real.exp 1)) ∧ (f a (Real.exp 1) = Real.exp 1 + 1) :=
by sorry

theorem problem_two (a : ℝ) :
  (∀ x, domain x → f a x ≤ 0) → (a ≤ -1/Real.exp 1) :=
by sorry

end problem_one_problem_two_l485_485297


namespace equal_numbers_on_circle_l485_485415

theorem equal_numbers_on_circle (n : ℕ) (a : ℤ → ℝ)
  (h : ∀ i, a i = (a (i - 1) + a (i + 1)) / 2) :
  ∃ c, ∀ i, a i = c :=
begin
  sorry
end

end equal_numbers_on_circle_l485_485415


namespace chadsRopeLength_l485_485027

-- Define the constants and conditions
def joeysRopeLength : ℕ := 56
def joeyChadRatioNumerator : ℕ := 8
def joeyChadRatioDenominator : ℕ := 3

-- Prove that Chad's rope length is 21 cm
theorem chadsRopeLength (C : ℕ) 
  (h_ratio : joeysRopeLength * joeyChadRatioDenominator = joeyChadRatioNumerator * C) : 
  C = 21 :=
sorry

end chadsRopeLength_l485_485027


namespace increase_by_percentage_l485_485498

-- Define the initial number.
def initial_number : ℝ := 75

-- Define the percentage increase as a decimal.
def percentage_increase : ℝ := 1.5

-- Define the expected final result after applying the increase.
def expected_result : ℝ := 187.5

-- The proof statement.
theorem increase_by_percentage : initial_number * (1 + percentage_increase) = expected_result :=
by
  sorry

end increase_by_percentage_l485_485498


namespace cosine_of_angle_between_lines_l485_485408

open Real

-- Define the vectors corresponding to the direction of the lines
def vec1 : ℝ × ℝ × ℝ := (4, 5, -3)
def vec2 : ℝ × ℝ × ℝ := (2, 6, -1)

-- Compute the dot product
def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

-- Compute the norm of a vector
def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2 + v.3^2)

-- Compute cos θ
def cos_theta (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (norm v1 * norm v2)

-- The main theorem
theorem cosine_of_angle_between_lines :
  cos_theta vec1 vec2 = 41 / sqrt 2050 :=
by
  sorry

end cosine_of_angle_between_lines_l485_485408


namespace circle_E_radius_sum_l485_485989

noncomputable def radius_A := 15
noncomputable def radius_B := 5
noncomputable def radius_C := 3
noncomputable def radius_D := 3

-- We need to find that the sum of m and n for the radius of circle E is 131.
theorem circle_E_radius_sum (m n : ℕ) (h1 : Nat.gcd m n = 1) (radius_E : ℚ := (m / n)) :
  m + n = 131 :=
  sorry

end circle_E_radius_sum_l485_485989


namespace append_five_to_two_digit_l485_485730

theorem append_five_to_two_digit (t u : ℕ) (ht : t < 10) (hu : u < 10) :
  let original_number := 10 * t + u in
  let new_number := original_number * 10 + 5 in
  new_number = 100 * t + 10 * u + 5 :=
by
  intros
  sorry

end append_five_to_two_digit_l485_485730


namespace expression_value_l485_485209

theorem expression_value (c : ℝ) (h : c = 1) :
  (1 + c + 1 / 1) * (1 + c + 1 / 2) * (1 + c + 1 / 3) * (1 + c + 1 / 4) * (1 + c + 1 / 5) = 133 / 20 :=
by
  rw h
  sorry

end expression_value_l485_485209


namespace price_decrease_percentage_l485_485332

theorem price_decrease_percentage (P₀ P₁ P₂ : ℝ) (x : ℝ) :
  P₀ = 1 → P₁ = P₀ * 1.25 → P₂ = P₁ * (1 - x / 100) → P₂ = 1 → x = 20 :=
by
  intros h₀ h₁ h₂ h₃
  sorry

end price_decrease_percentage_l485_485332


namespace find_angle_B_min_dot_product_l485_485765

-- Define the initial conditions
variables {a b c : ℝ} {A B C : ℝ}

-- Given condition: (2a - c) * cos B = b * cos C
axiom given_condition : (2 * a - c) * Real.cos B = b * Real.cos C

-- Definition of angles in a triangle constraint
axiom angle_triangle_constraint : A + B + C = Real.pi

-- Vectors m and n given
def m := (Real.sin A, 1)
def n := (-1, 1)

-- Dot product of m and n
def dot_product := m.1 * n.1 + m.2 * n.2

-- Theorem stating that B equals pi / 3 under given_condition
theorem find_angle_B : given_condition → angle_triangle_constraint → B = Real.pi / 3 := by
  sorry

-- Theorem for finding minimum value of the dot product
theorem min_dot_product : angle_triangle_constraint → B = Real.pi / 3 → ∃ A, A = Real.pi / 2 ∧ dot_product = 0 := by
  sorry

end find_angle_B_min_dot_product_l485_485765


namespace difference_30th_28th_triangular_l485_485971

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_30th_28th_triangular :
  triangular_number 30 - triangular_number 28 = 59 :=
by
  sorry

end difference_30th_28th_triangular_l485_485971


namespace salary_change_commute_l485_485832

theorem salary_change_commute :
  ∀ (x : ℝ),
    let initial_salary := 37500 in
    let final_salary_with_given_order := initial_salary * (1 + x / 100) ^ 2 * (1 - 2 * x / 100) in
    let final_salary_with_reversed_order := initial_salary * (1 - 2 * x / 100) * (1 + x / 100) ^ 2 in
    final_salary_with_given_order = 34825 →
    final_salary_with_reversed_order = 34825 := 
by
  intros x initial_salary final_salary_with_given_order final_salary_with_reversed_order h
  rw [final_salary_with_given_order, final_salary_with_reversed_order]
  assumption
  sorry

end salary_change_commute_l485_485832


namespace find_ages_l485_485034

theorem find_ages (F S : ℕ) (h1 : F + 2 * S = 110) (h2 : 3 * F = 186) :
  F = 62 ∧ S = 24 := by
  sorry

end find_ages_l485_485034


namespace gcd_45_75_l485_485111

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485111


namespace jeffrey_steps_l485_485506

theorem jeffrey_steps (distance : ℕ) (forward_steps : ℕ) (backward_steps : ℕ) (effective_distance : ℤ) 
  (total_actual_steps : ℤ) (h1 : forward_steps = 3) (h2 : backward_steps = 2)
  (h3 : effective_distance = 1) 
  (h4 : total_distance : effective_distance * distance) :
  total_actual_steps = 330 :=
  by sorry

end jeffrey_steps_l485_485506


namespace area_of_region_l485_485900

theorem area_of_region :
  {p : ℝ × ℝ | |4 * p.1 - 12| + |3 * p.2 - 9| ≤ 6}.measure = 6 :=
sorry

end area_of_region_l485_485900


namespace partition_sets_with_equal_sum_l485_485375

theorem partition_sets_with_equal_sum (p : ℕ) (hp : Nat.Prime p) (k : ℕ) :
  (∃ n : ℕ, n % 2 = 0 ∧ (k = n * p ∨ k = n * p - 1)) ↔
  ∃ (s : Finset (Finset ℕ)), s.card = p ∧ s.sum (λ t, t.sum id) = (Finset.range k).sum id :=
sorry

end partition_sets_with_equal_sum_l485_485375


namespace gcd_45_75_l485_485105

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485105


namespace number_of_students_l485_485436

/--
Statement: Several students are seated around a circular table. 
Each person takes one piece from a bag containing 120 pieces of candy 
before passing it to the next. Chris starts with the bag, takes one piece 
and also ends up with the last piece. Prove that the number of students
at the table could be 7 or 17.
-/
theorem number_of_students (n : Nat) (h : 120 > 0) :
  (∃ k, 119 = k * n ∧ n ≥ 1) → (n = 7 ∨ n = 17) :=
by
  sorry

end number_of_students_l485_485436


namespace quantile_75_l485_485337

def precipitation_data : List ℕ := [46, 48, 51, 53, 53, 56, 56, 56, 58, 64, 66, 71]

def quantile_index(data : List ℕ) (q : ℚ) : ℕ :=
  (q * data.length).to_nat

def average (a b : ℕ) : ℚ := (a + b) / 2

def quantile_value (data : List ℕ) (q : ℚ) : ℚ :=
  let idx := quantile_index data q
  average (data.nth_le idx sorry) (data.nth_le (idx + 1) sorry)

theorem quantile_75 : quantile_value precipitation_data (3/4) = 61 := 
by
  sorry

end quantile_75_l485_485337


namespace minimum_homework_assignments_for_20_points_l485_485557

theorem minimum_homework_assignments_for_20_points :
  let assignments (n : ℕ) := ((n + 3) / 4)
  (sum (assignments '' (finset.range 20).val.to_finset) = 48) :=
by 
  let assignments (n : ℕ) := (n + 3) / 4
  have h1 : finset.range 4 = {0, 1, 2, 3}, sorry
  have h2 : finset.range 8 = {0, 1, 2, 3, 4, 5, 6, 7}, sorry
  have h3 : finset.range 12 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, sorry
  have h4 : finset.range 16 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, sorry
  have h5 : finset.range 20 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, sorry
  have h_sum : sum (assignments '' {0, 1, 2, 3}) 
                + sum (assignments '' {4, 5, 6, 7}) 
                + sum (assignments '' {8, 9, 10, 11}) 
                + sum (assignments '' {12, 13, 14, 15}) 
                + sum (assignments '' {16, 17, 18, 19}) = 4 + 8 + 8 + 12 + 16 := sorry
  exact h_sum

end minimum_homework_assignments_for_20_points_l485_485557


namespace gcd_45_75_l485_485060

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485060


namespace vertical_line_divides_triangle_l485_485963

theorem vertical_line_divides_triangle (k : ℝ) :
  let triangle_area := 1 / 2 * |0 * (1 - 1) + 1 * (1 - 0) + 9 * (0 - 1)|
  let left_triangle_area := 1 / 2 * |0 * (1 - 1) + k * (1 - 0) + 1 * (0 - 1)|
  let right_triangle_area := triangle_area - left_triangle_area
  triangle_area = 4 
  ∧ left_triangle_area = 2
  ∧ right_triangle_area = 2
  ∧ (k = 5) ∨ (k = -3) → 
  k = 5 :=
by
  sorry

end vertical_line_divides_triangle_l485_485963


namespace prob_dist_and_expectation_prob_non_negative_l485_485569

-- Definitions based on problem conditions
def total_score (correct_answers : ℕ) : ℤ :=
  (correct_answers : ℤ) * 100 - (3 - correct_answers) * 100

def prob_correct : ℝ := 0.8
def prob_incorrect : ℝ := 0.2

def prob_dist (s : ℤ) : ℝ :=
  if s = -300 then prob_incorrect ^ 3 else
  if s = -100 then 3 * (prob_incorrect ^ 2) * prob_correct else
  if s = 100 then 3 * prob_incorrect * (prob_correct ^ 2) else
  if s = 300 then prob_correct ^ 3 else 0

-- Part Ⅰ: Probability distribution and mathematical expectation
theorem prob_dist_and_expectation :
  (prob_dist (-300) = 0.008) ∧
  (prob_dist (-100) = 0.096) ∧
  (prob_dist (100) = 0.384) ∧
  (prob_dist (300) = 0.512) ∧
  (∑ s in {-300, -100, 100, 300}, s * prob_dist s) = 180 := sorry

-- Part Ⅱ: Probability of a non-negative score
theorem prob_non_negative :
  (∑ s in {100, 300}, prob_dist s) = 0.896 := sorry

end prob_dist_and_expectation_prob_non_negative_l485_485569


namespace geometric_sequence_x_value_l485_485222

theorem geometric_sequence_x_value (x : ℝ) (r : ℝ) 
  (h1 : 12 * r = x) 
  (h2 : x * r = 2 / 3) 
  (h3 : 0 < x) :
  x = 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_x_value_l485_485222


namespace number_of_unique_circles_l485_485797

def is_square (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (P Q R T : ℝ × ℝ), 
  Set.finite S ∧
  S = {P, Q, R, T} ∧
  dist P Q = dist Q R ∧ dist R T = dist T P ∧ 
  dist P R = dist Q T ∧
  dist P R = (dist P Q * real.sqrt 2)

theorem number_of_unique_circles (S : Set (ℝ × ℝ)) (h : is_square S) : 
  ∃ n : ℕ, n = 3 :=
by
  sorry

end number_of_unique_circles_l485_485797


namespace shooter_prob_l485_485953

variable (hit_prob : ℝ)
variable (miss_prob : ℝ := 1 - hit_prob)
variable (p1 : hit_prob = 0.85)
variable (independent_shots : true)

theorem shooter_prob :
  miss_prob * miss_prob * hit_prob = 0.019125 :=
by
  rw [p1]
  sorry

end shooter_prob_l485_485953


namespace minimum_output_no_loss_l485_485464

theorem minimum_output_no_loss (x : ℕ) (h₁ : 0 < x) (h₂ : x < 240) : 25 * x ≥ 3000 + 20 * x - 0.1 * x^2 ↔ x = 150 :=
by
  sorry

end minimum_output_no_loss_l485_485464


namespace factor_polynomial_l485_485997

theorem factor_polynomial (t : ℝ) : (∀ x, (4 * x ^ 2 + 9 * x + 2) = (x - t) * f x) → t = -1/4 ∨ t = -2 :=
by
  intro h
  have h1 : 4 * t ^ 2 + 9 * t + 2 = 0 := by sorry -- Factor Theorem application step
  sorry

end factor_polynomial_l485_485997


namespace largest_20_supporting_number_l485_485403

theorem largest_20_supporting_number :
  ∃ X : ℝ, X = 0.025 ∧ (∀ (a : fin 20 → ℝ), (∑ i, a i).isInt → (∃ i, |a i - 0.5| ≥ X)) :=
sorry

end largest_20_supporting_number_l485_485403


namespace paws_on_ground_are_correct_l485_485205

-- Problem statement
def num_paws_on_ground (total_dogs : ℕ) (half_on_all_fours : ℕ) (paws_on_all_fours : ℕ) (half_on_two_legs : ℕ) (paws_on_two_legs : ℕ) : ℕ :=
  half_on_all_fours * paws_on_all_fours + half_on_two_legs * paws_on_two_legs

theorem paws_on_ground_are_correct :
  let total_dogs := 12
  let half_on_all_fours := 6
  let half_on_two_legs := 6
  let paws_on_all_fours := 4
  let paws_on_two_legs := 2
  num_paws_on_ground total_dogs half_on_all_fours paws_on_all_fours half_on_two_legs paws_on_two_legs = 36 :=
by sorry

end paws_on_ground_are_correct_l485_485205


namespace determinant_matrix_zero_l485_485991

theorem determinant_matrix_zero (θ φ : ℝ) : 
  Matrix.det ![
    ![0, Real.cos θ, -Real.sin θ],
    ![-Real.cos θ, 0, Real.cos φ],
    ![Real.sin θ, -Real.cos φ, 0]
  ] = 0 := by sorry

end determinant_matrix_zero_l485_485991


namespace find_complement_l485_485306

-- Define predicate for a specific universal set U and set A
def universal_set (a : ℤ) (x : ℤ) : Prop :=
  x = a^2 - 2 ∨ x = 2 ∨ x = 1

def set_A (a : ℤ) (x : ℤ) : Prop :=
  x = a ∨ x = 1

-- Define complement of A with respect to U
def complement_U_A (a : ℤ) (x : ℤ) : Prop :=
  universal_set a x ∧ ¬ set_A a x

-- Main theorem statement
theorem find_complement (a : ℤ) (h : a ≠ 2) : { x | complement_U_A a x } = {2} :=
by
  sorry

end find_complement_l485_485306


namespace rational_numbers_property_l485_485677

theorem rational_numbers_property (n : ℕ) (h : n > 0) :
  ∃ (a b : ℚ), a ≠ b ∧ (∀ k, 1 ≤ k ∧ k ≤ n → ∃ m : ℤ, a^k - b^k = m) ∧ 
  ∀ i, (a : ℝ) ≠ i ∧ (b : ℝ) ≠ i :=
sorry

end rational_numbers_property_l485_485677


namespace projection_of_c_onto_a_l485_485286

variables (a b : ℝ → ℝ → ℝ)
variable [is_unit_vector : ∀ v, ∥v∥ = 1]
variable [inner_product : ∀ v w, ∀ v w, real.arccos ⟪v, w⟫ = π / 3]
def c := λ x y, a x y - b x y

theorem projection_of_c_onto_a (a b : ℝ → ℝ → ℝ) [is_unit_vector] [inner_product] :
  (λ x y, (c x y) • a x y) = 1 / 2 * (λ x y, a x y) := sorry

end projection_of_c_onto_a_l485_485286


namespace graph_iso_prime_l485_485659

-- Define finite simple graph
structure SimpleGraph (V : Type) :=
(adj : V → V → Prop)
(sym : symmetric adj . obviously)  -- symmetry of edge relation
(loop_free : irreflexive adj . obviously)  -- no loops

-- Define the graph operation G'
def graph_prime {V : Type} (G : SimpleGraph V) : SimpleGraph V :=
{ adj := λ u v, ∃ w, u ≠ v ∧ G.adj u w ∧ G.adj v w,
  sym := by finish [symmetric],
  loop_free := by finish }

-- Graph isomorphism
def GraphIsom {V : Type} (G H : SimpleGraph V) :=
  ∃ (f : V → V), bijective f ∧ (∀ u v, G.adj u v ↔ H.adj (f u) (f v))

-- Main theorem
theorem graph_iso_prime {V : Type} (G : SimpleGraph V) :
  GraphIsom G (graph_prime (graph_prime G)) → GraphIsom G (graph_prime G) :=
sorry

end graph_iso_prime_l485_485659


namespace value_of_x_l485_485477

theorem value_of_x (w : ℝ) (hw : w = 90) (z : ℝ) (hz : z = 2 / 3 * w) (y : ℝ) (hy : y = 1 / 4 * z) (x : ℝ) (hx : x = 1 / 2 * y) : x = 7.5 :=
by
  -- Proof skipped; conclusion derived from conditions
  sorry

end value_of_x_l485_485477


namespace cover_6x6_with_L_pieces_l485_485914

-- Define the structure of an L-shaped piece using three squares.
structure L_Piece :=
(x1 y1 x2 y2 x3 y3 : Nat)
(h1 : (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x2, y2) ≠ (x3, y3))
(valid_positions : ∀ (i j : Nat), i < 3 → j < 3 → (i, j) ∈ {(x1, y1), (x2, y2), (x3, y3)})

-- The L_Piece pieces given
def pieces : List L_Piece := sorry -- Define the twelve pieces here.

-- Define the 6x6 grid
def grid : List (Nat × Nat) := 
  [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5),
   (1,0), (1,1), (1,2), (1,3), (1,4), (1,5),
   (2,0), (2,1), (2,2), (2,3), (2,4), (2,5),
   (3,0), (3,1), (3,2), (3,3), (3,4), (3,5),
   (4,0), (4,1), (4,2), (4,3), (4,4), (4,5),
   (5,0), (5,1), (5,2), (5,3), (5,4), (5,5)]

-- Define the target property to cover the grid correctly
def covers_correctly (pieces : List L_Piece) (grid : List (Nat × Nat)) : Prop :=
  covers_all_squares pieces grid ∧ no_2x3_rectangles_two_pieces pieces 

-- Formalize the proof problem
theorem cover_6x6_with_L_pieces : 
  ∃ pieces : List L_Piece, covers_correctly pieces grid := sorry 

end cover_6x6_with_L_pieces_l485_485914


namespace repeating_decimal_as_fraction_l485_485639

theorem repeating_decimal_as_fraction (x : ℝ) (h1 : x = 0.28282828...) : x = 28 / 99 :=
sorry

end repeating_decimal_as_fraction_l485_485639


namespace count_valid_m_l485_485250

theorem count_valid_m :
  {m : ℕ // 0 < m ∧ ∃ k : ℕ, 1764 = (m^2 - 3) * k} = {2, 3, 4, 7, 8} :=
by
  sorry

end count_valid_m_l485_485250


namespace dot_product_computation_l485_485689

open Real

variables (a b : ℝ) (θ : ℝ)

noncomputable def dot_product (u v : ℝ) : ℝ :=
  u * v * cos θ

noncomputable def magnitude (v : ℝ) : ℝ :=
  abs v

theorem dot_product_computation (a b : ℝ) (h1 : θ = 120) (h2 : magnitude a = 4) (h3 : magnitude b = 4) :
  dot_product b (3 * a + b) = -8 :=
by
  sorry

end dot_product_computation_l485_485689


namespace system_of_inequalities_l485_485013

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end system_of_inequalities_l485_485013


namespace erased_number_average_l485_485025

theorem erased_number_average {n k : ℕ}
  (h1 : n % 2 = 1)
  (h2 : (∑ i in Finset.range (n + 1), i - k) / (n - 1) = 22) :
  n = 43 ∧ k = 22 :=
by
  sorry

end erased_number_average_l485_485025


namespace modulus_of_z_l485_485695

-- Define the given complex number z
def z : ℂ := 2 / (1 - complex.sqrt 3 * complex.i)

-- State the theorem that |z| = 1
theorem modulus_of_z : |z| = 1 := by
  sorry

end modulus_of_z_l485_485695


namespace nested_g_evaluation_l485_485376

def g (x : ℝ) : ℝ :=
if x ≥ 0 then -x^3
else x + 10

theorem nested_g_evaluation : g (g (g (g (g 2)))) = -8 := 
by 
  -- use sorry to skip the proof
  sorry

end nested_g_evaluation_l485_485376


namespace variance_of_2X_minus_1_l485_485852

noncomputable def variance_of_transformed_variable (X : Fin 3 → ℝ) (P : Fin 3 → ℝ) : ℝ :=
  let a := 2
  let b := -1
  let μ := P 0 * X 0 + P 1 * X 1 + P 2 * X 2
  let D_X := P 0 * (X 0 - μ) ^ 2 + P 1 * (X 1 - μ) ^ 2 + P 2 * (X 2 - μ) ^ 2
  a ^ 2 * D_X

theorem variance_of_2X_minus_1 {p : ℝ} (h : 0.3 + p + 0.3 = 1) : 
  variance_of_transformed_variable (λ i, if i = 0 then 0 else if i = 1 then 1 else 2) (λ i, if i = 0 then 0.3 else if i = 1 then p else 0.3) = 2.4 :=
by
  have p_value : p = 0.4 := by linarith [h]
  sorry

end variance_of_2X_minus_1_l485_485852


namespace sum_of_p_for_circumcenter_on_Ox_l485_485454

/--
Given the quadratic equation: y = 2^p * x^2 + 5 * p * x - 2^(p^2)
and the triangle ABC formed by the intersections with the axes,
find the sum of all values of the parameter p for which the center of the
circle circumscribing the triangle ABC lies on the Ox axis.
-/
theorem sum_of_p_for_circumcenter_on_Ox {p : ℝ} :
  ∑ p ∈ {p | ∃ (x1 x2 : ℝ), 
        (2^p * x1^2 + 5*p*x1 - 2^(p^2) = 0) ∧ 
        (2^p * x2^2 + 5*p*x2 - 2^(p^2) = 0) ∧ 
        ∃ (C : ℝ × ℝ), 
          C = (0, -2^(p^2)) ∧ 
          ((-2^(p^2) / x1) * (-2^(p^2) / x2) = -1)},
    p = -1 :=
begin
  sorry
end

end sum_of_p_for_circumcenter_on_Ox_l485_485454


namespace gcd_45_75_l485_485112

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485112


namespace inequality_solution_l485_485010

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end inequality_solution_l485_485010


namespace smallest_abs_value_of_z_for_given_condition_l485_485389

noncomputable def smallest_possible_value_of_abs_z : ℂ → ℝ
| z := by sorry

theorem smallest_abs_value_of_z_for_given_condition (z : ℂ) 
  (h : abs (z - 15) + abs (z + 6 * Complex.I) = 22) : 
  smallest_possible_value_of_abs_z z = 45 / 11 := by sorry

end smallest_abs_value_of_z_for_given_condition_l485_485389


namespace pascals_triangle_even_count_15_l485_485613

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

def is_even (x : ℕ) : Prop := x % 2 = 0

def count_even_in_row (n : ℕ) : ℕ :=
  (finset.range (n + 1)).count (λ k, is_even (pascal n k))

def count_even_in_first_15_rows : ℕ :=
  (finset.range 15).sum count_even_in_row

theorem pascals_triangle_even_count_15 :
  count_even_in_first_15_rows = 64 :=
by sorry

end pascals_triangle_even_count_15_l485_485613


namespace find_vector_at_t4_l485_485160

def vector_at (t : ℝ) (a d : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := a
  let (dx, dy, dz) := d
  (x + t * dx, y + t * dy, z + t * dz)

theorem find_vector_at_t4 :
  ∀ (a d : ℝ × ℝ × ℝ),
    vector_at (-2) a d = (2, 6, 16) →
    vector_at 1 a d = (-1, -5, -10) →
    vector_at 4 a d = (-16, -60, -140) :=
by
  intros a d h1 h2
  sorry

end find_vector_at_t4_l485_485160


namespace xy_product_given_conditions_l485_485048

variable (x y : ℝ)

theorem xy_product_given_conditions (hx : x - y = 5) (hx3 : x^3 - y^3 = 35) : x * y = -6 :=
by
  sorry

end xy_product_given_conditions_l485_485048


namespace sum_of_remainders_l485_485576

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 11) : (n % 4) + (n % 5) = 4 :=
by
  -- sorry is here to skip the actual proof as per instructions
  sorry

end sum_of_remainders_l485_485576


namespace cannot_obtain_100_pieces_l485_485951

theorem cannot_obtain_100_pieces : ¬ ∃ n : ℕ, 1 + 2 * n = 100 := by
  sorry

end cannot_obtain_100_pieces_l485_485951


namespace possible_degrees_of_remainder_l485_485508

theorem possible_degrees_of_remainder (f : ℕ → ℕ) (p q r : polynomial ℕ) (h_divisor : q = 3 * X^2 - 4 * X + 9) 
  (h_degree_q : q.degree = 2) (h_remainder_deg : r < q): 
  r.degree < q.degree → (r.degree = 0 ∨ r.degree = 1) :=
by
  have h_dq : q.degree = 2, from h_degree_q
  sorry

end possible_degrees_of_remainder_l485_485508


namespace sum_perpendiculars_is_leg_length_l485_485212

theorem sum_perpendiculars_is_leg_length
  (A B C P D E F : Point)
  (s : ℝ)
  (h_triangle : isosceles_right_triangle A B C)
  (h_AB_AC_eq_s : distance A B = s ∧ distance A C = s)
  (h_BC_perpendicular : perpendicular_from_point(B, P, C, E))
  (h_CA_perpendicular : perpendicular_from_point(C, P, A, F))
  (h_AB_perpendicular : perpendicular_from_point(A, P, B, D))
  (h_angle : ∠BAC = 90)
  (h_point_in_triangle : P_triangle_interior A B C P) : 
  distance P D + distance P E + distance P F = s := 
by 
  sorry

end sum_perpendiculars_is_leg_length_l485_485212


namespace gcd_45_75_l485_485094

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485094


namespace circumradius_bounds_l485_485141

-- Definitions of points on a square
structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def circumradius (P Q R: Point) : ℝ :=
  let a := (Q.x - P.x)^2 + (Q.y - P.y)^2
  let b := (R.x - Q.x)^2 + (R.y - Q.y)^2
  let c := (P.x - R.x)^2 + (P.y - R.y)^2
  (a + b + c) / 4

noncomputable def is_on_side (P : Point) (a b : Point) : Prop :=
  (P.y = a.y ∧ P.y = b.y ∧ a.x ≤ P.x ∧ P.x ≤ b.x) ∨
  (P.x = a.x ∧ P.x = b.x ∧ a.y ≤ P.y ∧ P.y ≤ b.y)

theorem circumradius_bounds (P Q R: Point):
  is_on_side P ⟨0, 1⟩ ⟨1, 1⟩ → 
  is_on_side Q ⟨0, 1⟩ ⟨1, 1⟩ → 
  is_on_side R ⟨0, 0⟩ ⟨1, 0⟩ → 
  (0.5 < circumradius P Q R ∧ circumradius P Q R ≤ sqrt 2 / 2) :=
by
  sorry

end circumradius_bounds_l485_485141


namespace amy_hourly_rate_l485_485194

theorem amy_hourly_rate (hours_worked tips total_earnings : ℝ) (h1 : hours_worked = 7) (h2 : tips = 9) (h3 : total_earnings = 23) : 
  let earnings_from_hourly_wage := total_earnings - tips in
  let hourly_rate := earnings_from_hourly_wage / hours_worked in
  hourly_rate = 2 := by
  sorry

end amy_hourly_rate_l485_485194


namespace circle_equation_with_diameter_PQ_l485_485308

theorem circle_equation_with_diameter_PQ :
  let P := (4 : ℝ, 0 : ℝ)
  let Q := (0 : ℝ, 2 : ℝ)
  let center := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let radius := real.sqrt (((Q.1 - P.1) ^ 2) + ((Q.2 - P.2) ^ 2)) / 2
  (x y : ℝ), (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2 :=
sorry

end circle_equation_with_diameter_PQ_l485_485308


namespace height_of_cylinder_l485_485851

noncomputable def pi := Real.pi

constant diameter : ℝ
constant volume : ℝ

axiom diameter_eq_six : diameter = 6
axiom volume_eq_45 : volume = 45

theorem height_of_cylinder :
  ∃ height : ℝ, height = volume / (pi * (diameter / 2) ^ 2) :=
by
  -- Defining radius from diameter
  let radius := diameter / 2
  -- The height formula derived from given conditions
  have height : height = volume / (pi * radius ^ 2) := sorry
  -- Using given conditions to conclude height
  use height
  split
  exact height
  sorry

end height_of_cylinder_l485_485851


namespace bananas_oranges_equiv_l485_485203

def bananas_apples_equiv (x y : ℕ) : Prop :=
  4 * x = 3 * y

def apples_oranges_equiv (w z : ℕ) : Prop :=
  9 * w = 5 * z

theorem bananas_oranges_equiv (x y w z : ℕ) (h1 : bananas_apples_equiv x y) (h2 : apples_oranges_equiv y z) :
  bananas_apples_equiv 24 18 ∧ apples_oranges_equiv 18 10 :=
by sorry

end bananas_oranges_equiv_l485_485203


namespace probability_within_two_units_of_origin_l485_485560

noncomputable def probability_within_circle : ℝ :=
  let side := 6
  let square_area := side * side
  let radius := 2
  let circle_area := π * radius * radius
  circle_area / square_area

theorem probability_within_two_units_of_origin :
  probability_within_circle = π / 9 :=
by
  sorry

end probability_within_two_units_of_origin_l485_485560


namespace intersection_complement_equiv_subsets_l485_485664

variable (U : Type) [Universe U]
variables (A B : Set U)

theorem intersection_complement_equiv_subsets :
  (A ∩ B = A) ↔ (U \ B ⊆ U \ A) := 
sorry

end intersection_complement_equiv_subsets_l485_485664


namespace total_distance_is_13dot5_l485_485567

def radius_of_top_disk := 10
def radius_of_bottom_disk := 2
def decrement_per_disk := 0.5
def thickness_per_disk := 0.5
def peg_height := 5

def number_of_disks : ℕ := ((radius_of_top_disk - radius_of_bottom_disk) / decrement_per_disk + 1).to_nat
def height_of_disks : ℝ := number_of_disks * thickness_per_disk
def total_height : ℝ := height_of_disks + peg_height

theorem total_distance_is_13dot5 :
  total_height = 13.5 := by
  sorry

end total_distance_is_13dot5_l485_485567


namespace time_to_write_all_rearrangements_in_hours_l485_485412

/-- Michael's name length is 7 (number of unique letters) -/
def name_length : Nat := 7

/-- Michael can write 10 rearrangements per minute -/
def write_rate : Nat := 10

/-- Number of rearrangements of Michael's name -/
def num_rearrangements : Nat := (name_length.factorial)

theorem time_to_write_all_rearrangements_in_hours :
  (num_rearrangements / write_rate : ℚ) / 60 = 8.4 := by
  sorry

end time_to_write_all_rearrangements_in_hours_l485_485412


namespace tan_sec_relation_l485_485596

theorem tan_sec_relation : 
  (let tan30sq := (1 / 3: ℝ), sec30sq := (4 / 3: ℝ) in 
  (tan30sq - sec30sq) / (tan30sq * sec30sq) = - (9 / 4)) :=
by
  sorry

end tan_sec_relation_l485_485596


namespace man_born_in_1936_l485_485937

noncomputable def year_of_birth (x : ℕ) : ℕ :=
  x^2 - 2 * x

theorem man_born_in_1936 :
  ∃ x : ℕ, x < 50 ∧ year_of_birth x < 1950 ∧ year_of_birth x = 1892 :=
by
  sorry

end man_born_in_1936_l485_485937


namespace interval_satisfies_ineq_l485_485016

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end interval_satisfies_ineq_l485_485016


namespace number_of_monic_quadratic_polynomials_l485_485650

theorem number_of_monic_quadratic_polynomials :
  let p := 125^48 in
  ∃! (polynomials : List (ℚ[X])),
    (∀ (q : ℚ[X]), q ∈ polynomials → q.leadingCoeff = 1 ∧ 
      ∃ (a b : ℕ), a ≠ b ∧ 
        (q = X^2 - (5^a + 5^b)*X + 5^(a + b)) ∧
        ((5^a + 5^b ≤ p) ∧ (5^(a + b) ≤ p))) ∧
    polynomials.length = 5112 :=
sorry

end number_of_monic_quadratic_polynomials_l485_485650


namespace larger_number_l485_485457

theorem larger_number (L S : ℕ) (h1 : L - S = 1345) (h2 : L = 6 * S + 15) : L = 1611 :=
by
  sorry

end larger_number_l485_485457


namespace impossible_friend_distribution_l485_485038

theorem impossible_friend_distribution (students : ℕ) 
    (students_with_3_friends : ℕ) (students_with_4_friends : ℕ) 
    (students_with_5_friends : ℕ) 
    (total_students : ℕ) :
    total_students = 30 ∧ 
    students_with_3_friends = 9 ∧ 
    students_with_4_friends = 11 ∧ 
    students_with_5_friends = 10 
    → ¬∃ (d : ℕ), d = (students_with_3_friends * 3 + students_with_4_friends * 4 + students_with_5_friends * 5) ∧ d % 2 = 0 :=
by
  intros h
  cases h with ht h9
  cases h9 with h4 h5
  sorry

end impossible_friend_distribution_l485_485038


namespace edge_length_of_cube_l485_485941

theorem edge_length_of_cube (l w h : ℝ) (extra_vol : ℝ) (a : ℝ) :
  l = 27 ∧ w = 18 ∧ h = 12 ∧ extra_vol = 18 ∧ 
  (a = (27 * 18 * 12 + extra_vol) ^ (1/3) * 100) →
  a ≈ 1802 :=
by
  intros
  sorry

end edge_length_of_cube_l485_485941


namespace largest_4_digit_congruent_to_7_mod_19_l485_485120

theorem largest_4_digit_congruent_to_7_mod_19 : 
  ∃ x, (x % 19 = 7) ∧ 1000 ≤ x ∧ x < 10000 ∧ x = 9982 :=
by
  sorry

end largest_4_digit_congruent_to_7_mod_19_l485_485120


namespace min_value_of_f_solve_inequality_l485_485699

noncomputable def f (x : ℝ) : ℝ := abs (x - 5/2) + abs (x - 1/2)

theorem min_value_of_f : (∀ x : ℝ, f x ≥ 2) ∧ (∃ x : ℝ, f x = 2) := by
  sorry

theorem solve_inequality (x : ℝ) : (f x ≤ x + 4) ↔ (-1/3 ≤ x ∧ x ≤ 7) := by
  sorry

end min_value_of_f_solve_inequality_l485_485699


namespace divisible_by_p_l485_485246

theorem divisible_by_p {p m n : ℕ} (hp : Nat.Prime p) (hp_gt_3 : p > 3)
    (h : (∑ i in Finset.range (p - 1) \ {0, 1}, (p-i.succ) / i.succ) - 1 = m / n)
    (h_irr : Nat.coprime m n) : p ∣ m :=
by
  sorry

end divisible_by_p_l485_485246


namespace part1_monotonic_increasing_interval_part2_range_of_t_l485_485296

def f (x : ℝ) : ℝ := sqrt 3 * (sin x) ^ 2 + sin x * cos x

theorem part1_monotonic_increasing_interval 
  : ∀ k : ℤ, ∀ x : ℝ, 
    (-π/12 + (k:ℝ) * π ≤ x ∧ x ≤ 5 * π / 12 + (k:ℝ) * π) → 
    monotone_on f (-π/12 + (k:ℝ) * π) (5 * π / 12 + (k:ℝ) * π) := 
sorry

theorem part2_range_of_t
  : ∀ x : ℝ, (∀ t : ℝ, (t ∈ set.Icc 0 (π/3))
    → (x ∈ set.Icc t (π / 3))
    → abs (f x - sqrt 3 / 2) ≤ sqrt 3 / 2)
    → t ∈ set.Ico 0 (π / 3) :=
sorry

end part1_monotonic_increasing_interval_part2_range_of_t_l485_485296


namespace least_integer_value_l485_485648

theorem least_integer_value (x : ℤ) : 3 * abs x + 4 < 19 → x = -4 :=
by
  intro h
  sorry

end least_integer_value_l485_485648


namespace number_of_boys_in_class_l485_485190

-- Definitions and conditions
def circle_positions : Type := ℕ

variable (n : circle_positions) -- Total number of boys in the class
variable (b1 b2 : circle_positions) -- Positions of the boys in the class

-- Given conditions
def in_circle (n : circle_positions) (b1 b2 : circle_positions) : Prop :=
  b1 = 10 ∧ b2 = 45 ∧ (b2 - b1) * 2 = n

-- Statement to be proved
theorem number_of_boys_in_class : ∃ n, in_circle n 10 45 :=
begin
  use 70,
  unfold in_circle,
  split,
  { refl },
  split,
  { refl },
  simp,
  sorry
end

end number_of_boys_in_class_l485_485190


namespace axis_of_symmetry_cosine_l485_485854

theorem axis_of_symmetry_cosine (x : ℝ) : 
  (∃ k : ℤ, 2 * x + π / 3 = k * π) → x = -π / 6 :=
sorry

end axis_of_symmetry_cosine_l485_485854


namespace max_candy_for_one_student_l485_485960

-- Definition of the conditions
variables {students : ℕ} {average_candies : ℕ} {min_candies : ℕ}

-- Definition under given conditions
def conditions := students = 30 ∧ average_candies = 7 ∧ min_candies = 1

-- Theorem statement
theorem max_candy_for_one_student (h : conditions) : ∃ max_candies : ℕ, max_candies = 181 :=
begin
  sorry
end

end max_candy_for_one_student_l485_485960


namespace length_PA_sine_dihedral_angle_l485_485353

variables (PA base ABCD F PAF PB BAFD : ℝ)

-- Length of PA
theorem length_PA :
  (PA ⊥ base ABCD) ∧ 
  (BC = 2) ∧ 
  (CD = 2) ∧
  (AC = 4) ∧ 
  (∠ ACB = π/3) ∧ 
  (∠ ACD = π/3) ∧ 
  (midpoint F PC) ∧ 
  (AF ⊥ PB) →
  PA = 2 * sqrt 3 := 
sorry

-- Sine of the dihedral angle B-AF-D
theorem sine_dihedral_angle :
  (PA ⊥ base ABCD) ∧ 
  (BC = 2) ∧ 
  (CD = 2) ∧
  (AC = 4) ∧ 
  (∠ ACB = π/3) ∧ 
  (∠ ACD = π/3) ∧ 
  (midpoint F PC) ∧ 
  (AF ⊥ PB) →
  sin (dihedral_angle BAFD) = sqrt 63 / 8 := 
sorry

end length_PA_sine_dihedral_angle_l485_485353


namespace tangent_line_eqn_extremum_range_of_a_inequality_holds_l485_485298

noncomputable def f (x : ℝ) (a : ℝ) := Math.exp x + a * Real.log (x + 1)

theorem tangent_line_eqn (a : ℝ) (h : a = -2): 
    (∃ (m : ℝ), ∃ (b : ℝ), y = m * x + b ∧ m = -1 ∧ b = 1) → 
    x + y - 1 = 0 :=
  sorry

theorem extremum_range_of_a : 
    (a < 0) ↔ (∃ x : ℝ, x > -1 ∧ ∃ f' : ℝ → ℝ, f' x = Math.exp x + a / (x + 1) ∧ f' x = 0) :=
  sorry

theorem inequality_holds (a : ℝ) : 
    (a = -2) ↔ (∀ x : ℝ, x > -1 → (∃ g : ℝ → ℝ, g x = Math.sin x + Math.exp x + a * Real.log (x + 1) ∧ g x ≥ 1 - Math.sin x)) :=
  sorry

end tangent_line_eqn_extremum_range_of_a_inequality_holds_l485_485298


namespace gcd_45_75_l485_485116

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485116


namespace average_weight_increase_l485_485743

theorem average_weight_increase (W_new : ℝ) (W_old : ℝ) (num_persons : ℝ): 
  W_new = 94 ∧ W_old = 70 ∧ num_persons = 8 → 
  (W_new - W_old) / num_persons = 3 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end average_weight_increase_l485_485743


namespace cistern_fill_time_l485_485896

theorem cistern_fill_time {t : ℕ} :
  (∀ p q : ℕ, (p = 10 ∧ q = 15) →
   (4 * (1 / p + 1 / q) = 2 / 3) ∧
   (1 / q * t = 1 / 3)) → t = 5 :=
by
  intros p q h
  cases h with h1 h2
  sorry

end cistern_fill_time_l485_485896


namespace sum_of_digits_of_190_l485_485769

/-- 
Given:
1. An exchange rate of 8 U.S. dollars to 11 Canadian dollars.
2. After spending 70 Canadian dollars, the remaining Canadian dollars equals the original U.S. dollars exchanged.
3. The adjusted condition to ensure \( d \) is an integer: \( 3d = 570 \).

Prove:
The sum of the digits of \( d \) is 10.
--/
theorem sum_of_digits_of_190 :
  ∑ x in ([1, 9, 0] : List ℕ), x = 10 := 
  by
  simp

end sum_of_digits_of_190_l485_485769


namespace even_integers_in_pascals_triangle_top_15_rows_l485_485618

/-- Prove that the total number of even integers in the top 15 rows of Pascal's Triangle is exactly 90.
  Pascal's Triangle's elements, binomial(n,k), are even unless every binary digit of k 
  is present in n when both are expressed in binary. -/
theorem even_integers_in_pascals_triangle_top_15_rows : 
  ∑ n in Finset.range 15, ∑ k in Finset.range (n + 1), if (∀ i, (n.bits.get i) = 1 → (k.bits.get i) = 1) then 0 else 1 = 90 :=
sorry

end even_integers_in_pascals_triangle_top_15_rows_l485_485618


namespace inequality_solution_set_l485_485031

theorem inequality_solution_set :
  { x : ℝ | 1 < x ∧ x < 2 } = { x : ℝ | (x - 2) / (1 - x) > 0 } :=
by sorry

end inequality_solution_set_l485_485031


namespace average_weight_of_three_l485_485428

theorem average_weight_of_three
  (rachel_weight jimmy_weight adam_weight : ℝ)
  (h1 : rachel_weight = 75)
  (h2 : jimmy_weight = rachel_weight + 6)
  (h3 : adam_weight = rachel_weight - 15) :
  (rachel_weight + jimmy_weight + adam_weight) / 3 = 72 :=
by
  sorry

end average_weight_of_three_l485_485428


namespace ratio_CP_CN_l485_485977

theorem ratio_CP_CN (r : ℝ) (B C O1 O2 A M N P : Point)
  (h1 : Circle O1 r ∩ Circle O2 r = {B, C})
  (h2 : O1 ∈ Circle O2 r ∧ O2 ∈ Circle O1 r)
  (h3 : diameter AB O1)
  (h4 : segment_intersects_circle AO2 (Circle O2 r) M N ∧ M ∈ segment A O2)
  (h5 : extension_intersects CM NB P) :
  ratio CP CN = (Real.sqrt 3, 1) :=
sorry

end ratio_CP_CN_l485_485977


namespace good_numbers_count_1_to_50_l485_485051

def is_good_number (n : ℕ) : Prop :=
  ∃ (k l : ℕ), k ≠ 0 ∧ l ≠ 0 ∧ n = k * l + l - k

theorem good_numbers_count_1_to_50 : ∃ cnt, cnt = 49 ∧ (∀ n, n ∈ (Finset.range 51).erase 0 → is_good_number n) :=
  sorry

end good_numbers_count_1_to_50_l485_485051


namespace milk_production_time_l485_485725

noncomputable def milk_production_days (y : ℕ) : ℚ := 
  y * (y + 7) / (y + 2)

theorem milk_production_time (y : ℕ) (h: y ≠ 0):  
  let daily_production_per_cow := (y + 2) / (y * (y + 4))
  let total_daily_production := (y + 4) * daily_production_per_cow
  let required_days := (y + 7) / total_daily_production in
  required_days = milk_production_days y :=
by
  sorry

end milk_production_time_l485_485725


namespace pascals_triangle_even_count_15_l485_485617

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

def is_even (x : ℕ) : Prop := x % 2 = 0

def count_even_in_row (n : ℕ) : ℕ :=
  (finset.range (n + 1)).count (λ k, is_even (pascal n k))

def count_even_in_first_15_rows : ℕ :=
  (finset.range 15).sum count_even_in_row

theorem pascals_triangle_even_count_15 :
  count_even_in_first_15_rows = 64 :=
by sorry

end pascals_triangle_even_count_15_l485_485617


namespace sum_of_coefficients_eq_3_pow_10_l485_485665

/-- Given a = (1 / π) * ∫ (sqrt (4 - x^2)) dx over [-2, 2], prove that the sum of the
  coefficients in the expansion of (∛x + a / sqrt x) ^ 10 is 3^10 -/
theorem sum_of_coefficients_eq_3_pow_10 :
  let a := (1 / Real.pi) * ∫ x in -2..2, Real.sqrt (4 - x ^ 2)
  a = 2 → (∑ k in Finset.range (11), (10.choose k) * (Real.cbrt 1 ^ k) * (2 / Real.sqrt 1) ^ (10 - k)) = 3 ^ 10 := 
by
  let a := (1 / Real.pi) * ∫ x in -2..2, Real.sqrt (4 - x ^ 2)
  assume h : a = 2
  sorry

end sum_of_coefficients_eq_3_pow_10_l485_485665


namespace osmotic_pressure_independence_l485_485052

-- definitions for conditions
def osmotic_pressure_depends_on (osmotic_pressure protein_content Na_content Cl_content : Prop) : Prop :=
  (osmotic_pressure = protein_content ∧ osmotic_pressure = Na_content ∧ osmotic_pressure = Cl_content)

-- statement of the problem to be proved
theorem osmotic_pressure_independence 
  (osmotic_pressure : Prop) 
  (protein_content : Prop) 
  (Na_content : Prop) 
  (Cl_content : Prop) 
  (mw_plasma_protein : Prop)
  (dependence : osmotic_pressure_depends_on osmotic_pressure protein_content Na_content Cl_content) :
  ¬(osmotic_pressure = mw_plasma_protein) :=
sorry

end osmotic_pressure_independence_l485_485052


namespace perfect_cubes_between_200_and_1600_l485_485713

theorem perfect_cubes_between_200_and_1600 : 
  ∃ (count : ℕ), count = (finset.filter (λ n, 200 ≤ n^3 ∧ n^3 ≤ 1600) (finset.range 50)).card := 
begin
  use 6,
  sorry,
end

end perfect_cubes_between_200_and_1600_l485_485713


namespace smallest_n_log_sum_l485_485597

theorem smallest_n_log_sum :
  ∃ (n : ℕ), (∀ (m : ℕ), (m < n) → 
    ∑ k in Finset.range (m + 1), Real.logb 3 (1 + 1 / 3^(3^k)) < 1 + Real.logb 3 (1004 / 1005)) ∧ 
    ∑ k in Finset.range (n + 1), Real.logb 3 (1 + 1 / 3^(3^k)) ≥ 1 + Real.logb 3 (1004 / 1005) :=
begin
  sorry
end

end smallest_n_log_sum_l485_485597


namespace sum_of_angles_l485_485241

theorem sum_of_angles :
  ∑ x in {x | x ∈ (set.Icc 0 360) ∧ sin x ^ 6 - cos x ^ 6 = 1 / sin x ^ 2 - 1 / cos x ^ 2}.to_finset = 720 :=
begin
  sorry
end

end sum_of_angles_l485_485241


namespace race_course_distance_l485_485516

def race_course_length (v_A v_B L : ℝ) : ℝ := 
  if v_A = 4 * v_B then L else 0

theorem race_course_distance (v_B L : ℝ) (h : v_A = 4 * v_B) (start : 75) :
  (L / v_A = (L - start) / v_B) → L = 100 :=
by
  sorry

end race_course_distance_l485_485516


namespace smallest_number_sum_of_three_squares_distinct_ways_l485_485137

theorem smallest_number_sum_of_three_squares_distinct_ways :
  ∃ n : ℤ, n = 30 ∧
  (∃ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℤ),
    a1^2 + b1^2 + c1^2 = n ∧
    a2^2 + b2^2 + c2^2 = n ∧
    a3^2 + b3^2 + c3^2 = n ∧
    (a1, b1, c1) ≠ (a2, b2, c2) ∧
    (a1, b1, c1) ≠ (a3, b3, c3) ∧
    (a2, b2, c2) ≠ (a3, b3, c3)) := sorry

end smallest_number_sum_of_three_squares_distinct_ways_l485_485137


namespace gcd_of_45_and_75_l485_485086

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l485_485086


namespace stock_percent_change_l485_485966

theorem stock_percent_change (x : ℝ) (h1 : 0 < x) :
  let first_day_value := (1 - 0.1) * x in
  let second_day_value := (1 + 0.2) * first_day_value in
  second_day_value = x * 1.08 :=
by
  sorry

end stock_percent_change_l485_485966


namespace math_problem_l485_485726

theorem math_problem
  (x y z : ℕ)
  (h1 : z = 4)
  (h2 : x + y = 7)
  (h3 : x + z = 8) :
  x + y + z = 11 := 
by
  sorry

end math_problem_l485_485726


namespace smallest_positive_x_l485_485122

theorem smallest_positive_x (x : ℝ) (h : x > 0) (h_eq : x / 4 + 3 / (4 * x) = 1) : x = 1 :=
by
  sorry

end smallest_positive_x_l485_485122


namespace additional_men_needed_l485_485157

def tunnel_construction (L D N_initial W_completed T_elapsed : ℝ) : ℝ :=
  let rate_per_man_per_day := W_completed / (N_initial * T_elapsed)
  let work_left := L - W_completed
  let days_left := D - T_elapsed
  let required_rate_per_day := work_left / days_left
  let N_required := required_rate_per_day / rate_per_man_per_day
  N_required - N_initial

theorem additional_men_needed :
  tunnel_construction 1800 450 100 600 200 = 60 := 
by 
  sorry

end additional_men_needed_l485_485157


namespace common_chord_single_point_l485_485271

/-- Given four points A, B, C, and D, 
it is known that any two circles, one passing through A and B, and another passing through C and D, intersect. 
We should prove that the common chords of all such pairs of circles pass through a single point. -/
theorem common_chord_single_point {A B C D : Point}
  (h_intersect : ∀ (σ1 σ2 : Circle), (σ1.passes_through A B) → (σ2.passes_through C D) → (intersects σ1 σ2)) :
  ∃ P : Point, ∀ (σ1 σ2 : Circle), (σ1.passes_through A B) → (σ2.passes_through C D) → 
    ((σ1.common_chord_with σ2).passes_through P) := sorry

end common_chord_single_point_l485_485271


namespace line_slope_angle_y_intercept_l485_485879

theorem line_slope_angle_y_intercept :
  ∀ (x y : ℝ), x - y - 1 = 0 → 
    (∃ k b : ℝ, y = x - 1 ∧ k = 1 ∧ b = -1 ∧ θ = 45 ∧ θ = Real.arctan k) := 
    by
      sorry

end line_slope_angle_y_intercept_l485_485879


namespace rotation_matrix_determinant_45_l485_485385

theorem rotation_matrix_determinant_45 :
  let S := Matrix![(Real.cos (Real.pi / 4)), -(Real.sin (Real.pi / 4))],
                  [(Real.sin (Real.pi / 4)), (Real.cos (Real.pi / 4))]
  in Matrix.det S = 1 :=
by
  sorry

end rotation_matrix_determinant_45_l485_485385


namespace scientific_notation_correct_l485_485910

def decimal_number : ℝ := 0.000000022
def scientific_notation : ℝ := 2.2 * 10^(-8)

theorem scientific_notation_correct :
  decimal_number = scientific_notation :=
sorry

end scientific_notation_correct_l485_485910


namespace necessary_french_woman_l485_485585

structure MeetingConditions where
  total_money_women : ℝ
  total_money_men : ℝ
  total_money_french : ℝ
  total_money_russian : ℝ

axiom no_other_representatives : Prop
axiom money_french_vs_russian (conditions : MeetingConditions) : conditions.total_money_french > conditions.total_money_russian
axiom money_women_vs_men (conditions : MeetingConditions) : conditions.total_money_women > conditions.total_money_men

theorem necessary_french_woman (conditions : MeetingConditions) :
  ∃ w_f : ℝ, w_f > 0 ∧ conditions.total_money_french > w_f ∧ w_f + conditions.total_money_men > conditions.total_money_women :=
by
  sorry

end necessary_french_woman_l485_485585


namespace largest_20_supporting_number_l485_485402

theorem largest_20_supporting_number :
  ∃ X : ℝ, X = 0.025 ∧ (∀ (a : fin 20 → ℝ), (∑ i, a i).isInt → (∃ i, |a i - 0.5| ≥ X)) :=
sorry

end largest_20_supporting_number_l485_485402


namespace kenya_peanuts_l485_485782

def jose_peanuts : ℕ := 85
def difference : ℕ := 48

theorem kenya_peanuts : jose_peanuts + difference = 133 := by
  sorry

end kenya_peanuts_l485_485782


namespace gcd_45_75_l485_485066

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485066


namespace distance_between_foci_l485_485601

-- Define the equation of the ellipse as a condition
def ellipse_eq (x y : ℝ) : Prop :=
  real.sqrt ((x - 4)^2 + (y - 5)^2) + real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

-- Define the coordinates of the foci
def f1 : (ℝ × ℝ) := (4, 5)
def f2 : (ℝ × ℝ) := (-6, 9)

-- Define the distance formula between two points
def dist (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Prove that distance between foci of the ellipse is 2 * real.sqrt 29
theorem distance_between_foci :
  (∀ (x y : ℝ), ellipse_eq x y) → dist f1 f2 = 2 * real.sqrt 29 :=
by
  intro h
  sorry

end distance_between_foci_l485_485601


namespace num_solutions_g_l485_485600

noncomputable def g : ℝ → ℝ
| x := if -5 ≤ x ∧ x ≤ -1 then -(x + 3) ^ 2 + 4
       else if -1 < x ∧ x ≤ 3 then x - 1
       else if 3 < x ∧ x ≤ 5 then (x - 4) ^ 2 + 1
       else 0

theorem num_solutions_g (h : ∀ x, -5 ≤ x ∧ x ≤ 5) :
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ g(g(x₁)) = 3 ∧ g(g(x₂)) = 3 ∧
  ∀ y, g(g(y)) = 3 → (y = x₁ ∨ y = x₂) := by
  sorry

end num_solutions_g_l485_485600


namespace smallest_odd_integer_l485_485861

theorem smallest_odd_integer (S : Set ℤ) (h1 : S = {n : ℤ | ∃ k : ℤ, n = 143 + 2*k})
  (h2 : 150 ∈ S)
  (h3 : 161 ∈ S)
  (h4 : (n : ℤ) ∈ S → ∃ m : ℤ, S = (range ((m+1) : ℕ)).map (λ x : ℕ, 143 + 2*x))
  : 143 ∈ S :=
sorry

end smallest_odd_integer_l485_485861


namespace six_divisors_third_seven_times_second_fourth_ten_more_than_third_l485_485867

theorem six_divisors_third_seven_times_second_fourth_ten_more_than_third (n : ℕ) :
  (∀ d : ℕ, d ∣ n ↔ d ∈ [1, d2, d3, d4, d5, n]) ∧ 
  (d3 = 7 * d2) ∧ 
  (d4 = d3 + 10) → 
  n = 2891 :=
by
  sorry

end six_divisors_third_seven_times_second_fourth_ten_more_than_third_l485_485867


namespace find_angle_BAC_l485_485359

-- Define the conditions for the given triangle and median
variables {α : Type*} [linear_ordered_field α] {a b m : α}

-- Define angle BAC
def angle_BAC (A B C: EuclideanGeometry.point α) : α := 
  arccos ((4 * m^2 - a^2 - b^2) / (2 * a * b))

-- Declare that AD is a median in triangle ABC
def is_median_AD (A B C D : EuclideanGeometry.point α) : Prop :=
  (∃ M, M = EuclideanGeometry.midpoint B C ∧ EuclideanGeometry.line[AD].contains M ∧ EuclideanGeometry.distance A D = m ∧ EuclideanGeometry.distance A B = a ∧ EuclideanGeometry.distance A C = b)

-- The proof statement required
theorem find_angle_BAC (A B C D: EuclideanGeometry.point α) 
  (h_median : is_median_AD A B C D) : 
  angle_BAC A B C = arccos ((4 * m^2 - a^2 - b^2) / (2 * a * b)) :=
sorry

end find_angle_BAC_l485_485359


namespace probability_difference_l485_485150

-- Definitions for probabilities
def P_plane : ℚ := 7 / 10
def P_train : ℚ := 3 / 10
def P_on_time_plane : ℚ := 8 / 10
def P_on_time_train : ℚ := 9 / 10

-- Events definitions
def P_arrive_on_time : ℚ := (7 / 10) * (8 / 10) + (3 / 10) * (9 / 10)
def P_plane_and_on_time : ℚ := (7 / 10) * (8 / 10)
def P_train_and_on_time : ℚ := (3 / 10) * (9 / 10)
def P_conditional_plane_given_on_time : ℚ := P_plane_and_on_time / P_arrive_on_time
def P_conditional_train_given_on_time : ℚ := P_train_and_on_time / P_arrive_on_time

theorem probability_difference :
  P_conditional_plane_given_on_time - P_conditional_train_given_on_time = 29 / 83 :=
by sorry

end probability_difference_l485_485150


namespace ellipse_proof_line_proof_l485_485684

section

variables {a b x y k m : ℝ}

-- Conditions:
def condition1 : Prop := ∀ (F1 F2 P : ℝ × ℝ) (a b : ℝ),
  F1 = (-1, 0) ∧ F2 = (1, 0) ∧ P = (1, 3 / 2) ∧ a > b ∧ b > 0 → 
  dist F1 P + dist F2 P = 2 * a ∧ 
  F1 = (-a, 0) ∧ F2 = (a, 0)

def condition2 : Prop := ( 0 < m ) 

def ellipse_eq : Prop :=  ∀ (a b : ℝ), a = 2 ∧ b = sqrt 3 → 
  (x^2 / a^2) + (y^2 / b^2) = 1


def line_eq : Prop := ∀ (k : ℝ ) (sqrt3_2 k^2 :ℝ)
  (m > 0), k = sqrt 3 / 2 ∧ m = sqrt 6 → 
  (y = k * x + m ) ∨ (y = -k * x + m)

-- Proof:
theorem ellipse_proof : ∀ (F1 F2 P : ℝ × ℝ) (a b : ℝ),
  condition1 F1 F2 P a b →
  F1 = (-1, 0) ∧ F2 = (1, 0) ∧ P = (1, 3 / 2) ∧ a > b ∧ b > 0 →
  ellipse_eq a b := by
  sorry

theorem line_proof : ∀ (l_eq : ℝ) (m> 0 ),
  condition2 →
  ellipse_eq 2 (sqrt 3) →
  line_eq  l_eq m := by
  sorry

end

end ellipse_proof_line_proof_l485_485684


namespace parallel_neither_sufficient_nor_necessary_l485_485307

variables (a b : Line) (α : Plane)

-- Definition of the condition
def is_in_plane (l : Line) (p : Plane) : Prop := ∃ p1, p1 ∈ p ∧ l ⊆ p1

-- Parallellity definitions
def parallel_lines (l1 l2 : Line) : Prop := ∀ p1 ∈ l1, ∀ p2 ∈ l2, ∃ p3 ∈ l1, p1 ≠ p3 ∧ collinear p1 p2 p3
def parallel_line_plane (l : Line) (p : Plane) : Prop := ∀ p1 ∈ l, ∃ q1 q2 ∈ p, collinear p1 q1 q2

-- The theorem statement
theorem parallel_neither_sufficient_nor_necessary {a b : Line} {α : Plane} 
  (h1 : is_in_plane b α) : ¬ ((parallel_lines a b) ↔ (parallel_line_plane a α)) :=
by sorry

end parallel_neither_sufficient_nor_necessary_l485_485307


namespace digit_a_prime_l485_485220

theorem digit_a_prime {A : ℕ} (hA: A < 10) : prime (130400 + A) ↔ A = 9 :=
by sorry

end digit_a_prime_l485_485220


namespace gcd_45_75_l485_485059

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485059


namespace even_count_in_top_15_rows_l485_485625

-- We want to count the number of even binomial coefficients in the first 15 rows of Pascal's Triangle.
-- Specifically, we are considering rows 0 to 14.

def is_even (n : ℕ) := n % 2 = 0

def count_even_binomials_in_pascals_triangle_up_to_row (max_row : ℕ) : ℕ := 
  ∑ n in Finset.range (max_row + 1), 
    ∑ k in Finset.range (n + 1), 
      if is_even (Nat.choose n k) then 1 else 0

theorem even_count_in_top_15_rows : count_even_binomials_in_pascals_triangle_up_to_row 14 = 49 := 
by 
  sorry

end even_count_in_top_15_rows_l485_485625


namespace largest_twenty_supporting_l485_485400

def is_twenty_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 20 → ℝ),
    (∑ i, a i : ℝ).round = (∑ i, a i : ℝ) →
    ∃ i, |a i - 1/2| ≥ X

theorem largest_twenty_supporting :
  (∃ X : ℝ, is_twenty_supporting X ∧ ∀ Y : ℝ, is_twenty_supporting Y → Y ≤ X) ∧
  ∀ Z : ℝ, is_twenty_supporting Z → abs ((X : ℝ) - 0.025) < 0.001 :=
sorry

end largest_twenty_supporting_l485_485400


namespace remaining_trees_correct_l485_485039

def initial_oak_trees := 57
def initial_maple_trees := 43

def full_cut_oak := 13
def full_cut_maple := 8

def partial_cut_oak := 2.5
def partial_cut_maple := 1.5

def remaining_oak_trees := initial_oak_trees - full_cut_oak
def remaining_maple_trees := initial_maple_trees - full_cut_maple

def total_remaining_trees := remaining_oak_trees + remaining_maple_trees

theorem remaining_trees_correct : remaining_oak_trees = 44 ∧ remaining_maple_trees = 35 ∧ total_remaining_trees = 79 :=
by
  sorry

end remaining_trees_correct_l485_485039


namespace even_count_in_top_15_rows_l485_485623

-- We want to count the number of even binomial coefficients in the first 15 rows of Pascal's Triangle.
-- Specifically, we are considering rows 0 to 14.

def is_even (n : ℕ) := n % 2 = 0

def count_even_binomials_in_pascals_triangle_up_to_row (max_row : ℕ) : ℕ := 
  ∑ n in Finset.range (max_row + 1), 
    ∑ k in Finset.range (n + 1), 
      if is_even (Nat.choose n k) then 1 else 0

theorem even_count_in_top_15_rows : count_even_binomials_in_pascals_triangle_up_to_row 14 = 49 := 
by 
  sorry

end even_count_in_top_15_rows_l485_485623


namespace no_such_n_exists_l485_485728

noncomputable def problem_conditions (n : ℝ) : Prop :=
  (1 / 4) * (1 / 3) * (2 / 5) * n = 17 ∧
  real.sqrt (0.6 * n) = (1 / 2) * real.cbrt n ∧
  real.log n = 3 * n - nat.factorial n.to_nat

theorem no_such_n_exists :
  ¬ ∃ n : ℝ, problem_conditions n :=
begin
  sorry -- Proof goes here
end

end no_such_n_exists_l485_485728


namespace exterior_angle_BAC_coplanar_triangle_pentagon_l485_485180

-- Definitions of the problem
variables {Triangle Pentagon : Type}

-- Conditions
axiom coplanar_and_common_side (T : Triangle) (P : Pentagon) (AB : ℝ) : 
  -- Some definition ensuring coplanarity and common side, e.g.
  AB ∈ T ∧ AB ∈ P

-- Statement
theorem exterior_angle_BAC_coplanar_triangle_pentagon (T : Triangle) (P : Pentagon) (AB : ℝ) :
  coplanar_and_common_side T P AB → 
  exterior_angle_of_triangle_with_pentagon_is_192 (T : Triangle) (P : Pentagon) (A B C : ℝ) : 
    angle BAC = 192 :=
by sorry

end exterior_angle_BAC_coplanar_triangle_pentagon_l485_485180


namespace max_regular_hours_l485_485929

-- Define the conditions
def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate * 1.75
def total_hours_worked : ℝ := 54
def total_earnings : ℝ := 1032

-- Prove that the maximum number of hours at the regular rate is 40
theorem max_regular_hours : 
  ∃ x : ℝ, 16 * x + overtime_rate * (total_hours_worked - x) = total_earnings ∧ x = 40 :=
by
  -- Introduce the variable x representing the maximum hours at the regular rate
  let x := 40
  -- Calculate the overtime component
  have overtime_component : ℝ := overtime_rate * (total_hours_worked - x)
  -- Assert that the total earnings equation holds
  have earnings_equation : ℝ := 16 * x + overtime_component
  -- Verify the equation
  have : earnings_equation = 1032 := sorry
  -- Conclude the proof with the assertion about x
  exact ⟨x, this, rfl⟩

end max_regular_hours_l485_485929


namespace sector_area_is_4_l485_485284

/-- Given a sector of a circle with perimeter 8 and central angle 2 radians,
    the area of the sector is 4. -/
theorem sector_area_is_4 (r l : ℝ) (h1 : l + 2 * r = 8) (h2 : l / r = 2) : 
    (1 / 2) * l * r = 4 :=
sorry

end sector_area_is_4_l485_485284


namespace students_not_coming_l485_485995

-- Define the conditions
def pieces_per_student : ℕ := 4
def pieces_made_last_monday : ℕ := 40
def pieces_made_upcoming_monday : ℕ := 28

-- Define the number of students not coming to class
theorem students_not_coming :
  (pieces_made_last_monday / pieces_per_student) - 
  (pieces_made_upcoming_monday / pieces_per_student) = 3 :=
by sorry

end students_not_coming_l485_485995


namespace sqrt_20_minus_1_range_l485_485990

theorem sqrt_20_minus_1_range : 
  16 < 20 ∧ 20 < 25 ∧ Real.sqrt 16 = 4 ∧ Real.sqrt 25 = 5 → (3 < Real.sqrt 20 - 1 ∧ Real.sqrt 20 - 1 < 4) :=
by
  intro h
  sorry

end sqrt_20_minus_1_range_l485_485990


namespace common_ratio_is_2_l485_485690

noncomputable def arithmetic_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, 2 * (a (n + 2) - a n) = 3 * a (n + 1)

theorem common_ratio_is_2 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 > 0)
  (h3 : arithmetic_sequence_common_ratio a q) :
  q = 2 :=
sorry

end common_ratio_is_2_l485_485690


namespace probability_X_geq_1_l485_485327

noncomputable def X : ℝ → ℝ := sorry

variable (X : ℝ → ℝ) 
variable (σ : ℝ)
variable [normal_distribution X (-1) σ^2]

axiom prob_interval : ∀ {a b : ℝ}, P(a ≤ X ∧ X ≤ b) = 0.4

theorem probability_X_geq_1 : P(1 ≤ X) = 0.1 :=
by
  sorry

end probability_X_geq_1_l485_485327


namespace least_integer_value_of_x_l485_485646

theorem least_integer_value_of_x (x : ℤ) (h : 3 * |x| + 4 < 19) : x = -4 :=
by sorry

end least_integer_value_of_x_l485_485646


namespace circle_radius_approximation_l485_485547

theorem circle_radius_approximation :
  ∀ (C A : ℝ), C = 50.27 → A = 201.06 → ∃ r : ℝ, r ≈ 8.00 :=
by
  intros C A hC hA
  -- Proof goes here
  sorry

end circle_radius_approximation_l485_485547


namespace incorrect_factorization_l485_485127

theorem incorrect_factorization :
    ∀ (x : ℝ), 
    (x^2 - 4 = (x + 2) * (x - 2)) ∧
    (x^2 + x * y = x * (x + y)) ∧
    (x^3 + 6 * x^2 + 9 * x = x * (x + 3)^2) ∧
    ¬(x^2 - 7 * x + 12 = x * (x - 7) + 12) := 
by
  intro x
  split
  sorry
  split
  sorry
  split
  sorry
  intro h
  sorry

end incorrect_factorization_l485_485127


namespace find_y_l485_485363

theorem find_y (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a < b ∧ b < c) :
    -2272 * a + b * (1000 + 100 * c + 10 * b + a) + 1 = 1 → 
    10^3 + 10^2 * c + 10 * b + a = 1987 :=
by
  sorry

end find_y_l485_485363


namespace find_inscribed_circle_radius_l485_485361

-- Define the necessary parameters and conditions
def side_length_square : ℝ := 15
def triangle_side_length (s : ℝ) : Prop := s = (15 * Real.sqrt 6 - 15 * Real.sqrt 2) / 2
def inscribed_circle_radius (r : ℝ) : Prop := r = 7.5 - (1 / 4) * (15 * Real.sqrt 6 - 15 * Real.sqrt 2)

theorem find_inscribed_circle_radius :
  ∃ s r, triangle_side_length s ∧ inscribed_circle_radius r :=
by
  use (15 * Real.sqrt 6 - 15 * Real.sqrt 2) / 2
  use 7.5 - (1 / 4) * (15 * Real.sqrt 6 - 15 * Real.sqrt 2)
  split
  · simp [triangle_side_length]
  · simp [inscribed_circle_radius]
  sorry

end find_inscribed_circle_radius_l485_485361


namespace total_cost_of_cultivating_field_l485_485842

theorem total_cost_of_cultivating_field 
  (base height : ℕ) 
  (cost_per_hectare : ℝ) 
  (base_eq: base = 3 * height) 
  (height_eq: height = 300) 
  (cost_eq: cost_per_hectare = 24.68) 
  : (1/2 : ℝ) * base * height / 10000 * cost_per_hectare = 333.18 :=
by
  sorry

end total_cost_of_cultivating_field_l485_485842


namespace table_filling_impossible_l485_485972

theorem table_filling_impossible :
  ∀ (table : Fin 5 → Fin 8 → Fin 10),
  (∀ digit : Fin 10, ∃ row_set : Finset (Fin 5), row_set.card = 4 ∧
    (∀ row : Fin 5, row ∈ row_set → ∃ col_set : Finset (Fin 8), col_set.card = 4 ∧
      (∀ col : Fin 8, col ∈ col_set → table row col = digit))) →
  False :=
by
  sorry

end table_filling_impossible_l485_485972


namespace centroid_of_right_triangle_l485_485845

theorem centroid_of_right_triangle (a b : ℝ) :
  ∃ (G : ℝ × ℝ), (G = (b / 3, 2 * a / 3) ∧
  let A := (0, b), B := (a, 0), C := (0, 0),
      M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2),
      N := ((B.1 + C.1) / 2, (B.2 + C.2) / 2),
      med_1 := λ P Q R, ((R.1 + Q.1) / 2, (R.2 + Q.2) / 2),
      med_2 := λ P Q R, ((P.1 + R.1) / 2, (P.2 + R.2) / 2) in
  ∃ (G' : ℝ × ℝ), (G' = (med_1 A B C) = G ∧ (med_2 B C A) = G)) :=
begin
  sorry
end

end centroid_of_right_triangle_l485_485845


namespace find_AB_l485_485331

noncomputable def triangle_ABC (AB BC AC : ℝ) :=
  (AB^2 + BC^2 = AC^2) ∧ (AC ≠ 0)

variable {AB BC AC : ℝ}

theorem find_AB (h_angle : \u3c50 := 90) (h_tan : \u3dfrac {5} {12}) (h_AC : AC = 39) :
    triangle_ABC AB BC AC → AB = 36 :=
by
  unfold triangle_ABC at *
  intro h
  sorry

end find_AB_l485_485331


namespace determine_n_l485_485792

theorem determine_n (x n : ℝ) : 
  (∃ c d : ℝ, G = (c * x + d) ^ 2) ∧ (G = (8 * x^2 + 24 * x + 3 * n) / 8) → n = 6 :=
by {
  sorry
}

end determine_n_l485_485792


namespace simplify_and_evaluate_expression_l485_485834

theorem simplify_and_evaluate_expression (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -2) (hx3 : x ≠ 2) :
  ( ( (x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2 * x)) = x - 2 ) ∧ 
  ( (x = 1) → ((x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2 * x)) = -1 ) :=
by
  sorry

end simplify_and_evaluate_expression_l485_485834


namespace sum_divisible_by_3_probability_l485_485656

-- The 12 prime numbers
def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

-- Function to calculate the residue of a number modulo 3
def residue_mod_3 (n : ℕ) : ℕ := n % 3

-- List of residues of the first 12 primes
def residues : List ℕ := primes.map residue_mod_3

-- Assume residues
def prime_residues := [2, 0, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1]

-- Number of valid combinations where the sum of residues modulo 3 is zero
noncomputable def valid_combinations : ℕ := 150

-- Total combinations to choose 5 out of 12 prime numbers
noncomputable def total_combinations : ℕ := Nat.choose 12 5

-- Final probability
noncomputable def probability := Rat.mk valid_combinations total_combinations

theorem sum_divisible_by_3_probability :
  probability = Rat.mk 25 132 :=
by sorry

end sum_divisible_by_3_probability_l485_485656


namespace lowest_score_is_49_l485_485841

theorem lowest_score_is_49
    (mean_15 : Real)
    (mean_13 : Real)
    (highest : Real)
    (lowest : Real)
    (sum_all : 15 * mean_15 = 1350)
    (sum_13 : 13 * mean_13 = 1196)
    (highest_val : highest = 105)
    (lowest_val : lowest = 49) :
  ∃ lowest : Real, sum_all - sum_13 = highest + lowest := by
  sorry

end lowest_score_is_49_l485_485841


namespace even_count_in_top_15_rows_l485_485606

def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

def count_even_in_row (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ k, is_even (binom n k)).card

def count_even_in_top_15_rows : ℕ :=
  (Finset.range 15).sum count_even_in_row

theorem even_count_in_top_15_rows :
  count_even_in_top_15_rows = "Sum of all identified evens" := by
sorry

end even_count_in_top_15_rows_l485_485606


namespace cyclic_quadrilateral_concurrency_l485_485140

noncomputable theory

open_locale classical

structure Point := (x : ℝ) (y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

def is_cyclic_quadrilateral (A B C D : Point) (ω : Circle) : Prop :=
-- Definition checking if quadrilateral ABCD is cyclic with circle ω.

def circumcircle (A B C : Point) : Circle :=
-- Definition of circumcircle of triangle ABC.

def intersection (A B C D : Point) : Point :=
-- Definition of intersection point of lines AB and CD.

def show_concurrent (A B C D E F G H : Point) : Prop :=
-- Definition stating that lines AC, BD, and GH are concurrent.

theorem cyclic_quadrilateral_concurrency
  (A B C D E F G H : Point)
  (ω ω1 ω2 : Circle)
  (h1 : is_cyclic_quadrilateral A B C D ω)
  (h2 : intersection A B C D = E)
  (h3 : intersection A D B C = F)
  (h4 : ω1 = circumcircle A E F)
  (h5 : ω2 = circumcircle C E F)
  (h6 : ω ∩ ω1 = G)
  (h7 : ω ∩ ω2 = H) :
  show_concurrent A C B D G H :=
sorry

end cyclic_quadrilateral_concurrency_l485_485140


namespace find_M_coordinates_l485_485643

-- Definition: The parabola x^2 = 4y
def is_on_parabola (M : ℝ × ℝ) : Prop :=
  M.1^2 = 4 * M.2

-- Distance calculation from point to a fixed point function
def distance (A B : ℝ × ℝ) : ℝ :=
  ((A.1 - B.1)^2 + (A.2 - B.2)^2)^0.5

-- Focus of the parabola x^2 = 4y with coordinates (0, 1)
def focus := (0, 1)

-- Distance condition: The distance from point M to the focus is 10
def is_distance_to_focus_10 (M : ℝ × ℝ) : Prop :=
  distance M focus = 10

-- Cartesian coordinates for points that satisfy the problem
def M1 : ℝ × ℝ := (6, 9)
def M2 : ℝ × ℝ := (-6, 9)

-- Theorem statement to prove that M1 and M2 are the points satisfying both the parabola equation and distance condition
theorem find_M_coordinates : 
  (is_on_parabola M1 ∧ is_distance_to_focus_10 M1) ∧
  (is_on_parabola M2 ∧ is_distance_to_focus_10 M2) :=
by
  sorry

end find_M_coordinates_l485_485643


namespace boys_together_arrangements_no_adjacent_gender_a_left_of_b_l485_485479

-- Define the boys and girls
def boys := 3
def girls := 4
def total_people := boys + girls

-- Define the factorial function for convenience
noncomputable def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

-- 1. Proving the number of arrangements if the boys must be together
theorem boys_together_arrangements : (Σ x, x = fact boys * fact (boys + girls - boys + 1)) = 720 := 
  sorry

-- 2. Proving the number of arrangements where neither gender can stand next to same gender
theorem no_adjacent_gender : (Σ x, x = fact boys * fact girls) = 144 := 
  sorry

-- 3. Proving the number of arrangements if person A is to the left of person B
theorem a_left_of_b : (Σ x, x = fact total_people / 2) = 2520 := 
  sorry

end boys_together_arrangements_no_adjacent_gender_a_left_of_b_l485_485479


namespace gcd_45_75_l485_485095

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485095


namespace domain_log_base_3e_l485_485002

theorem domain_log_base_3e (x : ℝ) : x > 1 ↔ (∃ y : ℝ, y = log (3 * real.exp 1) (x - 1)) :=
by sorry

end domain_log_base_3e_l485_485002


namespace f_diff_l485_485667

open Nat

-- Define the function f
def f (k : ℕ) : ℝ :=
  (Finset.range (k + 1)) -- gives {0, 1, ..., k}
  .sum (λ i => 1 / (↑i + ↑k + 1))

-- The main theorem statement
theorem f_diff (k : ℕ) (h : 0 < k) : 
  f (k + 1) - f k = 1 / (2 * k + 1) + 1 / (2 * k + 2) - 1 / (k + 1) := by
  sorry

end f_diff_l485_485667


namespace calculate_total_cost_l485_485579

def total_cost (num_boxes : ℕ) (packs_per_box : ℕ) (tissues_per_pack : ℕ) (cost_per_tissue : ℝ) : ℝ :=
  num_boxes * packs_per_box * tissues_per_pack * cost_per_tissue

theorem calculate_total_cost :
  total_cost 10 20 100 0.05 = 1000 := 
by
  sorry

end calculate_total_cost_l485_485579


namespace ship_selection_and_arrangement_l485_485434

def numSelectionsAndArrangements (n m k : ℕ) : ℕ :=
  (Nat.choose n k - Nat.choose m k) * Nat.factorial k

theorem ship_selection_and_arrangement :
  numSelectionsAndArrangements 8 6 3 = 216 := by
  sorry

end ship_selection_and_arrangement_l485_485434


namespace ratio_of_speeds_l485_485636

-- Definitions based on the conditions provided
def timeEddy : ℝ := 3 -- hours
def distanceEddy : ℝ := 510 -- km
def timeFreddy : ℝ := 4 -- hours
def distanceFreddy : ℝ := 300 -- km

-- Helper definitions to compute average speeds
def averageSpeed (distance : ℝ) (time : ℝ) : ℝ := distance / time
def ratio (a : ℝ) (b : ℝ) : Rat := Rat.mk a b

theorem ratio_of_speeds :
  ratio (averageSpeed distanceEddy timeEddy) (averageSpeed distanceFreddy timeFreddy) = Rat.mk 34 15 :=
by
  sorry

end ratio_of_speeds_l485_485636


namespace find_n_with_divisors_conditions_l485_485865

theorem find_n_with_divisors_conditions :
  ∃ n : ℕ, 
    (∀ d : ℕ, d ∣ n → d ∈ [1, n] ∧ 
    (∃ a b c : ℕ, a = 1 ∧ b = d / a ∧ c = d / b ∧ b = 7 * a ∧ d = 10 + b)) →
    n = 2891 :=
by
  sorry

end find_n_with_divisors_conditions_l485_485865


namespace circle_cartesian_equation_max_distance_from_circle_to_line_l485_485760

-- Definition of the circle in polar coordinates and the corresponding proof in Cartesian coordinates.
theorem circle_cartesian_equation :
  ∀ (θ : ℝ), 
  (∃ (ρ : ℝ), ρ = 10 * Real.cos (π / 3 - θ)) ↔ 
  (∀ (x y : ℝ), x = ρ * cos θ ∧ y = ρ * sin θ → x^2 + y^2 - 5 * x - 5 * sqrt 3 * y = 0) := 
sorry

-- Definition and proof for the maximum distance from any point on the circle to the line.
theorem max_distance_from_circle_to_line :
  ∀ (P : ℝ × ℝ), 
  (P.1 ^ 2 + P.2 ^ 2 - 5 * P.1 - 5 * sqrt 3 * P.2 = 0) →
  (∃ (d : ℝ), d = abs (sqrt 3 * (5 / 2) - (5 * sqrt 3) / 2 + 2) / 2 + 5) :=
sorry

end circle_cartesian_equation_max_distance_from_circle_to_line_l485_485760


namespace number_of_valid_three_digit_numbers_l485_485193

-- Definitions for the conditions
def digits := {0, 1, 2, 3, 4, 5}

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def sum_of_digits_is_nine (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 + d2 + d3 = 9

def no_repeated_digits (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 

-- Statement of the theorem to be proved
theorem number_of_valid_three_digit_numbers : 
  finset.filter (λ n, is_three_digit_number n ∧ sum_of_digits_is_nine n ∧ no_repeated_digits n) 
  (finset.range 1000) -- all three-digit numbers
  .card = 16 := 
sorry

end number_of_valid_three_digit_numbers_l485_485193


namespace general_term_formulas_sum_S_n_l485_485694

noncomputable def sum_first_n_terms_a (n : ℕ) : ℚ :=
  n * (3 * n + 1) / 2

def a_n (n : ℕ) : ℚ :=
  if n = 1 then 2 else 3 * n - 1

noncomputable def sum_first_n_terms_b (n : ℕ) : ℚ :=
  (1 - (-2) ^ (n + 1)) / 3

def b_n (n : ℕ) : ℤ :=
  (-2) ^ n

noncomputable def S_n (n : ℕ) : ℤ :=
  -n * (-2) ^ (n + 1)

theorem general_term_formulas (n : ℕ) (hn : n ≥ 1) :
  a_n n = if n = 1 then 2 else 3 * n - 1 ∧
  b_n n = (-2) ^ n :=
by
  sorry

theorem sum_S_n (n : ℕ) (hn : n ≥ 1) :
  S_n n = ∑ i in finset.range n, a_n i.succ * b_n i.succ :=
by
  sorry

end general_term_formulas_sum_S_n_l485_485694


namespace impossible_digit_filling_l485_485975

theorem impossible_digit_filling (T : Fin 5 → Fin 8 → Fin 10) :
  (∀ d : Fin 10, (∃! r₁ r₂ r₃ r₄ : Fin 5, T r₁ = d ∧ T r₂ = d ∧ T r₃ = d ∧ T r₄ = d) ∧
                 (∃! c₁ c₂ c₃ c₄ : Fin 8, T c₁ = d ∧ T c₂ = d ∧ T c₃ = d ∧ T c₄ = d)) → False :=
by
  sorry

end impossible_digit_filling_l485_485975


namespace Alfred_win_condition_l485_485469

noncomputable def Alfred_can_force_win (n : ℕ) : Prop :=
  ∀ a : Fin n → ℤ, (∀ k : Fin n, (k : ℕ) = n - 1 → a k ≠ 0) →
  ∃ x : ℤ, ((Finset.range n).sum (λ k, a k * x ^ (n - 1 - k))) = 0

theorem Alfred_win_condition (n : ℕ) (h : n ≥ 2) :
  Alfred_can_force_win n ↔ n % 2 = 1 :=
by
  sorry

end Alfred_win_condition_l485_485469


namespace barry_time_saved_l485_485314

/-- 
Barry used his stationary bike on 4 days this week, cycling 3 miles each day.
- On Monday and Thursday, he cycled at a speed of 6 miles per hour.
- On Tuesday, he cycled at a speed of 3 miles per hour.
- On Wednesday, he cycled at a speed of 5 miles per hour.
- If Barry had cycled at 5 miles per hour on all days, he would have spent 42 minutes less.
-/
theorem barry_time_saved :
  let distance_per_day := 3 in
  let speed_mon_thu := 6 in
  let speed_tue := 3 in
  let speed_wed := 5 in
  let alternate_speed := 5 in
  let time_mon_thu := distance_per_day / speed_mon_thu in
  let time_tue := distance_per_day / speed_tue in
  let time_wed := distance_per_day / speed_wed in
  let total_time :=
    2 * time_mon_thu + time_tue + time_wed in
  let alternate_time :=
    4 * (distance_per_day / alternate_speed) in
  let time_saved :=
    total_time - alternate_time in
  let time_saved_minutes :=
    time_saved * 60 in
  time_saved_minutes = 42 := by
  sorry

end barry_time_saved_l485_485314


namespace eccentricity_of_ellipse_is_one_third_l485_485275

-- Definitions based on the problem conditions
def origin : (ℝ × ℝ) := (0, 0)

def ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
def focus_left (c : ℝ) : (ℝ × ℝ) := (-c, 0)
def vertex_left (a : ℝ) : (ℝ × ℝ) := (-a, 0)
def vertex_right (a : ℝ) : (ℝ × ℝ) := (a, 0)

def point_P (c b a : ℝ) := (-c, b^2 / a)
def line_equation (k a : ℝ) (x : ℝ) := k * (x + a)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Proof statement (without proof)
theorem eccentricity_of_ellipse_is_one_third
  (a b c : ℝ)
  (ha : a > b)
  (hb : b > 0)
  (hc : c > 0)
  (e : ℝ := c / a)
  (H1 :  c^2 = a^2 - b^2)
  (H2 : 2 * (a - c) = a + c) :
  e = 1 / 3 :=
by
  sorry

end eccentricity_of_ellipse_is_one_third_l485_485275


namespace gcd_45_75_l485_485088

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485088


namespace power_function_quadrant_l485_485706

theorem power_function_quadrant :
  let P := ∀ f : ℝ → ℝ, (∀ x, f x = x^n → ∀ x < 0, f x ≥ 0)
  let P_contrapositive := ∀ f, (∃ x < 0, f x < 0) → ∃ x, f x ≠ x^n
  let P_converse := ∀ f, (∀ x < 0, f x ≥ 0) → f x = x^n
  let P_inverse := ∀ f, (∃ x, f x ≠ x^n) → (∃ x < 0, f x < 0)
  true_count := [P_contrapositive, P_converse, P_inverse].filter(λ p, p).length
  true_count = 1 :=
by
  have P_true : P := sorry
  have P_contrapositive_true : P_contrapositive := sorry
  have P_converse_false : ¬P_converse := sorry
  have P_inverse_false : ¬P_inverse := sorry
  sorry

end power_function_quadrant_l485_485706


namespace distance_between_parallel_lines_is_correct_l485_485460

def line1 (x y: ℝ) : Prop :=
  x - 2 * y + 1 = 0

def line2 (x y : ℝ) : Prop :=
  2 * x - 4 * y - 3 = 0

def distance_between_lines : Prop :=
  (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 ∧ line2 x2 y2 ∧ 
  (real.abs ((x1 - x2) - 2 * (y1 - y2) + 4)) / sqrt (2^2 + (-4)^2) = sqrt (5) / 2)

theorem distance_between_parallel_lines_is_correct : distance_between_lines :=
sorry

end distance_between_parallel_lines_is_correct_l485_485460


namespace gcd_45_75_l485_485114

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485114


namespace domain_transform_l485_485282

theorem domain_transform
  (f : ℝ → ℝ)
  (domain_f : ∀ x : ℝ, ¬(x ∈ set.Icc (-1 : ℝ) 5) → f x = 0) :
  ∀ x : ℝ, ¬(x ∈ set.Icc (4 / 3 : ℝ) (10 / 3 : ℝ)) → f (3 * x - 5) = 0 :=
by
  sorry

end domain_transform_l485_485282


namespace at_least_one_not_less_than_two_l485_485800

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b) ≥ 2 ∨ (b + 1 / c) ≥ 2 ∨ (c + 1 / a) ≥ 2 :=
sorry

end at_least_one_not_less_than_two_l485_485800


namespace minyoung_sells_correct_number_of_fruits_l485_485906

theorem minyoung_sells_correct_number_of_fruits :
  ∀ (tangerines apples : ℕ), tangerines = 2 → apples = 7 → tangerines + apples = 9 :=
by
  intros tangerines apples htang happles
  rw [htang, happles]
  exact Nat.add_comm tangerines apples
  sorry

end minyoung_sells_correct_number_of_fruits_l485_485906


namespace largest_twenty_supporting_l485_485399

def is_twenty_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 20 → ℝ),
    (∑ i, a i : ℝ).round = (∑ i, a i : ℝ) →
    ∃ i, |a i - 1/2| ≥ X

theorem largest_twenty_supporting :
  (∃ X : ℝ, is_twenty_supporting X ∧ ∀ Y : ℝ, is_twenty_supporting Y → Y ≤ X) ∧
  ∀ Z : ℝ, is_twenty_supporting Z → abs ((X : ℝ) - 0.025) < 0.001 :=
sorry

end largest_twenty_supporting_l485_485399


namespace hash_calculation_l485_485723

-- Define the binary operation x#y
def hash : ℕ → ℕ → ℕ := λ x y, x * y - x - 2 * y

-- State the theorem
theorem hash_calculation : (hash 6 4) - (hash 4 6) = 2 := by
  sorry

end hash_calculation_l485_485723


namespace sum_first_2001_l485_485173

-- Define the sequence using the given recurrence relation and initial terms
def sequence (a : ℕ → ℤ) : Prop :=
  ∃ x y : ℤ, 
  (a 1 = x) ∧ (a 2 = y) ∧ (∀ n ≥ 3, a n  = a (n-1) - a (n-2))

-- Sum condition for first 1492 terms
def sum_first_1492 (a : ℕ → ℤ) : Prop :=
  ∑ i in Finset.range 1492, a (i + 1) = 1985

-- Sum condition for first 1985 terms
def sum_first_1985 (a : ℕ → ℤ) : Prop :=
  ∑ i in Finset.range 1985, a (i + 1) = 1492

-- Prove that the sum of the first 2001 terms is 986
theorem sum_first_2001 (a : ℕ → ℤ) :
  sequence a →
  sum_first_1492 a →
  sum_first_1985 a →
  ∑ i in Finset.range 2001, a (i + 1) = 986 :=
by
  sorry

end sum_first_2001_l485_485173


namespace total_apples_collected_l485_485195

-- Definitions based on conditions
def number_of_green_apples : ℕ := 124
def number_of_red_apples : ℕ := 3 * number_of_green_apples

-- Proof statement
theorem total_apples_collected : number_of_red_apples + number_of_green_apples = 496 := by
  sorry

end total_apples_collected_l485_485195


namespace minimum_value_of_expression_l485_485808

noncomputable def minimum_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0): ℝ :=
  ∑ i in [(5 * r) / (3 * p + 2 * q), (5 * p) / (2 * q + 3 * r), (2 * q) / (p + r)], id i

theorem minimum_value_of_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0):
  minimum_value_expression p q r hp hq hr = 151 / 18 := by
  sorry

end minimum_value_of_expression_l485_485808


namespace system_of_inequalities_l485_485015

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end system_of_inequalities_l485_485015


namespace prob_final_states_unchanged_l485_485757

-- Define the 4x4 grid of switches
inductive Switch: Type
| mk: ℕ → ℕ → Switch

-- Define the initial grid setup
def switchGrid := List (Switch.mk 1 1, Switch.mk 1 2, ..., Switch.mk 4 4)

-- Conditions: pressing a switch changes its state and its adjacent switches
def pressSwitch (s: Switch) (grid: List Switch) : List Switch := 
  sorry

-- Define the condition affecting specific switches
def affectSwitch (s target: Switch): Bool := 
  sorry

-- Define the total number of ways to press two different switches
def totalWays: ℕ := 
  sorry

-- Define the number of ways to press two switches without affecting (2,3) and (4,1)
def unaffectedWays: ℕ := 
  sorry

-- The target probability
theorem prob_final_states_unchanged:
    (unaffectedWays + ways_affecting_23 + ways_affecting_41) / totalWays = 41 / 120 :=
by
  sorry

end prob_final_states_unchanged_l485_485757


namespace gcd_45_75_l485_485064

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485064


namespace delightful_numbers_l485_485554

def is_delightful (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000 ∧
  n % 25 = 0 ∧
  let digits := [n / 1000, (n % 1000) / 100, (n % 100) / 10, n % 10] in
  (digits.sum % 25 = 0) ∧ (digits.product % 25 = 0)

theorem delightful_numbers :
  { n : ℕ | is_delightful n } = { 5875, 8575 } :=
by
  sorry

end delightful_numbers_l485_485554


namespace seating_arrangements_l485_485890

def number_of_seats_front : ℕ := 11
def number_of_seats_back : ℕ := 12
def middle_seats_unoccupied : set ℕ := {5, 6, 7}
def people_not_next_to_each_other : Prop := true -- We need this as a condition, stating the people are not adjacent

theorem seating_arrangements (front_seats back_seats : ℕ) 
  (middle_unoccupied : set ℕ) (no_adjacent_seats : Prop) :
  front_seats = 11 → back_seats = 12 → middle_unoccupied = {5, 6, 7} → no_adjacent_seats → ∃ (n : ℕ), n = 346 :=
by 
  intros
  existsi 346
  simp
  sorry

end seating_arrangements_l485_485890


namespace find_integer_pairs_exists_Z0_l485_485998

theorem find_integer_pairs_exists_Z0 (a b : ℤ) (h1 : a * b * (a - b) ≠ 0)
  (h2 : ∃ Z0 : set ℤ, ∀ n : ℤ, ↑n ∈ Z0 ∨ ↑(n + a) ∈ Z0 ∨ ↑(n + b) ∈ Z0) :
  ∃ k y z : ℤ, a = k * y ∧ b = k * z ∧ (∃ y3 z3 : ℤ, y3 = y ∧ z3 = z ∧ y3 % 3 ≠ 0 ∧ z3 % 3 ≠ 0 ∧ (y3 - z3) % 3 ≠ 0) :=
by
  sorry

end find_integer_pairs_exists_Z0_l485_485998


namespace bunnies_count_l485_485727

theorem bunnies_count 
  (bunny_rate : ℕ)
  (total_hours : ℕ)
  (total_bunny_out : ℕ) 
  (total_time : total_hours * 60 = 600)
  (single_bunny_out : bunny_rate * 600 = 1800)
  (combined_out : total_bunny_out = 36000)
  (calc_bunnies : total_bunny_out / single_bunny_out = 20) :
  ∃ n : ℕ, n = 20 :=
by
  use 20
  sorry

end bunnies_count_l485_485727


namespace gcd_150_m_l485_485895

theorem gcd_150_m (m : ℕ)
  (h : ∃ d : ℕ, d ∣ 150 ∧ d ∣ m ∧ (∀ x, x ∣ 150 → x ∣ m → x = 1 ∨ x = 5 ∨ x = 25)) :
  gcd 150 m = 25 :=
sorry

end gcd_150_m_l485_485895


namespace minimum_value_cos_shifted_l485_485862

theorem minimum_value_cos_shifted :
  ∀ x ∈ set.Icc (Real.pi / 6) (2 * Real.pi / 3), 
  ∃ y, y = Real.cos (x - Real.pi / 8) ∧ y ≥ 1 / 2 :=
by sorry

end minimum_value_cos_shifted_l485_485862


namespace ratio_of_segments_of_tangency_l485_485548

theorem ratio_of_segments_of_tangency (A B C L J K : Point) (AB AC BC : ℝ) (r s : ℝ) :
  let triangle := Triangle.mk A B C
  let circle := Circle.inscribed_in triangle
  r < s ∧ triangle.side_lengths = (9, 14, 20) ∧
  is_tangent circle L triangle.BC ∧
  is_tangent circle J triangle.AC ∧
  is_tangent circle K triangle.AB ∧
  segment_length triangle.AK = x ∧
  segment_length triangle.AJ = x ∧
  segment_length triangle.KB = 20 - x ∧
  segment_length triangle.BL = 20 - x ∧
  segment_length triangle.JC = 9 - x ∧
  segment_length triangle.LC = 9 - x ∧
  29 - 2 * x = 14 ∧
  x = 7.5 ∧
  segment_length triangle.LC = 1.5 ∧
  segment_length triangle.BL = 12.5 ∧
  ratio (segment_length triangle.LC) (segment_length triangle.BL) = 3 / 25 := 
sorry

end ratio_of_segments_of_tangency_l485_485548


namespace gcd_45_75_l485_485063

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485063


namespace total_steps_needed_l485_485503

def cycles_needed (dist : ℕ) : ℕ := dist
def steps_per_cycle : ℕ := 5
def effective_steps_per_pattern : ℕ := 1

theorem total_steps_needed (dist : ℕ) (h : dist = 66) : 
  steps_per_cycle * cycles_needed dist = 330 :=
by 
  -- Placeholder for proof
  sorry

end total_steps_needed_l485_485503


namespace range_of_m_l485_485691

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 4 * x - 2

theorem range_of_m (m : ℝ) :
  (∀ x ∈ set.Icc 0 m, quadratic_function x ∈ set.Icc (-6) (-2)) →
  2 ≤ m ∧ m ≤ 4 :=
by
  intro h
  have h0 : quadratic_function 0 = -2 := by norm_num [quadratic_function]
  have h2 : quadratic_function 2 = -6 := by norm_num [quadratic_function]
  have hbnds : (0 : ℝ) ∈ set.Icc 0 m ∧ (2 : ℝ) ∈ set.Icc 0 m := sorry
  sorry

end range_of_m_l485_485691


namespace remainder_when_c_divided_by_b_eq_2_l485_485040

theorem remainder_when_c_divided_by_b_eq_2 
(a b c : ℕ) 
(hb : b = 3 * a + 3) 
(hc : c = 9 * a + 11) : 
  c % b = 2 := 
sorry

end remainder_when_c_divided_by_b_eq_2_l485_485040


namespace max_k_inequality_l485_485240

theorem max_k_inequality (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  ∀ k ≤ 2, ( ( (b - c) ^ 2 * (b + c) / a ) + 
             ( (c - a) ^ 2 * (c + a) / b ) + 
             ( (a - b) ^ 2 * (a + b) / c ) 
             ≥ k * ( a^2 + b^2 + c^2 - a*b - b*c - c*a ) ) :=
by
  sorry

end max_k_inequality_l485_485240


namespace volume_of_earth_dug_out_l485_485517

-- Define the problem conditions
def diameter : ℝ := 4
def depth : ℝ := 24
def radius : ℝ := diameter / 2
def pi_approx : ℝ := 3.14159

-- The statement to prove
theorem volume_of_earth_dug_out : 
  π * radius^2 * depth ≈ 301.59264 := 
begin
  -- Definitions and substitutions based on conditions
  have r_def : radius = diameter / 2 := rfl,
  have vol_formula : π * radius^2 * depth = π * (2)^2 * depth,
  {
    rw [r_def],
    norm_num,
  },
  -- Calculation with given π approximation
  have vol_approx : π * (2)^2 * (24 : ℝ) ≈ 301.59264,
  {
    rw pi_approx,
    norm_num,
  },
  exact vol_approx,
end

end volume_of_earth_dug_out_l485_485517


namespace distance_walked_l485_485364

def walking_time := 6 -- minutes
def walking_speed := 83.33 -- meters per minute

theorem distance_walked : walking_speed * walking_time = 500 := 
by
  sorry

end distance_walked_l485_485364


namespace total_calories_consumed_l485_485217

theorem total_calories_consumed : ∀ (crackers_calories : ℕ) (cookies_calories : ℕ) (num_cookies : ℕ) (num_crackers : ℕ), 
  crackers_calories = 15 → cookies_calories = 50 → num_cookies = 7 → num_crackers = 10 →
  (cookies_calories * num_cookies + crackers_calories * num_crackers) = 500 :=
by
  intros crackers_calories cookies_calories num_cookies num_crackers h_crackers_calories h_cookies_calories h_num_cookies h_num_crackers
  rw [h_crackers_calories, h_cookies_calories, h_num_cookies, h_num_crackers]
  norm_num
  sorry

end total_calories_consumed_l485_485217


namespace time_ratio_l485_485571

theorem time_ratio (A : ℝ) (B : ℝ) (h1 : B = 18) (h2 : 1 / A + 1 / B = 1 / 3) : A / B = 1 / 5 :=
by
  sorry

end time_ratio_l485_485571


namespace other_asymptote_of_hyperbola_l485_485821

theorem other_asymptote_of_hyperbola (a b : ℝ) :
  (∀ x : ℝ, a * x + b = 2 * x) →
  (∀ p : ℝ × ℝ, (p.1 = 3)) →
  ∀ (c : ℝ × ℝ), (c.1 = 3 ∧ c.2 = 6) ->
  ∃ (m : ℝ), m = -1/2 ∧ (∀ x, c.2 = -1/2 * x + 15/2) :=
by
  sorry

end other_asymptote_of_hyperbola_l485_485821


namespace petya_cannot_have_equal_coins_l485_485859

theorem petya_cannot_have_equal_coins
  (transact : ℕ → ℕ)
  (initial_two_kopeck : ℕ)
  (total_operations : ℕ)
  (insertion_machine : ℕ)
  (by_insert_two : ℕ)
  (by_insert_ten : ℕ)
  (odd : ℕ)
  :
  (initial_two_kopeck = 1) ∧ 
  (by_insert_two = 5) ∧ 
  (by_insert_ten = 5) ∧
  (∀ n, transact n = 1 + 4 * n) →
  (odd % 2 = 1) →
  (total_operations = transact insertion_machine) →
  (total_operations % 2 = 1) →
  (∀ x y, (x + y = total_operations) → (x = y) → False) :=
sorry

end petya_cannot_have_equal_coins_l485_485859


namespace license_plates_total_l485_485316

def odd_digits := {1, 3, 5, 7, 9}
def even_digits := {0, 2, 4, 6, 8}
def letters := Finset.range 26  -- Assuming letters are coded as 0-25

def license_plates_count :=
  (letters.card * odd_digits.card * letters.card * even_digits.card * letters.card * odd_digits.card)

theorem license_plates_total :
  license_plates_count = 2_197_000 := by
  sorry

end license_plates_total_l485_485316


namespace general_formula_l485_485029

noncomputable def a_seq : ℕ → ℝ
| 0       := 0 -- typically, sequences in Lean are 0-indexed, but this can be adjusted if needed
| (n + 1) := if n = 0 then 1 else a_seq n + (1 / 3)^n

lemma geom_seq (n : ℕ) :
  (a_seq 1 = 1) ∧ (∀ n > 1, a_seq n - a_seq (n - 1) = (1 / 3)^(n - 1)) :=
by {
  sorry
}

theorem general_formula (n : ℕ) : a_seq n = (3 / 2) * (1 - 1 / 3^n) :=
by {
  sorry
}

end general_formula_l485_485029


namespace value_of_X_l485_485970

noncomputable def arithmetic_seq_value : ℤ :=
  let d1 := (25 - 1) / 4
  let d5 := (81 - 17) / 4
  let d3 := (49 - 13) / 4
  X

theorem value_of_X :
  (∀ i, (1 + (i - 1) * 6 = 1 + (i - 1) * 6)) →
  (∀ i, (17 + (i - 1) * 16 = 17 + (i - 1) * 16)) →
  (∀ j, (13 + (j - 1) * 9 = 13 + (j - 1) * 9)) →
  arithmetic_seq_value = 31 :=
  sorry

end value_of_X_l485_485970


namespace max_sums_greater_than_180_l485_485257

def quadrilateral_angles (A B C D : ℝ) : Prop :=
  A + B + C + D = 360

theorem max_sums_greater_than_180 
  (A B C D : ℝ) 
  (h : quadrilateral_angles A B C D) : 
  (∃ S : finset (ℝ × ℝ), 
    (∀ s ∈ S, s.1 + s.2 > 180) ∧ 
    S.card = 3) :=
sorry

end max_sums_greater_than_180_l485_485257


namespace upper_bound_density_function_l485_485381

-- Define a random variable X with the given conditions
axiom E_exp_sX_finite (X : ℝ → ℝ) (μ : MeasureTheory.Measure ℝ) (s : ℝ) : 
  Expectation (fun x => real.exp (s * X x)) < ⊤

axiom integral_E_exp_siX_finite (X : ℝ → ℝ) (μ : MeasureTheory.Measure ℝ) (s t : ℝ) :
  ∫ (z:ℝ), abs (Expectation (fun x => real.exp ((s + Complex.i * t) * X x))) dμ ≤ ⊤

-- Define the upper bound for density function f
theorem upper_bound_density_function (X : ℝ → ℝ) (μ : MeasureTheory.Measure ℝ) (x : ℝ) :
  ∀f : ℝ → ℝ, 
  (∀s : ℝ, f = (fun x => ∫ (t:ℝ), real.exp (-Complex.i * x * t) * (Expectation (fun x => real.exp (s + Complex.i * t) * X x)) dμ)) →
  f(x) ≤ inf (λs, (real.exp (-s * x) / (2 * real.pi)) * ∫ (t : ℝ), abs (Expectation (fun u => real.exp ((s + Complex.i * t) * X u))) dμ s := sorry

end upper_bound_density_function_l485_485381


namespace exists_x_gt_zero_negation_l485_485864

theorem exists_x_gt_zero_negation :
  (∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry  -- Proof goes here

end exists_x_gt_zero_negation_l485_485864


namespace popsicles_eaten_l485_485411

theorem popsicles_eaten (total_minutes : ℕ) (minutes_per_popsicle : ℕ) (h : total_minutes = 405) (k : minutes_per_popsicle = 12) :
  (total_minutes / minutes_per_popsicle) = 33 :=
by
  sorry

end popsicles_eaten_l485_485411


namespace problem_1_l485_485924

theorem problem_1 (m : ℝ) : (¬ ∃ x : ℝ, x^2 + 2 * x + m ≤ 0) ↔ m > 1 := sorry

end problem_1_l485_485924


namespace percentDecreaseInSquareArea_l485_485762

open Real

noncomputable def triangleArea_I : ℝ := 32 * sqrt 3
noncomputable def triangleArea_III : ℝ := 8 * sqrt 3
noncomputable def squareArea : ℝ := 32
noncomputable def decreasePercent : ℝ := 12.5 / 100
noncomputable def originalSideLength : ℝ := sqrt squareArea

theorem percentDecreaseInSquareArea :
  let newSideLength := originalSideLength * (1 - decreasePercent),
      newArea := newSideLength ^ 2,
      areaDecrease := squareArea - newArea,
      percentDecrease := (areaDecrease / squareArea) * 100
  in percentDecrease = 25 := by
  sorry

end percentDecreaseInSquareArea_l485_485762


namespace jessica_withdrawal_l485_485366

def jessica_balance (X : ℝ) : ℝ :=
  let remaining_balance := (3 / 5) * X
  let deposit := (1 / 2) * remaining_balance
  remaining_balance + deposit

theorem jessica_withdrawal : ∀ X : ℝ, jessica_balance X = 450 → (2 / 5) * X = 200 :=
by
  intro X h_bal
  have h : jessica_balance X = (3 / 5) * X + (1 / 2) * (3 / 5) * X := by sorry
  rw h at h_bal
  -- Further steps to show (2 / 5) * X = 200
  sorry

end jessica_withdrawal_l485_485366


namespace number_of_equilateral_triangles_l485_485229

-- Define the hexagonal lattice points and the distances
variable (LatticePoints : Finset (ℤ × ℤ))
variable [Nonempty (LatticePoints)] -- The lattice is not empty
variable (unit_dist : ℤ) -- The unit distance between nearest neighbors
variable (is_hexagon : Set (ℤ × ℤ) → Prop) -- Hexagon properties

-- Define the property that vertices 1, 3, and 5 are rotated by 120 degrees relative to vertices 2, 4, and 6
variable (rotated_by_120_degrees : (ℤ × ℤ) → Prop)

-- Theorem: Number of equilateral triangles in the hexagonal lattice
theorem number_of_equilateral_triangles
  (LatticePoints_prop : is_hexagon LatticePoints)
  (one_unit_apart : ∀ (p1 p2 : ℤ × ℤ), p1 ∈ LatticePoints ∧ p2 ∈ LatticePoints → dist p1 p2 = unit_dist → p1 ≠ p2)
  (hexagon_rotation : ∀ (p : ℤ × ℤ), p ∈ {1, 3, 5}.map (λ n, nth_vertex n LatticePoints) → rotated_by_120_degrees p)
  : ∃ (n : ℕ), n = 6 := 
sorry

end number_of_equilateral_triangles_l485_485229


namespace solve_for_x_l485_485697

theorem solve_for_x (x y : ℝ) (h : 3 * x - 4 * y = 5) : x = (1 / 3) * (5 + 4 * y) :=
  sorry

end solve_for_x_l485_485697


namespace n_plus_one_is_sum_of_p_perfect_squares_l485_485371

variable (p n : ℕ)
variable (h_prime : Nat.prime p)
variable (h1 : 1 + n * p = Nat.succNat v * Nat.succNat v)

theorem n_plus_one_is_sum_of_p_perfect_squares :
  ∃ (a : Fin p → ℕ), n + 1 = (Finset.univ : Finset (Fin p)).sum (λ i => a i ^ 2) := 
sorry

end n_plus_one_is_sum_of_p_perfect_squares_l485_485371


namespace enclosed_area_correct_l485_485005

noncomputable def enclosed_area (arc_length : ℝ) (octagon_side : ℝ) : ℝ :=
  let r := arc_length / (2 * π)
  let octagon_area := 2 * (1 + Real.sqrt 2) * octagon_side^2
  let sector_area := (arc_length / (2 * π))^2 * π / 8
  octagon_area + 16 * sector_area

theorem enclosed_area_correct :
  enclosed_area (π / 2) 3 = 54 * (1 + Real.sqrt 2) + 2 * π :=
by
  sorry

end enclosed_area_correct_l485_485005


namespace number_of_uncool_parents_l485_485338

variable (total_students cool_dads cool_moms cool_both : ℕ)

theorem number_of_uncool_parents (h1 : total_students = 40)
                                  (h2 : cool_dads = 18)
                                  (h3 : cool_moms = 22)
                                  (h4 : cool_both = 10) :
    total_students - (cool_dads + cool_moms - cool_both) = 10 := by
  sorry

end number_of_uncool_parents_l485_485338


namespace part1_part2_l485_485253

noncomputable def f (x : ℝ) : ℝ := (9 / (Real.sin x)^2) + (4 / (Real.cos x)^2)

theorem part1 : 
  (∀ x ∈ Ioo 0 (Real.pi / 2), f(x) ≥ 25) :=
sorry

theorem part2 :
  { x | abs (x + 5) + abs (2 * x - 1) ≤ 6 } = {x | 0 ≤ x ∧ x ≤ 2 / 3} :=
sorry

end part1_part2_l485_485253


namespace olivine_more_stones_l485_485573

theorem olivine_more_stones (x O D : ℕ) (h1 : O = 30 + x) (h2 : D = O + 11)
  (h3 : 30 + O + D = 111) : x = 5 :=
by
  sorry

end olivine_more_stones_l485_485573


namespace exterior_angle_octagon_degree_l485_485001

-- Conditions
def sum_of_exterior_angles (n : ℕ) : ℕ := 360
def number_of_sides_octagon : ℕ := 8

-- Question and correct answer
theorem exterior_angle_octagon_degree :
  (sum_of_exterior_angles 8) / number_of_sides_octagon = 45 :=
by
  sorry

end exterior_angle_octagon_degree_l485_485001


namespace find_a_perpendicular_lines_l485_485310

theorem find_a_perpendicular_lines {a : ℝ} :
  (∀ x, y = ax - 2 → y = (a + 2)x + 1) →
  a * (a + 2) = -1 →
  a = -1 :=
begin
  sorry
end

end find_a_perpendicular_lines_l485_485310


namespace a_n_formula_b_n_formula_T_n_formula_l485_485294

noncomputable def f : ℕ → ℕ := λ x, 2 * x + 1
def a : ℕ → ℕ 
| 0     := 1
| (n+1) := f (a n) - 1

def b : ℕ → ℕ := λ n, 2 * n + 1

def c (n : ℕ) : ℕ := a n + b n

def sum_c : ℕ → ℕ 
| 0     := 0
| (n+1) := sum_c n + c n

theorem a_n_formula : ∀ n : ℕ, a (n+1) = 2 * (a n) :=
by sorry

theorem b_n_formula : ∀ n : ℕ, b n = 2 * n + 1 :=
by sorry

theorem T_n_formula : ∀ n : ℕ, sum_c n = 2^n + n^2 - 1 :=
by sorry

end a_n_formula_b_n_formula_T_n_formula_l485_485294


namespace path_traveled_by_A_l485_485791

noncomputable def distance_traveled_A (AB CD BC DA : ℝ) (rotation_1 rotation_2 : ℝ) : ℝ :=
  if AB = 3 ∧ CD = 3 ∧ BC = 5 ∧ DA = 5 ∧ rotation_1 = 90 ∧ rotation_2 = 180 then
    (π * Real.sqrt (3^2 + 5^2)) / 2 + 5 * π
  else
    0

theorem path_traveled_by_A :
  ∀ (AB CD BC DA : ℝ) (rotation_1 rotation_2 : ℝ),
    (AB = 3 ∧ CD = 3 ∧ BC = 5 ∧ DA = 5 ∧ rotation_1 = 90 ∧ rotation_2 = 180) →
      distance_traveled_A AB CD BC DA rotation_1 rotation_2 = (π * Real.sqrt 34) / 2 + 5 * π :=
by
  intros AB CD BC DA rotation_1 rotation_2 h
  unfold distance_traveled_A
  rw [if_pos h]
  sorry

end path_traveled_by_A_l485_485791


namespace cube_chopped_off_height_zero_l485_485452

/-- Given a cube with side length 2, a corner of the cube is chopped off such that the cut runs through the three vertices adjacent to the vertex of the chosen corner. 
    The cube is then rotated to rest on one of its original faces. 
    Prove that the height of the remaining cube from the table to the highest point is 0. -/
theorem cube_chopped_off_height_zero : 
  ∀ (s : ℝ), s = 2 →
    ∃ h : ℝ, h = 0 :=
by
  assume s hs,
  have h : ∃ h, h = 2 - 2 := sorry
  show ∃ h, h = 0, from h

#check cube_chopped_off_height_zero

end cube_chopped_off_height_zero_l485_485452


namespace replace_digit_in_626844_with_0_divisible_by_8_and_5_l485_485510

theorem replace_digit_in_626844_with_0_divisible_by_8_and_5:
  ∃ (d: Nat), (6268440 = 626844 * 10 + d) ∧ (d = 0) ∧ (by d % 5 = 0 and (6268440 % 8 = 0)) :=
sorry

end replace_digit_in_626844_with_0_divisible_by_8_and_5_l485_485510


namespace increased_percentage_l485_485496

theorem increased_percentage (x : ℝ) (p : ℝ) (h : x = 75) (h₁ : p = 1.5) : x + (p * x) = 187.5 :=
by
  sorry

end increased_percentage_l485_485496


namespace increase_by_percentage_l485_485497

-- Define the initial number.
def initial_number : ℝ := 75

-- Define the percentage increase as a decimal.
def percentage_increase : ℝ := 1.5

-- Define the expected final result after applying the increase.
def expected_result : ℝ := 187.5

-- The proof statement.
theorem increase_by_percentage : initial_number * (1 + percentage_increase) = expected_result :=
by
  sorry

end increase_by_percentage_l485_485497


namespace gcd_of_45_and_75_l485_485076

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l485_485076


namespace hyperbola_c_eccentricity_is_five_l485_485683

noncomputable def hyperbola_eccentricity (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) 
    (D : ℝ × ℝ) (F : ℝ × ℝ) (DF_perpendicular : D.2 = F.2)
    (E M N : ℝ × ℝ) (DM_ON_relation : 3 * |D.2 - M.2| = 2 * |N.2|)
    (eccentricity : ℝ) : Prop :=
    let A : ℝ × ℝ := (-a, 0)
    let B : ℝ × ℝ := (a, 0)
    let O : ℝ × ℝ := (0, 0)
    (D_point_on_hyperbola : D.1^2 / a^2 - D.2^2 / b^2 = 1) →
    (F = (c := sqrt (a^2 + b^2) * 5, 0)) →
    eccentricity = 5

theorem hyperbola_c_eccentricity_is_five : ∀ (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
    (c : ℝ) (D : ℝ × ℝ) (F : ℝ × ℝ) (DF_perpendicular : D.2 = F.2)
    (E M N : ℝ × ℝ) (DM_ON_relation : 3 * |D.2 - M.2| = 2 * |N.2|)
    (eccentricity : ℝ),
    let A : ℝ × ℝ := (-a, 0),
        B : ℝ × ℝ := (a, 0),
        O : ℝ × ℝ := (0, 0)
    in
    (D.1^2 / a^2 - D.2^2 / b^2 = 1) →
    (F = (sqrt(a^2 + b^2) * 5, 0)) →
    eccentricity = 5 :=
by
intros a b a_pos b_pos c D F DF_perpendicular E M N DM_ON_relation eccentricity A B O
intro D_point_on_hyperbola
intro F_is_focus
apply hyperbola_eccentricity
exact a
exact b
exact a_pos
exact b_pos
exact D
exact F
exact DF_perpendicular
exact E
exact M
exact N
exact DM_ON_relation
exact eccentricity
sorry

end hyperbola_c_eccentricity_is_five_l485_485683


namespace xy_value_l485_485049

noncomputable def compute_xy : ℝ × ℝ → ℝ
| (x, y) := x * y

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : compute_xy (x, y) = 35 / 12 := by
  sorry

end xy_value_l485_485049


namespace problem_part_I_problem_part_II_l485_485295

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - φ)

theorem problem_part_I (ω φ : ℝ) 
  (h1 : ω > 0)
  (h2 : 0 < φ ∧ φ < Real.pi / 2)
  (h3 : f ω φ (Real.pi / 4) = Real.sqrt 3 / 2)
  (h4 : ∃ d, d = Real.pi / 2) :
  ∃ φ : ℝ, (φ = Real.pi / 6 ∧ ω = 2 ∧
    (∀ x, (0 < x ∧ x < Real.pi) →
    (0 < 2 * x - Real.pi / 6 ∧ 2 * x - Real.pi / 6 < Real.pi) →
    f 2 (Real.pi / 6) x > f 2 (Real.pi / 6) (x + 1) ∧ f 2 (Real.pi / 6) x < f 2 (Real.pi / 6) (x - 1)))
    ∧ 
    (∀ x, (0 < x ∧ x < Real.pi) →
    ((0 < x ∧ x < Real.pi / 3) ∨ (5 * Real.pi / 6 < x ∧ x < Real.pi)) →
    f 2 (Real.pi / 6) x = Real.sin (2 * x - Real.pi / 6)))
    :=
sorry

theorem problem_part_II (A : ℝ)
  (h1 : ∀ x, ¬ (x = Real.pi ∨ x = 0))
  (h2 : 0 < A ∧ A < Real.pi)
  (h3 : Real.sin (A - Real.pi / 6) + Real.cos A = 1 / 2) :
  A = 2 * Real.pi / 3 :=
sorry

end problem_part_I_problem_part_II_l485_485295


namespace cos_Z_l485_485357

namespace TriangleCosine

variables {X Y Z : ℝ}

def sin_X := 4 / 5
def cos_Y := 12 / 13

theorem cos_Z (h_triangle : ∠X + ∠Y + ∠Z = π) : cos Z = -16 / 65 := sorry

end TriangleCosine

end cos_Z_l485_485357


namespace quad_AB_CD_l485_485347

theorem quad_AB_CD {A B C D: Type} [Quadrilateral A B C D]
  (parallel: Parallel AD BC)
  (rhombus_angle_bisectors: Rhombus (angle_bisector ∠DAC) (angle_bisector ∠DBC) (angle_bisector ∠ACB) (angle_bisector ∠ADB)) :
  length AB = length CD :=
begin
  sorry
end

end quad_AB_CD_l485_485347


namespace range_of_t_l485_485263

-- Define the quadratic function f(x)
def f (x : ℝ) : ℝ := 2 * x^2 + b * x + c

-- Define the condition that f(x) < 0 on (0, 2)
def condition1 : Prop := ∀ x, 0 < x ∧ x < 2 → f x < 0

-- Define the inequality that must hold for all x in ℝ
def inequality (t : ℝ) (x : ℝ) : Prop := f x + t ≥ 2

-- Prove that the range of t is t ≥ 4
theorem range_of_t (b c t : ℝ) (h1 : condition1) (h2 : ∀ x : ℝ, inequality t x) : t ≥ 4 := 
sorry

end range_of_t_l485_485263


namespace even_count_in_top_15_rows_l485_485624

-- We want to count the number of even binomial coefficients in the first 15 rows of Pascal's Triangle.
-- Specifically, we are considering rows 0 to 14.

def is_even (n : ℕ) := n % 2 = 0

def count_even_binomials_in_pascals_triangle_up_to_row (max_row : ℕ) : ℕ := 
  ∑ n in Finset.range (max_row + 1), 
    ∑ k in Finset.range (n + 1), 
      if is_even (Nat.choose n k) then 1 else 0

theorem even_count_in_top_15_rows : count_even_binomials_in_pascals_triangle_up_to_row 14 = 49 := 
by 
  sorry

end even_count_in_top_15_rows_l485_485624


namespace kenya_peanuts_correct_l485_485776

def jose_peanuts : ℕ := 85
def kenya_more_peanuts : ℕ := 48

def kenya_peanuts : ℕ := jose_peanuts + kenya_more_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := by
  sorry

end kenya_peanuts_correct_l485_485776


namespace problem_l485_485669

theorem problem
  (x y : ℝ)
  (h₁ : x - 2 * y = -5)
  (h₂ : x * y = -2) :
  2 * x^2 * y - 4 * x * y^2 = 20 := 
by
  sorry

end problem_l485_485669


namespace find_coeffs_for_extended_segment_l485_485409

variables {A B Q : Type} [AddCommGroup A] [Module ℝ A]

/-- 
Given points A, B, and Q such that the line segment AB is extended past B to Q
such that AQ:QB = 5:2, and that Q can be represented as a linear combination
of A and B with coefficients s and v respectively, prove that these coefficients
are s = -2/3 and v = 5/3.
-/
theorem find_coeffs_for_extended_segment 
  (A B Q : A)
  (h : Q = (5 : ℝ) • B + (-2 : ℝ) • A) : 
  ∃ (s v : ℝ), Q = s • A + v • B ∧ s = -2/3 ∧ v = 5/3 := 
sorry

end find_coeffs_for_extended_segment_l485_485409


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l485_485439

-- Proof statement for problem 1
theorem solve_eq1 (x : ℝ) : (x - 3)^2 = 16 → x = 7 ∨ x = -1 := 
by
  sorry

-- Proof statement for problem 2
theorem solve_eq2 (x : ℝ) : x^2 - 4x = 5 → x = 5 ∨ x = -1 := 
by
  sorry

-- Proof statement for problem 3
theorem solve_eq3 (x : ℝ) : x^2 - 4x - 5 = 0 → x = 5 ∨ x = -1 := 
by
  sorry

-- Proof statement for problem 4
theorem solve_eq4 (x : ℝ) : x^2 - 5x = 0 → x = 0 ∨ x = 5 := 
by
  sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l485_485439


namespace coefficients_divisible_by_7_l485_485770

theorem coefficients_divisible_by_7 
  {a b c d e : ℤ}
  (h : ∀ x : ℤ, (a * x^4 + b * x^3 + c * x^2 + d * x + e) % 7 = 0) :
  ∃ k l m n o : ℤ, a = 7*k ∧ b = 7*l ∧ c = 7*m ∧ d = 7*n ∧ e = 7*o :=
by
  sorry

end coefficients_divisible_by_7_l485_485770


namespace volume_ratio_of_spheres_l485_485736

/-
  Suppose the surface area ratio of three spheres is 1:2:3.
  Let's prove that the volume ratio of these spheres is 1:2\sqrt{2}:3\sqrt{3}.
-/

theorem volume_ratio_of_spheres
  (r_1 r_2 r_3 : ℝ)
  (S1_eq : 4 * π * r_1^2 = 4 * π * r_2^2 / 2)
  (S2_eq : 4 * π * r_3^2 = 4 * π * r_2^2 * (3 / 2)) :
  (r_1^3 : r_2^3 : r_3^3) = (1 : 2 * real.sqrt 2 : 3 * real.sqrt 3) :=
by
  sorry

end volume_ratio_of_spheres_l485_485736


namespace constant_term_of_expr_is_84_l485_485847

-- Define the expression you will work with
def expr (x : ℂ) : ℂ := (1 + x) * (Complex.exp (-2 * x) - Complex.exp x) ^ 9

-- State the theorem about the constant term
theorem constant_term_of_expr_is_84 : 
    let x := (0 : ℂ) in 
    ∃ (c : ℝ), 
    (∃ k : ℂ, expr k = c * k ^ 0) ∧ 
    c = 84 :=
by
  sorry

end constant_term_of_expr_is_84_l485_485847


namespace magnitude_of_linear_combination_is_sqrt_65_l485_485705

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (2 * m - 1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (-2, 3 * m - 2)
noncomputable def perpendicular (u v : ℝ × ℝ) : Prop := (u.1 * v.1 + u.2 * v.2 = 0)

theorem magnitude_of_linear_combination_is_sqrt_65 (m : ℝ) 
  (h_perpendicular : perpendicular (vector_a m) (vector_b m)) : 
  ‖((2 : ℝ) • (vector_a 1) - (3 : ℝ) • (vector_b 1))‖ = Real.sqrt 65 := 
by
  sorry

end magnitude_of_linear_combination_is_sqrt_65_l485_485705


namespace a2017_is_2_l485_485269

def sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) = a (n + 1) - a n

theorem a2017_is_2 (a : ℕ → ℤ) (h_seq : sequence a) (h1 : a 1 = 2) (h2 : a 2 = 3) : 
  a 2017 = 2 :=
by
  sorry

end a2017_is_2_l485_485269


namespace rons_chocolate_cost_l485_485540

theorem rons_chocolate_cost :
  let cost_per_bar := 1.5
  let sections_per_bar := 3
  let scouts := 15
  let smores_per_scout := 2
  let total_smores := scouts * smores_per_scout
  let bars_needed := total_smores / sections_per_bar in
  bars_needed * cost_per_bar = 15.0 := by
  sorry

end rons_chocolate_cost_l485_485540


namespace perfect_cubes_between_200_and_1600_l485_485714

theorem perfect_cubes_between_200_and_1600 : 
  ∃ (count : ℕ), count = (finset.filter (λ n, 200 ≤ n^3 ∧ n^3 ≤ 1600) (finset.range 50)).card := 
begin
  use 6,
  sorry,
end

end perfect_cubes_between_200_and_1600_l485_485714


namespace problem_one_black_ball_l485_485928

/-
A box contains 105 black balls, 89 gray balls, and 5 white balls. We perform the following steps until only two balls remain:

1. Two balls are drawn from the box.
2. If the balls drawn are of different colors, the darker-colored ball is returned to the box.
3. If the balls drawn are of the same color, both are discarded and one white ball is added to the box.

Prove that exactly one of the two remaining balls will be black.
-/

theorem problem_one_black_ball (initial_black : ℕ) (initial_gray : ℕ) (initial_white : ℕ) (final_black : ℕ) (final_gray : ℕ) (final_white : ℕ) :
  initial_black = 105 →
  initial_gray = 89 →
  initial_white = 5 →
  final_black + final_gray + final_white = 2 →
  (final_black = 1 → final_gray = 0 ∧ final_white = 1) ∧
  (final_black = 1 → final_white = 1 ∧ final_gray = 0) :=
begin
  sorry
end

end problem_one_black_ball_l485_485928


namespace distribution_schemes_36_l485_485172

def num_distribution_schemes (total_students english_excellent computer_skills : ℕ) : ℕ :=
  if total_students = 8 ∧ english_excellent = 2 ∧ computer_skills = 3 then 36 else 0

theorem distribution_schemes_36 :
  num_distribution_schemes 8 2 3 = 36 :=
by
 sorry

end distribution_schemes_36_l485_485172


namespace diameter_circle_inscribed_triangle_l485_485053

noncomputable def diameter_of_inscribed_circle (XY XZ YZ : ℝ) : ℝ :=
  let s := (XY + XZ + YZ) / 2
  let K := Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))
  let r := K / s
  2 * r

theorem diameter_circle_inscribed_triangle (XY XZ YZ : ℝ) (hXY : XY = 13) (hXZ : XZ = 8) (hYZ : YZ = 9) :
  diameter_of_inscribed_circle XY XZ YZ = 2 * Real.sqrt 210 / 5 := by
{
  rw [hXY, hXZ, hYZ]
  sorry
}

end diameter_circle_inscribed_triangle_l485_485053


namespace a_and_b_worked_together_for_20_days_l485_485917

variable (W : ℝ)

def work_done_by_a_and_b (x : ℝ) : ℝ :=
  x * (W / 30)

def work_done_by_a_alone (days : ℝ) : ℝ :=
  days * (W / 60)

theorem a_and_b_worked_together_for_20_days
  (total_work_eq : work_done_by_a_and_b W x + work_done_by_a_alone W 20 = W) :
  x = 20 := by
  sorry

end a_and_b_worked_together_for_20_days_l485_485917


namespace equilateral_triangle_opposite_isosceles_l485_485943

theorem equilateral_triangle_opposite_isosceles (A B C D M : Point)
  (hSquare : square A B C D)
  (hInside : inside_square M A B C D)
  (hIsosceles : angle M D C = 15 ∧ angle M C D = 15) :
  ∃ N, opposite_triangle_is_equilateral M N A B C D :=
by { sorry }

end equilateral_triangle_opposite_isosceles_l485_485943


namespace gcd_45_75_l485_485106

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485106


namespace incorrect_statement_d_l485_485512

-- Definitions based on the problem's conditions
def is_acute (θ : ℝ) := 0 < θ ∧ θ < 90

def is_complementary (θ₁ θ₂ : ℝ) := θ₁ + θ₂ = 90

def is_supplementary (θ₁ θ₂ : ℝ) := θ₁ + θ₂ = 180

-- Statement D from the problem
def statement_d (θ : ℝ) := is_acute θ → ∀ θc, is_complementary θ θc → θ > θc

-- The theorem we want to prove
theorem incorrect_statement_d : ¬(∀ θ : ℝ, statement_d θ) := 
by sorry

end incorrect_statement_d_l485_485512


namespace find_n_l485_485717

theorem find_n (n : ℝ) (h : 7^(3 * n) = (1 / 7)^(n - 30)) : n = 7.5 :=
by
  sorry

end find_n_l485_485717


namespace radius_of_circle_is_ten_l485_485526

noncomputable def radius_of_circle (diameter : ℝ) : ℝ :=
  diameter / 2

theorem radius_of_circle_is_ten :
  radius_of_circle 20 = 10 :=
by
  unfold radius_of_circle
  sorry

end radius_of_circle_is_ten_l485_485526


namespace num_divisors_of_g_2010_l485_485251

-- Define a function g(n) that returns the smallest even integer such that 1/k has exactly n digits after the decimal point
def g (n : ℕ) : ℕ :=
  2^n

-- Theorem statement: For n=2010, the number of positive integer divisors of g(2010) is 2011
theorem num_divisors_of_g_2010 : 
  ∀ (n : ℕ), g(2010) = 2 ^ 2010 → fintype.card {d : ℕ | d ∣ g(2010)} = 2011 :=
by
  intro n
  assume h : g(2010) = 2 ^ 2010
  /- Proof should be provided here -/
  -- This is just a placeholder for now.
  sorry

end num_divisors_of_g_2010_l485_485251


namespace circles_intersect_on_altitude_l485_485377

open EuclideanGeometry

noncomputable def midpoint (A B: Point) : Point := sorry -- assuming existence of midpoint function

theorem circles_intersect_on_altitude (A B C : Point) (M : Point) (N : Point)
  (h_triangle : Triangle A B C)
  (hM : M = midpoint A B)
  (hN : N = midpoint A C) :
  ∃ P : Point, IsOnAltitudeFromA P ∧ IsOnCircleWithDiameter P (segment C M) ∧ IsOnCircleWithDiameter P (segment B N) :=
sorry

end circles_intersect_on_altitude_l485_485377


namespace quadratic_has_one_solution_iff_l485_485654

theorem quadratic_has_one_solution_iff (n : ℕ) : 
  (∃ x : ℝ, ∀ y : ℝ, 16 * y ^ 2 + n * y + 4 = 0 → y = x) ↔ (n = 16 ∨ n = -16) :=
by
  sorry

end quadratic_has_one_solution_iff_l485_485654


namespace average_people_per_minute_l485_485759

def people : ℕ := 1500
def hours_per_day : ℕ := 24
def minutes_per_hour : ℕ := 60

theorem average_people_per_minute :
  Nat.round ((people : ℝ) / (hours_per_day * 2 * minutes_per_hour : ℝ)) = 1 :=
by
  sorry

end average_people_per_minute_l485_485759


namespace calc_expression_l485_485981

theorem calc_expression : 2 / (-1 / 4) - | -Real.sqrt 18 | + (1 / 5)⁻¹ = -3 - 3 * Real.sqrt 2 :=
by
  sorry

end calc_expression_l485_485981


namespace find_abs_z_l485_485394

open Complex

theorem find_abs_z (z w : ℂ) (h1 : complex.abs (3 * z - 2 * w) = 15)
                         (h2 : complex.abs (2 * z + 3 * w) = 10)
                         (h3 : complex.abs (z - w) = 3)
  : complex.abs z = 4.5 :=
by
  sorry

end find_abs_z_l485_485394


namespace ratio_angle_OBD_BAC_l485_485183

-- Definitions based on the conditions given in the problem
def is_acute_angle (A B C : Point) : Prop := sorry
def is_inscribed (A B C : Point) (O : Point) : Prop := sorry
def arc_measure (A B : Point) (θ : Real) : Prop := sorry
def perpendicular (O D : Point) (A C : Point) : Prop := sorry

axiom point : Type
axiom Point : point
variables (A B C D O : Point)
variable (θ_AB θ_BC : Real)

-- The given conditions
axiom acute_triangle : is_acute_angle A B C
axiom inscribed_circle : is_inscribed A B C O
axiom arc_AB : arc_measure A B 40
axiom arc_BC : arc_measure B C 100
axiom perpendicular_OD_AC : perpendicular O D A C

-- The target assertion we need to prove
theorem ratio_angle_OBD_BAC : 
  (∠OBD / ∠BAC) = (7 / 5) :=
sorry

end ratio_angle_OBD_BAC_l485_485183


namespace sum_of_remainders_l485_485908

theorem sum_of_remainders (n : ℤ) (h₁ : n % 12 = 5) (h₂ : n % 3 = 2) (h₃ : n % 4 = 1) : 2 + 1 = 3 := by
  sorry

end sum_of_remainders_l485_485908


namespace evaluate_expression_l485_485993

noncomputable def expr : ℝ :=
  3 + real.sqrt 3 + 1 / (3 + real.sqrt 3) + 1 / (real.sqrt 3 - 3)

theorem evaluate_expression : expr = 3 :=
by
  -- proof goes here
  sorry

end evaluate_expression_l485_485993


namespace find_triplet_x_y_z_l485_485641

theorem find_triplet_x_y_z :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + 1 / (y + 1 / z : ℝ) = (10 : ℝ) / 7) ∧ (x = 1 ∧ y = 2 ∧ z = 3) :=
by
  sorry

end find_triplet_x_y_z_l485_485641


namespace unique_solution_xy_l485_485651

theorem unique_solution_xy : ∃! (x y : ℕ), (x > y) ∧ ((x - y) ^ (x * y) = x ^ y * y ^ x) ∧ x = 4 ∧ y = 2 :=
by
  use (4, 2)
  split
  sorry
  intros a b h
  sorry

end unique_solution_xy_l485_485651


namespace root_conditions_l485_485863

theorem root_conditions (m : ℝ) : (∃ a b : ℝ, a < 2 ∧ b > 2 ∧ a * b = -1 ∧ a + b = m) ↔ m > 3 / 2 := sorry

end root_conditions_l485_485863


namespace dog_total_distance_l485_485486

theorem dog_total_distance (d_villages d_dog : ℝ) (s1 s2 s_dog : ℝ) (t : ℝ)
  (h1 : d_villages = 18)
  (h2 : s1 = 5)
  (h3 : s2 = 4)
  (h4 : s_dog = 8)
  (h5 : t = d_villages / (s1 + s2)) :
  d_dog = s_dog * t :=
by
  rw [h1, h2, h3, h4, h5]
  norm_num
  exact rfl

end dog_total_distance_l485_485486


namespace min_expression_value_l485_485682

theorem min_expression_value (m n : ℝ) (h : m - n^2 = 1) : ∃ min_val : ℝ, min_val = 4 ∧ (∀ x y, x - y^2 = 1 → m^2 + 2 * y^2 + 4 * x - 1 ≥ min_val) :=
by
  sorry

end min_expression_value_l485_485682


namespace upper_bound_lengthAB_l485_485430

-- Define the regular tetrahedron and its properties
structure Tetrahedron (A B C D : Point) : Prop :=
  (regular : regular_tetrahedron A B C D)

-- Projection resulting in the convex quadrilateral
def Projection (A B C D A' B' C' D' : Point) : Prop :=
  convex_quadrilateral A' B' C' D' ∧
  A'B' = A'D' ∧
  C'B' = C'D' ∧
  area_quadrilateral A' B' C' D' = 4

noncomputable def lengthAB (A B : Point) : ℝ :=
  distance A B

theorem upper_bound_lengthAB 
  {A B C D A' B' C' D' : Point}
  (h1 : Tetrahedron A B C D)
  (h2 : Projection A B C D A' B' C' D') :
  ∃ b : ℝ, b = 2 * (6 : ℝ)^(1/4) ∧ ∀ x : ℝ, lengthAB A B < b :=
sorry

end upper_bound_lengthAB_l485_485430


namespace yellow_bows_count_l485_485744

noncomputable def total_bows (n : ℚ) : ℚ :=
  let non_black_fraction := (1/6) + (1/3) + (1/8)
  let black_fraction := 1 - non_black_fraction
  let total_bows := (40 / black_fraction)
  total_bows

noncomputable def yellow_bows (n : ℚ) : ℚ :=
  total_bows(n) * (1/6)

theorem yellow_bows_count : yellow_bows(1) = 160/9 := by
  sorry

end yellow_bows_count_l485_485744


namespace inequality_solution_l485_485006

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end inequality_solution_l485_485006


namespace gcd_of_45_and_75_l485_485079

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l485_485079


namespace angle_sum_A_P_Q_l485_485820

variables 
  (A B C P Q R : Type)
  [Nonempty A] [Nonempty B] [Nonempty C] -- To denote points in a triangle
  [Nonempty P] [Nonempty Q] [Nonempty R] -- To denote specific points as described
  [Triangle ABC] -- Denotes ABC as a non-isosceles triangle
  [OnSide P Q AB] -- Points P and Q on side AB
  (AP_eq_AC : distance A P = distance A C) -- AP = AC
  (BQ_eq_BC : distance B Q = distance B C) -- BQ = BC
  (PB_perpendicular_to_PQ : perpendicular_bisector PQ intersects angle_bisector ∠C at point R) -- Description of R

-- Goal: Prove that ∠ACB + ∠PRQ = 180 degrees
theorem angle_sum_A_P_Q :
  angle ACB + angle PRQ = 180 :=
begin
  sorry
end

end angle_sum_A_P_Q_l485_485820


namespace total_hours_worked_l485_485978

-- Definitions based on the conditions
def hours_per_day : ℕ := 3
def days_worked : ℕ := 6

-- Statement of the problem
theorem total_hours_worked : hours_per_day * days_worked = 18 := by
  sorry

end total_hours_worked_l485_485978


namespace probability_lt_7000_l485_485846

def cities : List String := ["Bangkok", "Cape Town", "Honolulu", "London"]

def distances : String → String → Option ℕ
| "Bangkok", "Cape Town" => some 6300
| "Bangkok", "Honolulu" => some 6609
| "Bangkok", "London" => some 5944
| "Cape Town", "Bangkok" => some 6300
| "Cape Town", "Honolulu" => some 11535
| "Cape Town", "London" => some 5989
| "Honolulu", "Bangkok" => some 6609
| "Honolulu", "Cape Town" => some 11535
| "Honolulu", "London" => some 7240
| "London", "Bangkok" => some 5944
| "London", "Cape Town" => some 5989
| "London", "Honolulu" => some 7240
| _, _ => none

def less_than_7000 (a b : String) : Prop :=
match distances a b with
| some d => d < 7000
| none => false

theorem probability_lt_7000 : 
  (let total_pairs := 6 in        -- Total pairs
  let pairs_lt_7000 := 4 in       -- Pairs with distance less than 7000 miles
  let probability := pairs_lt_7000 / total_pairs in
  probability = 2 / 3) :=
by
  sorry

end probability_lt_7000_l485_485846


namespace sum_of_p_for_circumcenter_on_Ox_l485_485453

/--
Given the quadratic equation: y = 2^p * x^2 + 5 * p * x - 2^(p^2)
and the triangle ABC formed by the intersections with the axes,
find the sum of all values of the parameter p for which the center of the
circle circumscribing the triangle ABC lies on the Ox axis.
-/
theorem sum_of_p_for_circumcenter_on_Ox {p : ℝ} :
  ∑ p ∈ {p | ∃ (x1 x2 : ℝ), 
        (2^p * x1^2 + 5*p*x1 - 2^(p^2) = 0) ∧ 
        (2^p * x2^2 + 5*p*x2 - 2^(p^2) = 0) ∧ 
        ∃ (C : ℝ × ℝ), 
          C = (0, -2^(p^2)) ∧ 
          ((-2^(p^2) / x1) * (-2^(p^2) / x2) = -1)},
    p = -1 :=
begin
  sorry
end

end sum_of_p_for_circumcenter_on_Ox_l485_485453


namespace cute_pairs_count_l485_485478

def is_cute_pair (a b : ℕ) : Prop :=
  a ≥ b / 2 + 7 ∧ b ≥ a / 2 + 7

def max_cute_pairs : Prop :=
  ∀ (ages : Finset ℕ), 
  (∀ x ∈ ages, 1 ≤ x ∧ x ≤ 100) →
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ pair ∈ pairs, is_cute_pair pair.1 pair.2) ∧
    (∀ x ∈ pairs, ∀ y ∈ pairs, x ≠ y → x.1 ≠ y.1 ∧ x.2 ≠ y.2) ∧
    pairs.card = 43)

theorem cute_pairs_count : max_cute_pairs := 
sorry

end cute_pairs_count_l485_485478


namespace cos_A_correct_tan_2A_correct_l485_485360

noncomputable def cos_A (sinC : ℝ) (cosB : ℝ) : ℝ :=
  √(1 - (sinC ^ 2)) -- dummy initial value, just for syntax purpose
by
  unfold cos_A
sorry

theorem cos_A_correct (sinC : ℝ) (cosB : ℝ) (h₁ : sinC = 3 / 5) (h₂ : cosB = -3 / 5) :
  cos_A sinC cosB = 24 / 25 :=
by
  unfold cos_A
  sorry

noncomputable def tan_2A (sinC : ℝ) (cosB : ℝ) : ℝ :=
  (2 * (7 / 24)) / (1 - (7 / 24) ^ 2) -- dummy initial value, just for syntax purpose
by
  unfold tan_2A
sorry

theorem tan_2A_correct (sinC : ℝ) (cosB : ℝ) (h₁ : sinC = 3 / 5) (h₂ : cosB = -3 / 5) :
  tan_2A sinC cosB = 336 / 527 :=
by
  unfold tan_2A
  sorry

end cos_A_correct_tan_2A_correct_l485_485360


namespace sqrt_inequality_l485_485139

theorem sqrt_inequality
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  :
  sqrt (a^2 + a * b + b^2) + sqrt (a^2 + a * c + c^2) ≥ 4 * sqrt ( (ab / (a + b))^2 + (ab / (a + b)) * (ac / (a + c)) + (ac / (a + c))^2 ) :=
by
  sorry

end sqrt_inequality_l485_485139


namespace calculate_3_to_2x_minus_y_l485_485663

theorem calculate_3_to_2x_minus_y 
  (x y : ℝ) 
  (hx : 3^x = 5) 
  (hy : 3^y = 4) : 
  3^(2 * x - y) = 25 / 4 := 
by
  sorry

end calculate_3_to_2x_minus_y_l485_485663


namespace composed_area_l485_485944

def AC : ℝ := 40
def AE : ℝ := 30
def AB : ℝ := AC / 3
def AF : ℝ := AE / 2
def EF : ℝ := AE / 2 -- Because F is midpoint of AE
def area_rect_ACDE : ℝ := AC * AE
def area_eq_triangle_CEF : ℝ := (Real.sqrt 3 / 4) * EF ^ 2
def area_triangle_ABF : ℝ := 0.5 * AB * AF

def total_area : ℝ := area_rect_ACDE + area_eq_triangle_CEF - area_triangle_ABF

theorem composed_area :
  total_area = 1100 + 225 * Real.sqrt 3 / 4 :=
by
  -- sorry for the proof; the statement intends to be demonstrated.
  sorry

end composed_area_l485_485944


namespace symmetric_point_coordinates_l485_485272

def point := ℝ × ℝ × ℝ

def symmetric_with_respect_to_x_axis (M : point) : point :=
  (M.1, -M.2, -M.3)

theorem symmetric_point_coordinates :
  symmetric_with_respect_to_x_axis (2, 1, 3) = (2, -1, -3) :=
by
  -- Proof is omitted as per instruction
  sorry

end symmetric_point_coordinates_l485_485272


namespace kenya_peanuts_correct_l485_485775

def jose_peanuts : ℕ := 85
def kenya_more_peanuts : ℕ := 48

def kenya_peanuts : ℕ := jose_peanuts + kenya_more_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := by
  sorry

end kenya_peanuts_correct_l485_485775


namespace find_f_value_l485_485685

noncomputable def omega (d : ℝ) : ℝ :=
if d = π/2 then 2 else 0

noncomputable def f (x ω φ : ℝ) : ℝ :=
Real.sin (ω * x + φ)

theorem find_f_value
  (φ : ℝ)
  (hq1 : Real.sin φ = 3 / 5)
  (hq2 : φ ∈ Set.Ioo (π / 2) π)
  (hq3 : (λ(ω : ℝ), 2 * π / ω = π / 2) (omega (π / 2))) :
  f (π / 4) (omega (π / 2)) φ = -4 / 5 :=
by
  sorry

end find_f_value_l485_485685


namespace impossible_digit_filling_l485_485974

theorem impossible_digit_filling (T : Fin 5 → Fin 8 → Fin 10) :
  (∀ d : Fin 10, (∃! r₁ r₂ r₃ r₄ : Fin 5, T r₁ = d ∧ T r₂ = d ∧ T r₃ = d ∧ T r₄ = d) ∧
                 (∃! c₁ c₂ c₃ c₄ : Fin 8, T c₁ = d ∧ T c₂ = d ∧ T c₃ = d ∧ T c₄ = d)) → False :=
by
  sorry

end impossible_digit_filling_l485_485974


namespace feuerbach_circle_l485_485849

-- Define a structure to represent a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the vertices of the triangle
def vertexA : Point := ⟨a, 0⟩
def vertexB : Point := ⟨b, 0⟩
def vertexC : Point := ⟨0, c⟩

-- Define the midpoints of the sides of the triangle
def midpoint_AB : Point := ⟨(a + b) / 2, 0⟩
def midpoint_AC : Point := ⟨a / 2, c / 2⟩
def midpoint_BC : Point := ⟨b / 2, c / 2⟩

-- Define the general equation of a circle
def circle_eq (A B C : ℝ) (p : Point) : ℝ :=
  p.x ^ 2 + p.y ^2 + A * p.x + B * p.y + C

-- Define the nine-point circle equation to be proved
def feuerbach_circle_eq (A B C : ℝ) : (ℝ → ℝ → ℝ) :=
  λ (x y : ℝ), 2 * c * (x ^ 2 + y ^ 2) - (a + b) * c * x + (a * b - c ^ 2) * y

-- Theorem to prove the nine-point circle equation
theorem feuerbach_circle (a b c : ℝ) :
  ∃ (A B C : ℝ), (∀ (p : Point), p = midpoint_AB ∨ p = midpoint_AC ∨ p = midpoint_BC → circle_eq A B C p = 0) →
    ∀ x y : ℝ, circle_eq A B C ⟨x, y⟩ = feuerbach_circle_eq A B C x y := by
  sorry

end feuerbach_circle_l485_485849


namespace area_of_kite_l485_485021

theorem area_of_kite (a b c d : ℂ) (h1 : a * b * c * d = 1 + 12*I)
  (h2 : a + b + c + d = -4*I)
  (h_poly : (a^4 + 4*I*a^3 + (7 - 7*I)*a^2 + (10 + 2*I)*a - (1 + 12*I)) = 0) 
  (h_poly_b : (b^4 + 4*I*b^3 + (7 - 7*I)*b^2 + (10 + 2*I)*b - (1 + 12*I)) = 0)
  (h_poly_c : (c^4 + 4*I*c^3 + (7 - 7*I)*c^2 + (10 + 2*I)*c - (1 + 12*I)) = 0)
  (h_poly_d : (d^4 + 4*I*d^3 + (7 - 7*I)*d^2 + (10 + 2*I)*d - (1 + 12*I)) = 0) :
  abs ((a - c) * (b - d) / 2) = 5 / 2 :=
by { sorry }

end area_of_kite_l485_485021


namespace lily_pads_half_coverage_l485_485342

-- Define the constants and conditions of the problem.
def days_to_cover_lake : ℕ := 48

-- The theorem corresponds to the mathematical proof problem.
theorem lily_pads_half_coverage (n : ℕ) (h : n = days_to_cover_lake - 1) :
  ∃ k : ℕ, k = (days_to_cover_lake - 1) ∧ k = 47 :=
by
  use days_to_cover_lake - 1
  split
  · exact h
  · simp [days_to_cover_lake]
  · sorry

end lily_pads_half_coverage_l485_485342


namespace bee_distance_from_initial_l485_485926

noncomputable def omega : ℂ := Complex.exp (-Complex.I * Real.pi / 4)

def bee_position (n : ℕ) : ℂ :=
  (Finset.range n).sum (λ j, (j + 1) * omega^j)

theorem bee_distance_from_initial : 
  Complex.abs (bee_position 2015) = 2016 * Real.sqrt (2 + Real.sqrt 2) :=
sorry

end bee_distance_from_initial_l485_485926


namespace average_speed_round_trip_l485_485177

-- Define average speed calculation for round trip

open Real

theorem average_speed_round_trip (S : ℝ) (hS : S > 0) :
  let t1 := S / 6
  let t2 := S / 4
  let total_distance := 2 * S
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  average_speed = 4.8 :=
  by
    sorry

end average_speed_round_trip_l485_485177


namespace floor_function_identity_l485_485830

theorem floor_function_identity (n : ℕ) (x : ℝ) (h : n > 0) :
  ∑ k in finset.range n, int.floor (x + k / n) = int.floor (n * x) := sorry

end floor_function_identity_l485_485830


namespace total_spokes_in_garage_l485_485587

theorem total_spokes_in_garage :
  let bicycle1_spokes := 12 + 10
  let bicycle2_spokes := 14 + 12
  let bicycle3_spokes := 10 + 14
  let tricycle_spokes := 14 + 12 + 16
  bicycle1_spokes + bicycle2_spokes + bicycle3_spokes + tricycle_spokes = 114 :=
by
  let bicycle1_spokes := 12 + 10
  let bicycle2_spokes := 14 + 12
  let bicycle3_spokes := 10 + 14
  let tricycle_spokes := 14 + 12 + 16
  show bicycle1_spokes + bicycle2_spokes + bicycle3_spokes + tricycle_spokes = 114
  sorry

end total_spokes_in_garage_l485_485587


namespace pills_per_day_john_needs_2_pills_per_day_l485_485774

variables (cost_per_pill insurance_coverage total_payment days_in_month : ℝ)
variables (P : ℝ) -- P is the number of pills John needs to take per day

-- Given conditions
def cost_paid_per_pill := cost_per_pill * (1 - insurance_coverage)
def total_pills_in_month := total_payment / cost_paid_per_pill

theorem pills_per_day
  (h_cost: cost_per_pill = 1.5)
  (h_insurance: insurance_coverage = 0.4)
  (h_payment: total_payment = 54)
  (h_days: days_in_month = 30) :
  P = total_pills_in_month / days_in_month :=
sorry

-- To prove that P = 2 given the conditions
theorem john_needs_2_pills_per_day
  (h_cost: cost_per_pill = 1.5)
  (h_insurance: insurance_coverage = 0.4)
  (h_payment: total_payment = 54)
  (h_days: days_in_month = 30) :
  P = 2 :=
begin
  unfold cost_paid_per_pill total_pills_in_month,
  simp [h_cost, h_insurance, h_payment, h_days],
  norm_num,
end

end pills_per_day_john_needs_2_pills_per_day_l485_485774


namespace smallest_square_contains_five_disks_l485_485653

noncomputable def smallest_side_length := 2 + 2 * Real.sqrt 2

theorem smallest_square_contains_five_disks :
  ∃ (a : ℝ), a = smallest_side_length ∧ (∃ (d : ℕ → ℝ × ℝ), 
    (∀ i, 0 ≤ i ∧ i < 5 → (d i).fst ^ 2 + (d i).snd ^ 2 < (a / 2 - 1) ^ 2) ∧ 
    (∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 ∧ i ≠ j → 
      (d i).fst ^ 2 + (d i).snd ^ 2 + (d j).fst ^ 2 + (d j).snd ^ 2 ≥ 4)) :=
sorry

end smallest_square_contains_five_disks_l485_485653


namespace cosine_expansion_sum_of_squares_l485_485602

noncomputable def constants := 
  let b1 := 35 / 64
  let b2 := 0
  let b3 := 21 / 64
  let b4 := 0
  let b5 := 7 / 64
  let b6 := 0
  let b7 := 1 / 64
  (b1, b2, b3, b4, b5, b6, b7)

theorem cosine_expansion (θ : ℝ) :
  let (b1, b2, b3, b4, b5, b6, b7) := constants
  cos (θ)^7 = b1 * cos (θ) + b2 * cos (2 * θ) + b3 * cos (3 * θ) + b4 * cos (4 * θ) + b5 * cos (5 * θ) + b6 * cos (6 * θ) + b7 * cos (7 * θ) := 
by 
  sorry

theorem sum_of_squares :
  let (b1, b2, b3, b4, b5, b6, b7) := constants
  b1^2 + b2^2 + b3^2 + b4^2 + b5^2 + b6^2 + b7^2 = 1785 / 4096 := 
by 
  sorry

end cosine_expansion_sum_of_squares_l485_485602


namespace eval_fraction_subtraction_l485_485234

theorem eval_fraction_subtraction : 
  (3^2 + 5^2 + 7^2 = 83) → 
  (2^2 + 4^2 + 6^2 = 56) → 
  (83/56 - 56/83 = 3753/4648) := 
by 
  intros h1 h2 
  rw [h1, h2] 
  norm_num 
  sorry

end eval_fraction_subtraction_l485_485234


namespace priyas_fathers_age_l485_485827

-- Define Priya's age P and her father's age F
variables (P F : ℕ)

-- Define the conditions
def conditions : Prop :=
  F - P = 31 ∧ P + F = 53

-- Define the theorem to be proved
theorem priyas_fathers_age (h : conditions P F) : F = 42 :=
sorry

end priyas_fathers_age_l485_485827


namespace part1_part2_l485_485703

open Real

def f (x a : ℝ) := abs (x + 2 * a) + abs (x - 1)

section part1

variable (x : ℝ)

theorem part1 (a : ℝ) (h : a = 1) : f x a ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
by
  sorry

end part1

section part2

noncomputable def g (a : ℝ) := abs ((1 : ℝ) / a + 2 * a) + abs ((1 : ℝ) / a - 1)

theorem part2 {a : ℝ} (h : a ≠ 0) : g a ≤ 4 ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end part2

end part1_part2_l485_485703


namespace tournament_committee_count_l485_485745

theorem tournament_committee_count :
  let teams := 6
  let members_per_team := 8
  let host_team_choices := Nat.choose 8 3
  let regular_non_host_choices := Nat.choose 8 2
  let special_non_host_choices := Nat.choose 8 3
  let total_regular_non_host_choices := regular_non_host_choices ^ 4 
  let combined_choices_non_host := total_regular_non_host_choices * special_non_host_choices
  let combined_choices_host_non_host := combined_choices_non_host * host_team_choices
  let total_choices := combined_choices_host_non_host * teams
  total_choices = 11568055296 := 
by {
  let teams := 6
  let members_per_team := 8
  let host_team_choices := Nat.choose 8 3
  let regular_non_host_choices := Nat.choose 8 2
  let special_non_host_choices := Nat.choose 8 3
  let total_regular_non_host_choices := regular_non_host_choices ^ 4 
  let combined_choices_non_host := total_regular_non_host_choices * special_non_host_choices
  let combined_choices_host_non_host := combined_choices_non_host * host_team_choices
  let total_choices := combined_choices_host_non_host * teams
  have h_total_choices_eq : total_choices = 11568055296 := sorry
  exact h_total_choices_eq
}

end tournament_committee_count_l485_485745


namespace shifted_sine_function_l485_485733

theorem shifted_sine_function :
  ∀ x : ℝ, (2 * Real.sin (2 * x + π / 6)) = (2 * Real.sin (2 * (x - π / 4) + π / 6)) ↔
            (2 * Real.sin (2 * x - π / 3)) :=
by
  sorry

end shifted_sine_function_l485_485733


namespace problem_solution_l485_485192

-- Define the function f(x) = |sin x|
def f (x : ℝ) : ℝ := abs (sin x)

-- Define the property of being even
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Define the property of being monotonically increasing in [0, 1]
def is_monotone_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) := ∀ x y : ℝ, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

-- The main theorem: f(x) = |sin x| is even and monotonically increasing in [0, 1]
theorem problem_solution :
  is_even f ∧ is_monotone_increasing_on_interval f 0 1 := 
sorry

end problem_solution_l485_485192


namespace gcd_of_45_and_75_l485_485084

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l485_485084


namespace range_f_l485_485223

-- Definition of the function
def f (k : ℤ) : ℤ := 
  let k3 := (3 * ((k + 1) / 3 - 1) + if (k % 3).abs ≤ 1 then (k % 3).abs else -1)
  let k5 := (5 * (((2 * k + 2) / 5 - 1) + if ((2 * k) % 5).abs ≤ 2 then ((2 * k) % 5).abs else -2))
  let k7 := (7 * (((3 * k + 3) / 7 - 1) + if ((3 * k) % 7).abs ≤ 3 then ((3 * k) % 7).abs else -3))
  k3 + k5 + k7 - 6 * k

-- The theorem stating that the range of f is {-6, -5, ... , 6}
theorem range_f : 
  set.range f = {i : ℤ | -6 ≤ i ∧ i ≤ 6} :=
  sorry

end range_f_l485_485223


namespace gcd_45_75_l485_485117

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485117


namespace max_value_of_f_sum_extreme_l485_485701

noncomputable def f (a x : ℝ) : ℝ := (x^2 / 2) - 4 * a * x + a * real.log x + 3 * a^2 + 2 * a

noncomputable def g (a : ℝ) : ℝ := a * real.log a - 2 * a^2 + 3 * a

theorem max_value_of_f_sum_extreme (a : ℝ) (h : 0 < a) :
    (∀ a ≥ 1/4, ∀ x_1 x_2, (f a x_1 = f a x_1) ∧ (f a x_2 = f a x_2) →
       (x_1 = 2 * a - sqrt (4 * a^2 - a)) ∧
       (x_2 = 2 * a + sqrt (4 * a^2 - a))) →
    ∀ a > 1/4, g a ≤ 1 :=
by
  sorry

end max_value_of_f_sum_extreme_l485_485701


namespace number_of_unique_circles_l485_485796

def is_square (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (P Q R T : ℝ × ℝ), 
  Set.finite S ∧
  S = {P, Q, R, T} ∧
  dist P Q = dist Q R ∧ dist R T = dist T P ∧ 
  dist P R = dist Q T ∧
  dist P R = (dist P Q * real.sqrt 2)

theorem number_of_unique_circles (S : Set (ℝ × ℝ)) (h : is_square S) : 
  ∃ n : ℕ, n = 3 :=
by
  sorry

end number_of_unique_circles_l485_485796


namespace _l485_485787

noncomputable def polynomial := λ t : ℝ, t^3 - 2022 * t^2 + 2022 * t - 337

noncomputable def roots := {p q r : ℝ // polynomial p = 0 ∧ polynomial q = 0 ∧ polynomial r = 0 ∧
                            p + q + r = 2022 ∧
                            p * q + q * r + r * p = 2022 ∧
                            p * q * r = 337}

noncomputable def x (p q r : ℝ) := (q-1) * ((2022 - q) / (r-1) + (2022 - r) / (p-1))
noncomputable def y (p q r : ℝ) := (r-1) * ((2022 - r) / (p-1) + (2022 - p) / (q-1))
noncomputable def z (p q r : ℝ) := (p-1) * ((2022 - p) / (q-1) + (2022 - q) / (r-1))

def main_theorem (p q r : ℝ) (h : p ∈ roots ∧ q ∈ roots ∧ r ∈ roots) : 
  x p q r * y p q r * z p q r - q * r * x p q r - r * p * y p q r - p * q * z p q r = -674 :=
sorry

end _l485_485787


namespace average_speed_is_approx_595_24_l485_485148

def distance_meters : ℕ := 10000
def time_minutes : ℕ := 28

def distance_centimeters : ℕ := distance_meters * 100
def time_seconds : ℕ := time_minutes * 60

def average_speed_cm_per_sec : ℝ := distance_centimeters / time_seconds

theorem average_speed_is_approx_595_24 :
  abs (average_speed_cm_per_sec - 595.24) < 0.01 := 
sorry

end average_speed_is_approx_595_24_l485_485148


namespace time_to_pass_tree_l485_485179

-- Definitions
def length_of_train : ℝ := 420 -- in meters
def speed_of_train_kmh : ℝ := 63 -- in km/hr
def kmh_to_mps (v_kmh : ℝ) : ℝ := v_kmh * (1000 / 3600)

-- Theorem statement
theorem time_to_pass_tree : 
  let speed_of_train_mps := kmh_to_mps speed_of_train_kmh in
  let time_to_pass := length_of_train / speed_of_train_mps in
  time_to_pass = 24 := 
by
  -- We will skip the proof steps here as instructed.
  sorry

end time_to_pass_tree_l485_485179


namespace sum_fraction_equals_result_l485_485985

theorem sum_fraction_equals_result :
  (finset.sum (finset.range 5000) (λ n, 1 / ((n + 1)^3 + (n + 1)^2))) = 
  1 - (1 / 25050001) :=
sorry

end sum_fraction_equals_result_l485_485985


namespace dwarf_wage_increase_factor_l485_485336

-- Definitions of labor supply and demand for dwarves
def w_dwar_S (L : ℝ) : ℝ := 1 + (L / 3)
def w_dwar_D (L : ℝ) : ℝ := 10 - (2 * L / 3)

-- Definitions of labor supply and demand for elves
def w_elf_S (L : ℝ) : ℝ := 3 + L
def w_elf_D (L : ℝ) : ℝ := 18 - (2 * L)

-- Define the condition that wages must be equal for both clans
def equal_wages (w : ℝ) : Prop :=
  ∃ L_dwar L_elf, w_dwar_S L_dwar = w_dwar_D L_dwar ∧ w_elf_S L_elf = w_elf_D L_elf ∧ w_dwar_S L_dwar = w_elf_S L_elf ∧ w = w_dwar_S L_dwar

-- Prove the wage increase factor for dwarves
theorem dwarf_wage_increase_factor : 
  ∃ w_initial w_new, (w_initial = 4) ∧ (w_new = 5) ∧ (w_new / w_initial = 1.25) := by
  sorry

end dwarf_wage_increase_factor_l485_485336


namespace inverse_function_of_pow2_l485_485239

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem inverse_function_of_pow2 :
  ∀ (x : ℝ), 4 ≤ x → f⁻¹(x) = Real.log x / Real.log 2 :=
by
  -- Skip the proof with sorry
  sorry

end inverse_function_of_pow2_l485_485239


namespace pyramid_edge_length_l485_485947

-- Define the problem parameters
def side_length : ℕ := 12
def height : ℕ := 15

-- Define the measure of the edges based on the conditions
def diagonal_square_base : ℝ := real.sqrt (side_length ^ 2 + side_length ^ 2)
def half_diagonal : ℝ := diagonal_square_base / 2
def slant_edge : ℝ := real.sqrt (height ^ 2 + half_diagonal ^ 2)
def total_edge_length : ℝ := 4 * side_length + 4 * slant_edge

-- Define the correct answer
def correct_answer : ℝ := 117

-- State the theorem
theorem pyramid_edge_length :
  (total_edge_length).round = correct_answer :=
sorry

end pyramid_edge_length_l485_485947


namespace wang_hao_not_last_l485_485204

-- Define the total number of ways to select and arrange 3 players out of 6
def ways_total : ℕ := Nat.factorial 6 / Nat.factorial (6 - 3)

-- Define the number of ways in which Wang Hao is the last player
def ways_wang_last : ℕ := Nat.factorial 5 / Nat.factorial (5 - 2)

-- Proof statement
theorem wang_hao_not_last : ways_total - ways_wang_last = 100 :=
by sorry

end wang_hao_not_last_l485_485204


namespace cherry_cost_l485_485839

theorem cherry_cost (x : ℝ) : (∃ y : ℝ, y = 16 * x) :=
by
  use 16 * x
  sorry

end cherry_cost_l485_485839


namespace mosquito_maximum_journey_length_l485_485165

theorem mosquito_maximum_journey_length :
  let L := λ (x : ℝ) (y : ℝ) (z : ℝ), real.sqrt (x^2 + y^2 + z^2) in
  let edge_lengths := [1, 2, 3] in
  let face_diagonal_lengths := [real.sqrt (1^2 + 2^2), real.sqrt (2^2 + 3^2), real.sqrt (1^2 + 3^2)] in
  let space_diagonal := real.sqrt (1^2 + 2^2 + 3^2) in
  (4 * space_diagonal + 2 * real.sqrt (2^2 + 3^2)) = 4 * real.sqrt (14) + 2 * real.sqrt (13) :=
sorry

end mosquito_maximum_journey_length_l485_485165


namespace allocation_schemes_l485_485566

theorem allocation_schemes (
  classes : ℕ := 3
  spots : ℕ := 6
  min_spots_per_class : ℕ := 1 
) : classes = 3 ∧ spots = 6 ∧ min_spots_per_class ≥ 1 → 
    ∃ (num_schemes : ℕ), num_schemes = 10 := 
by
  intros h
  sorry

end allocation_schemes_l485_485566


namespace jill_braiding_time_l485_485367

-- Definitions based on the conditions
def num_dancers : ℕ := 15
def braids_per_dancer : ℕ := 10
def seconds_per_braid : ℕ := 45
def seconds_per_minute : ℕ := 60

-- Statement of the problem
theorem jill_braiding_time : 
  (num_dancers * braids_per_dancer * seconds_per_braid) / seconds_per_minute = 112.5 := 
by
  sorry

end jill_braiding_time_l485_485367


namespace roots_sum_zero_l485_485386

def Vieta_roots (R : Type) [CommRing R] (a b c : R) : Prop := 
  a + b + c = 6 ∧ ab + ac + bc = 8 ∧ abc = 3

theorem roots_sum_zero {R : Type} [CommRing R] {a b c : R} 
  (h : Vieta_roots a b c) :
  (a / (b * c + 2) + b / (a * c + 2) + c / (a * b + 2) = 0) := 
sorry

end roots_sum_zero_l485_485386


namespace solveForY_l485_485493

-- Define the conditions of the problem
def givenEquation (y : ℝ) : Prop := (40 / 80) = (sqrt (y / 80))

-- Theorem statement to show that if the condition holds, then y = 20
theorem solveForY (y : ℝ) (h : givenEquation y) : y = 20 :=
sorry

end solveForY_l485_485493


namespace circle_radius_tangent_lines_l485_485931

noncomputable def circle_radius (k : ℝ) (r : ℝ) : Prop :=
  k > 8 ∧ r = k / Real.sqrt 2 ∧ r = |k - 8|

theorem circle_radius_tangent_lines :
  ∃ k r : ℝ, k > 8 ∧ r = (k / Real.sqrt 2) ∧ r = |k - 8| ∧ r = 8 * Real.sqrt 2 :=
by
  sorry

end circle_radius_tangent_lines_l485_485931


namespace volleyball_team_selection_l485_485418

/-- The volleyball team has 16 players including 4 specific quadruplets. The task is to choose 6 starters with exactly 1 quadruplet. -/
theorem volleyball_team_selection : 
  let players_with_quadruplets := 16
  let quadruplets := 4
  let starters := 6
  (∃ (quadruplet_starters: Finset (Fin 16)) (team: Finset (Fin 16)),
    quadruplet_starters.card = 1 ∧ team.card = 5 ∧ quadruplet_starters ∩ team = ∅ ∧
    quadruplet_starters ∪ team ⊆ (Finset.range 16) ∧ quadruplet_starters ∪ team).card = starters 
→  4 * (choose 12 5) = 3168 := 
by
  sorry

end volleyball_team_selection_l485_485418


namespace number_of_valid_paths_chessboard_6x6_l485_485166

theorem number_of_valid_paths_chessboard_6x6 : 
    let n := 6 in Catalan n = 132 :=
by
  sorry

end number_of_valid_paths_chessboard_6x6_l485_485166


namespace mean_median_mode_order_l485_485340

def dataset : List ℕ :=
  List.replicate 12 (List.range' 1 29).sum ++ List.replicate 13 29 ++ List.replicate 12 30 ++ List.replicate 8 31

noncomputable def mean (data : List ℕ) : ℕ :=
  (data.sum) / data.length

noncomputable def median (data : List ℕ) : ℕ :=
  let sorted_data := data.sort
  if sorted_data.length % 2 = 0 then
    (sorted_data[sorted_data.length / 2 - 1] + sorted_data[sorted_data.length / 2]) / 2
  else sorted_data[sorted_data.length / 2]

noncomputable def mode_median (data : List ℕ) : ℕ :=
  let occurrences := data.foldr (fun x countMap => countMap.insert x ((countMap.find! x).getOrElse 0 + 1)) (RBMap.empty ℕ ℕ)
  let modes := occurrences.toList.filter (fun (key, value) => value = occurrences.foldl (fun max_val (_, value) => max max_val value) 0)
  let sorted_modes := modes.map Prod.fst
  if sorted_modes.length % 2 = 0 then
    (sorted_modes[sorted_modes.length / 2 - 1] + sorted_modes[sorted_modes.length / 2]) / 2
  else sorted_modes[sorted_modes.length / 2]

theorem mean_median_mode_order :
  let data := dataset
  let μ := mean data
  let M := median data
  let d := mode_median data
  (d < M) ∧ (M < μ) :=
by
  -- Proof steps here
  sorry

end mean_median_mode_order_l485_485340


namespace problem_solution_l485_485420

noncomputable def time_without_distraction : ℝ :=
  let rate_A := 1 / 10
  let rate_B := 0.75 * rate_A
  let rate_C := 0.5 * rate_A
  let combined_rate := rate_A + rate_B + rate_C
  1 / combined_rate

noncomputable def time_with_distraction : ℝ :=
  let rate_A := 0.9 * (1 / 10)
  let rate_B := 0.9 * (0.75 * (1 / 10))
  let rate_C := 0.9 * (0.5 * (1 / 10))
  let combined_rate := rate_A + rate_B + rate_C
  1 / combined_rate

theorem problem_solution :
  time_without_distraction = 40 / 9 ∧
  time_with_distraction = 44.44 / 9 := by
  sorry

end problem_solution_l485_485420


namespace percentage_loss_in_wages_l485_485940

-- Define the initial wages and the transformations
def initial_wages : ℝ := 100
def decreased_wages (w : ℝ) : ℝ := w * 0.5
def increased_wages (w : ℝ) : ℝ := w * 1.5

-- Define the final wages after decrease and increase
def final_wages := increased_wages (decreased_wages initial_wages)

-- Calculate the percentage loss
def percentage_loss (initial final : ℝ) : ℝ := ((initial - final) / initial) * 100

-- Statement to prove
theorem percentage_loss_in_wages :
  percentage_loss initial_wages final_wages = 25 := by
  sorry

end percentage_loss_in_wages_l485_485940


namespace option_B_option_D_l485_485130

-- Option B: If c = 2*a*cos B, then Triangle ABC is isosceles
theorem option_B (A B C : ℝ) (a b c : ℝ) (h1 : c = 2 * a * Real.cos B) :
  (triangle_is_isosceles ABC a b c) := sorry

-- Option D: If sin^2 A + sin^2 B < sin^2 C, then Triangle ABC is obtuse
theorem option_D (A B C : ℝ) (a b c : ℝ) (h2 : Real.sin A ^ 2 + Real.sin B ^ 2 < Real.sin C ^ 2) :
  (triangle_is_obtuse ABC a b c) := sorry

-- Additional Definitions, Assume these helper definitions exist for the purposes of illustrating the statements in Lean 4

def triangle_is_isosceles (ABC : ℝ) (a b c : ℝ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

def triangle_is_obtuse (ABC : ℝ) (a b c : ℝ) : Prop :=
  ∃ (angle : ℝ), angle > π / 2 ∧ (angle = A ∨ angle = B ∨ angle = C)

end option_B_option_D_l485_485130


namespace rancher_cows_l485_485170

theorem rancher_cows : ∃ (C H : ℕ), (C = 5 * H) ∧ (C + H = 168) ∧ (C = 140) := by
  sorry

end rancher_cows_l485_485170


namespace rahul_deepak_age_ratio_l485_485472

-- Define the conditions
variables (R D : ℕ)
axiom deepak_age : D = 33
axiom rahul_future_age : R + 6 = 50

-- Define the theorem to prove the ratio
theorem rahul_deepak_age_ratio : R / D = 4 / 3 :=
by
  -- Placeholder for proof
  sorry

end rahul_deepak_age_ratio_l485_485472


namespace hyperbola_eccentricity_proof_l485_485704

def hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_asymptote : b = 2 * sqrt 2 * a) : ℝ :=
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e

theorem hyperbola_eccentricity_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_asymptote : b = 2 * sqrt 2 * a) : hyperbola_eccentricity a b ha hb h_asymptote = 3 := by  
  sorry

end hyperbola_eccentricity_proof_l485_485704


namespace even_count_in_pascal_triangle_l485_485612

-- Define the binomial coefficient as a function
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define a predicate to check if a number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Count the number of even integers in the top 15 rows of Pascal's Triangle
def count_even_pascal (rows : ℕ) : ℕ :=
  (Finset.range rows).sum (λ n => (Finset.range (n + 1)).count (λ k => is_even (binom n k)))

-- Statement of the problem
theorem even_count_in_pascal_triangle : count_even_pascal 15 = 84 :=
  by
    sorry

end even_count_in_pascal_triangle_l485_485612


namespace arithmetic_progression_pairs_count_l485_485221

theorem arithmetic_progression_pairs_count (x y : ℝ) 
  (h1 : x = (15 + y) / 2)
  (h2 : x + x * y = 2 * y) : 
  (∃ x1 y1, x1 = (15 + y1) / 2 ∧ x1 + x1 * y1 = 2 * y1 ∧ x1 = (9 + 3 * Real.sqrt 7) / 2 ∧ y1 = -6 + 3 * Real.sqrt 7) ∨ 
  (∃ x2 y2, x2 = (15 + y2) / 2 ∧ x2 + x2 * y2 = 2 * y2 ∧ x2 = (9 - 3 * Real.sqrt 7) / 2 ∧ y2 = -6 - 3 * Real.sqrt 7) := 
sorry

end arithmetic_progression_pairs_count_l485_485221


namespace final_number_independent_of_order_l485_485737

theorem final_number_independent_of_order 
  (p q r : ℕ) : 
  ∃ k : ℕ, 
    (p % 2 ≠ 0 ∨ q % 2 ≠ 0 ∨ r % 2 ≠ 0) ∧ 
    (∀ (p' q' r' : ℕ), 
       p' + q' + r' = p + q + r → 
       p' % 2 = p % 2 ∧ q' % 2 = q % 2 ∧ r' % 2 = r % 2 → 
       (p' = 1 ∧ q' = 0 ∧ r' = 0 ∨ 
        p' = 0 ∧ q' = 1 ∧ r' = 0 ∨ 
        p' = 0 ∧ q' = 0 ∧ r' = 1) → 
       k = p ∨ k = q ∨ k = r) := 
sorry

end final_number_independent_of_order_l485_485737


namespace evaluate_expression_l485_485992

theorem evaluate_expression :
  (1 / (-5^3)^4) * (-5)^15 * 5^2 = -3125 :=
by
  sorry

end evaluate_expression_l485_485992


namespace cross_country_race_winning_scores_l485_485741

-- Lean 4 statement
noncomputable def differentWinningScoresPossible : ℕ :=
  19

theorem cross_country_race_winning_scores :
  ∀ (total_positions : fin 12) 
    (sum_points : fin 12 → ℕ) 
    (min_score max_score : ℕ),
    (∀ n : ℕ, n ≤ 12 → sum_points ⟨n, nat.lt_succ_iff.mpr (nat.le_of_lt_succ (nat.lt_base n 12))⟩ = n) →
    (sum_points ⟨1, _⟩ + sum_points ⟨2, _⟩ + sum_points ⟨3, _⟩ + sum_points ⟨4, _⟩ + sum_points ⟨5, _⟩ + sum_points ⟨6, _⟩ = min_score) →
    (min_score = 21) →
    (max_score = 39) →
    (total_positions = 78) →
    ((λ total sum_points, differentWinningScoresPossible) total_positions.sum_points = 19) :=
by
  intros
  sorry

end cross_country_race_winning_scores_l485_485741


namespace probability_floor_sqrt_50y_226_given_floor_sqrt_y_16_l485_485388

open Real

noncomputable def y (α : Type) [is_uniform α 200 300] : α → ℝ := sorry

theorem probability_floor_sqrt_50y_226_given_floor_sqrt_y_16 :
  (∃ y : ℝ, 200 ≤ y ∧ y < 300 ∧ ⌊sqrt y⌋ = 16) →
  0 = (∑' (y : ℝ), cond (⌊sqrt (50 * y)⌋ = 226) 1 0) := 
sorry

end probability_floor_sqrt_50y_226_given_floor_sqrt_y_16_l485_485388


namespace rancher_cows_l485_485171

theorem rancher_cows : ∃ (C H : ℕ), (C = 5 * H) ∧ (C + H = 168) ∧ (C = 140) := by
  sorry

end rancher_cows_l485_485171


namespace sector_area_correct_l485_485922

def radius : ℝ := 18
def angle_degrees : ℝ := 42

def sector_area (r θ : ℝ) : ℝ :=
  (θ / 360) * Real.pi * r^2

theorem sector_area_correct :
  sector_area radius angle_degrees = 118.752 :=
by
  sorry

end sector_area_correct_l485_485922


namespace larger_number_is_84_l485_485465
open BigOperators

/-
  The h.c.f and l.c.m of two numbers are 84 and 21 respectively.
  The ratio of the two numbers is 1:4.
  Which of the two numbers is 84, the larger or the smaller one?
-/

noncomputable def hcf : ℕ := 84
noncomputable def lcm : ℕ := 21

def ratio_condition (A B : ℕ) : Prop := B = 4 * A

def product_condition (A B : ℕ) : Prop := A * B = hcf * lcm

theorem larger_number_is_84 (A B : ℕ)
  (h_hcf : nat.gcd A B = hcf)
  (h_lcm : nat.lcm A B = lcm)
  (h_ratio : ratio_condition A B)
  (h_prod : product_condition A B) :
  B = 84 :=
sorry

end larger_number_is_84_l485_485465


namespace x_seq_converges_to_sqrt2_l485_485352

-- Given function and initial value
def f (x : ℝ) : ℝ := x^2 - 2
def x1 : ℝ := 2

-- Newton's method recurrence relation
def x_next (xn : ℝ) : ℝ := 0.5 * (xn + 2 / xn)

-- Sequence definition
def x_seq (n : ℕ) : ℝ :=
  Nat.recOn n x1 (λ n xn, x_next xn)

-- Formal proof statement
theorem x_seq_converges_to_sqrt2 :
  filter.tendsto x_seq filter.at_top (nhds (Real.sqrt 2)) :=
sorry

end x_seq_converges_to_sqrt2_l485_485352


namespace g_neither_even_nor_odd_l485_485768

def g (x : ℝ) : ℝ := ⌈x⌉ + 1/2

theorem g_neither_even_nor_odd : ¬(∀ x : ℝ, g (-x) = g x) ∧ ¬(∀ x : ℝ, g (-x) = -g x) := by
  sorry

end g_neither_even_nor_odd_l485_485768


namespace find_C_probability_within_r_l485_485752

noncomputable def probability_density (x y R : ℝ) (C : ℝ) : ℝ :=
if x^2 + y^2 <= R^2 then C * (R - Real.sqrt (x^2 + y^2)) else 0

noncomputable def total_integral (R : ℝ) (C : ℝ) : ℝ :=
∫ (x : ℝ) in -R..R, ∫ (y : ℝ) in -R..R, probability_density x y R C

theorem find_C (R : ℝ) (hR : 0 < R) : 
  (∫ (x : ℝ) in -R..R, ∫ (y : ℝ) in -R..R, probability_density x y R C) = 1 ↔ 
  C = 3 / (π * R^3) := 
by 
  sorry

theorem probability_within_r (R r : ℝ) 
  (hR : 0 < R) (hr : 0 < r) (hrR : r <= R) (P : ℝ) : 
  (∫ (x : ℝ) in -r..r, ∫ (y : ℝ) in -r..r, probability_density x y R (3 / (π * R^3))) = P ↔ 
  (R = 2 ∧ r = 1 → P = 1 / 2) := 
by 
  sorry

end find_C_probability_within_r_l485_485752


namespace gcd_45_75_l485_485098

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485098


namespace table_tennis_matches_l485_485747

theorem table_tennis_matches (n : ℕ) :
  ∃ x : ℕ, 3 * 2 - x + n * (n - 1) / 2 = 50 ∧ x = 1 :=
by
  sorry

end table_tennis_matches_l485_485747


namespace unique_zero_in_interval_l485_485732

theorem unique_zero_in_interval (a : ℝ) :
  (∃! x ∈ set.Ioo (-1 : ℝ) (1 : ℝ), 3*x^2 + 2*x - a = 0) ↔ a = -1/3 ∨ (1 < a ∧ a < 5) :=
by { sorry }

end unique_zero_in_interval_l485_485732


namespace secret_eggs_count_l485_485891

theorem secret_eggs_count :
  ∀ (total_items candy_count : ℕ), 
  total_items = 3554 → candy_count = 3409 → total_items - candy_count = 145 :=
by
  intros total_items candy_count h_total h_candy
  rw [h_total, h_candy]
  rfl

end secret_eggs_count_l485_485891


namespace gcd_45_75_l485_485074

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485074


namespace evaluate_expression_l485_485233

variable (b : ℝ) (h : b ≠ 0)

theorem evaluate_expression :
  (1/9 * b^0 + (1/(9*b))^0 - 27^(-1/3) - (-27)^(-3/4) = 1 + 1/9 - 1/3 + 1/(3^(9/4))) :=
  by sorry

end evaluate_expression_l485_485233


namespace radius_of_circle_l485_485822

theorem radius_of_circle (M C A B: Point) (MC : ℝ) (angle_BMC : ℝ) (M_eq_A : A = midpoint B M) 
  (MC_value : MC = 2) (angle_BMC_value : angle_BMC = 45) :
  radius circle = 2 := 
by
  sorry

end radius_of_circle_l485_485822


namespace probability_y_eq_2x_l485_485485

theorem probability_y_eq_2x :
  let outcomes := { (x, y) | x ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧ y ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) },
      favorable := { (x, y) | (x, y) ∈ outcomes ∧ y = 2 * x }
  in (favorable.card : ℚ) / outcomes.card = 1 / 12 := 
sorry

end probability_y_eq_2x_l485_485485


namespace solution_set_inequality_l485_485881

-- Statement of the problem
theorem solution_set_inequality :
  {x : ℝ | 1 / x < 1 / 2} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 2} :=
sorry

end solution_set_inequality_l485_485881


namespace expression_evaluation_l485_485231

theorem expression_evaluation : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 :=
by
  sorry

end expression_evaluation_l485_485231


namespace zero_and_one_positions_l485_485161

theorem zero_and_one_positions (a : ℝ) :
    (0 = (a + (-a)) / 2) ∧ (1 = ((a + (-a)) / 2 + 1)) :=
by
  sorry

end zero_and_one_positions_l485_485161


namespace remainder_of_trailing_zeros_product_1_to_50_l485_485793

noncomputable def count_factors_5 (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

noncomputable def factorial_trailing_zeros (n : ℕ) : ℕ :=
  ∑ i in range (n + 1), count_factors_5 i

theorem remainder_of_trailing_zeros_product_1_to_50 : (factorial_trailing_zeros 50) % 100 = 14 :=
by
  sorry

end remainder_of_trailing_zeros_product_1_to_50_l485_485793


namespace smallest_positive_period_of_f_l485_485708

noncomputable def vec_a (x ϕ : ℝ) : ℝ × ℝ := (Real.sin (x + ϕ), 2)
noncomputable def vec_b (x ϕ : ℝ) : ℝ × ℝ := (1, Real.cos (x + ϕ))

noncomputable def f (x ϕ : ℝ) : ℝ :=
  let a := vec_a x ϕ
  let b := vec_b x ϕ
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2)

theorem smallest_positive_period_of_f (x ϕ : ℝ) : ∃ T > 0, (∀ t, f (x + T) ϕ = f x ϕ) ∧ (∀ T' > 0, (∀ t, f (x + T') ϕ = f x ϕ) → T ≤ T') :=
sorry

end smallest_positive_period_of_f_l485_485708


namespace remainder_when_xy_divided_by_n_l485_485387

theorem remainder_when_xy_divided_by_n (n : ℕ) (x y : ℤ)
  (hn : 0 < n)
  (hx : IsUnit x)
  (hy : IsUnit y)
  (hxy : x ≡ y⁻¹ ∧ y ∈ Units ℤ) :
  (x * y) % n = 1 % n :=
by
  sorry

end remainder_when_xy_divided_by_n_l485_485387


namespace cost_of_leveling_walk_is_660_rs_l485_485918

noncomputable def π : ℝ := Real.pi

def radius_small : ℝ := 16
def width_walk : ℝ := 3
def radius_large : ℝ := radius_small + width_walk
def cost_per_m2 : ℝ := 2

def area_circle (r : ℝ) : ℝ := π * r^2

def area_small_circle : ℝ := area_circle radius_small
def area_large_circle : ℝ := area_circle radius_large
def area_walk : ℝ := area_large_circle - area_small_circle
def cost : ℝ := area_walk * cost_per_m2

theorem cost_of_leveling_walk_is_660_rs : cost ≈ 660 := by
  sorry

end cost_of_leveling_walk_is_660_rs_l485_485918


namespace even_integers_in_pascals_triangle_top_15_rows_l485_485620

/-- Prove that the total number of even integers in the top 15 rows of Pascal's Triangle is exactly 90.
  Pascal's Triangle's elements, binomial(n,k), are even unless every binary digit of k 
  is present in n when both are expressed in binary. -/
theorem even_integers_in_pascals_triangle_top_15_rows : 
  ∑ n in Finset.range 15, ∑ k in Finset.range (n + 1), if (∀ i, (n.bits.get i) = 1 → (k.bits.get i) = 1) then 0 else 1 = 90 :=
sorry

end even_integers_in_pascals_triangle_top_15_rows_l485_485620


namespace imo_inequality_l485_485395

variable {a b c : ℝ}

theorem imo_inequality (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : (a + b) * (b + c) * (c + a) = 1) :
  (a^2 / (1 + Real.sqrt (b * c))) + (b^2 / (1 + Real.sqrt (c * a))) + (c^2 / (1 + Real.sqrt (a * b))) ≥ (1 / 2) := 
sorry

end imo_inequality_l485_485395


namespace increased_percentage_l485_485494

theorem increased_percentage (x : ℝ) (p : ℝ) (h : x = 75) (h₁ : p = 1.5) : x + (p * x) = 187.5 :=
by
  sorry

end increased_percentage_l485_485494


namespace center_of_circle_symmetry_l485_485853

theorem center_of_circle_symmetry (a : ℝ) (h : a ≠ 0) :
    ∃ x y : ℝ, (x = -a) ∧ (y = a) ∧ (x + y = 0) :=
by
  have center_x := -a
  have center_y := a
  use center_x, center_y
  split
  case left => exact rfl
  case right => split
  case left => exact rfl
  case right => sorry

end center_of_circle_symmetry_l485_485853


namespace Ron_spends_15_dollars_l485_485541

theorem Ron_spends_15_dollars (cost_per_bar : ℝ) (sections_per_bar : ℕ) (num_scouts : ℕ) (s'mores_per_scout : ℕ) :
  cost_per_bar = 1.50 ∧ sections_per_bar = 3 ∧ num_scouts = 15 ∧ s'mores_per_scout = 2 →
  cost_per_bar * (num_scouts * s'mores_per_scout / sections_per_bar) = 15 :=
by
  sorry

end Ron_spends_15_dollars_l485_485541


namespace tangent_line_eq_extremum_range_always_geq_l485_485301

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * Real.log (x + 1)

-- Part (1)
theorem tangent_line_eq (a : ℝ) (x y : ℝ) (h : a = -2 ∧ x = 0 ∧ y = f 0 -2) :
  x + y - 1 = 0 :=
by sorry

-- Part (2)
theorem extremum_range (a : ℝ) : 
  (∃ x : ℝ, ∀ {h : x > -1}, x * (Real.exp x + a / (x + 1)) = 0) ↔ (a < 0) :=
by sorry

-- Part (3)
theorem always_geq (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 1 - Real.sin x) ↔ (a = -2) :=
by sorry

end tangent_line_eq_extremum_range_always_geq_l485_485301


namespace correct_cube_die_layout_l485_485550

-- Define initial conditions
def faces := {1, 2, 3, 4, 5, 6}

-- Define opposite pairs
def opposite_pairs : (ℕ × ℕ) → Prop
| (2, 4) => true
| (4, 2) => true
| (1, 5) => true
| (5, 1) => true
| (3, 6) => true
| (6, 3) => true
| _ => false

-- Define initial known positions
def initial_layout : ℕ → option ℕ
| 1 => some 1
| 2 => some 2
| _ => none

-- Define the correct answer layout to be proven
def correct_layout : ℕ → option ℕ
| 1 => some 1  -- A
| 2 => some 2  -- B
| 3 => some 4  -- E (opposite to 2)
| 4 => some 6  -- D
| 5 => some 5  -- F (opposite to 1)
| 6 => some 3  -- C
| _ => none

-- The theorem that needs to be proven
theorem correct_cube_die_layout :
  ∀ n, correct_layout n = initial_layout n ∨
    (∃ m, opposite_pairs (n, m) ∧ correct_layout m = initial_layout n) :=
by
  sorry

end correct_cube_die_layout_l485_485550


namespace even_count_in_top_15_rows_l485_485626

-- We want to count the number of even binomial coefficients in the first 15 rows of Pascal's Triangle.
-- Specifically, we are considering rows 0 to 14.

def is_even (n : ℕ) := n % 2 = 0

def count_even_binomials_in_pascals_triangle_up_to_row (max_row : ℕ) : ℕ := 
  ∑ n in Finset.range (max_row + 1), 
    ∑ k in Finset.range (n + 1), 
      if is_even (Nat.choose n k) then 1 else 0

theorem even_count_in_top_15_rows : count_even_binomials_in_pascals_triangle_up_to_row 14 = 49 := 
by 
  sorry

end even_count_in_top_15_rows_l485_485626


namespace range_of_a_for_increasing_function_l485_485326

theorem range_of_a_for_increasing_function 
  (f : ℝ → ℝ := λ x, if x ≥ 1 then (2 * a - 1) * x - 1 else x + 1)
  (is_increasing : ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≤ f x₂) :
  1 / 2 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_for_increasing_function_l485_485326


namespace gcd_45_75_l485_485107

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485107


namespace pyramid_height_l485_485946

theorem pyramid_height (perimeter : ℝ) (distance : ℝ) (height : ℝ) (side_length : ℝ) (half_diagonal : ℝ) :
  perimeter = 40 ∧ distance = 15 ∧ 
  side_length = perimeter / 4 ∧ 
  half_diagonal = side_length * Real.sqrt(2) / 2 ∧ 
  height = Real.sqrt(distance^2 - half_diagonal^2) → 
  height = 5 * Real.sqrt(7) := by
  intros
  sorry

end pyramid_height_l485_485946


namespace tangent_line_eqn_extremum_range_of_a_inequality_holds_l485_485299

noncomputable def f (x : ℝ) (a : ℝ) := Math.exp x + a * Real.log (x + 1)

theorem tangent_line_eqn (a : ℝ) (h : a = -2): 
    (∃ (m : ℝ), ∃ (b : ℝ), y = m * x + b ∧ m = -1 ∧ b = 1) → 
    x + y - 1 = 0 :=
  sorry

theorem extremum_range_of_a : 
    (a < 0) ↔ (∃ x : ℝ, x > -1 ∧ ∃ f' : ℝ → ℝ, f' x = Math.exp x + a / (x + 1) ∧ f' x = 0) :=
  sorry

theorem inequality_holds (a : ℝ) : 
    (a = -2) ↔ (∀ x : ℝ, x > -1 → (∃ g : ℝ → ℝ, g x = Math.sin x + Math.exp x + a * Real.log (x + 1) ∧ g x ≥ 1 - Math.sin x)) :=
  sorry

end tangent_line_eqn_extremum_range_of_a_inequality_holds_l485_485299


namespace gcd_45_75_l485_485103

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485103


namespace divisibility_by_2880_l485_485254

theorem divisibility_by_2880 (n : ℕ) : 
  (∃ t u : ℕ, (n = 16 * t - 2 ∨ n = 16 * t + 2 ∨ n = 8 * u - 1 ∨ n = 8 * u + 1) ∧ ¬(n % 3 = 0) ∧ ¬(n % 5 = 0)) ↔
  2880 ∣ (n^2 - 4) * (n^2 - 1) * (n^2 + 3) :=
sorry

end divisibility_by_2880_l485_485254


namespace log_inequalities_l485_485128

theorem log_inequalities :
  let A := ln 2 + ln 3 > 2 * ln (5/2)
  let B := 1/3 < ln 3 - ln 2 ∧ ln 3 - ln 2 < 1/2
  let C := ln 2 * ln 3 > 1
  let D := ln 3 / ln 2 < 3/2
  B := true :=
by
  sorry

end log_inequalities_l485_485128


namespace number_is_450064_l485_485558

theorem number_is_450064 : (45 * 10000 + 64) = 450064 :=
by
  sorry

end number_is_450064_l485_485558


namespace cathy_milk_drinking_l485_485905

/-- 
Cathy's milk drinking problem:
- Cathy drinks 60 ml of milk per day when lazing around.
- Each day she chases mice, she drinks a third more milk.
- Each day she gets chased by a dog, she drinks half as much again as when she chases mice.
- In the last two weeks, Cathy has been chasing mice on alternate days and was chased by a dog on two other days.
- Prove that in the last two weeks, Cathy drinks 1100 ml of milk.
 -/
theorem cathy_milk_drinking :
  let milk_per_day_lazing := 60
  let milk_per_day_chasing_mice := milk_per_day_lazing + (1 / 3) * milk_per_day_lazing
  let milk_per_day_chased_by_dog := milk_per_day_chasing_mice + (1 / 2) * milk_per_day_chasing_mice
  let days_chasing_mice := 7
  let days_chased_by_dog := 2
  let days_lazing := 14 - days_chasing_mice - days_chased_by_dog
  total_milk_total_drunk := (days_chasing_mice * milk_per_day_chasing_mice) +
                            (days_chased_by_dog * milk_per_day_chased_by_dog) +
                            (days_lazing * milk_per_day_lazing)
  in total_milk_total_drunk = 1100 :=
by
  sorry

end cathy_milk_drinking_l485_485905


namespace fill_diagram_count_l485_485996

theorem fill_diagram_count :
  ∃ f : (fin 4 → ℕ), 
    {n | (∀ i : fin 4, 1 ≤ f i ∧ f i ≤ 9) ∧ (∀ x y : fin 4, x ≠ y → f x ≠ f y) ∧ 
          f 0 > f 1 ∧
          f 0 > f 2 ∧
          f 2 > f 3 ∧
          f 1 = 2 ∧ 
          f 3 = 3} = 16 :=
sorry

end fill_diagram_count_l485_485996


namespace greatest_integer_leq_fraction_l485_485489

theorem greatest_integer_leq_fraction (N D : ℝ) (hN : N = 4^103 + 3^103 + 2^103) (hD : D = 4^100 + 3^100 + 2^100) :
  ⌊N / D⌋ = 64 :=
by
  sorry

end greatest_integer_leq_fraction_l485_485489


namespace three_digit_solutions_exist_l485_485131

theorem three_digit_solutions_exist :
  ∃ (x y z : ℤ), 100 ≤ x ∧ x ≤ 999 ∧ 
                 100 ≤ y ∧ y ≤ 999 ∧
                 100 ≤ z ∧ z ≤ 999 ∧
                 17 * x + 15 * y - 28 * z = 61 ∧
                 19 * x - 25 * y + 12 * z = 31 :=
by
    sorry

end three_digit_solutions_exist_l485_485131


namespace similarity_of_triangles_l485_485844

-- Variables
variables {A B C D E F M N P : Type}

-- Definitions
def is_triangle (T : Type) : Prop := sorry -- Placeholder for is_triangle definition
def midpoint (a b m : Type) : Prop := sorry -- Placeholder for midpoint definition
def bisector_meets_circumcircle (T : Type) (p : Type) : Prop := sorry -- Placeholder

-- Given conditions
axiom 
  (triangle_ABC : is_triangle A)
  (D_bisector : bisector_meets_circumcircle A D)
  (E_bisector : bisector_meets_circumcircle A E)
  (F_bisector : bisector_meets_circumcircle A F)
  (midpoint_M : midpoint B C M)
  (midpoint_N : midpoint C A N)
  (midpoint_P : midpoint A B P)

-- To Prove
theorem similarity_of_triangles
  (h1 : triangle_ABC)
  (h2 : D_bisector)
  (h3 : E_bisector)
  (h4 : F_bisector)
  (h5 : midpoint_M)
  (h6 : midpoint_N)
  (h7 : midpoint_P)
  : similar_triangles A M N A D E F := sorry

end similarity_of_triangles_l485_485844


namespace similarity_of_MPQ_and_ABC_l485_485803

variable {A B C G M X Y Q P : Point}
variable {ABC : Triangle}

/-- Definition of centroid -/
def is_centroid (ABC G : Triangle) : Prop :=
  G divides each median in the ratio 2 : 1

/-- Definition of midpoint -/
def is_midpoint (M : Point) (BC : Segment) : Prop :=
  M divides BC into two equal segments

/-- Definition of collinear points -/
def collinear (X Y G : Point) : Prop :=
  ∃ l : Line, X ∈ l ∧ Y ∈ l ∧ G ∈ l

/-- Definition of parallel segments -/
def parallel (XY BC : Segment) : Prop :=
  XY ∥ BC

/-- Definition of similar triangles -/
def similar (T1 T2 : Triangle) : Prop :=
  ∃ k, k > 0 ∧ T1 ~ T2

theorem similarity_of_MPQ_and_ABC
  (hG_centroid : is_centroid ABC G)
  (hM_midpoint : is_midpoint M ⟦B, C⟧)
  (hXY_parallel : parallel ⟦X, Y⟧ ⟦B, C⟧)
  (hcollinear : collinear X Y G)
  (hQ_intersection : q_intersection ⟦X, C⟧ ⟦G, B⟧ Q)
  (hP_intersection : p_intersection ⟦Y, B⟧ ⟦G, C⟧ P) :
  similar ⟦M, P, Q⟧ ⟦A, B, C⟧ :=
sorry

end similarity_of_MPQ_and_ABC_l485_485803


namespace time_spent_moving_l485_485442

noncomputable def time_per_trip_filling : ℝ := 15
noncomputable def time_per_trip_driving : ℝ := 30
noncomputable def time_per_trip_unloading : ℝ := 20
noncomputable def number_of_trips : ℕ := 10

theorem time_spent_moving :
  10.83 = (time_per_trip_filling + time_per_trip_driving + time_per_trip_unloading) * number_of_trips / 60 :=
by
  sorry

end time_spent_moving_l485_485442


namespace cole_backyard_length_l485_485593

noncomputable def cost_of_fence (L : ℝ) : ℝ :=
  let back_side_cost := 18 * 3
  let right_side_cost := L * 3
  let left_side_cost := L * 3
  let cole_cost := (back_side_cost / 2) + right_side_cost + ((2 / 3) * left_side_cost)
  cole_cost

theorem cole_backyard_length : ∃ (L : ℝ), cost_of_fence L = 72 ∧ L = 9 :=
by
  have h1 : 18 * 3 = 54 := rfl
  have back_side_cost_contrib := h1 / 2
  rw [back_side_cost_contrib] -- Assert that the back side contribution is 27

  have left_side_eq : ∀ (L : ℝ), (2 / 3) * (L * 3) = 2 * L := 
    by intro L; rw [← mul_assoc]; ring

  have cost_eq : ∀ (L : ℝ), 27 + 3 * L + 2 * L = 5 * L + 27 := 
    by intro L; ring

  use 9
  split
  show cost_of_fence 9 = 72
  rw [cost_of_fence, h1, lef_side_eq 9, cost_eq 9]
  norm_num
  rfl

  show 9 = 9
  rfl
  sorry

end cole_backyard_length_l485_485593


namespace rectangle_measurement_error_l485_485749

theorem rectangle_measurement_error
  (L W : ℝ)
  (x : ℝ)
  (h1 : ∀ x, L' = L * (1 + x / 100))
  (h2 : W' = W * 0.9)
  (h3 : A = L * W)
  (h4 : A' = A * 1.08) :
  x = 20 :=
by
  sorry

end rectangle_measurement_error_l485_485749


namespace area_of_triangle_l485_485709

theorem area_of_triangle : 
  ∀ (x y : ℝ), 
  |5 * x| + |12 * y| + |60 - 5 * x - 12 * y| = 60 →
  x ≥ 0 →
  y ≥ 0 →
  5 * x + 12 * y ≤ 60 →
  let vertices := [(0, 0), (0, 5), (12, 0)] in
  let area := (1 / 2) * |0 * (5 - 0) + 0 * (0 - 0) + 12 * (0 - 5)| in
  area = 30 := 
sorry

end area_of_triangle_l485_485709


namespace range_of_a1_l485_485825

theorem range_of_a1 (a1 : ℝ) :
  let outcome1 := 4 * a1 - 36
  let outcome2 := a1 + 6
  let outcome3 := (1 / 4) * a1 + 18
  let outcome4 := a1 + 18
  let winProbability := (if outcome1 > a1 then 1 else 0) +
                         (if outcome2 > a1 then 1 else 0) +
                         (if outcome3 > a1 then 1 else 0) +
                         (if outcome4 > a1 then 1 else 0) / 4
  in winProbability = (3 : ℝ) / 4 → 
    (a1 ≤ 12 ∨ a1 ≥ 24) :=
begin
  sorry
end

end range_of_a1_l485_485825


namespace probability_at_least_one_male_l485_485419

theorem probability_at_least_one_male :
  let total_students := 5
  let males := 3
  let females := 2
  let choose_2_students := (choose total_students 2)
  let choose_2_females := (choose females 2)
  1 - (choose_2_females / choose_2_students) = 9 / 10 :=
by {
  sorry
}

end probability_at_least_one_male_l485_485419


namespace smallest_munificence_of_monic_cubic_l485_485245

-- Define a monic cubic polynomial
def monic_cubic_poly (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b * x^2 + c * x + d

-- Define the notion of munificence as the maximum absolute value on the interval [-1, 1]
def munificence (q : ℝ → ℝ) : ℝ := 
  Sup (set.image (λ x, |q x|) (set.Icc (-1 : ℝ) 1))

-- Prove the smallest possible munificence of a monic cubic polynomial is 0
theorem smallest_munificence_of_monic_cubic : 
  ∃ (b c d : ℝ), munificence (monic_cubic_poly b c d) = 0 := 
sorry

end smallest_munificence_of_monic_cubic_l485_485245


namespace smallest_integer_in_set_A_is_neg3_l485_485880

noncomputable def smallest_integer_in_set_A : ℤ :=
  let A := {x : ℝ | |x - 2| ≤ 5}
  in if h : ∃ (m : ℤ), m ∈ A
     then Inf {m : ℤ | (m : ℝ) ∈ A ∧ m <= Inf {x : ℝ | x ∈ A}}
     else 0 -- fallback in case no integer exists (which won't happen here)

theorem smallest_integer_in_set_A_is_neg3 :
  smallest_integer_in_set_A = -3 :=
by
  let A := {x : ℝ | |x - 2| ≤ 5}
  have h : A = {x : ℝ | -3 ≤ x ∧ x ≤ 7} := by
    sorry -- elaborate on the steps to show the equivalence between sets
  suffices -3 ∈ A by
    sorry -- show that -3 is indeed in A and is the smallest zlib in A
  show -3 ∈ A 
  sorry -- prove membership of -3 in set A

end smallest_integer_in_set_A_is_neg3_l485_485880


namespace inv_sum_range_l485_485396

open Set

def is_positive_real (x : ℝ) : Prop := 0 < x
noncomputable def inv_sum (a b : ℝ) (h : a + b = 3) : ℝ := (1 / a) + (1 / b)

theorem inv_sum_range :
  ∀ (a b : ℝ), is_positive_real a → is_positive_real b → (a + b = 3) →
  ∃ (s : Set ℝ), s = Ici (4 / 3) ∧ inv_sum a b (by assumption) ∈ s :=
by
  intros a b ha hb hsum
  use Ici (4 / 3)
  split
  · ext x
    split
    · exact λ hx, hx
    · exact λ hx, hx
  · sorry

end inv_sum_range_l485_485396


namespace triangle_APQ_area_l485_485045

theorem triangle_APQ_area (A B C P Q : Type) [is_triangle A B C]
  (AB BC AC : ℝ)
  (hAB : dist A B = 15)
  (hBC : dist B C = 30)
  (hAC : dist A C = 20)
  (I : Type) [is_incenter I A B C]
  (hP : line_through_parallels I B P)
  (hQ : line_through_parallels I C Q) :
  area_of_triangle A P Q = √284765.625 :=
by
  sorry

end triangle_APQ_area_l485_485045


namespace arithmetic_progression_sum_15_terms_l485_485884

def arithmetic_progression_sum (a₁ d : ℚ) : ℚ :=
  15 * (2 * a₁ + (15 - 1) * d) / 2

def am_prog3_and_9_sum_and_product (a₁ d : ℚ) : Prop :=
  (a₁ + 2 * d) + (a₁ + 8 * d) = 6 ∧ (a₁ + 2 * d) * (a₁ + 8 * d) = 135 / 16

theorem arithmetic_progression_sum_15_terms (a₁ d : ℚ)
  (h : am_prog3_and_9_sum_and_product a₁ d) :
  arithmetic_progression_sum a₁ d = 37.5 ∨ arithmetic_progression_sum a₁ d = 52.5 :=
sorry

end arithmetic_progression_sum_15_terms_l485_485884


namespace equivalent_proof_problem_l485_485274

-- Given conditions and expressions
def expression_1 (a b : ℕ) (x y : ℕ) : ℤ := -2 * a^(2 : ℤ) * b^((y + 3) : ℤ)
def expression_2 (a b : ℕ) (x y : ℕ) : ℤ := 4 * a^((x) : ℤ) * b^(2 : ℤ)
def P (x y : ℕ) : ℤ := 2 * (x^2 * y - 3 * y^3 + 2 * x) - 3 * (x + x^2 * y - 2 * y^3) - x

-- Problem statement
theorem equivalent_proof_problem (a b : ℕ) (h : expression_1 a b x y + expression_2 a b x y = expression_2 a b x y + expression_1 a b x y) :
  x = 2 ∧ y = -1 ∧ P 2 (-1) = 4 :=
sorry

end equivalent_proof_problem_l485_485274


namespace find_AX_l485_485236

-- Definitions from part (a)
variables {A B C X : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space X]
variables (AC BC BX AX : ℝ)

-- Given conditions
def conditions (CX_bisects_ACB : Prop) : Prop :=
  CX_bisects_ACB ∧ AC = 25 ∧ BC = 40 ∧ BX = 34

-- Angle Bisector Theorem expression
def angle_bisector_theorem (AC BC BX AX : ℝ) : Prop :=
  AC / AX = BC / BX

-- Proof problem
theorem find_AX (CX_bisects_ACB : Prop) (h : conditions CX_bisects_ACB) : AX = 85 / 4 :=
 by {
 sorry -- Proof would go here
}

end find_AX_l485_485236


namespace sequence_sum_difference_l485_485309

theorem sequence_sum_difference (a b c : ℝ) (n p : ℕ) (hp : p ≠ 1)
  (h : a + (p - 1) * b = a + (p^2 - 1) * c) :
  (∑ k in Finset.range n, (a + k * b)) - (∑ k in Finset.range n, (a + k * c)) = (1/2 : ℝ) * n * p * (n - 1) * c :=
by
  sorry

end sequence_sum_difference_l485_485309


namespace gcd_45_75_l485_485118

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485118


namespace kataleya_total_amount_paid_l485_485958

/-- A store offers a $2 discount for every $10 purchase on any item in the store.
Kataleya went to the store and bought 400 peaches sold at forty cents each.
Prove that the total amount of money she paid at the store for the fruits is $128. -/
theorem kataleya_total_amount_paid : 
  let price_per_peach : ℝ := 0.40
  let number_of_peaches : ℝ := 400 
  let total_cost : ℝ := number_of_peaches * price_per_peach
  let discount_per_10_dollars : ℝ := 2
  let number_of_discounts := total_cost / 10
  let total_discount := number_of_discounts * discount_per_10_dollars
  let amount_paid := total_cost - total_discount
  amount_paid = 128 :=
by
  sorry

end kataleya_total_amount_paid_l485_485958


namespace gcd_45_75_l485_485065

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485065


namespace emily_age_l485_485188

theorem emily_age (A B C D E : ℕ) (h1 : A = B - 4) (h2 : B = C + 5) (h3 : D = C + 2) (h4 : E = A + D - B) (h5 : B = 20) : E = 13 :=
by sorry

end emily_age_l485_485188


namespace sum_first_9_terms_of_sequence_l485_485270

noncomputable def a_n (n : ℕ) : ℤ := 11 - 2 * (n + 1)

def b_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

theorem sum_first_9_terms_of_sequence :
  (∑ n in finset.range 9, b_n n) = -1 / 9 := sorry

end sum_first_9_terms_of_sequence_l485_485270


namespace K1_K2_similar_equilateral_l485_485476

-- Definitions for the problem
noncomputable def K_1 : Type := sorry  -- Define K_1 as a non-right triangle
noncomputable def K_2 : Type := sorry  -- Define K_2 as the orthic triangle of K_1

-- Theorem statement
theorem K1_K2_similar_equilateral (A B C : ℝ) (h1 : K_1 = (Angle A B C)) (h2 : K_2 = orthic_triangle K_1)
  (h3 : similar K_1 K_2) : A = 60 ∧ B = 60 ∧ C = 60 :=
sorry

end K1_K2_similar_equilateral_l485_485476


namespace range_f_2019_l485_485652

noncomputable def f (x : ℝ) : ℝ := Real.log (0.5) (sin x / (sin x + 15))

theorem range_f_2019 (x : ℝ) : ∃ y, y = f^[2019] x ∧ y ∈ Set.Ici 4 := 
sorry

end range_f_2019_l485_485652


namespace carlos_earnings_l485_485351

theorem carlos_earnings (h1 : ∃ w, 18 * w = w * 18) (h2 : ∃ w, 30 * w = w * 30) (h3 : ∀ w, 30 * w - 18 * w = 54) : 
  ∃ w, 18 * w + 30 * w = 216 := 
sorry

end carlos_earnings_l485_485351


namespace even_count_in_top_15_rows_l485_485604

def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

def count_even_in_row (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ k, is_even (binom n k)).card

def count_even_in_top_15_rows : ℕ :=
  (Finset.range 15).sum count_even_in_row

theorem even_count_in_top_15_rows :
  count_even_in_top_15_rows = "Sum of all identified evens" := by
sorry

end even_count_in_top_15_rows_l485_485604


namespace find_b_minus_c_l485_485248

noncomputable def a (n : ℕ) : ℝ :=
  if h : n > 1 then 1 / Real.log 1009 * Real.log n else 0

noncomputable def b : ℝ :=
  a 2 + a 3 + a 4 + a 5 + a 6

noncomputable def c : ℝ :=
  a 15 + a 16 + a 17 + a 18 + a 19

theorem find_b_minus_c : b - c = -Real.logb 1009 1938 := by
  sorry

end find_b_minus_c_l485_485248


namespace angle_XOY_114_l485_485766

-- Define the triangle and angles
variables {X Y Z N M O : Point}
variables (triangle_XYZ : Triangle X Y Z)
variables (altitude_XN : Altitude X N Z)
variables (altitude_YM : Altitude Y M X)
variables (O_point : O ∈ altitude_XN ∩ altitude_YM)
variables (angle_XYZ : angle Y X Z = 52)
variables (angle_YXZ : angle X Y Z = 62)

-- Define the proof goal
theorem angle_XOY_114 :
  ∠X O Y = 114 :=
sorry

end angle_XOY_114_l485_485766


namespace min_value_of_sequence_l485_485443

theorem min_value_of_sequence 
  (a : ℤ) 
  (a_sequence : ℕ → ℤ) 
  (h₀ : a_sequence 0 = a)
  (h_rec : ∀ n, a_sequence (n + 1) = 2 * a_sequence n - n ^ 2)
  (h_pos : ∀ n, a_sequence n > 0) :
  ∃ k, a_sequence k = 3 := 
sorry

end min_value_of_sequence_l485_485443


namespace Ron_spends_15_dollars_l485_485543

theorem Ron_spends_15_dollars (cost_per_bar : ℝ) (sections_per_bar : ℕ) (num_scouts : ℕ) (s'mores_per_scout : ℕ) :
  cost_per_bar = 1.50 ∧ sections_per_bar = 3 ∧ num_scouts = 15 ∧ s'mores_per_scout = 2 →
  cost_per_bar * (num_scouts * s'mores_per_scout / sections_per_bar) = 15 :=
by
  sorry

end Ron_spends_15_dollars_l485_485543


namespace determine_values_l485_485987

variables (x y z v w : ℕ)

-- Given conditions
def right_angle_triangle1 : Prop := x > 0 ∧ y > 0 ∧ x * y = 180 ∧ y^2 = x^2 + 81
def right_angle_triangle2 : Prop := z = Real.sqrt (20^2 - x^2)
def right_angle_triangle3 : Prop := v = 72 / x ∧ w = Real.sqrt (8^2 + v^2)

-- Proving the correct values
theorem determine_values :
  right_angle_triangle1 x y ∧
  right_angle_triangle2 z ∧ 
  right_angle_triangle3 v w →
  x = 12 ∧ y = 15 ∧ z = 16 ∧ v = 6 ∧ w = 10 :=
by 
  intro h,
  sorry

end determine_values_l485_485987


namespace dormouse_stole_flour_l485_485513

-- Define the suspects
inductive Suspect 
| MarchHare 
| MadHatter 
| Dormouse 

open Suspect 

-- Condition 1: Only one of three suspects stole the flour
def only_one_thief (s : Suspect) : Prop := 
  s = MarchHare ∨ s = MadHatter ∨ s = Dormouse

-- Condition 2: Only the person who stole the flour gave a truthful testimony
def truthful (thief : Suspect) (testimony : Suspect → Prop) : Prop :=
  testimony thief

-- Condition 3: The March Hare testified that the Mad Hatter stole the flour
def marchHare_testimony (s : Suspect) : Prop := 
  s = MadHatter

-- The theorem to prove: Dormouse stole the flour
theorem dormouse_stole_flour : 
  ∃ thief : Suspect, only_one_thief thief ∧ 
    (∀ s : Suspect, (s = thief ↔ truthful s marchHare_testimony) → thief = Dormouse) :=
by
  sorry

end dormouse_stole_flour_l485_485513


namespace triangle_vertices_minimum_area_sum_l485_485871

noncomputable def minimum_area_triangle_sum (k : ℤ) : ℤ :=
if (k = 11 ∨ k = 13) then k else 0

theorem triangle_vertices_minimum_area_sum :
  (minimum_area_triangle_sum 11) + (minimum_area_triangle_sum 13) = 24 :=
by {
  simp [minimum_area_triangle_sum],
  sorry
}

end triangle_vertices_minimum_area_sum_l485_485871


namespace min_value_achieved_l485_485679

-- Definition of the conditions
def satisfies_condition (x y : ℝ) : Prop :=
  2 * x + y = 2

-- Mathematical statement to be proved
theorem min_value_achieved (x y : ℝ) (hx : satisfies_condition x y) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (1 / x - y) = 2 * real.sqrt 2 - 2 ↔ x = real.sqrt 2 / 2 :=
sorry

end min_value_achieved_l485_485679


namespace age_ratio_rahul_deepak_l485_485878

/--
Prove that the ratio between Rahul and Deepak's current ages is 4:3 given the following conditions:
1. After 10 years, Rahul's age will be 26 years.
2. Deepak's current age is 12 years.
-/
theorem age_ratio_rahul_deepak (R D : ℕ) (h1 : R + 10 = 26) (h2 : D = 12) : R / D = 4 / 3 :=
by sorry

end age_ratio_rahul_deepak_l485_485878


namespace total_steps_needed_l485_485504

def cycles_needed (dist : ℕ) : ℕ := dist
def steps_per_cycle : ℕ := 5
def effective_steps_per_pattern : ℕ := 1

theorem total_steps_needed (dist : ℕ) (h : dist = 66) : 
  steps_per_cycle * cycles_needed dist = 330 :=
by 
  -- Placeholder for proof
  sorry

end total_steps_needed_l485_485504


namespace even_integers_in_pascals_triangle_top_15_rows_l485_485619

/-- Prove that the total number of even integers in the top 15 rows of Pascal's Triangle is exactly 90.
  Pascal's Triangle's elements, binomial(n,k), are even unless every binary digit of k 
  is present in n when both are expressed in binary. -/
theorem even_integers_in_pascals_triangle_top_15_rows : 
  ∑ n in Finset.range 15, ∑ k in Finset.range (n + 1), if (∀ i, (n.bits.get i) = 1 → (k.bits.get i) = 1) then 0 else 1 = 90 :=
sorry

end even_integers_in_pascals_triangle_top_15_rows_l485_485619


namespace subtract_polynomial_result_l485_485208

-- Define the original polynomial
def A : Polynomial ℚ := x^2 + x - 1

-- Define the given polynomials
def P1 : Polynomial ℚ := x^2 - 3x + 2
def P2 : Polynomial ℚ := 2x^2 - 2x + 1

-- The theorem to prove
theorem subtract_polynomial_result :
  A + P1 = P2 →
  A - P1 = 4 * x - 3 :=
by
  intro h
  rw [←h, add_sub_cancel]
  sorry

end subtract_polynomial_result_l485_485208


namespace minimum_reciprocal_sum_l485_485806

theorem minimum_reciprocal_sum (b : Fin 15 → ℝ) (hpos : ∀ i, 0 < b i) (hsum : (∑ i, b i) = 1) :
  (∑ i, 1 / b i) ≥ 225 :=
sorry

end minimum_reciprocal_sum_l485_485806


namespace gcd_45_75_l485_485108

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485108


namespace quad_area_abcd_l485_485911

noncomputable def circle_eq := λ x y: ℝ, (x - 2)^2 + (y - 2)^2 = 10
def point_E := (0, 1 : ℝ × ℝ)

theorem quad_area_abcd : 
  ∃ (AC BD : ℝ), 
    (AC = 2 * Real.sqrt 10) ∧ 
    (BD = 2 * Real.sqrt 10) ∧ 
    (1 / 2 * (AC * BD) = 10 * Real.sqrt 2) :=
by
  sorry

end quad_area_abcd_l485_485911


namespace union_complement_l485_485303

open Set

variable U : Set ℕ
variable A : Set ℕ
variable B : Set ℕ

theorem union_complement :
  U = {1, 2, 3, 4, 5} →
  A = {3, 4} →
  B = {1, 4, 5} →
  A ∪ (U \ B) = {2, 3, 4} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  sorry

end union_complement_l485_485303


namespace hyperbola_sum_l485_485341

noncomputable def h : ℝ := -3
noncomputable def k : ℝ := 1
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := Real.sqrt 50
noncomputable def b : ℝ := Real.sqrt (c ^ 2 - a ^ 2)

theorem hyperbola_sum :
  h + k + a + b = 2 + Real.sqrt 34 := by
  sorry

end hyperbola_sum_l485_485341


namespace max_quarters_l485_485784

theorem max_quarters (q : ℕ) (h : 0.25 * q + 0.05 * (2 * q) = 4.85) : q ≤ 13 :=
by
  sorry

end max_quarters_l485_485784


namespace age_ratio_is_4_over_3_l485_485875

-- Define variables for ages
variable (R D : ℕ)

-- Conditions
axiom key_condition_R : R + 10 = 26
axiom key_condition_D : D = 12

-- Theorem statement: The ratio of Rahul's age to Deepak's age is 4/3
theorem age_ratio_is_4_over_3 (hR : R + 10 = 26) (hD : D = 12) : R / D = 4 / 3 :=
sorry

end age_ratio_is_4_over_3_l485_485875


namespace single_elimination_tournament_games_23_teams_l485_485961

noncomputable def single_elimination_tournament_games (num_teams : ℕ) : ℕ :=
  num_teams - 1

theorem single_elimination_tournament_games_23_teams :
  single_elimination_tournament_games 23 = 22 :=
by
  -- Proof has been intentionally omitted
  sorry

end single_elimination_tournament_games_23_teams_l485_485961


namespace equation_of_line_l485_485672

def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def lies_on_hyperbola (P : ℝ × ℝ) : Prop :=
  2 * P.1 * P.1 - P.2 * P.2 = 2

theorem equation_of_line {M A B : ℝ × ℝ}
  (hM : M = (4, 1))
  (h_midpoint : midpoint A B M)
  (hA_on_hyperbola : lies_on_hyperbola A)
  (hB_on_hyperbola : lies_on_hyperbola B) :
  ∃ k, (k = 8) ∧ ∀ x : ℝ, y : ℝ, y = 8 * x - 31 :=
begin
  sorry
end

end equation_of_line_l485_485672


namespace a_2012_minus_a_2011_l485_485673

def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ m n : ℕ, a (m + n) = 3 + a m + a n)

theorem a_2012_minus_a_2011 (a : ℕ → ℕ) (h : seq a) : a 2012 - a 2011 = 4 :=
by 
  cases h with h1 h2
  have h3 : a 2012 = 4 + a 2011 := by
    calc
      a 2012 = a (2011 + 1) : by rw [nat.add_comm]
      ... = 3 + a 2011 + a 1 : by rw [h2 2011 1]
      ... = 3 + a 2011 + 1 : by rw [h1]
      ... = 4 + a 2011 : by linarith
  linarith

#eval sorry -- This is a placeholder to ensure the code builds

end a_2012_minus_a_2011_l485_485673


namespace triangle_perimeter_l485_485354

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) (h1 : a = 3) (h2 : b = 3) 
    (h3 : c^2 = a * Real.cos B + b * Real.cos A) : 
    a + b + c = 7 :=
by 
  sorry

end triangle_perimeter_l485_485354


namespace sequence_inequality_l485_485805

theorem sequence_inequality (a : ℕ → ℝ) (h : ∀ (n : ℕ), 0 < n → (a (n - 1) + a (n + 1)) / 2 ≥ a n) :
    ∀ (n : ℕ), 0 < n → (a 0 + a (n + 1)) / 2 ≥ (∑ i in finset.range n, a (i + 1)) / n := 
by
  sorry

end sequence_inequality_l485_485805


namespace compare_magnitudes_l485_485666

theorem compare_magnitudes :
  let a := 0.2^3
  let b := Real.logBase 0.3 0.2
  let c := Real.logBase 3 0.2
  b > a ∧ a > c := by
  sorry

end compare_magnitudes_l485_485666


namespace intersection_S_lies_on_circumcircle_l485_485675

-- Define the necessary points and circles in Lean
variables (A B C D E I U P S : Type) [Point A] [Point B] [Point C] [Point D] [Point E] [Point I] [Point U] [Point P] [Point S]
variable (ABC : Triangle A B C) -- Triangle ABC
variable (circumcircle_ABC : Circle) -- Circumcircle of triangle ABC
variable (circ_incenter_ABC : Circle) -- Incircle of triangle ABC
variable (circ_CDE : Circle) -- Circumcircle of triangle CDE

-- Given conditions
axiom h_AC_neq_BC : AC ≠ BC
axiom h_incenter_I : Incenter I ABC
axiom h_circumcenter_U : Circumcenter U ABC
axiom h_touches_D : Touches circ_incenter_ABC BC D
axiom h_touches_E : Touches circ_incenter_ABC AC E
axiom h_circ_intersection_CP : Intersects circumcircle_ABC circ_CDE C P

-- Define intersection points as variables
def intersect_CU_PI : Point := intersection (line_segment C U) (line_segment P I)

-- The theorem to prove
theorem intersection_S_lies_on_circumcircle (S : Point) 
    (h_S : S = intersect_CU_PI):
    LiesOnCircle circumcircle_ABC S :=
sorry

end intersection_S_lies_on_circumcircle_l485_485675


namespace angle_RPS_is_27_l485_485350

-- Definitions corresponding to the conditions
def QRS_straight (Q R S : Point) : Prop := collinear Q R S
def angle_PQS := 48
def angle_PSQ := 38
def angle_QPR := 67

-- Proof statement
theorem angle_RPS_is_27 (Q R S P : Point) (hQRS : QRS_straight Q R S)
  (h_PQS : angle P Q S = 48) (h_PSQ : angle P S Q = 38) (h_QPR : angle Q P R = 67)
  : angle R P S = 27 := sorry

end angle_RPS_is_27_l485_485350


namespace colors_needed_l485_485742

theorem colors_needed (planets : ℕ) (people : ℕ) (unique_colors_per_person : ℕ) : 
  planets = 8 → people = 3 → unique_colors_per_person = 8 → 
  (people * unique_colors_per_person) = 24 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact eq.refl 24

end colors_needed_l485_485742


namespace gcd_45_75_l485_485058

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485058


namespace gcd_45_75_l485_485109

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485109


namespace unique_three_digit_base_g_l485_485427

theorem unique_three_digit_base_g (g : ℤ) (h : ℤ) (a b c : ℤ) 
  (hg : g > 2) 
  (h_h : h = g + 1 ∨ h = g - 1) 
  (habc_g : a * g^2 + b * g + c = c * h^2 + b * h + a) : 
  a = (g + 1) / 2 ∧ b = (g - 1) / 2 ∧ c = (g - 1) / 2 :=
  sorry

end unique_three_digit_base_g_l485_485427


namespace range_of_eccentricity_l485_485676

open Real

theorem range_of_eccentricity (a b c : ℝ) (α : ℝ) (e : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : a^2 * b^2 > 0)
  (h4 : α ∈ Ioo (π/4) (π/3))
  (h5 : e = c / a)
  (h6 : (2 * c) / (sin α + cos α) = 2 * a)
  : e ∈ Ioo (sqrt 2 / 2) (sqrt 3 - 1) :=
sorry

end range_of_eccentricity_l485_485676


namespace cost_per_pound_of_penne_is_four_l485_485365

-- Define the given conditions as constants
constant cost_mustard_oil_per_liter : ℕ := 13
constant liters_of_mustard_oil : ℕ := 2
constant pounds_of_penne : ℕ := 3
constant cost_pasta_sauce : ℕ := 5
constant amount_left : ℕ := 7
constant initial_amount : ℕ := 50

-- Define the problem of finding the cost per pound of gluten-free penne pasta
theorem cost_per_pound_of_penne_is_four :
  let total_spent := initial_amount - amount_left,
      cost_mustard_oil := liters_of_mustard_oil * cost_mustard_oil_per_liter,
      cost_penne_pasta := total_spent - cost_mustard_oil - cost_pasta_sauce,
      cost_per_pound_penne := cost_penne_pasta / pounds_of_penne in
  cost_per_pound_penne = 4 :=
by
  intros
  sorry

end cost_per_pound_of_penne_is_four_l485_485365


namespace painting_time_l485_485633

theorem painting_time (t : ℚ) : 
  let d_rate := (1 : ℚ) / 5
  let da_rate := (1 : ℚ) / 7
  let e_rate := (1 : ℚ) / 10
  let total_rate := d_rate + da_rate + e_rate
  ∃ t : ℚ, total_rate * (t - 2) = 1 ∧ t = 132 / 31 := 
by
  abbreviate d_rate as d_rate
  abbreviate da_rate as da_rate
  abbreviate e_rate as e_rate
  have total_rate := d_rate + da_rate + e_rate
  use 132 / 31
  split
  · sorry
  · simp


end painting_time_l485_485633


namespace new_device_mean_improved_l485_485151

/-- Indicator data for new device --/
def new_device_data : List ℚ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]

/-- Indicator data for old device --/
def old_device_data : List ℚ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

/-- Calculate mean of a list of rationals --/
def mean (data : List ℚ) : ℚ :=
  data.sum / data.length

/-- Mean values --/
def x_bar : ℚ := mean new_device_data
def y_bar : ℚ := mean old_device_data

/-- Calculate variance of a list of rationals given the mean --/
def variance (data : List ℚ) (μ : ℚ) : ℚ :=
  let n := data.length
  data.sum (λ x => (x - μ) * (x - μ)) / n

/-- Variance values --/
def s1_squared : ℚ := variance new_device_data x_bar
def s2_squared : ℚ := variance old_device_data y_bar

/-- Prove the mean improvement condition --/
noncomputable def mean_has_improved (x_bar y_bar s1_squared s2_squared : ℚ) : Prop :=
  (y_bar - x_bar) ≥ 2 * (Real.sqrt ((s1_squared + s2_squared) / 10))

theorem new_device_mean_improved :
  mean_has_improved x_bar y_bar s1_squared s2_squared := by
  -- the detailed proof steps would be provided here
  sorry

end new_device_mean_improved_l485_485151


namespace even_count_in_pascal_triangle_l485_485610

-- Define the binomial coefficient as a function
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define a predicate to check if a number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Count the number of even integers in the top 15 rows of Pascal's Triangle
def count_even_pascal (rows : ℕ) : ℕ :=
  (Finset.range rows).sum (λ n => (Finset.range (n + 1)).count (λ k => is_even (binom n k)))

-- Statement of the problem
theorem even_count_in_pascal_triangle : count_even_pascal 15 = 84 :=
  by
    sorry

end even_count_in_pascal_triangle_l485_485610


namespace terrier_hush_interval_l485_485942

-- Definitions based on conditions
def poodle_barks_per_terrier_bark : ℕ := 2
def total_poodle_barks : ℕ := 24
def terrier_hushes : ℕ := 6

-- Derived values based on definitions
def total_terrier_barks := total_poodle_barks / poodle_barks_per_terrier_bark
def interval_hush := total_terrier_barks / terrier_hushes

-- The theorem stating the terrier's hush interval
theorem terrier_hush_interval : interval_hush = 2 := by
  have h1 : total_terrier_barks = 12 := by sorry
  have h2 : interval_hush = 2 := by sorry
  exact h2

end terrier_hush_interval_l485_485942


namespace inequality_solution_l485_485007

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end inequality_solution_l485_485007


namespace angle_NHC_60_degrees_l485_485954

theorem angle_NHC_60_degrees 
  (A B C D S N H : Point)
  (square_ABC : is_square A B C D)
  (equilateral_BCD : is_equilateral_triangle B C S)
  (BC_eq : dist B C = dist C S)
  (mid_AS : is_midpoint N A S)
  (mid_CD : is_midpoint H C D) :
  angle N H C = 60 :=
sorry

end angle_NHC_60_degrees_l485_485954


namespace table_filling_impossible_l485_485973

theorem table_filling_impossible :
  ∀ (table : Fin 5 → Fin 8 → Fin 10),
  (∀ digit : Fin 10, ∃ row_set : Finset (Fin 5), row_set.card = 4 ∧
    (∀ row : Fin 5, row ∈ row_set → ∃ col_set : Finset (Fin 8), col_set.card = 4 ∧
      (∀ col : Fin 8, col ∈ col_set → table row col = digit))) →
  False :=
by
  sorry

end table_filling_impossible_l485_485973


namespace find_income_separator_l485_485216

-- Define the income and tax parameters
def income : ℝ := 60000
def total_tax : ℝ := 8000
def rate1 : ℝ := 0.10
def rate2 : ℝ := 0.20

-- Define the function for total tax calculation
def tax (I : ℝ) : ℝ := rate1 * I + rate2 * (income - I)

theorem find_income_separator (I : ℝ) (h: tax I = total_tax) : I = 40000 :=
by sorry

end find_income_separator_l485_485216


namespace parallel_lines_implies_slope_l485_485023

theorem parallel_lines_implies_slope (a : ℝ) :
  (∀ (x y: ℝ), ax + 2 * y = 0) ∧ (∀ (x y: ℝ), x + y = 1) → (a = 2) :=
by
  sorry

end parallel_lines_implies_slope_l485_485023


namespace partition_into_subsets_l485_485426

open Finset Nat

-- Define the properties of the subsets A_i
def property (A : Finset ℕ) (n : ℕ) : Prop :=
  (A.card = n) ∧ (A.sum id = 5043)

-- Define the main theorem: we can partition {1, 2, ... 2007} into subsets A_i
theorem partition_into_subsets :
  ∃ (A : Fin ℕ → Finset ℕ),
    (∀ i : Fin 223, property (A i) 9) ∧
    (range 2007).bUnion A = (range 2007).toFinset ∧
    (∀ i j : Fin 223, i ≠ j → Disjoint (A i) (A j)) :=
by
  sorry

end partition_into_subsets_l485_485426


namespace gcd_45_75_l485_485090

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485090


namespace increase_75_by_150_percent_l485_485500

noncomputable def original_number : Real := 75
noncomputable def percentage_increase : Real := 1.5
noncomputable def increase_amount : Real := original_number * percentage_increase
noncomputable def result : Real := original_number + increase_amount

theorem increase_75_by_150_percent : result = 187.5 := by
  sorry

end increase_75_by_150_percent_l485_485500


namespace six_divisors_third_seven_times_second_fourth_ten_more_than_third_l485_485868

theorem six_divisors_third_seven_times_second_fourth_ten_more_than_third (n : ℕ) :
  (∀ d : ℕ, d ∣ n ↔ d ∈ [1, d2, d3, d4, d5, n]) ∧ 
  (d3 = 7 * d2) ∧ 
  (d4 = d3 + 10) → 
  n = 2891 :=
by
  sorry

end six_divisors_third_seven_times_second_fourth_ten_more_than_third_l485_485868


namespace value_of_f_sum_l485_485264

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h_odd : ∀ x, f (-x) = -f x) : Prop
axiom period_9 (h_period : ∀ x, f (x + 9) = f x) : Prop
axiom f_one (h_f1 : f 1 = 5) : Prop

theorem value_of_f_sum (h_odd : ∀ x, f (-x) = -f x)
                       (h_period : ∀ x, f (x + 9) = f x)
                       (h_f1 : f 1 = 5) :
  f 2007 + f 2008 = 5 :=
sorry

end value_of_f_sum_l485_485264


namespace gcd_8251_6105_l485_485022

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l485_485022


namespace gcd_45_75_l485_485102

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485102


namespace length_of_BC_l485_485753

theorem length_of_BC
  (A B C M D A' B' C' E : Type)
  [AffineSpace A [VecSpace ℝ]] 
  (dist : A → A → ℝ)
  (is_median : A → A → A → Prop)
  (translated_along_median : Π (A B C D : A), is_median A B C → ℝ → Prop)
  (midpoint : A → A → A → Prop)
  (am_median : is_median A B M)
  (a_de : dist A E = 8)
  (ec_de : dist E C = 16)
  (bd_b': dist B' D = 10)
  (M_mid_bc : midpoint M B C)
  (translation_distance : ℝ)
  (A_1_eq_target_A : A' = M)
  (B'_on_line_bd : dist B' B = dist B' D):
  dist B C = 20 := by
sorry

end length_of_BC_l485_485753


namespace value_of_x_l485_485904

theorem value_of_x (x : ℝ) : (2010 + x) ^ 2 = 2 * x ^ 2 ↔ (x = 4850 ∨ x = -830) :=
by
  split
  · { intro h, sorry }
  · { intro h, cases h; rw h; norm_num }

end value_of_x_l485_485904


namespace original_profit_percentage_l485_485176

variable (C : ℝ) -- Cost of the computer

-- Conditions
def selling_price_2240 := 2240
def selling_price_60_percent_profit := 2560
def cost_eqn := selling_price_60_percent_profit = C + 0.60 * C

-- Theorem stating what we want to prove: the original profit percentage is 40%
theorem original_profit_percentage (h : cost_eqn) : (selling_price_2240 - C) / C * 100 = 40 := by
  sorry

end original_profit_percentage_l485_485176


namespace compute_difference_of_squares_l485_485984

theorem compute_difference_of_squares (a b : ℕ) (h₁ : a = 63) (h₂ : b = 57) : a^2 - b^2 = 720 :=
by
  rw [h₁, h₂]
  calc
    (63 : ℕ)^2 - (57 : ℕ)^2 = (63 + 57) * (63 - 57) : by rw Nat.sub_square
    ... = 120 * 6 : by norm_num
    ... = 720 : by norm_num

end compute_difference_of_squares_l485_485984


namespace tangent_length_l485_485379

theorem tangent_length (x y : ℝ) (C₃: (x - 8)^2 + (y - 3)^2 = 49) (C₄: (x + 12)^2 + (y + 4)^2 = 16) :
  (∃ R S : ℝ, shortest_tangent R S C₃ C₄ = (real.sqrt 7840 + real.sqrt 24181) / 11 - 11) :=
sorry

end tangent_length_l485_485379


namespace interval_satisfies_ineq_l485_485020

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end interval_satisfies_ineq_l485_485020


namespace quadrilateral_perimeter_div_a_l485_485474

theorem quadrilateral_perimeter_div_a (a : ℝ) (h : 0 < a) : 
  let s := Set.mk (λ x : ℝ × ℝ, (x.fst = -a ∧ x.snd = -a) ∨ (x.fst = a ∧ x.snd = -a) ∨ (x.fst = -a ∧ x.snd = a) ∨ (x.fst = a ∧ x.snd = a))
  let l := (λ x : ℝ × ℝ, x.snd = x.fst)
  let quadrilateral := Set.mk (λ x : ℝ × ℝ, (x.fst = a ∧ x.snd = a) ∨ (x.fst = a ∧ x.snd = -a) ∨ (x.fst = 0 ∧ x.snd = 0) ∨ (x.fst = -a ∧ x.snd = a))
  let perimeter := dist (a, a) (a, -a) + dist (a, -a) (0, 0) + dist (0, 0) (-a, a) + dist (-a, a) (a, a)
  in perimeter / a = 4 + 2 * Real.sqrt 2 := by
  sorry

end quadrilateral_perimeter_div_a_l485_485474


namespace express_g_over_ln_2_l485_485986

def g (n : ℕ) : ℝ := Real.log (2^(2 * n))

theorem express_g_over_ln_2 (n : ℕ) : (g n) / Real.log 2 = 2 * n := by
  sorry

end express_g_over_ln_2_l485_485986


namespace selling_price_is_100_l485_485938

def cost_price : ℝ := 50
def profit_rate : ℝ := 100
def profit : ℝ := profit_rate / 100 * cost_price
def selling_price : ℝ := cost_price + profit

theorem selling_price_is_100 : selling_price = 100 := by
  -- our conditions
  have h1 : cost_price = 50 := rfl
  have h2 : profit_rate = 100 := rfl
  
  -- noncomputable example for the profit step
  noncomputable example : profit = 50 := by sorry
  
  -- noncomputable example for the selling price step
  noncomputable example : selling_price = 100 := by sorry
  
  -- concluding the theorem proof sketch
  sorry

end selling_price_is_100_l485_485938


namespace limit_ant_path_length_l485_485948

-- Definition of the sequence L_n where each ant walks along n semicircles with radii 1/n
def L (n : ℕ) : ℝ := n * (Real.pi / n)

-- The statement to be proved
theorem limit_ant_path_length : 
  tendsto L atTop (𝓝 Real.pi) :=
sorry

end limit_ant_path_length_l485_485948


namespace inscribed_circle_radius_approx_l485_485041

noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  1 / (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))

theorem inscribed_circle_radius_approx (r : ℝ) : 
  inscribed_circle_radius 5 10 15 ≈ 1.304 :=
by
  sorry

end inscribed_circle_radius_approx_l485_485041


namespace measure_angle_A_l485_485767

theorem measure_angle_A (a b c : ℝ) (A B C : ℝ)
  (h1 : ∀ (Δ : Type), Δ → Δ → Δ)
  (h2 : a / Real.cos A = b / (2 * Real.cos B) ∧ 
        a / Real.cos A = c / (3 * Real.cos C))
  (h3 : A + B + C = Real.pi) : 
  A = Real.pi / 4 :=
sorry

end measure_angle_A_l485_485767


namespace craft_store_pricing_maximize_daily_profit_l485_485507

theorem craft_store_pricing (profit_per_item marked_price cost_price : ℝ)
  (h₁ : profit_per_item = marked_price - cost_price)
  (h₂ : 8 * 0.85 * marked_price + 12 * (marked_price - 35) = 20 * cost_price)
  : cost_price = 155 ∧ marked_price = 200 := 
sorry

theorem maximize_daily_profit (profit_per_item cost_price marked_price : ℝ)
  (h₁ : profit_per_item = marked_price - cost_price)
  (h₃ : ∀ p : ℝ, (100 + 4 * (200 - p)) * (p - cost_price) ≤ 4900)
  : p = 190 ∧ daily_profit = 4900 :=
sorry

end craft_store_pricing_maximize_daily_profit_l485_485507


namespace min_expression_value_l485_485681

theorem min_expression_value (m n : ℝ) (h : m - n^2 = 1) : ∃ min_val : ℝ, min_val = 4 ∧ (∀ x y, x - y^2 = 1 → m^2 + 2 * y^2 + 4 * x - 1 ≥ min_val) :=
by
  sorry

end min_expression_value_l485_485681


namespace small_cheese_pizza_slices_l485_485824

theorem small_cheese_pizza_slices (slices_large_pepperoni : ℕ) 
(slices_eaten_by_both : ℕ) (slices_left_for_both : ℕ) : 
  slices_large_pepperoni = 14 → 
  slices_eaten_by_both = 18 → 
  slices_left_for_both = 4 → 
  ∃ (slices_small_cheese : ℕ), slices_small_cheese = 8 :=
by
  intros h1 h2 h3
  use 8
  sorry

end small_cheese_pizza_slices_l485_485824


namespace exponent_problem_l485_485262

variable {a m n : ℝ}

theorem exponent_problem (h1 : a^m = 2) (h2 : a^n = 3) : a^(3*m + 2*n) = 72 := 
  sorry

end exponent_problem_l485_485262


namespace joan_gave_sam_43_seashells_l485_485368

theorem joan_gave_sam_43_seashells (seashells_initial : ℕ) (seashells_left : ℕ)
  (h_initial: seashells_initial = 70) (h_left: seashells_left = 27) : seashells_initial - seashells_left = 43 :=
by
  rw [h_initial, h_left]
  sorry

end joan_gave_sam_43_seashells_l485_485368


namespace card_value_decrease_l485_485925

-- Define the constants and conditions
def original_value : ℝ := 100
def first_year_decrease_percent : ℝ := 0.40
def second_year_decrease_percent : ℝ := 0.10

def first_year_value : ℝ := original_value * (1 - first_year_decrease_percent)
def second_year_value : ℝ := first_year_value * (1 - second_year_decrease_percent)
def total_decrease : ℝ := original_value - second_year_value
def total_percent_decrease : ℝ := (total_decrease / original_value) * 100

-- Lean statement to prove the total percent decrease is 46%
theorem card_value_decrease : total_percent_decrease = 46 := by
  sorry

end card_value_decrease_l485_485925


namespace triangle_ABT_equilateral_area_ratio_l485_485850

variables {A B C D O T : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited O] [Inhabited T]

-- Conditions of the problem
-- convex quadrilateral ABCD
def is_convex (A B C D O : Type) : Prop := sorry

-- Diagonals AC and BD intersect at O
def diagonals_intersect (A B C D O : Type) : Prop := sorry

-- Triangles BOC and AOD are equilateral
def equilateral_triangles (B O C A D : Type) : Prop := sorry

-- T is symmetric to O with respect to the midpoint of side CD
def symmetric_point (O T C D : Type) : Prop := sorry

-- Proof that triangle ABT is equilateral
theorem triangle_ABT_equilateral (A B C D O T : Type) [H1 : is_convex A B C D O]
  [H2 : diagonals_intersect A B C D O] [H3 : equilateral_triangles B O C A D]
  [H4 : symmetric_point O T C D] :
  ∃ T, sorry :=
sorryproof

-- Given BC = 2 and AD = 4, the ratio of the area of triangle ABT to the area of ABCD is 7/9
theorem area_ratio (A B C D O T : Type) [H1 : is_convex A B C D O]
  [H2 : diagonals_intersect A B C D O] [H3 : equilateral_triangles B O C A D]
  [H4 : symmetric_point O T C D] (BC_eq_2 : BC = 2) (AD_eq_4 : AD = 4) :
  ∃ ratio, ratio = 7/9 :=
sorry

end triangle_ABT_equilateral_area_ratio_l485_485850


namespace smallest_b_for_fraction_eq_l485_485534

theorem smallest_b_for_fraction_eq (a b : ℕ) (h1 : 1000 ≤ a ∧ a < 10000) (h2 : 100000 ≤ b ∧ b < 1000000)
(h3 : 1/2006 = 1/a + 1/b) : b = 120360 := sorry

end smallest_b_for_fraction_eq_l485_485534


namespace approximate_shaded_area_l485_485755

def radius : ℝ := 8
def quarter_circle_area : ℝ := (1 / 4) * π * (radius ^ 2)
def semicircle_area (d : ℝ) : ℝ := (1 / 2) * π * ((d / 2) ^ 2)

def area_shaded_region : ℝ :=
  let ab := radius / (2 ^ (1 / 2))
  let bc := radius / (2 ^ (1 / 2))
  let total_semicircle_area := semicircle_area ab + semicircle_area bc
  2 * (quarter_circle_area - total_semicircle_area / 2)

theorem approximate_shaded_area : |area_shaded_region - 18.3| < 0.1 := by
  sorry

end approximate_shaded_area_l485_485755


namespace find_p_q_l485_485661

def op (a b c d : ℝ) : ℝ × ℝ := (a * c - b * d, a * d + b * c)

theorem find_p_q :
  (∀ (a b c d : ℝ), (a = c ∧ b = d) ↔ (a, b) = (c, d)) →
  (op 1 2 p q = (5, 0)) →
  (p, q) = (1, -2) :=
by
  intro h
  intro eq_op
  sorry

end find_p_q_l485_485661


namespace kenya_peanuts_l485_485781

def jose_peanuts : ℕ := 85
def difference : ℕ := 48

theorem kenya_peanuts : jose_peanuts + difference = 133 := by
  sorry

end kenya_peanuts_l485_485781


namespace original_circle_area_l485_485199

theorem original_circle_area (A : ℝ) (h1 : ∃ sector_area : ℝ, sector_area = 5) (h2 : A / 64 = 5) : A = 320 := 
by sorry

end original_circle_area_l485_485199


namespace gcd_45_75_l485_485113

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485113


namespace gcd_45_75_l485_485100

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485100


namespace problem_proof_l485_485319

def P : Set ℝ := {x | x ≤ 3}

theorem problem_proof : {-1} ⊆ P := 
sorry

end problem_proof_l485_485319


namespace total_students_l485_485509

theorem total_students (students_in_front : ℕ) (position_from_back : ℕ) : 
  students_in_front = 6 ∧ position_from_back = 5 → 
  students_in_front + 1 + (position_from_back - 1) = 11 :=
by
  sorry

end total_students_l485_485509


namespace age_ratio_is_4_over_3_l485_485876

-- Define variables for ages
variable (R D : ℕ)

-- Conditions
axiom key_condition_R : R + 10 = 26
axiom key_condition_D : D = 12

-- Theorem statement: The ratio of Rahul's age to Deepak's age is 4/3
theorem age_ratio_is_4_over_3 (hR : R + 10 = 26) (hD : D = 12) : R / D = 4 / 3 :=
sorry

end age_ratio_is_4_over_3_l485_485876


namespace number_of_people_in_group_l485_485448

-- Define the conditions as Lean definitions
def avg_weight_increase : ℝ := 2.5
def replaced_person_weight : ℝ := 66
def new_person_weight : ℝ := 86

-- Define the number of people in the group
variable (n : ℕ)

-- The main theorem statement
theorem number_of_people_in_group : n = 8 :=
by
  -- Using the conditions and the given correct answer
  have h1 : new_person_weight - replaced_person_weight = avg_weight_increase * n :=
    by simp [new_person_weight, replaced_person_weight, avg_weight_increase]
  have h2: 20 = 2.5 * 8 :=
    by simp [h1, n]
  simp [h2]

-- Proof is omitted for this task.
sorry

end number_of_people_in_group_l485_485448


namespace gcd_45_75_l485_485067

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485067


namespace find_fraction_divide_equal_l485_485631

theorem find_fraction_divide_equal (x : ℚ) : 
  (3 * x = (1 / (5 / 2))) → (x = 2 / 15) :=
by
  intro h
  sorry

end find_fraction_divide_equal_l485_485631


namespace fraction_to_decimal_l485_485640

theorem fraction_to_decimal :
  (7 : ℝ) / (16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l485_485640


namespace determine_m_from_probability_l485_485561

theorem determine_m_from_probability :
  ∃ (m : ℝ), (∀ (x : ℝ), x ∈ set.Icc (-2) 4 → ∃ (P : ℝ), P = (set.Icc (-m) m).measure / (set.Icc (-2) 4).measure) ∧ P = 5/6 → m = 3 :=
sorry

end determine_m_from_probability_l485_485561


namespace cos_beta_half_l485_485276

theorem cos_beta_half (α β : ℝ) 
  (h1 : cos α = 1 / 7) 
  (h2 : cos (α - β) = 13 / 14) 
  (h3 : 0 < β) 
  (h4 : β < α) 
  (h5 : α < π / 2) 
  : cos β = 1 / 2 := 
sorry

end cos_beta_half_l485_485276


namespace juicy_pair_count_l485_485578

def is_odd_prime (p : ℕ) := p > 1 ∧ (∀ m : ℕ, m ∣ p → m = 1 ∨ m = p) ∧ ¬(even p)

def is_juicy_pair (n p : ℕ) : Prop :=
  n^2 ≡ 1 [MOD p^2] ∧ n ≡ -1 [MOD p]

theorem juicy_pair_count :
  ∃! k, k = 36 ∧
    k = Nat.card {np : ℕ × ℕ // np.1 ≤ 200 ∧ np.2 ≤ 200 ∧ is_odd_prime np.2 ∧ is_juicy_pair np.1 np.2} := sorry

end juicy_pair_count_l485_485578


namespace sum_of_inradii_l485_485738

theorem sum_of_inradii {A B C E : Type*}
  (hABC : triangle A B C) (hAB : distance A B = 7) (hAC : distance A C = 9)
  (hBC : distance B C = 12) (hBE : distance B E = 5) (hEC : distance E C = 7) 
  (hAEP : AE_perpendicular_BC : is_perpendicular (line_through A E) (line_through B C)) :
  let rABE := inradius (triangle A B E),
      rAEC := inradius (triangle A E C)
  in rABE + rAEC = 84 * sqrt 20 / (48 + 14 * sqrt 20) :=
  sorry

end sum_of_inradii_l485_485738


namespace num_distinct_circles_l485_485798

-- Define that S is a square
variable (S : Type) [square S]

-- State the theorem
theorem num_distinct_circles (vertices : set S) (diameters : set (S × S)) :
  (count_distinct_circles vertices diameters = 2) :=
sorry

end num_distinct_circles_l485_485798


namespace problem_statement_l485_485142

theorem problem_statement
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_le_sqrt2_a : a ≤ Real.sqrt 2)
  (h_le_sqrt2_b : b ≤ Real.sqrt 2)
  (h_le_sqrt2_c : c ≤ Real.sqrt 2)
  (h_abc : a * b * c = 2) :
  Real.sqrt 2 * ( (ab + 3 * c) / (3 * ab + c) + (bc + 3 * a) / (3 * bc + a) + (ca + 3 * b) / (3 * ca + b) ) ≥ a + b + c :=
begin
  sorry
end

end problem_statement_l485_485142


namespace increased_percentage_l485_485495

theorem increased_percentage (x : ℝ) (p : ℝ) (h : x = 75) (h₁ : p = 1.5) : x + (p * x) = 187.5 :=
by
  sorry

end increased_percentage_l485_485495


namespace exists_infinite_pairs_consecutive_numbers_sum_of_digits_divisible_by_13_l485_485632

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

def consecutive_numbers_sum_of_digits_divisible_by_13 : Prop :=
  ∃ (n : ℕ), (sum_of_digits n % 13 = 0) ∧ (sum_of_digits (n + 1) % 13 = 0)

theorem exists_infinite_pairs_consecutive_numbers_sum_of_digits_divisible_by_13 : 
  ∃ᶠ (n : ℕ), (sum_of_digits n % 13 = 0) ∧ (sum_of_digits (n + 1) % 13 = 0) :=
by
  sorry

end exists_infinite_pairs_consecutive_numbers_sum_of_digits_divisible_by_13_l485_485632


namespace green_pill_cost_l485_485185

theorem green_pill_cost (p g : ℕ) (h1 : g = p + 1) (h2 : 14 * (p + g) = 546) : g = 20 :=
by
  sorry

end green_pill_cost_l485_485185


namespace min_val_and_period_sinx_cosx_l485_485024

theorem min_val_and_period_sinx_cosx :
  (∀ x, (sin x + cos x) ≥ -√2 ∧ (∃ k: ℤ, x = k * (2 * π))) :=
by
  -- Here, you'd provide the proof which shows the given conditions and conclusion
  sorry

end min_val_and_period_sinx_cosx_l485_485024


namespace α_plus_β_value_l485_485279

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

axiom obtuse_α : π / 2 < α ∧ α < π
axiom obtuse_β : π / 2 < β ∧ β < π

axiom sin_α : Real.sin α = sqrt 5 / 5
axiom cos_β : Real.cos β = -3 * sqrt 10 / 10

theorem α_plus_β_value : α + β = 7 * π / 4 :=
by
  -- Proof will go here
  sorry

end α_plus_β_value_l485_485279


namespace conjugate_of_z_l485_485450

noncomputable def z (z : ℂ) : Prop := (z - 3) * (2 - I) = 5

theorem conjugate_of_z (z : ℂ) (h : z (z)) : conj z = 5 - I :=
sorry

end conjugate_of_z_l485_485450


namespace trapezoid_area_l485_485761

-- Define the conditions of the problem
variables (AB DC : ℝ) (area_AMD area_BCM : ℝ)
variables (AB_is_8 : AB = 8) (DC_is_10 : DC = 10) 
variables (area_AMD_is_10 : area_AMD = 10) (area_BCM_is_15 : area_BCM = 15)

-- Statement of the problem to prove
theorem trapezoid_area (AB DC area_AMD area_BCM : ℝ) 
  (AB_is_8 : AB = 8) (DC_is_10 : DC = 10)
  (area_AMD_is_10 : area_AMD = 10) (area_BCM_is_15 : area_BCM = 15) :
  let area_ABM := (area_AMD + area_BCM) * (AB / DC) 
  in area_AMD + area_BCM + area_ABM = 45 := by 
    sorry

end trapezoid_area_l485_485761


namespace inequality_correct_l485_485718

theorem inequality_correct {a b : ℝ} (h₁ : a < 0) (h₂ : -1 < b) (h₃ : b < 0) : a < a * b ^ 2 ∧ a * b ^ 2 < a * b := 
sorry

end inequality_correct_l485_485718


namespace angle_C_is_pi_over_3_and_area_of_triangle_ABC_l485_485329

theorem angle_C_is_pi_over_3_and_area_of_triangle_ABC {A B C : ℝ} 
    (h1 : 2 * sin (2 * C) * cos C - sin (3 * C) = sqrt 3 * (1 - cos C))
    (h2 : AB = 2)
    (h3 : sin C + sin (B - A) = 2 * sin (2 * A))
    (h4 : A + B + C = π) : 
    C = π / 3 ∧ (0.5 * AB * (AC:ℝ) * sin C = (2 * sqrt 3)/3) := sorry

end angle_C_is_pi_over_3_and_area_of_triangle_ABC_l485_485329


namespace possible_digits_count_l485_485551

lemma digits_361_sum (d : ℕ) (h : d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ((3 + 6 + 1) + d) % 3 = 0 ↔ d = 2 ∨ d = 5 ∨ d = 8 := by sorry

theorem possible_digits_count :
  ∃ S : finset ℕ, S = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (∀ d ∈ S, digits_361_sum d (finset.mem_of_mem_coe h) → d = 2 ∨ d = 5 ∨ d = 8) ∧
  S.filter (λ d, d = 2 ∨ d = 5 ∨ d = 8).card = 3 := by sorry

end possible_digits_count_l485_485551


namespace revenue_increase_12_percent_l485_485819

-- Declaring variables for original price, quantity and calculating the revenue
variables (P Q : ℝ) 

-- Calculations for the new price and new quantity sold based on the given conditions
def P_new := 1.40 * P
def Q_new := 0.80 * Q

-- Definitions for original revenue and new revenue
def R := P * Q
def R_new := P_new * Q_new

-- Stating the effect on the revenue as a Lean theorem
theorem revenue_increase_12_percent :
  R_new = 1.12 * R :=
by
  sorry -- Proof goes here

end revenue_increase_12_percent_l485_485819


namespace Quadratic_Equation_is_D_l485_485126

-- Definitions to encapsulate conditions
def eqA (x : ℝ) : Prop := (x - 3) * x = x^2 + 2
def eqB (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def eqC (x : ℝ) : Prop := 3 * x^2 - 1 / x + 2 = 0
def eqD (x : ℝ) : Prop := 2 * x^2 = 1

/-- Definition of a quadratic equation in one variable -/
def isQuadraticEquation (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), ∀ x, eq x ↔ a * x^2 + b * x + c = 0 ∧ a ≠ 0

-- Theorem statement
theorem Quadratic_Equation_is_D : 
  isQuadraticEquation eqA = false ∧ 
  isQuadraticEquation eqB = (∃ a b c, a ≠ 0) ∧ 
  isQuadraticEquation eqC = false ∧ 
  isQuadraticEquation eqD :=
by 
  sorry

end Quadratic_Equation_is_D_l485_485126


namespace acute_triangle_sine_inequality_l485_485660

theorem acute_triangle_sine_inequality
  (α β γ : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hγ : 0 < γ ∧ γ < π / 2)
  (h_sum : α + β + γ = π) :
  sin (2 * α) + sin (2 * β) + sin (2 * γ) ≤ sin (α + β) + sin (β + γ) + sin (γ + α) :=
sorry

end acute_triangle_sine_inequality_l485_485660


namespace original_speed_is_150_l485_485771

noncomputable def original_speed : ℝ :=
let final_speed : ℝ := 205
let additional_speed_due_to_weight_cut : ℝ := 10
let speed_increase_rate_due_to_supercharge : ℝ := 1.30
in (final_speed - additional_speed_due_to_weight_cut) / speed_increase_rate_due_to_supercharge

theorem original_speed_is_150 :
  original_speed = 150 :=
by
  unfold original_speed
  norm_num
  sorry

end original_speed_is_150_l485_485771


namespace equal_product_sequence_sum_l485_485218

def isEqualProductSequence (a : ℕ → ℝ) (p : ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = p

def sumOfFirstNTerms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, a i)

theorem equal_product_sequence_sum :
  ∃ (a : ℕ → ℝ), isEqualProductSequence a 10 → a 0 = 2 → sumOfFirstNTerms a 21 = 72 :=
begin
  sorry
end

end equal_product_sequence_sum_l485_485218


namespace prod_fraction_lt_two_l485_485530

theorem prod_fraction_lt_two (n : ℕ) (h : n > 0) :
  (∏ i in Finset.range n, (1 + (1 / (3 : ℝ)^((i + 1) : ℝ)))) < 2 :=
sorry

end prod_fraction_lt_two_l485_485530


namespace xiao_qiang_games_l485_485657

-- Definitions reflecting the given conditions
def five_students : Type := {A, B, C, D, XiaoQiang : ℕ}

-- Conditions given in the problem
def A_games : ℕ := 4
def B_games : ℕ := 3
def C_games : ℕ := 2
def D_games : ℕ := 1

-- Proof statement that Xiao Qiang has played 2 games
theorem xiao_qiang_games : ∃ games : ℕ, games = 2 ∧ 
  -- Each player competes against every other player exactly once
  (A_games = 4) ∧
  (B_games = 3) ∧
  (C_games = 2) ∧
  (D_games = 1) :=
sorry

end xiao_qiang_games_l485_485657


namespace angle_BAO_eq_angle_CAH_l485_485390

noncomputable def is_triangle (A B C : Type) : Prop := sorry
noncomputable def orthocenter (A B C H : Type) : Prop := sorry
noncomputable def circumcenter (A B C O : Type) : Prop := sorry
noncomputable def angle (A B C : Type) : Type := sorry

theorem angle_BAO_eq_angle_CAH (A B C H O : Type) 
  (hABC : is_triangle A B C)
  (hH : orthocenter A B C H)
  (hO : circumcenter A B C O):
  angle B A O = angle C A H := 
  sorry

end angle_BAO_eq_angle_CAH_l485_485390


namespace inequality_solution_l485_485009

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end inequality_solution_l485_485009


namespace gcd_of_45_and_75_l485_485083

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l485_485083


namespace doll_cost_l485_485044

theorem doll_cost (D : ℝ) (h : 4 * D = 60) : D = 15 :=
by {
  sorry
}

end doll_cost_l485_485044


namespace black_white_ratio_l485_485655

noncomputable def circle_area (r : ℝ) : ℝ := real.pi * r^2

def black_areas : List ℝ :=
  [circle_area 3,
   circle_area 7 - circle_area 5,
   circle_area 11 - circle_area 9]

def white_areas : List ℝ :=
  [circle_area 5 - circle_area 3,
   circle_area 9 - circle_area 7]

noncomputable def total_black_area : ℝ := black_areas.sum
noncomputable def total_white_area : ℝ := white_areas.sum

theorem black_white_ratio :
  total_black_area / total_white_area = 73 / 48 :=
by sorry

end black_white_ratio_l485_485655


namespace kenya_peanuts_eq_133_l485_485778

def num_peanuts_jose : Nat := 85
def additional_peanuts_kenya : Nat := 48

def peanuts_kenya (jose_peanuts : Nat) (additional_peanuts : Nat) : Nat :=
  jose_peanuts + additional_peanuts

theorem kenya_peanuts_eq_133 : peanuts_kenya num_peanuts_jose additional_peanuts_kenya = 133 := by
  sorry

end kenya_peanuts_eq_133_l485_485778


namespace calc_f_xh_min_f_x_l485_485323

def f (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 1

theorem calc_f_xh_min_f_x (x h : ℝ) : f (x + h) - f x = h * (10 * x + 5 * h - 2) := 
by
  sorry

end calc_f_xh_min_f_x_l485_485323


namespace conditional_expectation_inequality_l485_485391

open ProbabilityTheory

variables {Ω : Type*} [MeasureSpace Ω]
variable (X Y Z : Ω → ℝ)
variable [Measurable X] [Measurable Y] [Measurable Z]
variable (hXY_ind : IndepFun X Y) (h2X_fin : E[X^2] < ∞) (h2Y_fin : E[Y^2] < ∞)
variable (h2Z_fin : E[Z^2] < ∞) (hZ_zero_mean : E[Z] = 0)

theorem conditional_expectation_inequality
  (h_assumptions : IndepFun X Y ∧ E[X^2] < ∞ ∧ E[Y^2] < ∞ ∧ E[Z^2] < ∞ ∧ E[Z] = 0) :
  E[(∥ conditionalExpectations Z X ∥^2] + E[(∥ conditionalExpectations Z Y ∥^2] ≤ E[Z^2] := 
sorry

end conditional_expectation_inequality_l485_485391


namespace circumscribed_center_on_Ox_axis_l485_485456

-- Define the quadratic equation
noncomputable def quadratic_eq (p x : ℝ) : ℝ := 2^p * x^2 + 5 * p * x - 2^(p^2)

-- Define the conditions for the problem
def intersects_Ox (p : ℝ) : Prop := ∃ x1 x2 : ℝ, quadratic_eq p x1 = 0 ∧ quadratic_eq p x2 = 0 ∧ x1 ≠ x2

def intersects_Oy (p : ℝ) : Prop := quadratic_eq p 0 = -2^(p^2)

-- Define the problem statement
theorem circumscribed_center_on_Ox_axis :
  (∀ p : ℝ, intersects_Ox p ∧ intersects_Oy p → (p = 0 ∨ p = -1)) →
  (0 + (-1) = -1) :=
sorry

end circumscribed_center_on_Ox_axis_l485_485456


namespace pascals_triangle_even_count_15_l485_485616

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

def is_even (x : ℕ) : Prop := x % 2 = 0

def count_even_in_row (n : ℕ) : ℕ :=
  (finset.range (n + 1)).count (λ k, is_even (pascal n k))

def count_even_in_first_15_rows : ℕ :=
  (finset.range 15).sum count_even_in_row

theorem pascals_triangle_even_count_15 :
  count_even_in_first_15_rows = 64 :=
by sorry

end pascals_triangle_even_count_15_l485_485616


namespace range_of_g_on_interval_l485_485292

def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x)

theorem range_of_g_on_interval :
  ∀ x ∈ Icc (Real.pi / 6) (2 * Real.pi / 3), g x ∈ Icc (-2 : ℝ) 1 := by
  sorry

end range_of_g_on_interval_l485_485292


namespace total_clothing_l485_485983

def num_boxes : ℕ := 4
def scarves_per_box : ℕ := 2
def mittens_per_box : ℕ := 6

theorem total_clothing :
  num_boxes * scarves_per_box + num_boxes * mittens_per_box = 32 :=
by
  sorry

end total_clothing_l485_485983


namespace integral_equals_area_of_semicircle_l485_485638

noncomputable def integral_semicircle_area : ℝ :=
  ∫ x in -3..3, Real.sqrt (9 - x^2)

theorem integral_equals_area_of_semicircle :
  integral_semicircle_area = (9 * Real.pi) / 2 :=
by
  sorry

end integral_equals_area_of_semicircle_l485_485638


namespace terry_daily_driving_time_l485_485445

theorem terry_daily_driving_time 
  (d1: ℝ) (s1: ℝ)
  (d2: ℝ) (s2: ℝ)
  (d3: ℝ) (s3: ℝ)
  (h1 : d1 = 15) (h2 : s1 = 30)
  (h3 : d2 = 35) (h4 : s2 = 50)
  (h5 : d3 = 10) (h6 : s3 = 40) : 
  2 * ((d1 / s1) + (d2 / s2) + (d3 / s3)) = 2.9 := 
by
  sorry

end terry_daily_driving_time_l485_485445


namespace solve_linear_system_l485_485441

theorem solve_linear_system :
  ∃ (x y z : ℤ), 3 * x + 2 * y + 2 * z = 13 ∧ 
                2 * x + 3 * y + 2 * z = 14 ∧ 
                2 * x + 2 * y + 3 * z = 15 ∧ 
                x = 1 ∧ y = 2 ∧ z = 3 := by 
  use 1, 2, 3
  apply And.intro
  · exact calc
    3 * 1 + 2 * 2 + 2 * 3 = 3 + 4 + 6 : by norm_num
    ... = 13 : by norm_num
  ;
  apply And.intro
  · exact calc
    2 * 1 + 3 * 2 + 2 * 3 = 2 + 6 + 6 : by norm_num
    ... = 14 : by norm_num
  ;
  apply And.intro
  · exact calc
    2 * 1 + 2 * 2 + 3 * 3 = 2 + 4 + 9 : by norm_num
    ... = 15 : by norm_num
  ;
  exact And.intro rfl (And.intro rfl rfl)

end solve_linear_system_l485_485441


namespace ron_chocolate_bar_cost_l485_485537

-- Definitions of the conditions given in the problem
def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def s'mores_needed_per_scout : ℕ := 2
def total_s'mores_needed : ℕ := scouts * s'mores_needed_per_scout
def chocolate_bars_needed : ℕ := total_s'mores_needed / sections_per_chocolate_bar
def total_cost_of_chocolate_bars : ℝ := chocolate_bars_needed * cost_per_chocolate_bar

-- Proving the question equals the answer given conditions
theorem ron_chocolate_bar_cost : total_cost_of_chocolate_bars = 15.00 := by
  sorry

end ron_chocolate_bar_cost_l485_485537


namespace inner_square_area_l485_485754

theorem inner_square_area (ABCD : Type) [square ABCD] (side_length_ABCD : ℝ)
  (EFGH : Type) [square EFGH] (E : point ABCD) (B : point ABCD) (H : point EFGH) 
  (side_length_EFGH : ℝ) (distance_BE : ℝ) :
  side_length_ABCD = 10 ∧ distance_BE = 2 ∧ E.on_side AB ∧ H.on_side EF →
  side_length_EFGH^2 = 96 :=
sorry

end inner_square_area_l485_485754


namespace categorize_number_players_l485_485562

theorem categorize_number_players :
  -- Define the sets of categorized numbers
  let int_team := {0, -8}
  let frac_team := {1/7, 0.505}
  let irr_team := {Real.sqrt 13, Real.pi}
  -- The given numbers
  let numbers := {0, -8, Real.sqrt 13, 1/7, Real.pi, 0.505}
  -- Conditions for categorization
  (∀ x ∈ numbers, x ∈ int_team ∨ x ∈ frac_team ∨ x ∈ irr_team) ∧
  (∀ x ∈ int_team, x ∈ numbers ∧ Int x = x) ∧ 
  (∀ x ∈ frac_team, x ∈ numbers ∧ (∃ n d : ℕ, d ≠ 0 ∧ x = n/d)) ∧
  (∀ x ∈ irr_team, x ∈ numbers ∧ ¬ ∃ n d : ℕ, d ≠ 0 ∧ x = n/d) :=
by
  sorry

end categorize_number_players_l485_485562


namespace general_eq_C1_cartesian_eq_C2_min_MN_l485_485750

open Real

section CurveEquations

-- Parametric equations for C1
def C1_param (t : ℝ) : ℝ × ℝ := (-3 + t, -1 - t)

-- General equation of C1: x + y + 4 = 0
def C1_general (x y : ℝ) : Prop := x + y + 4 = 0

-- Verify C1 equation from parametric form
theorem general_eq_C1 : ∀ t, C1_general (fst (C1_param t)) (snd (C1_param t)) :=
by
  intros t
  simp [C1_param, C1_general]
  linarith

-- Polar equation for C2
def C2_polar (θ : ℝ) : ℝ := 4 * sqrt 2 * sin (3 * π / 4 - θ)

-- Cartesian equation of C2: (x - 2)² + (y - 2)² = 8
def C2_cartesian (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8

-- Verify C2 equation from polar form
theorem cartesian_eq_C2 : ∀ θ, ∃ (x y : ℝ), C2_cartesian x y ∧ x = 4* sqrt 2 * sin (3*π/4 - θ) * cos θ + 2 ∧ y = 4 * sqrt 2 * sin (3*π/4 - θ) * sin θ + 2 :=
by
  intros θ
  use [4 * cos θ + 4 * sin θ, 4 * cos θ + 4 * sin θ]
  simp only [C2_cartesian]
  sorry  -- detailed calculation is omitted

-- Minimum value of |MN|
def distance (p1 p2 : ℝ × ℝ) : ℝ := sqrt ((fst p1 - fst p2)^2 + (snd p1 - snd p2)^2)
def dist_center_to_line : ℝ := 4 * sqrt 2

-- Verify minimum value of |MN| is 2 sqrt 6
theorem min_MN : ∀ t, let M := C1_param t in abs (distance M (2, 2) - sqrt 8) = 2 * sqrt 6 :=
by
  intros t
  simp [distance, dist_center_to_line, C1_param]
  sorry  -- detailed calculation is omitted

end CurveEquations

end general_eq_C1_cartesian_eq_C2_min_MN_l485_485750


namespace find_cos_Z_l485_485356

-- Define sin X and cos Y
variables (X Y Z : ℝ)
def sin_X : ℝ := 4 / 5
def cos_Y : ℝ := 12 / 13

-- Define triangle relationship
noncomputable def cos_Z (X Y : ℝ) := - (cos X * cos Y - sin X * sin Y)

-- State the theorem that cos Z = -16/65 under given conditions
theorem find_cos_Z : 
  sin X = 4 / 5 → 
  cos Y = 12 / 13 → 
  cos Z = - (16 / 65) := 
by {
  sorry
}

end find_cos_Z_l485_485356


namespace sin_angle_GAC_l485_485143

-- Define the points
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

def A := Point3D.mk 0 0 0
def B := Point3D.mk 2 0 0
def C := Point3D.mk 2 3 0
def G := Point3D.mk 2 3 4

-- Function to calculate the distance between two points
def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2 + (p2.z - p1.z) ^ 2)

-- Calculate the distance AG
def AG := distance A G

-- Define the triangle side lengths
def opposite : ℝ := 4
def hypotenuse : ℝ := AG

-- Define the function to calculate the sine of the angle ∠GAC
def sin_GAC : ℝ := opposite / hypotenuse

-- Proof statement
theorem sin_angle_GAC :
  sin_GAC = 4 / Real.sqrt 29 := by
  sorry

end sin_angle_GAC_l485_485143


namespace both_dice_3_given_one_dice_3_l485_485715

noncomputable def both_dice_3_probability_given_one_is_3 : ℚ := 1 / 11

theorem both_dice_3_given_one_dice_3 :
  let S := ({1, 2, 3, 4, 5, 6} : set ℕ × {1, 2, 3, 4, 5, 6}) in
  let outcomes := { (d1, d2) ∈ S | d1 = 3 ∨ d2 = 3 } in
  let favorable := { (3, 3) } in
  outcomes ≠ ∅ → 
  (favorable.finite.to_finset.card : ℚ) / (outcomes.finite.to_finset.card : ℚ) = both_dice_3_probability_given_one_is_3 :=
by
  sorry

end both_dice_3_given_one_dice_3_l485_485715


namespace perpendicular_vectors_l485_485919

variable {x1 y1 x2 y2 : ℝ}

/-- Two vectors are perpendicular if their dot product is zero. -/
theorem perpendicular_vectors (h : x1 * x2 + y1 * y2 = 0) : 
  (x1, y1) ⊥ (x2, y2) :=
sorry

/-- The distance between two points in the plane. -/
def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The cosine of the angle between two vectors. -/
def cos_angle (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 * x2 + y1 * y2) / (Real.sqrt (x1^2 + y1^2) * Real.sqrt (x2^2 + y2^2))

end perpendicular_vectors_l485_485919


namespace convex_hexagon_possibilities_l485_485317

noncomputable def hexagon_side_lengths : List ℕ := [1, 2, 3, 4, 5, 6]

theorem convex_hexagon_possibilities : 
  ∃ (hexagons : List (List ℕ)), 
    (∀ h ∈ hexagons, 
      (h.length = 6) ∧ 
      (∀ a ∈ h, a ∈ hexagon_side_lengths)) ∧ 
      (hexagons.length = 3) := 
sorry

end convex_hexagon_possibilities_l485_485317


namespace sector_central_angle_l485_485281

theorem sector_central_angle (l r α : ℝ) (h1 : l = 6) (h2 : 0 < r) (h3 : 18 = 1 / 2 * l * r) : α = l / r :=
by
  -- Using the given conditions to establish the relationship
  have hr : r = 6 := by
    -- Simplify using the condition h3
    linarith

  -- Substitute r = 6 into α = l / r
  rw hr
  linarith

end sector_central_angle_l485_485281


namespace max_possible_median_l485_485838

/-- 
Given:
1. The Beverage Barn sold 300 cans of soda to 120 customers.
2. Every customer bought at least 1 can of soda but no more than 5 cans.
Prove that the maximum possible median number of cans of soda bought per customer is 5.
-/
theorem max_possible_median (total_cans : ℕ) (customers : ℕ) (min_can_per_customer : ℕ) (max_can_per_customer : ℕ) :
  total_cans = 300 ∧ customers = 120 ∧ min_can_per_customer = 1 ∧ max_can_per_customer = 5 →
  (∃ median : ℕ, median = 5) :=
by
  sorry

end max_possible_median_l485_485838


namespace coeff_x5_expansion_l485_485756

theorem coeff_x5_expansion : 
  let f := λ x : ℚ, (x^2 + (2/x) + 1) 
  in  polynomial.coeff (polynomial.expand ℚ (f 1)^7) 5 = 560 := sorry

end coeff_x5_expansion_l485_485756


namespace total_votes_l485_485343

theorem total_votes (V : ℝ) (win_percentage : ℝ) (majority : ℝ) (lose_percentage : ℝ)
  (h1 : win_percentage = 0.75) (h2 : lose_percentage = 0.25) (h3 : majority = 420) :
  V = 840 :=
by
  sorry

end total_votes_l485_485343


namespace tan_of_alpha_l485_485686

theorem tan_of_alpha
  (α : ℝ)
  (h1 : Real.sin (α + Real.pi / 2) = 1 / 3)
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan α = 2 * Real.sqrt 2 := 
sorry

end tan_of_alpha_l485_485686


namespace find_cos_Z_l485_485355

-- Define sin X and cos Y
variables (X Y Z : ℝ)
def sin_X : ℝ := 4 / 5
def cos_Y : ℝ := 12 / 13

-- Define triangle relationship
noncomputable def cos_Z (X Y : ℝ) := - (cos X * cos Y - sin X * sin Y)

-- State the theorem that cos Z = -16/65 under given conditions
theorem find_cos_Z : 
  sin X = 4 / 5 → 
  cos Y = 12 / 13 → 
  cos Z = - (16 / 65) := 
by {
  sorry
}

end find_cos_Z_l485_485355


namespace bike_travel_distance_l485_485927

def avg_speed : ℝ := 3  -- average speed in m/s
def time : ℝ := 7       -- time in seconds

theorem bike_travel_distance : avg_speed * time = 21 := by
  sorry

end bike_travel_distance_l485_485927


namespace Sam_has_most_pages_l485_485814

theorem Sam_has_most_pages :
  let pages_per_inch_miles := 5
  let inches_miles := 240
  let pages_per_inch_daphne := 50
  let inches_daphne := 25
  let pages_per_inch_sam := 30
  let inches_sam := 60

  let pages_miles := inches_miles * pages_per_inch_miles
  let pages_daphne := inches_daphne * pages_per_inch_daphne
  let pages_sam := inches_sam * pages_per_inch_sam
  pages_sam = 1800 ∧ pages_sam > pages_miles ∧ pages_sam > pages_daphne :=
by
  sorry

end Sam_has_most_pages_l485_485814


namespace child_sold_apples_correct_l485_485933

-- Definitions based on conditions
def initial_apples (children : ℕ) (apples_per_child : ℕ) : ℕ := children * apples_per_child
def eaten_apples (children_eating : ℕ) (apples_eaten_per_child : ℕ) : ℕ := children_eating * apples_eaten_per_child
def remaining_apples (initial : ℕ) (eaten : ℕ) : ℕ := initial - eaten
def sold_apples (remaining : ℕ) (final : ℕ) : ℕ := remaining - final

-- Given conditions
variable (children : ℕ := 5)
variable (apples_per_child : ℕ := 15)
variable (children_eating : ℕ := 2)
variable (apples_eaten_per_child : ℕ := 4)
variable (final_apples : ℕ := 60)

-- Theorem statement
theorem child_sold_apples_correct :
  sold_apples (remaining_apples (initial_apples children apples_per_child) (eaten_apples children_eating apples_eaten_per_child)) final_apples = 7 :=
by
  sorry -- Proof is omitted

end child_sold_apples_correct_l485_485933


namespace gcd_45_75_l485_485097

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485097


namespace price_reduction_l485_485930

theorem price_reduction (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : 150 * (1 - x) * (1 - x) = 96 :=
sorry

end price_reduction_l485_485930


namespace largest_20_supporting_X_l485_485406

-- Define the predicate for a number X being 20-supporting
def is_20_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 20 → ℝ), (∑ i, a i).toInt = ∑ i, a i →
  ∃ i, |a i - 0.5| ≥ X

-- Statement to prove the largest 20-supporting X, which is 1/40
theorem largest_20_supporting_X : ∃ X : ℝ, X = 1 / 40 ∧ is_20_supporting X := by
  sorry

end largest_20_supporting_X_l485_485406


namespace triangle_vertices_minimum_area_sum_l485_485872

noncomputable def minimum_area_triangle_sum (k : ℤ) : ℤ :=
if (k = 11 ∨ k = 13) then k else 0

theorem triangle_vertices_minimum_area_sum :
  (minimum_area_triangle_sum 11) + (minimum_area_triangle_sum 13) = 24 :=
by {
  simp [minimum_area_triangle_sum],
  sorry
}

end triangle_vertices_minimum_area_sum_l485_485872


namespace find_a_value_l485_485700

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else x^2

theorem find_a_value :
  (f 4 = 2 * f a) → (a = -1 ∨ a = 2) :=
by 
  sorry

end find_a_value_l485_485700


namespace sun_not_set_point_shadow_length_correct_l485_485816

-- Defining the known constants and values
def declination : ℝ := 22 + 4/60 -- Declination in degrees
def arctic_circle_latitude : ℝ := 66 + 33/60 -- Latitude of Arctic Circle in degrees
def shadow_location_latitude : ℝ := 36 + 22/60 -- Latitude of shadow location in degrees
def rod_height : ℝ := 1 -- Height of the rod in meters

-- Define the effective latitude where the sun does not set
def effective_latitude : ℝ := arctic_circle_latitude - declination

-- Define the sun elevation angle at the shadow location
def sun_elevation_angle : ℝ := 90 - (shadow_location_latitude - declination)

-- Define the length of the shadow
def shadow_length : ℝ := rod_height / (Real.tan (sun_elevation_angle * Real.pi / 180)) -- Conversion to radians for tan function

-- Prove that the sun does not set above a specific latitude on July 11
theorem sun_not_set_point : effective_latitude = 44 + 29/60 := 
by
  calc
    effective_latitude = arctic_circle_latitude - declination : rfl
    ... = (66 + 33/60) - (22 + 4/60) : rfl
    ... = 44 + 29/60 : by norm_num

-- Prove the length of the shadow at the specified location and time
theorem shadow_length_correct : shadow_length ≈ 0.264 :=
by
  calc
    shadow_length = rod_height / (Real.tan (sun_elevation_angle * Real.pi / 180)) : rfl
    ... = 1 / (Real.tan ((90 - (36 + 22/60 - (22 + 4/60)) * Real.pi / 180))) : rfl
    ... = 1 / (Real.tan (75.7 * Real.pi / 180)) : by norm_num
    ... ≈ 0.264 : by norm_num

#check sun_not_set_point -- Sanity check to ensure it compiles
#check shadow_length_correct -- Sanity check to ensure it compiles

end sun_not_set_point_shadow_length_correct_l485_485816


namespace number_of_solutions_pi_equation_l485_485004

theorem number_of_solutions_pi_equation : 
  ∃ (x0 x1 : ℝ), (x0 = 0 ∧ x1 = 1) ∧ ∀ x : ℝ, (π^(x-1) * x^2 + π^(x^2) * x - π^(x^2) = x^2 + x - 1 ↔ x = x0 ∨ x = x1)
:=
by sorry

end number_of_solutions_pi_equation_l485_485004


namespace tangent_line_at_origin_range_of_a_for_positive_f_l485_485293

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) * Real.exp x - (a + 1) * x - 1

theorem tangent_line_at_origin (a : ℝ) : 
  let f0 := f a 0 in 
  let f' := fun x => Deriv (fun x => f a x) x in
  f' 0 = 0 ∧ f0 = 0 ∧ (y = f0) = (y = 0) :=
by
  sorry

theorem range_of_a_for_positive_f (a : ℝ) : 
  (∀ x > 0, f a x > 0) ↔ 0 ≤ a :=
by
  sorry

end tangent_line_at_origin_range_of_a_for_positive_f_l485_485293


namespace binary_representation_of_21_l485_485215

theorem binary_representation_of_21 : Nat.binaryRep 21 = 10101 := 
sorry

end binary_representation_of_21_l485_485215


namespace smallest_winning_strategy_B_l485_485046

def winning_strategy_B (n : ℕ) : Prop :=
  n = 2048

theorem smallest_winning_strategy_B (n : ℕ) (h : n > 1992) : 
  ∃ m : ℕ, n = 2^m ∧ (winning_strategy_B n) :=
begin
  use (11 : ℕ), -- since 2^11 = 2048
  split,
  { -- proof that 2^11 = 2048
    exact rfl,
  },
  { -- proof that winning_strategy_B 2048 holds
    refl,
  }
end

end smallest_winning_strategy_B_l485_485046


namespace gcd_45_75_l485_485096

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485096


namespace swim_club_pass_percentage_l485_485546

theorem swim_club_pass_percentage
  (N : ℕ)
  (H_N : N = 60)
  (notPassedWithCourse : ℕ)
  (H_notPassedWithCourse : notPassedWithCourse = 12)
  (notPassedWithoutCourse : ℕ)
  (H_notPassedWithoutCourse : notPassedWithoutCourse = 30) :
  let notPassed := notPassedWithCourse + notPassedWithoutCourse in
  let P := 100 - (notPassed * 100 / N) in
  P = 30 :=
by
  sorry

end swim_club_pass_percentage_l485_485546


namespace gcd_45_75_l485_485055

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485055


namespace agency_comparison_l485_485856

variable (days m : ℝ)

theorem agency_comparison (h : 20.25 * days + 0.14 * m < 18.25 * days + 0.22 * m) : m > 25 * days :=
by
  sorry

end agency_comparison_l485_485856


namespace focus_of_parabola_l485_485848

theorem focus_of_parabola : 
  (∀ x : ℝ, y = -2 * x^2 → (focus_x, focus_y) = (0, -1/8)) :=
by
  assume x : ℝ
  assume y = -2 * x^2
  sorry

end focus_of_parabola_l485_485848


namespace man_work_rate_complete_work_in_5_days_l485_485939

theorem man_work_rate_complete_work_in_5_days (W : ℝ) (M S : ℝ) :
  ((M + S) * 3 = W) →
  (S * 7.5 = W) →
  (M * 5 = W) :=
by
  intros h1 h2
  -- We use the given conditions to derive the work rate of man.
  have hs : S = W / 7.5 := by
    rw [←h2]
    ring -- simplify the right-hand side
  have hms : M + W / 7.5 = W / 3 := by
    rw [hs] at h1
    exact h1
  have hM : M * 5 = W := by
    -- manipulate hms to derive M in terms of W
    sorry
  exact hM

end man_work_rate_complete_work_in_5_days_l485_485939


namespace xy_product_given_conditions_l485_485047

variable (x y : ℝ)

theorem xy_product_given_conditions (hx : x - y = 5) (hx3 : x^3 - y^3 = 35) : x * y = -6 :=
by
  sorry

end xy_product_given_conditions_l485_485047


namespace integral_evaluation_l485_485994

def integral_problem : Prop :=
  ∫ x in 2..3, (2 * x + 1) = 6

theorem integral_evaluation : integral_problem := by
  sorry

end integral_evaluation_l485_485994


namespace ratio_platform_to_pole_l485_485572

variables (l t T v : ℝ)
-- Conditions
axiom constant_velocity : ∀ t l, l = v * t
axiom pass_pole : l = v * t
axiom pass_platform : 6 * l = v * T 

theorem ratio_platform_to_pole (h1 : l = v * t) (h2 : 6 * l = v * T) : T / t = 6 := 
  by sorry

end ratio_platform_to_pole_l485_485572


namespace theta_sum_is_840_degrees_l485_485037

-- Definition of complex numbers satisfying the given polynomial and modulus conditions
def satisfies_conditions (z : ℂ) : Prop :=
  z^24 - z^6 - 1 = 0 ∧ abs z = 1

-- Define the angles theta_m corresponding to these complex numbers
def theta_m (z : ℂ) : ℝ := complex.arg z * 180 / real.pi

-- Sort the angles in ascending order and extract 2n of them within the given range
def sorted_thetas (n : ℕ) : list ℝ :=
  let angles := (list.range (2 * n)).map (λ i, theta_m ((λ (z : ℂ), satisfies_conditions z).steps i)) in
  list.sort (≤) angles

-- Extract even-indexed thetas
def even_thetas_sum (n : ℕ) : ℝ :=
  (list.range n).map (λ i, list.nth_le (sorted_thetas n) (2 * i) sorry).sum

-- The statement we need to prove
theorem theta_sum_is_840_degrees (n : ℕ) : even_thetas_sum n = 840 :=
by sorry

end theta_sum_is_840_degrees_l485_485037


namespace bug_probability_nine_moves_l485_485149

noncomputable def bug_cube_probability (moves : ℕ) : ℚ := sorry

/-- 
The probability that after exactly 9 moves, a bug starting at one vertex of a cube 
and moving randomly along the edges will have visited every vertex exactly once and 
revisited one vertex once more. 
-/
theorem bug_probability_nine_moves : bug_cube_probability 9 = 16 / 6561 := by
  sorry

end bug_probability_nine_moves_l485_485149


namespace integral_eval_l485_485235

theorem integral_eval : ∫ x in 0..1, (2 * x + Real.sqrt (1 - x^2)) = 1 + Real.pi / 4 := by
  sorry

end integral_eval_l485_485235


namespace log_addition_closed_l485_485794

def is_log_of_nat (n : ℝ) : Prop := ∃ k : ℕ, k > 0 ∧ n = Real.log k

theorem log_addition_closed (a b : ℝ) (ha : is_log_of_nat a) (hb : is_log_of_nat b) : is_log_of_nat (a + b) :=
by
  sorry

end log_addition_closed_l485_485794


namespace inequality_proof_l485_485322

def sqrt2 := Real.sqrt 2
def log_π_3 := Real.logBase (Real.pi) 3
def log2_0_5 := Real.logBase 2 0.5

theorem inequality_proof :
  let a := sqrt2
  let b := log_π_3
  let c := log2_0_5
  a > b ∧ b > c := by
    sorry

end inequality_proof_l485_485322


namespace impossible_to_get_100_pieces_l485_485950

/-- We start with 1 piece of paper. Each time a piece of paper is torn into 3 parts,
it increases the total number of pieces by 2.
Therefore, the number of pieces remains odd through any sequence of tears.
Prove that it is impossible to obtain exactly 100 pieces. -/
theorem impossible_to_get_100_pieces : 
  ∀ n, n = 1 ∨ (∃ k, n = 1 + 2 * k) → n ≠ 100 :=
by
  sorry

end impossible_to_get_100_pieces_l485_485950


namespace evaluate_expression_l485_485920

noncomputable def greatest_integer_le (x : Real) : Int := Int.floor x

theorem evaluate_expression :
  greatest_integer_le 6.5 * greatest_integer_le (2 / 3) + greatest_integer_le 2 * 7.2 + greatest_integer_le 8.3 - 6.6 = 15.8 :=
by
  -- Mathematical details have been verified in the solution.
  sorry

end evaluate_expression_l485_485920


namespace minimum_m_value_l485_485668

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * cos (2 * x) - sin (2 * x)

theorem minimum_m_value (m : ℝ) (h₀ : m > 0) (h₁ : ∀ x : ℝ, (f (x - m) = -f (-x + m))) :
  m = π / 3 :=
sorry

end minimum_m_value_l485_485668


namespace volume_of_body_l485_485590

noncomputable def volume_bounded_by_surface : ℝ :=
  ∫ x in 0..1, ∫ y in 0..2, x * y^2

theorem volume_of_body : volume_bounded_by_surface = 4 / 3 := 
by sorry

end volume_of_body_l485_485590


namespace grid_five_broken_lines_of_length_eight_grid_eight_broken_lines_of_length_five_l485_485362

theorem grid_five_broken_lines_of_length_eight (grid : Type*) (vertex : grid → ℕ) (degree : grid → ℕ → ℕ) :
  ¬ (∃ (paths : ℕ → grid → grid → Prop) (five_lines : ℕ → grid → ℕ), 
    (∀ i, (five_lines i) = 8 ∧
    ∀ v, degree vertex v = odd ∧ 
    (count_odd_vertices grid vertex degree = 10))) :=
sorry

theorem grid_eight_broken_lines_of_length_five (grid : Type*) (vertex : grid → ℕ) (degree : grid → ℕ → ℕ) :
  ∃ (paths : ℕ → grid → grid → Prop) (eight_lines : ℕ → grid → ℕ), 
    (∀ i, (eight_lines i) = 5 ∧
    ∀ v, degree vertex v = odd ∧ 
    (count_odd_vertices grid vertex degree = 12)) :=
sorry

end grid_five_broken_lines_of_length_eight_grid_eight_broken_lines_of_length_five_l485_485362


namespace Balaganov_made_a_mistake_l485_485965

variable (n1 n2 n3 : ℕ) (x : ℝ)
variable (average : ℝ)

def total_salary (n1 n2 : ℕ) (x : ℝ) (n3 : ℕ) : ℝ := 27 * n1 + 35 * n2 + x * n3

def number_of_employees (n1 n2 n3 : ℕ) : ℕ := n1 + n2 + n3

noncomputable def calculated_average_salary (n1 n2 : ℕ) (x : ℝ) (n3 : ℕ) : ℝ :=
 total_salary n1 n2 x n3 / number_of_employees n1 n2 n3

theorem Balaganov_made_a_mistake (h₀ : n1 > n2) 
  (h₁ : calculated_average_salary n1 n2 x n3 = average) 
  (h₂ : 31 < average) : false :=
sorry

end Balaganov_made_a_mistake_l485_485965


namespace sum_of_three_numbers_l485_485033

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1 : B = 30)
  (h2 : A * 3 = 2 * B)
  (h3 : C * 5 = 8 * B) : 
  A + B + C = 98 :=
by
  sorry

end sum_of_three_numbers_l485_485033


namespace center_square_side_length_approx_l485_485936

-- Define the dimensions of the large square
def largeSquareSide : ℝ := 144

-- Define the total area of the large square
def totalArea : ℝ := largeSquareSide * largeSquareSide

-- Define the fraction of the total area that each L-shaped region occupies
def lRegionFraction : ℝ := 1 / 9

-- Define the number of L-shaped regions
def numLRegions : ℕ := 4

-- Define the total area occupied by the L-shaped regions
def totalLRegionArea : ℝ := numLRegions * lRegionFraction * totalArea

-- Define the area of the center square
def centerSquareArea : ℝ := totalArea - totalLRegionArea

-- State the theorem to prove the side length of the center square is approximately 107 inches
theorem center_square_side_length_approx :
  sqrt centerSquareArea ≈ 107 := 
sorry

end center_square_side_length_approx_l485_485936


namespace solution_l485_485999

/-
Define the problem conditions using Lean 4
-/

def distinctPrimeTriplesAndK : Prop :=
  ∃ (p q r : ℕ) (k : ℕ), p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
    (pq - k) % r = 0 ∧ (qr - k) % p = 0 ∧ (rp - k) % q = 0 ∧ (pq - k) > 0

/-
Expected solution based on the solution steps
-/
theorem solution : distinctPrimeTriplesAndK :=
  ∃ (p q r k : ℕ), p = 2 ∧ q = 3 ∧ r = 5 ∧ k = 1 ∧ 
    p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
    (p * q - k) % r = 0 ∧ (q * r - k) % p = 0 ∧ (r * p - k) % q = 0 ∧ (p * q - k) > 0 := 
  by {
    sorry
  }

end solution_l485_485999


namespace carla_total_marbles_l485_485211

def initial_marbles : ℝ := 187.0
def bought_marbles : ℝ := 134.0

theorem carla_total_marbles : initial_marbles + bought_marbles = 321.0 := by
  sorry

end carla_total_marbles_l485_485211


namespace value_of_a_when_b_is_24_l485_485035

variable (a b k : ℝ)

theorem value_of_a_when_b_is_24 (h1 : a = k / b^2) (h2 : 40 = k / 12^2) (h3 : b = 24) : a = 10 :=
by
  sorry

end value_of_a_when_b_is_24_l485_485035


namespace circumcircle_eq_A1B1C1_incircle_eq_A2B2C2_l485_485923

-- Definitions for the vertices of the triangles
def A1 : (ℝ × ℝ) := (5, 1)
def B1 : (ℝ × ℝ) := (7, -3)
def C1 : (ℝ × ℝ) := (2, -8)

def A2 : (ℝ × ℝ) := (0, 0)
def B2 : (ℝ × ℝ) := (5, 0)
def C2 : (ℝ × ℝ) := (0, 12)

-- Define the equation of a circle centered at (a, b) with radius r
def circle_eq (a b r : ℝ) : ℝ × ℝ → Prop := λ p, (p.1 - a)^2 + (p.2 - b)^2 = r^2

-- Declare the theorems to be proved
theorem circumcircle_eq_A1B1C1 : circle_eq 2 (-3) 5 A1 ∧ circle_eq 2 (-3) 5 B1 ∧ circle_eq 2 (-3) 5 C1 :=
sorry

theorem incircle_eq_A2B2C2 : circle_eq 2 2 2 A2 ∧ circle_eq 2 2 2 B2 ∧ circle_eq 2 2 2 C2 :=
sorry

end circumcircle_eq_A1B1C1_incircle_eq_A2B2C2_l485_485923


namespace problem_conditions_l485_485720

theorem problem_conditions (x y : ℝ) (h : x^2 + y^2 - x * y = 1) :
  ¬ (x + y ≤ 1) ∧ (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) ∧ ¬ (x^2 + y^2 ≥ 1) :=
by
  sorry

end problem_conditions_l485_485720


namespace beverage_price_l485_485883

theorem beverage_price (P : ℝ) :
  (3 * 2.25 + 4 * P + 4 * 1.00) / 6 = 2.79 → P = 1.50 :=
by
  intro h -- Introduce the hypothesis.
  sorry  -- Proof is omitted.

end beverage_price_l485_485883


namespace expression_evaluation_l485_485232

theorem expression_evaluation : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 :=
by
  sorry

end expression_evaluation_l485_485232


namespace min_trials_correct_l485_485490

noncomputable def minimum_trials (α p : ℝ) (hα : 0 < α ∧ α < 1) (hp : 0 < p ∧ p < 1) : ℕ :=
  Nat.floor ((Real.log (1 - α)) / (Real.log (1 - p))) + 1

-- The theorem to prove the correctness of minimum_trials
theorem min_trials_correct (α p : ℝ) (hα : 0 < α ∧ α < 1) (hp : 0 < p ∧ p < 1) :
  ∃ n : ℕ, minimum_trials α p hα hp = n ∧ (1 - (1 - p)^n ≥ α) :=
by
  sorry

end min_trials_correct_l485_485490


namespace volume_of_given_wedge_l485_485932

noncomputable def volume_of_wedge (d : ℝ) (angle : ℝ) : ℝ := 
  let r := d / 2
  let height := d
  let cos_angle := Real.cos angle
  (r^2 * height * Real.pi / 2) * cos_angle

theorem volume_of_given_wedge :
  volume_of_wedge 20 (Real.pi / 6) = 1732 * Real.pi :=
by {
  -- The proof logic will go here.
  sorry
}

end volume_of_given_wedge_l485_485932


namespace nature_of_quadrilateral_I_A_I_B_I_C_I_D_l485_485790

-- Define cyclic quadrilateral and incircle centers
variables {A B C D I_A I_B I_C I_D : Type} [point I_A] [point I_B] [point I_C] [point I_D]
variables [cyclic_quadrilateral A B C D] 

-- Define the centers of the incircles of the given triangles
variables (is_incircle_center_I_A : is_incircle_center I_A (triangle B C D))
variables (is_incircle_center_I_B : is_incircle_center I_B (triangle D C A))
variables (is_incircle_center_I_C : is_incircle_center I_C (triangle A D B))
variables (is_incircle_center_I_D : is_incircle_center I_D (triangle B A C))

-- Prove the nature of the quadrilateral I_A I_B I_C I_D
theorem nature_of_quadrilateral_I_A_I_B_I_C_I_D :
  is_rectangle (quadrilateral I_A I_B I_C I_D) :=
by
  sorry

end nature_of_quadrilateral_I_A_I_B_I_C_I_D_l485_485790


namespace value_of_m_div_x_l485_485026

noncomputable def ratio_of_a_to_b (a b : ℝ) : Prop := a / b = 4 / 5
noncomputable def x_value (a : ℝ) : ℝ := a * 1.75
noncomputable def m_value (b : ℝ) : ℝ := b * 0.20

theorem value_of_m_div_x (a b : ℝ) (h1 : ratio_of_a_to_b a b) (h2 : 0 < a) (h3 : 0 < b) :
  (m_value b) / (x_value a) = 1 / 7 :=
by
  sorry

end value_of_m_div_x_l485_485026


namespace proof_problem_l485_485290

def f (x : ℝ) : ℝ := abs (cos x) * sin x

theorem proof_problem : 
  (f (2015 * π / 3) = - sqrt 3 / 4) ∧
  ¬ (∀ x1 x2 : ℝ, |f x1| = |f x2| → ∃ k : ℤ, x1 = x2 + k * π) ∧
  (∀ x : ℝ, -π / 4 ≤ x ∧ x ≤ π / 4 → f x ≤ f (x + π / 8)) ∧
  ¬ (∀ x : ℝ, f (x + π) = f x) ∧
  (∃ x : ℝ, f (π - x) = -f (π + x)) :=
sorry  -- proof is not required

end proof_problem_l485_485290


namespace kenya_peanuts_eq_133_l485_485779

def num_peanuts_jose : Nat := 85
def additional_peanuts_kenya : Nat := 48

def peanuts_kenya (jose_peanuts : Nat) (additional_peanuts : Nat) : Nat :=
  jose_peanuts + additional_peanuts

theorem kenya_peanuts_eq_133 : peanuts_kenya num_peanuts_jose additional_peanuts_kenya = 133 := by
  sorry

end kenya_peanuts_eq_133_l485_485779


namespace f_value_2022_l485_485462

def f (n : ℕ) : ℝ := sorry

axiom f_conditions_0 : f 4 = 2

axiom f_conditions_1 : ∀ (n : ℕ), f (n + 1) = (∑ i in finset.range (n+1), 1 / (f i + f (i + 1)))

theorem f_value_2022 : f 2022 = Real.sqrt (2022) :=
sorry

end f_value_2022_l485_485462


namespace largest_twenty_supporting_l485_485401

def is_twenty_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 20 → ℝ),
    (∑ i, a i : ℝ).round = (∑ i, a i : ℝ) →
    ∃ i, |a i - 1/2| ≥ X

theorem largest_twenty_supporting :
  (∃ X : ℝ, is_twenty_supporting X ∧ ∀ Y : ℝ, is_twenty_supporting Y → Y ≤ X) ∧
  ∀ Z : ℝ, is_twenty_supporting Z → abs ((X : ℝ) - 0.025) < 0.001 :=
sorry

end largest_twenty_supporting_l485_485401


namespace percent_increase_may_to_june_l485_485470

variables (P : ℝ) (x : ℝ)

-- Define the conditions provided in the problem
def profit_apr := 1.35 * P
def profit_may := 1.08 * P
def profit_jun_may_percent_increase := profit_may * (1 + x)

-- Define the given information
def overall_percent_increase := 1.62 * P

-- State the proof problem
theorem percent_increase_may_to_june
  (h1 : profit_jun_may_percent_increase = overall_percent_increase) :
  x = 0.5 :=
by
  -- Skipping the proof
  sorry

end percent_increase_may_to_june_l485_485470


namespace gcd_45_75_l485_485099

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485099


namespace gcd_45_75_l485_485075

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485075


namespace evaluate_g_sum_l485_485785

def g (x : ℝ) : ℝ :=
if x > 3 then x^2 + 4
else if -3 ≤ x ∧ x ≤ 3 then 3*x + 1
else -1

theorem evaluate_g_sum : g (-4) + g 0 + g 4 = 20 := 
by
  sorry

end evaluate_g_sum_l485_485785


namespace Carter_cards_l485_485812

variable (C : ℕ) -- Let C be the number of baseball cards Carter has.

-- Condition 1: Marcus has 210 baseball cards.
def Marcus_cards : ℕ := 210

-- Condition 2: Marcus has 58 more cards than Carter.
def Marcus_has_more (C : ℕ) : Prop := Marcus_cards = C + 58

theorem Carter_cards (C : ℕ) (h : Marcus_has_more C) : C = 152 :=
by
  -- Expand the condition
  unfold Marcus_has_more at h
  -- Simplify the given equation
  rw [Marcus_cards] at h
  -- Solve for C
  linarith

end Carter_cards_l485_485812


namespace luke_number_of_rounds_l485_485410

variable (points_per_round total_points : ℕ)

theorem luke_number_of_rounds 
  (h1 : points_per_round = 3)
  (h2 : total_points = 78) : 
  total_points / points_per_round = 26 := 
by 
  sorry

end luke_number_of_rounds_l485_485410


namespace sum_of_remainders_l485_485907

theorem sum_of_remainders (n : ℤ) (h₁ : n % 12 = 5) (h₂ : n % 3 = 2) (h₃ : n % 4 = 1) : 2 + 1 = 3 := by
  sorry

end sum_of_remainders_l485_485907


namespace kataleya_paid_correct_amount_l485_485956

-- Definitions based on the conditions
def cost_per_peach : ℝ := 0.40 -- dollars
def number_of_peaches : ℕ := 400
def discount_per_10_dollars : ℝ := 2 -- dollars
def threshold_purchase_amount : ℝ := 10 -- dollars

-- Calculation based on the problem statement
def total_cost : ℝ := number_of_peaches * cost_per_peach
def total_10_dollar_purchases : ℕ := total_cost / threshold_purchase_amount
def total_discount : ℝ := (total_10_dollar_purchases : ℝ) * discount_per_10_dollars
def final_amount_paid : ℝ := total_cost - total_discount

-- Statement to prove
theorem kataleya_paid_correct_amount : final_amount_paid = 128 := 
by
  sorry

end kataleya_paid_correct_amount_l485_485956


namespace length_of_platform_l485_485515

def train_length := 300 -- meters
def time_to_cross_platform := 40 -- seconds
def time_to_cross_signal := 18 -- seconds

theorem length_of_platform (L P T_p T_s : ℕ) (hL : L = 300) (hTp : T_p = 40) (hTs : T_s = 18) : 
P = 367 :=
by
    have hSpeed : L / T_s = 300 / 18 := by rw [hL, hTs]
    have hTotalDistance := L + P
    have equation := (L + P) / T_p
    rw [hTp, hTotalDistance, hSpeed, L] at equation
    sorry  -- completes with proof: P = 367  

end length_of_platform_l485_485515


namespace base_conversion_arithmetic_l485_485980

theorem base_conversion_arithmetic :
  let b10 := 2468 : ℕ,
      b3 := 1 * 3^0 + 1 * 3^1 + 1 * 3^2,
      b9 := 1 * 9^0 + 7 * 9^1 + 4 * 9^2 + 3 * 9^3,
      b7 := 4 * 7^0 + 3 * 7^1 + 2 * 7^2 + 1 * 7^3
  in b10 / b3 - b9 + b7 = -1919 :=
by 
  sorry

end base_conversion_arithmetic_l485_485980


namespace gcd_45_75_l485_485056

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485056


namespace colin_next_two_miles_time_l485_485594

theorem colin_next_two_miles_time : 
  ∃ x : ℝ, t1 = 6 ∧ t4 = 4 ∧ avg_time = 5 ∧ 
    (t1 + t2 + t3 + t4 = 4 * avg_time) ∧ 
    t2 = x ∧ t3 = x ∧ x = 5 :=
by
  let x := (20 - (6 + 4)) / 2
  use x
  have t1 := 6
  have t4 := 4
  have avg_time := 5
  have total_time := 4 * avg_time
  have first_fourth_time := t1 + t4
  have remaining_time := total_time - first_fourth_time
  have next_two_time := remaining_time / 2
  have x_eq_5 := (20 - 10) / 2
  have eq_proof : next_two_time = 5 := by linarith
  exact ⟨x_eq_5, t1, t4, avg_time, by linarith, by linarith, eq_proof⟩

end colin_next_two_miles_time_l485_485594


namespace interval_satisfies_ineq_l485_485017

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end interval_satisfies_ineq_l485_485017


namespace A_intersection_B_l485_485304

open Set

noncomputable def A : Set ℝ := { x | 1 / 2 ≤ 2^x ∧ 2^x < 16 }
noncomputable def B : Set ℝ := { x | ∃ y, y = log 2 (9 - x^2) }

theorem A_intersection_B :
  A ∩ B = { x | -1 ≤ x ∧ x < 3 } :=
sorry

end A_intersection_B_l485_485304


namespace arc_length_ln_sin_l485_485529

noncomputable def arcLength (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt (1 + (deriv f x)^2)

def f (x : ℝ) : ℝ := 1 - Real.log (Real.sin x)

theorem arc_length_ln_sin :
  arcLength f (Real.pi / 3) (Real.pi / 2) = (Real.log 3) / 2 := by
  sorry

end arc_length_ln_sin_l485_485529


namespace age_ratio_rahul_deepak_l485_485877

/--
Prove that the ratio between Rahul and Deepak's current ages is 4:3 given the following conditions:
1. After 10 years, Rahul's age will be 26 years.
2. Deepak's current age is 12 years.
-/
theorem age_ratio_rahul_deepak (R D : ℕ) (h1 : R + 10 = 26) (h2 : D = 12) : R / D = 4 / 3 :=
by sorry

end age_ratio_rahul_deepak_l485_485877


namespace part_a_part_b_l485_485658

variable (n : ℕ) (a : fin n → ℝ) (b : fin n → ℝ)
variable (S T : ℝ)

-- Condition: n ≥ 3
-- Positive real numbers
axiom hn : n ≥ 3
axiom ha_pos : ∀ i, 0 < a i
axiom hb_pos : ∀ i, 0 < b i
-- Pairwise distinct b_i
axiom hb_distinct : function.injective b

-- Definitions of S and T
def S := ∑ i, a i
def T := ∏ i, b i

-- Part (a): The polynomial f(x) has n-1 distinct real zeroes
def f (x : ℝ) := ∏ i, (x - b i) * ∑ i, a i / (x - b i)

theorem part_a : fin n → ℝ := sorry

-- Part (b): Proving the inequality
theorem part_b : (1 / (n - 1)) * ∑ i, (1 - a i / S) * b i > (T / S * ∑ i, a i / b i) ^ (1 / (n - 1)) := sorry

end part_a_part_b_l485_485658


namespace value_in_base_5_l485_485902

theorem value_in_base_5 :
  ∀ (n m : ℕ), 
    n = 30 ∧ m = 12 → 
    (nat.sub n m) = 18 → 
    nat.toDigits 5 (nat.sub n m) = [3, 3] :=
by
  intros n m h1 h2
  cases h1
  rw [h1_left, h1_right]
  rw h2
  sorry

end value_in_base_5_l485_485902


namespace smallest_solution_is_neg_sqrt_13_l485_485224

noncomputable def smallest_solution (x : ℝ) : Prop :=
  x^4 - 26 * x^2 + 169 = 0 ∧ ∀ y : ℝ, y^4 - 26 * y^2 + 169 = 0 → x ≤ y

theorem smallest_solution_is_neg_sqrt_13 :
  smallest_solution (-Real.sqrt 13) :=
by
  sorry

end smallest_solution_is_neg_sqrt_13_l485_485224


namespace combined_tax_rate_l485_485135

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (h1 : Mindy_income = 4 * Mork_income) :
  let Mork_tax := 0.45 * Mork_income;
  let Mindy_tax := 0.15 * Mindy_income;
  let combined_tax := Mork_tax + Mindy_tax;
  let combined_income := Mork_income + Mindy_income;
  combined_tax / combined_income * 100 = 21 := 
by
  sorry

end combined_tax_rate_l485_485135


namespace father_seven_times_son_years_ago_l485_485162

constant father_current_age : Nat := 38
constant son_current_age : Nat := 14

theorem father_seven_times_son_years_ago :
  ∃ (x : Nat), (father_current_age - x = 7 * (son_current_age - x)) ∧ x = 10 :=
by
  sorry

end father_seven_times_son_years_ago_l485_485162


namespace kataleya_total_amount_paid_l485_485959

/-- A store offers a $2 discount for every $10 purchase on any item in the store.
Kataleya went to the store and bought 400 peaches sold at forty cents each.
Prove that the total amount of money she paid at the store for the fruits is $128. -/
theorem kataleya_total_amount_paid : 
  let price_per_peach : ℝ := 0.40
  let number_of_peaches : ℝ := 400 
  let total_cost : ℝ := number_of_peaches * price_per_peach
  let discount_per_10_dollars : ℝ := 2
  let number_of_discounts := total_cost / 10
  let total_discount := number_of_discounts * discount_per_10_dollars
  let amount_paid := total_cost - total_discount
  amount_paid = 128 :=
by
  sorry

end kataleya_total_amount_paid_l485_485959


namespace area_of_triangle_l485_485527

-- We are given the perimeter and the inradius of the triangle.
variables (p : ℝ) (r : ℝ)

-- The conditions of the problem.
def given_conditions : Prop := (p = 24) ∧ (r = 2.5)

-- The formula for the area of a triangle using inradius and perimeter.
def area (p r : ℝ) : ℝ := r * p / 2

-- The statement we aim to prove: Given the perimeter is 24 and the inradius is 2.5, the area is 30.
theorem area_of_triangle : given_conditions p r → area p r = 30 := by
  intro h
  cases h with hp hr
  rw [hp, hr]
  simp [area]
  sorry

end area_of_triangle_l485_485527


namespace palindrome_percentage_l485_485123

-- Define the characteristics of the palindromes
def is_palindrome (n : ℕ) : Prop :=
  let digits := (list.reverse (nat.digits 10 n)) in
  digits = nat.digits 10 n

-- Define the predicate for the allowed range
def in_range (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 5000

-- Define the predicate for containing the digit 7
def contains_digit_7 (n : ℕ) : Prop :=
  7 ∈ (nat.digits 10 n)
  
-- Define the overall condition for a valid palindrome in the given range
def valid_palindrome (n : ℕ) : Prop :=
  is_palindrome n ∧ in_range n

theorem palindrome_percentage (p : ℕ → Prop) (h : ∀ n, p n ↔ valid_palindrome n ∧ contains_digit_7 n):
  (finset.card (finset.filter contains_digit_7 (finset.filter valid_palindrome (finset.range 5000)))) * 10 / (finset.card (finset.filter valid_palindrome (finset.range 5000))) = 10 :=
sorry

end palindrome_percentage_l485_485123


namespace incenter_locus_l485_485892

variables {A B C : Type} [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
  {O : Point} {R : Length} {ABC : Triangle A B C} {Ω : Circle O R}
  {d_a d_b d_c : Line}

def areSymmetricLines (l : Line) (side1 side2 : Line) : Prop :=
  -- Definition for symmetric lines could be implemented here if needed.
  sorry

/-- Given a triangle ABC with circumcircle Ω, center O, and radius R. 
Let d_a, d_b, d_c be lines parallel through A, B, C respectively. 
If lines symmetric to these with respect to sides BC, CA, AB form a triangle XYZ,
then the locus of the incenter of XYZ is the circle concentric with Ω of radius 2R. -/
theorem incenter_locus (h_parallels : ∀ {A B : Point}, parallel (d_a : Line A) (d_b : Line B))
  (h_symmetric_a : areSymmetricLines d_a (Line.mk B C))
  (h_symmetric_b : areSymmetricLines d_b (Line.mk C A))
  (h_symmetric_c : areSymmetricLines d_c (Line.mk A B)) :
  ∃ (locus : Circle), circumcenter locus = O ∧ radius locus = 2 * R :=
sorry

end incenter_locus_l485_485892


namespace sum_of_three_pentagons_l485_485189

variable (x y : ℚ)

axiom eq1 : 3 * x + 2 * y = 27
axiom eq2 : 2 * x + 3 * y = 25

theorem sum_of_three_pentagons : 3 * y = 63 / 5 := 
by {
  sorry -- No need to provide proof steps
}

end sum_of_three_pentagons_l485_485189


namespace probability_all_6_numbers_appear_l485_485491

noncomputable def probability_all_numbers_appear_at_least_once (n m : ℕ) (k : Fin n) : ℝ := 
  let dies := pmf.pure (λ (_ : Fin m) => fin.choose (Fin 6))
  let events := List.map (λ i => pmf.mass dies {ω | ω i = k}) (List.range m)
  1 - ∑ s in Finset.powersetFin (Finset.univ : Finset (Fin 6)), (-1) ^ (Finset.card s + 1) * ∏ i in s, events i

theorem probability_all_6_numbers_appear :
  probability_all_numbers_appear_at_least_once 10 10 6 = 0.2718 := 
sorry

end probability_all_6_numbers_appear_l485_485491


namespace additional_toothpicks_needed_l485_485196

theorem additional_toothpicks_needed :
  ∀ (initial_tooothpicks: ℕ) (steps_main: ℕ) (steps_smaller: ℕ) (toothpicks_initial_4step: ℕ),
  initial_tooothpicks = 26 → steps_main = 6 → steps_smaller = 3 → toothpicks_initial_4step = 26 →
  (let total_needed : ℕ := 42 + 18 in total_needed - initial_tooothpicks = 34) :=
by
  intros initial_tooothpicks steps_main steps_smaller toothpicks_initial_4step 
  intros h1 h2 h3 h4 
  dsimp
  sorry

end additional_toothpicks_needed_l485_485196


namespace sum_polynomials_constant_or_increasing_l485_485885

theorem sum_polynomials_constant_or_increasing (n : ℕ) :
  ∀ (A B : Type) (replaceA : A → ℝ) (replaceB : B → ℝ), 
    ( ∀ word : (fin n → A ⊕ B), 
      let polynomial := (λ letters, 
        (finset.range n).sum (λ i, 
          match letters i with 
          | (sum.inl a) := replaceA a
          | (sum.inr b) := replaceB b 
          end))
      in polynomial word)
  →
  (let polynomials := finset.range (2^n) in 
  ∃ sum_poly : ℝ → ℝ, 
    (∀ k, 1 ≤ k → k ≤ 2^n → 
      (sum_poly = polynomials.sum (λ i, polynomial i)) 
        ∧ (∀ x ∈ set.Icc (0:ℝ) 1, 
            (monotonic_increasing sum_poly x) 
            ∨ (is_constant_on sum_poly x))) :=
sorry

end sum_polynomials_constant_or_increasing_l485_485885


namespace d_share_l485_485570

theorem d_share (T : ℝ) (A B C D E : ℝ) 
  (h1 : A = 5 / 15 * T) 
  (h2 : B = 2 / 15 * T) 
  (h3 : C = 4 / 15 * T)
  (h4 : D = 3 / 15 * T)
  (h5 : E = 1 / 15 * T)
  (combined_AC : A + C = 3 / 5 * T)
  (diff_BE : B - E = 250) : 
  D = 750 :=
by
  sorry

end d_share_l485_485570


namespace percentage_increase_is_20_l485_485369

def first_hospital_patients_daily : ℕ := 20
def total_patients_yearly : ℕ := 11000
def work_days_per_week : ℕ := 5
def work_weeks_per_year : ℕ := 50

noncomputable def percentage_increase_in_patients
  (first_hospital_patients_daily : ℕ)
  (total_patients_yearly : ℕ)
  (work_days_per_week : ℕ)
  (work_weeks_per_year : ℕ) : ℕ :=
let first_hospital_patients_yearly := first_hospital_patients_daily * work_days_per_week * work_weeks_per_year in
let second_hospital_patients_yearly := total_patients_yearly - first_hospital_patients_yearly in
let second_hospital_patients_daily := second_hospital_patients_yearly / (work_days_per_week * work_weeks_per_year) in
let increase := second_hospital_patients_daily - first_hospital_patients_daily in
(increase * 100) / first_hospital_patients_daily

theorem percentage_increase_is_20 :
  percentage_increase_in_patients first_hospital_patients_daily total_patients_yearly work_days_per_week work_weeks_per_year = 20 :=
by
  -- Proof is omitted.
  sorry

end percentage_increase_is_20_l485_485369


namespace sum_lent_250_l485_485167

theorem sum_lent_250 (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (hR : R = 4) (hT : T = 8) (hSI1 : SI = P - 170) 
  (hSI2 : SI = (P * R * T) / 100) : 
  P = 250 := 
by 
  sorry

end sum_lent_250_l485_485167


namespace probability_sum_17_with_10_sided_dice_l485_485855

theorem probability_sum_17_with_10_sided_dice
  (die1 die2 : ℕ)
  (faces : finset ℕ)
  (h_faces : faces = finset.range 11 \ {0})
  (h_die1 : die1 ∈ faces)
  (h_die2 : die2 ∈ faces)
  (H : finset {pair : ℕ × ℕ | pair.1 ∈ faces ∧ pair.2 ∈ faces ∧ pair.1 + pair.2 = 17}.card = 4)
  (total_outcomes : ℕ)
  (h_total : total_outcomes = finset.card (finset.product faces faces)) :
  (4 / 100) = (1 / 25) :=
by sorry

end probability_sum_17_with_10_sided_dice_l485_485855


namespace even_count_in_top_15_rows_l485_485603

def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

def count_even_in_row (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ k, is_even (binom n k)).card

def count_even_in_top_15_rows : ℕ :=
  (Finset.range 15).sum count_even_in_row

theorem even_count_in_top_15_rows :
  count_even_in_top_15_rows = "Sum of all identified evens" := by
sorry

end even_count_in_top_15_rows_l485_485603


namespace trains_meet_480_km_away_l485_485522

-- Define the conditions
def bombay_express_speed : ℕ := 60 -- speed in km/h
def rajdhani_express_speed : ℕ := 80 -- speed in km/h
def bombay_express_start_time : ℕ := 1430 -- 14:30 in 24-hour format
def rajdhani_express_start_time : ℕ := 1630 -- 16:30 in 24-hour format

-- Define the function to calculate the meeting point distance
noncomputable def meeting_distance (bombay_speed rajdhani_speed : ℕ) (bombay_start rajdhani_start : ℕ) : ℕ :=
  let t := 6 -- time taken for Rajdhani to catch up in hours, derived from the solution
  rajdhani_speed * t

-- The statement we need to prove:
theorem trains_meet_480_km_away :
  meeting_distance bombay_express_speed rajdhani_express_speed bombay_express_start_time rajdhani_express_start_time = 480 := by
  sorry

end trains_meet_480_km_away_l485_485522


namespace solve_for_x_l485_485628

theorem solve_for_x (x : ℝ) (hx : Real.root 4 (x^3 * Real.sqrt x) = 4) : 
  x = 4^(8/7) :=
sorry

end solve_for_x_l485_485628


namespace find_inclination_angle_l485_485988

open Real

noncomputable def line_l_parametric (t α : ℝ) : ℝ × ℝ :=
  (-2 + t * cos α, -4 + t * sin α)

def curve_C_cartesian (x y : ℝ) : Prop :=
  y^2 = 2 * x

theorem find_inclination_angle (α : ℝ) (t1 t2 : ℝ) :
  curve_C_cartesian (-2 + t1 * cos α) (-4 + t1 * sin α) →
  curve_C_cartesian (-2 + t2 * cos α) (-4 + t2 * sin α) →
  (∃ t1 t2 : ℝ, (t1 * t2 = 20 / sin(α)^2) ∧ t1 + t2 = (2 * cos α + 8 * sin α) / sin(α)^2) →
  20 / sin(α)^2 = 40 →
  α = π / 4 :=
begin
  sorry
end

end find_inclination_angle_l485_485988


namespace gcd_45_75_l485_485062

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485062


namespace valid_votes_per_candidate_l485_485344

theorem valid_votes_per_candidate (total_votes : ℕ) (invalid_percentage valid_percentage_A valid_percentage_B : ℚ) 
                                  (A_votes B_votes C_votes valid_votes : ℕ) :
  total_votes = 1250000 →
  invalid_percentage = 20 →
  valid_percentage_A = 45 →
  valid_percentage_B = 35 →
  valid_votes = total_votes * (1 - invalid_percentage / 100) →
  A_votes = valid_votes * (valid_percentage_A / 100) →
  B_votes = valid_votes * (valid_percentage_B / 100) →
  C_votes = valid_votes - A_votes - B_votes →
  valid_votes = 1000000 ∧ A_votes = 450000 ∧ B_votes = 350000 ∧ C_votes = 200000 :=
by {
  sorry
}

end valid_votes_per_candidate_l485_485344


namespace M_gt_N_necessary_but_not_sufficient_l485_485451

theorem M_gt_N_necessary_but_not_sufficient (M N : ℝ) (h : M > N) :
  (M > N ↔ log 2 M > log 2 N) → false :=
by sorry

end M_gt_N_necessary_but_not_sufficient_l485_485451


namespace increase_75_by_150_percent_l485_485501

noncomputable def original_number : Real := 75
noncomputable def percentage_increase : Real := 1.5
noncomputable def increase_amount : Real := original_number * percentage_increase
noncomputable def result : Real := original_number + increase_amount

theorem increase_75_by_150_percent : result = 187.5 := by
  sorry

end increase_75_by_150_percent_l485_485501


namespace gcd_45_75_l485_485070

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485070


namespace perimeter_of_new_square_l485_485955

noncomputable def new_perimeter (area : ℝ) (multiplier : ℝ) : ℝ :=
  let s := real.sqrt area in
  let s' := multiplier * s in
  4 * s'

theorem perimeter_of_new_square (h1 : (real.sqrt 4) = 2) : new_perimeter 4 3 = 24 :=
  by
    rw [new_perimeter, h1]
    sorry

end perimeter_of_new_square_l485_485955


namespace first_term_of_geometric_sequence_l485_485475

theorem first_term_of_geometric_sequence 
  (a r : ℝ)
  (h1 : a * r^2 = 720)
  (h2 : a * r^6 = 9!) :
  a = 20 :=
sorry

end first_term_of_geometric_sequence_l485_485475


namespace gcd_45_75_l485_485068

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485068


namespace number_of_cows_l485_485169

-- Definitions
variable (H C : ℕ)
variable h1 : C = 5 * H
variable h2 : C + H = 168

-- Proof Statement
theorem number_of_cows : C = 140 :=
by
  -- each line needs to be filled according
  sorry

end number_of_cows_l485_485169


namespace distinct_paths_base_regular_octagonal_pyramid_l485_485564

def num_paths_pyramid (base_vertices triangular_faces : ℕ) : ℕ :=
  let direct_paths := base_vertices
  let one_triangle_paths := triangular_faces * 2
  let two_triangle_paths := triangular_faces
  direct_paths + one_triangle_paths + two_triangle_paths

theorem distinct_paths_base_regular_octagonal_pyramid 
  (base_vertices triangular_faces : ℕ)
  (h1 : base_vertices = 8)
  (h2 : triangular_faces = 8) 
  : num_paths_pyramid base_vertices triangular_faces = 32 := by
  rw [h1, h2]
  unfold num_paths_pyramid
  simp
  sorry

end distinct_paths_base_regular_octagonal_pyramid_l485_485564


namespace kenya_peanuts_correct_l485_485777

def jose_peanuts : ℕ := 85
def kenya_more_peanuts : ℕ := 48

def kenya_peanuts : ℕ := jose_peanuts + kenya_more_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := by
  sorry

end kenya_peanuts_correct_l485_485777


namespace n_greater_than_7_l485_485599

theorem n_greater_than_7 (m n : ℕ) (hmn : m > n) (h : ∃k:ℕ, 22220038^m - 22220038^n = 10^8 * k) : n > 7 :=
sorry

end n_greater_than_7_l485_485599


namespace probability_point_in_ellipse_l485_485716

-- Define the number 2014 and its base-5 conversion properties
def base5_repr (x : Nat) : Bool :=
  x = 2014 ∧
  ∃ (α₄ α₃ α₂ α₁ α₀ : ℕ),
    α₄ = 3 ∧ α₃ = 1 ∧ α₂ = 0 ∧ α₁ = 2 ∧ α₀ = 4 ∧
    x = α₄ * 5^4 + α₃ * 5^3 + α₂ * 5^2 + α₁ * 5^1 + α₀ * 5^0

-- Define the set of valid (x, y) coordinates and their probability of falling inside the ellipse
def valid_points (α₀ α₁ α₂ α₃ α₄ : ℕ) : Finset (ℕ × ℕ) :=
  {⟨0,0⟩, ⟨1,1⟩, ⟨2,2⟩, ⟨2,0⟩, ⟨2,1⟩, ⟨0,2⟩, ⟨0,1⟩, ⟨1,2⟩, ⟨1,0⟩, ⟨3,0⟩, ⟨3,1⟩}.toFinset

theorem probability_point_in_ellipse : Prop :=
  base5_repr 2014 →
  let S := Finset.product (Finset.range 5) (Finset.range 5) in
  let E := valid_points 4 2 0 1 3 in
  (E.card : ℚ) / (S.card : ℚ) = 11 / 25

end probability_point_in_ellipse_l485_485716


namespace p_is_prime_and_gt_3_l485_485729

theorem p_is_prime_and_gt_3 (p : ℤ) (h1 : p > 3)
  (h2 : (p^2 + 15) % 12 = 4) : p.prime := 
by
  sorry

end p_is_prime_and_gt_3_l485_485729


namespace focal_distance_of_hyperbola_l485_485265

noncomputable def hyperbola (a b : ℝ) := set_of (λ p : ℝ × ℝ, (p.1 ^ 2) / (a ^ 2) - (p.2 ^ 2) / (b ^ 2) = 1)

theorem focal_distance_of_hyperbola
  (a b: ℝ)
  (e: ℝ)
  (c: ℝ)
  (F: ℝ × ℝ)
  (O: ℝ × ℝ)
  (A: ℝ × ℝ) 
  (h_eq : hyperbola a b)
  (h_eccentricity : e = (Real.sqrt 5) / 2)
  (h_focus : F = (c, 0))
  (h_perpendicular_line : ∃ A_x, A = (A_x, 0) ∧ line_passes_perpendicular_through (F) (x_axis) (A))
  (h_area : 1/2 * a * b = 2)
  (h_pythagorean : a^2 + b^2 = c^2):
  2 * c = 2 * Real.sqrt 10 :=
sorry

end focal_distance_of_hyperbola_l485_485265


namespace molecular_weight_calculation_l485_485121

/-- Define the molecular weight of the compound as 972 grams per mole. -/
def molecular_weight : ℕ := 972

/-- Define the number of moles as 9 moles. -/
def number_of_moles : ℕ := 9

/-- Define the total weight of the compound for the given number of moles. -/
def total_weight : ℕ := number_of_moles * molecular_weight

/-- Prove the total weight is 8748 grams. -/
theorem molecular_weight_calculation : total_weight = 8748 := by
  sorry

end molecular_weight_calculation_l485_485121


namespace k_is_even_l485_485335

def is_divisor (x y : ℕ) : Prop :=
  y % x = 0

def circular_arrangement (nums: List ℕ) (k: ℕ) : Prop :=
  let n := nums.length
  (∀ i, is_divisor (nums.get! i) (nums.get! ((i + 1) % n) + nums.get! ((i - 1 + n) % n))) ∧
  (nums.get! ((nums.indexOf k + 1) % n) % 2 = 1) ∧
  (nums.get! ((nums.indexOf k - 1 + n) % n) % 2 = 1)

theorem k_is_even (nums: List ℕ) (k: ℕ) (h1 : nums = List.range 1 1001) (h2 : circular_arrangement nums k) : k % 2 = 0 := by
  sorry

end k_is_even_l485_485335


namespace find_matrix_A_l485_485642

-- Defining the vector t
def t : Fin 3 → ℝ := ![1, 2, 3]

-- Defining the given matrix A
def A : Matrix (Fin 3) (Fin 3) ℝ := !![
  ![-1, 1, 1],
  ![2, 0, 2],
  ![3, 3, 1]
]

-- The proof statement
theorem find_matrix_A : ∀ (v : Fin 3 → ℝ), (A.mulVec v) = (-2 • v + t) :=
by
  sorry

end find_matrix_A_l485_485642


namespace even_integers_in_pascals_triangle_top_15_rows_l485_485622

/-- Prove that the total number of even integers in the top 15 rows of Pascal's Triangle is exactly 90.
  Pascal's Triangle's elements, binomial(n,k), are even unless every binary digit of k 
  is present in n when both are expressed in binary. -/
theorem even_integers_in_pascals_triangle_top_15_rows : 
  ∑ n in Finset.range 15, ∑ k in Finset.range (n + 1), if (∀ i, (n.bits.get i) = 1 → (k.bits.get i) = 1) then 0 else 1 = 90 :=
sorry

end even_integers_in_pascals_triangle_top_15_rows_l485_485622


namespace num_distinct_circles_l485_485799

-- Define that S is a square
variable (S : Type) [square S]

-- State the theorem
theorem num_distinct_circles (vertices : set S) (diameters : set (S × S)) :
  (count_distinct_circles vertices diameters = 2) :=
sorry

end num_distinct_circles_l485_485799


namespace integers_representable_as_fraction_l485_485125

-- Define the statement that we need to prove
theorem integers_representable_as_fraction (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  let k := (x + y + z) ^ 2 / (x * y * z) in
  k ∈ {1, 2, 3, 4, 5, 6, 8, 9} :=
sorry

end integers_representable_as_fraction_l485_485125


namespace perfect_family_card_le_l485_485552

-- Define what it means for a set family to be perfect
def perfect_family (U : finset α) (F : finset (finset α)) : Prop :=
  ∀ (X1 X2 X3 : finset α), X1 ∈ F → X2 ∈ F → X3 ∈ F →
    (X1 \ X2) ∩ X3 = ∅ ∨ (X2 \ X1) ∩ X3 = ∅

theorem perfect_family_card_le (U : finset α) (F : finset (finset α)) 
  (hU : U.finite) (hF : perfect_family U F) : F.card ≤ U.card + 1 := 
  sorry

end perfect_family_card_le_l485_485552


namespace percent_less_than_mean_l485_485247

theorem percent_less_than_mean (s : List ℝ) 
  (h_len : s.length = 100)
  (h_sorted : s.sort = s) 
  (M Md : ℝ) 
  (h_between : ∃ l r, List.splitAt l.length s = (List.take l.length s, List.drop l.length s) ∧ List.splitAt r.length (List.drop l.length s) = (List.take r.length (List.drop l.length s), List.drop r.length (List.drop l.length s)) ∧ l.length = 50 ∧ r.length = 35 ∧ ∃ m n, ¬ List.mem M s ∧ ¬ List.mem Md s ∧ M = n.val ∧ Md = n.val ∧ (l < M ∧ M < Md) ∨ (Md < M ∧ M < r)) : 
  (∃ q, q = 65 ∨ q = 35) :=
by
  sorry

end percent_less_than_mean_l485_485247


namespace even_count_in_top_15_rows_l485_485607

def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

def count_even_in_row (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ k, is_even (binom n k)).card

def count_even_in_top_15_rows : ℕ :=
  (Finset.range 15).sum count_even_in_row

theorem even_count_in_top_15_rows :
  count_even_in_top_15_rows = "Sum of all identified evens" := by
sorry

end even_count_in_top_15_rows_l485_485607


namespace xy_value_l485_485050

noncomputable def compute_xy : ℝ × ℝ → ℝ
| (x, y) := x * y

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : compute_xy (x, y) = 35 / 12 := by
  sorry

end xy_value_l485_485050


namespace rhombus_segment_range_l485_485198

theorem rhombus_segment_range :
  ∀ {A B C D E F : ℝ^2}
    (h1 : dist A B = 2)
    (h2 : dist B C = 2)
    (h3 : dist C D = 2)
    (h4 : dist D A = 2)
    (h5 : dist B D = 2)
    (hE : E ∈ line_segment A D)
    (hF : F ∈ line_segment C D)
    (h_AE_CF : dist A E + dist C F = 2),
    √3 ≤ dist E F ∧ dist E F ≤ 2 := by
  sorry

end rhombus_segment_range_l485_485198


namespace find_k_no_solution_l485_485325

noncomputable def no_solution_equation (k : ℚ) : Prop :=
  ¬ ∃ x : ℚ, (kx / (x - 1)) - ((2 * k - 1) / (1 - x)) = 2

theorem find_k_no_solution :
  ∀ k : ℚ, no_solution_equation k → (k = 2 ∨ k = 1 / 3) :=
by
  sorry

end find_k_no_solution_l485_485325


namespace gcd_of_45_and_75_l485_485078

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l485_485078


namespace ratio_of_areas_l485_485913

theorem ratio_of_areas (s : ℝ) (h1 : s > 0) : 
  let small_square_area := s^2
  let total_small_squares_area := 4 * s^2
  let large_square_side_length := 4 * s
  let large_square_area := (4 * s)^2
  total_small_squares_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l485_485913


namespace reasoning_is_deductive_l485_485146

-- Define the conditions: properties of parallelogram and rectangle
structure Parallelogram where
  opposite_sides_parallel : Prop
  opposite_sides_equal : Prop

structure Rectangle extends Parallelogram

-- Define the question: Identifying the method of reasoning for the properties of a rectangle derived from a parallelogram
def method_of_reasoning (P : Type) [Parallelogram P] [Rectangle P] : Prop :=
  -- define what properties a rectangle should inherit
  P.opposite_sides_parallel ∧ P.opposite_sides_equal → True

-- Define the correct answer: Deductive reasoning
def deductive_reasoning : Prop := 
  method_of_reasoning Rectangle

-- The theorem stating the problem in c)
theorem reasoning_is_deductive : deductive_reasoning :=
  by
    sorry -- Proof omitted

end reasoning_is_deductive_l485_485146


namespace ron_chocolate_bar_cost_l485_485536

-- Definitions of the conditions given in the problem
def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def s'mores_needed_per_scout : ℕ := 2
def total_s'mores_needed : ℕ := scouts * s'mores_needed_per_scout
def chocolate_bars_needed : ℕ := total_s'mores_needed / sections_per_chocolate_bar
def total_cost_of_chocolate_bars : ℝ := chocolate_bars_needed * cost_per_chocolate_bar

-- Proving the question equals the answer given conditions
theorem ron_chocolate_bar_cost : total_cost_of_chocolate_bars = 15.00 := by
  sorry

end ron_chocolate_bar_cost_l485_485536


namespace sandy_friend_puppies_l485_485433

theorem sandy_friend_puppies (original_puppies friend_puppies final_puppies : ℕ)
    (h1 : original_puppies = 8) (h2 : final_puppies = 12) :
    friend_puppies = final_puppies - original_puppies := by
    sorry

end sandy_friend_puppies_l485_485433


namespace radius_of_inscribed_circle_l485_485256

variable (p q r : ℝ)

theorem radius_of_inscribed_circle (hp : p > 0) (hq : q > 0) (area_eq : q^2 = r * p) : r = q^2 / p :=
by
  sorry

end radius_of_inscribed_circle_l485_485256


namespace find_trapezoid_segment_length_l485_485858

variables (a b : ℝ) -- a is the length of AD and b is the length of BC

def trapezoid_segment (a b : ℝ) : ℝ :=
  (1 / 3) * (2 * a - b)

theorem find_trapezoid_segment_length :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  let MN := trapezoid_segment a b in
  MN = (1 / 3) * (2 * a - b) :=
by
  intros a b ha hb
  simp [trapezoid_segment]
  sorry

end find_trapezoid_segment_length_l485_485858


namespace find_number_l485_485586

theorem find_number (x : ℚ) (h : x / 11 + 156 = 178) : x = 242 :=
sorry

end find_number_l485_485586


namespace median_and_angle_bisector_perpendicular_l485_485860

theorem median_and_angle_bisector_perpendicular 
    (A B C A₁ B₁ C₁ O : Point)
    (h_median : is_median A A₁ C C₁)
    (h_angle_bisector : is_angle_bisector B B₁)
    (h_altitude : is_altitude C C₁)
    (h_intersect_at_O : intersects_at_single_point {A₁, B₁, C₁} O)
    (h_divide_ratio : divides_with_ratio O C C₁ 3 1) :
  is_perpendicular A A₁ B B₁ := 
sorry

end median_and_angle_bisector_perpendicular_l485_485860


namespace gcd_45_75_l485_485069

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485069


namespace find_p_plus_q_l485_485563

-- Defining basic conditions and known values
def width := 10
def length := 20
def area_triangle := 40

-- Defining the height of the box in terms of relatively prime numbers p and q
variables (p q : ℕ) (h : ℚ)
hypothesis h_eq : h = p / q
hypothesis rel_prime : Nat.Coprime p q
hypothesis pos_p : 0 < p
hypothesis pos_q : 0 < q

-- Given conditions about the triangle
def side1 := Real.sqrt ((width / 2) ^ 2 + (length / 2) ^ 2) -- 5√5
def side2 := Real.sqrt ((width / 2) ^ 2 + (h / 2) ^ 2)
def side3 := Real.sqrt ((length / 2) ^ 2 + (h / 2) ^ 2)

-- Conditions that need to be proven
def A := 16 * Real.sqrt 5 / 5
def hypotenuse := Real.sqrt ((5 * Real.sqrt 5 / 2) ^ 2 + (h / 2) ^ 2)
def target_h := 2 * Real.sqrt 131 / 5

-- Final goal
theorem find_p_plus_q : p + q = 133 :=
by
  -- Add your proof here
  sorry

end find_p_plus_q_l485_485563


namespace least_integer_value_l485_485649

theorem least_integer_value (x : ℤ) : 3 * abs x + 4 < 19 → x = -4 :=
by
  intro h
  sorry

end least_integer_value_l485_485649


namespace largest_20_supporting_X_l485_485405

-- Define the predicate for a number X being 20-supporting
def is_20_supporting (X : ℝ) : Prop :=
  ∀ (a : Fin 20 → ℝ), (∑ i, a i).toInt = ∑ i, a i →
  ∃ i, |a i - 0.5| ≥ X

-- Statement to prove the largest 20-supporting X, which is 1/40
theorem largest_20_supporting_X : ∃ X : ℝ, X = 1 / 40 ∧ is_20_supporting X := by
  sorry

end largest_20_supporting_X_l485_485405


namespace log_decreasing_l485_485036

open Real

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

def in_interval (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 4

theorem log_decreasing (a : ℝ) (h : ¬(a < 2)) :
  ¬is_decreasing (λ x, log (1/2) (2 - a * x)) 2 4 :=
sorry

end log_decreasing_l485_485036


namespace least_number_l485_485373

noncomputable def permutations := list.perm

noncomputable def alpha (n : ℕ) : list ℕ := list.range n
noncomputable def beta (n : ℕ) : list ℕ := alpha n
noncomputable def gamma (n : ℕ) : list ℕ := alpha n
noncomputable def delta (n : ℕ) : list ℕ := list.reverse (alpha n)

theorem least_number (n : ℕ) (h : n ≥ 2) :
  (∃ (α β γ δ : list ℕ), α.permutations ∧ β.permutations ∧ γ.permutations ∧ δ.permutations ∧ 
    list.sum (list.zip_with (*) α β) = (list.sum (list.zip_with (*) γ δ) * 19 / 10))
  → n = 28 :=
sorry

end least_number_l485_485373


namespace bisect_angle_l485_485531

variables (A B C D : Point)
variables [convex_quadrilateral A B C D]
variable (h1 : ∠ADC = 30)
variable (h2 : BD = AB + BC + CA)

theorem bisect_angle (ABC : Point) (h : convex_quadrilateral ABC D) (h₁ : ∠ADC = 30) (h₂ : BD = AB + BC + CA) :
  bisects (BD) (∠ABC) :=
by
  sorry

end bisect_angle_l485_485531


namespace geometric_ratio_theorem_l485_485197

theorem geometric_ratio_theorem 
  (A B C A1 A2 : Type) 
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited A1] [Inhabited A2] 
  (BC_side : ∀ x, x = A1 ∨ x = A2 → x ∈ [B, C])
  (angle_eq : ∀ (α : ℝ), ∠BAA1 = α ∧ ∠CAA2 = α) : 
  ∀ (AB AC BA1 A1C BA2 A2C : ℝ), 
  AB^2 / AC^2 = (BA1 / A1C) * (BA2 / A2C) :=
by
  sorry

end geometric_ratio_theorem_l485_485197


namespace earphone_cost_correct_l485_485164

-- Given conditions
def mean_expenditure : ℕ := 500

def expenditure_mon : ℕ := 450
def expenditure_tue : ℕ := 600
def expenditure_wed : ℕ := 400
def expenditure_thu : ℕ := 500
def expenditure_sat : ℕ := 550
def expenditure_sun : ℕ := 300

def pen_cost : ℕ := 30
def notebook_cost : ℕ := 50

-- Goal: cost of the earphone
def total_expenditure_week : ℕ := 7 * mean_expenditure
def expenditure_6days : ℕ := expenditure_mon + expenditure_tue + expenditure_wed + expenditure_thu + expenditure_sat + expenditure_sun
def expenditure_fri : ℕ := total_expenditure_week - expenditure_6days
def expenditure_fri_items : ℕ := pen_cost + notebook_cost
def earphone_cost : ℕ := expenditure_fri - expenditure_fri_items

theorem earphone_cost_correct :
  earphone_cost = 620 :=
by
  sorry

end earphone_cost_correct_l485_485164


namespace find_f_2017_5_l485_485692

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x : ℝ, f (x) = f (p - x)

noncomputable def f : ℝ → ℝ := sorry -- Definition will be provided when proving

theorem find_f_2017_5 (f : ℝ → ℝ)
    (h_odd : odd_function f)
    (h_periodic : periodic_function f 2)
    (h_def : ∀ x ∈ Icc (0 : ℝ) (1 : ℝ), f x = x^3) :
    f 2017.5 = 1/8 := 
begin
  sorry
end

end find_f_2017_5_l485_485692


namespace solve_problem_l485_485644

noncomputable def intersection_point (t : ℝ) : (ℝ × ℝ) := (1 + t, -5 + (real.sqrt 3) * t)

def line_eq (x y : ℝ) : Prop := x - y - 2 * real.sqrt 3 = 0

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem solve_problem :
  ∃ t : ℝ, (line_eq (1 + t) (-5 + (real.sqrt 3) * t)) ∧
  let P := intersection_point t in
  P = (1 + 2 * real.sqrt 3, 1) ∧
  distance P (1, -5) = 4 * real.sqrt 3 :=
by {
  sorry
}

end solve_problem_l485_485644


namespace tian_ji_probability_normal_tian_ji_probability_special_l485_485882

def relative_strengths (A a B b C c : ℕ) : Prop :=
  A > a ∧ a > B ∧ B > b ∧ b > C ∧ C > c

def normal_circumstances_pairings (events : List (Fin 6)) : List (Fin 6) :=
  [ (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1) ]

def special_circumstances_pairings (events : List (Fin 2)) : List (Fin 2) :=
  [(0, 1, 0), (0, 0, 1)]

theorem tian_ji_probability_normal (A a B b C c : ℕ) (h : relative_strengths A a B b C c) :
  (∑ event in normal_circumstances_pairings, if event = (1, 0, 1) then 1 else 0) / 6 = 1/6 :=
by sorry

theorem tian_ji_probability_special (A a B b C c : ℕ) (h : relative_strengths A a B b C c) :
  (∑ event in special_circumstances_pairings, if event = (0, 1, 0) then 1 else 0) / 2 = 1/2 :=
by sorry

end tian_ji_probability_normal_tian_ji_probability_special_l485_485882


namespace triangle_inequality_l485_485483

open Real

noncomputable def circle (R : ℝ) := {p : ℝ × ℝ // dist p (0, 0) = R}

variables {R1 R2 R3 R : ℝ}
variables (O A B C A1 B1 C1 : ℝ × ℝ)

axiom common_intersection : ∀ (k1 k2 k3 : circle (0, 0)), k1.1 = O ∧ k2.1 = O ∧ k3.1 = O
axiom triangle_points : ∀ {P Q : ℝ × ℝ}, dist O P = R1 ∧ dist O Q = R2 ∧ dist O O = R

def inside_triangle (O A B C : ℝ × ℝ) : Prop := -- Placeholder, defining "inside" might be complex
  sorry

def circumradius (A B C : ℝ × ℝ) : ℝ := -- Placeholder for function that calculates circumradius
  sorry

variables (α β γ : ℝ)
def ratios := (α = dist O A1 / dist A A1) ∧ (β = dist O B1 / dist B B1) ∧ (γ = dist O C1 / dist C C1)

theorem triangle_inequality (R1 R2 R3 : ℝ) (α β γ : ℝ)
  (hO_inside: inside_triangle O A B C)
  (h_ratios: ratios α β γ)
  (h_circumradius: circumradius A B C = R) :
  α * R1 + β * R2 + γ * R3 ≥ R :=
sorry

end triangle_inequality_l485_485483


namespace smallest_repunit_divisible_by_97_l485_485773

theorem smallest_repunit_divisible_by_97 :
  ∃ n : ℕ, (∃ d : ℤ, 10^n - 1 = 97 * 9 * d) ∧ (∀ m : ℕ, (∃ d : ℤ, 10^m - 1 = 97 * 9 * d) → n ≤ m) :=
by
  sorry

end smallest_repunit_divisible_by_97_l485_485773


namespace collinear_MNK_l485_485473

variables (A B C D E M N K P Q R S O: Type) 
variables (AB BC CD DA EC AD EA ED: Prop)

-- All sides touch the circle with center O at P, Q, R, S respectively
axiom sides_touch_circle : AB ∧ BC ∧ CD ∧ DA

-- Points of touch
axiom points_of_touch : (P: Prop) ∧ (Q: Prop) ∧ (R: Prop) ∧ (S: Prop)

-- Point E is on AB
axiom point_E_on_AB : AB

-- Lines EC, ED, EA intersect the respective lines at M, N, K
axiom line_EC_inter_AD_at_M : EC ∧ AD
axiom line_ED_inter_BC_at_N : ED ∧ BC
axiom line_EA_inter_CD_at_K : EA ∧ CD

theorem collinear_MNK : sides_touch_circle → 
                        points_of_touch → 
                        point_E_on_AB → 
                        line_EC_inter_AD_at_M → 
                        line_ED_inter_BC_at_N → 
                        line_EA_inter_CD_at_K → 
                        Collinear ℝ (M: ℝ) (N: ℝ) (K: ℝ) := 
sorry

end collinear_MNK_l485_485473


namespace angle_complement_supplement_l485_485219

theorem angle_complement_supplement (θ : ℝ) (h1 : 90 - θ = (1/3) * (180 - θ)) : θ = 45 :=
by
  sorry

end angle_complement_supplement_l485_485219


namespace min_val_m_l485_485392

theorem min_val_m (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h : 24 * m = n ^ 4) : m = 54 :=
sorry

end min_val_m_l485_485392


namespace gcd_45_75_l485_485101

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485101


namespace work_done_correct_l485_485897

open Real

noncomputable def work_done (a b g : ℝ) : ℝ :=
  -- Define potential function based on gravitational field
  let ϕ := fun (z : ℝ) => -g * z
  -- Calculate potential at point A and point B
  let ϕ_A := ϕ (2 * π * b)
  let ϕ_B := ϕ 0
  ϕ_B - ϕ_A

theorem work_done_correct (a b g : ℝ) : work_done a b g = 2 * π * g * b :=
by
  -- Simplify potential difference
  simp [work_done]
  unfold ϕ
  norm_num

end work_done_correct_l485_485897


namespace fireworks_display_l485_485178

def num_digits_year : ℕ := 4
def fireworks_per_digit : ℕ := 6
def regular_letters_phrase : ℕ := 12
def fireworks_per_regular_letter : ℕ := 5

def fireworks_H : ℕ := 8
def fireworks_E : ℕ := 7
def fireworks_L : ℕ := 6
def fireworks_O : ℕ := 9

def num_boxes : ℕ := 100
def fireworks_per_box : ℕ := 10

def total_fireworks : ℕ :=
  (num_digits_year * fireworks_per_digit) +
  (regular_letters_phrase * fireworks_per_regular_letter) +
  (fireworks_H + fireworks_E + 2 * fireworks_L + fireworks_O) + 
  (num_boxes * fireworks_per_box)

theorem fireworks_display : total_fireworks = 1120 := by
  sorry

end fireworks_display_l485_485178


namespace complex_numbers_sum_leq_l485_485595

noncomputable def main_statement (n : ℕ) (x y : ℂ) (x_i y_i : Fin n → ℂ) : Prop :=
  (∀ (i : Fin n), |x_i i| = 1) ∧ (∀ (i : Fin n), |y_i i| = 1) →
  let x := 1 / (n : ℂ) * ∑ i, x_i i
  let y := 1 / (n : ℂ) * ∑ i, y_i i
  let z_i := λ i, x * y_i i + y * x_i i - x_i i * y_i i
  (∑ i, |z_i i|) ≤ n

theorem complex_numbers_sum_leq (n : ℕ) (x y : ℂ) (x_i y_i : Fin n → ℂ) :
  main_statement n x y x_i y_i :=
by
  sorry

end complex_numbers_sum_leq_l485_485595


namespace circle_arcs_inequality_l485_485414

noncomputable def length_arc (X : Set ℝ) : ℝ := sorry

def set_rotation (A : Set ℝ) (j m : ℕ) : Set ℝ := sorry

variables (A B : Set ℝ) (m : ℕ)

theorem circle_arcs_inequality (hA : finite_non_intersecting_arcs A)
                              (hB : finite_non_intersecting_arcs B)
                              (hB_length : ∀ b ∈ B, length_arc b = π / (m:ℝ)) :
  ∃ k : ℕ, length_arc (set_rotation A k m ∩ B) ≥ (1 / (2 * Real.pi)) * length_arc A * length_arc B := 
sorry

end circle_arcs_inequality_l485_485414


namespace simplest_quadratic_radical_l485_485129

theorem simplest_quadratic_radical :
  (∀ a, (a = 17) → (¬ ∃ b, b^2 = a)) ↔ 
  (∀ a, (a = 12) → (∃ b, b^2 | a)) ∧
  (∀ a, (a = 24) → (∃ b, b^2 | a)) ∧
  (∀ a, (a = 1/3) → (∃ b, b^2 | a)) :=
begin
  sorry
end

end simplest_quadratic_radical_l485_485129


namespace price_increase_decrease_l485_485175

theorem price_increase_decrease (P : ℝ) (h : 0.84 * P = P * (1 - (x / 100)^2)) : x = 40 := by
  sorry

end price_increase_decrease_l485_485175


namespace find_x_l485_485523

theorem find_x : ∃ x : ℕ, x + 1 = 5 ∧ x = 4 :=
by
  sorry

end find_x_l485_485523


namespace simplify_and_evaluate_expression_l485_485438

theorem simplify_and_evaluate_expression (x y : ℝ) (hx : x = 4) (hy : y = -2) : 
  1 - (x - y) / (x + 2 * y) / ((x^2 - y^2) / (x^2 + 4 * x * y + 4 * y^2)) = 1 :=
by {
  rw [hx, hy],
  -- further simplifications, factorizations, and calculations would go here
  sorry
}

end simplify_and_evaluate_expression_l485_485438


namespace brick_surface_area_l485_485136

theorem brick_surface_area (l w h : ℝ) (hl : l = 10) (hw : w = 4) (hh : h = 3) : 
  2 * (l * w + l * h + w * h) = 164 := 
by
  sorry

end brick_surface_area_l485_485136


namespace expected_mean_of_weights_final_result_l485_485372

-- Define a permutation of (1, 2, ..., 13)
def is_permutation (lst : List ℕ) : Prop :=
  lst = List.range' 1 13

-- Determine the weight of a permutation as the number of adjacent swaps needed to sort it
def weight (lst : List ℕ) : ℕ :=
  sorry -- Implementation of calculating the weight

-- Define the problem conditions
def conditions (a : List ℕ) : Prop :=
  is_permutation a ∧ List.getD a 4 0 = 9

-- Define the calculation of the arithmetic mean of weights
def arithmetic_mean_weights (lst : List (List ℕ)) : ℚ :=
  if lst ≠ [] then
    (lst.map weight).sum / lst.length
  else
    0

-- Utilize arithmetic mean to calculate (100*m + n)
def solution (m n : ℕ) : ℕ :=
  100 * m + n

-- Prove the expected mean of weights with given condition
theorem expected_mean_of_weights :
  ∀ a, conditions a → arithmetic_mean_weights (filter conditions (List.permutations (List.range' 1 13))) = 137 / 3 :=
sorry

-- Given the mean is 137/3, find 100*m + n
theorem final_result : solution 137 3 = 13700 + 3 :=
begin
  simp [solution],
  norm_num,
end

end expected_mean_of_weights_final_result_l485_485372


namespace gcd_45_75_l485_485089

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485089


namespace find_x_l485_485893

variable {x : ℝ}

def work_time_P : ℝ := x + 8
def work_time_Q : ℝ := x + 2
def work_time_R : ℝ := 2 * x

theorem find_x : 
  (1 / (work_time_P) + 1 / (work_time_Q) + 1 / (work_time_R) = 1 / x) 
  → x = 2 :=
by
  sorry

end find_x_l485_485893


namespace least_tiles_needed_l485_485577

-- Define the conditions
def hallway_length_ft : ℕ := 18
def hallway_width_ft : ℕ := 6
def tile_side_in : ℕ := 6
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Translate conditions
def hallway_length_in := feet_to_inches hallway_length_ft
def hallway_width_in := feet_to_inches hallway_width_ft

-- Define the areas
def hallway_area : ℕ := hallway_length_in * hallway_width_in
def tile_area : ℕ := tile_side_in * tile_side_in

-- State the theorem to be proved
theorem least_tiles_needed :
  hallway_area / tile_area = 432 := 
sorry

end least_tiles_needed_l485_485577


namespace sqrt2_minus_1_power_eq_sqrt_diff_l485_485828

theorem sqrt2_minus_1_power_eq_sqrt_diff (n : ℤ) : 
  ∃ k : ℤ, (√2 - 1) ^ n = √(k + 1) - √k := 
by trivial -- the actual proof is omitted

end sqrt2_minus_1_power_eq_sqrt_diff_l485_485828


namespace impossible_to_get_100_pieces_l485_485949

/-- We start with 1 piece of paper. Each time a piece of paper is torn into 3 parts,
it increases the total number of pieces by 2.
Therefore, the number of pieces remains odd through any sequence of tears.
Prove that it is impossible to obtain exactly 100 pieces. -/
theorem impossible_to_get_100_pieces : 
  ∀ n, n = 1 ∨ (∃ k, n = 1 + 2 * k) → n ≠ 100 :=
by
  sorry

end impossible_to_get_100_pieces_l485_485949


namespace profit_percentage_l485_485521

theorem profit_percentage (investment_a investment_b profit total_received_a : ℝ)
                          (h_investment_a : investment_a = 2000)
                          (h_investment_b : investment_b = 3000)
                          (h_profit : profit = 9600)
                          (h_total_received_a : total_received_a = 4416) :
                          (∃ (p : ℝ), p = 6 ∧ ((p / 100) * profit) + ((2 / 5) * (profit - (p / 100) * profit)) = total_received_a) :=
by
  use 6
  split
  exact rfl
  sorry

end profit_percentage_l485_485521


namespace gcd_45_75_l485_485061

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485061


namespace problem_solution_l485_485707

variable {a b x y : ℝ}

-- Define the conditions as Lean assumptions
axiom cond1 : a * x + b * y = 3
axiom cond2 : a * x^2 + b * y^2 = 7
axiom cond3 : a * x^3 + b * y^3 = 16
axiom cond4 : a * x^4 + b * y^4 = 42

-- The main theorem statement: under these conditions, prove a * x^5 + b * y^5 = 99
theorem problem_solution : a * x^5 + b * y^5 = 99 := 
sorry -- proof omitted

end problem_solution_l485_485707


namespace conic_section_focus_l485_485696

theorem conic_section_focus {m : ℝ} (h_non_zero : m ≠ 0) (h_non_five : m ≠ 5)
  (h_focus : ∃ (x_focus y_focus : ℝ), (x_focus, y_focus) = (2, 0) 
  ∧ (x_focus = c ∧ x_focus^2 / 4 = 5 * (1 - c^2 / m))) : m = 9 := 
by
  sorry

end conic_section_focus_l485_485696


namespace value_of_x_plus_y_pow_2023_l485_485687

theorem value_of_x_plus_y_pow_2023 (x y : ℝ) (h : abs (x - 2) + abs (y + 3) = 0) : 
  (x + y) ^ 2023 = -1 := 
sorry

end value_of_x_plus_y_pow_2023_l485_485687


namespace perp_tangency_excircle_pass_single_point_l485_485833

open EuclideanGeometry

notation "∥" => Parallel

variables {A B C O K O' : Point}
variable [Triangle A B C]
variable (incenter_reflect : ∀ (P : Point), reflection P K ∈ [O, O'])

theorem perp_tangency_excircle_pass_single_point
  (O_incenter : isIncenter O A B C)
  (K_circumcenter : isCircumcenter K A B C)
  (O'_reflect : reflection O K = O')
  (tangency_reflection : ∀ (P Q : Point), isTangency P Q ∧ midpoint P Q ∈ [A, C, A, B, B, C] → reflection P (midpoint P Q) = Q)
  (perpendiculars : ∀ (P : Point), isTangency P ∧ isExcircle P → ⟂ P)
  :
  ∀ (P : Point), isTangency P ∧ isExcircle P → passesThrough P O' :=
sorry

end perp_tangency_excircle_pass_single_point_l485_485833


namespace find_x_perpendicular_l485_485312

/-- Given vectors a = ⟨-1, 2⟩ and b = ⟨1, x⟩, if a is perpendicular to (a + 2 * b),
    then x = -3/4. -/
theorem find_x_perpendicular
  (x : ℝ)
  (a : ℝ × ℝ := (-1, 2))
  (b : ℝ × ℝ := (1, x))
  (h : (a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0)) :
  x = -3 / 4 :=
sorry

end find_x_perpendicular_l485_485312


namespace mother_current_age_l485_485230

-- Define the conditions
def Eunji_current_age := 16
def Eunji_age_when_mother_35 := 8
def Mother_age_when_Eunji_8 := 35

-- State the theorem
theorem mother_current_age :
  let years_passed := Eunji_current_age - Eunji_age_when_mother_35 in
  let mother_age_now := Mother_age_when_Eunji_8 + years_passed in
  mother_age_now = 43 :=
by
  sorry

end mother_current_age_l485_485230


namespace volume_calculation_l485_485378

-- Definitions of the parameters
def a : ℝ := (4 * Real.pi) / 3
def b : ℝ := 14 * Real.pi
def c : ℝ := 118
def d : ℝ := 70

-- The proof statement
theorem volume_calculation : bc / ad = 5.9 :=
by
  let bc := b * c
  let ad := a * d
  have h : ad ≠ 0 := by sorry
  let ratio := bc / ad
  have : ratio = 5.9 := by sorry
  exact this

end volume_calculation_l485_485378


namespace circumscribed_center_on_Ox_axis_l485_485455

-- Define the quadratic equation
noncomputable def quadratic_eq (p x : ℝ) : ℝ := 2^p * x^2 + 5 * p * x - 2^(p^2)

-- Define the conditions for the problem
def intersects_Ox (p : ℝ) : Prop := ∃ x1 x2 : ℝ, quadratic_eq p x1 = 0 ∧ quadratic_eq p x2 = 0 ∧ x1 ≠ x2

def intersects_Oy (p : ℝ) : Prop := quadratic_eq p 0 = -2^(p^2)

-- Define the problem statement
theorem circumscribed_center_on_Ox_axis :
  (∀ p : ℝ, intersects_Ox p ∧ intersects_Oy p → (p = 0 ∨ p = -1)) →
  (0 + (-1) = -1) :=
sorry

end circumscribed_center_on_Ox_axis_l485_485455


namespace correct_operation_l485_485511

theorem correct_operation : (sqrt 3) * (sqrt 5) = sqrt 15 :=
by sorry

end correct_operation_l485_485511


namespace annalise_spending_l485_485582

theorem annalise_spending
  (n_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (cost_per_tissue : ℝ)
  (h1 : n_boxes = 10)
  (h2 : packs_per_box = 20)
  (h3 : tissues_per_pack = 100)
  (h4 : cost_per_tissue = 0.05) :
  n_boxes * packs_per_box * tissues_per_pack * cost_per_tissue = 1000 := 
  by
  sorry

end annalise_spending_l485_485582


namespace xz_squared_value_l485_485802

theorem xz_squared_value (x y z : ℝ) (h₁ : 3 * x * 5 * z = (4 * y)^2) (h₂ : (y^2 : ℝ) = (x^2 + z^2) / 2) :
  x^2 + z^2 = 16 := 
sorry

end xz_squared_value_l485_485802


namespace invitation_methods_l485_485962

/-- The number of ways to invite 6 out of 10 teachers to a seminar, 
    such that teachers A and B cannot attend at the same time, is 140. -/
theorem invitation_methods (teachers : Finset ℕ) (A B : ℕ) (h_size : teachers.card = 10)
    (h_AB : A ∈ teachers) (h_B : B ∈ teachers) :
    (Finset.card {s : Finset ℕ // s.card = 6 ∧ teachers.card = 10 ∧ (A ∈ s ↔ B ∉ s)}) = 140 := 
sorry

end invitation_methods_l485_485962


namespace even_count_in_pascal_triangle_l485_485608

-- Define the binomial coefficient as a function
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define a predicate to check if a number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Count the number of even integers in the top 15 rows of Pascal's Triangle
def count_even_pascal (rows : ℕ) : ℕ :=
  (Finset.range rows).sum (λ n => (Finset.range (n + 1)).count (λ k => is_even (binom n k)))

-- Statement of the problem
theorem even_count_in_pascal_triangle : count_even_pascal 15 = 84 :=
  by
    sorry

end even_count_in_pascal_triangle_l485_485608


namespace solution_of_equation_l485_485722

theorem solution_of_equation (a : ℝ) : (∃ x : ℝ, x = 4 ∧ (a * x - 3 = 4 * x + 1)) → a = 5 :=
by
  sorry

end solution_of_equation_l485_485722


namespace sum_remainder_l485_485124

theorem sum_remainder (a b c d : ℤ) (h1 : a % 53 = 33) (h2 : b % 53 = 11) 
                       (h3 : c % 53 = 49) (h4 : d % 53 = 2) :
  (a + b + c + d) % 53 = 42 :=
sorry

end sum_remainder_l485_485124


namespace symmetric_function_extreme_values_l485_485693

-- Define the given function and its symmetry condition
def h (x : ℝ) : ℝ := x + (1 / x) + 2

theorem symmetric_function (f : ℝ → ℝ) :
  (∀ x y, (0, 1) = (0, 1) → f x = x + (1 / x)) ↔
  (∀ x, f (-x) = 2 - h (-x)) :=
begin
  sorry
end

-- Finding the extreme values of f(x) = x + 1/x in the interval (0, 8]
theorem extreme_values :
  (∀ x, f x = x + (1 / x)) →
  (min (f 1) = 2) ∧ (max (f 8) = 65 / 8) :=
begin
  sorry
end

end symmetric_function_extreme_values_l485_485693


namespace calculate_diamond_value_l485_485249

def diamond (x y : ℝ) (h : x ≠ y) : ℝ := (x^2 + y^2) / (x - y)

theorem calculate_diamond_value : 
  ((diamond 3 1 (by norm_num)) = 5) → 
  ((diamond 5 2 (by norm_num)) = 29 / 3) →
  (diamond (diamond 3 1 (by norm_num)) 2 (by norm_num)) = 29 / 3 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end calculate_diamond_value_l485_485249


namespace odd_degree_polynomial_has_odd_number_of_real_roots_l485_485133

theorem odd_degree_polynomial_has_odd_number_of_real_roots (n : ℕ) (P : Polynomial ℝ) :
  P.degree = n ∧ n % 2 = 1 → ∃ (roots : Finset ℝ), (roots.sum (λ r, if P.eval r = 0 then 1 else 0)) % 2 = 1 :=
by
  sorry

end odd_degree_polynomial_has_odd_number_of_real_roots_l485_485133


namespace gcd_45_75_l485_485104

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end gcd_45_75_l485_485104


namespace min_area_sum_of_k_l485_485869

-- Definition of the points and condition
def pointA := (2, 9)
def pointB := (14, 18)
def pointC (k : ℤ) := (6, k)

-- Question: Prove that the minimum area sum of k values is 24
theorem min_area_sum_of_k : 
  let k_values := {k : ℤ | k = 11 ∨ k = 13} in
  ∑ k in k_values, k = 24 := by
sorry

end min_area_sum_of_k_l485_485869


namespace problem1_proof_problem2_proof_l485_485982

-- Definitions as per conditions from the problem statement
def mixedNumber1 : Real := 1 + 1/3
def mixedNumber2 : Real := 2 + 2/9
def fraction1 : Real := 6/5
def number1 : Real := 27
def number2 : Real := -3
def negativeSquareRoot : Real := (-2)^2
def cubeRoot : Real := (8 : Real)^(1/3 : Real)
def absoluteValue : Real := abs (3 - Real.pi)
def negativeCube : Real := (-1)^3

-- Problem 1: Mathematical equivalence proof statement
theorem problem1_proof : 
  mixedNumber1 - mixedNumber2 * fraction1 + number1 / number2 = -10 - 1/3 :=
by
  sorry

-- Problem 2: Mathematical equivalence proof statement
theorem problem2_proof : 
  Real.sqrt negativeSquareRoot - cubeRoot - absoluteValue + negativeCube = 2 - Real.pi :=
by
  sorry

end problem1_proof_problem2_proof_l485_485982


namespace intersect_heights_at_one_point_l485_485843

structure Triangle :=
  (A B C : Point)
  (is_acute : AcuteTriangle A B C)
  (is_isosceles : IsoscelesTriangle A B C)

structure Pyramid :=
  (S A B C : Point)
  (base_triangle : Triangle)
  (H : Point)
  (H_is_orthocenter : IsOrthocenter H A B C)
  (SH_is_height : IsHeight SH S H base_triangle)

-- Lean statement to prove the intersection of heights
theorem intersect_heights_at_one_point (S A B C H O : Point) (tri : Triangle A B C)
    (H_is_orthocenter : IsOrthocenter H A B C)
    (SH_is_height : IsHeight S H tri)
    (height_SA : IsHeight S A tri)
    (height_SB : IsHeight S B tri) :
  intersect_at_one_point (S, A, B, C) := sorry

end intersect_heights_at_one_point_l485_485843


namespace division_of_money_l485_485413

theorem division_of_money (total_amount : ℝ) (num_people : ℕ) (share : ℝ) :
  total_amount = 3.75 → num_people = 3 → share = total_amount / num_people → share = 1.25 :=
by
  intros h_total h_people h_share
  rw [h_total, h_people] at h_share
  exact h_share

end division_of_money_l485_485413


namespace joy_fourth_rod_count_is_26_l485_485370

noncomputable def valid_rods_count : ℕ :=
  let fourth_rod_range := (6, 34)
  let used_rods := {5, 10, 20}
  let total_rods := finset.range 41
  let valid_rods := total_rods.filter (λ x, x ∈ finset.Ico 6 35)
  valid_rods.card - (valid_rods.filter (λ x, x ∈ used_rods)).card

theorem joy_fourth_rod_count_is_26 :
  valid_rods_count = 26 :=
by
  sorry

end joy_fourth_rod_count_is_26_l485_485370


namespace problem_800_250_binary_representation_digits_difference_l485_485909

theorem problem_800_250_binary_representation_digits_difference :
  let num_digs := λ (n : ℕ), (Nat.log2 n) + 1
  in (num_digs 800 - num_digs 250) = 2 :=
by
  let num_digs := λ (n : ℕ), (Nat.log2 n) + 1
  sorry

end problem_800_250_binary_representation_digits_difference_l485_485909


namespace determine_a_l485_485731

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

theorem determine_a (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : a = 1 / 2 :=
by
  sorry

end determine_a_l485_485731


namespace correct_function_statements_l485_485384

def greatest_integer_le (a : ℝ) : ℤ := ⌊a⌋ 

def fract_part (x : ℝ) : ℝ := x - ⌊x⌋ 

def correct_statements : Prop :=
  let y := fract_part in
  -- Options provided as statements
  let statement1 := ∀ x : ℝ, 0 ≤ y x ∧ y x < 1 ∨ (x ≠ x → false)
  let statement2 := ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, y (x + T) = y x
  let statement3 := ∀ (k : ℤ) (x : ℝ), k ≤ x ∧ x < k + 1 → y x ≤ y (x + 1)
  let statement4 := ∀ x : ℝ, y (-x) = y x in
 
  -- Correct answer
  (¬ statement1) ∧ statement2 ∧ (¬ statement3) ∧ (¬ statement4)

theorem correct_function_statements : correct_statements :=
by
  sorry

end correct_function_statements_l485_485384


namespace cos_Z_l485_485358

namespace TriangleCosine

variables {X Y Z : ℝ}

def sin_X := 4 / 5
def cos_Y := 12 / 13

theorem cos_Z (h_triangle : ∠X + ∠Y + ∠Z = π) : cos Z = -16 / 65 := sorry

end TriangleCosine

end cos_Z_l485_485358


namespace totalPayment_l485_485887

def totalNumberOfTrees : Nat := 850
def pricePerDouglasFir : Nat := 300
def pricePerPonderosaPine : Nat := 225
def numberOfDouglasFirPurchased : Nat := 350
def numberOfPonderosaPinePurchased := totalNumberOfTrees - numberOfDouglasFirPurchased

def costDouglasFir := numberOfDouglasFirPurchased * pricePerDouglasFir
def costPonderosaPine := numberOfPonderosaPinePurchased * pricePerPonderosaPine

def totalCost := costDouglasFir + costPonderosaPine

theorem totalPayment : totalCost = 217500 := by
  sorry

end totalPayment_l485_485887


namespace number_of_true_propositions_l485_485471

theorem number_of_true_propositions (m : ℝ) (h : m > 0) :
  let Δ := 1 + 4 * m,
  let original_proposition := Δ > 0,
  let converse_proposition := ∀ x, (x^2 + x - m = 0) → m > 0,
  let inverse_proposition := ∃ x, (x^2 + x - m ≠ 0) ∧ m ≤ 0,
  let contrapositive_proposition := ∀ x, (x^2 + x - m ≠ 0) → m ≤ 0,
  (if original_proposition then 1 else 0) +
  (if converse_proposition then 1 else 0) +
  (if inverse_proposition then 1 else 0) +
  (if contrapositive_proposition then 1 else 0) = 2 :=
begin
  sorry
end

end number_of_true_propositions_l485_485471


namespace problem_statement_l485_485555

noncomputable def f : ℝ → ℝ :=
λ x, if (0 ≤ x ∧ x ≤ 1) then x^2 else
if (x ≥ 2) then f (2 - x) else
-f (-x)

theorem problem_statement :
  (∀ x, f (2 - x) = f x) →
  (∀ x, f (-x) = -f x) →
  (∀ x, (0 ≤ x ∧ x ≤ 1) → f x = x^2) →
  f (2019.5) = 1 / 4 :=
begin
  intros h1 h2 h3,
  -- proof skipped
  sorry
end

end problem_statement_l485_485555


namespace factorization_correct_l485_485968

theorem factorization_correct : 
  let optionA : Prop := (∀ x y : ℝ, (x - y) * (x + y) = x^2 - y^2)
  let optionB : Prop := (∀ a : ℝ, 4*a^2 - 4*a + 1 = 4*a*(a - 1) + 1)
  let optionC : Prop := (∀ x : ℝ, x^2 - 10^2 = (x + 3)*(x - 3) - 1)
  let optionD : Prop := (∀ m R r : ℝ, 2*m*R + 2*m*r = 2*m*(R + r))
  optionD := true := by sorry

end factorization_correct_l485_485968


namespace playground_width_l485_485901

open Nat

theorem playground_width (garden_width playground_length perimeter_garden : ℕ) (garden_area_eq_playground_area : Bool) :
  garden_width = 8 →
  playground_length = 16 →
  perimeter_garden = 64 →
  garden_area_eq_playground_area →
  ∃ (W : ℕ), W = 12 :=
by
  intros h_t1 h_t2 h_t3 h_t4
  sorry

end playground_width_l485_485901


namespace find_other_person_weight_l485_485449

theorem find_other_person_weight
    (initial_avg_weight : ℕ)
    (final_avg_weight : ℕ)
    (initial_group_size : ℕ)
    (new_person_weight : ℕ)
    (final_group_size : ℕ)
    (initial_total_weight : ℕ)
    (final_total_weight : ℕ)
    (new_total_weight : ℕ)
    (other_person_weight : ℕ) :
  initial_avg_weight = 48 →
  final_avg_weight = 51 →
  initial_group_size = 23 →
  final_group_size = 25 →
  new_person_weight = 93 →
  initial_total_weight = initial_group_size * initial_avg_weight →
  final_total_weight = final_group_size * final_avg_weight →
  new_total_weight = initial_total_weight + new_person_weight + other_person_weight →
  final_total_weight = new_total_weight →
  other_person_weight = 78 :=
by
  sorry

end find_other_person_weight_l485_485449


namespace g_g_g_25_l485_485809

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 18

theorem g_g_g_25 :
  g (g (g 25)) = 22 :=
by
  sorry

end g_g_g_25_l485_485809


namespace center_of_symmetry_sum_of_function_values_l485_485252

noncomputable def f : ℝ → ℝ := λ x, (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 + 2 * x + (1 / 12)

theorem center_of_symmetry :
  (∃ x0 y0, f' x0 = 0 ∧ (x0 = 1 / 2) ∧ (f x0 = 1) ∧ (x0, y0) = (1 / 2, 1)) := sorry

theorem sum_of_function_values :
  (∑ k in Finset.range 2016, f (k / 2017) = 2016) := sorry

end center_of_symmetry_sum_of_function_values_l485_485252


namespace find_dihedral_angle_and_distance_l485_485348

-- Conditions based on the problem
def Rectangle (A B C D : Point) : Prop :=
  AB = 3 ∧ AD = 4 ∧ isRectangle A B C D

theorem find_dihedral_angle_and_distance (A B C D B1 : Point)
(hRect: Rectangle A B C D)
(hFold: isFoldedAlongDiagonal A B C D B1):
(∃ angle : Real, angle = arctan (15 / 16)) ∧
(∃ dist : Real, dist = 10 * sqrt 34 / 17) :=
  sorry

end find_dihedral_angle_and_distance_l485_485348


namespace smallest_t_for_sine_polar_circle_l485_485225

theorem smallest_t_for_sine_polar_circle :
  ∃ t : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t) → ∃ r : ℝ, r = Real.sin θ) ∧
           (∀ θ : ℝ, (θ = t) → ∃ r : ℝ, r = 0) ∧
           (∀ t' : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t') → ∃ r : ℝ, r = Real.sin θ) →
                       (∀ θ : ℝ, (θ = t') → ∃ r : ℝ, r = 0) → t' ≥ t) :=
by
  sorry

end smallest_t_for_sine_polar_circle_l485_485225


namespace range_of_a_l485_485397

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + a > 0

def proposition_q (a : ℝ) : Prop :=
  a - 1 > 1

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ proposition_q a) ∧ ¬ (proposition_p a ∧ proposition_q a) ↔ 1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l485_485397


namespace no_2000_real_numbers_satisfying_conditions_l485_485227

theorem no_2000_real_numbers_satisfying_conditions :
  ¬ ∃ (a : Fin 2000 → ℝ),
    (∀ i, a i ≠ 0) ∧
    (∀ (s : Finset (Fin 2000)), s.card = 1000 →
      ∃ (perm : Fin 1000 ≃ Fin 1000), 
        ∀ (k : ℕ) (h1 : k ≤ 999) (h2 : k < 1000),
          polynomial.coeff (polynomial.monic_X_sub_C_coeffs (s.piecewise a 0)) k = a (perm ⟨k + 1000, by linarith⟩)) :=
by
  sorry

end no_2000_real_numbers_satisfying_conditions_l485_485227


namespace solution_interval_l485_485398

-- Define the differentiable function f over the interval (-∞, 0)
variable {f : ℝ → ℝ}
variable (hf : ∀ x < 0, HasDerivAt f (f' x) x)
variable (hx_cond : ∀ x < 0, 2 * f x + x * (deriv f x) > x^2)

-- Proof statement to show the solution interval
theorem solution_interval :
  {x : ℝ | (x + 2018)^2 * f (x + 2018) - 4 * f (-2) > 0} = {x | x < -2020} :=
sorry

end solution_interval_l485_485398


namespace cara_optimal_reroll_two_dice_probability_l485_485210

def probability_reroll_two_dice : ℚ :=
  -- Probability derived from Cara's optimal reroll decisions
  5 / 27

theorem cara_optimal_reroll_two_dice_probability :
  cara_probability_optimal_reroll_two_dice = 5 / 27 := by sorry

end cara_optimal_reroll_two_dice_probability_l485_485210


namespace problem_l485_485145

def f (x : ℝ) : ℝ := x^3 + 2 * x

theorem problem : f 5 + f (-5) = 0 := by
  sorry

end problem_l485_485145


namespace a_500_is_1173_l485_485739

-- Define the sequence a
def a : ℕ → ℤ
| 0       := 1007 -- Note: since Lean is zero-indexed, we interpret a_1 as a 0 in Lean
| 1       := 1008
| (n + 2) :=
  have h1 : a n + a (n + 1) + (a (n + 2)) = n + 1, from sorry, -- Corresponding to the given condition a_n + a_{n+1} + a_{n+2} = n
  sorry

-- The main statement we aim to prove
theorem a_500_is_1173 : a 499 = 1173 := 
by
  sorry

end a_500_is_1173_l485_485739


namespace example_problem_l485_485382

def greatest_integer_leq (x : Real) : Int :=
  floor x

def N (n : Nat) : Real :=
  Real.sqrt (n * (n + 4) * (n + 6) * (n + 10))

def satisfies_condition (n : Nat) : Prop :=
  (greatest_integer_leq (N n)) % 7 = 0

theorem example_problem (n : Nat) (hn : n = 2018) : satisfies_condition n :=
  sorry

end example_problem_l485_485382


namespace tangent_line_eq_extremum_range_always_geq_l485_485300

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * Real.log (x + 1)

-- Part (1)
theorem tangent_line_eq (a : ℝ) (x y : ℝ) (h : a = -2 ∧ x = 0 ∧ y = f 0 -2) :
  x + y - 1 = 0 :=
by sorry

-- Part (2)
theorem extremum_range (a : ℝ) : 
  (∃ x : ℝ, ∀ {h : x > -1}, x * (Real.exp x + a / (x + 1)) = 0) ↔ (a < 0) :=
by sorry

-- Part (3)
theorem always_geq (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 1 - Real.sin x) ↔ (a = -2) :=
by sorry

end tangent_line_eq_extremum_range_always_geq_l485_485300


namespace distance_from_A_to_C_l485_485191

open Complex

noncomputable def A : ℂ := 0
noncomputable def B : ℂ := 3000 * I
noncomputable def C : ℂ := 900 + 1200 * I

theorem distance_from_A_to_C : Complex.abs (C - A) = 1500 := 
by 
  have d : ℝ := Complex.abs (C - A)
  have d_eq : d = Real.sqrt ((900:ℝ)^2 + (1200:ℝ)^2) := by
    unfold Complex.abs
    rw [Complex.sub_zero]
    rw [sq_add_sq_eq (900:ℝ) (1200:ℝ)]
  calc
    d = Real.sqrt ((900:ℝ)^2 + (1200:ℝ)^2) : d_eq
    ... = Real.sqrt (810000 + 1440000) : by
      rw [sq 900, sq 1200]
    ... = Real.sqrt 2250000 : by norm_num
    ... = 1500 : by norm_num

end distance_from_A_to_C_l485_485191


namespace gcd_of_45_and_75_l485_485080

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l485_485080


namespace planes_perpendicular_l485_485283

def normal_vector_m := (1 : ℝ, -5 : ℝ, 2 : ℝ)
def normal_vector_n := (-3 : ℝ, 1 : ℝ, 4 : ℝ)

-- Problem statement: Prove that the planes are perpendicular given that their normal vectors have a dot product of zero.
theorem planes_perpendicular 
  (m : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) 
  (h1 : m = (1, -5, 2)) 
  (h2 : n = (-3, 1, 4)) : 
  m.1 * n.1 + m.2 * n.2 + m.3 * n.3 = 0 -> 
  m ⬝ n ⊥ n ∧ n ⬝ m ⊥ m :=
by 
  sorry  -- Proof not required

end planes_perpendicular_l485_485283


namespace BC_length_in_triangle_ABC_l485_485763

noncomputable def triangle_BC_length (AB AC AM : ℝ) (angleBAM : ℝ) : Prop :=
  ∃ (BC : ℝ), BC = 6 * real.sqrt 3 + 2 * real.sqrt 7 ∨ BC = 6 * real.sqrt 3 - 2 * real.sqrt 7

theorem BC_length_in_triangle_ABC :
  triangle_BC_length 6 10 4 (real.pi / 6) :=
begin
  sorry,
end

end BC_length_in_triangle_ABC_l485_485763


namespace max_distance_from_dock_l485_485202

theorem max_distance_from_dock {
  current_speed boat_speed : ℝ
  available_time : ℝ
  rowing_period rest_period : ℝ}
  (h_current_speed : current_speed = 1.4)
  (h_boat_speed : boat_speed = 3)
  (h_available_time : available_time = 2.75) -- 2.75 hours is 2 hours and 45 minutes
  (h_rowing_period : rowing_period = 0.5) -- 0.5 hours is 30 minutes
  (h_rest_period : rest_period = 0.25) -- 0.25 hours is 15 minutes
  : max_distance_from_dock = 1.7 := begin
  sorry
end

end max_distance_from_dock_l485_485202


namespace min_total_cost_of_tank_l485_485565

theorem min_total_cost_of_tank (V D c₁ c₂ : ℝ) (hV : V = 0.18) (hD : D = 0.5)
  (hc₁ : c₁ = 400) (hc₂ : c₂ = 100) : 
  ∃ x : ℝ, x > 0 ∧ (y = c₂*D*(2*x + 0.72/x) + c₁*0.36) ∧ y = 264 := 
sorry

end min_total_cost_of_tank_l485_485565


namespace abs_h_eq_17_div_4_l485_485589

theorem abs_h_eq_17_div_4 (h : ℚ) (sum_of_squares : 2 * (roots_sum_of_squares (QuadraticRoots (y^2 + 4*h*y - 2)) = 34)) : 
  |h| = 17 / 4 := 
sorry

end abs_h_eq_17_div_4_l485_485589


namespace sqrt_expression_evaluation_l485_485903

theorem sqrt_expression_evaluation :
  let x := Real.sqrt ((5 - 3 * Real.sqrt 3)^2) - Real.sqrt ((5 + 3 * Real.sqrt 3)^2) 
  in x = -6 * Real.sqrt 3 :=
by
  sorry

end sqrt_expression_evaluation_l485_485903


namespace count_of_grime_numbers_l485_485213

def is_of_form (a : ℕ) : Prop := ∃ n : ℕ, a = 10 * n + 1

def is_smaller_product (a : ℕ) : Prop :=
  ∃ r s : ℕ, is_of_form r ∧ is_of_form s ∧ r ≤ s ∧ r * s = a

def is_grime (a : ℕ) : Prop := is_of_form a ∧ ¬ is_smaller_product a

def numbers_of_interest : list ℕ := list.range' 11 (99 * 10) 10

noncomputable def count_grime_numbers : ℕ := 
  (numbers_of_interest.filter is_grime).length

theorem count_of_grime_numbers : count_grime_numbers = 87 := by
  sorry

end count_of_grime_numbers_l485_485213


namespace system_of_inequalities_l485_485012

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end system_of_inequalities_l485_485012


namespace map_distance_A_to_C_l485_485484

noncomputable def distance_map_to_real (distance_map : ℝ) : ℝ :=
  (distance_map / 0.2) * 2

theorem map_distance_A_to_C (d_AB d_BC : ℝ) (scale : ℝ) (B_between_AC : Prop) :
  d_AB = 10 → d_BC = 12 → scale = 0.2 → B_between_AC →
  distance_map_to_real (d_AB + d_BC) = 220 :=
by {
  intros h1 h2 h3 h4,
  unfold distance_map_to_real,
  simp [h1, h2, h3],
  sorry
}

end map_distance_A_to_C_l485_485484


namespace number_of_cows_l485_485168

-- Definitions
variable (H C : ℕ)
variable h1 : C = 5 * H
variable h2 : C + H = 168

-- Proof Statement
theorem number_of_cows : C = 140 :=
by
  -- each line needs to be filled according
  sorry

end number_of_cows_l485_485168


namespace water_consumption_150_litres_per_household_4_months_6000_litres_l485_485740

def number_of_households (household_water_use_per_month : ℕ) (water_supply : ℕ) (duration_months : ℕ) : ℕ :=
  water_supply / (household_water_use_per_month * duration_months)

theorem water_consumption_150_litres_per_household_4_months_6000_litres : 
  number_of_households 150 6000 4 = 10 :=
by
  sorry

end water_consumption_150_litres_per_household_4_months_6000_litres_l485_485740


namespace sin_2a_minus_cos2_a_l485_485277

variable (a : ℝ)
variable (h₁ : sin (Real.pi - a) = 4/5)
variable (h₂ : 0 < a ∧ a < Real.pi / 2)

theorem sin_2a_minus_cos2_a :
  sin (2 * a) - cos (a / 2) ^ 2 = 4 / 25 :=
  sorry

end sin_2a_minus_cos2_a_l485_485277


namespace gcd_of_45_and_75_l485_485082

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l485_485082


namespace factory_output_decrease_percentage_l485_485468

theorem factory_output_decrease_percentage :
  let original_output := 100
  let first_increase := original_output + (10% of original_output)
  let second_increase := first_increase + (40% of first_increase)
  let required_decrease := second_increase - original_output
  let percentage_decrease := (required_decrease / second_increase) * 100
  percentage_decrease ≈ 35.06 := by
  sorry

end factory_output_decrease_percentage_l485_485468


namespace greatest_perimeter_l485_485182

noncomputable def distance (x y : ℝ) : ℝ := 
  Real.sqrt (x ^ 2 + y ^ 2)

def perimeter (i : ℕ) : ℝ := 
  1 + distance 12 i + distance 12 (i + 1)

def max_perimeter (n : ℕ) : ℝ := 
  Nat.foldr (λ i acc, max (perimeter i) acc) 0 (List.range n)

theorem greatest_perimeter :
  max_perimeter 10 = 31.62 := 
by
  sorry

end greatest_perimeter_l485_485182


namespace jeffrey_steps_l485_485505

theorem jeffrey_steps (distance : ℕ) (forward_steps : ℕ) (backward_steps : ℕ) (effective_distance : ℤ) 
  (total_actual_steps : ℤ) (h1 : forward_steps = 3) (h2 : backward_steps = 2)
  (h3 : effective_distance = 1) 
  (h4 : total_distance : effective_distance * distance) :
  total_actual_steps = 330 :=
  by sorry

end jeffrey_steps_l485_485505


namespace gcd_45_75_l485_485093

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485093


namespace gcd_45_75_l485_485071

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485071


namespace gcd_45_75_l485_485092

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l485_485092


namespace sequence_and_arithmetic_seq_proof_l485_485268

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable (n : ℕ)

-- Define the sequence sum condition
axiom seq_sum_condition (n : ℕ) (hn : n > 0) : S_n n = 2 * (a_n n - 1)

-- Variable definitions for the second part
variable {k d : ℕ}
variable (d_cond : 3 < d ∧ d < 4)
variable (arithmetic_seq_sum : ℕ)
variable term_ak term_ak1 : ℕ -- To represent a_k and a_{k+1}

-- Define general term formula a_n = 2^n
def general_term_formula (n : ℕ) : ℕ :=
  2 ^ n

-- Formal theorem statement
theorem sequence_and_arithmetic_seq_proof :
  (∀ n > 0, general_term_formula n = a_n n) ∧
  ∃ k : ℕ, k = 4 ∧
  ∃ d : ℕ, 3 < d ∧ d < 4 ∧
  ∃ arithmetic_seq_sum : ℕ, arithmetic_seq_sum = 144 :=
by
  sorry

end sequence_and_arithmetic_seq_proof_l485_485268


namespace fraction_of_income_from_tips_l485_485181

variable (S T I : ℝ)

/- Definition of the conditions -/
def tips_condition : Prop := T = (3 / 4) * S
def income_condition : Prop := I = S + T

/- The proof problem statement, asserting the desired result -/
theorem fraction_of_income_from_tips (h1 : tips_condition S T) (h2 : income_condition S T I) : T / I = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l485_485181


namespace varja_miron_game_l485_485899

open Nat

def isPrimeOrPowerOfTwo (n : ℕ) : Prop :=
  Nat.Prime n ∨ ∃ k : ℕ, n = 2^k

theorem varja_miron_game (n : ℕ) : (Nat.Prime n ∨ ∃ k : ℕ, n = 2^k) ∨
  (¬ (Nat.Prime n ∨ ∃ k : ℕ, n = 2^k)) :=
begin
  sorry
end

end varja_miron_game_l485_485899


namespace range_of_a_div_b_l485_485748

/-- In acute triangle ABC, given A = 2B, find the range of a/b. -/
theorem range_of_a_div_b (A B C a b c : ℝ) (h1 : 0 < A ∧ A < π/2) 
  (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2) 
  (h4 : A + B + C = π) (h5 : A = 2 * B) (h6 : a / b = 2 * cos B) : 
  (sqrt 2) < a / b ∧ a / b < (sqrt 3) :=
sorry

end range_of_a_div_b_l485_485748


namespace least_integer_value_of_x_l485_485647

theorem least_integer_value_of_x (x : ℤ) (h : 3 * |x| + 4 < 19) : x = -4 :=
by sorry

end least_integer_value_of_x_l485_485647
