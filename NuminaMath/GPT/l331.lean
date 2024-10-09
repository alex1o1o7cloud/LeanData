import Mathlib

namespace count_integers_with_factors_12_and_7_l331_33123

theorem count_integers_with_factors_12_and_7 :
  ∃ k : ℕ, k = 4 ∧
    (∀ (n : ℕ), 500 ≤ n ∧ n ≤ 800 ∧ 12 ∣ n ∧ 7 ∣ n ↔ (84 ∣ n ∧
      n = 504 ∨ n = 588 ∨ n = 672 ∨ n = 756)) :=
sorry

end count_integers_with_factors_12_and_7_l331_33123


namespace factorize_expression_l331_33139

-- Define the variables a and b
variables (a b : ℝ)

-- State the theorem
theorem factorize_expression : 5*a^2*b - 20*b^3 = 5*b*(a + 2*b)*(a - 2*b) :=
by sorry

end factorize_expression_l331_33139


namespace arithmetic_sequence_problem_l331_33154

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) -- condition for arithmetic sequence
  (h_condition : a 3 + a 5 + a 7 + a 9 + a 11 = 100) : 
  3 * a 9 - a 13 = 40 :=
sorry

end arithmetic_sequence_problem_l331_33154


namespace completing_square_16x2_32x_512_eq_33_l331_33131

theorem completing_square_16x2_32x_512_eq_33:
  (∃ p q : ℝ, (16 * x ^ 2 + 32 * x - 512 = 0) → (x + p) ^ 2 = q ∧ q = 33) :=
by
  sorry

end completing_square_16x2_32x_512_eq_33_l331_33131


namespace arithmetic_sequence_b1_l331_33109

theorem arithmetic_sequence_b1 
  (b : ℕ → ℝ) 
  (U : ℕ → ℝ)
  (U2023 : ℝ) 
  (b2023 : ℝ)
  (hb2023 : b 2023 = b 1 + 2022 * (b 2 - b 1))
  (hU2023 : U 2023 = 2023 * (b 1 + 1011 * (b 2 - b 1))) 
  (hUn : ∀ n, U n = (n * (2 * b 1 + (n - 1) * (b 2 - b 1)) / 2)) :
  b 1 = (U 2023 - 2023 * b 2023) / 2023 :=
by
  sorry

end arithmetic_sequence_b1_l331_33109


namespace imaginary_part_of_z_l331_33138

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : I * z = 1 + I) : z.im = -1 := 
sorry

end imaginary_part_of_z_l331_33138


namespace average_sales_is_167_5_l331_33115

def sales_january : ℝ := 150
def sales_february : ℝ := 90
def sales_march : ℝ := 1.5 * sales_february
def sales_april : ℝ := 180
def sales_may : ℝ := 210
def sales_june : ℝ := 240
def total_sales : ℝ := sales_january + sales_february + sales_march + sales_april + sales_may + sales_june
def number_of_months : ℝ := 6

theorem average_sales_is_167_5 :
  total_sales / number_of_months = 167.5 :=
sorry

end average_sales_is_167_5_l331_33115


namespace sum_of_possible_values_of_N_l331_33186

theorem sum_of_possible_values_of_N (N : ℤ) : 
  (N * (N - 8) = 16) -> (∃ a b, N^2 - 8 * N - 16 = 0 ∧ (a + b = 8)) :=
sorry

end sum_of_possible_values_of_N_l331_33186


namespace distance_between_QY_l331_33104

theorem distance_between_QY 
  (m_rate : ℕ) (j_rate : ℕ) (j_distance : ℕ) (headstart : ℕ) 
  (t : ℕ) 
  (h1 : m_rate = 3) 
  (h2 : j_rate = 4) 
  (h3 : j_distance = 24) 
  (h4 : headstart = 1) 
  (h5 : j_distance = j_rate * (t - headstart)) 
  (h6 : t = 7) 
  (distance_m : ℕ := m_rate * t) 
  (distance_j : ℕ := j_distance) :
  distance_j + distance_m = 45 :=
by 
  sorry

end distance_between_QY_l331_33104


namespace complex_square_l331_33184

theorem complex_square (i : ℂ) (h : i^2 = -1) : (1 - i)^2 = -2 * i :=
by
  sorry

end complex_square_l331_33184


namespace proof_correct_chemical_information_l331_33192

def chemical_formula_starch : String := "(C_{6}H_{10}O_{5})_{n}"
def structural_formula_glycine : String := "H_{2}N-CH_{2}-COOH"
def element_in_glass_ceramics_cement : String := "Si"
def elements_cause_red_tide : List String := ["N", "P"]

theorem proof_correct_chemical_information :
  chemical_formula_starch = "(C_{6}H_{10}O_{5})_{n}" ∧
  structural_formula_glycine = "H_{2}N-CH_{2}-COOH" ∧
  element_in_glass_ceramics_cement = "Si" ∧
  elements_cause_red_tide = ["N", "P"] :=
by
  sorry

end proof_correct_chemical_information_l331_33192


namespace math_proof_l331_33127

-- Definitions
def U := Set ℝ
def A : Set ℝ := {x | x ≥ 3}
def B : Set ℝ := {x | x^2 - 8*x + 7 ≤ 0}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem
theorem math_proof (a : ℝ) :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x | x ≥ 1}) ∧
  (C a ∪ A = A → a ≥ 4) :=
by
  sorry

end math_proof_l331_33127


namespace seating_arrangements_l331_33132

def total_seats_front := 11
def total_seats_back := 12
def middle_seats_front := 3

def number_of_arrangements := 334

theorem seating_arrangements: 
  (total_seats_front - middle_seats_front) * (total_seats_front - middle_seats_front - 1) / 2 +
  (total_seats_back * (total_seats_back - 1)) / 2 +
  (total_seats_front - middle_seats_front) * total_seats_back +
  total_seats_back * (total_seats_front - middle_seats_front) = number_of_arrangements := 
sorry

end seating_arrangements_l331_33132


namespace cos_alpha_value_cos_2alpha_value_l331_33110

noncomputable def x : ℤ := -3
noncomputable def y : ℤ := 4
noncomputable def r : ℝ := Real.sqrt (x^2 + y^2)
noncomputable def cos_alpha : ℝ := x / r
noncomputable def cos_2alpha : ℝ := 2 * cos_alpha^2 - 1

theorem cos_alpha_value : cos_alpha = -3 / 5 := by
  sorry

theorem cos_2alpha_value : cos_2alpha = -7 / 25 := by
  sorry

end cos_alpha_value_cos_2alpha_value_l331_33110


namespace find_f_neg_9_over_2_l331_33176

noncomputable def f : ℝ → ℝ
| x => if 0 ≤ x ∧ x ≤ 1 then 2^x else sorry

theorem find_f_neg_9_over_2
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hf_periodic : ∀ x : ℝ, f (x + 2) = f x)
  (hf_definition : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = 2^x) :
  f (-9 / 2) = Real.sqrt 2 := by
  sorry

end find_f_neg_9_over_2_l331_33176


namespace diana_owes_l331_33163

-- Define the conditions
def initial_charge : ℝ := 60
def annual_interest_rate : ℝ := 0.06
def time_in_years : ℝ := 1

-- Define the simple interest calculation
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

-- Define the total amount owed calculation
def total_amount_owed (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

-- State the theorem: Diana will owe $63.60 after one year
theorem diana_owes : total_amount_owed initial_charge (simple_interest initial_charge annual_interest_rate time_in_years) = 63.60 :=
by sorry

end diana_owes_l331_33163


namespace maximum_value_at_2001_l331_33197
noncomputable def a_n (n : ℕ) : ℝ := n^2 / (1.001^n)

theorem maximum_value_at_2001 : ∃ n : ℕ, n = 2001 ∧ ∀ k : ℕ, a_n k ≤ a_n 2001 := by
  sorry

end maximum_value_at_2001_l331_33197


namespace sandwiches_bought_is_2_l331_33144

-- The given costs and totals
def sandwich_cost : ℝ := 3.49
def soda_cost : ℝ := 0.87
def total_cost : ℝ := 10.46
def sodas_bought : ℕ := 4

-- We need to prove that the number of sandwiches bought, S, is 2
theorem sandwiches_bought_is_2 (S : ℕ) :
  sandwich_cost * S + soda_cost * sodas_bought = total_cost → S = 2 :=
by
  intros h
  sorry

end sandwiches_bought_is_2_l331_33144


namespace cylinder_volume_ratio_l331_33155

theorem cylinder_volume_ratio (h1 h2 r1 r2 V1 V2 : ℝ)
  (h1_eq : h1 = 9)
  (h2_eq : h2 = 6)
  (circumference1_eq : 2 * π * r1 = 6)
  (circumference2_eq : 2 * π * r2 = 9)
  (V1_eq : V1 = π * r1^2 * h1)
  (V2_eq : V2 = π * r2^2 * h2)
  (V1_calculated : V1 = 81 / π)
  (V2_calculated : V2 = 243 / (4 * π)) :
  (max V1 V2) / (min V1 V2) = 3 / 4 :=
by
  sorry

end cylinder_volume_ratio_l331_33155


namespace probability_of_all_heads_or_tails_l331_33159

theorem probability_of_all_heads_or_tails :
  let possible_outcomes := 256
  let favorable_outcomes := 2
  favorable_outcomes / possible_outcomes = 1 / 128 := by
  sorry

end probability_of_all_heads_or_tails_l331_33159


namespace domain_ln_x_minus_x_sq_l331_33156

noncomputable def f (x : ℝ) : ℝ := Real.log (x - x^2)

theorem domain_ln_x_minus_x_sq : { x : ℝ | x - x^2 > 0 } = { x : ℝ | 0 < x ∧ x < 1 } :=
by {
  -- These are placeholders for conditions needed in the proof
  sorry
}

end domain_ln_x_minus_x_sq_l331_33156


namespace bakery_new_cakes_count_l331_33182

def cakes_sold := 91
def more_cakes_bought := 63

theorem bakery_new_cakes_count : (91 + 63) = 154 :=
by
  sorry

end bakery_new_cakes_count_l331_33182


namespace solve_equation_integers_l331_33107

theorem solve_equation_integers :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (1 + 1 / (x : ℚ)) * (1 + 1 / (y : ℚ)) * (1 + 1 / (z : ℚ)) = 2 ∧
  (x = 2 ∧ y = 4 ∧ z = 15 ∨
   x = 2 ∧ y = 5 ∧ z = 9 ∨
   x = 2 ∧ y = 6 ∧ z = 7 ∨
   x = 3 ∧ y = 4 ∧ z = 5 ∨
   x = 3 ∧ y = 3 ∧ z = 8 ∨
   x = 2 ∧ y = 15 ∧ z = 4 ∨
   x = 2 ∧ y = 9 ∧ z = 5 ∨
   x = 2 ∧ y = 7 ∧ z = 6 ∨
   x = 3 ∧ y = 5 ∧ z = 4 ∨
   x = 3 ∧ y = 8 ∧ z = 3) ∧
  (y = 2 ∧ x = 4 ∧ z = 15 ∨
   y = 2 ∧ x = 5 ∧ z = 9 ∨
   y = 2 ∧ x = 6 ∧ z = 7 ∨
   y = 3 ∧ x = 4 ∧ z = 5 ∨
   y = 3 ∧ x = 3 ∧ z = 8 ∨
   y = 15 ∧ x = 4 ∧ z = 2 ∨
   y = 9 ∧ x = 5 ∧ z = 2 ∨
   y = 7 ∧ x = 6 ∧ z = 2 ∨
   y = 5 ∧ x = 4 ∧ z = 3 ∨
   y = 8 ∧ x = 3 ∧ z = 3) ∧
  (z = 2 ∧ x = 4 ∧ y = 15 ∨
   z = 2 ∧ x = 5 ∧ y = 9 ∨
   z = 2 ∧ x = 6 ∧ y = 7 ∨
   z = 3 ∧ x = 4 ∧ y = 5 ∨
   z = 3 ∧ x = 3 ∧ y = 8 ∨
   z = 15 ∧ x = 4 ∧ y = 2 ∨
   z = 9 ∧ x = 5 ∧ y = 2 ∨
   z = 7 ∧ x = 6 ∧ y = 2 ∨
   z = 5 ∧ x = 4 ∧ y = 3 ∨
   z = 8 ∧ x = 3 ∧ y = 3)
:= sorry

end solve_equation_integers_l331_33107


namespace triangle_area_is_integer_l331_33160

theorem triangle_area_is_integer (x1 x2 x3 y1 y2 y3 : ℤ) 
  (hx_even : (x1 + x2 + x3) % 2 = 0) 
  (hy_even : (y1 + y2 + y3) % 2 = 0) : 
  ∃ k : ℤ, 
    abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) = 2 * k := 
sorry

end triangle_area_is_integer_l331_33160


namespace vasya_number_l331_33129

theorem vasya_number (a b c : ℕ) (h1 : 100 ≤ 100*a + 10*b + c) (h2 : 100*a + 10*b + c < 1000) 
  (h3 : a + c = 1) (h4 : a * b = 4) (h5 : a ≠ 0) : 100*a + 10*b + c = 140 :=
by
  sorry

end vasya_number_l331_33129


namespace probability_of_selection_l331_33172

noncomputable def probability_selected (total_students : ℕ) (excluded_students : ℕ) (selected_students : ℕ) : ℚ :=
  selected_students / (total_students - excluded_students)

theorem probability_of_selection :
  probability_selected 2008 8 50 = 25 / 1004 :=
by
  sorry

end probability_of_selection_l331_33172


namespace smallest_N_value_l331_33188

theorem smallest_N_value (a b c d : ℕ)
  (h1 : gcd a b = 1 ∧ gcd a c = 2 ∧ gcd a d = 4 ∧ gcd b c = 5 ∧ gcd b d = 3 ∧ gcd c d = N)
  (h2 : N > 5) : N = 14 := sorry

end smallest_N_value_l331_33188


namespace find_p_l331_33148

theorem find_p
  (p : ℝ)
  (h1 : ∃ (x y : ℝ), p * (x^2 - y^2) = (p^2 - 1) * x * y ∧ |x - 1| + |y| = 1)
  (h2 : ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
         x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
         p * (x₁^2 - y₁^2) = (p^2 - 1) * x₁ * y₁ ∧ |x₁ - 1| + |y₁| = 1 ∧
         p * (x₂^2 - y₂^2) = (p^2 - 1) * x₂ * y₂ ∧ |x₂ - 1| + |y₂| = 1 ∧
         p * (x₃^2 - y₃^2) = (p^2 - 1) * x₃ * y₃ ∧ |x₃ - 1| + |y₃| = 1) :
  p = 1 ∨ p = -1 :=
by sorry

end find_p_l331_33148


namespace hyperbola_asymptotes_l331_33106

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x^2 - 4 * y^2 = 1) → (x = 2 * y ∨ x = -2 * y) :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l331_33106


namespace circle_equation_l331_33130

theorem circle_equation (x y : ℝ) :
  (∀ (C P : ℝ × ℝ), C = (8, -3) ∧ P = (5, 1) →
    ∃ R : ℝ, (x - 8)^2 + (y + 3)^2 = R^2 ∧ R^2 = 25) :=
sorry

end circle_equation_l331_33130


namespace prove_sequences_and_sum_l331_33166

theorem prove_sequences_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (a 1 = 5) →
  (a 2 = 2) →
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) →
  (∀ n, ∃ r1, (a (n + 1) - 2 * a n) = (a 2 - 2 * a 1) * r1 ^ n) ∧
  (∀ n, ∃ r2, (a (n + 1) - (1 / 2) * a n) = (a 2 - (1 / 2) * a 1) * r2 ^ n) ∧
  (∀ n, S n = (4 * n) / 3 + (4 ^ n) / 36 - 1 / 36) :=
by
  sorry

end prove_sequences_and_sum_l331_33166


namespace lucy_total_fish_l331_33180

theorem lucy_total_fish (current fish_needed : ℕ) (h1 : current = 212) (h2 : fish_needed = 68) : 
  current + fish_needed = 280 := 
by
  sorry

end lucy_total_fish_l331_33180


namespace area_quadrilateral_EFGH_l331_33142

-- Define the rectangles ABCD and XYZR
def area_rectangle_ABCD : ℝ := 60 
def area_rectangle_XYZR : ℝ := 4

-- Define what needs to be proven: the area of quadrilateral EFGH
theorem area_quadrilateral_EFGH (a b c d : ℝ) :
  (area_rectangle_ABCD = area_rectangle_XYZR + 2 * (a + b + c + d)) →
  (a + b + c + d = 28) →
  (area_rectangle_XYZR = 4) →
  (area_rectangle_ABCD = 60) →
  (a + b + c + d + area_rectangle_XYZR = 32) :=
by
  intros h1 h2 h3 h4
  sorry

end area_quadrilateral_EFGH_l331_33142


namespace thalassa_population_2050_l331_33158

def population_in_2000 : ℕ := 250

def population_doubling_interval : ℕ := 20

def population_linear_increase_interval : ℕ := 10

def linear_increase_amount : ℕ := 500

noncomputable def population_in_2050 : ℕ :=
  let double1 := population_in_2000 * 2
  let double2 := double1 * 2
  double2 + linear_increase_amount

theorem thalassa_population_2050 : population_in_2050 = 1500 := by
  sorry

end thalassa_population_2050_l331_33158


namespace solution_set_of_inequality_l331_33121

theorem solution_set_of_inequality :
  {x : ℝ | (x-2)*(3-x) > 0} = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end solution_set_of_inequality_l331_33121


namespace number_of_bananas_l331_33191

-- Define costs as constants
def cost_per_banana := 1
def cost_per_apple := 2
def cost_per_twelve_strawberries := 4
def cost_per_avocado := 3
def cost_per_half_bunch_grapes := 2
def total_cost := 28

-- Define quantities as constants
def number_of_apples := 3
def number_of_strawberries := 24
def number_of_avocados := 2
def number_of_half_bunches_grapes := 2

-- Define calculated costs
def cost_of_apples := number_of_apples * cost_per_apple
def cost_of_strawberries := (number_of_strawberries / 12) * cost_per_twelve_strawberries
def cost_of_avocados := number_of_avocados * cost_per_avocado
def cost_of_grapes := number_of_half_bunches_grapes * cost_per_half_bunch_grapes

-- Define total cost of other fruits
def total_cost_of_other_fruits := cost_of_apples + cost_of_strawberries + cost_of_avocados + cost_of_grapes

-- Define the remaining cost for bananas
def remaining_cost := total_cost - total_cost_of_other_fruits

-- Prove the number of bananas
theorem number_of_bananas : remaining_cost / cost_per_banana = 4 :=
by
  -- This is a placeholder to indicate a non-implemented proof
  sorry

end number_of_bananas_l331_33191


namespace cos2alpha_plus_sin2alpha_l331_33124

theorem cos2alpha_plus_sin2alpha (α : Real) (h : Real.tan (Real.pi + α) = 2) : 
  Real.cos (2 * α) + Real.sin (2 * α) = 1 / 5 :=
sorry

end cos2alpha_plus_sin2alpha_l331_33124


namespace find_coordinates_of_P_l331_33140

-- Definitions based on the conditions:
-- Point P has coordinates (a, 2a-1) and lies on the line y = x.

def lies_on_bisector (a : ℝ) : Prop :=
  (2 * a - 1) = a -- This is derived from the line y = x for the given point coordinates.

-- The final statement to prove:
theorem find_coordinates_of_P (a : ℝ) (P : ℝ × ℝ) (h1 : P = (a, 2 * a - 1)) (h2 : lies_on_bisector a) :
  P = (1, 1) :=
by
  -- Proof steps are omitted and replaced with sorry.
  sorry

end find_coordinates_of_P_l331_33140


namespace solve_quadratic_inequality_l331_33102

theorem solve_quadratic_inequality (x : ℝ) :
  (-3 * x^2 + 8 * x + 5 > 0) ↔ (x < -1 / 3) :=
by
  sorry

end solve_quadratic_inequality_l331_33102


namespace tony_graduate_degree_years_l331_33112

-- Define the years spent for each degree and the total time
def D1 := 4 -- years for the first degree in science
def D2 := 4 -- years for each of the two additional degrees
def T := 14 -- total years spent in school
def G := 2 -- years spent for the graduate degree in physics

-- Theorem: Given the conditions, prove that Tony spent 2 years on his graduate degree in physics
theorem tony_graduate_degree_years : 
  D1 + 2 * D2 + G = T :=
by
  sorry

end tony_graduate_degree_years_l331_33112


namespace total_rainfall_November_l331_33122

def rain_first_15_days : ℕ := 4

def days_first_15 : ℕ := 15

def rain_last_15_days : ℕ := 2 * rain_first_15_days

def days_last_15 : ℕ := 15

def total_rainfall : ℕ := 
  (rain_first_15_days * days_first_15) + (rain_last_15_days * days_last_15)

theorem total_rainfall_November : total_rainfall = 180 := by
  sorry

end total_rainfall_November_l331_33122


namespace find_AX_length_l331_33175

noncomputable def AX_length (AC BC BX : ℕ) : ℚ :=
AC * (BX / BC)

theorem find_AX_length :
  let AC := 25
  let BC := 35
  let BX := 30
  AX_length AC BC BX = 150 / 7 :=
by
  -- proof is omitted using 'sorry'
  sorry

end find_AX_length_l331_33175


namespace power_of_two_representation_l331_33151

/-- Prove that any number 2^n, where n = 3,4,5,..., can be represented 
as 7x^2 + y^2 where x and y are odd numbers. -/
theorem power_of_two_representation (n : ℕ) (hn : n ≥ 3) : 
  ∃ x y : ℤ, (2*x ≠ 0 ∧ 2*y ≠ 0) ∧ 2^n = 7 * x^2 + y^2 :=
by
  sorry

end power_of_two_representation_l331_33151


namespace cos_sin_15_deg_l331_33170

theorem cos_sin_15_deg :
  400 * (Real.cos (15 * Real.pi / 180))^5 +  (Real.sin (15 * Real.pi / 180))^5 / (Real.cos (15 * Real.pi / 180) + Real.sin (15 * Real.pi / 180)) = 100 := 
sorry

end cos_sin_15_deg_l331_33170


namespace shopkeeper_profit_percentage_l331_33137

theorem shopkeeper_profit_percentage (C : ℝ) (hC : C > 0) :
  let selling_price := 12 * C
  let cost_price := 10 * C
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 :=
by
  sorry

end shopkeeper_profit_percentage_l331_33137


namespace team_members_run_distance_l331_33187

-- Define the given conditions
def total_distance : ℕ := 150
def members : ℕ := 5

-- Prove the question == answer given the conditions
theorem team_members_run_distance :
  total_distance / members = 30 :=
by
  sorry

end team_members_run_distance_l331_33187


namespace inheritance_calculation_l331_33165

theorem inheritance_calculation
  (x : ℝ)
  (h1 : 0.25 * x + 0.15 * (0.75 * x) = 14000) :
  x = 38600 := by
  sorry

end inheritance_calculation_l331_33165


namespace initial_money_is_correct_l331_33108

-- Given conditions
def spend_per_trip : ℕ := 2
def trips_per_month : ℕ := 4
def months_per_year : ℕ := 12
def money_left_after_year : ℕ := 104

-- Define the initial amount of money
def initial_amount_of_money (spend_per_trip trips_per_month months_per_year money_left_after_year : ℕ) : ℕ :=
  money_left_after_year + (spend_per_trip * trips_per_month * months_per_year)

-- Theorem stating that under the given conditions, the initial amount of money is 200
theorem initial_money_is_correct :
  initial_amount_of_money spend_per_trip trips_per_month months_per_year money_left_after_year = 200 :=
  sorry

end initial_money_is_correct_l331_33108


namespace GP_length_l331_33135

theorem GP_length (X Y Z G P Q : Type) 
  (XY XZ YZ : ℝ) 
  (hXY : XY = 12) 
  (hXZ : XZ = 9) 
  (hYZ : YZ = 15) 
  (hG_centroid : true)  -- Medians intersect at G (Centroid property)
  (hQ_altitude : true)  -- Q is the foot of the altitude from X to YZ
  (hP_below_G : true)  -- P is the point on YZ directly below G
  : GP = 2.4 := 
sorry

end GP_length_l331_33135


namespace work_days_l331_33114

theorem work_days (A B : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 18) :
    1 / (A + B) = 6 :=
by
  sorry

end work_days_l331_33114


namespace values_of_x_l331_33199

def f (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x (x : ℝ) : f (f x) = f x → x = 0 ∨ x = -2 ∨ x = 5 ∨ x = 6 :=
by {
  sorry
}

end values_of_x_l331_33199


namespace student_monthly_earnings_l331_33113

theorem student_monthly_earnings :
  let daily_rate := 1250
  let days_per_week := 4
  let weeks_per_month := 4
  let income_tax_rate := 0.13
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings_before_tax := weekly_earnings * weeks_per_month
  let income_tax_amount := monthly_earnings_before_tax * income_tax_rate
  let monthly_earnings_after_tax := monthly_earnings_before_tax - income_tax_amount
  monthly_earnings_after_tax = 17400 := by
  -- Proof steps here
  sorry

end student_monthly_earnings_l331_33113


namespace box_volume_l331_33143

structure Box where
  L : ℝ  -- Length
  W : ℝ  -- Width
  H : ℝ  -- Height

def front_face_area (box : Box) : ℝ := box.L * box.H
def top_face_area (box : Box) : ℝ := box.L * box.W
def side_face_area (box : Box) : ℝ := box.H * box.W

noncomputable def volume (box : Box) : ℝ := box.L * box.W * box.H

theorem box_volume (box : Box)
  (h1 : front_face_area box = 0.5 * top_face_area box)
  (h2 : top_face_area box = 1.5 * side_face_area box)
  (h3 : side_face_area box = 72) :
  volume box = 648 := by
  sorry

end box_volume_l331_33143


namespace journey_length_25_km_l331_33195

theorem journey_length_25_km:
  ∀ (D T : ℝ),
  (D = 100 * T) →
  (D = 50 * (T + 15/60)) →
  D = 25 :=
by
  intros D T h1 h2
  sorry

end journey_length_25_km_l331_33195


namespace fox_jeans_price_l331_33111

theorem fox_jeans_price (pony_price : ℝ)
                        (total_savings : ℝ)
                        (total_discount_rate : ℝ)
                        (pony_discount_rate : ℝ)
                        (fox_discount_rate : ℝ)
                        (fox_price : ℝ) :
    pony_price = 18 ∧
    total_savings = 8.91 ∧
    total_discount_rate = 0.22 ∧
    pony_discount_rate = 0.1099999999999996 ∧
    fox_discount_rate = 0.11 →
    (3 * fox_discount_rate * fox_price + 2 * pony_discount_rate * pony_price = total_savings) →
    fox_price = 15 :=
by
  intros h h_eq
  rcases h with ⟨h_pony, h_savings, h_total_rate, h_pony_rate, h_fox_rate⟩
  sorry

end fox_jeans_price_l331_33111


namespace simplify_fraction_l331_33167

variable {x y : ℝ}

theorem simplify_fraction (hx : x ≠ 0) : (x * y) / (3 * x) = y / 3 := by
  sorry

end simplify_fraction_l331_33167


namespace range_of_a_l331_33162

theorem range_of_a (x y a : ℝ) :
  (2 * x + y ≥ 4) → 
  (x - y ≥ 1) → 
  (x - 2 * y ≤ 2) → 
  (x = 2) → 
  (y = 0) → 
  (z = a * x + y) → 
  (Ax = 2) → 
  (Ay = 0) → 
  (-1/2 < a ∧ a < 2) := sorry

end range_of_a_l331_33162


namespace no_positive_integer_makes_expression_integer_l331_33145

theorem no_positive_integer_makes_expression_integer : 
  ∀ n : ℕ, n > 0 → ¬ ∃ k : ℤ, (n^(3 * n - 2) - 3 * n + 1) = k * (3 * n - 2) := 
by 
  intro n hn
  sorry

end no_positive_integer_makes_expression_integer_l331_33145


namespace exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest_l331_33128

def small_numbers (n : ℕ) : Prop := n ≤ 150

theorem exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest :
  ∃ (N : ℕ), (∃ (a b : ℕ), small_numbers a ∧ small_numbers b ∧ (a + 1 = b) ∧ ¬(N % a = 0) ∧ ¬(N % b = 0))
  ∧ (∀ (m : ℕ), small_numbers m → ¬(m = a ∨ m = b) → N % m = 0) :=
sorry

end exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest_l331_33128


namespace equation_of_the_line_l331_33103

noncomputable def line_equation (t : ℝ) : (ℝ × ℝ) := (3 * t + 6, 5 * t - 7)

theorem equation_of_the_line : ∃ m b : ℝ, (∀ t : ℝ, ∃ (x y : ℝ), line_equation t = (x, y) ∧ y = m * x + b) ∧ m = 5 / 3 ∧ b = -17 :=
by
  sorry

end equation_of_the_line_l331_33103


namespace simplify_expression_l331_33178

theorem simplify_expression (y : ℝ) : (3 * y^4)^4 = 81 * y^16 :=
by
  sorry

end simplify_expression_l331_33178


namespace Luca_milk_water_needed_l331_33168

def LucaMilk (flour : ℕ) : ℕ := (flour / 250) * 50
def LucaWater (flour : ℕ) : ℕ := (flour / 250) * 30

theorem Luca_milk_water_needed (flour : ℕ) (h : flour = 1250) : LucaMilk flour = 250 ∧ LucaWater flour = 150 := by
  rw [h]
  sorry

end Luca_milk_water_needed_l331_33168


namespace number_of_trailing_zeros_l331_33134

def trailing_zeros (n : Nat) : Nat :=
  let powers_of_two := 2 * 52^5
  let powers_of_five := 2 * 25^2
  min powers_of_two powers_of_five

theorem number_of_trailing_zeros : trailing_zeros (525^(25^2) * 252^(52^5)) = 1250 := 
by sorry

end number_of_trailing_zeros_l331_33134


namespace equilateral_triangles_circle_l331_33136

-- Definitions and conditions
structure Triangle :=
  (A B C : ℝ)
  (side_length : ℝ)
  (equilateral : side_length = 12)

structure Circle :=
  (S : ℝ)

def PointOnArc (P1 P2 P : ℝ) : Prop :=
  -- Definition to describe P lies on the arc P1P2
  sorry

-- Theorem stating the proof problem
theorem equilateral_triangles_circle
  (S : Circle)
  (T1 T2 : Triangle)
  (H1 : T1.side_length = 12)
  (H2 : T2.side_length = 12)
  (HAonArc : PointOnArc T2.B T2.C T1.A)
  (HBonArc : PointOnArc T2.A T2.B T1.B) :
  (T1.A - T2.A) ^ 2 + (T1.B - T2.B) ^ 2 + (T1.C - T2.C) ^ 2 = 288 :=
sorry

end equilateral_triangles_circle_l331_33136


namespace number_of_pairs_x_y_l331_33133

theorem number_of_pairs_x_y (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - y^2 = 85) : 
    (1 : ℕ) + (1 : ℕ) = 2 := 
by 
  sorry

end number_of_pairs_x_y_l331_33133


namespace Donny_change_l331_33179

theorem Donny_change (tank_capacity : ℕ) (initial_fuel : ℕ) (money_available : ℕ) (fuel_cost_per_liter : ℕ) 
  (h1 : tank_capacity = 150) 
  (h2 : initial_fuel = 38) 
  (h3 : money_available = 350) 
  (h4 : fuel_cost_per_liter = 3) : 
  money_available - (tank_capacity - initial_fuel) * fuel_cost_per_liter = 14 := 
by 
  sorry

end Donny_change_l331_33179


namespace sum_of_consecutive_odds_l331_33177

theorem sum_of_consecutive_odds (a : ℤ) (h : (a - 2) * a * (a + 2) = 9177) : (a - 2) + a + (a + 2) = 63 := 
sorry

end sum_of_consecutive_odds_l331_33177


namespace price_difference_proof_l331_33169

theorem price_difference_proof (y : ℝ) (n : ℕ) :
  ∃ n : ℕ, (4.20 + 0.45 * n) = (6.30 + 0.01 * y * n + 0.65) → 
  n = (275 / (45 - y)) :=
by
  sorry

end price_difference_proof_l331_33169


namespace geometric_sequence_sum_eq_80_243_l331_33101

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_eq_80_243 {n : ℕ} :
  let a := (1 / 3 : ℝ)
  let r := (1 / 3 : ℝ)
  geometric_sum a r n = 80 / 243 ↔ n = 3 :=
by
  intros a r
  sorry

end geometric_sequence_sum_eq_80_243_l331_33101


namespace gcd_squares_example_l331_33161

noncomputable def gcd_of_squares : ℕ :=
  Nat.gcd (101 ^ 2 + 202 ^ 2 + 303 ^ 2) (100 ^ 2 + 201 ^ 2 + 304 ^ 2)

theorem gcd_squares_example : gcd_of_squares = 3 :=
by
  sorry

end gcd_squares_example_l331_33161


namespace fraction_of_shoppers_avoiding_checkout_l331_33164

theorem fraction_of_shoppers_avoiding_checkout 
  (total_shoppers : ℕ) 
  (shoppers_at_checkout : ℕ) 
  (h1 : total_shoppers = 480) 
  (h2 : shoppers_at_checkout = 180) : 
  (total_shoppers - shoppers_at_checkout) / total_shoppers = 5 / 8 :=
by
  sorry

end fraction_of_shoppers_avoiding_checkout_l331_33164


namespace units_digit_n_squared_plus_two_pow_n_l331_33126

theorem units_digit_n_squared_plus_two_pow_n
  (n : ℕ)
  (h : n = 2018^2 + 2^2018) : 
  (n^2 + 2^n) % 10 = 5 := by
  sorry

end units_digit_n_squared_plus_two_pow_n_l331_33126


namespace min_positive_period_cos_2x_l331_33118

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem min_positive_period_cos_2x :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ ∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T := 
sorry

end min_positive_period_cos_2x_l331_33118


namespace find_family_ages_l331_33174

theorem find_family_ages :
  ∃ (a b father_age mother_age : ℕ), 
    (a < 21) ∧
    (b < 21) ∧
    (a^3 + b^2 > 1900) ∧
    (a^3 + b^2 < 1978) ∧
    (father_age = 1978 - (a^3 + b^2)) ∧
    (mother_age = father_age - 8) ∧
    (a = 12) ∧
    (b = 14) ∧
    (father_age = 54) ∧
    (mother_age = 46) := 
by 
  use 12, 14, 54, 46
  sorry

end find_family_ages_l331_33174


namespace age_ratio_in_ten_years_l331_33146

-- Definitions of given conditions
variable (A : ℕ) (B : ℕ)
axiom age_condition : A = 20
axiom sum_of_ages : A + 10 + (B + 10) = 45

-- Theorem and proof skeleton for the ratio of ages in ten years.
theorem age_ratio_in_ten_years (A B : ℕ) (hA : A = 20) (hSum : A + 10 + (B + 10) = 45) :
  (A + 10) / (B + 10) = 2 := by
  sorry

end age_ratio_in_ten_years_l331_33146


namespace real_solution_exists_l331_33149

theorem real_solution_exists : ∃ x : ℝ, x^3 + (x+1)^4 + (x+2)^3 = (x+3)^4 :=
sorry

end real_solution_exists_l331_33149


namespace ratio_of_3_numbers_l331_33116

variable (A B C : ℕ)
variable (k : ℕ)

theorem ratio_of_3_numbers (h₁ : A = 5 * k) (h₂ : B = k) (h₃ : C = 4 * k) (h_sum : A + B + C = 1000) : C = 400 :=
  sorry

end ratio_of_3_numbers_l331_33116


namespace line_through_points_l331_33141

theorem line_through_points (x1 y1 x2 y2 : ℝ) (h1 : x1 ≠ x2) (hx1 : x1 = -3) (hy1 : y1 = 1) (hx2 : x2 = 1) (hy2 : y2 = 5) :
  ∃ (m b : ℝ), (m + b = 5) ∧ (y1 = m * x1 + b) ∧ (y2 = m * x2 + b) :=
by
  sorry

end line_through_points_l331_33141


namespace daps_equiv_dirps_l331_33190

noncomputable def dops_equiv_daps : ℝ := 5 / 4
noncomputable def dips_equiv_dops : ℝ := 3 / 10
noncomputable def dirps_equiv_dips : ℝ := 2

theorem daps_equiv_dirps (n : ℝ) : 20 = (dops_equiv_daps * dips_equiv_dops * dirps_equiv_dips) * n → n = 15 :=
by sorry

end daps_equiv_dirps_l331_33190


namespace evaluate_expression_l331_33198

theorem evaluate_expression :
  (2 * 4 * 6) * (1 / 2 + 1 / 4 + 1 / 6) = 44 :=
by
  sorry

end evaluate_expression_l331_33198


namespace quadratic_part_of_equation_l331_33117

theorem quadratic_part_of_equation (x: ℝ) :
  (x^2 - 8*x + 21 = |x - 5| + 4) → (x^2 - 8*x + 21) = x^2 - 8*x + 21 :=
by
  intros h
  sorry

end quadratic_part_of_equation_l331_33117


namespace trigonometric_identity_l331_33171

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin α * Real.sin (3 * Real.pi / 2 - α) = -3 / 10 :=
by
  sorry

end trigonometric_identity_l331_33171


namespace minimum_sum_of_reciprocals_l331_33194

open BigOperators

theorem minimum_sum_of_reciprocals (b : Fin 15 → ℝ) (h_pos : ∀ i, 0 < b i)
    (h_sum : ∑ i, b i = 1) :
    ∑ i, 1 / (b i) ≥ 225 := sorry

end minimum_sum_of_reciprocals_l331_33194


namespace train_speed_in_kmph_l331_33173

theorem train_speed_in_kmph
  (train_length : ℝ) (bridge_length : ℝ) (time_seconds : ℝ)
  (H1: train_length = 200) (H2: bridge_length = 150) (H3: time_seconds = 34.997200223982084) :
  train_length + bridge_length = 200 + 150 →
  (train_length + bridge_length) / time_seconds * 3.6 = 36 :=
sorry

end train_speed_in_kmph_l331_33173


namespace exist_positive_integers_for_perfect_squares_l331_33147

theorem exist_positive_integers_for_perfect_squares :
  ∃ (x y : ℕ), (0 < x ∧ 0 < y) ∧ (∃ a b c : ℕ, x + y = a^2 ∧ x^2 + y^2 = b^2 ∧ x^3 + y^3 = c^2) :=
by
  sorry

end exist_positive_integers_for_perfect_squares_l331_33147


namespace least_value_expression_l331_33150

-- Definition of the expression
def expression (x y : ℝ) := (x * y - 2) ^ 2 + (x - 1 + y) ^ 2

-- Statement to prove the least possible value of the expression
theorem least_value_expression : ∃ x y : ℝ, expression x y = 2 := 
sorry

end least_value_expression_l331_33150


namespace alphabet_letters_l331_33189

theorem alphabet_letters (DS S_only Total D_only : ℕ) 
  (h_DS : DS = 9) 
  (h_S_only : S_only = 24) 
  (h_Total : Total = 40) 
  (h_eq : Total = D_only + S_only + DS) 
  : D_only = 7 := 
by
  sorry

end alphabet_letters_l331_33189


namespace find_x_value_l331_33196

theorem find_x_value (x y z k: ℚ)
  (h1 : x = k * (z^3) / (y^2))
  (h2 : y = 2) (h3 : z = 3)
  (h4 : x = 1)
  : x = (4 / 27) * (4^3) / (6^2) := by
  sorry

end find_x_value_l331_33196


namespace cube_tangent_ratio_l331_33100

theorem cube_tangent_ratio 
  (edge_length : ℝ) 
  (midpoint K : ℝ) 
  (tangent E : ℝ) 
  (intersection F : ℝ) 
  (radius R : ℝ)
  (h1 : edge_length = 2)
  (h2 : radius = 1)
  (h3 : K = midpoint)
  (h4 : ∃ E F, tangent = E ∧ intersection = F) :
  (K - E) / (F - E) = 4 / 5 :=
sorry

end cube_tangent_ratio_l331_33100


namespace find_number_l331_33120

theorem find_number :
  ∃ x : ℕ, (8 * x + 5400) / 12 = 530 ∧ x = 120 :=
by
  sorry

end find_number_l331_33120


namespace system1_solution_system2_solution_l331_33183

-- Part 1: Substitution Method
theorem system1_solution (x y : ℤ) :
  2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ↔ x = 2 ∧ y = 1 :=
by
  sorry

-- Part 2: Elimination Method
theorem system2_solution (x y : ℚ) :
  2 * x + y = 2 ∧ 8 * x + 3 * y = 9 ↔ x = 3 / 2 ∧ y = -1 :=
by
  sorry

end system1_solution_system2_solution_l331_33183


namespace smallest_integer_k_no_real_roots_l331_33185

def quadratic_no_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c < 0

theorem smallest_integer_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, quadratic_no_real_roots (2 * k - 1) (-8) 6) ∧ (k = 2) :=
by
  sorry

end smallest_integer_k_no_real_roots_l331_33185


namespace percent_profit_l331_33152

-- Definitions based on given conditions
variables (P : ℝ) -- original price of the car

def discounted_price := 0.90 * P
def first_year_value := 0.945 * P
def second_year_value := 0.9828 * P
def third_year_value := 1.012284 * P
def selling_price := 1.62 * P

-- Theorem statement
theorem percent_profit : (selling_price P - P) / P * 100 = 62 := by
  sorry

end percent_profit_l331_33152


namespace number_is_seven_l331_33153

theorem number_is_seven (x : ℝ) (h : x^2 + 120 = (x - 20)^2) : x = 7 := 
by
  sorry

end number_is_seven_l331_33153


namespace height_average_inequality_l331_33157

theorem height_average_inequality 
    (a b c d : ℝ)
    (h1 : 3 * a + 2 * b = 2 * c + 3 * d)
    (h2 : a > d) : 
    (|c + d| / 2 > |a + b| / 2) :=
sorry

end height_average_inequality_l331_33157


namespace books_more_than_figures_l331_33181

-- Definitions of initial conditions
def initial_action_figures := 2
def initial_books := 10
def added_action_figures := 4

-- Problem statement to prove
theorem books_more_than_figures :
  initial_books - (initial_action_figures + added_action_figures) = 4 :=
by
  -- Proof goes here
  sorry

end books_more_than_figures_l331_33181


namespace cistern_fill_time_l331_33125

theorem cistern_fill_time (A B : ℝ) (hA : A = 1/60) (hB : B = 1/45) : (|A - B|)⁻¹ = 180 := by
  sorry

end cistern_fill_time_l331_33125


namespace largest_digit_7182N_divisible_by_6_l331_33105

noncomputable def largest_digit_divisible_by_6 : ℕ := 6

theorem largest_digit_7182N_divisible_by_6 (N : ℕ) : 
  (N % 2 = 0) ∧ ((18 + N) % 3 = 0) ↔ (N ≤ 9) ∧ (N = 6) :=
by
  sorry

end largest_digit_7182N_divisible_by_6_l331_33105


namespace sum_of_nonneg_numbers_ineq_l331_33193

theorem sum_of_nonneg_numbers_ineq
  (a b c d : ℝ)
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0)
  (h_sum : a + b + c + d = 4) :
  (a * b + c * d) * (a * c + b * d) * (a * d + b * c) ≤ 8 := sorry

end sum_of_nonneg_numbers_ineq_l331_33193


namespace smallest_M_exists_l331_33119

theorem smallest_M_exists :
  ∃ M : ℕ, M = 249 ∧
  (∃ k1 : ℕ, (M + k1 = 8 * k1 ∨ M + k1 + 1 = 8 * k1 ∨ M + k1 + 2 = 8 * k1)) ∧
  (∃ k2 : ℕ, (M + k2 = 27 * k2 ∨ M + k2 + 1 = 27 * k2 ∨ M + k2 + 2 = 27 * k2)) ∧
  (∃ k3 : ℕ, (M + k3 = 125 * k3 ∨ M + k3 + 1 = 125 * k3 ∨ M + k3 + 2 = 125 * k3)) :=
by
  sorry

end smallest_M_exists_l331_33119
