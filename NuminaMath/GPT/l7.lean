import Mathlib

namespace geometric_seq_necessary_not_sufficient_l7_7160

theorem geometric_seq_necessary_not_sufficient {a b c : ℝ} :
  (b^2 = a * c) →
  ((∃ r : ℝ, b = r * a ∧ c = r * b) ↔ (b^2 = a * c)) :=
by
  intro h
  /- Proof of necessary condition -/
  split
  · intro h_geometric
    cases h_geometric with r hr
    cases hr with hr1 hr2
    rw [hr1, hr2]
    ring
  /- Proof of not sufficient condition -/
  intro h_cond
  use [0, 0]
  sorry

end geometric_seq_necessary_not_sufficient_l7_7160


namespace no_solution_a_squared_plus_b_squared_eq_2023_l7_7540

theorem no_solution_a_squared_plus_b_squared_eq_2023 :
  ∀ (a b : ℤ), a^2 + b^2 ≠ 2023 := 
by
  sorry

end no_solution_a_squared_plus_b_squared_eq_2023_l7_7540


namespace fib_sum_series_eq_l7_7048

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

-- Define the infinite sum of Fibonacci sequence divided by powers of 5
noncomputable def fib_sum_series : ℝ := ∑' n, (fib n : ℝ) / (5 : ℝ) ^ n

-- State the theorem that needs to be proven
theorem fib_sum_series_eq : fib_sum_series = 5 / 21 := by sorry

end fib_sum_series_eq_l7_7048


namespace years_between_cars_l7_7191

theorem years_between_cars:
  (year_first_car second_car_diff third_car third_car_diff : ℕ) 
  (h1: year_first_car = 1970)
  (h2: second_car_diff = 10)
  (h3: third_car = 2000) :
  third_car_diff = (2000 - (1970 + 10)) :=
begin
  sorry
end

end years_between_cars_l7_7191


namespace smaller_denom_is_five_l7_7687

-- Define the conditions
def num_smaller_bills : ℕ := 4
def num_ten_dollar_bills : ℕ := 8
def total_bills : ℕ := num_smaller_bills + num_ten_dollar_bills
def ten_dollar_bill_value : ℕ := 10
def total_value : ℕ := 100

-- Define the smaller denomination value
def value_smaller_denom (x : ℕ) : Prop :=
  num_smaller_bills * x + num_ten_dollar_bills * ten_dollar_bill_value = total_value

-- Prove that the value of the smaller denomination bill is 5
theorem smaller_denom_is_five : value_smaller_denom 5 :=
by
  sorry

end smaller_denom_is_five_l7_7687


namespace remainder_when_55_times_57_divided_by_8_l7_7605

theorem remainder_when_55_times_57_divided_by_8 :
  (55 * 57) % 8 = 7 :=
by
  -- Insert the proof here
  sorry

end remainder_when_55_times_57_divided_by_8_l7_7605


namespace find_slope_l7_7451

theorem find_slope (k : ℝ) : (∃ x : ℝ, (y = k * x + 2) ∧ (y = 0) ∧ (abs x = 4)) ↔ (k = 1/2 ∨ k = -1/2) := by
  sorry

end find_slope_l7_7451


namespace repeating_six_equals_fraction_l7_7749

theorem repeating_six_equals_fraction : ∃ f : ℚ, (∀ n : ℕ, (n ≥ 1 → (6 * (10 : ℕ) ^ (-n) : ℚ) + (f - (6 * (10 : ℕ) ^ (-n) : ℚ)) = f)) ∧ f = 2 / 3 := sorry

end repeating_six_equals_fraction_l7_7749


namespace geom_sequence_sum_first_ten_terms_l7_7379

noncomputable def geom_sequence_sum (a1 q n : ℕ) : ℕ :=
  a1 * (1 - q^n) / (1 - q)

theorem geom_sequence_sum_first_ten_terms (a : ℕ) (q : ℕ) (h1 : a * (1 + q) = 6) (h2 : a * q^3 * (1 + q) = 48) :
  geom_sequence_sum a q 10 = 2046 :=
sorry

end geom_sequence_sum_first_ten_terms_l7_7379


namespace train_speed_correct_l7_7640

-- Definitions for the given conditions
def train_length : ℝ := 320
def time_to_cross : ℝ := 6

-- The speed of the train
def train_speed : ℝ := 53.33

-- The proof statement
theorem train_speed_correct : train_speed = train_length / time_to_cross :=
by
  sorry

end train_speed_correct_l7_7640


namespace moles_NH3_formed_l7_7358

-- Definitions
def moles_Li3N := 3
def moles_LiOH := 9
def total_H2O_grams := 162
def molar_mass_H2O := 18.015
def moles_H2O := total_H2O_grams / molar_mass_H2O

-- Balanced chemical equation yields molar ratio
def moles_NH3_expected := moles_Li3N

-- Proof Statement
theorem moles_NH3_formed :
  moles_NH3_expected = 3 := 
sorry

end moles_NH3_formed_l7_7358


namespace time_to_fill_tank_l7_7591

noncomputable def pipeA_rate := (1 : ℚ) / 10
noncomputable def pipeB_rate := (1 : ℚ) / 15
noncomputable def pipeC_rate := - (1 : ℚ) / 20
noncomputable def combined_rate := pipeA_rate + pipeB_rate + pipeC_rate
noncomputable def time_to_fill := 1 / combined_rate

theorem time_to_fill_tank : time_to_fill = 60 / 7 :=
by
  sorry

end time_to_fill_tank_l7_7591


namespace sum_of_odd_digits_greater_by_49_l7_7679

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def sum_of_digits (ns : List ℕ) : ℕ :=
  ns.sumBy digit_sum

def odd_numbers : List ℕ :=
  List.range' 1 50 |> List.map (fun n => 2 * n - 1)

def even_numbers : List ℕ :=
  List.range' 1 50 |> List.map (fun n => 2 * n)

theorem sum_of_odd_digits_greater_by_49 :
  sum_of_digits odd_numbers - sum_of_digits even_numbers = 49 :=
by
  sorry

end sum_of_odd_digits_greater_by_49_l7_7679


namespace profit_functions_properties_l7_7721

noncomputable def R (x : ℝ) : ℝ := 3000 * x - 20 * x^2
noncomputable def C (x : ℝ) : ℝ := 500 * x + 4000
noncomputable def P (x : ℝ) : ℝ := R x - C x
noncomputable def MP (x : ℝ) : ℝ := P (x + 1) - P x

theorem profit_functions_properties :
  (P x = -20 * x^2 + 2500 * x - 4000) ∧ 
  (MP x = -40 * x + 2480) ∧ 
  (∃ x_max₁, ∀ x, P x_max₁ ≥ P x) ∧ 
  (∃ x_max₂, ∀ x, MP x_max₂ ≥ MP x) ∧ 
  P x_max₁ ≠ MP x_max₂ := by
  sorry

end profit_functions_properties_l7_7721


namespace investment_last_duration_l7_7685

noncomputable def compound_interest_time (P A r n : ℝ) : ℝ := 
  log (A / P) / (n * log (1 + r / n))

theorem investment_last_duration :
  compound_interest_time 7000 8470 0.10 1 = 2 :=
by
  sorry

end investment_last_duration_l7_7685


namespace probability_lt_one_third_l7_7858

theorem probability_lt_one_third :
  let total_interval := (0 : ℝ, 1/2)
  let desired_interval := (0 : ℝ, 1/3)
  let total_length := (total_interval.2 - total_interval.1 : ℝ) = 1/2
  let desired_length := (desired_interval.2 - desired_interval.1 : ℝ) = 1/3
  -- then the probability P is given by:
  (desired_length / total_length) = 2/3
:=
by {
  -- definition of intervals and lengths
  let total_interval := (0 : ℝ, 1/2)
  let desired_interval := (0 : ℝ, 1/3)
  have total_length : total_interval.2 - total_interval.1 = 1/2,
  -- verify total length calculation
  calc
    total_interval.2 - total_interval.1 = 1/2 - 0 : by simp
                                 ... = 1/2      : by norm_num,
  have desired_length : desired_interval.2 - desired_interval.1 = 1/3,
  -- verify desired interval length calculation
  calc
    desired_interval.2 - desired_interval.1 = 1/3 - 0 : by simp
                                    ... = 1/3      : by norm_num,
  -- calculate probability
  set P := desired_length / total_length
  -- compute correct answer
  have : P = (1/3) / (1/2),
  calc
    (1/3) / (1/2) = (1/3) * (2/1) : by field_simp
              ...  = 2/3      : by norm_num,
  exact this
}

end probability_lt_one_third_l7_7858


namespace probability_of_less_than_one_third_l7_7862

theorem probability_of_less_than_one_third :
  (prob_of_interval (0 : ℝ) (1 / 2 : ℝ) (1 / 3 : ℝ) = 2 / 3) :=
sorry

end probability_of_less_than_one_third_l7_7862


namespace number_of_classmates_l7_7121

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l7_7121


namespace number_of_owls_joined_l7_7273

-- Define the initial condition
def initial_owls : ℕ := 3

-- Define the current condition
def current_owls : ℕ := 5

-- Define the problem statement as a theorem
theorem number_of_owls_joined : (current_owls - initial_owls) = 2 :=
by
  sorry

end number_of_owls_joined_l7_7273


namespace number_of_classmates_l7_7119

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l7_7119


namespace four_lines_set_l7_7327

-- Define the ⬩ operation
def clubsuit (a b : ℝ) := a^3 * b - a * b^3

-- Define the main theorem
theorem four_lines_set (x y : ℝ) : 
  (clubsuit x y = clubsuit y x) ↔ (y = 0 ∨ x = 0 ∨ y = x ∨ y = -x) :=
by sorry

end four_lines_set_l7_7327


namespace P_positive_l7_7925

variable {P : ℕ → ℤ}
variable P_cond0 : P 0 > 0
variable P_cond1 : P 1 > P 0
variable P_cond2 : P 2 > 2 * P 1 - P 0
variable P_cond3 : P 3 > 3 * P 2 - 3 * P 1 + P 0
variable P_condN : ∀ n, P (n + 4) > 4 * P (n + 3) - 6 * P (n + 2) + 4 * P (n + 1) - P n

theorem P_positive (n : ℕ) (n_pos : 0 < n) : 0 < P n :=
sorry

end P_positive_l7_7925


namespace ratio_of_mpg_l7_7325

variable (Darlene_mpg Martha_gallons Martha_miles : ℕ)

-- Given conditions
def Darlene_mpg := 20
def Martha_gallons := 30
def Martha_miles := 300

noncomputable def Martha_mpg := Martha_miles / Martha_gallons

-- The statement to prove
theorem ratio_of_mpg : Martha_mpg Martha_gallons Martha_miles / Darlene_mpg = 0.5 :=
by
  sorry

end ratio_of_mpg_l7_7325


namespace two_pow_n_plus_one_divisible_by_three_l7_7332

theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h1 : n > 0) :
  (2 ^ n + 1) % 3 = 0 ↔ n % 2 = 1 := 
sorry

end two_pow_n_plus_one_divisible_by_three_l7_7332


namespace simplest_quadratic_radical_l7_7616

theorem simplest_quadratic_radical :
  ∀ (a b c d : ℝ),
    a = Real.sqrt 12 →
    b = Real.sqrt (2 / 3) →
    c = Real.sqrt 0.3 →
    d = Real.sqrt 7 →
    d = Real.sqrt 7 :=
by
  intros a b c d ha hb hc hd
  rw [hd]
  sorry

end simplest_quadratic_radical_l7_7616


namespace line_perpendicular_to_beta_l7_7932

variables (α β : Plane) (l : Line)

-- Plane line denotes α and β are different planes.
axiom diff_planes (h1 : α ≠ β) 

-- Define perpendicular and parallel relations
axiom perp_line_plane (l : Line) (π : Plane) : Prop
axiom parallel_plane (π1 π2 : Plane) : Prop

theorem line_perpendicular_to_beta (h2 : perp_line_plane l α) (h3 : parallel_plane α β) : perp_line_plane l β := 
sorry

end line_perpendicular_to_beta_l7_7932


namespace middle_and_oldest_son_ages_l7_7661

theorem middle_and_oldest_son_ages 
  (x y z : ℕ) 
  (father_age_current father_age_future : ℕ) 
  (youngest_age_increment : ℕ)
  (father_age_increment : ℕ) 
  (father_equals_sons_sum : father_age_future = (x + youngest_age_increment) + (y + father_age_increment) + (z + father_age_increment))
  (father_age_constraint : father_age_current + father_age_increment = father_age_future)
  (youngest_age_initial : x = 2)
  (father_age_current_value : father_age_current = 33)
  (youngest_age_increment_value : youngest_age_increment = 12)
  (father_age_increment_value : father_age_increment = 12) 
  :
  y = 3 ∧ z = 4 :=
begin
  sorry
end

end middle_and_oldest_son_ages_l7_7661


namespace fraction_for_repeating_decimal_l7_7762

variable (a r S : ℚ)
variable (h1 : a = 3/5)
variable (h2 : r = 1/10)
variable (h3 : S = a / (1 - r))

theorem fraction_for_repeating_decimal : S = 2 / 3 :=
by
  have h4 : 1 - r = 9 / 10, from sorry
  have h5 : S = (3 / 5) / (9 / 10), from sorry
  have h6 : S = (3 * 10) / (5 * 9), from sorry
  have h7 : S = 30 / 45, from sorry
  have h8 : 30 / 45 = 2 / 3, from sorry
  exact h8

end fraction_for_repeating_decimal_l7_7762


namespace student_sister_weight_ratio_l7_7871

theorem student_sister_weight_ratio 
(student_weight : ℕ) 
(sister_weight : ℕ) 
(total_weight : ℕ) 
(h1 : student_weight + sister_weight = total_weight) 
(h2 : student_weight = 75) 
(h3 : total_weight = 110) : 
(student_weight - 5) / (total_weight - student_weight) = 2 :=
by 
  -- Assigning the variables based on the given conditions
  have s_weight := 75 - 5
  have r_weight := 110 - 75
  -- Calculation of the ratio
  have ratio := (s_weight : ℚ) / (r_weight : ℚ)
  show ratio = 2 
  sorry

end student_sister_weight_ratio_l7_7871


namespace incorrect_calculation_l7_7612

theorem incorrect_calculation (a : ℝ) : (2 * a) ^ 3 ≠ 6 * a ^ 3 :=
by {
  sorry
}

end incorrect_calculation_l7_7612


namespace parallel_vectors_implies_x_l7_7833

-- a definition of the vectors a and b
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (1, x)

-- a definition for vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- a definition for scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- a definition for vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

-- the theorem statement
theorem parallel_vectors_implies_x (x : ℝ) (h : 
  vector_add vector_a (vector_b x) = ⟨3, 1 + x⟩ ∧
  vector_sub (scalar_mul 2 vector_a) (vector_b x) = ⟨3, 2 - x⟩ ∧
  ∃ k : ℝ, vector_add vector_a (vector_b x) = scalar_mul k (vector_sub (scalar_mul 2 vector_a) (vector_b x))
  ) : x = 1 / 2 :=
sorry

end parallel_vectors_implies_x_l7_7833


namespace geometric_sequence_product_l7_7934

variable {a : ℕ → ℝ} (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1)
variable (a_five : a 5 = Math.log 8 / Math.log 2)

theorem geometric_sequence_product (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1)
  (a_five : a 5 = Real.log 8 / Real.log 2) : a 4 * a 6 = 9 := by
  sorry

end geometric_sequence_product_l7_7934


namespace find_sinα_and_tanα_l7_7837

open Real 

noncomputable def vectors (α : ℝ) := (Real.cos α, 1)

noncomputable def vectors_perpendicular (α : ℝ) := (Real.sin α, -2)

theorem find_sinα_and_tanα (α: ℝ) (hα: π < α ∧ α < 3 * π / 2)
  (h_perp: vectors_perpendicular α = (Real.sin α, -2) ∧ vectors α = (Real.cos α, 1) ∧ (vectors α).1 * (vectors_perpendicular α).1 + (vectors α).2 * (vectors_perpendicular α).2 = 0):
  (Real.sin α = - (2 * Real.sqrt 5) / 5) ∧ 
  (Real.tan (α + π / 4) = -3) := 
sorry 

end find_sinα_and_tanα_l7_7837


namespace circle_diameter_l7_7711

noncomputable def diameter_of_circle_C (D: ℝ) (ratio: ℝ): ℝ :=
  let radius_D := D / 2
  let area_D := π * radius_D ^ 2
  let area_C := area_D / (ratio + 1)
  let radius_C := sqrt (area_C / π)
  2 * radius_C

theorem circle_diameter (D: ℝ) (ratio: ℝ) (hD: D = 20) (hRatio: ratio = 4):
  diameter_of_circle_C D ratio = 4 * sqrt 5 :=
by
  rw [hD, hRatio]
  dsimp [diameter_of_circle_C]
  rw [sqrt_div (eq.mpr _ (mul_pos zero_lt_four (by norm_num : 0 < π))), sqrt_mul]
  norm_num
  sorry

end circle_diameter_l7_7711


namespace domain_of_h_l7_7725

noncomputable def h (x : ℝ) : ℝ := (x^4 - 2*x^3 + 5*x^2 - 3*x + 2) / (x^2 - 5*x + 6)

theorem domain_of_h :
  {x : ℝ | ∃ y, h y = h x} = {x : ℝ | x ∈ Set.Ioo (-∞) 2 ∪ Set.Ioo 2 3 ∪ Set.Ioo 3 ∞} :=
by
  sorry

end domain_of_h_l7_7725


namespace range_of_a_l7_7559

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ x
def g (x : ℝ) : ℝ := x ^ 2

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a > 1 ∧ (f a x = g x) → x ∈ set.Icc (-((/+)/) x e) e) :
  1 < a ∧ a < Real.exp (2 / Real.exp 1) :=
by
  sorry

end range_of_a_l7_7559


namespace right_triangle_third_side_l7_7794

theorem right_triangle_third_side (a b: ℝ) (h₁: a = 3 ∨ a = 4) (h₂: b = 3 ∨ b = 4)
  (h₃: a ≠ b ∧ (a ≠ 4 ∨ b ≠ 3)): 
  (∃ c: ℝ, c^2 = a^2 + b^2 ∨ c^2 + a^2 = b^2) → (∃ k: ℝ, k = 5 ∨ k = real.sqrt 7) :=
by
  sorry

end right_triangle_third_side_l7_7794


namespace min_points_on_tetrahedron_l7_7807

theorem min_points_on_tetrahedron (a : ℝ) (h : 0 < a) : 
  ∃ (n : ℕ), n = 7 ∧ 
    (∀ p : fin n → (ℝ × ℝ × ℝ), 
      (∃ (i j : fin n), i ≠ j ∧ dist (p i) (p j) ≤ a / 2)) :=
sorry

end min_points_on_tetrahedron_l7_7807


namespace circle_area_from_intersection_l7_7387

-- Statement of the problem
theorem circle_area_from_intersection (r : ℝ) (A B : ℝ × ℝ)
  (h_circle : ∀ x y, (x + 2) ^ 2 + y ^ 2 = r ^ 2 ↔ (x, y) = A ∨ (x, y) = B)
  (h_parabola : ∀ x y, y ^ 2 = 20 * x ↔ (x, y) = A ∨ (x, y) = B)
  (h_axis_sym : A.1 = -5 ∧ B.1 = -5)
  (h_AB_dist : |A.2 - B.2| = 8) : π * r ^ 2 = 25 * π :=
by
  sorry

end circle_area_from_intersection_l7_7387


namespace transformations_correct_l7_7584

def y_sin_x := λ x : ℝ, Real.sin x
def y_cos_2x_plus_pi_over_6 := λ x : ℝ, Real.cos (2 * x + Real.pi / 6)

theorem transformations_correct :
  (∀ x : ℝ, y_sin_x x = Real.cos (x - Real.pi / 2)) ∧
  (∀ x : ℝ, y_cos_2x_plus_pi_over_6 x = Real.cos (2 * x + Real.pi / 6)) →
  (∀ x : ℝ, (λ x : ℝ, Real.cos (x - Real.pi / 2 - Real.pi / 3)) = λ x : ℝ, Real.cos (x + Real.pi / 6)) ∧
  (∀ x : ℝ, (λ x : ℝ, Real.cos (2 * (x + Real.pi / 3))) = λ x : ℝ, Real.cos (2 * x + Real.pi / 6)) ∧
  (∀ x : ℝ, (λ x : ℝ, Real.cos (2 * (x - Real.pi / 3))) = λ x : ℝ, Real.cos (2 * x - 2 * Real.pi / 3)) :=
sorry

end transformations_correct_l7_7584


namespace classmates_ate_cake_l7_7095

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l7_7095


namespace circle_diameter_l7_7712

noncomputable def diameter_of_circle_C (D: ℝ) (ratio: ℝ): ℝ :=
  let radius_D := D / 2
  let area_D := π * radius_D ^ 2
  let area_C := area_D / (ratio + 1)
  let radius_C := sqrt (area_C / π)
  2 * radius_C

theorem circle_diameter (D: ℝ) (ratio: ℝ) (hD: D = 20) (hRatio: ratio = 4):
  diameter_of_circle_C D ratio = 4 * sqrt 5 :=
by
  rw [hD, hRatio]
  dsimp [diameter_of_circle_C]
  rw [sqrt_div (eq.mpr _ (mul_pos zero_lt_four (by norm_num : 0 < π))), sqrt_mul]
  norm_num
  sorry

end circle_diameter_l7_7712


namespace range_of_x_l7_7789

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the relevant conditions
axiom decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) < 0
axiom symmetry : ∀ x : ℝ, f (1 - x) = -f (1 + x)
axiom f_one : f 1 = -1

-- Define the statement to be proved
theorem range_of_x : ∀ x : ℝ, -1 ≤ f (0.5 * x - 1) ∧ f (0.5 * x - 1) ≤ 1 → 0 ≤ x ∧ x ≤ 4 :=
sorry

end range_of_x_l7_7789


namespace triangle_division_parallel_segment_length_l7_7677

noncomputable def length_of_parallel_segment (base : ℝ) : ℝ :=
  let area_ratio : ℝ := 1 / 3
  let ratio_of_sides : ℝ := real.sqrt area_ratio
  in ratio_of_sides * base

theorem triangle_division_parallel_segment_length :
  ∀ (base : ℝ), base = 18 → length_of_parallel_segment base = 6 * real.sqrt 3 :=
by
  intro base base_eq
  rw [base_eq, length_of_parallel_segment]
  unfold real.sqrt
  norm_num

#check triangle_division_parallel_segment_length

end triangle_division_parallel_segment_length_l7_7677


namespace diameter_of_circle_C_l7_7709

-- Define the conditions

def circle_radius_D : ℝ := 10
def area_circle_D : ℝ := 100 * Real.pi
def ratio_shaded_to_circle_C : ℝ := 4

-- Statement to prove the diameter of circle C
theorem diameter_of_circle_C (d : ℝ) :
  let radius_C := d / 2 in
  let area_circle_C := Real.pi * (radius_C ^ 2) in
  let shaded_area := ratio_shaded_to_circle_C * area_circle_C in
  shaded_area + area_circle_C = area_circle_D → 
  d = 8 * Real.sqrt 5 :=
sorry

end diameter_of_circle_C_l7_7709


namespace isosceles_triangle_perimeter_l7_7456

-- Define the lengths of the sides
def side1 : ℕ := 4
def side2 : ℕ := 7

-- Condition: The given sides form an isosceles triangle
def is_isosceles_triangle (a b : ℕ) : Prop := a = b ∨ a = 4 ∧ b = 7 ∨ a = 7 ∧ b = 4

-- Condition: The triangle inequality theorem must be satisfied
def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem we want to prove
theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : is_isosceles_triangle a b) (h2 : triangle_inequality a a b ∨ triangle_inequality b b a) :
  a + a + b = 15 ∨ b + b + a = 18 := 
sorry

end isosceles_triangle_perimeter_l7_7456


namespace right_triangle_cosine_l7_7471

theorem right_triangle_cosine (DE EF : ℝ) (hDE : DE = 8) (hEF : EF = 15) (h_angleD_right : ∠ DEF = π / 2) :
  let DF := Real.sqrt (DE^2 + EF^2)
  ∃ (DF : ℝ), DF = 17 ∧ cos (atan2 EF DE) = 8 / DF :=
by
  sorry

end right_triangle_cosine_l7_7471


namespace number_of_sections_l7_7033

-- Definitions based on the conditions in a)
def num_reels : Nat := 3
def length_per_reel : Nat := 100
def section_length : Nat := 10

-- The math proof problem statement
theorem number_of_sections :
  (num_reels * length_per_reel) / section_length = 30 := by
  sorry

end number_of_sections_l7_7033


namespace compound_interest_time_l7_7181

variable (P_SI R_SI T_SI P_CI R_CI : ℝ)
variable (n : ℝ)

def simpleInterest (P R T : ℝ) : ℝ := P * R * T / 100

def compoundInterest (P R T : ℝ) : ℝ := P * ((1 + R / 100)^T - 1)

theorem compound_interest_time (h1 : P_SI = 1750)
                               (h2 : R_SI = 8)
                               (h3 : T_SI = 3)
                               (h4 : P_CI = 4000)
                               (h5 : R_CI = 10)
                               (h6 : simpleInterest P_SI R_SI T_SI = 420)
                               (h7 : simpleInterest P_SI R_SI T_SI * 2 = compoundInterest P_CI R_CI n) :
  n = 2 := by
  sorry

end compound_interest_time_l7_7181


namespace probability_white_ball_l7_7466

-- Definitions for the given conditions
def P_red : ℝ := 0.3
def P_black : ℝ := 0.5
def events_mutually_exclusive : Prop := 
    -- explanation: probabilities for mutually exclusive events sums to 1
    P_red + P_black + P_white = 1

-- Goal to prove
theorem probability_white_ball (P_white : ℝ) (h_red : P_red = 0.3)
    (h_black : P_black = 0.5) (h_mutual : events_mutually_exclusive) :
    P_white = 0.2 :=
by sorry

end probability_white_ball_l7_7466


namespace andy_max_cookies_l7_7592

theorem andy_max_cookies (total_cookies : ℕ) (andy_cookies : ℕ) (bella_cookies : ℕ)
  (h1 : total_cookies = 30)
  (h2 : bella_cookies = 2 * andy_cookies)
  (h3 : andy_cookies + bella_cookies = total_cookies) :
  andy_cookies = 10 := by
  sorry

end andy_max_cookies_l7_7592


namespace matrix_multiplication_and_vector_solution_l7_7389

-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![[4, 0], [0, 1]]

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![[1, 2], [0, 5]]

-- Define the column vector X
def X (a b : ℝ) : Matrix (Fin 2) (Fin 1) ℝ :=
  ![[a], [b]]

-- Define the product AB
def AB : Matrix (Fin 2) (Fin 2) ℝ :=
  A ⬝ B

-- Condition given in the problem
def B_inv_A_inv_X_eq : Prop :=
  (B⁻¹ ⬝ A⁻¹ ⬝ (X 28 5) = ![[5], [1]])

-- Proof problem statement
theorem matrix_multiplication_and_vector_solution (a b : ℝ) :
  AB = ![[4, 8], [0, 5]] ∧ B_inv_A_inv_X_eq → (a = 28 ∧ b = 5) :=
sorry

end matrix_multiplication_and_vector_solution_l7_7389


namespace possible_classmates_l7_7135

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l7_7135


namespace light_path_length_l7_7929

-- Definitions for the problem conditions
structure cube :=
  (length : ℝ)
  (width : ℝ)
  (height : ℝ)
  deriving Repr

def cube_ABCD_BCFG : cube :=
{ length := 10, width := 10, height := 10 }

def point_P_in_face_BCFG (P_distance_BG : ℝ) (P_distance_BC : ℝ) : Prop :=
  P_distance_BG = 3 ∧ P_distance_BC = 4

-- Statement of the proof problem
theorem light_path_length (c : cube) (P_distance_BG P_distance_BC : ℝ) 
  (h1 : c.length = 10) (h2 : c.width = 10) (h3 : c.height = 10) 
  (hP : point_P_in_face_BCFG P_distance_BG P_distance_BC) : 
  ∃ p q : ℕ, q % (p * p) ≠ 0 ∧ c.length + q = 55 := 
by
  -- Proof omitted
  sorry

end light_path_length_l7_7929


namespace value_of_expression_l7_7444

theorem value_of_expression (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  simp
  rfl

end value_of_expression_l7_7444


namespace granary_circumference_proof_l7_7475

noncomputable def cylinder_circumference : ℝ :=
  let height := (10 * 1 + 3 + (10 / 3) * (1 / 10)) -- height in chi
  let volume := 2000 * 1.62 -- volume in cubic chi
  let base_area := volume / height -- S in cubic chi
  let radius := real.sqrt (base_area / 3) -- r in chi, using π ≈ 3
  2 * 3 * radius -- circumference in chi

theorem granary_circumference_proof :
  cylinder_circumference = 54 := by
  sorry

end granary_circumference_proof_l7_7475


namespace common_chord_properties_l7_7834

noncomputable def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 1 = 0

noncomputable def length_common_chord : ℝ := 2 * Real.sqrt 5

theorem common_chord_properties :
  (∀ x y : ℝ, 
    x^2 + y^2 + 2 * x + 8 * y - 8 = 0 ∧
    x^2 + y^2 - 4 * x - 4 * y - 2 = 0 →
    line_equation x y) ∧ 
  length_common_chord = 2 * Real.sqrt 5 :=
by
  sorry

end common_chord_properties_l7_7834


namespace function_symmetry_l7_7822

-- Define the function
def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

-- Condition: The function is symmetric about the line x = 5π/12
theorem function_symmetry :
  ∀ x : ℝ, f (x) = f (Real.pi * (5 / 12) - x) := by
  -- Proof will come here
  sorry

end function_symmetry_l7_7822


namespace multiplication_difference_is_564_l7_7691

/-- 
Given:
1. The digits in the fourth line's front two digits are both 9.
2. The first three digits of the resulting number are 1, 0, and 0.
3. The resulting difference is 564.
-/
theorem multiplication_difference_is_564
  (a b : ℕ)
  (h1 : nat.digits 10 (a * b) = [1, 0, 0, ...])
  (h2 : nat.digits 10 a = 9 :: 9 :: _ )
  (h3 : 564 ∈ [564, 574, 664, 674]) :
  a - b = 564 := 
sorry

end multiplication_difference_is_564_l7_7691


namespace repeating_decimal_sum_l7_7703

theorem repeating_decimal_sum :
  (0.\overline{3} = 1 / 3) → (0.\overline{6} = 2 / 3) → (0.\overline{3} + 0.\overline{6} = 1) :=
by
  sorry

end repeating_decimal_sum_l7_7703


namespace repeating_six_as_fraction_l7_7741

theorem repeating_six_as_fraction : (∑' n : ℕ, 6 / (10 * (10 : ℝ)^n)) = (2 / 3) :=
by
  sorry

end repeating_six_as_fraction_l7_7741


namespace intersection_of_A_and_B_l7_7947

-- Definitions representing the conditions
def setA : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def setB : Set ℝ := {x | x < 2}

-- Proof problem statement
theorem intersection_of_A_and_B : setA ∩ setB = {x | -1 < x ∧ x < 2} :=
sorry

end intersection_of_A_and_B_l7_7947


namespace dividing_cotton_among_eight_sons_l7_7901

theorem dividing_cotton_among_eight_sons :
  ∃ a : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 8 → a + (i - 1) * 17) ∧
    (∑ i in finset.range 8, a + i * 17) = 996 := sorry

end dividing_cotton_among_eight_sons_l7_7901


namespace complex_purely_imaginary_a_eq_3_l7_7874

theorem complex_purely_imaginary_a_eq_3 (a : ℝ) :
  (∀ (a : ℝ), (a^2 - 2*a - 3) + (a + 1)*I = 0 + (a + 1)*I → a = 3) :=
by
  sorry

end complex_purely_imaginary_a_eq_3_l7_7874


namespace terminating_decimals_count_l7_7776

theorem terminating_decimals_count :
  let predicate (n : ℕ) := ∃ (k : ℕ), n = k * 49
  let valid_n_count := {n : ℕ | 1 ≤ n ∧ n ≤ 392 ∧ predicate n}.card
  valid_n_count = 8 := 
by
  let predicate (n : ℕ) := ∃ (k : ℕ), n = k * 49
  let valid_n_count := {n : ℕ | 1 ≤ n ∧ n ≤ 392 ∧ predicate n}.card
  sorry

end terminating_decimals_count_l7_7776


namespace nonpositive_sum_of_products_l7_7043

theorem nonpositive_sum_of_products {a b c d : ℝ} (h : a + b + c + d = 0) :
  ab + ac + ad + bc + bd + cd ≤ 0 :=
sorry

end nonpositive_sum_of_products_l7_7043


namespace quadratic_equation_terms_l7_7912

theorem quadratic_equation_terms (x : ℝ) :
  (∃ a b c : ℝ, a = 3 ∧ b = -6 ∧ c = -7 ∧ a * x^2 + b * x + c = 0) →
  (∃ (a : ℝ), a = 3 ∧ ∀ (x : ℝ), 3 * x^2 - 6 * x - 7 = a * x^2 - 6 * x - 7) ∧
  (∃ (c : ℝ), c = -7 ∧ ∀ (x : ℝ), 3 * x^2 - 6 * x - 7 = 3 * x^2 - 6 * x + c) :=
by
  sorry

end quadratic_equation_terms_l7_7912


namespace even_function_f_l7_7366

noncomputable def f (a b c x : ℝ) := a * Real.cos x + b * x^2 + c

theorem even_function_f (a b c : ℝ) (h1 : f a b c 1 = 1) : f a b c (-1) = f a b c 1 := by
  sorry

end even_function_f_l7_7366


namespace f_strictly_decreasing_l7_7334

-- Define the function f(x)
def f (x : ℝ) := (x^2 + x + 1) * exp x

-- Define the derivative of the function f(x)
def f_prime (x : ℝ) := exp x * (x^2 + 3 * x + 2)

-- State the problem in Lean to show that f is strictly decreasing in the interval (-2, -1)
theorem f_strictly_decreasing : ∀ x : ℝ, -2 < x ∧ x < -1 → f_prime x < 0 :=
by 
  {
    sorry
  }

end f_strictly_decreasing_l7_7334


namespace equilateral_triangle_area_increase_l7_7682

theorem equilateral_triangle_area_increase (A_orig : ℝ) (s : ℝ) :
  (A_orig = 36 * real.sqrt 3 ∧ A_orig = (s^2 * real.sqrt 3) / 4) →
  let s_new := s + 2 in
  let A_new := (s_new^2 * real.sqrt 3) / 4 in
  A_new - A_orig = 13 * real.sqrt 3 :=
sorry

end equilateral_triangle_area_increase_l7_7682


namespace classmates_ate_cake_l7_7096

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l7_7096


namespace total_valid_votes_l7_7899

theorem total_valid_votes (V : ℕ) (h : 0.55 * V = 0.55 * V - 250 + 250) : V = 385 :=
by sorry

end total_valid_votes_l7_7899


namespace line_parallel_plane_l7_7798

-- Definitions based on the problem statement
variable (Line Plane : Type) 
variable (m n : Line) 
variable (α β : Plane) 

-- Predicates for parallelism and perpendicularity
variables [HasParallel Line Plane] [HasSubset Line Plane]
variables [HasPerpendicular Line Plane] [HasPerpendicular Plane Plane]

-- Condition: α ⊥ β, m ⊥ β, m ⊈ α
variables (h1 : Perpendicular α β)
variables (h2 : Perpendicular m β)
variables (h3 : ¬ Subset m α)

-- Conclusion: m ∥ α
theorem line_parallel_plane : Parallel m α :=
  by
  sorry

end line_parallel_plane_l7_7798


namespace PQ_eq_QR_iff_bisectors_intersect_AC_l7_7058

variables (A B C D P Q R : Type*) [IsCyclicQuadrilateral A B C D]

-- Definitions of projections
def isProjection (X Y Z : Type*) : Prop := -- Assume some definition of projection

def onLine (X Y : Type*) : Prop := -- Assume some definition of being on the same line

axiom P_projection : isProjection D B P
axiom Q_projection : isProjection D C Q
axiom R_projection : isProjection D A R

-- Required condition for the cyclic quadrilateral
axiom cyclic_ABC (A B C : Type*) : IsCyclicQuadrilateral A B C D

-- Required proof
theorem PQ_eq_QR_iff_bisectors_intersect_AC :
  (onLine P Q R) ∧ (PQ = QR) ↔ (∃ E, angle_bisector A B C E ∧ angle_bisector A D C E ∧ E on AC) :=
sorry

end PQ_eq_QR_iff_bisectors_intersect_AC_l7_7058


namespace correct_option_is_B_l7_7622

noncomputable def smallest_absolute_value := 0

theorem correct_option_is_B :
  (∀ x : ℝ, |x| ≥ 0) ∧ |(0 : ℝ)| = 0 :=
by
  sorry

end correct_option_is_B_l7_7622


namespace infinite_solutions_system_l7_7727

theorem infinite_solutions_system : ∃ (S : set (ℚ × ℚ)), ∀ (x y : ℚ), (3 * x - 4 * y = 8 ∧ 6 * x - 8 * y = 16) ↔ (x, y) ∈ S :=
by
  sorry

end infinite_solutions_system_l7_7727


namespace tangent_line_equation_extreme_value_of_g_inequality_x1_x2_l7_7417

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := ln x - (1/2) * a * x^2 + x

-- Part I: Tangent line equation
theorem tangent_line_equation (a : ℝ) (h : a = 0) : 
  ∃ y, y = 2 * x - 1 ∧ y = 2 * (x - 1) :=
by
  sorry

-- Part II: Extreme value of g(x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a - a * x + 1

theorem extreme_value_of_g (a : ℝ) : 
  (a ≤ 0 → ¬(∃ x, ∀ y, g y a ≤ g x a)) ∧ 
  (0 < a → (∃ x, ∀ y, g y a ≤ g x a ∧ g x a = (1 / (2 * a)) - ln a)) :=
by
  sorry

-- Part III: Proving the inequality for x1 + x2
theorem inequality_x1_x2 
  (a : ℝ) (h : a = -2) (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : f x1 a + f x2 a + x1 * x2 = 0) : 
  x1 + x2 ≥ (sqrt 5 - 1) / 2 :=
by
  sorry

end tangent_line_equation_extreme_value_of_g_inequality_x1_x2_l7_7417


namespace options_implication_l7_7255

theorem options_implication (a b : ℝ) :
  ((b > 0 ∧ a < 0) ∨ (a < 0 ∧ b < 0 ∧ a > b) ∨ (a > 0 ∧ b > 0 ∧ a > b)) → (1 / a < 1 / b) :=
by sorry

end options_implication_l7_7255


namespace Ada_original_seat_2_l7_7543

def seat_number := Fin 6

structure FriendsState where
  Ada : Option seat_number
  Bea : seat_number
  Ceci : seat_number
  Dee : seat_number
  Edie : seat_number
  Fana : seat_number

variables (initial final : FriendsState)

-- Conditions
axiom Bea_moves_right_3 : final.Bea = if initial.Bea.val + 3 < 6 then ⟨initial.Bea.val + 3, by decide⟩ else ⟨(initial.Bea.val + 3) % 6, by decide⟩
axiom Ceci_moves_right_2 : final.Ceci = if initial.Ceci.val + 2 < 6 then ⟨initial.Ceci.val + 2, by decide⟩ else ⟨(initial.Ceci.val + 2) % 6, by decide⟩
axiom Dee_Edie_switch : final.Dee = initial.Edie ∧ final.Edie = initial.Dee
axiom Fana_moves_left_1 : final.Fana = if initial.Fana.val - 1 >= 0 then ⟨initial.Fana.val - 1, by decide⟩ else ⟨5, by decide⟩
axiom Ada_end_empty : final.Ada = none

-- Demonstrating each friend ends up in different seats
axiom all_different : ∀ {a b c d e f : seat_number}, List.nodup [a, b, c, d, e, f]

-- We prove Ada was initially in seat 2
theorem Ada_original_seat_2 : initial.Ada = some ⟨1, by decide⟩ :=
sorry

end Ada_original_seat_2_l7_7543


namespace distinct_three_digit_numbers_l7_7838

-- Definitions based on the conditions
def digits := {1, 2, 3, 4, 5}
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Statement of the problem to be proved
theorem distinct_three_digit_numbers : 
  (finset.card 
    (finset.filter 
      (λ (n : ℕ), 
        (∀ i j, i ≠ j → string.nth (to_string n) i ≠ string.nth (to_string n) j) ∧ 
        (∀ i, i < 2 → (string.nth (to_string n) i = some '2' ∨ string.nth (to_string n) i = some '4') →
            (string.nth (to_string n) (i + 1) ≠ some '2' ∨ string.nth (to_string n) (i + 1) ≠ some '4')) ∧ 
        (100 ≤ n ∧ n < 1000) ∧ 
        (∀ i, (string.nth (to_string n) i).is_some → (option.get (to_string n).nth i.iget).digit_character ∈ digits)
      )
      (finset.Ico 100 1000)
    )
  ) = 39 := sorry

end distinct_three_digit_numbers_l7_7838


namespace angular_speed_proportion_l7_7566

variable (w x y z : ℕ) (ω_A ω_B ω_C ω_D : ℝ) (k : ℝ)

-- Conditions on gears and angular speeds
def gears_meshed_and_velocities_constant : Prop :=
  w * ω_A = x * ω_B ∧
  w * ω_A = y * ω_C ∧
  w * ω_A = z * ω_D ∧
  ω_A = k / w ∧
  ω_B = k / x ∧
  ω_C = k / y ∧
  ω_D = k / z

-- Proportion of angular speeds
def angular_speed_proportion_correct : Prop :=
  (xyz : wyz : wxz : wxy)

-- Final proof statement
theorem angular_speed_proportion
  (h : gears_meshed_and_velocities_constant w x y z ω_A ω_B ω_C ω_D k) :
  angular_speed_proportion_correct w x y z ω_A ω_B ω_C ω_D k :=
sorry

end angular_speed_proportion_l7_7566


namespace original_shape_area_is_correct_l7_7882

/-!
If the oblique projection of a horizontal planar shape is an isosceles trapezoid 
with a base angle of 45 degrees, bases and top both of length 1, prove that the 
area of the original planar shape is 2 + sqrt 2.
-/

def base_angle : ℝ := 45
def top_length : ℝ := 1
def base_length : ℝ := 1

-- The original shape is a right trapezoid
def upper_base : ℝ := 1
def lower_base : ℝ := 1 + Real.sqrt 2
def height : ℝ := 2

def original_area : ℝ := (upper_base + lower_base) / 2 * height

theorem original_shape_area_is_correct :
  original_area = 2 + Real.sqrt 2 := by
  sorry

end original_shape_area_is_correct_l7_7882


namespace ratio_pokemon_cards_l7_7963

noncomputable def ratio_pokemon_cards_proof :
  (Nicole : ℕ) →
  (Cindy : ℕ) →
  (Rex_remaining : ℕ) →
  Prop :=
  λ Nicole Cindy Rex_remaining,
    (Nicole = 400) →
    (Cindy = 2 * Nicole) →
    (Rex_remaining = 150) →
    (let Rex := 4 * Rex_remaining in
     let combined_total := Nicole + Cindy in
     (Rex // Nat.gcd Rex combined_total) = 1 ∧ (combined_total // Nat.gcd Rex combined_total) = 2)

theorem ratio_pokemon_cards :
  ratio_pokemon_cards_proof 400 (2 * 400) 150 :=
by
    unfold ratio_pokemon_cards_proof
    intros
    sorry

end ratio_pokemon_cards_l7_7963


namespace weight_difference_l7_7522

theorem weight_difference :
  let Box_A := 2.4
  let Box_B := 5.3
  let Box_C := 13.7
  let Box_D := 7.1
  let Box_E := 10.2
  let Box_F := 3.6
  let Box_G := 9.5
  max Box_A (max Box_B (max Box_C (max Box_D (max Box_E (max Box_F Box_G))))) -
  min Box_A (min Box_B (min Box_C (min Box_D (min Box_E (min Box_F Box_G))))) = 11.3 :=
by
  sorry

end weight_difference_l7_7522


namespace required_equation_l7_7739

-- Define the given lines
def line1 (x y : ℝ) : Prop := 2 * x - y = 0
def line2 (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the equation to be proven for the line through the intersection point and perpendicular to perp_line
def required_line (x y : ℝ) : Prop := x - 2 * y + 6 = 0

-- Define the predicate that states a point (2, 4) lies on line1 and line2
def point_intersect (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- The main theorem to be proven in Lean 4
theorem required_equation : 
  point_intersect 2 4 ∧ perp_line 2 4 → required_line 2 4 := by
  sorry

end required_equation_l7_7739


namespace deepak_and_wife_meet_time_correct_l7_7172

noncomputable def time_to_meet (C : ℝ) (Deepak_speed_kmph : ℝ) (Wife_speed_kmph : ℝ) : ℝ :=
  let Deepak_speed_mpm := Deepak_speed_kmph * 1000 / 60
  let Wife_speed_mpm := Wife_speed_kmph * 1000 / 60
  let relative_speed := Deepak_speed_mpm + Wife_speed_mpm
  C / relative_speed

theorem deepak_and_wife_meet_time_correct : 
  time_to_meet 561 4.5 3.75 = 561 / ((4.5 * 1000 / 60) + (3.75 * 1000 / 60)) :=
by
  simp [time_to_meet]
  sorry

end deepak_and_wife_meet_time_correct_l7_7172


namespace lcm_from_1_to_12_eq_27720_l7_7237

theorem lcm_from_1_to_12_eq_27720 : nat.lcm (finset.range 12).succ = 27720 :=
  sorry

end lcm_from_1_to_12_eq_27720_l7_7237


namespace find_a_l7_7410

noncomputable def curve (x a : ℝ) : ℝ := 1/x + (Real.log x)/a
noncomputable def curve_derivative (x a : ℝ) : ℝ := 
  (-1/(x^2)) + (1/(a * x))

theorem find_a (a : ℝ) : 
  (curve_derivative 1 a = 3/2) ∧ ((∃ l : ℝ, curve 1 a = l) → ∃ m : ℝ, m * (-2/3) = -1)  → a = 2/5 :=
by
  sorry

end find_a_l7_7410


namespace xiaoming_pens_l7_7258

theorem xiaoming_pens (P M : ℝ) (hP : P > 0) (hM : M > 0) :
  (M / (7 / 8 * P) - M / P = 13) → (M / P = 91) := 
by
  sorry

end xiaoming_pens_l7_7258


namespace exists_non_cool_graph_l7_7663

-- Define what it means for a graph to be "cool".
def cool_graph (n : ℕ) (G : Type*) [graph G] : Prop :=
  ∃ (label : G → ℕ) (D : finset ℕ),
  (∀ v1 v2 : G, label v1 ≠ label v2 → label v1 ∈ finset.range (n * n / 4) ∧ 
                label v2 ∈ finset.range (n * n / 4) ∧ 
                (v1 ≠ v2 ↔ (abs (label v1 - label v2) ∈ D)))

-- Main theorem stating the existence of a non-"cool" graph for sufficiently large n.
theorem exists_non_cool_graph (n : ℕ) (hn : n > 100): ∃ (G : Type*) [graph G], ¬ cool_graph n G :=
by
  sorry

end exists_non_cool_graph_l7_7663


namespace find_angle_D_l7_7001

variables (A B C D angle : ℝ)

-- Assumptions based on the problem statement
axiom sum_A_B : A + B = 140
axiom C_eq_D : C = D

-- The claim we aim to prove
theorem find_angle_D (h₁ : A + B = 140) (h₂: C = D): D = 20 :=
by {
    sorry 
}

end find_angle_D_l7_7001


namespace minimum_h9_tenuous_l7_7684

theorem minimum_h9_tenuous : 
  (∃ h : ℕ → ℕ, (∀ x y : ℕ, 0 < x → 0 < y → h(x) + h(y) > y^2) ∧ 
                h(1) + h(2) + h(3) + h(4) + h(5) + h(6) + h(7) + h(8) + h(9) + h(10) + 
                h(11) + h(12) + h(13) + h(14) + h(15) ≤ 1625) 
  → ∃ b : ℕ, b = 50 :=
by
  sorry

end minimum_h9_tenuous_l7_7684


namespace exists_even_sum_l7_7057

-- Define the sets and conditions
variables (n : ℕ) (A B : set ℕ)
hypothesis (H1 : 0 < n)
hypothesis (H2 : ∀ a ∈ A, a ≥ 0 ∧ a ≤ n)
hypothesis (H3 : ∀ b ∈ B, b ≥ 0 ∧ b ≤ n)
hypothesis (H4 : A.card + B.card ≥ n + 2)

-- Statement to prove
theorem exists_even_sum : ∃ a ∈ A, ∃ b ∈ B, (a + b) % 2 = 0 :=
by sorry

end exists_even_sum_l7_7057


namespace minimum_distance_is_1500_l7_7003

-- Define Point A
def A : ℝ × ℝ := (0, 3)

-- Define Point B and its reflection B'
def B : ℝ × ℝ := (12, 6)
def B' : ℝ × ℝ := (12, -6)

-- Define the horizontal distance and vertical distance from A to B'
def horizontalDistance : ℝ := 1200
def verticalDistance : ℝ := 300 + 600

-- Define the function to calculate the Euclidean distance
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- State the theorem to be proved
theorem minimum_distance_is_1500 :
  round (distance A B') = 1500 :=
by
  sorry

end minimum_distance_is_1500_l7_7003


namespace distinct_divisors_count_l7_7959

noncomputable def M : ℕ := sorry -- Define M such that it has exactly 10 divisors
noncomputable def Z : ℕ := sorry -- Define Z such that it has exactly 9 divisors
def is_divisor (n d : ℕ) : Prop := d ∣ n
def num_distinct_divisors : ℕ :=
  ((finset.image (λ x, x) ((finset.filter (is_divisor M) (finset.range (M + 1))))) ∪
  (finset.image (λ x, x) ((finset.filter (is_divisor Z) (finset.range (Z + 1)))))).card

theorem distinct_divisors_count 
  (hM : (finset.filter (is_divisor M) (finset.range (M + 1))).card = 10)
  (hZ : (finset.filter (is_divisor Z) (finset.range (Z + 1))).card = 9)
  (h_common_divisor : is_divisor M 6 ∧ is_divisor Z 6) : 
  num_distinct_divisors = 13 := by sorry

end distinct_divisors_count_l7_7959


namespace tim_is_on_fourth_bicycle_l7_7582

-- Definitions of the participants
inductive Person
| Phil | Will | Tim | Jill | Bill
deriving DecidableEq

-- The arrangement of bicycles
def arrangement : ℕ → Person
| 1 => sorry -- This needs to be deduced
| 2 => Person.Phil
| 3 => sorry -- This needs to be deduced
| 4 => sorry -- This needs to be deduced
| 5 => sorry -- This needs to be deduced

-- The given conditions
axiom will_ahead_of_bill : ∀ i j : ℕ, arrangement i = Person.Will → arrangement j = Person.Bill → i < j
axiom two_bicycles_between_tim_and_phil : ∀ i : ℕ, arrangement i = Person.Tim → (i = 4 ∨ i = 5)

-- The statement to prove
theorem tim_is_on_fourth_bicycle : arrangement 4 = Person.Tim :=
by
  -- Proof goes here
  sorry

end tim_is_on_fourth_bicycle_l7_7582


namespace vertex_position_not_unique_l7_7915

structure Triangle :=
(A B C : Point)
(mid_D : Point)
(mid_E : Point)
(mid_F : Point)
(D_eq_mid_AB : midpoint D A B)
(E_eq_mid_AC : midpoint E A C)
(F_eq_mid_AL : midpoint F A L)
(distance_DF : dist D F = 1)
(distance_EF : dist E F = 2)
(horizontal_DF : is_horizontal D F)
(above_A_DF : above_line A D F)

noncomputable def restore_vertex_position (Δ : Triangle) : Vertex :=
  sorry

/-- Prove that it's not possible to uniquely determine the location of at least one vertex of triangle ABC just from the given conditions. -/
theorem vertex_position_not_unique (Δ : Triangle) : ¬ ∃ A : Vertex, restore_vertex_position Δ = A :=
sorry

end vertex_position_not_unique_l7_7915


namespace probability_product_multiple_of_60_l7_7195

theorem probability_product_multiple_of_60 :
  (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 ∧ (60 ∣ (a * b))) →
  (∑ a b : ℕ, if (60 ∣ (a * b)) then 1 else 0 = 732) →
  true := sorry

end probability_product_multiple_of_60_l7_7195


namespace segment_length_l7_7214

theorem segment_length (x : ℝ) (h : |x - (27)^(1/3)| = 5) : ∃ a b : ℝ, (a = 8 ∧ b = -2 ∨ a = -2 ∧ b = 8) ∧ real.dist a b = 10 :=
by
  use [8, -2] -- providing the endpoints explicitly
  split
  -- prove that these are the correct endpoints
  · left; exact ⟨rfl, rfl⟩
  -- prove the distance is 10
  · apply real.dist_eq; linarith
  

end segment_length_l7_7214


namespace rectangle_area_equals_l7_7291

def length_triangle_sides := {a : ℚ // a = 7} ∧ {b : ℚ // b = 10} ∧ {c : ℚ // c = 11}
def perimeter_triangle (a b c : ℚ) := a + b + c
def length_rectangle_perimeter := 2

noncomputable def area_rectangle (perimeter : ℚ) (width : ℚ) := width * (2 * width)

theorem rectangle_area_equals:
  ∀ (a b c : ℚ) (width : ℚ)
  (h1 : a = 7) 
  (h2 : b = 10) 
  (h3 : c = 11) 
  (h4 : 2 * (2 * width + width) = perimeter_triangle a b c), 
  area_rectangle (perimeter_triangle a b c) width = 392 / 9 :=
by
  sorry

end rectangle_area_equals_l7_7291


namespace average_discount_is_23_07_l7_7065

def cost_price1 : ℝ := 540
def cost_price2 : ℝ := 660
def cost_price3 : ℝ := 780

def markup1 : ℝ := 0.15
def markup2 : ℝ := 0.20
def markup3 : ℝ := 0.25

def selling_price1 : ℝ := 496.80
def selling_price2 : ℝ := 600
def selling_price3 : ℝ := 740

def marked_price1 : ℝ := cost_price1 * (1 + markup1)
def marked_price2 : ℝ := cost_price2 * (1 + markup2)
def marked_price3 : ℝ := cost_price3 * (1 + markup3)

def discount1 : ℝ := marked_price1 - selling_price1
def discount2 : ℝ := marked_price2 - selling_price2
def discount3 : ℝ := marked_price3 - selling_price3

def total_marked_price : ℝ := marked_price1 + marked_price2 + marked_price3
def total_discount : ℝ := discount1 + discount2 + discount3

def average_discount_percentage : ℝ := (total_discount / total_marked_price) * 100

theorem average_discount_is_23_07 : average_discount_percentage = 23.07 := by
  sorry

end average_discount_is_23_07_l7_7065


namespace sum_of_roots_l7_7184

theorem sum_of_roots (r s t : ℝ) (h : 3 * r * s * t - 9 * (r * s + s * t + t * r) - 28 * (r + s + t) + 12 = 0) : r + s + t = 3 :=
by sorry

end sum_of_roots_l7_7184


namespace average_age_of_inhabitants_l7_7569

theorem average_age_of_inhabitants (H M : ℕ) (avg_age_men avg_age_women : ℕ)
  (ratio_condition : 2 * M = 3 * H)
  (men_avg_age_condition : avg_age_men = 37)
  (women_avg_age_condition : avg_age_women = 42) :
  ((H * 37) + (M * 42)) / (H + M) = 40 :=
by
  sorry

end average_age_of_inhabitants_l7_7569


namespace range_of_z_l7_7045

variable (x y z : ℝ)

theorem range_of_z (hx : x ≥ 0) (hy : y ≥ x) (hxy : 4*x + 3*y ≤ 12) 
(hz : z = (x + 2 * y + 3) / (x + 1)) : 
2 ≤ z ∧ z ≤ 6 :=
sorry

end range_of_z_l7_7045


namespace min_3x_add_4y_on_circle_l7_7811

noncomputable def min_value (f : ℝ → ℝ → ℝ) (C : set (ℝ × ℝ)) : ℝ :=
  Inf (f '' C)

theorem min_3x_add_4y_on_circle :
  min_value (λ x y, 3*x + 4*y) {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} = -5 :=
by
  sorry

end min_3x_add_4y_on_circle_l7_7811


namespace find_bisecting_vector_b_l7_7040

noncomputable def vector_a : ℝ × ℝ × ℝ := (3, 1, -5)
noncomputable def vector_b : ℝ × ℝ × ℝ := (619 / 931, 1, 925 / 931)
noncomputable def vector_c : ℝ × ℝ × ℝ := (-3, 1, 2)

-- Function to compute dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Function to compute norm of a vector
def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem find_bisecting_vector_b :
  (∃ t : ℝ, (619 / 931, 1, 925 / 931) = (3 - 6 * t, 1, -5 + 7 * t)) ∧
  dot_product vector_a vector_b / (norm vector_a * norm vector_b) =
  dot_product vector_b vector_c / (norm vector_b * norm vector_c) :=
  sorry

end find_bisecting_vector_b_l7_7040


namespace seq_product_100_l7_7913

noncomputable def seq (n : ℕ) : ℚ :=
  if n = 1 then 1 / 2 else 1 / (2 - seq (n - 1))

noncomputable def product (n : ℕ) : ℚ :=
  (finset.range n).prod (λ i, seq (i + 1))

theorem seq_product_100 :
  product 100 = 1 / 101 :=
by
  sorry

end seq_product_100_l7_7913


namespace constant_a_of_square_binomial_l7_7848

theorem constant_a_of_square_binomial (a : ℝ) : (∃ b : ℝ, (9x^2 + 27x + a) = (3x + b)^2) -> a = 81 / 4 :=
by
  intro h
  sorry

end constant_a_of_square_binomial_l7_7848


namespace diameter_of_large_circle_l7_7544

theorem diameter_of_large_circle :
  ∀ (R : ℝ), (
    (∀ r, r = 4 ∧ 
         ∀ θ : fin 6, 
         (∃ p, p ∈ ℝ^2 ∧ dist p.origin ≤ R ∧ dist p.(cycle θ).center = 8) ∧
         (∀ y : ℝ, y = 0 → ∃ x y', (x, y') ∈ ℝ^2 ∧ y - y' = r)
    )) → 2*R = 20 :=
by
  intros R conditions
  sorry

end diameter_of_large_circle_l7_7544


namespace inequality_proof_l7_7972

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a + 1) * (b + 1) * (a + c) * (b + c) > 16 * a * b * c :=
by
  sorry

end inequality_proof_l7_7972


namespace coach_check_counts_l7_7271

-- Definitions according to the conditions in a)
def num_runners_blue : ℕ := 5
def num_runners_red : ℕ := 5
def speed_min : ℝ := 9 -- in km/h
def speed_max : ℝ := 12 -- in km/h

-- Define the problem statement
theorem coach_check_counts (n : ℕ) (m : ℕ)
  (speeds_blue : Fin n → ℝ)
  (speeds_red : Fin m → ℝ)
  (hs1 : n = num_runners_blue)
  (hs2 : m = num_runners_red)
  (hs3 : ∀ i, speeds_blue i > speed_min ∧ speeds_blue i < speed_max)
  (hs4 : ∀ j, speeds_red j > speed_min ∧ speeds_red j < speed_max)
  (hs5 : Function.Injective speeds_blue)
  (hs6 : Function.Injective speeds_red) :
  ∑ i in Finset.univ.product Finset.univ, ite ((i.1 ≠ i.2)) 2 0 = 50 :=
by {
  -- Problem statement in Lean 4 format, proof is not required and hence skipped.
  sorry
}

end coach_check_counts_l7_7271


namespace sum_first_n_terms_arithmetic_sequence_l7_7511

theorem sum_first_n_terms_arithmetic_sequence 
  (S : ℕ → ℕ) (m : ℕ) (h1 : S m = 2) (h2 : S (2 * m) = 10) :
  S (3 * m) = 24 :=
sorry

end sum_first_n_terms_arithmetic_sequence_l7_7511


namespace expansion_term_count_l7_7314

theorem expansion_term_count :
  ∃ (n k : ℕ), (n = 12) ∧ (k = 4) ∧ (finset.card (finset.filter (λ (s : fin₄ → ℕ),
    s.sum = n) (finset.of_multiset (multiset.replicate n + fin₄))) = 455) :=
begin
  use [12, 4],
  split,
  { refl },
  split,
  { refl },
  sorry
end

end expansion_term_count_l7_7314


namespace infinite_even_odd_l7_7531

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) = (3 * a n) / 2

theorem infinite_even_odd (a : ℕ → ℕ) (h : sequence a) : 
  ∃ S_even S_odd : set ℕ, set.infinite S_even ∧ set.infinite S_odd ∧ 
  (∀ n ∈ S_even, even (a n)) ∧ (∀ n ∈ S_odd, odd (a n)) :=
sorry

end infinite_even_odd_l7_7531


namespace population_2003_is_110_l7_7998

noncomputable def population (n : ℕ) : ℕ
def k : ℝ

-- Conditions
axiom condition1 (n : ℕ) : population (n + 2) - population n = k * (population (n + 1) + 10)
axiom pop_2001 : population 2001 = 50
axiom pop_2002 : population 2002 = 75
axiom pop_2004 : population 2004 = 160

-- Desired population in 2003
theorem population_2003_is_110 : population 2003 = 110 := 
sorry

end population_2003_is_110_l7_7998


namespace weight_difference_of_red_and_yellow_glass_marbles_l7_7577

theorem weight_difference_of_red_and_yellow_glass_marbles :
  ∀ (total_marbles : ℕ) (yellow_glass_marbles : ℕ) (blue_and_red_ratio_b : ℕ) (blue_and_red_ratio_r : ℕ) 
  (yellow_glass_marble_weight : ℕ) (red_to_yellow_weight_ratio : ℕ)
  (remaining_marbles := total_marbles - yellow_glass_marbles)
  (total_blue_and_red_sets := remaining_marbles / (blue_and_red_ratio_b + blue_and_red_ratio_r))
  (red_glass_marbles := blue_and_red_ratio_r * total_blue_and_red_sets)
  (total_red_glass_weight := red_glass_marbles * (yellow_glass_marble_weight * red_to_yellow_weight_ratio)),
  total_marbles = 19 → 
  yellow_glass_marbles = 5 → 
  blue_and_red_ratio_b = 3 → 
  blue_and_red_ratio_r = 4 → 
  yellow_glass_marble_weight = 8 → 
  red_to_yellow_weight_ratio = 2 →
  total_red_glass_weight - yellow_glass_marble_weight * yellow_glass_marbles = 120 :=
by
  intros total_marbles yellow_glass_marbles blue_and_red_ratio_b blue_and_red_ratio_r yellow_glass_marble_weight red_to_yellow_weight_ratio
  simp [total_marbles, yellow_glass_marbles, blue_and_red_ratio_b, blue_and_red_ratio_r, yellow_glass_marble_weight, red_to_yellow_weight_ratio]
  sorry

end weight_difference_of_red_and_yellow_glass_marbles_l7_7577


namespace number_of_different_grids_l7_7729

-- Definitions of the disks and the grid
def blue_disks : Nat := 5
def red_disks : Nat := 2
def yellow_disk : Nat := 1
def total_disks : Nat := blue_disks + red_disks + yellow_disk

-- Definition of valid placements
def valid_red_disk_placements (P Q R S T U V W : Bool) : Nat :=
  18

-- Proof statement to be proven
theorem number_of_different_grids :
  total_disks = 8 ∧ valid_red_disk_placements true true true true true true true true * 6 = 108 :=
begin
  sorry
end

end number_of_different_grids_l7_7729


namespace find_a1000_l7_7002

noncomputable def seq (a : ℕ → ℤ) : Prop :=
a 1 = 1009 ∧
a 2 = 1010 ∧
(∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n)

theorem find_a1000 (a : ℕ → ℤ) (h : seq a) : a 1000 = 1675 :=
sorry

end find_a1000_l7_7002


namespace total_balls_l7_7452

def black_balls : ℕ := 8
def white_balls : ℕ := 6 * black_balls
theorem total_balls : white_balls + black_balls = 56 := 
by 
  sorry

end total_balls_l7_7452


namespace af_vector_l7_7378

-- Define the given cube and points
variables (A B C D A' B' C' D' E F : Type)
variables [AddCommGroup A] [vector_space ℝ A]
variables [AddCommGroup B] [vector_space ℝ B]
variables [AddCommGroup C] [vector_space ℝ C]
variables [AddCommGroup D] [vector_space ℝ D]
variables [AddCommGroup A'] [vector_space ℝ A']
variables [AddCommGroup B'] [vector_space ℝ B']
variables [AddCommGroup C'] [vector_space ℝ C']
variables [AddCommGroup D'] [vector_space ℝ D']
variables [AddCommGroup E] [vector_space ℝ E]
variables [AddCommGroup F] [vector_space ℝ F]

-- Define the vectors
variables (AA' AB AD A'C' : A → A)
variables (AE A'B' A'D' AF EF : E → E)
variables [is_midpoint A'C' AE E]
variables [is_trisection_point AE F]
variables [eq_of_vectors AF (1 / 2 * EF)]

-- Statement to prove
theorem af_vector :
  AA' + (1 / 6) * AB + (1 / 6) * AD = (1 / 3) * AA' + (1 / 6) * AB + (1 / 6) * AD := by
    sorry

end af_vector_l7_7378


namespace train_length_l7_7676

-- Definitions based on the conditions
def speed_kmh : ℝ := 27
def time_seconds : ℝ := 16

-- Conversion from km/h to m/s
def speed_ms : ℝ := speed_kmh * (5/18)

-- Goal: Prove the length of the train in meters
theorem train_length :
  (speed_ms * time_seconds = 120) :=
by
  -- Placeholder for the proof
  sorry

end train_length_l7_7676


namespace find_distance_l7_7525

variable (y : ℚ) -- The circumference of the bicycle wheel
variable (x : ℚ) -- The distance between the village and the field

-- Condition 1: The circumference of the truck's wheel is 4/3 of the bicycle's wheel
def circum_truck_eq : Prop := (4 / 3 : ℚ) * y = y

-- Condition 2: The circumference of the truck's wheel is 2 meters shorter than the tractor's track
def circum_truck_less : Prop := (4 / 3 : ℚ) * y + 2 = y + 2

-- Condition 3: Truck's wheel makes 100 fewer revolutions than the bicycle's wheel
def truck_100_fewer : Prop := x / ((4 / 3 : ℚ) * y) = (x / y) - 100

-- Condition 4: Truck's wheel makes 150 more revolutions than the tractor track
def truck_150_more : Prop := x / ((4 / 3 : ℚ) * y) = (x / ((4 / 3 : ℚ) * y + 2)) + 150

theorem find_distance (y : ℚ) (x : ℚ) :
  circum_truck_eq y →
  circum_truck_less y →
  truck_100_fewer x y →
  truck_150_more x y →
  x = 600 :=
by
  intros
  sorry

end find_distance_l7_7525


namespace function_behaviour_l7_7819

theorem function_behaviour (a : ℝ) (h : a ≠ 0) :
  ¬ ((a * (-2)^2 + 2 * a * (-2) + 1 > a * (-1)^2 + 2 * a * (-1) + 1) ∧
     (a * (-1)^2 + 2 * a * (-1) + 1 > a * 0^2 + 2 * a * 0 + 1)) :=
by
  sorry

end function_behaviour_l7_7819


namespace muffin_mix_buyers_l7_7648

noncomputable def total_buyers : ℕ := 100
noncomputable def cake_mix_buyers : ℕ := 50
noncomputable def both_buyers : ℕ := 16
noncomputable def neither_probability : ℝ := 0.26

theorem muffin_mix_buyers : ∃ M : ℕ, M = 40 :=
by
  let neither_buyers := (neither_probability * total_buyers)
  have neither_buyers_eq: neither_buyers = 26, from sorry
  let total_cake_or_muffin := total_buyers - neither_buyers
  have total_cake_or_muffin_eq: total_cake_or_muffin = 74, from sorry
  let muffin_mix_buyers := total_cake_or_muffin - cake_mix_buyers + both_buyers
  have muffin_mix_buyers_eq: muffin_mix_buyers = 40, from sorry
  exact ⟨muffin_mix_buyers, muffin_mix_buyers_eq⟩

end muffin_mix_buyers_l7_7648


namespace find_middle_and_oldest_sons_l7_7657

-- Defining the conditions
def youngest_age : ℕ := 2
def father_age : ℕ := 33
def father_age_in_12_years : ℕ := father_age + 12
def youngest_age_in_12_years : ℕ := youngest_age + 12

-- Lean theorem statement to find the ages of the middle and oldest sons
theorem find_middle_and_oldest_sons (y z : ℕ) (h1 : father_age_in_12_years = (youngest_age_in_12_years + 12 + y + 12 + z + 12)) :
  y = 3 ∧ z = 4 :=
sorry

end find_middle_and_oldest_sons_l7_7657


namespace third_side_of_triangle_l7_7967

theorem third_side_of_triangle (a b : ℝ) (γ : ℝ) (x : ℝ) 
  (ha : a = 6) (hb : b = 2 * Real.sqrt 7) (hγ : γ = Real.pi / 3) :
  x = 2 ∨ x = 4 :=
by 
  sorry

end third_side_of_triangle_l7_7967


namespace anne_wandering_time_l7_7853

theorem anne_wandering_time (distance speed : ℝ) (h_dist : distance = 3.0) (h_speed : speed = 2.0) : 
  distance / speed = 1.5 :=
by
  rw [h_dist, h_speed]
  norm_num

end anne_wandering_time_l7_7853


namespace probability_is_4_over_5_l7_7259

variable (total_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ)
variable (total_balls_eq : total_balls = 60) (red_balls_eq : red_balls = 5) (purple_balls_eq : purple_balls = 7)

def probability_neither_red_nor_purple : ℚ :=
  let favorable_outcomes := total_balls - (red_balls + purple_balls)
  let total_outcomes := total_balls
  favorable_outcomes / total_outcomes

theorem probability_is_4_over_5 :
  probability_neither_red_nor_purple total_balls red_balls purple_balls = 4 / 5 :=
by
  have h1: total_balls = 60 := total_balls_eq
  have h2: red_balls = 5 := red_balls_eq
  have h3: purple_balls = 7 := purple_balls_eq
  sorry

end probability_is_4_over_5_l7_7259


namespace trapezoid_sides_l7_7551

-- The given conditions: area = 8 cm^2, base angle = 30 degrees
def isosceles_trapezoid_circumscribed (a b c R : ℝ) (area : ℝ) (angle : ℝ) :=
  -- Conditions of the problem
  area = 8 ∧ 
  angle = π / 6 ∧
  -- Definitions for an isosceles trapezoid and the properties of the circle
  2 * R = c / 2 ∧
  area = (1/2) * (a + b) * (2 * R)

-- The proof goal: determine the sides of the trapezoid
theorem trapezoid_sides :
  ∃ (a b c : ℝ),
    isosceles_trapezoid_circumscribed a b c (c / 4) 8 (π / 6) ∧
    c = 4 ∧
    a = 4 - 2 * real.sqrt 3 ∧
    b = 4 + 2 * real.sqrt 3 :=
by { sorry }

end trapezoid_sides_l7_7551


namespace pq_difference_l7_7262

theorem pq_difference (p q : ℝ) (h1 : 3 / p = 6) (h2 : 3 / q = 15) : p - q = 3 / 10 := by
  sorry

end pq_difference_l7_7262


namespace olivia_new_premium_l7_7076

-- Define the initial premium and the percentage increase for accidents
def initial_premium : ℕ := 125
def accident_percentage_increase : ℝ := 0.12
def ticket_increase : ℕ := 7
def late_payment_increase : ℕ := 15

-- Define the number of accidents, tickets, and late payments
def number_of_accidents : ℕ := 2
def number_of_tickets : ℕ := 4
def number_of_late_payments : ℕ := 3

-- Define the total increase due to accidents, tickets, and late payments
def total_increase_accidents : ℕ := (accident_percentage_increase * initial_premium).to_nat * number_of_accidents
def total_increase_tickets : ℕ := ticket_increase * number_of_tickets
def total_increase_late_payments : ℕ := late_payment_increase * number_of_late_payments

-- Define Olivia's new insurance premium
def new_premium : ℕ := initial_premium + total_increase_accidents + total_increase_tickets + total_increase_late_payments

-- Prove that Olivia's new insurance premium is $228/month
theorem olivia_new_premium : new_premium = 228 := by
  sorry

end olivia_new_premium_l7_7076


namespace find_slope_l7_7513

theorem find_slope (k : ℝ) :
  (∀ x y : ℝ, y = -2 * x + 3 → y = k * x + 4 → (x, y) = (1, 1)) → k = -3 :=
by
  sorry

end find_slope_l7_7513


namespace women_in_first_class_l7_7545

theorem women_in_first_class (P : ℕ) (W_perc F_perc : ℝ) : 
  P = 200 → W_perc = 0.6 → F_perc = 0.1 → 
  (P * W_perc * F_perc).to_nat = 12 :=
by
  intros hP hW_perc hF_perc
  sorry

end women_in_first_class_l7_7545


namespace lcm_from_1_to_12_eq_27720_l7_7234

theorem lcm_from_1_to_12_eq_27720 : nat.lcm (finset.range 12).succ = 27720 :=
  sorry

end lcm_from_1_to_12_eq_27720_l7_7234


namespace length_MN_range_l7_7304

-- Define the basic properties of square and coordinates
structure Square :=
  (side : ℝ)
  (angle_between_faces : ℝ) -- in radians

-- Define properties of points on segments
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the conditions of the problem
def squareABCD : Square :=
{ side := 2,
  angle_between_faces := 0 }
  
def squareABEF : Square :=
{ side := 2,
  angle_between_faces := (real.pi / 3) }  -- 60 degrees in radians

def M : Point := { x := by sorry, y := by sorry }
def N : Point := { x := by sorry, y := by sorry }

-- AM = FN property
axiom AM_eq_FN : (abs (M.x - A.x) + abs (M.y - A.y)) = (abs (N.x - F.x) + abs (N.y - F.y))

-- We need to prove the length of segment MN falls within [1, 2]
theorem length_MN_range : 1 ≤ sqrt ((M.x - N.x)^2 + (M.y - N.y)^2) ∧ sqrt ((M.x - N.x)^2 + (M.y - N.y)^2) ≤ 2 :=
by
  sorry

end length_MN_range_l7_7304


namespace proof_smallest_lcm_1_to_12_l7_7224

noncomputable def smallest_lcm_1_to_12 : ℕ := 27720

theorem proof_smallest_lcm_1_to_12 :
  ∀ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ↔ n = 27720 :=
by
  sorry

end proof_smallest_lcm_1_to_12_l7_7224


namespace solve_problem_1_solve_problem_2_l7_7802

open Set

variable (I A B : Set ℕ) (h_univ : Fintype.card I = 25)
  (hA : Fintype.card A = 18)
  (hB : Fintype.card B = 10)
  (hASub : A ⊆ I)
  (hBSub : B ⊆ I) 

theorem solve_problem_1 : 18 ≤ Fintype.card (A ∪ B) ∧ Fintype.card (A ∪ B) ≤ 25 := 
  sorry

theorem solve_problem_2 : 3 ≤ Fintype.card (A ∩ B) ∧ Fintype.card (A ∩ B) ≤ 10 := 
  sorry

end solve_problem_1_solve_problem_2_l7_7802


namespace eggs_to_buy_l7_7073

theorem eggs_to_buy (total_eggs_needed : ℕ) (eggs_given_by_Andrew : ℕ) (result : ℕ) :
  total_eggs_needed = 222 ∧ eggs_given_by_Andrew = 155 ∧ result = 222 - 155 → result = 67 :=
by
  intros h
  rw [←h.2.2]
  sorry

end eggs_to_buy_l7_7073


namespace classmates_ate_cake_l7_7086

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l7_7086


namespace repeating_six_to_fraction_l7_7758

-- Define the infinite geometric series representation of 0.666...
def infinite_geometric_series (n : ℕ) : ℝ := 6 / (10 ^ n)

-- Define the sum of the infinite geometric series for 0.666...
def sum_infinite_geometric_series : ℝ :=
  ∑' n, infinite_geometric_series n

-- Formally state the problem to prove that 0.666... equals 2/3
theorem repeating_six_to_fraction : sum_infinite_geometric_series = 2 / 3 :=
by
  -- Proof goes here, but for now we use sorry to denote it will be completed later
  sorry

end repeating_six_to_fraction_l7_7758


namespace find_other_integer_l7_7514

theorem find_other_integer (x y : ℤ) (h1 : 3 * x + 2 * y = 85) (h2 : x = 19 ∨ y = 19) : y = 14 ∨ x = 14 :=
  sorry

end find_other_integer_l7_7514


namespace interval_sum_l7_7406

theorem interval_sum (m n : ℚ) (h : ∀ x : ℚ, m < x ∧ x < n ↔ (mx - 1) / (x + 3) > 0) :
  m + n = -10 / 3 :=
sorry

end interval_sum_l7_7406


namespace two_pow_n_plus_one_divisible_by_three_l7_7330

-- defining what it means to be an odd number
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- stating the main theorem in Lean
theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h_pos : 0 < n) : (2^n + 1) % 3 = 0 ↔ is_odd n :=
by sorry

end two_pow_n_plus_one_divisible_by_three_l7_7330


namespace flipping_square_area_l7_7628

theorem flipping_square_area : 
  ∃ (s : ℝ), s = 1 ∧ 
  (let triangle_area := (s * s) / 2 in
   let sector_area := (π * (s^2 + s^2)) / 4 in
   2 * triangle_area + sector_area = 1 + π / 2) :=
sorry

end flipping_square_area_l7_7628


namespace number_of_possible_B_l7_7512

open Set

def universal_set : Set ℤ := {x | -2 ≤ x ∧ x ≤ 4}
def A : Set ℤ := {-1, 0, 1, 2, 3}
def complement_A : Set ℤ := universal_set \ A
def B (s : Set ℤ) := s ⊆ complement_A

theorem number_of_possible_B : ∃ n : ℕ, n = 4 ∧ ∀ (B : Set ℤ), B ⊆ complement_A → ∃ (f : Finset (Set ℤ)), f.card = n :=
by
  have : complement_A = {-2, 4} := by sorry
  use 4
  split
  · rfl
  · intro B hB
    let possible_subsets := {∅, {-2}, {4}, {-2, 4}}
    use (possible_subsets.to_finset)
    have card_poss_sub : (possible_subsets.to_finset).card = 4 := by sorry
    exact card_poss_sub

end number_of_possible_B_l7_7512


namespace switch_connections_l7_7733

theorem switch_connections (n m : ℕ) (h1 : n = 15) (h2 : m = 4) : (n * m) / 2 = 30 := by
  have h3 : n * m = 60 := by
    rw [h1, h2]
    norm_num
  rw [h3]
  norm_num
  done
  sorry

end switch_connections_l7_7733


namespace cake_eating_classmates_l7_7110

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l7_7110


namespace probability_of_selected_number_probability_of_selected_number_in_given_interval_l7_7866

noncomputable def probability_in_interval (a b c : ℝ) (hab : 0 < a) (hbc : b < c) : ℝ :=
  (b - a) / (c - a)

theorem probability_of_selected_number (x : ℝ) (hx : 0 < x) (hx_le : x < 1/2) : 
  probability_in_interval 0 (1/3) (1/2) 0 lt_one_half = 2/3 := 
by
  have p := probability_in_interval 0 (1/3) (1/2) 0 (by norm_num : 1 < 2)
  norm_num at p
  exact p

-- Helper theorem to convert the original question
theorem probability_of_selected_number_in_given_interval :
  ∀ x, 0 < x ∧ x < 1/2 → x < 1/3 → probability_of_selected_number x 0 (by norm_num) = 2/3 :=
by
  intros x _ _
  exact probability_of_selected_number x 0 (by norm_num)

-- Sorry to skip the proof as requested
sorry

end probability_of_selected_number_probability_of_selected_number_in_given_interval_l7_7866


namespace linear_function_no_third_quadrant_l7_7881

theorem linear_function_no_third_quadrant (k b : ℝ) (h : ∀ x y : ℝ, x < 0 → y < 0 → y ≠ k * x + b) : k < 0 ∧ 0 ≤ b :=
sorry

end linear_function_no_third_quadrant_l7_7881


namespace find_m_l7_7683

noncomputable def first_term_1 := 24
noncomputable def second_term_1 := 8
noncomputable def first_term_2 := 24
noncomputable def second_term_2 (m : ℚ) := 8 + m
noncomputable def sum_of_first_series := 36

theorem find_m (m : ℚ) :
  (sum_of_first_series = 36) →
  (first_term_1 = 24) →
  (second_term_1 = 8) →
  (first_term_2 = 24) →
  (second_term_2 m = 8 + m) →
  (let sum_of_first_series := 36 in 
  let sum_of_second_series := 3 * sum_of_first_series in
  sum_of_second_series = 108) →
  m = 32 / 3 :=
sorry

end find_m_l7_7683


namespace ratio_second_vessel_l7_7594

-- Definitions of the problem conditions
def total_volume (V : ℝ) := V

def ratio_first_vessel_milk : ℝ := 2 / 3
def ratio_first_vessel_water : ℝ := 1 / 3

def ratio_second_vessel_milk (x y : ℝ) : ℝ := x / (x + y)
def ratio_second_vessel_water (x y : ℝ) : ℝ := y / (x + y)

def combined_ratio_milk (V x y : ℝ) : ℝ := (2 / 3) * V + (x / (x + y)) * V
def combined_ratio_water (V x y : ℝ) : ℝ := (1 / 3) * V + (y / (x + y)) * V

-- The Lean theorem statement
theorem ratio_second_vessel (x y : ℝ) (h : (combined_ratio_milk 1 x y) / (combined_ratio_water 1 x y) = 3) : x / y = 5 / 7 :=
  sorry

end ratio_second_vessel_l7_7594


namespace bruce_total_payment_l7_7311

noncomputable def total_amount_paid : ℝ :=
  let grapes_cost := 8 * 70 * 0.90 in
  let mangoes_cost := 11 * 55 in
  let oranges_cost := 5 * 45 * 0.80 in
  let apples_cost := 3 * 90 * 0.95 in
  let cherries_cost := 4.5 * 120 in
  grapes_cost + mangoes_cost + oranges_cost + apples_cost + cherries_cost

theorem bruce_total_payment :
  total_amount_paid = 2085.50 :=
by
  sorry

end bruce_total_payment_l7_7311


namespace perimeter_triangle_ellipse_focus_l7_7828

-- Definitions and Theorems regarding ellipses
def ellipse (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def focus1 (a b : ℝ) : ℝ := -sqrt (a^2 - b^2)
def focus2 (a b : ℝ) : ℝ := sqrt (a^2 - b^2)

theorem perimeter_triangle_ellipse_focus (a b : ℝ) (h_ellipse : ellipse a b x y) :
  let A : ℝ := focus1 a b
  let B : ℝ := focus1 a b
  let F2 : ℝ := focus2 a b
  A = focus1 a b ∧ B = focus1 a b ∧ F2 = focus2 a b →
  (2 * (2 * a)) = 16 :=
by
  intros
  simp only [ellipse, focus1, focus2] at *
  sorry

end perimeter_triangle_ellipse_focus_l7_7828


namespace log_pi_inequality_l7_7079

theorem log_pi_inequality (a b : ℝ) (π : ℝ) (h1 : 2^a = π) (h2 : 5^b = π) (h3 : a = Real.log π / Real.log 2) (h4 : b = Real.log π / Real.log 5) :
  (1 / a) + (1 / b) > 2 :=
by
  sorry

end log_pi_inequality_l7_7079


namespace ones_digit_seventeen_pow_l7_7360

def units_digit_cycle (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 1
  end

noncomputable def pow_five_five := 5^5

theorem ones_digit_seventeen_pow :
  pow_five_five = 3125 ∧
  (17 * 3125) % 4 = 1 →
  units_digit_cycle (17 * pow_five_five) = 7 :=
by {
  sorry
}

end ones_digit_seventeen_pow_l7_7360


namespace transformed_center_coordinates_l7_7318

theorem transformed_center_coordinates (S : (ℝ × ℝ)) (hS : S = (3, -4)) : 
  let reflected_S := (S.1, -S.2)
  let translated_S := (reflected_S.1, reflected_S.2 + 5)
  translated_S = (3, 9) :=
by
  sorry

end transformed_center_coordinates_l7_7318


namespace sum_of_roots_sin_eq_l7_7363

theorem sum_of_roots_sin_eq (h : ∀ x ∈ set.Icc 0 (2 * Real.pi), sin x * sin x - 4 * sin x - 5 = 0 → x = 3 * Real.pi / 2) :
  ∑ x in {x ∈ set.Icc 0 (2 * Real.pi) | sin x * sin x - 4 * sin x - 5 = 0}.to_finset := 3 * Real.pi / 2 := sorry

end sum_of_roots_sin_eq_l7_7363


namespace number_of_red_balls_l7_7010

noncomputable def red_balls (n_black n_red draws black_draws : ℕ) : ℕ :=
  if black_draws = (draws * n_black) / (n_black + n_red) then n_red else sorry

theorem number_of_red_balls :
  ∀ (n_black draws black_draws : ℕ),
    n_black = 4 →
    draws = 100 →
    black_draws = 40 →
    red_balls n_black (red_balls 4 6 100 40) 100 40 = 6 :=
by
  intros n_black draws black_draws h_black h_draws h_blackdraws
  dsimp [red_balls]
  rw [h_black, h_draws, h_blackdraws]
  norm_num
  sorry

end number_of_red_balls_l7_7010


namespace medium_kite_area_l7_7364

-- Define the points and the spacing on the grid
structure Point :=
mk :: (x : ℕ) (y : ℕ)

def medium_kite_vertices : List Point :=
[Point.mk 0 4, Point.mk 4 10, Point.mk 12 4, Point.mk 4 0]

def grid_spacing : ℕ := 2

-- Function to calculate the area of a kite given list of vertices and spacing
noncomputable def area_medium_kite (vertices : List Point) (spacing : ℕ) : ℕ := sorry

-- The theorem to be proved
theorem medium_kite_area (vertices : List Point) (spacing : ℕ) :
  vertices = medium_kite_vertices ∧ spacing = grid_spacing → area_medium_kite vertices spacing = 288 := 
by {
  -- The detailed proof would go here
  sorry
}

end medium_kite_area_l7_7364


namespace linear_avoid_third_quadrant_l7_7879

theorem linear_avoid_third_quadrant (k b : ℝ) (h : ∀ x : ℝ, k * x + b ≥ 0 → k * x + b > 0 → (k * x + b ≥ 0) ∧ (x ≥ 0)) :
  k < 0 ∧ b ≥ 0 :=
by
  sorry

end linear_avoid_third_quadrant_l7_7879


namespace ballot_name_in_all_boxes_same_candidate_l7_7348

def ballot (candidate : Type) := fin 10 → candidate

def ballot_box (ballot : Type) := list ballot

def ballot_boxes (ballot : Type) := fin 11 → ballot_box ballot

theorem ballot_name_in_all_boxes_same_candidate 
  {candidate : Type}
  (boxes : ballot_boxes (ballot candidate))
  (h1 : ∀ i : fin 11, ∃ b : ballot candidate, b ∈ boxes i)
  (h2 : ∀ (selection : fin 11 → ballot candidate), (∀ i j : fin 11, selection i ∈ boxes i ∧ selection j ∈ boxes j → ∃ c : candidate, ∀ i : fin 11, c ∈ selection i))
  : ∃ i : fin 11, ∃ c : candidate, ∀ b : ballot candidate, b ∈ boxes i → c ∈ b :=
by
  sorry

end ballot_name_in_all_boxes_same_candidate_l7_7348


namespace sin_sq_cos_sq_eq_one_l7_7300

theorem sin_sq_cos_sq_eq_one (α : ℝ) : sin α * sin α + cos α * cos α = 1 :=
sorry

end sin_sq_cos_sq_eq_one_l7_7300


namespace probability_of_less_than_one_third_l7_7859

theorem probability_of_less_than_one_third :
  (prob_of_interval (0 : ℝ) (1 / 2 : ℝ) (1 / 3 : ℝ) = 2 / 3) :=
sorry

end probability_of_less_than_one_third_l7_7859


namespace range_of_lambda_over_m_l7_7836

variable (λ m α : ℝ)

def vector_a (λ : ℝ) (α : ℝ) : ℝ × ℝ := (λ + 2, λ^2 - cos α ^ 2)
def vector_b (m : ℝ) (α : ℝ) : ℝ × ℝ := (m, m / 2 + sin α)

theorem range_of_lambda_over_m 
  (h : vector_a λ α = (2 * (vector_b m α)).fst, 2 * (vector_b m α)).snd) :
  -6 ≤ λ / m ∧ λ / m ≤ 1 :=
sorry

end range_of_lambda_over_m_l7_7836


namespace union_eq_set_l7_7930

noncomputable def M : Set ℤ := {x | |x| < 2}
noncomputable def N : Set ℤ := {-2, -1, 0}

theorem union_eq_set : M ∪ N = {-2, -1, 0, 1} := by
  sorry

end union_eq_set_l7_7930


namespace perpendicular_condition_l7_7786

noncomputable def line := ℝ → (ℝ × ℝ × ℝ)
noncomputable def plane := (ℝ × ℝ × ℝ) → Prop

variable {l m : line}
variable {α : plane}

-- l and m are two different lines
axiom lines_are_different : l ≠ m

-- m is parallel to the plane α
axiom m_parallel_alpha : ∀ t : ℝ, α (m t)

-- Prove that l perpendicular to α is a sufficient but not necessary condition for l perpendicular to m
theorem perpendicular_condition :
  (∀ t : ℝ, ¬ α (l t)) → (∀ t₁ t₂ : ℝ, (l t₁) ≠ (m t₂)) ∧ ¬ (∀ t : ℝ, ¬ α (l t)) :=
by 
  sorry

end perpendicular_condition_l7_7786


namespace number_of_classmates_ate_cake_l7_7140

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l7_7140


namespace classmates_ate_cake_l7_7090

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l7_7090


namespace hypotenuse_of_right_triangle_l7_7993

theorem hypotenuse_of_right_triangle (h : height_dropped_to_hypotenuse = 1) (a : acute_angle = 15) :
∃ (hypotenuse : ℝ), hypotenuse = 4 :=
sorry

end hypotenuse_of_right_triangle_l7_7993


namespace min_diff_of_arithmetic_sequence_l7_7979

theorem min_diff_of_arithmetic_sequence :
  ∃ (a1 a2 a3 a4 a5 a6 a7 : ℕ), 
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 ∧ a1 ≠ a7 ∧ 
     a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 ∧ a2 ≠ a7 ∧ 
     a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 ∧ a3 ≠ a7 ∧ 
     a4 ≠ a5 ∧ a4 ≠ a6 ∧ a4 ≠ a7 ∧ 
     a5 ≠ a6 ∧ a5 ≠ a7 ∧ 
     a6 ≠ a7) ∧ 
    (a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0 ∧ a5 > 0 ∧ a6 > 0 ∧ a7 > 0) ∧
    (∃ d : ℤ, 
      2 * ↑a2 = ↑a1 + d ∧ 
      3 * ↑a3 = ↑a1 + 2 * d ∧ 
      4 * ↑a4 = ↑a1 + 3 * d ∧ 
      5 * ↑a5 = ↑a1 + 4 * d ∧ 
      6 * ↑a6 = ↑a1 + 5 * d ∧ 
      7 * ↑a7 = ↑a1 + 6 * d) ∧
    | ↑a7 - ↑a1 | = 1 :=
begin 
  sorry
end

end min_diff_of_arithmetic_sequence_l7_7979


namespace find_middle_and_oldest_sons_l7_7658

-- Defining the conditions
def youngest_age : ℕ := 2
def father_age : ℕ := 33
def father_age_in_12_years : ℕ := father_age + 12
def youngest_age_in_12_years : ℕ := youngest_age + 12

-- Lean theorem statement to find the ages of the middle and oldest sons
theorem find_middle_and_oldest_sons (y z : ℕ) (h1 : father_age_in_12_years = (youngest_age_in_12_years + 12 + y + 12 + z + 12)) :
  y = 3 ∧ z = 4 :=
sorry

end find_middle_and_oldest_sons_l7_7658


namespace complete_even_square_diff_eqn_even_square_diff_multiple_of_four_odd_square_diff_multiple_of_eight_l7_7708

theorem complete_even_square_diff_eqn : (10^2 - 8^2 = 4 * 9) :=
by sorry

theorem even_square_diff_multiple_of_four (n : ℕ) : (4 * (n + 1) * (n + 1) - 4 * n * n) % 4 = 0 :=
by sorry

theorem odd_square_diff_multiple_of_eight (m : ℕ) : ((2 * m + 1)^2 - (2 * m - 1)^2) % 8 = 0 :=
by sorry

end complete_even_square_diff_eqn_even_square_diff_multiple_of_four_odd_square_diff_multiple_of_eight_l7_7708


namespace rectangle_circle_area_ratio_l7_7290

theorem rectangle_circle_area_ratio
  (r : ℝ)
  (rectangle_longer_chord : ℝ = r)
  (rectangle_shorter_chord : ℝ = r / 2)
  (longer_is_twice_shorter : rectangle_longer_chord = 2 * rectangle_shorter_chord) :
  ((rectangle_shorter_chord * rectangle_longer_chord) / (π * r^2)) = 1 / (2 * π) :=
sorry

end rectangle_circle_area_ratio_l7_7290


namespace luke_keeps_lollipops_l7_7515

theorem luke_keeps_lollipops :
  let total_lollipops := 57 + 98 + 13 + 167
  let friends := 13
  total_lollipops % friends = 10 :=
by
  let total_lollipops := 57 + 98 + 13 + 167
  let friends := 13
  show total_lollipops % friends = 10
  calc
    total_lollipops % friends = 335 % 13 : by norm_num
    ... = 10 : by norm_num

end luke_keeps_lollipops_l7_7515


namespace infinite_series_value_l7_7320

noncomputable def infinite_series := ∑ n in filter (λ x, x ≥ 2) (range ∞), 
  (n^4 + 5 * n^2 + 15 * n + 15) / (2^n * (n^4 + 8))

theorem infinite_series_value : infinite_series = 1 / 2 :=
by sorry

end infinite_series_value_l7_7320


namespace annual_interest_rate_is_10_percent_l7_7602

noncomputable def principal (P : ℝ) := P = 1500
noncomputable def total_amount (A : ℝ) := A = 1815
noncomputable def time_period (t : ℝ) := t = 2
noncomputable def compounding_frequency (n : ℝ) := n = 1
noncomputable def interest_rate_compound_interest_formula (P A t n : ℝ) (r : ℝ) := 
  A = P * (1 + r / n) ^ (n * t)

theorem annual_interest_rate_is_10_percent : 
  ∀ (P A t n : ℝ) (r : ℝ), principal P → total_amount A → time_period t → compounding_frequency n → 
  interest_rate_compound_interest_formula P A t n r → r = 0.1 :=
by
  intros P A t n r hP hA ht hn h_formula
  sorry

end annual_interest_rate_is_10_percent_l7_7602


namespace price_before_tax_l7_7843

theorem price_before_tax (P : ℝ) (h : 1.15 * P = 1955) : P = 1700 :=
by sorry

end price_before_tax_l7_7843


namespace no_integer_solution_exists_l7_7546

theorem no_integer_solution_exists : ¬ ∃ (x y z t : ℤ), x^2 + y^2 + z^2 = 8 * t - 1 := 
by sorry

end no_integer_solution_exists_l7_7546


namespace probability_of_selected_number_probability_of_selected_number_in_given_interval_l7_7865

noncomputable def probability_in_interval (a b c : ℝ) (hab : 0 < a) (hbc : b < c) : ℝ :=
  (b - a) / (c - a)

theorem probability_of_selected_number (x : ℝ) (hx : 0 < x) (hx_le : x < 1/2) : 
  probability_in_interval 0 (1/3) (1/2) 0 lt_one_half = 2/3 := 
by
  have p := probability_in_interval 0 (1/3) (1/2) 0 (by norm_num : 1 < 2)
  norm_num at p
  exact p

-- Helper theorem to convert the original question
theorem probability_of_selected_number_in_given_interval :
  ∀ x, 0 < x ∧ x < 1/2 → x < 1/3 → probability_of_selected_number x 0 (by norm_num) = 2/3 :=
by
  intros x _ _
  exact probability_of_selected_number x 0 (by norm_num)

-- Sorry to skip the proof as requested
sorry

end probability_of_selected_number_probability_of_selected_number_in_given_interval_l7_7865


namespace sum_of_geometric_sequence_15_l7_7910

noncomputable def geometric_sequence (a r : ℕ → ℝ) := ∃ (a₀ r : ℝ), ∀ n, a n = a₀ * r ^ n

noncomputable def sum_of_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = (0 to n).sum a

theorem sum_of_geometric_sequence_15 (a S : ℕ → ℝ) (r : ℝ) (h_geo : geometric_sequence a r)
  (h_sum : sum_of_geometric_sequence a S)
  (h_S5 : S 5 = 3) (h_S10 : S 10 = 9) : S 15 = 21 :=
sorry

end sum_of_geometric_sequence_15_l7_7910


namespace segment_length_eq_ten_l7_7208

theorem segment_length_eq_ten (x : ℝ) (h : |x - 3| = 5) : |8 - (-2)| = 10 :=
by {
  sorry
}

end segment_length_eq_ten_l7_7208


namespace cake_sharing_l7_7105

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l7_7105


namespace inequality_product_l7_7941

theorem inequality_product {n : ℕ} (a b : Finₓ n → ℝ)
  (h1 : (∑ i, a i) = n)
  (h2 : (∏ i, b i) = 1) : 
  (∏ i, (1 + a i)) ≤ (∏ i, (1 + b i)) :=
sorry

end inequality_product_l7_7941


namespace amount_borrowed_eq_4137_84_l7_7518

noncomputable def compound_interest (initial : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  initial * (1 + rate/100) ^ time

theorem amount_borrowed_eq_4137_84 :
  ∃ P : ℝ, 
    (compound_interest (compound_interest (compound_interest P 6 3) 8 4) 10 2 = 8110) 
    ∧ (P = 4137.84) :=
by
  sorry

end amount_borrowed_eq_4137_84_l7_7518


namespace number_of_classmates_l7_7120

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l7_7120


namespace fixed_point_of_line_l7_7996

theorem fixed_point_of_line (k : ℝ) : ∀ x y : ℝ, (kx + y - 2k = 0) → (x = 2 ∧ y = 0) :=
by
  intros x y h
  have hx : k * (x - 2) + y = 0 := by { rw h }
  sorry

end fixed_point_of_line_l7_7996


namespace emily_sixth_quiz_score_l7_7349

theorem emily_sixth_quiz_score (q1 q2 q3 q4 q5 target_mean : ℕ) (required_sum : ℕ) (current_sum : ℕ) (s6 : ℕ)
  (h1 : q1 = 94) (h2 : q2 = 97) (h3 : q3 = 88) (h4 : q4 = 91) (h5 : q5 = 102) (h_target_mean : target_mean = 95)
  (h_required_sum : required_sum = 6 * target_mean) (h_current_sum : current_sum = q1 + q2 + q3 + q4 + q5)
  (h6 : s6 = required_sum - current_sum) :
  s6 = 98 :=
by
  sorry

end emily_sixth_quiz_score_l7_7349


namespace least_time_exact_7_horses_back_l7_7576

def horses_meeting_time (T : ℕ) : Prop :=
  ∃ (H : finset ℕ), H.card = 7 ∧ (∀ h ∈ H, ∃ k ∈ (finset.range 16), k > 0 ∧ T % k = 0) ∧ ∀ m < T, ∀ I : finset ℕ, I.card = 7 → (∀ h ∈ I, ∃ k ∈ (finset.range 16), k > 0 ∧ m % k = 0) → H = I

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem least_time_exact_7_horses_back :
  horses_meeting_time 210 ∧ sum_of_digits 210 = 3 :=
by
  sorry

end least_time_exact_7_horses_back_l7_7576


namespace find_digits_l7_7735

def divisible_45z_by_8 (z : ℕ) : Prop :=
  45 * z % 8 = 0

def sum_digits_divisible_by_9 (x y z : ℕ) : Prop :=
  (1 + 3 + x + y + 4 + 5 + z) % 9 = 0

def alternating_sum_digits_divisible_by_11 (x y z : ℕ) : Prop :=
  (1 - 3 + x - y + 4 - 5 + z) % 11 = 0

theorem find_digits (x y z : ℕ) (h_div8 : divisible_45z_by_8 z) (h_div9 : sum_digits_divisible_by_9 x y z) (h_div11 : alternating_sum_digits_divisible_by_11 x y z) :
  x = 2 ∧ y = 3 ∧ z = 6 := 
sorry

end find_digits_l7_7735


namespace isosceles_triangle_min_max_l7_7351

noncomputable def isosceles_triangle_minimization (a : ℝ) : Prop :=
  let A := (0 : ℝ, 0 : ℝ)
  let B := (a, 0)
  let C := (0, a)
  ∃ M : ℝ × ℝ, 
    let u := (λ (M : ℝ × ℝ), 
      (M.1 - A.1)^2 + (M.2 - A.2)^2 + 
      (M.1 - B.1)^2 + (M.2 - B.2)^2 + 
      (M.1 - C.1)^2 + (M.2 - C.2)^2) in
    (0 <= M.1 ∧ M.1 <= a) ∧ (0 <= M.2 ∧ M.2 <= a) ∧
    (u(M) = 3 * (a^2 / 3) - 2 * a * (a / 3) + 2 * a^2)
    ∧ (u(M) = 4 * a^2 / 3)

theorem isosceles_triangle_min_max (a : ℝ) : isosceles_triangle_minimization a :=
sorry

end isosceles_triangle_min_max_l7_7351


namespace curve_C1_eq_curve_C2_eq_min_distance_C1_C2_l7_7017

open Real 

-- Definitions for parametric curve C1
def CurveC1 (α : ℝ) : ℝ × ℝ :=
  (sqrt 2 * sin (α + π / 4), sin (2 * α) + 1)

-- Cartesian equation for curve C1.
theorem curve_C1_eq (x y α : ℝ) (h : (x, y) = CurveC1 α) :
  y = x^2 := 
sorry

-- Definitions for polar curve C2
def polarC2 (ρ θ : ℝ) : Prop :=
  ρ^2 = 4 * ρ * sin θ - 3

-- Cartesian equation for curve C2.
theorem curve_C2_eq (x y ρ θ : ℝ) (hx : ρ^2 = x^2 + y^2) (hy : ρ * sin θ = y) 
  (h : polarC2 ρ θ) :
  x^2 + (y - 2)^2 = 1 := 
sorry

-- Minimum distance between curve C1 and curve C2
theorem min_distance_C1_C2 : 
  ∀ x₀ y₀,
  (y₀ = x₀^2) → 
  (x₀, y₀) ∈ {p | p = CurveC1 (α)} → 
  ∃ (d : ℝ),
  (d = ( (x₀^2 - 3 / 2)^2 + 7 / 4) ) :=
sorry

example : 
  ∃ d,
  d = (sqrt 7) / 2 :=
  by { use (sqrt 7) / 2, sorry }

end curve_C1_eq_curve_C2_eq_min_distance_C1_C2_l7_7017


namespace train_station_travel_time_l7_7296

def max (a b : ℕ) : ℕ := if a > b then a else b

theorem train_station_travel_time :
    let coal_cars := 6
    let max_coal := 2
    let iron_cars := 12
    let max_iron := 3
    let wood_cars := 2
    let max_wood := 1
    let total_time := 100
    let stops_needed := max (coal_cars / max_coal) (max (iron_cars / max_iron) (wood_cars / max_wood))
    let travels := stops_needed - 1
    let travel_time := total_time / travels
    travel_time = 33.33 :=
by
    sorry

end train_station_travel_time_l7_7296


namespace is_even_iff_exists_Q_l7_7530

def is_even_function (P : ℂ[X]) : Prop :=
  ∀ z : ℂ, P.eval (-z) = P.eval z

theorem is_even_iff_exists_Q (P : ℂ[X]) : 
  is_even_function P ↔ ∃ Q : ℂ[X], ∀ z : ℂ, P.eval z = Q.eval z * Q.eval (-z) :=
by
  sorry

end is_even_iff_exists_Q_l7_7530


namespace ruth_hours_per_week_l7_7082

theorem ruth_hours_per_week :
  let daily_hours := 8
  let days_per_week := 5
  let monday_wednesday_friday := 3
  let tuesday_thursday := 2
  let percentage_to_hours (percent : ℝ) (hours : ℕ) : ℝ := percent * hours
  let total_weekly_hours := daily_hours * days_per_week
  let monday_wednesday_friday_math_hours := percentage_to_hours 0.25 daily_hours
  let monday_wednesday_friday_science_hours := percentage_to_hours 0.15 daily_hours
  let tuesday_thursday_math_hours := percentage_to_hours 0.2 daily_hours
  let tuesday_thursday_science_hours := percentage_to_hours 0.35 daily_hours
  let tuesday_thursday_history_hours := percentage_to_hours 0.15 daily_hours
  let weekly_math_hours := monday_wednesday_friday_math_hours * monday_wednesday_friday + tuesday_thursday_math_hours * tuesday_thursday
  let weekly_science_hours := monday_wednesday_friday_science_hours * monday_wednesday_friday + tuesday_thursday_science_hours * tuesday_thursday
  let weekly_history_hours := tuesday_thursday_history_hours * tuesday_thursday
  let total_hours := weekly_math_hours + weekly_science_hours + weekly_history_hours
  total_hours = 20.8 := by
  sorry

end ruth_hours_per_week_l7_7082


namespace find_x_l7_7938

variables (a b x : ℝ)
variable (h_b : b ≠ 0)

theorem find_x (h : (3 * a)^(2 * b) = a^b * x^(2 * b)) : x = 3 * real.sqrt a :=
by sorry

end find_x_l7_7938


namespace inequality_sqrt_d_l7_7054

theorem inequality_sqrt_d (d n : ℕ) (hd : d ≥ 1) (hd_not_square : ∀ m : ℕ, m * m ≠ d) (hn : n ≥ 1) :
  (n * Real.sqrt d + 1) * Real.abs (Real.sin (n * Real.pi * Real.sqrt d)) ≥ 1 :=
sorry

end inequality_sqrt_d_l7_7054


namespace a_values_general_formula_exists_int_quotient_l7_7495

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := if (n % 2 = 0) then a n + 1 else 2 * a n

def S (n : ℕ) : ℕ :=
(n + 1).sum a

theorem a_values (h1 : a 2 = 2) (h2 : a 3 = 3) :
  a 2 = 2 ∧ a 3 = 3 :=
by
  sorry

theorem general_formula (n : ℕ) :
  ( ∀ m, n = 2 * m + 1 → a n = 2 ^ ((n + 1) / 2) - 1 ) ∧ 
  ( ∀ m, n = 2 * m → a n = 2 ^ ((n + 2) / 2) - 2 ) :=
by
  sorry

theorem exists_int_quotient (h : ∃ n, (S n) / (a n) ∈ ℤ) :
  ∀ (n : ℕ), n = 1 ∨ n = 3 ∨ n = 4 ↔
  ((S n) / (a n) : ℤ) ∈ ℤ :=
by
  sorry

end a_values_general_formula_exists_int_quotient_l7_7495


namespace mary_keep_warm_hours_l7_7951

-- Definitions based on the conditions
def sticks_from_chairs (chairs : ℕ) : ℕ := chairs * 6
def sticks_from_tables (tables : ℕ) : ℕ := tables * 9
def sticks_from_stools (stools : ℕ) : ℕ := stools * 2
def sticks_needed_per_hour : ℕ := 5

-- Given counts of furniture
def chairs : ℕ := 18
def tables : ℕ := 6
def stools : ℕ := 4

-- Total number of sticks
def total_sticks : ℕ := (sticks_from_chairs chairs) + (sticks_from_tables tables) + (sticks_from_stools stools)

-- Proving the number of hours Mary can keep warm
theorem mary_keep_warm_hours : total_sticks / sticks_needed_per_hour = 34 := by
  sorry

end mary_keep_warm_hours_l7_7951


namespace AR_parallel_BC_l7_7034

noncomputable def point := sorry

variables {A B C D O P Q R E : point}
variables {circumcircle : point → point → point → point → Prop}
variables {angle_bisector : point → point → point → point → point → Prop}
variables {perpendicular_bisector : point → point → point → Prop}
variables {is_midpoint : point → point → point → Prop}

-- The conditions translated:
variables (hABC : ¬ (O = A ∨ O = B ∨ O = C)) -- O as the circumcenter of acute ΔABC
variables (h1 : angle_bisector A B C D) -- D is on the circumcircle and bisector of ∠A
variables (h2 : circumcircle A B C O) -- O is the circumcenter of ΔABC
variables (h3 : angle_bisector A O B P)  -- P is bisector point on circle with diameter AD
variables (h4 : angle_bisector A O C Q)   -- Q is bisector point on circle with diameter AD
variables (h5 : perpendicular_bisector A D R) -- R is point on perpendicular bisector of AD
variables (h6 : is_midpoint E A D) -- E is midpoint of AD
variables (h7 : circumcircle P Q A E) -- PEAQ are concyclic

-- The goal:
theorem AR_parallel_BC : (A ≠ B ∧ A ≠ C ∧ B ≠ C) → (O ≠ A ∧ O ≠ B ∧ O ≠ C) → A ≠ D → 
(A ≠ R) → (R ≠ D) → (P ≠ Q) → circ_left_pt Q P R = adj_circ Q R ABC → parallel A R B C := sorry

end AR_parallel_BC_l7_7034


namespace rahul_savings_is_correct_l7_7263

def Rahul_Savings_Problem : Prop :=
  ∃ (NSC PPF : ℝ), 
    (1/3) * NSC = (1/2) * PPF ∧ 
    NSC + PPF = 180000 ∧ 
    PPF = 72000

theorem rahul_savings_is_correct : Rahul_Savings_Problem :=
  sorry

end rahul_savings_is_correct_l7_7263


namespace fraction_for_repeating_decimal_l7_7765

variable (a r S : ℚ)
variable (h1 : a = 3/5)
variable (h2 : r = 1/10)
variable (h3 : S = a / (1 - r))

theorem fraction_for_repeating_decimal : S = 2 / 3 :=
by
  have h4 : 1 - r = 9 / 10, from sorry
  have h5 : S = (3 / 5) / (9 / 10), from sorry
  have h6 : S = (3 * 10) / (5 * 9), from sorry
  have h7 : S = 30 / 45, from sorry
  have h8 : 30 / 45 = 2 / 3, from sorry
  exact h8

end fraction_for_repeating_decimal_l7_7765


namespace tangent_line_at_point_is_correct_l7_7555

open Real

noncomputable def curve (x : ℝ) : ℝ :=
  2 * x - x ^ 3

theorem tangent_line_at_point_is_correct :
  let x₀ := 1
  let y₀ := 1
  let y' := deriv curve x₀
  tangent_line_eq : ((∀ x y : ℝ, (y - y₀ = y' * (x - x₀)) → (x + y - 2 = 0))) :=
by
  let x₀ := 1
  let y₀ := 1
  have y'_value : deriv curve x₀ = -1 := sorry
  intro x y
  intro H
  have H1 : y - y₀ = y' * (x - x₀) := H
  have H2 : y - 1 = -1 * (x - 1) := by sorry
  exact H2

end tangent_line_at_point_is_correct_l7_7555


namespace chocolate_bar_cost_l7_7490

theorem chocolate_bar_cost :
  ∀ (num_chocolate_bars : ℕ) (num_gummy_bears : ℕ) (num_chocolate_bags : ℕ)
    (gummy_bear_cost : ℝ) (chocolate_bag_cost : ℝ) (discount_rate : ℝ) (total_spent : ℝ),
    num_chocolate_bars = 10 →
    num_gummy_bears = 10 →
    num_chocolate_bags = 20 →
    gummy_bear_cost = 2 →
    chocolate_bag_cost = 5 →
    discount_rate = 0.1 →
    total_spent = 150 →
    let total_gummy_bears = num_gummy_bears * gummy_bear_cost in
    let total_chocolate_bags_before_discount = num_chocolate_bags * chocolate_bag_cost in
    let chocolate_discount = discount_rate * total_chocolate_bags_before_discount in
    let total_chocolate_bags_after_discount = total_chocolate_bags_before_discount - chocolate_discount in
    let total_other_sweets = total_gummy_bears + total_chocolate_bags_after_discount in
    let total_chocolate_bars_spent = total_spent - total_other_sweets in
    total_chocolate_bars_spent / num_chocolate_bars = 4 :=
begin
  intros num_chocolate_bars num_gummy_bears num_chocolate_bags
         gummy_bear_cost chocolate_bag_cost discount_rate total_spent,
  intros hc_bars hc_gb hc_c_bags hc_gummy_cost hc_choco_cost hc_discount hc_total_spent,
  let total_gummy_bears := num_gummy_bears * gummy_bear_cost,
  let total_chocolate_bags_before_discount := num_chocolate_bags * chocolate_bag_cost,
  let chocolate_discount := discount_rate * total_chocolate_bags_before_discount,
  let total_chocolate_bags_after_discount := total_chocolate_bags_before_discount - chocolate_discount,
  let total_other_sweets := total_gummy_bears + total_chocolate_bags_after_discount,
  let total_chocolate_bars_spent := total_spent - total_other_sweets,
  have h_cost := total_chocolate_bars_spent / num_chocolate_bars,
  sorry
end

end chocolate_bar_cost_l7_7490


namespace length_of_segment_l7_7201

theorem length_of_segment (x : ℝ) : 
  |x - (27^(1/3))| = 5 →
  ∃ a b : ℝ, a - b = 10 ∧ (|a - (27^(1/3))| = 5 ∧ |b - (27^(1/3))| = 5) :=
by
  sorry

end length_of_segment_l7_7201


namespace problem1_problem2_l7_7903

-- Defining the parametric form of C1
def parametric_curve_C1 (α : ℝ) : ℝ × ℝ :=
  (1 + Real.cos α, Real.sin α)

-- Defining the polar form of C2
def polar_curve_C2 (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ^2 = Real.sin θ

-- Main theorems to prove
theorem problem1 :
  (∀ ρ θ : ℝ, (ρ = 2 * Real.cos θ ↔ ∃ α : ℝ, (1 + Real.cos α, Real.sin α) = (ρ * Real.cos θ, ρ * Real.sin θ))) ∧
  (∀ x y : ℝ, (x^2 = y ↔ ∃ ρ θ : ℝ, ρ * Real.cos θ^2 = Real.sin θ ∧ (ρ * Real.cos θ, ρ * Real.sin θ) = (x, y))) :=
by
  sorry

-- Defining the ray l
def ray_l (k t : ℝ) : ℝ × ℝ :=
  (t, k * t)

-- Theorem for the range of |OA| * |OB|
theorem problem2 (k : ℝ) (hk : 1 < k ∧ k ≤ Real.sqrt 3) :
  (∃ t1 t2 : ℝ, parametric_curve_C1 (Real.arccos t1) = ray_l k t1 ∧ polar_curve_C2 t2 (Real.atan (t2/k)) ∧
  2 * Real.cos (Real.arccos t1) * (Real.sin (Real.atan (t2/k)) / Real.cos (Real.atan (t2/k))^2) ∈ Ioo 2 (2 * Real.sqrt 3)) :=
by
  sorry

end problem1_problem2_l7_7903


namespace intersection_union_m3_range_of_m_l7_7831

def A (x : ℝ) := -3 ≤ x ∧ x ≤ 2

def B (m : ℝ) (x : ℝ) := 1 - m ≤ x ∧ x ≤ 3 * m - 1

-- Part 1.
theorem intersection_union_m3 :
  (A ∩ B 3 = { x : ℝ | -2 ≤ x ∧ x ≤ 2 }) ∧
  (A ∪ B 3 = { x : ℝ | -3 ≤ x ∧ x ≤ 8 }) :=
by sorry

-- Part 2.
theorem range_of_m (m : ℝ) :
  (A ∩ B m = B m) → m ≤ 1 :=
by sorry

end intersection_union_m3_range_of_m_l7_7831


namespace possible_values_of_g_l7_7050

noncomputable def g (a b c : ℝ) : ℝ :=
  a / (a + b) + b / (b + c) + c / (c + a)

theorem possible_values_of_g (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 :=
by
  sorry

end possible_values_of_g_l7_7050


namespace surface_area_LShape_l7_7975

-- Define the structures and conditions
structure UnitCube where
  x : ℕ
  y : ℕ
  z : ℕ

def LShape (cubes : List UnitCube) : Prop :=
  -- Condition 1: Exactly 7 unit cubes
  cubes.length = 7 ∧
  -- Condition 2: 4 cubes in a line along x-axis (bottom row)
  ∃ a b c d : UnitCube, 
    (a.x + 1 = b.x ∧ b.x + 1 = c.x ∧ c.x + 1 = d.x ∧
     a.y = b.y ∧ b.y = c.y ∧ c.y = d.y ∧
     a.z = b.z ∧ b.z = c.z ∧ c.z = d.z) ∧
  -- Condition 3: 3 cubes stacked along z-axis at one end of the row
  ∃ e f g : UnitCube,
    (d.x = e.x ∧ e.x = f.x ∧ f.x = g.x ∧
     d.y = e.y ∧ e.y = f.y ∧ f.y = g.y ∧
     e.z + 1 = f.z ∧ f.z + 1 = g.z)

-- Define the surface area function
def surfaceArea (cubes : List UnitCube) : ℕ :=
  4*7 - 2*3 + 4 -- correct answer calculation according to manual analysis of exposed faces

-- The theorem to be proven
theorem surface_area_LShape : 
  ∀ (cubes : List UnitCube), LShape cubes → surfaceArea cubes = 26 :=
by sorry

end surface_area_LShape_l7_7975


namespace triangle_acute_angled_l7_7035

open Complex

theorem triangle_acute_angled (ABC : Triangle) (h_acute : ∀ ∠ ∈ ABC.angles, ∠ < π / 2)
  (h_BC_mid : ∃ M, M = (B + C) / 2)
  (h_EF_foot : ∃ E F, foot B E ∧ foot C F)
  (h_KL_mid : ∃ K L, mid K (M + E) ∧ mid L (M + F))
  (h_AT_parallel : ∃ T, T ∈ line KL ∧ parallel (line AT) (line BC))
  : distance T A = distance T M := by
  sorry

end triangle_acute_angled_l7_7035


namespace part_a_impossible_part_b_impossible_l7_7564

-- Definitions for the initial state and operations

inductive Vertex : Type
| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8

-- Initial labeling of the vertices
def initialLabeling : Vertex → ℕ
| Vertex.v1 := 1
| _ := 0

-- Operation that adds 1 to both ends of an edge; we abstract this as a sequence of operations
def operation (l : Vertex → ℕ) (v1 v2 : Vertex) : Vertex → ℕ :=
  fun v => if v = v1 ∨ v = v2 then l v + 1 else l v

-- Ultimately we want to prove that all labels (values) are divisible by n.
def all_divisible (n : ℕ) (labeling : Vertex → ℕ) : Prop :=
  ∀ v, n ∣ labeling v

theorem part_a_impossible : ¬ ∃ (f : (Vertex → ℕ) → (Vertex → ℕ)), all_divisible 2 (f initialLabeling) :=
  by
  sorry

theorem part_b_impossible : ¬ ∃ (f : (Vertex → ℕ) → (Vertex → ℕ)), all_divisible 3 (f initialLabeling) :=
  by
  sorry

end part_a_impossible_part_b_impossible_l7_7564


namespace relationship_x_y_z_l7_7393

noncomputable def a : ℝ := sorry
noncomputable def x : ℝ := log a (sqrt 2) + (1 / 2) * log a 3
noncomputable def y : ℝ := (1 / 2) * log a 5
noncomputable def z : ℝ := log a (sqrt 21) - log a (sqrt 3)

theorem relationship_x_y_z (ha : 1 < a) : z > x ∧ x > y :=
  sorry

end relationship_x_y_z_l7_7393


namespace number_of_sections_l7_7032

-- Definitions based on the conditions in a)
def num_reels : Nat := 3
def length_per_reel : Nat := 100
def section_length : Nat := 10

-- The math proof problem statement
theorem number_of_sections :
  (num_reels * length_per_reel) / section_length = 30 := by
  sorry

end number_of_sections_l7_7032


namespace find_a_find_m_no_M_exists_l7_7818

noncomputable def f (x a : ℝ) : ℝ := ((x + a) * Real.exp x) / (x + 1)

noncomputable def g (x a : ℝ) : ℝ := (x + 1) * f x a

noncomputable def T (n : ℕ) (f : ℝ → ℝ) : ℝ :=
1 + 2 * (List.sum (List.map (λ k, f (k / n)) (List.range (n - 1))))

noncomputable def g_T (n : ℕ) (x : ℝ → ℝ) : ℝ :=
T n (λ y, (y + 1) * f y 0 / (y * (Real.sqrt Real.exp 1 + Real.exp y)))

theorem find_a (a : ℝ) :
  let slope := (-(4:ℝ)) / (3 * Real.exp 1);
  let tangent := deriv (λ x, ((x + a) * Real.exp x) / (x + 1));
  slope = tangent 1 → a = 0 :=
by 
  intros slope tangent;
  sorry

theorem find_m (m : ℝ) :
  (∀ x : ℝ, x > (2 / 3) → ((x + 1) * f x 0) ≥ m * (2 * x - 1)) ↔ m ≤ Real.exp 1 :=
by
  intro m;
  sorry

theorem no_M_exists :
  ¬ ∃ M : ℝ, ∀ n : ℕ, n ≥ 2 → (List.sum (List.map (λ k, 1 / g_T (3 * n) k) (List.range n))) < M :=
by
  intro M;
  sorry

end find_a_find_m_no_M_exists_l7_7818


namespace area_triangle_DOB_eq_l7_7528

-- Define necessary points and properties
variables {A B C D O : Type}
variables [EuclideanGeometry A B C D O]

-- Define the area of triangles and the centroid property
def area (T : Triangle) : ℝ := sorry
def is_centroid (G : Point) (T : Triangle) : Prop := sorry
def lies_on_extension (D A C : Point) : Prop := sorry

-- Hypotheses in Lean
variables (S Sl : ℝ)
variables (TABC TDOC : Triangle)
variables (O_centroid : is_centroid O TABC)
variables (area_TABC : area TABC = S)
variables (area_TDOC : area TDOC = Sl)
variables (D_on_extension : lies_on_extension D A C)

-- Define the proposition we need to prove
theorem area_triangle_DOB_eq (S Sl : ℝ) (TABC TDOC : Triangle)
  (O_centroid : is_centroid O TABC)
  (area_TABC : area TABC = S)
  (area_TDOC : area TDOC = Sl)
  (D_on_extension : lies_on_extension D A C) :
  area (Triangle.mk D O B) = 2 * Sl - S / 3 :=
sorry

end area_triangle_DOB_eq_l7_7528


namespace repeating_decimal_sum_l7_7700
noncomputable def repeater := sorry  -- Definitions of repeating decimals would be more complex in Lean, skipping implementation.

-- Define the repeating decimals as constants for the purpose of this proof
def decimal1 : ℚ := 1 / 3
def decimal2 : ℚ := 2 / 3

-- Define the main theorem
theorem repeating_decimal_sum : decimal1 + decimal2 = 1 := by
  sorry  -- Placeholder for proof

end repeating_decimal_sum_l7_7700


namespace min_boxes_to_eliminate_for_one_third_chance_l7_7479

-- Define the number of boxes
def total_boxes := 26

-- Define the number of boxes with at least $250,000
def boxes_with_at_least_250k := 6

-- Define the condition for having a 1/3 chance
def one_third_chance (remaining_boxes : ℕ) : Prop :=
  6 / remaining_boxes = 1 / 3

-- Define the target number of boxes to eliminate
def boxes_to_eliminate := total_boxes - 18

theorem min_boxes_to_eliminate_for_one_third_chance :
  ∃ remaining_boxes : ℕ, one_third_chance remaining_boxes ∧ total_boxes - remaining_boxes = boxes_to_eliminate :=
sorry

end min_boxes_to_eliminate_for_one_third_chance_l7_7479


namespace segment_length_l7_7213

theorem segment_length (x : ℝ) (h : |x - (27)^(1/3)| = 5) : ∃ a b : ℝ, (a = 8 ∧ b = -2 ∨ a = -2 ∧ b = 8) ∧ real.dist a b = 10 :=
by
  use [8, -2] -- providing the endpoints explicitly
  split
  -- prove that these are the correct endpoints
  · left; exact ⟨rfl, rfl⟩
  -- prove the distance is 10
  · apply real.dist_eq; linarith
  

end segment_length_l7_7213


namespace parabola_focus_l7_7740

theorem parabola_focus (a : ℝ) (b : ℝ) (h : a = 9 ∧ b = -5) :
  let focus_y := (1 : ℝ) / (4 * a) + b in
  focus_y = -179 / 36 := by
  -- We skip the proof as we just need the statement.
  sorry

end parabola_focus_l7_7740


namespace linear_polynomial_of_conditions_l7_7940

noncomputable def sequence (P : ℤ → ℤ) (n : ℤ) : ℕ → ℤ
| 0       := n
| (k + 1) := P (sequence k)

theorem linear_polynomial_of_conditions (P : ℤ → ℤ) (n : ℤ) (h1 : ∃ b : ℕ, n > 0 ∧ ∀ b > 0, ∃ k : ℕ, ∃ t : ℤ, t ≥ 2 ∧ sequence P n k = t^b)
  (h2 : ∀ k : ℕ, sequence P n (k+1) = P (sequence P n k)) :
  ∃ a b : ℤ, ∀ x : ℤ, P x = a * x + b :=
begin
  -- Proof is omitted
  sorry
end

end linear_polynomial_of_conditions_l7_7940


namespace cake_eating_classmates_l7_7107

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l7_7107


namespace isosceles_triangle_ax_l7_7483

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem isosceles_triangle_ax (x : ℝ) :
  let A := (4, 1, 9)
      B := (10, -1, 6)
      C := (x, 4, 3) in
  distance A B = distance A C ↔ (x = 2 ∨ x = 6) := by
sorry

end isosceles_triangle_ax_l7_7483


namespace segment_length_of_absolute_value_l7_7205

theorem segment_length_of_absolute_value (x : ℝ) (h : abs (x - (27 : ℝ)^(1/3)) = 5) : 
  |8 - (-2)| = 10 :=
by
  sorry

end segment_length_of_absolute_value_l7_7205


namespace probability_lt_one_third_l7_7856

theorem probability_lt_one_third :
  let total_interval := (0 : ℝ, 1/2)
  let desired_interval := (0 : ℝ, 1/3)
  let total_length := (total_interval.2 - total_interval.1 : ℝ) = 1/2
  let desired_length := (desired_interval.2 - desired_interval.1 : ℝ) = 1/3
  -- then the probability P is given by:
  (desired_length / total_length) = 2/3
:=
by {
  -- definition of intervals and lengths
  let total_interval := (0 : ℝ, 1/2)
  let desired_interval := (0 : ℝ, 1/3)
  have total_length : total_interval.2 - total_interval.1 = 1/2,
  -- verify total length calculation
  calc
    total_interval.2 - total_interval.1 = 1/2 - 0 : by simp
                                 ... = 1/2      : by norm_num,
  have desired_length : desired_interval.2 - desired_interval.1 = 1/3,
  -- verify desired interval length calculation
  calc
    desired_interval.2 - desired_interval.1 = 1/3 - 0 : by simp
                                    ... = 1/3      : by norm_num,
  -- calculate probability
  set P := desired_length / total_length
  -- compute correct answer
  have : P = (1/3) / (1/2),
  calc
    (1/3) / (1/2) = (1/3) * (2/1) : by field_simp
              ...  = 2/3      : by norm_num,
  exact this
}

end probability_lt_one_third_l7_7856


namespace vote_count_l7_7437

theorem vote_count 
(h_total: 200 = h_votes + l_votes + y_votes)
(h_hl: 3 * l_votes = 2 * h_votes)
(l_ly: 6 * y_votes = 5 * l_votes):
h_votes = 90 ∧ l_votes = 60 ∧ y_votes = 50 := by 
sorry

end vote_count_l7_7437


namespace possible_classmates_l7_7138

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l7_7138


namespace lcm_of_1_to_12_l7_7231

noncomputable def lcm_1_to_12 : ℕ := 2^3 * 3^2 * 5 * 7 * 11

theorem lcm_of_1_to_12 : lcm_1_to_12 = 27720 := by
  sorry

end lcm_of_1_to_12_l7_7231


namespace other_candidate_votes_l7_7898

theorem other_candidate_votes (total_votes : ℕ) (invalid_percent : ℕ) (first_candidate_percent : ℕ) :
  total_votes = 8500 → invalid_percent = 25 → first_candidate_percent = 60 → 
  (total_votes * (100 - invalid_percent) / 100 * (100 - first_candidate_percent) / 100) = 2550 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end other_candidate_votes_l7_7898


namespace transformations_equivalent_l7_7585

def transform_graph_A (x : ℝ) : ℝ := 2 * x + π/6
def transform_graph_B (x : ℝ) : ℝ := 2 * x - π/3
def transform_graph_C (x : ℝ) : ℝ := 2 * x + 2 * π/3
def transform_graph_D (x : ℝ) : ℝ := 2 * (x + 2 * π/3)

theorem transformations_equivalent :
  (∀ x : ℝ, cos (transform_graph_A x) = cos (2 * x + π/6)) ∧
  (∀ x : ℝ, cos (transform_graph_B x) = cos (2 * x + π/6)) ∧
  (∀ x : ℝ, cos (transform_graph_C x) = cos (2 * x + π/6)) ∧
  (∀ x : ℝ, cos (transform_graph_D x) = cos (2 * x + π/6)) :=
by
  sorry

end transformations_equivalent_l7_7585


namespace eldest_sibling_age_correct_l7_7468

-- Definitions and conditions
def youngest_sibling_age (x : ℝ) := x
def second_youngest_sibling_age (x : ℝ) := x + 4
def third_youngest_sibling_age (x : ℝ) := x + 8
def fourth_youngest_sibling_age (x : ℝ) := x + 12
def fifth_youngest_sibling_age (x : ℝ) := x + 16
def sixth_youngest_sibling_age (x : ℝ) := x + 20
def seventh_youngest_sibling_age (x : ℝ) := x + 28
def eldest_sibling_age (x : ℝ) := x + 32

def combined_age_of_eight_siblings (x : ℝ) : ℝ := 
  youngest_sibling_age x +
  second_youngest_sibling_age x +
  third_youngest_sibling_age x +
  fourth_youngest_sibling_age x +
  fifth_youngest_sibling_age x +
  sixth_youngest_sibling_age x +
  seventh_youngest_sibling_age x +
  eldest_sibling_age x

-- Proving the combined age part
theorem eldest_sibling_age_correct (x : ℝ) (h : combined_age_of_eight_siblings x - youngest_sibling_age (x + 24) = 140) : 
  eldest_sibling_age x = 34.5 := by
  sorry

end eldest_sibling_age_correct_l7_7468


namespace optionA_optionB_optionC_optionD_l7_7256

theorem optionA (x : ℝ) (h : x ≠ 0) : x + 1/x ≠ 2 := sorry

theorem optionB (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) : 2/x + 1/y ≥ 8 := sorry

theorem optionC (x : ℝ) : sqrt (x^2 + 3) + 1/sqrt (x^2 + 3) ≠ 2 := sorry

theorem optionD (x : ℝ) (h : x < 0) : 2 + x + 1/x ≤ 0 := sorry

end optionA_optionB_optionC_optionD_l7_7256


namespace remaining_medieval_art_pieces_remaining_renaissance_art_pieces_remaining_modern_art_pieces_l7_7678

structure ArtCollection where
  medieval : ℕ
  renaissance : ℕ
  modern : ℕ

def AliciaArtCollection : ArtCollection := {
  medieval := 70,
  renaissance := 120,
  modern := 150
}

def donationPercentages : ArtCollection := {
  medieval := 65,
  renaissance := 30,
  modern := 45
}

def remainingArtPieces (initial : ℕ) (percent : ℕ) : ℕ :=
  initial - ((percent * initial) / 100)

theorem remaining_medieval_art_pieces :
  remainingArtPieces AliciaArtCollection.medieval donationPercentages.medieval = 25 := by
  sorry

theorem remaining_renaissance_art_pieces :
  remainingArtPieces AliciaArtCollection.renaissance donationPercentages.renaissance = 84 := by
  sorry

theorem remaining_modern_art_pieces :
  remainingArtPieces AliciaArtCollection.modern donationPercentages.modern = 83 := by
  sorry

end remaining_medieval_art_pieces_remaining_renaissance_art_pieces_remaining_modern_art_pieces_l7_7678


namespace fencing_required_l7_7292

theorem fencing_required
  (L : ℝ) (W : ℝ) (A : ℝ) (hL : L = 20) (hA : A = 80) (hW : A = L * W) :
  (L + 2 * W) = 28 :=
by {
  sorry
}

end fencing_required_l7_7292


namespace smallest_positive_period_of_f_minimum_value_of_f_value_of_f_inverse_at_1_l7_7414

noncomputable def f (x : ℝ) : ℝ := 2 * cos x * sin (x + (Real.pi / 3)) - sqrt 3 * (sin x)^2 + sin x * cos x

theorem smallest_positive_period_of_f : (∃ T > 0, ∀ x, f(x + T) = f x) ∧ (∀ T > 0, (∀ x, f(x + T) = f x) → T ≥ Real.pi) ∧ (∃ T > 0, ∀ x, f(x + T) = f x ∧ T = Real.pi) := 
sorry

theorem minimum_value_of_f : (∀ x : ℝ, (∃ k : ℤ, f x = -2 ∧ (x = k * Real.pi - (5 * Real.pi / 12)))) := 
sorry

theorem value_of_f_inverse_at_1 : (∀ x ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12), f (f⁻¹ 1) = 1 ∧ f⁻¹ 1 = Real.pi / 4) := 
sorry

end smallest_positive_period_of_f_minimum_value_of_f_value_of_f_inverse_at_1_l7_7414


namespace plantable_area_290_l7_7999

-- Definitions of the garden dimensions and conditions
def length (w : ℝ) : ℝ := 3 * w
def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w
def total_area (l w : ℝ) : ℝ := l * w
constant fountain_area : ℝ := 10
constant garden_perimeter : ℝ := 80

-- The problem statement we need to prove
theorem plantable_area_290 :
  ∀ (w : ℝ), perimeter (length w) w = garden_perimeter →
  total_area (length w) w - fountain_area = 290 :=
by
  intros
  sorry

end plantable_area_290_l7_7999


namespace shoes_difference_l7_7077

theorem shoes_difference :
  let pairs_per_box := 20
  let boxes_A := 8
  let boxes_B := 5 * boxes_A
  let total_pairs_A := boxes_A * pairs_per_box
  let total_pairs_B := boxes_B * pairs_per_box
  total_pairs_B - total_pairs_A = 640 :=
by
  sorry

end shoes_difference_l7_7077


namespace segment_length_l7_7215

theorem segment_length (x : ℝ) (h : |x - (27)^(1/3)| = 5) : ∃ a b : ℝ, (a = 8 ∧ b = -2 ∨ a = -2 ∧ b = 8) ∧ real.dist a b = 10 :=
by
  use [8, -2] -- providing the endpoints explicitly
  split
  -- prove that these are the correct endpoints
  · left; exact ⟨rfl, rfl⟩
  -- prove the distance is 10
  · apply real.dist_eq; linarith
  

end segment_length_l7_7215


namespace exist_triangle_with_given_conditions_l7_7324

noncomputable def construct_triangle (α : ℝ) (s_a : ℝ) (ε : ℝ) : Prop :=
  ∃ (A B C B* C* : Type) [triangle A B C] [triangle A B* C*],
  angle A B C = α ∧
  median_length A B C = s_a ∧ 
  angle_medians A B C = ε ∧
  (triangle A B C ∨ triangle A B* C*).

-- Example usage
theorem exist_triangle_with_given_conditions (α s_a ε : ℝ) : construct_triangle α s_a ε :=
  sorry

end exist_triangle_with_given_conditions_l7_7324


namespace sufficient_and_necessary_condition_l7_7936

def f (x : ℝ) : ℝ := x^3 + Real.logBase 2 (x + Real.sqrt (x^2 + 1))

theorem sufficient_and_necessary_condition {a b : ℝ} (h : a + b ≥ 0) :
  f(a) + f(b) ≥ 0 ↔ a + b ≥ 0 :=
by
  sorry

end sufficient_and_necessary_condition_l7_7936


namespace pascal_triangle_ratio_l7_7889

theorem pascal_triangle_ratio (n r : ℕ) 
  (h1 : (nat.choose n r) * 6 = (nat.choose n (r+1)) * 5)
  (h2 : (nat.choose n (r+1)) * 7 = (nat.choose n (r+2)) * 6) : 
  n = 142 :=
by 
  sorry

end pascal_triangle_ratio_l7_7889


namespace num_non_congruent_triangles_l7_7459

theorem num_non_congruent_triangles :
  (∀ (A B C : ℝ), ∃ (AC ∈ {4, 5, 7, 9, 11}), 
  let AB := 10 in 
  let angle_ABC := 30 in 
  number_of_non_congruent_triangles (A B C AC AB angle_ABC) = 6) :=
sorry

end num_non_congruent_triangles_l7_7459


namespace sin_translation_omega_min_l7_7193

theorem sin_translation_omega_min (ω : ℝ) (hω : ω > 0) :
  (∃ y : ℝ → ℝ, y = (λ x, sin (ω * (x - π / 4))) ∧ y (3 * π / 4) = 0) → ω = 2 :=
by
  sorry

end sin_translation_omega_min_l7_7193


namespace propositions_correct_l7_7395

axiom line : Type
axiom plane : Type
axiom perp : line → plane → Prop
axiom perp_line : line → line → Prop
axiom parallel : line → line → Prop
axiom parallel_plane : plane → plane → Prop
axiom in_plane : line → plane → Prop
axiom intersection : plane → plane → line

variables (alpha beta gamma : plane) (m n : line)

-- Proposition 1 conditions: if alpha is perpendicular to beta, alpha ∩ beta = m, and n is perpendicular to m
axiom prop1_cond1 : perp m alpha
axiom prop1_cond2 : perp m beta
axiom prop1_cond3 : perp_line n m

-- Proposition 2 conditions: if alpha is parallel to beta, and their intersections with gamma are m and n respectively, then m is parallel to n
axiom prop2_cond1 : parallel_plane alpha beta
axiom prop2_cond2 : intersection alpha gamma = m
axiom prop2_cond3 : intersection beta gamma = n

-- Proposition 3 conditions: if m is not perpendicular to alpha then
axiom prop3_cond : ¬ perp m alpha

-- Proposition 4 conditions: if alpha ∩ beta = m, and n is parallel to m, n not in alpha or beta
axiom prop4_cond1 : intersection alpha beta = m
axiom prop4_cond2 : parallel n m
axiom prop4_cond3 : ¬ in_plane n alpha
axiom prop4_cond4 : ¬ in_plane n beta

-- Proof statement combining the conditions and correct answers
theorem propositions_correct :
  (prop2_cond1 → prop2_cond2 → prop2_cond3 → parallel m n) ∧
  (prop4_cond1 → prop4_cond2 → prop4_cond3 → prop4_cond4 → (parallel_plane n alpha ∧ parallel_plane n beta)) :=
sorry

end propositions_correct_l7_7395


namespace cake_sharing_l7_7099

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l7_7099


namespace solve_for_x_l7_7782

theorem solve_for_x (a x : ℝ) (h1 : 9^a = 3) (h2 : Real.log x = a) : x = Real.exp (1/2) :=
by sorry

end solve_for_x_l7_7782


namespace find_ellipse_eq_l7_7813

-- Conditions
def center_at_origin (C : ℝ × ℝ) : Prop := C = (0, 0)
def focus_on_x_axis (focus : ℝ × ℝ) : Prop := ∃ x : ℝ, focus = (x, 0)
def eccentricity (c a : ℝ) : Prop := c / a = (Real.sqrt 2) / 2
def vertex_is_focus (vertex parabola_focus : ℝ × ℝ) : Prop := vertex = parabola_focus

-- Definitions based on the problem
def parabola_focus : ℝ × ℝ := (2, 1) -- Since vertex of the ellipse is the focus of parabola x^2 = 4y, focus here is calculated
def standard_eq (C : ℝ × ℝ) (a b : ℝ) : Prop := 𝒳 = C ∧ ∃ a b : ℝ, a^2 = b^2 + (b / ((Real.sqrt 2) / 2))^2


-- Proof Problem Statement
theorem find_ellipse_eq : 
  ∃ (a b : ℝ), center_at_origin (0, 0) ∧ 
               focus_on_x_axis (b / (Real.sqrt 2) / 2, 0) ∧ 
               eccentricity (b / ((Real.sqrt 2) / 2)) a ∧ 
               vertex_is_focus (√2, 1) (2, 1) → 
  (standard_eq (0, 0) √2 1) :=
by 
  sorry

end find_ellipse_eq_l7_7813


namespace quadratic_decreases_after_vertex_l7_7973

theorem quadratic_decreases_after_vertex :
  ∀ x : ℝ, (x > 2) → (y = -(x - 2)^2 + 3) → ∃ k : ℝ, k < 0 :=
by
  sorry

end quadratic_decreases_after_vertex_l7_7973


namespace part1_part2_part3_l7_7906

structure Point (α : Type) := 
  (x : α) 
  (y : α)

noncomputable def long_distance (P : Point ℝ) : ℝ := 
  max (abs P.x) (abs P.y)

def is_perfect_point (Q : Point ℝ) : Prop := 
  abs Q.x = abs Q.y

variable {A : Point ℝ} [hA : A = ⟨-1, 3⟩]
variable {B : Point ℝ} [hB : B = ⟨4 * a - 1, -3⟩] (a : ℝ)
variable {C : Point ℝ} (b : ℝ) [hC : C = ⟨-2, 3 * b - 2⟩] [hC_long : long_distance C = 4] [hC_quad : C.x < 0 ∧ C.y > 0]
variable D : Point ℝ [hD : D = ⟨9 - 2 * b, -5⟩]

theorem part1 : long_distance A = 3 := 
sorry

theorem part2 : abs (4 * a - 1) = 3 →
  a = 1 ∨ a = -1/2 := 
sorry

theorem part3 : b = 2 →
  is_perfect_point D := 
sorry

end part1_part2_part3_l7_7906


namespace mary_keep_warm_hours_l7_7952

-- Definitions based on the conditions
def sticks_from_chairs (chairs : ℕ) : ℕ := chairs * 6
def sticks_from_tables (tables : ℕ) : ℕ := tables * 9
def sticks_from_stools (stools : ℕ) : ℕ := stools * 2
def sticks_needed_per_hour : ℕ := 5

-- Given counts of furniture
def chairs : ℕ := 18
def tables : ℕ := 6
def stools : ℕ := 4

-- Total number of sticks
def total_sticks : ℕ := (sticks_from_chairs chairs) + (sticks_from_tables tables) + (sticks_from_stools stools)

-- Proving the number of hours Mary can keep warm
theorem mary_keep_warm_hours : total_sticks / sticks_needed_per_hour = 34 := by
  sorry

end mary_keep_warm_hours_l7_7952


namespace cannot_remain_2_l7_7199

theorem cannot_remain_2 : 
  ¬∃ (f : Fin 2013 → ℤ), 
    (∀ (A B : ℤ) (h : A ∈ set.range f.val ∧ B ∈ set.range f.val), 
      ∃ (g : Fin (2013 - 1) → ℤ), 
      (rforall (i : Fin (2013 - 1)), 
        if i.val = 0 
        then g i = A - B 
        else g i = f⟨i.val + 2, sorry⟩)) 
    ∧ ∀ (j : Fin 1), g j = 2 :=
begin
  sorry
end

end cannot_remain_2_l7_7199


namespace max_availability_day_exists_l7_7730

structure TeamMember :=
  (name : String)
  (availability : List Bool) -- True if available

def team : List TeamMember :=
  [⟨"Alice", [false, true, true, false, true]⟩,
   ⟨"Bob", [true, false, false, true, false]⟩,
   ⟨"Charlie", [true, true, false, false, false]⟩,
   ⟨"Diana", [false, false, true, true, true]⟩]

def count_available (day : Fin 5) : Nat :=
  team.foldl (fun acc member => if member.availability.get day.val then acc + 1 else acc) 0

theorem max_availability_day_exists :
  ∃ day : Fin 5, count_available day = 2 := by
  sorry

end max_availability_day_exists_l7_7730


namespace classify_numbers_l7_7074

def isDecimal (n : ℝ) : Prop :=
  ∃ (i : ℤ) (f : ℚ), n = i + f ∧ i ≠ 0

def isNatural (n : ℕ) : Prop :=
  n ≥ 0

theorem classify_numbers :
  (isDecimal 7.42) ∧ (isDecimal 3.6) ∧ (isDecimal 5.23) ∧ (isDecimal 37.8) ∧
  (isNatural 5) ∧ (isNatural 100) ∧ (isNatural 502) ∧ (isNatural 460) :=
by
  sorry

end classify_numbers_l7_7074


namespace mikhail_pronounces_more_words_l7_7492

theorem mikhail_pronounces_more_words :
  let konstantin_range := finset.Icc 180 220
  let mikhail_range := finset.Icc 191 231
  let konstantin_words := (konstantin_range.card - 2) * 3 + 2 + 2
  let mikhail_words := (mikhail_range.card - 3) * 3 + 2 + 3
  (mikhail_words > konstantin_words) ∧ (mikhail_words = konstantin_words + 1) :=
by
  sorry

end mikhail_pronounces_more_words_l7_7492


namespace curve_C_standard_equation_line_l1_equation_with_conditions_no_such_line_l2_exists_l7_7800

-- Define the conditions
variables {x0 y0 x y : ℝ}
variable A : ℝ × ℝ := (x0, 0)
variable B : ℝ × ℝ := (0, y0)

-- Define the given conditions in Lean
def point_P_movement_condition (P : ℝ × ℝ) : Prop :=
  let OA := 2 • A in
  let OB := (Real.sqrt 3) • B in
  P = OA + OB

def distance_AB_condition : Prop :=
  (x0^2 + y0^2 = 1)

-- Define the proof problem

-- Part (I): Curve C equation
theorem curve_C_standard_equation (P : ℝ × ℝ) (hP : point_P_movement_condition P) (hAB : distance_AB_condition) :
  ∃ (C: ℝ×ℝ → Prop), C = (λ p, (p.1^2 / 4 + p.2^2 / 3 = 1)) :=
sorry

-- Part (II): Equation of line l_1
theorem line_l1_equation_with_conditions :
  ∃ k : ℝ, (k = 2 * Real.sqrt 3 / 3 ∨ k = -2 * Real.sqrt 3 / 3) ∧
  ∀ P Q : ℝ × ℝ, (P.2 = k * P.1 + 2) ∧ (Q.2 = k * Q.1 + 2) ∧
  ((P.1 * Q.1 + P.2 * Q.2 = 0) → (P ≠ (0,0) ∧ Q ≠ (0,0))) :=
sorry

-- Part (III): Line l_2 and area condition
theorem no_such_line_l2_exists :
  ¬ ∃ t : ℝ, let E := (1, 0) in
  let A :=
    λ y, (ty + 1, y) in
  let B := (E, A) in
  let triangle_area :=
    0.5 * (3t^2 + 4) * (y.2 (1 - (√3.5)) ) in
  triangle_area = 2 * √3 :=
sorry

end curve_C_standard_equation_line_l1_equation_with_conditions_no_such_line_l2_exists_l7_7800


namespace chessboard_circle_area_ratio_l7_7004

theorem chessboard_circle_area_ratio :
  let side := 8
  let radius := 4
  let S1 := (Real.pi * radius^2 : ℝ)
  let S2 := (side^2 : ℝ) - S1 in
  ⌊S1 / S2⌋ = 3 :=
by
  let side := 8
  let radius := 4
  let S1 := (Real.pi * radius^2 : ℝ)
  let S2 := (side^2 : ℝ) - S1
  have : ⌊S1 / S2⌋ = 3 := sorry
  exact this

end chessboard_circle_area_ratio_l7_7004


namespace transformations_equivalent_l7_7586

def transform_graph_A (x : ℝ) : ℝ := 2 * x + π/6
def transform_graph_B (x : ℝ) : ℝ := 2 * x - π/3
def transform_graph_C (x : ℝ) : ℝ := 2 * x + 2 * π/3
def transform_graph_D (x : ℝ) : ℝ := 2 * (x + 2 * π/3)

theorem transformations_equivalent :
  (∀ x : ℝ, cos (transform_graph_A x) = cos (2 * x + π/6)) ∧
  (∀ x : ℝ, cos (transform_graph_B x) = cos (2 * x + π/6)) ∧
  (∀ x : ℝ, cos (transform_graph_C x) = cos (2 * x + π/6)) ∧
  (∀ x : ℝ, cos (transform_graph_D x) = cos (2 * x + π/6)) :=
by
  sorry

end transformations_equivalent_l7_7586


namespace dihedral_angle_relationships_l7_7508

variables {α β : Plane} {l a b : Line}

-- Definitions of dihedral angle and relationships
def is_dihedral_angle (α l β : Plane) : Prop := sorry
def line_in_plane (a : Line) (α : Plane) : Prop := sorry
def not_perpendicular (a l : Line) : Prop := ¬(a ⟂ l)

theorem dihedral_angle_relationships
  (h1 : is_dihedral_angle α l β)
  (h2 : line_in_plane a α)
  (h3 : line_in_plane b β)
  (h4 : not_perpendicular a l)
  (h5 : not_perpendicular b l) :
  (a ⟂ b ∨ a ∥ b) :=
sorry

end dihedral_angle_relationships_l7_7508


namespace degree_of_polynomial_l7_7792

-- Defining the polynomial P and the sequence a_1, a_2, a_3, ...
def infinite_sequence (P : ℝ → ℝ) (a : ℕ → ℕ) : Prop :=
  (P a 1 = 0) ∧ (P a 2 = a 1) ∧ (P a 3 = a 2) ∧ ∀ n : ℕ, P a (n + 1) = a n

-- Defining the condition that P is a polynomial with real coefficients and the degree is 1.
def is_degree_one_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ c1 c0 : ℝ, c1 ≠ 0 ∧ ∀ x : ℝ, P x = c1 * x + c0

-- The main theorem stating that the degree of polynomial P satisfying the given conditions is 1.
theorem degree_of_polynomial (P : ℝ → ℝ) (a : ℕ → ℕ) (h : infinite_sequence P a) :
  is_degree_one_polynomial P :=
sorry

end degree_of_polynomial_l7_7792


namespace smallest_five_digit_equiv_11_mod_13_l7_7217

open Nat

theorem smallest_five_digit_equiv_11_mod_13 :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 13 = 11 ∧ n = 10009 :=
by
  sorry

end smallest_five_digit_equiv_11_mod_13_l7_7217


namespace marlon_gift_card_balance_l7_7069

theorem marlon_gift_card_balance 
  (initial_amount : ℕ) 
  (spent_monday : initial_amount / 2 = 100)
  (spent_tuesday : (initial_amount / 2) / 4 = 25) 
  : (initial_amount / 2) - (initial_amount / 2 / 4) = 75 :=
by
  sorry

end marlon_gift_card_balance_l7_7069


namespace parabola_shift_l7_7988

theorem parabola_shift (x : ℝ) : 
  let y := x^2 - 4*x - 4
  in (y = (x + 1)^2 - 5) :=
by
  sorry

end parabola_shift_l7_7988


namespace sons_ages_l7_7436

theorem sons_ages (x y : ℕ) (h1 : 2 * x = x + y + 18) (h2 : y = (x - y) - 6) : 
  x = 30 ∧ y = 12 := by
  sorry

end sons_ages_l7_7436


namespace distance_between_points_on_line_l7_7323

theorem distance_between_points_on_line
  (p q r s m n : ℝ)
  (h1 : q = n * p + m)
  (h2 : s = n * r + m) :
  real.dist (p, q) (r, s) = |r - p| * real.sqrt (1 + n ^ 2) := by
  sorry

end distance_between_points_on_line_l7_7323


namespace mock_exam_girls_count_l7_7637

theorem mock_exam_girls_count
  (B G Bc Gc : ℕ)
  (h1: B + G = 400)
  (h2: Bc = 60 * B / 100)
  (h3: Gc = 80 * G / 100)
  (h4: Bc + Gc = 65 * 400 / 100)
  : G = 100 :=
sorry

end mock_exam_girls_count_l7_7637


namespace angle_A_is_60_degrees_l7_7460

theorem angle_A_is_60_degrees 
  (a b c : ℝ) 
  (h : (a + b + c) * (b + c - a) = 3 * b * c) : 
  real.angle = 60 := 
sorry

end angle_A_is_60_degrees_l7_7460


namespace final_combined_price_eq_original_price_l7_7280

variable {J : ℝ}
variable (h₁ : J > 0)
variable original_shoes_price : ℝ := 1.10 * J
variable reduced_jacket_price : ℝ := 0.85 * J
variable sale_jacket_price : ℝ := 0.595 * J
variable sale_shoes_price : ℝ := 0.77 * J
variable final_jacket_price : ℝ := 0.6545 * J
variable final_shoes_price : ℝ := 0.847 * J
variable combined_final_price : ℝ := final_jacket_price + final_shoes_price + (J - final_jacket_price) + (original_shoes_price - final_shoes_price)

/-- Proof that the combined final price of the jacket and the shoes, after all reductions, taxes, 
and price increases is equal to 2.10 times the original price of the jacket given initial conditions 
--/
theorem final_combined_price_eq_original_price : combined_final_price = 2.10 * J := by
  sorry

end final_combined_price_eq_original_price_l7_7280


namespace domain_range_0_1_domain_range_1_4_2_l7_7404

/-
  Given the range of the function y = x^2 - 2x + 2 is [1,2], prove that the possible domains of the function 
  that result in this range include [0,1] and [1/4, 2].
-/
def quadratic_function (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem domain_range_0_1 : 
  ∃ D : set ℝ, D = set.Icc 0 1 ∧ (∀ x ∈ D, 1 ≤ quadratic_function x ∧ quadratic_function x ≤ 2) :=
sorry

theorem domain_range_1_4_2 : 
  ∃ D : set ℝ, D = set.Icc (1/4 : ℝ) 2 ∧ (∀ x ∈ D, 1 ≤ quadratic_function x ∧ quadratic_function x ≤ 2) :=
sorry

end domain_range_0_1_domain_range_1_4_2_l7_7404


namespace napkin_length_proof_l7_7536

-- Definitions and conditions
def tablecloth_length : ℕ := 102
def tablecloth_width : ℕ := 54
def napkins_count : ℕ := 8
def napkin_width : ℕ := 7
def total_material : ℕ := 5844

def tablecloth_area : ℕ := tablecloth_length * tablecloth_width -- 5508 square inches

-- The proof goal
theorem napkin_length_proof 
  (total_material_eq: total_material = 5844) 
  (tablecloth_area_eq: tablecloth_area = tablecloth_length * tablecloth_width)
  (napkins_count_eq: napkins_count = 8) 
  (napkin_width_eq: napkin_width = 7) : 
  let area_needed_for_napkins := total_material - tablecloth_area in
  let area_one_napkin := area_needed_for_napkins / napkins_count in
  let napkin_length := area_one_napkin / napkin_width in
  napkin_length = 6 :=
by
  have h1 : tablecloth_area = 5508 := by sorry
  have h2 : area_needed_for_napkins = total_material - tablecloth_area := by sorry
  have h3 : area_needed_for_napkins = 336 := by sorry
  have h4 : area_one_napkin = area_needed_for_napkins / napkins_count := by sorry
  have h5 : area_one_napkin = 42 := by sorry
  have h6 : napkin_length = area_one_napkin / napkin_width := by sorry
  have h7 : napkin_length = 6 := by sorry
  exact h7

end napkin_length_proof_l7_7536


namespace magician_card_drawings_l7_7283

theorem magician_card_drawings : 
  let total_cards := 256
  let doubles_count := 16
  let valid_cards_per_double := total_cards - doubles_count - (15 * 2)
  let only_one_double_selections := doubles_count * valid_cards_per_double
  let both_doubles_selections := Nat.choose 16 2
  let total_ways := both_doubles_selections + only_one_double_selections
  total_ways = 3480 :=
by
  let total_cards := 256
  let doubles_count := 16
  let valid_cards_per_double := total_cards - doubles_count - 30
  let only_one_double_selections := doubles_count * valid_cards_per_double
  let both_doubles_selections := Nat.choose 16 2
  let total_ways := both_doubles_selections + only_one_double_selections
  exact Eq.refl total_ways

end magician_card_drawings_l7_7283


namespace inequality_and_equality_condition_l7_7051

open Real

theorem inequality_and_equality_condition
  {a b : ℝ} (ha : 0 < a) (hb : 0 < b) {n : ℕ} (hn : 2 ≤ n) :
  (n + 1) * (∑ i in Finset.range (n + 1), a^(n - i) * b^i) ≥ (a + b)^n * 2^n ∧
  (n + 1) * (∑ i in Finset.range (n + 1), a^(n - i) * b^i) = (a + b)^n * 2^n ↔ a = b := 
sorry

end inequality_and_equality_condition_l7_7051


namespace sum_of_digits_of_smallest_init_l7_7692

def bernardo_wins (init : ℕ) : Prop :=
  2 * init < 1000 ∧ 3 * (2 * init) ≥ 1000 ∧ init ≥ 0 ∧ init ≤ 999

def smallest_init_win_for_bernardo : ℕ :=
  Nat.find (⟨28, by
    apply And.intro
    apply and.intro
    norm_num
    norm_num
    norm_num
    norm_num⟩)

theorem sum_of_digits_of_smallest_init : Nat.digits 10 smallest_init_win_for_bernardo.sum = 10 :=
by
  have h_find : smallest_init_win_for_bernardo = 28 := by
    unfold smallest_init_win_for_bernardo
    apply Nat.find_spec
  rw h_find
  norm_num
  sorry

end sum_of_digits_of_smallest_init_l7_7692


namespace cost_of_eight_books_l7_7455

theorem cost_of_eight_books (x : ℝ) (h : 2 * x = 34) : 8 * x = 136 :=
by
  sorry

end cost_of_eight_books_l7_7455


namespace complex_fraction_evaluation_l7_7041

noncomputable theory
open Complex

theorem complex_fraction_evaluation {a b : ℂ} (nz_a : a ≠ 0) (nz_b : b ≠ 0) 
  (h : a^3 + a^2 * b + a * b^2 + b^3 = 0) : (a^12 + b^12) / (a + b)^12 = 2 / 81 := 
sorry

end complex_fraction_evaluation_l7_7041


namespace a2021_equals_6_l7_7810

def sequence (a : ℕ → ℕ) := 
  a 0 = 0 ∧ 
  a 1 = 1 ∧ 
  a 2 = 1 ∧ 
  (∀ n, a (3 * n) = a n) ∧ 
  (∀ n, a (3 * n + 1) = a n + 1) ∧ 
  (∀ n, a (3 * n + 2) = a n + 1)

theorem a2021_equals_6 (a : ℕ → ℕ) (h : sequence a) : a 2021 = 6 :=
by
  -- Proof omitted, insert proof here
  sorry

end a2021_equals_6_l7_7810


namespace f_sum_always_negative_l7_7722

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f (x + 4)) ∧ (∀ x, x > 2 → strict_mono f)

theorem f_sum_always_negative (f : ℝ → ℝ)
  (h_conditions : satisfies_conditions f) :
  ∀ x₁ x₂, x₁ + x₂ < 4 → (x₁ - 2) * (x₂ - 2) < 0 → f x₁ + f x₂ < 0 :=
by 
  sorry

end f_sum_always_negative_l7_7722


namespace circle_area_eq_25pi_l7_7724

theorem circle_area_eq_25pi :
  (∃ (x y : ℝ), x^2 + y^2 - 4 * x + 6 * y - 12 = 0) →
  (∃ (area : ℝ), area = 25 * Real.pi) :=
by
  sorry

end circle_area_eq_25pi_l7_7724


namespace even_integers_diff_digits_200_to_800_l7_7430

theorem even_integers_diff_digits_200_to_800 :
  ∃ n : ℕ, n = 131 ∧ (∀ x : ℕ, 200 ≤ x ∧ x < 800 ∧ (x % 2 = 0) ∧ (∀ i j : ℕ, i ≠ j → (x / 10^i % 10) ≠ (x / 10^j % 10)) ↔ x < n) :=
sorry

end even_integers_diff_digits_200_to_800_l7_7430


namespace arithmetic_sequence_ratio_l7_7168

theorem arithmetic_sequence_ratio (a x b : ℝ) 
  (h1 : x - a = b - x)
  (h2 : 2 * x - b = b - x) :
  a / b = 1 / 3 :=
by
  sorry

end arithmetic_sequence_ratio_l7_7168


namespace repeating_decimals_sum_l7_7706

-- Definitions of repeating decimals as fractions
def repeatingDecimalToFrac (a b : ℕ) : ℚ :=
  (a : ℚ) / (b : ℚ)

-- Given conditions
def x : ℚ := repeatingDecimalToFrac 1 3
def y : ℚ := repeatingDecimalToFrac 2 3

-- Theorem statement
theorem repeating_decimals_sum : x + y = 1 := 
begin
  sorry
end

end repeating_decimals_sum_l7_7706


namespace distance_AB_l7_7524

def A : ℝ := -1
def B : ℝ := 2023

theorem distance_AB : |B - A| = 2024 := by
  sorry

end distance_AB_l7_7524


namespace c_sum_formula_l7_7384

noncomputable section

def arithmetic_sequence (a : Nat -> ℚ) : Prop :=
  a 3 = 2 ∧ (a 1 + 2 * ((a 2 - a 1) : ℚ)) = 2

def geometric_sequence (b : Nat -> ℚ) (a : Nat -> ℚ) : Prop :=
  b 1 = a 1 ∧ b 4 = a 15

def c_sequence (a : Nat -> ℚ) (b : Nat -> ℚ) (n : Nat) : ℚ :=
  a n + b n

def Tn (c : Nat -> ℚ) (n : Nat) : ℚ :=
  (Finset.range n).sum c

theorem c_sum_formula
  (a b c : Nat -> ℚ)
  (k : Nat) 
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b a)
  (hc : ∀ n, c n = c_sequence a b n) :
  Tn c k = k * (k + 3) / 4 + 2^k - 1 :=
by
  sorry

end c_sum_formula_l7_7384


namespace sequence_2014_l7_7549

-- Definition of the sequence and its repetition pattern
def sequence : ℕ → ℕ
| n := let s := λ k, List.replicate k k;
       (List.join (List.range' 1 (n + 1)).map s).get! (n - 1)

-- The main theorem to be proven
theorem sequence_2014 : sequence 2014 = 13 :=
sorry

end sequence_2014_l7_7549


namespace parabola_focus_hyperbola_focus_l7_7450

open Real

theorem parabola_focus_hyperbola_focus (p : ℝ) (h : p > 0) 
  (focus_parabola : (2 * p, 0) ∈ {(4, 0), (-4, 0)}) :
  p = 8 :=
sorry

end parabola_focus_hyperbola_focus_l7_7450


namespace cost_of_each_math_book_l7_7599

theorem cost_of_each_math_book 
(total_books : ℕ) 
(math_books : ℕ) 
(history_books : ℕ) 
(cost_history_book : ℕ) 
(total_cost : ℕ) 
(cost_math_book : ℕ) :
  total_books = 80 →
  math_books = 32 →
  history_books = total_books - math_books →
  cost_history_book = 5 →
  total_cost = 368 →
  32 * cost_math_book + history_books * cost_history_book = total_cost →
  cost_math_book = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  -- The proof would continue from here to solve the equation
  -- 32 * cost_math_book + 48 * 5 = 368
  -- and show that cost_math_book = 4
  sorry

end cost_of_each_math_book_l7_7599


namespace how_many_more_slices_needed_l7_7688

theorem how_many_more_slices_needed
  (ham_per_sandwich : Nat)
  (slices_Available : Nat)
  (sandwiches_needed : Nat) :
  ham_per_sandwich = 3 →
  slices_Available = 31 →
  sandwiches_needed = 50 →
  (sandwiches_needed * ham_per_sandwich - slices_Available) = 119 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end how_many_more_slices_needed_l7_7688


namespace collinear_PDQ_l7_7943

structure Triangle :=
  (A B C : Point)

structure PointOfTangency (ABC : Triangle) :=
  (D E F : Point) -- D on [BC], E on [CA], F on [AB]

structure CollinearityConditions (ABC : Triangle) (tangency_points : PointOfTangency ABC) :=
  (P Q : Point)
  (PF_eq_FB : PF P tangency_points.F = PF tangency_points.F B)
  (FP_parallel_AC : parallel_line (FP P) (line AC))
  (P_C_same_side_AB : same_side (line AB) P C)
  (QE_eq_EC : QE Q tangency_points.E = QE tangency_points.E C)
  (EQ_parallel_AB : parallel_line (EQ Q) (line AB))
  (Q_B_same_side_AC : same_side (line AC) Q B)

theorem collinear_PDQ (ABC : Triangle) (tangency_points : PointOfTangency ABC)
  (conds : CollinearityConditions ABC tangency_points) : collinear conds.P tangency_points.D conds.Q := sorry

end collinear_PDQ_l7_7943


namespace evaluate_expression_l7_7167

theorem evaluate_expression : (20 + 22) / 2 = 21 := by
  sorry

end evaluate_expression_l7_7167


namespace canonical_equation_of_line_l7_7624

noncomputable theory

open_locale classical 

theorem canonical_equation_of_line 
  (x y z : ℝ)
  (h1 : 3 * x + 3 * y + z - 1 = 0)
  (h2 : 2 * x - 3 * y - 2 * z + 6 = 0) :
  (x + 1) / -3 = (y - 4 / 3) / 8 ∧ (x + 1) / -3 = z / -15 :=
sorry

end canonical_equation_of_line_l7_7624


namespace part_1_select_B_prob_part_2_select_BC_prob_l7_7000

-- Definitions for the four students
inductive Student
| A
| B
| C
| D

open Student

-- Definition for calculating probability
def probability (favorable total : Nat) : Rat :=
  favorable / total

-- Part (1)
theorem part_1_select_B_prob : probability 1 4 = 1 / 4 :=
  sorry

-- Part (2)
theorem part_2_select_BC_prob : probability 2 12 = 1 / 6 :=
  sorry

end part_1_select_B_prob_part_2_select_BC_prob_l7_7000


namespace sum_of_digits_nine_ab_is_28314_l7_7063

noncomputable def a_sequence_2023_nines : ℕ := (10 ^ 2023) - 1
noncomputable def b_sequence_2023_sixes : ℕ := (6 * ((10 ^ 2023) - 1)) / 9
noncomputable def nine_ab : ℕ := 9 * a_sequence_2023_nines * b_sequence_2023_sixes

theorem sum_of_digits_nine_ab_is_28314 :
  nat.digits 10 nine_ab.sum = 28314 := sorry

end sum_of_digits_nine_ab_is_28314_l7_7063


namespace mike_payment_l7_7958

def xray_cost : ℕ := 250
def mri_cost : ℕ := 3 * xray_cost
def ctscan_cost : ℕ := 2 * mri_cost
def blood_tests_cost : ℕ := 200

def insurance_xray_mri : ℚ := 0.80
def insurance_ctscan : ℚ := 0.70
def insurance_blood_tests : ℚ := 0.50

def total_payment : ℕ := 
  xray_cost - (insurance_xray_mri * xray_cost).toNat +
  mri_cost - (insurance_xray_mri * mri_cost).toNat +
  ctscan_cost - (insurance_ctscan * ctscan_cost).toNat +
  blood_tests_cost - (insurance_blood_tests * blood_tests_cost).toNat

theorem mike_payment : total_payment = 750 := by
  sorry

end mike_payment_l7_7958


namespace lcm_1_to_12_l7_7242

theorem lcm_1_to_12 : Nat.lcm_list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 27720 := by
  sorry

end lcm_1_to_12_l7_7242


namespace find_middle_and_oldest_sons_l7_7656

-- Defining the conditions
def youngest_age : ℕ := 2
def father_age : ℕ := 33
def father_age_in_12_years : ℕ := father_age + 12
def youngest_age_in_12_years : ℕ := youngest_age + 12

-- Lean theorem statement to find the ages of the middle and oldest sons
theorem find_middle_and_oldest_sons (y z : ℕ) (h1 : father_age_in_12_years = (youngest_age_in_12_years + 12 + y + 12 + z + 12)) :
  y = 3 ∧ z = 4 :=
sorry

end find_middle_and_oldest_sons_l7_7656


namespace range_abs_diff_function_l7_7713

noncomputable def abs_diff_function (x : ℝ) : ℝ :=
|x + 10| - |x - 5|

theorem range_abs_diff_function : (set.range abs_diff_function) = set.Icc (-15 : ℝ) 25 := 
sorry

end range_abs_diff_function_l7_7713


namespace inclination_angle_of_line_inclination_angle_of_line_l7_7561

theorem inclination_angle_of_line (m : ℝ) (h : m = 1) : inclination_angle (Line.mk m (-1)) = 45 :=
by 
  sorry

def inclination_angle (l : Line) : ℝ := 
  if l.slope = 1 then 45 else 0  -- Simplified definition for the purpose of this example, in practice use appropriate trigonometric functions

noncomputable def Line : Type := 
  {slope : ℝ  -- Slope of the line
   constant : ℝ} -- Intercept

theorem inclination_angle_of_line (m : ℝ) (h : m = 1) : inclination_angle (Line.mk m (-1)) = 45 := -- Proving that the inclination angle is 45 degrees when the slope is 1
by 
  sorry

end inclination_angle_of_line_inclination_angle_of_line_l7_7561


namespace classmates_ate_cake_l7_7097

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l7_7097


namespace fishing_line_sections_l7_7031

theorem fishing_line_sections (reels : ℕ) (length_per_reel : ℕ) (section_length : ℕ)
    (h_reels : reels = 3) (h_length_per_reel : length_per_reel = 100) (h_section_length : section_length = 10) :
    (reels * length_per_reel) / section_length = 30 := 
by
  rw [h_reels, h_length_per_reel, h_section_length]
  norm_num

end fishing_line_sections_l7_7031


namespace smallest_sum_S_l7_7245

def sum_dice_transformation (n : ℕ) (S : ℕ) : Prop :=
  (∀ (d : ℕ → ℕ), (∀ i, d i ∈ {1, 2, 3, 4, 5, 6, 7, 8}) → 
    (∑ i in finset.range n, d i = 2400) ↔ 
    (∑ i in finset.range n, (9 - d i) = S))

theorem smallest_sum_S : ∃ (n : ℕ), 8 * n ≥ 2400 ∧ ∃ (S : ℕ), S = 9 * n - 2400 ∧ sum_dice_transformation n S ∧ S = 300 :=
by
  sorry

end smallest_sum_S_l7_7245


namespace main_proof_l7_7926

variables {n : ℕ} {𝕜 : Type*} [Field 𝕜]
variables (X : Matrix (Fin n) (Fin n) 𝕜) (Y : Matrix (Fin n) (Fin n) 𝕜)
variables (X_inv : Matrix (Fin n) (Fin n) 𝕜) (Y_inv : Matrix (Fin n) (Fin n) 𝕜)

-- Condition: X is an invertible matrix
@[simp] def is_invertible_X : Prop := X.det ≠ 0

-- Condition: Y is defined as a matrix with columns X₂, X₃, ..., Xₙ, 0
@[simp] def def_Y : Prop :=
∀ (i : Fin (n-1)), Y.vecCons (X i.succ) 0 = X (i : Fin n)

-- Definitions of A and B
noncomputable def A : Matrix (Fin n) (Fin n) 𝕜 := Y * X⁻¹
noncomputable def B : Matrix (Fin n) (Fin n) 𝕜 := X⁻¹ * Y

-- Proof goals
def rank_and_eigenvalues_A : Prop :=
A.rank = n-1 ∧ ∀ λ, (A.eigenval λ) → λ = 0

def rank_and_eigenvalues_B : Prop :=
B.rank = n-1 ∧ ∀ λ, (B.eigenval λ) → λ = 0

-- Main statement
theorem main_proof : is_invertible_X X ∧ def_Y Y X → rank_and_eigenvalues_A X Y ∧ rank_and_eigenvalues_B X Y :=
by sorry

end main_proof_l7_7926


namespace union_intersection_cardinality_l7_7080

open_locale classical

variable {α : Type*}

theorem union_intersection_cardinality (A : finset (set α)) (k : ℕ) (h : A.card = k):
  (finset.sum (finset.powerset A) (λ s, (s.val.to_finset.sum (λ t, (t.card : ℕ))))).card = 
  (2^k - 1) * (finset.sum (finset.powerset A) (λ s, (s.val.to_finset.prod (λ t, (t.card : ℕ))))).card :=
sorry

end union_intersection_cardinality_l7_7080


namespace tangent_line_range_l7_7282

noncomputable def curve : ℝ → ℝ := λ x, x^3 - 12 * x

theorem tangent_line_range (t : ℝ) :
  (∃ (x : ℝ), 2 * x^3 - 3 * x^2 + 12 + t = 0 ∧ (x = 0 ∨ x = 1)) → 
  (t + 12) * (t + 11) < 0 → -12 < t ∧ t < -11 :=
  sorry

end tangent_line_range_l7_7282


namespace possible_number_of_classmates_l7_7128

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l7_7128


namespace sum_three_consecutive_integers_divisible_by_three_l7_7935

theorem sum_three_consecutive_integers_divisible_by_three (a : ℕ) (h : 1 < a) :
  (a - 1) + a + (a + 1) % 3 = 0 :=
by
  sorry

end sum_three_consecutive_integers_divisible_by_three_l7_7935


namespace possible_number_of_classmates_l7_7129

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l7_7129


namespace series_satisfies_l7_7362

noncomputable def series (x : ℝ) : ℝ :=
  let S₁ := 1 / (1 + x^2)
  let S₂ := x / (1 + x^2)
  (S₁ - S₂)

theorem series_satisfies (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  x = series x ↔ x^3 + 2 * x - 1 = 0 :=
by 
  -- Proof outline:
  -- 1. Calculate the series S as a function of x
  -- 2. Equate series x to x and simplify to derive the polynomial equation
  sorry

end series_satisfies_l7_7362


namespace cake_eating_classmates_l7_7114

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l7_7114


namespace solve_eq_sqrt_l7_7356

noncomputable def solution (p : ℝ) := (4 - p) / Real.sqrt (8 * (2 - p))

theorem solve_eq_sqrt (p x : ℝ) (h₀ : p ≥ 0) (h₁ : p ≤ 4/3) : 
  (Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x) ↔ 
  x = solution p :=
begin
  sorry
end

end solve_eq_sqrt_l7_7356


namespace range_of_a_l7_7801

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - 6*x + 8 > 0

theorem range_of_a (h : (∀ x a, p x a → q x) ∧ (∃ x a, q x ∧ ¬ p x a)) :
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3) :=
sorry

end range_of_a_l7_7801


namespace log2_30_integers_sum_l7_7575

-- Define the constants and log properties
noncomputable def log_base_2 : ℝ → ℝ := λ x, Real.log x / Real.log 2

-- Statement of the problem
theorem log2_30_integers_sum : ∃ (a b : ℤ), a = 4 ∧ b = 5 ∧ (a : ℝ) < log_base_2 30 ∧ log_base_2 30 < (b : ℝ) → a + b = 9 := 
by
  -- Initialize with the given statement
  use [4, 5]
  sorry

end log2_30_integers_sum_l7_7575


namespace probability_of_moving_to_Q_after_n_seconds_l7_7328

noncomputable def probability_after_n_seconds (n : ℕ) : ℝ :=
  let M : Matrix (Fin 3) (Fin 3) ℝ :=
    ![![2/3, 1/6, 1/6], ![1/6, 2/3, 1/6], ![1/6, 1/6, 2/3]]
  let v0 : Fin 3 → ℝ := fun i => if i = 0 then 1 else 0
  let v_n := M^n • v0
  v_n 1

theorem probability_of_moving_to_Q_after_n_seconds (n : ℕ) : probability_after_n_seconds n = sorry := sorry

end probability_of_moving_to_Q_after_n_seconds_l7_7328


namespace number_of_classmates_l7_7116

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l7_7116


namespace triangle_ABC_A_l7_7485

noncomputable def is_isosceles (A B C : Type) (AB AC : ℝ) :=
AB = AC

noncomputable def bisects_angle (X Y Z : Type) (P : Type) (XY XZ : ℝ) :=
XY + XZ = 180

noncomputable def measure_angle_A (A B C D E : Type) (AB AC BD BC CE : ℝ) (ABC ACB BDC BCD CEB EBC : ℝ) : Prop :=
  is_isosceles A B C AB AC →
  bisects_angle B D C A ABC →
  bisects_angle C E B A ACB →
  BD = BC →
  CE = CB →
  A = 36

theorem triangle_ABC_A (A B C D E : Type) (AB AC BD BC CE : ℝ) (ABC ACB BDC BCD CEB EBC : ℝ) :
  measure_angle_A A B C D E AB AC BD BC CE ABC ACB BDC BCD CEB EBC :=
begin
  -- The proof is omitted here.
  sorry
end

end triangle_ABC_A_l7_7485


namespace quadratic_one_solution_l7_7422

theorem quadratic_one_solution (b d : ℝ) (h1 : b + d = 35) (h2 : b < d) (h3 : (24 : ℝ)^2 - 4 * b * d = 0) :
  (b, d) = (35 - Real.sqrt 649 / 2, 35 + Real.sqrt 649 / 2) := 
sorry

end quadratic_one_solution_l7_7422


namespace repeating_six_to_fraction_l7_7756

-- Define the infinite geometric series representation of 0.666...
def infinite_geometric_series (n : ℕ) : ℝ := 6 / (10 ^ n)

-- Define the sum of the infinite geometric series for 0.666...
def sum_infinite_geometric_series : ℝ :=
  ∑' n, infinite_geometric_series n

-- Formally state the problem to prove that 0.666... equals 2/3
theorem repeating_six_to_fraction : sum_infinite_geometric_series = 2 / 3 :=
by
  -- Proof goes here, but for now we use sorry to denote it will be completed later
  sorry

end repeating_six_to_fraction_l7_7756


namespace fractional_exponent_representation_l7_7766

-- Define the expressions and the relationship to be proved
theorem fractional_exponent_representation (a : ℝ) (h : a ≥ 0) : 
  ∃ (r : ℝ), r = (sqrt (a * (cbrt (a * (sqrt a))))) ∧ r = a^(3/4) :=
by
  sorry

end fractional_exponent_representation_l7_7766


namespace range_of_f_l7_7361

noncomputable def f (x : ℝ) : ℝ :=
  if x = -5 then 0 else (3 * (x + 5) * (x - 4)) / (x + 5)

theorem range_of_f :
  ∃ R : Set ℝ, R = Set.Ioo (⊥ : ℝ) (-27) ∪ Set.Ioo (-27) (⊤ : ℝ) ∧
  ∀ y, y ∈ R ↔ ∃ x : ℝ, x ≠ -5 ∧ f x = y :=
by
  sorry

end range_of_f_l7_7361


namespace probability_of_interval_l7_7867

noncomputable def probability_less_than (a b : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < b then (x - a) / (b - a) else 0

theorem probability_of_interval (a b x : ℝ) (ha : 0 < a) (hb : a < b) :
  probability_less_than 0 (1/2) (1/3) = 2/3 :=
by
  have h_interval : (1 : ℝ)/2 - 0 = (1/2) := by norm_num,
  have h_favorable : (1 : ℝ)/3 - 0 = (1/3) := by norm_num,
  rw [← h_interval, ← h_favorable, probability_less_than],
  split_ifs,
  simpa using div_eq_mul_one_div (1/3) (1/2),
  sorry

end probability_of_interval_l7_7867


namespace find_lambda_l7_7267

def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (4, -2)
def vec_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : vec_perpendicular (λ • a + b) a) : λ = -1 :=
by
  sorry

end find_lambda_l7_7267


namespace estimate_red_balls_l7_7007

theorem estimate_red_balls (x : ℕ) (drawn_black_balls : ℕ) (total_draws : ℕ) (black_balls : ℕ) 
  (h1 : black_balls = 4) 
  (h2 : total_draws = 100) 
  (h3 : drawn_black_balls = 40) 
  (h4 : (black_balls : ℚ) / (black_balls + x) = drawn_black_balls / total_draws) : 
  x = 6 := 
sorry

end estimate_red_balls_l7_7007


namespace gift_card_remaining_l7_7068

theorem gift_card_remaining (initial_amount : ℕ) (half_monday : ℕ) (quarter_tuesday : ℕ) : 
  initial_amount = 200 → 
  half_monday = initial_amount / 2 →
  quarter_tuesday = (initial_amount - half_monday) / 4 →
  initial_amount - half_monday - quarter_tuesday = 75 :=
by
  intros h_init h_half h_quarter
  rw [h_init, h_half, h_quarter]
  sorry

end gift_card_remaining_l7_7068


namespace number_of_solutions_l7_7434

theorem number_of_solutions :
  {y : ℤ | 100 ≤ y ∧ y ≤ 999 ∧ (5327 * y + 673) % 17 = 1850 % 17}.toFinset.card = 53 := by
  sorry

end number_of_solutions_l7_7434


namespace cake_eating_classmates_l7_7108

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l7_7108


namespace number_of_red_balls_l7_7008

noncomputable def red_balls (n_black n_red draws black_draws : ℕ) : ℕ :=
  if black_draws = (draws * n_black) / (n_black + n_red) then n_red else sorry

theorem number_of_red_balls :
  ∀ (n_black draws black_draws : ℕ),
    n_black = 4 →
    draws = 100 →
    black_draws = 40 →
    red_balls n_black (red_balls 4 6 100 40) 100 40 = 6 :=
by
  intros n_black draws black_draws h_black h_draws h_blackdraws
  dsimp [red_balls]
  rw [h_black, h_draws, h_blackdraws]
  norm_num
  sorry

end number_of_red_balls_l7_7008


namespace combined_time_alligators_walked_l7_7028

-- Define the conditions
def original_time : ℕ := 4
def return_time := original_time + 2 * Int.sqrt original_time

-- State the theorem to be proven
theorem combined_time_alligators_walked : original_time + return_time = 12 := by
  sorry

end combined_time_alligators_walked_l7_7028


namespace expected_digits_fair_icosahedral_die_l7_7715

noncomputable def expected_number_of_digits : ℝ :=
  let one_digit_count := 9
  let two_digit_count := 11
  let total_faces := 20
  let prob_one_digit := one_digit_count / total_faces
  let prob_two_digit := two_digit_count / total_faces
  (prob_one_digit * 1) + (prob_two_digit * 2)

theorem expected_digits_fair_icosahedral_die :
  expected_number_of_digits = 1.55 :=
by
  sorry

end expected_digits_fair_icosahedral_die_l7_7715


namespace range_on_domain01_range_on_domain14over_l7_7402

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem range_on_domain01 : (∀ y ∈ set.range (λ x : set.Icc (0 : ℝ) 1, f x), 1 ≤ y ∧ y ≤ 2) ∧
                            (∀ y : ℝ, (1 ≤ y ∧ y ≤ 2) → ∃ x ∈ set.Icc (0 : ℝ) 1, y = f x) :=
sorry

theorem range_on_domain14over : (∀ y ∈ set.range (λ x : set.Icc (1/4 : ℝ) 2, f x), 1 ≤ y ∧ y ≤ 2) ∧
                                (∀ y : ℝ, (1 ≤ y ∧ y ≤ 2) → ∃ x ∈ set.Icc (1/4 : ℝ) 2, y = f x) :=
sorry

end range_on_domain01_range_on_domain14over_l7_7402


namespace simplest_quadratic_radical_l7_7620

def is_simplest_radical (r : ℝ) : Prop :=
  ∀ x : ℝ, (∃ a b : ℝ, r = a * Real.sqrt b) → (∃ c d : ℝ, x = c * Real.sqrt d) → a ≤ c

def sqrt_12 := Real.sqrt 12
def sqrt_2_3 := Real.sqrt (2 / 3)
def sqrt_0_3 := Real.sqrt 0.3
def sqrt_7 := Real.sqrt 7

theorem simplest_quadratic_radical :
  is_simplest_radical sqrt_7 ∧
  ¬ is_simplest_radical sqrt_12 ∧
  ¬ is_simplest_radical sqrt_2_3 ∧
  ¬ is_simplest_radical sqrt_0_3 :=
by
  sorry

end simplest_quadratic_radical_l7_7620


namespace max_valid_n_division_l7_7341

theorem max_valid_n_division (n : ℕ) (h1 : n > 4)
  (h2 : ∀ A B : set ℕ, (∀ x ∈ A, ∀ y ∈ A, (x ≠ y → ¬ (∃ m : ℕ, m^2 = x + y))) ∧ 
                       (∀ x ∈ B, ∀ y ∈ B, (x ≠ y → ¬ (∃ m : ℕ, m^2 = x + y)))) : n ≤ 28 :=
sorry

end max_valid_n_division_l7_7341


namespace maximum_triangle_perimeter_is_5_85_l7_7768

-- Definitions based on conditions
def regular_septagon_vertices (n : ℕ) : set (ℝ × ℝ) :=
  { (cos (2 * pi * k / 7), sin (2 * pi * k / 7)) | k : ℤ } -- Vertices of regular septagon with unit edge.

def vertex_distance (v1 v2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2)

def triangle_perimeter (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  vertex_distance v1 v2 + vertex_distance v2 v3 + vertex_distance v3 v1

-- Define the maximum perimeter problem
def max_triangle_perimeter_in_septagon : ℝ :=
  Sup {p | ∃ v1 v2 v3 ∈ regular_septagon_vertices 7, p = triangle_perimeter v1 v2 v3}

theorem maximum_triangle_perimeter_is_5_85 :
  abs (max_triangle_perimeter_in_septagon - 5.85) < 0.01 :=
sorry

end maximum_triangle_perimeter_is_5_85_l7_7768


namespace possible_classmates_l7_7131

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l7_7131


namespace function_decreasing_interval_l7_7995

def f (x : ℝ) : ℝ := sin x * cos x + sqrt 3 * (sin x)^2

theorem function_decreasing_interval (k : ℤ) :
  (∀ x y : ℝ, 
    (frac5pi12 + k * pi ≤ x ∧ x ≤ frac11pi12 + k * pi) →
    (frac5pi12 + k * pi ≤ y ∧ y ≤ frac11pi12 + k * pi) →
    x < y →
    f y < f x) :=
sorry

end function_decreasing_interval_l7_7995


namespace find_value_of_a_l7_7849

theorem find_value_of_a (a : ℚ) : (∃ (b : ℚ), (9 * (x : ℚ) ^ 2 + 27 * x + a) = (3 * x + b) ^ 2) → a = 81 / 4 :=
by
  intros h
  rcases h with ⟨b, hb⟩
  have hx := congr_arg (λ p, p.coeff 1) hb
  linarith
  sorry

end find_value_of_a_l7_7849


namespace point_outside_circle_of_radius_third_l7_7027

-- Define the conditions
variables (O A : E) [normed_group E] [inner_product_space ℝ E]
-- Define the circle with center O and radius R
def circle (O : E) (R : ℝ) : set E := { x | dist O x = R }

-- Define our theorem, which indicates the location of point A
theorem point_outside_circle_of_radius_third (R : ℝ) (hR : R > 0) (h : dist O A > R / 3) :
  ∃ ⦃B C : E⦄, B ≠ O ∧ C ≠ O ∧ (∃ ⦃D : E⦄, D ≠ O ∧ ∃ n : ℕ, n = 3 ∧
    (ith_reflection_point O B C D A n = A)) :=
sorry

-- Provide a definition for the nth reflection point (this is an abstraction)
def ith_reflection_point (O B C D : E) (A : E) (n : ℕ) : E := 
  sorry -- Placeholder for the actual reflection logic

end point_outside_circle_of_radius_third_l7_7027


namespace simplest_quadratic_radical_l7_7618

theorem simplest_quadratic_radical :
  ∀ (a b c d : ℝ),
    a = Real.sqrt 12 →
    b = Real.sqrt (2 / 3) →
    c = Real.sqrt 0.3 →
    d = Real.sqrt 7 →
    d = Real.sqrt 7 :=
by
  intros a b c d ha hb hc hd
  rw [hd]
  sorry

end simplest_quadratic_radical_l7_7618


namespace angle_pba_eq_angle_dbh_l7_7902

variables {A B C D H P : Point} (h1 : IsSquare A B C D)
  (h2 : ∠B A C = 90º) (h3 : IsAltitude A H D C)
  (h4 : OnLine AC P) (h5 : Tangent PD (circumcircle (triangle A B D)))

theorem angle_pba_eq_angle_dbh
  (h6 : ∠P B A = ∠D B H) : ∠ P B A = ∠ D B H :=
sorry

end angle_pba_eq_angle_dbh_l7_7902


namespace new_avg_weight_l7_7553

theorem new_avg_weight (avg_weight_29 : ℕ) (new_student_weight : ℕ) (num_students : ℕ) (new_students : ℕ) (initial_avg_weight : ℕ) (initial_avg_correct : avg_weight_29 = initial_avg_weight * num_students / 29) (new_students_correct : new_students = num_students + 1) (calc_weight_correct : (initial_avg_weight * 29 + new_student_weight) / new_students = 27.4) : new_avg_weight == 27.4 :=
by
  -- Provided conditions are used, hence it's guaranteed in the theorem statement
  sorry

end new_avg_weight_l7_7553


namespace problem_l7_7851

noncomputable def g (x : ℝ) : ℝ := 3^x + 2

theorem problem (x : ℝ) : g (x + 1) - g x = 2 * g x - 2 := sorry

end problem_l7_7851


namespace S_not_singleton_S_two_elements_S_total_number_l7_7931

noncomputable def S (S : Set ℕ) : Prop :=
  ∀ x ∈ S, 1 ≠ x ∧ (1 + 12 / (x - 1 : ℝ)).nat_floor ∈ S

theorem S_not_singleton :
  ∀ (S : Set ℕ), (S ⊆ Set.Icc 2 ∞) → S ≠ ∅ → S (S) → (¬ ∃ m, S = {m}) :=
sorry

theorem S_two_elements :
  ∀ (S : Set ℕ), (S ⊆ Set.Icc 2 ∞) → S ≠ ∅ → S (S) → (S = {2, 13} ∨
                                                    S = {3, 7} ∨
                                                    S = {4, 5}) :=
sorry

theorem S_total_number :
  ∃ (Ss : List (Set ℕ)), (∀ S ∈ Ss, (S ⊆ Set.Icc 2 ∞) ∧ S ≠ ∅ ∧ S (S)) ∧
                        Ss.length = 7 :=
sorry

end S_not_singleton_S_two_elements_S_total_number_l7_7931


namespace compare_abc_l7_7805

noncomputable def a := Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
noncomputable def b := Real.cos (Real.pi / 6) ^ 2 - Real.sin (Real.pi / 6) ^ 2
noncomputable def c := Real.tan (30 * Real.pi / 180) / (1 - Real.tan (30 * Real.pi / 180) ^ 2)

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l7_7805


namespace repeating_six_to_fraction_l7_7759

-- Define the infinite geometric series representation of 0.666...
def infinite_geometric_series (n : ℕ) : ℝ := 6 / (10 ^ n)

-- Define the sum of the infinite geometric series for 0.666...
def sum_infinite_geometric_series : ℝ :=
  ∑' n, infinite_geometric_series n

-- Formally state the problem to prove that 0.666... equals 2/3
theorem repeating_six_to_fraction : sum_infinite_geometric_series = 2 / 3 :=
by
  -- Proof goes here, but for now we use sorry to denote it will be completed later
  sorry

end repeating_six_to_fraction_l7_7759


namespace triangle_side_squares_sum_l7_7038

noncomputable def centroid (x y z : ℝ^3) : ℝ^3 := (x + y + z) / 3

noncomputable def distance_squared (a b : ℝ^3) : ℝ := (a - b).norm^2

theorem triangle_side_squares_sum (x y z : ℝ^3) (h1 : distance_squared (centroid x y z) x + distance_squared (centroid x y z) y + distance_squared (centroid x y z) z = 72) :
  distance_squared x y + distance_squared x z + distance_squared y z = 216 := 
sorry

end triangle_side_squares_sum_l7_7038


namespace average_cost_per_individual_before_gratuity_l7_7664

theorem average_cost_per_individual_before_gratuity (X : ℝ) :
  let Y := X / 1.25 in
  let individual_cost_before_gratuity := Y / 10 in
  individual_cost_before_gratuity = X / 12.5 :=
by
  let Y := X / 1.25
  let individual_cost_before_gratuity := Y / 10
  show individual_cost_before_gratuity = X / 12.5
  sorry

end average_cost_per_individual_before_gratuity_l7_7664


namespace hexagon_area_evaluation_l7_7046

theorem hexagon_area_evaluation
  (hexagon : Type)
  [regular_hexagon hexagon]
  (A B C D E F M X Y Z : hexagon.point)
  (area : hexagon.polygon → ℝ)
  (H1 : area ⟨[A, B, C, D, E, F], hexagon.regular⟩ = 1)
  (H2: M = midpoint D E)
  (H3: X = intersection (line A C) (line B M))
  (H4: Y = intersection (line B F) (line A M))
  (H5: Z = intersection (line A C) (line B F)) :
  area ⟨[B, X, C], triangle BXC⟩ + area ⟨[A, Y, F], triangle AYF⟩ + 
  area ⟨[A, B, Z], triangle ABZ⟩ - area ⟨[M, X, Z, Y], quadrilateral MXZY⟩ = 1 :=
by 
  sorry

end hexagon_area_evaluation_l7_7046


namespace angle_CPC_tangent_when_C_and_D_coincide_l7_7590

-- Let A, B be points where two circles intersect.
-- Let C, D be points on the first circle intersected by a secant through A.
-- Let C', D' be points on the second circle intersected by a secant through A.
-- Let P be the intersection point of lines (CD) and (C'D').

noncomputable def angle_between_lines {α : Type*} [linear_ordered_field α] (a b c d : euclidean_space α (fin 2)) : α :=
180 - real.arcsin ((d - c) x (a - b) / ((d - c).norm * (a - b).norm))

theorem angle_CPC'_equals_180_minus_theta {α : Type*} [linear_ordered_field α]
  (A B C D C' D' P : euclidean_space α (fin 2))
  (H : circles_intersect_at A B C D C' D') :
  angle_between_lines C P C' = 180 - angle_between_circles A B :=
sorry

theorem tangent_when_C_and_D_coincide {α : Type*} [linear_ordered_field α]
  (A B C C' D' P : euclidean_space α (fin 2))
  (H : circles_intersect_at A B C C' D') :
  lines_tangent_at P C C' :=
sorry

end angle_CPC_tangent_when_C_and_D_coincide_l7_7590


namespace three_integers_product_sum_l7_7176

theorem three_integers_product_sum (a b c : ℤ) (h : a * b * c = -5) :
    a + b + c = 5 ∨ a + b + c = -3 ∨ a + b + c = -7 :=
sorry

end three_integers_product_sum_l7_7176


namespace area_of_right_triangle_l7_7884

theorem area_of_right_triangle (a b c : ℝ) 
  (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 30 :=
by sorry

end area_of_right_triangle_l7_7884


namespace evaluate_negativity_l7_7169

theorem evaluate_negativity :
  let e1 := -(-5)
  let e2 := -abs(-5)
  let e3 := -(5^2)
  let e4 := (-5)^2
  let e5 := 1 / (-5)
  (if e1 < 0 then 1 else 0) +
  (if e2 < 0 then 1 else 0) +
  (if e3 < 0 then 1 else 0) +
  (if e4 < 0 then 1 else 0) +
  (if e5 < 0 then 1 else 0) = 3 :=
by
  let e1 := -(-5)
  let e2 := -abs(-5)
  let e3 := -(5^2)
  let e4 := (-5)^2
  let e5 := 1 / (-5)
  sorry

end evaluate_negativity_l7_7169


namespace proof_smallest_lcm_1_to_12_l7_7227

noncomputable def smallest_lcm_1_to_12 : ℕ := 27720

theorem proof_smallest_lcm_1_to_12 :
  ∀ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ↔ n = 27720 :=
by
  sorry

end proof_smallest_lcm_1_to_12_l7_7227


namespace line_equation_sum_l7_7281

theorem line_equation_sum (m b x y : ℝ) (hx : x = 4) (hy : y = 2) (hm : m = -5) (hline : y = m * x + b) : m + b = 17 := by
  sorry

end line_equation_sum_l7_7281


namespace bridge_extension_length_l7_7556

theorem bridge_extension_length (river_width bridge_length : ℕ) (h_river : river_width = 487) (h_bridge : bridge_length = 295) : river_width - bridge_length = 192 :=
by
  sorry

end bridge_extension_length_l7_7556


namespace equidistant_points_are_on_line_l7_7606

-- Define the types and conditions
structure Plane : Type where
  normal : ℝ³ -- Representation of plane using normal vector
  point  : ℝ³ -- A point on the plane

structure TrihedralAngle : Type where
  plane1 : Plane
  plane2 : Plane
  plane3 : Plane
  vertex : ℝ³ -- The common point where all three planes intersect

-- Define what it means to be equidistant from three planes
def equidistantFromThreePlanes (p : ℝ³) (ta : TrihedralAngle) : Prop :=
  dist_to_plane p ta.plane1 = dist_to_plane p ta.plane2 ∧
  dist_to_plane p ta.plane2 = dist_to_plane p ta.plane3

-- Define the line where the points equidistant from the three planes lie
def intersectionLine (ta : TrihedralAngle) : set ℝ³ :=
  { p : ℝ³ | ∃ t : ℝ, p = ta.vertex + t • (unit_vector_of_bisector ta.plane1 ta.plane2 ∩
                                           unit_vector_of_bisector ta.plane2 ta.plane3) }

-- Now, write the theorem statement
theorem equidistant_points_are_on_line (ta : TrihedralAngle) :
  ∀ p : ℝ³, (equidistantFromThreePlanes p ta ∧ insideTrihedralAngle p ta) ↔ p ∈ intersectionLine ta :=
by
  sorry

end equidistant_points_are_on_line_l7_7606


namespace union_area_of_rotated_triangle_l7_7025

noncomputable def heron (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  real.sqrt (s * (s - a) * (s - b) * (s - c))

def centroid_triangle (a b c : ℝ) : ℝ :=
  (2 / 3) * heron a b c

theorem union_area_of_rotated_triangle :
  (AB BC AC G : ℝ) (hAB : AB = 13) (hBC : BC = 14) (hAC : AC = 15)
  (hG_centroid : G = centroid_triangle) :
  let triangle_area := heron AB BC AC in
  let small_triangle_area := (1 / 9) * triangle_area in
  2 * triangle_area - 3 * small_triangle_area = 140 :=
by
  sorry

end union_area_of_rotated_triangle_l7_7025


namespace regression_analysis_residuals_l7_7015

variables (y : Type*) [metric_space y] (y_hat : Type*) [metric_space y_hat]

/-- Total Sum of Squares. -/
def SST : ℝ := sorry

/-- Sum of Squares of Residuals. -/
def SSR : ℝ := sorry

/-- Regression Sum of Squares. -/
def SSM : ℝ := sorry

/-- Correlation Coefficient. -/
def r : ℝ := sorry

theorem regression_analysis_residuals : 
  (∃ (SSR : ℝ), ∀ (y y_hat : Type*), SSR = ∑ (i : ℕ), (y i - y_hat i) ^ 2) :=
sorry

end regression_analysis_residuals_l7_7015


namespace pipe_filling_cistern_l7_7197

theorem pipe_filling_cistern (t : ℝ) (ht_pos : 0 < t) :
  let rate_q := 1 / 15 in
  let fill_in_2_minutes := 2 * (1 / t) + 2 * rate_q in
  let fill_in_10_5_minutes := 10.5 * rate_q in
  fill_in_2_minutes + fill_in_10_5_minutes = 1 → t = 12 :=
by
  intro rate_q fill_in_2_minutes fill_in_10_5_minutes h_fill_sum
  sorry

end pipe_filling_cistern_l7_7197


namespace determine_a_range_f_l7_7061

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - (2 / (2 ^ x + 1))

theorem determine_a (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) -> a = 1 :=
by
  sorry

theorem range_f (x : ℝ) : (∀ x : ℝ, f 1 (-x) = -f 1 x) -> -1 < f 1 x ∧ f 1 x < 1 :=
by
  sorry

end determine_a_range_f_l7_7061


namespace garrison_deployments_l7_7581

variable (x1 x2 x3 x4 x5 x6 x7 x8 : ℕ)

theorem garrison_deployments :
  x1 = 20 ∧ x2 = 30 ∧ x3 = 50 ∧ x4 = 40 ∧ x5 = 10 ∧ x6 = 40 ∧ x7 = 30 ∧ x8 = 70 →
  x1 + x2 + x3 = 100 ∧
  x4 + x5 + x3 = 100 ∧
  x4 + x1 + x6 = 100 ∧
  x2 + x4 + x7 = 100 ∧
  x5 + x1 + x8 = 100 :=
by
  intros h,
  cases h,
  sorry

end garrison_deployments_l7_7581


namespace correspond_half_l7_7345

theorem correspond_half (m n : ℕ) 
  (H : ∀ h : Fin m, ∃ g_set : Finset (Fin n), (g_set.card = n / 2) ∧ (∀ g : Fin n, g ∈ g_set))
  (G : ∀ g : Fin n, ∃ h_set : Finset (Fin m), (h_set.card ≤ m / 2) ∧ (∀ h : Fin m, h ∈ h_set)) :
  (∀ h : Fin m, ∀ g_set : Finset (Fin n), g_set.card = n / 2) ∧ (∀ g : Fin n, ∀ h_set : Finset (Fin m), h_set.card = m / 2) :=
by
  sorry

end correspond_half_l7_7345


namespace incorrect_statements_about_plane_vectors_l7_7257

theorem incorrect_statements_about_plane_vectors 
  (a b c : Vector)
  (G A B C D : Point)
  (λ : ℝ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (parallel : a.parallel b ↔ ∃ λ, b = λ • a)
  (collinear : collinear AB CD → ¬∃ l : Line, (A ∈ l) ∧ (B ∈ l) ∧ (C ∈ l) ∧ (D ∈ l))
  (dot_product : a.dot c = b.dot c → a = b)
  (nonzero_c : c ≠ 0)
  (centroid : G.is_centroid_of A B C → GA + GB + GC = 0) :
  ¬ (collinear AB CD → A, B, C, D collinear) ∧ ¬ (a.dot c = b.dot c → a = b) :=
sorry

end incorrect_statements_about_plane_vectors_l7_7257


namespace M_in_fourth_quadrant_l7_7872

-- Define the conditions
variables (a b : ℝ)

/-- Condition that point A(a, 3) and B(2, b) are symmetric with respect to the x-axis -/
def symmetric_points : Prop :=
  a = 2 ∧ 3 = -b

-- Define the point M and quadrant check
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a > 0 ∧ b < 0

-- The theorem stating that if A(a, 3) and B(2, b) are symmetric wrt x-axis, M is in the fourth quadrant
theorem M_in_fourth_quadrant (a b : ℝ) (h : symmetric_points a b) : in_fourth_quadrant a b :=
by {
  sorry
}

end M_in_fourth_quadrant_l7_7872


namespace equilateral_triangle_l7_7461

variables {A B C a b c : ℝ}

/-- Given conditions:
 1. (a + b + c) * (a + b - c) = 3 * a * b
 2. 2 * cos A * sin B = sin C
Determine the shape of ΔABC.
-/
theorem equilateral_triangle 
  (h1 : (a + b + c) * (a + b - c) = 3 * a * b)
  (h2 : 2 * cos A * sin B = sin C) : 
  A = B ∧ B = C ∧ C = A := 
sorry

end equilateral_triangle_l7_7461


namespace total_assembly_time_l7_7066

-- Define the conditions
def chairs : ℕ := 2
def tables : ℕ := 2
def time_per_piece : ℕ := 8
def total_pieces : ℕ := chairs + tables

-- State the theorem
theorem total_assembly_time :
  total_pieces * time_per_piece = 32 :=
sorry

end total_assembly_time_l7_7066


namespace fish_left_in_tank_l7_7516

theorem fish_left_in_tank (initial_fish : ℕ) (fish_taken_out : ℕ) (fish_left : ℕ) 
  (h1 : initial_fish = 19) (h2 : fish_taken_out = 16) : fish_left = initial_fish - fish_taken_out :=
by
  simp [h1, h2]
  sorry

end fish_left_in_tank_l7_7516


namespace midpoint_locus_and_integer_points_l7_7635

theorem midpoint_locus_and_integer_points 
  (p : ℕ)
  (hp_prime : Nat.Prime p)
  (hp_ne_two : p ≠ 2)
  {x y : ℝ} :
  (∃ k : ℝ, y^2 = 2 * p * x ∧ k ≠ 0) →
  (∃ R : ℝ × ℝ, (4 * (R.snd)^2 = p * (R.fst - p) ∧ R.snd ≠ 0)
      ∧ (∀ t : ℤ, ∃ (x y : ℤ), y ≠ 0 ∧ 4 * y^2 = p * (x - p))
      ∧ ¬∃ (x y : ℤ), x^2 + y^2 = m^2 where m : ℕ) :=
begin
  sorry
end

end midpoint_locus_and_integer_points_l7_7635


namespace convex_polyhedron_with_acute_dihedral_angles_has_four_faces_l7_7651

theorem convex_polyhedron_with_acute_dihedral_angles_has_four_faces
  (P : Polyhedron) (h_convex : convex P)
  (h_acute : ∀ e ∈ edges P, acute_dihedral_angle_at e) :
  faces P = 4 := 
sorry

end convex_polyhedron_with_acute_dihedral_angles_has_four_faces_l7_7651


namespace classmates_ate_cake_l7_7087

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l7_7087


namespace length_of_segment_l7_7203

theorem length_of_segment (x : ℝ) : 
  |x - (27^(1/3))| = 5 →
  ∃ a b : ℝ, a - b = 10 ∧ (|a - (27^(1/3))| = 5 ∧ |b - (27^(1/3))| = 5) :=
by
  sorry

end length_of_segment_l7_7203


namespace overlapping_area_ratio_l7_7369

theorem overlapping_area_ratio (s : ℝ) (A B E F G H I K X : ℝ) (overlap_factor : ℝ) 
  (sq1 sq2 sq3 sq4 : set (ℝ × ℝ)) 
  (H1 : ∀ {p : ℝ × ℝ}, p ∈ sq1 → p ∈ sq2 → dist p (0,0) = overlap_factor * s)
  (H2 : ∀ {p : ℝ × ℝ}, p ∈ sq2 → p ∈ sq3 → dist p (s,0) = overlap_factor * s)
  (H3 : ∀ {p : ℝ × ℝ}, p ∈ sq3 → p ∈ sq4 → dist p (2 * s,0) = overlap_factor * s)
  (H4 : s = 1)
  (H5 : overlap_factor = 0.25) :
  let overlapping_area := 3 * (overlap_factor * s)^2
  let total_area := 4 * s^2 in
  overlapping_area / total_area = 1 / 4 :=
by
  sorry

end overlapping_area_ratio_l7_7369


namespace even_square_minus_self_l7_7532

theorem even_square_minus_self (a : ℤ) : 2 ∣ (a^2 - a) :=
sorry

end even_square_minus_self_l7_7532


namespace stacy_has_2_more_than_triple_steve_l7_7156

-- Definitions based on the given conditions
def skylar_berries : ℕ := 20
def steve_berries : ℕ := skylar_berries / 2
def stacy_berries : ℕ := 32

-- Statement to be proved
theorem stacy_has_2_more_than_triple_steve :
  stacy_berries = 3 * steve_berries + 2 := by
  sorry

end stacy_has_2_more_than_triple_steve_l7_7156


namespace repeating_decimal_to_fraction_l7_7751

theorem repeating_decimal_to_fraction : (let a := (6 : Real) / 10 in
                                         let r := (1 : Real) / 10 in
                                         ∑' n : ℕ, a * r^n) = (2 : Real) / 3 :=
by
  sorry

end repeating_decimal_to_fraction_l7_7751


namespace increasing_function_range_l7_7416

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 4*a*x else (2*a + 3)*x - 4*a + 5

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1/2 ≤ a ∧ a ≤ 3/2) :=
sorry

end increasing_function_range_l7_7416


namespace sin_theta_value_l7_7438

theorem sin_theta_value (θ : ℝ) (h₁ : 8 * (Real.tan θ) = 3 * (Real.cos θ)) (h₂ : 0 < θ ∧ θ < Real.pi) : 
  Real.sin θ = 1 / 3 := 
by sorry

end sin_theta_value_l7_7438


namespace paint_required_for_statues_l7_7852

theorem paint_required_for_statues :
  (let height_ratio := (2 / 8 : ℝ), 
       surface_area_ratio := (height_ratio * height_ratio), 
       paint_per_small_statue := (1 / 16 * 2), 
       total_paint := 360 * paint_per_small_statue
  in total_paint = 45) :=
by
  let height_ratio := (2 / 8 : ℝ)
  let surface_area_ratio := height_ratio * height_ratio
  let paint_per_small_statue := 1 / 16 * 2
  let total_paint := 360 * paint_per_small_statue
  have : total_paint = 45 := by sorry
  exact this

end paint_required_for_statues_l7_7852


namespace f_g_of_3_l7_7498

def f (x : ℤ) : ℤ := 2 * x + 3
def g (x : ℤ) : ℤ := x^3 - 6

theorem f_g_of_3 : f (g 3) = 45 := by
  sorry

end f_g_of_3_l7_7498


namespace number_of_classmates_ate_cake_l7_7146

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l7_7146


namespace solve_for_x_l7_7643

variable (diamondsuit : ℝ → ℝ → ℝ)
variable (h1 : ∀ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 → a ⋄ (b ⋄ c) = (a ⋄ b) * c)
variable (h2 : ∀ a : ℝ, a ≠ 0 → a ⋄ a = 1)

theorem solve_for_x : (504 ⋄ (3 ⋄ x) = 50) → x = 25 / 84 := by
  sorry

end solve_for_x_l7_7643


namespace simplest_quadratic_radical_l7_7614

theorem simplest_quadratic_radical :
  let a := Real.sqrt 12
  let b := Real.sqrt (2 / 3)
  let c := Real.sqrt 0.3
  let d := Real.sqrt 7
  d < a ∧ d < b ∧ d < c := 
by {
  -- the proof steps will go here, but we use sorry for now
  sorry
}

end simplest_quadratic_radical_l7_7614


namespace trapezoid_area_l7_7484

-- Define the given conditions in the problem
variables (EF GH h EG FH : ℝ)
variables (EF_parallel_GH : true) -- EF and GH are parallel (not used in the calculation)
variables (EF_eq_70 : EF = 70)
variables (GH_eq_40 : GH = 40)
variables (h_eq_15 : h = 15)
variables (EG_eq_20 : EG = 20)
variables (FH_eq_25 : FH = 25)

-- Define the main theorem to prove
theorem trapezoid_area (EF GH h EG FH : ℝ) 
  (EF_eq_70 : EF = 70) 
  (GH_eq_40 : GH = 40) 
  (h_eq_15 : h = 15) 
  (EG_eq_20 : EG = 20) 
  (FH_eq_25 : FH = 25) : 
  0.5 * (EF + GH) * h = 825 := 
by 
  sorry

end trapezoid_area_l7_7484


namespace intersection_eq_l7_7832

def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6 * x + 8 < 0}

theorem intersection_eq : M ∩ N = {x | 2 < x ∧ x < 3} := 
by
  sorry

end intersection_eq_l7_7832


namespace area_OEF_l7_7019

variables {Point : Type} [affine_plane Point]

variables (A B C D E F O : Point)
variables {triangle_area : Point → Point → Point → ℝ}

-- Assume collinearity of F, C, D
axiom collinear_FCD : collinear F C D

-- Areas of triangles
axiom area_AOD : triangle_area A O D = S1
axiom area_DOC : triangle_area D O C = S2
axiom area_AOB : triangle_area A O B = S3

theorem area_OEF (S1 S2 S3 : ℝ) :
  triangle_area O E F = S1 + S2 + S3 :=
sorry

end area_OEF_l7_7019


namespace transform_circle_center_l7_7317

theorem transform_circle_center :
  ∀ (S : ℝ × ℝ), S = (3, -4) →
  let S1 := (-(S.1), S.2) in
  let S2 := (S1.1, -(S1.2)) in
  let S3 := (S2.1, S2.2 + 5) in
  S3 = (-3, 9) :=
by
  intros S h1,
  dsimp,
  rw [h1],
  dsimp,
  simp,
  sorry

end transform_circle_center_l7_7317


namespace find_n_has_no_constant_term_l7_7449

-- Let's define the problem in Lean
variables {x : ℝ} {n : ℕ}
noncomputable def has_no_constant_term (n : ℕ) : Prop :=
  (∀ r : ℕ, n ≠ 4 * r) ∧
  (∀ r : ℤ, n ≠ 4 * r + 1) ∧
  (∀ r : ℤ, n ≠ 4 * r + 2)

theorem find_n_has_no_constant_term :
  has_no_constant_term 9 :=
sorry

end find_n_has_no_constant_term_l7_7449


namespace evaluate_f_difference_l7_7044

def f (x : ℤ) : ℤ := x^6 + 3 * x^4 - 4 * x^3 + x^2 + 2 * x

theorem evaluate_f_difference : f 3 - f (-3) = -204 := by
  sorry

end evaluate_f_difference_l7_7044


namespace light_path_length_l7_7037

/-- Define the problem's conditions -/
structure Cube where
  side_length : ℝ
  A : ℝ × ℝ × ℝ
  EFGH : ℝ × ℝ × ℝ
  Q : ℝ × ℝ × ℝ

noncomputable def problem : Cube :=
{ side_length := 10,
  A := (0, 0, 0),
  EFGH := (10, 0, 0),
  Q := (10, 3, 4) }

theorem light_path_length (c : Cube) (H1 : c.side_length = 10)
  (H2 : c.A = (0, 0, 0)) (H3 : c.Q = (10, 3, 4)) :
  (∃ p q : ℕ, c.path_length_from_A_to_vertex c.A c.Q = p * Real.sqrt q ∧ q % (p*p) ≠ 0 ∧ p + q = 55) :=
by sorry

end light_path_length_l7_7037


namespace two_pow_n_plus_one_divisible_by_three_l7_7331

-- defining what it means to be an odd number
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- stating the main theorem in Lean
theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h_pos : 0 < n) : (2^n + 1) % 3 = 0 ↔ is_odd n :=
by sorry

end two_pow_n_plus_one_divisible_by_three_l7_7331


namespace solution_set_m_eq_0_m_range_for_x_in_2_3_l7_7419

theorem solution_set_m_eq_0 (x : ℝ) : (0 ≤ x * |x| - 2) = (√2 ≤ x) :=
by
  sorry

theorem m_range_for_x_in_2_3 (x m : ℝ) (hx : 2 ≤ x ∧ x ≤ 3) : 
  (x * |x - m| - 2 ≥ m) ↔ (m ≤ 2/3 ∨ 6 ≤ m) :=
by
  sorry

end solution_set_m_eq_0_m_range_for_x_in_2_3_l7_7419


namespace julia_investment_l7_7921

-- Define the total investment and the relationship between the investments
theorem julia_investment:
  ∀ (m : ℕ), 
  m + 6 * m = 200000 → 6 * m = 171428 := 
by
  sorry

end julia_investment_l7_7921


namespace alternating_sum_equals_l7_7728

-- Define the sequence with sign changes at perfect squares
def alternating_sum_sequence (n : ℕ) : ℤ :=
  if (∃ k : ℕ, k^2 ≤ n ∧ n < (k+1)^2) then
    if ∃ m : ℕ, n = m^2 then n else -n
  else 0 -- default case should not affect computation, refine as necessary

-- Define the problem statement
theorem alternating_sum_equals :
  (∑ n in (Finset.range 10002), alternating_sum_sequence n) = 101 ^ 3 :=
  sorry

end alternating_sum_equals_l7_7728


namespace proof_smallest_lcm_1_to_12_l7_7225

noncomputable def smallest_lcm_1_to_12 : ℕ := 27720

theorem proof_smallest_lcm_1_to_12 :
  ∀ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ↔ n = 27720 :=
by
  sorry

end proof_smallest_lcm_1_to_12_l7_7225


namespace simplest_quadratic_radical_l7_7619

def is_simplest_radical (r : ℝ) : Prop :=
  ∀ x : ℝ, (∃ a b : ℝ, r = a * Real.sqrt b) → (∃ c d : ℝ, x = c * Real.sqrt d) → a ≤ c

def sqrt_12 := Real.sqrt 12
def sqrt_2_3 := Real.sqrt (2 / 3)
def sqrt_0_3 := Real.sqrt 0.3
def sqrt_7 := Real.sqrt 7

theorem simplest_quadratic_radical :
  is_simplest_radical sqrt_7 ∧
  ¬ is_simplest_radical sqrt_12 ∧
  ¬ is_simplest_radical sqrt_2_3 ∧
  ¬ is_simplest_radical sqrt_0_3 :=
by
  sorry

end simplest_quadratic_radical_l7_7619


namespace divisor_increase_by_10_5_l7_7523

def condition_one (n t : ℕ) : Prop :=
  n * (t + 7) = t * (n + 2)

def condition_two (n t z : ℕ) : Prop :=
  n * (t + z) = t * (n + 3)

theorem divisor_increase_by_10_5 (n t : ℕ) (hz : ℕ) (nz : n ≠ 0) (tz : t ≠ 0)
  (h1 : condition_one n t) (h2 : condition_two n t hz) : hz = 21 / 2 :=
by {
  sorry
}

end divisor_increase_by_10_5_l7_7523


namespace no_intersection_of_circles_l7_7501

noncomputable def acute_triangle (A B C : Point) : Prop :=
  -- some conditions that define \Delta ABC as an acute triangle

noncomputable def circumcircle (A B C : Point) : Circle :=
  -- definition of \mathcal{C}, the circumcircle passing through A, B, and C with center O

noncomputable def euler_circle (A B C : Point) : Circle :=
  -- definition of \mathcal{E}, the Euler circle passing through mid-points of sides, and feet of the altitudes of \Delta ABC with center E

variable {A B C : Point}

theorem no_intersection_of_circles 
  (h₁ : acute_triangle A B C)
  (h₂ : circumcircle A B C)
  (h₃ : euler_circle A B C)
  (h₄ : distance (center_of (circumcircle A B C)) (center_of (euler_circle A B C)) < radius (euler_circle A B C))
  (h₅ : radius (circumcircle A B C) = 2 * radius (euler_circle A B C)) :
  ∀ X, X ∈ euler_circle A B C → ¬ (X ∈ circumcircle A B C) := 
by
  sorry

end no_intersection_of_circles_l7_7501


namespace probability_of_selected_number_probability_of_selected_number_in_given_interval_l7_7864

noncomputable def probability_in_interval (a b c : ℝ) (hab : 0 < a) (hbc : b < c) : ℝ :=
  (b - a) / (c - a)

theorem probability_of_selected_number (x : ℝ) (hx : 0 < x) (hx_le : x < 1/2) : 
  probability_in_interval 0 (1/3) (1/2) 0 lt_one_half = 2/3 := 
by
  have p := probability_in_interval 0 (1/3) (1/2) 0 (by norm_num : 1 < 2)
  norm_num at p
  exact p

-- Helper theorem to convert the original question
theorem probability_of_selected_number_in_given_interval :
  ∀ x, 0 < x ∧ x < 1/2 → x < 1/3 → probability_of_selected_number x 0 (by norm_num) = 2/3 :=
by
  intros x _ _
  exact probability_of_selected_number x 0 (by norm_num)

-- Sorry to skip the proof as requested
sorry

end probability_of_selected_number_probability_of_selected_number_in_given_interval_l7_7864


namespace students_count_inconsistent_l7_7568

-- Define the conditions
variables (total_students boys_more_than_girls : ℤ)

-- Define the main theorem: The computed number of girls is not an integer
theorem students_count_inconsistent 
  (h1 : total_students = 3688) 
  (h2 : boys_more_than_girls = 373) 
  : ¬ ∃ x : ℤ, 2 * x + boys_more_than_girls = total_students := 
by
  sorry

end students_count_inconsistent_l7_7568


namespace sin_double_angle_l7_7439

noncomputable def α : ℝ := sorry

axiom cos_condition : cos (π / 4 - α) = 3 / 5

theorem sin_double_angle : sin (2 * α) = -7 / 25 :=
by
  -- Proof omitted, focus is on correct problem statement
  sorry

end sin_double_angle_l7_7439


namespace contradiction_neznaika_l7_7059

-- Definitions based on problem conditions
variable (S T : ℝ)
variable (h1 : S ≤ 50 * T)
variable (h2 : 60 * T ≤ S)
variable (h3 : T > 0)

-- Proof that the conditions lead to a contradiction
theorem contradiction_neznaika : false :=
by
  have h4 : 60 * T ≤ 50 * T := le_trans h2 h1
  have h5 : 10 * T ≤ 0 := by linarith
  have h6 : 10 * T > 0 := by { linarith, exact h3 }
  exact lt_irrefl (10 * T) h6

end contradiction_neznaika_l7_7059


namespace probability_same_denomination_or_suit_l7_7294

-- Definitions based on conditions
def total_cards := 52
def suits := 4
def denominations := 13
def pairs_from_n (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define events
def same_denomination := 13 * (pairs_from_n suits)
def same_suit := 4 * (pairs_from_n denominations)
def total_pairs := pairs_from_n total_cards

-- Probability calculations
def P_A := same_denomination.to_rat / total_pairs.to_rat
def P_B := same_suit.to_rat / total_pairs.to_rat

-- Lean statement
theorem probability_same_denomination_or_suit :
  P_A + P_B = 5 / 17 :=
by
  -- Add proof steps here
  sorry

end probability_same_denomination_or_suit_l7_7294


namespace monotonicity_interval_l7_7415

theorem monotonicity_interval (ω : ℝ) (hω : ω > 0) 
  (h_max_period : ∀ x, 2 = (λ x, 2 * sin (2 * ω * x - π / 4)).period) :
  ∃ k : ℤ, ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 
  increasing_on (λ x, 2 * sin(2 * ω * x - π / 4)) [ - (1 / 4) + 2 * k, 3 / 4 + 2 * k ] :=
sorry

end monotonicity_interval_l7_7415


namespace workers_total_earnings_l7_7650

-- Definitions for basic worker wage and hours
def wageA := 12
def wageB := 14
def wageC := 16
def wageD := 18
def overtime_rateA := 1.5
def overtime_rateB := 1.75
def overtime_rateC := 2.0
def overtime_rateD := 2 * 1.5
def hours_per_day := 8
def total_jobs := 12
def hours_per_job := 1

-- Calculations for total hours and earnings
def total_hours := total_jobs * hours_per_job

def regular_hoursA := min 5 hours_per_day
def overtime_hoursA := max 0 (hours_per_day - 5)

def regular_hoursB := min 6 hours_per_day
def overtime_hoursB := max 0 (hours_per_day - 6)

def regular_hoursC := min 7 hours_per_day
def overtime_hoursC := max 0 (hours_per_day - 7)

def regular_hoursD := min 5 hours_per_day
def overtime_hoursD := max 0 (hours_per_day - 5)

-- Calculate total wages for one day
def daily_wageA := regular_hoursA * wageA + overtime_hoursA * wageA * overtime_rateA
def daily_wageB := regular_hoursB * wageB + overtime_hoursB * wageB * overtime_rateB
def daily_wageC := regular_hoursC * wageC + overtime_hoursC * wageC * overtime_rateC
def daily_wageD := regular_hoursD * wageD + overtime_hoursD * wageD * overtime_rateD

-- Calculate total earnings for completing all jobs
def total_earningsA := daily_wageA * total_jobs
def total_earningsB := daily_wageB * total_jobs
def total_earningsC := daily_wageC * total_jobs
def total_earningsD := daily_wageD * total_jobs

-- Theorem statement
theorem workers_total_earnings :
  total_earningsA = 1368 ∧
  total_earningsB = 1596 ∧
  total_earningsC = 1728 ∧
  total_earningsD = 2376 := by
    sorry

end workers_total_earnings_l7_7650


namespace find_a_l7_7440

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a x : α) : α := 3^x + a * 3^(-x)

theorem find_a (a : α) : 
  (∀ x : α, f a (-x) = - (f a x)) → a = -1 :=
by
  intro h
  sorry

end find_a_l7_7440


namespace classmates_ate_cake_l7_7084

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l7_7084


namespace other_store_pools_l7_7968

variable (P A : ℕ)
variable (three_times : P = 3 * A)
variable (total_pools : P + A = 800)

theorem other_store_pools (three_times : P = 3 * A) (total_pools : P + A = 800) : A = 266 := 
by
  sorry

end other_store_pools_l7_7968


namespace mixture_is_possible_l7_7266

noncomputable def uniform_mixture_possible (n : ℕ) : Prop :=
  ∀ (beakers : Fin (n + 1) → ℝ) (liquids : Fin n → ℝ),
    (∀ i : Fin (n + 1), beakers i = if i = 0 then 0 else liquids i) →
    (∃ (beakers' : Fin (n + 1) → ℝ), 
      (∀ i : Fin (n + 1), beakers' i = 
        if i = ∃ i', beakers i'
        then liquids i' 
        else 1/n * (∑ i' in Finset.range n, liquids i')) )

theorem mixture_is_possible : ∀ n : ℕ, uniform_mixture_possible n :=
begin
  intros,
  sorry,  -- The actual proof will go here
end

end mixture_is_possible_l7_7266


namespace middle_and_oldest_son_ages_l7_7659

theorem middle_and_oldest_son_ages 
  (x y z : ℕ) 
  (father_age_current father_age_future : ℕ) 
  (youngest_age_increment : ℕ)
  (father_age_increment : ℕ) 
  (father_equals_sons_sum : father_age_future = (x + youngest_age_increment) + (y + father_age_increment) + (z + father_age_increment))
  (father_age_constraint : father_age_current + father_age_increment = father_age_future)
  (youngest_age_initial : x = 2)
  (father_age_current_value : father_age_current = 33)
  (youngest_age_increment_value : youngest_age_increment = 12)
  (father_age_increment_value : father_age_increment = 12) 
  :
  y = 3 ∧ z = 4 :=
begin
  sorry
end

end middle_and_oldest_son_ages_l7_7659


namespace train_speed_correct_l7_7641

-- Definitions for the given conditions
def train_length : ℝ := 320
def time_to_cross : ℝ := 6

-- The speed of the train
def train_speed : ℝ := 53.33

-- The proof statement
theorem train_speed_correct : train_speed = train_length / time_to_cross :=
by
  sorry

end train_speed_correct_l7_7641


namespace maximum_pairs_l7_7924

noncomputable def max_shape_pairs (side_large_square : ℝ) (side_small_square : ℝ) (hypotenuse_triangle : ℝ) : ℕ :=
  ⌊(side_large_square * side_large_square) / (side_small_square * side_small_square + (hypotenuse_triangle * hypotenuse_triangle / 2))⌋ 

theorem maximum_pairs
  (side_large_square : ℝ)
  (side_small_square : ℝ)
  (hypotenuse_triangle : ℝ)
  (h1 : side_large_square = 7)
  (h2 : side_small_square = 2)
  (h3 : hypotenuse_triangle = 3) :
  max_shape_pairs side_large_square side_small_square hypotenuse_triangle = 7 :=
by {
  sorry
}

end maximum_pairs_l7_7924


namespace sum_of_all_three_digit_numbers_sum_of_all_seven_digit_permutations_l7_7260

/-- 
Proof problem for the first question:
If we consider all possible three-digit numbers formed by the digits {1, 2, 3, 4}, 
where repetition is allowed, then the sum of all those numbers is 17760.
-/
theorem sum_of_all_three_digit_numbers : 
  ∑ n in (finset.pi_finset_of_three (finset.range 1 5)), n = 17760 :=
sorry

/-- 
Proof problem for the second question:
If we consider all possible permutations of the digits {1, 2, 3, 4, 5, 6, 7} to 
form seven-digit numbers, then the sum of all those numbers is 
28 × 6! × 1111111.
-/
theorem sum_of_all_seven_digit_permutations :
  ∑ perm in (finset.perm (finset.range 1 8)), (nat.digits 10 perm.val).sum = 
  (28 * nat.factorial 6 * 1111111) :=
sorry

end sum_of_all_three_digit_numbers_sum_of_all_seven_digit_permutations_l7_7260


namespace goldfish_problem_l7_7071

theorem goldfish_problem (x : ℕ) : 
  (18 + (x - 5) * 7 = 4) → (x = 3) :=
by
  intros
  sorry

end goldfish_problem_l7_7071


namespace total_canoes_proof_l7_7694

def n_canoes_january : ℕ := 5
def n_canoes_february : ℕ := 3 * n_canoes_january
def n_canoes_march : ℕ := 3 * n_canoes_february
def n_canoes_april : ℕ := 3 * n_canoes_march

def total_canoes_built : ℕ :=
  n_canoes_january + n_canoes_february + n_canoes_march + n_canoes_april

theorem total_canoes_proof : total_canoes_built = 200 := 
  by
  sorry

end total_canoes_proof_l7_7694


namespace limit_problem_l7_7696

theorem limit_problem :
  (Real.Lim at_top (λ n : ℝ, ((3 - n)^4 - (2 - n)^4) / ((1 - n)^5 - (1 + n)^3))) = 0 :=
by
  sorry

end limit_problem_l7_7696


namespace sum_of_valid_x_l7_7243

def mean (a b c d e f : ℝ) : ℝ := (a + b + c + d + e + f) / 6
def median_of_sorted_four (a b c d : ℝ) : ℝ := (b + c) / 2

theorem sum_of_valid_x :
  let mean_val := mean 3 5 7 x 18 20 in
  let sorted_with_x := if x ≤ 3 then [x, 3, 5, 7, 18, 20]
                       else if x ≤ 5 then [3, x, 5, 7, 18, 20]
                       else if x ≤ 7 then [3, 5, x, 7, 18, 20]
                       else if x ≤ 18 then [3, 5, 7, x, 18, 20]
                       else if x ≤ 20 then [3, 5, 7, 18, x, 20]
                       else [3, 5, 7, 18, 20, x] in
  let med_val := if x = sorted_with_x[2] then x
                 else if x = sorted_with_x[3] then (sorted_with_x[2] + x) / 2
                 else (sorted_with_x[2] + sorted_with_x[3]) / 2 in
  ∀ x : ℝ, med_val = mean_val → x = 53 / 5 :=
by sorry

end sum_of_valid_x_l7_7243


namespace max_segments_in_square_network_l7_7470

theorem max_segments_in_square_network (n : Nat) : 
  let total_points := n + 4
  let vertices_at_square := 4
  let internal_points := n
  ∀ (segments : Finset (Finset (Fin total_points))), -- Each segment is represented as a set of two points
  (∀ seg ∈ segments, seg.card = 2) ∧                 -- Segments are pairs of points
  (∀ seg1 seg2 ∈ segments, seg1 ≠ seg2 → seg1 ∩ seg2 ⊆ ∅) ∧ -- No intersection except endpoints
  (∀ pt ∈ Finset.univ.filter (λ p, p ∉ Finset.image₂ Pair.mk Finset.univ (Finset.erase Finset.univ pt)), -- No segment contains other points other than endpoints
  True) →
  segments.card ≤ 3 * n + 5 := 
by
  sorry

end max_segments_in_square_network_l7_7470


namespace minimum_value_l7_7042

theorem minimum_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 2 * a + 3 * b = 1) : 
  26 ≤ (2 / a + 3 / b) :=
sorry

end minimum_value_l7_7042


namespace number_of_classmates_ate_cake_l7_7141

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l7_7141


namespace find_a_l7_7773

theorem find_a (a : ℝ) :
  (∃ k x₀, y₀ = x₀^3 ∧ k = 3 * x₀^2 ∧ k = (y₀ / (x₀ - 1)) ∧
    ((y = (k * (x - 1))) ℧ y = (ax * x + (15 / 4) * x - 9)) ∧
    (tangent_line = y - ax * x - 15 / 4 * x + 9) :=
  a = -25 / 64 ∨ a = -1 :=
sorry

end find_a_l7_7773


namespace classmates_ate_cake_l7_7085

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l7_7085


namespace min_workers_l7_7716

-- Define the conditions
variables (total_days : ℕ) (workers_initial : ℕ) (fraction_completed : ℚ) (days_passed : ℕ) (remaining_fraction : ℚ)

-- Initial values
def total_days := 20
def workers_initial := 10
def fraction_completed := 1 / 4
def days_passed := 5

-- Leaving a definition here to describe the remaining work and days left
def days_remaining := total_days - days_passed
def project_remaining := 1 - fraction_completed

-- Required daily rate to complete the remaining project on time
def required_rate_per_day := project_remaining / days_remaining

-- Calculate the rate of work per day by initial workers
def work_rate_per_day := fraction_completed / days_passed

-- Each worker's daily contribution to the project
def worker_contribution := work_rate_per_day / workers_initial

-- Minimum number of workers required to meet the necessary daily rate
noncomputable def workers_min_required := ⌈required_rate_per_day / worker_contribution⌉₊  -- ℕ for number of workers, ceil for minimum whole number

-- Theorem: proving the minimum number of workers Alex must retain
theorem min_workers (h1 : total_days = 20)
                    (h2 : workers_initial = 10)
                    (h3 : fraction_completed = 1 / 4)
                    (h4 : days_passed = 5)
                    : workers_min_required total_days workers_initial fraction_completed days_passed project_remaining days_remaining required_rate_per_day work_rate_per_day worker_contribution = 5 :=
begin
  -- The proof should be written here, but we will use sorry to skip it for now.
  sorry
end

end min_workers_l7_7716


namespace problem_statement_l7_7500

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eqn (x y : ℝ) : g (x * g y + 2 * x) = 2 * x * y + g x

def n : ℕ := sorry  -- n is the number of possible values of g(3)
def s : ℝ := sorry  -- s is the sum of all possible values of g(3)

theorem problem_statement : n * s = 0 := sorry

end problem_statement_l7_7500


namespace arithmetic_square_root_of_sqrt_81_squared_theorem_cube_root_of_one_over_27_theorem_reciprocal_of_sqrt_2_theorem_l7_7983

-- Definition and theorem for Arithmetic Square Root of sqrt((-81)^2)
def arithmetic_square_root_of_sqrt_81_squared := sqrt (sqrt ((-81)^2)) = 9

-- Definition and theorem for Cube Root of 1/27
def cube_root_of_one_over_27 := cbrt (1 / 27) = 1 / 3

-- Definition and theorem for Reciprocal of sqrt(2)
def reciprocal_of_sqrt_2 := 1 / sqrt 2 = sqrt 2 / 2

-- Mathematically state all the proofs
theorem arithmetic_square_root_of_sqrt_81_squared_theorem : arithmetic_square_root_of_sqrt_81_squared :=
by sorry

theorem cube_root_of_one_over_27_theorem : cube_root_of_one_over_27 :=
by sorry

theorem reciprocal_of_sqrt_2_theorem : reciprocal_of_sqrt_2 :=
by sorry

end arithmetic_square_root_of_sqrt_81_squared_theorem_cube_root_of_one_over_27_theorem_reciprocal_of_sqrt_2_theorem_l7_7983


namespace AZ_length_l7_7503

noncomputable def AZ_solution : ℝ :=
  1 / (1 + real.cbrt 2)

theorem AZ_length {A B C D E F X Y Z : Type} 
  (h_equilateral_ABC : equilateral ABC 1)
  (h_D_inside_ABC : inside_triangle D ABC)
  (h_E_inside_ABC : inside_triangle E ABC)
  (h_F_inside_ABC : inside_triangle F ABC)
  (h_A_E_F_collinear : collinear_points A E F)
  (h_B_F_D_collinear : collinear_points B F D)
  (h_C_D_E_collinear : collinear_points C D E)
  (h_equilateral_DEF : equilateral DEF)
  (h_unique_equilateral_XYZ : unique_equilateral_triangle XYZ (side BC) (side AB) (side AC))
  (h_X_on_BC : on_side X BC)
  (h_Y_on_AB : on_side Y AB)
  (h_Z_on_AC : on_side Z AC)
  (h_D_on_XZ : on_side D XZ)
  (h_E_on_YZ : on_side E YZ)
  (h_F_on_XY : on_side F XY) :
  AZ = AZ_solution :=
sorry

end AZ_length_l7_7503


namespace find_b_l7_7175

theorem find_b (b p : ℝ) (h_factor : ∃ k : ℝ, 3 * (x^3 : ℝ) + b * x + 9 = (x^2 + p * x + 3) * (k * x + k)) :
  b = -6 :=
by
  obtain ⟨k, h_eq⟩ := h_factor
  sorry

end find_b_l7_7175


namespace classmates_ate_cake_l7_7088

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l7_7088


namespace running_speeds_and_lcm_times_l7_7489

theorem running_speeds_and_lcm_times
  (T_jerome : ℕ) (T_nero : ℕ) (T_amelia : ℕ)
  (S_jerome : ℕ)
  (same_start : T_jerome = 6 ∧ T_nero = 3 ∧ T_amelia = 4 ∧ S_jerome = 4) :
  let trail_length := S_jerome * T_jerome in
  let S_nero := trail_length / T_nero in
  let S_amelia := trail_length / T_amelia in
  S_nero = 8 ∧ S_amelia = 6 ∧ nat.lcm (nat.lcm T_jerome T_nero) T_amelia = 12 :=
by
  sorry

end running_speeds_and_lcm_times_l7_7489


namespace fraction_for_repeating_decimal_l7_7763

variable (a r S : ℚ)
variable (h1 : a = 3/5)
variable (h2 : r = 1/10)
variable (h3 : S = a / (1 - r))

theorem fraction_for_repeating_decimal : S = 2 / 3 :=
by
  have h4 : 1 - r = 9 / 10, from sorry
  have h5 : S = (3 / 5) / (9 / 10), from sorry
  have h6 : S = (3 * 10) / (5 * 9), from sorry
  have h7 : S = 30 / 45, from sorry
  have h8 : 30 / 45 = 2 / 3, from sorry
  exact h8

end fraction_for_repeating_decimal_l7_7763


namespace find_n_minus_m_l7_7465

theorem find_n_minus_m (Q R S T: Type) (QT TS : Type) (m n : ℕ) (p : ℕ) 
  (hAngleBisector : ∠(R, Q, T) = ∠(R, S, T))
  (hQT : QT = m) (hTS : TS = n)
  (hn_gt_m : n > m) 
  (hDivisible : (n + m) % (n - m) = 0) 
  (hPerimeter : p) 
  (hPossibleValues : ∃ k : ℕ, k = m^2 + 2 * m - 1 ∧ p = k) :
  n - m = 4 := 
sorry

end find_n_minus_m_l7_7465


namespace find_k_and_a_l7_7247

def poly1 (x : ℝ) : ℝ := x^4 - 6*x^3 + 16*x^2 - 25*x + 10
def divisor (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + k
def remainder (a : ℝ) (x : ℝ) : ℝ := x + a

theorem find_k_and_a :
  ∃ (k a : ℝ), (poly1 = λ x, quotient (poly1 x) (divisor k x) * divisor k x + remainder a x) ∧ (k = 5) ∧ (a = -5) :=
sorry

end find_k_and_a_l7_7247


namespace geometric_sequence_sum_a_value_l7_7883

theorem geometric_sequence_sum_a_value
    (S : ℕ → ℝ)
    (a_1 a_2 a_3 a : ℝ)
    (h1 : ∀ n, S n = 4^(n + 1) + a)
    (h2 : a_1 = S 1)
    (h3 : ∀ n, a_(n + 1) = S (n + 1) - S n)
    (h4 : a_2^2 = a_1 * a_3) :
  a = -4 := by
  sorry

end geometric_sequence_sum_a_value_l7_7883


namespace meet_time_correct_meet_time_l7_7922

noncomputable theory

def time_when_meet (t : ℤ) : Prop :=
  ∃ t : ℝ, (15 * t) + (20 * (t - 0.75)) - 15 = 85

theorem meet_time : ∃ t : ℝ, 15 * t + 20 * (t - 0.75) = 85 := by
  sorry

theorem correct_meet_time : time_when_meet 2.857 := by
  use 2.857
  calc
    15 * 2.857 + 20 * (2.857 - 0.75) - 15 
      = (15 * 2.857) + 20 * (2.857 - 0.75) - 15 : by sorry
      = 85 : by sorry
  sorry

end meet_time_correct_meet_time_l7_7922


namespace candidateX_win_percentage_l7_7893

-- Definitions based on conditions
def ratio_republicans := 3
def ratio_democrats := 2
def ratio_independents := 1

def total_voters := 600
def republicans := (ratio_republicans / (ratio_republicans + ratio_democrats + ratio_independents)) * total_voters
def democrats := (ratio_democrats / (ratio_republicans + ratio_democrats + ratio_independents)) * total_voters
def independents := (ratio_independents / (ratio_republicans + ratio_democrats + ratio_independents)) * total_voters

def candidateX_votes :=
  0.70 * republicans +
  0.25 * democrats +
  0.40 * independents

def candidateY_votes :=
  (1 - 0.70 - 0.05) * republicans +
  (1 - 0.25 - 0.05) * democrats +
  (1 - 0.40 - 0.05) * independents

def candidateZ_votes :=
    0.05 * republicans +
    0.05 * democrats +
    0.05 * independents

def total_votes_XY := candidateX_votes + candidateY_votes

def vote_difference := candidateX_votes - candidateY_votes

def win_percentage_X_over_Y := (vote_difference / total_votes_XY) * 100

theorem candidateX_win_percentage : win_percentage_X_over_Y ≈ 5.26 := sorry

end candidateX_win_percentage_l7_7893


namespace house_number_is_110_l7_7343

def is_digit (d : ℕ) : Prop := d > 0 ∧ d < 10

def is_prime (n : ℕ) : Prop :=  -- Prime checking function
  n > 1 ∧ (∀ k : ℕ, (k ∣ n) → (k = 1 ∨ k = n))

def is_two_digit_prime (n : ℕ) : Prop := 
  n ≥ 10 ∧ n < 50 ∧ is_prime n

def house_number_count : ℕ := 
  -- All pairs (AB, CD) where AB and CD are distinct two-digit primes
  (List.filter is_two_digit_prime (List.range 50)).length * 
  ((List.filter is_two_digit_prime (List.range 50)).length - 1)

theorem house_number_is_110 : house_number_count = 110 := sorry

end house_number_is_110_l7_7343


namespace mary_can_keep_warm_l7_7954

theorem mary_can_keep_warm :
  let chairs := 18
  let chairs_sticks := 6
  let tables := 6
  let tables_sticks := 9
  let stools := 4
  let stools_sticks := 2
  let sticks_per_hour := 5
  let total_sticks := (chairs * chairs_sticks) + (tables * tables_sticks) + (stools * stools_sticks)
  let hours := total_sticks / sticks_per_hour
  hours = 34 := by
{
  sorry
}

end mary_can_keep_warm_l7_7954


namespace probability_of_interval_l7_7870

noncomputable def probability_less_than (a b : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < b then (x - a) / (b - a) else 0

theorem probability_of_interval (a b x : ℝ) (ha : 0 < a) (hb : a < b) :
  probability_less_than 0 (1/2) (1/3) = 2/3 :=
by
  have h_interval : (1 : ℝ)/2 - 0 = (1/2) := by norm_num,
  have h_favorable : (1 : ℝ)/3 - 0 = (1/3) := by norm_num,
  rw [← h_interval, ← h_favorable, probability_less_than],
  split_ifs,
  simpa using div_eq_mul_one_div (1/3) (1/2),
  sorry

end probability_of_interval_l7_7870


namespace total_money_calculated_l7_7719

namespace PastryShop

def original_price_cupcake : ℝ := 3.00
def original_price_cookie : ℝ := 2.00
def reduced_price (original_price : ℝ) : ℝ := original_price / 2
def num_cupcakes_sold : ℕ := 16
def num_cookies_sold : ℕ := 8

def total_money_made : ℝ :=
  reduced_price original_price_cupcake * num_cupcakes_sold + 
  reduced_price original_price_cookie * num_cookies_sold

theorem total_money_calculated :
  total_money_made = 32 := by
  sorry

end PastryShop

end total_money_calculated_l7_7719


namespace number_of_classmates_l7_7115

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l7_7115


namespace mary_speed_l7_7950

theorem mary_speed (M : ℝ) (h1 : 4 = jimmy_speed) 
  (h2 : distance_after_1_hour M 4 = 9) : M = 5 :=
by {
  sorry
}

end mary_speed_l7_7950


namespace degree_of_f_at_least_n_l7_7056

variables {R : Type*} [CommRing R] [Algebra R ℝ]

-- Define the polynomial f
noncomputable def f (n m : ℕ) (hn : 2 ≤ n) (hm : 2 ≤ m) : Polynomial ℝ :=
  Polynomial.ofFn (λ (xi : Vector ℕ n), ⌊(xi.toList.sum) / m⌋)

theorem degree_of_f_at_least_n (n m : ℕ) (hn : 2 ≤ n) (hm : 2 ≤ m) :
  ∀ (f : Vector ℕ n → ℝ),
  (∀ (xi : Vector ℕ n), (∀ i, xi.get i ∈ Finset.range m) → f xi = ⌊(xi.toList.sum : ℝ) / m⌋) →
  (Polynomial.totalDegree (f n m) ≥ n) :=
sorry

end degree_of_f_at_least_n_l7_7056


namespace num_outcomes_correct_l7_7476

-- Definitions based on given conditions
def Candidates : Type := {A, B, C, D}
def Positions := {Secretary, DeputySecretary, OrganizationCommitteeMember}

noncomputable def incumbents := {A, B, C}

def valid_selection (selection : Candidates → option Positions) : Prop :=
  ∀ (c : incumbents), selection c ≠ differentiate_position c

-- Assumed differentiate_position maps candidates to the positions they previously held.
axiom differentiate_position : Candidates → Positions

-- Number of different valid outcomes
noncomputable def num_valid_outcomes : ℕ := by sorry

-- Final Lean 4 statement proving the number of valid assignments is 11
theorem num_outcomes_correct : num_valid_outcomes = 11 :=
by sorry

end num_outcomes_correct_l7_7476


namespace salary_for_january_l7_7632

theorem salary_for_january (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8700)
  (h_may : May = 6500) :
  J = 3700 :=
by
  sorry

end salary_for_january_l7_7632


namespace display_after_200_presses_l7_7157

theorem display_after_200_presses : 
  (∀ f : ℚ → ℚ, f x = (x + 1) / (x - 1) ∧ ∃ x0 : ℚ, x0 = 3 ∧
  (∀ n : ℕ, (n % 2 = 0) → (iterate f n x0 = 3))) :=
sorry

end display_after_200_presses_l7_7157


namespace smallest_intervals_contain_first_100_terms_l7_7505

section main_problem

variables {n : ℕ} (x y z t : ℕ → ℝ)

def x_n := λ n, (-1 : ℝ)^n / n
def y_n := λ n, 1 - 1 / n
def z_n := λ n, |(n : ℝ) - 50|
def t_n := λ n, (-0.5) ^ n

theorem smallest_intervals_contain_first_100_terms :
  (∀ n, 1 ≤ n → n ≤ 100 → x n ∈ set.Icc (x_n 99) (x_n 2))
  ∧ (∀ n, 1 ≤ n → n ≤ 100 → y n ∈ set.Icc (y_n 1) (y_n 100))
  ∧ (∀ n, 1 ≤ n → n ≤ 100 → z n ∈ set.Icc (z_n 50) (z_n 1))
  ∧ (∀ n, 1 ≤ n → n ≤ 100 → t n ∈ set.Icc (t_n 99) (t_n 2))
  := by { sorry }

end main_problem

end smallest_intervals_contain_first_100_terms_l7_7505


namespace height_of_darkened_strip_l7_7521

theorem height_of_darkened_strip 
  (tv_aspect_width : ℕ) (tv_aspect_height : ℕ) 
  (tv_diagonal : ℝ) (movie_aspect_width : ℕ) (movie_aspect_height : ℕ) 
  (H1 : tv_aspect_width = 4) (H2 : tv_aspect_height = 3) 
  (H3 : tv_diagonal = 27) (H4 : movie_aspect_width = 2) (H5 : movie_aspect_height = 1) : 
  ∃ (strip_height : ℝ), strip_height = 2.7 :=
begin
  sorry
end

end height_of_darkened_strip_l7_7521


namespace remainder_2_pow_33_mod_9_l7_7178

theorem remainder_2_pow_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end remainder_2_pow_33_mod_9_l7_7178


namespace spherical_to_rectangular_l7_7717

-- Definitions of conditions
def rho := 4
def theta := (3 * Real.pi) / 2
def phi := Real.pi / 3

-- Conversion formulas from spherical coordinates to rectangular coordinates
def x := rho * Real.sin phi * Real.cos theta
def y := rho * Real.sin phi * Real.sin theta
def z := rho * Real.cos phi

-- The theorem we need to prove
theorem spherical_to_rectangular : (x, y, z) = (0, -2 * Real.sqrt 3, 2) :=
by
  -- Proof is omitted; using sorry as a placeholder.
  sorry

end spherical_to_rectangular_l7_7717


namespace maximum_sum_achieved_at_10_l7_7797

variable {α : Type*} [LinearOrderedAddCommGroup α]

-- Definitions for the arithmetic sequence and sum of the first n terms
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → α) : ℕ → α
| 0 => 0
| (n + 1) => sum_of_first_n_terms n + a n

-- The proof statement
theorem maximum_sum_achieved_at_10
  (a : ℕ → α)
  (d : α)
  (ha_seq : is_arithmetic_sequence a d)
  (h10_11 : a 10 + a 11 > 0)
  (h10_12 : a 10 + a 12 < 0) :
  ∀ n, sum_of_first_n_terms a n ≤ sum_of_first_n_terms a 10 :=
begin
  sorry -- The proof itself is not required.
end

end maximum_sum_achieved_at_10_l7_7797


namespace variance_nine_data_points_l7_7398

noncomputable def data_points_8_mean : ℝ := 5
noncomputable def data_points_8_variance : ℝ := 3
noncomputable def new_data_point : ℝ := 5

theorem variance_nine_data_points (mean_8 : data_points_8_mean = 5) 
                                   (variance_8 : data_points_8_variance = 3) 
                                   (new_data : new_data_point = 5) :
  let mean_9 : ℝ := (8 * mean_8 + new_data) / 9
  in mean_9 = 5 ∧ (1 / 9) * (8 * variance_8 + (new_data - mean_9)^2) = 8 / 3 := 
by
  sorry

end variance_nine_data_points_l7_7398


namespace soap_bars_packing_l7_7610

theorem soap_bars_packing : 
  ∃ N : ℕ, (0 < N) ∧ (2007 % N = 5) ∧ 
  (({n : ℕ | (2007 % n = 5) ∧ (n > 5)}).toFinset.card = 14) :=
by
  sorry

end soap_bars_packing_l7_7610


namespace number_of_classmates_l7_7117

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l7_7117


namespace num_even_factors_l7_7441

def n : ℕ := 2^4 * 3^3 * 5^2

theorem num_even_factors : 
  let num_factors := (λ (k : ℕ), (∃ (a b c : ℕ), a + b + c = k ∧ a ∈ {1, 2, 3, 4} ∧ b ∈ {0, 1, 2, 3} ∧ c ∈ {0, 1, 2})) in
  ∑ k in {k | num_factors k}, 1 = 48 :=
sorry

end num_even_factors_l7_7441


namespace min_value_expr_l7_7336

theorem min_value_expr : 
  ∃ (n : ℕ), 0 < n ∧ ∀ m : ℕ, 0 < m → (n = m  ∨ (n / 2 + 24 / n < m / 2 + 24 / m)) ∧ 
  (n / 2 + 24 / n = 7) :=
begin
  -- sorry, omitted
  sorry
end

end min_value_expr_l7_7336


namespace classmates_ate_cake_l7_7089

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l7_7089


namespace p_sufficient_but_not_necessary_q_l7_7391

-- Definitions and Conditions
variables {x0 y0 : ℝ} 

def p : Prop := (y0 ≠ 0) ∧ (y0 / x0 = 1 / -y0)
def q : Prop := (y0 ^ 2 = - x0)

-- Proof Statement
theorem p_sufficient_but_not_necessary_q : (p → q) ∧ ¬ (q → p) := by
  sorry

end p_sufficient_but_not_necessary_q_l7_7391


namespace eq_OP_OQ_l7_7049

variables (A B C P Q O K L M : Point)
variables (circumcircle : Circle A B C)
variables (Gamma : Circle K L M)

-- Conditions
variable (circumcenter : is_circumcenter O A B C)
variable (P_on_CA : on_line P (line C A))
variable (Q_on_AB : on_line Q (line A B))
variable (K_midpoint_BP : midpoint K (segment B P))
variable (L_midpoint_CQ : midpoint L (segment C Q))
variable (M_midpoint_PQ : midpoint M (segment P Q))
variable (Gamma_tangent_PQ : tangent Gamma (line P Q))

-- The goal
theorem eq_OP_OQ : dist O P = dist O Q :=
sorry

end eq_OP_OQ_l7_7049


namespace scale_down_multiplication_l7_7373

theorem scale_down_multiplication (h : 14.97 * 46 = 688.62) : 1.497 * 4.6 = 6.8862 :=
by
  -- here we assume the necessary steps to justify the statement.
  sorry

end scale_down_multiplication_l7_7373


namespace equation_no_solution_B_l7_7249

theorem equation_no_solution_B :
  ¬(∃ x : ℝ, |-3 * x| + 5 = 0) :=
sorry

end equation_no_solution_B_l7_7249


namespace find_n_l7_7355

variable {θ : ℝ}

theorem find_n (h : θ ∉ { k * (π/2) | k : ℤ }) : 
  ∀ n > 0,
  (∀ θ ∈ Ioo 0 π ∪ Ioo π (2*π), 
    (sin (n * θ) / sin θ - cos (n * θ) / cos θ = n - 1)) → 
  n = 1 := 
begin
  sorry
end

end find_n_l7_7355


namespace sara_golf_balls_l7_7974

theorem sara_golf_balls (n : ℕ) (h : n = 9 * 12) : n = 108 :=
by
  rw [h]
  rfl

end sara_golf_balls_l7_7974


namespace two_pow_n_plus_one_divisible_by_three_l7_7333

theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h1 : n > 0) :
  (2 ^ n + 1) % 3 = 0 ↔ n % 2 = 1 := 
sorry

end two_pow_n_plus_one_divisible_by_three_l7_7333


namespace construct_triangle_from_bisectors_and_altitude_l7_7409

theorem construct_triangle_from_bisectors_and_altitude 
  (α β γ : ℝ)   -- angles formed by the perpendicular bisectors of a triangle
  (h : ℝ)       -- one of the altitudes of the triangle
  : ∃ (△ : Type) (A B C : △), 
   let bisector1 := α ∧ let bisector2 := β ∧ let bisector3 := γ ∧ let altitude := h in
   is_triangle △ A B C ∧ 
   perpendicular_bisectors △ A B C α β γ ∧ 
   has_altitude △ A B C h := 
sorry

end construct_triangle_from_bisectors_and_altitude_l7_7409


namespace ages_of_sons_l7_7653

variable (x y z : ℕ)

def father_age_current : ℕ := 33

def youngest_age_current : ℕ := 2

def father_age_in_12_years : ℕ := father_age_current + 12

def sum_of_ages_in_12_years : ℕ := youngest_age_current + 12 + y + 12 + z + 12

theorem ages_of_sons (x y z : ℕ) 
  (h1 : x = 2)
  (h2 : father_age_current = 33)
  (h3 : father_age_in_12_years = 45)
  (h4 : sum_of_ages_in_12_years = 45) :
  x = 2 ∧ y + z = 7 ∧ ((y = 3 ∧ z = 4) ∨ (y = 4 ∧ z = 3)) :=
by
  sorry

end ages_of_sons_l7_7653


namespace find_x_in_interval_l7_7354

noncomputable def solution (x : ℝ) : Prop :=
  2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ∧
  abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2

theorem find_x_in_interval :
  ∀ x, x ∈ Set.Icc 0 (2 * Real.pi) → solution x ↔ x ∈ Set.Icc (Real.pi / 4) (7 * Real.pi / 4) :=
by
  intros x hx
  have : solution x ↔ x ∉ Set.Ioo (7 * Real.pi / 4) (2 * Real.pi) ∨ x ∉ Set.Ioo 0 (Real.pi / 4)
  sorry

end find_x_in_interval_l7_7354


namespace area_of_rectangle_PQRS_l7_7026

-- Definitions for the lengths of the sides of triangle ABC.
def AB : ℝ := 15
def AC : ℝ := 20
def BC : ℝ := 25

-- Definition for the length of PQ in rectangle PQRS.
def PQ : ℝ := 12

-- Definition for the condition that PQ is parallel to BC and RS is parallel to AB.
def PQ_parallel_BC : Prop := True
def RS_parallel_AB : Prop := True

-- The theorem to be proved: the area of rectangle PQRS is 115.2.
theorem area_of_rectangle_PQRS : 
  (∃ h: ℝ, h = (AC * PQ / BC) ∧ PQ * h = 115.2) :=
by {
  sorry
}

end area_of_rectangle_PQRS_l7_7026


namespace speed_of_train_b_l7_7634

-- Defining the known data
def train_a_speed := 60 -- km/h
def train_a_time_after_meeting := 9 -- hours
def train_b_time_after_meeting := 4 -- hours

-- Statement we want to prove
theorem speed_of_train_b : ∃ (V_b : ℝ), V_b = 135 :=
by
  -- Sorry placeholder, as the proof is not required
  sorry

end speed_of_train_b_l7_7634


namespace find_x_l7_7885

theorem find_x : ∃ x : ℕ, 19250 % x = 11 ∧ 20302 % x = 3 ∧ x = 53 :=
by
  use 53
  split
  . norm_num
  . split
    . norm_num
    . refl
  sorry

end find_x_l7_7885


namespace hercules_defeats_hydra_l7_7842

theorem hercules_defeats_hydra :
  ∃ N : ℕ, (∀ hydra : SimpleGraph, hydra.Adj.size = 100 ∧ hydra.Adj.card = 100 →
    (∀ k: ℕ, k < N → hydra.IsConnected)) → (N = 10) :=
begin
  sorry
end

end hercules_defeats_hydra_l7_7842


namespace cake_sharing_l7_7102

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l7_7102


namespace inheritance_amount_l7_7519

def is_inheritance_amount (x : ℝ) : Prop :=
  let federal_tax := 0.25 * x
  let remaining_after_fed := x - federal_tax
  let state_tax := 0.12 * remaining_after_fed
  let total_tax_paid := federal_tax + state_tax
  total_tax_paid = 15600

theorem inheritance_amount : 
  ∃ x, is_inheritance_amount x ∧ x = 45882 := 
by
  sorry

end inheritance_amount_l7_7519


namespace calculate_expression_l7_7608

theorem calculate_expression :
  3^(1+2+3) - (3^1 + 3^2 + 3^3) = 690 :=
by
  sorry

end calculate_expression_l7_7608


namespace simplify_expression_l7_7542

theorem simplify_expression (r : ℝ) : 100*r - 48*r + 10 = 52*r + 10 :=
by
  sorry

end simplify_expression_l7_7542


namespace eval_expression_l7_7315

theorem eval_expression : 3 - (-3)⁻³ = 82 / 27 := by
  sorry

end eval_expression_l7_7315


namespace reporters_covering_local_politics_l7_7649

theorem reporters_covering_local_politics (R : ℕ) (P Q A B : ℕ)
  (h1 : P = 70)
  (h2 : Q = 100 - P)
  (h3 : A = 40)
  (h4 : B = 100 - A) :
  B % 30 = 18 :=
by
  sorry

end reporters_covering_local_politics_l7_7649


namespace xyz_equivalence_l7_7781

theorem xyz_equivalence (x y z a b : ℝ) (h₁ : 4^x = a) (h₂: 2^y = b) (h₃ : 8^z = a * b) : 3 * z = 2 * x + y :=
by
  -- Here, we leave the proof as an exercise
  sorry

end xyz_equivalence_l7_7781


namespace reflected_ray_eq_l7_7669

theorem reflected_ray_eq (M : ℝ × ℝ) (α : ℝ) (tan_alpha : ℝ) :
  M = (5, 3) ∧ tan_alpha = 3 →
  ∃ b : ℝ, (∀ x : ℝ, ((tan (π - α) * x + b = -3 * x + 12) ∧ M = (5, 3)) ∧
  tan (π - α) = -3) :=
begin
  sorry
end

end reflected_ray_eq_l7_7669


namespace water_volume_per_minute_l7_7670

noncomputable def river_depth : ℝ := 3 -- in meters
noncomputable def river_width : ℝ := 55 -- in meters
noncomputable def river_flow_rate_kmph : ℝ := 1 -- in km/h

theorem water_volume_per_minute :
  let area := river_depth * river_width,
      flow_rate_m_per_min := (river_flow_rate_kmph * 1000) / 60,
      volume_per_minute := area * flow_rate_m_per_min
  in volume_per_minute = 2750.55 := 
by
  sorry

end water_volume_per_minute_l7_7670


namespace geometric_ratio_value_l7_7399
-- Import the necessary library

-- Define the geometric sequence with common ratio r
variable (a_1 : ℝ) (r : ℝ := -1/2)

-- Define the terms of the sequence as provided by the conditions
def a_2 : ℝ := a_1 * r
def a_3 : ℝ := a_1 * r^2
def a_4 : ℝ := a_1 * r^3
def a_5 : ℝ := a_1 * r^4
def a_6 : ℝ := a_1 * r^5

-- State the theorem we need to prove
theorem geometric_ratio_value :
  (a_1 + a_3 + a_5) / (a_2 + a_4 + a_6) = -2 :=
by
  sorry

end geometric_ratio_value_l7_7399


namespace fifth_root_of_102030201_l7_7329

theorem fifth_root_of_102030201 :
  ∃ n : ℕ, n = 101 ∧ n ^ 5 = 102030201 :=
by
  have h_exp : 102030201 = (10 ^ 2 + 1) ^ 5 := by sorry
  use 101
  split
  . rfl
  . rw h_exp

end fifth_root_of_102030201_l7_7329


namespace asymptotes_of_hyperbola_l7_7986

theorem asymptotes_of_hyperbola :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 9 = 1) → (y = 3/2 * x ∨ y = -3/2 * x) :=
by
  intro x y h
  -- Proof would go here
  sorry

end asymptotes_of_hyperbola_l7_7986


namespace football_banquet_min_guests_l7_7306

noncomputable def min_guests (total_food : ℕ) (meat_ratio : ℕ) (veg_ratio : ℕ) (dessert_ratio : ℕ)
  (meat_limit : ℕ) (veg_limit : ℕ) (dessert_limit : ℕ) : ℕ :=
(total_food + meat_limit + veg_limit + dessert_limit - 1) / (meat_limit + veg_limit + dessert_limit)

theorem football_banquet_min_guests :
  let total_food := 319 in
  let meat_ratio := 3 in
  let veg_ratio := 1 in
  let dessert_ratio := 1 in
  let meat_limit := 150 in -- 1.5 pounds in 0.1 pounds unit
  let veg_limit := 30 in -- 0.3 pounds in 0.1 pounds unit
  let dessert_limit := 20 in -- 0.2 pounds in 0.1 pounds unit
  min_guests total_food meat_ratio veg_ratio dessert_ratio meat_limit veg_limit dessert_limit 
  = 160 := sorry

end football_banquet_min_guests_l7_7306


namespace find_value_l7_7478

variable {a b c : ℝ}

def ellipse_eqn (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

theorem find_value 
  (h1 : a^2 + b^2 - 3*c^2 = 0)
  (h2 : a^2 = b^2 + c^2) :
  (a + c) / (a - c) = 3 + 2 * Real.sqrt 2 := 
  sorry

end find_value_l7_7478


namespace function_is_monotonically_decreasing_l7_7560

-- Define the conditions
def f (x : ℝ) : ℝ := x^3 - 48 * x

-- Define the problem statement
theorem function_is_monotonically_decreasing : ∀ x ∈ Icc (-4 : ℝ) (4 : ℝ), is_monotone_decreasing_on f (Icc (-4 : ℝ) (4 : ℝ)) :=
by
  sorry

end function_is_monotonically_decreasing_l7_7560


namespace repeating_decimal_sum_l7_7702

theorem repeating_decimal_sum :
  (0.\overline{3} = 1 / 3) → (0.\overline{6} = 2 / 3) → (0.\overline{3} + 0.\overline{6} = 1) :=
by
  sorry

end repeating_decimal_sum_l7_7702


namespace sum_of_roots_l7_7579

theorem sum_of_roots :
  let p : Polynomial ℝ := Polynomial.C 20 + Polynomial.X * (Polynomial.C (-6) + Polynomial.X * (Polynomial.C (-4) + Polynomial.X)) in
  p.roots.sum = 4 :=
by
  -- defining the polynomial
  let p : Polynomial ℝ := Polynomial.C 20 + Polynomial.X * (Polynomial.C (-6) + Polynomial.X * (Polynomial.C (-4) + Polynomial.X))
  -- asserting the expected result of the sum of its roots
  show p.roots.sum = 4
  sorry

end sum_of_roots_l7_7579


namespace find_a_l7_7375

def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + 3 * x ^ 2 + 2

def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * a * x ^ 2 + 6 * x

theorem find_a : (f_prime a (-1) = 3) → a = 3 :=
by
  sorry

end find_a_l7_7375


namespace exists_column_contains_exactly_n_colors_l7_7346

open Function

def checkered_plane (α : Type*) := ℕ × ℕ → α

variables {α : Type*} [fintype α]

noncomputable def cell_colored_in_n_squared_colors (plane : checkered_plane α) (n : ℕ) :=
∀ i j, plane (i, j) ∈ (finset.range (n ^ 2)).val

noncomputable def every_nxn_square_contains_all_colors (plane : checkered_plane α) (n : ℕ) :=
∀ i j, (finset.univ.image (λ k, plane (i + k / n, j + k % n))).card = n ^ 2

noncomputable def one_row_contains_all_colors (plane : checkered_plane α) (n : ℕ) :=
∃ i, (finset.univ.image (λ j, plane (i, j))).card = n ^ 2

theorem exists_column_contains_exactly_n_colors (plane : checkered_plane α) (n : ℕ) 
  (h1 : cell_colored_in_n_squared_colors plane n)
  (h2 : every_nxn_square_contains_all_colors plane n)
  (h3 : one_row_contains_all_colors plane n) :
  ∃ j, (finset.univ.image (λ i, plane (i, j))).card = n := 
sorry

end exists_column_contains_exactly_n_colors_l7_7346


namespace Q_10_equals_2_div_13_l7_7039

noncomputable def T (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def Q (n : ℕ) : ℚ :=
∏ i in Finset.range (n - 1) + 2, 
  (T i) / (T (i + 1) - 1)

theorem Q_10_equals_2_div_13 : Q 10 = 2 / 13 := 
sorry

end Q_10_equals_2_div_13_l7_7039


namespace probability_purple_greater_green_l7_7668

-- Definitions
def green_point_range := set.Icc (0 : ℝ) 2
def purple_point_range := set.Icc (0 : ℝ) 2

-- Probability calculation function
def probability_of_condition (x y : ℝ) :=
  x < y ∧ y < 3 * x

-- Theorem statement
theorem probability_purple_greater_green (green_point purple_point : ℝ) :
  green_point ∈ green_point_range →
  purple_point ∈ purple_point_range →
  (∫ x in 0..(2/3), ∫ y in x..(3*x), 1) / 4 = 1 / 9 :=
by
  sorry

end probability_purple_greater_green_l7_7668


namespace count_arithmetic_sequence_l7_7841

open Nat

theorem count_arithmetic_sequence :
  let seq := list.range' 42 (40 * 3 + 1) \ list.map (· * 3) $ list.range' 14 41
  seq.length = 41 := by
  sorry

end count_arithmetic_sequence_l7_7841


namespace monotonically_decreasing_iff_l7_7804

noncomputable def f (a x : ℝ) : ℝ := (x^2 - 2 * a * x) * Real.exp x

theorem monotonically_decreasing_iff (a : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ f a (-1) ∧ f a x ≤ f a 1) ↔ (a ≥ 3 / 4) :=
by
  sorry

end monotonically_decreasing_iff_l7_7804


namespace algebraic_expression_value_l7_7376

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y^2 = 1) : -2 * x + 4 * y^2 + 1 = -1 :=
by
  sorry

end algebraic_expression_value_l7_7376


namespace angles_equal_l7_7340

open Real EuclideanGeometry

theorem angles_equal 
  (O A B C D P: Point) 
  (h_circle: Circle O R)
  (h_diameter: Diameter AB) 
  (h_on_circle_C: OnCircle C O R)
  (h_on_circle_D: OnCircle D O R)
  (h_on_diameter_same_side: SameSideAB C D)
  (h_on_diameter: OnLine P AB)
  (h_different_O_P: P ≠ O) 
  (h_concyclic: Concyclic {P, O, D, C}) : 
  (∠APC = ∠BPD) :=
by
  sorry

end angles_equal_l7_7340


namespace find_m_of_quadratic_root_l7_7381

theorem find_m_of_quadratic_root
  (m : ℤ) 
  (h : ∃ x : ℤ, x^2 - (m+3)*x + m + 2 = 0 ∧ x = 81) : 
  m = 79 :=
by
  sorry

end find_m_of_quadratic_root_l7_7381


namespace problem_part1_problem_part2_l7_7808

theorem problem_part1 (x y : ℝ) (h1 : x = 1 / (3 - 2 * Real.sqrt 2)) (h2 : y = 1 / (3 + 2 * Real.sqrt 2)) : 
  x^2 * y - x * y^2 = 4 * Real.sqrt 2 := 
  sorry

theorem problem_part2 (x y : ℝ) (h1 : x = 1 / (3 - 2 * Real.sqrt 2)) (h2 : y = 1 / (3 + 2 * Real.sqrt 2)) : 
  x^2 - x * y + y^2 = 33 := 
  sorry

end problem_part1_problem_part2_l7_7808


namespace number_of_classmates_ate_cake_l7_7139

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l7_7139


namespace perimeter_of_staircase_region_l7_7022

-- Definitions according to the conditions.
def staircase_region.all_right_angles : Prop := True -- Given condition that all angles are right angles.
def staircase_region.side_length : ℕ := 1 -- Given condition that the side length of each congruent side is 1 foot.
def staircase_region.total_area : ℕ := 120 -- Given condition that the total area of the region is 120 square feet.
def num_sides : ℕ := 12 -- Number of congruent sides.

-- The question is to prove that the perimeter of the region is 36 feet.
theorem perimeter_of_staircase_region : 
  (num_sides * staircase_region.side_length + 
    15 + -- length added to complete the larger rectangle assuming x = 15
    9   -- length added to complete the larger rectangle assuming y = 9
  ) = 36 := 
by
  -- Given and facts are already logically considered to prove (conditions and right angles are trivial)
  sorry

end perimeter_of_staircase_region_l7_7022


namespace floor_factorial_expression_l7_7319

theorem floor_factorial_expression :
  ⌊(2011! + 2008!) / (2010! + 2009!)⌋ = 2010 := 
sorry

end floor_factorial_expression_l7_7319


namespace max_tiles_proof_l7_7631

def floor : ℝ × ℝ := (560, 240)
def tile : ℝ × ℝ := (60, 56)
def obstacle1 : ℝ × ℝ := (40, 30)
def obstacle2 : ℝ × ℝ := (56, 20)
def obstacle3 : ℝ × ℝ := (32, 60)

-- Area calculation functions
def area (dim : ℝ × ℝ) : ℝ := dim.1 * dim.2

-- Areas of the components
def area_floor := area floor
def area_obstacle1 := area obstacle1
def area_obstacle2 := area obstacle2
def area_obstacle3 := area obstacle3
def area_obstacles := area_obstacle1 + area_obstacle2 + area_obstacle3

def area_available := area_floor - area_obstacles
def area_tile := area tile

-- Maximum number of tiles without covering any part of the obstacles
def max_tiles := (area_available / area_tile).toInt

theorem max_tiles_proof : max_tiles = 38 := by
  sorry

end max_tiles_proof_l7_7631


namespace eccentricity_range_circle_equation_l7_7385

noncomputable def ellipse_equation : Prop :=
  ∀ (x y a b : ℝ), a > b ∧ b > 0 → (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def point_A (a : ℝ) : Prop := A = (-a:ℝ, 0)

noncomputable def point_F (c : ℝ) : Prop := F = (c:ℝ, 0)

noncomputable def line_m : Prop := 
  ∀ (x : ℝ), (x / (a^2 / a / e) + c / a^2) ∈ m

theorem eccentricity_range (a b : ℝ) : Prop :=
  0 < e ∧ e ≤ 1 / 2

theorem circle_equation (x1 y1 x2 y2 : ℝ) (e : ℝ) : 
  ellipse_equation x1 y1 2 (real.sqrt 3) ∧ 
  ellipse_equation x2 y2 2 (real.sqrt 3) ∧ 
  (3/5 * x1 + 4/5 * x2) ^ 2 / 4 + (3/5 * y1 + 4/5 * y2) ^ 2 / 3 = 1 → 
  ((x + 1/2)^2 + (y ± real.sqrt (21) /4)^2 = 57/16)

end eccentricity_range_circle_equation_l7_7385


namespace find_b1_over_b2_l7_7981

variable {a b k a1 a2 b1 b2 : ℝ}

-- Assuming a is inversely proportional to b
def inversely_proportional (a b : ℝ) (k : ℝ) : Prop :=
  a * b = k

-- Define that a_1 and a_2 are nonzero and their ratio is 3/4
def a1_a2_ratio (a1 a2 : ℝ) (ratio : ℝ) : Prop :=
  a1 / a2 = ratio

-- Define that b_1 and b_2 are nonzero
def nonzero (x : ℝ) : Prop :=
  x ≠ 0

theorem find_b1_over_b2 (a1 a2 b1 b2 : ℝ) (h1 : inversely_proportional a b k)
  (h2 : a1_a2_ratio a1 a2 (3 / 4))
  (h3 : nonzero a1) (h4 : nonzero a2) (h5 : nonzero b1) (h6 : nonzero b2) :
  b1 / b2 = 4 / 3 := 
sorry

end find_b1_over_b2_l7_7981


namespace segment_length_of_absolute_value_l7_7207

theorem segment_length_of_absolute_value (x : ℝ) (h : abs (x - (27 : ℝ)^(1/3)) = 5) : 
  |8 - (-2)| = 10 :=
by
  sorry

end segment_length_of_absolute_value_l7_7207


namespace animals_competition_l7_7469

-- The variables representing the animals' participation
variables (L T P E : Prop)

-- Conditions given in the problem
hypothesis lion_if_tiger : L → T
hypothesis no_leopard_no_tiger : ¬ P → ¬ T
hypothesis leopard_no_elephant : P → ¬ E

-- The theorem to be proven: the pair {T, P} can participate
theorem animals_competition (h1 : L → T) (h2 : ¬ P → ¬ T) (h3 : P → ¬ E) : T ∧ P :=
by
  -- Proof would go here, but we are skipping it as per instructions
  sorry

end animals_competition_l7_7469


namespace portion_length_of_XY_in_third_cube_l7_7777

noncomputable def length_portion (X Y : (ℝ × ℝ × ℝ)) (cubes : List (ℝ × ℝ × ℝ × ℝ)) : ℝ :=
  have segment := (0, 0, 4, 4, 4, 8)
  Real.sqrt ((segment.4 - segment.1)^2 + (segment.5 - segment.2)^2 + (segment.6 - segment.3)^2)

theorem portion_length_of_XY_in_third_cube :
  ∀ X Y, X = (0, 0, 0) ∧ Y = (5, 5, 13) →
  ∀ cubes, cubes = [(0, 0, 0, 1), (0, 0, 1, 3), (0, 0, 4, 4), (0, 0, 8, 5)] →
  length_portion X Y cubes = 4 * Real.sqrt 3 := by sorry

end portion_length_of_XY_in_third_cube_l7_7777


namespace find_expression_of_f_l7_7400

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def f : ℝ → ℝ
| x := if x < 0 then x * (x + 1) else sorry

theorem find_expression_of_f (f : ℝ → ℝ) (h_even : is_even_function f) (h_neg_expr : ∀ x, x < 0 → f x = x * (x + 1)) :
  ∀ x, x > 0 → f x = x * (x - 1) :=
by 
  intros x hx
  have : f (-x) = x * (x - 1), from sorry -- uses the fact that -x ∈ (-∞, 0) and h_neg_expr
  have : f x = f (-x), from h_even x
  rw [this, this]
  sorry

end find_expression_of_f_l7_7400


namespace polynomial_arrangement_l7_7690

def poly : ℕ → ℕ → ℤ → ℤ := λ x y c, c + (x^3 : ℤ) + (3 * x * y^2 : ℤ) - (x^2 * y : ℤ)

theorem polynomial_arrangement :
  poly 1 1 (-9) = 1^3 - 1^2 * 1 + 3 * 1 * 1^2 - 9 :=
by
  sorry

end polynomial_arrangement_l7_7690


namespace correct_propositions_l7_7060

-- Definitions from the conditions
variable (α β : Plane)
variable (m n : Line)

-- Given plane and line relationships
variable (distinct_planes : α ≠ β)
variable (different_lines : m ≠ n)

-- Propositions as Lean statements
def proposition1 := m ⟂ n ∧ m ⟂ α → n ∥ α
def proposition2 := n ⊆ α ∧ m ⊆ β ∧ intersects α β ∧ ¬(α ⟂ β) → ¬(n ⟂ m)
def proposition3 := α ⟂ β ∧ (α ∩ β = m) ∧ (n ⊆ α) ∧ (n ⟂ m) → n ⟂ β
def proposition4 := m ∥ n ∧ n ⟂ α ∧ α ∥ β → m ⟂ β

-- Statement to prove the correctness of propositions 3 and 4
theorem correct_propositions : 
  (proposition3 α β m n distinct_planes different_lines) ∧ (proposition4 α β m n distinct_planes different_lines) :=
sorry

end correct_propositions_l7_7060


namespace lcm_from_1_to_12_eq_27720_l7_7233

theorem lcm_from_1_to_12_eq_27720 : nat.lcm (finset.range 12).succ = 27720 :=
  sorry

end lcm_from_1_to_12_eq_27720_l7_7233


namespace range_f_part1_range_of_k_part2_l7_7820

noncomputable def f_part1 (x : ℝ) : ℝ := x * Real.exp x

theorem range_f_part1 : 
  set.range (f_part1 ∘ λ x : ℝ, -2 ≤ x ∧ x ≤ 2) = set.Icc (-(1 / Real.exp 1)) (2 * Real.exp 2) :=
sorry

noncomputable def f_part2 (k x : ℝ) : ℝ := x * (Real.exp x - k * x)

theorem range_of_k_part2 (k : ℝ) : 
  (∀ x ∈ set.Ioo 0 Real.infty, f_part2 k x = 0 → f_part2 k x = 0 ∧ 
  (∀ y ∈ set.Ioo 0 Real.infty, (y ≠ x) → f_part2 k y ≠ 0)) ↔ k > Real.exp 1 :=
sorry

end range_f_part1_range_of_k_part2_l7_7820


namespace complement_of_intersection_l7_7423

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

theorem complement_of_intersection (AuB AcB : Set ℤ) :
  (A ∪ B) = AuB ∧ (A ∩ B) = AcB → 
  A ∪ B = ∅ ∨ A ∪ B = AuB → 
  (AuB \ AcB) = {-1, 1} :=
by
  -- Proof construction method placeholder.
  sorry

end complement_of_intersection_l7_7423


namespace multiplied_factor_number_count_200_to_300_l7_7666

def is_multiplied_factor_number (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 = 2 ∧ d3 ≠ 0 ∧ n % (d1 * d2 * d3) = 0

def count_multiplied_factor_numbers (start : ℕ) (end : ℕ) : ℕ :=
  (List.range' start (end - start + 1)).countp is_multiplied_factor_number

theorem multiplied_factor_number_count_200_to_300 : count_multiplied_factor_numbers 200 300 = 15 :=
  sorry

end multiplied_factor_number_count_200_to_300_l7_7666


namespace yoga_studio_women_count_l7_7188

theorem yoga_studio_women_count :
  ∃ W : ℕ, 
  (8 * 190) + (W * 120) = 14 * 160 ∧ W = 6 :=
by 
  existsi (6);
  sorry

end yoga_studio_women_count_l7_7188


namespace probability_of_selected_number_probability_of_selected_number_in_given_interval_l7_7863

noncomputable def probability_in_interval (a b c : ℝ) (hab : 0 < a) (hbc : b < c) : ℝ :=
  (b - a) / (c - a)

theorem probability_of_selected_number (x : ℝ) (hx : 0 < x) (hx_le : x < 1/2) : 
  probability_in_interval 0 (1/3) (1/2) 0 lt_one_half = 2/3 := 
by
  have p := probability_in_interval 0 (1/3) (1/2) 0 (by norm_num : 1 < 2)
  norm_num at p
  exact p

-- Helper theorem to convert the original question
theorem probability_of_selected_number_in_given_interval :
  ∀ x, 0 < x ∧ x < 1/2 → x < 1/3 → probability_of_selected_number x 0 (by norm_num) = 2/3 :=
by
  intros x _ _
  exact probability_of_selected_number x 0 (by norm_num)

-- Sorry to skip the proof as requested
sorry

end probability_of_selected_number_probability_of_selected_number_in_given_interval_l7_7863


namespace concrete_mixture_weight_l7_7647

theorem concrete_mixture_weight
  (x : ℕ)
  (H_ratio : x = 5)
  (cement_ratio sand_ratio earth_ratio : ℕ)
  (H_ratios : cement_ratio = 1 ∧ sand_ratio = 3 ∧ earth_ratio = 5):
  let sand := sand_ratio * x,
      earth := earth_ratio * x,
      total := x + sand + earth
  in total = 45 := by
  sorry

end concrete_mixture_weight_l7_7647


namespace min_interval_exists_l7_7785

-- Definitions
def f (x : ℝ) : ℝ := Real.exp x
def g (x : ℝ) : ℝ := Real.log x

-- Theorem statement
theorem min_interval_exists :
  ∃ (a : ℝ), (0 < a) ∧ (1 / 2 < a) ∧ (a < Real.log 2) ∧ 
  f (Real.log a) = g (Real.exp a) := sorry

end min_interval_exists_l7_7785


namespace infinite_divisors_l7_7055

theorem infinite_divisors (k : ℕ) (hk : 0 < k) :
  ∃ᶠ n in at_top, (n - k) ∣ (binom (2 * n) n) :=
sorry

end infinite_divisors_l7_7055


namespace construct_equilateral_triangle_l7_7835

-- Define the conditions
variables {Point Line : Type}
variables (XX' YY' : Line) 
variables (A : Point)
variables [parallel : ∀ (l₁ l₂ : Line), Prop] 

-- Introduce the parallel condition
axiom parallel_lines : parallel XX' YY'

-- State the theorem
theorem construct_equilateral_triangle (XX' YY' : Line) (A : Point) [parallel XX' YY'] :
  ∃ (B C : Point), 
  (B ∈ XX') ∧ (C ∈ YY') ∧ (∀ (A B C : Point), is_equilateral_triangle A B C) := sorry

end construct_equilateral_triangle_l7_7835


namespace estimate_red_balls_l7_7006

theorem estimate_red_balls (x : ℕ) (drawn_black_balls : ℕ) (total_draws : ℕ) (black_balls : ℕ) 
  (h1 : black_balls = 4) 
  (h2 : total_draws = 100) 
  (h3 : drawn_black_balls = 40) 
  (h4 : (black_balls : ℚ) / (black_balls + x) = drawn_black_balls / total_draws) : 
  x = 6 := 
sorry

end estimate_red_balls_l7_7006


namespace fourth_bell_interval_l7_7313

-- Define intervals of the first three bells
def interval_bell1 : ℕ := 5
def interval_bell2 : ℕ := 8
def interval_bell3 : ℕ := 11

-- Interval for all bells tolling together
def all_toll_together : ℕ := 1320

theorem fourth_bell_interval (interval_bell4 : ℕ) :
  Nat.gcd (Nat.lcm (Nat.lcm interval_bell1 interval_bell2) interval_bell3) interval_bell4 = 1 →
  Nat.lcm (Nat.lcm (Nat.lcm interval_bell1 interval_bell2) interval_bell3) interval_bell4 = all_toll_together →
  interval_bell4 = 1320 := sorry

end fourth_bell_interval_l7_7313


namespace intersection_P_Q_l7_7424

def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = -x + 2}

theorem intersection_P_Q : P ∩ Q = {y | y ≤ 2} :=
sorry

end intersection_P_Q_l7_7424


namespace probability_of_interval_l7_7868

noncomputable def probability_less_than (a b : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < b then (x - a) / (b - a) else 0

theorem probability_of_interval (a b x : ℝ) (ha : 0 < a) (hb : a < b) :
  probability_less_than 0 (1/2) (1/3) = 2/3 :=
by
  have h_interval : (1 : ℝ)/2 - 0 = (1/2) := by norm_num,
  have h_favorable : (1 : ℝ)/3 - 0 = (1/3) := by norm_num,
  rw [← h_interval, ← h_favorable, probability_less_than],
  split_ifs,
  simpa using div_eq_mul_one_div (1/3) (1/2),
  sorry

end probability_of_interval_l7_7868


namespace find_minimum_value_l7_7769

noncomputable def minimum_value (a : ℝ) : ℝ :=
  if a ≤ -1 then 2 * a + 3 
  else if a < 1 then -a^2 + 2 
  else -2 * a + 3

theorem find_minimum_value (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = 3 - 2 * a * sin x - cos x ^ 2) →
  ∃ ymin : ℝ, ymin = minimum_value a ∧ (∀ x : ℝ, f x ≥ ymin) :=
sorry

end find_minimum_value_l7_7769


namespace probability_three_Beliy_Naliv_l7_7312

/--
Given that there are 9 bushes of the "Beliy Naliv" variety and 7 bushes of the "Verlioka" variety, 
prove that the probability that the first three consecutively planted bushes are seedlings of the "Beliy Naliv" variety is 0.15.
-/
theorem probability_three_Beliy_Naliv (total_bushes : ℕ) (Beliy_Naliv_bushes Verlioka_bushes : ℕ)
  (h_total : total_bushes = Beliy_Naliv_bushes + Verlioka_bushes)
  (h_Beliy_Naliv : Beliy_Naliv_bushes = 9)
  (h_Verlioka : Verlioka_bushes = 7) :
  let p := (9 / 16) * (8 / 15) * (7 / 14) in
  p = 0.15 :=
by
  sorry

end probability_three_Beliy_Naliv_l7_7312


namespace set_different_l7_7335

-- Definitions of the sets ①, ②, ③, and ④
def set1 : Set ℤ := {x | x = 1}
def set2 : Set ℤ := {y | (y - 1)^2 = 0}
def set3 : Set ℤ := {x | x = 1}
def set4 : Set ℤ := {1}

-- Lean statement to prove that set3 is different from the others
theorem set_different : set3 ≠ set1 ∧ set3 ≠ set2 ∧ set3 ≠ set4 :=
by
  -- Skipping the proof with sorry
  sorry

end set_different_l7_7335


namespace geometry_problem_l7_7477

theorem geometry_problem
  (PU PV PS QS QT RT : Type)
  [Segment PU] [Segment PV] [Segment PS] [Segment QS] [Segment QT] [Segment RT]
  (U V P Q S T : Point)
  (H1 : IsOnLine U P Q S T)
  (H2 : PU = PV)
  (H3 : ∠UPV = 24)
  (H4 : ∠PSQ = x)
  (H5 : ∠TQS = y) :
  x + y = 78 :=
sorry

end geometry_problem_l7_7477


namespace volume_of_cylindrical_tin_l7_7264

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem volume_of_cylindrical_tin : 
  cylinder_volume 2 5 = 62.8318 := 
by
  sorry

end volume_of_cylindrical_tin_l7_7264


namespace capacity_of_initial_20_buckets_l7_7272

theorem capacity_of_initial_20_buckets (x : ℝ) (h : 20 * x = 270) : x = 13.5 :=
by 
  sorry

end capacity_of_initial_20_buckets_l7_7272


namespace shortest_path_ball_reflected_l7_7301

-- Define the parameters of the ellipse
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 9) = 1

-- Define the major axis a of the ellipse
def a : ℝ := 4

-- Define the distance to be proven
def shortest_distance : ℝ := 4 * a

-- A proof statement
theorem shortest_path_ball_reflected (x y : ℝ) (hx : ellipse_eq x y) : shortest_distance = 16 := by
  unfold shortest_distance
  unfold a
  rw [mul_assoc, mul_one, mul_comm, mul_assoc 4 4, ←mul_assoc, ←mul_one]
  norm_num

end shortest_path_ball_reflected_l7_7301


namespace cake_eating_classmates_l7_7113

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l7_7113


namespace initial_investment_l7_7686

theorem initial_investment :
  ∃ (P : ℝ), 
    let r := 0.10 in
    let n := 2 in
    let t := 1 in
    let A := 882 in
    A = P * (1 + r / n) ^ (n * t) ∧ P = 800 :=
sorry

end initial_investment_l7_7686


namespace semicircle_ratio_l7_7894

variable (r : ℝ)

-- Conditions
def circle_radius (O : ℝ) : ℝ := 2 * r
def semicircle_radius_1 (POQ : ℝ) : ℝ := r
def semicircle_radius_2 (ROS : ℝ) : ℝ := r
def diametrically_opposite (PQ RS : ℝ) := true  -- Placeholder, since this condition is more geometric.

-- Question in Lean 4
theorem semicircle_ratio (r : ℝ) : 
  let area_circle := Real.pi * (2 * r) ^ 2,
      area_semicircle := (1 / 2) * Real.pi * r^2,
      combined_area_semicircles := 2 * ((1 / 2) * Real.pi * r^2),
      non_overlapping_area := combined_area_semicircles,
      ratio := non_overlapping_area / area_circle
  in ratio = (1/4) :=
by
  let area_circle := Real.pi * (2 * r) ^ 2
  let area_semicircle := (1 / 2) * Real.pi * r^2
  let combined_area_semicircles := 2 * ((1 / 2) * Real.pi * r^2)
  let non_overlapping_area := combined_area_semicircles
  let ratio := non_overlapping_area / area_circle
  have h : ratio = (1/4) := sorry
  exact h

end semicircle_ratio_l7_7894


namespace gcd_8994_13326_37566_l7_7767

-- Define the integers involved
def a := 8994
def b := 13326
def c := 37566

-- Assert the GCD relation
theorem gcd_8994_13326_37566 : Int.gcd a (Int.gcd b c) = 2 := by
  sorry

end gcd_8994_13326_37566_l7_7767


namespace distance_problem_l7_7738

noncomputable def distance_point_to_plane 
  (x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ) : ℝ :=
  -- Equation of the plane passing through three points derived using determinants
  let a := x2 - x1
  let b := y2 - y1
  let c := z2 - z1
  let d := x3 - x1
  let e := y3 - y1
  let f := z3 - z1
  let A := b*f - c*e
  let B := c*d - a*f
  let C := a*e - b*d
  let D := -(A*x1 + B*y1 + C*z1)
  -- Distance from the given point to the above plane
  (|A*x0 + B*y0 + C*z0 + D|) / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_problem :
  distance_point_to_plane 
  3 6 68 
  (-3) (-5) 6 
  2 1 (-4) 
  0 (-3) (-1) 
  = Real.sqrt 573 :=
by sorry

end distance_problem_l7_7738


namespace initial_candy_count_l7_7488

theorem initial_candy_count (N : ℕ) (sold_mon : ℕ) (sold_tue : ℕ) (left_wed : ℕ) 
  (h1 : sold_mon = 15) (h2 : sold_tue = 58) (h3 : left_wed = 7) (h4 : N - sold_mon - sold_tue = left_wed) : 
  N = 80 :=
by { rw [h1, h2, h3] at h4, linarith }

#check initial_candy_count

end initial_candy_count_l7_7488


namespace domain_range_0_1_domain_range_1_4_2_l7_7403

/-
  Given the range of the function y = x^2 - 2x + 2 is [1,2], prove that the possible domains of the function 
  that result in this range include [0,1] and [1/4, 2].
-/
def quadratic_function (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem domain_range_0_1 : 
  ∃ D : set ℝ, D = set.Icc 0 1 ∧ (∀ x ∈ D, 1 ≤ quadratic_function x ∧ quadratic_function x ≤ 2) :=
sorry

theorem domain_range_1_4_2 : 
  ∃ D : set ℝ, D = set.Icc (1/4 : ℝ) 2 ∧ (∀ x ∈ D, 1 ≤ quadratic_function x ∧ quadratic_function x ≤ 2) :=
sorry

end domain_range_0_1_domain_range_1_4_2_l7_7403


namespace simplify_sqrt_245_l7_7150

-- Define the necessary conditions
def is_product_of_5_and_49 : Prop := 245 = 5 * 49
def square_root_multiplication (a b : ℝ) : Prop := sqrt (a * b) = sqrt a * sqrt b
def sqrt_of_49_is_7 : Prop := sqrt 49 = 7

-- Theorem statement
theorem simplify_sqrt_245 (h1 : is_product_of_5_and_49) (h2 : square_root_multiplication 49 5) (h3 : sqrt_of_49_is_7) : sqrt 245 = 7 * sqrt 5 :=
sorry

end simplify_sqrt_245_l7_7150


namespace max_value_f_l7_7173

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem max_value_f : ∃ x ∈ (Set.Icc 0 (Real.pi / 2)), f x = Real.pi / 12 + Real.sqrt 3 / 2 :=
by
  sorry

end max_value_f_l7_7173


namespace fraction_for_repeating_decimal_l7_7761

variable (a r S : ℚ)
variable (h1 : a = 3/5)
variable (h2 : r = 1/10)
variable (h3 : S = a / (1 - r))

theorem fraction_for_repeating_decimal : S = 2 / 3 :=
by
  have h4 : 1 - r = 9 / 10, from sorry
  have h5 : S = (3 / 5) / (9 / 10), from sorry
  have h6 : S = (3 * 10) / (5 * 9), from sorry
  have h7 : S = 30 / 45, from sorry
  have h8 : 30 / 45 = 2 / 3, from sorry
  exact h8

end fraction_for_repeating_decimal_l7_7761


namespace jackson_grade_increase_per_hour_l7_7919

-- Define the necessary variables
variables (v s p G : ℕ)

-- The conditions from the problem
def study_condition1 : v = 9 := sorry
def study_condition2 : s = v / 3 := sorry
def grade_starts_at_zero : G = s * p := sorry
def final_grade : G = 45 := sorry

-- The final problem statement to prove
theorem jackson_grade_increase_per_hour :
  p = 15 :=
by
  -- Add our sorry to indicate the partial proof
  sorry

end jackson_grade_increase_per_hour_l7_7919


namespace cake_sharing_l7_7103

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l7_7103


namespace competition_problem_l7_7547

def x1 : Nat -- declaration for the variables.
def x2 : Nat
def x3 : Nat

theorem competition_problem :
  (0 < x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ 90 ∧ x1 + x2 + x3 = 180) →
  ((∃ n p, n / p = 0.005745) :=
by
  sorry -- Proof omitted

end competition_problem_l7_7547


namespace company_budget_salaries_degrees_l7_7274

theorem company_budget_salaries_degrees :
  let transportation := 0.20
  let research_and_development := 0.09
  let utilities := 0.05
  let equipment := 0.04
  let supplies := 0.02
  let total_budget := 1.0
  let total_percentage := transportation + research_and_development + utilities + equipment + supplies
  let salaries_percentage := total_budget - total_percentage
  let total_degrees := 360.0
  let degrees_salaries := salaries_percentage * total_degrees
  degrees_salaries = 216 :=
by
  sorry

end company_budget_salaries_degrees_l7_7274


namespace probability_product_is_multiple_of_105_l7_7457

-- Define the set of numbers
def numbers_set : Finset ℕ := {3, 10, 15, 21, 35, 45, 70}

-- Define what it means for the product to be a multiple of 105
def is_multiple_of_105 (a b : ℕ) : Prop := (a * b) % 105 = 0

-- Define the proof statement
theorem probability_product_is_multiple_of_105 :
  let pairs := (numbers_set.product numbers_set).filter (λ p => p.1 < p.2)
  let successful := pairs.filter (λ p => is_multiple_of_105 p.1 p.2)
  (successful.card : ℚ) / pairs.card = 2 / 7 := sorry

end probability_product_is_multiple_of_105_l7_7457


namespace lcm_1_to_12_l7_7241

theorem lcm_1_to_12 : Nat.lcm_list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 27720 := by
  sorry

end lcm_1_to_12_l7_7241


namespace quadrilateral_on_triangle_has_small_triangle_area_l7_7382

variables {α : Type*} [linear_ordered_field α] {A B C P1 P2 P3 P4 : Point α}

structure Point (α : Type*) :=
(x : α)
(y : α)

noncomputable def area (A B C : Point α) : α := 
  abs ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)) / 2

theorem quadrilateral_on_triangle_has_small_triangle_area
  (hP1 : lies_on_boundary P1 (triangle A B C))
  (hP2 : lies_on_boundary P2 (triangle A B C))
  (hP3 : lies_on_boundary P3 (triangle A B C))
  (hP4 : lies_on_boundary P4 (triangle A B C)) :
  min (min (area P1 P2 P3) (area P1 P2 P4)) (min (area P1 P3 P4) (area P2 P3 P4)) ≤ (area A B C) / 4 :=
sorry

end quadrilateral_on_triangle_has_small_triangle_area_l7_7382


namespace percent_profit_is_25_percent_l7_7875

theorem percent_profit_is_25_percent
  (CP SP : ℝ)
  (h : 75 * (CP - 0.05 * CP) = 60 * SP) :
  let profit := SP - (0.95 * CP)
  let percent_profit := (profit / (0.95 * CP)) * 100
  percent_profit = 25 :=
by
  sorry

end percent_profit_is_25_percent_l7_7875


namespace calculate_arithmetic_sequence_sum_l7_7472

noncomputable def arithmetic_sequence_sum_condition (a : ℕ → ℤ) (d : ℤ) (h_d_pos : 0 < d) :=
  (2 * a 7 - a 13 = 1) ∧ ((a 3 - 1) ^ 2 = a 1 * (a 6 + 5))

theorem calculate_arithmetic_sequence_sum :
  ∃ a : ℕ → ℤ, ∃ d : ℤ, 
    0 < d ∧ (
      -- Conditions for arithmetic sequence and geometric progression
      arithmetic_sequence_sum_condition a d (by linarith) 
      ∧ -- Goal: Sum of the first 21 terms of the sequence {(-1)^{n-1} a_n} is 21
      (finset.sum (finset.range 21) (λ n, (-1 : ℤ) ^ n * a (n + 1)) = 21)
    ) :=
sorry

end calculate_arithmetic_sequence_sum_l7_7472


namespace number_of_classmates_l7_7118

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l7_7118


namespace repeating_six_as_fraction_l7_7743

theorem repeating_six_as_fraction : (∑' n : ℕ, 6 / (10 * (10 : ℝ)^n)) = (2 / 3) :=
by
  sorry

end repeating_six_as_fraction_l7_7743


namespace min_value_of_M_l7_7991

noncomputable def F (x : ℝ) (A B : ℝ) : ℂ := 
  complex.I * real.cos x ^ 2 + 2 * real.sin x * real.cos x - real.sin x ^ 2 + A * x + B

def interval := set.Icc 0 (3 / 2 * real.pi)

def M (A B : ℝ) : ℂ := 
  real.Sup (set.image (λ x, F x A B) interval)

theorem min_value_of_M : ∀ A B : ℝ, 
  (∃ x ∈ interval, F x A B = M A B) → ∃ A B : ℝ, M A B = real.sqrt 2 :=
sorry

end min_value_of_M_l7_7991


namespace fraction_of_water_in_mixture_l7_7896

theorem fraction_of_water_in_mixture (r : ℚ) (h : r = 2 / 3) : (3 / (2 + 3) : ℚ) = 3 / 5 :=
by
  sorry

end fraction_of_water_in_mixture_l7_7896


namespace isosceles_triangle_perimeter_l7_7481

theorem isosceles_triangle_perimeter (base side : ℕ) (h_base : base = 10) (h_side : side = 7) : 
  2 * side + base = 24 := by
  -- Given conditions
  have h_perimeter : 2 * side + base = 2 * 7 + 10 := by
    rw [h_base, h_side]
  -- Simplify to get the final answer
  exact calc
    2 * 7 + 10 = 14 + 10 : by rfl
    ... = 24 : by rfl

end isosceles_triangle_perimeter_l7_7481


namespace repeating_six_to_fraction_l7_7760

-- Define the infinite geometric series representation of 0.666...
def infinite_geometric_series (n : ℕ) : ℝ := 6 / (10 ^ n)

-- Define the sum of the infinite geometric series for 0.666...
def sum_infinite_geometric_series : ℝ :=
  ∑' n, infinite_geometric_series n

-- Formally state the problem to prove that 0.666... equals 2/3
theorem repeating_six_to_fraction : sum_infinite_geometric_series = 2 / 3 :=
by
  -- Proof goes here, but for now we use sorry to denote it will be completed later
  sorry

end repeating_six_to_fraction_l7_7760


namespace num_odd_three_digit_numbers_l7_7598

theorem num_odd_three_digit_numbers : 
    let digits := {1, 2, 3, 4, 5}
    let hundred_choices := {d ∈ digits | d < 4}
    let ten_choices := digits
    let unit_choices := {d ∈ digits | d % 2 = 1}
    #[
    (set.size hundred_choices) * (set.size ten_choices) * (set.size unit_choices) = 45
    #.

end num_odd_three_digit_numbers_l7_7598


namespace solution_l7_7946

open Real

def problem (f : ℝ → ℝ) : Prop :=
  (∃ m, f = λ x, m * sin x + cos x ∧ f (π / 2) = 1) ∧
  (∃ T, ∀ x, f (x + T) = f x) ∧
  (∀ x, ∃ a b, f x = a * sin (x + b) ∧ f (x + π/2) = sqrt 2) ∧
  (∃ A, ∃ ABC : Triangle, ABC.is_acute ∧ ABC.area = (3 * sqrt3) / 2 ∧ 
   ABC.AB = 2 ∧ ∃ AC BC, f (π / 12) = sqrt 2 * sin A ∧ AC = 3 ∧ BC = sqrt 7)

theorem solution : problem (λ x, sin x + cos x) :=
sorry

end solution_l7_7946


namespace joe_trip_time_l7_7491

/-
  Joe walked one-third of the way from home to school,
  ran the rest of the way, and ran 4 times as fast as he walked.
  Joe took 9 minutes to walk the one-third distance to school.
  Prove that Joe took 13.5 minutes to get from home to school.
-/

def joe_trip_duration := 13.5

theorem joe_trip_time
  (d : ℝ) -- Total distance from home to school
  (rw : ℝ) -- Joe's walking rate (speed)
  (rr : ℝ) -- Joe's running rate (speed)
  (tw : ℝ) -- Time Joe walked initially (9 minutes)
  (tr : ℝ) -- Time Joe ran the rest of the way
  (h1 : tw = 9) -- Joe walked the initial 9 minutes
  (h2 : rr = 4 * rw) -- Running rate is 4 times walking rate
  (h3 : rw * tw = d / 3) -- Distance for walking
  (h4 : rr * tr = 2 * d / 3) -- Distance for running
  : tw + tr = joe_trip_duration := 
by
  sorry

end joe_trip_time_l7_7491


namespace operation_result_l7_7458

-- Define x and the operations
def x : ℕ := 40

-- Define the operation sequence
def operation (y : ℕ) : ℕ :=
  let step1 := y / 4
  let step2 := step1 * 5
  let step3 := step2 + 10
  let step4 := step3 - 12
  step4

-- The statement we need to prove
theorem operation_result : operation x = 48 := by
  sorry

end operation_result_l7_7458


namespace statement_B_is_incorrect_l7_7784

variables {α β : Type} [Plane α] [Plane β] [Different α β]
variables {m n : Type} [Line m] [Line n] [Different m n]
variables [Parallel m α] [Intersection a β n]

theorem statement_B_is_incorrect : ¬ (Parallel m n) :=
sorry

end statement_B_is_incorrect_l7_7784


namespace solution_exists_l7_7557

def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

def f' (x a b : ℝ) : ℝ := 3 * x^2 - 2 * a * x - b

theorem solution_exists (a b : ℝ) :
    f 1 a b = 10 ∧ f' 1 a b = 0 ↔ (a = -4 ∧ b = 11) :=
by 
  sorry

end solution_exists_l7_7557


namespace possible_number_of_classmates_l7_7126

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l7_7126


namespace S_2006_is_rhombus_not_rectangle_l7_7793

-- Define the quadrilateral type
structure Quadrilateral :=
  (a b c d : ℝ)

-- Initial condition: S1 has equal but not perpendicular diagonals
def S1 : Quadrilateral := sorry

-- Function to generate the next quadrilateral by connecting midpoints
def next_quadrilateral (S : Quadrilateral) : Quadrilateral := sorry

-- Definition of S_2006
noncomputable def S_2006 : Quadrilateral :=
  Nat.iterate next_quadrilateral 2005 S1

-- Theorem statement
theorem S_2006_is_rhombus_not_rectangle : (is_rhombus S_2006) ∧ ¬ (is_rectangle S_2006) :=
  sorry

-- Auxiliary definitions to determine if a quadrilateral is a rhombus or rectangle
def is_rhombus (S : Quadrilateral) : Prop := sorry
def is_rectangle (S : Quadrilateral) : Prop := sorry

end S_2006_is_rhombus_not_rectangle_l7_7793


namespace proof_smallest_lcm_1_to_12_l7_7226

noncomputable def smallest_lcm_1_to_12 : ℕ := 27720

theorem proof_smallest_lcm_1_to_12 :
  ∀ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ↔ n = 27720 :=
by
  sorry

end proof_smallest_lcm_1_to_12_l7_7226


namespace ages_of_sons_l7_7654

variable (x y z : ℕ)

def father_age_current : ℕ := 33

def youngest_age_current : ℕ := 2

def father_age_in_12_years : ℕ := father_age_current + 12

def sum_of_ages_in_12_years : ℕ := youngest_age_current + 12 + y + 12 + z + 12

theorem ages_of_sons (x y z : ℕ) 
  (h1 : x = 2)
  (h2 : father_age_current = 33)
  (h3 : father_age_in_12_years = 45)
  (h4 : sum_of_ages_in_12_years = 45) :
  x = 2 ∧ y + z = 7 ∧ ((y = 3 ∧ z = 4) ∨ (y = 4 ∧ z = 3)) :=
by
  sorry

end ages_of_sons_l7_7654


namespace sample_size_for_probability_l7_7277

noncomputable def sample_size_needed 
    (p : ℝ) (ε : ℝ) (P : ℝ) (t : ℝ) : ℕ :=
  let pq := p * (1 - p) in
  Nat.ceil (pq * (t / ε) ^ 2)

theorem sample_size_for_probability 
    : ∀ (p ε P : ℝ), p = 0.85 → ε = 0.01 → P = 0.997 →
      let t := 3 in 
      sample_size_needed p ε P t = 11475 := 
by
  intros p ε P hp hε hP t ht
  have pq := p * (1 - p)
  rw [hp, hε, hP, ht]
  have : pq = 0.85 * 0.15 := by norm_num
  rw [this]
  simp [sample_size_needed]
  sorry

end sample_size_for_probability_l7_7277


namespace lcm_from_1_to_12_eq_27720_l7_7235

theorem lcm_from_1_to_12_eq_27720 : nat.lcm (finset.range 12).succ = 27720 :=
  sorry

end lcm_from_1_to_12_eq_27720_l7_7235


namespace terminal_side_of_half_θ_l7_7446

-- Define the conditions given in the problem
variables (θ : Real) (cos_θ : Real) (tan_θ : Real)
-- Assume the conditions
hypothesis hcos : cos_θ = |cos θ|
hypothesis htan : tan_θ = |-tan θ|

-- Define the requirement to prove
theorem terminal_side_of_half_θ (hcos : |cos θ| = cos θ) (htan : |tan θ| = -tan θ) :
  (∃ k : Int, k * 360 + 135 < θ / 2 ∧ θ / 2 ≤ k * 360 + 180) ∨ 
  (∃ k : Int, k * 360 + 315 < θ / 2 ∧ θ / 2 ≤ k * 360 + 360) :=
sorry

end terminal_side_of_half_θ_l7_7446


namespace find_principal_amount_l7_7633

def si (P : ℝ) (R : ℝ) (T : ℝ) := P * R * T / 100
def ci (P : ℝ) (R : ℝ) (T : ℝ) := P * (1 + R / 100) ^ T - P

theorem find_principal_amount :
  ∃ (P : ℝ), ci P 5 2 - si P 5 2 = 15 ∧ P = 6000 :=
by
  use 6000
  have si_val : si 6000 5 2 = 6000 * 5 * 2 / 100 := by sorry
  have ci_val : ci 6000 5 2 = 6000 * (1 + 5 / 100) ^ 2 - 6000 := by sorry
  simp only [si, ci] at *
  rw [si_val, ci_val]
  sorry

end find_principal_amount_l7_7633


namespace ratio_of_products_l7_7854

variable (a b c d : ℚ) -- assuming a, b, c, d are rational numbers

theorem ratio_of_products (h1 : a = 3 * b) (h2 : b = 2 * c) (h3 : c = 5 * d) :
  a * c / (b * d) = 15 := by
  sorry

end ratio_of_products_l7_7854


namespace original_cost_l7_7186

theorem original_cost (original_cost : ℝ) (h : 0.30 * original_cost = 588) : original_cost = 1960 :=
sorry

end original_cost_l7_7186


namespace length_of_QR_l7_7487

theorem length_of_QR (P Q R N : Type) [InnerProductSpace ℝ (P × Q × R)]
  (PQ PR PN QR : ℝ)
  (h1 : PQ = 5)
  (h2 : PR = 12)
  (h3 : PN = 6)
  (h4 : N = (Q + R) / 2)
  : QR = 14 := by 
  sorry

end length_of_QR_l7_7487


namespace root_product_minus_sums_l7_7496

variable {b c : ℝ}

theorem root_product_minus_sums
  (h1 : 3 * b^2 + 5 * b - 2 = 0)
  (h2 : 3 * c^2 + 5 * c - 2 = 0)
  : (b - 1) * (c - 1) = 2 := 
by
  sorry

end root_product_minus_sums_l7_7496


namespace monotonic_increasing_interval_ln_1_minus_x_sq_l7_7997

-- Define the function
def f (x : ℝ) : ℝ := Real.log (1 - x^2)

-- Define the proof problem
theorem monotonic_increasing_interval_ln_1_minus_x_sq :
  ∀ x y, (-1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ x < y) → f x < f y := 
by
  -- The actual proof goes here
  sorry

end monotonic_increasing_interval_ln_1_minus_x_sq_l7_7997


namespace bridge_length_is_240_l7_7627

noncomputable def length_of_bridge 
  (train_length : ℕ) 
  (train_speed_kmh : ℕ) 
  (crossing_time_s : ℕ) : ℕ :=
let train_speed_ms := (train_speed_kmh * 1000) / 3600 in
let total_distance := train_speed_ms * crossing_time_s in
total_distance - train_length

theorem bridge_length_is_240 :
  length_of_bridge 135 45 30 = 240 := 
by
  unfold length_of_bridge
  -- conversion and calculation steps can be assumed correct.
  sorry

end bridge_length_is_240_l7_7627


namespace find_M_plus_10m_l7_7081

variable {x y z : ℝ}

theorem find_M_plus_10m (h : 8 * (x + y + z) = x^2 + y^2 + z^2) : 
  let M := max (xy + xz + yz) 
  let m := min (xy + xz + yz) 
  M + 10 * m = 144 :=
by 
    -- We state the assumptions and definitions
    sorry -- actual proof steps to be filled in

end find_M_plus_10m_l7_7081


namespace general_term_arithmetic_sequence_inequality_l7_7795

-- Define the sequences {a_n} and {b_n}
def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * a (n - 1) + 1

-- Define the sequence {b_n} satisfying the condition
def b (n : ℕ) : ℕ -- b_n follows an arithmetic sequence b_n = b_1 + (n-1)d, needs to be proven
  -- Assuming some initial value b_1 and difference d
  -- This is not needed for the statement, proof will handle the definition
  := sorry

-- State the general term formula theorem
theorem general_term (n : ℕ) (h : n > 0) : a n = 2^n - 1 :=
by {
  sorry
}

-- State the arithmetic sequence theorem
theorem arithmetic_sequence (n : ℕ) : 4 ^ (b 1 - 1) * 4 ^ (b 2 - 1) * ... * 4 ^ (b n - 1) = (a n + 1) ^ b n → 
                                      ∃ d, ∀ m, b (m + 1) - b m = d :=
by {
  sorry
}

-- State the inequality involving a_n
theorem inequality (n : ℕ) (h : n > 0) : 
  (n / 2 - 1 / 3 : ℝ) < ∑ i in finset.range (n + 1), (a i / a (i + 1) : ℝ) ∧
  (∑ i in finset.range (n + 1), (a i / a (i + 1) : ℝ)) < n / 2 := 
by {
  sorry
}

end general_term_arithmetic_sequence_inequality_l7_7795


namespace solve_quartic_l7_7736

theorem solve_quartic (x : ℝ) (a b : ℝ)
    (h₁ : a = real.sqrt (real.sqrt (63 - 3 * x)))
    (h₂ : b = real.sqrt (real.sqrt (27 + 3 * x)))
    (h₃ : a + b = 5) :
    x = 5 := 
  sorry

end solve_quartic_l7_7736


namespace min_value_of_y_l7_7396

variable {x k : ℝ}

theorem min_value_of_y (h₁ : ∀ x > 0, 0 < k) 
  (h₂ : ∀ x > 0, (x^2 + k / x) ≥ 3) : k = 2 :=
sorry

end min_value_of_y_l7_7396


namespace mike_total_hours_l7_7517

theorem mike_total_hours (hours_per_day : ℕ) (days : ℕ) (total_hours : ℕ) : 
  hours_per_day = 3 → days = 5 → total_hours = hours_per_day * days → total_hours = 15 :=
by 
  intros hpd hdays hth
  rw [hpd, hdays] at hth
  exact hth

#eval mike_total_hours 3 5 15 sorry sorry sorry

end mike_total_hours_l7_7517


namespace sqrt_108_eq_6_sqrt_3_l7_7151

theorem sqrt_108_eq_6_sqrt_3 : Real.sqrt 108 = 6 * Real.sqrt 3 := 
sorry

end sqrt_108_eq_6_sqrt_3_l7_7151


namespace positional_relationship_uncertain_l7_7269

variable {l1 l2 l3 l4 : Type}

axiom four_distinct_lines : ∀ (l1 l2 l3 l4 : Type), Prop
axiom perp : ∀ (l m : Type), Prop

-- Conditions
axiom h1 : four_distinct_lines l1 l2 l3 l4
axiom h2 : perp l1 l2
axiom h3 : perp l2 l3
axiom h4 : perp l3 l4

-- Proof problem
theorem positional_relationship_uncertain : ∀ {l1 l4 : Type}, four_distinct_lines l1 l2 l3 l4 → perp l1 l2 → perp l2 l3 → perp l3 l4 → ∃ l : Type, ¬ (perp l1 l4 ∨ parallel l1 l4) :=
begin
  sorry
end

end positional_relationship_uncertain_l7_7269


namespace exists_100_numbers_with_property_l7_7916

theorem exists_100_numbers_with_property :
  ∃ (n : Fin 100 → ℕ), (∀ (a b c d e : Fin 100),
    a ≠ b → a ≠ c → a ≠ d → a ≠ e →
    b ≠ c → b ≠ d → b ≠ e →
    c ≠ d → c ≠ e →
    d ≠ e →
    let sum := n a + n b + n c + n d + n e,
        prod := n a * n b * n c * n d * n e
    in sum ∣ prod) := sorry

end exists_100_numbers_with_property_l7_7916


namespace problem_statement_l7_7806

noncomputable def is_interval_strictly_increasing (k : ℤ) : Prop :=
  let f (x : ℝ) := Real.sin (π / 4 + x / 2) * Real.cos (π / 4 + x / 2) in
  ∀ x y : ℝ, (2 * k + 1) * π ≤ x ∧ x < y ∧ y ≤ 2 * (k + 1) * π → f x < f y

theorem problem_statement (k : ℤ) : is_interval_strictly_increasing k :=
  sorry

end problem_statement_l7_7806


namespace probability_of_less_than_one_third_l7_7860

theorem probability_of_less_than_one_third :
  (prob_of_interval (0 : ℝ) (1 / 2 : ℝ) (1 / 3 : ℝ) = 2 / 3) :=
sorry

end probability_of_less_than_one_third_l7_7860


namespace quadratic_eq_real_equal_roots_l7_7337

theorem quadratic_eq_real_equal_roots (m : ℝ) :
    (∀ x : ℝ, 3 * x^2 - (m + 6) * x + 18 = 0) ∧
    ((m + 6)^2 - 216 = 0) ∧
    (m + 6) / 3 < -2 →
    m = -6 - 6 * real.sqrt(6) :=
by
  sorry

end quadratic_eq_real_equal_roots_l7_7337


namespace sin_alpha_minus_pi_over_3_l7_7780

theorem sin_alpha_minus_pi_over_3 (α : ℝ) (h1 : -π/2 < α ∧ α < 0) (h2 : 2 * tan α * sin α = 3) :
  sin (α - π / 3) = - sqrt 3 / 2 :=
by sorry

end sin_alpha_minus_pi_over_3_l7_7780


namespace find_angle_B_find_area_triangle_l7_7886

-- Problem 1: Find angle B
theorem find_angle_B
  (a b c : ℝ)
  (h1 : (a + c)^2 - b^2 = 3 * a * c) : 
  ∠ B = π / 3 :=
sorry

-- Problem 2: Find the area of triangle ABC
theorem find_area_triangle
  (a b c : ℝ)
  (h1 : b = 6)
  (h2 : sin C = 2 * sin A)
  (h3 : (a + c)^2 - b^2 = 3 * a * c) : 
  ((1 / 2) * a * b * sin C) = 6 * sqrt 3 :=
sorry

end find_angle_B_find_area_triangle_l7_7886


namespace cake_sharing_l7_7104

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l7_7104


namespace percentage_gain_is_correct_l7_7287

-- Define the given conditions
def bowls_bought := 118
def cost_per_bowl := 12
def bowls_sold := 102
def selling_price_per_bowl := 15

-- Define the calculations based on conditions
def total_cost := cost_per_bowl * bowls_bought
def total_selling_price := selling_price_per_bowl * bowls_sold
def gain := total_selling_price - total_cost
def percentage_gain := (gain.toRat / total_cost) * 100

-- Proof statement
theorem percentage_gain_is_correct : percentage_gain = 8.05 := 
sorry

end percentage_gain_is_correct_l7_7287


namespace classmates_ate_cake_l7_7092

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l7_7092


namespace cistern_empty_time_l7_7625

theorem cistern_empty_time
  (fill_time_without_leak : ℝ := 4)
  (additional_time_due_to_leak : ℝ := 2) :
  (1 / (fill_time_without_leak + additional_time_due_to_leak - fill_time_without_leak / fill_time_without_leak)) = 12 :=
by
  sorry

end cistern_empty_time_l7_7625


namespace correct_option_C_l7_7251

noncomputable def question := "Which of the following operations is correct?"
noncomputable def option_A := (-2)^2
noncomputable def option_B := (-2)^3
noncomputable def option_C := (-1/2)^3
noncomputable def option_D := (-7/3)^3
noncomputable def correct_answer := -1/8

theorem correct_option_C :
  option_C = correct_answer := by
  sorry

end correct_option_C_l7_7251


namespace possible_number_of_classmates_l7_7127

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l7_7127


namespace geometric_sequence_third_term_find_geometric_sequence_third_term_l7_7909

-- Define the geometric sequence
noncomputable def a : ℕ → ℤ
| 1     := -2
| 5     := -8
| n     := sorry  -- For the other terms, we haven't defined explicitly.

-- The property of a geometric sequence: any term squared is the product of equidistant terms
theorem geometric_sequence_third_term :
  (a 3) ^ 2 = (a 1) * (a 5) :=
sorry

-- The final answer based on the geometric sequence properties
theorem find_geometric_sequence_third_term :
  a 3 = -4 :=
sorry

end geometric_sequence_third_term_find_geometric_sequence_third_term_l7_7909


namespace a_n_nonzero_l7_7377

/-- Recurrence relation for the sequence a_n --/
def a : ℕ → ℤ
| 0 => 1
| 1 => 2
| (n + 2) => if (a n * a (n + 1)) % 2 = 1 then 5 * a (n + 1) - 3 * a n else a (n + 1) - a n

/-- Proof that for all n, a_n is non-zero --/
theorem a_n_nonzero : ∀ n : ℕ, a n ≠ 0 := 
sorry

end a_n_nonzero_l7_7377


namespace sid_spent_on_computer_accessories_l7_7541

theorem sid_spent_on_computer_accessories :
  ∃ (x : ℝ), let original_money := 48
             let spent_on_snacks := 8
             let half_of_original := original_money / 2
             let amount_left := half_of_original + 4
             let total_spent := original_money - amount_left
             total_spent = spent_on_snacks + x ∧ x = 12 :=
begin
  sorry
end

end sid_spent_on_computer_accessories_l7_7541


namespace translated_properties_l7_7194

def f (x : ℝ) : ℝ := sqrt 2 * sin (2 * x) - sqrt 2 * cos (2 * x) + 1
def g (x : ℝ) : ℝ := 2 * sin (2 * x + π / 4)

theorem translated_properties :
  (∀ x, (g (x + π / 4) - 1 = f x)) ∧
  (∀ T, (T = π) ↔ (∃ x, g (x + T) = g x)) ∧
  (g (π / 8) = g (π / 8)) ∧
  (∫ x in 0..(π / 2), g x = sqrt 2) ∧
  (¬ (∀ x ∈ Icc (π / 12) (5 * π / 8), ∀ y ∈ Icc (π / 12) (5 * π / 8), x ≤ y → g x ≥ g y)) :=
by
  -- Translate f(x) to g(x), check period, symmetry, integral, and monotonicity properties
  sorry

end translated_properties_l7_7194


namespace length_AB_l7_7021

def polarLength (θ : ℝ) (ρ1 ρ2 : ℝ) : ℝ :=
  |ρ1 - ρ2|

theorem length_AB :
  let θ := 2 * Real.pi / 3 in
  let ρ1 := 4 * Real.sin θ in
  let ρ2 := 2 * Real.sin θ in
  polarLength θ ρ1 ρ2 = sqrt 3 :=
  by
    sorry

end length_AB_l7_7021


namespace value_of_expression_l7_7443

theorem value_of_expression (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  simp
  rfl

end value_of_expression_l7_7443


namespace function_properties_l7_7945

open Real

noncomputable def f (x : ℝ) : ℝ := log (2 + x) + log (2 - x)

theorem function_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ ⦃a b : ℝ⦄, 0 < a → a < b → b < 2 → f b < f a) := by
  sorry

end function_properties_l7_7945


namespace possible_classmates_l7_7132

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l7_7132


namespace angle_between_vectors_l7_7933

variable 
  (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 2)
  (hperp : inner (a + b) a = 0)

theorem angle_between_vectors :
  real.angle (a : ℝ^3) (b : ℝ^3) = real.pi * (2 / 3) :=
by 
  sorry

end angle_between_vectors_l7_7933


namespace smallest_positive_integer_a_not_prime_l7_7771

theorem smallest_positive_integer_a_not_prime:
  ∃ (a: ℕ), 
  (∀ (x: ℤ), 1 ≤ a ∧ (x^4 + (a + 4)^2) ∉ prime) ∧ 
  (∀ b: ℕ, b < a → ∃ c: ℤ, c^4 + (b + 4)^2 ∈ prime) := by
  sorry

end smallest_positive_integer_a_not_prime_l7_7771


namespace even_increasing_decreasing_l7_7170

theorem even_increasing_decreasing (f : ℝ → ℝ) (h_def : ∀ x : ℝ, f x = -x^2) :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, x < 0 → f x < f (x + 1)) ∧ (∀ x : ℝ, x > 0 → f x > f (x + 1)) :=
by
  sorry

end even_increasing_decreasing_l7_7170


namespace trig_sum_identity_l7_7732

theorem trig_sum_identity :
  Real.sin (47 * Real.pi / 180) * Real.cos (43 * Real.pi / 180) 
  + Real.sin (137 * Real.pi / 180) * Real.sin (43 * Real.pi / 180) = 1 :=
by
  sorry

end trig_sum_identity_l7_7732


namespace square_properties_l7_7673

noncomputable def side_length : ℝ := 24 / 7

theorem square_properties :
  let s := side_length in
  let area := s^2 in
  let diagonal := s * real.sqrt 2 in
  let volume := s^3 in
  let angle := 45 in
  area = 576 / 49 ∧
  diagonal = 24 / 7 * real.sqrt 2 ∧
  volume = 13824 / 343 ∧
  angle = 45 :=
by
  sorry

end square_properties_l7_7673


namespace analytical_expression_g_max_min_values_g_l7_7558

theorem analytical_expression_g (ω φ : ℝ) (h₁ : ω > 0) (h₂ : |φ| < π / 2)
  (h₃ : ∀ x ∈ Icc (5*π/12) (11*π/12), deriv (λ x, sin (ω * x + φ)) x < 0) :
  g(x) = sin (4*x + π/6) :=
sorry

theorem max_min_values_g :
  max_value_on_interval = 1 ∧ min_value_on_interval = -1/2 :=
sorry

end analytical_expression_g_max_min_values_g_l7_7558


namespace number_of_arrangements_l7_7520

-- definitions based on conditions
def mr : Type := Unit
def mrs : Type := Unit
def eldest_child : Type := Unit
def middle_child : Type := Unit
def youngest_child : Type := Unit
def person := mr ⊕ mrs ⊕ eldest_child ⊕ middle_child ⊕ youngest_child

def is_driver (p : person) : Prop :=
p = Sum.inl () ∨ p = Sum.inr (Sum.inl ())

def in_front (p1 p2 : person) : Prop :=
is_driver p1 ∧ ¬ is_driver p2

def in_back (p1 p2 p3 : person) : Prop :=
p1 = Sum.inr (Sum.inr (Sum.inl ())) ∧ p2 ≠ Sum.inr (Sum.inr (Sum.inl ())) ∧ p3 ≠ Sum.inr (Sum.inr (Sum.inl ()))

def valid_arrangement (front1 front2 : person) (back1 back2 back3 : person) : Prop :=
in_front front1 front2 ∧ in_back back1 back2 back3

theorem number_of_arrangements : ∃ (front1 front2 back1 back2 back3 : person), valid_arrangement front1 front2 back1 back2 back3 ∧ 16 := by
  sorry

end number_of_arrangements_l7_7520


namespace xy_divides_x2_plus_2y_minus_1_l7_7734

theorem xy_divides_x2_plus_2y_minus_1 (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * y ∣ x^2 + 2 * y - 1) ↔ (∃ t : ℕ, t > 0 ∧ ((x = 1 ∧ y = t) ∨ (x = 2 * t - 1 ∧ y = t)
  ∨ (x = 3 ∧ y = 8) ∨ (x = 5 ∧ y = 8))) :=
by
  sorry

end xy_divides_x2_plus_2y_minus_1_l7_7734


namespace toothpaste_amount_in_tube_l7_7574

def dad_usage_per_brush : ℕ := 3
def mom_usage_per_brush : ℕ := 2
def kid_usage_per_brush : ℕ := 1
def brushes_per_day : ℕ := 3
def days : ℕ := 5

theorem toothpaste_amount_in_tube (dad_usage_per_brush mom_usage_per_brush kid_usage_per_brush brushes_per_day days : ℕ) : 
  dad_usage_per_brush * brushes_per_day * days + 
  mom_usage_per_brush * brushes_per_day * days + 
  (kid_usage_per_brush * brushes_per_day * days * 2) = 105 := 
  by sorry

end toothpaste_amount_in_tube_l7_7574


namespace repeating_decimals_sum_l7_7704

-- Definitions of repeating decimals as fractions
def repeatingDecimalToFrac (a b : ℕ) : ℚ :=
  (a : ℚ) / (b : ℚ)

-- Given conditions
def x : ℚ := repeatingDecimalToFrac 1 3
def y : ℚ := repeatingDecimalToFrac 2 3

-- Theorem statement
theorem repeating_decimals_sum : x + y = 1 := 
begin
  sorry
end

end repeating_decimals_sum_l7_7704


namespace solve_equation_l7_7154

noncomputable def f (x : ℝ) : ℝ :=
  2 * x + 1 + Real.arctan x * Real.sqrt (x^2 + 1)

theorem solve_equation : ∃ x : ℝ, f x + f (x + 1) = 0 ∧ x = -1/2 :=
  by
    use -1/2
    simp [f]
    sorry

end solve_equation_l7_7154


namespace prob_0_3_of_prob_lt_6_l7_7548

variables (σ : ℝ) (ξ : ℝ → ℝ) (h_norm_dist : ∀ x, ξ x = pdf_normal 3 σ x)
noncomputable def prob_xi_lt_6 : ℝ := integrate (λ x, pdf_normal 3 σ x) (-∞) 6

noncomputable def prob_xi_in_03 : ℝ := integrate (λ x, pdf_normal 3 σ x) 0 3

theorem prob_0_3_of_prob_lt_6 (h : prob_xi_lt_6 = 0.8) : prob_xi_in_03 = 0.3 := 
by  
  have h0 : integrate (λ x, pdf_normal 3 σ x) (-∞) 0 = 0.2 :=
    calc 
      integrate (λ x, pdf_normal 3 σ x) (-∞) 0 = 1 - integrate (λ x, pdf_normal 3 σ x) 0 ∞ : by sorry

  have h1 : integrate (λ x, pdf_normal 3 σ x) 0 6 = 0.6 :=
    calc 
      integrate (λ x, pdf_normal 3 σ x) 0 6 = integrate (λ x, pdf_normal 3 σ x) (-∞) 6 - integrate (λ x, pdf_normal 3 σ x) (-∞) 0 : by sorry **,

  show integrate (λ x, pdf_normal 3 σ x) 0 3 = 0.3 := 
    calc 
      integrate (λ x, pdf_normal 3 σ x) 0 3 = 0.5 * integrate (λ x, pdf_normal 3 σ x) 0 6 : by sorry
      ... = 0.3

end prob_0_3_of_prob_lt_6_l7_7548


namespace classmates_ate_cake_l7_7091

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l7_7091


namespace repeating_six_as_fraction_l7_7742

theorem repeating_six_as_fraction : (∑' n : ℕ, 6 / (10 * (10 : ℝ)^n)) = (2 / 3) :=
by
  sorry

end repeating_six_as_fraction_l7_7742


namespace classmates_ate_cake_l7_7093

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l7_7093


namespace simplest_quadratic_radical_l7_7617

theorem simplest_quadratic_radical :
  ∀ (a b c d : ℝ),
    a = Real.sqrt 12 →
    b = Real.sqrt (2 / 3) →
    c = Real.sqrt 0.3 →
    d = Real.sqrt 7 →
    d = Real.sqrt 7 :=
by
  intros a b c d ha hb hc hd
  rw [hd]
  sorry

end simplest_quadratic_radical_l7_7617


namespace dot_product_example_l7_7427

variables (a : ℝ × ℝ) (b : ℝ × ℝ)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_example 
  (ha : a = (-1, 1)) 
  (hb : b = (3, -2)) : dot_product a b = -5 := by
  sorry

end dot_product_example_l7_7427


namespace branches_count_eq_6_l7_7344

theorem branches_count_eq_6 (x : ℕ) (h : 1 + x + x^2 = 43) : x = 6 :=
sorry

end branches_count_eq_6_l7_7344


namespace lowest_score_85_avg_l7_7147

theorem lowest_score_85_avg (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 = 79) (h2 : a2 = 88) (h3 : a3 = 94) 
  (h4 : a4 = 91) (h5 : 75 ≤ a5) (h6 : 75 ≤ a6) 
  (h7 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 85) : (a5 = 75 ∨ a6 = 75) ∧ (a5 = 75 ∨ a5 > 75) := 
by
  sorry

end lowest_score_85_avg_l7_7147


namespace leopards_arrangement_l7_7949

theorem leopards_arrangement (leopards : Fin 9 → Nat) (tallest shortest1 shortest2 : Fin 9) :
  (∃ i, leopards i = tallest ∧ i = 4) ∧
  ((leopards 0 = shortest1 ∧ leopards 8 = shortest2) ∨ (leopards 0 = shortest2 ∧ leopards 8 = shortest1)) →
  2 * 6.factorial = 1440 := by
  sorry

end leopards_arrangement_l7_7949


namespace CosSinQuartic_l7_7923

variable {x : ℝ} {a : ℝ}

theorem CosSinQuartic (h1 : cos x ^ 6 + sin x ^ 6 = a) (h2 : cos x ^ 2 + sin x ^ 2 = 1) :
  cos x ^ 4 + sin x ^ 4 = (1 + 2 * a) / 3 := by
  sorry

end CosSinQuartic_l7_7923


namespace drink_costs_l7_7693

theorem drink_costs (cost_of_steak_per_person : ℝ) (total_tip_paid : ℝ) (tip_percentage : ℝ) (billy_tip_coverage_percentage : ℝ) (total_tip_percentage : ℝ) :
  cost_of_steak_per_person = 20 → 
  total_tip_paid = 8 → 
  tip_percentage = 0.20 → 
  billy_tip_coverage_percentage = 0.80 → 
  total_tip_percentage = 0.20 → 
  ∃ (cost_of_drink : ℝ), cost_of_drink = 1.60 :=
by
  intros
  sorry

end drink_costs_l7_7693


namespace percentile_25_eq_6_l7_7159

-- Define the satisfaction index numbers
def data : List ℕ := [8, 4, 5, 6, 9, 8, 9, 7, 10, 10]

-- Define the percentile calculation function
noncomputable def percentile (p : ℝ) (l : List ℕ) : ℝ :=
  let sorted_l := l.qsort (· < ·)
  let pos := (p / 100) * sorted_l.length
  if pos % 1 = 0 then sorted_l.nth (pos.toInt - 1) else sorted_l.nth pos.ceil.toInt.sorry

-- Assume p = 25
def p := 25

-- Assert the 25% percentile is 6
theorem percentile_25_eq_6 : percentile p data = 6 := by
  sorry

end percentile_25_eq_6_l7_7159


namespace complement_union_sets_l7_7425

open Set

theorem complement_union_sets :
  let U := univ : Set ℝ
  let A := {x : ℝ | x ≤ 1}
  let B := {x : ℝ | x ≥ 2}
  compl (A ∪ B) = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end complement_union_sets_l7_7425


namespace smallest_positive_period_intervals_monotonically_increasing_max_min_values_l7_7413

noncomputable def f (x : ℝ) : ℝ := 2 * sin (3 * x + π/4)

-- (I) Prove the smallest positive period of the function f(x) is T = 2π/3
theorem smallest_positive_period :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ T = 2 * π / 3 := by
  sorry

-- (II) Prove the intervals where the function f(x) is monotonically increasing
theorem intervals_monotonically_increasing (k : ℤ) :
  ∀ x, -2 * k * π / 3 - π / 4 ≤ x ∧ x ≤ 2 * k * π / 3 + π / 12 → 
    ∃ I, ∀ x₁ x₂ ∈ I, (x₁ < x₂ → f x₁ < f x₂) := by
  sorry

-- (III) Prove the maximum and minimum values of the function f(x) in [-π/6, π/6]
theorem max_min_values :
  ∃ x ∈ Icc (-π/6) (π/6), (∀ y ∈ Icc (-π/6) (π/6), f y ≤ f x ∧ f (-π/4) = 2) ∧ 
  (∀ y ∈ Icc (-π/6) (π/6), f y ≥ f x ∧ f (-π/4) = -sqrt 2) := by
  sorry

end smallest_positive_period_intervals_monotonically_increasing_max_min_values_l7_7413


namespace percentage_error_l7_7302

-- Define the conditions
def actual_side (a : ℝ) := a
def measured_side (a : ℝ) := 1.05 * a
def actual_area (a : ℝ) := a^2
def calculated_area (a : ℝ) := (1.05 * a)^2

-- Define the statement that we need to prove
theorem percentage_error (a : ℝ) (h : a > 0) :
  (calculated_area a - actual_area a) / actual_area a * 100 = 10.25 :=
by
  -- Proof goes here
  sorry

end percentage_error_l7_7302


namespace segment_length_l7_7212

theorem segment_length (x : ℝ) (h : |x - (27)^(1/3)| = 5) : ∃ a b : ℝ, (a = 8 ∧ b = -2 ∨ a = -2 ∧ b = 8) ∧ real.dist a b = 10 :=
by
  use [8, -2] -- providing the endpoints explicitly
  split
  -- prove that these are the correct endpoints
  · left; exact ⟨rfl, rfl⟩
  -- prove the distance is 10
  · apply real.dist_eq; linarith
  

end segment_length_l7_7212


namespace find_GQ_in_triangle_XYZ_l7_7914

noncomputable def GQ_in_triangle_XYZ_centroid : ℝ :=
  let XY := 13
  let XZ := 15
  let YZ := 24
  let centroid_ratio := 1 / 3
  let semi_perimeter := (XY + XZ + YZ) / 2
  let area := Real.sqrt (semi_perimeter * (semi_perimeter - XY) * (semi_perimeter - XZ) * (semi_perimeter - YZ))
  let heightXR := (2 * area) / YZ
  (heightXR * centroid_ratio)

theorem find_GQ_in_triangle_XYZ :
  GQ_in_triangle_XYZ_centroid = 2.4 :=
sorry

end find_GQ_in_triangle_XYZ_l7_7914


namespace average_age_in_2030_is_correct_l7_7463

def Wayne_age_2021 := 37
def Peter_age_2021 := Wayne_age_2021 + 3
def Julia_age_2021 := Peter_age_2021 + 2
def Greg_age_2021 := Wayne_age_2021 + 30
def Karen_age_2021 := Wayne_age_2021 + 28

def Wayne_age_2030 := Wayne_age_2021 + 9
def Peter_age_2030 := Peter_age_2021 + 9
def Julia_age_2030 := Julia_age_2021 + 9
def Greg_age_2030 := Greg_age_2021 + 9
def Karen_age_2030 := Karen_age_2021 + 9

def family_average_age_2030 := (Wayne_age_2030 + Peter_age_2030 + Julia_age_2030 + Greg_age_2030 + Karen_age_2030) / 5

theorem average_age_in_2030_is_correct : family_average_age_2030 = 59.2 := by
  sorry

end average_age_in_2030_is_correct_l7_7463


namespace find_ABC_plus_DE_l7_7611

theorem find_ABC_plus_DE (ABCDE : Nat) (h1 : ABCDE = 13579 * 6) : (ABCDE / 1000 + ABCDE % 1000 % 100) = 888 :=
by
  sorry

end find_ABC_plus_DE_l7_7611


namespace alice_bob_age_difference_18_l7_7185

-- Define Alice's and Bob's ages with the given constraints
def is_odd (n : ℕ) : Prop := n % 2 = 1

def alice_age (a b : ℕ) : ℕ := 10 * a + b
def bob_age (a b : ℕ) : ℕ := 10 * b + a

theorem alice_bob_age_difference_18 (a b : ℕ) (ha : is_odd a) (hb : is_odd b)
  (h : alice_age a b + 7 = 3 * (bob_age a b + 7)) : alice_age a b - bob_age a b = 18 :=
sorry

end alice_bob_age_difference_18_l7_7185


namespace trig_identity_simplification_l7_7809

theorem trig_identity_simplification (α : ℝ) (h : π < α ∧ α < 3 * π / 2) :
  (sin (α - π / 2) * cos (3 / 2 * π + α) * tan (π - α)) 
  /
  (tan (-α - π) * sin (-π - α)) = -cos α :=
by
  sorry

end trig_identity_simplification_l7_7809


namespace minimum_value_of_x_y_l7_7429

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  x + y

theorem minimum_value_of_x_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1 - x) * (-y) = x) : minimum_value x y = 4 :=
  sorry

end minimum_value_of_x_y_l7_7429


namespace sasha_cannot_identical_fruits_l7_7955

-- Definitions
def oranges : Int := 13
def apples : Int := 3
def total_fruits : Int := 16

-- Enum for Fruit representing +1 for orange and -1 for apple
inductive Fruit
| orange : Fruit
| apple : Fruit

open Fruit

-- Function to give value to a fruit
def fruit_value : Fruit → Int
| orange => 1
| apple => -1

-- Initial fruit configuration in terms of values
def initial_configuration : List (List Int) := 
  [[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1]]

-- Function to flip a fruit's value
def flip (f : Int) : Int := -f

-- Function that flips all fruits in a row
def flip_row (row : List Int) : List Int :=
  List.map flip row

-- Function that flips all fruits in a column
def flip_column (config : List (List Int)) (col_idx : Nat) : List (List Int) :=
  List.mapIdx (λ i row => row.set col_idx (flip (row.get col_idx))) config

-- Function that flips one fruit in each row
def flip_diagonal (config : List (List Int)) : List (List Int) :=
  List.mapIdx (λ i row => row.set i (flip (row.get i))) config

-- Function to compute the product of all fruits' values
def product_of_all_fruits (config : List (List Int)) : Int :=
  List.foldl (*) 1 (List.concat config)

-- Theorem statement
theorem sasha_cannot_identical_fruits :
  ∀ config : List (List Int),
  config = initial_configuration →
  (∀ config' : List (List Int),
    config' = flip_row config !!1-- One transformation
    ∨ config' = flip_column config !!1-- Another transformation
    ∨ config' = flip_diagonal config,
    product_of_all_fruits config' = product_of_all_fruits config) →
  ¬(∀ f : Int, f ∈ List.concat config → f = 1 ∨ f = -1) → -- cannot become all identical
  False :=
begin
  sorry
end

end sasha_cannot_identical_fruits_l7_7955


namespace lcm_1_to_12_l7_7218

theorem lcm_1_to_12 : nat.lcm (list.range (12 + 1)) = 27720 :=
begin
  sorry
end

end lcm_1_to_12_l7_7218


namespace train_speed_l7_7638

theorem train_speed (d t s : ℝ) (h1 : d = 320) (h2 : t = 6) (h3 : s = 53.33) :
  s = d / t :=
by
  rw [h1, h2]
  sorry

end train_speed_l7_7638


namespace limit_exists_l7_7572

noncomputable def seq (a₁ : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a₁ else (2 * seq a₁ (n - 1))^(1 / seq a₁ (n - 1))

theorem limit_exists (a₁ : ℝ) (h : a₁ > 0) : ∃ L : ℝ, filter.tendsto (seq a₁) filter.nat_at_top (filter.pure L) :=
sorry

end limit_exists_l7_7572


namespace simplest_quadratic_radical_l7_7613

theorem simplest_quadratic_radical :
  let a := Real.sqrt 12
  let b := Real.sqrt (2 / 3)
  let c := Real.sqrt 0.3
  let d := Real.sqrt 7
  d < a ∧ d < b ∧ d < c := 
by {
  -- the proof steps will go here, but we use sorry for now
  sorry
}

end simplest_quadratic_radical_l7_7613


namespace sets_equivalence_l7_7681

theorem sets_equivalence :
  (∀ M N, (M = {(3, 2)} ∧ N = {(2, 3)} → M ≠ N) ∧
          (M = {4, 5} ∧ N = {5, 4} → M = N) ∧
          (M = {1, 2} ∧ N = {(1, 2)} → M ≠ N) ∧
          (M = {(x, y) | x + y = 1} ∧ N = {y | ∃ x, x + y = 1} → M ≠ N)) :=
by sorry

end sets_equivalence_l7_7681


namespace part1_monotonicity_part2_range_b_part3_range_a_l7_7824

noncomputable def f (a x : ℝ) : ℝ := exp x + a * x - 1

-- Part 1: Monotonicity
theorem part1_monotonicity (a : ℝ) :
  (a ≥ 0 → ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (a < 0 → ∃ c : ℝ, c = log (-a) ∧ 
    (∀ x y : ℝ, x < y ∧ y ≤ c → f a x > f a y) ∧ 
    (∀ x y : ℝ, x ≥ c ∧ x < y → f a x < f a y)) :=
sorry

-- Part 2: Range of b
theorem part2_range_b (b : ℝ) (hx : ∀ x : ℝ, 0 < x → f (-exp 1) x ≥ b * x - 1) :
  b ≤ 0 :=
sorry

-- Part 3: Range of a
theorem part3_range_a (a : ℝ) (hz : ∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) :
  a < 0 :=
sorry

end part1_monotonicity_part2_range_b_part3_range_a_l7_7824


namespace units_digit_diff_l7_7397

theorem units_digit_diff (p : ℕ) (hp : p > 0) (even_p : p % 2 = 0) (units_p1_7 : (p + 1) % 10 = 7) : (p^3 % 10) = (p^2 % 10) :=
by
  sorry

end units_digit_diff_l7_7397


namespace original_triangle_area_l7_7161

theorem original_triangle_area (A_new : ℝ) (k : ℝ) (h1 : k = 4) (h2 : A_new = 64) :
  let A_orig := A_new / (k * k) in A_orig = 4 :=
by
  let A_orig := A_new / (k * k)
  sorry

end original_triangle_area_l7_7161


namespace equation_of_line_l7_7987

theorem equation_of_line (h₁ : (2 : ℝ), -1) :=
  let m := Real.tan (Real.pi / 4)
  m = 1 → ∃ (a b c : ℝ), a * x + b * y + c = 0 :=
begin
  sorry,
end

end equation_of_line_l7_7987


namespace repeating_decimal_sum_l7_7699
noncomputable def repeater := sorry  -- Definitions of repeating decimals would be more complex in Lean, skipping implementation.

-- Define the repeating decimals as constants for the purpose of this proof
def decimal1 : ℚ := 1 / 3
def decimal2 : ℚ := 2 / 3

-- Define the main theorem
theorem repeating_decimal_sum : decimal1 + decimal2 = 1 := by
  sorry  -- Placeholder for proof

end repeating_decimal_sum_l7_7699


namespace calc_f_g_2_l7_7827

theorem calc_f_g_2 :
  let f := λ x : ℝ, x - 4
  let g := λ x : ℝ, x^2 - 1
  f (g 2) = -1 := by
  sorry

end calc_f_g_2_l7_7827


namespace solve_equation_l7_7153

theorem solve_equation (x : ℝ) :
  (∃ x, (x = (27 + real.sqrt 769) / 10 ∨ x = (27 - real.sqrt 769) / 10) → 
  real.cbrt (5 * x - 2 / x) = 3) :=
by
  assume h,
  cases h with x1 x2,
  { rw x1,
    sorry
  },
  { rw x2,
    sorry
  }

end solve_equation_l7_7153


namespace units_digit_of_k_l7_7942

-- Definitions
variable {k : ℤ}
variable {a : ℝ}
variable {n : ℕ}

-- Conditions
def k_gt_1 : Prop := k > 1
def a_root : Prop := a^2 - k * a + 1 = 0
def units_digit_7 (n : ℕ) : Prop := natDigits 10 (int.natAbs ((a^(2^n) + a^(-2^n)).toInt)).last = some 7

-- Question as Lean Theorem
theorem units_digit_of_k (k : ℤ) (a : ℝ) (n : ℕ) [nat_gt_10 : n > 10] : 
  k_gt_1 → 
  a_root → 
  ∀ n > 10, 
  units_digit_7 n → 
  natDigits 10 (int.natAbs k).last = some 3 ∨ 
  natDigits 10 (int.natAbs k).last = some 5 ∨ 
  natDigits 10 (int.natAbs k).last = some 7 :=
sorry

end units_digit_of_k_l7_7942


namespace infinitely_many_partitions_l7_7534

theorem infinitely_many_partitions :
  ∃ (A B C : List ℕ), 
  ∀ (n : ℕ), 
    ∃ (m : ℕ), n = 3 * m ∧ 
    (∀ (i : ℕ), i < m → (A.nth i).getOrElse 0 + (B.nth i).getOrElse 0 = (C.nth i).getOrElse 0) ∧ 
    (A ++ B ++ C = List.range (1, n + 1)) :=
begin
  sorry
end

end infinitely_many_partitions_l7_7534


namespace marlon_gift_card_balance_l7_7070

theorem marlon_gift_card_balance 
  (initial_amount : ℕ) 
  (spent_monday : initial_amount / 2 = 100)
  (spent_tuesday : (initial_amount / 2) / 4 = 25) 
  : (initial_amount / 2) - (initial_amount / 2 / 4) = 75 :=
by
  sorry

end marlon_gift_card_balance_l7_7070


namespace distance_downstream_l7_7183

-- Definitions of given variables
def speed_boat_still_water : ℝ := 65  -- speed in km/hr
def rate_current : ℝ := 15            -- rate of the current in km/hr
def time_minutes : ℝ := 25            -- time in minutes

-- Convert time to hours
def time_hours : ℝ := time_minutes / 60

-- Calculate downstream speed
def downstream_speed : ℝ := speed_boat_still_water + rate_current

-- Calculate the distance covered
def distance_covered : ℝ := downstream_speed * time_hours

-- Proof statement placeholder (goal description)
theorem distance_downstream : distance_covered = 33.33 := by
  sorry

end distance_downstream_l7_7183


namespace lcm_1_to_12_l7_7222

theorem lcm_1_to_12 : nat.lcm (list.range (12 + 1)) = 27720 :=
begin
  sorry
end

end lcm_1_to_12_l7_7222


namespace car_production_total_l7_7646

theorem car_production_total (northAmericaCars europeCars : ℕ) (h1 : northAmericaCars = 3884) (h2 : europeCars = 2871) : northAmericaCars + europeCars = 6755 := by
  sorry

end car_production_total_l7_7646


namespace original_triangle_area_l7_7163

-- Define the conditions
def dimensions_quadrupled (original_area new_area : ℝ) : Prop :=
  4^2 * original_area = new_area

-- Define the statement to be proved
theorem original_triangle_area {new_area : ℝ} (h : new_area = 64) :
  ∃ (original_area : ℝ), dimensions_quadrupled original_area new_area ∧ original_area = 4 :=
by
  sorry

end original_triangle_area_l7_7163


namespace height_from_AC_l7_7890

-- Given the sides of the triangle
variables {A B C : Type} [metric_space A] [normed_group B]

-- Define the lengths
def length_AB : ℝ := 3
def length_BC : ℝ := real.sqrt 13
def length_AC : ℝ := 4

-- Proving the height from side AC is the given value
theorem height_from_AC (h : height_from_AC A B C AC = length_AB) 
  (ha : height_from_AC B A C BC = length_BC) 
  (hb : height_from_AC C A B AB = length_AC) : 
  height_from_AC = 3 * real.sqrt 3 / 2 :=
sorry

end height_from_AC_l7_7890


namespace train_speed_l7_7675

theorem train_speed :
  ∀ (length_train : ℕ) (time_cross : ℕ) (speed : ℕ),
    length_train = 1500 → 
    time_cross = 50 → 
    speed = 108 ↔
    ((length_train / 1000) / (time_cross / 3600) = speed) :=
by
  intros length_train time_cross speed h_length h_time
  rw [h_length, h_time]
  norm_num
  sorry

end train_speed_l7_7675


namespace segment_length_of_absolute_value_l7_7206

theorem segment_length_of_absolute_value (x : ℝ) (h : abs (x - (27 : ℝ)^(1/3)) = 5) : 
  |8 - (-2)| = 10 :=
by
  sorry

end segment_length_of_absolute_value_l7_7206


namespace f_injective_on_restricted_domain_l7_7497

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2)^2 - 5

-- Define the restricted domain
def f_restricted (x : ℝ) (h : -2 <= x) : ℝ := f x

-- The main statement to be proved
theorem f_injective_on_restricted_domain : 
  (∀ x1 x2 : {x // -2 <= x}, f_restricted x1.val x1.property = f_restricted x2.val x2.property → x1 = x2) := 
sorry

end f_injective_on_restricted_domain_l7_7497


namespace find_k_l7_7420

noncomputable def vector_a : ℝ × ℝ := (-1, 1)
noncomputable def vector_b : ℝ × ℝ := (2, 3)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (-2, k)

def perp (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_k (k : ℝ) (h : perp (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) (vector_c k)) : k = 1 / 2 :=
by
  sorry

end find_k_l7_7420


namespace lcm_1_to_12_l7_7220

theorem lcm_1_to_12 : nat.lcm (list.range (12 + 1)) = 27720 :=
begin
  sorry
end

end lcm_1_to_12_l7_7220


namespace sequence_length_divide_by_3_l7_7179

theorem sequence_length_divide_by_3 (n : ℕ) (h : n = 4860) : 
  ∃ k : ℕ, k = 6 ∧ ( ∀ m : ℕ, m < 6 → n / (3^m) ∈ ℕ ) :=
by
  sorry

end sequence_length_divide_by_3_l7_7179


namespace shoe_price_calculation_l7_7948

theorem shoe_price_calculation :
  let initialPrice : ℕ := 50
  let increasedPrice : ℕ := 60  -- initialPrice * 1.2
  let discountAmount : ℕ := 9    -- increasedPrice * 0.15
  increasedPrice - discountAmount = 51 := 
by
  sorry

end shoe_price_calculation_l7_7948


namespace dot_product_problem_l7_7814

variables (a b : ℝ^3) (theta : ℝ)
variables (h1 : real.cos theta = 1 / 3)
variables (h2 : ∥a∥ = 3)
variables (h3 : ∥b∥ = 2)

theorem dot_product_problem : (2 • a - 3 • b) ⬝ b = -8 := 
by sorry

end dot_product_problem_l7_7814


namespace arithmetic_sum_l7_7244

theorem arithmetic_sum :
  let a₁ := 4
  let d := 5
  let n := 12
  let aₙ := a₁ + (n - 1) * d
  Sₙ := (n * (a₁ + aₙ)) / 2
  in Sₙ = 378 := sorry

end arithmetic_sum_l7_7244


namespace red_balls_estimate_l7_7013

/-- There are several red balls and 4 black balls in a bag.
Each ball is identical except for color.
A ball is drawn and put back into the bag. This process is repeated 100 times.
Among those 100 draws, 40 times a black ball is drawn.
Prove that the number of red balls (x) is 6. -/
theorem red_balls_estimate (x : ℕ) (h_condition : (4 / (4 + x) = 40 / 100)) : x = 6 :=
by
    sorry

end red_balls_estimate_l7_7013


namespace teacher_students_and_ticket_cost_l7_7982

theorem teacher_students_and_ticket_cost 
    (C_s C_a : ℝ) 
    (n_k n_h : ℕ)
    (hk_total ht_total : ℝ) 
    (h_students : n_h = n_k + 3)
    (hk  : n_k * C_s + C_a = hk_total)
    (ht : n_h * C_s + C_a = ht_total)
    (hk_total_val : hk_total = 994)
    (ht_total_val : ht_total = 1120)
    (C_s_val : C_s = 42) : 
    (n_h = 25) ∧ (C_a = 70) := 
by
  -- Proof steps would be provided here
  sorry

end teacher_students_and_ticket_cost_l7_7982


namespace positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l7_7352

theorem positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5 : 
  ∃ (x : ℕ), (x = 594) ∧ (18 ∣ x) ∧ (24 ≤ Real.sqrt (x) ∧ Real.sqrt (x) ≤ 24.5) := 
by 
  sorry

end positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l7_7352


namespace floor_sequence_eq_l7_7510

noncomputable def sequence (a0 : ℕ) : ℕ → ℝ
| 0       := a0
| (n + 1) := sequence n ^ 2 / (sequence n + 1)

theorem floor_sequence_eq (a0 : ℕ) (n : ℕ) (h₀ : a0 > 0) (h₁ : n ≤ a0 / 2 + 1) :
  ∀ (k : ℕ), 0 ≤ k ∧ k ≤ n → ⌊sequence a0 k⌋ = a0 - k :=
begin
  sorry
end

end floor_sequence_eq_l7_7510


namespace minimal_possible_M_l7_7671

theorem minimal_possible_M (n : ℕ) (board : ℕ → ℕ → ℕ) 
    (h1 : ∀ i j : ℕ, i < n ∧ j < n → 1 ≤ board i j ∧ board i j ≤ n^2) 
    (h2 : ∀ i j : ℕ, i < n ∧ j < n → 
          ((j + 1 < n → abs (board i j - board i (j + 1)) = 1) ∧ 
           (i + 1 < n → abs (board i j - board (i + 1) j) = 2*n - 1))) : 
    ∃ M, M = 2*n - 1 ∧ 
    ∀ u v i j : ℕ, i < n ∧ j < n ∧ (abs (u - v) = 1 → M = max M (abs (board i j - board u v))) :=
sorry

end minimal_possible_M_l7_7671


namespace label_elements_with_zero_or_one_l7_7787

theorem label_elements_with_zero_or_one (n : ℕ) (h : n > 0)
    (B : Type) (A : Fin (2 * n + 1) → Set B)
    (h1 : ∀ i, (A i).card = 2 * n)
    (h2 : ∀ i j, i ≠ j → (A i ∩ A j).card = 1)
    (h3 : ∀ b : B, (∃ i j, i ≠ j ∧ b ∈ A i ∧ b ∈ A j)) :
    (∃ f : B → ℕ, ∀ i, (A i).to_finset.filter (λ b, f b = 0) .card = n) ↔ Even n :=
by sorry

end label_elements_with_zero_or_one_l7_7787


namespace minimum_value_of_f_maximum_value_of_k_l7_7394

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_f : ∃ x : ℝ, 0 < x ∧ f x = -1 / Real.exp 1 :=
sorry

theorem maximum_value_of_k : ∀ x > 2, ∀ k : ℤ, (f x ≥ k * x - 2 * (k + 1)) → k ≤ 3 :=
sorry

end minimum_value_of_f_maximum_value_of_k_l7_7394


namespace lcm_1_to_12_l7_7221

theorem lcm_1_to_12 : nat.lcm (list.range (12 + 1)) = 27720 :=
begin
  sorry
end

end lcm_1_to_12_l7_7221


namespace find_D_coords_find_area_trapezoid_l7_7023

-- Definitions of the given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨1, 7⟩
def B : Point := ⟨7, 5⟩
def C : Point := ⟨4, 1⟩

-- Theorem statement for the coordinates of point D
theorem find_D_coords :
  ∃ D : Point, (D.x = 1 ∧ D.y = 2) ∧
  (B.x - A.x) / (B.y - A.y) = (D.x - C.x) / (D.y - C.y) ∧
  ((C.x - A.x) * (B.x - D.x) + (C.y - A.y) * (B.y - D.y) = 0) :=
sorry

-- Theorem statement for the area of trapezoid ABCD
theorem find_area_trapezoid :
  ∀ D : Point, (D.x = 1 ∧ D.y = 2) →
  let AC := real.sqrt ((C.x - A.x) ^ 2 + (C.y - A.y) ^ 2),
      BD := real.sqrt ((D.x - B.x) ^ 2 + (D.y - B.y) ^ 2)
  in (AC * BD / 2) = 45 / 2 := 
sorry

end find_D_coords_find_area_trapezoid_l7_7023


namespace last_rope_length_l7_7588

def totalRopeLength : ℝ := 35
def rope1 : ℝ := 8
def rope2 : ℝ := 20
def rope3a : ℝ := 2
def rope3b : ℝ := 2
def rope3c : ℝ := 2
def knotLoss : ℝ := 1.2
def numKnots : ℝ := 4

theorem last_rope_length : 
  (35 + (4 * 1.2)) = (8 + 20 + 2 + 2 + 2 + x) → (x = 5.8) :=
sorry

end last_rope_length_l7_7588


namespace area_trapezoid_DEFG_l7_7024

-- Define the parameters of the problem
variables (AB CD AD BC DEFG : ℝ)
variables (E G : ℝ)
variables (Area_ABCD : ℝ)

-- Conditions
def is_parallel (x y : ℝ) : Prop := True  -- Simplified since any real numbers can be parallel in this context

def is_midpoint (x y : ℝ) : Prop := x = y

-- Given conditions
def given_conditions : Prop :=
  is_parallel AB CD ∧
  is_midpoint E (AD / 2) ∧
  is_midpoint G (BC / 2) ∧
  Area_ABCD = 96

-- Prove that the area of trapezoid DEFG is 24 given the conditions
theorem area_trapezoid_DEFG
  (AB CD AD BC DEFG E G : ℝ)
  (Area_ABCD : ℝ)
  (h1 : is_parallel AB CD)
  (h2 : is_midpoint E (AD / 2))
  (h3 : is_midpoint G (BC / 2))
  (h4 : Area_ABCD = 96) :
  DEFG = 24 :=
begin
  -- proof goes here
  sorry
end

end area_trapezoid_DEFG_l7_7024


namespace polynomial_no_real_roots_l7_7539

def f (x : ℝ) : ℝ := 4 * x ^ 8 - 2 * x ^ 7 + x ^ 6 - 3 * x ^ 4 + x ^ 2 - x + 1

theorem polynomial_no_real_roots : ∀ x : ℝ, f x > 0 := by
  sorry

end polynomial_no_real_roots_l7_7539


namespace decreasing_range_of_a_l7_7877

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x + 1 / x

theorem decreasing_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, 1 ≤ x → deriv (f a) x ≤ 0) ↔ a ≤ -1 / 4 :=
by
  intros a
  have h_deriv : ∀ x, deriv (f a) x = 1 / x + a - 1 / x ^ 2
    sorry
  sorry

end decreasing_range_of_a_l7_7877


namespace lego_triple_pieces_sold_l7_7957

theorem lego_triple_pieces_sold (n m t f q : ℕ) 
  (h_single : n = 100)
  (h_double : m = 45)
  (h_quadruple : q = 165)
  (h_total_earnings : (n * 1 + m * 2 + t * 3 + q * 4) * 0.01 = 10) : t = 50 :=
by
  have single_dollars : n * 0.01 = 1 := by rw [h_single]; norm_num
  have double_dollars : m * 0.02 = 0.9 := by rw [h_double]; norm_num
  have quadruple_dollars : q * 0.04 = 6.6 := by rw [h_quadruple]; norm_num
  have calc_total : 1 + 0.9 + 6.6 = 8.5 := by norm_num
  have in_dollar_t : (t * 3) * 0.01 = 10 - 8.5 := by rw h_total_earnings; field_simp [single_dollars, double_dollars, quadruple_dollars]; norm_num
  have calc_t : t * 3 * 0.01 = 1.5 := by rw in_dollar_t; norm_num
  have result_t : t * 0.03 = 1.5 := calc_t
  norm_num at result_t
  exact result_t

end lego_triple_pieces_sold_l7_7957


namespace hyperbola_equation_unique_l7_7791

noncomputable def hyperbola_asymptote_distance (a b : ℝ) := 
  ( ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1 ) ∧ 
  ( ∀ x, y = (√3 / 3) * x ∨ y = -(√3 / 3) * x ) ∧ 
  ( ( |√3 * a - 3 * 0| / √( ( √3 )^2 + (-3)^2 ) ) = 1 )

theorem hyperbola_equation_unique : 
  {a : ℝ // 0 < a} → 
  {b : ℝ // 0 < b} →
  hyperbola_asymptote_distance a b →
  (x^2 / 3) - y^2 = 1 := 
by 
  intros,
  sorry

end hyperbola_equation_unique_l7_7791


namespace maximum_ratio_is_2_plus_2_sqrt2_l7_7904

noncomputable def C1_polar_eq (θ : ℝ) : Prop :=
  ∀ ρ : ℝ, ρ * (Real.cos θ + Real.sin θ) = 1

noncomputable def C2_polar_eq (θ : ℝ) : Prop :=
  ∀ ρ : ℝ, ρ = 4 * Real.cos θ

theorem maximum_ratio_is_2_plus_2_sqrt2 (α : ℝ) (hα : 0 ≤ α ∧ α ≤ Real.pi / 2) :
  ∃ ρA ρB : ℝ, (ρA = 1 / (Real.cos α + Real.sin α)) ∧ (ρB = 4 * Real.cos α) ∧ 
  (4 * Real.cos α * (Real.cos α + Real.sin α) = 2 + 2 * Real.sqrt 2) :=
sorry

end maximum_ratio_is_2_plus_2_sqrt2_l7_7904


namespace mary_can_keep_warm_l7_7953

theorem mary_can_keep_warm :
  let chairs := 18
  let chairs_sticks := 6
  let tables := 6
  let tables_sticks := 9
  let stools := 4
  let stools_sticks := 2
  let sticks_per_hour := 5
  let total_sticks := (chairs * chairs_sticks) + (tables * tables_sticks) + (stools * stools_sticks)
  let hours := total_sticks / sticks_per_hour
  hours = 34 := by
{
  sorry
}

end mary_can_keep_warm_l7_7953


namespace perfect_square_if_integer_l7_7149

theorem perfect_square_if_integer (n : ℤ) (k : ℤ) 
  (h : k = 2 + 2 * Int.sqrt (28 * n^2 + 1)) : ∃ m : ℤ, k = m^2 :=
by 
  sorry

end perfect_square_if_integer_l7_7149


namespace function_properties_l7_7790

variable {𝓡: Type*} [OrderedRing 𝓡]

def odd_function (f : 𝓡 → 𝓡) := ∀ x, f (-x) = -f x
def periodic_function (f : 𝓡 → 𝓡) (p : 𝓡) := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∀ x, f(x) + f(4 - x) = 0) 
  (h2 : ∀ x, f(x + 2) - f(x - 2) = 0) :
  odd_function f ∧ periodic_function f 4 :=
by
  sorry

end function_properties_l7_7790


namespace find_hyperbola_equation_l7_7359

/-- 
  Given a hyperbola with the equation x²/2 - y² = 1 and an ellipse with the equation y²/8 + x²/2 = 1,
  we want to find the equation of a hyperbola that shares the same asymptotes as the given hyperbola 
  and shares a common focus with the ellipse.
-/

noncomputable def hyperbola_equation : Prop :=
  ∃ (x y : ℝ), (y^2 / 2) - (x^2 / 4) = 1 ∧ 
               -- Asymptotes same as x²/2 - y² = 1
               ∀ (x y : ℝ), 2 * y = ± (sqrt (1/8)) * x + C ∧ 
               -- Shares a common focus with y²/8 + x²/2 = 1 (focus distance sqrt 6)
               ∀ (x y : ℝ), focus_distance = sqrt 6

theorem find_hyperbola_equation :
  hyperbola_equation :=
sorry

end find_hyperbola_equation_l7_7359


namespace find_a_l7_7812

theorem find_a (a : ℝ) : (∃ p : ℝ × ℝ, p = (2 - a, a - 3) ∧ p.fst = 0) → a = 2 := by
  sorry

end find_a_l7_7812


namespace force_required_for_bolt_b_20_inch_l7_7989

noncomputable def force_inversely_proportional (F L : ℝ) : ℝ := F * L

theorem force_required_for_bolt_b_20_inch (F L : ℝ) :
  let handle_length_10 := 10
  let force_length_product_bolt_a := 3000
  let force_length_product_bolt_b := 4000
  let new_handle_length := 20
  (F * handle_length_10 = 400)
  ∧ (F * new_handle_length = 200)
  → force_inversely_proportional 400 10 = 4000
  ∧ force_inversely_proportional 200 20 = 4000
:=
by
  sorry

end force_required_for_bolt_b_20_inch_l7_7989


namespace maximal_disconnected_squares_l7_7036

theorem maximal_disconnected_squares (n : ℕ) (h_n : n ≥ 3) (h_odd : n % 2 = 1) :
  ∃ M, M = n^2 ∧ (∃ (coloring : Fin n → Fin n → bool), 
    ∀ (a b : Fin n) (i j : Fin n), 
    ((coloring a i = coloring b j) 
    ∧ (a ≠ b ∨ i ≠ j) 
    ∧ (¬is_connected coloring a i b j)) → 
    ∀ (c d : Fin n) (m o : Fin n), 
    (¬is_connected coloring c m d o)) 
:=
sorry

end maximal_disconnected_squares_l7_7036


namespace lcm_1_to_12_l7_7238

theorem lcm_1_to_12 : Nat.lcm_list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 27720 := by
  sorry

end lcm_1_to_12_l7_7238


namespace largest_value_is_B_l7_7250

-- Definitions of the expressions
def A : ℝ := real.quartic_root (7 * real.cubic_root 8)
def B : ℝ := real.sqrt (8 * real.quartic_root 7)
def C : ℝ := real.sqrt (7 * real.quartic_root 8)
def D : ℝ := real.cubic_root (7 * real.sqrt 8)
def E : ℝ := real.cubic_root (8 * real.sqrt 7)

-- The proof statement asserting B is the largest value among the options
theorem largest_value_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end largest_value_is_B_l7_7250


namespace debby_pancakes_l7_7326

def total_pancakes (B A P : ℕ) : ℕ := B + A + P

theorem debby_pancakes : 
  total_pancakes 20 24 23 = 67 := by 
  sorry

end debby_pancakes_l7_7326


namespace repeating_decimal_to_fraction_l7_7755

theorem repeating_decimal_to_fraction : (let a := (6 : Real) / 10 in
                                         let r := (1 : Real) / 10 in
                                         ∑' n : ℕ, a * r^n) = (2 : Real) / 3 :=
by
  sorry

end repeating_decimal_to_fraction_l7_7755


namespace which_options_imply_inverse_order_l7_7252

theorem which_options_imply_inverse_order (a b : ℝ) :
  ((b > 0 ∧ 0 > a) ∨ (0 > a ∧ a > b) ∨ (a > b ∧ b > 0)) →
  (1 / a < 1 / b) :=
by
  intro h
  cases h
  case inl h1 =>
    have ha : a < 0 := h1.2
    have hb : b > 0 := h1.1
    have hab : a < 0 ∧ 0 < b := ⟨ha, hb⟩
    calc
      1 / a < 1 / b := by sorry
  case inr h2 =>
    cases h2
    case inl h3 =>
      have ha : a < 0 := h3.1.1
      have hb : b < a  := h3.1.2
      have hb_lt_a : b < a ∧ a < 0 := ⟨hb, ha⟩
      calc
        1 / a < 1 / b := by sorry
    case inr h4 =>
      have ha : a > b := h4.1
      have hb : b > 0 := h4.2
      have a_gt_b_and_b_gt_0 : a > b ∧ b > 0 := ⟨ha, hb⟩
      calc
        1 / a < 1 / b := by sorry

end which_options_imply_inverse_order_l7_7252


namespace circumcenter_inequality_l7_7803

variables (x y : ℝ)
variables (O A B C : Type)
variables [inner_product_space ℝ O]

-- Define vectors OA, OB, and OC 
variables (OA OB OC : O)
-- Define scalar product
variables (dot_product : O → O → ℝ)

-- Conditions from Problem
def is_circumcenter (O : O) (A B C : O) : Prop := inner_product_space ℝ O

-- Magnitude condition
def magnitude_condition (x y : ℝ) (A B : O) : Prop :=
∀ (r : ℝ), (r = 1) → (dot_product A B < 1) → (1 = x^2 + y^2 + 2*x*y*(dot_product A B))

-- Vector equation OC = x * OA + y * OB
def vector_equation (OC OA OB : O) (x y : ℝ) : Prop :=
OC = x • OA + y • OB

-- The final theorem to prove
theorem circumcenter_inequality (h1 : is_circumcenter O A B C)
                                (h2 : magnitude_condition x y OA OB)
                                (h3 : vector_equation OC OA OB x y) : 
  x + y < -1 := sorry

end circumcenter_inequality_l7_7803


namespace number_of_classmates_ate_cake_l7_7145

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l7_7145


namespace largest_four_digit_perfect_cube_is_9261_l7_7603

-- Define the notion of a four-digit number and perfect cube
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

-- The main theorem statement
theorem largest_four_digit_perfect_cube_is_9261 :
  ∃ n, is_four_digit n ∧ is_perfect_cube n ∧ (∀ m, is_four_digit m ∧ is_perfect_cube m → m ≤ n) ∧ n = 9261 :=
sorry -- Proof is omitted

end largest_four_digit_perfect_cube_is_9261_l7_7603


namespace expected_games_per_match_l7_7778

theorem expected_games_per_match 
  (P_F : ℝ) (P_J : ℝ) (n : ℕ) 
  (hF : P_F = 0.3) (hJ : P_J = 0.7) (hn : n = 21) : 
  expected_value P_F P_J n = 30 :=
sorry

end expected_games_per_match_l7_7778


namespace volunteers_adjacent_l7_7014

theorem volunteers_adjacent (volunteers : Fin 6 → ℕ) (A B : ℕ) :
  (∀ i, volunteers i = A ∨ volunteers i = B ∨ 1 ≤ volunteers i ∧ volunteers i ≤ 6) →
  (∃ (i : Fin 5), (volunteers i = A ∧ volunteers (i + 1) = B) ∨ (volunteers i = B ∧ volunteers (i + 1) = A)) →
  240 = 5! * 2 := sorry

end volunteers_adjacent_l7_7014


namespace counterexample_to_symmetry_statement_l7_7917

open Set

-- Definition of quadrilateral with an axis of symmetry
structure QuadrilateralWithSymmetry :=
  (A B C D : Point)
  (axis : Line)
  (symmetry_condition : symmetric_about A axis ∧ symmetric_about B axis ∧ symmetric_about C axis ∧ symmetric_about D axis)

-- Definitions of types of quadrilaterals
def IsIsoscelesTrapezoid (q : QuadrilateralWithSymmetry) : Prop :=
  ∃ p1 p2 : Line, p1 ∥ p2 ∧ q.has_parallel_sides

def IsRectangle (q : QuadrilateralWithSymmetry) : Prop :=
  ∀ angles : q.A apex B = pi, q.B apex C = pi, q.C apex D = pi, q.D apex A = pi

def IsRhombus (q : QuadrilateralWithSymmetry) : Prop :=
  q.AB.length = q.BC.length ∧ q.BC.length = q.CD.length ∧ q.CD.length = q.DA.length

-- The statement to be proved: the existence of a counterexample to the given statement
theorem counterexample_to_symmetry_statement :
  ∃ (q : QuadrilateralWithSymmetry), ¬(IsIsoscelesTrapezoid q ∨ IsRectangle q ∨ IsRhombus q) :=
by
  sorry -- Proof of the counterexample

end counterexample_to_symmetry_statement_l7_7917


namespace sum_of_numbers_without_0_and_9_l7_7772

theorem sum_of_numbers_without_0_and_9 (a b c : ℕ) (h_a : 1 ≤ a ∧ a ≤ 8) (h_b : 1 ≤ b ∧ b ≤ 8) (h_c : 1 ≤ c ∧ c ≤ 8) :
  ∑ (n : ℕ) in (finset.filter (λ x, 
    let d₁ := x / 100,
        d₂ := (x % 100) / 10,
        d₃ := x % 10 in
    1 ≤ d₁ ∧ d₁ ≤ 8 ∧ 1 ≤ d₂ ∧ d₂ ≤ 8 ∧ 1 ≤ d₃ ∧ d₃ ≤ 8
  ) (finset.range 1000)), n = 255744 :=
sorry

end sum_of_numbers_without_0_and_9_l7_7772


namespace pairs_symmetry_l7_7962

theorem pairs_symmetry (N : ℕ) (hN : N > 2) :
  ∃ f : {ab : ℕ × ℕ // ab.1 < ab.2 ∧ ab.2 ≤ N ∧ ab.2 / ab.1 > 2} ≃ 
           {ab : ℕ × ℕ // ab.1 < ab.2 ∧ ab.2 ≤ N ∧ ab.2 / ab.1 < 2}, 
  true :=
sorry

end pairs_symmetry_l7_7962


namespace bear_cubs_count_l7_7642

theorem bear_cubs_count (total_meat : ℕ) (meat_per_cub : ℕ) (rabbits_per_day : ℕ) (weeks_days : ℕ) (meat_per_rabbit : ℕ)
  (mother_total_meat : ℕ) (number_of_cubs : ℕ) : 
  total_meat = 210 →
  meat_per_cub = 35 →
  rabbits_per_day = 10 →
  weeks_days = 7 →
  meat_per_rabbit = 5 →
  mother_total_meat = rabbits_per_day * weeks_days * meat_per_rabbit →
  meat_per_cub * number_of_cubs + mother_total_meat = total_meat →
  number_of_cubs = 4 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end bear_cubs_count_l7_7642


namespace percentage_of_non_technicians_l7_7895

theorem percentage_of_non_technicians (total_workers technicians non_technicians permanent_technicians permanent_non_technicians temporary_workers : ℝ)
  (h1 : technicians = 0.5 * total_workers)
  (h2 : non_technicians = total_workers - technicians)
  (h3 : permanent_technicians = 0.5 * technicians)
  (h4 : permanent_non_technicians = 0.5 * non_technicians)
  (h5 : temporary_workers = 0.5 * total_workers) :
  (non_technicians / total_workers) * 100 = 50 :=
by
  -- Proof is omitted
  sorry

end percentage_of_non_technicians_l7_7895


namespace cake_sharing_l7_7106

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l7_7106


namespace repeating_six_equals_fraction_l7_7750

theorem repeating_six_equals_fraction : ∃ f : ℚ, (∀ n : ℕ, (n ≥ 1 → (6 * (10 : ℕ) ^ (-n) : ℚ) + (f - (6 * (10 : ℕ) ^ (-n) : ℚ)) = f)) ∧ f = 2 / 3 := sorry

end repeating_six_equals_fraction_l7_7750


namespace profit_functions_properties_l7_7473

noncomputable def R (x : ℝ) := 3000 * x - 20 * x^2
noncomputable def C (x : ℝ) := 600 * x + 2000
noncomputable def p (x : ℝ) := R x - C x
noncomputable def Mp (x : ℝ) := p (x + 1) - p x

-- Main theorem to prove
theorem profit_functions_properties :
  (∀ x, p(x) = -20 * x^2 + 2400 * x - 2000) 
  ∧ (∀ x, 0 < x ∧ x ≤ 100 → Mp(x) = -40 * x + 2380)
  ∧ (∃ x, p(x) = 74000)
  ∧ (∃ x, Mp(x) = 2340)
  ∧ (74000 ≠ 2340)
:= 
sorry

end profit_functions_properties_l7_7473


namespace arithmetic_sequence_problem_l7_7796

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∃ A, ∀ n : ℕ, a n = A * (q ^ (n - 1))

theorem arithmetic_sequence_problem
  (q : ℝ) 
  (h1 : q > 1)
  (h2 : a 1 + a 4 = 9)
  (h3 : a 2 * a 3 = 8)
  (h_seq : is_arithmetic_sequence a q) : 
  (a 2015 + a 2016) / (a 2013 + a 2014) = 4 := 
by 
  sorry

end arithmetic_sequence_problem_l7_7796


namespace carol_remaining_distance_l7_7316

def fuel_efficiency : ℕ := 25 -- miles per gallon
def gas_tank_capacity : ℕ := 18 -- gallons
def distance_to_home : ℕ := 350 -- miles

def total_distance_on_full_tank : ℕ := fuel_efficiency * gas_tank_capacity
def distance_after_home : ℕ := total_distance_on_full_tank - distance_to_home

theorem carol_remaining_distance :
  distance_after_home = 100 :=
sorry

end carol_remaining_distance_l7_7316


namespace marked_price_l7_7309

noncomputable def MP (CP : ℝ) (profit_perc : ℝ) (taxes : List ℝ) (discounts : List ℝ) : ℝ :=
  let SP := CP * (1 + profit_perc)
  let P_tax := SP * (1 + (taxes.foldr (λ x acc => x + acc) 0 - taxes.length))
  let D := discounts.foldr (λ x acc => acc * (1 - x)) 1
  P_tax / D

theorem marked_price (CP : ℝ) (profit_perc : ℝ) (taxes discounts : List ℝ) :
  CP = 85.5 → profit_perc = 0.75 →
  taxes = [0.12, 0.07, 0.05] → discounts = [0.10, 0.05, 0.03] →
  MP CP profit_perc taxes discounts ≈ 227.31 :=
by
  intros hCP hProfit hTaxes hDiscounts
  rw [hCP, hProfit, hTaxes, hDiscounts]
  sorry

end marked_price_l7_7309


namespace repeating_six_equals_fraction_l7_7748

theorem repeating_six_equals_fraction : ∃ f : ℚ, (∀ n : ℕ, (n ≥ 1 → (6 * (10 : ℕ) ^ (-n) : ℚ) + (f - (6 * (10 : ℕ) ^ (-n) : ℚ)) = f)) ∧ f = 2 / 3 := sorry

end repeating_six_equals_fraction_l7_7748


namespace functional_eq_l7_7723

theorem functional_eq (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x * f y + y) = f (x * y) + f y) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) :=
sorry

end functional_eq_l7_7723


namespace fraction_of_students_getting_B_l7_7888

theorem fraction_of_students_getting_B (fA fC fD fF total_pass : ℚ) :
  fA = 1 / 4 → 
  fC = 1 / 8 → 
  fD = 1 / 12 → 
  fF = 1 / 24 → 
  total_pass = 0.875 →
  (total_pass - (fA + fC + fD + fF) = 3 / 8) :=
by
  intros hA hC hD hF htotal_pass
  have common_denom : ℚ := 24
  have fA' := fA * common_denom
  have fC' := fC * common_denom
  have fD' := fD * common_denom
  have fF' := fF * common_denom
  have total_pass' := total_pass * common_denom
  have numerator_pass := total_pass' - (fA' + fC' + fD' + fF')
  have hnum : numerator_pass = 21 - 12 
  rw [hnum]
  eval_sorry -- Placeholder for further necessary steps

end fraction_of_students_getting_B_l7_7888


namespace lcm_from_1_to_12_eq_27720_l7_7236

theorem lcm_from_1_to_12_eq_27720 : nat.lcm (finset.range 12).succ = 27720 :=
  sorry

end lcm_from_1_to_12_eq_27720_l7_7236


namespace repeating_decimal_to_fraction_l7_7752

theorem repeating_decimal_to_fraction : (let a := (6 : Real) / 10 in
                                         let r := (1 : Real) / 10 in
                                         ∑' n : ℕ, a * r^n) = (2 : Real) / 3 :=
by
  sorry

end repeating_decimal_to_fraction_l7_7752


namespace coeff_x6_in_f_l7_7480

noncomputable def f (n : ℕ) : (ℕ → ℕ) := 
  λ x, (finset.range (n+1)).sum (λ k, nat.choose n k * (x-1)^k)

theorem coeff_x6_in_f (n : ℕ) (hn : n ≥ 10) : 
  (∑ i in (finset.range (n - 6 + 1)), (-1)^i * nat.choose (6 + i) 6 * nat.choose n (6 + i)) = 
  nat.choose n 6 * (1 - (-1)^(n-6)) :=
sorry

end coeff_x6_in_f_l7_7480


namespace students_dormitories_arrangement_l7_7152

noncomputable def num_arrangements : ℕ := 6

theorem students_dormitories_arrangement :
  let students := {A, B, C, D, E, F}
  let dormitories := {1, 2, 3}
  let student_in_dormitory1 := {A}
  let student_not_in_dormitory3 := {B, C}
  let arrangements (students : finset) (dormitories : finset) :=
    ∃ f : students → dormitories, 
      (∀ s ∈ student_in_dormitory1, f s = 1) ∧
      (∀ s ∈ student_not_in_dormitory3, f s ≠ 3) ∧
      fintype.card {d : dormitories // finset.filter (λ s, f s = d) students = 2} = 3
  in fintype.card arrangements = 18 :=
by
  -- here will be the detailed proof to arrive at the solution
  sorry

end students_dormitories_arrangement_l7_7152


namespace inequality_solutions_l7_7357

theorem inequality_solutions (y : ℝ) :
  (2 / (y + 2) + 4 / (y + 8) ≥ 1 ↔ (y > -8 ∧ y ≤ -4) ∨ (y ≥ -2 ∧ y ≤ 2)) :=
by
  sorry

end inequality_solutions_l7_7357


namespace range_of_m_l7_7448

noncomputable def equation_has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, 25^(-|x+1|) - 4 * 5^(-|x+1|) - m = 0

theorem range_of_m : ∀ m : ℝ, equation_has_real_roots m ↔ (-3 ≤ m ∧ m < 0) :=
by
  -- Proof omitted
  sorry

end range_of_m_l7_7448


namespace no_integer_roots_l7_7737

theorem no_integer_roots (x : ℤ) : ¬ (x^3 - 5 * x^2 - 11 * x + 35 = 0) := 
sorry

end no_integer_roots_l7_7737


namespace count_valid_pairs_l7_7431

open Nat

def is_valid_pair (x y : ℕ) : Prop :=
  x < 165 ∧ y < 165 ∧ (y^2 % 165 = (x^3 + x) % 165)

theorem count_valid_pairs : (finset.univ.filter (λ (xy : ℕ × ℕ), is_valid_pair xy.1 xy.2)).card = 99 :=
sorry

end count_valid_pairs_l7_7431


namespace evaluate_f_inner_l7_7821

def f (x : ℝ) : ℝ :=
  if x < 3 then 1 / (x^2 - 1) else 2 * x^(-1/2)

theorem evaluate_f_inner : f(f(√5 / 2)) = 1 := by
  sorry

end evaluate_f_inner_l7_7821


namespace minimum_at_x_eq_2_l7_7990

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 4

theorem minimum_at_x_eq_2 : 
  has_local_minimum f 2 := 
sorry

end minimum_at_x_eq_2_l7_7990


namespace saree_final_price_l7_7652

noncomputable def final_price_of_saree (original_price : ℝ) (discounts : List ℝ) (sales_tax : ℝ) (handling_fee : ℝ) : ℝ :=
  let price_after_discounts := discounts.foldl (λ acc d, acc * (1 - d)) original_price
  price_after_discounts * (1 + sales_tax + handling_fee)

theorem saree_final_price :
  final_price_of_saree 1200 [0.18, 0.12, 0.05] 0.03 0.02 = 863.76 :=
by
    sorry

end saree_final_price_l7_7652


namespace repeating_decimals_sum_l7_7705

-- Definitions of repeating decimals as fractions
def repeatingDecimalToFrac (a b : ℕ) : ℚ :=
  (a : ℚ) / (b : ℚ)

-- Given conditions
def x : ℚ := repeatingDecimalToFrac 1 3
def y : ℚ := repeatingDecimalToFrac 2 3

-- Theorem statement
theorem repeating_decimals_sum : x + y = 1 := 
begin
  sorry
end

end repeating_decimals_sum_l7_7705


namespace p_q_sum_l7_7992

noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry

theorem p_q_sum :
  (∃ p q : ℝ → ℝ,
    (∀ x, (q(x) = (a : ℝ) * (x - 1) * (x - 2) * (x - c)) ∧
          (p(4) = 4) ∧
          (q(3) = 3) ∧
          (q polynomial.degree = 3))) →
          (p + q = (λ x, -\frac{1}{2}x^3 + \frac{3}{2}x^2)) :=
by
 sorry

end p_q_sum_l7_7992


namespace max_height_reached_l7_7672

def height (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

theorem max_height_reached : ∃ t : ℝ, ∀ t' : ℝ, height t ≤ height t' :=
sorry

end max_height_reached_l7_7672


namespace box_volume_l7_7338

-- Given conditions
variables (a b c : ℝ)
axiom ab_eq : a * b = 30
axiom bc_eq : b * c = 18
axiom ca_eq : c * a = 45

-- Prove that the volume of the box (a * b * c) equals 90 * sqrt(3)
theorem box_volume : a * b * c = 90 * Real.sqrt 3 :=
by
  sorry

end box_volume_l7_7338


namespace min_product_of_three_different_numbers_from_set_l7_7604

theorem min_product_of_three_different_numbers_from_set :
  let s := {-10, -7, -5, -3, 0, 2, 4, 6, 8}
  ∃ a b c ∈ s, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ∀ (x y z ∈ s), x ≠ y ∧ y ≠ z ∧ x ≠ z → (a * b * c ≤ x * y * z) → (a * b * c = -480) := sorry

end min_product_of_three_different_numbers_from_set_l7_7604


namespace lcm_1_to_12_l7_7239

theorem lcm_1_to_12 : Nat.lcm_list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 27720 := by
  sorry

end lcm_1_to_12_l7_7239


namespace triangle_problem_l7_7462

noncomputable def side_opposite_to_angle_C (a b : ℝ) : ℝ :=
  sqrt (a^2 + b^2 - 2 * a * b * cos (π / 3))

noncomputable def cos_B_minus_C (a b c : ℝ) : ℝ :=
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c) in
  let sin_B := sqrt(1 - cos_B^2) in
  cos_B * (cos (π / 3)) + sin_B * (sin (π / 3))

theorem triangle_problem (a b : ℝ) (A B C : ℝ) (area : ℝ) (hC : C = π / 3) (hb : b = 8)
                          (harea : area = 10 * sqrt 3) :
  let c := side_opposite_to_angle_C a b in
  c = 7 ∧ cos_B_minus_C a b c = 13 / 14 :=
by
  -- Definitions to use within the proof
  sorry

end triangle_problem_l7_7462


namespace angle_DAF_30_degrees_l7_7908

theorem angle_DAF_30_degrees 
  (A B C D E F : Type)
  [affine_space ℝ (E : Type)] 
  (is_square : square A B C D)
  (is_equilateral_abe : equilateral_triangle A B E)
  (is_equilateral_aef : equilateral_triangle A E F) :
  angle D A F = 30 :=
sorry

end angle_DAF_30_degrees_l7_7908


namespace options_implication_l7_7254

theorem options_implication (a b : ℝ) :
  ((b > 0 ∧ a < 0) ∨ (a < 0 ∧ b < 0 ∧ a > b) ∨ (a > 0 ∧ b > 0 ∧ a > b)) → (1 / a < 1 / b) :=
by sorry

end options_implication_l7_7254


namespace variance_of_data_set_l7_7873

noncomputable def variance (s : Finset ℝ) : ℝ :=
  let μ := (∑ x in s, x) / s.card
  (∑ x in s, (x - μ)^2) / s.card

theorem variance_of_data_set :
  ∀ (x y : ℝ), 
  (4 + x + 5 + y + 7 + 9) / 6 = 6 →
  (x = 5 ∨ y = 5) →
  (x + y = 11) →
  variance {4, x, 5, y, 7, 9} = 8 / 3 :=
by 
  intros x y avg mode_sum xy_sum; 
  sorry

end variance_of_data_set_l7_7873


namespace incorrect_inequality_in_triangle_l7_7486

theorem incorrect_inequality_in_triangle 
  (A B C : ℝ) (hB0 : 0 < B) (hBA : B < A) (hAπ : A < π) (h_sum : A + B + C = π) :
  ¬ (sin A ^ 2 > sin B ^ 2) := 
  by sorry

end incorrect_inequality_in_triangle_l7_7486


namespace reciprocal_eq_self_l7_7454

open Classical

theorem reciprocal_eq_self (a : ℝ) (h : a = 1 / a) : a = 1 ∨ a = -1 := 
sorry

end reciprocal_eq_self_l7_7454


namespace determine_card_numbers_l7_7562

noncomputable def prime_powers_encoding (a : ℕ → ℕ) : ℕ :=
  ∏ i in finset.range 100, (nat.prime_seq i)^(a i)

theorem determine_card_numbers 
  (a : ℕ → ℕ)  -- the card numbers function
  (encoded_value : ℕ := prime_powers_encoding a)  -- the encoded product of primes
  (decode : Π (n : ℕ), ℕ)  -- the decoding function to retrieve exponents
  (h : ∀ i, decode (nat.prime_seq i) = a i)  -- the decoding correctly identifies the exponents
  : ∀ i, a i = decode (nat.prime_seq i) :=
by 
  intros i
  exact h i

end determine_card_numbers_l7_7562


namespace probability_of_less_than_one_third_l7_7861

theorem probability_of_less_than_one_third :
  (prob_of_interval (0 : ℝ) (1 / 2 : ℝ) (1 / 3 : ℝ) = 2 / 3) :=
sorry

end probability_of_less_than_one_third_l7_7861


namespace lcm_of_1_to_12_l7_7230

noncomputable def lcm_1_to_12 : ℕ := 2^3 * 3^2 * 5 * 7 * 11

theorem lcm_of_1_to_12 : lcm_1_to_12 = 27720 := by
  sorry

end lcm_of_1_to_12_l7_7230


namespace quadrilateral_inequality_l7_7297

theorem quadrilateral_inequality
  {A B C D : Type*}
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  {a b c d : ℝ}
  (h_AB_CD : a = d)
  (h_angle_ABC_BCD : ∠(A B C) > ∠(B C D)) :
  dist A C > dist B D := by
  sorry

end quadrilateral_inequality_l7_7297


namespace distance_A_A_double_prime_l7_7527

theorem distance_A_A_double_prime (A A' A'' O O' : Type) [AffineSpace ℝ A] [AffineSpace ℝ O] [AffineSpace ℝ O'] :
  let dist := distance (O xposition) (O' yposition)
  let h1 := midpoint O A A'
  let h2 := midpoint O' A' A''
  dist O O' = a → distance A A'' = 2 * a := by
  sorry

end distance_A_A_double_prime_l7_7527


namespace total_squares_l7_7347

def columns : ℕ := 5
def rows : ℕ := 6

def count_squares_of_size (n : ℕ) : ℕ := (columns - n + 1) * (rows - n + 1)

theorem total_squares : 
  (count_squares_of_size 1) + 
  (count_squares_of_size 2) + 
  (count_squares_of_size 3) + 
  (count_squares_of_size 4) = 68 := 
by 
  calc 
  (count_squares_of_size 1) + 
  (count_squares_of_size 2) + 
  (count_squares_of_size 3) + 
  (count_squares_of_size 4) 
  = 30 + 20 + 12 + 6 : by sorry
  ... = 68 : by sorry

end total_squares_l7_7347


namespace minimum_value_of_function_l7_7563

def function (x : ℝ) : ℝ :=
  2 + 4 * x + 1 / x

theorem minimum_value_of_function : ∃ x > 0, function x = 6 :=
by
  sorry

end minimum_value_of_function_l7_7563


namespace nate_reading_percentage_l7_7072

-- Given conditions
def total_pages := 400
def pages_to_read := 320

-- Calculate the number of pages he has already read
def pages_read := total_pages - pages_to_read

-- Prove the percentage of the book Nate has finished reading
theorem nate_reading_percentage : (pages_read / total_pages) * 100 = 20 := by
  sorry

end nate_reading_percentage_l7_7072


namespace transformations_correct_l7_7583

def y_sin_x := λ x : ℝ, Real.sin x
def y_cos_2x_plus_pi_over_6 := λ x : ℝ, Real.cos (2 * x + Real.pi / 6)

theorem transformations_correct :
  (∀ x : ℝ, y_sin_x x = Real.cos (x - Real.pi / 2)) ∧
  (∀ x : ℝ, y_cos_2x_plus_pi_over_6 x = Real.cos (2 * x + Real.pi / 6)) →
  (∀ x : ℝ, (λ x : ℝ, Real.cos (x - Real.pi / 2 - Real.pi / 3)) = λ x : ℝ, Real.cos (x + Real.pi / 6)) ∧
  (∀ x : ℝ, (λ x : ℝ, Real.cos (2 * (x + Real.pi / 3))) = λ x : ℝ, Real.cos (2 * x + Real.pi / 6)) ∧
  (∀ x : ℝ, (λ x : ℝ, Real.cos (2 * (x - Real.pi / 3))) = λ x : ℝ, Real.cos (2 * x - 2 * Real.pi / 3)) :=
sorry

end transformations_correct_l7_7583


namespace repeating_six_to_fraction_l7_7757

-- Define the infinite geometric series representation of 0.666...
def infinite_geometric_series (n : ℕ) : ℝ := 6 / (10 ^ n)

-- Define the sum of the infinite geometric series for 0.666...
def sum_infinite_geometric_series : ℝ :=
  ∑' n, infinite_geometric_series n

-- Formally state the problem to prove that 0.666... equals 2/3
theorem repeating_six_to_fraction : sum_infinite_geometric_series = 2 / 3 :=
by
  -- Proof goes here, but for now we use sorry to denote it will be completed later
  sorry

end repeating_six_to_fraction_l7_7757


namespace find_intersection_line_of_planes_l7_7412

-- Definitions needed for the problem
variables {P : Type} [plane P]
variables {A B : P} -- planes A and B
variables {t₁ t₂ : line P} -- first traces of the planes
variables {α β : ℝ} -- first angles of inclination of planes

-- The theorem statement we want to prove
theorem find_intersection_line_of_planes 
  (hA1 : A.first_trace = t₁) (hA2 : A.angle_of_inclination = α)
  (hB1 : B.first_trace = t₂) (hB2 : B.angle_of_inclination = β) :
  ∃ L : line P, (L.intersects A) ∧ (L.intersects B) := 
sorry

end find_intersection_line_of_planes_l7_7412


namespace widescreen_horizontal_length_correct_l7_7960

noncomputable def widescreen_horizontal_length (aspect_ratio_width aspect_ratio_height diagonal_length : ℝ) : ℝ :=
  let hypot := real.sqrt (aspect_ratio_width^2 + aspect_ratio_height^2)
  aspect_ratio_width * diagonal_length / hypot

theorem widescreen_horizontal_length_correct : widescreen_horizontal_length 16 9 50 ≈ 43.56 :=
by {
  sorry
}

end widescreen_horizontal_length_correct_l7_7960


namespace simple_interest_calculation_l7_7453

-- Define the known quantities
def principal : ℕ := 400
def rate_of_interest : ℕ := 15
def time : ℕ := 2

-- Define the formula for simple interest
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Statement to be proved
theorem simple_interest_calculation :
  simple_interest principal rate_of_interest time = 60 :=
by
  -- This space is used for the proof, We assume the user will complete it
  sorry

end simple_interest_calculation_l7_7453


namespace current_time_is_6_10_pm_l7_7964

noncomputable def current_time_after_40_days : String :=
let sunset_march_1 := 18 * 60 -- 6 PM in minutes
let delayed_minutes := 1.2
let days_elapsed := 40
let total_delay := delayed_minutes * days_elapsed
let new_sunset_time := sunset_march_1 + total_delay
let sunset_in_38_minutes := new_sunset_time - 38
let hours := sunset_in_38_minutes / 60
let minutes := sunset_in_38_minutes % 60
hours.toString ++ ":" ++ minutes.toString

theorem current_time_is_6_10_pm :
current_time_after_40_days = "18:10" := 
by
  sorry

end current_time_is_6_10_pm_l7_7964


namespace total_students_l7_7892

theorem total_students (x y : ℕ) : 
  let n := x + 900 + y in
  370 - 120 - 100 = 1 / 6 * 900 → 
  370 / n = 1 / 6 → 
  n = 2220 :=
by
  intros
  sorry

end total_students_l7_7892


namespace collinear_vectors_y_value_l7_7428

-- Define the vectors as pairs of real numbers
variables (a b : ℝ × ℝ)

-- Define the collinearity condition
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Given vectors a and b, prove that y = -2 if they are collinear
theorem collinear_vectors_y_value :
  collinear (-3, 1) (6, y) → y = -2 :=
begin
  unfold collinear,
  intro h,
  simp at h,
  exact Eq.trans h (Eq.refl (-2))
end

end collinear_vectors_y_value_l7_7428


namespace circles_intersect_l7_7388

def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6 * x - 4 * y = 0

theorem circles_intersect:
  let C1_center := (0, 0)
  let C2_center := (-3, 2)
  let r1 := 2
  let r2 := sqrt 13
  let d := sqrt ((0 + 3)^2 + (0 - 2)^2)
  d = sqrt 13 ∧ sqrt 13 - 2 < sqrt 13 ∧ sqrt 13 < sqrt 13 + 2
→ ∃ x y : ℝ, circle1 x y ∧ circle2 x y :=
by
  sorry

end circles_intersect_l7_7388


namespace range_of_alpha_for_intersection_range_of_x_plus_y_l7_7905

-- Problem 1
theorem range_of_alpha_for_intersection (P : ℝ × ℝ) (hP : P = (-1, 0)) 
  (C : ℝ → ℝ → Prop) (hC : ∀ x y, C x y ↔ x^2 + y^2 - 6 * x + 1 = 0) :
  ∀ α : ℝ, (∃ t : ℝ, let x := -1 + t * cos α, y := t * sin α in C x y) →
  (α ≥ 0 ∧ α < π/4) ∨ (α ≥ 3*π/4 ∧ α < π) :=
by sorry

-- Problem 2
theorem range_of_x_plus_y (C : ℝ → ℝ → Prop) (hC : ∀ x y, C x y ↔ (x-3)^2 + y^2 = 8) :
  ∀ x y, C x y → -1 ≤ x + y ∧ x + y ≤ 7 :=
by sorry

end range_of_alpha_for_intersection_range_of_x_plus_y_l7_7905


namespace crease_length_l7_7667

theorem crease_length (AB BC AC : ℝ) (h1 : AB = 6) (h2 : BC = 8) (h3 : AC = 10) (right_triangle : AB^2 + BC^2 = AC^2) :
  let D := (3 : ℝ, 0 : ℝ) in
  let C := (0 : ℝ, 8 : ℝ) in
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = 73 :=
by {
  -- We need to show (3 - 0)^2 + (0 - 8)^2 = 73
  sorry
}

end crease_length_l7_7667


namespace correct_options_l7_7799

-- Definitions for the plane vectors and their magnitudes
variables (m n : EuclideanSpace ℝ (Fin 2)) (t : ℝ)
axiom mag_m : ‖ m ‖ = 1
axiom mag_n : ‖ n ‖ = 1
axiom inequality_condition : ∀ t : ℝ, ‖ m - (1 / 2) • n ‖ ≤ ‖ m + t • n ‖

-- Theorem: correct options (A and D) from the problem
theorem correct_options (m n : EuclideanSpace ℝ (Fin 2)) (t : ℝ) (h1 : ‖ m ‖ = 1) (h2 : ‖ n ‖ = 1) 
  (h3 : ∀ t : ℝ, ‖ m - (1 / 2) • n ‖ ≤ ‖ m + t • n ‖) :
  angle m n = (Real.pi / 3) ∧ (orthogonalProjection (ℝ ∙ (m + n)) m = (1 / 2) • (m + n)) :=
by
  sorry

end correct_options_l7_7799


namespace odell_and_kershaw_meetings_l7_7075

-- Define the conditions.
def odell_speed : ℝ := 250  -- Odell's speed in meters per minute.
def kershaw_speed : ℝ := 300  -- Kershaw's speed in meters per minute.
def odell_radius : ℝ := 50  -- Radius of the inner lane in meters.
def kershaw_radius : ℝ := 60  -- Radius of the outer lane in meters.
def run_time : ℝ := 30  -- Running time in minutes.

-- Circumferences of the tracks.
def odell_circumference : ℝ := 2 * Real.pi * odell_radius
def kershaw_circumference : ℝ := 2 * Real.pi * kershaw_radius

-- Angular speeds in radians per minute.
def odell_angular_speed : ℝ := (odell_speed / odell_circumference) * 2 * Real.pi
def kershaw_angular_speed : ℝ := (kershaw_speed / kershaw_circumference) * 2 * Real.pi

-- Relative angular speed in radians per minute.
def relative_angular_speed : ℝ := odell_angular_speed + kershaw_angular_speed

-- Time to meet once in minutes.
def time_to_meet : ℝ := (2 * Real.pi) / relative_angular_speed

-- Number of meetings in 30 minutes.
def number_of_meetings : ℝ := run_time / time_to_meet

-- Final number of meetings. This is what we want to prove.
def result : ℕ := Real.floor number_of_meetings

theorem odell_and_kershaw_meetings : result = 47 := by
  sorry

end odell_and_kershaw_meetings_l7_7075


namespace sum_slope_y_intercept_l7_7907

/-
Given:
- Points A, B, and C with coordinates A(0,6), B(0,0), and C(8,0).
- Point D is the midpoint of segment AB, hence has coordinates (0,3).

Goal:
- Prove the sum of the slope and y-intercept of the line passing through points C and D is 21/8.
-/

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def slope (P Q : ℝ × ℝ) : ℝ := 
  (Q.2 - P.2) / (Q.1 - P.1)

def y_intercept (m : ℝ) (P : ℝ × ℝ) : ℝ := 
  P.2 - m * P.1

theorem sum_slope_y_intercept :
  let A := (0, 6)
  let B := (0, 0)
  let C := (8, 0)
  let D := midpoint A B
  let m := slope C D
  let b := y_intercept m D
  (m + b) = 21 / 8 :=
by 
  sorry

end sum_slope_y_intercept_l7_7907


namespace segment_length_eq_ten_l7_7211

theorem segment_length_eq_ten (x : ℝ) (h : |x - 3| = 5) : |8 - (-2)| = 10 :=
by {
  sorry
}

end segment_length_eq_ten_l7_7211


namespace hexagon_sides_length_5_l7_7718

-- Define the necessary conditions
variables {K L M N O P : Type} [convex_space K] [convex_space L] [convex_space M] [convex_space N] [convex_space O] [convex_space P]

def is_convex_hexagon (hexagon : list (convex_space ℕ)) : Prop :=
  hexagon.length = 6

def side_length (side : convex_space ℕ) : ℕ :=
  match side with
  | K => 7
  | L => 7
  | M => 7
  | N => 5
  | O => 5
  | P => 7
  end

def perimeter (hexagon : list (convex_space ℕ)) : ℕ :=
  hexagon.foldr (+) 0 (hexagon.map side_length)

-- Define the hexagon
def KLMNOP_hexagon := [K, L, M, N, O, P]

-- Define how many sides measure 5 units
def count_sides_of_length_5 (hexagon : list (convex_space ℕ)) : ℕ :=
  hexagon.count (λ side, side_length side = 5)

-- Define the proof problem
theorem hexagon_sides_length_5 :
  is_convex_hexagon KLMNOP_hexagon →
  side_length K = 7 →
  side_length L = 7 →
  side_length M = 7 →
  side_length N = 5 →
  side_length O = 5 →
  side_length P = 7 →
  perimeter KLMNOP_hexagon = 38 →
  count_sides_of_length_5 KLMNOP_hexagon = 2 := by
  sorry

end hexagon_sides_length_5_l7_7718


namespace relationship_x2_ax_bx_l7_7939

variable {x a b : ℝ}

theorem relationship_x2_ax_bx (h1 : x < a) (h2 : a < 0) (h3 : b > 0) : x^2 > ax ∧ ax > bx :=
by
  sorry

end relationship_x2_ax_bx_l7_7939


namespace repeating_decimal_sum_l7_7701

theorem repeating_decimal_sum :
  (0.\overline{3} = 1 / 3) → (0.\overline{6} = 2 / 3) → (0.\overline{3} + 0.\overline{6} = 1) :=
by
  sorry

end repeating_decimal_sum_l7_7701


namespace find_shortage_l7_7630

def total_capacity (T : ℝ) : Prop :=
  0.70 * T = 14

def normal_level (normal : ℝ) : Prop :=
  normal = 14 / 2

def capacity_shortage (T : ℝ) (normal : ℝ) : Prop :=
  T - normal = 13

theorem find_shortage (T : ℝ) (normal : ℝ) : 
  total_capacity T →
  normal_level normal →
  capacity_shortage T normal :=
by
  sorry

end find_shortage_l7_7630


namespace fraction_of_C_grades_l7_7467

theorem fraction_of_C_grades (total_students : ℕ) (A_fraction B_fraction D_count : ℕ)
  (hA : A_fraction * total_students = 160) 
  (hB : B_fraction * total_students = 200)
  (hD : D_count = 40) 
  (h_total : total_students = 800) :
  (total_students - (160 + 200 + D_count)) / total_students = 1 / 2 := 
by
  have h1 : total_students - (160 + 200 + 40) = 400 := sorry
  have h2 : 400 / total_students = 1 / 2 := sorry
  exact h2

end fraction_of_C_grades_l7_7467


namespace good_numbers_count_20_l7_7775

def isGoodNumber (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = a + b + a * b

def numGoodNumbers (N : ℕ) : ℕ :=
  (finset.range N).filter isGoodNumber |>.card

theorem good_numbers_count_20 : numGoodNumbers 20 = 12 := by
  sorry

end good_numbers_count_20_l7_7775


namespace value_of_y_l7_7368

theorem value_of_y (y : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) :
  a = 10^3 → b = 10^4 → 
  a^y * 10^(3 * y) = (b^4) → 
  y = 8 / 3 :=
by 
  intro ha hb hc
  rw [ha, hb] at hc
  sorry

end value_of_y_l7_7368


namespace car_selection_proportion_l7_7980

def production_volume_emgrand : ℕ := 1600
def production_volume_king_kong : ℕ := 6000
def production_volume_freedom_ship : ℕ := 2000
def total_selected_cars : ℕ := 48

theorem car_selection_proportion :
  (8, 30, 10) = (
    total_selected_cars * production_volume_emgrand /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship),
    total_selected_cars * production_volume_king_kong /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship),
    total_selected_cars * production_volume_freedom_ship /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship)
  ) :=
by sorry

end car_selection_proportion_l7_7980


namespace possible_number_of_classmates_l7_7123

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l7_7123


namespace max_geometric_sequence_sum_l7_7994

theorem max_geometric_sequence_sum (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a * b * c = 216) (h4 : ∃ r : ℕ, b = a * r ∧ c = b * r) : 
  a + b + c ≤ 43 :=
sorry

end max_geometric_sequence_sum_l7_7994


namespace possible_classmates_l7_7137

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l7_7137


namespace cards_left_l7_7920
noncomputable section

def initial_cards : ℕ := 676
def bought_cards : ℕ := 224

theorem cards_left : initial_cards - bought_cards = 452 := 
by
  sorry

end cards_left_l7_7920


namespace proof_at_eq_l7_7418

noncomputable def f (x a : ℝ) := Real.exp x - a * x + a
def x1 := sorry -- Placeholder for x1 where f(x1, a) = 0
def x2 := sorry -- Placeholder for x2 where f(x2, a) = 0
def t (x1 x2 : ℝ) := Real.sqrt ((x2 - 1) / (x1 - 1))

theorem proof_at_eq (a x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) (hx1ltx2 : x1 < x2) :
  at - (a + t x1 x2) = 1 :=
sorry

end proof_at_eq_l7_7418


namespace cake_eating_classmates_l7_7112

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l7_7112


namespace number_as_A_times_10_pow_N_integer_l7_7285

theorem number_as_A_times_10_pow_N_integer (A : ℝ) (N : ℝ) (hA1 : 1 ≤ A) (hA2 : A < 10) (hN : A * 10^N > 10) : ∃ (n : ℤ), N = n := 
sorry

end number_as_A_times_10_pow_N_integer_l7_7285


namespace fishing_line_sections_l7_7030

theorem fishing_line_sections (reels : ℕ) (length_per_reel : ℕ) (section_length : ℕ)
    (h_reels : reels = 3) (h_length_per_reel : length_per_reel = 100) (h_section_length : section_length = 10) :
    (reels * length_per_reel) / section_length = 30 := 
by
  rw [h_reels, h_length_per_reel, h_section_length]
  norm_num

end fishing_line_sections_l7_7030


namespace ciphertext_to_plaintext_l7_7171

theorem ciphertext_to_plaintext :
  ∃ (a b c d : ℕ), (a + 2 * b = 14) ∧ (2 * b + c = 9) ∧ (2 * c + 3 * d = 23) ∧ (4 * d = 28) ∧ a = 6 ∧ b = 4 ∧ c = 1 ∧ d = 7 :=
by 
  sorry

end ciphertext_to_plaintext_l7_7171


namespace flattest_ellipse_is_B_l7_7726

-- Definitions for the given ellipses
def ellipseA : Prop := ∀ (x y : ℝ), (x^2 / 16 + y^2 / 12 = 1)
def ellipseB : Prop := ∀ (x y : ℝ), (x^2 / 4 + y^2 = 1)
def ellipseC : Prop := ∀ (x y : ℝ), (x^2 / 6 + y^2 / 3 = 1)
def ellipseD : Prop := ∀ (x y : ℝ), (x^2 / 9 + y^2 / 8 = 1)

-- The proof to show that ellipseB is the flattest
theorem flattest_ellipse_is_B : ellipseB := by
  sorry

end flattest_ellipse_is_B_l7_7726


namespace lcm_of_1_to_12_l7_7229

noncomputable def lcm_1_to_12 : ℕ := 2^3 * 3^2 * 5 * 7 * 11

theorem lcm_of_1_to_12 : lcm_1_to_12 = 27720 := by
  sorry

end lcm_of_1_to_12_l7_7229


namespace part_a_l7_7365

def L (n : ℕ) : ℕ :=
  {a : ℕ | 1 ≤ a ∧ a ≤ n ∧ n ∣ a^n - 1}.toFinset.card

def T (n : ℕ) : ℕ :=
  let primes := (n.factorization.keys.toList)
  primes.foldr (fun p acc => acc * (p - 1)) 1

theorem part_a (n : ℕ) : n ∣ L n * T n := by
  sorry

end part_a_l7_7365


namespace pascal_triangle_sum_l7_7976

open Nat

open BigOperators

theorem pascal_triangle_sum :
  let d_i : ℕ → ℕ := λ i, Nat.choose 2010 i
  let e_i : ℕ → ℕ := λ i, Nat.choose 2011 i
  let f_i : ℕ → ℕ := λ i, Nat.choose 2012 i
  (∑ i in Finset.range 2012, e_i i / f_i i) - 
  (∑ i in Finset.range 2011, d_i i / e_i i) = 1 / 2 :=
by
  sorry

end pascal_triangle_sum_l7_7976


namespace possible_classmates_l7_7134

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l7_7134


namespace count_integers_satisfying_inequality_l7_7839

theorem count_integers_satisfying_inequality :
  {n : ℤ | (n + 5) * (n - 9) ≤ 0}.finite.to_finset.card = 15 := 
by
  sorry

end count_integers_satisfying_inequality_l7_7839


namespace no_such_continuous_function_exists_l7_7969

theorem no_such_continuous_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (Continuous f) ∧ ∀ x : ℝ, ((∃ q : ℚ, f x = q) ↔ ∀ q' : ℚ, f (x + 1) ≠ q') :=
sorry

end no_such_continuous_function_exists_l7_7969


namespace repeating_decimal_to_fraction_l7_7754

theorem repeating_decimal_to_fraction : (let a := (6 : Real) / 10 in
                                         let r := (1 : Real) / 10 in
                                         ∑' n : ℕ, a * r^n) = (2 : Real) / 3 :=
by
  sorry

end repeating_decimal_to_fraction_l7_7754


namespace lcm_1_to_12_l7_7219

theorem lcm_1_to_12 : nat.lcm (list.range (12 + 1)) = 27720 :=
begin
  sorry
end

end lcm_1_to_12_l7_7219


namespace modulus_of_z_l7_7350

-- Define the complex number z
def z : ℂ := -3 + (11 / 3) * complex.I

-- Theorem stating that the modulus of z is equal to the given expression
theorem modulus_of_z : complex.abs z = real.sqrt 202 / 3 := by
  sorry

end modulus_of_z_l7_7350


namespace cost_to_produce_program_l7_7276

theorem cost_to_produce_program
  (advertisement_revenue : ℝ)
  (number_of_copies : ℝ)
  (price_per_copy : ℝ)
  (desired_profit : ℝ)
  (total_revenue : ℝ)
  (revenue_from_sales : ℝ)
  (cost_to_produce : ℝ) :
  advertisement_revenue = 15000 →
  number_of_copies = 35000 →
  price_per_copy = 0.5 →
  desired_profit = 8000 →
  total_revenue = advertisement_revenue + desired_profit →
  revenue_from_sales = number_of_copies * price_per_copy →
  total_revenue = revenue_from_sales + cost_to_produce →
  cost_to_produce = 5500 :=
by
  sorry

end cost_to_produce_program_l7_7276


namespace value_of_expression_l7_7442

theorem value_of_expression (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  simp
  rfl

end value_of_expression_l7_7442


namespace comparison_inequalities_l7_7374

open Real

theorem comparison_inequalities
  (m : ℝ) (h1 : 3 ^ m = Real.exp 1) 
  (a : ℝ) (h2 : a = cos m) 
  (b : ℝ) (h3 : b = 1 - 1/2 * m^2)
  (c : ℝ) (h4 : c = sin m / m) :
  c > a ∧ a > b := by
  sorry

end comparison_inequalities_l7_7374


namespace train_speed_l7_7639

theorem train_speed (d t s : ℝ) (h1 : d = 320) (h2 : t = 6) (h3 : s = 53.33) :
  s = d / t :=
by
  rw [h1, h2]
  sorry

end train_speed_l7_7639


namespace quadratic_solution_set_equiv_l7_7405

theorem quadratic_solution_set_equiv (a b c : ℝ) 
  (H1: a < 0)
  (H2: −b / (2 * a) = (3 + 6) / 2)
  (H3: c / a = 3 * 6)
  : ∀ x, cx^2 + bx + a < 0 ↔ x < 1/6 ∨ x > 1/3 := 
    sorry

end quadratic_solution_set_equiv_l7_7405


namespace solution_set_cannot_be_134_l7_7823

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a ^ |x - b|

variables (a m n p b : ℝ)
variables (ha : a > 0) (ha_ne : a ≠ 1) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0)

theorem solution_set_cannot_be_134 (h : ∀ x, m * (f x a b)^2 + n * (f x a b) + p = 0) :
  ¬(set_of x, h x = {1, 3, 4}) :=
sorry

end solution_set_cannot_be_134_l7_7823


namespace man_and_son_together_days_l7_7284

noncomputable def man_days : ℝ := 7
noncomputable def son_days : ℝ := 5.25
noncomputable def combined_days : ℝ := man_days * son_days / (man_days + son_days)

theorem man_and_son_together_days :
  combined_days = 7 / 5 :=
by
  sorry

end man_and_son_together_days_l7_7284


namespace ln_inequality_l7_7826

theorem ln_inequality (x : ℝ) (hx : 0 < x) : log x ≤ x - 1 :=
sorry

end ln_inequality_l7_7826


namespace curve_equation_of_Γ_minimum_distance_to_origin_l7_7918

-- Problem (I) statement
theorem curve_equation_of_Γ (P B T : ℝ × ℝ)
  (hP_cond : ∃ xP yP : ℝ, P = (xP, yP) ∧ (xP + 1)^2 + yP^2 = 12)
  (hB : B = (1, 0))
  (hT_cond : ∃ xT yT : ℝ, T = (xT, yT) ∧ 
    (dist B T) = (dist P T) ∧
    ∃ xA yA : ℝ, let A := (xA, yA) in 
    (dist A T) + (dist B T) = (dist A T) + (dist P T) ∧ 
    xA = -1 ∧ yA = 0 ∧ ∃ c : ℝ, 4 ≤ c ∧ sqrt 12 = 2 * c ∧
    (2 * (dist A T))^2 = (2 * sqrt 3)^2) :
  ∀ x y : ℝ, (x, y) ∈ set_of (λ (p : ℝ × ℝ), (fst p)^2 / 3 + (snd p)^2 / 2 = 1) := sorry

-- Problem (II) statement
theorem minimum_distance_to_origin (M N H : ℝ × ℝ)
  (hM_cond : ∃ xM yM : ℝ, M = (xM, yM) ∧ (fst M)^2 / 3 + (snd M)^2 / 2 = 1)
  (hN_cond : ∃ xN yN : ℝ, N = (xN, yN) ∧ (fst N)^2 / 3 + (snd N)^2 / 2 = 1)
  (hH_cond : ∃ xH yH : ℝ, H = (xH, yH) ∧ xH^2 + yH^2 = 1 ∧ fst H = (fst M + fst N) / 2 ∧ snd H = (snd M + snd N) / 2) :
  dist (0, 0) ((fst M + fst N) / 2, (snd M + snd N) / 2) = 2 * sqrt(6) / 5 := sorry

end curve_equation_of_Γ_minimum_distance_to_origin_l7_7918


namespace original_triangle_area_l7_7164

-- Define the conditions
def dimensions_quadrupled (original_area new_area : ℝ) : Prop :=
  4^2 * original_area = new_area

-- Define the statement to be proved
theorem original_triangle_area {new_area : ℝ} (h : new_area = 64) :
  ∃ (original_area : ℝ), dimensions_quadrupled original_area new_area ∧ original_area = 4 :=
by
  sorry

end original_triangle_area_l7_7164


namespace equilibrium_possible_l7_7261

theorem equilibrium_possible (n : ℕ) : (∃ k : ℕ, 4 * k = n) ∨ (∃ k : ℕ, 4 * k + 3 = n) ↔
  (∃ S1 S2 : Finset ℕ, S1 ∪ S2 = Finset.range (n+1) ∧
                     S1 ∩ S2 = ∅ ∧
                     S1.sum id = S2.sum id) := 
sorry

end equilibrium_possible_l7_7261


namespace inverse_p_l7_7421

variables (x : ℝ)

def p : Prop := x < -3 → x^2 - 2*x - 8 > 0

theorem inverse_p : p → (x ≥ -3 → x^2 - 2*x - 8 ≤ 0) :=
by
  intro h
  sorry

end inverse_p_l7_7421


namespace parabola_intersection_sum_l7_7367

noncomputable theory
open BigOperators

theorem parabola_intersection_sum :
  let roots_x := {x : ℝ | ∃ y : ℝ, (y = (x + 2)^2) ∧ (x + 5 = (y - 4)^2)} in
  let roots_y := {y : ℝ | ∃ x : ℝ, (y = (x + 2)^2) ∧ (x + 5 = (y - 4)^2)} in
  (∑ (x ∈ roots_x), x) + (∑ (y ∈ roots_y), y) = 8 :=
sorry

end parabola_intersection_sum_l7_7367


namespace ratio_of_areas_l7_7020

-- Define the problem conditions
variables (d1 d2 : ℝ) (α : ℝ)
-- Define the areas of the quadrilaterals
def S_new := (d1 * d2 * (Real.cos α)^2) / (Real.sin α)
def S_ABCD := (1 / 2) * d1 * d2 * (Real.sin α)

-- Prove the ratio of the areas
theorem ratio_of_areas (hα : 0 < α ∧ α < Real.pi / 2) :
  (S_new d1 d2 α) / (S_ABCD d1 d2 α) = 2 * (Real.cot α)^2 := 
by
  sorry

end ratio_of_areas_l7_7020


namespace slower_pump_time_l7_7595

theorem slower_pump_time (R : ℝ) (hours : ℝ) (combined_rate : ℝ) (faster_rate_adj : ℝ) (time_both : ℝ) :
  (combined_rate = R * (1 + faster_rate_adj)) →
  (faster_rate_adj = 1.5) →
  (time_both = 5) →
  (combined_rate * time_both = 1) →
  (hours = 1 / R) →
  hours = 12.5 :=
by
  sorry

end slower_pump_time_l7_7595


namespace calculate_AB_l7_7529

-- Definitions of our constants and variables
def A_B_C_D_straight_line (A B C D : ℝ) : Prop :=
  A < B ∧ B < C ∧ C < D

def AB_CD_eq_x (A B C D x : ℝ) : Prop :=
  abs (B - A) = x ∧ abs (D - C) = x

def BC_eq_16 (B C : ℝ) : Prop :=
  abs (C - B) = 16

def point_E_off_line (E B C : ℝ) : Prop :=
  abs (B - E) = 14 ∧ abs (C - E) = 14

def perimeter_relation (A B C D E : ℝ) : Prop :=
  let AED_perimeter := 2 * real.sqrt ((B - A + 8)^2 + 132) + 2 * abs (B - A) + 16
  let BEC_perimeter := 44
  AED_perimeter = BEC_perimeter + 10

def correct_answer := 55 / 18

-- The main theorem to be proved
theorem calculate_AB (A B C D E x : ℝ) :
  A_B_C_D_straight_line A B C D →
  AB_CD_eq_x A B C D x →
  BC_eq_16 B C →
  point_E_off_line E B C →
  perimeter_relation A B C D E →
  abs (B - A) = correct_answer :=
sorry

end calculate_AB_l7_7529


namespace find_equation_of_line_midpoint_find_equation_of_line_vector_l7_7665

-- Definition for Problem 1
def equation_of_line_midpoint (x y : ℝ) : Prop :=
  ∃ l : ℝ → ℝ, (l x = 0 ∧ l 0 = y ∧ (x / (-6) + y / 2 = 1) ∧ l (-3) = 1)

-- Proof Statement for Problem 1
theorem find_equation_of_line_midpoint : equation_of_line_midpoint (-6) 2 :=
sorry

-- Definition for Problem 2
def equation_of_line_vector (x y : ℝ) : Prop :=
  ∃ l : ℝ → ℝ, (l x = 0 ∧ l 0 = y ∧ (y - 1) / (-1) = (x + 3) / (-6) ∧ l (-3) = 1)

-- Proof Statement for Problem 2
theorem find_equation_of_line_vector : equation_of_line_vector (-9) (3 / 2) :=
sorry

end find_equation_of_line_midpoint_find_equation_of_line_vector_l7_7665


namespace cake_eating_classmates_l7_7111

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l7_7111


namespace gift_card_remaining_l7_7067

theorem gift_card_remaining (initial_amount : ℕ) (half_monday : ℕ) (quarter_tuesday : ℕ) : 
  initial_amount = 200 → 
  half_monday = initial_amount / 2 →
  quarter_tuesday = (initial_amount - half_monday) / 4 →
  initial_amount - half_monday - quarter_tuesday = 75 :=
by
  intros h_init h_half h_quarter
  rw [h_init, h_half, h_quarter]
  sorry

end gift_card_remaining_l7_7067


namespace minimum_value_problem_l7_7407

-- Define the inequality and the solution set conditions
def inequality_solution_set (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 < 0

-- Define the interval notation for A
def interval_A (m : ℝ) : Set ℝ := {x | 1 - m < x ∧ x < 1 + m}

-- Given conditions
def given_conditions (m : ℝ) : Prop := 0 < m ∧ m < 1

-- Main proof problem
theorem minimum_value_problem (m : ℝ) (h : given_conditions m) :
  @Set.sSubset ℝ (interval_A m) ∅ ∧ 
  ∃ a b : ℝ, a = 1 - m ∧ b = 1 + m ∧
  (∀ a b, a = 1 - m ∧ b = 1 + m → (1 / (8 * a + 2 * b) - 1 / (3 * a - 3 * b)) = 2 / 5) := 
begin
  sorry
end

end minimum_value_problem_l7_7407


namespace triangle_area_l7_7891

noncomputable def sin_30 : ℝ := 1 / 2 -- Using sine of 30 degrees

theorem triangle_area (B : ℝ) (AB : ℝ) (AC : ℝ) 
  (hB : B = π / 6) (hAB : AB = sqrt 3) (hAC : AC = 1) :
  (1 / 2 * AB * AC * sin(B) = sqrt 3 / 4 ∨ 1 / 2 * AB * AC * sin(B) = sqrt 3 / 2) := by
{
  rw [hB, hAB, hAC],
  -- Need to show that the area satisfies one of the given conditions
  rw [sin_30],
  -- The value simplifies to sqrt(3)/4 or sqrt(3)/2
  sorry
}

end triangle_area_l7_7891


namespace lcm_ge_max_A_div_B_l7_7053

-- Definitions and conditions
variables {n A B : ℕ} (a : Fin n → ℕ)
hypothesis h1 : ∀ i, 1 ≤ a i
hypothesis h2 : ∀ i, 1 ≤ i ∧ i ≤ n → (a i) ≥ A
hypothesis h3 : ∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → Nat.gcd (a i) (a j) ≤ B

-- The proof statement
theorem lcm_ge_max_A_div_B (A B : ℕ) (a : Fin n → ℕ) (h1 : ∀ i, 1 ≤ a i)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → (a i) ≥ A) 
  (h3 : ∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → Nat.gcd (a i) (a j) ≤ B) : 
  Nat.lcm (Fin n) a ≥ (Fin n).sup (λ i => A ^ i / B ^ (i * (i - 1) / 2)) := 
begin
  sorry
end

end lcm_ge_max_A_div_B_l7_7053


namespace min_forget_all_three_l7_7977

theorem min_forget_all_three (total_students students_forgot_gloves students_forgot_scarves students_forgot_hats : ℕ) (h_total : total_students = 60) (h_gloves : students_forgot_gloves = 55) (h_scarves : students_forgot_scarves = 52) (h_hats : students_forgot_hats = 50) :
  ∃ min_students_forget_three, min_students_forget_three = total_students - (total_students - students_forgot_gloves + total_students - students_forgot_scarves + total_students - students_forgot_hats) :=
by
  use 37
  sorry

end min_forget_all_three_l7_7977


namespace problem_I_l7_7270

theorem problem_I (x m : ℝ) (h1 : |x - m| < 1) (h2 : (1/3 : ℝ) < x ∧ x < (1/2 : ℝ)) : (-1/2 : ℝ) ≤ m ∧ m ≤ (4/3 : ℝ) :=
sorry

end problem_I_l7_7270


namespace population_present_l7_7567

theorem population_present (P : ℝ) (h : P * (1.1)^3 = 79860) : P = 60000 :=
sorry

end population_present_l7_7567


namespace geometric_sequence_b_l7_7380

variables {q : ℝ} [hq : fact (q ≠ 1)] [hq0 : fact (q ≠ 0)]
noncomputable def a (n : ℕ) := q ^ (n - 1)
noncomputable def b (n : ℕ) := a (n + 1) - a n

theorem geometric_sequence_b :
  (∀ n : ℕ, b n = (q - 1) * q ^ (n - 1)) :=
begin
  assume n,
  calc b n = a (n + 1) - a n : by rfl
       ... = q ^ n - q ^ (n - 1) : by rfl
       ... = q ^ (n - 1) * (q - 1) : by { ring }
end

end geometric_sequence_b_l7_7380


namespace ratio_H_OH_l7_7597

-- Definitions of conditions
def standard_temp_pressure : Prop := true
def product_constant (H OH : ℝ) : Prop := H * OH = 10^(-14)
def pH (H : ℝ) : ℝ := - (Real.log H / Real.log 10)
def pH_range (pH_value : ℝ) : Prop := 7.35 ≤ pH_value ∧ pH_value ≤ 7.45
def log_values : Prop := Real.log 2 ≈ 0.30 ∧ Real.log 3 ≈ 0.48

-- The main theorem
theorem ratio_H_OH (H OH : ℝ) (h1 : standard_temp_pressure) (h2 : product_constant H OH) (h3 : pH_range (pH H)) (h4 : log_values) : 
  H / OH = 1 / 6 :=
sorry

end ratio_H_OH_l7_7597


namespace range_x_plus_y_l7_7390

theorem range_x_plus_y (x y: ℝ) (h: x^2 + y^2 - 4 * x + 3 = 0) : 
  2 - Real.sqrt 2 ≤ x + y ∧ x + y ≤ 2 + Real.sqrt 2 :=
by 
  sorry

end range_x_plus_y_l7_7390


namespace red_balls_estimate_l7_7012

/-- There are several red balls and 4 black balls in a bag.
Each ball is identical except for color.
A ball is drawn and put back into the bag. This process is repeated 100 times.
Among those 100 draws, 40 times a black ball is drawn.
Prove that the number of red balls (x) is 6. -/
theorem red_balls_estimate (x : ℕ) (h_condition : (4 / (4 + x) = 40 / 100)) : x = 6 :=
by
    sorry

end red_balls_estimate_l7_7012


namespace count_sum_of_primes_58_l7_7474

def is_prime (n : ℕ) : Prop := Nat.Prime n

def count_sum_of_primes (n : ℕ) : ℕ :=
  (List.filter (λ (p : ℕ × ℕ), is_prime p.fst ∧ is_prime p.snd ∧ p.fst + p.snd = n)
     ((List.range (n + 1)).product (List.range (n + 1)))).length

theorem count_sum_of_primes_58 : count_sum_of_primes 58 = 3 :=
by
  sorry

end count_sum_of_primes_58_l7_7474


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_solve_quadratic_5_solve_quadratic_6_l7_7155

-- Problem 1: 5x² = 40x
theorem solve_quadratic_1 (x : ℝ) : 5 * x^2 = 40 * x ↔ (x = 0 ∨ x = 8) :=
by sorry

-- Problem 2: 25/9 x² = 100
theorem solve_quadratic_2 (x : ℝ) : (25 / 9) * x^2 = 100 ↔ (x = 6 ∨ x = -6) :=
by sorry

-- Problem 3: 10x = x² + 21
theorem solve_quadratic_3 (x : ℝ) : 10 * x = x^2 + 21 ↔ (x = 7 ∨ x = 3) :=
by sorry

-- Problem 4: x² = 12x + 288
theorem solve_quadratic_4 (x : ℝ) : x^2 = 12 * x + 288 ↔ (x = 24 ∨ x = -12) :=
by sorry

-- Problem 5: x² + 20 1/4 = 11 1/4 x
theorem solve_quadratic_5 (x : ℝ) : x^2 + 81 / 4 = 45 / 4 * x ↔ (x = 9 / 4 ∨ x = 9) :=
by sorry

-- Problem 6: 1/12 x² + 7/12 x = 19
theorem solve_quadratic_6 (x : ℝ) : (1 / 12) * x^2 + (7 / 12) * x = 19 ↔ (x = 12 ∨ x = -19) :=
by sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_solve_quadratic_5_solve_quadratic_6_l7_7155


namespace correct_calculation_l7_7248

theorem correct_calculation (a : ℕ) :
  ¬ (a^3 + a^4 = a^7) ∧
  ¬ (2 * a - a = 2) ∧
  2 * a + a = 3 * a ∧
  ¬ (a^4 - a^3 = a) :=
by
  sorry

end correct_calculation_l7_7248


namespace number_of_red_balls_l7_7009

noncomputable def red_balls (n_black n_red draws black_draws : ℕ) : ℕ :=
  if black_draws = (draws * n_black) / (n_black + n_red) then n_red else sorry

theorem number_of_red_balls :
  ∀ (n_black draws black_draws : ℕ),
    n_black = 4 →
    draws = 100 →
    black_draws = 40 →
    red_balls n_black (red_balls 4 6 100 40) 100 40 = 6 :=
by
  intros n_black draws black_draws h_black h_draws h_blackdraws
  dsimp [red_balls]
  rw [h_black, h_draws, h_blackdraws]
  norm_num
  sorry

end number_of_red_balls_l7_7009


namespace simplest_quadratic_radical_l7_7615

theorem simplest_quadratic_radical :
  let a := Real.sqrt 12
  let b := Real.sqrt (2 / 3)
  let c := Real.sqrt 0.3
  let d := Real.sqrt 7
  d < a ∧ d < b ∧ d < c := 
by {
  -- the proof steps will go here, but we use sorry for now
  sorry
}

end simplest_quadratic_radical_l7_7615


namespace max_f_in_interval_l7_7788

noncomputable def f (x : ℚ) : ℚ :=
  if x.denom ≠ 0 ∧ x.num.gcd x.denom = 1 then 
    (x.num + 1) / x.denom 
  else 0

theorem max_f_in_interval : 
  ∃ x ∈ set.Ioc (7/8 : ℚ) (8/9 : ℚ), f x = 16/17 := 
sorry

end max_f_in_interval_l7_7788


namespace difference_of_digits_l7_7580

theorem difference_of_digits (A B : ℕ) (h1 : 6 * 10 + A - (B * 10 + 2) = 36) (h2 : A ≠ B) : A - B = 5 :=
sorry

end difference_of_digits_l7_7580


namespace min_volunteers_l7_7303

-- Definitions of sets and their sizes
variables (L C : Set ℕ)
hypothesis hL : L.countable
hypothesis hC : C.countable
hypothesis hL_size : L.card = 95
hypothesis hC_size : C.card = 78
hypothesis h_intersect_size : (L ∩ C).card = 30

-- Statement to prove
theorem min_volunteers : (L ∪ C).card = 143 := 
by
  sorry

end min_volunteers_l7_7303


namespace brady_hours_in_september_l7_7695

/-- Given conditions: 
  Brady worked 6 hours every day in April,
  Brady worked 5 hours every day in June, 
  The average amount of hours that Brady worked per month in those 3 months is 190. 
  Prove that Brady worked 8 hours every day in September.
-/
theorem brady_hours_in_september 
  (days_in_april : ℕ)
  (hours_per_day_in_april : ℕ)
  (days_in_june : ℕ)
  (hours_per_day_in_june : ℕ)
  (days_in_september : ℕ)
  (avg_hours_per_month : ℕ)
  (total_months : ℕ := 3)
  (h_April : days_in_april = 30)
  (h_June : days_in_june = 30)
  (h_September : days_in_september = 30)
  (h_hours_per_day_in_april : hours_per_day_in_april = 6)
  (h_hours_per_day_in_june : hours_per_day_in_june = 5)
  (h_avg_hours_per_month : avg_hours_per_month = 190) :
  let total_hours_in_april := days_in_april * hours_per_day_in_april,
      total_hours_in_june := days_in_june * hours_per_day_in_june,
      total_hours_three_months := avg_hours_per_month * total_months,
      total_hours_in_september := total_hours_three_months - (total_hours_in_april + total_hours_in_june),
      daily_hours_in_september := total_hours_in_september / days_in_september
  in daily_hours_in_september = 8 :=
by
  sorry

end brady_hours_in_september_l7_7695


namespace length_of_segment_l7_7202

theorem length_of_segment (x : ℝ) : 
  |x - (27^(1/3))| = 5 →
  ∃ a b : ℝ, a - b = 10 ∧ (|a - (27^(1/3))| = 5 ∧ |b - (27^(1/3))| = 5) :=
by
  sorry

end length_of_segment_l7_7202


namespace diane_total_harvest_l7_7339

def total_harvest (h1 i1 i2 : Nat) : Nat :=
  h1 + (h1 + i1) + ((h1 + i1) + i2)

theorem diane_total_harvest :
  total_harvest 2479 6085 7890 = 27497 := 
by 
  sorry

end diane_total_harvest_l7_7339


namespace length_of_base_of_isosceles_l7_7900

-- Definitions based on conditions
variables (A B C M : Type) [point A] [point B] [point C] [point M]
variables (AB AC BC AM AD : ℝ)
hypothesis h_isosceles : AB = 4 ∧ AC = 4
hypothesis h_median : AM = 3
hypothesis h_median_midpoint : ∃ M, midpoint A BC M ∧ AM = MD
hypothesis h_parallelogram : ∃ D, parallelogram A B C D ∧ BD = 3 ∧ AD = 6

-- The theorem we want to prove
theorem length_of_base_of_isosceles (h : AB = 4 ∧ AC = 4 ∧ AM = 3 ∧ AMD = 6 ∧ exists D, parallelogram ABDC) : BC = sqrt(10) :=
begin
  sorry
end

end length_of_base_of_isosceles_l7_7900


namespace bananas_to_kiwis_l7_7307

/-- Variables representing the costs of each fruit. -/
variables (cost_banana cost_pear cost_kiwi : ℝ)

/-- Given conditions -/
axiom h1 : 4 * cost_banana = 3 * cost_pear
axiom h2 : 9 * cost_pear = 6 * cost_kiwi

/-- Proof to show the number of kiwis equivalent to 24 bananas. -/
theorem bananas_to_kiwis :
  ∃ x : ℝ, 24 * cost_banana = x * cost_kiwi ∧ x = 12 :=
by
  sorry

end bananas_to_kiwis_l7_7307


namespace actual_height_of_boy_l7_7552

theorem actual_height_of_boy
  (n : ℕ) (heights_avg_wrong : ℝ) (heights_avg_correct : ℝ) (wrong_height : ℝ) (actual_height : ℝ) 
  (h_n : n = 35) (h_heights_avg_wrong : heights_avg_wrong = 180) 
  (h_heights_avg_correct : heights_avg_correct = 178) (h_wrong_height : wrong_height = 156) 
  (h_actual_height : actual_height = 226) :
  ((n * heights_avg_wrong) - (n * heights_avg_correct)) = (actual_height - wrong_height) :=
by {
  cases h_n,
  cases h_heights_avg_wrong,
  cases h_heights_avg_correct,
  cases h_wrong_height,
  cases h_actual_height,
  simp,
}

end actual_height_of_boy_l7_7552


namespace nonzero_perfect_square_appears_l7_7571

def sequence_a : ℕ → ℕ
| 0     := 1
| (n+1) := if (sequence_a n - 2 ∉ (list.range (n+1)).map sequence_a) then sequence_a n - 2 else sequence_a n + 3

theorem nonzero_perfect_square_appears :
  ∀ k : ℕ, k ≠ 0 ∧ (∃ m : ℕ, m * m = k) → (∃ n : ℕ, sequence_a (n+1) = k) :=
sorry

end nonzero_perfect_square_appears_l7_7571


namespace number_of_zeros_at_end_of_factorial_30_l7_7697

-- Lean statement for the equivalence proof problem
def count_factors_of (p n : Nat) : Nat :=
  n / p + n / (p * p) + n / (p * p * p) + n / (p * p * p * p) + n / (p * p * p * p * p)

def zeros_at_end_of_factorial (n : Nat) : Nat :=
  count_factors_of 5 n

theorem number_of_zeros_at_end_of_factorial_30 : zeros_at_end_of_factorial 30 = 7 :=
by 
  sorry

end number_of_zeros_at_end_of_factorial_30_l7_7697


namespace reflection_of_vector_l7_7293

open Real EuclideanSpace

-- Definitions for vectors in ℝ²
def vector1 : ℝ × ℝ := (3, -2)
def vector2 : ℝ × ℝ := (-2, 6)
def vector_to_reflect : ℝ × ℝ := (5, 1)
def reflected_vector : ℝ × ℝ := (-67 / 17, 55 / 17)

-- We will set up a theorem stating the reflection property
theorem reflection_of_vector :
  ∃ u v w x y z, u = 3 ∧ v = -2 ∧ w = -2 ∧ x = 6 ∧ y = 5 ∧ z = 1 ∧ 
  reflection (u, v) (w, x) (y, z) = (-67 / 17, 55 / 17) := 
sorry

-- Defining the reflection function as described 
noncomputable def reflection (u v w x y z: ℝ): ℝ × ℝ :=
  let midpoint := ((u + w) / 2, (v + x) / 2 )
  let reflection_direction := (1, 4) -- assumed based on solution steps
  let proj := (((y * 1 + z * 4)/(1^2 + 4^2)) * (1, 4))
  (2 * proj.fst - y, 2 * proj.snd - z)


end reflection_of_vector_l7_7293


namespace problem_l7_7815

variable {R : Type*} [OrderedField R]

/-- The function f : R → R satisfies properties f(-x) = -f(x+4) and is strictly increasing for x > 2, 
    and given conditions x₁ + x₂ < 4 and (x₁ - 2)(x₂ - 2) < 0, we prove that f(x₁) + f(x₂) < 0. -/
theorem problem (f : R → R) (hf_eq : ∀ x, f (-x) = -f (x + 4)) 
  (hf_inc : ∀ x, 2 < x → f x > f 2) (x₁ x₂ : R) (h₁ : x₁ + x₂ < 4) 
  (h₂ : (x₁ - 2) * (x₂ - 2) < 0) : f x₁ + f x₂ < 0 := 
sorry

end problem_l7_7815


namespace trains_crossing_time_same_direction_l7_7593

theorem trains_crossing_time_same_direction :
  ∀ (L : ℝ) (speed1_kmh speed2_kmh time_opposite seconds_in_hour meters_in_km : ℝ),
    speed1_kmh = 60 →
    speed2_kmh = 40 →
    time_opposite = 8.000000000000002 →
    seconds_in_hour = 3600 →
    meters_in_km = 1000 →
    let speed_opposite := speed1_kmh + speed2_kmh in
    let speed_opposite_mps := (speed_opposite * meters_in_km) / seconds_in_hour in
    let distance_opposite := speed_opposite_mps * time_opposite in
    let L := distance_opposite / 2 in
    let speed_same := speed1_kmh - speed2_kmh in
    let speed_same_mps := (speed_same * meters_in_km) / seconds_in_hour in
    let time_same := (2 * L) / speed_same_mps in
    time_same = 40 :=
by
  intros L speed1_kmh speed2_kmh time_opposite seconds_in_hour meters_in_km
  assume h1 h2 h3 h4 h5
  let speed_opposite := speed1_kmh + speed2_kmh
  let speed_opposite_mps := (speed_opposite * meters_in_km) / seconds_in_hour
  let distance_opposite := speed_opposite_mps * time_opposite
  let L := distance_opposite / 2
  let speed_same := speed1_kmh - speed2_kmh
  let speed_same_mps := (speed_same * meters_in_km) / seconds_in_hour
  let time_same := (2 * L) / speed_same_mps
  sorry

end trains_crossing_time_same_direction_l7_7593


namespace n_mul_n_plus_one_even_l7_7148

theorem n_mul_n_plus_one_even (n : ℤ) : Even (n * (n + 1)) := 
sorry

end n_mul_n_plus_one_even_l7_7148


namespace circle_tangent_to_y_axis_l7_7216

theorem circle_tangent_to_y_axis (p : ℝ) (hp : p > 0) :
  ∀ (x y : ℝ), y^2 = 2 * p * x → 
              let focus := (p / 2, 0) in
              let P := (x, y) in
              let center := ((x + p) / 4, y / 2) in
              let radius := (2 * x + p) / 4 in
              ∀ (a b : ℝ), a = 0 → (x - a)^2 + (y - b)^2 = radius^2
              → (x = p / 4) :=
by
  sorry

end circle_tangent_to_y_axis_l7_7216


namespace trig_identity_l7_7392

variable (α : ℝ)

theorem trig_identity (h : cos (π / 3 + α) = 1 / 3) : 
  sin (5 * π / 6 + α) = 1 / 3 := 
by
  sorry

end trig_identity_l7_7392


namespace minimum_num_small_pipes_required_l7_7288

open Real

-- Definitions based on conditions
def diameter_large : ℝ := 20 -- Diameter of the larger pipe in inches
def diameter_small : ℝ := 4 -- Diameter of the smaller pipe in inches
def radius_large : ℝ := diameter_large / 2 -- Radius of the larger pipe
def radius_small : ℝ := diameter_small / 2 -- Radius of the smaller pipe

-- Volume per unit height for the larger and smaller pipes
def volume_large (h : ℝ) : ℝ := π * radius_large^2 * h
def volume_small (h : ℝ) : ℝ := π * radius_small^2 * h

-- Define the equivalent number of smaller pipes needed
def num_small_pipes_required (h : ℝ) : ℝ := volume_large h / volume_small h

theorem minimum_num_small_pipes_required : ∀ (h : ℝ), num_small_pipes_required h = 25 := by
  intro h
  -- Proof goes here
  sorry

end minimum_num_small_pipes_required_l7_7288


namespace total_pencils_children_l7_7578

theorem total_pencils_children :
  let c1 := 6
  let c2 := 9
  let c3 := 12
  let c4 := 15
  let c5 := 18
  c1 + c2 + c3 + c4 + c5 = 60 :=
by
  let c1 := 6
  let c2 := 9
  let c3 := 12
  let c4 := 15
  let c5 := 18
  show c1 + c2 + c3 + c4 + c5 = 60
  sorry

end total_pencils_children_l7_7578


namespace distance_to_y_axis_of_point_on_ellipse_l7_7411

-- Definitions required for the problem
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1
def foci1 := (sqrt 3, 0)
def foci2 := (-sqrt 3, 0)
def M : Type := { p : ℝ × ℝ // ellipse p.1 p.2 }

noncomputable def mf1 (x y : ℝ) : ℝ := (x - sqrt 3) * x + y * y
noncomputable def mf2 (x y : ℝ) : ℝ := (x + sqrt 3) * x + y * y

-- Distances and conditions
def perpendicular (x y : ℝ) : Prop := mf1 x y * mf2 x y = 0

-- Final proof statement
theorem distance_to_y_axis_of_point_on_ellipse
  (x y : ℝ)
  (hx : ellipse x y)
  (hp : perpendicular x y) :
  abs x = (2 * real.sqrt 6) / 3 :=
sorry

end distance_to_y_axis_of_point_on_ellipse_l7_7411


namespace probability_lt_one_third_l7_7855

theorem probability_lt_one_third :
  let total_interval := (0 : ℝ, 1/2)
  let desired_interval := (0 : ℝ, 1/3)
  let total_length := (total_interval.2 - total_interval.1 : ℝ) = 1/2
  let desired_length := (desired_interval.2 - desired_interval.1 : ℝ) = 1/3
  -- then the probability P is given by:
  (desired_length / total_length) = 2/3
:=
by {
  -- definition of intervals and lengths
  let total_interval := (0 : ℝ, 1/2)
  let desired_interval := (0 : ℝ, 1/3)
  have total_length : total_interval.2 - total_interval.1 = 1/2,
  -- verify total length calculation
  calc
    total_interval.2 - total_interval.1 = 1/2 - 0 : by simp
                                 ... = 1/2      : by norm_num,
  have desired_length : desired_interval.2 - desired_interval.1 = 1/3,
  -- verify desired interval length calculation
  calc
    desired_interval.2 - desired_interval.1 = 1/3 - 0 : by simp
                                    ... = 1/3      : by norm_num,
  -- calculate probability
  set P := desired_length / total_length
  -- compute correct answer
  have : P = (1/3) / (1/2),
  calc
    (1/3) / (1/2) = (1/3) * (2/1) : by field_simp
              ...  = 2/3      : by norm_num,
  exact this
}

end probability_lt_one_third_l7_7855


namespace max_points_in_k_dim_subspace_le_pow_l7_7493

variables {n k : ℕ} (V : Submodule ℝ (EuclideanSpace ℝ (Fin n)))

def Z : Set (EuclideanSpace ℝ (Fin n)) := 
  {v | ∀ i, v i = 0 ∨ v i = 1}

noncomputable def Z_V (V : Submodule ℝ (EuclideanSpace ℝ (Fin n))) : ℕ :=
  Finset.card (Z ∩ V)

noncomputable def max_points_in_subspace (k n : ℕ) : ℕ :=
  Sup (Function.const (set_of (λ (V : Submodule ℝ (EuclideanSpace ℝ (Fin n))), 
       finrank ℝ V = k)) (Z_V V))

theorem max_points_in_k_dim_subspace_le_pow {k n : ℕ} 
    (V : Submodule ℝ (EuclideanSpace ℝ (Fin n))) 
    (h_dim : finrank ℝ V = k) :
  Z_V V ≤ 2^k := sorry

end max_points_in_k_dim_subspace_le_pow_l7_7493


namespace repeating_six_equals_fraction_l7_7747

theorem repeating_six_equals_fraction : ∃ f : ℚ, (∀ n : ℕ, (n ≥ 1 → (6 * (10 : ℕ) ^ (-n) : ℚ) + (f - (6 * (10 : ℕ) ^ (-n) : ℚ)) = f)) ∧ f = 2 / 3 := sorry

end repeating_six_equals_fraction_l7_7747


namespace original_triangle_area_l7_7162

theorem original_triangle_area (A_new : ℝ) (k : ℝ) (h1 : k = 4) (h2 : A_new = 64) :
  let A_orig := A_new / (k * k) in A_orig = 4 :=
by
  let A_orig := A_new / (k * k)
  sorry

end original_triangle_area_l7_7162


namespace proof_smallest_lcm_1_to_12_l7_7223

noncomputable def smallest_lcm_1_to_12 : ℕ := 27720

theorem proof_smallest_lcm_1_to_12 :
  ∀ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ↔ n = 27720 :=
by
  sorry

end proof_smallest_lcm_1_to_12_l7_7223


namespace constant_a_of_square_binomial_l7_7847

theorem constant_a_of_square_binomial (a : ℝ) : (∃ b : ℝ, (9x^2 + 27x + a) = (3x + b)^2) -> a = 81 / 4 :=
by
  intro h
  sorry

end constant_a_of_square_binomial_l7_7847


namespace sum_of_angles_at_shared_vertex_l7_7911

-- Define the number of sides of the polygons
def n_octagon : ℕ := 8
def n_pentagon : ℕ := 5

-- Calculate the interior angles of the polygons
def interior_angle (n : ℕ) : ℝ := 180 * (n - 2) / n

-- Interior angle of the octagon and pentagon
def angle_PQR : ℝ := interior_angle n_octagon
def angle_PQS : ℝ := interior_angle n_pentagon

-- Sum of the angles
def sum_of_angles : ℝ := angle_PQR + angle_PQS

-- The theorem to prove
theorem sum_of_angles_at_shared_vertex : sum_of_angles = 243 := by
  unfold sum_of_angles
  unfold angle_PQR
  unfold angle_PQS
  unfold interior_angle
  simp
  norm_num
  sorry

end sum_of_angles_at_shared_vertex_l7_7911


namespace ages_of_sons_l7_7655

variable (x y z : ℕ)

def father_age_current : ℕ := 33

def youngest_age_current : ℕ := 2

def father_age_in_12_years : ℕ := father_age_current + 12

def sum_of_ages_in_12_years : ℕ := youngest_age_current + 12 + y + 12 + z + 12

theorem ages_of_sons (x y z : ℕ) 
  (h1 : x = 2)
  (h2 : father_age_current = 33)
  (h3 : father_age_in_12_years = 45)
  (h4 : sum_of_ages_in_12_years = 45) :
  x = 2 ∧ y + z = 7 ∧ ((y = 3 ∧ z = 4) ∨ (y = 4 ∧ z = 3)) :=
by
  sorry

end ages_of_sons_l7_7655


namespace prob_odd_in_set_l7_7779

open Set

def draw_odd_probability (s : Set ℕ) (odds : Set ℕ) : ℚ :=
  (odds.toFinset.card : ℚ) / (s.toFinset.card : ℚ)

theorem prob_odd_in_set :
  let s := {1, 2, 3, 4, 5}
  let odds := {x ∈ s | x % 2 = 1}
  draw_odd_probability s odds = 3 / 5 := by
  sorry

end prob_odd_in_set_l7_7779


namespace minimize_y_at_x_l7_7506

-- Define the function y
def y (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * (x - a) ^ 2 + (x - b) ^ 2 

-- State the theorem
theorem minimize_y_at_x (a b : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x' a b ≥ y ((3 * a + b) / 4) a b) :=
by
  sorry

end minimize_y_at_x_l7_7506


namespace distance_A_B_eq_5_sqrt_10_l7_7289

noncomputable def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_A_B_eq_5_sqrt_10 :
  let A := (-3, 5) in
  let B := (2, 10) in
  distance A B = 5 * real.sqrt 10 :=
by
  sorry

end distance_A_B_eq_5_sqrt_10_l7_7289


namespace funcC_has_properties_l7_7299

noncomputable def period_cos_2x_minus_pi_over_6 : ℝ := sorry

-- Define the functions
def funcA (x : ℝ) := sin (x / 2 + π / 6)
def funcB (x : ℝ) := cos (2 * x - π / 3)
def funcC (x : ℝ) := cos (2 * x - π / 6)
def funcD (x : ℝ) := sin (2 * x - π / 6)

-- Define the problem conditions
def point_of_symmetry : ℝ × ℝ := (π / 3, 0)

-- Lean statement to prove the function with properties
theorem funcC_has_properties :
  (∃ T > 0, ∀ x, funcC (x + T) = funcC x) ∧
  (∃ x₀, funcC x₀ = 0 ∧ x₀ = π / 3) :=
by
  sorry

end funcC_has_properties_l7_7299


namespace gift_cost_calc_l7_7589

theorem gift_cost_calc (C N : ℕ) (hN : N = 12)
    (h : C / (N - 4) = C / N + 10) : C = 240 := by
  sorry

end gift_cost_calc_l7_7589


namespace classmates_ate_cake_l7_7083

theorem classmates_ate_cake (n : ℕ) (h_least_ate : (1 : ℚ) / 14) (h_most_ate : (1 : ℚ) / 11)
  (h_total_cake : 1 = ((1 : ℚ) / 11 + (1 : ℚ) / 14 + (n - 2) * x)) :
  n = 12 ∨ n = 13 :=
by
  sorry

end classmates_ate_cake_l7_7083


namespace distinct_triangles_count_l7_7322

-- Define points and conditions
structure Point2D :=
  (x : ℤ)
  (y : ℤ)

def validPoints (p : Point2D) : Prop :=
  (p.x % 3 = 0) ∧ (p.y % 3 = 0) ∧ (47 * p.x + p.y = 2353)

noncomputable def area (p q : Point2D) : ℝ :=
  0.5 * Real.abs (p.x * q.y - q.x * p.y)

theorem distinct_triangles_count :
  ∃ t : Finset (Point2D × Point2D), (∀ (p q : Point2D), (p, q) ∈ t → validPoints p ∧ validPoints q ∧ p ≠ q ∧ (∃ n ∈ Set.Icc 1 ∞, area p q = n)) ∧ t.card = 64 :=
sorry

end distinct_triangles_count_l7_7322


namespace tom_months_between_visits_l7_7192

def cost_per_pill := 5
def insurance_coverage := 0.8
def cost_per_visit := 400
def daily_pills := 2
def total_annual_cost := 1530

theorem tom_months_between_visits :
  let cost_per_pill_for_tom := cost_per_pill * (1 - insurance_coverage) in 
  let annual_medication_cost := daily_pills * cost_per_pill_for_tom * 365 in
  let annual_doctor_visit_cost := total_annual_cost - annual_medication_cost in
  let visits_per_year := annual_doctor_visit_cost / cost_per_visit in
  12 / visits_per_year = 6 :=
by
  sorry

end tom_months_between_visits_l7_7192


namespace NK_parallel_A5A2_l7_7078

noncomputable theory
open_locale classical

variables {A1 A2 A3 A4 A5 A6 K L M N : Point}

-- Given conditions
def lie_on_circle (A1 A2 A3 A4 A5 A6 : Point) : Prop :=
  ∃ (C : Circle), A1 ∈ C ∧ A2 ∈ C ∧ A3 ∈ C ∧ A4 ∈ C ∧ A5 ∈ C ∧ A6 ∈ C

def on_lines (K : Point) : Prop := lies_on K (line_through A1 A2)
def on_lines (L : Point) : Prop := lies_on L (line_through A3 A4)
def on_lines (M : Point) : Prop := lies_on M (line_through A1 A6)
def on_lines (N : Point) : Prop := lies_on N (line_through A4 A5)

def parallel_KL_A2A3 : Prop := parallel (line_through K L) (line_through A2 A3)
def parallel_LM_A3A6 : Prop := parallel (line_through L M) (line_through A3 A6)
def parallel_MN_A6A5 : Prop := parallel (line_through M N) (line_through A6 A5)

-- The theorem statement
theorem NK_parallel_A5A2
  (circ : lie_on_circle A1 A2 A3 A4 A5 A6)
  (on_lines_K : on_lines K)
  (on_lines_L : on_lines L)
  (on_lines_M : on_lines M)
  (on_lines_N : on_lines N)
  (par1 : parallel_KL_A2A3)
  (par2 : parallel_LM_A3A6)
  (par3 : parallel_MN_A6A5) :
  parallel (line_through N K) (line_through A5 A2) :=
sorry

end NK_parallel_A5A2_l7_7078


namespace arithmetic_sequence_sum_l7_7816

-- Use noncomputable theory to avoid issues with real number computations
noncomputable theory

-- Define the sequence
def a (n : ℕ) : ℝ := 2 * n - 1

-- Define the sum of the first n terms
def S (n : ℕ) : ℝ := (n * (2 * 1 + (n - 1) * 2)) / 2

-- Define A
def A (n : ℕ) : ℝ :=
  -a 1 * a 2 + a 2 * a 3 - a 3 * a 4 + a 4 * a 5 - 
  ∑ i in Finset.range n, a (2 * i + 1) * a (2 * i + 2)

-- The theorem we need to prove
theorem arithmetic_sequence_sum (n : ℕ) : A n = 8 * n^2 + 4 * n :=
  sorry

end arithmetic_sequence_sum_l7_7816


namespace geometric_sequence_S_n_l7_7830

-- Definitions related to the sequence
def a_n (n : ℕ) : ℕ := sorry  -- Placeholder for the actual sequence

-- Sum of the first n terms
def S_n (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms

-- Given conditions
axiom a1 : a_n 1 = 1
axiom Sn_eq_2an_plus1 : ∀ (n : ℕ), S_n n = 2 * a_n (n + 1)

-- Theorem to be proved
theorem geometric_sequence_S_n 
    (n : ℕ) (h : n > 1) 
    : S_n n = (3/2)^(n-1) := 
by 
  sorry

end geometric_sequence_S_n_l7_7830


namespace stratified_sampling_l7_7177

theorem stratified_sampling (n : ℕ) (h_ratio : 2 + 3 + 5 = 10) (h_sample : (5 : ℚ) / 10 = 150 / n) : n = 300 :=
by
  sorry

end stratified_sampling_l7_7177


namespace social_network_stablization_l7_7965

variable {Person : Type}
variable (friend : Person → Person → Prop) [Symmetric friend]

-- Definition of initial conditions
noncomputable def UserNetwork : Type :=
  { p : Person // ∃ k : Nat, (k = 1009 ∨ k = 1010) ∧ (∥friends_of p∥ = k) }

-- Definition of refriending operation
def refriending (A B C : Person) (hAB : friend A B) (hAC : friend A C) (hBC : ¬friend B C) : 
    (new_friendship : Person → Person → Prop) :=
  (λ x y : Person, if x = B ∧ y = C ∨ x = C ∧ y = B then True else friend x y ∧ not (x = A ∨ y = A))

-- Problem statement as a theorem
theorem social_network_stablization : 
  ∀ (people : Fin 2019 → UserNetwork) 
    (initial_friendship : ∀ p q : UserNetwork, friend p q → friend q p),
    ∃ (final_friendship : Person → Person → Prop), 
      (∀ p q : Person, final_friendship p q → final_friendship q p) ∧ 
      (∀ p : Person, ∥friends_of p with respect to final_friendship∥ ≤ 1) :=
sorry

end social_network_stablization_l7_7965


namespace determine_2023rd_digit_l7_7507

/- Conditions -/
def segment_A_digits : ℕ := 9
def segment_B_digits : ℕ := 90 * 2
def segment_C_start : ℕ := 100
def segment_C_length : ℕ := 900
def segment_C_digits : ℕ := segment_C_length * 3

/- Result to Prove -/
def digit_2023_position : ℕ := 2023 - segment_A_digits - segment_B_digits
def numbers_in_segment_C : ℕ := digit_2023_position / 3
def remainder_digit : ℕ := digit_2023_position % 3
def resulting_digit : ℕ := Nat.digits 10 (segment_C_start + numbers_in_segment_C - 1) !! remainder_digit

theorem determine_2023rd_digit :
  resulting_digit = 7 := by
  sorry

end determine_2023rd_digit_l7_7507


namespace log_expression_l7_7321

-- Define the logarithm functions and their properties
axiom log_mul (a b : ℝ) : real.log (a * b) = real.log a + real.log b
axiom log_pow (a : ℝ) (n : ℕ) : real.log (a^n) = n * real.log a
axiom log_10 : real.log 2 + real.log 5 = 1

-- Lean statement for the proof problem
theorem log_expression :
  real.log 4 + real.log 5 * real.log 20 + (real.log 5)^2 = 2 :=
by
  have h1 : real.log 4 = 2 * real.log 2 := by { rw [←real.log_pow 2 2], try {ring} }
  have h2 : real.log 20 = real.log (4 * 5) := by rw [←log_mul 4 5]

  -- Given conditions
  have h3 : real.log (4 * 5) = real.log 4 + real.log 5 := by apply log_mul
  have h4 : real.log 4 + real.log 5 = 2 * real.log 2 + real.log 5 := by rw h1

  -- Variables simplified by log_10 axiom
  have h5 : (2 : ℝ) = 1 := by from log_10.symm

  -- Constructing the main proof
  sorry

end log_expression_l7_7321


namespace sum_of_abc_l7_7570

theorem sum_of_abc (a b c : ℕ) (h1 : a = 2) (h2 : b = 1) (h3 : c = 1)
    (h4 : (180 : ℚ) / 45 = 4) (h5 : (4 : ℚ).sqrt = 2) :
    a + b + c = 4 := by
  rfl  -- Proof directly matches the conditions, ensuring definitions have been used correctly.

end sum_of_abc_l7_7570


namespace simplify_expression_l7_7182

theorem simplify_expression (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) : 
  1 - (1 / (1 + (a^2 / (1 - a^2)))) = a^2 :=
sorry

end simplify_expression_l7_7182


namespace min_positive_period_of_cosine_function_l7_7174

theorem min_positive_period_of_cosine_function :
  (∀ x : ℝ, ∃ T : ℝ, T > 0 ∧ ∀ x, 4 * cos (2 * (x + T)) + 3 = 4 * cos (2 * x) + 3) → T = π :=
sorry

end min_positive_period_of_cosine_function_l7_7174


namespace notAPrpos_l7_7623

def isProposition (s : String) : Prop :=
  s = "6 > 4" ∨ s = "If f(x) is a sine function, then f(x) is a periodic function." ∨ s = "1 ∈ {1, 2, 3}"

theorem notAPrpos (s : String) : ¬isProposition "Is a linear function an increasing function?" :=
by
  sorry

end notAPrpos_l7_7623


namespace switches_in_position_A_after_500_steps_l7_7190

def switches := Finset (Fin 5)
def positions := Fin 5
def cyclic_transitions := {A := 0, B := 1, C := 2, D := 3, E := 4}

def label (x y z : ℕ) : ℕ := (2^x) * (3^y) * (7^z)
def labels := {label x y z | x y z in Finset.range 5}
def N : ℕ := (2^4) * (3^4) * (7^4)

noncomputable def step_advance (label : ℕ) (i : ℕ) : ℕ // divides label i → ℕ := sorry

noncomputable def count_final_position_A : ℕ :=
  let valid_combinations := [(0,0,0), (0,0,4), (0,4,0), (4,0,0), (4,4,4), (4,4,0), (4,0,4), (0,4,4)] in
  valid_combinations.card

theorem switches_in_position_A_after_500_steps :
  switches_in_position_A_after_500_steps = 10 := sorry

end switches_in_position_A_after_500_steps_l7_7190


namespace smallest_m_satisfying_conditions_l7_7607

theorem smallest_m_satisfying_conditions :
  ∃ m : ℕ, m = 4 ∧ (∃ k : ℕ, 0 ≤ k ∧ k ≤ m ∧ (m^2 + m) % k ≠ 0) ∧ (∀ k : ℕ, (0 ≤ k ∧ k ≤ m) → (k ≠ 0 → (m^2 + m) % k = 0)) :=
sorry

end smallest_m_satisfying_conditions_l7_7607


namespace centroid_positions_l7_7158

-- Define the points on the perimeter of the square
def isValidPoint (x y : ℚ) : Prop :=
  (0 ≤ x ∧ x ≤ 12 ∧ (y = 0 ∨ y = 12)) ∨ (0 ≤ y ∧ y ≤ 12 ∧ (x = 0 ∨ x = 12)) ∧ (x * 11 / 12 ∈ ℤ) ∧ (y * 11 / 12 ∈ ℤ)

-- Define the points P, Q, and R with no two being consecutive
def noTwoConsecutive (P Q R : ℚ × ℚ) : Prop :=
  -- Define no two points should be consecutive logic here
  sorry

-- Define the centroid of the triangle PQR
def centroid (P Q R : ℚ × ℚ) : ℚ × ℚ :=
  (1 / 3 * (P.1 + Q.1 + R.1), 1 / 3 * (P.2 + Q.2 + R.2))

-- Define the range of valid centroids
def isValidCentroid (G : ℚ × ℚ) : Prop :=
  ∃ (m n : ℤ), 1 ≤ m ∧ m ≤ 34 ∧ 1 ≤ n ∧ n ≤ 34 ∧ G = (m / 3, n / 3)

-- Write the theorem statement
theorem centroid_positions :
  ∀ P Q R : ℚ × ℚ, isValidPoint P.1 P.2 → isValidPoint Q.1 Q.2 → isValidPoint R.1 R.2 →
  noTwoConsecutive P Q R → 
  ∃ (count : ℤ), count = 1156 ∧ ∀ G, isValidCentroid (centroid P Q R) ↔ G ∈ set.univ :=
sorry

end centroid_positions_l7_7158


namespace initial_albums_in_cart_l7_7956

theorem initial_albums_in_cart (total_songs : ℕ) (songs_per_album : ℕ) (removed_albums : ℕ) 
  (h_total: total_songs = 42) 
  (h_songs_per_album: songs_per_album = 7)
  (h_removed: removed_albums = 2): 
  (total_songs / songs_per_album) + removed_albums = 8 := 
by
  sorry

end initial_albums_in_cart_l7_7956


namespace no_such_function_proof_l7_7970

noncomputable def no_such_function_exists : Prop :=
  ¬ ∃ (f : ℝ → ℝ), 
    (∀ x > 0, f x > 0) ∧
    (∀ x y > 0, (f x) ^ 2 ≥ f (x + y) * (f x + y))

theorem no_such_function_proof : no_such_function_exists :=
by 
  sorry

end no_such_function_proof_l7_7970


namespace set_A_range_l7_7927

def A := {y : ℝ | ∃ x : ℝ, y = -x^2 ∧ (-1 ≤ x ∧ x ≤ 2)}

theorem set_A_range :
  A = {y : ℝ | -4 ≤ y ∧ y ≤ 0} :=
sorry

end set_A_range_l7_7927


namespace scale_down_multiplication_l7_7372

theorem scale_down_multiplication (h : 14.97 * 46 = 688.62) : 1.497 * 4.6 = 6.8862 :=
by
  -- here we assume the necessary steps to justify the statement.
  sorry

end scale_down_multiplication_l7_7372


namespace continuous_function_inequality_l7_7714

theorem continuous_function_inequality (f : ℝ → ℝ) (h_cont : continuous f)
  (h_deriv : ∀ x, (x - 1) * (deriv f x) < 0) :
  f 0 + f 2 < 2 * f 1 :=
by
  sorry

end continuous_function_inequality_l7_7714


namespace nested_roots_identity_l7_7445

theorem nested_roots_identity (x : ℝ) (hx : x ≥ 0) : Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15 / 16) :=
sorry

end nested_roots_identity_l7_7445


namespace lcm_of_1_to_12_l7_7228

noncomputable def lcm_1_to_12 : ℕ := 2^3 * 3^2 * 5 * 7 * 11

theorem lcm_of_1_to_12 : lcm_1_to_12 = 27720 := by
  sorry

end lcm_of_1_to_12_l7_7228


namespace solve_for_f_x_plus_2_l7_7504

def f (x : ℝ) : ℝ := (x^2 + 1) / (x^2 - 1)

theorem solve_for_f_x_plus_2 (x : ℝ) (hx : x^2 ≠ 1) : 
  f (x + 2) = (x^2 + 4 * x + 5) / (x^2 + 4 * x + 3) := 
by 
  sorry

end solve_for_f_x_plus_2_l7_7504


namespace function_passes_through_fixed_point_l7_7817

noncomputable def f (a : ℝ) (x : ℝ) := 4 + Real.log (x + 1) / Real.log a

theorem function_passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  f a 0 = 4 := 
by
  sorry

end function_passes_through_fixed_point_l7_7817


namespace points_remain_odd_l7_7538

-- Define an inductive process to describe the insertion of points
def points_after_operations (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then n else 2 * points_after_operations n (k - 1) - 1

-- Main proposition to prove
theorem points_remain_odd (n k : ℕ) : odd (points_after_operations n k) :=
by sorry

end points_remain_odd_l7_7538


namespace find_original_table_price_l7_7295

variable (C T : ℝ)

-- Definitions
def C_discounted := 0.9 * C
def T_tax := 1.05 * T
def eq1 := 2 * C_discounted + T_tax = 0.6 * (C_discounted + 2 * T_tax)
def eq2 := T_tax + C_discounted = 60

theorem find_original_table_price (h1 : eq1) (h2 : eq2) : T = 50 :=
by sorry

end find_original_table_price_l7_7295


namespace simplify_expression_frac_l7_7166

theorem simplify_expression_frac (a b k : ℤ) (h : (6*k + 12) / 6 = a * k + b) : a = 1 ∧ b = 2 → a / b = 1 / 2 := by
  sorry

end simplify_expression_frac_l7_7166


namespace number_of_classmates_l7_7122

theorem number_of_classmates (n : ℕ) (Alex Aleena : ℝ) 
  (h_Alex : Alex = 1/11) (h_Aleena : Aleena = 1/14) 
  (h_bound : ∀ (x : ℝ), Aleena ≤ x ∧ x ≤ Alex → ∃ c : ℕ, n - 2 > 0 ∧ c = n - 2) : 
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_l7_7122


namespace find_d_l7_7353

theorem find_d (d : ℝ) :
  (d / 4 ≤ 3 - d) ∧ (3 - d < 1 - 2d) ↔ -2 < d ∧ d ≤ 12 / 5 :=
by
  sorry

end find_d_l7_7353


namespace find_certain_number_l7_7342

theorem find_certain_number (x : ℤ) (h : ((x / 4) + 25) * 3 = 150) : x = 100 :=
by
  sorry

end find_certain_number_l7_7342


namespace calculate_b4_b6_l7_7447

-- Define a harmonic sequence
def harmonic_sequence (an : ℕ → ℚ) (d : ℚ) :=
  ∀ n, (1 / an (n + 1)) - (1 / an n) = d

-- Define the problem conditions
def problem_conditions (bn : ℕ → ℚ) (d : ℚ) :=
  harmonic_sequence (λ n, bn n) d ∧
  ∑ i in finset.range 9, bn (i + 1) = 90

-- Statement to be proven
theorem calculate_b4_b6 (bn : ℕ → ℚ) (d : ℚ) (h : problem_conditions bn d) :
  bn 4 + bn 6 = 20 :=
sorry

end calculate_b4_b6_l7_7447


namespace distribute_balls_ways_l7_7435

theorem distribute_balls_ways :
  let arrangements := [(7,0,0), (6,1,0), (5,2,0), (5,1,1), (4,3,0), (4,2,1), (3,3,1), (3,2,2)] in
  let ways (a : ℕ × ℕ × ℕ) : ℕ :=
    match a with
    | (7,0,0) => 3
    | (6,1,0) => 6
    | (5,2,0) => 6
    | (5,1,1) => 3
    | (4,3,0) => 6
    | (4,2,1) => 6
    | (3,3,1) => 3
    | (3,2,2) => 3
    | _       => 0
  in
  arrangements.foldl (λ acc a => acc + ways a) 0 = 36 :=
by
  sorry

end distribute_balls_ways_l7_7435


namespace cake_eating_classmates_l7_7109

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l7_7109


namespace lcm_of_1_to_12_l7_7232

noncomputable def lcm_1_to_12 : ℕ := 2^3 * 3^2 * 5 * 7 * 11

theorem lcm_of_1_to_12 : lcm_1_to_12 = 27720 := by
  sorry

end lcm_of_1_to_12_l7_7232


namespace lcm_triples_count_l7_7432

theorem lcm_triples_count :
  (finset.univ.filter (λ ⟨x, y, z⟩ : ℕ × ℕ × ℕ,
    nat.lcm x y = 180 ∧ nat.lcm x z = 1000 ∧ nat.lcm y z = 1200)).card = 5 :=
begin
  sorry
end

end lcm_triples_count_l7_7432


namespace division_theorem_l7_7770

-- Define the main polynomials involved in the problem
def poly : ℝ[X] := X^4 - 3 * X^2 + 2
def divisor : ℝ[X] := (X - 3)^2

-- Define the expected remainder
def remainder : ℝ[X] := 56 * X - 78

-- State the theorem to be proved
theorem division_theorem : ∃ Q : ℝ[X], poly = divisor * Q + remainder :=
by
  sorry

end division_theorem_l7_7770


namespace cake_sharing_l7_7100

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l7_7100


namespace flight_distance_calculation_l7_7966

theorem flight_distance_calculation (speed distance_from_departure time_remaining : ℝ) (h1 : speed = 864) (h2 : distance_from_departure = 1222) (h3 : time_remaining = 4 / 3) :
  distance_from_departure + speed * time_remaining ≈ 2400 :=
by sorry

end flight_distance_calculation_l7_7966


namespace tony_water_trips_calculation_l7_7587

noncomputable def tony_drinks_water_after_every_n_trips (bucket_capacity_sand : ℤ) 
                                                        (sandbox_depth : ℤ) (sandbox_width : ℤ) 
                                                        (sandbox_length : ℤ) (sand_weight_cubic_foot : ℤ) 
                                                        (water_consumption : ℤ) (water_bottle_ounces : ℤ) 
                                                        (water_bottle_cost : ℤ) (money_with_tony : ℤ) 
                                                        (expected_change : ℤ) : ℤ :=
  let volume_sandbox := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := volume_sandbox * sand_weight_cubic_foot
  let trips_needed := total_sand_weight / bucket_capacity_sand
  let money_spent_on_water := money_with_tony - expected_change
  let water_bottles_bought := money_spent_on_water / water_bottle_cost
  let total_water_ounces := water_bottles_bought * water_bottle_ounces
  let drinking_sessions := total_water_ounces / water_consumption
  trips_needed / drinking_sessions

theorem tony_water_trips_calculation : 
  tony_drinks_water_after_every_n_trips 2 2 4 5 3 3 15 2 10 4 = 4 := 
by 
  sorry

end tony_water_trips_calculation_l7_7587


namespace number_of_solutions_to_equation_l7_7985

theorem number_of_solutions_to_equation : 
  (∃ (x y z : ℕ), xyz + xy + yz + zx + x + y + z = 2014) → 
  (∃ solutions : finset (ℕ × ℕ × ℕ), solutions.card = 27) :=
  by
  sorry

end number_of_solutions_to_equation_l7_7985


namespace scaled_multiplication_l7_7370

theorem scaled_multiplication
  (h : 14.97 * 46 = 688.62) :
  1.497 * 4.6 = 6.8862 :=
by
  sorry

end scaled_multiplication_l7_7370


namespace position_of_23_in_sequence_l7_7180

theorem position_of_23_in_sequence :
  ∃ n : ℕ, 2 + (n - 1) * 3 = 23 ∧ n = 8 :=
by
  use 8
  split
  {
    rw [nat.succ_sub_succ_eq_sub, nat.sub_zero],
    norm_num,
  }
  {
    refl,
  }  

end position_of_23_in_sequence_l7_7180


namespace classmates_ate_cake_l7_7094

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l7_7094


namespace cheesecake_total_calories_l7_7731

-- Define the conditions
def slice_calories : ℕ := 350

def percent_eaten : ℕ := 25
def slices_eaten : ℕ := 2

-- Define the total number of slices in a cheesecake
def total_slices (percent_eaten slices_eaten : ℕ) : ℕ :=
  slices_eaten * (100 / percent_eaten)

-- Define the total calories in a cheesecake given the above conditions
def total_calories (slice_calories slices : ℕ) : ℕ :=
  slice_calories * slices

-- State the theorem
theorem cheesecake_total_calories :
  total_calories slice_calories (total_slices percent_eaten slices_eaten) = 2800 :=
by
  sorry

end cheesecake_total_calories_l7_7731


namespace three_digit_number_452_l7_7674

theorem three_digit_number_452 (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 1 ≤ b) (h4 : b ≤ 9) (h5 : 1 ≤ c) (h6 : c ≤ 9) 
  (h7 : 100 * a + 10 * b + c % (a + b + c) = 1)
  (h8 : 100 * c + 10 * b + a % (a + b + c) = 1)
  (h9 : a ≠ b) (h10 : b ≠ c) (h11 : a ≠ c)
  (h12 : a > c) :
  100 * a + 10 * b + c = 452 :=
sorry

end three_digit_number_452_l7_7674


namespace number_of_lines_passing_through_five_points_l7_7840

theorem number_of_lines_passing_through_five_points : 
  let points := {(i, j, k) | i ∈ {1, 2, 3, 4, 5}, j ∈ {1, 2, 3, 4, 5}, k ∈ {1, 2, 3, 4, 5}} in
  let directions := {(a, b, c) | a ∈ {-1, 0, 1}, b ∈ {-1, 0, 1}, c ∈ {-1, 0, 1}, (a, b, c) ≠ (0, 0, 0)} in
  ∃ (lines : finset (finset (ℕ × ℕ × ℕ))),
  lines.card = 120 ∧ 
  ∀ (line ∈ lines), 
    ∃ (p₀ : ℕ × ℕ × ℕ) (d : ℤ × ℤ × ℤ),
    d ∈ directions ∧
    ∀ n : ℕ, n < 5 → ((p₀.1 + n * d.1).nat_abs, (p₀.2 + n * d.2).nat_abs, (p₀.3 + n * d.3).nat_abs) ∈ points 
    := sorry

end number_of_lines_passing_through_five_points_l7_7840


namespace average_speed_ratio_l7_7644

-- Define the given conditions
def speed_in_still_water : ℝ := 10
def speed_of_current : ℝ := 3
def distance : ℝ := 1 -- assuming 1 mile for simplicity

-- Calculate derived values
def downstream_speed : ℝ := speed_in_still_water + speed_of_current
def upstream_speed : ℝ := speed_in_still_water - speed_of_current

-- Times for downstream and upstream journey
def time_downstream : ℝ := distance / downstream_speed
def time_upstream : ℝ := distance / upstream_speed

-- Total time and total distance for the round trip
def total_time : ℝ := time_downstream + time_upstream
def total_distance : ℝ := 2 * distance

-- Average speed for the round trip
def average_speed : ℝ := total_distance / total_time

-- Prove the ratio of the average speed to the speed in still water
theorem average_speed_ratio :
  average_speed / speed_in_still_water = 91 / 100 :=
sorry

end average_speed_ratio_l7_7644


namespace linemen_ounces_per_drink_l7_7662

-- Definitions corresponding to the conditions.
def linemen := 12
def skill_position_drink := 6
def skill_position_before_refill := 5
def cooler_capacity := 126

-- The theorem that requires proof.
theorem linemen_ounces_per_drink (L : ℕ) (h : 12 * L + 5 * skill_position_drink = cooler_capacity) : L = 8 :=
by
  sorry

end linemen_ounces_per_drink_l7_7662


namespace repeating_six_as_fraction_l7_7744

theorem repeating_six_as_fraction : (∑' n : ℕ, 6 / (10 * (10 : ℝ)^n)) = (2 / 3) :=
by
  sorry

end repeating_six_as_fraction_l7_7744


namespace uncle_bob_can_park_l7_7286

-- Define the conditions
def total_spaces : Nat := 18
def cars : Nat := 15
def rv_spaces : Nat := 3

-- Define a function to calculate the probability (without implementation)
noncomputable def probability_RV_can_park (total_spaces cars rv_spaces : Nat) : Rat :=
  if h : rv_spaces <= total_spaces - cars then
    -- The probability calculation logic would go here
    16 / 51
  else
    0

-- The theorem stating the desired result
theorem uncle_bob_can_park : probability_RV_can_park total_spaces cars rv_spaces = 16 / 51 :=
  sorry

end uncle_bob_can_park_l7_7286


namespace cake_sharing_l7_7101

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l7_7101


namespace isosceles_triangle_leg_less_than_five_times_base_l7_7386

theorem isosceles_triangle_leg_less_than_five_times_base
(base leg : ℝ) (h1 : isosceles_triangle_with_base_and_leg_and_vertex_angle base leg (12 * π / 180)) :
  leg < 5 * base := 
by sorry

end isosceles_triangle_leg_less_than_five_times_base_l7_7386


namespace sqrt_ineq_l7_7533

theorem sqrt_ineq (m n : ℕ) : max (n^(1/m : ℝ)) (m^(1/n : ℝ)) ≤ 3^(1/3 : ℝ) := 
sorry

end sqrt_ineq_l7_7533


namespace total_money_calculated_l7_7720

namespace PastryShop

def original_price_cupcake : ℝ := 3.00
def original_price_cookie : ℝ := 2.00
def reduced_price (original_price : ℝ) : ℝ := original_price / 2
def num_cupcakes_sold : ℕ := 16
def num_cookies_sold : ℕ := 8

def total_money_made : ℝ :=
  reduced_price original_price_cupcake * num_cupcakes_sold + 
  reduced_price original_price_cookie * num_cookies_sold

theorem total_money_calculated :
  total_money_made = 32 := by
  sorry

end PastryShop

end total_money_calculated_l7_7720


namespace regular_milk_cartons_l7_7645

variable (R C : ℕ)
variable (h1 : C + R = 24)
variable (h2 : C = 7 * R)

theorem regular_milk_cartons : R = 3 :=
by
  sorry

end regular_milk_cartons_l7_7645


namespace functional_relationship_max_annual_profit_l7_7278

namespace FactoryProfit

-- Definitions of conditions
def fixed_annual_investment : ℕ := 100
def unit_investment : ℕ := 1
def sales_revenue (x : ℕ) : ℕ :=
  if x > 20 then 260 
  else 33 * x - x^2

def annual_profit (x : ℕ) : ℤ :=
  let revenue := sales_revenue x
  let total_investment := fixed_annual_investment + x
  revenue - total_investment

-- Statements to prove
theorem functional_relationship (x : ℕ) (hx : x > 0) :
  annual_profit x =
  if x ≤ 20 then
    (-x^2 : ℤ) + 32 * x - 100
  else
    160 - x :=
by sorry

theorem max_annual_profit : 
  ∃ x, annual_profit x = 144 ∧
  ∀ y, annual_profit y ≤ 144 :=
by sorry

end FactoryProfit

end functional_relationship_max_annual_profit_l7_7278


namespace intersection_points_of_internal_tangents_l7_7596

theorem intersection_points_of_internal_tangents (r1 r2 : ℝ) (l : set (ℝ × ℝ)) :
  (∀ (O1 O2 M : ℝ × ℝ),
    (∃ t : ℝ, O1 = (t, 0) ∧ O2 = (t + r1 + r2, 0))
    ∧ (M.1 - O1.1) = (r1/(r1 + r2)) * (O2.1 - O1.1)
    ∧ (abs ((M.2 - O1.2) * (O2.1 - O1.1) - (M.1 - O1.1) * (O2.2 - O1.2)) = 0))
    → (abs (l.distance_to M) = 2 * r1 * r2 / (r1 + r2)) := sorry

end intersection_points_of_internal_tangents_l7_7596


namespace possible_number_of_classmates_l7_7130

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l7_7130


namespace number_of_valid_squares_l7_7573

def set_H : set (ℤ × ℤ) := { p | let (x, y) := p in 1 ≤ |x| ∧ |x| ≤ 9 ∧ 1 ≤ |y| ∧ |y| ≤ 9 }

def valid_square (p1 p2 p3 p4 : ℤ × ℤ) : Prop :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  let (x3, y3) := p3 in
  let (x4, y4) := p4 in
  (x1 = x2 ∧ x3 = x4 ∧ y1 = y3 ∧ y2 = y4) ∧
  abs (x2 - x1) = abs (y3 - y1) ∧
  abs (x2 - x1) ≥ 8

theorem number_of_valid_squares :
  (∃ n, n = 20 ∧ 
    ∀ (p1 p2 p3 p4 : ℤ × ℤ), 
    p1 ∈ set_H ∧ p2 ∈ set_H ∧ p3 ∈ set_H ∧ p4 ∈ set_H →
    valid_square p1 p2 p3 p4 → true) :=
sorry

end number_of_valid_squares_l7_7573


namespace sum_of_Q_and_R_in_base_8_l7_7845

theorem sum_of_Q_and_R_in_base_8 (P Q R : ℕ) (hp : 1 ≤ P ∧ P < 8) (hq : 1 ≤ Q ∧ Q < 8) (hr : 1 ≤ R ∧ R < 8) 
  (hdistinct : P ≠ Q ∧ Q ≠ R ∧ P ≠ R) (H : 8^2 * P + 8 * Q + R + (8^2 * R + 8 * Q + P) + (8^2 * Q + 8 * P + R) 
  = 8^3 * P + 8^2 * P + 8 * P) : Q + R = 7 := 
sorry

end sum_of_Q_and_R_in_base_8_l7_7845


namespace possible_number_of_classmates_l7_7125

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l7_7125


namespace two_digit_integer_one_less_than_lcm_of_3_4_7_l7_7609

theorem two_digit_integer_one_less_than_lcm_of_3_4_7 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n + 1) % (Nat.lcm (Nat.lcm 3 4) 7) = 0 ∧ n = 83 := by
  sorry

end two_digit_integer_one_less_than_lcm_of_3_4_7_l7_7609


namespace expected_heads_after_tosses_l7_7029

noncomputable def expected_heads (total_coins : ℕ) (tosses : ℕ) : ℚ :=
  let probability_of_heads : ℚ := 1/2 + 1/4 + 1/8 + 1/16  in
  total_coins * probability_of_heads

theorem expected_heads_after_tosses :
  expected_heads 64 4 = 60 := by
  sorry

end expected_heads_after_tosses_l7_7029


namespace year_2017_cycle_l7_7550

-- Definitions based on conditions
def heavenly_stems : list string := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
def earthly_branches : list string := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

def year_2016_stem := "Bing"
def year_2016_branch := "Shen"

-- Function to find the next element in a cyclic list
def next_in_cycle {α : Type} (cycle : list α) (current : α) : α :=
  cycle.nth ((cycle.index_of current + 1) % cycle.length) or_else current

-- Theorem based on the conditions and the correct answer
theorem year_2017_cycle : 
  next_in_cycle heavenly_stems year_2016_stem = "Ding" ∧
  next_in_cycle earthly_branches year_2016_branch = "You" :=
by {
  sorry -- Proof placeholder
}

end year_2017_cycle_l7_7550


namespace highest_prob_of_red_card_l7_7246

theorem highest_prob_of_red_card :
  let deck_size := 52
  let num_aces := 4
  let num_hearts := 13
  let num_kings := 4
  let num_reds := 26
  -- Event probabilities
  let prob_ace := num_aces / deck_size
  let prob_heart := num_hearts / deck_size
  let prob_king := num_kings / deck_size
  let prob_red := num_reds / deck_size
  prob_red > prob_heart ∧ prob_heart > prob_ace ∧ prob_ace = prob_king :=
sorry

end highest_prob_of_red_card_l7_7246


namespace geom_progression_common_ratio_l7_7876

theorem geom_progression_common_ratio (x y z r : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : ∃ a, a ≠ 0 ∧ x * (2 * y - z) = a ∧ y * (2 * z - x) = a * r ∧ z * (2 * x - y) = a * r^2) :
  r^2 + r + 1 = 0 :=
sorry

end geom_progression_common_ratio_l7_7876


namespace inequality_solution_set_ab2_bc_ca_a3b_ge_1_4_l7_7268

theorem inequality_solution_set (x : ℝ) : (|x - 1| + |2 * x + 5| < 8) ↔ (-4 < x ∧ x < 4 / 3) :=
by
  sorry

theorem ab2_bc_ca_a3b_ge_1_4 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  (a^2 / (b + 3 * c) + b^2 / (c + 3 * a) + c^2 / (a + 3 * b) ≥ 1 / 4) :=
by
  sorry

end inequality_solution_set_ab2_bc_ca_a3b_ge_1_4_l7_7268


namespace sewing_project_thread_length_l7_7535

noncomputable def final_thread_length_in_inches (initial_length_cm : ℝ) : ℝ :=
  let added_length := (2 / 3) * initial_length_cm
  let new_length := initial_length_cm + added_length
  let multiplied_length := (3 / 4) * new_length
  let divided_length := multiplied_length * (2 / 3)
  let removed_length := (1 / 4) * divided_length
  let remaining_length_cm := divided_length - removed_length
  remaining_length_cm / 2.54

theorem sewing_project_thread_length : final_thread_length_in_inches 12 ≈ 2.95 :=
by
  sorry

end sewing_project_thread_length_l7_7535


namespace p_is_necessary_but_not_sufficient_for_q_l7_7509

variable (m : ℝ)

-- Given conditions
def p : Prop := ∀ x : ℝ, (3 * m * x^2 + 6 * x - 1) ≤ 0
def q : Prop := m < -3

-- The proof statement we need to show
theorem p_is_necessary_but_not_sufficient_for_q : (p → q) ∧ ¬(q → p) := 
sorry

end p_is_necessary_but_not_sufficient_for_q_l7_7509


namespace simplest_quadratic_radical_l7_7621

def is_simplest_radical (r : ℝ) : Prop :=
  ∀ x : ℝ, (∃ a b : ℝ, r = a * Real.sqrt b) → (∃ c d : ℝ, x = c * Real.sqrt d) → a ≤ c

def sqrt_12 := Real.sqrt 12
def sqrt_2_3 := Real.sqrt (2 / 3)
def sqrt_0_3 := Real.sqrt 0.3
def sqrt_7 := Real.sqrt 7

theorem simplest_quadratic_radical :
  is_simplest_radical sqrt_7 ∧
  ¬ is_simplest_radical sqrt_12 ∧
  ¬ is_simplest_radical sqrt_2_3 ∧
  ¬ is_simplest_radical sqrt_0_3 :=
by
  sorry

end simplest_quadratic_radical_l7_7621


namespace chessboard_transformation_irreversible_l7_7600

def transformation (n : ℕ) (A : ℕ × ℕ → ℤ) : ℕ × ℕ → ℤ :=
  λ (i j : ℕ), 
      if i = 0 ∨ i = n - 1 ∨ j = 0 ∨ j = n - 1
      then 0    -- Assuming boundary condition cells do not change
      else A (i-1, j) * A (i+1, j) * A (i, j-1) * A (i, j+1)

theorem chessboard_transformation_irreversible :
  ¬ ∃ N : ℕ, ∀ A : ℕ × ℕ → ℤ, A = transformation 9^2 N A := 
by
  sorry

end chessboard_transformation_irreversible_l7_7600


namespace range_on_domain01_range_on_domain14over_l7_7401

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem range_on_domain01 : (∀ y ∈ set.range (λ x : set.Icc (0 : ℝ) 1, f x), 1 ≤ y ∧ y ≤ 2) ∧
                            (∀ y : ℝ, (1 ≤ y ∧ y ≤ 2) → ∃ x ∈ set.Icc (0 : ℝ) 1, y = f x) :=
sorry

theorem range_on_domain14over : (∀ y ∈ set.range (λ x : set.Icc (1/4 : ℝ) 2, f x), 1 ≤ y ∧ y ≤ 2) ∧
                                (∀ y : ℝ, (1 ≤ y ∧ y ≤ 2) → ∃ x ∈ set.Icc (1/4 : ℝ) 2, y = f x) :=
sorry

end range_on_domain01_range_on_domain14over_l7_7401


namespace problem1_problem2_l7_7636

-- Problem 1
theorem problem1 (x : ℝ) : 
  (1 - 2 * sin x * cos x) / (cos x^2 - sin x^2) = (1 - tan x) / (1 + tan x) :=
by sorry

-- Problem 2
theorem problem2 (θ a b : ℝ) 
  (h1 : tan θ + sin θ = a) 
  (h2 : tan θ - sin θ = b) : 
  (a^2 - b^2)^2 = 16 * a * b :=
by sorry

end problem1_problem2_l7_7636


namespace second_concert_attendance_l7_7961

def first_concert_attendance : ℕ := 65899
def additional_people : ℕ := 119

theorem second_concert_attendance : first_concert_attendance + additional_people = 66018 := 
by 
  -- Proof is not discussed here, only the statement is required.
sorry

end second_concert_attendance_l7_7961


namespace probability_lt_one_third_l7_7857

theorem probability_lt_one_third :
  let total_interval := (0 : ℝ, 1/2)
  let desired_interval := (0 : ℝ, 1/3)
  let total_length := (total_interval.2 - total_interval.1 : ℝ) = 1/2
  let desired_length := (desired_interval.2 - desired_interval.1 : ℝ) = 1/3
  -- then the probability P is given by:
  (desired_length / total_length) = 2/3
:=
by {
  -- definition of intervals and lengths
  let total_interval := (0 : ℝ, 1/2)
  let desired_interval := (0 : ℝ, 1/3)
  have total_length : total_interval.2 - total_interval.1 = 1/2,
  -- verify total length calculation
  calc
    total_interval.2 - total_interval.1 = 1/2 - 0 : by simp
                                 ... = 1/2      : by norm_num,
  have desired_length : desired_interval.2 - desired_interval.1 = 1/3,
  -- verify desired interval length calculation
  calc
    desired_interval.2 - desired_interval.1 = 1/3 - 0 : by simp
                                    ... = 1/3      : by norm_num,
  -- calculate probability
  set P := desired_length / total_length
  -- compute correct answer
  have : P = (1/3) / (1/2),
  calc
    (1/3) / (1/2) = (1/3) * (2/1) : by field_simp
              ...  = 2/3      : by norm_num,
  exact this
}

end probability_lt_one_third_l7_7857


namespace sum_of_angles_l7_7196

-- Definitions of acute, right, and obtuse angles
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def is_right (θ : ℝ) : Prop := θ = 90
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- The main statement we want to prove
theorem sum_of_angles :
  (∀ (α β : ℝ), is_acute α ∧ is_acute β → is_acute (α + β) ∨ is_right (α + β) ∨ is_obtuse (α + β)) ∧
  (∀ (α β : ℝ), is_acute α ∧ is_right β → is_obtuse (α + β)) :=
by sorry

end sum_of_angles_l7_7196


namespace find_slope_of_line_l_l7_7783

theorem find_slope_of_line_l :
  ∃ k : ℝ, (k = 3 * Real.sqrt 5 / 10 ∨ k = -3 * Real.sqrt 5 / 10) :=
by
  -- Given conditions
  let F1 : ℝ := 6 / 5 * Real.sqrt 5
  let PF : ℝ := 4 / 5 * Real.sqrt 5
  let slope_PQ : ℝ := 1
  let slope_RF1 : ℝ := sorry  -- we need to prove/extract this from the given
  let k := 3 / 2 * slope_RF1
  -- to prove this
  sorry

end find_slope_of_line_l_l7_7783


namespace line_through_A_B_circle_with_center_on_y_axis_l7_7426

-- Definitions for points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 3)

-- Line equation passing through points A and B
theorem line_through_A_B : ∃ m b : ℝ, (A.1, A.2) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b} ∧ (B.1, B.2) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b} ∧ ∀ x y, y = m * x + b → x - y + 2 = 0 :=
by sorry

-- Circle equation with center on the y-axis
theorem circle_with_center_on_y_axis : ∃ a r : ℝ, (A.1, A.2).fst² + ((A.1).snd - a)² = r² ∧ (B.1, B.2).fst² + ((B.1).snd - a)² = r² ∧ (a = 2) ∧ (r = sqrt 2) ∧ ∀ x y, (x² + (y - 2)² = 2) :=
by sorry

end line_through_A_B_circle_with_center_on_y_axis_l7_7426


namespace repeating_six_equals_fraction_l7_7746

theorem repeating_six_equals_fraction : ∃ f : ℚ, (∀ n : ℕ, (n ≥ 1 → (6 * (10 : ℕ) ^ (-n) : ℚ) + (f - (6 * (10 : ℕ) ^ (-n) : ℚ)) = f)) ∧ f = 2 / 3 := sorry

end repeating_six_equals_fraction_l7_7746


namespace lincoln_high_school_overlap_l7_7305

theorem lincoln_high_school_overlap 
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (either_club : ℕ)
  (H1 : total_students = 250)
  (H2 : drama_club = 100)
  (H3 : science_club = 130)
  (H4 : either_club = 210) : 
  ∃ both_clubs : ℕ, both_clubs = 20 :=
by
  use 230 - 210
  sorry 

end lincoln_high_school_overlap_l7_7305


namespace angle_MKR_is_15_degrees_l7_7887

-- Definitions of the conditions given in the problem
variables (P Q R K M : Type*)
variables (triangle_PQR : Triangle P Q R)

-- Conditions for given angles
variable (angle_P : angle triangle_PQR P = 80)
variable (angle_Q : angle triangle_PQR Q = 70)
variable (angle_R : angle triangle_PQR R = 30)

-- Definitions of altitude and median
variable (altitude_PK : Altitude triangle_PQR PK)
variable (median_QM : Median triangle_PQR QM)

-- Statement to prove
theorem angle_MKR_is_15_degrees :
  angle MKR = 15 :=
sorry

end angle_MKR_is_15_degrees_l7_7887


namespace possible_number_of_classmates_l7_7124

theorem possible_number_of_classmates
  (h1 : ∃ x : ℝ, x = 1 / 11)
  (h2 : ∃ y : ℝ, y = 1 / 14)
  : ∃ n : ℕ, (n = 12 ∨ n = 13) :=
begin
  sorry
end

end possible_number_of_classmates_l7_7124


namespace angle_solution_l7_7018

/-!
  Given:
  k + 90° = 360°

  Prove:
  k = 270°
-/

theorem angle_solution (k : ℝ) (h : k + 90 = 360) : k = 270 :=
by
  sorry

end angle_solution_l7_7018


namespace number_of_classmates_ate_cake_l7_7142

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l7_7142


namespace segment_length_of_absolute_value_l7_7204

theorem segment_length_of_absolute_value (x : ℝ) (h : abs (x - (27 : ℝ)^(1/3)) = 5) : 
  |8 - (-2)| = 10 :=
by
  sorry

end segment_length_of_absolute_value_l7_7204


namespace combined_share_correct_l7_7626

-- Definitions for conditions
def total_money := 12000
def ratio_a := 2
def ratio_b := 4
def ratio_c := 3
def ratio_d := 1
def ratio_e := 5

-- Sum of all parts in the ratio
def total_parts := ratio_a + ratio_b + ratio_c + ratio_d + ratio_e

-- The value of one part
def value_per_part : ℝ := total_money / total_parts

-- Individual shares based on the ratio
def share_d := ratio_d * value_per_part
def share_e := ratio_e * value_per_part

-- The combined share of children d and e
def combined_share_d_e := share_d + share_e

-- Prove that the combined share of d and e is $4800
theorem combined_share_correct : combined_share_d_e = 4800 := by
  sorry

end combined_share_correct_l7_7626


namespace find_y_values_l7_7774

theorem find_y_values (x y : ℝ) 
  (h1 : 3 * x^2 + 9 * x + 3 * y + 2 = 0)
  (h2 : 3 * x + y + 4 = 0) :
  y = -4 + real.sqrt 30 ∨ y = -4 - real.sqrt 30 :=
by 
  sorry

end find_y_values_l7_7774


namespace probability_of_interval_l7_7869

noncomputable def probability_less_than (a b : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < b then (x - a) / (b - a) else 0

theorem probability_of_interval (a b x : ℝ) (ha : 0 < a) (hb : a < b) :
  probability_less_than 0 (1/2) (1/3) = 2/3 :=
by
  have h_interval : (1 : ℝ)/2 - 0 = (1/2) := by norm_num,
  have h_favorable : (1 : ℝ)/3 - 0 = (1/3) := by norm_num,
  rw [← h_interval, ← h_favorable, probability_less_than],
  split_ifs,
  simpa using div_eq_mul_one_div (1/3) (1/2),
  sorry

end probability_of_interval_l7_7869


namespace min_value_of_quadratic_l7_7825

theorem min_value_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, y = x^2 - 8 * x + 15 → y ≥ -1) ∧ (∃ x₀ : ℝ, x₀ = 4 ∧ (x₀^2 - 8 * x₀ + 15 = -1)) :=
by
  sorry

end min_value_of_quadratic_l7_7825


namespace red_balls_estimate_l7_7011

/-- There are several red balls and 4 black balls in a bag.
Each ball is identical except for color.
A ball is drawn and put back into the bag. This process is repeated 100 times.
Among those 100 draws, 40 times a black ball is drawn.
Prove that the number of red balls (x) is 6. -/
theorem red_balls_estimate (x : ℕ) (h_condition : (4 / (4 + x) = 40 / 100)) : x = 6 :=
by
    sorry

end red_balls_estimate_l7_7011


namespace sequence_inequality_l7_7383

-- Define the sequence a_n such that a_(n+1) - a_n = 1 and a_1 = 1
def a : ℕ → ℕ
| 0     => 1
| (n+1) => a n + 1

noncomputable def sum_of_reciprocals (n : ℕ) : ℝ :=
∑ k in Finset.range (2^n), (1 : ℝ) / a (k+1)

theorem sequence_inequality (n : ℕ) (h : n > 0) :
  sum_of_reciprocals n ≥ (n + 2) / 2 :=
sorry

end sequence_inequality_l7_7383


namespace number_of_five_digit_even_numbers_l7_7565

theorem number_of_five_digit_even_numbers : 
  ∃ n : ℕ, n = (number_of_permutations (finset.filter (λ x, x ∈ {2, 4}) (finset.range 6)).card * 24) ∧ n = 48 := 
by 
  let digits := {1, 2, 3, 4, 5}
  let evens := digits.filter (λ x, x % 2 = 0)
  have h1 : evens = {2, 4} := by simp
  have h2 : evens.card = 2 := by simp [h1]
  have h3 : finset.powerset_len (digits.card - evens.card) digits.card = 24 := 
    by simp [digits.card, evens.card]; sorry
  exact ⟨_, mul_comm h2 h3, sorry⟩

end number_of_five_digit_even_numbers_l7_7565


namespace possible_classmates_l7_7133

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l7_7133


namespace average_additional_minutes_per_day_l7_7897

variable (Asha_times : List ℚ := [40, 60, 50, 70, 30, 55, 45])
variable (Sasha_times : List ℚ := [50, 70, 40, 100, 10, 55, 0])

theorem average_additional_minutes_per_day :
  (Sasha_times.zip Asha_times).map (λ pair => pair.1 - pair.2).sum / (Sasha_times.length) = -25 / 7 :=
by
  sorry

end average_additional_minutes_per_day_l7_7897


namespace repeating_decimal_sum_l7_7698
noncomputable def repeater := sorry  -- Definitions of repeating decimals would be more complex in Lean, skipping implementation.

-- Define the repeating decimals as constants for the purpose of this proof
def decimal1 : ℚ := 1 / 3
def decimal2 : ℚ := 2 / 3

-- Define the main theorem
theorem repeating_decimal_sum : decimal1 + decimal2 = 1 := by
  sorry  -- Placeholder for proof

end repeating_decimal_sum_l7_7698


namespace diameter_of_circle_C_l7_7710

-- Define the conditions

def circle_radius_D : ℝ := 10
def area_circle_D : ℝ := 100 * Real.pi
def ratio_shaded_to_circle_C : ℝ := 4

-- Statement to prove the diameter of circle C
theorem diameter_of_circle_C (d : ℝ) :
  let radius_C := d / 2 in
  let area_circle_C := Real.pi * (radius_C ^ 2) in
  let shaded_area := ratio_shaded_to_circle_C * area_circle_C in
  shaded_area + area_circle_C = area_circle_D → 
  d = 8 * Real.sqrt 5 :=
sorry

end diameter_of_circle_C_l7_7710


namespace const_product_AN_AN_l7_7928

variable {O B B' A M M' N N' : Point}
variable {c : Circle O}
variable (hA : OnLine A (DiameterLine B B'))
variable (hSecant : Secant A M M' c)
variable (hPerpendicular1 : Perpendicular (LineThroughPoints A N) (LineThroughPoints B B'))
variable (hPerpendicular2 : Perpendicular (LineThroughPoints A N') (LineThroughPoints B B'))

theorem const_product_AN_AN' : 
  (A.dist N) * (A.dist N') = (A.dist B) * (A.dist B') := 
  sorry

end const_product_AN_AN_l7_7928


namespace Alex_last_5_shots_l7_7298

theorem Alex_last_5_shots:
  let initial_attempts := 50 in
  let initial_percentage := 0.60 in
  let additional_attempts_1 := 10 in
  let additional_percentage_1 := 0.62 in
  let additional_attempts_2 := 5 in
  let final_percentage := 0.60 in
  let initial_made := initial_percentage * initial_attempts in
  let total_after_10 := initial_attempts + additional_attempts_1 in
  let made_after_10 := additional_percentage_1 * total_after_10 in
  let made_during_10 := made_after_10 - initial_made in
  let total_attempts := total_after_10 + additional_attempts_2 in
  let final_made := final_percentage * total_attempts in
  let made_during_5 := final_made - made_after_10 in
  made_during_5 = 2 :=
by
  sorry

end Alex_last_5_shots_l7_7298


namespace repeating_six_as_fraction_l7_7745

theorem repeating_six_as_fraction : (∑' n : ℕ, 6 / (10 * (10 : ℝ)^n)) = (2 / 3) :=
by
  sorry

end repeating_six_as_fraction_l7_7745


namespace maximum_candies_in_34_minutes_l7_7187

/-- There are initially 34 ones written on a board. Each minute, Carlson erases two arbitrary numbers,
    writes their sum on the board, and consumes a number of candies equal to the product of the two erased numbers.
    Prove that the maximum number of candies Carlson could eat in 34 minutes is 561. -/
theorem maximum_candies_in_34_minutes :
  ∃ (c : ℕ), c = 561 ∧ (∀ (board : list ℕ) (minutes_remaining : ℕ),
    board.length = 34 →
    minutes_remaining = 34 →
    (∀ (x y : ℕ), x ∈ board → y ∈ board → 
      ∃ (new_board : list ℕ), 
      (board.erase x).erase y = new_board.erase (x + y) ∧
      ∃ (candies : ℕ), candies = x * y)) →
    c = 561 := 
begin
  sorry
end

end maximum_candies_in_34_minutes_l7_7187


namespace domain_correct_l7_7165

noncomputable def domain_of_function : Set ℝ :=
  { x : ℝ | 2 ≤ x ∧ x < 3 }

theorem domain_correct (x : ℝ) :
  (3 - x > 0 ∧ 2^x - 4 ≥ 0) ↔ (2 ≤ x ∧ x < 3) :=
by
  sorry

end domain_correct_l7_7165


namespace max_elements_A_l7_7502

theorem max_elements_A :
  ∀ (A : Set ℕ), (∀ (x y : ℕ), x ∈ A → y ∈ A → |x - y| ≥ (x * y) / 30) →
    Set.Finite A → ∃ n, n ≤ 10 ∧ ¬∃ B, B ⊃ A ∧ ∀ (x y : ℕ), x ∈ B → y ∈ B → |x - y| ≥ (x * y) / 30 := 
sorry

end max_elements_A_l7_7502


namespace scaled_multiplication_l7_7371

theorem scaled_multiplication
  (h : 14.97 * 46 = 688.62) :
  1.497 * 4.6 = 6.8862 :=
by
  sorry

end scaled_multiplication_l7_7371


namespace range_of_x0_l7_7064

def A : set ℝ := {x | 0 ≤ x ∧ x < 1}
def B : set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ A then 2^x
  else if x ∈ B then 4 - 2*x
  else 0

theorem range_of_x0 :
  ∀ (x₀ : ℝ), x₀ ∈ A ∧ f (f x₀) ∈ A → (real.log 3 - real.log 2) / real.log 2 < x₀ ∧ x₀ < 1 :=
by 
  sorry

end range_of_x0_l7_7064


namespace dominos_sum_condition_satisfied_l7_7689

-- Represent a domino as a pair of integers (left side dots, right side dots)
structure Domino := (left : ℕ) (right : ℕ)

-- Define a list of dominos representing the arrangement
def dominos : List Domino := [
  -- fill in with appropriate values
]

-- Define a predicate to check the sum condition for any three consecutive dominoes
def sum_condition (dominos : List Domino) (i : ℕ) : Prop :=
  dominos.get_or_else i ⟨0, 0⟩.left + dominos.get_or_else (i+1) ⟨0, 0⟩.left + dominos.get_or_else (i+2) ⟨0, 0⟩.left = 
  dominos.get_or_else i ⟨0, 0⟩.right + dominos.get_or_else (i+1) ⟨0, 0⟩.right + dominos.get_or_else (i+2) ⟨0, 0⟩.right

-- The main statement to prove
theorem dominos_sum_condition_satisfied : 
  ∃ dominos : List Domino, dominos.length = 28 ∧ 
  ∀ i, i < 26 → sum_condition dominos i := 
sorry

end dominos_sum_condition_satisfied_l7_7689


namespace I_is_circumcenter_l7_7984

variables (A B C A1 B1 C1 I Ia Ib Ic A2 B2 C2 : Type)

-- Define that points are distinct and lines are constructed as described
axiom point_distinct : ∀ p1 p2 : Type, p1 ≠ p2
axiom incircle_touching : ∀ I A B C bc ca ab A1 B1 C1 : Type, true
axiom excenter_touching : ∀ Ia Ib Ic A B C bc ca ab : Type, true
axiom C2_intersection : ∀ Ia B1 Ib A1 C2 : Type, true
axiom A2_intersection : ∀ Ib C1 Ic B1 A2 : Type, true
axiom B2_intersection : ∀ Ic A1 Ia C1 B2 : Type, true

-- Given these points and constraints, we need to prove that I is the circumcenter
theorem I_is_circumcenter : ∀ (I A2 B2 C2 : Type) 
  (H_inc : incircle_touching I A B C bc ca ab A1 B1 C1)
  (H_exc : excenter_touching Ia Ib Ic A B C bc ca ab)
  (H_C2 : C2_intersection Ia B1 Ib A1 C2)
  (H_A2 : A2_intersection Ib C1 Ic B1 A2)
  (H_B2 : B2_intersection Ic A1 Ia C1 B2), 
  true := sorry

end I_is_circumcenter_l7_7984


namespace linear_function_no_third_quadrant_l7_7880

theorem linear_function_no_third_quadrant (k b : ℝ) (h : ∀ x y : ℝ, x < 0 → y < 0 → y ≠ k * x + b) : k < 0 ∧ 0 ≤ b :=
sorry

end linear_function_no_third_quadrant_l7_7880


namespace classmates_ate_cake_l7_7098

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l7_7098


namespace repeating_decimal_to_fraction_l7_7753

theorem repeating_decimal_to_fraction : (let a := (6 : Real) / 10 in
                                         let r := (1 : Real) / 10 in
                                         ∑' n : ℕ, a * r^n) = (2 : Real) / 3 :=
by
  sorry

end repeating_decimal_to_fraction_l7_7753


namespace expected_non_empty_urns_correct_l7_7464

open ProbabilityTheory

noncomputable def expected_non_empty_urns (n k : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n) ^ k)

theorem expected_non_empty_urns_correct (n k : ℕ) : expected_non_empty_urns n k = n * (1 - ((n - 1) / n) ^ k) :=
by 
  sorry

end expected_non_empty_urns_correct_l7_7464


namespace M_gt_N_l7_7844

variable (a b : ℝ)

def M := 10 * a^2 + 2 * b^2 - 7 * a + 6
def N := a^2 + 2 * b^2 + 5 * a + 1

theorem M_gt_N : M a b > N a b := by
  sorry

end M_gt_N_l7_7844


namespace calc_q1_plus_r_neg1_l7_7499

noncomputable def f : Polynomial ℚ := 4 * X^4 + 12 * X^3 - 9 * X^2 + X + 3
noncomputable def d : Polynomial ℚ := X^3 + 3 * X - 2
noncomputable def q : Polynomial ℚ := 4 * X
noncomputable def r : Polynomial ℚ := 4 * X + 1

theorem calc_q1_plus_r_neg1 : q.eval 1 + r.eval (-1) = 1 := by
  have h_decomposition : f = q * d + r := by
    -- Polynomial arithmetic to check the equality
    sorry
  have h_deg_r : r.degree < d.degree := by
    -- Degree arithmetic to check the degrees
    sorry
  rw [Polynomial.eval_add, Polynomial.eval_mul]
  simp
  sorry

end calc_q1_plus_r_neg1_l7_7499


namespace sqrt_meaningful_iff_x_ge_5_l7_7846

theorem sqrt_meaningful_iff_x_ge_5 (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 5)) ↔ x ≥ 5 :=
by sorry

end sqrt_meaningful_iff_x_ge_5_l7_7846


namespace multiple_of_six_l7_7052

theorem multiple_of_six
  (a d : ℤ)
  (h1 : prime a)
  (h2 : prime (a + d))
  (h3 : prime (a + 2 * d))
  (h4 : a > 3)
  (h5 : a + d > 3)
  (h6 : a + 2 * d > 3) :
  ∃ k : ℤ, d = 6 * k :=
by
  sorry

end multiple_of_six_l7_7052


namespace inscribed_circle_radius_correct_l7_7275

noncomputable def inscribed_circle_radius (α : ℝ) : ℝ :=
  let r := 15 * tan (α / 2) / (3 + 4 * (tan (α / 2))^2)
  r

theorem inscribed_circle_radius_correct (α : ℝ) (h1 : has_incircle ABC)
  (h2 : tangent_to_incircle_DE_parallel_to_BC)
  (h3 : perimeter ABC = 40)
  (h4 : perimeter ADE = 30)
  (h5 : ∠ABC = α) :
  ∃ r : ℝ, r = 15 * tan (α / 2) / (3 + 4 * (tan (α / 2))^2) :=
by
  use inscribed_circle_radius α
  sorry

end inscribed_circle_radius_correct_l7_7275


namespace segment_length_eq_ten_l7_7209

theorem segment_length_eq_ten (x : ℝ) (h : |x - 3| = 5) : |8 - (-2)| = 10 :=
by {
  sorry
}

end segment_length_eq_ten_l7_7209


namespace subsequence_non_decreasing_or_non_increasing_l7_7494

theorem subsequence_non_decreasing_or_non_increasing
    (a : ℕ → ℝ) : 
    ∃ f : ℕ → ℕ, strict_mono f ∧ 
        (∀ n m, n ≤ m → a (f n) ≤ a (f m)) ∨
        (∀ n m, n ≤ m → a (f n) ≥ a (f m)) :=
by
  sorry

end subsequence_non_decreasing_or_non_increasing_l7_7494


namespace union_P_Q_l7_7062

open Set

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5, 6}

theorem union_P_Q : P ∪ Q = {1, 2, 3, 4, 5, 6} :=
by 
  -- Proof goes here
  sorry

end union_P_Q_l7_7062


namespace totalSelectionSchemes_l7_7016

-- Definition of the available roles
inductive Role where
  | Translator
  | TourGuide
  | Etiquette
  | Driver

-- Volunteers
inductive Volunteer where
  | A
  | B
  | C
  | D
  | E

-- Roles that A and B can take (only first three)
def canTakeRole : Volunteer → (Role → Prop)
  | Volunteer.A => λ r => r ≠ Role.Driver
  | Volunteer.B => λ r => r ≠ Role.Driver
  | _ => λ _ => True

-- Definition of the problem: Total number of ways to assign roles
theorem totalSelectionSchemes : (totalAssignments canTakeRole) = 72 := 
sorry

end totalSelectionSchemes_l7_7016


namespace equal_distances_l7_7047

noncomputable def projection_point (O : Point) (l : Line) : Point := sorry
noncomputable def secant_through_points (p1 p2 : Point) (circle : Circle) : (Point × Point) := sorry
noncomputable def line_intersection (l1 l2 : Line) : Point := sorry
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

theorem equal_distances (O l : Line) (B C P Q M N R S : Point) (circle : Circle)
  (hA_proj : A = projection_point O l)
  (hAB_eq_AC : distance A B = distance A C)
  (hSecant_B : (secant_through_points B (circle)).1 = P) (secant_through_points B (circle)).2 = Q)
  (hSecant_C : (secant_through_points C (circle)).1 = M) (secant_through_points C (circle)).2 = N)
  (hR : line_intersection (line_through_points N P) l = R)
  (hS : line_intersection (line_through_points M Q) l = S)
  : distance R A = distance A S := 
sorry

end equal_distances_l7_7047


namespace estimate_red_balls_l7_7005

theorem estimate_red_balls (x : ℕ) (drawn_black_balls : ℕ) (total_draws : ℕ) (black_balls : ℕ) 
  (h1 : black_balls = 4) 
  (h2 : total_draws = 100) 
  (h3 : drawn_black_balls = 40) 
  (h4 : (black_balls : ℚ) / (black_balls + x) = drawn_black_balls / total_draws) : 
  x = 6 := 
sorry

end estimate_red_balls_l7_7005


namespace number_of_classmates_ate_cake_l7_7143

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l7_7143


namespace shaded_area_in_two_foot_length_pattern_l7_7537

theorem shaded_area_in_two_foot_length_pattern 
  (d : ℝ := 4) -- diameter of semicircles in inches
  (l : ℝ := 24) -- length of the pattern in inches
  (n : ℝ := l / d) -- number of semicircles along each boundary
  (r : ℝ := d / 2) -- radius of each semicircle
  : area : ℝ := 6 * 4 * Real.pi := 24 * Real.pi :=
begin
  sorry
end

end shaded_area_in_two_foot_length_pattern_l7_7537


namespace cows_relationship_l7_7601

theorem cows_relationship (H : ℕ) (W : ℕ) (T : ℕ) (hcows : W = 17) (tcows : T = 70) (together : H + W = T) : H = 53 :=
by
  rw [hcows, tcows] at together
  linarith
  -- sorry

end cows_relationship_l7_7601


namespace number_of_solutions_l7_7433

theorem number_of_solutions : ∃ n : ℕ, n = 43 ∧ 
  ∀ x : ℕ, x ∈ {1..50} → (x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 9 ∧ x ≠ 16 ∧ x ≠ 25 ∧ x ≠ 36 ∧ x ≠ 49) → 
  (∃ num den : ℕ, (x - num) * (x - den) = 0) := 
sorry

end number_of_solutions_l7_7433


namespace sum_f_2018_l7_7944

def f (n : ℕ) : ℝ := Real.cos ((n * Real.pi / 2) + (Real.pi / 4))

theorem sum_f_2018 : (∑ n in Finset.range 2018, f (n + 1)) = -Real.sqrt 2 := 
by 
  sorry

end sum_f_2018_l7_7944


namespace part1_part2_l7_7829

-- Definition of curve C1
def C1 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

-- Definition of line l
def l (t : ℝ) : ℝ × ℝ := (1 + t, Real.sqrt 3 * t)

-- Definition of transformed curve C2
def C2 (x y : ℝ) : Prop :=
  (x^2 / (1 / 4) + y^2 / (3 / 4) = 1)

-- Proving |AB|
theorem part1 :
  (∃ t1 t2 : ℝ, (1 + t1, Real.sqrt 3 * t1) ∈ Set.range C1 ∧
                (1 + t2, Real.sqrt 3 * t2) ∈ Set.range C1 ∧
                t1 ≠ t2) →
  |((1 + t1, Real.sqrt 3 * t1) - (1 + t2, Real.sqrt 3 * t2))| = Real.sqrt 13 :=
sorry

-- Proving the minimum distance
theorem part2 :
  ∃ P : ℝ × ℝ, C2 P.1 P.2 →
  ∃ t : ℝ, (1 + t, Real.sqrt 3 * t) →
  dist P (1 + t, Real.sqrt 3 * t) = (2 * Real.sqrt 3 - Real.sqrt 6) / 4 :=
sorry

end part1_part2_l7_7829


namespace frog_arrangement_l7_7189

def arrangementCount (total_frogs green_frogs red_frogs blue_frog : ℕ) : ℕ :=
  if (green_frogs + red_frogs + blue_frog = total_frogs ∧ 
      green_frogs = 3 ∧ red_frogs = 4 ∧ blue_frog = 1) then 40320 else 0

theorem frog_arrangement :
  arrangementCount 8 3 4 1 = 40320 :=
by {
  -- Proof omitted
  sorry
}

end frog_arrangement_l7_7189


namespace segment_length_eq_ten_l7_7210

theorem segment_length_eq_ten (x : ℝ) (h : |x - 3| = 5) : |8 - (-2)| = 10 :=
by {
  sorry
}

end segment_length_eq_ten_l7_7210


namespace total_fruit_punch_l7_7978

/-- Conditions -/
def orange_punch : ℝ := 4.5
def cherry_punch : ℝ := 2 * orange_punch
def apple_juice : ℝ := cherry_punch - 1.5
def pineapple_juice : ℝ := 3
def grape_punch : ℝ := 1.5 * apple_juice

/-- Proof that total fruit punch is 35.25 liters -/
theorem total_fruit_punch :
  orange_punch + cherry_punch + apple_juice + pineapple_juice + grape_punch = 35.25 := by
  sorry

end total_fruit_punch_l7_7978


namespace which_options_imply_inverse_order_l7_7253

theorem which_options_imply_inverse_order (a b : ℝ) :
  ((b > 0 ∧ 0 > a) ∨ (0 > a ∧ a > b) ∨ (a > b ∧ b > 0)) →
  (1 / a < 1 / b) :=
by
  intro h
  cases h
  case inl h1 =>
    have ha : a < 0 := h1.2
    have hb : b > 0 := h1.1
    have hab : a < 0 ∧ 0 < b := ⟨ha, hb⟩
    calc
      1 / a < 1 / b := by sorry
  case inr h2 =>
    cases h2
    case inl h3 =>
      have ha : a < 0 := h3.1.1
      have hb : b < a  := h3.1.2
      have hb_lt_a : b < a ∧ a < 0 := ⟨hb, ha⟩
      calc
        1 / a < 1 / b := by sorry
    case inr h4 =>
      have ha : a > b := h4.1
      have hb : b > 0 := h4.2
      have a_gt_b_and_b_gt_0 : a > b ∧ b > 0 := ⟨ha, hb⟩
      calc
        1 / a < 1 / b := by sorry

end which_options_imply_inverse_order_l7_7253


namespace number_of_classmates_ate_cake_l7_7144

theorem number_of_classmates_ate_cake (n : ℕ) 
  (Alyosha : ℝ) (Alena : ℝ) 
  (H1 : Alyosha = 1 / 11) 
  (H2 : Alena = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
sorry

end number_of_classmates_ate_cake_l7_7144


namespace lcm_1_to_12_l7_7240

theorem lcm_1_to_12 : Nat.lcm_list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 27720 := by
  sorry

end lcm_1_to_12_l7_7240


namespace linear_avoid_third_quadrant_l7_7878

theorem linear_avoid_third_quadrant (k b : ℝ) (h : ∀ x : ℝ, k * x + b ≥ 0 → k * x + b > 0 → (k * x + b ≥ 0) ∧ (x ≥ 0)) :
  k < 0 ∧ b ≥ 0 :=
by
  sorry

end linear_avoid_third_quadrant_l7_7878


namespace oranges_in_total_l7_7310

def number_of_boxes := 3
def oranges_per_box := 8
def total_oranges := 24

theorem oranges_in_total : number_of_boxes * oranges_per_box = total_oranges := 
by {
  -- sorry skips the proof part
  sorry 
}

end oranges_in_total_l7_7310


namespace base_four_30121_eq_793_l7_7554

-- Definition to convert a base-four (radix 4) number 30121_4 to its base-ten equivalent
def base_four_to_base_ten (d4 d3 d2 d1 d0 : ℕ) : ℕ :=
  d4 * 4^4 + d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0

theorem base_four_30121_eq_793 : base_four_to_base_ten 3 0 1 2 1 = 793 := 
by
  sorry

end base_four_30121_eq_793_l7_7554


namespace point_not_in_any_quadrilateral_l7_7971

theorem point_not_in_any_quadrilateral 
  (H : ∀ (V : Fin 7 → ℝ × ℝ), convex_heptagon V) :
  ∃ (p : ℝ × ℝ), p ∈ interior (polygon (Fin 7) V) ∧
    ∀ (i : Fin 7), p ∉ interior (polygon (Fin 4) (λ j, V ((i + j) % 7))) :=
sorry

end point_not_in_any_quadrilateral_l7_7971


namespace possible_classmates_l7_7136

noncomputable def cake_distribution (n : ℕ) : Prop :=
  n > 11 ∧ n < 14 ∧ 
  (let remaining_fraction n := 1 - ((1 / 11) + (1 / 14)) in 
    (if n = 12 then 
      ∀ i, 1 ≤ i ∧ i ≤ 10 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
    else
      if n = 13 then 
        ∀ i, 1 ≤ i ∧ i ≤ 11 → 1 / 14 < remaining_fraction 1 / n < 1 / 11 
     else False))

theorem possible_classmates (n : ℕ) : cake_distribution n ↔ n = 12 ∨ n = 13 := by
  sorry

end possible_classmates_l7_7136


namespace fraction_value_l7_7707

theorem fraction_value : (1 - 1 / 4) / (1 - 1 / 3) = 9 / 8 := 
by sorry

end fraction_value_l7_7707


namespace impossible_to_form_palindrome_l7_7629

-- Define the possible cards
inductive Card
| abc | bca | cab

-- Define the rule for palindrome formation
def canFormPalindrome (w : List Card) : Prop :=
  sorry  -- Placeholder for the actual formation rule

-- Define the theorem statement
theorem impossible_to_form_palindrome (w : List Card) :
  ¬canFormPalindrome w :=
sorry

end impossible_to_form_palindrome_l7_7629


namespace sequence_term_is_100th_term_l7_7482

theorem sequence_term_is_100th_term (a : ℕ → ℝ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, a (n + 1) = 2 * a n / (a n + 2)) :
  (∃ n : ℕ, a n = 2 / 101) ∧ ((∃ n : ℕ, a n = 2 / 101) → n = 100) :=
by
  sorry

end sequence_term_is_100th_term_l7_7482


namespace find_k_l7_7279

def f (n : ℤ) : ℤ :=
if n % 2 = 1 then n + 5 else n / 2

variable (k : ℤ)
hypothesis hk : k % 2 = 1

theorem find_k (h : f (f (f k)) = 35) : k = 55 := by
  sorry

end find_k_l7_7279


namespace middle_and_oldest_son_ages_l7_7660

theorem middle_and_oldest_son_ages 
  (x y z : ℕ) 
  (father_age_current father_age_future : ℕ) 
  (youngest_age_increment : ℕ)
  (father_age_increment : ℕ) 
  (father_equals_sons_sum : father_age_future = (x + youngest_age_increment) + (y + father_age_increment) + (z + father_age_increment))
  (father_age_constraint : father_age_current + father_age_increment = father_age_future)
  (youngest_age_initial : x = 2)
  (father_age_current_value : father_age_current = 33)
  (youngest_age_increment_value : youngest_age_increment = 12)
  (father_age_increment_value : father_age_increment = 12) 
  :
  y = 3 ∧ z = 4 :=
begin
  sorry
end

end middle_and_oldest_son_ages_l7_7660


namespace find_x_condition_l7_7408

theorem find_x_condition :
  ∀ x : ℝ, (x^2 - 1) / (x + 1) = 0 → x = 1 := by
  intros x h
  have num_zero : x^2 - 1 = 0 := by
    -- Proof that the numerator is zero
    sorry
  have denom_nonzero : x ≠ -1 := by
    -- Proof that the denominator is non-zero
    sorry
  have x_solves : x = 1 := by
    -- Final proof to show x = 1
    sorry
  exact x_solves

end find_x_condition_l7_7408


namespace fraction_for_repeating_decimal_l7_7764

variable (a r S : ℚ)
variable (h1 : a = 3/5)
variable (h2 : r = 1/10)
variable (h3 : S = a / (1 - r))

theorem fraction_for_repeating_decimal : S = 2 / 3 :=
by
  have h4 : 1 - r = 9 / 10, from sorry
  have h5 : S = (3 / 5) / (9 / 10), from sorry
  have h6 : S = (3 * 10) / (5 * 9), from sorry
  have h7 : S = 30 / 45, from sorry
  have h8 : 30 / 45 = 2 / 3, from sorry
  exact h8

end fraction_for_repeating_decimal_l7_7764


namespace length_of_segment_l7_7200

theorem length_of_segment (x : ℝ) : 
  |x - (27^(1/3))| = 5 →
  ∃ a b : ℝ, a - b = 10 ∧ (|a - (27^(1/3))| = 5 ∧ |b - (27^(1/3))| = 5) :=
by
  sorry

end length_of_segment_l7_7200


namespace find_value_of_a_l7_7850

theorem find_value_of_a (a : ℚ) : (∃ (b : ℚ), (9 * (x : ℚ) ^ 2 + 27 * x + a) = (3 * x + b) ^ 2) → a = 81 / 4 :=
by
  intros h
  rcases h with ⟨b, hb⟩
  have hx := congr_arg (λ p, p.coeff 1) hb
  linarith
  sorry

end find_value_of_a_l7_7850


namespace minimum_travel_cost_l7_7526

-- Define the distances and costs
def XZ := 4000
def XY := 5000
def cost_bus_per_km := 0.20
def airplane_base_cost := 150
def cost_airplane_per_km := 0.15

-- Calculate the YZ distance
def YZ := Real.sqrt (XY ^ 2 - XZ ^ 2)

-- Calculate the costs for each travel segment
def cost_XY_airplane := XY * cost_airplane_per_km + airplane_base_cost
def cost_XY_bus := XY * cost_bus_per_km

def cost_YZ_airplane := YZ * cost_airplane_per_km + airplane_base_cost
def cost_YZ_bus := YZ * cost_bus_per_km

def cost_ZX_airplane := XZ * cost_airplane_per_km + airplane_base_cost
def cost_ZX_bus := XZ * cost_bus_per_km

-- Define the minimum costs explicitly based on the choices in the solution
def min_cost := cost_XY_airplane + cost_YZ_airplane + cost_ZX_airplane

theorem minimum_travel_cost : min_cost = 2250 := by 
  sorry

end minimum_travel_cost_l7_7526


namespace length_of_BC_is_eight_l7_7680

theorem length_of_BC_is_eight (a : ℝ) (h_area : (1 / 2) * (2 * a) * a^2 = 64) : 2 * a = 8 := 
by { sorry }

end length_of_BC_is_eight_l7_7680


namespace proof_main_proof_l7_7265

noncomputable def main_proof : Prop :=
  2 * Real.logb 5 10 + Real.logb 5 0.25 = 2

theorem proof_main_proof : main_proof :=
  by
    sorry

end proof_main_proof_l7_7265


namespace gym_needs_868_towels_l7_7308

noncomputable def total_towels_needed : Nat :=
  let hour1 := 40
  let hour2 := hour1 + Nat.floor (0.20 * hour1)
  let hour3 := hour2 + Nat.floor (0.25 * hour2)
  let hour4 := hour3 + Nat.floor ((1 : Float) / 3 * hour3)
  let hour5 := hour4 - Nat.floor (0.15 * hour4)
  let hour6 := hour5
  let hour7 := hour6 - Nat.floor (0.30 * hour6)
  let hour8 := hour7 - Nat.floor (0.50 * hour7)
  2 * (hour1 + hour2 + hour3 + hour4 + hour5 + hour6 + hour7 + hour8)

theorem gym_needs_868_towels : total_towels_needed = 868 :=
by
  calc
    total_towels_needed
    = 2 * (40 + Nat.floor (0.20 * 40) + (hour2 + Nat.floor (0.25 * hour2)) + (hour3 + Nat.floor ((1 : Float) / 3 * hour3)) + (hour4 - Nat.floor (0.15 * hour4)) + hour5 + (hour6 - Nat.floor (0.30 * hour6)) + (hour7 - Nat.floor (0.50 * hour7))) := rfl
    -- detailed calculation steps would follow
  sorry

end gym_needs_868_towels_l7_7308


namespace product_divisible_by_8_probability_l7_7198

noncomputable def probability_product_divisible_by_8 (dice_rolls : Fin 6 → Fin 8) : ℚ :=
  -- Function to calculate the probability that the product of numbers is divisible by 8
  sorry

theorem product_divisible_by_8_probability :
  ∀ (dice_rolls : Fin 6 → Fin 8),
  probability_product_divisible_by_8 dice_rolls = 177 / 256 :=
sorry

end product_divisible_by_8_probability_l7_7198


namespace smallest_possible_N_l7_7937

theorem smallest_possible_N (p q r s t : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0)
    (h_sum : p + q + r + s + t = 2025) : 
    let N := max (max (p + q) (q + r)) (max (r + s) (s + t)) in
    N = 676 :=
by
    sorry

end smallest_possible_N_l7_7937
