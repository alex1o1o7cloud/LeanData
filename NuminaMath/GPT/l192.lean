import Mathlib

namespace min_distance_sum_l192_192879

theorem min_distance_sum (x : ℝ) : 
  ∃ y, y = |x + 1| + 2 * |x - 5| + |2 * x - 7| + |(x - 11) / 2| ∧ y = 45 / 8 :=
sorry

end min_distance_sum_l192_192879


namespace cal_fraction_of_anthony_l192_192584

theorem cal_fraction_of_anthony (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ)
  (h_mabel : mabel_transactions = 90)
  (h_anthony : anthony_transactions = mabel_transactions + mabel_transactions / 10)
  (h_jade : jade_transactions = 82)
  (h_jade_cal : jade_transactions = cal_transactions + 16) :
  (cal_transactions : ℚ) / (anthony_transactions : ℚ) = 2 / 3 :=
by
  -- The proof would be here, but it is omitted as per the requirement.
  sorry

end cal_fraction_of_anthony_l192_192584


namespace problem_correct_l192_192046

noncomputable def S : Set ℕ := {x | x^2 - x = 0}
noncomputable def T : Set ℕ := {x | x ∈ Set.univ ∧ 6 % (x - 2) = 0}

theorem problem_correct : S ∩ T = ∅ :=
by sorry

end problem_correct_l192_192046


namespace non_fiction_vs_fiction_diff_l192_192121

def total_books : Nat := 35
def fiction_books : Nat := 5
def picture_books : Nat := 11
def autobiography_books : Nat := 2 * fiction_books

def accounted_books : Nat := fiction_books + autobiography_books + picture_books
def non_fiction_books : Nat := total_books - accounted_books

theorem non_fiction_vs_fiction_diff :
  non_fiction_books - fiction_books = 4 := by 
  sorry

end non_fiction_vs_fiction_diff_l192_192121


namespace min_value_expression_l192_192676

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem min_value_expression (a b : ℝ) (h1 : b > 0) (h2 : f a b 1 = 3) :
  ∃ x, x = (4 / (a - 1) + 1 / b) ∧ x = 9 / 2 :=
by
  sorry

end min_value_expression_l192_192676


namespace ratio_of_perimeters_l192_192828

theorem ratio_of_perimeters (L : ℝ) (H : ℝ) (hL1 : L = 8) 
  (hH1 : H = 8) (hH2 : H = 2 * (H / 2)) (hH3 : 4 > 0) (hH4 : 0 < 4 / 3)
  (hW1 : ∀ a, a / 3 > 0 → 8 = L )
  (hPsmall : ∀ P, P = 2 * ((4 / 3) + 8) )
  (hPlarge : ∀ P, P = 2 * ((H - 4 / 3) + 8) )
  :
  (2 * ((4 / 3) + 8)) / (2 * ((8 - (4 / 3)) + 8)) = (7 / 11) := by
  sorry

end ratio_of_perimeters_l192_192828


namespace Alex_has_more_than_200_marbles_on_Monday_of_next_week_l192_192853

theorem Alex_has_more_than_200_marbles_on_Monday_of_next_week :
  ∃ k : ℕ, k > 0 ∧ 3 * 2^k > 200 ∧ k % 7 = 1 := by
  sorry

end Alex_has_more_than_200_marbles_on_Monday_of_next_week_l192_192853


namespace unique_perpendicular_line_through_point_l192_192290

variables (a b : ℝ → ℝ) (P : ℝ)

def are_skew_lines (a b : ℝ → ℝ) : Prop :=
  ¬∃ (t₁ t₂ : ℝ), a t₁ = b t₂

def is_point_not_on_lines (P : ℝ) (a b : ℝ → ℝ) : Prop :=
  ∀ (t : ℝ), P ≠ a t ∧ P ≠ b t

theorem unique_perpendicular_line_through_point (ha : are_skew_lines a b) (hp : is_point_not_on_lines P a b) :
  ∃! (L : ℝ → ℝ), (∀ (t : ℝ), L t ≠ P) ∧ (∀ (L' : ℝ → ℝ), (∀ (t : ℝ), L' t ≠ P) → L' = L) := sorry

end unique_perpendicular_line_through_point_l192_192290


namespace tension_limit_l192_192585

theorem tension_limit (M m g : ℝ) (hM : 0 < M) (hg : 0 < g) :
  (∀ T, (T = Mg ↔ m = 0) → (∀ ε, 0 < ε → ∃ m₀, m > m₀ → |T - 2 * M * g| < ε)) :=
by 
  sorry

end tension_limit_l192_192585


namespace solution_set_equivalence_l192_192010

noncomputable def f : ℝ → ℝ := sorry

axiom f_derivative : ∀ x : ℝ, deriv f x > 1 - f x
axiom f_at_0 : f 0 = 3

theorem solution_set_equivalence :
  {x : ℝ | (Real.exp x) * f x > (Real.exp x) + 2} = {x : ℝ | x > 0} :=
by sorry

end solution_set_equivalence_l192_192010


namespace number_of_girls_l192_192271

theorem number_of_girls (total_students : ℕ) (sample_size : ℕ) (girls_sampled_minus : ℕ) (girls_sampled_ratio : ℚ) :
  total_students = 1600 →
  sample_size = 200 →
  girls_sampled_minus = 20 →
  girls_sampled_ratio = 90 / 200 →
  (∃ x, x / (total_students : ℚ) = girls_sampled_ratio ∧ x = 720) :=
by intros _ _ _ _; sorry

end number_of_girls_l192_192271


namespace largest_perimeter_l192_192262

-- Define the problem's conditions
def side1 := 7
def side2 := 9
def integer_side (x : ℕ) : Prop := (x > 2) ∧ (x < 16)

-- Define the perimeter calculation
def perimeter (a b c : ℕ) := a + b + c

-- The theorem statement which we want to prove
theorem largest_perimeter : ∃ x : ℕ, integer_side x ∧ perimeter side1 side2 x = 31 :=
by
  sorry

end largest_perimeter_l192_192262


namespace gold_stickers_for_second_student_l192_192259

theorem gold_stickers_for_second_student :
  (exists f : ℕ → ℕ,
      f 1 = 29 ∧
      f 3 = 41 ∧
      f 4 = 47 ∧
      f 5 = 53 ∧
      f 6 = 59 ∧
      (∀ n, f (n + 1) - f n = 6 ∨ f (n + 2) - f n = 12)) →
  (∃ f : ℕ → ℕ, f 2 = 35) :=
by
  sorry

end gold_stickers_for_second_student_l192_192259


namespace exists_integers_A_B_C_l192_192346

theorem exists_integers_A_B_C (a b : ℚ) (N_star : Set ℕ) (Q : Set ℚ)
  (h : ∀ x ∈ N_star, (a * (x : ℚ) + b) / (x : ℚ) ∈ Q) : 
  ∃ A B C : ℤ, ∀ x ∈ N_star, 
    (a * (x : ℚ) + b) / (x : ℚ) = (A * (x : ℚ) + B) / (C * (x : ℚ)) := 
sorry

end exists_integers_A_B_C_l192_192346


namespace smallest_integer_to_multiply_y_to_make_perfect_square_l192_192694

noncomputable def y : ℕ :=
  3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_multiply_y_to_make_perfect_square :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (k * y) = m^2) ∧ k = 3 := by
  sorry

end smallest_integer_to_multiply_y_to_make_perfect_square_l192_192694


namespace investment_worth_l192_192314

theorem investment_worth {x : ℝ} (x_pos : 0 < x) :
  ∀ (initial_investment final_value : ℝ) (years : ℕ),
  (initial_investment * 3^years = final_value) → 
  initial_investment = 1500 → final_value = 13500 → 
  8 = x → years = 2 →
  years * (112 / x) = 28 := 
by
  sorry

end investment_worth_l192_192314


namespace solve_for_x_l192_192179

theorem solve_for_x (x : ℚ) (h₁ : (7 * x + 2) / (x - 4) = -6 / (x - 4)) (h₂ : x ≠ 4) :
  x = -8 / 7 := 
  sorry

end solve_for_x_l192_192179


namespace no_line_bisected_by_P_exists_l192_192626

theorem no_line_bisected_by_P_exists (P : ℝ × ℝ) (H : ∀ x y : ℝ, (x / 3)^2 - (y / 2)^2 = 1) : 
  P ≠ (2, 1) := 
sorry

end no_line_bisected_by_P_exists_l192_192626


namespace work_done_by_forces_l192_192839

-- Definitions of given forces and displacement
noncomputable def F1 : ℝ × ℝ := (Real.log 2, Real.log 2)
noncomputable def F2 : ℝ × ℝ := (Real.log 5, Real.log 2)
noncomputable def S : ℝ × ℝ := (2 * Real.log 5, 1)

-- Statement of the theorem
theorem work_done_by_forces :
  let F := (F1.1 + F2.1, F1.2 + F2.2)
  let W := F.1 * S.1 + F.2 * S.2
  W = 2 :=
by
  sorry

end work_done_by_forces_l192_192839


namespace second_train_speed_l192_192882

theorem second_train_speed (len1 len2 dist t : ℕ) (h1 : len1 = 100) (h2 : len2 = 150) (h3 : dist = 50) (h4 : t = 60) : 
  (len1 + len2 + dist) / t = 5 := 
  by
  -- Definitions from conditions
  have h_len1 : len1 = 100 := h1
  have h_len2 : len2 = 150 := h2
  have h_dist : dist = 50 := h3
  have h_time : t = 60 := h4
  
  -- Proof deferred
  sorry

end second_train_speed_l192_192882


namespace arithmetic_sequence_fourth_term_l192_192157

theorem arithmetic_sequence_fourth_term (b d : ℝ) (h : 2 * b + 2 * d = 10) : b + d = 5 :=
by
  sorry

end arithmetic_sequence_fourth_term_l192_192157


namespace bicycle_helmet_savings_l192_192111

theorem bicycle_helmet_savings :
  let bicycle_regular_price := 320
  let bicycle_discount := 0.2
  let helmet_regular_price := 80
  let helmet_discount := 0.1
  let bicycle_savings := bicycle_regular_price * bicycle_discount
  let helmet_savings := helmet_regular_price * helmet_discount
  let total_savings := bicycle_savings + helmet_savings
  let total_regular_price := bicycle_regular_price + helmet_regular_price
  let percentage_savings := (total_savings / total_regular_price) * 100
  percentage_savings = 18 := 
by sorry

end bicycle_helmet_savings_l192_192111


namespace petya_coloring_l192_192797

theorem petya_coloring (k : ℕ) : k = 1 :=
  sorry

end petya_coloring_l192_192797


namespace cube_surface_area_example_l192_192009

def cube_surface_area (V : ℝ) (S : ℝ) : Prop :=
  (∃ s : ℝ, s ^ 3 = V ∧ S = 6 * s ^ 2)

theorem cube_surface_area_example : cube_surface_area 8 24 :=
by
  sorry

end cube_surface_area_example_l192_192009


namespace distance_CD_l192_192752

-- Conditions
variable (width_small : ℝ) 
variable (length_small : ℝ := 2 * width_small) 
variable (perimeter_small : ℝ := 2 * (width_small + length_small))
variable (width_large : ℝ := 3 * width_small)
variable (length_large : ℝ := 2 * length_small)
variable (area_large : ℝ := width_large * length_large)

-- Condition assertions
axiom smaller_rectangle_perimeter : perimeter_small = 6
axiom larger_rectangle_area : area_large = 12

-- Calculating distance hypothesis
theorem distance_CD (CD_x CD_y : ℝ) (width_small length_small width_large length_large : ℝ) 
  (smaller_rectangle_perimeter : 2 * (width_small + length_small) = 6)
  (larger_rectangle_area : (3 * width_small) * (2 * length_small) = 12)
  (CD_x_def : CD_x = 2 * length_small)
  (CD_y_def : CD_y = 2 * width_large - width_small)
  : Real.sqrt ((CD_x) ^ 2 + (CD_y) ^ 2) = Real.sqrt 45 := 
sorry

end distance_CD_l192_192752


namespace ratio_neha_mother_age_12_years_ago_l192_192685

variables (N : ℕ) (M : ℕ) (X : ℕ)

theorem ratio_neha_mother_age_12_years_ago 
  (hM : M = 60)
  (h_future : M + 12 = 2 * (N + 12)) :
  (12 : ℕ) * (M - 12) = (48 : ℕ) * (N - 12) :=
by
  sorry

end ratio_neha_mother_age_12_years_ago_l192_192685


namespace conditional_probability_P_B_given_A_l192_192431

-- Let E be an enumeration type with exactly five values, each representing one attraction.
inductive Attraction : Type
| dayu_yashan : Attraction
| qiyunshan : Attraction
| tianlongshan : Attraction
| jiulianshan : Attraction
| sanbaishan : Attraction

open Attraction

-- Define A and B's choices as random variables.
axiom A_choice : Attraction
axiom B_choice : Attraction

-- Event A is that A and B choose different attractions.
def event_A : Prop := A_choice ≠ B_choice

-- Event B is that A and B each choose Chongyi Qiyunshan.
def event_B : Prop := A_choice = qiyunshan ∧ B_choice = qiyunshan

-- Calculate the conditional probability P(B|A)
theorem conditional_probability_P_B_given_A : 
  (1 - (1 / 5)) * (1 - (1 / 5)) = 2 / 5 :=
sorry

end conditional_probability_P_B_given_A_l192_192431


namespace CarrieSpent_l192_192141

variable (CostPerShirt NumberOfShirts : ℝ)

def TotalCost (CostPerShirt NumberOfShirts : ℝ) : ℝ :=
  CostPerShirt * NumberOfShirts

theorem CarrieSpent {CostPerShirt NumberOfShirts : ℝ} 
  (h1 : CostPerShirt = 9.95) 
  (h2 : NumberOfShirts = 20) : 
  TotalCost CostPerShirt NumberOfShirts = 199.00 :=
by
  sorry

end CarrieSpent_l192_192141


namespace tangent_addition_l192_192378

theorem tangent_addition (y : ℝ) (h : Real.tan y = -1) : Real.tan (y + Real.pi / 3) = -1 :=
sorry

end tangent_addition_l192_192378


namespace coloring_count_l192_192996

theorem coloring_count : 
  ∀ (n : ℕ), n = 2021 → 
  ∃ (ways : ℕ), ways = 3 * 2 ^ 2020 :=
by
  intros n hn
  existsi 3 * 2 ^ 2020
  sorry

end coloring_count_l192_192996


namespace g_at_3_l192_192877

def g (x : ℝ) : ℝ := -3 * x^4 + 4 * x^3 - 7 * x^2 + 5 * x - 2

theorem g_at_3 : g 3 = -185 := by
  sorry

end g_at_3_l192_192877


namespace part1_part2_part3_l192_192269

variable {x y z : ℝ}

-- Given condition
variables (hx : x > 0) (hy : y > 0) (hz : z > 0)

theorem part1 : 
  (x / y + y / z + z / x) / 3 ≥ 1 := sorry

theorem part2 :
  x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ (x / y + y / z + z / x)^2 / 3 := sorry

theorem part3 :
  x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ x / y + y / z + z / x := sorry

end part1_part2_part3_l192_192269


namespace max_temperature_when_80_l192_192863

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 10 * t + 60

-- State the theorem
theorem max_temperature_when_80 : ∃ t : ℝ, temperature t = 80 ∧ t = 5 + Real.sqrt 5 := 
by {
  -- Theorem proof is skipped with sorry
  sorry
}

end max_temperature_when_80_l192_192863


namespace determine_g_l192_192008

def real_function (g : ℝ → ℝ) :=
  ∀ c d : ℝ, g (c + d) + g (c - d) = g (c) * g (d) + g (d)

def non_zero_function (g : ℝ → ℝ) :=
  ∃ x : ℝ, g x ≠ 0

theorem determine_g (g : ℝ → ℝ) (h1 : real_function g) (h2 : non_zero_function g) : g 0 = 1 ∧ ∀ x : ℝ, g (-x) = g x := 
sorry

end determine_g_l192_192008


namespace sequence_properties_sum_Tn_l192_192916

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℤ := 2^(n - 1)
noncomputable def c_n (n : ℕ) : ℤ := (2 * n - 1) / 2^(n - 1)
noncomputable def T_n (n : ℕ) : ℤ := 6 - (2 * n + 3) / 2^(n - 1)

theorem sequence_properties : (d = 2) → (S₁₀ = 100) → 
  (∀ n : ℕ, a_n n = 2 * n - 1) ∧ (∀ n : ℕ, b_n n = 2^(n - 1)) := by
  sorry

theorem sum_Tn : (d > 1) → 
  (∀ n : ℕ, T_n n = 6 - (2 * n + 3) / 2^(n - 1)) := by
  sorry

end sequence_properties_sum_Tn_l192_192916


namespace equality_of_fractions_l192_192322

theorem equality_of_fractions
  (a b c x y z : ℝ)
  (h1 : a = b * z + c * y)
  (h2 : b = c * x + a * z)
  (h3 : c = a * y + b * x)
  (hx : x ≠ 1 ∧ x ≠ -1)
  (hy : y ≠ 1 ∧ y ≠ -1)
  (hz : z ≠ 1 ∧ z ≠ -1) :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  (a^2) / (1 - x^2) = (b^2) / (1 - y^2) ∧ (b^2) / (1 - y^2) = (c^2) / (1 - z^2) :=
by
  sorry

end equality_of_fractions_l192_192322


namespace monotonically_decreasing_when_a_half_l192_192143

noncomputable def f (x a : ℝ) : ℝ := x * (Real.log x - a * x)

theorem monotonically_decreasing_when_a_half :
  ∀ x : ℝ, 0 < x → (f x (1 / 2)) ≤ 0 :=
by
  sorry

end monotonically_decreasing_when_a_half_l192_192143


namespace polynomial_coefficient_sum_l192_192297

theorem polynomial_coefficient_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (2 * x - 3) ^ 5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  sorry

end polynomial_coefficient_sum_l192_192297


namespace triangle_side_b_length_l192_192648

noncomputable def length_of_side_b (A B C a b c : ℝ) (h1 : a = 1)
  (h2 : Real.cos A = 4/5) (h3 : Real.cos C = 5/13) : Prop :=
  b = 21 / 13

theorem triangle_side_b_length (A B C a b c : ℝ) (h1 : a = 1)
  (h2 : Real.cos A = 4/5) (h3 : Real.cos C = 5/13) :
  length_of_side_b A B C a b c h1 h2 h3 :=
by
  sorry

end triangle_side_b_length_l192_192648


namespace geometric_sequence_value_sum_l192_192984

variable {a : ℕ → ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) :=
  ∀ n m, a (n + m) * a 0 = a n * a m

theorem geometric_sequence_value_sum {a : ℕ → ℝ}
  (hpos : ∀ n, a n > 0)
  (geom : is_geometric_sequence a)
  (given : a 0 * a 2 + 2 * a 1 * a 3 + a 2 * a 4 = 16) 
  : a 1 + a 3 = 4 :=
sorry

end geometric_sequence_value_sum_l192_192984


namespace homework_points_l192_192815

variable (H Q T : ℕ)

theorem homework_points (h1 : T = 4 * Q)
                        (h2 : Q = H + 5)
                        (h3 : H + Q + T = 265) : 
  H = 40 :=
sorry

end homework_points_l192_192815


namespace distance_ratio_l192_192333

variables (KD DM : ℝ)

theorem distance_ratio : 
  KD = 4 ∧ (KD + DM + DM + KD = 12) → (KD / DM = 2) := 
by
  sorry

end distance_ratio_l192_192333


namespace inequality_problem_l192_192348

theorem inequality_problem
  (a b c : ℝ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( ( (2 * a + b + c) ^ 2 ) / ( 2 * a ^ 2 + (b + c) ^ 2 ) ) +
  ( ( (a + 2 * b + c) ^ 2 ) / ( 2 * b ^ 2 + (c + a) ^ 2 ) ) +
  ( ( (a + b + 2 * c) ^ 2 ) / ( 2 * c ^ 2 + (a + b) ^ 2 ) ) ≤ 8 :=
by
  sorry

end inequality_problem_l192_192348


namespace find_special_number_l192_192373

theorem find_special_number : 
  ∃ n, 
  (n % 12 = 11) ∧ 
  (n % 11 = 10) ∧ 
  (n % 10 = 9) ∧ 
  (n % 9 = 8) ∧ 
  (n % 8 = 7) ∧ 
  (n % 7 = 6) ∧ 
  (n % 6 = 5) ∧ 
  (n % 5 = 4) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (n % 2 = 1) ∧ 
  (n = 27719) :=
  sorry

end find_special_number_l192_192373


namespace factory_produces_correct_number_of_doors_l192_192524

variable (initial_planned_production : ℕ) (metal_shortage_decrease : ℕ) (pandemic_decrease_factor : ℕ)
variable (doors_per_car : ℕ)

theorem factory_produces_correct_number_of_doors
  (h1 : initial_planned_production = 200)
  (h2 : metal_shortage_decrease = 50)
  (h3 : pandemic_decrease_factor = 50)
  (h4 : doors_per_car = 5) :
  (initial_planned_production - metal_shortage_decrease) * (100 - pandemic_decrease_factor) * doors_per_car / 100 = 375 :=
by
  sorry

end factory_produces_correct_number_of_doors_l192_192524


namespace fractional_part_zero_l192_192679

noncomputable def fractional_part (z : ℝ) : ℝ := z - (⌊z⌋ : ℝ)

theorem fractional_part_zero (x : ℝ) :
  fractional_part (1 / 3 * (1 / 3 * (1 / 3 * x - 3) - 3) - 3) = 0 ↔ 
  ∃ k : ℤ, 27 * k + 9 ≤ x ∧ x < 27 * k + 18 :=
by
  sorry

end fractional_part_zero_l192_192679


namespace intersection_sets_l192_192649

theorem intersection_sets :
  let M := {x : ℝ | (x + 3) * (x - 2) < 0 }
  let N := {x : ℝ | 1 ≤ x ∧ x ≤ 3 }
  M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_sets_l192_192649


namespace students_growth_rate_l192_192176

theorem students_growth_rate (x : ℝ) 
  (h_total : 728 = 200 + 200 * (1+x) + 200 * (1+x)^2) : 
  200 + 200 * (1+x) + 200*(1+x)^2 = 728 := 
  by
  sorry

end students_growth_rate_l192_192176


namespace intervals_equinumerous_l192_192014

-- Definitions and statements
theorem intervals_equinumerous (a : ℝ) (h : 0 < a) : 
  ∃ (f : Set.Icc 0 1 → Set.Icc 0 a), Function.Bijective f :=
by
  sorry

end intervals_equinumerous_l192_192014


namespace product_mod_7_l192_192613

theorem product_mod_7 : (2021 * 2022 * 2023 * 2024) % 7 = 0 :=
by
  have h1 : 2021 % 7 = 6 := by sorry
  have h2 : 2022 % 7 = 0 := by sorry
  have h3 : 2023 % 7 = 1 := by sorry
  have h4 : 2024 % 7 = 2 := by sorry
  sorry

end product_mod_7_l192_192613


namespace inequality_proof_l192_192699

theorem inequality_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
  (1 - 2 * x) / Real.sqrt (x * (1 - x)) + 
  (1 - 2 * y) / Real.sqrt (y * (1 - y)) + 
  (1 - 2 * z) / Real.sqrt (z * (1 - z)) ≥ 
  Real.sqrt (x / (1 - x)) + 
  Real.sqrt (y / (1 - y)) + 
  Real.sqrt (z / (1 - z)) :=
by
  sorry

end inequality_proof_l192_192699


namespace inequality_proof_l192_192208

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ( (2 * a + b + c) ^ 2 / (2 * a ^ 2 + (b + c) ^ 2) +
    (2 * b + a + c) ^ 2 / (2 * b ^ 2 + (c + a) ^ 2) +
    (2 * c + a + b) ^ 2 / (2 * c ^ 2 + (a + b) ^ 2)
  ) ≤ 8 := 
by
  sorry

end inequality_proof_l192_192208


namespace find_x_l192_192276

theorem find_x (x : ℝ) (h : (2 * x + 8 + 5 * x + 3 + 3 * x + 9) / 3 = 3 * x + 2) : x = -14 :=
by
  sorry

end find_x_l192_192276


namespace blue_tissue_length_exists_l192_192977

theorem blue_tissue_length_exists (B R : ℝ) (h1 : R = B + 12) (h2 : 2 * R = 3 * B) : B = 24 := 
by
  sorry

end blue_tissue_length_exists_l192_192977


namespace ashley_family_spending_l192_192096

theorem ashley_family_spending:
  let child_ticket := 4.25
  let adult_ticket := child_ticket + 3.50
  let senior_ticket := adult_ticket - 1.75
  let morning_discount := 0.10
  let total_morning_tickets := 2 * adult_ticket + 4 * child_ticket + senior_ticket
  let morning_tickets_after_discount := total_morning_tickets * (1 - morning_discount)
  let buy_2_get_1_free_discount := child_ticket
  let discount_for_5_or_more := 4.00
  let total_tickets_after_vouchers := morning_tickets_after_discount - buy_2_get_1_free_discount - discount_for_5_or_more
  let popcorn := 5.25
  let soda := 3.50
  let candy := 4.00
  let concession_total := 3 * popcorn + 2 * soda + candy
  let concession_discount := concession_total * 0.10
  let concession_after_discount := concession_total - concession_discount
  let final_total := total_tickets_after_vouchers + concession_after_discount
  final_total = 50.47 := by
  sorry

end ashley_family_spending_l192_192096


namespace ticket_cost_difference_l192_192837

theorem ticket_cost_difference (num_prebuy : ℕ) (price_prebuy : ℕ) (num_gate : ℕ) (price_gate : ℕ)
  (h_prebuy : num_prebuy = 20) (h_price_prebuy : price_prebuy = 155)
  (h_gate : num_gate = 30) (h_price_gate : price_gate = 200) :
  num_gate * price_gate - num_prebuy * price_prebuy = 2900 :=
by
  sorry

end ticket_cost_difference_l192_192837


namespace length_cut_XY_l192_192565

theorem length_cut_XY (a x : ℝ) (h1 : 4 * a = 100) (h2 : a + a + 2 * x = 56) : x = 3 :=
by { sorry }

end length_cut_XY_l192_192565


namespace y_intercept_of_line_l192_192804

theorem y_intercept_of_line (m x1 y1 : ℝ) (x_intercept : x1 = 4) (y_intercept_at_x1_zero : y1 = 0) (m_value : m = -3) :
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ∧ x = 0 → y = b) ∧ b = 12 :=
by
  sorry

end y_intercept_of_line_l192_192804


namespace problem1_problem2_l192_192932

-- Definition of a double root equation with the given condition
def is_double_root_equation (a b c : ℝ) := 
  ∃ x1 x2 : ℝ, a * x1 = 2 * a * x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0

-- Proving that x² - 6x + 8 = 0 is a double root equation
theorem problem1 : is_double_root_equation 1 (-6) 8 :=
  sorry

-- Proving that if (x-8)(x-n) = 0 is a double root equation, n is either 4 or 16
theorem problem2 (n : ℝ) (h : is_double_root_equation 1 (-8 - n) (8 * n)) :
  n = 4 ∨ n = 16 :=
  sorry

end problem1_problem2_l192_192932


namespace bicycle_wheels_l192_192062

theorem bicycle_wheels :
  ∃ b : ℕ, 
  (∃ (num_bicycles : ℕ) (num_tricycles : ℕ) (wheels_per_tricycle : ℕ) (total_wheels : ℕ),
    num_bicycles = 16 ∧ 
    num_tricycles = 7 ∧ 
    wheels_per_tricycle = 3 ∧ 
    total_wheels = 53 ∧ 
    16 * b + num_tricycles * wheels_per_tricycle = total_wheels) ∧ 
  b = 2 :=
by
  sorry

end bicycle_wheels_l192_192062


namespace expression_value_l192_192257

theorem expression_value (m n a b x : ℤ) (h1 : m = -n) (h2 : a * b = 1) (h3 : |x| = 3) :
  x = 3 ∨ x = -3 → (x = 3 → x^3 - (1 + m + n - a * b) * x^2010 + (m + n) * x^2007 + (-a * b)^2009 = 26) ∧
                  (x = -3 → x^3 - (1 + m + n - a * b) * x^2010 + (m + n) * x^2007 + (-a * b)^2009 = -28) := by
  sorry

end expression_value_l192_192257


namespace probability_of_selecting_cooking_l192_192928

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l192_192928


namespace proposition_a_is_true_l192_192687

-- Define a quadrilateral
structure Quadrilateral (α : Type*) [Ring α] :=
(a b c d : α)

-- Define properties of a Quadrilateral
def parallel_and_equal_opposite_sides (Q : Quadrilateral ℝ) : Prop := sorry  -- Assumes parallel and equal opposite sides
def is_parallelogram (Q : Quadrilateral ℝ) : Prop := sorry  -- Defines a parallelogram

-- The theorem we need to prove
theorem proposition_a_is_true (Q : Quadrilateral ℝ) (h : parallel_and_equal_opposite_sides Q) : is_parallelogram Q :=
sorry

end proposition_a_is_true_l192_192687


namespace area_of_BCD_l192_192857

variables (a b c x y : ℝ)

-- Conditions
axiom h1 : x = (1 / 2) * a * b
axiom h2 : y = (1 / 2) * b * c

-- Conclusion to prove
theorem area_of_BCD (a b c x y : ℝ) (h1 : x = (1 / 2) * a * b) (h2 : y = (1 / 2) * b * c) : 
  (1 / 2) * b * c = y :=
sorry

end area_of_BCD_l192_192857


namespace volume_of_mixture_removed_replaced_l192_192089

noncomputable def volume_removed (initial_mixture: ℝ) (initial_milk: ℝ) (final_concentration: ℝ): ℝ :=
  (1 - final_concentration / initial_milk) * initial_mixture

theorem volume_of_mixture_removed_replaced (initial_mixture: ℝ) (initial_milk: ℝ) (final_concentration: ℝ) (V: ℝ):
  initial_mixture = 100 →
  initial_milk = 36 →
  final_concentration = 9 →
  V = 50 →
  volume_removed initial_mixture initial_milk final_concentration = V :=
by
  intros h1 h2 h3 h4
  have h5 : initial_mixture = 100 := h1
  have h6 : initial_milk = 36 := h2
  have h7 : final_concentration = 9 := h3
  rw [h5, h6, h7]
  sorry

end volume_of_mixture_removed_replaced_l192_192089


namespace cos_sub_sin_alpha_l192_192997

theorem cos_sub_sin_alpha (alpha : ℝ) (h1 : π / 4 < alpha) (h2 : alpha < π / 2)
    (h3 : Real.sin (2 * alpha) = 24 / 25) : Real.cos alpha - Real.sin alpha = -1 / 5 :=
by
  sorry

end cos_sub_sin_alpha_l192_192997


namespace minimum_value_of_quad_func_l192_192788

def quad_func (x : ℝ) : ℝ :=
  2 * x^2 - 8 * x + 15

theorem minimum_value_of_quad_func :
  (∀ x : ℝ, quad_func 2 ≤ quad_func x) ∧ (quad_func 2 = 7) :=
by
  -- sorry to skip proof
  sorry

end minimum_value_of_quad_func_l192_192788


namespace find_angle_complement_supplement_l192_192188

theorem find_angle_complement_supplement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end find_angle_complement_supplement_l192_192188


namespace total_pencils_l192_192194

-- Define the initial conditions
def initial_pencils : ℕ := 41
def added_pencils : ℕ := 30

-- Define the statement to be proven
theorem total_pencils :
  initial_pencils + added_pencils = 71 :=
by
  sorry

end total_pencils_l192_192194


namespace combined_score_is_75_l192_192954

variable (score1 : ℕ) (total1 : ℕ)
variable (score2 : ℕ) (total2 : ℕ)
variable (score3 : ℕ) (total3 : ℕ)

-- Conditions: Antonette's scores and the number of problems in each test
def Antonette_scores : Prop :=
  score1 = 60 * total1 / 100 ∧ total1 = 15 ∧
  score2 = 85 * total2 / 100 ∧ total2 = 20 ∧
  score3 = 75 * total3 / 100 ∧ total3 = 25

-- Theorem to prove the combined score is 75% (45 out of 60) rounded to the nearest percent
theorem combined_score_is_75
  (h : Antonette_scores score1 total1 score2 total2 score3 total3) :
  100 * (score1 + score2 + score3) / (total1 + total2 + total3) = 75 :=
by sorry

end combined_score_is_75_l192_192954


namespace hyperbola_parabola_intersection_l192_192100

open Real

theorem hyperbola_parabola_intersection :
  let A := (4, 4)
  let B := (4, -4)
  |dist A B| = 8 :=
by
  let hyperbola_asymptote (x y: ℝ) := x^2 - y^2 = 1
  let parabola_equation (x y : ℝ) := y^2 = 4 * x
  sorry

end hyperbola_parabola_intersection_l192_192100


namespace area_ratio_of_isosceles_triangle_l192_192479

variable (x : ℝ)
variable (hx : 0 < x)

def isosceles_triangle (AB AC : ℝ) (BC : ℝ) : Prop :=
  AB = AC ∧ AB = 2 * x ∧ BC = x

def extend_side (B_length AB_length : ℝ) : Prop :=
  B_length = 2 * AB_length

def ratio_of_areas (area_AB'B'C' area_ABC : ℝ) : Prop :=
  area_AB'B'C' / area_ABC = 9

theorem area_ratio_of_isosceles_triangle
  (AB AC BC : ℝ) (BB' B'C' area_ABC area_AB'B'C' : ℝ)
  (h_isosceles : isosceles_triangle x AB AC BC)
  (h_extend_A : extend_side BB' AB)
  (h_extend_C : extend_side B'C' AC) :
  ratio_of_areas area_AB'B'C' area_ABC := by
  sorry

end area_ratio_of_isosceles_triangle_l192_192479


namespace log_inequality_l192_192980

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) :
  log ((a + b) / 2) + log ((b + c) / 2) + log ((c + a) / 2) > log a + log b + log c :=
by
  sorry

end log_inequality_l192_192980


namespace area_to_paint_l192_192376

def wall_height : ℕ := 10
def wall_length : ℕ := 15
def bookshelf_height : ℕ := 3
def bookshelf_length : ℕ := 5

theorem area_to_paint : (wall_height * wall_length) - (bookshelf_height * bookshelf_length) = 135 :=
by 
  sorry

end area_to_paint_l192_192376


namespace trig_expression_equality_l192_192631

theorem trig_expression_equality (α : ℝ) (h : Real.tan α = 1 / 2) : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -4 :=
by
  sorry

end trig_expression_equality_l192_192631


namespace Heather_delay_l192_192929

noncomputable def find_start_time : ℝ :=
  let d := 15 -- Initial distance between Stacy and Heather in miles
  let H := 5 -- Heather's speed in miles/hour
  let S := H + 1 -- Stacy's speed in miles/hour
  let d_H := 5.7272727272727275 -- Distance Heather walked when they meet
  let t_H := d_H / H -- Time Heather walked till they meet in hours
  let d_S := S * t_H -- Distance Stacy walked till they meet in miles
  let total_distance := d_H + d_S -- Total distance covered when they meet in miles
  let remaining_distance := d - total_distance -- Remaining distance Stacy covers alone before Heather starts in miles
  let t_S := remaining_distance / S -- Time Stacy walked alone in hours
  let minutes := t_S * 60 -- Convert time Stacy walked alone to minutes
  minutes -- Result in minutes

theorem Heather_delay : find_start_time = 24 := by
  sorry -- Proof of the theorem

end Heather_delay_l192_192929


namespace max_value_of_f_l192_192981

theorem max_value_of_f :
  ∀ (x : ℝ), -5 ≤ x ∧ x ≤ 13 → ∃ (y : ℝ), y = x - 5 ∧ y ≤ 8 ∧ y >= -10 ∧ 
  (∀ (z : ℝ), z = (x - 5) → z ≤ 8) := 
by
  sorry

end max_value_of_f_l192_192981


namespace hour_hand_degrees_per_hour_l192_192224

-- Definitions based on the conditions
def number_of_rotations_in_6_days : ℕ := 12
def degrees_per_rotation : ℕ := 360
def hours_in_6_days : ℕ := 6 * 24

-- Statement to prove
theorem hour_hand_degrees_per_hour :
  (number_of_rotations_in_6_days * degrees_per_rotation) / hours_in_6_days = 30 :=
by sorry

end hour_hand_degrees_per_hour_l192_192224


namespace total_miles_run_correct_l192_192782

-- Define the number of people on the sprint team and the miles each person runs.
def number_of_people : Float := 150.0
def miles_per_person : Float := 5.0

-- Define the total miles run by the sprint team.
def total_miles_run : Float := number_of_people * miles_per_person

-- State the theorem to prove that the total miles run is equal to 750.0 miles.
theorem total_miles_run_correct : total_miles_run = 750.0 := sorry

end total_miles_run_correct_l192_192782


namespace intensity_of_replacement_paint_l192_192869

theorem intensity_of_replacement_paint (f : ℚ) (I_new : ℚ) (I_orig : ℚ) (I_repl : ℚ) :
  f = 2/3 → I_new = 40 → I_orig = 60 → I_repl = (40 - 1/3 * 60) * (3/2) := by
  sorry

end intensity_of_replacement_paint_l192_192869


namespace plane_equation_l192_192573

theorem plane_equation
  (A B C D : ℤ)
  (hA : A > 0)
  (h_gcd : Int.gcd A B = 1 ∧ Int.gcd A C = 1 ∧ Int.gcd A D = 1)
  (h_point : (A * 4 + B * (-4) + C * 5 + D = 0)) :
  A = 4 ∧ B = -4 ∧ C = 5 ∧ D = -57 :=
  sorry

end plane_equation_l192_192573


namespace chosen_number_l192_192918

theorem chosen_number (x : ℤ) (h : x / 12 - 240 = 8) : x = 2976 :=
by sorry

end chosen_number_l192_192918


namespace number_of_divisions_l192_192298

-- Definitions
def hour_in_seconds : ℕ := 3600

def is_division (n m : ℕ) : Prop :=
  n * m = hour_in_seconds ∧ n > 0 ∧ m > 0

-- Proof problem statement
theorem number_of_divisions : ∃ (count : ℕ), count = 44 ∧ 
  (∀ (n m : ℕ), is_division n m → ∃ (d : ℕ), d = count) :=
sorry

end number_of_divisions_l192_192298


namespace principal_sum_l192_192273

theorem principal_sum (R P : ℝ) (h : (P * (R + 3) * 3) / 100 = (P * R * 3) / 100 + 81) : P = 900 :=
by
  sorry

end principal_sum_l192_192273


namespace find_m_l192_192019

theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 2 ↔ m * (x - 1) > x^2 - x) : m = 2 :=
sorry

end find_m_l192_192019


namespace non_congruent_squares_on_5x5_grid_l192_192620

def is_lattice_point (x y : ℕ) : Prop := x ≤ 4 ∧ y ≤ 4

def is_square {a b c d : (ℕ × ℕ)} : Prop :=
((a.1 - b.1)^2 + (a.2 - b.2)^2 = (c.1 - d.1)^2 + (c.2 - d.2)^2) ∧ 
((c.1 - b.1)^2 + (c.2 - b.2)^2 = (a.1 - d.1)^2 + (a.2 - d.2)^2)

def number_of_non_congruent_squares : ℕ :=
  4 + -- Standard squares: 1x1, 2x2, 3x3, 4x4
  2 + -- Diagonal squares: with sides √2 and 2√2
  2   -- Diagonal sides of 1x2 and 1x3 rectangles

theorem non_congruent_squares_on_5x5_grid :
  number_of_non_congruent_squares = 8 :=
by
  -- proof goes here
  sorry

end non_congruent_squares_on_5x5_grid_l192_192620


namespace eduardo_needs_l192_192494

variable (flour_per_24_cookies sugar_per_24_cookies : ℝ)
variable (num_cookies : ℝ)

axiom h_flour : flour_per_24_cookies = 1.5
axiom h_sugar : sugar_per_24_cookies = 0.5
axiom h_cookies : num_cookies = 120

theorem eduardo_needs (scaling_factor : ℝ) 
    (flour_needed : ℝ)
    (sugar_needed : ℝ)
    (h_scaling : scaling_factor = num_cookies / 24)
    (h_flour_needed : flour_needed = flour_per_24_cookies * scaling_factor)
    (h_sugar_needed : sugar_needed = sugar_per_24_cookies * scaling_factor) :
  flour_needed = 7.5 ∧ sugar_needed = 2.5 :=
sorry

end eduardo_needs_l192_192494


namespace algebraic_expression_value_l192_192382

theorem algebraic_expression_value (a : ℝ) (h : (a^2 - 3) * (a^2 + 1) = 0) : a^2 = 3 :=
by
  sorry

end algebraic_expression_value_l192_192382


namespace conversion_200_meters_to_kilometers_l192_192744

noncomputable def meters_to_kilometers (meters : ℕ) : ℝ :=
  meters / 1000

theorem conversion_200_meters_to_kilometers :
  meters_to_kilometers 200 = 0.2 :=
by
  sorry

end conversion_200_meters_to_kilometers_l192_192744


namespace no_common_period_l192_192956

theorem no_common_period (g h : ℝ → ℝ) 
  (hg : ∀ x, g (x + 2) = g x) 
  (hh : ∀ x, h (x + π/2) = h x) : 
  ¬ (∃ T > 0, ∀ x, g (x + T) + h (x + T) = g x + h x) :=
sorry

end no_common_period_l192_192956


namespace ripe_mangoes_remaining_l192_192773

theorem ripe_mangoes_remaining
  (initial_mangoes : ℕ)
  (ripe_fraction : ℚ)
  (consume_fraction : ℚ)
  (initial_total : initial_mangoes = 400)
  (ripe_ratio : ripe_fraction = 3 / 5)
  (consume_ratio : consume_fraction = 60 / 100) :
  (initial_mangoes * ripe_fraction - initial_mangoes * ripe_fraction * consume_fraction) = 96 :=
by
  sorry

end ripe_mangoes_remaining_l192_192773


namespace journey_total_distance_l192_192343

theorem journey_total_distance :
  let speed1 := 40 -- in kmph
  let time1 := 3 -- in hours
  let speed2 := 60 -- in kmph
  let totalTime := 5 -- in hours
  let distance1 := speed1 * time1
  let time2 := totalTime - time1
  let distance2 := speed2 * time2
  let totalDistance := distance1 + distance2
  totalDistance = 240 := 
by
  sorry

end journey_total_distance_l192_192343


namespace parity_of_f_find_a_l192_192011

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x + a * Real.exp (-x)

theorem parity_of_f (a : ℝ) :
  (∀ x : ℝ, f (-x) a = f x a ↔ a = 1 ∨ a = -1) ∧
  (∀ x : ℝ, f (-x) a = -f x a ↔ a = -1) ∧
  (∀ x : ℝ, ¬(f (-x) a = f x a) ∧ ¬(f (-x) a = -f x a) ↔ ¬(a = 1 ∨ a = -1)) :=
by
  sorry

theorem find_a (h : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x a ≥ f 0 a) : 
  a = 1 :=
by
  sorry

end parity_of_f_find_a_l192_192011


namespace at_least_100_valid_pairs_l192_192347

-- Define the conditions
def boots_distribution (L41 L42 L43 R41 R42 R43 : ℕ) : Prop :=
  L41 + L42 + L43 = 300 ∧ R41 + R42 + R43 = 300 ∧
  (L41 = 200 ∨ L42 = 200 ∨ L43 = 200) ∧
  (R41 = 200 ∨ R42 = 200 ∨ R43 = 200)

-- Define the theorem to be proven
theorem at_least_100_valid_pairs (L41 L42 L43 R41 R42 R43 : ℕ) :
  boots_distribution L41 L42 L43 R41 R42 R43 → 
  (L41 ≥ 100 ∧ R41 ≥ 100 ∨ L42 ≥ 100 ∧ R42 ≥ 100 ∨ L43 ≥ 100 ∧ R43 ≥ 100) → 100 ≤ min L41 R41 ∨ 100 ≤ min L42 R42 ∨ 100 ≤ min L43 R43 :=
  sorry

end at_least_100_valid_pairs_l192_192347


namespace valid_pairs_l192_192720

def valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def valid_number (n : ℕ) : Prop :=
  let digits := [5, 3, 2, 9, n / 10 % 10, n % 10]
  (n % 2 = 0) ∧ (digits.sum % 3 = 0)

theorem valid_pairs (d₀ d₁ : ℕ) :
  valid_digit d₀ →
  valid_digit d₁ →
  (d₀ % 2 = 0) →
  valid_number (53290 * 10 + d₀ * 10 + d₁) →
  (d₀, d₁) = (0, 3) ∨ (d₀, d₁) = (2, 0) ∨ (d₀, d₁) = (2, 3) ∨ (d₀, d₁) = (2, 6) ∨
  (d₀, d₁) = (2, 9) ∨ (d₀, d₁) = (4, 1) ∨ (d₀, d₁) = (4, 4) ∨ (d₀, d₁) = (4, 7) ∨
  (d₀, d₁) = (6, 2) ∨ (d₀, d₁) = (6, 5) ∨ (d₀, d₁) = (6, 8) ∨ (d₀, d₁) = (8, 0) :=
by sorry

end valid_pairs_l192_192720


namespace quadratic_no_real_roots_implies_inequality_l192_192465

theorem quadratic_no_real_roots_implies_inequality (a b c : ℝ) :
  let A := b + c
  let B := a + c
  let C := a + b
  (B^2 - 4 * A * C < 0) → 4 * a * c - b^2 ≤ 3 * a * (a + b + c) :=
by
  intro h
  sorry

end quadratic_no_real_roots_implies_inequality_l192_192465


namespace largest_six_consecutive_nonprime_under_50_l192_192424

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → (m = 1 ∨ m = n)

def consecutiveNonPrimes (m : ℕ) : Prop :=
  ∀ i : ℕ, i < 6 → ¬ isPrime (m + i)

theorem largest_six_consecutive_nonprime_under_50 (n : ℕ) :
  (n < 50 ∧ consecutiveNonPrimes n) →
  n + 5 = 35 :=
by
  intro h
  sorry

end largest_six_consecutive_nonprime_under_50_l192_192424


namespace congruence_example_l192_192646

theorem congruence_example (x : ℤ) (h : 5 * x + 3 ≡ 1 [ZMOD 18]) : 3 * x + 8 ≡ 14 [ZMOD 18] :=
sorry

end congruence_example_l192_192646


namespace solution_set_l192_192594

open Real

noncomputable def f : ℝ → ℝ := sorry -- The function f is abstractly defined
axiom f_point : f 1 = 0 -- f passes through (1, 0)
axiom f_deriv_pos : ∀ (x : ℝ), x > 0 → x * (deriv f x) > 1 -- xf'(x) > 1 for x > 0

theorem solution_set (x : ℝ) : f x ≤ log x ↔ 0 < x ∧ x ≤ 1 :=
by
  sorry

end solution_set_l192_192594


namespace minimum_days_to_owe_double_l192_192569

/-- Kim borrows $100$ dollars from Sam with a simple interest rate of $10\%$ per day.
    There's a one-time borrowing fee of $10$ dollars that is added to the debt immediately.
    We need to prove that the least integer number of days after which Kim will owe 
    Sam at least twice as much as she borrowed is 9 days.
-/
theorem minimum_days_to_owe_double :
  ∀ (x : ℕ), 100 + 10 + 10 * x ≥ 200 → x ≥ 9 :=
by
  intros x h
  sorry

end minimum_days_to_owe_double_l192_192569


namespace complementary_event_equivalence_l192_192169

-- Define the event E: hitting the target at least once in two shots.
-- Event E complementary: missing the target both times.

def eventE := "hitting the target at least once"
def complementaryEvent := "missing the target both times"

theorem complementary_event_equivalence :
  (complementaryEvent = "missing the target both times") ↔ (eventE = "hitting the target at least once") :=
by
  sorry

end complementary_event_equivalence_l192_192169


namespace tiles_needed_l192_192389

def ft_to_inch (x : ℕ) : ℕ := x * 12

def height_ft : ℕ := 10
def length_ft : ℕ := 15
def tile_size_sq_inch : ℕ := 1

def height_inch : ℕ := ft_to_inch height_ft
def length_inch : ℕ := ft_to_inch length_ft
def area_sq_inch : ℕ := height_inch * length_inch

theorem tiles_needed : 
  height_ft = 10 ∧ length_ft = 15 ∧ tile_size_sq_inch = 1 →
  area_sq_inch = 21600 :=
by
  intro h
  exact sorry

end tiles_needed_l192_192389


namespace exists_n_with_common_divisor_l192_192844

theorem exists_n_with_common_divisor :
  ∃ (n : ℕ), ∀ (k : ℕ), (k ≤ 20) → Nat.gcd (n + k) 30030 > 1 :=
by
  sorry

end exists_n_with_common_divisor_l192_192844


namespace polygon_perpendiculars_length_l192_192381

noncomputable def RegularPolygon := { n : ℕ // n ≥ 3 }

structure Perpendiculars (P : RegularPolygon) (i : ℕ) :=
  (d_i     : ℝ)
  (d_i_minus_1 : ℝ)
  (d_i_plus_1 : ℝ)
  (line_crosses_interior : Bool)

theorem polygon_perpendiculars_length {P : RegularPolygon} {i : ℕ}
  (hyp : Perpendiculars P i) :
  hyp.d_i = if hyp.line_crosses_interior 
            then hyp.d_i_minus_1 + hyp.d_i_plus_1 
            else abs (hyp.d_i_minus_1 - hyp.d_i_plus_1) :=
sorry

end polygon_perpendiculars_length_l192_192381


namespace unit_price_quantity_inverse_proportion_map_distance_actual_distance_direct_proportion_l192_192556

-- Definitions based on conditions
variable (unit_price quantity total_price : ℕ)
variable (map_distance actual_distance scale : ℕ)

-- Given conditions
def total_price_fixed := unit_price * quantity = total_price
def scale_fixed := map_distance * scale = actual_distance

-- Proof problem statements
theorem unit_price_quantity_inverse_proportion (h : total_price_fixed unit_price quantity total_price) :
  ∃ k : ℕ, unit_price = k / quantity := sorry

theorem map_distance_actual_distance_direct_proportion (h : scale_fixed map_distance actual_distance scale) :
  ∃ k : ℕ, map_distance * scale = k * actual_distance := sorry

end unit_price_quantity_inverse_proportion_map_distance_actual_distance_direct_proportion_l192_192556


namespace difference_of_squares_l192_192076

variable (x y : ℚ)

theorem difference_of_squares (h1 : x + y = 3 / 8) (h2 : x - y = 1 / 8) : x^2 - y^2 = 3 / 64 := 
by
  sorry

end difference_of_squares_l192_192076


namespace sophomores_more_than_first_graders_l192_192998

def total_students : ℕ := 95
def first_graders : ℕ := 32
def second_graders : ℕ := total_students - first_graders

theorem sophomores_more_than_first_graders : second_graders - first_graders = 31 := by
  sorry

end sophomores_more_than_first_graders_l192_192998


namespace find_s_l192_192526

theorem find_s (n : ℤ) (hn : n ≠ 0) (s : ℝ)
  (hs : s = (20 / (2^(2*n+4) + 2^(2*n+2)))^(1 / n)) :
  s = 1 / 4 :=
by
  sorry

end find_s_l192_192526


namespace max_abs_sum_sqrt2_l192_192636

theorem max_abs_sum_sqrt2 (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (a : ℝ), (a = |x| + |y| ∧ a ≤ 2 * Real.sqrt 2) ∧ 
             ∀ (x y : ℝ), x^2 + y^2 = 4 → (|x| + |y|) ≤ 2 * Real.sqrt 2 :=
sorry

end max_abs_sum_sqrt2_l192_192636


namespace physics_class_size_l192_192355

theorem physics_class_size (total_students physics_only math_only both : ℕ) 
  (h1 : total_students = 53)
  (h2 : both = 7)
  (h3 : physics_only = 2 * (math_only + both))
  (h4 : total_students = physics_only + math_only + both) :
  physics_only + both = 40 :=
by
  sorry

end physics_class_size_l192_192355


namespace number_of_boys_took_exam_l192_192770

theorem number_of_boys_took_exam (T F : ℕ) (h_avg_all : 35 * T = 39 * 100 + 15 * F)
                                (h_total_boys : T = 100 + F) : T = 120 :=
sorry

end number_of_boys_took_exam_l192_192770


namespace value_of_expression_l192_192199

theorem value_of_expression (m : ℝ) (h : 2 * m ^ 2 - 3 * m - 1 = 0) : 4 * m ^ 2 - 6 * m = 2 :=
sorry

end value_of_expression_l192_192199


namespace problem_1_problem_2_l192_192912

open Real

noncomputable def f (omega : ℝ) (x : ℝ) : ℝ := 
  (cos (omega * x) * cos (omega * x) + sqrt 3 * cos (omega * x) * sin (omega * x) - 1/2)

theorem problem_1 (ω : ℝ) (hω : ω > 0):
 (f ω x = sin (2 * x + π / 6)) ∧ 
 (∀ k : ℤ, ∀ x : ℝ, (-π / 3 + ↑k * π) ≤ x ∧ x ≤ (π / 6 + ↑k * π) → f ω x = sin (2 * x + π / 6)) :=
sorry

theorem problem_2 (A b S a : ℝ) (hA : A / 2 = π / 3)
  (hb : b = 1) (hS: S = sqrt 3) :
  a = sqrt 13 :=
sorry

end problem_1_problem_2_l192_192912


namespace recurring_decimal_sum_is_13_over_33_l192_192910

noncomputable def recurring_decimal_sum : ℚ :=
  let x := 1/3 -- 0.\overline{3}
  let y := 2/33 -- 0.\overline{06}
  x + y

theorem recurring_decimal_sum_is_13_over_33 : recurring_decimal_sum = 13/33 := by
  sorry

end recurring_decimal_sum_is_13_over_33_l192_192910


namespace positive_difference_of_y_l192_192758

theorem positive_difference_of_y (y : ℝ) (h : (50 + y) / 2 = 35) : |50 - y| = 30 :=
by
  sorry

end positive_difference_of_y_l192_192758


namespace statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_l192_192536

theorem statement_A_correct :
  (∃ x0 : ℝ, x0^2 + 2 * x0 + 2 < 0) ↔ (¬ ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0) :=
sorry

theorem statement_B_incorrect :
  ¬ (∀ x y : ℝ, x > y → |x| > |y|) :=
sorry

theorem statement_C_incorrect :
  ¬ ∀ x : ℤ, x^2 > 0 :=
sorry

theorem statement_D_correct :
  (∀ m : ℝ, (∃ x1 x2 : ℝ, x1 + x2 = 2 ∧ x1 * x2 = m ∧ x1 * x2 > 0) ↔ m < 0) :=
sorry

end statement_A_correct_statement_B_incorrect_statement_C_incorrect_statement_D_correct_l192_192536


namespace log_sum_eval_l192_192812

theorem log_sum_eval :
  (Real.logb 5 625 + Real.logb 5 5 - Real.logb 5 (1 / 25)) = 7 :=
by
  have h1 : Real.logb 5 625 = 4 := by sorry
  have h2 : Real.logb 5 5 = 1 := by sorry
  have h3 : Real.logb 5 (1 / 25) = -2 := by sorry
  rw [h1, h2, h3]
  norm_num

end log_sum_eval_l192_192812


namespace doubled_container_volume_l192_192436

theorem doubled_container_volume (v : ℝ) (h₁ : v = 4) (h₂ : ∀ l w h : ℝ, v = l * w * h) : 8 * v = 32 := 
by
  -- The proof will go here, this is just the statement
  sorry

end doubled_container_volume_l192_192436


namespace parts_per_hour_equality_l192_192575

variable {x : ℝ}

theorem parts_per_hour_equality (h1 : x - 4 > 0) :
  (100 / x) = (80 / (x - 4)) :=
sorry

end parts_per_hour_equality_l192_192575


namespace solution_set_inequality_l192_192128

theorem solution_set_inequality (x : ℝ) : 
  (∃ x, (x-1)/((x^2) - x - 30) > 0) ↔ (x > -5 ∧ x < 1) ∨ (x > 6) :=
by
  sorry

end solution_set_inequality_l192_192128


namespace david_biology_marks_l192_192451

theorem david_biology_marks
  (english math physics chemistry avg_marks num_subjects : ℕ)
  (h_english : english = 86)
  (h_math : math = 85)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 87)
  (h_avg_marks : avg_marks = 85)
  (h_num_subjects : num_subjects = 5) :
  ∃ (biology : ℕ), biology = 85 :=
by
  -- Total marks for all subjects
  let total_marks_for_all_subjects := avg_marks * num_subjects
  -- Total marks in English, Mathematics, Physics, and Chemistry
  let total_marks_in_other_subjects := english + math + physics + chemistry
  -- Marks in Biology
  let biology := total_marks_for_all_subjects - total_marks_in_other_subjects
  existsi biology
  sorry

end david_biology_marks_l192_192451


namespace sum_gcd_lcm_of_4_and_10_l192_192051

theorem sum_gcd_lcm_of_4_and_10 : Nat.gcd 4 10 + Nat.lcm 4 10 = 22 :=
by
  sorry

end sum_gcd_lcm_of_4_and_10_l192_192051


namespace bus_seat_problem_l192_192985

theorem bus_seat_problem 
  (left_seats : ℕ) 
  (right_seats := left_seats - 3) 
  (left_capacity := 3 * left_seats)
  (right_capacity := 3 * right_seats)
  (back_seat_capacity := 12)
  (total_capacity := left_capacity + right_capacity + back_seat_capacity)
  (h1 : total_capacity = 93) 
  : left_seats = 15 := 
by 
  sorry

end bus_seat_problem_l192_192985


namespace relationship_f_l192_192335

-- Define the function f which is defined on the reals and even
variable (f : ℝ → ℝ)
-- Condition: f is an even function
axiom even_f : ∀ x, f (-x) = f x
-- Condition: (x₁ - x₂)[f(x₁) - f(x₂)] > 0 for all x₁, x₂ ∈ [0, +∞)
axiom increasing_cond : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem relationship_f : f (1/2) < f 1 ∧ f 1 < f (-2) := by
  sorry

end relationship_f_l192_192335


namespace cat_food_more_than_dog_food_l192_192733

-- Define the number of packages and cans per package for cat food
def cat_food_packages : ℕ := 9
def cat_food_cans_per_package : ℕ := 10

-- Define the number of packages and cans per package for dog food
def dog_food_packages : ℕ := 7
def dog_food_cans_per_package : ℕ := 5

-- Total number of cans of cat food
def total_cat_food_cans : ℕ := cat_food_packages * cat_food_cans_per_package

-- Total number of cans of dog food
def total_dog_food_cans : ℕ := dog_food_packages * dog_food_cans_per_package

-- Prove the difference between the total cans of cat food and total cans of dog food
theorem cat_food_more_than_dog_food : total_cat_food_cans - total_dog_food_cans = 55 := by
  -- Provide the calculation results directly
  have h_cat : total_cat_food_cans = 90 := by rfl
  have h_dog : total_dog_food_cans = 35 := by rfl
  calc
    total_cat_food_cans - total_dog_food_cans = 90 - 35 := by rw [h_cat, h_dog]
    _ = 55 := rfl

end cat_food_more_than_dog_food_l192_192733


namespace angle_sum_triangle_l192_192318

theorem angle_sum_triangle (A B C : ℝ) 
  (hA : A = 20)
  (hC : C = 90) :
  B = 70 := 
by
  -- In a triangle the sum of angles is 180 degrees
  have h_sum : A + B + C = 180 := sorry
  -- Substitute the given angles A and C
  rw [hA, hC] at h_sum
  -- Simplify the equation to find B
  have hB : 20 + B + 90 = 180 := sorry
  linarith

end angle_sum_triangle_l192_192318


namespace correct_conclusions_l192_192765

noncomputable def f1 (x : ℝ) : ℝ := 2^x - 1
noncomputable def f2 (x : ℝ) : ℝ := x^3
noncomputable def f3 (x : ℝ) : ℝ := x
noncomputable def f4 (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2

theorem correct_conclusions :
  ((∀ x, 0 < x ∧ x < 1 → f4 x > f1 x ∧ f4 x > f2 x ∧ f4 x > f3 x) ∧
  (∀ x, x > 1 → f4 x < f1 x ∧ f4 x < f2 x ∧ f4 x < f3 x)) ∧
  (∀ x, ¬(f3 x > f1 x ∧ f3 x > f2 x ∧ f3 x > f4 x) ∧
        ¬(f3 x < f1 x ∧ f3 x < f2 x ∧ f3 x < f4 x)) ∧
  (∃ x, x > 0 ∧ ∀ y, y > x → f1 y > f2 y ∧ f1 y > f3 y ∧ f1 y > f4 y) := by
  sorry

end correct_conclusions_l192_192765


namespace father_ate_8_brownies_l192_192255

noncomputable def brownies_initial := 24
noncomputable def brownies_mooney_ate := 4
noncomputable def brownies_after_mooney := brownies_initial - brownies_mooney_ate
noncomputable def brownies_mother_made_next_day := 24
noncomputable def brownies_total_expected := brownies_after_mooney + brownies_mother_made_next_day
noncomputable def brownies_actual_on_counter := 36

theorem father_ate_8_brownies :
  brownies_total_expected - brownies_actual_on_counter = 8 :=
by
  sorry

end father_ate_8_brownies_l192_192255


namespace clock_correct_after_240_days_l192_192688

theorem clock_correct_after_240_days (days : ℕ) (minutes_fast_per_day : ℕ) (hours_to_be_correct : ℕ) 
  (h1 : minutes_fast_per_day = 3) (h2 : hours_to_be_correct = 12) : 
  (days * minutes_fast_per_day) % (hours_to_be_correct * 60) = 0 :=
by 
  -- Proof skipped
  sorry

end clock_correct_after_240_days_l192_192688


namespace mean_equality_l192_192147

theorem mean_equality (z : ℚ) :
  (8 + 12 + 24) / 3 = (16 + z) / 2 ↔ z = 40 / 3 :=
by
  sorry

end mean_equality_l192_192147


namespace distinct_ratios_zero_l192_192038

theorem distinct_ratios_zero (p q r : ℝ) (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p) 
  (h : p / (q - r) + q / (r - p) + r / (p - q) = 3) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 0 :=
sorry

end distinct_ratios_zero_l192_192038


namespace lizzie_garbage_l192_192357

/-- Let G be the amount of garbage Lizzie's group collected. 
We are given that the second group collected G - 39 pounds of garbage,
and the total amount collected by both groups is 735 pounds.
We need to prove that G is 387 pounds. -/
theorem lizzie_garbage (G : ℕ) (h1 : G + (G - 39) = 735) : G = 387 :=
sorry

end lizzie_garbage_l192_192357


namespace rate_is_five_l192_192989

noncomputable def rate_per_sq_meter (total_cost : ℕ) (total_area : ℕ) : ℕ :=
  total_cost / total_area

theorem rate_is_five :
  let length := 80
  let breadth := 60
  let road_width := 10
  let total_cost := 6500
  let area_road1 := road_width * breadth
  let area_road2 := road_width * length
  let area_intersection := road_width * road_width
  let total_area := area_road1 + area_road2 - area_intersection
  rate_per_sq_meter total_cost total_area = 5 :=
by
  sorry

end rate_is_five_l192_192989


namespace molecular_weight_N2O3_correct_l192_192820

/-- Conditions -/
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

/-- Proof statement -/
theorem molecular_weight_N2O3_correct :
  (2 * atomic_weight_N + 3 * atomic_weight_O) = 76.02 ∧
  name_of_N2O3 = "dinitrogen trioxide" := sorry

/-- Definition of the compound name based on formula -/
def name_of_N2O3 : String := "dinitrogen trioxide"

end molecular_weight_N2O3_correct_l192_192820


namespace sum_of_squares_l192_192941

theorem sum_of_squares (x y : ℝ) (h1 : y + 6 = (x - 3)^2) (h2 : x + 6 = (y - 3)^2) (hxy : x ≠ y) : x^2 + y^2 = 43 :=
sorry

end sum_of_squares_l192_192941


namespace evaluate_nav_expression_l192_192808
noncomputable def nav (k m : ℕ) := k * (k - m)

theorem evaluate_nav_expression : (nav 5 1) + (nav 4 1) = 32 :=
by
  -- Skipping the proof as instructed
  sorry

end evaluate_nav_expression_l192_192808


namespace height_of_box_l192_192301

-- Definitions of given conditions
def length_box : ℕ := 9
def width_box : ℕ := 12
def num_cubes : ℕ := 108
def volume_cube : ℕ := 3
def volume_box : ℕ := num_cubes * volume_cube  -- Volume calculated from number of cubes and volume of each cube

-- The statement to prove
theorem height_of_box : 
  ∃ h : ℕ, volume_box = length_box * width_box * h ∧ h = 3 := by
  sorry

end height_of_box_l192_192301


namespace sufficient_but_not_necessary_l192_192277

theorem sufficient_but_not_necessary (x y : ℝ) (h : x ≥ 1 ∧ y ≥ 1) : x ^ 2 + y ^ 2 ≥ 2 ∧ ∃ (x y : ℝ), x ^ 2 + y ^ 2 ≥ 2 ∧ (¬ (x ≥ 1 ∧ y ≥ 1)) :=
by
  sorry

end sufficient_but_not_necessary_l192_192277


namespace find_a7_a8_a9_l192_192098

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → α) (n : ℕ) : α :=
  n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

variables {a : ℕ → α}
variables (S : ℕ → α)
variables (S_3 S_6 : α)

-- Given conditions
axiom is_arith_seq : is_arithmetic_sequence a
axiom S_def : ∀ n, S n = sum_of_arithmetic_sequence a n
axiom S_3_eq : S 3 = 9
axiom S_6_eq : S 6 = 36

-- Theorem to prove
theorem find_a7_a8_a9 : a 7 + a 8 + a 9 = 45 :=
sorry

end find_a7_a8_a9_l192_192098


namespace find_k_l192_192328

theorem find_k (k : ℝ) :
  (∃ x : ℝ, 8 * x - k = 2 * (x + 1) ∧ 2 * (2 * x - 3) = 1 - 3 * x) → k = 4 :=
by
  sorry

end find_k_l192_192328


namespace first_scenario_machines_l192_192725

theorem first_scenario_machines (M : ℕ) (h1 : 20 = 10 * 2 * M) (h2 : 140 = 20 * 17.5 * 2) : M = 5 :=
by sorry

end first_scenario_machines_l192_192725


namespace smallest_integer_ends_in_3_divisible_by_11_correct_l192_192693

def ends_in_3 (n : ℕ) : Prop :=
  n % 10 = 3

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def smallest_ends_in_3_divisible_by_11 : ℕ :=
  33

theorem smallest_integer_ends_in_3_divisible_by_11_correct :
  smallest_ends_in_3_divisible_by_11 = 33 ∧ ends_in_3 smallest_ends_in_3_divisible_by_11 ∧ divisible_by_11 smallest_ends_in_3_divisible_by_11 := 
by
  sorry

end smallest_integer_ends_in_3_divisible_by_11_correct_l192_192693


namespace number_of_pictures_l192_192482

theorem number_of_pictures (x : ℕ) (h : x - (x / 2 - 1) = 25) : x = 48 :=
sorry

end number_of_pictures_l192_192482


namespace time_in_3467_hours_l192_192883

-- Define the current time, the number of hours, and the modulus
def current_time : ℕ := 2
def hours_from_now : ℕ := 3467
def clock_modulus : ℕ := 12

-- Define the function to calculate the future time on a 12-hour clock
def future_time (current_time : ℕ) (hours_from_now : ℕ) (modulus : ℕ) : ℕ := 
  (current_time + hours_from_now) % modulus

-- Theorem statement
theorem time_in_3467_hours :
  future_time current_time hours_from_now clock_modulus = 9 :=
by
  -- Proof would go here
  sorry

end time_in_3467_hours_l192_192883


namespace sum_of_x_values_proof_l192_192212

noncomputable def sum_of_x_values : ℝ := 
  (-(-4)) / 1 -- Sum of roots of x^2 - 4x - 7 = 0

theorem sum_of_x_values_proof (x : ℝ) (h : 7 = (x^3 - 2 * x^2 - 8 * x) / (x + 2)) : sum_of_x_values = 4 :=
sorry

end sum_of_x_values_proof_l192_192212


namespace remainder_when_divided_by_296_l192_192411

theorem remainder_when_divided_by_296 (N : ℤ) (Q : ℤ) (R : ℤ)
  (h1 : N % 37 = 1)
  (h2 : N = 296 * Q + R)
  (h3 : 0 ≤ R) 
  (h4 : R < 296) :
  R = 260 := 
sorry

end remainder_when_divided_by_296_l192_192411


namespace solve_for_x_l192_192750

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 ↔ x = 2 / 9 := by
  sorry

end solve_for_x_l192_192750


namespace eliza_total_clothes_l192_192719

def time_per_blouse : ℕ := 15
def time_per_dress : ℕ := 20
def blouse_time : ℕ := 2 * 60   -- 2 hours in minutes
def dress_time : ℕ := 3 * 60    -- 3 hours in minutes

theorem eliza_total_clothes :
  (blouse_time / time_per_blouse) + (dress_time / time_per_dress) = 17 :=
by
  sorry

end eliza_total_clothes_l192_192719


namespace abs_expression_value_l192_192042

theorem abs_expression_value : (abs (2 * Real.pi - abs (Real.pi - 9))) = 3 * Real.pi - 9 := 
by
  sorry

end abs_expression_value_l192_192042


namespace average_marks_combined_l192_192074

theorem average_marks_combined (P C M B E : ℕ) (h : P + C + M + B + E = P + 280) : 
  (C + M + B + E) / 4 = 70 :=
by 
  sorry

end average_marks_combined_l192_192074


namespace fencing_rate_3_rs_per_meter_l192_192055

noncomputable def rate_per_meter (A_hectares : ℝ) (total_cost : ℝ) : ℝ := 
  let A_m2 := A_hectares * 10000
  let r := Real.sqrt (A_m2 / Real.pi)
  let C := 2 * Real.pi * r
  total_cost / C

theorem fencing_rate_3_rs_per_meter : rate_per_meter 17.56 4456.44 = 3.00 :=
by 
  sorry

end fencing_rate_3_rs_per_meter_l192_192055


namespace sequence_twice_square_l192_192651

theorem sequence_twice_square (n : ℕ) (a : ℕ → ℕ) :
    (∀ i : ℕ, a i = 0) →
    (∀ m : ℕ, 1 ≤ m ∧ m ≤ n → 
        ∀ i : ℕ, i % (2 * m) = 0 → 
            a i = if a i = 0 then 1 else 0) →
    (∀ i : ℕ, a i = 1 ↔ ∃ k : ℕ, i = 2 * k^2) :=
by
  sorry

end sequence_twice_square_l192_192651


namespace children_on_playground_l192_192567

theorem children_on_playground (boys_soccer girls_soccer boys_swings girls_swings boys_snacks girls_snacks : ℕ)
(h1 : boys_soccer = 27) (h2 : girls_soccer = 35)
(h3 : boys_swings = 15) (h4 : girls_swings = 20)
(h5 : boys_snacks = 10) (h6 : girls_snacks = 5) :
boys_soccer + girls_soccer + boys_swings + girls_swings + boys_snacks + girls_snacks = 112 := by
  sorry

end children_on_playground_l192_192567


namespace five_digit_number_divisibility_l192_192739

theorem five_digit_number_divisibility (a : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) : 11 ∣ 100001 * a :=
by
  sorry

end five_digit_number_divisibility_l192_192739


namespace parabola_problem_l192_192695

theorem parabola_problem (a x1 x2 y1 y2 : ℝ)
  (h1 : y1^2 = a * x1)
  (h2 : y2^2 = a * x2)
  (h3 : x1 + x2 = 8)
  (h4 : (x2 - x1)^2 + (y2 - y1)^2 = 144) : 
  a = 8 := 
sorry

end parabola_problem_l192_192695


namespace sin_A_mul_sin_B_find_c_l192_192548

-- Definitions for the triangle and the given conditions
variable (A B C : ℝ) -- Angles of the triangle
variable (a b c : ℝ) -- Opposite sides of the triangle

-- Given conditions
axiom h1 : c^2 = 4 * a * b * (Real.sin C)^2

-- The first proof problem statement
theorem sin_A_mul_sin_B (ha : A + B + C = π) (h2 : Real.sin C ≠ 0) :
  Real.sin A * Real.sin B = 1/4 :=
by
  sorry

-- The second proof problem statement with additional given conditions
theorem find_c (ha : A = π / 6) (ha2 : a = 3) (hb2 : b = 3) : 
  c = 3 * Real.sqrt 3 :=
by
  sorry

end sin_A_mul_sin_B_find_c_l192_192548


namespace trig_identity_proof_l192_192755

noncomputable def trig_identity (α β : ℝ) (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 2) : ℝ :=
  (Real.sin (2 * α)) / (Real.cos (2 * β))

theorem trig_identity_proof (α β : ℝ) (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 2) :
  trig_identity α β h1 h2 = 1 :=
sorry

end trig_identity_proof_l192_192755


namespace years_to_earn_house_l192_192715

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

end years_to_earn_house_l192_192715


namespace percentage_of_truth_speakers_l192_192909

theorem percentage_of_truth_speakers
  (L : ℝ) (hL: L = 0.2)
  (B : ℝ) (hB: B = 0.1)
  (prob_truth_or_lies : ℝ) (hProb: prob_truth_or_lies = 0.4)
  (T : ℝ)
: T = prob_truth_or_lies - L + B :=
sorry

end percentage_of_truth_speakers_l192_192909


namespace Sean_Julie_ratio_l192_192270

-- Define the sum of the first n natural numbers
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of even numbers up to 2n
def sum_even (n : ℕ) : ℕ := 2 * sum_n n

theorem Sean_Julie_ratio : 
  (sum_even 250) / (sum_n 250) = 2 := 
by
  sorry

end Sean_Julie_ratio_l192_192270


namespace lcm_hcf_relationship_l192_192933

theorem lcm_hcf_relationship (a b : ℕ) (h_prod : a * b = 84942) (h_hcf : Nat.gcd a b = 33) : Nat.lcm a b = 2574 :=
by
  sorry

end lcm_hcf_relationship_l192_192933


namespace even_and_multiple_of_3_l192_192060

theorem even_and_multiple_of_3 (a b : ℤ) (h1 : ∃ k : ℤ, a = 2 * k) (h2 : ∃ n : ℤ, b = 6 * n) :
  (∃ m : ℤ, a + b = 2 * m) ∧ (∃ p : ℤ, a + b = 3 * p) :=
by
  sorry

end even_and_multiple_of_3_l192_192060


namespace sum_of_first_three_tests_l192_192007

variable (A B C: ℕ)

def scores (A B C test4 : ℕ) : Prop := (A + B + C + test4) / 4 = 85

theorem sum_of_first_three_tests (h : scores A B C 100) : A + B + C = 240 :=
by
  -- Proof goes here
  sorry

end sum_of_first_three_tests_l192_192007


namespace smallest_solution_l192_192559

theorem smallest_solution (x : ℝ) : 
  (∃ x, (3 * x / (x - 3)) + ((3 * x^2 - 27) / x) = 15 ∧ ∀ y, (3 * y / (y - 3)) + ((3 * y^2 - 27) / y) = 15 → y ≥ x) → 
  x = -1 := 
by
  sorry

end smallest_solution_l192_192559


namespace bonnie_roark_wire_length_ratio_l192_192023

-- Define the conditions
def bonnie_wire_pieces : ℕ := 12
def bonnie_wire_length_per_piece : ℕ := 8
def roark_wire_length_per_piece : ℕ := 2
def bonnie_cube_volume : ℕ := 8 * 8 * 8
def roark_total_cube_volume : ℕ := bonnie_cube_volume
def roark_unit_cube_volume : ℕ := 1
def roark_unit_cube_wires : ℕ := 12

-- Calculate Bonnie's total wire length
noncomputable def bonnie_total_wire_length : ℕ := bonnie_wire_pieces * bonnie_wire_length_per_piece

-- Calculate the number of Roark's unit cubes
noncomputable def roark_number_of_unit_cubes : ℕ := roark_total_cube_volume / roark_unit_cube_volume

-- Calculate the total wire used by Roark
noncomputable def roark_total_wire_length : ℕ := roark_number_of_unit_cubes * roark_unit_cube_wires * roark_wire_length_per_piece

-- Calculate the ratio of Bonnie's total wire length to Roark's total wire length
noncomputable def wire_length_ratio : ℚ := bonnie_total_wire_length / roark_total_wire_length

-- State the theorem
theorem bonnie_roark_wire_length_ratio : wire_length_ratio = 1 / 128 := 
by 
  sorry

end bonnie_roark_wire_length_ratio_l192_192023


namespace remainder_of_xyz_l192_192478

theorem remainder_of_xyz {x y z : ℕ} (hx: x < 9) (hy: y < 9) (hz: z < 9)
  (h1: (x + 3*y + 2*z) % 9 = 0)
  (h2: (2*x + 2*y + z) % 9 = 7)
  (h3: (x + 2*y + 3*z) % 9 = 5) :
  (x * y * z) % 9 = 5 :=
sorry

end remainder_of_xyz_l192_192478


namespace triangle_inequality_iff_inequality_l192_192177

theorem triangle_inequality_iff_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) ↔ (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  sorry

end triangle_inequality_iff_inequality_l192_192177


namespace first_investment_percentage_l192_192117

theorem first_investment_percentage :
  let total_inheritance := 4000
  let invested_6_5 := 1800
  let interest_rate_6_5 := 0.065
  let total_interest := 227
  let remaining_investment := total_inheritance - invested_6_5
  let interest_from_6_5 := invested_6_5 * interest_rate_6_5
  let interest_from_remaining := total_interest - interest_from_6_5
  let P := interest_from_remaining / remaining_investment
  P = 0.05 :=
by 
  sorry

end first_investment_percentage_l192_192117


namespace proof_option_b_and_c_l192_192914

variable (a b c : ℝ)

theorem proof_option_b_and_c (h₀ : a > b) (h₁ : b > 0) (h₂ : c ≠ 0) :
  (b / a < (b + c^2) / (a + c^2)) ∧ (a^2 - 1 / a > b^2 - 1 / b) :=
by
  sorry

end proof_option_b_and_c_l192_192914


namespace pyramid_height_l192_192236

noncomputable def height_of_pyramid (h : ℝ) : Prop :=
  let cube_edge_length := 6
  let pyramid_base_edge_length := 12
  let V_cube := cube_edge_length ^ 3
  let V_pyramid := (1 / 3) * (pyramid_base_edge_length ^ 2) * h
  V_cube = V_pyramid → h = 4.5

theorem pyramid_height : height_of_pyramid 4.5 :=
by {
  sorry
}

end pyramid_height_l192_192236


namespace xiaotian_sep_usage_plan_cost_effectiveness_l192_192202

noncomputable def problem₁ (units : List Int) : Real :=
  units.sum / 1024 + 5 * 6

theorem xiaotian_sep_usage (units : List Int) (h : units = [200, -100, 100, -100, 212, 200]) :
  problem₁ units = 30.5 :=
sorry

def plan_cost_a (x : Int) : Real := 5 * x + 4

def plan_cost_b (x : Int) : Real :=
  if h : 20 < x ∧ x <= 23 then 5 * x - 1
  else 3 * x + 45

theorem plan_cost_effectiveness (x : Int) (h : x > 23) :
  plan_cost_a x > plan_cost_b x :=
sorry

end xiaotian_sep_usage_plan_cost_effectiveness_l192_192202


namespace consecutive_integers_sum_l192_192226

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l192_192226


namespace even_n_equals_identical_numbers_l192_192734

theorem even_n_equals_identical_numbers (n : ℕ) (h1 : n ≥ 2) : 
  (∃ f : ℕ → ℕ, (∀ a b, f a = f b + f b) ∧ n % 2 = 0) :=
sorry


end even_n_equals_identical_numbers_l192_192734


namespace probability_top_two_same_suit_l192_192650

theorem probability_top_two_same_suit :
  let deck_size := 52
  let suits := 4
  let cards_per_suit := 13
  let first_card_prob := (13 / 52 : ℚ)
  let remaining_cards := 51
  let second_card_same_suit_prob := (12 / 51 : ℚ)
  first_card_prob * second_card_same_suit_prob = (1 / 17 : ℚ) :=
by
  sorry

end probability_top_two_same_suit_l192_192650


namespace inequality_solution_set_l192_192903

theorem inequality_solution_set {a : ℝ} (x : ℝ) :
  (∀ x, (x - a) / (x^2 - 3 * x + 2) ≥ 0 ↔ (1 < x ∧ x ≤ a) ∨ (2 < x)) → (1 < a ∧ a < 2) :=
by 
  -- We would fill in the proof here. 
  sorry

end inequality_solution_set_l192_192903


namespace square_area_from_diagonal_l192_192241

theorem square_area_from_diagonal (d : ℝ) (h : d = 28) : (∃ A : ℝ, A = 392) :=
by
  sorry

end square_area_from_diagonal_l192_192241


namespace f_g_2_eq_neg_19_l192_192130

def f (x : ℝ) : ℝ := 5 - 4 * x

def g (x : ℝ) : ℝ := x^2 + 2

theorem f_g_2_eq_neg_19 : f (g 2) = -19 := 
by
  -- The proof is omitted
  sorry

end f_g_2_eq_neg_19_l192_192130


namespace sum_of_incircle_areas_l192_192596

variables {a b c : ℝ} (ABC : Triangle ℝ) (s K r : ℝ)
  (hs : s = (a + b + c) / 2)
  (hK : K = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
  (hr : r = K / s)

theorem sum_of_incircle_areas :
  let larger_circle_area := π * r^2
  let smaller_circle_area := π * (r / 2)^2
  larger_circle_area + 3 * smaller_circle_area = 7 * π * r^2 / 4 :=
sorry

end sum_of_incircle_areas_l192_192596


namespace all_roots_are_nth_roots_of_unity_l192_192221

noncomputable def smallest_positive_integer_n : ℕ :=
  5
  
theorem all_roots_are_nth_roots_of_unity :
  (∀ z : ℂ, (z^4 + z^3 + z^2 + z + 1 = 0) → z^(smallest_positive_integer_n) = 1) :=
  by
    sorry

end all_roots_are_nth_roots_of_unity_l192_192221


namespace Y_subset_X_l192_192456

def X : Set ℕ := {n | ∃ m : ℕ, n = 4 * m + 2}

def Y : Set ℕ := {t | ∃ k : ℕ, t = (2 * k - 1)^2 + 1}

theorem Y_subset_X : Y ⊆ X := by
  sorry

end Y_subset_X_l192_192456


namespace exchange_rate_l192_192295

theorem exchange_rate (a b : ℕ) (h : 5000 = 60 * a) : b = 75 * a → b = 6250 := by
  sorry

end exchange_rate_l192_192295


namespace commercial_duration_l192_192073

/-- Michael was watching a TV show, which was aired for 1.5 hours. 
    During this time, there were 3 commercials. 
    The TV show itself, not counting commercials, was 1 hour long. 
    Prove that each commercial lasted 10 minutes. -/
theorem commercial_duration (total_time : ℝ) (num_commercials : ℕ) (show_time : ℝ)
  (h1 : total_time = 1.5) (h2 : num_commercials = 3) (h3 : show_time = 1) :
  (total_time - show_time) / num_commercials * 60 = 10 := 
sorry

end commercial_duration_l192_192073


namespace derivative_at_0_l192_192230

noncomputable def f (x : ℝ) := Real.exp x / (x + 2)

theorem derivative_at_0 : deriv f 0 = 1 / 4 := sorry

end derivative_at_0_l192_192230


namespace total_balloons_l192_192260

theorem total_balloons (Gold Silver Black Total : Nat) (h1 : Gold = 141)
  (h2 : Silver = 2 * Gold) (h3 : Black = 150) (h4 : Total = Gold + Silver + Black) :
  Total = 573 := 
by
  sorry

end total_balloons_l192_192260


namespace intersection_eq_M_l192_192403

-- Define the sets M and N according to the given conditions
def M : Set ℝ := {x : ℝ | x^2 - x < 0}
def N : Set ℝ := {x : ℝ | |x| < 2}

-- The 'theorem' statement to prove M ∩ N = M
theorem intersection_eq_M : M ∩ N = M :=
  sorry

end intersection_eq_M_l192_192403


namespace right_angled_triangle_exists_l192_192970

def is_triangle (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_right_angled_triangle (a b c : ℕ) : Prop :=
  (a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)

theorem right_angled_triangle_exists :
  is_triangle 3 4 5 ∧ is_right_angled_triangle 3 4 5 :=
by
  sorry

end right_angled_triangle_exists_l192_192970


namespace geometric_sequence_product_l192_192003

-- Defining the geometric sequence and the equation
noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def satisfies_quadratic_roots (a : ℕ → ℝ) : Prop :=
  (a 2 = -1 ∧ a 18 = -16 / (-1 + 16 / -1) ∨
  a 18 = -1 ∧ a 2 = -16 / (-1 + 16 / -1))

-- Problem statement
theorem geometric_sequence_product (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_roots : satisfies_quadratic_roots a) : 
  a 3 * a 10 * a 17 = -64 :=
sorry

end geometric_sequence_product_l192_192003


namespace age_problem_l192_192057

theorem age_problem (S F : ℕ) (h1 : F = S + 27) (h2 : F + 2 = 2 * (S + 2)) :
  S = 25 := by
  sorry

end age_problem_l192_192057


namespace optimal_room_rate_to_maximize_income_l192_192589

noncomputable def max_income (x : ℝ) : ℝ := x * (300 - 0.5 * (x - 200))

theorem optimal_room_rate_to_maximize_income :
  ∀ x, 200 ≤ x → x ≤ 800 → max_income x ≤ max_income 400 :=
by
  sorry

end optimal_room_rate_to_maximize_income_l192_192589


namespace delores_initial_money_l192_192446

theorem delores_initial_money (cost_computer : ℕ) (cost_printer : ℕ) (money_left : ℕ) (initial_money : ℕ) :
  cost_computer = 400 → cost_printer = 40 → money_left = 10 → initial_money = cost_computer + cost_printer + money_left → initial_money = 450 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end delores_initial_money_l192_192446


namespace radius_of_sphere_l192_192138

-- Define the conditions.
def radius_wire : ℝ := 8
def length_wire : ℝ := 36

-- Given the volume of the metallic sphere is equal to the volume of the wire,
-- Prove that the radius of the sphere is 12 cm.
theorem radius_of_sphere (r_wire : ℝ) (h_wire : ℝ) (r_sphere : ℝ) : 
    r_wire = radius_wire → h_wire = length_wire →
    (π * r_wire^2 * h_wire = (4/3) * π * r_sphere^3) → 
    r_sphere = 12 :=
by
  intros h₁ h₂ h₃
  -- Add proof steps here.
  sorry

end radius_of_sphere_l192_192138


namespace correlation_coefficients_l192_192551

-- Definition of the variables and constants
def relative_risks_starting_age : List (ℕ × ℝ) := [(16, 15.10), (18, 12.81), (20, 9.72), (22, 3.21)]
def relative_risks_cigarettes_per_day : List (ℕ × ℝ) := [(10, 7.5), (20, 9.5), (30, 16.6)]

def r1 : ℝ := -- The correlation coefficient between starting age and relative risk
  sorry

def r2 : ℝ := -- The correlation coefficient between number of cigarettes per day and relative risk
  sorry

theorem correlation_coefficients :
  r1 < 0 ∧ 0 < r2 :=
by {
  -- Proof is skipped with sorry
  sorry
}

end correlation_coefficients_l192_192551


namespace cab_income_third_day_l192_192811

noncomputable def cab_driver_income (day1 day2 day3 day4 day5 : ℕ) : ℕ := 
day1 + day2 + day3 + day4 + day5

theorem cab_income_third_day 
  (day1 day2 day4 day5 avg_income total_income day3 : ℕ)
  (h1 : day1 = 45)
  (h2 : day2 = 50)
  (h3 : day4 = 65)
  (h4 : day5 = 70)
  (h_avg : avg_income = 58)
  (h_total : total_income = 5 * avg_income)
  (h_day_sum : day1 + day2 + day4 + day5 = 230) :
  total_income - 230 = 60 :=
sorry

end cab_income_third_day_l192_192811


namespace existence_of_xyz_l192_192265

theorem existence_of_xyz (n : ℕ) (hn_pos : 0 < n)
    (a b c : ℕ) (ha : 0 < a ∧ a ≤ 3 * n^2 + 4 * n) 
                (hb : 0 < b ∧ b ≤ 3 * n^2 + 4 * n) 
                (hc : 0 < c ∧ c ≤ 3 * n^2 + 4 * n) : 
  ∃ (x y z : ℤ), (|x| ≤ 2 * n) ∧ (|y| ≤ 2 * n) ∧ (|z| ≤ 2 * n) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 := by
  sorry

end existence_of_xyz_l192_192265


namespace unit_prices_max_colored_tiles_l192_192511

-- Define the given conditions
def condition1 (x y : ℝ) := 40 * x + 60 * y = 5600
def condition2 (x y : ℝ) := 50 * x + 50 * y = 6000

-- Prove the solution for part 1
theorem unit_prices (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 80 ∧ y = 40 := 
sorry

-- Define the condition for the kitchen tiles
def condition3 (a : ℝ) := 80 * a + 40 * (60 - a) ≤ 3400

-- Prove the maximum number of colored tiles for the kitchen
theorem max_colored_tiles (a : ℝ) (h3 : condition3 a) :
  a ≤ 25 := 
sorry

end unit_prices_max_colored_tiles_l192_192511


namespace range_of_m_l192_192763

theorem range_of_m (m : ℝ) (x : ℝ) (h₁ : x^2 - 8*x - 20 ≤ 0) 
  (h₂ : (x - 1 - m) * (x - 1 + m) ≤ 0) (h₃ : 0 < m) : 
  m ≤ 3 := sorry

end range_of_m_l192_192763


namespace triangle_inequality_proof_l192_192974

theorem triangle_inequality_proof (a b c : ℝ) (PA QA PB QB PC QC : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hpa : PA ≥ 0) (hqa : QA ≥ 0) (hpb : PB ≥ 0) (hqb : QB ≥ 0) 
  (hpc : PC ≥ 0) (hqc : QC ≥ 0):
  a * PA * QA + b * PB * QB + c * PC * QC ≥ a * b * c := 
sorry

end triangle_inequality_proof_l192_192974


namespace rectangle_area_l192_192170

namespace RectangleAreaProof

theorem rectangle_area (SqrArea : ℝ) (SqrSide : ℝ) (RectWidth : ℝ) (RectLength : ℝ) (RectArea : ℝ) :
  SqrArea = 36 →
  SqrSide = Real.sqrt SqrArea →
  RectWidth = SqrSide →
  RectLength = 3 * RectWidth →
  RectArea = RectWidth * RectLength →
  RectArea = 108 := by
  sorry

end RectangleAreaProof

end rectangle_area_l192_192170


namespace vacant_seats_l192_192037

theorem vacant_seats (total_seats : ℕ) (filled_percent vacant_percent : ℚ) 
  (h_total : total_seats = 600)
  (h_filled_percent : filled_percent = 75)
  (h_vacant_percent : vacant_percent = 100 - filled_percent)
  (h_vacant_percent_25 : vacant_percent = 25) :
  (25 / 100) * 600 = 150 :=
by 
  -- this is the final answer we want to prove, replace with sorry to skip the proof just for statement validation
  sorry

end vacant_seats_l192_192037


namespace dishonest_shopkeeper_gain_l192_192437

-- Conditions: false weight used by shopkeeper
def false_weight : ℚ := 930
def true_weight : ℚ := 1000

-- Correct answer: gain percentage
def gain_percentage (false_weight true_weight : ℚ) : ℚ :=
  ((true_weight - false_weight) / false_weight) * 100

theorem dishonest_shopkeeper_gain :
  gain_percentage false_weight true_weight = 7.53 := by
  sorry

end dishonest_shopkeeper_gain_l192_192437


namespace train_distance_difference_l192_192396

theorem train_distance_difference:
  ∀ (D1 D2 : ℕ) (t : ℕ), 
    (D1 = 20 * t) →            -- Slower train's distance
    (D2 = 25 * t) →           -- Faster train's distance
    (D1 + D2 = 450) →         -- Total distance between stations
    (D2 - D1 = 50) := 
by
  intros D1 D2 t h1 h2 h3
  sorry

end train_distance_difference_l192_192396


namespace alexa_pages_left_l192_192751

theorem alexa_pages_left 
  (total_pages : ℕ) 
  (first_day_read : ℕ) 
  (next_day_read : ℕ) 
  (total_pages_val : total_pages = 95) 
  (first_day_read_val : first_day_read = 18) 
  (next_day_read_val : next_day_read = 58) : 
  total_pages - (first_day_read + next_day_read) = 19 := by
  sorry

end alexa_pages_left_l192_192751


namespace smallest_integer_with_remainders_l192_192213

theorem smallest_integer_with_remainders :
  ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 10 = 9 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m) := 
sorry

end smallest_integer_with_remainders_l192_192213


namespace total_distance_walked_l192_192075

def distance_to_fountain : ℕ := 30
def number_of_trips : ℕ := 4
def round_trip_distance : ℕ := 2 * distance_to_fountain

theorem total_distance_walked : (number_of_trips * round_trip_distance) = 240 := by
  sorry

end total_distance_walked_l192_192075


namespace find_value_of_2a_plus_c_l192_192713

theorem find_value_of_2a_plus_c (a b c : ℝ) (h1 : 3 * a + b + 2 * c = 3) (h2 : a + 3 * b + 2 * c = 1) :
  2 * a + c = 2 :=
sorry

end find_value_of_2a_plus_c_l192_192713


namespace min_value_geometric_sequence_l192_192283

theorem min_value_geometric_sequence (a_2 a_3 : ℝ) (r : ℝ) 
(h_a2 : a_2 = 2 * r) (h_a3 : a_3 = 2 * r^2) : 
  (6 * a_2 + 7 * a_3) = -18 / 7 :=
by
  sorry

end min_value_geometric_sequence_l192_192283


namespace carpet_shaded_area_is_correct_l192_192669

def total_shaded_area (carpet_side_length : ℝ) (large_square_side : ℝ) (small_square_side : ℝ) : ℝ :=
  let large_shaded_area := large_square_side * large_square_side
  let small_shaded_area := small_square_side * small_square_side
  large_shaded_area + 12 * small_shaded_area

theorem carpet_shaded_area_is_correct :
  ∀ (S T : ℝ), 
  12 / S = 4 →
  S / T = 4 →
  total_shaded_area 12 S T = 15.75 :=
by
  intros S T h1 h2
  sorry

end carpet_shaded_area_is_correct_l192_192669


namespace michael_watermelon_weight_l192_192965

theorem michael_watermelon_weight (m c j : ℝ) (h1 : c = 3 * m) (h2 : j = c / 2) (h3 : j = 12) : m = 8 :=
by
  sorry

end michael_watermelon_weight_l192_192965


namespace intersection_of_solution_sets_solution_set_of_modified_inequality_l192_192756

open Set Real

theorem intersection_of_solution_sets :
  let A := {x | x ^ 2 - 2 * x - 3 < 0}
  let B := {x | x ^ 2 + x - 6 < 0}
  A ∩ B = {x | -1 < x ∧ x < 2} := by {
  sorry
}

theorem solution_set_of_modified_inequality :
  let A := {x | x ^ 2 + (-1) * x + (-2) < 0}
  A = {x | true} := by {
  sorry
}

end intersection_of_solution_sets_solution_set_of_modified_inequality_l192_192756


namespace sequence_term_1000_l192_192513

theorem sequence_term_1000 :
  ∃ (a : ℕ → ℤ), a 1 = 2007 ∧ a 2 = 2008 ∧ (∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n) ∧ a 1000 = 2340 := 
by
  sorry

end sequence_term_1000_l192_192513


namespace speed_ratio_l192_192552

variable (v_A v_B : ℝ)

def equidistant_3min : Prop := 3 * v_A = abs (-800 + 3 * v_B)
def equidistant_8min : Prop := 8 * v_A = abs (-800 + 8 * v_B)
def speed_ratio_correct : Prop := v_A / v_B = 1 / 2

theorem speed_ratio (h1 : equidistant_3min v_A v_B) (h2 : equidistant_8min v_A v_B) : speed_ratio_correct v_A v_B :=
by
  sorry

end speed_ratio_l192_192552


namespace partial_fraction_sum_inverse_l192_192872

theorem partial_fraction_sum_inverse (p q r A B C : ℝ)
  (hroots : (∀ s, s^3 - 20 * s^2 + 96 * s - 91 = (s - p) * (s - q) * (s - r)))
  (hA : ∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 20 * s^2 + 96 * s - 91) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1 / A + 1 / B + 1 / C = 225 :=
sorry

end partial_fraction_sum_inverse_l192_192872


namespace xiao_ming_selects_cooking_probability_l192_192541

theorem xiao_ming_selects_cooking_probability :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let probability (event: String) := if event ∈ courses then 1 / (courses.length : ℝ) else 0
  probability "cooking" = 1 / 4 :=
by
  sorry

end xiao_ming_selects_cooking_probability_l192_192541


namespace total_volume_structure_l192_192420

theorem total_volume_structure (d : ℝ) (h_cone : ℝ) (h_cylinder : ℝ) 
  (r := d / 2) 
  (V_cone := (1 / 3) * π * r^2 * h_cone) 
  (V_cylinder := π * r^2 * h_cylinder) 
  (V_total := V_cone + V_cylinder) :
  d = 8 → h_cone = 9 → h_cylinder = 4 → V_total = 112 * π :=
by
  intros
  sorry

end total_volume_structure_l192_192420


namespace solve_equation_l192_192505

theorem solve_equation :
  ∀ x : ℝ, (4 * x - 2 * x + 1 - 3 = 0) ↔ (x = 1 ∨ x = -1) :=
by
  intro x
  sorry

end solve_equation_l192_192505


namespace root_of_quadratic_l192_192945

theorem root_of_quadratic :
  (∀ x : ℝ, 2 * x^2 + 3 * x - 65 = 0 → x = 5 ∨ x = -6.5) :=
sorry

end root_of_quadratic_l192_192945


namespace probability_meeting_part_a_l192_192299

theorem probability_meeting_part_a :
  ∃ p : ℝ, p = (11 : ℝ) / 36 :=
sorry

end probability_meeting_part_a_l192_192299


namespace find_a_l192_192874

-- We define the conditions given in the problem
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The expression defined as per the problem statement
def expansion_coeff_x2 (a : ℝ) : ℝ :=
  (binom 4 2) * 4 - 2 * (binom 4 1) * (binom 5 1) * a + (binom 5 2) * a^2

-- We now express the proof statement in Lean 4. 
-- We need to prove that given the coefficient of x^2 is -16, then a = 2
theorem find_a (a : ℝ) (h : expansion_coeff_x2 a = -16) : a = 2 :=
  by sorry

end find_a_l192_192874


namespace wall_area_l192_192517

-- Definition of the width and length of the wall
def width : ℝ := 5.4
def length : ℝ := 2.5

-- Statement of the theorem
theorem wall_area : (width * length) = 13.5 :=
by
  sorry

end wall_area_l192_192517


namespace remainder_division_l192_192077

theorem remainder_division (n r : ℕ) (k : ℤ) (h1 : n % 25 = r) (h2 : (n + 15) % 5 = r) (h3 : 0 ≤ r ∧ r < 25) : r = 5 :=
sorry

end remainder_division_l192_192077


namespace lcm_16_24_45_l192_192069

-- Define the numbers
def a : Nat := 16
def b : Nat := 24
def c : Nat := 45

-- State the theorem that the least common multiple of these numbers is 720
theorem lcm_16_24_45 : Nat.lcm (Nat.lcm 16 24) 45 = 720 := by
  sorry

end lcm_16_24_45_l192_192069


namespace painting_frame_ratio_proof_l192_192006

def framed_painting_ratio (x : ℝ) : Prop :=
  let width := 20
  let height := 20
  let side_border := x
  let top_bottom_border := 3 * x
  let framed_width := width + 2 * side_border
  let framed_height := height + 2 * top_bottom_border
  let painting_area := width * height
  let frame_area := painting_area
  let total_area := framed_width * framed_height - painting_area
  total_area = frame_area ∧ (width + 2 * side_border) ≤ (height + 2 * top_bottom_border) → 
  framed_width / framed_height = 4 / 7

theorem painting_frame_ratio_proof (x : ℝ) (h : framed_painting_ratio x) : (20 + 2 * x) / (20 + 6 * x) = 4 / 7 :=
  sorry

end painting_frame_ratio_proof_l192_192006


namespace ratio_of_length_to_width_l192_192386

variable (L W : ℕ)
variable (H1 : W = 50)
variable (H2 : 2 * L + 2 * W = 240)

theorem ratio_of_length_to_width : L / W = 7 / 5 := 
by sorry

end ratio_of_length_to_width_l192_192386


namespace smallest_abcd_value_l192_192963

theorem smallest_abcd_value (A B C D : ℕ) (h1 : A ≠ B) (h2 : 1 ≤ A) (h3 : A ≤ 9) (h4 : 0 ≤ B) 
                            (h5 : B ≤ 9) (h6 : 1 ≤ C) (h7 : C ≤ 9) (h8 : 1 ≤ D) (h9 : D ≤ 9)
                            (h10 : 10 * A * A + A * B = 1000 * A + 100 * B + 10 * C + D)
                            (h11 : A ≠ C) (h12 : A ≠ D) (h13 : B ≠ C) (h14 : B ≠ D) (h15 : C ≠ D) :
  1000 * A + 100 * B + 10 * C + D = 2046 :=
sorry

end smallest_abcd_value_l192_192963


namespace find_b_find_area_l192_192714

open Real

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := A + π / 2
noncomputable def a : ℝ := 3
noncomputable def cos_A : ℝ := sqrt 6 / 3
noncomputable def b : ℝ := 3 * sqrt 2
noncomputable def area : ℝ := 3 * sqrt 2 / 2

theorem find_b (A : ℝ) (H1 : a = 3) (H2 : cos A = sqrt 6 / 3) (H3 : B = A + π / 2) : 
  b = 3 * sqrt 2 := 
  sorry

theorem find_area (A : ℝ) (H1 : a = 3) (H2 : cos A = sqrt 6 / 3) (H3 : B = A + π / 2) : 
  area = 3 * sqrt 2 / 2 := 
  sorry

end find_b_find_area_l192_192714


namespace find_x_in_inches_l192_192040

theorem find_x_in_inches (x : ℝ) :
  let area_smaller_square := 9 * x^2
  let area_larger_square := 36 * x^2
  let area_triangle := 9 * x^2
  area_smaller_square + area_larger_square + area_triangle = 1950 → x = (5 * Real.sqrt 13) / 3 :=
by
  sorry

end find_x_in_inches_l192_192040


namespace max_convex_quadrilaterals_l192_192204

-- Define the points on the plane and the conditions
variable (A : Fin 7 → (ℝ × ℝ))

-- Hypothesis that any 3 given points are not collinear
def not_collinear (P Q R : (ℝ × ℝ)) : Prop :=
  (Q.1 - P.1) * (R.2 - P.2) ≠ (Q.2 - P.2) * (R.1 - P.1)

-- Hypothesis that the convex hull of all points is \triangle A1 A2 A3
def convex_hull_triangle (A : Fin 7 → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : Fin 7), i ≠ j → j ≠ k → i ≠ k → not_collinear (A i) (A j) (A k)

-- The theorem to be proven
theorem max_convex_quadrilaterals :
  convex_hull_triangle A →
  (∀ i j k : Fin 7, i ≠ j → j ≠ k → i ≠ k → not_collinear (A i) (A j) (A k)) →
  ∃ n, n = 17 := 
by
  sorry

end max_convex_quadrilaterals_l192_192204


namespace algebra_inequality_l192_192870

theorem algebra_inequality (a b c : ℝ) 
  (H : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) :=
sorry

end algebra_inequality_l192_192870


namespace function_range_l192_192447

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sin x

theorem function_range : 
  ∀ x : ℝ, (0 < x ∧ x < Real.pi) → 1 ≤ f x ∧ f x ≤ 3 / 2 :=
by
  intro x
  sorry

end function_range_l192_192447


namespace birdhouse_flown_distance_l192_192313

-- Definition of the given conditions.
def car_distance : ℕ := 200
def lawn_chair_distance : ℕ := 2 * car_distance
def birdhouse_distance : ℕ := 3 * lawn_chair_distance

-- Statement of the proof problem.
theorem birdhouse_flown_distance : birdhouse_distance = 1200 := by
  sorry

end birdhouse_flown_distance_l192_192313


namespace circle_area_eq_25pi_l192_192825

theorem circle_area_eq_25pi :
  (∃ (x y : ℝ), x^2 + y^2 - 4 * x + 6 * y - 12 = 0) →
  (∃ (area : ℝ), area = 25 * Real.pi) :=
by
  sorry

end circle_area_eq_25pi_l192_192825


namespace geometric_sequence_and_sum_l192_192743

theorem geometric_sequence_and_sum (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ)
  (h_a1 : a 1 = 3/2)
  (h_a_recur : ∀ n : ℕ, a (n + 1) = 3 * a n - 1)
  (h_b_def : ∀ n : ℕ, b n = a n - 1/2) :
  (∀ n : ℕ, b (n + 1) = 3 * b n ∧ b 1 = 1) ∧ 
  (∀ n : ℕ, S n = (3^n + n - 1) / 2) :=
sorry

end geometric_sequence_and_sum_l192_192743


namespace gcd_f_x_l192_192498

-- Define that x is a multiple of 23478
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

-- Define the function f(x)
noncomputable def f (x : ℕ) : ℕ := (2 * x + 3) * (7 * x + 2) * (13 * x + 7) * (x + 13)

-- Assert the proof problem
theorem gcd_f_x (x : ℕ) (h : is_multiple_of x 23478) : Nat.gcd (f x) x = 546 :=
by 
  sorry

end gcd_f_x_l192_192498


namespace problem_statement_l192_192760

theorem problem_statement (x : ℝ) (h : (2024 - x)^2 + (2022 - x)^2 = 4038) : 
  (2024 - x) * (2022 - x) = 2017 :=
sorry

end problem_statement_l192_192760


namespace inequality_N_value_l192_192268

theorem inequality_N_value (a c : ℝ) (ha : 0 < a) (hc : 0 < c) (b : ℝ) (hb : b = 2 * a) : 
  (a^2 + b^2) / c^2 > 5 / 9 := 
by sorry

end inequality_N_value_l192_192268


namespace betty_needs_more_flies_l192_192439

def betty_frog_food (daily_flies: ℕ) (days_per_week: ℕ) (morning_catch: ℕ) 
  (afternoon_catch: ℕ) (flies_escaped: ℕ) : ℕ :=
  days_per_week * daily_flies - (morning_catch + afternoon_catch - flies_escaped)

theorem betty_needs_more_flies :
  betty_frog_food 2 7 5 6 1 = 4 :=
by
  sorry

end betty_needs_more_flies_l192_192439


namespace seats_needed_l192_192986

theorem seats_needed (children seats_per_seat : ℕ) (h1 : children = 58) (h2 : seats_per_seat = 2) : children / seats_per_seat = 29 :=
by sorry

end seats_needed_l192_192986


namespace opening_price_calculation_l192_192476

variable (Closing_Price : ℝ)
variable (Percent_Increase : ℝ)
variable (Opening_Price : ℝ)

theorem opening_price_calculation
    (H1 : Closing_Price = 28)
    (H2 : Percent_Increase = 0.1200000000000001) :
    Opening_Price = Closing_Price / (1 + Percent_Increase) := by
  sorry

end opening_price_calculation_l192_192476


namespace oxygen_atom_diameter_in_scientific_notation_l192_192852

theorem oxygen_atom_diameter_in_scientific_notation :
  0.000000000148 = 1.48 * 10^(-10) :=
sorry

end oxygen_atom_diameter_in_scientific_notation_l192_192852


namespace son_l192_192239

theorem son's_age (S F : ℕ) (h1: F = S + 27) (h2: F + 2 = 2 * (S + 2)) : S = 25 := by
  sorry

end son_l192_192239


namespace additional_people_proof_l192_192545

variable (initialPeople additionalPeople mowingHours trimmingRate totalNewPeople totalMowingPeople requiredPersonHours totalPersonHours: ℕ)

noncomputable def mowingLawn (initialPeople mowingHours : ℕ) : ℕ :=
  initialPeople * mowingHours

noncomputable def mowingRate (requiredPersonHours : ℕ) (mowingHours : ℕ) : ℕ :=
  (requiredPersonHours / mowingHours)

noncomputable def trimmingEdges (totalMowingPeople trimmingRate : ℕ) : ℕ :=
  (totalMowingPeople / trimmingRate)

noncomputable def totalPeople (mowingPeople trimmingPeople : ℕ) : ℕ :=
  (mowingPeople + trimmingPeople)

noncomputable def additionalPeopleNeeded (totalPeople initialPeople : ℕ) : ℕ :=
  (totalPeople - initialPeople)

theorem additional_people_proof :
  initialPeople = 8 →
  mowingHours = 3 →
  totalPersonHours = mowingLawn initialPeople mowingHours →
  totalMowingPeople = mowingRate totalPersonHours 2 →
  trimmingRate = 3 →
  requiredPersonHours = totalPersonHours →
  totalNewPeople = totalPeople totalMowingPeople (trimmingEdges totalMowingPeople trimmingRate) →
  additionalPeople = additionalPeopleNeeded totalNewPeople initialPeople →
  additionalPeople = 8 :=
by
  sorry

end additional_people_proof_l192_192545


namespace ratio_of_numbers_l192_192380

theorem ratio_of_numbers (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) (h₄ : a + b = 7 * (a - b) + 14) : a / b = 4 / 3 :=
sorry

end ratio_of_numbers_l192_192380


namespace jackie_more_apples_oranges_l192_192554

-- Definitions of initial conditions
def adams_apples : ℕ := 25
def adams_oranges : ℕ := 34
def jackies_apples : ℕ := 43
def jackies_oranges : ℕ := 29

-- The proof statement
theorem jackie_more_apples_oranges :
  (jackies_apples - adams_apples) + (jackies_oranges - adams_oranges) = 13 :=
by
  sorry

end jackie_more_apples_oranges_l192_192554


namespace perimeter_of_quadrilateral_l192_192579

theorem perimeter_of_quadrilateral 
  (WXYZ_area : ℝ)
  (h_area : WXYZ_area = 2500)
  (WQ XQ YQ ZQ : ℝ)
  (h_WQ : WQ = 30)
  (h_XQ : XQ = 40)
  (h_YQ : YQ = 35)
  (h_ZQ : ZQ = 50) :
  ∃ (P : ℝ), P = 155 + 10 * Real.sqrt 34 + 5 * Real.sqrt 113 :=
by
  sorry

end perimeter_of_quadrilateral_l192_192579


namespace total_marks_calculation_l192_192021

def average (total_marks : ℕ) (num_candidates : ℕ) : ℕ := total_marks / num_candidates
def total_marks (average : ℕ) (num_candidates : ℕ) : ℕ := average * num_candidates

theorem total_marks_calculation
  (num_candidates : ℕ)
  (average_marks : ℕ)
  (range_min : ℕ)
  (range_max : ℕ)
  (h1 : num_candidates = 250)
  (h2 : average_marks = 42)
  (h3 : range_min = 10)
  (h4 : range_max = 80) :
  total_marks average_marks num_candidates = 10500 :=
by 
  sorry

end total_marks_calculation_l192_192021


namespace trigonometric_identity_l192_192488

open Real

theorem trigonometric_identity (α : ℝ) (h1 : tan α = 4/3) (h2 : 0 < α ∧ α < π / 2) :
  sin (π + α) + cos (π - α) = -7/5 :=
by
  sorry

end trigonometric_identity_l192_192488


namespace candy_per_day_eq_eight_l192_192967

def candy_received_from_neighbors : ℝ := 11.0
def candy_received_from_sister : ℝ := 5.0
def days_candy_lasted : ℝ := 2.0

theorem candy_per_day_eq_eight :
  (candy_received_from_neighbors + candy_received_from_sister) / days_candy_lasted = 8.0 :=
by
  sorry

end candy_per_day_eq_eight_l192_192967


namespace selected_numbers_count_l192_192377

noncomputable def check_num_of_selected_numbers : ℕ := 
  let n := 2015
  let max_num := n * n
  let common_difference := 15
  let starting_number := 14
  let count := (max_num - starting_number) / common_difference + 1
  count

theorem selected_numbers_count : check_num_of_selected_numbers = 270681 := by
  -- Skipping the actual proof
  sorry

end selected_numbers_count_l192_192377


namespace charles_initial_bananas_l192_192312

theorem charles_initial_bananas (W C : ℕ) (h1 : W = 48) (h2 : C = C - 35 + W - 13) : C = 35 := by
  -- W = 48
  -- Charles loses 35 bananas
  -- Willie will have 13 bananas
  sorry

end charles_initial_bananas_l192_192312


namespace order_of_a_b_c_l192_192838

noncomputable def a := 2 + Real.sqrt 3
noncomputable def b := 1 + Real.sqrt 6
noncomputable def c := Real.sqrt 2 + Real.sqrt 5

theorem order_of_a_b_c : a > c ∧ c > b := 
by {
  sorry
}

end order_of_a_b_c_l192_192838


namespace triangle_inequality_difference_l192_192571

theorem triangle_inequality_difference :
  (∀ (x : ℤ), (x + 7 > 9) ∧ (x + 9 > 7) ∧ (7 + 9 > x) → (3 ≤ x ∧ x ≤ 15) ∧ (15 - 3 = 12)) :=
by
  sorry

end triangle_inequality_difference_l192_192571


namespace number_of_articles_l192_192627

variables (C S N : ℝ)
noncomputable def gain : ℝ := 3 / 7

-- Cost price of 50 articles is equal to the selling price of N articles
axiom cost_price_eq_selling_price : 50 * C = N * S

-- Selling price is cost price plus gain percentage
axiom selling_price_with_gain : S = C * (1 + gain)

-- Goal: Prove that N = 35
theorem number_of_articles (h1 : 50 * C = N * C * (10 / 7)) : N = 35 := by
  sorry

end number_of_articles_l192_192627


namespace max_xyz_l192_192686

theorem max_xyz (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : 5 * x + 8 * y + 3 * z = 90) : xyz ≤ 225 :=
by
  sorry

end max_xyz_l192_192686


namespace decision_making_system_reliability_l192_192369

theorem decision_making_system_reliability (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  (10 * p^3 - 15 * p^4 + 6 * p^5 > 3 * p^2 - 2 * p^3) -> (1 / 2 < p) ∧ (p < 1) :=
by
  sorry

end decision_making_system_reliability_l192_192369


namespace power_inequality_l192_192549

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) := 
by 
  sorry

end power_inequality_l192_192549


namespace boiling_point_water_standard_l192_192508

def boiling_point_water_celsius : ℝ := 100

theorem boiling_point_water_standard (bp_f : ℝ := 212) (ice_melting_c : ℝ := 0) (ice_melting_f : ℝ := 32) (pot_temp_c : ℝ := 55) (pot_temp_f : ℝ := 131) : boiling_point_water_celsius = 100 :=
by 
  -- Assuming standard atmospheric conditions, the boiling point of water in Celsius is 100.
  sorry

end boiling_point_water_standard_l192_192508


namespace train_crossing_time_l192_192757

-- Conditions
def length_train1 : ℕ := 200 -- Train 1 length in meters
def length_train2 : ℕ := 160 -- Train 2 length in meters
def speed_train1 : ℕ := 68 -- Train 1 speed in kmph
def speed_train2 : ℕ := 40 -- Train 2 speed in kmph

-- Conversion factors and formulas
def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600
def total_distance (l1 l2 : ℕ) := l1 + l2
def relative_speed (s1 s2 : ℕ) := kmph_to_mps (s1 + s2)
def crossing_time (dist speed : ℕ) := dist / speed

-- Proof statement
theorem train_crossing_time : 
  crossing_time (total_distance length_train1 length_train2) (relative_speed speed_train1 speed_train2) = 12 := by sorry

end train_crossing_time_l192_192757


namespace geometric_series_sum_l192_192814

theorem geometric_series_sum : 
  let a : ℕ := 2
  let r : ℕ := 3
  let n : ℕ := 6
  let S_n := (a * (r^n - 1)) / (r - 1)
  S_n = 728 :=
by
  sorry

end geometric_series_sum_l192_192814


namespace cube_side_length_l192_192413

theorem cube_side_length (s : ℝ) (h : 6 * s^2 = 864) : s = 12 := by
  sorry

end cube_side_length_l192_192413


namespace boat_distance_ratio_l192_192047

theorem boat_distance_ratio :
  ∀ (D_u D_d : ℝ),
  (3.6 = (D_u + D_d) / ((D_u / 4) + (D_d / 6))) →
  D_u / D_d = 4 :=
by
  intros D_u D_d h
  sorry

end boat_distance_ratio_l192_192047


namespace female_officers_on_duty_percentage_l192_192041

   def percentage_of_females_on_duty (total_on_duty : ℕ) (female_on_duty : ℕ) (total_females : ℕ) : ℕ :=
   (female_on_duty * 100) / total_females
  
   theorem female_officers_on_duty_percentage
     (total_on_duty : ℕ) (h1 : total_on_duty = 180)
     (female_on_duty : ℕ) (h2 : female_on_duty = total_on_duty / 2)
     (total_females : ℕ) (h3 : total_females = 500) :
     percentage_of_females_on_duty total_on_duty female_on_duty total_females = 18 :=
   by
     rw [h1, h2, h3]
     sorry
   
end female_officers_on_duty_percentage_l192_192041


namespace perimeter_of_rectangle_l192_192851

theorem perimeter_of_rectangle (s : ℝ) (h1 : 4 * s = 160) : 2 * (s + s / 4) = 100 :=
by
  sorry

end perimeter_of_rectangle_l192_192851


namespace simplify_expr_l192_192161

variable (a b : ℝ)

theorem simplify_expr (h : a + b ≠ 0) : 
  a - b + 2 * b^2 / (a + b) = (a^2 + b^2) / (a + b) :=
sorry

end simplify_expr_l192_192161


namespace volume_of_inequality_region_l192_192135

-- Define the inequality condition as a predicate
def region (x y z : ℝ) : Prop :=
  |4 * x - 20| + |3 * y + 9| + |z - 2| ≤ 6

-- Define the volume calculation for the region
def volume_of_region := 36

-- The proof statement
theorem volume_of_inequality_region : 
  (∃ x y z : ℝ, region x y z) → volume_of_region = 36 :=
by
  sorry

end volume_of_inequality_region_l192_192135


namespace sqrt_product_equals_l192_192473

noncomputable def sqrt128 : ℝ := Real.sqrt 128
noncomputable def sqrt50 : ℝ := Real.sqrt 50
noncomputable def sqrt18 : ℝ := Real.sqrt 18

theorem sqrt_product_equals : sqrt128 * sqrt50 * sqrt18 = 240 * Real.sqrt 2 := 
by
  sorry

end sqrt_product_equals_l192_192473


namespace total_treats_l192_192492

theorem total_treats (children : ℕ) (hours : ℕ) (houses_per_hour : ℕ) (treats_per_house_per_kid : ℕ) :
  children = 3 → hours = 4 → houses_per_hour = 5 → treats_per_house_per_kid = 3 → 
  (children * hours * houses_per_hour * treats_per_house_per_kid) = 180 :=
by
  intros
  sorry

end total_treats_l192_192492


namespace polyhedron_has_triangular_face_l192_192917

-- Let's define the structure of a polyhedron, its vertices, edges, and faces.
structure Polyhedron :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)

-- Let's assume a function that indicates if a polyhedron is convex.
def is_convex (P : Polyhedron) : Prop := sorry  -- Convexity needs a rigorous formal definition.

-- Define a face of a polyhedron as an n-sided polygon.
structure Face :=
(sides : ℕ)

-- Predicate to check if a face is triangular.
def is_triangle (F : Face) : Prop := F.sides = 3

-- Predicate to check if each vertex has at least four edges meeting at it.
def each_vertex_has_at_least_four_edges (P : Polyhedron) : Prop := 
  sorry  -- This would need a more intricate definition involving the degrees of vertices.

-- We state the theorem using the defined concepts.
theorem polyhedron_has_triangular_face 
(P : Polyhedron) 
(h1 : is_convex P) 
(h2 : each_vertex_has_at_least_four_edges P) :
∃ (F : Face), is_triangle F :=
sorry

end polyhedron_has_triangular_face_l192_192917


namespace factorize_polynomial_l192_192619

theorem factorize_polynomial (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := 
by 
  sorry

end factorize_polynomial_l192_192619


namespace yellow_highlighters_count_l192_192973

theorem yellow_highlighters_count 
  (Y : ℕ) 
  (pink_highlighters : ℕ := Y + 7) 
  (blue_highlighters : ℕ := Y + 12) 
  (total_highlighters : ℕ := Y + pink_highlighters + blue_highlighters) : 
  total_highlighters = 40 → Y = 7 :=
by
  sorry

end yellow_highlighters_count_l192_192973


namespace monkey_distance_l192_192742

-- Define the initial speeds and percentage adjustments
def swing_speed : ℝ := 10
def run_speed : ℝ := 15
def wind_resistance_percentage : ℝ := 0.10
def branch_assistance_percentage : ℝ := 0.05

-- Conditions
def adjusted_swing_speed : ℝ := swing_speed * (1 - wind_resistance_percentage)
def adjusted_run_speed : ℝ := run_speed * (1 + branch_assistance_percentage)
def run_time : ℝ := 5
def swing_time : ℝ := 10

-- Define the distance formulas based on the conditions
def run_distance : ℝ := adjusted_run_speed * run_time
def swing_distance : ℝ := adjusted_swing_speed * swing_time

-- Total distance calculation
def total_distance : ℝ := run_distance + swing_distance

-- Statement for the proof
theorem monkey_distance : total_distance = 168.75 := by
  sorry

end monkey_distance_l192_192742


namespace angle_B_of_triangle_l192_192205

theorem angle_B_of_triangle {A B C a b c : ℝ} (h1 : b^2 = a * c) (h2 : Real.sin A + Real.sin C = 2 * Real.sin B) : 
  B = Real.pi / 3 :=
sorry

end angle_B_of_triangle_l192_192205


namespace simplify_fraction_150_div_225_l192_192700

theorem simplify_fraction_150_div_225 :
  let a := 150
  let b := 225
  let gcd_ab := Nat.gcd a b
  let num_fact := 2 * 3 * 5^2
  let den_fact := 3^2 * 5^2
  gcd_ab = 75 →
  num_fact = a →
  den_fact = b →
  (a / gcd_ab) / (b / gcd_ab) = (2 / 3) :=
  by
    intros 
    sorry

end simplify_fraction_150_div_225_l192_192700


namespace greatest_possible_y_l192_192116

theorem greatest_possible_y
  (x y : ℤ)
  (h : x * y + 7 * x + 6 * y = -14) : y ≤ 21 :=
sorry

end greatest_possible_y_l192_192116


namespace solve_for_xy_l192_192586

-- The conditions given in the problem
variables (x y : ℝ)
axiom cond1 : 1 / 2 * x - y = 5
axiom cond2 : y - 1 / 3 * x = 2

-- The theorem we need to prove
theorem solve_for_xy (x y : ℝ) (cond1 : 1 / 2 * x - y = 5) (cond2 : y - 1 / 3 * x = 2) : 
  x = 42 ∧ y = 16 := sorry

end solve_for_xy_l192_192586


namespace truck_capacity_l192_192899

-- Definitions based on conditions
def initial_fuel : ℕ := 38
def total_money : ℕ := 350
def change : ℕ := 14
def cost_per_liter : ℕ := 3

-- Theorem statement
theorem truck_capacity :
  initial_fuel + (total_money - change) / cost_per_liter = 150 := by
  sorry

end truck_capacity_l192_192899


namespace minimum_value_of_a_l192_192655

variable (a x y : ℝ)

-- Condition
def condition (x y : ℝ) (a : ℝ) : Prop := 
  (x + y) * ((1/x) + (a/y)) ≥ 9

-- Main statement
theorem minimum_value_of_a : (∀ x > 0, ∀ y > 0, condition x y a) → a ≥ 4 :=
sorry

end minimum_value_of_a_l192_192655


namespace value_of_a_minus_b_l192_192145

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a + b > 0) :
  a - b = 4 ∨ a - b = 8 :=
  sorry

end value_of_a_minus_b_l192_192145


namespace sample_group_b_correct_l192_192150

noncomputable def stratified_sample_group_b (total_cities: ℕ) (group_b_cities: ℕ) (sample_size: ℕ) : ℕ :=
  (sample_size * group_b_cities) / total_cities

theorem sample_group_b_correct : stratified_sample_group_b 36 12 12 = 4 := by
  sorry

end sample_group_b_correct_l192_192150


namespace four_is_square_root_of_sixteen_l192_192746

theorem four_is_square_root_of_sixteen : (4 : ℝ) * (4 : ℝ) = 16 :=
by
  sorry

end four_is_square_root_of_sixteen_l192_192746


namespace relationship_a_b_c_l192_192025

open Real

theorem relationship_a_b_c (x : ℝ) (hx1 : e < x) (hx2 : x < e^2)
  (a : ℝ) (ha : a = log x)
  (b : ℝ) (hb : b = (1 / 2) ^ log x)
  (c : ℝ) (hc : c = exp (log x)) :
  c > a ∧ a > b :=
by {
  -- we state the theorem without providing the proof for now
  sorry
}

end relationship_a_b_c_l192_192025


namespace factor_of_quadratic_implies_m_value_l192_192475

theorem factor_of_quadratic_implies_m_value (m : ℤ) : (∀ x : ℤ, (x + 6) ∣ (x^2 - m * x - 42)) → m = 1 := by
  sorry

end factor_of_quadratic_implies_m_value_l192_192475


namespace project_B_days_l192_192880

theorem project_B_days (B : ℕ) : 
  (1 / 20 + 1 / B) * 10 + (1 / B) * 5 = 1 -> B = 30 :=
by
  sorry

end project_B_days_l192_192880


namespace valid_license_plates_count_l192_192504

def num_valid_license_plates := (26 ^ 3) * (10 ^ 4)

theorem valid_license_plates_count : num_valid_license_plates = 175760000 :=
by
  sorry

end valid_license_plates_count_l192_192504


namespace gcd_problem_l192_192490

def a : ℕ := 101^5 + 1
def b : ℕ := 101^5 + 101^3 + 1

theorem gcd_problem : Nat.gcd a b = 1 := by
  sorry

end gcd_problem_l192_192490


namespace compute_x_squared_y_plus_x_y_squared_l192_192421

open Real

theorem compute_x_squared_y_plus_x_y_squared (x y : ℝ) 
  (h1 : (1/x) + (1/y) = 5) 
  (h2 : x * y + 2 * x + 2 * y = 7) : 
  x^2 * y + x * y^2 = 245 / 121 := 
by 
  sorry

end compute_x_squared_y_plus_x_y_squared_l192_192421


namespace no_solution_for_k_eq_4_l192_192048

theorem no_solution_for_k_eq_4 (x k : ℝ) (h₁ : x ≠ 4) (h₂ : x ≠ 8) : (k = 4) → ¬ ((x - 3) * (x - 8) = (x - k) * (x - 4)) :=
by
  sorry

end no_solution_for_k_eq_4_l192_192048


namespace wheat_field_problem_l192_192623

def equations (x F : ℕ) :=
  (6 * x - 300 = F) ∧ (5 * x + 200 = F)

theorem wheat_field_problem :
  ∃ (x F : ℕ), equations x F ∧ x = 500 ∧ F = 2700 :=
by
  sorry

end wheat_field_problem_l192_192623


namespace contradiction_in_stock_price_l192_192868

noncomputable def stock_price_contradiction : Prop :=
  ∃ (P D : ℝ), (D = 0.20 * P) ∧ (0.10 = (D / P) * 100)

theorem contradiction_in_stock_price : ¬(stock_price_contradiction) := sorry

end contradiction_in_stock_price_l192_192868


namespace arithmetic_sequence_inequality_l192_192339

theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : ∀ n : ℕ, n ≥ 2 → 2 * a n = a (n - 1) + a (n + 1)) :
  a 2 * a 4 ≤ a 3 ^ 2 :=
sorry

end arithmetic_sequence_inequality_l192_192339


namespace find_b_l192_192780

theorem find_b
  (a b c : ℤ)
  (h1 : a + 5 = b)
  (h2 : 5 + b = c)
  (h3 : b + c = a) : b = -10 :=
by
  sorry

end find_b_l192_192780


namespace arc_length_of_polar_curve_l192_192435

noncomputable def arc_length (f : ℝ → ℝ) (df : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt ((f x)^2 + (df x)^2)

theorem arc_length_of_polar_curve :
  arc_length (λ φ => 3 * (1 + Real.sin φ)) (λ φ => 3 * Real.cos φ) (-Real.pi / 6) 0 = 
  6 * (Real.sqrt 3 - Real.sqrt 2) :=
by
  sorry -- Proof goes here

end arc_length_of_polar_curve_l192_192435


namespace combined_soldiers_correct_l192_192103

-- Define the parameters for the problem
def interval : ℕ := 5
def wall_length : ℕ := 7300
def soldiers_per_tower : ℕ := 2

-- Calculate the number of towers and the total number of soldiers
def num_towers : ℕ := wall_length / interval
def combined_soldiers : ℕ := num_towers * soldiers_per_tower

-- Prove that the combined number of soldiers is as expected
theorem combined_soldiers_correct : combined_soldiers = 2920 := 
by
  sorry

end combined_soldiers_correct_l192_192103


namespace sin_of_7pi_over_6_l192_192054

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  sorry

end sin_of_7pi_over_6_l192_192054


namespace difference_between_eights_l192_192028

theorem difference_between_eights (value_tenths : ℝ) (value_hundredths : ℝ) (h1 : value_tenths = 0.8) (h2 : value_hundredths = 0.08) : 
  value_tenths - value_hundredths = 0.72 :=
by 
  sorry

end difference_between_eights_l192_192028


namespace find_extrema_l192_192423

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

theorem find_extrema :
  (∀ x, f x ≤ 17) ∧ (∃ x, f x = 17) ∧ (∀ x, f x ≥ -47) ∧ (∃ x, f x = -47) :=
by
  sorry

end find_extrema_l192_192423


namespace transformed_cubic_polynomial_l192_192950

theorem transformed_cubic_polynomial (x z : ℂ) 
    (h1 : z = x + x⁻¹) (h2 : x^3 - 3 * x^2 + x + 2 = 0) : 
    x^2 * (z^2 - z - 1) + 3 = 0 :=
sorry

end transformed_cubic_polynomial_l192_192950


namespace five_digit_palindromes_count_l192_192846

theorem five_digit_palindromes_count : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9) → 
  900 = 9 * 10 * 10 := 
by
  intro h
  sorry

end five_digit_palindromes_count_l192_192846


namespace multiple_of_savings_l192_192282

theorem multiple_of_savings (P : ℝ) (h : P > 0) :
  let monthly_savings := (1 / 4) * P
  let monthly_non_savings := (3 / 4) * P
  let total_yearly_savings := 12 * monthly_savings
  ∃ M : ℝ, total_yearly_savings = M * monthly_non_savings ∧ M = 4 := 
by
  sorry

end multiple_of_savings_l192_192282


namespace claire_balloon_count_l192_192949

variable (start_balloons lost_balloons initial_give_away more_give_away final_balloons grabbed_from_coworker : ℕ)

theorem claire_balloon_count (h1 : start_balloons = 50)
                           (h2 : lost_balloons = 12)
                           (h3 : initial_give_away = 1)
                           (h4 : more_give_away = 9)
                           (h5 : final_balloons = 39)
                           (h6 : start_balloons - initial_give_away - lost_balloons - more_give_away + grabbed_from_coworker = final_balloons) :
                           grabbed_from_coworker = 11 :=
by
  sorry

end claire_balloon_count_l192_192949


namespace K9_le_89_K9_example_171_l192_192020

section weights_proof

def K (n : ℕ) (P : ℕ) : ℕ := sorry -- Assume the definition of K given by the problem

theorem K9_le_89 : ∀ P, K 9 P ≤ 89 := by
  sorry -- Proof to be filled

def example_weight : ℕ := 171

theorem K9_example_171 : K 9 example_weight = 89 := by
  sorry -- Proof to be filled

end weights_proof

end K9_le_89_K9_example_171_l192_192020


namespace ray_steps_problem_l192_192220

theorem ray_steps_problem : ∃ n, n > 15 ∧ n % 3 = 2 ∧ n % 7 = 1 ∧ n % 4 = 3 ∧ n = 71 :=
by
  sorry

end ray_steps_problem_l192_192220


namespace no_intersection_of_graphs_l192_192477

theorem no_intersection_of_graphs :
  ∃ x y : ℝ, y = |3 * x + 6| ∧ y = -|4 * x - 3| → false := by
  sorry

end no_intersection_of_graphs_l192_192477


namespace divisibility_theorem_l192_192737

theorem divisibility_theorem {a m x n : ℕ} : (m ∣ n) ↔ (x^m - a^m ∣ x^n - a^n) :=
by
  sorry

end divisibility_theorem_l192_192737


namespace five_n_plus_3_composite_l192_192657

theorem five_n_plus_3_composite (n : ℕ)
  (h1 : ∃ k : ℤ, 2 * n + 1 = k^2)
  (h2 : ∃ m : ℤ, 3 * n + 1 = m^2) :
  ¬ Prime (5 * n + 3) :=
by
  sorry

end five_n_plus_3_composite_l192_192657


namespace even_sum_probability_l192_192612

-- Define the probabilities for the first wheel
def prob_even_first_wheel : ℚ := 3 / 6
def prob_odd_first_wheel : ℚ := 3 / 6

-- Define the probabilities for the second wheel
def prob_even_second_wheel : ℚ := 3 / 4
def prob_odd_second_wheel : ℚ := 1 / 4

-- Probability that the sum of the two selected numbers is even
def prob_even_sum : ℚ :=
  (prob_even_first_wheel * prob_even_second_wheel) +
  (prob_odd_first_wheel * prob_odd_second_wheel)

-- The theorem to prove
theorem even_sum_probability : prob_even_sum = 13 / 24 := by
  sorry

end even_sum_probability_l192_192612


namespace vec_parallel_l192_192466

variable {R : Type*} [LinearOrderedField R]

def is_parallel (a b : R × R) : Prop :=
  ∃ k : R, a = (k * b.1, k * b.2)

theorem vec_parallel {x : R} : 
  is_parallel (1, x) (-3, 4) ↔ x = -4/3 := by
  sorry

end vec_parallel_l192_192466


namespace channels_taken_away_l192_192562

theorem channels_taken_away (X : ℕ) : 
  (150 - X + 12 - 10 + 8 + 7 = 147) -> X = 20 :=
by
  sorry

end channels_taken_away_l192_192562


namespace food_insufficiency_l192_192633

-- Given conditions
def number_of_dogs : ℕ := 5
def food_per_meal : ℚ := 3 / 4
def meals_per_day : ℕ := 3
def initial_food : ℚ := 45
def days_in_two_weeks : ℕ := 14

-- Definitions derived from conditions
def daily_food_per_dog : ℚ := food_per_meal * meals_per_day
def daily_food_for_all_dogs : ℚ := daily_food_per_dog * number_of_dogs
def total_food_in_two_weeks : ℚ := daily_food_for_all_dogs * days_in_two_weeks

-- Proof statement: proving the food consumed exceeds the initial amount
theorem food_insufficiency : total_food_in_two_weeks > initial_food :=
by {
  sorry
}

end food_insufficiency_l192_192633


namespace pedal_triangle_angle_pedal_triangle_angle_equality_l192_192432

variables {A B C T_A T_B T_C: Type*}
variables {α β γ : Real}
variables {triangle : ∀ (A B C : Type*) (α β γ : Real), α ≤ β ∧ β ≤ γ ∧ γ < 90}

theorem pedal_triangle_angle
  (h : α ≤ β ∧ β ≤ γ ∧ γ < 90)
  (angles : 180 - 2 * α ≥ γ) :
  true :=
sorry

theorem pedal_triangle_angle_equality
  (h : α = β)
  (angles : (45 < α ∧ α = β ∧ α ≤ 60) ∧ (60 ≤ γ ∧ γ < 90)) :
  true :=
sorry

end pedal_triangle_angle_pedal_triangle_angle_equality_l192_192432


namespace limit_exists_implies_d_eq_zero_l192_192859

variable (a₁ d : ℝ) (S : ℕ → ℝ)

noncomputable def limExists := ∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (S n - L) < ε

def is_sum_of_arithmetic_sequence (S : ℕ → ℝ) (a₁ d : ℝ) :=
  ∀ n : ℕ, S n = (a₁ * n + d * (n * (n - 1) / 2))

theorem limit_exists_implies_d_eq_zero (h₁ : ∀ n : ℕ, n > 0 → S n = (a₁ * n + d * (n * (n - 1) / 2))) :
  limExists S → d = 0 :=
by sorry

end limit_exists_implies_d_eq_zero_l192_192859


namespace geometric_sequence_ratios_l192_192618

theorem geometric_sequence_ratios {n : ℕ} {r : ℝ}
  (h1 : 85 = (1 - r^(2*n)) / (1 - r^2))
  (h2 : 170 = r * 85) :
  r = 2 ∧ 2*n = 8 :=
by
  sorry

end geometric_sequence_ratios_l192_192618


namespace base6_problem_l192_192958

theorem base6_problem
  (x y : ℕ)
  (h1 : 453 = 2 * x * 10 + y) -- Constraint from base-6 to base-10 conversion
  (h2 : 0 ≤ x ∧ x ≤ 9) -- x is a base-10 digit
  (h3 : 0 ≤ y ∧ y ≤ 9) -- y is a base-10 digit
  (h4 : 4 * 6^2 + 5 * 6 + 3 = 177) -- Conversion result for 453_6
  (h5 : 2 * x * 10 + y = 177) -- Conversion from condition
  (hx : x = 7) -- x value from solution
  (hy : y = 7) -- y value from solution
  : (x * y) / 10 = 49 / 10 := 
by 
  sorry

end base6_problem_l192_192958


namespace profit_per_piece_correct_sales_volume_correct_maximum_monthly_profit_optimum_selling_price_is_130_l192_192175

-- Define the selling price and cost price
def cost_price : ℝ := 60
def sales_price (x : ℝ) := x

-- 1. Prove the profit per piece
def profit_per_piece (x : ℝ) : ℝ := sales_price x - cost_price

theorem profit_per_piece_correct (x : ℝ) : profit_per_piece x = x - 60 :=
by 
  -- it follows directly from the definition of profit_per_piece
  sorry

-- 2. Define the linear function relationship between monthly sales volume and selling price
def sales_volume (x : ℝ) : ℝ := -2 * x + 400

theorem sales_volume_correct (x : ℝ) : sales_volume x = -2 * x + 400 :=
by 
  -- it follows directly from the definition of sales_volume
  sorry

-- 3. Define the monthly profit and prove the maximized profit
def monthly_profit (x : ℝ) : ℝ := profit_per_piece x * sales_volume x

theorem maximum_monthly_profit (x : ℝ) : 
  monthly_profit x = -2 * x^2 + 520 * x - 24000 :=
by 
  -- it follows directly from the definition of monthly_profit
  sorry

theorem optimum_selling_price_is_130 : ∃ (x : ℝ), (monthly_profit x = 9800) ∧ (x = 130) :=
by
  -- solve this using the properties of quadratic functions
  sorry

end profit_per_piece_correct_sales_volume_correct_maximum_monthly_profit_optimum_selling_price_is_130_l192_192175


namespace opposite_of_neg_2023_l192_192881

/-- Define the opposite of a number -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove the opposite of -2023 is 2023 -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l192_192881


namespace length_of_first_train_l192_192972

noncomputable def length_first_train
  (speed_train1_kmh : ℝ)
  (speed_train2_kmh : ℝ)
  (time_sec : ℝ)
  (length_train2_m : ℝ) : ℝ :=
  let relative_speed_mps := (speed_train1_kmh + speed_train2_kmh) * (1000 / 3600)
  let total_distance_m := relative_speed_mps * time_sec
  total_distance_m - length_train2_m

theorem length_of_first_train :
  length_first_train 80 65 7.82006405004841 165 = 150.106201 :=
  by
  -- Proof steps would go here.
  sorry

end length_of_first_train_l192_192972


namespace avg_price_of_pen_l192_192458

theorem avg_price_of_pen 
  (total_pens : ℕ) (total_pencils : ℕ) (total_cost : ℕ) 
  (avg_price_pencil : ℕ) (total_pens_cost : ℕ) (total_pencils_cost : ℕ)
  (total_cost_eq : total_cost = total_pens_cost + total_pencils_cost)
  (total_pencils_cost_eq : total_pencils_cost = total_pencils * avg_price_pencil)
  (pencils_count : total_pencils = 75) (pens_count : total_pens = 30) 
  (avg_price_pencil_eq : avg_price_pencil = 2)
  (total_cost_eq' : total_cost = 450) :
  total_pens_cost / total_pens = 10 :=
by
  sorry

end avg_price_of_pen_l192_192458


namespace find_value_of_a_l192_192079

theorem find_value_of_a 
  (P : ℝ × ℝ)
  (a : ℝ)
  (α : ℝ)
  (point_on_terminal_side : P = (-4, a))
  (sin_cos_condition : Real.sin α * Real.cos α = Real.sqrt 3 / 4) : 
  a = -4 * Real.sqrt 3 ∨ a = - (4 * Real.sqrt 3 / 3) :=
sorry

end find_value_of_a_l192_192079


namespace aquarium_length_l192_192120

theorem aquarium_length {L : ℝ} (W H : ℝ) (final_volume : ℝ)
  (hW : W = 6) (hH : H = 3) (h_final_volume : final_volume = 54)
  (h_volume_relation : final_volume = 3 * (1/4 * L * W * H)) :
  L = 4 := by
  -- Mathematically translate the problem given conditions and resulting in L = 4.
  sorry

end aquarium_length_l192_192120


namespace new_paint_intensity_l192_192590

-- Definition of the given conditions
def original_paint_intensity : ℝ := 0.15
def replacement_paint_intensity : ℝ := 0.25
def fraction_replaced : ℝ := 1.5
def original_volume : ℝ := 100

-- Proof statement
theorem new_paint_intensity :
  (original_volume * original_paint_intensity + original_volume * fraction_replaced * replacement_paint_intensity) /
  (original_volume + original_volume * fraction_replaced) = 0.21 :=
by
  sorry

end new_paint_intensity_l192_192590


namespace evaluate_polynomial_at_2_l192_192806

def polynomial (x : ℕ) : ℕ := 3 * x^4 + x^3 + 2 * x^2 + x + 4

def horner_method (x : ℕ) : ℕ :=
  let v_0 := x
  let v_1 := 3 * v_0 + 1
  let v_2 := v_1 * v_0 + 2
  v_2

theorem evaluate_polynomial_at_2 :
  horner_method 2 = 16 :=
by
  sorry

end evaluate_polynomial_at_2_l192_192806


namespace line_does_not_pass_through_second_quadrant_l192_192329
-- Import the Mathlib library

-- Define the properties of the line
def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the condition for a point to be in the second quadrant:
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the proof statement
theorem line_does_not_pass_through_second_quadrant:
  ∀ x y : ℝ, line_eq x y → ¬ in_second_quadrant x y :=
by
  sorry

end line_does_not_pass_through_second_quadrant_l192_192329


namespace expression_value_l192_192140

theorem expression_value (x y : ℝ) (h : x - y = 1) :
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 :=
by
  sorry

end expression_value_l192_192140


namespace hired_waiters_l192_192533

theorem hired_waiters (W H : Nat) (hcooks : Nat := 9) 
                      (initial_ratio : 3 * W = 11 * hcooks)
                      (new_ratio : 9 = 5 * (W + H)) 
                      (original_waiters : W = 33) 
                      : H = 12 :=
by
  sorry

end hired_waiters_l192_192533


namespace points_distance_le_sqrt5_l192_192638

theorem points_distance_le_sqrt5 :
  ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (0 ≤ (points i).1 ∧ (points i).1 ≤ 4) ∧ (0 ≤ (points i).2 ∧ (points i).2 ≤ 3)) →
  ∃ (i j : Fin 6), i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end points_distance_le_sqrt5_l192_192638


namespace inequality_and_equality_condition_l192_192324

theorem inequality_and_equality_condition (x : ℝ) (hx : x > 0) : 
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) :=
  sorry

end inequality_and_equality_condition_l192_192324


namespace max_reflections_l192_192850

theorem max_reflections (P Q R M : Type) (angle : ℝ) :
  0 < angle ∧ angle ≤ 30 ∧ (∃ n : ℕ, 10 * n = angle) →
  ∃ n : ℕ, n ≤ 3 :=
by
  sorry

end max_reflections_l192_192850


namespace combinations_of_coins_l192_192263

theorem combinations_of_coins (p n d : ℕ) (h₁ : p ≥ 0) (h₂ : n ≥ 0) (h₃ : d ≥ 0) 
  (value_eq : p + 5 * n + 10 * d = 25) : 
  ∃! c : ℕ, c = 12 :=
sorry

end combinations_of_coins_l192_192263


namespace dog_bones_remaining_l192_192735

noncomputable def initial_bones : ℕ := 350
noncomputable def factor : ℕ := 9
noncomputable def found_bones : ℕ := factor * initial_bones
noncomputable def total_bones : ℕ := initial_bones + found_bones
noncomputable def bones_given_away : ℕ := 120
noncomputable def bones_remaining : ℕ := total_bones - bones_given_away

theorem dog_bones_remaining : bones_remaining = 3380 :=
by
  sorry

end dog_bones_remaining_l192_192735


namespace frustum_has_only_two_parallel_surfaces_l192_192486

-- Definitions for the geometric bodies in terms of their properties
structure Pyramid where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 0

structure Prism where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 6

structure Frustum where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 2

structure Cuboid where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 6

-- The main theorem stating that the Frustum is the one with exactly two parallel surfaces.
theorem frustum_has_only_two_parallel_surfaces (pyramid : Pyramid) (prism : Prism) (frustum : Frustum) (cuboid : Cuboid) :
  frustum.parallel_surfaces = 2 ∧
  pyramid.parallel_surfaces ≠ 2 ∧
  prism.parallel_surfaces ≠ 2 ∧
  cuboid.parallel_surfaces ≠ 2 :=
by
  sorry

end frustum_has_only_two_parallel_surfaces_l192_192486


namespace geometric_sequence_sum_l192_192999

theorem geometric_sequence_sum (a r : ℚ) (h_a : a = 1/3) (h_r : r = 1/2) (S_n : ℚ) (h_S_n : S_n = 80/243) : ∃ n : ℕ, S_n = a * ((1 - r^n) / (1 - r)) ∧ n = 4 := by
  sorry

end geometric_sequence_sum_l192_192999


namespace point_on_line_l192_192741

theorem point_on_line (A B C x₀ y₀ : ℝ) :
  (A * x₀ + B * y₀ + C = 0) ↔ (A * (x₀ - x₀) + B * (y₀ - y₀) = 0) :=
by 
  sorry

end point_on_line_l192_192741


namespace time_to_store_vaccine_l192_192992

def final_temp : ℤ := -24
def current_temp : ℤ := -4
def rate_of_change : ℤ := -5

theorem time_to_store_vaccine : 
  ∃ t : ℤ, current_temp + rate_of_change * t = final_temp ∧ t = 4 :=
by
  use 4
  sorry

end time_to_store_vaccine_l192_192992


namespace num_times_teams_face_each_other_l192_192785

-- Conditions
variable (teams games total_games : ℕ)
variable (k : ℕ)
variable (h1 : teams = 17)
variable (h2 : games = teams * (teams - 1) * k / 2)
variable (h3 : total_games = 1360)

-- Proof problem
theorem num_times_teams_face_each_other : k = 5 := 
by 
  sorry

end num_times_teams_face_each_other_l192_192785


namespace rectangle_y_value_l192_192472

theorem rectangle_y_value (y : ℝ) (h₁ : (-2, y) ≠ (10, y))
  (h₂ : (-2, -1) ≠ (10, -1))
  (h₃ : 12 * (y + 1) = 108)
  (y_pos : 0 < y) :
  y = 8 :=
by
  sorry

end rectangle_y_value_l192_192472


namespace small_triangle_count_l192_192090

theorem small_triangle_count (n : ℕ) (h : n = 2009) : (2 * n + 1) = 4019 := 
by {
    sorry
}

end small_triangle_count_l192_192090


namespace distance_to_second_picture_edge_l192_192209

/-- Given a wall of width 25 feet, with a first picture 5 feet wide centered on the wall,
and a second picture 3 feet wide centered in the remaining space, the distance 
from the nearest edge of the second picture to the end of the wall is 13.5 feet. -/
theorem distance_to_second_picture_edge :
  let wall_width := 25
  let first_picture_width := 5
  let second_picture_width := 3
  let side_space := (wall_width - first_picture_width) / 2
  let remaining_space := side_space
  let second_picture_side_space := (remaining_space - second_picture_width) / 2
  10 + 3.5 = 13.5 :=
by
  sorry

end distance_to_second_picture_edge_l192_192209


namespace yvette_sundae_cost_l192_192943

noncomputable def cost_friends : ℝ := 7.50 + 10.00 + 8.50
noncomputable def final_bill : ℝ := 42.00
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def tip_amount : ℝ := tip_percentage * final_bill

theorem yvette_sundae_cost : 
  final_bill - (cost_friends + tip_amount) = 7.60 := by
  sorry

end yvette_sundae_cost_l192_192943


namespace infinite_series_sum_zero_l192_192561

theorem infinite_series_sum_zero : ∑' n : ℕ, (3 * n + 4) / ((n + 1) * (n + 2) * (n + 3)) = 0 :=
by
  sorry

end infinite_series_sum_zero_l192_192561


namespace regular_polygons_from_cube_intersection_l192_192670

noncomputable def cube : Type := sorry  -- Define a 3D cube type
noncomputable def plane : Type := sorry  -- Define a plane type

-- Define what it means for a polygon to be regular (equilateral and equiangular)
def is_regular_polygon (polygon : Type) : Prop := sorry

-- Define a function that describes the intersection of a plane with a cube,
-- resulting in a polygon
noncomputable def intersection (c : cube) (p : plane) : Type := sorry

-- Define predicates for the specific regular polygons: triangle, quadrilateral, and hexagon
def is_triangle (polygon : Type) : Prop := sorry
def is_quadrilateral (polygon : Type) : Prop := sorry
def is_hexagon (polygon : Type) : Prop := sorry

-- Ensure these predicates imply regular polygons
axiom triangle_is_regular : ∀ (t : Type), is_triangle t → is_regular_polygon t
axiom quadrilateral_is_regular : ∀ (q : Type), is_quadrilateral q → is_regular_polygon q
axiom hexagon_is_regular : ∀ (h : Type), is_hexagon h → is_regular_polygon h

-- The main theorem statement
theorem regular_polygons_from_cube_intersection (c : cube) (p : plane) :
  is_regular_polygon (intersection c p) →
  is_triangle (intersection c p) ∨ is_quadrilateral (intersection c p) ∨ is_hexagon (intersection c p) :=
sorry

end regular_polygons_from_cube_intersection_l192_192670


namespace xy_product_l192_192286

theorem xy_product (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 44) : x * y = -24 := 
by {
  sorry
}

end xy_product_l192_192286


namespace solve_inequality_l192_192414

def p (x : ℝ) : ℝ := x^2 - 5*x + 3

theorem solve_inequality (x : ℝ) : 
  abs (p x) < 9 ↔ (-1 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) :=
sorry

end solve_inequality_l192_192414


namespace problem_1_problem_2_l192_192786

-- Definition of the operation ⊕
def my_oplus (a b : ℚ) : ℚ := (a + 3 * b) / 2

-- Prove that 4(2 ⊕ 5) = 34
theorem problem_1 : 4 * my_oplus 2 5 = 34 := 
by sorry

-- Definitions of A and B
def A (x y : ℚ) : ℚ := x^2 + 2 * x * y + y^2
def B (x y : ℚ) : ℚ := -2 * x * y + y^2

-- Prove that (A ⊕ B) + (B ⊕ A) = 2x^2 + 4y^2
theorem problem_2 (x y : ℚ) : 
  my_oplus (A x y) (B x y) + my_oplus (B x y) (A x y) = 2 * x^2 + 4 * y^2 := 
by sorry

end problem_1_problem_2_l192_192786


namespace C_share_per_rs_equals_l192_192359

-- Definitions based on given conditions
def A_share_per_rs (x : ℝ) : ℝ := x
def B_share_per_rs : ℝ := 0.65
def C_share : ℝ := 48
def total_sum : ℝ := 246

-- The target statement to prove
theorem C_share_per_rs_equals : C_share / total_sum = 0.195122 :=
by
  sorry

end C_share_per_rs_equals_l192_192359


namespace distinct_real_solutions_l192_192762

open Real Nat

noncomputable def p_n : ℕ → ℝ → ℝ 
| 0, x => x
| (n+1), x => (p_n n (x^2 - 2))

theorem distinct_real_solutions (n : ℕ) : 
  ∃ S : Finset ℝ, S.card = 2^n ∧ ∀ x ∈ S, p_n n x = x ∧ (∀ y ∈ S, x ≠ y → x ≠ y) := 
sorry

end distinct_real_solutions_l192_192762


namespace total_amount_paid_l192_192491

theorem total_amount_paid :
  let pizzas := 3
  let cost_per_pizza := 8
  let total_cost := pizzas * cost_per_pizza
  total_cost = 24 :=
by
  sorry

end total_amount_paid_l192_192491


namespace relationship_among_a_b_c_l192_192392

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.1 * Real.log 2)
noncomputable def c : ℝ := Real.exp (1.3 * Real.log 0.2)

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  have a_neg : a < 0 :=
    by sorry
  have b_pos : b > 1 :=
    by sorry
  have c_pos : c < 1 :=
    by sorry
  sorry

end relationship_among_a_b_c_l192_192392


namespace find_term_in_sequence_l192_192564

theorem find_term_in_sequence (n : ℕ) (k : ℕ) (term_2020: ℚ) : 
  (3^7 = 2187) → 
  (2020 : ℕ) / (2187 : ℕ) = term_2020 → 
  (term_2020 = 2020 / 2187) →
  (∃ (k : ℕ), k = 2020 ∧ (2 ≤ k ∧ k < 2187 ∧ k % 3 ≠ 0)) → 
  (2020 / 2187 = (1347 / 2187 : ℚ)) :=
by {
  sorry
}

end find_term_in_sequence_l192_192564


namespace elberta_has_22_dollars_l192_192231

theorem elberta_has_22_dollars (granny_smith : ℝ) (anjou : ℝ) (elberta : ℝ) 
  (h1 : granny_smith = 75) 
  (h2 : anjou = granny_smith / 4)
  (h3 : elberta = anjou + 3) : 
  elberta = 22 := 
by
  sorry

end elberta_has_22_dollars_l192_192231


namespace sum_and_count_l192_192126

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count (x y : ℕ) (hx : x = sum_of_integers 30 50) (hy : y = count_even_integers 30 50) :
  x + y = 851 :=
by
  -- proof goes here
  sorry

end sum_and_count_l192_192126


namespace value_of_f_5_l192_192086

theorem value_of_f_5 (f : ℕ → ℕ) (y : ℕ)
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 20) : f 5 = 62 :=
sorry

end value_of_f_5_l192_192086


namespace relationship_among_a_b_c_l192_192367

noncomputable def a : ℝ := (1 / 3) ^ 3
noncomputable def b (x : ℝ) : ℝ := x ^ 3
noncomputable def c (x : ℝ) : ℝ := Real.log x

theorem relationship_among_a_b_c (x : ℝ) (h : x > 2) : a < c x ∧ c x < b x :=
by {
  -- proof steps are skipped
  sorry
}

end relationship_among_a_b_c_l192_192367


namespace smallest_positive_integer_l192_192410

theorem smallest_positive_integer (a : ℤ)
  (h1 : a % 2 = 1)
  (h2 : a % 3 = 2)
  (h3 : a % 4 = 3)
  (h4 : a % 5 = 4) :
  a = 59 :=
sorry

end smallest_positive_integer_l192_192410


namespace profit_relationship_profit_range_max_profit_l192_192624

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

end profit_relationship_profit_range_max_profit_l192_192624


namespace expected_value_of_winnings_after_one_flip_l192_192722

-- Definitions based on conditions from part a)
def prob_heads : ℚ := 1 / 3
def prob_tails : ℚ := 2 / 3
def win_heads : ℚ := 3
def lose_tails : ℚ := -2

-- The statement to prove:
theorem expected_value_of_winnings_after_one_flip :
  prob_heads * win_heads + prob_tails * lose_tails = -1 / 3 :=
by
  sorry

end expected_value_of_winnings_after_one_flip_l192_192722


namespace solve_trig_eq_l192_192583

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  (x = (π / 12) + 2 * k * π ∨
   x = (7 * π / 12) + 2 * k * π ∨
   x = (7 * π / 6) + 2 * k * π ∨
   x = -(5 * π / 6) + 2 * k * π) →
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 :=
sorry

end solve_trig_eq_l192_192583


namespace smallest_multiple_of_18_all_digits_9_or_0_l192_192449

theorem smallest_multiple_of_18_all_digits_9_or_0 :
  ∃ (m : ℕ), (m > 0) ∧ (m % 18 = 0) ∧ (∀ d ∈ (m.digits 10), d = 9 ∨ d = 0) ∧ (m / 18 = 5) :=
sorry

end smallest_multiple_of_18_all_digits_9_or_0_l192_192449


namespace inverse_proportion_l192_192441

theorem inverse_proportion (x : ℝ) (y : ℝ) (f₁ f₂ f₃ f₄ : ℝ → ℝ) (h₁ : f₁ x = 2 * x) (h₂ : f₂ x = x / 2) (h₃ : f₃ x = 2 / x) (h₄ : f₄ x = 2 / (x - 1)) :
  f₃ x * x = 2 := sorry

end inverse_proportion_l192_192441


namespace max_correct_answers_l192_192092

theorem max_correct_answers (c w b : ℕ) (h1 : c + w + b = 30) (h2 : 4 * c - w = 85) : c ≤ 23 :=
  sorry

end max_correct_answers_l192_192092


namespace orthogonal_pairs_in_cube_is_36_l192_192018

-- Define a cube based on its properties, i.e., having vertices, edges, and faces.
structure Cube :=
(vertices : Fin 8 → Fin 3)
(edges : Fin 12 → (Fin 2 → Fin 8))
(faces : Fin 6 → (Fin 4 → Fin 8))

-- Define orthogonal pairs of a cube as an axiom.
axiom orthogonal_line_plane_pairs (c : Cube) : ℕ

-- The main theorem stating the problem's conclusion.
theorem orthogonal_pairs_in_cube_is_36 (c : Cube): orthogonal_line_plane_pairs c = 36 :=
by { sorry }

end orthogonal_pairs_in_cube_is_36_l192_192018


namespace least_number_remainder_l192_192027

theorem least_number_remainder (N k : ℕ) (h : N = 18 * k + 4) : N = 256 :=
by
  sorry

end least_number_remainder_l192_192027


namespace abc_equal_l192_192452

theorem abc_equal (a b c : ℝ) (h : a^2 + b^2 + c^2 - ab - bc - ac = 0) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l192_192452


namespace tan_diff_identity_l192_192399

theorem tan_diff_identity 
  (α : ℝ)
  (h : Real.tan α = -4/3) : Real.tan (α - Real.pi / 4) = 7 := 
sorry

end tan_diff_identity_l192_192399


namespace tank_loss_rate_after_first_repair_l192_192017

def initial_capacity : ℕ := 350000
def first_loss_rate : ℕ := 32000
def first_loss_duration : ℕ := 5
def second_loss_duration : ℕ := 10
def filling_rate : ℕ := 40000
def filling_duration : ℕ := 3
def missing_gallons : ℕ := 140000

noncomputable def first_repair_loss_rate := (initial_capacity - (first_loss_rate * first_loss_duration) + (filling_rate * filling_duration) - (initial_capacity - missing_gallons)) / second_loss_duration

theorem tank_loss_rate_after_first_repair : first_repair_loss_rate = 10000 := by sorry

end tank_loss_rate_after_first_repair_l192_192017


namespace geometric_seq_a4_l192_192848

theorem geometric_seq_a4 (a : ℕ → ℕ) (q : ℕ) (h_q : q = 2) 
  (h_a1a3 : a 0 * a 2 = 6 * a 1) : a 3 = 24 :=
by
  -- Skipped proof
  sorry

end geometric_seq_a4_l192_192848


namespace division_4073_by_38_l192_192798

theorem division_4073_by_38 :
  ∃ q r, 4073 = 38 * q + r ∧ 0 ≤ r ∧ r < 38 ∧ q = 107 ∧ r = 7 := by
  sorry

end division_4073_by_38_l192_192798


namespace logically_follows_l192_192550

-- Define the predicates P and Q
variables {Student : Type} {P Q : Student → Prop}

-- The given condition
axiom Turner_statement : ∀ (x : Student), P x → Q x

-- The statement that necessarily follows
theorem logically_follows : (∀ (x : Student), ¬ Q x → ¬ P x) :=
sorry

end logically_follows_l192_192550


namespace education_fund_growth_l192_192558

theorem education_fund_growth (x : ℝ) :
  2500 * (1 + x)^2 = 3600 :=
sorry

end education_fund_growth_l192_192558


namespace remainder_of_division_l192_192818

noncomputable def dividend : Polynomial ℤ := Polynomial.C 1 * Polynomial.X^4 +
                                             Polynomial.C 3 * Polynomial.X^2 + 
                                             Polynomial.C (-4)

noncomputable def divisor : Polynomial ℤ := Polynomial.C 1 * Polynomial.X^3 +
                                            Polynomial.C (-3)

theorem remainder_of_division :
  Polynomial.modByMonic dividend divisor = Polynomial.C 3 * Polynomial.X^2 +
                                            Polynomial.C 3 * Polynomial.X +
                                            Polynomial.C (-4) :=
by
  sorry

end remainder_of_division_l192_192818


namespace orange_harvest_exists_l192_192292

theorem orange_harvest_exists :
  ∃ (A B C D : ℕ), A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ A + B + C + D = 56 :=
by
  use 10
  use 15
  use 16
  use 15
  repeat {split};
  sorry

end orange_harvest_exists_l192_192292


namespace blocks_for_tower_l192_192630

theorem blocks_for_tower (total_blocks : ℕ) (house_blocks : ℕ) (extra_blocks : ℕ) (tower_blocks : ℕ) 
  (h1 : total_blocks = 95) 
  (h2 : house_blocks = 20) 
  (h3 : extra_blocks = 30) 
  (h4 : tower_blocks = house_blocks + extra_blocks) : 
  tower_blocks = 50 :=
sorry

end blocks_for_tower_l192_192630


namespace arithmetic_identity_l192_192342

theorem arithmetic_identity : 3 * 5 * 7 + 15 / 3 = 110 := by
  sorry

end arithmetic_identity_l192_192342


namespace find_positive_n_l192_192107

theorem find_positive_n (n x : ℝ) (h : 16 * x ^ 2 + n * x + 4 = 0) : n = 16 :=
by
  sorry

end find_positive_n_l192_192107


namespace investment_amount_l192_192944

noncomputable def calculate_principal (A : ℕ) (r t : ℝ) (n : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem investment_amount (A : ℕ) (r t : ℝ) (n P : ℕ) :
  A = 70000 → r = 0.08 → t = 5 → n = 12 →
  P = 46994 →
  calculate_principal A r t n = P :=
by
  intros hA hr ht hn hP
  rw [hA, hr, ht, hn, hP]
  sorry

end investment_amount_l192_192944


namespace min_k_intersects_circle_l192_192133

def circle_eq (x y : ℝ) := (x + 2)^2 + y^2 = 4
def line_eq (x y k : ℝ) := k * x - y - 2 * k = 0

theorem min_k_intersects_circle :
  (∀ k : ℝ, (∃ x y : ℝ, circle_eq x y ∧ line_eq x y k) → k ≥ - (Real.sqrt 3) / 3) :=
sorry

end min_k_intersects_circle_l192_192133


namespace find_extra_digit_l192_192454

theorem find_extra_digit (x y a : ℕ) (hx : x + y = 23456) (h10x : 10 * x + a + y = 55555) (ha : 0 ≤ a ∧ a ≤ 9) : a = 5 :=
by
  sorry

end find_extra_digit_l192_192454


namespace range_of_x_l192_192229

theorem range_of_x (x : ℝ) (h : ∃ y : ℝ, y = (x - 3) ∧ y > 0) : x > 3 :=
sorry

end range_of_x_l192_192229


namespace wendy_time_correct_l192_192442

variable (bonnie_time wendy_difference : ℝ)

theorem wendy_time_correct (h1 : bonnie_time = 7.80) (h2 : wendy_difference = 0.25) : 
  (bonnie_time - wendy_difference = 7.55) :=
by
  sorry

end wendy_time_correct_l192_192442


namespace sum_first_n_terms_arithmetic_sequence_l192_192275

theorem sum_first_n_terms_arithmetic_sequence 
  (S : ℕ → ℕ) (m : ℕ) (h1 : S m = 2) (h2 : S (2 * m) = 10) :
  S (3 * m) = 24 :=
sorry

end sum_first_n_terms_arithmetic_sequence_l192_192275


namespace balance_force_l192_192864

structure Vector2D where
  x : ℝ
  y : ℝ

def F1 : Vector2D := ⟨1, 1⟩
def F2 : Vector2D := ⟨2, 3⟩

def vector_add (a b : Vector2D) : Vector2D := ⟨a.x + b.x, a.y + b.y⟩
def vector_neg (a : Vector2D) : Vector2D := ⟨-a.x, -a.y⟩

theorem balance_force : 
  ∃ F3 : Vector2D, vector_add (vector_add F1 F2) F3 = ⟨0, 0⟩ ∧ F3 = ⟨-3, -4⟩ := 
by
  sorry

end balance_force_l192_192864


namespace total_fish_count_l192_192316

theorem total_fish_count (kyle_caught_same_as_tasha : ∀ kyle tasha : ℕ, kyle = tasha) 
  (carla_caught : ℕ) (kyle_caught : ℕ) (tasha_caught : ℕ)
  (h0 : carla_caught = 8) (h1 : kyle_caught = 14) (h2 : tasha_caught = kyle_caught) : 
  8 + 14 + 14 = 36 :=
by sorry

end total_fish_count_l192_192316


namespace calculate_hardcover_volumes_l192_192993

theorem calculate_hardcover_volumes (h p : ℕ) 
  (h_total_volumes : h + p = 12)
  (h_cost_equation : 27 * h + 16 * p = 284)
  (h_p_relation : p = 12 - h) : h = 8 :=
by
  sorry

end calculate_hardcover_volumes_l192_192993


namespace flowers_left_l192_192068

theorem flowers_left (flowers_picked_A : Nat) (flowers_picked_M : Nat) (flowers_given : Nat)
  (h_a : flowers_picked_A = 16)
  (h_m : flowers_picked_M = 16)
  (h_g : flowers_given = 18) :
  flowers_picked_A + flowers_picked_M - flowers_given = 14 :=
by
  sorry

end flowers_left_l192_192068


namespace angle_BDC_is_55_l192_192534

def right_triangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] : Prop :=
  ∃ (angle_A angle_B angle_C : ℝ), angle_A + angle_B + angle_C = 180 ∧
  angle_A = 20 ∧ angle_C = 90

def bisector (B D : Type) [Inhabited B] [Inhabited D] (angle_ABC : ℝ) : Prop :=
  ∃ (angle_DBC : ℝ), angle_DBC = angle_ABC / 2

theorem angle_BDC_is_55 (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] :
  right_triangle A B C →
  bisector B D 70 →
  ∃ angle_BDC : ℝ, angle_BDC = 55 :=
by sorry

end angle_BDC_is_55_l192_192534


namespace metallic_sheet_dimension_l192_192393

theorem metallic_sheet_dimension (x : ℝ) (h₁ : ∀ (l w h : ℝ), l = x - 8 → w = 28 → h = 4 → l * w * h = 4480) : x = 48 :=
sorry

end metallic_sheet_dimension_l192_192393


namespace range_of_k_l192_192599

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ 0 ≤ k ∧ k < 4 := sorry

end range_of_k_l192_192599


namespace max_ab_l192_192966

theorem max_ab (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ∃ (M : ℝ), M = 1 / 8 ∧ ∀ (a b : ℝ), (a + 2 * b = 1) → 0 < a → 0 < b → ab ≤ M :=
sorry

end max_ab_l192_192966


namespace middle_school_mentoring_l192_192529

theorem middle_school_mentoring (s n : ℕ) (h1 : s ≠ 0) (h2 : n ≠ 0) 
  (h3 : (n : ℚ) / 3 = (2 : ℚ) * (s : ℚ) / 5) : 
  (n / 3 + 2 * s / 5) / (n + s) = 4 / 11 := by
  sorry

end middle_school_mentoring_l192_192529


namespace triangle_side_relation_l192_192896

theorem triangle_side_relation
  (A B C : ℝ)
  (a b c : ℝ)
  (h : 3 * (Real.sin (A / 2)) * (Real.sin (B / 2)) * (Real.cos (C / 2)) + (Real.sin (3 * A / 2)) * (Real.sin (3 * B / 2)) * (Real.cos (3 * C / 2)) = 0)
  (law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
  a^3 + b^3 = c^3 :=
by
  sorry

end triangle_side_relation_l192_192896


namespace identity_holds_for_all_real_numbers_l192_192667

theorem identity_holds_for_all_real_numbers (a b : ℝ) : 
  a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by sorry

end identity_holds_for_all_real_numbers_l192_192667


namespace ratio_of_areas_of_similar_triangles_l192_192530

theorem ratio_of_areas_of_similar_triangles (a b a1 b1 S S1 : ℝ) (α k : ℝ) :
  S = (1/2) * a * b * (Real.sin α) →
  S1 = (1/2) * a1 * b1 * (Real.sin α) →
  a1 = k * a →
  b1 = k * b →
  S1 / S = k^2 := by
  intros h1 h2 h3 h4
  sorry

end ratio_of_areas_of_similar_triangles_l192_192530


namespace angle_of_inclination_range_l192_192709

noncomputable def curve (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

noncomputable def tangent_slope (x : ℝ) : ℝ := 
  -4 * Real.exp x / (Real.exp x + 1) ^ 2

theorem angle_of_inclination_range (x : ℝ) (a : ℝ) 
  (hx : tangent_slope x = Real.tan a) : 
  (3 * Real.pi / 4 ≤ a ∧ a < Real.pi) :=
by 
  sorry

end angle_of_inclination_range_l192_192709


namespace common_ratio_geometric_series_l192_192171

theorem common_ratio_geometric_series (a r S : ℝ) (h₁ : S = a / (1 - r))
  (h₂ : r ≠ 1)
  (h₃ : r^4 * S = S / 81) :
  r = 1/3 :=
by 
  sorry

end common_ratio_geometric_series_l192_192171


namespace checker_moves_10_cells_l192_192796

theorem checker_moves_10_cells :
  ∃ (a : ℕ → ℕ), a 1 = 1 ∧ a 2 = 2 ∧ (∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) ∧ a 10 = 89 :=
by
  -- mathematical proof goes here
  sorry

end checker_moves_10_cells_l192_192796


namespace fraction_Renz_Miles_l192_192955

-- Given definitions and conditions
def Mitch_macarons : ℕ := 20
def Joshua_diff : ℕ := 6
def kids : ℕ := 68
def macarons_per_kid : ℕ := 2
def total_macarons_given : ℕ := kids * macarons_per_kid
def Joshua_macarons : ℕ := Mitch_macarons + Joshua_diff
def Miles_macarons : ℕ := 2 * Joshua_macarons
def Mitch_Joshua_Miles_macarons : ℕ := Mitch_macarons + Joshua_macarons + Miles_macarons
def Renz_macarons : ℕ := total_macarons_given - Mitch_Joshua_Miles_macarons

-- The theorem to prove
theorem fraction_Renz_Miles : (Renz_macarons : ℚ) / (Miles_macarons : ℚ) = 19 / 26 :=
by
  sorry

end fraction_Renz_Miles_l192_192955


namespace find_x_given_sin_interval_l192_192644

open Real

theorem find_x_given_sin_interval (x : ℝ) (h1 : sin x = -3 / 5) (h2 : π < x ∧ x < 3 / 2 * π) :
  x = π + arcsin (3 / 5) :=
sorry

end find_x_given_sin_interval_l192_192644


namespace domain_of_f_l192_192118

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (1 - x^2)) + x^0

theorem domain_of_f :
  {x : ℝ | 1 - x^2 > 0 ∧ x ≠ 0} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end domain_of_f_l192_192118


namespace area_of_right_triangle_integers_l192_192724

theorem area_of_right_triangle_integers (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ (A : ℤ), A = (a * b) / 2 := 
sorry

end area_of_right_triangle_integers_l192_192724


namespace find_x_value_l192_192807

theorem find_x_value : ∃ x : ℝ, 35 - (23 - (15 - x)) = 12 * 2 / 1 / 2 → x = -21 :=
by
  sorry

end find_x_value_l192_192807


namespace grocery_packs_l192_192976

theorem grocery_packs (cookie_packs cake_packs : ℕ)
  (h1 : cookie_packs = 23)
  (h2 : cake_packs = 4) :
  cookie_packs + cake_packs = 27 :=
by
  sorry

end grocery_packs_l192_192976


namespace annual_rent_per_sqft_l192_192264

theorem annual_rent_per_sqft
  (length width monthly_rent : ℕ)
  (H_length : length = 10)
  (H_width : width = 8)
  (H_monthly_rent : monthly_rent = 2400) :
  (12 * monthly_rent) / (length * width) = 360 := by
  sorry

end annual_rent_per_sqft_l192_192264


namespace arithmetic_sequence_sum_l192_192122

/-
The sum of the first 20 terms of the arithmetic sequence 8, 5, 2, ... is -410.
-/

theorem arithmetic_sequence_sum :
  let a : ℤ := 8
  let d : ℤ := -3
  let n : ℤ := 20
  let S_n : ℤ := n * a + (d * n * (n - 1)) / 2
  S_n = -410 := by
  sorry

end arithmetic_sequence_sum_l192_192122


namespace yuan_to_scientific_notation_l192_192035

/-- Express 2.175 billion yuan in scientific notation,
preserving three significant figures. --/
theorem yuan_to_scientific_notation (a : ℝ) (h : a = 2.175 * 10^9) : a = 2.18 * 10^9 :=
sorry

end yuan_to_scientific_notation_l192_192035


namespace union_intersection_l192_192827

-- Define the sets A, B, and C
def A : Set ℕ := {1, 2, 6}
def B : Set ℕ := {2, 4}
def C : Set ℕ := {1, 2, 3, 4}

-- The theorem stating that (A ∪ B) ∩ C = {1, 2, 4}
theorem union_intersection : (A ∪ B) ∩ C = {1, 2, 4} := sorry

end union_intersection_l192_192827


namespace find_cost_price_l192_192064

theorem find_cost_price (C : ℝ) (h1 : 1.12 * C + 18 = 1.18 * C) : C = 300 :=
by
  sorry

end find_cost_price_l192_192064


namespace stationery_problem_l192_192930

variables (S E : ℕ)

theorem stationery_problem
  (h1 : S - E = 30)
  (h2 : 4 * E = S) :
  S = 40 :=
by
  sorry

end stationery_problem_l192_192930


namespace total_cost_correct_l192_192938

-- Condition C1: There are 13 hearts in a deck of 52 playing cards. 
def hearts_in_deck : ℕ := 13

-- Condition C2: The number of cows is twice the number of hearts.
def cows_in_Devonshire : ℕ := 2 * hearts_in_deck

-- Condition C3: Each cow is sold at $200.
def cost_per_cow : ℕ := 200

-- Question Q1: Calculate the total cost of the cows.
def total_cost_of_cows : ℕ := cows_in_Devonshire * cost_per_cow

-- Final statement we need to prove
theorem total_cost_correct : total_cost_of_cows = 5200 := by
  -- This will be proven in the proof body
  sorry

end total_cost_correct_l192_192938


namespace volume_of_cuboid_l192_192747

theorem volume_of_cuboid (l w h : ℝ) (hlw: l * w = 120) (hwh: w * h = 72) (hhl: h * l = 60) : l * w * h = 720 :=
  sorry

end volume_of_cuboid_l192_192747


namespace triploid_fruit_fly_chromosome_periodicity_l192_192291

-- Define the conditions
def normal_chromosome_count (organism: Type) : ℕ := 8
def triploid_fruit_fly (organism: Type) : Prop := true
def XXY_sex_chromosome_composition (organism: Type) : Prop := true
def periodic_change (counts: List ℕ) : Prop := counts = [9, 18, 9]

-- State the theorem
theorem triploid_fruit_fly_chromosome_periodicity (organism: Type)
  (h1: triploid_fruit_fly organism) 
  (h2: XXY_sex_chromosome_composition organism)
  (h3: normal_chromosome_count organism = 8) : 
  periodic_change [9, 18, 9] :=
sorry

end triploid_fruit_fly_chromosome_periodicity_l192_192291


namespace find_width_l192_192855

theorem find_width (A : ℕ) (hA : A ≥ 120) (w : ℕ) (l : ℕ) (hl : l = w + 20) (h_area : w * l = A) : w = 4 :=
by sorry

end find_width_l192_192855


namespace blue_pill_cost_correct_l192_192532

-- Defining the conditions
def num_days : Nat := 21
def total_cost : Nat := 672
def red_pill_cost (blue_pill_cost : Nat) : Nat := blue_pill_cost - 2
def daily_cost (blue_pill_cost : Nat) : Nat := blue_pill_cost + red_pill_cost blue_pill_cost

-- The statement to prove
theorem blue_pill_cost_correct : ∃ (y : Nat), daily_cost y * num_days = total_cost ∧ y = 17 :=
by
  sorry

end blue_pill_cost_correct_l192_192532


namespace roots_square_sum_l192_192379

theorem roots_square_sum (a b : ℝ) 
  (h1 : a^2 - 4 * a + 4 = 0) 
  (h2 : b^2 - 4 * b + 4 = 0) 
  (h3 : a = b) :
  a^2 + b^2 = 8 := 
sorry

end roots_square_sum_l192_192379


namespace area_of_square_with_diagonal_l192_192919

theorem area_of_square_with_diagonal (c : ℝ) : 
  (∃ (s : ℝ), 2 * s^2 = c^4) → (∃ (A : ℝ), A = (c^4 / 2)) :=
  by
    sorry

end area_of_square_with_diagonal_l192_192919


namespace y_intercept_of_line_l192_192225

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 2 * y + 7 = 0) (hx : x = 0) : y = 7 / 2 :=
by
  sorry

end y_intercept_of_line_l192_192225


namespace total_stars_l192_192197

-- Define the daily stars earned by Shelby
def shelby_monday : Nat := 4
def shelby_tuesday : Nat := 6
def shelby_wednesday : Nat := 3
def shelby_thursday : Nat := 5
def shelby_friday : Nat := 2
def shelby_saturday : Nat := 3
def shelby_sunday : Nat := 7

-- Define the daily stars earned by Alex
def alex_monday : Nat := 5
def alex_tuesday : Nat := 3
def alex_wednesday : Nat := 6
def alex_thursday : Nat := 4
def alex_friday : Nat := 7
def alex_saturday : Nat := 2
def alex_sunday : Nat := 5

-- Define the total stars earned by Shelby in a week
def total_shelby_stars : Nat := shelby_monday + shelby_tuesday + shelby_wednesday + shelby_thursday + shelby_friday + shelby_saturday + shelby_sunday

-- Define the total stars earned by Alex in a week
def total_alex_stars : Nat := alex_monday + alex_tuesday + alex_wednesday + alex_thursday + alex_friday + alex_saturday + alex_sunday

-- The proof problem statement
theorem total_stars (total_shelby_stars total_alex_stars : Nat) : total_shelby_stars + total_alex_stars = 62 := by
  sorry

end total_stars_l192_192197


namespace solution_set_inequality_l192_192875

theorem solution_set_inequality (x : ℝ) : x * (x - 1) > 0 ↔ x < 0 ∨ x > 1 :=
by sorry

end solution_set_inequality_l192_192875


namespace multiplication_correct_l192_192349

theorem multiplication_correct (x : ℤ) (h : x - 6 = 51) : x * 6 = 342 := by
  sorry

end multiplication_correct_l192_192349


namespace magician_hat_probability_l192_192913

def total_arrangements : ℕ := Nat.choose 6 2
def favorable_arrangements : ℕ := Nat.choose 5 1
def probability_red_chips_drawn_first : ℚ := favorable_arrangements / total_arrangements

theorem magician_hat_probability :
  probability_red_chips_drawn_first = 1 / 3 :=
by
  sorry

end magician_hat_probability_l192_192913


namespace set_in_quadrant_I_l192_192053

theorem set_in_quadrant_I (x y : ℝ) (h1 : y ≥ 3 * x) (h2 : y ≥ 5 - x) (h3 : y < 7) : 
  x > 0 ∧ y > 0 :=
sorry

end set_in_quadrant_I_l192_192053


namespace exp4_is_odd_l192_192206

-- Define the domain for n to be integers and the expressions used in the conditions
variable (n : ℤ)

-- Define the expressions
def exp1 := (n + 1) ^ 2
def exp2 := (n + 1) ^ 2 - (n - 1)
def exp3 := (n + 1) ^ 3
def exp4 := (n + 1) ^ 3 - n ^ 3

-- Prove that exp4 is always odd
theorem exp4_is_odd : ∀ n : ℤ, exp4 n % 2 = 1 := by {
  -- Lean code does not require a proof here, we'll put sorry to skip the proof
  sorry
}

end exp4_is_odd_l192_192206


namespace num_integers_between_700_and_900_with_sum_of_digits_18_l192_192165

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem num_integers_between_700_and_900_with_sum_of_digits_18 : 
  ∃ k, k = 17 ∧ ∀ n, 700 ≤ n ∧ n ≤ 900 ∧ sum_of_digits n = 18 ↔ (1 ≤ k) := 
sorry

end num_integers_between_700_and_900_with_sum_of_digits_18_l192_192165


namespace union_A_B_l192_192805

open Set

-- Define the sets A and B
def setA : Set ℝ := { x | abs x < 3 }
def setB : Set ℝ := { x | x - 1 ≤ 0 }

-- State the theorem we want to prove
theorem union_A_B : setA ∪ setB = { x : ℝ | x < 3 } :=
by
  -- Skip the proof
  sorry

end union_A_B_l192_192805


namespace jessies_weight_loss_l192_192142

-- Definitions based on the given conditions
def initial_weight : ℝ := 74
def weight_loss_rate_even_days : ℝ := 0.2 + 0.15
def weight_loss_rate_odd_days : ℝ := 0.3
def total_exercise_days : ℕ := 25
def even_days : ℕ := (total_exercise_days - 1) / 2
def odd_days : ℕ := even_days + 1

-- The goal is to prove the total weight loss is 8.1 kg
theorem jessies_weight_loss : 
  (even_days * weight_loss_rate_even_days + odd_days * weight_loss_rate_odd_days) = 8.1 := 
by
  sorry

end jessies_weight_loss_l192_192142


namespace circle_center_sum_l192_192190

theorem circle_center_sum (x y : ℝ) (h : (x - 5)^2 + (y - 2)^2 = 38) : x + y = 7 := 
  sorry

end circle_center_sum_l192_192190


namespace f_no_zero_point_l192_192091

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem f_no_zero_point (x : ℝ) (h : x > 0) : f x ≠ 0 :=
by 
  sorry

end f_no_zero_point_l192_192091


namespace find_function_l192_192690

variable (R : Type) [LinearOrderedField R]

theorem find_function
  (f : R → R)
  (h : ∀ x y : R, f (x + y) + y ≤ f (f (f x))) :
  ∃ c : R, ∀ x : R, f x = c - x :=
sorry

end find_function_l192_192690


namespace babjis_height_less_by_20_percent_l192_192167

variable (B A : ℝ) (h : A = 1.25 * B)

theorem babjis_height_less_by_20_percent : ((A - B) / A) * 100 = 20 := by
  sorry

end babjis_height_less_by_20_percent_l192_192167


namespace number_of_self_inverse_subsets_is_15_l192_192267

-- Define the set M
def M : Set ℚ := ({-1, 0, 1/2, 1/3, 1, 2, 3, 4} : Set ℚ)

-- Definition of self-inverse set
def is_self_inverse (A : Set ℚ) : Prop := ∀ x ∈ A, 1/x ∈ A

-- Theorem stating the number of non-empty self-inverse subsets of M
theorem number_of_self_inverse_subsets_is_15 :
  (∃ S : Finset (Set ℚ), S.card = 15 ∧ ∀ A ∈ S, A ⊆ M ∧ is_self_inverse A) :=
sorry

end number_of_self_inverse_subsets_is_15_l192_192267


namespace hallie_hours_worked_on_tuesday_l192_192983

theorem hallie_hours_worked_on_tuesday
    (hourly_wage : ℝ := 10)
    (hours_monday : ℝ := 7)
    (tips_monday : ℝ := 18)
    (hours_wednesday : ℝ := 7)
    (tips_wednesday : ℝ := 20)
    (tips_tuesday : ℝ := 12)
    (total_earnings : ℝ := 240)
    (tuesday_hours : ℝ) :
    (hourly_wage * hours_monday + tips_monday) +
    (hourly_wage * hours_wednesday + tips_wednesday) +
    (hourly_wage * tuesday_hours + tips_tuesday) = total_earnings →
    tuesday_hours = 5 :=
by
  sorry

end hallie_hours_worked_on_tuesday_l192_192983


namespace relationship_between_number_and_value_l192_192384

theorem relationship_between_number_and_value (n v : ℝ) (h1 : n = 7) (h2 : n - 4 = 21 * v) : v = 1 / 7 :=
  sorry

end relationship_between_number_and_value_l192_192384


namespace domain_of_sqrt_tan_minus_one_l192_192164

open Real
open Set

def domain_sqrt_tan_minus_one : Set ℝ := 
  ⋃ k : ℤ, Ico (π/4 + k * π) (π/2 + k * π)

theorem domain_of_sqrt_tan_minus_one :
  {x : ℝ | ∃ y : ℝ, y = sqrt (tan x - 1)} = domain_sqrt_tan_minus_one :=
sorry

end domain_of_sqrt_tan_minus_one_l192_192164


namespace ratio_of_side_lengths_l192_192058

theorem ratio_of_side_lengths (t s : ℕ) (ht : 2 * t + (20 - 2 * t) = 20) (hs : 4 * s = 20) :
  t / s = 4 / 3 :=
by
  sorry

end ratio_of_side_lengths_l192_192058


namespace opposite_of_pi_l192_192110

theorem opposite_of_pi : -1 * Real.pi = -Real.pi := 
by sorry

end opposite_of_pi_l192_192110


namespace average_score_l192_192210

theorem average_score (classA_students classB_students : ℕ)
  (avg_score_classA avg_score_classB : ℕ)
  (h_classA : classA_students = 40)
  (h_classB : classB_students = 50)
  (h_avg_classA : avg_score_classA = 90)
  (h_avg_classB : avg_score_classB = 81) :
  (classA_students * avg_score_classA + classB_students * avg_score_classB) / 
  (classA_students + classB_students) = 85 := 
  by sorry

end average_score_l192_192210


namespace distance_to_school_l192_192615

theorem distance_to_school (d : ℝ) (h1 : d / 5 + d / 25 = 1) : d = 25 / 6 :=
by
  sorry

end distance_to_school_l192_192615


namespace ratio_of_sides_l192_192353

variable {A B C a b c : ℝ}

theorem ratio_of_sides
  (h1 : 2 * b * Real.sin (2 * A) = 3 * a * Real.sin B)
  (h2 : c = 2 * b) :
  a / b = Real.sqrt 2 := by
  sorry

end ratio_of_sides_l192_192353


namespace find_a_for_tangent_parallel_l192_192614

theorem find_a_for_tangent_parallel : 
  ∀ a : ℝ,
  (∀ (x y : ℝ), y = Real.log x - a * x → x = 1 → 2 * x + y - 1 = 0) →
  a = 3 :=
by
  sorry

end find_a_for_tangent_parallel_l192_192614


namespace dan_speed_must_exceed_48_l192_192889

theorem dan_speed_must_exceed_48 (d : ℕ) (s_cara : ℕ) (time_delay : ℕ) : 
  d = 120 → s_cara = 30 → time_delay = 3 / 2 → ∃ v : ℕ, v > 48 :=
by
  intro h1 h2 h3
  use 49
  sorry

end dan_speed_must_exceed_48_l192_192889


namespace probability_ratio_l192_192108

-- Defining the total number of cards and each number's frequency
def total_cards := 60
def each_number_frequency := 4
def distinct_numbers := 15

-- Defining probability p' and q'
def p' := (15: ℕ) * (Nat.choose 4 4) / (Nat.choose 60 4)
def q' := 210 * (Nat.choose 4 3) * (Nat.choose 4 1) / (Nat.choose 60 4)

-- Prove the value of q'/p'
theorem probability_ratio : (q' / p') = 224 := by
  sorry

end probability_ratio_l192_192108


namespace min_value_expr_ge_52_l192_192730

open Real

theorem min_value_expr_ge_52 (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  (sin x + 3 * (1 / sin x)) ^ 2 + (cos x + 3 * (1 / cos x)) ^ 2 ≥ 52 := 
by
  sorry

end min_value_expr_ge_52_l192_192730


namespace tangent_circle_equation_l192_192474

theorem tangent_circle_equation :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi →
    ∃ c : ℝ × ℝ, ∃ r : ℝ,
      (∀ (a b : ℝ), c = (a, b) →
        (|a * Real.cos θ + b * Real.sin θ - Real.cos θ - 2 * Real.sin θ - 2| = r) ∧
        (r = 2)) ∧
      (∃ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = r^2)) :=
by
  sorry

end tangent_circle_equation_l192_192474


namespace claire_gerbils_l192_192603

variables (G H : ℕ)

-- Claire's total pets
def total_pets : Prop := G + H = 92

-- One-quarter of the gerbils are male
def male_gerbils (G : ℕ) : ℕ := G / 4

-- One-third of the hamsters are male
def male_hamsters (H : ℕ) : ℕ := H / 3

-- Total males are 25
def total_males : Prop := male_gerbils G + male_hamsters H = 25

theorem claire_gerbils : total_pets G H → total_males G H → G = 68 :=
by
  intro h1 h2
  sorry

end claire_gerbils_l192_192603


namespace not_diff_of_squares_count_l192_192768

theorem not_diff_of_squares_count :
  ∃ n : ℕ, n ≤ 1000 ∧
  (∀ k : ℕ, k > 0 ∧ k ≤ 1000 → (∃ a b : ℤ, k = a^2 - b^2 ↔ k % 4 ≠ 2)) ∧
  n = 250 :=
sorry

end not_diff_of_squares_count_l192_192768


namespace bobby_last_10_throws_successful_l192_192660

theorem bobby_last_10_throws_successful :
    let initial_successful := 18 -- Bobby makes 18 successful throws out of his initial 30 throws.
    let total_throws := 30 + 10 -- Bobby makes a total of 40 throws.
    let final_successful := 0.64 * total_throws -- Bobby needs to make 64% of 40 throws to achieve a 64% success rate.
    let required_successful := 26 -- Adjusted to the nearest whole number.
    -- Bobby makes 8 successful throws in his last 10 attempts.
    required_successful - initial_successful = 8 := by
  sorry

end bobby_last_10_throws_successful_l192_192660


namespace number_of_unique_outfits_l192_192462

-- Define the given conditions
def num_shirts : ℕ := 8
def num_ties : ℕ := 6
def special_shirt_ties : ℕ := 3
def remaining_shirts := num_shirts - 1
def remaining_ties := num_ties

-- Define the proof problem
theorem number_of_unique_outfits : num_shirts * num_ties - remaining_shirts * remaining_ties + special_shirt_ties = 45 :=
by
  sorry

end number_of_unique_outfits_l192_192462


namespace max_distance_l192_192892

noncomputable def starting_cost : ℝ := 10
noncomputable def additional_cost_per_km : ℝ := 1.5
noncomputable def round_up : ℝ := 1
noncomputable def total_fare : ℝ := 19

theorem max_distance (x : ℝ) : (starting_cost + additional_cost_per_km * (x - 4)) = total_fare → x = 10 :=
by sorry

end max_distance_l192_192892


namespace geese_in_marsh_l192_192320

theorem geese_in_marsh (D : ℝ) (hD : D = 37.0) (G : ℝ) (hG : G = D + 21) : G = 58.0 := 
by 
  sorry

end geese_in_marsh_l192_192320


namespace O_is_incenter_l192_192112

variable {n : ℕ}
variable (A : Fin n → ℝ × ℝ)
variable (O : ℝ × ℝ)

-- Conditions
def inside_convex_ngon (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry
def angles_acute (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry
def angles_inequality (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry

-- This is the statement that we need to prove.
theorem O_is_incenter 
  (h1 : inside_convex_ngon O A)
  (h2 : angles_acute O A) 
  (h3 : angles_inequality O A) 
: sorry := sorry

end O_is_incenter_l192_192112


namespace max_donation_amount_l192_192723

theorem max_donation_amount (x : ℝ) : 
  (500 * x + 1500 * (x / 2) = 0.4 * 3750000) → x = 1200 :=
by 
  sorry

end max_donation_amount_l192_192723


namespace matts_weight_l192_192493

theorem matts_weight (protein_per_powder_rate : ℝ)
                     (weekly_intake_powder : ℝ)
                     (daily_protein_required_per_kg : ℝ)
                     (days_in_week : ℝ)
                     (expected_weight : ℝ)
    (h1 : protein_per_powder_rate = 0.8)
    (h2 : weekly_intake_powder = 1400)
    (h3 : daily_protein_required_per_kg = 2)
    (h4 : days_in_week = 7)
    (h5 : expected_weight = 80) :
    (weekly_intake_powder / days_in_week) * protein_per_powder_rate / daily_protein_required_per_kg = expected_weight := by
  sorry

end matts_weight_l192_192493


namespace cute_2020_all_integers_cute_l192_192450

-- Definition of "cute" integer
def is_cute (n : ℤ) : Prop :=
  ∃ (a b c d : ℤ), n = a^2 + b^3 + c^3 + d^5

-- Proof problem 1: Assert that 2020 is cute
theorem cute_2020 : is_cute 2020 :=
sorry

-- Proof problem 2: Assert that every integer is cute
theorem all_integers_cute (n : ℤ) : is_cute n :=
sorry

end cute_2020_all_integers_cute_l192_192450


namespace probability_male_monday_female_tuesday_l192_192043

structure Volunteers where
  men : ℕ
  women : ℕ
  total : ℕ

def group : Volunteers := {men := 2, women := 2, total := 4}

def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_male_monday_female_tuesday :
  let n := permutations group.total 2
  let m := combinations group.men 1 * combinations group.women 1
  (m / n : ℚ) = 1 / 3 :=
by
  sorry

end probability_male_monday_female_tuesday_l192_192043


namespace prove_bounds_l192_192192

variable (a b : ℝ)

-- Conditions
def condition1 : Prop := 6 * a - b = 45
def condition2 : Prop := 4 * a + b > 60

-- Proof problem statement
theorem prove_bounds (h1 : condition1 a b) (h2 : condition2 a b) : a > 10.5 ∧ b > 18 :=
sorry

end prove_bounds_l192_192192


namespace molecular_weight_calculation_l192_192969

-- Define the condition given in the problem
def molecular_weight_of_4_moles := 488 -- molecular weight of 4 moles in g/mol

-- Define the number of moles
def number_of_moles := 4

-- Define the expected molecular weight of 1 mole
def expected_molecular_weight_of_1_mole := 122 -- molecular weight of 1 mole in g/mol

-- Theorem statement
theorem molecular_weight_calculation : 
  molecular_weight_of_4_moles / number_of_moles = expected_molecular_weight_of_1_mole := 
by
  sorry

end molecular_weight_calculation_l192_192969


namespace last_digit_to_appear_is_6_l192_192711

def modified_fib (n : ℕ) : ℕ :=
match n with
| 1 => 2
| 2 => 3
| n + 3 => modified_fib (n + 2) + modified_fib (n + 1)
| _ => 0 -- To silence the "missing cases" warning; won't be hit.

theorem last_digit_to_appear_is_6 :
  ∃ N : ℕ, ∀ n : ℕ, (n < N → ∃ d, d < 10 ∧ 
    (∀ m < n, (modified_fib m) % 10 ≠ d) ∧ d = 6) := sorry

end last_digit_to_appear_is_6_l192_192711


namespace coprime_powers_l192_192495

theorem coprime_powers (n : ℕ) : Nat.gcd (n^5 + 4 * n^3 + 3 * n) (n^4 + 3 * n^2 + 1) = 1 :=
sorry

end coprime_powers_l192_192495


namespace find_unit_prices_minimize_cost_l192_192893

-- Definitions for the given prices and conditions
def cypress_price := 200
def pine_price := 150

def cost_eq1 (x y : ℕ) : Prop := 2 * x + 3 * y = 850
def cost_eq2 (x y : ℕ) : Prop := 3 * x + 2 * y = 900

-- Proving the unit prices of cypress and pine trees
theorem find_unit_prices (x y : ℕ) (h1 : cost_eq1 x y) (h2 : cost_eq2 x y) :
  x = cypress_price ∧ y = pine_price :=
sorry

-- Definitions for the number of trees and their costs
def total_trees := 80
def cypress_min (a : ℕ) : Prop := a ≥ 2 * (total_trees - a)
def total_cost (a : ℕ) : ℕ := 200 * a + 150 * (total_trees - a)

-- Conditions given for minimizing the cost
theorem minimize_cost (a : ℕ) (h1 : cypress_min a) : 
  a = 54 ∧ (total_trees - a) = 26 ∧ total_cost a = 14700 :=
sorry

end find_unit_prices_minimize_cost_l192_192893


namespace inf_pos_integers_n_sum_two_squares_l192_192253

theorem inf_pos_integers_n_sum_two_squares:
  ∃ (s : ℕ → ℕ), (∀ (k : ℕ), ∃ (a₁ b₁ a₂ b₂ : ℕ),
   a₁ > 0 ∧ b₁ > 0 ∧ a₂ > 0 ∧ b₂ > 0 ∧ s k = n ∧
   n = a₁^2 + b₁^2 ∧ n = a₂^2 + b₂^2 ∧ 
  (a₁ ≠ a₂ ∨ b₁ ≠ b₂)) := sorry

end inf_pos_integers_n_sum_two_squares_l192_192253


namespace combined_variance_is_178_l192_192459

noncomputable def average_weight_A := 60
noncomputable def variance_A := 100
noncomputable def average_weight_B := 64
noncomputable def variance_B := 200
noncomputable def ratio_A_B := (1, 3)

theorem combined_variance_is_178 :
  let nA := ratio_A_B.1
  let nB := ratio_A_B.2
  let avg_comb := (nA * average_weight_A + nB * average_weight_B) / (nA + nB)
  let var_comb := (nA * (variance_A + (average_weight_A - avg_comb)^2) + 
                   nB * (variance_B + (average_weight_B - avg_comb)^2)) / 
                   (nA + nB)
  var_comb = 178 := 
by
  sorry

end combined_variance_is_178_l192_192459


namespace smallest_x_for_perfect_cube_l192_192365

theorem smallest_x_for_perfect_cube (M : ℤ) :
  ∃ x : ℕ, 1680 * x = M^3 ∧ ∀ y : ℕ, 1680 * y = M^3 → 44100 ≤ y := 
sorry

end smallest_x_for_perfect_cube_l192_192365


namespace correct_statement_2_l192_192683

-- Definitions of parallel and perpendicular relationships
variables (a b : line) (α β : plane)

-- Conditions
def parallel (x y : plane) : Prop := sorry -- definition not provided
def perpendicular (x y : plane) : Prop := sorry -- definition not provided
def line_parallel_plane (l : line) (p : plane) : Prop := sorry -- definition not provided
def line_perpendicular_plane (l : line) (p : plane) : Prop := sorry -- definition not provided
def line_perpendicular (l1 l2 : line) : Prop := sorry -- definition not provided

-- Proof of the correct statement among the choices
theorem correct_statement_2 :
  line_perpendicular a b → line_perpendicular_plane a α → line_perpendicular_plane b β → perpendicular α β :=
by
  intros h1 h2 h3
  sorry

end correct_statement_2_l192_192683


namespace trigonometric_identities_l192_192801

theorem trigonometric_identities (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : Real.sin α = 4 / 5) :
    (Real.tan α = 4 / 3) ∧ 
    ((Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) :=
by
  sorry

end trigonometric_identities_l192_192801


namespace solve_for_x_l192_192160

theorem solve_for_x (x : ℚ) (h : 1/4 + 7/x = 13/x + 1/9) : x = 216/5 :=
by
  sorry

end solve_for_x_l192_192160


namespace min_students_solving_most_l192_192609

theorem min_students_solving_most (students problems : Nat) 
    (total_students : students = 10) 
    (problems_per_student : Nat → Nat) 
    (problems_per_student_property : ∀ s, s < students → problems_per_student s = 3) 
    (common_problem : ∀ s1 s2, s1 < students → s2 < students → s1 ≠ s2 → ∃ p, p < problems ∧ (∃ (solves1 solves2 : Nat → Nat), (solves1 p = 1 ∧ solves2 p = 1) ∧ s1 < students ∧ s2 < students)): 
  ∃ min_students, min_students = 5 :=
by
  sorry

end min_students_solving_most_l192_192609


namespace solve_equation_l192_192523

def equation_holds (x : ℝ) : Prop := 
  (1 / (x + 10)) + (1 / (x + 8)) = (1 / (x + 11)) + (1 / (x + 7))

theorem solve_equation : equation_holds (-9) :=
by
  sorry

end solve_equation_l192_192523


namespace range_of_m_l192_192481

theorem range_of_m (m : ℝ) :
  (∀ x, |x^2 - 4 * x + m| ≤ x + 4 ↔ (-4 ≤ m ∧ m ≤ 4)) ∧
  (∀ x, (x = 0 → |0^2 - 4 * 0 + m| ≤ 0 + 4) ∧ (x = 2 → ¬(|2^2 - 4 * 2 + m| ≤ 2 + 4))) →
  (-4 ≤ m ∧ m < -2) :=
by
  sorry

end range_of_m_l192_192481


namespace slope_range_l192_192154

variables (x y k : ℝ)

theorem slope_range :
  (2 ≤ x ∧ x ≤ 3) ∧ (y = -2 * x + 8) ∧ (k = -3 * y / (2 * x)) →
  -3 ≤ k ∧ k ≤ -1 :=
by
  sorry

end slope_range_l192_192154


namespace total_produce_of_mangoes_is_400_l192_192563

variable (A M O : ℕ)  -- Defines variables for total produce of apples, mangoes, and oranges respectively
variable (P : ℕ := 50)  -- Price per kg
variable (R : ℕ := 90000)  -- Total revenue

-- Definition of conditions
def apples_total_produce := 2 * M
def oranges_total_produce := M + 200
def total_weight_of_fruits := apples_total_produce + M + oranges_total_produce

-- Statement to prove
theorem total_produce_of_mangoes_is_400 :
  (total_weight_of_fruits = R / P) → (M = 400) :=
by
  sorry

end total_produce_of_mangoes_is_400_l192_192563


namespace amount_needed_for_free_delivery_l192_192816

theorem amount_needed_for_free_delivery :
  let chicken_cost := 1.5 * 6.00
  let lettuce_cost := 3.00
  let tomatoes_cost := 2.50
  let sweet_potatoes_cost := 4 * 0.75
  let broccoli_cost := 2 * 2.00
  let brussel_sprouts_cost := 2.50
  let total_cost := chicken_cost + lettuce_cost + tomatoes_cost + sweet_potatoes_cost + broccoli_cost + brussel_sprouts_cost
  let min_spend_for_free_delivery := 35.00
  min_spend_for_free_delivery - total_cost = 11.00 := sorry

end amount_needed_for_free_delivery_l192_192816


namespace karlsson_candies_28_l192_192228

def karlsson_max_candies (n : ℕ) : ℕ := (n * (n - 1)) / 2

theorem karlsson_candies_28 : karlsson_max_candies 28 = 378 := by
  sorry

end karlsson_candies_28_l192_192228


namespace teacher_zhang_friends_l192_192285

-- Define the conditions
def num_students : ℕ := 50
def both_friends : ℕ := 30
def neither_friend : ℕ := 1
def diff_in_friends : ℕ := 7

-- Prove that Teacher Zhang has 43 friends on social media
theorem teacher_zhang_friends : ∃ x : ℕ, 
  x + (x - diff_in_friends) - both_friends + neither_friend = num_students ∧ x = 43 := 
by
  sorry

end teacher_zhang_friends_l192_192285


namespace perpendicular_lines_have_a_zero_l192_192370

theorem perpendicular_lines_have_a_zero {a : ℝ} :
  ∀ x y : ℝ, (ax + y - 1 = 0) ∧ (x + a*y - 1 = 0) → a = 0 :=
by
  sorry

end perpendicular_lines_have_a_zero_l192_192370


namespace scalene_triangle_cannot_be_divided_into_two_congruent_triangles_l192_192106

-- Definitions and Conditions
structure Triangle :=
(a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)

-- Statement of the problem
theorem scalene_triangle_cannot_be_divided_into_two_congruent_triangles (T : Triangle) :
  ¬(∃ (D : ℝ) (ABD ACD : Triangle), ABD.a = ACD.a ∧ ABD.b = ACD.b ∧ ABD.c = ACD.c) :=
sorry

end scalene_triangle_cannot_be_divided_into_two_congruent_triangles_l192_192106


namespace pow_mod_sub_l192_192315

theorem pow_mod_sub (a b : ℕ) (n : ℕ) (h1 : a ≡ 5 [MOD 6]) (h2 : b ≡ 4 [MOD 6]) : (a^n - b^n) % 6 = 1 :=
by
  let a := 47
  let b := 22
  let n := 1987
  sorry

end pow_mod_sub_l192_192315


namespace complement_union_eq_l192_192056

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l192_192056


namespace length_of_diagonal_l192_192888

theorem length_of_diagonal (area : ℝ) (h1 h2 : ℝ) (d : ℝ) 
  (h_area : area = 75)
  (h_offsets : h1 = 6 ∧ h2 = 4) :
  d = 15 :=
by
  -- Given the conditions and formula, we can conclude
  sorry

end length_of_diagonal_l192_192888


namespace combined_weight_of_boxes_l192_192538

def first_box_weight := 2
def second_box_weight := 11
def last_box_weight := 5

theorem combined_weight_of_boxes :
  first_box_weight + second_box_weight + last_box_weight = 18 := by
  sorry

end combined_weight_of_boxes_l192_192538


namespace jack_total_dollars_l192_192012

-- Constants
def initial_dollars : ℝ := 45
def euro_amount : ℝ := 36
def yen_amount : ℝ := 1350
def ruble_amount : ℝ := 1500
def euro_to_dollar : ℝ := 2
def yen_to_dollar : ℝ := 0.009
def ruble_to_dollar : ℝ := 0.013
def transaction_fee_rate : ℝ := 0.01
def spending_rate : ℝ := 0.1

-- Convert each foreign currency to dollars
def euros_to_dollars : ℝ := euro_amount * euro_to_dollar
def yen_to_dollars : ℝ := yen_amount * yen_to_dollar
def rubles_to_dollars : ℝ := ruble_amount * ruble_to_dollar

-- Calculate transaction fees for each currency conversion
def euros_fee : ℝ := euros_to_dollars * transaction_fee_rate
def yen_fee : ℝ := yen_to_dollars * transaction_fee_rate
def rubles_fee : ℝ := rubles_to_dollars * transaction_fee_rate

-- Subtract transaction fees from the converted amounts
def euros_after_fee : ℝ := euros_to_dollars - euros_fee
def yen_after_fee : ℝ := yen_to_dollars - yen_fee
def rubles_after_fee : ℝ := rubles_to_dollars - rubles_fee

-- Calculate total dollars after conversion and fees
def total_dollars_before_spending : ℝ := initial_dollars + euros_after_fee + yen_after_fee + rubles_after_fee

-- Calculate 10% expenditure
def spending_amount : ℝ := total_dollars_before_spending * spending_rate

-- Calculate final amount after spending
def final_amount : ℝ := total_dollars_before_spending - spending_amount

theorem jack_total_dollars : final_amount = 132.85 := by
  sorry

end jack_total_dollars_l192_192012


namespace intersection_of_A_and_B_l192_192158

open Set

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 9} :=
by
  sorry

end intersection_of_A_and_B_l192_192158


namespace red_light_probability_l192_192754

theorem red_light_probability (n : ℕ) (p_r : ℚ) (waiting_time_for_two_red : ℚ) 
    (prob_two_red : ℚ) :
    n = 4 →
    p_r = (1/3 : ℚ) →
    waiting_time_for_two_red = 4 →
    prob_two_red = (8/27 : ℚ) :=
by
  intros hn hp hw
  sorry

end red_light_probability_l192_192754


namespace gcd_612_468_l192_192886

theorem gcd_612_468 : gcd 612 468 = 36 :=
by
  sorry

end gcd_612_468_l192_192886


namespace power_function_nature_l192_192581

def f (x : ℝ) : ℝ := x ^ (1/2)

theorem power_function_nature:
  (f 3 = Real.sqrt 3) ∧
  (¬ (∀ x, f (-x) = f x)) ∧
  (¬ (∀ x, f (-x) = -f x)) ∧
  (∀ x, 0 < x → 0 < f x) := 
by
  sorry

end power_function_nature_l192_192581


namespace solve_abs_inequality_l192_192689

theorem solve_abs_inequality (x : ℝ) (h : abs ((8 - x) / 4) < 3) : -4 < x ∧ x < 20 := 
  sorry

end solve_abs_inequality_l192_192689


namespace total_colors_needed_l192_192776

def num_planets : ℕ := 8
def num_people : ℕ := 3

theorem total_colors_needed : num_people * num_planets = 24 := by
  sorry

end total_colors_needed_l192_192776


namespace height_of_original_triangle_l192_192242

variable (a b c : ℝ)

theorem height_of_original_triangle (a b c : ℝ) : 
  ∃ h : ℝ, h = a + b + c :=
  sorry

end height_of_original_triangle_l192_192242


namespace partial_fraction_product_l192_192123

theorem partial_fraction_product : 
  (∃ A B C : ℚ, 
    (∀ x : ℚ, x ≠ 3 ∧ x ≠ -3 ∧ x ≠ 5 → 
      (x^2 - 21) / ((x - 3) * (x + 3) * (x - 5)) = A / (x - 3) + B / (x + 3) + C / (x - 5))
      ∧ (A * B * C = -1/16)) := 
    sorry

end partial_fraction_product_l192_192123


namespace probability_of_blue_or_orange_jelly_bean_is_5_over_13_l192_192235

def total_jelly_beans : ℕ := 7 + 9 + 8 + 10 + 5

def blue_or_orange_jelly_beans : ℕ := 10 + 5

def probability_blue_or_orange : ℚ := blue_or_orange_jelly_beans / total_jelly_beans

theorem probability_of_blue_or_orange_jelly_bean_is_5_over_13 :
  probability_blue_or_orange = 5 / 13 :=
by
  sorry

end probability_of_blue_or_orange_jelly_bean_is_5_over_13_l192_192235


namespace negate_proposition_l192_192764

-- Definitions of sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

-- The original proposition p
def p : Prop := ∀ x, is_odd x → is_even (2 * x)

-- The negation of the proposition p
def neg_p : Prop := ∃ x, is_odd x ∧ ¬ is_even (2 * x)

-- Proof problem statement: Prove that the negation of proposition p is as defined in neg_p
theorem negate_proposition :
  (∀ x, is_odd x → is_even (2 * x)) ↔ (∃ x, is_odd x ∧ ¬ is_even (2 * x)) :=
sorry

end negate_proposition_l192_192764


namespace find_n_values_l192_192487

theorem find_n_values : {n : ℕ | n ≥ 1 ∧ n ≤ 6 ∧ ∃ a b c : ℤ, a^n + b^n = c^n + n} = {1, 2, 3} :=
by sorry

end find_n_values_l192_192487


namespace find_locus_of_T_l192_192925

section Locus

variables {x y m : ℝ}
variable (M : ℝ × ℝ)

-- Condition: The equation of the ellipse
def ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1

-- Condition: Point P
def P := (1, 0)

-- Condition: M is any point on the ellipse, except A and B
def on_ellipse (M : ℝ × ℝ) := ellipse M.1 M.2 ∧ M ≠ (-2, 0) ∧ M ≠ (2, 0)

-- Condition: The intersection point N of line MP with the ellipse
def line_eq (m y : ℝ) := m * y + 1

-- Proposition: Locus of intersection point T of lines AM and BN
theorem find_locus_of_T 
  (hM : on_ellipse M)
  (hN : line_eq m M.2 = M.1)
  (hT : M.2 ≠ 0) :
  M.1 = 4 :=
sorry

end Locus

end find_locus_of_T_l192_192925


namespace interview_room_count_l192_192366

-- Define the number of people in the waiting room
def people_in_waiting_room : ℕ := 22

-- Define the increase in number of people
def extra_people_arrive : ℕ := 3

-- Define the total number of people after more people arrive
def total_people_after_arrival : ℕ := people_in_waiting_room + extra_people_arrive

-- Define the relationship between people in waiting room and interview room
def relation (x : ℕ) : Prop := total_people_after_arrival = 5 * x

theorem interview_room_count : ∃ x : ℕ, relation x ∧ x = 5 :=
by
  -- The proof will be provided here
  sorry

end interview_room_count_l192_192366


namespace proposition_1_proposition_4_l192_192809

-- Definitions
variable {a b c : Type} (Line : Type) (Plane : Type)
variable (a b c : Line) (γ : Plane)

-- Given conditions
variable (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)

-- Parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Propositions to prove
theorem proposition_1 (H1 : parallel a b) (H2 : parallel b c) : parallel a c := sorry

theorem proposition_4 (H3 : perpendicular a γ) (H4 : perpendicular b γ) : parallel a b := sorry

end proposition_1_proposition_4_l192_192809


namespace construct_angle_approx_l192_192707
-- Use a broader import to bring in the entirety of the necessary library

-- Define the problem 
theorem construct_angle_approx (α : ℝ) (m : ℕ) (h : ∃ l : ℕ, (l : ℝ) / 2^m * 90 ≤ α ∧ α ≤ ((l+1) : ℝ) / 2^m * 90) :
  ∃ β : ℝ, β ∈ { β | ∃ l : ℕ, β = (l : ℝ) / 2^m * 90} ∧ |α - β| ≤ 90 / 2^m :=
sorry

end construct_angle_approx_l192_192707


namespace solution_of_system_l192_192153

theorem solution_of_system (x y : ℝ) (h1 : x - 2 * y = 1) (h2 : x^3 - 6 * x * y - 8 * y^3 = 1) :
  y = (x - 1) / 2 :=
by
  sorry

end solution_of_system_l192_192153


namespace find_x_l192_192398

theorem find_x {x y : ℝ} (h1 : 3 * x - 2 * y = 7) (h2 : x^2 + 3 * y = 17) : x = 3.5 :=
sorry

end find_x_l192_192398


namespace positive_root_in_range_l192_192726

theorem positive_root_in_range : ∃ x > 0, (x^2 - 2 * x - 1 = 0) ∧ (2 < x ∧ x < 3) :=
by
  sorry

end positive_root_in_range_l192_192726


namespace stratified_sample_size_is_correct_l192_192404

def workshop_A_produces : ℕ := 120
def workshop_B_produces : ℕ := 90
def workshop_C_produces : ℕ := 60
def sample_from_C : ℕ := 4

def total_products : ℕ := workshop_A_produces + workshop_B_produces + workshop_C_produces

noncomputable def sampling_ratio : ℚ := (sample_from_C:ℚ) / (workshop_C_produces:ℚ)

noncomputable def sample_size : ℚ := total_products * sampling_ratio

theorem stratified_sample_size_is_correct :
  sample_size = 18 := by
  sorry

end stratified_sample_size_is_correct_l192_192404


namespace triangle_RS_length_l192_192673

theorem triangle_RS_length (PQ QR PS QS RS : ℝ)
  (h1 : PQ = 8) (h2 : QR = 8) (h3 : PS = 10) (h4 : QS = 5) :
  RS = 3.5 :=
by
  sorry

end triangle_RS_length_l192_192673


namespace more_girls_than_boys_l192_192887

theorem more_girls_than_boys
  (b g : ℕ)
  (ratio : b / g = 3 / 4)
  (total : b + g = 42) :
  g - b = 6 :=
sorry

end more_girls_than_boys_l192_192887


namespace Aiyanna_has_more_cookies_l192_192214

def Alyssa_cookies : ℕ := 129
def Aiyanna_cookies : ℕ := 140

theorem Aiyanna_has_more_cookies :
  Aiyanna_cookies - Alyssa_cookies = 11 := by
  sorry

end Aiyanna_has_more_cookies_l192_192214


namespace greatest_integer_y_l192_192677

theorem greatest_integer_y (y : ℤ) : (8 : ℚ) / 11 > y / 17 ↔ y ≤ 12 := 
sorry

end greatest_integer_y_l192_192677


namespace correct_propositions_l192_192862

structure Proposition :=
  (statement : String)
  (is_correct : Prop)

def prop1 : Proposition := {
  statement := "All sufficiently small positive numbers form a set.",
  is_correct := False -- From step b
}

def prop2 : Proposition := {
  statement := "The set containing 1, 2, 3, 1, 9 is represented by enumeration as {1, 2, 3, 1, 9}.",
  is_correct := False -- From step b
}

def prop3 : Proposition := {
  statement := "{1, 3, 5, 7} and {7, 5, 3, 1} denote the same set.",
  is_correct := True -- From step b
}

def prop4 : Proposition := {
  statement := "{y = -x} represents the collection of all points on the graph of the function y = -x.",
  is_correct := False -- From step b
}

theorem correct_propositions :
  prop3.is_correct ∧ ¬prop1.is_correct ∧ ¬prop2.is_correct ∧ ¬prop4.is_correct :=
by
  -- Here we put the proof steps, but for the exercise's purpose, we use sorry.
  sorry

end correct_propositions_l192_192862


namespace Phone_Bill_October_Phone_Bill_November_December_Extra_Cost_November_December_l192_192510

/-- Definitions for phone plans A and B and phone call durations -/
def fixed_cost_A : ℕ := 18
def free_minutes_A : ℕ := 1500
def price_per_minute_A : ℕ → ℚ := λ t => 0.1 * t

def fixed_cost_B : ℕ := 38
def free_minutes_B : ℕ := 4000
def price_per_minute_B : ℕ → ℚ := λ t => 0.07 * t

def call_duration_October : ℕ := 2600
def total_bill_November_December : ℚ := 176
def total_call_duration_November_December : ℕ := 5200

/-- Problem statements to be proven -/

theorem Phone_Bill_October : 
  fixed_cost_A + price_per_minute_A (call_duration_October - free_minutes_A) = 128 :=
  sorry

theorem Phone_Bill_November_December (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ total_call_duration_November_December) : 
  let bill_November := fixed_cost_A + price_per_minute_A (x - free_minutes_A)
  let bill_December := fixed_cost_B + price_per_minute_B (total_call_duration_November_December - x - free_minutes_B)
  bill_November + bill_December = total_bill_November_December :=
  sorry
  
theorem Extra_Cost_November_December :
  let actual_cost := 138 + 38
  let hypothetical_cost := fixed_cost_A + price_per_minute_A (total_call_duration_November_December - free_minutes_A)
  hypothetical_cost - actual_cost = 80 :=
  sorry

end Phone_Bill_October_Phone_Bill_November_December_Extra_Cost_November_December_l192_192510


namespace weight_of_milk_l192_192066

def max_bag_capacity : ℕ := 20
def green_beans : ℕ := 4
def carrots : ℕ := 2 * green_beans
def fit_more : ℕ := 2
def current_weight : ℕ := max_bag_capacity - fit_more
def total_weight_of_green_beans_and_carrots : ℕ := green_beans + carrots

theorem weight_of_milk : (current_weight - total_weight_of_green_beans_and_carrots) = 6 := by
  -- Proof to be written here
  sorry

end weight_of_milk_l192_192066


namespace slips_with_number_three_l192_192034

theorem slips_with_number_three : 
  ∀ (total_slips : ℕ) (number3 number8 : ℕ) (E : ℚ), 
  total_slips = 15 → 
  E = 5.6 → 
  number3 + number8 = total_slips → 
  (number3 : ℚ) / total_slips * 3 + (number8 : ℚ) / total_slips * 8 = E →
  number3 = 8 :=
by
  intros total_slips number3 number8 E h1 h2 h3 h4
  sorry

end slips_with_number_three_l192_192034


namespace work_completion_time_l192_192915

theorem work_completion_time (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 12) (hAC : A + C = 1 / 2) :
  1 / (B + C) = 3 :=
by
  -- The proof goes here
  sorry

end work_completion_time_l192_192915


namespace pour_tea_into_containers_l192_192352

-- Define the total number of containers
def total_containers : ℕ := 80

-- Define the amount of tea that Geraldo drank in terms of containers
def geraldo_drank_containers : ℚ := 3.5

-- Define the amount of tea that Geraldo consumed in terms of pints
def geraldo_drank_pints : ℕ := 7

-- Define the conversion factor from pints to gallons
def pints_per_gallon : ℕ := 8

-- Question: How many gallons of tea were poured into the containers?
theorem pour_tea_into_containers 
  (total_containers : ℕ)
  (geraldo_drank_containers : ℚ)
  (geraldo_drank_pints : ℕ)
  (pints_per_gallon : ℕ) :
  (total_containers * (geraldo_drank_pints / geraldo_drank_containers) / pints_per_gallon) = 20 :=
by
  sorry

end pour_tea_into_containers_l192_192352


namespace volume_not_occupied_by_cones_l192_192830

/-- Two cones with given dimensions are enclosed in a cylinder, and we want to find the volume 
    in the cylinder not occupied by the cones. -/
theorem volume_not_occupied_by_cones : 
  let radius := 10
  let height_cylinder := 26
  let height_cone1 := 10
  let height_cone2 := 16
  let volume_cylinder := π * (radius ^ 2) * height_cylinder
  let volume_cone1 := (1 / 3) * π * (radius ^ 2) * height_cone1
  let volume_cone2 := (1 / 3) * π * (radius ^ 2) * height_cone2
  let total_volume_cones := volume_cone1 + volume_cone2
  volume_cylinder - total_volume_cones = (2600 / 3) * π :=
by
  let radius := 10
  let height_cylinder := 26
  let height_cone1 := 10
  let height_cone2 := 16
  let volume_cylinder := π * (radius ^ 2) * height_cylinder
  let volume_cone1 := (1 / 3) * π * (radius ^ 2) * height_cone1
  let volume_cone2 := (1 / 3) * π * (radius ^ 2) * height_cone2
  let total_volume_cones := volume_cone1 + volume_cone2
  sorry

end volume_not_occupied_by_cones_l192_192830


namespace initial_percentage_correct_l192_192152

noncomputable def percentInitiallyFull (initialWater: ℕ) (waterAdded: ℕ) (fractionFull: ℚ) (capacity: ℕ) : ℚ :=
  (initialWater : ℚ) / (capacity : ℚ) * 100

theorem initial_percentage_correct (initialWater waterAdded capacity: ℕ) (fractionFull: ℚ) :
  waterAdded = 14 →
  fractionFull = 3/4 →
  capacity = 40 →
  initialWater + waterAdded = fractionFull * capacity →
  percentInitiallyFull initialWater waterAdded fractionFull capacity = 40 :=
by
  intros h1 h2 h3 h4
  unfold percentInitiallyFull
  sorry

end initial_percentage_correct_l192_192152


namespace max_value_of_expression_l192_192543

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l192_192543


namespace determine_numbers_l192_192114

theorem determine_numbers (a b c : ℕ) (h₁ : a + b + c = 15) 
  (h₂ : (1 / (a : ℝ)) + (1 / (b : ℝ)) + (1 / (c : ℝ)) = 71 / 105) : 
  (a = 3 ∧ b = 5 ∧ c = 7) ∨ (a = 3 ∧ b = 7 ∧ c = 5) ∨ (a = 5 ∧ b = 3 ∧ c = 7) ∨ 
  (a = 5 ∧ b = 7 ∧ c = 3) ∨ (a = 7 ∧ b = 3 ∧ c = 5) ∨ (a = 7 ∧ b = 5 ∧ c = 3) :=
sorry

end determine_numbers_l192_192114


namespace area_of_triangle_DOE_l192_192895

-- Definitions of points D, O, and E
def D (p : ℝ) : ℝ × ℝ := (0, p)
def O : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := (15, 0)

-- Theorem statement
theorem area_of_triangle_DOE (p : ℝ) : 
  let base := 15
  let height := p
  let area := (1/2) * base * height
  area = (15 * p) / 2 :=
by sorry

end area_of_triangle_DOE_l192_192895


namespace speed_last_segment_l192_192172

-- Definitions corresponding to conditions
def drove_total_distance : ℝ := 150
def total_time_minutes : ℝ := 120
def time_first_segment_minutes : ℝ := 40
def speed_first_segment_mph : ℝ := 70
def speed_second_segment_mph : ℝ := 75

-- The statement of the problem
theorem speed_last_segment :
  let total_distance : ℝ := drove_total_distance
  let total_time : ℝ := total_time_minutes / 60
  let time_first_segment : ℝ := time_first_segment_minutes / 60
  let time_second_segment : ℝ := time_first_segment
  let time_last_segment : ℝ := time_first_segment
  let distance_first_segment : ℝ := speed_first_segment_mph * time_first_segment
  let distance_second_segment : ℝ := speed_second_segment_mph * time_second_segment
  let distance_two_segments : ℝ := distance_first_segment + distance_second_segment
  let distance_last_segment : ℝ := total_distance - distance_two_segments
  let speed_last_segment := distance_last_segment / time_last_segment
  speed_last_segment = 80 := 
  sorry

end speed_last_segment_l192_192172


namespace range_of_x_l192_192512

variable (a x : ℝ)

theorem range_of_x :
  (∃ a ∈ Set.Icc 2 4, a * x ^ 2 + (a - 3) * x - 3 > 0) →
  x ∈ Set.Iio (-1) ∪ Set.Ioi (3 / 4) :=
by
  sorry

end range_of_x_l192_192512


namespace sam_gave_joan_seashells_l192_192661

variable (original_seashells : ℕ) (total_seashells : ℕ)

theorem sam_gave_joan_seashells (h1 : original_seashells = 70) (h2 : total_seashells = 97) :
  total_seashells - original_seashells = 27 :=
by
  sorry

end sam_gave_joan_seashells_l192_192661


namespace new_average_rent_l192_192356

theorem new_average_rent 
  (n : ℕ) (h_n : n = 4) 
  (avg_old : ℝ) (h_avg_old : avg_old = 800) 
  (inc_rate : ℝ) (h_inc_rate : inc_rate = 0.16) 
  (old_rent : ℝ) (h_old_rent : old_rent = 1250) 
  (new_rent : ℝ) (h_new_rent : new_rent = old_rent * (1 + inc_rate)) 
  (total_rent_old : ℝ) (h_total_rent_old : total_rent_old = n * avg_old)
  (total_rent_new : ℝ) (h_total_rent_new : total_rent_new = total_rent_old - old_rent + new_rent)
  (avg_new : ℝ) (h_avg_new : avg_new = total_rent_new / n) : 
  avg_new = 850 := 
sorry

end new_average_rent_l192_192356


namespace line_intersection_l192_192745

-- Parameters for the first line
def line1_param (s : ℝ) : ℝ × ℝ := (1 - 2 * s, 4 + 3 * s)

-- Parameters for the second line
def line2_param (v : ℝ) : ℝ × ℝ := (-v, 5 + 6 * v)

-- Statement of the intersection point
theorem line_intersection :
  ∃ (s v : ℝ), line1_param s = (-1 / 9, 17 / 3) ∧ line2_param v = (-1 / 9, 17 / 3) :=
by
  -- Placeholder for the proof, which we are not providing as per instructions
  sorry

end line_intersection_l192_192745


namespace convert_spherical_coords_l192_192668

theorem convert_spherical_coords (ρ θ φ : ℝ) (hρ : ρ > 0) (hθ : 0 ≤ θ ∧ θ < 2 * π) (hφ : 0 ≤ φ ∧ φ ≤ π) :
  (ρ = 4 ∧ θ = 4 * π / 3 ∧ φ = π / 4) ↔ (ρ, θ, φ) = (4, 4 * π / 3, π / 4) :=
by { sorry }

end convert_spherical_coords_l192_192668


namespace factorize_expression_l192_192247

theorem factorize_expression (x y : ℝ) : x^2 * y + 2 * x * y + y = y * (x + 1)^2 :=
by
  sorry

end factorize_expression_l192_192247


namespace sum_of_products_lt_zero_l192_192736

theorem sum_of_products_lt_zero (a b c d e f : ℤ) (h : ∃ (i : ℕ), i ≤ 6 ∧ i ≠ 6 ∧ (∀ i ∈ [a, b, c, d, e, f], i < 0 → i ≤ i)) :
  ab + cdef < 0 :=
sorry

end sum_of_products_lt_zero_l192_192736


namespace cocktail_cost_l192_192990

noncomputable def costPerLitreCocktail (cost_mixed_fruit_juice : ℝ) (cost_acai_juice : ℝ) (volume_mixed_fruit : ℝ) (volume_acai : ℝ) : ℝ :=
  let total_cost := cost_mixed_fruit_juice * volume_mixed_fruit + cost_acai_juice * volume_acai
  let total_volume := volume_mixed_fruit + volume_acai
  total_cost / total_volume

theorem cocktail_cost : costPerLitreCocktail 262.85 3104.35 32 21.333333333333332 = 1399.99 :=
  by
    sorry

end cocktail_cost_l192_192990


namespace math_problem_l192_192705

theorem math_problem : ((3.6 * 0.3) / 0.6 = 1.8) :=
by
  sorry

end math_problem_l192_192705


namespace calculate_f_5_l192_192826

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

theorem calculate_f_5 : f 5 = 4485 := 
by {
  -- The proof of the theorem will go here, using the Horner's method as described.
  sorry
}

end calculate_f_5_l192_192826


namespace miaCompletedAdditionalTasksOn6Days_l192_192251

def numDaysCompletingAdditionalTasks (n m : ℕ) : Prop :=
  n + m = 15 ∧ 4 * n + 7 * m = 78

theorem miaCompletedAdditionalTasksOn6Days (n m : ℕ): numDaysCompletingAdditionalTasks n m -> m = 6 :=
by
  intro h
  sorry

end miaCompletedAdditionalTasksOn6Days_l192_192251


namespace common_ratio_of_geometric_sequence_l192_192304

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ)
  (h_geom : ∃ q, ∀ n, a (n+1) = a n * q)
  (h1 : a 1 = 1 / 8)
  (h4 : a 4 = -1) :
  ∃ q, q = -2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l192_192304


namespace calc_expression_l192_192867

variable {x : ℝ}

theorem calc_expression :
    (2 + 3 * x) * (-2 + 3 * x) = 9 * x ^ 2 - 4 := sorry

end calc_expression_l192_192867


namespace length_of_goods_train_l192_192521

theorem length_of_goods_train
  (speed_man_train : ℕ) (speed_goods_train : ℕ) (passing_time : ℕ)
  (h1 : speed_man_train = 40)
  (h2 : speed_goods_train = 72)
  (h3 : passing_time = 9) :
  (112 * 1000 / 3600) * passing_time = 280 := 
by
  sorry

end length_of_goods_train_l192_192521


namespace function_equality_l192_192684

theorem function_equality (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 / x) = x / (1 - x)) :
  ∀ x : ℝ, f x = 1 / (x - 1) :=
by
  sorry

end function_equality_l192_192684


namespace area_enclosed_by_curves_l192_192250

theorem area_enclosed_by_curves (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, (x + a * y)^2 = 16 * a^2) ∧ (∀ x y : ℝ, (a * x - y)^2 = 4 * a^2) →
  ∃ A : ℝ, A = 32 * a^2 / (1 + a^2) :=
by
  sorry

end area_enclosed_by_curves_l192_192250


namespace iodine_initial_amount_l192_192574

theorem iodine_initial_amount (half_life : ℕ) (days_elapsed : ℕ) (final_amount : ℕ) (initial_amount : ℕ) :
  half_life = 8 → days_elapsed = 24 → final_amount = 2 → initial_amount = final_amount * 2 ^ (days_elapsed / half_life) → initial_amount = 16 :=
by
  intros h_half_life h_days_elapsed h_final_amount h_initial_exp
  rw [h_half_life, h_days_elapsed, h_final_amount] at h_initial_exp
  norm_num at h_initial_exp
  exact h_initial_exp

end iodine_initial_amount_l192_192574


namespace max_length_CD_l192_192608

open Real

/-- Given a circle with center O and diameter AB = 20 units,
    with points C and D positioned such that C is 6 units away from A
    and D is 7 units away from B on the diameter AB,
    prove that the maximum length of the direct path from C to D is 7 units.
-/
theorem max_length_CD {A B C D : ℝ} 
    (diameter : dist A B = 20) 
    (C_pos : dist A C = 6) 
    (D_pos : dist B D = 7) : 
    dist C D = 7 :=
by
  -- Details of the proof would go here
  sorry

end max_length_CD_l192_192608


namespace number_of_female_officers_is_382_l192_192463

noncomputable def F : ℝ := 
  let total_on_duty := 210
  let ratio_male_female := 3 / 2
  let percent_female_on_duty := 22 / 100
  let female_on_duty := total_on_duty * (2 / (3 + 2))
  let total_females := female_on_duty / percent_female_on_duty
  total_females

theorem number_of_female_officers_is_382 : F = 382 := 
by
  sorry

end number_of_female_officers_is_382_l192_192463


namespace basketball_substitution_mod_1000_l192_192522

def basketball_substitution_count_mod (n_playing n_substitutes max_subs : ℕ) : ℕ :=
  let no_subs := 1
  let one_sub := n_playing * n_substitutes
  let two_subs := n_playing * (n_playing - 1) * (n_substitutes * (n_substitutes - 1)) / 2
  let three_subs := n_playing * (n_playing - 1) * (n_playing - 2) *
                    (n_substitutes * (n_substitutes - 1) * (n_substitutes - 2)) / 6
  no_subs + one_sub + two_subs + three_subs 

theorem basketball_substitution_mod_1000 :
  basketball_substitution_count_mod 9 9 3 % 1000 = 10 :=
  by 
    -- Here the proof would be implemented
    sorry

end basketball_substitution_mod_1000_l192_192522


namespace scientific_notation_of_probe_unit_area_l192_192790

def probe_unit_area : ℝ := 0.0000064

theorem scientific_notation_of_probe_unit_area :
  ∃ (mantissa : ℝ) (exponent : ℤ), probe_unit_area = mantissa * 10^exponent ∧ mantissa = 6.4 ∧ exponent = -6 :=
by
  sorry

end scientific_notation_of_probe_unit_area_l192_192790


namespace smallest_y_in_arithmetic_series_l192_192731

theorem smallest_y_in_arithmetic_series (x y z : ℝ) (h1 : x < y) (h2 : y < z) (h3 : (x * y * z) = 216) : y = 6 :=
by 
  sorry

end smallest_y_in_arithmetic_series_l192_192731


namespace train_carriages_l192_192272

theorem train_carriages (num_trains : ℕ) (total_wheels : ℕ) (rows_per_carriage : ℕ) 
  (wheels_per_row : ℕ) (carriages_per_train : ℕ) :
  num_trains = 4 →
  total_wheels = 240 →
  rows_per_carriage = 3 →
  wheels_per_row = 5 →
  carriages_per_train = 
    (total_wheels / (rows_per_carriage * wheels_per_row)) / num_trains →
  carriages_per_train = 4 :=
by
  sorry

end train_carriages_l192_192272


namespace larger_angle_measure_l192_192467

theorem larger_angle_measure (x : ℝ) (hx : 7 * x = 90) : 4 * x = 360 / 7 := by
sorry

end larger_angle_measure_l192_192467


namespace mimi_spent_on_clothes_l192_192468

noncomputable def total_cost : ℤ := 8000
noncomputable def cost_adidas : ℤ := 600
noncomputable def cost_nike : ℤ := 3 * cost_adidas
noncomputable def cost_skechers : ℤ := 5 * cost_adidas
noncomputable def cost_clothes : ℤ := total_cost - (cost_adidas + cost_nike + cost_skechers)

theorem mimi_spent_on_clothes :
  cost_clothes = 2600 :=
by
  sorry

end mimi_spent_on_clothes_l192_192468


namespace alice_oranges_l192_192772

theorem alice_oranges (E A : ℕ) 
  (h1 : A = 2 * E) 
  (h2 : E + A = 180) : 
  A = 120 :=
by
  sorry

end alice_oranges_l192_192772


namespace red_balls_count_l192_192988

-- Define the conditions
def white_red_ratio : ℕ × ℕ := (5, 3)
def num_white_balls : ℕ := 15

-- Define the theorem to prove
theorem red_balls_count (r : ℕ) : r = num_white_balls / (white_red_ratio.1) * (white_red_ratio.2) :=
by sorry

end red_balls_count_l192_192988


namespace summer_camp_activity_l192_192240

theorem summer_camp_activity :
  ∃ (a b c d e f : ℕ), 
  a + b + c + d + 3 * e + 4 * f = 12 ∧ 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧
  f = 1 := by
  sorry

end summer_camp_activity_l192_192240


namespace initial_principal_amount_l192_192445

theorem initial_principal_amount
  (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (P : ℝ)
  (hA : A = 8400) 
  (hr : r = 0.05)
  (hn : n = 1) 
  (ht : t = 1) 
  (hformula : A = P * (1 + r / n) ^ (n * t)) : 
  P = 8000 :=
by
  rw [hA, hr, hn, ht] at hformula
  sorry

end initial_principal_amount_l192_192445


namespace work_completion_l192_192858

theorem work_completion (x y : ℕ) : 
  (1 / (x + y) = 1 / 12) ∧ (1 / y = 1 / 24) → x = 24 :=
by
  sorry

end work_completion_l192_192858


namespace jane_mistake_corrected_l192_192078

-- Conditions translated to Lean definitions
variables (x y z : ℤ)
variable (h1 : x - (y + z) = 15)
variable (h2 : x - y + z = 7)

-- Statement to prove
theorem jane_mistake_corrected : x - y = 11 :=
by
  -- Placeholder for the proof
  sorry

end jane_mistake_corrected_l192_192078


namespace stocks_closed_higher_l192_192824

-- Definition of the conditions:
def stocks : Nat := 1980
def increased (H L : Nat) : Prop := H = (1.20 : ℝ) * L
def total_stocks (H L : Nat) : Prop := H + L = stocks

-- Claim to prove
theorem stocks_closed_higher (H L : Nat) (h1 : increased H L) (h2 : total_stocks H L) : H = 1080 :=
by
  sorry

end stocks_closed_higher_l192_192824


namespace ratio_of_areas_of_concentric_circles_l192_192501

theorem ratio_of_areas_of_concentric_circles :
  (∀ (r1 r2 : ℝ), 
    r1 > 0 ∧ r2 > 0 ∧ 
    ((60 / 360) * 2 * Real.pi * r1 = (48 / 360) * 2 * Real.pi * r2)) →
    ((Real.pi * r1 ^ 2) / (Real.pi * r2 ^ 2) = (16 / 25)) :=
by
  intro h
  sorry

end ratio_of_areas_of_concentric_circles_l192_192501


namespace speed_of_current_l192_192233

-- Definitions for the conditions
variables (m c : ℝ)

-- Condition 1: man's speed with the current
def speed_with_current := m + c = 16

-- Condition 2: man's speed against the current
def speed_against_current := m - c = 9.6

-- The goal is to prove c = 3.2 given the conditions
theorem speed_of_current (h1 : speed_with_current m c) 
                         (h2 : speed_against_current m c) :
  c = 3.2 := 
sorry

end speed_of_current_l192_192233


namespace units_digit_n_is_7_l192_192371

def units_digit (x : ℕ) : ℕ := x % 10

theorem units_digit_n_is_7 (m n : ℕ) (h1 : m * n = 31 ^ 4) (h2 : units_digit m = 6) :
  units_digit n = 7 :=
sorry

end units_digit_n_is_7_l192_192371


namespace sum_arithmetic_sequence_100_to_110_l192_192004

theorem sum_arithmetic_sequence_100_to_110 :
  let a := 100
  let l := 110
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 1155 := by
  sorry

end sum_arithmetic_sequence_100_to_110_l192_192004


namespace parallel_line_plane_l192_192582

noncomputable def line : Type := sorry
noncomputable def plane : Type := sorry

-- Predicate for parallel lines
noncomputable def is_parallel_line (a b : line) : Prop := sorry

-- Predicate for parallel line and plane
noncomputable def is_parallel_plane (a : line) (α : plane) : Prop := sorry

-- Predicate for line contained within the plane
noncomputable def contained_in_plane (b : line) (α : plane) : Prop := sorry

theorem parallel_line_plane
  (a b : line) (α : plane)
  (h1 : is_parallel_line a b)
  (h2 : ¬ contained_in_plane a α)
  (h3 : contained_in_plane b α) :
  is_parallel_plane a α :=
sorry

end parallel_line_plane_l192_192582


namespace solve_system_of_equations_l192_192201

theorem solve_system_of_equations
  (x y : ℚ)
  (h1 : 5 * x - 3 * y = -7)
  (h2 : 4 * x + 6 * y = 34) :
  x = 10 / 7 ∧ y = 33 / 7 :=
by
  sorry

end solve_system_of_equations_l192_192201


namespace volume_box_constraint_l192_192767

theorem volume_box_constraint : ∀ x : ℕ, ((2 * x + 6) * (x^3 - 8) * (x^2 + 4) < 1200) → x = 2 :=
by
  intros x h
  -- Proof is skipped
  sorry

end volume_box_constraint_l192_192767


namespace inequality_solution_l192_192701

theorem inequality_solution (x : ℝ) :
  2 * (2 * x - 1) > 3 * x - 1 → x > 1 :=
by
  sorry

end inequality_solution_l192_192701


namespace solution_set_quadratic_l192_192088

-- Define the quadratic equation as a function
def quadratic_eq (x : ℝ) : ℝ := x^2 - 3 * x + 2

-- The theorem to prove
theorem solution_set_quadratic :
  {x : ℝ | quadratic_eq x = 0} = {1, 2} :=
by
  sorry

end solution_set_quadratic_l192_192088


namespace circle_radius_order_l192_192629

theorem circle_radius_order (r_X r_Y r_Z : ℝ)
  (hX : r_X = π)
  (hY : 2 * π * r_Y = 8 * π)
  (hZ : π * r_Z^2 = 9 * π) :
  r_Z < r_X ∧ r_X < r_Y :=
by {
  sorry
}

end circle_radius_order_l192_192629


namespace david_marks_in_physics_l192_192628

theorem david_marks_in_physics : 
  ∀ (P : ℝ), 
  let english := 72 
  let mathematics := 60 
  let chemistry := 62 
  let biology := 84 
  let average_marks := 62.6 
  let num_subjects := 5 
  let total_marks := average_marks * num_subjects 
  let known_marks := english + mathematics + chemistry + biology 
  total_marks - known_marks = P → P = 35 :=
by
  sorry

end david_marks_in_physics_l192_192628


namespace minimize_expression_at_9_l192_192643

noncomputable def minimize_expression (n : ℕ) : ℚ :=
  n / 3 + 27 / n

theorem minimize_expression_at_9 : minimize_expression 9 = 6 := by
  sorry

end minimize_expression_at_9_l192_192643


namespace Jamie_liquid_limit_l192_192258

theorem Jamie_liquid_limit :
  let milk_ounces := 8
  let grape_juice_ounces := 16
  let water_bottle_limit := 8
  let already_consumed := milk_ounces + grape_juice_ounces
  let max_before_bathroom := already_consumed + water_bottle_limit
  max_before_bathroom = 32 :=
by
  sorry

end Jamie_liquid_limit_l192_192258


namespace total_books_gwen_has_l192_192156

-- Definitions based on conditions in part a
def mystery_shelves : ℕ := 5
def picture_shelves : ℕ := 3
def books_per_shelf : ℕ := 4

-- Problem statement in Lean 4
theorem total_books_gwen_has : 
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 32 := by
  -- This is where the proof would go, but we include sorry to skip for now
  sorry

end total_books_gwen_has_l192_192156


namespace hyperbola_iff_m_lt_0_l192_192084

theorem hyperbola_iff_m_lt_0 (m : ℝ) : (m < 0) ↔ (∃ x y : ℝ,  x^2 + m * y^2 = m) :=
by sorry

end hyperbola_iff_m_lt_0_l192_192084


namespace time_for_runnerA_to_complete_race_l192_192015

variable (speedA : ℝ) -- speed of runner A in meters per second
variable (t : ℝ) -- time taken by runner A to complete the race in seconds
variable (tB : ℝ) -- time taken by runner B to complete the race in seconds

noncomputable def distanceA : ℝ := 1000 -- distance covered by runner A in meters
noncomputable def distanceB : ℝ := 950 -- distance covered by runner B in meters when A finishes
noncomputable def speedB : ℝ := distanceB / tB -- speed of runner B in meters per second

theorem time_for_runnerA_to_complete_race
    (h1 : distanceA = speedA * t)
    (h2 : distanceB = speedA * (t + 20)) :
    t = 400 :=
by
  sorry

end time_for_runnerA_to_complete_race_l192_192015


namespace simplify_fraction_144_12672_l192_192717

theorem simplify_fraction_144_12672 : (144 / 12672 : ℚ) = 1 / 88 :=
by
  sorry

end simplify_fraction_144_12672_l192_192717


namespace height_of_building_l192_192637

-- Define the conditions
def height_flagpole : ℝ := 18
def shadow_flagpole : ℝ := 45
def shadow_building : ℝ := 55

-- State the theorem to prove the height of the building
theorem height_of_building (h : ℝ) : (height_flagpole / shadow_flagpole) = (h / shadow_building) → h = 22 :=
by
  sorry

end height_of_building_l192_192637


namespace reciprocal_neg_2023_l192_192044

theorem reciprocal_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by 
  sorry

end reciprocal_neg_2023_l192_192044


namespace base_7_to_base_10_equiv_l192_192223

theorem base_7_to_base_10_equiv : 
  ∀ (d2 d1 d0 : ℕ), 
      d2 = 3 → d1 = 4 → d0 = 6 → 
      (d2 * 7^2 + d1 * 7^1 + d0 * 7^0) = 181 := 
by 
  sorry

end base_7_to_base_10_equiv_l192_192223


namespace modulus_of_2_plus_i_over_1_plus_2i_l192_192308

open Complex

noncomputable def modulus_of_complex_fraction : ℂ := 
  let z : ℂ := (2 + I) / (1 + 2 * I)
  abs z

theorem modulus_of_2_plus_i_over_1_plus_2i :
  modulus_of_complex_fraction = 1 := by
  sorry

end modulus_of_2_plus_i_over_1_plus_2i_l192_192308


namespace students_per_group_l192_192960

-- Defining the conditions
def total_students : ℕ := 256
def number_of_teachers : ℕ := 8

-- The statement to prove
theorem students_per_group :
  total_students / number_of_teachers = 32 :=
by
  sorry

end students_per_group_l192_192960


namespace candidate_p_wage_difference_l192_192964

theorem candidate_p_wage_difference
  (P Q : ℝ)    -- Candidate p's hourly wage is P, Candidate q's hourly wage is Q
  (H : ℝ)      -- Candidate p's working hours
  (total_payment : ℝ)
  (wage_ratio : P = 1.5 * Q)  -- Candidate p is paid 50% more per hour than candidate q
  (hours_diff : Q * (H + 10) = total_payment)  -- Candidate q's total payment equation
  (candidate_q_payment : Q * (H + 10) = 480)   -- total payment for candidate q
  (candidate_p_payment : 1.5 * Q * H = 480)    -- total payment for candidate p
  : P - Q = 8 := sorry

end candidate_p_wage_difference_l192_192964


namespace fraction_to_decimal_l192_192278

theorem fraction_to_decimal :
  ∀ x : ℚ, x = 52 / 180 → x = 0.1444 := 
sorry

end fraction_to_decimal_l192_192278


namespace equilateral_triangle_isosceles_triangle_l192_192971

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

noncomputable def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem equilateral_triangle (a b c : ℝ) (h : abs (a - b) + abs (b - c) = 0) : is_equilateral a b c :=
  sorry

theorem isosceles_triangle (a b c : ℝ) (h : (a - b) * (b - c) = 0) : is_isosceles a b c :=
  sorry

end equilateral_triangle_isosceles_triangle_l192_192971


namespace max_sum_of_squares_l192_192139

theorem max_sum_of_squares :
  ∃ m n : ℕ, (m ∈ Finset.range 101) ∧ (n ∈ Finset.range 101) ∧ ((n^2 - n * m - m^2)^2 = 1) ∧ (m^2 + n^2 = 10946) :=
by
  sorry

end max_sum_of_squares_l192_192139


namespace B_subset_complementA_A_intersection_B_nonempty_A_union_B_eq_A_l192_192506

-- Define the sets A and B
def setA : Set ℝ := {x : ℝ | x < 1 ∨ x > 2}
def setB (m : ℝ) : Set ℝ := 
  if m = 0 then {x : ℝ | x > 1} 
  else if m < 0 then {x : ℝ | x > 1 ∨ x < (2/m)}
  else if 0 < m ∧ m < 2 then {x : ℝ | 1 < x ∧ x < (2/m)}
  else if m = 2 then ∅
  else {x : ℝ | (2/m) < x ∧ x < 1}

-- Complement of set A
def complementA : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 2}

-- Proposition: if B subset of complement of A
theorem B_subset_complementA (m : ℝ) : setB m ⊆ complementA ↔ 1 ≤ m ∧ m ≤ 2 := by
  sorry

-- Similarly, we can define the other two propositions
theorem A_intersection_B_nonempty (m : ℝ) : (setA ∩ setB m).Nonempty ↔ m < 1 ∨ m > 2 := by
  sorry

theorem A_union_B_eq_A (m : ℝ) : setA ∪ setB m = setA ↔ m ≥ 2 := by
  sorry

end B_subset_complementA_A_intersection_B_nonempty_A_union_B_eq_A_l192_192506


namespace solve_linear_system_l192_192083

theorem solve_linear_system (x y z : ℝ) 
  (h1 : y + z = 20 - 4 * x)
  (h2 : x + z = -10 - 4 * y)
  (h3 : x + y = 14 - 4 * z)
  : 2 * x + 2 * y + 2 * z = 8 :=
by
  sorry

end solve_linear_system_l192_192083


namespace physical_education_class_min_size_l192_192338

theorem physical_education_class_min_size :
  ∃ (x : Nat), 3 * x + 2 * (x + 1) > 50 ∧ 5 * x + 2 = 52 := by
  sorry

end physical_education_class_min_size_l192_192338


namespace perpendicular_lines_l192_192507

theorem perpendicular_lines (a : ℝ) :
  (∃ x y : ℝ, x * a + 3 * y - 1 = 0) ∧ (∃ x y : ℝ, 2 * x + (a - 1) * y + 1 = 0) ∧
  (∀ m1 m2 : ℝ, m1 = - a / 3 → m2 = - 2 / (a - 1) → m1 * m2 = -1) →
  a = 3 / 5 :=
sorry

end perpendicular_lines_l192_192507


namespace maximum_value_of_expression_l192_192527

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2)

theorem maximum_value_of_expression (x y z : ℝ ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 3) :
  problem_statement x y z ≤ 81 / 4 :=
sorry

end maximum_value_of_expression_l192_192527


namespace total_books_l192_192539

theorem total_books (joan_books tom_books sarah_books alex_books : ℕ) 
  (h1 : joan_books = 10)
  (h2 : tom_books = 38)
  (h3 : sarah_books = 25)
  (h4 : alex_books = 45) : 
  joan_books + tom_books + sarah_books + alex_books = 118 := 
by 
  sorry

end total_books_l192_192539


namespace probability_non_defective_pens_l192_192753

theorem probability_non_defective_pens :
  let total_pens := 12
  let defective_pens := 6
  let non_defective_pens := total_pens - defective_pens
  let probability_first_non_defective := non_defective_pens / total_pens
  let probability_second_non_defective := (non_defective_pens - 1) / (total_pens - 1)
  (probability_first_non_defective * probability_second_non_defective = 5 / 22) :=
by
  rfl

end probability_non_defective_pens_l192_192753


namespace mutually_exclusive_one_two_odd_l192_192665

-- Define the event that describes rolling a fair die
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- Event: Exactly one die shows an odd number -/
def exactly_one_odd (d1 d2 : ℕ) : Prop :=
  (is_odd d1 ∧ ¬ is_odd d2) ∨ (¬ is_odd d1 ∧ is_odd d2)

/-- Event: Exactly two dice show odd numbers -/
def exactly_two_odd (d1 d2 : ℕ) : Prop :=
  is_odd d1 ∧ is_odd d2

/-- Main theorem: Exactly one odd number and exactly two odd numbers are mutually exclusive but not converse-/
theorem mutually_exclusive_one_two_odd (d1 d2 : ℕ) :
  (exactly_one_odd d1 d2 ∧ ¬ exactly_two_odd d1 d2) ∧
  (¬ exactly_one_odd d1 d2 ∧ exactly_two_odd d1 d2) ∧
  (exactly_one_odd d1 d2 ∨ exactly_two_odd d1 d2) :=
by
  sorry

end mutually_exclusive_one_two_odd_l192_192665


namespace surface_area_of_cube_l192_192368

noncomputable def cube_edge_length : ℝ := 20

theorem surface_area_of_cube (edge_length : ℝ) (h : edge_length = cube_edge_length) : 
    6 * edge_length ^ 2 = 2400 :=
by
  rw [h]
  sorry  -- proof placeholder

end surface_area_of_cube_l192_192368


namespace max_value_of_sum_l192_192759

open Real

theorem max_value_of_sum (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : a + b + c = 3) :
  (ab / (a + b) + bc / (b + c) + ca / (c + a)) ≤ 3 / 2 :=
sorry

end max_value_of_sum_l192_192759


namespace probability_at_least_one_multiple_of_4_is_correct_l192_192418

noncomputable def probability_at_least_one_multiple_of_4 : ℚ :=
  let total_numbers := 100
  let multiples_of_4 := 25
  let non_multiples_of_4 := total_numbers - multiples_of_4
  let p_non_multiple := (non_multiples_of_4 : ℚ) / total_numbers
  let p_both_non_multiples := p_non_multiple^2
  let p_at_least_one_multiple := 1 - p_both_non_multiples
  p_at_least_one_multiple

theorem probability_at_least_one_multiple_of_4_is_correct :
  probability_at_least_one_multiple_of_4 = 7 / 16 :=
by
  sorry

end probability_at_least_one_multiple_of_4_is_correct_l192_192418


namespace find_f_l192_192924

theorem find_f (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x^2 + x) :
  ∀ x : ℤ, f x = x^2 - x :=
by
  intro x
  sorry

end find_f_l192_192924


namespace composite_sum_l192_192354

theorem composite_sum (a b : ℤ) (h : 56 * a = 65 * b) : ∃ m n : ℤ,  m > 1 ∧ n > 1 ∧ a + b = m * n :=
sorry

end composite_sum_l192_192354


namespace area_of_triangle_ABF_l192_192337

theorem area_of_triangle_ABF :
  let C : Set (ℝ × ℝ) := {p | (p.1 ^ 2 / 4) + (p.2 ^ 2 / 3) = 1}
  let line : Set (ℝ × ℝ) := {p | p.1 - p.2 - 1 = 0}
  let F : ℝ × ℝ := (-1, 0)
  let AB := C ∩ line
  ∃ A B : ℝ × ℝ, A ∈ AB ∧ B ∈ AB ∧ A ≠ B ∧ 
  (1/2) * (2 : ℝ) * (12 * Real.sqrt (2 : ℝ) / 7) = (12 * Real.sqrt (2 : ℝ) / 7) :=
sorry

end area_of_triangle_ABF_l192_192337


namespace range_of_k_l192_192821

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x + 3| + |x - 1| > k) ↔ k < 4 :=
by sorry

end range_of_k_l192_192821


namespace find_other_integer_l192_192708

theorem find_other_integer (x y : ℤ) (h1 : 3*x + 4*y = 103) (h2 : x = 19 ∨ y = 19) : x = 9 ∨ y = 9 :=
by sorry

end find_other_integer_l192_192708


namespace sqrt_fraction_l192_192675

theorem sqrt_fraction {a b c : ℝ}
  (h1 : a = Real.sqrt 27)
  (h2 : b = Real.sqrt 243)
  (h3 : c = Real.sqrt 48) :
  (a + b) / c = 3 := by
  sorry

end sqrt_fraction_l192_192675


namespace dave_hourly_wage_l192_192659

theorem dave_hourly_wage :
  ∀ (hours_monday hours_tuesday total_money : ℝ),
  hours_monday = 6 → hours_tuesday = 2 → total_money = 48 →
  (total_money / (hours_monday + hours_tuesday) = 6) :=
by
  intros hours_monday hours_tuesday total_money h_monday h_tuesday h_money
  sorry

end dave_hourly_wage_l192_192659


namespace bert_money_problem_l192_192249

-- Define the conditions as hypotheses
theorem bert_money_problem
  (n : ℝ)
  (h1 : n > 0)  -- Since he can't have negative or zero dollars initially
  (h2 : (1/2) * ((3/4) * n - 9) = 15) :
  n = 52 :=
sorry

end bert_money_problem_l192_192249


namespace subtract_fractions_correct_l192_192962

theorem subtract_fractions_correct :
  (3 / 8 + 5 / 12 - 1 / 6) = (5 / 8) := by
sorry

end subtract_fractions_correct_l192_192962


namespace height_of_scale_model_eq_29_l192_192843

def empireStateBuildingHeight : ℕ := 1454

def scaleRatio : ℕ := 50

def scaleModelHeight (actualHeight : ℕ) (ratio : ℕ) : ℤ :=
  Int.ofNat actualHeight / ratio

theorem height_of_scale_model_eq_29 : scaleModelHeight empireStateBuildingHeight scaleRatio = 29 :=
by
  -- Proof would go here
  sorry

end height_of_scale_model_eq_29_l192_192843


namespace range_of_x_l192_192991

theorem range_of_x (x : ℝ) : x + 2 ≥ 0 ∧ x - 3 ≠ 0 → x ≥ -2 ∧ x ≠ 3 :=
by
  sorry

end range_of_x_l192_192991


namespace F_of_3153_max_value_of_N_l192_192823

-- Define friendly number predicate
def is_friendly (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  a - b = c - d

-- Define F(M)
def F (M : ℕ) : ℕ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let s := M / 10
  let t := M % 1000
  s - t - 10 * b

-- Prove F(3153) = 152
theorem F_of_3153 : F 3153 = 152 :=
by sorry

-- Define the given predicate for N
def is_k_special (N : ℕ) : Prop :=
  let x := N / 1000
  let y := (N / 100) % 10
  let m := (N / 30) % 10
  let n := N % 10
  (N % 5 = 1) ∧ (1000 * x + 100 * y + 30 * m + n + 1001 = N) ∧
  (0 ≤ y ∧ y < x ∧ x ≤ 8) ∧ (0 ≤ m ∧ m ≤ 3) ∧ (0 ≤ n ∧ n ≤ 8) ∧ 
  is_friendly N

-- Prove the maximum value satisfying the given constraints
theorem max_value_of_N : ∀ N, is_k_special N → N ≤ 9696 :=
by sorry

end F_of_3153_max_value_of_N_l192_192823


namespace no_perfect_squares_l192_192186

theorem no_perfect_squares (x y z t : ℕ) (h1 : xy - zt = k) (h2 : x + y = k) (h3 : z + t = k) :
  ¬ (∃ m n : ℕ, x * y = m^2 ∧ z * t = n^2) := by
  sorry

end no_perfect_squares_l192_192186


namespace time_left_for_nap_l192_192647

noncomputable def total_time : ℝ := 20
noncomputable def first_train_time : ℝ := 2 + 1
noncomputable def second_train_time : ℝ := 3 + 1
noncomputable def transfer_one_time : ℝ := 0.75 + 0.5
noncomputable def third_train_time : ℝ := 2 + 1
noncomputable def transfer_two_time : ℝ := 1
noncomputable def fourth_train_time : ℝ := 1
noncomputable def transfer_three_time : ℝ := 0.5
noncomputable def fifth_train_time_before_nap : ℝ := 1.5

noncomputable def total_activities_time : ℝ :=
  first_train_time +
  second_train_time +
  transfer_one_time +
  third_train_time +
  transfer_two_time +
  fourth_train_time +
  transfer_three_time +
  fifth_train_time_before_nap

theorem time_left_for_nap : total_time - total_activities_time = 4.75 := by
  sorry

end time_left_for_nap_l192_192647


namespace largest_even_two_digit_largest_odd_two_digit_l192_192819

-- Define conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Theorem statements
theorem largest_even_two_digit : ∃ n, is_two_digit n ∧ is_even n ∧ ∀ m, is_two_digit m ∧ is_even m → m ≤ n := 
sorry

theorem largest_odd_two_digit : ∃ n, is_two_digit n ∧ is_odd n ∧ ∀ m, is_two_digit m ∧ is_odd m → m ≤ n := 
sorry

end largest_even_two_digit_largest_odd_two_digit_l192_192819


namespace range_of_a_if_p_is_false_l192_192576

theorem range_of_a_if_p_is_false :
  (∀ x : ℝ, x^2 + a * x + a ≥ 0) → (0 ≤ a ∧ a ≤ 4) := 
sorry

end range_of_a_if_p_is_false_l192_192576


namespace find_term_number_l192_192577

noncomputable def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem find_term_number
  (a₁ : ℤ)
  (d : ℤ)
  (n : ℕ)
  (h₀ : a₁ = 1)
  (h₁ : d = 3)
  (h₂ : arithmetic_sequence a₁ d n = 2011) :
  n = 671 :=
  sorry

end find_term_number_l192_192577


namespace quadrilateral_inequality_l192_192029

theorem quadrilateral_inequality (A C : ℝ) (AB AC AD BC CD : ℝ) (h1 : A + C < 180) (h2 : A > 0) (h3 : C > 0) (h4 : AB > 0) (h5 : AC > 0) (h6 : AD > 0) (h7 : BC > 0) (h8 : CD > 0) : 
  AB * CD + AD * BC < AC * (AB + AD) := 
sorry

end quadrilateral_inequality_l192_192029


namespace prove_square_ratio_l192_192300
noncomputable section

-- Definitions from given conditions
variables (a b : ℝ) (d : ℝ := Real.sqrt (a^2 + b^2))

-- Condition from the problem
def ratio_condition : Prop := a / b = (a + 2 * b) / d

-- The theorem we need to prove
theorem prove_square_ratio (h : ratio_condition a b d) : 
  ∃ k : ℝ, k = a / b ∧ k^4 - 3*k^2 - 4*k - 4 = 0 := 
by
  sorry

end prove_square_ratio_l192_192300


namespace lenny_initial_money_l192_192860

-- Definitions based on the conditions
def spent_on_video_games : ℕ := 24
def spent_at_grocery_store : ℕ := 21
def amount_left : ℕ := 39

-- Statement of the problem
theorem lenny_initial_money : spent_on_video_games + spent_at_grocery_store + amount_left = 84 :=
by
  sorry

end lenny_initial_money_l192_192860


namespace angle_sum_unique_l192_192427

theorem angle_sum_unique (α β : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : β ∈ Set.Ioo (π / 2) π) 
  (h3 : Real.tan α + Real.tan β - Real.tan α * Real.tan β + 1 = 0) : 
  α + β = 7 * π / 4 :=
sorry

end angle_sum_unique_l192_192427


namespace map_representation_l192_192388

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l192_192388


namespace sum_of_A_B_in_B_l192_192706

def A : Set ℤ := { x | ∃ k : ℤ, x = 2 * k }
def B : Set ℤ := { x | ∃ k : ℤ, x = 2 * k + 1 }
def C : Set ℤ := { x | ∃ k : ℤ, x = 4 * k + 1 }

theorem sum_of_A_B_in_B (a b : ℤ) (ha : a ∈ A) (hb : b ∈ B) : a + b ∈ B := by
  sorry

end sum_of_A_B_in_B_l192_192706


namespace tan_sum_l192_192119

-- Define the conditions as local variables
variables {α β : ℝ} (h₁ : Real.tan α = -2) (h₂ : Real.tan β = 5)

-- The statement to prove
theorem tan_sum : Real.tan (α + β) = 3 / 11 :=
by 
  -- Proof goes here, using 'sorry' as placeholder
  sorry

end tan_sum_l192_192119


namespace parabola_directrix_l192_192904

theorem parabola_directrix (x y : ℝ) (h : x^2 = 12 * y) : y = -3 :=
sorry

end parabola_directrix_l192_192904


namespace girls_tried_out_l192_192031

-- Definitions for conditions
def boys_trying_out : ℕ := 4
def students_called_back : ℕ := 26
def students_did_not_make_cut : ℕ := 17

-- Definition to calculate total students who tried out
def total_students_who_tried_out : ℕ := students_called_back + students_did_not_make_cut

-- Proof statement
theorem girls_tried_out : ∀ (G : ℕ), G + boys_trying_out = total_students_who_tried_out → G = 39 :=
by
  intro G
  intro h
  rw [total_students_who_tried_out, boys_trying_out] at h
  sorry

end girls_tried_out_l192_192031


namespace minimum_dimes_to_afford_sneakers_l192_192218

-- Define constants and conditions using Lean
def sneaker_cost : ℝ := 45.35
def ten_dollar_bills_count : ℕ := 3
def quarter_count : ℕ := 4
def dime_value : ℝ := 0.1
def quarter_value : ℝ := 0.25
def ten_dollar_bill_value : ℝ := 10.0

-- Define a function to calculate the total amount based on the number of dimes
def total_amount (dimes : ℕ) : ℝ :=
  (ten_dollar_bills_count * ten_dollar_bill_value) +
  (quarter_count * quarter_value) +
  (dimes * dime_value)

-- The main theorem to be proven
theorem minimum_dimes_to_afford_sneakers (n : ℕ) : total_amount n ≥ sneaker_cost ↔ n ≥ 144 :=
by
  sorry

end minimum_dimes_to_afford_sneakers_l192_192218


namespace speed_increase_l192_192931

theorem speed_increase (v_initial: ℝ) (t_initial: ℝ) (t_new: ℝ) :
  v_initial = 60 → t_initial = 1 → t_new = 0.5 →
  v_new = (1 / (t_new / 60)) →
  v_increase = v_new - v_initial →
  v_increase = 60 :=
by
  sorry

end speed_increase_l192_192931


namespace segment_MN_length_l192_192640

theorem segment_MN_length
  (A B C D M N : ℝ)
  (hA : A < B)
  (hB : B < C)
  (hC : C < D)
  (hM : M = (A + C) / 2)
  (hN : N = (B + D) / 2)
  (hAD : D - A = 68)
  (hBC : C - B = 26) :
  |M - N| = 21 :=
sorry

end segment_MN_length_l192_192640


namespace quadratic_inequality_solution_l192_192617

theorem quadratic_inequality_solution (x : ℝ) : -3 < x ∧ x < 4 → x^2 - x - 12 < 0 := by
  sorry

end quadratic_inequality_solution_l192_192617


namespace mostWaterIntake_l192_192311

noncomputable def dailyWaterIntakeDongguk : ℝ := 5 * 0.2 -- Total water intake in liters per day for Dongguk
noncomputable def dailyWaterIntakeYoonji : ℝ := 6 * 0.3 -- Total water intake in liters per day for Yoonji
noncomputable def dailyWaterIntakeHeejin : ℝ := 4 * 500 / 1000 -- Total water intake in liters per day for Heejin (converted from milliliters)

theorem mostWaterIntake :
  dailyWaterIntakeHeejin = max dailyWaterIntakeDongguk (max dailyWaterIntakeYoonji dailyWaterIntakeHeejin) :=
by
  sorry

end mostWaterIntake_l192_192311


namespace cubic_poly_real_roots_l192_192072

theorem cubic_poly_real_roots (a b c d : ℝ) (h : a ≠ 0) : 
  ∃ (min_roots max_roots : ℕ), 1 ≤ min_roots ∧ max_roots ≤ 3 ∧ min_roots = 1 ∧ max_roots = 3 :=
by
  sorry

end cubic_poly_real_roots_l192_192072


namespace inequality_one_inequality_two_l192_192672

variable (a b c : ℝ)

-- First Inequality Proof Statement
theorem inequality_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) : 
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 := 
sorry

-- Second Inequality Proof Statement
theorem inequality_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) : 
  (a^3 + b^3 + c^3 + 1/a + 1/b + 1/c) ≥ 2 * (a + b + c) := 
sorry

end inequality_one_inequality_two_l192_192672


namespace pastry_problem_minimum_n_l192_192927

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end pastry_problem_minimum_n_l192_192927


namespace angle_measure_l192_192922

theorem angle_measure (x : ℝ) (h : 90 - x = 3 * (180 - x)) : x = 45 := by
  sorry

end angle_measure_l192_192922


namespace opposite_of_neg_2023_l192_192588

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by sorry

end opposite_of_neg_2023_l192_192588


namespace luka_age_difference_l192_192957

theorem luka_age_difference (a l : ℕ) (h1 : a = 8) (h2 : ∀ m : ℕ, m = 6 → l = m + 4) : l - a = 2 :=
by
  -- Assume Aubrey's age is 8
  have ha : a = 8 := h1
  -- Assume Max's age at Aubrey's 8th birthday is 6
  have hl : l = 10 := h2 6 rfl
  -- Hence, Luka is 2 years older than Aubrey
  sorry

end luka_age_difference_l192_192957


namespace paige_mp3_player_songs_l192_192105

/--
Paige had 11 songs on her mp3 player.
She deleted 9 old songs.
She added 8 new songs.

We are to prove:
- The final number of songs on her mp3 player is 10.
-/
theorem paige_mp3_player_songs (initial_songs deleted_songs added_songs final_songs : ℕ)
  (h₁ : initial_songs = 11)
  (h₂ : deleted_songs = 9)
  (h₃ : added_songs = 8) :
  final_songs = initial_songs - deleted_songs + added_songs :=
by
  sorry

end paige_mp3_player_songs_l192_192105


namespace range_m_for_p_range_m_for_q_range_m_for_not_p_or_q_l192_192321

-- Define the propositions p and q
def p (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + 2 + m = 0

def q (m : ℝ) : Prop :=
  1 - 2 * m < 0 ∧ m + 2 > 0 ∨ 1 - 2 * m > 0 ∧ m + 2 < 0 -- Hyperbola condition

-- Prove the ranges of m
theorem range_m_for_p {m : ℝ} (hp : p m) : m ≤ -2 ∨ m ≥ 1 :=
sorry

theorem range_m_for_q {m : ℝ} (hq : q m) : m < -2 ∨ m > (1 / 2) :=
sorry

theorem range_m_for_not_p_or_q {m : ℝ} (h_not_p : ¬ (p m)) (h_not_q : ¬ (q m)) : -2 < m ∧ m ≤ (1 / 2) :=
sorry

end range_m_for_p_range_m_for_q_range_m_for_not_p_or_q_l192_192321


namespace marie_eggs_total_l192_192234

variable (x : ℕ) -- Number of eggs in each box

-- Conditions as definitions
def egg_weight := 10 -- weight of each egg in ounces
def total_boxes := 4 -- total number of boxes
def remaining_boxes := 3 -- boxes left after one is discarded
def remaining_weight := 90 -- total weight of remaining eggs in ounces

-- Proof statement
theorem marie_eggs_total : remaining_boxes * egg_weight * x = remaining_weight → total_boxes * x = 12 :=
by
  intros h
  sorry

end marie_eggs_total_l192_192234


namespace positive_difference_of_b_values_l192_192525

noncomputable def g (n : ℤ) : ℤ :=
if n ≤ 0 then n^2 + 3 * n + 2 else 3 * n - 15

theorem positive_difference_of_b_values : 
  abs (-5 - 9) = 14 :=
by {
  sorry
}

end positive_difference_of_b_values_l192_192525


namespace layla_goldfish_count_l192_192793

def goldfish_count (total_food : ℕ) (swordtails_count : ℕ) (swordtails_food : ℕ) (guppies_count : ℕ) (guppies_food : ℕ) (goldfish_food : ℕ) : ℕ :=
  total_food - (swordtails_count * swordtails_food + guppies_count * guppies_food) / goldfish_food

theorem layla_goldfish_count : goldfish_count 12 3 2 8 1 1 = 2 := by
  sorry

end layla_goldfish_count_l192_192793


namespace quadratic_has_one_solution_l192_192222

theorem quadratic_has_one_solution (n : ℤ) : 
  (n ^ 2 - 64 = 0) ↔ (n = 8 ∨ n = -8) := 
by
  sorry

end quadratic_has_one_solution_l192_192222


namespace mudit_age_l192_192394

theorem mudit_age :
    ∃ x : ℤ, x + 16 = 3 * (x - 4) ∧ x = 14 :=
by
  use 14
  sorry -- Proof goes here

end mudit_age_l192_192394


namespace arithmetic_seq_8th_term_l192_192323

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_seq_8th_term_l192_192323


namespace ratio_of_floors_l192_192926

-- Define the number of floors of each building
def floors_building_A := 4
def floors_building_B := 4 + 9
def floors_building_C := 59

-- Prove the ratio of floors in Building C to Building B
theorem ratio_of_floors :
  floors_building_C / floors_building_B = 59 / 13 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_floors_l192_192926


namespace roots_of_quadratic_sum_cube_l192_192803

noncomputable def quadratic_roots (a b c : ℤ) (p q : ℤ) : Prop :=
  p^2 - b * p + c = 0 ∧ q^2 - b * q + c = 0

theorem roots_of_quadratic_sum_cube (p q : ℤ) :
  quadratic_roots 1 (-5) 6 p q →
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 503 :=
by
  sorry

end roots_of_quadratic_sum_cube_l192_192803


namespace jeans_price_increase_l192_192560

theorem jeans_price_increase
  (C R P : ℝ)
  (h1 : P = 1.15 * R)
  (h2 : P = 1.6100000000000001 * C) :
  R = 1.4 * C :=
by
  sorry

end jeans_price_increase_l192_192560


namespace percentage_calculation_l192_192766

def part : ℝ := 12.356
def whole : ℝ := 12356
def expected_percentage : ℝ := 0.1

theorem percentage_calculation (p w : ℝ) (h_p : p = part) (h_w : w = whole) : 
  (p / w) * 100 = expected_percentage :=
sorry

end percentage_calculation_l192_192766


namespace total_boxes_moved_l192_192587

-- Define a truck's capacity and number of trips
def truck_capacity : ℕ := 4
def trips : ℕ := 218

-- Prove that the total number of boxes is 872
theorem total_boxes_moved : truck_capacity * trips = 872 := by
  sorry

end total_boxes_moved_l192_192587


namespace crates_of_mangoes_sold_l192_192148

def total_crates_sold := 50
def crates_grapes_sold := 13
def crates_passion_fruits_sold := 17

theorem crates_of_mangoes_sold : 
  (total_crates_sold - (crates_grapes_sold + crates_passion_fruits_sold) = 20) :=
by 
  sorry

end crates_of_mangoes_sold_l192_192148


namespace angle_A_area_of_triangle_l192_192189

open Real

theorem angle_A (a : ℝ) (A B C : ℝ) 
  (h_a : a = 2 * sqrt 3)
  (h_condition1 : 4 * cos A ^ 2 + 4 * cos B * cos C + 1 = 4 * sin B * sin C) :
  A = π / 3 := 
sorry

theorem area_of_triangle (a b c A : ℝ) 
  (h_A : A = π / 3)
  (h_a : a = 2 * sqrt 3)
  (h_b : b = 3 * c) :
  (1 / 2) * b * c * sin A = 9 * sqrt 3 / 7 := 
sorry

end angle_A_area_of_triangle_l192_192189


namespace tankard_one_quarter_full_l192_192748

theorem tankard_one_quarter_full
  (C : ℝ) 
  (h : (3 / 4) * C = 480) : 
  (1 / 4) * C = 160 := 
by
  sorry

end tankard_one_quarter_full_l192_192748


namespace numbers_of_form_xy9z_div_by_132_l192_192168

theorem numbers_of_form_xy9z_div_by_132 (x y z : ℕ) :
  let N := 1000 * x + 100 * y + 90 + z
  (N % 4 = 0) ∧ ((x + y + 9 + z) % 3 = 0) ∧ ((x + 9 - y - z) % 11 = 0) ↔ 
  (N = 3696) ∨ (N = 4092) ∨ (N = 6996) ∨ (N = 7392) :=
by
  intros
  let N := 1000 * x + 100 * y + 90 + z
  sorry

end numbers_of_form_xy9z_div_by_132_l192_192168


namespace robin_packages_l192_192833

theorem robin_packages (p t n : ℕ) (h1 : p = 18) (h2 : t = 486) : t / p = n ↔ n = 27 :=
by
  rw [h1, h2]
  norm_num
  sorry

end robin_packages_l192_192833


namespace map_length_scale_l192_192905

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l192_192905


namespace find_m_of_quadratic_root_zero_l192_192792

theorem find_m_of_quadratic_root_zero (m : ℝ) (h : ∃ x, (m * x^2 + 5 * x + m^2 - 2 * m = 0) ∧ x = 0) : m = 2 :=
sorry

end find_m_of_quadratic_root_zero_l192_192792


namespace inverse_of_h_l192_192440

def h (x : ℝ) : ℝ := 3 + 6 * x

noncomputable def k (x : ℝ) : ℝ := (x - 3) / 6

theorem inverse_of_h : ∀ x, h (k x) = x :=
by
  intro x
  unfold h k
  sorry

end inverse_of_h_l192_192440


namespace financial_outcome_l192_192039

theorem financial_outcome :
  let initial_value : ℝ := 12000
  let selling_price : ℝ := initial_value * 1.20
  let buying_price : ℝ := selling_price * 0.85
  let financial_outcome : ℝ := buying_price - initial_value
  financial_outcome = 240 :=
by
  sorry

end financial_outcome_l192_192039


namespace sum_of_areas_of_six_rectangles_eq_572_l192_192489

theorem sum_of_areas_of_six_rectangles_eq_572 :
  let lengths := [1, 3, 5, 7, 9, 11]
  let areas := lengths.map (λ x => 2 * x^2)
  areas.sum = 572 :=
by 
  sorry

end sum_of_areas_of_six_rectangles_eq_572_l192_192489


namespace area_of_triangle_l192_192032

-- Definitions of the conditions
def hypotenuse_AC (a b c : ℝ) : Prop := c = 50
def sum_of_legs (a b : ℝ) : Prop := a + b = 70
def pythagorean_theorem (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- The main theorem statement
theorem area_of_triangle (a b c : ℝ) (h1 : hypotenuse_AC a b c)
  (h2 : sum_of_legs a b) (h3 : pythagorean_theorem a b c) : 
  (1/2) * a * b = 300 := 
by
  sorry

end area_of_triangle_l192_192032


namespace exists_pos_integer_n_l192_192081

theorem exists_pos_integer_n (n : ℕ) (hn_pos : n > 0) (h : ∃ m : ℕ, m * m = 1575 * n) : n = 7 :=
sorry

end exists_pos_integer_n_l192_192081


namespace sum_of_two_numbers_l192_192611

theorem sum_of_two_numbers (S L : ℝ) (h1 : S = 10.0) (h2 : 7 * S = 5 * L) : S + L = 24.0 :=
by
  -- proof goes here
  sorry

end sum_of_two_numbers_l192_192611


namespace ratio_is_three_l192_192697

-- Define the conditions
def area_of_garden : ℕ := 588
def width_of_garden : ℕ := 14
def length_of_garden : ℕ := area_of_garden / width_of_garden

-- Define the ratio
def ratio_length_to_width := length_of_garden / width_of_garden

-- The proof statement
theorem ratio_is_three : ratio_length_to_width = 3 := 
by sorry

end ratio_is_three_l192_192697


namespace probability_red_or_white_is_11_over_13_l192_192829

-- Given data
def total_marbles : ℕ := 60
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - blue_marbles - red_marbles

def blue_size : ℕ := 2
def red_size : ℕ := 1
def white_size : ℕ := 1

-- Total size value of all marbles
def total_size_value : ℕ := (blue_size * blue_marbles) + (red_size * red_marbles) + (white_size * white_marbles)

-- Probability of selecting a red or white marble
def probability_red_or_white : ℚ := (red_size * red_marbles + white_size * white_marbles) / total_size_value

-- Theorem to prove
theorem probability_red_or_white_is_11_over_13 : probability_red_or_white = 11 / 13 :=
by sorry

end probability_red_or_white_is_11_over_13_l192_192829


namespace compare_a_x_l192_192775

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem compare_a_x (x a b : ℝ) (h1 : a = log_base 5 (3^x + 4^x))
                    (h2 : b = log_base 4 (5^x - 3^x)) (h3 : a ≥ b) : x ≤ a :=
by
  sorry

end compare_a_x_l192_192775


namespace num_squares_less_than_1000_with_ones_digit_2_3_or_4_l192_192419

-- Define a function that checks if the one's digit of a number is one of 2, 3, or 4.
def ends_in (n : ℕ) (d : ℕ) : Prop := n % 10 = d

-- Define the main theorem to prove
theorem num_squares_less_than_1000_with_ones_digit_2_3_or_4 : 
  ∃ n, n = 6 ∧ ∀ m < 1000, ∃ k, m = k^2 → ends_in m 2 ∨ ends_in m 3 ∨ ends_in m 4 :=
sorry

end num_squares_less_than_1000_with_ones_digit_2_3_or_4_l192_192419


namespace probability_at_least_two_meters_l192_192795

def rope_length : ℝ := 6
def num_nodes : ℕ := 5
def equal_parts : ℕ := 6
def min_length : ℝ := 2

theorem probability_at_least_two_meters (h_rope_division : rope_length / equal_parts = 1) :
  let favorable_cuts := 3
  let total_cuts := num_nodes
  (favorable_cuts : ℝ) / total_cuts = 3 / 5 :=
by
  sorry

end probability_at_least_two_meters_l192_192795


namespace arithmetic_sequence_sum_abs_values_l192_192151

theorem arithmetic_sequence_sum_abs_values (n : ℕ) (a : ℕ → ℤ)
  (h₁ : a 1 = 13)
  (h₂ : ∀ k, a (k + 1) = a k + (-4)) :
  T_n = if n ≤ 4 then 15 * n - 2 * n^2 else 2 * n^2 - 15 * n + 56 :=
by sorry

end arithmetic_sequence_sum_abs_values_l192_192151


namespace no_real_solution_exists_l192_192799

theorem no_real_solution_exists:
  ¬ ∃ (x y z : ℝ), (x ^ 2 + 4 * y * z + 2 * z = 0) ∧
                   (x + 2 * x * y + 2 * z ^ 2 = 0) ∧
                   (2 * x * z + y ^ 2 + y + 1 = 0) :=
by
  sorry

end no_real_solution_exists_l192_192799


namespace negation_proposition_equivalence_l192_192294

theorem negation_proposition_equivalence :
  (¬ ∃ x₀ : ℝ, (2 / x₀ + Real.log x₀ ≤ 0)) ↔ (∀ x : ℝ, 2 / x + Real.log x > 0) := 
sorry

end negation_proposition_equivalence_l192_192294


namespace weight_of_new_person_l192_192331

theorem weight_of_new_person (W : ℝ) : 
  let avg_original := W / 5
  let avg_new := avg_original + 4
  let total_new := 5 * avg_new
  let weight_new_person := total_new - (W - 50)
  weight_new_person = 70 :=
by
  let avg_original := W / 5
  let avg_new := avg_original + 4
  let total_new := 5 * avg_new
  let weight_new_person := total_new - (W - 50)
  have : weight_new_person = 70 := sorry
  exact this

end weight_of_new_person_l192_192331


namespace cost_of_one_basketball_deck_l192_192470

theorem cost_of_one_basketball_deck (total_money_spent : ℕ) 
  (mary_sunglasses_cost : ℕ) (mary_jeans_cost : ℕ) 
  (rose_shoes_cost : ℕ) (rose_decks_count : ℕ) 
  (mary_total_cost : total_money_spent = 2 * mary_sunglasses_cost + mary_jeans_cost)
  (rose_total_cost : total_money_spent = rose_shoes_cost + 2 * (total_money_spent - rose_shoes_cost) / rose_decks_count) :
  (total_money_spent - rose_shoes_cost) / rose_decks_count = 25 := 
by 
  sorry

end cost_of_one_basketball_deck_l192_192470


namespace complex_omega_sum_l192_192791

open Complex

theorem complex_omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 + ω^48 + ω^51 + ω^54 + ω^57 + ω^60 + ω^63 = 1 := 
by
  sorry

end complex_omega_sum_l192_192791


namespace min_2a_plus_3b_l192_192184

theorem min_2a_plus_3b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_parallel : (a * (b - 3) - 2 * b = 0)) :
  (2 * a + 3 * b) = 25 :=
by
  -- proof goes here
  sorry

end min_2a_plus_3b_l192_192184


namespace abs_add_conditions_l192_192364

theorem abs_add_conditions (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a < b) :
  a + b = 1 ∨ a + b = 7 :=
by
  sorry

end abs_add_conditions_l192_192364


namespace man_speed_is_approximately_54_009_l192_192293

noncomputable def speed_in_kmh (d : ℝ) (t : ℝ) : ℝ := 
  -- Convert distance to kilometers and time to hours
  let distance_km := d / 1000
  let time_hours := t / 3600
  distance_km / time_hours

theorem man_speed_is_approximately_54_009 :
  abs (speed_in_kmh 375.03 25 - 54.009) < 0.001 := 
by
  sorry

end man_speed_is_approximately_54_009_l192_192293


namespace first_term_of_arithmetic_sequence_l192_192402

theorem first_term_of_arithmetic_sequence (a : ℕ) (median last_term : ℕ) 
  (h_arithmetic_progression : true) (h_median : median = 1010) (h_last_term : last_term = 2015) :
  a = 5 :=
by
  have h1 : 2 * median = 2020 := by sorry
  have h2 : last_term + a = 2020 := by sorry
  have h3 : 2015 + a = 2020 := by sorry
  have h4 : a = 2020 - 2015 := by sorry
  have h5 : a = 5 := by sorry
  exact h5

end first_term_of_arithmetic_sequence_l192_192402


namespace fraction_of_girls_is_one_third_l192_192455

-- Define the number of children and number of boys
def total_children : Nat := 45
def boys : Nat := 30

-- Calculate the number of girls
def girls : Nat := total_children - boys

-- Calculate the fraction of girls
def fraction_of_girls : Rat := (girls : Rat) / (total_children : Rat)

theorem fraction_of_girls_is_one_third : fraction_of_girls = 1 / 3 :=
by
  sorry -- Proof is not required

end fraction_of_girls_is_one_third_l192_192455


namespace num_passenger_cars_l192_192245

noncomputable def passengerCars (p c : ℕ) : Prop :=
  c = p / 2 + 3 ∧ p + c = 69

theorem num_passenger_cars (p c : ℕ) (h : passengerCars p c) : p = 44 :=
by
  unfold passengerCars at h
  cases h
  sorry

end num_passenger_cars_l192_192245


namespace smallest_y_not_defined_l192_192727

theorem smallest_y_not_defined : 
  ∃ y : ℝ, (6 * y^2 - 37 * y + 6 = 0) ∧ (∀ z : ℝ, (6 * z^2 - 37 * z + 6 = 0) → y ≤ z) ∧ y = 1 / 6 :=
by
  sorry

end smallest_y_not_defined_l192_192727


namespace books_remaining_in_library_l192_192127

def initial_books : ℕ := 250
def books_taken_out_Tuesday : ℕ := 120
def books_returned_Wednesday : ℕ := 35
def books_withdrawn_Thursday : ℕ := 15

theorem books_remaining_in_library :
  initial_books
  - books_taken_out_Tuesday
  + books_returned_Wednesday
  - books_withdrawn_Thursday = 150 :=
by
  sorry

end books_remaining_in_library_l192_192127


namespace infinite_solutions_congruence_l192_192387

theorem infinite_solutions_congruence (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ᶠ x in at_top, a ^ x + x ≡ b [MOD c] :=
sorry

end infinite_solutions_congruence_l192_192387


namespace quadratic_inequality_solution_l192_192215

open Real

theorem quadratic_inequality_solution :
    ∀ x : ℝ, -8 * x^2 + 6 * x - 1 < 0 ↔ 0.25 < x ∧ x < 0.5 :=
by sorry

end quadratic_inequality_solution_l192_192215


namespace min_distance_line_curve_l192_192136

/-- 
  Given line l with parametric equations:
    x = 1 + t * cos α,
    y = t * sin α,
  and curve C with the polar equation:
    ρ * sin^2 θ = 4 * cos θ,
  prove:
    1. The Cartesian coordinate equation of C is y^2 = 4x.
    2. The minimum value of the distance |AB|, where line l intersects curve C, is 4.
-/
theorem min_distance_line_curve {t α θ ρ x y : ℝ} 
  (h_line_x: x = 1 + t * Real.cos α)
  (h_line_y: y = t * Real.sin α)
  (h_curve_polar: ρ * (Real.sin θ)^2 = 4 * Real.cos θ)
  (h_alpha_range: 0 < α ∧ α < Real.pi) : 
  (∀ {x y}, y^2 = 4 * x) ∧ (min_value_of_AB = 4) :=
sorry

end min_distance_line_curve_l192_192136


namespace solve_for_x_l192_192417

theorem solve_for_x (x : ℝ) (h : 0.60 * 500 = 0.50 * x) : x = 600 :=
  sorry

end solve_for_x_l192_192417


namespace part_a_l192_192406

theorem part_a (a b c : ℝ) (m : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (a + b)^m + (b + c)^m + (c + a)^m ≤ 2^m * (a^m + b^m + c^m) :=
by
  sorry

end part_a_l192_192406


namespace sum_of_cubes_consecutive_integers_l192_192412

theorem sum_of_cubes_consecutive_integers (x : ℕ) (h1 : 0 < x) (h2 : x * (x + 1) * (x + 2) = 12 * (3 * x + 3)) :
  x^3 + (x + 1)^3 + (x + 2)^3 = 216 :=
by
  -- proof will go here
  sorry

end sum_of_cubes_consecutive_integers_l192_192412


namespace no_3_digit_number_with_digit_sum_27_and_even_l192_192605

-- Define what it means for a number to be 3-digit
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the digit-sum function
def digitSum (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

-- Define what it means for a number to be even
def isEven (n : ℕ) : Prop := n % 2 = 0

-- State the proof problem
theorem no_3_digit_number_with_digit_sum_27_and_even :
  ∀ n : ℕ, isThreeDigit n → digitSum n = 27 → isEven n → false :=
by
  -- Proof should go here
  sorry

end no_3_digit_number_with_digit_sum_27_and_even_l192_192605


namespace storm_first_thirty_minutes_rain_l192_192952

theorem storm_first_thirty_minutes_rain 
  (R: ℝ)
  (H1: R + (R / 2) + (1 / 2) = 8)
  : R = 5 :=
by
  sorry

end storm_first_thirty_minutes_rain_l192_192952


namespace max_integer_value_of_f_l192_192187

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 9 * x + 13) / (3 * x^2 + 9 * x + 5)

theorem max_integer_value_of_f : ∀ x : ℝ, ∃ n : ℤ, f x ≤ n ∧ n = 2 :=
by 
  sorry

end max_integer_value_of_f_l192_192187


namespace initial_number_l192_192284

theorem initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end initial_number_l192_192284


namespace division_quotient_is_correct_l192_192155

noncomputable def polynomial_division_quotient : Polynomial ℚ :=
  Polynomial.div (Polynomial.C 8 * Polynomial.X ^ 3 + 
                  Polynomial.C 16 * Polynomial.X ^ 2 + 
                  Polynomial.C (-7) * Polynomial.X + 
                  Polynomial.C 4) 
                 (Polynomial.C 2 * Polynomial.X + Polynomial.C 5)

theorem division_quotient_is_correct :
  polynomial_division_quotient =
    Polynomial.C 4 * Polynomial.X ^ 2 +
    Polynomial.C (-2) * Polynomial.X +
    Polynomial.C (3 / 2) :=
by
  sorry

end division_quotient_is_correct_l192_192155


namespace petya_vasya_problem_l192_192703

theorem petya_vasya_problem :
  ∀ n : ℕ, (∀ x : ℕ, x = 12320 * 10 ^ (10 * n + 1) - 1 →
    (∃ p q : ℕ, (p ≠ q ∧ ∀ r : ℕ, (r ∣ x → (r = p ∨ r = q))))) → n = 0 :=
by
  sorry

end petya_vasya_problem_l192_192703


namespace inverse_proportion_function_sol_l192_192634

theorem inverse_proportion_function_sol (k m x : ℝ) (h1 : k ≠ 0) (h2 : (m - 1) * x ^ (m ^ 2 - 2) = k / x) : m = -1 :=
by
  sorry

end inverse_proportion_function_sol_l192_192634


namespace circle_area_ratio_l192_192546

theorem circle_area_ratio (R_C R_D : ℝ)
  (h₁ : (60 / 360 * 2 * Real.pi * R_C) = (40 / 360 * 2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 9 / 4 :=
by 
  sorry

end circle_area_ratio_l192_192546


namespace missing_side_length_of_pan_l192_192884

-- Definition of the given problem's conditions
def pan_side_length := 29
def total_fudge_pieces := 522
def fudge_piece_area := 1

-- Proof statement in Lean 4
theorem missing_side_length_of_pan : 
  (total_fudge_pieces * fudge_piece_area) = (pan_side_length * 18) :=
by
  sorry

end missing_side_length_of_pan_l192_192884


namespace parallel_line_through_A_is_2x_3y_minus_15_line_with_twice_slope_angle_l192_192622

open Real

-- Conditions:
def l1 (x y : ℝ) : Prop := x - 2 * y + 3 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y - 9 = 0
def intersection_point (x y : ℝ) : Prop := l1 x y ∧ l2 x y

-- Point A is the intersection of l1 and l2
def A : ℝ × ℝ := ⟨3, 3⟩

-- Question 1
def line_parallel (x y : ℝ) (c : ℝ) : Prop := 2 * x + 3 * y + c = 0
def line_parallel_passing_through_A : Prop := line_parallel A.1 A.2 (-15)

theorem parallel_line_through_A_is_2x_3y_minus_15 : line_parallel_passing_through_A :=
sorry

-- Question 2
def slope_angle (tan_alpha : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ y, ∃ x, y ≠ 0 ∧ l x y ∧ (tan_alpha = x / y)

def required_slope (tan_alpha : ℝ) : Prop :=
  tan_alpha = 4 / 3

def line_with_slope (x y slope : ℝ) : Prop :=
  y - A.2 = slope * (x - A.1)

def line_with_required_slope : Prop := 
  line_with_slope A.1 A.2 (4 / 3)

theorem line_with_twice_slope_angle : line_with_required_slope :=
sorry

end parallel_line_through_A_is_2x_3y_minus_15_line_with_twice_slope_angle_l192_192622


namespace cube_surface_area_ratio_l192_192362

variable (x : ℝ) (hx : x > 0)

theorem cube_surface_area_ratio (hx : x > 0):
  let side1 := 7 * x
  let side2 := x
  let SA1 := 6 * side1^2
  let SA2 := 6 * side2^2
  (SA1 / SA2) = 49 := 
by 
  sorry

end cube_surface_area_ratio_l192_192362


namespace man_speed_in_still_water_l192_192327

theorem man_speed_in_still_water (upstream_speed downstream_speed : ℝ) (h1 : upstream_speed = 25) (h2 : downstream_speed = 45) :
  (upstream_speed + downstream_speed) / 2 = 35 :=
by
  sorry

end man_speed_in_still_water_l192_192327


namespace abigail_savings_l192_192738

-- Define the parameters for monthly savings and number of months in a year.
def monthlySavings : ℕ := 4000
def numberOfMonthsInYear : ℕ := 12

-- Define the total savings calculation.
def totalSavings (monthlySavings : ℕ) (numberOfMonths : ℕ) : ℕ :=
  monthlySavings * numberOfMonths

-- State the theorem that we need to prove.
theorem abigail_savings : totalSavings monthlySavings numberOfMonthsInYear = 48000 := by
  sorry

end abigail_savings_l192_192738


namespace least_number_to_add_to_246835_l192_192953

-- Define relevant conditions and computations
def lcm_of_169_and_289 : ℕ := Nat.lcm 169 289
def remainder_246835_mod_lcm : ℕ := 246835 % lcm_of_169_and_289
def least_number_to_add : ℕ := lcm_of_169_and_289 - remainder_246835_mod_lcm

-- The theorem statement
theorem least_number_to_add_to_246835 : least_number_to_add = 52 :=
by
  sorry

end least_number_to_add_to_246835_l192_192953


namespace matrix_unique_solution_l192_192137

-- Definitions for the conditions given in the problem
def vec_i : Fin 3 → ℤ := ![1, 0, 0]
def vec_j : Fin 3 → ℤ := ![0, 1, 0]
def vec_k : Fin 3 → ℤ := ![0, 0, 1]

def matrix_M : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![5, -3, 8],
  ![4, 6, -2],
  ![-9, 0, 5]
]

-- Define the target vectors
def target_i : Fin 3 → ℤ := ![5, 4, -9]
def target_j : Fin 3 → ℤ := ![-3, 6, 0]
def target_k : Fin 3 → ℤ := ![8, -2, 5]

-- The statement of the proof
theorem matrix_unique_solution : 
  (matrix_M.mulVec vec_i = target_i) ∧
  (matrix_M.mulVec vec_j = target_j) ∧
  (matrix_M.mulVec vec_k = target_k) :=
  by {
    sorry
  }

end matrix_unique_solution_l192_192137


namespace arithmetic_sequence_has_correct_number_of_terms_l192_192902

theorem arithmetic_sequence_has_correct_number_of_terms :
  ∀ (a₁ d : ℤ) (n : ℕ), a₁ = 1 ∧ d = -2 ∧ (n : ℤ) = (a₁ + (n - 1 : ℕ) * d) → n = 46 := by
  intros a₁ d n
  sorry

end arithmetic_sequence_has_correct_number_of_terms_l192_192902


namespace intersection_M_N_l192_192480

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {x | x ≥ 3}

theorem intersection_M_N : M ∩ N = {3, 4} := 
by
  sorry

end intersection_M_N_l192_192480


namespace solve_equation_l192_192537

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l192_192537


namespace smallest_x_value_min_smallest_x_value_l192_192885

noncomputable def smallest_x_not_defined : ℝ := ( 47 - (Real.sqrt 2041) ) / 12

theorem smallest_x_value :
  ∀ x : ℝ, (6 * x^2 - 47 * x + 7 = 0) → x = smallest_x_not_defined ∨ (x = (47 + (Real.sqrt 2041)) / 12) :=
sorry

theorem min_smallest_x_value :
  smallest_x_not_defined < (47 + (Real.sqrt 2041)) / 12 :=
sorry

end smallest_x_value_min_smallest_x_value_l192_192885


namespace div_difference_l192_192500

theorem div_difference {a b n : ℕ} (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) (h : n ∣ a^n - b^n) :
  n ∣ ((a^n - b^n) / (a - b)) :=
by
  sorry

end div_difference_l192_192500


namespace power_equation_l192_192499

theorem power_equation (m : ℤ) (h : 16 = 2 ^ 4) : (16 : ℝ) ^ (3 / 4) = (2 : ℝ) ^ (m : ℝ) → m = 3 := by
  intros
  sorry

end power_equation_l192_192499


namespace find_x_plus_inv_x_l192_192296

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + (1/x)^3 = 110) : x + (1/x) = 5 :=
sorry

end find_x_plus_inv_x_l192_192296


namespace solution_is_x_l192_192191

def find_x (x : ℝ) : Prop :=
  64 * (x + 1)^3 - 27 = 0

theorem solution_is_x : ∃ x : ℝ, find_x x ∧ x = -1 / 4 :=
by
  sorry

end solution_is_x_l192_192191


namespace selling_price_for_loss_l192_192198

noncomputable def cp : ℝ := 640
def sp1 : ℝ := 768
def sp2 : ℝ := 448
def sp_profitable_sale : ℝ := 832

theorem selling_price_for_loss :
  sp_profitable_sale - cp = cp - sp2 :=
by
  sorry

end selling_price_for_loss_l192_192198


namespace number_of_black_balls_l192_192036

theorem number_of_black_balls
  (total_balls : ℕ)  -- define the total number of balls
  (B : ℕ)            -- define B as the number of black balls
  (prob_red : ℚ := 1/4) -- define the probability of drawing a red ball as 1/4
  (red_balls : ℕ := 3)  -- define the number of red balls as 3
  (h1 : total_balls = red_balls + B) -- total balls is the sum of red and black balls
  (h2 : red_balls / total_balls = prob_red) -- given probability
  : B = 9 :=              -- we need to prove that B is 9
by
  sorry

end number_of_black_balls_l192_192036


namespace inequality_proof_l192_192621

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (h : 1/a + 1/b + 1/c = a + b + c) :
  1/(2*a + b + c)^2 + 1/(2*b + c + a)^2 + 1/(2*c + a + b)^2 ≤ 3/16 :=
by
  sorry

end inequality_proof_l192_192621


namespace problem_solution_l192_192216

theorem problem_solution :
  ∀ x y : ℝ, 9 * y^2 + 6 * x * y + x + 12 = 0 → (x ≤ -3 ∨ x ≥ 4) :=
  sorry

end problem_solution_l192_192216


namespace largest_solution_l192_192652

-- Define the largest solution to the equation |5x - 3| = 28 as 31/5.
theorem largest_solution (x : ℝ) (h : |5 * x - 3| = 28) : x ≤ 31 / 5 := 
  sorry

end largest_solution_l192_192652


namespace point_on_opposite_sides_l192_192658

theorem point_on_opposite_sides (y_0 : ℝ) :
  (2 - 2 * 3 + 5 > 0) ∧ (6 - 2 * y_0 < 0) → y_0 > 3 :=
by
  sorry

end point_on_opposite_sides_l192_192658


namespace bake_four_pans_l192_192503

-- Define the conditions
def bake_time_one_pan : ℕ := 7
def total_bake_time (n : ℕ) : ℕ := 28

-- Define the theorem statement
theorem bake_four_pans : total_bake_time 4 = 28 :=
by
  -- Proof is omitted
  sorry

end bake_four_pans_l192_192503


namespace only_one_true_l192_192219

def statement_dong (xi: Prop) := ¬ xi
def statement_xi (nan: Prop) := ¬ nan
def statement_nan (dong: Prop) := ¬ dong
def statement_bei (nan: Prop) := ¬ (statement_nan nan) 

-- Define the main proof problem assuming all statements
theorem only_one_true : (statement_dong xi → false ∧ statement_xi nan → false ∧ statement_nan dong → true ∧ statement_bei nan → false) 
                        ∨ (statement_dong xi → false ∧ statement_xi nan → true ∧ statement_nan dong → false ∧ statement_bei nan → false) 
                        ∨ (statement_dong xi → true ∧ statement_xi nan → false ∧ statement_nan dong → false ∧ statement_bei nan → true) 
                        ∨ (statement_dong xi → false ∧ statement_xi nan → false ∧ statement_nan dong → false ∧ statement_bei nan → true) 
                        ∧ (statement_nan (statement_dong xi)) = true :=
sorry

end only_one_true_l192_192219


namespace term_2_6_position_l192_192580

theorem term_2_6_position : 
  ∃ (seq : ℕ → ℚ), 
    (seq 23 = 2 / 6) ∧ 
    (∀ n, ∃ k, (n = (k * (k + 1)) / 2 ∧ k > 0 ∧ k <= n)) :=
by sorry

end term_2_6_position_l192_192580


namespace squad_sizes_l192_192544

-- Definitions for conditions
def total_students (x y : ℕ) : Prop := x + y = 146
def equal_after_transfer (x y : ℕ) : Prop := x - 11 = y + 11

-- Theorem to prove the number of students in first and second-year squads
theorem squad_sizes (x y : ℕ) (h1 : total_students x y) (h2 : equal_after_transfer x y) : 
  x = 84 ∧ y = 62 :=
by
  sorry

end squad_sizes_l192_192544


namespace Vann_total_teeth_cleaned_l192_192341

theorem Vann_total_teeth_cleaned :
  let dogs := 7
  let cats := 12
  let pigs := 9
  let horses := 4
  let rabbits := 15
  let dogs_teeth := 42
  let cats_teeth := 30
  let pigs_teeth := 44
  let horses_teeth := 40
  let rabbits_teeth := 28
  (dogs * dogs_teeth) + (cats * cats_teeth) + (pigs * pigs_teeth) + (horses * horses_teeth) + (rabbits * rabbits_teeth) = 1630 :=
by
  sorry

end Vann_total_teeth_cleaned_l192_192341


namespace mark_and_carolyn_total_l192_192817

theorem mark_and_carolyn_total (m c : ℝ) (hm : m = 3 / 4) (hc : c = 3 / 10) :
    m + c = 1.05 :=
by
  sorry

end mark_and_carolyn_total_l192_192817


namespace value_of_a_minus_b_l192_192600

variable {R : Type} [Field R]

noncomputable def f (a b x : R) : R := a * x + b
noncomputable def g (x : R) : R := -2 * x + 7
noncomputable def h (a b x : R) : R := f a b (g x)

theorem value_of_a_minus_b (a b : R) (h_inv : R → R) 
  (h_def : ∀ x, h_inv x = x + 9)
  (h_eq : ∀ x, h a b x = x - 9) : 
  a - b = 5 := by
  sorry

end value_of_a_minus_b_l192_192600


namespace find_m_value_l192_192429

theorem find_m_value
    (x y m : ℝ)
    (hx : x = -1)
    (hy : y = 2)
    (hxy : m * x + 2 * y = 1) :
    m = 3 :=
by
  sorry

end find_m_value_l192_192429


namespace geometric_sequence_sixth_term_l192_192671

variable (a r : ℝ) 

theorem geometric_sequence_sixth_term (h1 : a * (1 + r + r^2 + r^3) = 40)
                                    (h2 : a * r^4 = 32) :
  a * r^5 = 1280 / 15 :=
by sorry

end geometric_sequence_sixth_term_l192_192671


namespace degree_difference_l192_192397

variable (S J : ℕ)

theorem degree_difference :
  S = 150 → S + J = 295 → S - J = 5 :=
by
  intros h₁ h₂
  sorry

end degree_difference_l192_192397


namespace complement_union_l192_192831

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8}) 
  (hM : M = {1, 3, 5, 7}) (hN : N = {5, 6, 7}) : U \ (M ∪ N) = {2, 4, 8} :=
by
  sorry

end complement_union_l192_192831


namespace ramon_3_enchiladas_4_tacos_cost_l192_192907

theorem ramon_3_enchiladas_4_tacos_cost :
  ∃ (e t : ℝ), 2 * e + 3 * t = 2.50 ∧ 3 * e + 2 * t = 2.70 ∧ 3 * e + 4 * t = 3.54 :=
by {
  sorry
}

end ramon_3_enchiladas_4_tacos_cost_l192_192907


namespace my_cousin_reading_time_l192_192813

-- Define the conditions
def reading_time_me_hours : ℕ := 3
def reading_speed_ratio : ℕ := 5
def reading_time_me_min : ℕ := reading_time_me_hours * 60

-- Define the statement to be proved
theorem my_cousin_reading_time : (reading_time_me_min / reading_speed_ratio) = 36 := by
  sorry

end my_cousin_reading_time_l192_192813


namespace symmetric_curve_eq_l192_192639

-- Definitions from the problem conditions
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y + 1) ^ 2 = 1
def line_of_symmetry (x y : ℝ) : Prop := x - y + 3 = 0

-- Problem statement derived from the translation step
theorem symmetric_curve_eq (x y : ℝ) : (x - 2) ^ 2 + (y + 1) ^ 2 = 1 ∧ x - y + 3 = 0 → (x + 4) ^ 2 + (y - 5) ^ 2 = 1 := 
by
  sorry

end symmetric_curve_eq_l192_192639


namespace part1_inequality_part2_range_of_a_l192_192430

-- Definitions and conditions
def f (x a : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- First proof problem for a = 1
theorem part1_inequality (x : ℝ) : f x 1 > 1 ↔ x > 1/2 :=
by sorry

-- Second proof problem for range of a when f(x) > x in (0, 1)
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → f x a > x) → 0 < a ∧ a ≤ 2 :=
by sorry

end part1_inequality_part2_range_of_a_l192_192430


namespace total_cost_sean_bought_l192_192898

theorem total_cost_sean_bought (cost_soda cost_soup cost_sandwich : ℕ) 
  (h_soda : cost_soda = 1)
  (h_soup : cost_soup = 3 * cost_soda)
  (h_sandwich : cost_sandwich = 3 * cost_soup) :
  3 * cost_soda + 2 * cost_soup + cost_sandwich = 18 := 
by
  sorry

end total_cost_sean_bought_l192_192898


namespace option_d_is_correct_l192_192425

theorem option_d_is_correct : (-2 : ℤ) ^ 3 = -8 := by
  sorry

end option_d_is_correct_l192_192425


namespace january_1_is_monday_l192_192856

theorem january_1_is_monday
  (days_in_january : ℕ)
  (mondays_in_january : ℕ)
  (thursdays_in_january : ℕ) :
  days_in_january = 31 ∧ mondays_in_january = 5 ∧ thursdays_in_january = 5 → 
  ∃ d : ℕ, d = 1 ∧ (d % 7 = 1) :=
by
  sorry

end january_1_is_monday_l192_192856


namespace system_solutions_a_l192_192704

theorem system_solutions_a (x y z : ℝ) :
  (2 * x = (y + z) ^ 2) ∧ (2 * y = (z + x) ^ 2) ∧ (2 * z = (x + y) ^ 2) ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solutions_a_l192_192704


namespace gazprom_rnd_costs_calc_l192_192405

theorem gazprom_rnd_costs_calc (R_D_t ΔAPL_t1 : ℝ) (h1 : R_D_t = 3157.61) (h2 : ΔAPL_t1 = 0.69) :
  R_D_t / ΔAPL_t1 = 4576 :=
by
  sorry

end gazprom_rnd_costs_calc_l192_192405


namespace tyler_meals_l192_192840

def num_meals : ℕ := 
  let num_meats := 3
  let num_vegetable_combinations := Nat.choose 5 3
  let num_desserts := 5
  num_meats * num_vegetable_combinations * num_desserts

theorem tyler_meals :
  num_meals = 150 := by
  sorry

end tyler_meals_l192_192840


namespace avg_two_expressions_l192_192232

theorem avg_two_expressions (a : ℝ) (h : ((2 * a + 16) + (3 * a - 8)) / 2 = 84) : a = 32 := sorry

end avg_two_expressions_l192_192232


namespace sum_local_values_of_digits_l192_192181

theorem sum_local_values_of_digits :
  let d2 := 2000
  let d3 := 300
  let d4 := 40
  let d5 := 5
  d2 + d3 + d4 + d5 = 2345 :=
by
  sorry

end sum_local_values_of_digits_l192_192181


namespace andrea_needs_1500_sod_squares_l192_192395

-- Define the measurements of the yard sections
def section1_length : ℕ := 30
def section1_width : ℕ := 40
def section2_length : ℕ := 60
def section2_width : ℕ := 80

-- Define the measurements of the sod square
def sod_length : ℕ := 2
def sod_width : ℕ := 2

-- Compute the areas
def area_section1 : ℕ := section1_length * section1_width
def area_section2 : ℕ := section2_length * section2_width
def total_area : ℕ := area_section1 + area_section2

-- Compute the area of one sod square
def area_sod : ℕ := sod_length * sod_width

-- Compute the number of sod squares needed
def num_sod_squares : ℕ := total_area / area_sod

-- Theorem and proof placeholder
theorem andrea_needs_1500_sod_squares : num_sod_squares = 1500 :=
by {
  -- Place proof here
  sorry
}

end andrea_needs_1500_sod_squares_l192_192395


namespace range_a_mul_b_sub_three_half_l192_192052

theorem range_a_mul_b_sub_three_half (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) (h4 : b = (1 + Real.sqrt 5) / 2 * a) :
  (∃ l u : ℝ, ∀ f, l ≤ f ∧ f < u ↔ f = a * (b - 3 / 2)) :=
sorry

end range_a_mul_b_sub_three_half_l192_192052


namespace simplify_expression_l192_192769

theorem simplify_expression (a : ℝ) (h : a ≠ 0) : (a^9 * a^15) / a^3 = a^21 :=
by sorry

end simplify_expression_l192_192769


namespace percent_of_a_is_4b_l192_192173

variable (a b : ℝ)

theorem percent_of_a_is_4b (hab : a = 1.8 * b) :
  (4 * b / a) * 100 = 222.22 := by
  sorry

end percent_of_a_is_4b_l192_192173


namespace solution_system_inequalities_l192_192897

theorem solution_system_inequalities (x : ℝ) : 
  (x - 4 ≤ 0 ∧ 2 * (x + 1) < 3 * x) ↔ (2 < x ∧ x ≤ 4) := 
sorry

end solution_system_inequalities_l192_192897


namespace mindy_earns_k_times_more_than_mork_l192_192050

-- Given the following conditions:
-- Mork's tax rate: 0.45
-- Mindy's tax rate: 0.25
-- Combined tax rate: 0.29
-- Mindy earns k times more than Mork

theorem mindy_earns_k_times_more_than_mork (M : ℝ) (k : ℝ) (hM : M > 0) :
  (0.45 * M + 0.25 * k * M) / (M * (1 + k)) = 0.29 → k = 4 :=
by
  sorry

end mindy_earns_k_times_more_than_mork_l192_192050


namespace michael_bought_crates_on_thursday_l192_192951

theorem michael_bought_crates_on_thursday :
  ∀ (eggs_per_crate crates_tuesday crates_given current_eggs bought_on_thursday : ℕ),
    crates_tuesday = 6 →
    crates_given = 2 →
    eggs_per_crate = 30 →
    current_eggs = 270 →
    bought_on_thursday = (current_eggs - (crates_tuesday * eggs_per_crate - crates_given * eggs_per_crate)) / eggs_per_crate →
    bought_on_thursday = 5 :=
by
  intros _ _ _ _ _
  sorry

end michael_bought_crates_on_thursday_l192_192951


namespace value_of_other_bills_l192_192878

theorem value_of_other_bills (total_payment : ℕ) (num_fifty_dollar_bills : ℕ) (value_fifty_dollar_bill : ℕ) (num_other_bills : ℕ) 
  (total_fifty_dollars : ℕ) (remaining_payment : ℕ) (value_of_each_other_bill : ℕ) :
  total_payment = 170 →
  num_fifty_dollar_bills = 3 →
  value_fifty_dollar_bill = 50 →
  num_other_bills = 2 →
  total_fifty_dollars = num_fifty_dollar_bills * value_fifty_dollar_bill →
  remaining_payment = total_payment - total_fifty_dollars →
  value_of_each_other_bill = remaining_payment / num_other_bills →
  value_of_each_other_bill = 10 :=
by
  intros t_total_payment t_num_fifty_dollar_bills t_value_fifty_dollar_bill t_num_other_bills t_total_fifty_dollars t_remaining_payment t_value_of_each_other_bill
  sorry

end value_of_other_bills_l192_192878


namespace sheela_monthly_income_l192_192237

variable (deposits : ℝ) (percentage : ℝ) (monthly_income : ℝ)

-- Conditions
axiom deposit_condition : deposits = 3400
axiom percentage_condition : percentage = 0.15
axiom income_condition : deposits = percentage * monthly_income

-- Proof goal
theorem sheela_monthly_income :
  monthly_income = 3400 / 0.15 :=
sorry

end sheela_monthly_income_l192_192237


namespace convert_base_9A3_16_to_4_l192_192666

theorem convert_base_9A3_16_to_4 :
  let h₁ := 9
  let h₂ := 10 -- A in hexadecimal
  let h₃ := 3
  let b₁ := 21 -- h₁ converted to base 4
  let b₂ := 22 -- h₂ converted to base 4
  let b₃ := 3  -- h₃ converted to base 4
  9 * 16^2 + 10 * 16^1 + 3 * 16^0 = 2 * 4^5 + 1 * 4^4 + 2 * 4^3 + 2 * 4^2 + 0 * 4^1 + 3 * 4^0 :=
by
  sorry

end convert_base_9A3_16_to_4_l192_192666


namespace people_present_l192_192959

-- Define the number of parents, pupils, and teachers as constants
def p := 73
def s := 724
def t := 744

-- The theorem to prove the total number of people present
theorem people_present : p + s + t = 1541 := 
by
  -- Proof is inserted here
  sorry

end people_present_l192_192959


namespace simplify_eval_expr_l192_192710

noncomputable def a : ℝ := (Real.sqrt 2) + 1
noncomputable def b : ℝ := (Real.sqrt 2) - 1

theorem simplify_eval_expr (a b : ℝ) (ha : a = (Real.sqrt 2) + 1) (hb : b = (Real.sqrt 2) - 1) : 
  (a^2 - b^2) / a / (a + (2 * a * b + b^2) / a) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_eval_expr_l192_192710


namespace sum_first_6_is_correct_l192_192094

namespace ProofProblem

def sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 1) ∧ ∀ n : ℕ, n ≥ 2 → a (n - 1) = 2 * a n

def sum_first_6 (a : ℕ → ℚ) : ℚ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_first_6_is_correct (a : ℕ → ℚ) (h : sequence a) :
  sum_first_6 a = 63 / 32 :=
sorry

end ProofProblem

end sum_first_6_is_correct_l192_192094


namespace investment_at_6_percent_l192_192832

variables (x y : ℝ)

-- Conditions from the problem
def total_investment : Prop := x + y = 15000
def total_interest : Prop := 0.06 * x + 0.075 * y = 1023

-- Conclusion to prove
def invest_6_percent (x : ℝ) : Prop := x = 6800

theorem investment_at_6_percent (h1 : total_investment x y) (h2 : total_interest x y) : invest_6_percent x :=
by
  sorry

end investment_at_6_percent_l192_192832


namespace crimson_valley_skirts_l192_192923

theorem crimson_valley_skirts (e : ℕ) (a : ℕ) (s : ℕ) (p : ℕ) (c : ℕ) 
  (h1 : e = 120) 
  (h2 : a = 2 * e) 
  (h3 : s = 3 * a / 5) 
  (h4 : p = s / 4) 
  (h5 : c = p / 3) : 
  c = 12 := 
by 
  sorry

end crimson_valley_skirts_l192_192923


namespace car_Z_probability_l192_192866

theorem car_Z_probability :
  let P_X := 1/6
  let P_Y := 1/10
  let P_XYZ := 0.39166666666666666
  ∃ P_Z : ℝ, P_X + P_Y + P_Z = P_XYZ ∧ P_Z = 0.125 :=
by
  sorry

end car_Z_probability_l192_192866


namespace measure_time_with_hourglasses_l192_192217

def hourglass7 : ℕ := 7
def hourglass11 : ℕ := 11
def target_time : ℕ := 15

theorem measure_time_with_hourglasses :
  ∃ (time_elapsed : ℕ), time_elapsed = target_time :=
by
  use 15
  sorry

end measure_time_with_hourglasses_l192_192217


namespace intersection_product_l192_192547

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 - 4 * x + y^2 - 6 * y + 9 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 - 8 * x + y^2 - 6 * y + 25 = 0

-- Define the theorem to prove the product of the coordinates of the intersection points
theorem intersection_product : ∀ x y : ℝ, circle1 x y → circle2 x y → x * y = 12 :=
by
  intro x y h1 h2
  -- Insert proof here
  sorry

end intersection_product_l192_192547


namespace never_sunday_l192_192625

theorem never_sunday (n : ℕ) (days_in_month : ℕ → ℕ) (is_leap_year : Bool) : 
  (∀ (month : ℕ), 1 ≤ month ∧ month ≤ 12 → (days_in_month month = 28 ∨ days_in_month month = 29 ∨ days_in_month month = 30 ∨ days_in_month month = 31) ∧
  (∃ (k : ℕ), k < 7 ∧ ∀ (d : ℕ), d < days_in_month month → (d % 7 = k ↔ n ≠ d))) → n = 31 := 
by
  sorry

end never_sunday_l192_192625


namespace number_of_toddlers_l192_192336

-- Definitions based on the conditions provided in the problem
def total_children := 40
def newborns := 4
def toddlers (T : ℕ) := T
def teenagers (T : ℕ) := 5 * T

-- The theorem to prove
theorem number_of_toddlers : ∃ T : ℕ, newborns + toddlers T + teenagers T = total_children ∧ T = 6 :=
by
  sorry

end number_of_toddlers_l192_192336


namespace dogwood_tree_cut_count_l192_192749

theorem dogwood_tree_cut_count
  (trees_part1 : ℝ) (trees_part2 : ℝ) (trees_left : ℝ)
  (h1 : trees_part1 = 5.0) (h2 : trees_part2 = 4.0)
  (h3 : trees_left = 2.0) :
  trees_part1 + trees_part2 - trees_left = 7.0 :=
by
  sorry

end dogwood_tree_cut_count_l192_192749


namespace remainder_of_8_pow_2023_l192_192309

theorem remainder_of_8_pow_2023 :
  8^2023 % 100 = 12 :=
sorry

end remainder_of_8_pow_2023_l192_192309


namespace infection_average_l192_192934

theorem infection_average (x : ℕ) (h : 1 + x + x * (1 + x) = 196) : x = 13 :=
sorry

end infection_average_l192_192934


namespace triangle_angle_C_l192_192696

theorem triangle_angle_C (A B C : ℝ) (h : A + B = 80) : C = 100 :=
sorry

end triangle_angle_C_l192_192696


namespace sum_even_minus_sum_odd_l192_192606

theorem sum_even_minus_sum_odd :
  let x := (100 / 2) * (2 + 200)
  let y := (100 / 2) * (1 + 199)
  x - y = 100 := by
sorry

end sum_even_minus_sum_odd_l192_192606


namespace average_four_numbers_l192_192485

variable {x : ℝ}

theorem average_four_numbers (h : (15 + 25 + x + 30) / 4 = 23) : x = 22 :=
by
  sorry

end average_four_numbers_l192_192485


namespace yogurt_combinations_l192_192182

theorem yogurt_combinations (flavors toppings : ℕ) (h_flavors : flavors = 5) (h_toppings : toppings = 7) :
  (flavors * Nat.choose toppings 3) = 175 := by
  sorry

end yogurt_combinations_l192_192182


namespace cos_seven_pi_over_six_l192_192363

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := 
by
  sorry

end cos_seven_pi_over_six_l192_192363


namespace find_m_value_l192_192400

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)
def vector_sum (m : ℝ) : ℝ × ℝ := (1 + 3, m - 2)

-- Define the condition that vector_sum is parallel to vector_b
def vectors_parallel (m : ℝ) : Prop :=
  let (x1, y1) := vector_sum m
  let (x2, y2) := vector_b
  x1 * y2 - x2 * y1 = 0

-- The statement to prove
theorem find_m_value : ∃ m : ℝ, vectors_parallel m ∧ m = -2 / 3 :=
by {
  sorry
}

end find_m_value_l192_192400


namespace percentage_increase_l192_192740

theorem percentage_increase (x : ℝ) (h1 : x = 99.9) : 
  ((x - 90) / 90) * 100 = 11 :=
by 
  -- Add the required proof steps here
  sorry

end percentage_increase_l192_192740


namespace calc_value_l192_192514

noncomputable def f : ℝ → ℝ := sorry 

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom non_const_zero : ∃ x : ℝ, f x ≠ 0
axiom functional_eq : ∀ x : ℝ, x * f (x + 1) = (x + 1) * f x

theorem calc_value : f (f (5 / 2)) = 0 :=
sorry

end calc_value_l192_192514


namespace elizabeth_husband_weight_l192_192716

-- Defining the variables for weights of the three wives
variable (s : ℝ) -- Weight of Simona
def elizabeta_weight : ℝ := s + 5
def georgetta_weight : ℝ := s + 10

-- Condition: The total weight of all wives
def total_wives_weight : ℝ := s + elizabeta_weight s + georgetta_weight s

-- Given: The total weight of all wives is 171 kg
def total_wives_weight_cond : Prop := total_wives_weight s = 171

-- Given:
-- Leon weighs the same as his wife.
-- Victor weighs one and a half times more than his wife.
-- Maurice weighs twice as much as his wife.

-- Given: Elizabeth's weight relationship
def elizabeth_weight_cond : Prop := (s + 5 * 1.5) = 85.5

-- Main proof problem:
theorem elizabeth_husband_weight (s : ℝ) (h1: total_wives_weight_cond s) : elizabeth_weight_cond s :=
by
  sorry

end elizabeth_husband_weight_l192_192716


namespace no_integer_pair_satisfies_conditions_l192_192771

theorem no_integer_pair_satisfies_conditions :
  ¬ ∃ (x y : ℤ), x = x^2 + y^2 + 1 ∧ y = 3 * x * y := 
by
  sorry

end no_integer_pair_satisfies_conditions_l192_192771


namespace math_problem_l192_192920

theorem math_problem
  (x : ℝ)
  (h : (1/2) * x - 300 = 350) :
  (x + 200) * 2 = 3000 :=
by
  sorry

end math_problem_l192_192920


namespace magnitude_of_z_l192_192281

open Complex

theorem magnitude_of_z :
  ∃ z : ℂ, (1 + 2 * Complex.I) * z = -1 + 3 * Complex.I ∧ Complex.abs z = Real.sqrt 2 :=
by
  sorry

end magnitude_of_z_l192_192281


namespace increase_in_lines_l192_192729

variable (L : ℝ)
variable (h1 : L + (1 / 3) * L = 240)

theorem increase_in_lines : (240 - L) = 60 := by
  sorry

end increase_in_lines_l192_192729


namespace number_of_beetles_in_sixth_jar_l192_192842

theorem number_of_beetles_in_sixth_jar :
  ∃ (x : ℕ), 
      (x + (x+1) + (x+2) + (x+3) + (x+4) + (x+5) + (x+6) + (x+7) + (x+8) + (x+9) = 150) ∧
      (2 * x ≥ x + 9) ∧
      (x + 5 = 16) :=
by {
  -- This is just the statement, the proof steps are ommited.
  -- You can fill in the proof here using Lean tactics as necessary.
  sorry
}

end number_of_beetles_in_sixth_jar_l192_192842


namespace real_number_a_l192_192783

theorem real_number_a (a : ℝ) (ha : ∃ b : ℝ, z = 0 + bi) : a = 1 :=
sorry

end real_number_a_l192_192783


namespace man_speed_l192_192732

theorem man_speed (train_length : ℝ) (time_to_cross : ℝ) (train_speed_kmph : ℝ) 
  (h1 : train_length = 100) (h2 : time_to_cross = 6) (h3 : train_speed_kmph = 54.99520038396929) : 
  ∃ man_speed : ℝ, man_speed = 16.66666666666667 - 15.27644455165814 :=
by sorry

end man_speed_l192_192732


namespace sum_of_cubes_l192_192022

theorem sum_of_cubes
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y)
  (h1 : (x + y)^2 = 2500) 
  (h2 : x * y = 500) :
  x^3 + y^3 = 50000 := 
by
  sorry

end sum_of_cubes_l192_192022


namespace quadrilateral_is_parallelogram_l192_192721

theorem quadrilateral_is_parallelogram
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 - 2 * a * c - 2 * b * d = 0) :
  (a = c) ∧ (b = d) :=
by {
  sorry
}

end quadrilateral_is_parallelogram_l192_192721


namespace right_triangle_angles_l192_192302

theorem right_triangle_angles (α β : ℝ) (h : α + β = 90) 
  (h_ratio : (180 - α) / (90 + α) = 9 / 11) : 
  (α = 58.5 ∧ β = 31.5) :=
by sorry

end right_triangle_angles_l192_192302


namespace students_both_courses_l192_192097

-- Definitions from conditions
def total_students : ℕ := 87
def students_french : ℕ := 41
def students_german : ℕ := 22
def students_neither : ℕ := 33

-- The statement we need to prove
theorem students_both_courses : (students_french + students_german - 9 + students_neither = total_students) → (9 = 96 - total_students) :=
by
  -- The proof would go here, but we leave it as sorry for now
  sorry

end students_both_courses_l192_192097


namespace calculation_not_minus_one_l192_192416

theorem calculation_not_minus_one :
  (-1 : ℤ) * 1 ≠ 1 ∧
  (-1 : ℤ) / (-1) = 1 ∧
  (-2015 : ℤ) / 2015 ≠ 1 ∧
  (-1 : ℤ)^9 * (-1 : ℤ)^2 ≠ 1 := by 
  sorry

end calculation_not_minus_one_l192_192416


namespace prime_square_minus_one_divisible_by_24_l192_192995

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p) : 
  ∃ k : ℤ, p^2 - 1 = 24 * k :=
  sorry

end prime_square_minus_one_divisible_by_24_l192_192995


namespace ratio_of_volumes_l192_192146

-- Definitions based on given conditions
def V1 : ℝ := sorry -- Volume of the first vessel
def V2 : ℝ := sorry -- Volume of the second vessel

-- Given condition
def condition : Prop := (3 / 4) * V1 = (5 / 8) * V2

-- The theorem to prove the ratio V1 / V2 is 5 / 6
theorem ratio_of_volumes (h : condition) : V1 / V2 = 5 / 6 :=
sorry

end ratio_of_volumes_l192_192146


namespace gcd_8994_13326_37566_l192_192515

-- Define the integers involved
def a := 8994
def b := 13326
def c := 37566

-- Assert the GCD relation
theorem gcd_8994_13326_37566 : Int.gcd a (Int.gcd b c) = 2 := by
  sorry

end gcd_8994_13326_37566_l192_192515


namespace greatest_four_digit_divisible_by_3_5_6_l192_192781

theorem greatest_four_digit_divisible_by_3_5_6 : 
  ∃ n, n ≤ 9999 ∧ n ≥ 1000 ∧ (∀ m, m ≤ 9999 ∧ m ≥ 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n) ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n = 9990 :=
by 
  sorry

end greatest_four_digit_divisible_by_3_5_6_l192_192781


namespace no_solutions_exists_unique_l192_192428

def is_solution (a b c x y z : ℤ) : Prop :=
  2 * x - b * y + z = 2 * b ∧
  a * x + 5 * y - c * z = a

def no_solutions_for (a b c : ℤ) : Prop :=
  ∀ x y z : ℤ, ¬ is_solution a b c x y z

theorem no_solutions_exists_unique (a b c : ℤ) :
  (a = -2 ∧ b = 5 ∧ c = 1) ∨
  (a = 2 ∧ b = -5 ∧ c = -1) ∨
  (a = 10 ∧ b = -1 ∧ c = -5) ↔
  no_solutions_for a b c := 
sorry

end no_solutions_exists_unique_l192_192428


namespace find_correct_speed_l192_192597

variables (d t : ℝ) -- Defining distance and time as real numbers

theorem find_correct_speed
  (h1 : d = 30 * (t + 5 / 60))
  (h2 : d = 50 * (t - 5 / 60)) :
  ∃ r : ℝ, r = 37.5 ∧ d = r * t :=
by 
  -- Skip the proof for now
  sorry

end find_correct_speed_l192_192597


namespace part1_part2_l192_192516

open Set

variable {m x : ℝ}

def A (m : ℝ) : Set ℝ := { x | x^2 - (m+1)*x + m = 0 }
def B (m : ℝ) : Set ℝ := { x | x * m - 1 = 0 }

theorem part1 (h : A m ⊆ B m) : m = 1 :=
by
  sorry

theorem part2 (h : B m ⊂ A m) : m = 0 ∨ m = -1 :=
by
  sorry

end part1_part2_l192_192516


namespace fraction_exp_3_4_cubed_l192_192016

def fraction_exp (a b n : ℕ) : ℚ := (a : ℚ) ^ n / (b : ℚ) ^ n

theorem fraction_exp_3_4_cubed : fraction_exp 3 4 3 = 27 / 64 :=
by
  sorry

end fraction_exp_3_4_cubed_l192_192016


namespace parking_average_cost_l192_192678

noncomputable def parking_cost_per_hour := 
  let cost_two_hours : ℝ := 20.00
  let cost_per_excess_hour : ℝ := 1.75
  let weekend_surcharge : ℝ := 5.00
  let discount_rate : ℝ := 0.10
  let total_hours : ℝ := 9.00
  let excess_hours : ℝ := total_hours - 2.00
  let remaining_cost := cost_per_excess_hour * excess_hours
  let total_cost_before_discount := cost_two_hours + remaining_cost + weekend_surcharge
  let discount := discount_rate * total_cost_before_discount
  let discounted_total_cost := total_cost_before_discount - discount
  let average_cost_per_hour := discounted_total_cost / total_hours
  average_cost_per_hour

theorem parking_average_cost :
  parking_cost_per_hour = 3.725 := 
by
  sorry

end parking_average_cost_l192_192678


namespace union_is_faction_l192_192698

variable {D : Type} (is_faction : Set D → Prop)
variable (A B : Set D)

-- Define the complement
def complement (S : Set D) : Set D := {x | x ∉ S}

-- State the given condition
axiom faction_complement_union (A B : Set D) : 
  is_faction A → is_faction B → is_faction (complement (A ∪ B))

-- The theorem to prove
theorem union_is_faction (A B : Set D) :
  is_faction A → is_faction B → is_faction (A ∪ B) := 
by
  -- Proof goes here
  sorry

end union_is_faction_l192_192698


namespace increased_percentage_l192_192942

theorem increased_percentage (P : ℝ) (N : ℝ) (hN : N = 80) 
  (h : (N + (P / 100) * N) - (N - (25 / 100) * N) = 30) : P = 12.5 := 
by 
  sorry

end increased_percentage_l192_192942


namespace total_cups_for_8_batches_l192_192361

def cups_of_flour (batches : ℕ) : ℝ := 4 * batches
def cups_of_sugar (batches : ℕ) : ℝ := 1.5 * batches
def total_cups (batches : ℕ) : ℝ := cups_of_flour batches + cups_of_sugar batches

theorem total_cups_for_8_batches : total_cups 8 = 44 := 
by
  -- This is where the proof would go
  sorry

end total_cups_for_8_batches_l192_192361


namespace largest_integer_in_mean_set_l192_192681

theorem largest_integer_in_mean_set :
  ∃ (A B C D : ℕ), 
    A < B ∧ B < C ∧ C < D ∧
    (A + B + C + D) = 4 * 68 ∧
    A ≥ 5 ∧
    D = 254 :=
sorry

end largest_integer_in_mean_set_l192_192681


namespace alice_is_10_years_older_l192_192289

-- Problem definitions
variables (A B : ℕ)

-- Conditions of the problem
def condition1 := A + 5 = 19
def condition2 := A + 6 = 2 * (B + 6)

-- Question to prove
theorem alice_is_10_years_older (h1 : condition1 A) (h2 : condition2 A B) : A - B = 10 := 
by
  sorry

end alice_is_10_years_older_l192_192289


namespace ducks_counted_l192_192063

theorem ducks_counted (x y : ℕ) (h1 : x + y = 300) (h2 : 2 * x + 4 * y = 688) : x = 256 :=
by
  sorry

end ducks_counted_l192_192063


namespace temperature_on_friday_l192_192306

def temperatures (M T W Th F : ℝ) : Prop :=
  (M + T + W + Th) / 4 = 48 ∧
  (T + W + Th + F) / 4 = 40 ∧
  M = 42

theorem temperature_on_friday (M T W Th F : ℝ) (h : temperatures M T W Th F) : 
  F = 10 :=
  by
    -- problem statement
    sorry

end temperature_on_friday_l192_192306


namespace cos_identity_l192_192553

theorem cos_identity (θ : ℝ) (h : Real.cos (π / 6 + θ) = (Real.sqrt 3) / 3) : 
  Real.cos (5 * π / 6 - θ) = - (Real.sqrt 3 / 3) :=
by
  sorry

end cos_identity_l192_192553


namespace carpooling_plans_l192_192030

def last_digits (jia : ℕ) (friend1 : ℕ) (friend2 : ℕ) (friend3 : ℕ) (friend4 : ℕ) : Prop :=
  jia = 0 ∧ friend1 = 0 ∧ friend2 = 2 ∧ friend3 = 1 ∧ friend4 = 5

def total_car_plans : Prop :=
  ∀ (jia friend1 friend2 friend3 friend4 : ℕ),
    last_digits jia friend1 friend2 friend3 friend4 →
    (∃ num_ways : ℕ, num_ways = 64)

theorem carpooling_plans : total_car_plans :=
sorry

end carpooling_plans_l192_192030


namespace student_needs_33_percent_to_pass_l192_192566

-- Define the conditions
def obtained_marks : ℕ := 125
def failed_by : ℕ := 40
def max_marks : ℕ := 500

-- The Lean statement to prove the required percentage
theorem student_needs_33_percent_to_pass : (obtained_marks + failed_by) * 100 / max_marks = 33 := by
  sorry

end student_needs_33_percent_to_pass_l192_192566


namespace total_legs_in_household_l192_192674

def number_of_legs (humans children dogs cats : ℕ) (human_legs child_legs dog_legs cat_legs : ℕ) : ℕ :=
  humans * human_legs + children * child_legs + dogs * dog_legs + cats * cat_legs

theorem total_legs_in_household : number_of_legs 2 3 2 1 2 2 4 4 = 22 :=
  by
    -- The statement ensures the total number of legs is 22, given the defined conditions.
    sorry

end total_legs_in_household_l192_192674


namespace part1_part2_l192_192434

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - (a / 2) * x^2

-- Define the line l
noncomputable def l (k : ℤ) (x : ℝ) : ℝ := (k - 2) * x - k + 1

-- Theorem for part (1)
theorem part1 (x : ℝ) (a : ℝ) (h₁ : e ≤ x) (h₂ : x ≤ e^2) (h₃ : f a x > 0) : a < 2 / e :=
sorry

-- Theorem for part (2)
theorem part2 (k : ℤ) (h₁ : a = 0) (h₂ : ∀ (x : ℝ), 1 < x → f 0 x > l k x) : k ≤ 4 :=
sorry

end part1_part2_l192_192434


namespace greatest_good_t_l192_192936

noncomputable def S (a t : ℕ) : Set ℕ := {x | ∃ n : ℕ, x = a + 1 + n ∧ n < t}

def is_good (S : Set ℕ) (k : ℕ) : Prop :=
∃ (coloring : ℕ → Fin k), ∀ (x y : ℕ), x ≠ y → x + y ∈ S → coloring x ≠ coloring y

theorem greatest_good_t {k : ℕ} (hk : k > 1) : ∃ t, ∀ a, is_good (S a t) k ∧ 
  ∀ t' > t, ¬ ∀ a, is_good (S a t') k := 
sorry

end greatest_good_t_l192_192936


namespace sum_of_sides_eq_13_or_15_l192_192185

noncomputable def squares_side_lengths (b d : ℕ) : Prop :=
  15^2 = b^2 + 10^2 + d^2

theorem sum_of_sides_eq_13_or_15 :
  ∃ b d : ℕ, squares_side_lengths b d ∧ (b + d = 13 ∨ b + d = 15) :=
sorry

end sum_of_sides_eq_13_or_15_l192_192185


namespace solution_set_ineq_l192_192178

theorem solution_set_ineq (x : ℝ) : 
  (x - 1) / (2 * x + 3) > 1 ↔ -4 < x ∧ x < -3 / 2 :=
by
  sorry

end solution_set_ineq_l192_192178


namespace lizard_eyes_fewer_than_spots_and_wrinkles_l192_192835

noncomputable def lizard_problem : Nat :=
  let eyes_jan := 3
  let wrinkles_jan := 3 * eyes_jan
  let spots_jan := 7 * (wrinkles_jan ^ 2)
  let eyes_cousin := 3
  let wrinkles_cousin := 2 * eyes_cousin
  let spots_cousin := 5 * (wrinkles_cousin ^ 2)
  let total_eyes := eyes_jan + eyes_cousin
  let total_wrinkles := wrinkles_jan + wrinkles_cousin
  let total_spots := spots_jan + spots_cousin
  (total_spots + total_wrinkles) - total_eyes

theorem lizard_eyes_fewer_than_spots_and_wrinkles :
  lizard_problem = 756 :=
by
  sorry

end lizard_eyes_fewer_than_spots_and_wrinkles_l192_192835


namespace four_digit_number_count_l192_192471

theorem four_digit_number_count (A : ℕ → ℕ → ℕ)
  (odd_digits even_digits : Finset ℕ)
  (odds : ∀ x ∈ odd_digits, x % 2 = 1)
  (evens : ∀ x ∈ even_digits, x % 2 = 0) :
  odd_digits = {1, 3, 5, 7, 9} ∧ 
  even_digits = {2, 4, 6, 8} →
  A 5 2 * A 7 2 = 840 :=
by
  intros h1
  sorry

end four_digit_number_count_l192_192471


namespace unique_solution_for_y_l192_192604

def operation (x y : ℝ) : ℝ := 4 * x - 2 * y + x^2 * y

theorem unique_solution_for_y : ∃! (y : ℝ), operation 3 y = 20 :=
by {
  sorry
}

end unique_solution_for_y_l192_192604


namespace correct_average_marks_l192_192180

theorem correct_average_marks 
  (n : ℕ) (wrong_avg : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ)
  (h1 : n = 10)
  (h2 : wrong_avg = 100)
  (h3 : wrong_mark = 90)
  (h4 : correct_mark = 10) :
  (n * wrong_avg - wrong_mark + correct_mark) / n = 92 :=
by
  sorry

end correct_average_marks_l192_192180


namespace part1_part2_l192_192409

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 / (3 ^ x + 1) + a

theorem part1 (h : ∀ x : ℝ, f (-x) a = -f x a) : a = -1 :=
by sorry

noncomputable def f' (x : ℝ) : ℝ := 2 / (3 ^ x + 1) - 1

theorem part2 : ∀ t : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f' x + 1 = t ↔ 1 / 2 ≤ t ∧ t ≤ 1 :=
by sorry

end part1_part2_l192_192409


namespace asymptotes_of_hyperbola_l192_192682

-- Definition of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 16 - (y^2) / 9 = 1

-- Definition of the equations of the asymptotes
def asymptote_eq (x y : ℝ) : Prop := y = (3/4)*x ∨ y = -(3/4)*x

-- Theorem statement
theorem asymptotes_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_eq x y → asymptote_eq x y :=
sorry

end asymptotes_of_hyperbola_l192_192682


namespace valve_XY_time_correct_l192_192001

-- Given conditions
def valve_rates (x y z : ℝ) := (x + y + z = 1/2 ∧ x + z = 1/4 ∧ y + z = 1/3)
def total_fill_time (t : ℝ) (x y : ℝ) := t = 1 / (x + y)

-- The proof problem
theorem valve_XY_time_correct (x y z : ℝ) (t : ℝ) 
  (h : valve_rates x y z) : total_fill_time t x y → t = 2.4 :=
by
  -- Assume h defines the rates
  have h1 : x + y + z = 1/2 := h.1
  have h2 : x + z = 1/4 := h.2.1
  have h3 : y + z = 1/3 := h.2.2
  
  sorry

end valve_XY_time_correct_l192_192001


namespace find_x_l192_192635

theorem find_x (x : ℚ) (h1 : 8 * x^2 + 9 * x - 2 = 0) (h2 : 16 * x^2 + 35 * x - 4 = 0) : 
  x = 1 / 8 :=
by sorry

end find_x_l192_192635


namespace absolute_difference_AB_l192_192438

noncomputable def A : Real := 12 / 7
noncomputable def B : Real := 20 / 7

theorem absolute_difference_AB : |A - B| = 8 / 7 := by
  sorry

end absolute_difference_AB_l192_192438


namespace square_pyramid_sum_l192_192422

-- Define the number of faces, edges, and vertices of a square pyramid.
def faces_square_base : Nat := 1
def faces_lateral : Nat := 4
def edges_base : Nat := 4
def edges_lateral : Nat := 4
def vertices_base : Nat := 4
def vertices_apex : Nat := 1

-- Summing the faces, edges, and vertices
def total_faces : Nat := faces_square_base + faces_lateral
def total_edges : Nat := edges_base + edges_lateral
def total_vertices : Nat := vertices_base + vertices_apex

theorem square_pyramid_sum : (total_faces + total_edges + total_vertices = 18) :=
by
  sorry

end square_pyramid_sum_l192_192422


namespace victoria_donuts_cost_l192_192334

theorem victoria_donuts_cost (n : ℕ) (cost_per_dozen : ℝ) (total_donuts_needed : ℕ) 
  (dozens_needed : ℕ) (actual_total_donuts : ℕ) (total_cost : ℝ) :
  total_donuts_needed ≥ 550 ∧ cost_per_dozen = 7.49 ∧ (total_donuts_needed = 12 * dozens_needed) ∧
  (dozens_needed = Nat.ceil (total_donuts_needed / 12)) ∧ 
  (actual_total_donuts = 12 * dozens_needed) ∧ actual_total_donuts ≥ 550 ∧ 
  (total_cost = dozens_needed * cost_per_dozen) →
  total_cost = 344.54 :=
by
  sorry

end victoria_donuts_cost_l192_192334


namespace biology_to_general_ratio_l192_192279

variable (g b m : ℚ)

theorem biology_to_general_ratio (h1 : g = 30) 
                                (h2 : m = (3/5) * (g + b)) 
                                (h3 : g + b + m = 144) : 
                                b / g = 2 / 1 := 
by 
  sorry

end biology_to_general_ratio_l192_192279


namespace range_of_a_l192_192227

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 4 → (x^2 + 2 * (a - 1) * x + 2) ≥ (y^2 + 2 * (a - 1) * y + 2)) →
  a ≤ -3 :=
by
  sorry

end range_of_a_l192_192227


namespace Chad_saves_40_percent_of_his_earnings_l192_192935

theorem Chad_saves_40_percent_of_his_earnings :
  let earnings_mow := 600
  let earnings_birthday := 250
  let earnings_games := 150
  let earnings_oddjobs := 150
  let amount_saved := 460
  (amount_saved / (earnings_mow + earnings_birthday + earnings_games + earnings_oddjobs) * 100) = 40 :=
by
  sorry

end Chad_saves_40_percent_of_his_earnings_l192_192935


namespace marble_weights_total_l192_192149

theorem marble_weights_total:
  0.33 + 0.33 + 0.08 + 0.25 + 0.02 + 0.12 + 0.15 = 1.28 :=
by {
  sorry
}

end marble_weights_total_l192_192149


namespace unique_k_satisfying_eq_l192_192059

theorem unique_k_satisfying_eq (k : ℤ) :
  (∀ a b c : ℝ, (a + b + c) * (a * b + b * c + c * a) + k * a * b * c = (a + b) * (b + c) * (c + a)) ↔ k = -1 :=
sorry

end unique_k_satisfying_eq_l192_192059


namespace problem_solution_l192_192195

theorem problem_solution (a d e : ℕ) (ha : 0 < a ∧ a < 10) (hd : 0 < d ∧ d < 10) (he : 0 < e ∧ e < 10) :
  ((10 * a + d) * (10 * a + e) = 100 * a ^ 2 + 110 * a + d * e) ↔ (d + e = 11) := by
  sorry

end problem_solution_l192_192195


namespace find_d_l192_192464

theorem find_d (c a m d : ℝ) (h : m = (c * a * d) / (a - d)) : d = (m * a) / (m + c * a) :=
by sorry

end find_d_l192_192464


namespace sin_360_eq_0_l192_192254

theorem sin_360_eq_0 : Real.sin (360 * Real.pi / 180) = 0 := by
  sorry

end sin_360_eq_0_l192_192254


namespace pumpkin_pie_degrees_l192_192375

theorem pumpkin_pie_degrees (total_students : ℕ) (peach_pie : ℕ) (apple_pie : ℕ) (blueberry_pie : ℕ)
                               (pumpkin_pie : ℕ) (banana_pie : ℕ)
                               (h_total : total_students = 40)
                               (h_peach : peach_pie = 14)
                               (h_apple : apple_pie = 9)
                               (h_blueberry : blueberry_pie = 7)
                               (h_remaining : pumpkin_pie = banana_pie)
                               (h_half_remaining : 2 * pumpkin_pie = 40 - (peach_pie + apple_pie + blueberry_pie)) :
  (pumpkin_pie * 360) / total_students = 45 := by
sorry

end pumpkin_pie_degrees_l192_192375


namespace abs_eq_sets_l192_192261

theorem abs_eq_sets (x : ℝ) : 
  (|x - 25| + |x - 15| = |2 * x - 40|) → (x ≤ 15 ∨ x ≥ 25) :=
by
  sorry

end abs_eq_sets_l192_192261


namespace tangent_line_circle_l192_192102

theorem tangent_line_circle (r : ℝ) (h : r > 0) :
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = r^2 → x + y = 2 * r) ↔ r = 2 + Real.sqrt 2 :=
by
  sorry

end tangent_line_circle_l192_192102


namespace select_4_officers_from_7_members_l192_192873

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Statement of the problem
theorem select_4_officers_from_7_members : binom 7 4 = 35 :=
by
  -- Proof not required, so we use sorry to skip it
  sorry

end select_4_officers_from_7_members_l192_192873


namespace water_added_is_five_l192_192702

theorem water_added_is_five :
  ∃ W x : ℝ, (4 / 3 = 10 / W) ∧ (4 / 5 = 10 / (W + x)) ∧ x = 5 := by
  sorry

end water_added_is_five_l192_192702


namespace sector_area_l192_192444

theorem sector_area (s θ r : ℝ) (hs : s = 4) (hθ : θ = 2) (hr : r = s / θ) : (1/2) * r^2 * θ = 4 := by
  sorry

end sector_area_l192_192444


namespace smallest_prime_with_prime_digit_sum_l192_192340

def is_prime (n : ℕ) : Prop := ¬ ∃ m, m ∣ n ∧ 1 < m ∧ m < n

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_prime_digit_sum :
  ∃ p, is_prime p ∧ is_prime (digit_sum p) ∧ 10 < digit_sum p ∧ p = 29 :=
by
  sorry

end smallest_prime_with_prime_digit_sum_l192_192340


namespace wall_cost_equal_l192_192129

theorem wall_cost_equal (A B C : ℝ) (d_1 d_2 : ℝ) (h1 : A = B) (h2 : B = C) : d_1 = d_2 :=
by
  -- sorry is used to skip the proof
  sorry

end wall_cost_equal_l192_192129


namespace train_total_distance_l192_192598

theorem train_total_distance (x : ℝ) (h1 : x > 0) 
  (h_speed_avg : 48 = ((3 * x) / (x / 8))) : 
  3 * x = 6 := 
by
  sorry

end train_total_distance_l192_192598


namespace tan_alpha_value_complex_expression_value_l192_192095

theorem tan_alpha_value (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : Real.tan α = -1 / 3 :=
sorry

theorem complex_expression_value 
(α : ℝ) 
(h1 : Real.tan (π / 4 + α) = 1 / 2) 
(h2 : Real.tan α = -1 / 3) : 
Real.sin (2 * α + 2 * π) - (Real.sin (π / 2 - α))^2 / 
(1 - Real.cos (π - 2 * α) + (Real.sin α)^2) = -15 / 19 :=
sorry

end tan_alpha_value_complex_expression_value_l192_192095


namespace circle_radius_tangents_l192_192065

theorem circle_radius_tangents
  (AB CD EF r : ℝ)
  (circle_tangent_AB : AB = 5)
  (circle_tangent_CD : CD = 11)
  (circle_tangent_EF : EF = 15) :
  r = 2.5 := by
  sorry

end circle_radius_tangents_l192_192065


namespace system_of_equations_solution_l192_192024

theorem system_of_equations_solution (x y z : ℝ) :
  (4 * x^2 / (1 + 4 * x^2) = y ∧
   4 * y^2 / (1 + 4 * y^2) = z ∧
   4 * z^2 / (1 + 4 * z^2) = x) →
  ((x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨ (x = 0 ∧ y = 0 ∧ z = 0)) :=
by
  sorry

end system_of_equations_solution_l192_192024


namespace long_letter_time_ratio_l192_192777

-- Definitions based on conditions
def letters_per_month := (30 / 3 : Nat)
def regular_letter_pages := (20 / 10 : Nat)
def total_regular_pages := letters_per_month * regular_letter_pages
def long_letter_pages := 24 - total_regular_pages

-- Define the times and calculate the ratios
def time_spent_per_page_regular := (20 / regular_letter_pages : Nat)
def time_spent_per_page_long := (80 / long_letter_pages : Nat)
def time_ratio := time_spent_per_page_long / time_spent_per_page_regular

-- Theorem to prove the ratio
theorem long_letter_time_ratio : time_ratio = 2 := by
  sorry

end long_letter_time_ratio_l192_192777


namespace total_marbles_proof_l192_192287

def dan_violet_marbles : Nat := 64
def mary_red_marbles : Nat := 14
def john_blue_marbles (x : Nat) : Nat := x

def total_marble (x : Nat) : Nat := dan_violet_marbles + mary_red_marbles + john_blue_marbles x

theorem total_marbles_proof (x : Nat) : total_marble x = 78 + x := by
  sorry

end total_marbles_proof_l192_192287


namespace number_of_bouncy_balls_per_package_l192_192345

theorem number_of_bouncy_balls_per_package (x : ℕ) (h : 4 * x + 8 * x + 4 * x = 160) : x = 10 :=
by
  sorry

end number_of_bouncy_balls_per_package_l192_192345


namespace gift_card_value_l192_192374

def latte_cost : ℝ := 3.75
def croissant_cost : ℝ := 3.50
def daily_treat_cost : ℝ := latte_cost + croissant_cost
def weekly_treat_cost : ℝ := daily_treat_cost * 7

def cookie_cost : ℝ := 1.25
def total_cookie_cost : ℝ := cookie_cost * 5

def total_spent : ℝ := weekly_treat_cost + total_cookie_cost
def remaining_balance : ℝ := 43.00

theorem gift_card_value : (total_spent + remaining_balance) = 100 := 
by sorry

end gift_card_value_l192_192374


namespace intersection_of_A_and_B_l192_192002

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end intersection_of_A_and_B_l192_192002


namespace birch_trees_probability_l192_192568

/--
A gardener plants four pine trees, five oak trees, and six birch trees in a row. He plants them in random order, each arrangement being equally likely.
Prove that no two birch trees are next to one another is \(\frac{2}{45}\).
--/
theorem birch_trees_probability: (∃ (m n : ℕ), (m = 2) ∧ (n = 45) ∧ (no_two_birch_trees_adjacent_probability = m / n)) := 
sorry

end birch_trees_probability_l192_192568


namespace catalyst_second_addition_is_882_l192_192728

-- Constants for the problem
def lower_bound : ℝ := 500
def upper_bound : ℝ := 1500
def golden_ratio_method : ℝ := 0.618

-- Calculated values
def first_addition : ℝ := lower_bound + golden_ratio_method * (upper_bound - lower_bound)
def second_bound : ℝ := first_addition - lower_bound
def second_addition : ℝ := lower_bound + golden_ratio_method * second_bound

theorem catalyst_second_addition_is_882 :
  lower_bound = 500 → upper_bound = 1500 → golden_ratio_method = 0.618 → second_addition = 882 := by
  -- Proof goes here
  sorry

end catalyst_second_addition_is_882_l192_192728


namespace slope_angle_tangent_line_at_zero_l192_192461

noncomputable def curve (x : ℝ) : ℝ := 2 * x - Real.exp x

noncomputable def slope_at (x : ℝ) : ℝ := 
  (deriv curve) x

theorem slope_angle_tangent_line_at_zero : 
  Real.arctan (slope_at 0) = Real.pi / 4 :=
by
  sorry

end slope_angle_tangent_line_at_zero_l192_192461


namespace number_of_trees_l192_192979

theorem number_of_trees (l d : ℕ) (h_l : l = 441) (h_d : d = 21) : (l / d) + 1 = 22 :=
by
  sorry

end number_of_trees_l192_192979


namespace repeating_decimal_as_fraction_l192_192207

/-- Define x as the repeating decimal 7.182182... -/
def x : ℚ := 
  7 + 182 / 999

/-- Define y as the fraction 7175/999 -/
def y : ℚ := 
  7175 / 999

/-- Theorem stating that the repeating decimal 7.182182... is equal to the fraction 7175/999 -/
theorem repeating_decimal_as_fraction : x = y :=
sorry

end repeating_decimal_as_fraction_l192_192207


namespace nth_equation_l192_192460

theorem nth_equation (n : ℕ) (h : 0 < n) : (10 * n + 5) ^ 2 = n * (n + 1) * 100 + 5 ^ 2 := 
sorry

end nth_equation_l192_192460


namespace continuous_linear_function_l192_192266

theorem continuous_linear_function {f : ℝ → ℝ} (h_cont : Continuous f) 
  (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_a_half : a < 1/2) (h_b_half : b < 1/2) 
  (h_eq : ∀ x : ℝ, f (f x) = a * f x + b * x) : 
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ (k * k - a * k - b = 0) := 
sorry

end continuous_linear_function_l192_192266


namespace part1_part2_l192_192978

variable (a b c : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (h_sum : a + b + c = 1)

theorem part1 : 2 * a * b + b * c + c * a + c ^ 2 / 2 ≤ 1 / 2 := sorry

theorem part2 : (a ^ 2 + c ^ 2) / b + (b ^ 2 + a ^ 2) / c + (c ^ 2 + b ^ 2) / a ≥ 2 := sorry

end part1_part2_l192_192978


namespace math_proof_problem_l192_192653

theorem math_proof_problem
  (n m k l : ℕ)
  (hpos_n : n > 0)
  (hpos_m : m > 0)
  (hpos_k : k > 0)
  (hpos_l : l > 0)
  (hneq_n : n ≠ 1)
  (hdiv : n^k + m*n^l + 1 ∣ n^(k+l) - 1) :
  (m = 1 ∧ l = 2*k) ∨ (l ∣ k ∧ m = (n^(k-l) - 1) / (n^l - 1)) :=
by 
  sorry

end math_proof_problem_l192_192653


namespace problem_l192_192841

theorem problem (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, f (x + 2) = -f x) :
  f 4 = 0 ∧ (∀ x, f (x + 4) = f x) ∧ (∀ x, f (2 - x) = f (2 + x)) :=
sorry

end problem_l192_192841


namespace tilly_bag_cost_l192_192087

noncomputable def cost_per_bag (n s P τ F : ℕ) : ℕ :=
  let revenue := n * s
  let total_sales_tax := n * (s * τ / 100)
  let total_additional_expenses := total_sales_tax + F
  (revenue - (P + total_additional_expenses)) / n

theorem tilly_bag_cost :
  let n := 100
  let s := 10
  let P := 300
  let τ := 5
  let F := 50
  cost_per_bag n s P τ F = 6 :=
  by
    let n := 100
    let s := 10
    let P := 300
    let τ := 5
    let F := 50
    have : cost_per_bag n s P τ F = 6 := sorry
    exact this

end tilly_bag_cost_l192_192087


namespace range_of_m_decreasing_l192_192238

theorem range_of_m_decreasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (m - 3) * x₁ + 5 > (m - 3) * x₂ + 5) ↔ m < 3 :=
by
  sorry

end range_of_m_decreasing_l192_192238


namespace pedestrian_speeds_unique_l192_192871

variables 
  (x y : ℝ)
  (d : ℝ := 105)  -- Distance between cities
  (t1 : ℝ := 7.5) -- Time for current speeds
  (t2 : ℝ := 105 / 13) -- Time for adjusted speeds

theorem pedestrian_speeds_unique :
  (x + y = 14) →
  (3 * x + y = 14) →
  x = 6 ∧ y = 8 :=
by
  intros h1 h2
  have : 2 * x = 12 :=
    by ring_nf; sorry
  have hx : x = 6 :=
    by linarith
  have hy : y = 8 :=
    by linarith
  exact ⟨hx, hy⟩

end pedestrian_speeds_unique_l192_192871


namespace commuting_days_l192_192385

theorem commuting_days 
  (a b c d x : ℕ)
  (cond1 : b + c = 12)
  (cond2 : a + c = 20)
  (cond3 : a + b + 2 * d = 14)
  (cond4 : d = 2) :
  a + b + c + d = 23 := sorry

end commuting_days_l192_192385


namespace all_zero_l192_192592

def circle_condition (x : Fin 2007 → ℤ) : Prop :=
  ∀ i : Fin 2007, x i + x (i+1) + x (i+2) + x (i+3) + x (i+4) = 2 * (x (i+1) + x (i+2)) + 2 * (x (i+3) + x (i+4))

theorem all_zero (x : Fin 2007 → ℤ) (h : circle_condition x) : ∀ i, x i = 0 :=
sorry

end all_zero_l192_192592


namespace multiple_of_4_and_8_l192_192616

theorem multiple_of_4_and_8 (a b : ℤ) (h1 : ∃ k1 : ℤ, a = 4 * k1) (h2 : ∃ k2 : ℤ, b = 8 * k2) :
  (∃ k3 : ℤ, b = 4 * k3) ∧ (∃ k4 : ℤ, a - b = 4 * k4) :=
by
  sorry

end multiple_of_4_and_8_l192_192616


namespace greatest_m_value_l192_192113

theorem greatest_m_value (x y m : ℝ) 
  (h₁: x^2 + y^2 = 1)
  (h₂ : |x^3 - y^3| + |x - y| = m^3) : 
  m ≤ 2^(1/3) :=
sorry

end greatest_m_value_l192_192113


namespace sarah_bottle_caps_l192_192555

theorem sarah_bottle_caps (initial_caps : ℕ) (additional_caps : ℕ) (total_caps : ℕ) : initial_caps = 26 → additional_caps = 3 → total_caps = initial_caps + additional_caps → total_caps = 29 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sarah_bottle_caps_l192_192555


namespace find_xy_solution_l192_192124

theorem find_xy_solution (x y : ℕ) (hx : x > 0) (hy : y > 0) 
    (h : 3^x + x^4 = y.factorial + 2019) : 
    (x = 6 ∧ y = 3) :=
by {
  sorry
}

end find_xy_solution_l192_192124


namespace monotonic_quadratic_range_l192_192067

-- Define a quadratic function
noncomputable def quadratic (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- The theorem
theorem monotonic_quadratic_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → quadratic a x₁ ≤ quadratic a x₂) ∨
  (∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → quadratic a x₁ ≥ quadratic a x₂) →
  (a ≤ 2 ∨ 3 ≤ a) :=
sorry

end monotonic_quadratic_range_l192_192067


namespace triangle_ineq_sqrt_triangle_l192_192519

open Real

theorem triangle_ineq_sqrt_triangle (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a):
  (∃ u v w : ℝ, u > 0 ∧ v > 0 ∧ w > 0 ∧ a = v + w ∧ b = u + w ∧ c = u + v) ∧ 
  (sqrt (a * b) + sqrt (b * c) + sqrt (c * a) ≤ a + b + c ∧ a + b + c ≤ 2 * sqrt (a * b) + 2 * sqrt (b * c) + 2 * sqrt (c * a)) :=
  sorry

end triangle_ineq_sqrt_triangle_l192_192519


namespace percentage_length_more_than_breadth_l192_192641

-- Define the basic conditions
variables {C r l b : ℝ}
variable {p : ℝ}

-- Assume the conditions
def conditions (C r l b : ℝ) : Prop :=
  C = 400 ∧ r = 3 ∧ l = 20 ∧ 20 * b = 400 / 3

-- Define the statement that we want to prove
theorem percentage_length_more_than_breadth (C r l b : ℝ) (h : conditions C r l b) :
  ∃ (p : ℝ), l = b * (1 + p / 100) ∧ p = 200 :=
sorry

end percentage_length_more_than_breadth_l192_192641


namespace man_l192_192540

theorem man's_speed_downstream (v : ℕ) (h1 : v - 3 = 8) (s : ℕ := 3) : v + s = 14 :=
by
  sorry

end man_l192_192540


namespace find_abs_x_l192_192794

-- Given conditions
def A (x : ℝ) : ℝ := 3 + x
def B (x : ℝ) : ℝ := 3 - x
def distance (a b : ℝ) : ℝ := abs (a - b)

-- Problem statement: Prove |x| = 4 given the conditions
theorem find_abs_x (x : ℝ) (h : distance (A x) (B x) = 8) : abs x = 4 := 
  sorry

end find_abs_x_l192_192794


namespace trebled_principal_after_5_years_l192_192502

theorem trebled_principal_after_5_years 
(P R : ℝ) (T total_interest : ℝ) (n : ℝ) 
(h1 : T = 10) 
(h2 : total_interest = 800) 
(h3 : (P * R * 10) / 100 = 400) 
(h4 : (P * R * n) / 100 + (3 * P * R * (10 - n)) / 100 = 800) :
n = 5 :=
by
-- The Lean proof will go here
sorry

end trebled_principal_after_5_years_l192_192502


namespace HCF_48_99_l192_192948

-- definitions and theorem stating the problem
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

theorem HCF_48_99 : HCF 48 99 = 3 :=
by
  sorry

end HCF_48_99_l192_192948


namespace lottery_blanks_l192_192601

theorem lottery_blanks (P B : ℕ) (h₁ : P = 10) (h₂ : (P : ℝ) / (P + B) = 0.2857142857142857) : B = 25 := 
by
  sorry

end lottery_blanks_l192_192601


namespace smallest_n_reducible_fraction_l192_192810

theorem smallest_n_reducible_fraction : ∀ (n : ℕ), (∃ (k : ℕ), gcd (n - 13) (5 * n + 6) = k ∧ k > 1) ↔ n = 84 := by
  sorry

end smallest_n_reducible_fraction_l192_192810


namespace car_speed_in_second_hour_l192_192900

theorem car_speed_in_second_hour (x : ℕ) : 84 = (98 + x) / 2 → x = 70 := 
sorry

end car_speed_in_second_hour_l192_192900


namespace simplify_and_evaluate_expression_l192_192632

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_and_evaluate_expression :
  ((x + 1) / (x^2 + 2 * x + 1)) / (1 - (2 / (x + 1))) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l192_192632


namespace minimum_value_proof_l192_192049

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + b = 12

theorem minimum_value_proof : ∀ (a b : ℝ), minimum_value_condition a b → (1 / a + 1 / b ≥ 1 / 3) := 
by
  intros a b h
  sorry

end minimum_value_proof_l192_192049


namespace selling_price_equivalence_l192_192360

noncomputable def cost_price_25_profit : ℝ := 1750 / 1.25
def selling_price_profit := 1520
def selling_price_loss := 1280

theorem selling_price_equivalence
  (cp : ℝ)
  (h1 : cp = cost_price_25_profit)
  (h2 : cp = 1400) :
  (selling_price_profit - cp = cp - selling_price_loss) → (selling_price_loss = 1280) := 
  by
  unfold cost_price_25_profit at h1
  simp [h1] at h2
  sorry

end selling_price_equivalence_l192_192360


namespace Ann_age_is_46_l192_192071

theorem Ann_age_is_46
  (a b : ℕ) 
  (h1 : a + b = 72)
  (h2 : b = (a / 3) + 2 * (a - b)) : a = 46 :=
by
  sorry

end Ann_age_is_46_l192_192071


namespace willie_bananas_l192_192248

variable (W : ℝ) 

theorem willie_bananas (h1 : 35.0 - 14.0 = 21.0) (h2: W + 35.0 = 83.0) : 
  W = 48.0 :=
by
  sorry

end willie_bananas_l192_192248


namespace frog_escape_l192_192663

theorem frog_escape (wellDepth dayClimb nightSlide escapeDays : ℕ)
  (h_depth : wellDepth = 30)
  (h_dayClimb : dayClimb = 3)
  (h_nightSlide : nightSlide = 2)
  (h_escape : escapeDays = 28) :
  ∃ n, n = escapeDays ∧
       ((wellDepth ≤ (n - 1) * (dayClimb - nightSlide) + dayClimb)) :=
by
  sorry

end frog_escape_l192_192663


namespace set_intersection_l192_192163

-- Define set A
def A := {x : ℝ | x^2 - 4 * x < 0}

-- Define set B
def B := {x : ℤ | -2 < x ∧ x ≤ 2}

-- Define the intersection of A and B in ℝ
def A_inter_B := {x : ℝ | (x ∈ A) ∧ (∃ (z : ℤ), (x = z) ∧ (z ∈ B))}

-- Proof statement
theorem set_intersection : A_inter_B = {1, 2} :=
by sorry

end set_intersection_l192_192163


namespace red_balls_in_bag_l192_192443

theorem red_balls_in_bag : 
  ∃ (r : ℕ), (r * (r - 1) = 22) ∧ (r ≤ 12) :=
by { sorry }

end red_balls_in_bag_l192_192443


namespace range_of_m_l192_192125

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem range_of_m (m : ℝ) : (∀ x > 0, m * f x ≤ Real.exp (-x) + m - 1) ↔ m ≤ -1/3 :=
by
  sorry

end range_of_m_l192_192125


namespace angle_sum_l192_192894

theorem angle_sum (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) (h_triangle : A + B + C = 180) (h_complement : 180 - C = 130) :
  A + B = 130 :=
by
  sorry

end angle_sum_l192_192894


namespace problem_statement_l192_192162

variable {x : ℝ}
noncomputable def A : ℝ := 39
noncomputable def B : ℝ := -5

theorem problem_statement (h : ∀ x ≠ 3, (A / (x - 3) + B * (x + 2)) = (-5 * x ^ 2 + 18 * x + 30) / (x - 3)) : A + B = 34 := 
sorry

end problem_statement_l192_192162


namespace geometric_arithmetic_sequence_difference_l192_192595

theorem geometric_arithmetic_sequence_difference
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (hq : q > 0)
  (ha1 : a 1 = 2)
  (ha2 : a 2 = a 1 * q)
  (ha4 : a 4 = a 1 * q ^ 3)
  (ha5 : a 5 = a 1 * q ^ 4)
  (harith : 2 * (a 4 + 2 * a 5) = 2 * a 2 + (a 4 + 2 * a 5))
  (hS : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  S 10 - S 4 = 2016 :=
by
  sorry

end geometric_arithmetic_sequence_difference_l192_192595


namespace seq_15_l192_192570

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else if n = 2 then 2 else 2 * (n - 1) + 1 -- form inferred from solution

theorem seq_15 : seq 15 = 29 := by
  sorry

end seq_15_l192_192570


namespace union_of_A_B_l192_192656

def A (p q : ℝ) : Set ℝ := {x | x^2 + p * x + q = 0}
def B (p q : ℝ) : Set ℝ := {x | x^2 - p * x - 2 * q = 0}

theorem union_of_A_B (p q : ℝ)
  (h1 : A p q ∩ B p q = {-1}) :
  A p q ∪ B p q = {-1, -2, 4} := by
sorry

end union_of_A_B_l192_192656


namespace mary_talking_ratio_l192_192861

theorem mary_talking_ratio:
  let mac_download_time := 10
  let windows_download_time := 3 * mac_download_time
  let audio_glitch_time := 2 * 4
  let video_glitch_time := 6
  let total_glitch_time := audio_glitch_time + video_glitch_time
  let total_download_time := mac_download_time + windows_download_time
  let total_time := 82
  let talking_time := total_time - total_download_time
  let talking_time_without_glitch := talking_time - total_glitch_time
  talking_time_without_glitch / total_glitch_time = 2 :=
by
  sorry

end mary_talking_ratio_l192_192861


namespace rectangle_perimeter_gt_16_l192_192305

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
  sorry

end rectangle_perimeter_gt_16_l192_192305


namespace goose_survived_first_year_l192_192865

theorem goose_survived_first_year (total_eggs : ℕ) (eggs_hatched_ratio : ℚ) (first_month_survival_ratio : ℚ) 
  (first_year_no_survival_ratio : ℚ) 
  (eggs_hatched_ratio_eq : eggs_hatched_ratio = 2/3) 
  (first_month_survival_ratio_eq : first_month_survival_ratio = 3/4)
  (first_year_no_survival_ratio_eq : first_year_no_survival_ratio = 3/5)
  (total_eggs_eq : total_eggs = 500) :
  ∃ (survived_first_year : ℕ), survived_first_year = 100 :=
by
  sorry

end goose_survived_first_year_l192_192865


namespace ratio_of_x_intercepts_l192_192822

theorem ratio_of_x_intercepts (b s t : ℝ) (hb : b ≠ 0) (h1 : s = -b / 8) (h2 : t = -b / 4) : s / t = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l192_192822


namespace vehicle_flow_mod_15_l192_192005

theorem vehicle_flow_mod_15
  (vehicle_length : ℝ := 5)
  (max_speed : ℕ := 100)
  (speed_interval : ℕ := 10)
  (distance_multiplier : ℕ := 10)
  (N : ℕ := 2000) :
  (N % 15) = 5 := 
sorry

end vehicle_flow_mod_15_l192_192005


namespace marco_paint_fraction_l192_192131

theorem marco_paint_fraction (W : ℝ) (M : ℝ) (minutes_paint : ℝ) (fraction_paint : ℝ) :
  M = 60 ∧ W = 1 ∧ minutes_paint = 12 ∧ fraction_paint = 1/5 → 
  (minutes_paint / M) * W = fraction_paint := 
by
  sorry

end marco_paint_fraction_l192_192131


namespace salon_visitors_l192_192784

noncomputable def total_customers (x : ℕ) : ℕ :=
  let revenue_customers_with_one_visit := 10 * x
  let revenue_customers_with_two_visits := 30 * 18
  let revenue_customers_with_three_visits := 10 * 26
  let total_revenue := revenue_customers_with_one_visit + revenue_customers_with_two_visits + revenue_customers_with_three_visits
  if total_revenue = 1240 then
    x + 30 + 10
  else
    0

theorem salon_visitors : 
  ∃ x, total_customers x = 84 :=
by
  use 44
  sorry

end salon_visitors_l192_192784


namespace volume_truncated_cone_l192_192330

/-- 
Given a truncated right circular cone with a large base radius of 10 cm,
a smaller base radius of 3 cm, and a height of 9 cm, 
prove that the volume of the truncated cone is 417 π cubic centimeters.
-/
theorem volume_truncated_cone :
  let R := 10
  let r := 3
  let h := 9
  let V := (1/3) * Real.pi * h * (R^2 + R*r + r^2)
  V = 417 * Real.pi :=
by 
  sorry

end volume_truncated_cone_l192_192330


namespace translated_coordinates_of_B_l192_192144

-- Definitions and conditions
def pointA : ℝ × ℝ := (-2, 3)

def translate_right (x : ℝ) (units : ℝ) : ℝ := x + units
def translate_down (y : ℝ) (units : ℝ) : ℝ := y - units

-- Theorem statement
theorem translated_coordinates_of_B :
  let Bx := translate_right (-2) 3
  let By := translate_down 3 5
  (Bx, By) = (1, -2) :=
by
  -- This is where the proof would go, but we're using sorry to skip the proof steps.
  sorry

end translated_coordinates_of_B_l192_192144


namespace inheritance_amount_l192_192982

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

end inheritance_amount_l192_192982


namespace overtaking_time_l192_192845

theorem overtaking_time (t_a t_b t_k : ℝ) (t_b_start : t_b = t_a - 5) 
                       (overtake_eq1 : 40 * t_b = 30 * t_a)
                       (overtake_eq2 : 60 * (t_a - 10) = 30 * t_a) :
                       t_b = 15 :=
by
  sorry

end overtaking_time_l192_192845


namespace sin_gamma_delta_l192_192787

theorem sin_gamma_delta (γ δ : ℝ)
  (hγ : Complex.exp (Complex.I * γ) = Complex.ofReal 4 / 5 + Complex.I * (3 / 5))
  (hδ : Complex.exp (Complex.I * δ) = Complex.ofReal (-5 / 13) + Complex.I * (12 / 13)) :
  Real.sin (γ + δ) = 21 / 65 :=
by
  sorry

end sin_gamma_delta_l192_192787


namespace slower_train_pass_time_l192_192712

noncomputable def time_to_pass (length_train : ℕ) (speed_faster_kmh : ℕ) (speed_slower_kmh : ℕ) : ℕ :=
  let speed_faster_mps := speed_faster_kmh * 5 / 18
  let speed_slower_mps := speed_slower_kmh * 5 / 18
  let relative_speed := speed_faster_mps + speed_slower_mps
  let distance := length_train
  distance * 18 / (relative_speed * 5)

theorem slower_train_pass_time :
  time_to_pass 500 45 15 = 300 :=
by
  sorry

end slower_train_pass_time_l192_192712


namespace total_number_of_people_l192_192718

theorem total_number_of_people
  (cannoneers : ℕ) 
  (women : ℕ) 
  (men : ℕ) 
  (hc : cannoneers = 63)
  (hw : women = 2 * cannoneers)
  (hm : men = 2 * women) :
  cannoneers + women + men = 378 := 
sorry

end total_number_of_people_l192_192718


namespace taylor_pets_count_l192_192093

noncomputable def totalPetsTaylorFriends (T : ℕ) (x1 : ℕ) (x2 : ℕ) : ℕ :=
  T + 3 * x1 + 2 * x2

theorem taylor_pets_count (T : ℕ) (x1 x2 : ℕ) (h1 : x1 = 2 * T) (h2 : x2 = 2) (h3 : totalPetsTaylorFriends T x1 x2 = 32) :
  T = 4 :=
by
  sorry

end taylor_pets_count_l192_192093


namespace modulus_product_l192_192082

open Complex -- to open the complex namespace

-- Define the complex numbers
def z1 : ℂ := 10 - 5 * Complex.I
def z2 : ℂ := 7 + 24 * Complex.I

-- State the theorem to prove
theorem modulus_product : abs (z1 * z2) = 125 * Real.sqrt 5 := by
  sorry

end modulus_product_l192_192082


namespace second_car_speed_correct_l192_192692

noncomputable def first_car_speed : ℝ := 90

noncomputable def time_elapsed (h : ℕ) (m : ℕ) : ℝ := h + m / 60

noncomputable def distance_travelled (speed : ℝ) (time : ℝ) : ℝ := speed * time

def distance_ratio_at_832 (dist1 dist2 : ℝ) : Prop := dist1 = 1.2 * dist2
def distance_ratio_at_920 (dist1 dist2 : ℝ) : Prop := dist1 = 2 * dist2

noncomputable def time_first_car_832 : ℝ := time_elapsed 0 24
noncomputable def dist_first_car_832 : ℝ := distance_travelled first_car_speed time_first_car_832

noncomputable def dist_second_car_832 : ℝ := dist_first_car_832 / 1.2

noncomputable def time_first_car_920 : ℝ := time_elapsed 1 12
noncomputable def dist_first_car_920 : ℝ := distance_travelled first_car_speed time_first_car_920

noncomputable def dist_second_car_920 : ℝ := dist_first_car_920 / 2

noncomputable def time_second_car_travel : ℝ := time_elapsed 0 42

noncomputable def second_car_speed : ℝ := (dist_second_car_920 - dist_second_car_832) / time_second_car_travel

theorem second_car_speed_correct :
  second_car_speed = 34.2857 := by
  sorry

end second_car_speed_correct_l192_192692


namespace carnations_in_first_bouquet_l192_192383

theorem carnations_in_first_bouquet 
  (c2 : ℕ) (c3 : ℕ) (avg : ℕ) (n : ℕ) (total_carnations : ℕ) : 
  c2 = 14 → c3 = 13 → avg = 12 → n = 3 → total_carnations = avg * n →
  (total_carnations - (c2 + c3) = 9) :=
by
  sorry

end carnations_in_first_bouquet_l192_192383


namespace no_four_primes_exist_l192_192183

theorem no_four_primes_exist (a b c d : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b)
  (hc : Nat.Prime c) (hd : Nat.Prime d) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : (1 / a : ℚ) + (1 / d) = (1 / b) + (1 / c)) : False := sorry

end no_four_primes_exist_l192_192183


namespace sum_of_undefined_domain_values_l192_192085

theorem sum_of_undefined_domain_values :
  ∀ (x : ℝ), (x = 0 ∨ (1 + 1/x) = 0 ∨ (1 + 1/(1 + 1/x)) = 0 ∨ (1 + 1/(1 + 1/(1 + 1/x))) = 0) →
  x = 0 ∧ x = -1 ∧ x = -1/2 ∧ x = -1/3 →
  (0 + (-1) + (-1/2) + (-1/3) = -11/6) := sorry

end sum_of_undefined_domain_values_l192_192085


namespace simplify_expression_l192_192372

theorem simplify_expression :
  (4 + 2 + 6) / 3 - (2 + 1) / 3 = 3 := by
  sorry

end simplify_expression_l192_192372


namespace net_rate_of_pay_is_25_l192_192778

-- Define the conditions 
variables (hours : ℕ) (speed : ℕ) (efficiency : ℕ)
variables (pay_per_mile : ℝ) (cost_per_gallon : ℝ)
variables (total_distance : ℕ) (gas_used : ℕ)
variables (total_earnings : ℝ) (total_cost : ℝ) (net_earnings : ℝ) (net_rate_of_pay : ℝ)

-- Assume the given conditions are as stated in the problem
axiom hrs : hours = 3
axiom spd : speed = 50
axiom eff : efficiency = 25
axiom ppm : pay_per_mile = 0.60
axiom cpg : cost_per_gallon = 2.50

-- Assuming intermediate computations
axiom distance_calc : total_distance = speed * hours
axiom gas_calc : gas_used = total_distance / efficiency
axiom earnings_calc : total_earnings = pay_per_mile * total_distance
axiom cost_calc : total_cost = cost_per_gallon * gas_used
axiom net_earnings_calc : net_earnings = total_earnings - total_cost
axiom pay_rate_calc : net_rate_of_pay = net_earnings / hours

-- Proving the final result
theorem net_rate_of_pay_is_25 :
  net_rate_of_pay = 25 :=
by
  -- Proof goes here
  sorry

end net_rate_of_pay_is_25_l192_192778


namespace algebraic_expression_positive_l192_192415

theorem algebraic_expression_positive (a b : ℝ) : 
  a^2 + b^2 + 4*b - 2*a + 6 > 0 :=
by sorry

end algebraic_expression_positive_l192_192415


namespace race_time_A_l192_192802

theorem race_time_A (v_A v_B : ℝ) (t_A t_B : ℝ) (hA_time_eq : v_A = 1000 / t_A)
  (hB_time_eq : v_B = 960 / t_B) (hA_beats_B_40m : 1000 / v_A = 960 / v_B)
  (hA_beats_B_8s : t_B = t_A + 8) : t_A = 200 := 
  sorry

end race_time_A_l192_192802


namespace sum_of_repeating_decimals_l192_192987

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l192_192987


namespace passing_marks_l192_192542

theorem passing_marks
  (T P : ℝ)
  (h1 : 0.20 * T = P - 40)
  (h2 : 0.30 * T = P + 20) :
  P = 160 :=
by
  sorry

end passing_marks_l192_192542


namespace garden_to_land_area_ratio_l192_192509

variables (l_ter w_ter l_gard w_gard : ℝ)

-- Condition 1: Width of the land rectangle is 3/5 of its length
def land_conditions : Prop := w_ter = (3 / 5) * l_ter

-- Condition 2: Width of the garden rectangle is 3/5 of its length
def garden_conditions : Prop := w_gard = (3 / 5) * l_gard

-- Problem: Ratio of the area of the garden to the area of the land is 36%.
theorem garden_to_land_area_ratio
  (h_land : land_conditions l_ter w_ter)
  (h_garden : garden_conditions l_gard w_gard) :
  (l_gard * w_gard) / (l_ter * w_ter) = 0.36 := sorry

end garden_to_land_area_ratio_l192_192509


namespace bc_money_l192_192426

variables (A B C : ℕ)

theorem bc_money (h1 : A + B + C = 400) (h2 : A + C = 300) (h3 : C = 50) : B + C = 150 :=
sorry

end bc_money_l192_192426


namespace mass_of_empty_glass_l192_192908

theorem mass_of_empty_glass (mass_full : ℕ) (mass_half : ℕ) (G : ℕ) :
  mass_full = 1000 →
  mass_half = 700 →
  G = mass_full - (mass_full - mass_half) * 2 →
  G = 400 :=
by
  intros h_full h_half h_G_eq
  sorry

end mass_of_empty_glass_l192_192908


namespace train_length_l192_192854

variable (L_train : ℝ)
variable (speed_kmhr : ℝ := 45)
variable (time_seconds : ℝ := 30)
variable (bridge_length_m : ℝ := 275)
variable (train_speed_ms : ℝ := speed_kmhr * (1000 / 3600))
variable (total_distance : ℝ := train_speed_ms * time_seconds)

theorem train_length
  (h_total : total_distance = L_train + bridge_length_m) :
  L_train = 100 :=
by 
  sorry

end train_length_l192_192854


namespace probability_wife_selection_l192_192408

theorem probability_wife_selection (P_H P_only_one P_W : ℝ)
  (h1 : P_H = 1 / 7)
  (h2 : P_only_one = 0.28571428571428575)
  (h3 : P_only_one = (P_H * (1 - P_W)) + (P_W * (1 - P_H))) :
  P_W = 1 / 5 :=
by
  sorry

end probability_wife_selection_l192_192408


namespace fenced_area_l192_192193

theorem fenced_area (L W : ℝ) (square_side triangle_leg : ℝ) :
  L = 20 ∧ W = 18 ∧ square_side = 4 ∧ triangle_leg = 3 →
  (L * W - square_side^2 - (1 / 2) * triangle_leg^2 = 339.5) := by
  intros h
  rcases h with ⟨hL, hW, hs, ht⟩
  rw [hL, hW, hs, ht]
  simp
  sorry

end fenced_area_l192_192193


namespace clever_question_l192_192244

-- Define the conditions as predicates
def inhabitants_truthful (city : String) : Prop := 
  city = "Mars-Polis"

def inhabitants_lying (city : String) : Prop := 
  city = "Mars-City"

def responses (question : String) (city : String) : String :=
  if question = "Are we in Mars-City?" then
    if city = "Mars-City" then "No" else "Yes"
  else if question = "Do you live here?" then
    if city = "Mars-City" then "No" else "Yes"
  else "Unknown"

-- Define the main theorem
theorem clever_question (city : String) (initial_response : String) :
  (inhabitants_truthful city ∨ inhabitants_lying city) →
  responses "Are we in Mars-City?" city = initial_response →
  responses "Do you live here?" city = "Yes" ∨ responses "Do you live here?" city = "No" :=
by
  sorry

end clever_question_l192_192244


namespace find_x_l192_192834

variable (x : ℝ)

def delta (x : ℝ) : ℝ := 4 * x + 5
def phi (x : ℝ) : ℝ := 9 * x + 6

theorem find_x : delta (phi x) = 23 → x = -1 / 6 := by
  intro h
  sorry

end find_x_l192_192834


namespace max_value_of_f_l192_192326

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_of_f : ∀ x : ℝ, x > 0 → f x ≤ (Real.log (Real.exp 1)) / (Real.exp 1) :=
by
  sorry

end max_value_of_f_l192_192326


namespace probability_is_two_thirds_l192_192680

-- Define the general framework and conditions
def total_students : ℕ := 4
def students_from_first_grade : ℕ := 2
def students_from_second_grade : ℕ := 2

-- Define the combinations for selecting 2 students out of 4
def total_ways_to_select_2_students : ℕ := Nat.choose total_students 2

-- Define the combinations for selecting 1 student from each grade
def ways_to_select_1_from_first : ℕ := Nat.choose students_from_first_grade 1
def ways_to_select_1_from_second : ℕ := Nat.choose students_from_second_grade 1
def favorable_ways : ℕ := ways_to_select_1_from_first * ways_to_select_1_from_second

-- The target probability calculation
noncomputable def probability_of_different_grades : ℚ :=
  favorable_ways / total_ways_to_select_2_students

-- The statement and proof requirement (proof is deferred with sorry)
theorem probability_is_two_thirds :
  probability_of_different_grades = 2 / 3 :=
by sorry

end probability_is_two_thirds_l192_192680


namespace translate_parabola_l192_192045

theorem translate_parabola :
  (∀ x, y = 1/2 * x^2 + 1 → y = 1/2 * (x - 1)^2 - 2) :=
by
  sorry

end translate_parabola_l192_192045


namespace max_roads_no_intersections_l192_192518

theorem max_roads_no_intersections (V : ℕ) (hV : V = 100) : 
  ∃ E : ℕ, E ≤ 3 * V - 6 ∧ E = 294 := 
by 
  sorry

end max_roads_no_intersections_l192_192518


namespace maximize_revenue_l192_192070

-- Define the problem conditions
def is_valid (x y : ℕ) : Prop :=
  x + y ≤ 60 ∧ 6 * x + 30 * y ≤ 600

-- Define the objective function
def revenue (x y : ℕ) : ℚ :=
  2.5 * x + 7.5 * y

-- State the theorem with the given conditions
theorem maximize_revenue : 
  (∃ x y : ℕ, is_valid x y ∧ ∀ a b : ℕ, is_valid a b → revenue x y >= revenue a b) ∧
  ∃ x y, is_valid x y ∧ revenue x y = revenue 50 10 := 
sorry

end maximize_revenue_l192_192070


namespace part1_part2_l192_192836

namespace ClothingFactory

variables {x y m : ℝ} -- defining variables

-- The conditions
def condition1 : Prop := x + 2 * y = 5
def condition2 : Prop := 3 * x + y = 7
def condition3 : Prop := 1.8 * (100 - m) + 1.6 * m ≤ 168

-- Theorems to Prove
theorem part1 (h1 : x + 2 * y = 5) (h2 : 3 * x + y = 7) : 
  x = 1.8 ∧ y = 1.6 := 
sorry

theorem part2 (h1 : x = 1.8) (h2 : y = 1.6) (h3 : 1.8 * (100 - m) + 1.6 * m ≤ 168) : 
  m ≥ 60 := 
sorry

end ClothingFactory

end part1_part2_l192_192836


namespace difference_between_numbers_l192_192968

theorem difference_between_numbers 
  (L S : ℕ) 
  (hL : L = 1584) 
  (hDiv : L = 6 * S + 15) : 
  L - S = 1323 := 
by
  sorry

end difference_between_numbers_l192_192968


namespace william_library_visits_l192_192319

variable (W : ℕ) (J : ℕ)
variable (h1 : J = 4 * W)
variable (h2 : 4 * J = 32)

theorem william_library_visits : W = 2 :=
by
  sorry

end william_library_visits_l192_192319


namespace letters_identity_l192_192911

def identity_of_letters (first second third : ℕ) : Prop :=
  (first, second, third) = (1, 0, 1)

theorem letters_identity (first second third : ℕ) :
  first + second + third = 1 →
  (first = 1 → 1 ≠ first + second) →
  (second = 0 → first + second < 2) →
  (third = 0 → first + second = 1) →
  identity_of_letters first second third :=
by sorry

end letters_identity_l192_192911


namespace solution_set_of_inequality_l192_192591

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} := sorry

end solution_set_of_inequality_l192_192591


namespace batsman_average_increase_l192_192800

theorem batsman_average_increase 
    (A : ℝ) 
    (h1 : 11 * A + 80 = 12 * 47) : 
    47 - A = 3 := 
by 
  -- Proof goes here
  sorry

end batsman_average_increase_l192_192800


namespace fraction_inequality_l192_192610

theorem fraction_inequality {a b : ℝ} (h1 : a < b) (h2 : b < 0) : (1 / a) > (1 / b) :=
by
  sorry

end fraction_inequality_l192_192610


namespace zero_is_multiple_of_all_primes_l192_192906

theorem zero_is_multiple_of_all_primes :
  ∀ (x : ℕ), (∀ p : ℕ, Prime p → ∃ n : ℕ, x = n * p) ↔ x = 0 := by
sorry

end zero_is_multiple_of_all_primes_l192_192906


namespace problem_statement_l192_192174

variable (x1 x2 x3 x4 x5 x6 x7 : ℝ)

theorem problem_statement
  (h1 : x1 + 4*x2 + 9*x3 + 16*x4 + 25*x5 + 36*x6 + 49*x7 = 5)
  (h2 : 4*x1 + 9*x2 + 16*x3 + 25*x4 + 36*x5 + 49*x6 + 64*x7 = 20)
  (h3 : 9*x1 + 16*x2 + 25*x3 + 36*x4 + 49*x5 + 64*x6 + 81*x7 = 145) :
  16*x1 + 25*x2 + 36*x3 + 49*x4 + 64*x5 + 81*x6 + 100*x7 = 380 :=
sorry

end problem_statement_l192_192174


namespace fraction_difference_eq_l192_192961

theorem fraction_difference_eq (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
sorry

end fraction_difference_eq_l192_192961


namespace bacteria_growth_time_l192_192211

theorem bacteria_growth_time : 
  (∀ n : ℕ, 2 ^ n = 4096 → (n * 15) / 60 = 3) :=
by
  sorry

end bacteria_growth_time_l192_192211


namespace T_number_square_l192_192080

theorem T_number_square (a b : ℤ) : ∃ c d : ℤ, (a^2 + a * b + b^2)^2 = c^2 + c * d + d^2 := by
  sorry

end T_number_square_l192_192080


namespace q_domain_range_l192_192531

open Set

-- Given the function h with the specified domain and range
variable (h : ℝ → ℝ) (h_domain : ∀ x, -1 ≤ x ∧ x ≤ 3 → h x ∈ Icc 0 2)

def q (x : ℝ) : ℝ := 2 - h (x - 2)

theorem q_domain_range :
  (∀ x, 1 ≤ x ∧ x ≤ 5 → (q h x) ∈ Icc 0 2) ∧
  (∀ y, q h y ∈ Icc 0 2 ↔ y ∈ Icc 1 5) :=
by
  sorry

end q_domain_range_l192_192531


namespace algebraic_expression_value_l192_192940

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 11 = -5 :=
by
  sorry

end algebraic_expression_value_l192_192940


namespace platform_length_l192_192013

theorem platform_length (speed_kmh : ℕ) (time_min : ℕ) (train_length_m : ℕ) (distance_covered_m : ℕ) : 
  speed_kmh = 90 → time_min = 1 → train_length_m = 750 → distance_covered_m = 1500 →
  train_length_m + (distance_covered_m - train_length_m) = 750 + (1500 - 750) :=
by sorry

end platform_length_l192_192013


namespace product_of_four_consecutive_integers_is_perfect_square_l192_192350

-- Define the main statement we want to prove
theorem product_of_four_consecutive_integers_is_perfect_square (n : ℤ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 :=
by
  -- Proof is omitted
  sorry

end product_of_four_consecutive_integers_is_perfect_square_l192_192350


namespace don_can_have_more_rum_l192_192774

-- Definitions based on conditions:
def given_rum : ℕ := 10
def max_consumption_rate : ℕ := 3
def already_had : ℕ := 12

-- Maximum allowed consumption calculation:
def max_allowed_rum : ℕ := max_consumption_rate * given_rum

-- Remaining rum calculation:
def remaining_rum : ℕ := max_allowed_rum - already_had

-- Proof statement of the problem:
theorem don_can_have_more_rum : remaining_rum = 18 := by
  -- Let's compute directly:
  have h1 : max_allowed_rum = 30 := by
    simp [max_allowed_rum, max_consumption_rate, given_rum]

  have h2 : remaining_rum = 18 := by
    simp [remaining_rum, h1, already_had]

  exact h2

end don_can_have_more_rum_l192_192774


namespace option_B_correct_l192_192975

theorem option_B_correct : 1 ∈ ({0, 1} : Set ℕ) := 
by
  sorry

end option_B_correct_l192_192975


namespace M_is_real_l192_192453

open Complex

-- Define the condition that characterizes the set M
def M (Z : ℂ) : Prop := (Z - 1)^2 = abs (Z - 1)^2

-- Prove that M is exactly the set of real numbers
theorem M_is_real : ∀ (Z : ℂ), M Z ↔ Z.im = 0 :=
by
  sorry

end M_is_real_l192_192453


namespace positive_difference_of_perimeters_l192_192947

noncomputable def perimeter_figure1 : ℕ :=
  let outer_rectangle := 2 * (5 + 1)
  let inner_extension := 2 * (2 + 1)
  outer_rectangle + inner_extension

noncomputable def perimeter_figure2 : ℕ :=
  2 * (5 + 2)

theorem positive_difference_of_perimeters :
  (perimeter_figure1 - perimeter_figure2 = 4) :=
by
  let perimeter1 := perimeter_figure1
  let perimeter2 := perimeter_figure2
  sorry

end positive_difference_of_perimeters_l192_192947


namespace find_annual_interest_rate_l192_192557

noncomputable def compound_interest_problem : Prop :=
  ∃ (r : ℝ),
    let P := 8000
    let CI := 3109
    let t := 2.3333
    let A := 11109
    let n := 1
    A = P * (1 + r/n)^(n*t) ∧ r = 0.1505

theorem find_annual_interest_rate : compound_interest_problem :=
by sorry

end find_annual_interest_rate_l192_192557


namespace no_five_consecutive_terms_divisible_by_2005_l192_192274

noncomputable def a (n : ℕ) : ℤ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_terms_divisible_by_2005 : ¬ ∃ n : ℕ, (a n % 2005 = 0) ∧ (a (n+1) % 2005 = 0) ∧ (a (n+2) % 2005 = 0) ∧ (a (n+3) % 2005 = 0) ∧ (a (n+4) % 2005 = 0) := sorry

end no_five_consecutive_terms_divisible_by_2005_l192_192274


namespace range_of_k_l192_192252

theorem range_of_k (k : ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, a n = 2 * (n:ℕ)^2 + k * (n:ℕ)) 
  (increasing : ∀ n : ℕ+, a n < a (n + 1)) : 
  k > -6 := 
by 
  sorry

end range_of_k_l192_192252


namespace increasing_interval_f_l192_192303

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval_f :
  ∀ x, (2 < x) → (∃ ε > 0, ∀ δ > 0, δ < ε → f (x + δ) ≥ f x) :=
by
  sorry

end increasing_interval_f_l192_192303


namespace value_of_f_3_div_2_l192_192401

noncomputable def f : ℝ → ℝ := sorry

axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x
axiom even_f : ∀ x : ℝ, f (x) = f (-x)
axiom f_in_0_1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f (x) = x + 1

theorem value_of_f_3_div_2 : f (3 / 2) = 3 / 2 := by
  sorry

end value_of_f_3_div_2_l192_192401


namespace arithmetic_sequence_sum_l192_192196

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
    (h_arith_seq : ∀ n, a (n + 1) = a n + d)
    (h_a5 : a 5 = 3)
    (h_a6 : a 6 = -2) :
  a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = -49 :=
by
  sorry

end arithmetic_sequence_sum_l192_192196


namespace incorrect_statement_d_l192_192572

noncomputable def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem incorrect_statement_d (n : ℤ) :
  (n < cbrt 9 ∧ cbrt 9 < n+1) → n ≠ 3 :=
by
  intro h
  have h2 : (2 : ℤ) < cbrt 9 := sorry
  have h3 : cbrt 9 < (3 : ℤ) := sorry
  exact sorry

end incorrect_statement_d_l192_192572


namespace smallest_n_exists_l192_192642

def connected (a b : ℕ) : Prop := -- define connection based on a picture not specified here, placeholder
sorry

def not_connected (a b : ℕ) : Prop := ¬ connected a b

def coprime (a n : ℕ) : Prop := ∀ k : ℕ, k > 1 → k ∣ a → ¬ k ∣ n

def common_divisor_greater_than_one (a n : ℕ) : Prop := ∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ n

theorem smallest_n_exists :
  ∃ n : ℕ,
  (n = 35) ∧
  ∀ (numbers : Fin 7 → ℕ),
  (∀ i j, not_connected (numbers i) (numbers j) → coprime (numbers i + numbers j) n) ∧
  (∀ i j, connected (numbers i) (numbers j) → common_divisor_greater_than_one (numbers i + numbers j) n) := 
sorry

end smallest_n_exists_l192_192642


namespace polyhedron_volume_l192_192483

/-- Each 12 cm × 12 cm square is cut into two right-angled isosceles triangles by joining the midpoints of two adjacent sides. 
    These six triangles are attached to a regular hexagon to form a polyhedron.
    Prove that the volume of the resulting polyhedron is 864 cubic cm. -/
theorem polyhedron_volume :
  let s : ℝ := 12
  let volume_of_cube := s^3
  let volume_of_polyhedron := volume_of_cube / 2
  volume_of_polyhedron = 864 := 
by
  sorry

end polyhedron_volume_l192_192483


namespace triangle_area_l192_192484

theorem triangle_area (c b : ℝ) (c_eq : c = 15) (b_eq : b = 9) :
  ∃ a : ℝ, a^2 = c^2 - b^2 ∧ (b * a) / 2 = 54 := by
  sorry

end triangle_area_l192_192484


namespace calculation_correct_l192_192132

theorem calculation_correct : 
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by
  sorry

end calculation_correct_l192_192132


namespace ratio_of_areas_l192_192789

variable (s' : ℝ) -- Let s' be the side length of square S'

def area_square : ℝ := s' ^ 2
def length_longer_side_rectangle : ℝ := 1.15 * s'
def length_shorter_side_rectangle : ℝ := 0.95 * s'
def area_rectangle : ℝ := length_longer_side_rectangle s' * length_shorter_side_rectangle s'

theorem ratio_of_areas :
  (area_rectangle s') / (area_square s') = (10925 / 10000) :=
by
  -- skip the proof for now
  sorry

end ratio_of_areas_l192_192789


namespace count_FourDigitNumsWithThousandsDigitFive_is_1000_l192_192761

def count_FourDigitNumsWithThousandsDigitFive : Nat :=
  let minNum := 5000
  let maxNum := 5999
  maxNum - minNum + 1

theorem count_FourDigitNumsWithThousandsDigitFive_is_1000 :
  count_FourDigitNumsWithThousandsDigitFive = 1000 :=
by
  sorry

end count_FourDigitNumsWithThousandsDigitFive_is_1000_l192_192761


namespace trig_expression_value_l192_192645

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 1/2) :
  (1 + 2 * Real.sin (π - α) * Real.cos (-2 * π - α)) / 
  (Real.sin (-α) ^ 2 - Real.sin (5 * π / 2 - α) ^ 2) = -3 :=
by
  sorry

end trig_expression_value_l192_192645


namespace factory_output_l192_192026

variable (a : ℝ)
variable (n : ℕ)
variable (r : ℝ)

-- Initial condition: the output value increases by 10% each year for 5 years
def annual_growth (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^n

-- Theorem statement
theorem factory_output (a : ℝ) : annual_growth a 1.1 5 = 1.1^5 * a :=
by
  sorry

end factory_output_l192_192026


namespace max_quadratic_in_interval_l192_192166

-- Define the quadratic function
noncomputable def quadratic_fun (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the closed interval
def interval (a b : ℝ) (x : ℝ) : Prop := a ≤ x ∧ x ≤ b

-- Define the maximum value property
def is_max_value (f : ℝ → ℝ) (a b max_val : ℝ) : Prop :=
  ∀ x, interval a b x → f x ≤ max_val

-- State the problem in Lean 4
theorem max_quadratic_in_interval : 
  is_max_value quadratic_fun (-5) 3 36 := 
sorry

end max_quadratic_in_interval_l192_192166


namespace total_practice_hours_l192_192891

def schedule : List ℕ := [6, 4, 5, 7, 3]

-- We define the conditions
def total_scheduled_hours : ℕ := schedule.sum

def average_daily_practice_time (total : ℕ) : ℕ := total / schedule.length

def rainy_day_lost_hours : ℕ := average_daily_practice_time total_scheduled_hours

def player_A_missed_hours : ℕ := 2

def player_B_missed_hours : ℕ := 3

def total_missed_hours : ℕ := player_A_missed_hours + player_B_missed_hours

def total_hours_practiced : ℕ := total_scheduled_hours - (rainy_day_lost_hours + total_missed_hours)

-- Now we state the theorem we want to prove
theorem total_practice_hours : total_hours_practiced = 15 := by
  -- omitted proof
  sorry

end total_practice_hours_l192_192891


namespace sum_base5_eq_l192_192310

theorem sum_base5_eq :
  (432 + 43 + 4 : ℕ) = 1034 :=
by sorry

end sum_base5_eq_l192_192310


namespace abs_eq_1_solution_set_l192_192351

theorem abs_eq_1_solution_set (x : ℝ) : (|x| + |x + 1| = 1) ↔ (x ∈ Set.Icc (-1 : ℝ) 0) := by
  sorry

end abs_eq_1_solution_set_l192_192351


namespace smallest_positive_integer_l192_192994

theorem smallest_positive_integer 
  (x : ℤ) (h1 : x % 6 = 3) (h2 : x % 8 = 2) : x = 33 :=
sorry

end smallest_positive_integer_l192_192994


namespace marble_weight_l192_192578

-- Define the conditions
def condition1 (m k : ℝ) : Prop := 9 * m = 5 * k
def condition2 (k : ℝ) : Prop := 4 * k = 120

-- Define the main goal, i.e., proving m = 50/3 given the conditions
theorem marble_weight (m k : ℝ) 
  (h1 : condition1 m k) 
  (h2 : condition2 k) : 
  m = 50 / 3 := by 
  sorry

end marble_weight_l192_192578


namespace average_velocity_instantaneous_velocity_l192_192946

noncomputable def s (t : ℝ) : ℝ := 8 - 3 * t^2

theorem average_velocity {Δt : ℝ} (h : Δt ≠ 0) :
  (s (1 + Δt) - s 1) / Δt = -6 - 3 * Δt :=
sorry

theorem instantaneous_velocity :
  deriv s 1 = -6 :=
sorry

end average_velocity_instantaneous_velocity_l192_192946


namespace expression_evaluation_l192_192033

theorem expression_evaluation :
  (0.15)^3 - (0.06)^3 / (0.15)^2 + 0.009 + (0.06)^2 = 0.006375 :=
by
  sorry

end expression_evaluation_l192_192033


namespace bus_journey_distance_l192_192664

theorem bus_journey_distance (x : ℝ) (h1 : 0 ≤ x)
  (h2 : 0 ≤ 250 - x)
  (h3 : x / 40 + (250 - x) / 60 = 5.2) :
  x = 124 :=
sorry

end bus_journey_distance_l192_192664


namespace boy_work_completion_days_l192_192901

theorem boy_work_completion_days (M W B : ℚ) (D : ℚ)
  (h1 : M + W + B = 1 / 4)
  (h2 : M = 1 / 6)
  (h3 : W = 1 / 36)
  (h4 : B = 1 / D) :
  D = 18 := by
  sorry

end boy_work_completion_days_l192_192901


namespace height_difference_between_crates_l192_192607

theorem height_difference_between_crates 
  (n : ℕ) (diameter : ℝ) 
  (height_A : ℝ) (height_B : ℝ) :
  n = 200 →
  diameter = 12 →
  height_A = n / 10 * diameter →
  height_B = n / 20 * (diameter + 6 * Real.sqrt 3) →
  height_A - height_B = 120 - 60 * Real.sqrt 3 :=
sorry

end height_difference_between_crates_l192_192607


namespace molecular_weight_correct_l192_192496

-- Define the atomic weights
def atomic_weight_Cu : ℝ := 63.546
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

-- Define the number of atoms in the compound
def num_atoms_Cu : ℕ := 1
def num_atoms_C : ℕ := 1
def num_atoms_O : ℕ := 3

-- Define the molecular weight calculation
def molecular_weight : ℝ :=
  num_atoms_Cu * atomic_weight_Cu + 
  num_atoms_C * atomic_weight_C + 
  num_atoms_O * atomic_weight_O

-- Prove the molecular weight of the compound
theorem molecular_weight_correct : molecular_weight = 123.554 :=
by
  sorry

end molecular_weight_correct_l192_192496


namespace sum_of_coordinates_l192_192448

theorem sum_of_coordinates (C D : ℝ × ℝ) (hC : C = (0, 0)) (hD : D.snd = 6) (h_slope : (D.snd - C.snd) / (D.fst - C.fst) = 3/4) : D.fst + D.snd = 14 :=
sorry

end sum_of_coordinates_l192_192448


namespace sum_square_ends_same_digit_l192_192344

theorem sum_square_ends_same_digit {a b : ℤ} (h : (a + b) % 10 = 0) :
  (a^2 % 10) = (b^2 % 10) :=
by
  sorry

end sum_square_ends_same_digit_l192_192344


namespace center_square_is_15_l192_192099

noncomputable def center_square_value : ℤ :=
  let d1 := (15 - 3) / 2
  let d3 := (33 - 9) / 2
  let middle_first_row := 3 + d1
  let middle_last_row := 9 + d3
  let d2 := (middle_last_row - middle_first_row) / 2
  middle_first_row + d2

theorem center_square_is_15 : center_square_value = 15 := by
  sorry

end center_square_is_15_l192_192099


namespace beads_probability_l192_192407

/-
  Four red beads, three white beads, and two blue beads are placed in a line in random order.
  Prove that the probability that no two neighboring beads are the same color is 1/70.
-/
theorem beads_probability :
  let total_permutations := Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 2)
  let valid_permutations := 18 -- conservative estimate from the solution
  (valid_permutations : ℚ) / total_permutations = 1 / 70 :=
by
  let total_permutations := Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 2)
  let valid_permutations := 18
  show (valid_permutations : ℚ) / total_permutations = 1 / 70
  -- skipping proof details
  sorry

end beads_probability_l192_192407


namespace samuel_faster_than_sarah_l192_192779

theorem samuel_faster_than_sarah
  (efficiency_samuel : ℝ := 0.90)
  (efficiency_sarah : ℝ := 0.75)
  (efficiency_tim : ℝ := 0.80)
  (time_tim : ℝ := 45)
  : (time_tim * efficiency_tim / efficiency_sarah) - (time_tim * efficiency_tim / efficiency_samuel) = 8 :=
by
  sorry

end samuel_faster_than_sarah_l192_192779


namespace remainder_76_pow_77_mod_7_l192_192876

/-- Statement of the problem:
Prove that the remainder of \(76^{77}\) divided by 7 is 6.
-/
theorem remainder_76_pow_77_mod_7 :
  (76 ^ 77) % 7 = 6 := 
by
  sorry

end remainder_76_pow_77_mod_7_l192_192876


namespace Trisha_total_distance_l192_192939

theorem Trisha_total_distance :
  let d1 := 0.11  -- hotel to postcard shop
  let d2 := 0.11  -- postcard shop back to hotel
  let d3 := 1.52  -- hotel to T-shirt shop
  let d4 := 0.45  -- T-shirt shop to hat shop
  let d5 := 0.87  -- hat shop to purse shop
  let d6 := 2.32  -- purse shop back to hotel
  d1 + d2 + d3 + d4 + d5 + d6 = 5.38 :=
by
  sorry

end Trisha_total_distance_l192_192939


namespace min_S_value_l192_192847

noncomputable def S (x y z : ℝ) : ℝ := (1 + z) / (2 * x * y * z)

theorem min_S_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x^2 + y^2 + z^2 = 1) :
  S x y z ≥ 4 := 
sorry

end min_S_value_l192_192847


namespace last_digit_of_large_prime_l192_192497

theorem last_digit_of_large_prime :
  let n := 2^859433 - 1
  let last_digit := n % 10
  last_digit = 1 :=
by
  sorry

end last_digit_of_large_prime_l192_192497


namespace greatest_difference_is_124_l192_192332

-- Define the variables a, b, c, and x
variables (a b c x : ℕ)

-- Define the conditions of the problem
def conditions (a b c : ℕ) := 
  (4 * a = 2 * b) ∧ 
  (4 * a = c) ∧ 
  (a > 0) ∧ 
  (a < 10) ∧ 
  (b < 10) ∧ 
  (c < 10)

-- Define the value of a number given its digits
def number (a b c : ℕ) := 100 * a + 10 * b + c

-- Define the maximum and minimum values of x
def max_val (a : ℕ) := number a (2 * a) (4 * a)
def min_val (a : ℕ) := number a (2 * a) (4 * a)

-- Define the greatest difference
def greatest_difference := max_val 2 - min_val 1

-- Prove that the greatest difference is 124
theorem greatest_difference_is_124 : greatest_difference = 124 :=
by 
  unfold greatest_difference 
  unfold max_val 
  unfold min_val 
  unfold number 
  sorry

end greatest_difference_is_124_l192_192332


namespace earrings_ratio_l192_192390

theorem earrings_ratio :
  ∀ (total_pairs : ℕ) (given_pairs : ℕ) (total_earrings : ℕ) (given_earrings : ℕ),
    total_pairs = 12 →
    given_pairs = total_pairs / 2 →
    total_earrings = total_pairs * 2 →
    given_earrings = total_earrings / 2 →
    total_earrings = 36 →
    given_earrings = 12 →
    (total_earrings / given_earrings = 3) :=
by
  sorry

end earrings_ratio_l192_192390


namespace friends_belong_special_team_l192_192134

-- Define a type for students
universe u
variable {Student : Type u}

-- Assume a friendship relation among students
variable (friend : Student → Student → Prop)

-- Assume the conditions as given in the problem
variable (S : Student → Set (Set Student))
variable (students : Set Student)
variable (S_non_empty : ∀ v : Student, S v ≠ ∅)
variable (friendship_condition : 
  ∀ u v : Student, friend u v → 
    (∃ w : Student, S u ∩ S v ⊇ S w))
variable (special_team : ∀ (T : Set Student),
  (∃ v ∈ T, ∀ w : Student, w ∈ T → friend v w) ↔
  (∃ v ∈ T, ∀ w : Student, friend v w → w ∈ T))

-- Prove that any two friends belong to some special team
theorem friends_belong_special_team :
  ∀ u v : Student, friend u v → 
    (∃ T : Set Student, T ∈ S u ∩ S v ∧ 
      (∃ w ∈ T, ∀ x : Student, friend w x → x ∈ T)) :=
by
  sorry  -- Proof omitted


end friends_belong_special_team_l192_192134


namespace roots_geom_prog_eq_neg_cbrt_c_l192_192602

theorem roots_geom_prog_eq_neg_cbrt_c {a b c : ℝ} (h : ∀ (x1 x2 x3 : ℝ), 
  (x1^3 + a * x1^2 + b * x1 + c = 0) ∧ (x2^3 + a * x2^2 + b * x2 + c = 0) ∧ (x3^3 + a * x3^2 + b * x3 + c = 0) ∧ 
  (∃ (r : ℝ), (x2 = r * x1) ∧ (x3 = r^2 * x1))) : 
  ∃ (x : ℝ), (x^3 = c) ∧ (x = - ((c) ^ (1/3))) :=
by 
  sorry

end roots_geom_prog_eq_neg_cbrt_c_l192_192602


namespace A_and_C_mutually_exclusive_l192_192921

/-- Definitions for the problem conditions. -/
def A (all_non_defective : Prop) : Prop := all_non_defective
def B (all_defective : Prop) : Prop := all_defective
def C (at_least_one_defective : Prop) : Prop := at_least_one_defective

/-- Theorem stating that A and C are mutually exclusive. -/
theorem A_and_C_mutually_exclusive (all_non_defective at_least_one_defective : Prop) :
  A all_non_defective ∧ C at_least_one_defective → false :=
  sorry

end A_and_C_mutually_exclusive_l192_192921


namespace three_digit_integer_conditions_l192_192593

theorem three_digit_integer_conditions:
  ∃ n : ℕ, 
    n % 5 = 3 ∧ 
    n % 7 = 4 ∧ 
    n % 4 = 2 ∧
    100 ≤ n ∧ n < 1000 ∧ 
    n = 548 :=
sorry

end three_digit_integer_conditions_l192_192593


namespace part1_l192_192654

theorem part1 (a b : ℝ) : 3*(a - b)^2 - 6*(a - b)^2 + 2*(a - b)^2 = - (a - b)^2 :=
by
  sorry

end part1_l192_192654


namespace Emma_age_ratio_l192_192457

theorem Emma_age_ratio (E M : ℕ) (h1 : E = E) (h2 : E = E) 
(h3 : E - M = 3 * (E - 4 * M)) : E / M = 11 / 2 :=
sorry

end Emma_age_ratio_l192_192457


namespace room_length_l192_192115

theorem room_length
  (width : ℝ)
  (cost_rate : ℝ)
  (total_cost : ℝ)
  (h_width : width = 4)
  (h_cost_rate : cost_rate = 850)
  (h_total_cost : total_cost = 18700) :
  ∃ L : ℝ, L = 5.5 ∧ total_cost = cost_rate * (L * width) :=
by
  sorry

end room_length_l192_192115


namespace find_a_l192_192391

def A (x : ℝ) : Prop := x^2 + 6 * x < 0
def B (a x : ℝ) : Prop := x^2 - (a - 2) * x - 2 * a < 0
def U (x : ℝ) : Prop := -6 < x ∧ x < 5

theorem find_a : (∀ x, A x ∨ ∃ a, B a x) = U x -> a = 5 :=
by
  sorry

end find_a_l192_192391


namespace lowest_height_l192_192109

noncomputable def length_A : ℝ := 2.4
noncomputable def length_B : ℝ := 3.2
noncomputable def length_C : ℝ := 2.8

noncomputable def height_Eunji : ℝ := 8 * length_A
noncomputable def height_Namjoon : ℝ := 4 * length_B
noncomputable def height_Hoseok : ℝ := 5 * length_C

theorem lowest_height :
  height_Namjoon = 12.8 ∧ 
  height_Namjoon < height_Eunji ∧ 
  height_Namjoon < height_Hoseok :=
by
  sorry

end lowest_height_l192_192109


namespace relationship_among_a_b_c_l192_192203

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_deriv : ∀ x ≠ 0, f'' x + f x / x > 0)

noncomputable def a : ℝ := (1 / Real.exp 1) * f (1 / Real.exp 1)
noncomputable def b : ℝ := -Real.exp 1 * f (-Real.exp 1)
noncomputable def c : ℝ := f 1

theorem relationship_among_a_b_c :
  a < c ∧ c < b :=
by
  -- sorry to skip the proof steps
  sorry

end relationship_among_a_b_c_l192_192203


namespace r_minus_s_l192_192358

-- Define the equation whose roots are r and s
def equation (x : ℝ) := (6 * x - 18) / (x ^ 2 + 4 * x - 21) = x + 3

-- Define the condition that r and s are distinct roots of the equation and r > s
def is_solution_pair (r s : ℝ) :=
  equation r ∧ equation s ∧ r ≠ s ∧ r > s

-- The main theorem we need to prove
theorem r_minus_s (r s : ℝ) (h : is_solution_pair r s) : r - s = 12 :=
by
  sorry

end r_minus_s_l192_192358


namespace value_of_y_at_x8_l192_192061

theorem value_of_y_at_x8 (k : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = k * x^(1 / 3)) (h2 : f 64 = 4) : f 8 = 2 :=
sorry

end value_of_y_at_x8_l192_192061


namespace sum_first_five_terms_geometric_sequence_l192_192288

noncomputable def sum_first_five_geometric (a0 : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a0 * (1 - r^n) / (1 - r)

theorem sum_first_five_terms_geometric_sequence : 
  sum_first_five_geometric (1/3) (1/3) 5 = 121 / 243 := 
by 
  sorry

end sum_first_five_terms_geometric_sequence_l192_192288


namespace evaluate_expr_l192_192528

theorem evaluate_expr :
  (150^2 - 12^2) / (90^2 - 21^2) * ((90 + 21) * (90 - 21)) / ((150 + 12) * (150 - 12)) = 2 :=
by sorry

end evaluate_expr_l192_192528


namespace curve_equation_l192_192243

noncomputable def curve_passing_condition (x y : ℝ) : Prop :=
  (∃ (f : ℝ → ℝ), f 2 = 3 ∧ ∀ (t : ℝ), (f t) * t = 6 ∧ ((t ≠ 0 ∧ f t ≠ 0) → (t, f t) = (x, y)))

theorem curve_equation (x y : ℝ) (h1 : curve_passing_condition x y) : x * y = 6 :=
  sorry

end curve_equation_l192_192243


namespace sum_of_squares_l192_192691

theorem sum_of_squares (x y z : ℕ) (h1 : x + y + z = 30)
  (h2 : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12) :
  x^2 + y^2 + z^2 = 504 :=
by
  sorry

end sum_of_squares_l192_192691


namespace correct_system_of_equations_l192_192662

theorem correct_system_of_equations (x y : ℝ) :
  (5 * x + 6 * y = 16) ∧ (4 * x + y = x + 5 * y) :=
sorry

end correct_system_of_equations_l192_192662


namespace product_of_geometric_progressions_is_geometric_general_function_form_geometric_l192_192280

variables {α β γ : Type*} [CommSemiring α] [CommSemiring β] [CommSemiring γ]

-- Define the terms of geometric progressions
def term (a r : α) (k : ℕ) : α := a * r ^ (k - 1)

-- Define a general function with respective powers
def general_term (a r : α) (k p : ℕ) : α := a ^ p * (r ^ p) ^ (k - 1)

theorem product_of_geometric_progressions_is_geometric
  {a b c : α} {r1 r2 r3 : α} (k : ℕ) :
  term a r1 k * term b r2 k * term c r3 k = 
  (a * b * c) * (r1 * r2 * r3) ^ (k - 1) := 
sorry

theorem general_function_form_geometric
  {a b c : α} {r1 r2 r3 : α} {p q r : ℕ} (k : ℕ) :
  general_term a r1 k p * general_term b r2 k q * general_term c r3 k r = 
  (a^p * b^q * c^r) * (r1^p * r2^q * r3^r) ^ (k - 1) := 
sorry

end product_of_geometric_progressions_is_geometric_general_function_form_geometric_l192_192280


namespace problem_l192_192159

namespace MathProof

variable {p a b : ℕ}

theorem problem (h1 : Nat.Prime p) (h2 : p % 2 = 1) (h3 : a > 0) (h4 : b > 0) (h5 : (p + 1)^a - p^b = 1) : a = 1 ∧ b = 1 := 
sorry

end MathProof

end problem_l192_192159


namespace range_of_a_l192_192325

open Real

-- Definitions based on given conditions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := ∀ (x : ℝ), x > 0 → -3^x ≤ a

-- The main proposition combining the conditions
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l192_192325


namespace complex_expression_l192_192520

theorem complex_expression (i : ℂ) (h₁ : i^2 = -1) (h₂ : i^4 = 1) :
  (i + i^3)^100 + (i + i^2 + i^3 + i^4 + i^5)^120 = 1 := by
  sorry

end complex_expression_l192_192520


namespace Lakota_spent_l192_192307

-- Define the conditions
def U : ℝ := 9.99
def Mackenzies_cost (N : ℝ) : ℝ := 3 * N + 8 * U
def cost_of_Lakotas_disks (N : ℝ) : ℝ := 6 * N + 2 * U

-- State the theorem
theorem Lakota_spent (N : ℝ) (h : Mackenzies_cost N = 133.89) : cost_of_Lakotas_disks N = 127.92 :=
by
  sorry

end Lakota_spent_l192_192307


namespace sum_of_vars_l192_192101

theorem sum_of_vars 
  (x y z : ℝ) 
  (h1 : x + y = 4) 
  (h2 : y + z = 6) 
  (h3 : z + x = 8) : 
  x + y + z = 9 := 
by 
  sorry

end sum_of_vars_l192_192101


namespace interval_length_l192_192104

theorem interval_length (c d : ℝ) (h : (d - 5) / 3 - (c - 5) / 3 = 15) : d - c = 45 :=
sorry

end interval_length_l192_192104


namespace symmetric_curve_wrt_line_l192_192200

theorem symmetric_curve_wrt_line {f : ℝ → ℝ → ℝ} :
  (∀ x y : ℝ, f x y = 0 → f (y + 3) (x - 3) = 0) := by
  sorry

end symmetric_curve_wrt_line_l192_192200


namespace base_conversion_least_sum_l192_192256

theorem base_conversion_least_sum (a b : ℕ) (h : 3 * a + 5 = 5 * b + 3) : a + b = 10 :=
sorry

end base_conversion_least_sum_l192_192256


namespace time_A_to_complete_race_l192_192000

noncomputable def km_race_time (V_B : ℕ) : ℚ :=
  940 / V_B

theorem time_A_to_complete_race : km_race_time 6 = 156.67 := by
  sorry

end time_A_to_complete_race_l192_192000


namespace min_score_guarantees_payoff_l192_192849

-- Defining the probability of a single roll being a six
def prob_single_six : ℚ := 1 / 6 

-- Defining the event of rolling two sixes independently
def prob_two_sixes : ℚ := prob_single_six * prob_single_six

-- Defining the score of two die rolls summing up to 12
def is_score_twelve (a b : ℕ) : Prop := a + b = 12

-- Proving the probability of Jim scoring 12 in two rolls guarantees some monetary payoff.
theorem min_score_guarantees_payoff :
  (prob_two_sixes = 1/36) :=
by
  sorry

end min_score_guarantees_payoff_l192_192849


namespace remaining_structure_volume_and_surface_area_l192_192433

-- Define the dimensions of the large cube and the small cubes
def large_cube_volume := 12 * 12 * 12
def small_cube_volume := 2 * 2 * 2

-- Define the number of smaller cubes in the large cube
def num_small_cubes := (12 / 2) * (12 / 2) * (12 / 2)

-- Define the number of smaller cubes removed (central on each face and very center)
def removed_cubes := 7

-- The volume of a small cube after removing its center unit
def single_small_cube_remaining_volume := small_cube_volume - 1

-- Calculate the remaining volume after all removals
def remaining_volume := (num_small_cubes - removed_cubes) * single_small_cube_remaining_volume

-- Initial surface area of a small cube and increase per removal of central unit
def single_small_cube_initial_surface_area := 6 * 4 -- 6 faces of 2*2*2 cube, each face has 4 units
def single_small_cube_surface_increase := 6

-- Calculate the adjusted surface area considering internal faces' reduction
def single_cube_adjusted_surface_area := single_small_cube_initial_surface_area + single_small_cube_surface_increase
def total_initial_surface_area := single_cube_adjusted_surface_area * (num_small_cubes - removed_cubes)
def total_internal_faces_area := (num_small_cubes - removed_cubes) * 2 * 4
def final_surface_area := total_initial_surface_area - total_internal_faces_area

theorem remaining_structure_volume_and_surface_area :
  remaining_volume = 1463 ∧ final_surface_area = 4598 :=
by
  -- Proof logic goes here
  sorry

end remaining_structure_volume_and_surface_area_l192_192433


namespace intersection_M_N_l192_192535

noncomputable def set_M : Set ℝ := {x | x^2 - 3 * x - 4 ≤ 0}
noncomputable def set_N : Set ℝ := {x | Real.log x ≥ 0}

theorem intersection_M_N :
  {x | x ∈ set_M ∧ x ∈ set_N} = {x | 1 ≤ x ∧ x ≤ 4} :=
sorry

end intersection_M_N_l192_192535


namespace roots_ratio_sum_l192_192890

theorem roots_ratio_sum (α β : ℝ) (hαβ : α > β) (h1 : 3*α^2 + α - 1 = 0) (h2 : 3*β^2 + β - 1 = 0) :
  α / β + β / α = -7 / 3 :=
sorry

end roots_ratio_sum_l192_192890


namespace sector_area_l192_192317

theorem sector_area (r θ : ℝ) (hr : r = 2) (hθ : θ = (45 : ℝ) * (Real.pi / 180)) : 
  (1 / 2) * r^2 * θ = Real.pi / 2 := 
by
  sorry

end sector_area_l192_192317


namespace ratio_blue_to_total_l192_192937

theorem ratio_blue_to_total (total_marbles red_marbles green_marbles yellow_marbles blue_marbles : ℕ)
    (h_total : total_marbles = 164)
    (h_red : red_marbles = total_marbles / 4)
    (h_green : green_marbles = 27)
    (h_yellow : yellow_marbles = 14)
    (h_blue : blue_marbles = total_marbles - (red_marbles + green_marbles + yellow_marbles)) :
  blue_marbles / total_marbles = 1 / 2 :=
by
  sorry

end ratio_blue_to_total_l192_192937


namespace pencils_multiple_of_40_l192_192469

theorem pencils_multiple_of_40 :
  ∃ n : ℕ, 640 % n = 0 ∧ n ≤ 40 → ∃ m : ℕ, 40 * m = 40 * n :=
by
  sorry

end pencils_multiple_of_40_l192_192469


namespace ratio_of_times_gina_chooses_to_her_sister_l192_192246

theorem ratio_of_times_gina_chooses_to_her_sister (sister_shows : ℕ) (minutes_per_show : ℕ) (gina_minutes : ℕ) (ratio : ℕ × ℕ) :
  sister_shows = 24 →
  minutes_per_show = 50 →
  gina_minutes = 900 →
  ratio = (900 / Nat.gcd 900 1200, 1200 / Nat.gcd 900 1200) →
  ratio = (3, 4) :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_times_gina_chooses_to_her_sister_l192_192246
