import Mathlib

namespace tetrahedron_ineq_l881_88161

variable (P Q R S : ℝ)

-- Given conditions
axiom ortho_condition : S^2 = P^2 + Q^2 + R^2

theorem tetrahedron_ineq (P Q R S : ℝ) (ortho_condition : S^2 = P^2 + Q^2 + R^2) :
  (P + Q + R) / S ≤ Real.sqrt 3 := by
  sorry

end tetrahedron_ineq_l881_88161


namespace maria_trip_distance_l881_88185

variable (D : ℕ) -- Defining the total distance D as a natural number

-- Defining the conditions given in the problem
def first_stop_distance := D / 2
def second_stop_distance := first_stop_distance - (1 / 3 * first_stop_distance)
def third_stop_distance := second_stop_distance - (2 / 5 * second_stop_distance)
def remaining_distance := 180

-- The statement to prove
theorem maria_trip_distance : third_stop_distance = remaining_distance → D = 900 :=
by
  sorry

end maria_trip_distance_l881_88185


namespace distance_centers_triangle_l881_88180

noncomputable def distance_between_centers (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := K / s
  let circumradius := (a * b * c) / (4 * K)
  let hypotenuse := by
    by_cases hc : a * a + b * b = c * c
    exact c
    by_cases hb : a * a + c * c = b * b
    exact b
    by_cases ha : b * b + c * c = a * a
    exact a
    exact 0
  let oc := hypotenuse / 2
  Real.sqrt (oc * oc + r * r)

theorem distance_centers_triangle :
  distance_between_centers 7 24 25 = Real.sqrt 165.25 := sorry

end distance_centers_triangle_l881_88180


namespace conclusion_1_conclusion_3_l881_88136

def tensor (a b : ℝ) : ℝ := a * (1 - b)

theorem conclusion_1 : tensor 2 (-2) = 6 :=
by sorry

theorem conclusion_3 (a b : ℝ) (h : a + b = 0) : tensor a a + tensor b b = 2 * a * b :=
by sorry

end conclusion_1_conclusion_3_l881_88136


namespace problem_statement_l881_88175

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f a b α β 2007 = 5) :
  f a b α β 2008 = 3 := 
by
  sorry

end problem_statement_l881_88175


namespace complex_number_solution_l881_88107

def i : ℂ := Complex.I

theorem complex_number_solution (z : ℂ) (h : z * (1 - i) = 2 * i) : z = -1 + i :=
by
  sorry

end complex_number_solution_l881_88107


namespace average_mark_of_all_three_boys_is_432_l881_88130

noncomputable def max_score : ℝ := 900
noncomputable def get_score (percent : ℝ) : ℝ := (percent / 100) * max_score

noncomputable def amar_score : ℝ := get_score 64
noncomputable def bhavan_score : ℝ := get_score 36
noncomputable def chetan_score : ℝ := get_score 44

noncomputable def total_score : ℝ := amar_score + bhavan_score + chetan_score
noncomputable def average_score : ℝ := total_score / 3

theorem average_mark_of_all_three_boys_is_432 : average_score = 432 := 
by
  sorry

end average_mark_of_all_three_boys_is_432_l881_88130


namespace probability_of_rolling_perfect_square_l881_88149

theorem probability_of_rolling_perfect_square :
  (3 / 12 : ℚ) = 1 / 4 :=
by
  sorry

end probability_of_rolling_perfect_square_l881_88149


namespace necessarily_positive_y_plus_xsq_l881_88160

theorem necessarily_positive_y_plus_xsq {x y z : ℝ} 
  (hx : 0 < x ∧ x < 2) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 1) : 
  y + x^2 > 0 :=
sorry

end necessarily_positive_y_plus_xsq_l881_88160


namespace largest_divisor_of_n4_minus_n_l881_88179

theorem largest_divisor_of_n4_minus_n (n : ℕ) (h : ¬(Prime n) ∧ n ≠ 1) : 6 ∣ (n^4 - n) :=
by sorry

end largest_divisor_of_n4_minus_n_l881_88179


namespace find_ab_l881_88183

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by 
  sorry

end find_ab_l881_88183


namespace correct_subtraction_l881_88109

theorem correct_subtraction (x : ℕ) (h : x - 32 = 25) : x - 23 = 34 :=
by
  sorry

end correct_subtraction_l881_88109


namespace mixed_operations_with_decimals_false_l881_88188

-- Definitions and conditions
def operations_same_level_with_decimals : Prop :=
  ∀ (a b c : ℝ), a + b - c = (a + b) - c

def calculate_left_to_right_with_decimals : Prop :=
  ∀ (a b c : ℝ), (a - b + c) = a - b + c ∧ (a + b - c) = a + b - c

-- Proposition we're proving
theorem mixed_operations_with_decimals_false :
  ¬ ∀ (a b c : ℝ), (a + b - c) ≠ (a - b + c) :=
by
  intro h
  sorry

end mixed_operations_with_decimals_false_l881_88188


namespace probability_of_red_then_blue_is_correct_l881_88101

noncomputable def probability_red_then_blue : ℚ :=
  let total_marbles := 5 + 4 + 12 + 2
  let prob_red := 5 / total_marbles
  let remaining_marbles := total_marbles - 1
  let prob_blue_given_red := 2 / remaining_marbles
  prob_red * prob_blue_given_red

theorem probability_of_red_then_blue_is_correct :
  probability_red_then_blue = 5 / 253 := 
by 
  sorry

end probability_of_red_then_blue_is_correct_l881_88101


namespace trig_inequalities_l881_88151

theorem trig_inequalities :
  let sin_168 := Real.sin (168 * Real.pi / 180)
  let cos_10 := Real.cos (10 * Real.pi / 180)
  let tan_58 := Real.tan (58 * Real.pi / 180)
  let tan_45 := Real.tan (45 * Real.pi / 180)
  sin_168 < cos_10 ∧ cos_10 < tan_58 :=
  sorry

end trig_inequalities_l881_88151


namespace area_of_rectangle_R_l881_88168

-- Define the side lengths of the squares and rectangles involved
def larger_square_side := 4
def smaller_square_side := 2
def rectangle_side1 := 1
def rectangle_side2 := 4

-- The areas of these shapes
def area_larger_square := larger_square_side * larger_square_side
def area_smaller_square := smaller_square_side * smaller_square_side
def area_first_rectangle := rectangle_side1 * rectangle_side2

-- Define the sum of all possible values for the area of rectangle R
def area_remaining := area_larger_square - (area_smaller_square + area_first_rectangle)

theorem area_of_rectangle_R : area_remaining = 8 := sorry

end area_of_rectangle_R_l881_88168


namespace ratio_a7_b7_l881_88195

variable {α : Type*}
variables {a_n b_n : ℕ → α} [AddGroup α] [Field α]
variables {S_n T_n : ℕ → α}

-- Define the sum of the first n terms for sequences a_n and b_n
def sum_of_first_terms_a (n : ℕ) := S_n n = (n * (a_n n + a_n (n-1))) / 2
def sum_of_first_terms_b (n : ℕ) := T_n n = (n * (b_n n + b_n (n-1))) / 2

-- Given condition about the ratio of sums
axiom ratio_condition (n : ℕ) : S_n n / T_n n = (3 * n - 2) / (2 * n + 1)

-- The statement to be proved
theorem ratio_a7_b7 : (a_n 7 / b_n 7) = (37 / 27) := sorry

end ratio_a7_b7_l881_88195


namespace rebecca_charge_for_dye_job_l881_88122

def charges_for_services (haircuts per perms per dye_jobs hair_dye_per_dye_job tips : ℕ) : ℕ := 
  4 * 30 + 1 * 40 + 2 * (dye_jobs - hair_dye_per_dye_job) + tips

theorem rebecca_charge_for_dye_job 
  (haircuts: ℕ) (perms: ℕ) (hair_dye_per_dye_job: ℕ) (tips: ℕ) (end_of_day_amount: ℕ) : 
  haircuts = 4 → perms = 1 → hair_dye_per_dye_job = 10 → tips = 50 → 
  end_of_day_amount = 310 → 
  ∃ D: ℕ, D = 60 := 
by
  sorry

end rebecca_charge_for_dye_job_l881_88122


namespace equivalent_single_discount_l881_88144

theorem equivalent_single_discount :
  ∀ (x : ℝ), ((1 - 0.15) * (1 - 0.10) * (1 - 0.05) * x) = (1 - 0.273) * x :=
by
  intros x
  --- This proof is left blank intentionally.
  sorry

end equivalent_single_discount_l881_88144


namespace evaluate_f_l881_88118

def f (x : ℚ) : ℚ := (2 * x - 3) / (3 * x ^ 2 - 1)

theorem evaluate_f :
  f (-2) = -7 / 11 ∧ f (0) = 3 ∧ f (1) = -1 / 2 :=
by
  sorry

end evaluate_f_l881_88118


namespace trigonometric_identity_l881_88156

theorem trigonometric_identity (α : Real) (h : Real.tan (α + Real.pi / 4) = -3) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 9 / 5 :=
sorry

end trigonometric_identity_l881_88156


namespace maximize_Sn_l881_88123

def a_n (n : ℕ) : ℤ := 26 - 2 * n

def S_n (n : ℕ) : ℤ := n * (26 - 2 * (n + 1)) / 2 + 26 * n

theorem maximize_Sn : (n = 12 ∨ n = 13) ↔ (∀ m : ℕ, S_n m ≤ S_n 12 ∨ S_n m ≤ S_n 13) :=
by sorry

end maximize_Sn_l881_88123


namespace path_count_from_E_to_G_passing_through_F_l881_88142

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem path_count_from_E_to_G_passing_through_F :
  let E := (0, 0)
  let F := (5, 2)
  let G := (6, 5)
  ∃ (paths_EF paths_FG total_paths : ℕ),
  paths_EF = binom (5 + 2) 5 ∧
  paths_FG = binom (1 + 3) 1 ∧
  total_paths = paths_EF * paths_FG ∧
  total_paths = 84 := 
by
  sorry

end path_count_from_E_to_G_passing_through_F_l881_88142


namespace value_of_f_at_5_l881_88145

theorem value_of_f_at_5 (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = - f x) 
  (h_period : ∀ x, f (x + 4) = f x)
  (h_func : ∀ x, -2 ≤ x ∧ x < 0 → f x = 3 * x + 1) : 
  f 5 = 2 :=
  sorry

end value_of_f_at_5_l881_88145


namespace triangle_statements_l881_88170

-- Definitions of internal angles and sides of the triangle
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = Real.pi ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Statement A: If ABC is an acute triangle, then sin A > cos B
lemma statement_A (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2) :
  Real.sin A > Real.cos B := 
sorry

-- Statement B: If A > B, then sin A > sin B
lemma statement_B (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_AB : A > B) : 
  Real.sin A > Real.sin B := 
sorry

-- Statement C: If ABC is a non-right triangle, then tan A + tan B + tan C = tan A * tan B * tan C
lemma statement_C (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_non_right : A ≠ Real.pi / 2 ∧ B ≠ Real.pi / 2 ∧ C ≠ Real.pi / 2) : 
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := 
sorry

-- Statement D: If a cos A = b cos B, then triangle ABC must be isosceles
lemma statement_D (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_cos : a * Real.cos A = b * Real.cos B) : 
  ¬(A = B) ∧ ¬(B = C) := 
sorry

-- Theorem to combine all statements
theorem triangle_statements (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2)
  (h_AB : A > B)
  (h_non_right : A ≠ Real.pi / 2 ∧ B ≠ Real.pi / 2 ∧ C ≠ Real.pi / 2)
  (h_cos : a * Real.cos A = b * Real.cos B) : 
  Real.sin A > Real.cos B ∧ Real.sin A > Real.sin B ∧ 
  (Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) ∧ 
  ¬(A = B) ∧ ¬(B = C) := 
by
  exact ⟨statement_A A B C a b c h_triangle h_acute, statement_B A B C a b c h_triangle h_AB, statement_C A B C a b c h_triangle h_non_right, statement_D A B C a b c h_triangle h_cos⟩

end triangle_statements_l881_88170


namespace probability_one_hits_l881_88173

theorem probability_one_hits (P_A P_B : ℝ) (h_A : P_A = 0.6) (h_B : P_B = 0.6) :
  (P_A * (1 - P_B) + (1 - P_A) * P_B) = 0.48 :=
by
  sorry

end probability_one_hits_l881_88173


namespace band_formation_max_l881_88176

-- Define the conditions provided in the problem
theorem band_formation_max (m r x : ℕ) (h1 : m = r * x + 5)
  (h2 : (r - 3) * (x + 2) = m) (h3 : m < 100) :
  m = 70 :=
sorry

end band_formation_max_l881_88176


namespace express_set_A_l881_88157

def A := {x : ℤ | -1 < abs (x - 1) ∧ abs (x - 1) < 2}

theorem express_set_A : A = {0, 1, 2} := 
by
  sorry

end express_set_A_l881_88157


namespace option_a_is_fraction_option_b_is_fraction_option_c_is_fraction_option_d_is_fraction_l881_88197

section

variable (π : Real) (x : Real)

-- Definition of a fraction in this context
def is_fraction (num denom : Real) : Prop := denom ≠ 0

-- Proving each given option is a fraction
theorem option_a_is_fraction : is_fraction 1 π := 
sorry

theorem option_b_is_fraction : is_fraction x 3 :=
sorry

theorem option_c_is_fraction : is_fraction 2 5 :=
sorry

theorem option_d_is_fraction : is_fraction 1 (x - 1) :=
sorry

end

end option_a_is_fraction_option_b_is_fraction_option_c_is_fraction_option_d_is_fraction_l881_88197


namespace determine_k_l881_88113

theorem determine_k (k r s : ℝ) (h1 : r + s = -k) (h2 : (r + 3) + (s + 3) = k) : k = 3 :=
by
  sorry

end determine_k_l881_88113


namespace travel_time_comparison_l881_88115

theorem travel_time_comparison
  (v : ℝ) -- speed during the first trip
  (t1 : ℝ) (t2 : ℝ)
  (h1 : t1 = 80 / v) -- time for the first trip
  (h2 : t2 = 100 / v) -- time for the second trip
  : t2 = 1.25 * t1 :=
by
  sorry

end travel_time_comparison_l881_88115


namespace quadratic_eq_unique_k_l881_88132

theorem quadratic_eq_unique_k (k : ℝ) (x1 x2 : ℝ) 
  (h_quad : x1^2 - 3*x1 + k = 0 ∧ x2^2 - 3*x2 + k = 0)
  (h_cond : x1 * x2 + 2 * x1 + 2 * x2 = 1) : k = -5 :=
by 
  sorry

end quadratic_eq_unique_k_l881_88132


namespace max_groups_l881_88106

theorem max_groups (cards : ℕ) (sum_group : ℕ) (c5 c2 c1 : ℕ) (cond1 : cards = 600) (cond2 : c5 = 200)
  (cond3 : c2 = 200) (cond4 : c1 = 200) (cond5 : sum_group = 9) :
  ∃ max_g : ℕ, max_g = 100 :=
by
  sorry

end max_groups_l881_88106


namespace cost_of_six_hotdogs_and_seven_burgers_l881_88153

theorem cost_of_six_hotdogs_and_seven_burgers :
  ∀ (h b : ℝ), 4 * h + 5 * b = 3.75 → 5 * h + 3 * b = 3.45 → 6 * h + 7 * b = 5.43 :=
by
  intros h b h_eqn b_eqn
  sorry

end cost_of_six_hotdogs_and_seven_burgers_l881_88153


namespace candy_bar_cost_is_7_l881_88114

-- Define the conditions
def chocolate_cost : Nat := 3
def candy_additional_cost : Nat := 4

-- Define the expression for the cost of the candy bar
def candy_cost : Nat := chocolate_cost + candy_additional_cost

-- State the theorem to prove the cost of the candy bar
theorem candy_bar_cost_is_7 : candy_cost = 7 :=
by
  sorry

end candy_bar_cost_is_7_l881_88114


namespace calc_c_15_l881_88152

noncomputable def c : ℕ → ℝ
| 0 => 1 -- This case won't be used, setup for pattern match
| 1 => 3
| 2 => 5
| (n+3) => c (n+2) * c (n+1)

theorem calc_c_15 : c 15 = 3 ^ 235 :=
sorry

end calc_c_15_l881_88152


namespace complex_division_l881_88198

-- Conditions: i is the imaginary unit
def i : ℂ := Complex.I

-- Question: Prove the complex division
theorem complex_division (h : i = Complex.I) : (8 - i) / (2 + i) = 3 - 2 * i :=
by sorry

end complex_division_l881_88198


namespace total_money_l881_88194

-- Conditions
def mark_amount : ℚ := 5 / 6
def carolyn_amount : ℚ := 2 / 5

-- Combine both amounts and state the theorem to be proved
theorem total_money : mark_amount + carolyn_amount = 1.233 := by
  -- placeholder for the actual proof
  sorry

end total_money_l881_88194


namespace prism_sphere_surface_area_l881_88103

theorem prism_sphere_surface_area :
  ∀ (a b c : ℝ), (a * b = 6) → (b * c = 2) → (a * c = 3) → 
  4 * Real.pi * ((Real.sqrt ((a ^ 2) + (b ^ 2) + (c ^ 2))) / 2) ^ 2 = 14 * Real.pi :=
by
  intros a b c hab hbc hac
  sorry

end prism_sphere_surface_area_l881_88103


namespace range_of_a_l881_88147

theorem range_of_a (a : ℝ) : (1 ∉ {x : ℝ | (x - a) / (x + a) < 0}) → ( -1 ≤ a ∧ a ≤ 1 ) := 
by
  intro h
  sorry

end range_of_a_l881_88147


namespace remainder_500th_in_T_l881_88172

def sequence_T (n : ℕ) : ℕ := sorry -- Assume a definition for the sequence T where n represents the position and the sequence contains numbers having exactly 9 ones in their binary representation.

theorem remainder_500th_in_T :
  (sequence_T 500) % 500 = 191 := 
sorry

end remainder_500th_in_T_l881_88172


namespace max_value_of_x_l881_88190

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem max_value_of_x 
  (x : ℤ) 
  (h : log_base (1 / 4 : ℝ) (2 * x + 1) < log_base (1 / 2 : ℝ) (x - 1)) : x ≤ 3 :=
sorry

end max_value_of_x_l881_88190


namespace number_of_ways_split_2000_cents_l881_88133

theorem number_of_ways_split_2000_cents : 
  ∃ n : ℕ, n = 357 ∧ (∃ (nick d q : ℕ), 
    nick > 0 ∧ d > 0 ∧ q > 0 ∧ 5 * nick + 10 * d + 25 * q = 2000) :=
sorry

end number_of_ways_split_2000_cents_l881_88133


namespace second_fisherman_more_fish_l881_88129

-- Defining the conditions
def total_days : ℕ := 213
def first_fisherman_rate : ℕ := 3
def second_fisherman_rate1 : ℕ := 1
def second_fisherman_rate2 : ℕ := 2
def second_fisherman_rate3 : ℕ := 4
def days_rate1 : ℕ := 30
def days_rate2 : ℕ := 60
def days_rate3 : ℕ := total_days - (days_rate1 + days_rate2)

-- Calculating the total number of fish caught by both fishermen
def total_fish_first_fisherman : ℕ := first_fisherman_rate * total_days
def total_fish_second_fisherman : ℕ := (second_fisherman_rate1 * days_rate1) + 
                                        (second_fisherman_rate2 * days_rate2) + 
                                        (second_fisherman_rate3 * days_rate3)

-- Theorem stating the difference in the number of fish caught
theorem second_fisherman_more_fish : (total_fish_second_fisherman - total_fish_first_fisherman) = 3 := 
by
  sorry

end second_fisherman_more_fish_l881_88129


namespace board_total_length_l881_88112

-- Definitions based on conditions
def S : ℝ := 2
def L : ℝ := 2 * S

-- Define the total length of the board
def T : ℝ := S + L

-- The theorem asserting the total length of the board is 6 ft
theorem board_total_length : T = 6 := 
by
  sorry

end board_total_length_l881_88112


namespace min_value_is_five_l881_88100

noncomputable def min_value (x y : ℝ) : ℝ :=
  if x + 3 * y = 5 * x * y then 3 * x + 4 * y else 0

theorem min_value_is_five {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : min_value x y = 5 :=
by
  sorry

end min_value_is_five_l881_88100


namespace polynomial_expansion_sum_l881_88182

theorem polynomial_expansion_sum (a_6 a_5 a_4 a_3 a_2 a_1 a : ℝ) :
  (∀ x : ℝ, (3 * x - 1)^6 = a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) →
  a_6 + a_5 + a_4 + a_3 + a_2 + a_1 + a = 64 :=
by
  -- Proof is not needed, placeholder here.
  sorry

end polynomial_expansion_sum_l881_88182


namespace A_time_240m_race_l881_88166

theorem A_time_240m_race (t : ℕ) :
  (∀ t, (240 / t) = (184 / t) * (t + 7) ∧ 240 = 184 + ((184 * 7) / t)) → t = 23 :=
by
  sorry

end A_time_240m_race_l881_88166


namespace reggies_brother_long_shots_l881_88128

-- Define the number of points per type of shot
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define the number of shots made by Reggie
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := 1

-- Define the total number of points made by Reggie
def reggie_points : ℕ :=
  reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points

-- Define the total points by which Reggie loses
def points_lost_by : ℕ := 2

-- Prove the number of long shots made by Reggie's brother
theorem reggies_brother_long_shots : 
  (reggie_points + points_lost_by) / long_shot_points = 4 := by
  sorry

end reggies_brother_long_shots_l881_88128


namespace jane_sleep_hours_for_second_exam_l881_88138

theorem jane_sleep_hours_for_second_exam :
  ∀ (score1 score2 hours1 hours2 : ℝ),
  score1 * hours1 = 675 →
  (score1 + score2) / 2 = 85 →
  score2 * hours2 = 675 →
  hours2 = 135 / 19 :=
by
  intros score1 score2 hours1 hours2 h1 h2 h3
  sorry

end jane_sleep_hours_for_second_exam_l881_88138


namespace johns_age_is_15_l881_88125

-- Definitions from conditions
variables (J F : ℕ) -- J is John's age, F is his father's age
axiom sum_of_ages : J + F = 77
axiom father_age : F = 2 * J + 32

-- Target statement to prove
theorem johns_age_is_15 : J = 15 :=
by
  sorry

end johns_age_is_15_l881_88125


namespace problem_l881_88104

def gcf (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem problem (A B : ℕ) (hA : A = gcf 9 15 27) (hB : B = lcm 9 15 27) : A + B = 138 :=
by
  sorry

end problem_l881_88104


namespace abs_diff_gt_half_prob_l881_88127

noncomputable def probability_abs_diff_gt_half : ℝ :=
  ((1 / 4) * (1 / 8) + 
   (1 / 8) * (1 / 2) + 
   (1 / 8) * 1) * 2

theorem abs_diff_gt_half_prob : probability_abs_diff_gt_half = 5 / 16 := by 
  sorry

end abs_diff_gt_half_prob_l881_88127


namespace simplify_fraction_l881_88162

theorem simplify_fraction :
  (5 : ℚ) / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 27) = Real.sqrt 3 / 12 := by
sorry

end simplify_fraction_l881_88162


namespace dan_initial_amount_l881_88111

theorem dan_initial_amount (left_amount : ℕ) (candy_cost : ℕ) : left_amount = 3 ∧ candy_cost = 2 → left_amount + candy_cost = 5 :=
by
  sorry

end dan_initial_amount_l881_88111


namespace election_win_by_votes_l881_88199

/-- Two candidates in an election, the winner received 56% of votes and won the election
by receiving 1344 votes. We aim to prove that the winner won by 288 votes. -/
theorem election_win_by_votes
  (V : ℝ)  -- total number of votes
  (w : ℝ)  -- percentage of votes received by the winner
  (w_votes : ℝ)  -- votes received by the winner
  (l_votes : ℝ)  -- votes received by the loser
  (w_percentage : w = 0.56)
  (w_votes_given : w_votes = 1344)
  (total_votes : V = 1344 / 0.56)
  (l_votes_calc : l_votes = (V * 0.44)) :
  1344 - l_votes = 288 :=
by
  -- Proof goes here
  sorry

end election_win_by_votes_l881_88199


namespace cubic_no_maximum_value_l881_88193

theorem cubic_no_maximum_value (x : ℝ) : ¬ ∃ M, ∀ x : ℝ, 3 * x^2 + 6 * x^3 + 27 * x + 100 ≤ M := 
by
  sorry

end cubic_no_maximum_value_l881_88193


namespace garden_breadth_l881_88105

theorem garden_breadth (P L B : ℕ) (h1 : P = 700) (h2 : L = 250) (h3 : P = 2 * (L + B)) : B = 100 :=
by
  sorry

end garden_breadth_l881_88105


namespace ratio_books_donated_l881_88110

theorem ratio_books_donated (initial_books: ℕ) (books_given_nephew: ℕ) (books_after_nephew: ℕ) 
  (books_final: ℕ) (books_purchased: ℕ) (books_donated_library: ℕ) (ratio: ℕ):
    initial_books = 40 → 
    books_given_nephew = initial_books / 4 → 
    books_after_nephew = initial_books - books_given_nephew →
    books_final = 23 →
    books_purchased = 3 →
    books_donated_library = books_after_nephew - (books_final - books_purchased) →
    ratio = books_donated_library / books_after_nephew →
    ratio = 1 / 3 := sorry

end ratio_books_donated_l881_88110


namespace number_of_penguins_l881_88108

-- Define the number of animals and zookeepers
def zebras : ℕ := 22
def tigers : ℕ := 8
def zookeepers : ℕ := 12
def headsLessThanFeetBy : ℕ := 132

-- Define the theorem to prove the number of penguins (P)
theorem number_of_penguins (P : ℕ) (H : P + zebras + tigers + zookeepers + headsLessThanFeetBy = 4 * P + 4 * zebras + 4 * tigers + 2 * zookeepers) : P = 10 :=
by
  sorry

end number_of_penguins_l881_88108


namespace problem_statement_l881_88164

noncomputable def probability_different_colors : ℚ :=
  let p_red := 7 / 11
  let p_green := 4 / 11
  (p_red * p_green) + (p_green * p_red)

theorem problem_statement :
  let p_red := 7 / 11
  let p_green := 4 / 11
  (p_red * p_green) + (p_green * p_red) = 56 / 121 := by
  sorry

end problem_statement_l881_88164


namespace probability_of_bug9_is_zero_l881_88102

-- Definitions based on conditions provided
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def non_vowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
def digits_or_vowels : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'E', 'I', 'O', 'U']

-- Defining the number of choices for each position
def first_symbol_choices : Nat := 5
def second_symbol_choices : Nat := 21
def third_symbol_choices : Nat := 20
def fourth_symbol_choices : Nat := 15

-- Total number of possible license plates
def total_plates : Nat := first_symbol_choices * second_symbol_choices * third_symbol_choices * fourth_symbol_choices

-- Probability calculation for the specific license plate "BUG9"
def probability_bug9 : Nat := 0

theorem probability_of_bug9_is_zero : probability_bug9 = 0 := by sorry

end probability_of_bug9_is_zero_l881_88102


namespace domain_of_f_l881_88167

noncomputable def f (x : ℝ) : ℝ := 1 / x + Real.sqrt (-x^2 + x + 2)

theorem domain_of_f :
  {x : ℝ | -1 ≤ x ∧ x ≤ 2 ∧ x ≠ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2 ∧ x ≠ 0} :=
by
  sorry

end domain_of_f_l881_88167


namespace december_28_is_saturday_l881_88169

def days_per_week := 7

def thanksgiving_day : Nat := 28

def november_length : Nat := 30

def december_28_day_of_week : Nat :=
  (thanksgiving_day % days_per_week + november_length + 28 - thanksgiving_day) % days_per_week

theorem december_28_is_saturday :
  (december_28_day_of_week = 6) :=
by
  sorry

end december_28_is_saturday_l881_88169


namespace infinite_solutions_l881_88177

-- Define the system of linear equations
def eq1 (x y : ℝ) : Prop := 3 * x - 4 * y = 1
def eq2 (x y : ℝ) : Prop := 6 * x - 8 * y = 2

-- State that there are an unlimited number of solutions
theorem infinite_solutions : ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧
  ∀ y : ℝ, ∃ x : ℝ, eq1 x y :=
by
  sorry

end infinite_solutions_l881_88177


namespace inverseP_l881_88143

-- Mathematical definitions
def isOdd (a : ℕ) : Prop := a % 2 = 1
def isPrime (a : ℕ) : Prop := Nat.Prime a

-- Given proposition P (hypothesis)
def P (a : ℕ) : Prop := isOdd a → isPrime a

-- Inverse proposition: if a is prime, then a is odd
theorem inverseP (a : ℕ) (h : isPrime a) : isOdd a :=
sorry

end inverseP_l881_88143


namespace floor_sqrt_23_squared_eq_16_l881_88119

theorem floor_sqrt_23_squared_eq_16 :
  (Int.floor (Real.sqrt 23))^2 = 16 :=
by
  have h1 : 4 < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < 5 := sorry
  have floor_sqrt_23 : Int.floor (Real.sqrt 23) = 4 := sorry
  rw [floor_sqrt_23]
  norm_num

end floor_sqrt_23_squared_eq_16_l881_88119


namespace find_a_l881_88141

noncomputable def A : Set ℝ := {1, 2, 3, 4}
noncomputable def B (a : ℝ) : Set ℝ := { x | x ≤ a }

theorem find_a (a : ℝ) (h_union : A ∪ B a = Set.Iic 5) : a = 5 := by
  sorry

end find_a_l881_88141


namespace operation_proof_l881_88124

def operation (x y : ℤ) : ℤ := x * y - 3 * x - 4 * y

theorem operation_proof : (operation 7 2) - (operation 2 7) = 5 :=
by
  sorry

end operation_proof_l881_88124


namespace mike_books_l881_88146

theorem mike_books : 51 - 45 = 6 := 
by 
  rfl

end mike_books_l881_88146


namespace ratio_of_cars_to_trucks_l881_88116

-- Definitions based on conditions
def total_vehicles : ℕ := 60
def trucks : ℕ := 20
def cars : ℕ := total_vehicles - trucks

-- Theorem to prove
theorem ratio_of_cars_to_trucks : (cars / trucks : ℚ) = 2 := by
  sorry

end ratio_of_cars_to_trucks_l881_88116


namespace find_k_l881_88187

def vector := (ℝ × ℝ)

def a : vector := (3, 1)
def b : vector := (1, 3)
def c (k : ℝ) : vector := (k, 2)

def subtract (v1 v2 : vector) : vector :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_k (k : ℝ) (h : dot_product (subtract a (c k)) b = 0) : k = 0 := by
  sorry

end find_k_l881_88187


namespace find_XY_length_l881_88174

variables (a b c : ℝ) -- sides of triangle ABC
variables (s : ℝ) -- semi-perimeter s = (a + b + c) / 2

-- Definition of similar triangles and perimeter condition
noncomputable def XY_length
  (AX : ℝ) (XY : ℝ) (AY : ℝ) (BX : ℝ) 
  (BC : ℝ) (CY : ℝ) 
  (h1 : AX + AY + XY = BX + BC + CY)
  (h2 : AX = c * XY / a) 
  (h3 : AY = b * XY / a) : ℝ :=
  s * a / (b + c) -- by the given solution

-- The theorem statement
theorem find_XY_length
  (a b c : ℝ) (s : ℝ) -- given sides and semi-perimeter
  (AX : ℝ) (XY : ℝ) (AY : ℝ) (BX : ℝ)
  (BC : ℝ) (CY : ℝ) 
  (h1 : AX + AY + XY = BX + BC + CY)
  (h2 : AX = c * XY / a) 
  (h3 : AY = b * XY / a) :
  XY = s * a / (b + c) :=
sorry -- proof


end find_XY_length_l881_88174


namespace frog_hops_ratio_l881_88148

theorem frog_hops_ratio :
  ∀ (F1 F2 F3 : ℕ),
    F1 = 4 * F2 →
    F1 + F2 + F3 = 99 →
    F2 = 18 →
    (F2 : ℚ) / (F3 : ℚ) = 2 :=
by
  intros F1 F2 F3 h1 h2 h3
  -- algebraic manipulations and proof to be filled here
  sorry

end frog_hops_ratio_l881_88148


namespace john_annual_patients_l881_88131

-- Definitions for the various conditions
def first_hospital_patients_per_day := 20
def second_hospital_patients_per_day := first_hospital_patients_per_day + (first_hospital_patients_per_day * 20 / 100)
def third_hospital_patients_per_day := first_hospital_patients_per_day + (first_hospital_patients_per_day * 15 / 100)
def total_patients_per_day := first_hospital_patients_per_day + second_hospital_patients_per_day + third_hospital_patients_per_day
def workdays_per_week := 5
def total_patients_per_week := total_patients_per_day * workdays_per_week
def working_weeks_per_year := 50 - 2 -- considering 2 weeks of vacation
def total_patients_per_year := total_patients_per_week * working_weeks_per_year

-- The statement to prove
theorem john_annual_patients : total_patients_per_year = 16080 := by
  sorry

end john_annual_patients_l881_88131


namespace find_sin_expression_l881_88134

noncomputable def trigonometric_identity (γ : ℝ) : Prop :=
  3 * (Real.tan γ)^2 + 3 * (1 / (Real.tan γ))^2 + 2 / (Real.sin γ)^2 + 2 / (Real.cos γ)^2 = 19

theorem find_sin_expression (γ : ℝ) (h : trigonometric_identity γ) : 
  (Real.sin γ)^4 - (Real.sin γ)^2 = -1 / 5 :=
sorry

end find_sin_expression_l881_88134


namespace shifted_parabola_sum_constants_l881_88137

theorem shifted_parabola_sum_constants :
  let a := 2
  let b := -17
  let c := 43
  a + b + c = 28 := sorry

end shifted_parabola_sum_constants_l881_88137


namespace final_value_of_A_l881_88181

theorem final_value_of_A : 
  ∀ (A : Int), 
    (A = 20) → 
    (A = -A + 10) → 
    A = -10 :=
by
  intros A h1 h2
  sorry

end final_value_of_A_l881_88181


namespace fraction_start_with_9_end_with_0_is_1_over_72_l881_88163

-- Definition of valid 8-digit telephone number
def valid_phone_number (d : Fin 10) (n : Fin 10) (m : Fin (10 ^ 6)) : Prop :=
  2 ≤ d.val ∧ d.val ≤ 9 ∧ n.val ≤ 8

-- Definition of phone numbers that start with 9 and end with 0
def starts_with_9_ends_with_0 (d : Fin 10) (n : Fin 10) (m : Fin (10 ^ 6)) : Prop :=
  d.val = 9 ∧ n.val = 0

-- The total number of valid 8-digit phone numbers
noncomputable def total_valid_numbers : ℕ :=
  8 * (10 ^ 6) * 9

-- The number of valid phone numbers that start with 9 and end with 0
noncomputable def valid_start_with_9_end_with_0 : ℕ :=
  10 ^ 6

-- The target fraction
noncomputable def target_fraction : ℚ :=
  valid_start_with_9_end_with_0 / total_valid_numbers

-- Main theorem
theorem fraction_start_with_9_end_with_0_is_1_over_72 :
  target_fraction = (1 / 72 : ℚ) :=
by
  sorry

end fraction_start_with_9_end_with_0_is_1_over_72_l881_88163


namespace cyclists_no_point_b_l881_88126

theorem cyclists_no_point_b (v1 v2 t d : ℝ) (h1 : v1 = 35) (h2 : v2 = 25) (h3 : t = 2) (h4 : d = 30) :
  ∀ (ta tb : ℝ), ta + tb = t ∧ ta * v1 + tb * v2 < d → false :=
by
  sorry

end cyclists_no_point_b_l881_88126


namespace substitution_result_l881_88154

-- Conditions
def eq1 (x y : ℝ) : Prop := y = 2 * x - 3
def eq2 (x y : ℝ) : Prop := x - 2 * y = 6

-- The statement to be proven
theorem substitution_result (x y : ℝ) (h1 : eq1 x y) : (x - 4 * x + 6 = 6) :=
by sorry

end substitution_result_l881_88154


namespace average_length_one_third_of_strings_l881_88165

theorem average_length_one_third_of_strings (average_six_strings : ℕ → ℕ → ℕ)
    (average_four_strings : ℕ → ℕ → ℕ)
    (total_length : ℕ → ℕ → ℕ)
    (n m : ℕ) :
    (n = 6) →
    (m = 4) →
    (average_six_strings 80 n = 480) →
    (average_four_strings 85 m = 340) →
    (total_length 2 70 = 140) →
    70 = (480 - 340) / 2 :=
by
  intros h_n h_m avg_six avg_four total_len
  sorry

end average_length_one_third_of_strings_l881_88165


namespace sheet_width_l881_88189

theorem sheet_width (L : ℕ) (w : ℕ) (A_typist : ℚ) 
  (L_length : L = 30)
  (A_typist_percentage : A_typist = 0.64) 
  (width_used : ∀ w, w > 0 → (w - 4) * (24 : ℕ) = A_typist * w * 30) : 
  w = 20 :=
by
  intros
  sorry

end sheet_width_l881_88189


namespace university_diploma_percentage_l881_88121

-- Define variables
variables (P U J : ℝ)  -- P: Percentage of total population (i.e., 1 or 100%), U: Having a university diploma, J: having the job of their choice
variables (h1 : 10 / 100 * P = 10 / 100 * P * (1 - U) * J)        -- 10% of the people do not have a university diploma but have the job of their choice
variables (h2 : 30 / 100 * (P * (1 - J)) = 30 / 100 * P * U * (1 - J))  -- 30% of the people who do not have the job of their choice have a university diploma
variables (h3 : 40 / 100 * P = 40 / 100 * P * J)                   -- 40% of the people have the job of their choice

-- Statement to prove
theorem university_diploma_percentage : 
  48 / 100 * P = (30 / 100 * P * J) + (18 / 100 * P * (1 - J)) :=
by sorry

end university_diploma_percentage_l881_88121


namespace soccer_boys_percentage_l881_88184

theorem soccer_boys_percentage (total_students boys total_playing_soccer girls_not_playing_soccer : ℕ)
  (h_total_students : total_students = 500)
  (h_boys : boys = 350)
  (h_total_playing_soccer : total_playing_soccer = 250)
  (h_girls_not_playing_soccer : girls_not_playing_soccer = 115) :
  (boys - (total_students - total_playing_soccer) / total_playing_soccer * 100) = 86 :=
by
  sorry

end soccer_boys_percentage_l881_88184


namespace set_A_is_correct_l881_88155

open Complex

def A : Set ℤ := {x | ∃ n : ℕ, n > 0 ∧ x = (I ^ n + (-I) ^ n).re}

theorem set_A_is_correct : A = {-2, 0, 2} :=
sorry

end set_A_is_correct_l881_88155


namespace radian_to_degree_conversion_l881_88159

theorem radian_to_degree_conversion
: (π : ℝ) = 180 → ((-23 / 12) * π) = -345 :=
by
  sorry

end radian_to_degree_conversion_l881_88159


namespace marbles_cost_correct_l881_88135

def total_cost : ℝ := 20.52
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52

-- The problem is to prove that the amount spent on marbles is $9.05
def amount_spent_on_marbles : ℝ :=
  total_cost - (cost_football + cost_baseball)

theorem marbles_cost_correct :
  amount_spent_on_marbles = 9.05 :=
by
  -- The proof goes here.
  sorry

end marbles_cost_correct_l881_88135


namespace certain_number_is_3500_l881_88186

theorem certain_number_is_3500 :
  ∃ x : ℝ, x - (1000 / 20.50) = 3451.2195121951218 ∧ x = 3500 :=
by
  sorry

end certain_number_is_3500_l881_88186


namespace total_cost_is_160_l881_88139

-- Define the costs of each dress
def CostOfPaulineDress := 30
def CostOfJeansDress := CostOfPaulineDress - 10
def CostOfIdasDress := CostOfJeansDress + 30
def CostOfPattysDress := CostOfIdasDress + 10

-- The total cost
def TotalCost := CostOfPaulineDress + CostOfJeansDress + CostOfIdasDress + CostOfPattysDress

-- Prove the total cost is $160
theorem total_cost_is_160 : TotalCost = 160 := by
  -- skipping the proof steps
  sorry

end total_cost_is_160_l881_88139


namespace houses_with_white_mailboxes_l881_88140

theorem houses_with_white_mailboxes (total_mail : ℕ) (total_houses : ℕ) (red_mailboxes : ℕ) (mail_per_house : ℕ)
    (h1 : total_mail = 48) (h2 : total_houses = 8) (h3 : red_mailboxes = 3) (h4 : mail_per_house = 6) :
  total_houses - red_mailboxes = 5 :=
by
  sorry

end houses_with_white_mailboxes_l881_88140


namespace cloth_sold_l881_88178

theorem cloth_sold (total_sell_price : ℤ) (loss_per_meter : ℤ) (cost_price_per_meter : ℤ) (x : ℤ) 
    (h1 : total_sell_price = 18000) 
    (h2 : loss_per_meter = 5) 
    (h3 : cost_price_per_meter = 50) 
    (h4 : (cost_price_per_meter - loss_per_meter) * x = total_sell_price) : 
    x = 400 :=
by
  sorry

end cloth_sold_l881_88178


namespace determine_k_l881_88171

theorem determine_k (a b c k : ℤ) (h1 : c = -a - b) 
  (h2 : 60 < 6 * (8 * a + b) ∧ 6 * (8 * a + b) < 70)
  (h3 : 80 < 7 * (9 * a + b) ∧ 7 * (9 * a + b) < 90)
  (h4 : 2000 * k < (50^2 * a + 50 * b + c) ∧ (50^2 * a + 50 * b + c) < 2000 * (k + 1)) :
  k = 1 :=
  sorry

end determine_k_l881_88171


namespace Gage_skating_minutes_l881_88196

theorem Gage_skating_minutes (d1 d2 d3 : ℕ) (m1 m2 : ℕ) (avg : ℕ) (h1 : d1 = 6) (h2 : d2 = 4) (h3 : d3 = 1) (h4 : m1 = 80) (h5 : m2 = 105) (h6 : avg = 95) : 
  (d1 * m1 + d2 * m2 + d3 * x) / (d1 + d2 + d3) = avg ↔ x = 145 := 
by 
  sorry

end Gage_skating_minutes_l881_88196


namespace tickets_total_l881_88192

theorem tickets_total (x y : ℕ) 
  (h1 : 12 * x + 8 * y = 3320)
  (h2 : y = x + 190) : 
  x + y = 370 :=
by
  sorry

end tickets_total_l881_88192


namespace tangent_line_parallel_x_axis_coordinates_l881_88158

theorem tangent_line_parallel_x_axis_coordinates :
  (∃ P : ℝ × ℝ, P = (1, -2) ∨ P = (-1, 2)) ↔
  (∃ x y : ℝ, y = x^3 - 3 * x ∧ ∃ y', y' = 3 * x^2 - 3 ∧ y' = 0) :=
by
  sorry

end tangent_line_parallel_x_axis_coordinates_l881_88158


namespace Madison_minimum_score_l881_88117

theorem Madison_minimum_score (q1 q2 q3 q4 q5 : ℕ) (h1 : q1 = 84) (h2 : q2 = 81) (h3 : q3 = 87) (h4 : q4 = 83) (h5 : 85 * 5 ≤ q1 + q2 + q3 + q4 + q5) : 
  90 ≤ q5 := 
by
  sorry

end Madison_minimum_score_l881_88117


namespace alan_more_wings_per_minute_to_beat_record_l881_88191

-- Define relevant parameters and conditions
def kevin_wings := 64
def time_minutes := 8
def alan_rate := 5

-- Theorem: Alan must eat 3 more wings per minute to beat Kevin's record
theorem alan_more_wings_per_minute_to_beat_record : 
  (kevin_wings > alan_rate * time_minutes) → ((kevin_wings - (alan_rate * time_minutes)) / time_minutes = 3) :=
by
  sorry

end alan_more_wings_per_minute_to_beat_record_l881_88191


namespace max_intersection_value_l881_88120

noncomputable def max_intersection_size (A B C : Finset ℕ) (h1 : (A.card = 2019) ∧ (B.card = 2019)) 
  (h2 : (2 ^ A.card + 2 ^ B.card + 2 ^ C.card = 2 ^ (A ∪ B ∪ C).card)) : ℕ :=
  if ((A.card = 2019) ∧ (B.card = 2019) ∧ (A ∩ B ∩ C).card = 2018)
  then (A ∩ B ∩ C).card 
  else 0

theorem max_intersection_value (A B C : Finset ℕ) (h1 : (A.card = 2019) ∧ (B.card = 2019)) 
  (h2 : (2 ^ A.card + 2 ^ B.card + 2 ^ C.card = 2 ^ (A ∪ B ∪ C).card)) :
  max_intersection_size A B C h1 h2 = 2018 :=
sorry

end max_intersection_value_l881_88120


namespace Nara_height_is_1_69_l881_88150

-- Definitions of the conditions
def SangheonHeight : ℝ := 1.56
def ChihoHeight : ℝ := SangheonHeight - 0.14
def NaraHeight : ℝ := ChihoHeight + 0.27

-- The statement to prove
theorem Nara_height_is_1_69 : NaraHeight = 1.69 :=
by {
  sorry
}

end Nara_height_is_1_69_l881_88150
