import Mathlib

namespace ratio_seconds_l361_36147

theorem ratio_seconds (x : ℕ) (h : 12 / x = 6 / 240) : x = 480 :=
sorry

end ratio_seconds_l361_36147


namespace friend_wants_to_take_5_marbles_l361_36106

theorem friend_wants_to_take_5_marbles
  (total_marbles : ℝ)
  (clear_marbles : ℝ)
  (black_marbles : ℝ)
  (other_marbles : ℝ)
  (friend_marbles : ℝ)
  (h1 : clear_marbles = 0.4 * total_marbles)
  (h2 : black_marbles = 0.2 * total_marbles)
  (h3 : other_marbles = total_marbles - clear_marbles - black_marbles)
  (h4 : friend_marbles = 2)
  (friend_total_marbles : ℝ)
  (h5 : friend_marbles = 0.4 * friend_total_marbles) :
  friend_total_marbles = 5 := by
  sorry

end friend_wants_to_take_5_marbles_l361_36106


namespace class_contribution_Miss_Evans_class_contribution_Mr_Smith_class_contribution_Mrs_Johnson_l361_36145

theorem class_contribution_Miss_Evans :
  let total_contribution : ℝ := 90
  let class_funds_Evans : ℝ := 14
  let num_students_Evans : ℕ := 19
  let individual_contribution_Evans : ℝ := (total_contribution - class_funds_Evans) / num_students_Evans
  individual_contribution_Evans = 4 := 
sorry

theorem class_contribution_Mr_Smith :
  let total_contribution : ℝ := 90
  let class_funds_Smith : ℝ := 20
  let num_students_Smith : ℕ := 15
  let individual_contribution_Smith : ℝ := (total_contribution - class_funds_Smith) / num_students_Smith
  individual_contribution_Smith = 4.67 := 
sorry

theorem class_contribution_Mrs_Johnson :
  let total_contribution : ℝ := 90
  let class_funds_Johnson : ℝ := 30
  let num_students_Johnson : ℕ := 25
  let individual_contribution_Johnson : ℝ := (total_contribution - class_funds_Johnson) / num_students_Johnson
  individual_contribution_Johnson = 2.40 := 
sorry

end class_contribution_Miss_Evans_class_contribution_Mr_Smith_class_contribution_Mrs_Johnson_l361_36145


namespace function_is_odd_and_monotonically_increasing_on_pos_l361_36186

-- Define odd function
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

-- Define monotonically increasing on (0, +∞)
def monotonically_increasing_on_pos (f : ℝ → ℝ) := ∀ x y : ℝ, (0 < x ∧ x < y) → f (x) < f (y)

-- Define the function in question
def f (x : ℝ) := x * |x|

-- Prove the function is odd and monotonically increasing on (0, +∞)
theorem function_is_odd_and_monotonically_increasing_on_pos :
  odd_function f ∧ monotonically_increasing_on_pos f :=
by
  sorry

end function_is_odd_and_monotonically_increasing_on_pos_l361_36186


namespace least_number_to_add_1055_to_div_by_23_l361_36176

theorem least_number_to_add_1055_to_div_by_23 : ∃ k : ℕ, (1055 + k) % 23 = 0 ∧ k = 3 :=
by
  sorry

end least_number_to_add_1055_to_div_by_23_l361_36176


namespace value_of_b_minus_a_l361_36171

variable (a b : ℕ)

theorem value_of_b_minus_a 
  (h1 : b = 10)
  (h2 : a * b = 2 * (a + b) + 12) : b - a = 6 :=
by sorry

end value_of_b_minus_a_l361_36171


namespace smallest_root_abs_eq_six_l361_36131

theorem smallest_root_abs_eq_six : 
  (∃ x : ℝ, (abs (x - 1)) / (x^2) = 6 ∧ ∀ y : ℝ, (abs (y - 1)) / (y^2) = 6 → y ≥ x) → x = -1 / 2 := by
  sorry

end smallest_root_abs_eq_six_l361_36131


namespace distinct_divisor_sum_l361_36133

theorem distinct_divisor_sum (n : ℕ) (x : ℕ) (h : x < n.factorial) :
  ∃ (k : ℕ) (d : Fin k → ℕ), (k ≤ n) ∧ (∀ i j, i ≠ j → d i ≠ d j) ∧ (∀ i, d i ∣ n.factorial) ∧ (x = Finset.sum Finset.univ d) :=
sorry

end distinct_divisor_sum_l361_36133


namespace solve_exponential_equation_l361_36140

theorem solve_exponential_equation :
  ∃ x, (2:ℝ)^(2*x) - 8 * (2:ℝ)^x + 12 = 0 ↔ x = 1 ∨ x = 1 + Real.log 3 / Real.log 2 :=
by
  sorry

end solve_exponential_equation_l361_36140


namespace smallest_value_of_m_plus_n_l361_36165

theorem smallest_value_of_m_plus_n :
  ∃ m n : ℕ, 1 < m ∧ 
  (∃ l : ℝ, l = (m^2 - 1 : ℝ) / (m * n) ∧ l = 1 / 2021) ∧
  m + n = 85987 := 
sorry

end smallest_value_of_m_plus_n_l361_36165


namespace find_a_for_unique_solution_l361_36164

theorem find_a_for_unique_solution :
  ∃ a : ℝ, (∀ x : ℝ, 0 ≤ x^2 - a * x + a ∧ x^2 - a * x + a ≤ 1) ↔ a = 2 :=
by
  sorry

end find_a_for_unique_solution_l361_36164


namespace range_of_s_l361_36182

def double_value_point (s t : ℝ) (ht : t ≠ -1) :
  Prop := 
  ∀ k : ℝ, (t + 1) * k^2 + t * k + s = 0 →
  (t^2 - 4 * s * (t + 1) > 0)

theorem range_of_s (s t : ℝ) (ht : t ≠ -1) :
  double_value_point s t ht ↔ -1 < s ∧ s < 0 :=
sorry

end range_of_s_l361_36182


namespace Clarence_total_oranges_l361_36120

def Clarence_oranges_initial := 5
def oranges_from_Joyce := 3

theorem Clarence_total_oranges : Clarence_oranges_initial + oranges_from_Joyce = 8 := by
  sorry

end Clarence_total_oranges_l361_36120


namespace total_steps_l361_36146

theorem total_steps (up_steps down_steps : ℕ) (h1 : up_steps = 567) (h2 : down_steps = 325) : up_steps + down_steps = 892 := by
  sorry

end total_steps_l361_36146


namespace faye_initial_coloring_books_l361_36119

theorem faye_initial_coloring_books (gave_away1 gave_away2 remaining : ℝ) 
    (h1 : gave_away1 = 34.0) (h2 : gave_away2 = 3.0) (h3 : remaining = 11.0) :
    gave_away1 + gave_away2 + remaining = 48.0 := 
by
  sorry

end faye_initial_coloring_books_l361_36119


namespace sum_of_squares_l361_36162

theorem sum_of_squares (a b c d : ℝ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : d = a + 3) :
  a^2 + b^2 = c^2 + d^2 := by
  sorry

end sum_of_squares_l361_36162


namespace roots_sum_equality_l361_36141

theorem roots_sum_equality {a b c : ℝ} {x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ} :
  (∀ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x - 1 = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) →
  (∀ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x - 2 = 0 → x = y₁ ∨ x = y₂ ∨ x = y₃ ∨ x = y₄) →
  x₁ + x₂ = x₃ + x_₄ →
  y₁ + y₂ = y₃ + y₄ :=
sorry

end roots_sum_equality_l361_36141


namespace num_ways_to_use_100_yuan_l361_36127

noncomputable def x : ℕ → ℝ
| 0       => 0
| 1       => 1
| 2       => 3
| (n + 3) => x (n + 2) + 2 * x (n + 1)

theorem num_ways_to_use_100_yuan :
  x 100 = (1 / 3) * (2 ^ 101 + 1) :=
sorry

end num_ways_to_use_100_yuan_l361_36127


namespace container_capacity_in_liters_l361_36139

-- Defining the conditions
def portions : Nat := 10
def portion_size_ml : Nat := 200

-- Statement to prove
theorem container_capacity_in_liters : (portions * portion_size_ml / 1000 = 2) :=
by 
  sorry

end container_capacity_in_liters_l361_36139


namespace percent_increase_correct_l361_36122

noncomputable def last_year_ticket_price : ℝ := 85
noncomputable def last_year_tax_rate : ℝ := 0.10
noncomputable def this_year_ticket_price : ℝ := 102
noncomputable def this_year_tax_rate : ℝ := 0.12
noncomputable def student_discount_rate : ℝ := 0.15

noncomputable def last_year_total_cost : ℝ := last_year_ticket_price * (1 + last_year_tax_rate)
noncomputable def discounted_ticket_price_this_year : ℝ := this_year_ticket_price * (1 - student_discount_rate)
noncomputable def total_cost_this_year : ℝ := discounted_ticket_price_this_year * (1 + this_year_tax_rate)

noncomputable def percent_increase : ℝ := ((total_cost_this_year - last_year_total_cost) / last_year_total_cost) * 100

theorem percent_increase_correct :
  abs (percent_increase - 3.854) < 0.001 := sorry

end percent_increase_correct_l361_36122


namespace part1_part2_l361_36172

variable {U : Type} [TopologicalSpace U]

-- Definitions of the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2 * a}

-- Part (1): 
theorem part1 (U : Set ℝ) (a : ℝ) (h : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part (2):
theorem part2 (a : ℝ) (h : ¬ (A ∩ B a = B a)) : a < 1 / 2 := sorry

end part1_part2_l361_36172


namespace min_number_of_stamps_exists_l361_36168

theorem min_number_of_stamps_exists : 
  ∃ s t : ℕ, 5 * s + 7 * t = 50 ∧ ∀ (s' t' : ℕ), 5 * s' + 7 * t' = 50 → s + t ≤ s' + t' := 
by
  sorry

end min_number_of_stamps_exists_l361_36168


namespace geometric_sequence_a7_l361_36159

theorem geometric_sequence_a7 (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = 3)
  (h2 : a 2 + a 3 = 6)
  (h_geometric : ∀ n, a (n + 1) = q * a n) :
  a 7 = 64 := by
  sorry

end geometric_sequence_a7_l361_36159


namespace tan_alpha_plus_pi_over_4_l361_36160

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (Real.cos (2 * α), Real.sin α)
noncomputable def vec_b (α : ℝ) : ℝ × ℝ := (1, 2 * Real.sin α - 1)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi)
    (h3 : dot_product (vec_a α) (vec_b α) = 0) :
    Real.tan (α + Real.pi / 4) = -1 := sorry

end tan_alpha_plus_pi_over_4_l361_36160


namespace geometric_sequence_sum_l361_36121

variable {a : ℕ → ℝ} -- Sequence terms
variable {S : ℕ → ℝ} -- Sum of the first n terms

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, S n = a 0 * (1 - (a n)) / (1 - a 1)
def is_arithmetic_sequence (x y z : ℝ) := 2 * y = x + z
def term_1_equals_1 (a : ℕ → ℝ) := a 0 = 1

-- Question: Prove that given the conditions, S_5 = 31
theorem geometric_sequence_sum (q : ℝ) (h_geom : is_geometric_sequence a q) 
  (h_sum : sum_of_first_n_terms a S) (h_arith : is_arithmetic_sequence (4 * a 0) (2 * a 1) (a 2)) 
  (h_a1 : term_1_equals_1 a) : S 5 = 31 :=
sorry

end geometric_sequence_sum_l361_36121


namespace initial_speeds_l361_36130

/-- Motorcyclists Vasya and Petya ride at constant speeds around a circular track 1 km long.
    Petya overtakes Vasya every 2 minutes. Then Vasya doubles his speed and now he himself 
    overtakes Petya every 2 minutes. What were the initial speeds of Vasya and Petya? 
    Answer: 1000 and 1500 meters per minute.
-/

theorem initial_speeds (V_v V_p : ℕ) (track_length : ℕ) (time_interval : ℕ) 
  (h1 : track_length = 1000)
  (h2 : time_interval = 2)
  (h3 : V_p - V_v = track_length / time_interval)
  (h4 : 2 * V_v - V_p = track_length / time_interval):
  V_v = 1000 ∧ V_p = 1500 :=
by
  sorry

end initial_speeds_l361_36130


namespace triangle_base_l361_36113

theorem triangle_base (h : ℝ) (A : ℝ) (b : ℝ) (h_eq : h = 10) (A_eq : A = 46) (area_eq : A = (b * h) / 2) : b = 9.2 :=
by
  -- sorry to be replaced with the actual proof
  sorry

end triangle_base_l361_36113


namespace K1K2_eq_one_over_four_l361_36132

theorem K1K2_eq_one_over_four
  (K1 : ℝ) (hK1 : K1 ≠ 0)
  (K2 : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (hx1y1 : x1^2 - 4 * y1^2 = 4)
  (hx2y2 : x2^2 - 4 * y2^2 = 4)
  (hx0 : x0 = (x1 + x2) / 2)
  (hy0 : y0 = (y1 + y2) / 2)
  (K1_eq : K1 = (y1 - y2) / (x1 - x2))
  (K2_eq : K2 = y0 / x0) :
  K1 * K2 = 1 / 4 :=
sorry

end K1K2_eq_one_over_four_l361_36132


namespace simplify_fraction_l361_36185

theorem simplify_fraction : (48 / 72 : ℚ) = (2 / 3) := 
by
  sorry

end simplify_fraction_l361_36185


namespace ratio_population_XZ_l361_36100

variable (Population : Type) [Field Population]
variable (Z : Population) -- Population of City Z
variable (Y : Population) -- Population of City Y
variable (X : Population) -- Population of City X

-- Conditions
def population_Y : Y = 2 * Z := sorry
def population_X : X = 7 * Y := sorry

-- Theorem stating the ratio of populations
theorem ratio_population_XZ : (X / Z) = 14 := by
  -- The proof will use the conditions population_Y and population_X
  sorry

end ratio_population_XZ_l361_36100


namespace A_can_complete_work_in_4_days_l361_36116

-- Definitions based on conditions
def work_done_in_one_day (days : ℕ) : ℚ := 1 / days

def combined_work_done_in_two_days (a b c : ℕ) : ℚ :=
  work_done_in_one_day a + work_done_in_one_day b + work_done_in_one_day c

-- Theorem statement based on the problem
theorem A_can_complete_work_in_4_days (A B C : ℕ) 
  (hB : B = 8) (hC : C = 8) 
  (h_combined : combined_work_done_in_two_days A B C = work_done_in_one_day 2) :
  A = 4 :=
sorry

end A_can_complete_work_in_4_days_l361_36116


namespace central_angle_of_sector_l361_36188

theorem central_angle_of_sector (l S : ℝ) (r : ℝ) (θ : ℝ) 
  (h1 : l = 5) 
  (h2 : S = 5) 
  (h3 : S = (1 / 2) * l * r) 
  (h4 : l = θ * r): θ = 2.5 := by
  sorry

end central_angle_of_sector_l361_36188


namespace greg_savings_l361_36187

-- Definitions based on the conditions
def scooter_cost : ℕ := 90
def money_needed : ℕ := 33

-- The theorem to prove
theorem greg_savings : scooter_cost - money_needed = 57 := 
by
  -- sorry is used to skip the actual mathematical proof steps
  sorry

end greg_savings_l361_36187


namespace amy_small_gardens_l361_36193

-- Define the initial number of seeds
def initial_seeds : ℕ := 101

-- Define the number of seeds planted in the big garden
def big_garden_seeds : ℕ := 47

-- Define the number of seeds planted in each small garden
def seeds_per_small_garden : ℕ := 6

-- Define the number of small gardens
def number_of_small_gardens : ℕ := (initial_seeds - big_garden_seeds) / seeds_per_small_garden

-- Prove that Amy has 9 small gardens
theorem amy_small_gardens : number_of_small_gardens = 9 := by
  sorry

end amy_small_gardens_l361_36193


namespace log_mult_l361_36107

theorem log_mult : 
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 :=
by 
  sorry

end log_mult_l361_36107


namespace min_value_a_plus_b_l361_36134

theorem min_value_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : (2 / a) + (2 / b) = 1) :
  a + b >= 8 :=
sorry

end min_value_a_plus_b_l361_36134


namespace find_number_l361_36135

theorem find_number (x : ℤ) (h : 4 * x - 7 = 13) : x = 5 := 
sorry

end find_number_l361_36135


namespace arctan_sum_pi_div_two_l361_36151

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l361_36151


namespace sum_of_terms_in_arithmetic_sequence_eq_l361_36179

variable {a : ℕ → ℕ}

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_terms_in_arithmetic_sequence_eq :
  arithmetic_sequence a →
  (a 2 + a 3 + a 10 + a 11 = 36) →
  (a 3 + a 10 = 18) :=
by
  intros h_seq h_sum
  -- Proof placeholder
  sorry

end sum_of_terms_in_arithmetic_sequence_eq_l361_36179


namespace binary_division_remainder_l361_36155

theorem binary_division_remainder (n : ℕ) (h_n : n = 0b110110011011) : n % 8 = 3 :=
by {
  -- This sorry statement skips the actual proof
  sorry
}

end binary_division_remainder_l361_36155


namespace sheets_in_backpack_l361_36114

-- Definitions for the conditions
def total_sheets := 91
def desk_sheets := 50

-- Theorem statement with the goal
theorem sheets_in_backpack (total_sheets : ℕ) (desk_sheets : ℕ) (h1 : total_sheets = 91) (h2 : desk_sheets = 50) : 
  ∃ backpack_sheets : ℕ, backpack_sheets = total_sheets - desk_sheets ∧ backpack_sheets = 41 :=
by
  -- The proof is omitted here
  sorry

end sheets_in_backpack_l361_36114


namespace find_x_for_collinear_vectors_l361_36184

noncomputable def collinear_vectors (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem find_x_for_collinear_vectors : ∀ (x : ℝ), collinear_vectors (2, -3) (x, 6) → x = -4 := by
  intros x h
  sorry

end find_x_for_collinear_vectors_l361_36184


namespace machining_defect_probability_l361_36150

theorem machining_defect_probability :
  let defect_rate_process1 := 0.03
  let defect_rate_process2 := 0.05
  let non_defective_rate_process1 := 1 - defect_rate_process1
  let non_defective_rate_process2 := 1 - defect_rate_process2
  let non_defective_rate := non_defective_rate_process1 * non_defective_rate_process2
  let defective_rate := 1 - non_defective_rate
  defective_rate = 0.0785 :=
by
  sorry

end machining_defect_probability_l361_36150


namespace total_nails_polished_l361_36136

-- Defining the number of girls
def num_girls : ℕ := 5

-- Defining the number of fingers and toes per person
def num_fingers_per_person : ℕ := 10
def num_toes_per_person : ℕ := 10

-- Defining the total number of nails per person
def nails_per_person : ℕ := num_fingers_per_person + num_toes_per_person

-- The theorem stating that the total number of nails polished for 5 girls is 100 nails
theorem total_nails_polished : num_girls * nails_per_person = 100 := by
  sorry

end total_nails_polished_l361_36136


namespace f_f_of_2_l361_36148

def f (x : ℤ) : ℤ := 4 * x ^ 3 - 3 * x + 1

theorem f_f_of_2 : f (f 2) = 78652 := 
by
  sorry

end f_f_of_2_l361_36148


namespace barry_more_votes_than_joey_l361_36183

theorem barry_more_votes_than_joey {M B J X : ℕ} 
  (h1 : M = 66)
  (h2 : J = 8)
  (h3 : M = 3 * B)
  (h4 : B = 2 * (J + X)) :
  B - J = 14 := by
  sorry

end barry_more_votes_than_joey_l361_36183


namespace percentage_of_paycheck_went_to_taxes_l361_36126

-- Definitions
def original_paycheck : ℝ := 125
def savings : ℝ := 20
def spend_percentage : ℝ := 0.80
def save_percentage : ℝ := 0.20

-- Statement that needs to be proved
theorem percentage_of_paycheck_went_to_taxes (T : ℝ) :
  (0.20 * (1 - T / 100) * original_paycheck = savings) → T = 20 := 
by
  sorry

end percentage_of_paycheck_went_to_taxes_l361_36126


namespace zero_of_f_a_neg_sqrt2_range_of_a_no_zero_l361_36181

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 2 then 2^x + a else a - x

theorem zero_of_f_a_neg_sqrt2 : 
  ∀ x, f x (- Real.sqrt 2) = 0 ↔ x = 1/2 :=
by
  sorry

theorem range_of_a_no_zero :
  ∀ a, (¬∃ x, f x a = 0) ↔ a ∈ Set.Iic (-4) ∪ Set.Ico 0 2 :=
by
  sorry

end zero_of_f_a_neg_sqrt2_range_of_a_no_zero_l361_36181


namespace simplify_evaluate_expr_l361_36111

noncomputable def expr (x : ℝ) : ℝ := 
  ( ( (x^2 - 3) / (x + 2) - x + 2 ) / ( (x^2 - 4) / (x^2 + 4*x + 4) ) )

theorem simplify_evaluate_expr : 
  expr (Real.sqrt 2 + 1) = Real.sqrt 2 + 1 := by
  sorry

end simplify_evaluate_expr_l361_36111


namespace non_negative_solutions_l361_36104

theorem non_negative_solutions (x : ℕ) (h : 1 + x ≥ 2 * x - 1) : x = 0 ∨ x = 1 ∨ x = 2 := 
by {
  sorry
}

end non_negative_solutions_l361_36104


namespace kids_stayed_home_l361_36174

open Nat

theorem kids_stayed_home (kids_camp : ℕ) (additional_kids_home : ℕ) (total_kids_home : ℕ) 
  (h1 : kids_camp = 202958) 
  (h2 : additional_kids_home = 574664) 
  (h3 : total_kids_home = kids_camp + additional_kids_home) : 
  total_kids_home = 777622 := 
by 
  rw [h1, h2] at h3
  exact h3

end kids_stayed_home_l361_36174


namespace fisher_needed_score_l361_36195

-- Condition 1: To have an average of at least 85% over all four quarters
def average_score_threshold := 85
def total_score := 4 * average_score_threshold

-- Condition 2: Fisher's scores for the first three quarters
def first_three_scores := [82, 77, 75]
def current_total_score := first_three_scores.sum

-- Define the Lean statement to prove
theorem fisher_needed_score : ∃ x, current_total_score + x = total_score ∧ x = 106 := by
  sorry

end fisher_needed_score_l361_36195


namespace sequence_equality_l361_36123

theorem sequence_equality (a : ℕ → ℤ) (h : ∀ n, a (n + 2) ^ 2 + a (n + 1) * a n ≤ a (n + 2) * (a (n + 1) + a n)) :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 2) = a n :=
by sorry

end sequence_equality_l361_36123


namespace sugar_per_larger_cookie_l361_36191

theorem sugar_per_larger_cookie (c₁ c₂ : ℕ) (s₁ s₂ : ℝ) (h₁ : c₁ = 50) (h₂ : s₁ = 1 / 10) (h₃ : c₂ = 25) (h₄ : c₁ * s₁ = c₂ * s₂) : s₂ = 1 / 5 :=
by
  simp [h₁, h₂, h₃, h₄]
  sorry

end sugar_per_larger_cookie_l361_36191


namespace find_k_range_l361_36103

theorem find_k_range (k : ℝ) : 
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ -6 < k ∧ k < -2 :=
by
  sorry

end find_k_range_l361_36103


namespace right_triangle_side_length_l361_36109

theorem right_triangle_side_length (c a b : ℕ) (h1 : c = 5) (h2 : a = 3) (h3 : c^2 = a^2 + b^2) : b = 4 :=
  by
  sorry

end right_triangle_side_length_l361_36109


namespace polynomial_identity_l361_36170

theorem polynomial_identity (a0 a1 a2 a3 a4 a5 : ℤ) (x : ℤ) :
  (1 + 3 * x) ^ 5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 →
  a0 - a1 + a2 - a3 + a4 - a5 = -32 :=
by
  sorry

end polynomial_identity_l361_36170


namespace wine_problem_solution_l361_36110

theorem wine_problem_solution (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 200) (h2 : (200 - x) * (180 - x) / 200 = 144) : x = 20 := 
by
  sorry

end wine_problem_solution_l361_36110


namespace total_volume_of_water_l361_36144

-- Define the conditions
def volume_of_hemisphere : ℕ := 4
def number_of_hemispheres : ℕ := 2734

-- Define the total volume
def total_volume : ℕ := volume_of_hemisphere * number_of_hemispheres

-- State the theorem
theorem total_volume_of_water : total_volume = 10936 :=
by
  -- Proof placeholder
  sorry

end total_volume_of_water_l361_36144


namespace smallest_positive_real_x_l361_36108

theorem smallest_positive_real_x :
  ∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧ x = 89 / 9 :=
by
  sorry

end smallest_positive_real_x_l361_36108


namespace smallest_x_undefined_l361_36112

theorem smallest_x_undefined : ∃ x : ℝ, (10 * x^2 - 90 * x + 20 = 0) ∧ x = 1 / 4 :=
by sorry

end smallest_x_undefined_l361_36112


namespace interest_rate_l361_36128

theorem interest_rate (P CI SI: ℝ) (r: ℝ) : P = 5100 → CI = P * (1 + r)^2 - P → SI = P * r * 2 → (CI - SI = 51) → r = 0.1 :=
by
  intros
  -- skipping the proof
  sorry

end interest_rate_l361_36128


namespace range_j_l361_36158

def h (x : ℝ) : ℝ := 4 * x - 3
def j (x : ℝ) : ℝ := h (h (h x))

theorem range_j : ∀ x, 0 ≤ x ∧ x ≤ 3 → -63 ≤ j x ∧ j x ≤ 129 :=
by
  intro x
  intro hx
  sorry

end range_j_l361_36158


namespace tetrahedron_cube_volume_ratio_l361_36166

theorem tetrahedron_cube_volume_ratio (a : ℝ) :
  let V_tetrahedron := (a * Real.sqrt 2)^3 * Real.sqrt 2 / 12
  let V_cube := a^3
  (V_tetrahedron / V_cube) = 1 / 3 :=
by
  sorry

end tetrahedron_cube_volume_ratio_l361_36166


namespace number_of_lightsabers_in_order_l361_36156

-- Let's define the given conditions
def metal_arcs_per_lightsaber : ℕ := 2
def cost_per_metal_arc : ℕ := 400
def apparatus_production_rate : ℕ := 20 -- lightsabers per hour
def combined_app_expense_rate : ℕ := 300 -- units per hour
def total_order_cost : ℕ := 65200
def lightsaber_cost : ℕ := metal_arcs_per_lightsaber * cost_per_metal_arc + (combined_app_expense_rate / apparatus_production_rate)

-- Define the main theorem to prove
theorem number_of_lightsabers_in_order : 
  (total_order_cost / lightsaber_cost) = 80 :=
by
  sorry

end number_of_lightsabers_in_order_l361_36156


namespace steven_ships_boxes_l361_36102

-- Translate the conditions into Lean definitions and state the theorem
def truck_weight_limit : ℕ := 2000
def truck_count : ℕ := 3
def pair_weight : ℕ := 10 + 40
def boxes_per_pair : ℕ := 2

theorem steven_ships_boxes :
  ((truck_weight_limit / pair_weight) * boxes_per_pair * truck_count) = 240 := by
  sorry

end steven_ships_boxes_l361_36102


namespace problem_inequality_a3_a2_problem_inequality_relaxed_general_inequality_l361_36175

theorem problem_inequality_a3_a2 (a : ℝ) (ha : a > 1) : 
  a^3 + (1 / a^3) > a^2 + (1 / a^2) := 
sorry

theorem problem_inequality_relaxed (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : 
  a^3 + (1 / a^3) > a^2 + (1 / a^2) := 
sorry

theorem general_inequality (a : ℝ) (m n : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hmn1 : m > n) (hmn2 : n > 0) : 
  a^m + (1 / a^m) > a^n + (1 / a^n) := 
sorry

end problem_inequality_a3_a2_problem_inequality_relaxed_general_inequality_l361_36175


namespace max_sum_xy_l361_36129

theorem max_sum_xy (x y : ℤ) (h1 : x^2 + y^2 = 64) (h2 : x ≥ 0) (h3 : y ≥ 0) : x + y ≤ 8 :=
by sorry

end max_sum_xy_l361_36129


namespace relation_among_a_b_c_l361_36194

open Real

theorem relation_among_a_b_c 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : a = log 3 / log 2)
  (h2 : b = log 7 / (2 * log 2))
  (h3 : c = 0.7 ^ 4) :
  a > b ∧ b > c :=
by
  -- we leave the proof as an exercise
  sorry

end relation_among_a_b_c_l361_36194


namespace maximize_profit_l361_36192

variables (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ)

-- Definitions for the conditions
def nonneg_x := (0 ≤ x)
def nonneg_y := (0 ≤ y)
def constraint1 := (a1 * x + a2 * y ≤ c1)
def constraint2 := (b1 * x + b2 * y ≤ c2)
def profit := (z = d1 * x + d2 * y)

-- Proof of constraints and profit condition
theorem maximize_profit (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ) :
    nonneg_x x ∧ nonneg_y y ∧ constraint1 a1 a2 c1 x y ∧ constraint2 b1 b2 c2 x y → profit d1 d2 x y z :=
by
  sorry

end maximize_profit_l361_36192


namespace andrew_paid_1428_to_shopkeeper_l361_36169

-- Given conditions
def rate_per_kg_grapes : ℕ := 98
def quantity_of_grapes : ℕ := 11
def rate_per_kg_mangoes : ℕ := 50
def quantity_of_mangoes : ℕ := 7

-- Definitions for costs
def cost_of_grapes : ℕ := rate_per_kg_grapes * quantity_of_grapes
def cost_of_mangoes : ℕ := rate_per_kg_mangoes * quantity_of_mangoes
def total_amount_paid : ℕ := cost_of_grapes + cost_of_mangoes

-- Theorem to prove the total amount paid
theorem andrew_paid_1428_to_shopkeeper : total_amount_paid = 1428 := by
  sorry

end andrew_paid_1428_to_shopkeeper_l361_36169


namespace orange_ring_weight_l361_36154

theorem orange_ring_weight :
  ∀ (p w t o : ℝ), 
  p = 0.33 → w = 0.42 → t = 0.83 → t - (p + w) = o → 
  o = 0.08 :=
by
  intro p w t o hp hw ht h
  rw [hp, hw, ht] at h
  -- Additional steps would go here, but
  sorry -- Skipping the proof as instructed

end orange_ring_weight_l361_36154


namespace grass_coverage_day_l361_36180

theorem grass_coverage_day (coverage : ℕ → ℚ) : 
  (∀ n : ℕ, coverage (n + 1) = 2 * coverage n) → 
  coverage 24 = 1 → 
  coverage 21 = 1 / 8 := 
by
  sorry

end grass_coverage_day_l361_36180


namespace min_a_b_l361_36152

theorem min_a_b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 45 * a + b = 2021) : a + b = 85 :=
sorry

end min_a_b_l361_36152


namespace apples_left_over_l361_36142

-- Defining the number of apples collected by Liam, Mia, and Noah
def liam_apples := 53
def mia_apples := 68
def noah_apples := 22

-- The total number of apples collected
def total_apples := liam_apples + mia_apples + noah_apples

-- Proving that the remainder when the total number of apples is divided by 10 is 3
theorem apples_left_over : total_apples % 10 = 3 := by
  -- Placeholder for proof
  sorry

end apples_left_over_l361_36142


namespace solution_set_condition_l361_36189

theorem solution_set_condition (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - 1| > a) ↔ a < 1 :=
by
  sorry

end solution_set_condition_l361_36189


namespace total_votes_cast_l361_36190

def votes_witch : ℕ := 7
def votes_unicorn : ℕ := 3 * votes_witch
def votes_dragon : ℕ := votes_witch + 25
def votes_total : ℕ := votes_witch + votes_unicorn + votes_dragon

theorem total_votes_cast : votes_total = 60 := by
  sorry

end total_votes_cast_l361_36190


namespace find_b_l361_36157

-- Definitions
variable (k : ℤ) (b : ℤ)
def x := 3 * k
def y := 4 * k
def z := 7 * k

-- Conditions
axiom ratio : x / y = 3 / 4 ∧ y / z = 4 / 7
axiom equation : y = 15 * b - 5

-- Theorem statement
theorem find_b : ∃ b : ℤ, 4 * k = 15 * b - 5 ∧ b = 3 :=
by
  sorry

end find_b_l361_36157


namespace no_such_functions_exist_l361_36153

open Function

theorem no_such_functions_exist : ¬ (∃ (f g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^3) := 
sorry

end no_such_functions_exist_l361_36153


namespace unique_solution_l361_36197

theorem unique_solution (x y a : ℝ) :
  (x^2 + y^2 = 2 * a ∧ x + Real.log (y^2 + 1) / Real.log 2 = a) ↔ a = 0 ∧ x = 0 ∧ y = 0 :=
by
  sorry

end unique_solution_l361_36197


namespace marble_ratio_correct_l361_36178

-- Necessary given conditions
variables (x : ℕ) (Ben_initial John_initial : ℕ) (John_post Ben_post : ℕ)
variables (h1 : Ben_initial = 18)
variables (h2 : John_initial = 17)
variables (h3 : Ben_post = Ben_initial - x)
variables (h4 : John_post = John_initial + x)
variables (h5 : John_post = Ben_post + 17)

-- Define the ratio of the number of marbles Ben gave to John to the number of marbles Ben had initially
def marble_ratio := (x : ℕ) / Ben_initial

-- The theorem we want to prove
theorem marble_ratio_correct (h1 : Ben_initial = 18) (h2 : John_initial = 17) (h3 : Ben_post = Ben_initial - x)
(h4 : John_post = John_initial + x) (h5 : John_post = Ben_post + 17) : marble_ratio x Ben_initial = 1/2 := by 
  sorry

end marble_ratio_correct_l361_36178


namespace odds_against_horse_C_winning_l361_36115

theorem odds_against_horse_C_winning (odds_A : ℚ) (odds_B : ℚ) (odds_C : ℚ) 
  (cond1 : odds_A = 5 / 2) 
  (cond2 : odds_B = 3 / 1) 
  (race_condition : odds_C = 1 - ((2 / (5 + 2)) + (1 / (3 + 1))))
  : odds_C / (1 - odds_C) = 15 / 13 := 
sorry

end odds_against_horse_C_winning_l361_36115


namespace series_sum_eq_one_sixth_l361_36105

noncomputable def a (n : ℕ) : ℝ := 2^n / (7^(2^n) + 1)

theorem series_sum_eq_one_sixth :
  (∑' (n : ℕ), a n) = 1 / 6 :=
sorry

end series_sum_eq_one_sixth_l361_36105


namespace number_of_men_l361_36125

variable (W D X : ℝ)

theorem number_of_men (M_eq_2W : M = 2 * W)
  (wages_40_women : 21600 = 40 * W * D)
  (men_wages : 14400 = X * M * 20) :
  X = (2 / 3) * D :=
  by
  sorry

end number_of_men_l361_36125


namespace arithmetic_sequence_length_l361_36177

theorem arithmetic_sequence_length :
  ∀ (a₁ d an : ℤ), a₁ = -5 → d = 3 → an = 40 → (∃ n : ℕ, an = a₁ + (n - 1) * d ∧ n = 16) :=
by
  intros a₁ d an h₁ hd han
  sorry

end arithmetic_sequence_length_l361_36177


namespace max_ab_value_l361_36138

theorem max_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 4) : ab ≤ 2 :=
sorry

end max_ab_value_l361_36138


namespace number_of_chocolate_bars_l361_36198

theorem number_of_chocolate_bars (C : ℕ) (h1 : 50 * C = 250) : C = 5 := by
  sorry

end number_of_chocolate_bars_l361_36198


namespace election_votes_l361_36199

theorem election_votes (V : ℝ) (h1 : ∃ geoff_votes : ℝ, geoff_votes = 0.01 * V)
                       (h2 : ∀ candidate_votes : ℝ, (candidate_votes > 0.51 * V) → candidate_votes > 0.51 * V)
                       (h3 : ∃ needed_votes : ℝ, needed_votes = 3000 ∧ 0.01 * V + needed_votes = 0.51 * V) :
                       V = 6000 :=
by sorry

end election_votes_l361_36199


namespace abc_eq_bc_l361_36101

theorem abc_eq_bc (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) 
(h : 4 * a * b * c * (a + b + c) = (a + b)^2 * (a + c)^2) :
  a * (a + b + c) = b * c :=
by 
  sorry

end abc_eq_bc_l361_36101


namespace tissue_magnification_l361_36161

theorem tissue_magnification (d_image d_actual : ℝ) (h_image : d_image = 0.3) (h_actual : d_actual = 0.0003) :
  (d_image / d_actual) = 1000 :=
by
  sorry

end tissue_magnification_l361_36161


namespace Susie_possible_values_l361_36124

theorem Susie_possible_values (n : ℕ) (h1 : n > 43) (h2 : 2023 % n = 43) : 
  (∃ count : ℕ, count = 19 ∧ ∀ n, n > 43 ∧ 2023 % n = 43 → 1980 ∣ (2023 - 43)) :=
sorry

end Susie_possible_values_l361_36124


namespace janet_spending_difference_l361_36118

-- Defining hourly rates and weekly hours for each type of lessons
def clarinet_hourly_rate := 40
def clarinet_weekly_hours := 3
def piano_hourly_rate := 28
def piano_weekly_hours := 5
def violin_hourly_rate := 35
def violin_weekly_hours := 2
def singing_hourly_rate := 45
def singing_weekly_hours := 1

-- Calculating weekly costs
def clarinet_weekly_cost := clarinet_hourly_rate * clarinet_weekly_hours
def piano_weekly_cost := piano_hourly_rate * piano_weekly_hours
def violin_weekly_cost := violin_hourly_rate * violin_weekly_hours
def singing_weekly_cost := singing_hourly_rate * singing_weekly_hours
def combined_weekly_cost := piano_weekly_cost + violin_weekly_cost + singing_weekly_cost

-- Calculating annual costs with 52 weeks in a year
def weeks_per_year := 52
def clarinet_annual_cost := clarinet_weekly_cost * weeks_per_year
def combined_annual_cost := combined_weekly_cost * weeks_per_year

-- Proving the final statement
theorem janet_spending_difference :
  combined_annual_cost - clarinet_annual_cost = 7020 := by sorry

end janet_spending_difference_l361_36118


namespace compute_x_l361_36149

theorem compute_x :
  (∑' n : ℕ, (1 / (3^n)) * (1 / (3^n) * (-1)^n)) = (∑' n : ℕ, 1 / (9^n)) →
  (∑' n : ℕ, (1 / (3^n)) * (1 / (3^n) * (-1)^n)) = 1 / (1 - (1 / 9)) →
  9 = 9 :=
by
  sorry

end compute_x_l361_36149


namespace rational_function_solution_l361_36173

theorem rational_function_solution (g : ℝ → ℝ) (h : ∀ x ≠ 0, 4 * g (1 / x) + 3 * g x / x = x^3) :
  g (-3) = 135 / 4 := 
sorry

end rational_function_solution_l361_36173


namespace find_a_and_b_l361_36143

theorem find_a_and_b (a b : ℚ) (h : ∀ (n : ℕ), 1 / ((2 * n - 1) * (2 * n + 1)) = a / (2 * n - 1) + b / (2 * n + 1)) : 
  a = 1/2 ∧ b = -1/2 := 
by 
  sorry

end find_a_and_b_l361_36143


namespace expected_winnings_l361_36196

-- Define the probabilities
def prob_heads : ℚ := 1/2
def prob_tails : ℚ := 1/3
def prob_edge : ℚ := 1/6

-- Define the winnings
def win_heads : ℚ := 1
def win_tails : ℚ := 3
def lose_edge : ℚ := -5

-- Define the expected value function
def expected_value (p1 p2 p3 : ℚ) (w1 w2 w3 : ℚ) : ℚ :=
  p1 * w1 + p2 * w2 + p3 * w3

-- The expected winnings from flipping this coin
theorem expected_winnings : expected_value prob_heads prob_tails prob_edge win_heads win_tails lose_edge = 2/3 :=
by
  sorry

end expected_winnings_l361_36196


namespace min_removed_numbers_l361_36117

theorem min_removed_numbers : 
  ∃ S : Finset ℤ, 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 1982) ∧ 
    (∀ a b c : ℤ, a ∈ S → b ∈ S → c ∈ S → c ≠ a * b) ∧
    ∀ T : Finset ℤ, 
      ((∀ y ∈ T, 1 ≤ y ∧ y ≤ 1982) ∧ 
       (∀ p q r : ℤ, p ∈ T → q ∈ T → r ∈ T → r ≠ p * q) → 
       T.card ≥ 1982 - 43) :=
sorry

end min_removed_numbers_l361_36117


namespace right_triangle_shorter_leg_l361_36163
-- Import all necessary libraries

-- Define the problem
theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c = 65) (h4 : a^2 + b^2 = c^2) :
  a = 25 :=
sorry

end right_triangle_shorter_leg_l361_36163


namespace probability_is_7_over_26_l361_36167

section VowelProbability

def num_students : Nat := 26

def is_vowel (c : Char) : Bool :=
  c = 'A' || c = 'E' || c = 'I' || c = 'O' || c = 'U' || c = 'Y' || c = 'W'

def num_vowels : Nat := 7

def probability_of_vowel_initials : Rat :=
  (num_vowels : Nat) / (num_students : Nat)

theorem probability_is_7_over_26 :
  probability_of_vowel_initials = 7 / 26 := by
  sorry

end VowelProbability

end probability_is_7_over_26_l361_36167


namespace angle_F_measure_l361_36137

theorem angle_F_measure (α β γ : ℝ) (hD : α = 84) (hAngleSum : α + β + γ = 180) (hBeta : β = 4 * γ + 18) :
  γ = 15.6 := by
  sorry

end angle_F_measure_l361_36137
