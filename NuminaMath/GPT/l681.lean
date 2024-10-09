import Mathlib

namespace prove_necessary_but_not_sufficient_l681_68176

noncomputable def necessary_but_not_sufficient_condition (m : ℝ) :=
  (∀ x : ℝ, x^2 + 2*x + m > 0) → (m > 0) ∧ ¬ (∀ x : ℝ, x^2 + 2*x + m > 0 → m <= 1)

theorem prove_necessary_but_not_sufficient
    (m : ℝ) :
    necessary_but_not_sufficient_condition m :=
by
  sorry

end prove_necessary_but_not_sufficient_l681_68176


namespace subset_N_M_l681_68117

-- Define the sets M and N
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | x^2 - x < 0 }

-- The proof goal
theorem subset_N_M : N ⊆ M := by
  sorry

end subset_N_M_l681_68117


namespace find_original_class_strength_l681_68189

-- Definitions based on given conditions
def original_average_age : ℝ := 40
def additional_students : ℕ := 12
def new_students_average_age : ℝ := 32
def decrease_in_average : ℝ := 4
def new_average_age : ℝ := original_average_age - decrease_in_average

-- The equation setup
theorem find_original_class_strength (N : ℕ) (T : ℝ) 
  (h1 : T = original_average_age * N) 
  (h2 : T + additional_students * new_students_average_age = new_average_age * (N + additional_students)) : 
  N = 12 := 
sorry

end find_original_class_strength_l681_68189


namespace part1_part2_part3_l681_68133

noncomputable def functional_relationship (x : ℝ) : ℝ := -x + 26

theorem part1 (x y : ℝ) (hx6 : x = 6 ∧ y = 20) (hx8 : x = 8 ∧ y = 18) (hx10 : x = 10 ∧ y = 16) :
  ∀ (x : ℝ), functional_relationship x = -x + 26 := 
by
  sorry

theorem part2 (x : ℝ) (h_price_range : 6 ≤ x ∧ x ≤ 12) : 
  14 ≤ functional_relationship x ∧ functional_relationship x ≤ 20 :=
by
  sorry

noncomputable def gross_profit (x : ℝ) : ℝ := x * (functional_relationship x - 4)

theorem part3 (hx : 1 ≤ x) (hy : functional_relationship x ≤ 10):
  gross_profit (16 : ℝ) = 120 :=
by
  sorry

end part1_part2_part3_l681_68133


namespace selection_probability_correct_l681_68146

def percentage_women : ℝ := 0.55
def percentage_men : ℝ := 0.45

def women_below_35 : ℝ := 0.20
def women_35_to_50 : ℝ := 0.35
def women_above_50 : ℝ := 0.45

def men_below_35 : ℝ := 0.30
def men_35_to_50 : ℝ := 0.40
def men_above_50 : ℝ := 0.30

def women_below_35_lawyers : ℝ := 0.35
def women_below_35_doctors : ℝ := 0.45
def women_below_35_engineers : ℝ := 0.20

def women_35_to_50_lawyers : ℝ := 0.25
def women_35_to_50_doctors : ℝ := 0.50
def women_35_to_50_engineers : ℝ := 0.25

def women_above_50_lawyers : ℝ := 0.20
def women_above_50_doctors : ℝ := 0.30
def women_above_50_engineers : ℝ := 0.50

def men_below_35_lawyers : ℝ := 0.40
def men_below_35_doctors : ℝ := 0.30
def men_below_35_engineers : ℝ := 0.30

def men_35_to_50_lawyers : ℝ := 0.45
def men_35_to_50_doctors : ℝ := 0.25
def men_35_to_50_engineers : ℝ := 0.30

def men_above_50_lawyers : ℝ := 0.30
def men_above_50_doctors : ℝ := 0.40
def men_above_50_engineers : ℝ := 0.30

theorem selection_probability_correct :
  (percentage_women * women_below_35 * women_below_35_lawyers +
   percentage_men * men_above_50 * men_above_50_engineers +
   percentage_women * women_35_to_50 * women_35_to_50_doctors +
   percentage_men * men_35_to_50 * men_35_to_50_doctors) = 0.22025 :=
by
  sorry

end selection_probability_correct_l681_68146


namespace find_b2_l681_68198

theorem find_b2 (b : ℕ → ℝ) (h1 : b 1 = 23) (h10 : b 10 = 123) 
  (h : ∀ n ≥ 3, b n = (b 1 + b 2 + (n - 3) * b 3) / (n - 1)) : b 2 = 223 :=
sorry

end find_b2_l681_68198


namespace max_area_circle_center_l681_68148

theorem max_area_circle_center (k : ℝ) :
  (∃ (x y : ℝ), (x + k / 2)^2 + (y + 1)^2 = 1 - 3 / 4 * k^2 ∧ k = 0) →
  x = 0 ∧ y = -1 :=
sorry

end max_area_circle_center_l681_68148


namespace train_cross_time_l681_68132

-- Define the given conditions
def train_length : ℕ := 100
def train_speed_kmph : ℕ := 45
def total_length : ℕ := 275
def seconds_in_hour : ℕ := 3600
def meters_in_km : ℕ := 1000

-- Convert the speed from km/hr to m/s
noncomputable def train_speed_mps : ℚ := (train_speed_kmph * meters_in_km) / seconds_in_hour

-- The time to cross the bridge
noncomputable def time_to_cross (train_length total_length : ℕ) (train_speed_mps : ℚ) : ℚ :=
  total_length / train_speed_mps

-- The statement we want to prove
theorem train_cross_time : time_to_cross train_length total_length train_speed_mps = 30 :=
by
  sorry

end train_cross_time_l681_68132


namespace min_value_frac_l681_68100

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ (c : ℝ), (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → c ≤ 8 / x + 2 / y) ∧ c = 18 :=
sorry

end min_value_frac_l681_68100


namespace avg_temp_in_october_l681_68173

theorem avg_temp_in_october (a A : ℝ)
  (h1 : 28 = a + A)
  (h2 : 18 = a - A)
  (x := 10)
  (temperature : ℝ := a + A * Real.cos (π / 6 * (x - 6))) :
  temperature = 20.5 :=
by
  sorry

end avg_temp_in_october_l681_68173


namespace arithmetic_sequence_general_formula_l681_68172

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (h₁ : a 1 = 39) (h₂ : a 1 + a 3 = 74) : 
  ∀ n, a n = 41 - 2 * n :=
sorry

end arithmetic_sequence_general_formula_l681_68172


namespace find_divisor_l681_68111

-- Define the given conditions
def dividend : ℕ := 122
def quotient : ℕ := 6
def remainder : ℕ := 2

-- Define the proof problem to find the divisor
theorem find_divisor : 
  ∃ D : ℕ, dividend = (D * quotient) + remainder ∧ D = 20 :=
by sorry

end find_divisor_l681_68111


namespace joann_lollipops_l681_68160

theorem joann_lollipops : 
  ∃ (a : ℚ), 
  (7 * a  + 3 * (1 + 2 + 3 + 4 + 5 + 6) = 150) ∧ 
  (a_4 = a + 9) ∧ 
  (a_4 = 150 / 7) :=
by
  sorry

end joann_lollipops_l681_68160


namespace remainder_of_sum_of_primes_l681_68145

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l681_68145


namespace greatest_integer_x_l681_68181

theorem greatest_integer_x
    (x : ℤ) : 
    (7 / 9 : ℚ) > (x : ℚ) / 13 → x ≤ 10 :=
by
    sorry

end greatest_integer_x_l681_68181


namespace solve_f_g_f_3_l681_68110

def f (x : ℤ) : ℤ := 2 * x + 4

def g (x : ℤ) : ℤ := 5 * x + 2

theorem solve_f_g_f_3 :
  f (g (f 3)) = 108 := by
  sorry

end solve_f_g_f_3_l681_68110


namespace mikes_remaining_cards_l681_68163

variable (original_number_of_cards : ℕ)
variable (sam_bought : ℤ)
variable (alex_bought : ℤ)

theorem mikes_remaining_cards :
  original_number_of_cards = 87 →
  sam_bought = 8 →
  alex_bought = 13 →
  original_number_of_cards - (sam_bought + alex_bought) = 66 :=
by
  intros h_original h_sam h_alex
  rw [h_original, h_sam, h_alex]
  norm_num

end mikes_remaining_cards_l681_68163


namespace third_term_arithmetic_sequence_l681_68125

theorem third_term_arithmetic_sequence (a x : ℝ) 
  (h : 2 * a + 2 * x = 10) : a + x = 5 := 
by
  sorry

end third_term_arithmetic_sequence_l681_68125


namespace circle_equation_m_l681_68199
open Real

theorem circle_equation_m (m : ℝ) : (x^2 + y^2 + 4 * x + 2 * y + m = 0 → m < 5) := sorry

end circle_equation_m_l681_68199


namespace pie_distribution_l681_68193

theorem pie_distribution (x y : ℕ) (h1 : x + y + 2 * x = 13) (h2 : x < y) (h3 : y < 2 * x) :
  x = 3 ∧ y = 4 ∧ 2 * x = 6 := by
  sorry

end pie_distribution_l681_68193


namespace trapezoid_geometry_proof_l681_68171

theorem trapezoid_geometry_proof
  (midline_length : ℝ)
  (segment_midpoints : ℝ)
  (angle1 angle2 : ℝ)
  (h_midline : midline_length = 5)
  (h_segment_midpoints : segment_midpoints = 3)
  (h_angle1 : angle1 = 30)
  (h_angle2 : angle2 = 60) :
  ∃ (AD BC AB : ℝ), AD = 8 ∧ BC = 2 ∧ AB = 3 :=
by
  sorry

end trapezoid_geometry_proof_l681_68171


namespace train_pass_jogger_in_36_sec_l681_68120

noncomputable def time_to_pass_jogger (speed_jogger speed_train : ℝ) (lead_jogger len_train : ℝ) : ℝ :=
  let speed_jogger_mps := speed_jogger * (1000 / 3600)
  let speed_train_mps := speed_train * (1000 / 3600)
  let relative_speed := speed_train_mps - speed_jogger_mps
  let total_distance := lead_jogger + len_train
  total_distance / relative_speed

theorem train_pass_jogger_in_36_sec :
  time_to_pass_jogger 9 45 240 120 = 36 := by
  sorry

end train_pass_jogger_in_36_sec_l681_68120


namespace can_combine_with_sqrt2_l681_68118

theorem can_combine_with_sqrt2 :
  (∃ (x : ℝ), x = 2 * Real.sqrt 6 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 2 * Real.sqrt 3 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 2 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 3 * Real.sqrt 2 ∧ ∃ (y : ℝ), y = Real.sqrt 2) :=
sorry

end can_combine_with_sqrt2_l681_68118


namespace min_value_expression_l681_68105

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 5) :
  ∃ (min_val : ℝ), min_val = ( (x + 1) * (2 * y + 1) ) / (Real.sqrt (x * y)) ∧ min_val = 4 * Real.sqrt 3 :=
by
  sorry

end min_value_expression_l681_68105


namespace union_sets_l681_68119

-- Define the sets A and B based on their conditions
def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 < x ∧ x < 9 }

-- Statement of the proof problem
theorem union_sets (x : ℝ) : (x ∈ A ∪ B) ↔ (x ∈ { x | -1 ≤ x ∧ x < 9 }) := sorry

end union_sets_l681_68119


namespace num_photos_to_include_l681_68155

-- Define the conditions
def num_preselected_photos : ℕ := 7
def total_choices : ℕ := 56

-- Define the statement to prove
theorem num_photos_to_include : total_choices / num_preselected_photos = 8 :=
by sorry

end num_photos_to_include_l681_68155


namespace point_inside_circle_l681_68180

theorem point_inside_circle (O A : Type) (r OA : ℝ) (h1 : r = 6) (h2 : OA = 5) :
  OA < r :=
by
  sorry

end point_inside_circle_l681_68180


namespace find_f_expression_l681_68188

theorem find_f_expression (f : ℝ → ℝ) (x : ℝ) (h : f (Real.log x) = 3 * x + 4) : 
  f x = 3 * Real.exp x + 4 := 
by
  sorry

end find_f_expression_l681_68188


namespace louie_pie_share_l681_68186

theorem louie_pie_share :
  let leftover := (6 : ℝ) / 7
  let people := 3
  leftover / people = (2 : ℝ) / 7 := 
by
  sorry

end louie_pie_share_l681_68186


namespace rational_product_nonpositive_l681_68144

open Classical

theorem rational_product_nonpositive (a b : ℚ) (ha : |a| = a) (hb : |b| ≠ b) : a * b ≤ 0 :=
by
  sorry

end rational_product_nonpositive_l681_68144


namespace village_population_equal_in_years_l681_68141

theorem village_population_equal_in_years :
  ∀ (n : ℕ), (70000 - 1200 * n = 42000 + 800 * n) ↔ n = 14 :=
by {
  sorry
}

end village_population_equal_in_years_l681_68141


namespace restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l681_68124

-- Defining the given conditions
noncomputable def market_demand (P : ℝ) : ℝ := 688 - 4 * P
noncomputable def post_tax_producer_price : ℝ := 64
noncomputable def per_unit_tax : ℝ := 90
noncomputable def elasticity_supply_no_tax (P_e : ℝ) (Q_e : ℝ) : ℝ :=
  1.5 * (-(4 * P_e / Q_e))

-- Supply function to be proven
noncomputable def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Total tax revenue to be proven
noncomputable def total_tax_revenue : ℝ := 6480

-- Optimal tax rate to be proven
noncomputable def optimal_tax_rate : ℝ := 60

-- Maximum tax revenue to be proven
noncomputable def maximum_tax_revenue : ℝ := 8640

-- Theorem statements that need to be proven
theorem restore_supply_function (P : ℝ) : 
  supply_function P = 6 * P - 312 := sorry

theorem determine_tax_revenue : 
  total_tax_revenue = 6480 := sorry

theorem determine_optimal_tax_rate : 
  optimal_tax_rate = 60 := sorry

theorem determine_maximum_tax_revenue : 
  maximum_tax_revenue = 8640 := sorry

end restore_supply_function_determine_tax_revenue_determine_optimal_tax_rate_determine_maximum_tax_revenue_l681_68124


namespace eight_x_plus_y_l681_68135

theorem eight_x_plus_y (x y z : ℝ) (h1 : x + 2 * y - 3 * z = 7) (h2 : 2 * x - y + 2 * z = 6) : 
  8 * x + y = 32 :=
sorry

end eight_x_plus_y_l681_68135


namespace work_completion_days_l681_68149

theorem work_completion_days (x : ℕ) 
  (h1 : (1 : ℚ) / x + 1 / 9 = 1 / 6) :
  x = 18 := 
sorry

end work_completion_days_l681_68149


namespace find_n_l681_68164

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 103) (h3 : 100 * n ≡ 85 [MOD 103]) : n = 6 := 
sorry

end find_n_l681_68164


namespace john_more_needed_l681_68136

def john_needs : ℝ := 2.5
def john_has : ℝ := 0.75
def john_needs_more : ℝ := 1.75

theorem john_more_needed : (john_needs - john_has) = john_needs_more :=
by
  sorry

end john_more_needed_l681_68136


namespace other_divisor_l681_68121

theorem other_divisor (x : ℕ) (h₁ : 261 % 7 = 2) (h₂ : 261 % x = 2) : x = 259 :=
sorry

end other_divisor_l681_68121


namespace value_of_y_l681_68170

theorem value_of_y (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 24) : y = 96 :=
by
  sorry

end value_of_y_l681_68170


namespace max_value_of_m_l681_68190

theorem max_value_of_m (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (2 / a) + (1 / b) = 1 / 4) : 2 * a + b ≥ 36 :=
by 
  -- Skipping the proof
  sorry

end max_value_of_m_l681_68190


namespace smallest_number_is_10_l681_68191

/-- Define the set of numbers. -/
def numbers : List Int := [10, 11, 12, 13, 14]

theorem smallest_number_is_10 :
  ∃ n ∈ numbers, (∀ m ∈ numbers, n ≤ m) ∧ n = 10 :=
by
  sorry

end smallest_number_is_10_l681_68191


namespace find_a_l681_68195

-- Given function and its condition
def f (a x : ℝ) := a * x ^ 3 + 3 * x ^ 2 + 2
def f' (a x : ℝ) := 3 * a * x ^ 2 + 6 * x

-- Condition and proof that a = -2 given the condition f'(-1) = -12
theorem find_a 
  (a : ℝ)
  (h : f' a (-1) = -12) : 
  a = -2 := 
by 
  sorry

end find_a_l681_68195


namespace limit_of_an_l681_68112

theorem limit_of_an (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n, a_n n = (4 * n - 3) / (2 * n + 1)) → 
  a = 2 → 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) :=
by
  intros ha hA ε hε
  sorry

end limit_of_an_l681_68112


namespace minimum_value_of_objective_function_l681_68102

theorem minimum_value_of_objective_function :
  ∃ (x y : ℝ), x - y + 2 ≥ 0 ∧ 2 * x + 3 * y - 6 ≥ 0 ∧ 3 * x + 2 * y - 9 ≤ 0 ∧ (∀ (x' y' : ℝ), x' - y' + 2 ≥ 0 ∧ 2 * x' + 3 * y' - 6 ≥ 0 ∧ 3 * x' + 2 * y' - 9 ≤ 0 → 2 * x + 5 * y ≤ 2 * x' + 5 * y') ∧ 2 * x + 5 * y = 6 :=
sorry

end minimum_value_of_objective_function_l681_68102


namespace find_angle_C_max_area_l681_68104

-- Define the conditions as hypotheses
variable (a b c : ℝ) (A B C : ℝ)
variable (h1 : c = 2 * Real.sqrt 3)
variable (h2 : c * Real.cos B + (b - 2 * a) * Real.cos C = 0)

-- Problem (1): Prove that angle C is π/3
theorem find_angle_C : C = Real.pi / 3 :=
by
  sorry

-- Problem (2): Prove that the maximum area of triangle ABC is 3√3
theorem max_area : (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 :=
by
  sorry

end find_angle_C_max_area_l681_68104


namespace stream_speed_l681_68157

theorem stream_speed (u v : ℝ) (h1 : 27 = 9 * (u - v)) (h2 : 81 = 9 * (u + v)) : v = 3 :=
by
  sorry

end stream_speed_l681_68157


namespace length_of_second_train_correct_l681_68140

noncomputable def length_of_second_train : ℝ :=
  let speed_first_train := 60 / 3.6
  let speed_second_train := 90 / 3.6
  let relative_speed := speed_first_train + speed_second_train
  let time_to_clear := 6.623470122390208
  let total_distance := relative_speed * time_to_clear
  let length_first_train := 111
  total_distance - length_first_train

theorem length_of_second_train_correct :
  length_of_second_train = 164.978 :=
by
  unfold length_of_second_train
  sorry

end length_of_second_train_correct_l681_68140


namespace minimum_bamboo_fencing_length_l681_68174

theorem minimum_bamboo_fencing_length 
  (a b z : ℝ) 
  (h1 : a * b = 50)
  (h2 : a + 2 * b = z) : 
  z ≥ 20 := 
  sorry

end minimum_bamboo_fencing_length_l681_68174


namespace inequality_solution_set_l681_68185

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else -1

theorem inequality_solution_set :
  { x : ℝ | (x+1) * f x > 2 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end inequality_solution_set_l681_68185


namespace child_ticket_cost_l681_68152

-- Define the conditions
def adult_ticket_cost : ℕ := 11
def total_people : ℕ := 23
def total_revenue : ℕ := 246
def children_count : ℕ := 7
def adults_count := total_people - children_count

-- Define the target to prove that the child ticket cost is 10
theorem child_ticket_cost (child_ticket_cost : ℕ) :
  16 * adult_ticket_cost + 7 * child_ticket_cost = total_revenue → 
  child_ticket_cost = 10 := by
  -- The proof is omitted
  sorry

end child_ticket_cost_l681_68152


namespace abs_abc_eq_one_l681_68159

theorem abs_abc_eq_one 
  (a b c : ℝ)
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0)
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hca : c ≠ a)
  (h_eq : a + 1/b^2 = b + 1/c^2 ∧ b + 1/c^2 = c + 1/a^2) : 
  |a * b * c| = 1 := 
sorry

end abs_abc_eq_one_l681_68159


namespace coefficient_of_neg2ab_is_neg2_l681_68126

-- Define the term -2ab
def term : ℤ := -2

-- Define the function to get the coefficient from term -2ab
def coefficient (t : ℤ) : ℤ := t

-- The theorem stating the coefficient of -2ab is -2
theorem coefficient_of_neg2ab_is_neg2 : coefficient term = -2 :=
by
  -- Proof can be filled later
  sorry

end coefficient_of_neg2ab_is_neg2_l681_68126


namespace simplify_expression_l681_68113

-- Define general term for y
variable (y : ℤ)

-- Statement representing the given proof problem
theorem simplify_expression :
  4 * y + 5 * y + 6 * y + 2 = 15 * y + 2 := 
sorry

end simplify_expression_l681_68113


namespace difference_of_squares_l681_68197

theorem difference_of_squares (a b : ℝ) : -4 * a^2 + b^2 = (b + 2 * a) * (b - 2 * a) :=
by
  sorry

end difference_of_squares_l681_68197


namespace volume_relation_l681_68129

noncomputable def A (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3
noncomputable def M (r : ℝ) : ℝ := 2 * Real.pi * r^3
noncomputable def C (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_relation (r : ℝ) : A r - M r + C r = 0 :=
by
  sorry

end volume_relation_l681_68129


namespace expenditures_ratio_l681_68192

open Real

variables (I1 I2 E1 E2 : ℝ)
variables (x : ℝ)

theorem expenditures_ratio 
  (h1 : I1 = 4500)
  (h2 : I1 / I2 = 5 / 4)
  (h3 : I1 - E1 = 1800)
  (h4 : I2 - E2 = 1800) : 
  E1 / E2 = 3 / 2 :=
by
  have h5 : I1 / 5 = x := by sorry
  have h6 : I2 = 4 * x := by sorry
  have h7 : I2 = 3600 := by sorry
  have h8 : E1 = 2700 := by sorry
  have h9 : E2 = 1800 := by sorry
  exact sorry 

end expenditures_ratio_l681_68192


namespace time_via_route_B_l681_68123

-- Given conditions
def time_via_route_A : ℕ := 5
def time_saved_round_trip : ℕ := 6

-- Defining the proof problem
theorem time_via_route_B : time_via_route_A - (time_saved_round_trip / 2) = 2 :=
by
  -- Expected proof here
  sorry

end time_via_route_B_l681_68123


namespace return_trip_amount_l681_68167

noncomputable def gasoline_expense : ℝ := 8
noncomputable def lunch_expense : ℝ := 15.65
noncomputable def gift_expense_per_person : ℝ := 5
noncomputable def grandma_gift_per_person : ℝ := 10
noncomputable def initial_amount : ℝ := 50

theorem return_trip_amount : 
  let total_expense := gasoline_expense + lunch_expense + (gift_expense_per_person * 2)
  let total_money_gifted := grandma_gift_per_person * 2
  initial_amount - total_expense + total_money_gifted = 36.35 :=
by 
  let total_expense := gasoline_expense + lunch_expense + (gift_expense_per_person * 2)
  let total_money_gifted := grandma_gift_per_person * 2
  sorry

end return_trip_amount_l681_68167


namespace monotonic_intervals_of_f_g_minus_f_less_than_3_l681_68116

noncomputable def f (x : ℝ) : ℝ := -x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - x

theorem monotonic_intervals_of_f :
  ∀ x : ℝ, x < -1 / Real.exp 1 → f x < f (-1 / Real.exp 1) ∧ x > -1 / Real.exp 1 → f x > f (-1 / Real.exp 1) := sorry

theorem g_minus_f_less_than_3 :
  ∀ x : ℝ, x < 0 → g x - f x < 3 := sorry

end monotonic_intervals_of_f_g_minus_f_less_than_3_l681_68116


namespace main_theorem_l681_68108

-- Declare nonzero complex numbers
variables {x y z : ℂ} 

-- State the conditions
def conditions (x y z : ℂ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  x + y + z = 30 ∧
  (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z

-- Prove the main statement given the conditions
theorem main_theorem (h : conditions x y z) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 33 :=
by
  sorry

end main_theorem_l681_68108


namespace relationship_y1_y2_l681_68128

theorem relationship_y1_y2 (x1 x2 y1 y2 : ℝ) 
  (h1: x1 > 0) 
  (h2: 0 > x2) 
  (h3: y1 = 2 / x1)
  (h4: y2 = 2 / x2) : 
  y1 > y2 :=
by
  sorry

end relationship_y1_y2_l681_68128


namespace percentage_girls_l681_68165

theorem percentage_girls (initial_boys : ℕ) (initial_girls : ℕ) (added_boys : ℕ) :
  initial_boys = 11 → initial_girls = 13 → added_boys = 1 → 
  100 * initial_girls / (initial_boys + added_boys + initial_girls) = 52 :=
by
  intros h_boys h_girls h_added
  sorry

end percentage_girls_l681_68165


namespace balloon_volume_safety_l681_68194

theorem balloon_volume_safety (p V : ℝ) (h_prop : p = 90 / V) (h_burst : p ≤ 150) : 0.6 ≤ V :=
by {
  sorry
}

end balloon_volume_safety_l681_68194


namespace smallest_irreducible_l681_68107

def is_irreducible (n : ℕ) : Prop :=
  ∀ k : ℕ, 19 ≤ k ∧ k ≤ 91 → Nat.gcd k (n + k + 2) = 1

theorem smallest_irreducible : ∃ n : ℕ, is_irreducible n ∧ ∀ m : ℕ, m < n → ¬ is_irreducible m :=
  by
  exists 95
  sorry

end smallest_irreducible_l681_68107


namespace total_students_in_school_l681_68151

variable (TotalStudents : ℕ)
variable (num_students_8_years_old : ℕ := 48)
variable (percent_students_below_8 : ℝ := 0.20)
variable (num_students_above_8 : ℕ := (2 / 3) * num_students_8_years_old)

theorem total_students_in_school :
  percent_students_below_8 * TotalStudents + (num_students_8_years_old + num_students_above_8) = TotalStudents :=
by
  sorry

end total_students_in_school_l681_68151


namespace sixth_day_is_wednesday_l681_68154

noncomputable def day_of_week : Type := 
  { d // d ∈ ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"] }

def five_fridays_sum_correct (x : ℤ) : Prop :=
  x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 75

def first_is_friday (x : ℤ) : Prop :=
  x = 1

def day_of_6th_is_wednesday (d : day_of_week) : Prop :=
  d.1 = "Wednesday"

theorem sixth_day_is_wednesday (x : ℤ) (d : day_of_week) :
  five_fridays_sum_correct x → first_is_friday x → day_of_6th_is_wednesday d :=
by
  sorry

end sixth_day_is_wednesday_l681_68154


namespace Roselyn_initial_books_correct_l681_68115

variables (Roselyn_initial_books Mara_books Rebecca_books : ℕ)

-- Conditions
axiom A1 : Rebecca_books = 40
axiom A2 : Mara_books = 3 * Rebecca_books
axiom A3 : Roselyn_initial_books - (Rebecca_books + Mara_books) = 60

-- Proof statement
theorem Roselyn_initial_books_correct : Roselyn_initial_books = 220 :=
sorry

end Roselyn_initial_books_correct_l681_68115


namespace distribute_diamonds_among_two_safes_l681_68158

theorem distribute_diamonds_among_two_safes (N : ℕ) :
  ∀ banker : ℕ, banker < 777 → ∃ s1 s2 : ℕ, s1 ≠ s2 ∧ s1 + s2 = N := sorry

end distribute_diamonds_among_two_safes_l681_68158


namespace net_price_change_l681_68179

theorem net_price_change (P : ℝ) : 
  let decreased_price := P * (1 - 0.30)
  let increased_price := decreased_price * (1 + 0.20)
  increased_price - P = -0.16 * P :=
by
  -- The proof would go here. We just need the statement as per the prompt.
  sorry

end net_price_change_l681_68179


namespace dvd_blu_ratio_l681_68130

theorem dvd_blu_ratio (D B : ℕ) (h1 : D + B = 378) (h2 : (D : ℚ) / (B - 4 : ℚ) = 9 / 2) :
  D / Nat.gcd D B = 51 ∧ B / Nat.gcd D B = 12 :=
by
  sorry

end dvd_blu_ratio_l681_68130


namespace num_adult_tickets_l681_68178

theorem num_adult_tickets (adult_ticket_cost child_ticket_cost total_tickets_sold total_receipts : ℕ) 
  (h1 : adult_ticket_cost = 12) 
  (h2 : child_ticket_cost = 4) 
  (h3 : total_tickets_sold = 130) 
  (h4 : total_receipts = 840) :
  ∃ A C : ℕ, A + C = total_tickets_sold ∧ adult_ticket_cost * A + child_ticket_cost * C = total_receipts ∧ A = 40 :=
by {
  sorry
}

end num_adult_tickets_l681_68178


namespace eval_expression_l681_68196

-- We define the expression that needs to be evaluated
def expression := (0.76)^3 - (0.1)^3 / (0.76)^2 + 0.076 + (0.1)^2

-- The statement to prove
theorem eval_expression : expression = 0.5232443982683983 :=
by
  sorry

end eval_expression_l681_68196


namespace embankment_construction_l681_68183

theorem embankment_construction :
  (∃ r : ℚ, 0 < r ∧ (1 / 2 = 60 * r * 3)) →
  (∃ t : ℕ, 1 = 45 * 1 / 360 * t) :=
by
  sorry

end embankment_construction_l681_68183


namespace side_length_of_square_l681_68137

-- Define the conditions
def area_rectangle (length width : ℝ) : ℝ := length * width
def area_square (side : ℝ) : ℝ := side * side

-- Given conditions
def rect_length : ℝ := 2
def rect_width : ℝ := 8
def area_of_rectangle : ℝ := area_rectangle rect_length rect_width
def area_of_square : ℝ := area_of_rectangle

-- Main statement to prove
theorem side_length_of_square : ∃ (s : ℝ), s^2 = 16 ∧ s = 4 :=
by {
  -- use the conditions here
  sorry
}

end side_length_of_square_l681_68137


namespace applicant_overall_score_l681_68106

-- Definitions for the conditions
def writtenTestScore : ℝ := 80
def interviewScore : ℝ := 60
def weightWrittenTest : ℝ := 0.6
def weightInterview : ℝ := 0.4

-- Theorem statement
theorem applicant_overall_score : 
  (writtenTestScore * weightWrittenTest) + (interviewScore * weightInterview) = 72 := 
by
  sorry

end applicant_overall_score_l681_68106


namespace filled_sacks_count_l681_68114

-- Definitions from the problem conditions
def pieces_per_sack := 20
def total_pieces := 80

theorem filled_sacks_count : total_pieces / pieces_per_sack = 4 := 
by sorry

end filled_sacks_count_l681_68114


namespace gridiron_football_club_members_count_l681_68127

theorem gridiron_football_club_members_count :
  let sock_price := 6
  let tshirt_price := sock_price + 7
  let helmet_price := 2 * tshirt_price
  let total_cost_per_member := sock_price + tshirt_price + helmet_price
  let total_expenditure := 4680
  total_expenditure / total_cost_per_member = 104 :=
by
  let sock_price := 6
  let tshirt_price := sock_price + 7
  let helmet_price := 2 * tshirt_price
  let total_cost_per_member := sock_price + tshirt_price + helmet_price
  let total_expenditure := 4680
  sorry

end gridiron_football_club_members_count_l681_68127


namespace more_girls_than_boys_l681_68139

def ratio_boys_girls (B G : ℕ) : Prop := B = (3/5 : ℚ) * G

def total_students (B G : ℕ) : Prop := B + G = 16

theorem more_girls_than_boys (B G : ℕ) (h1 : ratio_boys_girls B G) (h2 : total_students B G) : G - B = 4 :=
by
  sorry

end more_girls_than_boys_l681_68139


namespace vector_subtraction_l681_68153

def a : Real × Real := (2, -1)
def b : Real × Real := (-2, 3)

theorem vector_subtraction :
  a.1 - 2 * b.1 = 6 ∧ a.2 - 2 * b.2 = -7 := by
  sorry

end vector_subtraction_l681_68153


namespace hcf_36_84_l681_68182

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := by
  sorry

end hcf_36_84_l681_68182


namespace find_other_intersection_point_l681_68138

-- Definitions
def parabola_eq (x : ℝ) : ℝ := x^2 - 2 * x - 3
def intersection_point1 : Prop := parabola_eq (-1) = 0
def intersection_point2 : Prop := parabola_eq 3 = 0

-- Proof problem
theorem find_other_intersection_point :
  intersection_point1 → intersection_point2 := by
  sorry

end find_other_intersection_point_l681_68138


namespace solve_fruit_juice_problem_l681_68168

open Real

noncomputable def fruit_juice_problem : Prop :=
  ∃ x, ((0.12 * 3 + x) / (3 + x) = 0.185) ∧ (x = 0.239)

theorem solve_fruit_juice_problem : fruit_juice_problem :=
sorry

end solve_fruit_juice_problem_l681_68168


namespace percentage_less_than_l681_68177

variable (x y z n : ℝ)
variable (hx : x = 8 * y)
variable (hy : y = 2 * |z - n|)
variable (hz : z = 1.1 * n)

theorem percentage_less_than (hx : x = 8 * y) (hy : y = 2 * |z - n|) (hz : z = 1.1 * n) :
  ((x - y) / x) * 100 = 87.5 := sorry

end percentage_less_than_l681_68177


namespace find_C_l681_68175

-- Variables and conditions
variables (A B C : ℝ)

-- Conditions given in the problem
def condition1 : Prop := A + B + C = 1000
def condition2 : Prop := A + C = 700
def condition3 : Prop := B + C = 600

-- The statement to be proved
theorem find_C (h1 : condition1 A B C) (h2 : condition2 A C) (h3 : condition3 B C) : C = 300 :=
sorry

end find_C_l681_68175


namespace article_final_price_l681_68134

theorem article_final_price (list_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) : 
  first_discount = 0.1 → 
  second_discount = 0.01999999999999997 → 
  list_price = 70 → 
  ∃ final_price, final_price = 61.74 := 
by {
  sorry
}

end article_final_price_l681_68134


namespace john_needs_392_tanks_l681_68122

/- Variables representing the conditions -/
def small_balloons : ℕ := 5000
def medium_balloons : ℕ := 5000
def large_balloons : ℕ := 5000

def small_balloon_volume : ℕ := 20
def medium_balloon_volume : ℕ := 30
def large_balloon_volume : ℕ := 50

def helium_tank_capacity : ℕ := 1000
def hydrogen_tank_capacity : ℕ := 1200
def mixture_tank_capacity : ℕ := 1500

/- Mathematical calculations -/
def helium_volume : ℕ := small_balloons * small_balloon_volume
def hydrogen_volume : ℕ := medium_balloons * medium_balloon_volume
def mixture_volume : ℕ := large_balloons * large_balloon_volume

def helium_tanks : ℕ := (helium_volume + helium_tank_capacity - 1) / helium_tank_capacity
def hydrogen_tanks : ℕ := (hydrogen_volume + hydrogen_tank_capacity - 1) / hydrogen_tank_capacity
def mixture_tanks : ℕ := (mixture_volume + mixture_tank_capacity - 1) / mixture_tank_capacity

def total_tanks : ℕ := helium_tanks + hydrogen_tanks + mixture_tanks

theorem john_needs_392_tanks : total_tanks = 392 :=
by {
  -- calculation proof goes here
  sorry
}

end john_needs_392_tanks_l681_68122


namespace unrealistic_data_l681_68103

theorem unrealistic_data :
  let A := 1000
  let A1 := 265
  let A2 := 51
  let A3 := 803
  let A1U2 := 287
  let A2U3 := 843
  let A1U3 := 919
  let A1I2 := A1 + A2 - A1U2
  let A2I3 := A2 + A3 - A2U3
  let A3I1 := A3 + A1 - A1U3
  let U := A1 + A2 + A3 - A1I2 - A2I3 - A3I1
  let A1I2I3 := A - U
  A1I2I3 > A2 :=
by
   sorry

end unrealistic_data_l681_68103


namespace running_race_total_students_l681_68187

theorem running_race_total_students 
  (number_of_first_grade_students number_of_second_grade_students : ℕ)
  (h1 : number_of_first_grade_students = 8)
  (h2 : number_of_second_grade_students = 5 * number_of_first_grade_students) :
  number_of_first_grade_students + number_of_second_grade_students = 48 := 
by
  -- we will leave the proof empty
  sorry

end running_race_total_students_l681_68187


namespace number_of_gigs_played_l681_68131

/-- Given earnings per gig for each band member and the total earnings, prove the total number of gigs played -/

def lead_singer_earnings : ℕ := 30
def guitarist_earnings : ℕ := 25
def bassist_earnings : ℕ := 20
def drummer_earnings : ℕ := 25
def keyboardist_earnings : ℕ := 20
def backup_singer1_earnings : ℕ := 15
def backup_singer2_earnings : ℕ := 18
def backup_singer3_earnings : ℕ := 12
def total_earnings : ℕ := 3465

def total_earnings_per_gig : ℕ :=
  lead_singer_earnings +
  guitarist_earnings +
  bassist_earnings +
  drummer_earnings +
  keyboardist_earnings +
  backup_singer1_earnings +
  backup_singer2_earnings +
  backup_singer3_earnings

theorem number_of_gigs_played : (total_earnings / total_earnings_per_gig) = 21 := by
  sorry

end number_of_gigs_played_l681_68131


namespace portia_high_school_students_l681_68184

theorem portia_high_school_students
  (L P M : ℕ)
  (h1 : P = 4 * L)
  (h2 : M = 2 * L)
  (h3 : P + L + M = 4200) :
  P = 2400 :=
sorry

end portia_high_school_students_l681_68184


namespace xiaoLiangComprehensiveScore_l681_68147

-- Define the scores for the three aspects
def contentScore : ℝ := 88
def deliveryAbilityScore : ℝ := 95
def effectivenessScore : ℝ := 90

-- Define the weights for the three aspects
def contentWeight : ℝ := 0.5
def deliveryAbilityWeight : ℝ := 0.4
def effectivenessWeight : ℝ := 0.1

-- Define the comprehensive score
def comprehensiveScore : ℝ :=
  (contentScore * contentWeight) +
  (deliveryAbilityScore * deliveryAbilityWeight) +
  (effectivenessScore * effectivenessWeight)

-- The theorem stating that the comprehensive score equals 91
theorem xiaoLiangComprehensiveScore : comprehensiveScore = 91 := by
  -- proof here (omitted)
  sorry

end xiaoLiangComprehensiveScore_l681_68147


namespace quadrilateral_area_l681_68166

/-
Proof Statement: For a square with a side length of 8 cm, each of whose sides is divided by a point into two equal segments, 
prove that the area of the quadrilateral formed by connecting these points is 32 cm².
-/

theorem quadrilateral_area (side_len : ℝ) (h : side_len = 8) :
  let quadrilateral_area := (side_len * side_len) / 2
  quadrilateral_area = 32 :=
by
  sorry

end quadrilateral_area_l681_68166


namespace eval_expression_l681_68143

theorem eval_expression :
  6 - 9 * (1 / 2 - 3^3) * 2 = 483 := 
sorry

end eval_expression_l681_68143


namespace sum_of_roots_eq_9_div_4_l681_68142

-- Define the values for the coefficients
def a : ℝ := -48
def b : ℝ := 108
def c : ℝ := -27

-- Define the quadratic equation and the function that represents the sum of the roots
def quadratic_eq (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Statement of the problem: Prove the sum of the roots of the quadratic equation equals 9/4
theorem sum_of_roots_eq_9_div_4 : 
  (∀ x y : ℝ, quadratic_eq x = 0 → quadratic_eq y = 0 → x ≠ y → x + y = - (b/a)) → - (b / a) = 9 / 4 :=
by
  sorry

end sum_of_roots_eq_9_div_4_l681_68142


namespace smallest_x_l681_68161

theorem smallest_x (x : ℕ) (h : 450 * x % 648 = 0) : x = 36 := 
sorry

end smallest_x_l681_68161


namespace biggest_number_l681_68169

theorem biggest_number (A B C D : ℕ) (h1 : A / B = 2 / 3) (h2 : B / C = 3 / 4) (h3 : C / D = 4 / 5) (h4 : A + B + C + D = 1344) : D = 480 := 
sorry

end biggest_number_l681_68169


namespace evaluate_expression_l681_68109

theorem evaluate_expression : (64^(1 / 6) * 16^(1 / 4) * 8^(1 / 3) = 8) :=
by
  -- sorry added to skip the proof
  sorry

end evaluate_expression_l681_68109


namespace rectangular_prism_height_eq_17_l681_68150

-- Defining the lengths of the edges of the cubes and rectangular prism
def side_length_cube1 := 10
def edges_cube := 12
def length_rect_prism := 8
def width_rect_prism := 5

-- The total length of the wire used for each shape must be equal
def wire_length_cube1 := edges_cube * side_length_cube1
def wire_length_rect_prism (h : ℕ) := 4 * length_rect_prism + 4 * width_rect_prism + 4 * h

theorem rectangular_prism_height_eq_17 (h : ℕ) :
  wire_length_cube1 = wire_length_rect_prism h → h = 17 := 
by
  -- The proof goes here
  sorry

end rectangular_prism_height_eq_17_l681_68150


namespace at_most_n_pairs_with_distance_d_l681_68162

theorem at_most_n_pairs_with_distance_d
  (n : ℕ) (hn : n ≥ 3)
  (points : Fin n → ℝ × ℝ)
  (d : ℝ)
  (hd : ∀ i j, i ≠ j → dist (points i) (points j) ≤ d)
  (dmax : ∃ i j, i ≠ j ∧ dist (points i) (points j) = d) :
  ∃ (pairs : Finset (Fin n × Fin n)), ∀ p ∈ pairs, dist (points p.1) (points p.2) = d ∧ pairs.card ≤ n := 
sorry

end at_most_n_pairs_with_distance_d_l681_68162


namespace logarithm_equation_l681_68101

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem logarithm_equation (a : ℝ) : 
  (1 / log_base 2 a + 1 / log_base 3 a + 1 / log_base 4 a = 1) → a = 24 :=
by
  sorry

end logarithm_equation_l681_68101


namespace trajectory_equation_l681_68156

theorem trajectory_equation (x y : ℝ) : x^2 + y^2 = 2 * |x| + 2 * |y| → x^2 + y^2 = 2 * |x| + 2 * |y| :=
by
  sorry

end trajectory_equation_l681_68156
