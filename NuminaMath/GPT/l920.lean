import Mathlib

namespace tan_half_theta_l920_92056

theorem tan_half_theta (θ : ℝ) (h1 : Real.sin θ = -3 / 5) (h2 : 3 * Real.pi < θ ∧ θ < 7 / 2 * Real.pi) :
  Real.tan (θ / 2) = -3 :=
sorry

end tan_half_theta_l920_92056


namespace ana_salary_after_changes_l920_92055

-- Definitions based on conditions in part (a)
def initial_salary : ℝ := 2000
def raise_factor : ℝ := 1.20
def cut_factor : ℝ := 0.80

-- Statement of the proof problem
theorem ana_salary_after_changes : 
  (initial_salary * raise_factor * cut_factor) = 1920 :=
by
  sorry

end ana_salary_after_changes_l920_92055


namespace problem1_problem2_l920_92097

-- Problem 1
theorem problem1 : (1/4 / 1/5) - 1/4 = 1 := 
by 
  sorry

-- Problem 2
theorem problem2 : ∃ x : ℚ, x + 1/2 * x = 12/5 ∧ x = 4 :=
by
  sorry

end problem1_problem2_l920_92097


namespace find_range_of_x_l920_92057

noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

theorem find_range_of_x (x : ℝ) :
  (f (2 * x) > f (x - 3)) ↔ (x < -3 ∨ x > 1) :=
sorry

end find_range_of_x_l920_92057


namespace ratio_of_segments_l920_92066

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end ratio_of_segments_l920_92066


namespace unique_geometric_progression_12_a_b_ab_l920_92003

noncomputable def geometric_progression_12_a_b_ab : Prop :=
  ∃ (a b : ℝ), ∃ r : ℝ, a = 12 * r ∧ b = 12 * r^2 ∧ 12 * r * (12 * r^2) = 144 * r^3

theorem unique_geometric_progression_12_a_b_ab :
  ∃! (a b : ℝ), ∃ r : ℝ, a = 12 * r ∧ b = 12 * r^2 ∧ 12 * r * (12 * r^2) = 144 * r^3 :=
by
  sorry

end unique_geometric_progression_12_a_b_ab_l920_92003


namespace total_amount_spent_l920_92035

noncomputable def food_price : ℝ := 160
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def tip_rate : ℝ := 0.20

theorem total_amount_spent :
  let sales_tax := sales_tax_rate * food_price
  let total_before_tip := food_price + sales_tax
  let tip := tip_rate * total_before_tip
  let total_amount := total_before_tip + tip
  total_amount = 211.20 :=
by
  -- include the proof logic here if necessary
  sorry

end total_amount_spent_l920_92035


namespace third_person_profit_share_l920_92089

noncomputable def investment_first : ℤ := 9000
noncomputable def investment_second : ℤ := investment_first + 2000
noncomputable def investment_third : ℤ := investment_second - 3000
noncomputable def investment_fourth : ℤ := 2 * investment_third
noncomputable def investment_fifth : ℤ := investment_fourth + 4000
noncomputable def total_investment : ℤ := investment_first + investment_second + investment_third + investment_fourth + investment_fifth

noncomputable def total_profit : ℤ := 25000
noncomputable def third_person_share : ℚ := (investment_third : ℚ) / (total_investment : ℚ) * (total_profit : ℚ)

theorem third_person_profit_share :
  third_person_share = 3076.92 := sorry

end third_person_profit_share_l920_92089


namespace evaluate_expression_l920_92047

theorem evaluate_expression (x : ℝ) : (1 - x^2) * (1 + x^4) = 1 - x^2 + x^4 - x^6 :=
by
  sorry

end evaluate_expression_l920_92047


namespace ticket_price_difference_l920_92043

noncomputable def price_difference (adult_price total_cost : ℕ) (num_adults num_children : ℕ) (child_price : ℕ) : ℕ :=
  adult_price - child_price

theorem ticket_price_difference :
  ∀ (adult_price total_cost num_adults num_children child_price : ℕ),
  adult_price = 19 →
  total_cost = 77 →
  num_adults = 2 →
  num_children = 3 →
  num_adults * adult_price + num_children * child_price = total_cost →
  price_difference adult_price total_cost num_adults num_children child_price = 6 :=
by
  intros
  simp [price_difference]
  sorry

end ticket_price_difference_l920_92043


namespace compound_interest_principal_l920_92033

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (Real.exp (T * Real.log (1 + R / 100)) - 1)

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem compound_interest_principal :
  let P_SI := 2800.0000000000027
  let R_SI := 5
  let T_SI := 3
  let P_CI := 4000
  let R_CI := 10
  let T_CI := 2
  let SI := simple_interest P_SI R_SI T_SI
  let CI := 2 * SI
  CI = compound_interest P_CI R_CI T_CI → P_CI = 4000 :=
by
  intros
  sorry

end compound_interest_principal_l920_92033


namespace coat_total_selling_price_l920_92008

theorem coat_total_selling_price :
  let original_price := 120
  let discount_percent := 30
  let tax_percent := 8
  let discount_amount := (discount_percent / 100) * original_price
  let sale_price := original_price - discount_amount
  let tax_amount := (tax_percent / 100) * sale_price
  let total_selling_price := sale_price + tax_amount
  total_selling_price = 90.72 :=
by
  sorry

end coat_total_selling_price_l920_92008


namespace total_votes_is_240_l920_92054

-- Defining the problem conditions
variables (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ)
def score : ℤ := likes - dislikes
def percentage_likes : ℚ := 3 / 4
def percentage_dislikes : ℚ := 1 / 4

-- Stating the given conditions
axiom h1 : total_votes = likes + dislikes
axiom h2 : (likes : ℤ) = (percentage_likes * total_votes)
axiom h3 : (dislikes : ℤ) = (percentage_dislikes * total_votes)
axiom h4 : score = 120

-- The statement to prove
theorem total_votes_is_240 : total_votes = 240 :=
by
  sorry

end total_votes_is_240_l920_92054


namespace percentage_of_passengers_in_first_class_l920_92016

theorem percentage_of_passengers_in_first_class (total_passengers : ℕ) (percentage_female : ℝ) (females_coach : ℕ) 
  (males_perc_first_class : ℝ) (Perc_first_class : ℝ) : 
  total_passengers = 120 → percentage_female = 0.45 → females_coach = 46 → males_perc_first_class = (1/3) → 
  Perc_first_class = 10 := by
  sorry

end percentage_of_passengers_in_first_class_l920_92016


namespace jim_reads_less_hours_l920_92038

-- Conditions
def initial_speed : ℕ := 40 -- pages per hour
def initial_pages_per_week : ℕ := 600 -- pages
def speed_increase_factor : ℚ := 1.5
def new_pages_per_week : ℕ := 660 -- pages

-- Calculations based on conditions
def initial_hours_per_week : ℚ := initial_pages_per_week / initial_speed
def new_speed : ℚ := initial_speed * speed_increase_factor
def new_hours_per_week : ℚ := new_pages_per_week / new_speed

-- Theorem Statement
theorem jim_reads_less_hours :
  initial_hours_per_week - new_hours_per_week = 4 :=
  sorry

end jim_reads_less_hours_l920_92038


namespace additional_time_required_l920_92017

-- Definitions based on conditions
def time_to_clean_three_sections : ℕ := 24
def total_sections : ℕ := 27

-- Rate of cleaning
def cleaning_rate_per_section (t : ℕ) (n : ℕ) : ℕ := t / n

-- Total time required to clean all sections
def total_cleaning_time (n : ℕ) (r : ℕ) : ℕ := n * r

-- Additional time required to clean the remaining sections
def additional_cleaning_time (t_total : ℕ) (t_spent : ℕ) : ℕ := t_total - t_spent

-- Theorem statement
theorem additional_time_required 
  (t3 : ℕ) (n : ℕ) (t_spent : ℕ) 
  (h₁ : t3 = time_to_clean_three_sections)
  (h₂ : n = total_sections)
  (h₃ : t_spent = time_to_clean_three_sections)
  : additional_cleaning_time (total_cleaning_time n (cleaning_rate_per_section t3 3)) t_spent = 192 :=
by
  sorry

end additional_time_required_l920_92017


namespace find_units_digit_of_n_l920_92030

-- Define the problem conditions
def units_digit (a : ℕ) : ℕ := a % 10

theorem find_units_digit_of_n (m n : ℕ) (h1 : units_digit m = 3) (h2 : units_digit (m * n) = 6) (h3 : units_digit (14^8) = 6) :
  units_digit n = 2 :=
  sorry

end find_units_digit_of_n_l920_92030


namespace circle_point_outside_range_l920_92036

theorem circle_point_outside_range (m : ℝ) :
  ¬ (1 + 1 + 4 * m - 2 * 1 + 5 * m = 0) → 
  (m > 1 ∨ (0 < m ∧ m < 1 / 4)) := 
sorry

end circle_point_outside_range_l920_92036


namespace pair_of_operations_equal_l920_92032

theorem pair_of_operations_equal :
  (-3) ^ 3 = -(3 ^ 3) ∧
  (¬((-2) ^ 4 = -(2 ^ 4))) ∧
  (¬((3 / 2) ^ 2 = (2 / 3) ^ 2)) ∧
  (¬(2 ^ 3 = 3 ^ 2)) :=
by 
  sorry

end pair_of_operations_equal_l920_92032


namespace evaluate_expression_l920_92021

theorem evaluate_expression : 5000 * 5000^3000 = 5000^3001 := 
by sorry

end evaluate_expression_l920_92021


namespace height_of_flagpole_l920_92039

-- Define the given conditions
variables (h : ℝ) -- height of the flagpole
variables (s_f : ℝ) (s_b : ℝ) (h_b : ℝ) -- s_f: shadow length of flagpole, s_b: shadow length of building, h_b: height of building

-- Problem conditions
def flagpole_shadow := (s_f = 45)
def building_shadow := (s_b = 50)
def building_height := (h_b = 20)

-- Mathematically equivalent statement
theorem height_of_flagpole
  (h_f : ℝ) (hsf : flagpole_shadow s_f) (hsb : building_shadow s_b) (hhb : building_height h_b)
  (similar_conditions : h / s_f = h_b / s_b) :
  h_f = 18 :=
by
  sorry

end height_of_flagpole_l920_92039


namespace find_x_2y_3z_l920_92059

theorem find_x_2y_3z (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (h1 : x ≤ y) (h2 : y ≤ z) (h3 : x + y + z = 12) (h4 : x * y + y * z + z * x = 41) :
  x + 2 * y + 3 * z = 29 :=
by
  sorry

end find_x_2y_3z_l920_92059


namespace least_positive_integer_lemma_l920_92050

theorem least_positive_integer_lemma :
  ∃ x : ℕ, x > 0 ∧ x + 7237 ≡ 5017 [MOD 12] ∧ (∀ y : ℕ, y > 0 ∧ y + 7237 ≡ 5017 [MOD 12] → x ≤ y) :=
by
  sorry

end least_positive_integer_lemma_l920_92050


namespace smallest_integer_larger_than_expression_l920_92012

theorem smallest_integer_larger_than_expression :
  ∃ n : ℤ, n = 248 ∧ (↑n > ((Real.sqrt 5 + Real.sqrt 3) ^ 4 : ℝ)) :=
by
  sorry

end smallest_integer_larger_than_expression_l920_92012


namespace teaching_arrangements_l920_92088

-- Define the conditions
structure Conditions :=
  (teach_A : ℕ)
  (teach_B : ℕ)
  (teach_C : ℕ)
  (teach_D : ℕ)
  (max_teach_AB : ∀ t, t = teach_A ∨ t = teach_B → t ≤ 2)
  (max_teach_CD : ∀ t, t = teach_C ∨ t = teach_D → t ≤ 1)
  (total_periods : ℕ)
  (teachers_per_period : ℕ)

-- Constants and assumptions
def problem_conditions : Conditions := {
  teach_A := 2,
  teach_B := 2,
  teach_C := 1,
  teach_D := 1,
  max_teach_AB := by sorry,
  max_teach_CD := by sorry,
  total_periods := 2,
  teachers_per_period := 2
}

-- Define the proof goal
theorem teaching_arrangements (c : Conditions) :
  c = problem_conditions → ∃ arrangements, arrangements = 19 :=
by
  sorry

end teaching_arrangements_l920_92088


namespace smallest_positive_period_monotonic_decreasing_interval_l920_92049

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.sin x * Real.cos x

theorem smallest_positive_period (T : ℝ) :
  (∀ x, f (x + T) = f x) ∧ T > 0 → T = Real.pi :=
by
  sorry

theorem monotonic_decreasing_interval :
  (∀ x, x ∈ Set.Icc (3 * Real.pi / 8) (7 * Real.pi / 8) → ∃ k : ℤ, 
     f (x + k * π) = f x ∧ f (x + k * π) ≤ f (x + (k + 1) * π)) :=
by
  sorry

end smallest_positive_period_monotonic_decreasing_interval_l920_92049


namespace whatsapp_messages_total_l920_92048

-- Define conditions
def messages_monday : ℕ := 300
def messages_tuesday : ℕ := 200
def messages_wednesday : ℕ := messages_tuesday + 300
def messages_thursday : ℕ := 2 * messages_wednesday
def messages_friday : ℕ := messages_thursday + (20 * messages_thursday) / 100
def messages_saturday : ℕ := messages_friday - (10 * messages_friday) / 100

-- Theorem statement to be proved
theorem whatsapp_messages_total :
  messages_monday + messages_tuesday + messages_wednesday + messages_thursday + messages_friday + messages_saturday = 4280 :=
by 
  sorry

end whatsapp_messages_total_l920_92048


namespace sum_first_n_terms_of_geometric_seq_l920_92093

variable {α : Type*} [LinearOrderedField α] (a r : α) (n : ℕ)

def geometric_sequence (a r : α) (n : ℕ) : α :=
  a * r ^ (n - 1)

def sum_geometric_sequence (a r : α) (n : ℕ) : α :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_first_n_terms_of_geometric_seq (h₁ : a * r + a * r^3 = 20) 
    (h₂ : a * r^2 + a * r^4 = 40) :
  sum_geometric_sequence a r n = 2^(n + 1) - 2 := 
sorry

end sum_first_n_terms_of_geometric_seq_l920_92093


namespace area_of_square_l920_92010

-- Define the conditions given in the problem
def radius_circle := 7 -- radius of each circle in inches

def diameter_circle := 2 * radius_circle -- diameter of each circle

def side_length_square := 2 * diameter_circle -- side length of the square

-- State the theorem we want to prove
theorem area_of_square : side_length_square ^ 2 = 784 := 
by
  sorry

end area_of_square_l920_92010


namespace union_eq_set_l920_92025

noncomputable def M : Set ℤ := {x | |x| < 2}
noncomputable def N : Set ℤ := {-2, -1, 0}

theorem union_eq_set : M ∪ N = {-2, -1, 0, 1} := by
  sorry

end union_eq_set_l920_92025


namespace find_c_l920_92094

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : c = 7 :=
sorry

end find_c_l920_92094


namespace triangle_ABC_right_angled_l920_92098

variable {α : Type*} [LinearOrderedField α]

variables (a b c : α)
variables (A B C : ℝ)

theorem triangle_ABC_right_angled
  (h1 : b^2 = c^2 + a^2 - c * a)
  (h2 : Real.sin A = 2 * Real.sin C)
  (h3 : Real.cos B = 1 / 2) :
  B = (Real.pi / 2) := by
  sorry

end triangle_ABC_right_angled_l920_92098


namespace stating_area_trapezoid_AMBQ_is_18_l920_92002

/-- Definition of the 20-sided polygon configuration with 2 unit sides and right-angle turns. -/
structure Polygon20 where
  sides : ℕ → ℝ
  units : ∀ i, sides i = 2
  right_angles : ∀ i, (i + 1) % 20 ≠ i -- Right angles between consecutive sides

/-- Intersection point of AJ and DP, named M, under the given polygon configuration. -/
def intersection_point (p : Polygon20) : ℝ × ℝ :=
  (5 * p.sides 0, 5 * p.sides 1)  -- Assuming relevant distances for simplicity

/-- Area of the trapezoid AMBQ formed given the defined Polygon20. -/
noncomputable def area_trapezoid_AMBQ (p : Polygon20) : ℝ :=
  let base1 := 10 * p.sides 0
  let base2 := 8 * p.sides 0
  let height := p.sides 0
  (base1 + base2) * height / 2

/-- 
  Theorem stating the area of the trapezoid AMBQ in the given configuration.
  We prove that the area is 18 units.
-/
theorem area_trapezoid_AMBQ_is_18 (p : Polygon20) :
  area_trapezoid_AMBQ p = 18 :=
sorry -- Proof to be done

end stating_area_trapezoid_AMBQ_is_18_l920_92002


namespace count_five_digit_progressive_numbers_find_110th_five_digit_progressive_number_l920_92045

def is_progressive_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 : ℕ), 1 ≤ d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 ≤ 9 ∧
                          n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5

theorem count_five_digit_progressive_numbers : ∃ n, n = 126 :=
by
  sorry

theorem find_110th_five_digit_progressive_number : ∃ n, n = 34579 :=
by
  sorry

end count_five_digit_progressive_numbers_find_110th_five_digit_progressive_number_l920_92045


namespace savings_fraction_l920_92041

variable (P : ℝ) 
variable (S : ℝ)
variable (E : ℝ)
variable (T : ℝ)

theorem savings_fraction :
  (12 * P * S) = 2 * P * (1 - S) → S = 1 / 7 :=
by
  intro h
  sorry

end savings_fraction_l920_92041


namespace largest_whole_number_satisfying_inequality_l920_92040

theorem largest_whole_number_satisfying_inequality : ∃ n : ℤ, (1 / 3 + n / 7 < 1) ∧ (∀ m : ℤ, (1 / 3 + m / 7 < 1) → m ≤ n) ∧ n = 4 :=
sorry

end largest_whole_number_satisfying_inequality_l920_92040


namespace max_pies_without_ingredients_l920_92099

def total_pies : ℕ := 30
def blueberry_pies : ℕ := total_pies / 3
def raspberry_pies : ℕ := (3 * total_pies) / 5
def blackberry_pies : ℕ := (5 * total_pies) / 6
def walnut_pies : ℕ := total_pies / 10

theorem max_pies_without_ingredients : 
  (total_pies - blackberry_pies) = 5 :=
by 
  -- We only require the proof part.
  sorry

end max_pies_without_ingredients_l920_92099


namespace solutions_in_nat_solutions_in_non_neg_int_l920_92015

-- Definitions for Part A
def nat_sol_count (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem solutions_in_nat (x1 x2 x3 : ℕ) : 
  (x1 > 0) → (x2 > 0) → (x3 > 0) → (x1 + x2 + x3 = 1000) → 
  nat_sol_count 997 3 = Nat.choose 999 2 := sorry

-- Definitions for Part B
theorem solutions_in_non_neg_int (x1 x2 x3 : ℕ) : 
  (x1 + x2 + x3 = 1000) → 
  nat_sol_count 1000 3 = Nat.choose 1002 2 := sorry

end solutions_in_nat_solutions_in_non_neg_int_l920_92015


namespace star_operation_result_l920_92060

def set_minus (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∉ B}

def set_star (A B : Set ℝ) : Set ℝ :=
  set_minus A B ∪ set_minus B A

def A : Set ℝ := { y : ℝ | y ≥ 0 }
def B : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 3 }

theorem star_operation_result :
  set_star A B = {x : ℝ | (-3 ≤ x ∧ x < 0) ∨ (x > 3)} :=
  sorry

end star_operation_result_l920_92060


namespace distance_between_planes_is_zero_l920_92058

def plane1 (x y z : ℝ) : Prop := x - 2 * y + 2 * z = 9
def plane2 (x y z : ℝ) : Prop := 2 * x - 4 * y + 4 * z = 18

theorem distance_between_planes_is_zero :
  (∀ x y z : ℝ, plane1 x y z ↔ plane2 x y z) → 0 = 0 :=
by
  sorry

end distance_between_planes_is_zero_l920_92058


namespace simplify_expression_l920_92080

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by 
  sorry

end simplify_expression_l920_92080


namespace verify_total_bill_l920_92007

def fixed_charge : ℝ := 20
def daytime_rate : ℝ := 0.10
def evening_rate : ℝ := 0.05
def free_evening_minutes : ℕ := 200

def daytime_minutes : ℕ := 200
def evening_minutes : ℕ := 300

noncomputable def total_bill : ℝ :=
  fixed_charge + (daytime_minutes * daytime_rate) +
  ((evening_minutes - free_evening_minutes) * evening_rate)

theorem verify_total_bill : total_bill = 45 := by
  sorry

end verify_total_bill_l920_92007


namespace bubble_gum_cost_l920_92076

-- Define the conditions
def total_cost : ℕ := 2448
def number_of_pieces : ℕ := 136

-- Main theorem to state that each piece of bubble gum costs 18 cents
theorem bubble_gum_cost : total_cost / number_of_pieces = 18 :=
by
  sorry

end bubble_gum_cost_l920_92076


namespace jake_peaches_l920_92052

noncomputable def steven_peaches : ℕ := 15
noncomputable def jake_fewer : ℕ := 7

theorem jake_peaches : steven_peaches - jake_fewer = 8 :=
by
  sorry

end jake_peaches_l920_92052


namespace cube_loop_probability_l920_92001

-- Define the number of faces and alignments for a cube
def total_faces := 6
def stripe_orientations_per_face := 2

-- Define the total possible stripe combinations
def total_stripe_combinations := stripe_orientations_per_face ^ total_faces

-- Define the combinations for both vertical and horizontal loops
def vertical_and_horizontal_loop_combinations := 64

-- Define the probability space
def probability_at_least_one_each := vertical_and_horizontal_loop_combinations / total_stripe_combinations

-- The main theorem to state the probability of having at least one vertical and one horizontal loop
theorem cube_loop_probability : probability_at_least_one_each = 1 := by
  sorry

end cube_loop_probability_l920_92001


namespace cylinder_to_sphere_volume_ratio_l920_92044

theorem cylinder_to_sphere_volume_ratio:
  ∀ (a r : ℝ), (a^2 = π * r^2) → (a^3)/( (4/3) * π * r^3) = 3/2 :=
by
  intros a r h
  sorry

end cylinder_to_sphere_volume_ratio_l920_92044


namespace basketball_card_price_l920_92046

variable (x : ℝ)

def total_cost_basketball_cards (x : ℝ) : ℝ := 2 * x
def total_cost_baseball_cards : ℝ := 5 * 4
def total_spent : ℝ := 50 - 24

theorem basketball_card_price :
  total_cost_basketball_cards x + total_cost_baseball_cards = total_spent ↔ x = 3 := by
  sorry

end basketball_card_price_l920_92046


namespace appointment_duration_l920_92013

-- Define the given conditions
def total_workday_hours : ℕ := 8
def permits_per_hour : ℕ := 50
def total_permits : ℕ := 100
def stamping_time : ℕ := total_permits / permits_per_hour
def appointment_time : ℕ := (total_workday_hours - stamping_time) / 2

-- State the theorem and ignore the proof part by adding sorry
theorem appointment_duration : appointment_time = 3 := by
  -- skipping the proof steps
  sorry

end appointment_duration_l920_92013


namespace divisible_by_56_l920_92087

theorem divisible_by_56 (n : ℕ) (h1 : ∃ k, 3 * n + 1 = k * k) (h2 : ∃ m, 4 * n + 1 = m * m) : 56 ∣ n := 
sorry

end divisible_by_56_l920_92087


namespace parallel_trans_l920_92078

variables {Line : Type} (a b c : Line)

-- Define parallel relation
def parallel (x y : Line) : Prop := sorry -- Replace 'sorry' with the actual definition

-- The main theorem
theorem parallel_trans (h1 : parallel a c) (h2 : parallel b c) : parallel a b :=
sorry

end parallel_trans_l920_92078


namespace sum_arithmetic_sequence_l920_92072

theorem sum_arithmetic_sequence (m : ℕ) (S : ℕ → ℕ) 
  (h1 : S m = 30) 
  (h2 : S (3 * m) = 90) : 
  S (2 * m) = 60 := 
sorry

end sum_arithmetic_sequence_l920_92072


namespace circle_m_range_l920_92018

theorem circle_m_range (m : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 - 2 * x + 6 * y + m = 0 → m < 10) :=
sorry

end circle_m_range_l920_92018


namespace range_of_x_l920_92083

noncomputable def problem_statement (x : ℝ) : Prop :=
  ∀ m : ℝ, abs m ≤ 2 → m * x^2 - 2 * x - m + 1 < 0 

theorem range_of_x (x : ℝ) :
  problem_statement x → ( ( -1 + Real.sqrt 7) / 2 < x ∧ x < ( 1 + Real.sqrt 3) / 2) :=
by
  intros h
  sorry

end range_of_x_l920_92083


namespace work_completion_days_l920_92023

theorem work_completion_days (d : ℝ) : (1 / 15 + 1 / d = 1 / 11.25) → d = 45 := sorry

end work_completion_days_l920_92023


namespace remainder_M_mod_32_l920_92090

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l920_92090


namespace one_minus_repeating_decimal_l920_92065

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ := x

theorem one_minus_repeating_decimal:
  ∀ (x : ℚ), x = 1/3 → 1 - x = 2/3 :=
by
  sorry

end one_minus_repeating_decimal_l920_92065


namespace anna_stamp_count_correct_l920_92026

-- Defining the initial counts of stamps
def anna_initial := 37
def alison_initial := 28
def jeff_initial := 31

-- Defining the operations
def alison_gives_half_to_anna := alison_initial / 2
def anna_after_receiving_from_alison := anna_initial + alison_gives_half_to_anna
def anna_final := anna_after_receiving_from_alison - 2 + 1

-- Formalizing the proof problem
theorem anna_stamp_count_correct : anna_final = 50 := by
  -- proof omitted
  sorry

end anna_stamp_count_correct_l920_92026


namespace find_fff_l920_92009

def f (x : ℚ) : ℚ :=
  if x ≥ 2 then x + 2 else x * x

theorem find_fff : f (f (3/2)) = 17/4 := by
  sorry

end find_fff_l920_92009


namespace Austin_work_hours_on_Wednesdays_l920_92091

variable {W : ℕ}

theorem Austin_work_hours_on_Wednesdays
  (h1 : 5 * 2 + 5 * W + 5 * 3 = 25 + 5 * W)
  (h2 : 6 * (25 + 5 * W) = 180)
  : W = 1 := by
  sorry

end Austin_work_hours_on_Wednesdays_l920_92091


namespace find_5y_45_sevenths_l920_92034

theorem find_5y_45_sevenths (x y : ℝ) 
(h1 : 3 * x + 4 * y = 0) 
(h2 : x = y + 3) : 
5 * y = -45 / 7 :=
by
  sorry

end find_5y_45_sevenths_l920_92034


namespace find_x_when_fx_eq_3_l920_92000

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ -1 then x + 2 else
if x < 2 then x^2 else
2 * x

theorem find_x_when_fx_eq_3 : ∃ x : ℝ, f x = 3 ∧ x = Real.sqrt 3 := by
  sorry

end find_x_when_fx_eq_3_l920_92000


namespace original_number_is_3_l920_92084

theorem original_number_is_3 
  (A B C D E : ℝ) 
  (h1 : (A + B + C + D + E) / 5 = 8) 
  (h2 : (8 + B + C + D + E) / 5 = 9): 
  A = 3 :=
sorry

end original_number_is_3_l920_92084


namespace giants_need_to_win_more_games_l920_92042

/-- The Giants baseball team is trying to make their league playoff.
They have played 20 games and won 12 of them. To make the playoffs, they need to win 2/3 of 
their games over the season. If there are 10 games left, how many do they have to win to
make the playoffs? 
-/
theorem giants_need_to_win_more_games (played won needed_won total remaining required_wins additional_wins : ℕ)
    (h1 : played = 20)
    (h2 : won = 12)
    (h3 : remaining = 10)
    (h4 : total = played + remaining)
    (h5 : total = 30)
    (h6 : required_wins = 2 * total / 3)
    (h7 : additional_wins = required_wins - won) :
    additional_wins = 8 := 
    by
      -- sorry should be used if the proof steps were required.
sorry

end giants_need_to_win_more_games_l920_92042


namespace remainder_is_correct_l920_92006

def dividend : ℕ := 725
def divisor : ℕ := 36
def quotient : ℕ := 20

theorem remainder_is_correct : ∃ (remainder : ℕ), dividend = (divisor * quotient) + remainder ∧ remainder = 5 := by
  sorry

end remainder_is_correct_l920_92006


namespace red_or_black_prob_red_black_or_white_prob_l920_92070

-- Defining the probabilities
def prob_red : ℚ := 5 / 12
def prob_black : ℚ := 4 / 12
def prob_white : ℚ := 2 / 12
def prob_green : ℚ := 1 / 12

-- Question 1: Probability of drawing a red or black ball
theorem red_or_black_prob : prob_red + prob_black = 3 / 4 :=
by sorry

-- Question 2: Probability of drawing a red, black, or white ball
theorem red_black_or_white_prob : prob_red + prob_black + prob_white = 11 / 12 :=
by sorry

end red_or_black_prob_red_black_or_white_prob_l920_92070


namespace intersection_A_complement_B_eq_minus_three_to_zero_l920_92096

-- Define the set A
def A : Set ℝ := { x : ℝ | x^2 + x - 6 ≤ 0 }

-- Define the set B
def B : Set ℝ := { y : ℝ | ∃ x : ℝ, y = Real.sqrt x ∧ 0 ≤ x ∧ x ≤ 4 }

-- Define the complement of B
def C_RB : Set ℝ := { y : ℝ | ¬ (y ∈ B) }

-- The proof problem
theorem intersection_A_complement_B_eq_minus_three_to_zero :
  (A ∩ C_RB) = { x : ℝ | -3 ≤ x ∧ x < 0 } :=
by
  sorry

end intersection_A_complement_B_eq_minus_three_to_zero_l920_92096


namespace zero_point_interval_l920_92092

noncomputable def f (x : ℝ) : ℝ := -x^3 - 3 * x + 5

theorem zero_point_interval: 
  ∃ x₀ : ℝ, f x₀ = 0 → 1 < x₀ ∧ x₀ < 2 :=
sorry

end zero_point_interval_l920_92092


namespace totalFriendsAreFour_l920_92069

-- Define the friends
def friends := ["Mary", "Sam", "Keith", "Alyssa"]

-- Define the number of friends
def numberOfFriends (f : List String) : ℕ := f.length

-- Claim that the number of friends is 4
theorem totalFriendsAreFour : numberOfFriends friends = 4 :=
by
  -- Skip proof
  sorry

end totalFriendsAreFour_l920_92069


namespace imag_part_z_is_3_l920_92068

namespace ComplexMultiplication

-- Define the imaginary unit i
def i := Complex.I

-- Define the complex number z
def z := (1 + 2 * i) * (2 - i)

-- Define the imaginary part of a complex number
def imag_part (z : ℂ) : ℂ := Complex.im z

-- Statement to prove: The imaginary part of z = 3
theorem imag_part_z_is_3 : imag_part z = 3 := by
  sorry

end ComplexMultiplication

end imag_part_z_is_3_l920_92068


namespace cube_volume_given_face_area_l920_92082

theorem cube_volume_given_face_area (s : ℝ) (h : s^2 = 36) : s^3 = 216 := by
  sorry

end cube_volume_given_face_area_l920_92082


namespace find_coordinates_of_D_l920_92064

theorem find_coordinates_of_D
  (A B C D : ℝ × ℝ)
  (hA : A = (-1, 2))
  (hB : B = (0, 0))
  (hC : C = (1, 7))
  (hParallelogram : ∃ u v, u * (B - A) + v * (C - D) = (0, 0) ∧ u * (C - D) + v * (B - A) = (0, 0)) :
  D = (0, 9) :=
sorry

end find_coordinates_of_D_l920_92064


namespace pentagon_square_ratio_l920_92077

theorem pentagon_square_ratio (s p : ℕ) (h1 : 4 * s = 20) (h2 : 5 * p = 20) :
  p / s = 4 / 5 :=
by
  sorry

end pentagon_square_ratio_l920_92077


namespace binomial_distribution_parameters_l920_92095

noncomputable def E (n : ℕ) (p : ℝ) : ℝ := n * p
noncomputable def D (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_distribution_parameters (n : ℕ) (p : ℝ) 
  (h1 : E n p = 2.4) (h2 : D n p = 1.44) : 
  n = 6 ∧ p = 0.4 :=
by
  sorry

end binomial_distribution_parameters_l920_92095


namespace fifteenth_number_in_base_5_l920_92074

theorem fifteenth_number_in_base_5 :
  ∃ n : ℕ, n = 15 ∧ (n : ℕ) = 3 * 5^1 + 0 * 5^0 :=
by
  sorry

end fifteenth_number_in_base_5_l920_92074


namespace inequality_AM_GM_l920_92067

theorem inequality_AM_GM (a b t : ℝ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : 0 < t) : 
  (a^2 / (b^t - 1) + b^(2 * t) / (a^t - 1)) ≥ 8 :=
by
  sorry

end inequality_AM_GM_l920_92067


namespace probability_blue_given_glass_l920_92075

-- Defining the various conditions given in the problem
def total_red_balls : ℕ := 5
def total_blue_balls : ℕ := 11
def red_glass_balls : ℕ := 2
def red_wooden_balls : ℕ := 3
def blue_glass_balls : ℕ := 4
def blue_wooden_balls : ℕ := 7
def total_balls : ℕ := total_red_balls + total_blue_balls
def total_glass_balls : ℕ := red_glass_balls + blue_glass_balls

-- The mathematically equivalent proof problem statement.
theorem probability_blue_given_glass :
  (blue_glass_balls : ℚ) / (total_glass_balls : ℚ) = 2 / 3 := by
sorry

end probability_blue_given_glass_l920_92075


namespace color_divisors_with_conditions_l920_92063

/-- Define the primes, product of the first 100 primes, and set S -/
def first_100_primes : List Nat := sorry -- Assume we have the list of first 100 primes
def product_of_first_100_primes : Nat := first_100_primes.foldr (· * ·) 1
def S := {d : Nat | d > 1 ∧ ∃ m, product_of_first_100_primes = m * d}

/-- Statement of the problem in Lean 4 -/
theorem color_divisors_with_conditions :
  (∃ (k : Nat), (∀ (coloring : S → Fin k), 
    (∀ s1 s2 s3 : S, (s1 * s2 * s3 = product_of_first_100_primes) → (coloring s1 = coloring s2 ∨ coloring s1 = coloring s3 ∨ coloring s2 = coloring s3)) ∧
    (∀ c : Fin k, ∃ s : S, coloring s = c))) ↔ k = 100 := 
by
  sorry

end color_divisors_with_conditions_l920_92063


namespace inequality_proof_l920_92011

-- Definitions for the conditions
variable (x y : ℝ)

-- Conditions
def conditions : Prop := 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Problem statement to be proven
theorem inequality_proof (h : conditions x y) : 
  x^3 + x * y^2 + 2 * x * y ≤ 2 * x^2 * y + x^2 + x + y := 
by 
  sorry

end inequality_proof_l920_92011


namespace point_on_x_axis_m_eq_2_l920_92071

theorem point_on_x_axis_m_eq_2 (m : ℝ) (h : (m + 5, m - 2).2 = 0) : m = 2 :=
sorry

end point_on_x_axis_m_eq_2_l920_92071


namespace second_occurrence_at_55_l920_92031

/-- On the highway, starting from 3 kilometers, there is a speed limit sign every 4 kilometers,
and starting from 10 kilometers, there is a speed monitoring device every 9 kilometers.
The first time both types of facilities are encountered simultaneously is at 19 kilometers.
The second time both types of facilities are encountered simultaneously is at 55 kilometers. -/
theorem second_occurrence_at_55 :
  ∀ (k : ℕ), (∃ n m : ℕ, 3 + 4 * n = k ∧ 10 + 9 * m = k ∧ 19 + 36 = k) := sorry

end second_occurrence_at_55_l920_92031


namespace factorial_sum_power_of_two_l920_92022

theorem factorial_sum_power_of_two (a b c n : ℕ) (h : a ≤ b ∧ b ≤ c) :
  a! + b! + c! = 2^n →
  (a = 1 ∧ b = 1 ∧ c = 2) ∨
  (a = 1 ∧ b = 1 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 4) ∨
  (a = 2 ∧ b = 3 ∧ c = 5) :=
by
  sorry

end factorial_sum_power_of_two_l920_92022


namespace avg_rate_of_change_eq_l920_92085

variable (Δx : ℝ)

def function_y (x : ℝ) : ℝ := x^2 + 1

theorem avg_rate_of_change_eq : (function_y (1 + Δx) - function_y 1) / Δx = 2 + Δx :=
by
  sorry

end avg_rate_of_change_eq_l920_92085


namespace ellipse_foci_distance_l920_92051

theorem ellipse_foci_distance :
  (∀ x y : ℝ, x^2 / 56 + y^2 / 14 = 8) →
  ∃ d : ℝ, d = 8 * Real.sqrt 21 :=
by
  sorry

end ellipse_foci_distance_l920_92051


namespace triangle_perimeter_l920_92020

theorem triangle_perimeter (m : ℝ) (a b : ℝ) (h1 : 3 ^ 2 - 3 * (m + 1) + 2 * m = 0)
  (h2 : a ^ 2 - (m + 1) * a + 2 * m = 0)
  (h3 : b ^ 2 - (m + 1) * b + 2 * m = 0)
  (h4 : a = 3 ∨ b = 3)
  (h5 : a ≠ b ∨ a = b)
  (hAB : a ≠ b ∨ a = b) :
  (∀ s₁ s₂ : ℝ, s₁ = a ∨ s₁ = b ∧ s₂ = a ∨ s₂ = b ∧ s₁ ≠ s₂ → s₁ + s₁ + s₂ = 10 ∨ s₁ + s₁ + s₂ = 11) ∨
  (∀ s₁ s₂ : ℝ, s₁ = a ∨ s₁ = b ∧ s₂ = a ∨ s₂ = b ∧ s₁ = s₂ → b + b + a = 10 ∨ b + b + a = 11) := by
  sorry

end triangle_perimeter_l920_92020


namespace base_conversion_addition_l920_92028

theorem base_conversion_addition :
  (214 % 8 / 32 % 5 + 343 % 9 / 133 % 4) = 9134 / 527 :=
by sorry

end base_conversion_addition_l920_92028


namespace ticTacToeConfigCorrect_l920_92086

def ticTacToeConfigCount (board : Fin 3 → Fin 3 → Option Char) : Nat := 
  sorry -- this function will count the configurations according to the game rules

theorem ticTacToeConfigCorrect (board : Fin 3 → Fin 3 → Option Char) :
  ticTacToeConfigCount board = 438 := 
  sorry

end ticTacToeConfigCorrect_l920_92086


namespace cubs_more_home_runs_l920_92073

-- Define the conditions for the Chicago Cubs
def cubs_home_runs_third_inning : Nat := 2
def cubs_home_runs_fifth_inning : Nat := 1
def cubs_home_runs_eighth_inning : Nat := 2

-- Define the conditions for the Cardinals
def cardinals_home_runs_second_inning : Nat := 1
def cardinals_home_runs_fifth_inning : Nat := 1

-- Total home runs scored by each team
def total_cubs_home_runs : Nat :=
  cubs_home_runs_third_inning + cubs_home_runs_fifth_inning + cubs_home_runs_eighth_inning

def total_cardinals_home_runs : Nat :=
  cardinals_home_runs_second_inning + cardinals_home_runs_fifth_inning

-- The statement to prove
theorem cubs_more_home_runs : total_cubs_home_runs - total_cardinals_home_runs = 3 := by
  sorry

end cubs_more_home_runs_l920_92073


namespace classmates_ate_cake_l920_92004

theorem classmates_ate_cake (n : ℕ) 
  (h1 : ∀ i, 1 ≤ i → i ≤ n → (1 : ℚ) / 11 ≥ 1 / i ∧ 1 / i ≥ 1 / 14) : 
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end classmates_ate_cake_l920_92004


namespace number_divisible_by_45_and_6_l920_92014

theorem number_divisible_by_45_and_6 (k : ℕ) (h1 : 1 ≤ k) (h2 : ∃ n : ℕ, 190 + 90 * (k - 1) ≤  n ∧ n < 190 + 90 * k) 
: 190 + 90 * 5 = 720 := by
  sorry

end number_divisible_by_45_and_6_l920_92014


namespace angle_problem_l920_92081

-- Definitions for degrees and minutes
structure Angle where
  degrees : ℕ
  minutes : ℕ

-- Adding two angles
def add_angles (a1 a2 : Angle) : Angle :=
  let total_minutes := a1.minutes + a2.minutes
  let extra_degrees := total_minutes / 60
  { degrees := a1.degrees + a2.degrees + extra_degrees,
    minutes := total_minutes % 60 }

-- Subtracting two angles
def sub_angles (a1 a2 : Angle) : Angle :=
  let total_minutes := if a1.minutes < a2.minutes then a1.minutes + 60 else a1.minutes
  let extra_deg := if a1.minutes < a2.minutes then 1 else 0
  { degrees := a1.degrees - a2.degrees - extra_deg,
    minutes := total_minutes - a2.minutes }

-- Multiplying an angle by a constant
def mul_angle (a : Angle) (k : ℕ) : Angle :=
  let total_minutes := a.minutes * k
  let extra_degrees := total_minutes / 60
  { degrees := a.degrees * k + extra_degrees,
    minutes := total_minutes % 60 }

-- Given angles
def angle1 : Angle := { degrees := 24, minutes := 31}
def angle2 : Angle := { degrees := 62, minutes := 10}

-- Prove the problem statement
theorem angle_problem : sub_angles (mul_angle angle1 4) angle2 = { degrees := 35, minutes := 54} :=
  sorry

end angle_problem_l920_92081


namespace smallest_x_for_quadratic_l920_92061

theorem smallest_x_for_quadratic :
  ∃ x, 8 * x^2 - 38 * x + 35 = 0 ∧ (∀ y, 8 * y^2 - 38 * y + 35 = 0 → x ≤ y) ∧ x = 1.25 :=
by
  sorry

end smallest_x_for_quadratic_l920_92061


namespace roots_are_reciprocals_eq_a_minus_one_l920_92019

theorem roots_are_reciprocals_eq_a_minus_one (a : ℝ) :
  (∀ x y : ℝ, x + y = -(a - 1) ∧ x * y = a^2 → x * y = 1) → a = -1 :=
by
  intro h
  sorry

end roots_are_reciprocals_eq_a_minus_one_l920_92019


namespace jordan_final_weight_l920_92005

-- Defining the initial weight and the weight losses over the specified weeks
def initial_weight := 250
def loss_first_4_weeks := 4 * 3
def loss_next_8_weeks := 8 * 2
def total_loss := loss_first_4_weeks + loss_next_8_weeks

-- Theorem stating the final weight of Jordan
theorem jordan_final_weight : initial_weight - total_loss = 222 := by
  -- We skip the proof as requested
  sorry

end jordan_final_weight_l920_92005


namespace length_de_l920_92029

theorem length_de (a b c d e : ℝ) (ab bc cd de ac ae : ℝ)
  (H1 : ab = 5)
  (H2 : bc = 2 * cd)
  (H3 : ac = ab + bc)
  (H4 : ac = 11)
  (H5 : ae = ab + bc + cd + de)
  (H6 : ae = 18) :
  de = 4 :=
by {
  sorry
}

-- Explanation:
-- a, b, c, d, e are points on a straight line
-- ab, bc, cd, de, ac, ae are lengths of segments between these points
-- H1: ab = 5
-- H2: bc = 2 * cd
-- H3: ac = ab + bc
-- H4: ac = 11
-- H5: ae = ab + bc + cd + de
-- H6: ae = 18
-- Prove that de = 4

end length_de_l920_92029


namespace inequality_solution_set_l920_92062

theorem inequality_solution_set (a b c : ℝ)
  (h1 : a < 0)
  (h2 : b = -a)
  (h3 : c = -2 * a) :
  ∀ x : ℝ, (c * x^2 + b * x + a > 0) ↔ (x < -1 ∨ x > 1 / 2) :=
by
  sorry

end inequality_solution_set_l920_92062


namespace monotonically_increasing_interval_l920_92024

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos ((2 / 3) * x - (5 * Real.pi / 12))

theorem monotonically_increasing_interval 
  (φ : ℝ) (h1 : -Real.pi / 2 < φ) (h2 : φ < 0) 
  (h3 : 2 * (Real.pi / 8) + φ = Real.pi / 4) : 
  ∀ x : ℝ, (-(Real.pi / 2) ≤ x) ∧ (x ≤ Real.pi / 2) ↔ ∃ k : ℤ, x ∈ [(-7 * Real.pi / 8 + 3 * k * Real.pi), (5 * Real.pi / 8 + 3 * k * Real.pi)] :=
sorry

end monotonically_increasing_interval_l920_92024


namespace minimum_gloves_needed_l920_92037

-- Definitions based on conditions:
def participants : Nat := 43
def gloves_per_participant : Nat := 2

-- Problem statement proving the minimum number of gloves needed
theorem minimum_gloves_needed : participants * gloves_per_participant = 86 := by
  -- sorry allows us to omit the proof, focusing only on the formal statement
  sorry

end minimum_gloves_needed_l920_92037


namespace least_value_of_sum_l920_92053

theorem least_value_of_sum (x y z : ℤ) 
  (h_cond : (x - 10) * (y - 5) * (z - 2) = 1000) : x + y + z ≥ 56 :=
sorry

end least_value_of_sum_l920_92053


namespace johns_average_speed_remaining_duration_l920_92027

noncomputable def average_speed_remaining_duration : ℝ :=
  let total_distance := 150
  let total_time := 3
  let first_hour_speed := 45
  let stop_time := 0.5
  let next_45_minutes_speed := 50
  let next_45_minutes_time := 0.75
  let driving_time := total_time - stop_time
  let distance_first_hour := first_hour_speed * 1
  let distance_next_45_minutes := next_45_minutes_speed * next_45_minutes_time
  let remaining_distance := total_distance - distance_first_hour - distance_next_45_minutes
  let remaining_time := driving_time - (1 + next_45_minutes_time)
  remaining_distance / remaining_time

theorem johns_average_speed_remaining_duration : average_speed_remaining_duration = 90 := by
  sorry

end johns_average_speed_remaining_duration_l920_92027


namespace katie_speed_l920_92079

theorem katie_speed (eugene_speed : ℝ)
  (brianna_ratio : ℝ)
  (katie_ratio : ℝ)
  (h1 : eugene_speed = 4)
  (h2 : brianna_ratio = 2 / 3)
  (h3 : katie_ratio = 7 / 5) :
  katie_ratio * (brianna_ratio * eugene_speed) = 56 / 15 := 
by
  sorry

end katie_speed_l920_92079
