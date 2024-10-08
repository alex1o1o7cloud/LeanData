import Mathlib

namespace megan_earnings_l156_156558

-- Define the given conditions
def bead_necklaces : ℕ := 7
def gem_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 9

-- Define the total number of necklaces
def total_necklaces : ℕ := bead_necklaces + gem_necklaces

-- Define the total earnings
def total_earnings : ℕ := total_necklaces * cost_per_necklace

-- Prove that the total earnings are 90 dollars
theorem megan_earnings : total_earnings = 90 := by
  sorry

end megan_earnings_l156_156558


namespace sum_of_remainders_l156_156286

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : ((n % 4) + (n % 5) = 4) :=
sorry

end sum_of_remainders_l156_156286


namespace ellie_runs_8_miles_in_24_minutes_l156_156704

theorem ellie_runs_8_miles_in_24_minutes (time_max : ℝ) (distance_max : ℝ) 
  (time_ellie_fraction : ℝ) (distance_ellie : ℝ) (distance_ellie_final : ℝ)
  (h1 : distance_max = 6) 
  (h2 : time_max = 36) 
  (h3 : time_ellie_fraction = 1/3) 
  (h4 : distance_ellie = 4) 
  (h5 : distance_ellie_final = 8) :
  ((time_ellie_fraction * time_max) / distance_ellie) * distance_ellie_final = 24 :=
by
  sorry

end ellie_runs_8_miles_in_24_minutes_l156_156704


namespace find_fraction_l156_156225

theorem find_fraction :
  ∀ (t k : ℝ) (frac : ℝ),
    t = frac * (k - 32) →
    t = 20 → 
    k = 68 → 
    frac = 5 / 9 :=
by
  intro t k frac h_eq h_t h_k
  -- Start from the conditions and end up showing frac = 5/9
  sorry

end find_fraction_l156_156225


namespace books_difference_l156_156478

theorem books_difference (bobby_books : ℕ) (kristi_books : ℕ) (h1 : bobby_books = 142) (h2 : kristi_books = 78) : bobby_books - kristi_books = 64 :=
by {
  -- Placeholder for the proof
  sorry
}

end books_difference_l156_156478


namespace sum_of_powers_of_two_l156_156978

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 = 2^5 :=
by
  sorry

end sum_of_powers_of_two_l156_156978


namespace calculate_new_measure_l156_156025

noncomputable def equilateral_triangle_side_length : ℝ := 7.5

theorem calculate_new_measure :
  3 * (equilateral_triangle_side_length ^ 2) = 168.75 :=
by
  sorry

end calculate_new_measure_l156_156025


namespace total_supervisors_l156_156436

def buses : ℕ := 7
def supervisors_per_bus : ℕ := 3

theorem total_supervisors : buses * supervisors_per_bus = 21 := 
by
  have h : buses * supervisors_per_bus = 21 := by sorry
  exact h

end total_supervisors_l156_156436


namespace rhombus_diagonal_difference_l156_156167

theorem rhombus_diagonal_difference (a d : ℝ) (h_a_pos : a > 0) (h_d_pos : d > 0):
  (∃ (e f : ℝ), e > f ∧ e - f = d ∧ a^2 = (e/2)^2 + (f/2)^2) ↔ d < 2 * a :=
sorry

end rhombus_diagonal_difference_l156_156167


namespace alternating_sum_of_coefficients_l156_156554

noncomputable def polynomial_expansion (x : ℝ) : ℝ :=
  (2 * x + 1)^5

theorem alternating_sum_of_coefficients :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), polynomial_expansion x = 
    a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
    a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = -1 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h
  sorry

end alternating_sum_of_coefficients_l156_156554


namespace necessary_but_not_sufficient_l156_156033

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a + b > 4) ↔ (¬ (a > 2 ∧ b > 2)) ∧ ((a > 2 ∧ b > 2) → (a + b > 4)) :=
by
  sorry

end necessary_but_not_sufficient_l156_156033


namespace total_people_on_boats_l156_156928

-- Definitions based on the given conditions
def boats : Nat := 5
def people_per_boat : Nat := 3

-- Theorem statement to prove the total number of people on boats in the lake
theorem total_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end total_people_on_boats_l156_156928


namespace sum_of_three_equal_expressions_l156_156453

-- Definitions of variables and conditions
variables (a b c d e f g h i S : ℤ)
variable (ha : a = 4)
variable (hg : g = 13)
variable (hh : h = 6)
variable (heq1 : a + b + c + d = S)
variable (heq2 : d + e + f + g = S)
variable (heq3 : g + h + i = S)

-- Main statement we want to prove
theorem sum_of_three_equal_expressions : S = 19 + i :=
by
  -- substitution steps and equality reasoning would be carried out here
  sorry

end sum_of_three_equal_expressions_l156_156453


namespace digit_sum_26_l156_156797

theorem digit_sum_26 
  (A B C D E : ℕ)
  (h1 : 1 ≤ A ∧ A ≤ 9)
  (h2 : 0 ≤ B ∧ B ≤ 9)
  (h3 : 0 ≤ C ∧ C ≤ 9)
  (h4 : 0 ≤ D ∧ D ≤ 9)
  (h5 : 0 ≤ E ∧ E ≤ 9)
  (h6 : 100000 + 10000 * A + 1000 * B + 100 * C + 10 * D + E * 3 = 100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + 1):
  A + B + C + D + E = 26 
  := 
  by
    sorry

end digit_sum_26_l156_156797


namespace value_of_a_minus_b_l156_156556

theorem value_of_a_minus_b (a b : ℝ)
  (h1 : ∃ (x : ℝ), x = 3 ∧ (ax / (x - 1)) = 1)
  (h2 : ∀ (x : ℝ), (ax / (x - 1)) < 1 ↔ (x < b ∨ x > 3)) :
  a - b = -1 / 3 :=
by
  sorry

end value_of_a_minus_b_l156_156556


namespace find_t_l156_156230

def vector (α : Type) : Type := (α × α)

def dot_product (v1 v2 : vector ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def orthogonal (v1 v2 : vector ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_t (t : ℝ) :
  let a : vector ℝ := (1, -1)
  let b : vector ℝ := (2, t)
  orthogonal a b → t = 2 := by
  sorry

end find_t_l156_156230


namespace male_population_half_total_l156_156060

theorem male_population_half_total (total_population : ℕ) (segments : ℕ) (male_segment : ℕ) :
  total_population = 800 ∧ segments = 4 ∧ male_segment = 1 ∧ male_segment = segments / 2 →
  total_population / 2 = 400 :=
by
  intro h
  sorry

end male_population_half_total_l156_156060


namespace top_triangle_is_multiple_of_5_l156_156117

-- Definitions of the conditions given in the problem

def lower_left_triangle := 12
def lower_right_triangle := 3

-- Let a, b, c, d be the four remaining numbers in the bottom row
variables (a b c d : ℤ)

-- Conditions that the sums of triangles must be congruent to multiples of 5
def second_lowest_row : Prop :=
  (3 - a) % 5 = 0 ∧
  (-a - b) % 5 = 0 ∧
  (-b - c) % 5 = 0 ∧
  (-c - d) % 5 = 0 ∧
  (2 - d) % 5 = 0

def third_lowest_row : Prop :=
  (2 + 2*a + b) % 5 = 0 ∧
  (a + 2*b + c) % 5 = 0 ∧
  (b + 2*c + d) % 5 = 0 ∧
  (3 + c + 2*d) % 5 = 0

def fourth_lowest_row : Prop :=
  (3 + 2*a + 2*b - c) % 5 = 0 ∧
  (-a + 2*b + 2*c - d) % 5 = 0 ∧
  (2 - b + 2*c + 2*d) % 5 = 0

def second_highest_row : Prop :=
  (2 - a + b - c + d) % 5 = 0 ∧
  (3 + a - b + c - d) % 5 = 0

def top_triangle : Prop :=
  (2 - a + b - c + d + 3 + a - b + c - d) % 5 = 0

theorem top_triangle_is_multiple_of_5 (a b c d : ℤ) :
  second_lowest_row a b c d →
  third_lowest_row a b c d →
  fourth_lowest_row a b c d →
  second_highest_row a b c d →
  top_triangle a b c d →
  ∃ k : ℤ, (2 - a + b - c + d + 3 + a - b + c - d) = 5 * k :=
by sorry

end top_triangle_is_multiple_of_5_l156_156117


namespace find_value_of_a4_plus_a5_l156_156744

variables {S_n : ℕ → ℕ} {a_n : ℕ → ℕ} {d : ℤ} 

-- Conditions
def arithmetic_sequence_sum (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) (d : ℤ) : Prop :=
∀ n : ℕ, S_n n = n * a_n 1 + (n * (n - 1) / 2) * d

def a_3_S_3_condition (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) : Prop := 
a_n 3 = 3 ∧ S_n 3 = 3

-- Question
theorem find_value_of_a4_plus_a5 (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (d : ℤ):
  arithmetic_sequence_sum S_n a_n d →
  a_3_S_3_condition a_n S_n →
  a_n 4 + a_n 5 = 12 :=
by
  sorry

end find_value_of_a4_plus_a5_l156_156744


namespace partI_solution_partII_solution_l156_156125

-- Part (I)
theorem partI_solution (x : ℝ) (a : ℝ) (h : a = 5) : (|x + a| + |x - 2| > 9) ↔ (x < -6 ∨ x > 3) :=
by
  sorry

-- Part (II)
theorem partII_solution (a : ℝ) :
  (∀ x : ℝ, (|2*x - 1| ≤ 3) → (|x + a| + |x - 2| ≤ |x - 4|)) → (-1 ≤ a ∧ a ≤ 0) :=
by
  sorry

end partI_solution_partII_solution_l156_156125


namespace bucket_problem_l156_156400

theorem bucket_problem 
  (C : ℝ) -- original capacity of the bucket
  (N : ℕ) -- number of buckets required to fill the tank with the original bucket size
  (h : N * C = 25 * (2/5) * C) : 
  N = 10 :=
by
  sorry

end bucket_problem_l156_156400


namespace carter_students_received_grades_l156_156228

theorem carter_students_received_grades
  (students_thompson : ℕ)
  (a_thompson : ℕ)
  (remaining_students_thompson : ℕ)
  (b_thompson : ℕ)
  (students_carter : ℕ)
  (ratio_A_thompson : ℚ)
  (ratio_B_thompson : ℚ)
  (A_carter : ℕ)
  (B_carter : ℕ) :
  students_thompson = 20 →
  a_thompson = 12 →
  remaining_students_thompson = 8 →
  b_thompson = 5 →
  students_carter = 30 →
  ratio_A_thompson = (a_thompson : ℚ) / students_thompson →
  ratio_B_thompson = (b_thompson : ℚ) / remaining_students_thompson →
  A_carter = ratio_A_thompson * students_carter →
  B_carter = (b_thompson : ℚ) / remaining_students_thompson * (students_carter - A_carter) →
  A_carter = 18 ∧ B_carter = 8 := 
by 
  intros;
  sorry

end carter_students_received_grades_l156_156228


namespace find_b_l156_156102

variable (x : ℝ)

noncomputable def d : ℝ := 3

theorem find_b (b c : ℝ) :
  (7 * x^2 - 5 * x + 11 / 4) * (d * x^2 + b * x + c) = 21 * x^4 - 26 * x^3 + 34 * x^2 - 55 / 4 * x + 33 / 4 →
  b = -11 / 7 :=
by
  sorry

end find_b_l156_156102


namespace line_through_fixed_point_l156_156510

-- Define the arithmetic sequence condition
def arithmetic_sequence (k b : ℝ) : Prop :=
  k + b = -2

-- Define the line passing through a fixed point
def line_passes_through (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ (x = 1 ∧ y = -2)

-- The theorem stating the main problem
theorem line_through_fixed_point (k b : ℝ) (h : arithmetic_sequence k b) : line_passes_through k b :=
  sorry

end line_through_fixed_point_l156_156510


namespace sequence_a8_equals_neg2_l156_156407

theorem sequence_a8_equals_neg2 (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, a n * a (n + 1) = -2) 
  : a 8 = -2 :=
sorry

end sequence_a8_equals_neg2_l156_156407


namespace porter_monthly_earnings_l156_156425

-- Definitions
def regular_daily_rate : ℝ := 8
def days_per_week : ℕ := 5
def overtime_rate : ℝ := 1.5
def tax_deduction_rate : ℝ := 0.10
def insurance_deduction_rate : ℝ := 0.05
def weeks_per_month : ℕ := 4

-- Intermediate Calculations
def regular_weekly_earnings := regular_daily_rate * days_per_week
def extra_day_rate := regular_daily_rate * overtime_rate
def total_weekly_earnings := regular_weekly_earnings + extra_day_rate
def total_monthly_earnings_before_deductions := total_weekly_earnings * weeks_per_month

-- Deductions
def tax_deduction := total_monthly_earnings_before_deductions * tax_deduction_rate
def insurance_deduction := total_monthly_earnings_before_deductions * insurance_deduction_rate
def total_deductions := tax_deduction + insurance_deduction
def total_monthly_earnings_after_deductions := total_monthly_earnings_before_deductions - total_deductions

-- Theorem Statement
theorem porter_monthly_earnings : total_monthly_earnings_after_deductions = 176.80 := by
  sorry

end porter_monthly_earnings_l156_156425


namespace popsicle_sticks_left_correct_l156_156859

noncomputable def popsicle_sticks_left (initial : ℝ) (given : ℝ) : ℝ :=
  initial - given

theorem popsicle_sticks_left_correct :
  popsicle_sticks_left 63 50 = 13 :=
by
  sorry

end popsicle_sticks_left_correct_l156_156859


namespace arithmetic_progression_even_terms_l156_156739

theorem arithmetic_progression_even_terms (a d n : ℕ) (h_even : n % 2 = 0)
  (h_last_first_diff : (n - 1) * d = 16)
  (h_sum_odd : n * (a + (n - 2) * d / 2) = 81)
  (h_sum_even : n * (a + d + (n - 2) * d / 2) = 75) :
  n = 8 :=
by sorry

end arithmetic_progression_even_terms_l156_156739


namespace slices_remaining_l156_156587

theorem slices_remaining (large_pizza_slices : ℕ) (xl_pizza_slices : ℕ) (large_pizza_ordered : ℕ) (xl_pizza_ordered : ℕ) (mary_eats_large : ℕ) (mary_eats_xl : ℕ) :
  large_pizza_slices = 8 →
  xl_pizza_slices = 12 →
  large_pizza_ordered = 1 →
  xl_pizza_ordered = 1 →
  mary_eats_large = 7 →
  mary_eats_xl = 3 →
  (large_pizza_slices * large_pizza_ordered - mary_eats_large + xl_pizza_slices * xl_pizza_ordered - mary_eats_xl) = 10 := 
by
  intros
  sorry

end slices_remaining_l156_156587


namespace subset_complU_N_l156_156993

variable {U : Type} {M N : Set U}

-- Given conditions
axiom non_empty_M : ∃ x, x ∈ M
axiom non_empty_N : ∃ y, y ∈ N
axiom subset_complU_M : N ⊆ Mᶜ

-- Prove the statement that M is a subset of the complement of N
theorem subset_complU_N : M ⊆ Nᶜ := by
  sorry

end subset_complU_N_l156_156993


namespace translated_line_eqn_l156_156621

theorem translated_line_eqn
  (c : ℝ) :
  ∀ (y_eqn : ℝ → ℝ), 
    (∀ x, y_eqn x = 2 * x + 1) →
    (∀ x, (y_eqn (x - 2) - 3) = (2 * x - 6)) :=
by
  sorry

end translated_line_eqn_l156_156621


namespace chemistry_class_students_l156_156029

theorem chemistry_class_students (total_students both_classes biology_class only_chemistry_class : ℕ)
    (h1: total_students = 100)
    (h2 : both_classes = 10)
    (h3 : total_students = biology_class + only_chemistry_class + both_classes)
    (h4 : only_chemistry_class = 4 * (biology_class + both_classes)) : 
    only_chemistry_class = 80 :=
by
  sorry

end chemistry_class_students_l156_156029


namespace find_d_l156_156590

theorem find_d (d : ℝ) (h : ∃ (x y : ℝ), 3 * x + 5 * y + d = 0 ∧ x = -d / 3 ∧ y = -d / 5 ∧ -d / 3 + (-d / 5) = 15) : d = -225 / 8 :=
by 
  sorry

end find_d_l156_156590


namespace max_matching_pairs_l156_156539

theorem max_matching_pairs 
  (total_pairs : ℕ := 23) 
  (total_colors : ℕ := 6) 
  (total_sizes : ℕ := 3) 
  (lost_shoes : ℕ := 9)
  (shoes_per_pair : ℕ := 2) 
  (total_shoes := total_pairs * shoes_per_pair) 
  (remaining_shoes := total_shoes - lost_shoes) :
  ∃ max_pairs : ℕ, max_pairs = total_pairs - lost_shoes / shoes_per_pair :=
sorry

end max_matching_pairs_l156_156539


namespace min_bottles_needed_l156_156267

theorem min_bottles_needed (fluid_ounces_needed : ℝ) (bottle_size_ml : ℝ) (conversion_factor : ℝ) :
  fluid_ounces_needed = 60 ∧ bottle_size_ml = 250 ∧ conversion_factor = 33.8 →
  ∃ (n : ℕ), n = 8 ∧ (fluid_ounces_needed / conversion_factor * 1000 / bottle_size_ml) <= ↑n :=
by
  sorry

end min_bottles_needed_l156_156267


namespace find_other_number_l156_156247

def a : ℝ := 0.5
def d : ℝ := 0.16666666666666669
def b : ℝ := 0.3333333333333333

theorem find_other_number : a - d = b := by
  sorry

end find_other_number_l156_156247


namespace number_base_addition_l156_156758

theorem number_base_addition (A B : ℕ) (h1: A = 2 * B) (h2: 2 * B^2 + 2 * B + 4 + 10 * B + 5 = (3 * B)^2 + 3 * (3 * B) + 4) : 
  A + B = 9 := 
by 
  sorry

end number_base_addition_l156_156758


namespace average_last_three_l156_156434

/-- The average of the last three numbers is 65, given that the average of six numbers is 60
  and the average of the first three numbers is 55. -/
theorem average_last_three (a b c d e f : ℝ) (h1 : (a + b + c + d + e + f) / 6 = 60) (h2 : (a + b + c) / 3 = 55) :
  (d + e + f) / 3 = 65 :=
by
  sorry

end average_last_three_l156_156434


namespace problem_statement_l156_156068

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (-1, 1)

-- Define vector addition
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

-- Define dot product
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define perpendicular condition
def perp (u v : ℝ × ℝ) : Prop := dot_prod u v = 0

theorem problem_statement : perp (vec_add a b) a :=
by
  sorry

end problem_statement_l156_156068


namespace dave_final_tickets_l156_156236

variable (initial_tickets_set1_won : ℕ) (initial_tickets_set1_lost : ℕ)
variable (initial_tickets_set2_won : ℕ) (initial_tickets_set2_lost : ℕ)
variable (multiplier_set3 : ℕ)
variable (initial_tickets_set3_lost : ℕ)
variable (used_tickets : ℕ)
variable (additional_tickets : ℕ)

theorem dave_final_tickets :
  let net_gain_set1 := initial_tickets_set1_won - initial_tickets_set1_lost
  let net_gain_set2 := initial_tickets_set2_won - initial_tickets_set2_lost
  let net_gain_set3 := multiplier_set3 * net_gain_set1 - initial_tickets_set3_lost
  let total_tickets_after_sets := net_gain_set1 + net_gain_set2 + net_gain_set3
  let tickets_after_buying := total_tickets_after_sets - used_tickets
  let final_tickets := tickets_after_buying + additional_tickets
  initial_tickets_set1_won = 14 →
  initial_tickets_set1_lost = 2 →
  initial_tickets_set2_won = 8 →
  initial_tickets_set2_lost = 5 →
  multiplier_set3 = 3 →
  initial_tickets_set3_lost = 15 →
  used_tickets = 25 →
  additional_tickets = 7 →
  final_tickets = 18 :=
by
  intros
  sorry

end dave_final_tickets_l156_156236


namespace required_number_of_shirts_l156_156900

/-
In a shop, there is a sale of clothes. Every shirt costs $5, every hat $4, and a pair of jeans $10.
You need to pay $51 for a certain number of shirts, two pairs of jeans, and four hats.
Prove that the number of shirts you need to buy is 3.
-/

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_payment : ℕ := 51
def number_of_jeans : ℕ := 2
def number_of_hats : ℕ := 4

theorem required_number_of_shirts (S : ℕ) (h : 5 * S + 2 * jeans_cost + 4 * hat_cost = total_payment) : S = 3 :=
by
  -- This statement asserts that given the defined conditions, the number of shirts that satisfies the equation is 3.
  sorry

end required_number_of_shirts_l156_156900


namespace calculate_value_of_squares_difference_l156_156890

theorem calculate_value_of_squares_difference : 305^2 - 301^2 = 2424 :=
by {
  sorry
}

end calculate_value_of_squares_difference_l156_156890


namespace find_intersection_l156_156089

def A : Set ℝ := {x | abs (x + 1) = x + 1}

def B : Set ℝ := {x | x^2 + x < 0}

def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem find_intersection : intersection A B = {x | -1 < x ∧ x < 0} :=
by
  sorry

end find_intersection_l156_156089


namespace trees_died_l156_156814

theorem trees_died 
  (original_trees : ℕ) 
  (cut_trees : ℕ) 
  (remaining_trees : ℕ) 
  (died_trees : ℕ)
  (h1 : original_trees = 86)
  (h2 : cut_trees = 23)
  (h3 : remaining_trees = 48)
  (h4 : original_trees - died_trees - cut_trees = remaining_trees) : 
  died_trees = 15 :=
by
  sorry

end trees_died_l156_156814


namespace gu_xian_expression_right_triangle_l156_156438

-- Definitions for Part 1
def gu (n : ℕ) (h : n ≥ 3 ∧ n % 2 = 1) : ℕ := (n^2 - 1) / 2
def xian (n : ℕ) (h : n ≥ 3 ∧ n % 2 = 1) : ℕ := (n^2 + 1) / 2

-- Definitions for Part 2
def a (m : ℕ) (h : m > 1) : ℕ := m^2 - 1
def b (m : ℕ) (h : m > 1) : ℕ := 2 * m
def c (m : ℕ) (h : m > 1) : ℕ := m^2 + 1

-- Proof statement for Part 1
theorem gu_xian_expression (n : ℕ) (hn : n ≥ 3 ∧ n % 2 = 1) :
  gu n hn = (n^2 - 1) / 2 ∧ xian n hn = (n^2 + 1) / 2 :=
sorry

-- Proof statement for Part 2
theorem right_triangle (m : ℕ) (hm: m > 1) :
  (a m hm)^2 + (b m hm)^2 = (c m hm)^2 :=
sorry

end gu_xian_expression_right_triangle_l156_156438


namespace positive_difference_of_perimeters_l156_156795

theorem positive_difference_of_perimeters :
  let length1 := 6
  let width1 := 1
  let length2 := 3
  let width2 := 2
  let perimeter1 := 2 * (length1 + width1)
  let perimeter2 := 2 * (length2 + width2)
  (perimeter1 - perimeter2) = 4 :=
by
  let length1 := 6
  let width1 := 1
  let length2 := 3
  let width2 := 2
  let perimeter1 := 2 * (length1 + width1)
  let perimeter2 := 2 * (length2 + width2)
  show (perimeter1 - perimeter2) = 4
  sorry

end positive_difference_of_perimeters_l156_156795


namespace no_solution_for_inequality_l156_156672

theorem no_solution_for_inequality (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end no_solution_for_inequality_l156_156672


namespace fraction_of_raisins_l156_156105

-- Define the cost of a single pound of raisins
variables (R : ℝ) -- R represents the cost of one pound of raisins

-- Conditions
def mixed_raisins := 5 -- Chris mixed 5 pounds of raisins
def mixed_nuts := 4 -- with 4 pounds of nuts
def nuts_cost_ratio := 3 -- A pound of nuts costs 3 times as much as a pound of raisins

-- Statement to prove
theorem fraction_of_raisins
  (R_pos : R > 0) : (5 * R) / ((5 * R) + (4 * (3 * R))) = 5 / 17 :=
by
  -- The proof is omitted here.
  sorry

end fraction_of_raisins_l156_156105


namespace proposition_C_correct_l156_156129

theorem proposition_C_correct (a b c : ℝ) (h : a * c ^ 2 > b * c ^ 2) : a > b :=
sorry

end proposition_C_correct_l156_156129


namespace mabel_tomatoes_l156_156947

theorem mabel_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 := by
  sorry

end mabel_tomatoes_l156_156947


namespace average_weight_increase_l156_156628

theorem average_weight_increase
  (initial_weight replaced_weight : ℝ)
  (num_persons : ℕ)
  (avg_increase : ℝ)
  (h₁ : num_persons = 5)
  (h₂ : replaced_weight = 65)
  (h₃ : avg_increase = 1.5)
  (total_increase : ℝ)
  (new_weight : ℝ)
  (h₄ : total_increase = num_persons * avg_increase)
  (h₅ : total_increase = new_weight - replaced_weight) :
  new_weight = 72.5 :=
by
  sorry

end average_weight_increase_l156_156628


namespace find_r_l156_156404

theorem find_r (a b m p r : ℝ) (h1 : a * b = 4)
  (h2 : ∃ (q w : ℝ), (a + 2 / b = q ∧ b + 2 / a = w) ∧ q * w = r) :
  r = 9 :=
sorry

end find_r_l156_156404


namespace expected_length_after_2012_repetitions_l156_156643

noncomputable def expected_length_remaining (n : ℕ) := (11/18 : ℚ)^n

theorem expected_length_after_2012_repetitions :
  expected_length_remaining 2012 = (11 / 18 : ℚ) ^ 2012 :=
by
  sorry

end expected_length_after_2012_repetitions_l156_156643


namespace sum_of_squares_of_roots_l156_156820

theorem sum_of_squares_of_roots : 
  (∃ (a b c d : ℝ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^4 - 15 * x^2 + 56 = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d)) ∧
    (a^2 + b^2 + c^2 + d^2 = 30)) :=
sorry

end sum_of_squares_of_roots_l156_156820


namespace purple_chip_value_l156_156817

theorem purple_chip_value 
  (x : ℕ)
  (blue_chip_value : 1 = 1)
  (green_chip_value : 5 = 5)
  (red_chip_value : 11 = 11)
  (purple_chip_condition1 : x > 5)
  (purple_chip_condition2 : x < 11)
  (product_of_points : ∃ b g p r, (b = 1 ∨ b = 1) ∧ (g = 5 ∨ g = 5) ∧ (p = x ∨ p = x) ∧ (r = 11 ∨ r = 11) ∧ b * g * p * r = 28160) : 
  x = 7 :=
sorry

end purple_chip_value_l156_156817


namespace problem_1_l156_156412

theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, |2 * x + 1| + |x - 2| ≥ a ^ 2 - a + (1 / 2)) ↔ -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end problem_1_l156_156412


namespace ratio_of_x_to_y_l156_156956

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
sorry

end ratio_of_x_to_y_l156_156956


namespace part1_part2_l156_156979

noncomputable def f (x a : ℝ) : ℝ := x^2 - (a+1)*x + a

theorem part1 (a x : ℝ) :
  (a < 1 ∧ f x a < 0 ↔ a < x ∧ x < 1) ∧
  (a = 1 ∧ ¬(f x a < 0)) ∧
  (a > 1 ∧ f x a < 0 ↔ 1 < x ∧ x < a) :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, 1 < x → f x a ≥ -1) → a ≤ 3 :=
sorry

end part1_part2_l156_156979


namespace book_pages_total_l156_156050

-- Define the conditions
def pagesPerNight : ℝ := 120.0
def nights : ℝ := 10.0

-- State the theorem to prove
theorem book_pages_total : pagesPerNight * nights = 1200.0 := by
  sorry

end book_pages_total_l156_156050


namespace total_rowing_time_l156_156016

theorem total_rowing_time (s_b : ℕ) (s_s : ℕ) (d : ℕ) : 
  s_b = 9 → s_s = 6 → d = 170 → 
  (d / (s_b + s_s) + d / (s_b - s_s)) = 68 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_rowing_time_l156_156016


namespace fencing_rate_correct_l156_156687

noncomputable def rate_per_meter (d : ℝ) (cost : ℝ) : ℝ :=
  cost / (Real.pi * d)

theorem fencing_rate_correct : rate_per_meter 26 122.52211349000194 = 1.5 := by
  sorry

end fencing_rate_correct_l156_156687


namespace bottom_level_legos_l156_156719

theorem bottom_level_legos
  (x : ℕ)
  (h : x^2 + (x - 1)^2 + (x - 2)^2 = 110) :
  x = 7 :=
by {
  sorry
}

end bottom_level_legos_l156_156719


namespace integer_solution_l156_156648

theorem integer_solution (x : ℕ) (h : (4 * x)^2 - 2 * x = 3178) : x = 226 :=
by
  sorry

end integer_solution_l156_156648


namespace quadratic_intersects_x_axis_iff_l156_156424

theorem quadratic_intersects_x_axis_iff (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x - m = 0) ↔ m ≥ -1 := 
by
  sorry

end quadratic_intersects_x_axis_iff_l156_156424


namespace product_of_differing_inputs_equal_l156_156042

theorem product_of_differing_inputs_equal (a b : ℝ) (h₁ : a ≠ b)
(h₂ : |Real.log a - (1 / 2)| = |Real.log b - (1 / 2)|) : a * b = Real.exp 1 :=
sorry

end product_of_differing_inputs_equal_l156_156042


namespace cost_of_each_burger_l156_156475

theorem cost_of_each_burger (purchases_per_day : ℕ) (total_days : ℕ) (total_amount_spent : ℕ)
  (h1 : purchases_per_day = 4) (h2 : total_days = 30) (h3 : total_amount_spent = 1560) : 
  total_amount_spent / (purchases_per_day * total_days) = 13 :=
by
  subst h1
  subst h2
  subst h3
  sorry

end cost_of_each_burger_l156_156475


namespace least_number_to_add_1054_23_l156_156683

def least_number_to_add (n k : ℕ) : ℕ :=
  let remainder := n % k
  if remainder = 0 then 0 else k - remainder

theorem least_number_to_add_1054_23 : least_number_to_add 1054 23 = 4 :=
by
  -- This is a placeholder for the actual proof
  sorry

end least_number_to_add_1054_23_l156_156683


namespace problem_log_inequality_l156_156401

noncomputable def f (x m : ℝ) := x - |x + 2| - |x - 3| - m

theorem problem (m : ℝ) (h1 : ∀ x : ℝ, (1 / m) - 4 ≥ f x m) :
  m > 0 :=
sorry

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_inequality (m : ℝ) (h2 : m > 0) :
  log_base (m + 1) (m + 2) > log_base (m + 2) (m + 3) :=
sorry

end problem_log_inequality_l156_156401


namespace solve_problem_l156_156870

-- Conditions from the problem
def is_prime (p : ℕ) : Prop := Nat.Prime p

def satisfies_conditions (n p : ℕ) : Prop := 
  (p > 1) ∧ is_prime p ∧ (n > 0) ∧ (n ≤ 2 * p)

-- Main proof statement
theorem solve_problem (n p : ℕ) (h1 : satisfies_conditions n p)
    (h2 : (p - 1) ^ n + 1 ∣ n ^ (p - 1)) :
    (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1 ∧ is_prime p) :=
sorry

end solve_problem_l156_156870


namespace sellable_fruit_l156_156340

theorem sellable_fruit :
  let total_oranges := 30 * 300
  let total_damaged_oranges := total_oranges * 10 / 100
  let sellable_oranges := total_oranges - total_damaged_oranges

  let total_nectarines := 45 * 80
  let nectarines_taken := 5 * 20
  let sellable_nectarines := total_nectarines - nectarines_taken

  let total_apples := 20 * 120
  let bad_apples := 50
  let sellable_apples := total_apples - bad_apples

  sellable_oranges + sellable_nectarines + sellable_apples = 13950 :=
by
  sorry

end sellable_fruit_l156_156340


namespace smallest_integer_l156_156782

theorem smallest_integer {x y z : ℕ} (h1 : 2*y = x) (h2 : 3*y = z) (h3 : x + y + z = 60) : y = 6 :=
by
  sorry

end smallest_integer_l156_156782


namespace lucy_groceries_total_l156_156798

theorem lucy_groceries_total (cookies noodles : ℕ) (h1 : cookies = 12) (h2 : noodles = 16) : cookies + noodles = 28 :=
by
  sorry

end lucy_groceries_total_l156_156798


namespace a8_eq_64_l156_156141

variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

axiom a1_eq_2 : a 1 = 2
axiom S_recurrence : ∀ (n : ℕ), S (n + 1) = 2 * S n - 1

theorem a8_eq_64 : a 8 = 64 := 
by
sorry

end a8_eq_64_l156_156141


namespace repeating_prime_exists_l156_156098

open Nat

theorem repeating_prime_exists (p : Fin 2021 → ℕ) 
  (prime_seq : ∀ i : Fin 2021, Nat.Prime (p i))
  (diff_condition : ∀ i : Fin 2019, (p (i + 1) - p i = 6 ∨ p (i + 1) - p i = 12) ∧ (p (i + 2) - p (i + 1) = 6 ∨ p (i + 2) - p (i + 1) = 12)) : 
  ∃ i j : Fin 2021, i ≠ j ∧ p i = p j := by
  sorry

end repeating_prime_exists_l156_156098


namespace insufficient_data_to_compare_l156_156417

variable (M P O : ℝ)

theorem insufficient_data_to_compare (h1 : M < P) (h2 : O > M) : ¬(P > O) ∧ ¬(O > P) :=
sorry

end insufficient_data_to_compare_l156_156417


namespace intersect_at_one_point_l156_156682

-- Definitions of points and circles
variable (Point : Type)
variable (Circle : Type)
variable (A : Point)
variable (C1 C2 C3 C4 : Circle)

-- Definition of intersection points
variable (B12 B13 B14 B23 B24 B34 : Point)

-- Note: Assumptions around the geometry structure axioms need to be defined
-- Assuming we have a function that checks if three points are collinear:
variable (are_collinear : Point → Point → Point → Prop)
-- Assuming we have a function that checks if a point is part of a circle:
variable (on_circle : Point → Circle → Prop)

-- Axioms related to the conditions
axiom collinear_B12_B34_B (hC1 : on_circle B12 C1) (hC2 : on_circle B12 C2) (hC3 : on_circle B34 C3) (hC4 : on_circle B34 C4) : 
  ∃ P : Point, are_collinear B12 P B34 

axiom collinear_B13_B24_B (hC1 : on_circle B13 C1) (hC2 : on_circle B13 C3) (hC3 : on_circle B24 C2) (hC4 : on_circle B24 C4) : 
  ∃ P : Point, are_collinear B13 P B24 

axiom collinear_B14_B23_B (hC1 : on_circle B14 C1) (hC2 : on_circle B14 C4) (hC3 : on_circle B23 C2) (hC4 : on_circle B23 C3) : 
  ∃ P : Point, are_collinear B14 P B23 

-- The theorem to be proved
theorem intersect_at_one_point :
  ∃ P : Point, 
    are_collinear B12 P B34 ∧ are_collinear B13 P B24 ∧ are_collinear B14 P B23 := 
sorry

end intersect_at_one_point_l156_156682


namespace percent_of_rs_600_l156_156430

theorem percent_of_rs_600 : (600 * 0.25 = 150) :=
by
  sorry

end percent_of_rs_600_l156_156430


namespace average_temperature_l156_156072

theorem average_temperature (T_NY T_Miami T_SD : ℝ) (h1 : T_NY = 80) (h2 : T_Miami = T_NY + 10) (h3 : T_SD = T_Miami + 25) :
  (T_NY + T_Miami + T_SD) / 3 = 95 :=
by
  sorry

end average_temperature_l156_156072


namespace central_angle_measure_l156_156335

theorem central_angle_measure (α r : ℝ) (h1 : α * r = 2) (h2 : 1/2 * α * r^2 = 2) : α = 1 := 
sorry

end central_angle_measure_l156_156335


namespace stickers_started_with_l156_156356

-- Definitions for the conditions
def stickers_given (Emily : ℕ) : Prop := Emily = 7
def stickers_ended_with (Willie_end : ℕ) : Prop := Willie_end = 43

-- The main proof statement
theorem stickers_started_with (Willie_start : ℕ) :
  stickers_given 7 →
  stickers_ended_with 43 →
  Willie_start = 43 - 7 :=
by
  intros h₁ h₂
  sorry

end stickers_started_with_l156_156356


namespace seq1_general_formula_seq2_general_formula_l156_156307

-- Sequence (1): Initial condition and recurrence relation
def seq1 (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + (2 * n - 1)

-- Proving the general formula for sequence (1)
theorem seq1_general_formula (a : ℕ → ℕ) (n : ℕ) (h : seq1 a) :
  a n = (n - 1) ^ 2 :=
sorry

-- Sequence (2): Initial condition and recurrence relation
def seq2 (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = 3 * a n

-- Proving the general formula for sequence (2)
theorem seq2_general_formula (a : ℕ → ℕ) (n : ℕ) (h : seq2 a) :
  a n = 3 ^ n :=
sorry

end seq1_general_formula_seq2_general_formula_l156_156307


namespace max_happy_times_l156_156680

theorem max_happy_times (weights : Fin 2021 → ℝ) (unique_mass : Function.Injective weights) : 
  ∃ max_happy : Nat, max_happy = 673 :=
by
  sorry

end max_happy_times_l156_156680


namespace geometric_sequence_arith_condition_l156_156850

-- Definitions of geometric sequence and arithmetic sequence condition
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions: \( \{a_n\} \) is a geometric sequence with \( a_2 \), \( \frac{1}{2}a_3 \), \( a_1 \) forming an arithmetic sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n, a (n + 1) = q * a n

def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop := a 2 = (1 / 2) * a 3 + a 1

-- Final theorem to prove
theorem geometric_sequence_arith_condition (hq : q^2 - q - 1 = 0) 
  (hgeo : is_geometric_sequence a q) 
  (harith : arithmetic_sequence_condition a) : 
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end geometric_sequence_arith_condition_l156_156850


namespace oranges_sold_in_the_morning_eq_30_l156_156192

variable (O : ℝ)  -- Denote the number of oranges Wendy sold in the morning

-- Conditions as assumptions
def price_per_apple : ℝ := 1.5
def price_per_orange : ℝ := 1
def morning_apples_sold : ℝ := 40
def afternoon_apples_sold : ℝ := 50
def afternoon_oranges_sold : ℝ := 40
def total_sales_for_day : ℝ := 205

-- Prove that O, satisfying the given conditions, equals 30
theorem oranges_sold_in_the_morning_eq_30 (h : 
    (morning_apples_sold * price_per_apple) +
    (O * price_per_orange) +
    (afternoon_apples_sold * price_per_apple) +
    (afternoon_oranges_sold * price_per_orange) = 
    total_sales_for_day
  ) : O = 30 :=
by
  sorry

end oranges_sold_in_the_morning_eq_30_l156_156192


namespace area_of_right_triangle_l156_156005

theorem area_of_right_triangle (hypotenuse : ℝ) (angle : ℝ) (h_hypotenuse : hypotenuse = 10) (h_angle : angle = 45) :
  (1 / 2) * (5 * Real.sqrt 2) * (5 * Real.sqrt 2) = 25 :=
by
  -- Proof goes here
  sorry

end area_of_right_triangle_l156_156005


namespace alcohol_mix_problem_l156_156731

theorem alcohol_mix_problem
  (x_volume : ℕ) (y_volume : ℕ)
  (x_percentage : ℝ) (y_percentage : ℝ)
  (target_percentage : ℝ)
  (x_volume_eq : x_volume = 200)
  (x_percentage_eq : x_percentage = 0.10)
  (y_percentage_eq : y_percentage = 0.30)
  (target_percentage_eq : target_percentage = 0.14)
  (y_solution : ℝ)
  (h : y_volume = 50) :
  (20 + 0.3 * y_solution) / (200 + y_solution) = target_percentage := by sorry

end alcohol_mix_problem_l156_156731


namespace find_c_l156_156742

theorem find_c (c : ℝ) : (∃ a : ℝ, (x : ℝ) → (x^2 + 80*x + c = (x + a)^2)) → (c = 1600) := by
  sorry

end find_c_l156_156742


namespace calculate_f_f_2_l156_156939

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 3 * x ^ 2 - 4
else if x = 0 then 2
else -1

theorem calculate_f_f_2 : f (f 2) = 188 :=
by
  sorry

end calculate_f_f_2_l156_156939


namespace RectangleAreaDiagonalk_l156_156700

theorem RectangleAreaDiagonalk {length width : ℝ} {d : ℝ}
  (h_ratio : length / width = 5 / 2)
  (h_perimeter : 2 * (length + width) = 42)
  (h_diagonal : d = Real.sqrt (length^2 + width^2))
  : (∃ k, k = 10 / 29 ∧ ∀ A, A = k * d^2) :=
by {
  sorry
}

end RectangleAreaDiagonalk_l156_156700


namespace total_initial_amounts_l156_156583

theorem total_initial_amounts :
  ∃ (a j t : ℝ), a = 50 ∧ t = 50 ∧ (50 + j + 50 = 187.5) :=
sorry

end total_initial_amounts_l156_156583


namespace solution_set_of_inequality_l156_156143

theorem solution_set_of_inequality :
  { x : ℝ | |1 - 2 * x| < 3 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_of_inequality_l156_156143


namespace average_book_width_is_3_point_9375_l156_156109

def book_widths : List ℚ := [3, 4, 3/4, 1.5, 7, 2, 5.25, 8]
def number_of_books : ℚ := 8
def total_width : ℚ := List.sum book_widths
def average_width : ℚ := total_width / number_of_books

theorem average_book_width_is_3_point_9375 :
  average_width = 3.9375 := by
  sorry

end average_book_width_is_3_point_9375_l156_156109


namespace imag_part_z_l156_156383

theorem imag_part_z {z : ℂ} (h : i * (z - 3) = -1 + 3 * i) : z.im = 1 :=
sorry

end imag_part_z_l156_156383


namespace find_x_plus_y_l156_156904

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := 
sorry

end find_x_plus_y_l156_156904


namespace area_is_300_l156_156512

variable (l w : ℝ) -- Length and Width of the playground

-- Conditions
def condition1 : Prop := 2 * l + 2 * w = 80
def condition2 : Prop := l = 3 * w

-- Question and Answer
def area_of_playground : ℝ := l * w

theorem area_is_300 (h1 : condition1 l w) (h2 : condition2 l w) : area_of_playground l w = 300 := 
by
  sorry

end area_is_300_l156_156512


namespace production_relationship_l156_156148

noncomputable def production_function (a : ℕ) (p : ℝ) (x : ℕ) : ℝ := a * (1 + p / 100)^x

theorem production_relationship (a : ℕ) (p : ℝ) (m : ℕ) (x : ℕ) (hx : 0 ≤ x ∧ x ≤ m) :
  production_function a p x = a * (1 + p / 100)^x := by
  sorry

end production_relationship_l156_156148


namespace correct_speed_to_reach_on_time_l156_156903

theorem correct_speed_to_reach_on_time
  (d : ℝ)
  (t : ℝ)
  (h1 : d = 50 * (t + 1 / 12))
  (h2 : d = 70 * (t - 1 / 12)) :
  d / t = 58 := 
by
  sorry

end correct_speed_to_reach_on_time_l156_156903


namespace f_value_plus_deriv_l156_156667

noncomputable def f : ℝ → ℝ := sorry

-- Define the function f and its derivative at x = 1
axiom f_deriv_at_1 : deriv f 1 = 1 / 2

-- Define the value of the function f at x = 1
axiom f_value_at_1 : f 1 = 5 / 2

-- Prove that f(1) + f'(1) = 3
theorem f_value_plus_deriv : f 1 + deriv f 1 = 3 :=
by
  rw [f_value_at_1, f_deriv_at_1]
  norm_num

end f_value_plus_deriv_l156_156667


namespace book_pages_l156_156373

theorem book_pages (P D : ℕ) 
  (h1 : P = 23 * D + 9) 
  (h2 : ∃ D, P = 23 * (D + 1) - 14) : 
  P = 32 :=
by sorry

end book_pages_l156_156373


namespace bounds_for_f3_l156_156881

variable (a c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 - c

theorem bounds_for_f3 (h1 : -4 ≤ f a c 1 ∧ f a c 1 ≤ -1)
                      (h2 : -1 ≤ f a c 2 ∧ f a c 2 ≤ 5) :
  -1 ≤ f a c 3 ∧ f a c 3 ≤ 20 := 
sorry

end bounds_for_f3_l156_156881


namespace log15_12_eq_l156_156385

-- Goal: Define the constants and statement per the identified conditions and goal
variable (a b : ℝ)
#check Real.log
#check Real.logb

-- Math conditions
def lg2_eq_a := Real.log 2 = a
def lg3_eq_b := Real.log 3 = b

-- Math proof problem statement
theorem log15_12_eq : lg2_eq_a a → lg3_eq_b b → Real.logb 15 12 = (2 * a + b) / (1 - a + b) :=
by intros h1 h2; sorry

end log15_12_eq_l156_156385


namespace initial_investment_calculation_l156_156369

theorem initial_investment_calculation
  (x : ℝ)  -- initial investment at 5% per annum
  (h₁ : x * 0.05 + 4000 * 0.08 = (x + 4000) * 0.06) :
  x = 8000 :=
by
  -- skip the proof
  sorry

end initial_investment_calculation_l156_156369


namespace cuboid_surface_area_cuboid_volume_not_unique_l156_156363

theorem cuboid_surface_area
    (a b c p q : ℝ)
    (h1 : a + b + c = p)
    (h2 : a^2 + b^2 + c^2 = q^2) :
    2 * (a * b + b * c + a * c) = p^2 - q^2 :=
by
  sorry

theorem cuboid_volume_not_unique
    (a b c p q v1 v2 : ℝ)
    (h1 : a + b + c = p)
    (h2 : a^2 + b^2 + c^2 = q^2)
    : ¬ (∀ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ), 
          a₁ + b₁ + c₁ = p ∧ a₁^2 + b₁^2 + c₁^2 = q^2 →
          a₂ + b₂ + c₂ = p ∧ a₂^2 + b₂^2 + c₂^2 = q^2 →
          (a₁ * b₁ * c₁ = a₂ * b₂ * c₂)) :=
by
  -- Provide counterexamples (4, 4, 7) and (3, 6, 6) for p = 15, q = 9
  sorry

end cuboid_surface_area_cuboid_volume_not_unique_l156_156363


namespace polynomials_common_zero_k_l156_156960

theorem polynomials_common_zero_k
  (k : ℝ) :
  (∃ x : ℝ, (1988 * x^2 + k * x + 8891 = 0) ∧ (8891 * x^2 + k * x + 1988 = 0)) ↔ (k = 10879 ∨ k = -10879) :=
sorry

end polynomials_common_zero_k_l156_156960


namespace solve_linear_equation_l156_156618

theorem solve_linear_equation (x : ℝ) (h : 2 * x - 1 = 1) : x = 1 :=
sorry

end solve_linear_equation_l156_156618


namespace total_dots_on_left_faces_l156_156258

-- Define the number of dots on the faces A, B, C, and D
def d_A : ℕ := 3
def d_B : ℕ := 5
def d_C : ℕ := 6
def d_D : ℕ := 5

-- The statement we need to prove
theorem total_dots_on_left_faces : d_A + d_B + d_C + d_D = 19 := by
  sorry

end total_dots_on_left_faces_l156_156258


namespace sum_gcf_lcm_36_56_84_l156_156898

def gcf (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem sum_gcf_lcm_36_56_84 :
  let gcf_36_56_84 := gcf 36 56 84
  let lcm_36_56_84 := lcm 36 56 84
  gcf_36_56_84 + lcm_36_56_84 = 516 :=
by
  let gcf_36_56_84 := gcf 36 56 84
  let lcm_36_56_84 := lcm 36 56 84
  show gcf_36_56_84 + lcm_36_56_84 = 516
  sorry

end sum_gcf_lcm_36_56_84_l156_156898


namespace y_value_is_32_l156_156789

-- Define the conditions
variables (y : ℝ) (hy_pos : y > 0) (hy_eq : y^2 = 1024)

-- State the theorem
theorem y_value_is_32 : y = 32 :=
by
  -- The proof will be written here
  sorry

end y_value_is_32_l156_156789


namespace m_range_circle_l156_156846

noncomputable def circle_equation (m : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + 2 * x + 4 * y + m = 0

theorem m_range_circle (m : ℝ) : circle_equation m → m < 5 := by
  sorry

end m_range_circle_l156_156846


namespace total_brushing_time_in_hours_l156_156131

-- Define the conditions as Lean definitions
def brushing_duration : ℕ := 2   -- 2 minutes per brushing session
def brushing_times_per_day : ℕ := 3  -- brushes 3 times a day
def days : ℕ := 30  -- for 30 days

-- Define the calculation of total brushing time in hours
theorem total_brushing_time_in_hours : (brushing_duration * brushing_times_per_day * days) / 60 = 3 := 
by 
  -- Sorry to skip the proof
  sorry

end total_brushing_time_in_hours_l156_156131


namespace kevin_correct_answer_l156_156826

theorem kevin_correct_answer (k : ℝ) (h : (20 + 1) * (6 + k) = 126 + 21 * k) :
  (20 + 1 * 6 + k) = 21 := by
sorry

end kevin_correct_answer_l156_156826


namespace quadratic_symmetric_l156_156422

-- Conditions: Graph passes through the point P(-2,4)
-- y = ax^2 is symmetric with respect to the y-axis

theorem quadratic_symmetric (a : ℝ) (h : a * (-2)^2 = 4) : a * 2^2 = 4 :=
by
  sorry

end quadratic_symmetric_l156_156422


namespace gcd_problem_l156_156999

-- Define the variables according to the conditions
def m : ℤ := 123^2 + 235^2 + 347^2
def n : ℤ := 122^2 + 234^2 + 348^2

-- Lean statement for the proof problem
theorem gcd_problem : Int.gcd m n = 1 := sorry

end gcd_problem_l156_156999


namespace quadrant_of_complex_number_l156_156544

theorem quadrant_of_complex_number
  (h : ∀ x : ℝ, 0 < x → (a^2 + a + 2)/x < 1/x^2 + 1) :
  ∃ a : ℝ, -1 < a ∧ a < 0 ∧ i^27 = -i :=
sorry

end quadrant_of_complex_number_l156_156544


namespace hockey_games_per_month_calculation_l156_156049

-- Define the given conditions
def months_in_season : Nat := 14
def total_hockey_games : Nat := 182

-- Prove the number of hockey games played each month
theorem hockey_games_per_month_calculation :
  total_hockey_games / months_in_season = 13 := by
  sorry

end hockey_games_per_month_calculation_l156_156049


namespace find_smaller_number_l156_156816

theorem find_smaller_number (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 124) : x = 31 := 
by 
  -- Proof would go here
  sorry

end find_smaller_number_l156_156816


namespace probability_of_both_white_l156_156155

namespace UrnProblem

-- Define the conditions
def firstUrnWhiteBalls : ℕ := 4
def firstUrnTotalBalls : ℕ := 10
def secondUrnWhiteBalls : ℕ := 7
def secondUrnTotalBalls : ℕ := 12

-- Define the probabilities of drawing a white ball from each urn
def P_A1 : ℚ := firstUrnWhiteBalls / firstUrnTotalBalls
def P_A2 : ℚ := secondUrnWhiteBalls / secondUrnTotalBalls

-- Define the combined probability of both events occurring
def P_A1_and_A2 : ℚ := P_A1 * P_A2

-- Theorem statement that checks the combined probability
theorem probability_of_both_white : P_A1_and_A2 = 7 / 30 := by
  sorry

end UrnProblem

end probability_of_both_white_l156_156155


namespace initial_students_l156_156498

variable (x : ℕ) -- let x be the initial number of students

-- each condition defined as a function
def first_round_rem (x : ℕ) : ℕ := (40 * x) / 100
def second_round_rem (x : ℕ) : ℕ := first_round_rem x / 2
def third_round_rem (x : ℕ) : ℕ := second_round_rem x / 4

theorem initial_students (x : ℕ) (h : third_round_rem x = 15) : x = 300 := 
by sorry  -- proof will be inserted here

end initial_students_l156_156498


namespace ball_probability_l156_156120

theorem ball_probability:
  let total_balls := 120
  let red_balls := 12
  let purple_balls := 18
  let yellow_balls := 15
  let desired_probability := 33 / 1190
  let probability_red := red_balls / total_balls
  let probability_purple_or_yellow := (purple_balls + yellow_balls) / (total_balls - 1)
  (probability_red * probability_purple_or_yellow = desired_probability) :=
sorry

end ball_probability_l156_156120


namespace find_velocity_of_current_l156_156250

-- Define the conditions given in the problem
def rowing_speed_in_still_water : ℤ := 10
def distance_to_place : ℤ := 48
def total_travel_time : ℤ := 10

-- Define the primary goal, which is to find the velocity of the current given the conditions
theorem find_velocity_of_current (v : ℤ) 
  (h1 : rowing_speed_in_still_water = 10)
  (h2 : distance_to_place = 48)
  (h3 : total_travel_time = 10) 
  (h4 : rowing_speed_in_still_water * 2 + v * 0 = 
   rowing_speed_in_still_water - v) :
  v = 2 := 
sorry

end find_velocity_of_current_l156_156250


namespace distance_from_integer_l156_156785

theorem distance_from_integer (a : ℝ) (h : a > 0) (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, ∃ m : ℕ, 1 ≤ m ∧ m < n ∧ abs (m * a - k) ≤ (1 / n) :=
by
  sorry

end distance_from_integer_l156_156785


namespace centroid_of_triangle_l156_156725

theorem centroid_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) :
  let x_centroid := (x1 + x2 + x3) / 3
  let y_centroid := (y1 + y2 + y3) / 3
  (x_centroid, y_centroid) = (1/3 * (x1 + x2 + x3), 1/3 * (y1 + y2 + y3)) :=
by
  sorry

end centroid_of_triangle_l156_156725


namespace white_area_of_sign_l156_156747

theorem white_area_of_sign :
  let total_area : ℕ := 6 * 18
  let black_area_C : ℕ := 11
  let black_area_A : ℕ := 10
  let black_area_F : ℕ := 12
  let black_area_E : ℕ := 9
  let total_black_area : ℕ := black_area_C + black_area_A + black_area_F + black_area_E
  let white_area : ℕ := total_area - total_black_area
  white_area = 66 := by
  sorry

end white_area_of_sign_l156_156747


namespace find_x_l156_156409

def x_condition (x : ℤ) : Prop :=
  (120 ≤ x ∧ x ≤ 150) ∧ (x % 5 = 2) ∧ (x % 6 = 5)

theorem find_x :
  ∃ x : ℤ, x_condition x ∧ x = 137 :=
by
  sorry

end find_x_l156_156409


namespace sum_of_products_of_roots_l156_156257

noncomputable def poly : Polynomial ℝ := 5 * Polynomial.X^3 - 10 * Polynomial.X^2 + 17 * Polynomial.X - 7

theorem sum_of_products_of_roots :
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ poly.eval p = 0 ∧ poly.eval q = 0 ∧ poly.eval r = 0) →
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ ((p * q + p * r + q * r) = 17 / 5)) :=
by
  sorry

end sum_of_products_of_roots_l156_156257


namespace transformed_sum_l156_156074

open BigOperators -- Open namespace to use big operators like summation

theorem transformed_sum (n : ℕ) (x : Fin n → ℝ) (s : ℝ) 
  (h_sum : ∑ i, x i = s) : 
  ∑ i, ((3 * (x i + 10)) - 10) = 3 * s + 20 * n :=
by
  sorry

end transformed_sum_l156_156074


namespace probability_yellow_second_l156_156612

section MarbleProbabilities

def bag_A := (5, 6)     -- (white marbles, black marbles)
def bag_B := (3, 7)     -- (yellow marbles, blue marbles)
def bag_C := (5, 6)     -- (yellow marbles, blue marbles)

def P_white_A := 5 / 11
def P_black_A := 6 / 11
def P_yellow_given_B := 3 / 10
def P_yellow_given_C := 5 / 11

theorem probability_yellow_second :
  P_white_A * P_yellow_given_B + P_black_A * P_yellow_given_C = 33 / 121 :=
by
  -- Proof would be provided here
  sorry

end MarbleProbabilities

end probability_yellow_second_l156_156612


namespace gcd_8251_6105_l156_156455

theorem gcd_8251_6105 :
  Nat.gcd 8251 6105 = 37 :=
by
  sorry

end gcd_8251_6105_l156_156455


namespace polygon_quadrilateral_l156_156218

theorem polygon_quadrilateral {n : ℕ} (h : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end polygon_quadrilateral_l156_156218


namespace total_salmons_caught_l156_156673

theorem total_salmons_caught :
  let hazel_salmons := 24
  let dad_salmons := 27
  hazel_salmons + dad_salmons = 51 :=
by
  sorry

end total_salmons_caught_l156_156673


namespace sin_identity_l156_156771

theorem sin_identity (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) :
  Real.sin ((3 * π / 4) - α) = 3 / 5 :=
by
  sorry

end sin_identity_l156_156771


namespace complex_quadrant_l156_156216

-- Define the imaginary unit
def i := Complex.I

-- Define the complex number z satisfying the given condition
variables (z : Complex)
axiom h : (3 - 2 * i) * z = 4 + 3 * i

-- Statement for the proof problem
theorem complex_quadrant (h : (3 - 2 * i) * z = 4 + 3 * i) : 
  (0 < z.re ∧ 0 < z.im) :=
sorry

end complex_quadrant_l156_156216


namespace find_fraction_sum_l156_156981

theorem find_fraction_sum (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -2) : (1 / x) + (1 / y) = -3 :=
by
  sorry

end find_fraction_sum_l156_156981


namespace shape_is_cone_l156_156675

-- Define spherical coordinates
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the positive constant c
def c : ℝ := sorry

-- Assume c is positive
axiom c_positive : c > 0

-- Define the shape equation in spherical coordinates
def shape_equation (p : SphericalCoordinates) : Prop :=
  p.ρ = c * Real.sin p.φ

-- The theorem statement
theorem shape_is_cone (p : SphericalCoordinates) : shape_equation p → 
  ∃ z : ℝ, (z = p.ρ * Real.cos p.φ) ∧ (p.ρ ^ 2 = (c * Real.sin p.φ) ^ 2 + z ^ 2) :=
sorry

end shape_is_cone_l156_156675


namespace real_root_if_and_only_if_l156_156624

theorem real_root_if_and_only_if (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end real_root_if_and_only_if_l156_156624


namespace school_growth_difference_l156_156113

theorem school_growth_difference (X Y : ℕ) (H₁ : Y = 2400)
  (H₂ : X + Y = 4000) : (X + 7 * X / 100 - X) - (Y + 3 * Y / 100 - Y) = 40 :=
by
  sorry

end school_growth_difference_l156_156113


namespace work_done_together_l156_156347

theorem work_done_together
    (fraction_work_left : ℚ)
    (A_days : ℕ)
    (B_days : ℚ) :
    A_days = 20 →
    fraction_work_left = 2 / 3 →
    4 * (1 / 20 + 1 / B_days) = 1 / 3 →
    B_days = 30 := 
by
  intros hA hfrac heq
  sorry

end work_done_together_l156_156347


namespace arithmetic_mean_difference_l156_156709

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 26) : 
  r - p = 32 := by
  sorry

end arithmetic_mean_difference_l156_156709


namespace sum_of_squares_divisible_by_sum_l156_156248

theorem sum_of_squares_divisible_by_sum (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
    (h_bound : a < 2017 ∧ b < 2017 ∧ c < 2017)
    (h_mod : (a^3 - b^3) % 2017 = 0 ∧ (b^3 - c^3) % 2017 = 0 ∧ (c^3 - a^3) % 2017 = 0) :
    (a^2 + b^2 + c^2) % (a + b + c) = 0 :=
by
  sorry

end sum_of_squares_divisible_by_sum_l156_156248


namespace part1_part2_l156_156212

-- Part 1: Proving the value of a given f(x) = a/x + 1 and f(-2) = 0
theorem part1 (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a / x + 1) (h2 : f (-2) = 0) : a = 2 := 
by 
-- Placeholder for the proof
sorry

-- Part 2: Proving the value of f(4) given f(x) = 6/x + 1
theorem part2 (f : ℝ → ℝ) (h1 : ∀ x, f x = 6 / x + 1) : f 4 = 5 / 2 := 
by 
-- Placeholder for the proof
sorry

end part1_part2_l156_156212


namespace bell_rings_before_geography_l156_156173

def number_of_bell_rings : Nat :=
  let assembly_start := 1
  let assembly_end := 1
  let maths_start := 1
  let maths_end := 1
  let history_start := 1
  let history_end := 1
  let quiz_start := 1
  let quiz_end := 1
  let geography_start := 1
  assembly_start + assembly_end + maths_start + maths_end + 
  history_start + history_end + quiz_start + quiz_end + 
  geography_start

theorem bell_rings_before_geography : number_of_bell_rings = 9 := 
by
  -- Proof omitted
  sorry

end bell_rings_before_geography_l156_156173


namespace find_original_production_planned_l156_156349

-- Definition of the problem
variables (x : ℕ)
noncomputable def original_production_planned (x : ℕ) :=
  (6000 / (x + 500)) = (4500 / x)

-- The theorem to prove the original number planned is 1500
theorem find_original_production_planned (x : ℕ) (h : original_production_planned x) : x = 1500 :=
sorry

end find_original_production_planned_l156_156349


namespace part_a_l156_156988

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem part_a :
  ∀ (N : ℕ), (N = (sum_of_digits N) ^ 2) → (N = 1 ∨ N = 81) :=
by
  intros N h
  sorry

end part_a_l156_156988


namespace williams_land_percentage_l156_156681

variable (total_tax : ℕ) (williams_tax : ℕ)

theorem williams_land_percentage (h1 : total_tax = 3840) (h2 : williams_tax = 480) : 
  (williams_tax:ℚ) / (total_tax:ℚ) * 100 = 12.5 := 
  sorry

end williams_land_percentage_l156_156681


namespace six_digit_number_property_l156_156411

theorem six_digit_number_property {a b c d e f : ℕ} 
  (h1 : 1 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10)
  (h3 : 0 ≤ c ∧ c < 10) (h4 : 0 ≤ d ∧ d < 10)
  (h5 : 0 ≤ e ∧ e < 10) (h6 : 0 ≤ f ∧ f < 10) 
  (h7 : 100000 ≤ a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f ∧
        a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f < 1000000) :
  (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 3 * (f * 10^5 + a * 10^4 + b * 10^3 + c * 10^2 + d * 10 + e)) ↔ 
  (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 428571 ∨ a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 857142) :=
sorry

end six_digit_number_property_l156_156411


namespace octagon_perimeter_l156_156080

-- Definitions based on conditions
def is_octagon (n : ℕ) : Prop := n = 8
def side_length : ℕ := 12

-- The proof problem statement
theorem octagon_perimeter (n : ℕ) (h : is_octagon n) : n * side_length = 96 := by
  sorry

end octagon_perimeter_l156_156080


namespace base7_divisibility_rules_2_base7_divisibility_rules_3_l156_156596

def divisible_by_2 (d : Nat) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4

def divisible_by_3 (d : Nat) : Prop :=
  d = 0 ∨ d = 3

def last_digit_base7 (n : Nat) : Nat :=
  n % 7

theorem base7_divisibility_rules_2 (n : Nat) :
  (∃ k, n = 2 * k) ↔ divisible_by_2 (last_digit_base7 n) :=
by
  sorry

theorem base7_divisibility_rules_3 (n : Nat) :
  (∃ k, n = 3 * k) ↔ divisible_by_3 (last_digit_base7 n) :=
by
  sorry

end base7_divisibility_rules_2_base7_divisibility_rules_3_l156_156596


namespace problem1_problem2_l156_156884

variable {n : ℕ}
variable {a b : ℝ}

-- Part 1
theorem problem1 (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |(2 * n^2 / (n + 2) - n * a) - b| < ε) :
  a = 2 ∧ b = 4 := sorry

-- Part 2
theorem problem2 (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |(3^n / (3^(n + 1) + (a + 1)^n) - 1/3)| < ε) :
  -4 < a ∧ a < 2 := sorry

end problem1_problem2_l156_156884


namespace find_fiona_experience_l156_156644

namespace Experience

variables (d e f : ℚ)

def avg_experience_equation : Prop := d + e + f = 36
def fiona_david_equation : Prop := f - 5 = d
def emma_david_future_equation : Prop := e + 4 = (3/4) * (d + 4)

theorem find_fiona_experience (h1 : avg_experience_equation d e f) (h2 : fiona_david_equation d f) (h3 : emma_david_future_equation d e) :
  f = 183 / 11 :=
by
  sorry

end Experience

end find_fiona_experience_l156_156644


namespace green_tea_price_decrease_l156_156136

def percentage_change (old_price new_price : ℚ) : ℚ :=
  ((new_price - old_price) / old_price) * 100

theorem green_tea_price_decrease
  (C : ℚ)
  (h1 : C > 0)
  (july_coffee_price : ℚ := 2 * C)
  (mixture_price : ℚ := 3.45)
  (july_green_tea_price : ℚ := 0.3)
  (old_green_tea_price : ℚ := C)
  (equal_mixture : ℚ := (1.5 * july_green_tea_price) + (1.5 * july_coffee_price)) :
  mixture_price = equal_mixture →
  percentage_change old_green_tea_price july_green_tea_price = -70 :=
by
  sorry

end green_tea_price_decrease_l156_156136


namespace bus_passenger_count_l156_156178

-- Definition of the function f representing the number of passengers per trip
def passengers (n : ℕ) : ℕ :=
  120 - 2 * n

-- The total number of trips is 18 (from 9 AM to 5:30 PM inclusive)
def total_trips : ℕ := 18

-- Sum of passengers over all trips
def total_passengers : ℕ :=
  List.sum (List.map passengers (List.range total_trips))

-- Problem statement
theorem bus_passenger_count :
  total_passengers = 1854 :=
sorry

end bus_passenger_count_l156_156178


namespace min_x_value_l156_156708

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 18 * x + 50 * y + 56

theorem min_x_value : 
  ∃ (x : ℝ), ∃ (y : ℝ), circle_eq x y ∧ x = 9 - Real.sqrt 762 :=
by
  sorry

end min_x_value_l156_156708


namespace total_cost_toys_l156_156651

variable (c_e_actionfigs : ℕ := 60) -- number of action figures for elder son
variable (cost_e_actionfig : ℕ := 5) -- cost per action figure for elder son
variable (c_y_actionfigs : ℕ := 3 * c_e_actionfigs) -- number of action figures for younger son
variable (cost_y_actionfig : ℕ := 4) -- cost per action figure for younger son
variable (c_y_cars : ℕ := 20) -- number of cars for younger son
variable (cost_car : ℕ := 3) -- cost per car
variable (c_y_animals : ℕ := 10) -- number of stuffed animals for younger son
variable (cost_animal : ℕ := 7) -- cost per stuffed animal

theorem total_cost_toys (c_e_actionfigs c_y_actionfigs c_y_cars c_y_animals : ℕ)
                         (cost_e_actionfig cost_y_actionfig cost_car cost_animal : ℕ) :
  (c_e_actionfigs * cost_e_actionfig + c_y_actionfigs * cost_y_actionfig + 
  c_y_cars * cost_car + c_y_animals * cost_animal) = 1150 := by
  sorry

end total_cost_toys_l156_156651


namespace cos_two_pi_over_three_plus_two_alpha_l156_156606

theorem cos_two_pi_over_three_plus_two_alpha 
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := 
by
  sorry

end cos_two_pi_over_three_plus_two_alpha_l156_156606


namespace tautology_a_tautology_b_tautology_c_tautology_d_l156_156669

variable (p q : Prop)

theorem tautology_a : p ∨ ¬ p := by
  sorry

theorem tautology_b : ¬ ¬ p ↔ p := by
  sorry

theorem tautology_c : ((p → q) → p) → p := by
  sorry

theorem tautology_d : ¬ (p ∧ ¬ p) := by
  sorry

end tautology_a_tautology_b_tautology_c_tautology_d_l156_156669


namespace minimum_value_f_l156_156845

open Real

noncomputable def f (x : ℝ) : ℝ :=
  x + (3 * x) / (x^2 + 3) + (x * (x + 3)) / (x^2 + 1) + (3 * (x + 1)) / (x * (x^2 + 1))

theorem minimum_value_f (x : ℝ) (hx : x > 0) : f x ≥ 7 :=
by
  -- Proof omitted
  sorry

end minimum_value_f_l156_156845


namespace find_g3_l156_156445

theorem find_g3 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 ^ x) + 2 * x * g (3 ^ (-x)) = 1) : 
  g 3 = 1 / 5 := 
sorry

end find_g3_l156_156445


namespace volume_of_revolved_region_l156_156039

theorem volume_of_revolved_region :
  let R := {p : ℝ × ℝ | |8 - p.1| + p.2 ≤ 10 ∧ 3 * p.2 - p.1 ≥ 15}
  let volume := (1 / 3) * Real.pi * (7 / Real.sqrt 10)^2 * (7 * Real.sqrt 10 / 4)
  let m := 343
  let n := 12
  let p := 10
  m + n + p = 365 := by
  sorry

end volume_of_revolved_region_l156_156039


namespace find_p_tilde_one_l156_156575

noncomputable def p (x : ℝ) : ℝ :=
  let r : ℝ := -1 / 9
  let s : ℝ := 1
  x^2 - (r + s) * x + (r * s)

theorem find_p_tilde_one : p 1 = 0 := by
  sorry

end find_p_tilde_one_l156_156575


namespace axis_of_symmetry_range_l156_156557

theorem axis_of_symmetry_range (a : ℝ) : (-(a + 2) / (3 - 4 * a) > 0) ↔ (a < -2 ∨ a > 3 / 4) :=
by
  sorry

end axis_of_symmetry_range_l156_156557


namespace evaluate_expression_l156_156950

theorem evaluate_expression : 
  abs (abs (-abs (3 - 5) + 2) - 4) = 4 :=
by
  sorry

end evaluate_expression_l156_156950


namespace tangent_line_of_cubic_at_l156_156705

theorem tangent_line_of_cubic_at (x y : ℝ) (h : y = x^3) (hx : x = 1) (hy : y = 1) : 
  3 * x - y - 2 = 0 :=
sorry

end tangent_line_of_cubic_at_l156_156705


namespace perfect_square_polynomial_l156_156476

theorem perfect_square_polynomial (m : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x : ℝ, x^2 + 2*(m-3)*x + 25 = f x * f x) ↔ (m = 8 ∨ m = -2) :=
by
  sorry

end perfect_square_polynomial_l156_156476


namespace number_is_93_75_l156_156976

theorem number_is_93_75 (x : ℝ) (h : 0.16 * (0.40 * x) = 6) : x = 93.75 :=
by
  -- The proof is omitted.
  sorry

end number_is_93_75_l156_156976


namespace triangle_is_isosceles_l156_156015

theorem triangle_is_isosceles {a b c : ℝ} {A B C : ℝ} (h1 : b * Real.cos A = a * Real.cos B) : 
  a = b ∨ a = c ∨ b = c :=
sorry

end triangle_is_isosceles_l156_156015


namespace smallest_of_five_consecutive_even_sum_500_l156_156160

theorem smallest_of_five_consecutive_even_sum_500 : 
  ∃ (n : Int), (n - 4, n - 2, n, n + 2, n + 4).1 = 96 ∧ 
  ((n - 4) + (n - 2) + n + (n + 2) + (n + 4) = 500) :=
by
  sorry

end smallest_of_five_consecutive_even_sum_500_l156_156160


namespace triangle_right_triangle_l156_156778

variable {A B C : Real}  -- Define the angles A, B, and C

theorem triangle_right_triangle (sin_A sin_B sin_C : Real)
  (h : sin_A^2 + sin_B^2 = sin_C^2) 
  (triangle_cond : A + B + C = 180) : 
  (A = 90) ∨ (B = 90) ∨ (C = 90) := 
  sorry

end triangle_right_triangle_l156_156778


namespace number_of_points_marked_l156_156863

theorem number_of_points_marked (a₁ a₂ b₁ b₂ : ℕ) 
  (h₁ : a₁ * a₂ = 50) (h₂ : b₁ * b₂ = 56) (h₃ : a₁ + a₂ = b₁ + b₂) : 
  (a₁ + a₂ + 1 = 16) :=
sorry

end number_of_points_marked_l156_156863


namespace max_min_diff_eq_four_l156_156402

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * |x - a|

theorem max_min_diff_eq_four (a : ℝ) (h_a : a ≥ 2) : 
    let M := max (f a (-1)) (f a 1)
    let m := min (f a (-1)) (f a 1)
    M - m = 4 :=
by
  sorry

end max_min_diff_eq_four_l156_156402


namespace number_of_packages_l156_156471

theorem number_of_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 56) (h2 : tshirts_per_package = 2) : 
  (total_tshirts / tshirts_per_package) = 28 := 
  by
    sorry

end number_of_packages_l156_156471


namespace f_2015_l156_156781

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 3) = f x
axiom f_interval : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = 2^x

theorem f_2015 : f 2015 = -2 := sorry

end f_2015_l156_156781


namespace problem_real_numbers_l156_156319

theorem problem_real_numbers (a b : ℝ) (n : ℕ) (h : 2 * a + 3 * b = 12) : 
  ((a / 3) ^ n + (b / 2) ^ n) ≥ 2 := 
sorry

end problem_real_numbers_l156_156319


namespace winning_strategy_l156_156891

theorem winning_strategy (n : ℕ) (take_stones : ℕ → Prop) :
  n = 13 ∧ (∀ k, (k = 1 ∨ k = 2) → take_stones k) →
  (take_stones 12 ∨ take_stones 9 ∨ take_stones 6 ∨ take_stones 3) :=
by sorry

end winning_strategy_l156_156891


namespace grape_juice_amount_l156_156570

-- Definitions for the conditions
def total_weight : ℝ := 150
def orange_percentage : ℝ := 0.35
def watermelon_percentage : ℝ := 0.35

-- Theorem statement to prove the amount of grape juice
theorem grape_juice_amount : 
  (total_weight * (1 - orange_percentage - watermelon_percentage)) = 45 :=
by
  sorry

end grape_juice_amount_l156_156570


namespace ratio_of_boxes_loaded_l156_156437

variable (D N B : ℕ) 

-- Definitions as conditions
def night_crew_workers (D : ℕ) : ℕ := (4 * D) / 9
def day_crew_boxes (B : ℕ) : ℕ := (3 * B) / 4
def night_crew_boxes (B : ℕ) : ℕ := B / 4

theorem ratio_of_boxes_loaded :
  ∀ {D B : ℕ}, 
    night_crew_workers D ≠ 0 → 
    D ≠ 0 → 
    B ≠ 0 → 
    ((night_crew_boxes B) / (night_crew_workers D)) / ((day_crew_boxes B) / D) = 3 / 4 :=
by
  -- Proof
  sorry

end ratio_of_boxes_loaded_l156_156437


namespace cheesecake_factory_savings_l156_156922

noncomputable def combined_savings : ℕ := 3000

theorem cheesecake_factory_savings :
  let hourly_wage := 10
  let daily_hours := 10
  let working_days := 5
  let weekly_hours := daily_hours * working_days
  let weekly_salary := weekly_hours * hourly_wage
  let robby_savings := (2/5 : ℚ) * weekly_salary
  let jaylen_savings := (3/5 : ℚ) * weekly_salary
  let miranda_savings := (1/2 : ℚ) * weekly_salary
  let combined_weekly_savings := robby_savings + jaylen_savings + miranda_savings
  4 * combined_weekly_savings = combined_savings :=
by
  sorry

end cheesecake_factory_savings_l156_156922


namespace symmetric_line_equation_l156_156569

theorem symmetric_line_equation (x y : ℝ) :
  let line_original := x - 2 * y + 1 = 0
  let line_symmetry := x = 1
  let line_symmetric := x + 2 * y - 3 = 0
  ∀ (x y : ℝ), (2 - x - 2 * y + 1 = 0) ↔ (x + 2 * y - 3 = 0) := by
sorry

end symmetric_line_equation_l156_156569


namespace triangle_angle_side_inequality_l156_156754

variable {A B C : Type} -- Variables for points in the triangle
variable {a b : ℝ} -- Variables for the lengths of sides opposite to angles A and B
variable {A_angle B_angle : ℝ} -- Variables for the angles at A and B in triangle ABC

-- Define that we are in a triangle setting
def is_triangle (A B C : Type) := True

-- Define the assumption for the proof by contradiction
def assumption (a b : ℝ) := a ≤ b

theorem triangle_angle_side_inequality (h_triangle : is_triangle A B C)
(h_angle : A_angle > B_angle) 
(h_assumption : assumption a b) : a > b := 
sorry

end triangle_angle_side_inequality_l156_156754


namespace pool_capacity_l156_156501

variables {T : ℕ} {A B C : ℕ → ℕ}

-- Conditions
def valve_rate_A (T : ℕ) : ℕ := T / 180
def valve_rate_B (T : ℕ) := valve_rate_A T + 60
def valve_rate_C (T : ℕ) := valve_rate_A T + 75

def combined_rate (T : ℕ) := valve_rate_A T + valve_rate_B T + valve_rate_C T

-- Theorem to prove
theorem pool_capacity (T : ℕ) (h1 : combined_rate T = T / 40) : T = 16200 :=
by
  sorry

end pool_capacity_l156_156501


namespace thirteen_power_1997_tens_digit_l156_156888

def tens_digit (n : ℕ) := (n / 10) % 10

theorem thirteen_power_1997_tens_digit :
  tens_digit (13 ^ 1997 % 100) = 5 := by
  sorry

end thirteen_power_1997_tens_digit_l156_156888


namespace range_of_a_plus_b_l156_156087

theorem range_of_a_plus_b 
  (a b : ℝ)
  (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) ≥ -1) : 
  -1 ≤ a + b ∧ a + b ≤ 2 :=
sorry

end range_of_a_plus_b_l156_156087


namespace find_annual_pension_l156_156431

variable (P k x a b p q : ℝ) (h1 : k * Real.sqrt (x + a) = k * Real.sqrt x + p)
                                   (h2 : k * Real.sqrt (x + b) = k * Real.sqrt x + q)

theorem find_annual_pension (h_nonzero_proportionality_constant : k ≠ 0) 
(h_year_difference : a ≠ b) : 
P = (a * q ^ 2 - b * p ^ 2) / (2 * (b * p - a * q)) := 
by
  sorry

end find_annual_pension_l156_156431


namespace largest_divisor_of_5_consecutive_integers_l156_156970

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l156_156970


namespace number_of_planters_l156_156034

variable (a b : ℕ)

-- Conditions
def tree_planting_condition_1 : Prop := a * b = 2013
def tree_planting_condition_2 : Prop := (a - 5) * (b + 2) < 2013
def tree_planting_condition_3 : Prop := (a - 5) * (b + 3) > 2013

-- Theorem stating the number of people who participated in the planting is 61
theorem number_of_planters (h1 : tree_planting_condition_1 a b) 
                           (h2 : tree_planting_condition_2 a b) 
                           (h3 : tree_planting_condition_3 a b) : 
                           a = 61 := 
sorry

end number_of_planters_l156_156034


namespace solve_equation1_solve_equation2_l156_156095

-- Define the first equation as a condition
def equation1 (x : ℝ) : Prop :=
  3 * x + 20 = 4 * x - 25

-- Prove that x = 45 satisfies equation1
theorem solve_equation1 : equation1 45 :=
by 
  -- Proof steps would go here
  sorry

-- Define the second equation as a condition
def equation2 (x : ℝ) : Prop :=
  (2 * x - 1) / 3 = 1 - (2 * x - 1) / 6

-- Prove that x = 3/2 satisfies equation2
theorem solve_equation2 : equation2 (3 / 2) :=
by 
  -- Proof steps would go here
  sorry

end solve_equation1_solve_equation2_l156_156095


namespace distinct_points_count_l156_156048

-- Definitions based on conditions
def eq1 (x y : ℝ) : Prop := (x + y = 7) ∨ (2 * x - 3 * y = -7)
def eq2 (x y : ℝ) : Prop := (x - y = 3) ∨ (3 * x + 2 * y = 18)

-- The statement combining conditions and requiring the proof of 3 distinct solutions
theorem distinct_points_count : 
  ∃! p1 p2 p3 : ℝ × ℝ, 
    (eq1 p1.1 p1.2 ∧ eq2 p1.1 p1.2) ∧ 
    (eq1 p2.1 p2.2 ∧ eq2 p2.1 p2.2) ∧ 
    (eq1 p3.1 p3.2 ∧ eq2 p3.1 p3.2) ∧ 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 :=
sorry

end distinct_points_count_l156_156048


namespace find_y_six_l156_156837

theorem find_y_six (y : ℝ) (h : y > 0) (h_eq : (2 - y^3)^(1/3) + (2 + y^3)^(1/3) = 2) : 
    y^6 = 116 / 27 :=
by
  sorry

end find_y_six_l156_156837


namespace max_z_value_l156_156310

theorem max_z_value (x y z : ℝ) (h : x + y + z = 3) (h' : x * y + y * z + z * x = 2) : z ≤ 5 / 3 :=
  sorry


end max_z_value_l156_156310


namespace example_is_fraction_l156_156222

def is_fraction (a b : ℚ) : Prop := ∃ x y : ℚ, a = x ∧ b = y ∧ y ≠ 0

-- Example condition relevant to the problem
theorem example_is_fraction (x : ℚ) : is_fraction x (x + 2) :=
by
  sorry

end example_is_fraction_l156_156222


namespace additional_houses_built_by_october_l156_156578

def total_houses : ℕ := 2000
def fraction_built_first_half : ℚ := 3 / 5
def houses_needed_by_october : ℕ := 500

def houses_built_first_half : ℚ := fraction_built_first_half * total_houses
def houses_built_by_october : ℕ := total_houses - houses_needed_by_october

theorem additional_houses_built_by_october :
  (houses_built_by_october - houses_built_first_half) = 300 := by
  sorry

end additional_houses_built_by_october_l156_156578


namespace Amanda_notebooks_l156_156912

theorem Amanda_notebooks (initial ordered lost final : ℕ) 
  (h_initial: initial = 65) 
  (h_ordered: ordered = 23) 
  (h_lost: lost = 14) : 
  final = 74 := 
by 
  -- calculation and proof will go here
  sorry 

end Amanda_notebooks_l156_156912


namespace find_possible_values_of_b_l156_156055

def good_number (x : ℕ) : Prop :=
  ∃ p n : ℕ, Nat.Prime p ∧ n ≥ 2 ∧ x = p^n

theorem find_possible_values_of_b (b : ℕ) : 
  (b ≥ 4) ∧ good_number (b^2 - 2 * b - 3) ↔ b = 87 := sorry

end find_possible_values_of_b_l156_156055


namespace calculate_volume_from_measurements_l156_156251

variables (r h : ℝ) (P : ℝ × ℝ)

noncomputable def volume_truncated_cylinder (area_base : ℝ) (height_segment : ℝ) : ℝ :=
  area_base * height_segment

theorem calculate_volume_from_measurements
    (radius : ℝ) (height : ℝ)
    (area_base : ℝ := π * radius^2)
    (P : ℝ × ℝ)  -- intersection point on the axis
    (height_segment : ℝ) : 
    volume_truncated_cylinder area_base height_segment = area_base * height_segment :=
by
  -- The proof would involve demonstrating the relationship mathematically
  sorry

end calculate_volume_from_measurements_l156_156251


namespace solution_set_of_abs_inequality_l156_156202

theorem solution_set_of_abs_inequality (x : ℝ) : |x| - |x - 3| < 2 ↔ x < 2.5 :=
by
  sorry

end solution_set_of_abs_inequality_l156_156202


namespace alice_average_speed_l156_156341

def average_speed (distance1 speed1 distance2 speed2 totalDistance totalTime : ℚ) :=
  totalDistance / totalTime

theorem alice_average_speed : 
  let d1 := 45
  let s1 := 15
  let d2 := 15
  let s2 := 45
  let totalDistance := d1 + d2
  let totalTime := (d1 / s1) + (d2 / s2)
  average_speed d1 s1 d2 s2 totalDistance totalTime = 18 :=
by
  sorry

end alice_average_speed_l156_156341


namespace probability_white_ball_l156_156663

theorem probability_white_ball (num_black_balls num_white_balls : ℕ) 
  (black_balls : num_black_balls = 6) 
  (white_balls : num_white_balls = 5) : 
  (num_white_balls / (num_black_balls + num_white_balls) : ℚ) = 5 / 11 :=
by
  sorry

end probability_white_ball_l156_156663


namespace part_a_part_b_l156_156915

-- Part (a)
theorem part_a (n : ℕ) (a b : ℝ) : 
  a^(n+1) + b^(n+1) = (a + b) * (a^n + b^n) - a * b * (a^(n - 1) + b^(n - 1)) :=
by sorry

-- Part (b)
theorem part_b {a b : ℝ} (h1 : a + b = 1) (h2: a * b = -1) : 
  a^10 + b^10 = 123 :=
by sorry

end part_a_part_b_l156_156915


namespace base_of_third_term_l156_156526

theorem base_of_third_term (x : ℝ) (some_number : ℝ) :
  625^(-x) + 25^(-2 * x) + some_number^(-4 * x) = 14 → x = 0.25 → some_number = 125 / 1744 :=
by
  intros h1 h2
  sorry

end base_of_third_term_l156_156526


namespace janet_speed_l156_156918

def janet_sister_speed : ℝ := 12
def lake_width : ℝ := 60
def wait_time : ℝ := 3

theorem janet_speed :
  (lake_width / (lake_width / janet_sister_speed - wait_time)) = 30 := 
sorry

end janet_speed_l156_156918


namespace maximum_piles_l156_156344

theorem maximum_piles (n : ℕ) (h : n = 660) : 
  ∃ m, m = 30 ∧ 
       ∀ (piles : Finset ℕ), (piles.sum id = n) →
       (∀ x ∈ piles, ∀ y ∈ piles, x ≤ y → y < 2 * x) → 
       (piles.card ≤ m) :=
by
  sorry

end maximum_piles_l156_156344


namespace jerry_gets_logs_l156_156696

def logs_per_pine_tree : ℕ := 80
def logs_per_maple_tree : ℕ := 60
def logs_per_walnut_tree : ℕ := 100
def logs_per_oak_tree : ℕ := 90
def logs_per_birch_tree : ℕ := 55

def pine_trees_cut : ℕ := 8
def maple_trees_cut : ℕ := 3
def walnut_trees_cut : ℕ := 4
def oak_trees_cut : ℕ := 7
def birch_trees_cut : ℕ := 5

def total_logs : ℕ :=
  pine_trees_cut * logs_per_pine_tree +
  maple_trees_cut * logs_per_maple_tree +
  walnut_trees_cut * logs_per_walnut_tree +
  oak_trees_cut * logs_per_oak_tree +
  birch_trees_cut * logs_per_birch_tree

theorem jerry_gets_logs : total_logs = 2125 :=
by
  sorry

end jerry_gets_logs_l156_156696


namespace volume_of_prism_l156_156268

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 20) (h3 : c * a = 12) (h4 : a + b + c = 11) :
  a * b * c = 12 * Real.sqrt 15 :=
by
  sorry

end volume_of_prism_l156_156268


namespace product_evaluation_l156_156530

theorem product_evaluation : 
  (7 - 5) * (7 - 4) * (7 - 3) * (7 - 2) * (7- 1) * 7 = 5040 := 
by 
  sorry

end product_evaluation_l156_156530


namespace additional_license_plates_l156_156562

def original_license_plates : ℕ := 5 * 3 * 5
def new_license_plates : ℕ := 6 * 4 * 5

theorem additional_license_plates : new_license_plates - original_license_plates = 45 := by
  sorry

end additional_license_plates_l156_156562


namespace points_on_line_l156_156282

theorem points_on_line (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l156_156282


namespace sin_cos_identity_l156_156351

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
by
  sorry

end sin_cos_identity_l156_156351


namespace cage_chicken_problem_l156_156253

theorem cage_chicken_problem :
  (∃ x : ℕ, 6 ≤ x ∧ x ≤ 10 ∧ (4 * x + 1 = 5 * (x - 1))) ∧
  (∀ x : ℕ, 6 ≤ x ∧ x ≤ 10 → (4 * x + 1 ≥ 25 ∧ 4 * x + 1 ≤ 41)) :=
by
  sorry

end cage_chicken_problem_l156_156253


namespace closest_point_on_plane_exists_l156_156580

def point_on_plane : Type := {P : ℝ × ℝ × ℝ // ∃ (x y z : ℝ), P = (x, y, z) ∧ 2 * x - 3 * y + 4 * z = 20}

def point_A : ℝ × ℝ × ℝ := (0, 1, -1)

theorem closest_point_on_plane_exists (P : point_on_plane) :
  ∃ (x y z : ℝ), (x, y, z) = (54 / 29, -80 / 29, 83 / 29) := sorry

end closest_point_on_plane_exists_l156_156580


namespace opposite_sides_of_line_l156_156653

theorem opposite_sides_of_line (m : ℝ) (h : (2 * (-2 : ℝ) + m - 2) * (2 * m + 4 - 2) < 0) : -1 < m ∧ m < 6 :=
sorry

end opposite_sides_of_line_l156_156653


namespace volume_of_cone_l156_156588

theorem volume_of_cone
  (r h l : ℝ) -- declaring variables
  (base_area : ℝ) (lateral_surface_is_semicircle : ℝ) 
  (h_eq : h = Real.sqrt (l^2 - r^2))
  (base_area_eq : π * r^2 = π)
  (lateral_surface_eq : π * l = 2 * π * r) : 
  (∀ (V : ℝ), V = (1 / 3) * π * r^2 * h → V = (Real.sqrt 3) / 3 * π) :=
by
  sorry

end volume_of_cone_l156_156588


namespace positive_solution_iff_abs_a_b_lt_one_l156_156692

theorem positive_solution_iff_abs_a_b_lt_one
  (a b : ℝ)
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 - x2 = a)
  (h2 : x3 - x4 = b)
  (h3 : x1 + x2 + x3 + x4 = 1)
  (h4 : x1 > 0)
  (h5 : x2 > 0)
  (h6 : x3 > 0)
  (h7 : x4 > 0) :
  |a| + |b| < 1 :=
sorry

end positive_solution_iff_abs_a_b_lt_one_l156_156692


namespace pulley_distance_l156_156764

theorem pulley_distance (r₁ r₂ d l : ℝ):
    r₁ = 10 →
    r₂ = 6 →
    l = 30 →
    (d = 2 * Real.sqrt 229) :=
by
    intros h₁ h₂ h₃
    sorry

end pulley_distance_l156_156764


namespace problem_proof_l156_156640

theorem problem_proof:
  (∃ n : ℕ, 25 = n ^ 2) ∧
  (Prime 31) ∧
  (¬ ∀ p : ℕ, Prime p → p >= 3 → p = 2) ∧
  (∃ m : ℕ, 8 = m ^ 3) ∧
  (∃ a b : ℕ, Prime a ∧ Prime b ∧ 15 = a * b) :=
by
  sorry

end problem_proof_l156_156640


namespace cone_surface_area_ratio_l156_156823

theorem cone_surface_area_ratio (l : ℝ) (h_l_pos : 0 < l) :
  let θ := (120 * Real.pi) / 180 -- converting 120 degrees to radians
  let side_area := (1/2) * l^2 * θ
  let r := l / 3
  let base_area := Real.pi * r^2
  let surface_area := side_area + base_area
  side_area ≠ 0 → 
  surface_area / side_area = 4 / 3 := 
by
  -- Provide the proof here
  sorry

end cone_surface_area_ratio_l156_156823


namespace smallest_sum_minimum_l156_156886

noncomputable def smallest_sum (x y : ℕ) : ℕ :=
if h₁ : x ≠ y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) then x + y else 0

theorem smallest_sum_minimum (x y : ℕ) (h₁ : x ≠ y) (h₂ : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 24) :
  smallest_sum x y = 96 := sorry

end smallest_sum_minimum_l156_156886


namespace quadratic_inequality_solution_l156_156581

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 4 * x > 45 ↔ x < -9 ∨ x > 5 := 
  sorry

end quadratic_inequality_solution_l156_156581


namespace f_at_8_l156_156670

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

theorem f_at_8 : f 8 = -1 := 
by
-- The following will be filled with the proof, hence sorry for now.
sorry

end f_at_8_l156_156670


namespace collinear_points_in_cube_l156_156299

def collinear_groups_in_cube : Prop :=
  let vertices := 8
  let edge_midpoints := 12
  let face_centers := 6
  let center_point := 1
  let total_groups :=
    (vertices * (vertices - 1) / 2) + (face_centers * 1 / 2) + (edge_midpoints * 3 / 2)
  total_groups = 49

theorem collinear_points_in_cube : collinear_groups_in_cube :=
  by
    sorry

end collinear_points_in_cube_l156_156299


namespace percentage_of_Hindu_boys_l156_156326

theorem percentage_of_Hindu_boys (total_boys : ℕ) (muslim_percentage : ℕ) (sikh_percentage : ℕ)
  (other_community_boys : ℕ) (H : total_boys = 850) (H1 : muslim_percentage = 44) 
  (H2 : sikh_percentage = 10) (H3 : other_community_boys = 153) :
  let muslim_boys := muslim_percentage * total_boys / 100
  let sikh_boys := sikh_percentage * total_boys / 100
  let non_hindu_boys := muslim_boys + sikh_boys + other_community_boys
  let hindu_boys := total_boys - non_hindu_boys
  (hindu_boys * 100 / total_boys : ℚ) = 28 := 
by
  sorry

end percentage_of_Hindu_boys_l156_156326


namespace jiahao_estimate_larger_l156_156925

variable (x y : ℝ)
variable (hxy : x > y)
variable (hy0 : y > 0)

theorem jiahao_estimate_larger (x y : ℝ) (hxy : x > y) (hy0 : y > 0) :
  (x + 2) - (y - 1) > x - y :=
by
  sorry

end jiahao_estimate_larger_l156_156925


namespace compute_expr_l156_156962

theorem compute_expr : 5^2 - 3 * 4 + 3^2 = 22 := by
  sorry

end compute_expr_l156_156962


namespace negation_of_exists_leq_l156_156440

theorem negation_of_exists_leq (x : ℝ) : ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_of_exists_leq_l156_156440


namespace inequality_solution_l156_156691

theorem inequality_solution (x : ℝ) :
  (2 * x^2 - 4 * x - 70 > 0) ∧ (x ≠ -2) ∧ (x ≠ 0) ↔ (x < -5 ∨ x > 7) :=
by
  sorry

end inequality_solution_l156_156691


namespace prod_mod_11_remainder_zero_l156_156377

theorem prod_mod_11_remainder_zero : (108 * 110) % 11 = 0 := 
by sorry

end prod_mod_11_remainder_zero_l156_156377


namespace solve_fraction_l156_156301

theorem solve_fraction (x : ℝ) (h1 : x + 2 = 0) (h2 : 2 * x - 4 ≠ 0) : x = -2 := 
by 
  sorry

end solve_fraction_l156_156301


namespace compare_neg_fractions_l156_156423

theorem compare_neg_fractions : (- (3 : ℚ) / 5) > (- (3 : ℚ) / 4) := sorry

end compare_neg_fractions_l156_156423


namespace exists_h_not_divisible_l156_156733

theorem exists_h_not_divisible : ∃ (h : ℝ), ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ % ⌊h * 1969^(n-1)⌋ = 0) :=
by
  sorry

end exists_h_not_divisible_l156_156733


namespace solve_indeterminate_equation_l156_156632

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem solve_indeterminate_equation (x y : ℕ) (hx : is_prime x) (hy : is_prime y) :
  x^2 - y^2 = x * y^2 - 19 ↔ (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) :=
by
  sorry

end solve_indeterminate_equation_l156_156632


namespace benny_gave_sandy_books_l156_156470

theorem benny_gave_sandy_books :
  ∀ (Benny_initial Tim_books total_books Benny_after_giving : ℕ), 
    Benny_initial = 24 → 
    Tim_books = 33 →
    total_books = 47 → 
    total_books - Tim_books = Benny_after_giving →
    Benny_initial - Benny_after_giving = 10 :=
by
  intros Benny_initial Tim_books total_books Benny_after_giving
  intros hBenny_initial hTim_books htotal_books hBooks_after
  simp [hBenny_initial, hTim_books, htotal_books, hBooks_after]
  sorry


end benny_gave_sandy_books_l156_156470


namespace arccos_of_sqrt3_div_2_l156_156839

theorem arccos_of_sqrt3_div_2 : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end arccos_of_sqrt3_div_2_l156_156839


namespace rectangle_side_greater_than_12_l156_156565

theorem rectangle_side_greater_than_12 
  (a b : ℝ) (h₁ : a ≠ b) (h₂ : a * b = 6 * (a + b)) : a > 12 ∨ b > 12 := 
by
  sorry

end rectangle_side_greater_than_12_l156_156565


namespace inscribed_square_area_l156_156949

noncomputable def area_inscribed_square (AB CD : ℕ) (BCFE : ℕ) : Prop :=
  AB = 36 ∧ CD = 64 ∧ BCFE = (AB * CD)

theorem inscribed_square_area :
  ∀ (AB CD : ℕ),
  area_inscribed_square AB CD 2304 :=
by
  intros
  sorry

end inscribed_square_area_l156_156949


namespace value_of_g_at_2_l156_156503

def g (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

theorem value_of_g_at_2 : g 2 = 11 := 
by
  sorry

end value_of_g_at_2_l156_156503


namespace willy_crayons_difference_l156_156597

def willy : Int := 5092
def lucy : Int := 3971
def jake : Int := 2435

theorem willy_crayons_difference : willy - (lucy + jake) = -1314 := by
  sorry

end willy_crayons_difference_l156_156597


namespace number_of_people_on_boats_l156_156630

def boats := 5
def people_per_boat := 3

theorem number_of_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end number_of_people_on_boats_l156_156630


namespace drunk_drivers_traffic_class_l156_156540

-- Define the variables for drunk drivers and speeders
variable (d s : ℕ)

-- Define the given conditions as hypotheses
theorem drunk_drivers_traffic_class (h1 : d + s = 45) (h2 : s = 7 * d - 3) : d = 6 := by
  sorry

end drunk_drivers_traffic_class_l156_156540


namespace y_is_never_perfect_square_l156_156568

theorem y_is_never_perfect_square (x : ℕ) : ¬ ∃ k : ℕ, k^2 = x^4 + 2*x^3 + 2*x^2 + 2*x + 1 :=
sorry

end y_is_never_perfect_square_l156_156568


namespace largest_number_is_B_l156_156190

-- Define the numbers as constants
def A : ℝ := 0.989
def B : ℝ := 0.998
def C : ℝ := 0.981
def D : ℝ := 0.899
def E : ℝ := 0.9801

-- State the theorem that B is the largest number
theorem largest_number_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  -- By comparison
  sorry

end largest_number_is_B_l156_156190


namespace bonus_received_l156_156306

-- Definitions based on the conditions
def total_sales (S : ℝ) : Prop :=
  S > 10000

def commission (S : ℝ) : ℝ :=
  0.09 * S

def excess_amount (S : ℝ) : ℝ :=
  S - 10000

def additional_commission (S : ℝ) : ℝ :=
  0.03 * (S - 10000)

def total_commission (S : ℝ) : ℝ :=
  commission S + additional_commission S

-- Given the conditions
axiom total_sales_commission : ∀ S : ℝ, total_sales S → total_commission S = 1380

-- The goal is to prove the bonus
theorem bonus_received (S : ℝ) (h : total_sales S) : additional_commission S = 120 := 
by 
  sorry

end bonus_received_l156_156306


namespace abs_eq_condition_l156_156574

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l156_156574


namespace prove_expression_value_l156_156376

theorem prove_expression_value (a b c d : ℝ) (h1 : a + b = 0) (h2 : c = -1) (h3 : d = 1 ∨ d = -1) :
  2 * a + 2 * b - c * d = 1 ∨ 2 * a + 2 * b - c * d = -1 := 
by sorry

end prove_expression_value_l156_156376


namespace num_even_multiple_5_perfect_squares_lt_1000_l156_156092

theorem num_even_multiple_5_perfect_squares_lt_1000 : 
  ∃ n, n = 3 ∧ ∀ x, (x < 1000) ∧ (x > 0) ∧ (∃ k, x = 100 * k^2) → (n = 3) := by 
  sorry

end num_even_multiple_5_perfect_squares_lt_1000_l156_156092


namespace line_slope_l156_156553

theorem line_slope (x1 y1 x2 y2 : ℝ) (h1 : x1 = 0) (h2 : y1 = 100) (h3 : x2 = 50) (h4 : y2 = 300) :
  (y2 - y1) / (x2 - x1) = 4 :=
by sorry

end line_slope_l156_156553


namespace part1_part2_l156_156975

def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

theorem part1 (a : ℝ) :
  (Set.Icc (a - 7) (a - 3)) = (Set.Icc (-5 : ℝ) (-1 : ℝ)) -> a = 2 :=
by
  intro h
  sorry

theorem part2 (m : ℝ) :
  (∃ x_0 : ℝ, f x_0 2 < 4 * m + m^2) -> (m < -5 ∨ m > 1) :=
by
  intro h
  sorry

end part1_part2_l156_156975


namespace ratio_of_ages_ten_years_ago_l156_156505

theorem ratio_of_ages_ten_years_ago (A T : ℕ) 
    (h1: A = 30) 
    (h2: T = A - 15) : 
    (A - 10) / (T - 10) = 4 :=
by
  sorry

end ratio_of_ages_ten_years_ago_l156_156505


namespace color_tv_cost_l156_156122

theorem color_tv_cost (x : ℝ) (y : ℝ) (z : ℝ)
  (h1 : y = x * 1.4)
  (h2 : z = y * 0.8)
  (h3 : z = 360 + x) :
  x = 3000 :=
sorry

end color_tv_cost_l156_156122


namespace find_larger_number_l156_156386

theorem find_larger_number (x y : ℤ) (h1 : x - y = 7) (h2 : x + y = 41) : x = 24 :=
by sorry

end find_larger_number_l156_156386


namespace rachel_homework_difference_l156_156997

def pages_of_math_homework : Nat := 5
def pages_of_reading_homework : Nat := 2

theorem rachel_homework_difference : 
  pages_of_math_homework - pages_of_reading_homework = 3 :=
sorry

end rachel_homework_difference_l156_156997


namespace abs_neg_one_third_l156_156127

theorem abs_neg_one_third : abs (- (1 / 3 : ℚ)) = 1 / 3 := 
by sorry

end abs_neg_one_third_l156_156127


namespace domain_of_f_l156_156259

noncomputable def f (x : ℝ) : ℝ := (x + 6) / Real.sqrt (x^2 - 5*x + 6)

theorem domain_of_f : 
  {x : ℝ | x^2 - 5*x + 6 > 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l156_156259


namespace percentage_of_allowance_spent_l156_156279

noncomputable def amount_spent : ℝ := 14
noncomputable def amount_left : ℝ := 26
noncomputable def total_allowance : ℝ := amount_spent + amount_left

theorem percentage_of_allowance_spent :
  ((amount_spent / total_allowance) * 100) = 35 := 
by 
  sorry

end percentage_of_allowance_spent_l156_156279


namespace ellen_golf_cart_trips_l156_156374

def patrons_from_cars : ℕ := 12
def patrons_from_bus : ℕ := 27
def patrons_per_cart : ℕ := 3

theorem ellen_golf_cart_trips : (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := by
  sorry

end ellen_golf_cart_trips_l156_156374


namespace meet_time_l156_156967

theorem meet_time 
  (circumference : ℝ) 
  (deepak_speed_kmph : ℝ) 
  (wife_speed_kmph : ℝ) 
  (deepak_speed_mpm : ℝ := deepak_speed_kmph * 1000 / 60) 
  (wife_speed_mpm : ℝ := wife_speed_kmph * 1000 / 60) 
  (relative_speed : ℝ := deepak_speed_mpm + wife_speed_mpm)
  (time_to_meet : ℝ := circumference / relative_speed) :
  circumference = 660 → 
  deepak_speed_kmph = 4.5 → 
  wife_speed_kmph = 3.75 → 
  time_to_meet = 4.8 :=
by 
  intros h1 h2 h3 
  sorry

end meet_time_l156_156967


namespace problem_equivalent_proof_l156_156291

def sequence_row1 (n : ℕ) : ℤ := 2 * (-2)^(n - 1)
def sequence_row2 (n : ℕ) : ℤ := sequence_row1 n - 1
def sequence_row3 (n : ℕ) : ℤ := (-2)^n - sequence_row2 n

theorem problem_equivalent_proof :
  let a := sequence_row1 7
  let b := sequence_row2 7
  let c := sequence_row3 7
  a - b + c = -254 :=
by
  sorry

end problem_equivalent_proof_l156_156291


namespace unplanted_fraction_l156_156536

theorem unplanted_fraction (a b hypotenuse : ℕ) (side_length_P : ℚ) 
                          (h1 : a = 5) (h2 : b = 12) (h3 : hypotenuse = 13)
                          (h4 : side_length_P = 5 / 3) : 
                          (side_length_P * side_length_P) / ((a * b) / 2) = 5 / 54 :=
by
  sorry

end unplanted_fraction_l156_156536


namespace unique_two_digit_integer_l156_156882

theorem unique_two_digit_integer (s : ℕ) (hs : s > 9 ∧ s < 100) (h : 13 * s ≡ 42 [MOD 100]) : s = 34 :=
by sorry

end unique_two_digit_integer_l156_156882


namespace find_abc_squared_sum_l156_156024

theorem find_abc_squared_sum (a b c : ℕ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a^3 + 32 * b + 2 * c = 2018) (h₃ : b^3 + 32 * a + 2 * c = 1115) :
  a^2 + b^2 + c^2 = 226 :=
sorry

end find_abc_squared_sum_l156_156024


namespace dogs_remaining_end_month_l156_156106

theorem dogs_remaining_end_month :
  let initial_dogs := 200
  let dogs_arrive_w1 := 30
  let dogs_adopt_w1 := 40
  let dogs_arrive_w2 := 40
  let dogs_adopt_w2 := 50
  let dogs_arrive_w3 := 30
  let dogs_adopt_w3 := 30
  let dogs_adopt_w4 := 70
  let dogs_return_w4 := 20
  initial_dogs + (dogs_arrive_w1 - dogs_adopt_w1) + 
  (dogs_arrive_w2 - dogs_adopt_w2) +
  (dogs_arrive_w3 - dogs_adopt_w3) + 
  (-dogs_adopt_w4 - dogs_return_w4) = 90 := by
  sorry

end dogs_remaining_end_month_l156_156106


namespace value_of_a7_minus_a8_l156_156885

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem value_of_a7_minus_a8
  (h_seq: arithmetic_sequence a d)
  (h_sum: a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - a 8 = d :=
sorry

end value_of_a7_minus_a8_l156_156885


namespace overall_percentage_gain_l156_156019

theorem overall_percentage_gain
    (original_price : ℝ)
    (first_increase : ℝ)
    (first_discount : ℝ)
    (second_discount : ℝ)
    (third_discount : ℝ)
    (final_increase : ℝ)
    (final_price : ℝ)
    (overall_gain : ℝ)
    (overall_percentage_gain : ℝ)
    (h1 : original_price = 100)
    (h2 : first_increase = original_price * 1.5)
    (h3 : first_discount = first_increase * 0.9)
    (h4 : second_discount = first_discount * 0.85)
    (h5 : third_discount = second_discount * 0.8)
    (h6 : final_increase = third_discount * 1.1)
    (h7 : final_price = final_increase)
    (h8 : overall_gain = final_price - original_price)
    (h9 : overall_percentage_gain = (overall_gain / original_price) * 100) :
  overall_percentage_gain = 0.98 := by
  sorry

end overall_percentage_gain_l156_156019


namespace trapezoid_height_l156_156193

variables (a b h : ℝ)

def is_trapezoid (a b h : ℝ) (angle_diag : ℝ) (angle_ext : ℝ) : Prop :=
a < b ∧ angle_diag = 90 ∧ angle_ext = 45

theorem trapezoid_height
  (a b : ℝ) (ha : a < b)
  (angle_diag : ℝ) (h_angle_diag : angle_diag = 90)
  (angle_ext : ℝ) (h_angle_ext : angle_ext = 45)
  (h_def : is_trapezoid a b h angle_diag angle_ext) :
  h = a * b / (b - a) :=
sorry

end trapezoid_height_l156_156193


namespace cloth_cost_price_per_metre_l156_156467

theorem cloth_cost_price_per_metre (total_metres : ℕ) (total_price : ℕ) (loss_per_metre : ℕ) :
  total_metres = 300 → total_price = 18000 → loss_per_metre = 5 → (total_price / total_metres + loss_per_metre) = 65 :=
by
  intros
  sorry

end cloth_cost_price_per_metre_l156_156467


namespace quadratic_complete_square_l156_156285

theorem quadratic_complete_square :
  ∃ d e : ℝ, (∀ x : ℝ, x^2 + 800 * x + 500 = (x + d)^2 + e) ∧
    (e / d = -398.75) :=
by
  use 400
  use -159500
  sorry

end quadratic_complete_square_l156_156285


namespace determine_F_l156_156011

def f1 (x : ℝ) : ℝ := x^2 + x
def f2 (x : ℝ) : ℝ := 2 * x^2 - x
def f3 (x : ℝ) : ℝ := x^2 + x

def g1 (x : ℝ) : ℝ := x - 2
def g2 (x : ℝ) : ℝ := 2 * x
def g3 (x : ℝ) : ℝ := x + 2

def h (x : ℝ) : ℝ := x

theorem determine_F (F1 F2 F3 : ℕ) : 
  (F1 = 0 ∧ F2 = 0 ∧ F3 = 1) :=
by
  sorry

end determine_F_l156_156011


namespace greatest_odd_factors_below_200_l156_156017

theorem greatest_odd_factors_below_200 : ∃ n : ℕ, (n < 200) ∧ (n = 196) ∧ (∃ k : ℕ, n = k^2) ∧ ∀ m : ℕ, (m < 200) ∧ (∃ j : ℕ, m = j^2) → m ≤ n := by
  sorry

end greatest_odd_factors_below_200_l156_156017


namespace problem_proof_l156_156333

theorem problem_proof (A B : ℝ) (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x^2 + A)^2 + B) - (B * (A * x^2 + B)^2 + A) = A^2 - B^2) :
  A^2 + B^2 = - (A * B) := 
sorry

end problem_proof_l156_156333


namespace p_arithmetic_square_root_l156_156765

theorem p_arithmetic_square_root {p : ℕ} (hp : p ≠ 2) (a : ℤ) (ha : a ≠ 0) :
  (∃ x1 x2 : ℤ, x1^2 = a ∧ x2^2 = a ∧ x1 ≠ x2) ∨ ¬ (∃ x : ℤ, x^2 = a) :=
  sorry

end p_arithmetic_square_root_l156_156765


namespace number_of_cubes_with_icing_on_two_sides_l156_156598

def cake_cube : ℕ := 3
def smaller_cubes : ℕ := 27
def covered_faces : ℕ := 3
def layers_with_icing : ℕ := 2
def edge_cubes_per_layer_per_face : ℕ := 2

theorem number_of_cubes_with_icing_on_two_sides :
  (covered_faces * edge_cubes_per_layer_per_face * layers_with_icing) = 12 := by
  sorry

end number_of_cubes_with_icing_on_two_sides_l156_156598


namespace hair_length_correct_l156_156441

-- Define the initial hair length, the cut length, and the growth length as constants
def l_initial : ℕ := 16
def l_cut : ℕ := 11
def l_growth : ℕ := 12

-- Define the final hair length as the result of the operations described
def l_final : ℕ := l_initial - l_cut + l_growth

-- State the theorem we want to prove
theorem hair_length_correct : l_final = 17 :=
by
  sorry

end hair_length_correct_l156_156441


namespace max_sum_of_digits_of_S_l156_156287

def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def distinctDigits (n : ℕ) : Prop :=
  let digits := (n.digits 10).toFinset
  digits.card = (n.digits 10).length

def digitsRange (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → 1 ≤ d ∧ d ≤ 9

theorem max_sum_of_digits_of_S : ∃ a b S, 
  isThreeDigit a ∧ 
  isThreeDigit b ∧ 
  distinctDigits a ∧ 
  distinctDigits b ∧ 
  digitsRange a ∧ 
  digitsRange b ∧ 
  isThreeDigit S ∧ 
  S = a + b ∧ 
  (S.digits 10).sum = 12 :=
sorry

end max_sum_of_digits_of_S_l156_156287


namespace sauna_max_couples_l156_156715

def max_couples (n : ℕ) : ℕ :=
  n - 1

theorem sauna_max_couples (n : ℕ) (rooms unlimited_capacity : Prop) (no_female_male_cohabsimult : Prop)
                          (males_shared_room_constraint females_shared_room_constraint : Prop)
                          (males_known_iff_wives_known : Prop) : max_couples n = n - 1 := 
  sorry

end sauna_max_couples_l156_156715


namespace find_A_l156_156205

theorem find_A : ∃ (A : ℕ), 
  (A > 0) ∧ (A ∣ (270 * 2 - 312)) ∧ (A ∣ (211 * 2 - 270)) ∧ 
  (∃ (rA rB rC : ℕ), 312 % A = rA ∧ 270 % A = rB ∧ 211 % A = rC ∧ 
                      rA = 2 * rB ∧ rB = 2 * rC ∧ A = 19) :=
by sorry

end find_A_l156_156205


namespace quadratic_max_value_4_at_2_l156_156300

theorem quadratic_max_value_4_at_2 (a b c : ℝ) (m : ℝ)
  (h1 : ∀ x : ℝ, x ≠ 2 → (a * 2^2 + b * 2 + c) = 4)
  (h2 : a * 0^2 + b * 0 + c = -20)
  (h3 : a * 5^2 + b * 5 + c = m) :
  m = -50 :=
sorry

end quadratic_max_value_4_at_2_l156_156300


namespace sum_of_roots_l156_156099

theorem sum_of_roots 
  (a b c : ℝ)
  (h1 : 1^2 + a * 1 + 2 = 0)
  (h2 : (∀ x : ℝ, x^2 + 5 * x + c = 0 → (x = a ∨ x = b))) :
  a + b + c = 1 :=
by
  sorry

end sum_of_roots_l156_156099


namespace scientific_notation_example_l156_156183

theorem scientific_notation_example :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (3650000 : ℝ) = a * 10 ^ n :=
sorry

end scientific_notation_example_l156_156183


namespace t_n_minus_n_even_l156_156469

noncomputable def number_of_nonempty_subsets_with_integer_average (n : ℕ) : ℕ := 
  sorry

theorem t_n_minus_n_even (N : ℕ) (hN : N > 1) :
  ∃ T_n, T_n = number_of_nonempty_subsets_with_integer_average N ∧ (T_n - N) % 2 = 0 :=
by
  sorry

end t_n_minus_n_even_l156_156469


namespace total_pupils_in_school_l156_156364

theorem total_pupils_in_school (girls boys : ℕ) (h_girls : girls = 542) (h_boys : boys = 387) : girls + boys = 929 := by
  sorry

end total_pupils_in_school_l156_156364


namespace range_of_a_l156_156174

theorem range_of_a
  (x0 : ℝ) (a : ℝ)
  (hx0 : x0 > 1)
  (hineq : (x0 + 1) * Real.log x0 < a * (x0 - 1)) :
  a > 2 :=
sorry

end range_of_a_l156_156174


namespace evaluate_x2_plus_y2_plus_z2_l156_156573

theorem evaluate_x2_plus_y2_plus_z2 (x y z : ℤ) 
  (h1 : x^2 * y + y^2 * z + z^2 * x = 2186)
  (h2 : x * y^2 + y * z^2 + z * x^2 = 2188) 
  : x^2 + y^2 + z^2 = 245 := 
sorry

end evaluate_x2_plus_y2_plus_z2_l156_156573


namespace polynomial_irreducible_over_Z_iff_Q_l156_156290

theorem polynomial_irreducible_over_Z_iff_Q (f : Polynomial ℤ) :
  Irreducible f ↔ Irreducible (f.map (Int.castRingHom ℚ)) :=
sorry

end polynomial_irreducible_over_Z_iff_Q_l156_156290


namespace remainder_when_divided_by_100_l156_156921

theorem remainder_when_divided_by_100 (n : ℤ) (h : ∃ a : ℤ, n = 100 * a - 1) : 
  (n^3 + n^2 + 2 * n + 3) % 100 = 1 :=
by 
  sorry

end remainder_when_divided_by_100_l156_156921


namespace circle_range_of_t_max_radius_t_value_l156_156408

open Real

theorem circle_range_of_t {t x y : ℝ} :
  (x^2 + y^2 - 2 * (t + 3) * x + 2 * (1 - 4 * t^2) * y + 16*t^4 + 9 = 0) →
  (- (1:ℝ)/7 < t ∧ t < 1) :=
by
  sorry

theorem max_radius_t_value {t x y : ℝ} :
  (x^2 + y^2 - 2 * (t + 3) * x + 2 * (1 - 4 * t^2) * y + 16*t^4 + 9 = 0) →
  (- (1:ℝ)/7 < t ∧ t < 1) →
  (∃ r, r^2 = -7*t^2 + 6*t + 1) →
  t = 3 / 7 :=
by
  sorry

end circle_range_of_t_max_radius_t_value_l156_156408


namespace player_c_wins_l156_156710

theorem player_c_wins :
  ∀ (A_wins A_losses B_wins B_losses C_losses C_wins : ℕ),
  A_wins = 4 →
  A_losses = 2 →
  B_wins = 3 →
  B_losses = 3 →
  C_losses = 3 →
  A_wins + B_wins + C_wins = A_losses + B_losses + C_losses →
  C_wins = 2 :=
by
  intros A_wins A_losses B_wins B_losses C_losses C_wins
  sorry

end player_c_wins_l156_156710


namespace scientist_born_on_saturday_l156_156256

noncomputable def day_of_week := List String

noncomputable def calculate_day := 
  let days_in_regular_years := 113
  let days_in_leap_years := 2 * 37
  let total_days_back := days_in_regular_years + days_in_leap_years
  total_days_back % 7

theorem scientist_born_on_saturday :
  let anniversary_day := 4  -- 0=Sunday, 1=Monday, ..., 4=Thursday
  calculate_day = 5 → 
  let birth_day := (anniversary_day + 7 - calculate_day) % 7 
  birth_day = 6 := sorry

end scientist_born_on_saturday_l156_156256


namespace introduce_people_no_three_same_acquaintances_l156_156393

theorem introduce_people_no_three_same_acquaintances (n : ℕ) :
  ∃ f : ℕ → ℕ, (∀ i, i < n → f i ≤ n - 1) ∧ (∀ i j k, i < n → j < n → k < n → i ≠ j → j ≠ k → i ≠ k → ¬(f i = f j ∧ f j = f k)) := 
sorry

end introduce_people_no_three_same_acquaintances_l156_156393


namespace solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism_l156_156003

-- Definitions as conditions
def is_cone (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_cylinder (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_triangular_pyramid (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_rectangular_prism (solid : Type) : Prop := -- Definition placeholder
sorry 

-- Predicate to check if the front view of a solid is a quadrilateral
def front_view_is_quadrilateral (solid : Type) : Prop :=
  (is_cylinder solid ∨ is_rectangular_prism solid)

-- Theorem stating the problem
theorem solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism
    (s : Type) :
  front_view_is_quadrilateral s ↔ is_cylinder s ∨ is_rectangular_prism s :=
by
  sorry

end solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism_l156_156003


namespace erdos_problem_l156_156309

variable (X : Type) [Infinite X] (𝓗 : Set (Set X))
variable (h1 : ∀ (A : Set X) (hA : A.Finite), ∃ (H1 H2 : Set X) (hH1 : H1 ∈ 𝓗) (hH2 : H2 ∈ 𝓗), H1 ∩ H2 = ∅ ∧ H1 ∪ H2 = A)

theorem erdos_problem (k : ℕ) (hk : k > 0) : 
  ∃ (A : Set X) (ways : Finset (Set X × Set X)), A.Finite ∧ (∀ (p : Set X × Set X), p ∈ ways → p.1 ∈ 𝓗 ∧ p.2 ∈ 𝓗 ∧ p.1 ∩ p.2 = ∅ ∧ p.1 ∪ p.2 = A) ∧ ways.card ≥ k :=
by
  sorry

end erdos_problem_l156_156309


namespace product_of_numbers_l156_156899

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : x * y = 72 :=
by
  sorry

end product_of_numbers_l156_156899


namespace equation_zero_solution_l156_156790

-- Define the conditions and the answer
def equation_zero (x : ℝ) : Prop := (x^2 + x - 2) / (x - 1) = 0
def non_zero_denominator (x : ℝ) : Prop := x - 1 ≠ 0
def solution_x (x : ℝ) : Prop := x = -2

-- The main theorem
theorem equation_zero_solution (x : ℝ) (h1 : equation_zero x) (h2 : non_zero_denominator x) : solution_x x := 
sorry

end equation_zero_solution_l156_156790


namespace krishan_money_l156_156613

theorem krishan_money
  (R G K : ℕ)
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (h3 : R = 637) : 
  K = 3774 := 
by
  sorry

end krishan_money_l156_156613


namespace sin_square_pi_over_4_l156_156320

theorem sin_square_pi_over_4 (β : ℝ) (h : Real.sin (2 * β) = 2 / 3) : 
  Real.sin (β + π/4) ^ 2 = 5 / 6 :=
by
  sorry

end sin_square_pi_over_4_l156_156320


namespace find_number_l156_156776

theorem find_number (n : ℕ) : gcd 30 n = 10 ∧ 70 ≤ n ∧ n ≤ 80 ∧ 200 ≤ lcm 30 n ∧ lcm 30 n ≤ 300 → (n = 70 ∨ n = 80) :=
sorry

end find_number_l156_156776


namespace area_of_ground_l156_156854

def height_of_rain : ℝ := 0.05
def volume_of_water : ℝ := 750

theorem area_of_ground : ∃ A : ℝ, A = (volume_of_water / height_of_rain) ∧ A = 15000 := by
  sorry

end area_of_ground_l156_156854


namespace time_to_fill_pool_l156_156866

noncomputable def slower_pump_rate : ℝ := 1 / 12.5
noncomputable def faster_pump_rate : ℝ := 1.5 * slower_pump_rate
noncomputable def combined_rate : ℝ := slower_pump_rate + faster_pump_rate

theorem time_to_fill_pool : (1 / combined_rate) = 5 := 
by
  sorry

end time_to_fill_pool_l156_156866


namespace number_of_boys_l156_156151

theorem number_of_boys 
    (B : ℕ) 
    (total_boys_sticks : ℕ := 15 * B)
    (total_girls_sticks : ℕ := 12 * 12)
    (sticks_relation : total_girls_sticks = total_boys_sticks - 6) : 
    B = 10 :=
by
    sorry

end number_of_boys_l156_156151


namespace triangle_area_decrease_l156_156780

theorem triangle_area_decrease (B H : ℝ) : 
  let A_original := (B * H) / 2
  let H_new := 0.60 * H
  let B_new := 1.40 * B
  let A_new := (B_new * H_new) / 2
  A_new = 0.42 * A_original :=
by
  sorry

end triangle_area_decrease_l156_156780


namespace total_food_in_10_days_l156_156161

theorem total_food_in_10_days :
  (let ella_food_per_day := 20
   let days := 10
   let dog_food_ratio := 4
   let ella_total_food := ella_food_per_day * days
   let dog_total_food := dog_food_ratio * ella_total_food
   ella_total_food + dog_total_food = 1000) :=
by
  sorry

end total_food_in_10_days_l156_156161


namespace friend_gain_percentage_l156_156451

noncomputable def gain_percentage (original_cost_price sold_price_friend : ℝ) : ℝ :=
  ((sold_price_friend - (original_cost_price - 0.12 * original_cost_price)) / (original_cost_price - 0.12 * original_cost_price)) * 100

theorem friend_gain_percentage (original_cost_price sold_price_friend gain_pct : ℝ) 
  (H1 : original_cost_price = 51136.36) 
  (H2 : sold_price_friend = 54000) 
  (H3 : gain_pct = 20) : 
  gain_percentage original_cost_price sold_price_friend = gain_pct := 
by
  sorry

end friend_gain_percentage_l156_156451


namespace probability_not_within_B_l156_156862

-- Definition representing the problem context
structure Squares where
  areaA : ℝ
  areaA_pos : areaA = 65
  perimeterB : ℝ
  perimeterB_pos : perimeterB = 16

-- The theorem to be proved
theorem probability_not_within_B (s : Squares) : 
  let sideA := Real.sqrt s.areaA
  let sideB := s.perimeterB / 4
  let areaB := sideB^2
  let area_not_covered := s.areaA - areaB
  let probability := area_not_covered / s.areaA
  probability = 49 / 65 := 
by
  sorry

end probability_not_within_B_l156_156862


namespace at_least_one_div_by_5_l156_156201

-- Define natural numbers and divisibility by 5
def is_div_by_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

-- Proposition: If a, b are natural numbers and ab is divisible by 5, then at least one of a or b must be divisible by 5.
theorem at_least_one_div_by_5 (a b : ℕ) (h_ab : is_div_by_5 (a * b)) : is_div_by_5 a ∨ is_div_by_5 b :=
  by
    sorry

end at_least_one_div_by_5_l156_156201


namespace find_number_l156_156046

variable (x : ℕ)

theorem find_number (h : (10 + 20 + x) / 3 = ((10 + 40 + 25) / 3) + 5) : x = 60 :=
by
  sorry

end find_number_l156_156046


namespace football_club_initial_balance_l156_156079

noncomputable def initial_balance (final_balance income expense : ℕ) : ℕ :=
  final_balance + income - expense

theorem football_club_initial_balance :
  initial_balance 60 (2 * 10) (4 * 15) = 20 := by
sorry

end football_club_initial_balance_l156_156079


namespace trevor_pages_l156_156468

theorem trevor_pages (p1 p2 p3 : ℕ) (h1 : p1 = 72) (h2 : p2 = 72) (h3 : p3 = p1 + 4) : 
    p1 + p2 + p3 = 220 := 
by 
    sorry

end trevor_pages_l156_156468


namespace height_of_smaller_cone_is_18_l156_156895

theorem height_of_smaller_cone_is_18
  (height_frustum : ℝ)
  (area_larger_base : ℝ)
  (area_smaller_base : ℝ) :
  let R := (area_larger_base / π).sqrt
  let r := (area_smaller_base / π).sqrt
  let ratio := r / R
  let H := height_frustum / (1 - ratio)
  let h := ratio * H
  height_frustum = 18 ∧ area_larger_base = 400 * π ∧ area_smaller_base = 100 * π
  → h = 18 := by
  sorry

end height_of_smaller_cone_is_18_l156_156895


namespace smallest_number_of_butterflies_l156_156246

theorem smallest_number_of_butterflies 
  (identical_groups : ℕ) 
  (groups_of_butterflies : ℕ) 
  (groups_of_fireflies : ℕ) 
  (groups_of_ladybugs : ℕ)
  (h1 : groups_of_butterflies = 44)
  (h2 : groups_of_fireflies = 17)
  (h3 : groups_of_ladybugs = 25)
  (h4 : identical_groups * (groups_of_butterflies + groups_of_fireflies + groups_of_ladybugs) % 60 = 0) :
  identical_groups * groups_of_butterflies = 425 :=
sorry

end smallest_number_of_butterflies_l156_156246


namespace vector_parallel_solution_l156_156690

theorem vector_parallel_solution (x : ℝ) : 
  let a := (2, 3)
  let b := (x, -9)
  (a.snd = 3) → (a.fst = 2) → (b.snd = -9) → (a.fst * b.snd = a.snd * (b.fst)) → x = -6 := 
by
  intros 
  sorry

end vector_parallel_solution_l156_156690


namespace paul_mowing_lawns_l156_156084

theorem paul_mowing_lawns : 
  ∃ M : ℕ, 
    (∃ money_made_weeating : ℕ, money_made_weeating = 13) ∧
    (∃ spending_per_week : ℕ, spending_per_week = 9) ∧
    (∃ weeks_last : ℕ, weeks_last = 9) ∧
    (M + 13 = 9 * 9) → 
    M = 68 := by
sorry

end paul_mowing_lawns_l156_156084


namespace steves_earning_l156_156311

variable (pounds_picked : ℕ → ℕ) -- pounds picked on day i: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday

def payment_per_pound : ℕ := 2

def total_money_made : ℕ :=
  (pounds_picked 0 * payment_per_pound) + 
  (pounds_picked 1 * payment_per_pound) + 
  (pounds_picked 2 * payment_per_pound) + 
  (pounds_picked 3 * payment_per_pound)

theorem steves_earning 
  (h0 : pounds_picked 0 = 8)
  (h1 : pounds_picked 1 = 3 * pounds_picked 0)
  (h2 : pounds_picked 2 = 0)
  (h3 : pounds_picked 3 = 18) : 
  total_money_made pounds_picked = 100 := by
  sorry

end steves_earning_l156_156311


namespace count_perfect_square_factors_of_360_l156_156946

def is_prime_fact_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_perfect_square (d : ℕ) : Prop :=
  ∃ a b c : ℕ, d = 2^(2*a) * 3^(2*b) * 5^(2*c)

def prime_factorization_360 : Prop :=
  ∀ d : ℕ, d ∣ 360 → is_perfect_square d

theorem count_perfect_square_factors_of_360 : ∃ count : ℕ, count = 4 :=
  sorry

end count_perfect_square_factors_of_360_l156_156946


namespace smaller_angle_at_3_45_l156_156112

def minute_hand_angle : ℝ := 270
def hour_hand_angle : ℝ := 90 + 0.75 * 30

theorem smaller_angle_at_3_45 :
  min (|minute_hand_angle - hour_hand_angle|) (360 - |minute_hand_angle - hour_hand_angle|) = 202.5 := 
by
  sorry

end smaller_angle_at_3_45_l156_156112


namespace calculate_fraction_l156_156130

def x : ℚ := 2 / 3
def y : ℚ := 8 / 10

theorem calculate_fraction :
  (6 * x + 10 * y) / (60 * x * y) = 3 / 8 := by
  sorry

end calculate_fraction_l156_156130


namespace median_price_l156_156892

-- Definitions from conditions
def price1 : ℝ := 10
def price2 : ℝ := 12
def price3 : ℝ := 15

def sales1 : ℝ := 0.50
def sales2 : ℝ := 0.30
def sales3 : ℝ := 0.20

-- Statement of the problem
theorem median_price : (price1 * sales1 + price2 * sales2 + price3 * sales3) / 2 = 11 := by
  sorry

end median_price_l156_156892


namespace range_of_m_l156_156071

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 + 2 * x + m > 0) → m > 1 :=
by
  -- Proof goes here
  sorry

end range_of_m_l156_156071


namespace rate_percent_l156_156711

theorem rate_percent (SI P T: ℝ) (h₁: SI = 250) (h₂: P = 1500) (h₃: T = 5) : 
  ∃ R : ℝ, R = (SI * 100) / (P * T) := 
by
  use (250 * 100) / (1500 * 5)
  sorry

end rate_percent_l156_156711


namespace initial_paint_l156_156909

variable (total_needed : ℕ) (paint_bought : ℕ) (still_needed : ℕ)

theorem initial_paint (h_total_needed : total_needed = 70)
                      (h_paint_bought : paint_bought = 23)
                      (h_still_needed : still_needed = 11) : 
                      ∃ x : ℕ, x = 36 :=
by
  sorry

end initial_paint_l156_156909


namespace yan_distance_ratio_l156_156284

-- Define conditions
variable (x z w: ℝ)  -- x: distance from Yan to his home, z: distance from Yan to the school, w: Yan's walking speed
variable (h1: z / w = x / w + (x + z) / (5 * w))  -- Both choices require the same amount of time

-- The ratio of Yan's distance from his home to his distance from the school is 2/3
theorem yan_distance_ratio :
    x / z = 2 / 3 :=
by
  sorry

end yan_distance_ratio_l156_156284


namespace smallest_possible_c_minus_a_l156_156022

theorem smallest_possible_c_minus_a :
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a * b * c = Nat.factorial 9 ∧ c - a = 216 := 
by
  sorry

end smallest_possible_c_minus_a_l156_156022


namespace selling_price_per_unit_profit_per_unit_after_discount_l156_156815

-- Define the initial cost per unit
variable (a : ℝ)

-- Problem statement for part 1: Selling price per unit is 1.22a yuan
theorem selling_price_per_unit (a : ℝ) : 1.22 * a = a + 0.22 * a :=
by
  sorry

-- Problem statement for part 2: Profit per unit after 15% discount is still 0.037a yuan
theorem profit_per_unit_after_discount (a : ℝ) : 
  (1.22 * a * 0.85) - a = 0.037 * a :=
by
  sorry

end selling_price_per_unit_profit_per_unit_after_discount_l156_156815


namespace volume_of_snow_l156_156119

theorem volume_of_snow (L W H : ℝ) (hL : L = 30) (hW : W = 3) (hH : H = 0.75) :
  L * W * H = 67.5 := by
  sorry

end volume_of_snow_l156_156119


namespace candy_problem_l156_156298

-- Define conditions and the statement
theorem candy_problem (K : ℕ) (h1 : 49 = K + 3 * K + 8 + 6 + 10 + 5) : K = 5 :=
sorry

end candy_problem_l156_156298


namespace hyperbola_distance_condition_l156_156906

open Real

theorem hyperbola_distance_condition (a b c x: ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
    (h_dist : abs (b^4 / a^2 / (a - c)) < a + sqrt (a^2 + b^2)) :
    0 < b / a ∧ b / a < 1 :=
by
  sorry

end hyperbola_distance_condition_l156_156906


namespace solve_equation1_solve_equation2_l156_156008

-- Define the first equation and state the theorem that proves its roots
def equation1 (x : ℝ) : Prop := 2 * x^2 + 1 = 3 * x

theorem solve_equation1 (x : ℝ) : equation1 x ↔ (x = 1 ∨ x = 1/2) :=
by sorry

-- Define the second equation and state the theorem that proves its roots
def equation2 (x : ℝ) : Prop := (2 * x - 1)^2 = (3 - x)^2

theorem solve_equation2 (x : ℝ) : equation2 x ↔ (x = -2 ∨ x = 4 / 3) :=
by sorry

end solve_equation1_solve_equation2_l156_156008


namespace geometric_sequence_y_l156_156943

theorem geometric_sequence_y (x y z : ℝ) (h1 : 1 ≠ 0) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : z ≠ 0) (h5 : 9 ≠ 0)
  (h_seq : ∀ a b c d e : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ a * e = b * d ∧ b * d = c^2) →
           (a, b, c, d, e) = (1, x, y, z, 9)) :
  y = 3 :=
sorry

end geometric_sequence_y_l156_156943


namespace robert_turns_30_after_2_years_l156_156741

variable (P R : ℕ) -- P for Patrick's age, R for Robert's age
variable (h1 : P = 14) -- Patrick is 14 years old now
variable (h2 : P * 2 = R) -- Patrick is half the age of Robert

theorem robert_turns_30_after_2_years : R + 2 = 30 :=
by
  -- Here should be the proof, but for now we skip it with sorry
  sorry

end robert_turns_30_after_2_years_l156_156741


namespace fireflies_remaining_l156_156360

theorem fireflies_remaining
  (initial_fireflies : ℕ)
  (fireflies_joined : ℕ)
  (fireflies_flew_away : ℕ)
  (h_initial : initial_fireflies = 3)
  (h_joined : fireflies_joined = 12 - 4)
  (h_flew_away : fireflies_flew_away = 2)
  : initial_fireflies + fireflies_joined - fireflies_flew_away = 9 := by
  sorry

end fireflies_remaining_l156_156360


namespace part1_part2_l156_156211

noncomputable def problem1 (x y: ℕ) : Prop := 
  (2 * x + 3 * y = 44) ∧ (4 * x = 5 * y)

noncomputable def solution1 (x y: ℕ) : Prop :=
  (x = 10) ∧ (y = 8)

theorem part1 : ∃ x y: ℕ, problem1 x y → solution1 x y :=
by sorry

noncomputable def problem2 (a b: ℕ) : Prop := 
  25 * (10 * a + 8 * b) = 3500

noncomputable def solution2 (a b: ℕ) : Prop :=
  ((a = 2 ∧ b = 15) ∨ (a = 6 ∧ b = 10) ∨ (a = 10 ∧ b = 5))

theorem part2 : ∃ a b: ℕ, problem2 a b → solution2 a b :=
by sorry

end part1_part2_l156_156211


namespace combinations_count_l156_156196

def colorChoices := 4
def decorationChoices := 3
def methodChoices := 3

theorem combinations_count : colorChoices * decorationChoices * methodChoices = 36 := by
  sorry

end combinations_count_l156_156196


namespace tank_third_dimension_l156_156717

theorem tank_third_dimension (x : ℕ) (h1 : 4 * 5 = 20) (h2 : 2 * (4 * x) + 2 * (5 * x) = 18 * x) (h3 : (40 + 18 * x) * 20 = 1520) :
  x = 2 :=
by
  sorry

end tank_third_dimension_l156_156717


namespace ant_rest_position_l156_156722

noncomputable def percent_way_B_to_C (s : ℕ) : ℕ :=
  let perimeter := 3 * s
  let distance_traveled := (42 * perimeter) / 100
  let distance_AB := s
  let remaining_distance := distance_traveled - distance_AB
  (remaining_distance * 100) / s

theorem ant_rest_position :
  ∀ (s : ℕ), percent_way_B_to_C s = 26 :=
by
  intros
  unfold percent_way_B_to_C
  sorry

end ant_rest_position_l156_156722


namespace necessary_condition_for_line_passes_quadrants_l156_156343

theorem necessary_condition_for_line_passes_quadrants (m n : ℝ) (h_line : ∀ x : ℝ, x * (m / n) - (1 / n) < 0 ∨ x * (m / n) - (1 / n) > 0) : m * n < 0 :=
by
  sorry

end necessary_condition_for_line_passes_quadrants_l156_156343


namespace solution_of_equation_l156_156825

theorem solution_of_equation (x : ℤ) : 7 * x - 5 = 6 * x → x = 5 := by
  intro h
  sorry

end solution_of_equation_l156_156825


namespace c_left_days_before_completion_l156_156582

-- Definitions for the given conditions
def work_done_by_a_in_one_day := 1 / 30
def work_done_by_b_in_one_day := 1 / 30
def work_done_by_c_in_one_day := 1 / 40
def total_days := 12

-- Proof problem statement (to prove that c left 8 days before the completion)
theorem c_left_days_before_completion :
  ∃ x : ℝ, 
  (12 - x) * (7 / 60) + x * (1 / 15) = 1 → 
  x = 8 := sorry

end c_left_days_before_completion_l156_156582


namespace professor_has_to_grade_405_more_problems_l156_156760

theorem professor_has_to_grade_405_more_problems
  (problems_per_paper : ℕ)
  (total_papers : ℕ)
  (graded_papers : ℕ)
  (remaining_papers := total_papers - graded_papers)
  (p : ℕ := remaining_papers * problems_per_paper) :
  problems_per_paper = 15 ∧ total_papers = 45 ∧ graded_papers = 18 → p = 405 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end professor_has_to_grade_405_more_problems_l156_156760


namespace abs_inequality_solution_rational_inequality_solution_l156_156779

theorem abs_inequality_solution (x : ℝ) : (|x - 2| + |2 * x - 3| < 4) ↔ (1 / 3 < x ∧ x < 3) :=
sorry

theorem rational_inequality_solution (x : ℝ) : 
  (x^2 - 3 * x) / (x^2 - x - 2) ≤ x ↔ (x ∈ Set.Icc (-1) 0 ∪ {1} ∪ Set.Ioi 2) := 
sorry

#check abs_inequality_solution
#check rational_inequality_solution

end abs_inequality_solution_rational_inequality_solution_l156_156779


namespace extreme_value_f_g_gt_one_l156_156679

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.exp x * (a * x + x * Real.cos x + 1)

theorem extreme_value_f : f 0 = 0 :=
by
  sorry

theorem g_gt_one (a : ℝ) (h : a > -1) (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : g x a > 1 :=
by
  sorry

end extreme_value_f_g_gt_one_l156_156679


namespace carla_sharpening_time_l156_156829

theorem carla_sharpening_time (x : ℕ) (h : x + 3 * x = 40) : x = 10 :=
by
  sorry

end carla_sharpening_time_l156_156829


namespace difference_of_numbers_l156_156065

noncomputable def larger_num : ℕ := 1495
noncomputable def quotient : ℕ := 5
noncomputable def remainder : ℕ := 4

theorem difference_of_numbers :
  ∃ S : ℕ, larger_num = quotient * S + remainder ∧ (larger_num - S = 1197) :=
by 
  sorry

end difference_of_numbers_l156_156065


namespace sum_of_roots_l156_156490

theorem sum_of_roots : (x₁ x₂ : ℝ) → (h : 2 * x₁^2 + 6 * x₁ - 1 = 0) → (h₂ : 2 * x₂^2 + 6 * x₂ - 1 = 0) → x₁ + x₂ = -3 :=
by 
  sorry

end sum_of_roots_l156_156490


namespace product_of_last_two_digits_l156_156499

theorem product_of_last_two_digits (A B : ℕ) (h1 : B = 0 ∨ B = 5) (h2 : A + B = 12) : A * B = 35 :=
by {
  -- proof omitted
  sorry
}

end product_of_last_two_digits_l156_156499


namespace expression_divisible_by_24_l156_156447

theorem expression_divisible_by_24 (n : ℕ) (hn : 0 < n) : ∃ k : ℕ, (n + 7)^2 - (n - 5)^2 = 24 * k := by
  sorry

end expression_divisible_by_24_l156_156447


namespace sum_of_f10_values_l156_156362

noncomputable def f : ℕ → ℝ := sorry

axiom f_cond1 : f 1 = 4

axiom f_cond2 : ∀ (m n : ℕ), m ≥ n → f (m + n) + f (m - n) = (f (2 * m) + f (2 * n)) / 2

theorem sum_of_f10_values : f 10 = 400 :=
sorry

end sum_of_f10_values_l156_156362


namespace male_students_plant_trees_l156_156808

theorem male_students_plant_trees (total_avg : ℕ) (female_trees : ℕ) (male_trees : ℕ) 
  (h1 : total_avg = 6) 
  (h2 : female_trees = 15)
  (h3 : 1 / (male_trees : ℝ) + 1 / (female_trees : ℝ) = 1 / (total_avg : ℝ)) : 
  male_trees = 10 := 
sorry

end male_students_plant_trees_l156_156808


namespace volume_formula_l156_156842

noncomputable def volume_of_parallelepiped
  (a b : ℝ) (h : ℝ) (θ : ℝ) 
  (θ_eq : θ = Real.pi / 3) 
  (base_diagonal : ℝ)
  (base_diagonal_eq : base_diagonal = Real.sqrt (a ^ 2 + b ^ 2)) : ℝ :=
  a * b * h 

theorem volume_formula 
  (a b : ℝ) (h : ℝ) (θ : ℝ)
  (area_base : ℝ) 
  (area_of_base_eq : area_base = a * b) 
  (θ_eq : θ = Real.pi / 3) 
  (base_diagonal : ℝ) 
  (base_diagonal_eq : base_diagonal = Real.sqrt (a ^ 2 + b ^ 2))
  (height_eq : h = (base_diagonal / 2) * (Real.sqrt 3)): 
  volume_of_parallelepiped a b h θ θ_eq base_diagonal base_diagonal_eq 
  = (144 * Real.sqrt 3) / 5 :=
by {
  sorry
}

end volume_formula_l156_156842


namespace relationship_between_vars_l156_156229

-- Define the variables a, b, c, d as real numbers
variables (a b c d : ℝ)

-- Define the initial condition
def initial_condition := (a + 2 * b) / (2 * b + c) = (c + 2 * d) / (2 * d + a)

-- State the theorem to be proved
theorem relationship_between_vars (h : initial_condition a b c d) : 
  a = c ∨ a + c + 2 * (b + d) = 0 :=
sorry

end relationship_between_vars_l156_156229


namespace measure_8_liters_possible_l156_156429

-- Define the types for buckets
structure Bucket :=
  (capacity : ℕ)
  (water : ℕ := 0)

-- Initial state with a 10-liter bucket and a 6-liter bucket, both empty
def B10_init := Bucket.mk 10 0
def B6_init := Bucket.mk 6 0

-- Define a function to check if we can measure 8 liters in B10
def can_measure_8_liters (B10 B6 : Bucket) : Prop :=
  (B10.water = 8 ∧ B10.capacity = 10 ∧ B6.capacity = 6)

-- The statement to prove there exists a sequence of operations to measure 8 liters in B10
theorem measure_8_liters_possible : ∃ (B10 B6 : Bucket), can_measure_8_liters B10 B6 :=
by
  -- Proof omitted
  sorry

end measure_8_liters_possible_l156_156429


namespace toy_cost_price_l156_156697

theorem toy_cost_price (C : ℕ) (h : 18 * C + 3 * C = 25200) : C = 1200 := by
  -- The proof is not required
  sorry

end toy_cost_price_l156_156697


namespace quadratic_roots_properties_quadratic_roots_max_min_l156_156726

theorem quadratic_roots_properties (k : ℝ) (h : 2 ≤ k ∧ k ≤ 8)
  (x1 x2 : ℝ) (h_roots : x1 + x2 = 2 * (k - 1) ∧ x1 * x2 = 2 * k^2 - 12 * k + 17) :
  (x1^2 + x2^2) = 16 * k - 30 :=
sorry

theorem quadratic_roots_max_min :
  (∀ k ∈ { k : ℝ | 2 ≤ k ∧ k ≤ 8 }, 
    ∃ (x1 x2 : ℝ), 
      (x1 + x2 = 2 * (k - 1) ∧ x1 * x2 = 2 * k^2 - 12 * k + 17) 
      ∧ (x1^2 + x2^2) = (if k = 8 then 98 else if k = 2 then 2 else 16 * k - 30)) :=
sorry

end quadratic_roots_properties_quadratic_roots_max_min_l156_156726


namespace work_combined_days_l156_156625

theorem work_combined_days (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 12) (hC : C = 1 / 6) :
  1 / (A + B + C) = 2 :=
by
  sorry

end work_combined_days_l156_156625


namespace find_x_y_n_l156_156237

def is_reverse_digit (x y : ℕ) : Prop := 
  x / 10 = y % 10 ∧ x % 10 = y / 10

def is_two_digit_nonzero (z : ℕ) : Prop := 
  10 ≤ z ∧ z < 100

theorem find_x_y_n : 
  ∃ (x y n : ℕ), is_two_digit_nonzero x ∧ is_two_digit_nonzero y ∧ is_reverse_digit x y ∧ (x^2 - y^2 = 44 * n) ∧ (x + y + n = 93) :=
sorry

end find_x_y_n_l156_156237


namespace point_A_coordinates_l156_156600

variable {a : ℝ}
variable {f : ℝ → ℝ}

theorem point_A_coordinates (h1 : a > 0) (h2 : a ≠ 1) (hf : ∀ x, f x = a^(x - 1)) :
  f 1 = 1 :=
by
  sorry

end point_A_coordinates_l156_156600


namespace algebraic_identity_l156_156713

theorem algebraic_identity (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) :
    a^2 - b^2 = -8 := by
  sorry

end algebraic_identity_l156_156713


namespace other_endpoint_l156_156996

theorem other_endpoint (M : ℝ × ℝ) (A : ℝ × ℝ) (x y : ℝ) :
  M = (2, 3) ∧ A = (5, -1) ∧ (M = ((A.1 + x) / 2, (A.2 + y) / 2)) → (x, y) = (-1, 7) := by
  sorry

end other_endpoint_l156_156996


namespace find_vidya_age_l156_156023

theorem find_vidya_age (V M : ℕ) (h1: M = 3 * V + 5) (h2: M = 44) : V = 13 :=
by {
  sorry
}

end find_vidya_age_l156_156023


namespace total_journey_distance_l156_156081

-- Definitions of the conditions

def journey_time : ℝ := 40
def first_half_speed : ℝ := 20
def second_half_speed : ℝ := 30

-- Proof statement
theorem total_journey_distance : ∃ D : ℝ, (D / first_half_speed + D / second_half_speed = journey_time) ∧ (D = 960) :=
by 
  sorry

end total_journey_distance_l156_156081


namespace imo_34_l156_156768

-- Define the input conditions
variables (R r ρ : ℝ)

-- The main theorem we need to prove
theorem imo_34 { R r ρ : ℝ } (hR : R = 1) : 
  ρ ≤ 1 - (1/3) * (1 + r)^2 :=
sorry

end imo_34_l156_156768


namespace find_f_2000_l156_156149

noncomputable def f : ℝ → ℝ := sorry

axiom f_property1 : ∀ (x y : ℝ), f (x + y) = f (x * y)
axiom f_property2 : f (-1/2) = -1/2

theorem find_f_2000 : f 2000 = -1/2 := 
sorry

end find_f_2000_l156_156149


namespace pyramid_z_value_l156_156156

-- Define the conditions and the proof problem
theorem pyramid_z_value {z x y : ℕ} :
  (x = z * y) →
  (8 = z * x) →
  (40 = x * y) →
  (10 = y * x) →
  z = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end pyramid_z_value_l156_156156


namespace snacks_displayed_at_dawn_l156_156757

variable (S : ℝ)
variable (SoldMorning : ℝ)
variable (SoldAfternoon : ℝ)

axiom cond1 : SoldMorning = (3 / 5) * S
axiom cond2 : SoldAfternoon = 180
axiom cond3 : SoldMorning = SoldAfternoon

theorem snacks_displayed_at_dawn : S = 300 :=
by
  sorry

end snacks_displayed_at_dawn_l156_156757


namespace tan_shift_monotonic_interval_l156_156699

noncomputable def monotonic_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | k * Real.pi - 3 * Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 4}

theorem tan_shift_monotonic_interval {k : ℤ} :
  ∀ x, (monotonic_interval k x) → (Real.tan (x + Real.pi / 4)) = (Real.tan x) := sorry

end tan_shift_monotonic_interval_l156_156699


namespace marco_older_than_twice_marie_l156_156923

variable (M m x : ℕ)

def marie_age : ℕ := 12
def sum_of_ages : ℕ := 37

theorem marco_older_than_twice_marie :
  m = marie_age → (M = 2 * m + x) → (M + m = sum_of_ages) → x = 1 :=
by
  intros h1 h2 h3
  rw [h1] at h2 h3
  sorry

end marco_older_than_twice_marie_l156_156923


namespace domain_of_sqrt_cos_minus_half_correct_l156_156365

noncomputable def domain_of_sqrt_cos_minus_half (x : ℝ) : Prop :=
  ∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3

theorem domain_of_sqrt_cos_minus_half_correct :
  ∀ x, (∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3) ↔
    ∃ (k : ℤ), 2 * (k : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * (k : ℝ) * Real.pi + Real.pi / 3 :=
by sorry

end domain_of_sqrt_cos_minus_half_correct_l156_156365


namespace decrease_neg_of_odd_and_decrease_nonneg_l156_156551

-- Define the properties of the function f
variable (f : ℝ → ℝ)

-- f is odd
def odd_function : Prop := ∀ x : ℝ, f (-x) = - f x

-- f is decreasing on [0, +∞)
def decreasing_on_nonneg : Prop := ∀ x1 x2 : ℝ, (0 ≤ x1) → (0 ≤ x2) → (x1 < x2 → f x1 > f x2)

-- Goal: f is decreasing on (-∞, 0)
def decreasing_on_neg : Prop := ∀ x1 x2 : ℝ, (x1 < 0) → (x2 < 0) → (x1 < x2) → f x1 > f x2

-- The theorem to be proved
theorem decrease_neg_of_odd_and_decrease_nonneg 
  (h_odd : odd_function f) (h_decreasing_nonneg : decreasing_on_nonneg f) :
  decreasing_on_neg f :=
sorry

end decrease_neg_of_odd_and_decrease_nonneg_l156_156551


namespace team_A_win_probability_l156_156302

theorem team_A_win_probability :
  let win_prob := (1 / 3 : ℝ)
  let team_A_lead := 2
  let total_sets := 5
  let require_wins := 3
  let remaining_sets := total_sets - team_A_lead
  let prob_team_B_win_remaining := (1 - win_prob) ^ remaining_sets
  let prob_team_A_win := 1 - prob_team_B_win_remaining
  prob_team_A_win = 19 / 27 := by
    sorry

end team_A_win_probability_l156_156302


namespace largest_integer_inequality_l156_156332

theorem largest_integer_inequality (x : ℤ) (h : 10 - 3 * x > 25) : x = -6 :=
sorry

end largest_integer_inequality_l156_156332


namespace gray_eyed_black_haired_students_l156_156608

theorem gray_eyed_black_haired_students :
  ∀ (students : ℕ)
    (green_eyed_red_haired : ℕ)
    (black_haired : ℕ)
    (gray_eyed : ℕ),
    students = 60 →
    green_eyed_red_haired = 20 →
    black_haired = 40 →
    gray_eyed = 25 →
    (gray_eyed - (students - black_haired - green_eyed_red_haired)) = 25 := by
  intros students green_eyed_red_haired black_haired gray_eyed
  intros h_students h_green h_black h_gray
  sorry

end gray_eyed_black_haired_students_l156_156608


namespace expression_evaluation_l156_156006

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end expression_evaluation_l156_156006


namespace simplify_polynomial_l156_156935

theorem simplify_polynomial (x : ℤ) :
  (3 * x - 2) * (6 * x^12 + 3 * x^11 + 5 * x^9 + x^8 + 7 * x^7) =
  18 * x^13 - 3 * x^12 + 15 * x^10 - 7 * x^9 + 19 * x^8 - 14 * x^7 :=
by
  sorry

end simplify_polynomial_l156_156935


namespace kevin_trip_distance_l156_156158

theorem kevin_trip_distance :
  let D := 600
  (∃ T : ℕ, D = 50 * T ∧ D = 75 * (T - 4)) := 
sorry

end kevin_trip_distance_l156_156158


namespace circle_equation_l156_156972

theorem circle_equation :
  ∃ (h k r : ℝ), 
    (∀ (x y : ℝ), (x, y) = (-6, 2) → (x - h)^2 + (y - k)^2 = r^2)
    ∧ (∀ (x y : ℝ), (x, y) = (2, -2) → (x - h)^2 + (y - k)^2 = r^2)
    ∧ r = 5
    ∧ h - k = -1
    ∧ (x + 3)^2 + (y + 2)^2 = 25 :=
by
  sorry

end circle_equation_l156_156972


namespace cat_food_sufficiency_l156_156482

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l156_156482


namespace flea_can_visit_all_points_l156_156394

def flea_maximum_jump (max_point : ℕ) : ℕ :=
  1006

theorem flea_can_visit_all_points (n : ℕ) (max_point : ℕ) (h_nonneg_max_point : 0 ≤ max_point) (h_segment : max_point = 2013) :
  n ≤ flea_maximum_jump max_point :=
by
  sorry

end flea_can_visit_all_points_l156_156394


namespace train_stoppage_time_l156_156799

-- Definitions of the conditions
def speed_excluding_stoppages : ℝ := 48 -- in kmph
def speed_including_stoppages : ℝ := 32 -- in kmph
def time_per_hour : ℝ := 60 -- 60 minutes in an hour

-- The problem statement
theorem train_stoppage_time :
  (speed_excluding_stoppages - speed_including_stoppages) * time_per_hour / speed_excluding_stoppages = 20 :=
by
  -- Initial statement
  sorry

end train_stoppage_time_l156_156799


namespace green_disks_more_than_blue_l156_156353

theorem green_disks_more_than_blue 
  (total_disks : ℕ) (blue_ratio yellow_ratio green_ratio red_ratio : ℕ)
  (h1 : total_disks = 132)
  (h2 : blue_ratio = 3)
  (h3 : yellow_ratio = 7)
  (h4 : green_ratio = 8)
  (h5 : red_ratio = 4)
  : 6 * green_ratio - 6 * blue_ratio = 30 :=
by
  sorry

end green_disks_more_than_blue_l156_156353


namespace arithmetic_sequence_a13_l156_156223

variable (a1 d : ℤ)

theorem arithmetic_sequence_a13 (h : a1 + 2 * d + a1 + 8 * d + a1 + 26 * d = 12) : a1 + 12 * d = 4 :=
by
  sorry

end arithmetic_sequence_a13_l156_156223


namespace range_of_a_for_common_tangents_l156_156197

theorem range_of_a_for_common_tangents :
  ∃ (a : ℝ), ∀ (x y : ℝ),
    ((x - 2)^2 + y^2 = 4) ∧ ((x - a)^2 + (y + 3)^2 = 9) →
    (-2 < a) ∧ (a < 6) := by
  sorry

end range_of_a_for_common_tangents_l156_156197


namespace contrapositive_statement_l156_156971

theorem contrapositive_statement (m : ℝ) (h : ¬ ∃ x : ℝ, x^2 = m) : m < 0 :=
sorry

end contrapositive_statement_l156_156971


namespace factorize_3x_squared_minus_7x_minus_6_factorize_6x_squared_minus_7x_minus_5_l156_156242

theorem factorize_3x_squared_minus_7x_minus_6 (x : ℝ) :
  3 * x^2 - 7 * x - 6 = (x - 3) * (3 * x + 2) :=
sorry

theorem factorize_6x_squared_minus_7x_minus_5 (x : ℝ) :
  6 * x^2 - 7 * x - 5 = (2 * x + 1) * (3 * x - 5) :=
sorry

end factorize_3x_squared_minus_7x_minus_6_factorize_6x_squared_minus_7x_minus_5_l156_156242


namespace lucas_total_pages_l156_156723

-- Define the variables and conditions
def lucas_read_pages : Nat :=
  let pages_first_four_days := 4 * 20
  let pages_break_day := 0
  let pages_next_four_days := 4 * 30
  let pages_last_day := 15
  pages_first_four_days + pages_break_day + pages_next_four_days + pages_last_day

-- State the theorem
theorem lucas_total_pages :
  lucas_read_pages = 215 :=
sorry

end lucas_total_pages_l156_156723


namespace sum_of_three_terms_divisible_by_3_l156_156738

theorem sum_of_three_terms_divisible_by_3 (a : Fin 5 → ℤ) :
  ∃ (i j k : Fin 5), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ (a i + a j + a k) % 3 = 0 :=
by
  sorry

end sum_of_three_terms_divisible_by_3_l156_156738


namespace remainder_when_divided_by_6_l156_156901

theorem remainder_when_divided_by_6 :
  ∃ (n : ℕ), (∃ k : ℕ, n = 3 * k + 2 ∧ ∃ m : ℕ, k = 4 * m + 3) → n % 6 = 5 :=
by
  sorry

end remainder_when_divided_by_6_l156_156901


namespace original_total_thumbtacks_l156_156602

-- Conditions
def num_cans : ℕ := 3
def num_boards_tested : ℕ := 120
def thumbtacks_per_board : ℕ := 3
def thumbtacks_remaining_per_can : ℕ := 30

-- Question
theorem original_total_thumbtacks :
  (num_cans * num_boards_tested * thumbtacks_per_board) + (num_cans * thumbtacks_remaining_per_can) = 450 :=
sorry

end original_total_thumbtacks_l156_156602


namespace closest_integer_to_2_plus_sqrt_6_l156_156847

theorem closest_integer_to_2_plus_sqrt_6 (sqrt6_lower : 2 < Real.sqrt 6) (sqrt6_upper : Real.sqrt 6 < 2.5) : 
  abs (2 + Real.sqrt 6 - 4) < abs (2 + Real.sqrt 6 - 3) ∧ abs (2 + Real.sqrt 6 - 4) < abs (2 + Real.sqrt 6 - 5) :=
by
  sorry

end closest_integer_to_2_plus_sqrt_6_l156_156847


namespace rain_probability_l156_156416

/-
Theorem: Given that the probability it will rain on Monday is 40%
and the probability it will rain on Tuesday is 30%, and the probability of
rain on a given day is independent of the weather on any other day,
the probability it will rain on both Monday and Tuesday is 12%.
-/
theorem rain_probability (p_monday : ℝ) (p_tuesday : ℝ) (independent : Prop) :
  p_monday = 0.4 ∧ p_tuesday = 0.3 ∧ independent → (p_monday * p_tuesday) * 100 = 12 :=
by sorry

end rain_probability_l156_156416


namespace volume_of_cube_l156_156684

theorem volume_of_cube (SA : ℝ) (H : SA = 600) : (10^3 : ℝ) = 1000 :=
by
  sorry

end volume_of_cube_l156_156684


namespace shaded_region_area_correct_l156_156154

noncomputable def hexagon_side : ℝ := 4
noncomputable def major_axis : ℝ := 4
noncomputable def minor_axis : ℝ := 2

noncomputable def hexagon_area := (3 * Real.sqrt 3 / 2) * hexagon_side^2

noncomputable def semi_ellipse_area : ℝ :=
  (1 / 2) * Real.pi * major_axis * minor_axis

noncomputable def total_semi_ellipse_area := 4 * semi_ellipse_area 

noncomputable def shaded_region_area := hexagon_area - total_semi_ellipse_area

theorem shaded_region_area_correct : shaded_region_area = 48 * Real.sqrt 3 - 16 * Real.pi :=
by
  sorry

end shaded_region_area_correct_l156_156154


namespace oliver_remaining_dishes_l156_156668

def num_dishes := 36
def dishes_with_mango_salsa := 3
def dishes_with_fresh_mango := num_dishes / 6
def dishes_with_mango_jelly := 1
def oliver_picks_two := 2

theorem oliver_remaining_dishes : 
  num_dishes - (dishes_with_mango_salsa + dishes_with_fresh_mango + dishes_with_mango_jelly - oliver_picks_two) = 28 := by
  sorry

end oliver_remaining_dishes_l156_156668


namespace sufficient_condition_for_lg_m_lt_1_l156_156601

theorem sufficient_condition_for_lg_m_lt_1 (m : ℝ) (h1 : m ∈ ({1, 2} : Set ℝ)) : Real.log m < 1 :=
sorry

end sufficient_condition_for_lg_m_lt_1_l156_156601


namespace find_a_plus_b_l156_156479

theorem find_a_plus_b (a b : ℤ) (h : 2*x^3 - a*x^2 - 5*x + 5 = (2*x^2 + a*x - 1)*(x - b) + 3) : a + b = 4 :=
by {
  -- Proof omitted
  sorry
}

end find_a_plus_b_l156_156479


namespace quadrilateral_diagonals_perpendicular_l156_156314

def convex_quadrilateral (A B C D : Type) : Prop := sorry -- Assume it’s defined elsewhere 
def tangent_to_all_sides (circle : Type) (A B C D : Type) : Prop := sorry -- Assume it’s properly specified with its conditions elsewhere
def tangent_to_all_extensions (circle : Type) (A B C D : Type) : Prop := sorry -- Same as above

theorem quadrilateral_diagonals_perpendicular
  (A B C D : Type)
  (h_convex : convex_quadrilateral A B C D)
  (incircle excircle : Type)
  (h_incircle : tangent_to_all_sides incircle A B C D)
  (h_excircle : tangent_to_all_extensions excircle A B C D) : 
  (⊥ : Prop) :=  -- statement indicating perpendicularity 
sorry

end quadrilateral_diagonals_perpendicular_l156_156314


namespace find_abc_l156_156774

theorem find_abc (a b c : ℝ)
  (h1 : ∀ x : ℝ, (x < -6 ∨ (|x - 31| ≤ 1)) ↔ (x - a) * (x - b) / (x - c) ≤ 0)
  (h2 : a < b) :
  a + 2 * b + 3 * c = 76 :=
sorry

end find_abc_l156_156774


namespace smallest_number_greater_than_l156_156200

theorem smallest_number_greater_than : 
  ∀ (S : Set ℝ), S = {0.8, 0.5, 0.3} → 
  (∃ x ∈ S, x > 0.4 ∧ (∀ y ∈ S, y > 0.4 → x ≤ y)) → 
  x = 0.5 :=
by
  sorry

end smallest_number_greater_than_l156_156200


namespace work_days_l156_156260

/-- A needs 20 days to complete the work alone. B needs 10 days to complete the work alone.
    The total work must be completed in 12 days. We need to find how many days B must work 
    before A continues, such that the total work equals the full task. -/
theorem work_days (x : ℝ) (h0 : 0 ≤ x ∧ x ≤ 12) (h1 : 1 / 10 * x + 1 / 20 * (12 - x) = 1) : x = 8 := by
  sorry

end work_days_l156_156260


namespace total_number_of_birds_l156_156414

variable (swallows : ℕ) (bluebirds : ℕ) (cardinals : ℕ)
variable (h1 : swallows = 2)
variable (h2 : bluebirds = 2 * swallows)
variable (h3 : cardinals = 3 * bluebirds)

theorem total_number_of_birds : 
  swallows + bluebirds + cardinals = 18 := by
  sorry

end total_number_of_birds_l156_156414


namespace sin_double_angle_sub_pi_over_4_l156_156497

open Real

theorem sin_double_angle_sub_pi_over_4 (x : ℝ) (h : sin x = (sqrt 5 - 1) / 2) : 
  sin (2 * (x - π / 4)) = 2 - sqrt 5 :=
by
  sorry

end sin_double_angle_sub_pi_over_4_l156_156497


namespace sum_pairwise_relatively_prime_integers_eq_160_l156_156007

theorem sum_pairwise_relatively_prime_integers_eq_160
  (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h_prod : a * b * c = 27000)
  (h_coprime_ab : Nat.gcd a b = 1)
  (h_coprime_bc : Nat.gcd b c = 1)
  (h_coprime_ac : Nat.gcd a c = 1) :
  a + b + c = 160 :=
by
  sorry

end sum_pairwise_relatively_prime_integers_eq_160_l156_156007


namespace smallest_n_boxes_l156_156191

theorem smallest_n_boxes (n : ℕ) : (15 * n - 1) % 11 = 0 ↔ n = 3 :=
by
  sorry

end smallest_n_boxes_l156_156191


namespace arithmetic_geometric_sum_l156_156062

theorem arithmetic_geometric_sum {n : ℕ} (a : ℕ → ℤ) (S : ℕ → ℚ) 
  (h1 : ∀ k, a (k + 1) = a k + 2) 
  (h2 : (a 1) * (a 1 + a 4) = (a 1 + a 2) ^ 2 / 2) :
  S n = 6 - (4 * n + 6) / 2^n :=
by
  sorry

end arithmetic_geometric_sum_l156_156062


namespace sum_first_8_terms_64_l156_156038

-- Define the problem conditions
def isArithmeticSeq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def isGeometricSeq (a : ℕ → ℤ) : Prop :=
  ∀ (m n k : ℕ), m < n → n < k → (a n)^2 = a m * a k

-- Given arithmetic sequence with a common difference 2
def arithmeticSeqWithDiff2 (a : ℕ → ℤ) : Prop :=
  isArithmeticSeq a ∧ (∃ d : ℤ, d = 2 ∧ ∀ (n : ℕ), a (n + 1) = a n + d)

-- Given a₁, a₂, a₅ form a geometric sequence
def a1_a2_a5_formGeometricSeq (a: ℕ → ℤ) : Prop :=
  (a 2)^2 = (a 1) * (a 5)

-- Sum of the first 8 terms of the arithmetic sequence
def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a (n - 1)) / 2

-- Main statement
theorem sum_first_8_terms_64 (a : ℕ → ℤ) (h1 : arithmeticSeqWithDiff2 a) (h2 : a1_a2_a5_formGeometricSeq a) : 
  sum_of_first_n_terms a 8 = 64 := 
sorry

end sum_first_8_terms_64_l156_156038


namespace path_count_in_grid_l156_156534

theorem path_count_in_grid :
  let grid_width := 6
  let grid_height := 5
  let total_steps := 8
  let right_steps := 5
  let up_steps := 3
  ∃ (C : Nat), C = Nat.choose total_steps up_steps ∧ C = 56 :=
by
  sorry

end path_count_in_grid_l156_156534


namespace quadratic_root_difference_l156_156328

theorem quadratic_root_difference (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ + x₂ = 2 ∧ x₁ * x₂ = a ∧ (x₁ - x₂)^2 = 20) → a = -4 := 
by
  sorry

end quadratic_root_difference_l156_156328


namespace sum_x_y_eq_2_l156_156924

open Real

theorem sum_x_y_eq_2 (x y : ℝ) (h : x - 1 = 1 - y) : x + y = 2 :=
by
  sorry

end sum_x_y_eq_2_l156_156924


namespace graph_squares_count_l156_156822

theorem graph_squares_count :
  let x_intercept := 45
  let y_intercept := 5
  let total_squares := x_intercept * y_intercept
  let diagonal_squares := x_intercept + y_intercept - 1
  let non_diagonal_squares := total_squares - diagonal_squares
  non_diagonal_squares / 2 = 88 :=
by
  let x_intercept := 45
  let y_intercept := 5
  let total_squares := x_intercept * y_intercept
  let diagonal_squares := x_intercept + y_intercept - 1
  let non_diagonal_squares := total_squares - diagonal_squares
  have h : (non_diagonal_squares / 2 = 88) := sorry
  exact h

end graph_squares_count_l156_156822


namespace total_crayons_in_drawer_l156_156984

-- Definitions of conditions from a)
def initial_crayons : Nat := 9
def additional_crayons : Nat := 3

-- Statement to prove that total crayons in the drawer is 12
theorem total_crayons_in_drawer : initial_crayons + additional_crayons = 12 := sorry

end total_crayons_in_drawer_l156_156984


namespace trip_time_total_l156_156914

noncomputable def wrong_direction_time : ℝ := 75 / 60
noncomputable def return_time : ℝ := 75 / 45
noncomputable def normal_trip_time : ℝ := 250 / 45

theorem trip_time_total :
  wrong_direction_time + return_time + normal_trip_time = 8.48 := by
  sorry

end trip_time_total_l156_156914


namespace missing_digit_divisible_by_11_l156_156420

theorem missing_digit_divisible_by_11 (A : ℕ) (h : 1 ≤ A ∧ A ≤ 9) (div_11 : (100 + 10 * A + 2) % 11 = 0) : A = 3 :=
sorry

end missing_digit_divisible_by_11_l156_156420


namespace subcommittee_count_l156_156067

theorem subcommittee_count :
  let total_members := 12
  let teachers := 5
  let total_subcommittees := (Nat.choose total_members 4)
  let subcommittees_with_zero_teachers := (Nat.choose 7 4)
  let subcommittees_with_one_teacher := (Nat.choose teachers 1) * (Nat.choose 7 3)
  let subcommittees_with_fewer_than_two_teachers := subcommittees_with_zero_teachers + subcommittees_with_one_teacher
  let subcommittees_with_at_least_two_teachers := total_subcommittees - subcommittees_with_fewer_than_two_teachers
  subcommittees_with_at_least_two_teachers = 285 := by
  sorry

end subcommittee_count_l156_156067


namespace shaltaev_boltaev_proof_l156_156069

variable (S B : ℝ)

axiom cond1 : 175 * S > 125 * B
axiom cond2 : 175 * S < 126 * B

theorem shaltaev_boltaev_proof : 3 * S + B ≥ 1 :=
by {
  sorry
}

end shaltaev_boltaev_proof_l156_156069


namespace fraction_of_full_tank_used_l156_156101

-- Define the initial conditions as per the problem statement
def speed : ℝ := 50 -- miles per hour
def time : ℝ := 5   -- hours
def miles_per_gallon : ℝ := 30
def full_tank_capacity : ℝ := 15 -- gallons

-- We need to prove that the fraction of gasoline used is 5/9
theorem fraction_of_full_tank_used : 
  ((speed * time) / miles_per_gallon) / full_tank_capacity = 5 / 9 := by
sorry

end fraction_of_full_tank_used_l156_156101


namespace arithmetic_seq_a7_l156_156381

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 10)
  (h3 : ∀ n, a (n + 1) = a n + d) : a 7 = 8 :=
sorry

end arithmetic_seq_a7_l156_156381


namespace sum_symmetry_l156_156415

-- Definitions of minimum and maximum faces for dice in the problem
def min_face := 2
def max_face := 7
def num_dice := 8

-- Definitions of the minimum and maximum sum outcomes
def min_sum := num_dice * min_face
def max_sum := num_dice * max_face

-- Definition of the average value for symmetry
def avg_sum := (min_sum + max_sum) / 2

-- Definition of the probability symmetry theorem
theorem sum_symmetry (S : ℕ) : 
  (min_face <= S) ∧ (S <= max_face * num_dice) → 
  ∃ T, T = 2 * avg_sum - S ∧ T = 52 :=
by
  sorry

end sum_symmetry_l156_156415


namespace books_left_after_giveaways_l156_156685

def initial_books : ℝ := 48.0
def first_giveaway : ℝ := 34.0
def second_giveaway : ℝ := 3.0

theorem books_left_after_giveaways : 
  initial_books - first_giveaway - second_giveaway = 11.0 :=
by
  sorry

end books_left_after_giveaways_l156_156685


namespace part_I_part_II_l156_156271

noncomputable def f (x : ℝ) := |x - 2| - |2 * x + 1|

theorem part_I :
  { x : ℝ | f x ≤ 0 } = { x : ℝ | x ≤ -3 ∨ x ≥ (1 : ℝ) / 3 } :=
by
  sorry

theorem part_II :
  ∀ x : ℝ, f x - 2 * m^2 ≤ 4 * m :=
by
  sorry

end part_I_part_II_l156_156271


namespace line_through_point_outside_plane_l156_156399

-- Definitions based on conditions
variable {Point Line Plane : Type}
variable (P : Point) (a : Line) (α : Plane)

-- Define the conditions
variable (passes_through : Point → Line → Prop)
variable (outside_of : Point → Plane → Prop)

-- State the theorem
theorem line_through_point_outside_plane :
  (passes_through P a) ∧ (¬ outside_of P α) :=
sorry

end line_through_point_outside_plane_l156_156399


namespace integer_roots_condition_l156_156614

noncomputable def has_integer_roots (n : ℕ) : Prop :=
  ∃ x : ℤ, x * x - 4 * x + n = 0

theorem integer_roots_condition (n : ℕ) (h : n > 0) :
  has_integer_roots n ↔ n = 3 ∨ n = 4 :=
by 
  sorry

end integer_roots_condition_l156_156614


namespace sum_divisible_by_17_l156_156877

theorem sum_divisible_by_17 :
    (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92 + 93 + 94) % 17 = 0 := 
by 
  sorry

end sum_divisible_by_17_l156_156877


namespace decreasing_function_range_of_a_l156_156378

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem decreasing_function_range_of_a :
  (∀ x y : ℝ, x < y → f a y ≤ f a x) ↔ (1/7 ≤ a ∧ a < 1/3) :=
by
  sorry

end decreasing_function_range_of_a_l156_156378


namespace inequality_solution_set_result_l156_156484

theorem inequality_solution_set_result (a b x : ℝ) :
  (∀ x, a ≤ (3/4) * x^2 - 3 * x + 4 ∧ (3/4) * x^2 - 3 * x + 4 ≤ b) ∧ 
  (∀ x, x ∈ Set.Icc a b ↔ a ≤ x ∧ x ≤ b) →
  a + b = 4 := 
by
  sorry

end inequality_solution_set_result_l156_156484


namespace hcf_of_two_numbers_l156_156456
-- Importing the entire Mathlib library for mathematical functions

-- Define the two numbers and the conditions given in the problem
variables (x y : ℕ)

-- State the conditions as hypotheses
def conditions (h1 : x + y = 45) (h2 : Nat.lcm x y = 120) (h3 : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 11 / 120) : Prop :=
  True

-- State the theorem we want to prove
theorem hcf_of_two_numbers (x y : ℕ)
  (h1 : x + y = 45)
  (h2 : Nat.lcm x y = 120)
  (h3 : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 11 / 120) : Nat.gcd x y = 1 :=
  sorry

end hcf_of_two_numbers_l156_156456


namespace work_days_together_l156_156519

theorem work_days_together (d : ℕ) (h : d * (17 / 140) = 6 / 7) : d = 17 := by
  sorry

end work_days_together_l156_156519


namespace sum_of_underlined_numbers_non_negative_l156_156830

-- Definitions used in the problem
def is_positive (n : Int) : Prop := n > 0
def underlined (nums : List Int) : List Int := sorry -- Define underlining based on conditions

def sum_of_underlined_numbers (nums : List Int) : Int :=
  (underlined nums).sum

-- The proof problem statement
theorem sum_of_underlined_numbers_non_negative
  (nums : List Int)
  (h_len : nums.length = 100) :
  0 < sum_of_underlined_numbers nums := sorry

end sum_of_underlined_numbers_non_negative_l156_156830


namespace remaining_episodes_l156_156856

theorem remaining_episodes (seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (h_seasons : seasons = 12) (h_episodes_per_season : episodes_per_season = 20) 
  (h_fraction_watched : fraction_watched = 1/3) : 
  (seasons * episodes_per_season - fraction_watched * (seasons * episodes_per_season) = 160) := 
by
  sorry

end remaining_episodes_l156_156856


namespace difference_between_largest_and_smallest_quarters_l156_156658

noncomputable def coin_collection : Prop :=
  ∃ (n d q : ℕ), 
    (n + d + q = 150) ∧ 
    (5 * n + 10 * d + 25 * q = 2000) ∧ 
    (forall (q1 q2 : ℕ), (n + d + q1 = 150) ∧ (5 * n + 10 * d + 25 * q1 = 2000) → 
     (n + d + q2 = 150) ∧ (5 * n + 10 * d + 25 * q2 = 2000) → 
     (q1 = q2))

theorem difference_between_largest_and_smallest_quarters : coin_collection :=
  sorry

end difference_between_largest_and_smallest_quarters_l156_156658


namespace probability_two_consecutive_pairs_of_four_dice_correct_l156_156983

open Classical

noncomputable def probability_two_consecutive_pairs_of_four_dice : ℚ :=
  let total_outcomes := 6^4
  let favorable_outcomes := 48
  favorable_outcomes / total_outcomes

theorem probability_two_consecutive_pairs_of_four_dice_correct :
  probability_two_consecutive_pairs_of_four_dice = 1 / 27 := 
by
  sorry

end probability_two_consecutive_pairs_of_four_dice_correct_l156_156983


namespace donut_selection_count_l156_156094

def num_donut_selections : ℕ :=
  Nat.choose 9 3

theorem donut_selection_count : num_donut_selections = 84 := 
by
  sorry

end donut_selection_count_l156_156094


namespace find_a_plus_b_l156_156524

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 2^(a * x + b)

theorem find_a_plus_b
  (a b : ℝ)
  (h1 : f a b 2 = 1 / 2)
  (h2 : f a b (1 / 2) = 2) :
  a + b = 1 / 3 :=
sorry

end find_a_plus_b_l156_156524


namespace find_five_digit_number_l156_156611

theorem find_five_digit_number
  (x y : ℕ)
  (h1 : 10 * y + x - (10000 * x + y) = 34767)
  (h2 : 10 * y + x + (10000 * x + y) = 86937) :
  10000 * x + y = 26035 := by
  sorry

end find_five_digit_number_l156_156611


namespace sum_of_final_two_numbers_l156_156121

theorem sum_of_final_two_numbers (a b S : ℝ) (h : a + b = S) : 
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 :=
by
  sorry

end sum_of_final_two_numbers_l156_156121


namespace correct_option_l156_156289

theorem correct_option : (∃ x, x = -3 ∧ x^3 = -27) :=
by {
  -- Given conditions
  let x := -3
  use x
  constructor
  . rfl
  . norm_num
}

end correct_option_l156_156289


namespace guinea_pigs_food_difference_l156_156604

theorem guinea_pigs_food_difference :
  ∀ (first second third total : ℕ),
  first = 2 →
  second = first * 2 →
  total = 13 →
  first + second + third = total →
  third - second = 3 :=
by 
  intros first second third total h1 h2 h3 h4
  sorry

end guinea_pigs_food_difference_l156_156604


namespace evaluate_expression_equals_three_plus_sqrt_three_l156_156134

noncomputable def tan_sixty_squared_plus_one := Real.tan (60 * Real.pi / 180) ^ 2 + 1
noncomputable def tan_fortyfive_minus_twocos_thirty := Real.tan (45 * Real.pi / 180) - 2 * Real.cos (30 * Real.pi / 180)
noncomputable def expression (x y : ℝ) : ℝ := (x - (2 * x * y - y ^ 2) / x) / ((x ^ 2 - y ^ 2) / (x ^ 2 + x * y))

theorem evaluate_expression_equals_three_plus_sqrt_three :
  expression tan_sixty_squared_plus_one tan_fortyfive_minus_twocos_thirty = 3 + Real.sqrt 3 :=
sorry

end evaluate_expression_equals_three_plus_sqrt_three_l156_156134


namespace cubical_pyramidal_segment_volume_and_area_l156_156883

noncomputable def volume_and_area_sum (a : ℝ) : ℝ :=
  (1/4 * (9 + 27 * Real.sqrt 13))

theorem cubical_pyramidal_segment_volume_and_area :
  ∀ a : ℝ, a = 3 → volume_and_area_sum a = (9/2 + 27 * Real.sqrt 13 / 8) := by
  intro a ha
  sorry

end cubical_pyramidal_segment_volume_and_area_l156_156883


namespace width_of_rectangular_plot_l156_156961

theorem width_of_rectangular_plot 
  (length : ℝ) 
  (poles : ℕ) 
  (distance_between_poles : ℝ) 
  (num_poles : ℕ) 
  (total_wire_length : ℝ) 
  (perimeter : ℝ) 
  (width : ℝ) :
  length = 90 ∧ 
  distance_between_poles = 5 ∧ 
  num_poles = 56 ∧ 
  total_wire_length = (num_poles - 1) * distance_between_poles ∧ 
  total_wire_length = 275 ∧ 
  perimeter = 2 * (length + width) 
  → width = 47.5 :=
by
  sorry

end width_of_rectangular_plot_l156_156961


namespace f_f_2_equals_l156_156028

def f (x : ℕ) : ℕ := 4 * x ^ 3 - 6 * x + 2

theorem f_f_2_equals :
  f (f 2) = 42462 :=
by
  sorry

end f_f_2_equals_l156_156028


namespace lattice_midpoint_l156_156635

theorem lattice_midpoint (points : Fin 5 → ℤ × ℤ) :
  ∃ (i j : Fin 5), i ≠ j ∧ 
  let (x1, y1) := points i 
  let (x2, y2) := points j
  (x1 + x2) % 2 = 0 ∧ (y1 + y2) % 2 = 0 := 
sorry

end lattice_midpoint_l156_156635


namespace juniper_remaining_bones_l156_156045

-- Conditions
def initial_bones : ℕ := 4
def doubled_bones (b : ℕ) : ℕ := 2 * b
def stolen_bones (b : ℕ) : ℕ := b - 2

-- Theorem Statement
theorem juniper_remaining_bones : stolen_bones (doubled_bones initial_bones) = 6 := by
  -- Proof is omitted, only the statement is required as per instructions
  sorry

end juniper_remaining_bones_l156_156045


namespace isosceles_triangle_perimeter_l156_156342

theorem isosceles_triangle_perimeter
  (x y : ℝ)
  (h : |x - 3| + (y - 1)^2 = 0)
  (isosceles_triangle : ∃ a b c, (a = x ∧ b = x ∧ c = y) ∨ (a = x ∧ b = y ∧ c = y) ∨ (a = y ∧ b = y ∧ c = x)):
  ∃ perimeter : ℝ, perimeter = 7 :=
by
  sorry

end isosceles_triangle_perimeter_l156_156342


namespace problem_a_problem_b_problem_c_problem_d_l156_156844

theorem problem_a : 37.3 / (1 / 2) = 74.6 := by
  sorry

theorem problem_b : 0.45 - (1 / 20) = 0.4 := by
  sorry

theorem problem_c : (33 / 40) * (10 / 11) = 0.75 := by
  sorry

theorem problem_d : 0.375 + (1 / 40) = 0.4 := by
  sorry

end problem_a_problem_b_problem_c_problem_d_l156_156844


namespace sum_of_7a_and_3b_l156_156118

theorem sum_of_7a_and_3b (a b : ℤ) (h : a + b = 1998) : 7 * a + 3 * b ≠ 6799 :=
by sorry

end sum_of_7a_and_3b_l156_156118


namespace square_side_length_l156_156674

theorem square_side_length (d s : ℝ) (h_diag : d = 2) (h_rel : d = s * Real.sqrt 2) : s = Real.sqrt 2 :=
sorry

end square_side_length_l156_156674


namespace arithmetic_sequence_sum_l156_156639

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h0 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2) 
  (h1 : S 10 = 12)
  (h2 : S 20 = 17) :
  S 30 = 15 := by
  sorry

end arithmetic_sequence_sum_l156_156639


namespace inequality_not_less_than_l156_156591

theorem inequality_not_less_than (y : ℝ) : 2 * y + 8 ≥ -3 := 
sorry

end inequality_not_less_than_l156_156591


namespace cost_of_paving_floor_l156_156181

-- Conditions
def length_of_room : ℝ := 8
def width_of_room : ℝ := 4.75
def rate_per_sq_metre : ℝ := 900

-- Statement to prove
theorem cost_of_paving_floor : (length_of_room * width_of_room * rate_per_sq_metre) = 34200 :=
by
  sorry

end cost_of_paving_floor_l156_156181


namespace find_f_x_minus_1_l156_156243

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem find_f_x_minus_1 (x : ℝ) : f (x - 1) = 2 * x - 3 := by
  sorry

end find_f_x_minus_1_l156_156243


namespace one_corresponds_to_36_l156_156085

-- Define the given conditions
def corresponds (n : Nat) (s : String) : Prop :=
match n with
| 2  => s = "36"
| 3  => s = "363"
| 4  => s = "364"
| 5  => s = "365"
| 36 => s = "2"
| _  => False

-- Statement for the proof problem: Prove that 1 corresponds to 36
theorem one_corresponds_to_36 : corresponds 1 "36" :=
by
  sorry

end one_corresponds_to_36_l156_156085


namespace total_tomato_seeds_l156_156395

theorem total_tomato_seeds (mike_morning mike_afternoon : ℕ) 
  (ted_morning : mike_morning = 50) 
  (ted_afternoon : mike_afternoon = 60) 
  (ted_morning_eq : 2 * mike_morning = 100) 
  (ted_afternoon_eq : mike_afternoon - 20 = 40)
  (total_seeds : mike_morning + mike_afternoon + (2 * mike_morning) + (mike_afternoon - 20) = 250) : 
  (50 + 60 + 100 + 40 = 250) :=
sorry

end total_tomato_seeds_l156_156395


namespace inverse_fourier_transform_l156_156330

noncomputable def F (p : ℝ) : ℂ :=
if 0 < p ∧ p < 1 then 1 else 0

noncomputable def f (x : ℝ) : ℂ :=
(1 / Real.sqrt (2 * Real.pi)) * ((1 - Complex.exp (-Complex.I * x)) / (Complex.I * x))

theorem inverse_fourier_transform :
  ∀ x, (f x) = (1 / (Real.sqrt (2 * Real.pi))) * ((1 - Complex.exp (-Complex.I * x)) / (Complex.I * x)) := by
  intros
  sorry

end inverse_fourier_transform_l156_156330


namespace min_value_expression_l156_156807

theorem min_value_expression (a b c : ℝ) (h : 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 4) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (4 / c - 1)^2 = 12 - 8 * Real.sqrt 2 :=
sorry

end min_value_expression_l156_156807


namespace num_positive_integer_solutions_l156_156053

theorem num_positive_integer_solutions : 
  ∃ n : ℕ, (∀ x : ℕ, x ≤ n → x - 1 < Real.sqrt 5) ∧ n = 3 :=
by
  sorry

end num_positive_integer_solutions_l156_156053


namespace minimum_value_of_f_l156_156313

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - x + (1 / 3)

theorem minimum_value_of_f :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ 1) → (∀ x : ℝ, f 1 = -(1 / 3)) :=
by
  sorry

end minimum_value_of_f_l156_156313


namespace cubic_root_identity_l156_156529

theorem cubic_root_identity (r : ℝ) (h : (r^(1/3)) - (1/(r^(1/3))) = 2) : r^3 - (1/r^3) = 14 := 
by 
  sorry

end cubic_root_identity_l156_156529


namespace alcohol_solution_volume_l156_156172

theorem alcohol_solution_volume (V : ℝ) (h1 : 0.42 * V = 0.33 * (V + 3)) : V = 11 :=
by
  sorry

end alcohol_solution_volume_l156_156172


namespace equation_has_no_solution_l156_156305

theorem equation_has_no_solution (k : ℝ) : ¬ (∃ x : ℝ , (x ≠ 3 ∧ x ≠ 4) ∧ (x - 1) / (x - 3) = (x - k) / (x - 4)) ↔ k = 2 :=
by
  sorry

end equation_has_no_solution_l156_156305


namespace percentage_to_pass_l156_156427

theorem percentage_to_pass (score shortfall max_marks : ℕ) (h_score : score = 212) (h_shortfall : shortfall = 13) (h_max_marks : max_marks = 750) :
  (score + shortfall) / max_marks * 100 = 30 :=
by
  sorry

end percentage_to_pass_l156_156427


namespace binom_13_11_eq_78_l156_156114

theorem binom_13_11_eq_78 : Nat.choose 13 11 = 78 := by
  sorry

end binom_13_11_eq_78_l156_156114


namespace arithmetic_sequence_propositions_l156_156716

theorem arithmetic_sequence_propositions (a_n : ℕ → ℤ) (S : ℕ → ℤ)
  (h_S_def : ∀ n, S n = n * (a_n 1 + (a_n (n - 1))) / 2)
  (h_cond : S 6 > S 7 ∧ S 7 > S 5) :
  (∃ d, d < 0 ∧ S 11 > 0) :=
by
  sorry

end arithmetic_sequence_propositions_l156_156716


namespace solve_for_a_minus_b_l156_156496

theorem solve_for_a_minus_b (a b : ℚ) 
  (h1 : 2020 * a + 2024 * b = 2030) 
  (h2 : 2022 * a + 2026 * b = 2032) : 
  a - b = -4 := 
sorry

end solve_for_a_minus_b_l156_156496


namespace cos_neg_17pi_over_4_l156_156336

noncomputable def cos_value : ℝ := (Real.pi / 4).cos

theorem cos_neg_17pi_over_4 :
  (Real.cos (-17 * Real.pi / 4)) = cos_value :=
by
  -- Define even property of cosine and angle simplification
  sorry

end cos_neg_17pi_over_4_l156_156336


namespace sum_lengths_DE_EF_equals_9_l156_156730

variable (AB BC FA : ℝ)
variable (area_ABCDEF : ℝ)
variable (DE EF : ℝ)

theorem sum_lengths_DE_EF_equals_9 (h1 : area_ABCDEF = 52) (h2 : AB = 8) (h3 : BC = 9) (h4 : FA = 5)
  (h5 : AB * BC - area_ABCDEF = DE * EF) (h6 : BC - FA = DE) : DE + EF = 9 := 
by 
  sorry

end sum_lengths_DE_EF_equals_9_l156_156730


namespace polynomial_is_monic_l156_156968

noncomputable def f : ℝ → ℝ := sorry

variables (h1 : f 1 = 3) (h2 : f 2 = 12) (h3 : ∀ x : ℝ, f x = x^2 + 6*x - 4)

theorem polynomial_is_monic (f : ℝ → ℝ) (h1 : f 1 = 3) (h2 : f 2 = 12) (h3 : ∀ x : ℝ, f x = x^2 + x + b) : 
  ∀ x : ℝ, f x = x^2 + 6*x - 4 :=
by sorry

end polynomial_is_monic_l156_156968


namespace martin_improved_lap_time_l156_156828

def initial_laps := 15
def initial_time := 45 -- in minutes
def final_laps := 18
def final_time := 42 -- in minutes

noncomputable def initial_lap_time := initial_time / initial_laps
noncomputable def final_lap_time := final_time / final_laps
noncomputable def improvement := initial_lap_time - final_lap_time

theorem martin_improved_lap_time : improvement = 2 / 3 := by 
  sorry

end martin_improved_lap_time_l156_156828


namespace official_exchange_rate_l156_156773

theorem official_exchange_rate (E : ℝ)
  (h1 : 70 = 10 * (7 / 5) * E) :
  E = 5 :=
by
  sorry

end official_exchange_rate_l156_156773


namespace eighth_term_geometric_seq_l156_156855

theorem eighth_term_geometric_seq (a1 a2 : ℚ) (a1_val : a1 = 3) (a2_val : a2 = 9 / 2) :
  (a1 * (a2 / a1)^(7) = 6561 / 128) :=
  by
    sorry

end eighth_term_geometric_seq_l156_156855


namespace x_lt_y_l156_156357

theorem x_lt_y (n : ℕ) (h_n : n > 2) (x y : ℝ) (h_x_pos : 0 < x) (h_y_pos : 0 < y) (h_x : x ^ n = x + 1) (h_y : y ^ (n + 1) = y ^ 3 + 1) : x < y :=
sorry

end x_lt_y_l156_156357


namespace fifth_term_sequence_l156_156701

theorem fifth_term_sequence : 
  (4 + 8 + 16 + 32 + 64) = 124 := 
by 
  sorry

end fifth_term_sequence_l156_156701


namespace sum_of_first_six_terms_l156_156317

variable {a_n : ℕ → ℕ}
variable {d : ℕ}

def is_arithmetic_sequence (a_n : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d 

theorem sum_of_first_six_terms (a_3 a_4 : ℕ) (h : a_3 + a_4 = 30) :
  ∃ a_n d, is_arithmetic_sequence a_n d ∧ 
  a_n 3 = a_3 ∧ a_n 4 = a_4 ∧ 
  (3 * (a_n 1 + (a_n 1 + 5 * d))) = 90 := 
sorry

end sum_of_first_six_terms_l156_156317


namespace not_both_267_and_269_non_standard_l156_156686

def G : ℤ → ℤ := sorry

def exists_x_ne_c (G : ℤ → ℤ) : Prop :=
  ∀ c : ℤ, ∃ x : ℤ, G x ≠ c

def non_standard (G : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, G x = G (a - x)

theorem not_both_267_and_269_non_standard (G : ℤ → ℤ)
  (h1 : exists_x_ne_c G) :
  ¬ (non_standard G 267 ∧ non_standard G 269) :=
sorry

end not_both_267_and_269_non_standard_l156_156686


namespace value_of_business_l156_156627

theorem value_of_business 
  (ownership : ℚ)
  (sale_fraction : ℚ)
  (sale_value : ℚ) 
  (h_ownership : ownership = 2/3) 
  (h_sale_fraction : sale_fraction = 3/4) 
  (h_sale_value : sale_value = 6500) : 
  2 * sale_value = 13000 := 
by
  -- mathematical equivalent proof here
  -- This is a placeholder.
  sorry

end value_of_business_l156_156627


namespace total_shells_is_correct_l156_156137

def morning_shells : Nat := 292
def afternoon_shells : Nat := 324
def total_shells : Nat := morning_shells + afternoon_shells

theorem total_shells_is_correct : total_shells = 616 :=
by
  sorry

end total_shells_is_correct_l156_156137


namespace baron_not_boasting_l156_156537

-- Define a function to verify if a given list of digits is a palindrome
def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

-- Define a list that represents the sequence given in the solution
def sequence_19 : List ℕ :=
  [9, 18, 7, 16, 5, 14, 3, 12, 1, 10, 11, 2, 13, 4, 15, 6, 17, 8, 19]

-- Prove that the sequence forms a palindrome
theorem baron_not_boasting : is_palindrome sequence_19 :=
by {
  -- Insert actual proof steps here
  sorry
}

end baron_not_boasting_l156_156537


namespace art_club_activity_l156_156217

theorem art_club_activity (n p s b : ℕ) (h1 : n = 150) (h2 : p = 80) (h3 : s = 60) (h4 : b = 20) :
  (n - (p + s - b) = 30) :=
by
  sorry

end art_club_activity_l156_156217


namespace solution_set_inequality_l156_156426

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) * (x - 1) > 0 ↔ x < 1 ∨ x > 3 :=
sorry

end solution_set_inequality_l156_156426


namespace max_enclosed_area_perimeter_160_length_twice_width_l156_156533

theorem max_enclosed_area_perimeter_160_length_twice_width 
  (W L : ℕ) 
  (h1 : 2 * (L + W) = 160) 
  (h2 : L = 2 * W) : 
  L * W = 1352 := 
sorry

end max_enclosed_area_perimeter_160_length_twice_width_l156_156533


namespace range_of_half_alpha_minus_beta_l156_156104

theorem range_of_half_alpha_minus_beta (α β : ℝ) (hα : 1 < α ∧ α < 3) (hβ : -4 < β ∧ β < 2) :
  -3 / 2 < (1 / 2) * α - β ∧ (1 / 2) * α - β < 11 / 2 :=
sorry

end range_of_half_alpha_minus_beta_l156_156104


namespace championship_outcomes_l156_156027

theorem championship_outcomes (students events : ℕ) (hs : students = 5) (he : events = 3) :
  ∃ outcomes : ℕ, outcomes = 5 ^ 3 := by
  sorry

end championship_outcomes_l156_156027


namespace arrival_time_difference_l156_156753

theorem arrival_time_difference
  (d : ℝ) (r_H : ℝ) (r_A : ℝ) (h₁ : d = 2) (h₂ : r_H = 12) (h₃ : r_A = 6) :
  (d / r_A * 60) - (d / r_H * 60) = 10 :=
by
  sorry

end arrival_time_difference_l156_156753


namespace max_value_even_function_1_2_l156_156458

-- Define the even function property
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Given conditions
variables (f : ℝ → ℝ)
variable (h1 : even_function f)
variable (h2 : ∀ x, -2 ≤ x ∧ x ≤ -1 → f x ≤ -2)

-- Prove the maximum value on [1, 2] is -2
theorem max_value_even_function_1_2 : (∀ x, 1 ≤ x ∧ x ≤ 2 → f x ≤ -2) :=
sorry

end max_value_even_function_1_2_l156_156458


namespace daniel_fraction_l156_156876

theorem daniel_fraction (A B C D : Type) (money : A → ℝ) 
  (adriano bruno cesar daniel : A)
  (h1 : money daniel = 0)
  (given_amount : ℝ)
  (h2 : money adriano = 5 * given_amount)
  (h3 : money bruno = 4 * given_amount)
  (h4 : money cesar = 3 * given_amount)
  (h5 : money daniel = (1 / 5) * money adriano + (1 / 4) * money bruno + (1 / 3) * money cesar) :
  money daniel / (money adriano + money bruno + money cesar) = 1 / 4 := 
by
  sorry

end daniel_fraction_l156_156876


namespace smallest_possible_n_l156_156873

theorem smallest_possible_n (x n : ℤ) (hx : 0 < x) (m : ℤ) (hm : m = 30) (h1 : m.gcd n = x + 1) (h2 : m.lcm n = x * (x + 1)) : n = 6 := sorry

end smallest_possible_n_l156_156873


namespace long_side_length_l156_156550

variable {a b d : ℝ}

theorem long_side_length (h1 : a / b = 2 * (b / d)) (h2 : a = 4) (hd : d = Real.sqrt (a^2 + b^2)) :
  b = Real.sqrt (2 + 4 * Real.sqrt 17) :=
sorry

end long_side_length_l156_156550


namespace apples_left_l156_156036

def Mike_apples : ℝ := 7.0
def Nancy_apples : ℝ := 3.0
def Keith_ate_apples : ℝ := 6.0

theorem apples_left : Mike_apples + Nancy_apples - Keith_ate_apples = 4.0 := by
  sorry

end apples_left_l156_156036


namespace equivalence_sufficient_necessary_l156_156728

-- Definitions for conditions
variables (A B : Prop)

-- Statement to prove
theorem equivalence_sufficient_necessary :
  (A → B) ↔ (¬B → ¬A) :=
by sorry

end equivalence_sufficient_necessary_l156_156728


namespace compare_three_and_negfour_l156_156348

theorem compare_three_and_negfour : 3 > -4 := by
  sorry

end compare_three_and_negfour_l156_156348


namespace ab_plus_cd_eq_12_l156_156525

theorem ab_plus_cd_eq_12 (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = -1) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 5) :
  a * b + c * d = 12 := by
  sorry

end ab_plus_cd_eq_12_l156_156525


namespace total_wicks_l156_156075

-- Amy bought a 15-foot spool of string.
def spool_length_feet : ℕ := 15

-- Since there are 12 inches in a foot, convert the spool length to inches.
def spool_length_inches : ℕ := spool_length_feet * 12

-- The string is cut into an equal number of 6-inch and 12-inch wicks.
def wick_pair_length : ℕ := 6 + 12

-- Prove that the total number of wicks she cuts is 20.
theorem total_wicks : (spool_length_inches / wick_pair_length) * 2 = 20 := by
  sorry

end total_wicks_l156_156075


namespace minimum_product_value_l156_156474

-- Problem conditions
def total_stones : ℕ := 40
def b_min : ℕ := 20
def b_max : ℕ := 32

-- Define the product function
def P (b : ℕ) : ℕ := b * (total_stones - b)

-- Goal: Prove the minimum value of P(b) for b in [20, 32] is 256
theorem minimum_product_value : ∃ (b : ℕ), b_min ≤ b ∧ b ≤ b_max ∧ P b = 256 := by
  sorry

end minimum_product_value_l156_156474


namespace find_f_nine_l156_156887

-- Define the function f that satisfies the conditions
def f (x : ℝ) : ℝ := sorry

-- Define the condition that f(x + y) = f(x) * f(y) for all real x and y
axiom functional_equation : ∀ (x y : ℝ), f (x + y) = f x * f y

-- Define the condition that f(3) = 4
axiom f_three : f 3 = 4

-- State the main theorem to prove that f(9) = 64
theorem find_f_nine : f 9 = 64 := by
  sorry

end find_f_nine_l156_156887


namespace largest_divisor_of_prime_squares_l156_156238

theorem largest_divisor_of_prime_squares (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : q < p) : 
  ∃ d : ℕ, ∀ p q : ℕ, Prime p → Prime q → q < p → d ∣ (p^2 - q^2) ∧ ∀ k : ℕ, (∀ p q : ℕ, Prime p → Prime q → q < p → k ∣ (p^2 - q^2)) → k ≤ d :=
by 
  use 2
  {
    sorry
  }

end largest_divisor_of_prime_squares_l156_156238


namespace sum_of_squares_l156_156457

def positive_integers (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0

def sum_of_values (x y z : ℕ) : Prop :=
  x + y + z = 24 ∧ Int.gcd x y + Int.gcd y z + Int.gcd z x = 10

theorem sum_of_squares (x y z : ℕ) (h1 : positive_integers x y z) (h2 : sum_of_values x y z) :
  x^2 + y^2 + z^2 = 296 :=
by sorry

end sum_of_squares_l156_156457


namespace max_sum_length_le_98306_l156_156086

noncomputable def L (k : ℕ) : ℕ := sorry

theorem max_sum_length_le_98306 (x y : ℕ) (hx : x > 1) (hy : y > 1) (hl : L x + L y = 16) : x + 3 * y < 98306 :=
sorry

end max_sum_length_le_98306_l156_156086


namespace range_of_m_l156_156170

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 + 2 * x + m^2 > 0) ↔ -1 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l156_156170


namespace clive_money_l156_156018

noncomputable def clive_initial_money : ℝ  :=
  let total_olives := 80
  let olives_per_jar := 20
  let cost_per_jar := 1.5
  let change := 4
  let jars_needed := total_olives / olives_per_jar
  let total_cost := jars_needed * cost_per_jar
  total_cost + change

theorem clive_money (h1 : clive_initial_money = 10) : clive_initial_money = 10 :=
by sorry

end clive_money_l156_156018


namespace integer_square_mod_4_l156_156561

theorem integer_square_mod_4 (N : ℤ) : (N^2 % 4 = 0) ∨ (N^2 % 4 = 1) :=
by sorry

end integer_square_mod_4_l156_156561


namespace no_equal_refereed_matches_l156_156629

theorem no_equal_refereed_matches {k : ℕ} (h1 : ∀ {n : ℕ}, n > k → n = 2 * k) 
    (h2 : ∀ {n : ℕ}, n > k → ∃ m, m = k * (2 * k - 1))
    (h3 : ∀ {n : ℕ}, n > k → ∃ r, r = (2 * k - 1) / 2): 
    False := 
by
  sorry

end no_equal_refereed_matches_l156_156629


namespace product_is_58_l156_156620

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def p := 2
def q := 29

-- Conditions based on the problem
axiom prime_p : is_prime p
axiom prime_q : is_prime q
axiom sum_eq_31 : p + q = 31

-- Theorem to be proven
theorem product_is_58 : p * q = 58 :=
by
  sorry

end product_is_58_l156_156620


namespace ratio_d_c_l156_156793

theorem ratio_d_c (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hc : c ≠ 0) 
  (h1 : 8 * x - 5 * y = c) (h2 : 10 * y - 16 * x = d) : d / c = -2 :=
by
  sorry

end ratio_d_c_l156_156793


namespace gcd_a_b_l156_156169

def a := 130^2 + 250^2 + 360^2
def b := 129^2 + 249^2 + 361^2

theorem gcd_a_b : Int.gcd a b = 1 := 
by
  sorry

end gcd_a_b_l156_156169


namespace bob_pennies_l156_156352

variable (a b : ℕ)

theorem bob_pennies : 
  (b + 2 = 4 * (a - 2)) →
  (b - 3 = 3 * (a + 3)) →
  b = 78 :=
by
  intros h1 h2
  sorry

end bob_pennies_l156_156352


namespace Mrs_Hilt_walks_to_fountain_l156_156843

theorem Mrs_Hilt_walks_to_fountain :
  ∀ (distance trips : ℕ), distance = 30 → trips = 4 → distance * trips = 120 :=
by
  intros distance trips h_distance h_trips
  sorry

end Mrs_Hilt_walks_to_fountain_l156_156843


namespace total_days_off_l156_156107

-- Definitions for the problem conditions
def days_off_personal (months_in_year : ℕ) (days_per_month : ℕ) : ℕ :=
  days_per_month * months_in_year

def days_off_professional (months_in_year : ℕ) (days_per_month : ℕ) : ℕ :=
  days_per_month * months_in_year

def days_off_teambuilding (quarters_in_year : ℕ) (days_per_quarter : ℕ) : ℕ :=
  days_per_quarter * quarters_in_year

-- Main theorem to prove
theorem total_days_off
  (months_in_year : ℕ) (quarters_in_year : ℕ)
  (days_per_month_personal : ℕ) (days_per_month_professional : ℕ) (days_per_quarter_teambuilding: ℕ)
  (h_months : months_in_year = 12) (h_quarters : quarters_in_year = 4) 
  (h_days_personal : days_per_month_personal = 4) (h_days_professional : days_per_month_professional = 2) (h_days_teambuilding : days_per_quarter_teambuilding = 1) :
  days_off_personal months_in_year days_per_month_personal
  + days_off_professional months_in_year days_per_month_professional
  + days_off_teambuilding quarters_in_year days_per_quarter_teambuilding
  = 76 := 
by {
  -- Calculation
  sorry
}

end total_days_off_l156_156107


namespace min_value_expression_l156_156303

theorem min_value_expression (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : a + b = 2) :
  (∃ x : ℝ, x = (1 / (a - 1)) + (1 / (2 * b)) ∧ x ≥ (3 / 2 + Real.sqrt 2)) :=
sorry

end min_value_expression_l156_156303


namespace radius_of_sphere_is_two_sqrt_46_l156_156485

theorem radius_of_sphere_is_two_sqrt_46
  (a b c : ℝ)
  (s : ℝ)
  (h1 : 4 * (a + b + c) = 160)
  (h2 : 2 * (a * b + b * c + c * a) = 864)
  (h3 : s = Real.sqrt ((a^2 + b^2 + c^2) / 4)) :
  s = 2 * Real.sqrt 46 :=
by
  -- proof placeholder
  sorry

end radius_of_sphere_is_two_sqrt_46_l156_156485


namespace exists_x0_f_leq_one_tenth_l156_156720

noncomputable def f (x a : ℝ) : ℝ := x^2 + (Real.log (3*x))^2 - 2*a*x - 6*a*(Real.log (3*x)) + 10*a^2

theorem exists_x0_f_leq_one_tenth (a : ℝ) : (∃ x₀, f x₀ a ≤ 1/10) ↔ a = 1/30 := by
  sorry

end exists_x0_f_leq_one_tenth_l156_156720


namespace ellipse_area_l156_156703

theorem ellipse_area :
  ∃ a b : ℝ, 
    (∀ x y : ℝ, (x^2 - 2 * x + 9 * y^2 + 18 * y + 16 = 0) → 
    (a = 2 ∧ b = (2 / 3) ∧ (π * a * b = 4 * π / 3))) :=
sorry

end ellipse_area_l156_156703


namespace find_p_if_parabola_axis_tangent_to_circle_l156_156449

theorem find_p_if_parabola_axis_tangent_to_circle :
  ∀ (p : ℝ), 0 < p →
    (∃ (C : ℝ × ℝ) (r : ℝ), 
      (C = (2, 0)) ∧ (r = 3) ∧ (dist (C.1 + p / 2, C.2) (C.1, C.2) = r) 
    ) → p = 2 :=
by
  intro p hp h
  rcases h with ⟨C, r, hC, hr, h_dist⟩ 
  have h_eq : C = (2, 0) := hC
  have hr_eq : r = 3 := hr
  rw [h_eq, hr_eq] at h_dist
  sorry

end find_p_if_parabola_axis_tangent_to_circle_l156_156449


namespace range_of_x_l156_156397

theorem range_of_x (x : ℝ) : -2 * x + 3 ≤ 6 → x ≥ -3 / 2 :=
sorry

end range_of_x_l156_156397


namespace monthly_income_l156_156444

def average_expenditure_6_months (expenditure_6_months : ℕ) (average : ℕ) : Prop :=
  average = expenditure_6_months / 6

def expenditure_next_4_months (expenditure_4_months : ℕ) (monthly_expense : ℕ) : Prop :=
  expenditure_4_months = 4 * monthly_expense

def cleared_debt_and_saved (income_4_months : ℕ) (debt : ℕ) (savings : ℕ)  (condition : ℕ) : Prop :=
  income_4_months = debt + savings + condition

theorem monthly_income 
(income : ℕ) 
(avg_6m_exp : ℕ) 
(exp_4m : ℕ) 
(debt: ℕ) 
(savings: ℕ )
(condition: ℕ) 
    (h1 : average_expenditure_6_months avg_6m_exp 85) 
    (h2 : expenditure_next_4_months exp_4m 60) 
    (h3 : cleared_debt_and_saved (income * 4) debt savings 30) 
    (h4 : income * 6 < 6 * avg_6m_exp) 
    : income = 78 :=
sorry

end monthly_income_l156_156444


namespace part1_intersection_part2_sufficient_not_necessary_l156_156338

open Set

-- Definition of sets A and B
def set_A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def set_B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

-- Part (1)
theorem part1_intersection (a : ℝ) (h : a = -2) : set_A a ∩ set_B = {x | -3 ≤ x ∧ x ≤ -2} := by
  sorry

-- Part (2)
theorem part2_sufficient_not_necessary (p q : Prop) (hp : ∀ x, set_A a x → set_B x) (h_suff : p → q) (h_not_necess : ¬(q → p)) : set_A a ⊆ set_B → a ∈ Iic (-3) ∪ Ici 4 := by
  sorry

end part1_intersection_part2_sufficient_not_necessary_l156_156338


namespace four_digit_numbers_divisible_by_17_l156_156775

theorem four_digit_numbers_divisible_by_17 :
  ∃ n, (∀ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 17 = 0 ↔ ∃ k, x = 17 * k ∧ 59 ≤ k ∧ k ≤ 588) ∧ n = 530 := 
sorry

end four_digit_numbers_divisible_by_17_l156_156775


namespace students_arrangement_l156_156186

theorem students_arrangement (B1 B2 S1 S2 T1 T2 C1 C2 : ℕ) :
  (B1 = B2 ∧ S1 ≠ S2 ∧ T1 ≠ T2 ∧ C1 ≠ C2) →
  (C1 ≠ C2) →
  (arrangements = 7200) :=
by
  sorry

end students_arrangement_l156_156186


namespace right_triangle_perimeter_l156_156123

noncomputable def perimeter_of_right_triangle (x : ℝ) : ℝ :=
  let y := x + 15
  let c := Real.sqrt (x^2 + y^2)
  x + y + c

theorem right_triangle_perimeter
  (h₁ : ∀ a b : ℝ, a * b = 2 * 150)  -- The area condition
  (h₂ : ∀ a b : ℝ, b = a + 15)       -- One leg is 15 units longer than the other
  : perimeter_of_right_triangle 11.375 = 66.47 :=
by
  sorry

end right_triangle_perimeter_l156_156123


namespace four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime_l156_156275

theorem four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime
  (N : ℕ) (hN : N ≥ 2) :
  (∀ n : ℕ, n < N → ¬ ∃ k : ℕ, k^2 = 4 * n * (N - n) + 1) ↔ Nat.Prime (N^2 + 1) :=
by sorry

end four_n_N_minus_n_plus_1_not_square_iff_N_sq_plus_1_prime_l156_156275


namespace probability_fx_lt_0_l156_156599

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem probability_fx_lt_0 :
  (∫ x in -Real.pi..Real.pi, if f x < 0 then 1 else 0) / (2 * Real.pi) = 2 / Real.pi :=
by sorry

end probability_fx_lt_0_l156_156599


namespace digit_in_base_l156_156052

theorem digit_in_base (t : ℕ) (h1 : t ≤ 9) (h2 : 5 * 7 + t = t * 9 + 3) : t = 4 := by
  sorry

end digit_in_base_l156_156052


namespace Suzanna_rides_8_miles_in_40_minutes_l156_156784

theorem Suzanna_rides_8_miles_in_40_minutes :
  (∀ n : ℕ, Suzanna_distance_in_n_minutes = (n / 10) * 2) → Suzanna_distance_in_40_minutes = 8 :=
by
  sorry

-- Definitions for Suzanna's distance conditions
def Suzanna_distance_in_n_minutes (n : ℕ) : ℕ := (n / 10) * 2

noncomputable def Suzanna_distance_in_40_minutes := Suzanna_distance_in_n_minutes 40

#check Suzanna_rides_8_miles_in_40_minutes

end Suzanna_rides_8_miles_in_40_minutes_l156_156784


namespace compare_fractions_difference_l156_156963

theorem compare_fractions_difference :
  let a := (1 : ℝ) / 2
  let b := (1 : ℝ) / 3
  a - b = 1 / 6 :=
by
  sorry

end compare_fractions_difference_l156_156963


namespace find_c_plus_one_div_b_l156_156491

-- Assume that a, b, and c are positive real numbers such that the given conditions hold.
variables (a b c : ℝ)
variables (habc : a * b * c = 1)
variables (hac : a + 1 / c = 7)
variables (hba : b + 1 / a = 11)

-- The goal is to show that c + 1 / b = 5 / 19.
theorem find_c_plus_one_div_b : c + 1 / b = 5 / 19 :=
by 
  sorry

end find_c_plus_one_div_b_l156_156491


namespace Harold_speed_is_one_more_l156_156761

variable (Adrienne_speed Harold_speed : ℝ)
variable (distance_when_Harold_catches_Adr : ℝ)
variable (time_difference : ℝ)

axiom Adrienne_speed_def : Adrienne_speed = 3
axiom Harold_catches_distance : distance_when_Harold_catches_Adr = 12
axiom time_difference_def : time_difference = 1

theorem Harold_speed_is_one_more :
  Harold_speed - Adrienne_speed = 1 :=
by 
  have Adrienne_time := (distance_when_Harold_catches_Adr - Adrienne_speed * time_difference) / Adrienne_speed 
  have Harold_time := distance_when_Harold_catches_Adr / Harold_speed
  have := Adrienne_time = Harold_time - time_difference
  sorry

end Harold_speed_is_one_more_l156_156761


namespace parabola_vertex_in_other_l156_156806

theorem parabola_vertex_in_other (p q a : ℝ) (h₁ : a ≠ 0) 
  (h₂ : ∀ (x : ℝ),  x = a → pa^2 = p * x^2) 
  (h₃ : ∀ (x : ℝ),  x = 0 → 0 = q * (x - a)^2 + pa^2) : 
  p + q = 0 := 
sorry

end parabola_vertex_in_other_l156_156806


namespace gcd_eq_55_l156_156108

theorem gcd_eq_55 : Nat.gcd 5280 12155 = 55 := sorry

end gcd_eq_55_l156_156108


namespace absolute_value_solution_l156_156546

theorem absolute_value_solution (m : ℤ) (h : abs m = abs (-7)) : m = 7 ∨ m = -7 := by
  sorry

end absolute_value_solution_l156_156546


namespace average_of_angles_l156_156869

theorem average_of_angles (p q r s t : ℝ) (h : p + q + r + s + t = 180) : 
  (p + q + r + s + t) / 5 = 36 :=
by
  sorry

end average_of_angles_l156_156869


namespace salary_increase_gt_90_percent_l156_156566

theorem salary_increase_gt_90_percent (S : ℝ) : 
  (S * (1.12^6) - S) / S > 0.90 :=
by
  -- Here we skip the proof with sorry
  sorry

end salary_increase_gt_90_percent_l156_156566


namespace catch_bus_probability_within_5_minutes_l156_156664

theorem catch_bus_probability_within_5_minutes :
  (Pbus3 : ℝ) → (Pbus6 : ℝ) → (Pbus3 = 0.20) → (Pbus6 = 0.60) → (Pcatch : ℝ) → (Pcatch = Pbus3 + Pbus6) → (Pcatch = 0.80) :=
by
  intros Pbus3 Pbus6 hPbus3 hPbus6 Pcatch hPcatch
  sorry

end catch_bus_probability_within_5_minutes_l156_156664


namespace tan_sub_theta_cos_double_theta_l156_156093

variables (θ : ℝ)

-- Condition: given tan θ = 2
axiom tan_theta_eq_two : Real.tan θ = 2

-- Proof problem 1: Prove tan (π/4 - θ) = -1/3
theorem tan_sub_theta (h : Real.tan θ = 2) : Real.tan (Real.pi / 4 - θ) = -1/3 :=
by sorry

-- Proof problem 2: Prove cos 2θ = -3/5
theorem cos_double_theta (h : Real.tan θ = 2) : Real.cos (2 * θ) = -3/5 :=
by sorry

end tan_sub_theta_cos_double_theta_l156_156093


namespace sphere_radius_volume_eq_surface_area_l156_156194

theorem sphere_radius_volume_eq_surface_area (r : ℝ) (h₁ : (4 / 3) * π * r^3 = 4 * π * r^2) : r = 3 :=
by
  sorry

end sphere_radius_volume_eq_surface_area_l156_156194


namespace sun_xing_zhe_problem_l156_156489

theorem sun_xing_zhe_problem (S X Z : ℕ) (h : S < 10 ∧ X < 10 ∧ Z < 10)
  (hprod : (100 * S + 10 * X + Z) * (100 * Z + 10 * X + S) = 78445) :
  (100 * S + 10 * X + Z) + (100 * Z + 10 * X + S) = 1372 := 
by
  sorry

end sun_xing_zhe_problem_l156_156489


namespace fraction_evaluation_l156_156792

theorem fraction_evaluation :
  ( (1 / 2 * 1 / 3 * 1 / 4 * 1 / 5 + 3 / 2 * 3 / 4 * 3 / 5) / 
    (1 / 2 * 2 / 3 * 2 / 5) ) = 41 / 8 :=
by
  sorry

end fraction_evaluation_l156_156792


namespace contingency_fund_correct_l156_156318

def annual_donation := 240
def community_pantry_share := (1 / 3 : ℚ)
def local_crisis_fund_share := (1 / 2 : ℚ)
def remaining_share := (1 / 4 : ℚ)

def community_pantry_amount : ℚ := annual_donation * community_pantry_share
def local_crisis_amount : ℚ := annual_donation * local_crisis_fund_share
def remaining_amount : ℚ := annual_donation - community_pantry_amount - local_crisis_amount
def livelihood_amount : ℚ := remaining_amount * remaining_share
def contingency_amount : ℚ := remaining_amount - livelihood_amount

theorem contingency_fund_correct :
  contingency_amount = 30 := by
  -- Proof goes here (to be completed)
  sorry

end contingency_fund_correct_l156_156318


namespace solution_set_of_quadratic_inequality_l156_156463

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + 2 * x - 3 < 0} = {x : ℝ | -3 < x ∧ x < 1} :=
sorry

end solution_set_of_quadratic_inequality_l156_156463


namespace fixed_point_coordinates_l156_156769

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  2 * a^(x + 1) - 3

theorem fixed_point_coordinates (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = -1 :=
by
  sorry

end fixed_point_coordinates_l156_156769


namespace card_sequence_probability_l156_156571

-- Conditions about the deck and card suits
def standard_deck : ℕ := 52
def diamond_count : ℕ := 13
def spade_count : ℕ := 13
def heart_count : ℕ := 13

-- Definition of the problem statement
def diamond_first_prob : ℚ := diamond_count / standard_deck
def spade_second_prob : ℚ := spade_count / (standard_deck - 1)
def heart_third_prob : ℚ := heart_count / (standard_deck - 2)

-- Theorem statement for the required probability
theorem card_sequence_probability : 
    diamond_first_prob * spade_second_prob * heart_third_prob = 13 / 780 :=
by
  sorry

end card_sequence_probability_l156_156571


namespace convex_g_inequality_l156_156446

noncomputable def g (x : ℝ) : ℝ := x * Real.log x

theorem convex_g_inequality (a b : ℝ) (h : 0 < a ∧ a < b) :
  g a + g b - 2 * g ((a + b) / 2) > 0 := 
sorry

end convex_g_inequality_l156_156446


namespace value_subtracted_l156_156495

theorem value_subtracted (x y : ℤ) (h1 : (x - 5) / 7 = 7) (h2 : (x - y) / 13 = 4) : y = 2 :=
sorry

end value_subtracted_l156_156495


namespace remainder_when_divided_by_7_l156_156078

theorem remainder_when_divided_by_7 :
  let a := -1234
  let b := 1984
  let c := -1460
  let d := 2008
  (a * b * c * d) % 7 = 0 :=
by
  sorry

end remainder_when_divided_by_7_l156_156078


namespace area_of_EFGH_l156_156389

-- Definitions based on given conditions
def shorter_side : ℝ := 4
def longer_side : ℝ := 8
def smaller_rectangle_area : ℝ := shorter_side * longer_side
def larger_rectangle_width : ℝ := longer_side
def larger_rectangle_height : ℝ := 2 * longer_side

-- Theorem stating the area of the larger rectangle
theorem area_of_EFGH : larger_rectangle_width * larger_rectangle_height = 128 := by
  -- Proof goes here
  sorry

end area_of_EFGH_l156_156389


namespace not_on_graph_ln_l156_156269

theorem not_on_graph_ln {a b : ℝ} (h : b = Real.log a) : ¬ (1 + b = Real.log (a + Real.exp 1)) :=
by
  sorry

end not_on_graph_ln_l156_156269


namespace range_of_a_l156_156504

theorem range_of_a :
  (∃ a : ℝ, (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0)
    ↔ a ≤ -2 ∨ a = 1) := 
sorry

end range_of_a_l156_156504


namespace intersection_when_a_eq_4_range_for_A_subset_B_l156_156176

-- Define the conditions
def setA : Set ℝ := { x | (1 - x) / (x - 7) > 0 }
def setB (a : ℝ) : Set ℝ := { x | x^2 - 2 * x - a^2 - 2 * a < 0 }

-- First proof goal: When a = 4, find A ∩ B
theorem intersection_when_a_eq_4 :
  setA ∩ (setB 4) = { x : ℝ | 1 < x ∧ x < 6 } :=
sorry

-- Second proof goal: Find the range for a such that A ⊆ B
theorem range_for_A_subset_B :
  { a : ℝ | setA ⊆ setB a } = { a : ℝ | a ≤ -7 ∨ a ≥ 5 } :=
sorry

end intersection_when_a_eq_4_range_for_A_subset_B_l156_156176


namespace alfred_bill_days_l156_156140

-- Definitions based on conditions
def combined_work_rate := 1 / 24
def alfred_to_bill_ratio := 2 / 3

-- Theorem statement
theorem alfred_bill_days (A B : ℝ) (ha : A = alfred_to_bill_ratio * B) (hcombined : A + B = combined_work_rate) : 
  A = 1 / 60 ∧ B = 1 / 40 :=
by
  sorry

end alfred_bill_days_l156_156140


namespace find_initial_number_of_girls_l156_156304

theorem find_initial_number_of_girls (b g : ℕ) : 
  (b = 3 * (g - 12)) ∧ (4 * (b - 36) = g - 12) → g = 25 :=
by
  intros h
  sorry

end find_initial_number_of_girls_l156_156304


namespace min_sum_abc_l156_156646

theorem min_sum_abc (a b c : ℕ) (h1 : a * b * c = 3960) : a + b + c ≥ 150 :=
sorry

end min_sum_abc_l156_156646


namespace a_plus_b_minus_c_in_S_l156_156555

-- Define the sets P, Q, and S
def P := {x : ℤ | ∃ k : ℤ, x = 3 * k}
def Q := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def S := {x : ℤ | ∃ k : ℤ, x = 3 * k - 1}

-- Define the elements a, b, and c as members of sets P, Q, and S respectively
variables (a b c : ℤ)
variable (ha : a ∈ P) -- a ∈ P
variable (hb : b ∈ Q) -- b ∈ Q
variable (hc : c ∈ S) -- c ∈ S

-- Theorem statement proving the question
theorem a_plus_b_minus_c_in_S : a + b - c ∈ S := sorry

end a_plus_b_minus_c_in_S_l156_156555


namespace geometric_sequence_fifth_term_l156_156354

theorem geometric_sequence_fifth_term (x y : ℝ) (r : ℝ) 
  (h1 : x + y ≠ 0) (h2 : x - y ≠ 0) (h3 : x ≠ 0) (h4 : y ≠ 0)
  (h_ratio_1 : (x - y) / (x + y) = r)
  (h_ratio_2 : (x^2 * y) / (x - y) = r)
  (h_ratio_3 : (x * y^2) / (x^2 * y) = r) :
  (x * y^2 * ((y / x) * r)) = y^3 := 
by 
  sorry

end geometric_sequence_fifth_term_l156_156354


namespace find_height_of_cuboid_l156_156777

variable (A : ℝ) (V : ℝ) (h : ℝ)

theorem find_height_of_cuboid (h_eq : h = V / A) (A_eq : A = 36) (V_eq : V = 252) : h = 7 :=
by
  sorry

end find_height_of_cuboid_l156_156777


namespace incorrect_expression_l156_156756

theorem incorrect_expression : ¬ (5 = (Real.sqrt (-5))^2) :=
by
  sorry

end incorrect_expression_l156_156756


namespace triangle_centers_exist_l156_156189

structure Triangle (α : Type _) [OrderedCommSemiring α] :=
(A B C : α × α)

noncomputable def circumcenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def incenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def excenter {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

noncomputable def centroid {α : Type _} [OrderedCommSemiring α] (T : Triangle α) : α × α :=
sorry

theorem triangle_centers_exist {α : Type _} [OrderedCommSemiring α] (T : Triangle α) :
  ∃ K O Oc S : α × α, K = circumcenter T ∧ O = incenter T ∧ Oc = excenter T ∧ S = centroid T :=
by
  refine ⟨circumcenter T, incenter T, excenter T, centroid T, ⟨rfl, rfl, rfl, rfl⟩⟩

end triangle_centers_exist_l156_156189


namespace total_cost_is_103_l156_156185

-- Base cost of the plan is 20 dollars
def base_cost : ℝ := 20

-- Cost per text message in dollars
def cost_per_text : ℝ := 0.10

-- Cost per minute over 25 hours in dollars
def cost_per_minute_over_limit : ℝ := 0.15

-- Number of text messages sent
def text_messages : ℕ := 200

-- Total hours talked
def hours_talked : ℝ := 32

-- Free minutes (25 hours)
def free_minutes : ℝ := 25 * 60

-- Calculating the extra minutes talked
def extra_minutes : ℝ := (hours_talked * 60) - free_minutes

-- Total cost
def total_cost : ℝ :=
  base_cost +
  (text_messages * cost_per_text) +
  (extra_minutes * cost_per_minute_over_limit)

-- Proving that the total cost is 103 dollars
theorem total_cost_is_103 : total_cost = 103 := by
  sorry

end total_cost_is_103_l156_156185


namespace dice_surface_dots_l156_156707

def total_dots_on_die := 1 + 2 + 3 + 4 + 5 + 6

def total_dots_on_seven_dice := 7 * total_dots_on_die

def hidden_dots_on_central_die := total_dots_on_die

def visible_dots_on_surface := total_dots_on_seven_dice - hidden_dots_on_central_die

theorem dice_surface_dots : visible_dots_on_surface = 105 := by
  sorry

end dice_surface_dots_l156_156707


namespace smallest_b_l156_156466

theorem smallest_b (a b : ℕ) (hp : a > 0) (hq : b > 0) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 8) : b = 4 :=
sorry

end smallest_b_l156_156466


namespace abs_inequality_solution_l156_156821

theorem abs_inequality_solution (x : ℝ) : 
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) := 
sorry

end abs_inequality_solution_l156_156821


namespace jacket_cost_correct_l156_156452

-- Definitions based on given conditions
def total_cost : ℝ := 33.56
def cost_shorts : ℝ := 13.99
def cost_shirt : ℝ := 12.14
def cost_jacket : ℝ := 7.43

-- Formal statement of the proof problem in Lean 4
theorem jacket_cost_correct :
  total_cost = cost_shorts + cost_shirt + cost_jacket :=
by
  sorry

end jacket_cost_correct_l156_156452


namespace probability_YW_correct_l156_156619

noncomputable def probability_YW_greater_than_six_sqrt_three (XY YZ XZ YW : ℝ) : ℝ :=
  if H : XY = 12 ∧ YZ = 6 ∧ XZ = 6 * Real.sqrt 3 then 
    if YW > 6 * Real.sqrt 3 then (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3
    else 0
  else 0

theorem probability_YW_correct : probability_YW_greater_than_six_sqrt_three 12 6 (6 * Real.sqrt 3) (6 * Real.sqrt 3) = (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3 :=
sorry

end probability_YW_correct_l156_156619


namespace mary_baseball_cards_count_l156_156020

def mary_initial_cards := 18
def mary_torn_cards := 8
def fred_gift_cards := 26
def mary_bought_cards := 40
def exchange_with_tom := 0
def mary_lost_cards := 5
def trade_with_lisa_gain := 1
def exchange_with_alex_loss := 2

theorem mary_baseball_cards_count : 
  mary_initial_cards - mary_torn_cards
  + fred_gift_cards
  + mary_bought_cards 
  + exchange_with_tom
  - mary_lost_cards
  + trade_with_lisa_gain 
  - exchange_with_alex_loss 
  = 70 := 
by
  sorry

end mary_baseball_cards_count_l156_156020


namespace problem1_problem2_l156_156889

theorem problem1 (a b : ℤ) (h : Even (5 * b + a)) : Even (a - 3 * b) :=
sorry

theorem problem2 (a b : ℤ) (h : Odd (5 * b + a)) : Odd (a - 3 * b) :=
sorry

end problem1_problem2_l156_156889


namespace arithmetic_seq_a9_l156_156037

theorem arithmetic_seq_a9 (a : ℕ → ℤ) (h1 : a 3 - a 2 = -2) (h2 : a 7 = -2) : a 9 = -6 := 
by sorry

end arithmetic_seq_a9_l156_156037


namespace reciprocal_of_neg_one_seventh_l156_156171

theorem reciprocal_of_neg_one_seventh :
  (∃ x : ℚ, - (1 / 7) * x = 1) → (-7) * (- (1 / 7)) = 1 :=
by
  sorry

end reciprocal_of_neg_one_seventh_l156_156171


namespace quadratic_polynomial_fourth_power_l156_156872

theorem quadratic_polynomial_fourth_power {a b c : ℤ} (h : ∀ x : ℤ, ∃ k : ℤ, ax^2 + bx + c = k^4) : a = 0 ∧ b = 0 :=
sorry

end quadratic_polynomial_fourth_power_l156_156872


namespace trajectory_is_eight_rays_l156_156763

open Real

def trajectory_of_point (x y : ℝ) : Prop :=
  abs (abs x - abs y) = 2

theorem trajectory_is_eight_rays :
  ∃ (x y : ℝ), trajectory_of_point x y :=
sorry

end trajectory_is_eight_rays_l156_156763


namespace tangent_line_a_zero_range_a_if_fx_neg_max_value_a_one_l156_156857

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + Real.log x) - x * Real.exp x

theorem tangent_line_a_zero (x : ℝ) (y : ℝ) : 
  a = 0 ∧ x = 1 → (2 * Real.exp 1) * x + y - Real.exp 1 = 0 :=
sorry

theorem range_a_if_fx_neg (a : ℝ) : 
  (∀ x ≥ 1, f a x < 0) → a < Real.exp 1 :=
sorry

theorem max_value_a_one (x : ℝ) : 
  a = 1 → x = (Real.exp 1)⁻¹ → f 1 x = -1 :=
sorry

end tangent_line_a_zero_range_a_if_fx_neg_max_value_a_one_l156_156857


namespace percentage_increase_in_pay_rate_l156_156813

-- Given conditions
def regular_rate : ℝ := 10
def total_surveys : ℕ := 50
def cellphone_surveys : ℕ := 35
def total_earnings : ℝ := 605

-- We need to demonstrate that the percentage increase in the pay rate for surveys involving the use of her cellphone is 30%
theorem percentage_increase_in_pay_rate :
  let earnings_at_regular_rate := regular_rate * total_surveys
  let earnings_from_cellphone_surveys := total_earnings - earnings_at_regular_rate
  let rate_per_cellphone_survey := earnings_from_cellphone_surveys / cellphone_surveys
  let increase_in_rate := rate_per_cellphone_survey - regular_rate
  let percentage_increase := (increase_in_rate / regular_rate) * 100
  percentage_increase = 30 :=
by
  sorry

end percentage_increase_in_pay_rate_l156_156813


namespace dhoni_savings_l156_156188

theorem dhoni_savings :
  let earnings := 100
  let rent := 0.25 * earnings
  let dishwasher := rent - (0.10 * rent)
  let utilities := 0.15 * earnings
  let groceries := 0.20 * earnings
  let transportation := 0.12 * earnings
  let total_spent := rent + dishwasher + utilities + groceries + transportation
  earnings - total_spent = 0.055 * earnings :=
by
  sorry

end dhoni_savings_l156_156188


namespace hoodies_ownership_l156_156274

-- Step a): Defining conditions
variables (Fiona_casey_hoodies_total: ℕ) (Casey_difference: ℕ) (Alex_hoodies: ℕ)

-- Functions representing the constraints
def hoodies_owned_by_Fiona (F : ℕ) : Prop :=
  (F + (F + 2) + 3 = 15)

-- Step c): Prove the correct number of hoodies owned by each
theorem hoodies_ownership (F : ℕ) (H1 : hoodies_owned_by_Fiona F) : 
  F = 5 ∧ (F + 2 = 7) ∧ (3 = 3) :=
by {
  -- Skipping proof details
  sorry
}

end hoodies_ownership_l156_156274


namespace constant_function_on_chessboard_l156_156879

theorem constant_function_on_chessboard
  (f : ℤ × ℤ → ℝ)
  (h_nonneg : ∀ (m n : ℤ), 0 ≤ f (m, n))
  (h_mean : ∀ (m n : ℤ), f (m, n) = (f (m + 1, n) + f (m - 1, n) + f (m, n + 1) + f (m, n - 1)) / 4) :
  ∃ c : ℝ, ∀ (m n : ℤ), f (m, n) = c :=
sorry

end constant_function_on_chessboard_l156_156879


namespace inequality_pow_l156_156982

variable {n : ℕ}

theorem inequality_pow (hn : n > 0) : 
  (3:ℝ) / 2 ≤ (1 + (1:ℝ) / (2 * n)) ^ n ∧ (1 + (1:ℝ) / (2 * n)) ^ n < 2 := 
sorry

end inequality_pow_l156_156982


namespace crackers_shared_equally_l156_156076

theorem crackers_shared_equally : ∀ (matthew_crackers friends_crackers left_crackers friends : ℕ),
  matthew_crackers = 23 →
  left_crackers = 11 →
  friends = 2 →
  matthew_crackers - left_crackers = friends_crackers →
  friends_crackers = friends * 6 :=
by
  intro matthew_crackers friends_crackers left_crackers friends
  sorry

end crackers_shared_equally_l156_156076


namespace Q1_no_such_a_b_Q2_no_such_a_b_c_l156_156252

theorem Q1_no_such_a_b :
  ∀ (a b : ℕ), (0 < a) ∧ (0 < b) → ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, k^2 = 2^n * a + 5^n * b) := sorry

theorem Q2_no_such_a_b_c :
  ∀ (a b c : ℕ), (0 < a) ∧ (0 < b) ∧ (0 < c) → ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, k^2 = 2^n * a + 5^n * b + c) := sorry

end Q1_no_such_a_b_Q2_no_such_a_b_c_l156_156252


namespace households_soap_usage_l156_156637

theorem households_soap_usage
  (total_households : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (only_B_ratio : ℕ)
  (B := only_B_ratio * both) :
  total_households = 200 →
  neither = 80 →
  both = 40 →
  only_B_ratio = 3 →
  (total_households - neither - both - B = 40) :=
by
  intros
  sorry

end households_soap_usage_l156_156637


namespace simplify_tan_product_l156_156528

-- Mathematical Conditions
def tan_inv (x : ℝ) : ℝ := sorry
noncomputable def tan (θ : ℝ) : ℝ := sorry

-- Problem statement to be proven
theorem simplify_tan_product (x y : ℝ) (hx : tan_inv x = 10) (hy : tan_inv y = 35) :
  (1 + x) * (1 + y) = 2 :=
sorry

end simplify_tan_product_l156_156528


namespace total_marbles_l156_156652

theorem total_marbles (marbles_per_row_8 : ℕ) (rows_of_9 : ℕ) (marbles_per_row_1 : ℕ) (rows_of_4 : ℕ) 
  (h1 : marbles_per_row_8 = 9) 
  (h2 : rows_of_9 = 8) 
  (h3 : marbles_per_row_1 = 4) 
  (h4 : rows_of_4 = 1) : 
  (marbles_per_row_8 * rows_of_9 + marbles_per_row_1 * rows_of_4) = 76 :=
by
  sorry

end total_marbles_l156_156652


namespace student_failed_by_l156_156465

-- Conditions
def total_marks : ℕ := 440
def passing_percentage : ℝ := 0.50
def marks_obtained : ℕ := 200

-- Calculate passing marks
noncomputable def passing_marks : ℝ := passing_percentage * total_marks

-- Definition of the problem to be proved
theorem student_failed_by : passing_marks - marks_obtained = 20 := 
by
  sorry

end student_failed_by_l156_156465


namespace problem_condition_l156_156226

variable {f : ℝ → ℝ}

theorem problem_condition (h_diff : Differentiable ℝ f) (h_ineq : ∀ x : ℝ, f x < iteratedDeriv 2 f x) : 
  e^2019 * f (-2019) < f 0 ∧ f 2019 > e^2019 * f 0 :=
by
  sorry

end problem_condition_l156_156226


namespace theater_seat_count_l156_156396

theorem theater_seat_count :
  ∃ n : ℕ, n < 60 ∧ n % 9 = 5 ∧ n % 6 = 3 ∧ n = 41 :=
sorry

end theater_seat_count_l156_156396


namespace jill_sod_area_l156_156851

noncomputable def area_of_sod (yard_width yard_length sidewalk_width sidewalk_length flower_bed1_depth flower_bed1_length flower_bed2_depth flower_bed2_length flower_bed3_width flower_bed3_length flower_bed4_width flower_bed4_length : ℝ) : ℝ :=
  let yard_area := yard_width * yard_length
  let sidewalk_area := sidewalk_width * sidewalk_length
  let flower_bed1_area := flower_bed1_depth * flower_bed1_length
  let flower_bed2_area := flower_bed2_depth * flower_bed2_length
  let flower_bed3_area := flower_bed3_width * flower_bed3_length
  let flower_bed4_area := flower_bed4_width * flower_bed4_length
  let total_non_sod_area := sidewalk_area + 2 * flower_bed1_area + flower_bed2_area + flower_bed3_area + flower_bed4_area
  yard_area - total_non_sod_area

theorem jill_sod_area : 
  area_of_sod 200 50 3 50 4 25 4 25 10 12 7 8 = 9474 := by sorry

end jill_sod_area_l156_156851


namespace print_time_l156_156145

theorem print_time (P R: ℕ) (hR : R = 24) (hP : P = 360) (T : ℕ) : T = P / R → T = 15 := by
  intros h
  rw [hR, hP] at h
  exact h

end print_time_l156_156145


namespace cost_price_is_50_l156_156297

-- Define the conditions
def selling_price : ℝ := 80
def profit_rate : ℝ := 0.6

-- The cost price should be proven to be 50
def cost_price (C : ℝ) : Prop :=
  selling_price = C + (C * profit_rate)

theorem cost_price_is_50 : ∃ C : ℝ, cost_price C ∧ C = 50 := by
  sorry

end cost_price_is_50_l156_156297


namespace total_dogs_l156_156002

-- Definitions of conditions
def brown_dogs : Nat := 20
def white_dogs : Nat := 10
def black_dogs : Nat := 15

-- Theorem to prove the total number of dogs
theorem total_dogs : brown_dogs + white_dogs + black_dogs = 45 := by
  -- Placeholder for proof
  sorry

end total_dogs_l156_156002


namespace cream_ratio_l156_156064

variable (servings : ℕ) (fat_per_serving : ℕ) (fat_per_cup : ℕ)
variable (h_servings : servings = 4) (h_fat_per_serving : fat_per_serving = 11) (h_fat_per_cup : fat_per_cup = 88)

theorem cream_ratio (total_fat : ℕ) (h_total_fat : total_fat = fat_per_serving * servings) :
  (total_fat : ℚ) / fat_per_cup = 1 / 2 :=
by
  sorry

end cream_ratio_l156_156064


namespace find_f_at_3_l156_156432

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq : ∀ (x : ℝ), x ≠ 2 / 3 → f x + f ((x + 2) / (2 - 3 * x)) = 2 * x

theorem find_f_at_3 : f 3 = 3 :=
by {
  sorry
}

end find_f_at_3_l156_156432


namespace balloon_altitude_l156_156896

theorem balloon_altitude 
  (temp_diff_per_1000m : ℝ)
  (altitude_temp : ℝ) 
  (ground_temp : ℝ)
  (altitude : ℝ) 
  (h1 : temp_diff_per_1000m = 6) 
  (h2 : altitude_temp = -2)
  (h3 : ground_temp = 5) :
  altitude = 7/6 :=
by sorry

end balloon_altitude_l156_156896


namespace solve_linear_eq_l156_156871

theorem solve_linear_eq (x y : ℤ) : 2 * x + 3 * y = 0 ↔ (x, y) = (3, -2) := sorry

end solve_linear_eq_l156_156871


namespace bake_sale_earnings_eq_400_l156_156207

/-
  The problem statement derived from the given bake sale problem.
  We are to verify that the bake sale earned 400 dollars.
-/

def total_donation (bake_sale_earnings : ℕ) :=
  ((bake_sale_earnings - 100) / 2) + 10

theorem bake_sale_earnings_eq_400 (X : ℕ) (h : total_donation X = 160) : X = 400 :=
by
  sorry

end bake_sale_earnings_eq_400_l156_156207


namespace dynaco_shares_l156_156919

theorem dynaco_shares (M D : ℕ) 
  (h1 : M + D = 300)
  (h2 : 36 * M + 44 * D = 12000) : 
  D = 150 :=
sorry

end dynaco_shares_l156_156919


namespace paint_grid_l156_156157

theorem paint_grid (paint : Fin 3 × Fin 3 → Bool) (no_adjacent : ∀ i j, (paint (i, j) = true) → (paint (i+1, j) = false) ∧ (paint (i-1, j) = false) ∧ (paint (i, j+1) = false) ∧ (paint (i, j-1) = false)) : 
  ∃! (count : ℕ), count = 8 :=
sorry

end paint_grid_l156_156157


namespace average_of_25_results_is_24_l156_156579

theorem average_of_25_results_is_24 
  (first12_sum : ℕ)
  (last12_sum : ℕ)
  (result13 : ℕ)
  (n1 n2 n3 : ℕ)
  (h1 : n1 = 12)
  (h2 : n2 = 12)
  (h3 : n3 = 25)
  (avg_first12 : first12_sum = 14 * n1)
  (avg_last12 : last12_sum = 17 * n2)
  (res_13 : result13 = 228) :
  (first12_sum + last12_sum + result13) / n3 = 24 :=
by
  sorry

end average_of_25_results_is_24_l156_156579


namespace g_is_odd_l156_156198

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  sorry

end g_is_odd_l156_156198


namespace max_profit_jars_max_tax_value_l156_156004

-- Part a: Prove the optimal number of jars for maximum profit
theorem max_profit_jars (Q : ℝ) 
  (h : ∀ Q, Q >= 0 → (310 - 3 * Q) * Q - 10 * Q ≤ (310 - 3 * 50) * 50 - 10 * 50):
  Q = 50 :=
sorry

-- Part b: Prove the optimal tax for maximum tax revenue
theorem max_tax_value (t : ℝ) 
  (h : ∀ t, t >= 0 → ((300 * t - t^2) / 6) ≤ ((300 * 150 - 150^2) / 6)):
  t = 150 :=
sorry

end max_profit_jars_max_tax_value_l156_156004


namespace negation_of_universal_abs_nonneg_l156_156657

theorem negation_of_universal_abs_nonneg :
  (¬ (∀ x : ℝ, |x| ≥ 0)) ↔ (∃ x : ℝ, |x| < 0) :=
by
  sorry

end negation_of_universal_abs_nonneg_l156_156657


namespace candy_count_after_giving_l156_156631

def numKitKats : ℕ := 5
def numHersheyKisses : ℕ := 3 * numKitKats
def numNerds : ℕ := 8
def numLollipops : ℕ := 11
def numBabyRuths : ℕ := 10
def numReeseCups : ℕ := numBabyRuths / 2
def numLollipopsGivenAway : ℕ := 5

def totalCandyBefore : ℕ := numKitKats + numHersheyKisses + numNerds + numLollipops + numBabyRuths + numReeseCups
def totalCandyAfter : ℕ := totalCandyBefore - numLollipopsGivenAway

theorem candy_count_after_giving : totalCandyAfter = 49 := by
  sorry

end candy_count_after_giving_l156_156631


namespace find_x_l156_156702

-- Introducing the main theorem
theorem find_x (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (x : ℝ) (h_x : 0 < x) : 
  let r := (4 * a) ^ (4 * b)
  let y := x ^ 2
  r = a ^ b * y → 
  x = 16 ^ b * a ^ (1.5 * b) :=
by
  sorry

end find_x_l156_156702


namespace company_profit_growth_l156_156324

theorem company_profit_growth (x : ℝ) (h : 1.6 * (1 + x / 100)^2 = 2.5) : x = 25 :=
sorry

end company_profit_growth_l156_156324


namespace magician_draws_two_cards_l156_156908

-- Define the total number of unique cards
def total_cards : ℕ := 15^2

-- Define the number of duplicate cards
def duplicate_cards : ℕ := 15

-- Define the number of ways to choose 2 cards from the duplicate cards
def choose_two_duplicates : ℕ := Nat.choose 15 2

-- Define the number of ways to choose 1 duplicate card and 1 non-duplicate card
def choose_one_duplicate_one_nonduplicate : ℕ := (15 * (total_cards - 15 - 14 - 14))

-- The main theorem to prove
theorem magician_draws_two_cards : choose_two_duplicates + choose_one_duplicate_one_nonduplicate = 2835 := by
  sorry

end magician_draws_two_cards_l156_156908


namespace pure_alcohol_to_add_l156_156208

-- Variables and known values
variables (x : ℝ) -- amount of pure alcohol added
def initial_volume : ℝ := 6 -- initial solution volume in liters
def initial_concentration : ℝ := 0.35 -- initial alcohol concentration
def target_concentration : ℝ := 0.50 -- target alcohol concentration

-- Conditions
def initial_pure_alcohol : ℝ := initial_volume * initial_concentration

-- Statement of the problem
theorem pure_alcohol_to_add :
  (2.1 + x) / (initial_volume + x) = target_concentration ↔ x = 1.8 :=
by
  sorry

end pure_alcohol_to_add_l156_156208


namespace pull_ups_per_time_l156_156508

theorem pull_ups_per_time (pull_ups_week : ℕ) (times_day : ℕ) (days_week : ℕ)
  (h1 : pull_ups_week = 70) (h2 : times_day = 5) (h3 : days_week = 7) :
  pull_ups_week / (times_day * days_week) = 2 := by
  sorry

end pull_ups_per_time_l156_156508


namespace area_of_new_shape_l156_156214

noncomputable def unit_equilateral_triangle_area : ℝ :=
  (1 : ℝ)^2 * Real.sqrt 3 / 4

noncomputable def area_removed_each_step (k : ℕ) : ℝ :=
  3 * (4 ^ (k - 1)) * (Real.sqrt 3 / (4 * (9 ^ k)))

noncomputable def total_removed_area : ℝ :=
  ∑' k, 3 * (4 ^ (k - 1)) * (Real.sqrt 3 / (4 * (9 ^ k)))

noncomputable def final_area := unit_equilateral_triangle_area - total_removed_area

theorem area_of_new_shape :
  final_area = Real.sqrt 3 / 10 := sorry

end area_of_new_shape_l156_156214


namespace price_of_first_variety_l156_156461

theorem price_of_first_variety (P : ℝ) (h1 : 1 * P + 1 * 135 + 2 * 175.5 = 153 * 4) : P = 126 :=
sorry

end price_of_first_variety_l156_156461


namespace find_m_l156_156040

theorem find_m (m : ℤ) : m < 2 * Real.sqrt 3 ∧ 2 * Real.sqrt 3 < m + 1 → m = 3 :=
sorry

end find_m_l156_156040


namespace find_a6_l156_156277

variable {a : ℕ → ℝ} -- Sequence a is indexed by natural numbers and the terms are real numbers.

-- Conditions
def a_is_geom_seq (a : ℕ → ℝ) := ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q)
def a1_eq_4 (a : ℕ → ℝ) := a 1 = 4
def a3_eq_a2_mul_a4 (a : ℕ → ℝ) := a 3 = a 2 * a 4

theorem find_a6 (a : ℕ → ℝ) 
  (h1 : a_is_geom_seq a)
  (h2 : a1_eq_4 a)
  (h3 : a3_eq_a2_mul_a4 a) : 
  a 6 = 1 / 8 ∨ a 6 = - (1 / 8) := 
by 
  sorry

end find_a6_l156_156277


namespace min_2x3y2z_l156_156221

noncomputable def min_value (x y z : ℝ) : ℝ := 2 * (x^3) * (y^2) * z

theorem min_2x3y2z (x y z : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h : (1/x) + (1/y) + (1/z) = 9) :
  min_value x y z = 2 / 675 :=
sorry

end min_2x3y2z_l156_156221


namespace find_t_l156_156772

-- Define the utility function based on hours of reading and playing basketball
def utility (reading_hours : ℝ) (basketball_hours : ℝ) : ℝ :=
  reading_hours * basketball_hours

-- Define the conditions for Wednesday and Thursday utilities
def wednesday_utility (t : ℝ) : ℝ :=
  t * (10 - t)

def thursday_utility (t : ℝ) : ℝ :=
  (3 - t) * (t + 4)

-- The main theorem stating the equivalence of utilities implies t = 3
theorem find_t (t : ℝ) (h : wednesday_utility t = thursday_utility t) : t = 3 :=
by
  -- Skip proof with sorry
  sorry

end find_t_l156_156772


namespace brianne_january_savings_l156_156111

theorem brianne_january_savings (S : ℝ) (h : 16 * S = 160) : S = 10 :=
sorry

end brianne_january_savings_l156_156111


namespace solution_set_of_inequality_l156_156116

theorem solution_set_of_inequality:
  {x : ℝ | |x - 5| + |x + 1| < 8} = {x : ℝ | -2 < x ∧ x < 6} :=
sorry

end solution_set_of_inequality_l156_156116


namespace emmalyn_earnings_l156_156531

theorem emmalyn_earnings :
  let rate_per_meter := 0.20
  let number_of_fences := 50
  let length_of_each_fence := 500
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * rate_per_meter
  total_income = 5000 :=
by
  sorry

end emmalyn_earnings_l156_156531


namespace find_largest_t_l156_156126

theorem find_largest_t (t : ℝ) : 
  (15 * t^2 - 38 * t + 14) / (4 * t - 3) + 6 * t = 7 * t - 2 → t ≤ 1 := 
by 
  intro h
  sorry

end find_largest_t_l156_156126


namespace triangle_area_correct_l156_156128

open Real

def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2 - v1.2 * v2.1 - v2.2 * v3.1 - v3.2 * v1.1)

theorem triangle_area_correct :
  triangle_area (4, 6) (-4, 6) (0, 2) = 16 :=
by
  sorry

end triangle_area_correct_l156_156128


namespace age_proof_l156_156986

theorem age_proof (y d : ℕ)
  (h1 : y = 4 * d)
  (h2 : y - 7 = 11 * (d - 7)) :
  y = 48 ∧ d = 12 :=
by
  -- The proof is omitted
  sorry

end age_proof_l156_156986


namespace number_of_ways_to_take_one_ball_from_pockets_number_of_ways_to_take_one_ball_each_from_pockets_l156_156932

-- Let's define the conditions.
def balls_in_first_pocket : Nat := 2
def balls_in_second_pocket : Nat := 4
def balls_in_third_pocket : Nat := 5

-- Proof for the first question
theorem number_of_ways_to_take_one_ball_from_pockets : 
  balls_in_first_pocket + balls_in_second_pocket + balls_in_third_pocket = 11 := 
by
  sorry

-- Proof for the second question
theorem number_of_ways_to_take_one_ball_each_from_pockets : 
  balls_in_first_pocket * balls_in_second_pocket * balls_in_third_pocket = 40 := 
by
  sorry

end number_of_ways_to_take_one_ball_from_pockets_number_of_ways_to_take_one_ball_each_from_pockets_l156_156932


namespace remainder_67_pow_67_plus_67_mod_68_l156_156472

theorem remainder_67_pow_67_plus_67_mod_68 :
  (67 ^ 67 + 67) % 68 = 66 :=
by
  -- Skip the proof for now
  sorry

end remainder_67_pow_67_plus_67_mod_68_l156_156472


namespace square_side_length_l156_156270

variables (s : ℝ) (π : ℝ)
  
theorem square_side_length (h : 4 * s = π * s^2 / 2) : s = 8 / π :=
by sorry

end square_side_length_l156_156270


namespace marble_distribution_l156_156000

theorem marble_distribution (a b c : ℚ) (h1 : a + b + c = 78) (h2 : a = 3 * b + 2) (h3 : b = c / 2) : 
  a = 40 ∧ b = 38 / 3 ∧ c = 76 / 3 :=
by
  sorry

end marble_distribution_l156_156000


namespace central_angle_relation_l156_156736

theorem central_angle_relation
  (R L : ℝ)
  (α : ℝ)
  (r l β : ℝ)
  (h1 : r = 0.5 * R)
  (h2 : l = 1.5 * L)
  (h3 : L = R * α)
  (h4 : l = r * β) : 
  β = 3 * α :=
by
  sorry

end central_angle_relation_l156_156736


namespace bread_cost_l156_156288

theorem bread_cost {packs_meat packs_cheese sandwiches : ℕ} 
  (cost_meat cost_cheese cost_sandwich coupon_meat coupon_cheese total_cost : ℝ) 
  (h_meat_cost : cost_meat = 5.00) 
  (h_cheese_cost : cost_cheese = 4.00)
  (h_coupon_meat : coupon_meat = 1.00)
  (h_coupon_cheese : coupon_cheese = 1.00)
  (h_cost_sandwich : cost_sandwich = 2.00)
  (h_packs_meat : packs_meat = 2)
  (h_packs_cheese : packs_cheese = 2)
  (h_sandwiches : sandwiches = 10)
  (h_total_revenue : total_cost = sandwiches * cost_sandwich) :
  ∃ (bread_cost : ℝ), bread_cost = total_cost - ((packs_meat * cost_meat - coupon_meat) + (packs_cheese * cost_cheese - coupon_cheese)) :=
sorry

end bread_cost_l156_156288


namespace andrei_kolya_ages_l156_156678

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + (n / 1000)

theorem andrei_kolya_ages :
  ∃ (y1 y2 : ℕ), (sum_of_digits y1 = 2021 - y1) ∧ (sum_of_digits y2 = 2021 - y2) ∧ (y1 ≠ y2) ∧ ((2022 - y1 = 8 ∧ 2022 - y2 = 26) ∨ (2022 - y1 = 26 ∧ 2022 - y2 = 8)) :=
by
  sorry

end andrei_kolya_ages_l156_156678


namespace problem_solution_l156_156255

theorem problem_solution:
  2019 ^ Real.log (Real.log 2019) - Real.log 2019 ^ Real.log 2019 = 0 :=
by
  sorry

end problem_solution_l156_156255


namespace percentage_increase_l156_156868

theorem percentage_increase (N : ℝ) (P : ℝ) (h1 : N + (P / 100) * N - (N - 25 / 100 * N) = 30) (h2 : N = 80) : P = 12.5 :=
by
  sorry

end percentage_increase_l156_156868


namespace buzz_waiter_ratio_l156_156749

def total_slices : Nat := 78
def waiter_condition (W : Nat) : Prop := W - 20 = 28

theorem buzz_waiter_ratio (W : Nat) (h : waiter_condition W) : 
  let buzz_slices := total_slices - W
  let ratio_buzz_waiter := buzz_slices / W
  ratio_buzz_waiter = 5 / 8 :=
by
  sorry

end buzz_waiter_ratio_l156_156749


namespace yearly_exports_calculation_l156_156280

variable (Y : Type) 
variable (fruit_exports_total yearly_exports : ℝ)
variable (orange_exports : ℝ := 4.25 * 10^6)
variable (fruit_exports_percent : ℝ := 0.20)
variable (orange_exports_fraction : ℝ := 1/6)

-- The main statement to prove
theorem yearly_exports_calculation
  (h1 : yearly_exports * fruit_exports_percent = fruit_exports_total)
  (h2 : fruit_exports_total * orange_exports_fraction = orange_exports) :
  yearly_exports = 127.5 * 10^6 :=
by
  -- Proof (omitted)
  sorry

end yearly_exports_calculation_l156_156280


namespace f_at_five_l156_156974

def f (n : ℕ) : ℕ := n^3 + 2 * n^2 + 3 * n + 17

theorem f_at_five : f 5 = 207 := 
by 
sorry

end f_at_five_l156_156974


namespace bus_interval_three_buses_l156_156375

theorem bus_interval_three_buses (T : ℕ) (h : T = 21) : (T * 2) / 3 = 14 :=
by
  sorry

end bus_interval_three_buses_l156_156375


namespace quadrilateral_probability_l156_156955

def total_shapes : ℕ := 6
def quadrilateral_shapes : ℕ := 3

theorem quadrilateral_probability : (quadrilateral_shapes : ℚ) / (total_shapes : ℚ) = 1 / 2 :=
by
  sorry

end quadrilateral_probability_l156_156955


namespace rectangle_dimensions_l156_156012

theorem rectangle_dimensions (a b : ℝ) 
  (h_area : a * b = 12) 
  (h_perimeter : 2 * (a + b) = 26) : 
  (a = 1 ∧ b = 12) ∨ (a = 12 ∧ b = 1) :=
sorry

end rectangle_dimensions_l156_156012


namespace josh_bought_6_CDs_l156_156788

theorem josh_bought_6_CDs 
  (numFilms : ℕ)   (numBooks : ℕ) (numCDs : ℕ)
  (costFilm : ℕ)   (costBook : ℕ) (costCD : ℕ)
  (totalSpent : ℕ) :
  numFilms = 9 → 
  numBooks = 4 → 
  costFilm = 5 → 
  costBook = 4 → 
  costCD = 3 → 
  totalSpent = 79 → 
  numCDs = (totalSpent - numFilms * costFilm - numBooks * costBook) / costCD → 
  numCDs = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  exact h7

end josh_bought_6_CDs_l156_156788


namespace remainder_of_6_power_700_mod_72_l156_156464

theorem remainder_of_6_power_700_mod_72 : (6^700) % 72 = 0 :=
by
  sorry

end remainder_of_6_power_700_mod_72_l156_156464


namespace max_students_l156_156398

theorem max_students : 
  ∃ x : ℕ, x < 100 ∧ x % 9 = 4 ∧ x % 7 = 3 ∧ ∀ y : ℕ, (y < 100 ∧ y % 9 = 4 ∧ y % 7 = 3) → y ≤ x := 
by
  sorry

end max_students_l156_156398


namespace abs_diff_of_prod_and_sum_l156_156435

theorem abs_diff_of_prod_and_sum (m n : ℝ) (h1 : m * n = 8) (h2 : m + n = 6) : |m - n| = 2 :=
by
  -- The proof is not required as per the instructions.
  sorry

end abs_diff_of_prod_and_sum_l156_156435


namespace fermat_coprime_l156_156861

theorem fermat_coprime (m n : ℕ) (hmn : m ≠ n) (hm_pos : m > 0) (hn_pos : n > 0) :
  gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 :=
sorry

end fermat_coprime_l156_156861


namespace prime_factor_of_difference_l156_156732

theorem prime_factor_of_difference (A B C : ℕ) (hA : A ≠ 0) (hABC_digits : A ≠ B ∧ A ≠ C ∧ B ≠ C) 
  (hA_range : 0 ≤ A ∧ A ≤ 9) (hB_range : 0 ≤ B ∧ B ≤ 9) (hC_range : 0 ≤ C ∧ C ≤ 9) :
  11 ∣ (100 * A + 10 * B + C) - (100 * C + 10 * B + A) :=
by
  sorry

end prime_factor_of_difference_l156_156732


namespace Pyarelal_loss_is_1800_l156_156957

noncomputable def Ashok_and_Pyarelal_loss (P L : ℝ) : Prop :=
  let Ashok_cap := (1 / 9) * P
  let total_cap := P + Ashok_cap
  let Pyarelal_ratio := P / total_cap
  let total_loss := 2000
  let Pyarelal_loss := Pyarelal_ratio * total_loss
  Pyarelal_loss = 1800

theorem Pyarelal_loss_is_1800 (P : ℝ) (h1 : P > 0) (h2 : L = 2000) :
  Ashok_and_Pyarelal_loss P L := sorry

end Pyarelal_loss_is_1800_l156_156957


namespace largest_perfect_square_factor_of_882_l156_156215

theorem largest_perfect_square_factor_of_882 : ∃ n, n * n = 441 ∧ ∀ m, m * m ∣ 882 → m * m ≤ 441 := 
by 
 sorry

end largest_perfect_square_factor_of_882_l156_156215


namespace closest_perfect_square_l156_156603

theorem closest_perfect_square (n : ℕ) (h1 : n = 325) : 
    ∃ m : ℕ, m^2 = 324 ∧ 
    (∀ k : ℕ, (k^2 ≤ n ∨ k^2 ≥ n) → (k = 18 ∨ k^2 > 361 ∨ k^2 < 289)) := 
by
  sorry

end closest_perfect_square_l156_156603


namespace solve_inequalities_l156_156083

theorem solve_inequalities (x : ℤ) :
  (1 ≤ x ∧ x < 3) ↔ 
  ((↑x - 1) / 2 < (↑x : ℝ) / 3 ∧ 2 * (↑x : ℝ) - 5 ≤ 3 * (↑x : ℝ) - 6) :=
by
  sorry

end solve_inequalities_l156_156083


namespace total_cost_kept_l156_156133

def prices_all : List ℕ := [15, 18, 20, 15, 25, 30, 20, 17, 22, 23, 29]
def prices_returned : List ℕ := [20, 25, 30, 22, 23, 29]

def total_cost (prices : List ℕ) : ℕ :=
  prices.foldl (· + ·) 0

theorem total_cost_kept :
  total_cost prices_all - total_cost prices_returned = 85 :=
by
  -- The proof steps go here
  sorry

end total_cost_kept_l156_156133


namespace polygon_interior_angle_increase_l156_156831

theorem polygon_interior_angle_increase (n : ℕ) (h : 3 ≤ n) :
  ((n + 1 - 2) * 180 - (n - 2) * 180 = 180) :=
by sorry

end polygon_interior_angle_increase_l156_156831


namespace necessary_for_A_l156_156791

-- Define the sets A, B, C as non-empty sets
variables {α : Type*} (A B C : Set α)
-- Non-empty sets
axiom non_empty_A : ∃ x, x ∈ A
axiom non_empty_B : ∃ x, x ∈ B
axiom non_empty_C : ∃ x, x ∈ C

-- Conditions
axiom union_condition : A ∪ B = C
axiom subset_condition : ¬ (B ⊆ A)

-- Statement to prove
theorem necessary_for_A (x : α) : (x ∈ C → x ∈ A) ∧ ¬(x ∈ C ↔ x ∈ A) :=
sorry

end necessary_for_A_l156_156791


namespace wendy_pictures_in_one_album_l156_156548

theorem wendy_pictures_in_one_album 
  (total_pictures : ℕ) (pictures_per_album : ℕ) (num_other_albums : ℕ)
  (h_total : total_pictures = 45) (h_pictures_per_album : pictures_per_album = 2) 
  (h_num_other_albums : num_other_albums = 9) : 
  ∃ (pictures_in_one_album : ℕ), pictures_in_one_album = 27 :=
by {
  sorry
}

end wendy_pictures_in_one_album_l156_156548


namespace three_digit_number_possibilities_l156_156605

theorem three_digit_number_possibilities (A B C : ℕ) (hA : A ≠ 0) (hC : C ≠ 0) (h_diff : A - C = 5) :
  ∃ (x : ℕ), x = 100 * A + 10 * B + C ∧ (x - (100 * C + 10 * B + A) = 495) ∧ ∃ n, n = 40 :=
by
  sorry

end three_digit_number_possibilities_l156_156605


namespace relationship_x_a_b_l156_156329

theorem relationship_x_a_b (x a b : ℝ) (h1 : x < b) (h2 : b < a) (h3 : a < 0) : 
  x^2 > a * b ∧ a * b > a^2 :=
by
  sorry

end relationship_x_a_b_l156_156329


namespace percentage_of_x_is_2x_minus_y_l156_156013

variable (x y : ℝ)
variable (h1 : x / y = 4)
variable (h2 : y ≠ 0)

theorem percentage_of_x_is_2x_minus_y :
  (2 * x - y) / x * 100 = 175 := by
  sorry

end percentage_of_x_is_2x_minus_y_l156_156013


namespace washing_time_per_cycle_l156_156880

theorem washing_time_per_cycle
    (shirts pants sweaters jeans : ℕ)
    (items_per_cycle total_hours : ℕ)
    (h1 : shirts = 18)
    (h2 : pants = 12)
    (h3 : sweaters = 17)
    (h4 : jeans = 13)
    (h5 : items_per_cycle = 15)
    (h6 : total_hours = 3) :
    ((shirts + pants + sweaters + jeans) / items_per_cycle) * (total_hours * 60) / ((shirts + pants + sweaters + jeans) / items_per_cycle) = 45 := 
by
  sorry

end washing_time_per_cycle_l156_156880


namespace find_max_sum_of_squares_l156_156535

open Real

theorem find_max_sum_of_squares 
  (a b c d : ℝ)
  (h1 : a + b = 17)
  (h2 : ab + c + d = 98)
  (h3 : ad + bc = 176)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 770 :=
sorry

end find_max_sum_of_squares_l156_156535


namespace find_train_speed_l156_156413

def length_of_platform : ℝ := 210.0168
def time_to_pass_platform : ℝ := 34
def time_to_pass_man : ℝ := 20 
def speed_of_train (L : ℝ) (V : ℝ) : Prop :=
  V = (L + length_of_platform) / time_to_pass_platform ∧ V = L / time_to_pass_man

theorem find_train_speed (L V : ℝ) (h : speed_of_train L V) : V = 54.00432 := sorry

end find_train_speed_l156_156413


namespace find_length_of_FC_l156_156802

theorem find_length_of_FC (DC CB AD AB ED FC : ℝ) (h1 : DC = 9) (h2 : CB = 10) (h3 : AB = (1 / 3) * AD) (h4 : ED = (2 / 3) * AD) : 
  FC = 13 := by
  sorry

end find_length_of_FC_l156_156802


namespace roads_with_five_possible_roads_with_four_not_possible_l156_156931

-- Problem (a)
theorem roads_with_five_possible :
  ∃ (cities : Fin 16 → Finset (Fin 16)),
  (∀ c, cities c = {d | d ≠ c ∧ d ∈ cities c}) ∧
  (∀ c, (cities c).card ≤ 5) ∧
  (∀ c d, d ≠ c → ∃ e, e ≠ c ∧ e ≠ d ∧ d ∈ cities c ∪ {e}) := by
  sorry

-- Problem (b)
theorem roads_with_four_not_possible :
  ¬ ∃ (cities : Fin 16 → Finset (Fin 16)),
  (∀ c, cities c = {d | d ≠ c ∧ d ∈ cities c}) ∧
  (∀ c, (cities c).card ≤ 4) ∧
  (∀ c d, d ≠ c → ∃ e, e ≠ c ∧ e ≠ d ∧ d ∈ cities c ∪ {e}) := by
  sorry

end roads_with_five_possible_roads_with_four_not_possible_l156_156931


namespace percentage_female_officers_on_duty_correct_l156_156964

-- Define the conditions
def total_officers_on_duty : ℕ := 144
def total_female_officers : ℕ := 400
def female_officers_on_duty : ℕ := total_officers_on_duty / 2

-- Define the percentage calculation
def percentage_female_officers_on_duty : ℕ :=
  (female_officers_on_duty * 100) / total_female_officers

-- The theorem that what we need to prove
theorem percentage_female_officers_on_duty_correct :
  percentage_female_officers_on_duty = 18 :=
by
  sorry

end percentage_female_officers_on_duty_correct_l156_156964


namespace scrap_cookie_radius_is_correct_l156_156905

noncomputable def radius_of_scrap_cookie (large_radius small_radius : ℝ) (number_of_cookies : ℕ) : ℝ :=
  have large_area : ℝ := Real.pi * large_radius^2
  have small_area : ℝ := Real.pi * small_radius^2
  have total_small_area : ℝ := small_area * number_of_cookies
  have scrap_area : ℝ := large_area - total_small_area
  Real.sqrt (scrap_area / Real.pi)

theorem scrap_cookie_radius_is_correct :
  radius_of_scrap_cookie 8 2 9 = 2 * Real.sqrt 7 :=
sorry

end scrap_cookie_radius_is_correct_l156_156905


namespace main_theorem_l156_156262

noncomputable def problem_statement : Prop :=
  ∀ x : ℂ, (x ≠ -2) →
  ((15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48) ↔
  (x = 12 + 2 * Real.sqrt 38 ∨ x = 12 - 2 * Real.sqrt 38 ∨
  x = -1/2 + Complex.I * Real.sqrt 95 / 2 ∨
  x = -1/2 - Complex.I * Real.sqrt 95 / 2)

-- Provide the main statement without the proof
theorem main_theorem : problem_statement := sorry

end main_theorem_l156_156262


namespace ronald_laundry_frequency_l156_156762

variable (Tim_laundry_frequency Ronald_laundry_frequency : ℕ)

theorem ronald_laundry_frequency :
  (Tim_laundry_frequency = 9) →
  (18 % Ronald_laundry_frequency = 0) →
  (18 % Tim_laundry_frequency = 0) →
  (Ronald_laundry_frequency ≠ 1) →
  (Ronald_laundry_frequency ≠ 18) →
  (Ronald_laundry_frequency ≠ 9) →
  (Ronald_laundry_frequency = 3) :=
by
  intros hTim hRonaldMultiple hTimMultiple hNot1 hNot18 hNot9
  sorry

end ronald_laundry_frequency_l156_156762


namespace number_of_solution_pairs_l156_156514

theorem number_of_solution_pairs : 
  ∃ n, (∀ x y : ℕ, 4 * x + 7 * y = 548 → (x > 0 ∧ y > 0) → n = 19) :=
sorry

end number_of_solution_pairs_l156_156514


namespace find_r_l156_156810

variable (k r : ℝ)

theorem find_r (h1 : 5 = k * 2^r) (h2 : 45 = k * 8^r) : r = (1/2) * Real.log 9 / Real.log 2 :=
sorry

end find_r_l156_156810


namespace time_to_meet_l156_156090

variable (distance : ℕ)
variable (speed1 speed2 time : ℕ)

-- Given conditions
def distanceAB := 480
def speedPassengerCar := 65
def speedCargoTruck := 55

-- Sum of the speeds of the two vehicles
def sumSpeeds := speedPassengerCar + speedCargoTruck

-- Prove that the time it takes for the two vehicles to meet is 4 hours
theorem time_to_meet : sumSpeeds * time = distanceAB → time = 4 :=
by
  sorry

end time_to_meet_l156_156090


namespace cans_of_soda_l156_156245

theorem cans_of_soda (S Q D : ℕ) : (4 * D * S) / Q = x :=
by
  sorry

end cans_of_soda_l156_156245


namespace least_integer_square_double_condition_l156_156641

theorem least_integer_square_double_condition : ∃ x : ℤ, x^2 = 2 * x + 75 ∧ ∀ y : ℤ, y^2 = 2 * y + 75 → x ≤ y :=
by
  use -8
  sorry

end least_integer_square_double_condition_l156_156641


namespace simplify_fraction_fraction_c_over_d_l156_156666

-- Define necessary constants and variables
variable (k : ℤ)

/-- Original expression -/
def original_expr := (6 * k + 12 + 3 : ℤ)

/-- Simplified expression -/
def simplified_expr := (2 * k + 5 : ℤ)

/-- The main theorem to prove the equivalent mathematical proof problem -/
theorem simplify_fraction : (original_expr / 3) = simplified_expr :=
by
  sorry

-- The final fraction to prove the answer
theorem fraction_c_over_d : (2 / 5 : ℚ) = 2 / 5 :=
by
  sorry

end simplify_fraction_fraction_c_over_d_l156_156666


namespace thabo_books_l156_156493

variable (P F H : Nat)

theorem thabo_books :
  P > 55 ∧ F = 2 * P ∧ H = 55 ∧ H + P + F = 280 → P - H = 20 :=
by
  sorry

end thabo_books_l156_156493


namespace side_length_is_prime_l156_156231

-- Define the integer side length of the square
variable (a : ℕ)

-- Define the conditions
def impossible_rectangle (m n : ℕ) : Prop :=
  m * n = a^2 ∧ m ≠ 1 ∧ n ≠ 1

-- Declare the theorem to be proved
theorem side_length_is_prime (h : ∀ m n : ℕ, impossible_rectangle a m n → false) : Nat.Prime a := sorry

end side_length_is_prime_l156_156231


namespace total_wheels_in_storage_l156_156345

def wheels (n_bicycles n_tricycles n_unicycles n_quadbikes : ℕ) : ℕ :=
  (n_bicycles * 2) + (n_tricycles * 3) + (n_unicycles * 1) + (n_quadbikes * 4)

theorem total_wheels_in_storage :
  let n_bicycles := 24
  let n_tricycles := 14
  let n_unicycles := 10
  let n_quadbikes := 8
  wheels n_bicycles n_tricycles n_unicycles n_quadbikes = 132 :=
by
  let n_bicycles := 24
  let n_tricycles := 14
  let n_unicycles := 10
  let n_quadbikes := 8
  show wheels n_bicycles n_tricycles n_unicycles n_quadbikes = 132
  sorry

end total_wheels_in_storage_l156_156345


namespace number_of_girls_l156_156203

theorem number_of_girls
  (B G : ℕ)
  (ratio_condition : B * 8 = 5 * G)
  (total_condition : B + G = 260) :
  G = 160 :=
by
  sorry

end number_of_girls_l156_156203


namespace systematic_sampling_student_selection_l156_156577

theorem systematic_sampling_student_selection
    (total_students : ℕ)
    (num_groups : ℕ)
    (students_per_group : ℕ)
    (third_group_selected : ℕ)
    (third_group_num : ℕ)
    (eighth_group_num : ℕ)
    (h1 : total_students = 50)
    (h2 : num_groups = 10)
    (h3 : students_per_group = total_students / num_groups)
    (h4 : students_per_group = 5)
    (h5 : 11 ≤ third_group_selected ∧ third_group_selected ≤ 15)
    (h6 : third_group_selected = 12)
    (h7 : third_group_num = 3)
    (h8 : eighth_group_num = 8) :
  eighth_group_selected = 37 :=
by
  sorry

end systematic_sampling_student_selection_l156_156577


namespace not_traversable_n_62_l156_156592

theorem not_traversable_n_62 :
  ¬ (∃ (path : ℕ → ℕ), ∀ i < 62, path (i + 1) = (path i + 8) % 62 ∨ path (i + 1) = (path i + 9) % 62 ∨ path (i + 1) = (path i + 10) % 62) :=
by sorry

end not_traversable_n_62_l156_156592


namespace ab_fraction_l156_156059

theorem ab_fraction (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 9) (h2 : a * b = 20) : 
  (1 / a + 1 / b) = 9 / 20 := 
by 
  sorry

end ab_fraction_l156_156059


namespace quadratic_polynomial_discriminant_l156_156368

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l156_156368


namespace ball_falls_in_middle_pocket_l156_156448

theorem ball_falls_in_middle_pocket (p q : ℕ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  ∃ k : ℕ, (k * p) % (2 * q) = 0 :=
by
  sorry

end ball_falls_in_middle_pocket_l156_156448


namespace tan_75_degrees_eq_l156_156073

noncomputable def tan_75_degrees : ℝ := Real.tan (75 * Real.pi / 180)

theorem tan_75_degrees_eq : tan_75_degrees = 2 + Real.sqrt 3 := by
  sorry

end tan_75_degrees_eq_l156_156073


namespace min_x2_plus_y2_l156_156662

noncomputable def min_val (x y : ℝ) : ℝ :=
  if h : (x + 1)^2 + y^2 = 1/4 then x^2 + y^2 else 0

theorem min_x2_plus_y2 : 
  ∃ x y : ℝ, (x + 1)^2 + y^2 = 1/4 ∧ x^2 + y^2 = 1/4 :=
by
  sorry

end min_x2_plus_y2_l156_156662


namespace four_digit_composite_l156_156538

theorem four_digit_composite (abcd : ℕ) (h : 1000 ≤ abcd ∧ abcd < 10000) :
  ∃ (m n : ℕ), m ≥ 2 ∧ n ≥ 2 ∧ m * n = (abcd * 10001) :=
by
  sorry

end four_digit_composite_l156_156538


namespace number_added_to_x_l156_156507

theorem number_added_to_x (x : ℕ) (some_number : ℕ) (h1 : x = 3) (h2 : x + some_number = 4) : some_number = 1 := 
by
  -- Given hypotheses can be used here
  sorry

end number_added_to_x_l156_156507


namespace combinations_medical_team_l156_156266

noncomputable def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem combinations_medical_team : 
  let maleDoctors := 6
  let femaleDoctors := 5
  let numWaysMale := num_combinations maleDoctors 2
  let numWaysFemale := num_combinations femaleDoctors 1
  numWaysMale * numWaysFemale = 75 :=
by
  let maleDoctors := 6
  let femaleDoctors := 5
  let numWaysMale := num_combinations maleDoctors 2
  let numWaysFemale := num_combinations femaleDoctors 1
  show numWaysMale * numWaysFemale = 75 
  sorry

end combinations_medical_team_l156_156266


namespace sum_abc_geq_half_l156_156339

theorem sum_abc_geq_half (a b c : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) 
(h_abs_sum : |a - b| + |b - c| + |c - a| = 1) : 
a + b + c ≥ 0.5 := 
sorry

end sum_abc_geq_half_l156_156339


namespace weight_of_each_pack_l156_156047

-- Definitions based on conditions
def total_sugar : ℕ := 3020
def leftover_sugar : ℕ := 20
def number_of_packs : ℕ := 12

-- Definition of sugar used for packs
def sugar_used_for_packs : ℕ := total_sugar - leftover_sugar

-- Proof statement to be verified
theorem weight_of_each_pack : sugar_used_for_packs / number_of_packs = 250 := by
  sorry

end weight_of_each_pack_l156_156047


namespace advertisement_broadcasting_methods_l156_156770

/-- A TV station is broadcasting 5 different advertisements.
There are 3 different commercial advertisements.
There are 2 different Olympic promotional advertisements.
The last advertisement must be an Olympic promotional advertisement.
The two Olympic promotional advertisements cannot be broadcast consecutively.
Prove that the total number of different broadcasting methods is 18. -/
theorem advertisement_broadcasting_methods : 
  ∃ (arrangements : ℕ), arrangements = 18 := sorry

end advertisement_broadcasting_methods_l156_156770


namespace probability_ball_sports_l156_156911

theorem probability_ball_sports (clubs : Finset String)
  (ball_clubs : Finset String)
  (count_clubs : clubs.card = 5)
  (count_ball_clubs : ball_clubs.card = 3)
  (h1 : "basketball" ∈ clubs)
  (h2 : "soccer" ∈ clubs)
  (h3 : "volleyball" ∈ clubs)
  (h4 : "swimming" ∈ clubs)
  (h5 : "gymnastics" ∈ clubs)
  (h6 : "basketball" ∈ ball_clubs)
  (h7 : "soccer" ∈ ball_clubs)
  (h8 : "volleyball" ∈ ball_clubs) :
  (2 / ((5 : ℝ) * (4 : ℝ)) * ((3 : ℝ) * (2 : ℝ)) = (3 / 10)) :=
by
  sorry

end probability_ball_sports_l156_156911


namespace factorize_expression_l156_156210

-- Define the expression E
def E (x y z : ℝ) : ℝ := x^2 + x*y - x*z - y*z

-- State the theorem to prove \(E = (x + y)(x - z)\)
theorem factorize_expression (x y z : ℝ) : 
  E x y z = (x + y) * (x - z) := 
sorry

end factorize_expression_l156_156210


namespace part_I_part_II_l156_156655

noncomputable def f (a : ℝ) (x : ℝ) := a * x - Real.log x
noncomputable def F (a : ℝ) (x : ℝ) := Real.exp x + a * x
noncomputable def g (a : ℝ) (x : ℝ) := x * Real.exp (a * x - 1) - 2 * a * x + f a x

def monotonicity_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < b ∧ a < y ∧ y < b ∧ x < y → f x < f y

theorem part_I (a : ℝ) : 
  monotonicity_in_interval (f a) 0 (Real.log 3) = monotonicity_in_interval (F a) 0 (Real.log 3) ↔ a ≤ -3 :=
sorry

theorem part_II (a : ℝ) (ha : a ∈ Set.Iic (-1 / Real.exp 2)) : 
  (∃ x, x > 0 ∧ g a x = M) → M ≥ 0 :=
sorry

end part_I_part_II_l156_156655


namespace tan_simplify_l156_156849

theorem tan_simplify (α : ℝ) (h : Real.tan α = 1 / 2) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = - 3 / 4 :=
by
  sorry

end tan_simplify_l156_156849


namespace percentage_mr_william_land_l156_156958

theorem percentage_mr_william_land 
  (T W : ℝ) -- Total taxable land of the village and the total land of Mr. William
  (tax_collected_village : ℝ) -- Total tax collected from the village
  (tax_paid_william : ℝ) -- Tax paid by Mr. William
  (h1 : tax_collected_village = 3840) 
  (h2 : tax_paid_william = 480) 
  (h3 : (480 / 3840) = (25 / 100) * (W / T)) 
: (W / T) * 100 = 12.5 :=
by sorry

end percentage_mr_william_land_l156_156958


namespace quadratic_roots_l156_156487

theorem quadratic_roots (x : ℝ) : 
  (x^2 + 3 * x - 1 = 0) ↔ (x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) := 
by
  sorry

end quadratic_roots_l156_156487


namespace polar_to_cartesian_l156_156159

theorem polar_to_cartesian (θ : ℝ) (ρ : ℝ) (x y : ℝ) :
  (ρ = 2 * Real.sin θ + 4 * Real.cos θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x - 8)^2 + (y - 2)^2 = 68 :=
by
  intros hρ hx hy
  -- Proof steps would go here
  sorry

end polar_to_cartesian_l156_156159


namespace union_sets_eq_real_l156_156809

def A : Set ℝ := {x | x ≥ 0}
def B : Set ℝ := {x | x < 1}

theorem union_sets_eq_real : A ∪ B = Set.univ :=
by
  sorry

end union_sets_eq_real_l156_156809


namespace probability_p_eq_l156_156135

theorem probability_p_eq (p q : ℝ) (h_q : q = 1 - p)
  (h_eq : (Nat.choose 10 5) * p^5 * q^5 = (Nat.choose 10 6) * p^6 * q^4) : 
  p = 6 / 11 :=
by
  sorry

end probability_p_eq_l156_156135


namespace solve_monetary_prize_problem_l156_156428

def monetary_prize_problem : Prop :=
  ∃ (P x y : ℝ), 
    P = x + y + 30000 ∧
    x = (1/2) * P - (3/22) * (y + 30000) ∧
    y = (1/4) * P + (1/56) * x ∧
    P = 95000 ∧
    x = 40000 ∧
    y = 25000

theorem solve_monetary_prize_problem : monetary_prize_problem :=
  sorry

end solve_monetary_prize_problem_l156_156428


namespace ap_number_of_terms_l156_156103

theorem ap_number_of_terms (a d : ℕ) (n : ℕ) (ha1 : (n - 1) * d = 12) (ha2 : a + 2 * d = 6)
  (h_odd_sum : (n / 2) * (2 * a + (n - 2) * d) = 36) (h_even_sum : (n / 2) * (2 * a + n * d) = 42) :
    n = 12 :=
by
  sorry

end ap_number_of_terms_l156_156103


namespace percent_yz_of_x_l156_156097

theorem percent_yz_of_x (x y z : ℝ) 
  (h₁ : 0.6 * (x - y) = 0.3 * (x + y))
  (h₂ : 0.4 * (x + z) = 0.2 * (y + z))
  (h₃ : 0.5 * (x - z) = 0.25 * (x + y + z)) :
  y + z = 0.0 * x :=
sorry

end percent_yz_of_x_l156_156097


namespace cricket_bat_cost_l156_156220

variable (CP_A : ℝ) (CP_B : ℝ) (CP_C : ℝ)

-- Conditions
def CP_B_def : Prop := CP_B = 1.20 * CP_A
def CP_C_def : Prop := CP_C = 1.25 * CP_B
def CP_C_val : Prop := CP_C = 234

-- Theorem statement
theorem cricket_bat_cost (h1 : CP_B_def CP_A CP_B) (h2 : CP_C_def CP_B CP_C) (h3 : CP_C_val CP_C) : CP_A = 156 :=by
  sorry

end cricket_bat_cost_l156_156220


namespace total_cubes_l156_156743

noncomputable def original_cubes : ℕ := 2
noncomputable def additional_cubes : ℕ := 7

theorem total_cubes : original_cubes + additional_cubes = 9 := by
  sorry

end total_cubes_l156_156743


namespace complement_intersection_l156_156294

-- Definitions to set the universal set and other sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {2, 4, 5}

-- Complement of M with respect to U
def CU_M : Set ℕ := U \ M

-- Intersection of (CU_M) and N
def intersection_CU_M_N : Set ℕ := CU_M ∩ N

-- The proof problem statement
theorem complement_intersection :
  intersection_CU_M_N = {2, 5} :=
sorry

end complement_intersection_l156_156294


namespace find_initial_men_l156_156312

noncomputable def initial_men_planned (M : ℕ) : Prop :=
  let initial_days := 10
  let additional_days := 20
  let total_days := initial_days + additional_days
  let men_sent := 25
  let initial_work := M * initial_days
  let remaining_men := M - men_sent
  let remaining_work := remaining_men * total_days
  initial_work = remaining_work 

theorem find_initial_men :
  ∃ M : ℕ, initial_men_planned M ∧ M = 38 :=
by
  have h : initial_men_planned 38 :=
    by
      sorry
  exact ⟨38, h, rfl⟩

end find_initial_men_l156_156312


namespace no_solution_in_natural_numbers_l156_156056

theorem no_solution_in_natural_numbers :
  ¬ ∃ (x y : ℕ), 2^x + 21^x = y^3 :=
sorry

end no_solution_in_natural_numbers_l156_156056


namespace distance_from_Beijing_to_Lanzhou_l156_156175

-- Conditions
def distance_Beijing_Lanzhou_Lhasa : ℕ := 3985
def distance_Lanzhou_Lhasa : ℕ := 2054

-- Define the distance from Beijing to Lanzhou
def distance_Beijing_Lanzhou : ℕ := distance_Beijing_Lanzhou_Lhasa - distance_Lanzhou_Lhasa

-- Proof statement that given conditions imply the correct answer
theorem distance_from_Beijing_to_Lanzhou :
  distance_Beijing_Lanzhou = 1931 :=
by
  -- conditions and definitions are already given
  sorry

end distance_from_Beijing_to_Lanzhou_l156_156175


namespace solve_graph_equation_l156_156506

/- Problem:
Solve for the graph of the equation x^2(x+y+2)=y^2(x+y+2)
Given condition: equation x^2(x+y+2)=y^2(x+y+2)
Conclusion: Three lines that do not all pass through a common point
The final answer should be formally proven.
-/

theorem solve_graph_equation (x y : ℝ) :
  (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (∃ a b c d : ℝ,  (a = -x - 2 ∧ b = -x ∧ c = x ∧ (a ≠ b ∧ a ≠ c ∧ b ≠ c)) ∧
   (d = 0) ∧ ¬ ∀ p q r : ℝ, p = q ∧ q = r ∧ r = p) :=
by
  sorry

end solve_graph_equation_l156_156506


namespace principal_amount_l156_156315

theorem principal_amount (P : ℝ) (CI SI : ℝ) 
  (H1 : CI = P * 0.44) 
  (H2 : SI = P * 0.4) 
  (H3 : CI - SI = 216) : 
  P = 5400 :=
by {
  sorry
}

end principal_amount_l156_156315


namespace negation_of_quadratic_inequality_l156_156391

-- Definitions
def quadratic_inequality (a : ℝ) : Prop := ∃ x : ℝ, x * x + a * x + 1 < 0

-- Theorem statement
theorem negation_of_quadratic_inequality (a : ℝ) : ¬ (quadratic_inequality a) ↔ ∀ x : ℝ, x * x + a * x + 1 ≥ 0 :=
by sorry

end negation_of_quadratic_inequality_l156_156391


namespace base_conversion_to_zero_l156_156818

theorem base_conversion_to_zero (A B : ℕ) (hA : 0 ≤ A ∧ A < 12) (hB : 0 ≤ B ∧ B < 5) 
    (h1 : 12 * A + B = 5 * B + A) : 12 * A + B = 0 :=
by
  sorry

end base_conversion_to_zero_l156_156818


namespace find_chocolate_boxes_l156_156936

section
variable (x : Nat)
variable (candy_per_box : Nat := 8)
variable (caramel_boxes : Nat := 3)
variable (total_candy : Nat := 80)

theorem find_chocolate_boxes :
  8 * x + candy_per_box * caramel_boxes = total_candy -> x = 7 :=
by
  sorry
end

end find_chocolate_boxes_l156_156936


namespace crates_second_trip_l156_156442

theorem crates_second_trip
  (x y : Nat) 
  (h1 : x + y = 12)
  (h2 : x = 5) :
  y = 7 :=
by
  sorry

end crates_second_trip_l156_156442


namespace train_speed_kmph_l156_156549

noncomputable def speed_of_train
  (train_length : ℝ) (bridge_cross_time : ℝ) (total_length : ℝ) : ℝ :=
  (total_length / bridge_cross_time) * 3.6

theorem train_speed_kmph
  (train_length : ℝ := 130) 
  (bridge_cross_time : ℝ := 30) 
  (total_length : ℝ := 245) : 
  speed_of_train train_length bridge_cross_time total_length = 29.4 := by
  sorry

end train_speed_kmph_l156_156549


namespace meters_to_kilometers_kilograms_to_grams_centimeters_to_decimeters_hours_to_minutes_l156_156204

theorem meters_to_kilometers (h : 1 = 1000) : 6000 / 1000 = 6 := by
  sorry

theorem kilograms_to_grams (h : 1 = 1000) : (5 + 2) * 1000 = 7000 := by
  sorry

theorem centimeters_to_decimeters (h : 10 = 1) : (58 + 32) / 10 = 9 := by
  sorry

theorem hours_to_minutes (h : 60 = 1) : 3 * 60 + 30 = 210 := by
  sorry

end meters_to_kilometers_kilograms_to_grams_centimeters_to_decimeters_hours_to_minutes_l156_156204


namespace student_passing_percentage_l156_156998

variable (marks_obtained failed_by max_marks : ℕ)

def passing_marks (marks_obtained failed_by : ℕ) : ℕ :=
  marks_obtained + failed_by

def percentage_needed (passing_marks max_marks : ℕ) : ℚ :=
  (passing_marks : ℚ) / (max_marks : ℚ) * 100

theorem student_passing_percentage
  (h1 : marks_obtained = 125)
  (h2 : failed_by = 40)
  (h3 : max_marks = 500) :
  percentage_needed (passing_marks marks_obtained failed_by) max_marks = 33 := by
  sorry

end student_passing_percentage_l156_156998


namespace A_subset_B_l156_156585

def A (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 ≤ 5 / 4

def B (x y : ℝ) (a : ℝ) : Prop :=
  abs (x - 1) + 2 * abs (y - 2) ≤ a

theorem A_subset_B (a : ℝ) (h : a ≥ 5 / 2) : 
  ∀ x y : ℝ, A x y → B x y a := 
sorry

end A_subset_B_l156_156585


namespace average_loss_per_loot_box_l156_156241

theorem average_loss_per_loot_box
  (cost_per_loot_box : ℝ := 5)
  (value_standard_item : ℝ := 3.5)
  (probability_rare_item_A : ℝ := 0.05)
  (value_rare_item_A : ℝ := 10)
  (probability_rare_item_B : ℝ := 0.03)
  (value_rare_item_B : ℝ := 15)
  (probability_rare_item_C : ℝ := 0.02)
  (value_rare_item_C : ℝ := 20) 
  : (cost_per_loot_box 
      - (0.90 * value_standard_item 
      + probability_rare_item_A * value_rare_item_A 
      + probability_rare_item_B * value_rare_item_B 
      + probability_rare_item_C * value_rare_item_C)) = 0.50 := by 
  sorry

end average_loss_per_loot_box_l156_156241


namespace num_tuples_abc_l156_156473

theorem num_tuples_abc (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 2019 ≥ 10 * a) (h5 : 10 * a ≥ 100 * b) (h6 : 100 * b ≥ 1000 * c) : 
  ∃ n, n = 574 := sorry

end num_tuples_abc_l156_156473


namespace base_seven_sum_of_product_l156_156615

def base_seven_to_decimal (d1 d0 : ℕ) : ℕ :=
  7 * d1 + d0

def decimal_to_base_seven (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let d3 := n / (7 ^ 3)
  let r3 := n % (7 ^ 3)
  let d2 := r3 / (7 ^ 2)
  let r2 := r3 % (7 ^ 2)
  let d1 := r2 / 7
  let d0 := r2 % 7
  (d3, d2, d1, d0)

def sum_of_base_seven_digits (d3 d2 d1 d0 : ℕ) : ℕ :=
  d3 + d2 + d1 + d0

theorem base_seven_sum_of_product :
  let n1 := base_seven_to_decimal 3 5
  let n2 := base_seven_to_decimal 4 2
  let product := n1 * n2
  let (d3, d2, d1, d0) := decimal_to_base_seven product
  sum_of_base_seven_digits d3 d2 d1 d0 = 18 :=
  by
    sorry

end base_seven_sum_of_product_l156_156615


namespace kim_candy_bars_saved_l156_156261

theorem kim_candy_bars_saved
  (n : ℕ)
  (c : ℕ)
  (w : ℕ)
  (total_bought : ℕ := n * c)
  (total_eaten : ℕ := n / w)
  (candy_bars_saved : ℕ := total_bought - total_eaten) :
  candy_bars_saved = 28 :=
by
  sorry

end kim_candy_bars_saved_l156_156261


namespace radius_of_circle_l156_156966

theorem radius_of_circle (r x y : ℝ): 
  x = π * r^2 → 
  y = 2 * π * r → 
  x - y = 72 * π → 
  r = 12 := 
by 
  sorry

end radius_of_circle_l156_156966


namespace shaded_area_percentage_l156_156734

theorem shaded_area_percentage (side : ℕ) (total_shaded_area : ℕ) (expected_percentage : ℕ)
  (h1 : side = 5)
  (h2 : total_shaded_area = 15)
  (h3 : expected_percentage = 60) :
  ((total_shaded_area : ℚ) / (side * side) * 100) = expected_percentage :=
by
  sorry

end shaded_area_percentage_l156_156734


namespace second_discount_percentage_l156_156941

-- Defining the variables
variables (P S : ℝ) (d1 d2 : ℝ)

-- Given conditions
def original_price : P = 200 := by sorry
def sale_price_after_initial_discount : S = 171 := by sorry
def first_discount_rate : d1 = 0.10 := by sorry

-- Required to prove
theorem second_discount_percentage :
  ∃ d2, (d2 = 0.05) :=
sorry

end second_discount_percentage_l156_156941


namespace eugene_initial_pencils_l156_156721

theorem eugene_initial_pencils (P : ℕ) (h1 : P + 6 = 57) : P = 51 :=
by
  sorry

end eugene_initial_pencils_l156_156721


namespace find_crayons_in_pack_l156_156929

variables (crayons_in_locker : ℕ) (crayons_given_by_bobby : ℕ) (crayons_given_to_mary : ℕ) (crayons_final_count : ℕ) (crayons_in_pack : ℕ)

-- Definitions from the conditions
def initial_crayons := 36
def bobby_gave := initial_crayons / 2
def mary_crayons := 25
def final_crayons := initial_crayons + bobby_gave - mary_crayons

-- The theorem to prove
theorem find_crayons_in_pack : initial_crayons = 36 ∧ bobby_gave = 18 ∧ mary_crayons = 25 ∧ final_crayons = 29 → crayons_in_pack = 29 :=
by
  sorry

end find_crayons_in_pack_l156_156929


namespace arithmetic_sequences_ratio_l156_156182

theorem arithmetic_sequences_ratio (x y a1 a2 a3 b1 b2 b3 b4 : Real) (hxy : x ≠ y) 
  (h_arith1 : a1 = x + (y - x) / 4 ∧ a2 = x + 2 * (y - x) / 4 ∧ a3 = x + 3 * (y - x) / 4 ∧ y = x + 4 * (y - x) / 4)
  (h_arith2 : b1 = x - (y - x) / 2 ∧ b2 = x + (y - x) / 2 ∧ b3 = x + 2 * (y - x) / 2 ∧ y = x + 2 * (y - x) / 2 ∧ b4 = y + (y - x) / 2):
  (b4 - b3) / (a2 - a1) = 8 / 3 := 
sorry

end arithmetic_sequences_ratio_l156_156182


namespace train_length_is_correct_l156_156516

noncomputable def length_of_train (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_s

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 :=
by 
  -- Here, a proof would be provided, eventually using the definitions and conditions given
  sorry

end train_length_is_correct_l156_156516


namespace impossible_to_all_minus_l156_156543

def initial_grid : List (List Int) :=
  [[1, 1, -1, 1], 
   [-1, -1, 1, 1], 
   [1, 1, 1, 1], 
   [1, -1, 1, -1]]

-- Define the operation of flipping a row
def flip_row (grid : List (List Int)) (r : Nat) : List (List Int) :=
  grid.mapIdx (fun i row => if i == r then row.map (fun x => -x) else row)

-- Define the operation of flipping a column
def flip_col (grid : List (List Int)) (c : Nat) : List (List Int) :=
  grid.map (fun row => row.mapIdx (fun j x => if j == c then -x else x))

-- Predicate to check if all elements in the grid are -1
def all_minus (grid : List (List Int)) : Prop :=
  grid.all (fun row => row.all (fun x => x = -1))

-- The main theorem
theorem impossible_to_all_minus (init : List (List Int)) (hf1 : init = initial_grid) :
  ∀ grid, (grid = init ∨ ∃ r, grid = flip_row grid r ∨ ∃ c, grid = flip_col grid c) →
  ¬ all_minus grid := by
    sorry

end impossible_to_all_minus_l156_156543


namespace circle_radius_l156_156026

theorem circle_radius
  (area_sector : ℝ)
  (arc_length : ℝ)
  (h_area : area_sector = 8.75)
  (h_arc : arc_length = 3.5) :
  ∃ r : ℝ, r = 5 :=
by
  let r := 5
  use r
  sorry

end circle_radius_l156_156026


namespace expected_value_of_winnings_l156_156647

noncomputable def expected_value : ℝ :=
  (1 / 8) * (1 / 2) + (1 / 8) * (3 / 2) + (1 / 8) * (5 / 2) + (1 / 8) * (7 / 2) +
  (1 / 8) * 2 + (1 / 8) * 4 + (1 / 8) * 6 + (1 / 8) * 8

theorem expected_value_of_winnings : expected_value = 3.5 :=
by
  -- the proof steps will go here
  sorry

end expected_value_of_winnings_l156_156647


namespace Douglas_weight_correct_l156_156609

def Anne_weight : ℕ := 67
def weight_diff : ℕ := 15
def Douglas_weight : ℕ := 52

theorem Douglas_weight_correct : Douglas_weight = Anne_weight - weight_diff := by
  sorry

end Douglas_weight_correct_l156_156609


namespace simple_sampling_methods_l156_156589

theorem simple_sampling_methods :
  methods_of_implementing_simple_sampling = ["lottery method", "random number table method"] :=
sorry

end simple_sampling_methods_l156_156589


namespace part1_part2_l156_156848

noncomputable def f (x : ℝ) : ℝ := x^2 - 1
noncomputable def g (a x : ℝ) := a * |x - 1|

theorem part1 (a : ℝ) :
  (∀ x : ℝ, |f x| = g a x → x = 1) → a < 0 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) → a ≤ -2 :=
sorry

end part1_part2_l156_156848


namespace length_PT_30_l156_156502

noncomputable def length_PT (PQ QR : ℝ) (angle_QRT : ℝ) (T_on_RS : Prop) : ℝ := 
  if h : PQ = 30 ∧ QR = 15 ∧ angle_QRT = 75 then 30 else 0

theorem length_PT_30 (PQ QR : ℝ) (angle_QRT : ℝ) (T_on_RS : Prop) :
  PQ = 30 → QR = 15 → angle_QRT = 75 → length_PT PQ QR angle_QRT T_on_RS = 30 :=
sorry

end length_PT_30_l156_156502


namespace calculate_expression_l156_156224

theorem calculate_expression : 
  (2^1002 + 5^1003)^2 - (2^1002 - 5^1003)^2 = 20 * 10^1002 := 
by
  sorry

end calculate_expression_l156_156224


namespace water_tank_equilibrium_l156_156235

theorem water_tank_equilibrium :
  (1 / 15 : ℝ) + (1 / 10 : ℝ) - (1 / 6 : ℝ) = 0 :=
by
  sorry

end water_tank_equilibrium_l156_156235


namespace verify_BG_BF_verify_FG_EG_find_x_l156_156712

noncomputable def verify_angles (CBG GBE EBF BCF FCE : ℝ) :=
  CBG = 20 ∧ GBE = 40 ∧ EBF = 20 ∧ BCF = 50 ∧ FCE = 30

theorem verify_BG_BF (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → BG = BF :=
by
  sorry

theorem verify_FG_EG (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → FG = EG :=
by
  sorry

theorem find_x (CBG GBE EBF BCF FCE : ℝ) :
  verify_angles CBG GBE EBF BCF FCE → x = 30 :=
by
  sorry

end verify_BG_BF_verify_FG_EG_find_x_l156_156712


namespace bill_profit_difference_l156_156063

theorem bill_profit_difference (P : ℝ) 
  (h1 : 1.10 * P = 549.9999999999995)
  (h2 : ∀ NP NSP, NP = 0.90 * P ∧ NSP = 1.30 * NP →
  NSP - 549.9999999999995 = 35) :
  true :=
by {
  sorry
}

end bill_profit_difference_l156_156063


namespace Y_4_3_l156_156379

def Y (x y : ℝ) : ℝ := x^2 - 3 * x * y + y^2

theorem Y_4_3 : Y 4 3 = -11 :=
by
  -- This line is added to skip the proof and focus on the statement.
  sorry

end Y_4_3_l156_156379


namespace probability_between_C_and_E_l156_156833

theorem probability_between_C_and_E
  (AB AD BC BE : ℝ)
  (h₁ : AB = 4 * AD)
  (h₂ : AB = 8 * BC)
  (h₃ : AB = 2 * BE) : 
  (AB / 2 - AB / 8) / AB = 3 / 8 :=
by 
  sorry

end probability_between_C_and_E_l156_156833


namespace curve_is_circle_l156_156584

noncomputable def curve_eqn_polar (r θ : ℝ) : Prop :=
  r = 1 / (Real.sin θ + Real.cos θ)

theorem curve_is_circle : ∀ r θ, curve_eqn_polar r θ →
  ∃ x y : ℝ, r = Real.sqrt (x^2 + y^2) ∧ 
  x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ 
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2 :=
by
  sorry

end curve_is_circle_l156_156584


namespace entry_cost_proof_l156_156894

variable (hitting_rate : ℕ → ℝ)
variable (entry_cost : ℝ)
variable (total_hits : ℕ)
variable (money_lost : ℝ)

-- Conditions
axiom hitting_rate_condition : hitting_rate 200 = 0.025
axiom total_hits_condition : total_hits = 300
axiom money_lost_condition : money_lost = 7.5

-- Question: Prove that the cost to enter the contest equals $10.00
theorem entry_cost_proof : entry_cost = 10 := by
  sorry

end entry_cost_proof_l156_156894


namespace ratio_diff_squares_eq_16_l156_156907

theorem ratio_diff_squares_eq_16 (x y : ℕ) (h1 : x + y = 16) (h2 : x ≠ y) :
  (x^2 - y^2) / (x - y) = 16 :=
by
  sorry

end ratio_diff_squares_eq_16_l156_156907


namespace find_k_l156_156388

theorem find_k (k : ℝ) : (∃ x y : ℝ, y = -2 * x + 4 ∧ y = k * x ∧ y = x + 2) → k = 4 :=
by
  sorry

end find_k_l156_156388


namespace trapezoid_LM_sqrt2_l156_156295

theorem trapezoid_LM_sqrt2 (K L M N P Q : Type*)
  (KM : ℝ)
  (KN MQ LM MP : ℝ)
  (h_KM : KM = 1)
  (h_KN_MQ : KN = MQ)
  (h_LM_MP : LM = MP) 
  (h_KP_1 : KN = 1) 
  (h_MQ_1 : MQ = 1) :
  LM = Real.sqrt 2 :=
by
  sorry

end trapezoid_LM_sqrt2_l156_156295


namespace interest_rate_compound_interest_l156_156292

theorem interest_rate_compound_interest :
  ∀ (P A : ℝ) (t n : ℕ), 
  P = 156.25 → A = 169 → t = 2 → n = 1 → 
  (∃ r : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ r * 100 = 4) :=
by
  intros P A t n hP hA ht hn
  use 0.04
  rw [hP, hA, ht, hn]
  sorry

end interest_rate_compound_interest_l156_156292


namespace minimize_x_l156_156492

theorem minimize_x (x y : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : x + y^2 = x * y) : x ≥ 3 :=
sorry

end minimize_x_l156_156492


namespace sandy_saved_percentage_last_year_l156_156166

noncomputable def sandys_saved_percentage (S : ℝ) (P : ℝ) : ℝ :=
  (P / 100) * S

noncomputable def salary_with_10_percent_more (S : ℝ) : ℝ :=
  1.1 * S

noncomputable def amount_saved_this_year (S : ℝ) : ℝ :=
  0.15 * (salary_with_10_percent_more S)

noncomputable def amount_saved_this_year_compare_last_year (S : ℝ) (P : ℝ) : Prop :=
  amount_saved_this_year S = 1.65 * sandys_saved_percentage S P

theorem sandy_saved_percentage_last_year (S : ℝ) (P : ℝ) :
  amount_saved_this_year_compare_last_year S P → P = 10 :=
by
  sorry

end sandy_saved_percentage_last_year_l156_156166


namespace range_of_a_l156_156066

def A : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a < x ∧ x < a + 1}

theorem range_of_a (a : ℝ)
  (h₀ : a < 1)
  (h₁ : B a ⊆ A) :
  a ∈ {x : ℝ | x ≤ -2 ∨ (1 / 2 ≤ x ∧ x < 1)} :=
by
  sorry

end range_of_a_l156_156066


namespace sum_first_10_terms_l156_156952

-- Define the general arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Define the conditions of the problem
def given_conditions (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 2 ∧ (a 2) ^ 2 = 2 * a 4 ∧ arithmetic_seq a d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 1 + a n)) / 2

-- Statement of the problem
theorem sum_first_10_terms (a : ℕ → ℤ) (d : ℤ) (S₁₀ : ℤ) :
  given_conditions a d →
  (S₁₀ = 20 ∨ S₁₀ = 110) :=
sorry

end sum_first_10_terms_l156_156952


namespace solution_inequality_l156_156454

noncomputable def solution_set (a : ℝ) (x : ℝ) := x < (1 - a) / (1 + a)

theorem solution_inequality 
  (a : ℝ) 
  (h1 : a^3 < a) 
  (h2 : a < a^2) :
  ∀ (x : ℝ), x + a > 1 - a * x ↔ solution_set a x :=
sorry

end solution_inequality_l156_156454


namespace percentage_increase_in_consumption_l156_156195

-- Define the conditions
variables {T C : ℝ}  -- T: original tax, C: original consumption
variables (P : ℝ)    -- P: percentage increase in consumption

-- Non-zero conditions
variables (hT : T ≠ 0) (hC : C ≠ 0)

-- Define the Lean theorem
theorem percentage_increase_in_consumption 
  (h : 0.8 * (1 + P / 100) = 0.96) : 
  P = 20 :=
by
  sorry

end percentage_increase_in_consumption_l156_156195


namespace face_sum_l156_156433

theorem face_sum (a b c d e f : ℕ) (h : (a + d) * (b + e) * (c + f) = 1008) : 
  a + b + c + d + e + f = 173 :=
by
  sorry

end face_sum_l156_156433


namespace eval_expression_l156_156852

theorem eval_expression : 
  3000^3 - 2998 * 3000^2 - 2998^2 * 3000 + 2998^3 = 23992 := 
by 
  sorry

end eval_expression_l156_156852


namespace sum_of_other_endpoint_l156_156030

theorem sum_of_other_endpoint (x y : ℝ) (h₁ : (9 + x) / 2 = 5) (h₂ : (-6 + y) / 2 = -8) :
  x + y = -9 :=
sorry

end sum_of_other_endpoint_l156_156030


namespace difference_between_max_and_min_l156_156819

noncomputable def maxThree (a b c : ℝ) : ℝ :=
  max a (max b c)

noncomputable def minThree (a b c : ℝ) : ℝ :=
  min a (min b c)

theorem difference_between_max_and_min :
  maxThree 0.12 0.23 0.22 - minThree 0.12 0.23 0.22 = 0.11 :=
by
  sorry

end difference_between_max_and_min_l156_156819


namespace divisors_remainders_l156_156991

theorem divisors_remainders (n : ℕ) (h : ∀ k : ℕ, 1001 ≤ k ∧ k ≤ 2012 → ∃ d : ℕ, d ∣ n ∧ d % 2013 = k) :
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 2012 → ∃ d : ℕ, d ∣ n^2 ∧ d % 2013 = m :=
by sorry

end divisors_remainders_l156_156991


namespace sum_of_4n_pos_integers_l156_156595

theorem sum_of_4n_pos_integers (n : ℕ) (Sn : ℕ → ℕ)
  (hSn : ∀ k, Sn k = k * (k + 1) / 2)
  (h_condition : Sn (3 * n) - Sn n = 150) :
  Sn (4 * n) = 300 :=
by {
  sorry
}

end sum_of_4n_pos_integers_l156_156595


namespace inequality_1_inequality_2_l156_156878

noncomputable def f (x : ℝ) : ℝ := |x - 2| - 3
noncomputable def g (x : ℝ) : ℝ := |x + 3|

theorem inequality_1 (x : ℝ) : f x < g x ↔ x > -2 := 
by sorry

theorem inequality_2 (a : ℝ) : (∀ x : ℝ, f x < g x + a) ↔ a > 2 := 
by sorry

end inequality_1_inequality_2_l156_156878


namespace cos_pi_div_4_minus_alpha_l156_156460

theorem cos_pi_div_4_minus_alpha (α : ℝ) (h : Real.sin (α + π/4) = 5/13) : 
  Real.cos (π/4 - α) = 5/13 :=
by
  sorry

end cos_pi_div_4_minus_alpha_l156_156460


namespace domain_of_f_x_squared_l156_156969

theorem domain_of_f_x_squared {f : ℝ → ℝ} (h : ∀ x, -2 ≤ x ∧ x ≤ 3 → ∃ y, f (x + 1) = y) : 
  ∀ x, -2 ≤ x ∧ x ≤ 2 → ∃ y, f (x ^ 2) = y := 
by 
  sorry

end domain_of_f_x_squared_l156_156969


namespace y_value_when_x_is_3_l156_156532

theorem y_value_when_x_is_3 :
  (x + y = 30) → (x - y = 12) → (x * y = 189) → (x = 3) → y = 63 :=
by 
  intros h1 h2 h3 h4
  sorry

end y_value_when_x_is_3_l156_156532


namespace add_to_37_eq_52_l156_156358

theorem add_to_37_eq_52 (x : ℕ) (h : 37 + x = 52) : x = 15 := by
  sorry

end add_to_37_eq_52_l156_156358


namespace calc1_calc2_calc3_calc4_l156_156865

-- Proof problem definitions
theorem calc1 : 15 + (-22) = -7 := sorry

theorem calc2 : (-13) + (-8) = -21 := sorry

theorem calc3 : (-0.9) + 1.5 = 0.6 := sorry

theorem calc4 : (1 / 2) + (-2 / 3) = -1 / 6 := sorry

end calc1_calc2_calc3_calc4_l156_156865


namespace china_nhsm_league_2021_zhejiang_p15_l156_156179

variable (x y z : ℝ)

theorem china_nhsm_league_2021_zhejiang_p15 (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x ^ 4 + y ^ 2 * z ^ 2) / (x ^ (5 / 2) * (y + z)) + 
  (y ^ 4 + z ^ 2 * x ^ 2) / (y ^ (5 / 2) * (z + x)) + 
  (z ^ 4 + y ^ 2 * x ^ 2) / (z ^ (5 / 2) * (y + x)) ≥ 1 := 
sorry

end china_nhsm_league_2021_zhejiang_p15_l156_156179


namespace distribution_ways_l156_156542

theorem distribution_ways :
  ∃ (n : ℕ) (erasers pencils notebooks pens : ℕ),
  pencils = 4 ∧ notebooks = 2 ∧ pens = 3 ∧ 
  n = 6 := sorry

end distribution_ways_l156_156542


namespace felicia_flour_amount_l156_156840

-- Define the conditions as constants
def white_sugar := 1 -- cups
def brown_sugar := 1 / 4 -- cups
def oil := 1 / 2 -- cups
def scoop := 1 / 4 -- cups
def total_scoops := 15 -- number of scoops

-- Define the proof statement
theorem felicia_flour_amount : 
  (total_scoops * scoop - (white_sugar + brown_sugar / scoop + oil / scoop)) * scoop = 2 :=
by
  sorry

end felicia_flour_amount_l156_156840


namespace binary_multiplication_correct_l156_156748

theorem binary_multiplication_correct :
  (0b1101 : ℕ) * (0b1011 : ℕ) = (0b10011011 : ℕ) :=
by
  sorry

end binary_multiplication_correct_l156_156748


namespace number_of_pieces_of_paper_used_l156_156759

theorem number_of_pieces_of_paper_used
  (P : ℕ)
  (h1 : 1 / 5 > 0)
  (h2 : 2 / 5 > 0)
  (h3 : 1 < (P : ℝ) * (1 / 5) + 2 / 5 ∧ (P : ℝ) * (1 / 5) + 2 / 5 ≤ 2) : 
  P = 8 :=
sorry

end number_of_pieces_of_paper_used_l156_156759


namespace sequence_probability_correct_l156_156439

noncomputable def m : ℕ := 377
noncomputable def n : ℕ := 4096

theorem sequence_probability_correct :
  let m := 377
  let n := 4096
  (m.gcd n = 1) ∧ (m + n = 4473) := 
by
  -- Proof requires the given equivalent statement in Lean, so include here
  sorry

end sequence_probability_correct_l156_156439


namespace guo_can_pay_exact_amount_l156_156321

-- Define the denominations and total amount Guo has
def note_denominations := [1, 10, 20, 50]
def total_amount := 20000
def cost_computer := 10000

-- The main theorem stating that Guo can pay exactly 10000 yuan
theorem guo_can_pay_exact_amount : ∃ bills : List ℕ, ∀ (b : ℕ), b ∈ bills → b ∈ note_denominations ∧
  bills.sum = cost_computer :=
sorry

end guo_can_pay_exact_amount_l156_156321


namespace binary_operation_l156_156132

theorem binary_operation : 
  let a := 0b11011
  let b := 0b1101
  let c := 0b1010
  let result := 0b110011101  
  ((a * b) - c) = result := by
  sorry

end binary_operation_l156_156132


namespace n_div_p_eq_27_l156_156124

theorem n_div_p_eq_27 (m n p : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : p ≠ 0)
    (h4 : ∃ r1 r2 : ℝ, r1 * r2 = m ∧ r1 + r2 = -p ∧ (3 * r1) * (3 * r2) = n ∧ 3 * (r1 + r2) = -m)
    : n / p = 27 := sorry

end n_div_p_eq_27_l156_156124


namespace basketball_probability_l156_156177

-- Define the probabilities of A and B making a shot
def prob_A : ℝ := 0.4
def prob_B : ℝ := 0.6

-- Define the probability that both miss their shots in one round
def prob_miss_one_round : ℝ := (1 - prob_A) * (1 - prob_B)

-- Define the probability that A takes k shots to make a basket
noncomputable def P_xi (k : ℕ) : ℝ := (prob_miss_one_round)^(k-1) * prob_A

-- State the theorem
theorem basketball_probability (k : ℕ) : 
  P_xi k = 0.24^(k-1) * 0.4 :=
by
  unfold P_xi
  unfold prob_miss_one_round
  sorry

end basketball_probability_l156_156177


namespace evaluate_tan_fraction_l156_156724

theorem evaluate_tan_fraction:
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 / 3 :=
by
  sorry

end evaluate_tan_fraction_l156_156724


namespace total_students_sampled_l156_156751

theorem total_students_sampled (freq_ratio : ℕ → ℕ → ℕ) (second_group_freq : ℕ) 
  (ratio_condition : freq_ratio 2 1 = 2 ∧ freq_ratio 2 3 = 3) : 
  (6 + second_group_freq + 18) = 48 := 
by 
  sorry

end total_students_sampled_l156_156751


namespace compare_fractions_l156_156766

variables {a b : ℝ}

theorem compare_fractions (h : a + b > 0) : 
  (a / (b^2)) + (b / (a^2)) ≥ (1 / a) + (1 / b) :=
sorry

end compare_fractions_l156_156766


namespace dan_initial_money_l156_156184

theorem dan_initial_money 
  (cost_chocolate : ℕ) 
  (cost_candy_bar : ℕ) 
  (h1 : cost_chocolate = 3) 
  (h2 : cost_candy_bar = 7)
  (h3 : cost_candy_bar - cost_chocolate = 4) : 
  cost_candy_bar + cost_chocolate = 10 := 
by
  sorry

end dan_initial_money_l156_156184


namespace score_seventy_five_can_be_achieved_three_ways_l156_156937

-- Defining the problem constraints and goal
def quiz_problem (c u i : ℕ) (S : ℝ) : Prop :=
  c + u + i = 20 ∧ S = 5 * (c : ℝ) + 1.5 * (u : ℝ)

theorem score_seventy_five_can_be_achieved_three_ways :
  ∃ (c1 u1 c2 u2 c3 u3 : ℕ), 0 ≤ (5 * (c1 : ℝ) + 1.5 * (u1 : ℝ)) ∧ (5 * (c1 : ℝ) + 1.5 * (u1 : ℝ)) ≤ 100 ∧
  (5 * (c2 : ℝ) + 1.5 * (u2 : ℝ)) = 75 ∧ (5 * (c3 : ℝ) + 1.5 * (u3 : ℝ)) = 75 ∧
  (c1 ≠ c2 ∧ u1 ≠ u2) ∧ (c2 ≠ c3 ∧ u2 ≠ u3) ∧ (c3 ≠ c1 ∧ u3 ≠ u1) ∧ 
  quiz_problem c1 u1 (20 - c1 - u1) 75 ∧
  quiz_problem c2 u2 (20 - c2 - u2) 75 ∧
  quiz_problem c3 u3 (20 - c3 - u3) 75 :=
sorry

end score_seventy_five_can_be_achieved_three_ways_l156_156937


namespace calculate_total_shaded_area_l156_156481

theorem calculate_total_shaded_area
(smaller_square_side larger_square_side smaller_circle_radius larger_circle_radius : ℝ)
(h1 : smaller_square_side = 6)
(h2 : larger_square_side = 12)
(h3 : smaller_circle_radius = 3)
(h4 : larger_circle_radius = 6) :
  (smaller_square_side^2 - π * smaller_circle_radius^2) + 
  (larger_square_side^2 - π * larger_circle_radius^2) = 180 - 45 * π :=
by
  sorry

end calculate_total_shaded_area_l156_156481


namespace sheila_attends_picnic_l156_156355

theorem sheila_attends_picnic :
  let probRain := 0.30
  let probSunny := 0.50
  let probCloudy := 0.20
  let probAttendIfRain := 0.15
  let probAttendIfSunny := 0.85
  let probAttendIfCloudy := 0.40
  (probRain * probAttendIfRain + probSunny * probAttendIfSunny + probCloudy * probAttendIfCloudy) = 0.55 :=
by sorry

end sheila_attends_picnic_l156_156355


namespace relationship_among_a_b_c_l156_156010

noncomputable def a : ℝ := Real.log (7 / 2) / Real.log 3
noncomputable def b : ℝ := (1 / 4)^(1 / 3)
noncomputable def c : ℝ := -Real.log 5 / Real.log 3

theorem relationship_among_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_among_a_b_c_l156_156010


namespace ice_cream_cones_sold_l156_156233

theorem ice_cream_cones_sold (T W : ℕ) (h1 : W = 2 * T) (h2 : T + W = 36000) : T = 12000 :=
by
  sorry

end ice_cream_cones_sold_l156_156233


namespace find_number_l156_156213

theorem find_number (x : ℝ) (h : 0.15 * 40 = 0.25 * x + 2) : x = 16 :=
by
  sorry

end find_number_l156_156213


namespace new_team_average_weight_is_113_l156_156694

-- Defining the given constants and conditions
def original_players := 7
def original_average_weight := 121 
def weight_new_player1 := 110 
def weight_new_player2 := 60 

-- Definition to calculate the new average weight
def new_average_weight : ℕ :=
  let original_total_weight := original_players * original_average_weight
  let new_total_weight := original_total_weight + weight_new_player1 + weight_new_player2
  let new_total_players := original_players + 2
  new_total_weight / new_total_players

-- Statement to prove
theorem new_team_average_weight_is_113 : new_average_weight = 113 :=
sorry

end new_team_average_weight_is_113_l156_156694


namespace cost_price_l156_156800

-- Given conditions
variable (x : ℝ)
def profit (x : ℝ) : ℝ := 54 - x
def loss (x : ℝ) : ℝ := x - 40

-- Claim
theorem cost_price (h : profit x = loss x) : x = 47 :=
by {
  -- This is where the proof would go
  sorry
}

end cost_price_l156_156800


namespace problem1_problem2_l156_156938

open Set

variable {U : Set ℝ} (A B : Set ℝ)

def UA : U = univ := by sorry
def A_def : A = { x : ℝ | 0 < x ∧ x ≤ 2 } := by sorry
def B_def : B = { x : ℝ | x < -3 ∨ x > 1 } := by sorry

theorem problem1 : A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 2 } := 
by sorry

theorem problem2 : (U \ A) ∩ (U \ B) = { x : ℝ | -3 ≤ x ∧ x ≤ 0 } := 
by sorry

end problem1_problem2_l156_156938


namespace num_quarters_left_l156_156283

-- Define initial amounts and costs
def initial_amount : ℝ := 40
def pizza_cost : ℝ := 2.75
def soda_cost : ℝ := 1.50
def jeans_cost : ℝ := 11.50
def quarter_value : ℝ := 0.25

-- Define the total amount spent
def total_spent : ℝ := pizza_cost + soda_cost + jeans_cost

-- Define the remaining amount
def remaining_amount : ℝ := initial_amount - total_spent

-- Prove the number of quarters left
theorem num_quarters_left : remaining_amount / quarter_value = 97 :=
by
  sorry

end num_quarters_left_l156_156283


namespace solve_for_x_l156_156783

theorem solve_for_x (x : ℝ) : (2 / 7) * (1 / 8) * x = 14 ↔ x = 392 :=
by {
  sorry
}

end solve_for_x_l156_156783


namespace inverse_matrix_l156_156234

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, -2], ![-3, 1]]

-- Define the supposed inverse B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2], ![3, 7]]

-- Define the condition that A * B should yield the identity matrix
theorem inverse_matrix :
  A * B = 1 :=
sorry

end inverse_matrix_l156_156234


namespace farthest_vertex_label_l156_156610

-- The vertices and their labeling
def cube_faces : List (List Nat) := [
  [1, 2, 5, 8],
  [3, 4, 6, 7],
  [2, 4, 5, 7],
  [1, 3, 6, 8],
  [2, 3, 7, 8],
  [1, 4, 5, 6]
]

-- Define the cube vertices labels
def vertices : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

-- Statement of the problem in Lean 4
theorem farthest_vertex_label (h : true) : 
  ∃ v : Nat, v ∈ vertices ∧ ∀ face ∈ cube_faces, v ∉ face → v = 6 := 
sorry

end farthest_vertex_label_l156_156610


namespace g_inv_g_inv_14_l156_156322

def g (x : ℝ) : ℝ := 5 * x - 3

noncomputable def g_inv (y : ℝ) : ℝ := (y + 3) / 5

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 32 / 25 :=
by
  sorry

end g_inv_g_inv_14_l156_156322


namespace exists_reals_condition_l156_156021

-- Define the conditions in Lean
theorem exists_reals_condition (n : ℕ) (h₁ : n ≥ 3) : 
  (∃ a : Fin (n + 2) → ℝ, a 0 = a n ∧ a 1 = a (n + 1) ∧ 
  ∀ i : Fin n, a i * a (i + 1) + 1 = a (i + 2)) ↔ 3 ∣ n := 
sorry

end exists_reals_condition_l156_156021


namespace parabola_latus_rectum_l156_156058

theorem parabola_latus_rectum (x p y : ℝ) (hp : p > 0) (h_eq : x^2 = 2 * p * y) (hl : y = -3) :
  p = 6 :=
by
  sorry

end parabola_latus_rectum_l156_156058


namespace Peter_drew_more_l156_156296

theorem Peter_drew_more :
  ∃ (P : ℕ), 5 + P + (P + 20) = 41 ∧ (P - 5 = 3) :=
sorry

end Peter_drew_more_l156_156296


namespace transmitter_finding_probability_l156_156567

/-- 
  A license plate in the country Kerrania consists of 4 digits followed by two letters.
  The letters A, B, and C are used only by government vehicles while the letters D through Z are used by non-government vehicles.
  Kerrania's intelligence agency has recently captured a message from the country Gonzalia indicating that an electronic transmitter 
  has been installed in a Kerrania government vehicle with a license plate starting with 79. 
  In addition, the message reveals that the last three digits of the license plate form a palindromic sequence (meaning that they are 
  the same forward and backward), and the second digit is either a 3 or a 5. 
  If it takes the police 10 minutes to inspect each vehicle, what is the probability that the police will find the transmitter 
  within 3 hours, considering the additional restrictions on the possible license plate combinations?
-/
theorem transmitter_finding_probability :
  0.1 = 18 / 180 :=
by
  sorry

end transmitter_finding_probability_l156_156567


namespace determine_b_l156_156541

-- Define the problem conditions
variable (n b : ℝ)
variable (h_pos_b : b > 0)
variable (h_eq : ∀ x : ℝ, (x + n) ^ 2 + 16 = x^2 + b * x + 88)

-- State that we want to prove that b equals 12 * sqrt(2)
theorem determine_b : b = 12 * Real.sqrt 2 :=
by
  sorry

end determine_b_l156_156541


namespace number_of_cases_in_top_level_l156_156240

-- Definitions for the total number of soda cases
def pyramid_cases (n : ℕ) : ℕ :=
  n^2 + (n + 1)^2 + (n + 2)^2 + (n + 3)^2

-- Theorem statement: proving the number of cases in the top level
theorem number_of_cases_in_top_level (n : ℕ) (h : pyramid_cases n = 30) : n = 1 :=
by {
  sorry
}

end number_of_cases_in_top_level_l156_156240


namespace determine_a_l156_156153
open Set

-- Given Condition Definitions
def U : Set ℕ := {1, 3, 5, 7}
def M (a : ℤ) : Set ℕ := {1, Int.natAbs (a - 5)} -- using ℤ for a and natAbs to get |a - 5|

-- Problem statement
theorem determine_a (a : ℤ) (hM_subset_U : M a ⊆ U) (h_complement : U \ M a = {5, 7}) : a = 2 ∨ a = 8 :=
by sorry

end determine_a_l156_156153


namespace symmetry_proof_l156_156057

-- Define the coordinates of point A
def A : ℝ × ℝ := (-1, 8)

-- Define the reflection property across the y-axis
def is_reflection_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

-- Define the point B which we need to prove
def B : ℝ × ℝ := (1, 8)

-- The proof statement
theorem symmetry_proof :
  is_reflection_y_axis A B :=
by
  sorry

end symmetry_proof_l156_156057


namespace simplify_sqrt_eight_l156_156459

theorem simplify_sqrt_eight : Real.sqrt 8 = 2 * Real.sqrt 2 :=
by
  -- Given that 8 can be factored into 4 * 2 and the property sqrt(a * b) = sqrt(a) * sqrt(b)
  sorry

end simplify_sqrt_eight_l156_156459


namespace road_renovation_l156_156272

theorem road_renovation (x : ℕ) (h : 200 / (x + 20) = 150 / x) : 
  x = 60 ∧ (x + 20) = 80 :=
by {
  sorry
}

end road_renovation_l156_156272


namespace LCM_4_6_15_is_60_l156_156096

def prime_factors (n : ℕ) : List ℕ :=
  [] -- placeholder, definition of prime_factor is not necessary for the problem statement, so we leave it abstract

def LCM (a b : ℕ) : ℕ := 
  sorry -- placeholder, definition of LCM not directly necessary for the statement

theorem LCM_4_6_15_is_60 : LCM (LCM 4 6) 15 = 60 := 
  sorry

end LCM_4_6_15_is_60_l156_156096


namespace time_needed_to_gather_remaining_flowers_l156_156676

-- conditions
def classmates : ℕ := 30
def time_per_flower : ℕ := 10
def gathering_time : ℕ := 2 * 60
def lost_flowers : ℕ := 3

-- question and proof goal
theorem time_needed_to_gather_remaining_flowers : 
  let flowers_needed := classmates - ((gathering_time / time_per_flower) - lost_flowers)
  flowers_needed * time_per_flower = 210 :=
by
  sorry

end time_needed_to_gather_remaining_flowers_l156_156676


namespace neg_q_necessary_not_sufficient_for_neg_p_l156_156163

-- Proposition p: |x + 2| > 2
def p (x : ℝ) : Prop := abs (x + 2) > 2

-- Proposition q: 1 / (3 - x) > 1
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Negation of p and q
def neg_p (x : ℝ) : Prop := -4 ≤ x ∧ x ≤ 0
def neg_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

-- Theorem: negation of q is a necessary but not sufficient condition for negation of p
theorem neg_q_necessary_not_sufficient_for_neg_p :
  (∀ x : ℝ, neg_p x → neg_q x) ∧ (∃ x : ℝ, neg_q x ∧ ¬neg_p x) :=
by
  sorry

end neg_q_necessary_not_sufficient_for_neg_p_l156_156163


namespace car_R_average_speed_l156_156091

theorem car_R_average_speed :
  ∃ (v : ℕ), (600 / v) - 2 = 600 / (v + 10) ∧ v = 50 :=
by sorry

end car_R_average_speed_l156_156091


namespace area_of_region_eq_24π_l156_156985

theorem area_of_region_eq_24π :
  (∃ R, R > 0 ∧ ∀ (x y : ℝ), x ^ 2 + y ^ 2 + 8 * x + 18 * y + 73 = R ^ 2) →
  ∃ π : ℝ, π > 0 ∧ area = 24 * π :=
by
  sorry

end area_of_region_eq_24π_l156_156985


namespace sin_of_5pi_over_6_l156_156933

theorem sin_of_5pi_over_6 : Real.sin (5 * Real.pi / 6) = 1 / 2 := 
by
  sorry

end sin_of_5pi_over_6_l156_156933


namespace watch_cost_price_l156_156636

theorem watch_cost_price (CP : ℝ) (H1 : 0.90 * CP = CP - 0.10 * CP)
(H2 : 1.04 * CP = CP + 0.04 * CP)
(H3 : 1.04 * CP - 0.90 * CP = 168) : CP = 1200 := by
sorry

end watch_cost_price_l156_156636


namespace compute_expression_l156_156965

theorem compute_expression : 7^3 - 5 * (6^2) + 2^4 = 179 :=
by
  sorry

end compute_expression_l156_156965


namespace determine_k_circle_l156_156515

theorem determine_k_circle (k : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 14*y - k = 0) ∧ ((∀ x y : ℝ, (x + 4)^2 + (y + 7)^2 = 25) ↔ k = -40) :=
by
  sorry

end determine_k_circle_l156_156515


namespace prob_board_251_l156_156509

noncomputable def probability_boarding_bus_251 (r1 r2 : ℕ) : ℚ :=
  let interval_152 := r1
  let interval_251 := r2
  let total_area := interval_152 * interval_251
  let triangle_area := 1 / 2 * interval_152 * interval_152
  triangle_area / total_area

theorem prob_board_251 : probability_boarding_bus_251 5 7 = 5 / 14 := by
  sorry

end prob_board_251_l156_156509


namespace probability_heart_or_king_l156_156910

theorem probability_heart_or_king (cards hearts kings : ℕ) (prob_non_heart_king : ℚ) 
    (prob_two_non_heart_king : ℚ) : 
    cards = 52 → hearts = 13 → kings = 4 → 
    prob_non_heart_king = 36 / 52 → prob_two_non_heart_king = (36 / 52) ^ 2 → 
    1 - prob_two_non_heart_king = 88 / 169 :=
by
  intros h_cards h_hearts h_kings h_prob_non_heart_king h_prob_two_non_heart_king
  sorry

end probability_heart_or_king_l156_156910


namespace find_original_shirt_price_l156_156948

noncomputable def original_shirt_price (S pants_orig_price jacket_orig_price total_paid : ℝ) :=
  let discounted_shirt := S * 0.5625
  let discounted_pants := pants_orig_price * 0.70
  let discounted_jacket := jacket_orig_price * 0.64
  let total_before_loyalty := discounted_shirt + discounted_pants + discounted_jacket
  let total_after_loyalty := total_before_loyalty * 0.90
  let total_after_tax := total_after_loyalty * 1.15
  total_after_tax = total_paid

theorem find_original_shirt_price : 
  original_shirt_price S 50 75 150 → S = 110.07 :=
by
  intro h
  sorry

end find_original_shirt_price_l156_156948


namespace placing_2_flowers_in_2_vases_l156_156995

noncomputable def num_ways_to_place_flowers (n k : ℕ) (h_n : n = 5) (h_k : k = 2) : ℕ :=
  Nat.choose n k * 2

theorem placing_2_flowers_in_2_vases :
  num_ways_to_place_flowers 5 2 rfl rfl = 20 := 
by
  sorry

end placing_2_flowers_in_2_vases_l156_156995


namespace find_a_l156_156560

open Complex

theorem find_a (a : ℝ) (i : ℂ := Complex.I) (h : (a - i) ^ 2 = 2 * i) : a = -1 :=
sorry

end find_a_l156_156560


namespace problem_acd_div_b_l156_156750

theorem problem_acd_div_b (a b c d : ℤ) (x : ℝ)
    (h1 : x = (a + b * Real.sqrt c) / d)
    (h2 : (7 * x) / 4 + 2 = 6 / x) :
    (a * c * d) / b = -322 := sorry

end problem_acd_div_b_l156_156750


namespace fraction_of_male_first_class_l156_156576

theorem fraction_of_male_first_class (total_passengers : ℕ) (percent_female : ℚ) (percent_first_class : ℚ)
    (females_in_coach : ℕ) (h1 : total_passengers = 120) (h2 : percent_female = 0.45) (h3 : percent_first_class = 0.10)
    (h4 : females_in_coach = 46) :
    (((percent_first_class * total_passengers - (percent_female * total_passengers - females_in_coach)))
    / (percent_first_class * total_passengers))  = 1 / 3 := 
by
  sorry

end fraction_of_male_first_class_l156_156576


namespace find_r_l156_156244

theorem find_r (r : ℝ) (h : ⌊r⌋ + r = 20.7) : r = 10.7 := 
by 
  sorry 

end find_r_l156_156244


namespace min_focal_length_hyperbola_l156_156371

theorem min_focal_length_hyperbola (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b - c = 2) : 
  2*c ≥ 4 + 4 * Real.sqrt 2 := 
sorry

end min_focal_length_hyperbola_l156_156371


namespace ratio_of_girls_with_long_hair_l156_156902

theorem ratio_of_girls_with_long_hair (total_people boys girls short_hair long_hair : ℕ)
  (h1 : total_people = 55)
  (h2 : boys = 30)
  (h3 : girls = total_people - boys)
  (h4 : short_hair = 10)
  (h5 : long_hair = girls - short_hhair) :
  long_hair / gcd long_hair girls = 3 ∧ girls / gcd long_hair girls = 5 := 
by {
  -- This placeholder indicates where the proof should be.
  sorry
}

end ratio_of_girls_with_long_hair_l156_156902


namespace no_five_coin_combination_for_70_cents_l156_156147

/-- Define the values of each coin type -/
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25

/-- Prove that it is not possible to achieve a total value of 70 cents with exactly five coins -/
theorem no_five_coin_combination_for_70_cents :
  ¬ ∃ a b c d e : ℕ, a + b + c + d + e = 5 ∧ a * penny + b * nickel + c * dime + d * quarter + e * quarter = 70 :=
sorry

end no_five_coin_combination_for_70_cents_l156_156147


namespace min_c_plus_3d_l156_156714

theorem min_c_plus_3d (c d : ℝ) (hc : 0 < c) (hd : 0 < d) 
    (h1 : c^2 ≥ 12 * d) (h2 : 9 * d^2 ≥ 4 * c) : 
  c + 3 * d ≥ 8 :=
  sorry

end min_c_plus_3d_l156_156714


namespace pig_problem_l156_156689

theorem pig_problem (x y : ℕ) (h₁ : y - 100 = 100 * x) (h₂ : y = 90 * x) : x = 10 ∧ y = 900 := 
by
  sorry

end pig_problem_l156_156689


namespace arithmetic_sequence_sum_l156_156206

-- Definitions used in the conditions
variable (a : ℕ → ℕ)
variable (n : ℕ)
variable (a_seq : Prop)
-- Declaring the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop := ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

noncomputable def a_5_is_2 : Prop := a 5 = 2

-- The statement we need to prove
theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h_arith_seq : is_arithmetic_sequence a) (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 := by
sorry

end arithmetic_sequence_sum_l156_156206


namespace scientific_notation_110_billion_l156_156893

theorem scientific_notation_110_billion :
  ∃ (n : ℝ) (e : ℤ), 110000000000 = n * 10 ^ e ∧ 1 ≤ n ∧ n < 10 ∧ n = 1.1 ∧ e = 11 :=
by
  sorry

end scientific_notation_110_billion_l156_156893


namespace initial_legos_500_l156_156488

-- Definitions and conditions from the problem
def initial_legos (x : ℕ) : Prop :=
  let used_pieces := x / 2
  let remaining_pieces := x - used_pieces
  let boxed_pieces := remaining_pieces - 5
  boxed_pieces = 245

-- Statement to be proven
theorem initial_legos_500 : initial_legos 500 :=
by
  -- Proof goes here
  sorry

end initial_legos_500_l156_156488


namespace total_credit_hours_l156_156752

def max_courses := 40
def max_courses_per_semester := 5
def max_courses_per_semester_credit := 3
def max_additional_courses_last_semester := 2
def max_additional_course_credit := 4
def sid_courses_multiplier := 4
def sid_additional_courses_multiplier := 2

theorem total_credit_hours (total_max_courses : Nat) 
                           (avg_max_courses_per_semester : Nat) 
                           (max_course_credit : Nat) 
                           (extra_max_courses_last_sem : Nat) 
                           (extra_max_course_credit : Nat) 
                           (sid_courses_mult : Nat) 
                           (sid_extra_courses_mult : Nat) 
                           (max_total_courses : total_max_courses = max_courses)
                           (max_avg_courses_per_semester : avg_max_courses_per_semester = max_courses_per_semester)
                           (max_course_credit_def : max_course_credit = max_courses_per_semester_credit)
                           (extra_max_courses_last_sem_def : extra_max_courses_last_sem = max_additional_courses_last_semester)
                           (extra_max_courses_credit_def : extra_max_course_credit = max_additional_course_credit)
                           (sid_courses_mult_def : sid_courses_mult = sid_courses_multiplier)
                           (sid_extra_courses_mult_def : sid_extra_courses_mult = sid_additional_courses_multiplier) : 
  total_max_courses * max_course_credit + extra_max_courses_last_sem * extra_max_course_credit + 
  (sid_courses_mult * total_max_courses - sid_extra_courses_mult * extra_max_courses_last_sem) * max_course_credit + sid_extra_courses_mult * extra_max_courses_last_sem * extra_max_course_credit = 606 := 
  by 
    sorry

end total_credit_hours_l156_156752


namespace prob1_prob2_l156_156522

-- Definition and theorems related to the calculations of the given problem.
theorem prob1 : ((-12) - 5 + (-14) - (-39)) = 8 := by 
  sorry

theorem prob2 : (-2^2 * 5 - (-12) / 4 - 4) = -21 := by
  sorry

end prob1_prob2_l156_156522


namespace mode_and_median_of_data_set_l156_156168

def data_set : List ℕ := [3, 5, 4, 6, 3, 3, 4]

noncomputable def mode_of_data_set : ℕ :=
  sorry  -- The mode calculation goes here (implementation is skipped)

noncomputable def median_of_data_set : ℕ :=
  sorry  -- The median calculation goes here (implementation is skipped)

theorem mode_and_median_of_data_set :
  mode_of_data_set = 3 ∧ median_of_data_set = 4 :=
  by
    sorry  -- Proof goes here

end mode_and_median_of_data_set_l156_156168


namespace credit_extended_l156_156035

noncomputable def automobile_installment_credit (total_consumer_credit : ℝ) : ℝ :=
  0.43 * total_consumer_credit

noncomputable def extended_by_finance_companies (auto_credit : ℝ) : ℝ :=
  0.25 * auto_credit

theorem credit_extended (total_consumer_credit : ℝ) (h : total_consumer_credit = 465.1162790697675) :
  extended_by_finance_companies (automobile_installment_credit total_consumer_credit) = 50.00 :=
by
  rw [h]
  sorry

end credit_extended_l156_156035


namespace difference_of_digits_l156_156867

theorem difference_of_digits (p q : ℕ) (h1 : ∀ n, n < 100 → n ≥ 10 → ∀ m, m < 100 → m ≥ 10 → 9 * (p - q) = 9) : 
  p - q = 1 :=
sorry

end difference_of_digits_l156_156867


namespace sum_eq_twenty_x_l156_156142

variable {R : Type*} [CommRing R] (x y z : R)

theorem sum_eq_twenty_x (h1 : y = 3 * x) (h2 : z = 3 * y) : 2 * x + 3 * y + z = 20 * x := by
  sorry

end sum_eq_twenty_x_l156_156142


namespace diagonal_length_of_quadrilateral_l156_156552

theorem diagonal_length_of_quadrilateral 
  (area : ℝ) (m n : ℝ) (d : ℝ) 
  (h_area : area = 210) 
  (h_m : m = 9) 
  (h_n : n = 6) 
  (h_formula : area = 0.5 * d * (m + n)) : 
  d = 28 :=
by 
  sorry

end diagonal_length_of_quadrilateral_l156_156552


namespace find_angle_x_l156_156403

theorem find_angle_x (A B C : Type) (angle_ABC angle_CAB x : ℝ) 
  (h1 : angle_ABC = 40) 
  (h2 : angle_CAB = 120)
  (triangle_sum : x + angle_ABC + (180 - angle_CAB) = 180) : 
  x = 80 :=
by 
  -- actual proof goes here
  sorry

end find_angle_x_l156_156403


namespace probability_neither_snow_nor_rain_in_5_days_l156_156139

def probability_no_snow (p_snow : ℚ) : ℚ := 1 - p_snow
def probability_no_rain (p_rain : ℚ) : ℚ := 1 - p_rain
def probability_no_snow_and_no_rain (p_no_snow p_no_rain : ℚ) : ℚ := p_no_snow * p_no_rain
def probability_no_snow_and_no_rain_5_days (p : ℚ) : ℚ := p ^ 5

theorem probability_neither_snow_nor_rain_in_5_days
    (p_snow : ℚ) (p_rain : ℚ)
    (h1 : p_snow = 2/3) (h2 : p_rain = 1/2) :
    probability_no_snow_and_no_rain_5_days (probability_no_snow_and_no_rain (probability_no_snow p_snow) (probability_no_rain p_rain)) = 1/7776 := by
  sorry

end probability_neither_snow_nor_rain_in_5_days_l156_156139


namespace range_of_k_real_roots_l156_156737

variable (k : ℝ)
def quadratic_has_real_roots : Prop :=
  let a := k - 1
  let b := 2
  let c := 1
  let Δ := b^2 - 4 * a * c
  Δ ≥ 0 ∧ a ≠ 0

theorem range_of_k_real_roots :
  quadratic_has_real_roots k ↔ (k ≤ 2 ∧ k ≠ 1) := by
  sorry

end range_of_k_real_roots_l156_156737


namespace quadratic_inequality_solution_l156_156477

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 3 * x + 2 < 0 ↔ 1 < x ∧ x < 2 :=
by
  sorry

end quadratic_inequality_solution_l156_156477


namespace depth_of_tank_proof_l156_156382

-- Definitions based on conditions
def length_of_tank : ℝ := 25
def width_of_tank : ℝ := 12
def cost_per_sq_meter : ℝ := 0.75
def total_cost : ℝ := 558

-- The depth of the tank to be proven as 6 meters
def depth_of_tank : ℝ := 6

-- Area of the tanks for walls and bottom
def plastered_area (d : ℝ) : ℝ := 2 * (length_of_tank * d) + 2 * (width_of_tank * d) + (length_of_tank * width_of_tank)

-- Final cost calculation
def plastering_cost (d : ℝ) : ℝ := cost_per_sq_meter * (plastered_area d)

-- Statement to be proven in Lean 4
theorem depth_of_tank_proof : plastering_cost depth_of_tank = total_cost :=
by
  sorry

end depth_of_tank_proof_l156_156382


namespace coffee_cost_per_week_l156_156521

theorem coffee_cost_per_week 
  (people_in_house : ℕ) 
  (drinks_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (num_days_in_week : ℕ) 
  (h1 : people_in_house = 4) 
  (h2 : drinks_per_person_per_day = 2)
  (h3 : ounces_per_cup = 0.5)
  (h4 : cost_per_ounce = 1.25)
  (h5 : num_days_in_week = 7) :
  people_in_house * drinks_per_person_per_day * ounces_per_cup * cost_per_ounce * num_days_in_week = 35 := 
by
  sorry

end coffee_cost_per_week_l156_156521


namespace range_of_square_of_difference_of_roots_l156_156308

theorem range_of_square_of_difference_of_roots (a : ℝ) (h : (a - 1) * (a - 2) < 0) :
  ∃ (S : Set ℝ), S = { x | 0 < x ∧ x ≤ 1 } ∧ ∀ (x1 x2 : ℝ),
  x1 + x2 = 2 * a ∧ x1 * x2 = 2 * a^2 - 3 * a + 2 → (x1 - x2)^2 ∈ S :=
sorry

end range_of_square_of_difference_of_roots_l156_156308


namespace quadratic_function_choice_l156_156380

-- Define what it means to be a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

-- Define the given equations as functions
def f_A (x : ℝ) : ℝ := 3 * x
def f_B (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def f_C (x : ℝ) : ℝ := (x - 1)^2
def f_D (x : ℝ) : ℝ := 2

-- State the Lean theorem statement
theorem quadratic_function_choice : is_quadratic f_C := sorry

end quadratic_function_choice_l156_156380


namespace total_people_clean_city_l156_156523

-- Define the conditions
def lizzie_group : Nat := 54
def group_difference : Nat := 17
def other_group := lizzie_group - group_difference

-- State the theorem
theorem total_people_clean_city : lizzie_group + other_group = 91 := by
  -- Proof would go here
  sorry

end total_people_clean_city_l156_156523


namespace wendy_tooth_extraction_cost_eq_290_l156_156665

def dentist_cleaning_cost : ℕ := 70
def dentist_filling_cost : ℕ := 120
def wendy_dentist_bill : ℕ := 5 * dentist_filling_cost
def wendy_cleaning_and_fillings_cost : ℕ := dentist_cleaning_cost + 2 * dentist_filling_cost
def wendy_tooth_extraction_cost : ℕ := wendy_dentist_bill - wendy_cleaning_and_fillings_cost

theorem wendy_tooth_extraction_cost_eq_290 : wendy_tooth_extraction_cost = 290 := by
  sorry

end wendy_tooth_extraction_cost_eq_290_l156_156665


namespace remainder_17_pow_2023_mod_28_l156_156980

theorem remainder_17_pow_2023_mod_28 :
  17^2023 % 28 = 17 := 
by sorry

end remainder_17_pow_2023_mod_28_l156_156980


namespace jennifer_boxes_l156_156572

theorem jennifer_boxes (kim_sold : ℕ) (h₁ : kim_sold = 54) (h₂ : ∃ jennifer_sold, jennifer_sold = kim_sold + 17) : ∃ jennifer_sold, jennifer_sold = 71 := by
  sorry

end jennifer_boxes_l156_156572


namespace bears_in_shipment_l156_156232

theorem bears_in_shipment (initial_bears shipment_bears bears_per_shelf total_shelves : ℕ)
  (h1 : initial_bears = 17)
  (h2 : bears_per_shelf = 9)
  (h3 : total_shelves = 3)
  (h4 : total_shelves * bears_per_shelf = 27) :
  shipment_bears = 10 :=
by
  sorry

end bears_in_shipment_l156_156232


namespace initial_population_is_10000_l156_156659

def population_growth (P : ℝ) : Prop :=
  let growth_rate := 0.20
  let final_population := 12000
  final_population = P * (1 + growth_rate)

theorem initial_population_is_10000 : population_growth 10000 :=
by
  unfold population_growth
  sorry

end initial_population_is_10000_l156_156659


namespace rowan_distance_downstream_l156_156009

-- Conditions
def speed_still : ℝ := 9.75
def downstream_time : ℝ := 2
def upstream_time : ℝ := 4

-- Statement to prove
theorem rowan_distance_downstream : ∃ (d : ℝ) (c : ℝ), 
  d / (speed_still + c) = downstream_time ∧
  d / (speed_still - c) = upstream_time ∧
  d = 26 := by
    sorry

end rowan_distance_downstream_l156_156009


namespace textbook_profit_l156_156366

theorem textbook_profit (cost_price selling_price : ℕ) (h1 : cost_price = 44) (h2 : selling_price = 55) :
  (selling_price - cost_price) = 11 := by
  sorry

end textbook_profit_l156_156366


namespace change_in_y_when_x_increases_l156_156634

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

-- State the theorem
theorem change_in_y_when_x_increases (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -5 :=
by
  sorry

end change_in_y_when_x_increases_l156_156634


namespace side_face_area_l156_156392

noncomputable def box_lengths (l w h : ℕ) : Prop :=
  (w * h = (1 / 2) * l * w ∧
   l * w = (3 / 2) * l * h ∧
   l * w * h = 5184 ∧
   2 * (l + h) = (6 / 5) * 2 * (l + w))

theorem side_face_area :
  ∃ (l w h : ℕ), box_lengths l w h ∧ l * h = 384 := by
  sorry

end side_face_area_l156_156392


namespace sequence_solution_l156_156517

theorem sequence_solution (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, (2*n - 1) * a (n + 1) = (2*n + 1) * a n) : 
∀ n : ℕ, a n = 2 * n - 1 := 
by
  sorry

end sequence_solution_l156_156517


namespace bowling_team_score_ratio_l156_156044

theorem bowling_team_score_ratio :
  ∀ (F S T : ℕ),
  F + S + T = 810 →
  F = (1 / 3 : ℚ) * S →
  T = 162 →
  S / T = 3 := 
by
  intros F S T h1 h2 h3
  sorry

end bowling_team_score_ratio_l156_156044


namespace sphere_tangent_plane_normal_line_l156_156803

variable {F : ℝ → ℝ → ℝ → ℝ}
def sphere (x y z : ℝ) : Prop := x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 5 = 0

def tangent_plane (x y z : ℝ) : Prop := 2*x + y + 2*z - 15 = 0

def normal_line (x y z : ℝ) : Prop := (x - 3) / 2 = (y + 1) / 1 ∧ (y + 1) / 1 = (z - 5) / 2

theorem sphere_tangent_plane_normal_line :
  sphere 3 (-1) 5 →
  tangent_plane 3 (-1) 5 ∧ normal_line 3 (-1) 5 :=
by
  intros h
  constructor
  sorry
  sorry

end sphere_tangent_plane_normal_line_l156_156803


namespace min_value_zero_l156_156977

noncomputable def f (k x y : ℝ) : ℝ :=
  3 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9

theorem min_value_zero (k : ℝ) :
  (∀ x y : ℝ, f k x y ≥ 0) ↔ (k = 3 / 2 ∨ k = -3 / 2) :=
by
  sorry

end min_value_zero_l156_156977


namespace solve_for_x_l156_156671

theorem solve_for_x (x y z : ℚ) (h1 : x * y = 2 * (x + y)) (h2 : y * z = 4 * (y + z)) (h3 : x * z = 8 * (x + z)) (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) : x = 16 / 3 := 
sorry

end solve_for_x_l156_156671


namespace cube_painting_l156_156219

theorem cube_painting (n : ℕ) (h1 : n > 3) 
  (h2 : 2 * (n-2) * (n-2) = 4 * (n-2)) :
  n = 4 :=
sorry

end cube_painting_l156_156219


namespace find_four_digit_number_l156_156164

theorem find_four_digit_number :
  ∃ A B C D : ℕ, 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
    (1001 * A + 100 * B + 10 * C + A) = 182 * (10 * C + D) ∧
    (1000 * A + 100 * B + 10 * C + D) = 2916 :=
by 
  sorry

end find_four_digit_number_l156_156164


namespace spent_more_on_candy_bar_l156_156913

-- Definitions of conditions
def money_Dan_has : ℕ := 2
def candy_bar_cost : ℕ := 6
def chocolate_cost : ℕ := 3

-- Statement of the proof problem
theorem spent_more_on_candy_bar : candy_bar_cost - chocolate_cost = 3 := by
  sorry

end spent_more_on_candy_bar_l156_156913


namespace valid_param_a_valid_param_c_l156_156051

/-
The task is to prove that the goals provided are valid parameterizations of the given line.
-/

def line_eqn (x y : ℝ) : Prop := y = -7/4 * x + 21/4

def is_valid_param (p₀ : ℝ × ℝ) (d : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line_eqn ((p₀.1 + t * d.1) : ℝ) ((p₀.2 + t * d.2) : ℝ)

theorem valid_param_a : is_valid_param (7, 0) (4, -7) :=
by
  sorry

theorem valid_param_c : is_valid_param (0, 21/4) (-4, 7) :=
by
  sorry


end valid_param_a_valid_param_c_l156_156051


namespace remainder_of_7n_div_4_l156_156165

theorem remainder_of_7n_div_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
sorry

end remainder_of_7n_div_4_l156_156165


namespace batsman_avg_l156_156735

variable (A : ℕ) -- The batting average in 46 innings

-- Given conditions
variables (highest lowest : ℕ)
variables (diff : ℕ) (avg_excl : ℕ) (num_excl : ℕ)

namespace cricket

-- Define the given values
def highest_score := 225
def difference := 150
def avg_excluding := 58
def num_excluding := 44

-- Calculate the lowest score
def lowest_score := highest_score - difference

-- Calculate the total runs in 44 innings excluding highest and lowest scores
def total_run_excluded := avg_excluding * num_excluding

-- Calculate the total runs in 46 innings
def total_runs := total_run_excluded + highest_score + lowest_score

-- Define the equation relating the average to everything else
def batting_avg_eq : Prop :=
  total_runs = 46 * A

-- Prove that the batting average A is 62 given the conditions
theorem batsman_avg :
  A = 62 :=
  by
    sorry

end cricket

end batsman_avg_l156_156735


namespace cent_piece_value_l156_156607

theorem cent_piece_value (Q P : ℕ) 
  (h1 : Q + P = 29)
  (h2 : 25 * Q + P = 545)
  (h3 : Q = 17) : 
  P = 120 := by
  sorry

end cent_piece_value_l156_156607


namespace value_of_x_minus_y_l156_156832

theorem value_of_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := 
by
  sorry

end value_of_x_minus_y_l156_156832


namespace parallelogram_height_base_difference_l156_156350

theorem parallelogram_height_base_difference (A B H : ℝ) (hA : A = 24) (hB : B = 4) (hArea : A = B * H) :
  H - B = 2 := by
  sorry

end parallelogram_height_base_difference_l156_156350


namespace total_fishes_l156_156500

theorem total_fishes (Will_catfish : ℕ) (Will_eels : ℕ) (Henry_multiplier : ℕ) (Henry_return_fraction : ℚ) :
  Will_catfish = 16 → Will_eels = 10 → Henry_multiplier = 3 → Henry_return_fraction = 1 / 2 →
  (Will_catfish + Will_eels) + (Henry_multiplier * Will_catfish - (Henry_multiplier * Will_catfish / 2)) = 50 := 
by
  intros h1 h2 h3 h4
  sorry

end total_fishes_l156_156500


namespace min_value_of_expression_l156_156593

open Real

theorem min_value_of_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 :=
by
  sorry

end min_value_of_expression_l156_156593


namespace band_row_lengths_l156_156688

theorem band_row_lengths (x y : ℕ) :
  (x * y = 90) → (5 ≤ x ∧ x ≤ 20) → (Even y) → False :=
by sorry

end band_row_lengths_l156_156688


namespace remainder_of_power_mod_l156_156239

noncomputable def carmichael (n : ℕ) : ℕ := sorry  -- Define Carmichael function (as a placeholder)

theorem remainder_of_power_mod :
  ∀ (n : ℕ), carmichael 1000 = 100 → carmichael 100 = 20 → 
    (5 ^ 5 ^ 5 ^ 5) % 1000 = 625 :=
by
  intros n h₁ h₂
  sorry

end remainder_of_power_mod_l156_156239


namespace no_cell_with_sum_2018_l156_156520

theorem no_cell_with_sum_2018 : ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 4900 → (5 * x = 2018 → false) := 
by
  intros x hx
  have h_bound : 1 ≤ x ∧ x ≤ 4900 := hx
  sorry

end no_cell_with_sum_2018_l156_156520


namespace three_digit_number_value_l156_156077

theorem three_digit_number_value (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
    (h4 : a > b) (h5 : b > c)
    (h6 : (10 * a + b) + (10 * b + a) = 55)  
    (h7 : 1300 < 222 * (a + b + c) ∧ 222 * (a + b + c) < 1400) : 
    (100 * a + 10 * b + c) = 321 := 
sorry

end three_digit_number_value_l156_156077


namespace inflation_over_two_years_real_interest_rate_l156_156150

-- Definitions for conditions
def annual_inflation_rate : ℝ := 0.025
def nominal_interest_rate : ℝ := 0.06

-- Lean statement for the first problem: Inflation over two years
theorem inflation_over_two_years :
  (1 + annual_inflation_rate) ^ 2 - 1 = 0.050625 := 
by sorry

-- Lean statement for the second problem: Real interest rate after inflation
theorem real_interest_rate (inflation_rate_two_years : ℝ)
  (h_inflation_rate : inflation_rate_two_years = 0.050625) :
  (nominal_interest_rate + 1) ^ 2 / (1 + inflation_rate_two_years) - 1 = 0.069459 :=
by sorry

end inflation_over_two_years_real_interest_rate_l156_156150


namespace expression_value_is_241_l156_156638

noncomputable def expression_value : ℕ :=
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2

theorem expression_value_is_241 : expression_value = 241 := 
by
  sorry

end expression_value_is_241_l156_156638


namespace total_yield_l156_156959

noncomputable def johnson_hectare_yield_2months : ℕ := 80
noncomputable def neighbor_hectare_yield_multiplier : ℕ := 2
noncomputable def neighbor_hectares : ℕ := 2
noncomputable def months : ℕ := 6

theorem total_yield (jh2 : ℕ := johnson_hectare_yield_2months) 
                    (nhm : ℕ := neighbor_hectare_yield_multiplier) 
                    (nh : ℕ := neighbor_hectares) 
                    (m : ℕ := months): 
                    3 * jh2 + 3 * nh * jh2 * nhm = 1200 :=
by
  sorry

end total_yield_l156_156959


namespace james_tip_percentage_l156_156786

theorem james_tip_percentage :
  let ticket_cost : ℝ := 100
  let dinner_cost : ℝ := 120
  let limo_cost_per_hour : ℝ := 80
  let limo_hours : ℕ := 6
  let total_cost_with_tip : ℝ := 836
  let total_cost_without_tip : ℝ := 2 * ticket_cost + limo_hours * limo_cost_per_hour + dinner_cost
  let tip : ℝ := total_cost_with_tip - total_cost_without_tip
  let percentage_tip : ℝ := (tip / dinner_cost) * 100
  percentage_tip = 30 :=
by
  sorry

end james_tip_percentage_l156_156786


namespace boys_or_girls_rink_l156_156650

variables (Class : Type) (is_boy : Class → Prop) (is_girl : Class → Prop) (visited_rink : Class → Prop) (met_at_rink : Class → Class → Prop)

-- Every student in the class visited the rink at least once.
axiom all_students_visited : ∀ (s : Class), visited_rink s

-- Every boy met every girl at the rink.
axiom boys_meet_girls : ∀ (b g : Class), is_boy b → is_girl g → met_at_rink b g

-- Prove that there exists a time when all the boys, or all the girls were simultaneously on the rink.
theorem boys_or_girls_rink : ∃ (t : Prop), (∀ b, is_boy b → visited_rink b) ∨ (∀ g, is_girl g → visited_rink g) :=
sorry

end boys_or_girls_rink_l156_156650


namespace sequence_recurrence_l156_156989

theorem sequence_recurrence (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : a 2 = 2) (h₃ : ∀ n, n ≥ 1 → a (n + 2) / a n = (a (n + 1) ^ 2 + 1) / (a n ^ 2 + 1)):
  (∀ n, a (n + 1) = a n + 1 / a n) ∧ 63 < a 2008 ∧ a 2008 < 78 :=
by
  sorry

end sequence_recurrence_l156_156989


namespace larger_number_is_17_l156_156853

noncomputable def x : ℤ := 17
noncomputable def y : ℤ := 12

def sum_condition : Prop := x + y = 29
def diff_condition : Prop := x - y = 5

theorem larger_number_is_17 (h_sum : sum_condition) (h_diff : diff_condition) : x = 17 :=
by {
  sorry
}

end larger_number_is_17_l156_156853


namespace product_gcd_lcm_4000_l156_156860

-- Definitions of gcd and lcm for the given numbers
def gcd_40_100 := Nat.gcd 40 100
def lcm_40_100 := Nat.lcm 40 100

-- Problem: Prove that the product of the gcd and lcm of 40 and 100 equals 4000
theorem product_gcd_lcm_4000 : gcd_40_100 * lcm_40_100 = 4000 := by
  sorry

end product_gcd_lcm_4000_l156_156860


namespace certain_number_l156_156838

theorem certain_number (n : ℕ) : 
  (55 * 57) % n = 6 ∧ n = 1043 :=
by
  sorry

end certain_number_l156_156838


namespace minimum_value_l156_156405

noncomputable def smallest_value_expression (x y : ℝ) := x^4 + y^4 - x^2 * y - x * y^2

theorem minimum_value (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y ≤ 1) :
  (smallest_value_expression x y) ≥ -1 / 8 :=
sorry

end minimum_value_l156_156405


namespace point_B_not_on_curve_C_l156_156926

theorem point_B_not_on_curve_C {a : ℝ} : 
  ¬ ((2 * a) ^ 2 + (4 * a) ^ 2 + 6 * a * (2 * a) - 8 * a * (4 * a) = 0) :=
by 
  sorry

end point_B_not_on_curve_C_l156_156926


namespace new_songs_added_l156_156755

-- Define the initial, deleted, and final total number of songs as constants
def initial_songs : ℕ := 8
def deleted_songs : ℕ := 5
def total_songs_now : ℕ := 33

-- Define and prove the number of new songs added
theorem new_songs_added : total_songs_now - (initial_songs - deleted_songs) = 30 :=
by
  sorry

end new_songs_added_l156_156755


namespace sodas_per_pack_l156_156897

theorem sodas_per_pack 
  (packs : ℕ) (initial_sodas : ℕ) (days_in_a_week : ℕ) (sodas_per_day : ℕ) 
  (total_sodas_consumed : ℕ) (sodas_per_pack : ℕ) :
  packs = 5 →
  initial_sodas = 10 →
  days_in_a_week = 7 →
  sodas_per_day = 10 →
  total_sodas_consumed = 70 →
  total_sodas_consumed - initial_sodas = packs * sodas_per_pack →
  sodas_per_pack = 12 :=
by
  intros hpacks hinitial hsodas hdaws htpd htcs
  sorry

end sodas_per_pack_l156_156897


namespace binary_addition_l156_156745

theorem binary_addition (a b : ℕ) :
  (a = (2^0 + 2^2 + 2^4 + 2^6)) → (b = (2^0 + 2^3 + 2^6)) →
  (a + b = 158) :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end binary_addition_l156_156745


namespace number_of_workers_l156_156858

-- Definitions for conditions
def initial_contribution (W C : ℕ) : Prop := W * C = 300000
def additional_contribution (W C : ℕ) : Prop := W * (C + 50) = 350000

-- Proof statement
theorem number_of_workers (W C : ℕ) (h1 : initial_contribution W C) (h2 : additional_contribution W C) : W = 1000 :=
by
  sorry

end number_of_workers_l156_156858


namespace irreducible_fraction_l156_156804

-- Definition of gcd
def my_gcd (m n : Int) : Int :=
  gcd m n

-- Statement of the problem
theorem irreducible_fraction (a : Int) : my_gcd (a^3 + 2 * a) (a^4 + 3 * a^2 + 1) = 1 :=
by
  sorry

end irreducible_fraction_l156_156804


namespace value_of_x_l156_156209

theorem value_of_x (x : ℝ) (h : x = 88 + 0.3 * 88) : x = 114.4 :=
by
  sorry

end value_of_x_l156_156209


namespace option_D_correct_l156_156249

theorem option_D_correct (a : ℝ) :
  3 * a ^ 2 - a ≠ 2 * a ∧
  a - (1 - 2 * a) ≠ a - 1 ∧
  -5 * (1 - a ^ 2) ≠ -5 - 5 * a ^ 2 ∧
  a ^ 3 + 7 * a ^ 3 - 5 * a ^ 3 = 3 * a ^ 3 :=
by
  sorry

end option_D_correct_l156_156249


namespace solve_for_A_l156_156138

variable (a b : ℝ) 

theorem solve_for_A (A : ℝ) (h : (5 * a + 3 * b)^2 = (5 * a - 3 * b)^2 + A) : 
  A = 60 * a * b := by
  sorry

end solve_for_A_l156_156138


namespace distinct_triangles_count_l156_156180

theorem distinct_triangles_count (n : ℕ) (hn : 0 < n) : 
  (∃ triangles_count, triangles_count = ⌊((n+1)^2 : ℝ)/4⌋) :=
sorry

end distinct_triangles_count_l156_156180


namespace number_of_single_rooms_l156_156014

theorem number_of_single_rooms (S : ℕ) : 
  (S + 13 * 2 = 40) ∧ (S * 10 + 13 * 2 * 10 = 400) → S = 14 :=
by 
  sorry

end number_of_single_rooms_l156_156014


namespace AC_plus_third_BA_l156_156836

def point := (ℝ × ℝ)

def A : point := (2, 4)
def B : point := (-1, -5)
def C : point := (3, -2)

noncomputable def vec (p₁ p₂ : point) : point :=
  (p₂.1 - p₁.1, p₂.2 - p₁.2)

noncomputable def scal_mult (scalar : ℝ) (v : point) : point :=
  (scalar * v.1, scalar * v.2)

noncomputable def vec_add (v₁ v₂ : point) : point :=
  (v₁.1 + v₂.1, v₁.2 + v₂.2)

theorem AC_plus_third_BA : 
  vec_add (vec A C) (scal_mult (1 / 3) (vec B A)) = (2, -3) :=
by
  sorry

end AC_plus_third_BA_l156_156836


namespace oranges_in_each_box_l156_156278

theorem oranges_in_each_box (O B : ℕ) (h1 : O = 24) (h2 : B = 3) :
  O / B = 8 :=
by
  sorry

end oranges_in_each_box_l156_156278


namespace tan_240_eq_sqrt3_l156_156110

theorem tan_240_eq_sqrt3 : Real.tan (240 * Real.pi / 180) = Real.sqrt 3 :=
by sorry

end tan_240_eq_sqrt3_l156_156110


namespace flower_bee_relationship_l156_156796

def numberOfBees (flowers : ℕ) (fewer_bees : ℕ) : ℕ :=
  flowers - fewer_bees

theorem flower_bee_relationship :
  numberOfBees 5 2 = 3 := by
  sorry

end flower_bee_relationship_l156_156796


namespace geometric_sequence_proof_l156_156421

theorem geometric_sequence_proof (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (n : ℕ) 
    (h1 : a 2 = 8) 
    (h2 : S 3 = 28) 
    (h3 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
    (h4 : ∀ n, a n = a 1 * q^(n-1)) 
    (h5 : q > 1) :
    (∀ n, a n = 2^(n + 1)) ∧ (∀ n, (a n)^2 > S n + 7) := sorry

end geometric_sequence_proof_l156_156421


namespace george_correct_answer_l156_156729

variable (y : ℝ)

theorem george_correct_answer (h : y / 7 = 30) : 70 + y = 280 :=
sorry

end george_correct_answer_l156_156729


namespace sphere_radius_twice_cone_volume_l156_156649

theorem sphere_radius_twice_cone_volume :
  ∀ (r_cone h_cone : ℝ) (r_sphere : ℝ), 
    r_cone = 2 → h_cone = 8 → 2 * (1 / 3 * Real.pi * r_cone^2 * h_cone) = (4/3 * Real.pi * r_sphere^3) → 
    r_sphere = 2^(4/3) :=
by
  intros r_cone h_cone r_sphere h_r_cone h_h_cone h_volume_equiv
  sorry

end sphere_radius_twice_cone_volume_l156_156649


namespace mans_speed_against_current_l156_156227

/-- Given the man's speed with the current and the speed of the current, prove the man's speed against the current. -/
theorem mans_speed_against_current
  (speed_with_current : ℝ) (speed_of_current : ℝ)
  (h1 : speed_with_current = 16)
  (h2 : speed_of_current = 3.2) :
  speed_with_current - 2 * speed_of_current = 9.6 :=
sorry

end mans_speed_against_current_l156_156227


namespace at_least_one_not_less_than_six_l156_156841

-- Definitions for the conditions.
variables {a b c : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The proof statement.
theorem at_least_one_not_less_than_six :
  (a + 4 / b) < 6 ∧ (b + 9 / c) < 6 ∧ (c + 16 / a) < 6 → false :=
by
  sorry

end at_least_one_not_less_than_six_l156_156841


namespace even_decreasing_function_l156_156827

theorem even_decreasing_function (f : ℝ → ℝ) (x1 x2 : ℝ)
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y → x < 0 → y < 0 → f y < f x)
  (hx1_neg : x1 < 0)
  (hx1x2_pos : x1 + x2 > 0) :
  f x1 < f x2 :=
sorry

end even_decreasing_function_l156_156827


namespace division_of_powers_l156_156645

theorem division_of_powers (a : ℝ) (h : a ≠ 0) : a^10 / a^9 = a := 
by sorry

end division_of_powers_l156_156645


namespace yanna_sandals_l156_156410

theorem yanna_sandals (shirts_cost: ℕ) (sandal_cost: ℕ) (total_money: ℕ) (change: ℕ) (num_shirts: ℕ)
  (h1: shirts_cost = 5)
  (h2: sandal_cost = 3)
  (h3: total_money = 100)
  (h4: change = 41)
  (h5: num_shirts = 10) : 
  ∃ num_sandals: ℕ, num_sandals = 3 :=
sorry

end yanna_sandals_l156_156410


namespace complex_number_purely_imaginary_l156_156945

theorem complex_number_purely_imaginary (a : ℝ) (i : ℂ) (h₁ : (a^2 - a - 2 : ℝ) = 0) (h₂ : (a + 1 ≠ 0)) : a = 2 := 
by {
  sorry
}

end complex_number_purely_imaginary_l156_156945


namespace equilateral_triangles_count_in_grid_of_side_4_l156_156951

-- Define a function to calculate the number of equilateral triangles in a triangular grid of side length n
def countEquilateralTriangles (n : ℕ) : ℕ :=
  (n * (n + 1) * (n + 2) * (n + 3)) / 24

-- Define the problem statement for n = 4
theorem equilateral_triangles_count_in_grid_of_side_4 :
  countEquilateralTriangles 4 = 35 := by
  sorry

end equilateral_triangles_count_in_grid_of_side_4_l156_156951


namespace square_diagonal_l156_156920

theorem square_diagonal (s d : ℝ) (h : 4 * s = 40) : d = s * Real.sqrt 2 → d = 10 * Real.sqrt 2 :=
by
  sorry

end square_diagonal_l156_156920


namespace certain_number_is_50_l156_156327

theorem certain_number_is_50 (x : ℝ) (h : 0.6 * x = 0.42 * 30 + 17.4) : x = 50 :=
by
  sorry

end certain_number_is_50_l156_156327


namespace tangent_line_length_l156_156835

noncomputable def curve_C (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

def problem_conditions : Prop :=
  curve_C 0 = 4 ∧ cartesian 4 0 = (4, 0)

theorem tangent_line_length :
  problem_conditions → 
  ∃ l : ℝ, l = 2 :=
by
  sorry

end tangent_line_length_l156_156835


namespace quadratic_inequality_always_holds_l156_156940

theorem quadratic_inequality_always_holds (k : ℝ) (h : ∀ x : ℝ, (x^2 - k*x + 1) > 0) : -2 < k ∧ k < 2 :=
  sorry

end quadratic_inequality_always_holds_l156_156940


namespace part1_part2_l156_156331

-- Conditions and the equation of the circle
def circleCenterLine (a : ℝ) : Prop := ∃ y, y = a + 2
def circleRadius : ℝ := 2
def pointOnCircle (A : ℝ × ℝ) (a : ℝ) : Prop := (A.1 - a)^2 + (A.2 - (a + 2))^2 = circleRadius^2
def tangentToYAxis (a : ℝ) : Prop := abs a = circleRadius

-- Problem 1: Proving the equation of the circle C
def circleEq (x y a : ℝ) : Prop := (x - a)^2 + (y - (a + 2))^2 = circleRadius^2

theorem part1 (a : ℝ) (h : abs a = circleRadius) (h1 : pointOnCircle (2, 2) a) 
    (h2 : circleCenterLine a) : circleEq 2 0 2 := 
sorry

-- Conditions and the properties for Problem 2
def distSquared (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
def QCondition (Q : ℝ × ℝ) : Prop := 
  distSquared Q (1, 3) - distSquared Q (1, 1) = 32
def onCircle (Q : ℝ × ℝ) (a : ℝ) : Prop := (Q.1 - a)^2 + (Q.2 - (a + 2))^2 = circleRadius^2

-- Problem 2: Proving the range of the abscissa a
theorem part2 (Q : ℝ × ℝ) (a : ℝ) 
    (hQ : QCondition Q) (hCircle : onCircle Q a) : 
    -3 ≤ a ∧ a ≤ 1 := 
sorry

end part1_part2_l156_156331


namespace proof_by_contradiction_example_l156_156346

theorem proof_by_contradiction_example (a b c : ℝ) (h : a < 3 ∧ b < 3 ∧ c < 3) : a < 1 ∨ b < 1 ∨ c < 1 := 
by
  have h1 : a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1 := sorry
  sorry

end proof_by_contradiction_example_l156_156346


namespace major_axis_endpoints_of_ellipse_l156_156740

theorem major_axis_endpoints_of_ellipse :
  ∀ x y, 6 * x^2 + y^2 = 6 ↔ (x = 0 ∧ (y = -Real.sqrt 6 ∨ y = Real.sqrt 6)) :=
by
  -- Proof
  sorry

end major_axis_endpoints_of_ellipse_l156_156740


namespace hannah_bought_two_sets_of_measuring_spoons_l156_156001

-- Definitions of conditions
def number_of_cookies_sold : ℕ := 40
def price_per_cookie : ℝ := 0.8
def number_of_cupcakes_sold : ℕ := 30
def price_per_cupcake : ℝ := 2.0
def cost_per_measuring_spoon_set : ℝ := 6.5
def remaining_money : ℝ := 79

-- Definition of total money made from selling cookies and cupcakes
def total_money_made : ℝ := (number_of_cookies_sold * price_per_cookie) + (number_of_cupcakes_sold * price_per_cupcake)

-- Definition of money spent on measuring spoons
def money_spent_on_measuring_spoons : ℝ := total_money_made - remaining_money

-- Theorem statement
theorem hannah_bought_two_sets_of_measuring_spoons :
  (money_spent_on_measuring_spoons / cost_per_measuring_spoon_set) = 2 := by
  sorry

end hannah_bought_two_sets_of_measuring_spoons_l156_156001


namespace polynomial_function_value_l156_156486

theorem polynomial_function_value 
  (p q r s : ℝ) 
  (h : p - q + r - s = 4) : 
  2 * p + q - 3 * r + 2 * s = -8 := 
by 
  sorry

end polynomial_function_value_l156_156486


namespace correct_transformation_l156_156622

variable {a b c : ℝ}

-- A: \frac{a+3}{b+3} = \frac{a}{b}
def transformation_A (a b : ℝ) : Prop := (a + 3) / (b + 3) = a / b

-- B: \frac{a}{b} = \frac{ac}{bc}
def transformation_B (a b c : ℝ) : Prop := a / b = (a * c) / (b * c)

-- C: \frac{3a}{3b} = \frac{a}{b}
def transformation_C (a b : ℝ) : Prop := (3 * a) / (3 * b) = a / b

-- D: \frac{a}{b} = \frac{a^2}{b^2}
def transformation_D (a b : ℝ) : Prop := a / b = (a ^ 2) / (b ^ 2)

-- The main theorem to prove
theorem correct_transformation : transformation_C a b :=
by
  sorry

end correct_transformation_l156_156622


namespace nested_geometric_sum_l156_156263

theorem nested_geometric_sum :
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4))))))))) = 1398100 :=
by
  sorry

end nested_geometric_sum_l156_156263


namespace range_of_a_l156_156623

-- Definition of sets A and B
def set_A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def set_B (a : ℝ) := {x : ℝ | 0 < x ∧ x < a}

-- Statement that if A ⊆ B, then a > 3
theorem range_of_a (a : ℝ) (h : set_A ⊆ set_B a) : 3 < a :=
by sorry

end range_of_a_l156_156623


namespace circle_general_eq_l156_156418
noncomputable def center_line (x : ℝ) := -4 * x
def tangent_line (x : ℝ) := 1 - x

def is_circle (center : ℝ × ℝ) (radius : ℝ) :=
  ∃ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2

def is_on_line (p : ℝ × ℝ) := (p.2 = center_line p.1)

def is_tangent_at_p (center : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) :=
  is_circle center r ∧ p.2 = tangent_line p.1 ∧ (center.1 - p.1)^2 + (center.2 - p.2)^2 = r^2

theorem circle_general_eq :
  ∀ (center : ℝ × ℝ), is_on_line center →
  ∀ r, is_tangent_at_p center (3, -2) r →
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = r^2 →
  x^2 + y^2 - 2 * x + 8 * y + 9 = 0 := by
  sorry

end circle_general_eq_l156_156418


namespace rotten_tomatoes_l156_156990

-- Conditions
def weight_per_crate := 20
def num_crates := 3
def total_cost := 330
def selling_price_per_kg := 6
def profit := 12

-- Derived data
def total_weight := num_crates * weight_per_crate
def total_revenue := profit + total_cost
def sold_weight := total_revenue / selling_price_per_kg

-- Proof statement
theorem rotten_tomatoes : total_weight - sold_weight = 3 := by
  sorry

end rotten_tomatoes_l156_156990


namespace factor_x4_plus_81_l156_156954

theorem factor_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9) :=
by 
  -- The proof is omitted.
  sorry

end factor_x4_plus_81_l156_156954


namespace inequality_solution_l156_156276

theorem inequality_solution (x : ℝ) :
  (-4 ≤ x ∧ x < -3 / 2) ↔ (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) :=
by
  sorry

end inequality_solution_l156_156276


namespace integer_solutions_of_log_inequality_l156_156070

def log_inequality_solution_set : Set ℤ := {0, 1, 2}

theorem integer_solutions_of_log_inequality (x : ℤ) (h : 2 < Real.log (x + 5) / Real.log 2 ∧ Real.log (x + 5) / Real.log 2 < 3) :
    x ∈ log_inequality_solution_set :=
sorry

end integer_solutions_of_log_inequality_l156_156070


namespace apricot_trees_count_l156_156390

theorem apricot_trees_count (peach_trees apricot_trees : ℕ) 
  (h1 : peach_trees = 300) 
  (h2 : peach_trees = 2 * apricot_trees + 30) : 
  apricot_trees = 135 := 
by 
  sorry

end apricot_trees_count_l156_156390


namespace digital_earth_concept_wrong_l156_156480

theorem digital_earth_concept_wrong :
  ∀ (A C D : Prop),
  (A → true) →
  (C → true) →
  (D → true) →
  ¬(B → true) :=
by
  sorry

end digital_earth_concept_wrong_l156_156480


namespace chickens_and_rabbits_l156_156654

theorem chickens_and_rabbits (total_animals : ℕ) (total_legs : ℕ) (chickens : ℕ) (rabbits : ℕ) 
    (h1 : total_animals = 40) 
    (h2 : total_legs = 108) 
    (h3 : total_animals = chickens + rabbits) 
    (h4 : total_legs = 2 * chickens + 4 * rabbits) : 
    chickens = 26 ∧ rabbits = 14 :=
by
  sorry

end chickens_and_rabbits_l156_156654


namespace sin_510_eq_1_div_2_l156_156406

theorem sin_510_eq_1_div_2 : Real.sin (510 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_510_eq_1_div_2_l156_156406


namespace age_difference_l156_156518

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 16) : A - C = 16 :=
sorry

end age_difference_l156_156518


namespace minimum_containers_needed_l156_156564

-- Definition of the problem conditions
def container_sizes := [5, 10, 20]
def target_units := 85

-- Proposition stating the minimum number of containers required
theorem minimum_containers_needed : 
  ∃ (x y z : ℕ), 
    5 * x + 10 * y + 20 * z = target_units ∧ 
    x + y + z = 5 :=
sorry

end minimum_containers_needed_l156_156564


namespace inequality_not_always_true_l156_156874

theorem inequality_not_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) : ¬ (∀ c, (a - b) / c > 0) := 
sorry

end inequality_not_always_true_l156_156874


namespace work_speed_ratio_l156_156930

open Real

theorem work_speed_ratio (A B : Type) 
  (A_work_speed B_work_speed : ℝ) 
  (combined_work_time : ℝ) 
  (B_work_time : ℝ)
  (h_combined : combined_work_time = 12)
  (h_B : B_work_time = 36)
  (combined_speed : A_work_speed + B_work_speed = 1 / combined_work_time)
  (B_speed : B_work_speed = 1 / B_work_time) :
  A_work_speed / B_work_speed = 2 :=
by sorry

end work_speed_ratio_l156_156930


namespace no_real_roots_quadratic_l156_156706

theorem no_real_roots_quadratic (a b c : ℝ) (h : a = 1 ∧ b = -4 ∧ c = 8) :
    (a ≠ 0) → (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) :=
by
  sorry

end no_real_roots_quadratic_l156_156706


namespace lia_quadrilateral_rod_count_l156_156698

theorem lia_quadrilateral_rod_count :
  let rods := {n : ℕ | 1 ≤ n ∧ n ≤ 40}
  let selected_rods := {5, 10, 20}
  let remaining_rods := rods \ selected_rods
  rod_count = 26 ∧ (∃ d ∈ remaining_rods, 
    (5 + 10 + 20) > d ∧ (10 + 20 + d) > 5 ∧ (5 + 20 + d) > 10 ∧ (5 + 10 + d) > 20)
:=
sorry

end lia_quadrilateral_rod_count_l156_156698


namespace four_digit_number_sum_of_digits_2023_l156_156462

theorem four_digit_number_sum_of_digits_2023 (a b c d : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) (hd : d < 10) :
  1000 * a + 100 * b + 10 * c + d = a + b + c + d + 2023 → 
  (1000 * a + 100 * b + 10 * c + d = 1997 ∨ 1000 * a + 100 * b + 10 * c + d = 2015) :=
by
  sorry

end four_digit_number_sum_of_digits_2023_l156_156462


namespace problem1_l156_156811

theorem problem1 (a b : ℝ) (i : ℝ) (h : (a-2*i)*i = b-i) : a^2 + b^2 = 5 := by
  sorry

end problem1_l156_156811


namespace scientific_notation_of_21500000_l156_156187

theorem scientific_notation_of_21500000 :
  21500000 = 2.15 * 10^7 :=
by
  sorry

end scientific_notation_of_21500000_l156_156187


namespace necessary_and_sufficient_problem_l156_156992

theorem necessary_and_sufficient_problem : 
  (¬ (∀ x : ℝ, (-2 < x ∧ x < 1) → (|x| > 1)) ∧ ¬ (∀ x : ℝ, (|x| > 1) → (-2 < x ∧ x < 1))) :=
by {
  sorry
}

end necessary_and_sufficient_problem_l156_156992


namespace benny_lunch_cost_l156_156746

theorem benny_lunch_cost :
  let person := 3;
  let cost_per_lunch := 8;
  let total_cost := person * cost_per_lunch;
  total_cost = 24 :=
by
  let person := 3;
  let cost_per_lunch := 8;
  let total_cost := person * cost_per_lunch;
  have h : total_cost = 24 := by
    sorry
  exact h

end benny_lunch_cost_l156_156746


namespace cistern_filled_in_12_hours_l156_156661

def fill_rate := 1 / 6
def empty_rate := 1 / 12
def net_rate := fill_rate - empty_rate

theorem cistern_filled_in_12_hours :
  (1 / net_rate) = 12 :=
by
  -- Proof omitted for clarity
  sorry

end cistern_filled_in_12_hours_l156_156661


namespace derivative_of_f_l156_156718

variable (x : ℝ)
def f (x : ℝ) := (5 * x - 4) ^ 3

theorem derivative_of_f :
  (deriv f x) = 15 * (5 * x - 4) ^ 2 :=
sorry

end derivative_of_f_l156_156718


namespace sum_of_k_values_l156_156281

-- Conditions
def P (x : ℝ) : ℝ := x^2 - 4 * x + 3
def Q (x k : ℝ) : ℝ := x^2 - 6 * x + k

-- Statement of the mathematical problem
theorem sum_of_k_values (k1 k2 : ℝ) (h1 : P 1 = 0) (h2 : P 3 = 0) 
  (h3 : Q 1 k1 = 0) (h4 : Q 3 k2 = 0) : k1 + k2 = 14 := 
by
  -- Here we would proceed with the proof steps corresponding to the solution
  sorry

end sum_of_k_values_l156_156281


namespace find_n_l156_156916

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end find_n_l156_156916


namespace decoration_sets_count_l156_156359

/-- 
Prove the number of different decoration sets that can be purchased for $120 dollars,
where each balloon costs $4, each ribbon costs $6, and the number of balloons must be even,
is exactly 2.
-/
theorem decoration_sets_count : 
  ∃ n : ℕ, n = 2 ∧ 
  (∃ (b r : ℕ), 
    4 * b + 6 * r = 120 ∧ 
    b % 2 = 0 ∧ 
    ∃ (i j : ℕ), 
      i ≠ j ∧ 
      (4 * i + 6 * (120 - 4 * i) / 6 = 120) ∧ 
      (4 * j + 6 * (120 - 4 * j) / 6 = 120) 
  )
:= sorry

end decoration_sets_count_l156_156359


namespace cost_per_bag_l156_156265

theorem cost_per_bag (total_friends: ℕ) (amount_paid_per_friend: ℕ) (total_bags: ℕ) 
  (h1 : total_friends = 3) (h2 : amount_paid_per_friend = 5) (h3 : total_bags = 5) 
  : total_friends * amount_paid_per_friend / total_bags = 3 := by
  sorry

end cost_per_bag_l156_156265


namespace sweets_remaining_l156_156419

def num_cherry := 30
def num_strawberry := 40
def num_pineapple := 50

def half (n : Nat) := n / 2

def num_eaten_cherry := half num_cherry
def num_eaten_strawberry := half num_strawberry
def num_eaten_pineapple := half num_pineapple

def num_given_away := 5

def total_initial := num_cherry + num_strawberry + num_pineapple

def total_eaten := num_eaten_cherry + num_eaten_strawberry + num_eaten_pineapple

def total_remaining_after_eating := total_initial - total_eaten
def total_remaining := total_remaining_after_eating - num_given_away

theorem sweets_remaining : total_remaining = 55 := by
  sorry

end sweets_remaining_l156_156419


namespace sufficient_not_necessary_condition_l156_156563

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (x - 1)^2 < 9 → (x + 2) * (x + a) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x + a) < 0 → ¬ (x - 1)^2 < 9) →
  a < -4 :=
by
  sorry

end sufficient_not_necessary_condition_l156_156563


namespace equation_of_plane_l156_156273

-- Definitions based on conditions
def line_equation (A B C x y : ℝ) : Prop :=
  A * x + B * y + C = 0

def A_B_nonzero (A B : ℝ) : Prop :=
  A ^ 2 + B ^ 2 ≠ 0

-- Statement for the problem
noncomputable def plane_equation (A B C D x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem equation_of_plane (A B C D : ℝ) :
  (A ^ 2 + B ^ 2 + C ^ 2 ≠ 0) → (∀ x y z : ℝ, plane_equation A B C D x y z) :=
by
  sorry

end equation_of_plane_l156_156273


namespace find_y_eq_l156_156483

theorem find_y_eq (y : ℝ) : (10 - y)^2 = 4 * y^2 → (y = 10 / 3 ∨ y = -10) :=
by
  intro h
  -- The detailed proof will be provided here
  sorry

end find_y_eq_l156_156483


namespace problem_solution_l156_156527

theorem problem_solution :
  ∃ n : ℕ, 50 < n ∧ n < 70 ∧ n % 5 = 3 ∧ n % 7 = 2 ∧ n = 58 :=
by
  -- Lean code to prove the statement
  sorry

end problem_solution_l156_156527


namespace estimate_fitness_population_l156_156944

theorem estimate_fitness_population :
  ∀ (sample_size total_population : ℕ) (sample_met_standards : Nat) (percentage_met_standards estimated_met_standards : ℝ),
  sample_size = 1000 →
  total_population = 1200000 →
  sample_met_standards = 950 →
  percentage_met_standards = (sample_met_standards : ℝ) / (sample_size : ℝ) →
  estimated_met_standards = percentage_met_standards * (total_population : ℝ) →
  estimated_met_standards = 1140000 := by sorry

end estimate_fitness_population_l156_156944


namespace total_floor_area_covered_l156_156801

theorem total_floor_area_covered (A B C : ℝ) 
  (h1 : A + B + C = 200) 
  (h2 : B = 24) 
  (h3 : C = 19) : 
  A - (B - C) - 2 * C = 138 := 
by sorry

end total_floor_area_covered_l156_156801


namespace wifi_cost_per_hour_l156_156088

-- Define the conditions as hypotheses
def ticket_cost : ℝ := 11
def snacks_cost : ℝ := 3
def headphones_cost : ℝ := 16
def hourly_income : ℝ := 12
def trip_duration : ℝ := 3
def total_expenses : ℝ := ticket_cost + snacks_cost + headphones_cost
def total_earnings : ℝ := hourly_income * trip_duration

-- Translate the proof problem to Lean 4 statement
theorem wifi_cost_per_hour: 
  (total_earnings - total_expenses) / trip_duration = 2 :=
by sorry

end wifi_cost_per_hour_l156_156088


namespace eval_expr1_l156_156334

theorem eval_expr1 : 
  ( (27 / 8) ^ (-2 / 3) - (49 / 9) ^ 0.5 + (0.008) ^ (-2 / 3) * (2 / 25) ) = 1 / 9 :=
by 
  sorry

end eval_expr1_l156_156334


namespace scout_weekend_earnings_l156_156144

-- Define the constants and conditions
def base_pay : ℝ := 10.0
def tip_per_delivery : ℝ := 5.0
def saturday_hours : ℝ := 4.0
def sunday_hours : ℝ := 5.0
def saturday_deliveries : ℝ := 5.0
def sunday_deliveries : ℝ := 8.0

-- Calculate total hours worked
def total_hours : ℝ := saturday_hours + sunday_hours

-- Calculate base pay for the weekend
def total_base_pay : ℝ := total_hours * base_pay

-- Calculate total number of deliveries
def total_deliveries : ℝ := saturday_deliveries + sunday_deliveries

-- Calculate total earnings from tips
def total_tips : ℝ := total_deliveries * tip_per_delivery

-- Calculate total earnings
def total_earnings : ℝ := total_base_pay + total_tips

-- Theorem to prove the total earnings is $155.00
theorem scout_weekend_earnings : total_earnings = 155.0 := by
  sorry

end scout_weekend_earnings_l156_156144


namespace solve_abs_system_eq_l156_156927

theorem solve_abs_system_eq (x y : ℝ) :
  (|x + y| + |1 - x| = 6) ∧ (|x + y + 1| + |1 - y| = 4) ↔ x = -2 ∧ y = -1 :=
by sorry

end solve_abs_system_eq_l156_156927


namespace magnitude_of_resultant_vector_is_sqrt_5_l156_156559

-- We denote the vectors a and b
def a : ℝ × ℝ := (1, -2)
def b (y : ℝ) : ℝ × ℝ := (-2, y)

-- We encode the condition that vectors are parallel
def parallel_vectors (y : ℝ) : Prop := 1 * y = (-2) * (-2)

-- We calculate the resultant vector and its magnitude
def resultant_vector (y : ℝ) : ℝ × ℝ :=
  ((3 * 1 + 2 * -2), (3 * -2 + 2 * y))

def magnitude_square (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

-- The target statement
theorem magnitude_of_resultant_vector_is_sqrt_5 (y : ℝ) (hy : parallel_vectors y) :
  magnitude_square (resultant_vector y) = 5 := by
  sorry

end magnitude_of_resultant_vector_is_sqrt_5_l156_156559


namespace n_greater_than_sqrt_p_sub_1_l156_156864

theorem n_greater_than_sqrt_p_sub_1 {p n : ℕ} (hp : Nat.Prime p) (hn : n ≥ 2) (hdiv : p ∣ (n^6 - 1)) : n > Nat.sqrt p - 1 := 
by
  sorry

end n_greater_than_sqrt_p_sub_1_l156_156864


namespace f_periodic_l156_156264

noncomputable def f : ℝ → ℝ := sorry

variable (a : ℝ) (h_a : 0 < a)
variable (h_cond : ∀ x : ℝ, f (x + a) = 1 / 2 + sqrt (f x - (f x)^2))

theorem f_periodic : ∀ x : ℝ, f (x + 2 * a) = f x := sorry

end f_periodic_l156_156264


namespace beaker_filling_l156_156361

theorem beaker_filling (C : ℝ) (hC : 0 < C) :
    let small_beaker_salt := (1/2) * C
    let large_beaker_capacity := 5 * C
    let large_beaker_fresh := large_beaker_capacity / 5
    let large_beaker_total_fill := large_beaker_fresh + small_beaker_salt
    (large_beaker_total_fill / large_beaker_capacity) = 3 / 10 :=
by
    let small_beaker_salt := (1/2) * C
    let large_beaker_capacity := 5 * C
    let large_beaker_fresh := large_beaker_capacity / 5
    let large_beaker_total_fill := large_beaker_fresh + small_beaker_salt
    show (large_beaker_total_fill / large_beaker_capacity) = 3 / 10
    sorry

end beaker_filling_l156_156361


namespace flat_tyre_problem_l156_156545

theorem flat_tyre_problem
    (x : ℝ)
    (h1 : 0 < x)
    (h2 : 1 / x + 1 / 6 = 1 / 5.6) :
  x = 84 :=
sorry

end flat_tyre_problem_l156_156545


namespace remainder_is_3_l156_156323

theorem remainder_is_3 (x y r : ℕ) (h1 : x = 7 * y + r) (h2 : 2 * x = 18 * y + 2) (h3 : 11 * y - x = 1)
  (hrange : 0 ≤ r ∧ r < 7) : r = 3 := 
sorry

end remainder_is_3_l156_156323


namespace inequality_holds_l156_156387

variable {a b c : ℝ}

theorem inequality_holds (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) : (a - b) * c ^ 2 ≤ 0 :=
sorry

end inequality_holds_l156_156387


namespace max_value_x_sub_2z_l156_156513

theorem max_value_x_sub_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 16) :
  ∃ m, m = 4 * Real.sqrt 5 ∧ ∀ x y z, x^2 + y^2 + z^2 = 16 → x - 2 * z ≤ m :=
sorry

end max_value_x_sub_2z_l156_156513


namespace total_number_of_meetings_proof_l156_156325

-- Define the conditions in Lean
variable (A B : Type)
variable (starting_time : ℕ)
variable (location_A location_B : A × B)

-- Define speeds
variable (speed_A speed_B : ℕ)

-- Define meeting counts
variable (total_meetings : ℕ)

-- Define A reaches point B 2015 times
variable (A_reaches_B_2015 : Prop)

-- Define that B travels twice as fast as A
axiom speed_ratio : speed_B = 2 * speed_A

-- Define that A reaches point B for the 5th time when B reaches it for the 9th time
axiom meeting_times : A_reaches_B_2015 → (total_meetings = 6044)

-- The Lean statement to prove
theorem total_number_of_meetings_proof : A_reaches_B_2015 → total_meetings = 6044 := by
  sorry

end total_number_of_meetings_proof_l156_156325


namespace train_time_to_cross_tree_l156_156834

-- Definitions based on conditions
def length_of_train := 1200 -- in meters
def time_to_pass_platform := 150 -- in seconds
def length_of_platform := 300 -- in meters
def total_distance := length_of_train + length_of_platform
def speed_of_train := total_distance / time_to_pass_platform
def time_to_cross_tree := length_of_train / speed_of_train

-- Theorem stating the main question
theorem train_time_to_cross_tree : time_to_cross_tree = 120 := by
  sorry

end train_time_to_cross_tree_l156_156834


namespace find_point_on_line_l156_156642

theorem find_point_on_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 3) : y = 16 / 7 :=
by
  sorry

end find_point_on_line_l156_156642


namespace triangle_isosceles_or_right_l156_156146

theorem triangle_isosceles_or_right (a b c : ℝ) (A B C : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (triangle_abc : A + B + C = 180)
  (opposite_sides : ∀ {x y}, x ≠ y → x + y < 180) 
  (condition : a * Real.cos A = b * Real.cos B) :
  (A = B ∨ A + B = 90) :=
by {
  sorry
}

end triangle_isosceles_or_right_l156_156146


namespace total_distance_traveled_l156_156115

noncomputable def totalDistance
  (d1 d2 : ℝ) (s1 s2 : ℝ) (average_speed : ℝ) (total_time : ℝ) : ℝ := 
  average_speed * total_time

theorem total_distance_traveled :
  let d1 := 160
  let s1 := 64
  let d2 := 160
  let s2 := 80
  let average_speed := 71.11111111111111
  let total_time := d1 / s1 + d2 / s2
  totalDistance d1 d2 s1 s2 average_speed total_time = 320 :=
by
  -- This is the main statement theorem
  sorry

end total_distance_traveled_l156_156115


namespace odds_against_C_l156_156032

theorem odds_against_C (pA pB : ℚ) (hA : pA = 1 / 5) (hB : pB = 2 / 3) :
  (1 - (1 - pA + 1 - pB)) / (1 - pA - pB) = 13 / 2 := 
sorry

end odds_against_C_l156_156032


namespace frog_ends_on_horizontal_side_l156_156693

-- Definitions for the problem conditions
def frog_jump_probability (x y : ℤ) : ℚ := sorry

-- Main theorem statement based on the identified question and correct answer
theorem frog_ends_on_horizontal_side :
  frog_jump_probability 2 3 = 13 / 14 :=
sorry

end frog_ends_on_horizontal_side_l156_156693


namespace complement_intersection_l156_156994

open Set

theorem complement_intersection (U A B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 5})
  (hB : B = {2, 4}) :
  ((U \ A) ∩ B) = {2, 4} :=
by
  sorry

end complement_intersection_l156_156994


namespace geometric_sequence_a5_l156_156152

theorem geometric_sequence_a5 (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r) (h2 : a 2 * a 8 = 4) : a 5 = 2 :=
sorry

end geometric_sequence_a5_l156_156152


namespace boat_cannot_complete_round_trip_l156_156942

theorem boat_cannot_complete_round_trip
  (speed_still_water : ℝ)
  (speed_current : ℝ)
  (distance : ℝ)
  (total_time : ℝ)
  (speed_still_water_pos : speed_still_water > 0)
  (speed_current_nonneg : speed_current ≥ 0)
  (distance_pos : distance > 0)
  (total_time_pos : total_time > 0) :
  let speed_downstream := speed_still_water + speed_current
  let speed_upstream := speed_still_water - speed_current
  let time_downstream := distance / speed_downstream
  let time_upstream := distance / speed_upstream
  let total_trip_time := time_downstream + time_upstream
  total_trip_time > total_time :=
by {
  -- Proof goes here
  sorry
}

end boat_cannot_complete_round_trip_l156_156942


namespace balance_proof_l156_156824

variable (a b c : ℕ)

theorem balance_proof (h1 : 5 * a + 2 * b = 15 * c) (h2 : 2 * a = b + 3 * c) : 4 * b = 7 * c :=
sorry

end balance_proof_l156_156824


namespace larger_integer_is_21_l156_156875

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l156_156875


namespace parallel_lines_necessary_not_sufficient_l156_156934

variables {a1 b1 a2 b2 c1 c2 : ℝ}

def determinant (a1 b1 a2 b2 : ℝ) : ℝ := a1 * b2 - a2 * b1

theorem parallel_lines_necessary_not_sufficient
  (h1 : a1^2 + b1^2 ≠ 0)
  (h2 : a2^2 + b2^2 ≠ 0)
  : (determinant a1 b1 a2 b2 = 0) → 
    (a1 * x + b1 * y + c1 = 0 ∧ a2 * x + b2 * y + c2 =0 → exists k : ℝ, (a1 = k ∧ b1 = k)) ∧ 
    (determinant a1 b1 a2 b2 = 0 → (a2 * x + b2 * y + c2 = a1 * x + b1 * y + c1 → false)) :=
sorry

end parallel_lines_necessary_not_sufficient_l156_156934


namespace solution_set_of_inequality_l156_156293

theorem solution_set_of_inequality (x : ℝ) : (1 / |x - 1| ≥ 1) ↔ (0 ≤ x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) :=
by
  sorry

end solution_set_of_inequality_l156_156293


namespace perpendicular_slope_l156_156656

theorem perpendicular_slope (x y : ℝ) (h : 3 * x - 4 * y = 8) :
  ∃ m : ℝ, m = - (4 / 3) :=
by
  sorry

end perpendicular_slope_l156_156656


namespace f_increasing_maximum_b_condition_approximate_ln2_l156_156987

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem f_increasing : ∀ x y : ℝ, x < y → f x ≤ f y := 
sorry

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := f (2 * x) - 4 * b * f x

theorem maximum_b_condition (x : ℝ) (H : 0 < x): ∃ b, g x b > 0 ∧ b ≤ 2 := 
sorry

theorem approximate_ln2 :
  0.692 ≤ Real.log 2 ∧ Real.log 2 ≤ 0.694 :=
sorry

end f_increasing_maximum_b_condition_approximate_ln2_l156_156987


namespace combined_weight_is_18442_l156_156337

noncomputable def combined_weight_proof : ℝ :=
  let elephant_weight_tons := 3
  let donkey_weight_percentage := 0.1
  let giraffe_weight_tons := 1.5
  let hippopotamus_weight_kg := 4000
  let elephant_food_oz := 16
  let donkey_food_lbs := 5
  let giraffe_food_kg := 3
  let hippopotamus_food_g := 5000

  let ton_to_pounds := 2000
  let kg_to_pounds := 2.20462
  let oz_to_pounds := 1 / 16
  let g_to_pounds := 0.00220462

  let elephant_weight_pounds := elephant_weight_tons * ton_to_pounds
  let donkey_weight_pounds := (1 - donkey_weight_percentage) * elephant_weight_pounds
  let giraffe_weight_pounds := giraffe_weight_tons * ton_to_pounds
  let hippopotamus_weight_pounds := hippopotamus_weight_kg * kg_to_pounds

  let elephant_food_pounds := elephant_food_oz * oz_to_pounds
  let giraffe_food_pounds := giraffe_food_kg * kg_to_pounds
  let hippopotamus_food_pounds := hippopotamus_food_g * g_to_pounds

  elephant_weight_pounds + donkey_weight_pounds + giraffe_weight_pounds + hippopotamus_weight_pounds +
  elephant_food_pounds + donkey_food_lbs + giraffe_food_pounds + hippopotamus_food_pounds

theorem combined_weight_is_18442 : combined_weight_proof = 18442 := by
  sorry

end combined_weight_is_18442_l156_156337


namespace average_boxes_per_day_by_third_day_l156_156767

theorem average_boxes_per_day_by_third_day (day1 day2 day3_part1 day3_part2 : ℕ) :
  day1 = 318 →
  day2 = 312 →
  day3_part1 = 180 →
  day3_part2 = 162 →
  ((day1 + day2 + (day3_part1 + day3_part2)) / 3) = 324 :=
by
  intros h1 h2 h3 h4
  sorry

end average_boxes_per_day_by_third_day_l156_156767


namespace twenty_five_billion_scientific_notation_l156_156660

theorem twenty_five_billion_scientific_notation :
  (25 * 10^9 : ℝ) = 2.5 * 10^10 := 
by simp only [←mul_assoc, ←@pow_add ℝ, pow_one, two_mul];
   norm_num

end twenty_five_billion_scientific_notation_l156_156660


namespace layla_more_than_nahima_l156_156616

-- Definitions for the conditions
def layla_points : ℕ := 70
def total_points : ℕ := 112
def nahima_points : ℕ := total_points - layla_points

-- Statement of the theorem
theorem layla_more_than_nahima : layla_points - nahima_points = 28 :=
by
  sorry

end layla_more_than_nahima_l156_156616


namespace calculate_120ab_l156_156316

variable (a b : ℚ)

theorem calculate_120ab (h1 : 10 * a = 20) (h2 : 6 * b = 20) : 120 * (a * b) = 800 := by
  sorry

end calculate_120ab_l156_156316


namespace find_stadium_width_l156_156794

-- Conditions
def stadium_length : ℝ := 24
def stadium_height : ℝ := 16
def longest_pole : ℝ := 34

-- Width to be solved
def stadium_width : ℝ := 18

-- Theorem stating that given the conditions, the width must be 18
theorem find_stadium_width :
  stadium_length^2 + stadium_width^2 + stadium_height^2 = longest_pole^2 :=
by
  sorry

end find_stadium_width_l156_156794


namespace range_of_a_quadratic_root_conditions_l156_156254

theorem range_of_a_quadratic_root_conditions (a : ℝ) :
  ((∃ x₁ x₂ : ℝ, x₁ > 2 ∧ x₂ < 2 ∧ (ax^2 - 2*(a+1)*x + a-1 = 0)) ↔ (0 < a ∧ a < 5)) :=
by
  sorry

end range_of_a_quadratic_root_conditions_l156_156254


namespace sixth_largest_divisor_correct_l156_156443

noncomputable def sixth_largest_divisor_of_4056600000 : ℕ :=
  50707500

theorem sixth_largest_divisor_correct : sixth_largest_divisor_of_4056600000 = 50707500 :=
sorry

end sixth_largest_divisor_correct_l156_156443


namespace bhanu_income_problem_l156_156061

-- Define the total income
def total_income (I : ℝ) : Prop :=
  let petrol_spent := 300
  let house_rent := 70
  (0.10 * (I - petrol_spent) = house_rent)

-- Define the percentage of income spent on petrol
def petrol_percentage (P : ℝ) (I : ℝ) : Prop :=
  0.01 * P * I = 300

-- The theorem we aim to prove
theorem bhanu_income_problem : 
  ∃ I P, total_income I ∧ petrol_percentage P I ∧ P = 30 :=
by
  sorry

end bhanu_income_problem_l156_156061


namespace find_polygon_sides_l156_156633

theorem find_polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by
  sorry

end find_polygon_sides_l156_156633


namespace books_in_bin_after_transactions_l156_156199

def initial_books : ℕ := 4
def sold_books : ℕ := 3
def added_books : ℕ := 10

def final_books (initial_books sold_books added_books : ℕ) : ℕ :=
  initial_books - sold_books + added_books

theorem books_in_bin_after_transactions :
  final_books initial_books sold_books added_books = 11 := by
  sorry

end books_in_bin_after_transactions_l156_156199


namespace x_varies_inversely_l156_156547

theorem x_varies_inversely (y: ℝ) (x: ℝ): (∃ k: ℝ, (∀ y: ℝ, x = k / y ^ 2) ∧ (1 = k / 3 ^ 2)) → x = 0.5625 :=
by
  sorry

end x_varies_inversely_l156_156547


namespace b_profit_share_l156_156917

theorem b_profit_share (total_capital : ℝ) (profit : ℝ) (A_invest : ℝ) (B_invest : ℝ) (C_invest : ℝ) (D_invest : ℝ)
 (A_time : ℝ) (B_time : ℝ) (C_time : ℝ) (D_time : ℝ) :
  total_capital = 100000 ∧
  A_invest = B_invest + 10000 ∧
  B_invest = C_invest + 5000 ∧
  D_invest = A_invest + 8000 ∧
  A_time = 12 ∧
  B_time = 10 ∧
  C_time = 8 ∧
  D_time = 6 ∧
  profit = 50000 →
  (B_invest * B_time / (A_invest * A_time + B_invest * B_time + C_invest * C_time + D_invest * D_time)) * profit = 10925 :=
by
  sorry

end b_profit_share_l156_156917


namespace determine_g_l156_156494

theorem determine_g (g : ℝ → ℝ) : (∀ x : ℝ, 4 * x^4 + x^3 - 2 * x + 5 + g x = 2 * x^3 - 7 * x^2 + 4) →
  (∀ x : ℝ, g x = -4 * x^4 + x^3 - 7 * x^2 + 2 * x - 1) :=
by
  intro h
  sorry

end determine_g_l156_156494


namespace quadratic_factor_conditions_l156_156162

theorem quadratic_factor_conditions (b : ℤ) :
  (∃ m n p q : ℤ, m * p = 15 ∧ n * q = 75 ∧ mq + np = b) → ∃ (c : ℤ), b = c :=
sorry

end quadratic_factor_conditions_l156_156162


namespace staples_left_in_stapler_l156_156586

def initial_staples : ℕ := 50
def reports_stapled : ℕ := 3 * 12
def staples_per_report : ℕ := 1
def remaining_staples : ℕ := initial_staples - (reports_stapled * staples_per_report)

theorem staples_left_in_stapler : remaining_staples = 14 :=
by
  sorry

end staples_left_in_stapler_l156_156586


namespace minimum_value_of_quadratic_l156_156805

theorem minimum_value_of_quadratic :
  ∃ x : ℝ, (x = 6) ∧ (∀ y : ℝ, (y^2 - 12 * y + 32) ≥ -4) :=
sorry

end minimum_value_of_quadratic_l156_156805


namespace university_students_l156_156082

theorem university_students (total_students students_both math_students physics_students : ℕ) 
  (h1 : total_students = 75) 
  (h2 : total_students = (math_students - students_both) + (physics_students - students_both) + students_both)
  (h3 : math_students = 2 * physics_students) 
  (h4 : students_both = 10) : 
  math_students = 56 := by
  sorry

end university_students_l156_156082


namespace Adam_current_money_is_8_l156_156594

variable (Adam_initial : ℕ) (spent_on_game : ℕ) (allowance : ℕ)

def money_left_after_spending (initial : ℕ) (spent : ℕ) := initial - spent
def current_money (money_left : ℕ) (allowance : ℕ) := money_left + allowance

theorem Adam_current_money_is_8 
    (h1 : Adam_initial = 5)
    (h2 : spent_on_game = 2)
    (h3 : allowance = 5) :
    current_money (money_left_after_spending Adam_initial spent_on_game) allowance = 8 := 
by sorry

end Adam_current_money_is_8_l156_156594


namespace find_f_42_div_17_l156_156367

def f : ℚ → ℤ := sorry

theorem find_f_42_div_17 : 
  (∀ x y : ℚ, x ≠ y → (x * y = 1 ∨ x + y = 1) → f x * f y = -1) → 
  f 0 = 1 →
  f (42 / 17) = -1 :=
sorry

end find_f_42_div_17_l156_156367


namespace inequality_chain_l156_156041

theorem inequality_chain (a b : ℝ) (h₁ : a < 0) (h₂ : -1 < b) (h₃ : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_chain_l156_156041


namespace max_digit_sum_watch_l156_156727

def digit_sum (n : Nat) : Nat :=
  (n / 10) + (n % 10)

theorem max_digit_sum_watch :
  ∃ (h m : Nat), (1 <= h ∧ h <= 12) ∧ (0 <= m ∧ m <= 59) 
  ∧ (digit_sum h + digit_sum m = 23) :=
by 
  sorry

end max_digit_sum_watch_l156_156727


namespace find_cuboid_length_l156_156054

theorem find_cuboid_length
  (b : ℝ) (h : ℝ) (S : ℝ)
  (hb : b = 10) (hh : h = 12) (hS : S = 960) :
  ∃ l : ℝ, 2 * (l * b + b * h + h * l) = S ∧ l = 16.36 :=
by
  sorry

end find_cuboid_length_l156_156054


namespace directrix_of_parabola_l156_156511

theorem directrix_of_parabola (p : ℝ) (y x : ℝ) :
  y = x^2 → x^2 = 4 * p * y → 4 * y + 1 = 0 :=
by
  intros hyp1 hyp2
  sorry

end directrix_of_parabola_l156_156511


namespace tan_315_eq_neg1_l156_156677

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 := 
by
  sorry

end tan_315_eq_neg1_l156_156677


namespace find_natural_pairs_l156_156031

-- Definitions
def is_natural (n : ℕ) : Prop := n > 0
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def satisfies_equation (x y : ℕ) : Prop := 2 * x^2 + 5 * x * y + 3 * y^2 = 41 * x + 62 * y + 21

-- Problem statement
theorem find_natural_pairs (x y : ℕ) (hx : is_natural x) (hy : is_natural y) (hrel : relatively_prime x y) :
  satisfies_equation x y ↔ (x = 2 ∧ y = 19) ∨ (x = 19 ∧ y = 2) :=
by
  sorry

end find_natural_pairs_l156_156031


namespace cakes_difference_l156_156973

-- Definitions of the given conditions
def cakes_sold : ℕ := 78
def cakes_bought : ℕ := 31

-- The theorem to prove
theorem cakes_difference : cakes_sold - cakes_bought = 47 :=
by sorry

end cakes_difference_l156_156973


namespace tower_total_surface_area_l156_156626

/-- Given seven cubes with volumes 1, 8, 27, 64, 125, 216, and 343 cubic units each, stacked vertically
    with volumes decreasing from bottom to top, compute their total surface area including the bottom. -/
theorem tower_total_surface_area :
  let volumes := [1, 8, 27, 64, 125, 216, 343]
  let side_lengths := volumes.map (fun v => v ^ (1 / 3))
  let surface_area (n : ℝ) (visible_faces : ℕ) := visible_faces * (n ^ 2)
  let total_surface_area := surface_area 7 5 + surface_area 6 4 + surface_area 5 4 + surface_area 4 4
                            + surface_area 3 4 + surface_area 2 4 + surface_area 1 5
  total_surface_area = 610 := sorry

end tower_total_surface_area_l156_156626


namespace total_rainfall_l156_156370

theorem total_rainfall
  (monday : ℝ)
  (tuesday : ℝ)
  (wednesday : ℝ)
  (h_monday : monday = 0.17)
  (h_tuesday : tuesday = 0.42)
  (h_wednesday : wednesday = 0.08) :
  monday + tuesday + wednesday = 0.67 :=
by
  sorry

end total_rainfall_l156_156370


namespace tutors_meet_in_360_days_l156_156372

noncomputable def lcm_four_days : ℕ := Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9)

theorem tutors_meet_in_360_days :
  lcm_four_days = 360 := 
by
  -- The proof steps are omitted.
  sorry

end tutors_meet_in_360_days_l156_156372


namespace smallest_bdf_value_l156_156450

open Nat

theorem smallest_bdf_value
  (a b c d e f : ℕ)
  (h1 : (a + 1) * c * e = a * c * e + 3 * b * d * f)
  (h2 : a * (c + 1) * e = a * c * e + 4 * b * d * f)
  (h3 : a * c * (e + 1) = a * c * e + 5 * b * d * f) :
  (b * d * f = 60) :=
sorry

end smallest_bdf_value_l156_156450


namespace find_value_sum_l156_156384

noncomputable def f : ℝ → ℝ
  := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 3) = f x
axiom value_at_minus_one : f (-1) = 1

theorem find_value_sum :
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry

end find_value_sum_l156_156384


namespace p_necessary_not_sufficient_for_q_l156_156617

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def p (a : ℝ) : Prop :=
  collinear (vec a (a^2)) (vec 1 2)

def q (a : ℝ) : Prop := a = 2

theorem p_necessary_not_sufficient_for_q :
  (∀ a : ℝ, q a → p a) ∧ ¬(∀ a : ℝ, p a → q a) :=
sorry

end p_necessary_not_sufficient_for_q_l156_156617


namespace total_pupils_correct_l156_156812

-- Definitions of the conditions
def number_of_girls : ℕ := 308
def number_of_boys : ℕ := 318

-- Definition of the number of pupils
def total_number_of_pupils : ℕ := number_of_girls + number_of_boys

-- The theorem to be proven
theorem total_pupils_correct : total_number_of_pupils = 626 := by
  -- The proof would go here
  sorry

end total_pupils_correct_l156_156812


namespace game_show_prize_guess_l156_156100

noncomputable def total_possible_guesses : ℕ :=
  (Nat.choose 8 3) * (Nat.choose 5 3) * (Nat.choose 2 2) * (Nat.choose 7 3)

theorem game_show_prize_guess :
  total_possible_guesses = 19600 :=
by
  -- Omitted proof steps
  sorry

end game_show_prize_guess_l156_156100


namespace number_of_divisors_30_l156_156787

theorem number_of_divisors_30 : 
  ∃ (d : ℕ), d = 2 * 2 * 2 ∧ d = 8 :=
  by sorry

end number_of_divisors_30_l156_156787


namespace parabola_vertex_and_point_l156_156953

theorem parabola_vertex_and_point (a b c : ℝ) : 
  (∀ x, y = a * x^2 + b * x + c) ∧ 
  ∃ x y, (y = a * (x - 4)^2 + 3) → 
  (a * 2^2 + b * 2 + c = 5) → 
  (a = 1/2 ∧ b = -4 ∧ c = 11) :=
by
  sorry

end parabola_vertex_and_point_l156_156953


namespace average_computer_time_per_person_is_95_l156_156695

def people : ℕ := 8
def computers : ℕ := 5
def work_time : ℕ := 152 -- total working day minutes

def total_computer_time : ℕ := work_time * computers
def average_time_per_person : ℕ := total_computer_time / people

theorem average_computer_time_per_person_is_95 :
  average_time_per_person = 95 := 
by
  sorry

end average_computer_time_per_person_is_95_l156_156695


namespace canada_population_l156_156043

theorem canada_population 
    (M : ℕ) (B : ℕ) (H : ℕ)
    (hM : M = 1000000)
    (hB : B = 2 * M)
    (hH : H = 19 * B) : 
    H = 38000000 := by
  sorry

end canada_population_l156_156043
