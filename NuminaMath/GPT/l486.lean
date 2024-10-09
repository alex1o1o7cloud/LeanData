import Mathlib

namespace speed_of_stream_l486_48669

def boatSpeedDownstream (V_b V_s : ℝ) : ℝ :=
  V_b + V_s

def boatSpeedUpstream (V_b V_s : ℝ) : ℝ :=
  V_b - V_s

theorem speed_of_stream (V_b V_s : ℝ) (h1 : V_b + V_s = 25) (h2 : V_b - V_s = 5) : V_s = 10 :=
by {
  sorry
}

end speed_of_stream_l486_48669


namespace fixed_constant_t_l486_48626

-- Representation of point on the Cartesian plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of the parabola y = 4x^2
def parabola (p : Point) : Prop := p.y = 4 * p.x^2

-- Definition of distance squared between two points
def distance_squared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Main theorem statement
theorem fixed_constant_t :
  ∃ (c : ℝ) (C : Point), c = 1/8 ∧ C = ⟨1, c⟩ ∧ 
  (∀ (A B : Point), parabola A ∧ parabola B ∧ 
  (∃ m k : ℝ, A.y = m * A.x + k ∧ B.y = m * B.x + k ∧ k = c - m) → 
  (1 / distance_squared A C + 1 / distance_squared B C = 16)) :=
by {
  -- Proof omitted
  sorry
}

end fixed_constant_t_l486_48626


namespace fractions_arithmetic_lemma_l486_48660

theorem fractions_arithmetic_lemma : (8 / 15 : ℚ) - (7 / 9) + (3 / 4) = 1 / 2 := 
by
  sorry

end fractions_arithmetic_lemma_l486_48660


namespace trapezoid_length_relation_l486_48627

variables {A B C D M N : Type}
variables [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]
variables (a b c d m n : A)
variables (h_parallel_ab_cd : A) (h_parallel_mn_ab : A) 

-- The required proof statement
theorem trapezoid_length_relation (H1 : a = h_parallel_ab_cd) 
(H2 : b = m * n + h_parallel_mn_ab - m * d)
(H3 : c = d * (h_parallel_mn_ab - a))
(H4 : n = d / (n - a))
(H5 : n = c - h_parallel_ab_cd) :
c * m * a + b * c * d = n * d * a :=
sorry

end trapezoid_length_relation_l486_48627


namespace max_marks_l486_48653

theorem max_marks (marks_obtained failed_by : ℝ) (passing_percentage : ℝ) (M : ℝ) : 
  marks_obtained = 180 ∧ failed_by = 40 ∧ passing_percentage = 0.45 ∧ (marks_obtained + failed_by = passing_percentage * M) → M = 489 :=
by 
  sorry

end max_marks_l486_48653


namespace H2O_required_for_NaH_reaction_l486_48629

theorem H2O_required_for_NaH_reaction
  (n_NaH : ℕ) (n_H2O : ℕ) (n_NaOH : ℕ) (n_H2 : ℕ)
  (h_eq : n_NaH = 2) (balanced_eq : n_NaH = n_H2O ∧ n_H2O = n_NaOH ∧ n_NaOH = n_H2) :
  n_H2O = 2 :=
by
  -- The proof is omitted as we only need to declare the statement.
  sorry

end H2O_required_for_NaH_reaction_l486_48629


namespace additional_discount_percentage_l486_48644

-- Define constants representing the conditions
def price_shoes : ℝ := 200
def discount_shoes : ℝ := 0.30
def price_shirt : ℝ := 80
def number_shirts : ℕ := 2
def final_spent : ℝ := 285

-- Define the theorem to prove the additional discount percentage
theorem additional_discount_percentage :
  let discounted_shoes := price_shoes * (1 - discount_shoes)
  let total_before_additional_discount := discounted_shoes + number_shirts * price_shirt
  let additional_discount := total_before_additional_discount - final_spent
  (additional_discount / total_before_additional_discount) * 100 = 5 :=
by
  -- Lean proof goes here, but we'll skip it for now with sorry
  sorry

end additional_discount_percentage_l486_48644


namespace count_positive_integers_satisfying_inequality_l486_48606

theorem count_positive_integers_satisfying_inequality :
  ∃ n : ℕ, n = 4 ∧ ∀ x : ℕ, (10 < x^2 + 6 * x + 9 ∧ x^2 + 6 * x + 9 < 50) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) := 
by
  sorry

end count_positive_integers_satisfying_inequality_l486_48606


namespace ascorbic_acid_weight_l486_48692

def molecular_weight (formula : String) : ℝ :=
  if formula = "C6H8O6" then 176.12 else 0

theorem ascorbic_acid_weight : molecular_weight "C6H8O6" = 176.12 :=
by {
  sorry
}

end ascorbic_acid_weight_l486_48692


namespace carla_initial_marbles_l486_48682

theorem carla_initial_marbles (total_marbles : ℕ) (bought_marbles : ℕ) (initial_marbles : ℕ) 
  (h1 : total_marbles = 187) (h2 : bought_marbles = 134) (h3 : total_marbles = initial_marbles + bought_marbles) : 
  initial_marbles = 53 := 
sorry

end carla_initial_marbles_l486_48682


namespace harmonious_division_condition_l486_48648

theorem harmonious_division_condition (a b c d e k : ℕ) (h : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e) (hk : 3 * k = a + b + c + d + e) (hk_pos : k > 0) :
  (∀ i j l : ℕ, i ≠ j ∧ j ≠ l ∧ i ≠ l → a ≤ k) ↔ (a ≤ k) :=
sorry

end harmonious_division_condition_l486_48648


namespace jordan_annual_income_l486_48601

theorem jordan_annual_income (q : ℝ) (I T : ℝ) 
  (h1 : T = q * 35000 + (q + 3) * (I - 35000))
  (h2 : T = (q + 0.4) * I) : 
  I = 40000 :=
by sorry

end jordan_annual_income_l486_48601


namespace mark_increase_reading_time_l486_48651

def initial_pages_per_day : ℕ := 100
def final_pages_per_week : ℕ := 1750
def days_in_week : ℕ := 7

def calculate_percentage_increase (initial_pages_per_day : ℕ) (final_pages_per_week : ℕ) (days_in_week : ℕ) : ℚ :=
  ((final_pages_per_week : ℚ) / ((initial_pages_per_day : ℚ) * (days_in_week : ℚ)) - 1) * 100

theorem mark_increase_reading_time :
  calculate_percentage_increase initial_pages_per_day final_pages_per_week days_in_week = 150 :=
by sorry

end mark_increase_reading_time_l486_48651


namespace largest_base8_3digit_to_base10_l486_48637

theorem largest_base8_3digit_to_base10 : (7 * 8^2 + 7 * 8^1 + 7 * 8^0) = 511 := by
  sorry

end largest_base8_3digit_to_base10_l486_48637


namespace length_of_shorter_piece_l486_48634

theorem length_of_shorter_piece (x : ℕ) (h1 : x + (x + 12) = 68) : x = 28 :=
by
  sorry

end length_of_shorter_piece_l486_48634


namespace new_mean_correct_l486_48689

-- Define the original condition data
def initial_mean : ℝ := 42
def total_numbers : ℕ := 60
def discard1 : ℝ := 50
def discard2 : ℝ := 60
def increment : ℝ := 2

-- A function representing the new arithmetic mean
noncomputable def new_arithmetic_mean : ℝ :=
  let initial_sum := initial_mean * total_numbers
  let sum_after_discard := initial_sum - (discard1 + discard2)
  let sum_after_increment := sum_after_discard + (increment * (total_numbers - 2))
  sum_after_increment / (total_numbers - 2)

-- The theorem statement
theorem new_mean_correct : new_arithmetic_mean = 43.55 :=
by 
  sorry

end new_mean_correct_l486_48689


namespace emails_received_l486_48681

variable (x y : ℕ)

theorem emails_received (h1 : 3 + 6 = 9) (h2 : x + y + 9 = 10) : x + y = 1 := by
  sorry

end emails_received_l486_48681


namespace smallest_three_digit_divisible_by_4_and_5_l486_48668

-- Define the problem conditions and goal as a Lean theorem statement
theorem smallest_three_digit_divisible_by_4_and_5 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 4 = 0 ∧ m % 5 = 0 → n ≤ m) :=
sorry

end smallest_three_digit_divisible_by_4_and_5_l486_48668


namespace max_ab_bc_cd_da_l486_48602

theorem max_ab_bc_cd_da (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h_sum : a + b + c + d = 200) :
  ab + bc + cd + da ≤ 10000 :=
by sorry

end max_ab_bc_cd_da_l486_48602


namespace triangle_side_lengths_relation_l486_48696

-- Given a triangle ABC with side lengths a, b, c
variables (a b c R d : ℝ)
-- Given orthocenter H and circumcenter O, and the radius of the circumcircle is R,
-- and distance between O and H is d.
-- Prove that a² + b² + c² = 9R² - d²

theorem triangle_side_lengths_relation (a b c R d : ℝ) (H O : Type) (orthocenter : H) (circumcenter : O)
  (radius_circumcircle : O → ℝ)
  (distance_OH : O → H → ℝ) :
  a^2 + b^2 + c^2 = 9 * R^2 - d^2 :=
sorry

end triangle_side_lengths_relation_l486_48696


namespace find_larger_number_l486_48663

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1375) (h2 : L = 6 * S + 15) : L = 1647 :=
by
  -- proof to be filled
  sorry

end find_larger_number_l486_48663


namespace real_solution_to_abs_equation_l486_48603

theorem real_solution_to_abs_equation :
  (∃! x : ℝ, |x - 2| = |x - 4| + |x - 6| + |x - 8|) :=
by
  sorry

end real_solution_to_abs_equation_l486_48603


namespace factor_t_squared_minus_81_l486_48673

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) :=
by
  sorry

end factor_t_squared_minus_81_l486_48673


namespace apple_pies_count_l486_48617

def total_pies := 13
def pecan_pies := 4
def pumpkin_pies := 7
def apple_pies := total_pies - pecan_pies - pumpkin_pies

theorem apple_pies_count : apple_pies = 2 := by
  sorry

end apple_pies_count_l486_48617


namespace triangle_cos_Z_l486_48654

theorem triangle_cos_Z (X Y Z : ℝ) (hXZ : X + Y + Z = π) 
  (sinX : Real.sin X = 4 / 5) (cosY : Real.cos Y = 3 / 5) : 
  Real.cos Z = 7 / 25 := 
sorry

end triangle_cos_Z_l486_48654


namespace line_intercepts_l486_48698

theorem line_intercepts :
  (exists a b : ℝ, (forall x y : ℝ, x - 2*y - 2 = 0 ↔ (x = 2 ∨ y = -1)) ∧ a = 2 ∧ b = -1) :=
by
  sorry

end line_intercepts_l486_48698


namespace triangle_perimeter_l486_48674

theorem triangle_perimeter (x : ℕ) 
  (h1 : x % 2 = 1) 
  (h2 : 7 - 2 < x)
  (h3 : x < 2 + 7) :
  2 + 7 + x = 16 := 
sorry

end triangle_perimeter_l486_48674


namespace sum_of_solutions_l486_48664

theorem sum_of_solutions (x : ℝ) (h : x^2 - 3 * x = 12) : x = 3 := by
  sorry

end sum_of_solutions_l486_48664


namespace sequence_properties_l486_48649

theorem sequence_properties (S : ℕ → ℝ) (a : ℕ → ℝ) :
  S 2 = 4 →
  (∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1) →
  a 1 = 1 ∧ S 5 = 121 :=
by
  intros hS2 ha
  sorry

end sequence_properties_l486_48649


namespace coords_of_point_P_l486_48612

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 2)

theorem coords_of_point_P :
  ∀ (a : ℝ), 0 < a ∧ a ≠ 1 → ∃ P : ℝ × ℝ, (P = (1, -2) ∧ ∀ y, f (f a (-2)) y = y) :=
by
  sorry

end coords_of_point_P_l486_48612


namespace largest_B_is_9_l486_48670

def is_divisible_by_three (n : ℕ) : Prop :=
  n % 3 = 0

def is_divisible_by_four (n : ℕ) : Prop :=
  n % 4 = 0

def largest_B_divisible_by_3_and_4 (B : ℕ) : Prop :=
  is_divisible_by_three (21 + B) ∧ is_divisible_by_four 32

theorem largest_B_is_9 : largest_B_divisible_by_3_and_4 9 :=
by
  have h1 : is_divisible_by_three (21 + 9) := by sorry
  have h2 : is_divisible_by_four 32 := by sorry
  exact ⟨h1, h2⟩

end largest_B_is_9_l486_48670


namespace range_of_m_for_distinct_real_roots_l486_48623

theorem range_of_m_for_distinct_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 4 * x1 - m = 0 ∧ x2^2 - 4 * x2 - m = 0) ↔ m > -4 :=
by
  sorry

end range_of_m_for_distinct_real_roots_l486_48623


namespace find_p_geometric_progression_l486_48607

theorem find_p_geometric_progression (p : ℝ) : 
  (p = -1 ∨ p = 40 / 9) ↔ ((9 * p + 10), (3 * p), |p - 8|) ∈ 
  {gp | ∃ r : ℝ, gp = (r, r * r, r * r * r)} :=
by sorry

end find_p_geometric_progression_l486_48607


namespace four_integers_product_sum_l486_48684

theorem four_integers_product_sum (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_prod : a * b * c * d = 2002) (h_sum : a + b + c + d < 40) :
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) :=
sorry

end four_integers_product_sum_l486_48684


namespace quarts_of_water_required_l486_48687

-- Define the ratio of water to juice
def ratio_water_to_juice : Nat := 5 / 3

-- Define the total punch to prepare in gallons
def total_punch_in_gallons : Nat := 2

-- Define the conversion factor from gallons to quarts
def quarts_per_gallon : Nat := 4

-- Define the total number of parts
def total_parts : Nat := 5 + 3

-- Define the total punch in quarts
def total_punch_in_quarts : Nat := total_punch_in_gallons * quarts_per_gallon

-- Define the amount of water per part
def quarts_per_part : Nat := total_punch_in_quarts / total_parts

-- Prove the required amount of water in quarts
theorem quarts_of_water_required : quarts_per_part * 5 = 5 := 
by
  -- Proof is omitted, represented by sorry
  sorry

end quarts_of_water_required_l486_48687


namespace average_of_three_l486_48699

-- Definitions of Conditions
variables (A B C : ℝ)
variables (h1 : A + B = 147) (h2 : B + C = 123) (h3 : A + C = 132)

-- The proof problem stating the goal
theorem average_of_three (A B C : ℝ) 
    (h1 : A + B = 147) (h2 : B + C = 123) (h3 : A + C = 132) : 
    (A + B + C) / 3 = 67 := 
sorry

end average_of_three_l486_48699


namespace unique_abc_solution_l486_48631

theorem unique_abc_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
    (h4 : a^4 + b^2 * c^2 = 16 * a) (h5 : b^4 + c^2 * a^2 = 16 * b) (h6 : c^4 + a^2 * b^2 = 16 * c) : 
    (a, b, c) = (2, 2, 2) :=
  by
    sorry

end unique_abc_solution_l486_48631


namespace part1_l486_48633

theorem part1 (m : ℝ) (a b : ℝ) (h : m > 0) : 
  ( (a + m * b) / (1 + m) )^2 ≤ (a^2 + m * b^2) / (1 + m) :=
sorry

end part1_l486_48633


namespace exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4_l486_48620

theorem exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4
  (n : ℕ) (hn₁ : Odd n) (hn₂ : 0 < n) :
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (4 : ℚ) / n = 1 / a + 1 / b) ↔
  ∃ p, p ∣ n ∧ Prime p ∧ p % 4 = 1 :=
by
  sorry

end exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4_l486_48620


namespace min_value_l486_48666

theorem min_value (a : ℝ) (h : a > 1) : a + 1 / (a - 1) ≥ 3 :=
sorry

end min_value_l486_48666


namespace interest_rate_increase_60_percent_l486_48659

noncomputable def percentage_increase (A P A' t : ℝ) : ℝ :=
  let r₁ := (A - P) / (P * t)
  let r₂ := (A' - P) / (P * t)
  ((r₂ - r₁) / r₁) * 100

theorem interest_rate_increase_60_percent :
  percentage_increase 920 800 992 3 = 60 := by
  sorry

end interest_rate_increase_60_percent_l486_48659


namespace counting_4digit_integers_l486_48630

theorem counting_4digit_integers (x y : ℕ) (a b c d : ℕ) :
  (x = 1000 * a + 100 * b + 10 * c + d) →
  (y = 1000 * d + 100 * c + 10 * b + a) →
  (y - x = 3177) →
  (1 ≤ a) → (a ≤ 6) →
  (0 ≤ b) → (b ≤ 7) →
  (c = b + 2) →
  (d = a + 3) →
  ∃ n : ℕ, n = 48 := 
sorry

end counting_4digit_integers_l486_48630


namespace range_of_b_l486_48671

theorem range_of_b {b : ℝ} (h_b_ne_zero : b ≠ 0) :
  (∃ x : ℝ, (0 ≤ x ∧ x ≤ 3) ∧ (2 * x + b = 3)) ↔ -3 ≤ b ∧ b ≤ 3 ∧ b ≠ 0 :=
by
  sorry

end range_of_b_l486_48671


namespace solve_x_if_alpha_beta_eq_8_l486_48618

variable (x : ℝ)

def alpha (x : ℝ) := 4 * x + 9
def beta (x : ℝ) := 9 * x + 6

theorem solve_x_if_alpha_beta_eq_8 (hx : alpha (beta x) = 8) : x = (-25 / 36) :=
by
  sorry

end solve_x_if_alpha_beta_eq_8_l486_48618


namespace total_cakes_served_l486_48609

-- Defining the values for cakes served during lunch and dinner
def lunch_cakes : ℤ := 6
def dinner_cakes : ℤ := 9

-- Stating the theorem that the total number of cakes served today is 15
theorem total_cakes_served : lunch_cakes + dinner_cakes = 15 :=
by
  sorry

end total_cakes_served_l486_48609


namespace arithmetic_series_sum_l486_48675

variable (a₁ aₙ d S : ℝ)
variable (n : ℕ)

-- Defining the conditions (a₁, aₙ, d, and the formula for arithmetic series sum)
def first_term : a₁ = 10 := sorry
def last_term : aₙ = 70 := sorry
def common_diff : d = 1 / 7 := sorry

-- Equation to find number of terms (n)
def find_n : 70 = 10 + (n - 1) * (1 / 7) := sorry

-- Formula for the sum of an arithmetic series
def series_sum : S = (n * (10 + 70)) / 2 := sorry

-- The proof problem statement
theorem arithmetic_series_sum : 
  a₁ = 10 → 
  aₙ = 70 → 
  d = 1 / 7 → 
  (70 = 10 + (n - 1) * (1 / 7)) → 
  S = (n * (10 + 70)) / 2 → 
  S = 16840 := by 
  intros h1 h2 h3 h4 h5 
  -- proof steps would go here
  sorry

end arithmetic_series_sum_l486_48675


namespace equation_solution_l486_48605

theorem equation_solution (x y : ℝ) (h : x^2 + (1 - y)^2 + (x - y)^2 = (1 / 3)) : 
  x = (1 / 3) ∧ y = (2 / 3) := 
  sorry

end equation_solution_l486_48605


namespace complement_intersection_l486_48691

-- Conditions
def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0}
def B : Set Int := {0, 1, 2}

-- Theorem statement (proof not included)
theorem complement_intersection :
  let C_UA : Set Int := U \ A
  (C_UA ∩ B) = {1, 2} := 
by
  sorry

end complement_intersection_l486_48691


namespace mechanic_worked_days_l486_48686

-- Definitions of conditions as variables
def hourly_rate : ℝ := 60
def hours_per_day : ℝ := 8
def cost_of_parts : ℝ := 2500
def total_amount_paid : ℝ := 9220

-- Definition to calculate the total labor cost
def total_labor_cost : ℝ := total_amount_paid - cost_of_parts

-- Definition to calculate the daily labor cost
def daily_labor_cost : ℝ := hourly_rate * hours_per_day

-- Proof (statement only) that the number of days the mechanic worked on the car is 14
theorem mechanic_worked_days : total_labor_cost / daily_labor_cost = 14 := by
  sorry

end mechanic_worked_days_l486_48686


namespace hyperbola_equation_l486_48655

variable (a b c : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (asymptote_cond : -b / a = -1 / 2)
variable (foci_cond : c = 5)
variable (hyperbola_rel : a^2 + b^2 = c^2)

theorem hyperbola_equation : 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ -b / a = -1 / 2 ∧ c = 5 ∧ a^2 + b^2 = c^2 
  ∧ ∀ x y : ℝ, (x^2 / 20 - y^2 / 5 = 1)) := 
sorry

end hyperbola_equation_l486_48655


namespace lunch_cost_before_tax_and_tip_l486_48678

theorem lunch_cost_before_tax_and_tip (C : ℝ) (h1 : 1.10 * C = 110) : C = 100 := by
  sorry

end lunch_cost_before_tax_and_tip_l486_48678


namespace smallest_divisor_after_391_l486_48616

theorem smallest_divisor_after_391 (m : ℕ) (h₁ : 1000 ≤ m ∧ m < 10000) (h₂ : Even m) (h₃ : 391 ∣ m) : 
  ∃ d, d > 391 ∧ d ∣ m ∧ ∀ e, 391 < e ∧ e ∣ m → e ≥ d :=
by
  use 441
  sorry

end smallest_divisor_after_391_l486_48616


namespace attendance_second_concert_l486_48625

-- Define the given conditions
def attendance_first_concert : ℕ := 65899
def additional_people : ℕ := 119

-- Prove the number of people at the second concert
theorem attendance_second_concert : 
  attendance_first_concert + additional_people = 66018 := 
by
  -- Placeholder for the proof
  sorry

end attendance_second_concert_l486_48625


namespace total_marbles_l486_48622

theorem total_marbles :
  let marbles_second_bowl := 600
  let marbles_first_bowl := (3/4) * marbles_second_bowl
  let total_marbles := marbles_first_bowl + marbles_second_bowl
  total_marbles = 1050 := by
  sorry -- proof skipped

end total_marbles_l486_48622


namespace binom_sum_l486_48652

theorem binom_sum : Nat.choose 18 4 + Nat.choose 5 2 = 3070 := 
by
  sorry

end binom_sum_l486_48652


namespace probability_of_sequence_HTHT_l486_48680

noncomputable def prob_sequence_HTHT : ℚ :=
  let p := 1 / 2
  (p * p * p * p)

theorem probability_of_sequence_HTHT :
  prob_sequence_HTHT = 1 / 16 := 
by
  sorry

end probability_of_sequence_HTHT_l486_48680


namespace omitted_angle_measure_l486_48662

theorem omitted_angle_measure (initial_sum correct_sum : ℝ) (H_initial : initial_sum = 2083) (H_correct : correct_sum = 2160) :
  correct_sum - initial_sum = 77 :=
by sorry

end omitted_angle_measure_l486_48662


namespace parabola_shift_right_l486_48621

theorem parabola_shift_right (x : ℝ) :
  let original_parabola := - (1 / 2) * x^2
  let shifted_parabola := - (1 / 2) * (x - 1)^2
  original_parabola = shifted_parabola :=
sorry

end parabola_shift_right_l486_48621


namespace adiabatic_compression_work_l486_48614

noncomputable def adiabatic_work (p1 V1 V2 k : ℝ) (h₁ : k > 1) (h₂ : V1 > 0) (h₃ : V2 > 0) : ℝ :=
  (p1 * V1) / (k - 1) * (1 - (V1 / V2)^(k - 1))

theorem adiabatic_compression_work (p1 V1 V2 k W : ℝ) (h₁ : k > 1) (h₂ : V1 > 0) (h₃ : V2 > 0) :
  W = adiabatic_work p1 V1 V2 k h₁ h₂ h₃ :=
sorry

end adiabatic_compression_work_l486_48614


namespace exists_h_l486_48685

noncomputable def F (x : ℝ) : ℝ := x^2 + 12 / x^2
noncomputable def G (x : ℝ) : ℝ := Real.sin (Real.pi * x^2)
noncomputable def H (x : ℝ) : ℝ := 1

theorem exists_h (h : ℝ → ℝ) (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 10) :
  |h x - x| < 1 / 3 :=
sorry

end exists_h_l486_48685


namespace first_plane_passengers_l486_48613

-- Definitions and conditions
def speed_plane_empty : ℕ := 600
def slowdown_per_passenger : ℕ := 2
def second_plane_passengers : ℕ := 60
def third_plane_passengers : ℕ := 40
def average_speed : ℕ := 500

-- Definition of the speed of a plane given number of passengers
def speed (passengers : ℕ) : ℕ := speed_plane_empty - slowdown_per_passenger * passengers

-- The problem statement rewritten in Lean 4
theorem first_plane_passengers (P : ℕ) (h_avg : (speed P + speed second_plane_passengers + speed third_plane_passengers) / 3 = average_speed) : P = 50 :=
sorry

end first_plane_passengers_l486_48613


namespace count_integers_with_same_remainder_l486_48688

theorem count_integers_with_same_remainder (n : ℤ) : 
  (150 < n ∧ n < 250) ∧ 
  (∃ r : ℤ, 0 ≤ r ∧ r ≤ 6 ∧ ∃ a b : ℤ, n = 7 * a + r ∧ n = 9 * b + r) ↔ n = 7 :=
sorry

end count_integers_with_same_remainder_l486_48688


namespace michael_remaining_yards_l486_48657

theorem michael_remaining_yards (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) (m y : ℕ)
  (h1 : miles_per_marathon = 50)
  (h2 : yards_per_marathon = 800)
  (h3 : yards_per_mile = 1760)
  (h4 : num_marathons = 5)
  (h5 : y = (yards_per_marathon * num_marathons) % yards_per_mile)
  (h6 : m = miles_per_marathon * num_marathons + (yards_per_marathon * num_marathons) / yards_per_mile) :
  y = 480 :=
sorry

end michael_remaining_yards_l486_48657


namespace max_gcd_b_eq_1_l486_48661

-- Define bn as bn = 2^n - 1 for natural number n
def b (n : ℕ) : ℕ := 2^n - 1

-- Define en as the greatest common divisor of bn and bn+1
def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

-- The theorem to prove:
theorem max_gcd_b_eq_1 (n : ℕ) : e n = 1 :=
  sorry

end max_gcd_b_eq_1_l486_48661


namespace fewer_parking_spaces_on_fourth_level_l486_48615

theorem fewer_parking_spaces_on_fourth_level 
  (spaces_first_level : ℕ) (spaces_second_level : ℕ) (spaces_third_level : ℕ) (spaces_fourth_level : ℕ) 
  (total_spaces_garage : ℕ) (cars_parked : ℕ) 
  (h1 : spaces_first_level = 90)
  (h2 : spaces_second_level = spaces_first_level + 8)
  (h3 : spaces_third_level = spaces_second_level + 12)
  (h4 : total_spaces_garage = 299)
  (h5 : cars_parked = 100)
  (h6 : spaces_first_level + spaces_second_level + spaces_third_level + spaces_fourth_level = total_spaces_garage) :
  spaces_third_level - spaces_fourth_level = 109 := 
by
  sorry

end fewer_parking_spaces_on_fourth_level_l486_48615


namespace find_a_l486_48647

theorem find_a (x y a : ℝ) (h1 : x + 3 * y = 4 - a) 
  (h2 : x - y = -3 * a) (h3 : x + y = 0) : a = 1 :=
sorry

end find_a_l486_48647


namespace sin_cos_eq_values_l486_48656

theorem sin_cos_eq_values (θ : ℝ) (hθ : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  (∃ t : ℝ, 
    0 < t ∧ 
    t ≤ 2 * Real.pi ∧ 
    (2 + 4 * Real.sin t - 3 * Real.cos (2 * t) = 0)) ↔ (∃ n : ℕ, n = 4) :=
by 
  sorry

end sin_cos_eq_values_l486_48656


namespace outfits_count_l486_48639

def num_outfits (redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats : ℕ) : ℕ :=
  (redShirts * pairsPants * (greenHats + blueHats)) +
  (greenShirts * pairsPants * (redHats + blueHats)) +
  (blueShirts * pairsPants * (redHats + greenHats))

theorem outfits_count :
  ∀ (redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats : ℕ),
  redShirts = 4 → greenShirts = 4 → blueShirts = 4 →
  pairsPants = 7 →
  greenHats = 6 → redHats = 6 → blueHats = 6 →
  num_outfits redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats = 1008 :=
by
  intros redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats
  intros hredShirts hgreenShirts hblueShirts hpairsPants hgreenHats hredHats hblueHats
  rw [hredShirts, hgreenShirts, hblueShirts, hpairsPants, hgreenHats, hredHats, hblueHats]
  sorry

end outfits_count_l486_48639


namespace root_of_linear_eq_l486_48619

variable (a b : ℚ) -- Using rationals for coefficients

-- Define the linear equation
def linear_eq (x : ℚ) : Prop := a * x + b = 0

-- Define the root function
def root_function : ℚ := -b / a

-- State the goal
theorem root_of_linear_eq : linear_eq a b (root_function a b) :=
by
  unfold linear_eq
  unfold root_function
  sorry

end root_of_linear_eq_l486_48619


namespace min_triangle_perimeter_proof_l486_48624

noncomputable def min_triangle_perimeter (l m n : ℕ) : ℕ :=
  if l > m ∧ m > n ∧ (3^l % 10000 = 3^m % 10000) ∧ (3^m % 10000 = 3^n % 10000) then
    l + m + n
  else
    0

theorem min_triangle_perimeter_proof : ∃ (l m n : ℕ), l > m ∧ m > n ∧ 
  (3^l % 10000 = 3^m % 10000) ∧
  (3^m % 10000 = 3^n % 10000) ∧ min_triangle_perimeter l m n = 3003 :=
  sorry

end min_triangle_perimeter_proof_l486_48624


namespace complex_modulus_l486_48643

noncomputable def z : ℂ := (1 + 3 * Complex.I) / (1 + Complex.I)

theorem complex_modulus 
  (h : (1 + Complex.I) * z = 1 + 3 * Complex.I) : 
  Complex.abs (z^2) = 5 := 
by
  sorry

end complex_modulus_l486_48643


namespace remainder_of_3n_mod_9_l486_48695

theorem remainder_of_3n_mod_9 (n : ℕ) (h : n % 9 = 7) : (3 * n) % 9 = 3 :=
by
  sorry

end remainder_of_3n_mod_9_l486_48695


namespace exists_pos_int_n_l486_48610

def sequence_x (x : ℕ → ℝ) : Prop :=
  ∀ n, x (n + 2) = x n + (x (n + 1))^2

def sequence_y (y : ℕ → ℝ) : Prop :=
  ∀ n, y (n + 2) = y n^2 + y (n + 1)

def positive_initial_conditions (x y : ℕ → ℝ) : Prop :=
  x 1 > 1 ∧ x 2 > 1 ∧ y 1 > 1 ∧ y 2 > 1

theorem exists_pos_int_n (x y : ℕ → ℝ) (hx : sequence_x x) (hy : sequence_y y) 
  (ini : positive_initial_conditions x y) : ∃ n, x n > y n := 
sorry

end exists_pos_int_n_l486_48610


namespace rectangular_field_area_l486_48690

theorem rectangular_field_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) :
  w * l = 243 :=
by
  -- Proof goes here
  sorry

end rectangular_field_area_l486_48690


namespace line_through_points_decreasing_direct_proportion_function_m_l486_48672

theorem line_through_points_decreasing (x₁ x₂ y₁ y₂ k b : ℝ) (h1 : x₁ < x₂) (h2 : y₁ = k * x₁ + b) (h3 : y₂ = k * x₂ + b) (h4 : k < 0) : y₁ > y₂ :=
sorry

theorem direct_proportion_function_m (x₁ x₂ y₁ y₂ m : ℝ) (h1 : x₁ < x₂) (h2 : y₁ = (1 - 2 * m) * x₁) (h3 : y₂ = (1 - 2 * m) * x₂) (h4 : y₁ > y₂) : m > 1/2 :=
sorry

end line_through_points_decreasing_direct_proportion_function_m_l486_48672


namespace train_speed_in_kmph_l486_48640

theorem train_speed_in_kmph (length_in_m : ℝ) (time_in_s : ℝ) (length_in_m_eq : length_in_m = 800.064) (time_in_s_eq : time_in_s = 18) : 
  (length_in_m / 1000) / (time_in_s / 3600) = 160.0128 :=
by
  rw [length_in_m_eq, time_in_s_eq]
  /-
  To convert length in meters to kilometers, divide by 1000.
  To convert time in seconds to hours, divide by 3600.
  The speed is then computed by dividing the converted length by the converted time.
  -/
  sorry

end train_speed_in_kmph_l486_48640


namespace fixed_point_of_f_l486_48600

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) + 4

theorem fixed_point_of_f (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) : f a 1 = 5 :=
by
  unfold f
  -- Skip the proof; it will be filled in the subsequent steps
  sorry

end fixed_point_of_f_l486_48600


namespace jones_elementary_school_students_l486_48635

theorem jones_elementary_school_students
  (X : ℕ)
  (boys_percent_total : ℚ)
  (num_students_represented : ℕ)
  (percent_of_boys : ℚ)
  (h1 : boys_percent_total = 0.60)
  (h2 : num_students_represented = 90)
  (h3 : percent_of_boys * (boys_percent_total * X) = 90)
  : X = 150 :=
by
  sorry

end jones_elementary_school_students_l486_48635


namespace find_value_of_expression_l486_48658

noncomputable def root_finder (a b c : ℝ) : Prop :=
  a^3 - 30*a^2 + 65*a - 42 = 0 ∧
  b^3 - 30*b^2 + 65*b - 42 = 0 ∧
  c^3 - 30*c^2 + 65*c - 42 = 0

theorem find_value_of_expression {a b c : ℝ} (h : root_finder a b c) :
  a + b + c = 30 ∧ ab + bc + ca = 65 ∧ abc = 42 → 
  (a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)) = 770/43 :=
by
  sorry

end find_value_of_expression_l486_48658


namespace percent_of_x_is_y_l486_48628

-- Given the condition
def condition (x y : ℝ) : Prop :=
  0.70 * (x - y) = 0.30 * (x + y)

-- Prove y / x = 0.40
theorem percent_of_x_is_y (x y : ℝ) (h : condition x y) : y / x = 0.40 :=
by
  sorry

end percent_of_x_is_y_l486_48628


namespace base9_perfect_square_l486_48679

theorem base9_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : ∃ k : ℕ, (729 * a + 81 * b + 36 + d) = k * k) :
    d = 0 ∨ d = 1 ∨ d = 4 ∨ d = 7 :=
sorry

end base9_perfect_square_l486_48679


namespace odd_f_even_g_fg_eq_g_increasing_min_g_sum_l486_48667

noncomputable def f (x : ℝ) : ℝ := (0.5) * ((2:ℝ)^x - (2:ℝ)^(-x))
noncomputable def g (x : ℝ) : ℝ := (0.5) * ((2:ℝ)^x + (2:ℝ)^(-x))

theorem odd_f (x : ℝ) : f (-x) = -f (x) := sorry
theorem even_g (x : ℝ) : g (-x) = g (x) := sorry
theorem fg_eq (x : ℝ) : f (x) + g (x) = (2:ℝ)^x := sorry
theorem g_increasing (x : ℝ) : x ≥ 0 → ∀ y, 0 ≤ y ∧ y < x → g y < g x := sorry
theorem min_g_sum (x : ℝ) : ∃ t, t ≥ 2 ∧ (g x + g (2 * x) = 2) := sorry

end odd_f_even_g_fg_eq_g_increasing_min_g_sum_l486_48667


namespace right_vs_oblique_prism_similarities_and_differences_l486_48608

-- Definitions of Prisms and their properties
structure Prism where
  parallel_bases : Prop
  congruent_bases : Prop
  parallelogram_faces : Prop

structure RightPrism extends Prism where
  rectangular_faces : Prop
  perpendicular_sides : Prop

structure ObliquePrism extends Prism where
  non_perpendicular_sides : Prop

theorem right_vs_oblique_prism_similarities_and_differences 
  (p1 : RightPrism) (p2 : ObliquePrism) : 
    (p1.parallel_bases ↔ p2.parallel_bases) ∧ 
    (p1.congruent_bases ↔ p2.congruent_bases) ∧ 
    (p1.parallelogram_faces ↔ p2.parallelogram_faces) ∧
    (p1.rectangular_faces ∧ p1.perpendicular_sides ↔ p2.non_perpendicular_sides) := 
by 
  sorry

end right_vs_oblique_prism_similarities_and_differences_l486_48608


namespace smallest_n_for_sqrt_20n_int_l486_48646

theorem smallest_n_for_sqrt_20n_int (n : ℕ) (h : ∃ k : ℕ, 20 * n = k^2) : n = 5 :=
by sorry

end smallest_n_for_sqrt_20n_int_l486_48646


namespace average_weight_correct_l486_48645

-- Define the number of men and women
def number_of_men : ℕ := 8
def number_of_women : ℕ := 6

-- Define the average weights of men and women
def average_weight_men : ℕ := 190
def average_weight_women : ℕ := 120

-- Define the total weight of men and women
def total_weight_men : ℕ := number_of_men * average_weight_men
def total_weight_women : ℕ := number_of_women * average_weight_women

-- Define the total number of individuals
def total_individuals : ℕ := number_of_men + number_of_women

-- Define the combined total weight
def total_weight : ℕ := total_weight_men + total_weight_women

-- Define the average weight of all individuals
def average_weight_all : ℕ := total_weight / total_individuals

theorem average_weight_correct :
  average_weight_all = 160 :=
  by sorry

end average_weight_correct_l486_48645


namespace min_soldiers_in_square_formations_l486_48693

theorem min_soldiers_in_square_formations : ∃ (a : ℕ), 
  ∃ (k : ℕ), 
    (a = k^2 ∧ 
    11 * a + 1 = (m : ℕ) ^ 2) ∧ 
    (∀ (b : ℕ), 
      (∃ (j : ℕ), b = j^2 ∧ 11 * b + 1 = (n : ℕ) ^ 2) → a ≤ b) ∧ 
    a = 9 := 
sorry

end min_soldiers_in_square_formations_l486_48693


namespace determinant_computation_l486_48636

variable (x y z w : ℝ)
variable (det : ℝ)
variable (H : x * w - y * z = 7)

theorem determinant_computation : 
  (x + z) * w - (y + 2 * w) * z = 7 - w * z := by
  sorry

end determinant_computation_l486_48636


namespace fixed_point_of_invariant_line_l486_48665

theorem fixed_point_of_invariant_line :
  ∀ (m : ℝ) (x y : ℝ), (3 * m + 4) * x + (5 - 2 * m) * y + 7 * m - 6 = 0 →
  (x = -1 ∧ y = 2) :=
by
  intro m x y h
  sorry

end fixed_point_of_invariant_line_l486_48665


namespace particle_motion_inverse_relationship_l486_48632

theorem particle_motion_inverse_relationship 
  {k : ℝ} 
  (inverse_relationship : ∀ {n : ℕ}, ∃ t_n d_n, d_n = k / t_n)
  (second_mile : ∃ t_2 d_2, t_2 = 2 ∧ d_2 = 1) : 
  ∃ t_4 d_4, t_4 = 4 ∧ d_4 = 0.5 :=
by
  sorry

end particle_motion_inverse_relationship_l486_48632


namespace remaining_stock_weight_l486_48604

def green_beans_weight : ℕ := 80
def rice_weight : ℕ := green_beans_weight - 30
def sugar_weight : ℕ := green_beans_weight - 20
def flour_weight : ℕ := 2 * sugar_weight
def lentils_weight : ℕ := flour_weight - 10

def rice_remaining_weight : ℕ := rice_weight - rice_weight / 3
def sugar_remaining_weight : ℕ := sugar_weight - sugar_weight / 5
def flour_remaining_weight : ℕ := flour_weight - flour_weight / 4
def lentils_remaining_weight : ℕ := lentils_weight - lentils_weight / 6

def total_remaining_weight : ℕ :=
  rice_remaining_weight + sugar_remaining_weight + flour_remaining_weight + lentils_remaining_weight + green_beans_weight

theorem remaining_stock_weight :
  total_remaining_weight = 343 := by
  sorry

end remaining_stock_weight_l486_48604


namespace largest_power_of_three_dividing_A_l486_48641

theorem largest_power_of_three_dividing_A (A : ℕ)
  (h1 : ∃ (factors : List ℕ), (∀ b ∈ factors, b > 0) ∧ factors.sum = 2011 ∧ factors.prod = A)
  : ∃ k : ℕ, 3^k ∣ A ∧ ∀ m : ℕ, 3^m ∣ A → m ≤ 669 :=
by
  sorry

end largest_power_of_three_dividing_A_l486_48641


namespace least_n_condition_l486_48694

theorem least_n_condition (n : ℕ) (h1 : ∀ k : ℕ, 1 ≤ k → k ≤ n + 1 → (k ∣ n * (n - 1) → k ≠ n + 1)) : n = 4 :=
sorry

end least_n_condition_l486_48694


namespace skating_speeds_ratio_l486_48638

theorem skating_speeds_ratio (v_s v_f : ℝ) (h1 : v_f > v_s) (h2 : |v_f + v_s| / |v_f - v_s| = 5) :
  v_f / v_s = 3 / 2 :=
by
  sorry

end skating_speeds_ratio_l486_48638


namespace maximize_GDP_growth_l486_48676

def projectA_investment : ℕ := 20  -- million yuan
def projectB_investment : ℕ := 10  -- million yuan

def total_investment (a b : ℕ) : ℕ := a + b
def total_electricity (a b : ℕ) : ℕ := 20000 * a + 40000 * b
def total_jobs (a b : ℕ) : ℕ := 24 * a + 36 * b
def total_GDP_increase (a b : ℕ) : ℕ := 26 * a + 20 * b  -- scaled by 10 to avoid decimals

theorem maximize_GDP_growth : 
  total_investment projectA_investment projectB_investment ≤ 30 ∧
  total_electricity projectA_investment projectB_investment ≤ 1000000 ∧
  total_jobs projectA_investment projectB_investment ≥ 840 → 
  total_GDP_increase projectA_investment projectB_investment = 860 := 
by
  -- Proof would be provided here
  sorry

end maximize_GDP_growth_l486_48676


namespace birdhouse_price_l486_48677

theorem birdhouse_price (S : ℤ) : 
  (2 * 22) + (2 * 16) + (3 * S) = 97 → 
  S = 7 :=
by
  sorry

end birdhouse_price_l486_48677


namespace original_number_is_seven_l486_48611

theorem original_number_is_seven (x : ℤ) (h : 3 * x - 6 = 15) : x = 7 :=
by
  sorry

end original_number_is_seven_l486_48611


namespace probability_of_matching_colors_l486_48642

theorem probability_of_matching_colors :
  let abe_jelly_beans := ["green", "red", "blue"]
  let bob_jelly_beans := ["green", "green", "yellow", "yellow", "red", "red", "red"]
  let abe_probs := (1 / 3, 1 / 3, 1 / 3)
  let bob_probs := (2 / 7, 3 / 7, 0)
  let matching_prob := (1 / 3 * 2 / 7) + (1 / 3 * 3 / 7)
  matching_prob = 5 / 21 := by sorry

end probability_of_matching_colors_l486_48642


namespace coin_overlap_black_region_cd_sum_l486_48697

noncomputable def black_region_probability : ℝ := 
  let square_side := 10
  let triangle_leg := 3
  let diamond_side := 3 * Real.sqrt 2
  let coin_diameter := 2
  let coin_radius := coin_diameter / 2
  let reduced_square_side := square_side - coin_diameter
  let reduced_square_area := reduced_square_side * reduced_square_side
  let triangle_area := 4 * ((triangle_leg * triangle_leg) / 2)
  let extra_triangle_area := 4 * (Real.pi / 4 + 3)
  let diamond_area := (diamond_side * diamond_side) / 2
  let extra_diamond_area := Real.pi + 12 * Real.sqrt 2
  let total_black_area := triangle_area + extra_triangle_area + diamond_area + extra_diamond_area

  total_black_area / reduced_square_area

theorem coin_overlap_black_region: 
  black_region_probability = (1 / 64) * (30 + 12 * Real.sqrt 2 + Real.pi) := 
sorry

theorem cd_sum: 
  let c := 30
  let d := 12
  c + d = 42 := 
by
  trivial

end coin_overlap_black_region_cd_sum_l486_48697


namespace Jordan_Lee_debt_equal_l486_48683

theorem Jordan_Lee_debt_equal (initial_debt_jordan : ℝ) (additional_debt_jordan : ℝ)
  (rate_jordan : ℝ) (initial_debt_lee : ℝ) (rate_lee : ℝ) :
  initial_debt_jordan + additional_debt_jordan + (initial_debt_jordan + additional_debt_jordan) * rate_jordan * 33.333333333333336 
  = initial_debt_lee + initial_debt_lee * rate_lee * 33.333333333333336 :=
by
  let t := 33.333333333333336
  have rate_jordan := 0.12
  have rate_lee := 0.08
  have initial_debt_jordan := 200
  have additional_debt_jordan := 20
  have initial_debt_lee := 300
  sorry

end Jordan_Lee_debt_equal_l486_48683


namespace water_left_ratio_l486_48650

theorem water_left_ratio (h1: 2 * (30 / 10) = 6)
                        (h2: 2 * (30 / 10) = 6)
                        (h3: 4 * (60 / 10) = 24)
                        (water_left: ℕ)
                        (total_water_collected: ℕ) 
                        (h4: water_left = 18)
                        (h5: total_water_collected = 36) : 
  water_left * 2 = total_water_collected :=
by
  sorry

end water_left_ratio_l486_48650
