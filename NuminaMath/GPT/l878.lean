import Mathlib

namespace NUMINAMATH_GPT_smallest_positive_period_l878_87892

noncomputable def tan_period (a b x : ℝ) : ℝ := 
  Real.tan ((a + b) * x / 2)

theorem smallest_positive_period 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ p > 0, ∀ x, tan_period a b (x + p) = tan_period a b x ∧ p = 2 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_l878_87892


namespace NUMINAMATH_GPT_ratio_a_b_equals_sqrt2_l878_87877

variable (A B C a b c : ℝ) -- Define the variables representing the angles and sides.

-- Assuming the sides a, b, c are positive and a triangle is formed (non-degenerate)
axiom triangle_ABC : 0 < a ∧ 0 < b ∧ 0 < c

-- Assuming the sum of the angles in a triangle equals 180 degrees (π radians)
axiom sum_angles_triangle : A + B + C = Real.pi

-- Given condition
axiom given_condition : b * Real.cos C + c * Real.cos B = Real.sqrt 2 * b

-- Problem statement to be proven
theorem ratio_a_b_equals_sqrt2 : (a / b) = Real.sqrt 2 :=
by
  -- Assume the problem statement is correct
  sorry

end NUMINAMATH_GPT_ratio_a_b_equals_sqrt2_l878_87877


namespace NUMINAMATH_GPT_paul_initial_savings_l878_87846

theorem paul_initial_savings (additional_allowance: ℕ) (cost_per_toy: ℕ) (number_of_toys: ℕ) (total_savings: ℕ) :
  additional_allowance = 7 →
  cost_per_toy = 5 →
  number_of_toys = 2 →
  total_savings + additional_allowance = cost_per_toy * number_of_toys →
  total_savings = 3 :=
by
  intros h_additional h_cost h_number h_total
  sorry

end NUMINAMATH_GPT_paul_initial_savings_l878_87846


namespace NUMINAMATH_GPT_edge_of_new_cube_l878_87800

theorem edge_of_new_cube (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) :
  ∃ d : ℝ, d^3 = a^3 + b^3 + c^3 ∧ d = 12 :=
by
  sorry

end NUMINAMATH_GPT_edge_of_new_cube_l878_87800


namespace NUMINAMATH_GPT_sum_of_angles_l878_87807

namespace BridgeProblem

def is_isosceles (A B C : Type) (AB AC : ℝ) : Prop := AB = AC

def angle_bac (A B C : Type) : ℝ := 15

def angle_edf (D E F : Type) : ℝ := 45

theorem sum_of_angles (A B C D E F : Type) 
  (h_isosceles_ABC : is_isosceles A B C 1 1)
  (h_isosceles_DEF : is_isosceles D E F 1 1)
  (h_angle_BAC : angle_bac A B C = 15)
  (h_angle_EDF : angle_edf D E F = 45) :
  true := 
by 
  sorry

end BridgeProblem

end NUMINAMATH_GPT_sum_of_angles_l878_87807


namespace NUMINAMATH_GPT_min_value_fraction_l878_87863

open Real

theorem min_value_fraction (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (∃ x, x = (a + b) / (a * b * c) ∧ x = 16 / 9) :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l878_87863


namespace NUMINAMATH_GPT_problem_l878_87839

theorem problem : 
  let N := 63745.2981
  let place_value_7 := 1000 -- The place value of the digit 7 (thousands place)
  let place_value_2 := 0.1 -- The place value of the digit 2 (tenths place)
  place_value_7 / place_value_2 = 10000 :=
by
  sorry

end NUMINAMATH_GPT_problem_l878_87839


namespace NUMINAMATH_GPT_xyz_value_l878_87856

theorem xyz_value
  (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19)
  (h3 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = 11) :
  x * y * z = 26 / 3 :=
sorry

end NUMINAMATH_GPT_xyz_value_l878_87856


namespace NUMINAMATH_GPT_range_of_m_l878_87836

theorem range_of_m (x m : ℝ) (h1 : -2 ≤ 1 - (x-1)/3 ∧ 1 - (x-1)/3 ≤ 2)
                   (h2 : x^2 - 2*x + 1 - m^2 ≤ 0)
                   (h3 : m > 0)
                   (h4 : (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m))
                   (h5 : ¬((x < 1 - m ∨ x > 1 + m) → (x < -2 ∨ x > 10))) :
                   m ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l878_87836


namespace NUMINAMATH_GPT_length_of_platform_l878_87867

theorem length_of_platform (v t_m t_p L_t L_p : ℝ)
    (h1 : v = 33.3333333)
    (h2 : t_m = 22)
    (h3 : t_p = 45)
    (h4 : L_t = v * t_m)
    (h5 : L_t + L_p = v * t_p) :
    L_p = 766.666666 :=
by
  sorry

end NUMINAMATH_GPT_length_of_platform_l878_87867


namespace NUMINAMATH_GPT_cos_neg_300_eq_positive_half_l878_87872

theorem cos_neg_300_eq_positive_half : Real.cos (-300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_neg_300_eq_positive_half_l878_87872


namespace NUMINAMATH_GPT_sum_of_cubes_l878_87866

theorem sum_of_cubes (x y z : ℝ) (h1 : x + y + z = 7) (h2 : xy + xz + yz = 9) (h3 : xyz = -18) :
  x^3 + y^3 + z^3 = 100 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l878_87866


namespace NUMINAMATH_GPT_find_divisor_of_x_l878_87804

theorem find_divisor_of_x (x : ℕ) (q p : ℕ) (h1 : x % n = 5) (h2 : 4 * x % n = 2) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_of_x_l878_87804


namespace NUMINAMATH_GPT_chairs_made_after_tables_l878_87842

def pieces_of_wood : Nat := 672
def wood_per_table : Nat := 12
def wood_per_chair : Nat := 8
def number_of_tables : Nat := 24

theorem chairs_made_after_tables (pieces_of_wood wood_per_table wood_per_chair number_of_tables : Nat) :
  wood_per_table * number_of_tables <= pieces_of_wood ->
  (pieces_of_wood - wood_per_table * number_of_tables) / wood_per_chair = 48 :=
by
  sorry

end NUMINAMATH_GPT_chairs_made_after_tables_l878_87842


namespace NUMINAMATH_GPT_factor_expression_l878_87857

theorem factor_expression (b : ℝ) : 45 * b^2 + 135 * b^3 = 45 * b^2 * (1 + 3 * b) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l878_87857


namespace NUMINAMATH_GPT_propositions_correct_l878_87849

def vertical_angles (α β : ℝ) : Prop := ∃ γ, α = γ ∧ β = γ

def problem_statement : Prop :=
  (∀ α β, vertical_angles α β → α = β) ∧
  ¬(∀ α β, α = β → vertical_angles α β) ∧
  ¬(∀ α β, ¬vertical_angles α β → ¬(α = β)) ∧
  (∀ α β, ¬(α = β) → ¬vertical_angles α β)

theorem propositions_correct :
  problem_statement :=
by
  sorry

end NUMINAMATH_GPT_propositions_correct_l878_87849


namespace NUMINAMATH_GPT_math_proof_problem_l878_87858

-- Definitions for conditions:
def condition1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 2) = -f x
def condition2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x - 3 / 4) = -f (- (x - 3 / 4))

-- Statements to prove:
def statement1 (f : ℝ → ℝ) : Prop := ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x
def statement2 (f : ℝ → ℝ) : Prop := ∀ x, f (-(3 / 4) - x) = f (-(3 / 4) + x)
def statement3 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def statement4 (f : ℝ → ℝ) : Prop := ¬(∀ x y : ℝ, x < y → f x ≤ f y)

theorem math_proof_problem (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) :
  statement1 f ∧ statement2 f ∧ statement3 f ∧ statement4 f :=
by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l878_87858


namespace NUMINAMATH_GPT_interior_angle_ratio_l878_87894

theorem interior_angle_ratio (exterior_angle1 exterior_angle2 exterior_angle3 : ℝ)
  (h_ratio : 3 * exterior_angle1 = 4 * exterior_angle2 ∧ 
             4 * exterior_angle1 = 5 * exterior_angle3 ∧ 
             3 * exterior_angle1 + 4 * exterior_angle2 + 5 * exterior_angle3 = 360 ) : 
  3 * (180 - exterior_angle1) = 2 * (180 - exterior_angle2) ∧ 
  2 * (180 - exterior_angle2) = 1 * (180 - exterior_angle3) :=
sorry

end NUMINAMATH_GPT_interior_angle_ratio_l878_87894


namespace NUMINAMATH_GPT_theta_plus_2phi_eq_pi_div_4_l878_87855

noncomputable def theta (θ : ℝ) (φ : ℝ) : Prop := 
  ((Real.tan θ = 5 / 12) ∧ 
   (Real.sin φ = 1 / 2) ∧ 
   (0 < θ ∧ θ < Real.pi / 2) ∧ 
   (0 < φ ∧ φ < Real.pi / 2)  )

theorem theta_plus_2phi_eq_pi_div_4 (θ φ : ℝ) (h : theta θ φ) : 
    θ + 2 * φ = Real.pi / 4 :=
by 
  sorry

end NUMINAMATH_GPT_theta_plus_2phi_eq_pi_div_4_l878_87855


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_geometric_sequence_l878_87865

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * a (n - 1) / a n 

theorem necessary_but_not_sufficient_condition_geometric_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2) :
  (is_geometric_sequence a → (∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2)) ∧ (∃ b : ℕ → ℝ, (b n = 0 ∨ b n = b (n - 1) ∨ b n = b (n + 1)) ∧ ¬ is_geometric_sequence b) := 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_geometric_sequence_l878_87865


namespace NUMINAMATH_GPT_ellipse_properties_l878_87835

noncomputable def standard_equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2) / 4 + y^2 = 1

noncomputable def trajectory_equation_midpoint (x y : ℝ) : Prop :=
  ((2 * x - 1)^2) / 4 + (2 * y - 1 / 2)^2 = 1

theorem ellipse_properties :
  (∀ x y : ℝ, standard_equation_of_ellipse x y) ∧
  (∀ x y : ℝ, trajectory_equation_midpoint x y) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_properties_l878_87835


namespace NUMINAMATH_GPT_expression_square_l878_87829

theorem expression_square (a b c d : ℝ) :
  (2*a + b + 2*c - d)^2 - (3*a + 2*b + 3*c - 2*d)^2 - (4*a + 3*b + 4*c - 3*d)^2 + (5*a + 4*b + 5*c - 4*d)^2 =
  (2*(a + b + c - d))^2 := 
sorry

end NUMINAMATH_GPT_expression_square_l878_87829


namespace NUMINAMATH_GPT_chocolate_bars_produced_per_minute_l878_87871

theorem chocolate_bars_produced_per_minute
  (sugar_per_bar : ℝ)
  (total_sugar : ℝ)
  (time_in_minutes : ℝ) 
  (bars_per_min : ℝ) :
  sugar_per_bar = 1.5 →
  total_sugar = 108 →
  time_in_minutes = 2 →
  bars_per_min = 36 :=
sorry

end NUMINAMATH_GPT_chocolate_bars_produced_per_minute_l878_87871


namespace NUMINAMATH_GPT_simplify_expression_l878_87896

variable (y : ℝ)

theorem simplify_expression : 
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + 5 * y ^ 9 + 3 * y ^ 8) = 
  15 * y ^ 13 - y ^ 12 + 3 * y ^ 11 + 15 * y ^ 10 - y ^ 9 - 6 * y ^ 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l878_87896


namespace NUMINAMATH_GPT_complex_number_condition_l878_87812

theorem complex_number_condition (z : ℂ) (h : z^2 + z + 1 = 0) :
  2 * z^96 + 3 * z^97 + 4 * z^98 + 5 * z^99 + 6 * z^100 = 3 + 5 * z := 
by 
  sorry

end NUMINAMATH_GPT_complex_number_condition_l878_87812


namespace NUMINAMATH_GPT_cryptarithm_solved_l878_87810

-- Definitions for the digits A, B, C
def valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 10

-- Given conditions, where A, B, C are distinct non-zero digits
def conditions (A B C : ℕ) : Prop :=
  valid_digit A ∧ valid_digit B ∧ valid_digit C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C

-- Definitions of the two-digit and three-digit numbers
def two_digit (A B : ℕ) : ℕ := 10 * A + B
def three_digit_rep (C : ℕ) : ℕ := 111 * C

-- Main statement of the proof problem
theorem cryptarithm_solved (A B C : ℕ) (h : conditions A B C) :
  two_digit A B + A * three_digit_rep C = 247 → A * 100 + B * 10 + C = 251 :=
sorry -- Proof goes here

end NUMINAMATH_GPT_cryptarithm_solved_l878_87810


namespace NUMINAMATH_GPT_value_of_a_l878_87850

noncomputable def A : Set ℝ := { x | abs x = 1 }
def B (a : ℝ) : Set ℝ := { x | a * x = 1 }
def is_superset (A B : Set ℝ) : Prop := ∀ x, x ∈ B → x ∈ A

theorem value_of_a (a : ℝ) (h : is_superset A (B a)) : a = 1 ∨ a = 0 ∨ a = -1 :=
  sorry

end NUMINAMATH_GPT_value_of_a_l878_87850


namespace NUMINAMATH_GPT_monomial_properties_l878_87879

noncomputable def monomial_coeff : ℚ := -(3/5 : ℚ)

def monomial_degree (x y : ℤ) : ℕ :=
  1 + 2

theorem monomial_properties (x y : ℤ) :
  monomial_coeff = -(3/5) ∧ monomial_degree x y = 3 :=
by
  -- Proof is to be filled here
  sorry

end NUMINAMATH_GPT_monomial_properties_l878_87879


namespace NUMINAMATH_GPT_new_sales_volume_monthly_profit_maximize_profit_l878_87868

-- Define assumptions and variables
variables (x : ℝ) (p : ℝ) (v : ℝ) (profit : ℝ)

-- Part 1: New sales volume after price increase
theorem new_sales_volume (h : 0 < x ∧ x < 20) : v = 600 - 10 * x :=
sorry

-- Part 2: Price and quantity for a monthly profit of 10,000 yuan
theorem monthly_profit (h : profit = (40 + x - 30) * (600 - 10 * x)) (h2: profit = 10000) : p = 50 ∧ v = 500 :=
sorry

-- Part 3: Price for maximizing monthly sales profit
theorem maximize_profit (h : profit = (40 + x - 30) * (600 - 10 * x)) : (∃ x_max: ℝ, x_max < 20 ∧ ∀ x, x < 20 → profit ≤ -10 * (x - 25)^2 + 12250 ∧ p = 59 ∧ profit = 11890) :=
sorry

end NUMINAMATH_GPT_new_sales_volume_monthly_profit_maximize_profit_l878_87868


namespace NUMINAMATH_GPT_cistern_fill_time_l878_87825

theorem cistern_fill_time (F E : ℝ) (hF : F = 1 / 7) (hE : E = 1 / 9) : (1 / (F - E)) = 31.5 :=
by
  sorry

end NUMINAMATH_GPT_cistern_fill_time_l878_87825


namespace NUMINAMATH_GPT_estimate_contestants_l878_87822

theorem estimate_contestants :
  let total_contestants := 679
  let median_all_three := 188
  let median_two_tests := 159
  let median_one_test := 169
  total_contestants = 679 ∧
  median_all_three = 188 ∧
  median_two_tests = 159 ∧
  median_one_test = 169 →
  let approx_two_tests_per_pair := median_two_tests / 3
  let intersection_pairs_approx := approx_two_tests_per_pair + median_all_three
  let number_above_or_equal_median :=
    median_one_test + median_one_test + median_one_test -
    intersection_pairs_approx - intersection_pairs_approx - intersection_pairs_approx +
    median_all_three
  number_above_or_equal_median = 516 :=
by
  intros
  sorry

end NUMINAMATH_GPT_estimate_contestants_l878_87822


namespace NUMINAMATH_GPT_percent_decrease_l878_87862

theorem percent_decrease(call_cost_1980 call_cost_2010 : ℝ) (h₁ : call_cost_1980 = 50) (h₂ : call_cost_2010 = 5) :
  ((call_cost_1980 - call_cost_2010) / call_cost_1980 * 100) = 90 :=
by
  sorry

end NUMINAMATH_GPT_percent_decrease_l878_87862


namespace NUMINAMATH_GPT_supplement_greater_than_complement_l878_87870

variable (angle1 : ℝ)

def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

theorem supplement_greater_than_complement (h : is_acute angle1) :
  180 - angle1 = 90 + (90 - angle1) :=
by {
  sorry
}

end NUMINAMATH_GPT_supplement_greater_than_complement_l878_87870


namespace NUMINAMATH_GPT_sum_of_solutions_eq_35_over_3_l878_87876

theorem sum_of_solutions_eq_35_over_3 (a b : ℝ) 
  (h1 : 2 * a + b = 14) (h2 : a + 2 * b = 21) : 
  a + b = 35 / 3 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_35_over_3_l878_87876


namespace NUMINAMATH_GPT_average_weight_l878_87881

theorem average_weight {w : ℝ} 
  (h1 : 62 < w) 
  (h2 : w < 72) 
  (h3 : 60 < w) 
  (h4 : w < 70) 
  (h5 : w ≤ 65) : w = 63.5 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_l878_87881


namespace NUMINAMATH_GPT_inf_arith_seq_contains_inf_geo_seq_l878_87888

-- Condition: Infinite arithmetic sequence of natural numbers
variable (a d : ℕ) (h : ∀ n : ℕ, n ≥ 1 → ∃ k : ℕ, k = a + (n - 1) * d)

-- Theorem: There exists an infinite geometric sequence within the arithmetic sequence
theorem inf_arith_seq_contains_inf_geo_seq :
  ∃ r : ℕ, ∀ n : ℕ, ∃ k : ℕ, k = a * r ^ (n - 1) := sorry

end NUMINAMATH_GPT_inf_arith_seq_contains_inf_geo_seq_l878_87888


namespace NUMINAMATH_GPT_true_proposition_l878_87884

-- Define the propositions p and q
def p : Prop := ∃ x0 : ℝ, x0 ^ 2 - x0 + 1 ≥ 0

def q : Prop := ∀ (a b : ℝ), a < b → 1 / a > 1 / b

-- Prove that p ∧ ¬q is true
theorem true_proposition : p ∧ ¬q :=
by
  sorry

end NUMINAMATH_GPT_true_proposition_l878_87884


namespace NUMINAMATH_GPT_equation1_solutions_equation2_solutions_l878_87832

theorem equation1_solutions (x : ℝ) :
  (4 * x^2 = 12 * x) ↔ (x = 0 ∨ x = 3) := by
sorry

theorem equation2_solutions (x : ℝ) :
  ((3 / 4) * x^2 - 2 * x - (1 / 2) = 0) ↔ (x = (4 + Real.sqrt 22) / 3 ∨ x = (4 - Real.sqrt 22) / 3) := by
sorry

end NUMINAMATH_GPT_equation1_solutions_equation2_solutions_l878_87832


namespace NUMINAMATH_GPT_isabella_babysits_afternoons_per_week_l878_87831

-- Defining the conditions of Isabella's babysitting job
def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 5
def days_per_week (weeks : ℕ) (total_earnings : ℕ) : ℕ := total_earnings / (weeks * (hourly_rate * hours_per_day))

-- Total earnings after 7 weeks
def total_earnings : ℕ := 1050
def weeks : ℕ := 7

-- State the theorem
theorem isabella_babysits_afternoons_per_week :
  days_per_week weeks total_earnings = 6 :=
by
  sorry

end NUMINAMATH_GPT_isabella_babysits_afternoons_per_week_l878_87831


namespace NUMINAMATH_GPT_image_digit_sum_l878_87853

theorem image_digit_sum 
  (cat chicken crab bear goat: ℕ)
  (h1 : 5 * crab = 10)
  (h2 : 4 * crab + goat = 11)
  (h3 : 2 * goat + crab + 2 * bear = 16)
  (h4 : cat + bear + 2 * goat + crab = 13)
  (h5 : 2 * crab + 2 * chicken + goat = 17) :
  cat = 1 ∧ chicken = 5 ∧ crab = 2 ∧ bear = 4 ∧ goat = 3 := by
  sorry

end NUMINAMATH_GPT_image_digit_sum_l878_87853


namespace NUMINAMATH_GPT_people_visited_neither_l878_87834

-- Definitions based on conditions
def total_people : ℕ := 60
def visited_iceland : ℕ := 35
def visited_norway : ℕ := 23
def visited_both : ℕ := 31

-- Theorem statement
theorem people_visited_neither :
  total_people - (visited_iceland + visited_norway - visited_both) = 33 :=
by sorry

end NUMINAMATH_GPT_people_visited_neither_l878_87834


namespace NUMINAMATH_GPT_sum_of_three_numbers_l878_87875

theorem sum_of_three_numbers (a b c : ℤ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : a + 15 = (a + b + c) / 3) (h4 : (a + b + c) / 3 = c - 20) (h5 : b = 7) :
  a + b + c = 36 :=
sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l878_87875


namespace NUMINAMATH_GPT_value_of_n_l878_87803

theorem value_of_n (a : ℝ) (n : ℕ) (h : ∃ (k : ℕ), (n - 2 * k = 0) ∧ (k = 4)) : n = 8 :=
sorry

end NUMINAMATH_GPT_value_of_n_l878_87803


namespace NUMINAMATH_GPT_total_percent_decrease_baseball_card_l878_87852

theorem total_percent_decrease_baseball_card
  (original_value : ℝ)
  (first_year_decrease : ℝ := 0.20)
  (second_year_decrease : ℝ := 0.30)
  (value_after_first_year : ℝ := original_value * (1 - first_year_decrease))
  (final_value : ℝ := value_after_first_year * (1 - second_year_decrease))
  (total_percent_decrease : ℝ := ((original_value - final_value) / original_value) * 100) :
  total_percent_decrease = 44 :=
by 
  sorry

end NUMINAMATH_GPT_total_percent_decrease_baseball_card_l878_87852


namespace NUMINAMATH_GPT_complementary_supplementary_angle_l878_87830

theorem complementary_supplementary_angle (x : ℝ) :
  (90 - x) * 3 = 180 - x → x = 45 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_complementary_supplementary_angle_l878_87830


namespace NUMINAMATH_GPT_total_weight_on_scale_l878_87880

-- Define the weights of Alexa and Katerina
def alexa_weight : ℕ := 46
def katerina_weight : ℕ := 49

-- State the theorem to prove the total weight on the scale
theorem total_weight_on_scale : alexa_weight + katerina_weight = 95 := by
  sorry

end NUMINAMATH_GPT_total_weight_on_scale_l878_87880


namespace NUMINAMATH_GPT_increasing_order_magnitudes_l878_87882

variable (x : ℝ)

noncomputable def y := x^x
noncomputable def z := x^(x^x)

theorem increasing_order_magnitudes (h1 : 1 < x) (h2 : x < 1.1) : x < y x ∧ y x < z x :=
by
  have h3 : y x = x^x := rfl
  have h4 : z x = x^(x^x) := rfl
  sorry

end NUMINAMATH_GPT_increasing_order_magnitudes_l878_87882


namespace NUMINAMATH_GPT_smallest_n_contains_digit9_and_terminating_decimal_l878_87811

-- Define the condition that a number contains the digit 9
def contains_digit_9 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 9

-- Define the condition that a number is of the form 2^a * 5^b
def is_form_of_2a_5b (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2 ^ a * 5 ^ b

-- Define the main theorem
theorem smallest_n_contains_digit9_and_terminating_decimal : 
  ∃ (n : ℕ), contains_digit_9 n ∧ is_form_of_2a_5b n ∧ (∀ m, (contains_digit_9 m ∧ is_form_of_2a_5b m) → n ≤ m) ∧ n = 12500 :=
  sorry

end NUMINAMATH_GPT_smallest_n_contains_digit9_and_terminating_decimal_l878_87811


namespace NUMINAMATH_GPT_determinant_expression_l878_87889

noncomputable def matrixDet (α β : ℝ) : ℝ :=
  Matrix.det ![
    ![Real.sin α * Real.cos β, -Real.sin α * Real.sin β, Real.cos α],
    ![-Real.sin β, -Real.cos β, 0],
    ![Real.cos α * Real.cos β, Real.cos α * Real.sin β, Real.sin α]]

theorem determinant_expression (α β: ℝ) : matrixDet α β = Real.sin α ^ 3 := 
by 
  sorry

end NUMINAMATH_GPT_determinant_expression_l878_87889


namespace NUMINAMATH_GPT_xy_squared_l878_87838

theorem xy_squared (x y : ℚ) (h1 : x + y = 9 / 20) (h2 : x - y = 1 / 20) :
  x^2 - y^2 = 9 / 400 :=
by
  sorry

end NUMINAMATH_GPT_xy_squared_l878_87838


namespace NUMINAMATH_GPT_sarahs_score_l878_87897

theorem sarahs_score (g s : ℕ) (h₁ : s = g + 30) (h₂ : (s + g) / 2 = 95) : s = 110 := by
  sorry

end NUMINAMATH_GPT_sarahs_score_l878_87897


namespace NUMINAMATH_GPT_find_x_l878_87820

theorem find_x (x : ℝ) : 
  45 - (28 - (37 - (x - 17))) = 56 ↔ x = 15 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l878_87820


namespace NUMINAMATH_GPT_inequality_mn_l878_87864

theorem inequality_mn (m n : ℤ)
  (h : ∃ x : ℤ, (x + m) * (x + n) = x + m + n) : 
  2 * (m^2 + n^2) < 5 * m * n := 
sorry

end NUMINAMATH_GPT_inequality_mn_l878_87864


namespace NUMINAMATH_GPT_sum_angles_bisected_l878_87891

theorem sum_angles_bisected (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : 0 < θ₁) (h₂ : 0 < θ₂) (h₃ : 0 < θ₃) (h₄ : 0 < θ₄)
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = 360) :
  (θ₁ / 2 + θ₃ / 2 = 180 ∨ θ₂ / 2 + θ₄ / 2 = 180) ∧ (θ₂ / 2 + θ₄ / 2 = 180 ∨ θ₁ / 2 + θ₃ / 2 = 180) := 
by 
  sorry

end NUMINAMATH_GPT_sum_angles_bisected_l878_87891


namespace NUMINAMATH_GPT_original_mixture_percentage_l878_87827

def mixture_percentage_acid (a w : ℕ) : ℚ :=
  a / (a + w)

theorem original_mixture_percentage (a w : ℕ) :
  (a / (a + w+2) = 1 / 4) ∧ ((a + 2) / (a + w + 4) = 2 / 5) → 
  mixture_percentage_acid a w = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_original_mixture_percentage_l878_87827


namespace NUMINAMATH_GPT_triangle_area_and_fraction_of_square_l878_87818

theorem triangle_area_and_fraction_of_square 
  (a b c s : ℕ) 
  (h_triangle : a = 9 ∧ b = 40 ∧ c = 41)
  (h_square : s = 41)
  (h_right_angle : a^2 + b^2 = c^2) :
  let area_triangle := (a * b) / 2
  let area_square := s^2
  let fraction := (a * b) / (2 * s^2)
  area_triangle = 180 ∧ fraction = 180 / 1681 := 
by
  sorry

end NUMINAMATH_GPT_triangle_area_and_fraction_of_square_l878_87818


namespace NUMINAMATH_GPT_obtuse_triangles_in_17_gon_l878_87883

noncomputable def number_of_obtuse_triangles (n : ℕ): ℕ := 
  if h : n ≥ 3 then (n * (n - 1) * (n - 2)) / 6 else 0

theorem obtuse_triangles_in_17_gon : number_of_obtuse_triangles 17 = 476 := sorry

end NUMINAMATH_GPT_obtuse_triangles_in_17_gon_l878_87883


namespace NUMINAMATH_GPT_min_value_l878_87826

open Real

noncomputable def y1 (x1 : ℝ) : ℝ := x1 * log x1
noncomputable def y2 (x2 : ℝ) : ℝ := x2 - 3

theorem min_value :
  ∃ (x1 x2 : ℝ), (x1 - x2)^2 + (y1 x1 - y2 x2)^2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_l878_87826


namespace NUMINAMATH_GPT_fraction_spent_at_toy_store_l878_87895

theorem fraction_spent_at_toy_store 
  (total_allowance : ℝ)
  (arcade_fraction : ℝ)
  (candy_store_amount : ℝ) 
  (remaining_allowance : ℝ)
  (toy_store_amount : ℝ)
  (H1 : total_allowance = 2.40)
  (H2 : arcade_fraction = 3 / 5)
  (H3 : candy_store_amount = 0.64)
  (H4 : remaining_allowance = total_allowance - (arcade_fraction * total_allowance))
  (H5 : toy_store_amount = remaining_allowance - candy_store_amount) :
  toy_store_amount / remaining_allowance = 1 / 3 := 
sorry

end NUMINAMATH_GPT_fraction_spent_at_toy_store_l878_87895


namespace NUMINAMATH_GPT_trajectory_midpoints_parabola_l878_87893

theorem trajectory_midpoints_parabola {k : ℝ} (hk : k ≠ 0) :
  ∀ (x1 x2 y1 y2 : ℝ), 
    y1 = 2 * x1^2 → 
    y2 = 2 * x2^2 → 
    y2 - y1 = 2 * (x2 + x1) * (x2 - x1) → 
    x = (x1 + x2) / 2 → 
    k = (y2 - y1) / (x2 - x1) → 
    x = 1 / (4 * k) := 
sorry

end NUMINAMATH_GPT_trajectory_midpoints_parabola_l878_87893


namespace NUMINAMATH_GPT_sum_of_a_and_c_l878_87874

variable {R : Type} [LinearOrderedField R]

theorem sum_of_a_and_c
    (ha hb hc hd : R) 
    (h_intersect : (1, 7) ∈ {p | p.2 = -2 * abs (p.1 - ha) + hb} ∧ (1, 7) ∈ {p | p.2 = 2 * abs (p.1 - hc) + hd}
                 ∧ (9, 1) ∈ {p | p.2 = -2 * abs (p.1 - ha) + hb} ∧ (9, 1) ∈ {p | p.2 = 2 * abs (p.1 - hc) + hd}) :
  ha + hc = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_a_and_c_l878_87874


namespace NUMINAMATH_GPT_inequality_solution_set_l878_87878

theorem inequality_solution_set {m n : ℝ} (h : ∀ x : ℝ, -3 < x ∧ x < 6 ↔ x^2 - m * x - 6 * n < 0) : m + n = 6 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l878_87878


namespace NUMINAMATH_GPT_petals_in_garden_l878_87805

def lilies_count : ℕ := 8
def tulips_count : ℕ := 5
def petals_per_lily : ℕ := 6
def petals_per_tulip : ℕ := 3

def total_petals : ℕ := lilies_count * petals_per_lily + tulips_count * petals_per_tulip

theorem petals_in_garden : total_petals = 63 := by
  sorry

end NUMINAMATH_GPT_petals_in_garden_l878_87805


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_sum_l878_87885

theorem arithmetic_geometric_sequence_sum 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ x y z : ℝ, (x = a ∧ y = -4 ∧ z = b ∨ x = b ∧ y = -4 ∧ z = a) 
                   ∧ (x + z = 2 * y) ∧ (x * z = y^2)) : 
  a + b = 10 :=
by sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_sum_l878_87885


namespace NUMINAMATH_GPT_smallest_x_plus_y_l878_87886

theorem smallest_x_plus_y 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_neq : x ≠ y)
  (h_eq : (1/x + 1/y = 1/10)) : x + y = 45 :=
sorry

end NUMINAMATH_GPT_smallest_x_plus_y_l878_87886


namespace NUMINAMATH_GPT_cubes_sum_is_214_5_l878_87844

noncomputable def r_plus_s_plus_t : ℝ := 12
noncomputable def rs_plus_rt_plus_st : ℝ := 47
noncomputable def rst : ℝ := 59.5

theorem cubes_sum_is_214_5 :
    (r_plus_s_plus_t * ((r_plus_s_plus_t)^2 - 3 * rs_plus_rt_plus_st) + 3 * rst) = 214.5 := by
    sorry

end NUMINAMATH_GPT_cubes_sum_is_214_5_l878_87844


namespace NUMINAMATH_GPT_gcd_polynomial_l878_87840

-- Given definitions based on the conditions
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

-- Given the conditions: a is a multiple of 1610
variables (a : ℕ) (h : is_multiple_of a 1610)

-- Main theorem: Prove that gcd(a^2 + 9a + 35, a + 5) = 15
theorem gcd_polynomial (h : is_multiple_of a 1610) : gcd (a^2 + 9*a + 35) (a + 5) = 15 :=
sorry

end NUMINAMATH_GPT_gcd_polynomial_l878_87840


namespace NUMINAMATH_GPT_fraction_of_girls_on_trip_l878_87837

theorem fraction_of_girls_on_trip (b g : ℕ) (h : b = g) :
  ((2 / 3 * g) / (5 / 6 * b + 2 / 3 * g)) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_girls_on_trip_l878_87837


namespace NUMINAMATH_GPT_find_k_l878_87813

noncomputable def sequence_sum (n : ℕ) (k : ℝ) : ℝ :=
  3 * 2^n + k

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a n * a (n + 2) = (a (n + 1))^2

theorem find_k
  (a : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, a n = sequence_sum (n + 1) k - sequence_sum n k)
  (h2 : geometric_sequence a) :
  k = -3 :=
  by sorry

end NUMINAMATH_GPT_find_k_l878_87813


namespace NUMINAMATH_GPT_maximum_surface_area_of_cuboid_l878_87833

noncomputable def max_surface_area_of_inscribed_cuboid (R : ℝ) :=
  let (a, b, c) := (R, R, R) -- assuming cube dimensions where a=b=c
  2 * a * b + 2 * a * c + 2 * b * c

theorem maximum_surface_area_of_cuboid (R : ℝ) (h : ∃ a b c : ℝ, a^2 + b^2 + c^2 = 4 * R^2) :
  max_surface_area_of_inscribed_cuboid R = 8 * R^2 :=
sorry

end NUMINAMATH_GPT_maximum_surface_area_of_cuboid_l878_87833


namespace NUMINAMATH_GPT_fraction_meaningful_l878_87828

theorem fraction_meaningful (x : ℝ) : x ≠ 1 ↔ ∃ (f : ℝ → ℝ), f x = (x + 2) / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l878_87828


namespace NUMINAMATH_GPT_prove_ellipse_and_dot_product_l878_87841

open Real

-- Assume the given conditions
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : a > b)
variable (e : ℝ) (he : e = sqrt 2 / 2)
variable (h_chord : 2 = 2 * sqrt (a^2 - 1))
variables (k : ℝ) (hk : k ≠ 0)

-- Given equation of points on the line and the ellipse
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 2 + y^2 = 1
def line_eq (x y : ℝ) : Prop := y = k * (x - 1)

-- The points A and B lie on the ellipse and the line
variables (x1 y1 x2 y2 : ℝ)
variable (A : x1^2 / 2 + y1^2 = 1 ∧ y1 = k * (x1 - 1))
variable (B : x2^2 / 2 + y2^2 = 1 ∧ y2 = k * (x2 - 1))

-- Define the dot product condition
def MA_dot_MB (m : ℝ) : ℝ :=
  let x1_term := x1 - m
  let x2_term := x2 - m
  let dot_product := (x1_term * x2_term + y1 * y2)
  dot_product

-- The statement we need to prove
theorem prove_ellipse_and_dot_product :
  (a^2 = 2) ∧ (b = 1) ∧ (c = 1) ∧ (∃ (m : ℝ), m = 5 / 4 ∧ MA_dot_MB m = -7 / 16) :=
sorry

end NUMINAMATH_GPT_prove_ellipse_and_dot_product_l878_87841


namespace NUMINAMATH_GPT_goldfish_growth_solution_l878_87843

def goldfish_growth_problem : Prop :=
  ∃ n : ℕ, 
    (∀ k, (k < n → 3 * (5:ℕ)^k ≠ 243 * (3:ℕ)^k)) ∧
    3 * (5:ℕ)^n = 243 * (3:ℕ)^n

theorem goldfish_growth_solution : goldfish_growth_problem :=
sorry

end NUMINAMATH_GPT_goldfish_growth_solution_l878_87843


namespace NUMINAMATH_GPT_cubic_has_three_natural_roots_l878_87899

theorem cubic_has_three_natural_roots (p : ℝ) :
  (∃ (x1 x2 x3 : ℕ), 5 * (x1:ℝ)^3 - 5 * (p + 1) * (x1:ℝ)^2 + (71 * p - 1) * (x1:ℝ) + 1 = 66 * p ∧
                     5 * (x2:ℝ)^3 - 5 * (p + 1) * (x2:ℝ)^2 + (71 * p - 1) * (x2:ℝ) + 1 = 66 * p ∧
                     5 * (x3:ℝ)^3 - 5 * (p + 1) * (x3:ℝ)^2 + (71 * p - 1) * (x3:ℝ) + 1 = 66 * p) ↔ p = 76 :=
by sorry

end NUMINAMATH_GPT_cubic_has_three_natural_roots_l878_87899


namespace NUMINAMATH_GPT_determine_m_l878_87824

def setA_is_empty (m: ℝ) : Prop :=
  { x : ℝ | m * x = 1 } = ∅

theorem determine_m (m: ℝ) (h: setA_is_empty m) : m = 0 :=
by sorry

end NUMINAMATH_GPT_determine_m_l878_87824


namespace NUMINAMATH_GPT_avg_age_initial_group_l878_87869

theorem avg_age_initial_group (N : ℕ) (A avg_new_persons avg_entire_group : ℝ) (hN : N = 15)
  (h_avg_new_persons : avg_new_persons = 15) (h_avg_entire_group : avg_entire_group = 15.5) :
  (A * (N : ℝ) + 15 * avg_new_persons) = ((N + 15) : ℝ) * avg_entire_group → A = 16 :=
by
  intro h
  have h_initial : N = 15 := hN
  have h_new : avg_new_persons = 15 := h_avg_new_persons
  have h_group : avg_entire_group = 15.5 := h_avg_entire_group
  sorry

end NUMINAMATH_GPT_avg_age_initial_group_l878_87869


namespace NUMINAMATH_GPT_circumscribedCircleDiameter_is_10sqrt2_l878_87851

noncomputable def circumscribedCircleDiameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem circumscribedCircleDiameter_is_10sqrt2 :
  circumscribedCircleDiameter 10 (Real.pi / 4) = 10 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_circumscribedCircleDiameter_is_10sqrt2_l878_87851


namespace NUMINAMATH_GPT_find_f_neg_3_l878_87819

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def functional_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 - x) = f (1 + x)

def function_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2

theorem find_f_neg_3 
  (hf_even : even_function f) 
  (hf_condition : functional_condition f)
  (hf_interval : function_on_interval f) : 
  f (-3) = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_f_neg_3_l878_87819


namespace NUMINAMATH_GPT_find_x_l878_87860

theorem find_x
  (x : ℝ)
  (h1 : (x - 2)^2 + (15 - 5)^2 = 13^2)
  (h2 : x > 0) : 
  x = 2 + Real.sqrt 69 :=
sorry

end NUMINAMATH_GPT_find_x_l878_87860


namespace NUMINAMATH_GPT_correct_total_cost_correct_remaining_donuts_l878_87814

-- Conditions
def budget : ℝ := 50
def cost_per_box : ℝ := 12
def discount_percentage : ℝ := 0.10
def number_of_boxes_bought : ℕ := 4
def donuts_per_box : ℕ := 12
def boxes_given_away : ℕ := 1
def additional_donuts_given_away : ℕ := 6

-- Calculations based on conditions
def total_cost_before_discount : ℝ := number_of_boxes_bought * cost_per_box
def discount_amount : ℝ := discount_percentage * total_cost_before_discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

def total_donuts : ℕ := number_of_boxes_bought * donuts_per_box
def total_donuts_given_away : ℕ := (boxes_given_away * donuts_per_box) + additional_donuts_given_away
def remaining_donuts : ℕ := total_donuts - total_donuts_given_away

-- Theorems to prove
theorem correct_total_cost : total_cost_after_discount = 43.20 := by
  -- proof here
  sorry

theorem correct_remaining_donuts : remaining_donuts = 30 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_correct_total_cost_correct_remaining_donuts_l878_87814


namespace NUMINAMATH_GPT_minimum_a_l878_87854

noncomputable def f (x a : ℝ) := Real.exp x * (x^3 - 3 * x + 3) - a * Real.exp x - x

theorem minimum_a (a : ℝ) : (∃ x, x ≥ -2 ∧ f x a ≤ 0) ↔ a ≥ 1 - 1 / Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_a_l878_87854


namespace NUMINAMATH_GPT_kneading_time_is_correct_l878_87815

def total_time := 280
def rising_time_per_session := 120
def number_of_rising_sessions := 2
def baking_time := 30

def total_rising_time := rising_time_per_session * number_of_rising_sessions
def total_non_kneading_time := total_rising_time + baking_time
def kneading_time := total_time - total_non_kneading_time

theorem kneading_time_is_correct : kneading_time = 10 := by
  have h1 : total_rising_time = 240 := by
    sorry
  have h2 : total_non_kneading_time = 270 := by
    sorry
  have h3 : kneading_time = 10 := by
    sorry
  exact h3

end NUMINAMATH_GPT_kneading_time_is_correct_l878_87815


namespace NUMINAMATH_GPT_geometric_sum_six_l878_87816

theorem geometric_sum_six (a r : ℚ) (n : ℕ) 
  (hn₁ : a = 1/4) 
  (hn₂ : r = 1/2) 
  (hS: a * (1 - r^n) / (1 - r) = 63/128) : 
  n = 6 :=
by
  -- Statement to be Proven
  rw [hn₁, hn₂] at hS
  sorry

end NUMINAMATH_GPT_geometric_sum_six_l878_87816


namespace NUMINAMATH_GPT_find_fraction_l878_87806

variable {N : ℕ}
variable {f : ℚ}

theorem find_fraction (h1 : N = 150) (h2 : N - f * N = 60) : f = 3/5 := by
  sorry

end NUMINAMATH_GPT_find_fraction_l878_87806


namespace NUMINAMATH_GPT_cost_per_bracelet_l878_87873

/-- Each friend and the number of their name's letters -/
def friends_letters_counts : List (String × Nat) :=
  [("Jessica", 7), ("Tori", 4), ("Lily", 4), ("Patrice", 7)]

/-- Total cost spent by Robin -/
def total_cost : Nat := 44

/-- Calculate the total number of bracelets -/
def total_bracelets : Nat :=
  friends_letters_counts.foldr (λ p acc => p.snd + acc) 0

theorem cost_per_bracelet : (total_cost / total_bracelets) = 2 :=
  by
    sorry

end NUMINAMATH_GPT_cost_per_bracelet_l878_87873


namespace NUMINAMATH_GPT_molecular_weight_of_one_mole_l878_87848

noncomputable def molecular_weight (total_weight : ℝ) (moles : ℕ) : ℝ :=
total_weight / moles

theorem molecular_weight_of_one_mole (h : molecular_weight 252 6 = 42) : molecular_weight 252 6 = 42 := by
  exact h

end NUMINAMATH_GPT_molecular_weight_of_one_mole_l878_87848


namespace NUMINAMATH_GPT_first_candidate_percentage_l878_87859

theorem first_candidate_percentage (total_votes : ℕ) (invalid_percentage : ℕ) (second_candidate_votes : ℕ) 
  (h_total_votes : total_votes = 7500) 
  (h_invalid_percentage : invalid_percentage = 20) 
  (h_second_candidate_votes : second_candidate_votes = 2700) : 
  (100 * (total_votes * (1 - (invalid_percentage / 100)) - second_candidate_votes) / (total_votes * (1 - (invalid_percentage / 100)))) = 55 :=
by
  sorry

end NUMINAMATH_GPT_first_candidate_percentage_l878_87859


namespace NUMINAMATH_GPT_sum_of_coefficients_l878_87847

-- Define the polynomial
def polynomial (x : ℝ) : ℝ :=
  2 * (4 * x ^ 8 + 7 * x ^ 6 - 9 * x ^ 3 + 3) + 6 * (x ^ 7 - 2 * x ^ 4 + 8 * x ^ 2 - 2)

-- State the theorem to prove the sum of the coefficients
theorem sum_of_coefficients : polynomial 1 = 40 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l878_87847


namespace NUMINAMATH_GPT_count_males_not_in_orchestra_l878_87861

variable (females_band females_orchestra females_choir females_all
          males_band males_orchestra males_choir males_all total_students : ℕ)
variable (males_band_not_in_orchestra : ℕ)

theorem count_males_not_in_orchestra :
  females_band = 120 ∧ females_orchestra = 90 ∧ females_choir = 50 ∧ females_all = 30 ∧
  males_band = 90 ∧ males_orchestra = 120 ∧ males_choir = 40 ∧ males_all = 20 ∧
  total_students = 250 ∧ males_band_not_in_orchestra = (males_band - (males_band + males_orchestra + males_choir - males_all - total_students)) 
  → males_band_not_in_orchestra = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_count_males_not_in_orchestra_l878_87861


namespace NUMINAMATH_GPT_kite_area_eq_twenty_l878_87817

theorem kite_area_eq_twenty :
  let base := 10
  let height := 2
  let area_of_triangle := (1 / 2 : ℝ) * base * height
  let total_area := 2 * area_of_triangle
  total_area = 20 :=
by
  sorry

end NUMINAMATH_GPT_kite_area_eq_twenty_l878_87817


namespace NUMINAMATH_GPT_archie_touchdown_passes_l878_87890

-- Definitions based on the conditions
def richard_avg_first_14_games : ℕ := 6
def richard_avg_last_2_games : ℕ := 3
def richard_games_first : ℕ := 14
def richard_games_last : ℕ := 2

-- Total touchdowns Richard made in the first 14 games
def touchdowns_first_14 := richard_games_first * richard_avg_first_14_games

-- Total touchdowns Richard needs in the final 2 games
def touchdowns_last_2 := richard_games_last * richard_avg_last_2_games

-- Total touchdowns Richard made in the season
def richard_touchdowns_season := touchdowns_first_14 + touchdowns_last_2

-- Archie's record is one less than Richard's total touchdowns for the season
def archie_record := richard_touchdowns_season - 1

-- Proposition to prove Archie's touchdown passes in a season
theorem archie_touchdown_passes : archie_record = 89 := by
  sorry

end NUMINAMATH_GPT_archie_touchdown_passes_l878_87890


namespace NUMINAMATH_GPT_congruent_triangles_implies_corresponding_sides_equal_corresponding_sides_equal_implies_congruent_triangles_not_congruent_triangles_implies_not_corresponding_sides_equal_not_corresponding_sides_equal_implies_not_congruent_triangles_four_equal_sides_implies_is_square_is_square_implies_four_equal_sides_not_four_equal_sides_implies_not_is_square_not_is_square_implies_not_four_equal_sides_l878_87801

namespace GeometricPropositions

-- Definitions for congruence in triangles and quadrilaterals:
def congruent_triangles (Δ1 Δ2 : Type) : Prop := sorry
def corresponding_sides_equal (Δ1 Δ2 : Type) : Prop := sorry

def four_equal_sides (Q : Type) : Prop := sorry
def is_square (Q : Type) : Prop := sorry

-- Propositions and their logical forms for triangles
theorem congruent_triangles_implies_corresponding_sides_equal (Δ1 Δ2 : Type) : congruent_triangles Δ1 Δ2 → corresponding_sides_equal Δ1 Δ2 := sorry

theorem corresponding_sides_equal_implies_congruent_triangles (Δ1 Δ2 : Type) : corresponding_sides_equal Δ1 Δ2 → congruent_triangles Δ1 Δ2 := sorry

theorem not_congruent_triangles_implies_not_corresponding_sides_equal (Δ1 Δ2 : Type) : ¬ congruent_triangles Δ1 Δ2 → ¬ corresponding_sides_equal Δ1 Δ2 := sorry

theorem not_corresponding_sides_equal_implies_not_congruent_triangles (Δ1 Δ2 : Type) : ¬ corresponding_sides_equal Δ1 Δ2 → ¬ congruent_triangles Δ1 Δ2 := sorry

-- Propositions and their logical forms for quadrilaterals
theorem four_equal_sides_implies_is_square (Q : Type) : four_equal_sides Q → is_square Q := sorry

theorem is_square_implies_four_equal_sides (Q : Type) : is_square Q → four_equal_sides Q := sorry

theorem not_four_equal_sides_implies_not_is_square (Q : Type) : ¬ four_equal_sides Q → ¬ is_square Q := sorry

theorem not_is_square_implies_not_four_equal_sides (Q : Type) : ¬ is_square Q → ¬ four_equal_sides Q := sorry

end GeometricPropositions

end NUMINAMATH_GPT_congruent_triangles_implies_corresponding_sides_equal_corresponding_sides_equal_implies_congruent_triangles_not_congruent_triangles_implies_not_corresponding_sides_equal_not_corresponding_sides_equal_implies_not_congruent_triangles_four_equal_sides_implies_is_square_is_square_implies_four_equal_sides_not_four_equal_sides_implies_not_is_square_not_is_square_implies_not_four_equal_sides_l878_87801


namespace NUMINAMATH_GPT_find_k_l878_87821

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := x + 2 * y - 7 = 0
def l2 (x y : ℝ) (k : ℝ) : Prop := 2 * x + k * x + 3 = 0

-- Define the condition for parallel lines in our context
def parallel (k : ℝ) : Prop := - (1 / 2) = -(2 / k)

-- Prove that under the given conditions, k must be 4
theorem find_k (k : ℝ) : parallel k → k = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_k_l878_87821


namespace NUMINAMATH_GPT_minimize_M_l878_87808

noncomputable def M (x y : ℝ) : ℝ := 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9

theorem minimize_M : ∃ x y, M x y = 5 ∧ x = -3 ∧ y = -2 :=
by
  sorry

end NUMINAMATH_GPT_minimize_M_l878_87808


namespace NUMINAMATH_GPT_slips_drawn_l878_87845

theorem slips_drawn (P : ℚ) (P_value : P = 24⁻¹) :
  ∃ n : ℕ, (n ≤ 5 ∧ P = (Nat.choose 5 n) / (Nat.choose 10 n) ∧ n = 4) := by
{
  sorry
}

end NUMINAMATH_GPT_slips_drawn_l878_87845


namespace NUMINAMATH_GPT_taylor_probability_l878_87898

open Nat Real

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

theorem taylor_probability :
  (binomial_probability 5 2 (3/5) = 144 / 625) :=
by
  sorry

end NUMINAMATH_GPT_taylor_probability_l878_87898


namespace NUMINAMATH_GPT_original_price_of_goods_l878_87802

theorem original_price_of_goods
  (rebate_percent : ℝ := 0.06)
  (tax_percent : ℝ := 0.10)
  (total_paid : ℝ := 6876.1) :
  ∃ P : ℝ, (P - P * rebate_percent) * (1 + tax_percent) = total_paid ∧ P = 6650 :=
sorry

end NUMINAMATH_GPT_original_price_of_goods_l878_87802


namespace NUMINAMATH_GPT_find_missing_number_l878_87809

noncomputable def missing_number : Prop :=
  ∃ (y x a b : ℝ),
    a = y + x ∧
    b = x + 630 ∧
    28 = y * a ∧
    660 = a * b ∧
    y = 13

theorem find_missing_number : missing_number :=
  sorry

end NUMINAMATH_GPT_find_missing_number_l878_87809


namespace NUMINAMATH_GPT_compare_two_sqrt_five_five_l878_87823

theorem compare_two_sqrt_five_five : 2 * Real.sqrt 5 < 5 :=
sorry

end NUMINAMATH_GPT_compare_two_sqrt_five_five_l878_87823


namespace NUMINAMATH_GPT_find_b_value_l878_87887

-- Definitions based on the problem conditions
def line_bisects_circle (b : ℝ) : Prop :=
  ∃ c : ℝ × ℝ, (c.fst = 4 ∧ c.snd = -1) ∧
                (c.snd = c.fst + b)

-- Theorem statement for the problem
theorem find_b_value : line_bisects_circle (-5) :=
by
  sorry

end NUMINAMATH_GPT_find_b_value_l878_87887
