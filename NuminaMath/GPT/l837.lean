import Mathlib

namespace NUMINAMATH_GPT_identical_solutions_of_quadratic_linear_l837_83775

theorem identical_solutions_of_quadratic_linear (k : ℝ) :
  (∃ x : ℝ, x^2 = 4 * x + k ∧ x^2 = 4 * x + k) ↔ k = -4 :=
by
  sorry

end NUMINAMATH_GPT_identical_solutions_of_quadratic_linear_l837_83775


namespace NUMINAMATH_GPT_gcd_bezout_663_182_l837_83740

theorem gcd_bezout_663_182 :
  let a := 182
  let b := 663
  ∃ d u v : ℤ, d = Int.gcd a b ∧ d = a * u + b * v ∧ d = 13 ∧ u = 11 ∧ v = -3 :=
by 
  let a := 182
  let b := 663
  use 13, 11, -3
  sorry

end NUMINAMATH_GPT_gcd_bezout_663_182_l837_83740


namespace NUMINAMATH_GPT_factorization_of_polynomial_l837_83734

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 :=
by
  intros x
  sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l837_83734


namespace NUMINAMATH_GPT_evaluate_expression_l837_83725

theorem evaluate_expression :
  (305^2 - 275^2) / 30 = 580 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l837_83725


namespace NUMINAMATH_GPT_smallest_possible_abc_l837_83707

open Nat

theorem smallest_possible_abc (a b c : ℕ)
  (h₁ : 5 * c ∣ a * b)
  (h₂ : 13 * a ∣ b * c)
  (h₃ : 31 * b ∣ a * c) :
  abc = 4060225 :=
by sorry

end NUMINAMATH_GPT_smallest_possible_abc_l837_83707


namespace NUMINAMATH_GPT_coordinates_of_A_l837_83788

-- Definition of the point A with coordinates (-1, 3)
def point_A : ℝ × ℝ := (-1, 3)

-- Statement that the coordinates of point A with respect to the origin are (-1, 3)
theorem coordinates_of_A : point_A = (-1, 3) := by
  sorry

end NUMINAMATH_GPT_coordinates_of_A_l837_83788


namespace NUMINAMATH_GPT_find_xy_l837_83796

variable {x y : ℝ}

theorem find_xy (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l837_83796


namespace NUMINAMATH_GPT_construct_using_five_twos_l837_83741

theorem construct_using_five_twos :
  (∃ (a b c d e f : ℕ), (22 * (a / b)) / c = 11 ∧
                        (22 / d) + (e / f) = 12 ∧
                        (22 + g + h) / i = 13 ∧
                        (2 * 2 * 2 * 2 - j) = 14 ∧
                        (22 / k) + (2 * 2) = 15) := by
  sorry

end NUMINAMATH_GPT_construct_using_five_twos_l837_83741


namespace NUMINAMATH_GPT_total_difference_in_cups_l837_83748

theorem total_difference_in_cups (h1: Nat) (h2: Nat) (h3: Nat) (hrs: Nat) : 
  h1 = 4 → h2 = 7 → h3 = 5 → hrs = 3 → 
  ((h2 * hrs - h1 * hrs) + (h3 * hrs - h1 * hrs) + (h2 * hrs - h3 * hrs)) = 18 :=
by
  intros h1_eq h2_eq h3_eq hrs_eq
  sorry

end NUMINAMATH_GPT_total_difference_in_cups_l837_83748


namespace NUMINAMATH_GPT_Second_beats_Third_by_miles_l837_83731

theorem Second_beats_Third_by_miles
  (v1 v2 v3 : ℝ) -- speeds of First, Second, and Third
  (H1 : (10 / v1) = (8 / v2)) -- First beats Second by 2 miles in 10-mile race
  (H2 : (10 / v1) = (6 / v3)) -- First beats Third by 4 miles in 10-mile race
  : (10 - (v3 * (10 / v2))) = 2.5 := 
sorry

end NUMINAMATH_GPT_Second_beats_Third_by_miles_l837_83731


namespace NUMINAMATH_GPT_sum_consecutive_numbers_last_digit_diff_l837_83791

theorem sum_consecutive_numbers_last_digit_diff (a : ℕ) : 
    (2015 * (a + 1007) % 10) ≠ (2019 * (a + 3024) % 10) := 
by 
  sorry

end NUMINAMATH_GPT_sum_consecutive_numbers_last_digit_diff_l837_83791


namespace NUMINAMATH_GPT_diophantine_soln_l837_83769

-- Define the Diophantine equation as a predicate
def diophantine_eq (x y : ℤ) : Prop := x^3 - y^3 = 2 * x * y + 8

-- Theorem stating that the only solutions are (0, -2) and (2, 0)
theorem diophantine_soln :
  ∀ x y : ℤ, diophantine_eq x y ↔ (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_diophantine_soln_l837_83769


namespace NUMINAMATH_GPT_overall_percentage_badminton_l837_83772

theorem overall_percentage_badminton (N S : ℕ) (pN pS : ℝ) :
  N = 1500 → S = 1800 → pN = 0.30 → pS = 0.35 → 
  ( (N * pN + S * pS) / (N + S) ) * 100 = 33 := 
by
  intros hN hS hpN hpS
  sorry

end NUMINAMATH_GPT_overall_percentage_badminton_l837_83772


namespace NUMINAMATH_GPT_domain_eq_l837_83785

def domain_of_function :
    Set ℝ := {x | (x - 1 ≥ 0) ∧ (x + 1 > 0)}

theorem domain_eq :
    domain_of_function = {x | x ≥ 1} :=
by
  sorry

end NUMINAMATH_GPT_domain_eq_l837_83785


namespace NUMINAMATH_GPT_highest_of_seven_consecutive_with_average_33_l837_83790

theorem highest_of_seven_consecutive_with_average_33 (x : ℤ) 
    (h : (x - 3 + x - 2 + x - 1 + x + x + 1 + x + 2 + x + 3) / 7 = 33) : 
    x + 3 = 36 := 
sorry

end NUMINAMATH_GPT_highest_of_seven_consecutive_with_average_33_l837_83790


namespace NUMINAMATH_GPT_sum_gcd_lcm_75_4410_l837_83760

theorem sum_gcd_lcm_75_4410 :
  Nat.gcd 75 4410 + Nat.lcm 75 4410 = 22065 := by
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_75_4410_l837_83760


namespace NUMINAMATH_GPT_converse_proposition_l837_83781

theorem converse_proposition (a b c : ℝ) (h : c ≠ 0) :
  a * c^2 > b * c^2 → a > b :=
by
  sorry

end NUMINAMATH_GPT_converse_proposition_l837_83781


namespace NUMINAMATH_GPT_largest_power_of_2_dividing_n_l837_83722

open Nat

-- Defining given expressions
def n : ℕ := 17^4 - 9^4 + 8 * 17^2

-- The theorem to prove
theorem largest_power_of_2_dividing_n : 2^3 ∣ n ∧ ∀ k, (k > 3 → ¬ 2^k ∣ n) :=
by
  sorry

end NUMINAMATH_GPT_largest_power_of_2_dividing_n_l837_83722


namespace NUMINAMATH_GPT_max_value_of_xy_l837_83720

theorem max_value_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 2) :
  xy ≤ 1 / 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_xy_l837_83720


namespace NUMINAMATH_GPT_residue_11_pow_2021_mod_19_l837_83758

theorem residue_11_pow_2021_mod_19 : (11^2021) % 19 = 17 := 
by
  -- this is to ensure the theorem is syntactically correct in Lean but skips the proof for now
  sorry

end NUMINAMATH_GPT_residue_11_pow_2021_mod_19_l837_83758


namespace NUMINAMATH_GPT_inequality_solution_set_no_positive_a_b_exists_l837_83708

def f (x : ℝ) := abs (2 * x - 1) - abs (2 * x - 2)
def k := 1

theorem inequality_solution_set :
  { x : ℝ | f x ≥ x } = { x : ℝ | x ≤ -1 ∨ x = 1 } :=
sorry

theorem no_positive_a_b_exists (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  ¬ (a + 2 * b = k ∧ 2 / a + 1 / b = 4 - 1 / (a * b)) :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_no_positive_a_b_exists_l837_83708


namespace NUMINAMATH_GPT_rectangles_with_trapezoid_area_l837_83778

-- Define the necessary conditions
def small_square_area : ℝ := 1
def total_squares : ℕ := 12
def rows : ℕ := 4
def columns : ℕ := 3
def trapezoid_area : ℝ := 3

-- Statement of the proof problem
theorem rectangles_with_trapezoid_area :
  (∀ rows columns : ℕ, rows * columns = total_squares) →
  (∀ area : ℝ, area = small_square_area) →
  (∀ trapezoid_area : ℝ, trapezoid_area = 3) →
  (rows = 4) →
  (columns = 3) →
  ∃ rectangles : ℕ, rectangles = 10 :=
by
  sorry

end NUMINAMATH_GPT_rectangles_with_trapezoid_area_l837_83778


namespace NUMINAMATH_GPT_coefficient_x2_term_l837_83724

open Polynomial

noncomputable def poly1 : Polynomial ℝ := (X - 1)^3
noncomputable def poly2 : Polynomial ℝ := (X - 1)^4

theorem coefficient_x2_term :
  coeff (poly1 + poly2) 2 = 3 :=
sorry

end NUMINAMATH_GPT_coefficient_x2_term_l837_83724


namespace NUMINAMATH_GPT_oranges_in_bowl_l837_83768

-- Definitions (conditions)
def bananas : Nat := 2
def apples : Nat := 2 * bananas
def total_fruits : Nat := 12

-- Theorem (proof goal)
theorem oranges_in_bowl : 
  apples + bananas + oranges = total_fruits → oranges = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_oranges_in_bowl_l837_83768


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_for_monotonic_l837_83787

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
∀ x y, x ≤ y → f x ≤ f y

noncomputable def is_sufficient_condition (P Q : Prop) : Prop :=
P → Q

noncomputable def is_not_necessary_condition (P Q : Prop) : Prop :=
¬ Q → ¬ P

noncomputable def is_sufficient_but_not_necessary (P Q : Prop) : Prop :=
is_sufficient_condition P Q ∧ is_not_necessary_condition P Q

theorem sufficient_but_not_necessary_for_monotonic (f : ℝ → ℝ) :
  (∀ x, 0 ≤ deriv f x) → is_monotonically_increasing f :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_for_monotonic_l837_83787


namespace NUMINAMATH_GPT_find_sixth_number_l837_83746

theorem find_sixth_number (avg_all : ℝ) (avg_first6 : ℝ) (avg_last6 : ℝ) (total_avg : avg_all = 10.7) (first6_avg: avg_first6 = 10.5) (last6_avg: avg_last6 = 11.4) : 
  let S1 := 6 * avg_first6
  let S2 := 6 * avg_last6
  let total_sum := 11 * avg_all
  let X := total_sum - (S1 - X + S2 - X)
  X = 13.7 :=
by 
  sorry

end NUMINAMATH_GPT_find_sixth_number_l837_83746


namespace NUMINAMATH_GPT_tens_digit_19_pow_1987_l837_83751

theorem tens_digit_19_pow_1987 : (19 ^ 1987) % 100 / 10 = 3 := 
sorry

end NUMINAMATH_GPT_tens_digit_19_pow_1987_l837_83751


namespace NUMINAMATH_GPT_num_of_consec_int_sum_18_l837_83782

theorem num_of_consec_int_sum_18 : 
  ∃! (a n : ℕ), n ≥ 3 ∧ (n * (2 * a + n - 1)) = 36 :=
sorry

end NUMINAMATH_GPT_num_of_consec_int_sum_18_l837_83782


namespace NUMINAMATH_GPT_geometric_sequence_identity_l837_83759

variables {b : ℕ → ℝ} {m n p : ℕ}

def is_geometric_sequence (b : ℕ → ℝ) :=
  ∀ i j k : ℕ, i < j → j < k → b j^2 = b i * b k

noncomputable def distinct_pos_ints (m n p : ℕ) :=
  0 < m ∧ 0 < n ∧ 0 < p ∧ m ≠ n ∧ n ≠ p ∧ p ≠ m

theorem geometric_sequence_identity 
  (h_geom : is_geometric_sequence b) 
  (h_distinct : distinct_pos_ints m n p) : 
  b p ^ (m - n) * b m ^ (n - p) * b n ^ (p - m) = 1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_identity_l837_83759


namespace NUMINAMATH_GPT_lorry_sand_capacity_l837_83739

def cost_cement (bags : ℕ) (cost_per_bag : ℕ) : ℕ := bags * cost_per_bag
def total_cost (cement_cost : ℕ) (sand_cost : ℕ) : ℕ := cement_cost + sand_cost
def total_sand (sand_cost : ℕ) (cost_per_ton : ℕ) : ℕ := sand_cost / cost_per_ton
def sand_per_lorry (total_sand : ℕ) (lorries : ℕ) : ℕ := total_sand / lorries

theorem lorry_sand_capacity : 
  cost_cement 500 10 + (total_cost 5000 (total_sand 8000 40)) = 13000 ∧
  total_cost 5000 8000 = 13000 ∧
  total_sand 8000 40 = 200 ∧
  sand_per_lorry 200 20 = 10 :=
by
  sorry

end NUMINAMATH_GPT_lorry_sand_capacity_l837_83739


namespace NUMINAMATH_GPT_simplify_fraction_l837_83703

theorem simplify_fraction (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (15 * x^2 * y^3) / (9 * x * y^2) = 20 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l837_83703


namespace NUMINAMATH_GPT_range_of_m_l837_83784

-- Definitions of Propositions p and q
def Proposition_p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m > 0) ∧ (1 > 0)  -- where x₁ + x₂ = -m > 0 and x₁x₂ = 1

def Proposition_q (m : ℝ) : Prop :=
  16 * (m + 2)^2 - 16 < 0  -- discriminant of 4x^2 + 4(m+2)x + 1 = 0 is less than 0

-- Given: "Proposition p or Proposition q" is true
def given (m : ℝ) : Prop :=
  Proposition_p m ∨ Proposition_q m

-- Prove: Range of values for m is (-∞, -1)
theorem range_of_m (m : ℝ) (h : given m) : m < -1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l837_83784


namespace NUMINAMATH_GPT_parallel_vectors_l837_83744

variable (y : ℝ)

def vector_a : ℝ × ℝ := (-1, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

theorem parallel_vectors (h : (-1 * y - 3 * 2) = 0) : y = -6 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l837_83744


namespace NUMINAMATH_GPT_fraction_simplification_l837_83727

theorem fraction_simplification (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) :
  (x / (x - 1) = 3 / (2 * x - 2) - 3) → (2 * x = 3 - 6 * x + 6) :=
by 
  intro h1
  -- Proof steps would be here, but we are using sorry
  sorry

end NUMINAMATH_GPT_fraction_simplification_l837_83727


namespace NUMINAMATH_GPT_adrien_winning_strategy_l837_83761

/--
On the table, there are 2023 tokens. Adrien and Iris take turns removing at least one token and at most half of the remaining tokens at the time they play. The player who leaves a single token on the table loses the game. Adrien starts first. Prove that Adrien has a winning strategy.
-/
theorem adrien_winning_strategy : ∃ strategy : ℕ → ℕ, 
  ∀ n:ℕ, (n = 2023 ∧ 1 ≤ strategy n ∧ strategy n ≤ n / 2) → 
    (∀ u : ℕ, (u = n - strategy n) → (∃ strategy' : ℕ → ℕ , 
      ∀ m:ℕ, (m = u ∧ 1 ≤ strategy' m ∧ strategy' m ≤ m / 2) → 
        (∃ next_u : ℕ, (next_u = m - strategy' m → next_u ≠ 1 ∨ (m = 1 ∧ u ≠ 1 ∧ next_u = 1)))))
:= sorry

end NUMINAMATH_GPT_adrien_winning_strategy_l837_83761


namespace NUMINAMATH_GPT_moles_of_SO2_formed_l837_83794

variable (n_NaHSO3 n_HCl n_SO2 : ℕ)

/--
The reaction between sodium bisulfite (NaHSO3) and hydrochloric acid (HCl) is:
NaHSO3 + HCl → NaCl + H2O + SO2
Given 2 moles of NaHSO3 and 2 moles of HCl, prove that the number of moles of SO2 formed is 2.
-/
theorem moles_of_SO2_formed :
  (n_NaHSO3 = 2) →
  (n_HCl = 2) →
  (∀ (n : ℕ), (n_NaHSO3 = n) → (n_HCl = n) → (n_SO2 = n)) →
  n_SO2 = 2 :=
by 
  intros hNaHSO3 hHCl hReaction
  exact hReaction 2 hNaHSO3 hHCl

end NUMINAMATH_GPT_moles_of_SO2_formed_l837_83794


namespace NUMINAMATH_GPT_smallest_pos_int_terminating_decimal_with_9_l837_83776

theorem smallest_pos_int_terminating_decimal_with_9 : ∃ n : ℕ, (∃ m k : ℕ, n = 2^m * 5^k ∧ (∃ d : ℕ, d ∈ n.digits 10 ∧ d = 9)) ∧ n = 4096 :=
by {
    sorry
}

end NUMINAMATH_GPT_smallest_pos_int_terminating_decimal_with_9_l837_83776


namespace NUMINAMATH_GPT_product_increase_l837_83726

theorem product_increase (a b c : ℕ) (h1 : a ≥ 3) (h2 : b ≥ 3) (h3 : c ≥ 3) :
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2016 := by
  sorry

end NUMINAMATH_GPT_product_increase_l837_83726


namespace NUMINAMATH_GPT_no_solutions_system_l837_83767

theorem no_solutions_system :
  ∀ (x y : ℝ), 
  (x^3 + x + y + 1 = 0) →
  (y * x^2 + x + y = 0) →
  (y^2 + y - x^2 + 1 = 0) →
  false :=
by
  intro x y h1 h2 h3
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_no_solutions_system_l837_83767


namespace NUMINAMATH_GPT_compute_x_squared_first_compute_x_squared_second_l837_83763

variable (x : ℝ)
variable (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1)

theorem compute_x_squared_first : 
  1 / (1 / x - 1 / (x + 1)) - x = x^2 :=
by
  sorry

theorem compute_x_squared_second : 
  1 / (1 / (x - 1) - 1 / x) + x = x^2 :=
by
  sorry

end NUMINAMATH_GPT_compute_x_squared_first_compute_x_squared_second_l837_83763


namespace NUMINAMATH_GPT_value_of_x_squared_plus_9y_squared_l837_83786

theorem value_of_x_squared_plus_9y_squared (x y : ℝ) (h1 : x - 3 * y = 3) (h2 : x * y = -9) : x^2 + 9 * y^2 = -45 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_9y_squared_l837_83786


namespace NUMINAMATH_GPT_distance_to_y_axis_l837_83749

theorem distance_to_y_axis {x y : ℝ} (h : x = -3 ∧ y = 4) : abs x = 3 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_y_axis_l837_83749


namespace NUMINAMATH_GPT_total_cost_correct_l837_83710

-- Conditions given in the problem.
def net_profit : ℝ := 44
def gross_revenue : ℝ := 47
def lemonades_sold : ℝ := 50
def babysitting_income : ℝ := 31

def cost_per_lemon : ℝ := 0.20
def cost_per_sugar : ℝ := 0.15
def cost_per_ice : ℝ := 0.05

def one_time_cost_sunhat : ℝ := 10

-- Definition of variable cost per lemonade.
def variable_cost_per_lemonade : ℝ := cost_per_lemon + cost_per_sugar + cost_per_ice

-- Definition of total variable cost for all lemonades sold.
def total_variable_cost : ℝ := lemonades_sold * variable_cost_per_lemonade

-- Final total cost to operate the lemonade stand.
def total_cost : ℝ := total_variable_cost + one_time_cost_sunhat

-- The proof statement that total cost is equal to $30.
theorem total_cost_correct : total_cost = 30 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l837_83710


namespace NUMINAMATH_GPT_arcsin_of_half_l837_83723

theorem arcsin_of_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_arcsin_of_half_l837_83723


namespace NUMINAMATH_GPT_calc_is_a_pow4_l837_83798

theorem calc_is_a_pow4 (a : ℕ) : (a^2)^2 = a^4 := 
by 
  sorry

end NUMINAMATH_GPT_calc_is_a_pow4_l837_83798


namespace NUMINAMATH_GPT_max_plus_ten_min_eq_zero_l837_83770

theorem max_plus_ten_min_eq_zero (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  let M := max (x * y + x * z + y * z)
  let m := min (x * y + x * z + y * z)
  M + 10 * m = 0 :=
by
  sorry

end NUMINAMATH_GPT_max_plus_ten_min_eq_zero_l837_83770


namespace NUMINAMATH_GPT_complex_magnitude_difference_eq_one_l837_83719

noncomputable def magnitude (z : Complex) : ℝ := Complex.abs z

/-- Lean 4 statement of the problem -/
theorem complex_magnitude_difference_eq_one (z₁ z₂ : Complex) (h₁ : magnitude z₁ = 1) (h₂ : magnitude z₂ = 1) (h₃ : magnitude (z₁ + z₂) = Real.sqrt 3) : magnitude (z₁ - z₂) = 1 := 
sorry

end NUMINAMATH_GPT_complex_magnitude_difference_eq_one_l837_83719


namespace NUMINAMATH_GPT_simplify_expression_l837_83777

theorem simplify_expression (a : ℝ) : 
  ( (a^(16 / 8))^(1 / 4) )^3 * ( (a^(16 / 4))^(1 / 8) )^3 = a^3 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l837_83777


namespace NUMINAMATH_GPT_final_investment_amount_l837_83737

noncomputable def final_amount (P1 P2 : ℝ) (r1 r2 t1 t2 n1 n2 : ℝ) : ℝ :=
  let A1 := P1 * (1 + r1 / n1) ^ (n1 * t1)
  let A2 := (A1 + P2) * (1 + r2 / n2) ^ (n2 * t2)
  A2

theorem final_investment_amount :
  final_amount 6000 2000 0.10 0.08 2 1.5 2 4 = 10467.05 :=
by
  sorry

end NUMINAMATH_GPT_final_investment_amount_l837_83737


namespace NUMINAMATH_GPT_solve_equation_l837_83771

theorem solve_equation {n k l m : ℕ} (h_l : l > 1) :
  (1 + n^k)^l = 1 + n^m ↔ (n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3) :=
sorry

end NUMINAMATH_GPT_solve_equation_l837_83771


namespace NUMINAMATH_GPT_length_of_room_l837_83795

noncomputable def room_length (width cost rate : ℝ) : ℝ :=
  let area := cost / rate
  area / width

theorem length_of_room :
  room_length 4.75 38475 900 = 9 := by
  sorry

end NUMINAMATH_GPT_length_of_room_l837_83795


namespace NUMINAMATH_GPT_range_of_a_condition_l837_83745

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0

theorem range_of_a_condition :
  range_of_a a → -1 < a ∧ a < 3 := sorry

end NUMINAMATH_GPT_range_of_a_condition_l837_83745


namespace NUMINAMATH_GPT_remainder_of_99_pow_36_mod_100_l837_83764

theorem remainder_of_99_pow_36_mod_100 :
  (99 : ℤ)^36 % 100 = 1 := sorry

end NUMINAMATH_GPT_remainder_of_99_pow_36_mod_100_l837_83764


namespace NUMINAMATH_GPT_simplify_expression_l837_83714

theorem simplify_expression :
  (2^8 + 4^5) * (2^3 - (-2)^3)^2 = 327680 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l837_83714


namespace NUMINAMATH_GPT_fraction_subtraction_l837_83766

theorem fraction_subtraction (x : ℚ) : x - (1/5 : ℚ) = (3/5 : ℚ) → x = (4/5 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l837_83766


namespace NUMINAMATH_GPT_longest_side_length_quadrilateral_l837_83717

theorem longest_side_length_quadrilateral :
  (∀ (x y : ℝ),
    (x + y ≤ 4) ∧
    (2 * x + y ≥ 3) ∧
    (x ≥ 0) ∧
    (y ≥ 0)) →
  (∃ d : ℝ, d = 4 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_GPT_longest_side_length_quadrilateral_l837_83717


namespace NUMINAMATH_GPT_find_m_value_l837_83783

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_m_value (m : ℝ) 
  (h : dot_product (2 * m - 1, 3) (1, -1) = 2) : 
  m = 3 := by
  sorry

end NUMINAMATH_GPT_find_m_value_l837_83783


namespace NUMINAMATH_GPT_integer_solutions_no_solutions_2891_l837_83733

-- Define the main problem statement
-- Prove that if the equation x^3 - 3xy^2 + y^3 = n has a solution in integers x, y, then it has at least three such solutions.
theorem integer_solutions (n : ℕ) (x y : ℤ) (h : x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ x₁ y₁ x₂ y₂ : ℤ, x₁ ≠ x ∧ y₁ ≠ y ∧ x₂ ≠ x ∧ y₂ ≠ y ∧ 
  x₁^3 - 3 * x₁ * y₁^2 + y₁^3 = n ∧ 
  x₂^3 - 3 * x₂ * y₂^2 + y₂^3 = n := sorry

-- Prove that if n = 2891 then no such integer solutions exist.
theorem no_solutions_2891 (x y : ℤ) : ¬ (x^3 - 3 * x * y^2 + y^3 = 2891) := sorry

end NUMINAMATH_GPT_integer_solutions_no_solutions_2891_l837_83733


namespace NUMINAMATH_GPT_peanuts_weight_l837_83747

theorem peanuts_weight (total_snacks raisins : ℝ) (h_total : total_snacks = 0.5) (h_raisins : raisins = 0.4) : (total_snacks - raisins) = 0.1 :=
by
  rw [h_total, h_raisins]
  norm_num

end NUMINAMATH_GPT_peanuts_weight_l837_83747


namespace NUMINAMATH_GPT_P_sufficient_for_Q_P_not_necessary_for_Q_l837_83793

variable (x : ℝ)
def P : Prop := x >= 0
def Q : Prop := 2 * x + 1 / (2 * x + 1) >= 1

theorem P_sufficient_for_Q : P x -> Q x := 
by sorry

theorem P_not_necessary_for_Q : ¬ (Q x -> P x) := 
by sorry

end NUMINAMATH_GPT_P_sufficient_for_Q_P_not_necessary_for_Q_l837_83793


namespace NUMINAMATH_GPT_find_a_l837_83757

def A : Set ℝ := {x | x^2 - 2 * x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem find_a (a : ℝ) (h : A ∩ B a = B a) : a = 0 ∨ a = -1 ∨ a = 1/3 := by
  sorry

end NUMINAMATH_GPT_find_a_l837_83757


namespace NUMINAMATH_GPT_sum_of_triangle_angles_l837_83721

theorem sum_of_triangle_angles 
  (smallest largest middle : ℝ) 
  (h1 : smallest = 20) 
  (h2 : middle = 3 * smallest) 
  (h3 : largest = 5 * smallest) 
  (h4 : smallest + middle + largest = 180) :
  smallest + middle + largest = 180 :=
by sorry

end NUMINAMATH_GPT_sum_of_triangle_angles_l837_83721


namespace NUMINAMATH_GPT_valid_license_plates_l837_83728

def letters := 26
def digits := 10
def totalPlates := letters^3 * digits^4

theorem valid_license_plates : totalPlates = 175760000 := by
  sorry

end NUMINAMATH_GPT_valid_license_plates_l837_83728


namespace NUMINAMATH_GPT_chord_length_eq_l837_83774

noncomputable def length_of_chord (radius : ℝ) (distance_to_chord : ℝ) : ℝ :=
  2 * Real.sqrt (radius^2 - distance_to_chord^2)

theorem chord_length_eq {radius distance_to_chord : ℝ} (h_radius : radius = 5) (h_distance : distance_to_chord = 4) :
  length_of_chord radius distance_to_chord = 6 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_eq_l837_83774


namespace NUMINAMATH_GPT_time_to_Lake_Park_restaurant_l837_83716

def time_to_Hidden_Lake := 15
def time_back_to_Park_Office := 7
def total_time_gone := 32

theorem time_to_Lake_Park_restaurant : 
  (total_time_gone = time_to_Hidden_Lake + time_back_to_Park_Office +
  (32 - (time_to_Hidden_Lake + time_back_to_Park_Office))) -> 
  (32 - (time_to_Hidden_Lake + time_back_to_Park_Office) = 10) := by
  intros 
  sorry

end NUMINAMATH_GPT_time_to_Lake_Park_restaurant_l837_83716


namespace NUMINAMATH_GPT_radius_of_given_spherical_circle_l837_83754
noncomputable def circle_radius_spherical_coords : Real :=
  let spherical_to_cartesian (rho theta phi : Real) : (Real × Real × Real) :=
    (rho * (Real.sin phi) * (Real.cos theta), rho * (Real.sin phi) * (Real.sin theta), rho * (Real.cos phi))
  let (rho, theta, phi) := (1, 0, Real.pi / 3)
  let (x, y, z) := spherical_to_cartesian rho theta phi
  let radius := Real.sqrt (x^2 + y^2)
  radius

theorem radius_of_given_spherical_circle :
  circle_radius_spherical_coords = (Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_radius_of_given_spherical_circle_l837_83754


namespace NUMINAMATH_GPT_sum_first_15_odd_integers_l837_83712

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end NUMINAMATH_GPT_sum_first_15_odd_integers_l837_83712


namespace NUMINAMATH_GPT_num_positive_integers_satisfying_condition_l837_83750

theorem num_positive_integers_satisfying_condition :
  ∃! (n : ℕ), 30 - 6 * n > 18 := by
  sorry

end NUMINAMATH_GPT_num_positive_integers_satisfying_condition_l837_83750


namespace NUMINAMATH_GPT_no_digit_C_makes_2C4_multiple_of_5_l837_83779

theorem no_digit_C_makes_2C4_multiple_of_5 : ∀ (C : ℕ), (2 * 100 + C * 10 + 4 ≠ 0 ∨ 2 * 100 + C * 10 + 4 ≠ 5) := 
by 
  intros C
  have h : 4 ≠ 0 := by norm_num
  have h2 : 4 ≠ 5 := by norm_num
  sorry

end NUMINAMATH_GPT_no_digit_C_makes_2C4_multiple_of_5_l837_83779


namespace NUMINAMATH_GPT_kim_gum_distribution_l837_83753

theorem kim_gum_distribution (cousins : ℕ) (total_gum : ℕ) 
  (h1 : cousins = 4) (h2 : total_gum = 20) : 
  total_gum / cousins = 5 :=
by
  sorry

end NUMINAMATH_GPT_kim_gum_distribution_l837_83753


namespace NUMINAMATH_GPT_six_x_plus_four_eq_twenty_two_l837_83706

theorem six_x_plus_four_eq_twenty_two (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 := 
by
  sorry

end NUMINAMATH_GPT_six_x_plus_four_eq_twenty_two_l837_83706


namespace NUMINAMATH_GPT_complement_of_A_in_U_l837_83704

-- Define the universal set U and the subset A
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {1, 2, 5, 7}

-- Define the complement of A with respect to U
def complementU_A : Set Nat := {x ∈ U | x ∉ A}

-- Prove the complement of A in U is {3, 4, 6}
theorem complement_of_A_in_U :
  complementU_A = {3, 4, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l837_83704


namespace NUMINAMATH_GPT_sum_of_digits_divisible_by_six_l837_83730

theorem sum_of_digits_divisible_by_six (A B : ℕ) (h1 : 10 * A + B % 6 = 0) (h2 : A + B = 12) : A + B = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_divisible_by_six_l837_83730


namespace NUMINAMATH_GPT_nuts_division_pattern_l837_83700

noncomputable def smallest_number_of_nuts : ℕ := 15621

theorem nuts_division_pattern :
  ∃ N : ℕ, N = smallest_number_of_nuts ∧ 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → 
  (∃ M : ℕ, (N - k) % 4 = 0 ∧ (N - k) / 4 * 5 + 1 = N) := sorry

end NUMINAMATH_GPT_nuts_division_pattern_l837_83700


namespace NUMINAMATH_GPT_change_in_expression_l837_83797

theorem change_in_expression (x a : ℝ) (h : 0 < a) :
  (x + a)^3 - 3 * (x + a) - (x^3 - 3 * x) = 3 * a * x^2 + 3 * a^2 * x + a^3 - 3 * a
  ∨ (x - a)^3 - 3 * (x - a) - (x^3 - 3 * x) = -3 * a * x^2 + 3 * a^2 * x - a^3 + 3 * a :=
sorry

end NUMINAMATH_GPT_change_in_expression_l837_83797


namespace NUMINAMATH_GPT_smallest_positive_n_l837_83773

theorem smallest_positive_n (n : ℕ) (h : 77 * n ≡ 308 [MOD 385]) : n = 4 :=
sorry

end NUMINAMATH_GPT_smallest_positive_n_l837_83773


namespace NUMINAMATH_GPT_find_y_l837_83742

theorem find_y (y z : ℕ) (h1 : 50 = y * 10) (h2 : 300 = 50 * z) : y = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l837_83742


namespace NUMINAMATH_GPT_lesson_duration_tuesday_l837_83780

theorem lesson_duration_tuesday
  (monday_lessons : ℕ)
  (monday_duration : ℕ)
  (tuesday_lessons : ℕ)
  (wednesday_multiplier : ℕ)
  (total_time : ℕ)
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (H1 : monday_lessons = 6)
  (H2 : monday_duration = 30)
  (H3 : tuesday_lessons = 3)
  (H4 : wednesday_multiplier = 2)
  (H5 : total_time = 12)
  (H6 : monday_hours = monday_lessons * monday_duration / 60)
  (H7 : tuesday_hours = tuesday_lessons * T)
  (H8 : wednesday_hours = wednesday_multiplier * tuesday_hours)
  (H9 : monday_hours + tuesday_hours + wednesday_hours = total_time) :
  T = 1 := by
  sorry

end NUMINAMATH_GPT_lesson_duration_tuesday_l837_83780


namespace NUMINAMATH_GPT_discount_percentage_l837_83732

theorem discount_percentage (x : ℝ) : 
  let marked_price := 12000
  let final_price := 7752
  let second_discount := 0.15
  let third_discount := 0.05
  (marked_price * (1 - x / 100) * (1 - second_discount) * (1 - third_discount) = final_price) ↔ x = 20 := 
by
  let marked_price := 12000
  let final_price := 7752
  let second_discount := 0.15
  let third_discount := 0.05
  sorry

end NUMINAMATH_GPT_discount_percentage_l837_83732


namespace NUMINAMATH_GPT_no_real_solutions_l837_83709

theorem no_real_solutions (x : ℝ) : 
  x^(Real.log x / Real.log 2) ≠ x^4 / 256 :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l837_83709


namespace NUMINAMATH_GPT_unique_pair_prime_m_positive_l837_83756

theorem unique_pair_prime_m_positive (p m : ℕ) (hp : Nat.Prime p) (hm : 0 < m) :
  p * (p + m) + p = (m + 1) ^ 3 → (p = 2 ∧ m = 1) :=
by
  sorry

end NUMINAMATH_GPT_unique_pair_prime_m_positive_l837_83756


namespace NUMINAMATH_GPT_fraction_surface_area_red_l837_83715

theorem fraction_surface_area_red :
  ∀ (num_unit_cubes : ℕ) (side_length_large_cube : ℕ) (total_surface_area_painted : ℕ) (total_surface_area_unit_cubes : ℕ),
    num_unit_cubes = 8 →
    side_length_large_cube = 2 →
    total_surface_area_painted = 6 * (side_length_large_cube ^ 2) →
    total_surface_area_unit_cubes = num_unit_cubes * 6 →
    (total_surface_area_painted : ℝ) / total_surface_area_unit_cubes = 1 / 2 :=
by
  intros num_unit_cubes side_length_large_cube total_surface_area_painted total_surface_area_unit_cubes
  sorry

end NUMINAMATH_GPT_fraction_surface_area_red_l837_83715


namespace NUMINAMATH_GPT_symmetric_circle_eq_a_l837_83792

theorem symmetric_circle_eq_a :
  ∀ (a : ℝ), (∀ x y : ℝ, (x^2 + y^2 - a * x + 2 * y + 1 = 0) ↔ (∃ x y : ℝ, (x - y = 1) ∧ ( x^2 + y^2 = 1))) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_circle_eq_a_l837_83792


namespace NUMINAMATH_GPT_stratified_sampling_freshman_l837_83705

def total_students : ℕ := 1800 + 1500 + 1200
def sample_size : ℕ := 150
def freshman_students : ℕ := 1200

/-- if a sample of 150 students is drawn using stratified sampling, 40 students should be drawn from the freshman year -/
theorem stratified_sampling_freshman :
  (freshman_students * sample_size) / total_students = 40 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_freshman_l837_83705


namespace NUMINAMATH_GPT_range_of_m_l837_83701

open Set

def set_A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def set_B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (3 * m - 2)}

theorem range_of_m (m : ℝ) : (set_B m ⊆ set_A) ↔ m ≤ 4 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l837_83701


namespace NUMINAMATH_GPT_range_of_m_exacts_two_integers_l837_83713

theorem range_of_m_exacts_two_integers (m : ℝ) :
  (∀ x : ℝ, (x - 2) / 4 < (x - 1) / 3 ∧ 2 * x - m ≤ 2 - x) ↔ -2 ≤ m ∧ m < 1 := 
sorry

end NUMINAMATH_GPT_range_of_m_exacts_two_integers_l837_83713


namespace NUMINAMATH_GPT_length_of_platform_is_350_l837_83789

-- Define the parameters as given in the problem
def train_length : ℕ := 300
def time_to_cross_post : ℕ := 18
def time_to_cross_platform : ℕ := 39

-- Define the speed of the train as a ratio of the length of the train and the time to cross the post
def train_speed : ℚ := train_length / time_to_cross_post

-- Formalize the problem statement: Prove that the length of the platform is 350 meters
theorem length_of_platform_is_350 : ∃ (L : ℕ), (train_speed * time_to_cross_platform) = train_length + L := by
  use 350
  sorry

end NUMINAMATH_GPT_length_of_platform_is_350_l837_83789


namespace NUMINAMATH_GPT_A_investment_amount_l837_83765

-- Conditions
variable (B_investment : ℝ) (C_investment : ℝ) (total_profit : ℝ) (A_profit : ℝ)
variable (B_investment_value : B_investment = 4200)
variable (C_investment_value : C_investment = 10500)
variable (total_profit_value : total_profit = 13600)
variable (A_profit_value : A_profit = 4080)

-- Proof statement
theorem A_investment_amount : 
  (∃ x : ℝ, x = 4410) :=
by
  sorry

end NUMINAMATH_GPT_A_investment_amount_l837_83765


namespace NUMINAMATH_GPT_cara_constant_speed_l837_83718

noncomputable def cara_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

theorem cara_constant_speed
  ( distance : ℕ := 120 )
  ( dan_speed : ℕ := 40 )
  ( dan_time_offset : ℕ := 1 ) :
  cara_speed distance (3 + dan_time_offset) = 30 := 
by
  -- skip proof
  sorry

end NUMINAMATH_GPT_cara_constant_speed_l837_83718


namespace NUMINAMATH_GPT_brenda_distance_when_first_met_l837_83702

theorem brenda_distance_when_first_met
  (opposite_points : ∀ (d : ℕ), d = 150) -- Starting at diametrically opposite points on a 300m track means distance is 150m
  (constant_speeds : ∀ (B S x : ℕ), B * x = S * x) -- Brenda/ Sally run at constant speed
  (meet_again : ∀ (d₁ d₂ : ℕ), d₁ + d₂ = 300 + 100) -- Together they run 400 meters when they meet again, additional 100m by Sally
  : ∃ (x : ℕ), x = 150 :=
  by
    sorry

end NUMINAMATH_GPT_brenda_distance_when_first_met_l837_83702


namespace NUMINAMATH_GPT_sum_of_prime_factors_240345_l837_83799

theorem sum_of_prime_factors_240345 : ∀ {p1 p2 p3 : ℕ}, 
  Prime p1 → Prime p2 → Prime p3 →
  p1 * p2 * p3 = 240345 →
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
  p1 + p2 + p3 = 16011 :=
by
  intros p1 p2 p3 hp1 hp2 hp3 hprod hdiff
  sorry

end NUMINAMATH_GPT_sum_of_prime_factors_240345_l837_83799


namespace NUMINAMATH_GPT_last_digit_of_2_pow_2010_l837_83743

-- Define the pattern of last digits of powers of 2
def last_digit_of_power_of_2 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 0 -- This case is redundant as n % 4 ∈ {0, 1, 2, 3}

-- Main theorem stating the problem's assertion
theorem last_digit_of_2_pow_2010 : last_digit_of_power_of_2 2010 = 4 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_last_digit_of_2_pow_2010_l837_83743


namespace NUMINAMATH_GPT_no_integer_solutions_for_x2_minus_4y2_eq_2011_l837_83762

theorem no_integer_solutions_for_x2_minus_4y2_eq_2011 :
  ∀ (x y : ℤ), x^2 - 4 * y^2 ≠ 2011 := by
sorry

end NUMINAMATH_GPT_no_integer_solutions_for_x2_minus_4y2_eq_2011_l837_83762


namespace NUMINAMATH_GPT_cost_of_plastering_l837_83729

/-- 
Let's define the problem conditions
Length of the tank (in meters)
-/
def tank_length : ℕ := 25

/--
Width of the tank (in meters)
-/
def tank_width : ℕ := 12

/--
Depth of the tank (in meters)
-/
def tank_depth : ℕ := 6

/--
Cost of plastering per square meter (55 paise converted to rupees)
-/
def cost_per_sq_meter : ℝ := 0.55

/--
Prove that the cost of plastering the walls and bottom of the tank is 409.2 rupees
-/
theorem cost_of_plastering (total_cost : ℝ) : 
  total_cost = 409.2 :=
sorry

end NUMINAMATH_GPT_cost_of_plastering_l837_83729


namespace NUMINAMATH_GPT_find_solution_l837_83752

-- Define the setup for the problem
variables (k x y : ℝ)

-- Conditions from the problem
def cond1 : Prop := x - y = 9 * k
def cond2 : Prop := x + y = 5 * k
def cond3 : Prop := 2 * x + 3 * y = 8

-- Proof statement combining all conditions to show the values of k, x, and y that satisfy them
theorem find_solution :
  cond1 k x y →
  cond2 k x y →
  cond3 x y →
  k = 1 ∧ x = 7 ∧ y = -2 := by
  sorry

end NUMINAMATH_GPT_find_solution_l837_83752


namespace NUMINAMATH_GPT_steve_total_payment_l837_83711

def mike_dvd_cost : ℝ := 5
def steve_dvd_cost : ℝ := 2 * mike_dvd_cost
def additional_dvd_cost : ℝ := 7
def steve_additional_dvds : ℝ := 2 * additional_dvd_cost
def total_dvd_cost : ℝ := steve_dvd_cost + steve_additional_dvds
def shipping_cost : ℝ := 0.80 * total_dvd_cost
def subtotal_with_shipping : ℝ := total_dvd_cost + shipping_cost
def sales_tax : ℝ := 0.10 * subtotal_with_shipping
def total_amount_paid : ℝ := subtotal_with_shipping + sales_tax

theorem steve_total_payment : total_amount_paid = 47.52 := by
  sorry

end NUMINAMATH_GPT_steve_total_payment_l837_83711


namespace NUMINAMATH_GPT_percent_decrease_in_hours_l837_83755

variable {W H : ℝ} (W_nonzero : W ≠ 0) (H_nonzero : H ≠ 0)

theorem percent_decrease_in_hours
  (wage_increase : W' = 1.25 * W)
  (income_unchanged : W * H = W' * H')
  : (H' = 0.8 * H) → H' = H * (1 - 0.2) := by
  sorry

end NUMINAMATH_GPT_percent_decrease_in_hours_l837_83755


namespace NUMINAMATH_GPT_bala_age_difference_l837_83738

theorem bala_age_difference 
  (a10 : ℕ) -- Anand's age 10 years ago.
  (b10 : ℕ) -- Bala's age 10 years ago.
  (h1 : a10 = b10 / 3) -- 10 years ago, Anand's age was one-third Bala's age.
  (h2 : a10 = 15 - 10) -- Anand was 5 years old 10 years ago, given his current age is 15.
  : (b10 + 10) - 15 = 10 := -- Bala is 10 years older than Anand.
sorry

end NUMINAMATH_GPT_bala_age_difference_l837_83738


namespace NUMINAMATH_GPT_consecutive_odd_natural_numbers_sum_l837_83736

theorem consecutive_odd_natural_numbers_sum (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : b = a + 6) 
  (h4 : c = a + 12) 
  (h5 : c = 27) 
  (h6 : a % 2 = 1) 
  (h7 : b % 2 = 1) 
  (h8 : c % 2 = 1) 
  (h9 : a % 3 = 0) 
  (h10 : b % 3 = 0) 
  (h11 : c % 3 = 0) 
  : a + b + c = 63 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_odd_natural_numbers_sum_l837_83736


namespace NUMINAMATH_GPT_unique_pegboard_arrangement_l837_83735

/-- Conceptually, we will set up a function to count valid arrangements of pegs
based on the given conditions and prove that there is exactly one such arrangement. -/
def triangular_pegboard_arrangements (yellow red green blue orange black : ℕ) : ℕ :=
  if yellow = 6 ∧ red = 5 ∧ green = 4 ∧ blue = 3 ∧ orange = 2 ∧ black = 1 then 1 else 0

theorem unique_pegboard_arrangement :
  triangular_pegboard_arrangements 6 5 4 3 2 1 = 1 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_unique_pegboard_arrangement_l837_83735
