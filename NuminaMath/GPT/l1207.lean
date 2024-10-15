import Mathlib

namespace NUMINAMATH_GPT_sum_of_powers_modulo_l1207_120791

theorem sum_of_powers_modulo (R : Finset ℕ) (S : ℕ) :
  (∀ n < 100, ∃ r, r ∈ R ∧ r = 3^n % 500) →
  S = R.sum id →
  (S % 500) = 0 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_sum_of_powers_modulo_l1207_120791


namespace NUMINAMATH_GPT_digits_difference_l1207_120701

/-- Given a two-digit number represented as 10X + Y and the number obtained by interchanging its digits as 10Y + X,
    if the difference between the original number and the interchanged number is 81, 
    then the difference between the tens digit X and the units digit Y is 9. -/
theorem digits_difference (X Y : ℕ) (h : (10 * X + Y) - (10 * Y + X) = 81) : X - Y = 9 :=
by
  sorry

end NUMINAMATH_GPT_digits_difference_l1207_120701


namespace NUMINAMATH_GPT_graph_inequality_solution_l1207_120709

noncomputable def solution_set : Set (Real × Real) := {
  p | let x := p.1
       let y := p.2
       (y^2 - (Real.arcsin (Real.sin x))^2) *
       (y^2 - (Real.arcsin (Real.sin (x + Real.pi / 3)))^2) *
       (y^2 - (Real.arcsin (Real.sin (x - Real.pi / 3)))^2) < 0
}

theorem graph_inequality_solution
  (x y : ℝ) :
  (y^2 - (Real.arcsin (Real.sin x))^2) *
  (y^2 - (Real.arcsin (Real.sin (x + Real.pi / 3)))^2) *
  (y^2 - (Real.arcsin (Real.sin (x - Real.pi / 3)))^2) < 0 ↔
  (x, y) ∈ solution_set :=
by
  sorry

end NUMINAMATH_GPT_graph_inequality_solution_l1207_120709


namespace NUMINAMATH_GPT_prism_volume_l1207_120704

theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 24) 
  (h2 : x * z = 32) 
  (h3 : y * z = 48) : 
  x * y * z = 192 :=
by
  sorry

end NUMINAMATH_GPT_prism_volume_l1207_120704


namespace NUMINAMATH_GPT_quadratic_inequality_no_real_roots_l1207_120793

theorem quadratic_inequality_no_real_roots (a b c : ℝ) (h : a ≠ 0) (h_Δ : b^2 - 4 * a * c < 0) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_no_real_roots_l1207_120793


namespace NUMINAMATH_GPT_find_fraction_divide_equal_l1207_120728

theorem find_fraction_divide_equal (x : ℚ) : 
  (3 * x = (1 / (5 / 2))) → (x = 2 / 15) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_fraction_divide_equal_l1207_120728


namespace NUMINAMATH_GPT_trig_identity_equiv_l1207_120731

theorem trig_identity_equiv (α : ℝ) (h : Real.sin (Real.pi - α) = -2 * Real.cos (-α)) : 
  Real.sin (2 * α) - Real.cos α ^ 2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_equiv_l1207_120731


namespace NUMINAMATH_GPT_probability_of_observing_change_l1207_120732

noncomputable def traffic_light_cycle := 45 + 5 + 45
noncomputable def observable_duration := 5 + 5 + 5
noncomputable def probability_observe_change := observable_duration / (traffic_light_cycle : ℝ)

theorem probability_of_observing_change :
  probability_observe_change = (3 / 19 : ℝ) :=
  by sorry

end NUMINAMATH_GPT_probability_of_observing_change_l1207_120732


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_S30_l1207_120787

variable (S : ℕ → ℝ)

theorem arithmetic_geometric_sequence_S30 :
  S 10 = 10 →
  S 20 = 30 →
  S 30 = 70 := by
  intros h1 h2
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_S30_l1207_120787


namespace NUMINAMATH_GPT_sqrt_solution_range_l1207_120754

theorem sqrt_solution_range : 
  7 < (Real.sqrt 32) * (Real.sqrt (1 / 2)) + (Real.sqrt 12) ∧ (Real.sqrt 32) * (Real.sqrt (1 / 2)) + (Real.sqrt 12) < 8 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_solution_range_l1207_120754


namespace NUMINAMATH_GPT_sales_proof_valid_l1207_120796

variables (T: ℝ) (Teq: T = 30)
noncomputable def check_sales_proof : Prop :=
  (6.4 * T + 228 = 420)

theorem sales_proof_valid (T : ℝ) (Teq: T = 30) : check_sales_proof T :=
  by
    rw [Teq]
    norm_num
    sorry

end NUMINAMATH_GPT_sales_proof_valid_l1207_120796


namespace NUMINAMATH_GPT_arc_length_of_sector_l1207_120790

theorem arc_length_of_sector (r A l : ℝ) (h_r : r = 2) (h_A : A = π / 3) (h_area : A = 1 / 2 * r * l) : l = π / 3 :=
by
  rw [h_r, h_A] at h_area
  sorry

end NUMINAMATH_GPT_arc_length_of_sector_l1207_120790


namespace NUMINAMATH_GPT_total_wheels_correct_l1207_120784

def total_wheels (bicycles cars motorcycles tricycles quads : ℕ) 
(missing_bicycle_wheels broken_car_wheels missing_motorcycle_wheels : ℕ) : ℕ :=
  let bicycles_wheels := (bicycles - missing_bicycle_wheels) * 2 + missing_bicycle_wheels
  let cars_wheels := (cars - broken_car_wheels) * 4 + broken_car_wheels * 3
  let motorcycles_wheels := (motorcycles - missing_motorcycle_wheels) * 2
  let tricycles_wheels := tricycles * 3
  let quads_wheels := quads * 4
  bicycles_wheels + cars_wheels + motorcycles_wheels + tricycles_wheels + quads_wheels

theorem total_wheels_correct : total_wheels 25 15 8 3 2 5 2 1 = 134 := 
  by sorry

end NUMINAMATH_GPT_total_wheels_correct_l1207_120784


namespace NUMINAMATH_GPT_number_of_blue_socks_l1207_120736

theorem number_of_blue_socks (x : ℕ) (h : ((6 + x ^ 2 - x) / ((6 + x) * (5 + x)) = 1/5)) : x = 4 := 
sorry

end NUMINAMATH_GPT_number_of_blue_socks_l1207_120736


namespace NUMINAMATH_GPT_comprehensive_score_correct_l1207_120720

-- Conditions
def theoreticalWeight : ℝ := 0.20
def designWeight : ℝ := 0.50
def presentationWeight : ℝ := 0.30

def theoreticalScore : ℕ := 95
def designScore : ℕ := 88
def presentationScore : ℕ := 90

-- Calculate comprehensive score
def comprehensiveScore : ℝ :=
  theoreticalScore * theoreticalWeight +
  designScore * designWeight +
  presentationScore * presentationWeight

-- Lean statement to prove the comprehensive score using the conditions
theorem comprehensive_score_correct :
  comprehensiveScore = 90 := 
  sorry

end NUMINAMATH_GPT_comprehensive_score_correct_l1207_120720


namespace NUMINAMATH_GPT_time_to_finish_work_l1207_120786

theorem time_to_finish_work (a b c : ℕ) (h1 : 1/a + 1/9 + 1/18 = 1/4) : a = 12 :=
by
  sorry

end NUMINAMATH_GPT_time_to_finish_work_l1207_120786


namespace NUMINAMATH_GPT_computation_equal_l1207_120725

theorem computation_equal (a b c d : ℕ) (inv : ℚ → ℚ) (mul : ℚ → ℕ → ℚ) : 
  a = 3 → b = 1 → c = 6 → d = 2 → 
  inv ((a^b - d + c^2 + b) : ℚ) * 6 = (3 / 19) := by
  intros ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end NUMINAMATH_GPT_computation_equal_l1207_120725


namespace NUMINAMATH_GPT_minimum_erasures_correct_l1207_120742

open Nat List

-- define a function that checks if a number represented as a list of digits is a palindrome
def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

-- the given problem statement
def given_number := [1, 2, 3, 2, 3, 3, 1, 4]

-- function to find the minimum erasures to make a list a palindrome
noncomputable def min_erasures_to_palindrome (l : List ℕ) : ℕ :=
  sorry -- function implementation skipped

-- the main theorem statement
theorem minimum_erasures_correct : min_erasures_to_palindrome given_number = 3 :=
  sorry

end NUMINAMATH_GPT_minimum_erasures_correct_l1207_120742


namespace NUMINAMATH_GPT_find_natural_n_l1207_120729

theorem find_natural_n (n x y k : ℕ) (h_rel_prime : Nat.gcd x y = 1) (h_k_gt_one : k > 1) (h_eq : 3^n = x^k + y^k) :
  n = 2 := by
  sorry

end NUMINAMATH_GPT_find_natural_n_l1207_120729


namespace NUMINAMATH_GPT_coefficient_of_x2_in_expansion_l1207_120773

def binomial_coefficient (n k : Nat) : Nat := Nat.choose k n

def binomial_term (a x : ℕ) (n r : ℕ) : ℕ :=
  a^(n-r) * binomial_coefficient n r * x^r

theorem coefficient_of_x2_in_expansion : 
  binomial_term 2 1 5 2 = 80 := by sorry

end NUMINAMATH_GPT_coefficient_of_x2_in_expansion_l1207_120773


namespace NUMINAMATH_GPT_ellipse_to_parabola_standard_eq_l1207_120735

theorem ellipse_to_parabola_standard_eq :
  ∀ (x y : ℝ), (x^2 / 25 + y^2 / 16 = 1) → (y^2 = 12 * x) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_to_parabola_standard_eq_l1207_120735


namespace NUMINAMATH_GPT_find_f_l1207_120755

-- Define the conditions
def g (x : ℝ) : ℝ := 2 * x + 3
def f (x : ℝ) : ℝ := g (x + 2)

-- State the theorem
theorem find_f :
  ∀ x : ℝ, f x = 2 * x + 7 :=
by
  sorry

end NUMINAMATH_GPT_find_f_l1207_120755


namespace NUMINAMATH_GPT_part1_part2_l1207_120733

theorem part1 (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) : a^2 + b^2 = 22 :=
sorry

theorem part2 (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) : (a - 2) * (b + 2) = 7 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1207_120733


namespace NUMINAMATH_GPT_sum_squares_inequality_l1207_120776

theorem sum_squares_inequality {a b c : ℝ} 
  (h1 : a > 0)
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := 
sorry

end NUMINAMATH_GPT_sum_squares_inequality_l1207_120776


namespace NUMINAMATH_GPT_arithmetic_sequence_n_l1207_120702

theorem arithmetic_sequence_n (a_n : ℕ → ℕ) (S_n : ℕ) (n : ℕ) 
  (h1 : ∀ i, a_n i = 20 + (i - 1) * (54 - 20) / (n - 1)) 
  (h2 : S_n = 37 * n) 
  (h3 : S_n = 999) : 
  n = 27 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_l1207_120702


namespace NUMINAMATH_GPT_inequality_system_solution_exists_l1207_120714

theorem inequality_system_solution_exists (a : ℝ) : (∃ x : ℝ, x ≥ -1 ∧ 2 * x < a) ↔ a > -2 := 
sorry

end NUMINAMATH_GPT_inequality_system_solution_exists_l1207_120714


namespace NUMINAMATH_GPT_jack_initial_money_l1207_120746

-- Define the cost of one pair of socks
def cost_pair_socks : ℝ := 9.50

-- Define the cost of soccer shoes
def cost_soccer_shoes : ℝ := 92

-- Define the additional money Jack needs
def additional_money_needed : ℝ := 71

-- Define the total cost of two pairs of socks and one pair of soccer shoes
def total_cost : ℝ := 2 * cost_pair_socks + cost_soccer_shoes

-- Theorem to prove Jack's initial money
theorem jack_initial_money : ∃ m : ℝ, total_cost - additional_money_needed = 40 :=
by
  sorry

end NUMINAMATH_GPT_jack_initial_money_l1207_120746


namespace NUMINAMATH_GPT_range_of_m_l1207_120749

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (m^2 + 4 * m - 5) * x^2 - 4 * (m - 1) * x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1207_120749


namespace NUMINAMATH_GPT_foci_of_ellipse_l1207_120752

-- Define the ellipsis
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 25) = 1

-- Prove the coordinates of foci of the ellipse
theorem foci_of_ellipse :
  ∃ c : ℝ, c = 3 ∧ ((0, c) ∈ {p : ℝ × ℝ | ellipse p.1 p.2} ∧ (0, -c) ∈ {p : ℝ × ℝ | ellipse p.1 p.2}) :=
by
  sorry

end NUMINAMATH_GPT_foci_of_ellipse_l1207_120752


namespace NUMINAMATH_GPT_solution_set_for_inequality_l1207_120771

theorem solution_set_for_inequality : {x : ℝ | x ≠ 0 ∧ (x-1)/x ≤ 0} = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l1207_120771


namespace NUMINAMATH_GPT_angle_in_second_quadrant_l1207_120759

theorem angle_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : (2 * Real.tan (α / 2)) / (1 - (Real.tan (α / 2))^2) < 0) : 
  ∃ q, q = 2 ∧ α ∈ {α | 0 < α ∧ α < π} :=
by
  sorry

end NUMINAMATH_GPT_angle_in_second_quadrant_l1207_120759


namespace NUMINAMATH_GPT_missed_angle_l1207_120760

theorem missed_angle (sum_calculated : ℕ) (missed_angle_target : ℕ) 
  (h1 : sum_calculated = 2843) 
  (h2 : missed_angle_target = 37) : 
  ∃ n : ℕ, (sum_calculated + missed_angle_target = n * 180) :=
by {
  sorry
}

end NUMINAMATH_GPT_missed_angle_l1207_120760


namespace NUMINAMATH_GPT_math_problem_l1207_120789

variables {a b : ℝ}
open Real

theorem math_problem (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (a - 1) * (b - 1) = 1 ∧ 
  (∀ b : ℝ, (a = 2 * b → a + 4 * b = 9)) ∧ 
  (∀ b : ℝ, (b = 3 → (1 / a^2 + 2 / b^2) = 2 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1207_120789


namespace NUMINAMATH_GPT_zero_of_f_inequality_l1207_120718

noncomputable def f (x : ℝ) : ℝ := 2^(-x) - Real.log (x^3 + 1)

variable (a b c x : ℝ)
variable (h : 0 < a ∧ a < b ∧ b < c)
variable (hx : f x = 0)
variable (h₀ : f a * f b * f c < 0)

theorem zero_of_f_inequality :
  ¬ (x > c) :=
by 
  sorry

end NUMINAMATH_GPT_zero_of_f_inequality_l1207_120718


namespace NUMINAMATH_GPT_line_in_slope_intercept_form_l1207_120739

variable (x y : ℝ)

def line_eq (x y : ℝ) : Prop :=
  (3 : ℝ) * (x - 2) - (4 : ℝ) * (y + 1) = 0

theorem line_in_slope_intercept_form (x y : ℝ) (h: line_eq x y) :
  y = (3 / 4) * x - 5 / 2 :=
sorry

end NUMINAMATH_GPT_line_in_slope_intercept_form_l1207_120739


namespace NUMINAMATH_GPT_food_sufficient_days_l1207_120757

theorem food_sufficient_days (D : ℕ) (h1 : 1000 * D - 10000 = 800 * D) : D = 50 :=
sorry

end NUMINAMATH_GPT_food_sufficient_days_l1207_120757


namespace NUMINAMATH_GPT_ratio_costs_equal_l1207_120756

noncomputable def cost_first_8_years : ℝ := 10000 * 8
noncomputable def john_share_first_8_years : ℝ := cost_first_8_years / 2
noncomputable def university_tuition : ℝ := 250000
noncomputable def john_share_university : ℝ := university_tuition / 2
noncomputable def total_paid_by_john : ℝ := 265000
noncomputable def cost_between_8_and_18 : ℝ := total_paid_by_john - john_share_first_8_years - john_share_university
noncomputable def cost_per_year_8_to_18 : ℝ := cost_between_8_and_18 / 10
noncomputable def cost_per_year_first_8_years : ℝ := 10000

theorem ratio_costs_equal : cost_per_year_8_to_18 / cost_per_year_first_8_years = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_costs_equal_l1207_120756


namespace NUMINAMATH_GPT_juniors_more_than_seniors_l1207_120798

theorem juniors_more_than_seniors
  (j s : ℕ)
  (h1 : (1 / 3) * j = (2 / 3) * s)
  (h2 : j + s = 300) :
  j - s = 100 := 
sorry

end NUMINAMATH_GPT_juniors_more_than_seniors_l1207_120798


namespace NUMINAMATH_GPT_cos_sum_zero_l1207_120768

noncomputable def cos_sum : ℂ :=
  Real.cos (Real.pi / 15) + Real.cos (4 * Real.pi / 15) + Real.cos (7 * Real.pi / 15) + Real.cos (10 * Real.pi / 15)

theorem cos_sum_zero : cos_sum = 0 := by
  sorry

end NUMINAMATH_GPT_cos_sum_zero_l1207_120768


namespace NUMINAMATH_GPT_geometric_sequence_sum_ratio_l1207_120770

noncomputable def a (n : ℕ) (a1 q : ℝ) : ℝ :=
  a1 * q^n

-- Sum of the first 'n' terms of a geometric sequence
noncomputable def S (n : ℕ) (a1 q : ℝ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a1 q : ℝ) 
  (h : 8 * (a 11 a1 q) = (a 14 a1 q)) :
  (S 4 a1 q) / (S 2 a1 q) = 5 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_ratio_l1207_120770


namespace NUMINAMATH_GPT_clock_equiv_l1207_120750

theorem clock_equiv (h : ℕ) (h_gt_6 : h > 6) : h ≡ h^2 [MOD 12] ∧ h ≡ h^3 [MOD 12] → h = 9 :=
by
  sorry

end NUMINAMATH_GPT_clock_equiv_l1207_120750


namespace NUMINAMATH_GPT_average_first_n_numbers_eq_10_l1207_120763

theorem average_first_n_numbers_eq_10 (n : ℕ) 
  (h : (n * (n + 1)) / (2 * n) = 10) : n = 19 :=
  sorry

end NUMINAMATH_GPT_average_first_n_numbers_eq_10_l1207_120763


namespace NUMINAMATH_GPT_inequality_solution_set_l1207_120726

theorem inequality_solution_set (x : ℝ) : |x - 5| + |x + 3| ≤ 10 ↔ -4 ≤ x ∧ x ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1207_120726


namespace NUMINAMATH_GPT_line_through_points_l1207_120730

variable (A1 B1 A2 B2 : ℝ)

def line1 : Prop := -7 * A1 + 9 * B1 = 1
def line2 : Prop := -7 * A2 + 9 * B2 = 1

theorem line_through_points (h1 : line1 A1 B1) (h2 : line1 A2 B2) :
  ∃ (k : ℝ), (∀ (x y : ℝ), y - B1 = k * (x - A1)) ∧ (-7 * (x : ℝ) + 9 * y = 1) := 
by sorry

end NUMINAMATH_GPT_line_through_points_l1207_120730


namespace NUMINAMATH_GPT_optionA_incorrect_optionB_incorrect_optionC_incorrect_optionD_correct_l1207_120724

theorem optionA_incorrect (a x : ℝ) : 3 * a * x^2 - 6 * a * x ≠ 3 * (a * x^2 - 2 * a * x) :=
by sorry

theorem optionB_incorrect (a x : ℝ) : (x + a) * (x - a) ≠ x^2 - a^2 :=
by sorry

theorem optionC_incorrect (a b : ℝ) : a^2 + 2 * a * b - 4 * b^2 ≠ (a + 2 * b)^2 :=
by sorry

theorem optionD_correct (a x : ℝ) : -a * x^2 + 2 * a * x - a = -a * (x - 1)^2 :=
by sorry

end NUMINAMATH_GPT_optionA_incorrect_optionB_incorrect_optionC_incorrect_optionD_correct_l1207_120724


namespace NUMINAMATH_GPT_distinct_license_plates_l1207_120778

theorem distinct_license_plates :
  let num_digits := 10
  let num_letters := 26
  let num_digit_positions := 5
  let num_letter_pairs := num_letters * num_letters
  let num_letter_positions := num_digit_positions + 1
  num_digits^num_digit_positions * num_letter_pairs * num_letter_positions = 40560000 := by
  sorry

end NUMINAMATH_GPT_distinct_license_plates_l1207_120778


namespace NUMINAMATH_GPT_number_that_divides_and_leaves_remainder_54_l1207_120781

theorem number_that_divides_and_leaves_remainder_54 :
  ∃ n : ℕ, n > 0 ∧ (55 ^ 55 + 55) % n = 54 ∧ n = 56 :=
by
  sorry

end NUMINAMATH_GPT_number_that_divides_and_leaves_remainder_54_l1207_120781


namespace NUMINAMATH_GPT_angle_problem_l1207_120710

theorem angle_problem (θ : ℝ) (h1 : 90 - θ = 0.4 * (180 - θ)) (h2 : 180 - θ = 2 * θ) : θ = 30 :=
by
  sorry

end NUMINAMATH_GPT_angle_problem_l1207_120710


namespace NUMINAMATH_GPT_binary_arithmetic_l1207_120777

theorem binary_arithmetic :
  (110010:ℕ) * (1100:ℕ) / (100:ℕ) / (10:ℕ) = 100100 :=
by sorry

end NUMINAMATH_GPT_binary_arithmetic_l1207_120777


namespace NUMINAMATH_GPT_sam_wins_l1207_120795

variable (p : ℚ) -- p is the probability that Sam wins
variable (phit : ℚ) -- probability of hitting the target in one shot
variable (pmiss : ℚ) -- probability of missing the target in one shot

-- Define the problem and set up the conditions
def conditions : Prop := phit = 2 / 5 ∧ pmiss = 3 / 5

-- Define the equation derived from the problem
def equation (p : ℚ) (phit : ℚ) (pmiss : ℚ) : Prop :=
  p = phit + (pmiss * pmiss * p)

-- State the theorem that Sam wins with probability 5/8
theorem sam_wins (h : conditions phit pmiss) : 
  equation p phit pmiss → p = 5 / 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sam_wins_l1207_120795


namespace NUMINAMATH_GPT_radius_of_surrounding_circles_is_correct_l1207_120767

noncomputable def r : Real := 1 + Real.sqrt 2

theorem radius_of_surrounding_circles_is_correct (r: ℝ)
  (h₁: ∃c : ℝ, c = 2) -- central circle radius is 2
  (h₂: ∃far: ℝ, far = (1 + (Real.sqrt 2))) -- r is the solution as calculated
: 2 * r = 1 + Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_surrounding_circles_is_correct_l1207_120767


namespace NUMINAMATH_GPT_sum_arithmetic_seq_l1207_120785

theorem sum_arithmetic_seq (a d n : ℕ) :
  a = 2 → d = 2 → a + (n - 1) * d = 20 → (n / 2) * (a + (a + (n - 1) * d)) = 110 :=
by sorry

end NUMINAMATH_GPT_sum_arithmetic_seq_l1207_120785


namespace NUMINAMATH_GPT_log_condition_necessary_not_sufficient_l1207_120738

noncomputable def base_of_natural_logarithm := Real.exp 1

variable (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 0 < b) (h4 : b ≠ 1)

theorem log_condition_necessary_not_sufficient (h : 0 < a ∧ a < b ∧ b < 1) :
  (Real.log 2 / Real.log a > Real.log base_of_natural_logarithm / Real.log b) :=
sorry

end NUMINAMATH_GPT_log_condition_necessary_not_sufficient_l1207_120738


namespace NUMINAMATH_GPT_value_of_x_squared_plus_reciprocal_squared_l1207_120703

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (h : x^4 + (1 / x^4) = 23) :
  x^2 + (1 / x^2) = 5 := by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_reciprocal_squared_l1207_120703


namespace NUMINAMATH_GPT_map_float_time_l1207_120799

theorem map_float_time
  (t₀ t₁ : Nat) -- times representing 12:00 PM and 12:21 PM in minutes since midnight
  (v_w v_b : ℝ) -- constant speed of water current and boat in still water
  (h₀ : t₀ = 12 * 60) -- t₀ is 12:00 PM
  (h₁ : t₁ = 12 * 60 + 21) -- t₁ is 12:21 PM
  : t₁ - t₀ = 21 := 
  sorry

end NUMINAMATH_GPT_map_float_time_l1207_120799


namespace NUMINAMATH_GPT_first_player_wins_l1207_120706

-- Define the game state and requirements
inductive Player
| first : Player
| second : Player

-- Game state consists of a number of stones and whose turn it is
structure GameState where
  stones : Nat
  player : Player

-- Define a simple transition for the game
def take_stones (s : GameState) (n : Nat) : GameState :=
  { s with stones := s.stones - n, player := Player.second }

-- Determine if a player can take n stones
def can_take (s : GameState) (n : Nat) : Prop :=
  n >= 1 ∧ n <= 4 ∧ n <= s.stones

-- Define victory condition
def wins (s : GameState) : Prop :=
  s.stones = 0 ∧ s.player = Player.second

-- Prove that if the first player starts with 18 stones and picks 3 stones initially,
-- they can ensure victory
theorem first_player_wins :
  ∀ (s : GameState),
    s.stones = 18 ∧ s.player = Player.first →
    can_take s 3 →
    wins (take_stones s 3)
:= by
  sorry

end NUMINAMATH_GPT_first_player_wins_l1207_120706


namespace NUMINAMATH_GPT_abs_expression_equals_one_l1207_120745

theorem abs_expression_equals_one : 
  abs (abs (-(abs (2 - 3)) + 2) - 2) = 1 := 
  sorry

end NUMINAMATH_GPT_abs_expression_equals_one_l1207_120745


namespace NUMINAMATH_GPT_largest_cuts_9x9_l1207_120782

theorem largest_cuts_9x9 (k : ℕ) (V E F : ℕ) (hV : V = 81) (hE : E = 4 * k) (hF : F = 1 + 2 * k)
  (hEuler : V - E + F ≥ 2) : k ≤ 21 :=
by
  sorry

end NUMINAMATH_GPT_largest_cuts_9x9_l1207_120782


namespace NUMINAMATH_GPT_algebraic_expression_value_l1207_120753

theorem algebraic_expression_value (x y : ℝ) (h1 : x + 2 * y = 4) (h2 : x - 2 * y = -1) : 
  x^2 - 4 * y^2 = -4 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1207_120753


namespace NUMINAMATH_GPT_quadratic_roots_l1207_120707

theorem quadratic_roots (x : ℝ) (h : x^2 - 1 = 3) : x = 2 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1207_120707


namespace NUMINAMATH_GPT_range_of_m_l1207_120741

theorem range_of_m (x m : ℝ) (h1 : (2 * x + m) / (x - 1) = 1) (h2 : x ≥ 0) : m ≤ -1 ∧ m ≠ -2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1207_120741


namespace NUMINAMATH_GPT_loss_per_metre_proof_l1207_120705

-- Define the given conditions
def cost_price_per_metre : ℕ := 66
def quantity_sold : ℕ := 200
def total_selling_price : ℕ := 12000

-- Define total cost price based on cost price per metre and quantity sold
def total_cost_price : ℕ := cost_price_per_metre * quantity_sold

-- Define total loss based on total cost price and total selling price
def total_loss : ℕ := total_cost_price - total_selling_price

-- Define loss per metre
def loss_per_metre : ℕ := total_loss / quantity_sold

-- The theorem we need to prove:
theorem loss_per_metre_proof : loss_per_metre = 6 :=
  by
    sorry

end NUMINAMATH_GPT_loss_per_metre_proof_l1207_120705


namespace NUMINAMATH_GPT_set_of_a_where_A_subset_B_l1207_120737

variable {a x : ℝ}

theorem set_of_a_where_A_subset_B (h : ∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) :
  6 ≤ a ∧ a ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_set_of_a_where_A_subset_B_l1207_120737


namespace NUMINAMATH_GPT_no_valid_pairs_l1207_120711

theorem no_valid_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ¬ (a * b + 82 = 25 * Nat.lcm a b + 15 * Nat.gcd a b) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_valid_pairs_l1207_120711


namespace NUMINAMATH_GPT_find_x_parallel_vectors_l1207_120762

theorem find_x_parallel_vectors :
  ∀ x : ℝ, (∃ k : ℝ, (1, 2) = (k * (2 * x), k * (-3))) → x = -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_parallel_vectors_l1207_120762


namespace NUMINAMATH_GPT_hall_length_l1207_120743

variable (breadth length : ℝ)

def condition1 : Prop := length = breadth + 5
def condition2 : Prop := length * breadth = 750

theorem hall_length : condition1 breadth length ∧ condition2 breadth length → length = 30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_hall_length_l1207_120743


namespace NUMINAMATH_GPT_line_length_400_l1207_120721

noncomputable def length_of_line (speed_march_kmh speed_run_kmh total_time_min: ℝ) : ℝ :=
  let speed_march_mpm := (speed_march_kmh * 1000) / 60
  let speed_run_mpm := (speed_run_kmh * 1000) / 60
  let len_eq := 1 / (speed_run_mpm - speed_march_mpm) + 1 / (speed_run_mpm + speed_march_mpm)
  (total_time_min * 200 * len_eq) * 400 / len_eq

theorem line_length_400 :
  length_of_line 8 12 7.2 = 400 := by
  sorry

end NUMINAMATH_GPT_line_length_400_l1207_120721


namespace NUMINAMATH_GPT_Ariella_total_amount_l1207_120772

-- We define the conditions
def Daniella_initial (daniella_amount : ℝ) := daniella_amount = 400
def Ariella_initial (daniella_amount : ℝ) (ariella_amount : ℝ) := ariella_amount = daniella_amount + 200
def simple_interest_rate : ℝ := 0.10
def investment_period : ℕ := 2

-- We state the goal to prove
theorem Ariella_total_amount (daniella_amount ariella_amount : ℝ) :
  Daniella_initial daniella_amount →
  Ariella_initial daniella_amount ariella_amount →
  ariella_amount + ariella_amount * simple_interest_rate * (investment_period : ℝ) = 720 :=
by
  sorry

end NUMINAMATH_GPT_Ariella_total_amount_l1207_120772


namespace NUMINAMATH_GPT_circle_representation_l1207_120775

theorem circle_representation (m : ℝ) : 
  (∃ (x y : ℝ), (x^2 + y^2 + x + 2*m*y + m = 0)) → m ≠ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_circle_representation_l1207_120775


namespace NUMINAMATH_GPT_children_absent_l1207_120744

theorem children_absent (A : ℕ) (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ) :
  total_children = 660 →
  bananas_per_child = 2 →
  extra_bananas_per_child = 2 →
  (total_children * bananas_per_child) = 1320 →
  ((total_children - A) * (bananas_per_child + extra_bananas_per_child)) = 1320 →
  A = 330 :=
by
  intros
  sorry

end NUMINAMATH_GPT_children_absent_l1207_120744


namespace NUMINAMATH_GPT_solution_to_largest_four_digit_fulfilling_conditions_l1207_120717

def largest_four_digit_fulfilling_conditions : Prop :=
  ∃ (N : ℕ), N < 10000 ∧ N ≡ 2 [MOD 11] ∧ N ≡ 4 [MOD 7] ∧ N = 9979

theorem solution_to_largest_four_digit_fulfilling_conditions : largest_four_digit_fulfilling_conditions :=
  sorry

end NUMINAMATH_GPT_solution_to_largest_four_digit_fulfilling_conditions_l1207_120717


namespace NUMINAMATH_GPT_incorrect_statement_about_zero_l1207_120734

theorem incorrect_statement_about_zero :
  ¬ (0 > 0) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_about_zero_l1207_120734


namespace NUMINAMATH_GPT_remaining_surface_area_correct_l1207_120783

open Real

-- Define the original cube and the corner cubes
def orig_cube : ℝ × ℝ × ℝ := (5, 5, 5)
def corner_cube : ℝ × ℝ × ℝ := (2, 2, 2)

-- Define a function to compute the surface area of a cube given dimensions (a, b, c)
def surface_area (a b c : ℝ) : ℝ := 2 * (a * b + b * c + c * a)

-- Original surface area of the cube
def orig_surface_area : ℝ := surface_area 5 5 5

-- Total surface area of the remaining figure after removing 8 corner cubes
def remaining_surface_area : ℝ := 150  -- Calculated directly as 6 * 25

-- Theorem stating that the surface area of the remaining figure is 150 cm^2
theorem remaining_surface_area_correct :
  remaining_surface_area = 150 := sorry

end NUMINAMATH_GPT_remaining_surface_area_correct_l1207_120783


namespace NUMINAMATH_GPT_min_sum_distinct_positive_integers_l1207_120765

theorem min_sum_distinct_positive_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : (1 / a + 1 / b = k1 * (1 / c)) ∧ (1 / a + 1 / c = k2 * (1 / b)) ∧ (1 / b + 1 / c = k3 * (1 / a))) :
  a + b + c ≥ 11 :=
sorry

end NUMINAMATH_GPT_min_sum_distinct_positive_integers_l1207_120765


namespace NUMINAMATH_GPT_train_length_l1207_120713

open Real

/--
A train of a certain length can cross an electric pole in 30 sec with a speed of 43.2 km/h.
Prove that the length of the train is 360 meters.
-/
theorem train_length (t : ℝ) (v_kmh : ℝ) (length : ℝ) 
  (h_time : t = 30) 
  (h_speed_kmh : v_kmh = 43.2) 
  (h_length : length = v_kmh * (t * (1000 / 3600))) : 
  length = 360 := 
by
  -- skip the actual proof steps
  sorry

end NUMINAMATH_GPT_train_length_l1207_120713


namespace NUMINAMATH_GPT_average_marks_l1207_120769

theorem average_marks (total_students : ℕ) (first_group : ℕ) (first_group_marks : ℕ)
                      (second_group : ℕ) (second_group_marks_diff : ℕ) (third_group_marks : ℕ)
                      (total_marks : ℕ) (class_average : ℕ) :
  total_students = 50 → 
  first_group = 10 → 
  first_group_marks = 90 → 
  second_group = 15 → 
  second_group_marks_diff = 10 → 
  third_group_marks = 60 →
  total_marks = (first_group * first_group_marks) + (second_group * (first_group_marks - second_group_marks_diff)) + ((total_students - (first_group + second_group)) * third_group_marks) →
  class_average = total_marks / total_students →
  class_average = 72 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_marks_l1207_120769


namespace NUMINAMATH_GPT_combinations_sol_eq_l1207_120715

theorem combinations_sol_eq (x : ℕ) (h : Nat.choose 10 x = Nat.choose 10 (3 * x - 2)) : x = 1 ∨ x = 3 := sorry

end NUMINAMATH_GPT_combinations_sol_eq_l1207_120715


namespace NUMINAMATH_GPT_fourth_quadrant_for_m_negative_half_x_axis_for_m_upper_half_plane_for_m_l1207_120723

open Complex

def inFourthQuadrant (m : ℝ) : Prop :=
  (m^2 - 8*m + 15) > 0 ∧ (m^2 + 3*m - 28) < 0

def onNegativeHalfXAxis (m : ℝ) : Prop :=
  (m^2 - 8*m + 15) < 0 ∧ (m^2 + 3*m - 28) = 0

def inUpperHalfPlaneIncludingRealAxis (m : ℝ) : Prop :=
  (m^2 + 3*m - 28) ≥ 0

theorem fourth_quadrant_for_m (m : ℝ) :
  (-7 < m ∧ m < 3) ↔ inFourthQuadrant m := 
sorry

theorem negative_half_x_axis_for_m (m : ℝ) :
  (m = 4) ↔ onNegativeHalfXAxis m :=
sorry

theorem upper_half_plane_for_m (m : ℝ) :
  (m ≤ -7 ∨ m ≥ 4) ↔ inUpperHalfPlaneIncludingRealAxis m :=
sorry

end NUMINAMATH_GPT_fourth_quadrant_for_m_negative_half_x_axis_for_m_upper_half_plane_for_m_l1207_120723


namespace NUMINAMATH_GPT_checkerboard_pattern_exists_l1207_120722

-- Definitions for the given conditions
def is_black_white_board (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i j, i < n ∧ j < n → (board (i, j) = true ∨ board (i, j) = false)

def boundary_cells_black (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i, (i < n → (board (i, 0) = true ∧ board (i, n-1) = true ∧ 
                  board (0, i) = true ∧ board (n-1, i) = true))

def no_monochromatic_square (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i j, i < n-1 ∧ j < n-1 → ¬(board (i, j) = board (i+1, j) ∧ 
                               board (i, j) = board (i, j+1) ∧ 
                               board (i, j) = board (i+1, j+1))

def exists_checkerboard_2x2 (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∃ i j, i < n-1 ∧ j < n-1 ∧ 
         (board (i, j) ≠ board (i+1, j) ∧ board (i, j) ≠ board (i, j+1) ∧ 
          board (i+1, j) ≠ board (i+1, j+1) ∧ board (i, j+1) ≠ board (i+1, j+1))

-- The theorem statement
theorem checkerboard_pattern_exists (board : ℕ × ℕ → Prop) (n : ℕ) 
  (coloring : is_black_white_board board n)
  (boundary_black : boundary_cells_black board n)
  (no_mono_2x2 : no_monochromatic_square board n) : 
  exists_checkerboard_2x2 board n :=
by
  sorry

end NUMINAMATH_GPT_checkerboard_pattern_exists_l1207_120722


namespace NUMINAMATH_GPT_total_gas_cost_l1207_120766

def gas_price_station_1 : ℝ := 3
def gas_price_station_2 : ℝ := 3.5
def gas_price_station_3 : ℝ := 4
def gas_price_station_4 : ℝ := 4.5
def tank_capacity : ℝ := 12

theorem total_gas_cost :
  let cost_station_1 := tank_capacity * gas_price_station_1
  let cost_station_2 := tank_capacity * gas_price_station_2
  let cost_station_3 := tank_capacity * gas_price_station_3
  let cost_station_4 := tank_capacity * gas_price_station_4
  cost_station_1 + cost_station_2 + cost_station_3 + cost_station_4 = 180 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_total_gas_cost_l1207_120766


namespace NUMINAMATH_GPT_marble_158th_is_gray_l1207_120797

def marble_color (n : ℕ) : String :=
  if (n % 12 < 5) then "gray"
  else if (n % 12 < 9) then "white"
  else "black"

theorem marble_158th_is_gray : marble_color 157 = "gray" := 
by
  sorry

end NUMINAMATH_GPT_marble_158th_is_gray_l1207_120797


namespace NUMINAMATH_GPT_avg_salary_feb_mar_apr_may_l1207_120716

def avg_salary_4_months : ℝ := 8000
def salary_jan : ℝ := 3700
def salary_may : ℝ := 6500
def total_salary_4_months := 4 * avg_salary_4_months
def total_salary_feb_mar_apr := total_salary_4_months - salary_jan
def total_salary_feb_mar_apr_may := total_salary_feb_mar_apr + salary_may

theorem avg_salary_feb_mar_apr_may : total_salary_feb_mar_apr_may / 4 = 8700 := by
  sorry

end NUMINAMATH_GPT_avg_salary_feb_mar_apr_may_l1207_120716


namespace NUMINAMATH_GPT_determine_f_l1207_120719

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom f_eq (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem determine_f (x : ℝ) : f x = x + 1 := by
  sorry

end NUMINAMATH_GPT_determine_f_l1207_120719


namespace NUMINAMATH_GPT_equation_of_line_l1207_120792

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1 * v.1 + v.2 * v.2
  (dot_product / norm_sq * v.1, dot_product / norm_sq * v.2)

theorem equation_of_line (x y : ℝ) :
  projection (x, y) (7, 3) = (-7, -3) →
  y = -7/3 * x - 58/3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_equation_of_line_l1207_120792


namespace NUMINAMATH_GPT_smallest_base10_integer_l1207_120740

theorem smallest_base10_integer :
  ∃ (n A B : ℕ), 
    (A < 5) ∧ (B < 7) ∧ 
    (n = 6 * A) ∧ 
    (n = 8 * B) ∧ 
    n = 24 := 
sorry

end NUMINAMATH_GPT_smallest_base10_integer_l1207_120740


namespace NUMINAMATH_GPT_bonnets_difference_thursday_monday_l1207_120748

variable (Bm Bt Bf : ℕ)

-- Conditions
axiom monday_bonnets_made : Bm = 10
axiom tuesday_wednesday_bonnets_made : Bm + (2 * Bm) = 30
axiom bonnets_sent_to_orphanages : (Bm + Bt + (Bt - 5) + Bm + (2 * Bm)) / 5 = 11
axiom friday_bonnets_made : Bf = Bt - 5

theorem bonnets_difference_thursday_monday :
  Bt - Bm = 5 :=
sorry

end NUMINAMATH_GPT_bonnets_difference_thursday_monday_l1207_120748


namespace NUMINAMATH_GPT_find_value_l1207_120758

noncomputable def f : ℝ → ℝ := sorry

def tangent_line (x y : ℝ) : Prop := x + 2 * y + 1 = 0

def has_tangent_at (f : ℝ → ℝ) (x0 : ℝ) (L : ℝ → ℝ → Prop) : Prop :=
  L x0 (f x0)

theorem find_value (h : has_tangent_at f 2 tangent_line) :
  f 2 - 2 * (deriv f 2) = -1/2 :=
sorry

end NUMINAMATH_GPT_find_value_l1207_120758


namespace NUMINAMATH_GPT_correct_statements_l1207_120708

variable (P Q : Prop)

-- Define statements
def is_neg_false_if_orig_true := (P → ¬P) = False
def is_converse_not_nec_true_if_orig_true := (P → Q) → ¬(Q → P)
def is_neg_true_if_converse_true := (Q → P) → (¬P → ¬Q)
def is_neg_true_if_contrapositive_true := (¬Q → ¬P) → (¬P → False)

-- Main proposition
theorem correct_statements : 
  is_converse_not_nec_true_if_orig_true P Q ∧ 
  is_neg_true_if_converse_true P Q :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l1207_120708


namespace NUMINAMATH_GPT_sugar_at_home_l1207_120712

-- Definitions based on conditions
def bags_of_sugar := 2
def cups_per_bag := 6
def cups_for_batter_per_12_cupcakes := 1
def cups_for_frosting_per_12_cupcakes := 2
def dozens_of_cupcakes := 5

-- Calculation of total sugar needed and bought, in terms of definitions
def total_cupcakes := dozens_of_cupcakes * 12
def total_sugar_needed_for_batter := (total_cupcakes / 12) * cups_for_batter_per_12_cupcakes
def total_sugar_needed_for_frosting := dozens_of_cupcakes * cups_for_frosting_per_12_cupcakes
def total_sugar_needed := total_sugar_needed_for_batter + total_sugar_needed_for_frosting
def total_sugar_bought := bags_of_sugar * cups_per_bag

-- The statement to be proven in Lean
theorem sugar_at_home : total_sugar_needed - total_sugar_bought = 3 := by
  sorry

end NUMINAMATH_GPT_sugar_at_home_l1207_120712


namespace NUMINAMATH_GPT_number_in_interval_l1207_120788

def number := 0.2012
def lower_bound := 0.2
def upper_bound := 0.25

theorem number_in_interval : lower_bound < number ∧ number < upper_bound :=
by
  sorry

end NUMINAMATH_GPT_number_in_interval_l1207_120788


namespace NUMINAMATH_GPT_sale_in_third_month_l1207_120779

theorem sale_in_third_month (
  f1 f2 f4 f5 f6 average : ℕ
) (h1 : f1 = 7435) 
  (h2 : f2 = 7927) 
  (h4 : f4 = 8230) 
  (h5 : f5 = 7562) 
  (h6 : f6 = 5991) 
  (havg : average = 7500) :
  ∃ f3, f3 = 7855 ∧ f1 + f2 + f3 + f4 + f5 + f6 = average * 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_sale_in_third_month_l1207_120779


namespace NUMINAMATH_GPT_hyperbola_asymptote_l1207_120774

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 9 = 1 ∧ ∀ (x y : ℝ), (y = 3/5 * x ↔ y = 3 / 5 * x)) → a = 5 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l1207_120774


namespace NUMINAMATH_GPT_ratio_of_segments_l1207_120761

theorem ratio_of_segments (a b x : ℝ) (h₁ : a = 9 * x) (h₂ : b = 99 * x) : b / a = 11 := by
  sorry

end NUMINAMATH_GPT_ratio_of_segments_l1207_120761


namespace NUMINAMATH_GPT_min_n_constant_term_l1207_120747

theorem min_n_constant_term (x : ℕ) (hx : x > 0) : 
  ∃ n : ℕ, 
  (∀ r : ℕ, (2 * n = 5 * r) → n ≥ 5) ∧ 
  (∃ r : ℕ, (2 * n = 5 * r) ∧ n = 5) := by
  sorry

end NUMINAMATH_GPT_min_n_constant_term_l1207_120747


namespace NUMINAMATH_GPT_square_tile_area_l1207_120751

-- Definition and statement of the problem
theorem square_tile_area (side_length : ℝ) (h : side_length = 7) : 
  (side_length * side_length) = 49 :=
by
  sorry

end NUMINAMATH_GPT_square_tile_area_l1207_120751


namespace NUMINAMATH_GPT_question_inequality_l1207_120764

theorem question_inequality (x y z : ℝ) :
  x^2 + y^2 + z^2 - x*y - y*z - z*x ≥ max (3/4 * (x - y)^2) (max (3/4 * (y - z)^2) (3/4 * (z - x)^2)) := 
sorry

end NUMINAMATH_GPT_question_inequality_l1207_120764


namespace NUMINAMATH_GPT_maximum_distance_between_balls_l1207_120700

theorem maximum_distance_between_balls 
  (a b c : ℝ) 
  (aluminum_ball_heavier : true) -- Implicitly understood property rather than used in calculation directly
  (wood_ball_lighter : true) -- Implicitly understood property rather than used in calculation directly
  : ∃ d : ℝ, d = Real.sqrt (a^2 + b^2 + c^2) → d = Real.sqrt (3^2 + 4^2 + 2^2) := 
by
  use Real.sqrt (3^2 + 4^2 + 2^2)
  sorry

end NUMINAMATH_GPT_maximum_distance_between_balls_l1207_120700


namespace NUMINAMATH_GPT_sum_is_eighteen_or_twentyseven_l1207_120780

theorem sum_is_eighteen_or_twentyseven :
  ∀ (A B C D E I J K L M : ℕ),
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ I ∧ A ≠ J ∧ A ≠ K ∧ A ≠ L ∧ A ≠ M ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ I ∧ B ≠ J ∧ B ≠ K ∧ B ≠ L ∧ B ≠ M ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ I ∧ C ≠ J ∧ C ≠ K ∧ C ≠ L ∧ C ≠ M ∧
  D ≠ E ∧ D ≠ I ∧ D ≠ J ∧ D ≠ K ∧ D ≠ L ∧ D ≠ M ∧
  E ≠ I ∧ E ≠ J ∧ E ≠ K ∧ E ≠ L ∧ E ≠ M ∧
  I ≠ J ∧ I ≠ K ∧ I ≠ L ∧ I ≠ M ∧
  J ≠ K ∧ J ≠ L ∧ J ≠ M ∧
  K ≠ L ∧ K ≠ M ∧
  L ≠ M ∧
  (0 < I) ∧ (0 < J) ∧ (0 < K) ∧ (0 < L) ∧ (0 < M) ∧
  A + B + C + D + E + I + J + K + L + M = 45 ∧
  (I + J + K + L + M) % 10 = 0 →
  A + B + C + D + E + (I + J + K + L + M) / 10 = 18 ∨
  A + B + C + D + E + (I + J + K + L + M) / 10 = 27 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_is_eighteen_or_twentyseven_l1207_120780


namespace NUMINAMATH_GPT_men_joined_l1207_120727

-- Definitions for initial conditions
def initial_men : ℕ := 10
def initial_days : ℕ := 50
def extended_days : ℕ := 25

-- Theorem stating the number of men who joined the camp
theorem men_joined (x : ℕ) 
    (initial_food : initial_men * initial_days = (initial_men + x) * extended_days) : 
    x = 10 := 
sorry

end NUMINAMATH_GPT_men_joined_l1207_120727


namespace NUMINAMATH_GPT_original_price_of_shirt_l1207_120794

theorem original_price_of_shirt (P : ℝ) (h : 0.5625 * P = 18) : P = 32 := 
by 
sorry

end NUMINAMATH_GPT_original_price_of_shirt_l1207_120794
