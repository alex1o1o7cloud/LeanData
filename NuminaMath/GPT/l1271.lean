import Mathlib

namespace NUMINAMATH_GPT_unique_solution_l1271_127186

theorem unique_solution (k : ℝ) (h : k + 1 ≠ 0) : 
  (∀ x y : ℝ, ((x + 3) / (k * x + x - 3) = x) → ((y + 3) / (k * y + y - 3) = y) → x = y) ↔ k = -7/3 :=
by sorry

end NUMINAMATH_GPT_unique_solution_l1271_127186


namespace NUMINAMATH_GPT_solve_for_x_l1271_127156

theorem solve_for_x : 
  (∀ (x y : ℝ), y = 1 / (4 * x + 2) → y = 2 → x = -3 / 8) :=
by
  intro x y
  intro h₁ h₂
  rw [h₂] at h₁
  sorry

end NUMINAMATH_GPT_solve_for_x_l1271_127156


namespace NUMINAMATH_GPT_find_n_l1271_127157

theorem find_n (n : ℕ) (h_pos : 0 < n) (h_prime : Nat.Prime (n^4 - 16 * n^2 + 100)) : n = 3 := 
sorry

end NUMINAMATH_GPT_find_n_l1271_127157


namespace NUMINAMATH_GPT_sum_of_integers_l1271_127143

theorem sum_of_integers (a b : ℤ) (h : (Int.sqrt (a - 2023) + |b + 2023| = 1)) : a + b = 1 ∨ a + b = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1271_127143


namespace NUMINAMATH_GPT_polynomial_multiple_of_six_l1271_127107

theorem polynomial_multiple_of_six 
  (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) :
  6 ∣ P.eval 5 :=
sorry

end NUMINAMATH_GPT_polynomial_multiple_of_six_l1271_127107


namespace NUMINAMATH_GPT_calculate_molecular_weight_CaBr2_l1271_127188

def atomic_weight_Ca : ℝ := 40.08                 -- The atomic weight of calcium (Ca)
def atomic_weight_Br : ℝ := 79.904                -- The atomic weight of bromine (Br)
def molecular_weight_CaBr2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_Br  -- Definition of molecular weight of CaBr₂

theorem calculate_molecular_weight_CaBr2 : molecular_weight_CaBr2 = 199.888 := by
  sorry

end NUMINAMATH_GPT_calculate_molecular_weight_CaBr2_l1271_127188


namespace NUMINAMATH_GPT_monotone_increasing_interval_l1271_127153

def f (x : ℝ) := x^2 - 2

theorem monotone_increasing_interval : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y :=
by
  sorry

end NUMINAMATH_GPT_monotone_increasing_interval_l1271_127153


namespace NUMINAMATH_GPT_second_consecutive_odd_integer_l1271_127103

theorem second_consecutive_odd_integer (x : ℤ) 
  (h1 : ∃ x, x % 2 = 1 ∧ (x + 2) % 2 = 1 ∧ (x + 4) % 2 = 1) 
  (h2 : (x + 2) + (x + 4) = x + 17) : 
  (x + 2) = 13 :=
by
  sorry

end NUMINAMATH_GPT_second_consecutive_odd_integer_l1271_127103


namespace NUMINAMATH_GPT_find_c_for_minimum_value_l1271_127158

-- Definitions based on the conditions
def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

-- Main statement to be proved
theorem find_c_for_minimum_value (c : ℝ) : (∀ x, (3*x^2 - 4*c*x + c^2) = 0) → c = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_c_for_minimum_value_l1271_127158


namespace NUMINAMATH_GPT_least_zorgs_to_drop_more_points_than_eating_l1271_127184

theorem least_zorgs_to_drop_more_points_than_eating :
  ∃ (n : ℕ), (∀ m < n, m * (m + 1) / 2 ≤ 20 * m) ∧ n * (n + 1) / 2 > 20 * n :=
sorry

end NUMINAMATH_GPT_least_zorgs_to_drop_more_points_than_eating_l1271_127184


namespace NUMINAMATH_GPT_geometric_series_sum_l1271_127141

theorem geometric_series_sum (a r : ℝ)
  (h₁ : a / (1 - r) = 15)
  (h₂ : a / (1 - r^4) = 9) :
  r = 1 / 3 :=
sorry

end NUMINAMATH_GPT_geometric_series_sum_l1271_127141


namespace NUMINAMATH_GPT_isabel_remaining_pages_l1271_127114

def total_problems : ℕ := 72
def finished_problems : ℕ := 32
def problems_per_page : ℕ := 8

theorem isabel_remaining_pages :
  (total_problems - finished_problems) / problems_per_page = 5 := 
sorry

end NUMINAMATH_GPT_isabel_remaining_pages_l1271_127114


namespace NUMINAMATH_GPT_fewest_cookies_l1271_127118

theorem fewest_cookies
  (area_art_cookies : ℝ)
  (area_roger_cookies : ℝ)
  (area_paul_cookies : ℝ)
  (area_trisha_cookies : ℝ)
  (h_art : area_art_cookies = 12)
  (h_roger : area_roger_cookies = 8)
  (h_paul : area_paul_cookies = 6)
  (h_trisha : area_trisha_cookies = 6)
  (dough : ℝ) :
  (dough / area_art_cookies) < (dough / area_roger_cookies) ∧
  (dough / area_art_cookies) < (dough / area_paul_cookies) ∧
  (dough / area_art_cookies) < (dough / area_trisha_cookies) := by
  sorry

end NUMINAMATH_GPT_fewest_cookies_l1271_127118


namespace NUMINAMATH_GPT_average_score_l1271_127115

-- Definitions from conditions
def June_score := 97
def Patty_score := 85
def Josh_score := 100
def Henry_score := 94
def total_children := 4
def total_score := June_score + Patty_score + Josh_score + Henry_score

-- Prove the average score
theorem average_score : (total_score / total_children) = 94 :=
by
  sorry

end NUMINAMATH_GPT_average_score_l1271_127115


namespace NUMINAMATH_GPT_right_triangle_legs_l1271_127126

theorem right_triangle_legs (a b : ℕ) (h : a^2 + b^2 = 100) (h_r: a + b - 10 = 4) : (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) :=
sorry

end NUMINAMATH_GPT_right_triangle_legs_l1271_127126


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1271_127191

-- Arithmetic sequence with condition and proof of common difference
theorem arithmetic_sequence_common_difference (a : ℕ → ℚ) (d : ℚ) :
  (a 2015 = a 2013 + 6) → ((a 2015 - a 2013) = 2 * d) → (d = 3) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1271_127191


namespace NUMINAMATH_GPT_ellipse_equation_and_fixed_point_proof_l1271_127138

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_and_fixed_point_proof_l1271_127138


namespace NUMINAMATH_GPT_smallest_resolvable_debt_l1271_127169

def pig_value : ℤ := 450
def goat_value : ℤ := 330
def gcd_pig_goat : ℤ := Int.gcd pig_value goat_value

theorem smallest_resolvable_debt :
  ∃ p g : ℤ, gcd_pig_goat * 4 = pig_value * p + goat_value * g := 
by
  sorry

end NUMINAMATH_GPT_smallest_resolvable_debt_l1271_127169


namespace NUMINAMATH_GPT_andrew_age_l1271_127179

theorem andrew_age (a g : ℝ) (h1 : g = 9 * a) (h2 : g - a = 63) : a = 7.875 :=
by
  sorry

end NUMINAMATH_GPT_andrew_age_l1271_127179


namespace NUMINAMATH_GPT_two_largest_divisors_difference_l1271_127100

theorem two_largest_divisors_difference (N : ℕ) (h : N > 1) (a : ℕ) (ha : a ∣ N) (h6a : 6 * a ∣ N) :
  (N / 2 : ℚ) / (N / 3 : ℚ) = 1.5 := by
  sorry

end NUMINAMATH_GPT_two_largest_divisors_difference_l1271_127100


namespace NUMINAMATH_GPT_find_p4_q4_l1271_127189

-- Definitions
def p (x : ℝ) : ℝ := 3 * (x - 6) * (x - 2)
def q (x : ℝ) : ℝ := (x - 6) * (x + 3)

-- Statement to prove
theorem find_p4_q4 : (p 4) / (q 4) = 6 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_p4_q4_l1271_127189


namespace NUMINAMATH_GPT_remaining_medieval_art_pieces_remaining_renaissance_art_pieces_remaining_modern_art_pieces_l1271_127150

structure ArtCollection where
  medieval : ℕ
  renaissance : ℕ
  modern : ℕ

def AliciaArtCollection : ArtCollection := {
  medieval := 70,
  renaissance := 120,
  modern := 150
}

def donationPercentages : ArtCollection := {
  medieval := 65,
  renaissance := 30,
  modern := 45
}

def remainingArtPieces (initial : ℕ) (percent : ℕ) : ℕ :=
  initial - ((percent * initial) / 100)

theorem remaining_medieval_art_pieces :
  remainingArtPieces AliciaArtCollection.medieval donationPercentages.medieval = 25 := by
  sorry

theorem remaining_renaissance_art_pieces :
  remainingArtPieces AliciaArtCollection.renaissance donationPercentages.renaissance = 84 := by
  sorry

theorem remaining_modern_art_pieces :
  remainingArtPieces AliciaArtCollection.modern donationPercentages.modern = 83 := by
  sorry

end NUMINAMATH_GPT_remaining_medieval_art_pieces_remaining_renaissance_art_pieces_remaining_modern_art_pieces_l1271_127150


namespace NUMINAMATH_GPT_relationship_xyz_w_l1271_127122

theorem relationship_xyz_w (x y z w : ℝ) (h : (x + y) / (y + z) = (2 * z + w) / (w + x)) :
  x = 2 * z - w := 
sorry

end NUMINAMATH_GPT_relationship_xyz_w_l1271_127122


namespace NUMINAMATH_GPT_find_roots_l1271_127183

def polynomial (x: ℝ) := x^3 - 2*x^2 - x + 2

theorem find_roots : { x : ℝ // polynomial x = 0 } = ({1, -1, 2} : Set ℝ) :=
by
  sorry

end NUMINAMATH_GPT_find_roots_l1271_127183


namespace NUMINAMATH_GPT_total_cookies_in_box_l1271_127132

-- Definitions from the conditions
def oldest_son_cookies : ℕ := 4
def youngest_son_cookies : ℕ := 2
def days_box_lasts : ℕ := 9

-- Total cookies consumed per day
def daily_cookies_consumption : ℕ := oldest_son_cookies + youngest_son_cookies

-- Theorem statement: total number of cookies in the box
theorem total_cookies_in_box : (daily_cookies_consumption * days_box_lasts) = 54 := by
  sorry

end NUMINAMATH_GPT_total_cookies_in_box_l1271_127132


namespace NUMINAMATH_GPT_digit_Phi_l1271_127181

theorem digit_Phi (Phi : ℕ) (h1 : 220 / Phi = 40 + 3 * Phi) : Phi = 4 :=
by
  sorry

end NUMINAMATH_GPT_digit_Phi_l1271_127181


namespace NUMINAMATH_GPT_pagoda_lights_l1271_127109

/-- From afar, the magnificent pagoda has seven layers, with red lights doubling on each
ascending floor, totaling 381 lights. How many lights are there at the very top? -/
theorem pagoda_lights :
  ∃ x, (1 + 2 + 4 + 8 + 16 + 32 + 64) * x = 381 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_pagoda_lights_l1271_127109


namespace NUMINAMATH_GPT_movies_shown_eq_twenty_four_l1271_127160

-- Define conditions
variables (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ)

-- Define the total number of movies calculation
noncomputable def total_movies_shown (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  screens * (open_hours / movie_duration)

-- Theorem to prove the total number of movies shown is 24
theorem movies_shown_eq_twenty_four : 
  total_movies_shown 6 8 2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_movies_shown_eq_twenty_four_l1271_127160


namespace NUMINAMATH_GPT_lyle_percentage_l1271_127163

theorem lyle_percentage (chips : ℕ) (ian_ratio lyle_ratio : ℕ) (h_ratio_sum : ian_ratio + lyle_ratio = 10) (h_chips : chips = 100) :
  (lyle_ratio / (ian_ratio + lyle_ratio) : ℚ) * 100 = 60 := 
by
  sorry

end NUMINAMATH_GPT_lyle_percentage_l1271_127163


namespace NUMINAMATH_GPT_radius_of_first_cylinder_l1271_127110

theorem radius_of_first_cylinder :
  ∀ (rounds1 rounds2 : ℕ) (r2 r1 : ℝ), rounds1 = 70 → rounds2 = 49 → r2 = 20 → 
  (2 * Real.pi * r1 * rounds1 = 2 * Real.pi * r2 * rounds2) → r1 = 14 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_first_cylinder_l1271_127110


namespace NUMINAMATH_GPT_compound_carbon_atoms_l1271_127151

-- Definition of data given in the problem.
def molecular_weight : ℕ := 60
def hydrogen_atoms : ℕ := 4
def oxygen_atoms : ℕ := 2
def carbon_atomic_weight : ℕ := 12
def hydrogen_atomic_weight : ℕ := 1
def oxygen_atomic_weight : ℕ := 16

-- Statement to prove the number of carbon atoms in the compound.
theorem compound_carbon_atoms : 
  (molecular_weight - (hydrogen_atoms * hydrogen_atomic_weight + oxygen_atoms * oxygen_atomic_weight)) / carbon_atomic_weight = 2 := 
by
  sorry

end NUMINAMATH_GPT_compound_carbon_atoms_l1271_127151


namespace NUMINAMATH_GPT_threshold_mu_l1271_127176

/-- 
Find threshold values μ₁₀₀ and μ₁₀₀₀₀₀ such that 
F = m * n * sin (π / m) * sqrt (1 / n² + sin⁴ (π / m)) 
is definitely greater than 100 and 1,000,000 respectively for all m greater than μ₁₀₀ and μ₁₀₀₀₀₀, 
assuming n = m³. -/
theorem threshold_mu : 
  (∃ (μ₁₀₀ μ₁₀₀₀₀₀ : ℝ), ∀ (m : ℝ), m > μ₁₀₀ → 
    m * (m ^ 3) * (Real.sin (Real.pi / m)) * 
      (Real.sqrt ((1 : ℝ) / (m ^ 6) + (Real.sin (Real.pi / m)) ^ 4)) > 100) ∧ 
  (∃ (μ₁₀₀₀₀₀ μ₁₀₀₀₀₀ : ℝ), ∀ (m : ℝ), m > μ₁₀₀₀₀₀ → 
    m * (m ^ 3) * (Real.sin (Real.pi / m)) * 
      (Real.sqrt ((1 : ℝ) / (m ^ 6) + (Real.sin (Real.pi / m)) ^ 4)) > 1000000) :=
sorry

end NUMINAMATH_GPT_threshold_mu_l1271_127176


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1271_127137

variable (a b : ℤ)

theorem simplify_and_evaluate_expression (h1 : a = 1) (h2 : b = -1) :
  (3 * a^2 * b - 2 * (a * b - (3/2) * a^2 * b) + a * b - 2 * a^2 * b) = -3 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1271_127137


namespace NUMINAMATH_GPT_mean_of_five_numbers_l1271_127124

theorem mean_of_five_numbers (S : ℚ) (n : ℕ) (h1 : S = 3/4) (h2 : n = 5) :
  (S / n) = 3/20 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_mean_of_five_numbers_l1271_127124


namespace NUMINAMATH_GPT_heating_time_correct_l1271_127120

structure HeatingProblem where
  initial_temp : ℕ
  final_temp : ℕ
  heating_rate : ℕ

def time_to_heat (hp : HeatingProblem) : ℕ :=
  (hp.final_temp - hp.initial_temp) / hp.heating_rate

theorem heating_time_correct (hp : HeatingProblem) (h1 : hp.initial_temp = 20) (h2 : hp.final_temp = 100) (h3 : hp.heating_rate = 5) :
  time_to_heat hp = 16 :=
by
  sorry

end NUMINAMATH_GPT_heating_time_correct_l1271_127120


namespace NUMINAMATH_GPT_solution_set_for_f_geq_zero_l1271_127134

theorem solution_set_for_f_geq_zero (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_f3 : f 3 = 0) (h_cond : ∀ x : ℝ, x < 0 → x * (deriv f x) < f x) :
  {x : ℝ | f x ≥ 0} = {x : ℝ | -3 < x ∧ x < 0} ∪ {x : ℝ | 3 < x} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_for_f_geq_zero_l1271_127134


namespace NUMINAMATH_GPT_percentage_of_l_equals_150_percent_k_l1271_127155

section

variables (j k l m : ℝ) (x : ℝ)

-- Given conditions
axiom cond1 : 1.25 * j = 0.25 * k
axiom cond2 : 1.50 * k = x / 100 * l
axiom cond3 : 1.75 * l = 0.75 * m
axiom cond4 : 0.20 * m = 7.00 * j

-- Proof statement
theorem percentage_of_l_equals_150_percent_k : x = 50 :=
sorry

end

end NUMINAMATH_GPT_percentage_of_l_equals_150_percent_k_l1271_127155


namespace NUMINAMATH_GPT_total_ticket_revenue_l1271_127162

theorem total_ticket_revenue (total_seats : Nat) (price_adult_ticket : Nat) (price_child_ticket : Nat)
  (theatre_full : Bool) (child_tickets : Nat) (adult_tickets := total_seats - child_tickets)
  (rev_adult := adult_tickets * price_adult_ticket) (rev_child := child_tickets * price_child_ticket) :
  total_seats = 250 →
  price_adult_ticket = 6 →
  price_child_ticket = 4 →
  theatre_full = true →
  child_tickets = 188 →
  rev_adult + rev_child = 1124 := 
by
  intros h_total_seats h_price_adult h_price_child h_theatre_full h_child_tickets
  sorry

end NUMINAMATH_GPT_total_ticket_revenue_l1271_127162


namespace NUMINAMATH_GPT_marie_gift_boxes_l1271_127174

theorem marie_gift_boxes
  (total_eggs : ℕ)
  (weight_per_egg : ℕ)
  (remaining_weight : ℕ)
  (melted_eggs_weight : ℕ)
  (eggs_per_box : ℕ)
  (total_boxes : ℕ)
  (H1 : total_eggs = 12)
  (H2 : weight_per_egg = 10)
  (H3 : remaining_weight = 90)
  (H4 : melted_eggs_weight = total_eggs * weight_per_egg - remaining_weight)
  (H5 : melted_eggs_weight / weight_per_egg = eggs_per_box)
  (H6 : total_eggs / eggs_per_box = total_boxes) :
  total_boxes = 4 := 
sorry

end NUMINAMATH_GPT_marie_gift_boxes_l1271_127174


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1271_127108

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), 16 * x^2 - 9 * y^2 = -144 → (y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intros x y h1
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1271_127108


namespace NUMINAMATH_GPT_quadratic_eq_mn_sum_l1271_127146

theorem quadratic_eq_mn_sum (m n : ℤ) 
  (h1 : m - 1 = 2) 
  (h2 : 16 + 4 * n = 0) 
  : m + n = -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_mn_sum_l1271_127146


namespace NUMINAMATH_GPT_no_integer_solutions_l1271_127152

theorem no_integer_solutions :
  ∀ n m : ℤ, (n^2 + (n+1)^2 + (n+2)^2) ≠ m^2 :=
by
  intro n m
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1271_127152


namespace NUMINAMATH_GPT_general_term_arithmetic_sequence_l1271_127190

theorem general_term_arithmetic_sequence (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = a n + 2) : 
  ∀ n, a n = 2 * n - 1 := 
by 
  sorry

end NUMINAMATH_GPT_general_term_arithmetic_sequence_l1271_127190


namespace NUMINAMATH_GPT_gray_region_area_l1271_127149

theorem gray_region_area 
  (r : ℝ) 
  (h1 : ∀ r : ℝ, (3 * r) - r = 3) 
  (h2 : r = 1.5) 
  (inner_circle_area : ℝ := π * r * r) 
  (outer_circle_area : ℝ := π * (3 * r) * (3 * r)) : 
  outer_circle_area - inner_circle_area = 18 * π := 
by
  sorry

end NUMINAMATH_GPT_gray_region_area_l1271_127149


namespace NUMINAMATH_GPT_problem_l1271_127140

theorem problem (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 16) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 4) : 
  x * y * z = 4 := 
sorry

end NUMINAMATH_GPT_problem_l1271_127140


namespace NUMINAMATH_GPT_cube_volume_l1271_127192

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end NUMINAMATH_GPT_cube_volume_l1271_127192


namespace NUMINAMATH_GPT_Bernoulli_inequality_l1271_127125

theorem Bernoulli_inequality (n : ℕ) (a : ℝ) (h : a > -1) : (1 + a)^n ≥ n * a + 1 := 
sorry

end NUMINAMATH_GPT_Bernoulli_inequality_l1271_127125


namespace NUMINAMATH_GPT_num_valid_n_l1271_127172

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (Nat.succ n') => Nat.succ n' * factorial n'

def divisible (a b : ℕ) : Prop := b ∣ a

theorem num_valid_n (N : ℕ) :
  N ≤ 30 → 
  ¬ (∃ k, k + 1 ≤ 31 ∧ k + 1 > 1 ∧ (Prime (k + 1)) ∧ ¬ divisible (2 * factorial (k - 1)) (k + 1)) →
  ∃ m : ℕ, m = 20 :=
by
  sorry

end NUMINAMATH_GPT_num_valid_n_l1271_127172


namespace NUMINAMATH_GPT_f_2009_value_l1271_127136

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom even_function (f : ℝ → ℝ) : ∀ x, f x = f (-x)
axiom odd_function (g : ℝ → ℝ) : ∀ x, g x = -g (-x)
axiom f_value : f 1 = 0
axiom g_def : ∀ x, g x = f (x - 1)

theorem f_2009_value : f 2009 = 0 :=
by
  sorry

end NUMINAMATH_GPT_f_2009_value_l1271_127136


namespace NUMINAMATH_GPT_points_three_units_away_from_neg_two_on_number_line_l1271_127133

theorem points_three_units_away_from_neg_two_on_number_line :
  ∃! p1 p2 : ℤ, |p1 + 2| = 3 ∧ |p2 + 2| = 3 ∧ p1 ≠ p2 ∧ (p1 = -5 ∨ p2 = -5) ∧ (p1 = 1 ∨ p2 = 1) :=
sorry

end NUMINAMATH_GPT_points_three_units_away_from_neg_two_on_number_line_l1271_127133


namespace NUMINAMATH_GPT_remainder_equality_l1271_127113

theorem remainder_equality (a b s t d : ℕ) (h1 : a > b) (h2 : a % d = s % d) (h3 : b % d = t % d) :
  ((a + 1) * (b + 1)) % d = ((s + 1) * (t + 1)) % d :=
by
  sorry

end NUMINAMATH_GPT_remainder_equality_l1271_127113


namespace NUMINAMATH_GPT_problem_statement_l1271_127171

-- Defining the real numbers and the hypothesis
variables {a b c x y z : ℝ}
variables (h1 : 17 * x + b * y + c * z = 0)
variables (h2 : a * x + 31 * y + c * z = 0)
variables (h3 : a * x + b * y + 53 * z = 0)
variables (ha : a ≠ 17)
variables (hx : x ≠ 0)

-- State the theorem
theorem problem_statement : 
  (a / (a - 17) + b / (b - 31) + c / (c - 53) = 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1271_127171


namespace NUMINAMATH_GPT_sample_size_student_congress_l1271_127135

-- Definitions based on the conditions provided in the problem
def num_classes := 40
def students_per_class := 3

-- Theorem statement for the mathematically equivalent proof problem
theorem sample_size_student_congress : 
  (num_classes * students_per_class) = 120 := 
by 
  sorry

end NUMINAMATH_GPT_sample_size_student_congress_l1271_127135


namespace NUMINAMATH_GPT_sum_smallest_largest_l1271_127116

theorem sum_smallest_largest (z b : ℤ) (n : ℤ) (h_even_n : (n % 2 = 0)) (h_mean : z = (n * b + ((n - 1) * n) / 2) / n) : 
  (2 * (z - (n - 1) / 2) + n - 1) = 2 * z := by
  sorry

end NUMINAMATH_GPT_sum_smallest_largest_l1271_127116


namespace NUMINAMATH_GPT_sequence_S15_is_211_l1271_127178

theorem sequence_S15_is_211 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2)
  (h3 : ∀ n > 1, S (n + 1) + S (n - 1) = 2 * (S n + S 1)) :
  S 15 = 211 := 
sorry

end NUMINAMATH_GPT_sequence_S15_is_211_l1271_127178


namespace NUMINAMATH_GPT_players_at_least_two_sciences_l1271_127102

-- Define the conditions of the problem
def total_players : Nat := 30
def players_biology : Nat := 15
def players_chemistry : Nat := 10
def players_physics : Nat := 5
def players_all_three : Nat := 3

-- Define the main theorem we want to prove
theorem players_at_least_two_sciences :
  (players_biology + players_chemistry + players_physics 
    - players_all_three - total_players) = 9 :=
sorry

end NUMINAMATH_GPT_players_at_least_two_sciences_l1271_127102


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1271_127105

-- Step d) Rewrite the problem in Lean 4 statement
theorem geometric_sequence_problem 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (b_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (q : ℝ) 
  (h1 : ∀ n, n > 0 → a_n n = 1 * q^(n-1)) 
  (h2 : 1 + q + q^2 = 7)
  (h3 : 6 * 1 * q = 1 + 3 + 1 * q^2 + 4)
  :
  (∀ n, a_n n = 2^(n-1)) ∧ 
  (∀ n, T_n n = 4 - (n+2) / 2^(n-1)) :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1271_127105


namespace NUMINAMATH_GPT_overlap_area_rhombus_l1271_127185

noncomputable def area_of_overlap (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  1 / (Real.sin (α / 2))

theorem overlap_area_rhombus (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  area_of_overlap α hα = 1 / (Real.sin (α / 2)) :=
sorry

end NUMINAMATH_GPT_overlap_area_rhombus_l1271_127185


namespace NUMINAMATH_GPT_contradiction_method_example_l1271_127175

variables {a b c : ℝ}
variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variables (h4 : a + b + c > 0) (h5 : ab + bc + ca > 0)
variables (h6 : (a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (b < 0 ∧ c < 0))

theorem contradiction_method_example : false :=
by {
  sorry
}

end NUMINAMATH_GPT_contradiction_method_example_l1271_127175


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l1271_127142

-- Definitions based on conditions
variables {a_n b_n : ℕ → ℕ} -- Arithmetic sequences
variables {A_n B_n : ℕ → ℕ} -- Sums of the first n terms

-- Given condition
axiom sums_of_arithmetic_sequences (n : ℕ) : A_n n / B_n n = (7 * n + 1) / (4 * n + 27)

-- Theorem to prove
theorem arithmetic_sequence_ratio :
  ∀ (a_n b_n : ℕ → ℕ) (A_n B_n : ℕ → ℕ), 
    (∀ n, A_n n / B_n n = (7 * n + 1) / (4 * n + 27)) → 
    a_6 / b_6 = 78 / 71 := 
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l1271_127142


namespace NUMINAMATH_GPT_wire_not_used_l1271_127106

variable (total_wire length_cut_parts parts_used : ℕ)

theorem wire_not_used (h1 : total_wire = 50) (h2 : length_cut_parts = 5) (h3 : parts_used = 3) : 
  total_wire - (parts_used * (total_wire / length_cut_parts)) = 20 := 
  sorry

end NUMINAMATH_GPT_wire_not_used_l1271_127106


namespace NUMINAMATH_GPT_find_value_of_function_l1271_127173

theorem find_value_of_function (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x + f (-x) = 3 * x + 2) : 
  f 2 = 20 / 3 :=
sorry

end NUMINAMATH_GPT_find_value_of_function_l1271_127173


namespace NUMINAMATH_GPT_Gandalf_reachability_l1271_127161

theorem Gandalf_reachability (n : ℕ) (h : n ≥ 1) :
  ∃ (m : ℕ), m = 1 :=
sorry

end NUMINAMATH_GPT_Gandalf_reachability_l1271_127161


namespace NUMINAMATH_GPT_large_monkey_doll_cost_l1271_127180

theorem large_monkey_doll_cost (S L E : ℝ) 
  (h1 : S = L - 2) 
  (h2 : E = L + 1) 
  (h3 : 300 / S = 300 / L + 25) 
  (h4 : 300 / E = 300 / L - 15) : 
  L = 6 := 
sorry

end NUMINAMATH_GPT_large_monkey_doll_cost_l1271_127180


namespace NUMINAMATH_GPT_horse_revolutions_l1271_127197

theorem horse_revolutions (r1 r2 r3 : ℝ) (rev1 : ℕ) 
  (h1 : r1 = 30) (h2 : r2 = 15) (h3 : r3 = 10) (h4 : rev1 = 40) :
  (r2 / r1 = 1 / 2 ∧ 2 * rev1 = 80) ∧ (r3 / r1 = 1 / 3 ∧ 3 * rev1 = 120) :=
by
  sorry

end NUMINAMATH_GPT_horse_revolutions_l1271_127197


namespace NUMINAMATH_GPT_window_total_width_l1271_127182

theorem window_total_width 
  (panes : Nat := 6)
  (ratio_height_width : ℤ := 3)
  (border_width : ℤ := 1)
  (rows : Nat := 2)
  (columns : Nat := 3)
  (pane_width : ℤ := 12) :
  3 * pane_width + 2 * border_width + 2 * border_width = 40 := 
by
  sorry

end NUMINAMATH_GPT_window_total_width_l1271_127182


namespace NUMINAMATH_GPT_mean_equality_l1271_127199

theorem mean_equality (y : ℝ) :
  ((3 + 7 + 11 + 15) / 4 = (10 + 14 + y) / 3) → y = 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_equality_l1271_127199


namespace NUMINAMATH_GPT_playground_perimeter_l1271_127117

-- Defining the conditions
def length : ℕ := 100
def breadth : ℕ := 500
def perimeter (L B : ℕ) : ℕ := 2 * (L + B)

-- The theorem to prove
theorem playground_perimeter : perimeter length breadth = 1200 := 
by
  -- The actual proof will be filled later
  sorry

end NUMINAMATH_GPT_playground_perimeter_l1271_127117


namespace NUMINAMATH_GPT_single_discount_equivalence_l1271_127159

theorem single_discount_equivalence (original_price : ℝ) (first_discount second_discount : ℝ) (final_price : ℝ) :
  original_price = 50 →
  first_discount = 0.30 →
  second_discount = 0.10 →
  final_price = original_price * (1 - first_discount) * (1 - second_discount) →
  ((original_price - final_price) / original_price) * 100 = 37 := by
  sorry

end NUMINAMATH_GPT_single_discount_equivalence_l1271_127159


namespace NUMINAMATH_GPT_zeros_in_square_of_999_999_999_l1271_127131

noncomputable def number_of_zeros_in_square (n : ℕ) : ℕ :=
  if n ≥ 1 then n - 1 else 0

theorem zeros_in_square_of_999_999_999 :
  number_of_zeros_in_square 9 = 8 :=
sorry

end NUMINAMATH_GPT_zeros_in_square_of_999_999_999_l1271_127131


namespace NUMINAMATH_GPT_equal_shipments_by_truck_l1271_127101

theorem equal_shipments_by_truck (T : ℕ) (hT1 : 120 % T = 0) (hT2 : T ≠ 5) : T = 2 :=
by
  sorry

end NUMINAMATH_GPT_equal_shipments_by_truck_l1271_127101


namespace NUMINAMATH_GPT_sin_neg_30_eq_neg_one_half_l1271_127130

theorem sin_neg_30_eq_neg_one_half :
  Real.sin (-30 * Real.pi / 180) = -1 / 2 := 
by
  sorry -- Proof is skipped

end NUMINAMATH_GPT_sin_neg_30_eq_neg_one_half_l1271_127130


namespace NUMINAMATH_GPT_exam_percentage_l1271_127167

theorem exam_percentage (x : ℝ) (h_cond : 100 - x >= 0 ∧ x >= 0 ∧ 60 * x + 90 * (100 - x) = 69 * 100) : x = 70 := by
  sorry

end NUMINAMATH_GPT_exam_percentage_l1271_127167


namespace NUMINAMATH_GPT_darcy_commute_l1271_127148

theorem darcy_commute (d w r t x time_walk train_time : ℝ) 
  (h1 : d = 1.5)
  (h2 : w = 3)
  (h3 : r = 20)
  (h4 : train_time = t + x)
  (h5 : time_walk = 15 + train_time)
  (h6 : time_walk = d / w * 60)  -- Time taken to walk in minutes
  (h7 : t = d / r * 60)  -- Time taken on train in minutes
  : x = 10.5 :=
sorry

end NUMINAMATH_GPT_darcy_commute_l1271_127148


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l1271_127187

theorem right_triangle_hypotenuse (a b c : ℕ) (h : a = 6) (k : b = 8) (pt : a^2 + b^2 = c^2) : c = 10 := by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l1271_127187


namespace NUMINAMATH_GPT_probability_third_winning_l1271_127121

-- Definitions based on the conditions provided
def num_tickets : ℕ := 10
def num_winning_tickets : ℕ := 3
def num_non_winning_tickets : ℕ := num_tickets - num_winning_tickets

-- Define the probability function
def probability_of_third_draw_winning : ℚ :=
  (num_non_winning_tickets / num_tickets) * 
  ((num_non_winning_tickets - 1) / (num_tickets - 1)) * 
  (num_winning_tickets / (num_tickets - 2))

-- The theorem to prove
theorem probability_third_winning : probability_of_third_draw_winning = 7 / 40 :=
  by sorry

end NUMINAMATH_GPT_probability_third_winning_l1271_127121


namespace NUMINAMATH_GPT_computation_is_correct_l1271_127168

def large_multiplication : ℤ := 23457689 * 84736521

def denominator_subtraction : ℤ := 7589236 - 3145897

def computed_m : ℚ := large_multiplication / denominator_subtraction

theorem computation_is_correct : computed_m = 447214.999 :=
by 
  -- exact calculation to be provided
  sorry

end NUMINAMATH_GPT_computation_is_correct_l1271_127168


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1271_127165

theorem sufficient_but_not_necessary_condition :
  (∀ (x : ℝ), x = 1 → x^2 - 3 * x + 2 = 0) ∧ ¬(∀ (x : ℝ), x^2 - 3 * x + 2 = 0 → x = 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1271_127165


namespace NUMINAMATH_GPT_overtime_hours_l1271_127129

theorem overtime_hours
  (regularPayPerHour : ℝ)
  (regularHours : ℝ)
  (totalPay : ℝ)
  (overtimeRate : ℝ) 
  (h1 : regularPayPerHour = 3)
  (h2 : regularHours = 40)
  (h3 : totalPay = 168)
  (h4 : overtimeRate = 2 * regularPayPerHour) :
  (totalPay - (regularPayPerHour * regularHours)) / overtimeRate = 8 :=
by
  sorry

end NUMINAMATH_GPT_overtime_hours_l1271_127129


namespace NUMINAMATH_GPT_solve_for_b_l1271_127104

theorem solve_for_b (b : ℚ) : 
  (∃ m1 m2 : ℚ, 3 * m1 - 2 * 1 + 4 = 0 ∧ 5 * m2 + b * 1 - 1 = 0 ∧ m1 * m2 = -1) → b = 15 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_b_l1271_127104


namespace NUMINAMATH_GPT_trig_identity_l1271_127164

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1271_127164


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_23_l1271_127128

def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def is_multiple_of_23 (n : ℕ) : Prop :=
  n % 23 = 0

theorem greatest_three_digit_multiple_of_23 :
  ∀ n, is_three_digit n ∧ is_multiple_of_23 n → n ≤ 989 :=
by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_23_l1271_127128


namespace NUMINAMATH_GPT_number_of_ordered_triples_l1271_127193

theorem number_of_ordered_triples (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
    (h_eq : a * b * c - b * c - a * c - a * b + a + b + c = 2013) :
    ∃ n, n = 39 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ordered_triples_l1271_127193


namespace NUMINAMATH_GPT_theater_ticket_sales_l1271_127127

theorem theater_ticket_sales (O B : ℕ) 
  (h1 : O + B = 370) 
  (h2 : 12 * O + 8 * B = 3320) : 
  B - O = 190 := 
sorry

end NUMINAMATH_GPT_theater_ticket_sales_l1271_127127


namespace NUMINAMATH_GPT_sum_of_thetas_l1271_127154

noncomputable def theta (k : ℕ) : ℝ := (54 + 72 * k) % 360

theorem sum_of_thetas : (theta 0 + theta 1 + theta 2 + theta 3 + theta 4) = 990 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_thetas_l1271_127154


namespace NUMINAMATH_GPT_combined_mpg_l1271_127119

-- Definitions based on the conditions
def ray_miles : ℕ := 150
def tom_miles : ℕ := 100
def ray_mpg : ℕ := 30
def tom_mpg : ℕ := 20

-- Theorem statement
theorem combined_mpg : (ray_miles + tom_miles) / ((ray_miles / ray_mpg) + (tom_miles / tom_mpg)) = 25 := by
  sorry

end NUMINAMATH_GPT_combined_mpg_l1271_127119


namespace NUMINAMATH_GPT_geometric_sum_l1271_127112

theorem geometric_sum 
  (a : ℕ → ℝ) (q : ℝ) (h1 : a 2 + a 4 = 32) (h2 : a 6 + a 8 = 16) 
  (h_seq : ∀ n, a (n+2) = a n * q ^ 2):
  a 10 + a 12 + a 14 + a 16 = 12 :=
by
  -- Proof needs to be written here
  sorry

end NUMINAMATH_GPT_geometric_sum_l1271_127112


namespace NUMINAMATH_GPT_min_floor_sum_l1271_127147

-- Definitions of the conditions
variables (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 24)

-- Our main theorem statement
theorem min_floor_sum (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 24) :
  (Nat.floor ((a+b) / c) + Nat.floor ((b+c) / a) + Nat.floor ((c+a) / b)) = 6 := 
sorry

end NUMINAMATH_GPT_min_floor_sum_l1271_127147


namespace NUMINAMATH_GPT_acute_triangle_exists_l1271_127166

theorem acute_triangle_exists {a1 a2 a3 a4 a5 : ℝ} 
  (h1 : a1 + a2 > a3) (h2 : a1 + a3 > a2) (h3 : a2 + a3 > a1)
  (h4 : a2 + a3 > a4) (h5 : a3 + a4 > a2) (h6 : a2 + a4 > a3)
  (h7 : a3 + a4 > a5) (h8 : a4 + a5 > a3) (h9 : a3 + a5 > a4) : 
  ∃ (t1 t2 t3 : ℝ), (t1 + t2 > t3) ∧ (t1 + t3 > t2) ∧ (t2 + t3 > t1) ∧ (t3 ^ 2 < t1 ^ 2 + t2 ^ 2) :=
sorry

end NUMINAMATH_GPT_acute_triangle_exists_l1271_127166


namespace NUMINAMATH_GPT_part_a_l1271_127145

-- Part (a)
theorem part_a (x : ℕ)  : (x^2 - x + 2) % 7 = 0 → x % 7 = 4 := by 
  sorry

end NUMINAMATH_GPT_part_a_l1271_127145


namespace NUMINAMATH_GPT_gain_percent_l1271_127170

variable (MP CP SP : ℝ)

def costPrice (CP : ℝ) (MP : ℝ) := CP = 0.64 * MP

def sellingPrice (SP : ℝ) (MP : ℝ) := SP = MP * 0.88

theorem gain_percent (h1 : costPrice CP MP) (h2 : sellingPrice SP MP) : 
  ((SP - CP) / CP) * 100 = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_l1271_127170


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_and_k_l1271_127177

theorem arithmetic_sequence_general_term_and_k (a : ℕ → ℚ) (d : ℚ)
  (h1 : a 4 + a 7 + a 10 = 17)
  (h2 : a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + a 13 + a 14 = 77) :
  (∀ n : ℕ, a n = (2 * n + 3) / 3) ∧ (∃ k : ℕ, a k = 13 ∧ k = 18) := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_and_k_l1271_127177


namespace NUMINAMATH_GPT_joggers_meet_again_at_correct_time_l1271_127144

-- Define the joggers and their lap times
def bob_lap_time := 3
def carol_lap_time := 5
def ted_lap_time := 8

-- Calculate the Least Common Multiple (LCM) of their lap times
def lcm_joggers := Nat.lcm (Nat.lcm bob_lap_time carol_lap_time) ted_lap_time

-- Start time is 9:00 AM
def start_time := 9 * 60  -- in minutes

-- The time (in minutes) we get back together is start_time plus the LCM
def earliest_meeting_time := start_time + lcm_joggers

-- Convert the meeting time to hours and minutes
def hours := earliest_meeting_time / 60
def minutes := earliest_meeting_time % 60

-- Define an expected result
def expected_meeting_hour := 11
def expected_meeting_minute := 0

-- Prove that all joggers will meet again at the correct time
theorem joggers_meet_again_at_correct_time :
  hours = expected_meeting_hour ∧ minutes = expected_meeting_minute :=
by
  -- Here you would provide the proof, but we'll use sorry for brevity
  sorry

end NUMINAMATH_GPT_joggers_meet_again_at_correct_time_l1271_127144


namespace NUMINAMATH_GPT_greatest_three_digit_divisible_by_3_5_6_l1271_127139

theorem greatest_three_digit_divisible_by_3_5_6 : 
    ∃ n : ℕ, 
        (100 ≤ n ∧ n ≤ 999) ∧ 
        (∃ k₃ : ℕ, n = 3 * k₃) ∧ 
        (∃ k₅ : ℕ, n = 5 * k₅) ∧ 
        (∃ k₆ : ℕ, n = 6 * k₆) ∧ 
        (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999) ∧ (∃ k₃ : ℕ, m = 3 * k₃) ∧ (∃ k₅ : ℕ, m = 5 * k₅) ∧ (∃ k₆ : ℕ, m = 6 * k₆) → m ≤ 990) := by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_divisible_by_3_5_6_l1271_127139


namespace NUMINAMATH_GPT_ravi_first_has_more_than_500_paperclips_on_wednesday_l1271_127196

noncomputable def paperclips (k : Nat) : Nat :=
  5 * 4^k

theorem ravi_first_has_more_than_500_paperclips_on_wednesday :
  ∃ k : Nat, paperclips k > 500 ∧ k = 3 :=
by
  sorry

end NUMINAMATH_GPT_ravi_first_has_more_than_500_paperclips_on_wednesday_l1271_127196


namespace NUMINAMATH_GPT_cornelia_european_countries_l1271_127123

def total_countries : Nat := 42
def south_american_countries : Nat := 10
def asian_countries : Nat := 6

def non_european_countries : Nat :=
  south_american_countries + 2 * asian_countries

def european_countries : Nat :=
  total_countries - non_european_countries

theorem cornelia_european_countries :
  european_countries = 20 := by
  sorry

end NUMINAMATH_GPT_cornelia_european_countries_l1271_127123


namespace NUMINAMATH_GPT_oranges_for_juice_l1271_127111

theorem oranges_for_juice (total_oranges : ℝ) (exported_percentage : ℝ) (juice_percentage : ℝ) :
  total_oranges = 7 →
  exported_percentage = 0.30 →
  juice_percentage = 0.60 →
  (total_oranges * (1 - exported_percentage) * juice_percentage) = 2.9 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_oranges_for_juice_l1271_127111


namespace NUMINAMATH_GPT_first_student_can_ensure_one_real_root_l1271_127195

noncomputable def can_first_student_ensure_one_real_root : Prop :=
  ∀ (b c : ℝ), ∃ a : ℝ, ∃ d : ℝ, ∀ (e : ℝ), 
    (d = 0 ∧ (e = b ∨ e = c)) → 
    (∀ x : ℝ, x^3 + d * x^2 + e * x + (if e = b then c else b) = 0)

theorem first_student_can_ensure_one_real_root :
  can_first_student_ensure_one_real_root := sorry

end NUMINAMATH_GPT_first_student_can_ensure_one_real_root_l1271_127195


namespace NUMINAMATH_GPT_valid_functions_l1271_127194

theorem valid_functions (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x + y) * g (x - y) = (g x + g y)^2 - 4 * x^2 * g y + 2 * y^2 * g x) :
  (∀ x, g x = 0) ∨ (∀ x, g x = x^2) :=
by sorry

end NUMINAMATH_GPT_valid_functions_l1271_127194


namespace NUMINAMATH_GPT_total_sand_donated_l1271_127198

theorem total_sand_donated (A B C D: ℚ) (hA: A = 33 / 2) (hB: B = 26) (hC: C = 49 / 2) (hD: D = 28) : 
  A + B + C + D = 95 := by
  sorry

end NUMINAMATH_GPT_total_sand_donated_l1271_127198
