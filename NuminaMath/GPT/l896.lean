import Mathlib

namespace skylar_total_donations_l896_89608

-- Define the conditions
def start_age : ℕ := 17
def current_age : ℕ := 71
def annual_donation : ℕ := 8000

-- Define the statement to be proven
theorem skylar_total_donations : 
  (current_age - start_age) * annual_donation = 432000 := by
    sorry

end skylar_total_donations_l896_89608


namespace jane_rejected_percentage_l896_89605

theorem jane_rejected_percentage (P : ℕ) (John_rejected : ℤ) (Jane_inspected_rejected : ℤ) :
  John_rejected = 7 * P ∧
  Jane_inspected_rejected = 5 * P ∧
  (John_rejected + Jane_inspected_rejected) = 75 * P → 
  Jane_inspected_rejected = P  :=
by sorry

end jane_rejected_percentage_l896_89605


namespace pow_2023_eq_one_or_neg_one_l896_89688

theorem pow_2023_eq_one_or_neg_one (x : ℂ) (h : (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0) : 
  x^2023 = 1 ∨ x^2023 = -1 := 
by 
{
  sorry
}

end pow_2023_eq_one_or_neg_one_l896_89688


namespace abs_c_five_l896_89620

theorem abs_c_five (a b c : ℤ) (h_coprime : Int.gcd a (Int.gcd b c) = 1) 
  (h1 : a = 2 * (b + c)) 
  (h2 : b = 3 * (a + c)) : 
  |c| = 5 :=
by
  sorry

end abs_c_five_l896_89620


namespace value_of_x_l896_89623

theorem value_of_x :
  ∃ x : ℝ, x = 1.13 * 80 :=
sorry

end value_of_x_l896_89623


namespace katy_books_l896_89632

theorem katy_books (june july aug : ℕ) (h1 : june = 8) (h2 : july = 2 * june) (h3 : june + july + aug = 37) :
  july - aug = 3 :=
by sorry

end katy_books_l896_89632


namespace large_A_exists_l896_89686

noncomputable def F_n (n a : ℕ) : ℕ :=
  let q := a / n
  let r := a % n
  q + r

theorem large_A_exists : ∃ n1 n2 n3 n4 n5 n6 : ℕ,
  ∀ a : ℕ, a ≤ 53590 → 
  F_n n6 (F_n n5 (F_n n4 (F_n n3 (F_n n2 (F_n n1 a))))) = 1 :=
by
  sorry

end large_A_exists_l896_89686


namespace solve_x_l896_89625

theorem solve_x (x : ℝ) (h1 : 8 * x^2 + 7 * x - 1 = 0) (h2 : 24 * x^2 + 53 * x - 7 = 0) : x = 1 / 8 :=
by
  sorry

end solve_x_l896_89625


namespace complement_intersection_eq_l896_89627

open Set

def P : Set ℝ := { x | x^2 - 2 * x ≥ 0 }
def Q : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

theorem complement_intersection_eq :
  (compl P) ∩ Q = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end complement_intersection_eq_l896_89627


namespace meaningful_sqrt_domain_l896_89609

theorem meaningful_sqrt_domain (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) :=
by
  sorry

end meaningful_sqrt_domain_l896_89609


namespace perfect_square_expression_l896_89641

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, m^2 = (2 * l - n - k) * (2 * l - n + k) / 2 :=
by 
  sorry

end perfect_square_expression_l896_89641


namespace trigonometric_identity_l896_89658

noncomputable def sin110cos40_minus_cos70sin40 : ℝ := 
  Real.sin (110 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) - 
  Real.cos (70 * Real.pi / 180) * Real.sin (40 * Real.pi / 180)

theorem trigonometric_identity : 
  sin110cos40_minus_cos70sin40 = 1 / 2 := 
by sorry

end trigonometric_identity_l896_89658


namespace gain_percentage_l896_89612

theorem gain_percentage (SP1 SP2 CP: ℝ) (h1 : SP1 = 102) (h2 : SP2 = 144) (h3 : SP1 = CP - 0.15 * CP) :
  ((SP2 - CP) / CP) * 100 = 20 := by
sorry

end gain_percentage_l896_89612


namespace lawnmower_blades_l896_89683

theorem lawnmower_blades (B : ℤ) (h : 8 * B + 7 = 39) : B = 4 :=
by 
  sorry

end lawnmower_blades_l896_89683


namespace jonathan_tax_per_hour_l896_89693

-- Given conditions
def wage : ℝ := 25          -- wage in dollars per hour
def tax_rate : ℝ := 0.024    -- tax rate in decimal

-- Prove statement
theorem jonathan_tax_per_hour :
  (wage * 100) * tax_rate = 60 :=
sorry

end jonathan_tax_per_hour_l896_89693


namespace find_number_l896_89617

-- Define the variables and the conditions as theorems to be proven in Lean.
theorem find_number (x : ℤ) 
  (h1 : (x - 16) % 37 = 0)
  (h2 : (x - 16) / 37 = 23) :
  x = 867 :=
sorry

end find_number_l896_89617


namespace median_circumradius_altitude_inequality_l896_89670

variable (h R m_a m_b m_c : ℝ)

-- Define the condition for the lengths of the medians and other related parameters
-- m_a, m_b, m_c are medians, R is the circumradius, h is the greatest altitude

theorem median_circumradius_altitude_inequality :
  m_a + m_b + m_c ≤ 3 * R + h :=
sorry

end median_circumradius_altitude_inequality_l896_89670


namespace find_values_of_a_and_b_find_square_root_l896_89621

-- Define the conditions
def condition1 (a b : ℤ) : Prop := (2 * b - 2 * a)^3 = -8
def condition2 (a b : ℤ) : Prop := (4 * a + 3 * b)^2 = 9

-- State the problem to prove the values of a and b
theorem find_values_of_a_and_b (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) : 
  a = 3 ∧ b = -1 :=
sorry

-- State the problem to prove the square root of 5a - b
theorem find_square_root (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) (ha : a = 3) (hb : b = -1) :
  ∃ x : ℤ, x^2 = 5 * a - b ∧ (x = 4 ∨ x = -4) :=
sorry

end find_values_of_a_and_b_find_square_root_l896_89621


namespace sam_found_seashells_l896_89635

def seashells_given : Nat := 18
def seashells_left : Nat := 17
def seashells_found : Nat := seashells_given + seashells_left

theorem sam_found_seashells : seashells_found = 35 := by
  sorry

end sam_found_seashells_l896_89635


namespace dice_product_probability_is_one_l896_89689

def dice_probability_product_is_one : Prop :=
  ∀ (a b c d e : ℕ), (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 → 
    (a * b * c * d * e) = 1) ∧
  ∃ (p : ℚ), p = (1/6)^5 ∧ p = 1/7776

theorem dice_product_probability_is_one (a b c d e : ℕ) :
  dice_probability_product_is_one :=
by
  sorry

end dice_product_probability_is_one_l896_89689


namespace hash_op_correct_l896_89639

-- Definition of the custom operation #
def hash_op (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- The theorem to prove that 3 # 8 = 80
theorem hash_op_correct : hash_op 3 8 = 80 :=
by
  sorry

end hash_op_correct_l896_89639


namespace combination_coins_l896_89690

theorem combination_coins : ∃ n : ℕ, n = 23 ∧ ∀ (p n d q : ℕ), 
  p + 5 * n + 10 * d + 25 * q = 50 → n = 23 :=
sorry

end combination_coins_l896_89690


namespace product_xyz_equals_zero_l896_89674

theorem product_xyz_equals_zero (x y z : ℝ) 
    (h1 : x + 2 / y = 2) 
    (h2 : y + 2 / z = 2) 
    : x * y * z = 0 := 
by
  sorry

end product_xyz_equals_zero_l896_89674


namespace exists_divisible_triangle_l896_89660

theorem exists_divisible_triangle (p : ℕ) (n : ℕ) (m : ℕ) (points : Fin m → ℤ × ℤ) 
  (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_pos : 0 < n) (hm_eight : m = 8) 
  (on_circle : ∀ k : Fin m, (points k).fst ^ 2 + (points k).snd ^ 2 = (p ^ n) ^ 2) :
  ∃ (i j k : Fin m), (i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ (∃ d : ℕ, (points i).fst - (points j).fst = p ^ d ∧ 
  (points i).snd - (points j).snd = p ^ d ∧ d ≥ n + 1) :=
sorry

end exists_divisible_triangle_l896_89660


namespace work_completion_l896_89667

/-- 
  Let A, B, and C have work rates where:
  1. A completes the work in 4 days (work rate: 1/4 per day)
  2. C completes the work in 12 days (work rate: 1/12 per day)
  3. Together with B, they complete the work in 2 days (combined work rate: 1/2 per day)
  Prove that B alone can complete the work in 6 days.
--/
theorem work_completion (A B C : ℝ) (x : ℝ)
  (hA : A = 1/4)
  (hC : C = 1/12)
  (h_combined : A + 1/x + C = 1/2) :
  x = 6 := sorry

end work_completion_l896_89667


namespace max_pies_without_ingredients_l896_89624

theorem max_pies_without_ingredients
  (total_pies chocolate_pies berries_pies cinnamon_pies poppy_seeds_pies : ℕ)
  (h1 : total_pies = 60)
  (h2 : chocolate_pies = 1 / 3 * total_pies)
  (h3 : berries_pies = 3 / 5 * total_pies)
  (h4 : cinnamon_pies = 1 / 2 * total_pies)
  (h5 : poppy_seeds_pies = 1 / 5 * total_pies) : 
  total_pies - max chocolate_pies (max berries_pies (max cinnamon_pies poppy_seeds_pies)) = 24 := 
by
  sorry

end max_pies_without_ingredients_l896_89624


namespace pirates_on_schooner_l896_89661

def pirate_problem (N : ℝ) : Prop :=
  let total_pirates       := N
  let non_participants    := 10
  let participants        := total_pirates - non_participants
  let lost_arm            := 0.54 * participants
  let lost_arm_and_leg    := 0.34 * participants
  let lost_leg            := (2 / 3) * total_pirates
  -- The number of pirates who lost only a leg can be calculated.
  let lost_only_leg       := lost_leg - lost_arm_and_leg
  -- The equation that needs to be satisfied
  lost_leg = lost_arm_and_leg + lost_only_leg

theorem pirates_on_schooner : ∃ N : ℝ, N > 10 ∧ pirate_problem N :=
sorry

end pirates_on_schooner_l896_89661


namespace negation_equiv_l896_89680

variable (f : ℝ → ℝ)

theorem negation_equiv :
  ¬ (∀ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  ∃ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) < 0 := by
sorry

end negation_equiv_l896_89680


namespace necessary_but_not_sufficient_l896_89633

def M := {x : ℝ | 0 < x ∧ x ≤ 3}
def N := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient (a : ℝ) (haM : a ∈ M) : (a ∈ N → a ∈ M) ∧ ¬(a ∈ M → a ∈ N) :=
by {
  sorry
}

end necessary_but_not_sufficient_l896_89633


namespace hyperbola_eccentricity_l896_89684

theorem hyperbola_eccentricity (m : ℤ) (h1 : -2 < m) (h2 : m < 2) : 
  let a := m
  let b := (4 - m^2).sqrt 
  let c := (a^2 + b^2).sqrt
  let e := c / a
  e = 2 := by
sorry

end hyperbola_eccentricity_l896_89684


namespace sum_of_coordinates_eq_69_l896_89697

theorem sum_of_coordinates_eq_69 {f k : ℝ → ℝ} (h₁ : f 4 = 8) (h₂ : ∀ x, k x = (f x)^2 + 1) : 4 + k 4 = 69 :=
by
  sorry

end sum_of_coordinates_eq_69_l896_89697


namespace compound_interest_calculation_l896_89616

theorem compound_interest_calculation :
  let SI := (1833.33 * 16 * 6) / 100
  let CI := 2 * SI
  let principal_ci := 8000
  let rate_ci := 20
  let n := Real.log (1.4399995) / Real.log (1 + rate_ci / 100)
  n = 2 := by
  sorry

end compound_interest_calculation_l896_89616


namespace problem_statement_l896_89656

theorem problem_statement (x : ℕ) (h : 4 * (3^x) = 2187) : (x + 2) * (x - 2) = 21 := 
by
  sorry

end problem_statement_l896_89656


namespace inequality_solution_l896_89671

theorem inequality_solution (x : ℝ) : 4 * x - 1 < 0 ↔ x < 1 / 4 := 
sorry

end inequality_solution_l896_89671


namespace kitty_cleaning_time_l896_89604

theorem kitty_cleaning_time
    (picking_up_toys : ℕ := 5)
    (vacuuming : ℕ := 20)
    (dusting_furniture : ℕ := 10)
    (total_time_4_weeks : ℕ := 200)
    (weeks : ℕ := 4)
    : (total_time_4_weeks - weeks * (picking_up_toys + vacuuming + dusting_furniture)) / weeks = 15 := by
    sorry

end kitty_cleaning_time_l896_89604


namespace total_distance_run_l896_89662

def track_meters : ℕ := 9
def laps_already_run : ℕ := 6
def laps_to_run : ℕ := 5

theorem total_distance_run :
  (laps_already_run * track_meters) + (laps_to_run * track_meters) = 99 := by
  sorry

end total_distance_run_l896_89662


namespace eval_expr_ceil_floor_l896_89629

theorem eval_expr_ceil_floor (x y : ℚ) (h1 : x = 7 / 3) (h2 : y = -7 / 3) :
  (⌈x⌉ + ⌊y⌋ = 0) :=
sorry

end eval_expr_ceil_floor_l896_89629


namespace local_maximum_at_neg2_l896_89694

noncomputable def y (x : ℝ) : ℝ :=
  (1/3) * x^3 - 4 * x + 4

theorem local_maximum_at_neg2 :
  ∃ x : ℝ, x = -2 ∧ 
           y x = 28/3 ∧
           (∀ ε > 0, ∃ δ > 0, ∀ z, abs (z + 2) < δ → y z < y (-2)) := by
  sorry

end local_maximum_at_neg2_l896_89694


namespace min_fraction_value_l896_89644

theorem min_fraction_value
    (a x y : ℕ)
    (h1 : a > 100)
    (h2 : x > 100)
    (h3 : y > 100)
    (h4 : y^2 - 1 = a^2 * (x^2 - 1))
    : a / x ≥ 2 := 
sorry

end min_fraction_value_l896_89644


namespace linear_equation_in_options_l896_89672

def is_linear_equation_with_one_variable (eqn : String) : Prop :=
  eqn = "3 - 2x = 5"

theorem linear_equation_in_options :
  is_linear_equation_with_one_variable "3 - 2x = 5" :=
by
  sorry

end linear_equation_in_options_l896_89672


namespace sum_of_digits_B_l896_89682

/- 
  Let A be the natural number formed by concatenating integers from 1 to 100.
  Let B be the smallest possible natural number formed by removing 100 digits from A.
  We need to prove that the sum of the digits of B equals 486.
-/
def A : ℕ := sorry -- construct the natural number 1234567891011121314...99100

def sum_of_digits (n : ℕ) : ℕ := sorry -- function to calculate the sum of digits of a natural number

def B : ℕ := sorry -- construct the smallest possible number B by removing 100 digits from A

theorem sum_of_digits_B : sum_of_digits B = 486 := sorry

end sum_of_digits_B_l896_89682


namespace speed_of_first_boy_proof_l896_89673

noncomputable def speed_of_first_boy := 5.9

theorem speed_of_first_boy_proof :
  ∀ (x : ℝ) (t : ℝ) (d : ℝ),
    (d = x * t) → (d = (x - 5.6) * 35) →
    d = 10.5 →
    t = 35 →
    x = 5.9 := 
by
  intros x t d h1 h2 h3 h4
  sorry

end speed_of_first_boy_proof_l896_89673


namespace broken_glass_pieces_l896_89675

theorem broken_glass_pieces (x : ℕ) 
    (total_pieces : ℕ := 100) 
    (safe_fee : ℕ := 3) 
    (compensation : ℕ := 5) 
    (total_fee : ℕ := 260) 
    (h : safe_fee * (total_pieces - x) - compensation * x = total_fee) : x = 5 := by
  sorry

end broken_glass_pieces_l896_89675


namespace sum_pos_implies_one_pos_l896_89638

theorem sum_pos_implies_one_pos (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 := 
sorry

end sum_pos_implies_one_pos_l896_89638


namespace find_f_neg_one_l896_89630

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x then x^2 + 2 * x else - ( (x^2) + (2 * x))

theorem find_f_neg_one : 
  f (-1) = -3 :=
by 
  sorry

end find_f_neg_one_l896_89630


namespace kevin_food_expenditure_l896_89681

/-- Samuel and Kevin have a total budget of $20. Samuel spends $14 on his ticket 
and $6 on drinks and food. Kevin spends $2 on drinks. Prove that Kevin spent $4 on food. -/
theorem kevin_food_expenditure :
  ∀ (total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food : ℝ),
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_drinks_food = 6 →
  kevin_ticket = 14 →
  kevin_drinks = 2 →
  kevin_ticket + kevin_drinks + kevin_food = total_budget / 2 →
  kevin_food = 4 :=
by
  intros total_budget samuel_ticket samuel_drinks_food kevin_ticket kevin_drinks kevin_food
  intro h_budget h_sam_ticket h_sam_food_drinks h_kev_ticket h_kev_drinks h_kev_budget
  sorry

end kevin_food_expenditure_l896_89681


namespace minimum_z_value_l896_89648

theorem minimum_z_value (x y : ℝ) (h : (x - 2)^2 + (y - 3)^2 = 1) : x^2 + y^2 ≥ 14 - 2 * Real.sqrt 13 :=
sorry

end minimum_z_value_l896_89648


namespace proof_problem_l896_89602

variable {a_n : ℕ → ℤ}
variable {b_n : ℕ → ℤ}
variable {c_n : ℕ → ℤ}
variable {T_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}

-- Conditions

-- 1. The common difference d of the arithmetic sequence {a_n} is greater than 0
def common_difference_positive (d : ℤ) : Prop :=
  d > 0

-- 2. a_2 and a_5 are the two roots of the equation x^2 - 12x + 27 = 0
def roots_of_quadratic (a2 a5 : ℤ) : Prop :=
  a2^2 - 12 * a2 + 27 = 0 ∧ a5^2 - 12 * a5 + 27 = 0

-- 3. The sum of the first n terms of the sequence {b_n} is S_n, and it is given that S_n = (3 / 2)(b_n - 1)
def sum_of_b_n (S_n b_n : ℕ → ℤ) : Prop :=
  ∀ n, S_n n = 3/2 * (b_n n - 1)

-- Define the sequences to display further characteristics

-- 1. Find the general formula for the sequences {a_n} and {b_n}
def general_formula_a (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 2 * n - 1

def general_formula_b (b : ℕ → ℤ) : Prop :=
  ∀ n, b n = 3 ^ n

-- 2. Check if c_n = a_n * b_n and find the sum T_n
def c_n_equals_a_n_times_b_n (a b : ℕ → ℤ) (c : ℕ → ℤ) : Prop :=
  ∀ n, c n = a n * b n

def sum_T_n (T c : ℕ → ℤ) : Prop :=
  ∀ n, T n = 3 + (n - 1) * 3^(n + 1)

theorem proof_problem 
  (d : ℤ)
  (a2 a5 : ℤ)
  (S_n b_n : ℕ → ℤ)
  (a_n b_n c_n T_n : ℕ → ℤ) :
  common_difference_positive d ∧
  roots_of_quadratic a2 a5 ∧ 
  sum_of_b_n S_n b_n ∧ 
  general_formula_a a_n ∧ 
  general_formula_b b_n ∧ 
  c_n_equals_a_n_times_b_n a_n b_n c_n ∧ 
  sum_T_n T_n c_n :=
sorry

end proof_problem_l896_89602


namespace radius_of_circle_l896_89665

theorem radius_of_circle (d : ℝ) (h : d = 22) : (d / 2) = 11 := by
  sorry

end radius_of_circle_l896_89665


namespace circles_chord_length_l896_89654

theorem circles_chord_length (r1 r2 r3 : ℕ) (m n p : ℕ) (h1 : r1 = 4) (h2 : r2 = 10) (h3 : r3 = 14)
(h4 : gcd m p = 1) (h5 : ¬ (∃ (k : ℕ), k^2 ∣ n)) : m + n + p = 19 :=
by
  sorry

end circles_chord_length_l896_89654


namespace x_power_12_l896_89645

theorem x_power_12 (x : ℝ) (h : x + 1 / x = 2) : x^12 = 1 :=
by sorry

end x_power_12_l896_89645


namespace find_y_l896_89634

theorem find_y (steps distance : ℕ) (total_steps : ℕ) (marking_step : ℕ)
  (h1 : total_steps = 8)
  (h2 : distance = 48)
  (h3 : marking_step = 6) :
  steps = distance / total_steps * marking_step → steps = 36 :=
by
  intros
  sorry

end find_y_l896_89634


namespace roots_polynomial_value_l896_89600

theorem roots_polynomial_value (a b c : ℝ) 
  (h1 : a + b + c = 15)
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 12) :
  (2 + a) * (2 + b) * (2 + c) = 130 := 
by
  sorry

end roots_polynomial_value_l896_89600


namespace find_number_l896_89615

theorem find_number :
  ∃ x : ℝ, (x - 1.9) * 1.5 + 32 / 2.5 = 20 ∧ x = 13.9 :=
by
  sorry

end find_number_l896_89615


namespace division_of_decimals_l896_89699

theorem division_of_decimals : 0.08 / 0.002 = 40 :=
by
  sorry

end division_of_decimals_l896_89699


namespace mary_days_eq_11_l896_89679

variable (x : ℝ) -- Number of days Mary takes to complete the work
variable (m_eff : ℝ) -- Efficiency of Mary (work per day)
variable (r_eff : ℝ) -- Efficiency of Rosy (work per day)

-- Given conditions
axiom rosy_efficiency : r_eff = 1.1 * m_eff
axiom rosy_days : r_eff * 10 = 1

-- Define the efficiency of Mary in terms of days
axiom mary_efficiency : m_eff = 1 / x

-- The theorem to prove
theorem mary_days_eq_11 : x = 11 :=
by
  sorry

end mary_days_eq_11_l896_89679


namespace find_common_difference_l896_89611

variable (a : ℕ → ℤ)  -- define the arithmetic sequence as a function from ℕ to ℤ
variable (d : ℤ)      -- define the common difference

-- Define the conditions
def conditions := (a 5 = 10) ∧ (a 12 = 31)

-- Define the formula for the nth term of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (n : ℕ) := a 1 + d * (n - 1)

-- Prove that the common difference d is 3 given the conditions
theorem find_common_difference (h : conditions a) : d = 3 :=
sorry

end find_common_difference_l896_89611


namespace vector_c_correct_l896_89650

theorem vector_c_correct (a b c : ℤ × ℤ) (h_a : a = (1, -3)) (h_b : b = (-2, 4))
    (h_condition : 4 • a + (3 • b - 2 • a) + c = (0, 0)) :
    c = (4, -6) :=
by 
  -- The proof steps go here, but we'll skip them with 'sorry' for now.
  sorry

end vector_c_correct_l896_89650


namespace estimate_value_l896_89685

theorem estimate_value : 1 < (3 - Real.sqrt 3) ∧ (3 - Real.sqrt 3) < 2 :=
by
  have h₁ : Real.sqrt 18 = 3 * Real.sqrt 2 :=
    by sorry
  have h₂ : Real.sqrt 6 = Real.sqrt 3 * Real.sqrt 2 :=
    by sorry
  have h₃ : (Real.sqrt 18 - Real.sqrt 6) / Real.sqrt 2 = (3 * Real.sqrt 2 - Real.sqrt 3 * Real.sqrt 2) / Real.sqrt 2 :=
    by sorry
  have h₄ : (3 * Real.sqrt 2 - Real.sqrt 3 * Real.sqrt 2) / Real.sqrt 2 = 3 - Real.sqrt 3 :=
    by sorry
  have h₅ : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 :=
    by sorry
  sorry

end estimate_value_l896_89685


namespace minimum_xy_l896_89651

theorem minimum_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : 
  x * y ≥ 18 :=
sorry

end minimum_xy_l896_89651


namespace find_k_l896_89649

theorem find_k (k : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, 3 * x - k * y + c = 0) ∧ (∀ x y : ℝ, k * x + y + 1 = 0 → 3 * k + (-k) = 0) → k = 0 :=
by
  sorry

end find_k_l896_89649


namespace paul_completion_time_l896_89669

theorem paul_completion_time :
  let george_rate := 1 / 15
  let remaining_work := 2 / 5
  let combined_rate (P : ℚ) := george_rate + P
  let P_work := 4 * combined_rate P = remaining_work
  let paul_rate := 13 / 90
  let total_work := 1
  let time_paul_alone := total_work / paul_rate
  P_work → time_paul_alone = (90 / 13) := by
  intros
  -- all necessary definitions and conditions are used
  sorry

end paul_completion_time_l896_89669


namespace circle_equation_l896_89698

def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem circle_equation : ∃ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 5 ∧ (y = parabola x) ∧ (x = -1 ∨ x = 3 ∨ (x = 0 ∧ y = -3)) :=
by { sorry }

end circle_equation_l896_89698


namespace triangle_angle_relation_l896_89653

theorem triangle_angle_relation 
  (a b c : ℝ)
  (α β γ : ℝ)
  (h1 : b = (a + c) / Real.sqrt 2)
  (h2 : β = (α + γ) / 2)
  (h3 : c > a)
  : γ = α + 90 :=
sorry

end triangle_angle_relation_l896_89653


namespace AM_GM_contradiction_l896_89676

open Real

theorem AM_GM_contradiction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
      ¬ (6 < a + 4 / b ∧ 6 < b + 9 / c ∧ 6 < c + 16 / a) := by
  sorry

end AM_GM_contradiction_l896_89676


namespace part_I_part_II_l896_89642

open Real

def f (x m n : ℝ) := abs (x - m) + abs (x + n)

theorem part_I (m n M : ℝ) (h1 : m + n = 9) (h2 : ∀ x : ℝ, f x m n ≥ M) : M ≤ 9 := 
sorry

theorem part_II (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9) : (a + b) * (a^3 + b^3) ≥ 81 := 
sorry

end part_I_part_II_l896_89642


namespace vertical_asymptote_once_l896_89655

theorem vertical_asymptote_once (c : ℝ) : 
  (∀ x : ℝ, (x^2 + 2*x + c) / (x^2 - x - 12) = (x^2 + 2*x + c) / ((x - 4) * (x + 3))) → 
  (c = -24 ∨ c = -3) :=
by 
  sorry

end vertical_asymptote_once_l896_89655


namespace simplify_expr_l896_89610

theorem simplify_expr : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by 
  sorry

end simplify_expr_l896_89610


namespace value_of_t_plus_k_l896_89601

noncomputable def f (x t : ℝ) : ℝ := x^3 + (t - 1) * x^2 - 1

theorem value_of_t_plus_k (k t : ℝ)
  (h1 : k ≠ 0)
  (h2 : ∀ x, f x t = 2 * x - 1)
  (h3 : ∃ x₁ x₂, f x₁ t = 2 * x₁ - 1 ∧ f x₂ t = 2 * x₂ - 1) :
  t + k = 7 :=
sorry

end value_of_t_plus_k_l896_89601


namespace find_m_of_quad_roots_l896_89619

theorem find_m_of_quad_roots
  (a b : ℝ) (m : ℝ)
  (ha : a = 5)
  (hb : b = -4)
  (h_roots : ∀ x : ℂ, (x = (2 + Complex.I * Real.sqrt 143) / 5 ∨ x = (2 - Complex.I * Real.sqrt 143) / 5) →
                     (a * x^2 + b * x + m = 0)) :
  m = 7.95 :=
by
  -- Proof goes here
  sorry

end find_m_of_quad_roots_l896_89619


namespace calculate_a_plus_b_l896_89613

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def g (x : ℝ) : ℝ := 3 * x - 7

theorem calculate_a_plus_b (a b : ℝ) (h : ∀ x : ℝ, g (f a b x) = 4 * x + 6) : a + b = 17 / 3 :=
by
  sorry

end calculate_a_plus_b_l896_89613


namespace sum_q_p_values_l896_89666

def p (x : ℤ) : ℤ := x^2 - 4
def q (x : ℤ) : ℤ := -x

def q_p_composed (x : ℤ) : ℤ := q (p x)

theorem sum_q_p_values :
  q_p_composed (-3) + q_p_composed (-2) + q_p_composed (-1) + q_p_composed 0 + 
  q_p_composed 1 + q_p_composed 2 + q_p_composed 3 = 0 := by
  sorry

end sum_q_p_values_l896_89666


namespace ram_weight_increase_percentage_l896_89677

theorem ram_weight_increase_percentage :
  ∃ r s r_new: ℝ,
  r / s = 4 / 5 ∧ 
  r + s = 72 ∧ 
  s * 1.19 = 47.6 ∧
  r_new = 82.8 - 47.6 ∧ 
  (r_new - r) / r * 100 = 10 :=
by
  sorry

end ram_weight_increase_percentage_l896_89677


namespace P_lt_Q_l896_89691

variable {x : ℝ}

def P (x : ℝ) : ℝ := (x - 2) * (x - 4)
def Q (x : ℝ) : ℝ := (x - 3) ^ 2

theorem P_lt_Q : P x < Q x := by
  sorry

end P_lt_Q_l896_89691


namespace pony_jeans_discount_rate_l896_89646

noncomputable def fox_price : ℝ := 15
noncomputable def pony_price : ℝ := 18

-- Define the conditions
def total_savings (F P : ℝ) : Prop :=
  3 * (F / 100 * fox_price) + 2 * (P / 100 * pony_price) = 9

def discount_sum (F P : ℝ) : Prop :=
  F + P = 22

-- Main statement to be proven
theorem pony_jeans_discount_rate (F P : ℝ) (h1 : total_savings F P) (h2 : discount_sum F P) : P = 10 :=
by
  -- Proof goes here
  sorry

end pony_jeans_discount_rate_l896_89646


namespace negation_of_p_is_neg_p_l896_89636

-- Define the proposition p
def p : Prop := ∀ x : ℝ, x > 3 → x^3 - 27 > 0

-- Define the negation of proposition p
def neg_p : Prop := ∃ x : ℝ, x > 3 ∧ x^3 - 27 ≤ 0

-- The Lean statement that proves the problem
theorem negation_of_p_is_neg_p : ¬ p ↔ neg_p := by
  sorry

end negation_of_p_is_neg_p_l896_89636


namespace james_monthly_earnings_l896_89643

theorem james_monthly_earnings :
  let initial_subscribers := 150
  let gifted_subscribers := 50
  let rate_per_subscriber := 9
  let total_subscribers := initial_subscribers + gifted_subscribers
  let total_earnings := total_subscribers * rate_per_subscriber
  total_earnings = 1800 := by
  sorry

end james_monthly_earnings_l896_89643


namespace raisin_fraction_of_mixture_l896_89652

noncomputable def raisin_nut_cost_fraction (R : ℝ) : ℝ :=
  let raisin_cost := 3 * R
  let nut_cost := 4 * (4 * R)
  let total_cost := raisin_cost + nut_cost
  raisin_cost / total_cost

theorem raisin_fraction_of_mixture (R : ℝ) : raisin_nut_cost_fraction R = 3 / 19 :=
by
  sorry

end raisin_fraction_of_mixture_l896_89652


namespace bookstore_discount_l896_89614

noncomputable def discount_percentage (total_spent : ℝ) (over_22 : List ℝ) (under_20 : List ℝ) : ℝ :=
  let disc_over_22 := over_22.map (fun p => p * (1 - 0.30))
  let total_over_22 := disc_over_22.sum
  let total_with_under_20 := total_over_22 + 21
  let total_under_20 := under_20.sum
  let discount_received := total_spent - total_with_under_20
  let discount_percentage := (total_under_20 - discount_received) / total_under_20 * 100
  discount_percentage

theorem bookstore_discount :
  discount_percentage 95 [25.00, 35.00] [18.00, 12.00, 10.00] = 20 := by
  sorry

end bookstore_discount_l896_89614


namespace eugene_used_six_boxes_of_toothpicks_l896_89628

-- Define the given conditions
def toothpicks_per_card : ℕ := 75
def total_cards : ℕ := 52
def unused_cards : ℕ := 16
def toothpicks_per_box : ℕ := 450

-- Compute the required result
theorem eugene_used_six_boxes_of_toothpicks :
  ((total_cards - unused_cards) * toothpicks_per_card) / toothpicks_per_box = 6 :=
by
  sorry

end eugene_used_six_boxes_of_toothpicks_l896_89628


namespace petya_points_l896_89603

noncomputable def points_after_disqualification : ℕ :=
4

theorem petya_points (players: ℕ) (initial_points: ℕ) (disqualified: ℕ) (new_points: ℕ) : 
  players = 10 → 
  initial_points < (players * (players - 1) / 2) / players → 
  disqualified = 2 → 
  (players - disqualified) * (players - disqualified - 1) / 2 = new_points →
  new_points / (players - disqualified) < points_after_disqualification →
  points_after_disqualification > new_points / (players - disqualified) →
  points_after_disqualification = 4 :=
by 
  intros 
  exact sorry

end petya_points_l896_89603


namespace cookies_per_bag_l896_89637

theorem cookies_per_bag (n_bags : ℕ) (total_cookies : ℕ) (n_candies : ℕ) (h_bags : n_bags = 26) (h_cookies : total_cookies = 52) (h_candies : n_candies = 15) : (total_cookies / n_bags) = 2 :=
by sorry

end cookies_per_bag_l896_89637


namespace hcf_two_numbers_l896_89618

theorem hcf_two_numbers (H a b : ℕ) (coprime_ab : Nat.gcd a b = 1) 
    (lcm_factors : a * b = 150) (larger_num : H * a = 450 ∨ H * b = 450) : H = 30 := 
by
  sorry

end hcf_two_numbers_l896_89618


namespace wholesale_cost_calc_l896_89696

theorem wholesale_cost_calc (wholesale_cost : ℝ) 
  (h_profit : 0.15 * wholesale_cost = 28 - wholesale_cost) : 
  wholesale_cost = 28 / 1.15 :=
by
  sorry

end wholesale_cost_calc_l896_89696


namespace fifteenth_term_ratio_l896_89664

noncomputable def U (n : ℕ) (c f : ℚ) := n * (2 * c + (n - 1) * f) / 2
noncomputable def V (n : ℕ) (g h : ℚ) := n * (2 * g + (n - 1) * h) / 2

theorem fifteenth_term_ratio (c f g h : ℚ)
  (h1 : ∀ n : ℕ, (n > 0) → (U n c f) / (V n g h) = (5 * (n * n) + 3 * n + 2) / (3 * (n * n) + 2 * n + 30)) :
  (c + 14 * f) / (g + 14 * h) = 125 / 99 :=
by
  sorry

end fifteenth_term_ratio_l896_89664


namespace problem_statement_l896_89640

noncomputable def f : ℕ+ → ℝ := sorry

theorem problem_statement (x : ℕ+) :
  (f 1 = 1) →
  (∀ x, f (x + 1) = (2 * f x) / (f x + 2)) →
  f x = 2 / (x + 1) := 
sorry

end problem_statement_l896_89640


namespace probability_outside_circle_is_7_over_9_l896_89626

noncomputable def probability_point_outside_circle :
    ℚ :=
sorry

theorem probability_outside_circle_is_7_over_9 :
    probability_point_outside_circle = 7 / 9 :=
sorry

end probability_outside_circle_is_7_over_9_l896_89626


namespace find_nth_number_in_s_l896_89659

def s (k : ℕ) : ℕ := 8 * k + 5

theorem find_nth_number_in_s (n : ℕ) (number_in_s : ℕ) (h : number_in_s = 573) :
  ∃ k : ℕ, s k = number_in_s ∧ n = k + 1 := 
sorry

end find_nth_number_in_s_l896_89659


namespace arithmetic_sequence_term_count_l896_89668

theorem arithmetic_sequence_term_count (a1 d an : ℤ) (h₀ : a1 = -6) (h₁ : d = 5) (h₂ : an = 59) :
  ∃ n : ℤ, an = a1 + (n - 1) * d ∧ n = 14 :=
by
  sorry

end arithmetic_sequence_term_count_l896_89668


namespace aaron_brothers_l896_89663

theorem aaron_brothers (A : ℕ) (h1 : 6 = 2 * A - 2) : A = 4 :=
by
  sorry

end aaron_brothers_l896_89663


namespace total_athletes_l896_89647

theorem total_athletes (g : ℕ) (p : ℕ)
  (h₁ : g = 7)
  (h₂ : p = 5)
  (h₃ : 3 * (g + p - 1) = 33) : 
  3 * (g + p - 1) = 33 :=
sorry

end total_athletes_l896_89647


namespace sum_difference_l896_89687

def even_sum (n : ℕ) : ℕ :=
  n * (n + 1)

def odd_sum (n : ℕ) : ℕ :=
  n^2

theorem sum_difference : even_sum 100 - odd_sum 100 = 100 := by
  sorry

end sum_difference_l896_89687


namespace total_pitches_missed_l896_89657

theorem total_pitches_missed (tokens_to_pitches : ℕ → ℕ) 
  (macy_used : ℕ) (piper_used : ℕ) 
  (macy_hits : ℕ) (piper_hits : ℕ) 
  (h1 : tokens_to_pitches 1 = 15) 
  (h_macy_used : macy_used = 11) 
  (h_piper_used : piper_used = 17) 
  (h_macy_hits : macy_hits = 50) 
  (h_piper_hits : piper_hits = 55) :
  let total_pitches := tokens_to_pitches macy_used + tokens_to_pitches piper_used
  let total_hits := macy_hits + piper_hits
  total_pitches - total_hits = 315 :=
by
  sorry

end total_pitches_missed_l896_89657


namespace percentage_markup_l896_89607

theorem percentage_markup (CP SP : ℕ) (hCP : CP = 800) (hSP : SP = 1000) :
  let Markup := SP - CP
  let PercentageMarkup := (Markup : ℚ) / CP * 100
  PercentageMarkup = 25 := by
  sorry

end percentage_markup_l896_89607


namespace simplify_and_rationalize_l896_89606

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l896_89606


namespace infinite_powers_of_two_in_sequence_l896_89692

theorem infinite_powers_of_two_in_sequence :
  ∃ᶠ n in at_top, ∃ k : ℕ, ∃ a : ℕ, (a = ⌊n * Real.sqrt 2⌋ ∧ a = 2^k) :=
sorry

end infinite_powers_of_two_in_sequence_l896_89692


namespace gcd_of_2475_and_7350_is_225_l896_89631

-- Definitions and conditions based on the factorization of the given numbers
def factor_2475 := (5^2 * 3^2 * 11)
def factor_7350 := (2 * 3^2 * 5^2 * 7)

-- Proof problem: showing the GCD of 2475 and 7350 is 225
theorem gcd_of_2475_and_7350_is_225 : Nat.gcd 2475 7350 = 225 :=
by
  -- Formal proof would go here
  sorry

end gcd_of_2475_and_7350_is_225_l896_89631


namespace articles_count_l896_89695

noncomputable def cost_price_per_article : ℝ := 1
noncomputable def selling_price_per_article (x : ℝ) : ℝ := x / 16
noncomputable def profit : ℝ := 0.50

theorem articles_count (x : ℝ) (h1 : cost_price_per_article * x = selling_price_per_article x * 16)
                       (h2 : selling_price_per_article 16 = cost_price_per_article * (1 + profit)) :
  x = 24 :=
by
  sorry

end articles_count_l896_89695


namespace find_value_of_a2_b2_c2_l896_89622

variable {a b c : ℝ}

theorem find_value_of_a2_b2_c2
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 := 
sorry

end find_value_of_a2_b2_c2_l896_89622


namespace axis_of_symmetry_exists_l896_89678

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

theorem axis_of_symmetry_exists :
  ∃ k : ℤ, ∃ x : ℝ, (x = -5 * Real.pi / 12 ∧ f x = Real.sin (Real.pi / 2 + k * Real.pi))
  ∨ (x = Real.pi / 12 + k * Real.pi / 2 ∧ f x = Real.sin (Real.pi / 2 + k * Real.pi)) :=
sorry

end axis_of_symmetry_exists_l896_89678
