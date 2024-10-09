import Mathlib

namespace yen_checking_account_l1587_158706

theorem yen_checking_account (savings : ℕ) (total : ℕ) (checking : ℕ) (h1 : savings = 3485) (h2 : total = 9844) (h3 : checking = total - savings) :
  checking = 6359 :=
by
  rw [h1, h2] at h3
  exact h3

end yen_checking_account_l1587_158706


namespace radius_of_inscribed_circle_l1587_158729

theorem radius_of_inscribed_circle (a b c r : ℝ) (h : a^2 + b^2 = c^2) :
  r = a + b - c :=
sorry

end radius_of_inscribed_circle_l1587_158729


namespace f_at_7_l1587_158740

noncomputable def f (x : ℝ) (a b c d : ℝ) := a * x^7 + b * x^5 + c * x^3 + d * x + 5

theorem f_at_7 (a b c d : ℝ) (h : f (-7) a b c d = -7) : f 7 a b c d = 17 := 
by
  sorry

end f_at_7_l1587_158740


namespace common_difference_of_arithmetic_sequence_l1587_158758

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : a 1 + a 9 = 10)
  (h2 : a 2 = -1)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l1587_158758


namespace chocolates_vs_gums_l1587_158782

theorem chocolates_vs_gums 
    (c g : ℝ) 
    (Kolya_claim : 2 * c > 5 * g) 
    (Sasha_claim : ¬ ( 3 * c > 8 * g )) : 
    7 * c ≤ 19 * g := 
sorry

end chocolates_vs_gums_l1587_158782


namespace base_of_log_is_176_l1587_158755

theorem base_of_log_is_176 
    (x : ℕ)
    (h : ∃ q r : ℕ, x = 19 * q + r ∧ q = 9 ∧ r = 5) :
    x = 176 :=
by
  sorry

end base_of_log_is_176_l1587_158755


namespace D_96_equals_112_l1587_158785

def multiplicative_decompositions (n : ℕ) : ℕ :=
  sorry -- Define how to find the number of multiplicative decompositions

theorem D_96_equals_112 : multiplicative_decompositions 96 = 112 :=
  sorry

end D_96_equals_112_l1587_158785


namespace arrange_numbers_l1587_158728

theorem arrange_numbers :
  (2 : ℝ) ^ 1000 < (5 : ℝ) ^ 500 ∧ (5 : ℝ) ^ 500 < (3 : ℝ) ^ 750 :=
by
  sorry

end arrange_numbers_l1587_158728


namespace abs_neg_five_is_five_l1587_158776

theorem abs_neg_five_is_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_is_five_l1587_158776


namespace problem_statement_l1587_158795

theorem problem_statement : ((26.3 * 12 * 20) / 3 + 125 - Real.sqrt 576 = 21141) :=
by
  sorry

end problem_statement_l1587_158795


namespace center_of_circle_from_diameter_l1587_158787

theorem center_of_circle_from_diameter (x1 y1 x2 y2 : ℝ) 
  (h1 : x1 = 3) (h2 : y1 = -3) (h3 : x2 = 13) (h4 : y2 = 17) :
  (x1 + x2) / 2 = 8 ∧ (y1 + y2) / 2 = 7 :=
by
  sorry

end center_of_circle_from_diameter_l1587_158787


namespace parabola_vertex_n_l1587_158780

theorem parabola_vertex_n (x y : ℝ) (h : y = -3 * x^2 - 24 * x - 72) : ∃ m n : ℝ, (m, n) = (-4, -24) :=
by
  sorry

end parabola_vertex_n_l1587_158780


namespace calculate_fraction_l1587_158790

theorem calculate_fraction (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -1) : 
  (1 / (x - 1)) - (2 / (x^2 - 1)) = 1 / (x + 1) :=
by
  sorry

end calculate_fraction_l1587_158790


namespace type_B_ratio_l1587_158777

theorem type_B_ratio
    (num_A : ℕ)
    (total_bricks : ℕ)
    (other_bricks : ℕ)
    (h1 : num_A = 40)
    (h2 : total_bricks = 150)
    (h3 : other_bricks = 90) :
    (total_bricks - num_A - other_bricks) / num_A = 1 / 2 :=
by
  sorry

end type_B_ratio_l1587_158777


namespace cone_volume_l1587_158722

theorem cone_volume (r h : ℝ) (h_cylinder_vol : π * r^2 * h = 72 * π) : 
  (1 / 3) * π * r^2 * (h / 2) = 12 * π := by
  sorry

end cone_volume_l1587_158722


namespace power_first_digits_l1587_158723

theorem power_first_digits (n : ℕ) (h1 : ∀ k : ℕ, n ≠ 10^k) : ∃ j k : ℕ, 1973 ≤ n^j / 10^k ∧ n^j / 10^k < 1974 := by
  sorry

end power_first_digits_l1587_158723


namespace solution_l1587_158798

namespace ProofProblem

variables (a b : ℝ)

def five_times_a_minus_b_eq_60 := 5 * a - b = 60
def six_times_a_plus_b_lt_90 := 6 * a + b < 90

theorem solution (h1 : five_times_a_minus_b_eq_60 a b) (h2 : six_times_a_plus_b_lt_90 a b) :
  a < 150 / 11 ∧ b < 8.18 :=
sorry

end ProofProblem

end solution_l1587_158798


namespace container_ratio_l1587_158788

theorem container_ratio (V1 V2 V3 : ℝ)
  (h1 : (3 / 4) * V1 = (5 / 8) * V2)
  (h2 : (5 / 8) * V2 = (1 / 2) * V3) :
  V1 / V3 = 1 / 2 :=
by
  sorry

end container_ratio_l1587_158788


namespace quotient_of_2213_div_13_in_base4_is_53_l1587_158718

-- Definitions of the numbers in base 4
def n₁ : ℕ := 2 * 4^3 + 2 * 4^2 + 1 * 4^1 + 3 * 4^0  -- 2213_4 in base 10
def n₂ : ℕ := 1 * 4^1 + 3 * 4^0  -- 13_4 in base 10

-- The correct quotient in base 4 (converted from quotient in base 10)
def expected_quotient : ℕ := 5 * 4^1 + 3 * 4^0  -- 53_4 in base 10

-- The proposition we want to prove
theorem quotient_of_2213_div_13_in_base4_is_53 : n₁ / n₂ = expected_quotient := by
  sorry

end quotient_of_2213_div_13_in_base4_is_53_l1587_158718


namespace sqrt_205_between_14_and_15_l1587_158724

theorem sqrt_205_between_14_and_15 : 14 < Real.sqrt 205 ∧ Real.sqrt 205 < 15 := 
by
  sorry

end sqrt_205_between_14_and_15_l1587_158724


namespace value_of_y_l1587_158768

theorem value_of_y (x y : ℝ) (h1 : x^(2 * y) = 9) (h2 : x = 3) : y = 1 := by
  sorry

end value_of_y_l1587_158768


namespace max_profit_l1587_158713

noncomputable def profit (x : ℕ) : ℝ := -0.15 * (x : ℝ)^2 + 3.06 * (x : ℝ) + 30

theorem max_profit :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 15 ∧ profit x = 45.6 ∧ ∀ y : ℕ, 0 ≤ y ∧ y ≤ 15 → profit y ≤ profit x :=
by
  sorry

end max_profit_l1587_158713


namespace sum_is_two_l1587_158753

noncomputable def compute_sum (x : ℂ) (hx : x^7 = 1) (hx_ne : x ≠ 1) : ℂ :=
  (x^2 / (x - 1)) + (x^4 / (x^2 - 1)) + (x^6 / (x^3 - 1)) + (x^8 / (x^4 - 1)) + (x^10 / (x^5 - 1)) + (x^12 / (x^6 - 1))

theorem sum_is_two (x : ℂ) (hx : x^7 = 1) (hx_ne : x ≠ 1) : compute_sum x hx hx_ne = 2 :=
by
  sorry

end sum_is_two_l1587_158753


namespace min_units_for_profitability_profitability_during_epidemic_l1587_158725

-- Conditions
def assembly_line_cost : ℝ := 1.8
def selling_price_per_product : ℝ := 0.1
def max_annual_output : ℕ := 100

noncomputable def production_cost (x : ℕ) : ℝ := 5 + 135 / (x + 1)

-- Part 1: Prove Minimum x for profitability
theorem min_units_for_profitability (x : ℕ) :
  (10 - (production_cost x)) * x - assembly_line_cost > 0 ↔ x ≥ 63 := sorry

-- Part 2: Profitability and max profit output during epidemic
theorem profitability_during_epidemic (x : ℕ) :
  (60 < x ∧ x ≤ max_annual_output) → 
  ((10 - (production_cost x)) * 60 - (x - 60) - assembly_line_cost > 0) ↔ x = 89 := sorry

end min_units_for_profitability_profitability_during_epidemic_l1587_158725


namespace at_least_12_lyamziks_rowed_l1587_158702

-- Define the lyamziks, their weights, and constraints
def LyamzikWeight1 : ℕ := 7
def LyamzikWeight2 : ℕ := 14
def LyamzikWeight3 : ℕ := 21
def LyamzikWeight4 : ℕ := 28
def totalLyamziks : ℕ := LyamzikWeight1 + LyamzikWeight2 + LyamzikWeight3 + LyamzikWeight4
def boatCapacity : ℕ := 10
def maxRowsPerLyamzik : ℕ := 2

-- Question to prove
theorem at_least_12_lyamziks_rowed : totalLyamziks ≥ 12 :=
  by sorry


end at_least_12_lyamziks_rowed_l1587_158702


namespace cos_A_condition_is_isosceles_triangle_tan_sum_l1587_158759

variable {A B C a b c : ℝ}

theorem cos_A_condition (h : (3 * b - c) * Real.cos A - a * Real.cos C = 0) :
  Real.cos A = 1 / 3 := sorry

theorem is_isosceles_triangle (ha : a = 2 * Real.sqrt 3)
  (hs : 1 / 2 * b * c * Real.sin A = 3 * Real.sqrt 2) :
  c = 3 ∧ b = 3 := sorry

theorem tan_sum (h_sin : Real.sin B * Real.sin C = 2 / 3)
  (h_cos : Real.cos A = 1 / 3) :
  Real.tan A + Real.tan B + Real.tan C = 4 * Real.sqrt 2 := sorry

end cos_A_condition_is_isosceles_triangle_tan_sum_l1587_158759


namespace probability_heads_odd_l1587_158752

theorem probability_heads_odd (n : ℕ) (p : ℚ) (Q : ℕ → ℚ) (h : p = 3/4) (h_rec : ∀ n, Q (n + 1) = p * (1 - Q n) + (1 - p) * Q n) :
  Q 40 = 1/2 * (1 - 1/4^40) := 
sorry

end probability_heads_odd_l1587_158752


namespace inequalities_consistent_l1587_158784

theorem inequalities_consistent (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1) ^ 2) (h3 : y * (y - 1) ≤ x ^ 2) : true := 
by 
  sorry

end inequalities_consistent_l1587_158784


namespace number_of_seeds_per_row_l1587_158701

-- Define the conditions as variables
def rows : ℕ := 6
def total_potatoes : ℕ := 54
def seeds_per_row : ℕ := 9

-- State the theorem
theorem number_of_seeds_per_row :
  total_potatoes / rows = seeds_per_row :=
by
-- We ignore the proof here, it will be provided later
sorry

end number_of_seeds_per_row_l1587_158701


namespace percentage_of_towns_correct_l1587_158778

def percentage_of_towns_with_fewer_than_50000_residents (p1 p2 p3 : ℝ) : ℝ :=
  p1 + p2

theorem percentage_of_towns_correct (p1 p2 p3 : ℝ) (h1 : p1 = 0.15) (h2 : p2 = 0.30) (h3 : p3 = 0.55) :
  percentage_of_towns_with_fewer_than_50000_residents p1 p2 p3 = 0.45 :=
by 
  sorry

end percentage_of_towns_correct_l1587_158778


namespace cyclic_quadrilateral_area_l1587_158741

variable (a b c d R : ℝ)
noncomputable def p : ℝ := (a + b + c + d) / 2
noncomputable def Brahmagupta_area : ℝ := Real.sqrt ((p a b c d - a) * (p a b c d - b) * (p a b c d - c) * (p a b c d - d))

theorem cyclic_quadrilateral_area :
  Brahmagupta_area a b c d = Real.sqrt ((a * b + c * d) * (a * d + b * c) * (a * c + b * d)) / (4 * R) := sorry

end cyclic_quadrilateral_area_l1587_158741


namespace roses_cut_l1587_158766

variable (initial final : ℕ) -- Declare variables for initial and final numbers of roses

-- Define the theorem stating the solution
theorem roses_cut (h1 : initial = 6) (h2 : final = 16) : final - initial = 10 :=
sorry -- Use sorry to skip the proof

end roses_cut_l1587_158766


namespace simon_legos_l1587_158742

theorem simon_legos (k b s : ℕ) 
  (h_kent : k = 40)
  (h_bruce : b = k + 20)
  (h_simon : s = b + b / 5) : 
  s = 72 := by
  -- sorry, proof not required.
  sorry

end simon_legos_l1587_158742


namespace evaluate_expr_l1587_158745

theorem evaluate_expr :
  (3 * Real.sqrt 7) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 11) = 
  -1 / 6 * (Real.sqrt 21 + Real.sqrt 35 - Real.sqrt 77) - 7 / 3 := by
  sorry

end evaluate_expr_l1587_158745


namespace unique_sum_of_two_cubes_lt_1000_l1587_158733

theorem unique_sum_of_two_cubes_lt_1000 
  : ∃ (sums : Finset ℕ), 
    (∀ x ∈ sums, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ x = a^3 + b^3) 
    ∧ sums.card = 40 
    ∧ ∀ x ∈ sums, x < 1000 := 
by sorry

end unique_sum_of_two_cubes_lt_1000_l1587_158733


namespace common_solution_l1587_158721

-- Define the conditions of the equations as hypotheses
variables (x y : ℝ)

-- First equation
def eq1 := x^2 + y^2 = 4

-- Second equation
def eq2 := x^2 = 4*y - 8

-- Proof statement: If there exists real numbers x and y such that both equations hold,
-- then y must be equal to 2.
theorem common_solution (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) : y = 2 :=
sorry

end common_solution_l1587_158721


namespace weight_difference_l1587_158772

noncomputable def W_A : ℝ := 78

variable (W_B W_C W_D W_E : ℝ)

axiom cond1 : (W_A + W_B + W_C) / 3 = 84
axiom cond2 : (W_A + W_B + W_C + W_D) / 4 = 80
axiom cond3 : (W_B + W_C + W_D + W_E) / 4 = 79

theorem weight_difference : W_E - W_D = 6 :=
by
  have h1 : W_A = 78 := rfl
  sorry

end weight_difference_l1587_158772


namespace perpendicular_line_l1587_158705

theorem perpendicular_line 
  (a b c : ℝ) 
  (p : ℝ × ℝ) 
  (h₁ : p = (-1, 3)) 
  (h₂ : a * (-1) + b * 3 + c = 0) 
  (h₃ : a * p.fst + b * p.snd + c = 0) 
  (hp : a = 1 ∧ b = -2 ∧ c = 3) : 
  ∃ a₁ b₁ c₁ : ℝ, 
  a₁ * (-1) + b₁ * 3 + c₁ = 0 ∧ a₁ = 2 ∧ b₁ = 1 ∧ c₁ = -1 := 
by 
  sorry

end perpendicular_line_l1587_158705


namespace points_for_level_completion_l1587_158710

-- Condition definitions
def enemies_defeated : ℕ := 6
def points_per_enemy : ℕ := 9
def total_points : ℕ := 62

-- Derived definitions (based on the problem steps):
def points_from_enemies : ℕ := enemies_defeated * points_per_enemy
def points_for_completing_level : ℕ := total_points - points_from_enemies

-- Theorem statement
theorem points_for_level_completion : points_for_completing_level = 8 := by
  sorry

end points_for_level_completion_l1587_158710


namespace product_of_four_consecutive_integers_l1587_158781

theorem product_of_four_consecutive_integers (n : ℤ) : ∃ k : ℤ, k^2 = (n-1) * n * (n+1) * (n+2) + 1 :=
by
  sorry

end product_of_four_consecutive_integers_l1587_158781


namespace person_age_in_1893_l1587_158716

theorem person_age_in_1893 
    (x y : ℕ)
    (h1 : 0 ≤ x ∧ x < 10)
    (h2 : 0 ≤ y ∧ y < 10)
    (h3 : 1 + 8 + x + y = 93 - 10 * x - y) : 
    1893 - (1800 + 10 * x + y) = 24 :=
by
  sorry

end person_age_in_1893_l1587_158716


namespace distance_to_larger_cross_section_l1587_158731

theorem distance_to_larger_cross_section
    (A B : ℝ)
    (a b : ℝ)
    (d : ℝ)
    (h : ℝ)
    (h_eq : h = 30):
  A = 300 * Real.sqrt 2 → 
  B = 675 * Real.sqrt 2 → 
  a = Real.sqrt (A / B) → 
  b = d / (1 - a) → 
  d = 10 → 
  b = h :=
by
  sorry

end distance_to_larger_cross_section_l1587_158731


namespace no_such_function_exists_l1587_158709

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ m n : ℕ, (m + f n)^2 ≥ 3 * (f m)^2 + n^2 :=
by 
  sorry

end no_such_function_exists_l1587_158709


namespace rectangle_area_l1587_158703

theorem rectangle_area (y : ℝ) (h : y > 0) 
    (h_area : ∃ (E F G H : ℝ × ℝ), 
        E = (0, 0) ∧ 
        F = (0, 5) ∧ 
        G = (y, 5) ∧ 
        H = (y, 0) ∧ 
        5 * y = 45) : 
    y = 9 := 
by
    sorry

end rectangle_area_l1587_158703


namespace people_sharing_cookies_l1587_158756

theorem people_sharing_cookies (total_cookies : ℕ) (cookies_per_person : ℕ) (people : ℕ) 
  (h1 : total_cookies = 24) (h2 : cookies_per_person = 4) (h3 : total_cookies = cookies_per_person * people) : 
  people = 6 :=
by
  sorry

end people_sharing_cookies_l1587_158756


namespace smallest_term_at_n_is_4_or_5_l1587_158786

def a_n (n : ℕ) : ℝ :=
  n^2 - 9 * n - 100

theorem smallest_term_at_n_is_4_or_5 :
  ∃ n, n = 4 ∨ n = 5 ∧ a_n n = min (a_n 4) (a_n 5) :=
by
  sorry

end smallest_term_at_n_is_4_or_5_l1587_158786


namespace number_of_papers_l1587_158743

-- Define the conditions
def folded_pieces (folds : ℕ) : ℕ := 2 ^ folds
def notes_per_day : ℕ := 10
def days_per_notepad : ℕ := 4
def notes_per_notepad : ℕ := notes_per_day * days_per_notepad
def notes_per_paper (folds : ℕ) : ℕ := folded_pieces folds

-- Lean statement for the proof problem
theorem number_of_papers (folds : ℕ) (h_folds : folds = 3) :
  (notes_per_notepad / notes_per_paper folds) = 5 :=
by
  rw [h_folds]
  simp [notes_per_notepad, notes_per_paper, folded_pieces]
  sorry

end number_of_papers_l1587_158743


namespace bank_teller_bills_l1587_158746

theorem bank_teller_bills (x y : ℕ) (h1 : x + y = 54) (h2 : 5 * x + 20 * y = 780) : x = 20 :=
by
  sorry

end bank_teller_bills_l1587_158746


namespace find_center_and_radius_sum_l1587_158769

-- Define the given equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 16 * x + y^2 + 10 * y = -75

-- Define the center of the circle
def center (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → (x = a) ∧ (y = b)

-- Define the radius of the circle
def radius (r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → (x^2 - 16 * x + y^2 + 10 * y = r^2)

-- Main theorem to prove a + b + r = 3 + sqrt 14
theorem find_center_and_radius_sum (a b r : ℝ) (h_cen : center a b) (h_rad : radius r) : 
  a + b + r = 3 + Real.sqrt 14 :=
  sorry

end find_center_and_radius_sum_l1587_158769


namespace billiard_ball_radius_unique_l1587_158751

noncomputable def radius_of_billiard_balls (r : ℝ) : Prop :=
  let side_length := 292
  let lhs := (8 + 2 * Real.sqrt 3) * r
  lhs = side_length

theorem billiard_ball_radius_unique (r : ℝ) : radius_of_billiard_balls r → r = (146 / 13) * (4 - Real.sqrt 3 / 3) :=
by
  intro h1
  sorry

end billiard_ball_radius_unique_l1587_158751


namespace value_of_a_l1587_158712

theorem value_of_a (a : ℝ) : 
  (∀ (x : ℝ), (x < -4 ∨ x > 5) → x^2 + a * x + 20 > 0) → a = -1 :=
by
  sorry

end value_of_a_l1587_158712


namespace total_age_10_years_from_now_is_75_l1587_158738

-- Define the conditions
def eldest_age_now : ℕ := 20
def age_difference : ℕ := 5

-- Define the ages of the siblings 10 years from now
def eldest_age_10_years_from_now : ℕ := eldest_age_now + 10
def second_age_10_years_from_now : ℕ := (eldest_age_now - age_difference) + 10
def third_age_10_years_from_now : ℕ := (eldest_age_now - 2 * age_difference) + 10

-- Define the total age of the siblings 10 years from now
def total_age_10_years_from_now : ℕ := 
  eldest_age_10_years_from_now + 
  second_age_10_years_from_now + 
  third_age_10_years_from_now

-- The theorem statement
theorem total_age_10_years_from_now_is_75 : total_age_10_years_from_now = 75 := 
  by sorry

end total_age_10_years_from_now_is_75_l1587_158738


namespace project_completion_time_l1587_158708

theorem project_completion_time (m n : ℝ) (hm : m > 0) (hn : n > 0):
  (1 / (1 / m + 1 / n)) = (m * n) / (m + n) :=
by
  sorry

end project_completion_time_l1587_158708


namespace find_remainder_l1587_158750

theorem find_remainder (dividend divisor quotient : ℕ) (h1 : dividend = 686) (h2 : divisor = 36) (h3 : quotient = 19) :
  ∃ remainder, dividend = (divisor * quotient) + remainder ∧ remainder = 2 :=
by
  sorry

end find_remainder_l1587_158750


namespace interval_of_expression_l1587_158720

theorem interval_of_expression (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
  1 < (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) ∧ 
  (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) < 2 :=
by sorry

end interval_of_expression_l1587_158720


namespace problem_statement_l1587_158754

noncomputable def A := 5 * Real.pi / 12
noncomputable def B := Real.pi / 3
noncomputable def C := Real.pi / 4
noncomputable def b := Real.sqrt 3
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3

theorem problem_statement :
  (Set.Icc (-2 : ℝ) 2 = Set.image f Set.univ) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∃ (area : ℝ), area = (3 + Real.sqrt 3) / 4)
:= sorry

end problem_statement_l1587_158754


namespace fractions_order_l1587_158770

theorem fractions_order :
  (21 / 17) < (18 / 13) ∧ (18 / 13) < (16 / 11) := by
  sorry

end fractions_order_l1587_158770


namespace average_price_of_goat_l1587_158715

theorem average_price_of_goat (total_cost_goats_hens : ℕ) (num_goats num_hens : ℕ) (avg_price_hen : ℕ)
  (h1 : total_cost_goats_hens = 2500) (h2 : num_hens = 10) (h3 : avg_price_hen = 50) (h4 : num_goats = 5) :
  (total_cost_goats_hens - num_hens * avg_price_hen) / num_goats = 400 :=
sorry

end average_price_of_goat_l1587_158715


namespace negation_of_existential_l1587_158765

theorem negation_of_existential :
  (∀ x : ℝ, x^2 + x - 1 ≤ 0) ↔ ¬ (∃ x : ℝ, x^2 + x - 1 > 0) :=
by sorry

end negation_of_existential_l1587_158765


namespace express_b_c_range_a_not_monotonic_l1587_158797

noncomputable def f (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) * Real.exp (-x)
noncomputable def f' (a b c x : ℝ) : ℝ := 
    (a * x^2 + b * x + c) * (-Real.exp (-x)) + (2 * a * x + b) * Real.exp (-x)

theorem express_b_c (a : ℝ) : 
    (∃ b c : ℝ, f a b c 0 = 2 * a ∧ f' a b c 0 = Real.pi / 4) → 
    (∃ b c : ℝ, b = 1 + 2 * a ∧ c = 2 * a) := 
sorry

noncomputable def g (a x : ℝ) : ℝ := -a * x^2 - x + 1

theorem range_a_not_monotonic (a : ℝ) : 
    (¬ (∀ x y : ℝ, x ∈ Set.Ici (1 / 2) → y ∈ Set.Ici (1 / 2) → x < y → g a x ≤ g a y)) → 
    (-1 / 4 < a ∧ a < 2) := 
sorry

end express_b_c_range_a_not_monotonic_l1587_158797


namespace add_base3_numbers_l1587_158727

theorem add_base3_numbers : 
  (2 + 1 * 3) + (0 + 2 * 3 + 1 * 3^2) + 
  (1 + 2 * 3 + 0 * 3^2 + 2 * 3^3) + (2 + 0 * 3 + 1 * 3^2 + 2 * 3^3)
  = 2 + 2 * 3 + 2 * 3^2 + 2 * 3^3 := 
by sorry

end add_base3_numbers_l1587_158727


namespace chord_bisected_vertically_by_line_l1587_158737

theorem chord_bisected_vertically_by_line (p : ℝ) (h : p > 0) (l : ℝ → ℝ) (focus : ℝ × ℝ) 
  (h_focus: focus = (p / 2, 0)) (h_line: ∀ x, l x ≠ 0) :
  ¬ ∃ (A B : ℝ × ℝ), 
     A.1 ≠ B.1 ∧
     A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1 ∧ 
     (A.1 + B.1) / 2 = focus.1 ∧ 
     l ((A.1 + B.1) / 2) = focus.2 :=
sorry

end chord_bisected_vertically_by_line_l1587_158737


namespace repeating_decimal_fraction_l1587_158744

theorem repeating_decimal_fraction :
  let a := (9 : ℚ) / 25
  let r := (1 : ℚ) / 100
  (a / (1 - r)) = (4 : ℚ) / 11 :=
by
  sorry

end repeating_decimal_fraction_l1587_158744


namespace smallest_number_exists_l1587_158714

theorem smallest_number_exists (x : ℤ) :
  (x + 3) % 18 = 0 ∧ 
  (x + 3) % 70 = 0 ∧ 
  (x + 3) % 100 = 0 ∧ 
  (x + 3) % 84 = 0 → 
  x = 6297 :=
by
  sorry

end smallest_number_exists_l1587_158714


namespace determine_real_coins_l1587_158747

def has_fake_coin (coins : List ℝ) : Prop :=
  ∃ fake_coin ∈ coins, (∀ coin ∈ coins, coin ≠ fake_coin)

theorem determine_real_coins (coins : List ℝ) (h : has_fake_coin coins) (h_length : coins.length = 101) :
  ∃ real_coins : List ℝ, ∀ r ∈ real_coins, r ∈ coins ∧ real_coins.length ≥ 50 :=
by
  sorry

end determine_real_coins_l1587_158747


namespace cannot_have_1970_minus_signs_in_grid_l1587_158711

theorem cannot_have_1970_minus_signs_in_grid :
  ∀ (k l : ℕ), k ≤ 100 → l ≤ 100 → (k+l)*50 - k*l ≠ 985 :=
by
  intros k l hk hl
  sorry

end cannot_have_1970_minus_signs_in_grid_l1587_158711


namespace determine_k_l1587_158704

theorem determine_k (k : ℝ) (h1 : ∃ x y : ℝ, y = 4 * x + 3 ∧ y = -2 * x - 25 ∧ y = 3 * x + k) : k = -5 / 3 := by
  sorry

end determine_k_l1587_158704


namespace findAnalyticalExpression_l1587_158794

-- Defining the point A as a structure with x and y coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Defining a line as having a slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Condition: Line 1 is parallel to y = 2x - 3
def line1 : Line := {slope := 2, intercept := -3}

-- Condition: Line 2 passes through point A
def point_A : Point := {x := -2, y := -1}

-- The theorem statement:
theorem findAnalyticalExpression : 
  ∃ b : ℝ, (∀ x : ℝ, (point_A.y = line1.slope * point_A.x + b) → b = 3) ∧ 
            ∀ x : ℝ, (line1.slope * x + b = 2 * x + 3) :=
sorry

end findAnalyticalExpression_l1587_158794


namespace find_rate_of_interest_l1587_158796

variable (P : ℝ) (R : ℝ) (T : ℕ := 2)

-- Condition for Simple Interest (SI = Rs. 660 for 2 years)
def simple_interest :=
  P * R * ↑T / 100 = 660

-- Condition for Compound Interest (CI = Rs. 696.30 for 2 years)
def compound_interest :=
  P * ((1 + R / 100) ^ T - 1) = 696.30

-- We need to prove that R = 11
theorem find_rate_of_interest (P : ℝ) (h1 : simple_interest P R) (h2 : compound_interest P R) : 
  R = 11 := by
  sorry

end find_rate_of_interest_l1587_158796


namespace find_f_five_thirds_l1587_158767

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = - f x
def functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (-x)
def specific_value (f : ℝ → ℝ) := f (-1/3) = 1/3

theorem find_f_five_thirds
  (hf_odd : odd_function f)
  (hf_fe : functional_equation f)
  (hf_value : specific_value f) :
  f (5 / 3) = 1 / 3 := by
  sorry

end find_f_five_thirds_l1587_158767


namespace fish_eaten_by_new_fish_l1587_158719

def initial_original_fish := 14
def added_fish := 2
def exchange_new_fish := 3
def total_fish_now := 11

theorem fish_eaten_by_new_fish : initial_original_fish - (total_fish_now - exchange_new_fish) = 6 := by
  -- This is where the proof would go
  sorry

end fish_eaten_by_new_fish_l1587_158719


namespace circle_equation_through_points_l1587_158730

theorem circle_equation_through_points (A B: ℝ × ℝ) (C : ℝ × ℝ)
  (hA : A = (1, -1)) (hB : B = (-1, 1)) (hC : C.1 + C.2 = 2)
  (hAC : dist A C = dist B C) :
  (x - C.1) ^ 2 + (y - C.2) ^ 2 = 4 :=
by
  sorry

end circle_equation_through_points_l1587_158730


namespace solve_equation_l1587_158734

theorem solve_equation (x y z : ℕ) : (3 ^ x + 5 ^ y + 14 = z!) ↔ ((x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) :=
by
  sorry

end solve_equation_l1587_158734


namespace find_current_l1587_158732

theorem find_current (R Q t : ℝ) (hR : R = 8) (hQ : Q = 72) (ht : t = 2) :
  ∃ I : ℝ, Q = I^2 * R * t ∧ I = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end find_current_l1587_158732


namespace total_money_shared_l1587_158773

-- Define the variables and conditions
def joshua_share : ℕ := 30
def justin_share : ℕ := joshua_share / 3
def total_shared_money : ℕ := joshua_share + justin_share

-- State the theorem to prove
theorem total_money_shared : total_shared_money = 40 :=
by
  -- proof will go here
  sorry

end total_money_shared_l1587_158773


namespace intersection_of_function_and_inverse_l1587_158761

theorem intersection_of_function_and_inverse (b a : Int) 
  (h₁ : a = 2 * (-4) + b) 
  (h₂ : a = (-4 - b) / 2) 
  : a = -4 :=
by
  sorry

end intersection_of_function_and_inverse_l1587_158761


namespace four_digit_sum_l1587_158771

theorem four_digit_sum (A B : ℕ) (hA : 1000 ≤ A ∧ A < 10000) (hB : 1000 ≤ B ∧ B < 10000) (h : A * B = 16^5 + 2^10) : A + B = 2049 := 
by sorry

end four_digit_sum_l1587_158771


namespace expression_evaluation_l1587_158764

theorem expression_evaluation (a b c : ℤ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 3) :
  (2 * a - (b - 2 * c)) - ((2 * a - b) - 2 * c) + 3 * (a - c) = 27 :=
by
  have ha : a = 8 := h₁
  have hb : b = 10 := h₂
  have hc : c = 3 := h₃
  rw [ha, hb, hc]
  sorry

end expression_evaluation_l1587_158764


namespace angle_sum_proof_l1587_158789

theorem angle_sum_proof (A B C x y : ℝ) 
  (hA : A = 35) 
  (hB : B = 65) 
  (hC : C = 40) 
  (hx : x = 130 - C)
  (hy : y = 90 - A) :
  x + y = 140 := by
  sorry

end angle_sum_proof_l1587_158789


namespace P_projection_matrix_P_not_invertible_l1587_158717

noncomputable def v : ℝ × ℝ := (4, -1)

noncomputable def norm_v : ℝ := Real.sqrt (4^2 + (-1)^2)

noncomputable def u : ℝ × ℝ := (4 / norm_v, -1 / norm_v)

noncomputable def P : ℝ × ℝ × ℝ × ℝ :=
((4 * 4) / norm_v^2, (4 * -1) / norm_v^2, 
 (-1 * 4) / norm_v^2, (-1 * -1) / norm_v^2)

theorem P_projection_matrix :
  P = (16 / 17, -4 / 17, -4 / 17, 1 / 17) := by
  sorry

theorem P_not_invertible :
  ¬(∃ Q : ℝ × ℝ × ℝ × ℝ, P = Q) := by
  sorry

end P_projection_matrix_P_not_invertible_l1587_158717


namespace smallest_number_is_33_l1587_158749

theorem smallest_number_is_33 (x : ℝ) 
  (h1 : 2 * x = third)
  (h2 : 4 * x = second)
  (h3 : (x + 2 * x + 4 * x) / 3 = 77) : 
  x = 33 := 
by 
  sorry

end smallest_number_is_33_l1587_158749


namespace sets_equal_l1587_158739

def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }
def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem sets_equal : E = F :=
  sorry

end sets_equal_l1587_158739


namespace calc_h_one_l1587_158735

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 6
noncomputable def g (x : ℝ) : ℝ := Real.exp (f x) - 3
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- the final theorem that we are proving
theorem calc_h_one : h 1 = 3 * Real.exp 26 - 14 * Real.exp 13 + 21 := by
  sorry

end calc_h_one_l1587_158735


namespace average_inside_time_l1587_158774

theorem average_inside_time (j_awake_frac : ℚ) (j_inside_awake_frac : ℚ) (r_awake_frac : ℚ) (r_inside_day_frac : ℚ) :
  j_awake_frac = 2 / 3 →
  j_inside_awake_frac = 1 / 2 →
  r_awake_frac = 3 / 4 →
  r_inside_day_frac = 2 / 3 →
  (24 * j_awake_frac * j_inside_awake_frac + 24 * r_awake_frac * r_inside_day_frac) / 2 = 10 := 
by
    sorry

end average_inside_time_l1587_158774


namespace geom_seq_root_product_l1587_158792

theorem geom_seq_root_product
  (a : ℕ → ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * a 1)
  (h_root1 : 3 * (a 1)^2 + 7 * a 1 - 9 = 0)
  (h_root10 : 3 * (a 10)^2 + 7 * a 10 - 9 = 0) :
  a 4 * a 7 = -3 := 
by
  sorry

end geom_seq_root_product_l1587_158792


namespace problem_statement_l1587_158793

-- Let's define the conditions
def num_blue_balls : ℕ := 8
def num_green_balls : ℕ := 7
def total_balls : ℕ := num_blue_balls + num_green_balls

-- Function to calculate combinations (binomial coefficients)
def combination (n r : ℕ) : ℕ :=
  n.choose r

-- Specific combinations for this problem
def blue_ball_ways : ℕ := combination num_blue_balls 3
def green_ball_ways : ℕ := combination num_green_balls 2
def total_ways : ℕ := combination total_balls 5

-- The number of favorable outcomes
def favorable_outcomes : ℕ := blue_ball_ways * green_ball_ways

-- The probability
def probability : ℚ := favorable_outcomes / total_ways

-- The theorem stating our result
theorem problem_statement : probability = 1176/3003 := by
  sorry

end problem_statement_l1587_158793


namespace smallest_eraser_packs_needed_l1587_158760

def yazmin_packs_condition (pencils_packs erasers_packs pencils_per_pack erasers_per_pack : ℕ) : Prop :=
  pencils_packs * pencils_per_pack = erasers_packs * erasers_per_pack

theorem smallest_eraser_packs_needed (pencils_per_pack erasers_per_pack : ℕ) (h_pencils_5 : pencils_per_pack = 5) (h_erasers_7 : erasers_per_pack = 7) : ∃ erasers_packs, yazmin_packs_condition 7 erasers_packs pencils_per_pack erasers_per_pack ∧ erasers_packs = 5 :=
by
  sorry

end smallest_eraser_packs_needed_l1587_158760


namespace negation_of_proposition_p_l1587_158757

theorem negation_of_proposition_p :
  (¬(∃ x : ℝ, 0 < x ∧ Real.log x > x - 1)) ↔ (∀ x : ℝ, 0 < x → Real.log x ≤ x - 1) :=
by
  sorry

end negation_of_proposition_p_l1587_158757


namespace sum_of_first_60_digits_l1587_158700

noncomputable def decimal_expansion_period : List ℕ := [0, 0, 0, 8, 1, 0, 3, 7, 2, 7, 7, 1, 4, 7, 4, 8, 7, 8, 4, 4, 4, 0, 8, 4, 2, 7, 8, 7, 6, 8]

def sum_of_list (l : List ℕ) : ℕ := l.foldl (· + ·) 0

theorem sum_of_first_60_digits : sum_of_list (decimal_expansion_period ++ decimal_expansion_period) = 282 := 
by
  simp [decimal_expansion_period, sum_of_list]
  sorry

end sum_of_first_60_digits_l1587_158700


namespace sum_series_eq_one_l1587_158779

noncomputable def series : ℝ := ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1)

theorem sum_series_eq_one : series = 1 := 
by sorry

end sum_series_eq_one_l1587_158779


namespace probability_at_most_one_correct_in_two_rounds_l1587_158775

theorem probability_at_most_one_correct_in_two_rounds :
  let pA := 3 / 5
  let pB := 2 / 3
  let pA_incorrect := 1 - pA
  let pB_incorrect := 1 - pB
  let p_0_correct := pA_incorrect * pA_incorrect * pB_incorrect * pB_incorrect
  let p_1_correct_A1 := pA * pA_incorrect * pB_incorrect * pB_incorrect
  let p_1_correct_A2 := pA_incorrect * pA * pB_incorrect * pB_incorrect
  let p_1_correct_B1 := pA_incorrect * pA_incorrect * pB * pB_incorrect
  let p_1_correct_B2 := pA_incorrect * pA_incorrect * pB_incorrect * pB
  let p_at_most_one := p_0_correct + p_1_correct_A1 + p_1_correct_A2 + 
      p_1_correct_B1 + p_1_correct_B2
  p_at_most_one = 32 / 225 := 
  sorry

end probability_at_most_one_correct_in_two_rounds_l1587_158775


namespace rectangle_to_total_height_ratio_l1587_158748

theorem rectangle_to_total_height_ratio 
  (total_area : ℕ)
  (width : ℕ)
  (area_per_side : ℕ)
  (height : ℕ)
  (triangle_base : ℕ)
  (triangle_area : ℕ)
  (rect_area : ℕ)
  (total_height : ℕ)
  (ratio : ℚ)
  (h_eqn : 3 * height = area_per_side)
  (h_value : height = total_area / (2 * 3))
  (total_height_eqn : total_height = 2 * height)
  (ratio_eqn : ratio = height / total_height) :
  total_area = 12 → width = 3 → area_per_side = 6 → triangle_base = 3 →
  triangle_area = triangle_base * height / 2 → rect_area = width * height →
  rect_area = area_per_side → ratio = 1 / 2 :=
by
  intros
  sorry

end rectangle_to_total_height_ratio_l1587_158748


namespace People_Distribution_l1587_158726

theorem People_Distribution 
  (total_people : ℕ) 
  (total_buses : ℕ) 
  (equal_distribution : ℕ) 
  (h1 : total_people = 219) 
  (h2 : total_buses = 3) 
  (h3 : equal_distribution = total_people / total_buses) : 
  equal_distribution = 73 :=
by 
  intros 
  sorry

end People_Distribution_l1587_158726


namespace prob_union_of_mutually_exclusive_l1587_158799

-- Let's denote P as a probability function
variable {Ω : Type} (P : Set Ω → ℝ)

-- Define the mutually exclusive condition
def mutually_exclusive (A B : Set Ω) : Prop :=
  (A ∩ B) = ∅

-- State the theorem that we want to prove
theorem prob_union_of_mutually_exclusive (A B : Set Ω) 
  (h : mutually_exclusive A B) : P (A ∪ B) = P A + P B :=
sorry

end prob_union_of_mutually_exclusive_l1587_158799


namespace solve_inequalities_solve_linear_system_l1587_158762

-- System of Inequalities
theorem solve_inequalities (x : ℝ) (h1 : x + 2 > 1) (h2 : 2 * x < x + 3) : -1 < x ∧ x < 3 :=
by
  sorry

-- System of Linear Equations
theorem solve_linear_system (x y : ℝ) (h1 : 3 * x + 2 * y = 12) (h2 : 2 * x - y = 1) : x = 2 ∧ y = 3 :=
by
  sorry

end solve_inequalities_solve_linear_system_l1587_158762


namespace factor_of_land_increase_l1587_158736

-- Definitions of the conditions in the problem:
def initial_money_given_by_blake : ℝ := 20000
def money_received_by_blake_after_sale : ℝ := 30000

-- The main theorem to prove
theorem factor_of_land_increase (F : ℝ) : 
  (1/2) * (initial_money_given_by_blake * F) = money_received_by_blake_after_sale → 
  F = 3 :=
by sorry

end factor_of_land_increase_l1587_158736


namespace num_packs_blue_tshirts_l1587_158707

def num_white_tshirts_per_pack : ℕ := 6
def num_packs_white_tshirts : ℕ := 5
def num_blue_tshirts_per_pack : ℕ := 9
def total_num_tshirts : ℕ := 57

theorem num_packs_blue_tshirts : (total_num_tshirts - num_white_tshirts_per_pack * num_packs_white_tshirts) / num_blue_tshirts_per_pack = 3 := by
  sorry

end num_packs_blue_tshirts_l1587_158707


namespace train_stop_times_l1587_158763

theorem train_stop_times :
  ∀ (speed_without_stops_A speed_with_stops_A speed_without_stops_B speed_with_stops_B : ℕ),
  speed_without_stops_A = 45 →
  speed_with_stops_A = 30 →
  speed_without_stops_B = 60 →
  speed_with_stops_B = 40 →
  (60 * (speed_without_stops_A - speed_with_stops_A) / speed_without_stops_A = 20) ∧
  (60 * (speed_without_stops_B - speed_with_stops_B) / speed_without_stops_B = 20) :=
by
  intros
  sorry

end train_stop_times_l1587_158763


namespace negation_of_proposition_l1587_158791

theorem negation_of_proposition (x : ℝ) : ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_of_proposition_l1587_158791


namespace reduced_price_of_oil_is_40_l1587_158783

variables 
  (P R : ℝ) 
  (hP : 0 < P)
  (hR : R = 0.75 * P)
  (hw : 800 / (0.75 * P) = 800 / P + 5)

theorem reduced_price_of_oil_is_40 : R = 40 :=
sorry

end reduced_price_of_oil_is_40_l1587_158783
