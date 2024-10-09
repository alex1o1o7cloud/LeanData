import Mathlib

namespace bridge_length_l1991_199160

theorem bridge_length
  (train_length : ℝ)
  (train_speed_km_hr : ℝ)
  (crossing_time_sec : ℝ)
  (train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600)
  (total_distance : ℝ := train_speed_m_s * crossing_time_sec)
  (bridge_length : ℝ := total_distance - train_length)
  (train_length_val : train_length = 110)
  (train_speed_km_hr_val : train_speed_km_hr = 36)
  (crossing_time_sec_val : crossing_time_sec = 24.198064154867613) :
  bridge_length = 131.98064154867613 :=
by
  sorry

end bridge_length_l1991_199160


namespace sum_of_obtuse_angles_l1991_199143

theorem sum_of_obtuse_angles (A B : ℝ) (hA1 : A > π / 2) (hA2 : A < π)
  (hB1 : B > π / 2) (hB2 : B < π)
  (hSinA : Real.sin A = Real.sqrt 5 / 5)
  (hSinB : Real.sin B = Real.sqrt 10 / 10) :
  A + B = 7 * π / 4 := 
sorry

end sum_of_obtuse_angles_l1991_199143


namespace problem_part_1_problem_part_2_problem_part_3_l1991_199165

open Set

universe u

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := univ

theorem problem_part_1 : A ∪ B = {x | 1 < x ∧ x ≤ 8} :=
sorry

theorem problem_part_2 : (U \ A) ∩ B = {x | 1 < x ∧ x < 2} :=
sorry

theorem problem_part_3 (a : ℝ) (h : (A ∩ C a) ≠ ∅) : a < 8 :=
sorry

end problem_part_1_problem_part_2_problem_part_3_l1991_199165


namespace intersection_A_B_l1991_199146

def A : Set ℝ := {x | x ≤ 2*x + 1 ∧ 2*x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : 
  A ∩ B = {x | 0 < x ∧ x ≤ 2} :=
  sorry

end intersection_A_B_l1991_199146


namespace part1_part2_l1991_199117
noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

theorem part1 : {x : ℝ | f x ≥ 3} = {x | x ≤ 0} ∪ {x | x ≥ 3} :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x ≤ -a^2 + a + 7) ↔ -2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end part1_part2_l1991_199117


namespace angles_terminal_side_equiv_l1991_199161

theorem angles_terminal_side_equiv (k : ℤ) : (2 * k * Real.pi + Real.pi) % (2 * Real.pi) = (4 * k * Real.pi + Real.pi) % (2 * Real.pi) ∨ (2 * k * Real.pi + Real.pi) % (2 * Real.pi) = (4 * k * Real.pi - Real.pi) % (2 * Real.pi) :=
sorry

end angles_terminal_side_equiv_l1991_199161


namespace max_m_value_l1991_199135

theorem max_m_value (a : ℚ) (m : ℚ) : (∀ x : ℤ, 0 < x ∧ x ≤ 50 → ¬ ∃ y : ℤ, y = m * x + 3) ∧ (1 / 2 < m) ∧ (m < a) → a = 26 / 51 :=
by sorry

end max_m_value_l1991_199135


namespace simplify_power_of_product_l1991_199170

theorem simplify_power_of_product (x y : ℝ) : (3 * x^2 * y^3)^2 = 9 * x^4 * y^6 :=
by
  -- hint: begin proof here
  sorry

end simplify_power_of_product_l1991_199170


namespace intersection_M_N_l1991_199108

def M : Set ℕ := {1, 2, 4, 8}
def N : Set ℕ := {x | ∃ k : ℕ, x = 2 * k}

theorem intersection_M_N :
  M ∩ N = {2, 4, 8} :=
by sorry

end intersection_M_N_l1991_199108


namespace percentage_shaded_is_18_75_l1991_199178

-- conditions
def total_squares: ℕ := 16
def shaded_squares: ℕ := 3

-- claim to prove
theorem percentage_shaded_is_18_75 :
  ((shaded_squares : ℝ) / total_squares) * 100 = 18.75 := 
by
  sorry

end percentage_shaded_is_18_75_l1991_199178


namespace dan_has_13_limes_l1991_199169

theorem dan_has_13_limes (picked_limes : ℕ) (given_limes : ℕ) (h1 : picked_limes = 9) (h2 : given_limes = 4) : 
  picked_limes + given_limes = 13 := 
by
  sorry

end dan_has_13_limes_l1991_199169


namespace molecular_weight_calculated_l1991_199107

def atomic_weight_Ba : ℚ := 137.33
def atomic_weight_O  : ℚ := 16.00
def atomic_weight_H  : ℚ := 1.01

def molecular_weight_compound : ℚ :=
  (1 * atomic_weight_Ba) + (2 * atomic_weight_O) + (2 * atomic_weight_H)

theorem molecular_weight_calculated :
  molecular_weight_compound = 171.35 :=
by {
  sorry
}

end molecular_weight_calculated_l1991_199107


namespace unique_nonzero_solution_l1991_199179

theorem unique_nonzero_solution (x : ℝ) (h : x ≠ 0) : (3 * x)^3 = (9 * x)^2 → x = 3 :=
by
  sorry

end unique_nonzero_solution_l1991_199179


namespace new_ratio_of_milk_to_water_l1991_199110

theorem new_ratio_of_milk_to_water
  (total_volume : ℕ) (initial_ratio_milk : ℕ) (initial_ratio_water : ℕ) (added_water : ℕ)
  (h_total_volume : total_volume = 45)
  (h_initial_ratio : initial_ratio_milk = 4 ∧ initial_ratio_water = 1)
  (h_added_water : added_water = 11) :
  let initial_milk := (initial_ratio_milk * total_volume) / (initial_ratio_milk + initial_ratio_water)
  let initial_water := (initial_ratio_water * total_volume) / (initial_ratio_milk + initial_ratio_water)
  let new_water := initial_water + added_water
  let gcd := Nat.gcd initial_milk new_water
  (initial_milk / gcd : ℕ) = 9 ∧ (new_water / gcd : ℕ) = 5 :=
by
  sorry

end new_ratio_of_milk_to_water_l1991_199110


namespace find_positive_integers_l1991_199139

theorem find_positive_integers (a b c : ℕ) (ha : a ≥ b) (hb : b ≥ c) :
  (∃ n₁ : ℕ, a^2 + 3 * b = n₁^2) ∧ 
  (∃ n₂ : ℕ, b^2 + 3 * c = n₂^2) ∧ 
  (∃ n₃ : ℕ, c^2 + 3 * a = n₃^2) →
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 37 ∧ b = 25 ∧ c = 17) :=
by
  sorry

end find_positive_integers_l1991_199139


namespace find_coords_of_P_l1991_199126

-- Definitions from the conditions
def line_eq (x y : ℝ) : Prop := x - y - 7 = 0
def is_midpoint (P Q M : ℝ × ℝ) : Prop := 
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Coordinates given in the problem
def P : ℝ × ℝ := (-2, 1)

-- The proof goal
theorem find_coords_of_P : ∃ Q : ℝ × ℝ,
  is_midpoint P Q (1, -1) ∧ 
  line_eq Q.1 Q.2 :=
sorry

end find_coords_of_P_l1991_199126


namespace solve_fraction_equation_l1991_199181

theorem solve_fraction_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_fraction_equation_l1991_199181


namespace min_distance_to_line_value_of_AB_l1991_199198

noncomputable def point_B : ℝ × ℝ := (1, 1)
noncomputable def point_A : ℝ × ℝ := (4 * Real.sqrt 2, Real.pi / 4)

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def polar_line_l (a : ℝ) (θ : ℝ) : ℝ :=
  a * Real.cos (θ - Real.pi / 4)

noncomputable def line_l1 (m : ℝ) (x y : ℝ) : Prop :=
  x + y + m = 0

theorem min_distance_to_line {θ : ℝ} (a : ℝ) :
  polar_line_l a θ = 4 * Real.sqrt 2 → 
  ∃ d, d = (8 * Real.sqrt 2 - Real.sqrt 14) / 2 :=
by
  sorry

theorem value_of_AB :
  ∃ AB, AB = 12 * Real.sqrt 2 / 7 :=
by
  sorry

end min_distance_to_line_value_of_AB_l1991_199198


namespace men_absent_l1991_199125

theorem men_absent (original_men absent_men remaining_men : ℕ) (total_work : ℕ) 
  (h1 : original_men = 15) (h2 : total_work = original_men * 40) (h3 : 60 * remaining_men = total_work) : 
  remaining_men = original_men - absent_men → absent_men = 5 := 
by
  sorry

end men_absent_l1991_199125


namespace simultaneous_eq_solution_l1991_199127

theorem simultaneous_eq_solution (n : ℝ) (hn : n ≠ 1 / 2) : 
  ∃ (x y : ℝ), (y = (3 * n + 1) * x + 2) ∧ (y = (5 * n - 2) * x + 5) := 
sorry

end simultaneous_eq_solution_l1991_199127


namespace compounding_frequency_l1991_199124

variable (i : ℝ) (EAR : ℝ)

/-- Given the nominal annual rate (i = 6%) and the effective annual rate (EAR = 6.09%), 
    prove that the frequency of payment (n) is 4. -/
theorem compounding_frequency (h1 : i = 0.06) (h2 : EAR = 0.0609) : 
  ∃ n : ℕ, (1 + i / n)^n - 1 = EAR ∧ n = 4 := sorry

end compounding_frequency_l1991_199124


namespace profit_ratio_l1991_199100

variables (P_s : ℝ)

theorem profit_ratio (h1 : 21 * (7 / 3) + 3 * P_s = 175) : P_s / 21 = 2 :=
by
  sorry

end profit_ratio_l1991_199100


namespace complex_number_equality_l1991_199152

def is_imaginary_unit (i : ℂ) : Prop :=
  i^2 = -1

theorem complex_number_equality (a b : ℝ) (i : ℂ) (h1 : is_imaginary_unit i) (h2 : (a + 4 * i) * i = b + i) : a + b = -3 :=
sorry

end complex_number_equality_l1991_199152


namespace number_of_outfits_l1991_199112

-- Define the number of shirts, pants, and jacket options.
def shirts : Nat := 8
def pants : Nat := 5
def jackets : Nat := 3

-- The theorem statement for the total number of outfits.
theorem number_of_outfits : shirts * pants * jackets = 120 := 
by
  sorry

end number_of_outfits_l1991_199112


namespace christine_aquafaba_needed_l1991_199129

-- Define the number of tablespoons per egg white
def tablespoons_per_egg_white : ℕ := 2

-- Define the number of egg whites per cake
def egg_whites_per_cake : ℕ := 8

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Express the total amount of aquafaba needed
def aquafaba_needed : ℕ :=
  tablespoons_per_egg_white * egg_whites_per_cake * number_of_cakes

-- Statement asserting the amount of aquafaba needed is 32
theorem christine_aquafaba_needed : aquafaba_needed = 32 := by
  sorry

end christine_aquafaba_needed_l1991_199129


namespace find_f_correct_l1991_199120

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_con1 : ∀ x : ℝ, 2 * f x + f (-x) = 2 * x

theorem find_f_correct : ∀ x : ℝ, f x = 2 * x :=
by
  sorry

end find_f_correct_l1991_199120


namespace determine_d_l1991_199123

theorem determine_d (d c : ℕ) (hlcm : Nat.lcm 76 d = 456) (hhcf : Nat.gcd 76 d = c) : d = 24 :=
by
  sorry

end determine_d_l1991_199123


namespace truck_distance_on_7_liters_l1991_199130

-- Define the conditions
def truck_300_km_per_5_liters := 300
def liters_5 := 5
def liters_7 := 7
def expected_distance_7_liters := 420

-- The rate of distance (km per liter)
def rate := truck_300_km_per_5_liters / liters_5

-- Proof statement
theorem truck_distance_on_7_liters :
  rate * liters_7 = expected_distance_7_liters :=
  by
  sorry

end truck_distance_on_7_liters_l1991_199130


namespace fewer_white_chairs_than_green_blue_l1991_199189

-- Definitions of the conditions
def blue_chairs : ℕ := 10
def green_chairs : ℕ := 3 * blue_chairs
def total_chairs : ℕ := 67
def green_blue_chairs : ℕ := green_chairs + blue_chairs
def white_chairs : ℕ := total_chairs - green_blue_chairs

-- Statement of the theorem
theorem fewer_white_chairs_than_green_blue : green_blue_chairs - white_chairs = 13 :=
by
  -- This is where the proof would go, but we're omitting it as per instruction
  sorry

end fewer_white_chairs_than_green_blue_l1991_199189


namespace rectangle_area_l1991_199121

theorem rectangle_area (L B : ℕ) 
  (h1 : L - B = 23)
  (h2 : 2 * L + 2 * B = 186) :
  L * B = 2030 := by
  sorry

end rectangle_area_l1991_199121


namespace union_A_B_l1991_199194

def A (x : ℝ) : Set ℝ := {x ^ 2, 2 * x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem union_A_B (x : ℝ) (h : {9} = A x ∩ B x) :
  (A x ∪ B x) = {(-8 : ℝ), -7, -4, 4, 9} := by
  sorry

end union_A_B_l1991_199194


namespace max_sum_x1_x2_x3_l1991_199195

theorem max_sum_x1_x2_x3 : 
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℕ), 
    x1 < x2 → x2 < x3 → x3 < x4 → x4 < x5 → x5 < x6 → x6 < x7 →
    x1 + x2 + x3 + x4 + x5 + x6 + x7 = 159 →
    x1 + x2 + x3 = 61 :=
by
  intros x1 x2 x3 x4 x5 x6 x7 h1 h2 h3 h4 h5 h6 h_sum
  sorry

end max_sum_x1_x2_x3_l1991_199195


namespace perimeter_of_square_l1991_199196

theorem perimeter_of_square (A : ℝ) (hA : A = 400) : exists P : ℝ, P = 80 :=
by
  sorry

end perimeter_of_square_l1991_199196


namespace unique_positive_integer_n_l1991_199149

-- Definitions based on conditions
def is_divisor (n a : ℕ) : Prop := a % n = 0

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- The main theorem statement
theorem unique_positive_integer_n : ∃ (n : ℕ), n > 0 ∧ is_divisor n 1989 ∧
    is_perfect_square (n^2 - 1989 / n) ∧ n = 13 :=
by
  sorry

end unique_positive_integer_n_l1991_199149


namespace combined_capacity_eq_l1991_199192

variable {x y z : ℚ}

-- Container A condition
def containerA_full (x : ℚ) := 0.75 * x
def containerA_initial (x : ℚ) := 0.30 * x
def containerA_diff (x : ℚ) := containerA_full x - containerA_initial x = 36

-- Container B condition
def containerB_full (y : ℚ) := 0.70 * y
def containerB_initial (y : ℚ) := 0.40 * y
def containerB_diff (y : ℚ) := containerB_full y - containerB_initial y = 20

-- Container C condition
def containerC_full (z : ℚ) := (2 / 3) * z
def containerC_initial (z : ℚ) := 0.50 * z
def containerC_diff (z : ℚ) := containerC_full z - containerC_initial z = 12

-- Theorem to prove the total capacity
theorem combined_capacity_eq : containerA_diff x → containerB_diff y → containerC_diff z → 
(218 + 2 / 3 = x + y + z) :=
by
  intros hA hB hC
  sorry

end combined_capacity_eq_l1991_199192


namespace necessary_but_not_sufficient_l1991_199114

def lines_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, (a * x + 2 * y = 0) ↔ (x + (a + 1) * y + 4 = 0)

theorem necessary_but_not_sufficient (a : ℝ) :
  (a = 1 → lines_parallel a) ∧ ¬(lines_parallel a → a = 1) :=
by
  sorry

end necessary_but_not_sufficient_l1991_199114


namespace clock_spoke_angle_l1991_199144

-- Define the parameters of the clock face and the problem.
def num_spokes := 10
def total_degrees := 360
def degrees_per_spoke := total_degrees / num_spokes
def position_3_oclock := 3 -- the third spoke
def halfway_45_oclock := 5 -- approximately the fifth spoke
def spokes_between := halfway_45_oclock - position_3_oclock
def smaller_angle := spokes_between * degrees_per_spoke
def expected_angle := 72

-- Statement of the problem
theorem clock_spoke_angle :
  smaller_angle = expected_angle := by
    -- Proof is omitted
    sorry

end clock_spoke_angle_l1991_199144


namespace tan_7pi_over_6_eq_1_over_sqrt_3_l1991_199164

theorem tan_7pi_over_6_eq_1_over_sqrt_3 : 
  ∀ θ : ℝ, θ = (7 * Real.pi) / 6 → Real.tan θ = 1 / Real.sqrt 3 :=
by
  intros θ hθ
  rw [hθ]
  sorry  -- Proof to be completed

end tan_7pi_over_6_eq_1_over_sqrt_3_l1991_199164


namespace booth_makes_50_per_day_on_popcorn_l1991_199180

-- Define the conditions as provided
def daily_popcorn_revenue (P : ℝ) : Prop :=
  let cotton_candy_revenue := 3 * P
  let total_days := 5
  let rent := 30
  let ingredients := 75
  let total_expenses := rent + ingredients
  let profit := 895
  let total_revenue_before_expenses := profit + total_expenses
  total_revenue_before_expenses = 20 * P 

theorem booth_makes_50_per_day_on_popcorn : daily_popcorn_revenue 50 :=
  by sorry

end booth_makes_50_per_day_on_popcorn_l1991_199180


namespace tan_half_prod_eq_sqrt3_l1991_199153

theorem tan_half_prod_eq_sqrt3 (a b : ℝ) (h : 7 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ (xy : ℝ), xy = Real.tan (a / 2) * Real.tan (b / 2) ∧ (xy = Real.sqrt 3 ∨ xy = -Real.sqrt 3) :=
by
  sorry

end tan_half_prod_eq_sqrt3_l1991_199153


namespace total_points_first_half_l1991_199147

def geometric_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (1 - r ^ n) / (1 - r)

def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * a + d * (n * (n - 1) / 2)

-- Given conditions:
variables (a r b d : ℕ)
variables (h1 : a = b)
variables (h2 : geometric_sum a r 4 = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
variables (h3 : a * (1 + r + r^2 + r^3) ≤ 120)
variables (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 120)

theorem total_points_first_half (a r b d : ℕ) (h1 : a = b) (h2 : a * (1 + r + r ^ 2 + r ^ 3) = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
  (h3 : a * (1 + r + r ^ 2 + r ^ 3) ≤ 120) (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 120) : 
  a + a * r + b + (b + d) = 45 :=
by
  sorry

end total_points_first_half_l1991_199147


namespace question1_question2_l1991_199173

def f (x : ℝ) : ℝ := abs (x - 5) - abs (x - 2)

theorem question1 :
  (∃ x : ℝ, f x ≤ m) ↔ m ≥ -3 :=
sorry

theorem question2 :
  { x : ℝ | x^2 - 8*x + 15 + f x ≤ 0 } = { x | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6 } :=
sorry

end question1_question2_l1991_199173


namespace general_term_of_series_l1991_199151

def gen_term (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, a n = if n = 1 then 2 else 6 * n - 5

def series_sum (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, S n = 3 * n ^ 2 - 2 * n + 1

theorem general_term_of_series (a S : ℕ → ℕ) (h : series_sum S) :
  gen_term a ↔ (∀ n : ℕ, a n = if n = 1 then 2 else S n - S (n - 1)) :=
by sorry

end general_term_of_series_l1991_199151


namespace smallest_three_digit_integer_l1991_199187

theorem smallest_three_digit_integer (n : ℕ) (h : 75 * n ≡ 225 [MOD 345]) (hne : n ≥ 100) (hn : n < 1000) : n = 118 :=
sorry

end smallest_three_digit_integer_l1991_199187


namespace fixed_point_exists_l1991_199188

-- Defining the function f
def f (a x : ℝ) : ℝ := a * x - 3 + 3

-- Stating that there exists a fixed point (3, 3a)
theorem fixed_point_exists (a : ℝ) : ∃ y : ℝ, f a 3 = y :=
by
  use (3 * a)
  simp [f]
  sorry

end fixed_point_exists_l1991_199188


namespace M_subset_N_l1991_199193

def M : Set ℚ := { x | ∃ k : ℤ, x = k / 2 + 1 / 4 }
def N : Set ℚ := { x | ∃ k : ℤ, x = k / 4 + 1 / 2 }

theorem M_subset_N : M ⊆ N :=
sorry

end M_subset_N_l1991_199193


namespace right_triangle_area_l1991_199162

theorem right_triangle_area (a : ℝ) (h : a > 2)
  (h_arith_seq : a - 2 > 0)
  (pythagorean : (a - 2)^2 + a^2 = (a + 2)^2) :
  (1 / 2) * (a - 2) * a = 24 :=
by
  sorry

end right_triangle_area_l1991_199162


namespace expression_value_l1991_199136

theorem expression_value : 
  (Nat.factorial 10) / (2 * (Finset.sum (Finset.range 11) id)) = 33080 := by
  sorry

end expression_value_l1991_199136


namespace angle_ABC_is_50_l1991_199101

theorem angle_ABC_is_50
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h1 : a = 90)
  (h2 : b = 60)
  (h3 : a + b + c = 200): c = 50 := by
  rw [h1, h2] at h3
  linarith

end angle_ABC_is_50_l1991_199101


namespace exists_line_with_two_colors_l1991_199119

open Classical

/-- Given a grid with 1x1 squares where each vertex is painted one of four colors such that each 1x1 square's vertices are all different colors, 
    there exists a line in the grid with nodes of exactly two different colors. -/
theorem exists_line_with_two_colors 
  (A : Type)
  [Inhabited A]
  [DecidableEq A]
  (colors : Finset A) 
  (h_col : colors.card = 4) 
  (grid : ℤ × ℤ → A) 
  (h_diff_colors : ∀ (i j : ℤ), i ≠ j → ∀ (k l : ℤ), grid (i, k) ≠ grid (j, k) ∧ grid (i, l) ≠ grid (i, k)) :
  ∃ line : ℤ → ℤ × ℤ, ∃ a b : A, a ≠ b ∧ ∀ n : ℤ, grid (line n) = a ∨ grid (line n) = b :=
sorry

end exists_line_with_two_colors_l1991_199119


namespace solve_B_l1991_199199

theorem solve_B (B : ℕ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 7 ∣ (4000 + 110 * B + 2)) : B = 4 :=
by
  sorry

end solve_B_l1991_199199


namespace exists_n_not_represented_l1991_199113

theorem exists_n_not_represented (a b c d : ℤ) (a_gt_14 : a > 14)
  (h1 : 0 ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ a) :
  ∃ (n : ℕ), ¬ ∃ (x y z : ℤ), n = x * (a * x + b) + y * (a * y + c) + z * (a * z + d) :=
sorry

end exists_n_not_represented_l1991_199113


namespace least_possible_faces_combined_l1991_199132

noncomputable def hasValidDiceConfiguration : Prop :=
  ∃ a b : ℕ,
  (∃ s8 s12 s13 : ℕ,
    (s8 = 3) ∧
    (s12 = 4) ∧
    (a ≥ 5 ∧ b = 6 ∧ (a + b = 11) ∧
      (2 * s12 = s8) ∧
      (2 * s8 = s13))
  )

theorem least_possible_faces_combined : hasValidDiceConfiguration :=
  sorry

end least_possible_faces_combined_l1991_199132


namespace function_symmetry_property_l1991_199191

noncomputable def f (x : ℝ) : ℝ :=
  x ^ 2

def symmetry_property := 
  ∀ (x : ℝ), (-1 < x ∧ x ≤ 1) →
    (¬ (f (-x) = f x) ∧ ¬ (f (-x) = -f x))

theorem function_symmetry_property :
  symmetry_property :=
by
  sorry

end function_symmetry_property_l1991_199191


namespace expression_value_l1991_199176

theorem expression_value (a b c d : ℝ) 
  (intersect1 : 4 = a * (2:ℝ)^2 + b * 2 + 1) 
  (intersect2 : 4 = (2:ℝ)^2 + c * 2 + d) 
  (hc : b + c = 1) : 
  4 * a + d = 1 := 
sorry

end expression_value_l1991_199176


namespace percentage_increase_first_year_l1991_199140

theorem percentage_increase_first_year (P : ℝ) (X : ℝ) 
  (h1 : P * (1 + X / 100) * 0.75 * 1.15 = P * 1.035) : 
  X = 20 :=
by
  sorry

end percentage_increase_first_year_l1991_199140


namespace repeating_decimal_fraction_l1991_199131

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l1991_199131


namespace power_division_calculation_l1991_199150

theorem power_division_calculation :
  ( ( 5^13 / 5^11 )^2 * 5^2 ) / 2^5 = 15625 / 32 :=
by
  sorry

end power_division_calculation_l1991_199150


namespace intersection_A_B_range_of_a_l1991_199184

-- Problem 1: Prove the intersection of A and B when a = 4
theorem intersection_A_B (a : ℝ) (h : a = 4) :
  { x : ℝ | 5 ≤ x ∧ x ≤ 7 } ∩ { x : ℝ | x ≤ 3 ∨ 5 < x} = {6, 7} :=
by sorry

-- Problem 2: Prove the range of values for a such that A ⊆ B
theorem range_of_a :
  { a : ℝ | (a < 2) ∨ (a > 4) } :=
by sorry

end intersection_A_B_range_of_a_l1991_199184


namespace greatest_length_measures_exactly_l1991_199111

theorem greatest_length_measures_exactly 
    (a b c : ℕ) 
    (ha : a = 700)
    (hb : b = 385)
    (hc : c = 1295) : 
    Nat.gcd (Nat.gcd a b) c = 35 := 
by
  sorry

end greatest_length_measures_exactly_l1991_199111


namespace non_red_fraction_l1991_199109

-- Define the conditions
def cube_edge : ℕ := 4
def num_cubes : ℕ := 64
def num_red_cubes : ℕ := 48
def num_white_cubes : ℕ := 12
def num_blue_cubes : ℕ := 4
def total_surface_area : ℕ := 6 * (cube_edge * cube_edge)

-- Define the non-red surface area exposed
def white_cube_exposed_area : ℕ := 12
def blue_cube_exposed_area : ℕ := 0

-- Calculating non-red area
def non_red_surface_area : ℕ := white_cube_exposed_area + blue_cube_exposed_area

-- The theorem to prove
theorem non_red_fraction (cube_edge : ℕ) (num_cubes : ℕ) (num_red_cubes : ℕ) 
  (num_white_cubes : ℕ) (num_blue_cubes : ℕ) (total_surface_area : ℕ) 
  (non_red_surface_area : ℕ) : 
  (non_red_surface_area : ℚ) / (total_surface_area : ℚ) = 1 / 8 :=
by 
  sorry

end non_red_fraction_l1991_199109


namespace john_using_three_colors_l1991_199103

theorem john_using_three_colors {total_paint liters_per_color : ℕ} 
    (h1 : total_paint = 15) 
    (h2 : liters_per_color = 5) :
    total_ppaint / liters_per_color = 3 := 
by
  sorry

end john_using_three_colors_l1991_199103


namespace pond_fish_count_l1991_199133

theorem pond_fish_count :
  (∃ (N : ℕ), (2 / 50 : ℚ) = (40 / N : ℚ)) → N = 1000 :=
by
  sorry

end pond_fish_count_l1991_199133


namespace classify_quadrilateral_l1991_199166

structure Quadrilateral where
  sides : ℕ → ℝ 
  angle : ℕ → ℝ 
  diag_length : ℕ → ℝ 
  perpendicular_diagonals : Prop

def is_rhombus (q : Quadrilateral) : Prop :=
  (∀ i, q.sides i = q.sides 0) ∧ q.perpendicular_diagonals

def is_kite (q : Quadrilateral) : Prop :=
  (q.sides 1 = q.sides 2 ∧ q.sides 3 = q.sides 4) ∧ q.perpendicular_diagonals

def is_square (q : Quadrilateral) : Prop :=
  (∀ i, q.sides i = q.sides 0) ∧ (∀ i, q.angle i = 90) ∧ q.perpendicular_diagonals

theorem classify_quadrilateral (q : Quadrilateral) (h : q.perpendicular_diagonals) :
  is_rhombus q ∨ is_kite q ∨ is_square q :=
sorry

end classify_quadrilateral_l1991_199166


namespace eval_power_imaginary_unit_l1991_199158

noncomputable def i : ℂ := Complex.I

theorem eval_power_imaginary_unit :
  i^20 + i^39 = 1 - i := by
  -- Skipping the proof itself, indicating it with "sorry"
  sorry

end eval_power_imaginary_unit_l1991_199158


namespace find_large_number_l1991_199137

theorem find_large_number (L S : ℕ) 
  (h1 : L - S = 1335) 
  (h2 : L = 6 * S + 15) : 
  L = 1599 := 
by 
  -- proof omitted
  sorry

end find_large_number_l1991_199137


namespace square_park_area_l1991_199172

theorem square_park_area (side_length : ℝ) (h : side_length = 200) : side_length * side_length = 40000 := by
  sorry

end square_park_area_l1991_199172


namespace correct_propositions_for_curve_C_l1991_199155

def curve_C (k : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / (4 - k) + y^2 / (k - 1) = 1)

theorem correct_propositions_for_curve_C (k : ℝ) :
  (∀ x y : ℝ, curve_C k) →
  ((∃ k, ((4 - k) * (k - 1) < 0) ↔ (k < 1 ∨ k > 4)) ∧
  ((1 < k ∧ k < (5 : ℝ) / 2) ↔
  (4 - k > k - 1 ∧ 4 - k > 0 ∧ k - 1 > 0))) :=
by {
  sorry
}

end correct_propositions_for_curve_C_l1991_199155


namespace integral_abs_sin_from_0_to_2pi_l1991_199142

theorem integral_abs_sin_from_0_to_2pi : ∫ x in (0 : ℝ)..(2 * Real.pi), |Real.sin x| = 4 := 
by
  sorry

end integral_abs_sin_from_0_to_2pi_l1991_199142


namespace find_omega_l1991_199154

theorem find_omega (ω : ℝ) (h₀ : ω > 0) (h₁ : (π / ω = π / 2)) : ω = 2 :=
sorry

end find_omega_l1991_199154


namespace max_rectangle_area_l1991_199148

theorem max_rectangle_area (a b : ℝ) (h : 2 * a + 2 * b = 60) :
  a * b ≤ 225 :=
by
  sorry

end max_rectangle_area_l1991_199148


namespace smallest_b_for_factorable_polynomial_l1991_199167

theorem smallest_b_for_factorable_polynomial :
  ∃ (b : ℕ), b > 0 ∧ (∃ (p q : ℤ), x^2 + b * x + 1176 = (x + p) * (x + q) ∧ p * q = 1176 ∧ p + q = b) ∧ 
  (∀ (b' : ℕ), b' > 0 → (∃ (p' q' : ℤ), x^2 + b' * x + 1176 = (x + p') * (x + q') ∧ p' * q' = 1176 ∧ p' + q' = b') → b ≤ b') :=
sorry

end smallest_b_for_factorable_polynomial_l1991_199167


namespace bill_fine_amount_l1991_199186

-- Define the conditions
def ounces_sold : ℕ := 8
def earnings_per_ounce : ℕ := 9
def amount_left : ℕ := 22

-- Calculate the earnings
def earnings : ℕ := ounces_sold * earnings_per_ounce

-- Define the fine as the difference between earnings and amount left
def fine : ℕ := earnings - amount_left

-- The proof problem to solve
theorem bill_fine_amount : fine = 50 :=
by
  -- Statements and calculations would go here
  sorry

end bill_fine_amount_l1991_199186


namespace prime_divisor_problem_l1991_199157

theorem prime_divisor_problem (d r : ℕ) (h1 : d > 1) (h2 : Prime d)
  (h3 : 1274 % d = r) (h4 : 1841 % d = r) (h5 : 2866 % d = r) : d - r = 6 :=
by
  sorry

end prime_divisor_problem_l1991_199157


namespace variance_eta_l1991_199185

noncomputable def xi : ℝ := sorry -- Define ξ as a real number (will be specified later)
noncomputable def eta : ℝ := sorry -- Define η as a real number (will be specified later)

-- Conditions
axiom xi_distribution : xi = 3 + 2*Real.sqrt 4 -- ξ follows a normal distribution with mean 3 and variance 4
axiom relationship : xi = 2*eta + 3 -- Given relationship between ξ and η

-- Theorem to prove the question
theorem variance_eta : sorry := sorry

end variance_eta_l1991_199185


namespace power_of_two_as_sum_of_squares_l1991_199171

theorem power_of_two_as_sum_of_squares (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℤ), x % 2 = 1 ∧ y % 2 = 1 ∧ (2^n = 7*x^2 + y^2) :=
by
  sorry

end power_of_two_as_sum_of_squares_l1991_199171


namespace quotient_calc_l1991_199141

theorem quotient_calc (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h_dividend : dividend = 139)
  (h_divisor : divisor = 19)
  (h_remainder : remainder = 6)
  (h_formula : dividend - remainder = quotient * divisor):
  quotient = 7 :=
by {
  -- Insert proof here
  sorry
}

end quotient_calc_l1991_199141


namespace evaluate_expression_l1991_199105

-- Define the ceiling of square roots for the given numbers
def ceil_sqrt_3 := 2
def ceil_sqrt_27 := 6
def ceil_sqrt_243 := 16

-- Main theorem statement
theorem evaluate_expression :
  ceil_sqrt_3 + ceil_sqrt_27 * 2 + ceil_sqrt_243 = 30 :=
by
  -- Sorry to indicate that the proof is skipped
  sorry

end evaluate_expression_l1991_199105


namespace bounded_area_l1991_199115

noncomputable def f (x : ℝ) : ℝ := (x + Real.sqrt (x^2 + 1))^(1/3) + (x - Real.sqrt (x^2 + 1))^(1/3)

def g (y : ℝ) : ℝ := y + 1

theorem bounded_area : 
  (∫ y in (0:ℝ)..(1:ℝ), (g y - f (g y))) = (5/8 : ℝ) := by
  sorry

end bounded_area_l1991_199115


namespace factor_expression_l1991_199182

theorem factor_expression (x : ℝ) :
  (16 * x^4 + 36 * x^2 - 9) - (4 * x^4 - 6 * x^2 - 9) = 6 * x^2 * (2 * x^2 + 7) :=
by
  sorry

end factor_expression_l1991_199182


namespace part_I_part_II_l1991_199102

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - 2 * x + 2 * a

theorem part_I (a : ℝ) :
  let x := Real.log 2
  ∃ I₁ I₂ : Set ℝ,
    (∀ x ∈ I₁, f a x > f a (Real.log 2)) ∧
    (∀ x ∈ I₂, f a x < f a (Real.log 2)) ∧
    I₁ = Set.Iio (Real.log 2) ∧
    I₂ = Set.Ioi (Real.log 2) ∧
    f a (Real.log 2) = 2 * (1 - Real.log 2 + a) :=
by sorry

theorem part_II (a : ℝ) (h : a > Real.log 2 - 1) (x : ℝ) (hx : 0 < x) :
  Real.exp x > x^2 - 2 * a * x + 1 :=
by sorry

end part_I_part_II_l1991_199102


namespace frustum_volume_correct_l1991_199104

noncomputable def volume_frustum (base_edge_original base_edge_smaller altitude_original altitude_smaller : ℝ) : ℝ :=
  let base_area_original := base_edge_original ^ 2
  let base_area_smaller := base_edge_smaller ^ 2
  let volume_original := (1 / 3) * base_area_original * altitude_original
  let volume_smaller := (1 / 3) * base_area_smaller * altitude_smaller
  volume_original - volume_smaller

theorem frustum_volume_correct :
  volume_frustum 16 8 10 5 = 2240 / 3 :=
by
  have h1 : volume_frustum 16 8 10 5 = 
    (1 / 3) * (16^2) * 10 - (1 / 3) * (8^2) * 5 := rfl
  simp only [pow_two] at h1
  norm_num at h1
  exact h1

end frustum_volume_correct_l1991_199104


namespace min_value_expression_l1991_199174

theorem min_value_expression :
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → (∃ (c : ℝ), c = 16 ∧ ∀ z, z = (1 / x + 9 / y) → z ≥ c) :=
by
  sorry

end min_value_expression_l1991_199174


namespace line_through_P_midpoint_l1991_199122

noncomputable section

open Classical

variables (l l1 l2 : ℝ → ℝ → Prop) (P A B : ℝ × ℝ)

def line1 (x y : ℝ) := 2 * x - y - 2 = 0
def line2 (x y : ℝ) := x + y + 3 = 0

theorem line_through_P_midpoint (P A B : ℝ × ℝ)
  (hP : P = (3, 0))
  (hl1 : ∀ x y, line1 x y → l x y)
  (hl2 : ∀ x y, line2 x y → l x y)
  (hmid : (P.1 = (A.1 + B.1) / 2) ∧ (P.2 = (A.2 + B.2) / 2)) :
  ∃ k : ℝ, ∀ x y, (y = k * (x - 3)) ↔ (8 * x - y - 24 = 0) :=
by
  sorry

end line_through_P_midpoint_l1991_199122


namespace cost_per_serving_l1991_199175

def pasta_cost : ℝ := 1.0
def sauce_cost : ℝ := 2.0
def meatballs_cost : ℝ := 5.0
def total_servings : ℝ := 8.0

theorem cost_per_serving : (pasta_cost + sauce_cost + meatballs_cost) / total_servings = 1.0 :=
by sorry

end cost_per_serving_l1991_199175


namespace sum_of_four_numbers_in_ratio_is_correct_l1991_199118

variable (A B C D : ℝ)
variable (h_ratio : A / B = 2 / 3 ∧ B / C = 3 / 4 ∧ C / D = 4 / 5)
variable (h_biggest : D = 672)

theorem sum_of_four_numbers_in_ratio_is_correct :
  A + B + C + D = 1881.6 :=
by
  sorry

end sum_of_four_numbers_in_ratio_is_correct_l1991_199118


namespace abs_eq_iff_x_eq_2_l1991_199183

theorem abs_eq_iff_x_eq_2 (x : ℝ) : |x - 1| = |x - 3| → x = 2 := by
  sorry

end abs_eq_iff_x_eq_2_l1991_199183


namespace nested_f_has_zero_l1991_199168

def f (x : ℝ) : ℝ := x^2 + 2017 * x + 1

theorem nested_f_has_zero (n : ℕ) (hn : n ≥ 1) : ∃ x : ℝ, (Nat.iterate f n x) = 0 :=
by
  sorry

end nested_f_has_zero_l1991_199168


namespace count_ordered_pairs_squares_diff_l1991_199177

theorem count_ordered_pairs_squares_diff (m n : ℕ) (h1 : m ≥ n) (h2 : m^2 - n^2 = 72) : 
∃ (a : ℕ), a = 3 :=
sorry

end count_ordered_pairs_squares_diff_l1991_199177


namespace num_orange_juice_l1991_199163

-- Definitions based on the conditions in the problem
def O : ℝ := sorry -- To represent the number of bottles of orange juice
def A : ℝ := sorry -- To represent the number of bottles of apple juice
def cost_orange_juice : ℝ := 0.70
def cost_apple_juice : ℝ := 0.60
def total_cost : ℝ := 46.20
def total_bottles : ℝ := 70

-- Conditions used as definitions in Lean 4
axiom condition1 : O + A = total_bottles
axiom condition2 : cost_orange_juice * O + cost_apple_juice * A = total_cost

-- Proof statement with the correct answer
theorem num_orange_juice : O = 42 := by
  sorry

end num_orange_juice_l1991_199163


namespace lassie_original_bones_l1991_199128

variable (B : ℕ) -- B is the number of bones Lassie started with

-- Conditions translated into Lean statements
def eats_half_on_saturday (B : ℕ) : ℕ := B / 2
def receives_ten_more_on_sunday (B : ℕ) : ℕ := eats_half_on_saturday B + 10
def total_bones_after_sunday (B : ℕ) : Prop := receives_ten_more_on_sunday B = 35

-- Proof goal: B is equal to 50 given the conditions
theorem lassie_original_bones :
  total_bones_after_sunday B → B = 50 :=
sorry

end lassie_original_bones_l1991_199128


namespace cassandra_overall_score_l1991_199190

theorem cassandra_overall_score 
  (score1_percent : ℤ) (score1_total : ℕ)
  (score2_percent : ℤ) (score2_total : ℕ)
  (score3_percent : ℤ) (score3_total : ℕ) :
  score1_percent = 60 → score1_total = 15 →
  score2_percent = 75 → score2_total = 20 →
  score3_percent = 85 → score3_total = 25 →
  let correct1 := (score1_percent * score1_total) / 100
  let correct2 := (score2_percent * score2_total) / 100
  let correct3 := (score3_percent * score3_total) / 100
  let total_correct := correct1 + correct2 + correct3
  let total_problems := score1_total + score2_total + score3_total
  75 = (100 * total_correct) / total_problems := by
  intros h1 h2 h3 h4 h5 h6
  let correct1 := (60 * 15) / 100
  let correct2 := (75 * 20) / 100
  let correct3 := (85 * 25) / 100
  let total_correct := correct1 + correct2 + correct3
  let total_problems := 15 + 20 + 25
  suffices 75 = (100 * total_correct) / total_problems by sorry
  sorry

end cassandra_overall_score_l1991_199190


namespace problemStatement_l1991_199145

-- Define the set of values as a type
structure SetOfValues where
  k : ℤ
  b : ℤ

-- The given sets of values
def A : SetOfValues := ⟨2, 2⟩
def B : SetOfValues := ⟨2, -2⟩
def C : SetOfValues := ⟨-2, -2⟩
def D : SetOfValues := ⟨-2, 2⟩

-- Define the conditions for the function
def isValidSet (s : SetOfValues) : Prop :=
  s.k < 0 ∧ s.b > 0

-- The problem statement: Prove that D is a valid set
theorem problemStatement : isValidSet D := by
  sorry

end problemStatement_l1991_199145


namespace minimum_value_of_functions_l1991_199116

def linear_fn (a b c: ℝ) := a ≠ 0 
def f (a b: ℝ) (x: ℝ) := a * x + b 
def g (a c: ℝ) (x: ℝ) := a * x + c

theorem minimum_value_of_functions (a b c: ℝ) (hx: linear_fn a b c) :
  (∀ x: ℝ, 3 * (f a b x)^2 + 2 * g a c x ≥ -19 / 6) → (∀ x: ℝ, 3 * (g a c x)^2 + 2 * f a b x ≥ 5 / 2) :=
by
  sorry

end minimum_value_of_functions_l1991_199116


namespace number_of_boys_is_10_l1991_199134

-- Definitions based on given conditions
def num_children := 20
def has_blue_neighbor_clockwise (i : ℕ) : Prop := true -- Dummy predicate representing condition
def has_red_neighbor_counterclockwise (i : ℕ) : Prop := true -- Dummy predicate representing condition

axiom boys_and_girls_exist : ∃ b g : ℤ, b + g = num_children ∧ b > 0 ∧ g > 0

-- Theorem based on the problem statement
theorem number_of_boys_is_10 (b g : ℤ) 
  (total_children: b + g = num_children)
  (boys_exist: b > 0)
  (girls_exist: g > 0)
  (each_boy_has_blue_neighbor: ∀ i, has_blue_neighbor_clockwise i → true)
  (each_girl_has_red_neighbor: ∀ i, has_red_neighbor_counterclockwise i → true): 
  b = 10 :=
by
  sorry

end number_of_boys_is_10_l1991_199134


namespace calculate_expression_l1991_199197

theorem calculate_expression :
  ((-1 -2 -3 -4 -5 -6 -7 -8 -9 -10) * (1 -2 +3 -4 +5 -6 +7 -8 +9 -10) = 275) :=
by
  sorry

end calculate_expression_l1991_199197


namespace flower_bed_l1991_199138

def planting_schemes (A B C D E F : Prop) : Prop :=
  A ≠ B ∧ B ≠ C ∧ D ≠ E ∧ E ≠ F ∧ A ≠ D ∧ B ≠ D ∧ B ≠ E ∧ C ≠ E ∧ C ≠ F ∧ D ≠ F

theorem flower_bed (A B C D E F : Prop) (plant_choices : Finset (Fin 6))
  (h_choice : plant_choices.card = 6)
  (h_different : ∀ x ∈ plant_choices, ∀ y ∈ plant_choices, x ≠ y → x ≠ y)
  (h_adj : planting_schemes A B C D E F) :
  ∃! planting_schemes, planting_schemes ∧ plant_choices.card = 13230 :=
by sorry

end flower_bed_l1991_199138


namespace sam_total_cans_l1991_199106

theorem sam_total_cans (bags_saturday bags_sunday bags_total cans_per_bag total_cans : ℕ)
    (h1 : bags_saturday = 3)
    (h2 : bags_sunday = 4)
    (h3 : bags_total = bags_saturday + bags_sunday)
    (h4 : cans_per_bag = 9)
    (h5 : total_cans = bags_total * cans_per_bag) : total_cans = 63 :=
sorry

end sam_total_cans_l1991_199106


namespace inclination_angle_x_eq_one_l1991_199156

noncomputable def inclination_angle_of_vertical_line (x : ℝ) : ℝ :=
if x = 1 then 90 else 0

theorem inclination_angle_x_eq_one :
  inclination_angle_of_vertical_line 1 = 90 :=
by
  sorry

end inclination_angle_x_eq_one_l1991_199156


namespace negation_of_P_l1991_199159

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x^2 + 2*x + 2 > 0

-- State the negation of P
theorem negation_of_P : ¬P ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by
  sorry

end negation_of_P_l1991_199159
