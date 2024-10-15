import Mathlib

namespace NUMINAMATH_GPT_num_outfits_l578_57810

-- Define the number of trousers, shirts, and jackets available
def num_trousers : Nat := 5
def num_shirts : Nat := 6
def num_jackets : Nat := 4

-- Define the main theorem
theorem num_outfits (t : Nat) (s : Nat) (j : Nat) (ht : t = num_trousers) (hs : s = num_shirts) (hj : j = num_jackets) :
  t * s * j = 120 :=
by 
  rw [ht, hs, hj]
  exact rfl

end NUMINAMATH_GPT_num_outfits_l578_57810


namespace NUMINAMATH_GPT_area_of_rectangular_park_l578_57864

theorem area_of_rectangular_park
  (l w : ℕ) 
  (h_perimeter : 2 * l + 2 * w = 80)
  (h_length : l = 3 * w) :
  l * w = 300 :=
sorry

end NUMINAMATH_GPT_area_of_rectangular_park_l578_57864


namespace NUMINAMATH_GPT_peaches_eaten_l578_57826

theorem peaches_eaten (P B Baskets P_each R Boxes P_box : ℕ) 
  (h1 : B = 5) 
  (h2 : P_each = 25)
  (h3 : Baskets = B * P_each)
  (h4 : R = 8) 
  (h5 : P_box = 15)
  (h6 : Boxes = R * P_box)
  (h7 : P = Baskets - Boxes) : P = 5 :=
by sorry

end NUMINAMATH_GPT_peaches_eaten_l578_57826


namespace NUMINAMATH_GPT_twenty_percent_l578_57843

-- Given condition
def condition (X : ℝ) : Prop := 0.4 * X = 160

-- Theorem to show that 20% of X equals 80 given the condition
theorem twenty_percent (X : ℝ) (h : condition X) : 0.2 * X = 80 :=
by sorry

end NUMINAMATH_GPT_twenty_percent_l578_57843


namespace NUMINAMATH_GPT_john_ultramarathon_distance_l578_57816

theorem john_ultramarathon_distance :
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  initial_time * (1 + time_increase_percentage) * (initial_speed + speed_increase) = 168 :=
by
  let initial_time := 8
  let time_increase_percentage := 0.75
  let speed_increase := 4
  let initial_speed := 8
  sorry

end NUMINAMATH_GPT_john_ultramarathon_distance_l578_57816


namespace NUMINAMATH_GPT_georgia_vs_texas_license_plates_l578_57845

theorem georgia_vs_texas_license_plates :
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  georgia_plates - texas_plates = 731161600 :=
by
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  show georgia_plates - texas_plates = 731161600
  sorry

end NUMINAMATH_GPT_georgia_vs_texas_license_plates_l578_57845


namespace NUMINAMATH_GPT_sum_of_triangles_l578_57827

def triangle (a b c : ℕ) : ℕ :=
  (a * b) + c

theorem sum_of_triangles : 
  triangle 4 2 3 + triangle 5 3 2 = 28 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_triangles_l578_57827


namespace NUMINAMATH_GPT_Alyssa_has_37_balloons_l578_57847

variable (Sandy_balloons : ℕ) (Sally_balloons : ℕ) (Total_balloons : ℕ)

-- Conditions
axiom Sandy_Condition : Sandy_balloons = 28
axiom Sally_Condition : Sally_balloons = 39
axiom Total_Condition : Total_balloons = 104

-- Definition of Alyssa's balloons
def Alyssa_balloons : ℕ := Total_balloons - (Sandy_balloons + Sally_balloons)

-- The proof statement 
theorem Alyssa_has_37_balloons 
: Alyssa_balloons Sandy_balloons Sally_balloons Total_balloons = 37 :=
by
  -- The proof body will be placed here, but we will leave it as a placeholder for now
  sorry

end NUMINAMATH_GPT_Alyssa_has_37_balloons_l578_57847


namespace NUMINAMATH_GPT_probability_log2_x_between_1_and_2_l578_57872

noncomputable def probability_log_between : ℝ :=
  let favorable_range := (4:ℝ) - (2:ℝ)
  let total_range := (6:ℝ) - (0:ℝ)
  favorable_range / total_range

theorem probability_log2_x_between_1_and_2 :
  probability_log_between = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_log2_x_between_1_and_2_l578_57872


namespace NUMINAMATH_GPT_total_clouds_count_l578_57853

def carson_clouds := 12
def little_brother_clouds := 5 * carson_clouds
def older_sister_clouds := carson_clouds / 2
def cousin_clouds := 2 * (older_sister_clouds + carson_clouds)

theorem total_clouds_count : carson_clouds + little_brother_clouds + older_sister_clouds + cousin_clouds = 114 := by
  sorry

end NUMINAMATH_GPT_total_clouds_count_l578_57853


namespace NUMINAMATH_GPT_basin_more_than_tank2_l578_57866

/-- Define the water volumes in milliliters -/
def volume_bottle1 : ℕ := 1000 -- 1 liter = 1000 milliliters
def volume_bottle2 : ℕ := 400  -- 400 milliliters
def volume_tank : ℕ := 2800    -- 2800 milliliters
def volume_basin : ℕ := volume_bottle1 + volume_bottle2 + volume_tank -- total volume in basin
def volume_tank2 : ℕ := 4000 + 100 -- 4 liters 100 milliliters tank

/-- Theorem: The basin can hold 100 ml more water than the 4-liter 100-milliliter tank -/
theorem basin_more_than_tank2 : volume_basin = volume_tank2 + 100 :=
by
  -- This is where the proof would go, but it is not required for this exercise
  sorry

end NUMINAMATH_GPT_basin_more_than_tank2_l578_57866


namespace NUMINAMATH_GPT_difference_of_sums_1000_l578_57811

def sum_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_first_n_odd_not_divisible_by_5 (n : ℕ) : ℕ :=
  (n * n) - 5 * ((n / 5) * ((n / 5) + 1))

theorem difference_of_sums_1000 :
  (sum_first_n_even 1000) - (sum_first_n_odd_not_divisible_by_5 1000) = 51000 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_sums_1000_l578_57811


namespace NUMINAMATH_GPT_root_in_interval_l578_57823

theorem root_in_interval (a b c : ℝ) (h_a : a ≠ 0)
    (h_table : ∀ x y, (x = 1.2 ∧ y = -1.16) ∨ (x = 1.3 ∧ y = -0.71) ∨ (x = 1.4 ∧ y = -0.24) ∨ (x = 1.5 ∧ y = 0.25) ∨ (x = 1.6 ∧ y = 0.76) → y = a * x^2 + b * x + c ) :
  ∃ x₁, 1.4 < x₁ ∧ x₁ < 1.5 ∧ a * x₁^2 + b * x₁ + c = 0 :=
by sorry

end NUMINAMATH_GPT_root_in_interval_l578_57823


namespace NUMINAMATH_GPT_number_of_solutions_l578_57815

noncomputable def g (x : ℝ) : ℝ := -3 * Real.sin (2 * Real.pi * x)

theorem number_of_solutions (h : -1 ≤ x ∧ x ≤ 1) : 
  (∃ s : ℕ, s = 21 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g (g (g x)) = g x) :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l578_57815


namespace NUMINAMATH_GPT_cryptarithm_solution_l578_57829

theorem cryptarithm_solution (A B C D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_adjacent : A = C + 1 ∨ A = C - 1)
  (h_diff : B = D + 2 ∨ B = D - 2) :
  1000 * A + 100 * B + 10 * C + D = 5240 :=
sorry

end NUMINAMATH_GPT_cryptarithm_solution_l578_57829


namespace NUMINAMATH_GPT_Mary_and_Sandra_solution_l578_57873

theorem Mary_and_Sandra_solution (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) :
  (2 * 40 + 3 * 60) * n / (5 * n) = (4 * 30 * n + 80 * m) / (4 * n + m) →
  m + n = 29 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Mary_and_Sandra_solution_l578_57873


namespace NUMINAMATH_GPT_solve_2xx_eq_sqrt2_unique_solution_l578_57891

noncomputable def solve_equation_2xx_eq_sqrt2 (x : ℝ) : Prop :=
  2 * x^x = Real.sqrt 2

theorem solve_2xx_eq_sqrt2_unique_solution (x : ℝ) : solve_equation_2xx_eq_sqrt2 x ↔ (x = 1/2 ∨ x = 1/4) ∧ x > 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_2xx_eq_sqrt2_unique_solution_l578_57891


namespace NUMINAMATH_GPT_place_numbers_l578_57809

theorem place_numbers (a b c d : ℕ) (hab : Nat.gcd a b = 1) (hac : Nat.gcd a c = 1) 
  (had : Nat.gcd a d = 1) (hbc : Nat.gcd b c = 1) (hbd : Nat.gcd b d = 1) 
  (hcd : Nat.gcd c d = 1) :
  ∃ (bc ad ab cd abcd : ℕ), 
    bc = b * c ∧ ad = a * d ∧ ab = a * b ∧ cd = c * d ∧ abcd = a * b * c * d ∧
    Nat.gcd bc abcd > 1 ∧ Nat.gcd ad abcd > 1 ∧ Nat.gcd ab abcd > 1 ∧ 
    Nat.gcd cd abcd > 1 ∧
    Nat.gcd ab cd = 1 ∧ Nat.gcd ab ad = 1 ∧ Nat.gcd ab bc = 1 ∧ 
    Nat.gcd cd ad = 1 ∧ Nat.gcd cd bc = 1 ∧ Nat.gcd ad bc = 1 :=
by
  sorry

end NUMINAMATH_GPT_place_numbers_l578_57809


namespace NUMINAMATH_GPT_line_parabola_intersections_l578_57887

theorem line_parabola_intersections (k : ℝ) :
  ((∃ x y, y = k * (x - 2) + 1 ∧ y^2 = 4 * x) ↔ k = 0) ∧
  (¬∃ x₁ x₂, x₁ ≠ x₂ ∧ (k * (x₁ - 2) + 1)^2 = 4 * x₁ ∧ (k * (x₂ - 2) + 1)^2 = 4 * x₂) ∧
  (¬∃ x y, y = k * (x - 2) + 1 ∧ y^2 = 4 * x) :=
by sorry

end NUMINAMATH_GPT_line_parabola_intersections_l578_57887


namespace NUMINAMATH_GPT_brick_width_l578_57858

theorem brick_width (l_brick : ℕ) (w_courtyard l_courtyard : ℕ) (num_bricks : ℕ) (w_brick : ℕ)
  (H1 : l_courtyard = 24) 
  (H2 : w_courtyard = 14) 
  (H3 : num_bricks = 8960) 
  (H4 : l_brick = 25) 
  (H5 : (w_courtyard * 100 * l_courtyard * 100 = (num_bricks * (l_brick * w_brick)))) :
  w_brick = 15 :=
by
  sorry

end NUMINAMATH_GPT_brick_width_l578_57858


namespace NUMINAMATH_GPT_train_passing_platform_time_l578_57842

-- Conditions
variable (l t : ℝ) -- Length of the train and time to pass the pole
variable (v : ℝ) -- Velocity of the train
variable (n : ℝ) -- Multiple of t seconds to pass the platform
variable (d_platform : ℝ) -- Length of the platform

-- Theorem statement
theorem train_passing_platform_time (h1 : d_platform = 3 * l) (h2 : v = l / t) (h3 : n = (l + d_platform) / l) :
  n = 4 := by
  sorry

end NUMINAMATH_GPT_train_passing_platform_time_l578_57842


namespace NUMINAMATH_GPT_uphill_flat_road_system_l578_57817

variables {x y : ℝ}

theorem uphill_flat_road_system :
  (3 : ℝ)⁻¹ * x + (4 : ℝ)⁻¹ * y = 70 / 60 ∧
  (4 : ℝ)⁻¹ * y + (5 : ℝ)⁻¹ * x = 54 / 60 :=
sorry

end NUMINAMATH_GPT_uphill_flat_road_system_l578_57817


namespace NUMINAMATH_GPT_alice_bob_probability_l578_57813

noncomputable def probability_of_exactly_two_sunny_days : ℚ :=
  let p_sunny := 3 / 5
  let p_rain := 2 / 5
  3 * (p_sunny^2 * p_rain)

theorem alice_bob_probability :
  probability_of_exactly_two_sunny_days = 54 / 125 := 
sorry

end NUMINAMATH_GPT_alice_bob_probability_l578_57813


namespace NUMINAMATH_GPT_number_of_pencils_is_11_l578_57840

noncomputable def numberOfPencils (A B : ℕ) :  ℕ :=
  2 * A + 1 * B

theorem number_of_pencils_is_11 (A B : ℕ) (h1 : A + 2 * B = 16) (h2 : A + B = 9) : numberOfPencils A B = 11 :=
  sorry

end NUMINAMATH_GPT_number_of_pencils_is_11_l578_57840


namespace NUMINAMATH_GPT_nitin_borrowed_amount_l578_57803

theorem nitin_borrowed_amount (P : ℝ) (I1 I2 I3 : ℝ) :
  (I1 = P * 0.06 * 3) ∧
  (I2 = P * 0.09 * 5) ∧
  (I3 = P * 0.13 * 3) ∧
  (I1 + I2 + I3 = 8160) →
  P = 8000 :=
by
  sorry

end NUMINAMATH_GPT_nitin_borrowed_amount_l578_57803


namespace NUMINAMATH_GPT_sammy_remaining_problems_l578_57855

variable (total_problems : Nat)
variable (fraction_problems : Nat) (decimal_problems : Nat) (multiplication_problems : Nat) (division_problems : Nat)
variable (completed_fraction_problems : Nat) (completed_decimal_problems : Nat)
variable (completed_multiplication_problems : Nat) (completed_division_problems : Nat)
variable (remaining_problems : Nat)

theorem sammy_remaining_problems
  (h₁ : total_problems = 115)
  (h₂ : fraction_problems = 35)
  (h₃ : decimal_problems = 40)
  (h₄ : multiplication_problems = 20)
  (h₅ : division_problems = 20)
  (h₆ : completed_fraction_problems = 11)
  (h₇ : completed_decimal_problems = 17)
  (h₈ : completed_multiplication_problems = 9)
  (h₉ : completed_division_problems = 5)
  (h₁₀ : remaining_problems =
    fraction_problems - completed_fraction_problems +
    decimal_problems - completed_decimal_problems +
    multiplication_problems - completed_multiplication_problems +
    division_problems - completed_division_problems) :
  remaining_problems = 73 :=
  by
    -- proof to be written
    sorry

end NUMINAMATH_GPT_sammy_remaining_problems_l578_57855


namespace NUMINAMATH_GPT_twentieth_term_is_78_l578_57831

-- Define the arithmetic sequence parameters
def first_term : ℤ := 2
def common_difference : ℤ := 4

-- Define the function to compute the n-th term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ := first_term + (n - 1) * common_difference

-- Formulate the theorem to prove
theorem twentieth_term_is_78 : nth_term 20 = 78 :=
by
  sorry

end NUMINAMATH_GPT_twentieth_term_is_78_l578_57831


namespace NUMINAMATH_GPT_cost_of_nuts_l578_57894

/--
Adam bought 3 kilograms of nuts and 2.5 kilograms of dried fruits at a store. 
One kilogram of nuts costs a certain amount N and one kilogram of dried fruit costs $8. 
His purchases cost $56. Prove that one kilogram of nuts costs $12.
-/
theorem cost_of_nuts (N : ℝ) 
  (h1 : 3 * N + 2.5 * 8 = 56) 
  : N = 12 := by
  sorry

end NUMINAMATH_GPT_cost_of_nuts_l578_57894


namespace NUMINAMATH_GPT_min_value_of_squares_l578_57834

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) : a^2 + b^2 + c^2 ≥ t^2 / 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_squares_l578_57834


namespace NUMINAMATH_GPT_percentage_difference_y_less_than_z_l578_57837

-- Define the variables and the conditions
variables (x y z : ℝ)
variables (h₁ : x = 12 * y)
variables (h₂ : z = 1.2 * x)

-- Define the theorem statement
theorem percentage_difference_y_less_than_z (h₁ : x = 12 * y) (h₂ : z = 1.2 * x) :
  ((z - y) / z) * 100 = 93.06 := by
  sorry

end NUMINAMATH_GPT_percentage_difference_y_less_than_z_l578_57837


namespace NUMINAMATH_GPT_identify_correct_statement_l578_57804

-- Definitions based on conditions
def population (athletes : ℕ) : Prop := athletes = 1000
def is_individual (athlete : ℕ) : Prop := athlete ≤ 1000
def is_sample (sampled_athletes : ℕ) (sample_size : ℕ) : Prop := sampled_athletes = 100 ∧ sample_size = 100

-- Theorem statement based on the conclusion
theorem identify_correct_statement (athletes : ℕ) (sampled_athletes : ℕ) (sample_size : ℕ)
    (h1 : population athletes) (h2 : ∀ a, is_individual a) (h3 : is_sample sampled_athletes sample_size) : 
    (sampled_athletes = 100) ∧ (sample_size = 100) :=
by
  sorry

end NUMINAMATH_GPT_identify_correct_statement_l578_57804


namespace NUMINAMATH_GPT_village_Y_initial_population_l578_57844

def population_X := 76000
def decrease_rate_X := 1200
def increase_rate_Y := 800
def years := 17

def population_X_after_17_years := population_X - decrease_rate_X * years
def population_Y_after_17_years (P : Nat) := P + increase_rate_Y * years

theorem village_Y_initial_population (P : Nat) (h : population_Y_after_17_years P = population_X_after_17_years) : P = 42000 :=
by
  sorry

end NUMINAMATH_GPT_village_Y_initial_population_l578_57844


namespace NUMINAMATH_GPT_minimize_folded_area_l578_57851

-- defining the problem as statements in Lean
variables (a M N : ℝ) (M_on_AB : M > 0 ∧ M < a) (N_on_CD : N > 0 ∧ N < a)

-- main theorem statement
theorem minimize_folded_area :
  BM = 5 * a / 8 →
  CN = a / 8 →
  S = 3 * a ^ 2 / 8 := sorry

end NUMINAMATH_GPT_minimize_folded_area_l578_57851


namespace NUMINAMATH_GPT_parallelogram_larger_angle_l578_57884

theorem parallelogram_larger_angle (a b : ℕ) (h₁ : b = a + 50) (h₂ : a = 65) : b = 115 := 
by
  -- Use the conditions h₁ and h₂ to prove the statement.
  sorry

end NUMINAMATH_GPT_parallelogram_larger_angle_l578_57884


namespace NUMINAMATH_GPT_tree_height_is_12_l578_57846

-- Let h be the height of the tree in meters.
def height_of_tree (h : ℝ) : Prop :=
  ∃ h, (h / 8 = 150 / 100) → h = 12

theorem tree_height_is_12 : ∃ h : ℝ, height_of_tree h :=
by {
  sorry
}

end NUMINAMATH_GPT_tree_height_is_12_l578_57846


namespace NUMINAMATH_GPT_value_of_x_y_mn_l578_57880

variables (x y m n : ℝ)

-- Conditions for arithmetic sequence 2, x, y, 3
def arithmetic_sequence_condition_1 : Prop := 2 * x = 2 + y
def arithmetic_sequence_condition_2 : Prop := 2 * y = 3 + x

-- Conditions for geometric sequence 2, m, n, 3
def geometric_sequence_condition_1 : Prop := m^2 = 2 * n
def geometric_sequence_condition_2 : Prop := n^2 = 3 * m

theorem value_of_x_y_mn (h1 : arithmetic_sequence_condition_1 x y) 
                        (h2 : arithmetic_sequence_condition_2 x y) 
                        (h3 : geometric_sequence_condition_1 m n)
                        (h4 : geometric_sequence_condition_2 m n) : 
  x + y + m * n = 11 :=
sorry

end NUMINAMATH_GPT_value_of_x_y_mn_l578_57880


namespace NUMINAMATH_GPT_quadrilateral_ABCD_r_plus_s_l578_57807

noncomputable def AB_is (AB : Real) (r s : Nat) : Prop :=
  AB = r + Real.sqrt s

theorem quadrilateral_ABCD_r_plus_s :
  ∀ (BC CD AD : Real) (mA mB : ℕ) (r s : ℕ), 
  BC = 7 → 
  CD = 10 → 
  AD = 8 → 
  mA = 60 → 
  mB = 60 → 
  AB_is AB r s →
  r + s = 99 :=
by intros BC CD AD mA mB r s hBC hCD hAD hMA hMB hAB_is
   sorry

end NUMINAMATH_GPT_quadrilateral_ABCD_r_plus_s_l578_57807


namespace NUMINAMATH_GPT_no_constant_term_l578_57825

theorem no_constant_term (n : ℕ) (hn : ∀ r : ℕ, ¬(n = (4 * r) / 3)) : n ≠ 8 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_no_constant_term_l578_57825


namespace NUMINAMATH_GPT_power_function_no_origin_l578_57892

theorem power_function_no_origin (m : ℝ) :
  (m = 1 ∨ m = 2) → 
  (m^2 - 3 * m + 3 ≠ 0 ∧ (m - 2) * (m + 1) ≤ 0) :=
by
  intro h
  cases h
  case inl =>
    -- m = 1 case will be processed here
    sorry
  case inr =>
    -- m = 2 case will be processed here
    sorry

end NUMINAMATH_GPT_power_function_no_origin_l578_57892


namespace NUMINAMATH_GPT_algebraic_expression_value_l578_57824

theorem algebraic_expression_value : 
  ∀ (a b : ℝ), (∃ x, x = -2 ∧ a * x - b = 1) → 4 * a + 2 * b + 7 = 5 :=
by
  intros a b h
  cases' h with x hx
  cases' hx with hx1 hx2
  rw [hx1] at hx2
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l578_57824


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_l578_57890

-- Definition of the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 6 * x + 13

-- Statement of the proof problem
theorem minimum_value_of_quadratic : ∃ (y : ℝ), ∀ x : ℝ, quadratic x >= y ∧ y = 4 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_l578_57890


namespace NUMINAMATH_GPT_range_of_a_l578_57854

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ -x^2 + 4*x + a = 0) ↔ (-3 ≤ a ∧ a ≤ 21) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l578_57854


namespace NUMINAMATH_GPT_calculate_p_op_l578_57830

def op (x y : ℝ) := x * y^2 - x

theorem calculate_p_op (p : ℝ) : op p (op p p) = p^7 - 2*p^5 + p^3 - p :=
by
  sorry

end NUMINAMATH_GPT_calculate_p_op_l578_57830


namespace NUMINAMATH_GPT_fewest_four_dollar_frisbees_l578_57849

theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : x + y = 64) (h2 : 3 * x + 4 * y = 196) : y = 4 :=
by
  sorry

end NUMINAMATH_GPT_fewest_four_dollar_frisbees_l578_57849


namespace NUMINAMATH_GPT_evaluate_g_neg5_l578_57819

def g (x : ℝ) : ℝ := 4 * x - 2

theorem evaluate_g_neg5 : g (-5) = -22 := 
  by sorry

end NUMINAMATH_GPT_evaluate_g_neg5_l578_57819


namespace NUMINAMATH_GPT_relationship_between_abc_l578_57878

noncomputable def a := (4 / 5) ^ (1 / 2)
noncomputable def b := (5 / 4) ^ (1 / 5)
noncomputable def c := (3 / 4) ^ (3 / 4)

theorem relationship_between_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_GPT_relationship_between_abc_l578_57878


namespace NUMINAMATH_GPT_area_triangle_FQH_l578_57814

open Set

structure Point where
  x : ℝ
  y : ℝ

def Rectangle (A B C D : Point) : Prop :=
  A.x = B.x ∧ C.x = D.x ∧ A.y = D.y ∧ B.y = C.y

def IsMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def AreaTrapezoid (A B C D : Point) : ℝ :=
  0.5 * (B.x - A.x + D.x - C.x) * (A.y - C.y)

def AreaTriangle (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

variables (E P R H F Q G : Point)

-- Conditions
axiom h1 : Rectangle E F G H
axiom h2 : E.y - P.y = 8
axiom h3 : R.y - H.y = 8
axiom h4 : F.x - E.x = 16
axiom h5 : AreaTrapezoid P R H G = 160

-- Target to prove
theorem area_triangle_FQH : AreaTriangle F Q H = 80 :=
sorry

end NUMINAMATH_GPT_area_triangle_FQH_l578_57814


namespace NUMINAMATH_GPT_peaches_thrown_away_l578_57806

variables (total_peaches fresh_percentage peaches_left : ℕ) (thrown_away : ℕ)
variables (h1 : total_peaches = 250) (h2 : fresh_percentage = 60) (h3 : peaches_left = 135)

theorem peaches_thrown_away :
  thrown_away = (total_peaches * (fresh_percentage / 100)) - peaches_left :=
sorry

end NUMINAMATH_GPT_peaches_thrown_away_l578_57806


namespace NUMINAMATH_GPT_smallest_n_mod5_l578_57859

theorem smallest_n_mod5 :
  ∃ n : ℕ, n > 0 ∧ 6^n % 5 = n^6 % 5 ∧ ∀ m : ℕ, m > 0 ∧ 6^m % 5 = m^6 % 5 → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_mod5_l578_57859


namespace NUMINAMATH_GPT_ratio_pow_eq_l578_57899

theorem ratio_pow_eq {x y : ℝ} (h : x / y = 7 / 5) : (x^3 / y^2) = 343 / 25 :=
by sorry

end NUMINAMATH_GPT_ratio_pow_eq_l578_57899


namespace NUMINAMATH_GPT_road_network_possible_l578_57869

theorem road_network_possible (n : ℕ) :
  (n = 6 → true) ∧ (n = 1986 → false) :=
by {
  -- Proof of the statement goes here.
  sorry
}

end NUMINAMATH_GPT_road_network_possible_l578_57869


namespace NUMINAMATH_GPT_exists_small_area_triangle_l578_57801

structure Point :=
(x : ℝ)
(y : ℝ)

def is_valid_point (p : Point) : Prop :=
(|p.x| ≤ 2) ∧ (|p.y| ≤ 2)

def no_three_collinear (points : List Point) : Prop :=
  ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points →
  (p1 ≠ p2) → (p1 ≠ p3) → (p2 ≠ p3) →
  ((p1.y - p2.y) * (p1.x - p3.x) ≠ (p1.y - p3.y) * (p1.x - p2.x))

noncomputable def triangle_area (p1 p2 p3: Point) : ℝ :=
(abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))) / 2

theorem exists_small_area_triangle (points : List Point)
  (h_valid : ∀ p ∈ points, is_valid_point p)
  (h_no_collinear : no_three_collinear points)
  (h_len : points.length = 6) :
  ∃ (p1 p2 p3: Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
  triangle_area p1 p2 p3 ≤ 2 :=
sorry

end NUMINAMATH_GPT_exists_small_area_triangle_l578_57801


namespace NUMINAMATH_GPT_expression_c_is_positive_l578_57879

def A : ℝ := 2.1
def B : ℝ := -0.5
def C : ℝ := -3.0
def D : ℝ := 4.2
def E : ℝ := 0.8

theorem expression_c_is_positive : |C| + |B| > 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_expression_c_is_positive_l578_57879


namespace NUMINAMATH_GPT_johns_weekly_allowance_l578_57870

theorem johns_weekly_allowance
    (A : ℝ)
    (h1 : ∃ A, (4/15) * A = 0.64) :
    A = 2.40 :=
by
  sorry

end NUMINAMATH_GPT_johns_weekly_allowance_l578_57870


namespace NUMINAMATH_GPT_edward_candy_purchase_l578_57821

theorem edward_candy_purchase (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) 
  (h1 : whack_a_mole_tickets = 3) (h2 : skee_ball_tickets = 5) (h3 : candy_cost = 4) :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 := 
by 
  sorry

end NUMINAMATH_GPT_edward_candy_purchase_l578_57821


namespace NUMINAMATH_GPT_max_correct_questions_prime_score_l578_57876

-- Definitions and conditions
def total_questions := 20
def points_correct := 5
def points_no_answer := 0
def points_wrong := -2

-- Main statement to prove
theorem max_correct_questions_prime_score :
  ∃ (correct : ℕ) (no_answer wrong : ℕ), 
    correct + no_answer + wrong = total_questions ∧ 
    correct * points_correct + no_answer * points_no_answer + wrong * points_wrong = 83 ∧
    correct = 17 :=
sorry

end NUMINAMATH_GPT_max_correct_questions_prime_score_l578_57876


namespace NUMINAMATH_GPT_range_of_a_l578_57889

variable (f : ℝ → ℝ) (a : ℝ)

-- Definitions based on provided conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotonically_increasing (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x ≤ f y

-- Main statement
theorem range_of_a
    (hf_even : is_even f)
    (hf_mono : is_monotonically_increasing f)
    (h_ineq : ∀ x : ℝ, f (Real.log (a) / Real.log 2) ≤ f (x^2 - 2 * x + 2)) :
  (1/2 : ℝ) ≤ a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_range_of_a_l578_57889


namespace NUMINAMATH_GPT_problem_real_numbers_l578_57898

theorem problem_real_numbers (a b c d r : ℝ) 
  (h1 : b + c + d = r * a) 
  (h2 : a + c + d = r * b) 
  (h3 : a + b + d = r * c) 
  (h4 : a + b + c = r * d) : 
  r = 3 ∨ r = -1 :=
sorry

end NUMINAMATH_GPT_problem_real_numbers_l578_57898


namespace NUMINAMATH_GPT_goose_eggs_count_l578_57832

theorem goose_eggs_count (E : ℕ) 
  (h1 : (1/2 : ℝ) * E = E/2)
  (h2 : (3/4 : ℝ) * (E/2) = (3 * E) / 8)
  (h3 : (2/5 : ℝ) * ((3 * E) / 8) = (3 * E) / 20)
  (h4 : (3 * E) / 20 = 120) :
  E = 400 :=
sorry

end NUMINAMATH_GPT_goose_eggs_count_l578_57832


namespace NUMINAMATH_GPT_minimum_value_l578_57856

variable {x : ℝ}

theorem minimum_value (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l578_57856


namespace NUMINAMATH_GPT_contrapositive_l578_57861

theorem contrapositive (a b : ℕ) : (a = 0 → ab = 0) → (ab ≠ 0 → a ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_l578_57861


namespace NUMINAMATH_GPT_only_nonneg_solution_l578_57871

theorem only_nonneg_solution :
  ∀ (x y : ℕ), 2^x = y^2 + y + 1 → (x, y) = (0, 0) := by
  intros x y h
  sorry

end NUMINAMATH_GPT_only_nonneg_solution_l578_57871


namespace NUMINAMATH_GPT_cupboard_cost_price_l578_57886

noncomputable def cost_price_of_cupboard (C : ℝ) : Prop :=
  let SP := 0.88 * C
  let NSP := 1.12 * C
  NSP - SP = 1650

theorem cupboard_cost_price : ∃ (C : ℝ), cost_price_of_cupboard C ∧ C = 6875 := by
  sorry

end NUMINAMATH_GPT_cupboard_cost_price_l578_57886


namespace NUMINAMATH_GPT_nina_widgets_purchase_l578_57805

theorem nina_widgets_purchase (P : ℝ) (h1 : 8 * (P - 1) = 24) (h2 : 24 / P = 6) : true :=
by
  sorry

end NUMINAMATH_GPT_nina_widgets_purchase_l578_57805


namespace NUMINAMATH_GPT_children_ticket_price_l578_57868

theorem children_ticket_price
  (C : ℝ)
  (adult_ticket_price : ℝ)
  (total_payment : ℝ)
  (total_tickets : ℕ)
  (children_tickets : ℕ)
  (H1 : adult_ticket_price = 8)
  (H2 : total_payment = 201)
  (H3 : total_tickets = 33)
  (H4 : children_tickets = 21)
  : C = 5 :=
by
  sorry

end NUMINAMATH_GPT_children_ticket_price_l578_57868


namespace NUMINAMATH_GPT_shortest_side_of_similar_triangle_l578_57852

theorem shortest_side_of_similar_triangle (a1 a2 h1 h2 : ℝ)
  (h1_eq : a1 = 24)
  (h2_eq : h1 = 37)
  (h2_eq' : h2 = 74)
  (h_similar : h2 / h1 = 2)
  (h_a2_eq : a2 = 2 * Real.sqrt 793):
  a2 = 2 * Real.sqrt 793 := by
  sorry

end NUMINAMATH_GPT_shortest_side_of_similar_triangle_l578_57852


namespace NUMINAMATH_GPT_range_of_b_l578_57888

theorem range_of_b (b : ℝ) :
  (∀ x y : ℝ, (x ≠ y) → (y = 1/3 * x^3 + b * x^2 + (b + 2) * x + 3) → (y ≥ 1/3 * x^3 + b * x^2 + (b + 2) * x + 3))
  ↔ (-1 ≤ b ∧ b ≤ 2) :=
sorry

end NUMINAMATH_GPT_range_of_b_l578_57888


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l578_57822

theorem simplify_and_evaluate_expression (a : ℝ) (h : a^2 + 2 * a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2)) / (a^2 + 2 * a) / (a - 2) = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l578_57822


namespace NUMINAMATH_GPT_find_number_l578_57865

theorem find_number (x : ℝ) (h : 0.8 * x = (4/5 : ℝ) * 25 + 16) : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l578_57865


namespace NUMINAMATH_GPT_sum_abcd_eq_neg_46_div_3_l578_57835

theorem sum_abcd_eq_neg_46_div_3
  (a b c d : ℝ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 15) :
  a + b + c + d = -46 / 3 := 
by sorry

end NUMINAMATH_GPT_sum_abcd_eq_neg_46_div_3_l578_57835


namespace NUMINAMATH_GPT_function_point_proof_l578_57897

-- Given conditions
def condition (f : ℝ → ℝ) : Prop :=
  f 1 = 3

-- Prove the statement
theorem function_point_proof (f : ℝ → ℝ) (h : condition f) : f (-1) + 1 = 4 :=
by
  -- Adding the conditions here
  sorry -- proof is not required

end NUMINAMATH_GPT_function_point_proof_l578_57897


namespace NUMINAMATH_GPT_waiter_customer_count_l578_57818

def initial_customers := 33
def customers_left := 31
def new_customers := 26

theorem waiter_customer_count :
  (initial_customers - customers_left) + new_customers = 28 :=
by
  -- This is a placeholder for the proof that can be filled later.
  sorry

end NUMINAMATH_GPT_waiter_customer_count_l578_57818


namespace NUMINAMATH_GPT_alex_correct_percentage_l578_57802

theorem alex_correct_percentage (y : ℝ) (hy_pos : y > 0) : 
  (5 / 7) * 100 = 71.43 := 
by
  sorry

end NUMINAMATH_GPT_alex_correct_percentage_l578_57802


namespace NUMINAMATH_GPT_jack_travel_total_hours_l578_57828

theorem jack_travel_total_hours :
  (20 + 14 * 24) + (15 + 10 * 24) + (10 + 7 * 24) = 789 := by
  sorry

end NUMINAMATH_GPT_jack_travel_total_hours_l578_57828


namespace NUMINAMATH_GPT_problem_l578_57862

def m (x : ℝ) : ℝ := (x + 2) * (x + 3)
def n (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 9

theorem problem (x : ℝ) : m x < n x :=
by sorry

end NUMINAMATH_GPT_problem_l578_57862


namespace NUMINAMATH_GPT_value_of_x_minus_y_l578_57881

theorem value_of_x_minus_y 
  (x y : ℝ)
  (h1 : x + y = 2)
  (h2 : 3 * x - y = 8) :
  x - y = 3 := by
  sorry

end NUMINAMATH_GPT_value_of_x_minus_y_l578_57881


namespace NUMINAMATH_GPT_car_travel_distance_l578_57836

theorem car_travel_distance 
  (v_train : ℝ) (h_train_speed : v_train = 90) 
  (v_car : ℝ) (h_car_speed : v_car = (2 / 3) * v_train) 
  (t : ℝ) (h_time : t = 0.5) :
  ∃ d : ℝ, d = v_car * t ∧ d = 30 := 
sorry

end NUMINAMATH_GPT_car_travel_distance_l578_57836


namespace NUMINAMATH_GPT_fruit_juice_conversion_needed_l578_57896

theorem fruit_juice_conversion_needed
  (A_milk_parts B_milk_parts A_fruit_juice_parts B_fruit_juice_parts : ℕ)
  (y : ℕ)
  (x : ℕ)
  (convert_liters : ℕ)
  (A_juice_ratio_milk A_juice_ratio_fruit : ℚ)
  (B_juice_ratio_milk B_juice_ratio_fruit : ℚ) :
  (A_milk_parts : ℚ) / (A_milk_parts + A_fruit_juice_parts) = A_juice_ratio_milk →
  (A_fruit_juice_parts : ℚ) / (A_milk_parts + A_fruit_juice_parts) = A_juice_ratio_fruit →
  (B_milk_parts : ℚ) / (B_milk_parts + B_fruit_juice_parts) = B_juice_ratio_milk →
  (B_fruit_juice_parts : ℚ) / (B_milk_parts + B_fruit_juice_parts) = B_juice_ratio_fruit →
  (A_juice_ratio_milk * x = A_juice_ratio_fruit * x + y) →
  y = 14 →
  x = 98 :=
by sorry

end NUMINAMATH_GPT_fruit_juice_conversion_needed_l578_57896


namespace NUMINAMATH_GPT_find_a_if_f_is_odd_l578_57833

noncomputable def f (a x : ℝ) : ℝ := (Real.logb 2 ((a - x) / (1 + x))) 

theorem find_a_if_f_is_odd (a : ℝ) (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 1 := sorry

end NUMINAMATH_GPT_find_a_if_f_is_odd_l578_57833


namespace NUMINAMATH_GPT_remainder_7547_div_11_l578_57877

theorem remainder_7547_div_11 : 7547 % 11 = 10 :=
by
  sorry

end NUMINAMATH_GPT_remainder_7547_div_11_l578_57877


namespace NUMINAMATH_GPT_marching_band_formations_l578_57850

open Nat

theorem marching_band_formations :
  ∃ g, (g = 9) ∧ ∀ s t : ℕ, (s * t = 480 ∧ 15 ≤ t ∧ t ≤ 60) ↔ 
    (t = 15 ∨ t = 16 ∨ t = 20 ∨ t = 24 ∨ t = 30 ∨ t = 32 ∨ t = 40 ∨ t = 48 ∨ t = 60) :=
by
  -- Skipped proof.
  sorry

end NUMINAMATH_GPT_marching_band_formations_l578_57850


namespace NUMINAMATH_GPT_polynomial_integer_roots_a_value_l578_57800

open Polynomial

theorem polynomial_integer_roots_a_value (α β γ : ℤ) (a : ℤ) :
  (X - C α) * (X - C β) * (X - C γ) = X^3 - 2 * X^2 - 25 * X + C a →
  α + β + γ = 2 →
  α * β + α * γ + β * γ = -25 →
  a = -50 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_integer_roots_a_value_l578_57800


namespace NUMINAMATH_GPT_younger_son_age_in_30_years_l578_57883

theorem younger_son_age_in_30_years
  (age_difference : ℕ)
  (elder_son_current_age : ℕ)
  (younger_son_age_in_30_years : ℕ) :
  age_difference = 10 →
  elder_son_current_age = 40 →
  younger_son_age_in_30_years = elder_son_current_age - age_difference + 30 →
  younger_son_age_in_30_years = 60 :=
by
  intros h_diff h_elder h_calc
  sorry

end NUMINAMATH_GPT_younger_son_age_in_30_years_l578_57883


namespace NUMINAMATH_GPT_find_abc_l578_57838

theorem find_abc
  (a b c : ℝ)
  (h1 : a^2 * (b + c) = 2011)
  (h2 : b^2 * (a + c) = 2011)
  (h3 : a ≠ b) : 
  a * b * c = -2011 := 
by 
sorry

end NUMINAMATH_GPT_find_abc_l578_57838


namespace NUMINAMATH_GPT_randy_wipes_days_l578_57839

theorem randy_wipes_days (wipes_per_pack : ℕ) (packs_needed : ℕ) (wipes_per_walk : ℕ) (walks_per_day : ℕ) (total_wipes : ℕ) (wipes_per_day : ℕ) (days_needed : ℕ) 
(h1 : wipes_per_pack = 120)
(h2 : packs_needed = 6)
(h3 : wipes_per_walk = 4)
(h4 : walks_per_day = 2)
(h5 : total_wipes = packs_needed * wipes_per_pack)
(h6 : wipes_per_day = wipes_per_walk * walks_per_day)
(h7 : days_needed = total_wipes / wipes_per_day) : 
days_needed = 90 :=
by sorry

end NUMINAMATH_GPT_randy_wipes_days_l578_57839


namespace NUMINAMATH_GPT_Donovan_Mitchell_goal_average_l578_57895

theorem Donovan_Mitchell_goal_average 
  (current_avg_pg : ℕ)     -- Donovan's current average points per game.
  (played_games : ℕ)       -- Number of games played so far.
  (required_avg_pg : ℕ)    -- Required average points per game in remaining games.
  (total_games : ℕ)        -- Total number of games in the season.
  (goal_avg_pg : ℕ)        -- Goal average points per game for the entire season.
  (H1 : current_avg_pg = 26)
  (H2 : played_games = 15)
  (H3 : required_avg_pg = 42)
  (H4 : total_games = 20) :
  goal_avg_pg = 30 :=
by
  sorry

end NUMINAMATH_GPT_Donovan_Mitchell_goal_average_l578_57895


namespace NUMINAMATH_GPT_problem_solution_l578_57808

theorem problem_solution : 
  (∃ (N : ℕ), (1 + 2 + 3) / 6 = (1988 + 1989 + 1990) / N) → ∃ (N : ℕ), N = 5967 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_solution_l578_57808


namespace NUMINAMATH_GPT_markers_needed_total_l578_57875

noncomputable def markers_needed_first_group : ℕ := 10 * 2
noncomputable def markers_needed_second_group : ℕ := 15 * 4
noncomputable def students_last_group : ℕ := 30 - (10 + 15)
noncomputable def markers_needed_last_group : ℕ := students_last_group * 6

theorem markers_needed_total : markers_needed_first_group + markers_needed_second_group + markers_needed_last_group = 110 :=
by
  sorry

end NUMINAMATH_GPT_markers_needed_total_l578_57875


namespace NUMINAMATH_GPT_find_pairs_l578_57841

theorem find_pairs (n p : ℕ) (hp : Prime p) (hnp : n ≤ 2 * p) (hdiv : (p - 1) * n + 1 % n^(p-1) = 0) :
  (n = 1 ∧ Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
sorry

end NUMINAMATH_GPT_find_pairs_l578_57841


namespace NUMINAMATH_GPT_solve_for_x_l578_57860

theorem solve_for_x (x : ℝ) (h : (3 * x + 15)^2 = 3 * (4 * x + 40)) :
  x = -5 / 3 ∨ x = -7 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l578_57860


namespace NUMINAMATH_GPT_coefficient_of_x3_in_expansion_l578_57893

noncomputable def binomial_expansion_coefficient (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem coefficient_of_x3_in_expansion : 
  (∀ k : ℕ, binomial_expansion_coefficient 6 k ≤ binomial_expansion_coefficient 6 3) →
  binomial_expansion_coefficient 6 3 = 20 :=
by
  intro h
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_coefficient_of_x3_in_expansion_l578_57893


namespace NUMINAMATH_GPT_Shekar_biology_marks_l578_57863

theorem Shekar_biology_marks 
  (math_marks : ℕ := 76) 
  (science_marks : ℕ := 65) 
  (social_studies_marks : ℕ := 82) 
  (english_marks : ℕ := 47) 
  (average_marks : ℕ := 71) 
  (num_subjects : ℕ := 5) 
  (biology_marks : ℕ) :
  (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / num_subjects = average_marks → biology_marks = 85 := 
by 
  sorry

end NUMINAMATH_GPT_Shekar_biology_marks_l578_57863


namespace NUMINAMATH_GPT_total_cookies_baked_l578_57820

def cookies_baked_yesterday : ℕ := 435
def cookies_baked_today : ℕ := 139

theorem total_cookies_baked : cookies_baked_yesterday + cookies_baked_today = 574 := by
  sorry

end NUMINAMATH_GPT_total_cookies_baked_l578_57820


namespace NUMINAMATH_GPT_base_number_eq_2_l578_57885

theorem base_number_eq_2 (x : ℝ) (n : ℕ) (h₁ : x^(2 * n) + x^(2 * n) + x^(2 * n) + x^(2 * n) = 4^28) (h₂ : n = 27) : x = 2 := by
  sorry

end NUMINAMATH_GPT_base_number_eq_2_l578_57885


namespace NUMINAMATH_GPT_probability_of_B_l578_57857

-- Define the events and their probabilities according to the problem description
def A₁ := "Event where a red ball is taken from bag A"
def A₂ := "Event where a white ball is taken from bag A"
def A₃ := "Event where a black ball is taken from bag A"
def B := "Event where a red ball is taken from bag B"

-- Types of bags A and B containing balls
structure Bag where
  red : Nat
  white : Nat
  black : Nat

-- Initial bags
def bagA : Bag := ⟨ 3, 2, 5 ⟩
def bagB : Bag := ⟨ 3, 3, 4 ⟩

-- Probabilities of each event in bagA
def P_A₁ : ℚ := 3 / 10
def P_A₂ : ℚ := 2 / 10
def P_A₃ : ℚ := 5 / 10

-- Probability of event B under conditions A₁, A₂, A₃
def P_B_given_A₁ : ℚ := 4 / 11
def P_B_given_A₂ : ℚ := 3 / 11
def P_B_given_A₃ : ℚ := 3 / 11

-- Goal: Prove that the probability of drawing a red ball from bag B (P(B)) is 3/10
theorem probability_of_B : 
  (P_A₁ * P_B_given_A₁ + P_A₂ * P_B_given_A₂ + P_A₃ * P_B_given_A₃) = (3 / 10) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_probability_of_B_l578_57857


namespace NUMINAMATH_GPT_car_fuel_efficiency_l578_57874

theorem car_fuel_efficiency (distance gallons fuel_efficiency D : ℝ)
  (h₀ : fuel_efficiency = 40)
  (h₁ : gallons = 3.75)
  (h₂ : distance = 150)
  (h_eff : fuel_efficiency = distance / gallons) :
  fuel_efficiency = 40 ∧ (D / fuel_efficiency) = (D / 40) :=
by
  sorry

end NUMINAMATH_GPT_car_fuel_efficiency_l578_57874


namespace NUMINAMATH_GPT_remainder_x_plus_3uy_plus_u_div_y_l578_57848

theorem remainder_x_plus_3uy_plus_u_div_y (x y u v : ℕ) (hx : x = u * y + v) (hu : 0 ≤ v) (hv : v < y) (huv : u + v < y) : 
  (x + 3 * u * y + u) % y = u + v :=
by
  sorry

end NUMINAMATH_GPT_remainder_x_plus_3uy_plus_u_div_y_l578_57848


namespace NUMINAMATH_GPT_percent_women_surveryed_equal_40_l578_57882

theorem percent_women_surveryed_equal_40
  (W M : ℕ) 
  (h1 : W + M = 100)
  (h2 : (W / 100 * 1 / 10 : ℚ) + (M / 100 * 1 / 4 : ℚ) = (19 / 100 : ℚ))
  (h3 : (9 / 10 : ℚ) * (W / 100 : ℚ) + (3 / 4 : ℚ) * (M / 100 : ℚ) = (1 - 19 / 100 : ℚ)) :
  W = 40 := 
sorry

end NUMINAMATH_GPT_percent_women_surveryed_equal_40_l578_57882


namespace NUMINAMATH_GPT_difference_max_min_eq_2log2_minus_1_l578_57867

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem difference_max_min_eq_2log2_minus_1 :
  let M := max (f (1 / 2)) (max (f 1) (f 2))
  let N := min (f (1 / 2)) (min (f 1) (f 2))
  M - N = 2 * Real.log 2 - 1 :=
by
  let M := max (f (1 / 2)) (max (f 1) (f 2))
  let N := min (f (1 / 2)) (min (f 1) (f 2))
  sorry

end NUMINAMATH_GPT_difference_max_min_eq_2log2_minus_1_l578_57867


namespace NUMINAMATH_GPT_multiple_of_A_share_l578_57812

theorem multiple_of_A_share (a b c : ℤ) (hC : c = 84) (hSum : a + b + c = 427)
  (hEquality1 : ∃ x : ℤ, x * a = 4 * b) (hEquality2 : 7 * c = 4 * b) : ∃ x : ℤ, x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_multiple_of_A_share_l578_57812
