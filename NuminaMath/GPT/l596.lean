import Mathlib

namespace lcm_9_16_21_eq_1008_l596_59605

theorem lcm_9_16_21_eq_1008 : Nat.lcm (Nat.lcm 9 16) 21 = 1008 := by
  sorry

end lcm_9_16_21_eq_1008_l596_59605


namespace largest_n_with_100_trailing_zeros_l596_59617

def trailing_zeros_factorial (n : ℕ) : ℕ :=
  if n = 0 then 0 else n / 5 + trailing_zeros_factorial (n / 5)

theorem largest_n_with_100_trailing_zeros :
  ∃ (n : ℕ), trailing_zeros_factorial n = 100 ∧ ∀ (m : ℕ), (trailing_zeros_factorial m = 100 → m ≤ 409) :=
by
  sorry

end largest_n_with_100_trailing_zeros_l596_59617


namespace range_of_a_l596_59618

noncomputable def f (x : ℝ) : ℝ := sorry -- The actual definition of the function f is not given
def g (a x : ℝ) : ℝ := a * x - 1

theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc (-2 : ℝ) 2 → ∃ x₀ : ℝ, x₀ ∈ Set.Icc (-2 : ℝ) 2 ∧ g a x₀ = f x₁) ↔
  a ≤ -1/2 ∨ 5/2 ≤ a :=
by 
  sorry

end range_of_a_l596_59618


namespace inf_geometric_mean_gt_3_inf_geometric_mean_le_2_l596_59668

variables {x y g : ℝ}
variables (hx : 0 < x) (hy : 0 < y)
variable (hg : g = Real.sqrt (x * y))

theorem inf_geometric_mean_gt_3 :
  g ≥ 3 → (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≥ 2 / Real.sqrt (1 + g)) :=
by
  sorry

theorem inf_geometric_mean_le_2 :
  g ≤ 2 → (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≤ 2 / Real.sqrt (1 + g)) :=
by
  sorry

end inf_geometric_mean_gt_3_inf_geometric_mean_le_2_l596_59668


namespace number_of_digits_in_sum_l596_59653

def is_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

theorem number_of_digits_in_sum (C D : ℕ) (hC : is_digit C) (hD : is_digit D) :
  let n1 := 98765
  let n2 := C * 1000 + 433
  let n3 := D * 100 + 22
  let s := n1 + n2 + n3
  100000 ≤ s ∧ s < 1000000 :=
by {
  sorry
}

end number_of_digits_in_sum_l596_59653


namespace shift_line_down_4_units_l596_59682

theorem shift_line_down_4_units :
  ∀ (x : ℝ), y = - (3 / 4) * x → (y - 4 = - (3 / 4) * x - 4) := by
  sorry

end shift_line_down_4_units_l596_59682


namespace totalPizzaEaten_l596_59645

-- Define the conditions
def rachelAte : ℕ := 598
def bellaAte : ℕ := 354

-- State the theorem
theorem totalPizzaEaten : rachelAte + bellaAte = 952 :=
by
  -- Proof omitted
  sorry

end totalPizzaEaten_l596_59645


namespace inequality_proof_l596_59635

theorem inequality_proof (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 2) : 
  (1 / a + 1 / b) ≥ 2 :=
sorry

end inequality_proof_l596_59635


namespace equilibrium_problems_l596_59670

-- Definition of equilibrium constant and catalyst relations

def q1 := False -- Any concentration of substances in equilibrium constant
def q2 := False -- Catalysts changing equilibrium constant
def q3 := False -- No shift if equilibrium constant doesn't change
def q4 := False -- ΔH > 0 if K decreases with increasing temperature
def q5 := True  -- Stoichiometric differences affecting equilibrium constants
def q6 := True  -- Equilibrium shift not necessarily changing equilibrium constant
def q7 := True  -- Extent of reaction indicated by both equilibrium constant and conversion rate

-- The theorem includes our problem statements

theorem equilibrium_problems :
  q1 = False ∧ q2 = False ∧ q3 = False ∧
  q4 = False ∧ q5 = True ∧ q6 = True ∧ q7 = True := by
  sorry

end equilibrium_problems_l596_59670


namespace sam_pam_ratio_is_2_l596_59634

-- Definition of given conditions
def min_assigned_pages : ℕ := 25
def harrison_extra_read : ℕ := 10
def pam_extra_read : ℕ := 15
def sam_read : ℕ := 100

-- Calculations based on the given conditions
def harrison_read : ℕ := min_assigned_pages + harrison_extra_read
def pam_read : ℕ := harrison_read + pam_extra_read

-- Prove the ratio of the number of pages Sam read to the number of pages Pam read is 2
theorem sam_pam_ratio_is_2 : sam_read / pam_read = 2 := 
by
  sorry

end sam_pam_ratio_is_2_l596_59634


namespace coins_to_rubles_l596_59660

theorem coins_to_rubles (a1 a2 a3 a4 a5 a6 a7 k m : ℕ)
  (h1 : a1 + 2 * a2 + 5 * a3 + 10 * a4 + 20 * a5 + 50 * a6 + 100 * a7 = m)
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = k) :
  m * 100 = k :=
by sorry

end coins_to_rubles_l596_59660


namespace least_positive_24x_16y_l596_59693

theorem least_positive_24x_16y (x y : ℤ) : ∃ a : ℕ, a > 0 ∧ a = 24 * x + 16 * y ∧ ∀ b : ℕ, b = 24 * x + 16 * y → b > 0 → b ≥ a :=
sorry

end least_positive_24x_16y_l596_59693


namespace complex_number_on_imaginary_axis_l596_59656

theorem complex_number_on_imaginary_axis (a : ℝ) 
(h : ∃ z : ℂ, z = (a^2 - 2 * a) + (a^2 - a - 2) * Complex.I ∧ z.re = 0) : 
a = 0 ∨ a = 2 :=
by
  sorry

end complex_number_on_imaginary_axis_l596_59656


namespace steve_num_nickels_l596_59694

-- Definitions for the conditions
def num_nickels (N : ℕ) : Prop :=
  ∃ D Q : ℕ, D = N + 4 ∧ Q = D + 3 ∧ 5 * N + 10 * D + 25 * Q + 5 = 380

-- Statement of the problem
theorem steve_num_nickels : num_nickels 4 :=
sorry

end steve_num_nickels_l596_59694


namespace length_of_route_l596_59610

theorem length_of_route 
  (D vA vB : ℝ)
  (h_vA : vA = D / 10)
  (h_vB : vB = D / 6)
  (t : ℝ)
  (h_va_t : vA * t = 75)
  (h_vb_t : vB * t = D - 75) :
  D = 200 :=
by
  sorry

end length_of_route_l596_59610


namespace parallel_lines_slope_l596_59664

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 → 2 * x + (a + 1) * y + 1 = 0) →
  a = -3 :=
by
  sorry

end parallel_lines_slope_l596_59664


namespace sum_of_box_weights_l596_59672

theorem sum_of_box_weights (heavy_box_weight : ℚ) (difference : ℚ) 
  (h1 : heavy_box_weight = 14 / 15) (h2 : difference = 1 / 10) :
  heavy_box_weight + (heavy_box_weight - difference) = 53 / 30 := 
  by
  sorry

end sum_of_box_weights_l596_59672


namespace product_of_largest_two_and_four_digit_primes_l596_59632

theorem product_of_largest_two_and_four_digit_primes :
  let largest_two_digit_prime := 97
  let largest_four_digit_prime := 9973
  largest_two_digit_prime * largest_four_digit_prime = 967781 := by
  sorry

end product_of_largest_two_and_four_digit_primes_l596_59632


namespace at_least_one_gt_one_l596_59646

theorem at_least_one_gt_one (a b : ℝ) (h : a + b > 2) : a > 1 ∨ b > 1 :=
by
  sorry

end at_least_one_gt_one_l596_59646


namespace square_equiv_l596_59600

theorem square_equiv (x : ℝ) : 
  (7 - (x^3 - 49)^(1/3))^2 = 
  49 - 14 * (x^3 - 49)^(1/3) + ((x^3 - 49)^(1/3))^2 := 
by 
  sorry

end square_equiv_l596_59600


namespace suzanna_bike_distance_l596_59613

variable (constant_rate : ℝ) (time_minutes : ℝ) (interval : ℝ) (distance_per_interval : ℝ)

theorem suzanna_bike_distance :
  (constant_rate = 1 / interval) ∧ (interval = 5) ∧ (distance_per_interval = constant_rate * interval) ∧ (time_minutes = 30) →
  ((time_minutes / interval) * distance_per_interval = 6) :=
by
  intros
  sorry

end suzanna_bike_distance_l596_59613


namespace sum_of_numbers_l596_59654

theorem sum_of_numbers :
  36 + 17 + 32 + 54 + 28 + 3 = 170 :=
by
  sorry

end sum_of_numbers_l596_59654


namespace ratio_of_walkway_to_fountain_l596_59680

theorem ratio_of_walkway_to_fountain (n s d : ℝ) (h₀ : n = 10) (h₁ : n^2 * s^2 = 0.40 * (n*s + 2*n*d)^2) : 
  d / s = 1 / 3.44 := 
sorry

end ratio_of_walkway_to_fountain_l596_59680


namespace Clever_not_Green_l596_59630

variables {Lizard : Type}
variables [DecidableEq Lizard] (Clever Green CanJump CanSwim : Lizard → Prop)

theorem Clever_not_Green (h1 : ∀ x, Clever x → CanJump x)
                        (h2 : ∀ x, Green x → ¬ CanSwim x)
                        (h3 : ∀ x, ¬ CanSwim x → ¬ CanJump x) :
  ∀ x, Clever x → ¬ Green x :=
by
  intro x hClever hGreen
  apply h3 x
  apply h2 x hGreen
  exact h1 x hClever

end Clever_not_Green_l596_59630


namespace fraction_product_l596_59667

theorem fraction_product :
  (5 / 8) * (7 / 9) * (11 / 13) * (3 / 5) * (17 / 19) * (8 / 15) = 14280 / 1107000 :=
by sorry

end fraction_product_l596_59667


namespace line_equation_l596_59612

theorem line_equation
  (t : ℝ)
  (x : ℝ) (y : ℝ)
  (h1 : x = 3 * t + 6)
  (h2 : y = 5 * t - 10) :
  y = (5 / 3) * x - 20 :=
sorry

end line_equation_l596_59612


namespace maximum_f_l596_59603

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def f (p : ℝ) : ℝ :=
  binomial_coefficient 20 2 * p^2 * (1 - p)^18

theorem maximum_f :
  ∃ p_0 : ℝ, 0 < p_0 ∧ p_0 < 1 ∧ f p = f (0.1) := sorry

end maximum_f_l596_59603


namespace calculate_value_l596_59671

theorem calculate_value : (2 / 3 : ℝ)^0 + Real.log 2 + Real.log 5 = 2 :=
by 
  sorry

end calculate_value_l596_59671


namespace no_3_digit_even_sum_27_l596_59601

/-- Predicate for a 3-digit number -/
def is_3_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Predicate for an even number -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Function to compute the digit sum of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Theorem: There are no 3-digit numbers with a digit sum of 27 that are even -/
theorem no_3_digit_even_sum_27 : 
  ∀ n : ℕ, is_3_digit n → digit_sum n = 27 → is_even n → false :=
by
  sorry

end no_3_digit_even_sum_27_l596_59601


namespace exact_fraction_difference_l596_59679

theorem exact_fraction_difference :
  let x := (8:ℚ) / 11
  let y := (18:ℚ) / 25 
  x - y = (2:ℚ) / 275 :=
by
  -- Definitions from conditions: x = 0.\overline{72} and y = 0.72
  let x := (8:ℚ) / 11
  let y := (18:ℚ) / 25 
  -- Goal is to prove the exact fraction difference
  show x - y = (2:ℚ) / 275
  sorry

end exact_fraction_difference_l596_59679


namespace complex_power_of_sum_l596_59689

theorem complex_power_of_sum (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end complex_power_of_sum_l596_59689


namespace greatest_value_of_sum_l596_59639

theorem greatest_value_of_sum (x : ℝ) (h : 13 = x^2 + (1/x)^2) : x + 1/x ≤ Real.sqrt 15 :=
sorry

end greatest_value_of_sum_l596_59639


namespace money_collected_l596_59619

theorem money_collected
  (households_per_day : ℕ)
  (days : ℕ)
  (half_give_money : ℕ → ℕ)
  (total_money_collected : ℕ)
  (households_give_money : ℕ) :
  households_per_day = 20 →  
  days = 5 →
  total_money_collected = 2000 →
  half_give_money (households_per_day * days) = (households_per_day * days) / 2 →
  households_give_money = (households_per_day * days) / 2 →
  total_money_collected / households_give_money = 40
:= sorry

end money_collected_l596_59619


namespace possible_values_count_l596_59650

theorem possible_values_count {x y z : ℤ} (h₁ : x = 5) (h₂ : y = -3) (h₃ : z = -1) :
  ∃ v, v = x - y - z ∧ (v = 7 ∨ v = 8 ∨ v = 9) :=
by
  sorry

end possible_values_count_l596_59650


namespace largest_three_digit_number_divisible_by_six_l596_59629

theorem largest_three_digit_number_divisible_by_six : ∃ n : ℕ, (∃ m < 1000, m ≥ 100 ∧ m % 6 = 0 ∧ m = n) ∧ (∀ k < 1000, k ≥ 100 ∧ k % 6 = 0 → k ≤ n) ∧ n = 996 :=
by sorry

end largest_three_digit_number_divisible_by_six_l596_59629


namespace length_of_side_b_max_area_of_triangle_l596_59631

variable {A B C a b c : ℝ}
variable {triangle_ABC : a + c = 6}
variable {eq1 : (3 - Real.cos A) * Real.sin B = Real.sin A * (1 + Real.cos B)}

-- Theorem for part (1) length of side b
theorem length_of_side_b :
  b = 2 :=
sorry

-- Theorem for part (2) maximum area of the triangle
theorem max_area_of_triangle :
  ∃ (S : ℝ), S = 2 * Real.sqrt 2 :=
sorry

end length_of_side_b_max_area_of_triangle_l596_59631


namespace sum_of_first_four_terms_of_sequence_l596_59651

-- Define the sequence, its common difference, and the given initial condition
def a_sequence (a : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, a (n + 1) - a n = 2) ∧ (a 2 = 5)

-- Define the sum of the first four terms
def sum_first_four_terms (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_of_first_four_terms_of_sequence :
  ∀ (a : ℕ → ℤ), a_sequence a → sum_first_four_terms a = 24 :=
by
  intro a h
  rw [a_sequence] at h
  obtain ⟨h_diff, h_a2⟩ := h
  sorry

end sum_of_first_four_terms_of_sequence_l596_59651


namespace fraction_meaningful_range_l596_59649

-- Define the condition
def meaningful_fraction_condition (x : ℝ) : Prop := (x - 2023) ≠ 0

-- Define the conclusion that we need to prove
def meaningful_fraction_range (x : ℝ) : Prop := x ≠ 2023

theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction_condition x → meaningful_fraction_range x :=
by
  intro h
  -- Proof steps would go here
  sorry

end fraction_meaningful_range_l596_59649


namespace cone_sphere_ratio_l596_59621

-- Defining the conditions and proof goals
theorem cone_sphere_ratio (r h : ℝ) (h_cone_sphere_radius : r ≠ 0) 
  (h_cone_volume : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 :=
by
  -- All the assumptions / conditions given in the problem
  sorry -- Proof omitted

end cone_sphere_ratio_l596_59621


namespace maximize_a_minus_b_plus_c_l596_59676

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x)

theorem maximize_a_minus_b_plus_c
  {a b c : ℝ}
  (h : ∀ x : ℝ, f a b c x ≥ -1) :
  a - b + c ≤ 1 :=
sorry

end maximize_a_minus_b_plus_c_l596_59676


namespace min_mod_z_l596_59636

open Complex

theorem min_mod_z (z : ℂ) (hz : abs (z - 2 * I) + abs (z - 5) = 7) : abs z = 10 / 7 :=
sorry

end min_mod_z_l596_59636


namespace polynomial_equivalence_l596_59674

variable (x : ℝ) -- Define variable x

-- Define the expressions.
def expr1 := (3 * x^2 + 5 * x + 8) * (x + 2)
def expr2 := (x + 2) * (x^2 + 5 * x - 72)
def expr3 := (4 * x - 15) * (x + 2) * (x + 6)

-- Define the expression to be proved.
def original_expr := expr1 - expr2 + expr3
def simplified_expr := 6 * x^3 + 21 * x^2 + 18 * x

-- The theorem to prove the equivalence of the original and simplified expressions.
theorem polynomial_equivalence : original_expr = simplified_expr :=
by sorry -- proof to be filled in

end polynomial_equivalence_l596_59674


namespace original_triangle_area_l596_59673

theorem original_triangle_area (A_orig A_new : ℝ) (h1 : A_new = 256) (h2 : A_new = 16 * A_orig) : A_orig = 16 :=
by
  sorry

end original_triangle_area_l596_59673


namespace janice_total_earnings_l596_59675

-- Defining the working conditions as constants
def days_per_week : ℕ := 5  -- Janice works 5 days a week
def earning_per_day : ℕ := 30  -- Janice earns $30 per day
def overtime_earning_per_shift : ℕ := 15  -- Janice earns $15 per overtime shift
def overtime_shifts : ℕ := 3  -- Janice works three overtime shifts

-- Defining Janice's total earnings for the week
def total_earnings : ℕ := (days_per_week * earning_per_day) + (overtime_shifts * overtime_earning_per_shift)

-- Statement to prove that Janice's total earnings are $195
theorem janice_total_earnings : total_earnings = 195 :=
by
  -- The proof is omitted.
  sorry

end janice_total_earnings_l596_59675


namespace initial_treasure_amount_l596_59611

theorem initial_treasure_amount 
  (T : ℚ)
  (h₁ : T * (1 - 1/13) * (1 - 1/17) = 150) : 
  T = 172 + 21/32 :=
sorry

end initial_treasure_amount_l596_59611


namespace arc_length_l596_59681

-- Define the radius and central angle
def radius : ℝ := 10
def central_angle : ℝ := 240

-- Theorem to prove the arc length is (40 * π) / 3
theorem arc_length (r : ℝ) (n : ℝ) (h_r : r = radius) (h_n : n = central_angle) : 
  (n * π * r) / 180 = (40 * π) / 3 :=
by
  -- Proof omitted
  sorry

end arc_length_l596_59681


namespace inequality_solution_l596_59698

theorem inequality_solution (x : ℝ) :
  (x + 2) / (x^2 + 3 * x + 10) ≥ 0 ↔ x ≥ -2 := sorry

end inequality_solution_l596_59698


namespace divisible_by_square_of_k_l596_59648

theorem divisible_by_square_of_k (a b l : ℕ) (k : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : a % 2 = 1) (h4 : b % 2 = 1) (h5 : a + b = 2 ^ l) : k = 1 ↔ k^2 ∣ a^k + b^k := 
sorry

end divisible_by_square_of_k_l596_59648


namespace parabola_focus_directrix_distance_l596_59652

theorem parabola_focus_directrix_distance 
  (p : ℝ) 
  (hp : 3 = p * (1:ℝ)^2) 
  (hparabola : ∀ x : ℝ, y = p * x^2 → x^2 = (1/3:ℝ) * y)
  : (distance_focus_directrix : ℝ) = (1 / 6:ℝ) :=
  sorry

end parabola_focus_directrix_distance_l596_59652


namespace odd_function_evaluation_l596_59637

theorem odd_function_evaluation
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, x ≤ 0 → f x = 2 * x^2 - x) :
  f 1 = -3 :=
by {
  sorry
}

end odd_function_evaluation_l596_59637


namespace tangent_parallel_l596_59604

noncomputable def f (x : ℝ) := x^4 - x

theorem tangent_parallel (P : ℝ × ℝ) (hP : P = (1, 0)) :
  (∃ x y : ℝ, P = (x, y) ∧ (fderiv ℝ f x) 1 = 3 / 1) ↔ P = (1, 0) :=
by
  sorry

end tangent_parallel_l596_59604


namespace problem_solution_l596_59627

theorem problem_solution (x : ℝ) (h1 : x = 12) (h2 : 5 + 7 / x = some_number - 5 / x) : some_number = 6 := 
by
  sorry

end problem_solution_l596_59627


namespace correct_operation_l596_59659

variable {x y : ℝ}

theorem correct_operation :
  (2 * x^2 + 4 * x^2 = 6 * x^2) → 
  (x * x^3 = x^4) → 
  ((x^3)^2 = x^6) →
  ((xy)^5 = x^5 * y^5) →
  ((x^3)^2 = x^6) := 
by 
  intros h1 h2 h3 h4
  exact h3

end correct_operation_l596_59659


namespace find_all_possible_f_l596_59695

-- Noncomputability is needed here since we cannot construct a function 
-- like f deterministically via computation due to the nature of the problem.
noncomputable def functional_equation_solution (f : ℕ → ℕ) := 
  (∀ a b : ℕ, f a + f b ∣ 2 * (a + b - 1)) → 
  (∀ x : ℕ, f x = 1) ∨ (∀ x : ℕ, f x = 2 * x - 1)

-- Statement of the mathematically equivalent proof problem.
theorem find_all_possible_f (f : ℕ → ℕ) : functional_equation_solution f := 
sorry

end find_all_possible_f_l596_59695


namespace license_plate_count_l596_59687

-- Define the conditions as constants
def even_digit_count : Nat := 5
def consonant_count : Nat := 20
def vowel_count : Nat := 6

-- Define the problem as a theorem to prove
theorem license_plate_count : even_digit_count * consonant_count * vowel_count * consonant_count = 12000 := 
by
  -- The proof is not required, so we leave it as sorry
  sorry

end license_plate_count_l596_59687


namespace range_of_t_l596_59609

theorem range_of_t (a b : ℝ) (h : a^2 + a*b + b^2 = 1) :
  ∃ t : ℝ, (t = a^2 - a*b + b^2) ∧ (1/3 ≤ t ∧ t ≤ 3) :=
sorry

end range_of_t_l596_59609


namespace power_boat_travel_time_l596_59690

theorem power_boat_travel_time {r p t : ℝ} (h1 : r > 0) (h2 : p > 0) 
  (h3 : (p + r) * t + (p - r) * (9 - t) = 9 * r) : t = 4.5 :=
by
  sorry

end power_boat_travel_time_l596_59690


namespace simplify_polynomials_l596_59683

theorem simplify_polynomials :
  (4 * q ^ 4 + 2 * p ^ 3 - 7 * p + 8) + (3 * q ^ 4 - 2 * p ^ 3 + 3 * p ^ 2 - 5 * p + 6) =
  7 * q ^ 4 + 3 * p ^ 2 - 12 * p + 14 :=
by
  sorry

end simplify_polynomials_l596_59683


namespace equivalent_problem_l596_59691

variable {x y : Real}

theorem equivalent_problem 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 15) :
  (x - y)^2 = 21 ∧ (x + y) * (x - y) = Real.sqrt 1701 :=
by
  sorry

end equivalent_problem_l596_59691


namespace required_percentage_to_pass_l596_59678

-- Definitions based on conditions
def obtained_marks : ℕ := 175
def failed_by : ℕ := 56
def max_marks : ℕ := 700
def pass_marks : ℕ := obtained_marks + failed_by

-- Theorem stating the required percentage to pass
theorem required_percentage_to_pass : 
  (pass_marks : ℚ) / max_marks * 100 = 33 := 
by 
  sorry

end required_percentage_to_pass_l596_59678


namespace num_digits_expr_l596_59608

noncomputable def num_digits (n : ℕ) : ℕ :=
  (Int.ofNat n).natAbs.digits 10 |>.length

def expr : ℕ := 2^15 * 5^10 * 12

theorem num_digits_expr : num_digits expr = 13 := by
  sorry

end num_digits_expr_l596_59608


namespace point_in_third_quadrant_l596_59606

section quadrant_problem

variables (a b : ℝ)

-- Given: Point (a, b) is in the fourth quadrant
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a > 0 ∧ b < 0

-- To prove: Point (a / b, 2 * b - a) is in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- The theorem stating that if (a, b) is in the fourth quadrant,
-- then (a / b, 2 * b - a) is in the third quadrant
theorem point_in_third_quadrant (a b : ℝ) (h : in_fourth_quadrant a b) :
  in_third_quadrant (a / b) (2 * b - a) :=
  sorry

end quadrant_problem

end point_in_third_quadrant_l596_59606


namespace factorials_sum_of_two_squares_l596_59624

-- Define what it means for a number to be a sum of two squares.
def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

theorem factorials_sum_of_two_squares :
  {n : ℕ | n < 14 ∧ is_sum_of_two_squares (n!)} = {2, 6} :=
by
  sorry

end factorials_sum_of_two_squares_l596_59624


namespace range_of_m_l596_59688

theorem range_of_m (m : ℝ) : (-6 < m ∧ m < 2) ↔ ∃ x : ℝ, |x - m| + |x + 2| < 4 :=
by sorry

end range_of_m_l596_59688


namespace shifting_parabola_l596_59642

def original_function (x : ℝ) : ℝ := (x + 1)^2 + 3

def shifted_function (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem shifting_parabola : ∀ x : ℝ, shifted_function x = original_function (x + 2) - 1 := 
by 
  sorry

end shifting_parabola_l596_59642


namespace sin_fourth_plus_cos_fourth_l596_59614

theorem sin_fourth_plus_cos_fourth (α : ℝ) (h : Real.cos (2 * α) = 3 / 5) : 
  Real.sin α ^ 4 + Real.cos α ^ 4 = 17 / 25 := 
by
  sorry

end sin_fourth_plus_cos_fourth_l596_59614


namespace seq_sum_difference_l596_59620

-- Define the sequences
def seq1 : List ℕ := List.range 93 |> List.map (λ n => 2001 + n)
def seq2 : List ℕ := List.range 93 |> List.map (λ n => 301 + n)

-- Define the sum of the sequences
def sum_seq1 : ℕ := seq1.sum
def sum_seq2 : ℕ := seq2.sum

-- Define the difference between the sums of the sequences
def diff_seq_sum : ℕ := sum_seq1 - sum_seq2

-- Lean statement to prove the difference equals 158100
theorem seq_sum_difference : diff_seq_sum = 158100 := by
  sorry

end seq_sum_difference_l596_59620


namespace equality_of_areas_l596_59641

theorem equality_of_areas (d : ℝ) :
  (∀ d : ℝ, (1/2) * d * 3 = 9 / 2 → d = 3) ↔ d = 3 :=
by
  sorry

end equality_of_areas_l596_59641


namespace initial_number_of_women_l596_59640

variable (W : ℕ)

def work_done_by_women_per_day (W : ℕ) : ℚ := 1 / (8 * W)
def work_done_by_children_per_day (W : ℕ) : ℚ := 1 / (12 * W)

theorem initial_number_of_women :
  (6 * work_done_by_women_per_day W + 3 * work_done_by_children_per_day W = 1 / 10) → W = 10 :=
by
  sorry

end initial_number_of_women_l596_59640


namespace ap_number_of_terms_is_six_l596_59633

noncomputable def arithmetic_progression_number_of_terms (a d : ℕ) (n : ℕ) : Prop :=
  let odd_sum := (n / 2) * (2 * a + (n - 2) * d)
  let even_sum := (n / 2) * (2 * a + n * d)
  let last_term_condition := (n - 1) * d = 15
  n % 2 = 0 ∧ odd_sum = 30 ∧ even_sum = 36 ∧ last_term_condition

theorem ap_number_of_terms_is_six (a d n : ℕ) (h : arithmetic_progression_number_of_terms a d n) :
  n = 6 :=
by sorry

end ap_number_of_terms_is_six_l596_59633


namespace harmonic_mean_closest_to_one_l596_59684

-- Define the given conditions a = 1/4 and b = 2048
def a : ℚ := 1 / 4
def b : ℚ := 2048

-- Define the harmonic mean of two numbers
def harmonic_mean (x y : ℚ) : ℚ := 2 * x * y / (x + y)

-- State the theorem proving the harmonic mean is closest to 1
theorem harmonic_mean_closest_to_one : abs (harmonic_mean a b - 1) < 1 :=
sorry

end harmonic_mean_closest_to_one_l596_59684


namespace pencils_per_row_l596_59657

-- Definitions of conditions.
def num_pencils : ℕ := 35
def num_rows : ℕ := 7

-- Hypothesis: given the conditions, prove the number of pencils per row.
theorem pencils_per_row : num_pencils / num_rows = 5 := 
  by 
  -- Proof steps go here, but are replaced by sorry.
  sorry

end pencils_per_row_l596_59657


namespace liza_phone_bill_eq_70_l596_59696

theorem liza_phone_bill_eq_70 (initial_balance rent payment paycheck electricity internet final_balance phone_bill : ℝ)
  (h1 : initial_balance = 800)
  (h2 : rent = 450)
  (h3 : paycheck = 1500)
  (h4 : electricity = 117)
  (h5 : internet = 100)
  (h6 : final_balance = 1563)
  (h_balance_before_phone_bill : initial_balance - rent + paycheck - (electricity + internet) = 1633)
  (h_final_balance_def : 1633 - phone_bill = final_balance) :
  phone_bill = 70 := sorry

end liza_phone_bill_eq_70_l596_59696


namespace highest_student_id_in_sample_l596_59643

theorem highest_student_id_in_sample
    (total_students : ℕ)
    (sample_size : ℕ)
    (included_student_id : ℕ)
    (interval : ℕ)
    (first_id in_sample : ℕ)
    (k : ℕ)
    (highest_id : ℕ)
    (total_students_eq : total_students = 63)
    (sample_size_eq : sample_size = 7)
    (included_student_id_eq : included_student_id = 11)
    (k_def : k = total_students / sample_size)
    (included_student_id_in_second_pos : included_student_id = first_id + k)
    (interval_eq : interval = first_id - k)
    (in_sample_eq : in_sample = interval)
    (highest_id_eq : highest_id = in_sample + k * (sample_size - 1)) :
  highest_id = 56 := sorry

end highest_student_id_in_sample_l596_59643


namespace find_a_b_k_l596_59616

noncomputable def a (k : ℕ) : ℕ := if h : k = 9 then 243 else sorry
noncomputable def b (k : ℕ) : ℕ := if h : k = 9 then 3 else sorry

theorem find_a_b_k (a b k : ℕ) (hb : b = 3) (ha : a = 243) (hk : k = 9)
  (h1 : a * b = k^3) (h2 : a / b = k^2) (h3 : 100 ≤ a * b ∧ a * b < 1000) :
  a = 243 ∧ b = 3 ∧ k = 9 :=
by 
  sorry

end find_a_b_k_l596_59616


namespace circle_radius_k_l596_59638

theorem circle_radius_k (k : ℝ) : (∃ x y : ℝ, (x^2 + 14*x + y^2 + 8*y - k = 0) ∧ ((x + 7)^2 + (y + 4)^2 = 100)) → k = 35 :=
by
  sorry

end circle_radius_k_l596_59638


namespace miles_to_friends_house_l596_59622

-- Define the conditions as constants
def miles_per_gallon : ℕ := 19
def gallons : ℕ := 2
def miles_to_school : ℕ := 15
def miles_to_softball_park : ℕ := 6
def miles_to_burger_restaurant : ℕ := 2
def miles_home : ℕ := 11

-- Define the total miles driven
def total_miles_driven (miles_to_friend : ℕ) :=
  miles_to_school + miles_to_softball_park + miles_to_burger_restaurant + miles_to_friend + miles_home

-- Define the total miles possible with given gallons of gas
def total_miles_possible : ℕ :=
  miles_per_gallon * gallons

-- Prove that the miles driven to the friend's house is 4
theorem miles_to_friends_house : 
  ∃ miles_to_friend, total_miles_driven miles_to_friend = total_miles_possible ∧ miles_to_friend = 4 :=
by
  sorry

end miles_to_friends_house_l596_59622


namespace tens_digit_2015_pow_2016_minus_2017_l596_59699

theorem tens_digit_2015_pow_2016_minus_2017 :
  (2015^2016 - 2017) % 100 = 8 := 
sorry

end tens_digit_2015_pow_2016_minus_2017_l596_59699


namespace polynomial_has_three_real_roots_l596_59625

theorem polynomial_has_three_real_roots (a b c : ℝ) (h1 : b < 0) (h2 : a * b = 9 * c) :
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (x1^3 + a * x1^2 + b * x1 + c = 0) ∧ 
    (x2^3 + a * x2^2 + b * x2 + c = 0) ∧ 
    (x3^3 + a * x3^2 + b * x3 + c = 0) := sorry

end polynomial_has_three_real_roots_l596_59625


namespace sum_of_common_ratios_l596_59658

variable {k p r : ℝ}

theorem sum_of_common_ratios (h1 : k ≠ 0)
                             (h2 : p ≠ r)
                             (h3 : k * p^2 - k * r^2 = 5 * (k * p - k * r)) :
                             p + r = 5 := 
by
  sorry

end sum_of_common_ratios_l596_59658


namespace tank_capacity_l596_59697

theorem tank_capacity (C : ℝ) (h1 : 1/4 * C + 180 = 3/4 * C) : C = 360 :=
sorry

end tank_capacity_l596_59697


namespace fixed_monthly_charge_l596_59692

-- Given conditions
variable (F C_J : ℕ)
axiom january_bill : F + C_J = 46
axiom february_bill : F + 2 * C_J = 76

-- Proof problem
theorem fixed_monthly_charge : F = 16 :=
by
  sorry

end fixed_monthly_charge_l596_59692


namespace perfect_square_trinomial_l596_59607

theorem perfect_square_trinomial (c : ℝ) : (∃ a : ℝ, (x : ℝ) → x^2 + 150 * x + c = (x + a)^2) → c = 5625 :=
sorry

end perfect_square_trinomial_l596_59607


namespace find_t_l596_59628

variables (V V₀ g a S t : ℝ)

-- Conditions
axiom eq1 : V = 3 * g * t + V₀
axiom eq2 : S = (3 / 2) * g * t^2 + V₀ * t + (1 / 2) * a * t^2

-- Theorem to prove
theorem find_t : t = (9 * g * S) / (2 * (V - V₀)^2 + 3 * V₀ * (V - V₀)) :=
by
  sorry

end find_t_l596_59628


namespace queenie_total_earnings_l596_59665

-- Define the conditions
def daily_wage : ℕ := 150
def overtime_wage_per_hour : ℕ := 5
def days_worked : ℕ := 5
def overtime_hours : ℕ := 4

-- Define the main problem
theorem queenie_total_earnings : 
  (daily_wage * days_worked + overtime_wage_per_hour * overtime_hours) = 770 :=
by
  sorry

end queenie_total_earnings_l596_59665


namespace quadratic_equation_equivalence_l596_59602

theorem quadratic_equation_equivalence
  (a_0 a_1 a_2 : ℝ)
  (r s : ℝ)
  (h_roots : a_0 + a_1 * r + a_2 * r^2 = 0 ∧ a_0 + a_1 * s + a_2 * s^2 = 0)
  (h_a2_nonzero : a_2 ≠ 0) :
  (∀ x, a_0 ≠ 0 ↔ a_0 + a_1 * x + a_2 * x^2 = a_0 * (1 - x / r) * (1 - x / s)) :=
sorry

end quadratic_equation_equivalence_l596_59602


namespace find_m_l596_59644

theorem find_m (x m : ℝ) (h_eq : (x + m) / (x - 2) + 1 / (2 - x) = 3) (h_root : x = 2) : m = -1 :=
by
  sorry

end find_m_l596_59644


namespace find_positives_xyz_l596_59677

theorem find_positives_xyz (x y z : ℕ) (h : x > 0 ∧ y > 0 ∧ z > 0)
    (heq : (1 : ℚ)/x + (1 : ℚ)/y + (1 : ℚ)/z = 4 / 5) :
    (x = 2 ∧ y = 4 ∧ z = 20) ∨ (x = 2 ∧ y = 5 ∧ z = 10) :=
by
  sorry

-- This theorem states that there are only two sets of positive integers (x, y, z)
-- that satisfy the equation (1/x) + (1/y) + (1/z) = 4/5, specifically:
-- (2, 4, 20) and (2, 5, 10).

end find_positives_xyz_l596_59677


namespace multiple_of_669_l596_59655

theorem multiple_of_669 (k : ℕ) (h : ∃ a : ℤ, 2007 ∣ (a + k : ℤ)^3 - a^3) : 669 ∣ k :=
sorry

end multiple_of_669_l596_59655


namespace find_x_when_y_3_l596_59669

variable (y x k : ℝ)

axiom h₁ : x = k / (y ^ 2)
axiom h₂ : y = 9 → x = 0.1111111111111111
axiom y_eq_3 : y = 3

theorem find_x_when_y_3 : y = 3 → x = 1 :=
by
  sorry

end find_x_when_y_3_l596_59669


namespace smallest_x_fraction_floor_l596_59663

theorem smallest_x_fraction_floor (x : ℝ) (h : ⌊x⌋ / x = 7 / 8) : x = 48 / 7 :=
sorry

end smallest_x_fraction_floor_l596_59663


namespace vector_c_expression_l596_59662

-- Define the vectors a, b, c
def vector_a : ℤ × ℤ := (1, 2)
def vector_b : ℤ × ℤ := (-1, 1)
def vector_c : ℤ × ℤ := (1, 5)

-- Define the addition of vectors in ℤ × ℤ
def vec_add (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define the scalar multiplication of vectors in ℤ × ℤ
def scalar_mul (k : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (k * v.1, k * v.2)

-- Given the conditions
def condition1 := vector_a = (1, 2)
def condition2 := vec_add vector_a vector_b = (0, 3)

-- The goal is to prove that vector_c = 2 * vector_a + vector_b
theorem vector_c_expression : vec_add (scalar_mul 2 vector_a) vector_b = vector_c := by
  sorry

end vector_c_expression_l596_59662


namespace seashells_remainder_l596_59623

theorem seashells_remainder :
  let derek := 58
  let emily := 73
  let fiona := 31 
  let total_seashells := derek + emily + fiona
  total_seashells % 10 = 2 :=
by
  sorry

end seashells_remainder_l596_59623


namespace greatest_y_value_l596_59686

theorem greatest_y_value (x y : ℤ) (h : x * y + 7 * x + 2 * y = -8) : y ≤ -1 :=
by
  sorry

end greatest_y_value_l596_59686


namespace calculate_truck_loads_of_dirt_l596_59661

noncomputable def truck_loads_sand: ℚ := 0.16666666666666666
noncomputable def truck_loads_cement: ℚ := 0.16666666666666666
noncomputable def total_truck_loads_material: ℚ := 0.6666666666666666
noncomputable def truck_loads_dirt: ℚ := total_truck_loads_material - (truck_loads_sand + truck_loads_cement)

theorem calculate_truck_loads_of_dirt :
  truck_loads_dirt = 0.3333333333333333 := 
by
  sorry

end calculate_truck_loads_of_dirt_l596_59661


namespace sufficient_but_not_necessary_l596_59615

theorem sufficient_but_not_necessary (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) : 
  (a > 1 ∧ b > 1 → a * b > 1) ∧ ¬(a * b > 1 → a > 1 ∧ b > 1) :=
by
  sorry

end sufficient_but_not_necessary_l596_59615


namespace count_integers_within_range_l596_59685

theorem count_integers_within_range : 
  ∃ (count : ℕ), count = 57 ∧ ∀ n : ℤ, -5.5 * Real.pi ≤ n ∧ n ≤ 12.5 * Real.pi → n ≥ -17 ∧ n ≤ 39 :=
by
  sorry

end count_integers_within_range_l596_59685


namespace relationship_between_products_l596_59666

variable {a₁ a₂ b₁ b₂ : ℝ}

theorem relationship_between_products (h₁ : a₁ < a₂) (h₂ : b₁ < b₂) : a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := 
sorry

end relationship_between_products_l596_59666


namespace eleven_million_scientific_notation_l596_59626

-- Definition of the scientific notation condition and question
def scientific_notation (a n : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ ∃ k : ℤ, n = 10 ^ k

-- The main theorem stating that 11 million can be expressed as 1.1 * 10^7
theorem eleven_million_scientific_notation : scientific_notation 1.1 (10 ^ 7) :=
by 
  -- Adding sorry to skip the proof
  sorry

end eleven_million_scientific_notation_l596_59626


namespace linear_equation_m_equals_neg_3_l596_59647

theorem linear_equation_m_equals_neg_3 
  (m : ℤ)
  (h1 : |m| - 2 = 1)
  (h2 : m - 3 ≠ 0) :
  m = -3 :=
sorry

end linear_equation_m_equals_neg_3_l596_59647
