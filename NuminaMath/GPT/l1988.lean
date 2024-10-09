import Mathlib

namespace algebraic_identity_l1988_198847

theorem algebraic_identity 
  (p q r a b c : ℝ)
  (h₁ : p + q + r = 1)
  (h₂ : 1 / p + 1 / q + 1 / r = 0) :
  a^2 + b^2 + c^2 = (p * a + q * b + r * c)^2 + (q * a + r * b + p * c)^2 + (r * a + p * b + q * c)^2 := by
  sorry

end algebraic_identity_l1988_198847


namespace tank_capacity_l1988_198801

noncomputable def leak_rate (C : ℝ) := C / 6
noncomputable def inlet_rate := 240
noncomputable def net_emptying_rate (C : ℝ) := C / 8

theorem tank_capacity : ∀ (C : ℝ), 
  (inlet_rate - leak_rate C = net_emptying_rate C) → 
  C = 5760 / 7 :=
by 
  sorry

end tank_capacity_l1988_198801


namespace factory_output_l1988_198887

theorem factory_output :
  ∀ (J M : ℝ), M = J * 0.8 → J = M * 1.25 :=
by
  intros J M h
  sorry

end factory_output_l1988_198887


namespace number_of_boys_at_reunion_l1988_198839

theorem number_of_boys_at_reunion (n : ℕ) (h : n * (n - 1) / 2 = 66) : n = 12 :=
sorry

end number_of_boys_at_reunion_l1988_198839


namespace tan_five_pi_over_four_l1988_198899

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end tan_five_pi_over_four_l1988_198899


namespace connie_needs_more_money_l1988_198849

variable (cost_connie : ℕ) (cost_watch : ℕ)

theorem connie_needs_more_money 
  (h_connie : cost_connie = 39)
  (h_watch : cost_watch = 55) :
  cost_watch - cost_connie = 16 :=
by sorry

end connie_needs_more_money_l1988_198849


namespace abs_x_plus_1_plus_abs_x_minus_3_ge_a_l1988_198823

theorem abs_x_plus_1_plus_abs_x_minus_3_ge_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) ↔ a ≤ 4 :=
by
  sorry

end abs_x_plus_1_plus_abs_x_minus_3_ge_a_l1988_198823


namespace range_of_m_l1988_198821

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x + α / x + Real.log x

theorem range_of_m (e l : ℝ) (alpha : ℝ) :
  (∀ (α : ℝ), α ∈ Set.Icc (1 / Real.exp 1) (2 * Real.exp 1 ^ 2) → 
  ∀ (x : ℝ), x ∈ Set.Icc l e → f alpha x < m) →
  m ∈ Set.Ioi (1 + 2 * Real.exp 1 ^ 2) := sorry

end range_of_m_l1988_198821


namespace floor_eq_48_iff_l1988_198845

-- Define the real number set I to be [8, 49/6)
def I : Set ℝ := { x | 8 ≤ x ∧ x < 49/6 }

-- The main statement to be proven
theorem floor_eq_48_iff (x : ℝ) : (Int.floor (x * Int.floor x) = 48) ↔ x ∈ I := 
by
  sorry

end floor_eq_48_iff_l1988_198845


namespace resistance_parallel_l1988_198822

theorem resistance_parallel (x y r : ℝ) (hy : y = 6) (hr : r = 2.4) 
  (h : 1 / r = 1 / x + 1 / y) : x = 4 :=
  sorry

end resistance_parallel_l1988_198822


namespace meiosis_fertilization_correct_l1988_198890

theorem meiosis_fertilization_correct :
  (∀ (half_nuclear_sperm half_nuclear_egg mitochondrial_egg : Prop)
     (recognition_basis_clycoproteins : Prop)
     (fusion_basis_nuclei : Prop)
     (meiosis_eukaryotes : Prop)
     (random_fertilization : Prop),
    (half_nuclear_sperm ∧ half_nuclear_egg ∧ mitochondrial_egg ∧ recognition_basis_clycoproteins ∧ fusion_basis_nuclei ∧ meiosis_eukaryotes ∧ random_fertilization) →
    (D : Prop) ) := 
sorry

end meiosis_fertilization_correct_l1988_198890


namespace cd_leq_one_l1988_198835

variables {a b c d : ℝ}

theorem cd_leq_one (h1 : a * b = 1) (h2 : a * c + b * d = 2) : c * d ≤ 1 := 
sorry

end cd_leq_one_l1988_198835


namespace president_and_committee_l1988_198888

def combinatorial (n k : ℕ) : ℕ := Nat.choose n k

theorem president_and_committee :
  let num_people := 10
  let num_president := 1
  let num_committee := 3
  let num_ways_president := 10
  let num_remaining_people := num_people - num_president
  let num_ways_committee := combinatorial num_remaining_people num_committee
  num_ways_president * num_ways_committee = 840 := 
by
  sorry

end president_and_committee_l1988_198888


namespace measure_AB_l1988_198827

noncomputable def segment_measure (a b : ℝ) : ℝ :=
  a + (2 / 3) * b

theorem measure_AB (a b : ℝ) (parallel_AB_CD : true) (angle_B_three_times_angle_D : true) (measure_AD_eq_a : true) (measure_CD_eq_b : true) :
  segment_measure a b = a + (2 / 3) * b :=
by
  sorry

end measure_AB_l1988_198827


namespace tenth_term_of_sequence_l1988_198842

theorem tenth_term_of_sequence : 
  let a_1 := 3
  let d := 6 
  let n := 10 
  (a_1 + (n-1) * d) = 57 := by
  sorry

end tenth_term_of_sequence_l1988_198842


namespace ceil_minus_floor_eq_one_implies_ceil_minus_y_l1988_198896

noncomputable def fractional_part (y : ℝ) : ℝ := y - ⌊y⌋

theorem ceil_minus_floor_eq_one_implies_ceil_minus_y (y : ℝ) (h : ⌈y⌉ - ⌊y⌋ = 1) : ⌈y⌉ - y = 1 - fractional_part y :=
by
  sorry

end ceil_minus_floor_eq_one_implies_ceil_minus_y_l1988_198896


namespace abs_p_minus_1_ge_2_l1988_198828

theorem abs_p_minus_1_ge_2 (p : ℝ) (a : ℕ → ℝ) 
  (h₀ : a 0 = 1)
  (h₁ : a 1 = p)
  (h₂ : a 2 = p * (p - 1))
  (h₃ : ∀ n : ℕ, a (n + 3) = p * a (n + 2) - p * a (n + 1) + a n)
  (h₄ : ∀ n : ℕ, a n > 0)
  (h₅ : ∀ m n : ℕ, m ≥ n → a m * a n > a (m + 1) * a (n - 1)) :
  |p - 1| ≥ 2 :=
sorry

end abs_p_minus_1_ge_2_l1988_198828


namespace henry_total_fee_8_bikes_l1988_198802

def paint_fee := 5
def sell_fee := paint_fee + 8
def total_fee_per_bike := paint_fee + sell_fee
def total_fee (bikes : ℕ) := bikes * total_fee_per_bike

theorem henry_total_fee_8_bikes : total_fee 8 = 144 :=
by
  sorry

end henry_total_fee_8_bikes_l1988_198802


namespace smallest_palindromic_odd_integer_in_base2_and_4_l1988_198853

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := n.digits base
  digits = digits.reverse

theorem smallest_palindromic_odd_integer_in_base2_and_4 :
  ∃ n : ℕ, n > 10 ∧ is_palindrome n 2 ∧ is_palindrome n 4 ∧ Odd n ∧ ∀ m : ℕ, (m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 4 ∧ Odd m) → n <= m :=
  sorry

end smallest_palindromic_odd_integer_in_base2_and_4_l1988_198853


namespace balloon_difference_l1988_198809

theorem balloon_difference (x y : ℝ) (h1 : x = 2 * y - 3) (h2 : y = x / 4 + 1) : x - y = -2.5 :=
by 
  sorry

end balloon_difference_l1988_198809


namespace log_increasing_a_gt_one_l1988_198882

noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_increasing_a_gt_one (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log a 2 < log a 3) : a > 1 :=
by
  sorry

end log_increasing_a_gt_one_l1988_198882


namespace find_number_l1988_198840

theorem find_number (x : ℝ) (h : x / 3 = 1.005 * 400) : x = 1206 := 
by 
sorry

end find_number_l1988_198840


namespace base_8_to_10_conversion_l1988_198883

theorem base_8_to_10_conversion : (2 * 8^4 + 3 * 8^3 + 4 * 8^2 + 5 * 8^1 + 6 * 8^0) = 10030 := by 
  -- specify the summation directly 
  sorry

end base_8_to_10_conversion_l1988_198883


namespace initial_distance_l1988_198819

def relative_speed (v1 v2 : ℝ) : ℝ := v1 + v2

def total_distance (rel_speed time : ℝ) : ℝ := rel_speed * time

theorem initial_distance (v1 v2 time : ℝ) : (v1 = 1.6) → (v2 = 1.9) → 
                                            (time = 100) →
                                            total_distance (relative_speed v1 v2) time = 350 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [relative_speed, total_distance]
  sorry

end initial_distance_l1988_198819


namespace apple_price_l1988_198891

variable (p q : ℝ)

theorem apple_price :
  (30 * p + 3 * q = 168) →
  (30 * p + 6 * q = 186) →
  (20 * p = 100) →
  p = 5 :=
by
  intros h1 h2 h3
  have h4 : p = 5 := sorry
  exact h4

end apple_price_l1988_198891


namespace probability_is_correct_l1988_198862

-- Given definitions
def total_marbles : ℕ := 100
def red_marbles : ℕ := 35
def white_marbles : ℕ := 30
def green_marbles : ℕ := 10

-- Probe the probability
noncomputable def probability_red_white_green : ℚ :=
  (red_marbles + white_marbles + green_marbles : ℚ) / total_marbles

-- The theorem we need to prove
theorem probability_is_correct :
  probability_red_white_green = 0.75 := by
  sorry

end probability_is_correct_l1988_198862


namespace fermats_little_theorem_l1988_198824

theorem fermats_little_theorem (p : ℕ) (hp : Nat.Prime p) (a : ℤ) : (a^p - a) % p = 0 := 
by sorry

end fermats_little_theorem_l1988_198824


namespace divides_power_of_odd_l1988_198880

theorem divides_power_of_odd (k : ℕ) (hk : k % 2 = 1) (n : ℕ) (hn : n ≥ 1) : 2^(n + 2) ∣ (k^(2^n) - 1) :=
by
  sorry

end divides_power_of_odd_l1988_198880


namespace sine_addition_l1988_198844

noncomputable def sin_inv_45 := Real.arcsin (4 / 5)
noncomputable def tan_inv_12 := Real.arctan (1 / 2)

theorem sine_addition :
  Real.sin (sin_inv_45 + tan_inv_12) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sine_addition_l1988_198844


namespace all_plants_diseased_l1988_198850

theorem all_plants_diseased (n : ℕ) (h : n = 1007) : 
  n * 2 = 2014 := by
  sorry

end all_plants_diseased_l1988_198850


namespace perfect_squares_difference_l1988_198846

theorem perfect_squares_difference : 
  let N : ℕ := 20000;
  let diff_squared (b : ℤ) : ℤ := (b+2)^2 - b^2;
  ∃ k : ℕ, (1 ≤ k ∧ k ≤ 70) ∧ (∀ m : ℕ, (m < N) → (∃ b : ℤ, m = diff_squared b) → m = (2 * k)^2)
:= sorry

end perfect_squares_difference_l1988_198846


namespace unique_prime_p_l1988_198868

def f (x : ℤ) : ℤ := x^3 + 7 * x^2 + 9 * x + 10

theorem unique_prime_p (p : ℕ) (hp : p = 5 ∨ p = 7 ∨ p = 11 ∨ p = 13 ∨ p = 17) :
  (∀ a b : ℤ, f a ≡ f b [ZMOD p] → a ≡ b [ZMOD p]) ↔ p = 11 :=
by
  sorry

end unique_prime_p_l1988_198868


namespace jills_present_age_l1988_198843

-- Define the problem parameters and conditions
variables (H J : ℕ)
axiom cond1 : H + J = 43
axiom cond2 : H - 5 = 2 * (J - 5)

-- State the goal
theorem jills_present_age : J = 16 :=
sorry

end jills_present_age_l1988_198843


namespace total_hours_before_midterms_l1988_198893

-- Define the hours spent on each activity per week
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3

-- Sum up the total hours spent on extracurriculars per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Define semester information
def total_weeks_per_semester : ℕ := 12
def weeks_before_midterms : ℕ := total_weeks_per_semester / 2
def weeks_sick : ℕ := 2
def active_weeks_before_midterms : ℕ := weeks_before_midterms - weeks_sick

-- Define the theorem statement about total hours before midterms
theorem total_hours_before_midterms : total_hours_per_week * active_weeks_before_midterms = 52 := by
  -- We skip the actual proof here
  sorry

end total_hours_before_midterms_l1988_198893


namespace length_of_generatrix_l1988_198832

/-- Given that the base radius of a cone is sqrt(2), and its lateral surface is unfolded into a semicircle,
prove that the length of the generatrix of the cone is 2 sqrt(2). -/
theorem length_of_generatrix (r l : ℝ) (h1 : r = Real.sqrt 2)
    (h2 : 2 * Real.pi * r = Real.pi * l) : l = 2 * Real.sqrt 2 :=
by
  sorry

end length_of_generatrix_l1988_198832


namespace train_travel_distance_l1988_198881

def speed (miles : ℕ) (minutes : ℕ) : ℕ :=
  miles / minutes

def minutes_in_hours (hours : ℕ) : ℕ :=
  hours * 60

def distance_traveled (rate : ℕ) (time : ℕ) : ℕ :=
  rate * time

theorem train_travel_distance :
  (speed 2 2 = 1) →
  (minutes_in_hours 3 = 180) →
  distance_traveled (speed 2 2) (minutes_in_hours 3) = 180 :=
by
  intros h_speed h_minutes
  rw [h_speed, h_minutes]
  sorry

end train_travel_distance_l1988_198881


namespace frog_paths_l1988_198829

theorem frog_paths (n : ℕ) : (∃ e_2n e_2n_minus_1 : ℕ,
  e_2n_minus_1 = 0 ∧
  e_2n = (1 / Real.sqrt 2) * ((2 + Real.sqrt 2) ^ (n - 1) - (2 - Real.sqrt 2) ^ (n - 1))) :=
by {
  sorry
}

end frog_paths_l1988_198829


namespace train_length_l1988_198813

theorem train_length (speed_kmph : ℕ) (time_sec : ℕ) (length_meters : ℕ) : speed_kmph = 90 → time_sec = 4 → length_meters = 100 :=
by
  intros h₁ h₂
  have speed_mps : ℕ := speed_kmph * 1000 / 3600
  have speed_mps_val : speed_mps = 25 := sorry
  have distance : ℕ := speed_mps * time_sec
  have distance_val : distance = 100 := sorry
  exact sorry

end train_length_l1988_198813


namespace arithmetic_sequence_product_l1988_198894

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b n < b (n + 1))
  (h_condition : b 5 * b 6 = 14) :
  (b 4 * b 7 = -324) ∨ (b 4 * b 7 = -36) :=
sorry

end arithmetic_sequence_product_l1988_198894


namespace part1_part2_l1988_198815

def custom_operation (a b : ℝ) : ℝ := a^2 + 2*a*b

theorem part1 : custom_operation 2 3 = 16 :=
by sorry

theorem part2 (x : ℝ) (h : custom_operation (-2) x = -2 + x) : x = 6 / 5 :=
by sorry

end part1_part2_l1988_198815


namespace area_of_one_trapezoid_l1988_198807

theorem area_of_one_trapezoid (outer_area inner_area : ℝ) (num_trapezoids : ℕ) (h_outer : outer_area = 36) (h_inner : inner_area = 4) (h_num_trapezoids : num_trapezoids = 3) : (outer_area - inner_area) / num_trapezoids = 32 / 3 :=
by
  rw [h_outer, h_inner, h_num_trapezoids]
  norm_num

end area_of_one_trapezoid_l1988_198807


namespace distinct_arrangements_ballon_l1988_198851

theorem distinct_arrangements_ballon : 
  let n := 6
  let repetitions := 2
  n! / repetitions! = 360 :=
by
  sorry

end distinct_arrangements_ballon_l1988_198851


namespace math_problem_l1988_198861

def calc_expr : Int := 
  54322 * 32123 - 54321 * 32123 + 54322 * 99000 - 54321 * 99001

theorem math_problem :
  calc_expr = 76802 := 
by
  sorry

end math_problem_l1988_198861


namespace plane_speeds_l1988_198818

theorem plane_speeds (v : ℕ) 
    (h1 : ∀ (t : ℕ), t = 5 → 20 * v = 4800): 
  v = 240 ∧ 3 * v = 720 := by
  sorry

end plane_speeds_l1988_198818


namespace pure_imaginary_number_solution_l1988_198856

-- Definition of the problem
theorem pure_imaginary_number_solution (a : ℝ) (h1 : a^2 - 4 = 0) (h2 : a^2 - 3 * a + 2 ≠ 0) : a = -2 :=
sorry

end pure_imaginary_number_solution_l1988_198856


namespace icing_time_is_30_l1988_198898

def num_batches : Nat := 4
def baking_time_per_batch : Nat := 20
def total_time : Nat := 200

def baking_time_total : Nat := num_batches * baking_time_per_batch
def icing_time_total : Nat := total_time - baking_time_total
def icing_time_per_batch : Nat := icing_time_total / num_batches

theorem icing_time_is_30 :
  icing_time_per_batch = 30 := by
  sorry

end icing_time_is_30_l1988_198898


namespace copper_alloy_proof_l1988_198895

variable (x p : ℝ)

theorem copper_alloy_proof
  (copper_content1 copper_content2 weight1 weight2 total_weight : ℝ)
  (h1 : weight1 = 3)
  (h2 : copper_content1 = 0.4)
  (h3 : weight2 = 7)
  (h4 : copper_content2 = 0.3)
  (h5 : total_weight = 8)
  (h6 : 1 ≤ x ∧ x ≤ 3)
  (h7 : p = 100 * (copper_content1 * x + copper_content2 * (total_weight - x)) / total_weight) :
  31.25 ≤ p ∧ p ≤ 33.75 := 
  sorry

end copper_alloy_proof_l1988_198895


namespace parabola_vertex_location_l1988_198876

theorem parabola_vertex_location (a b c : ℝ) (h1 : ∀ x < 0, a * x^2 + b * x + c ≤ 0) (h2 : a < 0) : 
  -b / (2 * a) ≥ 0 :=
by
  sorry

end parabola_vertex_location_l1988_198876


namespace carmina_coins_l1988_198826

-- Define the conditions related to the problem
variables (n d : ℕ) -- number of nickels and dimes

theorem carmina_coins (h1 : 5 * n + 10 * d = 360) (h2 : 10 * n + 5 * d = 540) : n + d = 60 :=
sorry

end carmina_coins_l1988_198826


namespace sum_first_2017_terms_l1988_198852

theorem sum_first_2017_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → S (n + 1) - S n = 3^n / a n) :
  S 2017 = 3^1009 - 2 := sorry

end sum_first_2017_terms_l1988_198852


namespace eval_expression_l1988_198877

theorem eval_expression : ⌈- (7 / 3 : ℚ)⌉ + ⌊(7 / 3 : ℚ)⌋ = 0 := 
by 
  sorry

end eval_expression_l1988_198877


namespace distance_to_lake_l1988_198897

theorem distance_to_lake 
  {d : ℝ} 
  (h1 : ¬ (d ≥ 8))
  (h2 : ¬ (d ≤ 7))
  (h3 : ¬ (d ≤ 6)) : 
  (7 < d) ∧ (d < 8) :=
by
  sorry

end distance_to_lake_l1988_198897


namespace equilateral_triangle_perimeter_l1988_198803

theorem equilateral_triangle_perimeter (s : ℕ) (b : ℕ) (h1 : 40 = 2 * s + b) (h2 : b = 10) : 3 * s = 45 :=
by {
  sorry
}

end equilateral_triangle_perimeter_l1988_198803


namespace blue_pens_count_l1988_198874

variable (x y : ℕ) -- Define x as the number of red pens and y as the number of blue pens.
variable (h1 : 5 * x + 7 * y = 102) -- Condition 1: Total cost equation.
variable (h2 : x + y = 16) -- Condition 2: Total number of pens equation.

theorem blue_pens_count : y = 11 :=
by
  sorry

end blue_pens_count_l1988_198874


namespace angle_in_third_quadrant_l1988_198892

theorem angle_in_third_quadrant (θ : ℤ) (hθ : θ = -510) : 
  (210 % 360 > 180 ∧ 210 % 360 < 270) := 
by
  have h : 210 % 360 = 210 := by norm_num
  sorry

end angle_in_third_quadrant_l1988_198892


namespace remainder_2_pow_33_mod_9_l1988_198836

theorem remainder_2_pow_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end remainder_2_pow_33_mod_9_l1988_198836


namespace find_constants_l1988_198869

variable (x : ℝ)

/-- Restate the equation problem and the constants A, B, C, D to be found. -/
theorem find_constants 
  (A B C D : ℝ)
  (h : ∀ x, x^3 - 7 = A * (x - 3) * (x - 5) * (x - 7) + B * (x - 2) * (x - 5) * (x - 7) + C * (x - 2) * (x - 3) * (x - 7) + D * (x - 2) * (x - 3) * (x - 5)) :
  A = 1/15 ∧ B = 5/2 ∧ C = -59/6 ∧ D = 42/5 :=
  sorry

end find_constants_l1988_198869


namespace emily_furniture_assembly_time_l1988_198820

def num_chairs : Nat := 4
def num_tables : Nat := 2
def num_shelves : Nat := 3
def num_wardrobe : Nat := 1

def time_per_chair : Nat := 8
def time_per_table : Nat := 15
def time_per_shelf : Nat := 10
def time_per_wardrobe : Nat := 45

def total_time : Nat := 
  num_chairs * time_per_chair + 
  num_tables * time_per_table + 
  num_shelves * time_per_shelf + 
  num_wardrobe * time_per_wardrobe

theorem emily_furniture_assembly_time : total_time = 137 := by
  unfold total_time
  sorry

end emily_furniture_assembly_time_l1988_198820


namespace sin_solution_set_l1988_198812

open Real

theorem sin_solution_set (x : ℝ) : 
  (3 * sin x = 1 + cos (2 * x)) ↔ ∃ k : ℤ, x = k * π + (-1) ^ k * (π / 6) :=
by
  sorry

end sin_solution_set_l1988_198812


namespace energy_of_first_particle_l1988_198808

theorem energy_of_first_particle
  (E_1 E_2 E_3 : ℤ)
  (h1 : E_1^2 - E_2^2 - E_3^2 + E_1 * E_2 = 5040)
  (h2 : E_1^2 + 2 * E_2^2 + 2 * E_3^2 - 2 * E_1 * E_2 - E_1 * E_3 - E_2 * E_3 = -4968)
  (h3 : 0 < E_3)
  (h4 : E_3 ≤ E_2)
  (h5 : E_2 ≤ E_1) : E_1 = 12 :=
by sorry

end energy_of_first_particle_l1988_198808


namespace inequality_proof_l1988_198885

section
variable {a b x y : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hab : a + b = 1) :
  (1 / (a / x + b / y) ≤ a * x + b * y) ∧ (1 / (a / x + b / y) = a * x + b * y ↔ a * y = b * x) :=
  sorry
end

end inequality_proof_l1988_198885


namespace ellipse_C_properties_l1988_198830

open Real

noncomputable def ellipse_eq (b : ℝ) : Prop :=
  (∀ (x y : ℝ), (x = 1 ∧ y = sqrt 3 / 2) → (x^2 / 4 + y^2 / b^2 = 1))

theorem ellipse_C_properties : 
  (∀ (C : ℝ → ℝ → Prop), 
    (C 0 0) ∧ 
    (∀ x y, C x y → (x = 0 ↔ y = 0)) ∧ 
    (∀ x, C x 0) ∧ 
    (∃ x y, C x y ∧ x = 1 ∧ y = sqrt 3 / 2) →
    (∃ b, b > 0 ∧ b^2 = 1 ∧ ellipse_eq b)) ∧
  (∀ P A B : ℝ × ℝ, 
    (P.1 = P.1 ∧ P.1 ≠ 0 ∧ P.2 = 0 ∧ -2 ≤ P.1 ∧ P.1 ≤ 2) →
    (A.2 = 1/2 * (A.1 - P.1) ∧ B.2 = 1/2 * (B.1 - P.1)) →
    ((P.1 - A.1)^2 + A.2^2 + (P.1 - B.1)^2 + B.2^2 = 5)) :=
by sorry

end ellipse_C_properties_l1988_198830


namespace parabola_intercept_sum_l1988_198864

theorem parabola_intercept_sum : 
  let d := 4
  let e := (9 + Real.sqrt 33) / 6
  let f := (9 - Real.sqrt 33) / 6
  d + e + f = 7 :=
by 
  sorry

end parabola_intercept_sum_l1988_198864


namespace clara_gave_10_stickers_l1988_198854

-- Defining the conditions
def initial_stickers : ℕ := 100
def remaining_after_boy (B : ℕ) : ℕ := initial_stickers - B
def remaining_after_friends (B : ℕ) : ℕ := (remaining_after_boy B) / 2

-- Theorem stating that Clara gave 10 stickers to the boy
theorem clara_gave_10_stickers (B : ℕ) (h : remaining_after_friends B = 45) : B = 10 :=
by
  sorry

end clara_gave_10_stickers_l1988_198854


namespace coats_from_high_schools_l1988_198814

-- Define the total number of coats collected.
def total_coats_collected : ℕ := 9437

-- Define the number of coats collected from elementary schools.
def coats_from_elementary : ℕ := 2515

-- Goal: Prove that the number of coats collected from high schools is 6922.
theorem coats_from_high_schools : (total_coats_collected - coats_from_elementary) = 6922 := by
  sorry

end coats_from_high_schools_l1988_198814


namespace total_length_of_segments_l1988_198834

theorem total_length_of_segments
  (l1 l2 l3 l4 l5 l6 : ℕ) 
  (hl1 : l1 = 5) 
  (hl2 : l2 = 1) 
  (hl3 : l3 = 4) 
  (hl4 : l4 = 2) 
  (hl5 : l5 = 3) 
  (hl6 : l6 = 3) : 
  l1 + l2 + l3 + l4 + l5 + l6 = 18 := 
by 
  sorry

end total_length_of_segments_l1988_198834


namespace find_fraction_l1988_198889

variable (n : ℚ) (x : ℚ)

theorem find_fraction (h1 : n = 0.5833333333333333) (h2 : n = 1/3 + x) : x = 0.25 := by
  sorry

end find_fraction_l1988_198889


namespace money_left_l1988_198848

theorem money_left (olivia_money nigel_money ticket_cost tickets_purchased : ℕ) 
  (h1 : olivia_money = 112) 
  (h2 : nigel_money = 139) 
  (h3 : ticket_cost = 28) 
  (h4 : tickets_purchased = 6) : 
  olivia_money + nigel_money - tickets_purchased * ticket_cost = 83 := 
by 
  sorry

end money_left_l1988_198848


namespace four_pq_plus_four_qp_l1988_198817

theorem four_pq_plus_four_qp (p q : ℝ) (h : p / q - q / p = 21 / 10) : 
  4 * p / q + 4 * q / p = 16.8 :=
sorry

end four_pq_plus_four_qp_l1988_198817


namespace value_of_f_of_g_l1988_198858

def f (x : ℝ) : ℝ := 2 * x + 4
def g (x : ℝ) : ℝ := x^2 - 9

theorem value_of_f_of_g : f (g 3) = 4 :=
by
  -- The proof would go here. Since we are only defining the statement, we can leave this as 'sorry'.
  sorry

end value_of_f_of_g_l1988_198858


namespace dogs_with_no_accessories_l1988_198804

theorem dogs_with_no_accessories :
  let total := 120
  let tags := 60
  let flea_collars := 50
  let harnesses := 30
  let tags_and_flea_collars := 20
  let tags_and_harnesses := 15
  let flea_collars_and_harnesses := 10
  let all_three := 5
  total - (tags + flea_collars + harnesses - tags_and_flea_collars - tags_and_harnesses - flea_collars_and_harnesses + all_three) = 25 := by
  sorry

end dogs_with_no_accessories_l1988_198804


namespace correct_statements_count_l1988_198866

/-
  Question: How many students have given correct interpretations of the algebraic expression \( 7x \)?
  Conditions:
    - Xiaoming's Statement: \( 7x \) can represent the sum of \( 7 \) and \( x \).
    - Xiaogang's Statement: \( 7x \) can represent the product of \( 7 \) and \( x \).
    - Xiaoliang's Statement: \( 7x \) can represent the total price of buying \( x \) pens at a unit price of \( 7 \) yuan.
  Given these conditions, prove that the number of correct statements is \( 2 \).
-/

theorem correct_statements_count (x : ℕ) :
  (if 7 * x = 7 + x then 1 else 0) +
  (if 7 * x = 7 * x then 1 else 0) +
  (if 7 * x = 7 * x then 1 else 0) = 2 := sorry

end correct_statements_count_l1988_198866


namespace max_min_f_l1988_198806

-- Defining a and the set A
def a : ℤ := 2001

def A : Set (ℤ × ℤ) := {p | p.snd ≠ 0 ∧ p.fst < 2 * a ∧ (2 * p.snd) ∣ ((2 * a * p.fst) - (p.fst * p.fst) + (p.snd * p.snd)) ∧ ((p.snd * p.snd) - (p.fst * p.fst) + (2 * p.fst * p.snd) ≤ (2 * a * (p.snd - p.fst)))}

-- Defining the function f
def f (m n : ℤ): ℤ := (2 * a * m - m * m - m * n) / n

-- Main theorem: Proving that the maximum and minimum values of f over A are 3750 and 2 respectively
theorem max_min_f : 
  ∃ p ∈ A, f p.fst p.snd = 3750 ∧
  ∃ q ∈ A, f q.fst q.snd = 2 :=
sorry

end max_min_f_l1988_198806


namespace number_of_deluxe_volumes_l1988_198859

theorem number_of_deluxe_volumes (d s : ℕ) 
  (h1 : d + s = 15)
  (h2 : 30 * d + 20 * s = 390) : 
  d = 9 :=
by
  sorry

end number_of_deluxe_volumes_l1988_198859


namespace divisibility_problem_l1988_198870

theorem divisibility_problem (a b k : ℕ) :
  (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) →
  a * b^2 + b + 7 ∣ a^2 * b + a + b := by
  intro h
  cases h
  case inl h1 =>
    rw [h1.1, h1.2]
    sorry
  case inr h2 =>
    cases h2
    case inl h21 =>
      rw [h21.1, h21.2]
      sorry
    case inr h22 =>
      rw [h22.1, h22.2]
      sorry

end divisibility_problem_l1988_198870


namespace abs_inequality_solution_l1988_198867

theorem abs_inequality_solution (x : ℝ) :
  |2 * x - 2| + |2 * x + 4| < 10 ↔ x ∈ Set.Ioo (-4 : ℝ) (2 : ℝ) := 
by sorry

end abs_inequality_solution_l1988_198867


namespace inverse_B_squared_l1988_198811

-- Defining the inverse matrix B_inv
def B_inv : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; 0, 1]

-- Theorem to prove that the inverse of B^2 is a specific matrix
theorem inverse_B_squared :
  (B_inv * B_inv) = !![9, -6; 0, 1] :=
  by sorry


end inverse_B_squared_l1988_198811


namespace find_first_number_l1988_198831

variable (x y : ℕ)

theorem find_first_number (h1 : y = 11) (h2 : x + (y + 3) = 19) : x = 5 :=
by
  sorry

end find_first_number_l1988_198831


namespace exists_three_naturals_sum_perfect_square_no_four_naturals_sum_perfect_square_l1988_198865

-- Definition for the condition that ab + 10 is a perfect square
def is_perfect_square_sum (a b : ℕ) : Prop := ∃ k : ℕ, a * b + 10 = k * k

-- Problem: Existence of three different natural numbers for which the sum of the product of any two with 10 is a perfect square
theorem exists_three_naturals_sum_perfect_square :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_perfect_square_sum a b ∧ is_perfect_square_sum b c ∧ is_perfect_square_sum c a := sorry

-- Problem: Non-existence of four different natural numbers for which the sum of the product of any two with 10 is a perfect square
theorem no_four_naturals_sum_perfect_square :
  ¬ ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d ∧
    is_perfect_square_sum a b ∧ is_perfect_square_sum a c ∧ is_perfect_square_sum a d ∧
    is_perfect_square_sum b c ∧ is_perfect_square_sum b d ∧ is_perfect_square_sum c d := sorry

end exists_three_naturals_sum_perfect_square_no_four_naturals_sum_perfect_square_l1988_198865


namespace time_to_office_l1988_198871

theorem time_to_office (S T : ℝ) (h1 : T > 0) (h2 : S > 0) 
    (h : S * (T + 15) = (4/5) * S * T) :
    T = 75 := by
  sorry

end time_to_office_l1988_198871


namespace age_ratio_l1988_198838

theorem age_ratio (B_current A_current B_10_years_ago A_in_10_years : ℕ) 
  (h1 : B_current = 37) 
  (h2 : A_current = B_current + 7) 
  (h3 : B_10_years_ago = B_current - 10) 
  (h4 : A_in_10_years = A_current + 10) : 
  A_in_10_years / B_10_years_ago = 2 :=
by
  sorry

end age_ratio_l1988_198838


namespace smallest_positive_period_of_y_l1988_198805

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := Real.sin (-x / 2 + Real.pi / 4)

-- Statement we need to prove
theorem smallest_positive_period_of_y :
  ∃ T > 0, ∀ x : ℝ, y (x + T) = y x ∧ T = 4 * Real.pi := sorry

end smallest_positive_period_of_y_l1988_198805


namespace least_red_chips_l1988_198837

/--
  There are 70 chips in a box. Each chip is either red or blue.
  If the sum of the number of red chips and twice the number of blue chips equals a prime number,
  proving that the least possible number of red chips is 69.
-/
theorem least_red_chips (r b : ℕ) (p : ℕ) (h1 : r + b = 70) (h2 : r + 2 * b = p) (hp : Nat.Prime p) :
  r = 69 :=
by
  -- Proof goes here
  sorry

end least_red_chips_l1988_198837


namespace greatest_two_digit_number_with_digit_product_16_l1988_198878

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digit_product (n m : ℕ) : Prop :=
  n * m = 16

def from_digits (n m : ℕ) : ℕ :=
  10 * n + m

theorem greatest_two_digit_number_with_digit_product_16 :
  ∀ n m, is_two_digit_number (from_digits n m) → digit_product n m → (82 ≥ from_digits n m) :=
by
  intros n m h1 h2
  sorry

end greatest_two_digit_number_with_digit_product_16_l1988_198878


namespace sum_of_roots_of_quadratic_l1988_198841

open Polynomial

theorem sum_of_roots_of_quadratic :
  ∀ (m n : ℝ), (m ≠ n ∧ (∀ x, x^2 + 2*x - 1 = 0 → x = m ∨ x = n)) → m + n = -2 :=
by
  sorry

end sum_of_roots_of_quadratic_l1988_198841


namespace square_flag_side_length_side_length_of_square_flags_is_4_l1988_198872

theorem square_flag_side_length 
  (total_fabric : ℕ)
  (fabric_left : ℕ)
  (num_square_flags : ℕ)
  (num_wide_flags : ℕ)
  (num_tall_flags : ℕ)
  (wide_flag_length : ℕ)
  (wide_flag_width : ℕ)
  (tall_flag_length : ℕ)
  (tall_flag_width : ℕ)
  (fabric_used_on_wide_and_tall_flags : ℕ)
  (fabric_used_on_all_flags : ℕ)
  (fabric_used_on_square_flags : ℕ)
  (square_flag_area : ℕ)
  (side_length : ℕ) : Prop :=
  total_fabric = 1000 ∧
  fabric_left = 294 ∧
  num_square_flags = 16 ∧
  num_wide_flags = 20 ∧
  num_tall_flags = 10 ∧
  wide_flag_length = 5 ∧
  wide_flag_width = 3 ∧
  tall_flag_length = 5 ∧
  tall_flag_width = 3 ∧
  fabric_used_on_wide_and_tall_flags = (num_wide_flags + num_tall_flags) * (wide_flag_length * wide_flag_width) ∧
  fabric_used_on_all_flags = total_fabric - fabric_left ∧
  fabric_used_on_square_flags = fabric_used_on_all_flags - fabric_used_on_wide_and_tall_flags ∧
  square_flag_area = fabric_used_on_square_flags / num_square_flags ∧
  side_length = Int.sqrt square_flag_area ∧
  side_length = 4

theorem side_length_of_square_flags_is_4 : 
  square_flag_side_length 1000 294 16 20 10 5 3 5 3 450 706 256 16 4 :=
  by
    sorry

end square_flag_side_length_side_length_of_square_flags_is_4_l1988_198872


namespace nonnegative_values_ineq_l1988_198857

theorem nonnegative_values_ineq {x : ℝ} : 
  (x^2 - 6*x + 9) / (9 - x^3) ≥ 0 ↔ x ∈ Set.Iic 3 := 
sorry

end nonnegative_values_ineq_l1988_198857


namespace find_percentage_l1988_198810

theorem find_percentage (p : ℝ) (h : (p / 100) * 8 = 0.06) : p = 0.75 := 
by 
  sorry

end find_percentage_l1988_198810


namespace target1_target2_l1988_198833

variable (α : ℝ)

-- Define the condition
def tan_alpha := Real.tan α = 2

-- State the first target with the condition considered
theorem target1 (h : tan_alpha α) : 
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 := by
  sorry

-- State the second target with the condition considered
theorem target2 (h : tan_alpha α) : 
  4 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 5 * Real.cos α ^ 2 = 1 := by
  sorry

end target1_target2_l1988_198833


namespace find_B_value_l1988_198825

theorem find_B_value (A C B : ℕ) (h1 : A = 634) (h2 : A = C + 593) (h3 : B = C + 482) : B = 523 :=
by {
  -- Proof would go here
  sorry
}

end find_B_value_l1988_198825


namespace rectangle_area_l1988_198879

theorem rectangle_area 
  (P : ℝ) (r : ℝ) (hP : P = 40) (hr : r = 3 / 2) : 
  ∃ (length width : ℝ), 2 * (length + width) = P ∧ length = 3 * (width / 2) ∧ (length * width) = 96 :=
by
  sorry

end rectangle_area_l1988_198879


namespace tan_double_angle_third_quadrant_l1988_198863

open Real

theorem tan_double_angle_third_quadrant (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2) 
  (h2 : sin (π - α) = - (3 / 5)) : 
  tan (2 * α) = 24 / 7 :=
by
  sorry

end tan_double_angle_third_quadrant_l1988_198863


namespace unique_solution_l1988_198886

theorem unique_solution (x : ℝ) (h : (1 / (x - 1)) = (3 / (2 * x - 3))) : x = 0 := 
sorry

end unique_solution_l1988_198886


namespace valid_third_side_length_l1988_198884

theorem valid_third_side_length : 4 < 6 ∧ 6 < 10 :=
by
  exact ⟨by norm_num, by norm_num⟩

end valid_third_side_length_l1988_198884


namespace set_complement_intersection_l1988_198816

variable (U : Set ℕ) (M N : Set ℕ)

theorem set_complement_intersection
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {1, 4, 5})
  (hN : N = {2, 3}) :
  ((U \ N) ∩ M) = {1, 4, 5} :=
by
  sorry

end set_complement_intersection_l1988_198816


namespace sqrt3_times_3_minus_sqrt3_bound_l1988_198800

theorem sqrt3_times_3_minus_sqrt3_bound : 2 < (Real.sqrt 3) * (3 - (Real.sqrt 3)) ∧ (Real.sqrt 3) * (3 - (Real.sqrt 3)) < 3 := 
by 
  sorry

end sqrt3_times_3_minus_sqrt3_bound_l1988_198800


namespace remainder_of_power_mod_five_l1988_198875

theorem remainder_of_power_mod_five : (4 ^ 11) % 5 = 4 :=
by
  sorry

end remainder_of_power_mod_five_l1988_198875


namespace value_of_other_bills_l1988_198855

theorem value_of_other_bills (x : ℕ) : 
  (∃ (num_twenty num_x : ℕ), num_twenty = 3 ∧
                           num_x = 2 * num_twenty ∧
                           20 * num_twenty + x * num_x = 120) → 
  x * 6 = 60 :=
by
  intro h
  obtain ⟨num_twenty, num_x, h1, h2, h3⟩ := h
  have : num_twenty = 3 := h1
  have : num_x = 2 * num_twenty := h2
  have : x * 6 = 60 := sorry
  exact this

end value_of_other_bills_l1988_198855


namespace arithmetic_and_geometric_sequence_statement_l1988_198873

-- Arithmetic sequence definitions
def arithmetic_seq (a b d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Conditions
def a_2 : ℕ := 9
def a_5 : ℕ := 21

-- General formula and solution for part (Ⅰ)
def general_formula_arithmetic_sequence : Prop :=
  ∃ (a d : ℕ), (a + d = a_2 ∧ a + 4 * d = a_5) ∧ ∀ n : ℕ, arithmetic_seq a d n = 4 * n + 1

-- Definitions and conditions for geometric sequence derived from arithmetic sequence
def b_n (n : ℕ) : ℕ := 2 ^ (4 * n + 1)

-- Sum of the first n terms of the sequence {b_n}
def S_n (n : ℕ) : ℕ := (32 * (2 ^ (4 * n) - 1)) / 15

-- Statement that needs to be proven
theorem arithmetic_and_geometric_sequence_statement :
  general_formula_arithmetic_sequence ∧ (∀ n, S_n n = (32 * (2 ^ (4 * n) - 1)) / 15) := by
  sorry

end arithmetic_and_geometric_sequence_statement_l1988_198873


namespace angle_measure_l1988_198860

theorem angle_measure (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 :=
by
  sorry

end angle_measure_l1988_198860
