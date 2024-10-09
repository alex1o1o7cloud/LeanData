import Mathlib

namespace calculate_expression_l465_46569

theorem calculate_expression :
  (-0.125) ^ 2009 * (8 : ℝ) ^ 2009 = -1 :=
sorry

end calculate_expression_l465_46569


namespace f_characterization_l465_46526

noncomputable def op (a b : ℝ) := a * b

noncomputable def ot (a b : ℝ) := a + b

noncomputable def f (x : ℝ) := ot x 2 - op 2 x

-- Prove that f(x) is neither odd nor even and is a decreasing function
theorem f_characterization :
  (∀ x : ℝ, f x = -x + 2) ∧
  (∀ x : ℝ, f (-x) ≠ f x ∧ f (-x) ≠ -f x) ∧
  (∀ x y : ℝ, x < y → f x > f y) := sorry

end f_characterization_l465_46526


namespace find_f_neg_2_l465_46538

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 3^x - 1 else sorry -- we'll define this not for non-negative x properly later

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem find_f_neg_2 (hodd : is_odd_function f) (hpos : ∀ x : ℝ, 0 ≤ x → f x = 3^x - 1) :
  f (-2) = -8 :=
by
  -- Proof omitted
  sorry

end find_f_neg_2_l465_46538


namespace graphs_intersect_at_one_point_l465_46523

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.log (4 * x) / Real.log 2

theorem graphs_intersect_at_one_point : ∃! x, f x = g x :=
by {
  sorry
}

end graphs_intersect_at_one_point_l465_46523


namespace collinear_points_sum_l465_46549

variables {a b : ℝ}

/-- If the points (1, a, b), (a, b, 3), and (b, 3, a) are collinear, then b + a = 3.
-/
theorem collinear_points_sum (h : ∃ k : ℝ, 
  (a - 1, b - a, 3 - b) = k • (b - 1, 3 - a, a - b)) : b + a = 3 :=
sorry

end collinear_points_sum_l465_46549


namespace abs_lt_one_iff_sq_lt_one_l465_46516

variable {x : ℝ}

theorem abs_lt_one_iff_sq_lt_one : |x| < 1 ↔ x^2 < 1 := sorry

end abs_lt_one_iff_sq_lt_one_l465_46516


namespace max_sheep_pen_area_l465_46563

theorem max_sheep_pen_area :
  ∃ x y : ℝ, 15 * 2 = 30 ∧ (x + 2 * y = 30) ∧
  (x > 0 ∧ y > 0) ∧
  (x * y = 112) := by
  sorry

end max_sheep_pen_area_l465_46563


namespace min_value_f_l465_46541

def f (x y z : ℝ) : ℝ := 
  x^2 + 4 * x * y + 3 * y^2 + 2 * z^2 - 8 * x - 4 * y + 6 * z

theorem min_value_f : ∃ (x y z : ℝ), f x y z = -13.5 :=
  by
  use 1, 1.5, -1.5
  sorry

end min_value_f_l465_46541


namespace range_of_m_l465_46599

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^x 
else if 1 < x ∧ x ≤ 2 then Real.log (x - 1) 
else 0 -- function is not defined outside the given range

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 
  (x ≤ 1 → 2^x ≤ 4 - m * x) ∧ 
  (1 < x ∧ x ≤ 2 → Real.log (x - 1) ≤ 4 - m * x)) → 
  0 ≤ m ∧ m ≤ 2 := 
sorry

end range_of_m_l465_46599


namespace principal_amount_correct_l465_46577

noncomputable def initial_amount (A : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (A * 100) / (R * T + 100)

theorem principal_amount_correct : initial_amount 950 9.230769230769232 5 = 650 := by
  sorry

end principal_amount_correct_l465_46577


namespace perimeter_of_region_l465_46593

noncomputable def side_length : ℝ := 2 / Real.pi

noncomputable def semicircle_perimeter : ℝ := 2

theorem perimeter_of_region (s : ℝ) (p : ℝ) (h1 : s = 2 / Real.pi) (h2 : p = 2) :
  4 * (p / 2) = 4 :=
by
  sorry

end perimeter_of_region_l465_46593


namespace find_k_l465_46586

variable (x y z k : ℝ)

def fractions_are_equal : Prop := (9 / (x + y) = k / (x + z) ∧ k / (x + z) = 15 / (z - y))

theorem find_k (h : fractions_are_equal x y z k) : k = 24 := by
  sorry

end find_k_l465_46586


namespace john_beats_per_minute_l465_46597

theorem john_beats_per_minute :
  let hours_per_day := 2
  let days := 3
  let total_beats := 72000
  let minutes_per_hour := 60
  total_beats / (days * hours_per_day * minutes_per_hour) = 200 := 
by 
  sorry

end john_beats_per_minute_l465_46597


namespace power_mod_equiv_l465_46575

theorem power_mod_equiv :
  2^1000 % 17 = 1 := by
  sorry

end power_mod_equiv_l465_46575


namespace find_k_value_l465_46589

theorem find_k_value : 
  (∃ (x y k : ℝ), x = -6.8 ∧ 
  (y = 0.25 * x + 10) ∧ 
  (k = -3 * x + y) ∧ 
  k = 32.1) :=
sorry

end find_k_value_l465_46589


namespace inequality_solution_set_l465_46539

theorem inequality_solution_set (x : ℝ) : 
  ( (x - 1) / (x + 2) > 0 ) ↔ ( x > 1 ∨ x < -2 ) :=
by sorry

end inequality_solution_set_l465_46539


namespace theon_speed_l465_46536

theorem theon_speed (VTheon VYara D : ℕ) (h1 : VYara = 30) (h2 : D = 90) (h3 : D / VTheon = D / VYara + 3) : VTheon = 15 := by
  sorry

end theon_speed_l465_46536


namespace min_x_prime_factorization_sum_eq_31_l465_46554

theorem min_x_prime_factorization_sum_eq_31
    (x y a b c d : ℕ)
    (hx : x > 0)
    (hy : y > 0)
    (h : 7 * x^5 = 11 * y^13)
    (hx_prime_fact : ∃ a c b d : ℕ, x = a^c * b^d) :
    a + b + c + d = 31 :=
by
 sorry
 
end min_x_prime_factorization_sum_eq_31_l465_46554


namespace total_emeralds_l465_46591

theorem total_emeralds (D R E : ℕ) 
  (h1 : 2 * D + 2 * E + 2 * R = 6)
  (h2 : R = D + 15) : 
  E = 12 :=
by
  -- Proof omitted
  sorry

end total_emeralds_l465_46591


namespace group4_exceeds_group2_group4_exceeds_group3_l465_46553

-- Define conditions
def score_group1 : Int := 100
def score_group2 : Int := 150
def score_group3 : Int := -400
def score_group4 : Int := 350
def score_group5 : Int := -100

-- Theorem 1: Proving Group 4 exceeded Group 2 by 200 points
theorem group4_exceeds_group2 :
  score_group4 - score_group2 = 200 := by
  sorry

-- Theorem 2: Proving Group 4 exceeded Group 3 by 750 points
theorem group4_exceeds_group3 :
  score_group4 - score_group3 = 750 := by
  sorry

end group4_exceeds_group2_group4_exceeds_group3_l465_46553


namespace total_chickens_l465_46540

   def number_of_hens := 12
   def hens_to_roosters_ratio := 3
   def chicks_per_hen := 5

   theorem total_chickens (h : number_of_hens = 12)
                          (r : hens_to_roosters_ratio = 3)
                          (c : chicks_per_hen = 5) :
     number_of_hens + (number_of_hens / hens_to_roosters_ratio) + (number_of_hens * chicks_per_hen) = 76 :=
   by
     sorry
   
end total_chickens_l465_46540


namespace liangliang_speed_l465_46545

theorem liangliang_speed (d_initial : ℝ) (t : ℝ) (d_final : ℝ) (v_mingming : ℝ) (v_liangliang : ℝ) :
  d_initial = 3000 →
  t = 20 →
  d_final = 2900 →
  v_mingming = 80 →
  (v_liangliang = 85 ∨ v_liangliang = 75) :=
by
  sorry

end liangliang_speed_l465_46545


namespace min_sum_squares_l465_46503

noncomputable def distances (P : ℝ) : ℝ :=
  let AP := P
  let BP := |P - 1|
  let CP := |P - 2|
  let DP := |P - 5|
  let EP := |P - 13|
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_sum_squares : ∀ P : ℝ, distances P ≥ 88.2 :=
by
  sorry

end min_sum_squares_l465_46503


namespace equation_of_curve_C_range_of_m_l465_46558

theorem equation_of_curve_C (x y m : ℝ) (hx : x ≠ 0) (hm : m > 1) (k1 k2 : ℝ) 
  (h_k1 : k1 = (y - 1) / x) (h_k2 : k2 = (y + 1) / (2 * x))
  (h_prod : k1 * k2 = -1 / m^2) :
  (x^2) / (m^2) + (y^2) = 1 := 
sorry

theorem range_of_m (m : ℝ) :
  (1 < m ∧ m ≤ Real.sqrt 3)
  ∨ (m < 1 ∨ m > Real.sqrt 3) :=
sorry

end equation_of_curve_C_range_of_m_l465_46558


namespace repeating_decimal_equals_fraction_l465_46532

theorem repeating_decimal_equals_fraction : 
  let a := 58 / 100
  let r := 1 / 100
  let S := a / (1 - r)
  S = (58 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_equals_fraction_l465_46532


namespace area_of_triangle_PDE_l465_46592

noncomputable def length (a b : Point) : ℝ := -- define length between two points
sorry

def distance_from_line (P D E : Point) : ℝ := -- define perpendicular distance from P to line DE
sorry

structure Point :=
(x : ℝ)
(y : ℝ)

def area_triangle (P D E : Point) : ℝ :=
0.5 -- define area given conditions

theorem area_of_triangle_PDE (D E : Point) (hD_E : D ≠ E) :
  { P : Point | area_triangle P D E = 0.5 } =
  { P : Point | distance_from_line P D E = 1 / (length D E) } :=
sorry

end area_of_triangle_PDE_l465_46592


namespace units_digit_sum_l465_46529

theorem units_digit_sum (h₁ : (24 : ℕ) % 10 = 4) 
                        (h₂ : (42 : ℕ) % 10 = 2) : 
  ((24^3 + 42^3) % 10 = 2) :=
by
  sorry

end units_digit_sum_l465_46529


namespace necessary_but_not_sufficient_l465_46535

theorem necessary_but_not_sufficient (x y : ℝ) : 
  (x - y > -1) → (x^3 + x > x^2 * y + y) → 
  ∃ z : ℝ, z - y > -1 ∧ ¬ (z^3 + z > z^2 * y + y) :=
sorry

end necessary_but_not_sufficient_l465_46535


namespace find_dimensions_l465_46524

def is_solution (m n r : ℕ) : Prop :=
  ∃ k0 k1 k2 : ℕ, 
    k0 = (m - 2) * (n - 2) * (r - 2) ∧
    k1 = 2 * ((m - 2) * (n - 2) + (n - 2) * (r - 2) + (r - 2) * (m - 2)) ∧
    k2 = 4 * ((m - 2) + (n - 2) + (r - 2)) ∧
    k0 + k2 - k1 = 1985

theorem find_dimensions (m n r : ℕ) (h : m ≤ n ∧ n ≤ r) (hp : 0 < m ∧ 0 < n ∧ 0 < r) : 
  is_solution m n r :=
sorry

end find_dimensions_l465_46524


namespace inequality_always_true_l465_46550

theorem inequality_always_true (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b :=
by sorry

end inequality_always_true_l465_46550


namespace twice_original_price_l465_46507

theorem twice_original_price (P : ℝ) (h : 377 = 1.30 * P) : 2 * P = 580 :=
by {
  -- proof steps will go here
  sorry
}

end twice_original_price_l465_46507


namespace average_of_added_numbers_l465_46506

theorem average_of_added_numbers (sum_twelve : ℕ) (new_sum : ℕ) (x y z : ℕ) 
  (h_sum_twelve : sum_twelve = 12 * 45) 
  (h_new_sum : new_sum = 15 * 60) 
  (h_addition : x + y + z = new_sum - sum_twelve) : 
  (x + y + z) / 3 = 120 :=
by 
  sorry

end average_of_added_numbers_l465_46506


namespace bowling_average_decrease_l465_46548

/-- Represents data about the bowler's performance. -/
structure BowlerPerformance :=
(old_average : ℚ)
(last_match_runs : ℚ)
(last_match_wickets : ℕ)
(previous_wickets : ℕ)

/-- Calculates the new total runs given. -/
def new_total_runs (perf : BowlerPerformance) : ℚ :=
  perf.old_average * ↑perf.previous_wickets + perf.last_match_runs

/-- Calculates the new total number of wickets. -/
def new_total_wickets (perf : BowlerPerformance) : ℕ :=
  perf.previous_wickets + perf.last_match_wickets

/-- Calculates the new bowling average. -/
def new_average (perf : BowlerPerformance) : ℚ :=
  new_total_runs perf / ↑(new_total_wickets perf)

/-- Calculates the decrease in the bowling average. -/
def decrease_in_average (perf : BowlerPerformance) : ℚ :=
  perf.old_average - new_average perf

/-- The proof statement to be verified. -/
theorem bowling_average_decrease :
  ∀ (perf : BowlerPerformance),
    perf.old_average = 12.4 →
    perf.last_match_runs = 26 →
    perf.last_match_wickets = 6 →
    perf.previous_wickets = 115 →
    decrease_in_average perf = 0.4 :=
by
  intros
  sorry

end bowling_average_decrease_l465_46548


namespace original_price_of_coat_l465_46574

theorem original_price_of_coat (P : ℝ) (h : 0.70 * P = 350) : P = 500 :=
sorry

end original_price_of_coat_l465_46574


namespace total_spaces_in_game_l465_46533

-- Conditions
def first_turn : ℕ := 8
def second_turn_forward : ℕ := 2
def second_turn_backward : ℕ := 5
def third_turn : ℕ := 6
def total_to_end : ℕ := 37

-- Theorem stating the total number of spaces in the game
theorem total_spaces_in_game : first_turn + second_turn_forward - second_turn_backward + third_turn + (total_to_end - (first_turn + second_turn_forward - second_turn_backward + third_turn)) = total_to_end :=
by sorry

end total_spaces_in_game_l465_46533


namespace find_k_l465_46556

theorem find_k (k : ℝ) (A B : ℝ × ℝ) 
  (hA : A = (2, 3)) (hB : B = (4, k)) 
  (hAB_parallel : A.2 = B.2) : k = 3 := 
by 
  have hA_def : A = (2, 3) := hA 
  have hB_def : B = (4, k) := hB 
  have parallel_condition: A.2 = B.2 := hAB_parallel
  simp at parallel_condition
  sorry

end find_k_l465_46556


namespace integer_solutions_to_equation_l465_46513

theorem integer_solutions_to_equation :
  {p : ℤ × ℤ | (p.fst^2 - 2 * p.fst * p.snd - 3 * p.snd^2 = 5)} =
  {(4, 1), (2, -1), (-4, -1), (-2, 1)} :=
by {
  sorry
}

end integer_solutions_to_equation_l465_46513


namespace solve_for_y_l465_46505

theorem solve_for_y (x y : ℚ) (h₁ : x - y = 12) (h₂ : 2 * x + y = 10) : y = -14 / 3 :=
by
  sorry

end solve_for_y_l465_46505


namespace find_a_of_extreme_at_1_l465_46528

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - x - Real.log x

theorem find_a_of_extreme_at_1 :
  (∃ a : ℝ, ∃ f' : ℝ -> ℝ, (f' x = 3 * a * x^2 - 1 - 1/x) ∧ f' 1 = 0) →
  ∃ a : ℝ, a = 2 / 3 :=
by
  sorry

end find_a_of_extreme_at_1_l465_46528


namespace vegan_non_soy_fraction_l465_46587

theorem vegan_non_soy_fraction (total_menu : ℕ) (vegan_dishes soy_free_vegan_dish : ℕ) 
  (h1 : vegan_dishes = 6) (h2 : vegan_dishes = total_menu / 3) (h3 : soy_free_vegan_dish = vegan_dishes - 5) :
  (soy_free_vegan_dish / total_menu = 1 / 18) :=
by
  sorry

end vegan_non_soy_fraction_l465_46587


namespace ratio_of_distances_l465_46579

theorem ratio_of_distances 
  (x : ℝ) -- distance walked by the first lady
  (h1 : 4 + x = 12) -- combined total distance walked is 12 miles 
  (h2 : ¬(x < 0)) -- distance cannot be negative
  (h3 : 4 ≠ 0) : -- the second lady walked 4 miles which is not zero
  x / 4 = 2 := -- the ratio of the distances is 2
by
  sorry

end ratio_of_distances_l465_46579


namespace find_radius_of_sector_l465_46565

noncomputable def radius_of_sector (P : ℝ) (θ : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem find_radius_of_sector :
  radius_of_sector 144 180 = 144 / (Real.pi + 2) :=
by
  unfold radius_of_sector
  sorry

end find_radius_of_sector_l465_46565


namespace books_read_last_month_l465_46512

namespace BookReading

variable (W : ℕ) -- Number of books William read last month.

-- Conditions
axiom cond1 : ∃ B : ℕ, B = 3 * W -- Brad read thrice as many books as William did last month.
axiom cond2 : W = 2 * 8 -- This month, William read twice as much as Brad, who read 8 books.
axiom cond3 : ∃ (B_prev : ℕ) (B_curr : ℕ), B_prev = 3 * W ∧ B_curr = 8 ∧ W + 16 = B_prev + B_curr + 4 -- Total books equation

theorem books_read_last_month : W = 2 := by
  sorry

end BookReading

end books_read_last_month_l465_46512


namespace company_bought_14_02_tons_l465_46520

noncomputable def gravel := 5.91
noncomputable def sand := 8.11
noncomputable def total_material := gravel + sand

theorem company_bought_14_02_tons : total_material = 14.02 :=
by 
  sorry

end company_bought_14_02_tons_l465_46520


namespace ratio_of_ages_l465_46566

variable (F S : ℕ)

-- Condition 1: The product of father's age and son's age is 756
def cond1 := F * S = 756

-- Condition 2: The ratio of their ages after 6 years will be 2
def cond2 := (F + 6) / (S + 6) = 2

-- Theorem statement: The current ratio of the father's age to the son's age is 7:3
theorem ratio_of_ages (h1 : cond1 F S) (h2 : cond2 F S) : F / S = 7 / 3 :=
sorry

end ratio_of_ages_l465_46566


namespace correct_decimal_product_l465_46598

theorem correct_decimal_product : (0.125 * 3.2 = 4.0) :=
sorry

end correct_decimal_product_l465_46598


namespace algebraic_expression_value_l465_46537

theorem algebraic_expression_value (x : ℝ) (h : 3 * x^2 - 4 * x = 6): 6 * x^2 - 8 * x - 9 = 3 :=
by sorry

end algebraic_expression_value_l465_46537


namespace simplify_expression_l465_46500

theorem simplify_expression (z y : ℝ) :
  (4 - 5 * z + 2 * y) - (6 + 7 * z - 3 * y) = -2 - 12 * z + 5 * y :=
by
  sorry

end simplify_expression_l465_46500


namespace inequality_1_inequality_2_inequality_3_inequality_4_l465_46534

-- Definitions of distances
def d_a : ℝ := sorry
def d_b : ℝ := sorry
def d_c : ℝ := sorry
def R_a : ℝ := sorry
def R_b : ℝ := sorry
def R_c : ℝ := sorry
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

def R : ℝ := sorry -- Circumradius
def r : ℝ := sorry -- Inradius

-- Inequality 1
theorem inequality_1 : a * R_a ≥ c * d_c + b * d_b := 
  sorry

-- Inequality 2
theorem inequality_2 : d_a * R_a + d_b * R_b + d_c * R_c ≥ 2 * (d_a * d_b + d_b * d_c + d_c * d_a) :=
  sorry

-- Inequality 3
theorem inequality_3 : R_a + R_b + R_c ≥ 2 * (d_a + d_b + d_c) :=
  sorry

-- Inequality 4
theorem inequality_4 : R_a * R_b * R_c ≥ (R / (2 * r)) * (d_a + d_b) * (d_b + d_c) * (d_c + d_a) :=
  sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l465_46534


namespace arithmetic_sequence_nine_l465_46522

variable (a : ℕ → ℝ)
variable (d : ℝ)
-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nine (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h_cond : a 4 + a 14 = 2) : 
  a 9 = 1 := 
sorry

end arithmetic_sequence_nine_l465_46522


namespace distributive_laws_fail_for_all_l465_46581

def has_op_hash (a b : ℝ) : ℝ := a + 2 * b

theorem distributive_laws_fail_for_all (x y z : ℝ) : 
  ¬ (∀ x y z, has_op_hash x (y + z) = has_op_hash x y + has_op_hash x z) ∧
  ¬ (∀ x y z, x + has_op_hash y z = has_op_hash (x + y) (x + z)) ∧
  ¬ (∀ x y z, has_op_hash x (has_op_hash y z) = has_op_hash (has_op_hash x y) (has_op_hash x z)) := 
sorry

end distributive_laws_fail_for_all_l465_46581


namespace chance_Z_winning_l465_46531

-- Given conditions as Lean definitions
def p_x : ℚ := 1 / (3 + 1)
def p_y : ℚ := 3 / (2 + 3)
def p_z : ℚ := 1 - (p_x + p_y)

-- Theorem statement: Prove the equivalence of the winning ratio for Z
theorem chance_Z_winning : 
  p_z = 3 / (3 + 17) :=
by
  -- Since we include no proof, we use sorry to indicate it
  sorry

end chance_Z_winning_l465_46531


namespace sum_of_coefficients_l465_46542

theorem sum_of_coefficients (p : ℕ) (hp : Nat.Prime p) (a b c : ℕ)
    (f : ℕ → ℕ) (hf : ∀ x, f x = a * x ^ 2 + b * x + c)
    (h_range : 0 < a ∧ a ≤ p ∧ 0 < b ∧ b ≤ p ∧ 0 < c ∧ c ≤ p)
    (h_div : ∀ x, x > 0 → p ∣ (f x)) : 
    a + b + c = 3 * p := 
sorry

end sum_of_coefficients_l465_46542


namespace find_value_of_alpha_beta_plus_alpha_plus_beta_l465_46521

variable (α β : ℝ)

theorem find_value_of_alpha_beta_plus_alpha_plus_beta
  (hα : α^2 + α - 1 = 0)
  (hβ : β^2 + β - 1 = 0)
  (hαβ : α ≠ β) :
  α * β + α + β = -2 := 
by
  sorry

end find_value_of_alpha_beta_plus_alpha_plus_beta_l465_46521


namespace find_a_l465_46552

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 0 then x * 2^(x + a) - 1 else - (x * 2^(-x + a) - 1)

theorem find_a (a : ℝ) (h_odd: ∀ x : ℝ, f x a = -f (-x) a)
  (h_pos : ∀ x : ℝ, x > 0 → f x a = x * 2^(x + a) - 1)
  (h_neg : f (-1) a = 3 / 4) :
  a = -3 :=
by
  sorry

end find_a_l465_46552


namespace parabola_equation_and_orthogonality_l465_46584

theorem parabola_equation_and_orthogonality 
  (p : ℝ) (h_p_pos : p > 0) 
  (F : ℝ × ℝ) (h_focus : F = (p / 2, 0)) 
  (A B : ℝ × ℝ) (y : ℝ → ℝ) (C : ℝ × ℝ) 
  (h_parabola : ∀ (x y : ℝ), y^2 = 2 * p * x) 
  (h_line : ∀ (x : ℝ), y x = x - 8) 
  (h_intersect : ∃ x, y x = 0)
  (h_intersection_points : ∃ (x1 x2 : ℝ), y x1 = 0 ∧ y x2 = 0)
  (O : ℝ × ℝ) (h_origin : O = (0, 0)) 
  (h_vector_relation : 3 * F.fst = C.fst - F.fst)
  (h_C_x_axis : C = (8, 0)) :
  (p = 4 → y^2 = 8 * x) ∧ 
  (∀ (A B : ℝ × ℝ), (A.snd * B.snd = -64) ∧ 
  ((A.fst = (A.snd)^2 / 8) ∧ (B.fst = (B.snd)^2 / 8)) → 
  (A.fst * B.fst + A.snd * B.snd = 0)) := 
sorry

end parabola_equation_and_orthogonality_l465_46584


namespace sum_of_central_squares_is_34_l465_46562

-- Defining the parameters and conditions
def is_adjacent (i j : ℕ) : Prop := 
  (i = j + 1 ∨ i = j - 1 ∨ i = j + 4 ∨ i = j - 4)

def valid_matrix (M : Fin 4 → Fin 4 → ℕ) : Prop := 
  ∀ (i j : Fin 4), 
  i < 3 ∧ j < 3 → is_adjacent (M i j) (M (i + 1) j) ∧ is_adjacent (M i j) (M i (j + 1))

def corners_sum_to_34 (M : Fin 4 → Fin 4 → ℕ) : Prop :=
  M 0 0 + M 0 3 + M 3 0 + M 3 3 = 34

-- Stating the proof problem
theorem sum_of_central_squares_is_34 :
  ∃ (M : Fin 4 → Fin 4 → ℕ), valid_matrix M ∧ corners_sum_to_34 M → 
  (M 1 1 + M 1 2 + M 2 1 + M 2 2 = 34) :=
by
  sorry

end sum_of_central_squares_is_34_l465_46562


namespace percent_chemical_a_in_mixture_l465_46508

-- Define the given problem parameters
def percent_chemical_a_in_solution_x : ℝ := 0.30
def percent_chemical_a_in_solution_y : ℝ := 0.40
def proportion_of_solution_x_in_mixture : ℝ := 0.80
def proportion_of_solution_y_in_mixture : ℝ := 1.0 - proportion_of_solution_x_in_mixture

-- Define what we need to prove: the percentage of chemical a in the mixture
theorem percent_chemical_a_in_mixture:
  (percent_chemical_a_in_solution_x * proportion_of_solution_x_in_mixture) + 
  (percent_chemical_a_in_solution_y * proportion_of_solution_y_in_mixture) = 0.32 
:= by sorry

end percent_chemical_a_in_mixture_l465_46508


namespace one_fourth_difference_l465_46570

theorem one_fourth_difference :
  (1 / 4) * ((9 * 5) - (7 + 3)) = 35 / 4 :=
by sorry

end one_fourth_difference_l465_46570


namespace smallest_n_for_sqrt_50n_is_integer_l465_46576

theorem smallest_n_for_sqrt_50n_is_integer :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (50 * n) = k * k) ∧ n = 2 :=
by
  sorry

end smallest_n_for_sqrt_50n_is_integer_l465_46576


namespace profit_23_percent_of_cost_price_l465_46544

-- Define the conditions
variable (C : ℝ) -- Cost price of the turtleneck sweaters
variable (C_nonneg : 0 ≤ C) -- Ensure cost price is non-negative

-- Definitions based on conditions
def SP1 (C : ℝ) : ℝ := 1.20 * C
def SP2 (SP1 : ℝ) : ℝ := 1.25 * SP1
def SPF (SP2 : ℝ) : ℝ := 0.82 * SP2

-- Define the profit calculation
def Profit (C : ℝ) : ℝ := (SPF (SP2 (SP1 C))) - C

-- Statement of the theorem
theorem profit_23_percent_of_cost_price (C : ℝ) (C_nonneg : 0 ≤ C):
  Profit C = 0.23 * C :=
by
  -- The actual proof would go here
  sorry

end profit_23_percent_of_cost_price_l465_46544


namespace transform_sequence_zero_l465_46564

theorem transform_sequence_zero 
  (n : ℕ) 
  (a : ℕ → ℝ) 
  (h_nonempty : n > 0) :
  ∃ k : ℕ, k ≤ n ∧ ∀ k' ≤ k, ∃ α : ℝ, (∀ i, i < n → |a i - α| = 0) := 
sorry

end transform_sequence_zero_l465_46564


namespace f_is_odd_and_periodic_l465_46546

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (10 + x) = f (10 - x)
axiom h2 : ∀ x : ℝ, f (20 - x) = -f (20 + x)

theorem f_is_odd_and_periodic : 
  (∀ x : ℝ, f (-x) = -f x) ∧ (∃ T : ℝ, T = 40 ∧ ∀ x : ℝ, f (x + T) = f x) :=
by
  sorry

end f_is_odd_and_periodic_l465_46546


namespace find_prime_pairs_l465_46580

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_pair (p q : ℕ) : Prop := 
  p < 2023 ∧ q < 2023 ∧ 
  p ∣ q^2 + 8 ∧ q ∣ p^2 + 8

theorem find_prime_pairs : 
  ∀ (p q : ℕ), is_prime p → is_prime q → valid_pair p q → 
    (p = 2 ∧ q = 2) ∨ 
    (p = 17 ∧ q = 3) ∨ 
    (p = 11 ∧ q = 5) :=
by 
  sorry

end find_prime_pairs_l465_46580


namespace garage_motorcycles_l465_46588

theorem garage_motorcycles (bicycles cars motorcycles total_wheels : ℕ)
  (hb : bicycles = 20)
  (hc : cars = 10)
  (hw : total_wheels = 90)
  (wb : bicycles * 2 = 40)
  (wc : cars * 4 = 40)
  (wm : motorcycles * 2 = total_wheels - (bicycles * 2 + cars * 4)) :
  motorcycles = 5 := 
  by 
  sorry

end garage_motorcycles_l465_46588


namespace first_number_is_48_l465_46502

-- Definitions of the conditions
def ratio (A B : ℕ) := 8 * B = 9 * A
def lcm (A B : ℕ) := Nat.lcm A B = 432

-- The statement to prove
theorem first_number_is_48 (A B : ℕ) (h_ratio : ratio A B) (h_lcm : lcm A B) : A = 48 :=
by
  sorry

end first_number_is_48_l465_46502


namespace wire_ratio_bonnie_roark_l465_46582

-- Definitions from the conditions
def bonnie_wire_length : ℕ := 12 * 8
def bonnie_volume : ℕ := 8 ^ 3
def roark_cube_side : ℕ := 2
def roark_cube_volume : ℕ := roark_cube_side ^ 3
def num_roark_cubes : ℕ := bonnie_volume / roark_cube_volume
def roark_wire_length_per_cube : ℕ := 12 * roark_cube_side
def roark_total_wire_length : ℕ := num_roark_cubes * roark_wire_length_per_cube

-- Statement to prove
theorem wire_ratio_bonnie_roark : 
  ((bonnie_wire_length : ℚ) / roark_total_wire_length) = (1 / 16) :=
by
  sorry

end wire_ratio_bonnie_roark_l465_46582


namespace gain_percentage_l465_46547

theorem gain_percentage (x : ℝ) (CP : ℝ := 50 * x) (SP : ℝ := 60 * x) (Profit : ℝ := 10 * x) :
  ((Profit / CP) * 100) = 20 := 
by
  sorry

end gain_percentage_l465_46547


namespace jacob_age_proof_l465_46530

theorem jacob_age_proof
  (drew_age maya_age peter_age : ℕ)
  (john_age : ℕ := 30)
  (jacob_age : ℕ) :
  (drew_age = maya_age + 5) →
  (peter_age = drew_age + 4) →
  (john_age = 30 ∧ john_age = 2 * maya_age) →
  (jacob_age + 2 = (peter_age + 2) / 2) →
  jacob_age = 11 :=
by
  sorry

end jacob_age_proof_l465_46530


namespace peaches_division_l465_46561

theorem peaches_division (n k r : ℕ) 
  (h₁ : 100 = n * k + 10)
  (h₂ : 1000 = n * k * 11 + r) :
  r = 10 :=
by sorry

end peaches_division_l465_46561


namespace profit_percent_is_approx_6_point_35_l465_46511

noncomputable def selling_price : ℝ := 2552.36
noncomputable def cost_price : ℝ := 2400
noncomputable def profit_amount : ℝ := selling_price - cost_price
noncomputable def profit_percent : ℝ := (profit_amount / cost_price) * 100

theorem profit_percent_is_approx_6_point_35 : abs (profit_percent - 6.35) < 0.01 := sorry

end profit_percent_is_approx_6_point_35_l465_46511


namespace Yanna_apples_l465_46501

def total_apples_bought (given_to_zenny : ℕ) (given_to_andrea : ℕ) (kept : ℕ) : ℕ :=
  given_to_zenny + given_to_andrea + kept

theorem Yanna_apples {given_to_zenny given_to_andrea kept total : ℕ}:
  given_to_zenny = 18 →
  given_to_andrea = 6 →
  kept = 36 →
  total_apples_bought given_to_zenny given_to_andrea kept = 60 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end Yanna_apples_l465_46501


namespace ring_cost_l465_46572

theorem ring_cost (total_cost : ℕ) (rings : ℕ) (h1 : total_cost = 24) (h2 : rings = 2) : total_cost / rings = 12 :=
by
  sorry

end ring_cost_l465_46572


namespace or_is_true_given_p_true_q_false_l465_46568

theorem or_is_true_given_p_true_q_false (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q :=
by
  sorry

end or_is_true_given_p_true_q_false_l465_46568


namespace avg_age_increase_l465_46595

theorem avg_age_increase 
    (student_count : ℕ) (avg_student_age : ℕ) (teacher_age : ℕ) (new_count : ℕ) (new_avg_age : ℕ) (age_increase : ℕ)
    (hc1 : student_count = 23)
    (hc2 : avg_student_age = 22)
    (hc3 : teacher_age = 46)
    (hc4 : new_count = student_count + 1)
    (hc5 : new_avg_age = ((avg_student_age * student_count + teacher_age) / new_count))
    (hc6 : age_increase = new_avg_age - avg_student_age) :
  age_increase = 1 := 
sorry

end avg_age_increase_l465_46595


namespace power_function_value_at_minus_two_l465_46578

-- Define the power function assumption and points
variable (f : ℝ → ℝ)
variable (hf : f (1 / 2) = 8)

-- Prove that the given condition implies the required result
theorem power_function_value_at_minus_two : f (-2) = -1 / 8 := 
by {
  -- proof to be filled here
  sorry
}

end power_function_value_at_minus_two_l465_46578


namespace crocodile_length_in_meters_l465_46551

-- Definitions based on conditions
def ken_to_cm : ℕ := 180
def shaku_to_cm : ℕ := 30
def ken_to_shaku : ℕ := 6
def cm_to_m : ℕ := 100

-- Lengths given in the problem expressed in ken
def head_to_tail_in_ken (L : ℚ) : Prop := 3 * L = 10
def tail_to_head_in_ken (L : ℚ) : Prop := L = (3 + (2 / ken_to_shaku : ℚ))

-- Final length conversion to meters
def length_in_m (L : ℚ) : ℚ := L * ken_to_cm / cm_to_m

-- The length of the crocodile in meters
theorem crocodile_length_in_meters (L : ℚ) : head_to_tail_in_ken L → tail_to_head_in_ken L → length_in_m L = 6 :=
by
  intros _ _
  sorry

end crocodile_length_in_meters_l465_46551


namespace paper_cut_square_l465_46560

noncomputable def proof_paper_cut_square : Prop :=
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ ((2 * x - 2 = 2 - x) ∨ (2 * (2 * x - 2) = 2 - x)) ∧ (x = 1.2 ∨ x = 1.5)

theorem paper_cut_square : proof_paper_cut_square :=
sorry

end paper_cut_square_l465_46560


namespace initial_books_in_library_l465_46509

theorem initial_books_in_library 
  (initial_books : ℕ)
  (books_taken_out_Tuesday : ℕ := 120)
  (books_returned_Wednesday : ℕ := 35)
  (books_withdrawn_Thursday : ℕ := 15)
  (books_final_count : ℕ := 150)
  : initial_books - books_taken_out_Tuesday + books_returned_Wednesday - books_withdrawn_Thursday = books_final_count → initial_books = 250 :=
by
  intros h
  sorry

end initial_books_in_library_l465_46509


namespace geometric_sequence_general_formula_l465_46517

theorem geometric_sequence_general_formula (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h1 : a 1 = 2)
  (h_rec : ∀ n, (a (n + 2))^2 + 4 * (a n)^2 = 4 * (a (n + 1))^2) :
  ∀ n, a n = 2^(n + 1) / 2 := 
sorry

end geometric_sequence_general_formula_l465_46517


namespace smallest_three_digit_multiple_of_13_l465_46510

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 13 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 13 = 0 → n ≤ m :=
⟨104, by sorry⟩

end smallest_three_digit_multiple_of_13_l465_46510


namespace proof_problem_l465_46504

variable (p1 p2 p3 p4 : Prop)

theorem proof_problem (hp1 : p1) (hp2 : ¬ p2) (hp3 : ¬ p3) (hp4 : p4) :
  (p1 ∧ p4) ∧ (¬ p2 ∨ p3) ∧ (¬ p3 ∨ ¬ p4) := by
  sorry

end proof_problem_l465_46504


namespace percentage_increase_in_expenses_l465_46567

theorem percentage_increase_in_expenses:
  ∀ (S : ℝ) (original_save_percentage new_savings : ℝ), 
  S = 5750 → 
  original_save_percentage = 0.20 →
  new_savings = 230 →
  (original_save_percentage * S - new_savings) / (S - original_save_percentage * S) * 100 = 20 :=
by
  intros S original_save_percentage new_savings HS Horiginal_save_percentage Hnew_savings
  rw [HS, Horiginal_save_percentage, Hnew_savings]
  sorry

end percentage_increase_in_expenses_l465_46567


namespace translation_of_exponential_l465_46596

noncomputable def translated_function (a : ℝ × ℝ) (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => f (x - a.1) + a.2

theorem translation_of_exponential :
  translated_function (2, 3) (λ x => Real.exp x) = λ x => Real.exp (x - 2) + 3 :=
by
  sorry

end translation_of_exponential_l465_46596


namespace ax_by_powers_l465_46557

theorem ax_by_powers (a b x y : ℝ) (h1 : a * x + b * y = 5) 
                      (h2: a * x^2 + b * y^2 = 11)
                      (h3: a * x^3 + b * y^3 = 25)
                      (h4: a * x^4 + b * y^4 = 59) : 
                      a * x^5 + b * y^5 = 145 := 
by 
  -- Include the proof steps here if needed 
  sorry

end ax_by_powers_l465_46557


namespace RouteB_quicker_than_RouteA_l465_46543

def RouteA_segment1_time : ℚ := 4 / 40 -- time in hours
def RouteA_segment2_time : ℚ := 4 / 20 -- time in hours
def RouteA_total_time : ℚ := RouteA_segment1_time + RouteA_segment2_time -- total time in hours

def RouteB_segment1_time : ℚ := 6 / 35 -- time in hours
def RouteB_segment2_time : ℚ := 1 / 15 -- time in hours
def RouteB_total_time : ℚ := RouteB_segment1_time + RouteB_segment2_time -- total time in hours

def time_difference_minutes : ℚ := (RouteA_total_time - RouteB_total_time) * 60 -- difference in minutes

theorem RouteB_quicker_than_RouteA : time_difference_minutes = 3.71 := by
  sorry

end RouteB_quicker_than_RouteA_l465_46543


namespace variable_value_l465_46525

theorem variable_value 
  (x : ℝ)
  (a k some_variable : ℝ)
  (eqn1 : (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + some_variable)
  (eqn2 : a - some_variable + k = 3)
  (a_val : a = 6)
  (k_val : k = -17) :
  some_variable = -14 :=
by
  sorry

end variable_value_l465_46525


namespace prove_absolute_value_subtract_power_l465_46514

noncomputable def smallest_absolute_value : ℝ := 0

theorem prove_absolute_value_subtract_power (b : ℝ) 
  (h1 : smallest_absolute_value = 0) 
  (h2 : b * b = 1) : 
  (|smallest_absolute_value - 2| - b ^ 2023 = 1) 
  ∨ (|smallest_absolute_value - 2| - b ^ 2023 = 3) :=
sorry

end prove_absolute_value_subtract_power_l465_46514


namespace cheese_bread_grams_l465_46527

/-- Each 100 grams of cheese bread costs 3.20 BRL and corresponds to 10 pieces. 
Each person eats, on average, 5 pieces of cheese bread. Including the professor,
there are 16 students, 1 monitor, and 5 parents, making a total of 23 people. 
The precision of the bakery's scale is 100 grams. -/
theorem cheese_bread_grams : (5 * 23 / 10) * 100 = 1200 := 
by
  sorry

end cheese_bread_grams_l465_46527


namespace ratio_of_areas_l465_46585

-- Definition of sides and given condition
variables {a b c d : ℝ}
-- Given condition in the problem.
axiom condition : a / c = 3 / 5 ∧ b / d = 3 / 5

-- Statement of the theorem to be proved in Lean 4
theorem ratio_of_areas (h : a / c = 3 / 5) (h' : b / d = 3 / 5) : (a * b) / (c * d) = 9 / 25 :=
by sorry

end ratio_of_areas_l465_46585


namespace cos_seven_pi_over_six_l465_46518

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l465_46518


namespace incorrect_value_in_polynomial_progression_l465_46583

noncomputable def polynomial_values (x : ℕ) : ℕ :=
  match x with
  | 0 => 1
  | 1 => 9
  | 2 => 35
  | 3 => 99
  | 4 => 225
  | 5 => 441
  | 6 => 784
  | 7 => 1296
  | _ => 0  -- This is a dummy value just to complete the function

theorem incorrect_value_in_polynomial_progression :
  ¬ (∃ (a b c d : ℝ), ∀ x : ℕ,
    polynomial_values x = (a * x ^ 3 + b * x ^ 2 + c * x + d + if x ≤ 7 then 0 else 1)) :=
by
  intro h
  sorry

end incorrect_value_in_polynomial_progression_l465_46583


namespace find_some_value_l465_46555

theorem find_some_value (m n : ℝ) (some_value : ℝ) 
  (h₁ : m = n / 2 - 2 / 5)
  (h₂ : m + 2 = (n + some_value) / 2 - 2 / 5) :
  some_value = 4 := 
sorry

end find_some_value_l465_46555


namespace initial_roses_l465_46515

theorem initial_roses (x : ℕ) (h : x - 2 + 32 = 41) : x = 11 :=
sorry

end initial_roses_l465_46515


namespace find_x_when_y_is_sqrt_8_l465_46571

theorem find_x_when_y_is_sqrt_8
  (x y : ℝ)
  (h : ∀ x y : ℝ, (x^2 * y^4 = 1600) ↔ (x = 10 ∧ y = 2)) :
  x = 5 :=
by
  sorry

end find_x_when_y_is_sqrt_8_l465_46571


namespace pages_remaining_total_l465_46559

-- Define the conditions
def total_pages_book1 : ℕ := 563
def read_pages_book1 : ℕ := 147

def total_pages_book2 : ℕ := 849
def read_pages_book2 : ℕ := 389

def total_pages_book3 : ℕ := 700
def read_pages_book3 : ℕ := 134

-- The theorem to be proved
theorem pages_remaining_total :
  (total_pages_book1 - read_pages_book1) + 
  (total_pages_book2 - read_pages_book2) + 
  (total_pages_book3 - read_pages_book3) = 1442 := 
by
  sorry

end pages_remaining_total_l465_46559


namespace tank_leak_time_l465_46594

/--
The rate at which the tank is filled without a leak is R = 1/5 tank per hour.
The effective rate with the leak is 1/6 tank per hour.
Prove that the time it takes for the leak to empty the full tank is 30 hours.
-/
theorem tank_leak_time (R : ℝ) (L : ℝ) (h1 : R = 1 / 5) (h2 : R - L = 1 / 6) :
  1 / L = 30 :=
by
  sorry

end tank_leak_time_l465_46594


namespace smallest_even_number_of_seven_l465_46590

-- Conditions: The sum of seven consecutive even numbers is 700.
-- We need to prove that the smallest of these numbers is 94.

theorem smallest_even_number_of_seven (n : ℕ) (hn : 7 * n = 700) :
  ∃ (a b c d e f g : ℕ), 
  (2 * a + 4 * b + 6 * c + 8 * d + 10 * e + 12 * f + 14 * g = 700) ∧ 
  (a = b - 1) ∧ (b = c - 1) ∧ (c = d - 1) ∧ (d = e - 1) ∧ (e = f - 1) ∧ 
  (f = g - 1) ∧ (g = 100) ∧ (a = 94) :=
by
  -- This is the theorem statement. 
  sorry

end smallest_even_number_of_seven_l465_46590


namespace complete_the_square_l465_46519

theorem complete_the_square : ∀ x : ℝ, x^2 - 6 * x + 4 = 0 → (x - 3)^2 = 5 :=
by
  intro x h
  sorry

end complete_the_square_l465_46519


namespace arithmetic_sequence_a11_l465_46573

theorem arithmetic_sequence_a11 (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 2) - a n = 6) : 
  a 11 = 31 := 
sorry

end arithmetic_sequence_a11_l465_46573
