import Mathlib

namespace equal_cost_per_copy_l2027_202793

theorem equal_cost_per_copy 
    (x : ℕ) 
    (h₁ : 2000 % x = 0) 
    (h₂ : 3000 % (x + 50) = 0) 
    (h₃ : 2000 / x = 3000 / (x + 50)) :
    (2000 : ℕ) / x = (3000 : ℕ) / (x + 50) :=
by
  sorry

end equal_cost_per_copy_l2027_202793


namespace remainder_when_divided_by_DE_l2027_202715

theorem remainder_when_divided_by_DE (P D E Q R M S C : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = E * M + S) :
  (∃ quotient : ℕ, P = quotient * (D * E) + (S * D + R + C)) :=
by {
  sorry
}

end remainder_when_divided_by_DE_l2027_202715


namespace inequality_proof_l2027_202747

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := by
  sorry

end inequality_proof_l2027_202747


namespace find_multiple_of_son_age_l2027_202708

variable (F S k : ℕ)

theorem find_multiple_of_son_age
  (h1 : F = k * S + 4)
  (h2 : F + 4 = 2 * (S + 4) + 20)
  (h3 : F = 44) :
  k = 4 :=
by
  sorry

end find_multiple_of_son_age_l2027_202708


namespace arccos_one_eq_zero_l2027_202774

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l2027_202774


namespace smallest_n_exists_l2027_202726

theorem smallest_n_exists (n : ℕ) (h : n ≥ 4) :
  (∃ (S : Finset ℤ), S.card = n ∧
    (∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
        (a + b - c - d) % 20 = 0))
  ↔ n = 9 := sorry

end smallest_n_exists_l2027_202726


namespace perpendicular_lines_slope_l2027_202786

theorem perpendicular_lines_slope :
  ∀ (a : ℚ), (∀ x y : ℚ, y = 3 * x + 5) 
  ∧ (∀ x y : ℚ, 4 * y + a * x = 8) →
  a = 4 / 3 :=
by
  intro a
  intro h
  sorry

end perpendicular_lines_slope_l2027_202786


namespace ineq_condition_l2027_202717

theorem ineq_condition (a b : ℝ) : (a + 1 > b - 2) ↔ (a > b - 3 ∧ ¬(a > b)) :=
by
  sorry

end ineq_condition_l2027_202717


namespace pen_cost_difference_l2027_202713

theorem pen_cost_difference :
  ∀ (P : ℕ), (P + 2 = 13) → (P - 2 = 9) :=
by
  intro P
  intro h
  sorry

end pen_cost_difference_l2027_202713


namespace cube_faces_consecutive_sum_l2027_202772

noncomputable def cube_face_sum (n : ℕ) : ℕ :=
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)

theorem cube_faces_consecutive_sum (n : ℕ) (h1 : ∀ i, i ∈ [0, 5] -> (2 * n + 5 + n + 5 - 6) = 6) (h2 : n = 12) :
  cube_face_sum n = 87 :=
  sorry

end cube_faces_consecutive_sum_l2027_202772


namespace interest_rate_calculation_l2027_202718

theorem interest_rate_calculation :
  let P := 1599.9999999999998
  let A := 1792
  let T := 2 + 2 / 5
  let I := A - P
  I / (P * T) = 0.05 :=
  sorry

end interest_rate_calculation_l2027_202718


namespace smallest_difference_l2027_202789

variable (DE EF FD : ℕ)

def is_valid_triangle (DE EF FD : ℕ) : Prop :=
  DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

theorem smallest_difference (h1 : DE < EF)
                           (h2 : EF ≤ FD)
                           (h3 : DE + EF + FD = 1024)
                           (h4 : is_valid_triangle DE EF FD) :
  ∃ d, d = EF - DE ∧ d = 1 :=
by
  sorry

end smallest_difference_l2027_202789


namespace smallest_n_l2027_202781

theorem smallest_n (n : ℕ) (h₁ : ∃ k1 : ℕ, 4 * n = k1 ^ 2) (h₂ : ∃ k2 : ℕ, 3 * n = k2 ^ 3) : n = 18 :=
sorry

end smallest_n_l2027_202781


namespace ratio_of_ages_l2027_202782

noncomputable def ratio_4th_to_3rd (age1 age2 age3 age4 age5 : ℕ) : ℚ :=
  age4 / age3

theorem ratio_of_ages
  (age1 age2 age3 age4 age5 : ℕ)
  (h1 : (age1 + age5) / 2 = 18)
  (h2 : age1 = 10)
  (h3 : age2 = age1 - 2)
  (h4 : age3 = age2 + 4)
  (h5 : age4 = age3 / 2)
  (h6 : age5 = age4 + 20) :
  ratio_4th_to_3rd age1 age2 age3 age4 age5 = 1 / 2 :=
by
  sorry

end ratio_of_ages_l2027_202782


namespace three_digit_odd_sum_count_l2027_202707

def countOddSumDigits : Nat :=
  -- Count of three-digit numbers with an odd sum formed by (1, 2, 3, 4, 5)
  24

theorem three_digit_odd_sum_count :
  -- Guarantees that the count of three-digit numbers meeting the criteria is 24
  ∃ n : Nat, n = countOddSumDigits :=
by
  use 24
  sorry

end three_digit_odd_sum_count_l2027_202707


namespace oranges_in_box_l2027_202780

theorem oranges_in_box :
  ∃ (A P O : ℕ), A + P + O = 60 ∧ A = 3 * (P + O) ∧ P = (A + O) / 5 ∧ O = 5 :=
by
  sorry

end oranges_in_box_l2027_202780


namespace large_cube_surface_area_l2027_202742

-- Define given conditions
def small_cube_volume := 512 -- volume in cm^3
def num_small_cubes := 8

-- Define side length of small cube
def small_cube_side_length := (small_cube_volume : ℝ)^(1/3)

-- Define side length of large cube
def large_cube_side_length := 2 * small_cube_side_length

-- Surface area formula for a cube
def surface_area (side_length : ℝ) := 6 * side_length^2

-- Theorem: The surface area of the large cube is 1536 cm^2
theorem large_cube_surface_area :
  surface_area large_cube_side_length = 1536 :=
sorry

end large_cube_surface_area_l2027_202742


namespace no_integer_solutions_l2027_202702

theorem no_integer_solutions (x y z : ℤ) (h₀ : x ≠ 0) : ¬(2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) :=
sorry

end no_integer_solutions_l2027_202702


namespace number_of_students_third_l2027_202798

-- Define the ratio and the total number of samples.
def ratio_first : ℕ := 3
def ratio_second : ℕ := 3
def ratio_third : ℕ := 4
def total_sample : ℕ := 50

-- Define the condition that the sum of ratios equals the total proportion numerator.
def sum_ratios : ℕ := ratio_first + ratio_second + ratio_third

-- Final proposition: the number of students to be sampled from the third grade.
theorem number_of_students_third :
  (ratio_third * total_sample) / sum_ratios = 20 := by
  sorry

end number_of_students_third_l2027_202798


namespace tan_alpha_eq_two_and_expression_value_sin_tan_simplify_l2027_202731

-- First problem: Given condition and expression to be proved equal to the correct answer.
theorem tan_alpha_eq_two_and_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (2 * Real.pi - α) + Real.cos (Real.pi + α)) / 
  (Real.cos (α - Real.pi) - Real.cos (3 * Real.pi / 2 - α)) = -3 := sorry

-- Second problem: Given expression to be proved simplified to the correct answer.
theorem sin_tan_simplify :
  Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi/180)) = 1 := sorry

end tan_alpha_eq_two_and_expression_value_sin_tan_simplify_l2027_202731


namespace household_member_count_l2027_202740

variable (M : ℕ) -- the number of members in the household

-- Conditions
def slices_per_breakfast := 3
def slices_per_snack := 2
def slices_per_member_daily := slices_per_breakfast + slices_per_snack
def slices_per_loaf := 12
def loaves_last_days := 3
def loaves_given := 5
def total_slices := slices_per_loaf * loaves_given
def daily_consumption := total_slices / loaves_last_days

-- Proof statement
theorem household_member_count : daily_consumption = slices_per_member_daily * M → M = 4 :=
by
  sorry

end household_member_count_l2027_202740


namespace largest_initial_number_l2027_202741

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l2027_202741


namespace ratio_side_length_to_brush_width_l2027_202794

theorem ratio_side_length_to_brush_width (s w : ℝ) (h1 : w = s / 4) (h2 : s^2 / 3 = w^2 + ((s - w)^2) / 2) :
    s / w = 4 := by
  sorry

end ratio_side_length_to_brush_width_l2027_202794


namespace cone_height_l2027_202725

theorem cone_height (r_sector : ℝ) (θ_sector : ℝ) :
  r_sector = 3 → θ_sector = (2 * Real.pi / 3) → 
  ∃ (h : ℝ), h = 2 * Real.sqrt 2 := 
by 
  intros r_sector_eq θ_sector_eq
  sorry

end cone_height_l2027_202725


namespace problem_statement_l2027_202750

theorem problem_statement (f : ℝ → ℝ) (hf_odd : ∀ x, f (-x) = - f x)
  (hf_deriv : ∀ x < 0, 2 * f x + x * deriv f x < 0) :
  f 1 < 2016 * f (Real.sqrt 2016) ∧ 2016 * f (Real.sqrt 2016) < 2017 * f (Real.sqrt 2017) := 
  sorry

end problem_statement_l2027_202750


namespace no_common_solution_general_case_l2027_202745

-- Define the context: three linear equations in two variables
variables {a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℝ}

-- Statement of the theorem
theorem no_common_solution_general_case :
  (∃ (x y : ℝ), a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2 ∧ a3 * x + b3 * y = c3) →
  (a1 * b2 ≠ a2 * b1 ∧ a1 * b3 ≠ a3 * b1 ∧ a2 * b3 ≠ a3 * b2) →
  false := 
sorry

end no_common_solution_general_case_l2027_202745


namespace vertical_asymptotes_sum_l2027_202728

theorem vertical_asymptotes_sum (A B C : ℤ)
  (h : ∀ x : ℝ, x = -1 ∨ x = 2 ∨ x = 3 → x^3 + A * x^2 + B * x + C = 0)
  : A + B + C = -3 :=
sorry

end vertical_asymptotes_sum_l2027_202728


namespace find_x_l2027_202709

theorem find_x (x y : ℝ) (h₁ : 2 * x - y = 14) (h₂ : y = 2) : x = 8 :=
by
  sorry

end find_x_l2027_202709


namespace number_of_days_A_to_finish_remaining_work_l2027_202765

theorem number_of_days_A_to_finish_remaining_work
  (A_days : ℕ) (B_days : ℕ) (B_work_days : ℕ) : 
  A_days = 9 → 
  B_days = 15 → 
  B_work_days = 10 → 
  ∃ d : ℕ, d = 3 :=
by 
  intros hA hB hBw
  sorry

end number_of_days_A_to_finish_remaining_work_l2027_202765


namespace total_people_on_hike_l2027_202758

theorem total_people_on_hike
  (cars : ℕ) (cars_people : ℕ)
  (taxis : ℕ) (taxis_people : ℕ)
  (vans : ℕ) (vans_people : ℕ)
  (buses : ℕ) (buses_people : ℕ)
  (minibuses : ℕ) (minibuses_people : ℕ)
  (h_cars : cars = 7) (h_cars_people : cars_people = 4)
  (h_taxis : taxis = 10) (h_taxis_people : taxis_people = 6)
  (h_vans : vans = 4) (h_vans_people : vans_people = 5)
  (h_buses : buses = 3) (h_buses_people : buses_people = 20)
  (h_minibuses : minibuses = 2) (h_minibuses_people : minibuses_people = 8) :
  cars * cars_people + taxis * taxis_people + vans * vans_people + buses * buses_people + minibuses * minibuses_people = 184 :=
by
  sorry

end total_people_on_hike_l2027_202758


namespace last_digit_7_powers_l2027_202732

theorem last_digit_7_powers :
  (∃ n : ℕ, (∀ k < 4004, k.mod 2002 == n))
  := sorry

end last_digit_7_powers_l2027_202732


namespace quadratic_roots_m_eq_2_quadratic_discriminant_pos_l2027_202736

theorem quadratic_roots_m_eq_2 (x : ℝ) (m : ℝ) (h1 : m = 2) : x^2 + 2 * x - 3 = 0 ↔ (x = -3 ∨ x = 1) :=
by sorry

theorem quadratic_discriminant_pos (m : ℝ) : m^2 + 12 > 0 :=
by sorry

end quadratic_roots_m_eq_2_quadratic_discriminant_pos_l2027_202736


namespace molecular_weight_of_3_moles_l2027_202778

namespace AscorbicAcid

def molecular_form : List (String × ℕ) := [("C", 6), ("H", 8), ("O", 6)]

def atomic_weight : String → ℝ
| "C" => 12.01
| "H" => 1.008
| "O" => 16.00
| _ => 0

noncomputable def molecular_weight (molecular_form : List (String × ℕ)) : ℝ :=
molecular_form.foldr (λ (x : (String × ℕ)) acc => acc + (x.snd * atomic_weight x.fst)) 0

noncomputable def weight_of_3_moles (mw : ℝ) : ℝ := mw * 3

theorem molecular_weight_of_3_moles :
  weight_of_3_moles (molecular_weight molecular_form) = 528.372 :=
by
  sorry

end AscorbicAcid

end molecular_weight_of_3_moles_l2027_202778


namespace lucas_fraction_to_emma_l2027_202703

variable (n : ℕ)

-- Define initial stickers
def noah_stickers := n
def emma_stickers := 3 * n
def lucas_stickers := 12 * n

-- Define the final state where each has the same number of stickers
def final_stickers_per_person := (16 * n) / 3

-- Lucas gives some stickers to Emma. Calculate the fraction of Lucas's stickers given to Emma
theorem lucas_fraction_to_emma :
  (7 * n / 3) / (12 * n) = 7 / 36 := by
  sorry

end lucas_fraction_to_emma_l2027_202703


namespace common_root_value_l2027_202762

theorem common_root_value (a : ℝ) : 
  (∃ x : ℝ, x^2 + a * x + 8 = 0 ∧ x^2 + x + a = 0) ↔ a = -6 :=
sorry

end common_root_value_l2027_202762


namespace convex_quadrilateral_area_lt_a_sq_l2027_202767

theorem convex_quadrilateral_area_lt_a_sq {a x y z t : ℝ} (hx : x < a) (hy : y < a) (hz : z < a) (ht : t < a) :
  (∃ S : ℝ, S < a^2) :=
sorry

end convex_quadrilateral_area_lt_a_sq_l2027_202767


namespace greatest_ABCBA_divisible_by_13_l2027_202785

theorem greatest_ABCBA_divisible_by_13 :
  ∃ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  1 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ 0 ≤ C ∧ C < 10 ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 13 = 0 ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) = 95159 :=
by
  sorry

end greatest_ABCBA_divisible_by_13_l2027_202785


namespace remove_wallpaper_time_l2027_202735

theorem remove_wallpaper_time 
    (total_walls : ℕ := 8)
    (remaining_walls : ℕ := 7)
    (time_for_remaining_walls : ℕ := 14) :
    time_for_remaining_walls / remaining_walls = 2 :=
by
sorry

end remove_wallpaper_time_l2027_202735


namespace correct_factorization_l2027_202761

theorem correct_factorization (x m n a : ℝ) : 
  (¬ (x^2 + 2 * x + 1 = x * (x + 2) + 1)) ∧
  (¬ (m^2 - 2 * m * n + n^2 = (m + n)^2)) ∧
  (¬ (-a^4 + 16 = -(a^2 + 4) * (a^2 - 4))) ∧
  (x^3 - 4 * x = x * (x + 2) * (x - 2)) :=
by
  sorry

end correct_factorization_l2027_202761


namespace min_vertical_segment_length_l2027_202753

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := -x^2 - 4 * x - 3
def L (x : ℝ) : ℝ := f x - g x

theorem min_vertical_segment_length : ∃ (x : ℝ), L x = 10 :=
by
  sorry

end min_vertical_segment_length_l2027_202753


namespace sum_of_remainders_l2027_202775

theorem sum_of_remainders (a b c d : ℤ) (h1 : a % 53 = 33) (h2 : b % 53 = 25) (h3 : c % 53 = 6) (h4 : d % 53 = 12) : 
  (a + b + c + d) % 53 = 23 :=
by {
  sorry
}

end sum_of_remainders_l2027_202775


namespace parabola_axis_of_symmetry_is_x_eq_1_l2027_202795

theorem parabola_axis_of_symmetry_is_x_eq_1 :
  ∀ x : ℝ, ∀ y : ℝ, y = -2 * (x - 1)^2 + 3 → (∀ c : ℝ, c = 1 → ∃ x1 x2 : ℝ, x1 = c ∧ x2 = c) := 
by
  sorry

end parabola_axis_of_symmetry_is_x_eq_1_l2027_202795


namespace no_real_roots_range_k_l2027_202756

theorem no_real_roots_range_k (k : ℝ) : (x^2 - 2 * x - k = 0) ∧ (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) → k < -1 := 
by
  sorry

end no_real_roots_range_k_l2027_202756


namespace amount_borrowed_l2027_202746

variable (P : ℝ)
variable (interest_paid : ℝ) -- Interest paid on borrowing
variable (interest_earned : ℝ) -- Interest earned on lending
variable (gain_per_year : ℝ)

variable (h1 : interest_paid = P * 4 * 2 / 100)
variable (h2 : interest_earned = P * 6 * 2 / 100)
variable (h3 : gain_per_year = 160)
variable (h4 : gain_per_year = (interest_earned - interest_paid) / 2)

theorem amount_borrowed : P = 8000 := by
  sorry

end amount_borrowed_l2027_202746


namespace sum_of_numbers_l2027_202749

theorem sum_of_numbers :
  15.58 + 21.32 + 642.51 + 51.51 = 730.92 := 
  by
  sorry

end sum_of_numbers_l2027_202749


namespace milk_for_9_cookies_l2027_202797

def quarts_to_pints (q : ℕ) : ℕ := q * 2

def milk_for_cookies (cookies : ℕ) (milk_in_quarts : ℕ) : ℕ :=
  quarts_to_pints milk_in_quarts * cookies / 18

theorem milk_for_9_cookies :
  milk_for_cookies 9 3 = 3 :=
by
  -- We define the conversion and proportional conditions explicitly here.
  unfold milk_for_cookies
  unfold quarts_to_pints
  sorry

end milk_for_9_cookies_l2027_202797


namespace monotonic_increasing_on_interval_l2027_202779

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

theorem monotonic_increasing_on_interval (ω : ℝ) (h1 : ω > 0) (h2 : 2 * Real.pi / (2 * ω) = 4 * Real.pi) :
  ∀ x y : ℝ, (x ∈ Set.Icc (Real.pi / 2) Real.pi) → (y ∈ Set.Icc (Real.pi / 2) Real.pi) → x ≤ y → f ω x ≤ f ω y := 
by
  sorry

end monotonic_increasing_on_interval_l2027_202779


namespace algae_colony_growth_l2027_202744

def initial_cells : ℕ := 5
def days : ℕ := 10
def tripling_period : ℕ := 3
def cell_growth_ratio : ℕ := 3

noncomputable def cells_after_n_days (init_cells : ℕ) (day_count : ℕ) (period : ℕ) (growth_ratio : ℕ) : ℕ :=
  let steps := day_count / period
  init_cells * growth_ratio^steps

theorem algae_colony_growth : cells_after_n_days initial_cells days tripling_period cell_growth_ratio = 135 :=
  by sorry

end algae_colony_growth_l2027_202744


namespace nth_equation_pattern_l2027_202729

theorem nth_equation_pattern (n : ℕ) : 
  (List.range' n (2 * n - 1)).sum = (2 * n - 1) ^ 2 :=
by
  sorry

end nth_equation_pattern_l2027_202729


namespace perfect_square_expression_l2027_202751

theorem perfect_square_expression (p : ℝ) : 
  (12.86^2 + 12.86 * p + 0.14^2) = (12.86 + 0.14)^2 → p = 0.28 :=
by
  sorry

end perfect_square_expression_l2027_202751


namespace calculate_ff2_l2027_202799

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 4

theorem calculate_ff2 : f (f 2) = 5450 := by
  sorry

end calculate_ff2_l2027_202799


namespace decimal_to_fraction_l2027_202768

theorem decimal_to_fraction :
  (368 / 100 : ℚ) = (92 / 25 : ℚ) := by
  sorry

end decimal_to_fraction_l2027_202768


namespace number_of_students_l2027_202704

noncomputable def is_handshakes_correct (m n : ℕ) : Prop :=
  m ≥ 3 ∧ n ≥ 3 ∧ 
  (1 / 2 : ℚ) * (12 + 10 * (m + n - 4) + 8 * (m - 2) * (n - 2)) = 1020

theorem number_of_students (m n : ℕ) (h : is_handshakes_correct m n) : m * n = 280 := sorry

end number_of_students_l2027_202704


namespace triangle_area_l2027_202722

theorem triangle_area (a b : ℝ) (sinC sinA : ℝ) 
  (h1 : a = Real.sqrt 5) 
  (h2 : b = 3) 
  (h3 : sinC = 2 * sinA) : 
  ∃ (area : ℝ), area = 3 := 
by 
  sorry

end triangle_area_l2027_202722


namespace simplify_fraction_l2027_202755

theorem simplify_fraction (a b x : ℝ) (h₁ : x = a / b) (h₂ : a ≠ b) (h₃ : b ≠ 0) : 
  (2 * a + b) / (a - 2 * b) = (2 * x + 1) / (x - 2) :=
sorry

end simplify_fraction_l2027_202755


namespace symmetric_curve_equation_l2027_202706

theorem symmetric_curve_equation (y x : ℝ) :
  (y^2 = 4 * x) → (y^2 = 16 - 4 * x) :=
sorry

end symmetric_curve_equation_l2027_202706


namespace determine_unique_row_weight_free_l2027_202754

theorem determine_unique_row_weight_free (t : ℝ) (rows : Fin 10 → ℝ) (unique_row : Fin 10)
  (h_weights_same : ∀ i : Fin 10, i ≠ unique_row → rows i = t) :
  0 = 0 := by
  sorry

end determine_unique_row_weight_free_l2027_202754


namespace ages_correct_l2027_202705

variables (Rehana_age Phoebe_age Jacob_age Xander_age : ℕ)

theorem ages_correct
  (h1 : Rehana_age = 25)
  (h2 : Rehana_age + 5 = 3 * (Phoebe_age + 5))
  (h3 : Jacob_age = 3 * Phoebe_age / 5)
  (h4 : Xander_age = Rehana_age + Jacob_age - 4) : 
  Rehana_age = 25 ∧ Phoebe_age = 5 ∧ Jacob_age = 3 ∧ Xander_age = 24 :=
by
  sorry

end ages_correct_l2027_202705


namespace solution_set_of_inequality_l2027_202738

theorem solution_set_of_inequality (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
by sorry

end solution_set_of_inequality_l2027_202738


namespace correct_transformation_option_c_l2027_202720

theorem correct_transformation_option_c (x : ℝ) (h : (x / 2) - (x / 3) = 1) : 3 * x - 2 * x = 6 :=
by
  sorry

end correct_transformation_option_c_l2027_202720


namespace andrew_age_l2027_202724

theorem andrew_age (a g : ℕ) (h1 : g = 10 * a) (h2 : g - a = 63) : a = 7 := by
  sorry

end andrew_age_l2027_202724


namespace find_omega_and_range_l2027_202743

noncomputable def f (ω : ℝ) (x : ℝ) := (Real.sin (ω * x))^2 + (Real.sqrt 3) * (Real.sin (ω * x)) * (Real.sin (ω * x + Real.pi / 2))

theorem find_omega_and_range :
  ∃ ω : ℝ, ω > 0 ∧ (∀ x, f ω x = (Real.sin (2 * ω * x - Real.pi / 6) + 1/2)) ∧
    (∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2),
      f 1 x ∈ Set.Icc ((1 - Real.sqrt 3) / 2) (3 / 2)) :=
by
  sorry

end find_omega_and_range_l2027_202743


namespace expression_not_defined_at_x_eq_5_l2027_202759

theorem expression_not_defined_at_x_eq_5 :
  ∃ x : ℝ, x^3 - 15 * x^2 + 75 * x - 125 = 0 ↔ x = 5 :=
by
  sorry

end expression_not_defined_at_x_eq_5_l2027_202759


namespace Lilith_caps_collection_l2027_202714

theorem Lilith_caps_collection
  (caps_per_month_first_year : ℕ)
  (caps_per_month_after_first_year : ℕ)
  (caps_received_each_christmas : ℕ)
  (caps_lost_per_year : ℕ)
  (total_caps_collected : ℕ)
  (first_year_caps : ℕ := caps_per_month_first_year * 12)
  (years_after_first_year : ℕ)
  (total_years : ℕ := years_after_first_year + 1)
  (caps_collected_after_first_year : ℕ := caps_per_month_after_first_year * 12 * years_after_first_year)
  (caps_received_total : ℕ := caps_received_each_christmas * total_years)
  (caps_lost_total : ℕ := caps_lost_per_year * total_years)
  (total_calculated_caps : ℕ := first_year_caps + caps_collected_after_first_year + caps_received_total - caps_lost_total) :
  total_caps_collected = 401 → total_years = 5 :=
by
  sorry

end Lilith_caps_collection_l2027_202714


namespace maximum_value_frac_l2027_202760

-- Let x and y be positive real numbers. Prove that (x + y)^3 / (x^3 + y^3) ≤ 4.
theorem maximum_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y)^3 / (x^3 + y^3) ≤ 4 := sorry

end maximum_value_frac_l2027_202760


namespace range_of_m_l2027_202710

theorem range_of_m (f : ℝ → ℝ) (h_decreasing : ∀ x y, x < y → f y ≤ f x) (m : ℝ) (h : f (m-1) > f (2*m-1)) : 0 < m :=
by
  sorry

end range_of_m_l2027_202710


namespace probability_N_lt_L_is_zero_l2027_202712

variable (M N L O : ℝ)
variable (hNM : N < M)
variable (hLO : L > O)

theorem probability_N_lt_L_is_zero : 
  (∃ (permutations : List (ℝ → ℝ)), 
  (∀ perm : ℝ → ℝ, perm ∈ permutations → N < M ∧ L > O) ∧ 
  ∀ perm : ℝ → ℝ, N > L) → false :=
by {
  sorry
}

end probability_N_lt_L_is_zero_l2027_202712


namespace scientific_notation_l2027_202748

theorem scientific_notation (a : ℝ) (n : ℤ) (h1 : 1 ≤ a ∧ a < 10) (h2 : 43050000 = a * 10^n) : a = 4.305 ∧ n = 7 :=
by
  sorry

end scientific_notation_l2027_202748


namespace average_of_three_marbles_l2027_202769

-- Define the conditions as hypotheses
theorem average_of_three_marbles (R Y B : ℕ) 
  (h1 : R + Y = 53)
  (h2 : B + Y = 69)
  (h3 : R + B = 58) :
  (R + Y + B) / 3 = 30 :=
by
  sorry

end average_of_three_marbles_l2027_202769


namespace width_of_metallic_sheet_l2027_202796

-- Define the given conditions
def length_of_sheet : ℝ := 48
def side_of_square_cut : ℝ := 7
def volume_of_box : ℝ := 5236

-- Define the question as a Lean theorem
theorem width_of_metallic_sheet : ∃ (w : ℝ), w = 36 ∧
  volume_of_box = (length_of_sheet - 2 * side_of_square_cut) * (w - 2 * side_of_square_cut) * side_of_square_cut := by
  sorry

end width_of_metallic_sheet_l2027_202796


namespace equal_charges_at_4_hours_l2027_202791

-- Define the charges for both companies
def PaulsPlumbingCharge (h : ℝ) : ℝ := 55 + 35 * h
def ReliablePlumbingCharge (h : ℝ) : ℝ := 75 + 30 * h

-- Prove that for 4 hours of labor, the charges are equal
theorem equal_charges_at_4_hours : PaulsPlumbingCharge 4 = ReliablePlumbingCharge 4 :=
by
  sorry

end equal_charges_at_4_hours_l2027_202791


namespace kernels_popped_in_final_bag_l2027_202757

/-- Parker wants to find out what the average percentage of kernels that pop in a bag is.
In the first bag he makes, 60 kernels pop and the bag has 75 kernels.
In the second bag, 42 kernels pop and there are 50 in the bag.
In the final bag, some kernels pop and the bag has 100 kernels.
The average percentage of kernels that pop in a bag is 82%.
How many kernels popped in the final bag?
We prove that given these conditions, the number of popped kernels in the final bag is 82.
-/
noncomputable def kernelsPoppedInFirstBag := 60
noncomputable def totalKernelsInFirstBag := 75
noncomputable def kernelsPoppedInSecondBag := 42
noncomputable def totalKernelsInSecondBag := 50
noncomputable def totalKernelsInFinalBag := 100
noncomputable def averagePoppedPercentage := 82

theorem kernels_popped_in_final_bag (x : ℕ) :
  (kernelsPoppedInFirstBag * 100 / totalKernelsInFirstBag +
   kernelsPoppedInSecondBag * 100 / totalKernelsInSecondBag +
   x * 100 / totalKernelsInFinalBag) / 3 = averagePoppedPercentage →
  x = 82 := 
by
  sorry

end kernels_popped_in_final_bag_l2027_202757


namespace cos_of_angle_B_l2027_202721

theorem cos_of_angle_B (A B C a b c : Real) (h₁ : A - C = Real.pi / 2) (h₂ : 2 * b = a + c) 
  (h₃ : 2 * a * Real.sin A = 2 * b * Real.sin B) (h₄ : 2 * c * Real.sin C = 2 * b * Real.sin B) :
  Real.cos B = 3 / 4 := by
  sorry

end cos_of_angle_B_l2027_202721


namespace area_of_triangle_is_correct_l2027_202792

def vector := (ℝ × ℝ)

def a : vector := (7, 3)
def b : vector := (-1, 5)

noncomputable def det2x2 (v1 v2 : vector) : ℝ :=
  (v1.1 * v2.2) - (v1.2 * v2.1)

theorem area_of_triangle_is_correct :
  let area := (det2x2 a b) / 2
  area = 19 := by
  -- defintions and conditions are set here, proof skipped
  sorry

end area_of_triangle_is_correct_l2027_202792


namespace polar_to_cartesian_max_and_min_x_plus_y_l2027_202752

-- Define the given polar equation and convert it to Cartesian equations
def polar_equation (rho θ : ℝ) : Prop :=
  rho^2 - 4 * (Real.sqrt 2) * rho * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Cartesian equation derived from the polar equation
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 4 * y + 6 = 0

-- Prove equivalence of the given polar equation and its equivalent Cartesian form for all ρ and \theta
theorem polar_to_cartesian (rho θ : ℝ) : 
  (∃ (x y : ℝ), polar_equation rho θ ∧ x = rho * Real.cos θ ∧ y = rho * Real.sin θ ∧ cartesian_equation x y) :=
by
  sorry

-- Property of points (x, y) on the circle defined by the Cartesian equation
def lies_on_circle (x y : ℝ) : Prop :=
  cartesian_equation x y

-- Given a point (x, y) on the circle defined by cartesian_equation, show bounds for x + y
theorem max_and_min_x_plus_y (x y : ℝ) (h : lies_on_circle x y) : 
  2 ≤ x + y ∧ x + y ≤ 6 :=
by
  sorry

end polar_to_cartesian_max_and_min_x_plus_y_l2027_202752


namespace power_mod_remainder_l2027_202727

theorem power_mod_remainder (a b : ℕ) (h1 : a = 3) (h2 : b = 167) :
  (3^167) % 11 = 9 := by
  sorry

end power_mod_remainder_l2027_202727


namespace jury_selection_duration_is_two_l2027_202763

variable (jury_selection_days : ℕ) (trial_days : ℕ) (deliberation_days : ℕ)

axiom trial_lasts_four_times_jury_selection : trial_days = 4 * jury_selection_days
axiom deliberation_is_six_full_days : deliberation_days = (6 * 24) / 16
axiom john_spends_nineteen_days : jury_selection_days + trial_days + deliberation_days = 19

theorem jury_selection_duration_is_two : jury_selection_days = 2 :=
by
  sorry

end jury_selection_duration_is_two_l2027_202763


namespace milkshake_hours_l2027_202733

theorem milkshake_hours (h : ℕ) : 
  (3 * h + 7 * h = 80) → h = 8 := 
by
  intro h_milkshake_eq
  sorry

end milkshake_hours_l2027_202733


namespace monotonic_intervals_l2027_202790

noncomputable def f (a x : ℝ) : ℝ := x^2 * Real.exp (a * x)

theorem monotonic_intervals (a : ℝ) :
  (a = 0 → (∀ x : ℝ, (x < 0 → f a x < f a (-1)) ∧ (x > 0 → f a x > f a 1))) ∧
  (a > 0 → (∀ x : ℝ, (x < -2 / a → f a x < f a (-2 / a - 1)) ∧ (x > 0 → f a x > f a 1) ∧ 
                  ((-2 / a) < x ∧ x < 0 → f a x < f a (-2 / a + 1)))) ∧
  (a < 0 → (∀ x : ℝ, (x < 0 → f a x < f a (-1)) ∧ (x > -2 / a → f a x < f a (-2 / a - 1)) ∧
                  (0 < x ∧ x < -2 / a → f a x > f a (-2 / a + 1))))
:= sorry

end monotonic_intervals_l2027_202790


namespace log_ratio_l2027_202737

theorem log_ratio : (Real.logb 2 16) / (Real.logb 2 4) = 2 := sorry

end log_ratio_l2027_202737


namespace calculate_bubble_bath_needed_l2027_202734

theorem calculate_bubble_bath_needed :
  let double_suites_capacity := 5 * 4
  let rooms_for_couples_capacity := 13 * 2
  let single_rooms_capacity := 14 * 1
  let family_rooms_capacity := 3 * 6
  let total_guests := double_suites_capacity + rooms_for_couples_capacity + single_rooms_capacity + family_rooms_capacity
  let bubble_bath_per_guest := 25
  total_guests * bubble_bath_per_guest = 1950 := by
  let double_suites_capacity := 5 * 4
  let rooms_for_couples_capacity := 13 * 2
  let single_rooms_capacity := 14 * 1
  let family_rooms_capacity := 3 * 6
  let total_guests := double_suites_capacity + rooms_for_couples_capacity + single_rooms_capacity + family_rooms_capacity
  let bubble_bath_per_guest := 25
  sorry

end calculate_bubble_bath_needed_l2027_202734


namespace ratio_of_volumes_l2027_202701

theorem ratio_of_volumes (r1 r2 : ℝ) (h : (4 * π * r1^2) / (4 * π * r2^2) = 4 / 9) :
  (4/3 * π * r1^3) / (4/3 * π * r2^3) = 8 / 27 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_volumes_l2027_202701


namespace speed_comparison_l2027_202788

theorem speed_comparison (v v2 : ℝ) (h1 : v2 > 0) (h2 : v = 5 * v2) : v = 5 * v2 :=
by
  exact h2 

end speed_comparison_l2027_202788


namespace samia_walked_distance_l2027_202773

theorem samia_walked_distance :
  ∀ (total_distance cycling_speed walking_speed total_time : ℝ), 
  total_distance = 18 → 
  cycling_speed = 20 → 
  walking_speed = 4 → 
  total_time = 1 + 10 / 60 → 
  2 / 3 * total_distance / cycling_speed + 1 / 3 * total_distance / walking_speed = total_time → 
  1 / 3 * total_distance = 6 := 
by
  intros total_distance cycling_speed walking_speed total_time h1 h2 h3 h4 h5
  sorry

end samia_walked_distance_l2027_202773


namespace smallest_divisible_by_1_to_10_l2027_202777

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end smallest_divisible_by_1_to_10_l2027_202777


namespace remaining_movies_l2027_202764

-- Definitions based on the problem's conditions
def total_movies : ℕ := 8
def watched_movies : ℕ := 4

-- Theorem statement to prove that you still have 4 movies left to watch
theorem remaining_movies : total_movies - watched_movies = 4 :=
by
  sorry

end remaining_movies_l2027_202764


namespace largest_possible_b_l2027_202776

theorem largest_possible_b (b : ℝ) (h : (3 * b + 6) * (b - 2) = 9 * b) : b ≤ 4 := 
by {
  -- leaving the proof as an exercise, using 'sorry' to complete the statement
  sorry
}

end largest_possible_b_l2027_202776


namespace percentage_increase_first_year_l2027_202700

-- Assume the original price of the painting is P and the percentage increase during the first year is X
variable {P : ℝ} (X : ℝ)

-- Condition: The price decreases by 15% during the second year
def condition_decrease (price : ℝ) : ℝ := price * 0.85

-- Condition: The price at the end of the 2-year period was 93.5% of the original price
axiom condition_end_price : ∀ (P : ℝ), (P + (X/100) * P) * 0.85 = 0.935 * P

-- Proof problem: What was the percentage increase during the first year?
theorem percentage_increase_first_year : X = 10 :=
by 
  sorry

end percentage_increase_first_year_l2027_202700


namespace account_balance_after_transfer_l2027_202719

def account_after_transfer (initial_balance transfer_amount : ℕ) : ℕ :=
  initial_balance - transfer_amount

theorem account_balance_after_transfer :
  account_after_transfer 27004 69 = 26935 :=
by
  sorry

end account_balance_after_transfer_l2027_202719


namespace interest_second_month_l2027_202730

theorem interest_second_month {P r n : ℝ} (hP : P = 200) (hr : r = 0.10) (hn : n = 12) :
  (P * (1 + r / n) ^ (n * (1/12)) - P) * r / n = 1.68 :=
by
  sorry

end interest_second_month_l2027_202730


namespace percentage_y_less_than_x_l2027_202783

theorem percentage_y_less_than_x (x y : ℝ) (h : x = 11 * y) : 
  ((x - y) / x) * 100 = 90.91 := 
by 
  sorry -- proof to be provided separately

end percentage_y_less_than_x_l2027_202783


namespace option_A_option_B_option_D_l2027_202787

-- Given real numbers a, b, c such that a > b > 1 and c > 0,
-- prove the following inequalities.
variables {a b c : ℝ}

-- Assume the conditions
axiom H1 : a > b
axiom H2 : b > 1
axiom H3 : c > 0

-- Statements to prove
theorem option_A (H1: a > b) (H2: b > 1) (H3: c > 0) : a^2 - bc > b^2 - ac := sorry
theorem option_B (H1: a > b) (H2: b > 1) : a^3 > b^2 := sorry
theorem option_D (H1: a > b) (H2: b > 1) : a + (1/a) > b + (1/b) := sorry
  
end option_A_option_B_option_D_l2027_202787


namespace evaluate_poly_at_2_l2027_202739

def my_op (x y : ℕ) : ℕ := (x + 1) * (y + 1)
def star2 (x : ℕ) : ℕ := my_op x x

theorem evaluate_poly_at_2 :
  3 * (star2 2) - 2 * 2 + 1 = 24 :=
by
  sorry

end evaluate_poly_at_2_l2027_202739


namespace flower_pots_on_path_count_l2027_202711

theorem flower_pots_on_path_count (L d : ℕ) (hL : L = 15) (hd : d = 3) : 
  (L / d) + 1 = 6 :=
by
  sorry

end flower_pots_on_path_count_l2027_202711


namespace exactly_one_defective_l2027_202771

theorem exactly_one_defective (p_A p_B : ℝ) (hA : p_A = 0.04) (hB : p_B = 0.05) :
  ((p_A * (1 - p_B)) + ((1 - p_A) * p_B)) = 0.086 :=
by
  sorry

end exactly_one_defective_l2027_202771


namespace fraction_sum_to_decimal_l2027_202784

theorem fraction_sum_to_decimal : 
  (3 / 10 : Rat) + (5 / 100) - (1 / 1000) = 349 / 1000 := 
by
  sorry

end fraction_sum_to_decimal_l2027_202784


namespace option_d_is_quadratic_equation_l2027_202723

theorem option_d_is_quadratic_equation (x y : ℝ) : 
  (x^2 + x - 4 = 0) ↔ (x^2 + x = 4) := 
by
  sorry

end option_d_is_quadratic_equation_l2027_202723


namespace two_digit_sabroso_numbers_l2027_202766

theorem two_digit_sabroso_numbers :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ (n + (10 * b + a) = k^2)} =
  {29, 38, 47, 56, 65, 74, 83, 92} :=
sorry

end two_digit_sabroso_numbers_l2027_202766


namespace bread_slices_per_friend_l2027_202716

theorem bread_slices_per_friend :
  (∀ (slices_per_loaf friends loaves total_slices_per_friend : ℕ),
    slices_per_loaf = 15 →
    friends = 10 →
    loaves = 4 →
    total_slices_per_friend = slices_per_loaf * loaves / friends →
    total_slices_per_friend = 6) :=
by 
  intros slices_per_loaf friends loaves total_slices_per_friend h1 h2 h3 h4
  sorry

end bread_slices_per_friend_l2027_202716


namespace fraction_to_decimal_l2027_202770

theorem fraction_to_decimal :
  (7 : ℝ) / (16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l2027_202770
