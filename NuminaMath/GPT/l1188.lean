import Mathlib

namespace count_integer_triangles_with_perimeter_12_l1188_118894

theorem count_integer_triangles_with_perimeter_12 : 
  ∃! (sides : ℕ × ℕ × ℕ), sides.1 + sides.2.1 + sides.2.2 = 12 ∧ sides.1 + sides.2.1 > sides.2.2 ∧ sides.1 + sides.2.2 > sides.2.1 ∧ sides.2.1 + sides.2.2 > sides.1 ∧
  (sides = (2, 5, 5) ∨ sides = (3, 4, 5) ∨ sides = (4, 4, 4)) :=
by 
  exists 3
  sorry

end count_integer_triangles_with_perimeter_12_l1188_118894


namespace program_total_cost_l1188_118804

-- Define the necessary variables and constants
def ms_to_s : Float := 0.001
def os_overhead : Float := 1.07
def cost_per_ms : Float := 0.023
def mount_cost : Float := 5.35
def time_required : Float := 1.5

-- Calculate components of the total cost
def total_cost_for_computer_time := (time_required * 1000) * cost_per_ms
def total_cost := os_overhead + total_cost_for_computer_time + mount_cost

-- State the theorem
theorem program_total_cost : total_cost = 40.92 := by
  sorry

end program_total_cost_l1188_118804


namespace nicholas_crackers_l1188_118838

theorem nicholas_crackers (marcus_crackers mona_crackers nicholas_crackers : ℕ) 
  (h1 : marcus_crackers = 3 * mona_crackers)
  (h2 : nicholas_crackers = mona_crackers + 6)
  (h3 : marcus_crackers = 27) : nicholas_crackers = 15 := by
  sorry

end nicholas_crackers_l1188_118838


namespace solve_system_of_equations_l1188_118899

theorem solve_system_of_equations (n : ℕ) (hn : n ≥ 3) (x : ℕ → ℝ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    x i ^ 3 = (x ((i % n) + 1) + x ((i % n) + 2) + 1)) →
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    (x i = -1 ∨ x i = (1 + Real.sqrt 5) / 2 ∨ x i = (1 - Real.sqrt 5) / 2)) :=
sorry

end solve_system_of_equations_l1188_118899


namespace quadratic_real_roots_opposite_signs_l1188_118818

theorem quadratic_real_roots_opposite_signs (c : ℝ) : 
  (c < 0 → (∃ x1 x2 : ℝ, x1 * x2 = c ∧ x1 + x2 = -1 ∧ x1 ≠ x2 ∧ (x1 < 0 ∧ x2 > 0 ∨ x1 > 0 ∧ x2 < 0))) ∧ 
  (∃ x1 x2 : ℝ, x1 * x2 = c ∧ x1 + x2 = -1 ∧ x1 ≠ x2 ∧ (x1 < 0 ∧ x2 > 0 ∨ x1 > 0 ∧ x2 < 0) → c < 0) :=
by 
  sorry

end quadratic_real_roots_opposite_signs_l1188_118818


namespace arithmetic_progression_25th_term_l1188_118847

theorem arithmetic_progression_25th_term (a1 d : ℤ) (n : ℕ) (h_a1 : a1 = 5) (h_d : d = 7) (h_n : n = 25) :
  a1 + (n - 1) * d = 173 :=
by
  sorry

end arithmetic_progression_25th_term_l1188_118847


namespace geom_mean_between_2_and_8_l1188_118820

theorem geom_mean_between_2_and_8 (b : ℝ) (h : b^2 = 16) : b = 4 ∨ b = -4 :=
by
  sorry

end geom_mean_between_2_and_8_l1188_118820


namespace roots_cubic_sum_of_cubes_l1188_118831

theorem roots_cubic_sum_of_cubes (a b c : ℝ)
  (h1 : Polynomial.eval a (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h3 : Polynomial.eval c (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h4 : a + b + c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 753 :=
by
  sorry

end roots_cubic_sum_of_cubes_l1188_118831


namespace summer_camp_skills_l1188_118839

theorem summer_camp_skills
  (x y z a b c : ℕ)
  (h1 : x + y + z + a + b + c = 100)
  (h2 : y + z + c = 42)
  (h3 : z + x + b = 65)
  (h4 : x + y + a = 29) :
  a + b + c = 64 :=
by sorry

end summer_camp_skills_l1188_118839


namespace batsman_total_score_l1188_118892

-- We establish our variables and conditions first
variables (T : ℕ) -- total score
variables (boundaries : ℕ := 3) -- number of boundaries
variables (sixes : ℕ := 8) -- number of sixes
variables (boundary_runs_per : ℕ := 4) -- runs per boundary
variables (six_runs_per : ℕ := 6) -- runs per six
variables (running_percentage : ℕ := 50) -- percentage of runs made by running

-- Define the amounts of runs from boundaries and sixes
def runs_from_boundaries := boundaries * boundary_runs_per
def runs_from_sixes := sixes * six_runs_per

-- Main theorem to prove
theorem batsman_total_score :
  T = runs_from_boundaries + runs_from_sixes + T / 2 → T = 120 :=
by
  sorry

end batsman_total_score_l1188_118892


namespace antecedent_is_50_l1188_118862

theorem antecedent_is_50 (antecedent consequent : ℕ) (h_ratio : 4 * consequent = 6 * antecedent) (h_consequent : consequent = 75) : antecedent = 50 := by
  sorry

end antecedent_is_50_l1188_118862


namespace marina_max_socks_l1188_118813

theorem marina_max_socks (white black : ℕ) (hw : white = 8) (hb : black = 15) :
  ∃ n, n = 17 ∧ ∀ w b, w + b = n → 0 ≤ w ∧ 0 ≤ b ∧ w ≤ black ∧ b ≤ black ∧ w ≤ white ∧ b ≤ black → b > w :=
sorry

end marina_max_socks_l1188_118813


namespace triangle_height_l1188_118881

def width := 10
def length := 2 * width
def area_rectangle := width * length
def base_triangle := width

theorem triangle_height (h : ℝ) : (1 / 2) * base_triangle * h = area_rectangle → h = 40 :=
by
  sorry

end triangle_height_l1188_118881


namespace four_bags_remainder_l1188_118854

theorem four_bags_remainder (n : ℤ) (hn : n % 11 = 5) : (4 * n) % 11 = 9 := 
by
  sorry

end four_bags_remainder_l1188_118854


namespace solution_intervals_l1188_118819

noncomputable def cubic_inequality (x : ℝ) : Prop :=
  x^3 - 3 * x^2 - 4 * x - 12 ≤ 0

noncomputable def linear_inequality (x : ℝ) : Prop :=
  2 * x + 6 > 0

theorem solution_intervals :
  { x : ℝ | cubic_inequality x ∧ linear_inequality x } = { x | -2 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end solution_intervals_l1188_118819


namespace identify_counterfeit_bag_l1188_118895

-- Definitions based on problem conditions
def num_bags := 10
def genuine_weight := 10
def counterfeit_weight := 11
def expected_total_weight := genuine_weight * ((num_bags * (num_bags + 1)) / 2 : ℕ)

-- Lean theorem for the above problem
theorem identify_counterfeit_bag (W : ℕ) (Δ := W - expected_total_weight) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ num_bags ∧ Δ = i :=
by sorry

end identify_counterfeit_bag_l1188_118895


namespace inequality_proof_l1188_118816

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
sorry

end inequality_proof_l1188_118816


namespace new_average_age_l1188_118866

theorem new_average_age (n : ℕ) (avg_old : ℕ) (new_person_age : ℕ) (new_avg_age : ℕ)
  (h1 : avg_old = 14)
  (h2 : n = 9)
  (h3 : new_person_age = 34)
  (h4 : new_avg_age = 16) :
  (n * avg_old + new_person_age) / (n + 1) = new_avg_age :=
sorry

end new_average_age_l1188_118866


namespace boris_clock_time_l1188_118803

-- Define a function to compute the sum of digits of a number.
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the problem
theorem boris_clock_time (h m : ℕ) :
  sum_digits h + sum_digits m = 6 ∧ h + m = 15 ↔
  (h, m) = (0, 15) ∨ (h, m) = (1, 14) ∨ (h, m) = (2, 13) ∨ (h, m) = (3, 12) ∨
  (h, m) = (4, 11) ∨ (h, m) = (5, 10) ∨ (h, m) = (10, 5) ∨ (h, m) = (11, 4) ∨
  (h, m) = (12, 3) ∨ (h, m) = (13, 2) ∨ (h, m) = (14, 1) ∨ (h, m) = (15, 0) :=
by sorry

end boris_clock_time_l1188_118803


namespace linda_total_profit_is_50_l1188_118884

def total_loaves : ℕ := 60
def loaves_sold_morning (total_loaves : ℕ) : ℕ := total_loaves / 3
def loaves_sold_afternoon (loaves_left_morning : ℕ) : ℕ := loaves_left_morning / 2
def loaves_sold_evening (loaves_left_afternoon : ℕ) : ℕ := loaves_left_afternoon

def price_per_loaf_morning : ℕ := 3
def price_per_loaf_afternoon : ℕ := 150 / 100 -- Representing $1.50 as 150 cents to use integer arithmetic
def price_per_loaf_evening : ℕ := 1

def cost_per_loaf : ℕ := 1

def calculate_profit (total_loaves loaves_sold_morning loaves_sold_afternoon loaves_sold_evening price_per_loaf_morning price_per_loaf_afternoon price_per_loaf_evening cost_per_loaf : ℕ) : ℕ := 
  let revenue_morning := loaves_sold_morning * price_per_loaf_morning
  let loaves_left_morning := total_loaves - loaves_sold_morning
  let revenue_afternoon := loaves_sold_afternoon * price_per_loaf_afternoon
  let loaves_left_afternoon := loaves_left_morning - loaves_sold_afternoon
  let revenue_evening := loaves_sold_evening * price_per_loaf_evening
  let total_revenue := revenue_morning + revenue_afternoon + revenue_evening
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

theorem linda_total_profit_is_50 : calculate_profit total_loaves (loaves_sold_morning total_loaves) (loaves_sold_afternoon (total_loaves - loaves_sold_morning total_loaves)) (total_loaves - loaves_sold_morning total_loaves - loaves_sold_afternoon (total_loaves - loaves_sold_morning total_loaves)) price_per_loaf_morning price_per_loaf_afternoon price_per_loaf_evening cost_per_loaf = 50 := 
  by 
    sorry

end linda_total_profit_is_50_l1188_118884


namespace sum_of_three_numbers_l1188_118817

theorem sum_of_three_numbers (a b c : ℝ) (h₁ : a + b = 31) (h₂ : b + c = 48) (h₃ : c + a = 59) :
  a + b + c = 69 :=
by
  sorry

end sum_of_three_numbers_l1188_118817


namespace total_students_l1188_118832

theorem total_students (S K : ℕ) (h1 : S = 4000) (h2 : K = 2 * S) :
  S + K = 12000 := by
  sorry

end total_students_l1188_118832


namespace height_of_highest_wave_l1188_118882

theorem height_of_highest_wave 
  (h_austin : ℝ) -- Austin's height
  (h_high : ℝ) -- Highest wave's height
  (h_short : ℝ) -- Shortest wave's height 
  (height_relation1 : h_high = 4 * h_austin + 2)
  (height_relation2 : h_short = h_austin + 4)
  (surfboard : ℝ) (surfboard_len : surfboard = 7)
  (short_wave_len : h_short = surfboard + 3) :
  h_high = 26 :=
by
  -- Define local variables with the values from given conditions
  let austin_height := 6        -- as per calculation: 10 - 4 = 6
  let highest_wave_height := 26 -- as per calculation: (6 * 4) + 2 = 26
  sorry

end height_of_highest_wave_l1188_118882


namespace ben_apples_difference_l1188_118897

theorem ben_apples_difference (B P T : ℕ) (h1 : P = 40) (h2 : T = 18) (h3 : (3 / 8) * B = T) :
  B - P = 8 :=
sorry

end ben_apples_difference_l1188_118897


namespace number_of_towers_l1188_118879

noncomputable def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def multinomial (n : ℕ) (k1 k2 k3 : ℕ) : ℕ :=
  factorial n / (factorial k1 * factorial k2 * factorial k3)

theorem number_of_towers :
  (multinomial 10 3 3 4 = 4200) :=
by
  sorry

end number_of_towers_l1188_118879


namespace range_of_x_if_cos2_gt_sin2_l1188_118837

theorem range_of_x_if_cos2_gt_sin2 (x : ℝ) (h1 : x ∈ Set.Icc 0 Real.pi) (h2 : Real.cos x ^ 2 > Real.sin x ^ 2) :
  x ∈ Set.Ico 0 (Real.pi / 4) ∪ Set.Ioc (3 * Real.pi / 4) Real.pi :=
by
  sorry

end range_of_x_if_cos2_gt_sin2_l1188_118837


namespace probability_of_pink_tie_l1188_118801

theorem probability_of_pink_tie 
  (black_ties gold_ties pink_ties : ℕ) 
  (h_black : black_ties = 5) 
  (h_gold : gold_ties = 7) 
  (h_pink : pink_ties = 8) 
  (h_total : (5 + 7 + 8) = (black_ties + gold_ties + pink_ties)) 
  : (pink_ties : ℚ) / (black_ties + gold_ties + pink_ties) = 2 / 5 := 
by 
  sorry

end probability_of_pink_tie_l1188_118801


namespace expand_polynomial_l1188_118829

theorem expand_polynomial :
  (5 * x^2 + 3 * x - 4) * 3 * x^3 = 15 * x^5 + 9 * x^4 - 12 * x^3 := 
by
  sorry

end expand_polynomial_l1188_118829


namespace water_added_l1188_118822

theorem water_added (initial_volume : ℕ) (initial_sugar_percentage : ℝ) (final_sugar_percentage : ℝ) (V : ℝ) : 
  initial_volume = 3 →
  initial_sugar_percentage = 0.4 →
  final_sugar_percentage = 0.3 →
  V = 1 :=
by
  sorry

end water_added_l1188_118822


namespace min_value_expr_l1188_118812

noncomputable def expr (θ : Real) : Real :=
  3 * (Real.cos θ) + 2 / (Real.sin θ) + 2 * Real.sqrt 2 * (Real.tan θ)

theorem min_value_expr :
  ∃ (θ : Real), 0 < θ ∧ θ < Real.pi / 2 ∧ expr θ = (7 * Real.sqrt 2) / 2 := 
by
  sorry

end min_value_expr_l1188_118812


namespace remainder_when_divided_by_4x_minus_8_l1188_118875

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30

-- Define the divisor d(x)
def d (x : ℝ) : ℝ := 4 * x - 8

-- The specific value where the remainder theorem applies (root of d(x) = 0 is x = 2)
def x₀ : ℝ := 2

-- Prove the remainder when p(x) is divided by d(x) is 10
theorem remainder_when_divided_by_4x_minus_8 :
  (p x₀ = 10) :=
by
  -- The proof will be filled in here.
  sorry

end remainder_when_divided_by_4x_minus_8_l1188_118875


namespace find_sum_of_a_b_l1188_118833

def star (a b : ℕ) : ℕ := a^b - a * b

theorem find_sum_of_a_b (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 2) : a + b = 5 := 
by
  sorry

end find_sum_of_a_b_l1188_118833


namespace remainder_4063_div_97_l1188_118810

theorem remainder_4063_div_97 : 4063 % 97 = 86 := 
by sorry

end remainder_4063_div_97_l1188_118810


namespace ratio_brown_eyes_l1188_118885

theorem ratio_brown_eyes (total_people : ℕ) (blue_eyes : ℕ) (black_eyes : ℕ) (green_eyes : ℕ) (brown_eyes : ℕ) 
    (h1 : total_people = 100) 
    (h2 : blue_eyes = 19) 
    (h3 : black_eyes = total_people / 4) 
    (h4 : green_eyes = 6) 
    (h5 : brown_eyes = total_people - (blue_eyes + black_eyes + green_eyes)) : 
    brown_eyes / total_people = 1 / 2 :=
by sorry

end ratio_brown_eyes_l1188_118885


namespace least_ab_value_l1188_118898

theorem least_ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h : (1 : ℚ)/a + (1 : ℚ)/(3 * b) = 1 / 6) : a * b = 98 :=
by
  sorry

end least_ab_value_l1188_118898


namespace prove_total_rent_of_field_l1188_118845

def totalRentField (A_cows A_months B_cows B_months C_cows C_months 
                    D_cows D_months E_cows E_months F_cows F_months 
                    G_cows G_months A_rent : ℕ) : ℕ := 
  let A_cow_months := A_cows * A_months
  let B_cow_months := B_cows * B_months
  let C_cow_months := C_cows * C_months
  let D_cow_months := D_cows * D_months
  let E_cow_months := E_cows * E_months
  let F_cow_months := F_cows * F_months
  let G_cow_months := G_cows * G_months
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + 
                          D_cow_months + E_cow_months + F_cow_months + G_cow_months
  let rent_per_cow_month := A_rent / A_cow_months
  total_cow_months * rent_per_cow_month

theorem prove_total_rent_of_field : totalRentField 24 3 10 5 35 4 21 3 15 6 40 2 28 (7/2) 720 = 5930 :=
  by
  sorry

end prove_total_rent_of_field_l1188_118845


namespace panthers_second_half_points_l1188_118842

theorem panthers_second_half_points (C1 P1 C2 P2 : ℕ) 
  (h1 : C1 + P1 = 38) 
  (h2 : C1 = P1 + 16) 
  (h3 : C1 + C2 + P1 + P2 = 58) 
  (h4 : C1 + C2 = P1 + P2 + 22) : 
  P2 = 7 :=
by 
  -- Definitions and substitutions are skipped here
  sorry

end panthers_second_half_points_l1188_118842


namespace amanda_bought_30_candy_bars_l1188_118807

noncomputable def candy_bars_bought (c1 c2 c3 c4 : ℕ) : ℕ :=
  let c5 := c4 * c2
  let c6 := c3 - c2
  let c7 := (c6 + c5) - c1
  c7

theorem amanda_bought_30_candy_bars :
  candy_bars_bought 7 3 22 4 = 30 :=
by
  sorry

end amanda_bought_30_candy_bars_l1188_118807


namespace power_six_rectangular_form_l1188_118865

noncomputable def sin (x : ℂ) : ℂ := (Complex.exp (-Complex.I * x) - Complex.exp (Complex.I * x)) / (2 * Complex.I)
noncomputable def cos (x : ℂ) : ℂ := (Complex.exp (Complex.I * x) + Complex.exp (-Complex.I * x)) / 2

theorem power_six_rectangular_form :
  (2 * cos (20 * Real.pi / 180) + 2 * Complex.I * sin (20 * Real.pi / 180))^6 = -32 + 32 * Complex.I * Real.sqrt 3 := sorry

end power_six_rectangular_form_l1188_118865


namespace base6_to_base10_product_zero_l1188_118857

theorem base6_to_base10_product_zero
  (c d e : ℕ)
  (h : (5 * 6^2 + 3 * 6^1 + 2 * 6^0) = (100 * c + 10 * d + e)) :
  (c * e) / 10 = 0 :=
by
  sorry

end base6_to_base10_product_zero_l1188_118857


namespace students_answered_both_correct_l1188_118893

theorem students_answered_both_correct (total_students : ℕ)
  (answered_sets_correctly : ℕ) (answered_functions_correctly : ℕ)
  (both_wrong : ℕ) (total : total_students = 50)
  (sets_correct : answered_sets_correctly = 40)
  (functions_correct : answered_functions_correctly = 31)
  (wrong_both : both_wrong = 4) :
  (40 + 31 - (total_students - 4) + both_wrong = 50) → total_students - (40 + 31 - (total_students - 4)) = 29 :=
by
  sorry

end students_answered_both_correct_l1188_118893


namespace coefficient_of_x_l1188_118886

theorem coefficient_of_x :
  let expr := (5 * (x - 6)) + (6 * (9 - 3 * x ^ 2 + 3 * x)) - (9 * (5 * x - 4))
  (expr : ℝ) → 
  let expr' := 5 * x - 30 + 54 - 18 * x ^ 2 + 18 * x - 45 * x + 36
  (expr' : ℝ) → 
  let coeff_x := 5 + 18 - 45
  coeff_x = -22 :=
by
  sorry

end coefficient_of_x_l1188_118886


namespace intersection_complement_l1188_118852

open Set

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | 2 < x}
def R_complement_N : Set ℝ := {x | x ≤ 2}

theorem intersection_complement : M ∩ R_complement_N = {x | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_complement_l1188_118852


namespace find_side_length_of_left_square_l1188_118846

theorem find_side_length_of_left_square (x : ℕ) 
  (h1 : x + (x + 17) + (x + 11) = 52) : 
  x = 8 :=
by
  -- The proof will go here
  sorry

end find_side_length_of_left_square_l1188_118846


namespace a2_value_for_cubic_expansion_l1188_118860

theorem a2_value_for_cubic_expansion (x a0 a1 a2 a3 : ℝ) : 
  (x ^ 3 = a0 + a1 * (x - 2) + a2 * (x - 2) ^ 2 + a3 * (x - 2) ^ 3) → a2 = 6 := by
  sorry

end a2_value_for_cubic_expansion_l1188_118860


namespace max_min_product_l1188_118828

theorem max_min_product (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h_sum : p + q + r = 13) (h_prod_sum : p * q + q * r + r * p = 30) :
  ∃ n, n = min (p * q) (min (q * r) (r * p)) ∧ n = 10 :=
by
  sorry

end max_min_product_l1188_118828


namespace percentage_of_z_equals_39_percent_of_y_l1188_118890

theorem percentage_of_z_equals_39_percent_of_y
    (x y z : ℝ)
    (h1 : y = 0.75 * x)
    (h2 : z = 0.65 * x)
    (P : ℝ)
    (h3 : (P / 100) * z = 0.39 * y) :
    P = 45 :=
by sorry

end percentage_of_z_equals_39_percent_of_y_l1188_118890


namespace graph_inverse_point_sum_l1188_118840

theorem graph_inverse_point_sum 
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ) 
  (h1 : ∀ x, f_inv (f x) = x) 
  (h2 : ∀ x, f (f_inv x) = x) 
  (h3 : f 2 = 6) 
  (h4 : (2, 3) ∈ {p : ℝ × ℝ | p.snd = f p.fst / 2}) :
  (6, 1) ∈ {p : ℝ × ℝ | p.snd = f_inv p.fst / 2} ∧ (6 + 1 = 7) :=
by
  sorry

end graph_inverse_point_sum_l1188_118840


namespace Iggy_miles_on_Monday_l1188_118841

theorem Iggy_miles_on_Monday 
  (tuesday_miles : ℕ)
  (wednesday_miles : ℕ)
  (thursday_miles : ℕ)
  (friday_miles : ℕ)
  (monday_minutes : ℕ)
  (pace : ℕ)
  (total_hours : ℕ)
  (total_minutes : ℕ)
  (total_tuesday_to_friday_miles : ℕ)
  (total_tuesday_to_friday_minutes : ℕ) :
  tuesday_miles = 4 →
  wednesday_miles = 6 →
  thursday_miles = 8 →
  friday_miles = 3 →
  pace = 10 →
  total_hours = 4 →
  total_minutes = total_hours * 60 →
  total_tuesday_to_friday_miles = tuesday_miles + wednesday_miles + thursday_miles + friday_miles →
  total_tuesday_to_friday_minutes = total_tuesday_to_friday_miles * pace →
  monday_minutes = total_minutes - total_tuesday_to_friday_minutes →
  (monday_minutes / pace) = 3 := sorry

end Iggy_miles_on_Monday_l1188_118841


namespace estimate_number_of_trees_l1188_118880

-- Definitions derived from the conditions
def forest_length : ℝ := 100
def forest_width : ℝ := 0.5
def plot_length : ℝ := 1
def plot_width : ℝ := 0.5
def tree_counts : List ℕ := [65110, 63200, 64600, 64700, 67300, 63300, 65100, 66600, 62800, 65500]

-- The main theorem stating the problem
theorem estimate_number_of_trees :
  let avg_trees_per_plot := tree_counts.sum / tree_counts.length
  let total_plots := (forest_length * forest_width) / (plot_length * plot_width)
  avg_trees_per_plot * total_plots = 6482100 :=
by
  sorry

end estimate_number_of_trees_l1188_118880


namespace inequality_proof_l1188_118808

open Real

theorem inequality_proof {x y : ℝ} (hx : x < 0) (hy : y < 0) : 
    (x ^ 4 / y ^ 4) + (y ^ 4 / x ^ 4) - (x ^ 2 / y ^ 2) - (y ^ 2 / x ^ 2) + (x / y) + (y / x) >= 2 := 
by
    sorry

end inequality_proof_l1188_118808


namespace probability_three_specific_cards_l1188_118883

theorem probability_three_specific_cards :
  let deck_size := 52
  let diamonds := 13
  let spades := 13
  let hearts := 13
  let p1 := diamonds / deck_size
  let p2 := spades / (deck_size - 1)
  let p3 := hearts / (deck_size - 2)
  p1 * p2 * p3 = 169 / 5100 :=
by
  sorry

end probability_three_specific_cards_l1188_118883


namespace milk_butterfat_mixture_l1188_118888

theorem milk_butterfat_mixture (x gallons_50 gall_10_perc final_gall mixture_perc: ℝ)
    (H1 : gall_10_perc = 24) 
    (H2 : mixture_perc = 0.20 * (x + gall_10_perc))
    (H3 : 0.50 * x + 0.10 * gall_10_perc = 0.20 * (x + gall_10_perc)) 
    (H4 : final_gall = 20) :
    x = 8 :=
sorry

end milk_butterfat_mixture_l1188_118888


namespace part1_tangent_line_at_x1_part2_a_range_l1188_118864

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * x

theorem part1_tangent_line_at_x1 (a : ℝ) (h1 : a = 1) : 
  let f' (x : ℝ) : ℝ := (x + 1) * Real.exp x - 1
  (2 * Real.exp 1 - 1) * 1 - (f 1 1) = Real.exp 1 :=
by 
  sorry

theorem part2_a_range (a : ℝ) (h2 : ∀ x > 0, f x a ≥ Real.log x - x + 1) : 
  0 < a ∧ a ≤ 2 :=
by 
  sorry

end part1_tangent_line_at_x1_part2_a_range_l1188_118864


namespace shopkeeper_discount_l1188_118867

theorem shopkeeper_discount
  (CP LP SP : ℝ)
  (H_CP : CP = 100)
  (H_LP : LP = CP + 0.4 * CP)
  (H_SP : SP = CP + 0.33 * CP)
  (discount_percent : ℝ) :
  discount_percent = ((LP - SP) / LP) * 100 → discount_percent = 5 :=
by
  sorry

end shopkeeper_discount_l1188_118867


namespace fuel_consumption_rate_l1188_118825

theorem fuel_consumption_rate (fuel_left time_left r: ℝ) 
    (h_fuel: fuel_left = 6.3333) 
    (h_time: time_left = 0.6667) 
    (h_rate: r = fuel_left / time_left) : r = 9.5 := 
by
    sorry

end fuel_consumption_rate_l1188_118825


namespace Alina_messages_comparison_l1188_118889

theorem Alina_messages_comparison 
  (lucia_day1 : ℕ) (alina_day1 : ℕ) (lucia_day2 : ℕ) (alina_day2 : ℕ) (lucia_day3 : ℕ) (alina_day3 : ℕ)
  (h1 : lucia_day1 = 120)
  (h2 : alina_day1 = lucia_day1 - 20)
  (h3 : lucia_day2 = lucia_day1 / 3)
  (h4 : lucia_day3 = lucia_day1)
  (h5 : alina_day3 = alina_day1)
  (h6 : lucia_day1 + lucia_day2 + lucia_day3 + alina_day1 + alina_day2 + alina_day3 = 680) :
  alina_day2 = alina_day1 + 100 :=
sorry

end Alina_messages_comparison_l1188_118889


namespace emery_family_trip_l1188_118876

theorem emery_family_trip 
  (first_part_distance : ℕ) (first_part_time : ℕ) (total_time : ℕ) (speed : ℕ) (second_part_time : ℕ) :
  first_part_distance = 100 ∧ first_part_time = 1 ∧ total_time = 4 ∧ speed = 100 ∧ second_part_time = 3 →
  second_part_time * speed = 300 :=
by 
  sorry

end emery_family_trip_l1188_118876


namespace sequence_inequality_l1188_118835

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
    (h_subadd : ∀ m n : ℕ, a (n + m) ≤ a n + a m) :
  ∀ (n m : ℕ), m ≤ n → a n ≤ m * a 1 + ((n : ℝ) / m - 1) * a m := 
by
  intros n m hnm
  sorry

end sequence_inequality_l1188_118835


namespace train_cross_time_l1188_118872

-- Definitions from the conditions
def length_of_train : ℤ := 600
def speed_of_man_kmh : ℤ := 2
def speed_of_train_kmh : ℤ := 56

-- Conversion factors and speed conversion
def kmh_to_mph_factor : ℤ := 1000 / 3600 -- 1 km/hr = 0.27778 m/s approximately

def speed_of_man_ms : ℤ := speed_of_man_kmh * kmh_to_mph_factor -- Convert speed of man to m/s
def speed_of_train_ms : ℤ := speed_of_train_kmh * kmh_to_mph_factor -- Convert speed of train to m/s

-- Calculating relative speed
def relative_speed_ms : ℤ := speed_of_train_ms - speed_of_man_ms

-- Calculating the time taken to cross
def time_to_cross : ℤ := length_of_train / relative_speed_ms 

-- The theorem to prove
theorem train_cross_time : time_to_cross = 40 := 
by sorry

end train_cross_time_l1188_118872


namespace P_inter_Q_eq_l1188_118809

def P (x : ℝ) : Prop := -1 < x ∧ x < 3
def Q (x : ℝ) : Prop := -2 < x ∧ x < 1

theorem P_inter_Q_eq : {x | P x} ∩ {x | Q x} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end P_inter_Q_eq_l1188_118809


namespace transfer_people_correct_equation_l1188_118824

theorem transfer_people_correct_equation (A B x : ℕ) (h1 : A = 28) (h2 : B = 20) : 
  A + x = 2 * (B - x) := 
by sorry

end transfer_people_correct_equation_l1188_118824


namespace cost_per_sq_meter_l1188_118823

def tank_dimensions : ℝ × ℝ × ℝ := (25, 12, 6)
def total_plastering_cost : ℝ := 186
def total_plastering_area : ℝ :=
  let (length, width, height) := tank_dimensions
  let area_bottom := length * width
  let area_longer_walls := length * height * 2
  let area_shorter_walls := width * height * 2
  area_bottom + area_longer_walls + area_shorter_walls

theorem cost_per_sq_meter : total_plastering_cost / total_plastering_area = 0.25 := by
  sorry

end cost_per_sq_meter_l1188_118823


namespace value_of_r_for_n_3_l1188_118856

theorem value_of_r_for_n_3 :
  ∀ (r s : ℕ), 
  (r = 4^s + 3 * s) → 
  (s = 2^3 + 2) → 
  r = 1048606 :=
by
  intros r s h1 h2
  sorry

end value_of_r_for_n_3_l1188_118856


namespace relationship_between_lines_l1188_118843

-- Define the type for a line and a plane
structure Line where
  -- some properties (to be defined as needed, omitted for brevity)

structure Plane where
  -- some properties (to be defined as needed, omitted for brevity)

-- Define parallelism between a line and a plane
def parallel_line_plane (m : Line) (α : Plane) : Prop := sorry

-- Define line within a plane
def line_within_plane (n : Line) (α : Plane) : Prop := sorry

-- Define parallelism between two lines
def parallel_lines (m n : Line) : Prop := sorry

-- Define skewness between two lines
def skew_lines (m n : Line) : Prop := sorry

-- The mathematically equivalent proof problem
theorem relationship_between_lines (m n : Line) (α : Plane)
  (h1 : parallel_line_plane m α)
  (h2 : line_within_plane n α) :
  parallel_lines m n ∨ skew_lines m n := 
sorry

end relationship_between_lines_l1188_118843


namespace staircase_steps_180_toothpicks_l1188_118861

-- Condition definition: total number of toothpicks for \( n \) steps is \( n(n + 1) \)
def total_toothpicks (n : ℕ) : ℕ := n * (n + 1)

-- Theorem statement: for 180 toothpicks, the number of steps \( n \) is 12
theorem staircase_steps_180_toothpicks : ∃ n : ℕ, total_toothpicks n = 180 ∧ n = 12 :=
by sorry

end staircase_steps_180_toothpicks_l1188_118861


namespace minimum_perimeter_is_728_l1188_118858

noncomputable def minimum_common_perimeter (a b c : ℤ) (h1 : 2 * a + 18 * c = 2 * b + 20 * c)
  (h2 : 9 * c * Real.sqrt (a^2 - (9 * c)^2) = 10 * c * Real.sqrt (b^2 - (10 * c)^2)) 
  (h3 : a = b + c) : ℤ :=
2 * a + 18 * c

theorem minimum_perimeter_is_728 (a b c : ℤ) 
  (h1 : 2 * a + 18 * c = 2 * b + 20 * c) 
  (h2 : 9 * c * Real.sqrt (a^2 - (9 * c)^2) = 10 * c * Real.sqrt (b^2 - (10 * c)^2)) 
  (h3 : a = b + c) : 
  minimum_common_perimeter a b c h1 h2 h3 = 728 :=
sorry

end minimum_perimeter_is_728_l1188_118858


namespace rubber_duck_cost_l1188_118814

theorem rubber_duck_cost 
  (price_large : ℕ)
  (num_regular : ℕ)
  (num_large : ℕ)
  (total_revenue : ℕ)
  (h1 : price_large = 5)
  (h2 : num_regular = 221)
  (h3 : num_large = 185)
  (h4 : total_revenue = 1588) :
  ∃ (cost_regular : ℕ), (num_regular * cost_regular + num_large * price_large = total_revenue) ∧ cost_regular = 3 :=
by
  exists 3
  sorry

end rubber_duck_cost_l1188_118814


namespace midpoint_sum_is_correct_l1188_118805

theorem midpoint_sum_is_correct:
  let A := (10, 8)
  let B := (-4, -6)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (midpoint.1 + midpoint.2) = 4 :=
by
  sorry

end midpoint_sum_is_correct_l1188_118805


namespace correct_division_l1188_118870

theorem correct_division (a : ℝ) : a^8 / a^2 = a^6 := by 
  sorry

end correct_division_l1188_118870


namespace circle_reflection_l1188_118849

-- Definitions provided in conditions
def initial_center : ℝ × ℝ := (6, -5)
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.snd, p.fst)
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.fst, p.snd)

-- The final statement we need to prove
theorem circle_reflection :
  reflect_y_axis (reflect_y_eq_x initial_center) = (5, 6) :=
by
  -- By reflecting the point (6, -5) over y = x and then over the y-axis, we should get (5, 6)
  sorry

end circle_reflection_l1188_118849


namespace total_water_bottles_needed_l1188_118874

def number_of_people : ℕ := 4
def travel_time_one_way : ℕ := 8
def number_of_way : ℕ := 2
def water_consumption_per_hour : ℚ := 1 / 2

theorem total_water_bottles_needed : (number_of_people * (travel_time_one_way * number_of_way) * water_consumption_per_hour) = 32 := by
  sorry

end total_water_bottles_needed_l1188_118874


namespace value_of_f_neg_one_l1188_118848

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_neg_one (f_def : ∀ x, f (Real.tan x) = Real.sin (2 * x)) : f (-1) = -1 := 
by
sorry

end value_of_f_neg_one_l1188_118848


namespace visible_yellow_bus_length_correct_l1188_118800

noncomputable def red_bus_length : ℝ := 48
noncomputable def orange_car_length : ℝ := red_bus_length / 4
noncomputable def yellow_bus_length : ℝ := 3.5 * orange_car_length
noncomputable def green_truck_length : ℝ := 2 * orange_car_length
noncomputable def total_vehicle_length : ℝ := yellow_bus_length + green_truck_length
noncomputable def visible_yellow_bus_length : ℝ := 0.75 * yellow_bus_length

theorem visible_yellow_bus_length_correct :
  visible_yellow_bus_length = 31.5 := 
sorry

end visible_yellow_bus_length_correct_l1188_118800


namespace time_to_save_for_downpayment_l1188_118826

-- Definitions based on conditions
def annual_saving : ℝ := 0.10 * 150000
def downpayment : ℝ := 0.20 * 450000

-- Statement of the theorem to be proved
theorem time_to_save_for_downpayment (T : ℝ) (H1 : annual_saving = 15000) (H2 : downpayment = 90000) : 
  T = 6 :=
by
  -- Placeholder for the proof
  sorry

end time_to_save_for_downpayment_l1188_118826


namespace count_satisfying_pairs_l1188_118802

theorem count_satisfying_pairs :
  ∃ (count : ℕ), count = 540 ∧ 
  (∀ (w n : ℕ), (w % 23 = 5) ∧ (w < 450) ∧ (n % 17 = 7) ∧ (n < 450) → w < 450 ∧ n < 450) := 
by
  sorry

end count_satisfying_pairs_l1188_118802


namespace books_of_jason_l1188_118851

theorem books_of_jason (M J : ℕ) (hM : M = 42) (hTotal : M + J = 60) : J = 18 :=
by
  sorry

end books_of_jason_l1188_118851


namespace height_percentage_l1188_118878

theorem height_percentage (a b c : ℝ) 
  (h1 : a = 0.6 * b) 
  (h2 : c = 1.25 * a) : 
  (b - a) / a * 100 = 66.67 ∧ (c - a) / a * 100 = 25 := 
by 
  sorry

end height_percentage_l1188_118878


namespace roger_toys_l1188_118873

theorem roger_toys (initial_money spent_money toy_cost remaining_money toys : ℕ) 
  (h1 : initial_money = 63) 
  (h2 : spent_money = 48) 
  (h3 : toy_cost = 3) 
  (h4 : remaining_money = initial_money - spent_money) 
  (h5 : toys = remaining_money / toy_cost) : 
  toys = 5 := 
by 
  sorry

end roger_toys_l1188_118873


namespace sum_of_coefficients_l1188_118855

theorem sum_of_coefficients (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℤ) :
  (5 * x - 2) ^ 6 = b_6 * x ^ 6 + b_5 * x ^ 5 + b_4 * x ^ 4 + b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0 →
  b_6 + b_5 + b_4 + b_3 + b_2 + b_1 + b_0 = 729 :=
by
  sorry

end sum_of_coefficients_l1188_118855


namespace sum_of_fourth_powers_eq_82_l1188_118869

theorem sum_of_fourth_powers_eq_82 (x y : ℝ) (hx : x + y = -2) (hy : x * y = -3) :
  x^4 + y^4 = 82 :=
by
  sorry

end sum_of_fourth_powers_eq_82_l1188_118869


namespace arithmetic_geometric_sum_l1188_118834

def a (n : ℕ) : ℕ := 3 * n - 2
def b (n : ℕ) : ℕ := 3 ^ (n - 1)

theorem arithmetic_geometric_sum :
  a (b 1) + a (b 2) + a (b 3) = 33 := by
  sorry

end arithmetic_geometric_sum_l1188_118834


namespace parallel_line_plane_l1188_118815

-- Define vectors
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Dot product definition
def dotProduct (u v : Vector3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

-- Options given
def optionA : Vector3D × Vector3D := (⟨1, 0, 0⟩, ⟨-2, 0, 0⟩)
def optionB : Vector3D × Vector3D := (⟨1, 3, 5⟩, ⟨1, 0, 1⟩)
def optionC : Vector3D × Vector3D := (⟨0, 2, 1⟩, ⟨-1, 0, -1⟩)
def optionD : Vector3D × Vector3D := (⟨1, -1, 3⟩, ⟨0, 3, 1⟩)

-- Main theorem
theorem parallel_line_plane :
  (dotProduct (optionA.fst) (optionA.snd) ≠ 0) ∧
  (dotProduct (optionB.fst) (optionB.snd) ≠ 0) ∧
  (dotProduct (optionC.fst) (optionC.snd) ≠ 0) ∧
  (dotProduct (optionD.fst) (optionD.snd) = 0) :=
by
  -- Using sorry to skip the proof
  sorry

end parallel_line_plane_l1188_118815


namespace compute_value_of_expression_l1188_118859

theorem compute_value_of_expression (p q : ℝ) (hpq : 3 * p^2 - 5 * p - 8 = 0) (hq : 3 * q^2 - 5 * q - 8 = 0) (hneq : p ≠ q) :
  3 * (p^2 - q^2) / (p - q) = 5 :=
by
  have hpq_sum : p + q = 5 / 3 := sorry
  exact sorry

end compute_value_of_expression_l1188_118859


namespace amount_after_two_years_l1188_118896

noncomputable def amountAfterYears (presentValue : ℝ) (rate : ℝ) (n : ℕ) : ℝ :=
  presentValue * (1 + rate) ^ n

theorem amount_after_two_years 
  (presentValue : ℝ := 62000) 
  (rate : ℝ := 1 / 8) 
  (n : ℕ := 2) : 
  amountAfterYears presentValue rate n = 78468.75 := 
  sorry

end amount_after_two_years_l1188_118896


namespace geom_seq_product_equals_16_l1188_118836

theorem geom_seq_product_equals_16
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : ∀ m n, a (m + 1) - a m = a (n + 1) - a n)
  (non_zero_diff : ∃ d, d ≠ 0 ∧ ∀ n, a (n + 1) - a n = d)
  (h_cond : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0)
  (h_geom : ∀ m n, b (m + 1) / b m = b (n + 1) / b n)
  (h_b7 : b 7 = a 7):
  b 6 * b 8 = 16 := 
sorry

end geom_seq_product_equals_16_l1188_118836


namespace bob_cleaning_time_is_correct_l1188_118877

-- Definitions for conditions
def timeAliceTakes : ℕ := 32
def bobTimeFactor : ℚ := 3 / 4

-- Theorem to prove
theorem bob_cleaning_time_is_correct : (bobTimeFactor * timeAliceTakes : ℚ) = 24 := 
by
  sorry

end bob_cleaning_time_is_correct_l1188_118877


namespace remainder_div_13_l1188_118868

theorem remainder_div_13 {N : ℕ} (k : ℕ) (h : N = 39 * k + 18) : N % 13 = 5 := sorry

end remainder_div_13_l1188_118868


namespace find_a_iff_l1188_118821

def non_deg_ellipse (k : ℝ) : Prop :=
  ∀ x y : ℝ, 9 * (x^2) + (y^2) - 36 * x + 8 * y = k → 
  (∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0))

theorem find_a_iff (k : ℝ) : non_deg_ellipse k ↔ k > -52 := by
  sorry

end find_a_iff_l1188_118821


namespace problem_inequality_l1188_118827

theorem problem_inequality (a b c d : ℝ) (h1 : d ≥ 0) (h2 : a + b = 2) (h3 : c + d = 2) :
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 :=
by sorry

end problem_inequality_l1188_118827


namespace max_annual_profit_l1188_118853

noncomputable def annual_sales_volume (x : ℝ) : ℝ := - (1 / 3) * x^2 + 2 * x + 21

noncomputable def annual_sales_profit (x : ℝ) : ℝ := (- (1 / 3) * x^3 + 4 * x^2 + 9 * x - 126)

theorem max_annual_profit :
  ∀ x : ℝ, (x > 6) →
  (annual_sales_volume x) = - (1 / 3) * x^2 + 2 * x + 21 →
  (annual_sales_volume 10 = 23 / 3) →
  (21 - annual_sales_volume x = (1 / 3) * (x^2 - 6 * x)) →
    (annual_sales_profit x = - (1 / 3) * x^3 + 4 * x^2 + 9 * x - 126) ∧
    ∃ x_max : ℝ, 
      (annual_sales_profit x_max = 36) ∧
      x_max = 9 :=
by
  sorry

end max_annual_profit_l1188_118853


namespace abc_divisible_by_7_l1188_118850

theorem abc_divisible_by_7 (a b c : ℤ) (h : 7 ∣ (a^3 + b^3 + c^3)) : 7 ∣ (a * b * c) :=
sorry

end abc_divisible_by_7_l1188_118850


namespace watermelon_heavier_than_pineapple_l1188_118891

noncomputable def watermelon_weight : ℕ := 1 * 1000 + 300 -- Weight of one watermelon in grams
noncomputable def pineapple_weight : ℕ := 450 -- Weight of one pineapple in grams

theorem watermelon_heavier_than_pineapple :
    (4 * watermelon_weight = 5 * 1000 + 200) →
    (3 * watermelon_weight + 4 * pineapple_weight = 5 * 1000 + 700) →
    watermelon_weight - pineapple_weight = 850 :=
by
    intros h1 h2
    sorry

end watermelon_heavier_than_pineapple_l1188_118891


namespace pencil_weight_l1188_118811

theorem pencil_weight (total_weight : ℝ) (empty_case_weight : ℝ) (num_pencils : ℕ)
  (h1 : total_weight = 11.14) 
  (h2 : empty_case_weight = 0.5) 
  (h3 : num_pencils = 14) :
  (total_weight - empty_case_weight) / num_pencils = 0.76 := by
  sorry

end pencil_weight_l1188_118811


namespace probability_of_matching_pair_l1188_118844

noncomputable def num_socks := 22
noncomputable def red_socks := 12
noncomputable def blue_socks := 10

def ways_to_choose_two (n : ℕ) : ℕ :=
  n * (n - 1) / 2

noncomputable def probability_same_color : ℚ :=
  (ways_to_choose_two red_socks + ways_to_choose_two blue_socks : ℚ) / ways_to_choose_two num_socks

theorem probability_of_matching_pair :
  probability_same_color = 37 / 77 := 
by
  -- proof goes here
  sorry

end probability_of_matching_pair_l1188_118844


namespace time_to_cover_length_l1188_118871

/-- Define the conditions for the problem -/
def angle_deg : ℝ := 30
def escalator_speed : ℝ := 12
def length_along_incline : ℝ := 160
def person_speed : ℝ := 8

/-- Define the combined speed as the sum of the escalator speed and the person speed -/
def combined_speed : ℝ := escalator_speed + person_speed

/-- Theorem stating the time taken to cover the length of the escalator is 8 seconds -/
theorem time_to_cover_length : (length_along_incline / combined_speed) = 8 := by
  sorry

end time_to_cover_length_l1188_118871


namespace reciprocal_of_36_recurring_decimal_l1188_118806

-- Definitions and conditions
def recurring_decimal (x : ℚ) : Prop := x = 36 / 99

-- Theorem statement
theorem reciprocal_of_36_recurring_decimal :
  recurring_decimal (36 / 99) → (1 / (36 / 99) = 11 / 4) :=
sorry

end reciprocal_of_36_recurring_decimal_l1188_118806


namespace travel_time_difference_l1188_118887

theorem travel_time_difference :
  (160 / 40) - (280 / 40) = 3 := by
  sorry

end travel_time_difference_l1188_118887


namespace lattice_points_on_segment_l1188_118863

theorem lattice_points_on_segment : 
  let x1 := 5 
  let y1 := 23 
  let x2 := 47 
  let y2 := 297 
  ∃ n, n = 3 ∧ ∀ p : ℕ × ℕ, (p = (x1, y1) ∨ p = (x2, y2) ∨ ∃ t : ℕ, p = (x1 + t * (x2 - x1) / 2, y1 + t * (y2 - y1) / 2)) := 
sorry

end lattice_points_on_segment_l1188_118863


namespace train_pass_time_l1188_118830

noncomputable def train_speed_kmh := 36  -- Speed in km/hr
noncomputable def train_speed_ms := 10   -- Speed in m/s (converted)
noncomputable def platform_length := 180 -- Length of the platform in meters
noncomputable def platform_pass_time := 30 -- Time in seconds to pass platform
noncomputable def train_length := 120    -- Train length derived from conditions

theorem train_pass_time 
  (speed_in_kmh : ℕ) (speed_in_ms : ℕ) (platform_len : ℕ) (pass_platform_time : ℕ) (train_len : ℕ)
  (h1 : speed_in_kmh = 36)
  (h2 : speed_in_ms = 10)
  (h3 : platform_len = 180)
  (h4 : pass_platform_time = 30)
  (h5 : train_len = 120) :
  (train_len / speed_in_ms) = 12 := by
  sorry

end train_pass_time_l1188_118830
