import Mathlib

namespace total_dresses_l198_198338

theorem total_dresses (E M D S: ℕ) 
  (h1 : D = M + 12)
  (h2 : M = E / 2)
  (h3 : E = 16)
  (h4 : S = D - 5) : 
  E + M + D + S = 59 :=
by
  sorry

end total_dresses_l198_198338


namespace hyperbola_eccentricity_correct_l198_198917

open Real

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
    let PF1 := (12 * a / 5)
    let PF2 := PF1 - 2 * a
    let c := (2 * sqrt 37 * a / 5)
    sqrt (1 + (b^2 / a^2))  -- Assuming the geometric properties hold, the eccentricity should match
-- Lean function to verify the result
def verify_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
    hyperbola_eccentricity a b ha hb = sqrt 37 / 5

-- Statement to be verified
theorem hyperbola_eccentricity_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
    verify_eccentricity a b ha hb := sorry

end hyperbola_eccentricity_correct_l198_198917


namespace balloons_problem_l198_198238

variable (b_J b_S b_J_f b_g : ℕ)

theorem balloons_problem
  (h1 : b_J = 9)
  (h2 : b_S = 5)
  (h3 : b_J_f = 12)
  (h4 : b_g = (b_J + b_S) - b_J_f)
  : b_g = 2 :=
by {
  sorry
}

end balloons_problem_l198_198238


namespace unable_to_determine_questions_answered_l198_198991

variable (total_questions : ℕ) (total_time : ℕ) (used_time : ℕ) (remaining_time : ℕ)

theorem unable_to_determine_questions_answered (total_questions_eq : total_questions = 80)
  (total_time_eq : total_time = 60)
  (used_time_eq : used_time = 12)
  (remaining_time_eq : remaining_time = 0) :
  ∀ (answered_rate : ℕ → ℕ), ¬ ∃ questions_answered, answered_rate used_time = questions_answered :=
by sorry

end unable_to_determine_questions_answered_l198_198991


namespace area_bounded_region_l198_198922

theorem area_bounded_region : 
  (∃ x y : ℝ, x^2 + y^2 = 2 * abs (x - y) + 2 * abs (x + y)) →
  (bounded_area : ℝ) = 16 * Real.pi :=
by
  sorry

end area_bounded_region_l198_198922


namespace amy_hours_per_week_l198_198204

theorem amy_hours_per_week (hours_summer_per_week : ℕ) (weeks_summer : ℕ) (earnings_summer : ℕ)
  (weeks_school_year : ℕ) (earnings_school_year_goal : ℕ) :
  (hours_summer_per_week = 40) →
  (weeks_summer = 12) →
  (earnings_summer = 4800) →
  (weeks_school_year = 36) →
  (earnings_school_year_goal = 7200) →
  (∃ hours_school_year_per_week : ℕ, hours_school_year_per_week = 20) :=
by
  sorry

end amy_hours_per_week_l198_198204


namespace evaluate_expression_l198_198144

noncomputable def a : ℝ := 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def b : ℝ := -2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def c : ℝ := 2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def d : ℝ := -2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6

theorem evaluate_expression : (1/a + 1/b + 1/c + 1/d)^2 = 952576 / 70225 := by
  sorry

end evaluate_expression_l198_198144


namespace symmetry_about_origin_l198_198115

def f (x : ℝ) : ℝ := x^3 - x

theorem symmetry_about_origin : 
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end symmetry_about_origin_l198_198115


namespace smallest_positive_debt_resolves_l198_198560

theorem smallest_positive_debt_resolves :
  ∃ (c t : ℤ), (240 * c + 180 * t = 60) ∧ (60 > 0) :=
by
  sorry

end smallest_positive_debt_resolves_l198_198560


namespace total_toys_l198_198828

theorem total_toys (bill_toys hana_toys hash_toys: ℕ) 
  (hb: bill_toys = 60)
  (hh: hana_toys = (5 * bill_toys) / 6)
  (hs: hash_toys = (hana_toys / 2) + 9) :
  (bill_toys + hana_toys + hash_toys) = 144 :=
by
  sorry

end total_toys_l198_198828


namespace smallest_non_factor_product_of_factors_of_60_l198_198034

theorem smallest_non_factor_product_of_factors_of_60 :
  ∃ x y : ℕ, x ≠ y ∧ x ∣ 60 ∧ y ∣ 60 ∧ ¬ (x * y ∣ 60) ∧ ∀ x' y' : ℕ, x' ≠ y' → x' ∣ 60 → y' ∣ 60 → ¬(x' * y' ∣ 60) → x * y ≤ x' * y' := 
sorry

end smallest_non_factor_product_of_factors_of_60_l198_198034


namespace number_of_rectangles_in_5x5_grid_l198_198494

-- Number of ways to choose k elements from a set of n elements
def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def points_in_each_direction : ℕ := 5
def number_of_rectangles : ℕ :=
  binomial points_in_each_direction 2 * binomial points_in_each_direction 2

-- Lean statement to prove the problem
theorem number_of_rectangles_in_5x5_grid :
  number_of_rectangles = 100 :=
by
  -- begin Lean proof
  sorry

end number_of_rectangles_in_5x5_grid_l198_198494


namespace polynomial_coeff_diff_l198_198477

theorem polynomial_coeff_diff (a b c d e f : ℝ) :
  ((3*x + 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (a - b + c - d + e - f = 32) :=
by
  sorry

end polynomial_coeff_diff_l198_198477


namespace trapezoid_area_division_l198_198561

theorem trapezoid_area_division (AD BC MN : ℝ) (h₁ : AD = 4) (h₂ : BC = 3)
  (h₃ : MN > 0) (area_ratio : ∃ (S_ABMD S_MBCN : ℝ), MN/BC = (S_ABMD + S_MBCN)/(S_ABMD) ∧ (S_ABMD/S_MBCN = 2/5)) :
  MN = Real.sqrt 14 :=
by
  sorry

end trapezoid_area_division_l198_198561


namespace grasshopper_cannot_move_3_cm_grasshopper_can_move_2_cm_grasshopper_can_move_1234_cm_l198_198923

def can_jump (x : Int) : Prop :=
  ∃ (k m : Int), x = k * 36 + m * 14

theorem grasshopper_cannot_move_3_cm :
  ¬ can_jump 3 :=
by
  sorry

theorem grasshopper_can_move_2_cm :
  can_jump 2 :=
by
  sorry

theorem grasshopper_can_move_1234_cm :
  can_jump 1234 :=
by
  sorry

end grasshopper_cannot_move_3_cm_grasshopper_can_move_2_cm_grasshopper_can_move_1234_cm_l198_198923


namespace parabola_chord_ratio_is_3_l198_198343

noncomputable def parabola_chord_ratio (p : ℝ) (h : p > 0) : ℝ :=
  let focus_x := p / 2
  let a_x := (3 * p) / 2
  let b_x := p / 6
  let af := a_x + (p / 2)
  let bf := b_x + (p / 2)
  af / bf

theorem parabola_chord_ratio_is_3 (p : ℝ) (h : p > 0) : parabola_chord_ratio p h = 3 := by
  sorry

end parabola_chord_ratio_is_3_l198_198343


namespace find_a_of_odd_function_l198_198504

theorem find_a_of_odd_function (a : ℝ) (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_neg : ∀ x, x < 0 → f x = x^2 + a * x)
  (h_pos_value : f 2 = 6) : a = 5 := by
  sorry

end find_a_of_odd_function_l198_198504


namespace cities_real_distance_l198_198652

def map_scale := 7 -- number of centimeters representing 35 kilometers
def real_distance_equiv := 35 -- number of kilometers that corresponds to map_scale

def centimeters_per_kilometer := real_distance_equiv / map_scale -- kilometers per centimeter

def distance_on_map := 49 -- number of centimeters cities are separated by on the map

theorem cities_real_distance : distance_on_map * centimeters_per_kilometer = 245 :=
by
  sorry

end cities_real_distance_l198_198652


namespace sqrt_5_is_quadratic_radical_l198_198167

variable (a : ℝ) -- a is a real number

-- Definition to check if a given expression is a quadratic radical
def is_quadratic_radical (x : ℝ) : Prop := ∃ y : ℝ, y^2 = x

theorem sqrt_5_is_quadratic_radical : is_quadratic_radical 5 :=
by
  -- Here, 'by' indicates the start of the proof block,
  -- but the actual content of the proof is replaced with 'sorry' as instructed.
  sorry

end sqrt_5_is_quadratic_radical_l198_198167


namespace range_of_m_l198_198944

theorem range_of_m (x m : ℝ) (h1 : -1 ≤ x ∧ x ≤ 1) (h2 : |x - m| ≤ 2) : -1 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l198_198944


namespace leak_empties_tank_in_4_hours_l198_198769

theorem leak_empties_tank_in_4_hours
  (A_fills_in : ℝ)
  (A_with_leak_fills_in : ℝ) : 
  (∀ (L : ℝ), A_fills_in = 2 ∧ A_with_leak_fills_in = 4 → L = (1 / 4) → 1 / L = 4) :=
by 
  sorry

end leak_empties_tank_in_4_hours_l198_198769


namespace number_of_males_is_one_part_l198_198532

-- Define the total population
def population : ℕ := 480

-- Define the number of divided parts
def parts : ℕ := 3

-- Define the population part represented by one square.
def part_population (total_population : ℕ) (n_parts : ℕ) : ℕ :=
  total_population / n_parts

-- The Lean statement for the problem
theorem number_of_males_is_one_part : part_population population parts = 160 :=
by
  -- Proof omitted
  sorry

end number_of_males_is_one_part_l198_198532


namespace spaces_per_tray_l198_198071

-- Conditions
def num_ice_cubes_glass : ℕ := 8
def num_ice_cubes_pitcher : ℕ := 2 * num_ice_cubes_glass
def total_ice_cubes_used : ℕ := num_ice_cubes_glass + num_ice_cubes_pitcher
def num_trays : ℕ := 2

-- Proof statement
theorem spaces_per_tray : total_ice_cubes_used / num_trays = 12 :=
by
  sorry

end spaces_per_tray_l198_198071


namespace prime_power_of_n_l198_198550

theorem prime_power_of_n (n : ℕ) (h : Nat.Prime (4^n + 2^n + 1)) : ∃ k : ℕ, n = 3^k := 
sorry

end prime_power_of_n_l198_198550


namespace proof_problem_l198_198476

noncomputable def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 3 else 3 * n

noncomputable def sequence_b (n : ℕ) : ℕ :=
  3 ^ n

noncomputable def sequence_c (n : ℕ) : ℕ :=
  sequence_b (sequence_a n)

theorem proof_problem :
  sequence_c 2017 = 27 ^ 2017 :=
by sorry

end proof_problem_l198_198476


namespace hyperbola_range_of_m_l198_198340

theorem hyperbola_range_of_m (m : ℝ) : (∃ f : ℝ → ℝ → ℝ, ∀ x y: ℝ, f x y = (x^2 / (4 - m) - y^2 / (2 + m))) → (4 - m) * (2 + m) > 0 → -2 < m ∧ m < 4 :=
by
  intros h_eq h_cond
  sorry

end hyperbola_range_of_m_l198_198340


namespace find_denominator_l198_198903

theorem find_denominator (x : ℕ) (dec_form_of_frac_4128 : ℝ) (h1: 4128 / x = dec_form_of_frac_4128) 
    : x = 4387 :=
by
  have h: dec_form_of_frac_4128 = 0.9411764705882353 := sorry
  sorry

end find_denominator_l198_198903


namespace value_of_expression_l198_198588

-- Define the conditions
def x := -2
def y := 1
def z := 1
def w := 3

-- The main theorem statement
theorem value_of_expression : 
  (x^2 * y^2 * z^2) - (x^2 * y * z^2) + (y / w) * Real.sin (x * z) = - (1 / 3) * Real.sin 2 := by
  sorry

end value_of_expression_l198_198588


namespace length_of_second_train_l198_198032

theorem length_of_second_train 
  (length_first_train : ℝ) 
  (speed_first_train_kmph : ℝ) 
  (speed_second_train_kmph : ℝ) 
  (time_to_cross : ℝ) 
  (h1 : length_first_train = 400)
  (h2 : speed_first_train_kmph = 72)
  (h3 : speed_second_train_kmph = 36)
  (h4 : time_to_cross = 69.99440044796417) :
  let speed_first_train := speed_first_train_kmph * (1000 / 3600)
  let speed_second_train := speed_second_train_kmph * (1000 / 3600)
  let relative_speed := speed_first_train - speed_second_train
  let distance := relative_speed * time_to_cross
  let length_second_train := distance - length_first_train
  length_second_train = 299.9440044796417 :=
  by
    sorry

end length_of_second_train_l198_198032


namespace divides_f_of_nat_l198_198680

variable {n : ℕ}

theorem divides_f_of_nat (n : ℕ) : 5 ∣ (76 * n^5 + 115 * n^4 + 19 * n) := 
sorry

end divides_f_of_nat_l198_198680


namespace correct_propositions_count_l198_198364

theorem correct_propositions_count (a b : ℝ) :
  (∀ a b, a > b → a + 1 > b + 1) ∧
  (∀ a b, a > b → a - 1 > b - 1) ∧
  (∀ a b, a > b → -2 * a < -2 * b) ∧
  (¬ ∀ a b, a > b → 2 * a < 2 * b) → 
  3 = 3 :=
by
  intro h
  sorry

end correct_propositions_count_l198_198364


namespace connie_total_markers_l198_198643

theorem connie_total_markers (red_markers : ℕ) (blue_markers : ℕ) 
                              (h1 : red_markers = 41)
                              (h2 : blue_markers = 64) : 
                              red_markers + blue_markers = 105 := by
  sorry

end connie_total_markers_l198_198643


namespace remaining_cubes_l198_198150

-- The configuration of the initial cube and the properties of a layer
def initial_cube : ℕ := 10
def total_cubes : ℕ := 1000
def layer_cubes : ℕ := (initial_cube * initial_cube)

-- The proof problem: Prove that the remaining number of cubes is 900 after removing one layer
theorem remaining_cubes : total_cubes - layer_cubes = 900 := 
by 
  sorry

end remaining_cubes_l198_198150


namespace total_hours_worked_l198_198182

theorem total_hours_worked :
  (∃ (hours_per_day : ℕ) (days : ℕ), hours_per_day = 3 ∧ days = 6) →
  (∃ (total_hours : ℕ), total_hours = 18) :=
by
  intros
  sorry

end total_hours_worked_l198_198182


namespace value_range_of_quadratic_function_l198_198605

def quadratic_function (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_range_of_quadratic_function :
  (∀ x : ℝ, 1 < x ∧ x ≤ 4 → -1 < quadratic_function x ∧ quadratic_function x ≤ 3) :=
sorry

end value_range_of_quadratic_function_l198_198605


namespace product_of_prs_eq_60_l198_198220

theorem product_of_prs_eq_60 (p r s : ℕ) (h1 : 3 ^ p + 3 ^ 5 = 270) (h2 : 2 ^ r + 46 = 94) (h3 : 6 ^ s + 5 ^ 4 = 1560) :
  p * r * s = 60 :=
  sorry

end product_of_prs_eq_60_l198_198220


namespace evaluate_expression_l198_198850

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end evaluate_expression_l198_198850


namespace cost_backpack_is_100_l198_198534

-- Definitions based on the conditions
def cost_wallet : ℕ := 50
def cost_sneakers_per_pair : ℕ := 100
def num_sneakers_pairs : ℕ := 2
def cost_jeans_per_pair : ℕ := 50
def num_jeans_pairs : ℕ := 2
def total_spent : ℕ := 450

-- The problem statement
theorem cost_backpack_is_100 (x : ℕ) 
  (leonard_total : ℕ := cost_wallet + num_sneakers_pairs * cost_sneakers_per_pair) 
  (michael_non_backpack_total : ℕ := num_jeans_pairs * cost_jeans_per_pair) :
  total_spent = leonard_total + michael_non_backpack_total + x → x = 100 := 
by
  unfold cost_wallet cost_sneakers_per_pair num_sneakers_pairs total_spent cost_jeans_per_pair num_jeans_pairs
  intro h
  sorry

end cost_backpack_is_100_l198_198534


namespace max_z_value_l198_198451

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) : z = 13 / 3 := 
sorry

end max_z_value_l198_198451


namespace triangle_area_in_circle_l198_198296

theorem triangle_area_in_circle (r : ℝ) (arc1 arc2 arc3 : ℝ) 
  (circumference_eq : arc1 + arc2 + arc3 = 24)
  (radius_eq : 2 * Real.pi * r = 24) : 
  1 / 2 * (r ^ 2) * (Real.sin (105 * Real.pi / 180) + Real.sin (120 * Real.pi / 180) + Real.sin (135 * Real.pi / 180)) = 364.416 / (Real.pi ^ 2) :=
by
  sorry

end triangle_area_in_circle_l198_198296


namespace jorge_total_goals_l198_198294

theorem jorge_total_goals (last_season_goals current_season_goals : ℕ) (h_last : last_season_goals = 156) (h_current : current_season_goals = 187) : 
  last_season_goals + current_season_goals = 343 :=
by
  sorry

end jorge_total_goals_l198_198294


namespace find_circumference_l198_198297

theorem find_circumference
  (C : ℕ)
  (h1 : ∃ (vA vB : ℕ), C > 0 ∧ vA > 0 ∧ vB > 0 ∧ 
                        (120 * (C/2 + 80)) = ((C - 80) * (C/2 - 120)) ∧
                        (C - 240) / vA = (C + 240) / vB) :
  C = 520 := 
  sorry

end find_circumference_l198_198297


namespace closest_fraction_to_team_aus_medals_l198_198234

theorem closest_fraction_to_team_aus_medals 
  (won_medals : ℕ) (total_medals : ℕ) 
  (choices : List ℚ)
  (fraction_won : ℚ)
  (c1 : won_medals = 28)
  (c2 : total_medals = 150)
  (c3 : choices = [1/4, 1/5, 1/6, 1/7, 1/8])
  (c4 : fraction_won = 28 / 150) :
  abs (fraction_won - 1/5) < abs (fraction_won - 1/4) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/6) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/7) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/8) := 
sorry

end closest_fraction_to_team_aus_medals_l198_198234


namespace binkie_gemstones_l198_198168

noncomputable def gemstones_solution : ℕ :=
sorry

theorem binkie_gemstones : ∀ (Binkie Frankie Spaatz Whiskers Snowball : ℕ),
  Spaatz = 1 ∧
  Whiskers = Spaatz + 3 ∧
  Snowball = 2 * Whiskers ∧ 
  Snowball % 2 = 0 ∧
  Whiskers % 2 = 0 ∧
  Spaatz = (1 / 2 * Frankie) - 2 ∧
  Binkie = 4 * Frankie ∧
  Binkie + Frankie + Spaatz + Whiskers + Snowball <= 50 →
  Binkie = 24 :=
sorry

end binkie_gemstones_l198_198168


namespace abs_distance_equation_1_abs_distance_equation_2_l198_198531

theorem abs_distance_equation_1 (x : ℚ) : |x - (3 : ℚ)| = 5 ↔ x = 8 ∨ x = -2 := 
sorry

theorem abs_distance_equation_2 (x : ℚ) : |x - (3 : ℚ)| = |x + (1 : ℚ)| ↔ x = 1 :=
sorry

end abs_distance_equation_1_abs_distance_equation_2_l198_198531


namespace probability_of_same_suit_or_number_but_not_both_l198_198933

def same_suit_or_number_but_not_both : Prop :=
  let total_outcomes := 52 * 52
  let prob_same_suit := 12 / 51
  let prob_same_number := 3 / 51
  let prob_same_suit_and_number := 1 / 51
  (prob_same_suit + prob_same_number - 2 * prob_same_suit_and_number) = 15 / 52

theorem probability_of_same_suit_or_number_but_not_both :
  same_suit_or_number_but_not_both :=
by sorry

end probability_of_same_suit_or_number_but_not_both_l198_198933


namespace total_insects_eaten_l198_198546

theorem total_insects_eaten :
  let geckos := 5
  let insects_per_gecko := 6
  let lizards := 3
  let insects_per_lizard := 2 * insects_per_gecko
  let total_insects := geckos * insects_per_gecko + lizards * insects_per_lizard
  total_insects = 66 := by
  sorry

end total_insects_eaten_l198_198546


namespace find_number_l198_198566

theorem find_number (x : ℝ) : 35 + 3 * x^2 = 89 ↔ x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2 := by
  sorry

end find_number_l198_198566


namespace find_larger_number_l198_198530

theorem find_larger_number (x y : ℤ) (h1 : 5 * y = 6 * x) (h2 : y - x = 12) : y = 72 :=
sorry

end find_larger_number_l198_198530


namespace train_passing_time_l198_198843

-- conditions
def train_length := 490 -- in meters
def train_speed_kmh := 63 -- in kilometers per hour
def conversion_factor := 1000 / 3600 -- to convert km/hr to m/s

-- conversion
def train_speed_ms := train_speed_kmh * conversion_factor -- speed in meters per second

-- expected correct answer
def expected_time := 28 -- in seconds

-- Theorem statement
theorem train_passing_time :
  train_length / train_speed_ms = expected_time :=
by
  sorry

end train_passing_time_l198_198843


namespace nancy_spelling_problems_l198_198458

structure NancyProblems where
  math_problems : ℝ
  rate : ℝ
  hours : ℝ
  total_problems : ℝ

noncomputable def calculate_spelling_problems (n : NancyProblems) : ℝ :=
  n.total_problems - n.math_problems

theorem nancy_spelling_problems :
  ∀ (n : NancyProblems), n.math_problems = 17.0 ∧ n.rate = 8.0 ∧ n.hours = 4.0 ∧ n.total_problems = 32.0 →
  calculate_spelling_problems n = 15.0 :=
by
  intros
  sorry

end nancy_spelling_problems_l198_198458


namespace determinant_of_non_right_triangle_l198_198074

theorem determinant_of_non_right_triangle (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
(h_sum_ABC : A + B + C = π) :
  Matrix.det ![
    ![2 * Real.sin A, 1, 1],
    ![1, 2 * Real.sin B, 1],
    ![1, 1, 2 * Real.sin C]
  ] = 2 := by
  sorry

end determinant_of_non_right_triangle_l198_198074


namespace surface_area_of_cube_given_sphere_surface_area_l198_198218

noncomputable def edge_length_of_cube (sphere_surface_area : ℝ) : ℝ :=
  let a_square := 2
  Real.sqrt a_square

def surface_area_of_cube (a : ℝ) : ℝ :=
  6 * a^2

theorem surface_area_of_cube_given_sphere_surface_area (sphere_surface_area : ℝ) :
  sphere_surface_area = 6 * Real.pi → 
  surface_area_of_cube (edge_length_of_cube sphere_surface_area) = 12 :=
by
  sorry

end surface_area_of_cube_given_sphere_surface_area_l198_198218


namespace complement_A_l198_198023

-- Definitions for the conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 1}

-- Proof statement
theorem complement_A : (U \ A) = {x | x ≥ 1} := by
  sorry

end complement_A_l198_198023


namespace y_value_for_equations_l198_198082

theorem y_value_for_equations (x y : ℝ) (h1 : x^2 + y^2 = 25) (h2 : x^2 + y = 10) :
  y = (1 - Real.sqrt 61) / 2 := by
  sorry

end y_value_for_equations_l198_198082


namespace number_of_pies_l198_198105

-- Definitions based on the conditions
def box_weight : ℕ := 120
def weight_for_applesauce : ℕ := box_weight / 2
def weight_per_pie : ℕ := 4
def remaining_weight : ℕ := box_weight - weight_for_applesauce

-- The proof problem statement
theorem number_of_pies : (remaining_weight / weight_per_pie) = 15 :=
by
  sorry

end number_of_pies_l198_198105


namespace find_fraction_value_l198_198634

variable {x y : ℂ}

theorem find_fraction_value
    (h1 : (x^2 + y^2) / (x + y) = 4)
    (h2 : (x^4 + y^4) / (x^3 + y^3) = 2) :
    (x^6 + y^6) / (x^5 + y^5) = 4 := by
  sorry

end find_fraction_value_l198_198634


namespace gcd_at_most_3_digits_l198_198247

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l198_198247


namespace number_of_people_and_price_l198_198589

theorem number_of_people_and_price 
  (x y : ℤ) 
  (h1 : 8 * x - y = 3) 
  (h2 : y - 7 * x = 4) : 
  x = 7 ∧ y = 53 :=
by
  sorry

end number_of_people_and_price_l198_198589


namespace probability_of_cold_given_rhinitis_l198_198121

/-- Define the events A and B as propositions --/
def A : Prop := sorry -- A represents having rhinitis
def B : Prop := sorry -- B represents having a cold

/-- Define the given probabilities as assumptions --/
axiom P_A : ℝ -- P(A) = 0.8
axiom P_A_and_B : ℝ -- P(A ∩ B) = 0.6

/-- Adding the conditions --/
axiom P_A_val : P_A = 0.8
axiom P_A_and_B_val : P_A_and_B = 0.6

/-- Define the conditional probability --/
noncomputable def P_B_given_A : ℝ := P_A_and_B / P_A

/-- The main theorem which states the problem --/
theorem probability_of_cold_given_rhinitis : P_B_given_A = 0.75 :=
by 
  sorry

end probability_of_cold_given_rhinitis_l198_198121


namespace extreme_points_inequality_l198_198301

noncomputable def f (x : ℝ) (m : ℝ) := (1 / 2) * x^2 + m * Real.log (1 - x)

theorem extreme_points_inequality (m x1 x2 : ℝ) 
  (h_m1 : 0 < m) (h_m2 : m < 1 / 4)
  (h_x1 : 0 < x1) (h_x2: x1 < 1 / 2)
  (h_x3: x2 > 1 / 2) (h_x4: x2 < 1)
  (h_x5 : x1 < x2)
  (h_sum : x1 + x2 = 1)
  (h_prod : x1 * x2 = m)
  : (1 / 4) - (1 / 2) * Real.log 2 < (f x1 m) / x2 ∧ (f x1 m) / x2 < 0 :=
by
  sorry

end extreme_points_inequality_l198_198301


namespace probability_of_selected_member_l198_198265

section Probability

variables {N : ℕ} -- Total number of members in the group

-- Conditions
-- Probabilities of selecting individuals by gender
def P_woman : ℝ := 0.70
def P_man : ℝ := 0.20
def P_non_binary : ℝ := 0.10

-- Conditional probabilities of occupations given gender
def P_engineer_given_woman : ℝ := 0.20
def P_doctor_given_man : ℝ := 0.20
def P_translator_given_non_binary : ℝ := 0.20

-- The main proof statement
theorem probability_of_selected_member :
  (P_woman * P_engineer_given_woman) + (P_man * P_doctor_given_man) + (P_non_binary * P_translator_given_non_binary) = 0.20 :=
by
  sorry

end Probability

end probability_of_selected_member_l198_198265


namespace min_jumps_required_to_visit_all_points_and_return_l198_198212

theorem min_jumps_required_to_visit_all_points_and_return :
  ∀ (n : ℕ), n = 2016 →
  ∀ jumps : ℕ → ℕ, (∀ i, jumps i = 2 ∨ jumps i = 3) →
  (∀ i, (jumps (i + 1) + jumps (i + 2)) % n = 0) →
  ∃ (min_jumps : ℕ), min_jumps = 2017 :=
by
  sorry

end min_jumps_required_to_visit_all_points_and_return_l198_198212


namespace find_x_floor_mul_eq_100_l198_198921

theorem find_x_floor_mul_eq_100 (x : ℝ) (h1 : 0 < x) (h2 : (⌊x⌋ : ℝ) * x = 100) : x = 10 :=
by
  sorry

end find_x_floor_mul_eq_100_l198_198921


namespace line_circle_chord_shortest_l198_198203

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

noncomputable def line_l (x y m : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem line_circle_chord_shortest (m : ℝ) :
  (∀ x y : ℝ, circle_C x y → line_l x y m → m = -3 / 4) :=
sorry

end line_circle_chord_shortest_l198_198203


namespace initial_budget_calculation_l198_198844

variable (flaskCost testTubeCost safetyGearCost totalExpenses remainingAmount initialBudget : ℕ)

theorem initial_budget_calculation (h1 : flaskCost = 150)
                               (h2 : testTubeCost = 2 * flaskCost / 3)
                               (h3 : safetyGearCost = testTubeCost / 2)
                               (h4 : totalExpenses = flaskCost + testTubeCost + safetyGearCost)
                               (h5 : remainingAmount = 25)
                               (h6 : initialBudget = totalExpenses + remainingAmount) :
                               initialBudget = 325 := by
  sorry

end initial_budget_calculation_l198_198844


namespace base_133_not_perfect_square_l198_198459

theorem base_133_not_perfect_square (b : ℤ) : ¬ ∃ k : ℤ, b^2 + 3 * b + 3 = k^2 := by
  sorry

end base_133_not_perfect_square_l198_198459


namespace fluctuations_B_greater_than_A_l198_198187

variable (A B : Type)
variable (mean_A mean_B : ℝ)
variable (var_A var_B : ℝ)

-- Given conditions
axiom avg_A : mean_A = 5
axiom avg_B : mean_B = 5
axiom variance_A : var_A = 0.1
axiom variance_B : var_B = 0.2

-- The proof problem statement
theorem fluctuations_B_greater_than_A : var_A < var_B :=
by sorry

end fluctuations_B_greater_than_A_l198_198187


namespace solve_system_l198_198545

theorem solve_system (x y z : ℤ) 
  (h1 : x + 3 * y = 20)
  (h2 : x + y + z = 25)
  (h3 : x - z = 5) : 
  x = 14 ∧ y = 2 ∧ z = 9 := 
  sorry

end solve_system_l198_198545


namespace inequality_proof_l198_198342

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
  (ab / Real.sqrt (c^2 + 3)) + (bc / Real.sqrt (a^2 + 3)) + (ca / Real.sqrt (b^2 + 3)) ≤ 3 / 2 :=
by
  sorry

end inequality_proof_l198_198342


namespace relationship_between_u_and_v_l198_198192

variables {r u v p : ℝ}
variables (AB G : ℝ)

theorem relationship_between_u_and_v (hAB : AB = 2 * r) (hAG_GF : u = (p^2 / (2 * r)) - p) :
    v^2 = u^3 / (2 * r - u) :=
sorry

end relationship_between_u_and_v_l198_198192


namespace unique_prime_value_l198_198354

def T : ℤ := 2161

theorem unique_prime_value :
  ∃ p : ℕ, (∃ n : ℤ, n^4 - 898 * n^2 + T - 2160 = p) ∧ Prime p ∧ (∀ q, (∃ n : ℤ, n^4 - 898 * n^2 + T - 2160 = q) → q = p) :=
  sorry

end unique_prime_value_l198_198354


namespace total_savings_l198_198191

-- Definition to specify the denomination of each bill
def bill_value : ℕ := 100

-- Condition: Number of $100 bills Michelle has
def num_bills : ℕ := 8

-- The theorem to prove the total savings amount
theorem total_savings : num_bills * bill_value = 800 :=
by
  sorry

end total_savings_l198_198191


namespace range_of_a_l198_198353

noncomputable def common_point_ellipse_parabola (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y

theorem range_of_a : ∀ a : ℝ, common_point_ellipse_parabola a → -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end range_of_a_l198_198353


namespace no_common_root_l198_198839

theorem no_common_root 
  (a b : ℚ) 
  (α : ℂ) 
  (h1 : α^5 = α + 1) 
  (h2 : α^2 = -a * α - b) : 
  False :=
sorry

end no_common_root_l198_198839


namespace intersection_A_B_complement_l198_198905

def universal_set : Set ℝ := {x : ℝ | True}
def A : Set ℝ := {x : ℝ | x^2 - 2 * x < 0}
def B : Set ℝ := {x : ℝ | x > 1}
def B_complement : Set ℝ := {x : ℝ | x ≤ 1}

theorem intersection_A_B_complement :
  (A ∩ B_complement) = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_A_B_complement_l198_198905


namespace kite_area_overlap_l198_198603

theorem kite_area_overlap (beta : Real) (h_beta : beta ≠ 0 ∧ beta ≠ π) : 
  ∃ (A : Real), A = 1 / Real.sin beta := by
  sorry

end kite_area_overlap_l198_198603


namespace total_baseball_cards_is_100_l198_198495

-- Define the initial number of baseball cards Mike has
def initial_baseball_cards : ℕ := 87

-- Define the number of baseball cards Sam gave to Mike
def given_baseball_cards : ℕ := 13

-- Define the total number of baseball cards Mike has now
def total_baseball_cards : ℕ := initial_baseball_cards + given_baseball_cards

-- State the theorem that the total number of baseball cards is 100
theorem total_baseball_cards_is_100 : total_baseball_cards = 100 := by
  sorry

end total_baseball_cards_is_100_l198_198495


namespace compute_P_2_4_8_l198_198948

noncomputable def P : ℝ → ℝ → ℝ → ℝ := sorry

axiom homogeneity (x y z k : ℝ) : P (k * x) (k * y) (k * z) = (k ^ 4) * P x y z

axiom symmetry (a b c : ℝ) : P a b c = P b c a

axiom zero_cond (a b : ℝ) : P a a b = 0

axiom initial_cond : P 1 2 3 = 1

theorem compute_P_2_4_8 : P 2 4 8 = 56 := sorry

end compute_P_2_4_8_l198_198948


namespace logarithmic_expression_evaluation_l198_198592

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem logarithmic_expression_evaluation : 
  log_base_10 (5 / 2) + 2 * log_base_10 2 - (1/2)⁻¹ = -1 := 
by 
  sorry

end logarithmic_expression_evaluation_l198_198592


namespace corrected_mean_l198_198355

theorem corrected_mean (n : ℕ) (mean : ℝ) (obs1 obs2 : ℝ) (inc1 inc2 cor1 cor2 : ℝ)
    (h_num_obs : n = 50)
    (h_initial_mean : mean = 36)
    (h_incorrect1 : inc1 = 23) (h_correct1 : cor1 = 34)
    (h_incorrect2 : inc2 = 55) (h_correct2 : cor2 = 45)
    : (mean * n + (cor1 - inc1) + (cor2 - inc2)) / n = 36.02 := 
by 
  -- Insert steps to prove the theorem here
  sorry

end corrected_mean_l198_198355


namespace min_expression_value_l198_198100

theorem min_expression_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + 2 * b = 1) : 
  ∃ x, (x = (a^2 + 1) / a + (2 * b^2 + 1) / b) ∧ x = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end min_expression_value_l198_198100


namespace dice_probability_l198_198436

theorem dice_probability :
  let prob_roll_less_than_four := 3 / 6
  let prob_roll_even := 3 / 6
  let prob_roll_greater_than_four := 2 / 6
  prob_roll_less_than_four * prob_roll_even * prob_roll_greater_than_four = 1 / 12 :=
by
  sorry

end dice_probability_l198_198436


namespace expression_value_l198_198972

theorem expression_value :
  2 - (-3) - 4 - (-5) - 6 - (-7) * 2 = 14 :=
by sorry

end expression_value_l198_198972


namespace print_time_l198_198785

/-- Define the number of pages per minute printed by the printer -/
def pages_per_minute : ℕ := 25

/-- Define the total number of pages to be printed -/
def total_pages : ℕ := 350

/-- Prove that the time to print 350 pages at a rate of 25 pages per minute is 14 minutes -/
theorem print_time :
  (total_pages / pages_per_minute) = 14 :=
by
  sorry

end print_time_l198_198785


namespace Danny_shorts_washed_l198_198394

-- Define the given conditions
def Cally_white_shirts : ℕ := 10
def Cally_colored_shirts : ℕ := 5
def Cally_shorts : ℕ := 7
def Cally_pants : ℕ := 6

def Danny_white_shirts : ℕ := 6
def Danny_colored_shirts : ℕ := 8
def Danny_pants : ℕ := 6

def total_clothes_washed : ℕ := 58

-- Calculate total clothes washed by Cally
def total_cally_clothes : ℕ := 
  Cally_white_shirts + Cally_colored_shirts + Cally_shorts + Cally_pants

-- Calculate total clothes washed by Danny (excluding shorts)
def total_danny_clothes_excl_shorts : ℕ := 
  Danny_white_shirts + Danny_colored_shirts + Danny_pants

-- Define the statement to be proven
theorem Danny_shorts_washed : 
  total_clothes_washed - (total_cally_clothes + total_danny_clothes_excl_shorts) = 10 := by
  sorry

end Danny_shorts_washed_l198_198394


namespace evaluate_expression_l198_198431

theorem evaluate_expression : (18 * 3 + 6) / (6 - 3) = 20 := by
  sorry

end evaluate_expression_l198_198431


namespace average_weight_bc_is_43_l198_198892

variable (a b c : ℝ)

-- Definitions of the conditions
def average_weight_abc (a b c : ℝ) : Prop := (a + b + c) / 3 = 45
def average_weight_ab (a b : ℝ) : Prop := (a + b) / 2 = 40
def weight_b (b : ℝ) : Prop := b = 31

-- The theorem to prove
theorem average_weight_bc_is_43 :
  ∀ (a b c : ℝ), average_weight_abc a b c → average_weight_ab a b → weight_b b → (b + c) / 2 = 43 :=
by
  intros a b c h_average_weight_abc h_average_weight_ab h_weight_b
  sorry

end average_weight_bc_is_43_l198_198892


namespace find_unknown_number_l198_198607

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l198_198607


namespace katrina_tax_deduction_l198_198391

variable (hourlyWage : ℚ) (taxRate : ℚ)

def wageInCents (wage : ℚ) : ℚ := wage * 100
def taxInCents (wageInCents : ℚ) (rate : ℚ) : ℚ := wageInCents * rate / 100

theorem katrina_tax_deduction : 
  hourlyWage = 25 ∧ taxRate = 2.5 → taxInCents (wageInCents hourlyWage) taxRate = 62.5 := 
by 
  sorry

end katrina_tax_deduction_l198_198391


namespace scientific_notation_of_1650000_l198_198617

theorem scientific_notation_of_1650000 : (1650000 : ℝ) = 1.65 * 10^6 := 
by {
  -- Proof goes here
  sorry
}

end scientific_notation_of_1650000_l198_198617


namespace radius_of_circle_with_area_3_14_l198_198249

theorem radius_of_circle_with_area_3_14 (A : ℝ) (π : ℝ) (hA : A = 3.14) (hπ : π = 3.14) (h_area : A = π * r^2) : r = 1 :=
by
  sorry

end radius_of_circle_with_area_3_14_l198_198249


namespace fraction_zero_implies_x_zero_l198_198613

theorem fraction_zero_implies_x_zero (x : ℝ) (h : (x^2 - x) / (x - 1) = 0) (h₁ : x ≠ 1) : x = 0 := by
  sorry

end fraction_zero_implies_x_zero_l198_198613


namespace distance_to_building_materials_l198_198694

theorem distance_to_building_materials (D : ℝ) 
  (h1 : 2 * 10 * 4 * D = 8000) : 
  D = 100 := 
by
  sorry

end distance_to_building_materials_l198_198694


namespace apples_given_by_nathan_l198_198581

theorem apples_given_by_nathan (initial_apples : ℕ) (total_apples : ℕ) (given_by_nathan : ℕ) :
  initial_apples = 6 → total_apples = 12 → given_by_nathan = (total_apples - initial_apples) → given_by_nathan = 6 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end apples_given_by_nathan_l198_198581


namespace product_of_four_consecutive_integers_is_not_square_l198_198518

theorem product_of_four_consecutive_integers_is_not_square (n : ℤ) : 
  ¬ ∃ k : ℤ, k * k = (n-1)*n*(n+1)*(n+2) :=
sorry

end product_of_four_consecutive_integers_is_not_square_l198_198518


namespace natural_number_between_squares_l198_198742

open Nat

theorem natural_number_between_squares (n m k l : ℕ)
  (h1 : n > m^2)
  (h2 : n < (m+1)^2)
  (h3 : n - k = m^2)
  (h4 : n + l = (m+1)^2) : ∃ x : ℕ, n - k * l = x^2 := by
  sorry

end natural_number_between_squares_l198_198742


namespace two_point_four_times_eight_point_two_l198_198004

theorem two_point_four_times_eight_point_two (x y z : ℝ) (hx : x = 2.4) (hy : y = 8.2) (hz : z = 4.8 + 5.2) :
  x * y * z = 2.4 * 8.2 * 10 ∧ abs (x * y * z - 200) < abs (x * y * z - 150) ∧
  abs (x * y * z - 200) < abs (x * y * z - 250) ∧
  abs (x * y * z - 200) < abs (x * y * z - 300) ∧
  abs (x * y * z - 200) < abs (x * y * z - 350) := by
  sorry

end two_point_four_times_eight_point_two_l198_198004


namespace largest_two_digit_number_with_remainder_2_div_13_l198_198771

theorem largest_two_digit_number_with_remainder_2_div_13 : 
  ∃ (N : ℕ), (10 ≤ N ∧ N ≤ 99) ∧ N % 13 = 2 ∧ ∀ (M : ℕ), (10 ≤ M ∧ M ≤ 99) ∧ M % 13 = 2 → M ≤ N :=
  sorry

end largest_two_digit_number_with_remainder_2_div_13_l198_198771


namespace all_elements_rational_l198_198646

open Set

def finite_set_in_interval (n : ℕ) : Set ℝ :=
  {x | ∃ i, i ∈ Finset.range (n + 1) ∧ (x = 0 ∨ x = 1 ∨ 0 < x ∧ x < 1)}

def unique_distance_condition (S : Set ℝ) : Prop :=
  ∀ d, d ≠ 1 → ∃ x_i x_j x_k x_l, x_i ∈ S ∧ x_j ∈ S ∧ x_k ∈ S ∧ x_l ∈ S ∧ 
        abs (x_i - x_j) = d ∧ abs (x_k - x_l) = d ∧ (x_i = x_k → x_j ≠ x_l)

theorem all_elements_rational
  (n : ℕ)
  (S : Set ℝ)
  (hS1 : ∀ x ∈ S, 0 ≤ x ∧ x ≤ 1)
  (hS2 : 0 ∈ S)
  (hS3 : 1 ∈ S)
  (hS4 : unique_distance_condition S) :
  ∀ x ∈ S, ∃ q : ℚ, (x : ℝ) = q := 
sorry

end all_elements_rational_l198_198646


namespace MapleLeafHigh_points_l198_198849

def MapleLeafHigh (x y : ℕ) : Prop :=
  (1/3 * x + 3/8 * x + 18 + y = x) ∧ (10 ≤ y) ∧ (y ≤ 30)

theorem MapleLeafHigh_points : ∃ y, MapleLeafHigh 104 y ∧ y = 21 := 
by
  use 21
  sorry

end MapleLeafHigh_points_l198_198849


namespace roots_quadratic_eq_identity1_roots_quadratic_eq_identity2_l198_198749

variables {α : Type*} [Field α] (a b c x1 x2 : α)

theorem roots_quadratic_eq_identity1 (h_eq_roots: ∀ x, a * x^2 + b * x + c = 0 → (x = x1 ∨ x = x2)) 
(h_root1: a * x1^2 + b * x1 + c = 0) (h_root2: a * x2^2 + b * x2 + c = 0) :
  x1^2 + x2^2 = (b^2 - 2 * a * c) / a^2 :=
sorry

theorem roots_quadratic_eq_identity2 (h_eq_roots: ∀ x, a * x^2 + b * x + c = 0 → (x = x1 ∨ x = x2)) 
(h_root1: a * x1^2 + b * x1 + c = 0) (h_root2: a * x2^2 + b * x2 + c = 0) :
  x1^3 + x2^3 = (3 * a * b * c - b^3) / a^3 :=
sorry

end roots_quadratic_eq_identity1_roots_quadratic_eq_identity2_l198_198749


namespace games_against_other_division_l198_198522

theorem games_against_other_division
  (N M : ℕ) (h1 : N > 2 * M) (h2 : M > 5)
  (total_games : N * 4 + 5 * M = 82) :
  5 * M = 30 :=
by
  sorry

end games_against_other_division_l198_198522


namespace eval_expr_l198_198151

theorem eval_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  (x^(2 * y) * y^(3 * x) / (y^(2 * y) * x^(3 * x))) = x^(2 * y - 3 * x) * y^(3 * x - 2 * y) :=
by
  sorry

end eval_expr_l198_198151


namespace only_solution_is_2_3_7_l198_198070

theorem only_solution_is_2_3_7 (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h4 : c ∣ (a * b + 1)) (h5 : a ∣ (b * c + 1)) (h6 : b ∣ (c * a + 1)) :
  (a = 2 ∧ b = 3 ∧ c = 7) ∨ (a = 3 ∧ b = 7 ∧ c = 2) ∨ (a = 7 ∧ b = 2 ∧ c = 3) ∨
  (a = 2 ∧ b = 7 ∧ c = 3) ∨ (a = 7 ∧ b = 3 ∧ c = 2) ∨ (a = 3 ∧ b = 2 ∧ c = 7) :=
  sorry

end only_solution_is_2_3_7_l198_198070


namespace coursework_materials_spending_l198_198003

def budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

theorem coursework_materials_spending : 
    budget - (budget * food_percentage + budget * accommodation_percentage + budget * entertainment_percentage) = 300 := 
by 
  -- steps you would use to prove this
  sorry

end coursework_materials_spending_l198_198003


namespace banana_distribution_correct_l198_198845

noncomputable def proof_problem : Prop :=
  let bananas := 40
  let marbles := 4
  let boys := 18
  let girls := 12
  let total_friends := 30
  let bananas_for_boys := (3/8 : ℝ) * bananas
  let bananas_for_girls := (1/4 : ℝ) * bananas
  let bananas_left := bananas - (bananas_for_boys + bananas_for_girls)
  let bananas_per_marble := bananas_left / marbles
  bananas_for_boys = 15 ∧ bananas_for_girls = 10 ∧ bananas_per_marble = 3.75

theorem banana_distribution_correct : proof_problem :=
by
  -- Proof is omitted
  sorry

end banana_distribution_correct_l198_198845


namespace prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_l198_198284

-- Definition of the inequalities to be proven using the rearrangement inequality
def inequality1 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * a * b
def inequality2 (a b c : ℝ) : Prop := a^2 + b^2 + c^2 ≥ a * b + b * c + c * a
def inequality3 (a b : ℝ) : Prop := a^2 + b^2 + 1 ≥ a * b + b + a
def inequality5 (x y : ℝ) : Prop := x^3 + y^3 ≥ x^2 * y + x * y^2

-- Proofs required for each inequality
theorem prove_inequality1 (a b : ℝ) : inequality1 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality2 (a b c : ℝ) : inequality2 a b c := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality3 (a b : ℝ) : inequality3 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality5 (x y : ℝ) (hx : x ≥ y) (hy : 0 < y) : inequality5 x y := 
by sorry  -- This can be proved using the rearrangement inequality

end prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_l198_198284


namespace temperature_at_midnight_l198_198816

-- Define temperature in the morning
def T_morning := -2 -- in degrees Celsius

-- Temperature change at noon
def delta_noon := 12 -- in degrees Celsius

-- Temperature change at midnight
def delta_midnight := -8 -- in degrees Celsius

-- Function to compute temperature
def compute_temperature (T : ℤ) (delta1 : ℤ) (delta2 : ℤ) : ℤ :=
  T + delta1 + delta2

-- The proposition to prove
theorem temperature_at_midnight :
  compute_temperature T_morning delta_noon delta_midnight = 2 :=
by
  sorry

end temperature_at_midnight_l198_198816


namespace value_of_m_squared_plus_2m_minus_3_l198_198035

theorem value_of_m_squared_plus_2m_minus_3 (m : ℤ) : 
  (∀ x : ℤ, 4 * (x - 1) - m * x + 6 = 8 → x = 3) →
  m^2 + 2 * m - 3 = 5 :=
by
  sorry

end value_of_m_squared_plus_2m_minus_3_l198_198035


namespace inequality_proof_l198_198053

theorem inequality_proof (x y z : ℝ) (n : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h_sum : x + y + z = 1) :
  (x^4 / (y * (1 - y^n)) + y^4 / (z * (1 - z^n)) + z^4 / (x * (1 - x^n))) ≥ (3^n) / (3^(n+2) - 9) :=
by
  sorry

end inequality_proof_l198_198053


namespace number_of_unsold_items_l198_198028

theorem number_of_unsold_items (v k : ℕ) (hv : v ≤ 53) (havg_int : ∃ n : ℕ, k = n * v)
  (hk_eq : k = 130*v - 1595) 
  (hnew_avg : (k + 2505) / (v + 7) = 130) :
  60 - (v + 7) = 24 :=
by
  sorry

end number_of_unsold_items_l198_198028


namespace time_to_fill_pool_l198_198998

-- Define the conditions given in the problem
def pool_volume_gallons : ℕ := 30000
def num_hoses : ℕ := 5
def hose_flow_rate_gpm : ℕ := 3

-- Define the total flow rate per minute
def total_flow_rate_gpm : ℕ := num_hoses * hose_flow_rate_gpm

-- Define the total flow rate per hour
def total_flow_rate_gph : ℕ := total_flow_rate_gpm * 60

-- Prove that the time to fill the pool is equal to 34 hours
theorem time_to_fill_pool : pool_volume_gallons / total_flow_rate_gph = 34 :=
by {
  -- Insert detailed proof steps here.
  sorry
}

end time_to_fill_pool_l198_198998


namespace gcd_111_148_l198_198200

theorem gcd_111_148 : Nat.gcd 111 148 = 37 :=
by
  sorry

end gcd_111_148_l198_198200


namespace min_colors_needed_correct_l198_198669

-- Define the 5x5 grid as a type
def Grid : Type := Fin 5 × Fin 5

-- Define a coloring as a function from Grid to a given number of colors
def Coloring (colors : Type) : Type := Grid → colors

-- Define the property where in any row, column, or diagonal, no three consecutive cells have the same color
def valid_coloring (colors : Type) (C : Coloring colors) : Prop :=
  ∀ i : Fin 5, ∀ j : Fin 3, ( C (i, j) ≠ C (i, j + 1) ∧ C (i, j + 1) ≠ C (i, j + 2) ) ∧
  ∀ i : Fin 3, ∀ j : Fin 5, ( C (i, j) ≠ C (i + 1, j) ∧ C (i + 1, j) ≠ C (i + 2, j) ) ∧
  ∀ i : Fin 3, ∀ j : Fin 3, ( C (i, j) ≠ C (i + 1, j + 1) ∧ C (i + 1, j + 1) ≠ C (i + 2, j + 2) )

-- Define the minimum number of colors required
def min_colors_needed : Nat := 5

-- Prove the statement
theorem min_colors_needed_correct : ∃ C : Coloring (Fin min_colors_needed), valid_coloring (Fin min_colors_needed) C :=
sorry

end min_colors_needed_correct_l198_198669


namespace largest_five_digit_product_l198_198547

theorem largest_five_digit_product
  (digs : List ℕ)
  (h_digit_count : digs.length = 5)
  (h_product : (digs.foldr (· * ·) 1) = 9 * 8 * 7 * 6 * 5) :
  (digs.foldr (λ a b => if a > b then 10 * a + b else 10 * b + a) 0) = 98765 :=
sorry

end largest_five_digit_product_l198_198547


namespace distinct_banners_count_l198_198718

def colors : Finset String := 
  {"red", "white", "blue", "green", "yellow"}

def valid_banners (strip1 strip2 strip3 : String) : Prop :=
  strip1 ∈ colors ∧ strip2 ∈ colors ∧ strip3 ∈ colors ∧
  strip1 ≠ strip2 ∧ strip2 ≠ strip3 ∧ strip3 ≠ strip1

theorem distinct_banners_count : 
  ∃ (banners : Finset (String × String × String)), 
    (∀ s1 s2 s3, (s1, s2, s3) ∈ banners ↔ valid_banners s1 s2 s3) ∧
    banners.card = 60 :=
by
  sorry

end distinct_banners_count_l198_198718


namespace no_real_x_for_sqrt_l198_198993

theorem no_real_x_for_sqrt :
  ¬ ∃ x : ℝ, - (x^2 + 2 * x + 5) ≥ 0 :=
sorry

end no_real_x_for_sqrt_l198_198993


namespace mixture_alcohol_quantity_l198_198357

theorem mixture_alcohol_quantity:
  ∀ (A W : ℝ), 
    A / W = 4 / 3 ∧ A / (W + 7) = 4 / 5 → A = 14 :=
by
  intros A W h
  sorry

end mixture_alcohol_quantity_l198_198357


namespace decagon_area_l198_198827

theorem decagon_area 
    (perimeter_square : ℝ) 
    (side_division : ℕ) 
    (side_length : ℝ) 
    (triangle_area : ℝ) 
    (total_triangle_area : ℝ) 
    (square_area : ℝ)
    (decagon_area : ℝ) :
    perimeter_square = 150 →
    side_division = 5 →
    side_length = perimeter_square / 4 →
    triangle_area = 1 / 2 * (side_length / side_division) * (side_length / side_division) →
    total_triangle_area = 8 * triangle_area →
    square_area = side_length * side_length →
    decagon_area = square_area - total_triangle_area →
    decagon_area = 1181.25 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end decagon_area_l198_198827


namespace commute_time_abs_diff_l198_198308

theorem commute_time_abs_diff (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2) :
  |x - y| = 4 := by
  sorry

end commute_time_abs_diff_l198_198308


namespace company_fund_initial_amount_l198_198799

theorem company_fund_initial_amount (n : ℕ) 
  (h : 45 * n + 95 = 50 * n - 5) : 50 * n - 5 = 995 := by
  sorry

end company_fund_initial_amount_l198_198799


namespace probability_of_drawing_jingyuetan_ticket_l198_198757

-- Definitions from the problem
def num_jingyuetan_tickets : ℕ := 3
def num_changying_tickets : ℕ := 2
def total_tickets : ℕ := num_jingyuetan_tickets + num_changying_tickets
def num_envelopes : ℕ := total_tickets

-- Probability calculation
def probability_jingyuetan : ℚ := (num_jingyuetan_tickets : ℚ) / (num_envelopes : ℚ)

-- Theorem statement
theorem probability_of_drawing_jingyuetan_ticket : probability_jingyuetan = 3 / 5 :=
by
  sorry

end probability_of_drawing_jingyuetan_ticket_l198_198757


namespace minimum_sum_at_nine_l198_198913

noncomputable def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem minimum_sum_at_nine {a1 d : ℤ} (h_a1_neg : a1 < 0) 
    (h_sum_equal : sum_of_arithmetic_sequence a1 d 12 = sum_of_arithmetic_sequence a1 d 6) :
  ∀ n : ℕ, (n = 9) → sum_of_arithmetic_sequence a1 d n ≤ sum_of_arithmetic_sequence a1 d m :=
sorry

end minimum_sum_at_nine_l198_198913


namespace math_marks_is_95_l198_198233

-- Define the conditions as Lean assumptions
variables (english_marks math_marks physics_marks chemistry_marks biology_marks : ℝ)
variable (average_marks : ℝ)
variable (num_subjects : ℝ)

-- State the conditions
axiom h1 : english_marks = 96
axiom h2 : physics_marks = 82
axiom h3 : chemistry_marks = 97
axiom h4 : biology_marks = 95
axiom h5 : average_marks = 93
axiom h6 : num_subjects = 5

-- Formalize the problem: Prove that math_marks = 95
theorem math_marks_is_95 : math_marks = 95 :=
by
  sorry

end math_marks_is_95_l198_198233


namespace inequality_m_2n_l198_198598

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

lemma max_f : ∃ x : ℝ, f x = 2 :=
sorry

theorem inequality_m_2n (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 1/(2*n) = 2) : m + 2*n ≥ 2 :=
sorry

end inequality_m_2n_l198_198598


namespace number_of_positive_real_solutions_l198_198407

noncomputable def p (x : ℝ) : ℝ := x^12 + 5 * x^11 + 20 * x^10 + 1300 * x^9 - 1105 * x^8

theorem number_of_positive_real_solutions : ∃! x : ℝ, 0 < x ∧ p x = 0 :=
sorry

end number_of_positive_real_solutions_l198_198407


namespace inequality_lt_l198_198510

theorem inequality_lt (x y : ℝ) (h1 : x > y) (h2 : y > 0) (n k : ℕ) (h3 : n > k) :
  (x^k - y^k) ^ n < (x^n - y^n) ^ k := 
  sorry

end inequality_lt_l198_198510


namespace prank_combinations_l198_198093

theorem prank_combinations :
  let monday := 1
  let tuesday := 4
  let wednesday := 7
  let thursday := 5
  let friday := 1
  (monday * tuesday * wednesday * thursday * friday) = 140 :=
by
  sorry

end prank_combinations_l198_198093


namespace eq_solution_l198_198805

theorem eq_solution (x : ℝ) (h : 2 / x = 3 / (x + 1)) : x = 2 :=
by
  sorry

end eq_solution_l198_198805


namespace correct_calculation_l198_198124

theorem correct_calculation (a b : ℝ) :
  (6 * a - 5 * a ≠ 1) ∧
  (a + 2 * a^2 ≠ 3 * a^3) ∧
  (- (a - b) = -a + b) ∧
  (2 * (a + b) ≠ 2 * a + b) :=
by 
  sorry

end correct_calculation_l198_198124


namespace map_length_representation_l198_198330

variable (x : ℕ)

theorem map_length_representation :
  (12 : ℕ) * x = 17 * (72 : ℕ) / 12
:=
sorry

end map_length_representation_l198_198330


namespace first_player_winning_strategy_l198_198954
noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem first_player_winning_strategy (x1 y1 : ℕ)
    (h1 : x1 > 0) (h2 : y1 > 0) :
    (x1 / y1 = 1) ∨ 
    (x1 / y1 > golden_ratio) ∨ 
    (x1 / y1 < 1 / golden_ratio) :=
sorry

end first_player_winning_strategy_l198_198954


namespace cost_price_equals_selling_price_l198_198422

theorem cost_price_equals_selling_price (C : ℝ) (x : ℝ) (hp : C > 0) (profit : ℝ := 0.25) (h : 30 * C = (1 + profit) * C * x) : x = 24 :=
by
  sorry

end cost_price_equals_selling_price_l198_198422


namespace similar_segments_areas_proportional_to_chords_squares_l198_198759

variables {k k₁ Δ Δ₁ r r₁ a a₁ S S₁ : ℝ}

-- Conditions given in the problem
def similar_segments (r r₁ a a₁ Δ Δ₁ k k₁ : ℝ) :=
  (Δ / Δ₁ = (a^2 / a₁^2) ∧ (Δ / Δ₁ = r^2 / r₁^2)) ∧ (k / k₁ = r^2 / r₁^2)

-- Given the areas of the segments in terms of sectors and triangles
def area_of_segment (k Δ : ℝ) := k - Δ

-- Theorem statement proving the desired relationship
theorem similar_segments_areas_proportional_to_chords_squares
  (h : similar_segments r r₁ a a₁ Δ Δ₁ k k₁) :
  (S = area_of_segment k Δ) → (S₁ = area_of_segment k₁ Δ₁) → (S / S₁ = a^2 / a₁^2) :=
by
  sorry

end similar_segments_areas_proportional_to_chords_squares_l198_198759


namespace value_of_expression_l198_198543

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) :
    11 / 7 + (2 * q - p) / (2 * q + p) = 2 :=
sorry

end value_of_expression_l198_198543


namespace ab_value_l198_198491

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end ab_value_l198_198491


namespace batsman_average_increase_l198_198159

theorem batsman_average_increase 
  (A : ℕ)
  (h1 : ∀ n ≤ 11, (1 / (n : ℝ)) * (A * n + 60) = 38) 
  (h2 : 1 / 12 * (A * 11 + 60) = 38)
  (h3 : ∀ n ≤ 12, (A * n : ℝ) ≤ (A * (n + 1) : ℝ)) :
  38 - A = 2 := 
sorry

end batsman_average_increase_l198_198159


namespace rhombus_other_diagonal_length_l198_198648

theorem rhombus_other_diagonal_length (area_square : ℝ) (side_length_square : ℝ) (d1_rhombus : ℝ) (d2_expected: ℝ) 
  (h1 : area_square = side_length_square^2) 
  (h2 : side_length_square = 8) 
  (h3 : d1_rhombus = 16) 
  (h4 : (d1_rhombus * d2_expected) / 2 = area_square) :
  d2_expected = 8 := 
by
  sorry

end rhombus_other_diagonal_length_l198_198648


namespace quadratic_inequality_solution_l198_198582

theorem quadratic_inequality_solution :
  {x : ℝ | 3 * x^2 + 5 * x < 8} = {x : ℝ | -4 < x ∧ x < 2 / 3} :=
sorry

end quadratic_inequality_solution_l198_198582


namespace modulus_of_z_l198_198499

open Complex

theorem modulus_of_z (z : ℂ) (hz : (1 + I) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l198_198499


namespace shorter_piece_length_l198_198470

-- Definitions for the conditions
def total_length : ℕ := 70
def ratio (short long : ℕ) : Prop := long = (5 * short) / 2

-- The proof problem statement
theorem shorter_piece_length (x : ℕ) (h1 : total_length = x + (5 * x) / 2) : x = 20 :=
sorry

end shorter_piece_length_l198_198470


namespace dogs_legs_l198_198984

theorem dogs_legs (num_dogs : ℕ) (legs_per_dog : ℕ) (h1 : num_dogs = 109) (h2 : legs_per_dog = 4) : num_dogs * legs_per_dog = 436 :=
by {
  -- The proof is omitted as it's indicated that it should contain "sorry"
  sorry
}

end dogs_legs_l198_198984


namespace count_four_digit_integers_l198_198935

theorem count_four_digit_integers :
    ∃! (a b c d : ℕ), 1 ≤ a ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    (10 * b + c)^2 = (10 * a + b) * (10 * c + d) := sorry

end count_four_digit_integers_l198_198935


namespace emerson_distance_l198_198398

theorem emerson_distance (d1 : ℕ) : 
  (d1 + 15 + 18 = 39) → d1 = 6 := 
by
  intro h
  have h1 : 33 = 39 - d1 := sorry -- Steps to manipulate equation to find d1
  sorry

end emerson_distance_l198_198398


namespace sum_of_two_integers_is_22_l198_198474

noncomputable def a_and_b_sum_to_S : Prop :=
  ∃ (a b S : ℕ), 
    a + b = S ∧ 
    a^2 - b^2 = 44 ∧ 
    a * b = 120 ∧ 
    S = 22

theorem sum_of_two_integers_is_22 : a_and_b_sum_to_S :=
by {
  sorry
}

end sum_of_two_integers_is_22_l198_198474


namespace geometric_sequence_common_ratio_l198_198117

theorem geometric_sequence_common_ratio (S : ℕ → ℝ) (a : ℕ → ℝ)
  (q : ℝ) (h1 : a 1 = 2) (h2 : S 3 = 6)
  (geo_sum : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  q = 1 ∨ q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l198_198117


namespace area_of_triangle_ADE_l198_198697

theorem area_of_triangle_ADE (A B C D E : Type) (AB BC AC : ℝ) (AD AE : ℝ)
  (h1 : AB = 8) (h2 : BC = 13) (h3 : AC = 15) (h4 : AD = 3) (h5 : AE = 11) :
  let s := (AB + BC + AC) / 2
  let area_ABC := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  let sinA := 2 * area_ABC / (AB * AC)
  let area_ADE := (1 / 2) * AD * AE * sinA
  area_ADE = (33 * Real.sqrt 3) / 4 :=
by 
  have s := (8 + 13 + 15) / 2
  have area_ABC := Real.sqrt (s * (s - 8) * (s - 13) * (s - 15))
  have sinA := 2 * area_ABC / (8 * 15)
  have area_ADE := (1 / 2) * 3 * 11 * sinA
  sorry

end area_of_triangle_ADE_l198_198697


namespace necessary_but_not_sufficient_l198_198708

variable (p q : Prop)
-- Condition p: The base of a right prism is a rhombus.
def base_of_right_prism_is_rhombus := p
-- Condition q: A prism is a right rectangular prism.
def prism_is_right_rectangular := q

-- Proof: p is a necessary but not sufficient condition for q.
theorem necessary_but_not_sufficient (p q : Prop) 
  (h1 : base_of_right_prism_is_rhombus p)
  (h2 : prism_is_right_rectangular q) : 
  (q → p) ∧ ¬ (p → q) :=
sorry

end necessary_but_not_sufficient_l198_198708


namespace Nancy_money_in_dollars_l198_198147

-- Condition: Nancy has saved 1 dozen quarters
def dozen : ℕ := 12

-- Condition: Each quarter is worth 25 cents
def value_of_quarter : ℕ := 25

-- Condition: 100 cents is equal to 1 dollar
def cents_per_dollar : ℕ := 100

-- Proving that Nancy has 3 dollars
theorem Nancy_money_in_dollars :
  (dozen * value_of_quarter) / cents_per_dollar = 3 := by
  sorry

end Nancy_money_in_dollars_l198_198147


namespace simplify_expression_l198_198157

theorem simplify_expression (x : ℝ) : 
  8 * x + 15 - 3 * x + 5 * 7 = 5 * x + 50 :=
by
  sorry

end simplify_expression_l198_198157


namespace container_holds_slices_l198_198141

theorem container_holds_slices (x : ℕ) 
  (h1 : x > 1) 
  (h2 : x ≠ 332) 
  (h3 : x ≠ 166) 
  (h4 : x ∣ 332) :
  x = 83 := 
sorry

end container_holds_slices_l198_198141


namespace reggie_father_money_l198_198488

theorem reggie_father_money :
  let books := 5
  let cost_per_book := 2
  let amount_left := 38
  books * cost_per_book + amount_left = 48 :=
by
  sorry

end reggie_father_money_l198_198488


namespace number_of_restaurants_l198_198307

def units_in_building : ℕ := 300
def residential_units := units_in_building / 2
def remaining_units := units_in_building - residential_units
def restaurants := remaining_units / 2

theorem number_of_restaurants : restaurants = 75 :=
by
  sorry

end number_of_restaurants_l198_198307


namespace distribute_paper_clips_l198_198239

theorem distribute_paper_clips (total_clips : ℕ) (boxes : ℕ) (clips_per_box : ℕ) 
  (h1 : total_clips = 81) (h2 : boxes = 9) :
  total_clips / boxes = clips_per_box ↔ clips_per_box = 9 :=
by
  sorry

end distribute_paper_clips_l198_198239


namespace total_education_duration_l198_198021

-- Definitions from the conditions
def high_school_duration : ℕ := 4 - 1
def tertiary_education_duration : ℕ := 3 * high_school_duration

-- The theorem statement
theorem total_education_duration : high_school_duration + tertiary_education_duration = 12 :=
by
  sorry

end total_education_duration_l198_198021


namespace money_left_after_shopping_l198_198710

def initial_amount : ℕ := 26
def cost_jumper : ℕ := 9
def cost_tshirt : ℕ := 4
def cost_heels : ℕ := 5

theorem money_left_after_shopping : initial_amount - (cost_jumper + cost_tshirt + cost_heels) = 8 :=
by
  sorry

end money_left_after_shopping_l198_198710


namespace no_x4_term_expansion_l198_198318

-- Mathematical condition and properties
variable {R : Type*} [CommRing R]

theorem no_x4_term_expansion (a : R) (h : a ≠ 0) :
  ∃ a, (a = 8) := 
by 
  sorry

end no_x4_term_expansion_l198_198318


namespace line_equation_through_point_with_intercepts_conditions_l198_198811

theorem line_equation_through_point_with_intercepts_conditions :
  ∃ (a b : ℚ) (m c : ℚ), 
    (-5) * m + c = 2 ∧ -- The line passes through A(-5, 2)
    a = 2 * b ∧       -- x-intercept is twice the y-intercept
    (a * m + c = 0 ∨ ((1/m)*a + (1/m)^2 * c+1 = 0)) :=         -- Equations of the line
sorry

end line_equation_through_point_with_intercepts_conditions_l198_198811


namespace who_drank_most_l198_198502

theorem who_drank_most (eunji yujeong yuna : ℝ) 
    (h1 : eunji = 0.5) 
    (h2 : yujeong = 7 / 10) 
    (h3 : yuna = 6 / 10) :
    max (max eunji yujeong) yuna = yujeong :=
by {
    sorry
}

end who_drank_most_l198_198502


namespace intersecting_lines_l198_198874

theorem intersecting_lines (a b c d : ℝ) (h₁ : a ≠ b) (h₂ : ∃ x y : ℝ, y = a*x + a ∧ y = b*x + b ∧ y = c*x + d) : c = d :=
sorry

end intersecting_lines_l198_198874


namespace sphere_volume_equals_surface_area_l198_198410

theorem sphere_volume_equals_surface_area (r : ℝ) (hr : r = 3) :
  (4 / 3) * π * r^3 = 4 * π * r^2 := by
  sorry

end sphere_volume_equals_surface_area_l198_198410


namespace evan_books_two_years_ago_l198_198526

theorem evan_books_two_years_ago (B B2 : ℕ) 
  (h1 : 860 = 5 * B + 60) 
  (h2 : B2 = B + 40) : 
  B2 = 200 := 
by 
  sorry

end evan_books_two_years_ago_l198_198526


namespace size_of_third_file_l198_198045

theorem size_of_third_file 
  (s : ℝ) (t : ℝ) (f1 : ℝ) (f2 : ℝ) (f3 : ℝ) 
  (h1 : s = 2) (h2 : t = 120) (h3 : f1 = 80) (h4 : f2 = 90) : 
  f3 = s * t - (f1 + f2) :=
by
  sorry

end size_of_third_file_l198_198045


namespace question_equals_answer_l198_198851

theorem question_equals_answer (x y : ℝ) (h : abs (x - 6) + (y + 4)^2 = 0) : x + y = 2 :=
sorry

end question_equals_answer_l198_198851


namespace length_of_one_side_of_regular_pentagon_l198_198027

-- Define the conditions
def is_regular_pentagon (P : ℝ) (n : ℕ) : Prop := n = 5 ∧ P = 23.4

-- State the theorem
theorem length_of_one_side_of_regular_pentagon (P : ℝ) (n : ℕ) 
  (h : is_regular_pentagon P n) : P / n = 4.68 :=
by
  sorry

end length_of_one_side_of_regular_pentagon_l198_198027


namespace cos_double_angle_of_tan_l198_198417

theorem cos_double_angle_of_tan (θ : ℝ) (h : Real.tan θ = -1 / 3) : Real.cos (2 * θ) = 4 / 5 :=
sorry

end cos_double_angle_of_tan_l198_198417


namespace cream_ratio_l198_198042

theorem cream_ratio (joe_initial_coffee joann_initial_coffee : ℝ)
                    (joe_drank_ounces joann_drank_ounces joe_added_cream joann_added_cream : ℝ) :
  joe_initial_coffee = 20 →
  joann_initial_coffee = 20 →
  joe_drank_ounces = 3 →
  joann_drank_ounces = 3 →
  joe_added_cream = 4 →
  joann_added_cream = 4 →
  (4 : ℝ) / ((21 / 24) * 24 - 3) = (8 : ℝ) / 7 :=
by
  intros h_ji h_ji h_jd h_jd h_jc h_jc
  sorry

end cream_ratio_l198_198042


namespace find_missing_number_l198_198351

theorem find_missing_number (x : ℕ) : 
  (1 + 22 + 23 + 24 + 25 + 26 + x + 2) / 8 = 20 → x = 37 := by
  sorry

end find_missing_number_l198_198351


namespace closest_time_to_1600_mirror_l198_198570

noncomputable def clock_in_mirror_time (hour_hand_minute: ℕ) (minute_hand_minute: ℕ) : (ℕ × ℕ) :=
  let hour_in_mirror := (12 - hour_hand_minute) % 12
  let minute_in_mirror := minute_hand_minute
  (hour_in_mirror, minute_in_mirror)

theorem closest_time_to_1600_mirror (A B C D : (ℕ × ℕ)) :
  clock_in_mirror_time 4 0 = D → D = (8, 0) :=
by
  -- Introduction of hypothesis that clock closest to 16:00 (4:00) is represented by D
  intro h
  -- State the conclusion based on the given hypothesis
  sorry

end closest_time_to_1600_mirror_l198_198570


namespace product_pos_implies_pos_or_neg_pos_pair_implies_product_pos_product_pos_necessary_for_pos_product_pos_not_sufficient_for_pos_l198_198740

variable {x y : ℝ}

-- The formal statement in Lean
theorem product_pos_implies_pos_or_neg (h : x * y > 0) : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
sorry

theorem pos_pair_implies_product_pos (hx : x > 0) (hy : y > 0) : x * y > 0 :=
sorry

theorem product_pos_necessary_for_pos (h : x > 0 ∧ y > 0) : x * y > 0 :=
pos_pair_implies_product_pos h.1 h.2

theorem product_pos_not_sufficient_for_pos (h : x * y > 0) : ¬ (x > 0 ∧ y > 0) :=
sorry

end product_pos_implies_pos_or_neg_pos_pair_implies_product_pos_product_pos_necessary_for_pos_product_pos_not_sufficient_for_pos_l198_198740


namespace net_percentage_change_l198_198344

-- Definitions based on given conditions
variables (P : ℝ) (P_post_decrease : ℝ) (P_post_increase : ℝ)

-- Conditions
def decreased_by_5_percent : Prop := P_post_decrease = P * (1 - 0.05)
def increased_by_10_percent : Prop := P_post_increase = P_post_decrease * (1 + 0.10)

-- Proof problem
theorem net_percentage_change (h1 : decreased_by_5_percent P P_post_decrease) (h2 : increased_by_10_percent P_post_decrease P_post_increase) : 
  ((P_post_increase - P) / P) * 100 = 4.5 :=
by
  -- The proof would go here
  sorry

end net_percentage_change_l198_198344


namespace distance_amanda_to_kimberly_l198_198409

-- Define the given conditions
def amanda_speed : ℝ := 2 -- miles per hour
def amanda_time : ℝ := 3 -- hours

-- Prove that the distance is 6 miles
theorem distance_amanda_to_kimberly : amanda_speed * amanda_time = 6 := by
  sorry

end distance_amanda_to_kimberly_l198_198409


namespace cmp_c_b_a_l198_198500

noncomputable def a : ℝ := 17 / 18
noncomputable def b : ℝ := Real.cos (1 / 3)
noncomputable def c : ℝ := 3 * Real.sin (1 / 3)

theorem cmp_c_b_a:
  c > b ∧ b > a := by
  sorry

end cmp_c_b_a_l198_198500


namespace point_B_between_A_and_C_l198_198282

theorem point_B_between_A_and_C (a b c : ℚ) (h_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : |a - b| + |b - c| = |a - c|) : 
  (a < b ∧ b < c) ∨ (c < b ∧ b < a) :=
sorry

end point_B_between_A_and_C_l198_198282


namespace number_of_boys_l198_198626

-- Definitions of the conditions
def total_students : ℕ := 30
def ratio_girls_parts : ℕ := 1
def ratio_boys_parts : ℕ := 2
def total_parts : ℕ := ratio_girls_parts + ratio_boys_parts

-- Statement of the problem
theorem number_of_boys :
  ∃ (boys : ℕ), boys = (total_students / total_parts) * ratio_boys_parts ∧ boys = 20 :=
by
  sorry

end number_of_boys_l198_198626


namespace polikarp_make_first_box_empty_l198_198600

theorem polikarp_make_first_box_empty (n : ℕ) (h : n ≤ 30) : ∃ (x y : ℕ), x + y ≤ 10 ∧ ∀ k : ℕ, k ≤ x → k + k * y = n :=
by
  sorry

end polikarp_make_first_box_empty_l198_198600


namespace marbles_each_friend_gets_l198_198820

-- Definitions for the conditions
def total_marbles : ℕ := 100
def marbles_kept : ℕ := 20
def number_of_friends : ℕ := 5

-- The math proof problem
theorem marbles_each_friend_gets :
  (total_marbles - marbles_kept) / number_of_friends = 16 :=
by
  -- We include the proof steps within a by notation but stop at sorry for automated completion skipping proof steps
  sorry

end marbles_each_friend_gets_l198_198820


namespace xiaoming_accuracy_l198_198448

theorem xiaoming_accuracy :
  ∀ (correct already_wrong extra_needed : ℕ),
  correct = 30 →
  already_wrong = 6 →
  (correct + extra_needed).toFloat / (correct + already_wrong + extra_needed).toFloat = 0.85 →
  extra_needed = 4 := by
  intros correct already_wrong extra_needed h_correct h_wrong h_accuracy
  sorry

end xiaoming_accuracy_l198_198448


namespace analogous_to_tetrahedron_is_triangle_l198_198830

-- Define the objects as types
inductive Object
| Quadrilateral
| Pyramid
| Triangle
| Prism
| Tetrahedron

-- Define the analogous relationship
def analogous (a b : Object) : Prop :=
  (a = Object.Tetrahedron ∧ b = Object.Triangle)
  ∨ (b = Object.Tetrahedron ∧ a = Object.Triangle)

-- The main statement to prove
theorem analogous_to_tetrahedron_is_triangle :
  ∃ (x : Object), analogous Object.Tetrahedron x ∧ x = Object.Triangle :=
by
  sorry

end analogous_to_tetrahedron_is_triangle_l198_198830


namespace grandmother_age_l198_198869

theorem grandmother_age 
  (avg_age : ℝ)
  (age1 age2 age3 grandma_age : ℝ)
  (h_avg_age : avg_age = 20)
  (h_ages : age1 = 5)
  (h_ages2 : age2 = 10)
  (h_ages3 : age3 = 13)
  (h_eq : (age1 + age2 + age3 + grandma_age) / 4 = avg_age) : 
  grandma_age = 52 := 
by
  sorry

end grandmother_age_l198_198869


namespace weavers_in_first_group_l198_198781

theorem weavers_in_first_group 
  (W : ℕ)
  (H1 : 4 / (W * 4) = 1 / W) 
  (H2 : (9 / 6) / 6 = 0.25) :
  W = 4 :=
sorry

end weavers_in_first_group_l198_198781


namespace train_passing_time_l198_198346

theorem train_passing_time (length_of_train : ℝ) (speed_of_train_kmhr : ℝ) :
  length_of_train = 180 → speed_of_train_kmhr = 36 → (length_of_train / (speed_of_train_kmhr * (1000 / 3600))) = 18 :=
by
  intro h1 h2
  sorry

end train_passing_time_l198_198346


namespace intersection_points_l198_198480

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
noncomputable def parabola2 (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem intersection_points :
  {p : ℝ × ℝ |
    (∃ x : ℝ, p = (x, parabola1 x) ∧ parabola1 x = parabola2 x)} =
  { 
    ( (3 + Real.sqrt 13) / 4, (74 + 14 * Real.sqrt 13) / 16 ),
    ( (3 - Real.sqrt 13) / 4, (74 - 14 * Real.sqrt 13) / 16 )
  } := sorry

end intersection_points_l198_198480


namespace total_ttaki_count_l198_198763

noncomputable def total_ttaki_used (n : ℕ): ℕ := n * n

theorem total_ttaki_count {n : ℕ} (h : 4 * n - 4 = 240) : total_ttaki_used n = 3721 := by
  sorry

end total_ttaki_count_l198_198763


namespace aspirin_mass_percentages_l198_198872

noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def molar_mass_aspirin : ℝ := (9 * atomic_mass_C) + (8 * atomic_mass_H) + (4 * atomic_mass_O)

theorem aspirin_mass_percentages :
  let mass_percent_H := ((8 * atomic_mass_H) / molar_mass_aspirin) * 100
  let mass_percent_C := ((9 * atomic_mass_C) / molar_mass_aspirin) * 100
  let mass_percent_O := ((4 * atomic_mass_O) / molar_mass_aspirin) * 100
  mass_percent_H = 4.48 ∧ mass_percent_C = 60.00 ∧ mass_percent_O = 35.52 :=
by
  -- Placeholder for the proof
  sorry

end aspirin_mass_percentages_l198_198872


namespace Arman_hours_worked_l198_198956

/--
  Given:
  - LastWeekHours = 35
  - LastWeekRate = 10 (in dollars per hour)
  - IncreaseRate = 0.5 (in dollars per hour)
  - TotalEarnings = 770 (in dollars)
  Prove that:
  - ThisWeekHours = 40
-/
theorem Arman_hours_worked (LastWeekHours : ℕ) (LastWeekRate : ℕ) (IncreaseRate : ℕ) (TotalEarnings : ℕ)
  (h1 : LastWeekHours = 35)
  (h2 : LastWeekRate = 10)
  (h3 : IncreaseRate = 1/2)  -- because 0.5 as a fraction is 1/2
  (h4 : TotalEarnings = 770)
  : ∃ ThisWeekHours : ℕ, ThisWeekHours = 40 :=
by
  sorry

end Arman_hours_worked_l198_198956


namespace triangle_side_relation_l198_198026

variable {α β γ : ℝ} -- angles in the triangle
variable {a b c : ℝ} -- sides opposite to the angles

theorem triangle_side_relation
  (h1 : α = 3 * β)
  (h2 : α = 6 * γ)
  (h_sum : α + β + γ = 180)
  : b * c^2 = (a + b) * (a - b)^2 := 
by
  sorry

end triangle_side_relation_l198_198026


namespace price_increase_for_desired_profit_l198_198735

/--
In Xianyou Yonghui Supermarket, the profit from selling Pomelos is 10 yuan per kilogram.
They can sell 500 kilograms per day. Market research has found that, with a constant cost price, if the price per kilogram increases by 1 yuan, the daily sales volume will decrease by 20 kilograms.
Now, the supermarket wants to ensure a daily profit of 6000 yuan while also offering the best deal to the customers.
-/
theorem price_increase_for_desired_profit :
  ∃ x : ℝ, (10 + x) * (500 - 20 * x) = 6000 ∧ x = 5 :=
sorry

end price_increase_for_desired_profit_l198_198735


namespace grade_assignment_ways_l198_198329

theorem grade_assignment_ways : (4^12 = 16777216) := 
by 
  sorry

end grade_assignment_ways_l198_198329


namespace max_loaves_given_l198_198196

variables {a1 d : ℕ}

-- Mathematical statement: The conditions given in the problem
def arith_sequence_correct (a1 d : ℕ) : Prop :=
  (5 * a1 + 10 * d = 60) ∧ (2 * a1 + 7 * d = 3 * a1 + 3 * d)

-- Lean theorem statement
theorem max_loaves_given (a1 d : ℕ) (h : arith_sequence_correct a1 d) : a1 + 4 * d = 16 :=
sorry

end max_loaves_given_l198_198196


namespace sum_of_squares_and_product_pos_ints_l198_198108

variable (x y : ℕ)

theorem sum_of_squares_and_product_pos_ints :
  x^2 + y^2 = 289 ∧ x * y = 120 → x + y = 23 :=
by
  intro h
  sorry

end sum_of_squares_and_product_pos_ints_l198_198108


namespace equilateral_triangle_side_length_l198_198863

theorem equilateral_triangle_side_length (total_length : ℕ) (h1 : total_length = 78) : (total_length / 3) = 26 :=
by
  sorry

end equilateral_triangle_side_length_l198_198863


namespace total_students_l198_198385

-- Definition of the problem conditions
def ratio_boys_girls : ℕ := 8
def ratio_girls : ℕ := 5
def number_girls : ℕ := 160

-- The main theorem statement
theorem total_students (b g : ℕ) (h1 : b * ratio_girls = g * ratio_boys_girls) (h2 : g = number_girls) :
  b + g = 416 :=
sorry

end total_students_l198_198385


namespace farmer_payment_per_acre_l198_198109

-- Define the conditions
def monthly_payment : ℝ := 300
def length_ft : ℝ := 360
def width_ft : ℝ := 1210
def sqft_per_acre : ℝ := 43560

-- Define the question and its correct answer
def payment_per_acre_per_month : ℝ := 30

-- Prove that the farmer pays $30 per acre per month
theorem farmer_payment_per_acre :
  (monthly_payment / ((length_ft * width_ft) / sqft_per_acre)) = payment_per_acre_per_month :=
by
  sorry

end farmer_payment_per_acre_l198_198109


namespace find_x_value_l198_198493

def my_operation (a b : ℝ) : ℝ := 2 * a * b + 3 * b - 2 * a

theorem find_x_value (x : ℝ) (h : my_operation 3 x = 60) : x = 7.33 := 
by 
  sorry

end find_x_value_l198_198493


namespace modulo_residue_addition_l198_198154

theorem modulo_residue_addition : 
  (368 + 3 * 78 + 8 * 242 + 6 * 22) % 11 = 8 := 
by
  have h1 : 368 % 11 = 5 := by sorry
  have h2 : 78 % 11 = 1 := by sorry
  have h3 : 242 % 11 = 0 := by sorry
  have h4 : 22 % 11 = 0 := by sorry
  sorry

end modulo_residue_addition_l198_198154


namespace total_fruits_l198_198714

theorem total_fruits (cucumbers : ℕ) (watermelons : ℕ) 
  (h1 : cucumbers = 18) 
  (h2 : watermelons = cucumbers + 8) : 
  cucumbers + watermelons = 44 := 
by {
  sorry
}

end total_fruits_l198_198714


namespace count_whole_numbers_in_interval_l198_198114

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l198_198114


namespace green_ball_removal_l198_198535

variable (total_balls : ℕ)
variable (initial_green_balls : ℕ)
variable (initial_yellow_balls : ℕ)
variable (desired_green_percentage : ℚ)
variable (removals : ℕ)

theorem green_ball_removal :
  initial_green_balls = 420 → 
  total_balls = 600 → 
  desired_green_percentage = 3 / 5 →
  (420 - removals) / (600 - removals) = desired_green_percentage → 
  removals = 150 :=
sorry

end green_ball_removal_l198_198535


namespace max_earnings_l198_198005

section MaryEarnings

def regular_rate : ℝ := 10
def first_period_hours : ℕ := 40
def second_period_hours : ℕ := 10
def third_period_hours : ℕ := 10
def weekend_days : ℕ := 2
def weekend_bonus_per_day : ℝ := 50
def bonus_threshold_hours : ℕ := 55
def overtime_multiplier_second_period : ℝ := 0.25
def overtime_multiplier_third_period : ℝ := 0.5
def milestone_bonus : ℝ := 100

def regular_pay := regular_rate * first_period_hours
def second_period_pay := (regular_rate * (1 + overtime_multiplier_second_period)) * second_period_hours
def third_period_pay := (regular_rate * (1 + overtime_multiplier_third_period)) * third_period_hours
def weekend_bonus := weekend_days * weekend_bonus_per_day
def milestone_pay := milestone_bonus

def total_earnings := regular_pay + second_period_pay + third_period_pay + weekend_bonus + milestone_pay

theorem max_earnings : total_earnings = 875 := by
  sorry

end MaryEarnings

end max_earnings_l198_198005


namespace profit_calculation_l198_198861

variable (price : ℕ) (cost : ℕ) (exchange_rate : ℕ) (profit_per_bottle : ℚ)

-- Conditions
def conditions := price = 2 ∧ cost = 1 ∧ exchange_rate = 5

-- Profit per bottle is 0.66 yuan considering the exchange policy
theorem profit_calculation (h : conditions price cost exchange_rate) : profit_per_bottle = 0.66 := sorry

end profit_calculation_l198_198861


namespace find_number_l198_198670

theorem find_number (x : ℕ) (h : x * 625 = 584638125) : x = 935420 :=
sorry

end find_number_l198_198670


namespace boy_usual_time_l198_198481

noncomputable def usual_rate (R : ℝ) := R
noncomputable def usual_time (T : ℝ) := T
noncomputable def faster_rate (R : ℝ) := (7 / 6) * R
noncomputable def faster_time (T : ℝ) := T - 5

theorem boy_usual_time
  (R : ℝ) (T : ℝ) 
  (h1 : usual_rate R * usual_time T = faster_rate R * faster_time T) :
  T = 35 :=
by 
  unfold usual_rate usual_time faster_rate faster_time at h1
  sorry

end boy_usual_time_l198_198481


namespace maggi_ate_5_cupcakes_l198_198184

theorem maggi_ate_5_cupcakes
  (packages : ℕ)
  (cupcakes_per_package : ℕ)
  (left_cupcakes : ℕ)
  (total_cupcakes : ℕ := packages * cupcakes_per_package)
  (eaten_cupcakes : ℕ := total_cupcakes - left_cupcakes)
  (h1 : packages = 3)
  (h2 : cupcakes_per_package = 4)
  (h3 : left_cupcakes = 7) :
  eaten_cupcakes = 5 :=
by
  sorry

end maggi_ate_5_cupcakes_l198_198184


namespace probability_red_ball_is_correct_l198_198772

noncomputable def probability_red_ball : ℚ :=
  let prob_A := 1 / 3
  let prob_B := 1 / 3
  let prob_C := 1 / 3
  let prob_red_A := 3 / 10
  let prob_red_B := 7 / 10
  let prob_red_C := 5 / 11
  (prob_A * prob_red_A) + (prob_B * prob_red_B) + (prob_C * prob_red_C)

theorem probability_red_ball_is_correct : probability_red_ball = 16 / 33 := 
by
  sorry

end probability_red_ball_is_correct_l198_198772


namespace tangent_line_eqn_l198_198131

theorem tangent_line_eqn 
  (x y : ℝ)
  (H_curve : y = x^3 + 3 * x^2 - 5)
  (H_point : (x, y) = (-1, -3)) :
  (3 * x + y + 6 = 0) := 
sorry

end tangent_line_eqn_l198_198131


namespace number_of_four_digit_numbers_with_two_identical_digits_l198_198467

-- Define the conditions
def starts_with_nine (n : ℕ) : Prop := n / 1000 = 9
def has_exactly_two_identical_digits (n : ℕ) : Prop := 
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d1 ∧ (n % 100) / 10 = d1 ∧ n % 10 = d2) ∨
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d2 ∧ (n % 100) / 10 = d1 ∧ n % 10 = d1) ∨
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d1 ∧ (n % 100) / 10 = d2 ∧ n % 10 = d1)

-- Define the proof problem
theorem number_of_four_digit_numbers_with_two_identical_digits : 
  ∃ n, starts_with_nine n ∧ has_exactly_two_identical_digits n ∧ n = 432 := 
sorry

end number_of_four_digit_numbers_with_two_identical_digits_l198_198467


namespace anniversary_sale_total_cost_l198_198727

-- Definitions of conditions
def original_price_ice_cream : ℕ := 12
def discount_ice_cream : ℕ := 2
def sale_price_ice_cream : ℕ := original_price_ice_cream - discount_ice_cream

def price_per_five_cans_juice : ℕ := 2
def cans_per_five_pack : ℕ := 5

-- Definition of total cost
def total_cost : ℕ := 2 * sale_price_ice_cream + (10 / cans_per_five_pack) * price_per_five_cans_juice

-- The goal is to prove that total_cost is 24
theorem anniversary_sale_total_cost : total_cost = 24 :=
by
  sorry

end anniversary_sale_total_cost_l198_198727


namespace smallest_n_divides_24_and_1024_l198_198269

theorem smallest_n_divides_24_and_1024 : ∃ n : ℕ, n > 0 ∧ (24 ∣ n^2) ∧ (1024 ∣ n^3) ∧ (∀ m : ℕ, (m > 0 ∧ (24 ∣ m^2) ∧ (1024 ∣ m^3)) → n ≤ m) :=
by
  sorry

end smallest_n_divides_24_and_1024_l198_198269


namespace susan_strawberries_per_handful_l198_198181

-- Definitions of the given conditions
def total_picked := 75
def total_needed := 60
def strawberries_per_handful := 5

-- Derived conditions
def total_eaten := total_picked - total_needed
def number_of_handfuls := total_picked / strawberries_per_handful
def strawberries_eaten_per_handful := total_eaten / number_of_handfuls

-- The theorem we want to prove
theorem susan_strawberries_per_handful : strawberries_eaten_per_handful = 1 :=
by sorry

end susan_strawberries_per_handful_l198_198181


namespace find_value_of_x_squared_plus_y_squared_l198_198641

theorem find_value_of_x_squared_plus_y_squared (x y : ℝ) (h : (x^2 + y^2 + 1)^2 - 4 = 0) : x^2 + y^2 = 1 :=
by
  sorry

end find_value_of_x_squared_plus_y_squared_l198_198641


namespace find_a_l198_198263

noncomputable def tangent_to_circle_and_parallel (a : ℝ) : Prop := 
  let P := (2, 2)
  let circle_center := (1, 0)
  let on_circle := (P.1 - 1)^2 + P.2^2 = 5
  let perpendicular_slope := (P.2 - circle_center.2) / (P.1 - circle_center.1) * (1 / a) = -1
  on_circle ∧ perpendicular_slope

theorem find_a (a : ℝ) : tangent_to_circle_and_parallel a ↔ a = -2 :=
by
  sorry

end find_a_l198_198263


namespace apples_picked_per_tree_l198_198103

-- Definitions
def num_trees : Nat := 4
def total_apples_picked : Nat := 28

-- Proving how many apples Rachel picked from each tree if the same number were picked from each tree
theorem apples_picked_per_tree (h : num_trees ≠ 0) :
  total_apples_picked / num_trees = 7 :=
by
  sorry

end apples_picked_per_tree_l198_198103


namespace proportion_correct_l198_198767

theorem proportion_correct (x y : ℝ) (h : 3 * x = 2 * y) (hy : y ≠ 0) : x / 2 = y / 3 :=
by
  sorry

end proportion_correct_l198_198767


namespace expand_and_simplify_expression_l198_198128

theorem expand_and_simplify_expression : 
  ∀ (x : ℝ), (3 * x - 4) * (2 * x + 6) = 6 * x^2 + 10 * x - 24 := 
by 
  intro x
  sorry

end expand_and_simplify_expression_l198_198128


namespace fred_walking_speed_l198_198958

/-- 
Fred and Sam are standing 55 miles apart and they start walking in a straight line toward each other
at the same time. Fred walks at a certain speed and Sam walks at a constant speed of 5 miles per hour.
Sam has walked 25 miles when they meet.
-/
theorem fred_walking_speed
  (initial_distance : ℕ) 
  (sam_speed : ℕ)
  (sam_distance : ℕ) 
  (meeting_time : ℕ)
  (fred_distance : ℕ) 
  (fred_speed : ℕ)
  (h_initial_distance : initial_distance = 55)
  (h_sam_speed : sam_speed = 5)
  (h_sam_distance : sam_distance = 25)
  (h_meeting_time : meeting_time = 5)
  (h_fred_distance : fred_distance = 30)
  (h_fred_speed : fred_speed = 6)
  : fred_speed = fred_distance / meeting_time :=
by sorry

end fred_walking_speed_l198_198958


namespace kelly_peanut_weight_l198_198940

-- Define the total weight of snacks and the weight of raisins
def total_snacks_weight : ℝ := 0.5
def raisins_weight : ℝ := 0.4

-- Define the weight of peanuts as the remaining part
def peanuts_weight : ℝ := total_snacks_weight - raisins_weight

-- Theorem stating Kelly bought 0.1 pounds of peanuts
theorem kelly_peanut_weight : peanuts_weight = 0.1 :=
by
  -- proof would go here
  sorry

end kelly_peanut_weight_l198_198940


namespace non_congruent_rectangles_with_even_dimensions_l198_198113

/-- Given a rectangle with perimeter 120 inches and even integer dimensions,
    prove that there are 15 non-congruent rectangles that meet these criteria. -/
theorem non_congruent_rectangles_with_even_dimensions (h w : ℕ) (h_even : h % 2 = 0) (w_even : w % 2 = 0) (perimeter_condition : 2 * (h + w) = 120) :
  ∃ n : ℕ, n = 15 := sorry

end non_congruent_rectangles_with_even_dimensions_l198_198113


namespace fourth_rectangle_area_is_112_l198_198427

def area_of_fourth_rectangle (length : ℕ) (width : ℕ) (area1 : ℕ) (area2 : ℕ) (area3 : ℕ) : ℕ :=
  length * width - area1 - area2 - area3

theorem fourth_rectangle_area_is_112 :
  area_of_fourth_rectangle 20 12 24 48 36 = 112 :=
by
  sorry

end fourth_rectangle_area_is_112_l198_198427


namespace determine_triangle_ratio_l198_198974

theorem determine_triangle_ratio (a d : ℝ) (h : (a + d) ^ 2 = (a - d) ^ 2 + a ^ 2) : a / d = 2 + Real.sqrt 3 :=
sorry

end determine_triangle_ratio_l198_198974


namespace ming_estimate_less_l198_198384

theorem ming_estimate_less (x y δ : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : δ > 0) : 
  (x + δ) - (y + 2 * δ) < x - y :=
by 
  sorry

end ming_estimate_less_l198_198384


namespace value_of_a7_l198_198677

-- Define an arithmetic sequence
structure ArithmeticSeq (a : Nat → ℤ) :=
  (d : ℤ)
  (a_eq : ∀ n, a (n+1) = a n + d)

-- Lean statement of the equivalent proof problem
theorem value_of_a7 (a : ℕ → ℤ) (H : ArithmeticSeq a) :
  (2 * a 4 - a 7 ^ 2 + 2 * a 10 = 0) → a 7 = 4 * H.d :=
by
  sorry

end value_of_a7_l198_198677


namespace ending_number_divisible_by_3_l198_198171

theorem ending_number_divisible_by_3 (n : ℕ) :
  (∀ k, 0 ≤ k ∧ k < 13 → ∃ m, 10 ≤ m ∧ m ≤ n ∧ m % 3 = 0) →
  n = 48 :=
by
  intro h
  sorry

end ending_number_divisible_by_3_l198_198171


namespace area_of_triangle_DEF_eq_480_l198_198232

theorem area_of_triangle_DEF_eq_480 (DE EF DF : ℝ) (h1 : DE = 20) (h2 : EF = 48) (h3 : DF = 52) :
  let s := (DE + EF + DF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - EF) * (s - DF))
  area = 480 :=
by
  sorry

end area_of_triangle_DEF_eq_480_l198_198232


namespace sum_possible_x_eq_16_5_l198_198043

open Real

noncomputable def sum_of_possible_x : Real :=
  let a := 2
  let b := -33
  let c := 87
  (-b) / (2 * a)

theorem sum_possible_x_eq_16_5 : sum_of_possible_x = 16.5 :=
  by
    -- The actual proof goes here
    sorry

end sum_possible_x_eq_16_5_l198_198043


namespace obtuse_triangle_condition_l198_198250

theorem obtuse_triangle_condition
  (a b c : ℝ) 
  (h : ∃ A B C : ℝ, A + B + C = 180 ∧ A > 90 ∧ a^2 + b^2 - c^2 < 0)
  : (∃ A B C : ℝ, A + B + C = 180 ∧ A > 90 → a^2 + b^2 - c^2 < 0) := 
sorry

end obtuse_triangle_condition_l198_198250


namespace average_percentage_for_all_students_l198_198031

-- Definitions of the variables
def students1 : Nat := 15
def average1 : Nat := 75
def students2 : Nat := 10
def average2 : Nat := 90
def total_students : Nat := students1 + students2
def total_percentage1 : Nat := students1 * average1
def total_percentage2 : Nat := students2 * average2
def total_percentage : Nat := total_percentage1 + total_percentage2

-- Main theorem stating the average percentage for all students.
theorem average_percentage_for_all_students :
  total_percentage / total_students = 81 := by
  sorry

end average_percentage_for_all_students_l198_198031


namespace necessary_but_not_sufficient_l198_198691

noncomputable def is_increasing_on_R (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem necessary_but_not_sufficient (f : ℝ → ℝ) :
  (f 1 < f 2) → (¬∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∨ (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
by
  sorry

end necessary_but_not_sufficient_l198_198691


namespace blake_initial_amount_l198_198761

theorem blake_initial_amount (X : ℝ) (h1 : X > 0) (h2 : 3 * X / 2 = 30000) : X = 20000 :=
sorry

end blake_initial_amount_l198_198761


namespace total_amount_paid_l198_198537

theorem total_amount_paid (monthly_payment_1 monthly_payment_2 : ℕ) (years_1 years_2 : ℕ)
  (monthly_payment_1_eq : monthly_payment_1 = 300)
  (monthly_payment_2_eq : monthly_payment_2 = 350)
  (years_1_eq : years_1 = 3)
  (years_2_eq : years_2 = 2) :
  let annual_payment_1 := monthly_payment_1 * 12
  let annual_payment_2 := monthly_payment_2 * 12
  let total_1 := annual_payment_1 * years_1
  let total_2 := annual_payment_2 * years_2
  total_1 + total_2 = 19200 :=
by
  sorry

end total_amount_paid_l198_198537


namespace height_min_surface_area_l198_198243

def height_of_box (x : ℝ) : ℝ := x + 4

def surface_area (x : ℝ) : ℝ := 2 * x^2 + 4 * x * (x + 4)

theorem height_min_surface_area :
  ∀ x : ℝ, surface_area x ≥ 150 → x ≥ 5 → height_of_box x = 9 :=
by
  intros x h1 h2
  sorry

end height_min_surface_area_l198_198243


namespace reduced_price_per_kg_l198_198089

-- Define the conditions
def reduction_factor : ℝ := 0.80
def extra_kg : ℝ := 4
def total_cost : ℝ := 684

-- Assume the original price P and reduced price R
variables (P R : ℝ)

-- Define the equations derived from the conditions
def original_cost_eq := (P * 16 = total_cost)
def reduced_cost_eq := (0.80 * P * (16 + extra_kg) = total_cost)

-- The final theorem stating the reduced price per kg of oil is 34.20 Rs
theorem reduced_price_per_kg : R = 34.20 :=
by
  have h1: P * 16 = total_cost := sorry -- This will establish the original cost
  have h2: 0.80 * P * (16 + extra_kg) = total_cost := sorry -- This will establish the reduced cost
  have Q: 16 = 16 := sorry -- Calculation of Q (original quantity)
  have h3: P = 42.75 := sorry -- Calculation of original price
  have h4: R = 0.80 * P := sorry -- Calculation of reduced price
  have h5: R = 34.20 := sorry -- Final calculation matching the required answer
  exact h5

end reduced_price_per_kg_l198_198089


namespace no_valid_pairs_l198_198046

theorem no_valid_pairs : ∀ (x y : ℕ), x > 0 → y > 0 → x^2 + y^2 + 1 = x^3 → false := 
by
  intros x y hx hy h
  sorry

end no_valid_pairs_l198_198046


namespace find_solutions_l198_198784

theorem find_solutions (x : ℝ) : (x = -9 ∨ x = -3 ∨ x = 3) →
  (1 / (x^2 + 12 * x - 9) + 1 / (x^2 + 3 * x - 9) + 1 / (x^2 - 14 * x - 9) = 0) :=
by {
  sorry
}

end find_solutions_l198_198784


namespace find_halls_per_floor_l198_198661

theorem find_halls_per_floor
  (H : ℤ)
  (floors_first_wing : ℤ := 9)
  (rooms_per_hall_first_wing : ℤ := 32)
  (floors_second_wing : ℤ := 7)
  (halls_per_floor_second_wing : ℤ := 9)
  (rooms_per_hall_second_wing : ℤ := 40)
  (total_rooms : ℤ := 4248) :
  9 * H * 32 + 7 * 9 * 40 = 4248 → H = 6 :=
by
  sorry

end find_halls_per_floor_l198_198661


namespace converse_and_inverse_false_l198_198245

variable (Polygon : Type)
variable (RegularHexagon : Polygon → Prop)
variable (AllSidesEqual : Polygon → Prop)

theorem converse_and_inverse_false (p : Polygon → Prop) (q : Polygon → Prop)
  (h : ∀ x, RegularHexagon x → AllSidesEqual x) :
  ¬ (∀ x, AllSidesEqual x → RegularHexagon x) ∧ ¬ (∀ x, ¬ RegularHexagon x → ¬ AllSidesEqual x) :=
by
  sorry

end converse_and_inverse_false_l198_198245


namespace stations_equation_l198_198793

theorem stations_equation (x : ℕ) (h : x * (x - 1) = 1482) : true :=
by
  sorry

end stations_equation_l198_198793


namespace ratio_of_discount_l198_198520

theorem ratio_of_discount (price_pair1 price_pair2 : ℕ) (total_paid : ℕ) (discount_percent : ℕ) (h1 : price_pair1 = 40)
    (h2 : price_pair2 = 60) (h3 : total_paid = 60) (h4 : discount_percent = 50) :
    (price_pair1 * discount_percent / 100) / (price_pair1 + (price_pair2 - price_pair1 * discount_percent / 100)) = 1 / 4 :=
by
  sorry

end ratio_of_discount_l198_198520


namespace derivative_at_pi_over_4_l198_198175

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_over_4 : (deriv f) (Real.pi / 4) = 0 := 
by
  sorry

end derivative_at_pi_over_4_l198_198175


namespace opposite_of_neg_2022_l198_198808

theorem opposite_of_neg_2022 : -(-2022) = 2022 :=
by
  sorry

end opposite_of_neg_2022_l198_198808


namespace elizabeth_money_l198_198773

theorem elizabeth_money :
  (∀ (P N : ℝ), P = 5 → N = 6 → 
    (P * 1.60 + N * 2.00) = 20.00) :=
by
  sorry

end elizabeth_money_l198_198773


namespace problem_U_l198_198965

theorem problem_U :
  ( (1 : ℝ) / (4 - Real.sqrt 15) - (1 / (Real.sqrt 15 - Real.sqrt 14))
  + (1 / (Real.sqrt 14 - 3)) - (1 / (3 - Real.sqrt 12))
  + (1 / (Real.sqrt 12 - Real.sqrt 11)) ) = 10 + Real.sqrt 11 :=
by
  sorry

end problem_U_l198_198965


namespace haley_total_expenditure_l198_198572

-- Definition of conditions
def ticket_cost : ℕ := 4
def tickets_bought_for_self_and_friends : ℕ := 3
def tickets_bought_for_others : ℕ := 5
def total_tickets : ℕ := tickets_bought_for_self_and_friends + tickets_bought_for_others

-- Proof statement
theorem haley_total_expenditure : total_tickets * ticket_cost = 32 := by
  sorry

end haley_total_expenditure_l198_198572


namespace car_win_probability_l198_198056

noncomputable def P (n : ℕ) : ℚ := 1 / n

theorem car_win_probability :
  let P_x := 1 / 7
  let P_y := 1 / 3
  let P_z := 1 / 5
  P_x + P_y + P_z = 71 / 105 :=
by
  sorry

end car_win_probability_l198_198056


namespace alicia_tax_cents_per_hour_l198_198796

-- Define Alicia's hourly wage in dollars.
def alicia_hourly_wage_dollars : ℝ := 25
-- Define the conversion rate from dollars to cents.
def cents_per_dollar : ℝ := 100
-- Define the local tax rate as a percentage.
def tax_rate_percent : ℝ := 2

-- Convert Alicia's hourly wage to cents.
def alicia_hourly_wage_cents : ℝ := alicia_hourly_wage_dollars * cents_per_dollar

-- Define the theorem that needs to be proved.
theorem alicia_tax_cents_per_hour : alicia_hourly_wage_cents * (tax_rate_percent / 100) = 50 := by
  sorry

end alicia_tax_cents_per_hour_l198_198796


namespace positive_real_solutions_unique_l198_198779

variable (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (x y z : ℝ)

theorem positive_real_solutions_unique :
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = abc →
    (x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2) :=
by
  intros
  sorry

end positive_real_solutions_unique_l198_198779


namespace largest_amount_received_back_l198_198606

theorem largest_amount_received_back 
  (x y x_lost y_lost : ℕ) 
  (h1 : 20 * x + 100 * y = 3000) 
  (h2 : x_lost + y_lost = 16) 
  (h3 : x_lost = y_lost + 2 ∨ x_lost = y_lost - 2) 
  : (3000 - (20 * x_lost + 100 * y_lost) = 2120) :=
sorry

end largest_amount_received_back_l198_198606


namespace area_of_region_l198_198516

theorem area_of_region :
  (∃ (x y: ℝ), x^2 + y^2 = 5 * |x - y| + 2 * |x + y|) → 
  (∃ (A : ℝ), A = 14.5 * Real.pi) :=
sorry

end area_of_region_l198_198516


namespace total_pencils_correct_total_erasers_correct_total_rulers_correct_total_sharpeners_correct_total_skittles_correct_l198_198110

-- Given conditions
def kids_A := 7
def kids_B := 9
def kids_C := 5

def pencils_per_child_A := 4
def erasers_per_child_A := 2
def skittles_per_child_A := 13

def pencils_per_child_B := 6
def rulers_per_child_B := 1
def skittles_per_child_B := 8

def pencils_per_child_C := 3
def sharpeners_per_child_C := 1
def skittles_per_child_C := 15

-- Calculated totals
def total_pencils := kids_A * pencils_per_child_A + kids_B * pencils_per_child_B + kids_C * pencils_per_child_C
def total_erasers := kids_A * erasers_per_child_A
def total_rulers := kids_B * rulers_per_child_B
def total_sharpeners := kids_C * sharpeners_per_child_C
def total_skittles := kids_A * skittles_per_child_A + kids_B * skittles_per_child_B + kids_C * skittles_per_child_C

-- Proof obligations
theorem total_pencils_correct : total_pencils = 97 := by
  sorry

theorem total_erasers_correct : total_erasers = 14 := by
  sorry

theorem total_rulers_correct : total_rulers = 9 := by
  sorry

theorem total_sharpeners_correct : total_sharpeners = 5 := by
  sorry

theorem total_skittles_correct : total_skittles = 238 := by
  sorry

end total_pencils_correct_total_erasers_correct_total_rulers_correct_total_sharpeners_correct_total_skittles_correct_l198_198110


namespace basketball_club_members_l198_198662

theorem basketball_club_members :
  let sock_cost := 6
  let tshirt_additional_cost := 8
  let total_cost := 4440
  let cost_per_member := sock_cost + 2 * (sock_cost + tshirt_additional_cost)
  total_cost / cost_per_member = 130 :=
by
  sorry

end basketball_club_members_l198_198662


namespace sum_of_squares_of_roots_eq_zero_l198_198707

theorem sum_of_squares_of_roots_eq_zero :
  let f : Polynomial ℝ := Polynomial.C 50 + Polynomial.monomial 3 (-2) + Polynomial.monomial 7 5 + Polynomial.monomial 10 1
  ∀ (r : ℝ), r ∈ Multiset.toFinset f.roots → r ^ 2 = 0 :=
by
  sorry

end sum_of_squares_of_roots_eq_zero_l198_198707


namespace parabola_intersection_square_l198_198251

theorem parabola_intersection_square (p : ℝ) :
   (∃ (x : ℝ), (x = 1 ∨ x = 2) ∧ x^2 * p = 1 ∨ x^2 * p = 2)
   → (1 / 4 ≤ p ∧ p ≤ 2) :=
by
  sorry

end parabola_intersection_square_l198_198251


namespace product_of_repeating_decimals_l198_198914

noncomputable def repeating_decimal_038 : ℚ := 38 / 999
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem product_of_repeating_decimals :
  repeating_decimal_038 * repeating_decimal_4 = 152 / 8991 :=
by
  sorry

end product_of_repeating_decimals_l198_198914


namespace range_of_a_l198_198949

/-- Given a fixed point A(a, 3) is outside the circle x^2 + y^2 - 2ax - 3y + a^2 + a = 0,
we want to show that the range of values for a is (0, 9/4). -/
theorem range_of_a (a : ℝ) :
  (∃ (A : ℝ × ℝ), A = (a, 3) ∧ ¬(∃ (x y : ℝ), x^2 + y^2 - 2*a*x - 3*y + a^2 + a = 0))
  ↔ (0 < a ∧ a < 9/4) :=
sorry

end range_of_a_l198_198949


namespace product_of_roots_eq_neg30_l198_198635

theorem product_of_roots_eq_neg30 (x : ℝ) (h : (x + 3) * (x - 4) = 18) : 
  (∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = -30) :=
sorry

end product_of_roots_eq_neg30_l198_198635


namespace work_problem_l198_198002

theorem work_problem 
  (A_work_time : ℤ) 
  (B_work_time : ℤ) 
  (x : ℤ)
  (A_work_rate : ℚ := 1 / 15 )
  (work_left : ℚ := 0.18333333333333335)
  (worked_together_for : ℚ := 7)
  (work_done : ℚ := 1 - work_left) :
  (7 * (1 / 15 + 1 / x) = work_done) → x = 20 :=
by sorry

end work_problem_l198_198002


namespace smith_a_students_l198_198719

-- Definitions representing the conditions

def johnson_a_students : ℕ := 12
def johnson_total_students : ℕ := 20
def smith_total_students : ℕ := 30

def johnson_ratio := johnson_a_students / johnson_total_students

-- Statement to prove
theorem smith_a_students :
  (johnson_a_students / johnson_total_students) = (18 / smith_total_students) :=
sorry

end smith_a_students_l198_198719


namespace inequality_proved_l198_198133

theorem inequality_proved (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_proved_l198_198133


namespace find_C_l198_198822

theorem find_C (A B C : ℝ) 
  (h1 : A + B + C = 600) 
  (h2 : A + C = 250) 
  (h3 : B + C = 450) : 
  C = 100 :=
sorry

end find_C_l198_198822


namespace find_principal_sum_l198_198411

theorem find_principal_sum
  (R : ℝ) (P : ℝ)
  (H1 : 0 < R)
  (H2 : 8 * 10 * P / 100 = 150) :
  P = 187.50 :=
by
  sorry

end find_principal_sum_l198_198411


namespace equation_has_exactly_one_real_solution_l198_198862

-- Definitions for the problem setup
def equation (k : ℝ) (x : ℝ) : Prop := (3 * x + 8) * (x - 6) = -54 + k * x

-- The property that we need to prove
theorem equation_has_exactly_one_real_solution (k : ℝ) :
  (∀ x : ℝ, equation k x → ∃! x : ℝ, equation k x) ↔ k = 6 * Real.sqrt 2 - 10 ∨ k = -6 * Real.sqrt 2 - 10 := 
sorry

end equation_has_exactly_one_real_solution_l198_198862


namespace odd_function_value_l198_198166

def f (a x : ℝ) : ℝ := -x^3 + (a-2)*x^2 + x

-- Test that f(x) is an odd function:
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem odd_function_value (a : ℝ) (h : is_odd_function (f a)) : f a a = -6 :=
by
  sorry

end odd_function_value_l198_198166


namespace sector_area_l198_198856

theorem sector_area (θ r : ℝ) (hθ : θ = 2) (hr : r = 1) :
  (1 / 2) * r^2 * θ = 1 :=
by
  -- Conditions are instantiated
  rw [hθ, hr]
  -- Simplification is left to the proof
  sorry

end sector_area_l198_198856


namespace max_non_cyclic_handshakes_l198_198936

theorem max_non_cyclic_handshakes (n : ℕ) (h : n = 18) : 
  (n * (n - 1)) / 2 = 153 := by
  sorry

end max_non_cyclic_handshakes_l198_198936


namespace equivalent_problem_l198_198571

theorem equivalent_problem (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n < 29) (h₃ : 2 * n % 29 = 1) :
  (3^n % 29)^3 - 3 % 29 = 3 :=
sorry

end equivalent_problem_l198_198571


namespace sum_powers_is_76_l198_198061

theorem sum_powers_is_76 (m n : ℕ) (h1 : m + n = 1) (h2 : m^2 + n^2 = 3)
                         (h3 : m^3 + n^3 = 4) (h4 : m^4 + n^4 = 7)
                         (h5 : m^5 + n^5 = 11) : m^9 + n^9 = 76 :=
sorry

end sum_powers_is_76_l198_198061


namespace least_integer_greater_than_sqrt_450_l198_198248

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l198_198248


namespace problem1_problem2_problem3_problem4_l198_198055

section

variables (x y : Real)

-- Given conditions
def x_def : x = 3 + 2 * Real.sqrt 2 := sorry
def y_def : y = 3 - 2 * Real.sqrt 2 := sorry

-- Problem 1: Prove x + y = 6
theorem problem1 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x + y = 6 := 
by sorry

-- Problem 2: Prove x - y = 4 * sqrt 2
theorem problem2 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x - y = 4 * Real.sqrt 2 :=
by sorry

-- Problem 3: Prove xy = 1
theorem problem3 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x * y = 1 := 
by sorry

-- Problem 4: Prove x^2 - 3xy + y^2 - x - y = 25
theorem problem4 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x^2 - 3 * x * y + y^2 - x - y = 25 :=
by sorry

end

end problem1_problem2_problem3_problem4_l198_198055


namespace Frank_worked_days_l198_198512

theorem Frank_worked_days
  (h_per_day : ℕ) (total_hours : ℕ) (d : ℕ) 
  (h_day_def : h_per_day = 8) 
  (total_hours_def : total_hours = 32) 
  (d_def : d = total_hours / h_per_day) : 
  d = 4 :=
by 
  rw [total_hours_def, h_day_def] at d_def
  exact d_def

end Frank_worked_days_l198_198512


namespace min_moves_seven_chests_l198_198205

/-
Problem:
Seven chests are placed in a circle, each containing a certain number of coins: [20, 15, 5, 6, 10, 17, 18].
Prove that the minimum number of moves required to equalize the number of coins in all chests is 22.
-/

def min_moves_to_equalize_coins (coins: List ℕ) : ℕ :=
  -- Function that would calculate the minimum number of moves
  sorry

theorem min_moves_seven_chests :
  min_moves_to_equalize_coins [20, 15, 5, 6, 10, 17, 18] = 22 :=
sorry

end min_moves_seven_chests_l198_198205


namespace smallest_integer_condition_l198_198274

theorem smallest_integer_condition (x : ℝ) (hz : 9 = 9) (hineq : 27^9 > x^24) : x < 27 :=
  by {
    sorry
  }

end smallest_integer_condition_l198_198274


namespace model_A_sampling_l198_198310

theorem model_A_sampling (prod_A prod_B prod_C total_prod total_sampled : ℕ)
    (hA : prod_A = 1200) (hB : prod_B = 6000) (hC : prod_C = 2000)
    (htotal : total_prod = prod_A + prod_B + prod_C) (htotal_car : total_prod = 9200)
    (hsampled : total_sampled = 46) :
    (prod_A * total_sampled) / total_prod = 6 := by
  sorry

end model_A_sampling_l198_198310


namespace solution_for_x2_l198_198966

def eq1 (x : ℝ) := 2 * x = 6
def eq2 (x : ℝ) := x + 2 = 0
def eq3 (x : ℝ) := x - 5 = 3
def eq4 (x : ℝ) := 3 * x - 6 = 0

theorem solution_for_x2 : ∀ x : ℝ, x = 2 → ¬eq1 x ∧ ¬eq2 x ∧ ¬eq3 x ∧ eq4 x :=
by 
  sorry

end solution_for_x2_l198_198966


namespace problem1_l198_198016

theorem problem1 (A B C : Prop) : (A ∨ (B ∧ C)) ↔ ((A ∨ B) ∧ (A ∨ C)) :=
sorry 

end problem1_l198_198016


namespace range_of_f_l198_198178

theorem range_of_f (x : ℝ) (h : x ∈ Set.Icc (-3 : ℝ) 3) : 
  ∃ y, y ∈ Set.Icc (0 : ℝ) 25 ∧ ∀ z, z = (x^2 - 4*x + 4) → y = z :=
sorry

end range_of_f_l198_198178


namespace find_r_divisibility_l198_198682

theorem find_r_divisibility (r : ℝ) :
  (∃ s : ℝ, 10 * (x - r)^2 * (x - s) = 10 * x^3 - 5 * x^2 - 52 * x + 56) → r = 4 / 3 :=
by
  sorry

end find_r_divisibility_l198_198682


namespace race_distance_100_l198_198063

noncomputable def race_distance (a b c d : ℝ) :=
  (d / a = (d - 20) / b) ∧
  (d / b = (d - 10) / c) ∧
  (d / a = (d - 28) / c) 

theorem race_distance_100 (a b c d : ℝ) (h1 : d / a = (d - 20) / b) (h2 : d / b = (d - 10) / c) (h3 : d / a = (d - 28) / c) : 
  d = 100 :=
  sorry

end race_distance_100_l198_198063


namespace parabola_equation_l198_198878

def equation_of_parabola (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = a * x^2 + b * x + c ↔ 
              (∃ a : ℝ, y = a * (x - 3)^2 + 5) ∧
              y = (if x = 0 then 2 else y)

theorem parabola_equation :
  equation_of_parabola (-1 / 3) 2 2 :=
by
  -- First, show that the vertex form (x-3)^2 + 5 meets the conditions
  sorry

end parabola_equation_l198_198878


namespace constant_term_in_modified_equation_l198_198612

theorem constant_term_in_modified_equation :
  ∃ (c : ℝ), ∀ (q : ℝ), (3 * (3 * 5 - 3) - 3 + c = 132) → c = 99 := 
by
  sorry

end constant_term_in_modified_equation_l198_198612


namespace increased_expenses_percent_l198_198471

theorem increased_expenses_percent (S : ℝ) (hS : S = 6250) (initial_save_percent : ℝ) (final_savings : ℝ) 
  (initial_save_percent_def : initial_save_percent = 20) 
  (final_savings_def : final_savings = 250) : 
  (initial_save_percent / 100 * S - final_savings) / (S - initial_save_percent / 100 * S) * 100 = 20 := by
  sorry

end increased_expenses_percent_l198_198471


namespace find_eighth_number_l198_198088

theorem find_eighth_number (x : ℕ) (h1 : (1 + 2 + 4 + 5 + 6 + 9 + 9 + x + 12) / 9 = 7) : x = 27 :=
sorry

end find_eighth_number_l198_198088


namespace watch_cost_price_l198_198766

noncomputable def cost_price : ℝ := 1166.67

theorem watch_cost_price (CP : ℝ) (loss_percent gain_percent : ℝ) (delta : ℝ) 
  (h1 : loss_percent = 0.10) 
  (h2 : gain_percent = 0.02) 
  (h3 : delta = 140) 
  (h4 : (1 - loss_percent) * CP + delta = (1 + gain_percent) * CP) : 
  CP = cost_price := 
by 
  sorry

end watch_cost_price_l198_198766


namespace abc_ineq_l198_198356

theorem abc_ineq (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
by 
  sorry

end abc_ineq_l198_198356


namespace count_paths_l198_198252

theorem count_paths (m n : ℕ) : (n + m).choose m = (n + m).choose n :=
by
  sorry

end count_paths_l198_198252


namespace area_of_rectangle_l198_198932

theorem area_of_rectangle (side radius length breadth : ℕ) (h1 : side^2 = 784) (h2 : radius = side) (h3 : length = radius / 4) (h4 : breadth = 5) : length * breadth = 35 :=
by
  -- proof to be filled here
  sorry

end area_of_rectangle_l198_198932


namespace units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial_l198_198207

/-- Find the units digit of the largest power of 2 that divides into (2^5)! -/
theorem units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial : ∃ d : ℕ, d = 8 := by
  sorry

end units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial_l198_198207


namespace jordan_novels_read_l198_198482

variable (J A : ℕ)

theorem jordan_novels_read (h1 : A = (1 / 10) * J)
                          (h2 : J = A + 108) :
                          J = 120 := 
by
  sorry

end jordan_novels_read_l198_198482


namespace find_C_l198_198090

noncomputable def A_annual_income : ℝ := 403200.0000000001
noncomputable def A_monthly_income : ℝ := A_annual_income / 12 -- 33600.00000000001

noncomputable def x : ℝ := A_monthly_income / 5 -- 6720.000000000002

noncomputable def C : ℝ := (2 * x) / 1.12 -- should be 12000.000000000004

theorem find_C : C = 12000.000000000004 := 
by sorry

end find_C_l198_198090


namespace saplings_problem_l198_198764

theorem saplings_problem (x : ℕ) :
  (∃ n : ℕ, 5 * x + 3 = n ∧ 6 * x - 4 = n) ↔ 5 * x + 3 = 6 * x - 4 :=
by
  sorry

end saplings_problem_l198_198764


namespace minimum_x2y3z_l198_198118

theorem minimum_x2y3z (x y z : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : x^3 + y^3 + z^3 - 3 * x * y * z = 607) : 
  x + 2 * y + 3 * z ≥ 1215 :=
sorry

end minimum_x2y3z_l198_198118


namespace product_remainder_mod_7_l198_198995

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l198_198995


namespace residue_neg_1234_mod_32_l198_198478

theorem residue_neg_1234_mod_32 : -1234 % 32 = 14 := 
by sorry

end residue_neg_1234_mod_32_l198_198478


namespace swim_speed_in_still_water_l198_198120

-- Definitions from conditions
def downstream_speed (v_man v_stream : ℝ) : ℝ := v_man + v_stream
def upstream_speed (v_man v_stream : ℝ) : ℝ := v_man - v_stream

-- Question formatted as a proof problem
theorem swim_speed_in_still_water (v_man v_stream : ℝ)
  (h1 : downstream_speed v_man v_stream = 6)
  (h2 : upstream_speed v_man v_stream = 10) : v_man = 8 :=
by
  -- The proof will come here
  sorry

end swim_speed_in_still_water_l198_198120


namespace solve_identity_l198_198077

theorem solve_identity (x : ℝ) (a b p q : ℝ)
  (h : (6 * x + 1) / (6 * x ^ 2 + 19 * x + 15) = a / (x - p) + b / (x - q)) :
  a = -1 ∧ b = 2 ∧ p = -3/4 ∧ q = -5/3 :=
by
  sorry

end solve_identity_l198_198077


namespace last_digit_of_prime_l198_198674

theorem last_digit_of_prime (n : ℕ) (h1 : 859433 = 214858 * 4 + 1) : (2 ^ 859433 - 1) % 10 = 1 := by
  sorry

end last_digit_of_prime_l198_198674


namespace floor_T_equals_150_l198_198647

variable {p q r s : ℝ}

theorem floor_T_equals_150
  (hpq_sum_of_squares : p^2 + q^2 = 2500)
  (hrs_sum_of_squares : r^2 + s^2 = 2500)
  (hpq_product : p * q = 1225)
  (hrs_product : r * s = 1225)
  (hp_plus_s : p + s = 75) :
  ∃ T : ℝ, T = p + q + r + s ∧ ⌊T⌋ = 150 :=
by
  sorry

end floor_T_equals_150_l198_198647


namespace final_number_l198_198323

variables (crab goat bear cat hen : ℕ)

-- Given conditions
def row4_sum : Prop := 5 * crab = 10
def col5_sum : Prop := 4 * crab + goat = 11
def row2_sum : Prop := 2 * goat + crab + 2 * bear = 16
def col2_sum : Prop := cat + bear + 2 * goat + crab = 13
def col3_sum : Prop := 2 * crab + 2 * hen + goat = 17

-- Theorem statement
theorem final_number
  (hcrab : row4_sum crab)
  (hgoat_col5 : col5_sum crab goat)
  (hbear_row2 : row2_sum crab goat bear)
  (hcat_col2 : col2_sum cat crab bear goat)
  (hhen_col3 : col3_sum crab goat hen) :
  crab = 2 ∧ goat = 3 ∧ bear = 4 ∧ cat = 1 ∧ hen = 5 → (cat * 10000 + hen * 1000 + crab * 100 + bear * 10 + goat = 15243) :=
sorry

end final_number_l198_198323


namespace cost_price_of_watch_l198_198262

theorem cost_price_of_watch (CP : ℝ) (h_loss : 0.54 * CP = SP_loss)
                            (h_gain : 1.04 * CP = SP_gain)
                            (h_diff : SP_gain - SP_loss = 140) :
                            CP = 280 :=
by {
    sorry
}

end cost_price_of_watch_l198_198262


namespace meeting_time_l198_198078

noncomputable def combined_speed : ℕ := 10 -- km/h
noncomputable def distance_to_cover : ℕ := 50 -- km
noncomputable def start_time : ℕ := 6 -- pm (in hours)
noncomputable def speed_a : ℕ := 6 -- km/h
noncomputable def speed_b : ℕ := 4 -- km/h

theorem meeting_time : start_time + (distance_to_cover / combined_speed) = 11 :=
by
  sorry

end meeting_time_l198_198078


namespace subjects_difference_marius_monica_l198_198559

-- Definitions of given conditions.
def Monica_subjects : ℕ := 10
def Total_subjects : ℕ := 41
def Millie_offset : ℕ := 3

-- Theorem to prove the question == answer given conditions
theorem subjects_difference_marius_monica : 
  ∃ (M : ℕ), (M + (M + Millie_offset) + Monica_subjects = Total_subjects) ∧ (M - Monica_subjects = 4) := 
by
  sorry

end subjects_difference_marius_monica_l198_198559


namespace total_number_of_animals_is_650_l198_198573

def snake_count : Nat := 100
def arctic_fox_count : Nat := 80
def leopard_count : Nat := 20
def bee_eater_count : Nat := 10 * leopard_count
def cheetah_count : Nat := snake_count / 2
def alligator_count : Nat := 2 * (arctic_fox_count + leopard_count)

def total_animal_count : Nat :=
  snake_count + arctic_fox_count + leopard_count + bee_eater_count + cheetah_count + alligator_count

theorem total_number_of_animals_is_650 :
  total_animal_count = 650 :=
by
  sorry

end total_number_of_animals_is_650_l198_198573


namespace percentage_music_l198_198258

variable (students_total : ℕ)
variable (percent_dance percent_art percent_drama percent_sports percent_photography percent_music : ℝ)

-- Define the problem conditions
def school_conditions : Prop :=
  students_total = 3000 ∧
  percent_dance = 0.125 ∧
  percent_art = 0.22 ∧
  percent_drama = 0.135 ∧
  percent_sports = 0.15 ∧
  percent_photography = 0.08 ∧
  percent_music = 1 - (percent_dance + percent_art + percent_drama + percent_sports + percent_photography)

-- Define the proof statement
theorem percentage_music : school_conditions students_total percent_dance percent_art percent_drama percent_sports percent_photography percent_music → percent_music = 0.29 :=
by
  intros h
  rw [school_conditions] at h
  sorry

end percentage_music_l198_198258


namespace cost_of_each_big_apple_l198_198687

theorem cost_of_each_big_apple :
  ∀ (small_cost medium_cost : ℝ) (big_cost : ℝ) (num_small num_medium num_big : ℕ) (total_cost : ℝ),
  small_cost = 1.5 →
  medium_cost = 2 →
  num_small = 6 →
  num_medium = 6 →
  num_big = 8 →
  total_cost = 45 →
  total_cost = num_small * small_cost + num_medium * medium_cost + num_big * big_cost →
  big_cost = 3 :=
by
  intros small_cost medium_cost big_cost num_small num_medium num_big total_cost
  sorry

end cost_of_each_big_apple_l198_198687


namespace find_nine_day_segment_l198_198217

/-- 
  Definitions:
  - ws_day: The Winter Solstice day, December 21, 2012.
  - j1_day: New Year's Day, January 1, 2013.
  - Calculate the total days difference between ws_day and j1_day.
  - Check that the distribution of days into 9-day segments leads to January 1, 2013, being the third day of the second segment.
-/
def ws_day : ℕ := 21
def j1_day : ℕ := 1
def days_in_december : ℕ := 31
def days_ws_to_end_dec : ℕ := days_in_december - ws_day + 1
def total_days : ℕ := days_ws_to_end_dec + j1_day

theorem find_nine_day_segment : (total_days % 9) = 3 ∧ (total_days / 9) = 1 := by
  sorry  -- Proof skipped

end find_nine_day_segment_l198_198217


namespace line_does_not_pass_second_quadrant_l198_198439

-- Definitions of conditions
variables (k b x y : ℝ)
variable  (h₁ : k > 0) -- condition k > 0
variable  (h₂ : b < 0) -- condition b < 0


theorem line_does_not_pass_second_quadrant : 
  ¬∃ (x y : ℝ), (x < 0 ∧ y > 0) ∧ (y = k * x + b) :=
sorry

end line_does_not_pass_second_quadrant_l198_198439


namespace incorrect_conclusion_l198_198009

theorem incorrect_conclusion (b x : ℂ) (h : x^2 - b * x + 1 = 0) : x = 1 ∨ x = -1
  ↔ (b = 2 ∨ b = -2) :=
by sorry

end incorrect_conclusion_l198_198009


namespace sum21_exists_l198_198440

theorem sum21_exists (S : Finset ℕ) (h_size : S.card = 11) (h_range : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 20) :
  ∃ a b, a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ a + b = 21 :=
by
  sorry

end sum21_exists_l198_198440


namespace proof_equivalence_l198_198610

variables {a b c d e f : Prop}

theorem proof_equivalence (h₁ : (a ≥ b) → (c > d)) 
                        (h₂ : (c > d) → (a ≥ b)) 
                        (h₃ : (a < b) ↔ (e ≤ f)) :
  (c ≤ d) ↔ (e ≤ f) :=
sorry

end proof_equivalence_l198_198610


namespace marbles_per_pack_l198_198595

theorem marbles_per_pack (total_marbles : ℕ) (leo_packs manny_packs neil_packs total_packs : ℕ) 
(h1 : total_marbles = 400) 
(h2 : leo_packs = 25) 
(h3 : manny_packs = total_packs / 4) 
(h4 : neil_packs = total_packs / 8) 
(h5 : leo_packs + manny_packs + neil_packs = total_packs) : 
total_marbles / total_packs = 10 := 
by sorry

end marbles_per_pack_l198_198595


namespace smallest_solution_eq_sqrt_104_l198_198637

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l198_198637


namespace trader_profit_percentage_l198_198840

-- Definitions for the conditions
def original_price (P : ℝ) : ℝ := P
def discount_price (P : ℝ) : ℝ := 0.80 * P
def selling_price (P : ℝ) : ℝ := 0.80 * P * 1.45

-- Theorem statement including the problem's question and the correct answer
theorem trader_profit_percentage (P : ℝ) (hP : 0 < P) : 
  (selling_price P - original_price P) / original_price P * 100 = 16 :=
by
  sorry

end trader_profit_percentage_l198_198840


namespace angle_C_in_parallelogram_l198_198806

theorem angle_C_in_parallelogram (ABCD : Type)
  (angle_A angle_B angle_C angle_D : ℝ)
  (h1 : angle_A = angle_C)
  (h2 : angle_B = angle_D)
  (h3 : angle_A + angle_B = 180)
  (h4 : angle_A / angle_B = 3) :
  angle_C = 135 :=
  sorry

end angle_C_in_parallelogram_l198_198806


namespace average_class_weight_l198_198197

theorem average_class_weight
  (n_boys n_girls n_total : ℕ)
  (avg_weight_boys avg_weight_girls total_students : ℕ)
  (h1 : n_boys = 15)
  (h2 : n_girls = 10)
  (h3 : n_total = 25)
  (h4 : avg_weight_boys = 48)
  (h5 : avg_weight_girls = 405 / 10) 
  (h6 : total_students = 25) :
  (48 * 15 + 40.5 * 10) / 25 = 45 := 
sorry

end average_class_weight_l198_198197


namespace chuck_total_time_on_trip_l198_198058

def distance_into_country : ℝ := 28.8
def rate_out : ℝ := 16
def rate_back : ℝ := 24

theorem chuck_total_time_on_trip : (distance_into_country / rate_out) + (distance_into_country / rate_back) = 3 := 
by sorry

end chuck_total_time_on_trip_l198_198058


namespace arnold_total_protein_l198_198211

-- Conditions
def protein_in_collagen_powder (scoops: ℕ) : ℕ := 9 * scoops
def protein_in_protein_powder (scoops: ℕ) : ℕ := 21 * scoops
def protein_in_steak : ℕ := 56
def protein_in_greek_yogurt : ℕ := 15
def protein_in_almonds (cups: ℕ) : ℕ := 6 * (cups * 4) / 4
def half_cup_almonds_protein : ℕ := 12

-- Statement
theorem arnold_total_protein : 
  protein_in_collagen_powder 1 + protein_in_protein_powder 2 + protein_in_steak + protein_in_greek_yogurt + half_cup_almonds_protein = 134 :=
  by
    sorry

end arnold_total_protein_l198_198211


namespace derivative_of_f_l198_198095

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((2 * x + 1) / Real.sqrt 2) + (2 * x + 1) / (4 * x^2 + 4 * x + 3)

theorem derivative_of_f (x : ℝ) : deriv f x = 8 / (4 * x^2 + 4 * x + 3)^2 :=
by
  -- Proof will be provided here
  sorry

end derivative_of_f_l198_198095


namespace vertex_of_parabola_l198_198837

theorem vertex_of_parabola (x y : ℝ) : (y^2 - 4 * y + 3 * x + 7 = 0) → (x, y) = (-1, 2) :=
by
  sorry

end vertex_of_parabola_l198_198837


namespace selling_price_l198_198579

theorem selling_price 
  (cost_price : ℝ) 
  (profit_percentage : ℝ) 
  (h_cost : cost_price = 192) 
  (h_profit : profit_percentage = 0.25) : 
  ∃ selling_price : ℝ, selling_price = cost_price * (1 + profit_percentage) := 
by {
  sorry
}

end selling_price_l198_198579


namespace bob_km_per_gallon_l198_198374

-- Define the total distance Bob can drive.
def total_distance : ℕ := 100

-- Define the total amount of gas in gallons Bob's car uses.
def total_gas : ℕ := 10

-- Define the expected kilometers per gallon
def expected_km_per_gallon : ℕ := 10

-- Define the statement we want to prove
theorem bob_km_per_gallon : total_distance / total_gas = expected_km_per_gallon :=
by 
  sorry

end bob_km_per_gallon_l198_198374


namespace ceil_of_neg_sqrt_frac_64_over_9_l198_198893

theorem ceil_of_neg_sqrt_frac_64_over_9 :
  ⌈-Real.sqrt (64 / 9)⌉ = -2 :=
by
  sorry

end ceil_of_neg_sqrt_frac_64_over_9_l198_198893


namespace remainder_sum_first_150_div_11300_l198_198810

theorem remainder_sum_first_150_div_11300 :
  let n := 150
  let S := n * (n + 1) / 2
  S % 11300 = 25 :=
by
  let n := 150
  let S := n * (n + 1) / 2
  show S % 11300 = 25
  sorry

end remainder_sum_first_150_div_11300_l198_198810


namespace correct_statement_l198_198755

-- Defining the conditions
def freq_eq_prob : Prop :=
  ∀ (f p : ℝ), f = p

def freq_objective : Prop :=
  ∀ (f : ℝ) (n : ℕ), f = f

def freq_stabilizes : Prop :=
  ∀ (p : ℝ), ∃ (f : ℝ) (n : ℕ), f = p

def prob_random : Prop :=
  ∀ (p : ℝ), p = p

-- The statement we need to prove
theorem correct_statement :
  ¬freq_eq_prob ∧ ¬freq_objective ∧ freq_stabilizes ∧ ¬prob_random :=
by
  sorry

end correct_statement_l198_198755


namespace no_base_for_final_digit_one_l198_198226

theorem no_base_for_final_digit_one (b : ℕ) (h : 3 ≤ b ∧ b ≤ 10) : ¬ (842 % b = 1) :=
by
  cases h with 
  | intro hb1 hb2 => sorry

end no_base_for_final_digit_one_l198_198226


namespace ellipse_focus_value_k_l198_198390

theorem ellipse_focus_value_k 
  (k : ℝ)
  (h : ∀ x y, 5 * x^2 + k * y^2 = 5 → abs y ≠ 2 → ∀ c : ℝ, c^2 = 4 → k = 1) :
  ∀ k : ℝ, (5 * (0:ℝ)^2 + k * (2:ℝ)^2 = 5) ∧ (5 * (0:ℝ)^2 + k * (-(2:ℝ))^2 = 5) → k = 1 := by
  sorry

end ellipse_focus_value_k_l198_198390


namespace team_problem_solved_probability_l198_198298

-- Defining the probabilities
def P_A : ℚ := 1 / 5
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Defining the probability that the problem is solved
def P_s : ℚ := 3 / 5

-- Lean 4 statement to prove that the calculated probability matches the expected solution
theorem team_problem_solved_probability :
  1 - (1 - P_A) * (1 - P_B) * (1 - P_C) = P_s :=
by
  sorry

end team_problem_solved_probability_l198_198298


namespace parabola_hyperbola_focus_l198_198885

theorem parabola_hyperbola_focus {p : ℝ} :
  let focus_parabola := (p / 2, 0)
  let focus_hyperbola := (2, 0)
  focus_parabola = focus_hyperbola -> p = 4 :=
by
  intro h
  sorry

end parabola_hyperbola_focus_l198_198885


namespace all_three_use_media_l198_198333

variable (U T R M T_and_M T_and_R R_and_M T_and_R_and_M : ℕ)

theorem all_three_use_media (hU : U = 180)
  (hT : T = 115)
  (hR : R = 110)
  (hM : M = 130)
  (hT_and_M : T_and_M = 85)
  (hT_and_R : T_and_R = 75)
  (hR_and_M : R_and_M = 95)
  (h_union : U = T + R + M - T_and_R - T_and_M - R_and_M + T_and_R_and_M) :
  T_and_R_and_M = 80 :=
by
  sorry

end all_three_use_media_l198_198333


namespace triangle_side_a_l198_198587

theorem triangle_side_a (c b : ℝ) (B : ℝ) (h₁ : c = 2) (h₂ : b = 6) (h₃ : B = 120) : a = 2 :=
by sorry

end triangle_side_a_l198_198587


namespace joe_paint_usage_l198_198352

theorem joe_paint_usage :
  let total_paint := 360
  let paint_first_week := total_paint * (1 / 4)
  let remaining_paint_after_first_week := total_paint - paint_first_week
  let paint_second_week := remaining_paint_after_first_week * (1 / 7)
  paint_first_week + paint_second_week = 128.57 :=
by
  sorry

end joe_paint_usage_l198_198352


namespace smallest_area_2020th_square_l198_198608

theorem smallest_area_2020th_square (n : ℕ) :
  (∃ n : ℕ, n^2 > 2019 ∧ ∃ A : ℕ, A = n^2 - 2019 ∧ A ≠ 1) →
  (∃ A : ℕ, A = n^2 - 2019 ∧ A ≠ 1 ∧ A = 6) :=
sorry

end smallest_area_2020th_square_l198_198608


namespace y_intercept_of_line_l198_198299

theorem y_intercept_of_line (x y : ℝ) (h : 5 * x - 3 * y = 15) : (0, -5) = (0, (-5 : ℝ)) :=
by
  sorry

end y_intercept_of_line_l198_198299


namespace ways_to_make_30_cents_is_17_l198_198083

-- Define the value of each type of coin
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Define the function that counts the number of ways to make 30 cents
def count_ways_to_make_30_cents : ℕ :=
  let ways_with_1_quarter := (if 30 - quarter_value == 5 then 2 else 0)
  let ways_with_0_quarters :=
    let ways_with_2_dimes := (if 30 - 2 * dime_value == 10 then 3 else 0)
    let ways_with_1_dime := (if 30 - dime_value == 20 then 5 else 0)
    let ways_with_0_dimes := (if 30 == 30 then 7 else 0)
    ways_with_2_dimes + ways_with_1_dime + ways_with_0_dimes
  2 + ways_with_1_quarter + ways_with_0_quarters

-- Proof statement
theorem ways_to_make_30_cents_is_17 : count_ways_to_make_30_cents = 17 := sorry

end ways_to_make_30_cents_is_17_l198_198083


namespace jake_total_distance_l198_198415

noncomputable def jake_rate : ℝ := 4 -- Jake's walking rate in miles per hour
noncomputable def total_time : ℝ := 2 -- Jake's total walking time in hours
noncomputable def break_time : ℝ := 0.5 -- Jake's break time in hours

theorem jake_total_distance :
  jake_rate * (total_time - break_time) = 6 :=
by
  sorry

end jake_total_distance_l198_198415


namespace max_consecutive_sum_l198_198469

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l198_198469


namespace benches_required_l198_198140

theorem benches_required (students_base5 : ℕ := 312) (base_student_seating : ℕ := 5) (seats_per_bench : ℕ := 3) : ℕ :=
  let chairs := 3 * base_student_seating^2 + 1 * base_student_seating^1 + 2 * base_student_seating^0
  let benches := (chairs / seats_per_bench) + if (chairs % seats_per_bench > 0) then 1 else 0
  benches

example : benches_required = 28 :=
by sorry

end benches_required_l198_198140


namespace t_shirts_to_buy_l198_198724

variable (P T : ℕ)

def condition1 : Prop := 3 * P + 6 * T = 750
def condition2 : Prop := P + 12 * T = 750

theorem t_shirts_to_buy (h1 : condition1 P T) (h2 : condition2 P T) :
  400 / T = 8 :=
by
  sorry

end t_shirts_to_buy_l198_198724


namespace set_representation_l198_198628

theorem set_representation : 
  { x : ℕ | x < 5 } = {0, 1, 2, 3, 4} :=
sorry

end set_representation_l198_198628


namespace amount_due_years_l198_198447

noncomputable def years_due (PV FV : ℝ) (r : ℝ) : ℝ :=
  (Real.log (FV / PV)) / (Real.log (1 + r))

theorem amount_due_years : 
  years_due 200 242 0.10 = 2 :=
by
  sorry

end amount_due_years_l198_198447


namespace evaluation_expression_l198_198237

theorem evaluation_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 8 * y^x - 2 * x * y = 893 :=
by
  rw [h1, h2]
  -- Here we would perform the arithmetic steps to show the equality
  sorry

end evaluation_expression_l198_198237


namespace plastic_skulls_number_l198_198847

-- Define the conditions
def num_broomsticks : ℕ := 4
def num_spiderwebs : ℕ := 12
def num_pumpkins := 2 * num_spiderwebs
def num_cauldron : ℕ := 1
def budget_left_to_buy : ℕ := 20
def num_left_to_put_up : ℕ := 10
def total_decorations : ℕ := 83

-- The number of plastic skulls calculation as a function
def num_other_decorations : ℕ :=
  num_broomsticks + num_spiderwebs + num_pumpkins + num_cauldron + budget_left_to_buy + num_left_to_put_up

def num_plastic_skulls := total_decorations - num_other_decorations

-- The theorem to be proved
theorem plastic_skulls_number : num_plastic_skulls = 12 := by
  sorry

end plastic_skulls_number_l198_198847


namespace ratio_sum_l198_198639

variable (x y z : ℝ)

-- Conditions
axiom geometric_sequence : 16 * y^2 = 15 * x * z
axiom arithmetic_sequence : 2 / y = 1 / x + 1 / z

-- Theorem to prove
theorem ratio_sum : x ≠ 0 → y ≠ 0 → z ≠ 0 → 
  (16 * y^2 = 15 * x * z) → (2 / y = 1 / x + 1 / z) → (x / z + z / x = 34 / 15) :=
by
  -- proof goes here
  sorry

end ratio_sum_l198_198639


namespace perpendicular_tangent_lines_l198_198395

def f (x : ℝ) : ℝ := x^3 + 1

noncomputable def tangent_line_eqs (x₀ : ℝ) (y₀ : ℝ) : Prop :=
  (3 * x₀ - y₀ - 1 = 0) ∨ (3 * x₀ - y₀ + 3 = 0)

theorem perpendicular_tangent_lines (x₀ : ℝ) (hx₀ : x₀ = 1 ∨ x₀ = -1) :
  tangent_line_eqs x₀ (f x₀) := by
  sorry

end perpendicular_tangent_lines_l198_198395


namespace james_weekly_pistachio_cost_l198_198014

def cost_per_can : ℕ := 10
def ounces_per_can : ℕ := 5
def consumption_per_5_days : ℕ := 30
def days_per_week : ℕ := 7

theorem james_weekly_pistachio_cost : (days_per_week / 5 * consumption_per_5_days) / ounces_per_can * cost_per_can = 90 := 
by
  sorry

end james_weekly_pistachio_cost_l198_198014


namespace total_registration_methods_l198_198725

theorem total_registration_methods (n : ℕ) (h : n = 5) : (2 ^ n) = 32 :=
by
  sorry

end total_registration_methods_l198_198725


namespace volume_of_rectangular_prism_l198_198210

variables (a b c : ℝ)

theorem volume_of_rectangular_prism 
  (h1 : a * b = 12) 
  (h2 : b * c = 18) 
  (h3 : c * a = 9) 
  (h4 : (1 / a) * (1 / b) * (1 / c) = (1 / 216)) :
  a * b * c = 216 :=
sorry

end volume_of_rectangular_prism_l198_198210


namespace trains_time_to_clear_each_other_l198_198396

noncomputable def relative_speed (v1 v2 : ℝ) : ℝ :=
  v1 + v2

noncomputable def speed_to_m_s (v_kmph : ℝ) : ℝ :=
  v_kmph * 1000 / 3600

noncomputable def total_length (l1 l2 : ℝ) : ℝ :=
  l1 + l2

theorem trains_time_to_clear_each_other :
  ∀ (l1 l2 : ℝ) (v1_kmph v2_kmph : ℝ),
    l1 = 100 → l2 = 280 →
    v1_kmph = 42 → v2_kmph = 30 →
    (total_length l1 l2) / (speed_to_m_s (relative_speed v1_kmph v2_kmph)) = 19 :=
by
  intros l1 l2 v1_kmph v2_kmph h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end trains_time_to_clear_each_other_l198_198396


namespace sum_of_number_and_conjugate_l198_198104

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l198_198104


namespace vertex_of_parabola_l198_198455

theorem vertex_of_parabola (a b c : ℝ) (h k : ℝ) (x y : ℝ) :
  (∀ x, y = (1/2) * (x - 1)^2 + 2) → (h, k) = (1, 2) :=
by
  intro hy
  exact sorry

end vertex_of_parabola_l198_198455


namespace find_abc_value_l198_198846

noncomputable def given_conditions (a b c : ℝ) : Prop :=
  (a * b / (a + b) = 2) ∧ (b * c / (b + c) = 5) ∧ (c * a / (c + a) = 9)

theorem find_abc_value (a b c : ℝ) (h : given_conditions a b c) :
  a * b * c / (a * b + b * c + c * a) = 90 / 73 :=
sorry

end find_abc_value_l198_198846


namespace chocolate_chips_needed_l198_198319

-- Define the variables used in the conditions
def cups_per_recipe := 2
def number_of_recipes := 23

-- State the theorem
theorem chocolate_chips_needed : (cups_per_recipe * number_of_recipes) = 46 := 
by sorry

end chocolate_chips_needed_l198_198319


namespace remaining_balance_is_correct_l198_198594

def initial_balance : ℕ := 50
def spent_coffee : ℕ := 10
def spent_tumbler : ℕ := 30

theorem remaining_balance_is_correct : initial_balance - (spent_coffee + spent_tumbler) = 10 := by
  sorry

end remaining_balance_is_correct_l198_198594


namespace intersection_A_B_l198_198388

def A : Set ℝ := {x | x > 3}
def B : Set ℝ := {x | (x - 1) / (x - 4) < 0}
def inter : Set ℝ := {x | 3 < x ∧ x < 4}

theorem intersection_A_B : A ∩ B = inter := 
by 
  sorry

end intersection_A_B_l198_198388


namespace diane_coffee_purchase_l198_198721

theorem diane_coffee_purchase (c d : ℕ) (h1 : c + d = 7) (h2 : 90 * c + 60 * d % 100 = 0) : c = 6 :=
by
  sorry

end diane_coffee_purchase_l198_198721


namespace cole_drive_time_to_work_l198_198461

theorem cole_drive_time_to_work :
  ∀ (D : ℝ),
    (D / 80 + D / 120 = 3) → (D / 80 * 60 = 108) :=
by
  intro D h
  sorry

end cole_drive_time_to_work_l198_198461


namespace no_int_solutions_for_equation_l198_198460

theorem no_int_solutions_for_equation :
  ¬ ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 3 * y^2 = x^4 + x := 
sorry

end no_int_solutions_for_equation_l198_198460


namespace molecular_weight_is_correct_l198_198706

structure Compound :=
  (H C N Br O : ℕ)

structure AtomicWeights :=
  (H C N Br O : ℝ)

noncomputable def molecularWeight (compound : Compound) (weights : AtomicWeights) : ℝ :=
  compound.H * weights.H +
  compound.C * weights.C +
  compound.N * weights.N +
  compound.Br * weights.Br +
  compound.O * weights.O

def givenCompound : Compound :=
  { H := 2, C := 2, N := 1, Br := 1, O := 4 }

def givenWeights : AtomicWeights :=
  { H := 1.008, C := 12.011, N := 14.007, Br := 79.904, O := 15.999 }

theorem molecular_weight_is_correct : molecularWeight givenCompound givenWeights = 183.945 := by
  sorry

end molecular_weight_is_correct_l198_198706


namespace income_ratio_l198_198525

theorem income_ratio (I1 I2 E1 E2 : ℕ)
  (hI1 : I1 = 3500)
  (hE_ratio : (E1:ℚ) / E2 = 3 / 2)
  (hSavings : ∀ (x y : ℕ), x - E1 = 1400 ∧ y - E2 = 1400 → x = I1 ∧ y = I2) :
  I1 / I2 = 5 / 4 :=
by
  -- The proof steps would go here
  sorry

end income_ratio_l198_198525


namespace police_emergency_number_has_prime_divisor_gt_7_l198_198190

theorem police_emergency_number_has_prime_divisor_gt_7 (n : ℕ) (h : n % 1000 = 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n := sorry

end police_emergency_number_has_prime_divisor_gt_7_l198_198190


namespace probability_odd_divisor_of_15_factorial_l198_198961

-- Define the factorial function
def fact : ℕ → ℕ
  | 0 => 1
  | (n+1) => (n+1) * fact n

-- Probability function for choosing an odd divisor
noncomputable def probability_odd_divisor (n : ℕ) : ℚ :=
  let prime_factors := [(2, 11), (3, 6), (5, 3), (7, 2), (11, 1), (13, 1)]
  let total_factors := prime_factors.foldr (λ p acc => (p.2 + 1) * acc) 1
  let odd_factors := ((prime_factors.filter (λ p => p.1 ≠ 2)).foldr (λ p acc => (p.2 + 1) * acc) 1)
  (odd_factors : ℚ) / (total_factors : ℚ)

-- Statement to prove the probability of an odd divisor
theorem probability_odd_divisor_of_15_factorial :
  probability_odd_divisor 15 = 1 / 12 :=
by
  -- Proof goes here, which is omitted as per the instructions
  sorry

end probability_odd_divisor_of_15_factorial_l198_198961


namespace Monica_saved_per_week_l198_198317

theorem Monica_saved_per_week(amount_per_cycle : ℕ) (weeks_per_cycle : ℕ) (num_cycles : ℕ) (total_saved : ℕ) :
  num_cycles = 5 →
  weeks_per_cycle = 60 →
  (amount_per_cycle * num_cycles) = total_saved →
  total_saved = 4500 →
  total_saved / (weeks_per_cycle * num_cycles) = 75 := 
by
  intros
  sorry

end Monica_saved_per_week_l198_198317


namespace eval_at_5_l198_198406

def g (x : ℝ) : ℝ := 3 * x^4 - 8 * x^3 + 15 * x^2 - 10 * x - 75

theorem eval_at_5 : g 5 = 1125 := by
  sorry

end eval_at_5_l198_198406


namespace solve_for_y_l198_198859

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 6 * y)) : y = 100 / 3 :=
by
  sorry

end solve_for_y_l198_198859


namespace time_to_paint_one_room_l198_198188

variables (rooms_total rooms_painted : ℕ) (hours_to_paint_remaining : ℕ)

-- The conditions
def painter_conditions : Prop :=
  rooms_total = 10 ∧ rooms_painted = 8 ∧ hours_to_paint_remaining = 16

-- The goal is to find out the hours to paint one room
theorem time_to_paint_one_room (h : painter_conditions rooms_total rooms_painted hours_to_paint_remaining) : 
  let rooms_remaining := rooms_total - rooms_painted
  let hours_per_room := hours_to_paint_remaining / rooms_remaining
  hours_per_room = 8 :=
by sorry

end time_to_paint_one_room_l198_198188


namespace circle_condition_l198_198744

-- Define the given equation
def equation (m x y : ℝ) : Prop := x^2 + y^2 + 4 * m * x - 2 * y + 5 * m = 0

-- Define the condition for the equation to represent a circle
def represents_circle (m x y : ℝ) : Prop :=
  (x + 2 * m)^2 + (y - 1)^2 = 4 * m^2 - 5 * m + 1 ∧ 4 * m^2 - 5 * m + 1 > 0

-- The main theorem to be proven
theorem circle_condition (m : ℝ) : represents_circle m x y → (m < 1/4 ∨ m > 1) := 
sorry

end circle_condition_l198_198744


namespace hyperbola_focus_exists_l198_198223

-- Define the basic premises of the problem
def is_hyperbola (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 - 8 * x - 24 * y + 4 = 0

-- Define a condition for the focusing property of the hyperbola.
def is_focus (x y : ℝ) : Prop :=
  (x = -2) ∧ (y = 4 + (10 * Real.sqrt 3 / 3))

-- The theorem to be proved
theorem hyperbola_focus_exists : ∃ x y : ℝ, is_hyperbola x y ∧ is_focus x y :=
by
  -- Proof to be filled in
  sorry

end hyperbola_focus_exists_l198_198223


namespace velocity_equal_distance_l198_198227

theorem velocity_equal_distance (v t : ℝ) (h : v * t = t) (ht : t ≠ 0) : v = 1 :=
by sorry

end velocity_equal_distance_l198_198227


namespace expression_equals_two_l198_198514

noncomputable def math_expression : ℝ :=
  27^(1/3) + Real.log 4 + 2 * Real.log 5 - Real.exp (Real.log 3)

theorem expression_equals_two : math_expression = 2 := by
  sorry

end expression_equals_two_l198_198514


namespace cone_radius_l198_198732

theorem cone_radius (r l : ℝ) 
  (surface_area_eq : π * r^2 + π * r * l = 12 * π)
  (net_is_semicircle : π * l = 2 * π * r) : 
  r = 2 :=
by
  sorry

end cone_radius_l198_198732


namespace polygon_vertices_product_at_least_2014_l198_198416

theorem polygon_vertices_product_at_least_2014 :
  ∀ (vertices : Fin 90 → ℕ), 
    (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 90) → 
    ∃ i, (vertices i) * (vertices ((i + 1) % 90)) ≥ 2014 :=
sorry

end polygon_vertices_product_at_least_2014_l198_198416


namespace quadratic_complete_square_l198_198656

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 - 7 * x + 6) = (x - 7 / 2) ^ 2 - 25 / 4 :=
by
  sorry

end quadratic_complete_square_l198_198656


namespace isosceles_trapezoid_rotation_produces_frustum_l198_198696

-- Definitions based purely on conditions
structure IsoscelesTrapezoid :=
(a b c d : ℝ) -- sides
(ha : a = c) -- isosceles property
(hb : b ≠ d) -- non-parallel sides

def rotateAroundSymmetryAxis (shape : IsoscelesTrapezoid) : Type :=
-- We need to define what the rotation of the trapezoid produces
sorry

theorem isosceles_trapezoid_rotation_produces_frustum (shape : IsoscelesTrapezoid) :
  rotateAroundSymmetryAxis shape = Frustum :=
sorry

end isosceles_trapezoid_rotation_produces_frustum_l198_198696


namespace trapezoid_area_l198_198939

theorem trapezoid_area (h_base : ℕ) (sum_bases : ℕ) (height : ℕ) (hsum : sum_bases = 36) (hheight : height = 15) :
    (sum_bases * height) / 2 = 270 := by
  sorry

end trapezoid_area_l198_198939


namespace geometric_seq_tenth_term_l198_198654

theorem geometric_seq_tenth_term :
  let a := 12
  let r := (1 / 2 : ℝ)
  (a * r^9) = (3 / 128 : ℝ) :=
by
  let a := 12
  let r := (1 / 2 : ℝ)
  show a * r^9 = 3 / 128
  sorry

end geometric_seq_tenth_term_l198_198654


namespace binomial_square_constant_l198_198891

theorem binomial_square_constant :
  ∃ c : ℝ, (∀ x : ℝ, 9*x^2 - 21*x + c = (3*x + -3.5)^2) → c = 12.25 :=
by
  sorry

end binomial_square_constant_l198_198891


namespace same_grade_percentage_l198_198951

theorem same_grade_percentage (total_students: ℕ)
  (a_students: ℕ) (b_students: ℕ) (c_students: ℕ) (d_students: ℕ)
  (total: total_students = 30)
  (a: a_students = 2) (b: b_students = 4) (c: c_students = 5) (d: d_students = 1)
  : (a_students + b_students + c_students + d_students) * 100 / total_students = 40 := by
  sorry

end same_grade_percentage_l198_198951


namespace relationship_between_a_and_b_l198_198496

noncomputable section
open Classical

theorem relationship_between_a_and_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b :=
sorry

end relationship_between_a_and_b_l198_198496


namespace system1_solution_system2_solution_l198_198963

theorem system1_solution :
  ∃ (x y : ℝ), 3 * x - 2 * y = -1 ∧ 2 * x + 3 * y = 8 ∧ x = 1 ∧ y = 2 :=
by {
  -- Proof skipped
  sorry
}

theorem system2_solution :
  ∃ (x y : ℝ), 2 * x + y = 1 ∧ 2 * x - y = 7 ∧ x = 2 ∧ y = -3 :=
by {
  -- Proof skipped
  sorry
}

end system1_solution_system2_solution_l198_198963


namespace part1_part2_l198_198438

open Set

/-- Define sets A and B as per given conditions --/
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

/-- Part 1: Prove the intersection and union with complements --/
theorem part1 :
  A ∩ B = {x | 3 ≤ x ∧ x < 6} ∧ (compl B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
by {
  sorry
}

/-- Part 2: Given C ⊆ B, prove the constraints on a --/
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem part2 (a : ℝ) (h : C a ⊆ B) : 2 ≤ a ∧ a ≤ 8 :=
by {
  sorry
}

end part1_part2_l198_198438


namespace molecular_weight_N2O5_l198_198723

theorem molecular_weight_N2O5 :
  let atomic_weight_N := 14.01
  let atomic_weight_O := 16.00
  let molecular_weight_N2O5 := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  molecular_weight_N2O5 = 108.02 := 
by
  sorry

end molecular_weight_N2O5_l198_198723


namespace john_salary_april_l198_198392

theorem john_salary_april 
  (initial_salary : ℤ)
  (raise_percentage : ℤ)
  (cut_percentage : ℤ)
  (bonus : ℤ)
  (february_salary : ℤ)
  (march_salary : ℤ)
  : initial_salary = 3000 →
    raise_percentage = 10 →
    cut_percentage = 15 →
    bonus = 500 →
    february_salary = initial_salary + (initial_salary * raise_percentage / 100) →
    march_salary = february_salary - (february_salary * cut_percentage / 100) →
    march_salary + bonus = 3305 :=
by
  intros
  sorry

end john_salary_april_l198_198392


namespace mass_percentage_of_Cl_in_bleach_l198_198528

-- Definitions based on conditions
def Na_molar_mass : Float := 22.99
def Cl_molar_mass : Float := 35.45
def O_molar_mass : Float := 16.00

def NaClO_molar_mass : Float := Na_molar_mass + Cl_molar_mass + O_molar_mass

def mass_NaClO (mass_na: Float) (mass_cl: Float) (mass_o: Float) : Float :=
  mass_na + mass_cl + mass_o

def mass_of_NaClO : Float := 100.0

def mass_of_Cl_in_NaClO (mass_of_NaClO: Float) : Float :=
  (Cl_molar_mass / NaClO_molar_mass) * mass_of_NaClO

-- Statement to prove
theorem mass_percentage_of_Cl_in_bleach :
  let mass_Cl := mass_of_Cl_in_NaClO mass_of_NaClO
  (mass_Cl / mass_of_NaClO) * 100 = 47.61 :=
by 
  -- Skip the proof
  sorry

end mass_percentage_of_Cl_in_bleach_l198_198528


namespace cosine_of_angle_l198_198679

theorem cosine_of_angle (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = Real.sqrt 3 / 2) : 
  Real.cos (Real.pi / 3 - α) = Real.sqrt 3 / 2 := 
by
  sorry

end cosine_of_angle_l198_198679


namespace max_trees_cut_l198_198012

theorem max_trees_cut (n : ℕ) (h : n = 2001) :
  (∃ m : ℕ, m = n * n ∧ ∀ (x y : ℕ), x < n ∧ y < n → (x % 2 = 0 ∧ y % 2 = 0 → m = 1001001)) := sorry

end max_trees_cut_l198_198012


namespace total_stamps_l198_198578

-- Definitions based on conditions
def kylies_stamps : ℕ := 34
def nellys_stamps : ℕ := kylies_stamps + 44

-- Statement of the proof problem
theorem total_stamps : kylies_stamps + nellys_stamps = 112 :=
by
  -- Proof goes here
  sorry

end total_stamps_l198_198578


namespace g_value_at_50_l198_198852

noncomputable def g : ℝ → ℝ := sorry

theorem g_value_at_50 (g : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, 0 < x → 0 < y → x * g y + y * g x = g (x * y)) :
  g 50 = 0 :=
sorry

end g_value_at_50_l198_198852


namespace common_root_equation_l198_198119

theorem common_root_equation {m : ℝ} (x : ℝ) (h1 : m * x - 1000 = 1001) (h2 : 1001 * x = m - 1000 * x) : m = 2001 ∨ m = -2001 :=
by
  -- Skipping the proof details
  sorry

end common_root_equation_l198_198119


namespace find_X_l198_198692

theorem find_X :
  (15.2 * 0.25 - 48.51 / 14.7) / X = ((13 / 44 - 2 / 11 - 5 / 66) / (5 / 2) * (6 / 5)) / (3.2 + 0.8 * (5.5 - 3.25)) ->
  X = 137.5 :=
by
  intro h
  sorry

end find_X_l198_198692


namespace lucy_fish_moved_l198_198304

theorem lucy_fish_moved (original_count moved_count remaining_count : ℝ)
  (h1: original_count = 212.0)
  (h2: remaining_count = 144.0) :
  moved_count = original_count - remaining_count :=
by sorry

end lucy_fish_moved_l198_198304


namespace final_alcohol_percentage_l198_198449

noncomputable def initial_volume : ℝ := 6
noncomputable def initial_percentage : ℝ := 0.25
noncomputable def added_alcohol : ℝ := 3
noncomputable def final_volume : ℝ := initial_volume + added_alcohol
noncomputable def final_percentage : ℝ := (initial_volume * initial_percentage + added_alcohol) / final_volume * 100

theorem final_alcohol_percentage :
  final_percentage = 50 := by
  sorry

end final_alcohol_percentage_l198_198449


namespace final_hair_length_l198_198399

-- Define the initial conditions and the expected final result.
def initial_hair_length : ℕ := 14
def hair_growth (x : ℕ) : ℕ := x
def hair_cut : ℕ := 20

-- Prove that the final hair length is x - 6.
theorem final_hair_length (x : ℕ) : initial_hair_length + hair_growth x - hair_cut = x - 6 :=
by
  sorry

end final_hair_length_l198_198399


namespace infinite_points_on_line_with_positive_rational_coordinates_l198_198367

theorem infinite_points_on_line_with_positive_rational_coordinates :
  ∃ (S : Set (ℚ × ℚ)), (∀ p ∈ S, p.1 + p.2 = 4 ∧ 0 < p.1 ∧ 0 < p.2) ∧ S.Infinite :=
sorry

end infinite_points_on_line_with_positive_rational_coordinates_l198_198367


namespace time_to_pass_jogger_l198_198383

noncomputable def jogger_speed_kmh : ℕ := 9
noncomputable def jogger_speed_ms : ℝ := jogger_speed_kmh * 1000 / 3600
noncomputable def train_length : ℕ := 130
noncomputable def jogger_ahead_distance : ℕ := 240
noncomputable def train_speed_kmh : ℕ := 45
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
noncomputable def relative_speed : ℝ := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_to_cover : ℕ := jogger_ahead_distance + train_length
noncomputable def time_taken_to_pass : ℝ := total_distance_to_cover / relative_speed

theorem time_to_pass_jogger : time_taken_to_pass = 37 := sorry

end time_to_pass_jogger_l198_198383


namespace martha_bedroom_size_l198_198463

theorem martha_bedroom_size (x jenny_size total_size : ℤ) (h₁ : jenny_size = x + 60) (h₂ : total_size = x + jenny_size) (h_total : total_size = 300) : x = 120 :=
by
  -- Adding conditions and the ultimate goal
  sorry


end martha_bedroom_size_l198_198463


namespace arithmetic_expression_eval_l198_198834

theorem arithmetic_expression_eval : 8 / 4 - 3 - 9 + 3 * 9 = 17 :=
by
  sorry

end arithmetic_expression_eval_l198_198834


namespace yards_dyed_green_calc_l198_198479

-- Given conditions: total yards dyed and yards dyed pink
def total_yards_dyed : ℕ := 111421
def yards_dyed_pink : ℕ := 49500

-- Goal: Prove the number of yards dyed green
theorem yards_dyed_green_calc : total_yards_dyed - yards_dyed_pink = 61921 :=
by 
-- sorry means that the proof is skipped.
sorry

end yards_dyed_green_calc_l198_198479


namespace misha_second_attempt_points_l198_198214

/--
Misha made a homemade dartboard at his summer cottage. The round board is 
divided into several sectors by circles, and you can throw darts at it. 
Points are awarded based on the sector hit.

Misha threw 8 darts three times. In his second attempt, he scored twice 
as many points as in his first attempt, and in his third attempt, he scored 
1.5 times more points than in his second attempt. How many points did he 
score in his second attempt?
-/
theorem misha_second_attempt_points:
  ∀ (x : ℕ), 
  (x ≥ 24) →
  (2 * x ≥ 48) →
  (3 * x = 72) →
  (2 * x = 48) :=
by
  intros x h1 h2 h3
  sorry

end misha_second_attempt_points_l198_198214


namespace white_pairs_coincide_l198_198746

def triangles_in_each_half (red blue white: Nat) : Prop :=
  red = 5 ∧ blue = 6 ∧ white = 9

def folding_over_centerline (r_pairs b_pairs rw_pairs bw_pairs: Nat) : Prop :=
  r_pairs = 3 ∧ b_pairs = 2 ∧ rw_pairs = 3 ∧ bw_pairs = 1

theorem white_pairs_coincide
    (red_triangles blue_triangles white_triangles : Nat)
    (r_pairs b_pairs rw_pairs bw_pairs : Nat) :
    triangles_in_each_half red_triangles blue_triangles white_triangles →
    folding_over_centerline r_pairs b_pairs rw_pairs bw_pairs →
    ∃ coinciding_white_pairs, coinciding_white_pairs = 5 :=
by
  intros half_cond fold_cond
  sorry

end white_pairs_coincide_l198_198746


namespace sum_of_ages_l198_198378

theorem sum_of_ages (rose_age mother_age : ℕ) (rose_age_eq : rose_age = 25) (mother_age_eq : mother_age = 75) : 
  rose_age + mother_age = 100 := 
by
  sorry

end sum_of_ages_l198_198378


namespace part1_part2_l198_198414

theorem part1 (x : ℝ) (m : ℝ) :
  (∃ x, x^2 - 2*(m-1)*x + m^2 = 0) → (m ≤ 1 / 2) := 
  sorry

theorem part2 (x1 x2 : ℝ) (m : ℝ) :
  (x1^2 - 2*(m-1)*x1 + m^2 = 0) ∧ (x2^2 - 2*(m-1)*x2 + m^2 = 0) ∧ 
  (x1^2 + x2^2 = 8 - 3*x1*x2) → (m = -2 / 5) := 
  sorry

end part1_part2_l198_198414


namespace main_theorem_l198_198926

variable (a : ℝ)

def M : Set ℝ := {x | x > 1 / 2 ∧ x < 1} ∪ {x | x > 1}
def N : Set ℝ := {x | x > 0 ∧ x ≤ 1 / 2}

theorem main_theorem : M ∩ N = ∅ :=
by
  sorry

end main_theorem_l198_198926


namespace max_sum_of_factors_of_1764_l198_198900

theorem max_sum_of_factors_of_1764 :
  ∃ (a b : ℕ), a * b = 1764 ∧ a + b = 884 :=
by
  sorry

end max_sum_of_factors_of_1764_l198_198900


namespace alyssa_total_games_l198_198485

def calc_total_games (games_this_year games_last_year games_next_year : ℕ) : ℕ :=
  games_this_year + games_last_year + games_next_year

theorem alyssa_total_games :
  calc_total_games 11 13 15 = 39 :=
by
  -- Proof goes here
  sorry

end alyssa_total_games_l198_198485


namespace parametric_equations_l198_198551

variables (t : ℝ)
def x_velocity : ℝ := 9
def y_velocity : ℝ := 12
def init_x : ℝ := 1
def init_y : ℝ := 1

theorem parametric_equations :
  (x = init_x + x_velocity * t) ∧ (y = init_y + y_velocity * t) :=
sorry

end parametric_equations_l198_198551


namespace problem_statement_l198_198873

def has_solutions (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - m * x - 1 = 0

def p : Prop := ∀ m : ℝ, has_solutions m

def q : Prop := ∃ x_0 : ℕ, x_0^2 - 2 * x_0 - 1 ≤ 0

theorem problem_statement : ¬ (p ∧ ¬ q) := 
sorry

end problem_statement_l198_198873


namespace part1_part2_l198_198968

-- Definition of Set A
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 6 }

-- Definition of Set B
def B : Set ℝ := { x | x ≥ 3 }

-- The Complement of the Intersection of A and B
def C_R (S : Set ℝ) : Set ℝ := { x | ¬ (x ∈ S) }

-- Set C
def C (a : ℝ) : Set ℝ := { x | x ≤ a }

-- Lean statement for part 1
theorem part1 : C_R (A ∩ B) = { x | x < 3 ∨ x > 6 } :=
by sorry

-- Lean statement for part 2
theorem part2 (a : ℝ) (hA_C : A ⊆ C a) : a ≥ 6 :=
by sorry

end part1_part2_l198_198968


namespace joan_mortgage_payback_months_l198_198701

-- Define the conditions and statement

def first_payment : ℕ := 100
def total_amount : ℕ := 2952400

theorem joan_mortgage_payback_months :
  ∃ n : ℕ, 100 * (3^n - 1) / (3 - 1) = 2952400 ∧ n = 10 :=
by
  sorry

end joan_mortgage_payback_months_l198_198701


namespace spherical_to_cartesian_l198_198236

theorem spherical_to_cartesian 
  (ρ θ φ : ℝ)
  (hρ : ρ = 3) 
  (hθ : θ = 7 * Real.pi / 12) 
  (hφ : φ = Real.pi / 4) :
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) = 
  (3 * Real.sqrt 2 / 2 * Real.cos (7 * Real.pi / 12), 
   3 * Real.sqrt 2 / 2 * Real.sin (7 * Real.pi / 12), 
   3 * Real.sqrt 2 / 2) :=
by
  sorry

end spherical_to_cartesian_l198_198236


namespace polynomial_coefficient_a5_l198_198177

theorem polynomial_coefficient_a5 : 
  (∃ (a0 a1 a2 a3 a4 a5 a6 : ℝ), 
    (∀ (x : ℝ), ((2 * x - 1)^5 * (x + 2) = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6)) ∧ 
    a5 = 176) := sorry

end polynomial_coefficient_a5_l198_198177


namespace necessary_and_sufficient_condition_l198_198257

theorem necessary_and_sufficient_condition (a b : ℝ) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := 
by 
  sorry

end necessary_and_sufficient_condition_l198_198257


namespace apples_to_eat_raw_l198_198734

/-- Proof of the number of apples left to eat raw given the conditions -/
theorem apples_to_eat_raw 
  (total_apples : ℕ)
  (pct_wormy : ℕ)
  (pct_moldy : ℕ)
  (wormy_apples_offset : ℕ)
  (wormy_apples bruised_apples moldy_apples apples_left : ℕ) 
  (h1 : total_apples = 120)
  (h2 : pct_wormy = 20)
  (h3 : pct_moldy = 30)
  (h4 : wormy_apples = pct_wormy * total_apples / 100)
  (h5 : moldy_apples = pct_moldy * total_apples / 100)
  (h6 : bruised_apples = wormy_apples + wormy_apples_offset)
  (h7 : wormy_apples_offset = 9)
  (h8 : apples_left = total_apples - (wormy_apples + moldy_apples + bruised_apples))
  : apples_left = 27 :=
sorry

end apples_to_eat_raw_l198_198734


namespace carol_initial_peanuts_l198_198256

theorem carol_initial_peanuts (p_initial p_additional p_total : Nat) (h1 : p_additional = 5) (h2 : p_total = 7) (h3 : p_initial + p_additional = p_total) : p_initial = 2 :=
by
  sorry

end carol_initial_peanuts_l198_198256


namespace volume_of_sand_pile_l198_198688

theorem volume_of_sand_pile (d h : ℝ) (π : ℝ) (r : ℝ) (vol : ℝ) :
  d = 8 →
  h = (3 / 4) * d →
  r = d / 2 →
  vol = (1 / 3) * π * r^2 * h →
  vol = 32 * π :=
by
  intros hd hh hr hv
  subst hd
  subst hh
  subst hr
  subst hv
  sorry

end volume_of_sand_pile_l198_198688


namespace regular_polygon_perimeter_l198_198419

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l198_198419


namespace positive_value_of_m_l198_198047

variable {m : ℝ}

theorem positive_value_of_m (h : ∃ x : ℝ, (3 * x^2 + m * x + 36) = 0 ∧ (∀ y : ℝ, (3 * y^2 + m * y + 36) = 0 → y = x)) :
  m = 12 * Real.sqrt 3 :=
sorry

end positive_value_of_m_l198_198047


namespace brother_combined_age_l198_198992

-- Define the ages of the brothers as integers
variable (x y : ℕ)

-- Define the condition given in the problem
def combined_age_six_years_ago : Prop := (x - 6) + (y - 6) = 100

-- State the theorem to prove the current combined age
theorem brother_combined_age (h : combined_age_six_years_ago x y): x + y = 112 :=
  sorry

end brother_combined_age_l198_198992


namespace AM_GM_Ineq_l198_198538

theorem AM_GM_Ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by
  sorry

end AM_GM_Ineq_l198_198538


namespace tetrahedron_edge_length_l198_198011

-- Define the problem as a Lean theorem statement
theorem tetrahedron_edge_length (r : ℝ) (a : ℝ) (h : r = 1) :
  a = 2 * Real.sqrt 2 :=
sorry

end tetrahedron_edge_length_l198_198011


namespace smallest_number_diminished_by_10_divisible_l198_198311

theorem smallest_number_diminished_by_10_divisible :
  ∃ (x : ℕ), (x - 10) % 24 = 0 ∧ x = 34 :=
by
  sorry

end smallest_number_diminished_by_10_divisible_l198_198311


namespace base_conversion_problem_l198_198788

variable (A C : ℕ)
variable (h1 : 0 ≤ A ∧ A < 8)
variable (h2 : 0 ≤ C ∧ C < 5)

theorem base_conversion_problem (h : 8 * A + C = 5 * C + A) : 8 * A + C = 39 := 
sorry

end base_conversion_problem_l198_198788


namespace age_double_condition_l198_198938

theorem age_double_condition (S M X : ℕ) (h1 : S = 44) (h2 : M = S + 46) (h3 : M + X = 2 * (S + X)) : X = 2 :=
by
  sorry

end age_double_condition_l198_198938


namespace problem_solution_l198_198665

theorem problem_solution (x1 x2 x3 : ℝ) (h1: x1 < x2) (h2: x2 < x3)
(h3 : 10 * x1^3 - 201 * x1^2 + 3 = 0)
(h4 : 10 * x2^3 - 201 * x2^2 + 3 = 0)
(h5 : 10 * x3^3 - 201 * x3^2 + 3 = 0) :
x2 * (x1 + x3) = 398 :=
sorry

end problem_solution_l198_198665


namespace sequence_product_is_128_l198_198729

-- Define the sequence of fractions
def fractional_sequence (n : ℕ) : Rat :=
  if n % 2 = 0 then 1 / (2 : ℕ) ^ ((n + 2) / 2)
  else (2 : ℕ) ^ ((n + 1) / 2)

-- The target theorem: prove the product of the sequence results in 128
theorem sequence_product_is_128 : 
  (List.prod (List.map fractional_sequence [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])) = 128 := 
by
  sorry

end sequence_product_is_128_l198_198729


namespace total_consultation_time_l198_198024

-- Define the times in which each chief finishes a pipe
def chief1_time := 10
def chief2_time := 30
def chief3_time := 60

theorem total_consultation_time : 
  ∃ (t : ℕ), (∃ x, ((x / chief1_time) + (x / chief2_time) + (x / chief3_time) = 1) ∧ t = 3 * x) ∧ t = 20 :=
sorry

end total_consultation_time_l198_198024


namespace zookeeper_fish_excess_l198_198324

theorem zookeeper_fish_excess :
  let emperor_ratio := 3
  let adelie_ratio := 5
  let total_penguins := 48
  let total_ratio := emperor_ratio + adelie_ratio
  let emperor_penguins := (emperor_ratio / total_ratio) * total_penguins
  let adelie_penguins := (adelie_ratio / total_ratio) * total_penguins
  let emperor_fish_needed := emperor_penguins * 1.5
  let adelie_fish_needed := adelie_penguins * 2
  let total_fish_needed := emperor_fish_needed + adelie_fish_needed
  let fish_zookeeper_has := total_penguins * 2.5
  (fish_zookeeper_has - total_fish_needed = 33) :=
  
by {
  sorry
}

end zookeeper_fish_excess_l198_198324


namespace hollow_iron_ball_diameter_l198_198193

theorem hollow_iron_ball_diameter (R r : ℝ) (s : ℝ) (thickness : ℝ) 
  (h1 : thickness = 1) (h2 : s = 7.5) 
  (h3 : R - r = thickness) 
  (h4 : 4 / 3 * π * R^3 = 4 / 3 * π * s * (R^3 - r^3)) : 
  2 * R = 44.44 := 
sorry

end hollow_iron_ball_diameter_l198_198193


namespace Karsyn_payment_l198_198149

def percentage : ℝ := 20
def initial_price : ℝ := 600

theorem Karsyn_payment : (percentage / 100) * initial_price = 120 :=
by 
  sorry

end Karsyn_payment_l198_198149


namespace vityas_miscalculation_l198_198142

/-- Vitya's miscalculated percentages problem -/
theorem vityas_miscalculation :
  ∀ (N : ℕ)
  (acute obtuse nonexistent right depends_geometry : ℕ)
  (H_acute : acute = 5)
  (H_obtuse : obtuse = 5)
  (H_nonexistent : nonexistent = 5)
  (H_right : right = 50)
  (H_total : acute + obtuse + nonexistent + right + depends_geometry = 100),
  depends_geometry = 110 :=
by
  intros
  sorry

end vityas_miscalculation_l198_198142


namespace pie_chart_shows_percentage_l198_198418

-- Define the different types of graphs
inductive GraphType
| PieChart
| BarGraph
| LineGraph
| Histogram

-- Define conditions of the problem
def shows_percentage_of_whole (g : GraphType) : Prop :=
  g = GraphType.PieChart

def displays_with_rectangular_bars (g : GraphType) : Prop :=
  g = GraphType.BarGraph

def shows_trends (g : GraphType) : Prop :=
  g = GraphType.LineGraph

def shows_frequency_distribution (g : GraphType) : Prop :=
  g = GraphType.Histogram

-- We need to prove that a pie chart satisfies the condition of showing percentages of parts in a whole
theorem pie_chart_shows_percentage : shows_percentage_of_whole GraphType.PieChart :=
  by
    -- Proof is skipped
    sorry

end pie_chart_shows_percentage_l198_198418


namespace airport_distance_l198_198208

theorem airport_distance (d t : ℝ) (h1 : d = 45 * (t + 0.75))
                         (h2 : d - 45 = 65 * (t - 1.25)) :
  d = 241.875 :=
by
  sorry

end airport_distance_l198_198208


namespace prove_smallest_geometric_third_term_value_l198_198143

noncomputable def smallest_value_geometric_third_term : ℝ :=
  let d_1 := -5 + 10 * Real.sqrt 2
  let d_2 := -5 - 10 * Real.sqrt 2
  let g3_1 := 39 + 2 * d_1
  let g3_2 := 39 + 2 * d_2
  min g3_1 g3_2

theorem prove_smallest_geometric_third_term_value :
  smallest_value_geometric_third_term = 29 - 20 * Real.sqrt 2 := by sorry

end prove_smallest_geometric_third_term_value_l198_198143


namespace post_spacing_change_l198_198428

theorem post_spacing_change :
  ∀ (posts : ℕ → ℝ) (constant_spacing : ℝ), 
  (∀ n, 1 ≤ n ∧ n < 16 → posts (n + 1) - posts n = constant_spacing) →
  posts 16 - posts 1 = 48 → 
  posts 28 - posts 16 = 36 →
  ∃ (k : ℕ), 16 < k ∧ k ≤ 28 ∧ posts (k + 1) - posts k ≠ constant_spacing ∧ posts (k + 1) - posts k = 2.9 ∧ k = 20 := 
  sorry

end post_spacing_change_l198_198428


namespace investment_time_ratio_l198_198630

theorem investment_time_ratio (x t : ℕ) (h_inv : 7 * x = t * 5) (h_prof_ratio : 7 / 10 = 70 / (5 * t)) : 
  t = 20 := sorry

end investment_time_ratio_l198_198630


namespace largest_polygon_is_E_l198_198511

def area (num_unit_squares num_right_triangles num_half_squares: ℕ) : ℚ :=
  num_unit_squares + num_right_triangles * 0.5 + num_half_squares * 0.25

def polygon_A_area := area 3 2 0
def polygon_B_area := area 4 1 0
def polygon_C_area := area 2 4 2
def polygon_D_area := area 5 0 0
def polygon_E_area := area 3 3 4

theorem largest_polygon_is_E :
  polygon_E_area > polygon_A_area ∧ 
  polygon_E_area > polygon_B_area ∧ 
  polygon_E_area > polygon_C_area ∧ 
  polygon_E_area > polygon_D_area :=
by
  sorry

end largest_polygon_is_E_l198_198511


namespace proof_problem_l198_198244

open Real

noncomputable def set_A : Set ℝ :=
  {x | x = tan (-19 * π / 6) ∨ x = sin (-19 * π / 6)}

noncomputable def set_B : Set ℝ :=
  {m | 0 <= m ∧ m <= 4}

noncomputable def set_C (a : ℝ) : Set ℝ :=
  {x | a + 1 < x ∧ x < 2 * a}

theorem proof_problem (a : ℝ) :
  set_A = {-sqrt 3 / 3, -1 / 2} ∧
  set_B = {m | 0 <= m ∧ m <= 4} ∧
  (set_A ∪ set_B) = {-sqrt 3 / 3, -1 / 2, 0, 4} →
  (∀ a, set_C a ⊆ (set_A ∪ set_B) → 1 < a ∧ a < 2) :=
sorry

end proof_problem_l198_198244


namespace largest_triangle_angle_l198_198685

theorem largest_triangle_angle (h_ratio : ∃ (a b c : ℕ), a / b = 3 / 4 ∧ b / c = 4 / 9) 
  (h_external_angle : ∃ (θ1 θ2 θ3 θ4 : ℝ), θ1 = 3 * x ∧ θ2 = 4 * x ∧ θ3 = 9 * x ∧ θ4 = 3 * x ∧ θ1 + θ2 + θ3 = 180) :
  ∃ (θ3 : ℝ), θ3 = 101.25 := by
  sorry

end largest_triangle_angle_l198_198685


namespace M_plus_2N_equals_330_l198_198221

theorem M_plus_2N_equals_330 (M N : ℕ) :
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + 2 * N = 330 := by
  sorry

end M_plus_2N_equals_330_l198_198221


namespace integer_solutions_yk_eq_x2_plus_x_l198_198758

-- Define the problem in Lean
theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (hk : k > 1) :
  ∀ (x y : ℤ), y^k = x^2 + x → (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
by
  sorry

end integer_solutions_yk_eq_x2_plus_x_l198_198758


namespace distance_left_to_drive_l198_198146

theorem distance_left_to_drive (total_distance : ℕ) (distance_driven : ℕ) 
  (h1 : total_distance = 78) (h2 : distance_driven = 32) : 
  total_distance - distance_driven = 46 := by
  sorry

end distance_left_to_drive_l198_198146


namespace prop_A_l198_198007

theorem prop_A (x : ℝ) (h : x > 1) : (x + (1 / (x - 1)) >= 3) :=
sorry

end prop_A_l198_198007


namespace surface_area_of_T_is_630_l198_198716

noncomputable def s : ℕ := 582
noncomputable def t : ℕ := 42
noncomputable def u : ℕ := 6

theorem surface_area_of_T_is_630 : s + t + u = 630 :=
by
  sorry

end surface_area_of_T_is_630_l198_198716


namespace total_tickets_l198_198219

theorem total_tickets (n_friends : ℕ) (tickets_per_friend : ℕ) (h1 : n_friends = 6) (h2 : tickets_per_friend = 39) : n_friends * tickets_per_friend = 234 :=
by
  -- Place for proof, to be constructed
  sorry

end total_tickets_l198_198219


namespace square_of_other_leg_l198_198377

variable {R : Type} [CommRing R]

theorem square_of_other_leg (a b c : R) (h1 : a^2 + b^2 = c^2) (h2 : c = a + 2) : b^2 = 4 * a + 4 :=
by
  sorry

end square_of_other_leg_l198_198377


namespace multiple_of_9_l198_198541

theorem multiple_of_9 (x : ℕ) (hx1 : ∃ k : ℕ, x = 9 * k) (hx2 : x^2 > 80) (hx3 : x < 30) : x = 9 ∨ x = 18 ∨ x = 27 :=
sorry

end multiple_of_9_l198_198541


namespace weight_difference_l198_198943

variables (W_A W_B W_C W_D W_E : ℝ)

def condition1 : Prop := (W_A + W_B + W_C) / 3 = 84
def condition2 : Prop := (W_A + W_B + W_C + W_D) / 4 = 80
def condition3 : Prop := (W_B + W_C + W_D + W_E) / 4 = 79
def condition4 : Prop := W_A = 80

theorem weight_difference (h1 : condition1 W_A W_B W_C) 
                          (h2 : condition2 W_A W_B W_C W_D) 
                          (h3 : condition3 W_B W_C W_D W_E) 
                          (h4 : condition4 W_A) : 
                          W_E - W_D = 8 :=
by
  sorry

end weight_difference_l198_198943


namespace marys_final_amount_l198_198405

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

def final_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P + simple_interest P r t

theorem marys_final_amount 
  (P : ℝ := 200)
  (A_after_2_years : ℝ := 260)
  (t1 : ℝ := 2)
  (t2 : ℝ := 6)
  (r : ℝ := (A_after_2_years - P) / (P * t1)) :
  final_amount P r t2 = 380 := 
by
  sorry

end marys_final_amount_l198_198405


namespace quadratic_sequence_exists_l198_198302

theorem quadratic_sequence_exists (b c : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), 
  a 0 = b ∧ 
  a n = c ∧ 
  ∀ i, 1 ≤ i → i ≤ n → |a i - a (i - 1)| = i^2 :=
sorry

end quadratic_sequence_exists_l198_198302


namespace Flora_initial_daily_milk_l198_198402

def total_gallons : ℕ := 105
def total_weeks : ℕ := 3
def days_per_week : ℕ := 7
def total_days : ℕ := total_weeks * days_per_week
def extra_gallons_daily : ℕ := 2

theorem Flora_initial_daily_milk : 
  (total_gallons / total_days) = 5 := by
  sorry

end Flora_initial_daily_milk_l198_198402


namespace animals_per_aquarium_l198_198264

theorem animals_per_aquarium
  (saltwater_aquariums : ℕ)
  (saltwater_animals : ℕ)
  (h1 : saltwater_aquariums = 56)
  (h2 : saltwater_animals = 2184)
  : saltwater_animals / saltwater_aquariums = 39 := by
  sorry

end animals_per_aquarium_l198_198264


namespace prove_pattern_example_l198_198868

noncomputable def pattern_example : Prop :=
  (1 * 9 + 2 = 11) ∧
  (12 * 9 + 3 = 111) ∧
  (123 * 9 + 4 = 1111) ∧
  (1234 * 9 + 5 = 11111) ∧
  (12345 * 9 + 6 = 111111) →
  (123456 * 9 + 7 = 1111111)

theorem prove_pattern_example : pattern_example := by
  sorry

end prove_pattern_example_l198_198868


namespace pradeep_pass_percentage_l198_198580

-- Define the given data as constants
def score : ℕ := 185
def shortfall : ℕ := 25
def maxMarks : ℕ := 840

-- Calculate the passing mark
def passingMark : ℕ := score + shortfall

-- Calculate the percentage needed to pass
def passPercentage (passingMark : ℕ) (maxMarks : ℕ) : ℕ :=
  (passingMark * 100) / maxMarks

-- Statement of the theorem that we aim to prove
theorem pradeep_pass_percentage (score shortfall maxMarks : ℕ)
  (h_score : score = 185) (h_shortfall : shortfall = 25) (h_maxMarks : maxMarks = 840) :
  passPercentage (score + shortfall) maxMarks = 25 :=
by
  -- This is where the proof would go
  sorry

-- Example of calling the function to ensure definitions are correct
#eval passPercentage (score + shortfall) maxMarks -- Should output 25

end pradeep_pass_percentage_l198_198580


namespace students_taking_German_l198_198699

theorem students_taking_German 
  (total_students : ℕ)
  (students_taking_French : ℕ)
  (students_taking_both : ℕ)
  (students_not_taking_either : ℕ) 
  (students_taking_German : ℕ) 
  (h1 : total_students = 69)
  (h2 : students_taking_French = 41)
  (h3 : students_taking_both = 9)
  (h4 : students_not_taking_either = 15)
  (h5 : students_taking_German = 22) :
  total_students - students_not_taking_either = students_taking_French + students_taking_German - students_taking_both :=
sorry

end students_taking_German_l198_198699


namespace chinese_horses_problem_l198_198051

variables (x y : ℕ)

theorem chinese_horses_problem (h1 : x + y = 100) (h2 : 3 * x + (y / 3) = 100) :
  (x + y = 100) ∧ (3 * x + (y / 3) = 100) :=
by
  sorry

end chinese_horses_problem_l198_198051


namespace quadratic_inequalities_l198_198486

variable (c x₁ y₁ y₂ y₃ : ℝ)
noncomputable def quadratic_function := -x₁^2 + 2*x₁ + c

theorem quadratic_inequalities
  (h_c : c < 0)
  (h_y₁ : quadratic_function c x₁ > 0)
  (h_y₂ : y₂ = quadratic_function c (x₁ - 2))
  (h_y₃ : y₃ = quadratic_function c (x₁ + 2)) :
  y₂ < 0 ∧ y₃ < 0 :=
by sorry

end quadratic_inequalities_l198_198486


namespace negation_proof_l198_198831

theorem negation_proof : ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by
  sorry

end negation_proof_l198_198831


namespace complex_number_property_l198_198261

noncomputable def imaginary_unit : Complex := Complex.I

theorem complex_number_property (n : ℕ) (hn : 4^n = 256) : (1 + imaginary_unit)^n = -4 :=
by
  sorry

end complex_number_property_l198_198261


namespace trig_identity_l198_198776

theorem trig_identity :
  (Real.tan (30 * Real.pi / 180) * Real.cos (60 * Real.pi / 180) + Real.tan (45 * Real.pi / 180) * Real.cos (30 * Real.pi / 180)) = (2 * Real.sqrt 3) / 3 :=
by
  -- Proof is omitted
  sorry

end trig_identity_l198_198776


namespace train_overtake_distance_l198_198341

theorem train_overtake_distance (speed_a speed_b hours_late time_to_overtake distance_a distance_b : ℝ) 
  (h1 : speed_a = 30)
  (h2 : speed_b = 38)
  (h3 : hours_late = 2) 
  (h4 : distance_a = speed_a * hours_late) 
  (h5 : distance_b = speed_b * time_to_overtake) 
  (h6 : time_to_overtake = distance_a / (speed_b - speed_a)) : 
  distance_b = 285 := sorry

end train_overtake_distance_l198_198341


namespace delta_four_equal_zero_l198_198967

-- Define the sequence u_n
def u (n : ℕ) : ℤ := n^3 + n

-- Define the ∆ operator
def delta1 (u : ℕ → ℤ) (n : ℕ) : ℤ := u (n + 1) - u n

def delta (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0   => u
  | k+1 => delta1 (delta k u)

-- The theorem statement
theorem delta_four_equal_zero (n : ℕ) : delta 4 u n = 0 :=
by sorry

end delta_four_equal_zero_l198_198967


namespace complex_magnitude_l198_198348

theorem complex_magnitude (z : ℂ) (h : z * (2 - 4 * Complex.I) = 1 + 3 * Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 :=
by
  sorry

end complex_magnitude_l198_198348


namespace probability_P_plus_S_is_two_less_than_multiple_of_7_l198_198895

def is_distinct (a b : ℕ) : Prop :=
  a ≠ b

def in_range (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100

def mod_condition (a b : ℕ) : Prop :=
  (a * b + a + b) % 7 = 5

noncomputable def probability_p_s (p q : ℕ) : ℚ :=
  p / q

theorem probability_P_plus_S_is_two_less_than_multiple_of_7 :
  probability_p_s (1295) (4950) = 259 / 990 := 
sorry

end probability_P_plus_S_is_two_less_than_multiple_of_7_l198_198895


namespace determine_y_l198_198835

variable {R : Type} [LinearOrderedField R]
variables {x y : R}

theorem determine_y (h1 : 2 * x - 3 * y = 5) (h2 : 4 * x + 9 * y = 6) : y = -4 / 15 :=
by
  sorry

end determine_y_l198_198835


namespace greatest_prime_factor_180_l198_198174

noncomputable def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

theorem greatest_prime_factor_180 : 
  ∃ p : ℕ, is_prime p ∧ p ∣ 180 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 180 → q ≤ p :=
  sorry

end greatest_prime_factor_180_l198_198174


namespace curve_cross_intersection_l198_198336

theorem curve_cross_intersection : 
  ∃ (t_a t_b : ℝ), t_a ≠ t_b ∧ 
  (3 * t_a^2 + 1 = 3 * t_b^2 + 1) ∧
  (t_a^3 - 6 * t_a^2 + 4 = t_b^3 - 6 * t_b^2 + 4) ∧
  (3 * t_a^2 + 1 = 109 ∧ t_a^3 - 6 * t_a^2 + 4 = -428) := by
  sorry

end curve_cross_intersection_l198_198336


namespace range_of_m_l198_198473

-- Definitions of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b (m : ℝ) : ℝ × ℝ := (m, 4)

-- Dot product function for two 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Condition for acute angle
def is_acute (m : ℝ) : Prop := dot_product a (b m) > 0

-- Definition of the range of m
def m_range : Set ℝ := {m | m > -12 ∧ m ≠ 4/3}

-- The theorem to prove
theorem range_of_m (m : ℝ) : is_acute m → m ∈ m_range :=
by
  sorry

end range_of_m_l198_198473


namespace inner_ring_speed_minimum_train_distribution_l198_198389

theorem inner_ring_speed_minimum
  (l_inner : ℝ) (num_trains_inner : ℕ) (max_wait_inner : ℝ) (speed_min : ℝ) :
  l_inner = 30 →
  num_trains_inner = 9 →
  max_wait_inner = 10 →
  speed_min = 20 :=
by 
  sorry

theorem train_distribution
  (l_inner : ℝ) (speed_inner : ℝ) (speed_outer : ℝ) (total_trains : ℕ) (max_wait_diff : ℝ) (trains_inner : ℕ) (trains_outer : ℕ) :
  l_inner = 30 →
  speed_inner = 25 →
  speed_outer = 30 →
  total_trains = 18 →
  max_wait_diff = 1 →
  trains_inner = 10 →
  trains_outer = 8 :=
by 
  sorry

end inner_ring_speed_minimum_train_distribution_l198_198389


namespace school_class_student_count_l198_198365

theorem school_class_student_count
  (num_classes : ℕ) (num_students : ℕ)
  (h_classes : num_classes = 30)
  (h_students : num_students = 1000)
  (h_max_students_per_class : ∀(n : ℕ), n < 30 → ∀(s : ℕ), s ≤ 33 → s ≤ 1000 / 30) :
  ∃ c, c ≤ num_classes ∧ ∃s, s ≥ 34 :=
by
  sorry

end school_class_student_count_l198_198365


namespace fruit_problem_l198_198632

theorem fruit_problem :
  let apples_initial := 7
  let oranges_initial := 8
  let mangoes_initial := 15
  let grapes_initial := 12
  let strawberries_initial := 5
  let apples_taken := 3
  let oranges_taken := 4
  let mangoes_taken := 4
  let grapes_taken := 7
  let strawberries_taken := 3
  let apples_remaining := apples_initial - apples_taken
  let oranges_remaining := oranges_initial - oranges_taken
  let mangoes_remaining := mangoes_initial - mangoes_taken
  let grapes_remaining := grapes_initial - grapes_taken
  let strawberries_remaining := strawberries_initial - strawberries_taken
  let total_remaining := apples_remaining + oranges_remaining + mangoes_remaining + grapes_remaining + strawberries_remaining
  let total_taken := apples_taken + oranges_taken + mangoes_taken + grapes_taken + strawberries_taken
  total_remaining = 26 ∧ total_taken = 21 := by
    sorry

end fruit_problem_l198_198632


namespace diet_sodas_sold_l198_198386

theorem diet_sodas_sold (R D : ℕ) (h1 : R + D = 64) (h2 : R / D = 9 / 7) : D = 28 := 
by
  sorry

end diet_sodas_sold_l198_198386


namespace fraction_power_multiplication_l198_198292

theorem fraction_power_multiplication :
  ( (5 / 8: ℚ) ^ 2 * (3 / 4) ^ 2 * (2 / 3) = 75 / 512) := 
  by
  sorry

end fraction_power_multiplication_l198_198292


namespace intersection_of_sets_l198_198268

def setA : Set ℝ := {x | -2 < x ∧ x < 3}
def setB : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_sets :
  setA ∩ setB = {x | 0 < x ∧ x < 3} :=
by
  sorry

end intersection_of_sets_l198_198268


namespace highway_length_is_105_l198_198075

-- Define the speeds of the two cars
def speed_car1 : ℝ := 15
def speed_car2 : ℝ := 20

-- Define the time they travel for
def time_travelled : ℝ := 3

-- Define the distances covered by the cars
def distance_car1 : ℝ := speed_car1 * time_travelled
def distance_car2 : ℝ := speed_car2 * time_travelled

-- Define the total length of the highway
def length_highway : ℝ := distance_car1 + distance_car2

-- The theorem statement
theorem highway_length_is_105 : length_highway = 105 :=
by
  -- Skipping the proof for now
  sorry

end highway_length_is_105_l198_198075


namespace people_speak_neither_l198_198978

-- Define the total number of people
def total_people : ℕ := 25

-- Define the number of people who can speak Latin
def speak_latin : ℕ := 13

-- Define the number of people who can speak French
def speak_french : ℕ := 15

-- Define the number of people who can speak both Latin and French
def speak_both : ℕ := 9

-- Prove that the number of people who don't speak either Latin or French is 6
theorem people_speak_neither : (total_people - (speak_latin + speak_french - speak_both)) = 6 := by
  sorry

end people_speak_neither_l198_198978


namespace correct_choice_of_f_l198_198069

def f1 (x : ℝ) : ℝ := (x - 1)^2 + 3 * (x - 1)
def f2 (x : ℝ) : ℝ := 2 * (x - 1)
def f3 (x : ℝ) : ℝ := 2 * (x - 1)^2
def f4 (x : ℝ) : ℝ := x - 1

theorem correct_choice_of_f (h : (deriv f1 1 = 3) ∧ (deriv f2 1 ≠ 3) ∧ (deriv f3 1 ≠ 3) ∧ (deriv f4 1 ≠ 3)) : 
  ∀ f, (f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4) → (deriv f 1 = 3 → f = f1) :=
by sorry

end correct_choice_of_f_l198_198069


namespace sin_double_pi_minus_theta_eq_l198_198999

variable {θ : ℝ}
variable {k : ℤ}
variable (h1 : 3 * (Real.cos θ) ^ 2 = Real.tan θ + 3)
variable (h2 : θ ≠ k * Real.pi)

theorem sin_double_pi_minus_theta_eq :
  Real.sin (2 * (Real.pi - θ)) = 2 / 3 :=
sorry

end sin_double_pi_minus_theta_eq_l198_198999


namespace find_value_l198_198413

noncomputable def roots_of_equation (a b c : ℝ) : Prop :=
  10 * a^3 + 502 * a + 3010 = 0 ∧
  10 * b^3 + 502 * b + 3010 = 0 ∧
  10 * c^3 + 502 * c + 3010 = 0

theorem find_value (a b c : ℝ)
  (h : roots_of_equation a b c) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 903 :=
by
  sorry

end find_value_l198_198413


namespace sum_abs_a_l198_198964

def S (n : ℕ) : ℤ := n^2 - 4 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem sum_abs_a :
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + 
   |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 67) :=
by
  sorry

end sum_abs_a_l198_198964


namespace Helen_taller_than_Amy_l198_198156

-- Definitions from conditions
def Angela_height : ℕ := 157
def Amy_height : ℕ := 150
def Helen_height := Angela_height - 4

-- Question as a theorem
theorem Helen_taller_than_Amy : Helen_height - Amy_height = 3 := by
  sorry

end Helen_taller_than_Amy_l198_198156


namespace consecutive_integers_exist_l198_198623

def good (n : ℕ) : Prop :=
∃ (k : ℕ) (a : ℕ → ℕ), 
  (∀ i j, 1 ≤ i → i < j → j ≤ k → a i < a j) ∧ 
  (∀ i j i' j', 1 ≤ i → i < j → j ≤ k → 1 ≤ i' → i' < j' → j' ≤ k → a i + a j = a i' + a j' → i = i' ∧ j = j') ∧ 
  (∃ (t : ℕ), ∀ m, 0 ≤ m → m < n → ∃ i j, 1 ≤ i → i < j → j ≤ k → a i + a j = t + m)

theorem consecutive_integers_exist (n : ℕ) (h : n = 1000) : good n :=
sorry

end consecutive_integers_exist_l198_198623


namespace problem_statement_l198_198590

theorem problem_statement (a b : ℝ) (h : a^2 > b^2) : a > b → a > 0 :=
sorry

end problem_statement_l198_198590


namespace malingerers_exposed_l198_198615

theorem malingerers_exposed (a b c : Nat) (ha : a > b) (hc : c = b + 9) :
  let aabbb := 10000 * a + 1000 * a + 100 * b + 10 * b + b
  let abccc := 10000 * a + 1000 * b + 100 * c + 10 * c + c
  (aabbb - 1 = abccc) -> abccc = 10999 :=
by
  sorry

end malingerers_exposed_l198_198615


namespace find_n_eq_6_l198_198533

theorem find_n_eq_6 (n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) : 2^n + n^2 + 25 = p^3 → n = 6 := by
  sorry

end find_n_eq_6_l198_198533


namespace valid_license_plates_count_l198_198066

theorem valid_license_plates_count :
  let letters := 26 * 26 * 26
  let digits := 9 * 10 * 10
  letters * digits = 15818400 :=
by
  sorry

end valid_license_plates_count_l198_198066


namespace mean_age_euler_family_l198_198899

theorem mean_age_euler_family :
  let ages := [6, 6, 9, 11, 13, 16]
  let total_children := 6
  let total_sum := 61
  (total_sum / total_children : ℝ) = (61 / 6 : ℝ) :=
by
  sorry

end mean_age_euler_family_l198_198899


namespace find_page_number_l198_198060

theorem find_page_number (n p : ℕ) (h1 : (n * (n + 1)) / 2 + 2 * p = 2046) : p = 15 :=
sorry

end find_page_number_l198_198060


namespace required_run_rate_per_batsman_l198_198819

variable (initial_run_rate : ℝ) (overs_played : ℕ) (remaining_overs : ℕ)
variable (remaining_wickets : ℕ) (total_target : ℕ) 

theorem required_run_rate_per_batsman 
  (h_initial_run_rate : initial_run_rate = 3.4)
  (h_overs_played : overs_played = 10)
  (h_remaining_overs  : remaining_overs = 40)
  (h_remaining_wickets : remaining_wickets = 7)
  (h_total_target : total_target = 282) :
  (total_target - initial_run_rate * overs_played) / remaining_overs = 6.2 :=
by
  sorry

end required_run_rate_per_batsman_l198_198819


namespace sum_of_digits_inequality_l198_198655

-- Assume that S(x) represents the sum of the digits of x in its decimal representation.
axiom sum_of_digits (x : ℕ) : ℕ

-- Given condition: for any natural numbers a and b, the sum of digits function satisfies the inequality
axiom sum_of_digits_add (a b : ℕ) : sum_of_digits (a + b) ≤ sum_of_digits a + sum_of_digits b

-- Theorem statement we want to prove
theorem sum_of_digits_inequality (k : ℕ) : sum_of_digits k ≤ 8 * sum_of_digits (8 * k) := 
  sorry

end sum_of_digits_inequality_l198_198655


namespace distance_MC_l198_198879

theorem distance_MC (MA MB MC : ℝ) (hMA : MA = 2) (hMB : MB = 3) (hABC : ∀ x y z : ℝ, x + y > z ∧ y + z > x ∧ z + x > y) :
  1 ≤ MC ∧ MC ≤ 5 := 
by 
  sorry

end distance_MC_l198_198879


namespace largest_fraction_is_D_l198_198627

-- Define the fractions as Lean variables
def A : ℚ := 2 / 6
def B : ℚ := 3 / 8
def C : ℚ := 4 / 12
def D : ℚ := 7 / 16
def E : ℚ := 9 / 24

-- Define a theorem to prove the largest fraction is D
theorem largest_fraction_is_D : max (max (max A B) (max C D)) E = D :=
by
  sorry

end largest_fraction_is_D_l198_198627


namespace magnitude_of_z_l198_198625

namespace ComplexNumberProof

open Complex

noncomputable def z (b : ℝ) : ℂ := (3 - b * Complex.I) / Complex.I

theorem magnitude_of_z (b : ℝ) (h : (z b).re = (z b).im) : Complex.abs (z b) = 3 * Real.sqrt 2 :=
by
  sorry

end ComplexNumberProof

end magnitude_of_z_l198_198625


namespace sum_odds_200_600_l198_198172

-- Define the bounds 200 and 600 for our range
def lower_bound := 200
def upper_bound := 600

-- Define first and last odd integers in the range
def first_odd := 201
def last_odd := 599

-- Define the common difference in our arithmetic sequence
def common_diff := 2

-- Number of terms in the sequence
def n := ((last_odd - first_odd) / common_diff) + 1

-- Sum of the arithmetic sequence formula
def sum_arithmetic_seq (n : ℕ) (a l : ℕ) : ℕ :=
  n * (a + l) / 2

-- Specifically, the sum of odd integers between 200 and 600
def sum_odd_integers : ℕ := sum_arithmetic_seq n first_odd last_odd

-- Theorem stating the sum is equal to 80000
theorem sum_odds_200_600 : sum_odd_integers = 80000 :=
by sorry

end sum_odds_200_600_l198_198172


namespace cos_pi_over_2_minus_2alpha_l198_198086

theorem cos_pi_over_2_minus_2alpha (α : ℝ) (h : Real.tan α = 2) : Real.cos (Real.pi / 2 - 2 * α) = 4 / 5 := 
by 
  sorry

end cos_pi_over_2_minus_2alpha_l198_198086


namespace term_2012_of_T_is_2057_l198_198285

-- Define a function that checks if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the sequence T as all natural numbers which are not perfect squares
def T (n : ℕ) : ℕ :=
  (n + Nat.sqrt (4 * n)) 

-- The theorem to state the mathematical proof problem
theorem term_2012_of_T_is_2057 :
  T 2012 = 2057 :=
sorry

end term_2012_of_T_is_2057_l198_198285


namespace find_a_of_inequality_solution_set_l198_198492

theorem find_a_of_inequality_solution_set
  (a : ℝ)
  (h : ∀ x : ℝ, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) :
  a = -4 :=
sorry

end find_a_of_inequality_solution_set_l198_198492


namespace third_number_is_507_l198_198942

theorem third_number_is_507 (x : ℕ) 
  (h1 : (55 + 48 + x + 2 + 684 + 42) / 6 = 223) : 
  x = 507 := by
  sorry

end third_number_is_507_l198_198942


namespace work_completion_time_l198_198271

theorem work_completion_time (d : ℕ) (h : d = 9) : 3 * d = 27 := by
  sorry

end work_completion_time_l198_198271


namespace cubical_box_edge_length_l198_198228

noncomputable def edge_length_of_box_in_meters : ℝ :=
  let number_of_cubes := 999.9999999999998
  let edge_length_cube_cm := 10
  let volume_cube_cm := edge_length_cube_cm^3
  let total_volume_box_cm := volume_cube_cm * number_of_cubes
  let total_volume_box_meters := total_volume_box_cm / (100^3)
  (total_volume_box_meters)^(1/3)

theorem cubical_box_edge_length :
  edge_length_of_box_in_meters = 1 := 
sorry

end cubical_box_edge_length_l198_198228


namespace sum_of_reciprocals_of_roots_l198_198753

theorem sum_of_reciprocals_of_roots 
  (r₁ r₂ : ℝ)
  (h_roots : ∀ (x : ℝ), x^2 - 17*x + 8 = 0 → (∃ r, (r = r₁ ∨ r = r₂) ∧ x = r))
  (h_sum : r₁ + r₂ = 17)
  (h_prod : r₁ * r₂ = 8) :
  1/r₁ + 1/r₂ = 17/8 := 
by
  sorry

end sum_of_reciprocals_of_roots_l198_198753


namespace math_problem_l198_198358

theorem math_problem 
  (x y : ℝ) 
  (h1 : 4 * x + y = 12) 
  (h2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := 
by
  sorry

end math_problem_l198_198358


namespace fans_per_bleacher_l198_198107

theorem fans_per_bleacher 
  (total_fans : ℕ) 
  (sets_of_bleachers : ℕ) 
  (h_total : total_fans = 2436) 
  (h_sets : sets_of_bleachers = 3) : 
  total_fans / sets_of_bleachers = 812 := 
by 
  sorry

end fans_per_bleacher_l198_198107


namespace square_of_binomial_l198_198789

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b)^2 = 9 * x^2 + 30 * x + a) → a = 25 :=
by
  sorry

end square_of_binomial_l198_198789


namespace NataliesSisterInitialDiaries_l198_198736

theorem NataliesSisterInitialDiaries (D : ℕ)
  (h1 : 2 * D - (1 / 4) * 2 * D = 18) : D = 12 :=
by sorry

end NataliesSisterInitialDiaries_l198_198736


namespace find_number_l198_198017

theorem find_number (x : ℝ) : (30 / 100) * x = (60 / 100) * 150 + 120 ↔ x = 700 :=
by
  sorry

end find_number_l198_198017


namespace reciprocal_sum_l198_198955

theorem reciprocal_sum (x1 x2 x3 k : ℝ) (h : ∀ x, x^2 + k * x - k * x3 = 0 ∧ x ≠ 0 → x = x1 ∨ x = x2) :
  (1 / x1 + 1 / x2 = 1 / x3) := by
  sorry

end reciprocal_sum_l198_198955


namespace cost_of_stuffers_number_of_combinations_l198_198309

noncomputable def candy_cane_cost : ℝ := 4 * 0.5
noncomputable def beanie_baby_cost : ℝ := 2 * 3
noncomputable def book_cost : ℝ := 5
noncomputable def toy_cost : ℝ := 3 * 1
noncomputable def gift_card_cost : ℝ := 10
noncomputable def one_child_stuffers_cost : ℝ := candy_cane_cost + beanie_baby_cost + book_cost + toy_cost + gift_card_cost
noncomputable def total_cost : ℝ := one_child_stuffers_cost * 4

def num_books := 5
def num_toys := 10
def toys_combinations : ℕ := Nat.choose num_toys 3
def total_combinations : ℕ := num_books * toys_combinations

theorem cost_of_stuffers (h : total_cost = 104) : total_cost = 104 := by
  sorry

theorem number_of_combinations (h : total_combinations = 600) : total_combinations = 600 := by
  sorry

end cost_of_stuffers_number_of_combinations_l198_198309


namespace coin_toss_min_n_l198_198375

theorem coin_toss_min_n (n : ℕ) :
  (1 : ℝ) - (1 / (2 : ℝ)) ^ n ≥ 15 / 16 → n ≥ 4 :=
by
  sorry

end coin_toss_min_n_l198_198375


namespace rhombus_diagonals_not_always_equal_l198_198602

structure Rhombus where
  all_four_sides_equal : Prop
  symmetrical : Prop
  centrally_symmetrical : Prop

theorem rhombus_diagonals_not_always_equal (R : Rhombus) :
  ¬ (∀ (d1 d2 : ℝ), d1 = d2) :=
sorry

end rhombus_diagonals_not_always_equal_l198_198602


namespace minimum_handshakes_l198_198242

def binom (n k : ℕ) : ℕ := n.choose k

theorem minimum_handshakes (n_A n_B k_A k_B : ℕ) (h1 : binom (n_A + n_B) 2 + n_A + n_B = 465)
  (h2 : n_A < n_B) (h3 : k_A = n_A) (h4 : k_B = n_B) : k_A = 15 :=
by sorry

end minimum_handshakes_l198_198242


namespace time_to_reach_ship_l198_198371

-- Define the conditions
def rate_of_descent := 30 -- feet per minute
def depth_to_ship := 2400 -- feet

-- Define the proof statement
theorem time_to_reach_ship : (depth_to_ship / rate_of_descent) = 80 :=
by
  -- The proof will be inserted here in practice
  sorry

end time_to_reach_ship_l198_198371


namespace find_y_minus_x_l198_198160

theorem find_y_minus_x (x y : ℕ) (hx : x + y = 540) (hxy : (x : ℚ) / (y : ℚ) = 7 / 8) : y - x = 36 :=
by
  sorry

end find_y_minus_x_l198_198160


namespace primitive_root_exists_mod_pow_of_two_l198_198783

theorem primitive_root_exists_mod_pow_of_two (n : ℕ) : 
  (∃ x : ℤ, ∀ k : ℕ, 1 ≤ k → x^k % (2^n) ≠ 1 % (2^n)) ↔ (n ≤ 2) := sorry

end primitive_root_exists_mod_pow_of_two_l198_198783


namespace distinct_integers_division_l198_198790

theorem distinct_integers_division (n : ℤ) (h : n > 1) :
  ∃ (a b c : ℤ), a = n^2 + n + 1 ∧ b = n^2 + 2 ∧ c = n^2 + 1 ∧
  n^2 < a ∧ a < (n + 1)^2 ∧ 
  n^2 < b ∧ b < (n + 1)^2 ∧ 
  n^2 < c ∧ c < (n + 1)^2 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ c ∣ (a ^ 2 + b ^ 2) := 
by
  sorry

end distinct_integers_division_l198_198790


namespace cos_30_eq_sqrt3_div_2_l198_198877

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_30_eq_sqrt3_div_2_l198_198877


namespace range_of_m_in_inverse_proportion_function_l198_198044

theorem range_of_m_in_inverse_proportion_function (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → (1 - m) / x > 0) ∧ (x < 0 → (1 - m) / x < 0))) ↔ m < 1 :=
by
  sorry

end range_of_m_in_inverse_proportion_function_l198_198044


namespace third_place_books_max_l198_198483

theorem third_place_books_max (x y z : ℕ) (hx : 100 ∣ x) (hxpos : 0 < x) (hy : 100 ∣ y) (hz : 100 ∣ z)
  (h_sum : 2 * x + 100 + x + 100 + x + y + z ≤ 10000)
  (h_first_eq : 2 * x + 100 = x + 100 + x)
  (h_second_eq : x + 100 = y + z) 
  : x ≤ 1900 := sorry

end third_place_books_max_l198_198483


namespace symmetric_point_with_respect_to_y_eq_x_l198_198985

variables (P : ℝ × ℝ) (line : ℝ → ℝ)

theorem symmetric_point_with_respect_to_y_eq_x (P : ℝ × ℝ) (hP : P = (1, 3)) (hline : ∀ x, line x = x) :
  (∃ Q : ℝ × ℝ, Q = (3, 1) ∧ Q = (P.snd, P.fst)) :=
by
  sorry

end symmetric_point_with_respect_to_y_eq_x_l198_198985


namespace Mehki_is_10_years_older_than_Jordyn_l198_198013

def Zrinka_age : Nat := 6
def Mehki_age : Nat := 22
def Jordyn_age : Nat := 2 * Zrinka_age

theorem Mehki_is_10_years_older_than_Jordyn : Mehki_age - Jordyn_age = 10 :=
by
  sorry

end Mehki_is_10_years_older_than_Jordyn_l198_198013


namespace sum_of_three_eq_six_l198_198122

theorem sum_of_three_eq_six
  (a b c : ℕ) (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 150) :
  a + b + c = 6 :=
sorry

end sum_of_three_eq_six_l198_198122


namespace exponents_mod_7_l198_198164

theorem exponents_mod_7 : (2222 ^ 5555 + 5555 ^ 2222) % 7 = 0 := 
by 
  -- sorries here because no proof is needed as stated
  sorry

end exponents_mod_7_l198_198164


namespace white_area_l198_198279

/-- The area of a 5 by 17 rectangular sign. -/
def sign_area : ℕ := 5 * 17

/-- The area covered by the letter L. -/
def L_area : ℕ := 5 * 1 + 1 * 2

/-- The area covered by the letter O. -/
def O_area : ℕ := (3 * 3) - (1 * 1)

/-- The area covered by the letter V. -/
def V_area : ℕ := 2 * (3 * 1)

/-- The area covered by the letter E. -/
def E_area : ℕ := 3 * (1 * 3)

/-- The total area covered by the letters L, O, V, E. -/
def sum_black_area : ℕ := L_area + O_area + V_area + E_area

/-- The problem statement: Calculate the area of the white portion of the sign. -/
theorem white_area : sign_area - sum_black_area = 55 :=
by
  -- Place the proof here
  sorry

end white_area_l198_198279


namespace tractors_planting_rate_l198_198614

theorem tractors_planting_rate (total_acres : ℕ) (total_days : ℕ) 
    (tractors_first_team : ℕ) (days_first_team : ℕ)
    (tractors_second_team : ℕ) (days_second_team : ℕ)
    (total_tractor_days : ℕ) :
    total_acres = 1700 →
    total_days = 5 →
    tractors_first_team = 2 →
    days_first_team = 2 →
    tractors_second_team = 7 →
    days_second_team = 3 →
    total_tractor_days = (tractors_first_team * days_first_team) + (tractors_second_team * days_second_team) →
    total_acres / total_tractor_days = 68 :=
by
  -- proof can be filled in later
  intros
  sorry

end tractors_planting_rate_l198_198614


namespace isosceles_triangle_perimeter_l198_198040

-- Define the lengths of the sides
def side1 := 2 -- 2 cm
def side2 := 4 -- 4 cm

-- Define the condition of being isosceles
def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (a = c) ∨ (b = c)

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Define the triangle perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the main theorem to prove
theorem isosceles_triangle_perimeter {a b : ℝ} (ha : a = side1) (hb : b = side2)
    (h1 : is_isosceles a b c) (h2 : triangle_inequality a b c) : perimeter a b c = 10 :=
sorry

end isosceles_triangle_perimeter_l198_198040


namespace f_zero_eq_zero_f_periodic_l198_198345

def odd_function {α : Type*} [AddGroup α] (f : α → α) : Prop :=
∀ x, f (-x) = -f (x)

def symmetric_about (c : ℝ) (f : ℝ → ℝ) : Prop :=
∀ x, f (c + x) = f (c - x)

variable (f : ℝ → ℝ)
variables (h_odd : odd_function f) (h_sym : symmetric_about 1 f)

theorem f_zero_eq_zero : f 0 = 0 :=
sorry

theorem f_periodic : ∀ x, f (x + 4) = f x :=
sorry

end f_zero_eq_zero_f_periodic_l198_198345


namespace solve_a_l198_198953

def custom_op (a b : ℝ) : ℝ := 2 * a - b^2

theorem solve_a :
  ∃ a : ℝ, custom_op a 7 = -20 ∧ a = 29 / 2 :=
by
  sorry

end solve_a_l198_198953


namespace mirella_read_more_pages_l198_198170

-- Define the number of books Mirella read
def num_purple_books := 8
def num_orange_books := 7
def num_blue_books := 5

-- Define the number of pages per book for each color
def pages_per_purple_book := 320
def pages_per_orange_book := 640
def pages_per_blue_book := 450

-- Calculate the total pages for each color
def total_purple_pages := num_purple_books * pages_per_purple_book
def total_orange_pages := num_orange_books * pages_per_orange_book
def total_blue_pages := num_blue_books * pages_per_blue_book

-- Calculate the combined total of orange and blue pages
def total_orange_blue_pages := total_orange_pages + total_blue_pages

-- Define the target value
def page_difference := 4170

-- State the theorem to prove
theorem mirella_read_more_pages :
  total_orange_blue_pages - total_purple_pages = page_difference := by
  sorry

end mirella_read_more_pages_l198_198170


namespace length_of_XY_l198_198567

-- Defining the points on the circle
variables (A B C D P Q X Y : Type*)
-- Lengths given in the problem
variables (AB_len CD_len AP_len CQ_len PQ_len : ℕ)
-- Points and lengths conditions
variables (h1 : AB_len = 11) (h2 : CD_len = 19)
variables (h3 : AP_len = 6) (h4 : CQ_len = 7)
variables (h5 : PQ_len = 27)

-- Assuming the Power of a Point theorem applied to P and Q
variables (PX_len PY_len QX_len QY_len : ℕ)
variables (h6 : PX_len = 1) (h7 : QY_len = 3)
variables (h8 : PX_len + PQ_len + QY_len = XY_len)

-- The final length of XY is to be found
def XY_len : ℕ := PX_len + PQ_len + QY_len

-- The goal is to show XY = 31
theorem length_of_XY : XY_len = 31 :=
  by
    sorry

end length_of_XY_l198_198567


namespace solve_y_l198_198555

theorem solve_y (y : ℝ) (h1 : y > 0) (h2 : (y - 6) / 16 = 6 / (y - 16)) : y = 22 :=
by
  sorry

end solve_y_l198_198555


namespace cube_mod_35_divisors_l198_198506

theorem cube_mod_35_divisors (a : ℤ) : (35 ∣ a^3 - 1) ↔
  (∃ k : ℤ, a = 35 * k + 1) ∨ 
  (∃ k : ℤ, a = 35 * k + 11) ∨ 
  (∃ k : ℤ, a = 35 * k + 16) :=
by sorry

end cube_mod_35_divisors_l198_198506


namespace neg_p_equiv_l198_198215

def p : Prop := ∃ x₀ : ℝ, x₀^2 + 1 > 3 * x₀

theorem neg_p_equiv :
  ¬ p ↔ ∀ x : ℝ, x^2 + 1 ≤ 3 * x := by
  sorry

end neg_p_equiv_l198_198215


namespace proof_problem_l198_198152

-- Defining the statement in Lean 4.

noncomputable def p : Prop :=
  ∀ x : ℝ, x > Real.sin x

noncomputable def neg_p : Prop :=
  ∃ x : ℝ, x ≤ Real.sin x

theorem proof_problem : ¬p ↔ neg_p := 
by sorry

end proof_problem_l198_198152


namespace temperature_on_Tuesday_l198_198134

variable (T W Th F : ℝ)

theorem temperature_on_Tuesday :
  (T + W + Th) / 3 = 52 →
  (W + Th + F) / 3 = 54 →
  F = 53 →
  T = 47 := by
  intros h₁ h₂ h₃
  sorry

end temperature_on_Tuesday_l198_198134


namespace trigonometric_identity_l198_198260

theorem trigonometric_identity (α : ℝ)
  (h1 : Real.sin (π + α) = 3 / 5)
  (h2 : π < α ∧ α < 3 * π / 2) :
  (Real.sin ((π + α) / 2) - Real.cos ((π + α) / 2)) / 
  (Real.sin ((π - α) / 2) - Real.cos ((π - α) / 2)) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l198_198260


namespace students_not_reading_novels_l198_198325

theorem students_not_reading_novels
  (total_students : ℕ)
  (students_three_or_more_novels : ℕ)
  (students_two_novels : ℕ)
  (students_one_novel : ℕ)
  (h_total_students : total_students = 240)
  (h_students_three_or_more_novels : students_three_or_more_novels = 1 / 6 * 240)
  (h_students_two_novels : students_two_novels = 35 / 100 * 240)
  (h_students_one_novel : students_one_novel = 5 / 12 * 240)
  :
  total_students - (students_three_or_more_novels + students_two_novels + students_one_novel) = 16 :=
by
  sorry

end students_not_reading_novels_l198_198325


namespace fraction_to_decimal_l198_198645

theorem fraction_to_decimal :
  (17 : ℚ) / (2^2 * 5^4) = 0.0068 :=
by
  sorry

end fraction_to_decimal_l198_198645


namespace age_difference_l198_198824

theorem age_difference (P M Mo : ℕ) (h1 : P = (3 * M) / 5) (h2 : Mo = (4 * M) / 3) (h3 : P + M + Mo = 88) : Mo - P = 22 := 
by sorry

end age_difference_l198_198824


namespace lines_through_origin_l198_198631

-- Define that a, b, c are in geometric progression
def geo_prog (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

-- Define the property of the line passing through the common point (0, 0)
def passes_through_origin (a b c : ℝ) : Prop :=
  ∀ x y, (a * x + b * y = c) → (x = 0 ∧ y = 0)

theorem lines_through_origin (a b c : ℝ) (h : geo_prog a b c) : passes_through_origin a b c :=
by
  sorry

end lines_through_origin_l198_198631


namespace felix_trees_chopped_l198_198659

-- Given conditions
def cost_per_sharpening : ℕ := 8
def total_spent : ℕ := 48
def trees_per_sharpening : ℕ := 25

-- Lean statement of the problem
theorem felix_trees_chopped (h : total_spent / cost_per_sharpening * trees_per_sharpening >= 150) : True :=
by {
  -- This is just a placeholder for the proof.
  sorry
}

end felix_trees_chopped_l198_198659


namespace no_solution_to_equation_l198_198695

theorem no_solution_to_equation :
  ¬ ∃ x : ℝ, x ≠ 5 ∧ (1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5)) :=
by 
  sorry

end no_solution_to_equation_l198_198695


namespace evaluate_expression_l198_198337

theorem evaluate_expression : 
  (1 / 2 + ((2 / 3 * (3 / 8)) + 4) - (8 / 16)) = (17 / 4) :=
by
  sorry

end evaluate_expression_l198_198337


namespace find_consecutive_numbers_l198_198180

theorem find_consecutive_numbers :
  ∃ (a b c d : ℕ),
      a % 11 = 0 ∧
      b % 7 = 0 ∧
      c % 5 = 0 ∧
      d % 4 = 0 ∧
      b = a + 1 ∧
      c = a + 2 ∧
      d = a + 3 ∧
      (a % 10) = 3 ∧
      (b % 10) = 4 ∧
      (c % 10) = 5 ∧
      (d % 10) = 6 :=
sorry

end find_consecutive_numbers_l198_198180


namespace part1_condition_represents_line_part2_slope_does_not_exist_part3_x_intercept_part4_angle_condition_l198_198059

theorem part1_condition_represents_line (m : ℝ) :
  (m^2 - 2 * m - 3 ≠ 0) ∧ (2 * m^2 + m - 1 ≠ 0) ↔ m ≠ -1 :=
sorry

theorem part2_slope_does_not_exist (m : ℝ) :
  (m = 1 / 2) ↔ (m^2 - 2 * m - 3 = 0 ∧ (2 * m^2 + m - 1 = 0) ∧ ((1 * x = (4 / 3)))) :=
sorry

theorem part3_x_intercept (m : ℝ) :
  (2 * m - 6) / (m^2 - 2 * m - 3) = -3 ↔ m = -5 / 3 :=
sorry

theorem part4_angle_condition (m : ℝ) :
  -((m^2 - 2 * m - 3) / (2 * m^2 + m - 1)) = 1 ↔ m = 4 / 3 :=
sorry

end part1_condition_represents_line_part2_slope_does_not_exist_part3_x_intercept_part4_angle_condition_l198_198059


namespace smallest_n_square_smallest_n_cube_l198_198684

theorem smallest_n_square (n : ℕ) : 
  (∃ x y : ℕ, x * (x + n) = y ^ 2) ↔ n = 3 := 
by sorry

theorem smallest_n_cube (n : ℕ) : 
  (∃ x y : ℕ, x * (x + n) = y ^ 3) ↔ n = 2 := 
by sorry

end smallest_n_square_smallest_n_cube_l198_198684


namespace students_in_each_group_is_9_l198_198094

-- Define the number of students trying out for the trivia teams
def total_students : ℕ := 36

-- Define the number of students who didn't get picked for the team
def students_not_picked : ℕ := 9

-- Define the number of groups the remaining students are divided into
def number_of_groups : ℕ := 3

-- Define the function that calculates the number of students in each group
def students_per_group (total students_not_picked number_of_groups : ℕ) : ℕ :=
  (total - students_not_picked) / number_of_groups

-- Theorem: Given the conditions, the number of students in each group is 9
theorem students_in_each_group_is_9 : students_per_group total_students students_not_picked number_of_groups = 9 := 
by 
  -- proof skipped
  sorry

end students_in_each_group_is_9_l198_198094


namespace sum_of_cousins_ages_l198_198437

theorem sum_of_cousins_ages :
  ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 ∧
    a * b = 36 ∧ c * d = 40 ∧ a + b + c + d + e = 33 :=
by
  sorry

end sum_of_cousins_ages_l198_198437


namespace interior_angle_of_arithmetic_sequence_triangle_l198_198169

theorem interior_angle_of_arithmetic_sequence_triangle :
  ∀ (α d : ℝ), (α - d) + α + (α + d) = 180 → α = 60 :=
by 
  sorry

end interior_angle_of_arithmetic_sequence_triangle_l198_198169


namespace power_modulus_difference_l198_198508

theorem power_modulus_difference (m : ℤ) :
  (51 % 6 = 3) → (9 % 6 = 3) → ((51 : ℤ)^1723 - (9 : ℤ)^1723) % 6 = 0 :=
by 
  intros h1 h2
  sorry

end power_modulus_difference_l198_198508


namespace arithmetic_sum_l198_198001

theorem arithmetic_sum :
  ∀ (a : ℕ → ℝ),
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →
  (∃ x : ℝ, ∃ y : ℝ, x^2 - 6 * x - 1 = 0 ∧ y^2 - 6 * y - 1 = 0 ∧ x = a 3 ∧ y = a 15) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by
  intros a h_arith_seq h_roots
  sorry

end arithmetic_sum_l198_198001


namespace arc_length_correct_l198_198102

noncomputable def arcLengthOfCurve : ℝ :=
  ∫ φ in (0 : ℝ)..(5 * Real.pi / 12), (2 : ℝ) * (Real.sqrt (φ ^ 2 + 1))

theorem arc_length_correct :
  arcLengthOfCurve = (65 / 144) + Real.log (3 / 2) := by
  sorry

end arc_length_correct_l198_198102


namespace beth_longer_distance_by_5_miles_l198_198523

noncomputable def average_speed_john : ℝ := 40
noncomputable def time_john_hours : ℝ := 30 / 60
noncomputable def distance_john : ℝ := average_speed_john * time_john_hours

noncomputable def average_speed_beth : ℝ := 30
noncomputable def time_beth_hours : ℝ := (30 + 20) / 60
noncomputable def distance_beth : ℝ := average_speed_beth * time_beth_hours

theorem beth_longer_distance_by_5_miles : distance_beth - distance_john = 5 := by 
  sorry

end beth_longer_distance_by_5_miles_l198_198523


namespace isosceles_triangle_sides_l198_198509

theorem isosceles_triangle_sides (P : ℝ) (a b c : ℝ) (h₀ : P = 26) (h₁ : a = 11) (h₂ : a = b ∨ a = c)
  (h₃ : a + b + c = P) : 
  (b = 11 ∧ c = 4) ∨ (b = 7.5 ∧ c = 7.5) :=
by
  sorry

end isosceles_triangle_sides_l198_198509


namespace min_value_of_abc_l198_198321

noncomputable def minimum_value_abc (a b c : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ ((a + c) * (a + b) = 6 - 2 * Real.sqrt 5) → (2 * a + b + c ≥ 2 * Real.sqrt 5 - 2)

theorem min_value_of_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : (a + c) * (a + b) = 6 - 2 * Real.sqrt 5) : 
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 :=
by 
  sorry

end min_value_of_abc_l198_198321


namespace temperature_on_friday_is_72_l198_198281

-- Define the temperatures on specific days
def temp_sunday := 40
def temp_monday := 50
def temp_tuesday := 65
def temp_wednesday := 36
def temp_thursday := 82
def temp_saturday := 26

-- Average temperature over the week
def average_temp := 53

-- Total number of days in a week
def days_in_week := 7

-- Calculate the total sum of temperatures given the average temperature
def total_sum_temp : ℤ := average_temp * days_in_week

-- Sum of known temperatures from specific days
def known_sum_temp : ℤ := temp_sunday + temp_monday + temp_tuesday + temp_wednesday + temp_thursday + temp_saturday

-- Define the temperature on Friday
def temp_friday : ℤ := total_sum_temp - known_sum_temp

theorem temperature_on_friday_is_72 : temp_friday = 72 :=
by
  -- Placeholder for the proof
  sorry

end temperature_on_friday_is_72_l198_198281


namespace find_b_l198_198332

theorem find_b (a b c : ℚ) :
  -- Condition from the problem, equivalence of polynomials for all x
  ((4 : ℚ) * x^2 - 2 * x + 5 / 2) * (a * x^2 + b * x + c) =
    12 * x^4 - 8 * x^3 + 15 * x^2 - 5 * x + 5 / 2 →
  -- Given we found that a = 3 from the solution
  a = 3 →
  -- We need to prove that b = -1/2
  b = -1 / 2 :=
sorry

end find_b_l198_198332


namespace expression_X_l198_198490

variable {a b X : ℝ}

theorem expression_X (h1 : a / b = 4 / 3) (h2 : (3 * a + 2 * b) / X = 3) : X = 2 * b := 
sorry

end expression_X_l198_198490


namespace football_team_starting_lineup_count_l198_198915

theorem football_team_starting_lineup_count :
  let total_members := 12
  let offensive_lineman_choices := 4
  let quarterback_choices := 2
  let remaining_after_ol := total_members - 1 -- after choosing one offensive lineman
  let remaining_after_qb := remaining_after_ol - 1 -- after choosing one quarterback
  let running_back_choices := remaining_after_ol
  let wide_receiver_choices := remaining_after_qb - 1
  let tight_end_choices := remaining_after_qb - 2
  offensive_lineman_choices * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 5760 := 
by
  sorry

end football_team_starting_lineup_count_l198_198915


namespace regular_price_of_polo_shirt_l198_198435

/--
Zane purchases 2 polo shirts from the 40% off rack at the men's store. 
The polo shirts are priced at a certain amount at the regular price. 
He paid $60 for the shirts. 
Prove that the regular price of each polo shirt is $50.
-/
theorem regular_price_of_polo_shirt (P : ℝ) 
  (h1 : ∀ (x : ℝ), x = 0.6 * P → 2 * x = 60) : 
  P = 50 :=
sorry

end regular_price_of_polo_shirt_l198_198435


namespace _l198_198946

section BoxProblem

open Nat

def volume_box (l w h : ℕ) : ℕ := l * w * h
def volume_block (l w h : ℕ) : ℕ := l * w * h

def can_fit_blocks (box_l box_w box_h block_l block_w block_h n_blocks : ℕ) : Prop :=
  (volume_box box_l box_w box_h) = (n_blocks * volume_block block_l block_w block_h)

example : can_fit_blocks 4 3 3 3 2 1 6 :=
by
  -- calculation that proves the theorem goes here, but no need to provide proof steps
  sorry

end BoxProblem

end _l198_198946


namespace second_card_is_three_l198_198018

theorem second_card_is_three (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
                             (h_sum : a + b + c + d = 30)
                             (h_increasing : a < b ∧ b < c ∧ c < d)
                             (h_dennis : ∀ x y z, x = a → (y ≠ b ∨ z ≠ c ∨ d ≠ 30 - a - y - z))
                             (h_mandy : ∀ x y z, x = b → (y ≠ a ∨ z ≠ c ∨ d ≠ 30 - x - y - z))
                             (h_sandy : ∀ x y z, x = c → (y ≠ a ∨ z ≠ b ∨ d ≠ 30 - x - y - z))
                             (h_randy : ∀ x y z, x = d → (y ≠ a ∨ z ≠ b ∨ c ≠ 30 - x - y - z)) :
  b = 3 := 
sorry

end second_card_is_three_l198_198018


namespace rational_solves_abs_eq_l198_198072

theorem rational_solves_abs_eq (x : ℚ) : |6 + x| = |6| + |x| → 0 ≤ x := 
sorry

end rational_solves_abs_eq_l198_198072


namespace smallest_6_digit_divisible_by_111_l198_198489

theorem smallest_6_digit_divisible_by_111 :
  ∃ x : ℕ, 100000 ≤ x ∧ x ≤ 999999 ∧ x % 111 = 0 ∧ x = 100011 :=
  by
    sorry

end smallest_6_digit_divisible_by_111_l198_198489


namespace parents_gave_money_l198_198201

def money_before_birthday : ℕ := 159
def money_from_grandmother : ℕ := 25
def money_from_aunt_uncle : ℕ := 20
def total_money_after_birthday : ℕ := 279

theorem parents_gave_money :
  total_money_after_birthday = money_before_birthday + money_from_grandmother + money_from_aunt_uncle + 75 :=
by
  sorry

end parents_gave_money_l198_198201


namespace find_missing_figure_l198_198380

theorem find_missing_figure (x : ℝ) (h : 0.003 * x = 0.15) : x = 50 :=
sorry

end find_missing_figure_l198_198380


namespace value_of_q_l198_198138

theorem value_of_q (m p q a b : ℝ) 
  (h₁ : a * b = 6) 
  (h₂ : (a + 1 / b) * (b + 1 / a) = q): 
  q = 49 / 6 := 
sorry

end value_of_q_l198_198138


namespace largest_base5_three_digit_to_base10_l198_198291

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l198_198291


namespace nonnegative_integer_solutions_l198_198553

theorem nonnegative_integer_solutions (x y : ℕ) :
  3 * x^2 + 2 * 9^y = x * (4^(y+1) - 1) ↔ (x, y) ∈ [(2, 1), (3, 1), (3, 2), (18, 2)] :=
by sorry

end nonnegative_integer_solutions_l198_198553


namespace correct_calculation_l198_198973

theorem correct_calculation :
  (2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  (5 * Real.sqrt 3 * 5 * Real.sqrt 2 ≠ 5 * Real.sqrt 6) ∧
  (Real.sqrt (4 + 1/2) ≠ 2 * Real.sqrt (1/2)) :=
by
  sorry

end correct_calculation_l198_198973


namespace cafeteria_total_cost_l198_198487

-- Definitions based on conditions
def cost_per_coffee := 4
def cost_per_cake := 7
def cost_per_ice_cream := 3
def mell_coffee := 2 
def mell_cake := 1 
def friends_coffee := 2 
def friends_cake := 1 
def friends_ice_cream := 1 
def num_friends := 2
def total_coffee := mell_coffee + num_friends * friends_coffee
def total_cake := mell_cake + num_friends * friends_cake
def total_ice_cream := num_friends * friends_ice_cream

-- Total cost
def total_cost := total_coffee * cost_per_coffee + total_cake * cost_per_cake + total_ice_cream * cost_per_ice_cream

-- Theorem statement
theorem cafeteria_total_cost : total_cost = 51 := by
  sorry

end cafeteria_total_cost_l198_198487


namespace LCM_of_two_numbers_l198_198928

theorem LCM_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 14) (h2 : a * b = 2562) : Nat.lcm a b = 183 :=
by
  sorry

end LCM_of_two_numbers_l198_198928


namespace investment_time_P_l198_198080

-- Variables and conditions
variables {x : ℕ} {time_P : ℕ}

-- Conditions as seen from the mathematical problem
def investment_P (x : ℕ) := 7 * x
def investment_Q (x : ℕ) := 5 * x
def profit_ratio := 1 / 2
def time_Q := 14

-- Statement of the problem
theorem investment_time_P : 
  (profit_ratio = (investment_P x * time_P) / (investment_Q x * time_Q)) → 
  time_P = 5 := 
sorry

end investment_time_P_l198_198080


namespace negation_statement_l198_198293

variables (Students : Type) (LeftHanded InChessClub : Students → Prop)

theorem negation_statement :
  (¬ ∃ x, LeftHanded x ∧ InChessClub x) ↔ (∃ x, LeftHanded x ∧ InChessClub x) :=
by
  sorry

end negation_statement_l198_198293


namespace ninth_term_of_geometric_sequence_l198_198148

theorem ninth_term_of_geometric_sequence (a r : ℕ) (h1 : a = 3) (h2 : a * r^6 = 2187) : a * r^8 = 19683 := by
  sorry

end ninth_term_of_geometric_sequence_l198_198148


namespace binomial_expansion_coeff_x10_sub_x5_eq_251_l198_198246

open BigOperators Polynomial

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_expansion_coeff_x10_sub_x5_eq_251 :
  ∀ (a : Fin 11 → ℤ), (fun (x : ℤ) =>
    x^10 - x^5 - (a 0 + a 1 * (x - 1) + a 2 * (x - 1)^2 + 
                  a 3 * (x - 1)^3 + a 4 * (x - 1)^4 + 
                  a 5 * (x - 1)^5 + a 6 * (x - 1)^6 + 
                  a 7 * (x - 1)^7 + a 8 * (x - 1)^8 + 
                  a 9 * (x - 1)^9 + a 10 * (x - 1)^10)) = 0 → 
  a 5 = 251 := 
by 
  sorry

end binomial_expansion_coeff_x10_sub_x5_eq_251_l198_198246


namespace vector_MN_l198_198199

def M : ℝ × ℝ := (-3, 3)
def N : ℝ × ℝ := (-5, -1)
def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

theorem vector_MN :
  vector_sub N M = (-2, -4) :=
by
  sorry

end vector_MN_l198_198199


namespace prove_a4_plus_1_div_a4_l198_198254

theorem prove_a4_plus_1_div_a4 (a : ℝ) (h : (a + 1/a)^2 = 5) : a^4 + 1/(a^4) = 7 :=
by
  sorry

end prove_a4_plus_1_div_a4_l198_198254


namespace investment_in_business_l198_198876

theorem investment_in_business (Q : ℕ) (P : ℕ) 
  (h1 : Q = 65000) 
  (h2 : 4 * Q = 5 * P) : 
  P = 52000 :=
by
  rw [h1] at h2
  linarith

end investment_in_business_l198_198876


namespace smallest_group_size_l198_198884

theorem smallest_group_size (n : ℕ) (k : ℕ) (hk : k > 2) (h1 : n % 2 = 0) (h2 : n % k = 0) :
  n = 6 :=
sorry

end smallest_group_size_l198_198884


namespace divisibility_by_65_product_of_four_natural_numbers_l198_198306

def N : ℕ := 2^2022 + 1

theorem divisibility_by_65 : ∃ k : ℕ, N = 65 * k := by
  sorry

theorem product_of_four_natural_numbers :
  ∃ a b c d : ℕ, 1 < a ∧ 1 < b ∧ 1 < c ∧ 1 < d ∧ N = a * b * c * d :=
  by sorry

end divisibility_by_65_product_of_four_natural_numbers_l198_198306


namespace filter_replacement_month_l198_198198

theorem filter_replacement_month (n : ℕ) (h : n = 25) : (7 * (n - 1)) % 12 = 0 → "January" = "January" :=
by
  intros
  sorry

end filter_replacement_month_l198_198198


namespace find_y_at_neg3_l198_198145

noncomputable def quadratic_solution (y x a b : ℝ) : Prop :=
  y = x ^ 2 + a * x + b

theorem find_y_at_neg3
    (a b : ℝ)
    (h1 : 1 + a + b = 2)
    (h2 : 4 - 2 * a + b = -1)
    : quadratic_solution 2 (-3) a b :=
by
  sorry

end find_y_at_neg3_l198_198145


namespace total_amount_received_l198_198584

theorem total_amount_received (h1 : 12 = 12)
                              (h2 : 10 = 10)
                              (h3 : 8 = 8)
                              (h4 : 14 = 14)
                              (rate : 15 = 15) :
  (3 * (12 + 10 + 8 + 14) * 15) = 1980 :=
by sorry

end total_amount_received_l198_198584


namespace math_club_team_selection_l198_198681

theorem math_club_team_selection :
  let boys := 10
  let girls := 12
  let total := boys + girls
  let team_size := 8
  (Nat.choose total team_size - Nat.choose girls team_size - Nat.choose boys team_size = 319230) :=
by
  sorry

end math_club_team_selection_l198_198681


namespace Q_joined_after_4_months_l198_198091

namespace Business

-- Definitions
def P_cap := 4000
def Q_cap := 9000
def P_time := 12
def profit_ratio := (2 : ℚ) / 3

-- Statement to prove
theorem Q_joined_after_4_months (x : ℕ) (h : P_cap * P_time / (Q_cap * (12 - x)) = profit_ratio) :
  x = 4 := 
sorry

end Business

end Q_joined_after_4_months_l198_198091


namespace rainfall_in_may_l198_198748

-- Define the rainfalls for the months
def march_rain : ℝ := 3.79
def april_rain : ℝ := 4.5
def june_rain : ℝ := 3.09
def july_rain : ℝ := 4.67

-- Define the average rainfall over five months
def avg_rain : ℝ := 4

-- Define total rainfall calculation
def calc_total_rain (may_rain : ℝ) : ℝ :=
  march_rain + april_rain + may_rain + june_rain + july_rain

-- Problem statement: proving the rainfall in May
theorem rainfall_in_may : ∃ (may_rain : ℝ), calc_total_rain may_rain = avg_rain * 5 ∧ may_rain = 3.95 :=
sorry

end rainfall_in_may_l198_198748


namespace first_person_days_l198_198987

-- Define the condition that Tanya is 25% more efficient than the first person and that Tanya takes 12 days to do the work.
def tanya_more_efficient (x : ℕ) : Prop :=
  -- Efficiency relationship: tanya (12 days) = 3 days less than the first person
  12 = x - (x / 4)

-- Define the theorem that the first person takes 15 days to do the work
theorem first_person_days : ∃ x : ℕ, tanya_more_efficient x ∧ x = 15 := 
by
  sorry -- proof is not required

end first_person_days_l198_198987


namespace price_of_each_movie_in_first_box_l198_198161

theorem price_of_each_movie_in_first_box (P : ℝ) (total_movies_box1 : ℕ) (total_movies_box2 : ℕ) (price_per_movie_box2 : ℝ) (average_price : ℝ) (total_movies : ℕ) :
  total_movies_box1 = 10 →
  total_movies_box2 = 5 →
  price_per_movie_box2 = 5 →
  average_price = 3 →
  total_movies = 15 →
  10 * P + 5 * price_per_movie_box2 = average_price * total_movies →
  P = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end price_of_each_movie_in_first_box_l198_198161


namespace sampling_is_stratified_l198_198583

-- Given Conditions
def number_of_male_students := 500
def number_of_female_students := 400
def sampled_male_students := 25
def sampled_female_students := 20

-- Definition of stratified sampling according to the problem context
def is_stratified_sampling (N_M F_M R_M R_F : ℕ) : Prop :=
  (R_M > 0 ∧ R_F > 0 ∧ R_M < N_M ∧ R_F < N_M ∧ N_M > 0 ∧ N_M > 0)

-- Proving that the sampling method is stratified sampling
theorem sampling_is_stratified : 
  is_stratified_sampling number_of_male_students number_of_female_students sampled_male_students sampled_female_students = true :=
by
  sorry

end sampling_is_stratified_l198_198583


namespace product_divisible_by_4_l198_198821

theorem product_divisible_by_4 (a b c d : ℤ) 
    (h : a^2 + b^2 + c^2 = d^2) : 4 ∣ (a * b * c) :=
sorry

end product_divisible_by_4_l198_198821


namespace negation_of_universal_statement_l198_198959

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2 ≤ 0 :=
sorry

end negation_of_universal_statement_l198_198959


namespace find_y_l198_198368

-- Define vectors as tuples
def vector_1 : ℝ × ℝ := (3, 4)
def vector_2 (y : ℝ) : ℝ × ℝ := (y, -5)

-- Define dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition for orthogonality
def orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- The theorem we want to prove
theorem find_y (y : ℝ) :
  orthogonal vector_1 (vector_2 y) → y = (20 / 3) :=
by
  sorry

end find_y_l198_198368


namespace part_I_part_II_l198_198638

noncomputable def general_term (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (a 2 = 1 ∧ ∀ n, a (n + 1) - a n = d) ∧
  (d ≠ 0 ∧ (a 3)^2 = (a 2) * (a 6))

theorem part_I (a : ℕ → ℤ) (d : ℤ) : general_term a d → 
  ∀ n, a n = 2 * n - 3 := 
sorry

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) : Prop :=
  (∀ n, S n = n * (a 1 + a n) / 2) ∧ 
  (general_term a d)

theorem part_II (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) : sum_of_first_n_terms a d S → 
  ∃ n, n > 7 ∧ S n > 35 :=
sorry

end part_I_part_II_l198_198638


namespace sufficient_condition_perpendicular_l198_198989

-- Definitions of perpendicularity and lines/planes intersections
variables {Plane : Type} {Line : Type}

variable (α β γ : Plane)
variable (m n l : Line)

-- Axioms representing the given conditions
axiom perp_planes (p₁ p₂ : Plane) : Prop -- p₁ is perpendicular to p₂
axiom perp_line_plane (line : Line) (plane : Plane) : Prop -- line is perpendicular to plane

-- Given conditions for the problem.
axiom n_perp_α : perp_line_plane n α
axiom n_perp_β : perp_line_plane n β
axiom m_perp_α : perp_line_plane m α

-- The proposition to be proved.
theorem sufficient_condition_perpendicular (h₁ : perp_line_plane n α)
                                           (h₂ : perp_line_plane n β)
                                           (h₃ : perp_line_plane m α) :
  perp_line_plane m β := sorry

end sufficient_condition_perpendicular_l198_198989


namespace sum_tripled_numbers_l198_198158

theorem sum_tripled_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 :=
by
  sorry

end sum_tripled_numbers_l198_198158


namespace solve_circle_tangent_and_intercept_l198_198315

namespace CircleProblems

-- Condition: Circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + 3 = 0

-- Problem 1: Equations of tangent lines with equal intercepts
def tangent_lines_with_equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, l x y ↔ x + y + 1 = 0) ∨ (∀ x y : ℝ, l x y ↔ x + y - 3 = 0)

-- Problem 2: Equations of lines passing through origin and intercepted by the circle with a segment length of 2
def lines_intercepted_by_circle (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, l x y ↔ x = 0) ∨ (∀ x y : ℝ, l x y ↔ y = - (3 / 4) * x)

theorem solve_circle_tangent_and_intercept (l_tangent l_origin : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, circle_eq x y → l_tangent x y) →
  tangent_lines_with_equal_intercepts l_tangent ∧ lines_intercepted_by_circle l_origin :=
by
  sorry

end CircleProblems

end solve_circle_tangent_and_intercept_l198_198315


namespace max_M_l198_198466

noncomputable def conditions (x y z u : ℝ) : Prop :=
  (x - 2 * y = z - 2 * u) ∧ (2 * y * z = u * x) ∧ (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < u) ∧ (z ≥ y)

theorem max_M (x y z u : ℝ) : conditions x y z u → ∃ M : ℝ, M = 6 + 4 * Real.sqrt 2 ∧ M ≤ z / y :=
by {
  sorry
}

end max_M_l198_198466


namespace lcm_of_two_numbers_l198_198266

theorem lcm_of_two_numbers (a b : ℕ) (h_prod : a * b = 145862784) (h_hcf : Nat.gcd a b = 792) : Nat.lcm a b = 184256 :=
by {
  sorry
}

end lcm_of_two_numbers_l198_198266


namespace fraction_to_decimal_l198_198101

/-- The decimal equivalent of 1/4 is 0.25. -/
theorem fraction_to_decimal : (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end fraction_to_decimal_l198_198101


namespace well_filled_ways_1_5_l198_198517

-- Define a structure for representing the conditions of the figure filled with integers
structure WellFilledFigure where
  top_circle : ℕ
  shaded_circle_possibilities : Finset ℕ
  sub_diagram_possibilities : ℕ

-- Define an example of this structure corresponding to our problem
def figure1_5 : WellFilledFigure :=
  { top_circle := 5,
    shaded_circle_possibilities := {1, 2, 3, 4},
    sub_diagram_possibilities := 2 }

-- Define the theorem statement
theorem well_filled_ways_1_5 (f : WellFilledFigure) : (f.top_circle = 5) → 
  (f.shaded_circle_possibilities.card = 4) → 
  (f.sub_diagram_possibilities = 2) → 
  (4 * 2 = 8) := by
  sorry

end well_filled_ways_1_5_l198_198517


namespace bucket_water_total_l198_198054

theorem bucket_water_total (initial_gallons : ℝ) (added_gallons : ℝ) (total_gallons : ℝ) : 
  initial_gallons = 3 ∧ added_gallons = 6.8 → total_gallons = 9.8 :=
by
  { sorry }

end bucket_water_total_l198_198054


namespace strictly_increasing_not_gamma_interval_gamma_interval_within_one_inf_l198_198062

def f (x : ℝ) : ℝ := -x * abs x + 2 * x

theorem strictly_increasing : ∃ A : Set ℝ, A = (Set.Ioo 0 1) ∧ (∀ x y, x ∈ A → y ∈ A → x < y → f x < f y) :=
  sorry

theorem not_gamma_interval : ¬(Set.Icc (1/2) (3/2) ⊆ Set.Ioo 0 1 ∧ 
  (∀ x ∈ Set.Icc (1/2) (3/2), f x ∈ Set.Icc (1/(3/2)) (1/(1/2)))) :=
  sorry

theorem gamma_interval_within_one_inf : ∃ m n : ℝ, 1 ≤ m ∧ m < n ∧ 
  Set.Icc m n = Set.Icc 1 ((1 + Real.sqrt 5) / 2) ∧ 
  (∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (1/n) (1/m)) :=
  sorry

end strictly_increasing_not_gamma_interval_gamma_interval_within_one_inf_l198_198062


namespace max_value_of_8a_5b_15c_l198_198446

theorem max_value_of_8a_5b_15c (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) : 
  8*a + 5*b + 15*c ≤ (Real.sqrt 115) / 2 :=
by
  sorry

end max_value_of_8a_5b_15c_l198_198446


namespace at_least_one_le_one_l198_198658

theorem at_least_one_le_one (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end at_least_one_le_one_l198_198658


namespace eggs_per_hen_l198_198362

theorem eggs_per_hen (total_chickens : ℕ) (num_roosters : ℕ) (non_laying_hens : ℕ) (total_eggs : ℕ) :
  total_chickens = 440 →
  num_roosters = 39 →
  non_laying_hens = 15 →
  total_eggs = 1158 →
  (total_eggs / (total_chickens - num_roosters - non_laying_hens) = 3) :=
by
  intros
  sorry

end eggs_per_hen_l198_198362


namespace total_worth_of_stock_l198_198585

noncomputable def shop_equation (X : ℝ) : Prop :=
  0.04 * X - 0.02 * X = 400

theorem total_worth_of_stock :
  ∃ (X : ℝ), shop_equation X ∧ X = 20000 :=
by
  use 20000
  have h : shop_equation 20000 := by
    unfold shop_equation
    norm_num
  exact ⟨h, rfl⟩

end total_worth_of_stock_l198_198585


namespace reflection_of_P_across_y_axis_l198_198287

-- Define the initial point P as a tuple
def P : ℝ × ℝ := (1, -2)

-- Define the reflection across the y-axis function
def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

-- State the theorem that we want to prove
theorem reflection_of_P_across_y_axis :
  reflect_y_axis P = (-1, -2) :=
by
  -- placeholder for the proof steps
  sorry

end reflection_of_P_across_y_axis_l198_198287


namespace number_in_central_region_l198_198800

theorem number_in_central_region (a b c d : ℤ) :
  a + b + c + d = -4 →
  ∃ x : ℤ, x = -4 + 2 :=
by
  intros h
  use -2
  sorry

end number_in_central_region_l198_198800


namespace max_value_of_f_l198_198667

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 - 18 * x^2 + 27 * x

theorem max_value_of_f (x : ℝ) (h : 0 ≤ x) : ∃ M, M = 12 ∧ ∀ y, 0 ≤ y → f y ≤ M :=
sorry

end max_value_of_f_l198_198667


namespace range_of_a_for_monotonic_function_l198_198798

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

def is_monotonic_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y : ℝ⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a_for_monotonic_function :
  ∀ (a : ℝ), is_monotonic_on (f · a) (Set.Iic (-1)) → a ≤ 3 :=
by
  intros a h
  sorry

end range_of_a_for_monotonic_function_l198_198798


namespace geologists_probability_l198_198775

theorem geologists_probability :
  let r := 4 -- speed of each geologist in km/h
  let d := 6 -- distance in km
  let sectors := 8 -- number of sectors (roads)
  let total_outcomes := sectors * sectors
  let favorable_outcomes := sectors * 3 -- when distance > 6 km

  -- Calculating probability
  let P := (favorable_outcomes: ℝ) / (total_outcomes: ℝ)

  P = 0.375 :=
by
  sorry

end geologists_probability_l198_198775


namespace sequence_sum_formula_l198_198507

theorem sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_S : ∀ n, S n = (1 / 6) * (a n ^ 2 + 3 * a n - 4)) : 
  ∀ n, S n = (3 / 2) * n ^ 2 + (5 / 2) * n :=
by
  sorry

end sequence_sum_formula_l198_198507


namespace ravi_overall_profit_l198_198073

-- Define the purchase prices
def refrigerator_purchase_price := 15000
def mobile_phone_purchase_price := 8000

-- Define the percentages
def refrigerator_loss_percent := 2
def mobile_phone_profit_percent := 10

-- Define the calculations for selling prices
def refrigerator_loss_amount := (refrigerator_loss_percent / 100) * refrigerator_purchase_price
def refrigerator_selling_price := refrigerator_purchase_price - refrigerator_loss_amount

def mobile_phone_profit_amount := (mobile_phone_profit_percent / 100) * mobile_phone_purchase_price
def mobile_phone_selling_price := mobile_phone_purchase_price + mobile_phone_profit_amount

-- Define the total purchase and selling prices
def total_purchase_price := refrigerator_purchase_price + mobile_phone_purchase_price
def total_selling_price := refrigerator_selling_price + mobile_phone_selling_price

-- Define the overall profit calculation
def overall_profit := total_selling_price - total_purchase_price

-- Statement of the theorem
theorem ravi_overall_profit :
  overall_profit = 500 := by
  sorry

end ravi_overall_profit_l198_198073


namespace average_height_of_trees_l198_198096

-- Define the heights of the trees
def height_tree1: ℕ := 1000
def height_tree2: ℕ := height_tree1 / 2
def height_tree3: ℕ := height_tree1 / 2
def height_tree4: ℕ := height_tree1 + 200

-- Calculate the total number of trees
def number_of_trees: ℕ := 4

-- Compute the total height climbed
def total_height: ℕ := height_tree1 + height_tree2 + height_tree3 + height_tree4

-- Define the average height
def average_height: ℕ := total_height / number_of_trees

-- The theorem statement
theorem average_height_of_trees: average_height = 800 := by
  sorry

end average_height_of_trees_l198_198096


namespace intersection_of_sets_l198_198910

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def set_B : Set ℝ := {x | (x + 1) * (x - 4) > 0}

theorem intersection_of_sets :
  {x | -2 ≤ x ∧ x ≤ 3} ∩ {x | (x + 1) * (x - 4) > 0} = {x | -2 ≤ x ∧ x < -1} :=
by
  sorry

end intersection_of_sets_l198_198910


namespace exactly_one_box_empty_count_l198_198347

-- Define the setting with four different balls and four boxes.
def numberOfWaysExactlyOneBoxEmpty (balls : Finset ℕ) (boxes : Finset ℕ) : ℕ :=
  if (balls.card = 4 ∧ boxes.card = 4) then
     Nat.choose 4 2 * Nat.factorial 3
  else 0

theorem exactly_one_box_empty_count :
  numberOfWaysExactlyOneBoxEmpty {1, 2, 3, 4} {1, 2, 3, 4} = 144 :=
by
  -- The proof is omitted
  sorry

end exactly_one_box_empty_count_l198_198347


namespace girl_weaves_on_tenth_day_l198_198401

theorem girl_weaves_on_tenth_day 
  (a1 d : ℝ)
  (h1 : 7 * a1 + 21 * d = 28)
  (h2 : a1 + d + a1 + 4 * d + a1 + 7 * d = 15) :
  a1 + 9 * d = 10 :=
by sorry

end girl_weaves_on_tenth_day_l198_198401


namespace helga_ratio_l198_198036

variable (a b c d : ℕ)

def helga_shopping (a b c d total_shoes pairs_first_three : ℕ) : Prop :=
  a = 7 ∧
  b = a + 2 ∧
  c = 0 ∧
  a + b + c + d = total_shoes ∧
  pairs_first_three = a + b + c ∧
  total_shoes = 48 ∧
  (d : ℚ) / (pairs_first_three : ℚ) = 2

theorem helga_ratio : helga_shopping 7 9 0 32 48 16 := by
  sorry

end helga_ratio_l198_198036


namespace depth_of_box_l198_198521

theorem depth_of_box (length width depth : ℕ) (side_length : ℕ)
  (h_length : length = 30)
  (h_width : width = 48)
  (h_side_length : Nat.gcd length width = side_length)
  (h_cubes : side_length ^ 3 = 216)
  (h_volume : 80 * (side_length ^ 3) = length * width * depth) :
  depth = 12 :=
by
  sorry

end depth_of_box_l198_198521


namespace total_eggs_emily_collected_l198_198975

theorem total_eggs_emily_collected :
  let number_of_baskets := 303
  let eggs_per_basket := 28
  number_of_baskets * eggs_per_basket = 8484 :=
by
  let number_of_baskets := 303
  let eggs_per_basket := 28
  sorry -- Proof to be provided

end total_eggs_emily_collected_l198_198975


namespace expression_equals_value_l198_198770

theorem expression_equals_value : 97^3 + 3 * (97^2) + 3 * 97 + 1 = 940792 := 
by
  sorry

end expression_equals_value_l198_198770


namespace abs_sum_a_to_7_l198_198754

-- Sequence definition with domain
def a (n : ℕ) : ℤ := 2 * (n + 1) - 7  -- Lean's ℕ includes 0, so use (n + 1) instead of n here.

-- Prove absolute value sum of first seven terms
theorem abs_sum_a_to_7 : (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 25) :=
by
  -- Placeholder for actual proof
  sorry

end abs_sum_a_to_7_l198_198754


namespace revenue_after_decrease_l198_198540

theorem revenue_after_decrease (original_revenue : ℝ) (percentage_decrease : ℝ) (final_revenue : ℝ) 
  (h1 : original_revenue = 69.0) 
  (h2 : percentage_decrease = 24.637681159420293) 
  (h3 : final_revenue = original_revenue - (original_revenue * (percentage_decrease / 100))) 
  : final_revenue = 52.0 :=
by
  sorry

end revenue_after_decrease_l198_198540


namespace solve_for_x_l198_198826

noncomputable def infinite_power_tower (x : ℝ) : ℝ := sorry

theorem solve_for_x (x : ℝ) 
  (h1 : infinite_power_tower x = 4) : 
  x = Real.sqrt 2 := 
sorry

end solve_for_x_l198_198826


namespace goldbach_conjecture_2024_l198_198222

-- Definitions for the problem
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Lean 4 statement for the proof problem
theorem goldbach_conjecture_2024 :
  is_even 2024 ∧ 2024 > 2 → ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 2024 = p1 + p2 :=
by
  sorry

end goldbach_conjecture_2024_l198_198222


namespace find_a_l198_198020

noncomputable def has_exactly_one_solution_in_x (a : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 + 2*a*x + a + 5| = 3 → x = -a

theorem find_a (a : ℝ) : has_exactly_one_solution_in_x a ↔ (a = 4 ∨ a = -2) :=
by
  sorry

end find_a_l198_198020


namespace determine_h_l198_198841

-- Define the initial quadratic expression
def quadratic (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

-- Define the form we want to prove
def completed_square_form (x h k : ℝ) : ℝ := 3 * (x - h)^2 + k

-- The proof problem translated to Lean 4
theorem determine_h : ∃ k : ℝ, ∀ x : ℝ, quadratic x = completed_square_form x (-4 / 3) k :=
by
  exists (29 / 3)
  intro x
  sorry

end determine_h_l198_198841


namespace total_passengers_landed_l198_198536

theorem total_passengers_landed (on_time late : ℕ) (h1 : on_time = 14507) (h2 : late = 213) : 
    on_time + late = 14720 :=
by
  sorry

end total_passengers_landed_l198_198536


namespace exponent_problem_l198_198057

variable {a m n : ℝ}

theorem exponent_problem (h1 : a^m = 2) (h2 : a^n = 3) : a^(3*m + 2*n) = 72 := 
  sorry

end exponent_problem_l198_198057


namespace num_of_chords_l198_198092

theorem num_of_chords (n : ℕ) (h : n = 8) : (n.choose 2) = 28 :=
by
  -- Proof of this theorem will be here
  sorry

end num_of_chords_l198_198092


namespace total_volume_of_five_boxes_l198_198038

-- Define the edge length of each cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_cube (s : ℕ) : ℕ := s ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 5

-- Define the total volume
def total_volume (s : ℕ) (n : ℕ) : ℕ := n * (volume_of_cube s)

-- The theorem to prove
theorem total_volume_of_five_boxes :
  total_volume edge_length number_of_cubes = 625 := 
by
  -- Proof is skipped
  sorry

end total_volume_of_five_boxes_l198_198038


namespace solve_log_equation_l198_198229

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem solve_log_equation (x : ℝ) (hx : 2 * log_base 5 x - 3 * log_base 5 4 = 1) :
  x = 4 * Real.sqrt 5 ∨ x = -4 * Real.sqrt 5 :=
sorry

end solve_log_equation_l198_198229


namespace sum_first_50_arithmetic_sequence_l198_198556

theorem sum_first_50_arithmetic_sequence : 
  let a : ℕ := 2
  let d : ℕ := 4
  let n : ℕ := 50
  let a_n (n : ℕ) : ℕ := a + (n - 1) * d
  let S_n (n : ℕ) : ℕ := n / 2 * (2 * a + (n - 1) * d)
  S_n n = 5000 :=
by
  sorry

end sum_first_50_arithmetic_sequence_l198_198556


namespace find_n_l198_198907

theorem find_n (x : ℝ) (n : ℝ) (G : ℝ) (hG : G = (7*x^2 + 21*x + 5*n) / 7) :
  (∃ c d : ℝ, c^2 * x^2 + 2*c*d*x + d^2 = G) ↔ n = 63 / 20 :=
by
  sorry

end find_n_l198_198907


namespace solution_to_problem_l198_198230

theorem solution_to_problem (a x y n m : ℕ) (h1 : a * (x^n - x^m) = (a * x^m - 4) * y^2)
  (h2 : m % 2 = n % 2) (h3 : (a * x) % 2 = 1) : 
  x = 1 :=
sorry

end solution_to_problem_l198_198230


namespace quadratic_function_solution_l198_198795

noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 + 1/2 * x

theorem quadratic_function_solution (f : ℝ → ℝ)
  (h1 : ∃ a b c : ℝ, (a ≠ 0) ∧ (∀ x, f x = a * x^2 + b * x + c))
  (h2 : f 0 = 0)
  (h3 : ∀ x, f (x+1) = f x + x + 1) :
  ∀ x, f x = 1/2 * x^2 + 1/2 * x :=
by
  sorry

end quadratic_function_solution_l198_198795


namespace closest_integer_to_10_minus_sqrt_12_l198_198453

theorem closest_integer_to_10_minus_sqrt_12 (a b c d : ℤ) (h_a : a = 4) (h_b : b = 5) (h_c : c = 6) (h_d : d = 7) :
  d = 7 :=
by
  sorry

end closest_integer_to_10_minus_sqrt_12_l198_198453


namespace sum_first_10_terms_eq_65_l198_198803

section ArithmeticSequence

variables (a d : ℕ) (S : ℕ → ℕ) 

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Condition 1: nth term at n = 3
axiom a3_eq_4 : nth_term 3 = 4

-- Condition 2: difference in sums between n = 9 and n = 6
axiom S9_minus_S6_eq_27 : sum_first_n_terms 9 - sum_first_n_terms 6 = 27

-- To prove: sum of the first 10 terms equals 65
theorem sum_first_10_terms_eq_65 : sum_first_n_terms 10 = 65 :=
sorry

end ArithmeticSequence

end sum_first_10_terms_eq_65_l198_198803


namespace spadesuit_eval_l198_198586

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 2 3) = 0 :=
by
  sorry

end spadesuit_eval_l198_198586


namespace areaOfTangencyTriangle_l198_198823

noncomputable def semiPerimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def areaABC (a b c : ℝ) : ℝ :=
  let p := semiPerimeter a b c
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

noncomputable def excircleRadius (a b c : ℝ) : ℝ :=
  let S := areaABC a b c
  let p := semiPerimeter a b c
  S / (p - a)

theorem areaOfTangencyTriangle (a b c R : ℝ) :
  let p := semiPerimeter a b c
  let S := areaABC a b c
  let ra := excircleRadius a b c
  (S * (ra / (2 * R))) = (S ^ 2 / (2 * R * (p - a))) :=
by
  let p := semiPerimeter a b c
  let S := areaABC a b c
  let ra := excircleRadius a b c
  sorry

end areaOfTangencyTriangle_l198_198823


namespace checkerboard_problem_l198_198778

def checkerboard_rectangles : ℕ := 2025
def checkerboard_squares : ℕ := 285

def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem checkerboard_problem :
  ∃ m n : ℕ, relatively_prime m n ∧ m + n = 154 ∧ (285 : ℚ) / 2025 = m / n :=
by {
  sorry
}

end checkerboard_problem_l198_198778


namespace partition_impossible_l198_198408

def sum_of_list (l : List Int) : Int := l.foldl (· + ·) 0

theorem partition_impossible
  (l : List Int)
  (h : l = [-7, -4, -2, 3, 5, 9, 10, 18, 21, 33])
  (total_sum : Int := sum_of_list l)
  (target_diff : Int := 9) :
  ¬∃ (l1 l2 : List Int), 
    (l1 ++ l2 = l ∧ 
     sum_of_list l1 - sum_of_list l2 = target_diff ∧
     total_sum  = 86) := 
sorry

end partition_impossible_l198_198408


namespace first_place_clay_l198_198797

def Clay := "Clay"
def Allen := "Allen"
def Bart := "Bart"
def Dick := "Dick"

-- Statements made by the participants
def Allen_statements := ["I finished right before Bart", "I am not the first"]
def Bart_statements := ["I finished right before Clay", "I am not the second"]
def Clay_statements := ["I finished right before Dick", "I am not the third"]
def Dick_statements := ["I finished right before Allen", "I am not the last"]

-- Conditions
def only_two_true_statements : Prop := sorry -- This represents the condition that only two of these statements are true.
def first_place_told_truth : Prop := sorry -- This represents the condition that the person who got first place told at least one truth.

def person_first_place := Clay

theorem first_place_clay : person_first_place = Clay ∧ only_two_true_statements ∧ first_place_told_truth := 
sorry

end first_place_clay_l198_198797


namespace inscribed_circle_radius_l198_198554

noncomputable def radius_inscribed_circle (O1 O2 D : ℝ × ℝ) (r1 r2 : ℝ) :=
  if (r1 = 2 ∧ r2 = 6) ∧ ((O1.fst - O2.fst)^2 + (O1.snd - O2.snd)^2 = 64) then
    2 * (Real.sqrt 3 - 1)
  else
    0

theorem inscribed_circle_radius (O1 O2 D : ℝ × ℝ) (r1 r2 : ℝ)
  (h1 : r1 = 2) (h2 : r2 = 6)
  (h3 : (O1.fst - O2.fst)^2 + (O1.snd - O2.snd)^2 = 64) :
  radius_inscribed_circle O1 O2 D r1 r2 = 2 * (Real.sqrt 3 - 1) :=
by
  sorry

end inscribed_circle_radius_l198_198554


namespace sum_of_squares_of_non_zero_digits_from_10_to_99_l198_198650

-- Definition of the sum of squares of digits from 1 to 9
def P : ℕ := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2)

-- Definition of the sum of squares of the non-zero digits of the integers from 10 to 99
def T : ℕ := 20 * P

-- Theorem stating that T equals 5700
theorem sum_of_squares_of_non_zero_digits_from_10_to_99 : T = 5700 :=
by
  sorry

end sum_of_squares_of_non_zero_digits_from_10_to_99_l198_198650


namespace sum_twice_father_age_plus_son_age_l198_198549

/-- 
  Given:
  1. Twice the son's age plus the father's age equals 70.
  2. Father's age is 40.
  3. Son's age is 15.

  Prove:
  The sum when twice the father's age is added to the son's age is 95.
-/
theorem sum_twice_father_age_plus_son_age :
  ∀ (father_age son_age : ℕ), 
    2 * son_age + father_age = 70 → 
    father_age = 40 → 
    son_age = 15 → 
    2 * father_age + son_age = 95 := by
  intros
  sorry

end sum_twice_father_age_plus_son_age_l198_198549


namespace sequence_solution_l198_198875

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → a (n + 1) = a n * ((n + 2) / n)

theorem sequence_solution (a : ℕ → ℝ) (h1 : seq a) (h2 : a 1 = 1) :
  ∀ n : ℕ, 0 < n → a n = (n * (n + 1)) / 2 :=
by
  sorry

end sequence_solution_l198_198875


namespace full_day_students_count_l198_198322

-- Define the conditions
def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

-- Define the statement to prove
theorem full_day_students_count :
  total_students - (total_students * percentage_half_day_students / 100) = 60 :=
by
  sorry

end full_day_students_count_l198_198322


namespace trigonometric_inequality_l198_198715

theorem trigonometric_inequality (x : Real) (h1 : 0 < x) (h2 : x < (3 * Real.pi) / 8) :
  (1 / Real.sin (x / 3) + 1 / Real.sin (8 * x / 3) > (Real.sin (3 * x / 2)) / (Real.sin (x / 2) * Real.sin (2 * x))) :=
  by
  sorry

end trigonometric_inequality_l198_198715


namespace find_m_l198_198376

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, 2)
def vec_b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the addition of vectors
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

-- State the main theorem without proof
theorem find_m (m : ℝ) : dot_product (vec_add vec_a (vec_b m)) vec_a = 0 ↔ m = -7/2 := by
  sorry

end find_m_l198_198376


namespace compatibility_condition_l198_198498

theorem compatibility_condition (a b c d x : ℝ) 
  (h1 : a * x + b = 0) (h2 : c * x + d = 0) : a * d - b * c = 0 :=
sorry

end compatibility_condition_l198_198498


namespace time_spent_washing_car_l198_198079

theorem time_spent_washing_car (x : ℝ) 
  (h1 : x + (1/4) * x = 100) : x = 80 := 
sorry  

end time_spent_washing_car_l198_198079


namespace inequality_solution_set_l198_198712

def solution_set (a b x : ℝ) : Set ℝ := {x | |a - b * x| - 5 ≤ 0}

theorem inequality_solution_set (x : ℝ) :
  solution_set 4 3 x = {x | - (1 : ℝ) / 3 ≤ x ∧ x ≤ 3} :=
by {
  sorry
}

end inequality_solution_set_l198_198712


namespace clothing_factory_exceeded_tasks_l198_198937

theorem clothing_factory_exceeded_tasks :
  let first_half := (2 : ℚ) / 3
  let second_half := (3 : ℚ) / 5
  first_half + second_half - 1 = (4 : ℚ) / 15 :=
by
  sorry

end clothing_factory_exceeded_tasks_l198_198937


namespace flag_design_combinations_l198_198741

-- Definitions
def colors : Nat := 3  -- Number of colors: purple, gold, and silver
def stripes : Nat := 3  -- Number of horizontal stripes in the flag

-- The Lean statement
theorem flag_design_combinations :
  (colors ^ stripes) = 27 :=
by
  sorry

end flag_design_combinations_l198_198741


namespace Tom_money_made_l198_198429

theorem Tom_money_made (money_last_week money_now : ℕ) (h1 : money_last_week = 74) (h2 : money_now = 160) : 
  (money_now - money_last_week = 86) :=
by 
  sorry

end Tom_money_made_l198_198429


namespace book_count_l198_198468

theorem book_count (P C B : ℕ) (h1 : P = 3 * C / 2) (h2 : B = 3 * C / 4) (h3 : P + C + B > 3000) : 
  P + C + B = 3003 := by
  sorry

end book_count_l198_198468


namespace fencing_required_l198_198050

theorem fencing_required (L : ℝ) (W : ℝ) (A : ℝ) (H1 : L = 20) (H2 : A = 720) 
  (H3 : A = L * W) : L + 2 * W = 92 := by 
{
  sorry
}

end fencing_required_l198_198050


namespace y1_lt_y2_l198_198866

-- Definitions of conditions
def linear_function (x : ℝ) : ℝ := 2 * x + 1

def y1 : ℝ := linear_function (-3)
def y2 : ℝ := linear_function 4

-- Proof statement
theorem y1_lt_y2 : y1 < y2 :=
by
  -- The proof step is omitted
  sorry

end y1_lt_y2_l198_198866


namespace percentage_increase_second_movie_l198_198894

def length_first_movie : ℕ := 2
def total_length_marathon : ℕ := 9
def length_last_movie (F S : ℕ) := S + F - 1

theorem percentage_increase_second_movie :
  ∀ (S : ℕ), 
  length_first_movie + S + length_last_movie length_first_movie S = total_length_marathon →
  ((S - length_first_movie) * 100) / length_first_movie = 50 :=
by
  sorry

end percentage_increase_second_movie_l198_198894


namespace positive_number_solution_exists_l198_198505

theorem positive_number_solution_exists (x : ℝ) (h₁ : 0 < x) (h₂ : (2 / 3) * x = (64 / 216) * (1 / x)) : x = 2 / 3 :=
by sorry

end positive_number_solution_exists_l198_198505


namespace apples_in_boxes_l198_198787

theorem apples_in_boxes (apples_per_box : ℕ) (number_of_boxes : ℕ) (total_apples : ℕ) 
  (h1 : apples_per_box = 12) (h2 : number_of_boxes = 90) : total_apples = 1080 :=
by
  sorry

end apples_in_boxes_l198_198787


namespace total_legs_l198_198672

def human_legs : Nat := 2
def num_humans : Nat := 2
def dog_legs : Nat := 4
def num_dogs : Nat := 2

theorem total_legs :
  num_humans * human_legs + num_dogs * dog_legs = 12 := by
  sorry

end total_legs_l198_198672


namespace find_radii_l198_198947

-- Definitions based on the problem conditions
def tangent_lengths (TP T'Q r r' PQ: ℝ) : Prop :=
  TP = 6 ∧ T'Q = 10 ∧ PQ = 16 ∧ r < r'

-- The main theorem to prove the radii are 15 and 5
theorem find_radii (TP T'Q r r' PQ: ℝ) 
  (h : tangent_lengths TP T'Q r r' PQ) :
  r = 15 ∧ r' = 5 :=
sorry

end find_radii_l198_198947


namespace smallest_four_digit_divisible_by_53_ending_in_3_l198_198690

theorem smallest_four_digit_divisible_by_53_ending_in_3 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n % 10 = 3 ∧ n = 1113 := 
by
  sorry

end smallest_four_digit_divisible_by_53_ending_in_3_l198_198690


namespace ratio_of_scores_l198_198129

theorem ratio_of_scores (Lizzie Nathalie Aimee teammates : ℕ) (combinedLN : ℕ)
    (team_total : ℕ) (m : ℕ) :
    Lizzie = 4 →
    Nathalie = Lizzie + 3 →
    combinedLN = Lizzie + Nathalie →
    Aimee = m * combinedLN →
    teammates = 17 →
    team_total = Lizzie + Nathalie + Aimee + teammates →
    team_total = 50 →
    (Aimee / combinedLN) = 2 :=
by 
    sorry

end ratio_of_scores_l198_198129


namespace ratio_MN_l198_198857

variables (Q P R M N : ℝ)

def satisfies_conditions (Q P R M N : ℝ) : Prop :=
  M = 0.40 * Q ∧
  Q = 0.25 * P ∧
  R = 0.60 * P ∧
  N = 0.50 * R

theorem ratio_MN (Q P R M N : ℝ) (h : satisfies_conditions Q P R M N) : M / N = 1 / 3 :=
by {
  sorry
}

end ratio_MN_l198_198857


namespace value_of_inverse_product_l198_198804

theorem value_of_inverse_product (x y : ℝ) (h1 : x * y > 0) (h2 : 1/x + 1/y = 15) (h3 : (x + y) / 5 = 0.6) :
  1 / (x * y) = 5 :=
by 
  sorry

end value_of_inverse_product_l198_198804


namespace general_term_a_general_term_b_sum_first_n_terms_l198_198809

def a : Nat → Nat
| 0     => 1
| (n+1) => 2 * a n

def b (n : Nat) : Int :=
  3 * (n + 1) - 2

def S (n : Nat) : Int :=
  2^n - (3 * n^2) / 2 + n / 2 - 1

-- We state the theorems with the conditions included.

theorem general_term_a (n : Nat) : a n = 2^(n - 1) := by
  sorry

theorem general_term_b (n : Nat) : b n = 3 * (n + 1) - 2 := by
  sorry

theorem sum_first_n_terms (n : Nat) : 
  (Finset.range n).sum (λ i => a i - b i) = 2^n - (3 * n^2) / 2 + n / 2 - 1 := by
  sorry

end general_term_a_general_term_b_sum_first_n_terms_l198_198809


namespace find_i_when_x_is_0_point3_l198_198990

noncomputable def find_i (x : ℝ) (i : ℝ) : Prop :=
  (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / i

theorem find_i_when_x_is_0_point3 : find_i 0.3 2.9993 :=
by
  sorry

end find_i_when_x_is_0_point3_l198_198990


namespace product_calculation_l198_198372

theorem product_calculation :
  12 * 0.5 * 3 * 0.2 * 5 = 18 := by
  sorry

end product_calculation_l198_198372


namespace total_points_first_half_l198_198726

noncomputable def raiders_wildcats_scores := 
  ∃ (a b d r : ℕ),
    (a = b + 1) ∧
    (a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 2) ∧
    (a + a * r ≤ 100) ∧
    (b + b + d ≤ 100)

theorem total_points_first_half : 
  raiders_wildcats_scores → 
  ∃ (total : ℕ), total = 25 :=
by
  sorry

end total_points_first_half_l198_198726


namespace circle_equation_l198_198980

theorem circle_equation (x y : ℝ) :
  (∃ a < 0, (x - a)^2 + y^2 = 4 ∧ (0 - a)^2 + 0^2 = 4) ↔ (x + 2)^2 + y^2 = 4 := 
sorry

end circle_equation_l198_198980


namespace odd_function_condition_l198_198163

noncomputable def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) ↔ (a = 0 ∧ b = 0) :=
by
  sorry

end odd_function_condition_l198_198163


namespace integer_squared_equals_product_l198_198882

theorem integer_squared_equals_product : 
  3^8 * 3^12 * 2^5 * 2^10 = 1889568^2 :=
by
  sorry

end integer_squared_equals_product_l198_198882


namespace second_share_interest_rate_is_11_l198_198833

noncomputable def calculate_interest_rate 
    (total_investment : ℝ)
    (amount_in_second_share : ℝ)
    (interest_rate_first : ℝ)
    (total_interest : ℝ) : ℝ := 
  let A := total_investment - amount_in_second_share
  let interest_first := (interest_rate_first / 100) * A
  let interest_second := total_interest - interest_first
  (100 * interest_second) / amount_in_second_share

theorem second_share_interest_rate_is_11 :
  calculate_interest_rate 100000 12499.999999999998 9 9250 = 11 := 
by
  sorry

end second_share_interest_rate_is_11_l198_198833


namespace problem1_problem2_problem3_l198_198255

theorem problem1 (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + a + 3 = 0) → (a ≤ -2 ∨ a ≥ 6) :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a + 3 ≥ 4) → 
    (if a > 2 then 
      ∀ x : ℝ, ((x ≤ 1) ∨ (x ≥ a-1)) 
    else if a = 2 then 
      ∀ x : ℝ, true
    else 
      ∀ x : ℝ, ((x ≤ a - 1) ∨ (x ≥ 1))) :=
sorry

theorem problem3 (a : ℝ) :
  (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ x^2 - a*x + a + 3 = 0) → (6 ≤ a ∧ a ≤ 7) :=
sorry

end problem1_problem2_problem3_l198_198255


namespace space_shuttle_speed_l198_198224

theorem space_shuttle_speed :
  ∀ (speed_kph : ℕ) (minutes_per_hour seconds_per_minute : ℕ),
    speed_kph = 32400 →
    minutes_per_hour = 60 →
    seconds_per_minute = 60 →
    (speed_kph / (minutes_per_hour * seconds_per_minute)) = 9 :=
by
  intros speed_kph minutes_per_hour seconds_per_minute
  intro h_speed
  intro h_minutes
  intro h_seconds
  sorry

end space_shuttle_speed_l198_198224


namespace supermarket_profit_and_discount_l198_198441

theorem supermarket_profit_and_discount :
  ∃ (x : ℕ) (nB1 nB2 : ℕ) (discount_rate : ℝ),
    22*x + 30*(nB1) = 6000 ∧
    nB1 = (1 / 2 : ℝ) * x + 15 ∧
    150 * (29 - 22) + 90 * (40 - 30) = 1950 ∧
    nB2 = 3 * nB1 ∧
    150 * (29 - 22) + 270 * (40 * (1 - discount_rate / 100) - 30) = 2130 ∧
    discount_rate = 8.5 := sorry

end supermarket_profit_and_discount_l198_198441


namespace KochCurve_MinkowskiDimension_l198_198273

noncomputable def minkowskiDimensionOfKochCurve : ℝ :=
  let N (n : ℕ) := 3 * (4 ^ (n - 1))
  (Real.log 4) / (Real.log 3)

theorem KochCurve_MinkowskiDimension : minkowskiDimensionOfKochCurve = (Real.log 4) / (Real.log 3) := by
  sorry

end KochCurve_MinkowskiDimension_l198_198273


namespace power_function_monotonic_l198_198786

theorem power_function_monotonic (m : ℝ) :
  2 * m^2 + m > 0 ∧ m > 0 → m = 1 / 2 := 
by
  intro h
  sorry

end power_function_monotonic_l198_198786


namespace find_packs_of_yellow_bouncy_balls_l198_198751

noncomputable def packs_of_yellow_bouncy_balls (red_packs : ℕ) (balls_per_pack : ℕ) (extra_balls : ℕ) : ℕ :=
  (red_packs * balls_per_pack - extra_balls) / balls_per_pack

theorem find_packs_of_yellow_bouncy_balls :
  packs_of_yellow_bouncy_balls 5 18 18 = 4 := 
by
  sorry

end find_packs_of_yellow_bouncy_balls_l198_198751


namespace problem1_l198_198231

theorem problem1 :
  (-1 : ℤ)^2024 - (-1 : ℤ)^2023 = 2 := by
  sorry

end problem1_l198_198231


namespace min_value_f_l198_198912

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 3) * Real.sin x + Real.sin (Real.pi / 2 + x)

theorem min_value_f : ∃ x : ℝ, f x = -2 := by
  sorry

end min_value_f_l198_198912


namespace average_check_l198_198387

variable (a b c d e f g x : ℕ)

def sum_natural (l : List ℕ) : ℕ := l.foldr (λ x y => x + y) 0

theorem average_check (h1 : a = 54) (h2 : b = 55) (h3 : c = 57) (h4 : d = 58) (h5 : e = 59) (h6 : f = 63) (h7 : g = 65) (h8 : x = 65) (avg : 60 * 8 = 480) :
    sum_natural [a, b, c, d, e, f, g, x] = 480 :=
by
  sorry

end average_check_l198_198387


namespace example_problem_l198_198752

def operation (a b : ℕ) : ℕ := (a + b) * (a - b)

theorem example_problem : 50 - operation 8 5 = 11 := by
  sorry

end example_problem_l198_198752


namespace total_surfers_l198_198314

theorem total_surfers (num_surfs_santa_monica : ℝ) (ratio_malibu : ℝ) (ratio_santa_monica : ℝ) (ratio_venice : ℝ) (ratio_huntington : ℝ) (ratio_newport : ℝ) :
    num_surfs_santa_monica = 36 ∧ ratio_malibu = 7 ∧ ratio_santa_monica = 4.5 ∧ ratio_venice = 3.5 ∧ ratio_huntington = 2 ∧ ratio_newport = 1.5 →
    (ratio_malibu * (num_surfs_santa_monica / ratio_santa_monica) +
     num_surfs_santa_monica +
     ratio_venice * (num_surfs_santa_monica / ratio_santa_monica) +
     ratio_huntington * (num_surfs_santa_monica / ratio_santa_monica) +
     ratio_newport * (num_surfs_santa_monica / ratio_santa_monica)) = 148 :=
by
  sorry

end total_surfers_l198_198314


namespace fraction_given_to_sofia_is_correct_l198_198704

-- Pablo, Sofia, Mia, and Ana's initial egg counts
variables {m : ℕ}
def mia_initial (m : ℕ) := m
def sofia_initial (m : ℕ) := 3 * m
def pablo_initial (m : ℕ) := 12 * m
def ana_initial (m : ℕ) := m / 2

-- Total eggs and desired equal distribution
def total_eggs (m : ℕ) := 12 * m + 3 * m + m + m / 2
def equal_distribution (m : ℕ) := 33 * m / 4

-- Eggs each need to be equal
def sofia_needed (m : ℕ) := equal_distribution m - sofia_initial m
def mia_needed (m : ℕ) := equal_distribution m - mia_initial m
def ana_needed (m : ℕ) := equal_distribution m - ana_initial m

-- Fraction of eggs given to Sofia
def pablo_fraction_to_sofia (m : ℕ) := sofia_needed m / pablo_initial m

theorem fraction_given_to_sofia_is_correct (m : ℕ) :
  pablo_fraction_to_sofia m = 7 / 16 :=
sorry

end fraction_given_to_sofia_is_correct_l198_198704


namespace ironman_age_l198_198853

theorem ironman_age (T C P I : ℕ) (h1 : T = 13 * C) (h2 : C = 7 * P) (h3 : I = P + 32) (h4 : T = 1456) : I = 48 := 
by
  sorry

end ironman_age_l198_198853


namespace greatest_distance_centers_of_circles_in_rectangle_l198_198668

/--
Two circles are drawn in a 20-inch by 16-inch rectangle,
each circle with a diameter of 8 inches.
Prove that the greatest possible distance between 
the centers of the two circles without extending beyond the 
rectangular region is 4 * sqrt 13 inches.
-/
theorem greatest_distance_centers_of_circles_in_rectangle :
  let diameter := 8
  let width := 20
  let height := 16
  let radius := diameter / 2
  let reduced_width := width - 2 * radius
  let reduced_height := height - 2 * radius
  let distance := Real.sqrt ((reduced_width ^ 2) + (reduced_height ^ 2))
  distance = 4 * Real.sqrt 13 := by
    sorry

end greatest_distance_centers_of_circles_in_rectangle_l198_198668


namespace anja_equal_integers_l198_198421

theorem anja_equal_integers (S : Finset ℤ) (h_card : S.card = 2014)
  (h_mean : ∀ (x y z : ℤ), x ∈ S → y ∈ S → z ∈ S → (x + y + z) / 3 ∈ S) :
  ∃ k, ∀ x ∈ S, x = k :=
sorry

end anja_equal_integers_l198_198421


namespace least_integer_value_l198_198442

theorem least_integer_value (x : ℤ) :
  (|3 * x + 4| ≤ 25) → ∃ y : ℤ, x = y ∧ y = -9 :=
by
  sorry

end least_integer_value_l198_198442


namespace omega_range_l198_198644

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem omega_range (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 4), f ω x ≥ -2) :
  0 < ω ∧ ω ≤ 3 / 2 :=
by
  sorry

end omega_range_l198_198644


namespace exists_infinite_B_with_property_l198_198994

-- Definition of the sequence A
def seqA (n : ℕ) : ℤ := 5 * n - 2

-- Definition of the sequence B with its general form
def seqB (k : ℕ) (d : ℤ) : ℤ := k * d + 7 - d

-- The proof problem statement
theorem exists_infinite_B_with_property :
  ∃ (B : ℕ → ℤ) (d : ℤ), B 1 = 7 ∧ 
  (∀ k, k > 1 → B k = B (k - 1) + d) ∧
  (∀ n : ℕ, ∃ (k : ℕ), seqB k d = seqA n) :=
sorry

end exists_infinite_B_with_property_l198_198994


namespace chameleons_all_red_l198_198126

theorem chameleons_all_red (Y G R : ℕ) (total : ℕ) (P : Y = 7) (Q : G = 10) (R_cond : R = 17) (total_cond : Y + G + R = total) (total_value : total = 34) :
  ∃ x, x = R ∧ x = total ∧ ∀ z : ℕ, z ≠ 0 → total % 3 = z % 3 → ((R : ℕ) % 3 = z) :=
by
  sorry

end chameleons_all_red_l198_198126


namespace sqrt_sum_4_pow_4_eq_32_l198_198097

theorem sqrt_sum_4_pow_4_eq_32 : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 :=
by
  sorry

end sqrt_sum_4_pow_4_eq_32_l198_198097


namespace smallest_intersection_value_l198_198025

theorem smallest_intersection_value (a b : ℝ) (f g : ℝ → ℝ)
    (Hf : ∀ x, f x = x^4 - 6 * x^3 + 11 * x^2 - 6 * x + a)
    (Hg : ∀ x, g x = x + b)
    (Hinter : ∀ x, f x = g x → true):
  ∃ x₀, x₀ = 0 :=
by
  intros
  -- Further steps would involve proving roots and conditions stated but omitted here.
  sorry

end smallest_intersection_value_l198_198025


namespace spadesuit_eval_l198_198430

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 6 3) = -704 := by
  sorry

end spadesuit_eval_l198_198430


namespace weight_of_fresh_grapes_is_40_l198_198950

-- Define the weight of fresh grapes and dried grapes
variables (F D : ℝ)

-- Fresh grapes contain 90% water by weight, so 10% is non-water
def fresh_grapes_non_water_content (F : ℝ) : ℝ := 0.10 * F

-- Dried grapes contain 20% water by weight, so 80% is non-water
def dried_grapes_non_water_content (D : ℝ) : ℝ := 0.80 * D

-- Given condition: weight of dried grapes is 5 kg
def weight_of_dried_grapes : ℝ := 5

-- The main theorem to prove
theorem weight_of_fresh_grapes_is_40 :
  fresh_grapes_non_water_content F = dried_grapes_non_water_content weight_of_dried_grapes →
  F = 40 := 
by
  sorry

end weight_of_fresh_grapes_is_40_l198_198950


namespace not_necessarily_divisor_of_44_l198_198865

theorem not_necessarily_divisor_of_44 {k : ℤ} (h1 : ∃ k, n = k * (k + 1) * (k + 2)) (h2 : 11 ∣ n) :
  ¬(44 ∣ n) :=
sorry

end not_necessarily_divisor_of_44_l198_198865


namespace no_solution_part_a_no_solution_part_b_l198_198660

theorem no_solution_part_a 
  (x y z : ℕ) :
  ¬(x^2 + y^2 + z^2 = 2 * x * y * z) := 
sorry

theorem no_solution_part_b 
  (x y z u : ℕ) :
  ¬(x^2 + y^2 + z^2 + u^2 = 2 * x * y * z * u) := 
sorry

end no_solution_part_a_no_solution_part_b_l198_198660


namespace total_pages_in_book_l198_198601

theorem total_pages_in_book :
  ∃ x : ℝ, (x - (1/6 * x + 10) - (1/3 * (x - (1/6 * x + 10)) + 20)
           - (1/2 * ((x - (1/6 * x + 10) - (1/3 * (x - (1/6 * x + 10)) + 20))) + 25) = 120) ∧
           x = 552 :=
by
  sorry

end total_pages_in_book_l198_198601


namespace largest_among_abcd_l198_198339

theorem largest_among_abcd (a b c d k : ℤ) (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) :
  c = k + 3 ∧
  a = k + 1 ∧
  b = k - 2 ∧
  d = k - 4 ∧
  c > a ∧
  c > b ∧
  c > d :=
by
  sorry

end largest_among_abcd_l198_198339


namespace smallest_possible_N_l198_198048

theorem smallest_possible_N (table_size N : ℕ) (h_table_size : table_size = 72) :
  (∀ seating : Finset ℕ, (seating.card = N) → (seating ⊆ Finset.range table_size) →
    ∃ i ∈ Finset.range table_size, (seating = ∅ ∨ ∃ j, (j ∈ seating) ∧ (i = (j + 1) % table_size ∨ i = (j - 1) % table_size)))
  → N = 18 :=
by sorry

end smallest_possible_N_l198_198048


namespace problem_l198_198313

theorem problem (a b : ℝ) (h1 : a^2 - b^2 = 10) (h2 : a^4 + b^4 = 228) :
  a * b = 8 :=
sorry

end problem_l198_198313


namespace unique_fraction_representation_l198_198777

theorem unique_fraction_representation (p : ℕ) (h_prime : Nat.Prime p) (h_gt_2 : p > 2) :
  ∃! (x y : ℕ), (x ≠ y) ∧ (2 * x * y = p * (x + y)) :=
by
  sorry

end unique_fraction_representation_l198_198777


namespace positive_number_property_l198_198730

theorem positive_number_property (x : ℝ) (h_pos : x > 0) (h_property : 0.01 * x * x = 4) : x = 20 :=
sorry

end positive_number_property_l198_198730


namespace perpendicular_slope_l198_198127

theorem perpendicular_slope :
  ∀ (x y : ℝ), 5 * x - 2 * y = 10 → y = ((5 : ℝ) / 2) * x - 5 → ∃ (m : ℝ), m = - (2 / 5) := by
  sorry

end perpendicular_slope_l198_198127


namespace distribute_problems_l198_198801

theorem distribute_problems :
  (12 ^ 6) = 2985984 := by
  sorry

end distribute_problems_l198_198801


namespace cone_in_sphere_less_half_volume_l198_198886

theorem cone_in_sphere_less_half_volume
  (R r m : ℝ)
  (h1 : m < 2 * R)
  (h2 : r <= R) :
  (1 / 3 * Real.pi * r^2 * m < 1 / 2 * 4 / 3 * Real.pi * R^3) :=
by
  sorry

end cone_in_sphere_less_half_volume_l198_198886


namespace bob_questions_created_l198_198930

theorem bob_questions_created :
  let q1 := 13
  let q2 := 2 * q1
  let q3 := 2 * q2
  q1 + q2 + q3 = 91 :=
by
  sorry

end bob_questions_created_l198_198930


namespace toys_left_l198_198068

-- Given conditions
def initial_toys := 7
def sold_toys := 3

-- Proven statement
theorem toys_left : initial_toys - sold_toys = 4 := by
  sorry

end toys_left_l198_198068


namespace pump_filling_time_without_leak_l198_198577

theorem pump_filling_time_without_leak (P : ℝ) (h1 : 1 / P - 1 / 14 = 3 / 7) : P = 2 :=
sorry

end pump_filling_time_without_leak_l198_198577


namespace hyperbola_h_k_a_b_sum_eq_l198_198671

theorem hyperbola_h_k_a_b_sum_eq :
  ∃ (h k a b : ℝ), 
  h = 0 ∧ 
  k = 0 ∧ 
  a = 4 ∧ 
  (c : ℝ) = 8 ∧ 
  c^2 = a^2 + b^2 ∧ 
  h + k + a + b = 4 + 4 * Real.sqrt 3 := by
{ sorry }

end hyperbola_h_k_a_b_sum_eq_l198_198671


namespace x_eq_1_sufficient_not_necessary_for_x_sq_eq_1_l198_198434

theorem x_eq_1_sufficient_not_necessary_for_x_sq_eq_1 (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ ((x^2 = 1) → (x = 1 ∨ x = -1)) :=
by 
  sorry

end x_eq_1_sufficient_not_necessary_for_x_sq_eq_1_l198_198434


namespace find_t_l198_198423

-- Define the elements and the conditions
def vector_a : ℝ × ℝ := (1, -1)
def vector_b (t : ℝ) : ℝ × ℝ := (t, 1)

def add_vectors (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def sub_vectors (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Lean statement of the problem
theorem find_t (t : ℝ) : 
  parallel (add_vectors vector_a (vector_b t)) (sub_vectors vector_a (vector_b t)) → t = -1 :=
by
  sorry

end find_t_l198_198423


namespace fourth_leg_length_l198_198649

theorem fourth_leg_length (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 9) (h₃ : c = 10) :
  (∃ x : ℕ, x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ (a + x = b + c ∨ b + x = a + c ∨ c + x = a + b) ∧ (x = 7 ∨ x = 11)) :=
by sorry

end fourth_leg_length_l198_198649


namespace dale_pasta_l198_198183

-- Define the conditions
def original_pasta : Nat := 2
def original_servings : Nat := 7
def final_servings : Nat := 35

-- Define the required calculation for the number of pounds of pasta needed
def required_pasta : Nat := 10

-- The theorem to prove
theorem dale_pasta : (final_servings / original_servings) * original_pasta = required_pasta := 
by
  sorry

end dale_pasta_l198_198183


namespace math_problem_solution_l198_198426

noncomputable def a_range : Set ℝ := {a : ℝ | (0 < a ∧ a ≤ 1) ∨ (5 ≤ a ∧ a < 6)}

theorem math_problem_solution (a : ℝ) :
  (1 - 4 * (a^2 - 6 * a) > 0 ∧ a^2 - 6 * a < 0) ∨ ((a - 3)^2 - 4 < 0)
  ∧ ¬((1 - 4 * (a^2 - 6 * a) > 0 ∧ a^2 - 6 * a < 0) ∧ ((a - 3)^2 - 4 < 0)) →
  a ∈ a_range :=
sorry

end math_problem_solution_l198_198426


namespace find_other_diagonal_l198_198814

theorem find_other_diagonal (A : ℝ) (d1 : ℝ) (hA : A = 80) (hd1 : d1 = 16) :
  ∃ d2 : ℝ, 2 * A / d1 = d2 :=
by
  use 10
  -- Rest of the proof goes here
  sorry

end find_other_diagonal_l198_198814


namespace relationship_between_y_and_x_fuel_remaining_after_35_kilometers_max_distance_without_refueling_l198_198185

variable (x y : ℝ)

-- Assume the initial fuel and consumption rate
def initial_fuel : ℝ := 48
def consumption_rate : ℝ := 0.6

-- Define the fuel consumption equation
def fuel_equation (distance : ℝ) : ℝ := -consumption_rate * distance + initial_fuel

-- Theorem proving the fuel equation satisfies the specific conditions
theorem relationship_between_y_and_x :
  ∀ (x : ℝ), y = fuel_equation x :=
by
  sorry

-- Theorem proving the fuel remaining after traveling 35 kilometers
theorem fuel_remaining_after_35_kilometers :
  fuel_equation 35 = 27 :=
by
  sorry

-- Theorem proving the maximum distance the car can travel without refueling
theorem max_distance_without_refueling :
  ∃ (x : ℝ), fuel_equation x = 0 ∧ x = 80 :=
by
  sorry

end relationship_between_y_and_x_fuel_remaining_after_35_kilometers_max_distance_without_refueling_l198_198185


namespace max_soap_boxes_in_carton_l198_198945

-- Define the measurements of the carton
def L_carton := 25
def W_carton := 42
def H_carton := 60

-- Define the measurements of the soap box
def L_soap_box := 7
def W_soap_box := 12
def H_soap_box := 5

-- Calculate the volume of the carton
def V_carton := L_carton * W_carton * H_carton

-- Calculate the volume of the soap box
def V_soap_box := L_soap_box * W_soap_box * H_soap_box

-- Define the number of soap boxes that can fit in the carton
def number_of_soap_boxes := V_carton / V_soap_box

-- Prove that the number of soap boxes that can fit in the carton is 150
theorem max_soap_boxes_in_carton : number_of_soap_boxes = 150 :=
by
  -- Placeholder for the proof
  sorry

end max_soap_boxes_in_carton_l198_198945


namespace find_divisor_l198_198683

theorem find_divisor (d : ℕ) (n : ℕ) (least : ℕ)
  (h1 : least = 2)
  (h2 : n = 433124)
  (h3 : ∀ d : ℕ, (d ∣ (n + least)) → d = 2) :
  d = 2 := 
sorry

end find_divisor_l198_198683


namespace expand_product_l198_198925

def poly1 (x : ℝ) := 4 * x + 2
def poly2 (x : ℝ) := 3 * x - 1
def poly3 (x : ℝ) := x + 6

theorem expand_product (x : ℝ) :
  (poly1 x) * (poly2 x) * (poly3 x) = 12 * x^3 + 74 * x^2 + 10 * x - 12 :=
by
  sorry

end expand_product_l198_198925


namespace train_is_late_l198_198622

theorem train_is_late (S : ℝ) (T : ℝ) (T' : ℝ) (h1 : T = 2) (h2 : T' = T * 5 / 4) :
  (T' - T) * 60 = 30 :=
by
  sorry

end train_is_late_l198_198622


namespace hannah_spent_on_dessert_l198_198919

theorem hannah_spent_on_dessert
  (initial_money : ℕ)
  (money_left : ℕ)
  (half_spent_on_rides : ℕ)
  (total_spent : ℕ)
  (spent_on_dessert : ℕ)
  (H1 : initial_money = 30)
  (H2 : money_left = 10)
  (H3 : half_spent_on_rides = initial_money / 2)
  (H4 : total_spent = initial_money - money_left)
  (H5 : spent_on_dessert = total_spent - half_spent_on_rides) : spent_on_dessert = 5 :=
by
  sorry

end hannah_spent_on_dessert_l198_198919


namespace most_reasonable_sampling_method_is_stratified_l198_198780

def population_has_significant_differences 
    (grades : List String)
    (understanding : String → ℕ)
    : Prop := sorry -- This would be defined based on the details of "significant differences"

theorem most_reasonable_sampling_method_is_stratified
    (grades : List String)
    (understanding : String → ℕ)
    (h : population_has_significant_differences grades understanding)
    : (method : String) → (method = "Stratified sampling") :=
sorry

end most_reasonable_sampling_method_is_stratified_l198_198780


namespace non_congruent_triangles_l198_198515

-- Definition of points and isosceles property
variable (A B C P Q R : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited R]

-- Conditions of the problem
def is_isosceles (A B C : Type) : Prop := (A = B) ∧ (A = C)
def is_midpoint (P Q R : Type) (A B C : Type) : Prop := sorry -- precise formal definition omitted for brevity

-- Theorem stating the final result
theorem non_congruent_triangles (A B C P Q R : Type)
  (h_iso : is_isosceles A B C)
  (h_midpoints : is_midpoint P Q R A B C) :
  ∃ (n : ℕ), n = 4 := 
  by 
    -- proof abbreviated
    sorry

end non_congruent_triangles_l198_198515


namespace sum_of_roots_l198_198286

-- Define the main condition
def equation (x : ℝ) : Prop :=
  (x + 3) * (x - 4) = 22

-- Define the statement we want to prove
theorem sum_of_roots : ∀ x1 x2 : ℝ, (equation x1 ∧ equation x2) → x1 + x2 = 1 := by
  intros x1 x2 h
  sorry

end sum_of_roots_l198_198286


namespace find_a_l198_198465

theorem find_a (a : ℝ) : 3 * a + 150 = 360 → a = 70 := 
by 
  intro h
  sorry

end find_a_l198_198465


namespace infinite_integer_solutions_iff_l198_198276

theorem infinite_integer_solutions_iff
  (a b c d : ℤ) :
  (∃ inf_int_sol : (ℤ → ℤ) → Prop, ∀ (f : (ℤ → ℤ)), inf_int_sol f) ↔ (a^2 - 4*b = c^2 - 4*d) :=
by
  sorry

end infinite_integer_solutions_iff_l198_198276


namespace solve_inequality_l198_198864

open Set

-- Define a predicate for the inequality solution sets
def inequality_solution_set (k : ℝ) : Set ℝ :=
  if h : k = 0 then {x | x < 1}
  else if h : 0 < k ∧ k < 2 then {x | x < 1 ∨ x > 2 / k}
  else if h : k = 2 then {x | True} \ {1}
  else if h : k > 2 then {x | x < 2 / k ∨ x > 1}
  else {x | 2 / k < x ∧ x < 1}

-- The statement of the proof
theorem solve_inequality (k : ℝ) :
  ∀ x : ℝ, k * x^2 - (k + 2) * x + 2 < 0 ↔ x ∈ inequality_solution_set k :=
by
  sorry

end solve_inequality_l198_198864


namespace direction_vector_correct_l198_198836

open Real

def line_eq (x y : ℝ) : Prop := x - 3 * y + 1 = 0

noncomputable def direction_vector : ℝ × ℝ := (3, 1)

theorem direction_vector_correct (x y : ℝ) (h : line_eq x y) : 
    ∃ k : ℝ, direction_vector = (k * (1 : ℝ), k * (1 / 3)) :=
by
  use 3
  sorry

end direction_vector_correct_l198_198836


namespace foldable_positions_are_7_l198_198503

-- Define the initial polygon with 6 congruent squares forming a cross shape
def initial_polygon : Prop :=
  -- placeholder definition, in practice, this would be a more detailed geometrical model
  sorry

-- Define the positions where an additional square can be attached (11 positions in total)
def position (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 11

-- Define the resulting polygon when an additional square is attached at position n
def resulting_polygon (n : ℕ) : Prop :=
  position n ∧ initial_polygon

-- Define the condition that a polygon can be folded into a cube with one face missing
def can_fold_to_cube_with_missing_face (p : Prop) : Prop := sorry

-- The theorem that needs to be proved
theorem foldable_positions_are_7 : 
  ∃ (positions : Finset ℕ), 
    positions.card = 7 ∧ 
    ∀ n ∈ positions, can_fold_to_cube_with_missing_face (resulting_polygon n) :=
  sorry

end foldable_positions_are_7_l198_198503


namespace induction_inequality_l198_198272

variable (n : ℕ) (h₁ : n ∈ Set.Icc 2 (2^n - 1))

theorem induction_inequality : 1 + 1/2 + 1/3 < 2 := 
  sorry

end induction_inequality_l198_198272


namespace participation_arrangements_l198_198116

def num_students : ℕ := 5
def num_competitions : ℕ := 3
def eligible_dance_students : ℕ := 4

def arrangements_singing : ℕ := num_students
def arrangements_chess : ℕ := num_students
def arrangements_dance : ℕ := eligible_dance_students

def total_arrangements : ℕ := arrangements_singing * arrangements_chess * arrangements_dance

theorem participation_arrangements :
  total_arrangements = 100 := by
  sorry

end participation_arrangements_l198_198116


namespace angle_C_of_quadrilateral_ABCD_l198_198162

theorem angle_C_of_quadrilateral_ABCD
  (AB CD BC AD : ℝ) (D : ℝ) (h_AB_CD : AB = CD) (h_BC_AD : BC = AD) (h_ang_D : D = 120) :
  ∃ C : ℝ, C = 60 :=
by
  sorry

end angle_C_of_quadrilateral_ABCD_l198_198162


namespace ratio_of_areas_l198_198765

theorem ratio_of_areas (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : s2 = 5) :
  let area_equilateral (s : ℝ) := (Real.sqrt 3 / 4) * s^2
  let area_large_triangle := area_equilateral s1
  let area_small_triangle := area_equilateral s2
  let area_trapezoid := area_large_triangle - area_small_triangle
  area_small_triangle / area_trapezoid = 1 / 3 := 
by
  sorry

end ratio_of_areas_l198_198765


namespace cubic_sum_identity_l198_198563

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b + a * c + b * c = -6) (h3 : a * b * c = -3) :
  a^3 + b^3 + c^3 = 27 :=
by
  sorry

end cubic_sum_identity_l198_198563


namespace negation_exists_zero_product_l198_198678

variable {R : Type} [LinearOrderedField R]

variable (f g : R → R)

theorem negation_exists_zero_product :
  (¬ ∃ x : R, f x * g x = 0) ↔ ∀ x : R, f x ≠ 0 ∧ g x ≠ 0 :=
by
  sorry

end negation_exists_zero_product_l198_198678


namespace cubic_expansion_solution_l198_198529

theorem cubic_expansion_solution (x y : ℕ) (h_x : x = 27) (h_y : y = 9) : 
  x^3 + 3 * x^2 * y + 3 * x * y^2 + y^3 = 46656 :=
by
  sorry

end cubic_expansion_solution_l198_198529


namespace total_time_to_clean_and_complete_l198_198855

def time_to_complete_assignment : Nat := 10
def num_remaining_keys : Nat := 14
def time_per_key : Nat := 3

theorem total_time_to_clean_and_complete :
  time_to_complete_assignment + num_remaining_keys * time_per_key = 52 :=
by
  sorry

end total_time_to_clean_and_complete_l198_198855


namespace highest_prob_of_red_card_l198_198575

theorem highest_prob_of_red_card :
  let deck_size := 52
  let num_aces := 4
  let num_hearts := 13
  let num_kings := 4
  let num_reds := 26
  -- Event probabilities
  let prob_ace := num_aces / deck_size
  let prob_heart := num_hearts / deck_size
  let prob_king := num_kings / deck_size
  let prob_red := num_reds / deck_size
  prob_red > prob_heart ∧ prob_heart > prob_ace ∧ prob_ace = prob_king :=
sorry

end highest_prob_of_red_card_l198_198575


namespace sufficient_but_not_necessary_l198_198213

-- Define what it means for a line to be perpendicular to a plane
def line_perpendicular_to_plane (l : Type) (alpha : Type) : Prop := 
  sorry -- Definition not provided

-- Define what it means for a line to be perpendicular to countless lines in a plane
def line_perpendicular_to_countless_lines_in_plane (l : Type) (alpha : Type) : Prop := 
  sorry -- Definition not provided

-- The formal statement
theorem sufficient_but_not_necessary (l : Type) (alpha : Type) :
  (line_perpendicular_to_plane l alpha) → (line_perpendicular_to_countless_lines_in_plane l alpha) ∧ 
  ¬ ((line_perpendicular_to_countless_lines_in_plane l alpha) → (line_perpendicular_to_plane l alpha)) :=
by sorry

end sufficient_but_not_necessary_l198_198213


namespace angles_arithmetic_progression_l198_198456

theorem angles_arithmetic_progression (A B C : ℝ) (h_sum : A + B + C = 180) :
  (B = 60) ↔ (A + C = 2 * B) :=
by
  sorry

end angles_arithmetic_progression_l198_198456


namespace inequality_solution_l198_198327

theorem inequality_solution (x : ℝ) : (x - 1) / 3 > 2 → x > 7 :=
by
  intros h
  sorry

end inequality_solution_l198_198327


namespace candy_distribution_l198_198472

theorem candy_distribution (n : Nat) : ∃ k : Nat, n = 2 ^ k :=
sorry

end candy_distribution_l198_198472


namespace vector_dot_product_sum_l198_198756

noncomputable def points_in_plane (A B C : Type) (dist_AB dist_BC dist_CA : ℝ) : Prop :=
  dist_AB = 3 ∧ dist_BC = 5 ∧ dist_CA = 6

theorem vector_dot_product_sum (A B C : Type) (dist_AB dist_BC dist_CA : ℝ) (HA : points_in_plane A B C dist_AB dist_BC dist_CA) :
    ∃ (AB BC CA : ℝ), AB * BC + BC * CA + CA * AB = -35 :=
by
  sorry

end vector_dot_product_sum_l198_198756


namespace courier_speeds_correctness_l198_198618

noncomputable def courier_speeds : Prop :=
  ∃ (s1 s2 : ℕ), (s1 * 8 + s2 * 8 = 176) ∧ (s1 = 60 / 5) ∧ (s2 = 60 / 6)

theorem courier_speeds_correctness : courier_speeds :=
by
  sorry

end courier_speeds_correctness_l198_198618


namespace sum_digits_500_l198_198037

noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_500 (k : ℕ) (h : k = 55) :
  sum_digits (63 * 10^k - 64) = 500 :=
by
  sorry

end sum_digits_500_l198_198037


namespace compound_bar_chart_must_clearly_indicate_legend_l198_198920

-- Definitions of the conditions
structure CompoundBarChart where
  distinguishes_two_quantities : Bool
  uses_bars_of_different_colors : Bool

-- The theorem stating that a compound bar chart must clearly indicate the legend
theorem compound_bar_chart_must_clearly_indicate_legend 
  (chart : CompoundBarChart)
  (distinguishes_quantities : chart.distinguishes_two_quantities = true)
  (uses_colors : chart.uses_bars_of_different_colors = true) :
  ∃ legend : String, legend ≠ "" := by
  sorry

end compound_bar_chart_must_clearly_indicate_legend_l198_198920


namespace investment_rate_l198_198130

theorem investment_rate (total_investment income1_rate income2_rate income_total remaining_investment expected_income : ℝ)
  (h1 : total_investment = 12000)
  (h2 : income1_rate = 0.03)
  (h3 : income2_rate = 0.045)
  (h4 : expected_income = 600)
  (h5 : (5000 * income1_rate + 4000 * income2_rate) = 330)
  (h6 : remaining_investment = total_investment - 5000 - 4000) :
  (remaining_investment * 0.09 = expected_income - (5000 * income1_rate + 4000 * income2_rate)) :=
by
  sorry

end investment_rate_l198_198130


namespace problem_geometric_description_of_set_T_l198_198792

open Complex

def set_T (a b : ℝ) : ℂ := a + b * I

theorem problem_geometric_description_of_set_T :
  {w : ℂ | ∃ a b : ℝ, w = set_T a b ∧
    (im ((5 - 3 * I) * w) = 2 * re ((5 - 3 * I) * w))} =
  {w : ℂ | ∃ a : ℝ, w = set_T a (-(13/5) * a)} :=
sorry

end problem_geometric_description_of_set_T_l198_198792


namespace line_contains_point_l198_198464

theorem line_contains_point {
    k : ℝ
} :
  (2 - k * 3 = -4 * 1) → k = 2 :=
by
  sorry

end line_contains_point_l198_198464


namespace oak_total_after_planting_l198_198960

-- Let oak_current represent the current number of oak trees in the park.
def oak_current : ℕ := 9

-- Let oak_new represent the number of new oak trees being planted.
def oak_new : ℕ := 2

-- The problem is to prove the total number of oak trees after planting equals 11
theorem oak_total_after_planting : oak_current + oak_new = 11 :=
by
  sorry

end oak_total_after_planting_l198_198960


namespace rachel_bella_total_distance_l198_198373

theorem rachel_bella_total_distance:
  ∀ (distance_land distance_sea total_distance: ℕ), 
  distance_land = 451 → 
  distance_sea = 150 → 
  total_distance = distance_land + distance_sea → 
  total_distance = 601 := 
by 
  intros distance_land distance_sea total_distance h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end rachel_bella_total_distance_l198_198373


namespace evaluate_expression_l198_198664

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l198_198664


namespace distance_last_pair_of_trees_l198_198052

theorem distance_last_pair_of_trees 
  (yard_length : ℝ := 1200)
  (num_trees : ℕ := 117)
  (initial_distance : ℝ := 5)
  (distance_increment : ℝ := 2) :
  let num_distances := num_trees - 1
  let last_distance := initial_distance + (num_distances - 1) * distance_increment
  last_distance = 235 := by 
  sorry

end distance_last_pair_of_trees_l198_198052


namespace find_s_l198_198179

theorem find_s : ∃ s : ℚ, (∀ x : ℚ, (3 * x^2 - 8 * x + 9) * (5 * x^2 + s * x + 15) = 15 * x^4 - 71 * x^3 + 174 * x^2 - 215 * x + 135) ∧ s = -95 / 9 := sorry

end find_s_l198_198179


namespace find_constants_l198_198941

theorem find_constants (P Q R : ℚ) :
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ -2 →
    (x^2 + x - 8) / ((x - 1) * (x - 4) * (x + 2)) = 
    P / (x - 1) + Q / (x - 4) + R / (x + 2))
  → (P = 2 / 3 ∧ Q = 8 / 9 ∧ R = -5 / 9) :=
by
  sorry

end find_constants_l198_198941


namespace water_bottles_needed_l198_198524

theorem water_bottles_needed : 
  let number_of_people := 4
  let hours_to_destination := 8
  let hours_to_return := 8
  let hours_total := hours_to_destination + hours_to_return
  let bottles_per_person_per_hour := 1 / 2
  let total_bottles_per_hour := number_of_people * bottles_per_person_per_hour
  let total_bottles := total_bottles_per_hour * hours_total
  total_bottles = 32 :=
by
  sorry

end water_bottles_needed_l198_198524


namespace min_value_of_expression_l198_198366

open Real

theorem min_value_of_expression (x y z : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : 0 < z) (h₃ : x * y * z = 18) :
  x^2 + 4*x*y + y^2 + 3*z^2 ≥ 63 :=
sorry

end min_value_of_expression_l198_198366


namespace solution_set_l198_198277

theorem solution_set (x : ℝ) : 
  1 < |x + 2| ∧ |x + 2| < 5 ↔ 
  (-7 < x ∧ x < -3) ∨ (-1 < x ∧ x < 3) := 
by 
  sorry

end solution_set_l198_198277


namespace correct_total_weight_6_moles_Al2_CO3_3_l198_198717

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

def num_atoms_Al : ℕ := 2
def num_atoms_C : ℕ := 3
def num_atoms_O : ℕ := 9

def molecular_weight_Al2_CO3_3 : ℝ :=
  (num_atoms_Al * atomic_weight_Al) +
  (num_atoms_C * atomic_weight_C) +
  (num_atoms_O * atomic_weight_O)

def num_moles : ℝ := 6

def total_weight_6_moles_Al2_CO3_3 : ℝ := num_moles * molecular_weight_Al2_CO3_3

theorem correct_total_weight_6_moles_Al2_CO3_3 :
  total_weight_6_moles_Al2_CO3_3 = 1403.94 :=
by
  unfold total_weight_6_moles_Al2_CO3_3
  unfold num_moles
  unfold molecular_weight_Al2_CO3_3
  unfold num_atoms_Al num_atoms_C num_atoms_O atomic_weight_Al atomic_weight_C atomic_weight_O
  sorry

end correct_total_weight_6_moles_Al2_CO3_3_l198_198717


namespace smallest_positive_angle_l198_198982

def coterminal_angle (θ : ℤ) : ℤ := θ % 360

theorem smallest_positive_angle (θ : ℤ) (hθ : θ % 360 ≠ 0) : 
  0 < coterminal_angle θ ∧ coterminal_angle θ = 158 :=
by
  sorry

end smallest_positive_angle_l198_198982


namespace midpoint_s2_l198_198289

structure Point where
  x : ℤ
  y : ℤ

def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def translate (p : Point) (dx dy : ℤ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem midpoint_s2 :
  let s1_p1 := ⟨6, -2⟩
  let s1_p2 := ⟨-4, 6⟩
  let s1_mid := midpoint s1_p1 s1_p2
  let s2_mid_translated := translate s1_mid (-3) (-4)
  s2_mid_translated = ⟨-2, -2⟩ := 
by
  sorry

end midpoint_s2_l198_198289


namespace find_b_in_expression_l198_198870

theorem find_b_in_expression
  (a b : ℚ)
  (h : (1 + Real.sqrt 3)^5 = a + b * Real.sqrt 3) :
  b = 44 :=
sorry

end find_b_in_expression_l198_198870


namespace altitude_identity_l198_198361

variable {a b c d : ℝ}

def is_right_triangle (A B C : ℝ) : Prop :=
  A^2 + B^2 = C^2

def right_angle_triangle (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2

def altitude_property (a b c d : ℝ) : Prop :=
  a * b = c * d

theorem altitude_identity (a b c d : ℝ) (h1: right_angle_triangle a b c) (h2: altitude_property a b c d) :
  1 / a^2 + 1 / b^2 = 1 / d^2 :=
sorry

end altitude_identity_l198_198361


namespace problem_l198_198908

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 10) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem problem : f (g 2) + g (f 2) = 38 / 5 :=
by
  sorry

end problem_l198_198908


namespace advertising_time_l198_198621

-- Define the conditions
def total_duration : ℕ := 30
def national_news : ℕ := 12
def international_news : ℕ := 5
def sports : ℕ := 5
def weather_forecasts : ℕ := 2

-- Calculate total content time
def total_content_time : ℕ := national_news + international_news + sports + weather_forecasts

-- Define the proof problem
theorem advertising_time (h : total_duration - total_content_time = 6) : (total_duration - total_content_time) = 6 :=
by
sorry

end advertising_time_l198_198621


namespace quadratic_equal_roots_l198_198902

theorem quadratic_equal_roots (k : ℝ) : (∃ r : ℝ, (r^2 - 2 * r + k = 0)) → k = 1 := 
by
  sorry

end quadratic_equal_roots_l198_198902


namespace distance_between_points_l198_198676

theorem distance_between_points : 
  let p1 := (3, -2) 
  let p2 := (-7, 4) 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 136 :=
by
  sorry

end distance_between_points_l198_198676


namespace cindy_correct_operation_l198_198548

-- Let's define the conditions and proof statement in Lean 4.

variable (x : ℝ)
axiom incorrect_operation : (x - 7) / 5 = 25

theorem cindy_correct_operation :
  (x - 5) / 7 = 18 + 1 / 7 :=
sorry

end cindy_correct_operation_l198_198548


namespace longest_interval_between_friday_13ths_l198_198791

theorem longest_interval_between_friday_13ths
  (friday_the_13th : ℕ → ℕ → Prop)
  (at_least_once_per_year : ∀ year, ∃ month, friday_the_13th year month)
  (friday_occurs : ℕ) :
  ∃ (interval : ℕ), interval = 14 :=
by
  sorry

end longest_interval_between_friday_13ths_l198_198791


namespace eliot_account_balance_l198_198067

theorem eliot_account_balance 
  (A E : ℝ) 
  (h1 : A > E)
  (h2 : A - E = (1 / 12) * (A + E))
  (h3 : 1.10 * A = 1.20 * E + 20) : 
  E = 200 :=
by 
  sorry

end eliot_account_balance_l198_198067


namespace find_intersecting_lines_l198_198609

theorem find_intersecting_lines (x y : ℝ) : 
  (2 * x - y)^2 - (x + 3 * y)^2 = 0 ↔ x = 4 * y ∨ x = - (2 / 3) * y :=
by
  sorry

end find_intersecting_lines_l198_198609


namespace matrix_pow_A4_l198_198125

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1], ![1, 1]]

-- State the theorem
theorem matrix_pow_A4 :
  A^4 = ![![0, -9], ![9, -9]] :=
by
  sorry -- Proof is omitted

end matrix_pow_A4_l198_198125


namespace number_of_girls_in_club_l198_198888

theorem number_of_girls_in_club (total : ℕ) (C1 : total = 36) 
    (C2 : ∀ (S : Finset ℕ), S.card = 33 → ∃ g b : ℕ, g + b = 33 ∧ g > b) 
    (C3 : ∃ (S : Finset ℕ), S.card = 31 ∧ ∃ g b : ℕ, g + b = 31 ∧ b > g) : 
    ∃ G : ℕ, G = 20 :=
by
  sorry

end number_of_girls_in_club_l198_198888


namespace fraction_computation_l198_198860

theorem fraction_computation : (2 / 3) * (3 / 4 * 40) = 20 := 
by
  -- The proof will go here, for now we use sorry to skip the proof.
  sorry

end fraction_computation_l198_198860


namespace all_statements_true_l198_198305

noncomputable def g : ℝ → ℝ := sorry

axiom g_defined (x : ℝ) : ∃ y, g x = y
axiom g_positive (x : ℝ) : g x > 0
axiom g_multiplicative (a b : ℝ) : g (a) * g (b) = g (a + b)
axiom g_div (a b : ℝ) (h : a > b) : g (a - b) = g (a) / g (b)

theorem all_statements_true :
  (g 0 = 1) ∧
  (∀ a, g (-a) = 1 / g (a)) ∧
  (∀ a, g (a) = (g (3 * a))^(1 / 3)) ∧
  (∀ a b, b > a → g (b - a) < g (b)) :=
by
  sorry

end all_statements_true_l198_198305


namespace july_16_2010_is_wednesday_l198_198838

-- Define necessary concepts for the problem

def is_tuesday (d : ℕ) : Prop := (d % 7 = 2)
def day_after_n_days (d n : ℕ) : ℕ := (d + n) % 7

-- The statement we want to prove
theorem july_16_2010_is_wednesday (h : is_tuesday 1) : day_after_n_days 1 15 = 3 := 
sorry

end july_16_2010_is_wednesday_l198_198838


namespace chess_tournament_green_teams_l198_198076

theorem chess_tournament_green_teams :
  ∀ (R G total_teams : ℕ)
  (red_team_count : ℕ → ℕ)
  (green_team_count : ℕ → ℕ)
  (mixed_team_count : ℕ → ℕ),
  R = 64 → G = 68 → total_teams = 66 →
  red_team_count R = 20 →
  (R + G = 132) →
  -- Details derived from mixed_team_count and green_team_count
  -- are inferred from the conditions provided
  mixed_team_count R + red_team_count R = 32 → 
  -- Total teams by definition including mixed teams 
  mixed_team_count G = G - (2 * red_team_count R) - green_team_count G →
  green_team_count (G - (mixed_team_count R)) = 2 → 
  2 * (green_team_count G) = 22 :=
by sorry

end chess_tournament_green_teams_l198_198076


namespace sufficiency_but_not_necessity_l198_198599

theorem sufficiency_but_not_necessity (a b : ℝ) :
  (a = 0 → a * b = 0) ∧ (a * b = 0 → a = 0) → False :=
by
   -- Proof is skipped
   sorry

end sufficiency_but_not_necessity_l198_198599


namespace hyperbolas_same_asymptotes_l198_198041

theorem hyperbolas_same_asymptotes :
  (∀ x y, (x^2 / 4 - y^2 / 9 = 1) → (∃ k, y = k * x)) →
  (∀ x y, (y^2 / 18 - x^2 / N = 1) → (∃ k, y = k * x)) →
  N = 8 :=
by sorry

end hyperbolas_same_asymptotes_l198_198041


namespace sum_of_first_3n_terms_l198_198288

theorem sum_of_first_3n_terms (S_n S_2n S_3n : ℕ) (h1 : S_n = 48) (h2 : S_2n = 60) :
  S_3n = 63 :=
by
  sorry

end sum_of_first_3n_terms_l198_198288


namespace average_age_of_cricket_team_l198_198475

theorem average_age_of_cricket_team
  (A : ℝ)
  (captain_age : ℝ) (wicket_keeper_age : ℝ)
  (team_size : ℕ) (remaining_players : ℕ)
  (captain_age_eq : captain_age = 24)
  (wicket_keeper_age_eq : wicket_keeper_age = 27)
  (remaining_players_eq : remaining_players = team_size - 2)
  (average_age_condition : (team_size * A - (captain_age + wicket_keeper_age)) = remaining_players * (A - 1)) : 
  A = 21 := by
  sorry

end average_age_of_cricket_team_l198_198475


namespace unique_line_equation_l198_198911

theorem unique_line_equation
  (k : ℝ)
  (m b : ℝ)
  (h1 : |(k^2 + 4*k + 3) - (m*k + b)| = 4)
  (h2 : 2*m + b = 8)
  (h3 : b ≠ 0) :
  (m = 6 ∧ b = -4) :=
by
  sorry

end unique_line_equation_l198_198911


namespace angle_alpha_not_2pi_over_9_l198_198897

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.cos (2 * x)) * (Real.cos (4 * x))

theorem angle_alpha_not_2pi_over_9 (α : ℝ) (h : f α = 1 / 8) : α ≠ 2 * π / 9 :=
sorry

end angle_alpha_not_2pi_over_9_l198_198897


namespace find_w_value_l198_198202

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_w_value
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : sqrt x / sqrt y - sqrt y / sqrt x = 7 / 12)
  (h2 : x - y = 7) :
  x + y = 25 := 
by
  sorry

end find_w_value_l198_198202


namespace max_m_eq_4_inequality_a_b_c_l198_198709

noncomputable def f (x : ℝ) : ℝ :=
  |x - 3| + |x + 2|

theorem max_m_eq_4 (m : ℝ) (h : ∀ x : ℝ, f x ≥ |m + 1|) : m ≤ 4 ∧ m ≥ -6 :=
  sorry

theorem inequality_a_b_c (a b c : ℝ) (h : a + 2 * b + c = 4) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a + b) + 1 / (b + c) ≥ 1 :=
  sorry

end max_m_eq_4_inequality_a_b_c_l198_198709


namespace tens_place_of_8_pow_1234_l198_198957

theorem tens_place_of_8_pow_1234 : (8^1234 / 10) % 10 = 0 := by
  sorry

end tens_place_of_8_pow_1234_l198_198957


namespace estimate_larger_than_difference_l198_198015

theorem estimate_larger_than_difference (x y z : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : z > 0) :
    (x + z) - (y - z) > x - y :=
    sorry

end estimate_larger_than_difference_l198_198015


namespace find_number_l198_198887

theorem find_number (x : ℝ) (h : 0.60 * 50 = 0.45 * x + 16.5) : x = 30 :=
by
  sorry

end find_number_l198_198887


namespace no_valid_n_for_conditions_l198_198363

theorem no_valid_n_for_conditions :
  ¬∃ n : ℕ, 1000 ≤ n / 4 ∧ n / 4 ≤ 9999 ∧ 1000 ≤ 4 * n ∧ 4 * n ≤ 9999 := by
  sorry

end no_valid_n_for_conditions_l198_198363


namespace mr_zander_total_payment_l198_198624

noncomputable def total_cost (cement_bags : ℕ) (price_per_bag : ℝ) (sand_lorries : ℕ) 
(tons_per_lorry : ℝ) (price_per_ton : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let cement_cost_before_discount := cement_bags * price_per_bag
  let discount := cement_cost_before_discount * discount_rate
  let cement_cost_after_discount := cement_cost_before_discount - discount
  let sand_cost_before_tax := sand_lorries * tons_per_lorry * price_per_ton
  let tax := sand_cost_before_tax * tax_rate
  let sand_cost_after_tax := sand_cost_before_tax + tax
  cement_cost_after_discount + sand_cost_after_tax

theorem mr_zander_total_payment :
  total_cost 500 10 20 10 40 0.05 0.07 = 13310 := 
sorry

end mr_zander_total_payment_l198_198624


namespace inequality_holds_for_n_ge_0_l198_198397

theorem inequality_holds_for_n_ge_0
  (n : ℤ)
  (h : n ≥ 0)
  (a b c x y z : ℝ)
  (Habc : 0 < a ∧ 0 < b ∧ 0 < c)
  (Hxyz : 0 < x ∧ 0 < y ∧ 0 < z)
  (Hmax : max a (max b (max c (max x (max y z)))) = a)
  (Hsum : a + b + c = x + y + z)
  (Hprod : a * b * c = x * y * z) : a^n + b^n + c^n ≥ x^n + y^n + z^n := 
sorry

end inequality_holds_for_n_ge_0_l198_198397


namespace general_term_seq_l198_198030

universe u

-- Define the sequence
def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 2 * a n + 1)

-- State the theorem
theorem general_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n, a n = 2^n - 1 :=
by
  sorry

end general_term_seq_l198_198030


namespace cube_root_simplification_l198_198176

theorem cube_root_simplification (N : ℝ) (h : N > 1) : (N^3)^(1/3) * ((N^5)^(1/3) * ((N^3)^(1/3)))^(1/3) = N^(5/3) :=
by sorry

end cube_root_simplification_l198_198176


namespace min_vertices_in_hex_grid_l198_198739

-- Define a hexagonal grid and the condition on the midpoint property.
def hexagonal_grid (p : ℤ × ℤ) : Prop :=
  ∃ m n : ℤ, p = (m, n)

-- Statement: Prove that among any 9 points in a hexagonal grid, there are two points whose midpoint is also a grid point.
theorem min_vertices_in_hex_grid :
  ∀ points : Finset (ℤ × ℤ), points.card = 9 →
  (∃ p1 p2 : (ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ 
  (∃ midpoint : ℤ × ℤ, hexagonal_grid midpoint ∧ midpoint = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2))) :=
by
  intros points h_points_card
  sorry

end min_vertices_in_hex_grid_l198_198739


namespace total_years_l198_198425

variable (T D : ℕ)
variable (Tom_years : T = 50)
variable (Devin_years : D = 25 - 5)

theorem total_years (hT : T = 50) (hD : D = 25 - 5) : T + D = 70 := by
  sorry

end total_years_l198_198425


namespace count_president_vp_secretary_l198_198591

theorem count_president_vp_secretary (total_members boys girls : ℕ) (total_members_eq : total_members = 30) 
(boys_eq : boys = 18) (girls_eq : girls = 12) :
  ∃ (ways : ℕ), 
  ways = (boys * girls * (boys - 1) + girls * boys * (girls - 1)) ∧
  ways = 6048 :=
by
  sorry

end count_president_vp_secretary_l198_198591


namespace quadratic_has_one_positive_and_one_negative_root_l198_198871

theorem quadratic_has_one_positive_and_one_negative_root
  (a : ℝ) (h₁ : a ≠ 0) (h₂ : a < -1) :
  ∃ x₁ x₂ : ℝ, (a * x₁^2 + 2 * x₁ + 1 = 0) ∧ (a * x₂^2 + 2 * x₂ + 1 = 0) ∧ (x₁ > 0) ∧ (x₂ < 0) :=
by
  sorry

end quadratic_has_one_positive_and_one_negative_root_l198_198871


namespace mixed_oil_rate_l198_198022

theorem mixed_oil_rate :
  let v₁ := 10
  let p₁ := 50
  let v₂ := 5
  let p₂ := 68
  let v₃ := 8
  let p₃ := 42
  let v₄ := 7
  let p₄ := 62
  let v₅ := 12
  let p₅ := 55
  let v₆ := 6
  let p₆ := 75
  let total_cost := v₁ * p₁ + v₂ * p₂ + v₃ * p₃ + v₄ * p₄ + v₅ * p₅ + v₆ * p₆
  let total_volume := v₁ + v₂ + v₃ + v₄ + v₅ + v₆
  let rate := total_cost / total_volume
  rate = 56.67 :=
by
  sorry

end mixed_oil_rate_l198_198022


namespace larger_jar_half_full_l198_198320

-- Defining the capacities of the jars
variables (S L W : ℚ)

-- Conditions
def equal_amount_water (S L W : ℚ) : Prop :=
  W = (1/5 : ℚ) * S ∧ W = (1/4 : ℚ) * L

-- Question: What fraction will the larger jar be filled if the water from the smaller jar is added to it?
theorem larger_jar_half_full (S L W : ℚ) (h : equal_amount_water S L W) :
  (2 * W) / L = (1 / 2 : ℚ) :=
sorry

end larger_jar_half_full_l198_198320


namespace boys_without_calculators_l198_198527

theorem boys_without_calculators :
    ∀ (total_boys students_with_calculators girls_with_calculators : ℕ),
    total_boys = 16 →
    students_with_calculators = 22 →
    girls_with_calculators = 13 →
    total_boys - (students_with_calculators - girls_with_calculators) = 7 :=
by
  intros
  sorry

end boys_without_calculators_l198_198527


namespace probability_not_rel_prime_50_l198_198381

theorem probability_not_rel_prime_50 : 
  let n := 50;
  let non_rel_primes_count := n - Nat.totient 50;
  let total_count := n;
  let probability := (non_rel_primes_count : ℚ) / (total_count : ℚ);
  probability = 3 / 5 :=
by
  sorry

end probability_not_rel_prime_50_l198_198381


namespace max_prob_games_4_choose_best_of_five_l198_198675

-- Definitions of probabilities for Team A and Team B in different game scenarios
def prob_win_deciding_game : ℝ := 0.5
def prob_A_non_deciding : ℝ := 0.6
def prob_B_non_deciding : ℝ := 0.4

-- Definitions of probabilities for different number of games in the series
def prob_xi_3 : ℝ := (prob_A_non_deciding)^3 + (prob_B_non_deciding)^3
def prob_xi_4 : ℝ := 3 * (prob_A_non_deciding^2 * prob_B_non_deciding * prob_A_non_deciding + prob_B_non_deciding^2 * prob_A_non_deciding * prob_B_non_deciding)
def prob_xi_5 : ℝ := 6 * (prob_A_non_deciding^2 * prob_B_non_deciding^2) * (2 * prob_win_deciding_game)

-- The statement that a series of 4 games has the highest probability
theorem max_prob_games_4 : prob_xi_4 > prob_xi_5 ∧ prob_xi_4 > prob_xi_3 :=
by {
  sorry
}

-- Definitions of winning probabilities in the series for Team A
def prob_A_win_best_of_3 : ℝ := (prob_A_non_deciding)^2 + 2 * (prob_A_non_deciding * prob_B_non_deciding * prob_win_deciding_game)
def prob_A_win_best_of_5 : ℝ := (prob_A_non_deciding)^3 + 3 * (prob_A_non_deciding^2 * prob_B_non_deciding) + 6 * (prob_A_non_deciding^2 * prob_B_non_deciding^2 * prob_win_deciding_game)

-- The statement that Team A has a higher chance of winning in a best-of-five series
theorem choose_best_of_five : prob_A_win_best_of_5 > prob_A_win_best_of_3 :=
by {
  sorry
}

end max_prob_games_4_choose_best_of_five_l198_198675


namespace rectangle_area_l198_198462

variable (L B : ℕ)

theorem rectangle_area :
  (L - B = 23) ∧ (2 * L + 2 * B = 166) → (L * B = 1590) :=
by
  sorry

end rectangle_area_l198_198462


namespace find_sum_of_squares_l198_198883

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- Given conditions
def condition1 : Prop := x + y = 12
def condition2 : Prop := x * y = 50

-- The statement we need to prove
theorem find_sum_of_squares (h1 : condition1 x y) (h2 : condition2 x y) : x^2 + y^2 = 44 := by
  sorry

end find_sum_of_squares_l198_198883


namespace function_relationship_profit_1200_max_profit_l198_198065

namespace SalesProblem

-- Define the linear relationship between sales quantity y and selling price x
def sales_quantity (x : ℝ) : ℝ := -2 * x + 160

-- Define the cost per item
def cost_per_item := 30

-- Define the profit given selling price x and quantity y
def profit (x : ℝ) (y : ℝ) : ℝ := (x - cost_per_item) * y

-- The given data points and conditions
def data_point_1 : (ℝ × ℝ) := (35, 90)
def data_point_2 : (ℝ × ℝ) := (40, 80)

-- Prove the linear relationship between y and x
theorem function_relationship : 
  sales_quantity data_point_1.1 = data_point_1.2 ∧ 
  sales_quantity data_point_2.1 = data_point_2.2 := 
  by sorry

-- Given daily profit of 1200, proves selling price should be 50 yuan
theorem profit_1200 (x : ℝ) (h₁ : 30 ≤ x ∧ x ≤ 54) 
  (h₂ : profit x (sales_quantity x) = 1200) : 
  x = 50 := 
  by sorry

-- Prove the maximum daily profit and corresponding selling price
theorem max_profit : 
  ∃ x, 30 ≤ x ∧ x ≤ 54 ∧ (∀ y, 30 ≤ y ∧ y ≤ 54 → profit y (sales_quantity y) ≤ profit x (sales_quantity x)) ∧ 
  profit x (sales_quantity x) = 1248 := 
  by sorry

end SalesProblem

end function_relationship_profit_1200_max_profit_l198_198065


namespace notebook_cost_l198_198817

theorem notebook_cost (s n c : ℕ) (h1 : s > 25)
                                 (h2 : n % 2 = 1)
                                 (h3 : n > 1)
                                 (h4 : c > n)
                                 (h5 : s * n * c = 2739) :
  c = 7 :=
sorry

end notebook_cost_l198_198817


namespace problem1_problem2_l198_198290

theorem problem1 : (40 * Real.sqrt 3 - 18 * Real.sqrt 3 + 8 * Real.sqrt 3) / 6 = 5 * Real.sqrt 3 := 
by sorry

theorem problem2 : (Real.sqrt 3 - 2)^2023 * (Real.sqrt 3 + 2)^2023
                 - Real.sqrt 4 * Real.sqrt (1 / 2)
                 - (Real.pi - 1)^0
                = -2 - Real.sqrt 2 :=
by sorry

end problem1_problem2_l198_198290


namespace max_handshakes_25_people_l198_198904

theorem max_handshakes_25_people : 
  (∃ n : ℕ, n = 25) → 
  (∀ p : ℕ, p ≤ 24) → 
  ∃ m : ℕ, m = 300 :=
by sorry

end max_handshakes_25_people_l198_198904


namespace peanuts_added_correct_l198_198112

-- Define the initial and final number of peanuts
def initial_peanuts : ℕ := 4
def final_peanuts : ℕ := 12

-- Define the number of peanuts Mary added
def peanuts_added : ℕ := final_peanuts - initial_peanuts

-- State the theorem that proves the number of peanuts Mary added
theorem peanuts_added_correct : peanuts_added = 8 :=
by
  -- Add the proof here
  sorry

end peanuts_added_correct_l198_198112


namespace rubies_in_treasure_l198_198544

theorem rubies_in_treasure (total_gems diamonds : ℕ) (h1 : total_gems = 5155) (h2 : diamonds = 45) : 
  total_gems - diamonds = 5110 := by
  sorry

end rubies_in_treasure_l198_198544


namespace gretchen_fewest_trips_l198_198657

def fewestTrips (total_objects : ℕ) (max_carry : ℕ) : ℕ :=
  (total_objects + max_carry - 1) / max_carry

theorem gretchen_fewest_trips : fewestTrips 17 3 = 6 := 
  sorry

end gretchen_fewest_trips_l198_198657


namespace profit_15000_l198_198663

theorem profit_15000
  (P : ℝ)
  (invest_mary : ℝ := 550)
  (invest_mike : ℝ := 450)
  (total_invest := invest_mary + invest_mike)
  (share_ratio_mary := invest_mary / total_invest)
  (share_ratio_mike := invest_mike / total_invest)
  (effort_share := P / 6)
  (invest_share_mary := share_ratio_mary * (2 * P / 3))
  (invest_share_mike := share_ratio_mike * (2 * P / 3))
  (mary_total := effort_share + invest_share_mary)
  (mike_total := effort_share + invest_share_mike)
  (condition : mary_total - mike_total = 1000) :
  P = 15000 :=  
sorry

end profit_15000_l198_198663


namespace problem1_problem2_l198_198099

/-- Problem 1 -/
theorem problem1 (a b : ℝ) : (a^2 - b)^2 = a^4 - 2 * a^2 * b + b^2 :=
by
  sorry

/-- Problem 2 -/
theorem problem2 (x : ℝ) : (2 * x + 1) * (4 * x^2 - 1) * (2 * x - 1) = 16 * x^4 - 8 * x^2 + 1 :=
by
  sorry

end problem1_problem2_l198_198099


namespace find_f_3_l198_198713

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end find_f_3_l198_198713


namespace max_n_satisfying_inequality_l198_198889

theorem max_n_satisfying_inequality : 
  ∃ (n : ℤ), 303 * n^3 ≤ 380000 ∧ ∀ m : ℤ, m > n → 303 * m^3 > 380000 := sorry

end max_n_satisfying_inequality_l198_198889


namespace uniquely_identify_figure_l198_198497

structure Figure where
  is_curve : Bool
  has_axis_of_symmetry : Bool
  has_center_of_symmetry : Bool

def Circle : Figure := { is_curve := true, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Ellipse : Figure := { is_curve := true, has_axis_of_symmetry := true, has_center_of_symmetry := false }
def Triangle : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := false }
def Square : Figure := { is_curve := false, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Rectangle : Figure := { is_curve := false, has_axis_of_symmetry := true, has_center_of_symmetry := true }
def Parallelogram : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := true }
def Trapezoid : Figure := { is_curve := false, has_axis_of_symmetry := false, has_center_of_symmetry := false }

theorem uniquely_identify_figure (figures : List Figure) (q1 q2 q3 : Figure → Bool) :
  ∀ (f : Figure), ∃! (f' : Figure), 
    q1 f' = q1 f ∧ q2 f' = q2 f ∧ q3 f' = q3 f :=
by
  sorry

end uniquely_identify_figure_l198_198497


namespace molecular_weight_of_benzene_l198_198445

def molecular_weight_C : ℝ := 12.01
def molecular_weight_H : ℝ := 1.008
def number_of_C_atoms : ℕ := 6
def number_of_H_atoms : ℕ := 6

theorem molecular_weight_of_benzene : 
  (number_of_C_atoms * molecular_weight_C + number_of_H_atoms * molecular_weight_H) = 78.108 :=
by
  sorry

end molecular_weight_of_benzene_l198_198445


namespace number_of_six_digit_palindromes_l198_198568

def is_six_digit_palindrome (n : ℕ) : Prop := 
  100000 ≤ n ∧ n ≤ 999999 ∧ (∀ a b c : ℕ, 
    n = 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a → a ≠ 0)

theorem number_of_six_digit_palindromes : 
  ∃ (count : ℕ), (count = 900 ∧ 
  ∀ n : ℕ, is_six_digit_palindrome n → true) 
:= 
by 
  use 900 
  sorry

end number_of_six_digit_palindromes_l198_198568


namespace Luke_mowing_lawns_l198_198106

theorem Luke_mowing_lawns (L : ℕ) (h1 : 18 + L = 27) : L = 9 :=
by
  sorry

end Luke_mowing_lawns_l198_198106


namespace linear_equation_a_zero_l198_198616

theorem linear_equation_a_zero (a : ℝ) : 
  ((a - 2) * x ^ (abs (a - 1)) + 3 = 9) ∧ (abs (a - 1) = 1) → a = 0 := by
  sorry

end linear_equation_a_zero_l198_198616


namespace intersecting_chords_length_l198_198111

theorem intersecting_chords_length
  (h1 : ∃ c1 c2 : ℝ, c1 = 8 ∧ c2 = x + 4 * x ∧ x = 2)
  (h2 : ∀ (a b c d : ℝ), a * b = c * d → a = 4 ∧ b = 4 ∧ c = x ∧ d = 4 * x ∧ x = 2):
  (10 : ℝ) = (x + 4 * x) := by
  sorry

end intersecting_chords_length_l198_198111


namespace max_value_exponent_l198_198165

theorem max_value_exponent {a b : ℝ} (h : 0 < b ∧ b < a ∧ a < 1) :
  max (max (a^b) (b^a)) (max (a^a) (b^b)) = a^b :=
sorry

end max_value_exponent_l198_198165


namespace inequality_holds_for_positive_x_l198_198812

theorem inequality_holds_for_positive_x (x : ℝ) (h : x > 0) : 
  x^8 - x^5 - 1/x + 1/(x^4) ≥ 0 := 
sorry

end inequality_holds_for_positive_x_l198_198812


namespace total_weeds_correct_l198_198173

def tuesday : ℕ := 25
def wednesday : ℕ := 3 * tuesday
def thursday : ℕ := wednesday / 5
def friday : ℕ := thursday - 10
def total_weeds : ℕ := tuesday + wednesday + thursday + friday

theorem total_weeds_correct : total_weeds = 120 :=
by
  sorry

end total_weeds_correct_l198_198173


namespace price_decrease_percentage_l198_198686

-- Definitions based on given conditions
def price_in_2007 (x : ℝ) : ℝ := x
def price_in_2008 (x : ℝ) : ℝ := 1.25 * x
def desired_price_in_2009 (x : ℝ) : ℝ := 1.1 * x

-- Theorem statement to prove the price decrease from 2008 to 2009
theorem price_decrease_percentage (x : ℝ) (h : x > 0) : 
  (1.25 * x - 1.1 * x) / (1.25 * x) = 0.12 := 
sorry

end price_decrease_percentage_l198_198686


namespace decreasing_function_solution_set_l198_198369

theorem decreasing_function_solution_set {f : ℝ → ℝ} (h : ∀ x y, x < y → f y < f x) :
  {x : ℝ | f 2 < f (2*x + 1)} = {x : ℝ | x < 1/2} :=
by
  sorry

end decreasing_function_solution_set_l198_198369


namespace quadratic_equation_root_conditions_quadratic_equation_distinct_real_roots_l198_198640

theorem quadratic_equation_root_conditions
  (k : ℝ)
  (h_discriminant : 4 * k - 3 > 0)
  (h_sum_product : ∀ (x1 x2 : ℝ),
    x1 + x2 = -(2 * k + 1) ∧ 
    x1 * x2 = k^2 + 1 →
    x1 + x2 + 2 * (x1 * x2) = 1) :
  k = 1 :=
by
  sorry

theorem quadratic_equation_distinct_real_roots
  (k : ℝ) :
  (∃ (x1 x2 : ℝ),
    x1 ≠ x2 ∧
    x1^2 + (2 * k + 1) * x1 + (k^2 + 1) = 0 ∧
    x2^2 + (2 * k + 1) * x2 + (k^2 + 1) = 0) ↔
  k > 3 / 4 :=
by
  sorry

end quadratic_equation_root_conditions_quadratic_equation_distinct_real_roots_l198_198640


namespace price_of_table_l198_198768

variable (C T : ℝ)

theorem price_of_table :
  2 * C + T = 0.6 * (C + 2 * T) ∧
  C + T = 96 →
  T = 84 := by
sorry

end price_of_table_l198_198768


namespace solution_set_of_x_squared_lt_one_l198_198206

theorem solution_set_of_x_squared_lt_one : {x : ℝ | x^2 < 1} = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end solution_set_of_x_squared_lt_one_l198_198206


namespace students_taking_music_l198_198604

theorem students_taking_music
  (total_students : Nat)
  (students_taking_art : Nat)
  (students_taking_both : Nat)
  (students_taking_neither : Nat)
  (total_eq : total_students = 500)
  (art_eq : students_taking_art = 20)
  (both_eq : students_taking_both = 10)
  (neither_eq : students_taking_neither = 440) :
  ∃ M : Nat, M = 50 := by
  sorry

end students_taking_music_l198_198604


namespace hours_learning_english_each_day_l198_198962

theorem hours_learning_english_each_day (E : ℕ) 
  (h_chinese_each_day : ∀ (d : ℕ), d = 7) 
  (days : ℕ) 
  (h_total_days : days = 5) 
  (h_total_hours : ∀ (t : ℕ), t = 65) 
  (total_learning_time : 5 * (E + 7) = 65) :
  E = 6 :=
by
  sorry

end hours_learning_english_each_day_l198_198962


namespace find_A_for_club_suit_l198_198400

def club_suit (A B : ℝ) : ℝ := 3 * A + 2 * B^2 + 5

theorem find_A_for_club_suit :
  ∃ A : ℝ, club_suit A 3 = 73 ∧ A = 50 / 3 :=
sorry

end find_A_for_club_suit_l198_198400


namespace eq1_solution_eq2_solution_l198_198979


-- Theorem for the first equation (4(x + 1)^2 - 25 = 0)
theorem eq1_solution (x : ℝ) : (4 * (x + 1)^2 - 25 = 0) ↔ (x = 3 / 2 ∨ x = -7 / 2) :=
by
  sorry

-- Theorem for the second equation ((x + 10)^3 = -125)
theorem eq2_solution (x : ℝ) : ((x + 10)^3 = -125) ↔ (x = -15) :=
by
  sorry

end eq1_solution_eq2_solution_l198_198979


namespace reciprocals_sum_l198_198370

theorem reciprocals_sum (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * a * b) : 
  (1 / a) + (1 / b) = 6 := 
sorry

end reciprocals_sum_l198_198370


namespace problem1_problem2_l198_198136

-- Problem 1
theorem problem1 (x : ℝ) : (1 : ℝ) * (-2 * x^2)^3 + x^2 * x^4 - (-3 * x^3)^2 = -16 * x^6 := 
sorry

-- Problem 2
theorem problem2 (a b : ℝ) : (a - b)^2 * (b - a)^4 + (b - a)^3 * (a - b)^3 = 0 := 
sorry

end problem1_problem2_l198_198136


namespace total_games_proof_l198_198802

def num_teams : ℕ := 20
def num_games_per_team_regular_season : ℕ := 38
def total_regular_season_games : ℕ := num_teams * (num_games_per_team_regular_season / 2)
def num_games_per_team_mid_season : ℕ := 3
def total_mid_season_games : ℕ := num_teams * num_games_per_team_mid_season
def quarter_finals_teams : ℕ := 8
def quarter_finals_matchups : ℕ := quarter_finals_teams / 2
def quarter_finals_games : ℕ := quarter_finals_matchups * 2
def semi_finals_teams : ℕ := quarter_finals_matchups
def semi_finals_matchups : ℕ := semi_finals_teams / 2
def semi_finals_games : ℕ := semi_finals_matchups * 2
def final_teams : ℕ := semi_finals_matchups
def final_games : ℕ := final_teams * 2
def total_playoff_games : ℕ := quarter_finals_games + semi_finals_games + final_games

def total_season_games : ℕ := total_regular_season_games + total_mid_season_games + total_playoff_games

theorem total_games_proof : total_season_games = 454 := by
  -- The actual proof will go here
  sorry

end total_games_proof_l198_198802


namespace average_minutes_run_per_day_l198_198412

theorem average_minutes_run_per_day (e : ℕ)
  (sixth_grade_avg : ℕ := 16)
  (seventh_grade_avg : ℕ := 18)
  (eighth_grade_avg : ℕ := 12)
  (sixth_graders : ℕ := 3 * e)
  (seventh_graders : ℕ := 2 * e)
  (eighth_graders : ℕ := e) :
  ((sixth_grade_avg * sixth_graders + seventh_grade_avg * seventh_graders + eighth_grade_avg * eighth_graders)
   / (sixth_graders + seventh_graders + eighth_graders) : ℕ) = 16 := 
by
  sorry

end average_minutes_run_per_day_l198_198412


namespace regular_polygon_enclosure_l198_198728

theorem regular_polygon_enclosure (m n : ℕ) (h : m = 12)
    (h_enc : ∀ p : ℝ, p = 360 / ↑n → (2 * (180 / ↑n)) = (360 / ↑m)) :
    n = 12 :=
by
  sorry

end regular_polygon_enclosure_l198_198728


namespace cost_of_one_jacket_l198_198854

theorem cost_of_one_jacket
  (S J : ℝ)
  (h1 : 10 * S + 20 * J = 800)
  (h2 : 5 * S + 15 * J = 550) : J = 30 :=
sorry

end cost_of_one_jacket_l198_198854


namespace sum_first_five_special_l198_198564

def is_special (n : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p^2 * q^2

theorem sum_first_five_special :
  let special_numbers := [36, 100, 196, 484, 676]
  (∀ n ∈ special_numbers, is_special n) →
  special_numbers.sum = 1492 := by {
  sorry
}

end sum_first_five_special_l198_198564


namespace incorrect_statement_l198_198760

noncomputable def function_y (x : ℝ) : ℝ := 4 / x

theorem incorrect_statement (x : ℝ) (hx : x ≠ 0) : ¬(∀ x1 x2 : ℝ, (hx1 : x1 ≠ 0) → (hx2 : x2 ≠ 0) → x1 < x2 → function_y x1 > function_y x2) := 
sorry

end incorrect_statement_l198_198760


namespace ratio_of_triangle_side_to_rectangle_width_l198_198818

theorem ratio_of_triangle_side_to_rectangle_width
  (t w : ℕ)
  (ht : 3 * t = 24)
  (hw : 6 * w = 24) :
  t / w = 2 := by
  sorry

end ratio_of_triangle_side_to_rectangle_width_l198_198818


namespace option_C_correct_l198_198906

theorem option_C_correct : (Real.sqrt 2) * (Real.sqrt 6) = 2 * (Real.sqrt 3) :=
by sorry

end option_C_correct_l198_198906


namespace sequence_expression_l198_198135

-- Given conditions
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)
variable (h1 : ∀ n, S n = (1/4) * (a n + 1)^2)

-- Theorem statement
theorem sequence_expression (n : ℕ) : a n = 2 * n - 1 :=
sorry

end sequence_expression_l198_198135


namespace min_knights_in_village_l198_198705

theorem min_knights_in_village :
  ∃ (K L : ℕ), K + L = 7 ∧ 2 * K * L = 24 ∧ K ≥ 3 :=
by
  sorry

end min_knights_in_village_l198_198705


namespace parking_space_length_l198_198597

theorem parking_space_length {L W : ℕ} 
  (h1 : 2 * W + L = 37) 
  (h2 : L * W = 126) : 
  L = 9 := 
sorry

end parking_space_length_l198_198597


namespace total_volume_of_five_cubes_l198_198562

-- Definition for volume of a cube function
def volume_of_cube (edge_length : ℝ) : ℝ :=
  edge_length ^ 3

-- Conditions
def edge_length : ℝ := 5
def number_of_cubes : ℝ := 5

-- Proof statement
theorem total_volume_of_five_cubes : 
  volume_of_cube edge_length * number_of_cubes = 625 := 
by
  sorry

end total_volume_of_five_cubes_l198_198562


namespace inequality_sqrt_l198_198593

theorem inequality_sqrt (m n : ℕ) (h : m < n) : 
  (m^2 + Real.sqrt (m^2 + m) < n^2 - Real.sqrt (n^2 - n)) :=
by
  sorry

end inequality_sqrt_l198_198593


namespace storybook_pages_l198_198270

theorem storybook_pages :
  (10 + 5) / (1 - (1 / 5) * 2) = 25 := by
  sorry

end storybook_pages_l198_198270


namespace combined_money_half_l198_198189

theorem combined_money_half
  (J S : ℚ)
  (h1 : J = S)
  (h2 : J - (3/7 * J + 2/5 * J + 1/4 * J) = 24)
  (h3 : S - (1/2 * S + 1/3 * S) = 36) :
  1.5 * J = 458.18 := 
by
  sorry

end combined_money_half_l198_198189


namespace parabola_directrix_l198_198081

theorem parabola_directrix (x : ℝ) (y : ℝ) (h : y = -4 * x ^ 2 - 3) : y = - 49 / 16 := sorry

end parabola_directrix_l198_198081


namespace sector_angle_l198_198703

theorem sector_angle (r : ℝ) (θ : ℝ) 
  (area_eq : (1 / 2) * θ * r^2 = 1)
  (perimeter_eq : 2 * r + θ * r = 4) : θ = 2 := 
by
  sorry

end sector_angle_l198_198703


namespace option_B_correct_option_C_correct_l198_198574

-- Define the permutation coefficient
def A (n m : ℕ) : ℕ := n * (n-1) * (n-2) * (n-m+1)

-- Prove the equation for option B
theorem option_B_correct (n m : ℕ) : A (n+1) (m+1) - A n m = n^2 * A (n-1) (m-1) :=
by
  sorry

-- Prove the equation for option C
theorem option_C_correct (n m : ℕ) : A n m = n * A (n-1) (m-1) :=
by
  sorry

end option_B_correct_option_C_correct_l198_198574


namespace infinite_series_closed_form_l198_198970

noncomputable def series (a : ℝ) : ℝ :=
  ∑' (k : ℕ), (2 * (k + 1) - 1) / a^k

theorem infinite_series_closed_form (a : ℝ) (ha : 1 < a) : 
  series a = (a^2 + a) / (a - 1)^2 :=
sorry

end infinite_series_closed_form_l198_198970


namespace scientific_notation_of_12_06_million_l198_198359

theorem scientific_notation_of_12_06_million :
  12.06 * 10^6 = 1.206 * 10^7 :=
sorry

end scientific_notation_of_12_06_million_l198_198359


namespace seq_sum_l198_198326

theorem seq_sum (r : ℚ) (x y : ℚ) (h1 : r = 1 / 4)
    (h2 : 1024 * r = x) (h3 : x * r = y) : 
    x + y = 320 := by
  sorry

end seq_sum_l198_198326


namespace find_positive_integers_l198_198901

theorem find_positive_integers (n : ℕ) : 
  (∀ a : ℕ, a.gcd n = 1 → 2 * n * n ∣ a ^ n - 1) ↔ (n = 2 ∨ n = 6 ∨ n = 42 ∨ n = 1806) :=
sorry

end find_positive_integers_l198_198901


namespace solution_set_of_inequality_l198_198283

theorem solution_set_of_inequality (x : ℝ) : x * (9 - x) > 0 ↔ 0 < x ∧ x < 9 := by
  sorry

end solution_set_of_inequality_l198_198283


namespace find_norm_b_projection_of_b_on_a_l198_198049

open Real EuclideanSpace

noncomputable def a : ℝ := 4

noncomputable def angle_ab : ℝ := π / 4  -- 45 degrees in radians

noncomputable def inner_prod_condition (b : ℝ) : ℝ := 
  (1 / 2 * a) * (2 * a) + 
  (1 / 2 * a) * (-3 * b) + 
  b * (2 * a) + 
  b * (-3 * b) - 12

theorem find_norm_b (b : ℝ) (hb : inner_prod_condition b = 0) : b = sqrt 2 :=
  sorry

theorem projection_of_b_on_a (b : ℝ) (hb : inner_prod_condition b = 0) : 
  (b * cos angle_ab) = 1 :=
  sorry

end find_norm_b_projection_of_b_on_a_l198_198049


namespace lions_at_sanctuary_l198_198267

variable (L C : ℕ)

noncomputable def is_solution :=
  C = 1 / 2 * (L + 14) ∧
  L + 14 + C = 39 ∧
  L = 12

theorem lions_at_sanctuary : is_solution L C :=
sorry

end lions_at_sanctuary_l198_198267


namespace fraction_historical_fiction_new_releases_l198_198433

-- Define constants for book categories and new releases
def historical_fiction_percentage : ℝ := 0.40
def science_fiction_percentage : ℝ := 0.25
def biographies_percentage : ℝ := 0.15
def mystery_novels_percentage : ℝ := 0.20

def historical_fiction_new_releases : ℝ := 0.45
def science_fiction_new_releases : ℝ := 0.30
def biographies_new_releases : ℝ := 0.50
def mystery_novels_new_releases : ℝ := 0.35

-- Statement of the problem to prove
theorem fraction_historical_fiction_new_releases :
  (historical_fiction_percentage * historical_fiction_new_releases) /
    (historical_fiction_percentage * historical_fiction_new_releases +
     science_fiction_percentage * science_fiction_new_releases +
     biographies_percentage * biographies_new_releases +
     mystery_novels_percentage * mystery_novels_new_releases) = 9 / 20 :=
by
  sorry

end fraction_historical_fiction_new_releases_l198_198433


namespace angle_in_fourth_quadrant_l198_198971

theorem angle_in_fourth_quadrant (θ : ℝ) (h : θ = -1445) : (θ % 360) > 270 ∧ (θ % 360) < 360 :=
by
  sorry

end angle_in_fourth_quadrant_l198_198971


namespace percentage_increase_of_y_over_x_l198_198552

variable (x y : ℝ) (h : x > 0 ∧ y > 0) 

theorem percentage_increase_of_y_over_x
  (h_ratio : (x / 8) = (y / 7)) :
  ((y - x) / x) * 100 = 12.5 := 
sorry

end percentage_increase_of_y_over_x_l198_198552


namespace distance_A_to_C_through_B_l198_198008

-- Define the distances on the map
def Distance_AB_map : ℝ := 20
def Distance_BC_map : ℝ := 10

-- Define the scale of the map
def scale : ℝ := 5

-- Define the actual distances
def Distance_AB := Distance_AB_map * scale
def Distance_BC := Distance_BC_map * scale

-- Define the total distance from A to C through B
def Distance_AC_through_B := Distance_AB + Distance_BC

-- Theorem to be proved
theorem distance_A_to_C_through_B : Distance_AC_through_B = 150 := by
  sorry

end distance_A_to_C_through_B_l198_198008


namespace increasing_iff_a_le_0_l198_198275

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - a * x + 1

theorem increasing_iff_a_le_0 : (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 0 :=
by
  sorry

end increasing_iff_a_le_0_l198_198275


namespace tangent_lines_through_point_l198_198722

theorem tangent_lines_through_point (x y : ℝ) (hp : (x, y) = (3, 1))
 : ∃ (a b c : ℝ), (y - 1 = (4 / 3) * (x - 3) ∨ x = 3) :=
by
  sorry

end tangent_lines_through_point_l198_198722


namespace odds_against_C_l198_198666

def odds_against_winning (p : ℚ) : ℚ := (1 - p) / p

theorem odds_against_C (pA pB pC : ℚ) (hA : pA = 1 / 3) (hB : pB = 1 / 5) (hC : pC = 7 / 15) :
  odds_against_winning pC = 8 / 7 :=
by
  -- Definitions based on the conditions provided in a)
  have h1 : odds_against_winning (1/3) = 2 := by sorry
  have h2 : odds_against_winning (1/5) = 4 := by sorry

  -- Odds against C
  have h3 : 1 - (pA + pB) = pC := by sorry
  have h4 : pA + pB = 8 / 15 := by sorry

  -- Show that odds against C winning is 8/7
  have h5 : odds_against_winning pC = 8 / 7 := by sorry
  exact h5

end odds_against_C_l198_198666


namespace radical_multiplication_l198_198557

noncomputable def root4 (x : ℝ) : ℝ := x ^ (1/4)
noncomputable def root3 (x : ℝ) : ℝ := x ^ (1/3)
noncomputable def root2 (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_multiplication : root4 256 * root3 8 * root2 16 = 32 := by
  sorry

end radical_multiplication_l198_198557


namespace central_angle_l198_198918

variable (O : Type)
variable (A B C : O)
variable (angle_ABC : ℝ) 

theorem central_angle (h : angle_ABC = 50) : 2 * angle_ABC = 100 := by
  sorry

end central_angle_l198_198918


namespace total_fireworks_l198_198832

-- Definitions based on conditions
def kobys_boxes := 2
def kobys_sparklers_per_box := 3
def kobys_whistlers_per_box := 5
def cheries_boxes := 1
def cheries_sparklers_per_box := 8
def cheries_whistlers_per_box := 9

-- Calculations
def total_kobys_fireworks := kobys_boxes * (kobys_sparklers_per_box + kobys_whistlers_per_box)
def total_cheries_fireworks := cheries_boxes * (cheries_sparklers_per_box + cheries_whistlers_per_box)

-- Theorem
theorem total_fireworks : total_kobys_fireworks + total_cheries_fireworks = 33 := 
by
  -- Can be elaborated and filled in with steps, if necessary.
  sorry

end total_fireworks_l198_198832


namespace trigonometric_identity_l198_198807

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.cos (2 * α) + Real.sin (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α) = -1 :=
by
  sorry

end trigonometric_identity_l198_198807


namespace monochromatic_triangle_in_K6_l198_198403

theorem monochromatic_triangle_in_K6 :
  ∀ (color : Fin 6 → Fin 6 → Prop),
  (∀ (a b : Fin 6), a ≠ b → (color a b ↔ color b a)) →
  (∃ (x y z : Fin 6), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (color x y = color y z ∧ color y z = color z x)) :=
by
  sorry

end monochromatic_triangle_in_K6_l198_198403


namespace parcels_division_l198_198569

theorem parcels_division (x y n : ℕ) (h : 5 + 2 * x + 3 * y = 4 * n) (hn : n = x + y) :
    n = 3 ∨ n = 4 ∨ n = 5 := 
sorry

end parcels_division_l198_198569


namespace expand_expression_l198_198084

theorem expand_expression (x : ℝ) : (7 * x + 5) * (3 * x^2) = 21 * x^3 + 15 * x^2 :=
by
  sorry

end expand_expression_l198_198084


namespace tens_digit_of_23_pow_2057_l198_198450

theorem tens_digit_of_23_pow_2057 : (23^2057 % 100) / 10 % 10 = 6 := 
by
  sorry

end tens_digit_of_23_pow_2057_l198_198450


namespace number_of_tables_l198_198186

noncomputable def stools_per_table : ℕ := 7
noncomputable def legs_per_stool : ℕ := 4
noncomputable def legs_per_table : ℕ := 5
noncomputable def total_legs : ℕ := 658

theorem number_of_tables : 
  ∃ t : ℕ, 
  (∃ s : ℕ, s = stools_per_table * t ∧ legs_per_stool * s + legs_per_table * t = total_legs) ∧ t = 20 :=
by {
  sorry
}

end number_of_tables_l198_198186


namespace time_to_pass_jogger_l198_198064

noncomputable def jogger_speed_kmh := 9 -- in km/hr
noncomputable def train_speed_kmh := 45 -- in km/hr
noncomputable def jogger_headstart_m := 240 -- in meters
noncomputable def train_length_m := 100 -- in meters

noncomputable def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

noncomputable def jogger_speed_mps := kmh_to_mps jogger_speed_kmh
noncomputable def train_speed_mps := kmh_to_mps train_speed_kmh
noncomputable def relative_speed := train_speed_mps - jogger_speed_mps
noncomputable def distance_to_be_covered := jogger_headstart_m + train_length_m

theorem time_to_pass_jogger : distance_to_be_covered / relative_speed = 34 := by
  sorry

end time_to_pass_jogger_l198_198064


namespace greene_family_total_spent_l198_198952

def adm_cost : ℕ := 45

def food_cost : ℕ := adm_cost - 13

def total_cost : ℕ := adm_cost + food_cost

theorem greene_family_total_spent : total_cost = 77 := 
by 
  sorry

end greene_family_total_spent_l198_198952


namespace train_B_speed_l198_198137

noncomputable def train_speed_B (V_A : ℕ) (T_A : ℕ) (T_B : ℕ) : ℕ :=
  V_A * T_A / T_B

theorem train_B_speed
  (V_A : ℕ := 60)
  (T_A : ℕ := 9)
  (T_B : ℕ := 4) :
  train_speed_B V_A T_A T_B = 135 := 
by
  sorry

end train_B_speed_l198_198137


namespace min_fence_dimensions_l198_198988

theorem min_fence_dimensions (A : ℝ) (hA : A ≥ 800) (x : ℝ) (hx : 2 * x * x = A) : x = 20 ∧ 2 * x = 40 := by
  sorry

end min_fence_dimensions_l198_198988


namespace f_zero_derivative_not_extremum_l198_198139

noncomputable def f (x : ℝ) : ℝ := x ^ 3

theorem f_zero_derivative_not_extremum (x : ℝ) : 
  deriv f 0 = 0 ∧ ∀ (y : ℝ), y ≠ 0 → (∃ δ > 0, ∀ z, abs (z - 0) < δ → (f z / z : ℝ) ≠ 0) :=
by
  sorry

end f_zero_derivative_not_extremum_l198_198139


namespace find_number_l198_198700

-- Given conditions
variables (x y : ℕ)

-- The conditions from the problem statement
def digit_sum : Prop := x + y = 12
def reverse_condition : Prop := (10 * x + y) + 36 = 10 * y + x

-- The final statement
theorem find_number (h1 : digit_sum x y) (h2 : reverse_condition x y) : 10 * x + y = 48 :=
sorry

end find_number_l198_198700


namespace num_proper_subsets_of_A_l198_198443

open Set

def A : Finset ℕ := {2, 3}

theorem num_proper_subsets_of_A : (A.powerset \ {A, ∅}).card = 3 := by
  sorry

end num_proper_subsets_of_A_l198_198443


namespace john_twice_as_old_in_x_years_l198_198295

def frank_is_younger (john_age frank_age : ℕ) : Prop :=
  frank_age = john_age - 15

def frank_future_age (frank_age : ℕ) : ℕ :=
  frank_age + 4

def john_future_age (john_age : ℕ) : ℕ :=
  john_age + 4

theorem john_twice_as_old_in_x_years (john_age frank_age x : ℕ) 
  (h1 : frank_is_younger john_age frank_age)
  (h2 : frank_future_age frank_age = 16)
  (h3 : john_age = frank_age + 15) :
  (john_age + x) = 2 * (frank_age + x) → x = 3 :=
by 
  -- Skip the proof part
  sorry

end john_twice_as_old_in_x_years_l198_198295


namespace bus_problem_l198_198316

-- Define the participants in 2005
def participants_2005 (k : ℕ) : ℕ := 27 * k + 19

-- Define the participants in 2006
def participants_2006 (k : ℕ) : ℕ := participants_2005 k + 53

-- Define the total number of buses needed in 2006
def buses_needed_2006 (k : ℕ) : ℕ := (participants_2006 k) / 27 + if (participants_2006 k) % 27 = 0 then 0 else 1

-- Define the total number of buses needed in 2005
def buses_needed_2005 (k : ℕ) : ℕ := k + 1

-- Define the additional buses needed in 2006 compared to 2005
def additional_buses_2006 (k : ℕ) := buses_needed_2006 k - buses_needed_2005 k

-- Define the number of people in the incomplete bus in 2006
def people_in_incomplete_bus_2006 (k : ℕ) := (participants_2006 k) % 27

-- The proof statement to be proved
theorem bus_problem (k : ℕ) : additional_buses_2006 k = 2 ∧ people_in_incomplete_bus_2006 k = 9 := by
  sorry

end bus_problem_l198_198316


namespace largest_even_number_in_series_l198_198424

/-- 
  If the sum of 25 consecutive even numbers is 10,000,
  what is the largest number among these 25 consecutive even numbers? 
-/
theorem largest_even_number_in_series (n : ℤ) (S : ℤ) (h : S = 25 * (n - 24)) (h_sum : S = 10000) :
  n = 424 :=
by {
  sorry -- proof goes here
}

end largest_even_number_in_series_l198_198424


namespace percentage_in_first_subject_l198_198240

theorem percentage_in_first_subject (P : ℝ) (H1 : 80 = 80) (H2 : 75 = 75) (H3 : (P + 80 + 75) / 3 = 75) : P = 70 :=
by
  sorry

end percentage_in_first_subject_l198_198240


namespace num_children_with_identical_cards_l198_198225

theorem num_children_with_identical_cards (children_mama children_nyanya children_manya total_children mixed_cards : ℕ) 
  (h_mama: children_mama = 20) 
  (h_nyanya: children_nyanya = 30) 
  (h_manya: children_manya = 40) 
  (h_total: total_children = children_mama + children_nyanya) 
  (h_mixed: mixed_cards = children_manya) 
  : total_children - children_manya = 10 :=
by
  -- Sorry to indicate the proof is skipped
  sorry

end num_children_with_identical_cards_l198_198225


namespace speed_with_current_l198_198393

theorem speed_with_current (v : ℝ) (current_speed : ℝ) (against_current_speed : ℝ) 
  (h1 : current_speed = 2.8) 
  (h2 : against_current_speed = 9.4) 
  (h3 : against_current_speed = v - current_speed) 
  : (v + current_speed) = 15 := by
  sorry

end speed_with_current_l198_198393


namespace sum_original_and_correct_value_l198_198006

theorem sum_original_and_correct_value (x : ℕ) (h : x + 14 = 68) :
  x + (x + 41) = 149 := by
  sorry

end sum_original_and_correct_value_l198_198006


namespace average_visitors_on_Sundays_l198_198981

theorem average_visitors_on_Sundays (S : ℕ) (h1 : 30 = 5 + 25) (h2 : 25 * 240 + 5 * S = 30 * 285) :
  S = 510 := sorry

end average_visitors_on_Sundays_l198_198981


namespace area_of_T_prime_l198_198153

-- Given conditions
def AreaBeforeTransformation : ℝ := 9

def TransformationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4],![-2, 5]]

def AreaAfterTransformation (M : Matrix (Fin 2) (Fin 2) ℝ) (area_before : ℝ) : ℝ :=
  (M.det) * area_before

-- Problem statement
theorem area_of_T_prime : 
  AreaAfterTransformation TransformationMatrix AreaBeforeTransformation = 207 :=
by
  sorry

end area_of_T_prime_l198_198153


namespace sum_geometric_series_is_correct_l198_198702

def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_geometric_series_is_correct
  (a r : ℚ) (n : ℕ)
  (h_a : a = 1/4)
  (h_r : r = 1/4)
  (h_n : n = 5) :
  geometric_series_sum a r n = 341 / 1024 :=
by
  rw [h_a, h_r, h_n]
  -- Now we can skip the proof.
  sorry

end sum_geometric_series_is_correct_l198_198702


namespace matchsticks_for_3_by_1996_grid_l198_198737

def total_matchsticks_needed (rows cols : ℕ) : ℕ :=
  (cols * (rows + 1)) + (rows * (cols + 1))

theorem matchsticks_for_3_by_1996_grid : total_matchsticks_needed 3 1996 = 13975 := by
  sorry

end matchsticks_for_3_by_1996_grid_l198_198737


namespace jackson_sandwiches_l198_198519

theorem jackson_sandwiches (weeks : ℕ) (missed_wednesdays : ℕ) (missed_fridays : ℕ)
    (h_weeks : weeks = 36) (h_missed_wednesdays : missed_wednesdays = 1) (h_missed_fridays : missed_fridays = 2) :
    let total_days := weeks * 2
    let missed_days := missed_wednesdays + missed_fridays
    total_days - missed_days = 69 :=
by
    sorry

end jackson_sandwiches_l198_198519


namespace angle_between_adjacent_triangles_l198_198033

-- Define the setup of the problem
def five_nonoverlapping_equilateral_triangles (angles : Fin 5 → ℝ) :=
  ∀ i, angles i = 60

def angles_between_adjacent_triangles (angles : Fin 5 → ℝ) :=
  ∀ i j, i ≠ j → angles i = angles j

-- State the main theorem
theorem angle_between_adjacent_triangles :
  ∀ (angles : Fin 5 → ℝ),
    five_nonoverlapping_equilateral_triangles angles →
    angles_between_adjacent_triangles angles →
    ((360 - 5 * 60) / 5) = 12 :=
by
  intros angles h1 h2
  sorry

end angle_between_adjacent_triangles_l198_198033


namespace average_income_of_all_customers_l198_198216

theorem average_income_of_all_customers
  (n m : ℕ) 
  (a b : ℝ) 
  (customers_responded : n = 50) 
  (wealthiest_count : m = 10) 
  (other_customers_count : n - m = 40) 
  (wealthiest_avg_income : a = 55000) 
  (other_avg_income : b = 42500) : 
  (m * a + (n - m) * b) / n = 45000 := 
by
  -- transforming given conditions into useful expressions
  have h1 : m = 10 := by assumption
  have h2 : n = 50 := by assumption
  have h3 : n - m = 40 := by assumption
  have h4 : a = 55000 := by assumption
  have h5 : b = 42500 := by assumption
  sorry

end average_income_of_all_customers_l198_198216


namespace tetrahedron_formable_l198_198596

theorem tetrahedron_formable (x : ℝ) (hx_pos : 0 < x) (hx_bound : x < (Real.sqrt 6 + Real.sqrt 2) / 2) :
  true := 
sorry

end tetrahedron_formable_l198_198596


namespace investor_share_price_l198_198782

theorem investor_share_price (dividend_rate : ℝ) (face_value : ℝ) (roi : ℝ) (price_per_share : ℝ) : 
  dividend_rate = 0.125 →
  face_value = 40 →
  roi = 0.25 →
  ((dividend_rate * face_value) / price_per_share) = roi →
  price_per_share = 20 :=
by 
  intros h1 h2 h3 h4
  sorry

end investor_share_price_l198_198782


namespace max_competitors_l198_198880

theorem max_competitors (P1 P2 P3 : ℕ → ℕ → ℕ)
(hP1 : ∀ i, 0 ≤ P1 i ∧ P1 i ≤ 7)
(hP2 : ∀ i, 0 ≤ P2 i ∧ P2 i ≤ 7)
(hP3 : ∀ i, 0 ≤ P3 i ∧ P3 i ≤ 7)
(hDistinct : ∀ i j, i ≠ j → (P1 i ≠ P1 j ∨ P2 i ≠ P2 j ∨ P3 i ≠ P3 j)) :
  ∃ n, n ≤ 64 ∧ ∀ k, k < n → (∀ i j, i < k → j < k → i ≠ j → (P1 i ≠ P1 j ∨ P2 i ≠ P2 j ∨ P3 i ≠ P3 j)) :=
sorry

end max_competitors_l198_198880


namespace minimum_value_expr_eq_neg6680_25_l198_198983

noncomputable def expr (x : ℝ) : ℝ := (15 - x) * (8 - x) * (15 + x) * (8 + x) - 200

theorem minimum_value_expr_eq_neg6680_25 : ∃ x : ℝ, (∀ y : ℝ, expr y ≥ expr x) ∧ expr x = -6680.25 :=
sorry

end minimum_value_expr_eq_neg6680_25_l198_198983


namespace acute_triangle_angle_A_range_of_bc_l198_198743

-- Definitions
variables {A B C : ℝ} {a b c : ℝ}
variable (Δ : ∃ (A B C : ℝ), a = sqrt 2 ∧ ∀ (a b c A B C : ℝ), 
  (a = sqrt 2) ∧ (b = b) ∧ (c = c) ∧ 
  (sin A * cos A / cos (A + C) = a * c / (b^2 - a^2 - c^2)))

-- Problem statement
theorem acute_triangle_angle_A (h : Δ) : A = π / 4 :=
sorry

theorem range_of_bc (h : Δ) : 0 < b * c ∧ b * c ≤ 2 + sqrt 2 :=
sorry

end acute_triangle_angle_A_range_of_bc_l198_198743


namespace abc_unique_l198_198636

theorem abc_unique (n : ℕ) (hn : 0 < n) (p : ℕ) (hp : Nat.Prime p) 
                   (a b c : ℤ) 
                   (h : a^n + p * b = b^n + p * c ∧ b^n + p * c = c^n + p * a) 
                   : a = b ∧ b = c :=
by
  sorry

end abc_unique_l198_198636


namespace option_a_is_odd_l198_198300

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem option_a_is_odd (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_odd (a + 2 * b + 1) :=
by sorry

end option_a_is_odd_l198_198300


namespace trigonometric_identity_l198_198813

theorem trigonometric_identity (α : ℝ) (h : Real.sin (π / 4 + α) = 1 / 2) : 
  (Real.sin (5 * π / 4 + α) / Real.cos (9 * π / 4 + α)) * Real.cos (7 * π / 4 - α) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l198_198813


namespace find_m_plus_n_l198_198123

def num_fir_trees : ℕ := 4
def num_pine_trees : ℕ := 5
def num_acacia_trees : ℕ := 6

def num_non_acacia_trees : ℕ := num_fir_trees + num_pine_trees
def total_trees : ℕ := num_fir_trees + num_pine_trees + num_acacia_trees

def prob_no_two_acacia_adj : ℚ :=
  (Nat.choose (num_non_acacia_trees + 1) num_acacia_trees * Nat.choose num_non_acacia_trees num_fir_trees : ℚ) /
  Nat.choose total_trees num_acacia_trees

theorem find_m_plus_n : (prob_no_two_acacia_adj = 84/159) -> (84 + 159 = 243) :=
by {
  admit
}

end find_m_plus_n_l198_198123


namespace exact_sunny_days_probability_l198_198484

noncomputable def choose (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

def rain_prob : ℚ := 3 / 4
def sun_prob : ℚ := 1 / 4
def days : ℕ := 5

theorem exact_sunny_days_probability : (choose days 2 * (sun_prob^2 * rain_prob^3) = 135 / 512) :=
by
  sorry

end exact_sunny_days_probability_l198_198484


namespace average_speed_ratio_l198_198349

def eddy_distance := 450 -- distance from A to B in km
def eddy_time := 3 -- time taken by Eddy in hours
def freddy_distance := 300 -- distance from A to C in km
def freddy_time := 4 -- time taken by Freddy in hours

def avg_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

def eddy_avg_speed := avg_speed eddy_distance eddy_time
def freddy_avg_speed := avg_speed freddy_distance freddy_time

def speed_ratio (speed1 : ℕ) (speed2 : ℕ) : ℕ × ℕ := (speed1 / (gcd speed1 speed2), speed2 / (gcd speed1 speed2))

theorem average_speed_ratio : speed_ratio eddy_avg_speed freddy_avg_speed = (2, 1) :=
by
  sorry

end average_speed_ratio_l198_198349


namespace base8_subtraction_l198_198457

theorem base8_subtraction : (53 - 26 : ℕ) = 25 :=
by sorry

end base8_subtraction_l198_198457


namespace suit_cost_l198_198312

theorem suit_cost :
  let shirt_cost := 15
  let pants_cost := 40
  let sweater_cost := 30
  let shirts := 4
  let pants := 2
  let sweaters := 2 
  let total_cost := shirts * shirt_cost + pants * pants_cost + sweaters * sweater_cost
  let discount_store := 0.80
  let discount_coupon := 0.90
  ∃ S, discount_coupon * discount_store * (total_cost + S) = 252 → S = 150 :=
by
  let shirt_cost := 15
  let pants_cost := 40
  let sweater_cost := 30
  let shirts := 4
  let pants := 2
  let sweaters := 2 
  let total_cost := shirts * shirt_cost + pants * pants_cost + sweaters * sweater_cost
  let discount_store := 0.80
  let discount_coupon := 0.90
  exists 150
  intro h
  sorry

end suit_cost_l198_198312


namespace first_day_revenue_l198_198927

theorem first_day_revenue :
  ∀ (S : ℕ), (12 * S + 90 = 246) → (4 * S + 3 * 9 = 79) :=
by
  intros S h1
  sorry

end first_day_revenue_l198_198927


namespace fg_difference_l198_198132

def f (x : ℝ) : ℝ := x^2 - 4 * x + 7
def g (x : ℝ) : ℝ := x + 4

theorem fg_difference : f (g 3) - g (f 3) = 20 :=
by
  sorry

end fg_difference_l198_198132


namespace product_of_five_consecutive_numbers_not_square_l198_198794

theorem product_of_five_consecutive_numbers_not_square (a b c d e : ℕ)
  (ha : a > 0) (hb : b = a + 1) (hc : c = b + 1) (hd : d = c + 1) (he : e = d + 1) :
  ¬ ∃ k : ℕ, a * b * c * d * e = k^2 := by
  sorry

end product_of_five_consecutive_numbers_not_square_l198_198794


namespace share_of_y_is_63_l198_198350

theorem share_of_y_is_63 (x y z : ℝ) (h1 : y = 0.45 * x) (h2 : z = 0.50 * x) (h3 : x + y + z = 273) : y = 63 :=
by
  -- The proof will go here
  sorry

end share_of_y_is_63_l198_198350


namespace find_ab_l198_198762

theorem find_ab (a b q r : ℕ) (h : a > 0) (h2 : b > 0) (h3 : (a^2 + b^2) / (a + b) = q) (h4 : (a^2 + b^2) % (a + b) = r) (h5 : q^2 + r = 2010) : a * b = 1643 :=
sorry

end find_ab_l198_198762


namespace solution_correctness_l198_198452

def is_prime (n : ℕ) : Prop := Nat.Prime n

def problem_statement (a b c : ℕ) : Prop :=
  (a * b * c = 56) ∧
  (a * b + b * c + a * c = 311) ∧
  is_prime a ∧ is_prime b ∧ is_prime c

theorem solution_correctness (a b c : ℕ) (h : problem_statement a b c) :
  a = 2 ∨ a = 13 ∨ a = 19 ∧
  b = 2 ∨ b = 13 ∨ b = 19 ∧
  c = 2 ∨ c = 13 ∨ c = 19 :=
by
  sorry

end solution_correctness_l198_198452


namespace original_cost_of_meal_l198_198432

-- Definitions for conditions
def meal_cost (initial_cost : ℝ) : ℝ :=
  initial_cost + 0.085 * initial_cost + 0.18 * initial_cost

-- The theorem we aim to prove
theorem original_cost_of_meal (total_cost : ℝ) (h : total_cost = 35.70) :
  ∃ initial_cost : ℝ, initial_cost = 28.23 ∧ meal_cost initial_cost = total_cost :=
by
  use 28.23
  rw [meal_cost, h]
  sorry

end original_cost_of_meal_l198_198432


namespace product_of_successive_numbers_l198_198969

-- Given conditions
def n : ℝ := 51.49757275833493

-- Proof statement
theorem product_of_successive_numbers : n * (n + 1) = 2703.0000000000005 :=
by
  -- Proof would be supplied here
  sorry

end product_of_successive_numbers_l198_198969


namespace cylinder_volume_rotation_l198_198087

theorem cylinder_volume_rotation (length width : ℝ) (π : ℝ) (h : length = 4) (w : width = 2) (V : ℝ) :
  (V = π * (4^2) * width ∨ V = π * (2^2) * length) :=
by
  sorry

end cylinder_volume_rotation_l198_198087


namespace min_ab_l198_198890

theorem min_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b - a * b + 3 = 0) : 
  9 ≤ a * b :=
sorry

end min_ab_l198_198890


namespace find_m_value_l198_198916

-- Definitions of the hyperbola and its focus condition
def hyperbola_eq (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / m) - (y^2 / (3 + m)) = 1

def focus_condition (m : ℝ) : Prop :=
  4 = (m) + (3 + m)

-- Theorem stating the value of m
theorem find_m_value (m : ℝ) : hyperbola_eq m → focus_condition m → m = 1 / 2 :=
by
  intros
  sorry

end find_m_value_l198_198916


namespace find_other_endpoint_l198_198334

theorem find_other_endpoint (mx my x₁ y₁ x₂ y₂ : ℤ) 
  (h1 : mx = (x₁ + x₂) / 2) 
  (h2 : my = (y₁ + y₂) / 2) 
  (h3 : mx = 3) 
  (h4 : my = 4) 
  (h5 : x₁ = -2) 
  (h6 : y₁ = -5) : 
  x₂ = 8 ∧ y₂ = 13 := 
by
  sorry

end find_other_endpoint_l198_198334


namespace multiplier_for_obsolete_books_l198_198774

theorem multiplier_for_obsolete_books 
  (x : ℕ) 
  (total_books_removed number_of_damaged_books : ℕ) 
  (h1 : total_books_removed = 69) 
  (h2 : number_of_damaged_books = 11) 
  (h3 : number_of_damaged_books + (x * number_of_damaged_books - 8) = total_books_removed) 
  : x = 6 := 
by 
  sorry

end multiplier_for_obsolete_books_l198_198774


namespace central_angle_of_sector_l198_198010

theorem central_angle_of_sector (P : ℝ) (x : ℝ) (h : P = 1 / 8) : x = 45 :=
by
  sorry

end central_angle_of_sector_l198_198010


namespace dolphins_to_be_trained_next_month_l198_198542

theorem dolphins_to_be_trained_next_month :
  ∀ (total_dolphins fully_trained remaining trained_next_month : ℕ),
    total_dolphins = 20 →
    fully_trained = (1 / 4 : ℚ) * total_dolphins →
    remaining = total_dolphins - fully_trained →
    (2 / 3 : ℚ) * remaining = 10 →
    trained_next_month = remaining - 10 →
    trained_next_month = 5 :=
by
  intros total_dolphins fully_trained remaining trained_next_month
  intro h1 h2 h3 h4 h5
  sorry

end dolphins_to_be_trained_next_month_l198_198542


namespace wheels_in_garage_l198_198867

theorem wheels_in_garage :
  let bicycles := 9
  let cars := 16
  let single_axle_trailers := 5
  let double_axle_trailers := 3
  let wheels_per_bicycle := 2
  let wheels_per_car := 4
  let wheels_per_single_axle_trailer := 2
  let wheels_per_double_axle_trailer := 4
  let total_wheels := bicycles * wheels_per_bicycle + cars * wheels_per_car + single_axle_trailers * wheels_per_single_axle_trailer + double_axle_trailers * wheels_per_double_axle_trailer
  total_wheels = 104 := by
  sorry

end wheels_in_garage_l198_198867


namespace max_value_of_s_l198_198382

theorem max_value_of_s (p q r s : ℝ) (h1 : p + q + r + s = 10)
  (h2 : p * q + p * r + p * s + q * r + q * s + r * s = 22) :
  s ≤ (5 + Real.sqrt 93) / 2 :=
sorry

end max_value_of_s_l198_198382


namespace fraction_of_students_with_partner_l198_198898

theorem fraction_of_students_with_partner (s t : ℕ) 
  (h : t = (4 * s) / 3) :
  (t / 4 + s / 3) / (t + s) = 2 / 7 :=
by
  -- Proof omitted
  sorry

end fraction_of_students_with_partner_l198_198898


namespace correct_solutions_l198_198858

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ (x y : ℝ), f (x * y) = f x * f y - 2 * x * y

theorem correct_solutions :
  (∀ x : ℝ, f x = 2 * x) ∨ (∀ x : ℝ, f x = -x) := sorry

end correct_solutions_l198_198858


namespace sum_of_possible_values_of_N_l198_198620

theorem sum_of_possible_values_of_N :
  (∃ N : ℝ, N * (N - 7) = 12) → (∃ N₁ N₂ : ℝ, (N₁ * (N₁ - 7) = 12 ∧ N₂ * (N₂ - 7) = 12) ∧ N₁ + N₂ = 7) :=
by
  sorry

end sum_of_possible_values_of_N_l198_198620


namespace convert_kmph_to_mps_l198_198280

theorem convert_kmph_to_mps (speed_kmph : ℝ) (km_to_m : ℝ) (hr_to_s : ℝ) : 
  speed_kmph = 56 → km_to_m = 1000 → hr_to_s = 3600 → 
  (speed_kmph * (km_to_m / hr_to_s) : ℝ) = 15.56 :=
by
  intros
  sorry

end convert_kmph_to_mps_l198_198280


namespace abs_nonneg_position_l198_198934

theorem abs_nonneg_position (a : ℝ) : 0 ≤ |a| ∧ |a| ≥ 0 → (exists x : ℝ, x = |a| ∧ x ≥ 0) :=
by 
  sorry

end abs_nonneg_position_l198_198934


namespace evaluate_expression_l198_198693

theorem evaluate_expression :
  1002^3 - 1001 * 1002^2 - 1001^2 * 1002 + 1001^3 - 1000^3 = 2009007 :=
by
  sorry

end evaluate_expression_l198_198693


namespace cube_volume_l198_198815

variables (x s : ℝ)
theorem cube_volume (h : 6 * s^2 = 6 * x^2) : s^3 = x^3 :=
by sorry

end cube_volume_l198_198815


namespace time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l198_198444

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l198_198444


namespace xy_zero_l198_198619

theorem xy_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 0 :=
by
  sorry

end xy_zero_l198_198619


namespace correct_option_C_l198_198558

variable {a : ℝ} (x : ℝ) (b : ℝ)

theorem correct_option_C : 
  (a^8 / a^2 = a^6) :=
by {
  sorry
}

end correct_option_C_l198_198558


namespace bob_total_candies_l198_198331

noncomputable def total_chewing_gums : ℕ := 45
noncomputable def total_chocolate_bars : ℕ := 60
noncomputable def total_assorted_candies : ℕ := 45

def chewing_gum_ratio_sam_bob : ℕ × ℕ := (2, 3)
def chocolate_bar_ratio_sam_bob : ℕ × ℕ := (3, 1)
def assorted_candy_ratio_sam_bob : ℕ × ℕ := (1, 1)

theorem bob_total_candies :
  let bob_chewing_gums := (total_chewing_gums * chewing_gum_ratio_sam_bob.snd) / (chewing_gum_ratio_sam_bob.fst + chewing_gum_ratio_sam_bob.snd)
  let bob_chocolate_bars := (total_chocolate_bars * chocolate_bar_ratio_sam_bob.snd) / (chocolate_bar_ratio_sam_bob.fst + chocolate_bar_ratio_sam_bob.snd)
  let bob_assorted_candies := (total_assorted_candies * assorted_candy_ratio_sam_bob.snd) / (assorted_candy_ratio_sam_bob.fst + assorted_candy_ratio_sam_bob.snd)
  bob_chewing_gums + bob_chocolate_bars + bob_assorted_candies = 64 := by
  sorry

end bob_total_candies_l198_198331


namespace recorded_instances_l198_198924

-- Define the conditions
def interval := 5
def total_time := 60 * 60  -- one hour in seconds

-- Define the theorem to prove the expected number of instances recorded
theorem recorded_instances : total_time / interval = 720 := by
  sorry

end recorded_instances_l198_198924


namespace wolf_does_not_catch_hare_l198_198085

-- Define the distance the hare needs to cover
def distanceHare := 250 -- meters

-- Define the initial separation between the wolf and the hare
def separation := 30 -- meters

-- Define the speed of the hare
def speedHare := 550 -- meters per minute

-- Define the speed of the wolf
def speedWolf := 600 -- meters per minute

-- Define the time it takes for the hare to reach the refuge
def tHare := (distanceHare : ℚ) / speedHare

-- Define the total distance the wolf needs to cover
def totalDistanceWolf := distanceHare + separation

-- Define the time it takes for the wolf to cover the total distance
def tWolf := (totalDistanceWolf : ℚ) / speedWolf

-- Final proposition to be proven
theorem wolf_does_not_catch_hare : tHare < tWolf :=
by
  sorry

end wolf_does_not_catch_hare_l198_198085


namespace fg_of_3_l198_198931

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := x^2 - 3 * x

theorem fg_of_3 : f (g 3) = -2 := by
  sorry

end fg_of_3_l198_198931


namespace range_of_a_l198_198000

theorem range_of_a {a : ℝ} :
  (∃ (x y : ℝ), (x - a)^2 + (y - a)^2 = 4 ∧ x^2 + y^2 = 4) ↔ (-2*Real.sqrt 2 < a ∧ a < 2*Real.sqrt 2 ∧ a ≠ 0) :=
sorry

end range_of_a_l198_198000


namespace initial_breads_count_l198_198194

theorem initial_breads_count :
  ∃ (X : ℕ), ((((X / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2 = 3 ∧ X = 127 :=
by sorry

end initial_breads_count_l198_198194


namespace exists_six_numbers_multiple_2002_l198_198335

theorem exists_six_numbers_multiple_2002 (a : Fin 41 → ℕ) (h : Function.Injective a) :
  ∃ (i j k l m n : Fin 41),
    i ≠ j ∧ k ≠ l ∧ m ≠ n ∧
    (a i - a j) * (a k - a l) * (a m - a n) % 2002 = 0 := sorry

end exists_six_numbers_multiple_2002_l198_198335


namespace percentage_of_males_l198_198731

theorem percentage_of_males (P : ℝ) (total_employees : ℝ) (below_50_male_count : ℝ) :
  total_employees = 2800 →
  0.70 * (P / 100 * total_employees) = below_50_male_count →
  below_50_male_count = 490 →
  P = 25 :=
by
  intros h_total h_eq h_below_50
  sorry

end percentage_of_males_l198_198731


namespace triangle_inequality_l198_198454

theorem triangle_inequality (a b c : ℝ) (S : ℝ) (hS : S = (1/4) * Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

end triangle_inequality_l198_198454


namespace sum_of_digits_in_product_is_fourteen_l198_198977

def first_number : ℕ := -- Define the 101-digit number 141,414,141,...,414,141
  141 * 10^98 + 141 * 10^95 + 141 * 10^92 -- continue this pattern...

def second_number : ℕ := -- Define the 101-digit number 707,070,707,...,070,707
  707 * 10^98 + 707 * 10^95 + 707 * 10^92 -- continue this pattern...

def units_digit (n : ℕ) : ℕ := n % 10
def ten_thousands_digit (n : ℕ) : ℕ := (n / 10000) % 10

theorem sum_of_digits_in_product_is_fourteen :
  units_digit (first_number * second_number) + ten_thousands_digit (first_number * second_number) = 14 :=
sorry

end sum_of_digits_in_product_is_fourteen_l198_198977


namespace uncle_gave_13_l198_198986

-- Define all the given constants based on the conditions.
def J := 7    -- cost of the jump rope
def B := 12   -- cost of the board game
def P := 4    -- cost of the playground ball
def S := 6    -- savings from Dalton's allowance
def N := 4    -- additional amount needed

-- Derived quantities
def total_cost := J + B + P

-- Statement: to prove Dalton's uncle gave him $13.
theorem uncle_gave_13 : (total_cost - N) - S = 13 := by
  sorry

end uncle_gave_13_l198_198986


namespace total_watermelons_l198_198745

theorem total_watermelons 
  (A B C : ℕ) 
  (h1 : A + B = C - 6) 
  (h2 : B + C = A + 16) 
  (h3 : C + A = B + 8) :
  A + B + C = 18 :=
by
  sorry

end total_watermelons_l198_198745


namespace number_of_children_l198_198360

theorem number_of_children (x : ℕ) : 3 * x + 12 = 5 * x - 10 → x = 11 :=
by
  intros h
  have : 3 * x + 12 = 5 * x - 10 := h
  sorry

end number_of_children_l198_198360


namespace range_of_p_add_q_l198_198098

theorem range_of_p_add_q (p q : ℝ) :
  (∀ x : ℝ, ¬(x^2 + 2 * p * x - (q^2 - 2) = 0)) → 
  (p + q) ∈ Set.Ioo (-2 : ℝ) (2 : ℝ) :=
by
  intro h
  sorry

end range_of_p_add_q_l198_198098


namespace andrea_needs_to_buy_sod_squares_l198_198611

theorem andrea_needs_to_buy_sod_squares :
  let area_section1 := 30 * 40
  let area_section2 := 60 * 80
  let total_area := area_section1 + area_section2
  let area_of_sod_square := 2 * 2
  1500 = total_area / area_of_sod_square :=
by
  let area_section1 := 30 * 40
  let area_section2 := 60 * 80
  let total_area := area_section1 + area_section2
  let area_of_sod_square := 2 * 2
  sorry

end andrea_needs_to_buy_sod_squares_l198_198611


namespace precision_mult_10_decreases_precision_mult_35_decreases_precision_div_10_increases_precision_div_35_increases_l198_198711

-- Given definitions for precision adjustment
def initial_precision := 3

def new_precision_mult (x : ℕ): ℕ :=
  initial_precision - 1   -- Example: Multiplying by 10 moves decimal point right decreasing precision by 1

def new_precision_mult_large (x : ℕ): ℕ := 
  initial_precision - 2   -- Example: Multiplying by 35 generally decreases precision by 2

def new_precision_div (x : ℕ): ℕ := 
  initial_precision + 1   -- Example: Dividing by 10 moves decimal point left increasing precision by 1

def new_precision_div_large (x : ℕ): ℕ := 
  initial_precision + 1   -- Example: Dividing by 35 generally increases precision by 1

-- Statements to prove
theorem precision_mult_10_decreases: 
  new_precision_mult 10 = 2 := 
by 
  sorry

theorem precision_mult_35_decreases: 
  new_precision_mult_large 35 = 1 := 
by 
  sorry

theorem precision_div_10_increases: 
  new_precision_div 10 = 4 := 
by 
  sorry

theorem precision_div_35_increases: 
  new_precision_div_large 35 = 4 := 
by 
  sorry

end precision_mult_10_decreases_precision_mult_35_decreases_precision_div_10_increases_precision_div_35_increases_l198_198711


namespace quadratic_bounds_l198_198235

variable (a b c: ℝ)

-- Conditions
def quadratic_function (x: ℝ) : ℝ := a * x^2 + b * x + c

def within_range_neg_1_to_1 (h : ∀ x: ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 1) : Prop :=
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -7 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 7

-- Main statement
theorem quadratic_bounds
  (h : ∀ x: ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 1) :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -7 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 7 := sorry

end quadratic_bounds_l198_198235


namespace abcd_inequality_l198_198501

theorem abcd_inequality (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
(h_eq : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) + (d^2 / (1 + d^2)) = 1) :
  a * b * c * d ≤ 1 / 9 :=
sorry

end abcd_inequality_l198_198501


namespace sin2x_value_l198_198747

theorem sin2x_value (x : ℝ) (h : Real.sin (x + π / 4) = 3 / 5) : 
  Real.sin (2 * x) = 8 * Real.sqrt 2 / 25 := 
by sorry

end sin2x_value_l198_198747


namespace vector_sum_eq_l198_198698

variables (x y : ℝ)
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (3, 3)
def c : ℝ × ℝ := (7, 8)

theorem vector_sum_eq :
  ∃ (x y : ℝ), c = (x • a.1 + y • b.1, x • a.2 + y • b.2) ∧ x + y = 8 / 3 :=
by
  have h1 : 7 = 2 * x + 3 * y := sorry
  have h2 : 8 = 3 * x + 3 * y := sorry
  sorry

end vector_sum_eq_l198_198698


namespace words_per_page_is_106_l198_198848

noncomputable def book_pages := 154
noncomputable def max_words_per_page := 120
noncomputable def total_words_mod := 221
noncomputable def mod_val := 217

def number_of_words_per_page (p : ℕ) : Prop :=
  (book_pages * p ≡ total_words_mod [MOD mod_val]) ∧ (p ≤ max_words_per_page)

theorem words_per_page_is_106 : number_of_words_per_page 106 :=
by
  sorry

end words_per_page_is_106_l198_198848


namespace problem1_problem2_l198_198909

-- Statement for Problem 1
theorem problem1 (x y : ℝ) : (x - y) ^ 2 + x * (x + 2 * y) = 2 * x ^ 2 + y ^ 2 :=
by sorry

-- Statement for Problem 2
theorem problem2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  ((-3 * x + 4) / (x - 1) + x) / ((x - 2) / (x ^ 2 - x)) = x ^ 2 - 2 * x :=
by sorry

end problem1_problem2_l198_198909


namespace integer_solutions_l198_198829

theorem integer_solutions (n : ℤ) : ∃ m : ℤ, n^2 + 15 = m^2 ↔ n = -7 ∨ n = -1 ∨ n = 1 ∨ n = 7 :=
by
  sorry

end integer_solutions_l198_198829


namespace composite_divides_factorial_l198_198976

-- Define the factorial of a number
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Statement of the problem
theorem composite_divides_factorial (m : ℕ) (hm : m ≠ 4) (hcomposite : ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = m) :
  m ∣ factorial (m - 1) :=
by
  sorry

end composite_divides_factorial_l198_198976


namespace min_value_1abc_l198_198881

theorem min_value_1abc (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9) (h₂ : 0 ≤ b ∧ b ≤ 9) (h₃ : c = 0) 
    (h₄ : (1000 + 100 * a + 10 * b + c) % 2 = 0) 
    (h₅ : (1000 + 100 * a + 10 * b + c) % 3 = 0) 
    (h₆ : (1000 + 100 * a + 10 * b + c) % 5 = 0)
  : 1000 + 100 * a + 10 * b + c = 1020 :=
by
  sorry

end min_value_1abc_l198_198881


namespace find_complex_number_z_l198_198733

-- Given the complex number z and the equation \(\frac{z}{1+i} = i^{2015} + i^{2016}\)
-- prove that z = -2i
theorem find_complex_number_z (z : ℂ) (h : z / (1 + (1 : ℂ) * I) = I ^ 2015 + I ^ 2016) : z = -2 * I := 
by
  sorry

end find_complex_number_z_l198_198733


namespace total_distance_traveled_by_children_l198_198842

theorem total_distance_traveled_by_children :
  let ap := 50
  let dist_1_vertex_skip := (50 : ℝ) * Real.sqrt 2
  let dist_2_vertices_skip := (50 : ℝ) * Real.sqrt (2 + 2 * Real.sqrt 2)
  let dist_diameter := (2 : ℝ) * 50
  let single_child_distance := 2 * dist_1_vertex_skip + 2 * dist_2_vertices_skip + dist_diameter
  8 * single_child_distance = 800 * Real.sqrt 2 + 800 * Real.sqrt (2 + 2 * Real.sqrt 2) + 800 :=
sorry

end total_distance_traveled_by_children_l198_198842


namespace inequality_xyz_geq_3_l198_198253

theorem inequality_xyz_geq_3
  (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_not_all_zero : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :
  (2 * x^2 - x + y + z) / (x + y^2 + z^2) +
  (2 * y^2 + x - y + z) / (x^2 + y + z^2) +
  (2 * z^2 + x + y - z) / (x^2 + y^2 + z) ≥ 3 := 
sorry

end inequality_xyz_geq_3_l198_198253


namespace p_is_sufficient_not_necessary_for_q_l198_198039

-- Definitions for conditions p and q
def p (x : ℝ) := x^2 - x - 20 > 0
def q (x : ℝ) := 1 - x^2 < 0

-- The main statement
theorem p_is_sufficient_not_necessary_for_q:
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end p_is_sufficient_not_necessary_for_q_l198_198039


namespace loggers_count_l198_198404

theorem loggers_count 
  (cut_rate : ℕ) 
  (forest_width : ℕ) 
  (forest_height : ℕ) 
  (tree_density : ℕ) 
  (days_per_month : ℕ) 
  (months : ℕ) 
  (total_loggers : ℕ)
  (total_trees : ℕ := forest_width * forest_height * tree_density) 
  (total_days : ℕ := days_per_month * months)
  (trees_cut_down_per_logger : ℕ := cut_rate * total_days) 
  (expected_loggers : ℕ := total_trees / trees_cut_down_per_logger) 
  (h1: cut_rate = 6)
  (h2: forest_width = 4)
  (h3: forest_height = 6)
  (h4: tree_density = 600)
  (h5: days_per_month = 30)
  (h6: months = 10)
  (h7: total_loggers = expected_loggers)
: total_loggers = 8 := 
by {
    sorry
}

end loggers_count_l198_198404


namespace at_least_one_not_less_than_two_l198_198259

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) :=
sorry

end at_least_one_not_less_than_two_l198_198259


namespace line_intersects_parabola_at_9_units_apart_l198_198565

theorem line_intersects_parabola_at_9_units_apart :
  ∃ m b, (∃ (k1 k2 : ℝ), 
              (y1 = k1^2 + 6*k1 - 4) ∧ 
              (y2 = k2^2 + 6*k2 - 4) ∧ 
              (y1 = m*k1 + b) ∧ 
              (y2 = m*k2 + b) ∧ 
              |y1 - y2| = 9) ∧ 
          (0 ≠ b) ∧ 
          ((1 : ℝ) = 2*m + b) ∧ 
          (m = 4 ∧ b = -7)
:= sorry

end line_intersects_parabola_at_9_units_apart_l198_198565


namespace exist_interval_l198_198633

noncomputable def f (x : ℝ) := Real.log x + x - 4

theorem exist_interval (x₀ : ℝ) (h₀ : f x₀ = 0) : 2 < x₀ ∧ x₀ < 3 :=
by
  sorry

end exist_interval_l198_198633


namespace distance_CD_l198_198738

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  16 * (x + 2)^2 + 4 * y^2 = 64

def major_axis_distance : ℝ := 4
def minor_axis_distance : ℝ := 2

theorem distance_CD : ∃ (d : ℝ), 16 * (x + 2)^2 + 4 * y^2 = 64 → d = 2 * Real.sqrt 5 :=
by
  sorry

end distance_CD_l198_198738


namespace square_side_length_tangent_circle_l198_198155

theorem square_side_length_tangent_circle (r s : ℝ) :
  (∃ (O : ℝ × ℝ) (A : ℝ × ℝ) (AB : ℝ) (AD : ℝ),
    AB = AD ∧
    O = (r, r) ∧
    A = (0, 0) ∧
    dist O A = r * Real.sqrt 2 ∧
    s = dist (O.fst, 0) A ∧
    s = dist (0, O.snd) A ∧
    ∀ x y, (O = (x, y) → x = r ∧ y = r)) → s = 2 * r :=
by
  sorry

end square_side_length_tangent_circle_l198_198155


namespace gcd_24_36_54_l198_198996

-- Define the numbers and the gcd function
def num1 : ℕ := 24
def num2 : ℕ := 36
def num3 : ℕ := 54

-- The Lean statement to prove that the gcd of num1, num2, and num3 is 6
theorem gcd_24_36_54 : Nat.gcd (Nat.gcd num1 num2) num3 = 6 := by
  sorry

end gcd_24_36_54_l198_198996


namespace john_total_distance_l198_198673

-- Define the parameters according to the conditions
def daily_distance : ℕ := 1700
def number_of_days : ℕ := 6
def total_distance : ℕ := daily_distance * number_of_days

-- Lean theorem statement to prove the total distance run by John
theorem john_total_distance : total_distance = 10200 := by
  -- Here, the proof would go, but it is omitted as per instructions
  sorry

end john_total_distance_l198_198673


namespace no_real_solutions_l198_198029

theorem no_real_solutions :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 4 → ¬(3 * x^2 - 15 * x) / (x^2 - 4 * x) = x - 2) :=
by
  sorry

end no_real_solutions_l198_198029


namespace win_sector_area_l198_198513

theorem win_sector_area (r : ℝ) (h1 : r = 8) (h2 : (1 / 4) = 1 / 4) : 
  ∃ (area : ℝ), area = 16 * Real.pi := 
by
  existsi (16 * Real.pi); exact sorry

end win_sector_area_l198_198513


namespace sequence_term_number_l198_198539

theorem sequence_term_number (n : ℕ) : (n ≥ 1) → (n + 3 = 17 ∧ n + 1 = 15) → n = 14 := 
by
  intro h1 h2
  sorry

end sequence_term_number_l198_198539


namespace find_a1_l198_198303

-- Definitions of the conditions
def Sn (n : ℕ) : ℕ := sorry  -- Sum of the first n terms of the sequence
def a₁ : ℤ := sorry          -- First term of the sequence

axiom S_2016_eq_2016 : Sn 2016 = 2016
axiom diff_seq_eq_2000 : (Sn 2016 / 2016) - (Sn 16 / 16) = 2000

-- Proof statement
theorem find_a1 : a₁ = -2014 :=
by
  -- The proof would go here
  sorry

end find_a1_l198_198303


namespace total_marbles_l198_198278

theorem total_marbles (r b g : ℕ) (h_ratio : r = 1 ∧ b = 5 ∧ g = 3) (h_green : g = 27) :
  (r + b + g) * 3 = 81 :=
  sorry

end total_marbles_l198_198278


namespace normal_line_at_x0_is_correct_l198_198689

noncomputable def curve (x : ℝ) : ℝ := x^(2/3) - 20

def x0 : ℝ := -8

def normal_line_equation (x : ℝ) : ℝ := 3 * x + 8

theorem normal_line_at_x0_is_correct : 
  ∃ y0 : ℝ, curve x0 = y0 ∧ y0 = curve x0 ∧ normal_line_equation x0 = y0 :=
sorry

end normal_line_at_x0_is_correct_l198_198689


namespace students_taking_neither_l198_198629

theorem students_taking_neither (total_students music_students art_students dance_students music_art music_dance art_dance music_art_dance : ℕ) :
  total_students = 2500 →
  music_students = 200 →
  art_students = 150 →
  dance_students = 100 →
  music_art = 75 →
  art_dance = 50 →
  music_dance = 40 →
  music_art_dance = 25 →
  total_students - ((music_students + art_students + dance_students) - (music_art + art_dance + music_dance) + music_art_dance) = 2190 :=
by
  intros
  sorry

end students_taking_neither_l198_198629


namespace radius_scientific_notation_l198_198651

theorem radius_scientific_notation :
  696000 = 6.96 * 10^5 :=
sorry

end radius_scientific_notation_l198_198651


namespace deductible_amount_l198_198750

-- This definition represents the conditions of the problem.
def current_annual_deductible_is_increased (D : ℝ) : Prop :=
  (2 / 3) * D = 2000

-- This is the Lean statement, expressing the problem that needs to be proven.
theorem deductible_amount (D : ℝ) (h : current_annual_deductible_is_increased D) : D = 3000 :=
by
  sorry

end deductible_amount_l198_198750


namespace gain_percent_is_80_l198_198825

noncomputable def cost_price : ℝ := 600
noncomputable def selling_price : ℝ := 1080
noncomputable def gain : ℝ := selling_price - cost_price
noncomputable def gain_percent : ℝ := (gain / cost_price) * 100

theorem gain_percent_is_80 :
  gain_percent = 80 := by
  sorry

end gain_percent_is_80_l198_198825


namespace nigel_gave_away_l198_198576

theorem nigel_gave_away :
  ∀ (original : ℕ) (gift_from_mother : ℕ) (final : ℕ) (money_given_away : ℕ),
    original = 45 →
    gift_from_mother = 80 →
    final = 2 * original + 10 →
    final = original - money_given_away + gift_from_mother →
    money_given_away = 25 :=
by
  intros original gift_from_mother final money_given_away
  sorry

end nigel_gave_away_l198_198576


namespace second_derivative_at_x₀_l198_198195

noncomputable def f (x : ℝ) : ℝ := sorry
variables (x₀ a b : ℝ)

-- Condition: f(x₀ + Δx) - f(x₀) = a * Δx + b * (Δx)^2
axiom condition : ∀ Δx, f (x₀ + Δx) - f x₀ = a * Δx + b * (Δx)^2

theorem second_derivative_at_x₀ : deriv (deriv f) x₀ = 2 * b :=
sorry

end second_derivative_at_x₀_l198_198195


namespace smallest_square_area_l198_198328

variable (M N : ℝ)

/-- Given that the largest square has an area of 1 cm^2, the middle square has an area M cm^2, and the smallest square has a vertex on the side of the middle square, prove that the area of the smallest square N is equal to ((1 - M) / 2)^2. -/
theorem smallest_square_area (h1 : 1 ≥ 0)
  (h2 : 0 ≤ M ∧ M ≤ 1)
  (h3 : 0 ≤ N) :
  N = (1 - M) ^ 2 / 4 := sorry

end smallest_square_area_l198_198328


namespace function_machine_output_is_17_l198_198653

def functionMachineOutput (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 <= 22 then step1 + 10 else step1 - 7

theorem function_machine_output_is_17 : functionMachineOutput 8 = 17 := by
  sorry

end function_machine_output_is_17_l198_198653


namespace sum_of_two_consecutive_squares_l198_198241

variable {k m A : ℕ}

theorem sum_of_two_consecutive_squares :
  (∃ k : ℕ, A^2 = (k+1)^3 - k^3) → (∃ m : ℕ, A = m^2 + (m+1)^2) :=
by sorry

end sum_of_two_consecutive_squares_l198_198241


namespace measure_of_angle_B_l198_198379

theorem measure_of_angle_B 
  (A B C: ℝ)
  (a b c: ℝ)
  (h1: A + B + C = π)
  (h2: B / A = C / B)
  (h3: b^2 - a^2 = a * c) : B = 2 * π / 7 :=
  sorry

end measure_of_angle_B_l198_198379


namespace number_of_houses_built_l198_198642

def original_houses : ℕ := 20817
def current_houses : ℕ := 118558
def houses_built : ℕ := current_houses - original_houses

theorem number_of_houses_built :
  houses_built = 97741 := by
  sorry

end number_of_houses_built_l198_198642


namespace logarithm_identity_l198_198997

theorem logarithm_identity :
  1 / (Real.log 3 / Real.log 8 + 1) + 
  1 / (Real.log 2 / Real.log 12 + 1) + 
  1 / (Real.log 4 / Real.log 9 + 1) = 3 := 
by
  sorry

end logarithm_identity_l198_198997


namespace output_in_scientific_notation_l198_198720

def output_kilowatt_hours : ℝ := 448000
def scientific_notation (n : ℝ) : Prop := n = 4.48 * 10^5

theorem output_in_scientific_notation : scientific_notation output_kilowatt_hours :=
by
  -- Proof steps are not required
  sorry

end output_in_scientific_notation_l198_198720


namespace smaller_circle_radius_l198_198019

-- Given conditions
def larger_circle_radius : ℝ := 10
def number_of_smaller_circles : ℕ := 7

-- The goal
theorem smaller_circle_radius :
  ∃ r : ℝ, (∃ D : ℝ, D = 2 * larger_circle_radius ∧ D = 4 * r) ∧ r = 2.5 :=
by
  sorry

end smaller_circle_radius_l198_198019


namespace lcm_36_105_l198_198929

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l198_198929


namespace frames_sharing_point_with_line_e_l198_198896

def frame_shares_common_point_with_line (n : ℕ) : Prop := 
  n = 0 ∨ n = 1 ∨ n = 9 ∨ n = 17 ∨ n = 25 ∨ n = 33 ∨ n = 41 ∨ n = 49 ∨
  n = 6 ∨ n = 14 ∨ n = 22 ∨ n = 30 ∨ n = 38 ∨ n = 46

theorem frames_sharing_point_with_line_e :
  ∀ (i : ℕ), i < 50 → frame_shares_common_point_with_line i = 
  (i = 0 ∨ i = 1 ∨ i = 9 ∨ i = 17 ∨ i = 25 ∨ i = 33 ∨ i = 41 ∨ i = 49 ∨
   i = 6 ∨ i = 14 ∨ i = 22 ∨ i = 30 ∨ i = 38 ∨ i = 46) := 
by 
  sorry

end frames_sharing_point_with_line_e_l198_198896


namespace min_distinct_sums_max_distinct_sums_l198_198209

theorem min_distinct_sums (n : ℕ) (h : 0 < n) : ∃ a b, (a + (n - 1) * b) = (n * (n + 1)) / 2 := sorry

theorem max_distinct_sums (n : ℕ) (h : 0 < n) : 
  ∃ m, m = 2^n - 1 := sorry

end min_distinct_sums_max_distinct_sums_l198_198209


namespace train_length_l198_198420

theorem train_length (V L : ℝ) (h1 : L = V * 18) (h2 : L + 550 = V * 51) : L = 300 := sorry

end train_length_l198_198420
