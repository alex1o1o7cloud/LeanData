import Mathlib

namespace NUMINAMATH_GPT_quadruplets_sets_l126_12625

theorem quadruplets_sets (a b c babies: ℕ) (h1: 2 * a + 3 * b + 4 * c = 1200) (h2: b = 5 * c) (h3: a = 2 * b) :
  4 * c = 123 :=
by
  sorry

end NUMINAMATH_GPT_quadruplets_sets_l126_12625


namespace NUMINAMATH_GPT_certain_number_l126_12643

theorem certain_number (x y a : ℤ) (h1 : 4 * x + y = a) (h2 : 2 * x - y = 20) 
  (h3 : y ^ 2 = 4) : a = 46 :=
sorry

end NUMINAMATH_GPT_certain_number_l126_12643


namespace NUMINAMATH_GPT_trapezoid_EC_length_l126_12644

-- Define a trapezoid and its properties.
structure Trapezoid (A B C D : Type) :=
(base1 : ℝ) -- AB
(base2 : ℝ) -- CD
(diagonal_AC : ℝ) -- AC
(AB_eq_3CD : base1 = 3 * base2)
(AC_length : diagonal_AC = 15)
(E : Type) -- point of intersection of diagonals

-- Proof statement that length of EC is 15/4
theorem trapezoid_EC_length
  {A B C D E : Type}
  (t : Trapezoid A B C D)
  (E : Type)
  (intersection_E : E) :
  ∃ (EC : ℝ), EC = 15 / 4 :=
by
  have h1 : t.base1 = 3 * t.base2 := t.AB_eq_3CD
  have h2 : t.diagonal_AC = 15 := t.AC_length
  -- Use the given conditions to derive the length of EC
  sorry

end NUMINAMATH_GPT_trapezoid_EC_length_l126_12644


namespace NUMINAMATH_GPT_initial_goats_l126_12637

theorem initial_goats (G : ℕ) (h1 : 2 + 3 + G + 3 + 5 + 2 = 21) : G = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_goats_l126_12637


namespace NUMINAMATH_GPT_keychain_arrangement_l126_12638

open Function

theorem keychain_arrangement (keys : Finset ℕ) (h : keys.card = 7)
  (house_key car_key office_key : ℕ) (hmem : house_key ∈ keys)
  (cmem : car_key ∈ keys) (omem : office_key ∈ keys) : 
  ∃ n : ℕ, n = 72 :=
by
  sorry

end NUMINAMATH_GPT_keychain_arrangement_l126_12638


namespace NUMINAMATH_GPT_age_difference_l126_12622

theorem age_difference (john_age father_age mother_age : ℕ) 
    (h1 : john_age * 2 = father_age) 
    (h2 : father_age = mother_age + 4) 
    (h3 : father_age = 40) :
    mother_age - john_age = 16 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l126_12622


namespace NUMINAMATH_GPT_relationship_between_m_and_n_l126_12635

theorem relationship_between_m_and_n
  (m n : ℝ)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 4 * x + 2 * y - 4 = 0)
  (line_eq : ∀ (x y : ℝ), m * x + 2 * n * y - 4 = 0) :
  m - n - 2 = 0 := 
  sorry

end NUMINAMATH_GPT_relationship_between_m_and_n_l126_12635


namespace NUMINAMATH_GPT_spadesuit_calculation_l126_12634

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_calculation : spadesuit 2 (spadesuit 6 1) = -1221 := by
  sorry

end NUMINAMATH_GPT_spadesuit_calculation_l126_12634


namespace NUMINAMATH_GPT_average_speed_last_segment_l126_12662

theorem average_speed_last_segment
  (total_distance : ℕ)
  (total_time : ℕ)
  (speed1 speed2 speed3 : ℕ)
  (last_segment_time : ℕ)
  (average_speed_total : ℕ) :
  total_distance = 180 →
  total_time = 180 →
  speed1 = 40 →
  speed2 = 50 →
  speed3 = 60 →
  average_speed_total = 60 →
  last_segment_time = 45 →
  ∃ (speed4 : ℕ), speed4 = 90 :=
by sorry

end NUMINAMATH_GPT_average_speed_last_segment_l126_12662


namespace NUMINAMATH_GPT_fractions_of_group_money_l126_12618

def moneyDistribution (m l n o : ℕ) (moeGave : ℕ) (lokiGave : ℕ) (nickGave : ℕ) : Prop :=
  moeGave = 1 / 5 * m ∧
  lokiGave = 1 / 4 * l ∧
  nickGave = 1 / 3 * n ∧
  moeGave = lokiGave ∧
  lokiGave = nickGave ∧
  o = moeGave + lokiGave + nickGave

theorem fractions_of_group_money (m l n o total : ℕ) :
  moneyDistribution m l n o 1 1 1 →
  total = m + l + n →
  (o : ℚ) / total = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_fractions_of_group_money_l126_12618


namespace NUMINAMATH_GPT_solve_equation_simplify_expression_l126_12623

-- Problem (1)
theorem solve_equation : ∀ x : ℝ, x * (x + 6) = 8 * (x + 3) ↔ x = 6 ∨ x = -4 := by
  sorry

-- Problem (2)
theorem simplify_expression : ∀ a b : ℝ, a ≠ b → (a ≠ 0 ∧ b ≠ 0) →
  (3 * a ^ 2 - 3 * b ^ 2) / (a ^ 2 * b + a * b ^ 2) /
  (1 - (a ^ 2 + b ^ 2) / (2 * a * b)) = -6 / (a - b) := by
  sorry

end NUMINAMATH_GPT_solve_equation_simplify_expression_l126_12623


namespace NUMINAMATH_GPT_pow_mod_eq_l126_12603

theorem pow_mod_eq (h : 101 % 100 = 1) : (101 ^ 50) % 100 = 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_pow_mod_eq_l126_12603


namespace NUMINAMATH_GPT_abs_less_than_2_sufficient_but_not_necessary_l126_12666

theorem abs_less_than_2_sufficient_but_not_necessary (x : ℝ) : 
  (|x| < 2 → (x^2 - x - 6 < 0)) ∧ ¬(x^2 - x - 6 < 0 → |x| < 2) :=
by
  sorry

end NUMINAMATH_GPT_abs_less_than_2_sufficient_but_not_necessary_l126_12666


namespace NUMINAMATH_GPT_inequality_sqrt_a_b_c_l126_12664

noncomputable def sqrt (x : ℝ) := x ^ (1 / 2)

theorem inequality_sqrt_a_b_c (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  sqrt (a ^ (1 - a) * b ^ (1 - b) * c ^ (1 - c)) ≤ 1 / 3 := 
sorry

end NUMINAMATH_GPT_inequality_sqrt_a_b_c_l126_12664


namespace NUMINAMATH_GPT_probability_below_8_l126_12629

def prob_hit_10 := 0.20
def prob_hit_9 := 0.30
def prob_hit_8 := 0.10

theorem probability_below_8 : (1 - (prob_hit_10 + prob_hit_9 + prob_hit_8) = 0.40) :=
by
  sorry

end NUMINAMATH_GPT_probability_below_8_l126_12629


namespace NUMINAMATH_GPT_total_books_in_bookcase_l126_12694

def num_bookshelves := 8
def num_layers_per_bookshelf := 5
def books_per_layer := 85

theorem total_books_in_bookcase : 
  (num_bookshelves * num_layers_per_bookshelf * books_per_layer) = 3400 := by
  sorry

end NUMINAMATH_GPT_total_books_in_bookcase_l126_12694


namespace NUMINAMATH_GPT_sin_135_degree_l126_12626

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_135_degree_l126_12626


namespace NUMINAMATH_GPT_original_price_l126_12613

-- Definitions of conditions
def SalePrice : Float := 70
def DecreasePercentage : Float := 30

-- Statement to prove
theorem original_price (P : Float) (h : 0.70 * P = SalePrice) : P = 100 := by
  sorry

end NUMINAMATH_GPT_original_price_l126_12613


namespace NUMINAMATH_GPT_maximize_daily_profit_l126_12681

noncomputable def daily_profit : ℝ → ℝ → ℝ
| x, c => if h : 0 < x ∧ x ≤ c then (3 * (9 * x - 2 * x^2)) / (2 * (6 - x)) else 0

theorem maximize_daily_profit (c : ℝ) (x : ℝ) (h1 : 0 < c) (h2 : c < 6) :
  (y = daily_profit x c) ∧
  (if 0 < c ∧ c < 3 then x = c else if 3 ≤ c ∧ c < 6 then x = 3 else False) :=
by
  sorry

end NUMINAMATH_GPT_maximize_daily_profit_l126_12681


namespace NUMINAMATH_GPT_percent_increase_l126_12620

theorem percent_increase (P x : ℝ) (h1 : P + x/100 * P - 0.2 * (P + x/100 * P) = P) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_l126_12620


namespace NUMINAMATH_GPT_players_count_l126_12655

def total_socks : ℕ := 22
def socks_per_player : ℕ := 2

theorem players_count : total_socks / socks_per_player = 11 :=
by
  sorry

end NUMINAMATH_GPT_players_count_l126_12655


namespace NUMINAMATH_GPT_odd_increasing_min_5_then_neg5_max_on_neg_interval_l126_12609

-- Definitions using the conditions given in the problem statement
variable {f : ℝ → ℝ}

-- Condition 1: f is odd
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Condition 2: f is increasing on the interval [3, 7]
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f (x) ≤ f (y)

-- Condition 3: Minimum value of f on [3, 7] is 5
def min_value_on_interval (f : ℝ → ℝ) (a b : ℝ) (min_val : ℝ) : Prop :=
  ∃ x, a ≤ x ∧ x ≤ b ∧ f (x) = min_val

-- Lean statement for the proof problem
theorem odd_increasing_min_5_then_neg5_max_on_neg_interval
  (f_odd: odd_function f)
  (f_increasing: increasing_on_interval f 3 7)
  (min_val: min_value_on_interval f 3 7 5) :
  increasing_on_interval f (-7) (-3) ∧ min_value_on_interval f (-7) (-3) (-5) :=
by sorry

end NUMINAMATH_GPT_odd_increasing_min_5_then_neg5_max_on_neg_interval_l126_12609


namespace NUMINAMATH_GPT_pentagon_edges_and_vertices_sum_l126_12646

theorem pentagon_edges_and_vertices_sum :
  let edges := 5
  let vertices := 5
  edges + vertices = 10 := by
  sorry

end NUMINAMATH_GPT_pentagon_edges_and_vertices_sum_l126_12646


namespace NUMINAMATH_GPT_entrants_total_l126_12648

theorem entrants_total (N : ℝ) (h1 : N > 800)
  (h2 : 0.35 * N = NumFemales)
  (h3 : 0.65 * N = NumMales)
  (h4 : NumMales - NumFemales = 252) :
  N = 840 := 
sorry

end NUMINAMATH_GPT_entrants_total_l126_12648


namespace NUMINAMATH_GPT_Jorge_is_24_years_younger_l126_12668

-- Define the conditions
def Jorge_age_2005 := 16
def Simon_age_2010 := 45

-- Prove that Jorge is 24 years younger than Simon
theorem Jorge_is_24_years_younger :
  (Simon_age_2010 - (Jorge_age_2005 + 5) = 24) :=
by
  sorry

end NUMINAMATH_GPT_Jorge_is_24_years_younger_l126_12668


namespace NUMINAMATH_GPT_opposite_of_two_l126_12617

def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_two : opposite 2 = -2 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_opposite_of_two_l126_12617


namespace NUMINAMATH_GPT_solution_set_l126_12627

open Real

noncomputable def condition (x : ℝ) := x ≥ 2

noncomputable def eq_1 (x : ℝ) := sqrt (x + 5 - 6 * sqrt (x - 2)) + sqrt (x + 12 - 8 * sqrt (x - 2)) = 2

theorem solution_set :
  {x : ℝ | condition x ∧ eq_1 x} = {x : ℝ | 11 ≤ x ∧ x ≤ 18} :=
by sorry

end NUMINAMATH_GPT_solution_set_l126_12627


namespace NUMINAMATH_GPT_paperclips_volume_75_l126_12651

noncomputable def paperclips (v : ℝ) : ℝ := 60 / Real.sqrt 27 * Real.sqrt v

theorem paperclips_volume_75 :
  paperclips 75 = 100 :=
by
  sorry

end NUMINAMATH_GPT_paperclips_volume_75_l126_12651


namespace NUMINAMATH_GPT_divisible_by_3_l126_12650

theorem divisible_by_3 :
  ∃ n : ℕ, (5 + 2 + n + 4 + 8) % 3 = 0 ∧ n = 2 := 
by
  sorry

end NUMINAMATH_GPT_divisible_by_3_l126_12650


namespace NUMINAMATH_GPT_points_three_units_away_from_neg3_l126_12675

theorem points_three_units_away_from_neg3 (x : ℝ) : (abs (x + 3) = 3) ↔ (x = 0 ∨ x = -6) :=
by
  sorry

end NUMINAMATH_GPT_points_three_units_away_from_neg3_l126_12675


namespace NUMINAMATH_GPT_tg_half_product_l126_12678

open Real

variable (α β : ℝ)

theorem tg_half_product (h1 : sin α + sin β = 2 * sin (α + β))
                        (h2 : ∀ n : ℤ, α + β ≠ 2 * π * n) :
  tan (α / 2) * tan (β / 2) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_tg_half_product_l126_12678


namespace NUMINAMATH_GPT_tiger_distance_traveled_l126_12600

theorem tiger_distance_traveled :
  let distance1 := 25 * 1
  let distance2 := 35 * 2
  let distance3 := 20 * 1.5
  let distance4 := 10 * 1
  let distance5 := 50 * 0.5
  distance1 + distance2 + distance3 + distance4 + distance5 = 160 := by
sorry

end NUMINAMATH_GPT_tiger_distance_traveled_l126_12600


namespace NUMINAMATH_GPT_gimbap_total_cost_l126_12649

theorem gimbap_total_cost :
  let basic_gimbap_cost := 2000
  let tuna_gimbap_cost := 3500
  let red_pepper_gimbap_cost := 3000
  let beef_gimbap_cost := 4000
  let nude_gimbap_cost := 3500
  let cost_of_two gimbaps := (tuna_gimbap_cost * 2) + (beef_gimbap_cost * 2) + (nude_gimbap_cost * 2)
  cost_of_two gimbaps = 22000 := 
by 
  sorry

end NUMINAMATH_GPT_gimbap_total_cost_l126_12649


namespace NUMINAMATH_GPT_probability_one_head_one_tail_l126_12653

def toss_outcomes : List (String × String) := [("head", "head"), ("head", "tail"), ("tail", "head"), ("tail", "tail")]

def favorable_outcomes (outcomes : List (String × String)) : List (String × String) :=
  outcomes.filter (fun x => (x = ("head", "tail")) ∨ (x = ("tail", "head")))

theorem probability_one_head_one_tail :
  (favorable_outcomes toss_outcomes).length / toss_outcomes.length = 1 / 2 :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_probability_one_head_one_tail_l126_12653


namespace NUMINAMATH_GPT_sum_mod_18_l126_12674

theorem sum_mod_18 :
  (65 + 66 + 67 + 68 + 69 + 70 + 71 + 72) % 18 = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_18_l126_12674


namespace NUMINAMATH_GPT_max_sequence_sum_l126_12695

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSequence (a1 d : α) (n : ℕ) : α :=
  a1 + d * n

noncomputable def sequenceSum (a1 d : α) (n : ℕ) : α :=
  n * (a1 + (a1 + d * (n - 1))) / 2

theorem max_sequence_sum (a1 d : α) (n : ℕ) (hn : 5 ≤ n ∧ n ≤ 10)
    (h1 : d < 0) (h2 : sequenceSum a1 d 5 = sequenceSum a1 d 10) :
    n = 7 ∨ n = 8 :=
  sorry

end NUMINAMATH_GPT_max_sequence_sum_l126_12695


namespace NUMINAMATH_GPT_walters_exceptional_days_l126_12696

variable (b w : ℕ)
variable (days_total dollars_total : ℕ)
variable (normal_earn exceptional_earn : ℕ)
variable (at_least_exceptional_days : ℕ)

-- Conditions
def conditions : Prop :=
  days_total = 15 ∧
  dollars_total = 70 ∧
  normal_earn = 4 ∧
  exceptional_earn = 6 ∧
  at_least_exceptional_days = 5 ∧
  b + w = days_total ∧
  normal_earn * b + exceptional_earn * w = dollars_total ∧
  w ≥ at_least_exceptional_days

-- Theorem to prove the number of exceptional days is 5
theorem walters_exceptional_days (h : conditions b w days_total dollars_total normal_earn exceptional_earn at_least_exceptional_days) : w = 5 :=
sorry

end NUMINAMATH_GPT_walters_exceptional_days_l126_12696


namespace NUMINAMATH_GPT_complex_problem_solution_l126_12691

noncomputable def complex_problem (c d : ℂ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : c^2 - c * d + d^2 = 0) : ℂ :=
  (c^12 + d^12) / (c + d)^12

theorem complex_problem_solution (c d : ℂ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : c^2 - c * d + d^2 = 0) :
  complex_problem c d h1 h2 h3 = 2 / 81 := 
sorry

end NUMINAMATH_GPT_complex_problem_solution_l126_12691


namespace NUMINAMATH_GPT_minimum_value_expression_l126_12642

theorem minimum_value_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l126_12642


namespace NUMINAMATH_GPT_initial_marbles_l126_12632

-- Define the conditions as constants
def marbles_given_to_Juan : ℕ := 73
def marbles_left_with_Connie : ℕ := 70

-- Prove that Connie initially had 143 marbles
theorem initial_marbles (initial_marbles : ℕ) :
  initial_marbles = marbles_given_to_Juan + marbles_left_with_Connie → 
  initial_marbles = 143 :=
by
  intro h
  rw [h]
  rfl

end NUMINAMATH_GPT_initial_marbles_l126_12632


namespace NUMINAMATH_GPT_at_least_one_pass_l126_12667

variable (n : ℕ) (p : ℝ)

theorem at_least_one_pass (h_p_range : 0 < p ∧ p < 1) :
  (1 - (1 - p) ^ n) = 1 - (1 - p) ^ n :=
sorry

end NUMINAMATH_GPT_at_least_one_pass_l126_12667


namespace NUMINAMATH_GPT_max_k_l126_12686

-- Definitions and conditions
def original_number (A B : ℕ) : ℕ := 10 * A + B
def new_number (A C B : ℕ) : ℕ := 100 * A + 10 * C + B

theorem max_k (A C B k : ℕ) (hA : A ≠ 0) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : 0 ≤ B ∧ B ≤ 9) (h3: 0 ≤ C ∧ C ≤ 9) :
  ((original_number A B) * k = (new_number A C B)) → 
  (∀ (A: ℕ), 1 ≤ k) → 
  k ≤ 19 :=
by
  sorry

end NUMINAMATH_GPT_max_k_l126_12686


namespace NUMINAMATH_GPT_radish_patch_area_l126_12611

-- Definitions from the conditions
variables (R P : ℕ) -- R: area of radish patch, P: area of pea patch
variable (h1 : P = 2 * R) -- The pea patch is twice as large as the radish patch
variable (h2 : P / 6 = 5) -- One-sixth of the pea patch is 5 square feet

-- Goal statement
theorem radish_patch_area : R = 15 :=
by
  sorry

end NUMINAMATH_GPT_radish_patch_area_l126_12611


namespace NUMINAMATH_GPT_initial_blue_balls_l126_12685

-- Define the initial conditions
variables (B : ℕ) (total_balls : ℕ := 15) (removed_blue_balls : ℕ := 3)
variable (prob_after_removal : ℚ := 1 / 3)
variable (remaining_balls : ℕ := total_balls - removed_blue_balls)
variable (remaining_blue_balls : ℕ := B - removed_blue_balls)

-- State the theorem
theorem initial_blue_balls : 
  remaining_balls = 12 → remaining_blue_balls = remaining_balls * prob_after_removal → B = 7 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_initial_blue_balls_l126_12685


namespace NUMINAMATH_GPT_rectangular_garden_length_l126_12692

theorem rectangular_garden_length (P B L : ℕ) (h1 : P = 1800) (h2 : B = 400) (h3 : P = 2 * (L + B)) : L = 500 :=
sorry

end NUMINAMATH_GPT_rectangular_garden_length_l126_12692


namespace NUMINAMATH_GPT_minimum_possible_length_of_third_side_l126_12665

theorem minimum_possible_length_of_third_side (a b : ℝ) (h : a = 8 ∧ b = 15 ∨ a = 15 ∧ b = 8) : 
  ∃ c : ℝ, (c * c = a * a + b * b ∨ c * c = a * a - b * b ∨ c * c = b * b - a * a) ∧ c = Real.sqrt 161 :=
by
  sorry

end NUMINAMATH_GPT_minimum_possible_length_of_third_side_l126_12665


namespace NUMINAMATH_GPT_find_n_from_exponent_equation_l126_12697

theorem find_n_from_exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_n_from_exponent_equation_l126_12697


namespace NUMINAMATH_GPT_bird_families_to_Asia_l126_12614

theorem bird_families_to_Asia (total_families initial_families left_families went_to_Africa went_to_Asia: ℕ) 
  (h1 : total_families = 85) 
  (h2 : went_to_Africa = 23) 
  (h3 : left_families = 25) 
  (h4 : went_to_Asia = total_families - left_families - went_to_Africa) 
  : went_to_Asia = 37 := 
by 
  rw [h1, h2, h3] at h4 
  simp at h4 
  exact h4

end NUMINAMATH_GPT_bird_families_to_Asia_l126_12614


namespace NUMINAMATH_GPT_exist_n_div_k_l126_12689

open Function

theorem exist_n_div_k (k : ℕ) (h1 : k ≥ 1) (h2 : Nat.gcd k 6 = 1) :
  ∃ n : ℕ, n ≥ 0 ∧ k ∣ (2^n + 3^n + 6^n - 1) := 
sorry

end NUMINAMATH_GPT_exist_n_div_k_l126_12689


namespace NUMINAMATH_GPT_parallel_perpendicular_implies_perpendicular_l126_12640

-- Definitions of the geometric relationships
variables {Line Plane : Type}
variables (a b : Line) (alpha beta : Plane)

-- Conditions as per the problem statement
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Lean statement of the proof problem
theorem parallel_perpendicular_implies_perpendicular
  (h1 : parallel_line_plane a alpha)
  (h2 : perpendicular_line_plane b alpha) :
  perpendicular_lines a b :=  
sorry

end NUMINAMATH_GPT_parallel_perpendicular_implies_perpendicular_l126_12640


namespace NUMINAMATH_GPT_max_value_expression_l126_12604

theorem max_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (eq_condition : x^2 - 3 * x * y + 4 * y^2 - z = 0) : 
  ∃ (M : ℝ), M = 1 ∧ (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x^2 - 3 * x * y + 4 * y^2 - z = 0 → (2/x + 1/y - 2/z) ≤ M) := 
by
  sorry

end NUMINAMATH_GPT_max_value_expression_l126_12604


namespace NUMINAMATH_GPT_gnomes_red_hats_small_noses_l126_12641

theorem gnomes_red_hats_small_noses :
  ∀ (total_gnomes red_hats blue_hats big_noses_blue_hats : ℕ),
  total_gnomes = 28 →
  red_hats = (3 * total_gnomes) / 4 →
  blue_hats = total_gnomes - red_hats →
  big_noses_blue_hats = 6 →
  (total_gnomes / 2) - big_noses_blue_hats = 8 →
  red_hats - 8 = 13 :=
by
  intros total_gnomes red_hats blue_hats big_noses_blue_hats
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_gnomes_red_hats_small_noses_l126_12641


namespace NUMINAMATH_GPT_problem_2002_multiples_l126_12647

theorem problem_2002_multiples :
  ∃ (n : ℕ), 
    n = 1800 ∧
    (∀ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 149 →
      2002 ∣ (10^j - 10^i) ↔ j - i ≡ 0 [MOD 6]) :=
sorry

end NUMINAMATH_GPT_problem_2002_multiples_l126_12647


namespace NUMINAMATH_GPT_find_f_l126_12670

theorem find_f (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x > 0)
  (h2 : f 1 = 1)
  (h3 : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2) : ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_GPT_find_f_l126_12670


namespace NUMINAMATH_GPT_bus_stoppage_time_per_hour_l126_12607

theorem bus_stoppage_time_per_hour
  (speed_excluding_stoppages : ℕ) 
  (speed_including_stoppages : ℕ)
  (h1 : speed_excluding_stoppages = 54) 
  (h2 : speed_including_stoppages = 45) 
  : (60 * (speed_excluding_stoppages - speed_including_stoppages) / speed_excluding_stoppages) = 10 :=
by sorry

end NUMINAMATH_GPT_bus_stoppage_time_per_hour_l126_12607


namespace NUMINAMATH_GPT_employed_females_percentage_l126_12652

theorem employed_females_percentage (total_employed_percentage employed_males_percentage employed_females_percentage : ℝ) 
    (h1 : total_employed_percentage = 64) 
    (h2 : employed_males_percentage = 48) 
    (h3 : employed_females_percentage = total_employed_percentage - employed_males_percentage) :
    (employed_females_percentage / total_employed_percentage * 100) = 25 :=
by
  sorry

end NUMINAMATH_GPT_employed_females_percentage_l126_12652


namespace NUMINAMATH_GPT_probability_of_composite_l126_12601

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ m < n ∧ 1 < k ∧ k < n ∧ m * k = n

def dice_outcomes (faces : ℕ) (rolls : ℕ) : ℕ :=
  faces ^ rolls

def non_composite_product_ways : ℕ :=
  1 + (3 * 4)  -- one way for all 1s, plus combinations of (1,1,1,{2,3,5})

def total_outcomes : ℕ :=
  dice_outcomes 6 4  -- 6^4 total possible outcomes

def probability_composite : ℚ :=
  1 - (non_composite_product_ways / total_outcomes)

theorem probability_of_composite:
  probability_composite = 1283 / 1296 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_composite_l126_12601


namespace NUMINAMATH_GPT_smallest_integer_m_l126_12628

theorem smallest_integer_m (m : ℕ) : m > 1 ∧ m % 13 = 2 ∧ m % 5 = 2 ∧ m % 3 = 2 → m = 197 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_integer_m_l126_12628


namespace NUMINAMATH_GPT_garden_dimensions_l126_12602

theorem garden_dimensions (l b : ℝ) (walkway_width total_area perimeter : ℝ) 
  (h1 : l = 3 * b)
  (h2 : perimeter = 2 * l + 2 * b)
  (h3 : walkway_width = 1)
  (h4 : total_area = (l + 2 * walkway_width) * (b + 2 * walkway_width))
  (h5 : perimeter = 40)
  (h6 : total_area = 120) : 
  l = 15 ∧ b = 5 ∧ total_area - l * b = 45 :=  
  by
  sorry

end NUMINAMATH_GPT_garden_dimensions_l126_12602


namespace NUMINAMATH_GPT_scoops_of_natural_seedless_raisins_l126_12693

theorem scoops_of_natural_seedless_raisins 
  (cost_natural : ℝ := 3.45) 
  (cost_golden : ℝ := 2.55) 
  (num_golden : ℝ := 20) 
  (cost_mixture : ℝ := 3) : 
  ∃ x : ℝ, (3.45 * x + 20 * 2.55 = 3 * (x + 20)) ∧ x = 20 :=
sorry

end NUMINAMATH_GPT_scoops_of_natural_seedless_raisins_l126_12693


namespace NUMINAMATH_GPT_james_bike_ride_total_distance_l126_12605

theorem james_bike_ride_total_distance 
  (d1 d2 d3 : ℝ)
  (H1 : d2 = 12)
  (H2 : d2 = 1.2 * d1)
  (H3 : d3 = 1.25 * d2) :
  d1 + d2 + d3 = 37 :=
by
  -- additional proof steps would go here
  sorry

end NUMINAMATH_GPT_james_bike_ride_total_distance_l126_12605


namespace NUMINAMATH_GPT_find_product_l126_12699

theorem find_product
  (a b c d : ℝ) :
  3 * a + 2 * b + 4 * c + 6 * d = 60 →
  4 * (d + c) = b^2 →
  4 * b + 2 * c = a →
  c - 2 = d →
  a * b * c * d = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_product_l126_12699


namespace NUMINAMATH_GPT_proof_of_a_neg_two_l126_12683

theorem proof_of_a_neg_two (a : ℝ) (i : ℂ) (h_i : i^2 = -1) (h_real : (1 + i)^2 - a / i = (a + 2) * i → ∃ r : ℝ, (1 + i)^2 - a / i = r) : a = -2 :=
sorry

end NUMINAMATH_GPT_proof_of_a_neg_two_l126_12683


namespace NUMINAMATH_GPT_vinny_fifth_month_loss_l126_12619

theorem vinny_fifth_month_loss (start_weight : ℝ) (end_weight : ℝ) (first_month_loss : ℝ) (second_month_loss : ℝ) (third_month_loss : ℝ) (fourth_month_loss : ℝ) (total_loss : ℝ):
  start_weight = 300 ∧
  first_month_loss = 20 ∧
  second_month_loss = first_month_loss / 2 ∧
  third_month_loss = second_month_loss / 2 ∧
  fourth_month_loss = third_month_loss / 2 ∧
  (start_weight - end_weight) = total_loss ∧
  end_weight = 250.5 →
  (total_loss - (first_month_loss + second_month_loss + third_month_loss + fourth_month_loss)) = 12 :=
by
  sorry

end NUMINAMATH_GPT_vinny_fifth_month_loss_l126_12619


namespace NUMINAMATH_GPT_coefficient_of_8th_term_l126_12656

-- Define the general term of the binomial expansion
def binomial_expansion_term (n r : ℕ) (a b : ℕ) : ℕ := 
  Nat.choose n r * a^(n - r) * b^r

-- Define the specific scenario given in the problem
def specific_binomial_expansion_term : ℕ := 
  binomial_expansion_term 8 7 2 1  -- a = 2, b = x (consider b as 1 for coefficient calculation)

-- Problem statement to prove the coefficient of the 8th term is 16
theorem coefficient_of_8th_term : specific_binomial_expansion_term = 16 := by
  sorry

end NUMINAMATH_GPT_coefficient_of_8th_term_l126_12656


namespace NUMINAMATH_GPT_hexagon_area_is_32_l126_12661

noncomputable def area_of_hexagon : ℝ := 
  let p0 : ℝ × ℝ := (0, 0)
  let p1 : ℝ × ℝ := (2, 4)
  let p2 : ℝ × ℝ := (5, 4)
  let p3 : ℝ × ℝ := (7, 0)
  let p4 : ℝ × ℝ := (5, -4)
  let p5 : ℝ × ℝ := (2, -4)
  -- Triangle 1: p0, p1, p2
  let area_tri1 := 1 / 2 * (3 : ℝ) * (4 : ℝ)
  -- Triangle 2: p2, p3, p4
  let area_tri2 := 1 / 2 * (8 : ℝ) * (2 : ℝ)
  -- Triangle 3: p4, p5, p0
  let area_tri3 := 1 / 2 * (3 : ℝ) * (4 : ℝ)
  -- Triangle 4: p1, p2, p5
  let area_tri4 := 1 / 2 * (8 : ℝ) * (3 : ℝ)
  area_tri1 + area_tri2 + area_tri3 + area_tri4

theorem hexagon_area_is_32 : area_of_hexagon = 32 := 
by
  sorry

end NUMINAMATH_GPT_hexagon_area_is_32_l126_12661


namespace NUMINAMATH_GPT_range_of_a_l126_12673

-- Defining the function f(x)
def f (a x : ℝ) := x^2 + (a^2 - 1) * x + (a - 2)

-- The statement of the problem in Lean 4
theorem range_of_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 > 1 ∧ x2 < 1 ∧ f a x1 = 0 ∧ f a x2 = 0) : -2 < a ∧ a < 1 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_range_of_a_l126_12673


namespace NUMINAMATH_GPT_granger_bought_4_loaves_of_bread_l126_12616

-- Define the prices of items
def price_of_spam : Nat := 3
def price_of_pb : Nat := 5
def price_of_bread : Nat := 2

-- Define the quantities bought by Granger
def qty_spam : Nat := 12
def qty_pb : Nat := 3
def total_amount_paid : Nat := 59

-- The problem statement in Lean: Prove the number of loaves of bread bought
theorem granger_bought_4_loaves_of_bread :
  (qty_spam * price_of_spam) + (qty_pb * price_of_pb) + (4 * price_of_bread) = total_amount_paid :=
sorry

end NUMINAMATH_GPT_granger_bought_4_loaves_of_bread_l126_12616


namespace NUMINAMATH_GPT_matt_and_peter_worked_together_days_l126_12682

variables (W : ℝ) -- Represents total work
noncomputable def work_rate_peter := W / 35
noncomputable def work_rate_together := W / 20

theorem matt_and_peter_worked_together_days (x : ℝ) :
  (x / 20) + (14 / 35) = 1 → x = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_matt_and_peter_worked_together_days_l126_12682


namespace NUMINAMATH_GPT_area_circle_l126_12615

-- Define the given condition
def polar_eq (r θ : ℝ) : Prop :=
  r = 3 * Real.cos θ - 4 * Real.sin θ

-- The goal is to prove the area of the circle described by the polar equation
theorem area_circle {r θ : ℝ} (h : polar_eq r θ) :
  ∃ A, A = π * (5 / 2) ^ 2 :=
sorry

end NUMINAMATH_GPT_area_circle_l126_12615


namespace NUMINAMATH_GPT_fans_per_set_l126_12639

theorem fans_per_set (total_fans : ℕ) (sets_of_bleachers : ℕ) (fans_per_set : ℕ)
  (h1 : total_fans = 2436) (h2 : sets_of_bleachers = 3) : fans_per_set = 812 :=
by
  sorry

end NUMINAMATH_GPT_fans_per_set_l126_12639


namespace NUMINAMATH_GPT_insurance_not_covered_percentage_l126_12690

noncomputable def insurance_monthly_cost : ℝ := 20
noncomputable def insurance_months : ℝ := 24
noncomputable def procedure_cost : ℝ := 5000
noncomputable def amount_saved : ℝ := 3520

theorem insurance_not_covered_percentage :
  ((procedure_cost - amount_saved - (insurance_monthly_cost * insurance_months)) / procedure_cost) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_insurance_not_covered_percentage_l126_12690


namespace NUMINAMATH_GPT_percentage_B_to_C_l126_12698

variables (total_students : ℕ)
variables (pct_A pct_B pct_C pct_A_to_C pct_B_to_C : ℝ)

-- Given conditions
axiom total_students_eq_100 : total_students = 100
axiom pct_A_eq_60 : pct_A = 60
axiom pct_B_eq_40 : pct_B = 40
axiom pct_A_to_C_eq_30 : pct_A_to_C = 30
axiom pct_C_eq_34 : pct_C = 34

-- Proof goal
theorem percentage_B_to_C :
  pct_B_to_C = 40 :=
sorry

end NUMINAMATH_GPT_percentage_B_to_C_l126_12698


namespace NUMINAMATH_GPT_magazines_sold_l126_12663

theorem magazines_sold (total_sold : Float) (newspapers_sold : Float) (magazines_sold : Float)
  (h1 : total_sold = 425.0)
  (h2 : newspapers_sold = 275.0) :
  magazines_sold = total_sold - newspapers_sold :=
by
  sorry

#check magazines_sold

end NUMINAMATH_GPT_magazines_sold_l126_12663


namespace NUMINAMATH_GPT_income_expenditure_ratio_l126_12621

theorem income_expenditure_ratio (I E S : ℕ) (hI : I = 19000) (hS : S = 11400) (hRel : S = I - E) :
  I / E = 95 / 38 :=
by
  sorry

end NUMINAMATH_GPT_income_expenditure_ratio_l126_12621


namespace NUMINAMATH_GPT_fewest_coach_handshakes_l126_12684

theorem fewest_coach_handshakes (n_A n_B k_A k_B : ℕ) (h1 : n_A = n_B + 2)
    (h2 : ((n_A * (n_A - 1)) / 2) + ((n_B * (n_B - 1)) / 2) + (n_A * n_B) + k_A + k_B = 620) :
  k_A + k_B = 189 := 
sorry

end NUMINAMATH_GPT_fewest_coach_handshakes_l126_12684


namespace NUMINAMATH_GPT_number_of_pipes_l126_12660

theorem number_of_pipes (h_same_height : forall (height : ℝ), height > 0)
  (diam_large : ℝ) (hl : diam_large = 6)
  (diam_small : ℝ) (hs : diam_small = 1) :
  (π * (diam_large / 2)^2) / (π * (diam_small / 2)^2) = 36 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pipes_l126_12660


namespace NUMINAMATH_GPT_pencils_multiple_of_28_l126_12624

theorem pencils_multiple_of_28 (students pens pencils : ℕ) 
  (h1 : students = 28) 
  (h2 : pens = 1204) 
  (h3 : ∃ k, pens = students * k) 
  (h4 : ∃ n, pencils = students * n) : 
  ∃ m, pencils = 28 * m :=
by
  sorry

end NUMINAMATH_GPT_pencils_multiple_of_28_l126_12624


namespace NUMINAMATH_GPT_ellipse_area_l126_12669

theorem ellipse_area (P : ℝ) (b : ℝ) (a : ℝ) (A : ℝ) (h1 : P = 18)
  (h2 : a = b + 4)
  (h3 : A = π * a * b) :
  A = 5 * π :=
by
  sorry

end NUMINAMATH_GPT_ellipse_area_l126_12669


namespace NUMINAMATH_GPT_sum_of_solutions_l126_12630

theorem sum_of_solutions (y : ℝ) (h : y^2 = 25) : ∃ (a b : ℝ), (a = 5 ∨ a = -5) ∧ (b = 5 ∨ b = -5) ∧ a + b = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_l126_12630


namespace NUMINAMATH_GPT_find_k_l126_12680

-- Definitions
variable (m n k : ℝ)

-- Given conditions
def on_line_1 : Prop := m = 2 * n + 5
def on_line_2 : Prop := (m + 5) = 2 * (n + k) + 5

-- Desired conclusion
theorem find_k (h1 : on_line_1 m n) (h2 : on_line_2 m n k) : k = 2.5 :=
sorry

end NUMINAMATH_GPT_find_k_l126_12680


namespace NUMINAMATH_GPT_endpoint_correctness_l126_12659

-- Define two points in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define start point (2, 2)
def startPoint : Point := ⟨2, 2⟩

-- Define the endpoint's conditions
def endPoint (x y : ℝ) : Prop :=
  y = 2 * x + 1 ∧ (x > 0) ∧ (Real.sqrt ((x - startPoint.x) ^ 2 + (y - startPoint.y) ^ 2) = 6)

-- The solution to the problem proving (3.4213, 7.8426) satisfies the conditions
theorem endpoint_correctness : ∃ (x y : ℝ), endPoint x y ∧ x = 3.4213 ∧ y = 7.8426 := by
  use 3.4213
  use 7.8426
  sorry

end NUMINAMATH_GPT_endpoint_correctness_l126_12659


namespace NUMINAMATH_GPT_prob_task1_and_not_task2_l126_12679

def prob_task1_completed : ℚ := 5 / 8
def prob_task2_completed : ℚ := 3 / 5

theorem prob_task1_and_not_task2 : 
  ((prob_task1_completed) * (1 - prob_task2_completed)) = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_prob_task1_and_not_task2_l126_12679


namespace NUMINAMATH_GPT_curves_intersect_at_three_points_l126_12606

theorem curves_intersect_at_three_points (b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = b^2 ∧ y = 2 * x^2 - b) ∧ 
  (∀ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (x₁^2 + y₁^2 = b^2) ∧ (x₂^2 + y₂^2 = b^2) ∧ (x₃^2 + y₃^2 = b^2) ∧
    (y₁ = 2 * x₁^2 - b) ∧ (y₂ = 2 * x₂^2 - b) ∧ (y₃ = 2 * x₃^2 - b)) ↔ b > 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_curves_intersect_at_three_points_l126_12606


namespace NUMINAMATH_GPT_coordinates_with_respect_to_origin_l126_12676

theorem coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (2, -6)) : (x, y) = (2, -6) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_with_respect_to_origin_l126_12676


namespace NUMINAMATH_GPT_fixed_point_of_line_l126_12671

theorem fixed_point_of_line (m : ℝ) : 
  (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_line_l126_12671


namespace NUMINAMATH_GPT_original_denominator_is_21_l126_12631

theorem original_denominator_is_21 (d : ℕ) : (3 + 6) / (d + 6) = 1 / 3 → d = 21 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_original_denominator_is_21_l126_12631


namespace NUMINAMATH_GPT_vehicle_capacity_rental_plans_l126_12658

variables (a b x y : ℕ)

/-- Conditions -/
axiom cond1 : 2*x + y = 11
axiom cond2 : x + 2*y = 13

/-- Resulting capacities for each vehicle type -/
theorem vehicle_capacity : 
  x = 3 ∧ y = 5 :=
by
  sorry

/-- Rental plans for transporting 33 tons of drugs -/
theorem rental_plans :
  3*a + 5*b = 33 ∧ ((a = 6 ∧ b = 3) ∨ (a = 1 ∧ b = 6)) :=
by
  sorry

end NUMINAMATH_GPT_vehicle_capacity_rental_plans_l126_12658


namespace NUMINAMATH_GPT_triangle_has_three_altitudes_l126_12672

-- Assuming a triangle in ℝ² space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Definition of an altitude in the context of Lean
def altitude (T : Triangle) (p : ℝ × ℝ) := 
  ∃ (a : ℝ) (b : ℝ), T.A.1 * p.1 + T.A.2 * p.2 = a * p.1 + b -- Placeholder, real definition of altitude may vary

-- Prove that a triangle has exactly 3 altitudes
theorem triangle_has_three_altitudes (T : Triangle) : ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
  altitude T p₁ ∧ altitude T p₂ ∧ altitude T p₃ :=
sorry

end NUMINAMATH_GPT_triangle_has_three_altitudes_l126_12672


namespace NUMINAMATH_GPT_travel_time_l126_12636

def speed : ℝ := 60  -- Speed of the car in miles per hour
def distance : ℝ := 300  -- Distance to the campground in miles

theorem travel_time : distance / speed = 5 := by
  sorry

end NUMINAMATH_GPT_travel_time_l126_12636


namespace NUMINAMATH_GPT_students_on_field_trip_l126_12688

theorem students_on_field_trip 
    (vans : ℕ)
    (van_capacity : ℕ)
    (adults : ℕ)
    (students : ℕ)
    (H1 : vans = 3)
    (H2 : van_capacity = 8)
    (H3 : adults = 2)
    (H4 : students = vans * van_capacity - adults) :
    students = 22 := 
by 
  sorry

end NUMINAMATH_GPT_students_on_field_trip_l126_12688


namespace NUMINAMATH_GPT_determine_p_and_q_l126_12612

theorem determine_p_and_q (x p q : ℝ) : 
  (x + 4) * (x - 1) = x^2 + p * x + q → (p = 3 ∧ q = -4) := 
by 
  sorry

end NUMINAMATH_GPT_determine_p_and_q_l126_12612


namespace NUMINAMATH_GPT_max_combined_weight_l126_12645

theorem max_combined_weight (E A : ℕ) (h1 : A = 2 * E) (h2 : A + E = 90) (w_A : ℕ := 5) (w_E : ℕ := 2 * w_A) :
  E * w_E + A * w_A = 600 :=
by
  sorry

end NUMINAMATH_GPT_max_combined_weight_l126_12645


namespace NUMINAMATH_GPT_john_spent_at_candy_store_l126_12654

-- Conditions
def weekly_allowance : ℚ := 2.25
def spent_at_arcade : ℚ := (3 / 5) * weekly_allowance
def remaining_after_arcade : ℚ := weekly_allowance - spent_at_arcade
def spent_at_toy_store : ℚ := (1 / 3) * remaining_after_arcade
def remaining_after_toy_store : ℚ := remaining_after_arcade - spent_at_toy_store

-- Problem: Prove that John spent $0.60 at the candy store
theorem john_spent_at_candy_store : remaining_after_toy_store = 0.60 :=
by
  sorry

end NUMINAMATH_GPT_john_spent_at_candy_store_l126_12654


namespace NUMINAMATH_GPT_projection_of_3_neg2_onto_v_l126_12610

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1 + a.2 * b.2)
  let scalar := (dot_product u v) / (dot_product v v)
  (scalar * v.1, scalar * v.2)

def v : ℝ × ℝ := (2, -8)

theorem projection_of_3_neg2_onto_v :
  projection (3, -2) v = (11/17, -44/17) :=
by sorry

end NUMINAMATH_GPT_projection_of_3_neg2_onto_v_l126_12610


namespace NUMINAMATH_GPT_relationship_between_sets_l126_12657

def M (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k - 2
def P (x : ℤ) : Prop := ∃ n : ℤ, x = 5 * n + 3
def S (x : ℤ) : Prop := ∃ m : ℤ, x = 10 * m + 3

theorem relationship_between_sets :
  (∀ x, S x → P x) ∧ (∀ x, P x → M x) ∧ (∀ x, M x → P x) :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_sets_l126_12657


namespace NUMINAMATH_GPT_domain_of_f_l126_12608

noncomputable def f (x : ℝ) := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ x + 1 ≠ 1 ∧ 4 - x ≥ 0} = { x : ℝ | (-1 < x ∧ x ≤ 4) ∧ x ≠ 0 } :=
sorry

end NUMINAMATH_GPT_domain_of_f_l126_12608


namespace NUMINAMATH_GPT_panic_percentage_l126_12633

theorem panic_percentage (original_population disappeared_after first_population second_population : ℝ) 
  (h₁ : original_population = 7200)
  (h₂ : disappeared_after = original_population * 0.10)
  (h₃ : first_population = original_population - disappeared_after)
  (h₄ : second_population = 4860)
  (h₅ : second_population = first_population - (first_population * 0.25)) : 
  second_population = first_population * (1 - 0.25) :=
by
  sorry

end NUMINAMATH_GPT_panic_percentage_l126_12633


namespace NUMINAMATH_GPT_polynomial_divisibility_by_120_l126_12687

theorem polynomial_divisibility_by_120 (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_by_120_l126_12687


namespace NUMINAMATH_GPT_solve_system_eq_l126_12677

theorem solve_system_eq (x y : ℝ) :
  x^2 * y - x * y^2 - 5 * x + 5 * y + 3 = 0 ∧
  x^3 * y - x * y^3 - 5 * x^2 + 5 * y^2 + 15 = 0 ↔
  x = 4 ∧ y = 1 :=
sorry

end NUMINAMATH_GPT_solve_system_eq_l126_12677
