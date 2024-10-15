import Mathlib

namespace NUMINAMATH_GPT_sequence_polynomial_exists_l1373_137388

noncomputable def sequence_exists (k : ℕ) : Prop :=
∃ u : ℕ → ℝ,
  (∀ n : ℕ, u (n + 1) - u n = (n : ℝ) ^ k) ∧
  (∃ p : Polynomial ℝ, (∀ n : ℕ, u n = Polynomial.eval (n : ℝ) p) ∧ p.degree = k + 1 ∧ p.leadingCoeff = 1 / (k + 1))

theorem sequence_polynomial_exists (k : ℕ) : sequence_exists k :=
sorry

end NUMINAMATH_GPT_sequence_polynomial_exists_l1373_137388


namespace NUMINAMATH_GPT_probability_sales_greater_than_10000_l1373_137376

/-- Define the probability that the sales of new energy vehicles in a randomly selected city are greater than 10000 -/
theorem probability_sales_greater_than_10000 :
  (1 / 2) * (2 / 10) + (1 / 2) * (6 / 10) = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_probability_sales_greater_than_10000_l1373_137376


namespace NUMINAMATH_GPT_hyperbola_asymptote_value_of_a_l1373_137308

-- Define the hyperbola and the conditions given
variables {a : ℝ} (h1 : a > 0) (h2 : ∀ x y : ℝ, 3 * x + 2 * y = 0 ∧ 3 * x - 2 * y = 0)

theorem hyperbola_asymptote_value_of_a :
  a = 2 := by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_value_of_a_l1373_137308


namespace NUMINAMATH_GPT_closest_multiple_of_21_to_2023_l1373_137325

theorem closest_multiple_of_21_to_2023 : ∃ k : ℤ, k * 21 = 2022 ∧ ∀ m : ℤ, m * 21 = 2023 → (abs (m - 2023)) > (abs (2022 - 2023)) :=
by
  sorry

end NUMINAMATH_GPT_closest_multiple_of_21_to_2023_l1373_137325


namespace NUMINAMATH_GPT_smallest_collected_l1373_137384

noncomputable def Yoongi_collections : ℕ := 4
noncomputable def Jungkook_collections : ℕ := 6 / 3
noncomputable def Yuna_collections : ℕ := 5

theorem smallest_collected : min (min Yoongi_collections Jungkook_collections) Yuna_collections = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_collected_l1373_137384


namespace NUMINAMATH_GPT_exercise_serial_matches_year_problem_serial_matches_year_l1373_137336

-- Definitions for the exercise
def exercise_initial := 1169
def exercises_per_issue := 8
def issues_per_year := 9
def exercise_year := 1979
def exercises_per_year := exercises_per_issue * issues_per_year

-- Definitions for the problem
def problem_initial := 1576
def problems_per_issue := 8
def problems_per_year := problems_per_issue * issues_per_year
def problem_year := 1973

theorem exercise_serial_matches_year :
  ∃ (issue_number : ℕ) (exercise_number : ℕ),
    (issue_number = 3) ∧
    (exercise_number = 2) ∧
    (exercise_initial + 11 * exercises_per_year + 16 = exercise_year) :=
by {
  sorry
}

theorem problem_serial_matches_year :
  ∃ (issue_number : ℕ) (problem_number : ℕ),
    (issue_number = 5) ∧
    (problem_number = 5) ∧
    (problem_initial + 5 * problems_per_year + 36 = problem_year) :=
by {
  sorry
}

end NUMINAMATH_GPT_exercise_serial_matches_year_problem_serial_matches_year_l1373_137336


namespace NUMINAMATH_GPT_cost_whitewashing_l1373_137358

theorem cost_whitewashing
  (length : ℝ) (breadth : ℝ) (height : ℝ)
  (door_height : ℝ) (door_width : ℝ)
  (window_height : ℝ) (window_width : ℝ)
  (num_windows : ℕ) (cost_per_square_foot : ℝ)
  (room_dimensions : length = 25 ∧ breadth = 15 ∧ height = 12)
  (door_dimensions : door_height = 6 ∧ door_width = 3)
  (window_dimensions : window_height = 4 ∧ window_width = 3)
  (num_windows_condition : num_windows = 3)
  (cost_condition : cost_per_square_foot = 8) :
  (2 * (length + breadth) * height - (door_height * door_width + num_windows * window_height * window_width)) * cost_per_square_foot = 7248 := 
by
  sorry

end NUMINAMATH_GPT_cost_whitewashing_l1373_137358


namespace NUMINAMATH_GPT_percentage_slump_in_business_l1373_137364

theorem percentage_slump_in_business (X Y : ℝ) (h1 : 0.04 * X = 0.05 * Y) : 
  (1 - Y / X) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_slump_in_business_l1373_137364


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_conditions_l1373_137375

theorem necessary_but_not_sufficient_conditions (x y : ℝ) :
  (|x| ≤ 1 ∧ |y| ≤ 1) → x^2 + y^2 ≤ 1 ∨ ¬(x^2 + y^2 ≤ 1) → 
  (|x| ≤ 1 ∧ |y| ≤ 1) → (x^2 + y^2 ≤ 1 → (|x| ≤ 1 ∧ |y| ≤ 1)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_conditions_l1373_137375


namespace NUMINAMATH_GPT_brownie_pan_dimensions_l1373_137315

def brownie_dimensions (m n : ℕ) : Prop :=
  let numSectionsLength := m - 1
  let numSectionsWidth := n - 1
  let totalPieces := (numSectionsLength + 1) * (numSectionsWidth + 1)
  let interiorPieces := (numSectionsLength - 1) * (numSectionsWidth - 1)
  let perimeterPieces := totalPieces - interiorPieces
  (numSectionsLength = 3) ∧ (numSectionsWidth = 5) ∧ (interiorPieces = 2 * perimeterPieces)

theorem brownie_pan_dimensions :
  ∃ (m n : ℕ), brownie_dimensions m n ∧ m = 6 ∧ n = 12 :=
by
  existsi 6
  existsi 12
  unfold brownie_dimensions
  simp
  exact sorry

end NUMINAMATH_GPT_brownie_pan_dimensions_l1373_137315


namespace NUMINAMATH_GPT_max_sides_in_subpolygon_l1373_137379

/-- In a convex 1950-sided polygon with all its diagonals drawn, the polygon with the greatest number of sides among these smaller polygons can have at most 1949 sides. -/
theorem max_sides_in_subpolygon (n : ℕ) (hn : n = 1950) : 
  ∃ p : ℕ, p = 1949 ∧ ∀ m, m ≤ n-2 → m ≤ 1949 :=
sorry

end NUMINAMATH_GPT_max_sides_in_subpolygon_l1373_137379


namespace NUMINAMATH_GPT_total_leaves_on_farm_l1373_137368

noncomputable def number_of_branches : ℕ := 10
noncomputable def sub_branches_per_branch : ℕ := 40
noncomputable def leaves_per_sub_branch : ℕ := 60
noncomputable def number_of_trees : ℕ := 4

theorem total_leaves_on_farm :
  number_of_branches * sub_branches_per_branch * leaves_per_sub_branch * number_of_trees = 96000 :=
by
  sorry

end NUMINAMATH_GPT_total_leaves_on_farm_l1373_137368


namespace NUMINAMATH_GPT_horner_method_v2_l1373_137344

def f(x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.reverse.foldl (λ acc c => acc * x + c) 0

theorem horner_method_v2 :
  horner_eval [1, 5, 10, 10, 5, 1] 2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_horner_method_v2_l1373_137344


namespace NUMINAMATH_GPT_range_of_a_l1373_137395

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1373_137395


namespace NUMINAMATH_GPT_smallest_four_digit_product_is_12_l1373_137341

theorem smallest_four_digit_product_is_12 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
           (∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧ a * b * c * d = 12 ∧ a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 6) ∧
           (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 →
                     (∃ a' b' c' d' : ℕ, m = 1000 * a' + 100 * b' + 10 * c' + d' ∧ a' * b' * c' * d' = 12) →
                     n ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_product_is_12_l1373_137341


namespace NUMINAMATH_GPT_largest_of_five_l1373_137373

def a : ℝ := 0.994
def b : ℝ := 0.9399
def c : ℝ := 0.933
def d : ℝ := 0.9940
def e : ℝ := 0.9309

theorem largest_of_five : (a > b ∧ a > c ∧ a ≥ d ∧ a > e) := by
  -- We add sorry here to skip the proof
  sorry

end NUMINAMATH_GPT_largest_of_five_l1373_137373


namespace NUMINAMATH_GPT_proposition_four_l1373_137323

variables (a b c : Type)

noncomputable def perpend_lines (a b : Type) : Prop := sorry
noncomputable def parallel_lines (a b : Type) : Prop := sorry

theorem proposition_four (a b c : Type) 
  (h1 : perpend_lines a b) (h2 : parallel_lines b c) :
  perpend_lines a c :=
sorry

end NUMINAMATH_GPT_proposition_four_l1373_137323


namespace NUMINAMATH_GPT_optimal_cookies_l1373_137331

-- Define the initial state and the game's rules
def initial_blackboard : List Int := List.replicate 2020 1

def erase_two (l : List Int) (x y : Int) : List Int :=
  l.erase x |>.erase y

def write_back (l : List Int) (n : Int) : List Int :=
  n :: l

-- Define termination conditions
def game_ends_condition1 (l : List Int) : Prop :=
  ∃ x ∈ l, x > l.sum - x

def game_ends_condition2 (l : List Int) : Prop :=
  l = List.replicate (l.length) 0

def game_ends (l : List Int) : Prop :=
  game_ends_condition1 l ∨ game_ends_condition2 l

-- Define the number of cookies given to Player A
def cookies (l : List Int) : Int :=
  l.length

-- Prove that if both players play optimally, Player A receives 7 cookies
theorem optimal_cookies : cookies (initial_blackboard) = 7 :=
  sorry

end NUMINAMATH_GPT_optimal_cookies_l1373_137331


namespace NUMINAMATH_GPT_root_implies_quadratic_eq_l1373_137343

theorem root_implies_quadratic_eq (m : ℝ) (h : (m + 2) - 2 + m^2 - 2 * m - 6 = 0) : 
  2 * m^2 - m - 6 = 0 :=
sorry

end NUMINAMATH_GPT_root_implies_quadratic_eq_l1373_137343


namespace NUMINAMATH_GPT_ellipse_eccentricity_l1373_137377

theorem ellipse_eccentricity
  {a b n : ℝ}
  (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ (P : ℝ × ℝ), P.1 = n ∧ P.2 = 4 ∧ (n^2 / a^2 + 16 / b^2 = 1))
  (F1 F2 : ℝ × ℝ)
  (h4 : F1 = (c, 0))        -- Placeholders for focus coordinates of the ellipse
  (h5 : F2 = (-c, 0))
  (h6 : ∃ c, 4*c = (3 / 2) * (a + c))
  : 3 * c = 5 * a → c / a = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l1373_137377


namespace NUMINAMATH_GPT_cost_of_ground_school_l1373_137351

theorem cost_of_ground_school (G : ℝ) (F : ℝ) (h1 : F = G + 625) (h2 : F = 950) :
  G = 325 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_ground_school_l1373_137351


namespace NUMINAMATH_GPT_prime_square_minus_one_divisible_by_24_l1373_137382

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : Prime p) (hp_ge_5 : 5 ≤ p) : 24 ∣ (p^2 - 1) := 
by 
sorry

end NUMINAMATH_GPT_prime_square_minus_one_divisible_by_24_l1373_137382


namespace NUMINAMATH_GPT_subset_A_inter_B_eq_A_l1373_137332

variable {x : ℝ}
def A (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2 * k}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem subset_A_inter_B_eq_A (k : ℝ) : (A k ∩ B = A k) ↔ (k ≤ 3 / 2) := 
sorry

end NUMINAMATH_GPT_subset_A_inter_B_eq_A_l1373_137332


namespace NUMINAMATH_GPT_distance_equal_x_value_l1373_137300

theorem distance_equal_x_value :
  (∀ P Q R : ℝ × ℝ × ℝ, P = (x, 2, 1) ∧ Q = (1, 1, 2) ∧ R = (2, 1, 1) →
  dist P Q = dist P R →
  x = 1) :=
by
  -- Define the points P, Q, R
  let P := (x, 2, 1)
  let Q := (1, 1, 2)
  let R := (2, 1, 1)

  -- Given the condition
  intro h
  sorry

end NUMINAMATH_GPT_distance_equal_x_value_l1373_137300


namespace NUMINAMATH_GPT_final_balance_is_103_5_percent_of_initial_l1373_137349

/-- Define Megan's initial balance. -/
def initial_balance : ℝ := 125

/-- Define the balance after 25% increase from babysitting. -/
def after_babysitting (balance : ℝ) : ℝ :=
  balance + (balance * 0.25)

/-- Define the balance after 20% decrease from buying shoes. -/
def after_shoes (balance : ℝ) : ℝ :=
  balance - (balance * 0.20)

/-- Define the balance after 15% increase by investing in stocks. -/
def after_stocks (balance : ℝ) : ℝ :=
  balance + (balance * 0.15)

/-- Define the balance after 10% decrease due to medical expenses. -/
def after_medical_expense (balance : ℝ) : ℝ :=
  balance - (balance * 0.10)

/-- Define the final balance. -/
def final_balance : ℝ :=
  let b1 := after_babysitting initial_balance
  let b2 := after_shoes b1
  let b3 := after_stocks b2
  after_medical_expense b3

/-- Prove that the final balance is 103.5% of the initial balance. -/
theorem final_balance_is_103_5_percent_of_initial :
  final_balance / initial_balance = 1.035 :=
by
  unfold final_balance
  unfold initial_balance
  unfold after_babysitting
  unfold after_shoes
  unfold after_stocks
  unfold after_medical_expense
  sorry

end NUMINAMATH_GPT_final_balance_is_103_5_percent_of_initial_l1373_137349


namespace NUMINAMATH_GPT_train_speed_is_correct_l1373_137367

-- Define the conditions
def length_of_train : ℕ := 140 -- length in meters
def time_to_cross_pole : ℕ := 7 -- time in seconds

-- Define the expected speed in km/h
def expected_speed_in_kmh : ℕ := 72 -- speed in km/h

-- Prove that the speed of the train in km/h is 72
theorem train_speed_is_correct :
  (length_of_train / time_to_cross_pole) * 36 / 10 = expected_speed_in_kmh :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_correct_l1373_137367


namespace NUMINAMATH_GPT_sequence_general_formula_l1373_137347

def sequence_term (n : ℕ) : ℕ :=
  if n = 0 then 3 else 3 + n * 5 

theorem sequence_general_formula (n : ℕ) : n > 0 → sequence_term n = 5 * n - 2 :=
by 
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l1373_137347


namespace NUMINAMATH_GPT_jerseys_sold_l1373_137366

theorem jerseys_sold (unit_price_jersey : ℕ) (total_revenue_jersey : ℕ) (n : ℕ) 
  (h_unit_price : unit_price_jersey = 165) 
  (h_total_revenue : total_revenue_jersey = 25740) 
  (h_eq : n * unit_price_jersey = total_revenue_jersey) : 
  n = 156 :=
by
  rw [h_unit_price, h_total_revenue] at h_eq
  sorry

end NUMINAMATH_GPT_jerseys_sold_l1373_137366


namespace NUMINAMATH_GPT_row_col_value_2002_2003_l1373_137302

theorem row_col_value_2002_2003 :
  let base_num := (2003 - 1)^2 + 1 
  let result := base_num + 2001 
  result = 2002 * 2003 :=
by
  sorry

end NUMINAMATH_GPT_row_col_value_2002_2003_l1373_137302


namespace NUMINAMATH_GPT_Carson_returned_l1373_137309

theorem Carson_returned :
  ∀ (initial_oranges ate_oranges stolen_oranges final_oranges : ℕ), 
  initial_oranges = 60 →
  ate_oranges = 10 →
  stolen_oranges = (initial_oranges - ate_oranges) / 2 →
  final_oranges = 30 →
  final_oranges = (initial_oranges - ate_oranges - stolen_oranges) + 5 :=
by 
  sorry

end NUMINAMATH_GPT_Carson_returned_l1373_137309


namespace NUMINAMATH_GPT_min_c_for_expression_not_min_abs_c_for_expression_l1373_137326

theorem min_c_for_expression :
  ∀ c : ℝ,
  (c - 3)^2 + (c - 4)^2 + (c - 8)^2 ≥ (5 - 3)^2 + (5 - 4)^2 + (5 - 8)^2 := 
by sorry

theorem not_min_abs_c_for_expression :
  ∃ c : ℝ, |c - 3| + |c - 4| + |c - 8| < |5 - 3| + |5 - 4| + |5 - 8| := 
by sorry

end NUMINAMATH_GPT_min_c_for_expression_not_min_abs_c_for_expression_l1373_137326


namespace NUMINAMATH_GPT_topsoil_cost_l1373_137319

theorem topsoil_cost :
  let cubic_yard_to_cubic_foot := 27
  let cubic_feet_in_5_cubic_yards := 5 * cubic_yard_to_cubic_foot
  let cost_per_cubic_foot := 6
  let total_cost := cubic_feet_in_5_cubic_yards * cost_per_cubic_foot
  total_cost = 810 :=
by
  sorry

end NUMINAMATH_GPT_topsoil_cost_l1373_137319


namespace NUMINAMATH_GPT_closest_ratio_of_adults_to_children_l1373_137330

def total_fees (a c : ℕ) : ℕ := 20 * a + 10 * c
def adults_children_equation (a c : ℕ) : Prop := 2 * a + c = 160

theorem closest_ratio_of_adults_to_children :
  ∃ a c : ℕ, 
    total_fees a c = 1600 ∧
    a ≥ 1 ∧ c ≥ 1 ∧
    adults_children_equation a c ∧
    (∀ a' c' : ℕ, total_fees a' c' = 1600 ∧ 
        a' ≥ 1 ∧ c' ≥ 1 ∧ 
        adults_children_equation a' c' → 
        abs ((a : ℝ) / c - 1) ≤ abs ((a' : ℝ) / c' - 1)) :=
  sorry

end NUMINAMATH_GPT_closest_ratio_of_adults_to_children_l1373_137330


namespace NUMINAMATH_GPT_prime_factor_of_difference_l1373_137305

theorem prime_factor_of_difference {A B : ℕ} (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (h_neq : A ≠ B) :
  Nat.Prime 2 ∧ (∃ B : ℕ, 20 * B = 20 * B) :=
by
  sorry

end NUMINAMATH_GPT_prime_factor_of_difference_l1373_137305


namespace NUMINAMATH_GPT_DennisHas70Marbles_l1373_137356

-- Definitions according to the conditions
def LaurieMarbles : Nat := 37
def KurtMarbles : Nat := LaurieMarbles - 12
def DennisMarbles : Nat := KurtMarbles + 45

-- The proof problem statement
theorem DennisHas70Marbles : DennisMarbles = 70 :=
by
  sorry

end NUMINAMATH_GPT_DennisHas70Marbles_l1373_137356


namespace NUMINAMATH_GPT_fraction_simplifies_correctly_l1373_137353

variable (a b : ℕ)

theorem fraction_simplifies_correctly (h : a ≠ b) : (1/2 * a) / (1/2 * b) = a / b := 
by sorry

end NUMINAMATH_GPT_fraction_simplifies_correctly_l1373_137353


namespace NUMINAMATH_GPT_num_girls_l1373_137324

-- Define conditions as constants
def ratio (B G : ℕ) : Prop := B = (5 * G) / 8
def total (B G : ℕ) : Prop := B + G = 260

-- State the proof problem
theorem num_girls (B G : ℕ) (h1 : ratio B G) (h2 : total B G) : G = 160 :=
by {
  -- actual proof omitted
  sorry
}

end NUMINAMATH_GPT_num_girls_l1373_137324


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1373_137369

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9
noncomputable def repeating_decimal_7 : ℚ := 7 / 9

theorem repeating_decimal_sum : 
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 - repeating_decimal_7 = -1 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_repeating_decimal_sum_l1373_137369


namespace NUMINAMATH_GPT_interval_contains_root_l1373_137322

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem interval_contains_root :
  f (-1) < 0 → 
  f 0 < 0 → 
  f 1 < 0 → 
  f 2 > 0 → 
  ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_interval_contains_root_l1373_137322


namespace NUMINAMATH_GPT_catherine_bottle_caps_l1373_137316

-- Definitions from conditions
def friends : ℕ := 6
def caps_per_friend : ℕ := 3

-- Theorem statement from question and correct answer
theorem catherine_bottle_caps : friends * caps_per_friend = 18 :=
by sorry

end NUMINAMATH_GPT_catherine_bottle_caps_l1373_137316


namespace NUMINAMATH_GPT_sum_of_powers_of_four_to_50_l1373_137378

theorem sum_of_powers_of_four_to_50 :
  2 * (Finset.sum (Finset.range 51) (λ x => x^4)) = 1301700 := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_four_to_50_l1373_137378


namespace NUMINAMATH_GPT_students_in_both_clubs_l1373_137386

theorem students_in_both_clubs
  (T R B total_club_students : ℕ)
  (hT : T = 85) (hR : R = 120)
  (hTotal : T + R - B = total_club_students)
  (hTotalVal : total_club_students = 180) :
  B = 25 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_students_in_both_clubs_l1373_137386


namespace NUMINAMATH_GPT_trigonometric_identity_l1373_137335

theorem trigonometric_identity (α : ℝ) (h : Real.tan (π + α) = 2) :
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π + α) - Real.cos (π + α)) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1373_137335


namespace NUMINAMATH_GPT_systematic_sampling_id_fourth_student_l1373_137387

theorem systematic_sampling_id_fourth_student (n : ℕ) (a b c d : ℕ) (h1 : n = 54) 
(h2 : a = 3) (h3 : b = 29) (h4 : c = 42) (h5 : d = a + 13) : d = 16 :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_id_fourth_student_l1373_137387


namespace NUMINAMATH_GPT_roots_of_equation_in_interval_l1373_137342

theorem roots_of_equation_in_interval (f : ℝ → ℝ) (interval : Set ℝ) (n_roots : ℕ) :
  (∀ x ∈ interval, f x = 8 * x * (1 - 2 * x^2) * (8 * x^4 - 8 * x^2 + 1) - 1) →
  (interval = Set.Icc 0 1) →
  (n_roots = 4) :=
by
  intros f_eq interval_eq
  sorry

end NUMINAMATH_GPT_roots_of_equation_in_interval_l1373_137342


namespace NUMINAMATH_GPT_coterminal_angle_neg_60_eq_300_l1373_137383

theorem coterminal_angle_neg_60_eq_300 :
  ∃ k : ℤ, 0 ≤ k * 360 - 60 ∧ k * 360 - 60 < 360 ∧ (k * 360 - 60 = 300) := by
  sorry

end NUMINAMATH_GPT_coterminal_angle_neg_60_eq_300_l1373_137383


namespace NUMINAMATH_GPT_percentage_difference_l1373_137317

open scoped Classical

theorem percentage_difference (original_number new_number : ℕ) (h₀ : original_number = 60) (h₁ : new_number = 30) :
  (original_number - new_number) / original_number * 100 = 50 :=
by
      sorry

end NUMINAMATH_GPT_percentage_difference_l1373_137317


namespace NUMINAMATH_GPT_no_values_of_expression_l1373_137340

theorem no_values_of_expression (x : ℝ) (h : x^2 - 4 * x + 4 < 0) :
  ¬ ∃ y, y = x^2 + 4 * x + 5 :=
by
  sorry

end NUMINAMATH_GPT_no_values_of_expression_l1373_137340


namespace NUMINAMATH_GPT_parallelogram_area_l1373_137320

-- Defining the vectors u and z
def u : ℝ × ℝ := (4, -1)
def z : ℝ × ℝ := (9, -3)

-- Computing the area of parallelogram formed by vectors u and z
def area_parallelogram (u z : ℝ × ℝ) : ℝ :=
  abs (u.1 * (z.2 + u.2) - u.2 * (z.1 + u.1))

-- Lean statement asserting that the area of the parallelogram is 3
theorem parallelogram_area : area_parallelogram u z = 3 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1373_137320


namespace NUMINAMATH_GPT_point_on_xaxis_equidistant_l1373_137345

theorem point_on_xaxis_equidistant :
  ∃ (A : ℝ × ℝ), A.2 = 0 ∧ 
                  dist A (-3, 2) = dist A (4, -5) ∧ 
                  A = (2, 0) :=
by
  sorry

end NUMINAMATH_GPT_point_on_xaxis_equidistant_l1373_137345


namespace NUMINAMATH_GPT_solve_eq_n_fact_plus_n_eq_n_pow_k_l1373_137321

theorem solve_eq_n_fact_plus_n_eq_n_pow_k :
  ∀ (n k : ℕ), 0 < n → 0 < k → (n! + n = n^k ↔ (n, k) = (2, 2) ∨ (n, k) = (3, 2) ∨ (n, k) = (5, 3)) :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_n_fact_plus_n_eq_n_pow_k_l1373_137321


namespace NUMINAMATH_GPT_casey_saving_l1373_137307

-- Define the conditions
def cost_per_hour_first_employee : ℝ := 20
def cost_per_hour_second_employee : ℝ := 22
def subsidy_per_hour : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost calculations
def weekly_cost_first_employee := cost_per_hour_first_employee * hours_per_week
def effective_cost_per_hour_second_employee := cost_per_hour_second_employee - subsidy_per_hour
def weekly_cost_second_employee := effective_cost_per_hour_second_employee * hours_per_week

-- State the theorem
theorem casey_saving :
    weekly_cost_first_employee - weekly_cost_second_employee = 160 := 
by
  sorry

end NUMINAMATH_GPT_casey_saving_l1373_137307


namespace NUMINAMATH_GPT_min_value_alpha_beta_gamma_l1373_137338

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

def A (α β γ : ℕ) : ℕ := 2 ^ α * 3 ^ β * 5 ^ γ

def condition_1 (α β γ : ℕ) : Prop :=
  is_square (A α β γ / 2)

def condition_2 (α β γ : ℕ) : Prop :=
  is_cube (A α β γ / 3)

def condition_3 (α β γ : ℕ) : Prop :=
  is_fifth_power (A α β γ / 5)

theorem min_value_alpha_beta_gamma (α β γ : ℕ) :
  condition_1 α β γ → condition_2 α β γ → condition_3 α β γ →
  α + β + γ = 31 :=
sorry

end NUMINAMATH_GPT_min_value_alpha_beta_gamma_l1373_137338


namespace NUMINAMATH_GPT_base8_satisfies_l1373_137357

noncomputable def check_base (c : ℕ) : Prop := 
  ((2 * c ^ 2 + 4 * c + 3) + (1 * c ^ 2 + 5 * c + 6)) = (4 * c ^ 2 + 2 * c + 1)

theorem base8_satisfies : check_base 8 := 
by
  -- conditions: (243_c, 156_c, 421_c) translated as provided
  -- proof is skipped here as specified
  sorry

end NUMINAMATH_GPT_base8_satisfies_l1373_137357


namespace NUMINAMATH_GPT_subset_sum_bounds_l1373_137372

theorem subset_sum_bounds (M m n : ℕ) (A : Finset ℕ)
  (h1 : 1 ≤ m) (h2 : m ≤ n) (h3 : 1 ≤ M) (h4 : M ≤ (m * (m + 1)) / 2) (hA : A.card = m) (hA_subset : ∀ x ∈ A, x ∈ Finset.range (n + 1)) :
  ∃ B ⊆ A, 0 ≤ (B.sum id) - M ∧ (B.sum id) - M ≤ n - m :=
by
  sorry

end NUMINAMATH_GPT_subset_sum_bounds_l1373_137372


namespace NUMINAMATH_GPT_equal_sum_sequence_even_odd_l1373_137304

-- Define the sequence a_n
variable {a : ℕ → ℤ}

-- Define the condition of the equal-sum sequence
def equal_sum_sequence (a : ℕ → ℤ) : Prop := ∀ n, a n + a (n + 1) = a (n + 1) + a (n + 2)

-- Statement to prove the odd terms are equal and the even terms are equal
theorem equal_sum_sequence_even_odd (a : ℕ → ℤ) (h : equal_sum_sequence a) : (∀ n, a (2 * n) = a 0) ∧ (∀ n, a (2 * n + 1) = a 1) :=
by
  sorry

end NUMINAMATH_GPT_equal_sum_sequence_even_odd_l1373_137304


namespace NUMINAMATH_GPT_area_of_polygon_l1373_137398

theorem area_of_polygon (side_length n : ℕ) (h1 : n = 36) (h2 : 36 * side_length = 72) (h3 : ∀ i, i < n → (∃ a, ∃ b, (a + b = 4) ∧ (i = 4 * a + b))) :
  (n / 4) * side_length ^ 2 = 144 :=
by
  sorry

end NUMINAMATH_GPT_area_of_polygon_l1373_137398


namespace NUMINAMATH_GPT_expected_value_equals_1_5_l1373_137350

noncomputable def expected_value_win (roll : ℕ) : ℚ :=
  if roll = 1 then -1
  else if roll = 4 then -4
  else if roll = 2 ∨ roll = 3 ∨ roll = 5 ∨ roll = 7 then roll
  else 0

noncomputable def expected_value_total : ℚ :=
  (1/8 : ℚ) * ((expected_value_win 1) + (expected_value_win 2) + (expected_value_win 3) +
               (expected_value_win 4) + (expected_value_win 5) + (expected_value_win 6) +
               (expected_value_win 7) + (expected_value_win 8))

theorem expected_value_equals_1_5 : expected_value_total = 1.5 := by
  sorry

end NUMINAMATH_GPT_expected_value_equals_1_5_l1373_137350


namespace NUMINAMATH_GPT_sticker_probability_l1373_137389

theorem sticker_probability 
  (n : ℕ) (k : ℕ) (uncollected : ℕ) (collected : ℕ) (C : ℕ → ℕ → ℕ) :
  n = 18 → k = 10 → uncollected = 6 → collected = 12 → 
  (C uncollected uncollected) * (C collected (k - uncollected)) = 495 → 
  C n k = 43758 → 
  (495 : ℚ) / 43758 = 5 / 442 := 
by
  intros h_n h_k h_uncollected h_collected h_C1 h_C2
  sorry

end NUMINAMATH_GPT_sticker_probability_l1373_137389


namespace NUMINAMATH_GPT_kids_still_awake_l1373_137334

theorem kids_still_awake (initial_count remaining_after_first remaining_after_second : ℕ) 
  (h_initial : initial_count = 20)
  (h_first_round : remaining_after_first = initial_count / 2)
  (h_second_round : remaining_after_second = remaining_after_first / 2) : 
  remaining_after_second = 5 := 
by
  sorry

end NUMINAMATH_GPT_kids_still_awake_l1373_137334


namespace NUMINAMATH_GPT_marble_count_l1373_137361

variable (initial_mar: Int) (lost_mar: Int)

def final_mar (initial_mar: Int) (lost_mar: Int) : Int :=
  initial_mar - lost_mar

theorem marble_count : final_mar 16 7 = 9 := by
  trivial

end NUMINAMATH_GPT_marble_count_l1373_137361


namespace NUMINAMATH_GPT_no_faces_painted_two_or_three_faces_painted_l1373_137397

-- Define the dimensions of the cuboid
def cuboid_length : ℕ := 3
def cuboid_width : ℕ := 4
def cuboid_height : ℕ := 5

-- Define the number of small cubes
def small_cubes_total : ℕ := 60

-- Define the number of small cubes with no faces painted
def small_cubes_no_faces_painted : ℕ := (cuboid_length - 2) * (cuboid_width - 2) * (cuboid_height - 2)

-- Define the number of small cubes with 2 faces painted
def small_cubes_two_faces_painted : ℕ := (cuboid_length - 2) * cuboid_width +
                                          (cuboid_width - 2) * cuboid_length +
                                          (cuboid_height - 2) * cuboid_width

-- Define the number of small cubes with 3 faces painted
def small_cubes_three_faces_painted : ℕ := 8

-- Define the probabilities
def probability_no_faces_painted : ℚ := small_cubes_no_faces_painted / small_cubes_total
def probability_two_or_three_faces_painted : ℚ := (small_cubes_two_faces_painted + small_cubes_three_faces_painted) / small_cubes_total

-- Theorems to prove
theorem no_faces_painted (h : cuboid_length = 3 ∧ cuboid_width = 4 ∧ cuboid_height = 5 ∧ 
                           small_cubes_total = 60 ∧ small_cubes_no_faces_painted = 6) :
  probability_no_faces_painted = 1 / 10 := by
  sorry

theorem two_or_three_faces_painted (h : cuboid_length = 3 ∧ cuboid_width = 4 ∧ cuboid_height = 5 ∧ 
                                    small_cubes_total = 60 ∧ small_cubes_two_faces_painted = 24 ∧
                                    small_cubes_three_faces_painted = 8) :
  probability_two_or_three_faces_painted = 8 / 15 := by
  sorry

end NUMINAMATH_GPT_no_faces_painted_two_or_three_faces_painted_l1373_137397


namespace NUMINAMATH_GPT_pow_congr_mod_eight_l1373_137328

theorem pow_congr_mod_eight (n : ℕ) : (5^n + 2 * 3^(n-1) + 1) % 8 = 0 := sorry

end NUMINAMATH_GPT_pow_congr_mod_eight_l1373_137328


namespace NUMINAMATH_GPT_solution_l1373_137354

noncomputable def problem_statement (a b : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)

theorem solution (a b : ℝ) :
  problem_statement a b ↔ (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ a' b' : ℤ, (a : ℝ) = a' ∧ (b : ℝ) = b')) :=
by
  sorry

end NUMINAMATH_GPT_solution_l1373_137354


namespace NUMINAMATH_GPT_mark_more_hours_l1373_137381

-- Definitions based on the conditions
variables (Pat Kate Mark Alex : ℝ)
variables (total_hours : ℝ)
variables (h1 : Pat + Kate + Mark + Alex = 350)
variables (h2 : Pat = 2 * Kate)
variables (h3 : Pat = (1 / 3) * Mark)
variables (h4 : Alex = 1.5 * Kate)

-- Theorem statement with the desired proof target
theorem mark_more_hours (Pat Kate Mark Alex : ℝ) (h1 : Pat + Kate + Mark + Alex = 350) 
(h2 : Pat = 2 * Kate) (h3 : Pat = (1 / 3) * Mark) (h4 : Alex = 1.5 * Kate) : 
Mark - (Kate + Alex) = 116.66666666666667 := sorry

end NUMINAMATH_GPT_mark_more_hours_l1373_137381


namespace NUMINAMATH_GPT_correct_operator_is_subtraction_l1373_137312

theorem correct_operator_is_subtraction :
  (8 - 2) + 5 * (3 - 2) = 11 :=
by
  sorry

end NUMINAMATH_GPT_correct_operator_is_subtraction_l1373_137312


namespace NUMINAMATH_GPT_geometric_sequence_decreasing_iff_l1373_137360

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def is_decreasing_sequence (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a n > a (n + 1)

theorem geometric_sequence_decreasing_iff (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (a 0 > a 1 ∧ a 1 > a 2) ↔ is_decreasing_sequence a :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_decreasing_iff_l1373_137360


namespace NUMINAMATH_GPT_volume_frustum_fraction_l1373_137362

-- Define the base edge and initial altitude of the pyramid.
def base_edge := 32 -- in inches
def altitude_original := 1 -- in feet

-- Define the fractional part representing the altitude of the smaller pyramid.
def altitude_fraction := 1/4

-- Define the volume of the original pyramid being V.
noncomputable def volume_original : ℝ := (1/3) * (base_edge ^ 2) * altitude_original

-- Define the volume of the smaller pyramid being removed.
noncomputable def volume_smaller : ℝ := (1/3) * ((altitude_fraction * base_edge) ^ 2) * (altitude_fraction * altitude_original)

-- We now state the proof
theorem volume_frustum_fraction : 
  (volume_original - volume_smaller) / volume_original = 63/64 :=
by
  sorry

end NUMINAMATH_GPT_volume_frustum_fraction_l1373_137362


namespace NUMINAMATH_GPT_ball_is_green_probability_l1373_137313

noncomputable def probability_green_ball : ℚ :=
  let containerI_red := 8
  let containerI_green := 4
  let containerII_red := 3
  let containerII_green := 5
  let containerIII_red := 4
  let containerIII_green := 6
  let probability_container := (1 : ℚ) / 3
  let probability_green_I := (containerI_green : ℚ) / (containerI_red + containerI_green)
  let probability_green_II := (containerII_green : ℚ) / (containerII_red + containerII_green)
  let probability_green_III := (containerIII_green : ℚ) / (containerIII_red + containerIII_green)
  probability_container * probability_green_I +
  probability_container * probability_green_II +
  probability_container * probability_green_III

theorem ball_is_green_probability :
  probability_green_ball = 187 / 360 :=
by
  -- The detailed proof is omitted and left as an exercise
  sorry

end NUMINAMATH_GPT_ball_is_green_probability_l1373_137313


namespace NUMINAMATH_GPT_double_decker_bus_total_capacity_l1373_137306

-- Define conditions for the lower floor seating
def lower_floor_left_seats : Nat := 15
def lower_floor_right_seats : Nat := 12
def lower_floor_priority_seats : Nat := 4

-- Each seat on the left and right side of the lower floor holds 2 people
def lower_floor_left_capacity : Nat := lower_floor_left_seats * 2
def lower_floor_right_capacity : Nat := lower_floor_right_seats * 2
def lower_floor_priority_capacity : Nat := lower_floor_priority_seats * 1

-- Define conditions for the upper floor seating
def upper_floor_left_seats : Nat := 20
def upper_floor_right_seats : Nat := 20
def upper_floor_back_capacity : Nat := 15

-- Each seat on the left and right side of the upper floor holds 3 people
def upper_floor_left_capacity : Nat := upper_floor_left_seats * 3
def upper_floor_right_capacity : Nat := upper_floor_right_seats * 3

-- Total capacity of lower and upper floors
def lower_floor_total_capacity : Nat := lower_floor_left_capacity + lower_floor_right_capacity + lower_floor_priority_capacity
def upper_floor_total_capacity : Nat := upper_floor_left_capacity + upper_floor_right_capacity + upper_floor_back_capacity

-- Assert the total capacity
def bus_total_capacity : Nat := lower_floor_total_capacity + upper_floor_total_capacity

-- Prove that the total bus capacity is 193 people
theorem double_decker_bus_total_capacity : bus_total_capacity = 193 := by
  sorry

end NUMINAMATH_GPT_double_decker_bus_total_capacity_l1373_137306


namespace NUMINAMATH_GPT_find_eccentricity_of_ellipse_l1373_137352

theorem find_eccentricity_of_ellipse
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (hx : ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x, y) ∈ { p | (p.1^2 / a^2 + p.2^2 / b^2 = 1) })
  (hk : ∀ k x1 y1 x2 y2 : ℝ, y1 = k * x1 ∧ y2 = k * x2 → x1 ≠ x2 → (y1 = x1 * k ∧ y2 = x2 * k))  -- intersection points condition
  (hAB_AC : ∀ m n : ℝ, m ≠ 0 → (n - b) / m * (-n - b) / (-m) = -3/4 )
  : ∃ e : ℝ, e = 1/2 :=
sorry

end NUMINAMATH_GPT_find_eccentricity_of_ellipse_l1373_137352


namespace NUMINAMATH_GPT_find_number_l1373_137339

theorem find_number (x : ℝ) (h : x * 9999 = 824777405) : x = 82482.5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1373_137339


namespace NUMINAMATH_GPT_smallest_S_value_l1373_137374

def num_list := {x : ℕ // 1 ≤ x ∧ x ≤ 9}

def S (a b c : num_list) (d e f : num_list) (g h i : num_list) : ℕ :=
  a.val * b.val * c.val + d.val * e.val * f.val + g.val * h.val * i.val

theorem smallest_S_value :
  ∃ a b c d e f g h i : num_list,
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  S a b c d e f g h i = 214 :=
sorry

end NUMINAMATH_GPT_smallest_S_value_l1373_137374


namespace NUMINAMATH_GPT_percent_difference_l1373_137359

theorem percent_difference : 0.12 * 24.2 - 0.10 * 14.2 = 1.484 := by
  sorry

end NUMINAMATH_GPT_percent_difference_l1373_137359


namespace NUMINAMATH_GPT_clark_discount_l1373_137363

theorem clark_discount (price_per_part : ℕ) (number_of_parts : ℕ) (amount_paid : ℕ)
  (h1 : price_per_part = 80)
  (h2 : number_of_parts = 7)
  (h3 : amount_paid = 439) : 
  (number_of_parts * price_per_part) - amount_paid = 121 := by
  sorry

end NUMINAMATH_GPT_clark_discount_l1373_137363


namespace NUMINAMATH_GPT_find_multiple_l1373_137327
-- Importing Mathlib to access any necessary math definitions.

-- Define the constants based on the given conditions.
def Darwin_money : ℝ := 45
def Mia_money : ℝ := 110
def additional_amount : ℝ := 20

-- The Lean theorem which encapsulates the proof problem.
theorem find_multiple (x : ℝ) : 
  Mia_money = x * Darwin_money + additional_amount → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l1373_137327


namespace NUMINAMATH_GPT_ball_bouncing_height_l1373_137365

theorem ball_bouncing_height : ∃ (b : ℕ), 400 * (3/4 : ℝ)^b < 50 ∧ ∀ n < b, 400 * (3/4 : ℝ)^n ≥ 50 :=
by
  use 8
  sorry

end NUMINAMATH_GPT_ball_bouncing_height_l1373_137365


namespace NUMINAMATH_GPT_B_greater_than_A_l1373_137348

def A := (54 : ℚ) / (5^7 * 11^4 : ℚ)
def B := (55 : ℚ) / (5^7 * 11^4 : ℚ)

theorem B_greater_than_A : B > A := by
  sorry

end NUMINAMATH_GPT_B_greater_than_A_l1373_137348


namespace NUMINAMATH_GPT_problem_statement_l1373_137355

theorem problem_statement (a b c : ℝ) 
  (h1 : a - 2 * b + c = 0) 
  (h2 : a + 2 * b + c < 0) : b < 0 ∧ b^2 - a * c ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1373_137355


namespace NUMINAMATH_GPT_solve_equation_2021_2020_l1373_137314

theorem solve_equation_2021_2020 (x : ℝ) (hx : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_equation_2021_2020_l1373_137314


namespace NUMINAMATH_GPT_total_spent_on_burgers_l1373_137318

def days_in_june := 30
def burgers_per_day := 4
def cost_per_burger := 13

theorem total_spent_on_burgers (total_spent : Nat) :
  total_spent = days_in_june * burgers_per_day * cost_per_burger :=
sorry

end NUMINAMATH_GPT_total_spent_on_burgers_l1373_137318


namespace NUMINAMATH_GPT_multiply_98_102_l1373_137303

theorem multiply_98_102 : 98 * 102 = 9996 :=
by sorry

end NUMINAMATH_GPT_multiply_98_102_l1373_137303


namespace NUMINAMATH_GPT_peggy_dolls_ratio_l1373_137393

noncomputable def peggy_dolls_original := 6
noncomputable def peggy_dolls_from_grandmother := 30
noncomputable def peggy_dolls_total := 51

theorem peggy_dolls_ratio :
  ∃ x, peggy_dolls_original + peggy_dolls_from_grandmother + x = peggy_dolls_total ∧ x / peggy_dolls_from_grandmother = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_peggy_dolls_ratio_l1373_137393


namespace NUMINAMATH_GPT_train_speed_clicks_l1373_137399

theorem train_speed_clicks (x : ℝ) (rail_length_feet : ℝ := 40) (clicks_per_mile : ℝ := 5280/ 40) :
  15 ≤ (2400/5280) * 60  * clicks_per_mile ∧ (2400/5280) * 60 * clicks_per_mile ≤ 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_train_speed_clicks_l1373_137399


namespace NUMINAMATH_GPT_probability_Hugo_first_roll_is_six_l1373_137385

/-
In a dice game, each of 5 players, including Hugo, rolls a standard 6-sided die. 
The winner is the player who rolls the highest number. 
In the event of a tie for the highest roll, those involved in the tie roll again until a clear winner emerges.
-/
variable (HugoRoll : Nat) (A1 B1 C1 D1 : Nat)
variable (W : Bool)

-- Conditions in the problem
def isWinner (HugoRoll : Nat) (W : Bool) : Prop := (W = true)
def firstRollAtLeastFour (HugoRoll : Nat) : Prop := HugoRoll >= 4
def firstRollIsSix (HugoRoll : Nat) : Prop := HugoRoll = 6

-- Hypotheses: Hugo's event conditions
axiom HugoWonAndRollsAtLeastFour : isWinner HugoRoll W ∧ firstRollAtLeastFour HugoRoll

-- Target probability based on problem statement
noncomputable def probability (p : ℚ) : Prop := p = 625 / 4626

-- Main statement
theorem probability_Hugo_first_roll_is_six (HugoRoll : Nat) (A1 B1 C1 D1 : Nat) (W : Bool) :
  isWinner HugoRoll W ∧ firstRollAtLeastFour HugoRoll → 
  probability (625 / 4626) := by
  sorry


end NUMINAMATH_GPT_probability_Hugo_first_roll_is_six_l1373_137385


namespace NUMINAMATH_GPT_min_squares_to_cover_staircase_l1373_137394

-- Definition of the staircase and the constraints
def is_staircase (n : ℕ) (s : ℕ → ℕ) : Prop :=
  ∀ i, i < n → s i = i + 1

-- The proof problem statement
theorem min_squares_to_cover_staircase : 
  ∀ n : ℕ, n = 15 →
  ∀ s : ℕ → ℕ, is_staircase n s →
  ∃ k : ℕ, k = 15 ∧ (∀ i, i < n → ∃ a b : ℕ, a ≤ i ∧ b ≤ s a ∧ ∃ (l : ℕ), l = 1) :=
by
  sorry

end NUMINAMATH_GPT_min_squares_to_cover_staircase_l1373_137394


namespace NUMINAMATH_GPT_triangle_area_solution_l1373_137392

noncomputable def triangle_area (a b : ℝ) : ℝ := 
  let r := 6 -- radius of each circle
  let d := 2 -- derived distance
  let s := 2 * Real.sqrt 3 * d -- side length of the equilateral triangle
  let area := (Real.sqrt 3 / 4) * s^2 
  area

theorem triangle_area_solution : ∃ a b : ℝ, 
  triangle_area a b = 3 * Real.sqrt 3 ∧ 
  a + b = 27 := 
by 
  exists 27
  exists 3
  sorry

end NUMINAMATH_GPT_triangle_area_solution_l1373_137392


namespace NUMINAMATH_GPT_max_number_of_books_laughlin_can_buy_l1373_137390

-- Definitions of costs and the budget constraint
def individual_book_cost : ℕ := 3
def four_book_bundle_cost : ℕ := 10
def seven_book_bundle_cost : ℕ := 15
def budget : ℕ := 20

-- Condition that Laughlin must buy at least one 4-book bundle
def minimum_required_four_book_bundles : ℕ := 1

-- Define the function to calculate the maximum number of books Laughlin can buy
def max_books (budget : ℕ) (individual_book_cost : ℕ) 
              (four_book_bundle_cost : ℕ) (seven_book_bundle_cost : ℕ) 
              (min_four_book_bundles : ℕ) : ℕ :=
  let remaining_budget_after_four_bundle := budget - (min_four_book_bundles * four_book_bundle_cost)
  if remaining_budget_after_four_bundle >= seven_book_bundle_cost then
    min_four_book_bundles * 4 + 7
  else if remaining_budget_after_four_bundle >= individual_book_cost then
    min_four_book_bundles * 4 + remaining_budget_after_four_bundle / individual_book_cost
  else
    min_four_book_bundles * 4

-- Proof statement: Laughlin can buy a maximum of 7 books
theorem max_number_of_books_laughlin_can_buy : 
  max_books budget individual_book_cost four_book_bundle_cost seven_book_bundle_cost minimum_required_four_book_bundles = 7 :=
by
  sorry

end NUMINAMATH_GPT_max_number_of_books_laughlin_can_buy_l1373_137390


namespace NUMINAMATH_GPT_largest_value_fraction_l1373_137329

theorem largest_value_fraction (x y : ℝ) (hx : 10 ≤ x ∧ x ≤ 20) (hy : 40 ≤ y ∧ y ≤ 60) :
  ∃ z, z = (x^2 / (2 * y)) ∧ z ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_largest_value_fraction_l1373_137329


namespace NUMINAMATH_GPT_gcd_75_100_l1373_137337

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_gcd_75_100_l1373_137337


namespace NUMINAMATH_GPT_inequality_sqrt_sum_l1373_137310

theorem inequality_sqrt_sum (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_sqrt_sum_l1373_137310


namespace NUMINAMATH_GPT_locus_of_vertex_P_l1373_137371

noncomputable def M : ℝ × ℝ := (0, 5)
noncomputable def N : ℝ × ℝ := (0, -5)
noncomputable def perimeter : ℝ := 36

theorem locus_of_vertex_P : ∃ (P : ℝ × ℝ), 
  (∃ (a b : ℝ), a = 13 ∧ b = 12 ∧ P ≠ (0,0) ∧
  (a^2 = b^2 + 5^2) ∧ 
  (perimeter = 2 * a + (5 - (-5))) ∧ 
  ((P.1)^2 / 144 + (P.2)^2 / 169 = 1)) :=
sorry

end NUMINAMATH_GPT_locus_of_vertex_P_l1373_137371


namespace NUMINAMATH_GPT_relations_of_sets_l1373_137346

open Set

theorem relations_of_sets {A B : Set ℝ} (h : ∃ x ∈ A, x ∉ B) : 
  ¬(A ⊆ B) ∧ ((A ∩ B ≠ ∅) ∨ (B ⊆ A) ∨ (A ∩ B = ∅)) := sorry

end NUMINAMATH_GPT_relations_of_sets_l1373_137346


namespace NUMINAMATH_GPT_find_fourth_vertex_l1373_137396

-- Given three vertices of a tetrahedron
def v1 : ℤ × ℤ × ℤ := (1, 1, 2)
def v2 : ℤ × ℤ × ℤ := (4, 2, 1)
def v3 : ℤ × ℤ × ℤ := (3, 1, 5)

-- The side length squared of the tetrahedron (computed from any pair of given points)
def side_length_squared : ℤ := 11

-- The goal is to find the fourth vertex with integer coordinates which maintains the distance
def is_fourth_vertex (x y z : ℤ) : Prop :=
  (x - 1)^2 + (y - 1)^2 + (z - 2)^2 = side_length_squared ∧
  (x - 4)^2 + (y - 2)^2 + (z - 1)^2 = side_length_squared ∧
  (x - 3)^2 + (y - 1)^2 + (z - 5)^2 = side_length_squared

theorem find_fourth_vertex : is_fourth_vertex 4 1 3 :=
  sorry

end NUMINAMATH_GPT_find_fourth_vertex_l1373_137396


namespace NUMINAMATH_GPT_time_with_family_l1373_137301

theorem time_with_family : 
    let hours_in_day := 24
    let sleep_fraction := 1 / 3
    let school_fraction := 1 / 6
    let assignment_fraction := 1 / 12
    let sleep_hours := sleep_fraction * hours_in_day
    let school_hours := school_fraction * hours_in_day
    let assignment_hours := assignment_fraction * hours_in_day
    let total_hours_occupied := sleep_hours + school_hours + assignment_hours
    hours_in_day - total_hours_occupied = 10 :=
by
  sorry

end NUMINAMATH_GPT_time_with_family_l1373_137301


namespace NUMINAMATH_GPT_points_per_touchdown_l1373_137311

theorem points_per_touchdown (total_points touchdowns : ℕ) (h1 : total_points = 21) (h2 : touchdowns = 3) :
  total_points / touchdowns = 7 :=
by
  sorry

end NUMINAMATH_GPT_points_per_touchdown_l1373_137311


namespace NUMINAMATH_GPT_find_phi_l1373_137391

theorem find_phi (ϕ : ℝ) (h1 : |ϕ| < π / 2)
  (h2 : ∃ k : ℤ, 3 * (π / 12) + ϕ = k * π + π / 2) :
  ϕ = π / 4 :=
by sorry

end NUMINAMATH_GPT_find_phi_l1373_137391


namespace NUMINAMATH_GPT_Petya_wins_l1373_137380

theorem Petya_wins (n : ℕ) (h₁ : n = 2016) : (∀ m : ℕ, m < n → ∀ k : ℕ, k ∣ m ∧ k ≠ m → m - k = 1 → false) :=
sorry

end NUMINAMATH_GPT_Petya_wins_l1373_137380


namespace NUMINAMATH_GPT_original_number_l1373_137333

theorem original_number (h : 2.04 / 1.275 = 1.6) : 204 / 12.75 = 16 := 
by
  sorry

end NUMINAMATH_GPT_original_number_l1373_137333


namespace NUMINAMATH_GPT_remainder_when_n_add_3006_divided_by_6_l1373_137370

theorem remainder_when_n_add_3006_divided_by_6 (n : ℤ) (h : n % 6 = 1) : (n + 3006) % 6 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_when_n_add_3006_divided_by_6_l1373_137370
